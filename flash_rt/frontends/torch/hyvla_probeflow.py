"""Native BF16 Hy-Embodied VLA with Euler or host-routed ProbeFlow.

The Thor frontend supplies checkpoint IO, ViT, prefix construction and raw
padded action outputs. This adapter deliberately uses its plain BF16 pipeline
on real SM80/87/89/110/120 CUDA devices, not the specialized quantized Orin or
RTX frontends. These are source execution paths, not runtime certifications.

Only prefix preparation and non-integrating velocity evaluations are captured.
Adaptive routing and action-vector integration always execute outside graphs.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch

from flash_rt.frontends.torch.hyvla_thor import (
    HyVLATorchFrontendThor, _BF16, _camera_tensor,
)
from flash_rt.sampling.probeflow import ProbeFlowConfig
from flash_rt.sampling.torch_probeflow import sample_probeflow_torch_


_HARDWARE_CAPABILITIES = {
    "cuda_sm80": (8, 0), "rtx_sm80": (8, 0),
    "cuda_sm87": (8, 7), "rtx_sm87": (8, 7), "orin": (8, 7),
    "cuda_sm89": (8, 9), "rtx_sm89": (8, 9),
    "cuda_sm110": (11, 0), "thor": (11, 0),
    "cuda_sm120": (12, 0), "rtx_sm120": (12, 0),
}


class HyVLAFlowFrontend(HyVLATorchFrontendThor):
    """BF16 native flow frontend, retaining the existing raw action domain.

    A frontend instance owns mutable graph buffers and is not reentrant.
    ``num_steps`` selects an Euler budget only. For ProbeFlow use
    ``probeflow_config.max_steps`` instead. Quantized/fused flags are accepted
    only as explicit ``False``; unknown options raise instead of being ignored.
    ``numerical_profile`` defaults to the unchanged semantic BF16 path; the
    official-local option matches verified local cast boundaries, not a full
    official model or a claim of passing its recorded action oracle.
    """

    _SUPPORTED_CAPABILITIES = ((8, 0), (8, 7), (8, 9), (11, 0), (12, 0))

    def __init__(self, checkpoint_dir: str, *, hardware: str | None = "auto",
                 sampler: str = "euler",
                 probeflow_config: ProbeFlowConfig | None = None,
                 num_steps: int | None = None, use_cuda_graph: bool = True,
                 numerical_profile: str = "semantic_bf16",
                 use_fp8: bool = False, use_fp8_vit: bool = False,
                 use_fused: bool = False, use_fp4: bool = False,
                 use_fused_quant: bool = False, use_autotune: bool = False,
                 use_ffn_mega: bool = False):
        # Validate all user options before checkpoint IO or device probing.
        if numerical_profile not in ("semantic_bf16", "official_local_fp32"):
            raise ValueError("numerical_profile must be semantic_bf16 or official_local_fp32")
        if sampler not in ("euler", "probeflow"):
            raise ValueError("sampler must be euler or probeflow")
        if probeflow_config is not None and not isinstance(probeflow_config, ProbeFlowConfig):
            raise TypeError("probeflow_config must be a ProbeFlowConfig")
        if num_steps is not None and (
                isinstance(num_steps, bool) or not isinstance(num_steps, Integral)
                or num_steps < 1):
            raise ValueError("num_steps must be a positive integer")
        if sampler == "euler" and probeflow_config is not None:
            raise ValueError('probeflow_config requires sampler="probeflow"')
        if sampler == "probeflow" and num_steps is not None:
            raise ValueError("Use probeflow_config.max_steps, not num_steps, for ProbeFlow")
        if not isinstance(use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be a bool")
        flags = {
            "use_fp8": use_fp8, "use_fp8_vit": use_fp8_vit,
            "use_fused": use_fused, "use_fp4": use_fp4,
            "use_fused_quant": use_fused_quant, "use_autotune": use_autotune,
            "use_ffn_mega": use_ffn_mega,
        }
        for name, value in flags.items():
            if value is not False:
                raise ValueError(f"HyVLAFlowFrontend is BF16-only; {name} must be False")
        if hardware is None:
            hardware = "auto"
        if not isinstance(hardware, str) or (
                hardware != "auto" and hardware not in _HARDWARE_CAPABILITIES):
            raise ValueError(f"Unknown HyVLA BF16 hardware {hardware!r}")
        self.hardware = hardware
        self._numerical_profile = numerical_profile
        self.sampler = sampler
        self.probeflow_config = (
            (probeflow_config or ProbeFlowConfig()) if sampler == "probeflow" else None)
        self._requested_num_steps = num_steps
        self.use_cuda_graph = use_cuda_graph
        self._last_sampler_metadata = None
        # The base constructor binds HyVLAThorBF16Pipeline. All its optional
        # SM110-specific kernels remain disabled by these explicit flags.
        super().__init__(checkpoint_dir, hardware=hardware, **flags)

    @property
    def numerical_profile(self):
        """Fixed for this graph-owning instance; not a runtime precision toggle."""
        return self._numerical_profile

    def _load_weights(self):
        if self.numerical_profile == "official_local_fp32":
            # Verify the only traced rotary initialization before reading any
            # weights. Do not silently emulate other HF rope configurations.
            self._official_rope_parameters()
        super()._load_weights()

    def _official_rope_parameters(self):
        if hasattr(self, "_official_rope_inv_freq"):
            return self._official_rope_inv_freq
        text = self.cfg.get("vlm_config_dict", {}).get("text_config", {})
        scaling = text.get("rope_scaling", {})
        alpha = scaling.get("alpha")
        if (scaling.get("type") != "dynamic" or not isinstance(alpha, (int, float))
                or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0):
            raise NotImplementedError(
                "official_local_fp32 currently requires the traced dynamic RoPE with positive alpha")
        # Official module construction initializes inv_freq on CPU in FP32;
        # policy.to(bfloat16) then rounds that nonpersistent buffer. Widening
        # it for forward must not recreate an unrounded FP32 frequency table.
        base = self.rope_theta * alpha ** (self.head_dim / (self.head_dim - 2))
        inverse = 1.0 / (base ** (
            torch.arange(0, self.head_dim, 2, device="cpu").float() / self.head_dim))
        self._official_rope_inv_freq = inverse.to(_BF16).to(self.device)
        return self._official_rope_inv_freq

    def _rope_cos_sin(self, positions):
        if self.numerical_profile == "semantic_bf16":
            return super()._rope_cos_sin(positions)
        inverse = self._official_rope_parameters()
        with torch.autocast(device_type=torch.device(self.device).type, enabled=False):
            expanded = inverse[None, :, None].float().expand(positions.shape[0], -1, 1)
            frequencies = (expanded @ positions[:, None, :].float()).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            # dual_tower passes _dtype_ref.float() to rotary_emb, so the
            # official output tables are FP32, not the weight/storage dtype.
            return embedding.cos()[:, None], embedding.sin()[:, None]

    def _require_arch(self):
        """Probe actual CUDA capability; never honor FORCE_ARCH or fake flags."""
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            raise RuntimeError("HyVLA BF16 frontend requires CUDA; CUDA is not available")
        capability = tuple(torch.cuda.get_device_capability())
        if capability not in self._SUPPORTED_CAPABILITIES:
            raise RuntimeError(
                f"HyVLA BF16 source path requires capability in "
                f"{self._SUPPORTED_CAPABILITIES}, found capability {capability}")
        expected = _HARDWARE_CAPABILITIES.get(self.hardware)
        if expected is not None and expected != capability:
            raise RuntimeError(
                f"Requested hardware={self.hardware!r} requires capability "
                f"{expected}, found actual capability {capability}")
        self.capability = capability

    def _precompute_time_embs(self):
        """Keep exact callback float keys and never treat a union as a budget."""
        if self.sampler == "probeflow":
            self._model_times = self.probeflow_config.timesteps()
        else:
            if self._requested_num_steps is not None:
                self.num_steps = int(self._requested_num_steps)
            if self.num_steps < 1:
                raise ValueError("Checkpoint num_steps must be positive")
            # Preserve the legacy Euler expression, including float order.
            self._model_times = tuple(
                1.0 + s * (-1.0 / self.num_steps) for s in range(self.num_steps))
        self._time_index = {time: index for index, time in enumerate(self._model_times)}
        self._embedding_times = self._model_times
        if self.numerical_profile == "official_local_fp32":
            if self.sampler == "euler":
                # Official sample_actions updates a FP32 scalar time in place.
                # Preserve the exact callback keys independently of those
                # recurrent, slightly different effective embedding values.
                time = torch.tensor(1.0, dtype=torch.float32, device="cpu")
                dt = torch.tensor(-1.0 / self.num_steps, dtype=torch.float32, device="cpu")
                effective = []
                for _ in range(self.num_steps):
                    effective.append(float(time))
                    time += dt
                self._embedding_times = tuple(effective)
                self._euler_step_size = (-dt).to(self.device)
            else:
                self._embedding_times = tuple(
                    float(torch.tensor(time, dtype=torch.float32, device="cpu")) for time in self._model_times)
        self._time_embs = torch.stack([
            self._sinusoidal_time(time) for time in self._embedding_times
        ]).to(self.device, _BF16)

    def _sinusoidal_time(self, t, min_period=4e-3, max_period=4.0):
        if self.numerical_profile == "semantic_bf16":
            return super()._sinusoidal_time(t, min_period, max_period)
        # Official create_sinusoidal_pos_embedding computes in FP64 on the
        # target device from a FP32 model-time tensor, then embed_suffix casts
        # to BF16. Large projection GEMMs are unaffected.
        fraction = torch.linspace(0.0, 1.0, self.proj_width // 2,
                                  dtype=torch.float64, device=self.device)
        period = min_period * (max_period / min_period) ** fraction
        scaling = 1.0 / period * 2 * math.pi
        time = torch.tensor([t], dtype=torch.float32, device=self.device)
        argument = scaling[None, :] * time[:, None]
        return torch.cat([torch.sin(argument), torch.cos(argument)], dim=1)

    def _allocate_flow_buffers(self, S_p):
        """Allocate long-lived prefix/KV/state/action and velocity addresses."""
        S_s, hd = 1 + self.chunk, self.head_dim
        local_dtype = torch.float32 if self.numerical_profile == "official_local_fp32" else _BF16

        def z(*shape, dtype=_BF16):
            return torch.zeros(*shape, dtype=dtype, device=self.device)

        return {
            "pe": z(1, S_p, self.d_vlm),
            "pmask": z(1, 1, S_p, S_p, dtype=torch.bool),
            "pcos": z(1, 1, S_p, hd, dtype=local_dtype),
            "psin": z(1, 1, S_p, hd, dtype=local_dtype),
            "smask": z(1, 1, S_s, S_p + S_s, dtype=torch.bool),
            "scos": z(1, 1, S_s, hd, dtype=local_dtype),
            "ssin": z(1, 1, S_s, hd, dtype=local_dtype),
            "state": z(1, self.max_state_dim),
            "state_emb": z(1, 1, self.proj_width),
            "x": z(1, self.chunk, self.max_action_dim, dtype=torch.float32),
            "velocity": z(1, self.chunk, self.max_action_dim, dtype=torch.float32),
            # Like the native base path, cache KV pre-expanded to query heads.
            "kbuf": z(32, 1, self.n_heads, S_p + S_s, hd, dtype=local_dtype),
            "vbuf": z(32, 1, self.n_heads, S_p + S_s, hd),
        }

    def _prefix_body(self, b, n_vis):
        self.pipe.prefill(b["pe"], n_vis, b["pmask"], b["pcos"], b["psin"],
                          b["kbuf"], b["vbuf"])
        b["state_emb"].copy_(self.pipe.embed_state(b["state"]))

    def _velocity_body(self, b, time_index, S_p):
        raw = self.pipe.evaluate_velocity(
            b["x"], self._time_embs[time_index], b["state_emb"],
            b["smask"], b["scos"], b["ssin"], b["kbuf"], b["vbuf"], S_p)
        # Native action latent is FP32, native head BF16. Convert first, then
        # negate exactly once for the shared progress-velocity contract u=-v.
        b["velocity"].copy_(raw)
        b["velocity"].neg_()

    @staticmethod
    def _capture_flow_body(body):
        """Warm up a side stream; retain each graph's own private scratch pool."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return graph

    def _build_graph(self, key):
        S_p, n_vis, model_times = key
        if model_times != self._model_times:
            raise ValueError("HyVLA graph time table does not match this sampler")
        b = self._allocate_flow_buffers(S_p)
        prefix_graph = self._capture_flow_body(lambda: self._prefix_body(b, n_vis))
        # All per-time graphs use the same mutable inputs and external output.
        # Never share private pools: adaptive branches replay in varying order.
        velocity_graphs = {
            time: self._capture_flow_body(
                lambda index=index: self._velocity_body(b, index, S_p))
            for index, time in enumerate(model_times)
        }
        return {"buf": b, "prefix_graph": prefix_graph,
                "velocity_graphs": velocity_graphs}

    def _sample_latent(self, b, evaluate):
        if self.sampler == "probeflow":
            metadata = sample_probeflow_torch_(b["x"], evaluate, self.probeflow_config)
        else:
            # Separate multiply then add, like the original dt * v + x Euler
            # body, rather than add_(..., alpha=...) with different rounding.
            for time in self._model_times:
                progress_velocity = evaluate(time)
                if self.numerical_profile == "official_local_fp32":
                    # Official dt is a 0-D FP32 device tensor but v is BF16:
                    # wrapped-scalar promotion leaves dt*v BF16, before the
                    # FP32 latent add. This is distinct from default FP32 Euler.
                    increment = self._euler_step_size * progress_velocity.to(_BF16)
                else:
                    increment = progress_velocity * (1.0 / self.num_steps)
                b["x"].add_(increment)
            metadata = {"sampler": "euler", "scheduled_steps": self.num_steps,
                        "nfe": self.num_steps, "timesteps": list(self._model_times)}
        metadata["numerical_profile"] = self.numerical_profile
        metadata["operation_dtypes"] = self._operation_dtypes()
        metadata["embedding_timesteps"] = [
            self._embedding_times[self._time_index[time]] for time in metadata["timesteps"]]
        metadata["profile_scope"] = "local_math_hooks_not_official_exact"
        self._last_sampler_metadata = metadata
        return b["x"].clone()

    def _operation_dtypes(self):
        official = self.numerical_profile == "official_local_fp32"
        return {
            "weights_and_projections": "bfloat16",
            "rope_inv_freq_storage": "bfloat16",
            "rope_tables_and_rotated_qk": "float32" if official else "bfloat16",
            "key_cache": "float32" if official else "bfloat16",
            "value_cache": "bfloat16",
            "attention_logits_and_softmax": "float32" if official else "sdpa_backend_selected",
            "attention_probabilities": "bfloat16" if official else "sdpa_backend_selected",
            "attention_value_matmul": "bfloat16" if official else "sdpa_backend_selected",
            "vision_attention": "unchanged_native_sdpa_and_bfloat16_temporal",
            "rms_norm": "float32_reduce_then_input_dtype_cast_then_weight" if official else "torch_rms_norm",
            "raw_velocity": "bfloat16",
            "progress_velocity_and_latent": "float32",
            "integration": "float32",
            "euler_increment": "bfloat16" if official and self.sampler == "euler" else "float32",
        }

    @torch.no_grad()
    def _graph_forward(self, S_p, n_vis, prefix_embs, pmask, pcos, psin,
                       smask, scos, ssin, state_t, noise_t, use_graph=True):
        key = (S_p, n_vis, self._model_times)
        if use_graph:
            entry = self._graph_cache.get(key)
            if entry is None:
                entry = self._build_graph(key)
                self._graph_cache[key] = entry
            b = entry["buf"]
        else:
            b = self._allocate_flow_buffers(S_p)
        # Every request refreshes all mutable prefix, mask, RoPE, state and
        # noise tensors. Same-shape prompt changes must not reuse old content.
        for name, value in (
                ("pe", prefix_embs), ("pmask", pmask), ("pcos", pcos), ("psin", psin),
                ("smask", smask), ("scos", scos), ("ssin", ssin),
                ("state", state_t), ("x", noise_t)):
            b[name].copy_(value)
        if use_graph:
            entry["prefix_graph"].replay()

            def evaluate(time):
                entry["velocity_graphs"][time].replay()
                return b["velocity"]
        else:
            self._prefix_body(b, n_vis)

            def evaluate(time):
                self._velocity_body(b, self._time_index[time], S_p)
                return b["velocity"]
        # Host scalar routing and all integration stay outside CUDA capture.
        return self._sample_latent(b, evaluate)

    @torch.no_grad()
    def predict_actions(self, images, prompt=None, state=None, noise=None,
                        use_graph=None):
        """Return the original (1, chunk, max_action_dim) raw FP32 NumPy chunk."""
        if use_graph is not None and not isinstance(use_graph, bool):
            raise TypeError("use_graph must be a bool or None")
        # The constructor switch covers ViT as well as prefix/velocity graphs.
        # A globally eager instance cannot accidentally re-enable ViT capture.
        use_graph = self.use_cuda_graph and (use_graph is not False)
        # The native preprocessing helper accepts camera dicts/lists, but the
        # base predict_actions coerces through np.asarray before reaching it.
        # Canonicalize containers with that same native camera helper first;
        # native resize/pad and [-1,1] normalization still run exactly once.
        if isinstance(images, dict):
            keys = [key for key in self.image_keys if key in images]
            if not keys:
                keys = [key for key in ("image", "wrist_image", "wrist_image_right")
                        if key in images]
            if not keys:
                raise ValueError("images dict does not contain configured camera keys")
            images = [images[key] for key in keys]
        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("images list must have at least one camera")
            images = torch.stack([_camera_tensor(image) for image in images], 0)
        return super().predict_actions(images, prompt=prompt, state=state,
                                       noise=noise, use_graph=use_graph)

    @torch.no_grad()
    def infer(self, obs):
        """Preserve native observation/action semantics and report actual NFE."""
        result = super().infer(obs)
        result["sampler_metadata"] = dict(self._last_sampler_metadata)
        return result


__all__ = ["HyVLAFlowFrontend"]
