"""Real FastWAM first-frame VAE/video cache and nonintegrating action field.

Dataflow follows upstream fastwam.py::infer_action at commit 7faa711.
Only the integration loop is separate; infer()/infer_joint()/infer_action()
are NEVER called to approximate a velocity. Model timestep=1000*sigma in
the released configs; the FM update delta is sigma_next-sigma, not dt*1000.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Integral
import threading

import torch

from flash_rt.models.fastwam.config import UPSTREAM_COMMIT, UPSTREAM_URL, positive_integer, vae_spatial_factor
from flash_rt.sampling.schedules import ShiftedSigmaSchedule
from flash_rt.sampling.torch_probeflow import sample_probeflow_torch_


@dataclass(frozen=True)
class FastWAMCondition:
    context: torch.Tensor
    context_mask: torch.Tensor
    video_cache_k: list[torch.Tensor]
    video_cache_v: list[torch.Tensor]
    action_attention_mask: torch.Tensor


class FastWAMActionPipeline:
    """Action-only orchestration, with per-observation cache reuse.

    Optional graphs capture the raw field only, never sampling decisions.
    Caches live only for one infer call; changing an image/prompt/proprio cannot
    accidentally reuse a previous observation's video KV. Concurrent calls on
    one frontend are serialized, including the native runner's workspace.
    """

    def __init__(self, model, *, action_horizon, sigma_shift, sampler,
                 num_inference_steps, probeflow_config, native_linear, device, dtype,
                  native_linear_backend="legacy", use_cuda_graph=False):
        if not isinstance(use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be bool")
        if use_cuda_graph and (device.type != "cuda" or (native_linear and native_linear_backend != "cached")):
            raise ValueError("FastWAM CUDA graphs require CUDA and Torch or cached native Linear")
        if native_linear_backend not in ("legacy", "cached"):
            raise ValueError("native_linear_backend must be 'legacy' or 'cached'")
        if native_linear_backend == "cached" and not native_linear:
            raise ValueError("native_linear_backend='cached' requires native_linear=True")
        self.model = model
        self.native_linear_backend = native_linear_backend
        self.device, self.dtype = device, dtype
        self.action_horizon, self.sigma_shift = action_horizon, sigma_shift
        self.schedule = ShiftedSigmaSchedule(sigma_shift)
        self.sampler, self.num_inference_steps = sampler, num_inference_steps
        self.probeflow_config = probeflow_config
        self._lock = threading.RLock()
        self._closing = self._closed = False
        self._execution_stream = self._velocity_graph = None
        self._last_eager_stream = None
        self._native_modules, self._native_skipped = {}, ()
        self._native_runner = None
        self._validate_model()
        self._build_time_bank()
        model.eval()
        if native_linear:
            from flash_rt.models.fastwam.native_linear import install_native_linears
            self._native_modules, self._native_skipped = install_native_linears(
                model, backend=native_linear_backend
            )
            self._native_runner = next(iter(self._native_modules.values()))._runner
            if any(module._runner is not self._native_runner for module in self._native_modules.values()):
                raise ValueError("FastWAM native Linear modules must share one owned runner")
        if use_cuda_graph:
            from flash_rt.models.fastwam.velocity_graph import FastWAMVelocityGraph
            with torch.cuda.device(self.device):
                self._execution_stream = torch.cuda.Stream(device=self.device)
                self._execution_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._velocity_graph = FastWAMVelocityGraph(model._denoise_action_with_video_cache)
        elif self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                self._last_eager_stream = torch.cuda.current_stream(self.device)

    @staticmethod
    def _record_stream(value, stream):
        if isinstance(value, torch.Tensor):
            if value.device.type == 'cuda':
                value.record_stream(stream)
        elif isinstance(value, FastWAMCondition):
            for item in vars(value).values():
                FastWAMActionPipeline._record_stream(item, stream)
        elif isinstance(value, Mapping):
            for item in value.values():
                FastWAMActionPipeline._record_stream(item, stream)
        elif isinstance(value, (list, tuple)):
            for item in value:
                FastWAMActionPipeline._record_stream(item, stream)

    @contextmanager
    def _execution_scope(self, *inputs):
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError('FastWAM pipeline is closed or closing')
            if self._velocity_graph is not None:
                self._velocity_graph._check_live()
            stream = self._execution_stream
            if stream is None:
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        caller = torch.cuda.current_stream(self.device)
                        previous = self._last_eager_stream
                        if previous is not None and caller != previous:
                            caller.wait_stream(previous)
                        # Chain eager callers too: the final stream represents
                        # all prior submitted work, including borrowed timestep
                        # storage and the shared native workspace. Retain it
                        # even when a public asynchronous helper raises.
                        self._last_eager_stream = caller
                        self._record_stream(inputs, caller)
                        yield
                else:
                    yield
                return
            with torch.cuda.device(self.device):
                caller = torch.cuda.current_stream(self.device)
                stream.wait_stream(caller)
                self._record_stream(inputs, stream)
                try:
                    with torch.cuda.stream(stream):
                        yield
                finally:
                    caller.wait_stream(stream)

    def _validate_model(self):
        for name in ("_encode_input_image_latents_tensor", "_append_proprio_to_context",
                     "_build_mot_attention_mask", "_denoise_action_with_video_cache"):
            if not callable(getattr(self.model, name, None)):
                raise TypeError(f"Initialized FastWAM model is missing the pinned action-only seam: {name}")
        if not callable(getattr(self.model.mot, "prefill_video_cache_tensor", None)):
            raise TypeError("FastWAM model requires MoT.prefill_video_cache_tensor")
        video, action = self.model.video_expert, self.model.action_expert
        if getattr(video, "video_attention_mask_mode", None) != "first_frame_causal":
            raise ValueError("FastWAM first-frame inference requires first_frame_causal video attention")
        if not video.fuse_vae_embedding_in_latents or not video.seperated_timestep:
            raise ValueError("FastWAM first-frame inference requires fused VAE and separated timesteps")
        patch_size = tuple(video.patch_size)
        if len(patch_size) != 3:
            raise ValueError("FastWAM video patch_size must be a positive THW triple")
        self.patch_size = tuple(positive_integer(value, "video.patch_size") for value in patch_size)
        if self.patch_size[0] != 1:
            raise ValueError("FastWAM first-frame inference requires temporal patch_size=1")
        self.vae_spatial_factor = vae_spatial_factor(getattr(self.model, "vae", None))
        self.image_multiple = tuple(scale * patch for scale, patch in zip(
            self.vae_spatial_factor, self.patch_size[1:]
        ))
        if action.action_dim <= 0 or not 0 < self.action_horizon <= action.freqs.shape[0]:
            raise ValueError("action_horizon must fit the upstream action RoPE cache")
        model_device = torch.device(self.model.device)
        if self.device.type == "cuda" and self.device.index is None:
            raise ValueError("FastWAM pipeline requires an explicit canonical CUDA device ordinal")
        if model_device.type != self.device.type or (
            model_device.index is not None and model_device != self.device
        ) or self.model.torch_dtype != self.dtype:
            raise ValueError("Injected model device/dtype must match the requested device/dtype")
        # All explicit device declarations must agree. An implicit top-level
        # FastWAM device is pinned below only AFTER validating actual tensors;
        # no injected parameter/buffer is moved or cast.
        for owner in (video, action, self.model.mot, self.model.vae,
                      getattr(self.model.vae, "model", None),
                      getattr(self.model, "text_encoder", None)):
            declared = getattr(owner, "device", None)
            if declared is not None and torch.device(declared) != self.device:
                raise ValueError("Injected FastWAM component device must match the fixed device")
        if callable(getattr(self.model, "named_parameters", None)):
            for name, parameter in self.model.named_parameters():
                if (parameter.device != self.device
                        or (parameter.is_floating_point() and parameter.dtype != self.dtype)):
                    raise ValueError(f"FastWAM parameter {name} does not match requested device/dtype")
        if callable(getattr(self.model, "named_buffers", None)):
            for name, buffer in self.model.named_buffers():
                if buffer.device != self.device:
                    raise ValueError(f"FastWAM buffer {name} does not match the fixed device")
        if model_device != self.device:
            # Upstream image/proprio helpers consult model.device internally.
            # Canonicalizing this metadata prevents ambient-device migration.
            self.model.device = self.device
        self.action_dim = int(action.action_dim)
        self.time_scale = int(self.model.infer_action_scheduler.num_train_timesteps)
        if self.time_scale <= 0:
            raise ValueError("FastWAM num_train_timesteps must be positive")

    def _build_time_bank(self):
        """Validate/cast the entire schedule on CPU, then upload once.

        The full ProbeFlow clock union includes both probes and every reachable
        dense branch. Callback lookup performs no allocation or CUDA host read.
        Metadata is retained from CPU casts, not read back from CUDA per NFE.
        """
        if self.sampler == "probeflow":
            clocks = tuple(self.probeflow_config.timesteps())
        else:
            clocks = tuple(1.0 - index / self.num_inference_steps
                           for index in range(self.num_inference_steps))
        sigmas = tuple(self.schedule(clock) for clock in clocks)
        cpu_times = torch.tensor([sigma * self.time_scale for sigma in sigmas],
                                 device="cpu", dtype=self.dtype)
        rounded_times = tuple(float(value) for value in cpu_times.tolist())
        if any(not math.isfinite(value) or value <= 0 for value in rounded_times):
            raise ValueError("Positive FastWAM timesteps must be representable in the model dtype")
        self._clock_keys = clocks
        self._sigma_by_clock = dict(zip(clocks, sigmas))
        self._model_time_by_clock = dict(zip(clocks, rounded_times))
        if self.sampler == "euler":
            endpoints = (*sigmas[1:], 0.0)
            cpu_deltas = torch.tensor([right - left for left, right in zip(sigmas, endpoints)],
                                      device="cpu", dtype=self.dtype)
            self._rounded_deltas = tuple(float(value) for value in cpu_deltas.tolist())
            if any(not math.isfinite(value) or value >= 0 for value in self._rounded_deltas):
                raise ValueError("Euler sigma deltas must be representable and strictly negative in the model dtype")
        # All CPU representability checks precede device upload. Preserve
        # tensor owners and pre-created [1] views for every callback clock.
        self._time_bank = cpu_times.reshape(-1, 1).to(device=self.device)
        self._time_rows = dict(zip(clocks, self._time_bank.unbind(0)))
        self._sigma_rows = {sigma: self._time_rows[clock] for clock, sigma in zip(clocks, sigmas)}
        if self.sampler == "euler":
            self._delta_bank = cpu_deltas.to(device=self.device)
            self._delta_rows = tuple(self._delta_bank.unbind(0))

    def execution_metadata(self):
        return {
            "model": "fastwam", "upstream_url": UPSTREAM_URL,
            "source_contract_commit": UPSTREAM_COMMIT,
            "action_infer_mode": "first_frame",
            "backend": "flashrt_bf16_linear_partial" if self._native_modules else "upstream_torch_reference",
            "fully_native": False,
            "native_linear_modules": tuple(self._native_modules),
            "native_linear_skipped": self._native_skipped,
            "native_linear_backend": self.native_linear_backend if self._native_modules else None,
            "upstream_operations": ("VAE", "patch_conv3d", "attention_sdpa", "norm", "RoPE", "activations", "modulation"),
            "cuda_graph": self._velocity_graph is not None,
            "native_counter_semantics": "Python Linear forwards / successful CPU GEMM enqueues, not graph GPU work",
            "device": str(self.device),
            "image_spatial_multiple": self.image_multiple,
        }

    def _native_plan_snapshot(self):
        """Read CPU counters once at request boundaries, never per Linear."""
        if self.native_linear_backend != "cached" or self._native_runner is None:
            return None
        counters = dict(self._native_runner.stats())
        keys = ("plan_builds", "heuristic_queries", "launches")
        if any(key not in counters or isinstance(counters[key], bool)
               or not isinstance(counters[key], Integral) or counters[key] < 0 for key in keys):
            raise ValueError("Cached GEMM stats require nonnegative integer CPU counters")
        return {key: int(counters[key]) for key in keys}

    @staticmethod
    def _floating_tensor(value, name):
        tensor = torch.as_tensor(value)
        if not tensor.is_floating_point() or tensor.numel() == 0 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must be a nonempty finite floating tensor")
        return tensor

    def _observation_tensors(self, obs):
        if not isinstance(obs, Mapping):
            raise TypeError("FastWAM obs must be a mapping of already processed upstream inputs")
        unknown = set(obs) - {"input_image", "context", "context_mask", "proprio", "prompt"}
        if unknown:
            raise ValueError(f"Unknown FastWAM observation keys: {sorted(unknown)}")
        if "input_image" not in obs:
            raise ValueError("FastWAM obs requires input_image")
        image = self._floating_tensor(obs["input_image"], "input_image")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
            raise ValueError("input_image must be [3,H,W] or [1,3,H,W]")
        if any(size % multiple for size, multiple in zip(image.shape[-2:], self.image_multiple)):
            raise ValueError(f"input_image spatial dimensions must be multiples of {self.image_multiple} "
                             "from the actual VAE factor and DiT patch size")
        if image.min().item() < -1 or image.max().item() > 1:
            raise ValueError("input_image must already be normalized to [-1,1]")

        prompt, context, mask = obs.get("prompt"), obs.get("context"), obs.get("context_mask")
        if prompt is not None:
            if context is not None or mask is not None:
                raise ValueError("prompt and context/context_mask are mutually exclusive")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be a nonempty string")
            if getattr(self.model, "text_encoder", None) is None or getattr(self.model, "tokenizer", None) is None:
                raise ValueError("Prompt encoding requires an already loaded upstream T5; provide context/context_mask")
        elif context is None or mask is None:
            raise ValueError("Both prepared context/context_mask are required when prompt is absent")

        proprio = obs.get("proprio")
        if self.model.proprio_dim is not None and proprio is None:
            raise ValueError("proprio is required by this checkpoint's proprio_dim")
        if proprio is not None:
            if self.model.proprio_dim is None:
                raise ValueError("proprio was provided but the checkpoint's proprio encoder is disabled")
            proprio = self._floating_tensor(proprio, "proprio")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if tuple(proprio.shape) != (1, self.model.proprio_dim):
                raise ValueError("proprio must have checkpoint-normalized shape [D] or [1,D]")
            proprio = proprio.to(device=self.device, dtype=self.dtype)

        if prompt is not None:
            context, mask = self.model.encode_prompt(prompt)
        context = self._floating_tensor(context, "context")
        mask = torch.as_tensor(mask)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if (context.ndim != 3 or context.shape[0] != 1 or context.shape[2] != self.model.text_dim
                or tuple(mask.shape) != tuple(context.shape[:2]) or mask.dtype != torch.bool):
            raise ValueError("context/context_mask must be [1,L,text_dim]/[1,L] with a boolean mask (unbatched accepted)")
        if not mask.any().item():
            raise ValueError("context_mask must contain at least one valid text token")
        image = image.to(device=self.device, dtype=self.dtype)
        context = context.to(device=self.device, dtype=self.dtype)
        if not torch.isfinite(context).all().item() or (
            proprio is not None and not torch.isfinite(proprio).all().item()
        ):
            raise ValueError("context/proprio is not finite after conversion to the model dtype")
        return image, context, mask.to(device=self.device), proprio

    def _initial_state(self, noise):
        shape = (1, self.action_horizon, self.action_dim)
        if noise is None:
            # Match upstream rand_device='cpu', float32 noise then model cast.
            return torch.randn(shape, device="cpu", dtype=torch.float32).to(device=self.device, dtype=self.dtype)
        state = self._floating_tensor(noise, "noise")
        if tuple(state.shape) != shape:
            raise ValueError(f"noise must have shape {shape}")
        state = state.to(device=self.device, dtype=self.dtype).clone()
        if not torch.isfinite(state).all().item():
            raise ValueError("noise is not finite after conversion to the model dtype")
        return state

    @torch.inference_mode()
    def prepare_condition(self, obs):
        with self._execution_scope(obs):
            result = self._prepare_condition(obs)
        if self._execution_stream is not None:
            self._record_stream(result, torch.cuda.current_stream(self.device))
        return result

    def _prepare_condition(self, obs):
        image, context, mask, proprio = self._observation_tensors(obs)
        if proprio is not None:
            context, mask = self.model._append_proprio_to_context(context, mask, proprio)
        latents = self.model._encode_input_image_latents_tensor(input_image=image, tiled=False)
        if latents.ndim != 5 or tuple(latents.shape[:1]) != (1,) or latents.shape[2] != 1:
            raise ValueError("First-frame VAE latents must be [1,C,1,H,W]")
        if any(size % patch for size, patch in zip(latents.shape[-3:], self.patch_size)):
            raise ValueError("VAE latent dimensions must be divisible by the actual DiT patch_size")
        expected_hw = tuple(size // scale for size, scale in zip(image.shape[-2:], self.vae_spatial_factor))
        if tuple(latents.shape[-2:]) != expected_hw:
            raise ValueError("VAE latent spatial dimensions do not match the declared upsampling_factor")
        (tokens, _t, t_mod, video_context, video_mask, freqs,
         _f, _h, _w, tokens_per_frame) = self.model.video_expert.prepare(
            x=latents,
            timestep=torch.zeros((1,), device=self.device, dtype=latents.dtype),
            context=context, context_mask=mask, action=None,
            fuse_vae_embedding_in_latents=True,
        )
        seq_len = int(tokens.shape[1])
        attention_mask = self.model._build_mot_attention_mask(
            video_seq_len=seq_len, action_seq_len=self.action_horizon,
            video_tokens_per_frame=tokens_per_frame, device=tokens.device,
        )
        cache_k, cache_v = self.model.mot.prefill_video_cache_tensor(
            video_tokens=tokens, video_freqs=freqs, video_t_mod=t_mod,
            video_context=video_context, video_context_mask=video_mask,
            video_attention_mask=attention_mask[:seq_len, :seq_len],
        )
        return FastWAMCondition(context, mask, cache_k, cache_v, attention_mask[seq_len:, :])

    def _evaluate_at_timestep(self, state, timestep, condition, *, graph=False):
        if timestep.shape != (1,) or timestep.device != state.device or timestep.dtype != state.dtype:
            raise ValueError("FastWAM banked timestep must match state device/dtype and have shape [1]")
        evaluator = self._velocity_graph if graph and self._velocity_graph is not None else self.model._denoise_action_with_video_cache
        raw = evaluator(
            latents_action=state, timestep_action=timestep,
            context=condition.context, context_mask=condition.context_mask,
            video_cache_k=condition.video_cache_k, video_cache_v=condition.video_cache_v,
            action_attention_mask=condition.action_attention_mask,
        )
        if raw.shape != state.shape or raw.device != state.device or raw.dtype != state.dtype:
            raise ValueError("FastWAM raw velocity must match action state shape/device/dtype")
        return raw

    def _evaluate_clock_velocity(self, state, clock, condition):
        try:
            timestep = self._time_rows[clock]
        except KeyError as exc:
            raise ValueError("Clock is outside the precomputed FastWAM schedule union") from exc
        return self._evaluate_at_timestep(state, timestep, condition, graph=True)

    @torch.inference_mode()
    def evaluate_velocity(self, state, sigma, condition):
        """Return raw v for a precomputed physical sigma, without integration.

        Arbitrary off-grid evaluations require rebuilding the sampler config;
        inference never creates a variable timestep allocation in the hot path.
        """
        with self._execution_scope(state, condition):
            if not math.isfinite(sigma) or not 0 < sigma <= 1:
                raise ValueError("FastWAM velocity sigma must lie in (0,1]")
            try:
                timestep = self._sigma_rows[sigma]
            except KeyError as exc:
                raise ValueError("Sigma is outside the precomputed FastWAM schedule union") from exc
            result = self._evaluate_at_timestep(state, timestep, condition)
        if self._execution_stream is not None:
            self._record_stream(result, torch.cuda.current_stream(self.device))
        return result

    def _euler(self, state, condition):
        scheduler = self.model.infer_action_scheduler
        for clock, delta in zip(self._clock_keys, self._delta_rows):
            velocity = self._evaluate_clock_velocity(state, clock, condition)
            # Preserve upstream multiply-then-add rounding (not add_(alpha=)).
            state = scheduler.step(velocity, delta, state)
        return state, {"sampler": "euler", "scheduled_steps": len(self._clock_keys),
                       "nfe": len(self._clock_keys),
                       "model_timesteps": [self._model_time_by_clock[t] for t in self._clock_keys],
                       "timesteps": list(self._clock_keys),
                       "integration_times": [self._sigma_by_clock[t] for t in self._clock_keys],
                       "time_grid": "mapped", "sigma_deltas": list(self._rounded_deltas)}

    @torch.inference_mode()
    def infer(self, obs, noise=None):
        with self._execution_scope(obs, noise):
            before = {name: module.calls for name, module in self._native_modules.items()}
            plans_before = self._native_plan_snapshot()
            state = self._initial_state(noise)
            condition = self._prepare_condition(obs)
            graph = self._velocity_graph
            if graph is not None:
                graph.begin_request()
            try:
                if self.sampler == "probeflow":
                    # Callback keys are logical u; model/integration use sigma.
                    # Neither a Jacobian nor time_scale multiplies the raw field.
                    metadata = sample_probeflow_torch_(
                        state, lambda clock: -self._evaluate_clock_velocity(state, clock, condition),
                        self.probeflow_config, time_map=self.schedule,
                    )
                    metadata["model_timesteps"] = [
                        self._model_time_by_clock[t] for t in metadata["timesteps"]
                    ]
                else:
                    state, metadata = self._euler(state, condition)
            finally:
                if graph is not None:
                    graph.end_request()
            if not torch.isfinite(state).all().item():
                raise ValueError("FastWAM integration produced nonfinite actions")
            metadata.update(sigma_shift=self.sigma_shift, model_time_scale=self.time_scale,
                            action_infer_mode="first_frame", video_prefill_count=1)
            if self.sampler == "probeflow" and self.sigma_shift != 1.0:
                metadata["adaptation"] = "shifted_sigma_grid_not_original_paper_validation"
            coverage = self.execution_metadata()
            if graph is not None:
                coverage['velocity_graph'] = graph.snapshot()
            coverage["native_linear_calls"] = {
                name: module.calls - before[name] for name, module in self._native_modules.items()
            }
            if plans_before is not None:
                plans_after = self._native_plan_snapshot()
                coverage["native_plan_stats"] = {
                    key: plans_after[key] - value for key, value in plans_before.items()
                }
            return {"actions": state.detach().to(device="cpu", dtype=torch.float32).numpy(),
                    "sampler_metadata": metadata, "execution_metadata": coverage,
                    "actions_normalized": True}

    def close(self):
        """Stop admission, drain own work, reset own graph, then drop owners."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            if self._execution_stream is not None:
                self._execution_stream.synchronize()
            elif self._last_eager_stream is not None:
                self._last_eager_stream.synchronize()
            if self._velocity_graph is not None:
                self._velocity_graph.close()
            self._velocity_graph = self._execution_stream = None
            self._last_eager_stream = None
            self._native_modules.clear()
            self._native_runner = self.model = None
            self._time_rows.clear()
            self._sigma_rows.clear()
            self._time_bank = self._delta_bank = None
            self._delta_rows = ()
            self._closed = True
