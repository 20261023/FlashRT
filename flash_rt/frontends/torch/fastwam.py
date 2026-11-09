"""FastWAM action-only frontend with explicit partial FlashRT execution.

This is yuantianyuan01/FastWAM, not Pi0-FAST and not Motus. Local loading
needs a full trained checkpoint, resolved upstream model config and local Wan
VAE. Input/output robot transforms belong to the caller's checkpoint processor.
Importing this module does not import Torch/upstream models or initialize CUDA.
"""

from __future__ import annotations

from flash_rt.models.fastwam.config import positive_integer, validate_sampler
from flash_rt.models.fastwam.loader import read_model_config, resolve_local_paths


def _canonical_device(device, torch):
    """Resolve an implicit CUDA ordinal once, not during later allocations."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class FastWAMTorchFrontend:
    """First-frame action inference only; no video rollout or IDM rollout.

    ``sigma_shift`` is deliberately mandatory: original releases used 5,
    while the optional-IDM release uses 1. Euler and ProbeFlow share the same
    scalar shifted-sigma mapping. Shifted ProbeFlow is an explicit adaptation,
    not an original-paper accuracy/performance validation claim.
    ``native_linear=True`` opts into FlashRT BF16 GEMMs for the supported
    ActionDiT/VideoDiT linear modules. Other operations remain upstream Torch;
    this is NOT a fully native FlashRT model port or a performance claim.
    ``native_linear_backend='legacy'`` preserves the old binary by default;
    ``'cached'`` explicitly requires the standalone host-side plan extension.
    ``use_cuda_graph=False`` keeps eager execution. Opt-in captures only the raw
    field, on one owned ordered stream, with one fixed input geometry. Native
    graphs require cached Linear; changing geometry raises instead of recapture
    or fallback. Prepare/capture must occur while other CUDA work in this process
    is quiescent; an instance lock cannot coordinate unrelated GPU threads.
    Sampler decisions, VAE and per-request video prefill remain outside capture.

    Alternatively pass ``model=`` with an already initialized upstream model;
    its weights/device/dtype are not silently changed. Source/checkpoint loading
    options are mutually exclusive with that injection boundary.
    """

    def __init__(
        self,
        checkpoint_dir=None,
        *,
        sigma_shift: float,
        action_horizon: int,
        source_dir=None,
        model_config=None,
        backbone_dir=None,
        model=None,
        device=None,
        dtype=None,
        sampler: str = "euler",
        num_inference_steps: int | None = None,
        probeflow_config=None,
        native_linear: bool = False,
        native_linear_backend: str = "legacy",
        use_cuda_graph: bool = False,
    ):
        steps, probe = validate_sampler(sampler, sigma_shift, num_inference_steps, probeflow_config)
        horizon = positive_integer(action_horizon, "action_horizon")
        if not isinstance(native_linear, bool):
            raise TypeError("native_linear must be bool")
        if not isinstance(use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be bool")
        if use_cuda_graph and native_linear and native_linear_backend != "cached":
            raise ValueError("FastWAM CUDA graph with native Linear requires cached backend")
        if use_cuda_graph and device is not None and not str(device).startswith("cuda"):
            raise ValueError("use_cuda_graph requires a CUDA device")
        if native_linear_backend not in ("legacy", "cached"):
            raise ValueError("native_linear_backend must be 'legacy' or 'cached'")
        if native_linear_backend == "cached" and not native_linear:
            raise ValueError("native_linear_backend='cached' requires native_linear=True")
        if native_linear and device is not None and not str(device).startswith("cuda"):
            raise ValueError("native_linear requires a CUDA device")
        if native_linear and dtype is not None and str(dtype) not in (
            "bf16", "bfloat16", "torch.bfloat16"
        ):
            raise ValueError("native_linear supports BF16 only")
        if model is not None:
            if any(value is not None for value in (checkpoint_dir, source_dir, model_config, backbone_dir)):
                raise ValueError("model injection is mutually exclusive with source/checkpoint loading options")
        else:
            # All path/config failures occur before importing Torch or any device API.
            config = read_model_config(model_config)
            paths = resolve_local_paths(checkpoint_dir, backbone_dir, source_dir)

        import torch
        from flash_rt.models.fastwam.pipeline import FastWAMActionPipeline

        if dtype is None:
            dtype = getattr(model, "torch_dtype", torch.bfloat16)
        elif isinstance(dtype, str):
            aliases = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                       "fp16": torch.float16, "float16": torch.float16,
                       "fp32": torch.float32, "float32": torch.float32}
            if dtype not in aliases:
                raise ValueError("dtype must be bf16/bfloat16, fp16/float16, or fp32/float32")
            dtype = aliases[dtype]
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("FastWAM requires a floating model dtype")
        device = _canonical_device(device if device is not None else getattr(model, "device", "cuda"), torch)
        if native_linear and (dtype != torch.bfloat16 or device.type != "cuda"):
            raise ValueError("native_linear requires CUDA BF16")
        if use_cuda_graph and device.type != "cuda":
            raise ValueError("use_cuda_graph requires a CUDA device")
        if model is None:
            from flash_rt.models.fastwam.loader import load_local_model
            model = load_local_model(config, *paths, device=device, dtype=dtype)
        self.model = model
        self.pipeline = FastWAMActionPipeline(
            model, action_horizon=horizon, sigma_shift=float(sigma_shift),
            sampler=sampler, num_inference_steps=steps, probeflow_config=probe,
            native_linear=native_linear, device=device, dtype=dtype,
            native_linear_backend=native_linear_backend,
            use_cuda_graph=use_cuda_graph,
        )
        self.sampler = sampler
        self.device, self.dtype = device, dtype
        self.sigma_shift = float(sigma_shift)
        self.action_horizon = horizon
        self.num_inference_steps = steps
        self.native_linear_backend = native_linear_backend
        self.use_cuda_graph = use_cuda_graph

    @property
    def execution_metadata(self):
        return self.pipeline.execution_metadata()

    def infer(self, obs, noise=None) -> dict:
        """Return normalized float32 NumPy ``actions[1,H,A]`` and actual NFE.

        ``obs`` accepts ``input_image`` ([3,H,W]/[1,3,H,W], already [-1,1],
        spatial multiples of the actual VAE factor times the DiT patch size;
        32 for the released VAE38/2x2 DiT), ``context/context_mask`` (precomputed upstream
        T5 embeddings/mask), and checkpoint-normalized ``proprio`` ([D]/[1,D]).
        An injected model with an already loaded T5 may use ``prompt`` instead
        of context/mask. Camera merging, image resizing, normalization stats,
        relative actions, rotations and action unnormalization are NOT guessed.
        Optional noise is floating [1,action_horizon,action_dim]; it is copied.
        """
        return self.pipeline.infer(obs, noise=noise)

    def close(self):
        """Release owned graph/pipeline references; failed resets can be retried."""
        self.pipeline.close()
        self.model = None
