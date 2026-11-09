"""Opt-in, instance-local FlashRT BF16 Linear bindings for real FastWAM.

The pinned upstream MoT explicitly calls q/k/v/o modules and Sequential FFNs;
it does not bypass these modules with F.linear. No global monkeypatch is used.
The native layout/stream contract is csrc/bindings.cpp::bf16_nt_bias:
    D[M,N] = A[M,K] @ B[N,K].T + bias[N]
This avoids a second transposed copy of the multi-billion-parameter towers.
"""

from __future__ import annotations

import importlib

import torch
from torch import nn


class FlashRTLinear(nn.Module):
    """Inference-only Linear retaining the original Parameter/state_dict names.

    Owns Parameters and the shared runner by strong references. Activations and
    weights are made contiguous in their existing [M,K]/[N,K] layouts. Native
    failures propagate: there is no silent PyTorch fallback. Installation is
    intended after loading/eval; training, hooks and parametrized linears are
    outside this narrow replacement contract.
    """

    def __init__(self, linear: nn.Linear, runner):
        super().__init__()
        if type(linear) is not nn.Linear:
            raise TypeError("FlashRTLinear replaces only ordinary nn.Linear modules")
        if linear.weight.dtype != torch.bfloat16 or linear.weight.device.type != "cuda":
            raise ValueError("FlashRTLinear weights must already be CUDA BF16")
        if linear.bias is not None and (
            linear.bias.dtype != torch.bfloat16 or linear.bias.device != linear.weight.device
        ):
            raise ValueError("FlashRTLinear bias must match CUDA BF16 weights")
        if not callable(getattr(runner, "bf16_nt_bias", None)):
            raise RuntimeError("FlashRT GemmRunner.bf16_nt_bias is required; rebuild the native extension")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight
        self.register_parameter("bias", linear.bias)
        self.register_buffer(
            "_zero_bias",
            torch.zeros(self.out_features, dtype=torch.bfloat16, device=linear.weight.device)
            if linear.bias is None else None,
            persistent=False,
        )
        self._runner = runner
        self.calls = 0
        self.train(linear.training)

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("FlashRTLinear is inference-only; use eval() and torch.inference_mode()")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("FlashRTLinear input last dimension must equal in_features")
        if x.device != self.weight.device or not x.is_floating_point():
            raise ValueError("FlashRTLinear input must be floating on the weight CUDA device")
        if self.weight.dtype != torch.bfloat16 or self.weight.device.type != "cuda":
            raise ValueError("FlashRTLinear must remain CUDA BF16 after installation")
        shape = (*x.shape[:-1], self.out_features)
        activation = x.to(dtype=torch.bfloat16).contiguous().reshape(-1, self.in_features)
        output = torch.empty((activation.shape[0], self.out_features),
                             device=x.device, dtype=torch.bfloat16)
        if activation.shape[0] == 0:
            return output.reshape(shape)
        weight = self.weight.contiguous()
        bias = self.bias if self.bias is not None else self._zero_bias
        if bias is None or bias.device != weight.device or bias.dtype != torch.bfloat16:
            raise ValueError("FlashRTLinear bias must remain CUDA BF16")
        bias = bias.contiguous()
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream(x.device)
            self._runner.bf16_nt_bias(
                activation.data_ptr(), weight.data_ptr(), output.data_ptr(), bias.data_ptr(),
                activation.shape[0], self.out_features, self.in_features, stream.cuda_stream,
            )
            # These are raw pointers, invisible to Torch's stream/lifetime tracker.
            for tensor in (activation, weight, bias, output):
                tensor.record_stream(stream)
        self.calls += 1
        return output.reshape(shape)


def _active_linear_path(expert_name, name):
    if name.startswith("blocks."):
        return True
    prefixes = ("text_embedding.", "time_embedding.", "time_projection.")
    if name.startswith(prefixes):
        return True
    return expert_name == "action_expert" and name in ("action_encoder", "head")


def create_native_runner(backend, device):
    """Create exactly the requested native runner; cached never falls back.

    The cached extension is deliberately independent of the legacy binary and
    has no Torch C++ dependency. Both runners share the same pointer API.
    """
    if backend not in ("legacy", "cached"):
        raise ValueError("native_linear_backend must be 'legacy' or 'cached'")
    if backend == "cached":
        try:
            module = importlib.import_module("flash_rt.flash_rt_gemm_plan")
        except ImportError as exc:
            raise ImportError(
                "Cached FastWAM GEMM extension is unavailable; in the allocated environment run "
                "python scripts/build_gemm_plan.py --output-dir <task-source>/flash_rt "
                "--cuda-home <CUDA-12.8-root>. No legacy fallback was selected."
            ) from exc
        factory = getattr(module, "CachedGemmRunner", None)
        if factory is None:
            raise ImportError("Rebuild CachedGemmRunner with scripts/build_gemm_plan.py")
    else:
        module = importlib.import_module("flash_rt.flash_rt_kernels")
        factory = module.GemmRunner
    with torch.cuda.device(device):
        runner = factory()
    if backend == "cached" and not callable(getattr(runner, "stats", None)):
        raise RuntimeError("CachedGemmRunner.stats is required; rebuild with scripts/build_gemm_plan.py")
    return runner


def install_native_linears(model, *, runner=None, backend="legacy"):
    """Atomically install only active first-frame/action-path Linear modules.

    Both aliases in MoT.mixtures must point at the same expert instances.
    Returns module references for truthful installed/executed coverage. The
    frontend serializes inference because one runner owns shared GEMM workspace.
    ``runner`` is an internal dependency-injection seam for contract tests.
    """
    if backend not in ("legacy", "cached"):
        raise ValueError("native_linear_backend must be 'legacy' or 'cached'")
    candidates, skipped = [], []
    for expert_name, mixture_name in (("video_expert", "video"), ("action_expert", "action")):
        expert = getattr(model, expert_name)
        if model.mot.mixtures[mixture_name] is not expert:
            raise ValueError("FastWAM expert/MoT aliases differ; native replacement would miss the real graph")
        for name, child in expert.named_modules():
            if not _active_linear_path(expert_name, name):
                continue
            path = f"{expert_name}.{name}"
            if isinstance(child, FlashRTLinear):
                raise ValueError("Native FastWAM linears are already installed on this model")
            if not isinstance(child, nn.Linear):
                continue
            if type(child) is not nn.Linear or any((
                child._forward_hooks, child._forward_pre_hooks, child._backward_hooks,
            )):
                skipped.append(path)
                continue
            candidates.append((expert, name, child, path))
    if not candidates:
        raise ValueError("No supported FastWAM hot-path Linear modules found")
    if runner is None:
        runner = create_native_runner(backend, candidates[0][2].weight.device)
    if backend == "cached" and not callable(getattr(runner, "stats", None)):
        raise ValueError("Cached native runner requires CPU stats() counters")
    replacements = [
        (expert, name, FlashRTLinear(child, runner), path)
        for expert, name, child, path in candidates
    ]
    installed = {}
    for expert, name, replacement, path in replacements:
        parent_name, _, attribute = name.rpartition(".")
        parent = expert.get_submodule(parent_name) if parent_name else expert
        setattr(parent, attribute, replacement)
        installed[path] = replacement
    return installed, tuple(skipped)
