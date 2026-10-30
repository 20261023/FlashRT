"""Torch vector operations for the native Pi0.5 ProbeFlow pipeline.

Only small integration vectors use Torch. Neural velocity evaluations are
supplied by the FlashRT pipeline and keep using the existing native kernels.
"""
import torch
import torch.nn.functional as F

from flash_rt.models.pi05.probeflow import ProbeFlowConfig, select_schedule


def native_bf16_view(buffer, shape):
    """Zero-copy bf16 view; the CUDA-array owner retains the native allocation."""
    owner = type('_ProbeFlowCudaArray', (), {})()
    owner.buffer = buffer
    owner.__cuda_array_interface__ = {
        'data': (buffer.ptr.value, False), 'shape': tuple(shape),
        'typestr': '<u2', 'version': 3,
    }
    tensor = torch.as_tensor(owner, device='cuda').view(torch.bfloat16)
    return tensor, owner


@torch.inference_mode()
def sample_probeflow_torch_(state, evaluate, config: ProbeFlowConfig):
    """Integrate in place; ``evaluate(model_time)`` returns progress velocity -v.

    Routing executes anew for each invocation, outside captured neural graphs.
    The one scalar host decision and vector-kernel costs are not hidden from
    policy timing. The evaluator must not integrate or mutate ``state``.
    """
    if state.numel() == 0 or not state.is_floating_point() or not torch.isfinite(state).all().item():
        raise ValueError('ProbeFlow state must be a nonempty finite floating tensor')
    initial = state.clone()
    times = []

    def velocity(t):
        result = evaluate(t)
        if (result.shape != state.shape or result.device != state.device
                or result.dtype != state.dtype):
            raise ValueError('ProbeFlow velocity must match state shape, device and dtype')
        times.append(t)
        return result

    first = velocity(1.).clone()
    state.copy_(initial).add_(first, alpha=config.probe_horizon)
    lookahead = velocity(1. - config.probe_horizon)
    a, b = first.float().flatten(), lookahead.float().flatten()
    # Scaling before the norm avoids FP32 overflow for large finite bf16 fields.
    a = a / a.abs().amax().clamp_min(1e-30)
    b = b / b.abs().amax().clamp_min(1e-30)
    cosine = F.cosine_similarity(a, b, dim=0).item()
    count = select_schedule(cosine, config)
    if count == 2:
        state.add_(lookahead, alpha=1. - config.probe_horizon)
    else:
        state.copy_(initial).add_(first, alpha=1. / count)
        for index in range(1, count):
            state.add_(velocity(1. - index / count), alpha=1. / count)
    if not torch.isfinite(state).all().item():
        raise ValueError('ProbeFlow integration produced nonfinite actions')
    return {'sampler': 'probeflow', 'scheduled_steps': count, 'nfe': len(times),
            'cosine': cosine, 'timesteps': times, 'probe_horizon': config.probe_horizon}
