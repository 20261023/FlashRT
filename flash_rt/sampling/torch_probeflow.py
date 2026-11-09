"""Torch vector operations shared by native flow-matching pipelines.

Only small integration vectors use Torch. Neural velocity evaluations are
supplied by the FlashRT pipeline and keep using the existing native kernels.
"""
import math

import torch
import torch.nn.functional as F

from flash_rt.sampling.probeflow import ProbeFlowConfig, select_schedule


def native_float_view(buffer, shape, dtype):
    """Zero-copy FP16/BF16 view retaining the native allocation's owner."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('native_float_view requires a two-byte floating dtype')
    owner = type('_ProbeFlowCudaArray', (), {})()
    owner.buffer = buffer
    owner.__cuda_array_interface__ = {
        'data': (buffer.ptr.value, False), 'shape': tuple(shape),
        'typestr': '<u2', 'version': 3,
    }
    tensor = torch.as_tensor(owner, device='cuda').view(dtype)
    return tensor, owner


def native_bf16_view(buffer, shape):
    return native_float_view(buffer, shape, torch.bfloat16)


@torch.inference_mode()
def sample_probeflow_torch_(state, evaluate, config: ProbeFlowConfig, *, time_map=None):
    """Integrate in place; ``evaluate(model_time)`` returns progress velocity -v.

    Routing executes anew for each invocation, outside captured neural graphs.
    The one scalar host decision and vector-kernel costs are not hidden from
    policy timing. The evaluator must not integrate or mutate ``state``.

    With ``time_map``, callback times remain the enumerable clock keys, while
    integration uses time_map(t_current)-time_map(t_next). The callback must
    evaluate raw -v at the corresponding physical sigma. This is a shifted-grid
    adaptation, not a claim that the original paper evaluated shifted grids.
    """
    if state.numel() == 0 or not state.is_floating_point() or not torch.isfinite(state).all().item():
        raise ValueError('ProbeFlow state must be a nonempty finite floating tensor')
    initial = state.clone()
    times = []

    def interval(start, end):
        high, low = float(time_map(start)), float(time_map(end))
        if (not math.isfinite(high) or not math.isfinite(low)
                or not 0 <= low < high <= 1):
            raise ValueError('time_map must increase strictly within [0, 1]')
        return high - low

    if time_map is not None and (time_map(0.0) != 0.0 or time_map(1.0) != 1.0):
        raise ValueError('time_map endpoints must be 0 and 1')

    def velocity(t):
        result = evaluate(t)
        if (result.shape != state.shape or result.device != state.device
                or result.dtype != state.dtype):
            raise ValueError('ProbeFlow velocity must match state shape, device and dtype')
        times.append(t)
        return result

    first = velocity(1.).clone()
    probe_weight = (config.probe_horizon if time_map is None
                    else interval(1.0, 1.0 - config.probe_horizon))
    state.copy_(initial).add_(first, alpha=probe_weight)
    lookahead = velocity(1. - config.probe_horizon)
    a, b = first.float().flatten(), lookahead.float().flatten()
    # Scaling before the norm avoids FP32 overflow for large finite bf16 fields.
    a = a / a.abs().amax().clamp_min(1e-30)
    b = b / b.abs().amax().clamp_min(1e-30)
    cosine = F.cosine_similarity(a, b, dim=0).item()
    count = select_schedule(cosine, config)
    if count == 2:
        final_weight = (1. - config.probe_horizon if time_map is None
                        else interval(1.0 - config.probe_horizon, 0.0))
        state.add_(lookahead, alpha=final_weight)
    else:
        first_weight = (1. / count if time_map is None
                        else interval(1.0, 1.0 - 1.0 / count))
        state.copy_(initial).add_(first, alpha=first_weight)
        for index in range(1, count):
            clock = 1. - index / count
            weight = (1. / count if time_map is None
                      else interval(clock, 1.0 - (index + 1) / count))
            state.add_(velocity(clock), alpha=weight)
    if not torch.isfinite(state).all().item():
        raise ValueError('ProbeFlow integration produced nonfinite actions')
    metadata = {'sampler': 'probeflow', 'scheduled_steps': count, 'nfe': len(times),
                'cosine': cosine, 'timesteps': times, 'probe_horizon': config.probe_horizon}
    if time_map is not None:
        metadata['integration_times'] = [float(time_map(t)) for t in times]
        metadata['time_grid'] = 'mapped'
    return metadata
