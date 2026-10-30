"""Training-free ProbeFlow routing and a CPU numerical reference.

Equations: https://arxiv.org/html/2603.17850v3, Algorithm 1.
Pi0.5 uses model time t=1-s and progress velocity u=-v_theta(x,t).
This module does not import Torch, load weights, or initialize CUDA.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class ProbeFlowConfig:
    epsilon: float = 0.008
    probe_horizon: float = 0.5
    min_steps: int = 2
    max_steps: int = 20
    step_increment: int = 2

    def __post_init__(self):
        for name in ('epsilon', 'probe_horizon'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite real number')
        if self.epsilon <= 0:
            raise ValueError('epsilon must be positive')
        if not 0 < self.probe_horizon < 1:
            raise ValueError('probe_horizon must lie strictly between 0 and 1')
        for name in ('min_steps', 'max_steps', 'step_increment'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f'{name} must be an integer')
        if self.min_steps != 2:
            raise ValueError('ProbeFlow requires min_steps=2 to reuse both probes')
        if not 2 <= self.max_steps <= 64:
            raise ValueError('max_steps must be in [2, 64]')
        if self.step_increment < 1:
            raise ValueError('step_increment must be positive')

    def schedules(self) -> tuple[int, ...]:
        """All reachable schedules, including a clipped non-aligned maximum."""
        return tuple(sorted(set(range(2, self.max_steps + 1, self.step_increment))
                            | {self.max_steps}))

    def timesteps(self) -> tuple[float, ...]:
        """Exact runtime float keys; never round a positive probe time to zero."""
        times = {1., 1. - self.probe_horizon}
        for count in self.schedules():
            if count > 2:
                times.update(1. - i / count for i in range(1, count))
        return tuple(sorted(times, reverse=True))


def select_schedule(cosine: float, config: ProbeFlowConfig) -> int:
    """Paper Eq. 5, clamped before division to avoid tiny-epsilon overflow."""
    if not math.isfinite(cosine):
        raise ValueError('ProbeFlow cosine must be finite')
    curvature = 1. - min(1., max(-1., float(cosine)))
    if curvature == 0 or config.max_steps == 2:
        return 2
    levels = math.ceil((config.max_steps - 2) / config.step_increment)
    if curvature >= config.epsilon * levels:
        return config.max_steps
    return min(config.max_steps,
               2 + math.floor(curvature / config.epsilon) * config.step_increment)


def sample_probeflow(noise, velocity, config: ProbeFlowConfig | None = None):
    """Reference integration; ``velocity(x, t)`` returns model-time velocity.

    The callback is never invoked at the decoded endpoint t=0. Dense branches
    reuse the first velocity, discard the lookahead velocity, and restart from
    the original noise. Input arrays are not mutated.
    """
    config = config or ProbeFlowConfig()
    initial = np.array(noise, dtype=np.float64, copy=True)
    if initial.size == 0 or initial.ndim == 0 or not np.isfinite(initial).all():
        raise ValueError('ProbeFlow noise must be a nonempty finite array')
    times = []

    def evaluate(x, progress):
        t = 1. - progress
        result = np.asarray(velocity(x.copy(), t), dtype=np.float64)
        if result.shape != initial.shape or not np.isfinite(result).all():
            raise ValueError('ProbeFlow velocity must be finite and match noise shape')
        times.append(t)
        return -result

    first = evaluate(initial, 0.)
    probe = initial + first * config.probe_horizon
    lookahead = evaluate(probe, config.probe_horizon)
    flat_first, flat_probe = first.ravel(), lookahead.ravel()
    norm_first, norm_probe = np.linalg.norm(flat_first), np.linalg.norm(flat_probe)
    cosine = (float(np.dot(flat_first / norm_first, flat_probe / norm_probe))
              if norm_first > 0 and norm_probe > 0 else 0.)
    count = select_schedule(cosine, config)
    if count == 2:
        actions = probe + lookahead * (1. - config.probe_horizon)
    else:
        actions = initial + first / count
        for index in range(1, count):
            actions = actions + evaluate(actions, index / count) / count
    if not np.isfinite(actions).all():
        raise ValueError('ProbeFlow integration produced nonfinite actions')
    return actions, {'sampler': 'probeflow', 'scheduled_steps': count,
                     'nfe': len(times), 'cosine': cosine, 'timesteps': times,
                     'probe_horizon': config.probe_horizon}
