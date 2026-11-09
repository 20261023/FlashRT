"""CPU-only descriptions of an execution backend and its sampling policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from os import fspath
from types import MappingProxyType
from typing import Any, Mapping

from flash_rt.sampling import ProbeFlowConfig


_ALIASES = {
    'pi0': 'pi0', 'π0': 'pi0',
    'pi05': 'pi05', 'pi0.5': 'pi05', 'π0.5': 'pi05',
    'fastwam': 'fastwam',
    'hyvla': 'hyvla', 'hy-embodied-vla': 'hyvla',
    'hy-embodied-0.5-vla': 'hyvla', 'hy-embadding-vla': 'hyvla',
}


@dataclass(frozen=True)
class BackendConfig:
    model: str
    checkpoint: str
    backend: str = 'flashrt'
    sampler: str = 'euler'
    hardware: str = 'auto'
    framework: str = 'torch'
    precision: str = 'auto'
    num_steps: int | None = None
    probeflow: ProbeFlowConfig | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            model = _ALIASES[self.model.lower()]
        except (KeyError, AttributeError):
            raise ValueError(f'Unknown VLA model {self.model!r}; use pi0, pi05, fastwam or hyvla') from None
        object.__setattr__(self, 'model', model)
        try:
            checkpoint = fspath(self.checkpoint)
        except TypeError:
            raise ValueError('checkpoint must be a nonempty local path') from None
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError('checkpoint must be a nonempty local path')
        object.__setattr__(self, 'checkpoint', checkpoint)
        if self.backend != 'flashrt' and not (
                self.backend == 'torch' and model == 'fastwam' and self.framework == 'torch'):
            raise ValueError('backend="torch" is supported only for Torch FastWAM; otherwise use backend="flashrt"')
        if self.sampler not in ('euler', 'probeflow'):
            raise ValueError('sampler must be euler or probeflow')
        if self.framework not in ('torch', 'jax'):
            raise ValueError('framework must be torch or jax')
        if self.precision not in ('auto', 'fp16', 'bf16'):
            raise ValueError('This backend supports auto/fp16/bf16; quantized tuning is not implemented')
        if self.num_steps is not None and (
                isinstance(self.num_steps, bool) or not isinstance(self.num_steps, Integral)
                or self.num_steps < 1):
            raise ValueError('num_steps must be a positive integer')
        if self.probeflow is not None and not isinstance(self.probeflow, ProbeFlowConfig):
            raise TypeError('probeflow must be a ProbeFlowConfig')
        if self.sampler == 'euler' and self.probeflow is not None:
            raise ValueError('probeflow requires sampler="probeflow"')
        if self.sampler == 'probeflow' and self.num_steps is not None:
            raise ValueError('Use probeflow.max_steps, not num_steps, for adaptive sampling')
        if self.sampler == 'probeflow' and self.probeflow is None:
            object.__setattr__(self, 'probeflow', ProbeFlowConfig())
        if not isinstance(self.options, Mapping):
            raise TypeError('options must be a mapping of frontend-specific options')
        object.__setattr__(self, 'options', MappingProxyType(dict(self.options)))
