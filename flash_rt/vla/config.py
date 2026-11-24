"""CPU-only descriptions of an execution backend and its sampling policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from os import fspath
from types import MappingProxyType
from typing import Any, Mapping

from flash_rt.sampling import ProbeFlowConfig
from .model_spec import get_spec


@dataclass(frozen=True)
class BackendConfig:
    model: str
    checkpoint: str
    backend: str = 'flashrt'
    sampler: str | None = None
    hardware: str = 'auto'
    framework: str = 'torch'
    precision: str = 'auto'
    num_steps: int | None = None
    probeflow: ProbeFlowConfig | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    spec_dir: str | None = None

    def __post_init__(self):
        if self.spec_dir is not None:
            directory = fspath(self.spec_dir)
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError('spec_dir must be a nonempty local directory')
            object.__setattr__(self, 'spec_dir', directory)
        spec = get_spec(self.model, spec_dir=self.spec_dir)
        model = spec['model']
        object.__setattr__(self, 'model', model)
        try:
            checkpoint = fspath(self.checkpoint)
        except TypeError:
            raise ValueError('checkpoint must be a nonempty local path') from None
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError('checkpoint must be a nonempty local path')
        object.__setattr__(self, 'checkpoint', checkpoint)
        caps = spec['capabilities']
        if self.backend not in caps['backends'] or self.framework not in caps['backends'][self.backend]:
            raise ValueError(f'{model} backend={self.backend!r} has no {self.framework!r} frontend; '
                             f'declared backends: {caps["backends"]}')
        if self.sampler is None:
            object.__setattr__(self, 'sampler', spec['head']['defaults']['sampler'])
        if self.sampler not in caps['samplers']:
            raise ValueError(f'{model} sampler must be one of {caps["samplers"]}')
        if self.framework not in caps['frameworks']:
            raise ValueError(f'{model} framework must be one of {caps["frameworks"]}')
        precisions = {'auto', 'fp16', 'bf16', caps['precision'], *caps.get('precisions', [])}
        if self.precision not in precisions:
            raise ValueError(f'precision must be one of {sorted(precisions)}')
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
        if spec['head']['type'] not in ('flow_matching', 'diffusion') and (
                self.probeflow is not None or self.num_steps is not None):
            raise ValueError(f'{spec["head"]["type"]} head does not accept denoising steps or probeflow')
        if not isinstance(self.options, Mapping):
            raise TypeError('options must be a mapping of frontend-specific options')
        object.__setattr__(self, 'options', MappingProxyType(dict(self.options)))
