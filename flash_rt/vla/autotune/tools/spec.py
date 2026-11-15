"""Tool descriptors: what a tool changes, where, for which models, and whether it runs today.

A ``ToolSpec`` is data plus an optional deterministic ``propose`` callable. It
never runs a model or an external project. Capability filtering uses the same
hardware labels as ``flash_rt.vla.registry`` (``_CUDA_TARGETS``) without
importing Torch; ``'auto'`` means the device is unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import re
from types import MappingProxyType
from typing import Any, Callable

from flash_rt.vla.registry import _CAPABILITIES, _CUDA_TARGETS

from ..stages import STAGE_IDS


LANES = ('config', 'port', 'patch', 'analysis', 'baseline')
EXACTNESS = ('exact', 'approximate', 'n/a')
STATUSES = ('ready', 'needs_setup', 'needs_implementation', 'blocked')
MODELS = tuple(_CAPABILITIES)
ANY_MODEL = '*'
# Hardware label -> CUDA capability; the inverse of registry._CUDA_TARGETS.
HARDWARE_CAPABILITY = MappingProxyType({label: capability for capability, label in _CUDA_TARGETS.items()})
HARDWARE_LABELS = tuple(HARDWARE_CAPABILITY) + ('auto',)

_ID = re.compile(r'[a-z0-9][a-z0-9_.-]*')
_SKILL = re.compile(r'[a-z0-9][a-z0-9-]*')
_COMMIT = re.compile(r'[0-9a-f]{40}')


def hardware_capability(label):
    """(major, minor) for a registry hardware label; None for 'auto' (unknown device)."""
    if label == 'auto':
        return None
    if label not in HARDWARE_CAPABILITY:
        raise ValueError(f'Unknown hardware label {label!r}; use one of {list(HARDWARE_LABELS)}')
    return HARDWARE_CAPABILITY[label]


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _json_data(value, name):
    """Detached JSON-compatible copy (tuples become lists)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'{name} holds a nonfinite number')
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f'{name} keys must be strings')
        return {key: _json_data(item, name) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(item, name) for item in value]
    raise ValueError(f'{name} holds a non-JSON value: {type(value).__name__}')


@dataclass(frozen=True)
class Upstream:
    """Pinned external project. ``notice``: upstream ships a NOTICE file to carry."""

    repo: str
    commit: str
    license: str
    path: str | None = None
    notice: bool = False

    def __post_init__(self):
        _nonempty(self.repo, 'upstream.repo')
        if not isinstance(self.commit, str) or not _COMMIT.fullmatch(self.commit):
            raise ValueError(f'upstream.commit must be a 40-hex commit, got {self.commit!r}')
        _nonempty(self.license, 'upstream.license')
        if self.path is not None:
            _nonempty(self.path, 'upstream.path')
        if not isinstance(self.notice, bool):
            raise ValueError('upstream.notice must be a bool')

    @classmethod
    def from_spec(cls, record) -> Upstream:
        """From an external adapter SPEC['upstream'] dict ({repo, commit, license, notice})."""
        if not isinstance(record, Mapping):
            raise ValueError('upstream record must be a dict')
        unknown = set(record) - {'repo', 'commit', 'license', 'path', 'notice'}
        if unknown:
            raise ValueError(f'Unknown upstream fields: {sorted(unknown)}')
        return cls(**dict(record))

    def to_dict(self) -> dict:
        return {'repo': self.repo, 'commit': self.commit, 'license': self.license,
                'path': self.path, 'notice': self.notice}


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool.

    ``models`` is a tuple of registry model ids or ``('*',)``. ``model_status``
    optionally overrides ``(status, reason)`` for single models (a tool can be
    ready for one model and unimplemented for the rest); ``status_for(model)``
    applies it. ``propose(baseline, **params) -> list[Candidate]`` exists only
    for deterministic config/analysis tools. ``adapter`` is the module that
    holds the tool's code or procedure.
    """

    tool_id: str
    title: str
    lane: str
    stages: tuple
    models: tuple
    exactness: str
    status: str
    reason: str = ''
    min_capability: tuple | None = None
    upstream: Upstream | None = None
    skills: tuple = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    propose: Callable | None = None
    adapter: str | None = None
    model_status: Mapping[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.tool_id, str) or not _ID.fullmatch(self.tool_id):
            raise ValueError(f'tool_id must match ^[a-z0-9][a-z0-9_.-]*$, got {self.tool_id!r}')
        _nonempty(self.title, f'{self.tool_id} title')
        if self.lane not in LANES:
            raise ValueError(f'{self.tool_id}: lane must be one of {LANES}')
        stages = tuple(self.stages)
        if not stages or any(stage not in STAGE_IDS for stage in stages) or len(set(stages)) != len(stages):
            raise ValueError(f'{self.tool_id}: stages must be distinct known stage ids, got {stages}')
        object.__setattr__(self, 'stages', stages)
        models = tuple(self.models)
        if models != (ANY_MODEL,) and (not models or any(model not in MODELS for model in models)
                                       or len(set(models)) != len(models)):
            raise ValueError(f'{self.tool_id}: models must be distinct ids from {MODELS} or ("*",)')
        object.__setattr__(self, 'models', models)
        if self.exactness not in EXACTNESS:
            raise ValueError(f'{self.tool_id}: exactness must be one of {EXACTNESS}')
        self._check_status(self.status, self.reason, self.tool_id)
        if self.min_capability is not None:
            capability = tuple(self.min_capability)
            if len(capability) != 2 or not all(type(part) is int and part >= 0 for part in capability):
                raise ValueError(f'{self.tool_id}: min_capability must be (major, minor) integers')
            object.__setattr__(self, 'min_capability', capability)
        if self.upstream is not None and not isinstance(self.upstream, Upstream):
            raise TypeError(f'{self.tool_id}: upstream must be an Upstream')
        skills = tuple(self.skills)
        if not skills or any(not isinstance(skill, str) or not _SKILL.fullmatch(skill) for skill in skills):
            raise ValueError(f'{self.tool_id}: skills must be a nonempty tuple of lowercase skill names')
        object.__setattr__(self, 'skills', skills)
        if not isinstance(self.parameters, Mapping):
            raise ValueError(f'{self.tool_id}: parameters must be a mapping of defaults')
        _json_data(dict(self.parameters), f'{self.tool_id} parameters')
        object.__setattr__(self, 'parameters', MappingProxyType(dict(self.parameters)))
        if self.propose is not None:
            if not callable(self.propose):
                raise TypeError(f'{self.tool_id}: propose must be callable')
            if self.lane not in ('config', 'analysis'):
                raise ValueError(f'{self.tool_id}: only config/analysis tools may propose candidates')
        if self.adapter is not None:
            _nonempty(self.adapter, f'{self.tool_id} adapter')
        overrides = {}
        for model, value in dict(self.model_status).items():
            if not self.applies_to(model):
                raise ValueError(f'{self.tool_id}: model_status for {model!r} outside its models')
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(f'{self.tool_id}: model_status values must be (status, reason)')
            self._check_status(value[0], value[1], f'{self.tool_id}/{model}')
            overrides[model] = value
        object.__setattr__(self, 'model_status', MappingProxyType(overrides))

    @staticmethod
    def _check_status(status, reason, name):
        if status not in STATUSES:
            raise ValueError(f'{name}: status must be one of {STATUSES}')
        if not isinstance(reason, str):
            raise ValueError(f'{name}: reason must be a string')
        if status != 'ready' and not reason.strip():
            raise ValueError(f'{name}: a {status} tool needs a reason')

    @property
    def primary_stage(self) -> str:
        return self.stages[0]

    def applies_to(self, model) -> bool:
        return self.models == (ANY_MODEL,) or model in self.models

    def status_for(self, model=None) -> str:
        return self.model_status.get(model, (self.status, self.reason))[0]

    def reason_for(self, model=None) -> str:
        return self.model_status.get(model, (self.status, self.reason))[1]

    def available_on(self, hardware) -> bool:
        """False when the tool needs a capability the hardware lacks or cannot be proven ('auto')."""
        if self.min_capability is None:
            return True
        capability = hardware_capability(hardware)
        return capability is not None and capability >= self.min_capability

    def to_dict(self) -> dict:
        return {
            'tool_id': self.tool_id, 'title': self.title, 'lane': self.lane,
            'stages': list(self.stages), 'models': list(self.models), 'exactness': self.exactness,
            'status': self.status, 'reason': self.reason,
            'model_status': {model: {'status': status, 'reason': reason}
                             for model, (status, reason) in self.model_status.items()},
            'min_capability': None if self.min_capability is None else list(self.min_capability),
            'upstream': None if self.upstream is None else self.upstream.to_dict(),
            'skills': list(self.skills), 'parameters': _json_data(dict(self.parameters), 'parameters'),
            'proposes_candidates': self.propose is not None, 'adapter': self.adapter,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


def spec_from_external(spec, adapter) -> ToolSpec:
    """Build a ToolSpec from an external adapter's SPEC dict (facts are not duplicated).

    The adapter's ``setup``/``risks`` lists stay in its SPEC (``show`` prints them).
    """
    if not isinstance(spec, Mapping):
        raise ValueError('external SPEC must be a dict')
    return ToolSpec(
        tool_id=spec['tool_id'], title=spec['title'], lane=spec['lane'], stages=tuple(spec['stages']),
        models=tuple(spec['models']), exactness=spec['exactness'], status=spec['status'],
        reason=spec['reason'], upstream=Upstream.from_spec(spec['upstream']),
        skills=tuple(spec['skills']), adapter=adapter)
