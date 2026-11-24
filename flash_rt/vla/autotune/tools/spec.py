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

from flash_rt.vla.model_spec import get_spec, list_models
from flash_rt.vla.registry import _CUDA_TARGETS

from ..stages import STAGE_IDS


LANES = ('config', 'port', 'patch', 'analysis', 'baseline')
EXACTNESS = ('exact', 'approximate', 'n/a')
STATUSES = ('ready', 'needs_setup', 'needs_implementation', 'blocked')
MODELS = list_models()
ANY_MODEL = '*'
# Hardware label -> CUDA capability; the inverse of registry._CUDA_TARGETS.
HARDWARE_CAPABILITY = MappingProxyType({label: capability for capability, label in _CUDA_TARGETS.items()})
HARDWARE_LABELS = tuple(HARDWARE_CAPABILITY) + ('auto',)

_ID = re.compile(r'[a-z0-9][a-z0-9_.-]*')
_SKILL = re.compile(r'[a-z0-9][a-z0-9-]*')
_COMMIT = re.compile(r'[0-9a-f]{40}')
_MISSING = object()


def _value_at(record, path):
    value = record
    for name in path.split('.'):
        if not isinstance(value, Mapping) or name not in value:
            return _MISSING
        value = value[name]
    return value


def validate_predicate(predicate):
    """Validate a JSON-only feature predicate; arbitrary executable selectors are forbidden."""
    if not isinstance(predicate, Mapping):
        raise ValueError('model_predicate must be a mapping')
    if not predicate:
        return
    groups = set(predicate) & {'all', 'any', 'not'}
    if groups:
        if len(predicate) != 1:
            raise ValueError('model_predicate boolean groups must have exactly one key')
        name = next(iter(groups))
        if name == 'not':
            validate_predicate(predicate[name])
        else:
            children = predicate[name]
            if not isinstance(children, (list, tuple)) or not children:
                raise ValueError(f'model_predicate {name} needs a nonempty list')
            for child in children:
                validate_predicate(child)
        return
    operators = set(predicate) - {'path'}
    if (not isinstance(predicate.get('path'), str) or not predicate['path']
            or any(not part for part in predicate['path'].split('.'))
            or len(operators) != 1 or not operators <= {'equals', 'contains', 'exists', 'in'}):
        raise ValueError('model_predicate leaf needs path and one of equals/contains/exists/in')
    if 'exists' in predicate and type(predicate['exists']) is not bool:
        raise ValueError('model_predicate exists must be a bool')
    if 'in' in predicate and not isinstance(predicate['in'], (list, tuple)):
        raise ValueError('model_predicate in must be a list')
    _json_data(predicate, 'model_predicate')


def match_predicate(predicate, record) -> tuple:
    """Return (matches, explanation), treating explicit unknown values as unverified."""
    if not predicate:
        return True, ''
    if 'all' in predicate:
        results = [match_predicate(child, record) for child in predicate['all']]
        failures = [reason for matches, reason in results if not matches]
        return not failures, '; '.join(failures)
    if 'any' in predicate:
        results = [match_predicate(child, record) for child in predicate['any']]
        if any(matches for matches, _ in results):
            return True, ''
        return False, ' or '.join(reason for _, reason in results)
    if 'not' in predicate:
        matches, _ = match_predicate(predicate['not'], record)
        return not matches, 'excluded by model feature predicate' if matches else ''
    path = predicate['path']
    value = _value_at(record, path)
    known = value is not _MISSING and not (isinstance(value, Mapping)
                                          and value.get('status') == 'unknown')
    if 'exists' in predicate:
        matches = known == predicate['exists']
        expected = f'{path} {"present and known" if predicate["exists"] else "absent or unknown"}'
    elif not known:
        return False, f'{path} is absent or unknown in ModelSpec'
    elif 'equals' in predicate:
        expected_value = predicate['equals']
        matches = value == expected_value
        if type(expected_value) is bool:
            matches = matches and type(value) is bool
        expected = f'{path}={predicate["equals"]!r}'
    elif 'contains' in predicate:
        item = predicate['contains']
        matches = ((isinstance(value, Mapping) and isinstance(item, str) and item in value)
                   or (isinstance(value, (list, tuple)) and item in value))
        expected = f'{path} contains {predicate["contains"]!r}'
    else:
        matches = value in predicate['in']
        expected = f'{path} in {predicate["in"]!r}'
    return bool(matches), '' if matches else f'requires {expected}'


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

    ``models`` is a legacy model restriction or ``('*',)``. New tools declare
    ``model_predicate`` over ModelSpec features, head and capabilities; importing
    a new spec automatically extends matching without editing the tool registry.
    ``model_status``
    optionally overrides ``(status, reason)`` for single models (a tool can be
    ready for one model and unimplemented for the rest); ``status_for(model)``
    applies it. ``propose(baseline, **params) -> list[Candidate]`` exists only
    for deterministic config/analysis tools. ``adapter`` is the module that
    holds the tool's code or procedure. ``requirements_path`` reads a declared
    model recipe for configuration constraints and stage scope; ``aliases``
    preserve historical ids without duplicating canonical registry entries.
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
    hardware_only: tuple = ()
    model_predicate: Mapping[str, Any] = field(default_factory=dict)
    frameworks: tuple = ()
    samplers: tuple = ()
    status_rules: tuple = ()
    aliases: tuple = ()
    requirements_path: str | None = None

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
        validate_predicate(self.model_predicate)
        object.__setattr__(self, 'model_predicate',
                           MappingProxyType(_json_data(self.model_predicate, 'model_predicate')))
        for name in ('frameworks', 'samplers'):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f'{self.tool_id}: {name} must contain distinct nonempty strings')
            object.__setattr__(self, name, values)
        aliases = tuple(self.aliases)
        if (len(set(aliases)) != len(aliases) or self.tool_id in aliases
                or any(not isinstance(alias, str) or not _ID.fullmatch(alias) for alias in aliases)):
            raise ValueError(f'{self.tool_id}: aliases must be distinct tool ids')
        object.__setattr__(self, 'aliases', aliases)
        if self.requirements_path is not None:
            _nonempty(self.requirements_path, f'{self.tool_id} requirements_path')
        rules = []
        for rule in self.status_rules:
            if not isinstance(rule, Mapping) or set(rule) != {'when', 'status', 'reason'}:
                raise ValueError(f'{self.tool_id}: status_rules need when, status and reason')
            validate_predicate(rule['when'])
            self._check_status(rule['status'], rule['reason'], self.tool_id)
            rules.append(_json_data(rule, 'status_rules'))
        object.__setattr__(self, 'status_rules', tuple(rules))
        if self.exactness not in EXACTNESS:
            raise ValueError(f'{self.tool_id}: exactness must be one of {EXACTNESS}')
        self._check_status(self.status, self.reason, self.tool_id)
        if self.min_capability is not None:
            capability = tuple(self.min_capability)
            if len(capability) != 2 or not all(type(part) is int and part >= 0 for part in capability):
                raise ValueError(f'{self.tool_id}: min_capability must be (major, minor) integers')
            object.__setattr__(self, 'min_capability', capability)
        labels = tuple(self.hardware_only)
        if len(set(labels)) != len(labels) or any(label not in HARDWARE_CAPABILITY for label in labels):
            raise ValueError(f'{self.tool_id}: hardware_only must name concrete CUDA targets')
        object.__setattr__(self, 'hardware_only', labels)
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

    def stages_for(self, model=None, *, spec_dir=None) -> tuple:
        stages = tuple(self._requirements(model, spec_dir=spec_dir).get('stages', self.stages))
        if not stages or len(set(stages)) != len(stages) or not set(stages) <= set(self.stages):
            raise ValueError(f'{self.tool_id}: ModelSpec recipe stages must be distinct supported stage ids')
        return stages

    def primary_stage_for(self, model=None, *, spec_dir=None) -> str:
        return self.stages_for(model, spec_dir=spec_dir)[0]

    def match_model(self, model, *, spec_dir=None) -> tuple:
        spec = get_spec(model, spec_dir=spec_dir)
        unsupported = set(spec.get('tools', {}).get('unsupported', []))
        if {self.tool_id, *self.aliases} & unsupported:
            return False, 'explicitly unsupported by ModelSpec tools.unsupported'
        if self.models != (ANY_MODEL,) and spec['model'] not in self.models:
            return False, f'legacy compatibility restriction: {", ".join(self.models)}'
        if self.frameworks and not set(self.frameworks) & set(spec['capabilities']['frameworks']):
            return False, f'requires frameworks in {list(self.frameworks)}'
        if self.samplers and not set(self.samplers) & set(spec['capabilities']['samplers']):
            return False, f'requires samplers in {list(self.samplers)}'
        return match_predicate(self.model_predicate, spec)

    def applies_to(self, model, *, spec_dir=None) -> bool:
        return self.match_model(model, spec_dir=spec_dir)[0]

    def matching_models(self, *, spec_dir=None) -> tuple:
        return tuple(model for model in list_models(spec_dir=spec_dir)
                     if self.applies_to(model, spec_dir=spec_dir))

    def _status(self, model=None, *, spec_dir=None) -> tuple:
        if model is not None:
            spec = get_spec(model, spec_dir=spec_dir)
            matches, reason = self.match_model(model, spec_dir=spec_dir)
            if not matches:
                return 'blocked', reason
            for rule in self.status_rules:
                if match_predicate(rule['when'], spec)[0]:
                    return rule['status'], rule['reason']
            if spec['model'] in self.model_status:
                return self.model_status[spec['model']]
        return self.status, self.reason

    def status_for(self, model=None, *, spec_dir=None) -> str:
        return self._status(model, spec_dir=spec_dir)[0]

    def reason_for(self, model=None, *, spec_dir=None) -> str:
        return self._status(model, spec_dir=spec_dir)[1]

    def configuration_reason(self, config) -> str:
        """Explain a baseline constraint before generating any candidates."""
        if self.frameworks and config.framework not in self.frameworks:
            return f'requires framework in {list(self.frameworks)}'
        if self.samplers and config.sampler not in self.samplers:
            return f'requires sampler in {list(self.samplers)}'
        requirements = self._requirements(config.model, spec_dir=getattr(config, 'spec_dir', None))
        for field, value in (('frameworks', config.framework), ('samplers', config.sampler),
                             ('hardware', config.hardware)):
            allowed = requirements.get(field, [])
            if allowed and value not in allowed and not (field == 'hardware' and value == 'auto'):
                return f'requires {field} in {allowed}'
        return ''

    def _requirements(self, model=None, *, spec_dir=None):
        if model is None or self.requirements_path is None:
            return {}
        value = _value_at(get_spec(model, spec_dir=spec_dir), self.requirements_path)
        return value if isinstance(value, Mapping) else {}

    def available_on(self, hardware, model=None, *, spec_dir=None) -> bool:
        """Match the compiled targets and minimum device capability."""
        return not self.hardware_reason(hardware, model, spec_dir=spec_dir)

    def hardware_reason(self, hardware, model=None, *, spec_dir=None) -> str:
        capability = hardware_capability(hardware)
        if self.hardware_only and hardware not in self.hardware_only:
            return f'requires hardware in {list(self.hardware_only)}; got {hardware}'
        allowed = self._requirements(model, spec_dir=spec_dir).get('hardware', [])
        if allowed and hardware not in allowed:
            return f'ModelSpec recipe requires hardware in {allowed}; got {hardware}'
        if self.min_capability is not None and (capability is None or capability < self.min_capability):
            return f'requires CUDA capability >= {self.min_capability}; got {hardware}'
        return ''

    def to_dict(self, *, model=None, spec_dir=None) -> dict:
        return {
            'tool_id': self.tool_id, 'title': self.title, 'lane': self.lane,
            'stages': list(self.stages_for(model, spec_dir=spec_dir)),
            'models': list(self.models), 'exactness': self.exactness,
            'status': self.status_for(model, spec_dir=spec_dir),
            'reason': self.reason_for(model, spec_dir=spec_dir),
            'model_status': {model: {'status': status, 'reason': reason}
                             for model, (status, reason) in self.model_status.items()},
            'min_capability': None if self.min_capability is None else list(self.min_capability),
            'upstream': None if self.upstream is None else self.upstream.to_dict(),
            'skills': list(self.skills), 'parameters': _json_data(dict(self.parameters), 'parameters'),
            'proposes_candidates': self.propose is not None, 'adapter': self.adapter,
            'model_predicate': _json_data(self.model_predicate, 'model_predicate'),
            'matching_models': list(self.matching_models(spec_dir=spec_dir)),
            'frameworks': list(self.frameworks), 'samplers': list(self.samplers),
            'status_rules': _json_data(self.status_rules, 'status_rules'),
            'aliases': list(self.aliases),
            'requirements_path': self.requirements_path,
            **({'hardware_only': list(self.hardware_only)} if self.hardware_only else {}),
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
        skills=tuple(spec['skills']), adapter=adapter,
        model_predicate=spec.get('model_predicate', {}),
        status_rules=tuple(spec.get('status_rules', ())),
        frameworks=tuple(spec.get('frameworks', ())),
        samplers=tuple(spec.get('samplers', ())),
        min_capability=spec.get('min_capability'),
        hardware_only=tuple(spec.get('hardware_only', ())),
        aliases=tuple(spec.get('aliases', ())),
        requirements_path=spec.get('requirements_path'))
