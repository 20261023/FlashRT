"""Explicit JSON codecs for immutable candidates and complete worker results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from flash_rt.sampling import ProbeFlowConfig
from flash_rt.vla.config import BackendConfig


_BACKEND_FIELDS = {
    'model', 'checkpoint', 'backend', 'sampler', 'hardware', 'framework',
    'precision', 'num_steps', 'probeflow', 'options',
}
_PROBEFLOW_FIELDS = {
    'epsilon', 'probe_horizon', 'min_steps', 'max_steps', 'step_increment',
}
_CANDIDATE_FIELDS = {
    'candidate_id', 'backend', 'workload_id', 'parent_id', 'hypothesis',
    'origin', 'patch_ref', 'source_ref',
}


def _json_value(value):
    """Copy JSON data without coercing keys, nonfinite numbers or opaque objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError('JSON object keys must be strings')
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    raise ValueError(f'Not a finite JSON value: {type(value).__name__}')


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a dict')
    return value


def _known_fields(record, allowed, name):
    _object(record, name)
    unknown = record.keys() - allowed
    if unknown:
        raise ValueError(f'Unknown {name} fields: {sorted(map(str, unknown))}')


def _text(value, name, *, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f'{name} must be a {"nonempty " if nonempty else ""}string')


def encode_backend(config: BackendConfig) -> dict:
    """Encode constructor fields, never the runtime-only describe_backend view.

    Frontend options must themselves be JSON data. Returned nested containers
    are detached from the input config, including its shallow read-only mapping.
    """
    if not isinstance(config, BackendConfig):
        raise TypeError('config must be a BackendConfig')
    policy = config.probeflow
    return _json_value({
        'model': config.model, 'checkpoint': config.checkpoint,
        'backend': config.backend, 'sampler': config.sampler,
        'hardware': config.hardware, 'framework': config.framework,
        'precision': config.precision,
        'num_steps': None if config.num_steps is None else int(config.num_steps),
        'probeflow': None if policy is None else {
            'epsilon': float(policy.epsilon),
            'probe_horizon': float(policy.probe_horizon),
            'min_steps': int(policy.min_steps), 'max_steps': int(policy.max_steps),
            'step_increment': int(policy.step_increment),
        },
        'options': config.options,
    })


def decode_backend(record: dict) -> BackendConfig:
    """Decode strict known fields; omitted optional fields use backend defaults."""
    _known_fields(record, _BACKEND_FIELDS, 'backend')
    record = _json_value(record)
    for name in ('model', 'checkpoint'):
        _text(record.get(name), name, nonempty=True)
    policy = record.get('probeflow')
    if policy is not None:
        _known_fields(policy, _PROBEFLOW_FIELDS, 'probeflow')
        policy = ProbeFlowConfig(**policy)
    fields = dict(record)
    if 'probeflow' in fields:
        fields['probeflow'] = policy
    return BackendConfig(**fields)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    backend: BackendConfig
    workload_id: str
    parent_id: str | None = None
    hypothesis: str = ''
    origin: str = 'human'
    patch_ref: str | None = None
    source_ref: str | None = None

    def __post_init__(self):
        for name in ('candidate_id', 'workload_id'):
            _text(getattr(self, name), name, nonempty=True)
        if not isinstance(self.backend, BackendConfig):
            raise TypeError('backend must be a BackendConfig')
        for name in ('parent_id', 'patch_ref', 'source_ref'):
            value = getattr(self, name)
            if value is not None:
                _text(value, name, nonempty=True)
        _text(self.hypothesis, 'hypothesis')
        if self.origin not in ('human', 'search', 'agent'):
            raise ValueError('origin must be human, search or agent')

    def to_dict(self) -> dict:
        return {
            'candidate_id': self.candidate_id, 'backend': encode_backend(self.backend),
            'workload_id': self.workload_id, 'parent_id': self.parent_id,
            'hypothesis': self.hypothesis, 'origin': self.origin,
            'patch_ref': self.patch_ref, 'source_ref': self.source_ref,
        }

    @classmethod
    def from_dict(cls, record: dict) -> Candidate:
        _known_fields(record, _CANDIDATE_FIELDS, 'candidate')
        for name in ('candidate_id', 'backend', 'workload_id'):
            if name not in record:
                raise ValueError(f'Missing candidate field: {name}')
        return cls(**{**record, 'backend': decode_backend(record['backend'])})


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _nonnegative(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be finite and nonnegative')


def validate_result(record: dict) -> None:
    """Reject partial/fabricated samples; leave all additional metrics untouched.

    Quality is evidence scoped by the caller, not an inferred task-success
    claim. A failed record carries a failure instead of a timing measurement.
    """
    _object(record, 'result')
    if type(record.get('schema_version')) is not int or record['schema_version'] not in (1, 2):
        raise ValueError('schema_version must be 1 (historical) or 2 (bound execution)')
    for name in ('candidate_id', 'attempt_id', 'comparison_id', 'workload_id'):
        _text(record.get(name), name, nonempty=True)
    if record['schema_version'] == 2:
        _text(record.get('campaign_instance_id'), 'campaign_instance_id', nonempty=True)
        candidate = Candidate.from_dict(record.get('candidate'))
        if candidate.candidate_id != record['candidate_id'] or candidate.workload_id != record['workload_id']:
            raise ValueError('Result candidate envelope identifiers do not match')
        if record.get('job_id') is not None:
            _text(record['job_id'], 'job_id', nonempty=True)
    state = record.get('state')
    if state not in ('measured', 'failed'):
        raise ValueError('result state must be measured or failed')
    if state == 'failed':
        failure = _object(record.get('failure'), 'failure')
        _text(failure.get('kind'), 'failure.kind', nonempty=True)
        _text(failure.get('message'), 'failure.message', nonempty=True)
        if record.get('measurement') is not None:
            raise ValueError('failed results must not contain a measurement')
    else:
        measurement = _object(record.get('measurement'), 'measurement')
        _integer(measurement.get('warmup'), 'warmup', 0)
        _integer(measurement.get('repeats'), 'repeats', 1)
        repeats = measurement['repeats']
        latency = measurement.get('latency_ms')
        if not isinstance(latency, list) or len(latency) != repeats:
            raise ValueError('latency_ms must have exactly repeats samples')
        for value in latency:
            _nonnegative(value, 'latency_ms')
        for name in ('load_ms', 'prepare_ms'):
            _nonnegative(measurement.get(name), name)
        samplers = measurement.get('sampler_metadata')
        if not isinstance(samplers, list) or len(samplers) != repeats:
            raise ValueError('sampler_metadata must have exactly repeats entries')
        for sampler in samplers:
            _object(sampler, 'sampler_metadata entry')
            if sampler.get('sampler') not in ('euler', 'probeflow'):
                raise ValueError('sampler must be euler or probeflow')
            _integer(sampler.get('nfe'), 'actual nfe', 1)
        _object(record.get('backend_metadata'), 'backend_metadata')
        _object(record.get('gpu'), 'gpu')
        _text(record.get('outputs_path'), 'outputs_path')
        quality = _object(record.get('quality'), 'quality')
        if quality.get('status') not in ('unknown', 'passed', 'failed'):
            raise ValueError('quality.status must be unknown, passed or failed')
        _text(quality.get('scope'), 'quality.scope')
        _object(quality.get('metrics'), 'quality.metrics')
    _json_value(record)
