"""Explicit JSON codecs for immutable candidates and complete worker results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re

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
    'origin', 'patch_ref', 'source_ref', 'tool', 'stage',
}
_PROVENANCE_ID = re.compile(r'[a-z0-9][a-z0-9_.-]*')
_SOURCE_IDENTITY_FIELDS = (
    'snapshot_id', 'source_dir', 'flash_rt_file', 'base_commit', 'patch_ref', 'head_commit',
    'head_tree', 'head_parent', 'harness_tree', 'tracked_clean', 'unexpected_files',
    'extensions', 'python', 'env',
)


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
    tool: str | None = None
    stage: str | None = None

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
        for name in ('tool', 'stage'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not _PROVENANCE_ID.fullmatch(value)):
                raise ValueError(f'{name} must be None or match ^[a-z0-9][a-z0-9_.-]*$')

    def to_dict(self) -> dict:
        record = {
            'candidate_id': self.candidate_id, 'backend': encode_backend(self.backend),
            'workload_id': self.workload_id, 'parent_id': self.parent_id,
            'hypothesis': self.hypothesis, 'origin': self.origin,
            'patch_ref': self.patch_ref, 'source_ref': self.source_ref,
        }
        # Optional provenance is emitted only when set so that the canonical
        # JSON of every pre-existing candidate (ledger identity) is unchanged.
        for name in ('tool', 'stage'):
            if getattr(self, name) is not None:
                record[name] = getattr(self, name)
        return record

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


def validate_source_identity(identity, candidate: Candidate) -> None:
    """A complete, clean source identity for the candidate's own snapshot."""
    # Imported lazily: ``python -m flash_rt.vla.autotune.snapshots`` must not
    # find its module pre-imported by this package's __init__.
    from .snapshots import check_commit, check_patch_ref, check_snapshot_id, snapshot_id
    _object(identity, 'source_identity')
    missing = [name for name in _SOURCE_IDENTITY_FIELDS if name not in identity]
    if missing:
        raise ValueError(f'source_identity is missing fields: {missing}')
    check_snapshot_id(identity['snapshot_id'], error=ValueError)
    for name in ('source_dir', 'flash_rt_file'):
        _text(identity[name], f'source_identity.{name}', nonempty=True)
    for name in ('base_commit', 'head_commit', 'head_tree', 'harness_tree'):
        check_commit(identity[name], f'source_identity.{name}', error=ValueError)
    if identity['head_parent'] is not None:
        check_commit(identity['head_parent'], 'source_identity.head_parent', error=ValueError)
    check_patch_ref(identity['patch_ref'], error=ValueError)
    if identity['tracked_clean'] is not True or identity['unexpected_files'] != []:
        raise ValueError('source_identity must prove a clean tree without unexpected files')
    extensions = _object(identity['extensions'], 'source_identity.extensions')
    for relpath, digest in extensions.items():
        if not isinstance(digest, str) or len(digest) != 64 or digest.strip('0123456789abcdef'):
            raise ValueError(f'source_identity extension {relpath} must have a 64-hex sha256')
    _object(identity['python'], 'source_identity.python')
    env = _object(identity['env'], 'source_identity.env')
    if not all(isinstance(value, str) for value in env.values()):
        raise ValueError('source_identity.env values must be strings')
    if (identity['base_commit'] != candidate.source_ref or identity['patch_ref'] != candidate.patch_ref
            or identity['snapshot_id'] != snapshot_id(candidate.source_ref, candidate.patch_ref)):
        raise ValueError('source_identity does not describe the candidate source_ref/patch_ref snapshot')


def validate_result(record: dict) -> None:
    """Reject partial/fabricated samples; leave all additional metrics untouched.

    Quality is evidence scoped by the caller, not an inferred task-success
    claim. A failed record carries a failure instead of a timing measurement.
    Schema 3 is schema 2 plus a verified ``source_identity`` (snapshot mode);
    a failed schema 3 record may carry none, e.g. when verification failed.
    """
    _object(record, 'result')
    if type(record.get('schema_version')) is not int or record['schema_version'] not in (1, 2, 3):
        raise ValueError('schema_version must be 1 (historical), 2 (bound execution) or 3 (bound source snapshot)')
    for name in ('candidate_id', 'attempt_id', 'comparison_id', 'workload_id'):
        _text(record.get(name), name, nonempty=True)
    candidate = None
    if record['schema_version'] in (2, 3):
        _text(record.get('campaign_instance_id'), 'campaign_instance_id', nonempty=True)
        candidate = Candidate.from_dict(record.get('candidate'))
        if candidate.candidate_id != record['candidate_id'] or candidate.workload_id != record['workload_id']:
            raise ValueError('Result candidate envelope identifiers do not match')
        if record.get('job_id') is not None:
            _text(record['job_id'], 'job_id', nonempty=True)
    if record['schema_version'] == 3:
        identity = record.get('source_identity')
        if record.get('state') == 'measured':
            if identity is None:
                raise ValueError('measured schema_version 3 results require source_identity')
            validate_source_identity(identity, candidate)
        elif identity is not None:
            _object(identity, 'source_identity')  # Partial evidence is allowed on failure.
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
