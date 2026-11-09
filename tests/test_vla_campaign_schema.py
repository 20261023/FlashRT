"""Persistent records must preserve policies, actual NFE, and full samples."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from importlib import import_module
import json

import pytest

from flash_rt.sampling import ProbeFlowConfig
from flash_rt.vla import BackendConfig


def _schema():
    return import_module('flash_rt.vla.autotune.schema')


def measured_result(**changes):
    record = {
        'schema_version': 1, 'candidate_id': 'base', 'attempt_id': 'attempt-1',
        'comparison_id': 'comparison-1', 'workload_id': 'workload-1',
        'state': 'measured',
        'measurement': {
            'warmup': 1, 'repeats': 2, 'latency_ms': [9.0, 11.0],
            'load_ms': 100.0, 'prepare_ms': 20.0,
            'sampler_metadata': [
                {'sampler': 'probeflow', 'scheduled_steps': 4, 'nfe': 5},
                {'sampler': 'probeflow', 'scheduled_steps': 2, 'nfe': 2},
            ],
        },
        'backend_metadata': {'action_domain': 'normalized_padded'},
        'gpu': {'name': 'NVIDIA A800', 'uuid': 'GPU-test-1'},
        'outputs_path': '/results/attempt-1.npz',
        'quality': {'status': 'unknown', 'scope': '', 'metrics': {}},
        'stage_metrics': {'prefix_ms': [2.0, 3.0]},
    }
    record.update(changes)
    return record


def test_backend_codec_emits_every_constructor_field_and_roundtrips_json():
    schema = _schema()
    config = BackendConfig(
        model='pi0.5', checkpoint='/weights/a b', hardware='cuda_sm80',
        framework='torch', precision='bf16', sampler='probeflow',
        probeflow=ProbeFlowConfig(epsilon=0.002, probe_horizon=0.4,
                                  max_steps=12, step_increment=3),
        options={'image_size': 224, 'nested': {'enabled': True, 'names': ['x']}},
    )
    encoded = schema.encode_backend(config)
    assert encoded == {
        'model': 'pi05', 'checkpoint': '/weights/a b', 'backend': 'flashrt',
        'sampler': 'probeflow', 'hardware': 'cuda_sm80', 'framework': 'torch',
        'precision': 'bf16', 'num_steps': None,
        'probeflow': {'epsilon': 0.002, 'probe_horizon': 0.4, 'min_steps': 2,
                      'max_steps': 12, 'step_increment': 3},
        'options': {'image_size': 224,
                    'nested': {'enabled': True, 'names': ['x']}},
    }
    restored = schema.decode_backend(json.loads(json.dumps(encoded)))
    assert restored == config
    encoded['options']['nested']['names'].append('mutated')
    assert config.options['nested']['names'] == ['x']


def test_euler_codec_preserves_steps_and_rejects_descriptive_unknown_fields():
    schema = _schema()
    encoded = schema.encode_backend(BackendConfig(
        model='pi0', checkpoint='/weights', num_steps=5))
    assert encoded['num_steps'] == 5
    assert encoded['probeflow'] is None
    assert schema.decode_backend(encoded).num_steps == 5
    with pytest.raises(ValueError, match='unknown|Unknown'):
        schema.decode_backend({**encoded, 'runtime_verified': False})
    policy = schema.encode_backend(BackendConfig(
        model='pi0', checkpoint='/weights', sampler='probeflow'))
    policy['probeflow']['typo'] = 1
    with pytest.raises(ValueError, match='unknown|Unknown'):
        schema.decode_backend(policy)


@pytest.mark.parametrize('bad', [object(), {'nested': float('nan')}, {1: 'not-json-key'}])
def test_backend_codec_rejects_non_json_options(bad):
    schema = _schema()
    with pytest.raises((TypeError, ValueError)):
        schema.encode_backend(BackendConfig(
            model='pi05', checkpoint='/weights', options={'bad': bad}))


def test_candidate_roundtrip_preserves_provenance_and_identity_is_frozen():
    schema = _schema()
    candidate = schema.Candidate(
        candidate_id='candidate-1', backend=BackendConfig(
            model='pi05', checkpoint='/weights', num_steps=5),
        workload_id='workload-1', parent_id='base', hypothesis='fewer evaluations',
        origin='agent', patch_ref='patch-1', source_ref='tree-1',
    )
    assert schema.Candidate.from_dict(json.loads(json.dumps(candidate.to_dict()))) == candidate
    assert candidate.to_dict()['source_ref'] == 'tree-1'
    with pytest.raises(FrozenInstanceError):
        candidate.candidate_id = 'replacement'
    with pytest.raises(ValueError, match='unknown|Unknown'):
        schema.Candidate.from_dict({**candidate.to_dict(), 'ignored': True})
    with pytest.raises(ValueError, match='origin'):
        schema.Candidate('bad', candidate.backend, 'workload-1', origin='random')
    with pytest.raises(ValueError, match='candidate_id'):
        schema.Candidate('', candidate.backend, 'workload-1')


def test_validate_result_preserves_actual_nfe_and_unrecognized_extra_metrics():
    schema = _schema()
    record = measured_result()
    before = deepcopy(record)
    assert schema.validate_result(record) is None
    assert record == before
    assert record['measurement']['sampler_metadata'][0]['nfe'] == 5
    assert record['stage_metrics'] == {'prefix_ms': [2.0, 3.0]}


@pytest.mark.parametrize('field', [
    'candidate_id', 'attempt_id', 'comparison_id', 'workload_id', 'schema_version',
    'measurement', 'backend_metadata', 'gpu', 'outputs_path', 'quality',
])
def test_measured_result_rejects_missing_required_field(field):
    record = measured_result()
    record.pop(field)
    with pytest.raises(ValueError):
        _schema().validate_result(record)


@pytest.mark.parametrize('field, value', [
    ('warmup', -1), ('warmup', True), ('repeats', 0), ('repeats', 2.0),
    ('latency_ms', [1.0]), ('latency_ms', [float('nan'), 1.0]),
    ('latency_ms', [float('inf'), 1.0]), ('latency_ms', [-1.0, 1.0]),
    ('latency_ms', [True, 1.0]), ('load_ms', -1), ('prepare_ms', float('inf')),
    ('sampler_metadata', [{'sampler': 'euler', 'nfe': 5}]),
    ('sampler_metadata', [{'sampler': 'euler', 'nfe': 0}] * 2),
    ('sampler_metadata', [{'sampler': 'euler', 'nfe': True}] * 2),
    ('sampler_metadata', [{'sampler': 'euler', 'scheduled_steps': 5}] * 2),
    ('sampler_metadata', [{'sampler': 'typo', 'nfe': 5}] * 2),
])
def test_measurement_rejects_partial_nonfinite_and_fake_actual_nfe(field, value):
    record = measured_result()
    record['measurement'][field] = value
    with pytest.raises(ValueError):
        _schema().validate_result(record)


@pytest.mark.parametrize('quality', [
    {}, {'status': 'success', 'scope': 'replay', 'metrics': {}},
    {'status': 'passed', 'scope': None, 'metrics': {}},
    {'status': 'passed', 'scope': 'replay', 'metrics': []},
])
def test_quality_requires_explicit_status_scope_and_metrics(quality):
    with pytest.raises(ValueError):
        _schema().validate_result(measured_result(quality=quality))


def test_failed_result_requires_failure_not_invented_measurements():
    record = {key: value for key, value in measured_result().items()
              if key in ('schema_version', 'candidate_id', 'attempt_id',
                         'comparison_id', 'workload_id')}
    record.update(state='failed', failure={'kind': 'worker_error', 'message': 'OOM'})
    assert _schema().validate_result(record) is None
    with pytest.raises(ValueError):
        _schema().validate_result({**record, 'failure': {'kind': 'worker_error'}})
    with pytest.raises(ValueError):
        _schema().validate_result({**record, 'measurement': {'latency_ms': [0]}})
