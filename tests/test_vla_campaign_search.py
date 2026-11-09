"""Explicit conditional search and strictly scoped, comparable winner ranking."""

from importlib import import_module

import pytest

from flash_rt.sampling import ProbeFlowConfig
from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.schema import Candidate, encode_backend
from test_vla_campaign_schema import measured_result


def _search():
    return import_module('flash_rt.vla.autotune.search')


def _baseline(*, sampler='euler'):
    return Candidate(
        'base', BackendConfig(
            model='pi05', checkpoint='/weights', hardware='cuda_sm80',
            precision='bf16', framework='torch', sampler=sampler,
            num_steps=10 if sampler == 'euler' else None,
            probeflow=None if sampler == 'euler' else ProbeFlowConfig(
                epsilon=0.01, probe_horizon=0.4, max_steps=12, step_increment=3),
            options={'image_size': 224, 'nested': {'names': ['front']}},
        ), 'workload-1', source_ref='tree-1', patch_ref='patch-1',
    )


def _record(candidate_id, samples, *, status='passed', scope='numerical_replay'):
    record = measured_result(
        candidate_id=candidate_id, attempt_id=f'attempt-{candidate_id}',
        quality={'status': status, 'scope': scope, 'metrics': {'max_abs_error': 0.01}},
    )
    record['measurement'].update(
        repeats=len(samples), latency_ms=samples,
        sampler_metadata=[{'sampler': 'euler', 'nfe': 5} for _ in samples],
    )
    return record


def test_enumeration_is_deterministic_readable_conditional_and_deduplicated():
    search = _search()
    baseline = _baseline()
    candidates = search.enumerate_candidates(
        baseline, euler_steps=(10, 2, 5, 2), epsilons=(0.008, 0.002, 0.008), max_steps=20)
    assert [point.candidate_id for point in candidates] == [
        'base-euler-steps-2', 'base-euler-steps-5', 'base-euler-steps-10',
        'base-probeflow-epsilon-0.002-max-20', 'base-probeflow-epsilon-0.008-max-20',
    ]
    assert candidates == search.enumerate_candidates(
        baseline, euler_steps=(2, 5, 10), epsilons=(0.002, 0.008), max_steps=20)
    for point in candidates[:3]:
        assert point.backend.sampler == 'euler'
        assert point.backend.probeflow is None
    assert [point.backend.num_steps for point in candidates[:3]] == [2, 5, 10]
    for point in candidates[3:]:
        assert point.backend.sampler == 'probeflow'
        assert point.backend.num_steps is None
        assert point.backend.probeflow.max_steps == 20
    assert [point.backend.probeflow.epsilon for point in candidates[3:]] == [0.002, 0.008]
    assert baseline.backend.num_steps == 10


def test_enumeration_preserves_source_workload_and_non_sampling_options():
    baseline = _baseline(sampler='probeflow')
    before = encode_backend(baseline.backend)
    candidates = _search().enumerate_candidates(baseline, euler_steps=(5,), epsilons=(0.002,))
    for point in candidates:
        encoded = encode_backend(point.backend)
        for name in ('model', 'checkpoint', 'backend', 'hardware', 'framework', 'precision', 'options'):
            assert encoded[name] == before[name]
        assert point.workload_id == 'workload-1'
        assert point.source_ref == 'tree-1'
        assert point.patch_ref == 'patch-1'
        assert point.parent_id == 'base'
        assert point.origin == 'search'
        assert 'evaluations' in point.hypothesis.lower()
        assert 'unchanged' in point.hypothesis.lower()
        assert 'prefix' in point.hypothesis.lower()
        assert 'head' in point.hypothesis.lower()
    assert candidates[1].backend.probeflow.probe_horizon == 0.4
    assert candidates[1].backend.probeflow.step_increment == 3
    candidates[0].backend.options['nested']['names'].append('mutated')
    assert baseline.backend.options['nested']['names'] == ['front']
    assert candidates[1].backend.options['nested']['names'] == ['front']


@pytest.mark.parametrize('kwargs', [
    {'euler_steps': (0,)}, {'euler_steps': (True,)}, {'euler_steps': (2.5,)},
    {'epsilons': (0,)}, {'epsilons': (True,)}, {'epsilons': (float('nan'),)},
    {'max_steps': 65}, {'max_steps': True},
])
def test_enumeration_rejects_invalid_sampling_points(kwargs):
    with pytest.raises(ValueError):
        _search().enumerate_candidates(_baseline(), **kwargs)


def test_winner_uses_median_not_average_and_quality_is_never_assumed():
    records = [
        _record('safe-fast', [1.0, 1.0, 1000.0]),
        _record('safe-slow', [4.0, 5.0, 6.0]),
        _record('unknown', [0.01], status='unknown'),
        _record('bad', [0.01], status='failed'),
        _record('wrong-scope', [0.01], scope='task_success'),
    ]
    assert _search().select_winner(records, quality_scope='numerical_replay') == 'safe-fast'
    assert _search().select_winner(records, quality_scope='task_success') == 'wrong-scope'


def test_winner_tie_break_is_candidate_id_not_input_order():
    records = [_record('z', [2, 4]), _record('a', [3, 3])]
    assert _search().select_winner(records, quality_scope='numerical_replay') == 'a'
    assert _search().select_winner(list(reversed(records)), quality_scope='numerical_replay') == 'a'


def test_gpu_job_id_is_submission_provenance_not_device_identity():
    first = _record('first', [2])
    second = _record('second', [1])
    first['gpu']['job_id'] = '1001'
    second['gpu']['job_id'] = '1002'
    assert _search().select_winner([first, second], quality_scope='numerical_replay') == 'second'


def test_no_eligible_quality_or_failed_worker_returns_none_without_task_claim():
    failed = {
        'schema_version': 1, 'candidate_id': 'failed-worker', 'attempt_id': 'failed-attempt',
        'workload_id': 'workload-1', 'comparison_id': 'comparison-1', 'state': 'failed',
        'failure': {'kind': 'runtime_error', 'message': 'OOM'},
    }
    assert _search().select_winner([
        _record('unknown', [1], status='unknown'),
        _record('bad', [1], status='failed'), failed,
    ], quality_scope='numerical_replay') is None
    assert _search().select_winner([
        _record('numerically-close', [1]),
    ], quality_scope='task_success') is None
    assert _search().select_winner([], quality_scope='numerical_replay') is None


@pytest.mark.parametrize('field, value', [
    ('workload_id', 'different-workload'), ('comparison_id', 'different-comparison'),
    ('gpu', {'name': 'NVIDIA A800', 'uuid': 'GPU-test-2'}),
    ('backend_metadata', {'action_domain': 'physical_robot_actions'}),
])
@pytest.mark.parametrize('status', ['passed', 'unknown'])
def test_mismatched_comparison_groups_are_rejected_not_silently_filtered(field, value, status):
    other = _record('other', [1], status=status)
    other[field] = value
    with pytest.raises(ValueError, match='compar|group|match'):
        _search().select_winner([
            _record('first', [2]), other,
        ], quality_scope='numerical_replay')


@pytest.mark.parametrize('field, value', [('gpu', {}), ('backend_metadata', {})])
def test_winner_requires_device_and_action_domain_evidence(field, value):
    record = _record('candidate', [1])
    record[field] = value
    with pytest.raises(ValueError):
        _search().select_winner([record], quality_scope='numerical_replay')


def test_incomplete_measurement_is_rejected_even_when_its_quality_is_unknown():
    broken = _record('broken', [1], status='unknown')
    broken['measurement']['latency_ms'] = []
    with pytest.raises(ValueError):
        _search().select_winner([
            _record('valid', [2]), broken,
        ], quality_scope='numerical_replay')


@pytest.mark.parametrize('field', ['workload_id', 'comparison_id'])
def test_failed_workers_cannot_hide_mismatched_comparison_identity(field):
    failed = {
        'schema_version': 1, 'candidate_id': 'failed-worker', 'attempt_id': 'failed-attempt',
        'workload_id': 'workload-1', 'comparison_id': 'comparison-1', 'state': 'failed',
        'failure': {'kind': 'runtime_error', 'message': 'OOM'},
    }
    failed[field] = 'different-group'
    with pytest.raises(ValueError, match='compar|group|match'):
        _search().select_winner([
            _record('valid', [2]), failed,
        ], quality_scope='numerical_replay')
