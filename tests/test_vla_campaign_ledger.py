"""Real SQLite recovery, CAS transitions, and immutable result ingestion."""

from copy import deepcopy
from dataclasses import replace
from importlib import import_module

import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.schema import Candidate
from test_vla_campaign_schema import measured_result


def _ledger(path):
    return import_module('flash_rt.vla.autotune.ledger').CampaignLedger(path)


def _candidate():
    return Candidate('base', BackendConfig(
        model='pi05', checkpoint='/weights', sampler='probeflow'), 'workload-1')


def _attempt(ledger, *, attempt_id='attempt-1', state='proposed'):
    ledger.register(_candidate())
    ledger.begin_attempt('base', 'comparison-1', attempt_id)
    route = ['proposed', 'submitting', 'queued', 'running', 'measured', 'validated']
    for current, next_state in zip(route, route[1:]):
        if current == state:
            break
        ledger.transition(attempt_id, current, next_state, job_id='12345' if next_state == 'queued' else None)


def _bound(ledger, record=None, **changes):
    record = measured_result(**changes) if record is None else record
    record.update(schema_version=2, campaign_instance_id=ledger.instance_id,
                  candidate=_candidate().to_dict(), job_id='12345')
    if record['state'] == 'measured':
        record['gpu'] = {**record['gpu'], 'job_id': '12345'}
    return record


def test_restart_recovers_candidate_attempt_and_submission_unknown_without_resubmit(tmp_path):
    path = tmp_path / 'campaign.sqlite'
    first = _ledger(path)
    first.register(_candidate())
    attempt = first.begin_attempt('base', 'comparison-1', 'attempt-1')
    assert attempt['state'] == 'proposed'
    assert attempt['workload_id'] == 'workload-1'
    first.transition('attempt-1', 'proposed', 'submitting')
    first.transition('attempt-1', 'submitting', 'submission_unknown', message='lost reply')
    first.close()
    first.close()
    with _ledger(path) as recovered:
        assert recovered.get_candidate('base') == _candidate()
        assert recovered.get_attempt('attempt-1')['state'] == 'submission_unknown'
        assert recovered.get_attempt('attempt-1')['message'] == 'lost reply'
        assert len(recovered.attempts()) == 1
        with pytest.raises(ValueError):
            recovered.transition('attempt-1', 'submission_unknown', 'submitting')
        recovered.transition('attempt-1', 'submission_unknown', 'queued', job_id='12345')
        assert recovered.get_attempt('attempt-1')['job_id'] == '12345'
    with pytest.raises(RuntimeError, match='closed'):
        recovered.attempts()


def test_registration_is_idempotent_but_candidate_and_attempt_ids_never_reused(tmp_path):
    with _ledger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.register(_candidate())
        ledger.register(_candidate())
        with pytest.raises(ValueError, match='conflict|already'):
            ledger.register(replace(_candidate(), hypothesis='different immutable record'))
        ledger.begin_attempt('base', 'comparison-1', 'attempt-1', job_id='99')
        with pytest.raises(ValueError, match='attempt|already'):
            ledger.begin_attempt('base', 'comparison-1', 'attempt-1')
        assert ledger.get_attempt('attempt-1')['job_id'] == '99'
        with pytest.raises(KeyError):
            ledger.begin_attempt('missing', 'comparison-1', 'new')
        with pytest.raises(KeyError):
            ledger.get_candidate('missing')
        with pytest.raises(KeyError):
            ledger.get_attempt('missing')


def test_two_controller_connections_cannot_apply_a_stale_transition(tmp_path):
    path = tmp_path / 'campaign.sqlite'
    with _ledger(path) as first, _ledger(path) as second:
        _attempt(first)
        first.transition('attempt-1', 'proposed', 'submitting', job_id='12')
        with pytest.raises(ValueError, match='stale|state'):
            second.transition('attempt-1', 'proposed', 'submitting', job_id='wrong')
        assert second.get_attempt('attempt-1')['job_id'] == '12'
        with pytest.raises(ValueError):
            second.transition('attempt-1', 'submitting', 'validated')
        assert first.get_attempt('attempt-1')['state'] == 'submitting'


@pytest.mark.parametrize('state', [
    'proposed', 'submitting', 'queued', 'running', 'submission_unknown',
])
def test_active_states_can_fail_but_retry_always_requires_a_new_attempt_id(tmp_path, state):
    with _ledger(tmp_path / 'campaign.sqlite') as ledger:
        _attempt(ledger, state='submitting' if state == 'submission_unknown' else state)
        if state == 'submission_unknown':
            ledger.transition('attempt-1', 'submitting', state)
        ledger.transition('attempt-1', state, 'failed',
                          failure_kind='scheduler_error', message='timeout')
        assert ledger.get_attempt('attempt-1')['failure_kind'] == 'scheduler_error'
        with pytest.raises(ValueError):
            ledger.transition('attempt-1', 'failed', 'proposed')
        ledger.begin_attempt('base', 'comparison-1', 'attempt-2')
        assert [attempt['state'] for attempt in ledger.attempts()] == ['failed', 'proposed']


@pytest.mark.parametrize('start', ['queued', 'running'])
@pytest.mark.parametrize('terminal', ['validated', 'rejected'])
def test_ingest_is_durable_idempotent_and_never_implicitly_validates(tmp_path, start, terminal):
    path = tmp_path / 'campaign.sqlite'
    with _ledger(path) as ledger:
        _attempt(ledger, state=start)
        record = _bound(ledger)
        ledger.ingest(record)
        assert ledger.get_attempt('attempt-1')['state'] == 'measured'
        assert ledger.get_attempt('attempt-1')['result'] == record
        ledger.transition('attempt-1', 'measured', terminal)
        ledger.ingest(dict(reversed(list(record.items()))))
        assert ledger.get_attempt('attempt-1')['state'] == terminal
    with _ledger(path) as ledger:
        ledger.ingest(record)
        assert ledger.get_attempt('attempt-1')['state'] == terminal
        assert ledger.get_attempt('attempt-1')['result']['measurement'][
            'sampler_metadata'][0]['nfe'] == 5
        different = deepcopy(record)
        different['measurement']['latency_ms'][0] = 1.0
        with pytest.raises(ValueError, match='[Cc]onflict|different'):
            ledger.ingest(different)
        assert ledger.get_attempt('attempt-1')['result']['measurement']['latency_ms'] == [9.0, 11.0]


@pytest.mark.parametrize('field, value', [
    ('candidate_id', 'another'), ('comparison_id', 'another'),
    ('workload_id', 'another'), ('attempt_id', 'missing'),
])
def test_ingest_rejects_identity_mismatch_without_mutating_attempt(tmp_path, field, value):
    with _ledger(tmp_path / 'campaign.sqlite') as ledger:
        _attempt(ledger, state='running')
        with pytest.raises((ValueError, KeyError)):
            ledger.ingest(_bound(ledger, **{field: value}))
        assert ledger.get_attempt('attempt-1')['state'] == 'running'
        assert ledger.get_attempt('attempt-1')['result'] is None


@pytest.mark.parametrize('state', ['proposed', 'submitting', 'failed'])
def test_measurement_cannot_skip_submission_or_revive_a_failed_attempt(tmp_path, state):
    with _ledger(tmp_path / 'campaign.sqlite') as ledger:
        _attempt(ledger, state='proposed' if state == 'failed' else state)
        if state == 'failed':
            ledger.transition('attempt-1', 'proposed', 'failed')
        with pytest.raises(ValueError):
            ledger.ingest(_bound(ledger))
        assert ledger.get_attempt('attempt-1')['state'] == state


def test_invalid_partial_result_does_not_poison_later_valid_ingestion(tmp_path):
    with _ledger(tmp_path / 'campaign.sqlite') as ledger:
        _attempt(ledger, state='queued')
        partial = _bound(ledger)
        partial['measurement'].pop('sampler_metadata')
        with pytest.raises(ValueError):
            ledger.ingest(partial)
        assert ledger.get_attempt('attempt-1')['state'] == 'queued'
        ledger.ingest(_bound(ledger))
        assert ledger.get_attempt('attempt-1')['state'] == 'measured'


def test_failed_worker_result_is_idempotent_and_preserves_failure(tmp_path):
    record = {
        'schema_version': 1, 'candidate_id': 'base', 'attempt_id': 'attempt-1',
        'comparison_id': 'comparison-1', 'workload_id': 'workload-1',
        'state': 'failed', 'failure': {'kind': 'runtime_error', 'message': 'OOM'},
    }
    with _ledger(tmp_path / 'campaign.sqlite') as ledger:
        _attempt(ledger, state='running')
        record = _bound(ledger, record)
        ledger.ingest(record)
        ledger.ingest(deepcopy(record))
        attempt = ledger.get_attempt('attempt-1')
        assert attempt['state'] == 'failed'
        assert attempt['failure_kind'] == 'runtime_error'
        assert attempt['message'] == 'OOM'
        assert attempt['result'] == record
