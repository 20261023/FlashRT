"""Atomic campaign initialization and state changes against real SQLite."""

from dataclasses import replace
import sqlite3

import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.ledger import CampaignLedger
from flash_rt.vla.autotune.schema import Candidate


def _candidate(candidate_id='first'):
    return Candidate(candidate_id, BackendConfig(
        model='pi05', checkpoint='/weights', num_steps=5), 'workload-1')


def _entries():
    return [(_candidate('first'), 'comparison-1', 'attempt-1'),
            (_candidate('second'), 'comparison-1', 'attempt-2')]


def _assert_no_candidates_or_attempts(ledger):
    assert ledger.attempts() == []
    for candidate_id in ('first', 'second'):
        with pytest.raises(KeyError):
            ledger.get_candidate(candidate_id)


def test_initialization_is_complete_durable_and_exact_retry_is_idempotent(tmp_path):
    path = tmp_path / 'campaign.sqlite'
    with CampaignLedger(path) as ledger:
        result = ledger.initialize_attempts(_entries())
        assert [(item['candidate_id'], item['attempt_id'], item['state']) for item in result] == [
            ('first', 'attempt-1', 'proposed'), ('second', 'attempt-2', 'proposed')]
        assert ledger.get_candidate('first') == _candidate('first')
        assert ledger.get_candidate('second') == _candidate('second')
        assert all(item['job_id'] is None and item['result'] is None for item in result)
    with CampaignLedger(path) as ledger:
        before = ledger.attempts()
        assert ledger.initialize_attempts(_entries()) == before
        assert ledger.attempts() == before
        # Caller mutations cannot change the ledger snapshot.
        result[0]['state'] = 'failed'
        assert ledger.get_attempt('attempt-1')['state'] == 'proposed'


@pytest.mark.parametrize('second', [
    (object(), 'comparison-1', 'attempt-2'),
    (_candidate('second'), '', 'attempt-2'),
    (_candidate('second'), 'comparison-1', ''),
    (_candidate('second'), 'comparison-1'),
    (_candidate('second'), 'comparison-1', 'attempt-2', 'extra'),
])
def test_invalid_second_entry_leaves_no_partial_candidates_or_attempts(tmp_path, second):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        with pytest.raises((TypeError, ValueError)):
            ledger.initialize_attempts([_entries()[0], second])
        _assert_no_candidates_or_attempts(ledger)


def test_second_candidate_with_non_json_options_leaves_no_partial_rows(tmp_path):
    invalid = Candidate('second', BackendConfig(
        model='pi05', checkpoint='/weights', options={'invalid': object()}), 'workload-1')
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        with pytest.raises((TypeError, ValueError)):
            ledger.initialize_attempts([_entries()[0], (invalid, 'comparison-1', 'attempt-2')])
        _assert_no_candidates_or_attempts(ledger)


def test_conflicting_candidate_ids_within_group_roll_back_whole_initialization(tmp_path):
    conflict = replace(_candidate('first'), hypothesis='different immutable candidate')
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        with pytest.raises(ValueError):
            ledger.initialize_attempts([
                _entries()[0], (conflict, 'comparison-1', 'attempt-2')])
        _assert_no_candidates_or_attempts(ledger)


def test_duplicate_attempt_ids_are_rejected_not_silently_deduplicated(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        with pytest.raises(ValueError):
            ledger.initialize_attempts([_entries()[0], _entries()[0]])
        _assert_no_candidates_or_attempts(ledger)


def test_existing_conflicting_second_candidate_does_not_register_first(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        conflict = replace(_candidate('second'), hypothesis='already registered')
        ledger.register(conflict)
        with pytest.raises(ValueError):
            ledger.initialize_attempts(_entries())
        assert ledger.attempts() == []
        assert ledger.get_candidate('second') == conflict
        with pytest.raises(KeyError):
            ledger.get_candidate('first')


def test_identical_candidate_can_have_distinct_attempts_in_the_initial_group(tmp_path):
    first = _entries()[0]
    second = (first[0], 'comparison-2', 'attempt-2')
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.register(first[0])
        ledger.initialize_attempts([first, second])
        ledger.initialize_attempts([second, first])
        assert [(row['candidate_id'], row['comparison_id'], row['attempt_id'])
                for row in ledger.attempts()] == [
                    ('first', 'comparison-1', 'attempt-1'),
                    ('first', 'comparison-2', 'attempt-2')]


@pytest.mark.parametrize('existing_ids', [('attempt-1',), ('unrelated',)])
def test_initialization_rejects_partial_or_extra_existing_attempt_groups(tmp_path, existing_ids):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.register(_candidate('first'))
        for attempt_id in existing_ids:
            ledger.begin_attempt('first', 'comparison-1', attempt_id)
        before = ledger.attempts()
        with pytest.raises(ValueError):
            ledger.initialize_attempts(_entries())
        assert ledger.attempts() == before
        with pytest.raises(KeyError):
            ledger.get_candidate('second')


@pytest.mark.parametrize('field, value', [
    ('candidate_id', 'first'), ('comparison_id', 'different'),
    ('workload_id', 'different'), ('state', 'submitting'), ('job_id', '123'),
    ('failure_kind', 'interrupted'), ('message', 'unexpected detail'),
    ('result_json', '{}'),
])
def test_initialization_retry_rejects_changed_identity_or_nonpristine_rows(tmp_path, field, value):
    path = tmp_path / 'campaign.sqlite'
    # SQL changes model an already advanced/inconsistent persisted second row;
    # literal columns are parametrized locally, not supplied by application input.
    with CampaignLedger(path) as ledger:
        for candidate, comparison_id, attempt_id in _entries():
            ledger.register(candidate)
            ledger.begin_attempt(candidate.candidate_id, comparison_id, attempt_id)
        with sqlite3.connect(path) as connection:
            connection.execute(f'UPDATE attempts SET {field} = ? WHERE attempt_id = ?',
                               (value, 'attempt-2'))
        before = ledger.attempts()
        with pytest.raises(ValueError):
            ledger.initialize_attempts(_entries())
        assert ledger.attempts() == before


def test_trigger_failure_during_second_attempt_insert_rolls_back_every_insert(tmp_path):
    path = tmp_path / 'campaign.sqlite'
    with CampaignLedger(path) as ledger:
        with sqlite3.connect(path) as connection:
            connection.execute('''CREATE TRIGGER interrupt_initialization
                BEFORE INSERT ON attempts WHEN NEW.attempt_id = 'attempt-2'
                BEGIN SELECT RAISE(ABORT, 'injected initialization failure'); END''')
        with pytest.raises(sqlite3.IntegrityError, match='injected initialization failure'):
            ledger.initialize_attempts(_entries())
        _assert_no_candidates_or_attempts(ledger)
    with CampaignLedger(path) as ledger:
        _assert_no_candidates_or_attempts(ledger)


@pytest.mark.parametrize('entries', [[], None, 'not-entries', (_entries()[0],)])
def test_initialization_requires_a_nonempty_list_of_entries(tmp_path, entries):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        with pytest.raises((TypeError, ValueError)):
            ledger.initialize_attempts(entries)
        _assert_no_candidates_or_attempts(ledger)


def _submitting(ledger, attempt_id):
    ledger.transition(attempt_id, 'proposed', 'submitting')


def test_transition_group_moves_all_proposed_rows_and_preserves_single_api(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        assert ledger.transition_group(
            ['attempt-1', 'attempt-2'], ('proposed',), 'submitting') is None
        assert [row['state'] for row in ledger.attempts()] == ['submitting', 'submitting']
        ledger.transition('attempt-1', 'submitting', 'queued', job_id='123')
        assert ledger.get_attempt('attempt-1')['job_id'] == '123'


def test_transition_group_accepts_mixed_expected_states_for_reconciliation(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        for attempt_id in ('attempt-1', 'attempt-2'):
            _submitting(ledger, attempt_id)
        ledger.transition('attempt-2', 'submitting', 'submission_unknown', message='lost reply')
        ledger.transition_group(
            ['attempt-1', 'attempt-2'], {'submitting', 'submission_unknown'}, 'queued',
            job_id='123', allow_already=True)
        assert [(row['state'], row['job_id']) for row in ledger.attempts()] == [
            ('queued', '123'), ('queued', '123')]


def test_queued_retry_same_job_is_idempotent_without_resetting_existing_details(tmp_path):
    path = tmp_path / 'campaign.sqlite'
    with CampaignLedger(path) as ledger:
        ledger.initialize_attempts(_entries())
        for attempt_id in ('attempt-1', 'attempt-2'):
            _submitting(ledger, attempt_id)
        ledger.transition('attempt-1', 'submitting', 'queued', job_id='123', message='first evidence')
        first = ledger.get_attempt('attempt-1')
        ledger.transition_group(
            ['attempt-1', 'attempt-2'], ('submitting', 'submission_unknown'), 'queued',
            job_id='123', message='retry evidence', allow_already=True)
        assert ledger.get_attempt('attempt-1') == first
        assert ledger.get_attempt('attempt-2')['message'] == 'retry evidence'
        before = ledger.attempts()
    with CampaignLedger(path) as ledger:
        ledger.transition_group(
            ['attempt-1', 'attempt-2'], ('submitting', 'submission_unknown'), 'queued',
            job_id='123', message='must not replace', allow_already=True)
        assert ledger.attempts() == before


@pytest.mark.parametrize('existing_job_id', [None, 'different-job'])
def test_conflicting_job_on_already_queued_second_row_aborts_whole_group(tmp_path, existing_job_id):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        for attempt_id in ('attempt-1', 'attempt-2'):
            _submitting(ledger, attempt_id)
        ledger.transition('attempt-2', 'submitting', 'queued', job_id=existing_job_id)
        before = ledger.attempts()
        with pytest.raises(ValueError):
            ledger.transition_group(
                ['attempt-1', 'attempt-2'], ('submitting', 'submission_unknown'), 'queued',
                job_id='123', allow_already=True)
        assert ledger.attempts() == before


def test_allow_already_without_job_id_does_not_clear_existing_job_or_metadata(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        for attempt_id in ('attempt-1', 'attempt-2'):
            _submitting(ledger, attempt_id)
            ledger.transition(attempt_id, 'submitting', 'queued', job_id='123')
        before = ledger.attempts()
        ledger.transition_group(
            ['attempt-1', 'attempt-2'], ('submitting',), 'queued', allow_already=True)
        assert ledger.attempts() == before


def test_already_queued_is_not_accepted_without_explicit_allow_already(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        for attempt_id in ('attempt-1', 'attempt-2'):
            _submitting(ledger, attempt_id)
        ledger.transition('attempt-2', 'submitting', 'queued', job_id='123')
        before = ledger.attempts()
        with pytest.raises(ValueError):
            ledger.transition_group(
                ['attempt-1', 'attempt-2'], ('submitting',), 'queued', job_id='123')
        assert ledger.attempts() == before


def test_all_rows_are_validated_before_first_update_not_just_rolled_back(tmp_path):
    path = tmp_path / 'campaign.sqlite'
    with CampaignLedger(path) as ledger:
        ledger.initialize_attempts(_entries())
        _submitting(ledger, 'attempt-1')
        with sqlite3.connect(path) as connection:
            connection.execute('''CREATE TRIGGER forbid_first_update
                BEFORE UPDATE ON attempts WHEN OLD.attempt_id = 'attempt-1'
                BEGIN SELECT RAISE(ABORT, 'first row must not be updated'); END''')
        before = ledger.attempts()
        # A naive validate/update loop would hit the trigger instead of finding
        # that the second row is stale. The expected error is the stale-row check.
        with pytest.raises(ValueError):
            ledger.transition_group(
                ['attempt-1', 'attempt-2'], ('submitting',), 'queued', job_id='123')
        assert ledger.attempts() == before


def test_trigger_failure_on_second_update_rolls_back_first_update_and_metadata(tmp_path):
    path = tmp_path / 'campaign.sqlite'
    with CampaignLedger(path) as ledger:
        ledger.initialize_attempts(_entries())
        with sqlite3.connect(path) as connection:
            connection.execute('''CREATE TRIGGER interrupt_transition
                BEFORE UPDATE ON attempts WHEN OLD.attempt_id = 'attempt-2'
                BEGIN SELECT RAISE(ABORT, 'injected transition failure'); END''')
        before = ledger.attempts()
        with pytest.raises(sqlite3.IntegrityError, match='injected transition failure'):
            ledger.transition_group(
                ['attempt-1', 'attempt-2'], ('proposed',), 'submitting',
                job_id='123', message='must roll back', failure_kind='submission')
        assert ledger.attempts() == before
    with CampaignLedger(path) as ledger:
        assert ledger.attempts() == before


@pytest.mark.parametrize('second_state', ['running', 'failed'])
def test_group_never_resets_an_advanced_or_terminal_second_row(tmp_path, second_state):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        for attempt_id in ('attempt-1', 'attempt-2'):
            _submitting(ledger, attempt_id)
        if second_state == 'running':
            ledger.transition('attempt-2', 'submitting', 'queued', job_id='123')
            ledger.transition('attempt-2', 'queued', 'running')
        else:
            ledger.transition('attempt-2', 'submitting', 'failed', failure_kind='timeout')
        before = ledger.attempts()
        with pytest.raises(ValueError):
            ledger.transition_group(
                ['attempt-1', 'attempt-2'], ('submitting', 'submission_unknown'), 'queued',
                job_id='123', allow_already=True)
        assert ledger.attempts() == before


def test_illegal_transition_for_second_expected_state_aborts_all_rows(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        _submitting(ledger, 'attempt-1')
        before = ledger.attempts()
        with pytest.raises(ValueError):
            ledger.transition_group(
                ['attempt-1', 'attempt-2'], ('submitting', 'proposed'), 'queued', job_id='123')
        assert ledger.attempts() == before


def test_missing_second_attempt_aborts_first_row(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        before = ledger.attempts()
        with pytest.raises(KeyError):
            ledger.transition_group(['attempt-1', 'missing'], ('proposed',), 'submitting')
        assert ledger.attempts() == before


def test_failure_details_update_atomically_and_failed_group_is_terminal(tmp_path):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        ledger.transition_group(
            ['attempt-1', 'attempt-2'], ('proposed',), 'failed',
            failure_kind='scheduler_error', message='submission refused')
        assert [(row['state'], row['failure_kind'], row['message']) for row in ledger.attempts()] == [
            ('failed', 'scheduler_error', 'submission refused'),
            ('failed', 'scheduler_error', 'submission refused')]
        before = ledger.attempts()
        with pytest.raises(ValueError):
            ledger.transition_group(['attempt-1', 'attempt-2'], ('failed',), 'proposed')
        assert ledger.attempts() == before


@pytest.mark.parametrize('kwargs', [
    {'attempt_ids': []}, {'attempt_ids': 'attempt-1'},
    {'attempt_ids': ['attempt-1', 'attempt-1']}, {'attempt_ids': ['attempt-1', '']},
    {'expected_states': 'proposed'}, {'expected_states': []}, {'expected_states': {1}},
    {'expected_states': ('not-a-state',)}, {'new_state': 'not-a-state'},
    {'job_id': ''}, {'failure_kind': 1}, {'message': object()}, {'allow_already': 'yes'},
])
def test_invalid_group_arguments_never_mutate_rows(tmp_path, kwargs):
    with CampaignLedger(tmp_path / 'campaign.sqlite') as ledger:
        ledger.initialize_attempts(_entries())
        before = ledger.attempts()
        arguments = dict(attempt_ids=['attempt-1', 'attempt-2'],
                         expected_states=('proposed',), new_state='submitting')
        arguments.update(kwargs)
        with pytest.raises((TypeError, ValueError)):
            ledger.transition_group(**arguments)
        assert ledger.attempts() == before
