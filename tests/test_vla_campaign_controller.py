import json
import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.schema import Candidate
from flash_rt.vla.autotune.controller import initialize_campaign, mark_submitting, mark_queued, ingest_campaign
from flash_rt.vla.autotune.ledger import CampaignLedger


def test_campaign_has_reference_bookends_and_persistent_submission_state(tmp_path):
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights', num_steps=10), 'frames')
    other = Candidate('fast', BackendConfig(model='pi05', checkpoint='/weights', num_steps=2), 'frames',
                       parent_id='base', origin='search', hypothesis='Reduce evaluations, same prefix')
    plan = initialize_campaign(tmp_path / 'campaign', comparison_id='round1',
                               baseline=baseline, candidates=[other],
                               workload_path='/remote/inputs.json', output_root='/remote/results',
                               warmup=2, repeats=3,
                               quality_gate={'scope': 'numerical_replay', 'max_relative_l2': 0.05})
    assert [item['role'] for item in plan['entries']] == ['reference', 'candidate', 'bookend']
    assert [item['candidate']['candidate_id'] for item in plan['entries']] == ['base', 'fast', 'base']
    mark_submitting(tmp_path / 'campaign')
    mark_queued(tmp_path / 'campaign', '123')
    with CampaignLedger(tmp_path / 'campaign' / 'ledger.sqlite') as ledger:
        assert {item['state'] for item in ledger.attempts()} == {'queued'}
        assert {item['job_id'] for item in ledger.attempts()} == {'123'}
    summary = ingest_campaign(tmp_path / 'campaign', tmp_path / 'not-finished')
    assert summary['complete'] is False and summary['winner'] is None


def test_finished_job_without_results_is_failed_not_ranked(tmp_path):
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights'), 'frames')
    initialize_campaign(tmp_path / 'campaign', comparison_id='round1', baseline=baseline,
                        candidates=[], workload_path='/remote/inputs.json', output_root='/remote/results')
    mark_submitting(tmp_path / 'campaign')
    mark_queued(tmp_path / 'campaign', '123')
    summary = ingest_campaign(tmp_path / 'campaign', tmp_path / 'missing', job_finished=True)
    assert summary['complete'] is True and summary['winner'] is None
    assert all(item['state'] == 'failed' for item in summary['attempts'])


@pytest.mark.parametrize('bookend_quality', ['failed', 'unknown'])
def test_unreliable_reference_bookend_blocks_winner_even_without_timing_drift(tmp_path, bookend_quality):
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights'), 'frames')
    fast = Candidate('fast', BackendConfig(model='pi05', checkpoint='/weights', num_steps=2), 'frames')
    root = tmp_path / 'campaign'
    plan = initialize_campaign(root, comparison_id='r1', baseline=baseline, candidates=[fast],
                               workload_path='/inputs.json', output_root='/results',
                               quality_gate={'scope': 'numerical_replay', 'max_relative_l2': 0.05})
    mark_submitting(root)
    mark_queued(root, '123')
    for entry in plan['entries']:
        status = bookend_quality if entry['role'] == 'bookend' else 'passed'
        value = 3.0 if entry['role'] == 'candidate' else 10.0
        record = {'schema_version': 2, 'candidate_id': entry['candidate']['candidate_id'],
                  'candidate': entry['candidate'], 'campaign_instance_id': plan['campaign_instance_id'],
                  'job_id': '123',
                  'attempt_id': entry['attempt_id'], 'comparison_id': 'r1', 'workload_id': 'frames',
                  'state': 'measured', 'measurement': {'warmup': 0, 'repeats': 2,
                    'latency_ms': [value, value], 'load_ms': 1.0, 'prepare_ms': 1.0,
                    'sampler_metadata': [{'sampler': 'euler', 'nfe': 2}] * 2},
                  'backend_metadata': {'action_domain': 'normalized'}, 'gpu': {'uuid': 'GPU-one', 'job_id': '123'},
                  'outputs_path': '/outputs.npz',
                  'quality': {'status': status, 'scope': 'numerical_replay',
                              'action_dim': None, 'gate': {'max_relative_l2': 0.05},
                              'metrics': {'max_relative_l2': 0.1 if status == 'failed' else 0.0}}}
        directory = tmp_path / 'results' / entry['attempt_id']
        directory.mkdir(parents=True)
        (directory / 'result.json').write_text(json.dumps(record))
    summary = ingest_campaign(root, tmp_path / 'results', job_finished=True)
    assert summary['winner'] is None


def test_unknown_submission_can_be_reconciled_without_resubmission(tmp_path):
    from flash_rt.vla.autotune.controller import mark_submission_unknown
    root = tmp_path / 'campaign'
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights'), 'frames')
    initialize_campaign(root, comparison_id='r1', baseline=baseline, candidates=[],
                        workload_path='/inputs.json', output_root='/results')
    mark_submitting(root)
    mark_submission_unknown(root, 'SSH response lost')
    mark_queued(root, '123')
    mark_queued(root, '123')
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        assert {item['state'] for item in ledger.attempts()} == {'queued'}


def test_bad_later_candidate_can_be_corrected_in_same_root(tmp_path):
    root = tmp_path / 'campaign'
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights'), 'frames')
    bad = Candidate('bad', BackendConfig(model='pi0', checkpoint='/weights'), 'frames')
    with pytest.raises(ValueError):
        initialize_campaign(root, comparison_id='r1', baseline=baseline, candidates=[bad],
                            workload_path='/inputs.json', output_root='/results')
    plan = initialize_campaign(root, comparison_id='r1', baseline=baseline, candidates=[],
                               workload_path='/inputs.json', output_root='/results')
    assert len(plan['entries']) == 2
