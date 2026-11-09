"""Real controller/worker/SQLite boundaries; only model execution is a CPU fixture."""
import json

import numpy as np
import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.controller import initialize_campaign, ingest_campaign, mark_submitting, mark_queued
from flash_rt.vla.autotune.schema import Candidate
from flash_rt.vla.autotune.worker import run_experiment
from flash_rt.vla.autotune.workloads import NPZWorkload
from flash_rt.vla.autotune.ledger import CampaignLedger


def campaign(tmp_path, name, *, checkpoint='/weights', gate=None, job_id='123'):
    root = tmp_path / name
    root.mkdir()
    np.savez(root / 'inputs.npz', noise=np.ones((2, 3), np.float32), state=np.zeros(3))
    workload = {'schema_version': 1, 'workload_id': 'frames', 'model': 'pi05', 'requests': [
        {'request_id': 'frame0', 'npz_path': 'inputs.npz', 'fields': {'state': 'state'},
         'noise_key': 'noise', 'prompt': 'pick'}]}
    (root / 'workload.json').write_text(json.dumps(workload))
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint=checkpoint, num_steps=2),
                         'frames', source_ref=f'{name}-source')
    point = Candidate('fast', BackendConfig(model='pi05', checkpoint=checkpoint, num_steps=2),
                      'frames', source_ref=f'{name}-source')
    plan = initialize_campaign(root / 'control', comparison_id='same-readable-id', baseline=baseline,
        candidates=[point], workload_path=root / 'workload.json', output_root=root / 'results',
        warmup=0, repeats=2, quality_gate=gate)
    mark_submitting(root / 'control')
    mark_queued(root / 'control', job_id)
    return root, plan


def bundles(root, plan, *, gate_override_role=None, job_id='123', bad_metric_role=None):
    class Backend:
        def prepare(self, *args, **kwargs):
            pass
        def infer(self, observation, *, prompt=None, noise=None):
            return {'actions': noise.copy(), 'raw_actions': noise.copy(),
                    'sampler_metadata': {'sampler': 'euler', 'nfe': 2},
                    'backend_metadata': {'action_domain': 'normalized'}}
        def close(self):
            pass
    reference = None
    for entry in plan['entries']:
        gate = dict(plan['quality_gate'] or {})
        if entry['role'] == gate_override_role:
            gate['max_relative_l2'] = 0.05
        output = root / 'results' / entry['attempt_id']
        record = run_experiment(Candidate.from_dict(entry['candidate']),
            NPZWorkload.from_file(root / 'workload.json'), output,
            attempt_id=entry['attempt_id'], comparison_id=plan['comparison_id'],
            campaign_instance_id=plan['campaign_instance_id'],
            warmup=0, repeats=2, quality_gate=gate, reference=reference,
            baseline_reference=entry['role'] == 'reference',
            backend_factory=lambda config: Backend(), device_info={'uuid': 'cpu-boundary', 'job_id': job_id},
            profile=False)
        if entry['role'] == bad_metric_role:
            record['quality']['metrics']['max_relative_l2'] = 0.03
        if entry['role'] == 'reference':
            reference = record['outputs_path']
        (output / 'result.json').write_text(json.dumps(record))


@pytest.mark.parametrize('role', ['candidate', 'bookend'])
def test_same_scope_looser_worker_budget_is_not_campaign_validation(tmp_path, role):
    root, plan = campaign(tmp_path, 'strict', gate={'scope': 'public-replay', 'action_dim': 2,
                                                  'max_relative_l2': 0.01})
    bundles(root, plan, gate_override_role=role)
    with pytest.raises(ValueError, match='quality.*contract|quality.*budget'):
        ingest_campaign(root / 'control', root / 'results', job_finished=True)


def test_worker_passed_flag_cannot_override_controller_metric_budget(tmp_path):
    root, plan = campaign(tmp_path, 'metrics', gate={'scope': 'public-replay', 'max_relative_l2': 0.01})
    bundles(root, plan, bad_metric_role='candidate')
    with pytest.raises(ValueError, match='quality.*budget'):
        ingest_campaign(root / 'control', root / 'results', job_finished=True)


def test_old_bundle_cannot_attach_to_same_readable_ids_in_another_campaign(tmp_path):
    gate = {'scope': 'public-replay', 'max_relative_l2': 0.05}
    old, old_plan = campaign(tmp_path, 'old', checkpoint='/old/weights', gate=gate, job_id='123')
    new, _ = campaign(tmp_path, 'new', checkpoint='/new/weights', gate=gate, job_id='456')
    bundles(old, old_plan)
    with pytest.raises(ValueError, match='execution|candidate|campaign|job'):
        ingest_campaign(new / 'control', old / 'results', job_finished=True)
    with CampaignLedger(new / 'control' / 'ledger.sqlite') as ledger:
        assert all(a['state'] == 'queued' and a['result'] is None for a in ledger.attempts())


@pytest.mark.parametrize('field', ['instance', 'job', 'gpu_job', 'checkpoint', 'options', 'source_ref', 'legacy'])
def test_execution_identity_fields_are_checked_before_any_attempt_is_imported(tmp_path, field):
    root, plan = campaign(tmp_path, 'bound', gate={'scope': 'replay', 'max_relative_l2': 0.05})
    bundles(root, plan)
    first = root / 'results' / plan['entries'][0]['attempt_id'] / 'result.json'
    record = json.loads(first.read_text())
    if field == 'instance':
        record['campaign_instance_id'] = 'different-instance'
    elif field == 'job':
        record['job_id'] = '456'
    elif field == 'gpu_job':
        record['gpu']['job_id'] = '456'
    elif field == 'source_ref':
        record['candidate']['source_ref'] = 'different-source'
    elif field == 'legacy':
        record['schema_version'] = 1
    else:
        record['candidate']['backend'][field] = '/other/weights' if field == 'checkpoint' else {'num_views': 2}
    first.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='identity|candidate|campaign|job'):
        ingest_campaign(root / 'control', root / 'results', job_finished=True)
    with CampaignLedger(root / 'control' / 'ledger.sqlite') as ledger:
        assert ledger.instance_id == plan['campaign_instance_id']
        assert all(a['state'] == 'queued' and a['result'] is None for a in ledger.attempts())
