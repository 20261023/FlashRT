import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.batch import run_batch
from flash_rt.vla.autotune.controller import initialize_campaign, ingest_campaign, mark_submitting, mark_queued
from flash_rt.vla.autotune.schema import Candidate
from flash_rt.vla.autotune.worker import run_experiment
from flash_rt.vla.autotune.workloads import NPZWorkload


@pytest.mark.parametrize('fail_fast', [False, True])
def test_controller_batch_worker_ledger_quality_loop(tmp_path, fail_fast):
    np.savez(tmp_path / 'inputs.npz', noise=np.ones((2, 3), np.float32), state=np.zeros(3))
    workload_path = tmp_path / 'workload.json'
    workload_path.write_text(json.dumps({
        'schema_version': 1, 'workload_id': 'frames', 'model': 'pi05',
        'requests': [{'request_id': 'frame0', 'npz_path': 'inputs.npz',
                      'fields': {'state': 'state'}, 'noise_key': 'noise', 'prompt': 'pick'}]}))
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights', num_steps=10), 'frames')
    fast = Candidate('fast', BackendConfig(model='pi05', checkpoint='/weights', num_steps=2), 'frames', parent_id='base')
    gate = {'scope': 'numerical_replay', 'max_relative_l2': 0.05}
    plan = initialize_campaign(tmp_path / 'control', comparison_id='round1', baseline=baseline,
                               candidates=[fast], workload_path=workload_path,
                               output_root=tmp_path / 'results', warmup=1, repeats=2, quality_gate=gate)
    mark_submitting(tmp_path / 'control')
    mark_queued(tmp_path / 'control', '123')
    class Backend:
        def __init__(self, config):
            self.steps = config.num_steps
        def prepare(self, *args, **kwargs):
            pass
        def infer(self, observation, *, prompt=None, noise=None):
            return {'actions': noise.copy(), 'sampler_metadata': {'sampler': 'euler', 'nfe': self.steps},
                    'backend_metadata': {'action_domain': 'normalized'}}
        def close(self):
            pass
    seen_references = []
    def execute(argv, **kwargs):
        def value(flag):
            return argv[argv.index(flag) + 1] if flag in argv else None
        candidate = Candidate.from_dict(json.loads(Path(value('--candidate')).read_text()))
        reference = value('--reference')
        seen_references.append(reference)
        def factory(config):
            if fail_fast and candidate.candidate_id == 'fast':
                raise AssertionError()
            return Backend(config)
        record = run_experiment(candidate, NPZWorkload.from_file(value('--workload')), value('--output'),
                                attempt_id=value('--attempt-id'), comparison_id=value('--comparison-id'),
                                campaign_instance_id=value('--campaign-instance-id'),
                                warmup=1, repeats=2, reference=reference, quality_gate=gate,
                                baseline_reference='--baseline-reference' in argv,
                                backend_factory=factory, device_info={'uuid': 'CPU-boundary', 'job_id': '123'}, profile=False)
        if record['state'] == 'measured':
            record['measurement']['latency_ms'] = [10.0, 10.0] if candidate.candidate_id == 'base' else [3.0, 3.0]
        (Path(value('--output')) / 'result.json').write_text(json.dumps(record))
        return subprocess.CompletedProcess(argv, 0 if record['state'] == 'measured' else 1)
    result = run_batch(plan, execute=execute, python='explicit-worker-python')
    assert result['complete'] is True
    assert seen_references[0] is None and seen_references[1] == seen_references[2]
    summary = ingest_campaign(tmp_path / 'control', tmp_path / 'results', job_finished=True)
    expected_winner = 'base' if fail_fast else 'fast'
    assert summary['winner'] == expected_winner and summary['baseline_drift_ratio'] == 0.0
    again = ingest_campaign(tmp_path / 'control', tmp_path / 'results', job_finished=True)
    assert again['winner'] == expected_winner
    assert [item['state'] for item in again['attempts']] == (
        ['validated', 'failed', 'validated'] if fail_fast else ['validated'] * 3)
