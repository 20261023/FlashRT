"""Experiment boundary behavior, using a CPU backend at the device boundary."""
import json

import numpy as np

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.schema import Candidate
from flash_rt.vla.autotune.workloads import NPZWorkload
from flash_rt.vla.autotune.worker import run_experiment


def workload(tmp_path):
    np.savez(tmp_path / 'inputs.npz', image=np.zeros((2, 4, 4, 3), np.uint8),
             noise=np.ones((2, 3), np.float32), state=np.zeros(3))
    manifest = {'schema_version': 1, 'workload_id': 'toy-replay', 'model': 'pi05',
                'requests': [{'request_id': 'frame-0', 'npz_path': 'inputs.npz',
                              'fields': {'images': 'image', 'state': 'state'},
                              'noise_key': 'noise', 'prompt': 'pick'}]}
    path = tmp_path / 'workload.json'
    path.write_text(json.dumps(manifest))
    return NPZWorkload.from_file(path)


def test_requests_return_fresh_noise_and_observations(tmp_path):
    data = workload(tmp_path)
    first = data.request(0)
    first.noise[:] = 99
    first.observation['images'][:] = 255
    second = data.request(0)
    np.testing.assert_array_equal(second.noise, np.ones((2, 3)))
    assert not second.observation['images'].any()


class CPUBackend:
    def __init__(self):
        self.calls, self.closed, self.prepares = 0, False, 0
    def prepare(self, observation, *, prompt=None):
        self.prepares += 1
    def infer(self, observation, *, prompt=None, noise=None):
        self.calls += 1
        assert prompt == 'pick'
        return {'actions': noise.copy(), 'raw_actions': noise.copy(),
                'sampler_metadata': {'sampler': 'euler', 'nfe': 2},
                'backend_metadata': {'action_domain': 'toy'}}
    def close(self):
        self.closed = True


def test_worker_keeps_warmup_out_of_samples_and_preserves_actual_nfe(tmp_path):
    data = workload(tmp_path)
    candidate = Candidate(candidate_id='euler-2', workload_id='toy-replay',
                                backend=BackendConfig(model='pi05', checkpoint='/weights', num_steps=2))
    backend = CPUBackend()
    record = run_experiment(candidate, data, tmp_path / 'run', attempt_id='a1',
                            comparison_id='c1', warmup=3, repeats=4,
                            backend_factory=lambda config: backend,
                            device_info={'uuid': 'cpu-fixture'}, profile=False)
    assert record['state'] == 'measured'
    assert backend.calls == 7 and backend.prepares == 1 and backend.closed
    assert len(record['measurement']['latency_ms']) == 4
    assert [item['nfe'] for item in record['sampler_metadata']] == [2, 2, 2, 2]
    assert record['quality']['status'] == 'unknown'
    assert (tmp_path / 'run' / 'result.json').is_file()
    with np.load(record['outputs_path']) as outputs:
        assert len(outputs['request_ids']) == 4


def test_failed_execution_has_no_fake_zero_latency(tmp_path):
    data = workload(tmp_path)
    candidate = Candidate(candidate_id='bad', workload_id='toy-replay',
                                backend=BackendConfig(model='pi05', checkpoint='/weights'))
    def fail(config):
        raise RuntimeError('native execution failed')
    record = run_experiment(candidate, data, tmp_path / 'failed', attempt_id='a2',
                            comparison_id='c1', backend_factory=fail,
                            device_info={'uuid': 'cpu-fixture'}, profile=False)
    assert record['state'] == 'failed'
    assert record['failure']['message'] == 'native execution failed'
    assert 'measurement' not in record


def test_dynamic_native_call_counts_are_evidence_not_backend_identity(tmp_path):
    class DynamicBackend(CPUBackend):
        def infer(self, *args, **kwargs):
            result = super().infer(*args, **kwargs)
            coverage = {'fully_native': False, 'native_linear_calls': {'head': self.calls}}
            result['backend_metadata']['execution_coverage'] = coverage
            result['execution_metadata'] = coverage
            return result
    candidate = Candidate(candidate_id='partial', workload_id='toy-replay',
                                backend=BackendConfig(model='pi05', checkpoint='/weights'))
    record = run_experiment(candidate, workload(tmp_path), tmp_path / 'dynamic',
                            attempt_id='a3', comparison_id='c1', warmup=0, repeats=2,
                            backend_factory=lambda config: DynamicBackend(),
                            device_info={'uuid': 'cpu-fixture'}, profile=False)
    assert record['state'] == 'measured'
    assert [item['native_linear_calls']['head'] for item in record['execution_metadata']] == [1, 2]


def test_real_worker_bundle_roundtrips_through_controller_ledger(tmp_path):
    from flash_rt.vla.autotune.schema import Candidate, validate_result
    from flash_rt.vla.autotune.ledger import CampaignLedger
    candidate = Candidate('euler-2', BackendConfig(model='pi05', checkpoint='/weights',
                                                  num_steps=2), 'toy-replay')
    with CampaignLedger(tmp_path / 'ledger.sqlite') as ledger:
        instance_id = ledger.instance_id
    record = run_experiment(candidate, workload(tmp_path), tmp_path / 'integrated',
                            attempt_id='a4', comparison_id='c1', repeats=2,
                            campaign_instance_id=instance_id,
                            backend_factory=lambda config: CPUBackend(),
                            device_info={'uuid': 'cpu-fixture', 'job_id': '123'}, profile=False)
    validate_result(record)
    with CampaignLedger(tmp_path / 'ledger.sqlite') as ledger:
        ledger.register(candidate)
        ledger.begin_attempt(candidate.candidate_id, 'c1', 'a4')
        ledger.transition('a4', 'proposed', 'submitting')
        ledger.transition('a4', 'submitting', 'queued', job_id='123')
        ledger.ingest(record)
        ledger.ingest(record)
        assert ledger.get_attempt('a4')['state'] == 'measured'


def test_cleanup_failure_retains_samples_as_partial_not_valid_measurement(tmp_path):
    from flash_rt.vla.autotune.schema import validate_result
    class FailingClose(CPUBackend):
        def close(self):
            raise RuntimeError('graph cleanup failed')
    candidate = Candidate(candidate_id='cleanup', workload_id='toy-replay',
                                backend=BackendConfig(model='pi05', checkpoint='/weights'))
    record = run_experiment(candidate, workload(tmp_path), tmp_path / 'cleanup',
                            attempt_id='a5', comparison_id='c1', repeats=2,
                            backend_factory=lambda config: FailingClose(),
                            device_info={'uuid': 'cpu-fixture'}, profile=False)
    assert record['state'] == 'failed'
    assert len(record['partial_measurement']['latency_ms']) == 2
    validate_result(record)


def test_quality_gate_excludes_explicitly_unused_padded_action_channels(tmp_path):
    from flash_rt.vla.autotune.worker import _quality
    path = tmp_path / 'reference.npz'
    np.savez(path, request_ids=np.asarray(['one']), actions_00000=np.array([[1.0, 900.0]]))
    result = _quality([np.array([[1.0, -900.0]])], ['one'], path,
                      {'scope': 'normalized-public-actions', 'action_dim': 1, 'max_relative_l2': 0.0})
    assert result['status'] == 'passed'
    assert result['metrics']['max_relative_l2'] == 0.0
    assert result['action_dim'] == 1


def test_blank_exception_message_still_produces_valid_failed_bundle(tmp_path):
    from flash_rt.vla.autotune.schema import validate_result
    candidate = Candidate(candidate_id='blank', workload_id='toy-replay',
                                backend=BackendConfig(model='pi05', checkpoint='/weights'))
    def fail(config):
        raise AssertionError()
    record = run_experiment(candidate, workload(tmp_path), tmp_path / 'blank',
                            attempt_id='a6', comparison_id='c1', backend_factory=fail,
                            device_info={'uuid': 'cpu-fixture'}, profile=False)
    validate_result(record)
    assert record['failure']['message'] == 'AssertionError'


def test_total_sample_budget_cannot_omit_workload_tail_from_quality(tmp_path):
    data = workload(tmp_path)
    manifest = dict(data.manifest)
    manifest['requests'] = [{**data.specs[0], 'request_id': f'frame-{i}'} for i in range(3)]
    data = NPZWorkload(manifest, tmp_path)
    candidate = Candidate(candidate_id='short', workload_id='toy-replay',
                                backend=BackendConfig(model='pi05', checkpoint='/weights'))
    record = run_experiment(candidate, data, tmp_path / 'short', attempt_id='a7',
                            comparison_id='c1', warmup=0, repeats=2,
                            backend_factory=lambda config: CPUBackend(), profile=False)
    assert record['state'] == 'failed'
    assert 'cover' in record['failure']['message']


def test_last_request_error_is_included_in_complete_workload_quality(tmp_path):
    data = workload(tmp_path)
    manifest = dict(data.manifest)
    np.savez(tmp_path / 'multi.npz', noise=np.ones((2, 3), np.float32),
             **{f'state_{i}': np.full(3, i, np.float32) for i in range(3)})
    manifest['requests'] = [{**data.specs[0], 'request_id': f'frame-{i}', 'npz_path': 'multi.npz',
                              'fields': {'state': f'state_{i}'}} for i in range(3)]
    data = NPZWorkload(manifest, tmp_path)
    candidate = Candidate(candidate_id='three', workload_id='toy-replay',
                                backend=BackendConfig(model='pi05', checkpoint='/weights'))
    gate = {'scope': 'all-three-requests', 'max_relative_l2': 0.05}
    reference = run_experiment(candidate, data, tmp_path / 'reference', attempt_id='a8',
        comparison_id='c1', warmup=0, repeats=3, baseline_reference=True, quality_gate=gate,
        backend_factory=lambda config: CPUBackend(), profile=False)
    class LastWrong(CPUBackend):
        def infer(self, observation, **kwargs):
            result = super().infer(observation, **kwargs)
            if observation['state'][0] == 2:
                result['raw_actions'] += 1
            return result
    record = run_experiment(candidate, data, tmp_path / 'tail', attempt_id='a9',
        comparison_id='c1', warmup=0, repeats=3, reference=reference['outputs_path'], quality_gate=gate,
        backend_factory=lambda config: LastWrong(), profile=False)
    assert record['quality']['status'] == 'failed'
    assert record['quality']['metrics']['max_relative_l2'] == 1.0
