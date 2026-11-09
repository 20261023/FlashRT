import subprocess

import pytest

from flash_rt.vla.autotune.scheduler import SlurmClient


def test_scheduler_uses_local_queue_query_and_retains_priority_reason():
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='123|PENDING|Priority\n', stderr='')
    status = SlurmClient(runner=run).status('123')
    assert '--local' in calls[0]
    assert status == {'job_id': '123', 'state': 'PENDING', 'reason': 'Priority', 'terminal': False}


def test_accounting_is_used_when_job_left_queue_but_never_implies_quality():
    def run(argv, **kwargs):
        output = '' if argv[0] == 'squeue' else '123|COMPLETED|0:0\n'
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr='')
    status = SlurmClient(runner=run).status('123')
    assert status['state'] == 'COMPLETED' and status['terminal'] is True
    assert 'quality' not in status


def test_submission_response_loss_is_not_automatically_retried(tmp_path):
    from flash_rt.vla import BackendConfig
    from flash_rt.vla.autotune.schema import Candidate
    from flash_rt.vla.autotune.controller import initialize_campaign, submit_campaign
    from flash_rt.vla.autotune.ledger import CampaignLedger
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights'), 'frames')
    root = tmp_path / 'campaign'
    initialize_campaign(root, comparison_id='r1', baseline=baseline, candidates=[],
                        workload_path='/inputs.json', output_root='/results')
    class LostResponse:
        calls = 0
        def submit(self, script):
            self.calls += 1
            raise subprocess.TimeoutExpired('sbatch', 60)
    client = LostResponse()
    with pytest.raises(subprocess.TimeoutExpired):
        submit_campaign(root, '/remote/run.sbatch', client)
    with pytest.raises(ValueError):
        submit_campaign(root, '/remote/run.sbatch', client)
    assert client.calls == 1
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        assert {item['state'] for item in ledger.attempts()} == {'submission_unknown'}


def test_wait_reconciles_known_job_without_submitting_or_validating_results(tmp_path):
    from flash_rt.vla import BackendConfig
    from flash_rt.vla.autotune.schema import Candidate
    from flash_rt.vla.autotune.controller import initialize_campaign, mark_submitting, mark_queued, wait_campaign
    from flash_rt.vla.autotune.ledger import CampaignLedger
    root = tmp_path / 'campaign'
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights'), 'frames')
    initialize_campaign(root, comparison_id='r1', baseline=baseline, candidates=[],
                        workload_path='/inputs.json', output_root='/results')
    mark_submitting(root)
    mark_queued(root, '123')
    class Client:
        def __init__(self):
            self.states = iter(['PENDING', 'RUNNING', 'COMPLETED'])
        def status(self, job_id):
            state = next(self.states)
            return {'job_id': job_id, 'state': state, 'terminal': state == 'COMPLETED', 'exit_code': '0:0'}
    sleeps = []
    result = wait_campaign(root, Client(), interval=15, timeout=60,
                           sleep=sleeps.append, clock=lambda: 0.0)
    assert result['scheduler_success'] is True
    assert sleeps == [15, 15]
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        assert {item['state'] for item in ledger.attempts()} == {'running'}
        assert all(item['result'] is None for item in ledger.attempts())


@pytest.mark.parametrize('terminal, exit_code', [('FAILED', '1:0'), ('COMPLETED', '0:9')])
def test_wait_terminal_scheduler_failure_does_not_invent_worker_failure(tmp_path, terminal, exit_code):
    from flash_rt.vla import BackendConfig
    from flash_rt.vla.autotune.schema import Candidate
    from flash_rt.vla.autotune.controller import initialize_campaign, mark_submitting, mark_queued, wait_campaign
    from flash_rt.vla.autotune.ledger import CampaignLedger
    root = tmp_path / 'campaign'
    initialize_campaign(root, comparison_id='r1', baseline=Candidate(
        'base', BackendConfig(model='pi05', checkpoint='/weights'), 'frames'), candidates=[],
        workload_path='/inputs.json', output_root='/results')
    mark_submitting(root)
    mark_queued(root, '123')
    class Client:
        def status(self, job_id):
            return {'job_id': job_id, 'state': terminal, 'terminal': True, 'exit_code': exit_code}
    result = wait_campaign(root, Client())
    assert result['scheduler_success'] is False and result['timed_out'] is False
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        assert all(item['state'] == 'queued' and item['result'] is None for item in ledger.attempts())


def test_wait_timeout_preserves_unknown_job_and_refuses_unconfirmed_submission(tmp_path):
    from flash_rt.vla import BackendConfig
    from flash_rt.vla.autotune.schema import Candidate
    from flash_rt.vla.autotune.controller import initialize_campaign, mark_submitting, mark_queued, wait_campaign
    root = tmp_path / 'campaign'
    initialize_campaign(root, comparison_id='r1', baseline=Candidate(
        'base', BackendConfig(model='pi05', checkpoint='/weights'), 'frames'), candidates=[],
        workload_path='/inputs.json', output_root='/results')
    class Client:
        calls = 0
        def status(self, job_id):
            self.calls += 1
            return {'job_id': job_id, 'state': 'UNKNOWN', 'terminal': False}
    client = Client()
    with pytest.raises(ValueError, match='confirmed'):
        wait_campaign(root, client)
    assert client.calls == 0
    mark_submitting(root)
    mark_queued(root, '123')
    now = [0.0]
    def sleep(seconds):
        now[0] += seconds
    result = wait_campaign(root, client, interval=15, timeout=20, sleep=sleep, clock=lambda: now[0])
    assert result['timed_out'] is True and result['scheduler_success'] is False
    assert now == [20.0]


def test_expired_controller_job_uses_accounting_but_connection_errors_propagate():
    calls = []
    def expired(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'squeue':
            raise subprocess.CalledProcessError(1, argv, stderr='slurm_load_jobs error: Invalid job id specified\n')
        return subprocess.CompletedProcess(argv, 0, stdout='123|COMPLETED|0:0\n', stderr='')
    status = SlurmClient(runner=expired).status('123')
    assert status['terminal'] is True and status['state'] == 'COMPLETED'
    assert len(calls) == 2 and calls[1][0] == 'sacct'
    def unavailable(argv, **kwargs):
        raise subprocess.CalledProcessError(255, argv, stderr='ssh: connect to host gpu: Connection refused')
    with pytest.raises(subprocess.CalledProcessError):
        SlurmClient(runner=unavailable).status('123')
