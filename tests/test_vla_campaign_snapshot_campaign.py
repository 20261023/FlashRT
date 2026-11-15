"""Snapshot-mode campaigns (plan schema 3): init, batch isolation, worker identity, ingest."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune import snapshots as snap
from flash_rt.vla.autotune.batch import run_batch
from flash_rt.vla.autotune.controller import (ingest_campaign, initialize_campaign, main as controller_main,
                                               mark_queued, mark_submitting)
from flash_rt.vla.autotune.ledger import CampaignLedger
from flash_rt.vla.autotune.schema import Candidate, validate_result
from flash_rt.vla.autotune.worker import _describe_dataflow, run_experiment
from flash_rt.vla.autotune.workloads import NPZWorkload

from test_vla_campaign_snapshot import STUB_WORKER, head, make_patch, make_repo, pytestmark, ready_snapshot  # noqa: F401


GATE = {'scope': 'numerical_replay', 'max_relative_l2': 0.05}


class Backend:
    def __init__(self, config=None):
        self.steps = 2 if config is None else config.num_steps

    def prepare(self, *args, **kwargs):
        pass

    def infer(self, observation, *, prompt=None, noise=None):
        return {'actions': noise.copy(), 'sampler_metadata': {'sampler': 'euler', 'nfe': self.steps},
                'backend_metadata': {'action_domain': 'normalized'}}

    def close(self):
        pass


@pytest.fixture
def world(tmp_path):
    """A fixture repo, a ready baseline snapshot, a ready patched snapshot and a workload."""
    repo = make_repo(tmp_path / 'repo')
    store = tmp_path / 'store'
    base = head(repo)
    baseline_manifest = ready_snapshot(repo, store)
    patched_manifest = ready_snapshot(repo, store, {'flash_rt/model.py': 'STEPS = 4\n'})
    np.savez(tmp_path / 'inputs.npz', noise=np.ones((2, 3), np.float32), state=np.zeros(3))
    workload = tmp_path / 'workload.json'
    workload.write_text(json.dumps({
        'schema_version': 1, 'workload_id': 'frames', 'model': 'pi05',
        'requests': [{'request_id': 'frame0', 'npz_path': 'inputs.npz', 'fields': {'state': 'state'},
                      'noise_key': 'noise', 'prompt': 'pick'}]}))
    baseline = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights', num_steps=2), 'frames',
                         source_ref=base)
    patched = Candidate('fast', BackendConfig(model='pi05', checkpoint='/weights', num_steps=2), 'frames',
                        parent_id='base', origin='agent', source_ref=base,
                        patch_ref=patched_manifest['patch_ref'], tool='kernel.test', stage='denoise.velocity')
    return {'tmp': tmp_path, 'repo': repo, 'store': store, 'base': base, 'baseline': baseline,
            'patched': patched, 'baseline_manifest': baseline_manifest,
            'patched_manifest': patched_manifest, 'workload': workload}


def init(world, name='control', **changes):
    arguments = dict(comparison_id='r1', baseline=world['baseline'], candidates=[world['patched']],
                     workload_path=world['workload'], output_root=world['tmp'] / 'results',
                     warmup=0, repeats=2, quality_gate=GATE, snapshot_store=world['store'])
    arguments.update(changes)
    return initialize_campaign(world['tmp'] / name, **arguments)


def queued(world, name='control'):
    mark_submitting(world['tmp'] / name)
    mark_queued(world['tmp'] / name, '123')


def in_process_execute(world, calls, *, factory_calls=None):
    """Fake executor: run the real worker in-process against the entry's own snapshot (PYTHONPATH[0])."""
    def execute(argv, **kwargs):
        calls.append((argv, kwargs))

        def value(flag):
            return argv[argv.index(flag) + 1] if flag in argv else None
        candidate = Candidate.from_dict(json.loads(Path(value('--candidate')).read_text()))

        def factory(config):
            if factory_calls is not None:
                factory_calls.append(candidate.candidate_id)
            return Backend(config)
        record = run_experiment(
            candidate, NPZWorkload.from_file(value('--workload')), value('--output'),
            attempt_id=value('--attempt-id'), comparison_id=value('--comparison-id'),
            campaign_instance_id=value('--campaign-instance-id'), warmup=0, repeats=2,
            reference=value('--reference'), quality_gate=GATE,
            baseline_reference='--baseline-reference' in argv, backend_factory=factory,
            device_info={'uuid': 'CPU-boundary', 'job_id': '123'}, profile=False,
            expected_source=json.loads(Path(value('--source-manifest')).read_text()),
            source_import_root=kwargs['env']['PYTHONPATH'].split(os.pathsep)[0])
        if record['state'] == 'measured':
            record['measurement']['latency_ms'] = [10.0, 10.0] if candidate.candidate_id == 'base' else [3.0, 3.0]
            (Path(value('--output')) / 'result.json').write_text(json.dumps(record))
        return subprocess.CompletedProcess(argv, 0 if record['state'] == 'measured' else 1)
    return execute


# --------------------------------------------------------------------------- controller init

def test_snapshot_init_writes_plan_v3_and_registers_snapshots_with_attempts(world):
    plan = init(world)
    baseline_id, patched_id = world['baseline_manifest']['snapshot_id'], world['patched_manifest']['snapshot_id']
    assert plan['schema_version'] == 3
    assert plan['harness'] == {'autotune_tree': world['baseline_manifest']['harness_tree']}
    assert plan['runtime'] == {'python': None, 'extra_pythonpath': []}
    assert [entry['snapshot_id'] for entry in plan['entries']] == [baseline_id, patched_id, baseline_id]
    assert [entry['role'] for entry in plan['entries']] == ['reference', 'candidate', 'bookend']
    assert set(plan['snapshots']) == {baseline_id, patched_id}
    for identifier, record in plan['snapshots'].items():
        manifest = {key: value for key, value in record.items() if key != 'manifest_path'}
        assert manifest == snap.load_manifest(world['store'], identifier)
        assert record['manifest_path'] == str(snap.manifest_path(world['store'], identifier).resolve())
    assert plan['entries'][1]['candidate']['tool'] == 'kernel.test'
    assert json.loads((world['tmp'] / 'control' / 'plan.json').read_text()) == plan
    with CampaignLedger(world['tmp'] / 'control' / 'ledger.sqlite') as ledger:
        assert ledger.snapshots() == {identifier: snap.load_manifest(world['store'], identifier)
                                      for identifier in (baseline_id, patched_id)}
        assert len(ledger.attempts()) == 3


def test_init_without_snapshot_store_is_unchanged_schema_2(world):
    plan = initialize_campaign(world['tmp'] / 'v2', comparison_id='r1', baseline=world['baseline'],
                               candidates=[world['patched']], workload_path='relative/workload.json',
                               output_root='relative/results', warmup=0, repeats=2)
    assert set(plan) == {'schema_version', 'comparison_id', 'campaign_instance_id', 'workload_path',
                         'output_root', 'warmup', 'repeats', 'quality_gate', 'entries'}
    assert plan['schema_version'] == 2
    assert all(set(entry) == {'role', 'attempt_id', 'candidate'} for entry in plan['entries'])
    with pytest.raises(ValueError, match='snapshot_store'):
        initialize_campaign(world['tmp'] / 'v2b', comparison_id='r1', baseline=world['baseline'],
                            candidates=[], workload_path='w', output_root='o', python='/usr/bin/python3')


def _assert_nothing_written(world, name='control'):
    assert not (world['tmp'] / name / 'plan.json').exists()
    assert not (world['tmp'] / name / 'ledger.sqlite').exists()


@pytest.mark.parametrize('source_ref', ['main', 'HEAD', 'abc123', None])
def test_snapshot_init_rejects_non_sha_source_refs(world, source_ref):
    bad = Candidate('bad', BackendConfig(model='pi05', checkpoint='/weights', num_steps=3), 'frames',
                    source_ref=source_ref)
    with pytest.raises(ValueError, match='source_ref'):
        init(world, candidates=[world['patched'], bad])
    _assert_nothing_written(world)


def test_snapshot_init_rejects_free_form_patch_refs(world):
    bad = Candidate('bad', BackendConfig(model='pi05', checkpoint='/weights', num_steps=3), 'frames',
                    source_ref=world['base'], patch_ref='my-local-change')
    with pytest.raises(ValueError, match='patch_ref'):
        init(world, candidates=[bad])
    _assert_nothing_written(world)


def test_snapshot_init_rejects_unready_or_missing_manifests(world):
    pending = snap.materialize(world['repo'], world['store'], world['base'],
                               snap.store_patch(world['store'], make_patch(world['repo'], {'csrc/k.cu': '// v2\n'})))
    point = Candidate('pending', BackendConfig(model='pi05', checkpoint='/weights', num_steps=3), 'frames',
                      source_ref=world['base'], patch_ref=pending['patch_ref'])
    with pytest.raises(ValueError, match='not ready'):
        init(world, candidates=[point])
    _assert_nothing_written(world)
    missing = Candidate('missing', BackendConfig(model='pi05', checkpoint='/weights', num_steps=3), 'frames',
                        source_ref=world['base'], patch_ref='sha256:' + 'e' * 64)
    with pytest.raises(ValueError, match='not materialized'):
        init(world, candidates=[missing])
    _assert_nothing_written(world)


def test_snapshot_init_rejects_mixed_harness_trees(world):
    repo = world['repo']
    (repo / 'flash_rt' / 'vla' / 'autotune' / '__init__.py').write_text('"""new harness"""\n')
    from test_vla_campaign_snapshot import git
    git(repo, 'commit', '-qam', 'harness change')
    other = ready_snapshot(repo, world['store'])
    assert other['harness_tree'] != world['baseline_manifest']['harness_tree']
    point = Candidate('other', BackendConfig(model='pi05', checkpoint='/weights', num_steps=3), 'frames',
                      source_ref=other['base_commit'])
    with pytest.raises(ValueError, match='harness'):
        init(world, candidates=[point])
    _assert_nothing_written(world)


@pytest.mark.parametrize('field,value', [('workload_path', 'workload.json'), ('output_root', 'results'),
                                         ('snapshot_store', 'store'), ('python', 'bin/python'),
                                         ('extra_pythonpath', ['relative/site'])])
def test_snapshot_init_requires_absolute_paths(world, field, value):
    with pytest.raises(ValueError, match='absolute'):
        init(world, **{field: value})
    _assert_nothing_written(world)


def test_snapshot_init_rejects_extra_pythonpath_with_another_flash_rt(world, tmp_path):
    (tmp_path / 'shadow' / 'flash_rt').mkdir(parents=True)
    (tmp_path / 'shadow' / 'flash_rt' / '__init__.py').write_text('')
    with pytest.raises(ValueError, match='another flash_rt'):
        init(world, extra_pythonpath=[str(tmp_path / 'shadow')])


def test_ledger_snapshot_registration_is_immutable(world):
    with CampaignLedger(world['tmp'] / 'ledger.sqlite') as ledger:
        ledger.register_snapshot(world['baseline_manifest'])
        ledger.register_snapshot(world['baseline_manifest'])
        changed = {**world['baseline_manifest'], 'native': {'status': 'reused', 'fingerprint': 'x',
                                                            'artifacts': {'flash_rt/k.so': 'f' * 64}}}
        with pytest.raises(ValueError, match='Conflicting snapshot'):
            ledger.register_snapshot(changed)
        pending = {**world['patched_manifest'], 'native': {'status': 'pending', 'fingerprint': None, 'artifacts': {}}}
        with pytest.raises(ValueError, match='not ready'):
            ledger.register_snapshot(pending)
        assert ledger.snapshots() == {world['baseline_manifest']['snapshot_id']: world['baseline_manifest']}


def test_interrupted_snapshot_init_cannot_be_retried_as_schema_2(world):
    init(world)
    (world['tmp'] / 'control' / 'plan.json').unlink()  # Crash between the ledger commit and the plan write.
    with pytest.raises(ValueError, match='snapshot set must match'):
        initialize_campaign(world['tmp'] / 'control', comparison_id='r1', baseline=world['baseline'],
                            candidates=[world['patched']], workload_path=world['workload'],
                            output_root=world['tmp'] / 'results', warmup=0, repeats=2, quality_gate=GATE)
    assert not (world['tmp'] / 'control' / 'plan.json').exists()
    assert init(world)['schema_version'] == 3  # The faithful retry still publishes the plan.


def test_ledger_reinitialization_requires_the_stored_snapshot_set(world):
    extra = ready_snapshot(world['repo'], world['store'], {'flash_rt/model.py': 'STEPS = 7\n'})
    entries = [(world['baseline'], 'c', 'a1'), (world['patched'], 'c', 'a2')]
    snapshots = [world['baseline_manifest'], world['patched_manifest']]
    with CampaignLedger(world['tmp'] / 'retry.sqlite') as ledger:
        first = ledger.initialize_attempts(entries, snapshots=snapshots)
        for changed in (None, [*snapshots, extra]):
            with pytest.raises(ValueError, match='snapshot set must match'):
                ledger.initialize_attempts(entries, snapshots=changed)
        assert set(ledger.snapshots()) == {manifest['snapshot_id'] for manifest in snapshots}
        assert ledger.initialize_attempts(entries, snapshots=snapshots) == first
    with CampaignLedger(world['tmp'] / 'v2.sqlite') as ledger:  # Schema 2 retries are unchanged.
        first = ledger.initialize_attempts(entries)
        assert ledger.initialize_attempts(entries) == first


def test_controller_init_cli_reports_snapshot_errors_as_json(world, capsys):
    baseline = world['tmp'] / 'baseline.json'
    baseline.write_text(json.dumps(world['baseline'].to_dict()))
    proposals = world['tmp'] / 'proposals.json'
    missing = Candidate('missing', BackendConfig(model='pi05', checkpoint='/weights', num_steps=3), 'frames',
                        source_ref=world['base'], patch_ref='sha256:' + 'e' * 64)
    proposals.write_text(json.dumps([missing.to_dict()]))
    code = controller_main(['init', '--root', str(world['tmp'] / 'cli'), '--comparison-id', 'r1',
                            '--baseline', str(baseline), '--proposals', str(proposals),
                            '--workload', str(world['workload']), '--output-root', str(world['tmp'] / 'out'),
                            '--snapshot-store', str(world['store'])])
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ''
    error = json.loads(captured.err)
    assert error['error'] == 'SnapshotError' and 'not materialized' in error['message']
    _assert_nothing_written(world, 'cli')
    code = controller_main(['init', '--root', str(world['tmp'] / 'cli'), '--comparison-id', 'r1',
                            '--baseline', str(baseline), '--workload', str(world['workload']),
                            '--output-root', str(world['tmp'] / 'out'), '--snapshot-store', str(world['store'])])
    assert code is None and json.loads(capsys.readouterr().out)['schema_version'] == 3


def test_snapshot_initialization_rolls_back_on_insert_failure(world):
    path = world['tmp'] / 'atomic.sqlite'
    with CampaignLedger(path) as ledger:
        ledger._connection().execute('''CREATE TRIGGER fail_second AFTER INSERT ON attempts
            WHEN NEW.attempt_id = 'a2' BEGIN SELECT RAISE(ABORT, 'injected'); END''')
        entries = [(world['baseline'], 'c', 'a1'), (world['patched'], 'c', 'a2')]
        with pytest.raises(Exception, match='injected'):
            ledger.initialize_attempts(entries, snapshots=[world['baseline_manifest'], world['patched_manifest']])
        assert ledger.snapshots() == {} and ledger.attempts() == []
        with pytest.raises(ValueError, match='no snapshot'):
            ledger.initialize_attempts(entries, snapshots=[world['baseline_manifest']])


# --------------------------------------------------------------------------- batch

def test_batch_v3_runs_each_entry_from_its_own_snapshot(world):
    plan = init(world)
    queued(world)
    calls, factory_calls = [], []
    result = run_batch(plan, execute=in_process_execute(world, calls, factory_calls=factory_calls),
                       python='/explicit/python')
    assert result['complete'] is True and [item['state'] for item in result['entries']] == ['measured'] * 3
    baseline_dir = world['baseline_manifest']['source_dir']
    patched_dir = world['patched_manifest']['source_dir']
    sources = [baseline_dir, patched_dir, baseline_dir]
    assert [kwargs['cwd'] for _, kwargs in calls] == [
        str(world['tmp'] / 'results' / 'work' / entry['attempt_id']) for entry in plan['entries']]
    for (argv, kwargs), entry, source in zip(calls, plan['entries'], sources):
        assert argv[:4] == ['/explicit/python', '-B', '-m', 'flash_rt.vla.autotune.worker']
        assert kwargs['check'] is False and kwargs['stderr'] == subprocess.STDOUT
        assert os.path.isdir(kwargs['cwd']) and not os.listdir(kwargs['cwd'])
        env = kwargs['env']
        assert env['PYTHONPATH'] == source
        assert (env['PYTHONSAFEPATH'], env['PYTHONNOUSERSITE'], env['PYTHONDONTWRITEBYTECODE']) == ('1', '1', '1')
        assert not {'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE'} & set(env)
        source_manifest = Path(argv[argv.index('--source-manifest') + 1])
        assert source_manifest == world['tmp'] / 'results' / 'control' / f'{entry["attempt_id"]}.source.json'
        assert json.loads(source_manifest.read_text()) == snap.load_manifest(world['store'], entry['snapshot_id'])
    assert factory_calls == ['base', 'fast', 'base']
    records = [json.loads(Path(item['result_path']).read_text()) for item in result['entries']]
    assert [record['schema_version'] for record in records] == [3, 3, 3]
    assert [record['source_identity']['source_dir'] for record in records] == [baseline_dir, patched_dir, baseline_dir]
    summary = ingest_campaign(world['tmp'] / 'control', world['tmp'] / 'results', job_finished=True)
    assert summary['winner'] == 'fast' and [item['state'] for item in summary['attempts']] == ['validated'] * 3


def test_batch_v3_uses_plan_runtime_python_and_extra_pythonpath(world, tmp_path):
    (tmp_path / 'site').mkdir()
    plan = init(world, python='/env/bin/python', extra_pythonpath=[str(tmp_path / 'site')])
    queued(world)
    calls = []
    run_batch(plan, execute=in_process_execute(world, calls), python='/batch/python')
    argv, kwargs = calls[0]
    assert argv[0] == '/env/bin/python'
    assert kwargs['env']['PYTHONPATH'] == os.pathsep.join([world['baseline_manifest']['source_dir'],
                                                           str(tmp_path / 'site')])


def test_batch_v3_attempt_writes_into_cwd_never_dirty_the_shared_snapshot(world):
    # Runtimes that drop files into their working directory (e.g. Ascend's
    # fusion_result.json) must not fail the bookend, which reuses the baseline snapshot.
    plan = init(world)
    queued(world)
    calls = []
    inner = in_process_execute(world, calls)

    def execute(argv, **kwargs):
        completed = inner(argv, **kwargs)
        (Path(kwargs['cwd']) / 'fusion_result.json').write_text('{}')
        return completed
    result = run_batch(plan, execute=execute, python='/explicit/python')
    assert [item['state'] for item in result['entries']] == ['measured'] * 3
    for manifest in (world['baseline_manifest'], world['patched_manifest']):
        snap.verify_source(manifest, import_root=manifest['source_dir'])
    assert sorted(path.name for path in (world['tmp'] / 'results' / 'work').iterdir()) == sorted(
        entry['attempt_id'] for entry in plan['entries'])


def _delete_manifest(world, plan):
    Path(plan['snapshots'][world['patched_manifest']['snapshot_id']]['manifest_path']).unlink()


def _stray_file(world, plan):
    (Path(world['patched_manifest']['source_dir']) / 'flash_rt' / 'sneaky.py').write_text('X = 1\n')


def _tracked_edit(world, plan):
    (Path(world['patched_manifest']['source_dir']) / 'flash_rt' / 'model.py').write_text('STEPS = 1\n')


def _drifted_manifest(world, plan):
    path = Path(plan['snapshots'][world['patched_manifest']['snapshot_id']]['manifest_path'])
    record = json.loads(path.read_text())
    record['native']['fingerprint'] = 'rebuilt-later'
    path.write_text(json.dumps(record))


def _wrong_entry_snapshot(world, plan):
    plan['entries'][1]['snapshot_id'] = world['baseline_manifest']['snapshot_id']


@pytest.mark.parametrize('break_plan', [_delete_manifest, _stray_file, _tracked_edit, _drifted_manifest,
                                        _wrong_entry_snapshot])
def test_batch_v3_preflight_failure_runs_nothing(world, break_plan):
    plan = init(world)
    break_plan(world, plan)
    calls = []
    with pytest.raises(ValueError):
        run_batch(plan, execute=lambda argv, **kwargs: calls.append(argv))
    assert calls == []
    assert not (world['tmp'] / 'results').exists()


def test_real_subprocess_workers_import_only_their_own_snapshot(world, tmp_path, monkeypatch):
    # A third FlashRT (with the same stub worker) in the launch cwd and on the
    # inherited PYTHONPATH must never be imported by any attempt.
    third = tmp_path / 'third'
    (third / 'flash_rt' / 'vla' / 'autotune').mkdir(parents=True)
    for relpath in ('flash_rt/__init__.py', 'flash_rt/vla/__init__.py', 'flash_rt/vla/autotune/__init__.py'):
        (third / relpath).write_text('')
    (third / 'flash_rt' / 'vla' / 'autotune' / 'worker.py').write_text(STUB_WORKER)
    monkeypatch.chdir(third)
    monkeypatch.setenv('PYTHONPATH', str(third))
    plan = init(world)
    result = run_batch(plan, execute=subprocess.run, python=sys.executable)
    assert result['complete'] is True
    assert [item['returncode'] for item in result['entries']] == [0, 1, 1]
    assert [item['state'] for item in result['entries']] == ['measured', 'failed', 'failed']
    for item, entry in zip(result['entries'], plan['entries']):
        record = json.loads(Path(item['result_path']).read_text())
        source = world['baseline_manifest' if entry['role'] != 'candidate' else 'patched_manifest']['source_dir']
        expected = os.path.join(source, 'flash_rt', 'vla', 'autotune', 'worker.py')
        assert record['stub_file'] == expected
        assert record['stub_flash_rt'] == os.path.join(source, 'flash_rt', '__init__.py')
        assert record['stub_cwd'] == os.path.realpath(world['tmp'] / 'results' / 'work' / entry['attempt_id'])
        assert record['stub_pythonpath'] == source
        if record['state'] == 'failed':
            assert record['failure']['message'] == expected
    # -B / PYTHONDONTWRITEBYTECODE kept the snapshots verifiably clean.
    for manifest in (world['baseline_manifest'], world['patched_manifest']):
        snap.verify_source(manifest, import_root=manifest['source_dir'])


# --------------------------------------------------------------------------- worker

def test_worker_identity_mismatch_fails_before_the_backend_factory(world):
    calls = []
    candidate = world['patched']
    for manifest, root in ((world['patched_manifest'], world['repo']),
                           (world['patched_manifest'], world['baseline_manifest']['source_dir']),
                           (world['baseline_manifest'], world['baseline_manifest']['source_dir'])):
        output = world['tmp'] / f'out-{len(calls)}-{manifest["snapshot_id"]}-{Path(root).name}'
        record = run_experiment(candidate, NPZWorkload.from_file(world['workload']), output,
                                attempt_id='a1', comparison_id='c1', campaign_instance_id='i1', repeats=2,
                                backend_factory=lambda config: calls.append(config) or Backend(config),
                                device_info={'job_id': '123'}, profile=False,
                                expected_source=manifest, source_import_root=root)
        assert record['state'] == 'failed' and record['failure']['kind'] == 'source_identity'
        assert record['schema_version'] == 3 and 'source_identity' not in record
        validate_result(record)
    assert calls == []


def test_worker_records_identity_and_measures_from_the_verified_snapshot(world):
    loaded = []
    record = run_experiment(world['patched'], NPZWorkload.from_file(world['workload']), world['tmp'] / 'ok',
                            attempt_id='a1', comparison_id='c1', campaign_instance_id='i1', repeats=2,
                            backend_factory=lambda config: loaded.append(config) or Backend(config),
                            device_info={'job_id': '123'}, profile=False,
                            expected_source=world['patched_manifest'],
                            source_import_root=world['patched_manifest']['source_dir'])
    assert record['state'] == 'measured' and record['schema_version'] == 3 and len(loaded) == 1
    identity = record['source_identity']
    assert identity['head_tree'] == world['patched_manifest']['snapshot_tree']
    assert identity['patch_ref'] == world['patched'].patch_ref
    assert identity['head_parent'] == world['base']
    validate_result(record)


def test_worker_without_expected_source_keeps_schema_2(world):
    record = run_experiment(world['patched'], NPZWorkload.from_file(world['workload']), world['tmp'] / 'v2',
                            attempt_id='a1', comparison_id='c1', campaign_instance_id='i1', repeats=2,
                            backend_factory=Backend, device_info={'job_id': '123'}, profile=False)
    assert record['schema_version'] == 2 and 'source_identity' not in record and record['state'] == 'measured'


def test_dataflow_error_no_longer_fails_a_good_measurement(tmp_path):
    class Pipeline:  # Lacks encoder_seq_len: describe_dataflow raises AttributeError.
        chunk_size = 2

    class Pi05Backend(Backend):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.frontend = type('Frontend', (), {'pipeline': Pipeline()})()
    np.savez(tmp_path / 'inputs.npz', noise=np.ones((2, 3), np.float32), state=np.zeros(3))
    (tmp_path / 'workload.json').write_text(json.dumps({
        'schema_version': 1, 'workload_id': 'frames', 'model': 'pi05',
        'requests': [{'request_id': 'frame0', 'npz_path': 'inputs.npz', 'fields': {'state': 'state'},
                      'noise_key': 'noise'}]}))
    candidate = Candidate('base', BackendConfig(model='pi05', checkpoint='/weights', num_steps=2), 'frames')
    record = run_experiment(candidate, NPZWorkload.from_file(tmp_path / 'workload.json'), tmp_path / 'out',
                            attempt_id='a1', comparison_id='c1', campaign_instance_id='i1', repeats=2,
                            backend_factory=Pi05Backend, device_info={'job_id': '123'}, profile=False)
    assert record['state'] == 'measured' and record['dataflow'] is None
    assert record['dataflow_failure'].startswith('AttributeError')
    validate_result(record)
    holder = {}
    _describe_dataflow(holder, object())
    assert holder == {'dataflow': None}


# --------------------------------------------------------------------------- ledger / ingest

def _bundles(world, plan):
    calls = []
    run_batch(plan, execute=in_process_execute(world, calls), python='/explicit/python')


@pytest.mark.parametrize('field,value', [('head_tree', '0' * 40), ('harness_tree', '1' * 40),
                                         ('patch_ref', 'sha256:' + '2' * 64), ('head_commit', '3' * 40),
                                         ('extensions', {'flash_rt/k.so': '4' * 64}),
                                         ('source_dir', '/elsewhere')])
def test_ingest_rejects_a_foreign_source_identity_and_leaves_attempts_unchanged(world, field, value):
    plan = init(world)
    queued(world)
    _bundles(world, plan)
    path = world['tmp'] / 'results' / plan['entries'][1]['attempt_id'] / 'result.json'
    record = json.loads(path.read_text())
    record['source_identity'][field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='source_identity|snapshot'):
        ingest_campaign(world['tmp'] / 'control', world['tmp'] / 'results', job_finished=True)
    with CampaignLedger(world['tmp'] / 'control' / 'ledger.sqlite') as ledger:
        assert [item['state'] for item in ledger.attempts()] == ['queued'] * 3
        assert all(item['result'] is None for item in ledger.attempts())


def test_snapshot_ledger_rejects_schema_2_results_and_v2_ledgers_reject_v3(world):
    plan = init(world)
    queued(world)
    _bundles(world, plan)
    path = world['tmp'] / 'results' / plan['entries'][0]['attempt_id'] / 'result.json'
    record = json.loads(path.read_text())
    downgraded = {key: value for key, value in record.items() if key != 'source_identity'}
    downgraded['schema_version'] = 2
    with CampaignLedger(world['tmp'] / 'control' / 'ledger.sqlite') as ledger:
        with pytest.raises(ValueError, match='schema_version 3'):
            ledger.validate_execution_identity(downgraded)
    v2 = initialize_campaign(world['tmp'] / 'v2', comparison_id='r1', baseline=world['baseline'],
                             candidates=[world['patched']], workload_path=world['workload'],
                             output_root=world['tmp'] / 'v2-results', warmup=0, repeats=2)
    queued(world, 'v2')
    with CampaignLedger(world['tmp'] / 'v2' / 'ledger.sqlite') as ledger:
        foreign = {**record, 'campaign_instance_id': v2['campaign_instance_id'],
                   'attempt_id': v2['entries'][0]['attempt_id']}
        with pytest.raises(ValueError, match='registered none'):
            ledger.validate_execution_identity(foreign)


def test_ingest_refuses_a_plan_whose_snapshots_drifted_from_the_ledger(world):
    plan = init(world)
    queued(world)
    identifier = world['patched_manifest']['snapshot_id']
    plan['snapshots'][identifier]['snapshot_tree'] = '5' * 40
    (world['tmp'] / 'control' / 'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Plan snapshots'):
        ingest_campaign(world['tmp'] / 'control', world['tmp'] / 'results', job_finished=True)
