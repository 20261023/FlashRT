"""CPU-only validation-runner contracts, not simulated model validation."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'examples/vla_autotune/validate_hyvla.py'


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location('hyvla_validation_runner', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def arrays():
    return {
        **{name: np.full((6, 480, 640, 3), value, np.uint8)
           for name, value in [('top_head', 11), ('hand_left', 22), ('hand_right', 33)]},
        'state': np.arange(20, dtype=np.float32) / 32,
        'noise': np.ones((40, 32), np.float32),
        'prompt': np.array('pick up the red block'),
        'expected_actions32': np.zeros((40, 32), np.float32),
        'expected_actions20': np.zeros((40, 20), np.float32),
    }


def test_help_runs_without_torch_or_a_slurm_job():
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--help'],
                            text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    assert '--compare' in result.stdout and '--sampler' in result.stdout


def test_import_does_not_import_torch():
    code = (
        'import importlib.util, sys; '
        f'spec=importlib.util.spec_from_file_location("runner", {str(SCRIPT)!r}); '
        'm=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); '
        'assert "torch" not in sys.modules'
    )
    result = subprocess.run([sys.executable, '-B', '-c', code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_real_shaped_npz_fields_are_used_without_loading_extra_pickles(runner, arrays, tmp_path):
    path = tmp_path / 'baseline.npz'
    np.savez(path, **arrays, irrelevant_pickle=np.array([{}], dtype=object))
    loaded = runner.load_fixture(path)
    assert loaded['noise'].shape == (40, 32)
    assert loaded['prompt'].item() == 'pick up the red block'
    assert 'irrelevant_pickle' not in loaded


@pytest.mark.parametrize('key,value', [
    ('top_head', np.zeros((6, 480, 640, 3), np.float32)),
    ('hand_left', np.zeros((480, 640, 3), np.uint8)),
    ('state', np.zeros(32, np.float32)),
    ('noise', np.zeros((1, 40, 32), np.float32)),
    ('noise', np.full((40, 32), np.nan, np.float32)),
    ('expected_actions32', np.full((40, 32), np.inf, np.float32)),
    ('expected_actions20', np.ones((40, 20), np.float32)),
    ('prompt', np.array(['not scalar'])),
    ('prompt', np.array(123)),
])
def test_rejects_wrong_raw_contract_without_coercion(runner, arrays, key, value):
    arrays[key] = value
    with pytest.raises(ValueError, match=key if key != 'expected_actions20' else 'slice'):
        runner.validate_fixture(arrays)


def test_checkpoint_camera_names_are_mapped_in_checkpoint_order(runner, arrays):
    keys = ['observation.images.hand_right', 'observation.images.top_head',
            'observation.images.hand_left']
    obs = runner.make_observation(arrays, keys)
    assert list(obs['images']) == keys
    assert [int(image[0, 0, 0, 0]) for image in obs['images'].values()] == [33, 11, 22]
    assert obs['state'].shape == (20,)  # padding belongs to the real frontend
    assert 'noise' not in obs


@pytest.mark.parametrize('keys', [[], ['top_head'],
                         ['observation.images.top_head'] * 3,
                         ['top_head', 'hand_left', 'unknown']])
def test_camera_mapping_cannot_silently_drop_a_camera(runner, arrays, keys):
    with pytest.raises(ValueError, match='camera'):
        runner.make_observation(arrays, keys)


def test_official_padded_state_must_match_raw_state(runner, arrays):
    arrays['expected_state'] = np.pad(arrays['state'], (0, 12))[None]
    runner.validate_fixture(arrays)
    arrays['expected_state'][0, -1] = 1
    with pytest.raises(ValueError, match='expected_state'):
        runner.validate_fixture(arrays)


@pytest.mark.parametrize('env', [
    {}, {'SLURM_JOB_ID': '123'},
    {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': 'batch'},
    {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': 'extern'},
    {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': '0'},
    {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': '0', 'CUDA_VISIBLE_DEVICES': '-1'},
])
def test_refuses_login_batch_and_non_gpu_steps_before_torch_import(runner, env):
    with pytest.raises(RuntimeError, match='Slurm|CUDA_VISIBLE_DEVICES'):
        runner.require_slurm_step(env)


def test_accepts_allocated_gpu_step_environment(runner):
    runner.require_slurm_step({'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': '0',
                              'CUDA_VISIBLE_DEVICES': '0'})


def test_missing_slurm_cli_never_attempts_model_or_torch_load(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
    result = subprocess.run([
        sys.executable, '-B', str(SCRIPT), '--checkpoint', '/does-not-exist',
        '--fixture', '/does-not-exist.npz', '--output', str(tmp_path / 'unused'),
        '--sampler', 'euler', '--num-steps', '2',
    ], text=True, capture_output=True, env=env)
    assert result.returncode != 0
    assert 'allocated Slurm' in result.stderr
    assert not (tmp_path / 'unused').exists()


def test_same_action_gate_preserves_old_absolute_relative_and_cosine_thresholds(runner):
    reference = np.zeros((40, 32), np.float32)
    actual = reference.copy()
    actual[0, 0] = 0.02
    metrics = runner.difference(actual, reference, atol=0.01, rtol=0.01)
    assert metrics['outside_tolerance'] == 1
    assert metrics['max_abs'] == pytest.approx(0.02)
    assert not metrics['allclose']
    equal = runner.difference(reference, reference, atol=0, rtol=0)
    assert equal['exact'] and equal['cosine'] == 1.0


def test_official_oracle_difference_is_not_a_runtime_accuracy_pass(runner, arrays):
    actual = np.full((40, 32), 0.1, np.float32)
    result = runner.oracle_comparison(actual, arrays, sampler='euler', num_steps=10)
    assert result['eligible_same_schedule']
    assert not result['same_action_gate_passed']
    assert result['actions32']['outside_tolerance'] == 1280
    short = runner.oracle_comparison(actual, arrays, sampler='euler', num_steps=2)
    assert not short['eligible_same_schedule']
    assert short['same_action_gate_passed'] is None


def test_each_request_gets_fresh_noise_and_warmup_is_excluded(runner):
    initial = np.ones((40, 32), np.float32)
    received = []
    ticks = iter([1_000_000, 3_000_000, 5_000_000, 8_000_000])

    def infer(noise):
        received.append(noise)
        output = noise.copy()
        noise.fill(99)
        return {'actions': output, 'sampler_metadata': {
            'sampler': 'euler', 'nfe': 2, 'scheduled_steps': 2, 'timesteps': [1., .5]},
            'backend_metadata': {'model': 'hyvla', 'precision': 'bf16',
                                 'action_domain': 'normalized-padded-model-actions'}}

    result = runner.measure_requests(infer, lambda: None, initial,
                                     warmup=2, repeats=2, clock=lambda: next(ticks))
    np.testing.assert_array_equal(result['latency_ms'], [2., 3.])
    assert len(received) == 4
    assert len({id(x) for x in received}) == 4
    np.testing.assert_array_equal(initial, np.ones((40, 32), np.float32))
    assert result['fixed_noise_replay']['exact']
    assert result['replay_actions32'].shape == (2, 40, 32)


def test_nonidentical_replays_are_reported_not_hidden(runner):
    index = iter(range(4))

    def infer(_noise):
        return {'actions': np.full((40, 32), next(index), np.float32),
                'sampler_metadata': {'sampler': 'euler', 'nfe': 2, 'scheduled_steps': 2}}

    result = runner.measure_requests(infer, lambda: None, np.zeros((40, 32), np.float32),
                                     warmup=1, repeats=2)
    assert not result['fixed_noise_replay']['exact']
    assert result['fixed_noise_replay']['max_abs'] == 1


def test_mutable_backend_output_is_snapshotted_before_next_replay(runner):
    scratch = np.zeros((40, 32), np.float32)

    def infer(_noise):
        scratch[:] += 1
        return {'actions': scratch, 'sampler_metadata': {
            'sampler': 'euler', 'nfe': 2, 'scheduled_steps': 2}}

    result = runner.measure_requests(infer, lambda: None, np.zeros((40, 32), np.float32),
                                     warmup=1, repeats=2)
    assert not result['fixed_noise_replay']['exact']
    assert result['actions32'][0, 0] == 2
    assert result['replay_actions32'][1, 0, 0] == 3


def test_metadata_rejects_incorrect_nfe_and_unrequested_schedule(runner):
    runner.validate_sampler_metadata({'sampler': 'probeflow', 'scheduled_steps': 2,
                                      'nfe': 2, 'timesteps': [1., .5]},
                                     sampler='probeflow', num_steps=None, max_steps=2)
    with pytest.raises(ValueError, match='nfe'):
        runner.validate_sampler_metadata({'sampler': 'probeflow', 'scheduled_steps': 4,
                                          'nfe': 4, 'timesteps': [1., .5, .75, .5, .25]},
                                         sampler='probeflow', num_steps=None, max_steps=4)
    with pytest.raises(ValueError, match='steps'):
        runner.validate_sampler_metadata({'sampler': 'euler', 'scheduled_steps': 10,
                                          'nfe': 10, 'timesteps': [1.] * 10},
                                         sampler='euler', num_steps=2, max_steps=None)


def _saved_run(folder, *, value=0, fixture='same.npz'):
    folder.mkdir()
    actions = np.full((40, 32), value, np.float32)
    np.savez(folder / 'actions.npz', actions32=actions, actions20=actions[:, :20],
             noise=np.ones((40, 32), np.float32), input_state=np.zeros(20, np.float32),
             input_prompt=np.array('same task'))
    record = {'schema_version': 1, 'runtime_checks_passed': True,
              'profile': 'flashrt_semantic_bf16_rope_qknorm_sdpa',
              'fixture': {'path': fixture, 'size_bytes': 10, 'mtime_ns': 20},
              'checkpoint': {'path': '/same/checkpoint', 'weights_size_bytes': 100,
                             'weights_mtime_ns': 200},
              'model_sources': {'frontend': {'path': '/same/frontend.py', 'mtime_ns': 20},
                                'pipeline': {'path': '/same/pipeline.py', 'mtime_ns': 30}},
              'config': {'sampler': 'euler', 'num_steps': 2, 'max_steps': None,
                         'use_graph': True},
              'sampler_metadata': [{'sampler': 'euler', 'nfe': 2, 'scheduled_steps': 2}]}
    (folder / 'result.json').write_text(json.dumps(record))
    return folder / 'actions.npz'


def test_cpu_compare_cli_preserves_threshold_and_nonzero_failure_exit(tmp_path):
    left = _saved_run(tmp_path / 'left')
    right = _saved_run(tmp_path / 'right', value=0.1)
    output = tmp_path / 'comparison'
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--compare', str(left),
                             str(right), '--output', str(output), '--atol', '0', '--rtol', '0'],
                            text=True, capture_output=True)
    assert result.returncode == 1, result.stderr
    record = json.loads((output / 'result.json').read_text())
    assert record['comparison']['actions32']['outside_tolerance'] == 1280
    assert record['status'] == 'comparison_mismatch'


def test_comparison_rejects_different_inputs_even_if_actions_match(runner, tmp_path):
    left = _saved_run(tmp_path / 'left')
    right = _saved_run(tmp_path / 'right', fixture='different.npz')
    with pytest.raises(ValueError, match='fixture'):
        runner.compare_saved_runs(left, right, atol=0, rtol=0, exact=False)


def test_output_validation_never_silently_slices_or_casts(runner):
    with pytest.raises(ValueError, match='actions'):
        runner.validate_actions(np.zeros((1, 40, 32), np.float32))
    with pytest.raises(ValueError, match='actions'):
        runner.validate_actions(np.zeros((40, 32), np.float64))


def test_graph_eager_check_uses_one_frontend_and_restores_graph_switch(runner):
    frontend = SimpleNamespace(use_cuda_graph=True)

    def infer(_obs, *, noise):
        # Distinguishable outputs prove that both execution modes were sampled.
        value = 0 if frontend.use_cuda_graph else 0.1
        return {'actions': np.full((40, 32), value, np.float32),
                'sampler_metadata': {'sampler': 'euler', 'scheduled_steps': 2, 'nfe': 2}}

    backend = SimpleNamespace(frontend=frontend, infer=infer)
    report, outputs = runner.check_graph_eager(
        backend, {}, np.zeros((40, 32), np.float32), lambda: None,
        free_bytes=20 * 2**30, min_free_gib=12, atol=1e-6, rtol=1e-6)
    assert not report['passed']
    assert report['comparison']['max_abs'] == pytest.approx(0.1)
    assert frontend.use_cuda_graph is True
    assert outputs['graph_actions32'][0, 0] == 0
    assert outputs['eager_actions32'][0, 0] == pytest.approx(0.1)


def test_graph_eager_check_does_not_run_when_free_memory_is_low(runner):
    report, outputs = runner.check_graph_eager(
        None, {}, np.zeros((40, 32), np.float32), lambda: None,
        free_bytes=2**30, min_free_gib=12, atol=0, rtol=0)
    assert report['status'] == 'skipped_insufficient_free_memory'
    assert report['passed'] is None and outputs == {}


def test_graph_eager_check_restores_switch_when_eager_raises(runner):
    frontend = SimpleNamespace(use_cuda_graph=True)

    def infer(_obs, *, noise):
        if not frontend.use_cuda_graph:
            raise RuntimeError('eager failed')
        return {'actions': np.zeros((40, 32), np.float32), 'sampler_metadata': {}}

    with pytest.raises(RuntimeError, match='eager failed'):
        runner.check_graph_eager(
            SimpleNamespace(frontend=frontend, infer=infer), {},
            np.zeros((40, 32), np.float32), lambda: None,
            free_bytes=20 * 2**30, min_free_gib=12, atol=0, rtol=0)
    assert frontend.use_cuda_graph is True


def test_cli_selects_real_sampler_budget_without_cross_sampler_options(runner):
    common = ['--checkpoint', '/checkpoint', '--fixture', '/fixture.npz', '--output', '/new-run']
    euler = runner.parse_args(common + ['--sampler', 'euler', '--num-steps', '2', '--eager'])
    assert euler.num_steps == 2 and euler.max_steps is None and not euler.use_graph
    adaptive = runner.parse_args(common + ['--sampler', 'probeflow'])
    assert adaptive.max_steps == 20 and adaptive.num_steps is None
    with pytest.raises(SystemExit):
        runner.parse_args(common + ['--sampler', 'probeflow', '--num-steps', '2'])
    with pytest.raises(SystemExit):
        runner.parse_args(common + ['--sampler', 'euler', '--max-steps', '2'])


def test_explicit_missing_manifest_is_rejected(runner, tmp_path):
    fixture = tmp_path / 'baseline.npz'
    fixture.write_bytes(b'just testing metadata, not loading arrays')
    with pytest.raises(ValueError, match='manifest'):
        runner.fixture_record(fixture, fixture.with_name('manifest.json'))


def test_compare_recognizes_euler2_vs_half_probe_only_with_matched_nfe(runner, tmp_path):
    left = _saved_run(tmp_path / 'euler')
    right = _saved_run(tmp_path / 'adaptive')
    record_path = right.with_name('result.json')
    record = json.loads(record_path.read_text())
    record['config'].update(sampler='probeflow', num_steps=None, max_steps=2, probe_horizon=.5)
    record['sampler_metadata'] = [{'sampler': 'probeflow', 'scheduled_steps': 2, 'nfe': 2}]
    record_path.write_text(json.dumps(record))
    report = runner.compare_saved_runs(left, right, atol=0, rtol=0, exact=True)
    assert report['passed']
    assert report['comparison_kind'] == 'same_head_euler2_vs_probeflow2'
    record['sampler_metadata'][0]['nfe'] = 3
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='NFE'):
        runner.compare_saved_runs(left, right, atol=0, rtol=0, exact=True)
    record['sampler_metadata'] = []
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='NFE'):
        runner.compare_saved_runs(left, right, atol=0, rtol=0, exact=True)


def test_same_head_comparison_rejects_changed_source_metadata(runner, tmp_path):
    left = _saved_run(tmp_path / 'left')
    right = _saved_run(tmp_path / 'right')
    record_path = right.with_name('result.json')
    record = json.loads(record_path.read_text())
    record['model_sources']['pipeline']['mtime_ns'] = 40
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='source'):
        runner.compare_saved_runs(left, right, atol=0, rtol=0, exact=True)


def test_official_profile_cli_builds_requested_real_backend_config(runner):
    args = runner.parse_args(['--checkpoint', '/checkpoint', '--fixture', '/fixture.npz',
                              '--output', '/new-run', '--sampler', 'euler', '--num-steps', '10',
                              '--numerical-profile', 'official_local_fp32'])
    config = runner.build_backend_config(args)
    assert config.options['numerical_profile'] == 'official_local_fp32'
    assert config.num_steps == 10 and config.precision == 'bf16'


def test_compare_recognizes_same_official_local_profile_without_claiming_same_integrator(runner, tmp_path):
    left = _saved_run(tmp_path / 'euler')
    right = _saved_run(tmp_path / 'pf')
    for path in (left, right):
        sidecar = path.with_name('result.json')
        record = json.loads(sidecar.read_text())
        record['profile'] = 'flashrt_official_local_fp32_hooks_not_exact'
        if path == right:
            record['config'].update(sampler='probeflow', num_steps=None, max_steps=2, probe_horizon=.5)
        sidecar.write_text(json.dumps(record))
    result = runner.compare_saved_runs(left, right, atol=0, rtol=0, exact=False)
    assert result['passed']
    assert result['comparison_kind'] == 'same_head_euler2_vs_probeflow2_different_integration_rounding'
