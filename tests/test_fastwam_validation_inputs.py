"""CPU-only runner contracts; no model imports, checkpoint reads or CUDA."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'examples/vla_autotune'
SPEC = importlib.util.spec_from_file_location('validate_fastwam', ROOT / 'validate_fastwam.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RunnerContracts(unittest.TestCase):
    def test_explicit_euler10_torch_graph_and_adaptive_pf_are_not_euler2_aliases(self):
        args = RUNNER.parser().parse_args(['--run', '--backend', 'torch', '--num-steps', '10',
                                         '--use-cuda-graph', '--output', '/unused'])
        config = RUNNER.build_backend_config(args, {'model': 'fixture'})
        self.assertEqual(config.num_steps, 10)
        self.assertEqual(config.backend, 'torch')
        self.assertTrue(config.options['use_cuda_graph'])
        self.assertNotIn('native_linear_backend', config.options)
        args = RUNNER.parser().parse_args(['--run', '--backend', 'torch', '--sampler', 'probeflow',
                                         '--pf-max', '10', '--pf-epsilon', '.008', '--output', '/unused'])
        config = RUNNER.build_backend_config(args, {'model': 'fixture'})
        self.assertIsNone(config.num_steps)
        self.assertEqual(config.probeflow.max_steps, 10)

    def test_graph_instrumentation_counts_logical_callbacks_not_capture_enqueues(self):
        model = SimpleNamespace(mot=SimpleNamespace(prefill_video_cache_tensor=lambda: 'kv'),
                                _denoise_action_with_video_cache=lambda: 'eager')
        pipeline = SimpleNamespace(_evaluate_clock_velocity=lambda *args: 'raw')
        captured_head = model._denoise_action_with_video_cache
        counts = RUNNER.instrument_model(model, pipeline=pipeline)
        for _ in range(4): self.assertEqual(captured_head(), 'eager')  # cold work, not sampler NFE
        self.assertEqual(pipeline._evaluate_clock_velocity(), 'raw')
        self.assertEqual(pipeline._evaluate_clock_velocity(), 'raw')
        self.assertEqual(model.mot.prefill_video_cache_tensor(), 'kv')
        self.assertEqual(counts, {'nfe': 2, 'video_prefill': 1})

    def test_library_graph_refresh_controls_cover_variable_nfe_and_restore_inputs(self):
        arrays = {'input_image': np.zeros((1, 3, 2, 2), np.float32),
                  'context': np.zeros((1, 3, 2), np.float32),
                  'context_mask': np.ones((1, 3), bool), 'proprio': np.zeros((1, 2), np.float32),
                  'noise': np.zeros((1, 2, 1), np.float32)}
        originals = {key: value.copy() for key, value in arrays.items()}
        driver = object(); pipe = SimpleNamespace(_velocity_graph=driver)
        counts = {'nfe': 0, 'video_prefill': 0}
        def infer(obs, noise):
            nfe = 5 if np.any(noise) else 2
            counts['nfe'] += nfe; counts['video_prefill'] += 1
            value = sum(float(value.sum()) for value in obs.values()) + float(noise.sum())
            return {'actions': np.full((1, 32, 7), value, np.float32),
                    'sampler_metadata': {'nfe': nfe, 'timesteps': list(range(nfe))}, 'execution_metadata': {}}
        owner = SimpleNamespace(frontend=SimpleNamespace(pipeline=pipe), infer=infer)
        checks, raw = RUNNER.graph_refresh_checks(owner, arrays, counts, synchronize=lambda: None)
        self.assertEqual([row['case'] for row in checks], ['original', 'noise', 'proprio', 'image', 'context', 'mask'])
        self.assertTrue(all(row['passed'] for row in checks))
        self.assertTrue(all(row['input_effect_observed'] for row in checks[1:]))
        self.assertEqual(checks[1]['execution']['graph']['actual_callbacks']['nfe'], 5)
        self.assertEqual(checks[2]['execution']['eager']['actual_callbacks']['nfe'], 2)
        self.assertEqual(counts['video_prefill'], 12)
        self.assertEqual(len(raw), 17)
        self.assertIs(pipe._velocity_graph, driver)
        for key in arrays: np.testing.assert_array_equal(arrays[key], originals[key])

    def test_library_graph_refresh_failure_restores_graph_and_all_inputs(self):
        arrays = {'input_image': np.zeros((1, 3, 2, 2), np.float32),
                  'context': np.zeros((1, 3, 2), np.float32),
                  'context_mask': np.ones((1, 3), bool), 'proprio': np.zeros((1, 2), np.float32),
                  'noise': np.zeros((1, 2, 1), np.float32)}
        originals = {key: value.copy() for key, value in arrays.items()}
        driver = object(); pipe = SimpleNamespace(_velocity_graph=driver)
        def fail(obs, noise): raise RuntimeError('injected infer failure')
        owner = SimpleNamespace(frontend=SimpleNamespace(pipeline=pipe), infer=fail)
        with self.assertRaisesRegex(RuntimeError, 'infer failure'):
            RUNNER.graph_refresh_checks(owner, arrays, {'nfe': 0, 'video_prefill': 0}, synchronize=lambda: None)
        self.assertIs(pipe._velocity_graph, driver)
        for key in arrays: np.testing.assert_array_equal(arrays[key], originals[key])

    def setUp(self):
        self.source_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.source_directory.cleanup)
        self.source = Path(self.source_directory.name) / 'upstream'
        entry = self.source / 'src/fastwam/models/wan22/fastwam.py'
        entry.parent.mkdir(parents=True)
        entry.write_text('# CPU source-boundary fixture, never imported as a model.\n')
        manifest = json.loads((ROOT / 'fastwam-source/SOURCE.json').read_text())
        manifest['integrity_note'] = 'Synthetic CPU boundary fixture; never imported or used for model inference.'
        (self.source.parent / 'SOURCE.json').write_text(json.dumps(manifest))

    def test_relocated_runner_uses_explicit_resolved_config_and_preflights_it(self):
        old_here = RUNNER.HERE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'model.json'
            config.write_text((ROOT / 'fastwam-source/libero_model_resolved.json').read_text())
            RUNNER.HERE = root / 'relocated-empty-folder'
            try:
                args = RUNNER.parser().parse_args(['--run', '--context', '/context.npz',
                    '--model-config', str(config), '--output', '/new'])
                resolved = RUNNER.resolved_model_config(self.source, Path('/wan'), args.model_config)
                self.assertEqual(resolved['action_dit_config']['action_dim'], 7)
                self.assertEqual(resolved['model_id'], '/wan')
            finally:
                RUNNER.HERE = old_here

    def test_cached_cli_builds_explicit_backend_without_changing_the_workload(self):
        args = RUNNER.parser().parse_args(['--run', '--context', '/context.npz',
                                           '--native-linear-backend', 'cached', '--output', '/new'])
        config = RUNNER.build_backend_config(args, {'same': 'model'})
        self.assertEqual(config.options['native_linear_backend'], 'cached')
        self.assertEqual(config.options['model_config'], {'same': 'model'})
        self.assertEqual(config.num_steps, 2)
        self.assertEqual(config.options['sigma_shift'], 5.)

    def test_login_rejected_even_with_a_copied_job_id(self):
        for environ, hostname in [({}, 'ln01'), ({'SLURM_JOB_ID': '123'}, 'ln01'),
                ({'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'gpu001'}, 'ln02')]:
            with self.subTest(environ=environ, hostname=hostname):
                with self.assertRaises(RuntimeError):
                    RUNNER.require_allocation(environ, hostname)

    def test_compute_requires_gpu_allocation_metadata(self):
        env = {'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'gpu001'}
        with self.assertRaises(RuntimeError):
            RUNNER.require_allocation(env, 'gpu001')
        env['SLURM_JOB_GPUS'] = '0'
        self.assertEqual(RUNNER.require_allocation(env, 'gpu001')['job_id'], '123')

    def test_zero_gpu_count_is_not_a_gpu_allocation(self):
        env = {'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'gpu001', 'SLURM_GPUS_ON_NODE': '0'}
        with self.assertRaises(RuntimeError):
            RUNNER.require_allocation(env, 'gpu001')

    def test_config_resolves_checkpoint_dimensions_and_original_shift(self):
        config = RUNNER.resolved_model_config(self.source, Path('/local/wan'))
        self.assertEqual(config['proprio_dim'], 8)
        self.assertEqual(config['action_dit_config']['action_dim'], 7)
        self.assertEqual(config['action_dit_config']['hidden_dim'], 1024)
        self.assertEqual(config['action_dit_config']['num_layers'], 30)
        self.assertEqual(config['action_dit_config']['num_heads'] *
                         config['action_dit_config']['attn_head_dim'], 3072)
        self.assertEqual(config['action_scheduler']['infer_shift'], 5.0)
        self.assertFalse(config['load_text_encoder'])
        self.assertTrue(config['skip_dit_load_from_pretrain'])
        self.assertNotIn('${', str(config))

    def fixture(self):
        return [np.zeros((1, 3, 224, 448), dtype=np.float32),
                np.zeros((1, 8), dtype=np.float32),
                np.zeros((1, 128, 4096), dtype=np.float32),
                np.ones((1, 128), dtype=bool),
                np.ones((1, 32, 7), dtype=np.float32)]

    def test_preprocessed_fixture_rejects_wrong_shapes_and_invalid_values(self):
        RUNNER.validate_fixture(*self.fixture())
        for index, replacement in [(0, np.zeros((1, 3, 224, 224), np.float32)),
                (0, np.full((1, 3, 224, 448), 2., np.float32)),
                (1, np.zeros((1, 7), np.float32)),
                (2, np.zeros((1, 128, 512), np.float32)),
                (3, np.ones((1, 128), np.int64)),
                (3, np.zeros((1, 128), bool)),
                (4, np.full((1, 32, 7), np.nan, np.float32))]:
            values = self.fixture()
            values[index] = replacement
            with self.subTest(index=index, replacement=replacement.shape):
                with self.assertRaises(ValueError):
                    RUNNER.validate_fixture(*values)

    def test_action_metrics_do_not_hide_nonfinite_or_shape_mismatches(self):
        values = np.zeros((1, 32, 7), np.float32)
        self.assertEqual(RUNNER.action_metrics(values, values)['max_abs'], 0.)
        changed = values.copy()
        changed[0, 0, 0] = 2.
        metrics = RUNNER.action_metrics(values, changed)
        self.assertEqual(metrics['max_abs'], 2.)
        self.assertAlmostEqual(metrics['rmse'], (4 / 224) ** .5)
        for invalid in [np.zeros((32, 7), np.float32),
                        np.full((1, 32, 7), np.inf, np.float32)]:
            with self.assertRaises(ValueError):
                RUNNER.action_metrics(values, invalid)

    def test_compare_checks_identical_inputs_before_computing_output_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / 'left', root / 'right'
            left.mkdir()
            right.mkdir()
            report = dict(source_contract_commit=RUNNER.COMMIT, model_config={'proprio_dim': 8},
                noise_seed=3, sigma_shift=5., action_horizon=32, status='passed_execution_checks',
                task_success_evaluated=False, checkpoint='/local/model.pt',
                assets=[dict(path='/local/model.pt', bytes=123, mtime_ns=5)],
                sampler_metadata=[dict(sampler='euler', nfe=2)], execution_metadata=dict(backend='unit_test_only'))
            image, proprio, context, mask, noise = self.fixture()
            arrays = dict(input_image=image, proprio=proprio, context=context,
                          context_mask=mask, noise=noise, actions=np.zeros((1, 32, 7), np.float32))
            for target in (left, right):
                (target / 'result.json').write_text(json.dumps(report))
                np.savez(target / 'actions.npz', **arrays)
            args = SimpleNamespace(compare=[left, right], compare_atol=None, compare_rtol=0.)
            output = {}
            RUNNER.compare_results(args, output)
            self.assertTrue(output['metrics']['bitwise_equal'])
            self.assertNotIn('allclose', output['metrics'])
            arrays['noise'] = arrays['noise'] + 1
            np.savez(right / 'actions.npz', **arrays)
            with self.assertRaisesRegex(ValueError, 'noise'):
                RUNNER.compare_results(args, {})

    def test_source_manifest_rejects_wrong_commit_before_import(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            (root / 'SOURCE.json').write_text(json.dumps({'commit': 'wrong', 'source_subdirectory': 'upstream'}))
            source = root / 'upstream'
            source.mkdir()
            with self.assertRaisesRegex(ValueError, 'commit'):
                RUNNER.validate_source_manifest(source)


if __name__ == '__main__':
    unittest.main()
