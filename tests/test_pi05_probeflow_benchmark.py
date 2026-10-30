"""CPU checks for the offline recorded-episode benchmark's common contract."""
import unittest

import numpy as np

from examples.pi05_probeflow_episode_benchmark import (
    bf16_round, decode_ur5_actions, latency_statistics, model_state,
    reference_bf16_probeflow,
)
from flash_rt.models.pi05.probeflow import ProbeFlowConfig


class RecordedBenchmarkTests(unittest.TestCase):
    def test_official_quantile_precision_preserves_discrete_state_boundary(self):
        from examples.pi05_probeflow_episode_benchmark import _normalized_state
        stats = {'state': {'q01': [0.] * 6 + [3.], 'q99': [1.] * 6 + [229.]}}
        state = np.array([0.] * 6 + [2.999997615814209], np.float32)
        normalized = _normalized_state(state, stats)
        # Raw Normalize retains float64 quantiles. Rounding first to float32
        # collapses this to -1 and changes the tokenizer's discrete value.
        self.assertLess(normalized[6], -1.)
        self.assertEqual(int(np.digitize(normalized[6], [-1., 0., 1.]) - 1), -1)

    def test_sharded_gpu_residency_uses_actual_devices_not_device_property(self):
        from types import SimpleNamespace
        from examples.pi05_probeflow_episode_benchmark import gpu_parameter_bytes
        gpu_array = SimpleNamespace(size=12, dtype=np.dtype(np.float32),
                                    device=SimpleNamespace(sharding='NamedSharding'),
                                    devices=lambda: [SimpleNamespace(platform='gpu')])
        cpu_array = SimpleNamespace(size=10000, dtype=np.dtype(np.float32),
                                    devices=lambda: [SimpleNamespace(platform='cpu')])
        self.assertEqual(gpu_parameter_bytes([gpu_array, cpu_array, None]), 48)

    def test_rounding_is_nearest_even_and_representable_in_bf16(self):
        values = np.array([1.00390625, 1.01171875, -1.00390625], np.float32)
        rounded = bf16_round(values)
        np.testing.assert_array_equal(rounded, [1., 1.015625, -1.])
        self.assertFalse(np.any(rounded.view(np.uint32) & 0xffff))

    def test_ur5_model_gripper_mapping_has_documented_endpoints(self):
        np.testing.assert_allclose(model_state([1., 2., 3., 4., 5., 6., 0.085]),
                                   [1., 2., 3., 4., 5., 6., 3.], atol=3e-6)
        self.assertAlmostEqual(model_state([0.] * 7)[6], 229.)

    def test_public_decode_uses_checkpoint_quantiles_and_absolute_joints(self):
        stats = {'actions': {'q01': [-1.] * 6 + [3.],
                             'q99': [1.] * 6 + [229.]}}
        result = decode_ur5_actions(np.zeros((50, 32), np.float32),
                                    np.array([1., 2., 3., 4., 5., 6., 3.]), stats)
        self.assertEqual(result.shape, (30, 7))
        np.testing.assert_allclose(result[0, :6], [1., 2., 3., 4., 5., 6.], atol=1e-6)
        self.assertAlmostEqual(float(result[0, 6]), 0.0425, places=6)

    def test_steady_percentiles_exclude_warmup_and_capture_not_slow_valid_calls(self):
        records = [{'run_policy_ms': ms, 'exclude': exclude}
                   for ms, exclude in [(10000., True), (1., False), (2., False),
                                       (3., False), (100., False), (20000., True)]]
        result = latency_statistics(records)
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['p50_ms'], 2.5)
        self.assertAlmostEqual(result['p99_ms'], 97.09)

    def test_no_samples_or_nonfinite_latency_is_not_success(self):
        for records in [[], [{'run_policy_ms': 10., 'exclude': True}],
                        [{'run_policy_ms': float('nan'), 'exclude': False}],
                        [{'run_policy_ms': 0., 'exclude': False}]]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                latency_statistics(records)

    def test_bf16_reference_reuses_probes_and_restarts_dense_original_noise(self):
        def curved(x, t):
            return np.array([1., 4. * (1. - t)], np.float32)
        result, metadata = reference_bf16_probeflow(
            np.zeros(2, np.float32), curved, ProbeFlowConfig(epsilon=.1, max_steps=4))
        np.testing.assert_array_equal(result, [1., 1.5])
        self.assertEqual(metadata['nfe'], 5)
        np.testing.assert_allclose(metadata['timesteps'], [1., .5, .75, .5, .25])
        result, metadata = reference_bf16_probeflow(
            np.zeros(2, np.float32), lambda x, t: np.ones(2), ProbeFlowConfig())
        np.testing.assert_array_equal(result, [1., 1.])
        self.assertEqual(metadata['nfe'], 2)


if __name__ == '__main__':
    unittest.main()
