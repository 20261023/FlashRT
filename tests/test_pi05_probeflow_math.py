"""Analytic oracles for ProbeFlow; runnable without CUDA, Torch or pytest."""
import unittest

import numpy as np

from flash_rt.models.pi05.probeflow import (
    ProbeFlowConfig, sample_probeflow, select_schedule,
)


class ProbeFlowMathTests(unittest.TestCase):
    def test_near_endpoint_horizons_keep_positive_times_and_exact_runtime_keys(self):
        for horizon in [1. - 1e-13, float(np.nextafter(1., 0.))]:
            with self.subTest(horizon=horizon):
                config = ProbeFlowConfig(probe_horizon=horizon)
                times = config.timesteps()
                self.assertTrue(all(0 < t <= 1 for t in times))
                requested = {1., 1. - horizon}
                for count in config.schedules():
                    if count > 2:
                        requested.update(1. - i / count for i in range(1, count))
                self.assertTrue(all(t in times for t in requested))

    def test_two_probe_branch_reuses_both_evaluations_and_reverses_model_time(self):
        queries = []
        noise = np.array([1., 2.])

        def velocity(x, t):
            queries.append((x.copy(), t))
            return np.array([-2., 1.])

        actions, metadata = sample_probeflow(
            noise, velocity, ProbeFlowConfig(probe_horizon=0.3))
        np.testing.assert_allclose(actions, [3., 1.])
        np.testing.assert_array_equal(noise, [1., 2.])
        np.testing.assert_allclose(queries[1][0], [1.6, 1.7])
        np.testing.assert_allclose([q[1] for q in queries], [1., 0.7])
        self.assertEqual(metadata['scheduled_steps'], 2)
        self.assertEqual(metadata['nfe'], 2)

    def test_dense_branch_restarts_original_noise_and_discards_probe_velocity(self):
        times = []

        def velocity(x, t):
            times.append(t)
            return -np.array([1., 4. * (1. - t)])

        actions, metadata = sample_probeflow(
            np.zeros(2), velocity, ProbeFlowConfig(epsilon=0.1, max_steps=4))
        # Four uniform Euler increments: [1,0], [1,1], [1,2], [1,3], each /4.
        np.testing.assert_allclose(actions, [1., 1.5])
        np.testing.assert_allclose(times, [1., 0.5, 0.75, 0.5, 0.25])
        self.assertEqual(metadata['scheduled_steps'], 4)
        self.assertEqual(metadata['nfe'], 5)

    def test_router_counts_real_evaluations_separately_from_selected_steps(self):
        config = ProbeFlowConfig()
        for cosine, expected in [(1., 2), (0.99, 4), (0.98, 6), (-1., 20),
                                 (1.00000001, 2)]:
            with self.subTest(cosine=cosine):
                self.assertEqual(select_schedule(cosine, config), expected)
        self.assertEqual(select_schedule(0.9, ProbeFlowConfig(epsilon=1e-320)), 20)

    def test_odd_maximum_is_reachable_and_all_runtime_times_are_precomputed(self):
        config = ProbeFlowConfig(max_steps=5, probe_horizon=0.3)
        self.assertEqual(config.schedules(), (2, 4, 5))
        np.testing.assert_allclose(config.timesteps(),
                                   [1., 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.25, 0.2])

    def test_zero_velocity_routes_conservatively_and_preserves_state(self):
        result, metadata = sample_probeflow(
            np.array([1., 2.]), lambda x, t: np.zeros(2), ProbeFlowConfig(max_steps=4))
        np.testing.assert_array_equal(result, [1., 2.])
        self.assertEqual(metadata['scheduled_steps'], 4)
        self.assertEqual(metadata['nfe'], 5)
        self.assertEqual(metadata['cosine'], 0.)

    def test_schedule_is_recomputed_for_each_call(self):
        config = ProbeFlowConfig(epsilon=0.1, max_steps=4)
        _, first = sample_probeflow(np.zeros(2), lambda x, t: -np.ones(2), config)
        _, second = sample_probeflow(
            np.zeros(2), lambda x, t: -np.array([1., 4. * (1. - t)]), config)
        self.assertEqual((first['scheduled_steps'], second['scheduled_steps']), (2, 4))

    def test_invalid_configuration_is_rejected(self):
        cases = [{'epsilon': x} for x in [0., -1., np.nan, np.inf, True]]
        cases += [{'probe_horizon': x} for x in [0., 1., -1., np.nan, np.inf, True]]
        cases += [{'max_steps': x} for x in [0, 1, 2.5, True, 65]]
        cases += [{'step_increment': x} for x in [0, -2, 1.5, True]]
        cases += [{'min_steps': 3}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                ProbeFlowConfig(**kwargs)

    def test_nonfinite_scores_noise_and_velocity_raise(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    select_schedule(value, ProbeFlowConfig())
                with self.assertRaises(ValueError):
                    sample_probeflow(np.array([value]), lambda x, t: np.ones(1))
                with self.assertRaises(ValueError):
                    sample_probeflow(np.zeros(1), lambda x, t: np.array([value]))
        with self.assertRaises(ValueError):
            sample_probeflow(np.zeros(2), lambda x, t: np.zeros(3))


if __name__ == '__main__':
    unittest.main()
