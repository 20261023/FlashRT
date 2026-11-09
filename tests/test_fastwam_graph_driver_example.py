import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip('torch')


def test_refresh_pairs_restore_inputs_seed_and_compare_identical_paired_requests():
    path = Path(__file__).resolve().parents[1] / 'examples/vla_autotune/fast_graph/run_velocity_graph.py'
    spec = importlib.util.spec_from_file_location('velocity_graph_driver', path)
    driver_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver_module)
    arrays = {'noise': np.zeros((1, 32, 7), np.float32), 'proprio': np.zeros((1, 8), np.float32),
              'input_image': np.zeros((1, 3, 2, 4), np.float32), 'context': np.zeros((1, 4, 5), np.float32),
              'context_mask': np.ones((1, 4), bool)}
    originals = {key: value.copy() for key, value in arrays.items()}
    args = SimpleNamespace(seed=7)
    counters = {'nfe': 0, 'video_prefill': 0}
    requests = []
    class Driver:
        enabled = True
        def begin_request(self):
            pass
        def snapshot(self):
            return {'replays': 2 if self.enabled else 0, 'captures': 0}
    graph = Driver()
    def infer():
        requests.append(({key: value.copy() for key, value in arrays.items()}, args.seed))
        counters['nfe'] += 2; counters['video_prefill'] += 1
        result = arrays['noise'] + sum(float(value.mean()) for key, value in arrays.items() if key != 'noise')
        return {'actions': result.copy()}
    checks, raw = driver_module.refresh_pairs(infer, graph, arrays, args, counters, torch,
        lambda a, b: {'bitwise_equal': bool(np.array_equal(a, b)), 'max_abs': float(abs(a-b).max())},
        baseline_actions=np.zeros((1, 32, 7), np.float32), synchronize=lambda: None)
    assert len(checks) == 6 and len(requests) == 12
    assert checks[0]['case'] == 'original'
    assert all(row['passed'] for row in checks)
    for first, second in zip(requests[::2], requests[1::2]):
        assert first[1] == second[1]
        assert all(np.array_equal(first[0][key], second[0][key]) for key in originals)
    assert args.seed == 7 and graph.enabled is True
    assert all(np.array_equal(arrays[key], value) for key, value in originals.items())
    assert len(raw) == 17
