import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples/vla_autotune'


@pytest.fixture
def profile():
    spec = importlib.util.spec_from_file_location('fast_residual_profile', EXAMPLES / 'fast_profile/profile_fastwam.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_timeline_union_does_not_double_count_overlapping_streams(profile):
    events = [{'cat': 'kernel', 'ph': 'X', 'ts': start, 'dur': dur,
               'name': name, 'args': {'device': 0}} for start, dur, name in
              [(1000, 3000, 'a'), (2000, 3000, 'b'), (7000, 1000, 'a')]]
    result = profile.summarize_kernels({'traceEvents': events})
    assert result['0']['kernel_sum_ms'] == 7.0
    assert result['0']['kernel_busy_union_ms'] == 5.0
    assert result['0']['kernel_span_ms'] == 7.0
    assert result['0']['kernel_count'] == 3


def test_missing_cuda_is_unknown_not_zero_and_bad_timestamps_reject(profile):
    with pytest.raises(ValueError, match='CUDA'):
        profile.summarize_kernels({'traceEvents': [{'cat': 'cpu_op', 'ph': 'X', 'ts': 1, 'dur': 5}]})
    with pytest.raises(ValueError, match='finite'):
        profile.summarize_kernels({'traceEvents': [{'cat': 'kernel', 'ph': 'X', 'ts': 1,
                                                   'dur': float('nan'), 'name': 'bad'}]})


def test_linear_ledger_counts_real_repetitions_without_claiming_hbm_traffic(profile):
    result = profile.linear_ledger({(2, 3, 4, 'torch.bfloat16'): 5})
    assert result['total_linear_flops'] == 240
    assert result['logical_weight_bytes'] == 120
    assert result['linear_calls'] == 5
    assert result['hbm_bytes'] is None


def test_cli_help_and_allocation_guard_do_not_load_models(tmp_path):
    script = EXAMPLES / 'fast_profile/profile_fastwam.py'
    runner = EXAMPLES / 'validate_fastwam.py'
    env = {key: value for key, value in os.environ.items() if not key.startswith('SLURM_')}
    help_result = subprocess.run([sys.executable, str(script), '--help'], env=env, text=True, capture_output=True)
    assert help_result.returncode == 0
    result = subprocess.run([sys.executable, str(script), '--runner', str(runner), '--run',
                              '--repeats', '1', '--context', '/missing', '--output', str(tmp_path / 'unused')],
                             env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'SLURM_JOB_ID' in result.stderr
    assert not (tmp_path / 'unused').exists()


def test_diagnostic_marker_is_attached_before_publishing_result_not_model_config(profile):
    original = {'status': 'passed_execution_checks', 'latency_scope': 'whole_infer'}
    tagged = profile.tag_result(Path('/new/result.json'), original)
    assert tagged['diagnostic_only'] is True
    assert tagged['matched_engine_speedup_eligible'] is False
    assert tagged['latency_scope'] == 'instrumented_profiler_request_not_an_unprofiled_latency_benchmark'
    assert original == {'status': 'passed_execution_checks', 'latency_scope': 'whole_infer'}
    assert profile.tag_result(Path('/new/model_config_resolved.json'), {'architecture': 1}) == {'architecture': 1}
