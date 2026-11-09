"""Warmed FastWAM CPU/CUDA diagnosis; never an uninstrumented latency result."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import importlib.util
import json
import math
from pathlib import Path


def summarize_kernels(trace):
    by_device = defaultdict(list)
    for event in trace.get('traceEvents', []):
        if event.get('ph') != 'X' or event.get('cat') != 'kernel':
            continue
        start, duration = event.get('ts'), event.get('dur')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in (start, duration)) or duration < 0:
            raise ValueError('CUDA kernel timestamps/durations must be finite and nonnegative')
        by_device[str(event.get('args', {}).get('device', 'unknown'))].append(
            (start, start + duration, event.get('name', 'unknown')))
    if not by_device:
        raise ValueError('No CUDA kernel events; profiler availability/coverage is unknown, not zero GPU time')
    result = {}
    for device, intervals in by_device.items():
        intervals.sort()
        left, right, _ = intervals[0]
        union = 0.0
        names = defaultdict(lambda: {'calls': 0, 'sum_us': 0.0})
        for start, stop, name in intervals:
            names[name]['calls'] += 1
            names[name]['sum_us'] += stop - start
            if start > right:
                union += right - left
                left, right = start, stop
            else:
                right = max(right, stop)
        union += right - left
        result[device] = {'kernel_count': len(intervals), 'kernel_sum_ms': sum(x['sum_us'] for x in names.values()) / 1000,
            'kernel_busy_union_ms': union / 1000, 'kernel_span_ms': (max(stop for _, stop, _ in intervals) - intervals[0][0]) / 1000,
            'top_kernels': sorted(({'name': name, **values} for name, values in names.items()),
                                  key=lambda row: row['sum_us'], reverse=True)[:30]}
    return result


def linear_ledger(counts):
    rows = []
    sizes = {'torch.bfloat16': 2, 'torch.float16': 2, 'torch.float32': 4, 'torch.float64': 8}
    for (m, n, k, dtype), count in sorted(counts.items()):
        if dtype not in sizes or any(type(x) is not int or x < 1 for x in (m, n, k, count)):
            raise ValueError('Unsupported Linear shape/count/dtype')
        rows.append({'M': m, 'N': n, 'K': k, 'weight_dtype': dtype, 'calls': count,
                     'linear_flops': 2*m*n*k*count, 'logical_weight_bytes': sizes[dtype]*n*k*count})
    return {'shapes': rows, 'linear_calls': sum(row['calls'] for row in rows),
            'total_linear_flops': sum(row['linear_flops'] for row in rows),
            'logical_weight_bytes': sum(row['logical_weight_bytes'] for row in rows), 'hbm_bytes': None,
            'scope': 'GEMM FMA=2 only; excludes bias/norm/attention/conv; logical weights are not HBM counters'}


def tag_result(path, payload):
    if Path(path).name != 'result.json':
        return payload
    return {**payload, 'diagnostic_only': True, 'matched_engine_speedup_eligible': False,
            'latency_scope': 'instrumented_profiler_request_not_an_unprofiled_latency_benchmark'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runner', required=True, type=Path, help='Explicit frozen validate_fastwam.py')
    args, forwarded = parser.parse_known_args(argv)
    spec = importlib.util.spec_from_file_location('profile_validation_runner', args.runner)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)  # stdlib-only module; no model imports
    helper.require_allocation()
    candidate = helper.parser().parse_args(forwarded)
    if not (candidate.run or candidate.reference) or candidate.repeats != 1:
        parser.error('Pass --run or --reference and exactly --repeats 1; this is a profile, not a benchmark')
    import cProfile
    import pstats
    import torch
    from torch import nn
    from flash_rt.models.fastwam.native_linear import FlashRTLinear

    original_timed, original_instrument, original_write = helper.timed_infer, helper.instrument_model, helper.write_json
    active = False
    calls, shapes, input_dtypes, implementations = 0, Counter(), Counter(), Counter()
    output = candidate.output.expanduser().resolve()
    captured = False
    with ExitStack() as stack:
        def instrument(model):
            counters = original_instrument(model)
            def observe(module, inputs):
                if not active:
                    return
                x = inputs[0]
                shapes[(x.numel() // x.shape[-1], module.out_features, module.in_features, str(module.weight.dtype))] += 1
                input_dtypes[str(x.dtype)] += 1
                implementations[type(module).__module__ + '.' + type(module).__name__] += 1
            for module in model.modules():
                if isinstance(module, (nn.Linear, FlashRTLinear)):
                    handle = module.register_forward_pre_hook(observe)
                    stack.callback(handle.remove)
            for owner, name, stage in ((model.mot, 'prefill_video_cache_tensor', 'video_prefill'),
                                        (model, '_denoise_action_with_video_cache', 'action_velocity')):
                original = getattr(owner, name)
                def wrapped(*a, _original=original, _stage=stage, **kw):
                    if active:
                        with torch.profiler.record_function(_stage):
                            return _original(*a, **kw)
                    return _original(*a, **kw)
                setattr(owner, name, wrapped)
                stack.callback(setattr, owner, name, original)
            return counters
        def timed(infer, counters, torch_module):
            nonlocal calls, active, captured
            calls += 1
            if calls != candidate.warmups + 2:
                return original_timed(infer, counters, torch_module)
            cpu = cProfile.Profile()
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False, with_stack=False, profile_memory=False) as profiler:
                    active = True
                    cpu.enable()
                    with torch.profiler.record_function('fastwam_request'):
                        result = original_timed(infer, counters, torch_module)
                    cpu.disable()
                    active = False
            finally:
                cpu.disable()
                active = False
            profiler.export_chrome_trace(str(output / 'trace.json'))
            (output / 'operator_cpu_table.txt').write_text(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=60))
            cpu.dump_stats(str(output / 'cpu.prof'))
            stats = pstats.Stats(cpu)
            rows = [{'file': key[0], 'line': key[1], 'function': key[2], 'primitive_calls': val[0],
                     'calls': val[1], 'self_seconds': val[2], 'cumulative_seconds': val[3]}
                    for key, val in stats.stats.items()]
            report = {'diagnostic_only': True, 'formal_latency_comparison_allowed': False,
                'profiler_overhead_present': True, 'record_shapes': False, 'with_stack': False,
                'linear_ledger': linear_ledger(shapes), 'linear_input_dtypes': dict(input_dtypes),
                'linear_implementations': dict(implementations),
                'gpu_kernel_trace': summarize_kernels(json.loads((output / 'trace.json').read_text())),
                'cpu_top_self': sorted(rows, key=lambda row: row['self_seconds'], reverse=True)[:40],
                'cpu_top_cumulative': sorted(rows, key=lambda row: row['cumulative_seconds'], reverse=True)[:40],
                'caveats': ['Kernel busy union is for instrumented execution only; gaps do not prove the unprofiled run is host-bound.',
                            'Kernel sums omit memory copies and overlap is accounted per device, not by summing streams.',
                            'cProfile records function elapsed wall time including GPU synchronization/blocking, not CPU cycle counters.',
                            'No compute/memory-bound claim without additional counters.']}
            (output / 'profile.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            captured = True
            return result
        helper.timed_infer, helper.instrument_model = timed, instrument
        helper.write_json = lambda path, payload: original_write(path, tag_result(path, payload))
        stack.callback(setattr, helper, 'timed_infer', original_timed)
        stack.callback(setattr, helper, 'instrument_model', original_instrument)
        stack.callback(setattr, helper, 'write_json', original_write)
        code = helper.main(forwarded)
    if not captured:
        raise RuntimeError('No warmed profiling request completed; retain failure artifact instead of a timing claim')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
