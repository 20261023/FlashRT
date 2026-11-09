"""Experimental non-integrating velocity graph pilot, with eager refresh controls."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path


def refresh_pairs(infer, driver, arrays, args, counters, torch, action_metrics, *,
                  baseline_actions, synchronize=None):
    import numpy as np
    synchronize = synchronize or torch.cuda.synchronize
    originals = {key: value.copy() for key, value in arrays.items()}
    seed = args.seed
    checks, raw = [], {}
    cases = (('original', None), ('noise', 'noise'), ('proprio', 'proprio'), ('image', 'input_image'),
             ('context', 'context'), ('mask', 'context_mask'))
    try:
        with torch.inference_mode():
            for name, field in cases:
                try:
                    if name == 'original':
                        pass
                    elif name == 'noise':
                        args.seed = seed + 1  # upstream generates its noise from this live seed
                        generator = torch.Generator(device='cpu').manual_seed(args.seed)
                        arrays[field][...] = torch.randn(arrays[field].shape, generator=generator, dtype=torch.float32).numpy()
                    elif name == 'mask':
                        valid = np.flatnonzero(arrays[field][0])
                        if len(valid) < 2:
                            raise ValueError('Mask freshness control requires at least two originally valid tokens')
                        arrays[field][0, valid[-1]] = False
                    elif name == 'image':
                        arrays[field][...] = np.clip(arrays[field] + .1, -1, 1)
                    else:
                        arrays[field][...] += .25
                    outputs, observations = {}, {}
                    for mode in ('graph', 'eager'):
                        driver.enabled = mode == 'graph'
                        driver.begin_request()
                        before = dict(counters)
                        outputs[mode] = np.asarray(infer()['actions']).copy()
                        synchronize()
                        delta = {key: counters[key] - before[key] for key in counters}
                        if delta != {'nfe': 2, 'video_prefill': 1}:
                            raise RuntimeError(f'Freshness control changed actual request work: {delta}')
                        observations[mode] = {'actual_model_calls': delta, 'velocity_graph': driver.snapshot()}
                    metric = action_metrics(outputs['graph'], outputs['eager'])
                    exact = bool(np.array_equal(outputs['graph'], outputs['eager']))
                    checks.append({'case': name, 'passed': exact, 'comparison': metric,
                                   'changed_input_field': field, 'seed': args.seed, 'execution': observations,
                                   'effect_vs_original': action_metrics(outputs['graph'], baseline_actions)})
                    raw[f'{name}_graph_actions'] = outputs['graph']
                    raw[f'{name}_eager_actions'] = outputs['eager']
                    if field is not None:
                        raw[f'{name}_changed_input'] = arrays[field].copy()
                finally:
                    args.seed = seed
                    for key, value in originals.items():
                        arrays[key][...] = value
    finally:
        driver.enabled = True
        args.seed = seed
        for key, value in originals.items():
            arrays[key][...] = value
    return checks, raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--runner', required=True, type=Path)
    args, forwarded = parser.parse_known_args(argv)
    spec = importlib.util.spec_from_file_location('velocity_graph_validation_runner', args.runner)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.require_allocation()  # before Torch/model imports
    point = helper.parser().parse_args(forwarded)
    if not (point.run or point.reference) or point.sampler != 'euler' or point.warmups < 1:
        parser.error('Pilot is Euler2 only with at least one excluded warmup, --run or --reference')
    import torch
    import numpy as np
    from velocity_graph import VelocityGraph

    output = point.output.expanduser().resolve()
    original_instrument, original_fixture = helper.instrument_model, helper.load_fixture
    original_timed, original_write = helper.timed_infer, helper.write_json
    hold = {'checks': None, 'cleanup_failure': None}
    request_index = 0
    with ExitStack() as stack:
        def fixture(live_args, torch_module):
            arrays, labels = original_fixture(live_args, torch_module)
            hold.update(arrays=arrays, live_args=live_args)
            return arrays, labels
        def instrument(model):
            original = model._denoise_action_with_video_cache
            driver = VelocityGraph(original)
            model._denoise_action_with_video_cache = driver
            hold['driver'] = driver
            stack.callback(driver.close)
            stack.callback(setattr, model, '_denoise_action_with_video_cache', original)
            hold['counters'] = original_instrument(model)
            return hold['counters']
        def timed(infer, counters, torch_module):
            nonlocal request_index
            request_index += 1
            driver = hold['driver']
            def invoke():
                driver.begin_request()
                return infer()
            result, elapsed, counted = original_timed(invoke, counters, torch_module)
            result['execution_metadata']['experimental_raw_velocity_graph'] = driver.snapshot()
            result['execution_metadata']['cuda_graph'] = True
            if 'backend_metadata' in result:
                result['backend_metadata']['experimental_execution_override'] = {
                    'raw_velocity_cuda_graph': True, 'product_default_unchanged': True}
            # All formal samples and replay are finished before extra refresh
            # controls. They are excluded from formal latency and restore arrays.
            if request_index == point.warmups + point.repeats + 2:
                checks, raw = refresh_pairs(infer, driver, hold['arrays'], hold['live_args'], counters,
                    torch_module, helper.action_metrics, baseline_actions=result['actions'])
                hold['checks'] = checks
                np.savez_compressed(output / 'refresh_checks.npz', **raw)
                (output / 'refresh_checks.json').write_text(json.dumps(checks, indent=2, allow_nan=False) + '\n')
            return result, elapsed, counted
        def publish(path, payload):
            if Path(path).name == 'result.json':
                driver = hold.get('driver')
                summary = None
                if driver is not None:
                    summary = {'total_captures': driver.total_captures, 'total_replays': driver.total_replays,
                               'capture_metadata': driver.capture_metadata,
                               'semantics': 'Raw velocity only; sampler/integration/video-prefill outside; CPU enqueues differ from encoded graph work'}
                    try:
                        driver.close()  # release only owned graphs before publishing success
                    except Exception as error:
                        hold['cleanup_failure'] = f'{type(error).__name__}: {error}'
                passed = hold['checks'] is not None and all(row['passed'] for row in hold['checks'])
                payload = {**payload, 'experimental_candidate': 'devlocal_fixed_geometry_raw_velocity_graph_not_product_capability',
                           'velocity_graph_summary': summary, 'refresh_checks_passed': passed,
                           'graph_cleanup_failure': hold['cleanup_failure'],
                           'extra_refresh_requests_excluded_from_samples': 2*len(hold['checks']) if hold['checks'] is not None else None}
                if payload['status'] == 'passed_execution_checks' and (not passed or hold['cleanup_failure']):
                    payload['status'] = 'failed_graph_refresh_or_cleanup'
            return original_write(path, payload)
        for name, replacement, original in (('instrument_model', instrument, original_instrument),
                ('load_fixture', fixture, original_fixture), ('timed_infer', timed, original_timed),
                ('write_json', publish, original_write)):
            setattr(helper, name, replacement)
            stack.callback(setattr, helper, name, original)
        code = helper.main(forwarded)
    if hold['checks'] is None or not all(row['passed'] for row in hold['checks']) or hold['cleanup_failure']:
        return 1
    return code


if __name__ == '__main__':
    raise SystemExit(main())
