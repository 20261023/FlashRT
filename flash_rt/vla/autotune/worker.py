"""One candidate per process; workers write bundles, never the controller DB."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback
from uuid import uuid4

import numpy as np

from .workloads import NPZWorkload
from .quality import quality_contract


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _device_info():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('ln'):
        raise RuntimeError('Real model experiments must run in an allocated Slurm compute step')
    import torch
    properties = torch.cuda.get_device_properties(0)
    return {'name': properties.name, 'uuid': str(getattr(properties, 'uuid', '')),
            'capability': list(torch.cuda.get_device_capability()),
            'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def _quality(actions, request_ids, reference, gate, *, baseline_reference=False):
    if baseline_reference:
        return {'status': 'passed', **quality_contract(gate),
                'metrics': {'max_relative_l2': 0.0, 'max_abs': 0.0, 'min_cosine': 1.0},
                'reference_role': True,
                'interpretation': 'Reference point, not a task-success assertion'}
    if reference is None:
        return {'status': 'unknown', **quality_contract(gate), 'metrics': {}}
    with np.load(reference, allow_pickle=False) as data:
        by_request = {}
        for index, request_id in enumerate(data['request_ids'].tolist()):
            by_request.setdefault(request_id, data[f'actions_{index:05d}'].copy())
    errors, absolutes, cosines = [], [], []
    action_dim = (gate or {}).get('action_dim')
    for actual, request_id in zip(actions, request_ids):
        if request_id not in by_request or actual.shape != by_request[request_id].shape:
            raise ValueError('Reference outputs do not cover the same request IDs/action shapes')
        expected = by_request[request_id]
        if action_dim is not None:
            if type(action_dim) is not int or not 1 <= action_dim <= actual.shape[-1]:
                raise ValueError('quality action_dim must select valid trained/public channels')
            expected, actual = expected[..., :action_dim], actual[..., :action_dim]
        expected = expected.astype(np.float64).ravel()
        observed = actual.astype(np.float64).ravel()
        delta = observed - expected
        norm = np.linalg.norm(expected)
        errors.append(float(np.linalg.norm(delta) / max(norm, 1e-12)))
        absolutes.append(float(np.abs(delta).max()))
        denom = norm * np.linalg.norm(observed)
        cosines.append(float(np.dot(expected, observed) / denom) if denom else float(np.array_equal(expected, observed)))
    metrics = {'max_relative_l2': max(errors), 'max_abs': max(absolutes), 'min_cosine': min(cosines)}
    thresholds = {key: gate[key] for key in metrics if gate is not None and key in gate}
    status = 'unknown'
    if thresholds:
        passed = all(metrics[key] >= value if key == 'min_cosine' else metrics[key] <= value
                     for key, value in thresholds.items())
        status = 'passed' if passed else 'failed'
    return {'status': status, 'scope': (gate or {}).get('scope', 'numerical_replay'),
            'metrics': metrics, 'gate': thresholds, 'action_dim': action_dim,
            'interpretation': 'Fixed-input numerical proxy only; not closed-loop task success'}


def run_experiment(candidate, workload, output, *, attempt_id, comparison_id,
                   warmup=3, repeats=10, reference=None, quality_gate=None,
                   baseline_reference=False, backend_factory=None, device_info=None,
                   synchronize=None, profile=True, campaign_instance_id=None):
    """Run a complete serial request measurement and retain all raw samples.

    Fixture file IO and per-request host copies are outside the timing boundary.
    Frontend input processing/transfers/model/routing/output return are inside.
    An injected backend/device boundary supports CPU contract tests; the real
    factory is only entered within Slurm. Output directories cannot be rerun.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('warmup must be a nonnegative integer')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'attempt.json').open('x') as stream:
        json.dump({'attempt_id': attempt_id, 'candidate_id': candidate.candidate_id}, stream)
    record = {'schema_version': 2, 'candidate_id': candidate.candidate_id,
              'attempt_id': attempt_id, 'comparison_id': comparison_id,
              'workload_id': candidate.workload_id, 'candidate': candidate.to_dict(),
              'campaign_instance_id': campaign_instance_id or f'standalone-{uuid4()}',
              'job_id': (device_info or {}).get('job_id') if backend_factory is not None else os.environ.get('SLURM_JOB_ID')}
    backend = None
    try:
        if workload.workload_id != candidate.workload_id or workload.model != candidate.backend.model:
            raise ValueError('Candidate model/workload does not match the input manifest')
        if repeats < len(workload):
            raise ValueError('Total repeat budget must cover every workload request for fixed-noise quality')
        if backend_factory is None:
            from flash_rt.vla import load_backend
            backend_factory = load_backend
            record['gpu'] = _device_info()
            import torch
            synchronize = torch.cuda.synchronize
        else:
            record['gpu'] = dict(device_info or {'name': 'injected-device-boundary'})
        sync = synchronize or (lambda: None)
        started = time.perf_counter_ns()
        backend = backend_factory(candidate.backend)
        sync()
        load_ms = (time.perf_counter_ns() - started) / 1e6
        started = time.perf_counter_ns()
        for index in range(len(workload)):
            request = workload.request(index)
            backend.prepare(request.observation, prompt=request.prompt)
        sync()
        prepare_ms = (time.perf_counter_ns() - started) / 1e6
        for index in range(warmup):
            request = workload.request(index)
            backend.infer(request.observation, prompt=request.prompt, noise=request.noise)
        sync()
        latency, actions, ids, sampler_metadata, execution_metadata = [], [], [], [], []
        backend_metadata = None
        for index in range(repeats):
            request = workload.request(index)
            sync()
            started = time.perf_counter_ns()
            result = backend.infer(request.observation, prompt=request.prompt, noise=request.noise)
            sync()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError('Invalid wall-clock measurement')
            action = np.asarray(result.get('raw_actions', result['actions']), dtype=np.float32)
            if action.size == 0 or not np.isfinite(action).all():
                raise ValueError('Backend returned empty/nonfinite actions')
            metadata = result.get('sampler_metadata')
            if not isinstance(metadata, dict) or metadata.get('nfe', 0) < 1:
                raise ValueError('Actual per-request NFE metadata is required')
            latency.append(elapsed)
            actions.append(action.copy())
            ids.append(request.request_id)
            sampler_metadata.append(metadata)
            current_backend = dict(result.get('backend_metadata', {}))
            if isinstance(current_backend.get('execution_coverage'), dict):
                coverage = dict(current_backend['execution_coverage'])
                # Request/cumulative work counters are evidence, not immutable
                # identity. Keep the engine, graph-mode flag and installed
                # modules here; retain ALL counters in execution_metadata.
                for key in ('native_linear_calls', 'native_plan_stats', 'velocity_graph'):
                    coverage.pop(key, None)
                current_backend['execution_coverage'] = coverage
            execution_metadata.append(result.get('execution_metadata', {}))
            if backend_metadata is None:
                backend_metadata = current_backend
            elif backend_metadata != current_backend:
                raise ValueError('Backend identity changed during one candidate measurement')
        outputs_path = output / 'outputs.npz'
        required_ids = {spec['request_id'] for spec in workload.specs}
        if set(ids) != required_ids:
            raise ValueError('Measured outputs do not cover every workload request')
        np.savez_compressed(outputs_path, request_ids=np.asarray(ids),
                            **{f'actions_{index:05d}': action for index, action in enumerate(actions)})
        record.update(state='measured',
                      measurement={'warmup': warmup, 'repeats': repeats,
                                   'latency_ms': latency, 'load_ms': load_ms, 'prepare_ms': prepare_ms,
                                   'sampler_metadata': sampler_metadata,
                                   'boundary': 'prepared-host-observation-and-noise_to_completed-CPU-actions'},
                      sampler_metadata=sampler_metadata, backend_metadata=backend_metadata,
                      execution_metadata=execution_metadata,
                      outputs_path=str(outputs_path),
                      workload_request_ids=sorted(required_ids),
                      quality=_quality(actions, ids, reference, quality_gate,
                                       baseline_reference=baseline_reference),
                      stage_metrics=None)
        from .dataflow import describe_dataflow
        record['dataflow'] = describe_dataflow(backend)
        if profile:
            try:
                from .profiling import profile_request
                record['stage_metrics'] = profile_request(backend, workload.request(0))
            except Exception as error:
                record['profiling_failure'] = f'{type(error).__name__}: {error}'
    except Exception as error:
        kind = ('oom' if 'OutOfMemory' in type(error).__name__ else
                'setup' if isinstance(error, (ImportError, FileNotFoundError)) else
                'unsupported_or_input' if isinstance(error, (ValueError, NotImplementedError)) else 'execution')
        record = {key: record[key] for key in ('schema_version', 'candidate_id', 'attempt_id',
                                              'comparison_id', 'workload_id', 'candidate',
                                              'campaign_instance_id', 'job_id')}
        record.update(state='failed', failure={'kind': kind, 'message': str(error).strip() or type(error).__name__,
                                               'exception': type(error).__name__})
        (output / 'failure.txt').write_text(traceback.format_exc())
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                record['cleanup_failure'] = f'{type(error).__name__}: {error}'
                if 'measurement' in record:
                    record['partial_measurement'] = record.pop('measurement')
                record['state'] = 'failed'
                record['failure'] = {'kind': 'cleanup',
                                     'message': str(error).strip() or type(error).__name__}
    _write_json(output / 'result.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--workload', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--comparison-id', required=True)
    parser.add_argument('--campaign-instance-id', required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--reference')
    parser.add_argument('--quality-gate', help='JSON string with an explicit numerical proxy gate')
    parser.add_argument('--baseline-reference', action='store_true')
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args(argv)
    from .schema import Candidate
    candidate = Candidate.from_dict(json.loads(Path(args.candidate).read_text()))
    record = run_experiment(candidate, NPZWorkload.from_file(args.workload), args.output,
                            attempt_id=args.attempt_id, comparison_id=args.comparison_id,
                            campaign_instance_id=args.campaign_instance_id,
                            warmup=args.warmup, repeats=args.repeats, reference=args.reference,
                            quality_gate=json.loads(args.quality_gate) if args.quality_gate else None,
                            baseline_reference=args.baseline_reference, profile=not args.no_profile)
    print(json.dumps(record, indent=2), flush=True)
    return 0 if record['state'] == 'measured' else 1


if __name__ == '__main__':
    raise SystemExit(main())
