"""One candidate per process; workers write bundles, never the controller DB."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from uuid import uuid4

import numpy as np

from .workloads import NPZWorkload
from .quality import compare_outputs, output_domain, quality_contract


# DM's coverage also contains graph addresses, prompt-dependent prefix lengths,
# cumulative enqueues and compile/capture telemetry. Only these fields identify
# the implementation. Full per-request evidence is retained separately.
_DM_EXECUTION_IDENTITY = (
    'engine', 'coverage', 'native_operators', 'torch_regions',
    'native_gemm_per_request', 'native_fa2_per_request', 'native_gelu_per_request',
    'setup_time_modulation_gemm', 'vision_patch_tokens_per_image', 'pooled_image_tokens',
    'use_cuda_graph', 'graph_scope', 'checkpoint', 'cpu_source',
    'reference_math', 'noise_contract', 'project_optimization',
)


def _backend_identity(metadata, model):
    current = deepcopy(metadata)
    coverage = current.get('execution_coverage')
    if isinstance(coverage, dict):
        if model == 'dm05':
            current['execution_coverage'] = {
                key: coverage[key] for key in _DM_EXECUTION_IDENTITY if key in coverage}
        else:
            for key in ('native_linear_calls', 'native_plan_stats', 'velocity_graph',
                        'text_crossattn_kv_cache_request'):
                coverage.pop(key, None)
    return current


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _device_info():
    from .slurm import require_gpu_allocation
    require_gpu_allocation()
    import torch
    properties = torch.cuda.get_device_properties(0)
    return {'name': properties.name, 'uuid': str(getattr(properties, 'uuid', '')),
            'capability': list(torch.cuda.get_device_capability()),
            'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def _quality(actions, request_ids, reference, gate, *, baseline_reference=False,
             domain=None, workload_identity=None):
    if baseline_reference:
        return {'status': 'passed', **quality_contract(gate),
                'metrics': {'max_relative_l2': 0.0, 'max_abs': 0.0, 'min_cosine': 1.0},
                'reference_role': True, 'reference_origin': 'internal',
                'output_domain': domain,
                'interpretation': 'Reference point, not a task-success assertion'}
    if reference is None:
        return {'status': 'unknown', **quality_contract(gate), 'metrics': {},
                'reference_origin': 'internal', 'output_domain': domain}
    with np.load(reference, allow_pickle=False) as data:
        if 'output_domain' in data:
            if domain is not None and str(data['output_domain'].item()) != domain:
                raise ValueError('Internal reference output domain differs')
        elif domain == output_domain('dm05'):
            raise ValueError('DM05 reference has no explicit absolute-action output domain; regenerate it')
        if workload_identity is not None and 'workload_digest' in data:
            if str(data['workload_digest'].item()) != workload_identity['sha256']:
                raise ValueError('Internal reference fixture content differs')
        reference_ids = data['request_ids'].tolist()
        references = [data[f'actions_{index:05d}'].copy() for index in range(len(reference_ids))]
    return compare_outputs(actions, request_ids, references, reference_ids, gate, domain=domain)


def _describe_dataflow(record, backend):
    """Diagnostic cost ledger; its failure must not void a valid measurement."""
    try:
        from .dataflow import describe_dataflow
        record['dataflow'] = describe_dataflow(backend)
    except Exception as error:
        record['dataflow'] = None
        record['dataflow_failure'] = f'{type(error).__name__}: {error}'


def _verify_expected_source(candidate, expected_source, import_root):
    """Prove the imported flash_rt is the candidate's snapshot (before any backend import)."""
    from .snapshots import check_candidate_source, snapshot_id, verify_source, SourceIdentityError
    if not isinstance(expected_source, dict):
        raise SourceIdentityError('expected_source must be a snapshot manifest dict')
    try:
        expected_id = snapshot_id(candidate.source_ref, candidate.patch_ref)
    except ValueError as error:
        raise SourceIdentityError(f'Candidate refs do not name a snapshot: {error}') from error
    if expected_source.get('snapshot_id') != expected_id:
        raise SourceIdentityError(f'Source manifest {expected_source.get("snapshot_id")!r} is not the '
                                  f'candidate snapshot {expected_id}')
    identity = verify_source(expected_source, import_root=import_root)
    check_candidate_source(expected_source, candidate.source_ref, candidate.patch_ref)
    return identity


def run_experiment(candidate, workload, output, *, attempt_id, comparison_id,
                   warmup=3, repeats=10, reference=None, quality_gate=None,
                   baseline_reference=False, backend_factory=None, device_info=None,
                   synchronize=None, profile=True, campaign_instance_id=None,
                   expected_source=None, source_import_root=None, author_reference=None):
    """Run a complete serial request measurement and retain all raw samples.

    Fixture file IO and per-request host copies are outside the timing boundary.
    Frontend input processing/transfers/model/routing/output return are inside.
    An injected backend/device boundary supports CPU contract tests; the real
    factory is only entered within Slurm. Output directories cannot be rerun.
    With ``expected_source`` (a snapshot manifest) the imported flash_rt tree
    is verified before the backend is imported or loaded; the result is then
    schema 3 and carries ``source_identity``. ``source_import_root`` overrides
    the import root (tests); by default it is the parent of ``flash_rt``.
    ``author_reference`` adds a separately bound author quality comparison;
    the normal ``quality`` field still refers to the internal baseline.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('warmup must be a nonnegative integer')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'attempt.json').open('x') as stream:
        json.dump({'attempt_id': attempt_id, 'candidate_id': candidate.candidate_id}, stream)
    record = {'schema_version': 2 if expected_source is None else 3,
              'candidate_id': candidate.candidate_id,
              'attempt_id': attempt_id, 'comparison_id': comparison_id,
              'workload_id': candidate.workload_id, 'candidate': candidate.to_dict(),
              'campaign_instance_id': campaign_instance_id or f'standalone-{uuid4()}',
              'job_id': (device_info or {}).get('job_id') if backend_factory is not None else os.environ.get('SLURM_JOB_ID')}
    backend = None
    try:
        if expected_source is not None:
            record['source_identity'] = _verify_expected_source(candidate, expected_source, source_import_root)
        if workload.workload_id != candidate.workload_id or workload.model != candidate.backend.model:
            raise ValueError('Candidate model/workload does not match the input manifest')
        if repeats < len(workload):
            raise ValueError('Total repeat budget must cover every workload request for fixed-noise quality')
        from .author import workload_identity as identify_workload
        fixture_identity = identify_workload(workload)
        domain = output_domain(candidate.backend.model)
        author = None
        if author_reference is not None:
            from .author import load_reference
            author = load_reference(author_reference, workload=workload, backend=candidate.backend)
        record['runtime'] = {'python': sys.version, 'executable': sys.executable}
        if backend_factory is None:
            from flash_rt.vla import load_backend
            backend_factory = load_backend
            record['gpu'] = _device_info()
            import torch
            synchronize = torch.cuda.synchronize
            record['runtime'].update(torch=torch.__version__, cuda=torch.version.cuda)
        else:
            record['gpu'] = dict(device_info or {'name': 'injected-device-boundary'})
        if author is not None and author['descriptor']['provenance'] == 'same_campaign':
            from .author import _check_device
            _check_device(record, author['record'])
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
        request_backend_metadata = []
        backend_metadata = None
        timing_contract = None
        for index in range(repeats):
            request = workload.request(index)
            sync()
            started = time.perf_counter_ns()
            result = backend.infer(request.observation, prompt=request.prompt, noise=request.noise)
            sync()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError('Invalid wall-clock measurement')
            # DM's raw_actions are normalized 14-D latents. Its public gate and
            # the author _predict API both use reconstructed absolute actions.
            value = (result['actions'] if candidate.backend.model == 'dm05'
                     else result.get('raw_actions', result['actions']))
            action = np.asarray(value, dtype=np.float32)
            if action.size == 0 or not np.isfinite(action).all():
                raise ValueError('Backend returned empty/nonfinite actions')
            metadata = result.get('sampler_metadata')
            if not isinstance(metadata, dict) or metadata.get('nfe', 0) < 1:
                raise ValueError('Actual per-request NFE metadata is required')
            latency.append(elapsed)
            actions.append(action.copy())
            ids.append(request.request_id)
            sampler_metadata.append(deepcopy(metadata))
            full_backend = deepcopy(result.get('backend_metadata', {}))
            request_backend_metadata.append(full_backend)
            cpu_actions = [
                {'field': key,
                 'domain': (full_backend.get('action_domain') if key == 'actions'
                            else 'normalized-model-actions' if candidate.backend.model == 'dm05' else domain),
                 'shape': list(np.asarray(result[key]).shape),
                 'dtype': np.asarray(result[key]).dtype.str}
                for key in ('actions', 'raw_actions') if key in result]
            current_contract = {'noise_source': 'provided-host-noise', 'cpu_actions': cpu_actions}
            if timing_contract is None:
                timing_contract = current_contract
            elif timing_contract != current_contract:
                raise ValueError('CPU action return contract changed during one candidate measurement')
            current_backend = _backend_identity(full_backend, candidate.backend.model)
            execution_metadata.append(deepcopy(result.get(
                'execution_metadata', full_backend.get('execution_coverage', {}))))
            if backend_metadata is None:
                backend_metadata = current_backend
            elif backend_metadata != current_backend:
                raise ValueError('Backend identity changed during one candidate measurement')
        outputs_path = output / 'outputs.npz'
        required_ids = {spec['request_id'] for spec in workload.specs}
        if set(ids) != required_ids:
            raise ValueError('Measured outputs do not cover every workload request')
        np.savez_compressed(outputs_path, request_ids=np.asarray(ids),
                            output_domain=np.asarray(domain),
                            workload_digest=np.asarray(fixture_identity['sha256']),
                            **{f'actions_{index:05d}': action for index, action in enumerate(actions)})
        record.update(state='measured',
                       measurement={'warmup': warmup, 'repeats': repeats,
                                    'latency_ms': latency, 'load_ms': load_ms, 'prepare_ms': prepare_ms,
                                    'sampler_metadata': sampler_metadata,
                                    'timing_contract': timing_contract,
                                    'boundary': 'prepared-host-observation-and-noise_to_completed-CPU-actions'},
                       sampler_metadata=sampler_metadata, backend_metadata=backend_metadata,
                       execution_metadata=execution_metadata,
                       request_backend_metadata=request_backend_metadata,
                       output_domain=domain,
                       output_field='actions' if candidate.backend.model == 'dm05' else 'raw_actions_if_available',
                       workload_identity=fixture_identity,
                      outputs_path=str(outputs_path),
                      workload_request_ids=sorted(required_ids),
                       quality=_quality(actions, ids, reference, quality_gate,
                                        baseline_reference=baseline_reference, domain=domain,
                                        workload_identity=fixture_identity),
                       stage_metrics=None)
        if author is not None:
            from .author import compare_reference
            record['author_quality'] = compare_reference(actions, ids, author, quality_gate)
        _describe_dataflow(record, backend)
        if profile:
            try:
                from .profiling import profile_request
                record['stage_metrics'] = profile_request(backend, workload.request(0))
            except Exception as error:
                record['profiling_failure'] = f'{type(error).__name__}: {error}'
    except Exception as error:
        from .author import AuthorReferenceError
        from .snapshots import SourceIdentityError
        kind = ('author_reference' if isinstance(error, AuthorReferenceError) else
                'source_identity' if isinstance(error, SourceIdentityError) else
                'oom' if 'OutOfMemory' in type(error).__name__ else
                'setup' if isinstance(error, (ImportError, FileNotFoundError)) else
                'unsupported_or_input' if isinstance(error, (ValueError, NotImplementedError)) else 'execution')
        record = {key: record[key] for key in ('schema_version', 'candidate_id', 'attempt_id',
                                               'comparison_id', 'workload_id', 'candidate',
                                               'campaign_instance_id', 'job_id', 'source_identity', 'runtime')
                  if key in record}
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
    parser.add_argument('--author-reference', help='Bound author reference descriptor written by batch')
    parser.add_argument('--quality-gate', help='JSON string with an explicit numerical proxy gate')
    parser.add_argument('--baseline-reference', action='store_true')
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--source-manifest', help='Snapshot manifest; verify the imported source tree first')
    args = parser.parse_args(argv)
    from .schema import Candidate
    candidate = Candidate.from_dict(json.loads(Path(args.candidate).read_text()))
    record = run_experiment(candidate, NPZWorkload.from_file(args.workload), args.output,
                            attempt_id=args.attempt_id, comparison_id=args.comparison_id,
                            campaign_instance_id=args.campaign_instance_id,
                            warmup=args.warmup, repeats=args.repeats, reference=args.reference,
                            quality_gate=json.loads(args.quality_gate) if args.quality_gate else None,
                             baseline_reference=args.baseline_reference, profile=not args.no_profile,
                             author_reference=args.author_reference,
                            expected_source=(json.loads(Path(args.source_manifest).read_text())
                                             if args.source_manifest else None))
    print(json.dumps(record, indent=2), flush=True)
    return 0 if record['state'] == 'measured' else 1


if __name__ == '__main__':
    raise SystemExit(main())
