"""Produce bound original-author outputs for the ordinary VLA experiment loop.

See author_baseline.md. This is a harness companion, never a project backend.
Model computation requires an allocated GPU compute node. The declaration
selects a model adapter; this module owns warmup, timing and evidence.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import time


HERE = Path(__file__).resolve().parent
# The producer is executed by absolute filename from a new empty cwd.
sys.path.insert(0, str(HERE.parents[1]))


def _require_step():
    host = socket.gethostname().split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID', '').isdigit()
            or not (os.environ.get('SLURM_STEP_ID', 'batch') == 'batch' or os.environ.get('SLURM_STEP_ID', '').isdigit())
            or host.startswith(('ln', 'login'))
            or host != (os.environ.get('SLURMD_NODENAME') or os.environ.get('SLURM_JOB_NODELIST', '')).split('.')[0]
            or os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1')):
        raise RuntimeError('Author model execution requires an actual allocated GPU compute node')


def _check_seeded_noise(infer, request, torch):
    """Observe an actual author RNG draw outside samples; never replace its result.

    The authors' seeded APIs cannot accept arbitrary fixed noise. A seed/name
    match alone is insufficient: verify the author's actual typed draw. Sources
    that use another RNG API fail explicitly and need their own observer.
    """
    import numpy as np
    shape = tuple(request.noise.shape)
    shapes = {shape, (1, *shape)} if len(shape) == 2 else {shape}
    original, observed = torch.randn, []

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        if tuple(result.shape) in shapes:
            observed.append(result.detach().clone())
        return result

    try:
        torch.randn = recorded
        infer(request)
        torch.cuda.synchronize()
    finally:
        torch.randn = original
    if len(observed) != 1:
        raise ValueError(f'Author RNG control must observe exactly one initial draw; got {len(observed)}')
    draw = observed[0]
    expected = torch.as_tensor(request.noise).reshape(draw.shape).to(dtype=draw.dtype)
    if not np.array_equal(draw.float().cpu().numpy(), expected.float().cpu().numpy()):
        raise ValueError(f'Author seed produces different typed initial noise for request {request.request_id}')
    return {'request_id': request.request_id, 'dtype': str(draw.dtype), 'shape': list(draw.shape),
            'exact_typed_noise_match': True, 'timed': False, 'rng_output_replaced': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--spec-json', required=True)
    parser.add_argument('--workload', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--comparison-id', required=True)
    parser.add_argument('--campaign-instance-id', required=True)
    parser.add_argument('--bind-saved', '--bind-dm05-saved', dest='bind_saved', type=Path,
                        help='Bind historical outputs with the declared adapter on CPU')
    args = parser.parse_args(argv)
    from flash_rt.vla.autotune.author import (
        BOUNDARY, _validate_spec, _write_json, content_identity, file_sha256, publish_outputs,
        prove_author_model, source_identity, workload_identity)
    from flash_rt.vla.autotune.workloads import NPZWorkload
    spec = _validate_spec(json.loads(args.spec_json))
    workload = NPZWorkload.from_file(args.workload, spec_dir=spec.get('spec_dir'))
    if workload.model != spec['model'] or args.warmup < 0 or args.repeats < len(workload):
        raise ValueError('Author producer requires matching model and a repeat budget covering the complete workload')
    from flash_rt.vla.model_spec import get_spec
    declared = get_spec(spec['model'], spec_dir=spec.get('spec_dir'))
    contract = declared['author']
    adapter = import_module(contract['adapter'])
    if contract['requires_noise_seeds'] and set(spec['options']['noise_seeds']) != {
            request['request_id'] for request in workload.specs}:
        raise ValueError('Author seed map must cover exactly the workload requests')
    if args.bind_saved is not None:
        if not hasattr(adapter, 'bind_saved'):
            raise ValueError('No saved-output binder for this author adapter')
        record = adapter.bind_saved(spec, workload, args.bind_saved, args.output)
        print(json.dumps({'origin': 'author', 'kind': record['kind'],
                          'repeats': record['repeats'], 'new_model_requests': 0}), flush=True)
        return 0
    _require_step()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        # Force local assets; never implicitly fetch a model during a campaign.
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        if hasattr(adapter, 'prepare_environment'):
            adapter.prepare_environment(spec)
        import numpy as np
        import torch
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Author baseline requires one allocated BF16-capable CUDA GPU')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        identity = {'model': spec['model'], 'workload': workload_identity(workload),
                    'checkpoint': content_identity(spec['checkpoint']),
                    'author_source': source_identity(spec), 'output_domain': declared['actions']['worker_domain']}
        started = time.perf_counter_ns()
        model, infer, metadata = adapter.load(spec, torch)
        torch.cuda.synchronize()
        load_ms = (time.perf_counter_ns() - started) / 1e6
        identity['author_model'] = prove_author_model(model, spec)
        if metadata.get('auxiliary_assets') is not None:
            identity['auxiliary_assets'] = metadata['auxiliary_assets']
        nfe_model = metadata.pop('nfe_control_model', None)
        controls = []
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            # Warm every request before formal samples, including when warmup=0,
            # just as worker.prepare initializes each workload request outside timing.
            for index in range(len(workload)):
                infer(workload.request(index))
            for index in range(args.warmup):
                infer(workload.request(index))
            torch.cuda.synchronize()
            if contract['requires_noise_seeds']:
                controls = [_check_seeded_noise(infer, workload.request(index), torch)
                            for index in range(len(workload))]
            if nfe_model is not None:
                control = adapter.check_nfe_once(
                    nfe_model, infer, workload.request(0), expected=spec['num_steps'],
                    synchronize=torch.cuda.synchronize)
                controls.append({key: value for key, value in control.items() if key != 'actions'})
            actions, ids, latency = [], [], []
            expected_shape = tuple(metadata['expected_shape'])
            for index in range(args.repeats):
                request = workload.request(index)
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                action = infer(request)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if action.shape != expected_shape or action.dtype != np.float32 or not np.isfinite(action).all():
                    raise ValueError(f'Author action shape/domain differs: {action.shape}, expected {expected_shape}')
                latency.append(elapsed)
                actions.append(action.copy())
                ids.append(request.request_id)
        # Check the actual imported implementation again after warmup/compile.
        prove_author_model(model, spec)
        properties = torch.cuda.get_device_properties(0)
        record = {
            'kind': 'bound_author_baseline', 'status': 'complete', 'origin': 'author',
            'model': spec['model'], 'identity': identity, 'producer_spec': spec,
            'producer_files': {file.name: file_sha256(file) for file in
                               (Path(__file__), Path(inspect.getfile(adapter)))},
            'producer_command': (list(sys.orig_argv) if argv is None else
                                 [sys.executable, str(Path(__file__).resolve()), *argv]),
            'comparison_id': args.comparison_id, 'campaign_instance_id': args.campaign_instance_id,
            'source_url': identity['author_source']['source_url'],
            'author_source_revision': spec['source_revision'],
            'implementation': metadata['implementation'], 'checkpoint': spec['checkpoint'],
            'gpu': {'name': properties.name, 'uuid': str(getattr(properties, 'uuid', '')),
                    'capability': list(torch.cuda.get_device_capability())},
            'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ.get('SLURM_STEP_ID', 'batch'),
            'runtime': {'python': sys.version, 'executable': sys.executable,
                        'torch': torch.__version__, 'cuda': torch.version.cuda},
            'warmup': args.warmup, 'untimed_prepare_requests': len(workload),
            'repeats': args.repeats, 'load_ms': load_ms, 'sampler': metadata.get('sampler', declared['head']['defaults'].get('sampler', 'euler')),
            'nfe': metadata.get('nfe', spec['num_steps']),
            'NFE_evidence': metadata.get('nfe_evidence', 'Author configuration and untimed controls'),
            'precision': metadata.get('precision', 'bf16'), 'boundary': BOUNDARY, 'request_ids': ids, 'latency_ms': latency,
            'timing_contract': {
                'noise_source': ('seeded-author-rng' if contract['requires_noise_seeds']
                                 else 'provided-host-noise'),
                'cpu_actions': [{'field': 'actions', 'domain': identity['output_domain'],
                                 'shape': metadata['expected_shape'], 'dtype': np.dtype(np.float32).str}]},
            'statistics': {'p50_ms': float(np.median(latency)), 'p95_ms': float(np.percentile(latency, 95)),
                           'mean_ms': float(np.mean(latency))},
            'controls': controls, 'metadata': metadata,
            'quality_interpretation': 'Author fixed-input numerical reference; no closed-loop task-success claim',
        }
        publish_outputs(args.output, record, actions=actions, request_ids=ids, latency_ms=latency)
        print(json.dumps({'origin': 'author', 'model': spec['model'], 'statistics': record['statistics']}), flush=True)
    except Exception as error:
        import traceback
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        _write_json(args.output / 'failure.json', {'origin': 'author', 'status': 'failed',
                                                  'exception': type(error).__name__, 'message': str(error)})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
