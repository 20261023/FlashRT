"""Formal DM0.5 public-path timings, not a test/smoke or the fixed harness.

Run serial independent processes for torch / flashrt / torch-bookend inside one
allocated Slurm GPU step. File decode/noise loading are outside timing; every
sample includes host transforms, current-state tokenization, H2D, the complete
BF16 Euler10 model, D2H and absolute FP32 CPU[50,14] reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def source_identity(root):
    """Verify the frozen bundle before loading models; never edit it in flight."""
    manifest = json.loads((root / 'DM05_SOURCE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if sha256(root / name) != expected:
            raise RuntimeError(f'Frozen DM05 source differs: {name}')
    return {'manifest_sha256': sha256(root / 'DM05_SOURCE_MANIFEST.json'),
            'base_commit': manifest['base_commit'], 'source_digest': manifest['source_digest'],
            'root': str(root), 'files': manifest['files']}


def numerical_metrics(actual, reference):
    import numpy as np
    left = np.asarray(actual, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError('Formal public outputs contain NaN/Inf')
    relative = float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-30))
    cosine = float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-30))
    return {'relative_l2': relative, 'cosine': cosine,
            'max_abs': float(np.max(np.abs(left - right))),
            'passed': relative <= 0.05 and cosine >= 0.999}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Pinned OpenDM CPU source')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Actual native329331 saved four-case outputs/noise')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('torch', 'flashrt'), required=True)
    parser.add_argument('--role', choices=('project', 'flashrt', 'bookend'), required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    from flash_rt.vla.autotune.slurm import require_gpu_allocation
    step = require_gpu_allocation()
    if args.warmup != 5 or args.repeats != 10:
        raise ValueError('This fixed formal profile requires five warmup and ten formal requests per case')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import numpy as np
    import torch
    from PIL import Image
    from flash_rt.vla import BackendConfig, load_backend
    import flash_rt

    root = Path(flash_rt.__file__).resolve().parents[1]
    identity = source_identity(root)
    if tuple(torch.cuda.get_device_capability()) != (8, 0):
        raise RuntimeError('Formal DM05 tuple requires SM80')
    torch.set_num_threads(2)
    config = BackendConfig(
        model='dm05', checkpoint=str(args.checkpoint), backend=args.backend,
        hardware='cuda_sm80', precision='bf16', num_steps=10,
        options={'source_dir': str(args.source), 'use_cuda_graph': True, 'num_views': 3})
    start = time.perf_counter()
    execution = load_backend(config)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    reference = json.loads((args.reference / 'result.json').read_text())
    images = [Image.open(args.source / 'assets/demo/images/episode0' / camera / '0.jpg').convert('RGB')
              for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')]
    cases, samples, execution_metadata = [], [], []
    try:
        for case in reference['cases']:
            target = args.output / case['id']
            target.mkdir()
            noise_path = args.reference / case['id'] / 'initial_noise.pt'
            # Trusted local tensor-only artifacts, explicitly weights_only.
            noise = torch.load(noise_path, map_location='cpu', weights_only=True)
            observation = {'images': images, 'state': np.asarray(case['state'], dtype=np.float32),
                           'prompt': case['prompt'], 'robot_type': 'DOS W1'}
            start = time.perf_counter()
            for _ in range(args.warmup):
                execution.infer(observation, noise=noise)
            torch.cuda.synchronize()
            warmup_s = time.perf_counter() - start
            actions, raw_actions, times, metadata = [], [], [], []
            for repeat in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                result = execution.infer(observation, noise=noise)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if result['actions'].shape != (50, 14) or result['actions'].dtype != np.float32:
                    raise RuntimeError('Public backend did not return complete FP32 CPU[50,14]')
                if result['sampler_metadata']['nfe'] != 10:
                    raise RuntimeError('Actual NFE differs from fixed Euler10')
                actions.append(np.array(result['actions'], copy=True))
                raw_actions.append(np.array(result['raw_actions'], copy=True))
                times.append(elapsed)
                metadata.append(result['execution_metadata'])
                samples.append({'case': case['id'], 'repeat': repeat,
                                'full_policy_ms': elapsed, 'nfe': 10})
            np.save(target / 'actions.npy', np.stack(actions))
            np.save(target / 'raw_actions.npy', np.stack(raw_actions))
            np.save(target / 'samples_ms.npy', times)
            torch.save(execution.frontend.last_prepared, target / 'prepared.pt')
            torch.save(noise, target / 'initial_noise.pt')
            (target / 'execution_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            # Arithmetic reads SAVED formal outputs, not another model request.
            saved = np.load(target / 'actions.npy', allow_pickle=False)
            oracle = np.load(args.reference / case['id'] / 'actions.npy', allow_pickle=False)
            quality = [numerical_metrics(saved[i], oracle[i]) for i in range(args.repeats)]
            record = {key: case[key] for key in ('id', 'seed', 'state', 'prompt', 'prefix_length')}
            record.update(warmup_s=warmup_s, full_policy_median_ms=statistics.median(times),
                          full_policy_p95_ms=float(np.percentile(times, 95)),
                          actual_prefix_length=metadata[-1]['prefix_length'],
                          identical_repeats=bool(np.array_equal(saved, np.broadcast_to(saved[:1], saved.shape))),
                          quality=quality, quality_passed=all(q['passed'] for q in quality),
                          fixed_noise_sha256=sha256(noise_path))
            cases.append(record)
            execution_metadata.append(metadata[-1])
            (target / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(record), flush=True)
        times = [sample['full_policy_ms'] for sample in samples]
        modules = {}
        if args.backend == 'flashrt':
            from flash_rt import flash_rt_fa2, flash_rt_gemm_plan, flash_rt_kernels
            for module in (flash_rt_fa2, flash_rt_gemm_plan, flash_rt_kernels):
                modules[module.__name__] = {'path': module.__file__, 'sha256': sha256(module.__file__)}
        report = {
            'model': 'dm05', 'backend': args.backend, 'role': args.role, 'origin': 'project',
            'job_id': os.environ['SLURM_JOB_ID'], 'step_id': step,
            'gpu': torch.cuda.get_device_name(), 'gpu_capability': [8, 0],
            'gpu_properties': str(torch.cuda.get_device_properties(0)),
            'source_identity': identity, 'native_modules': modules,
            'checkpoint': str(args.checkpoint), 'reference': str(args.reference),
            'checkpoint_identity': execution.frontend.pipeline.identity,
            'cpu_source_identity': execution.frontend.source_identity,
            'dtype': 'BF16-mixed-author-hot-suffix', 'batch': 1, 'cameras': 3,
            'history': 0, 'chunk': 50, 'internal_action_dim': 32, 'output_action_dim': 14,
            'steps': 10, 'nfe': 10, 'image_size': [448, 448],
            'boundary': 'host PIL RGB+state+prompt+provided noise -> full absolute FP32 CPU[50,14]; excludes file decode/network',
            'rng_caveat': 'author times include seeded CUDA RNG; project/FRT use its saved exact BF16 initial draw',
            'load_s': load_s, 'capture_s': execution.frontend.capture_s,
            'warmup_per_case': 5, 'samples_per_case': 10,
            'full_policy_median_ms': statistics.median(times),
            'full_policy_p95_ms': float(np.percentile(times, 95)),
            'quality_contract': {'scope': 'absolute-FP32-CPU-public14-vs-author329331',
                                 'max_relative_l2': 0.05, 'min_cosine': 0.999},
            'quality_passed': all(case['quality_passed'] for case in cases),
            'cases': cases, 'samples': samples, 'execution_metadata': execution_metadata,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'torchvision', 'transformers', 'safetensors', 'numpy')},
            'tests_run': False, 'extra_reviews': False,
            'functional_release_acceptance': 'not-run-user-waiver',
            'optimization_status': 'first-formal-baseline-not-best-or-winner',
        }
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: report[key] for key in
                          ('backend', 'role', 'full_policy_median_ms', 'full_policy_p95_ms',
                           'quality_passed', 'load_s', 'capture_s')}), flush=True)
    finally:
        execution.close()


if __name__ == '__main__':
    main()
