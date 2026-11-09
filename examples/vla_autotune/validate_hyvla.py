"""Real Hy-Embodied-0.5-VLA validation; one candidate per allocated GPU process.

Inference requires an actual Slurm srun GPU step. --compare is CPU-only. Nothing
imports Torch, loads weights, downloads assets, or submits jobs at module import.
This is raw normalized-model-action validation, NOT a robot success benchmark.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np


CAMERAS = ('top_head', 'hand_left', 'hand_right')
PROFILE = 'flashrt_semantic_bf16_rope_qknorm_sdpa'
PROFILES = {'semantic_bf16': PROFILE,
            'official_local_fp32': 'flashrt_official_local_fp32_hooks_not_exact'}
OFFICIAL_PROFILE = 'recorded_official_bf16_weights_local_fp32_rope_attention'
# EXACTLY the old scripts/probe_flashrt_hyvla.py same-action gate. Do not tune
# these to a candidate's observed error. Internal parity has separate CLI knobs.
ORACLE_ATOL, ORACLE_RTOL, ORACLE_COSINE = 0.01, 0.01, 0.999
SCHEMA_VERSION = 1


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, record):
    with Path(path).open('x') as stream:
        json.dump(jsonable(record), stream, indent=2, allow_nan=False)
        stream.write('\n')


def _float_array(arrays, key, shape):
    value = arrays.get(key)
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype != np.float32 or not np.isfinite(value).all()):
        raise ValueError(f'{key} must be finite float32{shape}; no implicit conversion')


def validate_fixture(arrays):
    for camera in CAMERAS:
        value = arrays.get(camera)
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
                or value.shape != (6, 480, 640, 3)):
            raise ValueError(f'{camera} must be raw uint8[6,480,640,3]')
    for key, shape in (('state', (20,)), ('noise', (40, 32)),
                       ('expected_actions32', (40, 32)), ('expected_actions20', (40, 20))):
        _float_array(arrays, key, shape)
    prompt = arrays.get('prompt')
    if (not isinstance(prompt, np.ndarray) or prompt.shape != ()
            or prompt.dtype.kind != 'U' or not str(prompt.item()).strip()):
        raise ValueError('prompt must be nonempty scalar Unicode text')
    if not np.array_equal(arrays['expected_actions32'][:, :20], arrays['expected_actions20']):
        raise ValueError('expected_actions20 oracle slice does not match expected_actions32')
    if 'expected_state' in arrays:
        _float_array(arrays, 'expected_state', (1, 32))
        padded = np.pad(arrays['state'], (0, 12))[None]
        if not np.array_equal(padded, arrays['expected_state']):
            raise ValueError('expected_state disagrees with zero-padded normalized raw state')


def load_fixture(path):
    required = (*CAMERAS, 'state', 'noise', 'prompt', 'expected_actions32', 'expected_actions20')
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f'Fixture missing actual raw/oracle fields: {sorted(missing)}')
        arrays = {key: archive[key].copy() for key in required}
        if 'expected_state' in archive.files:
            arrays['expected_state'] = archive['expected_state'].copy()
    validate_fixture(arrays)
    return arrays


def camera_order(camera_keys):
    keys = list(camera_keys)
    names = [key.removeprefix('observation.images.') for key in keys]
    if len(keys) != 3 or len(set(names)) != 3 or set(names) != set(CAMERAS):
        raise ValueError(f'Checkpoint camera order must contain all three known cameras, got {keys}')
    return keys, names


def make_observation(arrays, camera_keys):
    keys, names = camera_order(camera_keys)
    # Preserve raw uint8 histories and already-normalized 20-D state. The real
    # frontend owns uint8->float conversion, resize/pad, BF16 cast and 32-D pad.
    return {'images': {key: arrays[name] for key, name in zip(keys, names)},
            'state': arrays['state'].copy(), 'prompt': str(arrays['prompt'].item())}


def require_slurm_step(env=None):
    env = os.environ if env is None else env
    if (not env.get('SLURM_JOB_ID') or not env.get('SLURM_STEP_ID', '').isdigit()):
        raise RuntimeError('Model validation requires an allocated Slurm srun GPU step; never run on login')
    if env.get('CUDA_VISIBLE_DEVICES', '') in ('', '-1', 'NoDevFiles', 'none', 'void'):
        raise RuntimeError('Allocated GPU step must expose CUDA_VISIBLE_DEVICES')
    if env.get('FLASHRT_HYVLA_FORCE_ARCH', '0') not in ('', '0'):
        raise RuntimeError('New HyVLA BF16 path must run without FLASHRT_HYVLA_FORCE_ARCH')


def require_cuda(torch):
    if not torch.cuda.is_available() or getattr(torch.version, 'hip', None):
        raise RuntimeError('Allocated step has no usable NVIDIA compute GPU')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly the one allocated GPU to this candidate process')
    if tuple(torch.cuda.get_device_capability()) != (8, 0):
        raise RuntimeError('This A800 validation profile requires actual SM80, not a forced architecture')
    # Check a real compute operation before loading weights, not nvidia-smi alone.
    probe = torch.zeros(1, device='cuda').add_(1)
    torch.cuda.synchronize()
    if probe.item() != 1:
        raise RuntimeError('Allocated CUDA compute check failed')
    props = torch.cuda.get_device_properties(0)
    return {'name': props.name, 'capability': [props.major, props.minor],
            'total_memory_bytes': props.total_memory,
            'uuid': str(getattr(props, 'uuid', 'unavailable')),
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}


def validate_actions(actions):
    if (not isinstance(actions, np.ndarray) or actions.shape != (40, 32)
            or actions.dtype != np.float32 or not np.isfinite(actions).all()):
        raise ValueError('Backend actions must be finite float32[40,32] normalized padded model actions')


def difference(actual, reference, *, atol, rtol):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if (actual.shape != reference.shape or actual.size == 0
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError('Compared arrays must be finite, nonempty and have identical shapes')
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError('Comparison tolerances must be finite and nonnegative')
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    delta = np.abs(a - b)
    outside = int(np.count_nonzero(delta > atol + rtol * np.abs(b)))
    na, nb = float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel()))
    cosine = (float(np.dot((a / na).ravel(), (b / nb).ravel())) if na and nb
              else float(na == nb))
    return {'shape': list(actual.shape), 'exact': bool(np.array_equal(actual, reference)),
            'allclose': outside == 0, 'atol': atol, 'rtol': rtol,
            'outside_tolerance': outside, 'elements': int(actual.size),
            'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'rmse': float(np.sqrt(np.mean((a - b) ** 2))),
            'max_relative_to_reference_floor_1e_12': float((delta / np.maximum(np.abs(b), 1e-12)).max()),
            'cosine': max(-1., min(1., cosine))}


def oracle_comparison(actions, arrays, *, sampler, num_steps):
    eligible = sampler == 'euler' and num_steps == 10
    metrics = {name: difference(actual, arrays['expected_' + name], atol=ORACLE_ATOL, rtol=ORACLE_RTOL)
               for name, actual in (('actions32', actions), ('actions20', actions[:, :20]))}
    gate32 = metrics['actions32']['allclose'] and metrics['actions32']['cosine'] >= ORACLE_COSINE
    gate20 = metrics['actions20']['allclose'] and metrics['actions20']['cosine'] >= ORACLE_COSINE
    return {'reference_profile': OFFICIAL_PROFILE, 'eligible_same_schedule': eligible,
            'interpretation': ('same_schedule_official_recorded_oracle' if eligible
                               else 'different_schedule_diagnostic_only_not_a_parity_gate'),
            'gate': {'atol': ORACLE_ATOL, 'rtol': ORACLE_RTOL, 'minimum_cosine': ORACLE_COSINE,
                     'legacy_gate_domain': 'normalized_padded_40x32'},
            'same_action_gate_passed': bool(gate32) if eligible else None,
            'public20_gate_passed': bool(gate20) if eligible else None, **metrics}


def validate_sampler_metadata(metadata, *, sampler, num_steps, max_steps,
                              probe_horizon=0.5, step_increment=2):
    if not isinstance(metadata, dict) or metadata.get('sampler') != sampler:
        raise ValueError('Missing or incorrect actual sampler metadata')
    count, nfe = metadata.get('scheduled_steps'), metadata.get('nfe')
    if type(count) is not int or type(nfe) is not int:
        raise ValueError('Actual scheduled_steps/nfe must be integer counts')
    if sampler == 'euler':
        if count != num_steps:
            raise ValueError('Actual Euler scheduled_steps differs from requested num_steps')
        expected_nfe = count
        expected_times = [1. + index * (-1. / count) for index in range(count)]
    else:
        schedules = set(range(2, max_steps + 1, step_increment)) | {max_steps}
        if count not in schedules:
            raise ValueError('Actual ProbeFlow scheduled_steps is outside requested schedules')
        expected_nfe = 2 if count == 2 else count + 1
        expected_times = [1., 1. - probe_horizon]
        if count > 2:
            expected_times.extend(1. - index / count for index in range(1, count))
        if 'cosine' in metadata and not math.isfinite(metadata['cosine']):
            raise ValueError('Actual ProbeFlow routing cosine is nonfinite')
    if nfe != expected_nfe:
        raise ValueError(f'Actual nfe={nfe} does not match the requested integration contract ({expected_nfe})')
    if metadata.get('timesteps') != expected_times:
        raise ValueError('Actual sampler timesteps differ from the exact requested grid/probe accounting')


def measure_requests(infer, synchronize, initial_noise, *, warmup, repeats,
                     clock=time.perf_counter_ns, atol=1e-6, rtol=1e-6):
    if not 1 <= warmup <= 20 or not 2 <= repeats <= 100:
        raise ValueError('warmup must be in [1,20] and repeats in [2,100] for replay checks')
    for _ in range(warmup):
        validate_actions(infer(initial_noise.copy())['actions'])
    synchronize()
    outputs, samples, metadata, backend_metadata = [], [], [], []
    for _ in range(repeats):
        # Never let sample_actions or any frontend mutate next request's noise.
        fresh_noise = initial_noise.copy()
        synchronize()
        started = clock()
        result = infer(fresh_noise)
        synchronize()
        elapsed = (clock() - started) / 1e6
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Invalid measured latency')
        validate_actions(result['actions'])
        outputs.append(result['actions'].copy())
        samples.append(elapsed)
        metadata.append(jsonable(result.get('sampler_metadata')))
        backend_metadata.append(jsonable(result.get('backend_metadata')))
    comparisons = [difference(value, outputs[0], atol=atol, rtol=rtol) for value in outputs[1:]]
    return {'actions32': outputs[0], 'replay_actions32': np.stack(outputs),
            'latency_ms': np.asarray(samples, np.float64), 'sampler_metadata': metadata,
            'backend_metadata': backend_metadata,
            'sampler_metadata_replay_equal': all(value == metadata[0] for value in metadata),
            'fixed_noise_replay': {'exact': all(value['exact'] for value in comparisons),
                                   'allclose': all(value['allclose'] for value in comparisons),
                                   'max_abs': max(value['max_abs'] for value in comparisons),
                                   'comparisons_to_first': comparisons}}


def check_graph_eager(backend, observation, noise, synchronize, *, free_bytes,
                      min_free_gib, atol, rtol):
    required = int(min_free_gib * 2**30)
    memory = {'free_bytes_before': free_bytes, 'minimum_free_bytes': required,
              'memory_note': 'Conservative admission check, not a proof against OOM; one set of weights only'}
    if free_bytes < required:
        return {'status': 'skipped_insufficient_free_memory', 'passed': None, **memory}, {}
    frontend = backend.frontend
    if not frontend.use_cuda_graph:
        raise ValueError('Optional same-head graph/eager check requires --use-graph')
    graph_result = backend.infer(observation, noise=noise.copy())
    synchronize()
    validate_actions(graph_result['actions'])
    graph_actions = graph_result['actions'].copy()
    # This explicitly opt-in validation switch uses the SAME frontend/weights.
    # HyVLAFlowFrontend uses this flag for both ViT and prefix/velocity graphs.
    frontend.use_cuda_graph = False
    try:
        eager_result = backend.infer(observation, noise=noise.copy())
        synchronize()
        validate_actions(eager_result['actions'])
        eager_actions = eager_result['actions'].copy()
    finally:
        frontend.use_cuda_graph = True
    metrics = difference(graph_actions, eager_actions, atol=atol, rtol=rtol)
    return {'status': 'checked', 'passed': metrics['allclose'], 'comparison': metrics,
            'graph_sampler_metadata': jsonable(graph_result['sampler_metadata']),
            'eager_sampler_metadata': jsonable(eager_result['sampler_metadata']), **memory}, {
                'graph_actions32': graph_actions, 'eager_actions32': eager_actions}


def _file_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def fixture_record(path, manifest_path=None):
    path = Path(path).resolve()
    result = _file_record(path)
    explicit_manifest = manifest_path is not None
    manifest_path = manifest_path or path.with_name('manifest.json')
    if Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text())
        result['manifest'] = {
            'path': str(Path(manifest_path).resolve()),
            **{key: manifest.get(key) for key in ('job_id', 'source_revision', 'checkpoint_revision')},
            'load_report': manifest.get('load_report'),
            'variant': path.stem,
            'variant_bytes_recorded': manifest.get('variants', {}).get(path.stem, {}).get('bytes'),
            'provenance_note': 'Recorded manifest claims only; no content digests computed or verified',
        }
        expected_size = result['manifest']['variant_bytes_recorded']
        if expected_size is not None and expected_size != result['size_bytes']:
            raise ValueError('Fixture size differs from its recorded manifest')
    elif explicit_manifest:
        raise ValueError('Explicit fixture manifest does not exist')
    return result


def checkpoint_record(path):
    path = Path(path).resolve()
    cfg = json.loads((path / 'config.json').read_text())
    required = {'num_steps': 10, 'chunk_size': 40, 'n_action_steps': 40,
                'max_action_dim': 32, 'max_state_dim': 32, 'tokenizer_max_length': 64,
                'use_video_encoder': True, 'spacetime_layer_stride': 4,
                'past_drop_layer': None, 'visual_segment_isolation': False}
    for name, expected in required.items():
        if name not in cfg or cfg[name] != expected:
            raise ValueError(f'Checkpoint {name} must be {expected!r} for this fixture profile')
    keys, _ = camera_order(cfg.get('image_features', {}))
    for filename in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        if not (path / filename).is_file():
            raise ValueError(f'Missing local checkpoint tokenizer asset: {filename}')
    weights = _file_record(path / 'model.safetensors')
    return cfg, {'path': str(path), 'weights_size_bytes': weights['size_bytes'],
                 'weights_mtime_ns': weights['mtime_ns'], 'config': _file_record(path / 'config.json'),
                 'camera_order': keys, 'profile_fields': required,
                 'identity_note': 'Path/stat provenance only; no content digests computed'}


def model_sources(backend):
    sources = {}
    for name in ('torch', 'flash_rt', 'flash_rt.flash_rt_kernels'):
        module = sys.modules.get(name)
        if module is not None and getattr(module, '__file__', None):
            sources[name] = _file_record(module.__file__)
    for name, value in (('frontend', backend.frontend), ('pipeline', backend.frontend.pipe)):
        sources[name] = {'class': type(value).__module__ + '.' + type(value).__name__,
                         **_file_record(inspect.getfile(type(value)))}
    return sources


def latency_statistics(samples):
    return {'count': len(samples), 'p50_ms': float(np.percentile(samples, 50)),
            'p95_ms': float(np.percentile(samples, 95)), 'mean_ms': float(np.mean(samples)),
            'min_ms': float(np.min(samples)), 'max_ms': float(np.max(samples)),
            'std_ms': float(np.std(samples))}


def build_backend_config(args):
    from flash_rt.sampling import ProbeFlowConfig
    from flash_rt.vla import BackendConfig
    adaptive = (ProbeFlowConfig(epsilon=args.epsilon, probe_horizon=args.probe_horizon,
                               max_steps=args.max_steps, step_increment=args.step_increment)
                if args.sampler == 'probeflow' else None)
    return BackendConfig(model='hyvla', checkpoint=str(args.checkpoint.resolve()),
                         hardware='cuda_sm80', precision='bf16', framework='torch',
                         sampler=args.sampler, num_steps=args.num_steps, probeflow=adaptive,
                         options={'use_cuda_graph': args.use_graph,
                                  'numerical_profile': args.numerical_profile})


def run_candidate(args):
    # Defense in depth for callers invoking this function instead of main().
    require_slurm_step()
    arrays = load_fixture(args.fixture)
    fixture = fixture_record(args.fixture, args.manifest)
    cfg, checkpoint = checkpoint_record(args.checkpoint)
    observation = make_observation(arrays, cfg['image_features'])
    import torch
    from flash_rt.vla import describe_backend, load_backend

    gpu = require_cuda(torch)
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = build_backend_config(args)
    config_record = {'sampler': args.sampler, 'num_steps': args.num_steps,
                     'max_steps': args.max_steps, 'epsilon': args.epsilon,
                     'probe_horizon': args.probe_horizon, 'step_increment': args.step_increment,
                      'use_graph': args.use_graph, 'numerical_profile': args.numerical_profile}
    started = time.perf_counter()
    with torch.inference_mode(), load_backend(config) as backend:
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        sources = model_sources(backend)
        if sources['frontend']['class'] != 'flash_rt.frontends.torch.hyvla_probeflow.HyVLAFlowFrontend':
            raise RuntimeError('Unified backend did not construct the requested real HyVLA BF16 flow frontend')
        if any(getattr(backend.frontend, name) for name in
               ('use_fp8', 'use_fp8_vit', 'use_fused', 'use_fp4', 'use_fused_quant',
                'use_autotune', 'use_ffn_mega')):
            raise RuntimeError('Requested BF16 projection profile unexpectedly enabled quantized/fused kernels')
        if backend.frontend.numerical_profile != args.numerical_profile:
            raise RuntimeError('Frontend did not select the requested numerical profile')
        started = time.perf_counter()
        # prepare() performs a real discarded request and may capture graphs.
        backend.prepare({**observation, 'noise': arrays['noise'].copy()})
        torch.cuda.synchronize()
        prepare_seconds = time.perf_counter() - started
        graph_check, graph_outputs = {'status': 'not_requested', 'passed': None}, {}
        if args.check_graph_eager:
            free_bytes, _ = torch.cuda.mem_get_info()
            graph_check, graph_outputs = check_graph_eager(
                backend, observation, arrays['noise'], torch.cuda.synchronize,
                free_bytes=free_bytes, min_free_gib=args.graph_eager_min_free_gib,
                atol=args.atol, rtol=args.rtol)
        result = measure_requests(
            lambda noise: backend.infer(observation, noise=noise), torch.cuda.synchronize,
            arrays['noise'], warmup=args.warmup, repeats=args.repeats, atol=args.atol, rtol=args.rtol)
        for metadata in result['sampler_metadata']:
            if metadata.get('numerical_profile') != args.numerical_profile:
                raise ValueError('Actual sampler metadata does not match the requested numerical profile')
            validate_sampler_metadata(metadata, sampler=args.sampler, num_steps=args.num_steps,
                                      max_steps=args.max_steps, probe_horizon=args.probe_horizon,
                                      step_increment=args.step_increment)
        for metadata in result['backend_metadata']:
            if (not isinstance(metadata, dict) or metadata.get('model') != 'hyvla'
                    or metadata.get('precision') != 'bf16'
                    or metadata.get('action_domain') != 'normalized-padded-model-actions'):
                raise ValueError('Actual backend metadata does not describe raw HyVLA BF16 model actions')
        memory = {'allocated_bytes': torch.cuda.memory_allocated(),
                  'reserved_bytes': torch.cuda.memory_reserved(),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    oracle = oracle_comparison(result['actions32'], arrays,
                               sampler=args.sampler, num_steps=args.num_steps)
    runtime_ok = (result['fixed_noise_replay']['exact'] and result['sampler_metadata_replay_equal']
                  and graph_check['passed'] is not False)
    if not runtime_ok:
        status = 'runtime_check_failed'
    elif oracle['same_action_gate_passed'] is False:
        status = 'runtime_passed_oracle_mismatch'
    elif oracle['same_action_gate_passed'] is True:
        status = 'runtime_and_recorded_official_gate_passed'
    else:
        status = 'runtime_passed_official_parity_not_applicable'
    np.savez(args.output / 'actions.npz', actions32=result['actions32'],
             actions20=result['actions32'][:, :20], replay_actions32=result['replay_actions32'],
             latency_ms=result['latency_ms'], noise=arrays['noise'], input_state=arrays['state'],
             input_prompt=arrays['prompt'], oracle_actions32=arrays['expected_actions32'],
             oracle_actions20=arrays['expected_actions20'], **graph_outputs)
    record = {
        'schema_version': SCHEMA_VERSION, 'status': status, 'runtime_checks_passed': runtime_ok,
        'profile': PROFILES[args.numerical_profile], 'official_reference_profile': OFFICIAL_PROFILE,
        'config': config_record, 'backend_description': describe_backend(config),
        'checkpoint': checkpoint, 'fixture': fixture, 'model_sources': sources,
        'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
        'host': platform.node(), 'gpu': gpu,
        'runtime': {'python': platform.python_version(), 'executable': sys.executable,
                    'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__},
        'architecture_override_used': False, 'shape': [40, 32], 'dtype': 'float32', 'finite': True,
        'public_shape': [40, 20], 'action_domain': 'normalized-padded-model-actions',
        'fixed_noise_replay': result['fixed_noise_replay'],
        'sampler_metadata_replay_equal': result['sampler_metadata_replay_equal'],
        'sampler_metadata': result['sampler_metadata'], 'backend_metadata': result['backend_metadata'],
        'graph_eager_check': graph_check, 'official_oracle': oracle,
        'load_seconds_excluded': load_seconds, 'prepare_seconds_excluded': prepare_seconds,
        'warmup_requests_excluded': args.warmup, 'prepare_requests_excluded': 1,
        'latency_samples_ms': result['latency_ms'].tolist(),
        'latency_statistics': latency_statistics(result['latency_ms']),
        'latency_boundary': 'Synchronized backend.infer(raw CPU histories/state/prompt, fresh supplied noise) '
                            'through CPU actions; excludes load, prepare/capture, warmup, host noise copy, '
                            'post-return validation and artifact IO; NOT device-kernel-only timing',
        'memory': memory, 'artifacts': {'arrays': 'actions.npz'},
        'matched_engine_speedup_eligible': False,
        'scope': 'Real unified FlashRT backend fixture validation only; no official runtime executed here, '
                 'no matched end-to-end speedup or closed-loop task-success claim',
    }
    write_json(args.output / 'result.json', record)
    print('HYVLA_VALIDATION_RESULT', json.dumps(jsonable(record), allow_nan=False), flush=True)
    if not runtime_ok:
        return 1
    return 3 if args.require_oracle and oracle['same_action_gate_passed'] is not True else 0


def read_saved_run(path):
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    if path.is_file() and path.name not in ('actions.npz', 'result.json'):
        raise ValueError('Compare a runner output directory, result.json, or actions.npz with its sidecar')
    record = json.loads((folder / 'result.json').read_text())
    if record.get('schema_version') != SCHEMA_VERSION or record.get('runtime_checks_passed') is not True:
        raise ValueError('Comparison requires completed, runtime-validated runner artifacts')
    with np.load(folder / 'actions.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in
                  ('actions32', 'actions20', 'noise', 'input_state', 'input_prompt')}
    validate_actions(arrays['actions32'])
    if not np.array_equal(arrays['actions32'][:, :20], arrays['actions20']):
        raise ValueError('Saved public action slice is inconsistent')
    return record, arrays


def compare_saved_runs(left, right, *, atol, rtol, exact):
    a, aa = read_saved_run(left)
    b, bb = read_saved_run(right)
    if a.get('profile') != b.get('profile') or a.get('profile') not in PROFILES.values():
        raise ValueError('Cannot claim same-head parity across different numerical profiles')
    if not a.get('model_sources') or a.get('model_sources') != b.get('model_sources'):
        raise ValueError('Comparison model source/runtime origins differ (path/stat provenance)')
    for field in ('fixture', 'checkpoint'):
        if a.get(field) != b.get(field) or not a.get(field):
            raise ValueError(f'Comparison {field} provenance differs (path/stat, no content digests)')
    for field in ('noise', 'input_state', 'input_prompt'):
        if not np.array_equal(aa[field], bb[field]):
            raise ValueError(f'Comparison inputs differ: {field}')
    configs = (a['config'], b['config'])
    samplers = {config['sampler'] for config in configs}
    kind = 'numeric_difference_only'
    if samplers == {'euler', 'probeflow'}:
        euler = next(config for config in configs if config['sampler'] == 'euler')
        pf = next(config for config in configs if config['sampler'] == 'probeflow')
        if (euler['num_steps'] == 2 and pf['max_steps'] == 2
                and pf.get('probe_horizon') == 0.5 and euler['use_graph'] == pf['use_graph']):
            if (any(not record.get('sampler_metadata') for record in (a, b))
                    or any(row.get('nfe') != 2 or row.get('scheduled_steps') != 2
                           for record in (a, b) for row in record['sampler_metadata'])):
                raise ValueError('Euler-2/ProbeFlow-2 comparison lacks actual two-NFE evidence')
            kind = 'same_head_euler2_vs_probeflow2'
            if a['profile'] == PROFILES['official_local_fp32']:
                kind += '_different_integration_rounding'
    elif {config['use_graph'] for config in configs} == {True, False}:
        without_graph = [{key: value for key, value in config.items() if key != 'use_graph'}
                         for config in configs]
        if without_graph[0] == without_graph[1]:
            kind = 'same_head_graph_vs_eager_separate_processes'
    metrics = {name: difference(aa[name], bb[name], atol=atol, rtol=rtol)
               for name in ('actions32', 'actions20')}
    passed = all(row['exact' if exact else 'allclose'] for row in metrics.values())
    return {'schema_version': SCHEMA_VERSION,
            'status': 'comparison_passed' if passed else 'comparison_mismatch',
            'passed': passed, 'comparison_kind': kind, 'exact_required': exact,
            'left': str(Path(left).resolve()), 'right': str(Path(right).resolve()),
            'left_config': configs[0], 'right_config': configs[1],
            'comparison': metrics,
            'scope': 'CPU numerical comparison of same recorded raw inputs; not official parity, '
                     'not content-digest attestation, not task success or speedup'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--manifest', type=Path, help='Optional fixture manifest; sibling manifest.json is read by default')
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite an earlier run')
    parser.add_argument('--sampler', choices=('euler', 'probeflow'))
    parser.add_argument('--num-steps', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--numerical-profile', choices=tuple(PROFILES), default='semantic_bf16')
    graphs = parser.add_mutually_exclusive_group()
    graphs.add_argument('--use-graph', dest='use_graph', action='store_true')
    graphs.add_argument('--eager', dest='use_graph', action='store_false')
    parser.set_defaults(use_graph=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--epsilon', type=float, default=0.008)
    parser.add_argument('--probe-horizon', type=float, default=0.5)
    parser.add_argument('--step-increment', type=int, default=2)
    parser.add_argument('--check-graph-eager', action='store_true', help='Opt-in same-instance check; never load a second model')
    parser.add_argument('--graph-eager-min-free-gib', type=float, default=12.)
    parser.add_argument('--require-oracle', action='store_true', help='Euler-10 only: exit 3 on official recorded gate mismatch')
    parser.add_argument('--compare', nargs=2, metavar=('LEFT', 'RIGHT'), type=Path,
                        help='CPU-only compare of two result directories/NPZ files with JSON sidecars')
    parser.add_argument('--exact', action='store_true', help='CPU comparison must be exact, not just close')
    parser.add_argument('--atol', type=float, default=1e-6, help='Internal parity only; never changes the fixed official gate')
    parser.add_argument('--rtol', type=float, default=1e-6, help='Internal parity only; never changes the fixed official gate')
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        parser.error('atol/rtol must be finite and nonnegative')
    if args.compare:
        if any(value is not None for value in (args.checkpoint, args.fixture, args.sampler,
                                               args.num_steps, args.max_steps)):
            parser.error('--compare must not request model execution arguments')
        return args
    if args.checkpoint is None or args.fixture is None or args.sampler is None:
        parser.error('inference requires --checkpoint, --fixture and --sampler')
    if not 2 <= args.repeats <= 100 or not 1 <= args.warmup <= 20:
        parser.error('repeats must be in [2,100]; warmup in [1,20]')
    if args.sampler == 'euler':
        if args.max_steps is not None:
            parser.error('Euler uses --num-steps, not --max-steps')
        args.num_steps = 10 if args.num_steps is None else args.num_steps
        if not 1 <= args.num_steps <= 64:
            parser.error('num-steps must be in [1,64]')
    else:
        if args.num_steps is not None:
            parser.error('ProbeFlow uses --max-steps, not --num-steps')
        args.max_steps = 20 if args.max_steps is None else args.max_steps
        if not 2 <= args.max_steps <= 64:
            parser.error('max-steps must be in [2,64]')
    if (not math.isfinite(args.epsilon) or args.epsilon <= 0
            or not math.isfinite(args.probe_horizon) or not 0 < args.probe_horizon < 1
            or args.step_increment < 1):
        parser.error('Invalid ProbeFlow threshold, horizon or increment')
    if args.require_oracle and (args.sampler != 'euler' or args.num_steps != 10):
        parser.error('--require-oracle requires Euler num-steps=10')
    if args.check_graph_eager and not args.use_graph:
        parser.error('--check-graph-eager requires --use-graph')
    if not math.isfinite(args.graph_eager_min_free_gib) or args.graph_eager_min_free_gib <= 0:
        parser.error('graph-eager-min-free-gib must be finite and positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.compare:
        require_slurm_step()  # Must precede Torch/backend imports and even checkpoint reads.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.compare:
            record = compare_saved_runs(*args.compare, atol=args.atol, rtol=args.rtol, exact=args.exact)
            write_json(args.output / 'result.json', record)
            print('HYVLA_COMPARISON_RESULT', json.dumps(record, allow_nan=False), flush=True)
            return 0 if record['passed'] else 1
        return run_candidate(args)
    except Exception as error:
        write_json(args.output / 'failure.json', {
            'schema_version': SCHEMA_VERSION, 'status': 'failed', 'runtime_checks_passed': False,
            'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(),
            'job_id': os.environ.get('SLURM_JOB_ID'), 'step_id': os.environ.get('SLURM_STEP_ID'),
            'profile': PROFILE, 'arguments': jsonable(vars(args)),
        })
        raise


if __name__ == '__main__':
    raise SystemExit(main())
