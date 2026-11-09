"""Allocation-only real FastWAM validation; see the adjacent examples README.

No Torch/upstream imports occur before the Slurm guard. Each invocation owns at
most one model: T5-only preparation, upstream action-only reference, or unified
FlashRT backend. This is an execution/numerical fixture, NOT a task-success test.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
import traceback
import zipfile


COMMIT = '7faa71108368fbb3b6885649f112af607427a2d4'
HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = 'checkpoints/FastWAM/libero_uncond_2cam224.pt'
DEFAULT_BACKBONE = 'checkpoints/Wan2.2-TI2V-5B'
DEFAULT_TASK_ROOT = str(Path.cwd() / 'vla-runs')


def require_allocation(environ=None, hostname=None):
    """Reject login hosts, copied allocation variables, and CPU-only jobs.

    Environment checks are an accident-prevention boundary, not authentication.
    CUDA availability is checked separately, only after this guard passes.
    """
    env = os.environ if environ is None else environ
    host = (hostname or socket.gethostname()).split('.')[0]
    node = env.get('SLURMD_NODENAME', '').split('.')[0]
    job = env.get('SLURM_JOB_ID', '')
    if not job.isdigit() or not node or host != node or host.lower().startswith(('ln', 'login')):
        raise RuntimeError('Requires SLURM_JOB_ID on its SLURMD_NODENAME compute host; never run on login')
    allocated_ids = env.get('SLURM_JOB_GPUS') or env.get('SLURM_STEP_GPUS')
    count = env.get('SLURM_GPUS_ON_NODE', '')
    if not allocated_ids and not (count.isdigit() and int(count) > 0):
        raise RuntimeError('Requires Slurm GPU-allocation metadata (SLURM_JOB_GPUS/STEP_GPUS/GPUS_ON_NODE)')
    return {'job_id': job, 'node': node, 'step_id': env.get('SLURM_STEP_ID'),
            'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES')}


def validate_source_manifest(source):
    source = Path(source).resolve()
    manifest = json.loads((source.parent / 'SOURCE.json').read_text())
    if manifest.get('commit') != COMMIT:
        raise ValueError('Source manifest commit differs from the pinned contract')
    if manifest.get('source_subdirectory') != source.name:
        raise ValueError('Source directory differs from its pinned manifest')
    return manifest


def resolved_model_config(source, backbone, config_path=None):
    source = Path(source).resolve()
    validate_source_manifest(source)
    if not (source / 'src/fastwam/models/wan22/fastwam.py').is_file():
        raise FileNotFoundError(f'Pinned source checkout missing: {source}')
    path = Path(config_path) if config_path is not None else HERE / 'fastwam-source/libero_model_resolved.json'
    config = json.loads(path.read_text())
    config['model_id'] = config['tokenizer_model_id'] = str(Path(backbone).resolve())
    return config


def validate_fixture(image, proprio, context, mask, noise):
    import numpy as np
    for name, value, shape in [('input_image', image, (1, 3, 224, 448)),
            ('proprio', proprio, (1, 8)), ('noise', noise, (1, 32, 7))]:
        if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite floating {shape}; got {value.shape}/{value.dtype}')
    if image.min() < -1 or image.max() > 1:
        raise ValueError('input_image must be caller-preprocessed [-1,1], not raw RGB')
    if (context.ndim != 3 or context.shape[0] != 1 or context.shape[2] != 4096
            or not 0 < context.shape[1] <= 128 or context.dtype.kind != 'f'
            or not np.isfinite(context).all()):
        raise ValueError('context must be finite floating [1,L<=128,4096]')
    if mask.shape != context.shape[:2] or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('context_mask must be bool [1,L] with at least one valid token')


def action_metrics(left, right):
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != (1, 32, 7) or right.shape != left.shape
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError('Action comparison requires finite [1,32,7] arrays')
    a, b = left.astype(np.float64), right.astype(np.float64)
    delta = a - b
    denominator = float(np.linalg.norm(a))
    return {'bitwise_equal': bool(np.array_equal(left, right)),
            'max_abs': float(np.abs(delta).max()), 'mean_abs': float(np.abs(delta).mean()),
            'rmse': float(np.sqrt(np.mean(delta ** 2))),
            'relative_l2_vs_left': float(np.linalg.norm(delta) / denominator) if denominator else None}


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def local_assets(args, *, text_only=False):
    required = [args.source_dir / 'src/fastwam/models/wan22/fastwam.py']
    if text_only:
        required.extend([args.backbone_dir / 'models_t5_umt5-xxl-enc-bf16.pth',
                         args.backbone_dir / 'google/umt5-xxl/tokenizer_config.json',
                         args.backbone_dir / 'google/umt5-xxl/spiece.model'])
    else:
        required.extend([args.model_config, args.checkpoint, args.backbone_dir / 'Wan2.2_VAE.pth', args.context])
    metadata = []
    for path in required:
        if path is None or not path.is_file():
            raise FileNotFoundError(f'Required local asset missing: {path}; downloads are disabled')
        metadata.append({'path': str(path.resolve()), 'bytes': path.stat().st_size,
                         'mtime_ns': path.stat().st_mtime_ns})
    return metadata


def runtime_torch():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated process has no CUDA GPU')
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires BF16-capable allocated CUDA device')
    return torch


def prepare_context(args, report):
    """Exactly upstream encode_prompt semantics, without constructing VAE/DiTs."""
    import numpy as np
    torch = runtime_torch()
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

    checkpoint = args.backbone_dir / 'models_t5_umt5-xxl-enc-bf16.pth'
    # Meta construction avoids a roughly 22 GB FP32 random initialization.
    # mmap is used only when the resident checkpoint has PyTorch's ZIP format.
    with torch.device('meta'):
        encoder = WanTextEncoder()
    use_mmap = zipfile.is_zipfile(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=use_mmap)
    for key in ('state_dict', 'module', 'model_state'):
        if isinstance(state, Mapping) and len(state) == 1 and key in state:
            state = state[key]
            break
    encoder.load_state_dict(state, strict=True, assign=True)
    del state
    encoder.eval().to(device='cuda:0', dtype=torch.bfloat16)
    tokenizer = HuggingfaceTokenizer(name=str(args.backbone_dir / 'google/umt5-xxl'),
                                    seq_len=128, clean='whitespace', local_files_only=True)
    with torch.inference_mode():
        ids, original_mask = tokenizer(args.prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to('cuda:0')
        original_mask = original_mask.to('cuda:0', dtype=torch.bool)
        embedding = encoder(ids, original_mask)
        lengths = original_mask.gt(0).sum(dim=1).long()
        for index, length in enumerate(lengths):
            embedding[index, length:] = 0
        # Deliberate pinned upstream behavior: zero padding remains visible.
        mask = torch.ones_like(original_mask)
        context = embedding.float().cpu().numpy()
        context_mask = mask.cpu().numpy()
        np.savez(args.output / 'context.npz', context=context, context_mask=context_mask,
                 original_attention_mask=original_mask.cpu().numpy(), input_ids=ids.cpu().numpy(),
                 prompt=np.asarray(args.prompt), source_commit=np.asarray(COMMIT),
                 context_semantics=np.asarray('upstream_encode_prompt_zero_pad_all_true_mask'))
    report.update(status='prepared_context', prompt=args.prompt, t5_state_strict=True,
                  t5_meta_init=True, t5_mmap=use_mmap,
                  context_shape=list(context.shape), context_mask_shape=list(context_mask.shape),
                  context_finite=bool(np.isfinite(context).all()),
                  artifact=str(args.output / 'context.npz'))
    if not report['context_finite']:
        raise RuntimeError('Prepared context is nonfinite')
    return torch


def load_fixture(args, torch):
    import numpy as np
    with np.load(args.context, allow_pickle=False) as saved:
        context, mask = saved['context'].copy(), saved['context_mask'].copy()
        context_label = str(saved['prompt'].item()) if 'prompt' in saved else 'prepared_context_prompt_unknown'
        context_semantics = str(saved['context_semantics'].item()) if 'context_semantics' in saved else 'external_cache_semantics_unverified'
    if context.ndim == 2:
        context = context[None]
    if mask.ndim == 1:
        mask = mask[None]
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    noise = torch.randn((1, 32, 7), generator=generator, device='cpu', dtype=torch.float32).numpy()
    proprio = np.zeros((1, 8), np.float32)
    if args.input_fixture:
        with np.load(args.input_fixture, allow_pickle=False) as fixture:
            image = fixture['input_image'].copy()
            if 'proprio' in fixture:
                proprio = fixture['proprio'].copy()
            if 'noise' in fixture:
                provided_noise = fixture['noise']
                if not np.array_equal(provided_noise, noise):
                    raise ValueError('Fixture noise differs from explicit torch CPU seed; upstream reference cannot inject arbitrary noise')
        kind = 'provided_preprocessed_image_provenance_unverified'
        image_source = str(args.input_fixture.resolve())
    else:
        # Fixed, camera-distinct analytical RGB patterns; not a simulated scene.
        y, x = np.meshgrid(np.linspace(-1, 1, 224, dtype=np.float32),
                           np.linspace(-1, 1, 224, dtype=np.float32), indexing='ij')
        left = np.stack([x, y, (x + y) / 2])
        right = np.stack([-y, -x, (x - y) / 2])
        image = np.concatenate([left, right], axis=2)[None]
        kind = 'deterministic_synthetic_2cam_execution_fixture_not_task_validation'
        image_source = 'analytical_gradient_patterns_no_recorded_libero_image_found'
    if image.ndim == 3:
        image = image[None]
    if proprio.ndim == 1:
        proprio = proprio[None]
    validate_fixture(image, proprio, context, mask, noise)
    arrays = dict(input_image=image, proprio=proprio, context=context,
                  context_mask=mask, noise=noise)
    labels = dict(fixture_kind=kind, image_source=image_source,
                  context_prompt=context_label, context_semantics=context_semantics,
                  proprio_semantics='checkpoint_normalized_input_not_robot_state_validation')
    return arrays, labels


def load_upstream_reference(args, config, torch):
    """No FlashRT frontend or sampler in this reference inference path."""
    from fastwam.models.wan22.fastwam import FastWAM
    kwargs = {key: config[key] for key in ('video_dit_config', 'action_dit_config', 'proprio_dim')}
    kwargs.update(device='cpu', torch_dtype=torch.bfloat16, model_id=str(args.backbone_dir),
                  load_text_encoder=False, redirect_common_files=False,
                  skip_dit_load_from_pretrain=True, action_dit_pretrained_path=None,
                  mot_checkpoint_mixed_attn=False, compile_training_denoise=False)
    for kind in ('video', 'action'):
        for setting in ('train_shift', 'infer_shift', 'num_train_timesteps'):
            kwargs[f'{kind}_{setting}'] = config[f'{kind}_scheduler'][setting]
    model = FastWAM.from_wan22_pretrained(**kwargs)
    payload = model.load_checkpoint(str(args.checkpoint))
    if not isinstance(payload, Mapping) or not isinstance(payload.get('mot'), Mapping):
        raise ValueError('Released full MoT checkpoint required; no legacy video-only checkpoint')
    expected = model.mot.state_dict()
    if set(expected) != set(payload['mot']):
        raise ValueError('Full MoT checkpoint keys do not match resolved architecture; no random leftovers allowed')
    for key, value in expected.items():
        if payload['mot'][key].shape != value.shape:
            raise ValueError(f'MoT checkpoint shape mismatch: {key}')
    if not isinstance(payload.get('proprio_encoder'), Mapping):
        raise ValueError('Released checkpoint lacks required proprio_encoder weights')
    model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
    del expected, payload
    model.eval().to(device='cuda:0', dtype=torch.bfloat16)
    model.device, model.torch_dtype = torch.device('cuda:0'), torch.bfloat16
    return model


def instrument_model(model):
    """Count actual denoiser evaluations and video prefills, not requested NFE."""
    counters = {'nfe': 0, 'video_prefill': 0}
    denoise = model._denoise_action_with_video_cache
    prefill = model.mot.prefill_video_cache_tensor

    def counted_denoise(*args, **kwargs):
        counters['nfe'] += 1
        return denoise(*args, **kwargs)

    def counted_prefill(*args, **kwargs):
        counters['video_prefill'] += 1
        return prefill(*args, **kwargs)

    model._denoise_action_with_video_cache = counted_denoise
    model.mot.prefill_video_cache_tensor = counted_prefill
    return counters


def timed_infer(infer, counters, torch):
    before = dict(counters)
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    with torch.inference_mode():
        result = infer()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    counted = {key: value - before[key] for key, value in counters.items()}
    actions = result['actions']
    action_metrics(actions, actions)
    if counted['video_prefill'] != 1:
        raise RuntimeError(f'Expected exactly one video prefill per request, got {counted}')
    if counted['nfe'] != result['sampler_metadata']['nfe']:
        raise RuntimeError(f'Reported NFE differs from real model calls: {counted}')
    return result, elapsed_ms, counted


def build_backend_config(args, model_config):
    from flash_rt.sampling import ProbeFlowConfig
    from flash_rt.vla import BackendConfig
    probe = ProbeFlowConfig(epsilon=args.pf_epsilon, max_steps=args.pf_max) if args.sampler == 'probeflow' else None
    return BackendConfig(model='fastwam', checkpoint=str(args.checkpoint),
        sampler=args.sampler, precision='bf16', framework='torch', hardware='cuda_sm80',
        num_steps=2 if args.sampler == 'euler' else None, probeflow=probe,
        options={'model_config': model_config, 'backbone_dir': str(args.backbone_dir),
                 'source_dir': str(args.source_dir), 'sigma_shift': 5., 'action_horizon': 32,
                 'native_linear_backend': args.native_linear_backend})


def run_actions(args, report):
    import numpy as np
    torch = runtime_torch()
    arrays, labels = load_fixture(args, torch)
    config = resolved_model_config(args.source_dir, args.backbone_dir, args.model_config)
    report.update(labels, model_config=config, noise_seed=args.seed,
                  noise_generator='torch.Generator(device=cpu), float32 randn then bf16 cast',
                  input_shapes={key: list(value.shape) for key, value in arrays.items()},
                  checkpoint_full_state_required=True, sigma_shift=5., action_horizon=32)
    write_json(args.output / 'model_config_resolved.json', config)
    obs = {key: value for key, value in arrays.items() if key != 'noise'}
    owner = None
    load_start = time.perf_counter_ns()
    if args.reference:
        model = load_upstream_reference(args, config, torch)
        # CPU tensors preserve the same observation-upload boundary as unified infer.
        reference_obs = {key: torch.from_numpy(value) for key, value in obs.items()}
        schedule_t, schedule_d = model.infer_action_scheduler.build_inference_schedule(
            2, device=torch.device('cuda:0'), dtype=torch.bfloat16, shift_override=5.)
        reference_metadata = {'sampler': 'euler', 'scheduled_steps': 2, 'nfe': 2,
                              'model_timesteps': schedule_t.float().cpu().tolist(),
                              'sigma_deltas': schedule_d.float().cpu().tolist(),
                              'sigma_shift': 5., 'action_infer_mode': 'first_frame'}

        def infer():
            result = model.infer_action(prompt=None, action_horizon=32, num_inference_steps=2,
                                        sigma_shift=5., seed=args.seed, rand_device='cpu',
                                        tiled=False, compile_action_infer=False, **reference_obs)
            return {'actions': result['action'].numpy()[None],
                    'sampler_metadata': reference_metadata,
                    'execution_metadata': {'backend': 'direct_pinned_upstream_infer_action',
                                           'native_linear_modules': [], 'fully_native': False}}
    else:
        from flash_rt.vla import load_backend
        backend_config = build_backend_config(args, config)
        owner = load_backend(backend_config)
        model = owner.frontend.model

        def infer():
            return owner.infer(obs, noise=arrays['noise'])

    torch.cuda.synchronize()
    report['load_ms'] = (time.perf_counter_ns() - load_start) / 1e6
    counters = instrument_model(model)
    torch.cuda.reset_peak_memory_stats()
    first, first_ms, first_counts = timed_infer(infer, counters, torch)
    for _ in range(args.warmups):
        timed_infer(infer, counters, torch)
    samples, results, counts = [], [], []
    for _ in range(args.repeats):
        result, elapsed, counted = timed_infer(infer, counters, torch)
        samples.append(elapsed)
        results.append(result)
        counts.append(counted)
    replay, replay_ms, replay_counts = timed_infer(infer, counters, torch)
    baseline = results[0]['actions']
    replay_metrics = [action_metrics(baseline, result['actions']) for result in [first, *results[1:], replay]]
    replay_ok = all(np.allclose(baseline, result['actions'], atol=args.replay_atol,
                               rtol=args.replay_rtol, equal_nan=False)
                    for result in [first, *results[1:], replay])
    if not args.reference:
        for result in [first, *results, replay]:
            coverage = result['execution_metadata']
            if (coverage.get('backend') != 'flashrt_bf16_linear_partial'
                    or coverage.get('native_linear_backend') != args.native_linear_backend
                    or not coverage.get('native_linear_modules')
                    or not any(coverage.get('native_linear_calls', {}).values())):
                raise RuntimeError('Unified backend did not actually execute native linears; no silent fallback accepted')
    for result in [first, *results, replay]:
        metadata = result['sampler_metadata']
        if (args.sampler == 'euler' or args.pf_max == 2) and metadata['nfe'] != 2:
            raise RuntimeError('Euler2/PF2 candidate executed a different NFE')
    np.savez(args.output / 'actions.npz', **arrays, actions=baseline,
             repeat_actions=np.stack([result['actions'] for result in results]),
             first_actions=first['actions'], replay_actions=replay['actions'],
             latency_samples_ms=np.asarray(samples, dtype=np.float64))
    report.update(status='passed_execution_checks' if replay_ok else 'failed_replay',
        action_shape=list(baseline.shape), actions_finite=True, actions_normalized=True,
        first_call_ms=first_ms, first_call_counters=first_counts,
        warmups=args.warmups, repeats=args.repeats, latency_samples_ms=samples,
        latency_median_ms=float(np.median(samples)), latency_p95_ms=float(np.percentile(samples, 95)),
        latency_scope='synchronized_whole_infer_including_input_validation_upload_VAE_video_prefill_action_and_cpu_output',
        replay_ms=replay_ms, replay_counters=replay_counts,
        replay_metrics=replay_metrics, replay_allclose=bool(replay_ok),
        replay_atol=args.replay_atol, replay_rtol=args.replay_rtol,
        measured_call_counters=counts,
        sampler_metadata=[result['sampler_metadata'] for result in results],
        measured_execution_metadata=[result['execution_metadata'] for result in results],
        first_execution_metadata=first['execution_metadata'],
        execution_metadata=results[0]['execution_metadata'],
        backend_metadata=results[0].get('backend_metadata'),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        task_success_evaluated=False, no_video_rollout=True)
    if owner is not None:
        owner.close()
    if not replay_ok:
        raise RuntimeError('Fixed-input fixed-noise replay exceeded explicitly configured tolerances')
    return torch


def compare_results(args, report):
    import numpy as np
    left_path, right_path = args.compare
    left_report = json.loads((left_path / 'result.json').read_text())
    right_report = json.loads((right_path / 'result.json').read_text())
    for key in ('source_contract_commit', 'model_config', 'noise_seed', 'sigma_shift', 'action_horizon'):
        if left_report[key] != right_report[key]:
            raise ValueError(f'Cannot compare different {key}')
    if left_report['status'] != 'passed_execution_checks' or right_report['status'] != 'passed_execution_checks':
        raise ValueError('Cannot compare failed/unverified candidate runs')
    for label, candidate in [('left', left_report), ('right', right_report)]:
        if candidate.get('task_success_evaluated') is not False:
            raise ValueError(f'{label} report lacks explicit execution-only semantics')
    left_ckpt = [v for v in left_report['assets'] if v['path'] == left_report['checkpoint']]
    right_ckpt = [v for v in right_report['assets'] if v['path'] == right_report['checkpoint']]
    if not left_ckpt or left_ckpt != right_ckpt:
        raise ValueError('Checkpoint path/size/mtime differs across candidates')
    with np.load(left_path / 'actions.npz', allow_pickle=False) as left, np.load(right_path / 'actions.npz', allow_pickle=False) as right:
        for key in ('input_image', 'proprio', 'context', 'context_mask', 'noise'):
            if left[key].dtype != right[key].dtype or not np.array_equal(left[key], right[key]):
                raise ValueError(f'Comparison input differs: {key}')
        metrics = action_metrics(left['actions'], right['actions'])
        if args.compare_atol is not None:
            metrics['allclose'] = bool(np.allclose(left['actions'], right['actions'],
                atol=args.compare_atol, rtol=args.compare_rtol, equal_nan=False))
    report.update(status='compared_execution_outputs', left=str(left_path), right=str(right_path),
        metrics=metrics, compare_atol=args.compare_atol, compare_rtol=args.compare_rtol,
        threshold_note='No numerical correctness threshold assumed; optional explicit allclose is only a supplied criterion',
        left_sampler=left_report['sampler_metadata'], right_sampler=right_report['sampler_metadata'],
        left_execution=left_report['execution_metadata']['backend'],
        right_execution=right_report['execution_metadata']['backend'],
        task_success_evaluated=False, approximate_sampler_equality_not_required=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-context', action='store_true')
    modes.add_argument('--run', action='store_true', help='Unified native-linear BackendConfig candidate')
    modes.add_argument('--reference', action='store_true', help='Direct pinned upstream infer_action Euler2')
    modes.add_argument('--compare', nargs=2, type=Path, metavar=('LEFT_DIR', 'RIGHT_DIR'))
    p.add_argument('--source-dir', type=Path, default=HERE / 'fastwam-source/upstream')
    p.add_argument('--model-config', type=Path, default=HERE / 'fastwam-source/libero_model_resolved.json',
                   help='Explicit resolved architecture JSON; required sidecar checked before loading any model')
    p.add_argument('--checkpoint', type=Path, default=Path(DEFAULT_CHECKPOINT))
    p.add_argument('--backbone-dir', type=Path, default=Path(DEFAULT_BACKBONE))
    p.add_argument('--context', type=Path, help='Offline NPZ with context and bool context_mask')
    p.add_argument('--input-fixture', type=Path, help='Optional already preprocessed NPZ input_image/proprio/noise')
    p.add_argument('--prompt', default='pick up the black bowl between the plate and the ramekin and place it on the plate')
    p.add_argument('--sampler', choices=('euler', 'probeflow'), default='euler')
    p.add_argument('--native-linear-backend', choices=('legacy', 'cached'), default='legacy')
    p.add_argument('--pf-max', type=int, default=2, help='2 means fixed PF2; >2 is explicitly adaptive')
    p.add_argument('--pf-epsilon', type=float, default=.008)
    p.add_argument('--seed', type=int, default=20261109)
    p.add_argument('--warmups', type=int, default=1, help='Additional excluded warmups after separately timed first call')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--replay-atol', type=float, default=0.)
    p.add_argument('--replay-rtol', type=float, default=0.)
    p.add_argument('--compare-atol', type=float, help='Optional caller-selected tolerance, no implied task quality')
    p.add_argument('--compare-rtol', type=float, default=0.)
    p.add_argument('--output', required=True, type=Path, help='New result directory under VLA_TASK_ROOT')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    allocation = require_allocation()
    if args.repeats < 1 or args.warmups < 0 or not 2 <= args.pf_max <= 64:
        raise ValueError('repeats>=1, warmups>=0 and 2<=pf-max<=64 required')
    import math
    for value in (args.replay_atol, args.replay_rtol, args.compare_rtol, args.compare_atol):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError('Tolerances must be finite and nonnegative')
    if not math.isfinite(args.pf_epsilon) or args.pf_epsilon <= 0:
        raise ValueError('pf-epsilon must be finite and positive')
    if args.reference and args.sampler != 'euler':
        raise ValueError('Direct upstream reference is Euler2 only')
    if (args.run or args.reference) and args.context is None:
        raise ValueError('--context is required; encode T5 in a separate allocation process first')
    for key in ('source_dir', 'model_config', 'checkpoint', 'backbone_dir', 'context', 'input_fixture'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    args.output = args.output.expanduser().resolve()
    task_root = Path(os.environ.get('VLA_TASK_ROOT', DEFAULT_TASK_ROOT)).resolve()
    if not args.output.is_relative_to(task_root) or args.output == task_root:
        raise ValueError('Output must be a new isolated subdirectory under VLA_TASK_ROOT')
    # Existing results are never overwritten, even on a rerun in the same job.
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', allocation=allocation, source_contract_commit=COMMIT,
                  source_dir=str(args.source_dir), checkpoint=str(args.checkpoint),
                  task_success_evaluated=False, argv=sys.argv if argv is None else list(argv))
    try:
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
            os.environ[name] = '1'
        os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'true'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        sys.dont_write_bytecode = True
        # Prepend only our pinned source; do not mutate existing checkouts/envs.
        sys.path.insert(0, str(args.source_dir / 'src'))
        sys.path.insert(0, str(HERE.parents[1]))
        if args.compare:
            compare_results(args, report)
            torch = None
        else:
            report['source_provenance'] = validate_source_manifest(args.source_dir)
            report['assets'] = local_assets(args, text_only=args.prepare_context)
            torch = prepare_context(args, report) if args.prepare_context else run_actions(args, report)
        if torch is not None:
            report['runtime'] = {'python': sys.version, 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                'capability': list(torch.cuda.get_device_capability(0)),
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}
        report['cpu_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(args.output / 'result.json', report)
        print(json.dumps({'status': report['status'], 'result': str(args.output / 'result.json')}), flush=True)
        return 0
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(args.output / 'result.json', report)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
