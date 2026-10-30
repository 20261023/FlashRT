"""Deterministic offline UR5 episode replay; no robot, client or HTTP connection.

Primary timing: host RGB224 + raw UR5 state7 + shared bf16 noise -> completed
host public actions30x7. Includes tokenization, transfers, ProbeFlow routing,
both probe evaluations and public action decoding. Preparation/capture/warmup
and writing measurement files are outside steady-state statistics.
"""
import argparse
import ast
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from flash_rt.models.pi05.probeflow import ProbeFlowConfig, select_schedule


def bf16_round(values):
    array = np.asarray(values, np.float32)
    if not np.isfinite(array).all():
        raise ValueError('BF16 input must be finite')
    bits = array.reshape(-1).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32).reshape(array.shape).copy()


def model_state(raw):
    result = np.asarray(raw, np.float32).copy()
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError('UR5 raw state must contain seven finite values')
    result[6] = 3 + (0.085 - result[6]) * (226 / 0.085)
    return result


def decode_ur5_actions(raw, state, stats):
    raw = np.asarray(raw, np.float32)
    if raw.shape != (50, 32) or not np.isfinite(raw).all():
        raise ValueError('Expected finite normalized actions50x32')
    low, high = (np.asarray(stats['actions'][q], np.float32) for q in ('q01', 'q99'))
    actions = (raw[:, :7] + 1.) / 2. * (high - low + 1e-6) + low
    actions[:, :6] += state[:6]
    actions[:, 6] = 0.085 - (actions[:, 6] - 3) * 0.085 / 226
    return np.ascontiguousarray(actions[:30], np.float32)


def latency_statistics(records):
    values = np.asarray([row['run_policy_ms'] for row in records if not row['exclude']], np.float64)
    if not values.size or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('No valid finite positive steady latency samples')
    return {'count': len(values), 'p50_ms': float(np.percentile(values, 50)),
            'p99_ms': float(np.percentile(values, 99)), 'mean_ms': float(values.mean()),
            'min_ms': float(values.min()), 'max_ms': float(values.max())}


def gpu_parameter_bytes(leaves):
    """Inspect Array.devices(), including NamedSharding, without reading values."""
    total = 0
    for value in leaves:
        devices_method = getattr(value, 'devices', None)
        if not callable(devices_method):
            continue
        devices = devices_method()
        if devices and all(device.platform == 'gpu' for device in devices):
            total += value.size * value.dtype.itemsize
    return int(total)


def reference_bf16_probeflow(noise, evaluate, config):
    """Independent NumPy integration, rounding state after every BF16 update.

    ``evaluate(x, model_time)`` evaluates the actual native neural field and
    returns progress velocity -v. No Torch sampler is called by this oracle.
    """
    initial = bf16_round(noise)
    times = []

    def field(x, t):
        velocity = np.asarray(evaluate(x.copy(), t), np.float32)
        if velocity.shape != initial.shape or not np.isfinite(velocity).all():
            raise ValueError('Invalid reference velocity')
        times.append(t)
        return velocity

    first = field(initial, 1.)
    probe = bf16_round(initial + first * config.probe_horizon)
    lookahead = field(probe, 1. - config.probe_horizon)
    a, b = first.ravel().astype(np.float64), lookahead.ravel().astype(np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / norm) if norm else 0.
    count = select_schedule(cosine, config)
    if count == 2:
        result = bf16_round(probe + lookahead * (1. - config.probe_horizon))
    else:
        result = bf16_round(initial + first / count)
        for index in range(1, count):
            result = bf16_round(result + field(result, 1. - index / count) / count)
    return result, {'scheduled_steps': count, 'nfe': len(times), 'cosine': cosine,
                    'timesteps': times}


def _json_write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _tokenizer(source, model):
    import sentencepiece
    node = next(node for node in ast.parse(Path(source).read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == 'PaligemmaTokenizer')
    scope = {'np': np, 'logging': logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    result = scope['PaligemmaTokenizer'].__new__(scope['PaligemmaTokenizer'])
    result._max_len = 200
    result._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(model))
    return result


def _stats(checkpoint, asset_id):
    result = json.loads((Path(checkpoint) / 'assets' / asset_id / 'norm_stats.json').read_text())['norm_stats']
    for kind in ('state', 'actions'):
        for q in ('q01', 'q99'):
            value = np.asarray(result[kind][q])
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError('Checkpoint must provide finite UR5 state/action quantiles7')
    return result


def _normalized_state(state, stats):
    # Match openpi_rtc Normalize: JSON quantiles remain float64 until AFTER
    # state tokenization. A premature float32 cast changes bins near -1/+1.
    low, high = (np.asarray(stats['state'][q], np.float64) for q in ('q01', 'q99'))
    return (state - low) / (high - low + 1e-6) * 2 - 1


def _recorded_image(path, images, caption):
    from PIL import Image, ImageDraw
    width, height = images[0].shape[1], images[0].shape[0]
    canvas = Image.new('RGB', (width * 2, height + 50), 'white')
    for index, image in enumerate(images):
        canvas.paste(Image.fromarray(image), (index * width, 50))
    ImageDraw.Draw(canvas).text((10, 10), caption, fill='black')
    canvas.save(path)


def prepare(args):
    import cv2
    cv2.setNumThreads(1)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    states = [json.loads(line) for line in (args.episode / 'states/states.jsonl').read_text().splitlines()
              if line.strip()]
    if len(states) < 2:
        raise ValueError('Recorded episode must contain at least two frames')
    stats = _stats(args.checkpoint, args.asset_id)
    tokenizer = _tokenizer(args.tokenizer_source, args.tokenizer)
    image_path = args.openpi_dir / 'packages/openpi-client/src/openpi_client/image_tools.py'
    spec = importlib.util.spec_from_file_location('_readonly_openpi_image_tools', image_path)
    image_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_tools)
    count = len(states)
    rgb = np.lib.format.open_memmap(output / 'rgb.npy', mode='w+', dtype=np.uint8,
                                  shape=(count, 2, 224, 224, 3))
    noises = np.lib.format.open_memmap(output / 'noise.npy', mode='w+', dtype=np.float32,
                                     shape=(count, 50, 32))
    raw_states = np.empty((count, 7), np.float32)
    tokens = np.empty((count, 200), np.int32)
    masks = np.empty((count, 200), bool)
    captures = [cv2.VideoCapture(str(args.episode / f'videos/{camera}_rgb.mp4'))
                for camera in ('cam_global', 'cam_arm')]
    try:
        if any(not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != count for cap in captures):
            raise ValueError('Video/state frame counts disagree or video cannot be opened')
        for index, record in enumerate(states):
            pair = []
            for cap in captures:
                good, bgr = cap.read()
                if not good or bgr is None:
                    raise ValueError(f'Recorded video ended early at frame {index}')
                pair.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            rgb[index] = [image_tools.resize_with_pad(image, 224, 224) for image in pair]
            raw_states[index] = [*record['joint_positions'], float(record['gripper_width'])]
            state = model_state(raw_states[index])
            tokens[index], masks[index] = tokenizer.tokenize(args.prompt, _normalized_state(state, stats))
            noises[index] = bf16_round(np.random.default_rng(args.seed + index).standard_normal((50, 32)).astype(np.float32))
            if index in (0, count - 1):
                label = 'first' if index == 0 else 'last'
                _recorded_image(output / f'recorded-{label}.png', pair,
                                f'RECORDED INPUT / REPLAY - {label} frame {index}; not a closed-loop result')
        if any(cap.read()[0] for cap in captures):
            raise ValueError('Recorded video has additional frames beyond states')
    finally:
        for cap in captures:
            cap.release()
    rgb.flush()
    noises.flush()
    np.save(output / 'state_raw7.npy', raw_states)
    np.save(output / 'token_ids.npy', tokens)
    np.save(output / 'token_mask.npy', masks)
    manifest = {'status': 'prepared', 'frame_count': count, 'episode': str(args.episode.resolve()),
                'checkpoint': str(args.checkpoint.resolve()), 'asset_id': args.asset_id,
                'prompt': args.prompt, 'noise_seed_base': args.seed,
                'input': 'shared RGB224 uint8, raw state7; identical BF16-representable noise50x32',
                'tokenizer': str(args.tokenizer), 'tokenizer_source': str(args.tokenizer_source),
                'openpi_dir': str(args.openpi_dir), 'recorded_only': True,
                'state_quantile_precision': 'float64, matching original Normalize before tokenization'}
    _json_write(output / 'manifest.json', manifest)
    print('EPISODE_PREPARED', json.dumps(manifest), flush=True)


@contextmanager
def _canonical_tokens(ids):
    from flash_rt.frontends.torch import pi05_rtx
    original = pi05_rtx._prompt_token_ids
    pi05_rtx._prompt_token_ids = lambda *args, **kwargs: [int(value) for value in ids]
    try:
        yield
    finally:
        pi05_rtx._prompt_token_ids = original


class RawRecordedPolicy:
    def __init__(self, dataset, manifest, stats):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi_rtc.training.config import get_config
        from openpi_rtc.policies.policy_config import create_trained_policy
        from openpi_rtc import transforms
        from openpi_rtc.shared import normalize
        if jax.default_backend() != 'gpu' or not any(device.platform == 'gpu' for device in jax.devices()):
            raise RuntimeError('Raw baseline must execute on JAX GPU, not the CPU Orbax-loader environment')
        norm = normalize.load(Path(manifest['checkpoint']) / 'assets' / manifest['asset_id'])
        self.policy = create_trained_policy(
            get_config('cvpr_multitask_ur5_rtc'), manifest['checkpoint'],
            repack_transforms=transforms.Group(), norm_stats=norm, sample_kwargs={'num_steps': 10})
        self.jax, self.jnp = jax, jnp
        self.dataset, self.manifest, self.stats = dataset, manifest, stats
        self.raw = None
        original_output = self.policy._output_transform

        def output(data):
            self.raw = np.asarray(data['actions'], np.float32).copy()
            return original_output(data)

        self.policy._output_transform = output
        leaves = jax.tree.leaves(nnx.state(self.policy._model))
        gpu_bytes = gpu_parameter_bytes(value for value in leaves if isinstance(value, jax.Array))
        if gpu_bytes < 1_000_000_000:
            raise RuntimeError('Raw model parameters were not resident on the allocated GPU')
        self.metadata = {'framework': 'openpi_rtc/JAX GPU', 'jax_backend': jax.default_backend(),
                         'gpu_parameter_bytes': int(gpu_bytes), 'noise_dtype': 'bfloat16',
                         'model_weight_dtype': 'bfloat16', 'fixed_steps': 10}

    def run_frame(self, index):
        state = model_state(self.dataset['state'][index])
        obs = {'observation/cam_high': self.dataset['rgb'][index, 0],
               'observation/cam_wrist_left': self.dataset['rgb'][index, 1],
               'state': state.copy(), 'prompt': self.manifest['prompt'],
               'adarms_knob': np.asarray(0, np.int32)}
        noise = self.jnp.asarray(np.asarray(self.dataset['noise'][index]), dtype=self.jnp.bfloat16)
        result = np.asarray(self.policy.infer(obs, noise=noise)['actions'], np.float32).copy()
        result = result[:30, :7]
        result[:, 6] = 0.085 - (result[:, 6] - 3) * 0.085 / 226
        if result.shape != (30, 7) or not np.isfinite(result).all():
            raise ValueError('Raw returned invalid public actions')
        return result, self.raw, {'scheduled_steps': 10, 'nfe': 10, 'graph_setup': False}

    def verify_inputs(self, indices):
        for index in indices:
            state = model_state(self.dataset['state'][index])
            prepared = self.policy._input_transform({
                'observation/cam_high': self.dataset['rgb'][index, 0],
                'observation/cam_wrist_left': self.dataset['rgb'][index, 1],
                'state': state.copy(), 'prompt': self.manifest['prompt'],
                'adarms_knob': np.asarray(0, np.int32)})
            np.testing.assert_array_equal(prepared['tokenized_prompt'], self.dataset['tokens'][index])
            np.testing.assert_array_equal(prepared['tokenized_prompt_mask'], self.dataset['masks'][index])
            np.testing.assert_allclose(prepared['state'][:7], _normalized_state(state, self.stats), rtol=0, atol=1e-6)
            for camera, key in enumerate(('base_0_rgb', 'left_wrist_0_rgb')):
                np.testing.assert_array_equal(prepared['image'][key], self.dataset['rgb'][index, camera])
            if bool(prepared['image_mask']['right_wrist_0_rgb']):
                raise AssertionError('Padded third view must stay masked')
        return {'canonical_inputs_match_original_fork': True, 'checked_frames': indices}


class FlashRecordedPolicy:
    def __init__(self, dataset, manifest, stats, sampler, config):
        import torch
        from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx
        if not torch.cuda.is_available():
            raise RuntimeError('FlashRT requires Torch CUDA')
        torch.set_num_threads(2)
        self.torch = torch
        self.dataset, self.manifest, self.stats = dataset, manifest, stats
        self.tokenizer = _tokenizer(manifest['tokenizer_source'], manifest['tokenizer'])

        class UR5Frontend(Pi05JaxFrontendRtx):
            def _load_norm_stats(frontend, checkpoint_dir):
                frontend.norm_stats = stats

        self.pipe = UR5Frontend(
            manifest['checkpoint'], num_views=2, chunk_size=50, max_prompt_len=200,
            num_steps=10, use_fp8=False, state_prompt_mode='exact', use_cuda_graph=True,
            decoder_kernel='cublaslt', sampler=sampler,
            probeflow_config=config if sampler == 'probeflow' else None)
        self.metadata = {'framework': 'FlashRT/Torch CUDA, JAX CPU Orbax loader',
                         'noise_dtype': 'bfloat16', 'model_weight_dtype': 'bfloat16',
                         'sampler': sampler, 'config': vars(config) if sampler == 'probeflow' else {'num_steps': 10}}

    def _forward(self, images, noise):
        import ctypes
        pipe = self.pipe
        with self.torch.cuda.stream(pipe._graph_torch_stream):
            stream = pipe._graph_torch_stream.cuda_stream
            pipe._fill_noise(pipe._noise_buf, noise, None)
            pipe._copy_tensor_to_pipeline_buf_stream(pipe._noise_buf, pipe.pipeline.input_noise_buf, stream)
            pipe._fill_img_buf({'images': images})
            pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf, pipe.pipeline.input_images_buf, stream)
            pointer = pipe.pipeline.forward(stream=stream)
            rc = pipe._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(pipe._noise_out.data_ptr()), ctypes.c_void_p(pointer),
                pipe._noise_out.numel() * 2, 3, stream)
            if rc:
                raise RuntimeError(f'FlashRT copy failed: {rc}')
            rc = pipe._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
            if rc:
                raise RuntimeError(f'FlashRT synchronization failed: {rc}')
        return pipe._noise_out.float().cpu().numpy().copy()

    def run_frame(self, index):
        state = model_state(self.dataset['state'][index])
        normalized = _normalized_state(state, self.stats)
        ids, mask = self.tokenizer.tokenize(self.manifest['prompt'], normalized)
        images = list(self.dataset['rgb'][index])
        with _canonical_tokens(ids[mask]):
            self.pipe.set_prompt(self.manifest['prompt'], state=normalized)
        capture = not self.pipe.graph_recorded
        if capture:
            self.pipe.calibrate_with_real_data([{'images': images}])
        raw = self._forward(images, self.dataset['noise'][index])
        actions = decode_ur5_actions(raw, state, self.stats)
        metadata = dict(getattr(self.pipe.pipeline, 'last_sampler_metadata',
                                {'scheduled_steps': 10, 'nfe': 10}))
        metadata['graph_setup'] = capture
        return actions, raw, metadata

    def verify_reference(self, indices, output):
        from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
        results = []
        for index in indices:
            _, actual, metadata = self.run_frame(index)
            pipeline = self.pipe.pipeline
            config = pipeline.sampler_config
            cases = [('default', config)]
            if index == indices[0]:
                cases.append(('forced_dense', replace(config, epsilon=1e-12)))
            for label, active_config in cases:
                pipeline.sampler_config = active_config
                try:
                    actual = self._forward(list(self.dataset['rgb'][index]), self.dataset['noise'][index])
                    metadata = dict(pipeline.last_sampler_metadata)
                    with self.torch.cuda.stream(self.pipe._graph_torch_stream):
                        stream = self.pipe._graph_torch_stream.cuda_stream

                        def evaluate(state, t):
                            self.pipe._fill_noise(self.pipe._noise_buf, state, None)
                            self.pipe._copy_tensor_to_pipeline_buf_stream(
                                self.pipe._noise_buf, pipeline.input_noise_buf, stream)
                            Pi05Pipeline.transformer_decoder(
                                pipeline, stream=stream,
                                _step_indices=(pipeline._time_indices[t],), _integrate=False)
                            return pipeline._velocity_view.float().cpu().numpy().copy()

                        expected, oracle = reference_bf16_probeflow(
                            self.dataset['noise'][index], evaluate, active_config)
                    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
                    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    relative_l2 = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
                    row = {'frame_index': index, 'case': label, 'cosine': cosine,
                           'relative_l2': relative_l2, 'max_abs': float(np.max(np.abs(a - b))),
                           'actual_nfe': metadata['nfe'], 'reference_nfe': oracle['nfe'],
                           'actual_steps': metadata['scheduled_steps'], 'reference_steps': oracle['scheduled_steps']}
                    row['passed'] = (cosine >= .999 and relative_l2 <= .05
                                     and row['actual_nfe'] == row['reference_nfe']
                                     and row['actual_steps'] == row['reference_steps'])
                    np.save(output / f'reference-{index}-{label}.npy', expected)
                    results.append(row)
                    if not row['passed']:
                        raise AssertionError(f'Independent BF16 algorithm reference mismatch: {row}')
                finally:
                    pipeline.sampler_config = config
        return {'independent_bf16_algorithm_reference': results,
                'gate': {'min_cosine': .999, 'max_relative_l2': .05, 'matching_NFE': True}}


def worker(args):
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in (None, 'batch', 'extern')
            or os.environ.get('SLURM_JOB_PARTITION') != 'gpu_a800'):
        raise RuntimeError('Episode GPU comparison requires an allocated A800 srun step')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    # Load before timing: no method gets a cold memory-mapped file-I/O penalty.
    dataset = {key: np.load(args.prepared / filename, allow_pickle=False)
               for key, filename in [('rgb', 'rgb.npy'), ('noise', 'noise.npy'),
                                      ('state', 'state_raw7.npy'), ('tokens', 'token_ids.npy'),
                                      ('masks', 'token_mask.npy')]}
    stats = _stats(manifest['checkpoint'], manifest['asset_id'])
    config = ProbeFlowConfig(epsilon=args.epsilon, max_steps=args.max_steps)
    started = time.perf_counter()
    if args.backend == 'raw_euler':
        policy = RawRecordedPolicy(dataset, manifest, stats)
    else:
        policy = FlashRecordedPolicy(dataset, manifest, stats,
                                     'probeflow' if args.backend == 'flash_probeflow' else 'euler', config)
    load_seconds = time.perf_counter() - started
    count = manifest['frame_count']
    checks = [0, count // 2, count - 1]
    if args.backend == 'raw_euler':
        verification = policy.verify_inputs(checks)
    elif args.backend == 'flash_probeflow':
        verification = policy.verify_reference(checks, output)
    else:
        verification = {'native_fixed_steps': policy.pipe._num_steps}
    # Warm every exact token-length bucket before the timed, complete replay.
    lengths = dataset['masks'].sum(axis=1)
    bucket_frames = [int(np.flatnonzero(lengths == length)[0]) for length in sorted(set(lengths))]
    warm_frames = bucket_frames + np.linspace(0, count - 1, args.warmup, dtype=int).tolist()
    records = []
    actions = np.lib.format.open_memmap(output / 'public_actions.npy', mode='w+', dtype=np.float32,
                                      shape=(count, 30, 7))
    normalized = np.lib.format.open_memmap(output / 'normalized_actions.npy', mode='w+', dtype=np.float32,
                                         shape=(count, 50, 32))
    with (output / 'latency.jsonl').open('x') as stream:
        for warmup, indices in [(True, warm_frames), (False, range(count))]:
            for index in indices:
                start = time.perf_counter_ns()
                public, raw, metadata = policy.run_frame(index)
                milliseconds = (time.perf_counter_ns() - start) / 1e6
                if raw.shape != (50, 32) or not np.isfinite(raw).all() or not np.isfinite(public).all():
                    raise AssertionError(f'Invalid real model output at frame {index}')
                row = {'backend': args.backend, 'frame_index': int(index),
                       'noise_seed': manifest['noise_seed_base'] + int(index),
                       'token_count': int(lengths[index]), 'warmup': warmup,
                       'exclude': bool(warmup or metadata['graph_setup']),
                       'run_policy_ms': milliseconds, **metadata}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                if not warmup:
                    records.append(row)
                    actions[index], normalized[index] = public, raw
                    if index % 250 == 0:
                        print('EPISODE_PROGRESS', args.backend, index, count, milliseconds,
                              'NFE', metadata['nfe'], flush=True)
    actions.flush()
    normalized.flush()
    nfe = np.asarray([row['nfe'] for row in records])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        raise RuntimeError('Matched replay must have exactly one allocated visible GPU')
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    own_uuids = {line.split(',')[1].strip() for line in processes.splitlines()
                 if line.split(',')[0].strip() == str(os.getpid())}
    if len(own_uuids) != 1:
        raise RuntimeError(f'Cannot identify exactly one GPU context for benchmark PID {os.getpid()}')
    gpu = subprocess.check_output(['nvidia-smi', '-i', next(iter(own_uuids)),
                                   '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).strip()
    report = {'status': 'passed', 'backend': args.backend, 'job_id': os.environ['SLURM_JOB_ID'],
              'step_id': os.environ['SLURM_STEP_ID'], 'gpu': gpu, 'visible_gpu': visible,
              'episode': manifest['episode'], 'checkpoint': manifest['checkpoint'],
              'asset_id': manifest['asset_id'], 'prompt': manifest['prompt'],
              'input_residency': 'all shared input arrays loaded into host RAM before warmup',
              'frames': count, 'warmup_calls': len(warm_frames),
              'excluded_episode_calls': sum(row['exclude'] for row in records),
              'statistics': latency_statistics(records), 'nfe_mean': float(nfe.mean()),
              'nfe_min': int(nfe.min()), 'nfe_max': int(nfe.max()),
              'nfe_histogram': {str(value): int((nfe == value).sum()) for value in sorted(set(nfe))},
              'load_seconds': load_seconds, 'model': policy.metadata, 'verification': verification,
              'timing_boundary': 'host shared RGB224 uint8 + raw state7 + bf16 noise -> synchronized host public actions30x7; includes preprocessing, all probes/lookahead/routing and output decoding; excludes warmup/capture/log writes and network',
              'recorded_only': True}
    _json_write(output / 'result.json', report)
    print('EPISODE_WORKER_RESULT', json.dumps(report), flush=True)


def summarize(args):
    rows = [json.loads((args.results / backend / 'result.json').read_text())
            for backend in ('raw_euler', 'flash_euler', 'flash_probeflow')]
    identity = [(row['gpu'], row['checkpoint'], row['episode'], row['prompt'], row['frames']) for row in rows]
    if any(value != identity[0] for value in identity) or any(row['status'] != 'passed' for row in rows):
        raise AssertionError('The three runs do not share GPU/checkpoint/episode/prompt/frame count')
    comparisons = []
    reference = np.load(args.results / 'raw_euler/normalized_actions.npy', mmap_mode='r')
    actual = np.load(args.results / 'flash_euler/normalized_actions.npy', mmap_mode='r')
    for index in (0, len(reference) // 2, len(reference) - 1):
        a, b = reference[index].astype(np.float64).ravel(), actual[index].astype(np.float64).ravel()
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        relative = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
        comparisons.append({'frame_index': index, 'cosine': cosine,
                            'relative_l2': relative, 'max_abs': float(np.max(np.abs(a - b))),
                            'passed': cosine >= .999 and relative <= .05})
    report = {'status': 'passed' if all(row['passed'] for row in comparisons) else 'failed',
              'same_device_and_workload': True, 'runs': rows,
              'raw_vs_flash_euler': comparisons,
              'scope': 'Deterministic complete recorded-input replay, not a robot/client run or closed-loop success',
              'recorded_first_image': str(args.prepared / 'recorded-first.png'),
              'recorded_last_image': str(args.prepared / 'recorded-last.png')}
    _json_write(args.results / 'comparison.json', report)
    lines = ['# UR5 recorded episode — matched offline replay', '',
             'Recorded inputs only; not a closed-loop outcome. All three runs use the same allocated GPU, checkpoint, episode, prompt and BF16 noise.', '',
             '| Method | Frames | Steady P50 (ms) | Steady P99 (ms) | Mean NFE |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['backend']} | {row['frames']} | {row['statistics']['p50_ms']:.3f} | {row['statistics']['p99_ms']:.3f} | {row['nfe_mean']:.3f} |")
    lines += ['', rows[0]['timing_boundary'], '', f"Euler parity gate: {report['status']}", '',
              f"![Recorded first input]({os.path.relpath(args.prepared / 'recorded-first.png', args.results)})",
              f"![Recorded last input]({os.path.relpath(args.prepared / 'recorded-last.png', args.results)})", '']
    (args.results / 'comparison.md').write_text('\n'.join(lines))
    print('MATCHED_COMPARISON_RESULT', json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise AssertionError('Raw/FlashRT Euler parity gate failed; see actual metrics')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--episode', type=Path, required=True)
    prep.add_argument('--checkpoint', type=Path, required=True)
    prep.add_argument('--asset-id', default='ur5_three_tasks')
    prep.add_argument('--openpi-dir', type=Path, required=True)
    prep.add_argument('--tokenizer-source', type=Path, required=True)
    prep.add_argument('--tokenizer', type=Path, required=True)
    prep.add_argument('--prompt', default='Arrange the fruit in the basket.')
    prep.add_argument('--seed', type=int, default=37)
    prep.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('worker')
    run.add_argument('--backend', choices=['raw_euler', 'flash_euler', 'flash_probeflow'], required=True)
    run.add_argument('--prepared', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--warmup', type=int, default=20)
    run.add_argument('--epsilon', type=float, default=.008)
    run.add_argument('--max-steps', type=int, default=20)
    summary = commands.add_parser('summarize')
    summary.add_argument('--prepared', type=Path, required=True)
    summary.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    {'prepare': prepare, 'worker': worker, 'summarize': summarize}[args.mode](args)


if __name__ == '__main__':
    main()
