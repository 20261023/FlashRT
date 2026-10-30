"""Bounded UR5 acceptance check; invoke only inside an allocated Slurm GPU step.

This is a native-backend correctness fixture, NOT a matched episode benchmark.
The existing UR5 adapter supplies only its pinned tokenizer helpers, read-only.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--ur5-adapter', type=Path, required=True)
    parser.add_argument('--asset-id', default='ur5_three_tasks')
    parser.add_argument('--prompt', default='Arrange the fruit in the basket.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in (None, 'batch'):
        raise RuntimeError('GPU smoke requires an allocated Slurm srun step')
    if not 1 <= args.repeats <= 100:
        raise ValueError('repeats must be in [1, 100]')
    import ctypes
    import torch
    from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx
    from flash_rt.models.pi05.probeflow import ProbeFlowConfig

    args.output.mkdir(parents=True, exist_ok=False)
    stats_path = args.checkpoint / 'assets' / args.asset_id / 'norm_stats.json'
    stats = json.loads(stats_path.read_text())['norm_stats']
    spec = importlib.util.spec_from_file_location('_readonly_ur5_adapter', args.ur5_adapter)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    tokenizer = adapter.official_tokenizer()
    with np.load(args.fixture, allow_pickle=False) as fixture:
        images = [np.ascontiguousarray(image) for image in fixture['rgb'][:2]]
        model_state = fixture['state_model7'].copy()
        noise = fixture['noise'].copy()
    low, high = (np.asarray(stats['state'][key], np.float32) for key in ('q01', 'q99'))
    state = (model_state - low) / (high - low + 1e-6) * 2 - 1
    ids, mask = tokenizer.tokenize(args.prompt, state)
    ids = ids[mask]

    class UR5Frontend(Pi05JaxFrontendRtx):
        def _load_norm_stats(self, checkpoint_dir):
            self.norm_stats = stats

    def frontend(sampler, config=None, steps=10):
        value = UR5Frontend(
            args.checkpoint, num_views=2, chunk_size=50, max_prompt_len=200,
            num_steps=steps, use_fp8=False, state_prompt_mode='exact',
            use_cuda_graph=True, decoder_kernel='cublaslt',
            sampler=sampler, probeflow_config=config)
        with adapter.canonical_tokens(ids):
            value.set_prompt(args.prompt, state=state)
        value.calibrate_with_real_data([{'images': images}])
        return value

    def infer(value, initial, *, graph=True):
        with torch.cuda.stream(value._graph_torch_stream):
            stream = value._graph_torch_stream.cuda_stream
            value._fill_noise(value._noise_buf, initial, None)
            value._copy_tensor_to_pipeline_buf_stream(value._noise_buf, value.pipeline.input_noise_buf, stream)
            value._fill_img_buf({'images': images})
            value._copy_tensor_to_pipeline_buf_stream(value._img_buf, value.pipeline.input_images_buf, stream)
            if graph:
                pointer = value.pipeline.forward(stream=stream)
            else:
                value.pipeline.run_pipeline(stream=stream)
                pointer = value.pipeline.input_noise_buf.ptr.value
            rc = value._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(value._noise_out.data_ptr()), ctypes.c_void_p(pointer),
                value._noise_out.numel() * 2, 3, stream)
            if rc:
                raise RuntimeError(f'smoke output copy failed: {rc}')
            rc = value._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
            if rc:
                raise RuntimeError(f'smoke stream sync failed: {rc}')
        result = value._noise_out.float().cpu().numpy().copy()
        if result.shape != (50, 32) or not np.isfinite(result).all():
            raise AssertionError('nonfinite or wrong-shaped native actions')
        return result

    config = ProbeFlowConfig()
    started = time.perf_counter()
    adaptive = frontend('probeflow', config)
    load_and_capture = time.perf_counter() - started
    graph = infer(adaptive, noise)
    graph_metadata = dict(adaptive.pipeline.last_sampler_metadata)
    eager = infer(adaptive, noise, graph=False)
    np.testing.assert_array_equal(graph, eager, err_msg='ProbeFlow graph/eager disagreement')
    np.testing.assert_array_equal(graph, infer(adaptive, noise), err_msg='seeded replay changed')
    other = infer(adaptive, -noise)
    if np.array_equal(other, graph):
        raise AssertionError('different initial noise did not change output')
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        infer(adaptive, noise)
        timings.append((time.perf_counter() - start) * 1000)

    # Reuse the same captured graphs but force different routing thresholds.
    # This is an internal acceptance check, not a public config-mutation API.
    adaptive.pipeline.sampler_config = replace(config, epsilon=10.)
    two = infer(adaptive, noise)
    two_metadata = dict(adaptive.pipeline.last_sampler_metadata)
    if (two_metadata['scheduled_steps'], two_metadata['nfe']) != (2, 2):
        raise AssertionError('minimum branch did not reuse both probes')
    adaptive.pipeline.sampler_config = replace(config, epsilon=1e-12)
    dense = infer(adaptive, noise)
    dense_metadata = dict(adaptive.pipeline.last_sampler_metadata)
    if dense_metadata['scheduled_steps'] <= 2 or dense_metadata['nfe'] != dense_metadata['scheduled_steps'] + 1:
        raise AssertionError('dense branch was not rerouted after capture')
    adaptive.pipeline.sampler_config = config

    # Two equal segments must reproduce a native two-step Euler reference.
    euler = frontend('euler', steps=2)
    reference = infer(euler, noise)
    np.testing.assert_array_equal(two, reference, err_msg='ProbeFlow sign/time/scaling differs from Euler N=2')
    for name, array in [('graph', graph), ('eager', eager), ('two-probe', two),
                        ('dense', dense), ('euler-two-step', reference)]:
        np.save(args.output / f'{name}.npy', array)
    report = {
        'status': 'passed', 'job_id': os.environ['SLURM_JOB_ID'],
        'step_id': os.environ['SLURM_STEP_ID'], 'gpu': torch.cuda.get_device_name(),
        'checkpoint': str(args.checkpoint.resolve()), 'norm_stats': str(stats_path),
        'fixture': str(args.fixture), 'token_count': len(ids), 'native_views': 2,
        'load_and_capture_seconds': load_and_capture,
        'graph_eager_equal': True, 'seeded_replay_equal': True,
        'two_step_euler_equal': True, 'shape': list(graph.shape),
        'default_metadata': graph_metadata, 'minimum_metadata': two_metadata,
        'dense_metadata': dense_metadata,
        'fixture_native_p50_ms': float(np.median(timings)), 'repeats': args.repeats,
        'scope': 'Native correctness fixture only; not matched Mock P50, run_policy E2E, closed-loop success or a speedup claim',
    }
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PROBEFLOW_SMOKE_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
