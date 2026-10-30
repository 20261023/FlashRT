"""Real-UR5 GPU regressions for batching refusal and near-endpoint horizons.

Uses the existing recorded inputs/checkpoint, never a robot/client connection.
The alternate frame is a previously observed dense-routing case with the same
prompt length, so A/B/A exercises the SAME captured pipeline with fixed config.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from examples.pi05_probeflow_episode_benchmark import FlashRecordedPolicy, _stats
from flash_rt.models.pi05.probeflow import ProbeFlowConfig


def eager(policy, index):
    pipeline = policy.pipe.pipeline
    captured = pipeline._graph
    pipeline._graph = None
    try:
        return policy._forward(list(policy.dataset['rgb'][index]), policy.dataset['noise'][index])
    finally:
        pipeline._graph = captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alternate-frame', type=int, default=1305)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_STEP_ID') in (None, 'batch', 'extern')
            or os.environ.get('SLURM_JOB_PARTITION') != 'gpu_a800'):
        raise RuntimeError('GPU review regressions require an allocated A800 srun step')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    dataset = {key: np.load(args.prepared / filename, allow_pickle=False)
               for key, filename in [('rgb', 'rgb.npy'), ('noise', 'noise.npy'),
                                      ('state', 'state_raw7.npy'), ('tokens', 'token_ids.npy'),
                                      ('masks', 'token_mask.npy')]}
    stats = _stats(manifest['checkpoint'], manifest['asset_id'])
    b = args.alternate_frame
    if not 0 < b < manifest['frame_count'] or dataset['masks'][0].sum() != dataset['masks'][b].sum():
        raise ValueError('Alternate frame must exist and use the same captured prompt length')
    if np.array_equal(dataset['rgb'][0], dataset['rgb'][b]):
        raise AssertionError('A/B regression requires different real camera observations')
    config = ProbeFlowConfig()
    policy = FlashRecordedPolicy(dataset, manifest, stats, 'probeflow', config)
    _, initial, _ = policy.run_frame(0)
    frontend = policy.pipe
    keys = ('pipeline', 'attn_backend', '_last_prompt_call', 'current_prompt_len',
            'graph_recorded', 'calibrated')
    before = {key: getattr(frontend, key) for key in keys}
    try:
        frontend.select_batch_size(2)
    except NotImplementedError as error:
        if 'ProbeFlow' not in str(error):
            raise
    else:
        raise AssertionError('Unsupported batching was not rejected')
    for key, value in before.items():
        if getattr(frontend, key) is not value:
            raise AssertionError(f'Batch refusal mutated {key}')
    _, after_refusal, _ = policy.run_frame(0)
    np.testing.assert_array_equal(initial, after_refusal)
    pipeline = frontend.pipeline
    captured = pipeline._graph
    rows = []
    first = None
    for index in (0, b, 0):
        _, actual, metadata = policy.run_frame(index)
        if frontend.pipeline is not pipeline or pipeline._graph is not captured or pipeline.sampler_config is not config:
            raise AssertionError('A/B/A unexpectedly switched capture or sampler config')
        expected = eager(policy, index)
        np.testing.assert_array_equal(actual, expected)
        if first is None:
            first = actual.copy()
        rows.append({'frame_index': index, 'nfe': metadata['nfe'],
                     'steps': metadata['scheduled_steps'], 'cosine': metadata['cosine'],
                     'graph_eager_equal': True})
    np.testing.assert_array_equal(actual, first)
    if rows[0]['nfe'] == rows[1]['nfe'] or rows[0]['nfe'] != rows[2]['nfe']:
        raise AssertionError('Fixed-config routing did not change and recover on A/B/A')
    horizon_results = []
    for number, horizon in enumerate((1. - 1e-13, float(np.nextafter(1., 0.)))):
        edge = ProbeFlowConfig(probe_horizon=horizon)
        edge_policy = FlashRecordedPolicy(dataset, manifest, stats, 'probeflow', edge)
        _, graph, metadata = edge_policy.run_frame(0)
        edge_pipeline = edge_policy.pipe.pipeline
        if not all(0 < t <= 1 for t in edge_pipeline._sampler_times):
            raise AssertionError('Valid horizon generated an endpoint model time')
        if 1. - horizon not in edge_pipeline._time_indices:
            raise AssertionError('Probe time does not match its exact runtime key')
        np.testing.assert_array_equal(graph, eager(edge_policy, 0))
        output = args.output / f'near-endpoint-{number}'
        output.mkdir()
        reference = edge_policy.verify_reference([0], output)
        horizon_results.append({'probe_horizon': horizon, 'probe_model_time': 1. - horizon,
                                'all_times_positive': True, 'graph_eager_equal': True,
                                'nfe': metadata['nfe'], 'reference': reference})
    report = {'status': 'passed', 'job_id': os.environ['SLURM_JOB_ID'],
              'batch_refusal_preserves_live_frontend': True,
              'post_refusal_inference_equal': True,
              'fixed_config_same_capture_ABA': rows,
              'near_endpoint_horizons': horizon_results,
              'scope': 'Real UR5 recorded-input GPU regression; no robot/client execution'}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('REVIEW_FIXES_GPU_RESULT', json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
