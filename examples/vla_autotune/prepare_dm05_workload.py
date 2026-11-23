"""Export the four author DM0.5 requests to the loop's NPZ workload format (CPU only)."""
import argparse
import json
from pathlib import Path
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='official_dm05_latency.py output')
    parser.add_argument('--source', type=Path, required=True, help='pinned OpenDM checkout with demo images')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from PIL import Image

    record = json.loads((args.reference / 'result.json').read_text())
    if (record.get('model'), record.get('origin'), record.get('steps')) != ('dm05', 'author', 10):
        raise ValueError('Expected official DM0.5 Euler10 outputs')
    cases = record['cases']
    expected = {'demo-seed0', 'demo-seed17', 'state-seed0', 'prompt-seed0'}
    if len(cases) != len(expected) or {case['id'] for case in cases} != expected:
        raise ValueError('Expected the four distinct fixed DM0.5 author cases')
    cameras = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
    images = {}
    for camera in cameras:
        with Image.open(args.source / 'assets/demo/images/episode0' / camera / '0.jpg') as image:
            images[camera] = np.array(image.convert('RGB'))
    workload = {'schema_version': 1, 'model': 'dm05', 'workload_id': 'dm05-base-public14-e10-v1', 'requests': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        bundle = Path(tmp) / 'bundle'
        bundle.mkdir()
        for case in cases:
            name = case['id']
            noise = torch.load(args.reference / name / 'initial_noise.pt', map_location='cpu', weights_only=True)
            if noise.dtype != torch.bfloat16 or tuple(noise.shape) != (1, 50, 32):
                raise ValueError(f'Unexpected author BF16 noise for {name}')
            state = np.asarray(case['state'], dtype=np.float32)
            noise = noise.float().numpy()
            if state.shape != (14,) or not np.isfinite(state).all() or not np.isfinite(noise).all():
                raise ValueError(f'Invalid state or noise for {name}')
            filename = name + '.npz'
            np.savez_compressed(bundle / filename, state=state, noise=noise, **images)
            workload['requests'].append({'request_id': name, 'npz_path': filename,
                                         'fields': {'images': list(cameras), 'state': 'state'},
                                         'prompt': case['prompt'], 'noise_key': 'noise'})
        baseline = {'candidate_id': 'dm05-mixed-e10', 'workload_id': workload['workload_id'],
                    'backend': {'model': 'dm05', 'checkpoint': str(args.checkpoint.resolve()),
                                'backend': 'flashrt', 'framework': 'torch', 'precision': 'bf16',
                                'hardware': 'cuda_sm80', 'sampler': 'euler', 'num_steps': 10,
                                'options': {'source_dir': str(args.source.resolve()), 'use_cuda_graph': True}},
                    'hypothesis': 'Fixed mixed FlashRT BF16 Euler10 baseline for four OpenDM requests'}
        gate = {'scope': 'dm05-base-absolute-public14-v1', 'action_dim': 14,
                'max_relative_l2': 0.05, 'min_cosine': 0.999}
        for name, item in [('workload', workload), ('baseline', baseline), ('quality', gate)]:
            (bundle / f'{name}.json').write_text(json.dumps(item, indent=2, allow_nan=False) + '\n')
        bundle.rename(args.output)


if __name__ == '__main__':
    main()
