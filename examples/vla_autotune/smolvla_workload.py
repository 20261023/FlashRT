"""Create explicit real LIBERO requests from a local pinned LeRobot v3 dataset.

The command is CPU-only and never loads a policy. Raw RGB frames, state and
task text come from the dataset; only initial diffusion noise is generated.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATASET_REVISION = 'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4'
CAMERAS = ('observation.images.image', 'observation.images.image2')
DATASET_FILES = {
    'meta/info.json': '630630a1a4f07fc9fc1f1b2b7987ed3bd034bd8cffaaf8a0e6939ae9c468445d',
    'meta/tasks.parquet': '80bbbcdf623b7447c0f187af8ea70aa3a5c0018aaf56adba1731aeddcf18f5ff',
    'meta/episodes/chunk-000/file-000.parquet': '324db6d3f8efbfb340177507c11bfa2e78b1753c46802b3700fa8c8b4be461db',
    'videos/observation.images.image2/chunk-000/file-000.mp4': 'ba95edeb5190528b45364d7c8e518b7f12afa8ded72108e19a8349aa3f0a7d58',
    'videos/observation.images.image/chunk-000/file-000.mp4': '77d59b50e309e2099e562e70806603a1bdb66b893efcaecc2534b2c9fa3e02ae',
    'data/chunk-000/file-000.parquet': '71e4088ce5877ba3234d0cde284c403f449ed560a656a2626e72fc2c3399502d',
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset_files(root, expected):
    """Refuse to attach a known upstream revision to altered local data."""
    observed = {}
    for name, digest in expected.items():
        path = Path(root) / name
        if not path.is_file():
            raise ValueError(f'Pinned LIBERO dataset file is missing: {name}')
        observed[name] = _sha(path)
        if observed[name] != digest:
            raise ValueError(f'Pinned LIBERO dataset bytes differ: {name}')
    return observed


def write_workload(cases, output, *, dataset_identity, seed=7, spec_dir=None):
    """Serialize already extracted real observations with auditable provenance."""
    from flash_rt.models.smolvla.inputs import validate_observation
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite SmolVLA workload {output}')
    if len(cases) < 4:
        raise ValueError('SmolVLA onboarding requires at least four varied real cases')
    for case in cases:
        validate_observation(case)
        if not isinstance(case['prompt'], str) or not case['prompt'].strip():
            raise ValueError('Real LIBERO task text is required')
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    requests = []
    for index, case in enumerate(cases):
        name = f'libero-{index:03d}'
        np.savez_compressed(
            output / f'{name}.npz', image_0=case['images'][0], image_1=case['images'][1],
            state=case['state'], noise=rng.standard_normal((1, 50, 32)).astype(np.float32))
        requests.append({
            'request_id': name, 'npz_path': f'{name}.npz',
            'fields': {'images': ['image_0', 'image_1'], 'state': 'state'},
            'noise_key': 'noise', 'prompt': case['prompt'],
            'dataset_episode': int(case['episode']), 'dataset_frame': int(case['frame']),
        })
    path = output / 'workload.json'
    path.write_text(json.dumps({
        'schema_version': 1, 'model': 'smolvla', 'workload_id': 'smolvla-libero-real-v1',
        'spec_dir': str(Path(spec_dir).resolve()) if spec_dir is not None else str(
            Path(__file__).resolve().parents[2] / 'flash_rt/models/smolvla'),
        'requests': requests,
        'provenance': {'kind': 'real_dataset', 'dataset': dataset_identity, 'noise_seed': seed,
                       'camera_order': list(CAMERAS), 'image_transform': 'none; decoded RGB24',
                       'state_transform': 'none; dataset float32[8]'},
    }, indent=2) + '\n')
    return path


def _video_frame(path, timestamp):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if abs(float(frame.time) - timestamp) < 0.025:
                return frame.to_ndarray(format='rgb24')
            if float(frame.time) > timestamp + 0.025:
                break
    raise ValueError(f'No dataset video frame matches timestamp {timestamp} in {path}')


def extract_dataset(dataset, *, episode_frames=((0, 0), (0, 40), (1, 0), (1, 40))):
    import pyarrow.parquet as pq
    root = Path(dataset).resolve(strict=True)
    verified = verify_dataset_files(root, DATASET_FILES)
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if info['codebase_version'] != 'v3.0' or info['features']['observation.state']['shape'] != [8]:
        raise ValueError('SmolVLA requires the pinned LeRobot v3 LIBERO state[8] dataset')
    files = {info_path}
    task_path = root / 'meta/tasks.parquet'
    task_rows = pq.read_table(task_path).to_pylist()
    tasks = {int(row['task_index']): row['__index_level_0__'] for row in task_rows}
    files.add(task_path)
    episodes = {}
    for path in sorted((root / 'meta/episodes').glob('*/*.parquet')):
        files.add(path)
        episodes.update({int(row['episode_index']): row for row in pq.read_table(path).to_pylist()})
    cases = []
    for episode_index, frame_index in episode_frames:
        episode = episodes[episode_index]
        path = root / info['data_path'].format(
            chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index'])
        files.add(path)
        rows = [row for row in pq.read_table(path).to_pylist()
                if row['episode_index'] == episode_index and row['frame_index'] == frame_index]
        if len(rows) != 1:
            raise ValueError(f'Dataset does not uniquely cover episode={episode_index} frame={frame_index}')
        row = rows[0]
        images = []
        for camera in CAMERAS:
            prefix = f'videos/{camera}'
            path = root / info['video_path'].format(
                video_key=camera, chunk_index=episode[f'{prefix}/chunk_index'],
                file_index=episode[f'{prefix}/file_index'])
            files.add(path)
            images.append(_video_frame(path, episode[f'{prefix}/from_timestamp'] + row['timestamp']))
        cases.append({'images': images, 'state': np.asarray(row['observation.state'], dtype=np.float32),
                      'prompt': tasks[int(row['task_index'])], 'episode': episode_index, 'frame': frame_index})
    if any(str(path.relative_to(root)) not in verified for path in files):
        raise ValueError('Selected LIBERO episodes require files outside the pinned onboarding subset')
    return cases, {
        'repository': 'https://huggingface.co/datasets/lerobot/libero',
        'revision': DATASET_REVISION, 'license_declared': 'Apache-2.0',
        'files': {str(path.relative_to(root)): verified[str(path.relative_to(root))] for path in sorted(files)},
        'quality_scope': 'fixed-input numerical comparisons only; no task-success estimate',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--spec-dir', type=Path)
    args = parser.parse_args(argv)
    cases, identity = extract_dataset(args.dataset)
    path = write_workload(cases, args.output, dataset_identity=identity, seed=args.seed,
                          spec_dir=args.spec_dir)
    print(path)


if __name__ == '__main__':
    main()
