"""Original dm05 policy loader and request contract."""

from pathlib import Path
import sys
import hashlib
import json
import os
import numpy as np
from .common import _cameras, _prompt


def load(spec, torch):
    """Use the author's complete _prepare_model_input -> _predict public policy."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, spec['source'])
    from opendm.exp.dm05_exp import DM05Exp
    exp = DM05Exp(task='inference', logger_level='WARNING')
    exp.model_config.model_name_or_path = spec['checkpoint']
    exp.model_config.vision_attn_implementation = 'sdpa'
    exp.model_config.liger_kernel = False
    exp.model_config.chunk_size = 50
    exp.inference_config.backend = 'default'
    exp.inference_config.diffusion_steps = 10
    exp.inference_config.enable_vlm_chat = False
    exp.data_config.is_history = False
    exp._initialize_inference_runtime()
    policy = exp.inference_config

    def infer(request):
        os.environ['DM05_INFERENCE_SEED'] = str(spec['options']['noise_seeds'][request.request_id])
        images = _cameras(request, ('image', 'wrist_image', 'wrist_image_right'))
        converted = []
        for image in images:
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError('Author DM requires uint8 HWC RGB images, without implicit normalization')
                image = Image.fromarray(image)
            converted.append(image)
        state = np.asarray(request.observation['state'], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError('Author DM requires the same absolute state[14]')
        data = policy._prepare_model_input(
            text=_prompt(request), images=converted, states=state.tolist(), robot_type='DOS W1')
        return np.asarray(policy._predict(data), dtype=np.float32)

    return policy.model, infer, {
        'implementation': 'author OpenDM DM05Exp default backend _prepare_model_input + _predict',
        'noise_contract': 'original DM05_INFERENCE_SEED CUDA RNG, verified against typed workload noise by an untimed observation',
        'expected_shape': [50, 14],
        'output_semantics': 'full denormalized reconstructed absolute DOS W1 CPU actions',
    }

_DM_SOURCE_REVISION = "03f799120d425ead51d9482bb1da950b6dabade9"
_DM_WEIGHT_REVISION = "5cd18734814abb075a9ccfd9ad6d16777b5cf10e"
_DM_WEIGHT_SHA256 = "b7da77f516ebd0c0c68faed2f0cfe1c68985b7b874e49dda4292d8d286de20dd"
_DM_PYTHON_SHA256 = "28617ae3b6db5a887d6291e32bfac0406666f79188ec759dad5e22e1404dd462"


def _dm_python_digest(root):
    package, digest = Path(root) / 'opendm', hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(path.relative_to(package).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def verified_export(root):
    return _dm_python_digest(root) == _DM_PYTHON_SHA256


def legacy_inputs(path, spec):
    """Pin the actual legacy author bundle, without importing Torch on init."""
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    from flash_rt.vla.autotune.author import AuthorReferenceError, file_sha256
    required = {'model': 'dm05', 'origin': 'author', 'source_revision': _DM_SOURCE_REVISION,
                'checkpoint_revision': _DM_WEIGHT_REVISION, 'steps': 10, 'nfe': 10,
                'output_action_dim': 14, 'internal_action_dim': 32, 'chunk': 50}
    if (any(record.get(key) != value for key, value in required.items())
            or spec['source_revision'] != _DM_SOURCE_REVISION or spec['num_steps'] != 10
            or record.get('script_sha256') != '599d9e318deda0717277e909802f597154128071ce86ef15429841979c020cd7'):
        raise AuthorReferenceError('Legacy DM reference does not match the pinned original author producer/profile')
    cases = record.get('cases')
    if not isinstance(cases, list) or len(cases) != 4 or len({case['id'] for case in cases}) != 4:
        raise AuthorReferenceError('Legacy DM reference requires all four original cases')
    files = {str(path): file_sha256(path)}
    for case in cases:
        if not isinstance(case['id'], str) or Path(case['id']).name != case['id'] or case['id'] in ('.', '..'):
            raise AuthorReferenceError('Invalid legacy DM case directory')
        for filename in ('actions.npy', 'samples_ms.npy', 'initial_noise.pt', 'prepared.pt'):
            artifact = path.parent / case['id'] / filename
            files[str(artifact)] = file_sha256(artifact)
    return {'result_path': str(path), 'files': files}


def bind_saved(spec, workload, saved_path, output):
    """CPU-only binding of original DM four-case outputs, never new inference.

    Exact current CPU transforms must reproduce the saved prepared tensors.
    This is needed because the old report did not record raw input digests.
    """
    import importlib
    from flash_rt.vla.autotune.author import (AuthorReferenceError, BOUNDARY, _validate_spec, _write_json,
        file_sha256, source_identity, prove_author_model, workload_identity, content_identity)
    from flash_rt.vla.autotune.quality import output_domain
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from PIL import Image
    spec = _validate_spec(spec)
    inputs = legacy_inputs(saved_path, spec)
    raw = json.loads(Path(saved_path).read_text())
    if _dm_python_digest(spec['source']) != _DM_PYTHON_SHA256:
        raise AuthorReferenceError('Legacy DM binding requires the pinned original author CPU source bytes')
    if file_sha256(Path(spec['checkpoint']) / 'model.safetensors') != _DM_WEIGHT_SHA256:
        raise AuthorReferenceError('Legacy DM binding requires the independently pinned base weight bytes')
    cases = {case['id']: case for case in raw['cases']}
    if set(cases) != {case['request_id'] for case in workload.specs} or workload.model != 'dm05':
        raise AuthorReferenceError('Legacy DM workload must retain all four original author case IDs')
    if {key: case['seed'] for key, case in cases.items()} != spec['options']['noise_seeds']:
        raise AuthorReferenceError('Legacy DM author seeds differ from the reference plan')
    source = source_identity(spec)
    sys.path.insert(0, spec['source'])
    module = importlib.import_module('opendm.exp.dm05_exp')
    if Path(module.__file__).resolve() != Path(spec['source']) / 'opendm/exp/dm05_exp.py':
        raise AuthorReferenceError('Another OpenDM package is already imported')
    exp = module.DM05Exp(task='inference', logger_level='WARNING')
    exp.model_config.model_name_or_path = spec['checkpoint']
    exp.model_config.chunk_size = 50
    exp.data_config.is_history = False
    exp.inference_config.backend = 'default'
    exp.inference_config.diffusion_steps = 10
    exp.inference_config.enable_vlm_chat = False
    exp.initialize_inference_preprocessing(device='cpu')
    policy = exp.inference_config
    model_proof = prove_author_model(exp, spec)
    actions, ids, latency = [], [], []
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for index in range(len(workload)):
        request = workload.request(index)
        case = cases[request.request_id]
        state = np.asarray(request.observation['state'], dtype=np.float32)
        prompt = request.prompt if request.prompt is not None else request.observation.get('prompt')
        if hasattr(prompt, 'item'):
            prompt = prompt.item()
        if prompt != case['prompt'] or not np.array_equal(state, np.asarray(case['state'], dtype=np.float32)):
            raise AuthorReferenceError('Legacy DM prompt/state differs from the actual author case')
        images = request.observation.get('images')
        if not isinstance(images, (list, tuple)) or len(images) != 3:
            raise AuthorReferenceError('Legacy DM binding requires the three workload camera arrays in author order')
        author_images = [Image.open(Path(spec['source']) / 'assets/demo/images/episode0' / name / '0.jpg').convert('RGB')
                         for name in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')]
        if any(not np.array_equal(np.asarray(actual), np.asarray(expected))
               for actual, expected in zip(images, author_images)):
            raise AuthorReferenceError('Legacy DM workload images differ from the original author assets')
        directory = Path(saved_path).parent / case['id']
        noise = torch.load(directory / 'initial_noise.pt', map_location='cpu', weights_only=True)
        observed = torch.as_tensor(request.noise)
        if (tuple(observed.shape) not in ((50, 32), (1, 50, 32))
                or tuple(noise.shape) != (1, 50, 32)
                or not torch.equal(observed.reshape(noise.shape).float(), noise.float())):
            raise AuthorReferenceError('Legacy DM workload noise differs from the actual saved author draw')
        saved_prepared = torch.load(directory / 'prepared.pt', map_location='cpu', weights_only=True)
        data = policy._prepare_model_input(
            text=prompt, images=author_images, states=state.tolist(), robot_type='DOS W1')
        prepared = policy.input_transform(data)
        prepared = {key: value.detach().cpu() for key, value in prepared.items() if isinstance(value, torch.Tensor)}
        if set(prepared) != set(saved_prepared) or any(
                prepared[key].dtype != saved_prepared[key].dtype
                or not torch.equal(prepared[key], saved_prepared[key]) for key in prepared):
            raise AuthorReferenceError('Legacy DM author prepared tensors do not match this exact workload/checkpoint')
        saved_actions = np.load(directory / 'actions.npy', allow_pickle=False)
        samples = np.load(directory / 'samples_ms.npy', allow_pickle=False)
        count = raw['samples_per_case']
        if (saved_actions.shape != (count, 50, 14) or saved_actions.dtype != np.float32
                or samples.shape != (count,) or not np.isfinite(saved_actions).all()
                or not np.isfinite(samples).all() or np.any(samples <= 0)):
            raise AuthorReferenceError('Legacy DM saved output/sample coverage is incomplete or nonfinite')
        recorded = [sample['full_policy_ms'] for sample in raw['samples'] if sample['case'] == case['id']]
        if recorded != samples.tolist():
            raise AuthorReferenceError('Legacy DM JSON/NPY latency samples differ')
        actions.extend(saved_actions)
        ids.extend([case['id']] * count)
        latency.extend(samples.tolist())
    identity = {'model': 'dm05', 'workload': workload_identity(workload),
                'checkpoint': content_identity(spec['checkpoint']), 'author_source': source,
                'author_model': model_proof, 'author_model_proof_kind': 'original-producer-CPU-entrypoint',
                'output_domain': output_domain('dm05')}
    outputs = output / 'outputs.npz'
    np.savez_compressed(outputs, actions=np.stack(actions), request_ids=np.asarray(ids),
                        latency_ms=np.asarray(latency), output_domain=np.asarray(identity['output_domain']))
    original_command = [sys.executable, '-B', 'examples/vla_autotune/official_dm05_latency.py',
                        '--source', raw['source'], '--checkpoint', raw['checkpoint'],
                        '--output', str(Path(saved_path).parent), '--warmup', str(raw['warmup_per_case']),
                        '--repeats', str(raw['samples_per_case'])]
    record = {
        'kind': 'bound_saved_author_dm05', 'status': 'complete', 'origin': 'author', 'model': 'dm05',
        'identity': identity, 'producer_spec': spec, 'producer_command': original_command,
        'producer_command_provenance': 'Historical producer command reconstructed from its verified script SHA '
                                       'and recorded arguments; the legacy interpreter path was not recorded.',
        'binding_inputs': inputs, 'binding_command': list(getattr(sys, 'orig_argv', sys.argv)),
        'implementation': raw['execution'], 'job_id': raw['job_id'],
        'comparison_id': None, 'campaign_instance_id': None,
        'gpu': {'name': raw['gpu'], 'uuid': ''}, 'precision': 'bf16', 'sampler': 'euler', 'nfe': 10,
        'boundary': BOUNDARY, 'original_boundary': raw['boundary'],
        'warmup': raw['warmup_per_case'], 'repeats': len(actions),
        'latency_ms': latency, 'request_ids': ids,
        'outputs_path': str(outputs), 'outputs_sha256': file_sha256(outputs),
        'limitations': 'Saved author timing starts with PIL inputs and includes seeded RNG. '
                       'Historical weights are identified by the author revision claim plus the independent base weight pin; '
                       'the legacy producer did not hash its loaded weights. Prepared tensors and saved noise are verified exactly. '
                       'No same-campaign latency ratio or new model measurement is claimed.',
    }
    _write_json(output / 'result.json', record)
    return record


legacy_shape = [50, 14]
