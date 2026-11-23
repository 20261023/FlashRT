"""Opt-in author baselines, independent of the internal reference/bookend.

This is measurement-harness maintenance, never a candidate implementation.
The producer invokes the original authors' APIs in a separate process. Saved
outputs must retain their producer-time input/checkpoint/domain binding; older
unbound reports are evidence only and produce an actionable refusal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
import sys

import numpy as np

from .quality import compare_outputs, output_domain, quality_contract, validate_quality_contract
from .workloads import NPZWorkload


AUTHOR_SOURCES = {
    'pi0': ('https://github.com/Physical-Intelligence/openpi', 'src/openpi'),
    'pi05': ('https://github.com/Physical-Intelligence/openpi', 'src/openpi'),
    'hyvla': ('https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA', 'hy_vla'),
    'fastwam': ('https://github.com/yuantianyuan01/FastWAM', 'src/fastwam'),
    'dm05': ('https://github.com/dexmal/opendm', 'opendm'),
}
BOUNDARY = 'prepared-host-observation-and-noise_to_completed-CPU-actions'
_ENTRYPOINT = Path(__file__).resolve().parents[3] / 'examples/vla_autotune/author_baseline.py'
_OPTIONS = {
    'pi0': {'action_horizon', 'max_token_len'},
    'pi05': {'action_horizon', 'max_token_len'},
    'hyvla': set(),
    'fastwam': {'backbone_dir', 'model_config', 'sigma_shift', 'action_horizon',
                'compile_action_infer', 'noise_seeds'},
    'dm05': {'noise_seeds'},
}
_DM_SOURCE_REVISION = '03f799120d425ead51d9482bb1da950b6dabade9'
_DM_WEIGHT_REVISION = '5cd18734814abb075a9ccfd9ad6d16777b5cf10e'
_DM_WEIGHT_SHA256 = 'b7da77f516ebd0c0c68faed2f0cfe1c68985b7b874e49dda4292d8d286de20dd'
_DM_PYTHON_SHA256 = '28617ae3b6db5a887d6291e32bfac0406666f79188ec759dad5e22e1404dd462'
_PINNED_REVISIONS = {
    'dm05': _DM_SOURCE_REVISION,
    'hyvla': 'af57e7507ec5964b52fdf6296741e553cbcd3288',
    'fastwam': '7faa71108368fbb3b6885649f112af607427a2d4',
}


class AuthorReferenceError(ValueError):
    """An absent or incompatible author reference must never become a pass."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _write_json(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def content_identity(path):
    """Hash local checkpoint/config assets outside all measurement boundaries.

    Directory identities include every nonhidden file (including Orbax files
    without extensions). Hub cache metadata and Git internals are not weights.
    Paths are recorded separately and do not change content identity.
    """
    path = Path(path).resolve(strict=True)
    files = [path] if path.is_file() else sorted(
        entry for entry in path.rglob('*')
        if entry.is_file() and not any(part.startswith('.') for part in entry.relative_to(path).parts))
    if not files:
        raise AuthorReferenceError(f'No checkpoint assets found: {path}')
    records = [{'name': entry.name if path.is_file() else entry.relative_to(path).as_posix(),
                'bytes': entry.stat().st_size, 'sha256': file_sha256(entry)} for entry in files]
    return {'path': str(path), 'sha256': hashlib.sha256(_canonical(records).encode()).hexdigest(),
            'files': records}


def workload_identity(workload):
    """Bind the effective request values, order, prompts and typed fixed noise.

    An NPZ being repacked or copied changes neither its effective workload nor
    its digest. Changing a pixel, dtype, prompt, request ID or noise does.
    """
    def value(item):
        if isinstance(item, dict):
            return {key: value(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [value(part) for part in item]
        array = np.asarray(item)
        if array.dtype.hasobject:
            raise ValueError('Workload identities do not permit object arrays')
        return {'shape': list(array.shape), 'dtype': array.dtype.str,
                'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}
    requests = []
    for index in range(len(workload)):
        request = workload.request(index)
        requests.append({'request_id': request.request_id, 'prompt': request.prompt,
                         'observation': value(request.observation), 'noise': value(request.noise)})
    record = {'model': workload.model, 'workload_id': workload.workload_id, 'requests': requests}
    return {**record, 'sha256': hashlib.sha256(_canonical(record).encode()).hexdigest()}


def _absolute_text(value, name):
    if not isinstance(value, str) or not os.path.isabs(value):
        raise ValueError(f'Author baseline {name} must be an absolute path')
    return value


def _absolute(value, name):
    _absolute_text(value, name)
    return str(Path(value).resolve())


def _validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {
            'model', 'source', 'source_revision', 'checkpoint', 'num_steps', 'options'}:
        raise ValueError('Author producer spec requires model/source/source_revision/checkpoint/num_steps/options')
    model = spec['model']
    if model not in AUTHOR_SOURCES:
        raise ValueError('Author baseline supports pi0, pi05, hyvla, fastwam and dm05')
    for name in ('source', 'checkpoint'):
        _absolute_text(spec[name], name)
    if not isinstance(spec['source_revision'], str) or not re.fullmatch('[0-9a-f]{40}', spec['source_revision']):
        raise ValueError('Author source_revision must be a full lowercase Git commit')
    if model in _PINNED_REVISIONS and spec['source_revision'] != _PINNED_REVISIONS[model]:
        raise ValueError(f'This author {model} entrypoint requires source revision {_PINNED_REVISIONS[model]}')
    steps = spec['num_steps']
    if type(steps) is not int or steps < 1 or (model in ('hyvla', 'dm05') and steps != 10):
        raise ValueError('Author steps must be positive; Hy/DM currently require the original Euler10')
    options = spec['options']
    if not isinstance(options, dict) or set(options) - _OPTIONS[model]:
        raise ValueError(f'Unsupported author {model} options')
    for name in ('backbone_dir', 'model_config'):
        if name in options:
            _absolute_text(options[name], name)
    for name in ('action_horizon', 'max_token_len'):
        if name in options and (type(options[name]) is not int or options[name] < 1):
            raise ValueError(f'Author {name} must be a positive integer')
    if 'compile_action_infer' in options and type(options['compile_action_infer']) is not bool:
        raise ValueError('Author compile_action_infer must be bool')
    if 'sigma_shift' in options and (type(options['sigma_shift']) not in (int, float)
                                    or not math.isfinite(options['sigma_shift']) or options['sigma_shift'] <= 0):
        raise ValueError('Author sigma_shift must be positive finite')
    if model == 'fastwam' and 'backbone_dir' not in options:
        raise ValueError('Author FastWAM requires a local backbone_dir')
    if model in ('fastwam', 'dm05'):
        seeds = options.get('noise_seeds')
        if not isinstance(seeds, dict) or not seeds or any(
                not isinstance(key, str) or type(seed) is not int or not 0 <= seed < 2**63
                for key, seed in seeds.items()):
            raise ValueError('Author FastWAM/DM require options.noise_seeds mapping every request ID to a seed')
    _canonical(spec)
    return spec


def producer_command(spec, *, workload_path, output, warmup, repeats,
                     comparison_id, campaign_instance_id, python=None):
    """An executable argv, recorded verbatim; no shell or project model backend."""
    _validate_spec(spec)
    return [python or sys.executable, '-B', str(_ENTRYPOINT), '--spec-json', _canonical(spec),
            '--workload', str(workload_path), '--output', str(output),
            '--warmup', str(warmup), '--repeats', str(repeats),
            '--comparison-id', comparison_id, '--campaign-instance-id', campaign_instance_id]


def prepare_plan(config, *, baseline, workload_path, output_root, warmup, repeats,
                 comparison_id, campaign_instance_id):
    """Resolve one opt-in author reference; defaults are deliberately separate from E10."""
    allowed = {'mode', 'source', 'source_revision', 'num_steps', 'options', 'python', 'result_path',
               'extra_pythonpath'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Unknown author_baseline fields')
    mode = config.get('mode', 'produce')
    if mode not in ('produce', 'saved'):
        raise ValueError('author_baseline.mode must be produce or saved')
    model = baseline.backend.model
    spec = _validate_spec({
        'model': model, 'source': _absolute(config.get('source'), 'source'),
        'source_revision': config.get('source_revision'),
        'checkpoint': _absolute(baseline.backend.checkpoint, 'checkpoint'),
        'num_steps': config.get('num_steps', 10), 'options': config.get('options', {}),
    })
    workload_path, output_root = _absolute(str(workload_path), 'workload'), _absolute(str(output_root), 'output_root')
    workload = NPZWorkload.from_file(workload_path)
    if workload.model != model or workload.workload_id != baseline.workload_id:
        raise ValueError('Author baseline workload differs from the internal baseline')
    seeds = spec['options'].get('noise_seeds')
    if seeds is not None and set(seeds) != {item['request_id'] for item in workload.specs}:
        raise ValueError('Author noise_seeds must cover exactly the workload request IDs')
    python = config.get('python', sys.executable)
    # Resolving a venv's executable symlink bypasses pyvenv.cfg and switches
    # to the base installation, losing the selected author's dependencies.
    if not isinstance(python, str) or not os.path.isabs(python):
        raise ValueError('Author baseline python must be an absolute executable path')
    extras = config.get('extra_pythonpath', [])
    if not isinstance(extras, list):
        raise ValueError('extra_pythonpath must list absolute author dependency directories')
    for extra in extras:
        _absolute(extra, 'extra_pythonpath')
        if os.pathsep in extra or (Path(extra) / 'flash_rt').exists():
            raise ValueError('Author dependency entries must be single paths without another FlashRT checkout')
    target = str(Path(output_root) / 'author')
    command = producer_command(spec, workload_path=workload_path, output=target, warmup=warmup,
                               repeats=repeats, comparison_id=comparison_id,
                               campaign_instance_id=campaign_instance_id, python=python)
    record = {'schema_version': 1, 'mode': mode, 'spec': spec,
              'comparison_id': comparison_id, 'campaign_instance_id': campaign_instance_id,
              'workload_identity': workload_identity(workload), 'output_domain': output_domain(model),
              'producer_command': command, 'output': target,
              'result_path': str(Path(target) / 'result.json')}
    if extras:
        record['extra_pythonpath'] = list(extras)
    if mode == 'saved':
        record['result_path'] = _absolute(config.get('result_path'), 'result_path')
        try:
            saved = json.loads(Path(record['result_path']).read_text())
            if model == 'dm05' and saved.get('origin') == 'author' and 'identity' not in saved:
                inputs = _legacy_dm_inputs(record['result_path'], spec)
                target = str(Path(output_root) / 'author-bound-saved')
                binding = producer_command(spec, workload_path=workload_path, output=target, warmup=warmup,
                                           repeats=repeats, comparison_id=comparison_id,
                                           campaign_instance_id=campaign_instance_id, python=python)
                record.update(
                    saved_inputs=inputs, output=target, result_path=str(Path(target) / 'result.json'),
                    binding_command=[*binding, '--bind-dm05-saved', inputs['result_path']])
                return record
            result = _read_author_result(record['result_path'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise AuthorReferenceError(f'Saved author reference is unavailable or unbound: {error}. '
                                       f'Regenerate with argv: {_canonical(command)}') from error
        _check_expected(result, record)
        record['saved_digest'] = {
            'result_sha256': file_sha256(record['result_path']),
            'outputs_sha256': file_sha256(result['outputs_path'])}
        record['saved_producer_command'] = result['producer_command']
    elif config.get('result_path') is not None:
        raise ValueError('result_path is only accepted with author mode=saved')
    return record


def source_identity(spec):
    """Admit a clean author checkout or explicitly marked exported author files."""
    root = Path(spec['source']).resolve(strict=True)
    _, package = AUTHOR_SOURCES[spec['model']]
    package = root / package
    if not package.is_dir() or (root / 'flash_rt').exists():
        raise AuthorReferenceError(f'Expected author package is missing, or source is FlashRT: {package}')
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel', 'HEAD'],
                          capture_output=True, text=True, check=False)
    proof = 'export-marker-and-recorded-source-digests'
    if head.returncode == 0 and Path(head.stdout.splitlines()[0]).resolve() == root:
        if head.stdout.splitlines()[-1] != spec['source_revision']:
            raise AuthorReferenceError('Author checkout revision differs from the requested author commit')
        status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                                capture_output=True, text=True, check=True)
        if status.stdout.strip():
            raise AuthorReferenceError('Author checkout has tracked modifications')
        proof = 'clean-author-git-checkout-and-recorded-source-digests'
    else:
        markers = [root / name for name in ('AUTHOR_GIT_COMMIT', 'GIT_COMMIT')]
        marker_ok = any(path.is_file() and path.read_text().strip() == spec['source_revision'] for path in markers)
        source_manifest = root.parent / 'SOURCE.json'
        if source_manifest.is_file():
            manifest = json.loads(source_manifest.read_text())
            marker_ok |= (manifest.get('commit') == spec['source_revision']
                          and manifest.get('source_subdirectory') == root.name)
        if spec['model'] == 'dm05' and spec['source_revision'] == _DM_SOURCE_REVISION:
            pinned = _dm_python_digest(root) == _DM_PYTHON_SHA256
            marker_ok |= pinned
            if pinned:
                proof = 'independently-pinned-author-Python-bytes'
        if not marker_ok:
            raise AuthorReferenceError('Exported author source needs its exact AUTHOR_GIT_COMMIT/GIT_COMMIT or SOURCE.json')
    files = {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(package.rglob('*.py'))}
    if not files:
        raise AuthorReferenceError('Author source contains no Python implementation')
    return {'path': str(root), 'revision': spec['source_revision'],
            'source_url': AUTHOR_SOURCES[spec['model']][0], 'proof': proof, 'python_files': files}


def _dm_python_digest(root):
    package, digest = Path(root) / 'opendm', hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(path.relative_to(package).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def _legacy_dm_inputs(path, spec):
    """Pin the actual legacy author bundle, without importing Torch on init."""
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    script = _ENTRYPOINT.with_name('official_dm05_latency.py')
    required = {'model': 'dm05', 'origin': 'author', 'source_revision': _DM_SOURCE_REVISION,
                'checkpoint_revision': _DM_WEIGHT_REVISION, 'steps': 10, 'nfe': 10,
                'output_action_dim': 14, 'internal_action_dim': 32, 'chunk': 50}
    if (any(record.get(key) != value for key, value in required.items())
            or spec['source_revision'] != _DM_SOURCE_REVISION or spec['num_steps'] != 10
            or record.get('script_sha256') != file_sha256(script)):
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


def bind_dm05_saved(spec, workload, saved_path, output):
    """CPU-only binding of original DM four-case outputs, never new inference.

    Exact current CPU transforms must reproduce the saved prepared tensors.
    This is needed because the old report did not record raw input digests.
    """
    import importlib
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from PIL import Image
    spec = _validate_spec(spec)
    inputs = _legacy_dm_inputs(saved_path, spec)
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
    original_command = [sys.executable, '-B', str(_ENTRYPOINT.with_name('official_dm05_latency.py')),
                        '--source', raw['source'], '--checkpoint', raw['checkpoint'],
                        '--output', str(Path(saved_path).parent), '--warmup', str(raw['warmup_per_case']),
                        '--repeats', str(raw['samples_per_case'])]
    record = {
        'kind': 'bound_saved_author_dm05', 'status': 'complete', 'origin': 'author', 'model': 'dm05',
        'identity': identity, 'producer_spec': spec, 'producer_command': original_command,
        'producer_command_provenance': 'Reconstructed runnable original producer command from its verified script SHA '
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


def prove_author_model(model, spec):
    """Check the imported model's actual file; project code cannot impersonate it."""
    import inspect
    path = Path(inspect.getfile(type(model))).resolve()
    package = Path(spec['source']).resolve() / AUTHOR_SOURCES[spec['model']][1]
    if not path.is_relative_to(package):
        raise AuthorReferenceError(f'Imported author model is outside its declared author package: {path}')
    forbidden = ('flash_rt.models', 'flash_rt.frontends', 'flash_rt.flash_rt_kernels',
                 'flash_rt.flash_rt_fa2', 'flash_rt.flash_rt_gemm_plan')
    if any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in forbidden):
        raise AuthorReferenceError('Author producer imported a FlashRT neural implementation')
    return {'class': type(model).__module__ + '.' + type(model).__qualname__,
            'file': str(path), 'sha256': file_sha256(path)}


def _read_author_result(path, *, outputs_path=None):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get('origin') != 'author' or record.get('status') != 'complete':
        raise AuthorReferenceError('Reference must be a complete origin=author result, never a project/internal result')
    identity = record.get('identity')
    if (not isinstance(identity, dict) or any(
            not isinstance(identity.get(key), dict) for key in (
                'workload', 'checkpoint', 'author_source', 'author_model'))
            or not identity.get('workload', {}).get('sha256')
            or not identity.get('checkpoint', {}).get('sha256')
            or not identity.get('author_source', {}).get('revision')
            or not identity.get('author_model', {}).get('file')):
        raise AuthorReferenceError('Legacy author report lacks producer-time workload/checkpoint/source identity')
    spec = _validate_spec(record.get('producer_spec'))
    model_file = Path(identity['author_model']['file'])
    if not model_file.is_relative_to(Path(spec['source']) / AUTHOR_SOURCES[spec['model']][1]):
        raise AuthorReferenceError('Saved reference model file is not from the declared author package')
    if identity.get('output_domain') != output_domain(spec['model']):
        raise AuthorReferenceError('Saved author output domain differs from the model contract')
    source = identity['author_source']
    relative_model = model_file.relative_to(Path(spec['source'])).as_posix()
    if (identity.get('model') != spec['model'] or record.get('model') != spec['model']
            or source.get('revision') != spec['source_revision']
            or source.get('source_url') != AUTHOR_SOURCES[spec['model']][0]
            or not isinstance(source.get('python_files'), dict)
            or source.get('python_files', {}).get(relative_model) != identity['author_model'].get('sha256')
            or record.get('nfe') != spec['num_steps'] or record.get('sampler') != 'euler'):
        raise AuthorReferenceError('Saved author source/model/sampling proof is inconsistent')
    command = record.get('producer_command')
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise AuthorReferenceError('Saved author reference must retain its actual producer argv')
    outputs = Path(record.get('outputs_path', path.parent / 'outputs.npz'))
    if not outputs.is_absolute():
        outputs = path.parent / outputs
    record['outputs_path'] = str(outputs.resolve()) if outputs_path is None else str(outputs)
    payload = Path(outputs_path) if outputs_path is not None else outputs
    if not payload.is_file() or file_sha256(payload) != record.get('outputs_sha256'):
        raise AuthorReferenceError('Saved author output file is missing or its digest differs')
    samples = record.get('latency_ms')
    if (not isinstance(samples, list) or not samples or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples)):
        raise AuthorReferenceError('Author result has no complete finite latency samples')
    ids = record.get('request_ids')
    if (type(record.get('repeats')) is not int or record['repeats'] != len(samples)
            or not isinstance(ids, list) or len(ids) != len(samples)
            or any(not isinstance(item, str) or not item for item in ids)
            or record.get('boundary') != BOUNDARY):
        raise AuthorReferenceError('Author result sample coverage or measurement boundary differs')
    return record


def _check_expected(result, planned):
    if result['producer_spec'] != planned['spec']:
        raise AuthorReferenceError('Author producer source/checkpoint/options/steps differ from the plan')
    if result['identity']['workload'] != planned['workload_identity']:
        raise AuthorReferenceError('Author fixture values/request IDs/order/noise differ from the campaign')
    if result['identity']['output_domain'] != planned['output_domain']:
        raise AuthorReferenceError('Author and campaign output domains differ')
    if planned.get('saved_inputs') is not None and result.get('binding_inputs') != planned['saved_inputs']:
        raise AuthorReferenceError('Bound saved author output does not describe the pinned original bundle')
    if planned['mode'] == 'produce' and any(
            result.get(key) != planned[key] for key in ('comparison_id', 'campaign_instance_id')):
        raise AuthorReferenceError('Author result belongs to a different campaign instance')


def prepare_reference(plan, *, execute):
    """Run the producer before any internal attempt, or verify pinned saved data."""
    planned = plan['author_baseline']
    try:
        command = (planned['producer_command'] if planned['mode'] == 'produce'
                   else planned.get('binding_command'))
        if command is not None:
            if planned.get('saved_inputs') is not None:
                for path, digest in planned['saved_inputs']['files'].items():
                    if file_sha256(path) != digest:
                        raise AuthorReferenceError('Original saved author artifact changed after campaign initialization')
            # An explicit author python is part of the plan, independent of a
            # snapshot worker's interpreter (e.g. original Hy's Torch version).
            output = Path(plan['output_root'])
            cwd = output / 'work' / 'author'
            cwd.mkdir(parents=True)
            env = {key: value for key, value in os.environ.items() if key not in (
                'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE')}
            env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
            if planned.get('extra_pythonpath'):
                env['PYTHONPATH'] = os.pathsep.join(planned['extra_pythonpath'])
            with (output / 'author.log').open('x') as log:
                completed = execute(command, cwd=str(cwd), env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise AuthorReferenceError(f'Author producer failed ({completed.returncode}); inspect {output / "author.log"}')
        result_path = planned['result_path']
        result = _read_author_result(result_path)
        _check_expected(result, planned)
        digests = {'result_sha256': file_sha256(result_path),
                   'outputs_sha256': file_sha256(result['outputs_path'])}
        if planned['mode'] == 'saved' and 'saved_digest' in planned and digests != planned['saved_digest']:
            raise AuthorReferenceError('Saved author result changed after campaign initialization')
        if planned['mode'] == 'produce' and any(
                result.get(key) != plan[key] for key in ('comparison_id', 'campaign_instance_id')):
            raise AuthorReferenceError('Produced author output belongs to a different campaign instance')
        descriptor = {'schema_version': 1, 'origin': 'author', 'result_path': result_path,
                      **digests, 'planned': planned,
                      'provenance': 'same_campaign' if planned['mode'] == 'produce' else 'saved_author_outputs',
                      'producer_command': result['producer_command']}
        control = Path(plan['output_root']) / 'control'
        control.mkdir(parents=True, exist_ok=True)
        path = control / 'author_reference.json'
        _write_json(path, descriptor)
        return str(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, AuthorReferenceError):
            raise
        raise AuthorReferenceError(f'Author reference is missing or mismatched: {error}') from error


def load_reference(descriptor, *, workload, backend):
    """Verify current fixtures/checkpoint and read original-author NPZ arrays."""
    try:
        descriptor = json.loads(Path(descriptor).read_text()) if isinstance(descriptor, (str, Path)) else descriptor
        path = descriptor['result_path']
        planned = descriptor['planned']
        expected_provenance = 'same_campaign' if planned['mode'] == 'produce' else 'saved_author_outputs'
        if path != planned['result_path'] or descriptor.get('provenance') != expected_provenance:
            raise AuthorReferenceError('Author descriptor path/provenance differs from the plan')
        if descriptor.get('origin') != 'author' or file_sha256(path) != descriptor['result_sha256']:
            raise AuthorReferenceError('Author reference result is missing or changed')
        result = _read_author_result(path)
        _check_expected(result, descriptor['planned'])
        identity = result['identity']
        if identity['workload'] != workload_identity(workload) or result['producer_spec']['model'] != backend.model:
            raise AuthorReferenceError('Author reference fixture/model identity differs from this worker')
        if identity['output_domain'] != output_domain(backend.model):
            raise AuthorReferenceError('Author reference uses a different action domain')
        if identity['checkpoint']['sha256'] != content_identity(backend.checkpoint)['sha256']:
            raise AuthorReferenceError('Worker checkpoint content differs from the author checkpoint')
        if backend.model == 'fastwam':
            options = backend.options
            expected_options = result['producer_spec']['options']
            for key, default in (('sigma_shift', 5.0), ('action_horizon', 32)):
                if options.get(key, default) != expected_options.get(key, default):
                    raise AuthorReferenceError(f'Worker FastWAM {key} differs from the author model contract')
            vae = Path(options['backbone_dir']) / 'Wan2.2_VAE.pth'
            if content_identity(vae)['sha256'] != identity.get('auxiliary_assets', {}).get('vae', {}).get('sha256'):
                raise AuthorReferenceError('Worker FastWAM VAE weights differ from the author checkpoint')
        if file_sha256(result['outputs_path']) != descriptor['outputs_sha256']:
            raise AuthorReferenceError('Author outputs changed after reference preparation')
        actions, ids = _reference_actions(result, result['outputs_path'])
        return {'descriptor': descriptor, 'record': result, 'actions': actions, 'request_ids': ids}
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, AuthorReferenceError):
            raise
        raise AuthorReferenceError(f'Author reference is missing or mismatched: {error}') from error


def _reference_actions(result, path):
    """Read portable payloads without changing the archived cluster provenance."""
    identity = result['identity']
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['request_ids'].tolist()
        if ids != result['request_ids'] or len(ids) != len(result['latency_ms']):
            raise AuthorReferenceError('Author NPZ/JSON request coverage differs')
        if str(archive['output_domain'].item()) != identity['output_domain']:
            raise AuthorReferenceError('Author NPZ/JSON output domains differ')
        if not np.array_equal(archive['latency_ms'], np.asarray(result['latency_ms'], dtype=np.float64)):
            raise AuthorReferenceError('Author NPZ/JSON latency samples differ')
        actions = [np.array(action, copy=True) for action in archive['actions']]
    if (len(actions) != len(ids)
            or set(ids) != {item['request_id'] for item in identity['workload']['requests']}):
        raise AuthorReferenceError('Author outputs do not cover the full workload')
    if any(action.dtype != np.float32 or not action.size or not np.isfinite(action).all() for action in actions):
        raise AuthorReferenceError('Author reference actions must be nonempty finite FP32 CPU arrays')
    expected_shape = result.get('metadata', {}).get('expected_shape')
    if expected_shape is None and result['model'] == 'dm05':
        expected_shape = [50, 14]
    if expected_shape is None or any(list(action.shape) != expected_shape for action in actions):
        raise AuthorReferenceError('Author reference action shape differs from its declared model output')
    return actions, ids


def _plan_digest(plan):
    return hashlib.sha256(_canonical(plan).encode()).hexdigest()


def _portable_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AuthorReferenceError('Portable evidence paths must be relative to the fetched results')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise AuthorReferenceError('Portable evidence path escapes the fetched results')
    return path


def _checked_file(root, artifact):
    if not isinstance(artifact, dict):
        raise AuthorReferenceError('Portable evidence is missing a file binding')
    path = _portable_path(root, artifact['path'])
    if file_sha256(path) != artifact['sha256']:
        raise AuthorReferenceError(f'Portable evidence digest differs: {artifact["path"]}')
    return path


def _copy_evidence(source, target, digest):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source != target:
        temporary = target.with_suffix(target.suffix + '.tmp')
        with source.open('rb') as original, temporary.open('xb') as copied:
            shutil.copyfileobj(original, copied, length=8 * 1024 * 1024)
        if file_sha256(temporary) != digest:
            raise AuthorReferenceError('Author artifact changed during portable evidence export')
        temporary.replace(target)
    if file_sha256(target) != digest:
        raise AuthorReferenceError('Portable author artifact digest differs')


def export_portable_reference(plan, descriptor_path, *, batch_job_id=None):
    """Batch-side export: fetch control/ plus attempt bundles to ingest anywhere."""
    descriptor_path = Path(descriptor_path)
    descriptor = json.loads(descriptor_path.read_text())
    if descriptor['planned'] != plan['author_baseline']:
        raise AuthorReferenceError('Cannot export an author descriptor for a different plan')
    record = _read_author_result(descriptor['result_path'])
    _check_expected(record, plan['author_baseline'])
    if (file_sha256(descriptor['result_path']) != descriptor['result_sha256']
            or record['outputs_sha256'] != descriptor['outputs_sha256']):
        raise AuthorReferenceError('Author reference changed before portable evidence export')
    output = Path(plan['output_root'])
    control = output / 'control'
    control.mkdir(parents=True, exist_ok=True)
    descriptor_digest = file_sha256(descriptor_path)
    _copy_evidence(descriptor_path, control / 'author_reference.json', descriptor_digest)
    _copy_evidence(descriptor['result_path'], control / 'author_result.json', descriptor['result_sha256'])
    _copy_evidence(record['outputs_path'], control / 'author_outputs.npz', descriptor['outputs_sha256'])
    _write_json(control / 'campaign_plan.json', plan)
    job_id = batch_job_id if batch_job_id is not None else os.environ.get('SLURM_JOB_ID')
    if not job_id or (descriptor['provenance'] == 'same_campaign' and job_id != record.get('job_id')):
        raise AuthorReferenceError('Portable evidence must identify the batch allocation and original author job')
    files = {
        'plan': {'path': 'control/campaign_plan.json', 'sha256': file_sha256(control / 'campaign_plan.json')},
        'descriptor': {'path': 'control/author_reference.json', 'sha256': descriptor_digest},
        'author_result': {'path': 'control/author_result.json', 'sha256': descriptor['result_sha256']},
        'author_outputs': {'path': 'control/author_outputs.npz', 'sha256': descriptor['outputs_sha256']},
    }
    manifest = {'schema_version': 1, 'kind': 'portable_author_evidence',
                'comparison_id': plan['comparison_id'], 'campaign_instance_id': plan['campaign_instance_id'],
                'plan_sha256': _plan_digest(plan), 'job_id': job_id, 'files': files,
                'executed_gpu': None, 'attempts': {}}
    _write_json(control / 'author_portable.json', manifest)
    return manifest


def _check_plan_envelope(plan, manifest):
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
            or manifest.get('kind') != 'portable_author_evidence'
            or manifest.get('plan_sha256') != _plan_digest(plan)
            or any(manifest.get(key) != plan[key] for key in ('comparison_id', 'campaign_instance_id'))):
        raise AuthorReferenceError('Portable evidence belongs to another plan/campaign instance')
    if not isinstance(manifest.get('job_id'), str) or not manifest['job_id']:
        raise AuthorReferenceError('Portable evidence is missing its actual batch job identity')


def _check_device(actual, expected):
    """Same-job measurements require an actual common GPU, never a name alone."""
    if actual.get('job_id') != expected.get('job_id') or not actual.get('job_id'):
        raise AuthorReferenceError('Worker and author/batch job identities differ')
    left, right = actual.get('gpu', {}), expected.get('gpu', {})
    if (not isinstance(left, dict) or not isinstance(right, dict)
            or not isinstance(left.get('uuid'), str) or not isinstance(right.get('uuid'), str)
            or left['uuid'].lower() in ('', 'none', 'unknown', 'unavailable')
            or right['uuid'].lower() in ('', 'none', 'unknown', 'unavailable')
            or left['uuid'] != right['uuid']
            or not left.get('name') or left['name'] != right.get('name')
            or left.get('capability') != right.get('capability')
            or left.get('job_id', actual['job_id']) != actual['job_id']):
        raise AuthorReferenceError('Worker and author/batch GPU identities differ or are incomplete')


def record_portable_attempt(plan, entry, record, result_path):
    """Pin each complete remote worker result, its CPU outputs and runtime evidence."""
    output = Path(plan['output_root'])
    manifest_path = output / 'control/author_portable.json'
    manifest = json.loads(manifest_path.read_text())
    _check_plan_envelope(plan, manifest)
    for key in ('comparison_id', 'campaign_instance_id', 'job_id'):
        if record.get(key) != manifest[key]:
            raise AuthorReferenceError('Worker result does not belong to the exported batch')
    if record.get('attempt_id') != entry['attempt_id'] or record.get('candidate') != entry['candidate']:
        raise AuthorReferenceError('Worker attempt/candidate differs from the portable plan')
    relative = f'{entry["attempt_id"]}/result.json'
    if Path(result_path).resolve() != _portable_path(output, relative):
        raise AuthorReferenceError('Worker result file is outside its planned attempt directory')
    binding = {'result': {'path': relative, 'sha256': file_sha256(result_path)},
               'state': record['state'], 'runtime': record.get('runtime')}
    if record['state'] == 'measured':
        if manifest['executed_gpu'] is None:
            manifest['executed_gpu'] = record['gpu']
        _check_device(record, {'job_id': manifest['job_id'], 'gpu': manifest['executed_gpu']})
        relative = f'{entry["attempt_id"]}/outputs.npz'
        path = _portable_path(output, relative)
        if Path(record['outputs_path']).resolve() != path:
            raise AuthorReferenceError('Worker outputs are outside their planned attempt directory')
        binding['outputs'] = {'path': relative, 'sha256': file_sha256(path)}
    manifest['attempts'][entry['attempt_id']] = binding
    _write_json(manifest_path, manifest)


def export_portable_campaign(plan):
    """CPU-only upgrade of a trusted pre-portable batch; reuse its recorded job."""
    from .schema import validate_result
    output = Path(plan['output_root'])
    batch = json.loads((output / 'batch_result.json').read_text())
    if (any(batch.get(key) != plan[key] for key in ('comparison_id', 'campaign_instance_id'))
            or not isinstance(batch.get('job_id'), str) or not batch['job_id']):
        raise AuthorReferenceError('Archived batch identity differs from the original plan')
    points = {entry['attempt_id']: entry for entry in plan['entries']}
    completed = batch.get('entries')
    if (not isinstance(completed, list) or len({entry['attempt_id'] for entry in completed}) != len(completed)
            or any(entry['attempt_id'] not in points for entry in completed)):
        raise AuthorReferenceError('Archived batch has unexpected/duplicate attempt identities')
    records = []
    for item in completed:
        path = _portable_path(output, f'{item["attempt_id"]}/result.json')
        if not path.is_file() and item.get('state') is None:
            continue
        record = json.loads(path.read_text())
        validate_result(record)
        if (record.get('job_id') != batch['job_id']
                or record.get('candidate_id') != item['candidate_id']
                or record.get('state') != item.get('state')):
            raise AuthorReferenceError('Archived result differs from its batch envelope')
        records.append((points[item['attempt_id']], record, path))
    export_portable_reference(plan, output / 'control/author_reference.json', batch_job_id=batch['job_id'])
    for entry, record, path in records:
        record_portable_attempt(plan, entry, record, path)
    return output / 'control/author_portable.json'


def load_portable_reference(plan, results_dir):
    """Local ingest: verify fetched evidence without any cluster files or imports."""
    try:
        root = Path(results_dir)
        manifest = json.loads((root / 'control/author_portable.json').read_text())
        _check_plan_envelope(plan, manifest)
        files = manifest['files']
        names = {'plan': 'control/campaign_plan.json', 'descriptor': 'control/author_reference.json',
                 'author_result': 'control/author_result.json', 'author_outputs': 'control/author_outputs.npz'}
        if set(files) != set(names) or any(files[key]['path'] != path for key, path in names.items()):
            raise AuthorReferenceError('Fetched portable control files have unexpected names')
        if not isinstance(manifest.get('attempts'), dict) or set(manifest['attempts']) - {
                entry['attempt_id'] for entry in plan['entries']}:
            raise AuthorReferenceError('Portable evidence names an attempt outside the original plan')
        copy = json.loads(_checked_file(root, files['plan']).read_text())
        if copy != plan:
            raise AuthorReferenceError('Fetched batch plan differs from the original controller plan')
        descriptor = json.loads(_checked_file(root, files['descriptor']).read_text())
        if descriptor['planned'] != plan['author_baseline']:
            raise AuthorReferenceError('Fetched author descriptor differs from the original plan')
        expected = 'same_campaign' if plan['author_baseline']['mode'] == 'produce' else 'saved_author_outputs'
        if (descriptor.get('origin') != 'author' or descriptor.get('provenance') != expected
                or descriptor['result_path'] != plan['author_baseline']['result_path']
                or files['author_result']['sha256'] != descriptor['result_sha256']
                or files['author_outputs']['sha256'] != descriptor['outputs_sha256']):
            raise AuthorReferenceError('Portable author file bindings differ from the original descriptor')
        if 'saved_digest' in plan['author_baseline'] and plan['author_baseline']['saved_digest'] != {
                key: descriptor[key] for key in ('result_sha256', 'outputs_sha256')}:
            raise AuthorReferenceError('Fetched saved author reference differs from its initialized digests')
        payload = _checked_file(root, files['author_outputs'])
        record = _read_author_result(_checked_file(root, files['author_result']), outputs_path=payload)
        _check_expected(record, plan['author_baseline'])
        if descriptor['producer_command'] != record['producer_command']:
            raise AuthorReferenceError('Fetched producer argv differs from the bound author result')
        if expected == 'same_campaign' and record.get('job_id') != manifest['job_id']:
            raise AuthorReferenceError('Fetched author reference belongs to another batch job')
        actions, ids = _reference_actions(record, payload)
        return {'descriptor': descriptor, 'record': record, 'actions': actions, 'request_ids': ids,
                'portable': {'manifest': manifest, 'root': str(root)}}
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, AuthorReferenceError):
            raise
        raise AuthorReferenceError(f'Portable author evidence is missing or mismatched: {error}') from error


def _worker_payload(record, root, binding):
    path = _checked_file(root, binding['outputs'])
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['request_ids'].tolist()
        if (str(archive['output_domain'].item()) != record['output_domain']
                or str(archive['workload_digest'].item()) != record['workload_identity']['sha256']):
            raise AuthorReferenceError('Fetched worker action domain/fixture digest differs')
        actions = [archive[f'actions_{index:05d}'].copy() for index in range(len(ids))]
    if (len(ids) != record['measurement']['repeats']
            or set(ids) != set(record['workload_request_ids'])
            or any(action.dtype != np.float32 or not action.size or not np.isfinite(action).all() for action in actions)):
        raise AuthorReferenceError('Fetched worker action sample coverage/dtype is invalid')
    requests = record['workload_identity']['requests']
    if not requests or ids != [requests[index % len(requests)]['request_id'] for index in range(len(ids))]:
        raise AuthorReferenceError('Fetched worker request order differs from the original workload')
    return actions, ids


def _check_recomputed_quality(claimed, observed):
    for key in ('status', 'scope', 'action_dim', 'gate'):
        if claimed.get(key) != observed.get(key):
            raise AuthorReferenceError('Fetched actions do not reproduce the worker quality contract/verdict')
    for key, value in observed['metrics'].items():
        measured = claimed.get('metrics', {}).get(key)
        if (type(measured) not in (int, float) or not math.isfinite(measured)
                or not math.isclose(measured, value, rel_tol=1e-12, abs_tol=1e-15)):
            raise AuthorReferenceError(f'Fetched actions do not reproduce worker quality metric: {key}')


def validate_portable_attempt(plan, entry, record, result_path, reference):
    """Check transport integrity and recompute both gates from fetched CPU arrays."""
    try:
        portable = reference['portable']
        manifest, root = portable['manifest'], portable['root']
        binding = manifest['attempts'][entry['attempt_id']]
        if (binding['result']['path'] != f'{entry["attempt_id"]}/result.json'
                or (record['state'] == 'measured'
                    and binding['outputs']['path'] != f'{entry["attempt_id"]}/outputs.npz')):
            raise AuthorReferenceError('Portable worker files were swapped between attempts')
        expected = _checked_file(root, binding['result'])
        if expected != Path(result_path).resolve() or json.loads(expected.read_text()) != record:
            raise AuthorReferenceError('Fetched result is not the pinned worker envelope')
        if (record.get('candidate') != entry['candidate'] or record['attempt_id'] != entry['attempt_id']
                or any(record.get(key) != manifest[key]
                       for key in ('campaign_instance_id', 'comparison_id', 'job_id'))
                or binding['state'] != record['state'] or binding.get('runtime') != record.get('runtime')):
            raise AuthorReferenceError('Fetched worker candidate/instance/job/runtime identity differs')
        if record['state'] != 'measured':
            return
        _check_device(record, {'job_id': manifest['job_id'], 'gpu': manifest['executed_gpu']})
        validate_worker_reference(record, reference, plan.get('quality_gate'))
        actions, ids = _worker_payload(record, root, binding)
        observed = compare_reference(actions, ids, reference, plan.get('quality_gate'))
        _check_recomputed_quality(record['author_quality'], observed)
        baseline = next((item for item in plan['entries'] if item['role'] == 'reference'), None)
        if baseline is None:
            raise AuthorReferenceError('Portable plan has no internal reference attempt')
        baseline_binding = manifest['attempts'][baseline['attempt_id']]
        baseline_record = json.loads(_checked_file(root, baseline_binding['result']).read_text())
        baseline_actions, baseline_ids = _worker_payload(baseline_record, root, baseline_binding)
        if entry['role'] == 'reference':
            # The internal reference is defined as a self-reference. Its exact
            # identity metrics do not depend on cross-host BLAS rounding.
            observed = {'status': 'passed', **quality_contract(plan.get('quality_gate')),
                        'metrics': {'max_relative_l2': 0.0, 'max_abs': 0.0, 'min_cosine': 1.0}}
        else:
            observed = compare_outputs(actions, ids, baseline_actions, baseline_ids, plan.get('quality_gate'),
                                       domain=record['output_domain'])
        _check_recomputed_quality(record['quality'], observed)
        if bool(record['quality'].get('reference_role')) != (entry['role'] == 'reference'):
            raise AuthorReferenceError('Internal self-reference role differs from the plan')
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, AuthorReferenceError):
            raise
        raise AuthorReferenceError(f'Fetched worker evidence is missing or mismatched: {error}') from error


def reference_evidence(reference):
    descriptor, result = reference['descriptor'], reference['record']
    return {key: descriptor[key] for key in (
        'result_path', 'result_sha256', 'outputs_sha256', 'provenance', 'producer_command')} | {
        'origin': 'author', 'model': result['producer_spec']['model'],
        'output_domain': result['identity']['output_domain'],
        'workload_digest': result['identity']['workload']['sha256'],
        'checkpoint_digest': result['identity']['checkpoint']['sha256'],
        'author_source_revision': result['producer_spec']['source_revision'],
        'author_comparison_id': result.get('comparison_id'),
        'author_campaign_instance_id': result.get('campaign_instance_id'),
        'author_job_id': result.get('job_id'), 'author_gpu': result.get('gpu'),
        'source_url': result['identity']['author_source']['source_url'],
        'reference_kind': result.get('kind'),
        'producer_command_provenance': result.get('producer_command_provenance', 'captured_actual_argv'),
        'limitations': result.get('limitations'),
        'sampler': 'euler', 'nfe': result['producer_spec']['num_steps'],
    }


def compare_reference(actions, ids, reference, gate):
    try:
        return {**compare_outputs(actions, ids, reference['actions'], reference['request_ids'], gate,
                                  reference_origin='author', domain=reference['record']['identity']['output_domain']),
                'reference': reference_evidence(reference)}
    except ValueError as error:
        raise AuthorReferenceError(f'Author reference/action contract mismatch: {error}') from error


def validate_worker_reference(record, reference, gate):
    quality = record.get('author_quality')
    if not isinstance(quality, dict) or quality.get('reference_origin') != 'author':
        raise AuthorReferenceError('Opt-in campaign worker has no author quality evidence')
    validate_quality_contract(quality, gate)
    if (quality.get('reference') != reference_evidence(reference)
            or record.get('output_domain') != reference['record']['identity']['output_domain']
            or quality.get('output_domain') != record.get('output_domain')
            or record.get('workload_identity') != reference['record']['identity']['workload']):
        raise AuthorReferenceError('Worker author quality belongs to a different reference/domain/workload')
    if reference['descriptor']['provenance'] == 'same_campaign' and any(
            record.get(key) != reference['record'].get(key)
            for key in ('comparison_id', 'campaign_instance_id', 'job_id')):
        raise AuthorReferenceError('Worker and produced author reference do not belong to the same campaign/job')
    if reference['descriptor']['provenance'] == 'same_campaign':
        _check_device(record, reference['record'])


def latency_comparison(record, reference):
    """Keep author quality and strict latency comparability as separate evidence."""
    author = reference['record']
    reasons = []
    if reference['descriptor']['provenance'] != 'same_campaign':
        reasons.append('saved author outputs are a numerical reference, not a same-campaign latency measurement')
    for key in ('comparison_id', 'campaign_instance_id', 'job_id'):
        if not record.get(key) or record.get(key) != author.get(key):
            reasons.append(f'{key} differs (saved reference is not a same-campaign timing)')
    gpu, other = record.get('gpu', {}), author.get('gpu', {})
    if not gpu.get('uuid') or gpu['uuid'] != other.get('uuid') or gpu.get('name') != other.get('name'):
        reasons.append('GPU identity differs or is missing')
    measurement = record['measurement']
    if measurement.get('boundary') != author.get('boundary'):
        reasons.append('timing boundary differs')
    if record.get('output_domain') != author['identity']['output_domain']:
        reasons.append('output domain differs')
    sampler = measurement['sampler_metadata']
    if any(item.get('sampler') != 'euler' or item.get('nfe') != author['nfe'] for item in sampler):
        reasons.append('sampler/NFE differs')
    if record['backend_metadata'].get('precision') != author.get('precision'):
        reasons.append('precision differs')
    contract = measurement.get('timing_contract')
    if not isinstance(contract, dict) or not contract or contract != author.get('timing_contract'):
        reasons.append('included noise work and CPU return fields/domains are not proven to match')
    if record.get('author_quality', {}).get('status') != 'passed':
        reasons.append('author numerical quality budget did not pass')
    if (measurement['repeats'] != author['repeats']
            or record.get('workload_request_ids') != sorted(set(author['request_ids']))):
        reasons.append('request/sample coverage differs')
    internal_ms, author_ms = median(measurement['latency_ms']), median(author['latency_ms'])
    return {'origin': 'author', 'reference': reference_evidence(reference),
            'internal_median_ms': internal_ms, 'author_median_ms': author_ms,
            'strict_author_over_internal_ratio': None if reasons else author_ms / max(internal_ms, 1e-12),
            'incomparable_reasons': reasons,
            'author_quality_status': record.get('author_quality', {}).get('status', 'unknown'),
            'interpretation': 'Author API vs project full-request timing; internal E10 is a separate baseline. '
                              'Different NFE/precision or saved measurements are reported without a strict speedup ratio.'}
