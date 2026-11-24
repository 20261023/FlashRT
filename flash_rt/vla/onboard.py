"""CPU-only scaffolding and fail-closed evidence checks for VLA onboarding.

Model declarations belong to model_spec. This module adds no model registry,
does not import runtime/author adapters and never executes a model.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sqlite3
from statistics import median
import sys
import tempfile

import numpy as np

from .model_spec import get_spec, validate_spec


HEAD_TYPES = ('flow_matching', 'autoregressive_tokens', 'other')
BOUNDARY = 'prepared-host-observation-and-noise_to_completed-CPU-actions'


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key {key!r}')
            result[key] = value
        return result
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ValueError(f'nonfinite JSON value {value}')))
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot read {path}: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + '\n', encoding='utf-8')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root, value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty path')
    path = Path(value)
    return path if path.is_absolute() else root / path


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return float(value)


def _template(model, head_type):
    if head_type not in HEAD_TYPES:
        raise ValueError(f'head must be one of {HEAD_TYPES}')
    unknown = lambda fact: {
        'status': 'unknown', 'value': None,
        'sources': [f'onboarding template: independently establish {fact} before measurement']}
    flow = head_type == 'flow_matching'
    tokens = head_type == 'autoregressive_tokens'
    sampler = 'euler' if flow else 'autoregressive' if tokens else 'none'
    action_stage = 'denoise.velocity' if flow else 'action.decode' if tokens else 'action.regression'
    package = model.replace('-', '_').replace('.', '_') + '_onboarding'
    if not package.isidentifier():
        raise ValueError('A scaffold model id must produce a valid Python package identifier')
    keys = {'vision': ['vision.encode'], 'prefix': ['prefix.prefill'], 'action': [action_stage]}
    spec = {
        'schema_version': 1, 'model': model, 'name': f'Unverified template for {model}', 'aliases': [],
        'identity': {
            'source': {**unknown('author repository and exact commit'),
                       'repository': 'unknown-author-repository', 'revision': None},
            'weights': unknown('checkpoint revision and weight content'),
            'license': unknown('source and checkpoint licenses')},
        'inputs': {
            'image_0': {'shape': ['height', 'width', 3], 'dtype': 'uint8'},
            'image_1': {'shape': ['height', 'width', 3], 'dtype': 'uint8'},
            'state': {'shape': ['state_dim'], 'dtype': 'float32'},
            'noise': {'shape': ['horizon', 'model_dim'] if flow else [1], 'dtype': 'float32'}},
        'head': {'type': head_type, 'kind': 'velocity' if flow else 'tokens' if tokens else 'regression',
                 **({'vocab_size': 1024, 'max_tokens': 256} if tokens else {}),
                 'defaults': {
            'sampler': sampler, 'num_steps': 10 if flow else None}},
        'actions': {'model_dim': 32, 'effective_dim': 7, 'channels': list(range(7)),
                    'horizon': 50, 'domain': 'normalized-model-actions',
                    'worker_domain': 'normalized-model-actions', 'worker_field': 'raw_actions_if_available',
                    'notes': ['All template dimensions and domains are assumptions, not model facts.']},
        'stages': [{'stage_id': stage, 'seams': [], 'profiling_keys': [key], 'notes': []}
                   for key, stages in keys.items() for stage in stages],
        'profiling': {'keys': keys, 'expected_keys': {sampler: list(keys)}, 'strategy': {}},
        'attention': [],
        'capabilities': {
            'frameworks': ['torch'], 'hardware': ['cuda_sm80'], 'precision': 'bf16',
            'backends': {'torch': ['torch']}, 'samplers': [sampler],
            'options': {'use_cuda_graph': {'type': 'bool', 'default': True}},
            'action_domain_resolution': 'static',
            'execution_coverage': {'torch': 'Scaffold only; no execution has been measured.'}},
        'features': {'cuda_graph': unknown('graph capture and replay parity'),
                     'variable_steps': unknown('implemented Euler step budget selection') if flow else False},
        'tools': {'tags': [head_type], 'supported': [], 'unsupported': []},
        'adapter': {'hooks': {
            'binding': f'{package}.project_adapter:binding',
            'validate': f'{package}.project_adapter:validate'}},
        'author': {'adapter': f'{package}.author_adapter', 'source_package': 'replace_with_author_package',
                   'options': [], 'default_steps': 10 if flow else 1,
                   'fixed_steps': None if flow or tokens else 1,
                   'requires_noise_seeds': False},
        'fixture': {
            'recipe': 'onboarding_numpy_v1', 'fields': {
                'images': ['image_0', 'image_1'], 'state': 'state'},
            'noise_key': 'noise', 'seed': 0, 'cases': 4,
            'noise_role': 'fixed_initial_noise' if flow else 'unused',
            'dimensions': {'height': 224, 'width': 224, 'state_dim': 8,
                           'horizon': 50, 'model_dim': 32},
            'prompts': ['pick up the red object', 'move the blue object'],
            'sources': ['Synthetic CPU structural fixture; not a dataset or task-quality claim.']},
        'quality': {'scope': 'numerical_replay', 'action_dim': 7,
                    'thresholds': {'max_relative_l2': 0.05, 'min_cosine': 0.999},
                    'sources': ['Proposed onboarding budget; obtain confirmation before measurements.']},
    }
    return validate_spec(spec)


def _input_arrays(spec):
    dimensions = spec['fixture'].get('dimensions', {})
    result = {}
    for name, field in spec['inputs'].items():
        shape = []
        for dim in field['shape']:
            resolved = dimensions.get(dim) if isinstance(dim, str) else dim
            shape.append(_positive(resolved, f'inputs.{name} unresolved dimension {dim!r}'))
        try:
            dtype = np.dtype(field['dtype'])
        except (TypeError, ValueError) as error:
            raise ValueError(f'inputs.{name} invalid dtype: {error}') from error
        if dtype.kind not in 'buif' or dtype.hasobject:
            raise ValueError(f'inputs.{name} requires a real numeric/bool dtype, got {dtype}')
        result[name] = (tuple(shape), dtype)
    return result


def generate_fixtures(spec, root, *, spec_dir=None):
    """Generate varied deterministic NPZ requests from shared ModelSpec inputs.

    Synthetic data establish structural/numerical contracts only. Real datasets
    may instead supply an ordinary NPZWorkload manifest with their provenance.
    """
    spec = validate_spec(spec)
    if spec['head']['type'] not in HEAD_TYPES:
        raise ValueError(f'Fixture generation supports {HEAD_TYPES}')
    recipe = spec['fixture']
    if recipe['recipe'] != 'onboarding_numpy_v1':
        raise ValueError('This generator only implements fixture.recipe=onboarding_numpy_v1')
    arrays = _input_arrays(spec)
    cases = _positive(recipe.get('cases'), 'fixture.cases')
    if cases < 4:
        raise ValueError('Onboarding fixtures require at least four varied cases')
    seed = recipe.get('seed')
    if type(seed) is not int or seed < 0:
        raise ValueError('fixture.seed must be a nonnegative integer')
    fields = recipe['fields']
    names = []
    for field, source in fields.items():
        sources = source if isinstance(source, list) else [source]
        if not sources or any(not isinstance(name, str) or name not in arrays for name in sources):
            raise ValueError(f'fixture.fields.{field} must name declared input arrays')
        names.extend(sources)
    noise_key = recipe.get('noise_key')
    if not isinstance(noise_key, str) or noise_key not in arrays:
        raise ValueError('fixture.noise_key must name a declared input')
    if arrays[noise_key][1].kind != 'f':
        raise ValueError('fixture noise requires a floating dtype')
    prompts = recipe.get('prompts')
    if not isinstance(prompts, list) or not prompts or any(
            not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError('fixture.prompts must contain nonempty prompts')
    root = Path(root)
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite fixtures: {root}')
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    requests = []
    for index in range(cases):
        payload = {}
        for name in sorted(set(names) | {noise_key}):
            shape, dtype = arrays[name]
            if name == noise_key and spec['head']['type'] != 'flow_matching':
                value = np.zeros(shape, dtype=dtype)
            elif dtype.kind == 'f':
                value = rng.standard_normal(shape).astype(dtype)
            else:
                high = 2 if dtype.kind == 'b' else min(256, int(np.iinfo(dtype).max) + 1)
                value = rng.integers(0, high, size=shape).astype(dtype)
            payload[name] = value
        filename = f'case-{index:03d}.npz'
        np.savez_compressed(root / filename, **payload)
        requests.append({'request_id': f'case-{index:03d}', 'npz_path': filename,
                         'fields': fields, 'noise_key': noise_key,
                         'prompt': prompts[index % len(prompts)]})
    path = root / 'workload.json'
    _write_json(path, {'schema_version': 1, 'model': spec['model'],
                      **({'spec_dir': str(Path(spec_dir).resolve())} if spec_dir is not None else {}),
                      'workload_id': f'{spec["model"]}-onboarding-v1', 'requests': requests,
                      'provenance': {'kind': 'synthetic', 'seed': seed, 'sources': recipe['sources'],
                                     'noise_role': recipe.get('noise_role', 'fixed_initial_noise'),
                                     'limitations': 'CPU structural fixture, not real-data/task evidence'}})
    return path


_AUTHOR_TEMPLATE = '''"""Original-author adapter. Implement before running in a Slurm GPU job.

Load the pinned original author package and checkpoint through its own APIs.
Do not import a FlashRT neural implementation. The unified producer supplies
spec and torch, checks provenance and owns timing/quality evidence.
"""

ONBOARDING_IMPLEMENTED = False


def load(spec, torch):
    """Return (author_model, infer(request), metadata).

    infer consumes NPZWorkload Request, including prompt, observation and fixed
    noise, and returns CPU-compatible actions in the ModelSpec worker domain.
    metadata must contain implementation and expected_shape. Describe any
    additional work in the measurement boundary. Regression must ignore the
    inert noise placeholder and perform one direct action-head evaluation.
    Autoregressive decoding must report observed prefill+decoder forward counts
    and generated tokens; never use the placeholder author num_steps as NFE.
    """
    raise NotImplementedError("Implement the pinned original-author entrypoint")
'''


_PROJECT_ADAPTER_TEMPLATE = '''"""Explicit project binding. CPU discovery must not import this module."""

ONBOARDING_IMPLEMENTED = False


def validate(config):
    if not ONBOARDING_IMPLEMENTED:
        raise NotImplementedError("Implement and test the project backend before GPU loading")
    # Validate semantic shape/options interactions, without reading a device.


def binding(config, hardware):
    validate(config)
    # Split additional hardware implementations into their own backend modules.
    if hardware != "cuda_sm80":
        raise ValueError("Only the explicitly implemented SM80 path may be loaded")
    return (__package__ + ".project_backend", "OnboardingFrontend", {"config": config})
'''


_PROJECT_BACKEND_TEMPLATE = '''"""Project frontend skeleton; never an author-Native implementation.

Implement model math in a per-hardware pipeline and preserve owned static graph
buffers. The methods below intentionally refuse computation until implemented.
Autoregressive heads ignore inert host noise; use the checkpoint tokenizer and
action-token decoder, retain prefill/decode call counts and actual token counts.
"""

ONBOARDING_IMPLEMENTED = False


class OnboardingFrontend:
    def __init__(self, checkpoint, *, config):
        if not ONBOARDING_IMPLEMENTED:
            raise NotImplementedError("Implement loading, exact input and action-domain contracts")
        self.checkpoint = checkpoint
        self.config = config

    def infer(self, observation, noise=None):
        """Return CPU actions plus sampler_metadata and execution_metadata.

        Flow: sampler=euler, nfe=actual velocity calls, supplied fixed noise.
        Tokens: sampler=autoregressive, generated_tokens=actual token count,
        nfe=actual prefill_forward_calls+decoder_forward_calls. Retain these
        counts under execution_metadata with nfe_provenance=observed_forward_calls.
        Regression: sampler=none, nfe=1, no denoising/noise computation.
        Graph telemetry: use_cuda_graph, graph_replays, graph_capture_id.
        No NFE/replay fields may be fabricated from requested configuration.
        """
        raise NotImplementedError("Implement actual eager/graph execution and output decoding")

    def close(self):
        raise NotImplementedError("Release only this frontend's owned graph/model resources")
'''


def _contract_test_template(package):
    return f'''"""CPU scaffold contract; add model-specific tests before enabling implementation."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_scaffold_refuses_gpu_readiness_without_implementation():
    # No runtime, torch, weights or model imports during template tests.
    for name in ("project_adapter.py", "project_backend.py", "author_adapter.py"):
        tree = ast.parse((ROOT / "{package}" / name).read_text())
        marker = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "ONBOARDING_IMPLEMENTED"
                              for target in node.targets))
        assert isinstance(marker, ast.Constant) and marker.value is False


def test_fixed_fixture_manifest_is_complete():
    import json
    manifest = json.loads((ROOT / "fixtures/workload.json").read_text())
    assert len(manifest["requests"]) >= 4
    assert len({{request["request_id"] for request in manifest["requests"]}}) == len(manifest["requests"])
    assert all((ROOT / "fixtures" / request["npz_path"]).is_file() for request in manifest["requests"])
'''


def init_onboarding(root, *, model, head_type):
    """Create a fresh reviewable scaffold; no existing files are overwritten."""
    spec = _template(model, head_type)
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(f'Onboarding root already exists: {root}')
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.onboard-', dir=root.parent) as temporary:
        staged = Path(temporary) / 'root'
        (staged / 'specs').mkdir(parents=True)
        _write_json(staged / 'specs' / f'{model}.json', spec)
        generate_fixtures(spec, staged / 'fixtures', spec_dir=root / 'specs')
        package = spec['author']['adapter'].split('.')[0]
        (staged / package).mkdir()
        (staged / package / '__init__.py').write_text('"""Private onboarding adapters."""\n', encoding='utf-8')
        for name, source in (('author_adapter', _AUTHOR_TEMPLATE),
                             ('project_adapter', _PROJECT_ADAPTER_TEMPLATE),
                             ('project_backend', _PROJECT_BACKEND_TEMPLATE)):
            (staged / package / f'{name}.py').write_text(source, encoding='utf-8')
        (staged / 'tests').mkdir()
        (staged / 'tests' / f'test_{model.replace("-", "_").replace(".", "_")}_contract.py').write_text(
            _contract_test_template(package), encoding='utf-8')
        _write_json(staged / 'author_baseline.json', {
            'mode': 'produce', 'source': '/replace/with/clean/original/author/checkout',
            'source_revision': None, 'num_steps': spec['author']['default_steps'], 'options': {}})
        _write_json(staged / 'onboarding.json', {
            'schema_version': 1, 'model': model, 'spec_dir': 'specs',
            'workload': 'fixtures/workload.json', 'author_baseline': 'author_baseline.json',
            'fixture_kind': 'synthetic', 'quality_budget_confirmed': False,
            'template_assumptions_confirmed': False,
            'implementation_files': {name: f'{package}/{name}.py'
                                     for name in ('project_adapter', 'project_backend', 'author_adapter')},
            'assumptions': [
                'Two RGB cameras, 224x224 uint8; state[8]. Replace with actual model inputs.',
                'Horizon=50, model actions=32, effective channels=0..6 are template assumptions.',
                'Torch/BF16/SM80 are proposed targets; no execution capability has been established.',
                'Synthetic fixtures only validate fixed-input numerical/shape behavior.',
                'Non-flow heads keep inert fixed noise for the worker file envelope only.',
                'Token vocabulary and max token budget are assumptions; validate checkpoint token/action decoding.']})
        staged.rename(root)
    return {'status': 'initialized', 'root': str(root), 'model': model,
            'head': head_type, 'next': 'Fill pinned facts and adapters, then run check.'}


def _load_onboarding(root):
    root = Path(root).resolve()
    manifest = _read_json(root / 'onboarding.json')
    if manifest.get('schema_version') != 1:
        raise ValueError('onboarding schema_version must be 1')
    directory = manifest.get('spec_dir')
    spec_dir = _path(root, directory, 'spec_dir') if directory is not None else None
    spec = get_spec(manifest.get('model'), spec_dir=spec_dir)
    return root, manifest, spec


def _gate(spec):
    quality = spec['quality']
    thresholds = quality['thresholds']
    if (not isinstance(thresholds, dict) or thresholds.get('status') == 'unknown'
            or not {'max_relative_l2', 'min_cosine'} <= set(thresholds)):
        raise ValueError('quality.thresholds requires explicit max_relative_l2 and min_cosine budgets')
    channels = spec['actions']['channels']
    effective = spec['actions']['effective_dim']
    if (type(effective) is not int or not isinstance(channels, list) or len(channels) != effective
            or quality.get('action_dim') not in (None, effective)):
        raise ValueError('Quality gate requires explicit effective action channels consistent with ModelSpec')
    selection = {'action_dim': effective}
    if channels != list(range(effective)):
        selection['action_channels'] = channels
    return {'scope': quality['scope'], **selection, **thresholds}


def _check_workload(spec, workload):
    arrays = _input_arrays(spec)
    expected = spec['fixture']['fields']
    noise_key = spec['fixture'].get('noise_key')
    if workload.model != spec['model'] or len(workload) < 4:
        raise ValueError('workload must match the model and contain at least four distinct cases')
    if noise_key not in arrays:
        raise ValueError('fixture.noise_key must name a declared input')
    for index in range(len(workload)):
        item, request = workload.specs[index], workload.request(index)
        if item['fields'] != expected or item['noise_key'] != noise_key:
            raise ValueError('workload fields/noise differ from ModelSpec.fixture')
        for field, sources in expected.items():
            sources = sources if isinstance(sources, list) else [sources]
            values = request.observation[field]
            values = values if isinstance(values, list) else [values]
            if len(values) != len(sources):
                raise ValueError(f'workload field {field} has incomplete inputs')
            for name, value in zip(sources, values):
                shape, dtype = arrays[name]
                if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                    raise ValueError(f'workload input {name} has wrong shape/dtype or nonfinite values')
        shape, dtype = arrays[noise_key]
        if (request.noise.shape != shape or request.noise.dtype != dtype
                or not np.isfinite(request.noise).all()):
            raise ValueError('workload noise has wrong shape/dtype or nonfinite values')
        if spec['head']['type'] != 'flow_matching' and np.any(request.noise):
            raise ValueError('Non-flow fixture noise must be an inert zero placeholder')


def _implementation_issues(root, manifest):
    issues = []
    files = manifest.get('implementation_files')
    required = {'project_adapter', 'project_backend', 'author_adapter'}
    if not isinstance(files, dict) or not required <= set(files):
        return ['implementation_files must identify the project adapter/backend and author adapter source']
    for role in sorted(required):
        path = _path(root, files[role], f'implementation_files.{role}')
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (OSError, SyntaxError) as error:
            issues.append(f'{role} source unavailable or invalid: {error}')
            continue
        markers = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'ONBOARDING_IMPLEMENTED'
                           for target in node.targets)]
        if markers and not (len(markers) == 1 and isinstance(markers[0], ast.Constant)
                            and markers[0].value is True):
            issues.append(f'{role} is an unimplemented onboarding scaffold')
        if any(isinstance(node, ast.Raise) and isinstance(node.exc, (ast.Call, ast.Name))
               and (isinstance(node.exc, ast.Name) and node.exc.id == 'NotImplementedError'
                    or isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id == 'NotImplementedError') for node in ast.walk(tree)):
            issues.append(f'{role} contains unimplemented NotImplementedError paths')
    return issues


def check_onboarding(root):
    """Static readiness only; a ready result is not model/hardware verification."""
    issues, notes = [], []
    try:
        root, manifest, spec = _load_onboarding(root)
        for key, fact in spec['identity'].items():
            if isinstance(fact, dict) and fact.get('status') != 'pinned':
                issues.append(f'identity.{key} is not pinned to authoritative source evidence')
        digest = spec['identity']['weights'].get('sha256')
        if not isinstance(digest, str) or len(digest) != 64 or digest.strip('0123456789abcdef'):
            issues.append('identity.weights.sha256 must pin the actual checkpoint content identity')
        if spec['head']['type'] not in HEAD_TYPES:
            issues.append(f'head type must be one of {HEAD_TYPES}')
        issues.extend(_implementation_issues(root, manifest))
        if 'binding' not in spec['adapter']['hooks']:
            issues.append('adapter.hooks.binding must name an implemented runtime adapter')
        if spec['author']['adapter'] == 'author_adapter' or spec['author']['source_package'].startswith('replace_'):
            issues.append('author adapter/source_package still refer to the generated scaffold')
        for key in ('template_assumptions_confirmed', 'quality_budget_confirmed'):
            if manifest.get(key) is not True:
                issues.append(f'{key} must be explicitly confirmed before measurement')
        if not isinstance(manifest.get('assumptions'), list) or not manifest['assumptions']:
            issues.append('assumptions must document the intended input/domain/hardware limits')
        _gate(spec)
        from .autotune.workloads import NPZWorkload
        workload = NPZWorkload.from_file(_path(root, manifest.get('workload'), 'workload'),
                                        spec_dir=(_path(root, manifest['spec_dir'], 'spec_dir')
                                                  if manifest.get('spec_dir') is not None else None))
        _check_workload(spec, workload)
        author = _read_json(_path(root, manifest.get('author_baseline'), 'author_baseline'))
        source = spec['identity']['source']
        if (author.get('mode') not in ('produce', 'saved')
                or author.get('source_revision') is None
                or author.get('source_revision') != source.get('revision')
                or not isinstance(author.get('source'), str) or not Path(author['source']).is_absolute()):
            issues.append('author_baseline must bind an absolute original-author source and pinned revision')
        if manifest.get('fixture_kind') == 'synthetic':
            notes.append('Synthetic fixed-input numerical proxy; no dataset or closed-loop task-success claim.')
        elif (manifest.get('fixture_kind') != 'dataset' or not isinstance(manifest.get('fixture_sources'), list)
              or not manifest['fixture_sources'] or any(
                  not isinstance(source, str) or not source.strip() for source in manifest['fixture_sources'])):
            issues.append('Real data fixtures require fixture_kind=dataset and nonempty fixture_sources provenance')
        return {'status': 'incomplete' if issues else 'ready_for_measurement',
                'model': spec['model'], 'head': spec['head']['type'], 'issues': issues, 'notes': notes}
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append(str(error))
        return {'status': 'incomplete', 'issues': issues, 'notes': notes}


def _hardware(record):
    gpu = record.get('gpu', {})
    for key in ('name', 'uuid', 'host', 'job_id'):
        if not isinstance(gpu.get(key), str) or not gpu[key].strip() or gpu[key].lower() in (
                'unknown', 'none', 'unavailable'):
            raise ValueError(f'Real Slurm GPU evidence requires gpu.{key}')
    if gpu['job_id'] != record.get('job_id') or not gpu['job_id'].split('_')[0].isdigit():
        raise ValueError('GPU and worker Slurm job IDs differ or are invalid')
    if gpu['host'].lower().startswith(('ln', 'login')):
        raise ValueError('Model measurements on a login host cannot qualify onboarding')
    capability = gpu.get('capability')
    if (not isinstance(capability, list) or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)
            or capability[0] == 0):
        raise ValueError('Actual GPU compute capability is required')
    hardware = record['candidate']['backend']['hardware']
    if hardware == 'auto':
        raise ValueError('Onboarding requires an explicit hardware target')
    if hardware.startswith('cuda_sm') and hardware != f'cuda_sm{capability[0]}{capability[1]}':
        raise ValueError('Requested hardware differs from actual GPU compute capability')
    runtime = record.get('runtime', {})
    required = ('python', 'cuda', record['candidate']['backend']['framework'])
    if any(not isinstance(runtime.get(key), str) or not runtime[key] for key in required):
        raise ValueError('Actual Python/framework/CUDA versions must accompany hardware evidence')
    return {key: gpu[key] for key in ('name', 'uuid', 'host', 'job_id', 'capability')}


def _outputs(path, record, spec, expected_ids):
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['request_ids'].tolist()
        if (archive['output_domain'].item() != spec['actions']['worker_domain']
                or archive['workload_digest'].item() != record['workload_identity']['sha256']):
            raise ValueError('Worker NPZ has a different output domain or workload digest')
        actions = [archive[f'actions_{index:05d}'].copy() for index in range(len(ids))]
    counts = Counter(ids)
    if set(counts) != set(expected_ids) or min(counts.values()) < 2:
        raise ValueError('Every workload case requires at least two internal output samples per attempt')
    if len(ids) != record['measurement']['repeats']:
        raise ValueError('Output sample count differs from the authoritative measurement')
    channels = spec['actions']['channels']
    horizon = spec['actions']['horizon']
    for action in actions:
        if (action.ndim < 2 or action.shape[-2] != horizon
                or action.shape[-1] <= max(channels)
                or action.dtype != np.float32 or not np.isfinite(action).all()):
            raise ValueError('Output actions violate the declared horizon/channels/finite FP32 contract')
    return actions, ids, min(counts.values())


def _profile(record, spec, result_path, explanation):
    metrics = record.get('stage_metrics')
    if (not isinstance(metrics, dict) or 'profiling_failure' in record
            or metrics.get('measurement_kind') != 'diagnostic_cuda_event_intervals'):
        raise ValueError('Onboarding requires successful measured stage profile evidence')
    wall = _number(metrics.get('wall_ms'), 'profile.wall_ms', positive=True)
    intervals = metrics.get('interval_ms')
    mapping = spec['profiling']['keys']
    provenance = metrics.get('interval_provenance')
    sums = metrics.get('interval_sum_ms')
    if (not isinstance(intervals, dict) or not intervals
            or set(intervals) - set(mapping) or not isinstance(provenance, dict)
            or not isinstance(sums, dict) or set(sums) != set(intervals)):
        raise ValueError('Profile needs measured intervals and declared stage provenance')
    totals = {}
    for key, values in intervals.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f'Profile interval {key} has no samples')
        detail = provenance.get(key, {})
        if (not isinstance(detail, dict) or detail.get('additive') is not True
                or detail.get('stages') != mapping[key]):
            raise ValueError('Profile intervals must be additive and match ModelSpec stage ownership')
        totals[key] = math.fsum(_number(value, f'profile.{key}') for value in values)
        stored = sums[key]
        if stored is None or not math.isclose(
                totals[key], _number(stored, f'profile.interval_sum_ms.{key}'),
                rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError('Profile interval sums differ from the measured samples')
    covered = math.fsum(totals.values())
    if covered > wall * (1 + 1e-6):
        raise ValueError('Profile intervals exceed wall time; resolve overlapping intervals before onboarding')
    if metrics.get('diagnostic_graph_recapture') and metrics.get('capture_parity', {}).get('status') != 'bit_exact':
        raise ValueError('Diagnostic graph recapture requires measured bit-exact output parity')
    report = {'coverage_ratio': covered / wall, 'wall_ms': wall,
              'covered_ms': covered, 'unattributed_ms': wall - covered,
              'status': 'covered', 'intervals_ms': totals}
    if covered / wall >= 0.90:
        return report
    if explanation is None:
        raise ValueError('Profile covers less than 90%; supply a bound measured residual explanation')
    path = Path(explanation).resolve()
    extra = _read_json(path)
    binding = {
        'campaign_instance_id': record['campaign_instance_id'], 'attempt_id': record['attempt_id'],
        'job_id': record['job_id'], 'gpu_uuid': record['gpu']['uuid'],
        'workload_digest': record['workload_identity']['sha256'], 'result_sha256': _sha256(result_path)}
    if (extra.get('schema_version') != 1 or extra.get('kind') != 'onboarding_residual_explanation'
            or extra.get('binding') != binding):
        raise ValueError('Residual explanation does not bind this profile/campaign/job/GPU/workload')
    for name in ('reason', 'limitations'):
        if not isinstance(extra.get(name), str) or not extra[name].strip():
            raise ValueError(f'Measured residual explanation requires {name}')
    for name in ('wall_ms', 'covered_ms', 'unattributed_ms'):
        if not math.isclose(_number(extra.get(name), f'explanation.{name}'), report[name],
                            rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError('Residual explanation numbers do not reproduce the actual profile')
    observations = extra.get('observations')
    if not isinstance(observations, list) or not observations:
        raise ValueError('Residual explanation requires repeated quantitative observations')
    observed, seen = [], set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError('Residual observation must be an object')
        artifact = observation.get('artifact', {})
        artifact_path = _path(path.parent, artifact.get('path'), 'residual artifact').resolve()
        if artifact_path in seen or not artifact_path.is_file() or artifact_path.stat().st_size == 0:
            raise ValueError('Residual observations require distinct nonempty supporting artifacts')
        seen.add(artifact_path)
        if _sha256(artifact_path) != artifact.get('sha256'):
            raise ValueError('Residual measurement artifact digest differs')
        samples = observation.get('samples_ms')
        if (not isinstance(samples, list) or len(samples) < 2
                or observation.get('clock') not in ('host_wall', 'cuda_event', 'nsys')
                or not isinstance(observation.get('stage'), str) or not observation['stage']):
            raise ValueError('Residual observations require stage/clock and at least two measured samples')
        observed.append(median(_number(value, 'residual sample') for value in samples))
    explained = math.fsum(observed)
    if abs(explained - report['unattributed_ms']) > 0.10 * wall:
        raise ValueError('Repeated residual observations do not explain the measured residual within 10% wall time')
    return {**report, 'status': 'measured_residual_explained', 'explanation_path': str(path),
            'explanation_sha256': _sha256(path), 'reason': extra['reason'],
            'limitations': extra['limitations'], 'explained_ms': explained}


def _graph_evidence(record, *, graph):
    options = record['candidate']['backend']['options']
    if options.get('use_cuda_graph') is not graph:
        raise ValueError('Graph/eager records must explicitly select their execution modes')
    samples = record.get('execution_metadata')
    if not isinstance(samples, list) or len(samples) != record['measurement']['repeats']:
        raise ValueError('Graph/eager execution requires actual per-request execution telemetry')
    for sample in samples:
        if (not isinstance(sample, dict) or sample.get('use_cuda_graph') is not graph
                or type(sample.get('graph_replays')) is not int
                or (sample['graph_replays'] < 1 if graph else sample['graph_replays'] != 0)
                or (graph and (not isinstance(sample.get('graph_capture_id'), str)
                               or not sample['graph_capture_id']))):
            raise ValueError('Graph/eager telemetry does not prove the requested graph replay behavior')


def _sampler_evidence(record, spec):
    """NFE is observed neural calls, never a denoising config on token heads."""
    backend = record['candidate']['backend']
    samples = record['measurement']['sampler_metadata']
    repeats = record['measurement']['repeats']
    if not isinstance(samples, list) or len(samples) != repeats:
        raise ValueError('Actual sampler metadata is required for every output sample')
    for sample in samples:
        if (not isinstance(sample, dict) or sample.get('sampler') != backend['sampler']
                or type(sample.get('nfe')) is not int or sample['nfe'] < 1):
            raise ValueError('Every sampler sample requires the actual positive neural forward count')
    if spec['head']['type'] == 'autoregressive_tokens':
        if (backend['sampler'] != 'autoregressive' or backend.get('num_steps') is not None
                or backend.get('probeflow') is not None):
            raise ValueError('Autoregressive heads cannot use flow sampler/denoising steps')
        executions = record.get('execution_metadata')
        if not isinstance(executions, list) or len(executions) != repeats:
            raise ValueError('Autoregressive decoder NFE requires observed per-request execution counts')
        for sample, execution in zip(samples, executions):
            if not isinstance(execution, dict):
                raise ValueError('Autoregressive decoder execution telemetry must be an object')
            counts = [execution.get('prefill_forward_calls'), execution.get('decoder_forward_calls')]
            tokens = execution.get('generated_tokens')
            if (any(type(count) is not int or count < 0 for count in counts)
                    or execution.get('nfe_provenance') != 'observed_forward_calls'
                    or sum(counts) != sample['nfe']
                    or type(tokens) is not int or not 1 <= tokens <= spec['head']['max_tokens']
                    or sample.get('generated_tokens') != tokens):
                raise ValueError('Autoregressive NFE/token metadata must reproduce actual prefill+decoder forward counts')
    elif spec['head'].get('kind') == 'regression':
        if backend['sampler'] != 'none' or any(sample['nfe'] != 1 for sample in samples):
            raise ValueError('Regression requires one direct action-head evaluation and no flow sampler')


def _verify_evidence(root, *, campaign, results, graph_result, eager_result,
                     residual_explanation=None):
    readiness = check_onboarding(root)
    if readiness['status'] != 'ready_for_measurement':
        raise ValueError('Static onboarding checks failed: ' + '; '.join(readiness['issues']))
    root, manifest, spec = _load_onboarding(root)
    campaign, results = Path(campaign).resolve(), Path(results).resolve()
    graph_path, eager_path = Path(graph_result).resolve(), Path(eager_result).resolve()
    plan = _read_json(campaign / 'plan.json')
    gate = _gate(spec)
    if (plan.get('schema_version') not in (2, 3) or not (campaign / 'ledger.sqlite').is_file()
            or plan.get('author_baseline') is None):
        raise ValueError('An authoritative bound campaign ledger and author baseline are required')
    if plan.get('quality_gate') != gate:
        raise ValueError('Campaign quality budget/channels differ from the confirmed ModelSpec')
    if plan['author_baseline'].get('declaration') != spec:
        raise ValueError('Campaign must retain the exact onboarding ModelSpec declaration at initialization')
    entries = plan.get('entries')
    if (not isinstance(entries, list) or len(entries) < 3
            or entries[0].get('role') != 'reference' or entries[-1].get('role') != 'bookend'
            or entries[0].get('candidate') != entries[-1].get('candidate')
            or len({entry['attempt_id'] for entry in entries}) != len(entries)):
        raise ValueError('Campaign must bind a repeated internal reference/bookend and graph/eager attempts')
    from .autotune.author import load_portable_reference, validate_portable_attempt, workload_identity
    from .autotune.controller import _validate_checkout_candidate
    from .autotune.ledger import CampaignLedger
    from .autotune.quality import compare_outputs
    from .autotune.schema import Candidate
    from .autotune.workloads import NPZWorkload
    spec_dir = _path(root, manifest['spec_dir'], 'spec_dir') if manifest.get('spec_dir') is not None else None
    workload = NPZWorkload.from_file(_path(root, manifest['workload'], 'workload'), spec_dir=spec_dir)
    expected_workload = workload_identity(workload)
    reference = load_portable_reference(plan, results)
    source = reference['record']['identity']['author_source']
    if (source['revision'] != spec['identity']['source']['revision']
            or source['source_url'] != spec['identity']['source']['repository']):
        raise ValueError('Author source evidence differs from the declared original-author pin')
    pinned_weights = spec['identity']['weights'].get('sha256')
    if not pinned_weights or pinned_weights != reference['record']['identity']['checkpoint']['sha256']:
        raise ValueError('ModelSpec must pin the actual author/worker checkpoint content sha256')
    if reference['record']['identity']['workload'] != expected_workload:
        raise ValueError('Measured author/worker fixtures differ from the current onboarding workload')
    author_config = _read_json(_path(root, manifest['author_baseline'], 'author_baseline'))
    for key in ('mode', 'result_path'):
        if key in author_config and author_config[key] != plan['author_baseline'].get(key):
            raise ValueError('Author baseline mode/result path differs from onboarding configuration')
    if any(author_config.get(key) != plan['author_baseline']['spec'].get(key)
           for key in ('source', 'source_revision', 'num_steps', 'options')):
        raise ValueError('Campaign author entry differs from onboarding configuration')
    records, payloads, devices, repeats_per_case = {}, {}, {}, []
    class ReadOnlyLedger(CampaignLedger):
        # Reuse every authoritative identity query/validator while preventing
        # the ordinary controller constructor from creating/migrating evidence.
        def __init__(self, path):
            self._db = sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True)
            self._db.row_factory = sqlite3.Row
            self._db.execute('BEGIN')  # One consistent authoritative read snapshot.

    with ReadOnlyLedger(campaign / 'ledger.sqlite') as ledger:
        if plan.get('campaign_instance_id') != ledger.instance_id:
            raise ValueError('Campaign instance differs from the authoritative ledger')
        attempts = {attempt['attempt_id']: attempt for attempt in ledger.attempts()}
        if set(attempts) != {entry['attempt_id'] for entry in entries}:
            raise ValueError('Plan attempt set differs from the authoritative ledger')
        if plan['schema_version'] == 3:
            snapshots = {key: {field: value for field, value in item.items() if field != 'manifest_path'}
                         for key, item in plan.get('snapshots', {}).items()}
            if snapshots != ledger.snapshots():
                raise ValueError('Plan snapshots differ from the authoritative ledger')
        baseline = Candidate.from_dict(entries[0]['candidate'])
        for entry in entries:
            attempt = attempts[entry['attempt_id']]
            if attempt['state'] not in ('validated', 'rejected', 'failed'):
                raise ValueError('Campaign must be fully ingested before onboarding verification')
            path = results / entry['attempt_id'] / 'result.json'
            record = _read_json(path)
            ledger.validate_execution_identity(record)
            if record != attempt['result'] or record['candidate'] != entry['candidate']:
                raise ValueError('Fetched worker result/configuration differs from the authoritative ledger')
            candidate = Candidate.from_dict(entry['candidate'])
            if get_spec(candidate.backend.model, spec_dir=candidate.backend.spec_dir) != spec:
                raise ValueError('Executed candidate ModelSpec differs from the onboarding declaration')
            if plan['schema_version'] == 2:
                _validate_checkout_candidate(candidate, baseline)
            elif entry.get('snapshot_id') != record.get('source_identity', {}).get('snapshot_id'):
                raise ValueError('Plan/result snapshot identity differs')
            validate_portable_attempt(plan, entry, record, path, reference)
            if attempt['state'] != 'validated':
                if path in (graph_path, eager_path) or entry['role'] in ('reference', 'bookend'):
                    raise ValueError('Required graph/eager/reference/bookend attempt did not pass both quality gates')
                continue
            if (record['state'] != 'measured' or record['candidate']['backend']['model'] != spec['model']
                    or record.get('workload_identity') != expected_workload
                    or record.get('output_domain') != spec['actions']['worker_domain']
                    or record['quality']['status'] != 'passed' or record['author_quality']['status'] != 'passed'
                    or record['measurement'].get('boundary') != BOUNDARY):
                raise ValueError('Validated worker evidence is not the declared model/workload/domain/boundary')
            records[path] = record
            devices[path] = _hardware(record)
            _sampler_evidence(record, spec)
            payloads[path] = _outputs(path.parent / 'outputs.npz', record, spec, [
                request['request_id'] for request in workload.specs])
            repeats_per_case.append(payloads[path][2])
    if graph_path == eager_path or graph_path not in records or eager_path not in records:
        raise ValueError('Graph and eager evidence must be distinct validated attempts in this campaign')
    graph, eager = records[graph_path], records[eager_path]
    _graph_evidence(graph, graph=True)
    _graph_evidence(eager, graph=False)
    left, right = json.loads(json.dumps(graph['candidate'])), json.loads(json.dumps(eager['candidate']))
    for candidate in (left, right):
        for key in ('candidate_id', 'parent_id', 'hypothesis', 'origin', 'tool', 'stage'):
            candidate.pop(key, None)
        candidate['backend']['options'].pop('use_cuda_graph', None)
    if left != right:
        raise ValueError('Graph/eager comparison must use identical model/source/precision/sampling except use_cuda_graph')
    if any(device != devices[graph_path] for device in devices.values()):
        raise ValueError('Internal repeated measurements must use the same actual Slurm GPU')
    first_path = results / entries[0]['attempt_id'] / 'result.json'
    last_path = results / entries[-1]['attempt_id'] / 'result.json'
    base = median(records[first_path]['measurement']['latency_ms'])
    drift = median(records[last_path]['measurement']['latency_ms']) / max(base, 1e-12) - 1
    if abs(drift) > 0.10:
        raise ValueError('Internal reference/bookend drift exceeds 10%')
    graph_actions, graph_ids, _ = payloads[graph_path]
    eager_actions, eager_ids, _ = payloads[eager_path]
    parity = compare_outputs(graph_actions, graph_ids, eager_actions, eager_ids, gate,
                             domain=spec['actions']['worker_domain'])
    if parity['status'] != 'passed':
        raise ValueError('Graph outputs failed the confirmed numerical budget against eager outputs')
    profile = _profile(graph, spec, graph_path, residual_explanation)
    return {
        'schema_version': 1, 'status': 'evidence_verified', 'model': spec['model'],
        'head': spec['head'], 'campaign_instance_id': plan['campaign_instance_id'],
        'comparison_id': plan['comparison_id'], 'gpu': devices[graph_path],
        'internal_repeats_per_case': min(repeats_per_case), 'baseline_drift_ratio': drift,
        'profile': profile, 'graph_eager_quality': parity,
        'author_provenance': reference['descriptor']['provenance'],
        'assumptions': manifest['assumptions'], 'notes': readiness['notes'],
        'evidence_refs': [str(campaign / 'plan.json'), str(campaign / 'ledger.sqlite'),
                          str(results / 'control/author_portable.json'), str(graph_path), str(eager_path)],
        'interpretation': 'Fixed-input numerical and execution onboarding only; no task-success or speedup claim.'}


def verify_onboarding(root, *, campaign, results, graph_result, eager_result,
                      decisions, segments, session_id, actor='agent', residual_explanation=None):
    """Revalidate authoritative CPU evidence, append a note, close onboarding.

    A durable successful receipt is published only after H's segment close.
    Hardware experiments and authoritative controller ingest happen beforehand.
    """
    from .autotune.decisions import DecisionLog, build_fields
    from .autotune.token_usage import SegmentLog, segment_intervals
    root = Path(root).resolve()
    if (root / 'verification.json').exists():
        raise ValueError('Onboarding already has a verification receipt; preserve it and use a new onboarding root')
    try:
        report = _verify_evidence(
            root, campaign=campaign, results=results, graph_result=graph_result, eager_result=eager_result,
            residual_explanation=residual_explanation)
        log = SegmentLog(segments)
        spans = segment_intervals(log.records())
        opened = [span for span in spans if span['session_id'] == session_id and not span['closed']]
        if len(opened) != 1 or opened[0]['model'] != report['model'] or opened[0]['phase'] != 'onboarding':
            raise ValueError('Verification must close this session/model open onboarding segment')
        required = {'expected_model', 'expected_phase', 'expected_segment_id'}
        if not required <= set(inspect.signature(log.end).parameters):
            raise ValueError('H SegmentLog.end must support atomic expected model/phase/segment checks')
        fields = dict(model=report['model'], campaign=report['comparison_id'],
                      evidence_refs=report['evidence_refs'],
                      text=json.dumps(report, ensure_ascii=False, allow_nan=False))
        summary = f'{report["model"]} onboarding evidence verified; completion is recorded by verification.json'
        build_fields('note', actor=actor, summary=summary, **fields)
        DecisionLog(decisions).records()  # Refuse malformed logs before any completion append.
        note = DecisionLog(decisions).append('note', actor=actor, summary=summary, **fields)
        end = log.end(session_id, summary=f'{report["model"]} onboarding verified with author and internal gates',
                      expected_model=report['model'], expected_phase='onboarding',
                      expected_segment_id=opened[0]['segment_id'])
        report.update(status='verified', verified_at=datetime.now(timezone.utc).isoformat(),
                      decision_record_id=note['record_id'], segment_id=end['segment_id'])
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root,
                                         prefix='.verification-', suffix='.json', delete=False) as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        # Hard-link publication is atomic and cannot overwrite a receipt created
        # by a competing verifier after our preflight. Both paths share a disk.
        os.link(temporary, root / 'verification.json')
        temporary.unlink()
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return report
    except (OSError, KeyError, TypeError, sqlite3.Error) as error:
        raise ValueError(f'Onboarding evidence/completion failed: {error}') from error


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m flash_rt.vla.onboard', description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('init', help='Create fresh ModelSpec/fixture/author scaffolds, CPU only')
    create.add_argument('--root', required=True)
    create.add_argument('--model', required=True)
    create.add_argument('--head', choices=HEAD_TYPES, required=True)
    check = commands.add_parser('check', help='Check static readiness without loading any model')
    check.add_argument('--root', required=True)
    verify = commands.add_parser('verify', help='Revalidate authoritative campaign/author/graph evidence and close onboarding')
    for name in ('root', 'campaign', 'results', 'graph-result', 'eager-result', 'decisions', 'segments'):
        verify.add_argument('--' + name, required=True)
    verify.add_argument('--session-id', default=os.environ.get('OPENCODE_SESSION_ID'))
    verify.add_argument('--actor', choices=('human', 'agent', 'human+agent', 'search'), default='agent')
    verify.add_argument('--residual-explanation', help='Bound measured diagnosis when profile coverage is below 90%%')
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            output = init_onboarding(args.root, model=args.model, head_type=args.head)
        elif args.command == 'check':
            output = check_onboarding(args.root)
        else:
            if not args.session_id:
                raise ValueError('verify requires --session-id or OPENCODE_SESSION_ID')
            output = verify_onboarding(
                args.root, campaign=args.campaign, results=args.results, graph_result=args.graph_result,
                eager_result=args.eager_result, decisions=args.decisions, segments=args.segments,
                session_id=args.session_id, actor=args.actor, residual_explanation=args.residual_explanation)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'error', 'message': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    return 1 if output['status'] == 'incomplete' else 0


if __name__ == '__main__':
    raise SystemExit(main())
