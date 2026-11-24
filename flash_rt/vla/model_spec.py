"""Validated, declarative model metadata. Standard library only.

Discovery never imports adapters, reads checkpoints or probes a device. An
explicit ``spec_dir`` adds local JSON declarations; duplicates fail instead of
silently replacing builtins. Returned dictionaries are detached copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import json
import math
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
import re


SCHEMA_VERSION = 1
HEAD_TYPES = ('flow_matching', 'diffusion', 'autoregressive_tokens', 'other')
BUILTIN_SPEC_DIR = Path(__file__).with_name('specs')
_IDENTIFIER = re.compile(r'[a-z0-9][a-z0-9_.-]*')
_MODULE = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*')
_HOOK = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*')
_REQUIRED = {
    'schema_version', 'model', 'name', 'aliases', 'identity', 'inputs', 'head',
    'actions', 'stages', 'profiling', 'attention', 'capabilities', 'features',
    'tools', 'adapter', 'author', 'fixture', 'quality',
}


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _texts(value, name, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f'{name} must be a {"nonempty " if nonempty else ""}string list')
    for item in value:
        _text(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f'{name} contains duplicate values')
    return value


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _unknown(value, name):
    if not isinstance(value, dict) or value.get('status') != 'unknown' or value.get('value', 0) is not None:
        raise ValueError(f'{name} must be a value or an explicit unknown with value=null and sources')
    _texts(value.get('sources'), f'{name}.sources', nonempty=True)
    return True


def _dimension(value, name):
    if isinstance(value, dict):
        _unknown(value, name)
    else:
        _positive(value, name)


def _shape(value, name):
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a dimension list')
    for dim in value:
        if isinstance(dim, str):
            _text(dim, name)
        else:
            _dimension(dim, name)


def _finite_json(value, name='spec'):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f'{name} object keys must be strings')
        return {key: _finite_json(item, f'{name}.{key}') for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item, name) for item in value]
    raise ValueError(f'{name} must contain only finite JSON values')


def _identity(value, name):
    value = _object(value, name)
    status = value.get('status')
    _texts(value.get('sources'), f'{name}.sources', nonempty=True)
    if status == 'unknown':
        _unknown(value, name)
    elif status == 'pinned':
        _text(value.get('value'), f'{name}.value')
    else:
        raise ValueError(f'{name}.status must be pinned or unknown')
    revision = value.get('revision')
    if revision is not None:
        if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
            raise ValueError(f'{name}.revision must be a full lowercase Git commit or null')
        if value['value'] != revision:
            raise ValueError(f'{name}.revision differs from its pinned value')
    if name == 'identity.source':
        _text(value.get('repository'), f'{name}.repository')
        if status == 'pinned' and revision is None:
            raise ValueError(f'{name} requires a pinned revision')
    if 'sha256' in value and (not isinstance(value['sha256'], str)
                              or not re.fullmatch('[0-9a-f]{64}', value['sha256'])):
        raise ValueError(f'{name}.sha256 must be a lowercase SHA256')


def _seam(value):
    if not isinstance(value, str) or value.count(':') != 1:
        raise ValueError(f'seam must be relative/path.py:function or :Class.method, got {value!r}')
    path, symbol = value.split(':')
    parts = PurePosixPath(path).parts
    if (not path.endswith('.py') or path.startswith('/') or '\\' in path
            or '..' in parts or not 1 <= len(symbol.split('.')) <= 2
            or not all(part.isidentifier() for part in symbol.split('.'))):
        raise ValueError(f'Invalid seam {value!r}')


def _option_value(value, schema, name):
    kind = schema.get('type')
    valid = {
        'bool': lambda x: isinstance(x, bool),
        'integer': lambda x: isinstance(x, Integral) and not isinstance(x, bool),
        'number': lambda x: isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x),
        'string': lambda x: isinstance(x, str),
        'path': lambda x: isinstance(x, str) and bool(x.strip()),
        'object': lambda x: isinstance(x, Mapping),
        'array': lambda x: isinstance(x, (list, tuple)),
        'any': lambda x: True,
    }
    if kind not in valid:
        raise ValueError(f'{name} has unsupported option type {kind!r}')
    reason = None
    if not valid[kind](value):
        reason = f'must be {kind}'
    elif 'enum' in schema and value not in schema['enum']:
        reason = f'must be one of {schema["enum"]}'
    elif 'minimum' in schema and value < schema['minimum']:
        reason = f'must be >= {schema["minimum"]}'
    elif 'exclusive_minimum' in schema and value <= schema['exclusive_minimum']:
        reason = f'must be > {schema["exclusive_minimum"]}'
    if reason:
        error = TypeError if schema.get('error_type') == 'TypeError' else ValueError
        raise error(f'{name} {reason}')


def validate_options(spec, options):
    """Validate declared option types/ranges; adapters own semantic interactions."""
    if not isinstance(options, Mapping):
        raise TypeError('options must be a mapping')
    schemas = spec['capabilities']['options']
    unknown = set(options) - set(schemas)
    if unknown:
        raise ValueError(f'Unsupported {spec["model"]} options: {sorted(unknown)}')
    for name, value in options.items():
        _option_value(value, schemas[name], name)


def validate_spec(mapping, *, source='<memory>'):
    """Return a detached valid v1 declaration or raise a field-specific ValueError.

    The validator checks references within a declaration, not the existence of
    upstream assets or runtime hooks. Unknown facts require provenance, and
    cannot be mistaken for a pinned revision, numeric dimension or measurement.
    """
    spec = _finite_json(mapping)
    _object(spec, 'spec')
    missing = _REQUIRED - set(spec)
    if missing:
        raise ValueError(f'{source}: missing model spec fields {sorted(missing)}')
    if type(spec['schema_version']) is not int or spec['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'{source}: schema_version must be {SCHEMA_VERSION}')
    if not isinstance(spec['model'], str) or not _IDENTIFIER.fullmatch(spec['model']):
        raise ValueError('model must be a lowercase model identifier')
    _text(spec['name'], 'name')
    aliases = _texts(spec['aliases'], 'aliases')
    if any(alias != alias.lower() for alias in aliases):
        raise ValueError('aliases must be lowercase')
    identity = _object(spec['identity'], 'identity')
    for key in ('source', 'weights', 'license'):
        _identity(identity.get(key), f'identity.{key}')

    inputs = _object(spec['inputs'], 'inputs')
    if not inputs:
        raise ValueError('inputs must declare at least one field')
    for name, field in inputs.items():
        _object(field, f'inputs.{name}')
        _shape(field.get('shape'), f'inputs.{name}.shape')
        _text(field.get('dtype'), f'inputs.{name}.dtype')

    head = _object(spec['head'], 'head')
    if head.get('type') not in HEAD_TYPES:
        raise ValueError(f'head.type must be one of {HEAD_TYPES}')
    defaults = _object(head.get('defaults'), 'head.defaults')
    _text(defaults.get('sampler'), 'head.defaults.sampler')
    if head['type'] == 'autoregressive_tokens':
        for key in ('vocab_size', 'max_tokens'):
            _dimension(head.get(key), f'head.{key}')
    if head['type'] not in ('flow_matching', 'diffusion') and defaults.get('num_steps') is not None:
        raise ValueError('head.defaults.num_steps only applies to flow_matching/diffusion heads')
    actions = _object(spec['actions'], 'actions')
    for key in ('model_dim', 'effective_dim', 'horizon'):
        _dimension(actions.get(key), f'actions.{key}')
    model_dim, effective = actions['model_dim'], actions['effective_dim']
    if type(model_dim) is int and type(effective) is int and effective > model_dim:
        raise ValueError('actions.effective_dim cannot exceed model_dim')
    channels = actions.get('channels')
    if isinstance(channels, dict):
        _unknown(channels, 'actions.channels')
    elif not isinstance(channels, list) or not channels or any(
            type(channel) is not int or channel < 0 for channel in channels):
        raise ValueError('actions.channels must be nonnegative integer indices or an explicit unknown')
    elif len(set(channels)) != len(channels) or (
            type(model_dim) is int and max(channels) >= model_dim) or (
            type(effective) is int and len(channels) != effective):
        raise ValueError('actions.channels must be unique, in range, and match effective_dim')
    for key in ('domain', 'worker_domain'):
        _text(actions.get(key), f'actions.{key}')
    if actions.get('worker_field', 'raw_actions_if_available') not in ('actions', 'raw_actions_if_available'):
        raise ValueError('actions.worker_field must be actions or raw_actions_if_available')
    if 'raw_domain' in actions:
        _text(actions['raw_domain'], 'actions.raw_domain')

    stages = spec['stages']
    if not isinstance(stages, list) or not stages:
        raise ValueError('stages must be a nonempty list')
    stage_ids = []
    for item in stages:
        _object(item, 'stage')
        stage_id = _text(item.get('stage_id'), 'stage.stage_id')
        if not _IDENTIFIER.fullmatch(stage_id) or stage_id in stage_ids:
            raise ValueError(f'Invalid/duplicate stage id {stage_id!r}')
        stage_ids.append(stage_id)
        for seam in _texts(item.get('seams'), f'{stage_id}.seams'):
            _seam(seam)
        _texts(item.get('profiling_keys'), f'{stage_id}.profiling_keys')
        _texts(item.get('notes'), f'{stage_id}.notes')
    profiling = _object(spec['profiling'], 'profiling')
    keys = _object(profiling.get('keys'), 'profiling.keys')
    for key, covered in keys.items():
        _text(key, 'profiling key')
        _texts(covered, f'profiling.keys.{key}', nonempty=True)
        if set(covered) - set(stage_ids):
            raise ValueError(f'profiling key {key!r} refers to unknown stages')
    for item in stages:
        for key in item['profiling_keys']:
            if item['stage_id'] not in keys.get(key, []):
                raise ValueError(f'{item["stage_id"]} profiling key {key!r} does not cover the stage')
        missing = [key for key, covered in keys.items()
                   if item['stage_id'] in covered and key not in item['profiling_keys']]
        if missing:
            raise ValueError(f'{item["stage_id"]} profiling_keys omits covering intervals {missing}')
    expected = _object(profiling.get('expected_keys'), 'profiling.expected_keys')
    for sampler, intervals in expected.items():
        _texts(intervals, f'profiling.expected_keys.{sampler}')
        if set(intervals) - set(keys):
            raise ValueError(f'profiling.expected_keys.{sampler} refers to unknown keys')
    _object(profiling.get('strategy'), 'profiling.strategy')

    if not isinstance(spec['attention'], list):
        raise ValueError('attention must be a list')
    sites = set()
    for site in spec['attention']:
        _object(site, 'attention site')
        site_id = _text(site.get('id'), 'attention.id')
        if site_id in sites:
            raise ValueError(f'duplicate attention id {site_id!r}')
        sites.add(site_id)
        if site.get('stage') not in stage_ids:
            raise ValueError(f'attention {site_id} refers to an unknown stage')
        for key in ('head_dim', 'query_heads', 'kv_heads'):
            _dimension(site.get(key), f'attention.{site_id}.{key}')
        for key in ('mask', 'backend'):
            if isinstance(site.get(key), dict):
                _unknown(site[key], f'attention.{site_id}.{key}')
            else:
                _text(site.get(key), f'attention.{site_id}.{key}')
        qh, kh, gqa = site['query_heads'], site['kv_heads'], site.get('gqa')
        if isinstance(gqa, dict):
            _unknown(gqa, f'attention.{site_id}.gqa')
        elif type(gqa) is not bool:
            raise ValueError(f'attention.{site_id}.gqa must be bool or explicit unknown')
        if type(qh) is int and type(kh) is int:
            if qh % kh:
                raise ValueError(f'attention.{site_id}: query heads must be divisible by kv heads')
            if type(gqa) is bool and gqa != (qh != kh):
                raise ValueError(f'attention.{site_id}.gqa contradicts query/kv heads')
        _texts(site.get('sources'), f'attention.{site_id}.sources', nonempty=True)
        if 'precision' in site:
            if isinstance(site['precision'], dict):
                _unknown(site['precision'], f'attention.{site_id}.precision')
            else:
                _text(site['precision'], f'attention.{site_id}.precision')
        for name, shape in _object(site.get('shapes', {}), f'attention.{site_id}.shapes').items():
            _shape(shape, f'attention.{site_id}.shapes.{name}')

    caps = _object(spec['capabilities'], 'capabilities')
    for key in ('frameworks', 'hardware', 'samplers'):
        _texts(caps.get(key), f'capabilities.{key}', nonempty=True)
    _text(caps.get('precision'), 'capabilities.precision')
    if defaults['sampler'] not in caps['samplers']:
        raise ValueError('head default sampler is not declared in capabilities.samplers')
    if head.get('kind') == 'regression' and caps['samplers'] != ['none']:
        raise ValueError('regression head uses sampler="none"')
    if head['type'] == 'autoregressive_tokens' and caps['samplers'] != ['autoregressive']:
        raise ValueError('autoregressive head uses sampler="autoregressive"')
    if head['type'] == 'flow_matching' and set(caps['samplers']) - {'euler', 'probeflow'}:
        raise ValueError('flow_matching head supports Euler/ProbeFlow samplers')
    backends = _object(caps.get('backends'), 'capabilities.backends')
    if not backends:
        raise ValueError('capabilities.backends must declare an execution backend')
    for backend, frameworks in backends.items():
        _text(backend, 'backend')
        _texts(frameworks, f'capabilities.backends.{backend}', nonempty=True)
        if set(frameworks) - set(caps['frameworks']):
            raise ValueError(f'backend {backend} refers to unsupported frameworks')
    options = _object(caps.get('options'), 'capabilities.options')
    for name, schema in options.items():
        _object(schema, f'option {name}')
        if schema.get('type') not in ('bool', 'integer', 'number', 'string', 'path', 'object', 'array', 'any'):
            raise ValueError(f'option {name} requires a declared type')
        if 'enum' in schema and (not isinstance(schema['enum'], list) or not schema['enum']):
            raise ValueError(f'option {name}.enum must be a nonempty list')
        for bound in ('minimum', 'exclusive_minimum'):
            if bound in schema and (type(schema[bound]) not in (int, float) or not math.isfinite(schema[bound])):
                raise ValueError(f'option {name}.{bound} must be a finite number')
            if bound in schema and schema['type'] not in ('integer', 'number'):
                raise ValueError(f'option {name}.{bound} requires a numeric option type')
        if 'default' in schema:
            try:
                _option_value(schema['default'], schema, f'{name}.default')
            except (TypeError, ValueError) as error:
                raise ValueError(str(error)) from error
        for backend, value in _object(schema.get('default_by_backend', {}), f'{name}.default_by_backend').items():
            if backend not in backends:
                raise ValueError(f'{name}.default_by_backend refers to an unknown backend')
            try:
                _option_value(value, schema, f'{name}.default_by_backend.{backend}')
            except (TypeError, ValueError) as error:
                raise ValueError(str(error)) from error
        for value, precision in _object(schema.get('precision_by_value', {}), f'{name}.precision_by_value').items():
            if schema['type'] != 'string':
                raise ValueError(f'{name}.precision_by_value requires a string option')
            _option_value(value, schema, f'{name}.precision_by_value')
            _text(precision, f'{name}.precision_by_value.{value}')
    if caps.get('action_domain_resolution') not in ('static', 'checkpoint'):
        raise ValueError('capabilities.action_domain_resolution must be static or checkpoint')
    _object(caps.get('execution_coverage'), 'capabilities.execution_coverage')
    for name, value in _object(spec['features'], 'features').items():
        if type(value) is not bool:
            _unknown(value, f'features.{name}')
    tools = _object(spec['tools'], 'tools')
    for key in ('tags', 'supported', 'unsupported'):
        _texts(tools.get(key), f'tools.{key}')
    if set(tools['supported']) & set(tools['unsupported']):
        raise ValueError('a tool cannot be both supported and unsupported')
    defaults = _object(tools.get('defaults', {}), 'tools.defaults')
    if 'euler_steps' in defaults:
        steps = defaults['euler_steps']
        if not isinstance(steps, list) or not steps:
            raise ValueError('tools.defaults.euler_steps must be a nonempty integer list')
        for value in steps:
            _positive(value, 'tools.defaults.euler_steps')
        if len(set(steps)) != len(steps):
            raise ValueError('tools.defaults.euler_steps contains duplicate values')
    for key in ('num_steps', 'probeflow_max_steps'):
        if defaults.get(key) is not None:
            _positive(defaults[key], f'tools.defaults.{key}')
    if 'options' in defaults:
        validate_options(spec, _object(defaults['options'], 'tools.defaults.options'))
    if 'inductor' in tools:
        recipe = _object(tools['inductor'], 'tools.inductor')
        if recipe.get('backend') not in backends:
            raise ValueError('tools.inductor.backend must declare an available backend')
        for key in ('frameworks', 'samplers', 'hardware'):
            values = _texts(recipe.get(key, []), f'tools.inductor.{key}')
            if set(values) - set(caps[key]):
                raise ValueError(f'tools.inductor.{key} refers to unavailable capabilities')
        scoped_stages = _texts(recipe.get('stages'), 'tools.inductor.stages', nonempty=True)
        if set(scoped_stages) - set(stage_ids):
            raise ValueError('tools.inductor.stages refers to unknown model stages')
        values = _object(recipe.get('options'), 'tools.inductor.options')
        if not values:
            raise ValueError('tools.inductor.options must declare an implemented option')
        validate_options(spec, values)
        removed = _texts(recipe.get('remove_options', []), 'tools.inductor.remove_options')
        if set(removed) - set(options):
            raise ValueError('tools.inductor.remove_options refers to unknown options')
        if 'sources' in recipe:
            _texts(recipe['sources'], 'tools.inductor.sources', nonempty=True)
    adapter = _object(spec['adapter'], 'adapter')
    for name, hook in _object(adapter.get('hooks'), 'adapter.hooks').items():
        if not isinstance(hook, str) or not _HOOK.fullmatch(hook):
            raise ValueError(f'Invalid adapter hook {name!r}: {hook!r}')
    runtime = _object(adapter.get('runtime', {}), 'adapter.runtime')
    for name, value in runtime.items():
        if type(value) is not bool:
            raise ValueError(f'adapter.runtime.{name} must be bool')
    author = _object(spec['author'], 'author')
    if not isinstance(author.get('adapter'), str) or not _MODULE.fullmatch(author['adapter']):
        raise ValueError('author.adapter must be a dotted module name')
    _text(author.get('source_package'), 'author.source_package')
    _texts(author.get('options'), 'author.options')
    _positive(author.get('default_steps'), 'author.default_steps')
    if author.get('fixed_steps') is not None:
        _positive(author['fixed_steps'], 'author.fixed_steps')
    if type(author.get('requires_noise_seeds')) is not bool:
        raise ValueError('author.requires_noise_seeds must be bool')
    fixture = _object(spec['fixture'], 'fixture')
    _text(fixture.get('recipe'), 'fixture.recipe')
    _object(fixture.get('fields'), 'fixture.fields')
    _texts(fixture.get('sources'), 'fixture.sources', nonempty=True)
    quality = _object(spec['quality'], 'quality')
    _text(quality.get('scope'), 'quality.scope')
    if quality.get('action_dim') is not None:
        _positive(quality['action_dim'], 'quality.action_dim')
        if type(model_dim) is int and quality['action_dim'] > model_dim:
            raise ValueError('quality.action_dim exceeds model_dim')
    thresholds = quality.get('thresholds')
    if isinstance(thresholds, dict) and thresholds.get('status') == 'unknown':
        _unknown(thresholds, 'quality.thresholds')
    else:
        _object(thresholds, 'quality.thresholds')
        if not thresholds or set(thresholds) - {'max_relative_l2', 'max_abs', 'min_cosine'}:
            raise ValueError('quality.thresholds must declare known metrics or an explicit unknown')
        for key, value in thresholds.items():
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (not -1 <= value <= 1 if key == 'min_cosine' else value < 0)):
                raise ValueError(f'quality threshold {key} is invalid')
    _texts(quality.get('sources'), 'quality.sources', nonempty=True)
    return spec


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate JSON key {key!r}')
        value[key] = item
    return value


def load_spec(path):
    """Read one strict JSON declaration, rejecting duplicate and nonfinite keys."""
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
        return validate_spec(value, source=str(path))
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot load model spec {path}: {error}') from error


def _read_directory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'Model spec directory does not exist: {directory}')
    paths = sorted(directory.glob('*.json'))
    if not paths:
        raise ValueError(f'Model spec directory contains no JSON declarations: {directory}')
    return [load_spec(path) for path in paths]


@lru_cache(maxsize=1)
def _builtins():
    return _read_directory(BUILTIN_SPEC_DIR)


def discover_specs(*, spec_dir=None):
    """Return canonical id -> validated declaration, with optional local additions."""
    specs = list(_builtins())
    if spec_dir is not None and Path(spec_dir).resolve() != BUILTIN_SPEC_DIR.resolve():
        specs.extend(_read_directory(spec_dir))
    result, aliases = {}, {}
    for spec in specs:
        model = spec['model']
        if model in result:
            raise ValueError(f'duplicate model id {model!r}')
        for alias in {model, *spec['aliases']}:
            if alias in aliases:
                raise ValueError(f'duplicate model alias {alias!r} in {aliases[alias]!r} and {model!r}')
            aliases[alias] = model
        result[model] = spec
    return deepcopy(result)


def list_models(*, spec_dir=None):
    """Canonical model ids, in deterministic directory discovery order."""
    return tuple(discover_specs(spec_dir=spec_dir))


def get_spec(model, *, spec_dir=None):
    """Resolve a canonical id or alias and return its detached declaration."""
    specs = discover_specs(spec_dir=spec_dir)
    if isinstance(model, str):
        requested = model.lower()
        for key, spec in specs.items():
            if requested == key or requested in spec['aliases']:
                return spec
    raise ValueError(f'Unknown VLA model {model!r}; use {", ".join(specs)} or an explicit spec_dir')


def resolve_model(model, *, spec_dir=None):
    return get_spec(model, spec_dir=spec_dir)['model']
