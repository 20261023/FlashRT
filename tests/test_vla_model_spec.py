"""Declarative model discovery must stay CPU-only and fail on broken contracts."""

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


def api():
    from flash_rt.vla import model_spec
    return model_spec


def external_spec(head='other'):
    spec = api().get_spec('pi05')
    spec.update(model='external_robot', name='External test robot', aliases=['external-test'])
    sampler = 'none' if head == 'other' else 'autoregressive'
    spec['head'] = {'type': head, 'defaults': {'sampler': sampler},
                    **({'vocab_size': 256, 'max_tokens': 32} if head == 'autoregressive_tokens'
                       else {'kind': 'regression'})}
    spec['capabilities']['samplers'] = [sampler]
    spec['capabilities']['options'] = {}
    spec['adapter'] = {'hooks': {}, 'runtime': {}}
    stage = 'action.regression' if head == 'other' else 'token.decode'
    spec['stages'] = [{'stage_id': stage, 'kind': 'compute', 'title': 'Synthetic action head',
                       'seams': [], 'profiling_keys': [], 'notes': []}]
    spec['profiling'] = {'keys': {}, 'expected_keys': {sampler: []}, 'strategy': {}}
    spec['attention'] = []
    spec['features'] = {}
    spec['tools'] = {'tags': [head], 'supported': [], 'unsupported': []}
    for name in ('source', 'weights', 'license'):
        spec['identity'][name] = {
            'status': 'unknown', 'value': None, 'sources': ['Synthetic test declaration, not a model asset.']}
    spec['identity']['source'].update(repository='synthetic-test-source', revision=None)
    spec['author']['adapter'] = 'external_author_adapter'
    spec['fixture']['head_type'] = head
    return spec


def write_spec(directory, spec):
    path = directory / (spec['model'] + '.json')
    path.write_text(json.dumps(spec), encoding='utf-8')
    return path


def test_discovery_exposes_five_independent_specs_and_resolves_aliases():
    module = api()
    assert set(module.list_models()) == {'pi0', 'pi05', 'hyvla', 'fastwam', 'dm05'}
    spec = module.get_spec('π0.5')
    assert spec['model'] == 'pi05'
    assert spec['head']['type'] == 'flow_matching'
    spec['actions']['model_dim'] = 99
    assert module.get_spec('pi05')['actions']['model_dim'] == 32


def test_loader_can_be_imported_standalone_without_any_runtime_dependency():
    path = Path(__file__).parents[1] / 'flash_rt/vla/model_spec.py'
    assert path.is_file(), 'CPU-only spec loader is missing'
    program = """
import importlib.util, sys
path = sys.argv[1]
spec = importlib.util.spec_from_file_location('standalone_model_spec', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert len(module.discover_specs()) == 5
assert not any(name in sys.modules for name in ('torch', 'numpy', 'jax'))
assert not any(name.startswith('flash_rt.models.') for name in sys.modules)
"""
    subprocess.run([sys.executable, '-I', '-S', '-c', program, str(path)], check=True)


@pytest.mark.parametrize('head,sampler', [
    ('other', 'none'), ('autoregressive_tokens', 'autoregressive'),
])
def test_external_head_declaration_drives_alias_and_backend_config(tmp_path, head, sampler):
    module = api()
    spec = external_spec(head)
    write_spec(tmp_path, spec)
    discovered = module.discover_specs(spec_dir=tmp_path)
    assert discovered['external_robot']['head']['type'] == head
    from flash_rt.vla import BackendConfig, describe_backend
    from flash_rt.vla.registry import validate_config
    config = BackendConfig(model='external-test', checkpoint='/absent', spec_dir=tmp_path)
    assert config.model == 'external_robot'
    assert config.sampler == sampler
    assert validate_config(config)['head']['type'] == head
    result = describe_backend(config)
    assert result['head_type'] == head
    assert result['runtime_verified'] is False
    assert 'external_robot' not in module.list_models()


def test_external_spec_without_binding_refuses_loading_before_cuda(tmp_path, monkeypatch):
    spec = external_spec()
    write_spec(tmp_path, spec)
    from flash_rt.vla import BackendConfig, load_backend, registry
    def forbidden(*args):
        raise AssertionError('a metadata-only declaration cannot probe hardware')
    monkeypatch.setattr(registry, '_cuda_target', forbidden)
    with pytest.raises(NotImplementedError, match='binding'):
        load_backend(BackendConfig(model='external-test', checkpoint='/absent', spec_dir=tmp_path))


@pytest.mark.parametrize('head', ['other', 'autoregressive_tokens'])
def test_external_adapter_loads_through_registry_without_model_name_branches(tmp_path, monkeypatch, head):
    import numpy as np
    from flash_rt.vla import BackendConfig, load_backend, registry
    module = ModuleType('external_model_spec_fixture')
    class Frontend:
        def __init__(self, checkpoint, policy):
            self.policy = policy
        def infer(self, observation):
            return {'actions': np.ones((2, 7), dtype=np.float32),
                    'sampler_metadata': {'sampler': self.policy}}
    module.Frontend = Frontend
    module.binding = lambda config, hardware: (
        module.__name__, 'Frontend', {'policy': config.sampler})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(registry, '_cuda_target', lambda requested: 'cuda_sm80')
    spec = external_spec(head)
    spec['adapter']['hooks']['binding'] = module.__name__ + ':binding'
    write_spec(tmp_path, spec)
    config = BackendConfig(model='external-test', checkpoint='/absent', spec_dir=tmp_path)
    backend = load_backend(config)
    result = backend.infer({}, prompt='pick up the object')
    assert result['actions'].shape == (2, 7)
    assert result['backend_metadata']['model'] == 'external_robot'
    assert result['sampler_metadata'] == {'sampler': 'none' if head == 'other' else 'autoregressive'}
    backend.close()


@pytest.mark.parametrize('mutation,match', [
    (lambda s: s.update(schema_version=True), 'schema_version'),
    (lambda s: s['identity']['source'].update(status='pinned', value=None), 'identity.source'),
    (lambda s: s['identity']['weights'].update(sources=[]), 'sources'),
    (lambda s: s['actions'].update(effective_dim=33), 'effective_dim'),
    (lambda s: s['actions'].update(channels=[0, 0]), 'channels'),
    (lambda s: s['attention'][1].update(query_heads=7, kv_heads=2), 'heads'),
    (lambda s: s['attention'][1].update(head_dim=True), 'head_dim'),
    (lambda s: s['attention'][1].update(gqa=False), 'gqa'),
    (lambda s: s['attention'][1].update(precision=True), 'precision'),
    (lambda s: s['attention'][1]['shapes'].update(q=[1, True, 8, 256]), 'shapes'),
    (lambda s: s['profiling']['keys'].update(broken=['missing.stage']), 'stage'),
    (lambda s: s['stages'][1].update(profiling_keys=['missing_interval']), 'profiling'),
    (lambda s: s['stages'][0].update(profiling_keys=[]), 'profiling'),
    (lambda s: s['stages'][1].update(seams=['../outside.py:run']), 'seam'),
    (lambda s: s['adapter']['hooks'].update(binding='unsafe()'), 'hook'),
    (lambda s: s['quality'].update(thresholds={'max_relative_l2': -1}), 'threshold'),
    (lambda s: s['capabilities']['options'].update(foo={'type': 'bool', 'default': 'yes'}), 'default'),
    (lambda s: s['head'].update(type='invented'), 'head'),
])
def test_validation_rejects_inconsistent_declarations(mutation, match):
    spec = api().get_spec('pi05')
    mutation(spec)
    with pytest.raises(ValueError, match=match):
        api().validate_spec(spec)


def test_autoregressive_heads_need_token_geometry():
    spec = external_spec('autoregressive_tokens')
    spec['head'].pop('vocab_size')
    with pytest.raises(ValueError, match='vocab_size'):
        api().validate_spec(spec)


@pytest.mark.parametrize('field,value', [('vocab_size', 0), ('vocab_size', True), ('max_tokens', -1)])
def test_autoregressive_token_geometry_rejects_invalid_sizes(field, value):
    spec = external_spec('autoregressive_tokens')
    spec['head'][field] = value
    with pytest.raises(ValueError, match=field):
        api().validate_spec(spec)


def test_autoregressive_tokens_do_not_accept_denoising_defaults():
    spec = external_spec('autoregressive_tokens')
    spec['head']['defaults']['num_steps'] = 10
    with pytest.raises(ValueError, match='num_steps'):
        api().validate_spec(spec)


@pytest.mark.parametrize('alias', ['pi05', 'π0.5'])
def test_external_alias_collision_cannot_replace_builtin(tmp_path, alias):
    spec = external_spec()
    spec['aliases'] = [alias]
    write_spec(tmp_path, spec)
    with pytest.raises(ValueError, match='alias|duplicate'):
        api().discover_specs(spec_dir=tmp_path)


def test_duplicate_json_keys_are_not_silently_accepted(tmp_path):
    path = tmp_path / 'ambiguous.json'
    path.write_text('{"schema_version": 1, "model": "first", "model": "second"}')
    with pytest.raises(ValueError, match='duplicate'):
        api().load_spec(path)


def test_attention_and_quality_metadata_preserve_actual_comparison_domains():
    dm = api().get_spec('dm05')
    assert dm['actions']['worker_domain'] == 'denormalized-absolute-dos-w1-14d'
    assert dm['actions']['effective_dim'] == 14
    assert dm['actions']['channels'] == list(range(14))
    assert dm['quality']['thresholds'] == {'max_relative_l2': 0.05, 'min_cosine': 0.999}
    pi = api().get_spec('pi0')
    assert pi['actions']['worker_domain'] == 'normalized-padded-model-actions'
    assert pi['capabilities']['action_domain_resolution'] == 'checkpoint'
    assert pi['author']['adapter'] == 'flash_rt.vla.author_adapters.openpi'
    assert dm['identity']['source']['revision'] == '03f799120d425ead51d9482bb1da950b6dabade9'


def test_spec_option_schemas_reject_invalid_values_before_adapter_import():
    from flash_rt.vla import BackendConfig
    from flash_rt.vla.registry import validate_config
    with pytest.raises(ValueError, match='num_views'):
        validate_config(BackendConfig(model='pi0', checkpoint='/absent', options={'num_views': True}))


def test_external_discovery_scope_roundtrips_backend_codec_and_candidate(tmp_path):
    spec = external_spec()
    write_spec(tmp_path, spec)
    from flash_rt.vla import BackendConfig
    from flash_rt.vla.autotune.schema import Candidate, decode_backend, encode_backend
    config = BackendConfig(model='external-test', checkpoint='/absent', spec_dir=tmp_path)
    encoded = encode_backend(config)
    assert encoded['spec_dir'] == str(tmp_path)
    assert decode_backend(json.loads(json.dumps(encoded))) == config
    candidate = Candidate('external', config, 'fixture')
    assert Candidate.from_dict(json.loads(json.dumps(candidate.to_dict()))) == candidate


def test_legacy_backend_codec_does_not_change_existing_candidate_identity():
    from flash_rt.vla import BackendConfig
    from flash_rt.vla.autotune.schema import Candidate, decode_backend, encode_backend
    legacy = {
        'model': 'pi05', 'checkpoint': '/weights', 'backend': 'flashrt',
        'sampler': 'euler', 'hardware': 'auto', 'framework': 'torch',
        'precision': 'auto', 'num_steps': None, 'probeflow': None, 'options': {},
    }
    assert encode_backend(decode_backend(legacy)) == legacy
    assert 'spec_dir' not in Candidate('legacy', BackendConfig('pi05', '/weights'), 'w').to_dict()['backend']


def test_compiler_recipe_requires_declared_stage_scope():
    spec = api().get_spec('fastwam')
    spec['tools']['inductor']['stages'] = ['not.a.model.stage']
    with pytest.raises(ValueError, match='inductor.stages'):
        api().validate_spec(spec)


def test_external_compiler_recipe_can_use_parent_provenance_without_new_source_claim():
    spec = api().get_spec('fastwam')
    spec['tools']['inductor'].pop('sources')
    assert api().validate_spec(spec)['tools']['inductor']['options']['compile_action_infer'] is True


def test_backend_description_cannot_mutate_later_capability_identity():
    from flash_rt.vla import BackendConfig, describe_backend
    config = BackendConfig('pi05', '/absent')
    first = describe_backend(config)
    first['identity']['source']['revision'] = 'bad'
    first['action_channels'].append(31)
    second = describe_backend(config)
    assert second['identity']['source']['revision'] == 'dd5c5ca93ce229017477a75fdda9f16654ab91ef'
    assert second['action_channels'] == [0, 1, 2, 3, 4, 5, 6]
