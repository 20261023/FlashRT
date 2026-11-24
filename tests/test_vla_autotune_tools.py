"""Tool registry (DESIGN F): specs, config tools, lever pricing, patches and the CLI."""

import json
import importlib
import os
from pathlib import Path
import shutil

import pytest

from flash_rt.sampling import ProbeFlowConfig
from flash_rt.vla import BackendConfig
from flash_rt.vla.model_spec import get_spec, list_models
from flash_rt.vla.autotune import snapshots as snap
from flash_rt.vla.autotune import tools
from flash_rt.vla.autotune.controller import initialize_campaign, main as controller_main
from flash_rt.vla.autotune.schema import Candidate, encode_backend
from flash_rt.vla.autotune.search import enumerate_candidates
from flash_rt.vla.autotune.stages import REFERENCED_TOOL_IDS, STAGE_IDS, STAGES
from flash_rt.vla.autotune.tools import levers as levers_module
from flash_rt.vla.autotune.tools import patches
from flash_rt.vla.autotune.tools.__main__ import main as tools_main
from flash_rt.vla.autotune.tools.artifacts import PatchProposal
from flash_rt.vla.autotune.tools.builtin import effective_backend
from flash_rt.vla.autotune.tools.external import ADAPTERS, adapter_module, external_specs
from flash_rt.vla.autotune.tools.spec import HARDWARE_CAPABILITY, ToolSpec, Upstream, hardware_capability
from flash_rt.vla.autotune.tools.spec import match_predicate
from flash_rt.vla.registry import _CAPABILITIES, _CUDA_TARGETS, validate_config

from test_vla_campaign_snapshot import git, head, make_repo


needs_git = pytest.mark.skipif(shutil.which('git') is None, reason='git executable required')

FASTWAM_OPTIONS = {'sigma_shift': 5.0, 'action_horizon': 32, 'model_config': '/w/fastwam/model.json',
                   'backbone_dir': '/w/wan22'}
PINNED = {'kernel.forgekernel': 'b2c8ae8510e02c28576808910a6b8827721ab7dc',
          'kernel.kernelagent': 'e0647170da36ef9b059ac0bd3d60103aa4ed378b',
          'engine.apxinf': 'b05b40990387e8c48e7d6267a790664981a665e4',
          'compile.aitune': '1d3a4236ad8f458669d5633715b6956125f2289c'}
PI05_ENV = ('FVK_PI05_RTX_FORCE_INT8', 'FVK_PI05_RTX_INT8_ENCODER_ONLY', 'FVK_PI05_RTX_INT8_VISION',
            'FLASHRT_PI05_PREFIX_PRECISION', 'FLASHRT_PI05_DECODER_PRECISION', 'FLASHRT_PI05_DECODER_KERNEL',
            'FLASHRT_PI05_STATE_PROMPT_MODE')


@pytest.fixture(autouse=True)
def clean_pi05_env(monkeypatch):
    for name in PI05_ENV:
        monkeypatch.delenv(name, raising=False)


def candidate(candidate_id, model, **fields):
    return Candidate(candidate_id, BackendConfig(model=model, checkpoint=f'/w/{model}', **fields), 'frames')


def baselines():
    """Realistic baselines per model, including non-default samplers/backends."""
    return {
        'dm05': [candidate('dm-base', 'dm05', hardware='cuda_sm80', num_steps=10,
                           options={'source_dir': '/w/opendm'})],
        'pi0': [candidate('pi0-base', 'pi0', num_steps=10, options={'num_views': 2, 'chunk_size': 50}),
                candidate('pi0-eager', 'pi0', options={'use_cuda_graph': False})],
        'pi05': [candidate('pi05-base', 'pi05', num_steps=10,
                           options={'num_views': 2, 'chunk_size': 10, 'max_prompt_len': 48}),
                 candidate('pi05-pf', 'pi05', sampler='probeflow', probeflow=ProbeFlowConfig(max_steps=12),
                           options={'use_cuda_graph': False, 'vision_pool_factor': 2})],
        'hyvla': [candidate('hy-base', 'hyvla', num_steps=10),
                  candidate('hy-eager', 'hyvla', options={'use_cuda_graph': False,
                                                         'numerical_profile': 'semantic_bf16'})],
        'fastwam': [candidate('fw-base', 'fastwam', num_steps=10, options=FASTWAM_OPTIONS),
                    candidate('fw-graph', 'fastwam', num_steps=10,
                              options={**FASTWAM_OPTIONS, 'native_linear_backend': 'cached',
                                       'use_cuda_graph': True}),
                    candidate('fw-torch', 'fastwam', backend='torch', sampler='probeflow',
                              options=FASTWAM_OPTIONS)],
    }


def ready_config_tools():
    return [tool for tool in tools.all_tools() if tool.lane == 'config' and tool.status == 'ready']


# --------------------------------------------------------------------------- registry facts

def test_every_stage_default_tool_is_registered_and_stages_exist():
    ids = {tool.tool_id for tool in tools.all_tools()}
    assert REFERENCED_TOOL_IDS <= ids
    for info in STAGES.values():
        for tool_id in info.default_tools:
            tool = tools.get_tool(tool_id)
            assert tool.tool_id == tool_id or tool_id in tool.aliases
    for tool in tools.all_tools():
        assert tool.stages and set(tool.stages) <= set(STAGE_IDS), tool.tool_id
        assert tool.skills and all(isinstance(skill, str) and skill.strip() for skill in tool.skills)
        assert tool.status_for(None) in tools.STATUSES
        if tool.status != 'ready':
            assert tool.reason.strip(), tool.tool_id
        json.dumps(tool.to_dict(), allow_nan=False)
    assert len(ids) == len(tools.all_tools())
    assert {'cache.pi05_cross_frame_kv', 'throughput.pi05_batch'} <= ids


def test_stage_default_tools_are_listed_for_that_stage():
    # `tools list --stage S` must show every tool the stage map recommends for S.
    for stage_id, info in STAGES.items():
        listed = {tool.tool_id for tool in tools.list_tools(stage=stage_id, include_unavailable=True)}
        for tool_id in info.default_tools:
            assert stage_id in tools.get_tool(tool_id).stages, (stage_id, tool_id)
            assert tool_id in listed, (stage_id, tool_id)


def test_config_tools_do_not_route_to_the_operator_gate():
    # Config candidates are judged by the model-level campaign gate only.
    for tool in tools.all_tools():
        if tool.lane == 'config':
            assert 'vla-operator-gate' not in tool.skills, tool.tool_id
        if tool.lane == 'port':
            assert 'vla-port-workflow' in tool.skills, tool.tool_id


def test_unknown_tool_and_filters_raise():
    with pytest.raises(ValueError, match='Unknown tool'):
        tools.get_tool('nope')
    for kwargs in ({'model': 'groot'}, {'hardware': 'h100'}, {'stage': 'x'}, {'lane': 'magic'}):
        with pytest.raises(ValueError):
            tools.list_tools(**kwargs)


def test_hardware_labels_follow_registry_cuda_targets():
    assert dict(HARDWARE_CAPABILITY) == {label: capability for capability, label in _CUDA_TARGETS.items()}
    assert hardware_capability('cuda_sm80') == (8, 0) and hardware_capability('auto') is None


def test_external_specs_match_adapters_and_pinned_commits():
    specs = external_specs()
    assert set(specs) == set(ADAPTERS) == set(PINNED)
    for tool_id, spec in specs.items():
        tool = tools.get_tool(tool_id)
        module = adapter_module(tool_id)
        assert tool.upstream == Upstream.from_spec(spec['upstream'])
        assert tool.upstream.commit == spec['upstream']['commit'] == module.UPSTREAM_COMMIT == PINNED[tool_id]
        assert tool.upstream.notice == spec['upstream']['notice'] and tool.upstream.path is None
        for name in ('title', 'lane', 'exactness', 'status', 'reason'):
            assert getattr(tool, name) == spec[name]
        assert tool.stages == tuple(spec['stages']) and tool.models == tuple(spec['models'])
        assert tool.skills == tuple(spec['skills']) and tool.adapter == ADAPTERS[tool_id]
        assert tool.propose is None


def test_aitune_is_a_listed_unimplemented_patch_for_module_or_functional_torch_regions():
    tool = tools.get_tool('compile.aitune')
    assert tool.lane == 'patch' and tool.status == 'needs_implementation' and tool.propose is None
    assert tool.adapter == 'flash_rt.vla.autotune.tools.external.aitune'
    listed = {item.tool_id for item in tools.list_tools(model='fastwam', hardware='cuda_sm80',
                                                         stage='denoise.velocity',
                                                         include_unavailable=True)}
    assert 'compile.aitune' in listed
    assert 'compile.aitune' not in {item.tool_id for item in tools.list_tools(model='fastwam')}
    for model in ('pi0', 'pi05'):
        assert 'compile.aitune' not in {item.tool_id for item in tools.list_tools(
            model=model, include_unavailable=True)}
    with pytest.raises(ValueError, match='tools.external.aitune'):
        tools.propose('compile.aitune', baselines()['fastwam'][0])
    for stage_id in tool.stages:
        assert stage_id in STAGE_IDS
    assert {stage_id for stage_id, info in STAGES.items()
            if 'compile.aitune' in info.default_tools} <= set(tool.stages)
    skills_root = Path(__file__).resolve().parents[1] / '.agents' / 'skills'
    for skill in tool.skills:
        assert (skills_root / skill / 'SKILL.md').is_file(), skill


def test_fp8_and_nvfp4_never_available_on_a800():
    for include in (False, True):
        listed = {tool.tool_id for tool in tools.list_tools(hardware='cuda_sm80', include_unavailable=include)}
        assert not {'quant.fp8', 'quant.nvfp4'} & listed
        auto = {tool.tool_id for tool in tools.list_tools(hardware='auto', include_unavailable=include)}
        assert not {'quant.fp8', 'quant.nvfp4'} & auto
    assert 'quant.fp8' in {tool.tool_id for tool in tools.list_tools(hardware='rtx_sm89', include_unavailable=True)}
    assert 'quant.nvfp4' in {tool.tool_id for tool in tools.list_tools(hardware='rtx_sm120',
                                                                         include_unavailable=True)}
    assert 'quant.nvfp4' not in {tool.tool_id for tool in tools.list_tools(hardware='rtx_sm89',
                                                                             include_unavailable=True)}
    for tool_id in ('quant.fp8', 'quant.nvfp4'):
        assert tools.get_tool(tool_id).status == 'blocked'
        assert tool_id not in {tool.tool_id for tool in tools.list_tools(hardware='rtx_sm120')}


def test_list_filters_by_model_stage_lane_and_readiness():
    ready = tools.list_tools()
    assert ready and all(tool.status == 'ready' for tool in ready)
    everything = tools.list_tools(include_unavailable=True)
    assert len(everything) == len(tools.all_tools())
    pi05 = {tool.tool_id for tool in tools.list_tools(model='pi05')}
    assert {'prefix.vision_pool', 'prefix.state_prompt_mode', 'sampler.euler_steps',
            'profile.pi05_dataflow'} <= pi05
    assert not {'gemm.fastwam_linear_backend', 'precision.hy_numerical_profile'} & pi05
    # The author latency scripts make the official baseline ready for every model.
    assert 'baseline.official_impl' in pi05
    assert 'compile.inductor' not in pi05
    fastwam = {tool.tool_id for tool in tools.list_tools(model='fastwam')}
    assert {'gemm.fastwam_linear_backend', 'baseline.official_impl', 'compile.inductor'} <= fastwam
    assert not {'compile.fastwam_inductor', 'compile.dm05_inductor'} & fastwam
    assert 'prefix.vision_pool' not in fastwam
    hy = {tool.tool_id for tool in tools.list_tools(model='hyvla')}
    assert 'precision.hy_numerical_profile' in hy and 'gemm.fastwam_linear_backend' not in hy
    schedule = tools.list_tools(stage='denoise.schedule')
    assert {tool.tool_id for tool in schedule} >= {'sampler.euler_steps', 'sampler.probeflow'}
    assert all('denoise.schedule' in tool.stages for tool in schedule)
    assert all(tool.lane == 'patch' for tool in tools.list_tools(lane='patch', include_unavailable=True))


def test_official_impl_uses_declared_author_adapter_and_supported_options():
    tool = tools.get_tool('baseline.official_impl')
    for model in list_models():
        spec = get_spec(model)
        assert tool.status_for(model) == 'ready'
        assert callable(importlib.import_module(spec['author']['adapter']).load)
        assert isinstance(spec['author']['options'], list)
        assert all(isinstance(name, str) and name for name in spec['author']['options'])
    assert 'author.adapter' in tool.reason and 'author.options' in tool.reason


def _fastwam_baseline(**options):
    config = BackendConfig(model='fastwam', checkpoint='/ckpt', backend=options.pop('backend', 'torch'),
                           num_steps=20, options={'sigma_shift': 5.0, 'action_horizon': 32,
                                                  'model_config': '/cfg.json', 'backbone_dir': '/wan',
                                                  **options})
    return Candidate(candidate_id='fw', backend=config, workload_id='w')


def test_fastwam_compile_tool_emits_a_registry_valid_compiled_candidate():
    out = tools.propose('compile.inductor', _fastwam_baseline(use_cuda_graph=True))
    assert len(out) == 1
    backend = out[0].backend
    assert backend.backend == 'torch'
    assert backend.options['compile_action_infer'] is True and backend.options['use_cuda_graph'] is False
    assert 'native_linear_backend' not in backend.options
    validate_config(backend)
    assert out[0].tool == 'compile.inductor' and out[0].stage == 'host.launch'
    assert tools.propose('compile.inductor', _fastwam_baseline(compile_action_infer=True)) == []


def test_compiler_historical_ids_resolve_to_one_canonical_recipe_tool():
    canonical = tools.get_tool('compile.inductor')
    for alias in ('compile.fastwam_inductor', 'compile.dm05_inductor'):
        assert tools.get_tool(alias) is canonical
    for model in ('dm05', 'fastwam'):
        baseline = baselines()[model][0]
        alias = f'compile.{model}_inductor'
        assert tools.propose(alias, baseline) == tools.propose('compile.inductor', baseline)
        assert all(item.tool == 'compile.inductor' for item in tools.propose(alias, baseline))


def test_compiler_routes_only_to_stages_declared_by_each_existing_product_recipe():
    tool = tools.get_tool('compile.inductor')
    for model in ('dm05', 'fastwam'):
        recipe = get_spec(model)['tools']['inductor']
        assert tool.stages_for(model) == tuple(recipe['stages'])
    fastwam_vision = {item.tool_id for item in tools.list_tools(
        model='fastwam', hardware='cuda_sm80', stage='vision.encode')}
    assert 'compile.inductor' not in fastwam_vision
    dm05_vision = {item.tool_id for item in tools.list_tools(
        model='dm05', hardware='cuda_sm80', stage='vision.encode')}
    assert 'compile.inductor' in dm05_vision


def test_effective_backend_uses_declared_backend_specific_option_defaults(tmp_path):
    spec, path = external_spec(tmp_path)
    spec['capabilities']['options']['native_linear_backend'] = {
        'type': 'string', 'enum': ['legacy', 'cached'], 'default_by_backend': {'flashrt': 'cached'}}
    path.write_text(json.dumps(spec))
    native = BackendConfig(model='test_flow', checkpoint='/w', spec_dir=str(tmp_path))
    assert effective_backend(native)['options']['native_linear_backend'] == 'cached'
    torch = BackendConfig(model='test_flow', checkpoint='/w', backend='torch', spec_dir=str(tmp_path))
    assert 'native_linear_backend' not in effective_backend(torch)['options']


def test_graph_and_native_linear_tools_turn_compile_off_on_a_compiled_baseline():
    compiled = _fastwam_baseline(compile_action_infer=True)
    graph = tools.propose('launch.cuda_graph', compiled)
    assert len(graph) == 1 and graph[0].backend.options['compile_action_infer'] is False
    validate_config(graph[0].backend)
    for candidate in tools.propose('gemm.fastwam_linear_backend', compiled):
        validate_config(candidate.backend)
        if candidate.backend.backend == 'flashrt':
            assert candidate.backend.options['compile_action_infer'] is False


def test_tool_models_follow_registry_options():
    """A config tool that emits a model-specific option targets exactly the models that accept it."""
    option_tools = {'prefix.vision_pool': 'vision_pool_factor', 'prefix.state_prompt_mode': 'state_prompt_mode',
                    'precision.hy_numerical_profile': 'numerical_profile',
                    'gemm.fastwam_linear_backend': 'native_linear_backend'}
    for tool_id, option in option_tools.items():
        accepting = tuple(model for model, spec in _CAPABILITIES.items() if option in spec['options'])
        assert set(tools.get_tool(tool_id).matching_models()) == set(accepting), tool_id
        assert tools.get_tool(tool_id).models == ('*',), tool_id


def test_toolspec_validation():
    common = dict(title='t', lane='config', stages=('host.launch',), models=('*',), exactness='exact',
                  skills=('vla-config-sweep',))
    ToolSpec('ok.tool', status='ready', **common)
    with pytest.raises(ValueError, match='reason'):
        ToolSpec('bad.tool', status='blocked', **common)
    with pytest.raises(ValueError, match='stages'):
        ToolSpec('bad.tool', status='ready', **{**common, 'stages': ('nope',)})
    with pytest.raises(ValueError, match='skills'):
        ToolSpec('bad.tool', status='ready', **{**common, 'skills': ()})
    with pytest.raises(ValueError, match='only config/analysis'):
        ToolSpec('bad.tool', status='ready', propose=lambda baseline: [], **{**common, 'lane': 'patch'})
    with pytest.raises(ValueError, match='commit'):
        Upstream('https://example.invalid/repo', 'abc', 'MIT')
    for predicate in ({'path': 'features.variable_steps'}, {'any': []},
                      {'all': [], 'path': 'head.type'}, {'path': 'head.type', 'execute': 'os.system'}):
        with pytest.raises(ValueError, match='model_predicate'):
            ToolSpec('bad.selector', status='ready', model_predicate=predicate, **common)


def external_spec(tmp_path, *, model='test_flow', head='flow_matching', samplers=None, features=None):
    """A separate declaration; no model adapter is imported by the routing tests."""
    spec = get_spec('pi05')
    spec['model'], spec['name'], spec['aliases'] = model, f'Test {model}', []
    spec['adapter']['hooks'] = {}
    spec['head'] = {'type': head, 'defaults': {'sampler': 'euler', 'num_steps': 10}}
    spec['features'] = {'variable_steps': True, **(features or {})}
    spec['tools'] = {'tags': [], 'supported': [], 'unsupported': [],
                     'defaults': {'num_steps': 10, 'options': {'use_cuda_graph': False}}}
    if head == 'other':
        spec['head']['kind'] = 'regression'
        spec['head']['defaults'] = {'sampler': 'none'}
    elif head == 'autoregressive_tokens':
        spec['head'].update(vocab_size=256, max_tokens=32)
        spec['head']['defaults'] = {'sampler': 'autoregressive'}
    spec['capabilities']['samplers'] = samplers or [spec['head']['defaults']['sampler']]
    spec['capabilities']['options'] = {'use_cuda_graph': {'type': 'bool', 'default': False}}
    spec['capabilities']['backends'] = {'flashrt': ['torch'], 'torch': ['torch']}
    spec['capabilities']['frameworks'] = ['torch']
    spec['capabilities']['hardware'] = ['cuda_sm80', 'rtx_sm89']
    spec['profiling']['expected_keys'] = {name: [] for name in spec['capabilities']['samplers']}
    path = tmp_path / f'{model}.json'
    path.write_text(json.dumps(spec))
    return spec, path


def test_feature_routing_extends_to_explicit_spec_dir_without_model_lists(tmp_path):
    _, _ = external_spec(tmp_path, samplers=['euler', 'probeflow'],
                         features={'torch_modules': True})
    listed = {tool.tool_id for tool in tools.list_tools(model='test_flow', hardware='cuda_sm80',
                                                       spec_dir=tmp_path, include_unavailable=True)}
    assert {'sampler.euler_steps', 'sampler.probeflow', 'launch.cuda_graph', 'compile.aitune'} <= listed
    assert not {'compile.inductor', 'prefix.vision_pool', 'precision.hy_numerical_profile',
                'engine.apxinf'} & listed
    baseline = Candidate('new', BackendConfig(model='test_flow', checkpoint='/w', num_steps=10,
                                             spec_dir=str(tmp_path)), 'frames')
    points = tools.propose('sampler.euler_steps', baseline, steps=[4, 10])
    assert len(points) == 1 and points[0].backend.num_steps == 4
    assert points[0].backend.spec_dir == str(tmp_path)
    assert Candidate.from_dict(points[0].to_dict()) == points[0]


@pytest.mark.parametrize('head', ['other', 'autoregressive_tokens'])
def test_non_flow_heads_never_receive_flow_sampler_routes(tmp_path, head):
    external_spec(tmp_path, head=head)
    listed = {tool.tool_id for tool in tools.list_tools(model='test_flow', spec_dir=tmp_path,
                                                       include_unavailable=True)}
    assert not {'sampler.euler_steps', 'sampler.probeflow'} & listed
    schedule = next(row for row in tools.stage_coverage('test_flow', 'cuda_sm80', spec_dir=tmp_path)
                    if row['stage'] == 'denoise.schedule')
    sampler_entries = [entry for entry in schedule['tools'] if entry['lane'] == 'config']
    assert sampler_entries and all(entry['status'] == 'blocked' for entry in sampler_entries)
    assert all('head.type' in entry['reason'] for entry in sampler_entries)


def test_unknown_feature_is_unverified_and_explicit_unsupported_wins(tmp_path):
    spec, path = external_spec(tmp_path, features={
        'variable_steps': {'status': 'unknown', 'value': None, 'sources': ['test fixture']}})
    assert 'sampler.euler_steps' not in {tool.tool_id for tool in tools.list_tools(
        model='test_flow', spec_dir=tmp_path)}
    assert not match_predicate({'path': 'features.variable_steps', 'equals': True},
                               {'features': {'variable_steps': 1}})[0]
    spec['features']['variable_steps'] = True
    spec['tools']['unsupported'] = ['sampler.euler_steps']
    path.write_text(json.dumps(spec))
    matches, reason = tools.get_tool('sampler.euler_steps').match_model('test_flow', spec_dir=tmp_path)
    assert not matches and 'tools.unsupported' in reason


def test_external_baseline_requires_declared_compatibility_in_the_selected_spec(tmp_path):
    spec, path = external_spec(tmp_path)
    apxinf = tools.get_tool('engine.apxinf')
    assert not apxinf.applies_to('test_flow', spec_dir=tmp_path)
    spec['tools']['supported'] = ['engine.apxinf']
    path.write_text(json.dumps(spec))
    assert apxinf.applies_to('test_flow', spec_dir=tmp_path)
    assert apxinf.status_for('test_flow', spec_dir=tmp_path) == 'needs_setup'
    assert 'engine.apxinf' not in {tool.tool_id for tool in tools.list_tools(
        model='test_flow', spec_dir=tmp_path)}


def test_torch_tool_predicate_rejects_a_declaration_with_only_jax_execution(tmp_path):
    spec, path = external_spec(tmp_path, features={'torch_modules': True})
    spec['capabilities']['frameworks'] = ['jax']
    spec['capabilities']['backends'] = {'flashrt': ['jax']}
    path.write_text(json.dumps(spec))
    tool = tools.get_tool('compile.aitune')
    assert not tool.applies_to('test_flow', spec_dir=tmp_path)
    assert tool.status_for('test_flow', spec_dir=tmp_path) == 'blocked'
    assert 'frameworks' in tool.reason_for('test_flow', spec_dir=tmp_path)


def test_compiler_uses_declared_recipe_and_respects_sampler_framework_and_hardware(tmp_path):
    spec, path = external_spec(tmp_path, samplers=['euler', 'probeflow'])
    spec['capabilities']['options']['compile_region'] = {'type': 'bool', 'default': False}
    spec['capabilities']['options']['legacy_region'] = {'type': 'bool'}
    spec['tools']['inductor'] = {
        'backend': 'torch', 'options': {'compile_region': True, 'use_cuda_graph': False},
        'remove_options': ['legacy_region'], 'frameworks': ['torch'], 'samplers': ['euler'],
        'hardware': ['cuda_sm80'], 'stages': ['denoise.velocity'],
        'reason': 'Fixture implemented compiler option', 'sources': ['tests/test_vla_autotune_tools.py'],
    }
    path.write_text(json.dumps(spec))
    baseline = Candidate('new', BackendConfig(
        model='test_flow', checkpoint='/w', hardware='cuda_sm80', num_steps=10,
        options={'use_cuda_graph': True, 'legacy_region': True}, spec_dir=str(tmp_path)), 'frames')
    compiled = tools.propose('compile.inductor', baseline)
    assert len(compiled) == 1
    assert compiled[0].backend.backend == 'torch'
    assert compiled[0].backend.options == {'use_cuda_graph': False, 'compile_region': True}
    assert compiled[0].stage == 'denoise.velocity'
    assert tools.propose('compile.inductor', compiled[0]) == []
    assert 'compile.inductor' not in {tool.tool_id for tool in tools.list_tools(
        model='test_flow', hardware='rtx_sm89', spec_dir=tmp_path)}
    assert 'compile.inductor' not in {tool.tool_id for tool in tools.list_tools(
        model='test_flow', hardware='cuda_sm80', stage='vision.encode', spec_dir=tmp_path)}
    adaptive = Candidate('pf', BackendConfig(
        model='test_flow', checkpoint='/w', hardware='cuda_sm80', sampler='probeflow',
        spec_dir=str(tmp_path)), 'frames')
    with pytest.raises(ValueError, match='samplers'):
        tools.propose('compile.inductor', adaptive)


def test_int8_selector_refuses_incompatible_sampler_and_framework():
    adaptive = baselines()['pi05'][1]
    with pytest.raises(ValueError, match='sampler'):
        tools.propose('quant.int8_pi05', adaptive)
    jax = candidate('jax', 'pi05', framework='jax')
    with pytest.raises(ValueError, match='framework'):
        tools.propose('quant.int8_pi05', jax)


def test_attention_routes_are_checked_per_call_site_on_a800():
    def sites(tool, model, **kwargs):
        spec = tools.get_tool(tool)
        assert spec.lane == 'port' and spec.status == 'needs_implementation'
        assert spec.propose is None
        return spec.attention_sites(model, hardware='cuda_sm80', **kwargs)

    sage = sites('attention.sage', 'dm05')
    vision, = sage['eligible_sites']
    assert vision['id'] == 'vision'
    assert any(row['kind'] == 'head_dim_padding' and row['from'] == 72 and row['to'] == 128
               for row in vision['adaptations'])
    assert {row['id'] for row in sage['ineligible_sites']} == {'prefix', 'action'}
    assert not sites('attention.sage', 'dm05', stage='prefix.prefill')['eligible_sites']
    prefix, = sites('attention.flex', 'dm05', stage='prefix.prefill')['eligible_sites']
    assert prefix['id'] == 'prefix' and prefix['gqa'] and prefix['mask_kind'] == 'arbitrary'
    spatial, = sites('attention.turbo', 'hyvla')['eligible_sites']
    assert spatial['id'] == 'vision_spatial'


def test_attention_ports_remain_unimplemented_and_coverage_explains_hardware():
    for tool_id in ('attention.sdpa_backend', 'attention.flash_attention'):
        tool = tools.get_tool(tool_id)
        assert tool.lane == 'port' and tool.status == 'needs_implementation'
        assert tool.propose is None and 'vla-operator-gate' in tool.skills
    coverage = tools.stage_coverage('pi05', 'cuda_sm80')
    assert {row['stage'] for row in coverage} == set(STAGE_IDS)
    precision = next(row for row in coverage if row['stage'] == 'precision.weights')
    fp8 = next(entry for entry in precision['tools'] if entry['tool_id'] == 'quant.fp8')
    assert fp8['status'] == 'blocked' and fp8['reason']
    attention = next(row for row in coverage if row['stage'] == 'kernel.attention')
    fa = next(entry for entry in attention['tools'] if entry['tool_id'] == 'attention.flash_attention')
    assert fa['status'] == 'needs_implementation' and 'mask' in fa['reason']


# --------------------------------------------------------------------------- config tools

def test_every_ready_config_tool_produces_registry_valid_candidates_for_every_model():
    produced = set()
    for tool in ready_config_tools():
        for model, points in baselines().items():
            for baseline in points:
                if not tool.applies_to(model):
                    with pytest.raises(ValueError, match='does not apply'):
                        tools.propose(tool.tool_id, baseline)
                    continue
                for item in tools.propose(tool.tool_id, baseline):
                    produced.add(tool.tool_id)
                    assert validate_config(item.backend) is _CAPABILITIES[model]
                    assert set(item.backend.options) <= set(_CAPABILITIES[model]['options'])
                    assert item.origin == 'search' and item.tool == tool.tool_id
                    assert item.stage == tool.primary_stage_for(model)
                    assert item.parent_id == baseline.candidate_id
                    assert item.candidate_id.startswith(f'{baseline.candidate_id}-')
                    assert item.hypothesis.startswith(f'{tool.tool_id}: ')
                    assert item.hypothesis.endswith(f'; exactness={tool.exactness}')
                    assert item.source_ref == baseline.source_ref and item.patch_ref == baseline.patch_ref
                    assert effective_backend(item.backend) != effective_backend(baseline.backend)
                    assert Candidate.from_dict(json.loads(json.dumps(item.to_dict()))) == item
    assert produced == {tool.tool_id for tool in ready_config_tools()}


def test_sampler_tools_match_search_enumerate_candidates():
    for model, points in baselines().items():
        if model == 'dm05':
            with pytest.raises(ValueError, match='fixed Euler10'):
                enumerate_candidates(points[0])
            continue
        for baseline in points:
            steps = (2, 3, 7)
            ours = tools.propose('sampler.euler_steps', baseline, steps=list(steps))
            theirs = [point.backend for point in enumerate_candidates(baseline, euler_steps=steps,
                                                                      epsilons=())]
            assert ours and all(item.backend in theirs for item in ours)
            assert [item.backend.num_steps for item in ours] == [
                backend.num_steps for backend in theirs
                if effective_backend(backend) != effective_backend(baseline.backend)]
            pf = tools.propose('sampler.probeflow', baseline, epsilons=[0.008, 0.002], max_steps=12)
            theirs = [point.backend for point in enumerate_candidates(baseline, euler_steps=(),
                                                                      epsilons=(0.002, 0.008), max_steps=12)]
            assert [item.backend for item in pf] == [backend for backend in theirs
                                                     if backend != baseline.backend]


def test_sampler_defaults_ids_and_dropping_the_baseline_value():
    fw = candidate('fw', 'fastwam', num_steps=14, options=FASTWAM_OPTIONS)
    assert [item.candidate_id for item in tools.propose('sampler.euler_steps', fw)] == [
        'fw-euler-10', 'fw-euler-12', 'fw-euler-13', 'fw-euler-16', 'fw-euler-18']
    pi = candidate('p', 'pi05', num_steps=7)
    assert [item.backend.num_steps for item in tools.propose('sampler.euler_steps', pi)] == [5, 6, 8, 9]
    # pi05 default Euler steps are 10: an explicit 10 is the baseline and is dropped.
    default = candidate('d', 'pi05')
    assert tools.propose('sampler.euler_steps', default, steps=[10, 4])[0].candidate_id == 'd-euler-4'
    pf = tools.propose('sampler.probeflow', pi)
    assert [item.candidate_id for item in pf] == [
        'p-pf-eps0.002-max7', 'p-pf-eps0.004-max7', 'p-pf-eps0.008-max7', 'p-pf-eps0.016-max7']
    assert all(item.backend.probeflow.max_steps == 7 and item.backend.num_steps is None for item in pf)
    assert tools.propose('sampler.probeflow', default)[0].backend.probeflow.max_steps == 10
    assert tools.propose('sampler.probeflow', fw.__class__(
        'f2', BackendConfig(model='fastwam', checkpoint='/w', options=FASTWAM_OPTIONS), 'frames')
    )[0].backend.probeflow.max_steps == 20
    same = candidate('s', 'pi05', sampler='probeflow', probeflow=ProbeFlowConfig(epsilon=0.004, max_steps=10))
    assert [item.backend.probeflow.epsilon for item in tools.propose('sampler.probeflow', same)] == [
        0.002, 0.008, 0.016]
    with pytest.raises(ValueError):
        tools.propose('sampler.euler_steps', pi, steps=[0])
    with pytest.raises(ValueError):
        tools.propose('sampler.probeflow', pi, max_steps=1)


def test_config_tools_are_deterministic():
    for tool in ready_config_tools():
        for model, points in baselines().items():
            if tool.applies_to(model):
                first = [item.to_dict() for item in tools.propose(tool.tool_id, points[0])]
                assert first == [item.to_dict() for item in tools.propose(tool.tool_id, points[0])]


def test_cuda_graph_tool_respects_defaults_and_fastwam_requirements():
    assert tools.propose('launch.cuda_graph', baselines()['pi05'][0]) == []
    eager = tools.propose('launch.cuda_graph', baselines()['pi05'][1])
    assert [item.candidate_id for item in eager] == ['pi05-pf-graph-on']
    assert eager[0].backend.options['use_cuda_graph'] is True
    native = tools.propose('launch.cuda_graph', baselines()['fastwam'][0])[0]
    assert dict(native.backend.options) == {**FASTWAM_OPTIONS, 'use_cuda_graph': True,
                                            'native_linear_backend': 'cached'}
    assert tools.propose('launch.cuda_graph', baselines()['fastwam'][1]) == []
    torch = tools.propose('launch.cuda_graph', baselines()['fastwam'][2])[0]
    assert torch.backend.backend == 'torch' and 'native_linear_backend' not in torch.backend.options


def test_fastwam_linear_backend_variants():
    base = tools.propose('gemm.fastwam_linear_backend', baselines()['fastwam'][0])
    assert [item.candidate_id for item in base] == ['fw-base-linear-torch', 'fw-base-linear-flashrt-cached']
    graph = tools.propose('gemm.fastwam_linear_backend', baselines()['fastwam'][1])
    assert [item.candidate_id for item in graph] == ['fw-graph-linear-torch', 'fw-graph-linear-flashrt-legacy']
    legacy = graph[1].backend
    assert legacy.options['native_linear_backend'] == 'legacy' and legacy.options['use_cuda_graph'] is False
    assert 'CUDA graph disabled' in graph[1].hypothesis
    torch = tools.propose('gemm.fastwam_linear_backend', baselines()['fastwam'][2])
    assert [item.backend.backend for item in torch] == ['flashrt', 'flashrt']
    with pytest.raises(ValueError, match='variants'):
        tools.propose('gemm.fastwam_linear_backend', baselines()['fastwam'][0], variants=['cuda'])


def test_pi05_and_hy_option_tools():
    pool = tools.propose('prefix.vision_pool', baselines()['pi05'][1])
    assert [item.candidate_id for item in pool] == ['pi05-pf-vpool-4']  # factor 2 is the baseline
    assert [item.backend.options['state_prompt_mode']
            for item in tools.propose('prefix.state_prompt_mode', baselines()['pi05'][0])] == ['fixed']
    assert tools.propose('prefix.state_prompt_mode', baselines()['pi05'][0], modes=['exact']) == []
    hy = tools.propose('precision.hy_numerical_profile', baselines()['hyvla'][1])
    assert [item.candidate_id for item in hy] == ['hy-eager-numprof-official_local_fp32']
    assert 'exactness=n/a' in hy[0].hypothesis


def test_propose_refuses_non_generating_tools_with_reason_and_adapter():
    pi05 = baselines()['pi05'][0]
    with pytest.raises(ValueError, match='blocked') as error:
        tools.propose('quant.fp8', pi05)
    assert 'SM>=89' in str(error.value)
    with pytest.raises(ValueError, match='tools.external.forgekernel'):
        tools.propose('kernel.forgekernel', pi05)
    with pytest.raises(ValueError, match='author.adapter'):
        tools.propose('baseline.official_impl', baselines()['fastwam'][0])
    with pytest.raises(ValueError, match='stages attribute'):
        tools.propose('profile.stage_intervals', pi05)
    with pytest.raises(ValueError, match='no parameters'):
        tools.propose('sampler.euler_steps', pi05, epsilons=[0.1])
    with pytest.raises(TypeError):
        tools.propose('sampler.euler_steps', pi05.to_dict())


# --------------------------------------------------------------------------- lever pricing

def test_lever_combinations_follow_the_factorial_plan():
    assert levers_module.combinations(['b', 'a']) == [('a',), ('b',), ('a', 'b')]
    assert levers_module.combinations(['c', 'a', 'b']) == [
        ('a',), ('b',), ('c',), ('a', 'b', 'c'), ('b', 'c'), ('a', 'c'), ('a', 'b')]
    assert levers_module.combinations(['x']) == [('x',)]


def test_lever_pricing_dedupes_validates_and_is_deterministic():
    fw = baselines()['fastwam'][0]
    levers = {'steps': {'num_steps': 6},
              'graph': {'options': {'use_cuda_graph': True, 'native_linear_backend': 'cached'}},
              'cached': {'options': {'native_linear_backend': 'cached'}}}
    points = tools.propose('analysis.lever_pricing', fw, levers=levers)
    reordered = tools.propose('analysis.lever_pricing', fw, levers=dict(reversed(list(levers.items()))))
    assert [item.to_dict() for item in points] == [item.to_dict() for item in reordered]
    ids = [item.candidate_id for item in points]
    # cached+graph == graph alone and cached+graph+steps == graph+steps: duplicates dropped (first kept).
    assert ids == ['fw-base-lv-cached', 'fw-base-lv-graph', 'fw-base-lv-steps',
                   'fw-base-lv-cached+graph+steps', 'fw-base-lv-cached+steps']
    encoded = [json.dumps(encode_backend(item.backend), sort_keys=True) for item in points]
    assert len(set(encoded)) == len(encoded)
    for item in points:
        validate_config(item.backend)
        assert item.tool == 'analysis.lever_pricing' and item.stage is None and item.origin == 'search'
        assert item.parent_id == 'fw-base' and 'exactness=n/a' in item.hypothesis
    staged = levers_module.price_levers(fw, {'steps': {'num_steps': 6}}, stage='denoise.schedule')
    assert [item.stage for item in staged] == ['denoise.schedule']


def test_lever_pricing_excludes_the_baseline_and_rejects_conflicts():
    pi = baselines()['pi05'][0]
    assert levers_module.price_levers(pi, {'same': {'num_steps': 10}}) == []
    with pytest.raises(ValueError, match="set 'num_steps' differently"):
        levers_module.price_levers(pi, {'a': {'num_steps': 5}, 'b': {'num_steps': 6}})
    with pytest.raises(ValueError, match="set option 'vision_pool_factor' differently"):
        levers_module.price_levers(pi, {'a': {'options': {'vision_pool_factor': 2}},
                                        'b': {'options': {'vision_pool_factor': 4}}})
    with pytest.raises(ValueError, match='unknown fields'):
        levers_module.price_levers(pi, {'a': {'checkpoint': '/other'}})
    with pytest.raises(ValueError, match='is invalid'):
        levers_module.price_levers(pi, {'a': {'options': {'numerical_profile': 'official_local_fp32'}}})
    adaptive = levers_module.price_levers(pi, {'pf': {'sampler': 'probeflow', 'num_steps': None,
                                                      'probeflow': {'epsilon': 0.004, 'max_steps': 8}}})
    assert adaptive[0].backend.probeflow == ProbeFlowConfig(epsilon=0.004, max_steps=8)
    with pytest.raises(ValueError, match='levers'):
        tools.propose('analysis.lever_pricing', pi)


# --------------------------------------------------------------------------- CLI

def write_baseline(tmp_path, baseline):
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps(baseline.to_dict()))
    return path


def test_cli_list_show_and_errors(capsys):
    assert tools_main(['list', '--model', 'pi05', '--hardware', 'cuda_sm80']) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {item['tool_id'] for item in listed} == {tool.tool_id for tool in tools.list_tools(
        model='pi05', hardware='cuda_sm80')}
    assert tools_main(['list', '--all', '--markdown']) == 0
    table = capsys.readouterr().out
    assert table.startswith('| id | lane |') and '| quant.fp8 | patch |' in table
    assert tools_main(['show', 'kernel.forgekernel']) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown['upstream']['commit'] == PINNED['kernel.forgekernel'] and shown['setup'] and shown['risks']
    assert tools_main(['show', 'nope']) == 2
    assert 'Unknown tool' in capsys.readouterr().err


def test_cli_coverage_shows_all_stages_and_unavailable_reasons(capsys):
    assert tools_main(['coverage', '--model', 'pi05', '--hardware', 'cuda_sm80',
                       '--markdown']) == 0
    table = capsys.readouterr().out
    assert table.startswith('| stage | ready | unavailable / prerequisites |')
    assert len(table.strip().splitlines()) == len(STAGE_IDS) + 2
    assert 'quant.fp8 (patch): blocked:' in table
    assert 'attention.flash_attention (port): needs_implementation:' in table
    assert tools_main(['show', 'compile.fastwam_inductor', '--model', 'fastwam']) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown['tool_id'] == 'compile.inductor' and shown['requested_alias'] == 'compile.fastwam_inductor'
    assert shown['selected_model']['recipe'] == get_spec('fastwam')['tools']['inductor']


def test_cli_external_spec_dir_is_carried_through_listing_and_coverage(tmp_path, capsys):
    external_spec(tmp_path)
    assert tools_main(['list', '--model', 'test_flow', '--spec-dir', str(tmp_path)]) == 0
    listed = json.loads(capsys.readouterr().out)
    sampler = next(record for record in listed if record['tool_id'] == 'sampler.euler_steps')
    assert 'test_flow' in sampler['matching_models'] and sampler['model_predicate']
    assert tools_main(['coverage', '--model', 'test_flow', '--hardware', 'cuda_sm80',
                       '--spec-dir', str(tmp_path)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row['stage'] for row in rows} == set(STAGE_IDS)


def test_cli_propose_feeds_controller_init_proposals(tmp_path, capsys):
    baseline = baselines()['pi05'][0]
    baseline_path = write_baseline(tmp_path, baseline)
    out = tmp_path / 'proposals.json'
    assert tools_main(['propose', '--baseline', str(baseline_path), '--tool', 'sampler.probeflow',
                       '--param', 'epsilons=[0.004, 0.008]', '--param', 'max_steps=10',
                       '--out', str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['count'] == 2 and summary['out'] == str(out)
    records = json.loads(out.read_text())
    assert [Candidate.from_dict(record).candidate_id for record in records] == summary['candidate_ids']
    controller_main(['init', '--root', str(tmp_path / 'campaign'), '--comparison-id', 'r1',
                     '--baseline', str(baseline_path), '--workload', '/remote/inputs.json',
                     '--output-root', '/remote/results', '--proposals', str(out)])
    plan = json.loads((tmp_path / 'campaign' / 'plan.json').read_text())
    assert [entry['role'] for entry in plan['entries']] == ['reference', 'candidate', 'candidate', 'bookend']
    assert [entry['candidate'] for entry in plan['entries'][1:3]] == records
    assert all(entry['candidate']['tool'] == 'sampler.probeflow' for entry in plan['entries'][1:3])
    assert tools_main(['propose', '--baseline', str(baseline_path), '--tool', 'sampler.probeflow',
                       '--param', 'max_steps=ten', '--out', str(tmp_path / 'x.json')]) == 2
    assert 'not JSON' in capsys.readouterr().err
    assert tools_main(['propose', '--baseline', str(baseline_path), '--tool', 'quant.fp8',
                       '--out', str(tmp_path / 'y.json')]) == 2
    assert not (tmp_path / 'y.json').exists()


def test_cli_levers(tmp_path, capsys):
    baseline_path = write_baseline(tmp_path, baselines()['fastwam'][0])
    out = tmp_path / 'levers.json'
    assert tools_main(['levers', '--baseline', str(baseline_path), '--lever', 'steps={"num_steps": 6}',
                       '--lever', 'torch={"backend": "torch"}', '--stage', 'kernel.gemm',
                       '--out', str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['candidate_ids'] == ['fw-base-lv-steps', 'fw-base-lv-torch', 'fw-base-lv-steps+torch']
    assert all(Candidate.from_dict(record).stage == 'kernel.gemm' for record in json.loads(out.read_text()))


# --------------------------------------------------------------------------- patches

def snapshot_baseline(base):
    return Candidate('base', BackendConfig(model='pi05', checkpoint='/weights', num_steps=2), 'frames',
                     source_ref=base)


@needs_git
def test_patch_from_committed_staged_and_unstaged_changes_runs_as_snapshot_campaign(tmp_path):
    repo = make_repo(tmp_path / 'repo')
    store = tmp_path / 'store'
    base = head(repo)
    (repo / 'flash_rt' / 'model.py').write_text('STEPS = 4\n')
    git(repo, 'commit', '-q', '-am', 'agent change')
    (repo / 'csrc' / 'k.cu').write_text('__global__ void k() { /* staged */ }\n')
    git(repo, 'add', 'csrc/k.cu')
    (repo / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\n')  # unstaged
    index_before = git(repo, 'ls-files', '--stage').stdout
    proposal = patches.proposal_from_worktree(
        repo, 'main~1', store=store, tool_id='kernel.forgekernel', stage_id='denoise.velocity',
        hypothesis='fewer steps in the fixture model', options={'use_cuda_graph': False},
        provenance={'agent': 'test'})
    assert proposal.base_commit == base and proposal.patch_ref.startswith('sha256:')
    assert sorted(proposal.provenance['paths']) == ['CMakeLists.txt', 'csrc/k.cu', 'flash_rt/model.py']
    assert proposal.provenance['agent'] == 'test' and proposal.provenance['head_commit'] == head(repo)
    assert git(repo, 'ls-files', '--stage').stdout == index_before
    assert snap.patch_file(store, proposal.patch_ref).is_file()
    assert PatchProposal.from_dict(proposal.to_dict()) == proposal

    baseline = snapshot_baseline(base)
    patched = patches.candidate_from_proposal(baseline, proposal)
    assert patched.source_ref == base and patched.patch_ref == proposal.patch_ref
    assert patched.origin == 'agent' and patched.tool == 'kernel.forgekernel'
    assert patched.stage == 'denoise.velocity' and patched.parent_id == 'base'
    assert patched.candidate_id == f'base-patch-{proposal.patch_ref[7:19]}'
    assert patched.backend.options['use_cuda_graph'] is False
    assert patched.hypothesis == 'fewer steps in the fixture model'

    for patch_ref in (None, proposal.patch_ref):
        manifest = snap.materialize(repo, store, base, patch_ref)
        snap.adopt_native(store, manifest['snapshot_id'], None, fingerprint=None, none_expected=True)
    patched_dir = Path(snap.load_manifest(store, snap.snapshot_id(base, proposal.patch_ref))['source_dir'])
    assert (patched_dir / 'flash_rt' / 'model.py').read_text() == 'STEPS = 4\n'
    assert (patched_dir / 'CMakeLists.txt').read_text() == 'cmake_minimum_required(VERSION 3.20)\n'
    plan = initialize_campaign(tmp_path / 'campaign', comparison_id='r1', baseline=baseline,
                               candidates=[patched], workload_path=str(tmp_path / 'workload.json'),
                               output_root=str(tmp_path / 'results'), snapshot_store=str(store))
    assert plan['schema_version'] == 3
    assert [entry['candidate']['candidate_id'] for entry in plan['entries']] == ['base', patched.candidate_id,
                                                                                'base']
    assert plan['entries'][1]['snapshot_id'] == snap.snapshot_id(base, proposal.patch_ref)


@needs_git
def test_patch_refuses_untracked_unless_included_and_keeps_real_index(tmp_path):
    repo = make_repo(tmp_path / 'repo')
    store = tmp_path / 'store'
    base = head(repo)
    (repo / 'flash_rt' / 'new_kernel.py').write_text('X = 1\n')
    with pytest.raises(ValueError, match='untracked'):
        patches.proposal_from_worktree(repo, base, store=store, tool_id='kernel.kernelagent',
                                       stage_id='kernel.gemm', hypothesis='h')
    assert not (store / 'patches').exists()
    index_before = git(repo, 'ls-files', '--stage').stdout
    proposal = patches.proposal_from_worktree(repo, base, store=store, tool_id='kernel.kernelagent',
                                              stage_id='kernel.gemm', hypothesis='h', include_untracked=True)
    assert proposal.provenance['untracked_included'] == ['flash_rt/new_kernel.py']
    assert proposal.provenance['paths'] == ['flash_rt/new_kernel.py']
    assert git(repo, 'ls-files', '--stage').stdout == index_before
    assert git(repo, 'status', '--porcelain').stdout.decode().strip() == '?? flash_rt/new_kernel.py'
    manifest = snap.materialize(repo, store, base, proposal.patch_ref)
    assert (Path(manifest['source_dir']) / 'flash_rt' / 'new_kernel.py').read_text() == 'X = 1\n'
    # Ignored files never enter a patch.
    (repo / 'flash_rt' / 'ext.so').write_bytes(b'\0')
    os.remove(repo / 'flash_rt' / 'new_kernel.py')
    with pytest.raises(ValueError, match='no tracked change'):
        patches.proposal_from_worktree(repo, base, store=store, tool_id='kernel.kernelagent',
                                       stage_id='kernel.gemm', hypothesis='h')


@needs_git
def test_patch_rejects_harness_changes_bad_stage_and_mismatched_baseline(tmp_path):
    repo = make_repo(tmp_path / 'repo')
    store = tmp_path / 'store'
    base = head(repo)
    (repo / 'flash_rt' / 'vla' / 'autotune' / 'worker.py').write_text('# tampered\n')
    with pytest.raises(snap.SnapshotError, match='pinned'):
        patches.proposal_from_worktree(repo, base, store=store, tool_id='t', stage_id='kernel.gemm',
                                       hypothesis='h')
    assert not (store / 'patches').exists()
    git(repo, 'checkout', '--', 'flash_rt/vla/autotune/worker.py')
    with pytest.raises(ValueError, match='stage_id'):
        patches.proposal_from_worktree(repo, base, store=store, tool_id='t', stage_id='gpu', hypothesis='h')
    (repo / 'flash_rt' / 'model.py').write_text('STEPS = 3\n')
    proposal = patches.proposal_from_worktree(repo, base, store=store, tool_id='t', stage_id='kernel.gemm',
                                              hypothesis='h', options={'brand_new_option': 1})
    with pytest.raises(ValueError, match='Unsupported pi05 options'):
        patches.candidate_from_proposal(snapshot_baseline(base), proposal)
    unchecked = patches.candidate_from_proposal(snapshot_baseline(base), proposal, 'custom', validate=False)
    assert unchecked.candidate_id == 'custom' and unchecked.backend.options['brand_new_option'] == 1
    with pytest.raises(ValueError, match='source_ref'):
        patches.candidate_from_proposal(Candidate('b', BackendConfig(model='pi05', checkpoint='/w'), 'frames'),
                                        proposal, validate=False)
    with pytest.raises(ValueError, match='stacked'):
        patches.candidate_from_proposal(Candidate('b', BackendConfig(model='pi05', checkpoint='/w'), 'frames',
                                                  source_ref=base, patch_ref=proposal.patch_ref),
                                        proposal, validate=False)


def test_patch_candidate_records_the_requested_origin():
    base = 'a' * 40
    proposal = PatchProposal(base_commit=base, patch_ref='sha256:' + 'b' * 64, tool_id='kernel.forgekernel',
                             stage_id='denoise.velocity', hypothesis='h', options={}, provenance={})
    assert patches.candidate_from_proposal(snapshot_baseline(base), proposal).origin == 'agent'
    for origin in ('human', 'search', 'agent'):
        assert patches.candidate_from_proposal(snapshot_baseline(base), proposal, origin=origin).origin == origin
    for bad in ('user', 'Human', None, ['human']):
        with pytest.raises(ValueError, match='origin must be one of'):
            patches.candidate_from_proposal(snapshot_baseline(base), proposal, origin=bad)


def test_pi05_state_prompt_mode_env_override_is_refused_when_it_differs(monkeypatch):
    exact = BackendConfig(model='pi05', checkpoint='/w')
    fixed = BackendConfig(model='pi05', checkpoint='/w', options={'state_prompt_mode': 'fixed'})
    validate_config(exact)
    validate_config(fixed)
    monkeypatch.setenv('FLASHRT_PI05_STATE_PROMPT_MODE', 'fixed')
    with pytest.raises(ValueError, match='state_prompt_mode=exact conflicts with '
                                         'FLASHRT_PI05_STATE_PROMPT_MODE=fixed'):
        validate_config(exact)
    validate_config(fixed)
    monkeypatch.setenv('FLASHRT_PI05_STATE_PROMPT_MODE', 'exact')
    validate_config(exact)
    with pytest.raises(ValueError, match='state_prompt_mode=fixed conflicts with'):
        validate_config(fixed)
    # pi0 has no such override.
    validate_config(BackendConfig(model='pi0', checkpoint='/w'))
