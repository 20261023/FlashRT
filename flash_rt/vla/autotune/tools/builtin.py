"""In-repo tools: deterministic config tools plus explicit blocked/unimplemented entries.

Config tools vary only BackendConfig fields the unified backend already exposes
(``flash_rt.vla.registry._CAPABILITIES``); every candidate passes
``registry.validate_config``. They keep the baseline's source_ref/patch_ref
(same code tree, never a worktree). Non-ready entries record why, with
file:line evidence from this tree, so a planner never silently drops them.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

from flash_rt.sampling import ProbeFlowConfig
from flash_rt.vla.model_spec import get_spec
from flash_rt.vla.registry import validate_config

from ..schema import Candidate, decode_backend, encode_backend
from ..search import enumerate_candidates
from ..stages import STAGE_IDS
from .spec import ToolSpec


BUILTIN = 'flash_rt.vla.autotune.tools.builtin'

PROBEFLOW_EPSILONS = (0.002, 0.004, 0.008, 0.016)
FASTWAM_LINEAR_VARIANTS = ('torch', 'flashrt-cached', 'flashrt-legacy')

# Skill names the tools point to (procedures live in skills, never winners).
SKILLS = {
    'config': 'vla-config-sweep',
    'profile': 'vla-stage-profile',
    'kernel': 'vla-kernel-candidate',
    'gate': 'vla-operator-gate',
    'patch': 'vla-patch-candidate',
    'map': 'vla-model-impl-map',
    'port': 'vla-port-workflow',
}


# --------------------------------------------------------------------------- helpers

def _model_spec(config):
    return get_spec(config.model, spec_dir=getattr(config, 'spec_dir', None))


def _defaults(config):
    spec = _model_spec(config)
    defaults = dict(spec['head'].get('defaults', {}))
    defaults.update(spec.get('tools', {}).get('defaults', {}))
    options = {}
    for name, schema in spec['capabilities']['options'].items():
        if 'default' in schema:
            options[name] = schema['default']
        if config.backend in schema.get('default_by_backend', {}):
            options[name] = schema['default_by_backend'][config.backend]
    options.update(defaults.get('options', {}))
    defaults['options'] = options
    return defaults


def _has_option(config, option):
    return option in _model_spec(config)['capabilities']['options']


def effective_backend(config) -> dict:
    """Encoded backend with frontend defaults filled in, for equivalence checks only."""
    record = encode_backend(config)
    defaults = _defaults(config)
    options = {**defaults['options'], **record['options']}
    if record['sampler'] == 'euler' and record['num_steps'] is None and defaults.get('num_steps'):
        record['num_steps'] = defaults['num_steps']
    record['options'] = options
    return record


def check_baseline(tool, baseline):
    if not isinstance(baseline, Candidate):
        raise TypeError('baseline must be a Candidate')
    matches, reason = tool.match_model(baseline.backend.model,
                                      spec_dir=getattr(baseline.backend, 'spec_dir', None))
    if not matches:
        raise ValueError(f'{tool.tool_id} does not apply to model {baseline.backend.model!r} '
                         f'({reason})')


_PRIMARY = object()


def make_candidate(baseline, tool, short, value, backend_record, change, *, stage=_PRIMARY):
    """One validated config candidate derived from ``baseline``.

    ``stage`` defaults to the tool's primary stage; pass None for no stage.
    """
    backend = decode_backend(backend_record)  # Detaches nested options.
    allowed = set(_model_spec(backend)['capabilities']['options'])
    unknown = set(backend.options) - allowed
    if unknown:
        raise ValueError(f'{tool.tool_id} would emit unsupported {backend.model} options {sorted(unknown)}')
    validate_config(backend)
    return Candidate(
        candidate_id=f'{baseline.candidate_id}-{short}-{value}', backend=backend,
        workload_id=baseline.workload_id, parent_id=baseline.candidate_id,
        hypothesis=f'{tool.tool_id}: {change}; exactness={tool.exactness}',
        origin='search', patch_ref=baseline.patch_ref, source_ref=baseline.source_ref,
        tool=tool.tool_id,
        stage=(tool.primary_stage_for(backend.model, spec_dir=getattr(backend, 'spec_dir', None))
               if stage is _PRIMARY else stage))


def finalize(baseline, candidates, *, effective=True):
    """Drop points equivalent to the baseline and duplicate backends (first kept)."""
    reference = (effective_backend if effective else encode_backend)(baseline.backend)
    seen, kept = [], []
    for candidate in candidates:
        encoded = encode_backend(candidate.backend)
        if (effective_backend if effective else encode_backend)(candidate.backend) == reference:
            continue
        if encoded in seen:
            continue
        seen.append(encoded)
        kept.append(candidate)
    ids = [candidate.candidate_id for candidate in kept]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate candidate ids: {ids}')
    return kept


def _positive_ints(values, name):
    if isinstance(values, (str, bytes)) or not hasattr(values, '__iter__'):
        raise ValueError(f'{name} must be a sequence of positive integers')
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must contain positive integers, got {value!r}')
        result.append(int(value))
    if not result:
        raise ValueError(f'{name} must not be empty')
    return result


def _choices(values, allowed, name):
    if isinstance(values, (str, bytes)) or not hasattr(values, '__iter__'):
        raise ValueError(f'{name} must be a sequence drawn from {list(allowed)}')
    values = list(values)
    if not values or any(value not in allowed for value in values):
        raise ValueError(f'{name} must be a nonempty sequence drawn from {list(allowed)}, got {values!r}')
    return list(dict.fromkeys(values))


def _with(baseline, **changes):
    record = encode_backend(baseline.backend)
    options = changes.pop('options', None)
    record.update(changes)
    if options is not None:
        record['options'] = options
    return record


# --------------------------------------------------------------------------- config tools

def _propose_euler_steps(tool, baseline, *, steps=None):
    check_baseline(tool, baseline)
    if steps is None:
        steps = _defaults(baseline.backend).get('euler_steps')
        if steps is None:
            count = _defaults(baseline.backend).get('num_steps') or baseline.backend.num_steps
            if count is None:
                raise ValueError('Euler step defaults are unknown; pass explicit steps')
            steps = [max(1, round(count * fraction)) for fraction in (0.5, 0.6, 2 / 3, 0.7, 0.8, 0.9)]
    values = sorted(set(_positive_ints(steps, 'steps')))
    # Same backend content as search.enumerate_candidates for these values.
    points = enumerate_candidates(baseline, euler_steps=values, epsilons=())
    candidates = [make_candidate(baseline, tool, 'euler', count, encode_backend(point.backend),
                                 f'fixed Euler with {count} action-expert evaluations (NFE={count})')
                  for count, point in zip(values, points)]
    return finalize(baseline, candidates)


def _default_max_steps(baseline):
    config = baseline.backend
    if config.sampler == 'probeflow':
        return config.probeflow.max_steps
    if config.num_steps is not None:
        return int(config.num_steps)
    defaults = _defaults(config)
    value = defaults.get('probeflow_max_steps') or defaults.get('num_steps')
    if value is None:
        raise ValueError('ProbeFlow max_steps is unknown; pass explicit max_steps')
    return value


def _propose_probeflow(tool, baseline, *, epsilons=PROBEFLOW_EPSILONS, max_steps=None):
    check_baseline(tool, baseline)
    if isinstance(epsilons, (str, bytes)) or not hasattr(epsilons, '__iter__'):
        raise ValueError('epsilons must be a sequence of positive finite numbers')
    values = []
    for value in epsilons:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'epsilons must contain positive finite numbers, got {value!r}')
        values.append(float(value))
    if not values:
        raise ValueError('epsilons must not be empty')
    if max_steps is None:
        max_steps = _default_max_steps(baseline)
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral):
        raise ValueError('max_steps must be an integer')
    max_steps = int(max_steps)
    ProbeFlowConfig(max_steps=max_steps)  # Range check with the sampler's own message.
    values = sorted(set(values))
    points = enumerate_candidates(baseline, euler_steps=(), epsilons=values, max_steps=max_steps)
    candidates = [make_candidate(
        baseline, tool, 'pf', f'eps{epsilon:g}-max{max_steps}', encode_backend(point.backend),
        f'ProbeFlow adaptive schedule epsilon={epsilon:g}, max_steps={max_steps} '
        '(actual NFE is measured per request)') for epsilon, point in zip(values, points)]
    return finalize(baseline, candidates)


def _propose_cuda_graph(tool, baseline):
    check_baseline(tool, baseline)
    config = baseline.backend
    if effective_backend(config)['options'].get('use_cuda_graph', False):
        return []
    options = {**config.options, 'use_cuda_graph': True}
    change = 'capture the request in CUDA graphs (use_cuda_graph=True)'
    if _has_option(config, 'native_linear_backend') and config.backend == 'flashrt':
        # registry: FastWAM native CUDA graph requires native_linear_backend="cached".
        options['native_linear_backend'] = 'cached'
        change += ' with native_linear_backend=cached (required by the FastWAM native graph)'
    if _has_option(config, 'compile_action_infer') and options.get('compile_action_infer'):
        # registry: compile_action_infer requires use_cuda_graph=False.
        options['compile_action_infer'] = False
        change += '; replaces torch.compile (compile_action_infer=False, mutually exclusive with graphs)'
    return finalize(baseline, [make_candidate(baseline, tool, 'graph', 'on',
                                              _with(baseline, options=options), change)])


def _propose_fastwam_linear(tool, baseline, *, variants=FASTWAM_LINEAR_VARIANTS):
    check_baseline(tool, baseline)
    variants = _choices(variants, FASTWAM_LINEAR_VARIANTS, 'variants')
    options = dict(baseline.backend.options)
    candidates = []
    for variant in variants:
        changed = dict(options)
        if variant != 'torch' and changed.get('compile_action_infer'):
            # registry: compile_action_infer requires backend="torch".
            changed['compile_action_infer'] = False
        if variant == 'torch':
            changed.pop('native_linear_backend', None)
            record, change = _with(baseline, backend='torch', options=changed), \
                'upstream Torch linears (backend=torch; no native FlashRT GEMM)'
        elif variant == 'flashrt-cached':
            changed['native_linear_backend'] = 'cached'
            record, change = _with(baseline, backend='flashrt', options=changed), \
                'native FlashRT linears with the cached runner (native_linear_backend=cached)'
        else:
            changed['native_linear_backend'] = 'legacy'
            change = 'native FlashRT linears with the legacy runner (native_linear_backend=legacy)'
            if changed.get('use_cuda_graph'):
                changed['use_cuda_graph'] = False
                change += '; CUDA graph disabled (legacy cannot be captured)'
            record = _with(baseline, backend='flashrt', options=changed)
        candidates.append(make_candidate(baseline, tool, 'linear', variant, record, change))
    return finalize(baseline, candidates)


def _propose_inductor(tool, baseline):
    """Apply only a ModelSpec recipe for an already implemented product option."""
    check_baseline(tool, baseline)
    config = baseline.backend
    recipe = _model_spec(config).get('tools', {}).get('inductor')
    if not isinstance(recipe, dict) or not recipe.get('options'):
        raise ValueError('compile.inductor needs an implemented tools.inductor recipe in ModelSpec')
    for field in ('frameworks', 'samplers', 'hardware'):
        value = getattr(config, 'framework' if field == 'frameworks' else
                        'sampler' if field == 'samplers' else 'hardware')
        allowed = recipe.get(field, [])
        if allowed and value not in allowed and not (field == 'hardware' and value == 'auto'):
            raise ValueError(f'compile.inductor requires {field} in {allowed}, got {value!r}')
    options = {**config.options, **recipe['options']}
    for name in recipe.get('remove_options', []):
        options.pop(name, None)
    return finalize(baseline, [make_candidate(
        baseline, tool, 'compile', 'on',
        _with(baseline, backend=recipe.get('backend', config.backend), options=options),
        recipe.get('reason', 'enable the implemented ModelSpec Inductor recipe'))])


def _enabled_option(tool, baseline, option, *, sm80=False):
    changes = {'options': {**baseline.backend.options, option: True}}
    if sm80:
        changes['hardware'] = 'cuda_sm80'
    return finalize(baseline, [make_candidate(
        baseline, tool, option, 'on', _with(baseline, **changes), f'enable {option}')])


def _propose_hy_kv(tool, baseline):
    return _enabled_option(tool, baseline, 'fused_rope_qknorm_kvwrite', sm80=True)


def _propose_fastwam_text_kv(tool, baseline):
    return _enabled_option(tool, baseline, 'cache_text_crossattn_kv')


def _propose_pi05_int8(tool, baseline, *, modes=('encoder', 'encoder_decoder')):
    check_baseline(tool, baseline)
    modes = _choices(modes, ('encoder', 'encoder_decoder'), 'modes')
    candidates = []
    for mode in modes:
        record = _with(baseline, hardware='cuda_sm80', precision='auto',
                       options={**baseline.backend.options, 'int8_mode': mode})
        candidates.append(make_candidate(baseline, tool, 'int8', mode, record,
                                         f'dynamic W8A8 {mode} projections with BF16 output'))
    return finalize(baseline, candidates)


def _propose_vision_pool(tool, baseline, *, factors=(2, 4)):
    check_baseline(tool, baseline)
    factors = _choices(factors, (1, 2, 4), 'factors')
    return finalize(baseline, [make_candidate(
        baseline, tool, 'vpool', factor,
        _with(baseline, options={**baseline.backend.options, 'vision_pool_factor': factor}),
        f'average-pool vision tokens by {factor} before the prefix (fewer prefix rows; ViT work unchanged)')
        for factor in factors])


def _propose_state_prompt_mode(tool, baseline, *, modes=('fixed',)):
    check_baseline(tool, baseline)
    modes = _choices(modes, ('exact', 'fixed'), 'modes')
    return finalize(baseline, [make_candidate(
        baseline, tool, 'spm', mode,
        _with(baseline, options={**baseline.backend.options, 'state_prompt_mode': mode}),
        f'state_prompt_mode={mode} (fixed: one max-length captured pipeline with masked padding, '
        'no per-length re-capture; only affects prompts that embed state)')
        for mode in modes])


def _propose_hy_profile(tool, baseline, *, profiles=('official_local_fp32',)):
    check_baseline(tool, baseline)
    profiles = _choices(profiles, ('semantic_bf16', 'official_local_fp32'), 'profiles')
    return finalize(baseline, [make_candidate(
        baseline, tool, 'numprof', profile,
        _with(baseline, options={**baseline.backend.options, 'numerical_profile': profile}),
        f'numerical_profile={profile} (alignment with the official local-fp32 numerics, not a speed knob)')
        for profile in profiles])


def _propose_levers(tool, baseline, *, levers=None, stage=None):
    from .levers import price_levers
    check_baseline(tool, baseline)
    if levers is None:
        raise ValueError('analysis.lever_pricing requires levers={name: {field: value}}')
    return price_levers(baseline, levers, stage=stage)


def _bind(function):
    """Bind the ToolSpec into its propose callable after construction."""
    holder = {}

    def propose(baseline, **params):
        return function(holder['tool'], baseline, **params)
    propose.__name__ = function.__name__.lstrip('_')
    propose.__doc__ = function.__doc__
    return holder, propose


def _tool(function=None, **fields):
    fields.setdefault('models', ('*',))
    if function is None:
        return ToolSpec(**fields)
    holder, propose = _bind(function)
    tool = ToolSpec(propose=propose, adapter=BUILTIN, **fields)
    holder['tool'] = tool
    return tool


# --------------------------------------------------------------------------- registry

_NO_QUANT = ('pi-family adapter bindings fix use_fp8=False; '
             'HyVLAFlowFrontend requires its FP8/FP4 flags to be false')


def _option(name):
    return {'path': 'capabilities.options', 'contains': name}


def _feature(name):
    return {'path': f'features.{name}', 'equals': True}


def _sampler(name, *, variable=False):
    conditions = [{'path': 'head.type', 'equals': 'flow_matching'},
                  {'path': 'capabilities.samplers', 'contains': name}]
    if variable:
        conditions.append(_feature('variable_steps'))
    return {'all': conditions}


_ATTENTION = {'all': [{'path': 'attention', 'exists': True},
                      {'not': {'path': 'attention', 'equals': []}}]}


def builtin_tools() -> tuple:
    """Fresh ToolSpec objects for every in-repo tool, in registry order."""
    config_skills = (SKILLS['config'],)
    return (
        _tool(_propose_euler_steps, tool_id='sampler.euler_steps', title='Fixed Euler step count',
              lane='config', stages=('denoise.schedule',), model_predicate=_sampler('euler', variable=True),
              exactness='approximate',
              status='ready', skills=config_skills,
              parameters={'steps': None}),
        _tool(_propose_probeflow, tool_id='sampler.probeflow', title='ProbeFlow adaptive schedule',
              lane='config', stages=('denoise.schedule', 'denoise.routing'),
              model_predicate=_sampler('probeflow'),
              exactness='approximate', status='ready', skills=config_skills,
              parameters={'epsilons': list(PROBEFLOW_EPSILONS), 'max_steps': None}),
        _tool(_propose_cuda_graph, tool_id='launch.cuda_graph', title='CUDA graph capture',
              lane='config', stages=('host.launch', 'denoise.velocity'),
              model_predicate=_option('use_cuda_graph'), exactness='exact',
              status='ready', skills=config_skills, parameters={},
              reason='No candidate when the effective baseline already captures graphs '
                     '(frontend defaults are declared in ModelSpec).'),
        _tool(_propose_inductor, tool_id='compile.inductor',
              title='Inductor through an implemented model-spec compiler recipe', lane='config',
              stages=('host.launch', 'denoise.velocity', 'prefix.prefill', 'vision.encode', 'kernel.gemm'),
              model_predicate={'all': [{'path': 'tools.inductor.options', 'exists': True},
                                       {'not': {'path': 'tools.inductor.options', 'equals': {}}},
                                       {'path': 'tools.inductor.stages', 'exists': True},
                                       {'not': {'path': 'tools.inductor.stages', 'equals': []}}]},
              frameworks=('torch',), requirements_path='tools.inductor',
              exactness='approximate', status='ready', skills=config_skills,
              aliases=('compile.fastwam_inductor', 'compile.dm05_inductor'),
              reason='Only applies a declared, already implemented BackendConfig recipe; never wraps '
                     'or compiles model code in the measurement harness. Record cold compilation and '
                     'break-even calls; the model-level quality gate covers changed rounding.'),
        _tool(_propose_fastwam_linear, tool_id='gemm.fastwam_linear_backend',
              title='FastWAM linear backend (torch / native cached / native legacy)',
              lane='config', stages=('kernel.gemm', 'host.launch', 'denoise.velocity'),
              model_predicate=_option('native_linear_backend'), frameworks=('torch',),
              exactness='exact', status='ready', skills=config_skills,
              parameters={'variants': list(FASTWAM_LINEAR_VARIANTS)}),
        _tool(_propose_vision_pool, tool_id='prefix.vision_pool', title='pi0.5 vision token pooling',
              lane='config', stages=('prefix.prefill', 'vision.project'),
              model_predicate=_option('vision_pool_factor'), frameworks=('torch',),
              exactness='approximate', status='ready', skills=config_skills,
              parameters={'factors': [2, 4]}),
        _tool(_propose_state_prompt_mode, tool_id='prefix.state_prompt_mode',
              title='pi0.5 fixed-shape state prompt', lane='config',
              stages=('host.launch', 'prefix.prefill'), model_predicate=_option('state_prompt_mode'),
              frameworks=('torch',), exactness='exact',
              status='ready', skills=config_skills, parameters={'modes': ['fixed']}),
        _tool(_propose_hy_profile, tool_id='precision.hy_numerical_profile',
              title='Hy numerical profile (alignment, not speed)', lane='config',
              stages=('precision.weights',), model_predicate=_option('numerical_profile'),
              frameworks=('torch',), exactness='n/a', status='ready',
              skills=config_skills, parameters={'profiles': ['official_local_fp32']}),
        _tool(tool_id='quant.fp8', title='FP8 weights/activations', lane='patch',
              stages=('precision.weights',), model_predicate=_feature('fp8_weights'),
              exactness='approximate', status='blocked', min_capability=(8, 9),
              skills=(SKILLS['patch'], SKILLS['gate']),
              reason='Needs SM>=89 FP8 tensor cores (A800/sm_80 has none; the RTX frontend '
                     'falls back to BF16) and the unified backend forbids quantized precision: '
                     + _NO_QUANT + '.'),
        _tool(tool_id='quant.nvfp4', title='NVFP4 / W4A8 weights', lane='patch',
              stages=('precision.weights',), model_predicate=_feature('nvfp4_weights'), exactness='approximate',
              status='blocked', min_capability=(12, 0), skills=(SKILLS['patch'], SKILLS['gate']),
              reason='NVFP4 kernels are built only for sm_120a/121a (CMakeLists.txt); pi-family '
                     'adapter bindings pin internal prefix/decoder selector labels to fp8 while '
                     'use_fp8=False, and validation rejects conflicting FLASHRT_PI05_*_PRECISION '
                     'overrides. Hy use_fp4 must be False (hyvla_probeflow.py).'),
        _tool(_propose_pi05_int8, tool_id='quant.int8_pi05', title='pi0.5 INT8 CUTLASS GEMMs', lane='config',
              stages=('precision.weights', 'kernel.gemm'),
              model_predicate={'all': [_option('int8_mode'), _sampler('euler')]},
              frameworks=('torch',), samplers=('euler',), exactness='approximate',
              status='needs_setup', hardware_only=('cuda_sm80',), skills=config_skills,
              parameters={'modes': ['encoder', 'encoder_decoder']},
              reason='Build flash_rt_kernels with -DGPU_ARCH=80 -DENABLE_SM80_INT8_CUTLASS=ON. '
                     'Torch Euler, dynamic per-row activation scales. Unset ambient precision overrides.'),
        _tool(_propose_hy_kv, tool_id='fusion.hy_rope_qknorm_kvwrite', title='Hy fused RoPE + QK-norm + KV-write',
              lane='config', stages=('kernel.norm_elementwise', 'prefix.prefill', 'denoise.velocity'),
              model_predicate=_option('fused_rope_qknorm_kvwrite'), frameworks=('torch',),
              exactness='approximate', status='needs_setup',
              hardware_only=('cuda_sm80',), skills=config_skills,
              reason='Build flash_rt_hyvla_sm80 with -DGPU_ARCH=80 -DFLASHRT_ENABLE_HYVLA=ON. '
                     'One launch computes RoPE, QK norm and the layer KV window per prefix/expert call.'),
        _tool(_propose_fastwam_text_kv, tool_id='cache.fastwam_text_crossattn_kv',
              title='FastWAM per-request text cross-attention K/V', lane='config',
              stages=('denoise.velocity',), model_predicate=_option('cache_text_crossattn_kv'),
              frameworks=('torch',), exactness='exact', status='ready',
              skills=config_skills,
              reason='Prepare text embedding and action cross-attention K/V once after appending the '
                     'current proprio. Each request refreshes the cache, including captured graph inputs.'),
        _tool(tool_id='cache.pi05_cross_frame_kv', title='pi0.5 cross-frame prefix KV reuse',
              lane='port', stages=('prefix.prefill', 'vision.encode'),
              model_predicate=_feature('cross_frame_kv'),
              exactness='approximate', status='blocked', skills=(SKILLS['map'], SKILLS['port']),
              reason='Semantic change, not a speed knob: cache_frames>1 reuses stale encoder K/V '
                     '(pi05_rtx.py:759-765,2279-2303); it is not a registry option, ProbeFlow refuses '
                     'it (pi05_rtx.py:538-541) and it needs a multi-frame workload.'),
        _tool(tool_id='throughput.pi05_batch', title='pi0.5 batched inference', lane='port',
              stages=('prefix.prefill', 'denoise.velocity'), model_predicate=_feature('batched_inference'),
              exactness='approximate',
              status='blocked', skills=(SKILLS['map'], SKILLS['port']),
              reason='Throughput, not single-request latency: set_batched_mode/infer_batch '
                      '(pi05_rtx.py:2413,2663) are unreachable from the single-observation '
                      'VLABackend.infer (backend.py:141) and the worker API measures one request.'),
        _tool(tool_id='attention.sdpa_backend', title='Explicit SDPA backend selection',
              lane='port', stages=('kernel.attention', 'vision.encode', 'prefix.prefill', 'denoise.velocity'),
              model_predicate={'all': [_ATTENTION, {'any': [_feature('functional_torch'),
                                                           _feature('torch_modules')]}]},
              frameworks=('torch',), exactness='approximate', status='needs_implementation',
              skills=(SKILLS['map'], SKILLS['port'], SKILLS['gate']),
              reason='No validated BackendConfig selector exists. First identify the active attention '
                     'entry, mask, dtype, head_dim and GQA contract; implement selection in the product '
                     'path and validate eager, capture/replay and changed-input behavior before a campaign.'),
        _tool(tool_id='attention.flash_attention', title='FlashAttention operator integration',
              lane='port', stages=('kernel.attention', 'vision.encode', 'prefix.prefill', 'denoise.velocity'),
              model_predicate=_ATTENTION, min_capability=(8, 0), frameworks=('torch',),
              exactness='approximate', status='needs_implementation',
              skills=(SKILLS['map'], SKILLS['port'], SKILLS['gate']),
              reason='Existing FA2 calls do not establish a selectable replacement for every attention '
                     'entry. Check device, BF16/FP16, head_dim, Q/KV head divisibility, mask and layout '
                     'against the exact call site. No unverified attention port is ready.'),
        _tool(tool_id='profile.stage_intervals', title='Stage interval profiling + attribution',
              lane='analysis', stages=('host.input', 'vision.encode', 'vision.project', 'prefix.prefill',
                                       'denoise.velocity', 'denoise.routing', 'host.output'),
              models=('*',), exactness='n/a', status='ready', skills=(SKILLS['profile'],),
              adapter='flash_rt.vla.autotune.stages',
              reason='No candidates: a separate diagnostic request records stage_metrics '
                     '(--no-profile disables); attribute with `python -m flash_rt.vla.autotune.stages '
                     'attribute RESULT_JSON --rank`. Coverage follows ModelSpec profiling strategy; '
                     'missing intervals remain unknown and are never formal latency samples.'),
        _tool(tool_id='profile.pi05_dataflow', title='Declared model dataflow ledger',
              lane='analysis', stages=('prefix.prefill', 'denoise.velocity'),
              model_predicate={'path': 'profiling.dataflow', 'exists': True},
              exactness='n/a', status='ready', skills=(SKILLS['profile'],),
              adapter='flash_rt.vla.autotune.dataflow',
              reason='No candidates: call the declared profiling.dataflow adapter. It may emit geometry '
                     'or an implemented FLOP/weight-byte cost ledger; missing quantities stay unknown. '
                     'This is not profiler-measured HBM traffic.'),
        _tool(tool_id='profile.ncu_kernel', title='Nsight Compute kernel counters', lane='analysis',
              stages=('kernel.gemm', 'kernel.attention', 'kernel.norm_elementwise', 'kernel.conv',
                      'vision.encode'),
              models=('*',), exactness='n/a', status='needs_setup', skills=(SKILLS['profile'], SKILLS['kernel']),
              adapter='flash_rt.vla.autotune.tools.external.forgekernel',
              reason='Needs ncu with GPU counter permission on the compute node (or sudo); run '
                     'plain ncu or ForgeKernel inner/ncu.py; on A800/GA100 the PM-sampling interval '
                     'must be >= 20000 cycles. Not wired into the worker.'),
        _tool(_propose_levers, tool_id='analysis.lever_pricing',
              title='Factorial lever pricing (each lever, all, all-minus-one)', lane='analysis',
              stages=STAGE_IDS, models=('*',), exactness='n/a', status='ready',
              skills=(SKILLS['config'],), parameters={'levers': None, 'stage': None}),
        _tool(tool_id='baseline.official_impl', title='Direct official/author implementation baseline',
              lane='baseline', stages=('host.input', 'vision.encode', 'prefix.prefill', 'denoise.velocity',
                                       'host.output'),
              models=('*',), exactness='n/a', status='ready', skills=(SKILLS['map'],),
              adapter='flash_rt.vla.autotune.author',
              reason='controller init --author-baseline author.json: run the author on the same GPU or '
                     'bind saved author outputs. The unified producer calls the selected ModelSpec '
                     'author.adapter load(spec, torch); supported options come from author.options. '
                     'Every candidate has separate internal and author quality records.',
              model_predicate={'path': 'author.adapter', 'exists': True}),
    )
