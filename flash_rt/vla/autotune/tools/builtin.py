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
from flash_rt.vla.registry import _CAPABILITIES, validate_config

from ..schema import Candidate, decode_backend, encode_backend
from ..search import enumerate_candidates
from ..stages import STAGE_IDS
from .spec import ToolSpec


BUILTIN = 'flash_rt.vla.autotune.tools.builtin'

# Frontend defaults used to decide "identical to the baseline" (evidence):
# use_cuda_graph: pi0_rtx.py:408 True, pi05_rtx.py:669 True, hyvla_probeflow.py:52
# True, fastwam.py:64 False. Euler steps: NUM_STEPS_DEFAULT=10
# (pi0/pipeline_rtx.py:61, pi05/pipeline_rtx.py:100); FastWAM validate_sampler
# defaults to 20 (models/fastwam/config.py); Hy reads num_steps from the
# checkpoint config (hyvla_thor.py:125), so its default is unknown here.
GRAPH_DEFAULT = {'pi0': True, 'pi05': True, 'hyvla': True, 'fastwam': False, 'dm05': True}
EULER_DEFAULT_STEPS = {'pi0': 10, 'pi05': 10, 'hyvla': None, 'fastwam': 20, 'dm05': 10}
PROBEFLOW_DEFAULT_MAX = {'pi0': 10, 'pi05': 10, 'hyvla': 10, 'fastwam': 20}
OPTION_DEFAULTS = {
    'dm05': {'project_optimization': 'none'},
    'pi05': {'state_prompt_mode': 'exact', 'vision_pool_factor': 1},
    'hyvla': {'numerical_profile': 'semantic_bf16'},
    'fastwam': {'compile_action_infer': False},
}
EULER_STEPS = {'pi0': (5, 6, 7, 8, 9), 'pi05': (5, 6, 7, 8, 9), 'hyvla': (5, 6, 7, 8, 9),
               'fastwam': (10, 12, 13, 14, 16, 18)}
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

def effective_backend(config) -> dict:
    """Encoded backend with frontend defaults filled in, for equivalence checks only."""
    record = encode_backend(config)
    model = record['model']
    options = dict(record['options'])
    options.setdefault('use_cuda_graph', GRAPH_DEFAULT[model])
    for name, value in OPTION_DEFAULTS.get(model, {}).items():
        options.setdefault(name, value)
    if model == 'fastwam' and record['backend'] == 'flashrt':
        options.setdefault('native_linear_backend', 'legacy')
    if record['sampler'] == 'euler' and record['num_steps'] is None and EULER_DEFAULT_STEPS[model]:
        record['num_steps'] = EULER_DEFAULT_STEPS[model]
    record['options'] = options
    return record


def check_baseline(tool, baseline):
    if not isinstance(baseline, Candidate):
        raise TypeError('baseline must be a Candidate')
    if not tool.applies_to(baseline.backend.model):
        raise ValueError(f'{tool.tool_id} does not apply to model {baseline.backend.model!r} '
                         f'(models: {list(tool.models)})')


_PRIMARY = object()


def make_candidate(baseline, tool, short, value, backend_record, change, *, stage=_PRIMARY):
    """One validated config candidate derived from ``baseline``.

    ``stage`` defaults to the tool's primary stage; pass None for no stage.
    """
    backend = decode_backend(backend_record)  # Detaches nested options.
    allowed = set(_CAPABILITIES[backend.model]['options'])
    unknown = set(backend.options) - allowed
    if unknown:
        raise ValueError(f'{tool.tool_id} would emit unsupported {backend.model} options {sorted(unknown)}')
    validate_config(backend)
    return Candidate(
        candidate_id=f'{baseline.candidate_id}-{short}-{value}', backend=backend,
        workload_id=baseline.workload_id, parent_id=baseline.candidate_id,
        hypothesis=f'{tool.tool_id}: {change}; exactness={tool.exactness}',
        origin='search', patch_ref=baseline.patch_ref, source_ref=baseline.source_ref,
        tool=tool.tool_id, stage=tool.primary_stage if stage is _PRIMARY else stage)


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
    model = baseline.backend.model
    values = sorted(set(_positive_ints(EULER_STEPS[model] if steps is None else steps, 'steps')))
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
    return PROBEFLOW_DEFAULT_MAX[config.model]


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
    if effective_backend(config)['options']['use_cuda_graph']:
        return []
    options = {**config.options, 'use_cuda_graph': True}
    change = 'capture the request in CUDA graphs (use_cuda_graph=True)'
    if config.model == 'fastwam' and config.backend == 'flashrt':
        # registry: FastWAM native CUDA graph requires native_linear_backend="cached".
        options['native_linear_backend'] = 'cached'
        change += ' with native_linear_backend=cached (required by the FastWAM native graph)'
    if config.model == 'fastwam' and options.get('compile_action_infer'):
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


def _propose_fastwam_compile(tool, baseline):
    check_baseline(tool, baseline)
    config = baseline.backend
    if effective_backend(config)['options']['compile_action_infer']:
        return []
    options = {**config.options, 'compile_action_infer': True, 'use_cuda_graph': False}
    options.pop('native_linear_backend', None)
    change = ('compile video prefill and raw denoise with torch.compile (compile_action_infer=True; '
              'backend=torch, use_cuda_graph=False, as the registry requires)')
    return finalize(baseline, [make_candidate(baseline, tool, 'compile', 'on',
                                              _with(baseline, backend='torch', options=options), change)])


def _propose_dm05_compile(tool, baseline):
    check_baseline(tool, baseline)
    options = {**baseline.backend.options, 'project_optimization': 'compiled'}
    return finalize(baseline, [make_candidate(
        baseline, tool, 'compile', 'on', _with(baseline, backend='torch', options=options),
        'DM0.5 packed projections and compiled vision/prefix/action layers at fixed BF16 Euler10')])


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
    if baseline.backend.framework != 'torch' or baseline.backend.sampler != 'euler':
        raise ValueError('quant.int8_pi05 requires a Torch Euler baseline on SM80')
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
    if function is None:
        return ToolSpec(**fields)
    holder, propose = _bind(function)
    tool = ToolSpec(propose=propose, adapter=BUILTIN, **fields)
    holder['tool'] = tool
    return tool


# --------------------------------------------------------------------------- registry

_NO_QUANT = ('registry.load_backend fixes use_fp8=False for pi0/pi05; '
             'HyVLAFlowFrontend requires its FP8/FP4 flags to be false')


def builtin_tools() -> tuple:
    """Fresh ToolSpec objects for every in-repo tool, in registry order."""
    config_skills = (SKILLS['config'],)
    return (
        _tool(_propose_euler_steps, tool_id='sampler.euler_steps', title='Fixed Euler step count',
              lane='config', stages=('denoise.schedule',), models=tuple(EULER_STEPS), exactness='approximate',
              status='ready', skills=config_skills,
              parameters={'steps': {model: list(values) for model, values in EULER_STEPS.items()}}),
        _tool(_propose_probeflow, tool_id='sampler.probeflow', title='ProbeFlow adaptive schedule',
              lane='config', stages=('denoise.schedule', 'denoise.routing'), models=tuple(EULER_STEPS),
              exactness='approximate', status='ready', skills=config_skills,
              parameters={'epsilons': list(PROBEFLOW_EPSILONS), 'max_steps': None}),
        _tool(_propose_cuda_graph, tool_id='launch.cuda_graph', title='CUDA graph capture',
              lane='config', stages=('host.launch', 'denoise.velocity'), models=('*',), exactness='exact',
              status='ready', skills=config_skills, parameters={},
              reason='No candidate when the effective baseline already captures graphs '
                     '(default True for pi0/pi05/hyvla/dm05, False for FastWAM).'),
        _tool(_propose_dm05_compile, tool_id='compile.dm05_inductor',
              title='DM0.5 packed projections and compiled layers', lane='config',
              stages=('vision.encode', 'prefix.prefill', 'denoise.velocity', 'kernel.gemm'),
              models=('dm05',), hardware_only=('cuda_sm80',),
              exactness='approximate', status='ready', skills=config_skills,
              reason='Project Torch backend, BF16 Euler10 on SM80. Preserves the graph setting.'),
        _tool(_propose_fastwam_linear, tool_id='gemm.fastwam_linear_backend',
              title='FastWAM linear backend (torch / native cached / native legacy)',
              lane='config', stages=('kernel.gemm', 'host.launch', 'denoise.velocity'), models=('fastwam',),
              exactness='exact', status='ready', skills=config_skills,
              parameters={'variants': list(FASTWAM_LINEAR_VARIANTS)}),
        _tool(_propose_fastwam_compile, tool_id='compile.fastwam_inductor',
              title='FastWAM torch.compile of video prefill + raw denoise (author recipe)',
              lane='config', stages=('host.launch', 'denoise.velocity', 'prefix.prefill'),
              models=('fastwam',), exactness='approximate', status='ready', skills=config_skills,
              parameters={},
              reason='Opt-in compile_action_infer (frontends/torch/fastwam.py, models/fastwam/pipeline.py): '
                     'torch.compile(mode="reduce-overhead", fullgraph=True) of mot.prefill_video_cache_tensor '
                     'and _denoise_action_with_video_cache. Requires backend="torch" and use_cuda_graph=False '
                     '(registry.validate_config). Cold compile is outside steady-state latency: record it and '
                     'the break-even call count. Inductor fusions change BF16 rounding, so the quality gate '
                     'applies.'),
        _tool(_propose_vision_pool, tool_id='prefix.vision_pool', title='pi0.5 vision token pooling',
              lane='config', stages=('prefix.prefill', 'vision.project'), models=('pi05',),
              exactness='approximate', status='ready', skills=config_skills,
              parameters={'factors': [2, 4]}),
        _tool(_propose_state_prompt_mode, tool_id='prefix.state_prompt_mode',
              title='pi0.5 fixed-shape state prompt', lane='config',
              stages=('host.launch', 'prefix.prefill'), models=('pi05',), exactness='exact',
              status='ready', skills=config_skills, parameters={'modes': ['fixed']}),
        _tool(_propose_hy_profile, tool_id='precision.hy_numerical_profile',
              title='Hy numerical profile (alignment, not speed)', lane='config',
              stages=('precision.weights',), models=('hyvla',), exactness='n/a', status='ready',
              skills=config_skills, parameters={'profiles': ['official_local_fp32']}),
        _tool(tool_id='quant.fp8', title='FP8 weights/activations', lane='patch',
              stages=('precision.weights',), models=('pi0', 'pi05', 'hyvla'),
              exactness='approximate', status='blocked', min_capability=(8, 9),
              skills=(SKILLS['patch'], SKILLS['gate']),
              reason='Needs SM>=89 FP8 tensor cores (A800/sm_80 has none: pi05_rtx.py:665,816-819 '
                     'falls back to BF16) and the unified backend forbids quantized precision: '
                     + _NO_QUANT + '.'),
        _tool(tool_id='quant.nvfp4', title='NVFP4 / W4A8 weights', lane='patch',
              stages=('precision.weights',), models=('pi05', 'hyvla'), exactness='approximate',
              status='blocked', min_capability=(12, 0), skills=(SKILLS['patch'], SKILLS['gate']),
              reason='NVFP4 kernels are built only for sm_120a/121a (CMakeLists.txt:58-64); pi05 '
                     'nvfp4 prefix/decoder selectors (pi05_rtx.py:694-717) are pinned to fp8 labels '
                     'by registry.load_backend and FLASHRT_PI05_*_PRECISION env values are rejected '
                     'by registry.validate_config; Hy use_fp4 must be False (hyvla_probeflow.py:75-83).'),
        _tool(_propose_pi05_int8, tool_id='quant.int8_pi05', title='pi0.5 INT8 CUTLASS GEMMs', lane='config',
              stages=('precision.weights', 'kernel.gemm'), models=('pi05',), exactness='approximate',
              status='needs_setup', hardware_only=('cuda_sm80',), skills=config_skills,
              parameters={'modes': ['encoder', 'encoder_decoder']},
              reason='Build flash_rt_kernels with -DGPU_ARCH=80 -DENABLE_SM80_INT8_CUTLASS=ON. '
                     'Torch Euler, dynamic per-row activation scales. Unset ambient precision overrides.'),
        _tool(_propose_hy_kv, tool_id='fusion.hy_rope_qknorm_kvwrite', title='Hy fused RoPE + QK-norm + KV-write',
              lane='config', stages=('kernel.norm_elementwise', 'prefix.prefill', 'denoise.velocity'),
              models=('hyvla',), exactness='approximate', status='needs_setup',
              hardware_only=('cuda_sm80',), skills=config_skills,
              reason='Build flash_rt_hyvla_sm80 with -DGPU_ARCH=80 -DFLASHRT_ENABLE_HYVLA=ON. '
                     'One launch computes RoPE, QK norm and the layer KV window per prefix/expert call.'),
        _tool(_propose_fastwam_text_kv, tool_id='cache.fastwam_text_crossattn_kv',
              title='FastWAM per-request text cross-attention K/V', lane='config',
              stages=('denoise.velocity',), models=('fastwam',), exactness='exact', status='ready',
              skills=config_skills,
              reason='Prepare text embedding and action cross-attention K/V once after appending the '
                     'current proprio. Each request refreshes the cache, including captured graph inputs.'),
        _tool(tool_id='cache.pi05_cross_frame_kv', title='pi0.5 cross-frame prefix KV reuse',
              lane='port', stages=('prefix.prefill', 'vision.encode'), models=('pi05',),
              exactness='approximate', status='blocked', skills=(SKILLS['map'], SKILLS['port']),
              reason='Semantic change, not a speed knob: cache_frames>1 reuses stale encoder K/V '
                     '(pi05_rtx.py:759-765,2279-2303); it is not a registry option, ProbeFlow refuses '
                     'it (pi05_rtx.py:538-541) and it needs a multi-frame workload.'),
        _tool(tool_id='throughput.pi05_batch', title='pi0.5 batched inference', lane='port',
              stages=('prefix.prefill', 'denoise.velocity'), models=('pi05',), exactness='approximate',
              status='blocked', skills=(SKILLS['map'], SKILLS['port']),
              reason='Throughput, not single-request latency: set_batched_mode/infer_batch '
                     '(pi05_rtx.py:2413,2663) are unreachable from the single-observation '
                     'VLABackend.infer (backend.py:141) and the worker API measures one request.'),
        _tool(tool_id='profile.stage_intervals', title='Stage interval profiling + attribution',
              lane='analysis', stages=('host.input', 'vision.encode', 'vision.project', 'prefix.prefill',
                                       'denoise.velocity', 'denoise.routing', 'host.output'),
              models=('*',), exactness='n/a', status='ready', skills=(SKILLS['profile'],),
              adapter='flash_rt.vla.autotune.stages',
              reason='No candidates: the worker records stage_metrics by default (profiling.py; '
                     '--no-profile disables); attribute with `python -m flash_rt.vla.autotune.stages '
                     'attribute RESULT_JSON --rank`. No intervals without use_cuda_graph (pi0/pi05/hyvla).'),
        _tool(tool_id='profile.pi05_dataflow', title='pi0.5 implementation dataflow ledger',
              lane='analysis', stages=('prefix.prefill', 'denoise.velocity'), models=('pi05',),
              exactness='n/a', status='ready', skills=(SKILLS['profile'],),
              adapter='flash_rt.vla.autotune.dataflow',
              reason='No candidates: every pi05 worker result carries record["dataflow"] (prefix/action '
                     'rows, FFN FLOPs, weight bytes); a cost ledger, not profiler-measured HBM traffic.'),
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
                     'bind saved author outputs. Every candidate has separate internal and author quality records.',
              model_status={
                  'pi0': ('ready', 'examples/vla_autotune/official_native_latency.py --model pi0: author OpenPI '
                                   'JAX (module_jit) on the GPU, with --source/--source-revision/--checkpoint/'
                                   '--workload/--output/--repeats; allocation only.'),
                  'pi05': ('ready', 'examples/vla_autotune/official_native_latency.py --model pi05: author OpenPI '
                                    'PyTorch (default max-autotune compile); the resident LeRobot checkpoint uses '
                                    'shared-weight-aware safetensors loading.'),
                  'fastwam': ('ready', 'examples/vla_autotune/official_native_latency.py --model fastwam: author '
                                       'FastWAM.infer_action with the author compile_action_infer=True and cached '
                                       'T5 inputs (cold compile reported separately). Also '
                                       'examples/vla_autotune/validate_fastwam.py --reference (direct pinned '
                                       'upstream infer_action, Euler only).'),
                   'hyvla': ('ready', 'examples/vla_autotune/official_hyvla_latency.py: unmodified author HyVLA '
                                      'prepare_images/state/language + policy.model.sample_actions with fixed noise '
                                      '(--official-source/--checkpoint/--fixtures/--source-revision/--output).'),
                   'dm05': ('ready', 'examples/vla_autotune/official_dm05_latency.py: OpenDM base, '
                                     'three current RGB views, fixed Euler10, absolute public14 actions.'),
              }),
    )
