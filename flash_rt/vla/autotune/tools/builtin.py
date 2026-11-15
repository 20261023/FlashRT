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
GRAPH_DEFAULT = {'pi0': True, 'pi05': True, 'hyvla': True, 'fastwam': False}
EULER_DEFAULT_STEPS = {'pi0': 10, 'pi05': 10, 'hyvla': None, 'fastwam': 20}
PROBEFLOW_DEFAULT_MAX = {'pi0': 10, 'pi05': 10, 'hyvla': 10, 'fastwam': 20}
OPTION_DEFAULTS = {
    'pi05': {'state_prompt_mode': 'exact', 'vision_pool_factor': 1},
    'hyvla': {'numerical_profile': 'semantic_bf16'},
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
    return finalize(baseline, [make_candidate(baseline, tool, 'graph', 'on',
                                              _with(baseline, options=options), change)])


def _propose_fastwam_linear(tool, baseline, *, variants=FASTWAM_LINEAR_VARIANTS):
    check_baseline(tool, baseline)
    variants = _choices(variants, FASTWAM_LINEAR_VARIANTS, 'variants')
    options = dict(baseline.backend.options)
    candidates = []
    for variant in variants:
        changed = dict(options)
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

_NO_QUANT = ('config.py:56-57 rejects quantized precision ("quantized tuning is not implemented"); '
             'registry.load_backend forces use_fp8=False for pi0/pi05 and HyVLAFlowFrontend '
             'requires every FP8/FP4 flag False (hyvla_probeflow.py:75-83)')


def builtin_tools() -> tuple:
    """Fresh ToolSpec objects for every in-repo tool, in registry order."""
    config_skills = (SKILLS['config'],)
    return (
        _tool(_propose_euler_steps, tool_id='sampler.euler_steps', title='Fixed Euler step count',
              lane='config', stages=('denoise.schedule',), models=('*',), exactness='approximate',
              status='ready', skills=config_skills,
              parameters={'steps': {model: list(values) for model, values in EULER_STEPS.items()}}),
        _tool(_propose_probeflow, tool_id='sampler.probeflow', title='ProbeFlow adaptive schedule',
              lane='config', stages=('denoise.schedule', 'denoise.routing'), models=('*',),
              exactness='approximate', status='ready', skills=config_skills,
              parameters={'epsilons': list(PROBEFLOW_EPSILONS), 'max_steps': None}),
        _tool(_propose_cuda_graph, tool_id='launch.cuda_graph', title='CUDA graph capture',
              lane='config', stages=('host.launch', 'denoise.velocity'), models=('*',), exactness='exact',
              status='ready', skills=config_skills, parameters={},
              reason='No candidate when the effective baseline already captures graphs '
                     '(default True for pi0/pi05/hyvla, False for FastWAM).'),
        _tool(_propose_fastwam_linear, tool_id='gemm.fastwam_linear_backend',
              title='FastWAM linear backend (torch / native cached / native legacy)',
              lane='config', stages=('kernel.gemm', 'host.launch', 'denoise.velocity'), models=('fastwam',),
              exactness='exact', status='ready', skills=config_skills,
              parameters={'variants': list(FASTWAM_LINEAR_VARIANTS)}),
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
        _tool(tool_id='quant.int8_pi05', title='pi0.5 INT8 CUTLASS GEMMs', lane='port',
              stages=('precision.weights', 'kernel.gemm'), models=('pi05',), exactness='approximate',
              status='needs_implementation', min_capability=(8, 0),
              skills=(SKILLS['port'], SKILLS['patch'], SKILLS['gate']),
              reason='INT8 paths exist (pi05_rtx.py:795-819, FVK_PI05_RTX_FORCE_INT8/_INT8_ENCODER_ONLY/'
                     '_INT8_VISION) but need a -DENABLE_SM80_INT8_CUTLASS=ON build (default ON only for '
                     'GPU_ARCH 87: CMakeLists.txt:121-127) and a registry unblock: '
                     'registry.validate_config rejects those env vars for the BF16 backend.'),
        _tool(tool_id='fusion.hy_rope_qknorm_kvwrite', title='Hy fused RoPE + QK-norm + KV-write',
              lane='patch', stages=('kernel.norm_elementwise', 'prefix.prefill', 'denoise.velocity'),
              models=('hyvla',), exactness='approximate', status='needs_implementation',
              skills=(SKILLS['port'], SKILLS['patch'], SKILLS['kernel'], SKILLS['gate']),
              reason='fvk.hyvla_rope_qknorm_kvwrite_bf16 (used at pipeline_thor.py:425-432 when '
                     '_fused_attn) is plain CUDA but CMakeLists.txt:2168-2190 builds it only for GPU_ARCH '
                     '110, or 87/120/121 with FLASHRT_ENABLE_HYVLA=ON: add 80 to the arch list, build '
                     'with -DFLASHRT_ENABLE_HYVLA=ON and allow use_fused in HyVLAFlowFrontend '
                     '(rejected at hyvla_probeflow.py:75-83). Numerics differ from the unfused path.'),
        _tool(tool_id='cache.fastwam_text_crossattn_kv', title='FastWAM per-request text cross-attention K/V',
              lane='patch', stages=('denoise.velocity',), models=('fastwam',), exactness='exact',
              status='needs_implementation',
              skills=(SKILLS['port'], SKILLS['patch'], SKILLS['map'], SKILLS['gate']),
              reason='Every denoising step passes the same context to model._denoise_action_with_video_cache '
                     '(flash_rt/models/fastwam/pipeline.py _evaluate_at_timestep), which recomputes '
                     'ActionDiT text_embedding(context) and every layer\'s cross-attention K/V although '
                     'they are constant per request; only the first-frame video KV is cached. Needs a '
                     'patch that pre-projects them once per request.'),
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
        _tool(tool_id='baseline.official_impl', title='Direct official/upstream implementation baseline',
              lane='baseline', stages=('host.input', 'vision.encode', 'prefix.prefill', 'denoise.velocity',
                                       'host.output'),
              models=('*',), exactness='n/a', status='needs_implementation', skills=(SKILLS['map'],),
              adapter='examples/vla_autotune/validate_fastwam.py',
              reason='No direct-upstream execution mode for pi0/pi05/hyvla in this tree (Hy has only '
                     'recorded official traces: examples/vla_autotune/hy_trace/reference_observer.py).',
              model_status={'fastwam': (
                  'ready',
                  'examples/vla_autotune/validate_fastwam.py --reference (parser(): "Direct pinned upstream '
                  'infer_action Euler"; run_actions -> load_upstream_reference -> model.infer_action, '
                  'execution backend "direct_pinned_upstream_infer_action") at FastWAM@7faa711. Limits: '
                  'Euler only (main() rejects other samplers), sigma_shift=5 and action_horizon=32 '
                  'hard-coded, Slurm GPU allocation required (require_allocation), T5 context prepared '
                  'separately (--prepare-context); writes its own report (compare with --compare), not '
                  'a controller worker bundle.')}),
    )
