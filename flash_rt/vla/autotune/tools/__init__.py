"""Tool pool for the stage-aware autotune loop.

``all_tools()`` = the in-repo tools (``builtin``) plus one ToolSpec per external
adapter, built from that adapter's own ``SPEC`` (``external.external_specs``).
``list_tools`` filters by model, hardware capability, stage and lane, hiding
tools that are not ready unless ``include_unavailable``. ``propose`` runs a
config/analysis generator and returns Candidates. Native build prerequisites
are checked on the target, so ``needs_setup`` config tools can also be planned;
patch, port, baseline and external tools never generate candidates here: their
adapters prepare jobs and ``patches.candidate_from_proposal`` turns their
stored source change into a Candidate. Importing this package loads no model,
no Torch and no external project.
"""

from __future__ import annotations

from flash_rt.vla.model_spec import get_spec

from .spec import (ANY_MODEL, EXACTNESS, HARDWARE_LABELS, LANES, MODELS, STATUSES, ToolSpec, Upstream,
                   hardware_capability, spec_from_external)


_TOOLS = None
_ALIASES = {}


def _registry() -> dict:
    global _TOOLS, _ALIASES
    if _TOOLS is None:
        from .builtin import builtin_tools
        from .external import ADAPTERS, external_specs
        tools = {}
        external = [spec_from_external(spec, ADAPTERS[tool_id]) for tool_id, spec in external_specs().items()]
        for tool in (*builtin_tools(), *external):
            if tool.tool_id in tools:
                raise ValueError(f'Duplicate tool id {tool.tool_id!r}')
            tools[tool.tool_id] = tool
        aliases = {}
        for tool in tools.values():
            for alias in tool.aliases:
                if alias in tools or alias in aliases:
                    raise ValueError(f'Duplicate tool alias {alias!r}')
                aliases[alias] = tool.tool_id
        _ALIASES = aliases
        _TOOLS = tools
    return _TOOLS


def all_tools() -> tuple:
    """Every registered tool, builtin first, in registry order."""
    return tuple(_registry().values())


def get_tool(tool_id) -> ToolSpec:
    tools = _registry()
    tool_id = _ALIASES.get(tool_id, tool_id)
    if tool_id not in tools:
        raise ValueError(f'Unknown tool {tool_id!r}; known: {sorted(tools)}')
    return tools[tool_id]


def list_tools(model=None, hardware=None, stage=None, lane=None, include_unavailable=False, *,
               spec_dir=None) -> list:
    """Tools matching every given filter.

    ``model``: the tool applies to it (status/reason per model via status_for).
    ``hardware``: a registry hardware label; tools whose min_capability exceeds
    it are excluded, and so are capability-gated tools on 'auto' (unknown device).
    ``include_unavailable=False`` hides tools whose status is not 'ready'.
    """
    from ..stages import STAGE_IDS
    spec = get_spec(model, spec_dir=spec_dir) if model is not None else None
    if hardware is not None:
        hardware_capability(hardware)
        if model is not None and hardware != 'auto':
            if hardware not in spec['capabilities']['hardware']:
                return []
    if stage is not None and stage not in STAGE_IDS:
        raise ValueError(f'Unknown stage {stage!r}')
    if lane is not None and lane not in LANES:
        raise ValueError(f'Unknown lane {lane!r}; use one of {list(LANES)}')
    selected = []
    for tool in all_tools():
        if model is not None and not tool.applies_to(model, spec_dir=spec_dir):
            continue
        if hardware is not None and not tool.available_on(hardware, model, spec_dir=spec_dir):
            continue
        if stage is not None and stage not in tool.stages_for(model, spec_dir=spec_dir):
            continue
        if lane is not None and tool.lane != lane:
            continue
        if not include_unavailable and tool.status_for(model, spec_dir=spec_dir) != 'ready':
            continue
        selected.append(tool)
    return selected


def propose(tool_id, baseline, **params) -> list:
    """Candidates from a ready config/analysis tool; ValueError for every other tool."""
    from ..schema import Candidate
    tool = get_tool(tool_id)
    if not isinstance(baseline, Candidate):
        raise TypeError('baseline must be a Candidate')
    model = baseline.backend.model
    spec_dir = getattr(baseline.backend, 'spec_dir', None)
    status = tool.status_for(model, spec_dir=spec_dir)
    matches, reason = tool.match_model(model, spec_dir=spec_dir)
    if not matches:
        raise ValueError(f'{tool_id} does not apply to model {model!r}: {reason}')
    constraint = tool.configuration_reason(baseline.backend)
    if constraint:
        raise ValueError(f'{tool_id} {constraint}')
    pointer = f'; see {tool.adapter}' if tool.adapter else ''
    if tool.propose is None or tool.lane not in ('config', 'analysis'):
        raise ValueError(f'{tool_id} ({tool.lane}, {status}) does not generate candidates here: '
                         f'{tool.reason_for(model, spec_dir=spec_dir) or "no deterministic generator"}{pointer}')
    if (baseline.backend.hardware != 'auto'
            and not tool.available_on(baseline.backend.hardware, model, spec_dir=spec_dir)):
        raise ValueError(f'{tool_id} {tool.hardware_reason(baseline.backend.hardware, model, spec_dir=spec_dir)}')
    # Native options can be planned on a CPU host before the target extension
    # is built. The explicit device path checks the build at load time.
    if status != 'ready' and not (status == 'needs_setup' and tool.lane == 'config'):
        raise ValueError(f'{tool_id} is {status} for {model}: '
                         f'{tool.reason_for(model, spec_dir=spec_dir)}{pointer}')
    unknown = set(params) - set(tool.parameters)
    if unknown:
        raise ValueError(f'{tool_id} has no parameters {sorted(unknown)}; known: {sorted(tool.parameters)}')
    return tool.propose(baseline, **params)


def stage_coverage(model, hardware='auto', *, spec_dir=None) -> list:
    """One row per stage, including concrete reasons for every unavailable route.

    This is declarative planning coverage, never a device probe or a claim that
    an external tool/attention port has been installed or validated.
    """
    from ..stages import STAGES
    spec = get_spec(model, spec_dir=spec_dir)
    hardware_capability(hardware)
    declared = {entry['stage_id'] for entry in spec['stages']}
    rows = []
    for stage_id, info in STAGES.items():
        entries = []
        for tool in all_tools():
            if stage_id not in tool.stages:
                continue
            matches, reason = tool.match_model(model, spec_dir=spec_dir)
            status = tool.status_for(model, spec_dir=spec_dir)
            if not matches:
                status = 'blocked'
            elif stage_id not in tool.stages_for(model, spec_dir=spec_dir):
                status, reason = ('blocked', 'ModelSpec recipe does not target this stage; stages: '
                                  + ', '.join(tool.stages_for(model, spec_dir=spec_dir)))
            elif hardware != 'auto' and hardware not in spec['capabilities']['hardware']:
                status, reason = 'blocked', f'ModelSpec has no {hardware} product path'
            elif not tool.available_on(hardware, model, spec_dir=spec_dir):
                status, reason = 'blocked', tool.hardware_reason(hardware, model, spec_dir=spec_dir)
            else:
                reason = tool.reason_for(model, spec_dir=spec_dir)
            entries.append({'tool_id': tool.tool_id, 'lane': tool.lane, 'status': status,
                            'reason': reason, 'matches_model': matches,
                            'proposes_candidates': tool.propose is not None,
                            'frameworks': list(tool.frameworks), 'samplers': list(tool.samplers)})
        rows.append({'stage': stage_id, 'title': info.title, 'declared': stage_id in declared,
                     'tools': entries,
                     'note': '' if stage_id in declared else
                             'Stage is absent from this ModelSpec; cross-cutting diagnostics may still apply.'})
    return rows


__all__ = ['ANY_MODEL', 'EXACTNESS', 'HARDWARE_LABELS', 'LANES', 'MODELS', 'STATUSES', 'ToolSpec',
           'Upstream', 'all_tools', 'get_tool', 'hardware_capability', 'list_tools', 'propose',
           'spec_from_external', 'stage_coverage']
