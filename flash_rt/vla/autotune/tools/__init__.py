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

from .spec import (ANY_MODEL, EXACTNESS, HARDWARE_LABELS, LANES, MODELS, STATUSES, ToolSpec, Upstream,
                   hardware_capability, spec_from_external)


_TOOLS = None


def _registry() -> dict:
    global _TOOLS
    if _TOOLS is None:
        from .builtin import builtin_tools
        from .external import ADAPTERS, external_specs
        tools = {}
        external = [spec_from_external(spec, ADAPTERS[tool_id]) for tool_id, spec in external_specs().items()]
        for tool in (*builtin_tools(), *external):
            if tool.tool_id in tools:
                raise ValueError(f'Duplicate tool id {tool.tool_id!r}')
            tools[tool.tool_id] = tool
        _TOOLS = tools
    return _TOOLS


def all_tools() -> tuple:
    """Every registered tool, builtin first, in registry order."""
    return tuple(_registry().values())


def get_tool(tool_id) -> ToolSpec:
    tools = _registry()
    if tool_id not in tools:
        raise ValueError(f'Unknown tool {tool_id!r}; known: {sorted(tools)}')
    return tools[tool_id]


def list_tools(model=None, hardware=None, stage=None, lane=None, include_unavailable=False) -> list:
    """Tools matching every given filter.

    ``model``: the tool applies to it (status/reason per model via status_for).
    ``hardware``: a registry hardware label; tools whose min_capability exceeds
    it are excluded, and so are capability-gated tools on 'auto' (unknown device).
    ``include_unavailable=False`` hides tools whose status is not 'ready'.
    """
    from ..stages import STAGE_IDS
    if model is not None and model not in MODELS:
        raise ValueError(f'Unknown model {model!r}; use one of {list(MODELS)}')
    if hardware is not None:
        hardware_capability(hardware)
        if model is not None and hardware != 'auto':
            from flash_rt.vla.registry import _CAPABILITIES
            if hardware not in _CAPABILITIES[model]['hardware']:
                return []
    if stage is not None and stage not in STAGE_IDS:
        raise ValueError(f'Unknown stage {stage!r}')
    if lane is not None and lane not in LANES:
        raise ValueError(f'Unknown lane {lane!r}; use one of {list(LANES)}')
    selected = []
    for tool in all_tools():
        if model is not None and not tool.applies_to(model):
            continue
        if hardware is not None and not tool.available_on(hardware):
            continue
        if stage is not None and stage not in tool.stages:
            continue
        if lane is not None and tool.lane != lane:
            continue
        if not include_unavailable and tool.status_for(model) != 'ready':
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
    status = tool.status_for(model)
    pointer = f'; see {tool.adapter}' if tool.adapter else ''
    if tool.propose is None or tool.lane not in ('config', 'analysis'):
        raise ValueError(f'{tool_id} ({tool.lane}, {status}) does not generate candidates here: '
                         f'{tool.reason_for(model) or "no deterministic generator"}{pointer}')
    if not tool.applies_to(model):
        raise ValueError(f'{tool_id} does not apply to model {model!r} (models: {list(tool.models)})')
    if (tool.hardware_only and baseline.backend.hardware != 'auto'
            and not tool.available_on(baseline.backend.hardware)):
        raise ValueError(f'{tool_id} requires hardware in {list(tool.hardware_only)}')
    # Native options can be planned on a CPU host before the target extension
    # is built. The explicit device path checks the build at load time.
    if status != 'ready' and not (status == 'needs_setup' and tool.lane == 'config'):
        raise ValueError(f'{tool_id} is {status} for {model}: {tool.reason_for(model)}{pointer}')
    unknown = set(params) - set(tool.parameters)
    if unknown:
        raise ValueError(f'{tool_id} has no parameters {sorted(unknown)}; known: {sorted(tool.parameters)}')
    return tool.propose(baseline, **params)


__all__ = ['ANY_MODEL', 'EXACTNESS', 'HARDWARE_LABELS', 'LANES', 'MODELS', 'STATUSES', 'ToolSpec',
           'Upstream', 'all_tools', 'get_tool', 'hardware_capability', 'list_tools', 'propose',
           'spec_from_external']
