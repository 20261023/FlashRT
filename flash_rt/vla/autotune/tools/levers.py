"""Factorial lever pricing: what each change buys alone, together, and when left out.

A lever is ``name -> overrides`` over the BackendConfig fields ``num_steps``,
``sampler``, ``probeflow`` (a dict of ProbeFlowConfig fields, unspecified
fields at their defaults, or None) and ``backend``, plus ``options`` (merged
key by key into the baseline options). Points: each lever alone, all levers
(two or more), and all-minus-one (three or more). The baseline itself is not
emitted: it is the campaign reference. Lever names are sorted, so the output is
deterministic; a point whose encoded backend equals the baseline's or an
earlier point's is dropped (first kept). Two levers in one point that set the
same field (or option) to different values are an error, never a silent
precedence rule. Every point passes registry.validate_config.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re

from ..schema import Candidate, encode_backend
from ..stages import STAGE_IDS
from .builtin import finalize, make_candidate


TOOL_ID = 'analysis.lever_pricing'
LEVER_FIELDS = ('num_steps', 'sampler', 'probeflow', 'backend', 'options')
_NAME = re.compile(r'[a-z0-9][a-z0-9_-]*')


def _check_levers(levers):
    if not isinstance(levers, Mapping) or not levers:
        raise ValueError('levers must be a nonempty mapping name -> overrides')
    checked = {}
    for name, overrides in levers.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f'lever name must match ^[a-z0-9][a-z0-9_-]*$, got {name!r}')
        if not isinstance(overrides, Mapping) or not overrides:
            raise ValueError(f'lever {name!r} must be a nonempty mapping of overrides')
        unknown = set(overrides) - set(LEVER_FIELDS)
        if unknown:
            raise ValueError(f'lever {name!r} overrides unknown fields {sorted(unknown)}; '
                             f'allowed: {list(LEVER_FIELDS)}')
        if 'options' in overrides and (not isinstance(overrides['options'], Mapping)
                                       or not overrides['options']):
            raise ValueError(f'lever {name!r} options must be a nonempty mapping')
        if overrides.get('probeflow') is not None and not isinstance(overrides['probeflow'], Mapping):
            raise ValueError(f'lever {name!r} probeflow must be a mapping of ProbeFlowConfig fields or null')
        checked[name] = overrides
    return checked


def combinations(names) -> list:
    """Sorted names -> [each alone..., all (n>=2), all-minus-one... (n>=3)]."""
    names = sorted(names)
    points = [(name,) for name in names]
    if len(names) >= 2:
        points.append(tuple(names))
    if len(names) >= 3:
        points.extend(tuple(other for other in names if other != left_out) for left_out in names)
    return points


def _combine(baseline, levers, names):
    record = encode_backend(baseline.backend)
    fields, options, owners = {}, dict(record['options']), {}
    for name in names:
        for key, value in levers[name].items():
            if key == 'options':
                for option, option_value in value.items():
                    owner = owners.get(('options', option))
                    if owner is not None and options[option] != option_value:
                        raise ValueError(f'levers {owner!r} and {name!r} set option {option!r} differently')
                    options[option], owners[('options', option)] = option_value, name
                continue
            owner = owners.get(key)
            if owner is not None and fields[key] != value:
                raise ValueError(f'levers {owner!r} and {name!r} set {key!r} differently')
            fields[key], owners[key] = value, name
    record.update(fields)
    record['options'] = options
    return record


def price_levers(baseline, levers, *, stage=None, tool=None) -> list:
    """Candidates for every lever combination (see module docstring)."""
    from . import get_tool
    if not isinstance(baseline, Candidate):
        raise TypeError('baseline must be a Candidate')
    if stage is not None and stage not in STAGE_IDS:
        raise ValueError(f'stage must be None or a known stage id, got {stage!r}')
    levers = _check_levers(levers)
    tool = tool or get_tool(TOOL_ID)
    candidates = []
    for names in combinations(levers):
        record = _combine(baseline, levers, names)
        change = 'levers ' + '+'.join(names) + ' ' + json.dumps(
            {name: levers[name] for name in names}, sort_keys=True, separators=(',', ':'))
        try:
            candidates.append(make_candidate(baseline, tool, 'lv', '+'.join(names), record, change,
                                             stage=stage))
        except (TypeError, ValueError) as error:
            raise ValueError(f'lever combination {"+".join(names)} is invalid: {error}') from error
    return finalize(baseline, candidates, effective=False)
