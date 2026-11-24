"""CPU-only attention-site routing; no backend imports or model execution.

Every constraint is evaluated against one site. A mask on one layer cannot
license a compatible head width on another layer. Eligibility is a porting
precondition, never evidence that the package is installed or a port is ready.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


MASK_KINDS = ('none', 'causal', 'arbitrary')
SITE_KINDS = ('spatial', 'temporal', 'self', 'cross', 'joint')
PHASES = ('prefill', 'decode', 'denoise')
LAYOUT_AXES = {
    'BHSD': (1, 2),
    'BSHD': (2, 1),
    'logical-BSHD': (2, 1),
    'BSHD-before-transpose-to-BHSD': (2, 1),
}
SAGE_COMMIT = 'eb615cf6cf4d221338033340ee2de1c37fbdba4a'
SAGE_SOURCE = f'https://github.com/thu-ml/SageAttention/blob/{SAGE_COMMIT}'
FLEX_SOURCE = 'https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/attention/flex_attention.py'
TURBO_SOURCE = 'https://arxiv.org/html/2412.08585v3'


def validate_requirements(requirements):
    """Reject malformed tool rules before any model can match."""
    if not isinstance(requirements, Mapping):
        raise ValueError('attention_requirements must be a mapping')
    if not requirements:
        return
    required = {'mask_kinds', 'head_dims', 'precisions', 'gqa', 'sources'}
    optional = {'padding', 'site_kinds', 'phases', 'causal_equal_lengths'}
    if not required <= set(requirements) or set(requirements) - required - optional:
        raise ValueError('attention_requirements has missing or unknown fields')
    for name, allowed in (('mask_kinds', MASK_KINDS), ('site_kinds', SITE_KINDS), ('phases', PHASES)):
        if name not in requirements:
            continue
        values = requirements[name]
        if (not isinstance(values, (list, tuple)) or not values
                or any(not isinstance(value, str) or value not in allowed for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'attention_requirements.{name} must contain distinct supported values')
    dims = requirements['head_dims']
    if (not isinstance(dims, (list, tuple)) or not dims
            or any(type(dim) is not int or dim < 1 for dim in dims) or len(set(dims)) != len(dims)):
        raise ValueError('attention_requirements.head_dims must contain distinct positive integers')
    for name in ('precisions', 'sources'):
        values = requirements[name]
        if (not isinstance(values, (list, tuple)) or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)):
            raise ValueError(f'attention_requirements.{name} must be a nonempty string sequence')
    if type(requirements['gqa']) is not bool:
        raise ValueError('attention_requirements.gqa must be a bool')
    if type(requirements.get('causal_equal_lengths', False)) is not bool:
        raise ValueError('attention_requirements.causal_equal_lengths must be a bool')
    padding = requirements.get('padding', {})
    if not isinstance(padding, Mapping):
        raise ValueError('attention_requirements.padding must be a mapping')
    for width, recipe in padding.items():
        if (not isinstance(width, str) or not width.isdecimal() or int(width) < 1
                or not isinstance(recipe, Mapping) or type(recipe.get('to')) is not int
                or recipe['to'] <= int(width) or recipe['to'] not in dims):
            raise ValueError('attention_requirements.padding must map source widths to supported larger widths')
        if any(not isinstance(recipe.get(name), str) or not recipe[name].strip()
               for name in ('scale', 'cost')):
            raise ValueError('attention_requirements.padding needs explicit scale and cost')
        if (not isinstance(recipe.get('sources'), (list, tuple)) or not recipe['sources']
                or any(not isinstance(source, str) or not source for source in recipe['sources'])):
            raise ValueError('attention_requirements.padding needs explicit sources')


def _dimension(value):
    """Symbolic batch/length dimensions are declarations; explicit unknowns are not."""
    return (type(value) is int and value > 0) or (isinstance(value, str) and bool(value.strip()))


def _check_site(site, requirements):
    reasons, adaptations = [], []
    if not isinstance(site, Mapping):
        return ['attention site must be an object'], adaptations
    for name in ('id', 'stage'):
        if not isinstance(site.get(name), str) or not site[name]:
            reasons.append(f'{name} is absent or unknown')
    for name in ('head_dim', 'query_heads', 'kv_heads'):
        if type(site.get(name)) is not int or site[name] < 1:
            reasons.append(f'{name} must be a known positive integer')
    if type(site.get('gqa')) is not bool:
        reasons.append('gqa must be a known bool')
    if reasons:
        return reasons, adaptations
    hd, qh, kh = site['head_dim'], site['query_heads'], site['kv_heads']
    if qh % kh or site['gqa'] != (qh != kh):
        reasons.append('query_heads/kv_heads/gqa geometry is inconsistent')
    if site['gqa'] and not requirements['gqa']:
        reasons.append('GQA is unsupported by this tool')
    mask = site.get('mask_kind')
    if not isinstance(mask, str) or mask not in MASK_KINDS:
        reasons.append('mask_kind must be known none/causal/arbitrary')
    elif mask not in requirements['mask_kinds']:
        reasons.append(f'mask_kind={mask} is unsupported; requires {requirements["mask_kinds"]}')
    precision = site.get('precision')
    if not isinstance(precision, str) or precision not in requirements['precisions']:
        reasons.append(f'precision must be known and in {requirements["precisions"]}')
    for name, allowed in (('site_kind', SITE_KINDS), ('phase', PHASES)):
        value = site.get(name)
        if not isinstance(value, str) or value not in allowed:
            reasons.append(f'{name} is absent, unknown or unsupported')
        elif value not in requirements.get('site_kinds' if name == 'site_kind' else 'phases', allowed):
            reasons.append(f'{name}={value} is unsupported by this tool')
    if hd not in requirements['head_dims']:
        recipe = requirements.get('padding', {}).get(str(hd))
        if recipe is None:
            reasons.append(f'head_dim={hd} is unsupported; requires {requirements["head_dims"]} or explicit padding')
        else:
            adaptations.append({
                'kind': 'head_dim_padding', 'from': hd, **deepcopy(dict(recipe)),
                'qkv_width_multiplier': recipe['to'] / hd,
                'extra_elements_per_row_per_tensor': recipe['to'] - hd,
                'status': 'requires_port_and_operator_gate',
            })
    sources = site.get('sources')
    if (not isinstance(sources, (list, tuple)) or not sources
            or any(not isinstance(source, str) or not source for source in sources)):
        reasons.append('site source provenance is absent or unknown')
    layout = site.get('layout')
    axes = LAYOUT_AXES.get(layout) if isinstance(layout, str) else None
    if axes is None:
        reasons.append('layout is absent, unknown or unsupported')
        return reasons, adaptations
    shapes = site.get('shapes')
    if (not isinstance(shapes, Mapping) or any(
            not isinstance(shapes.get(name), (list, tuple)) or len(shapes[name]) != 4
            or any(not _dimension(value) for value in shapes[name]) for name in ('q', 'k', 'v'))):
        reasons.append('Q/K/V shapes must declare four known or symbolic dimensions')
        return reasons, adaptations
    head_axis, length_axis = axes
    for name, heads in (('q', qh), ('k', kh), ('v', kh)):
        if type(shapes[name][head_axis]) is not int or shapes[name][head_axis] != heads:
            reasons.append(f'{name} shape contradicts declared logical head count')
        if type(shapes[name][-1]) is not int or shapes[name][-1] != hd:
            reasons.append(f'{name} shape contradicts declared head_dim')
    if (shapes['k'] != shapes['v'] or shapes['q'][0] != shapes['k'][0]):
        reasons.append('Q/K/V batch or KV shape declarations are inconsistent')
    if (mask == 'causal' and requirements.get('causal_equal_lengths')
            and shapes['q'][length_axis] != shapes['k'][length_axis]):
        reasons.append('causal attention requires declared equal Q and KV lengths')
    if layout != 'BHSD':
        adaptations.append({'kind': 'layout', 'from': layout, 'to': 'BHSD',
                            'cost': 'Account for transpose/stride handling and any contiguous copy at the real call site.'})
    if site.get('storage_kv_heads', kh) != kh:
        adaptations.append({'kind': 'logical_gqa',
                            'cost': 'Current storage expands KV heads; a logical-GQA port must prove compact-KV equivalence.'})
    return reasons, adaptations


def match_sites(spec, requirements, *, stages, stage=None, blocked_reason=''):
    """Return eligible and rejected rows, retaining all facts from the SAME site."""
    eligible, rejected = [], []
    declared_stages = {entry['stage_id'] for entry in spec['stages']}
    sites = spec.get('attention', [])
    for site in sites:
        if stage not in (None, 'kernel.attention') and (
                not isinstance(site, Mapping) or site.get('stage') != stage):
            continue
        reasons, adaptations = _check_site(site, requirements)
        if isinstance(site, Mapping):
            row = deepcopy(dict(site))
            if site.get('stage') not in declared_stages:
                reasons.append('site stage is absent from ModelSpec stages')
            if site.get('stage') not in stages:
                reasons.append('site stage is outside this tool scope')
            variants = site.get('precision_by_profile', {})
            if not isinstance(variants, Mapping):
                reasons.append('precision_by_profile must be a mapping')
        else:
            row = {'id': None, 'stage': None}
        if blocked_reason:
            reasons.append(blocked_reason)
        row.update(model=spec['model'], eligible=not reasons, adaptations=adaptations)
        if reasons:
            row['reasons'] = reasons
            rejected.append(row)
        else:
            if variants:
                row['unsupported_precision_profiles'] = [
                    profile for profile, values in variants.items()
                    if not isinstance(values, Mapping) or any(
                        values.get(name) not in requirements['precisions'] for name in ('q', 'k', 'v'))]
            row['runtime_validation_required'] = True
            eligible.append(row)
    return {'eligible_sites': eligible, 'ineligible_sites': rejected}
