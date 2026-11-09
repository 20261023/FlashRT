"""One explicit numerical quality contract shared by worker and controller."""

import math


def quality_contract(gate):
    gate = gate or {}
    return {'scope': gate.get('scope', 'numerical_replay'),
            'action_dim': gate.get('action_dim'),
            'gate': {key: gate[key] for key in ('max_relative_l2', 'max_abs', 'min_cosine') if key in gate}}


def validate_quality_contract(quality, gate):
    """Never trust a passed flag from a different budget/channel contract."""
    expected = quality_contract(gate)
    if any(quality.get(key) != value for key, value in expected.items()):
        raise ValueError('Worker quality contract differs from campaign quality budget/channels')
    if quality['status'] != 'passed':
        return
    for key, threshold in expected['gate'].items():
        value = quality['metrics'].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Missing/nonfinite metric for campaign quality budget: {key}')
        passed = value >= threshold if key == 'min_cosine' else 0 <= value <= threshold
        if not passed:
            raise ValueError(f'Passed worker result violates campaign quality budget: {key}')
