"""One explicit numerical quality contract shared by worker and controller."""

import math

import numpy as np


OUTPUT_DOMAINS = {
    'pi0': 'normalized-padded-model-actions',
    'pi05': 'normalized-padded-model-actions',
    'hyvla': 'normalized-padded-model-actions',
    'fastwam': 'normalized-model-actions-batched',
    'dm05': 'denormalized-absolute-dos-w1-14d',
}


def output_domain(model):
    """Domain of the arrays retained by the experiment worker, before slicing."""
    return OUTPUT_DOMAINS[model]


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


def compare_outputs(actions, request_ids, references, reference_ids, gate, *,
                    reference_origin='internal', domain=None):
    """Compare every sample in the same output domain, retaining request identity.

    Repeated reference requests are matched by occurrence, cycling only if the
    candidate has more samples. No request or nonfinite reference is discarded.
    """
    if not actions or len(actions) != len(request_ids) or len(references) != len(reference_ids):
        raise ValueError('Quality comparison requires complete nonempty output samples')
    by_request = {}
    for action, request_id in zip(references, reference_ids):
        value = np.asarray(action)
        if not value.size or not np.isfinite(value).all():
            raise ValueError('Reference returned empty/nonfinite actions')
        by_request.setdefault(request_id, []).append(value)
    if set(request_ids) != set(by_request):
        raise ValueError('Reference outputs do not cover the same request IDs')
    errors, absolutes, cosines, occurrences = [], [], [], {}
    action_dim = (gate or {}).get('action_dim')
    for actual, request_id in zip(actions, request_ids):
        index = occurrences.get(request_id, 0)
        expected = by_request[request_id][index % len(by_request[request_id])]
        occurrences[request_id] = index + 1
        actual = np.asarray(actual)
        if actual.shape != expected.shape or not actual.size or not np.isfinite(actual).all():
            raise ValueError('Reference outputs do not have the same finite action shapes')
        if action_dim is not None:
            if type(action_dim) is not int or not 1 <= action_dim <= actual.shape[-1]:
                raise ValueError('quality action_dim must select valid trained/public channels')
            actual, expected = actual[..., :action_dim], expected[..., :action_dim]
        expected = expected.astype(np.float64).ravel()
        observed = actual.astype(np.float64).ravel()
        delta = observed - expected
        norm = np.linalg.norm(expected)
        errors.append(float(np.linalg.norm(delta) / max(norm, 1e-12)))
        absolutes.append(float(np.abs(delta).max()))
        denominator = norm * np.linalg.norm(observed)
        cosine = (float(np.dot(expected, observed) / denominator) if denominator
                  else float(np.array_equal(expected, observed)))
        cosines.append(max(-1.0, min(1.0, cosine)))
    metrics = {'max_relative_l2': max(errors), 'max_abs': max(absolutes), 'min_cosine': min(cosines)}
    contract = quality_contract(gate)
    passed = all(metrics[key] >= value if key == 'min_cosine' else metrics[key] <= value
                 for key, value in contract['gate'].items())
    return {'status': ('passed' if passed else 'failed') if contract['gate'] else 'unknown',
            **contract, 'metrics': metrics, 'reference_origin': reference_origin,
            'output_domain': domain,
            'interpretation': 'Fixed-input numerical proxy only; not closed-loop task success'}
