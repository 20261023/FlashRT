"""Deterministic conditional sampling search and quality-gated comparisons."""

from __future__ import annotations

from dataclasses import replace
import json
from numbers import Integral
from statistics import median

from flash_rt.sampling import ProbeFlowConfig

from .schema import Candidate, _text, decode_backend, encode_backend, validate_result


def enumerate_candidates(baseline: Candidate, *, euler_steps=(2, 5, 10),
                         epsilons=(0.002, 0.008), max_steps=20) -> list[Candidate]:
    """Vary denoising only; fixed steps and adaptive policy are never mixed.

    Order is numeric Euler steps followed by numeric ProbeFlow epsilons, not
    caller iteration order. IDs are readable extensions of the immutable parent
    ID. The unchanged parent source/options remain part of every explicit point.
    """
    if not isinstance(baseline, Candidate):
        raise TypeError('baseline must be a Candidate')
    steps = []
    for value in euler_steps:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError('euler_steps must contain positive integers')
        steps.append(int(value))
    policy = replace(baseline.backend.probeflow or ProbeFlowConfig(), max_steps=max_steps)
    policies = [replace(policy, epsilon=epsilon) for epsilon in epsilons]
    points = []
    base_record = encode_backend(baseline.backend)
    for count in sorted(set(steps)):
        backend = decode_backend({**base_record, 'sampler': 'euler',
                                  'num_steps': count, 'probeflow': None})
        points.append((f'euler-steps-{count}', backend,
                       f'Use {count} fixed Euler evaluations'))
    for epsilon in sorted({float(point.epsilon) for point in policies}):
        adaptive = replace(policy, epsilon=epsilon)
        # The codec also detaches nested options from the baseline/other points.
        backend = decode_backend(encode_backend(replace(
            baseline.backend, sampler='probeflow', num_steps=None, probeflow=adaptive)))
        points.append((f'probeflow-epsilon-{epsilon}-max-{max_steps}', backend,
                       f'Adapt evaluations with ProbeFlow epsilon={epsilon}, max_steps={max_steps}; '
                       'measure actual NFE separately from scheduled steps'))
    return [Candidate(
        candidate_id=f'{baseline.candidate_id}-{suffix}', backend=backend,
        workload_id=baseline.workload_id, parent_id=baseline.candidate_id,
        hypothesis=f'{hypothesis}; unchanged model, action head and prefix execution.',
        origin='search', patch_ref=baseline.patch_ref, source_ref=baseline.source_ref,
    ) for suffix, backend, hypothesis in points]


def select_winner(records: list[dict], *, quality_scope: str) -> str | None:
    """Return the fastest measured median with passed quality in exactly scope.

    All supplied measurements, including those with unknown/failed quality,
    must belong to one comparison group. GPU dictionaries are compared without
    the worker's per-submission ``job_id`` (key order is immaterial): callers
    must otherwise provide stable device identity, not transient telemetry.
    Action domain lives in
    ``backend_metadata.action_domain``. Numerical replay is never promoted to
    task success. Failed worker records are validated but cannot be ranked.
    """
    _text(quality_scope, 'quality_scope', nonempty=True)
    comparison = None
    group = None
    eligible = []
    for record in records:
        validate_result(record)
        record_comparison = (record['workload_id'], record['comparison_id'])
        if comparison is None:
            comparison = record_comparison
        elif comparison != record_comparison:
            raise ValueError('Inconsistent comparison groups: workload/comparison mismatch')
        if record['state'] != 'measured':
            continue
        gpu = {key: value for key, value in record['gpu'].items() if key != 'job_id'}
        if not gpu:
            raise ValueError('Comparison requires nonempty gpu identity')
        domain = record['backend_metadata'].get('action_domain')
        _text(domain, 'action_domain', nonempty=True)
        identity = (record['workload_id'], record['comparison_id'],
                    json.dumps(gpu, sort_keys=True, separators=(',', ':'), allow_nan=False), domain)
        if group is None:
            group = identity
        elif group != identity:
            raise ValueError('Inconsistent comparison groups: workload/comparison/gpu/action_domain mismatch')
        quality = record['quality']
        if quality['status'] == 'passed' and quality['scope'] == quality_scope:
            eligible.append((median(record['measurement']['latency_ms']),
                             record['candidate_id'], record['attempt_id']))
    return min(eligible)[1] if eligible else None
