"""Evidence-driven VLA outer-loop policy, deliberately OUTSIDE the pinned harness.

CPU only: read ingested controller summaries, recompute attribution from a worker
profile, consult the live feature-based tool registry, and record ONE next action.
Never submits a job, measures a model, accepts a candidate, or invents a winner.
See .agents/skills/vla-autotune/references/next-round.md for the history format.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import shlex
from statistics import median
import sys


# Source-checkout example, not a worker import-path override.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_rt.vla.autotune import decisions, stages
from flash_rt.vla.autotune.controller import _validate_checkout_candidate
from flash_rt.vla.autotune.schema import Candidate, validate_result
from flash_rt.vla.autotune.tools import all_tools, get_tool
from flash_rt.vla.model_spec import get_spec


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}
HARD_EXCLUSIONS = {'blocked', 'not_optimization', 'recorded_exhaustion',
                   'method_budget_exhausted', 'three_direction_rounds_below_2_percent',
                   'no_distinct_config_candidates'}


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return float(value)


def _load(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda x: _invalid_json(x))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def _invalid_json(value):
    raise ValueError(f'nonfinite JSON value: {value}')


def _resolve(value, root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('evidence paths must be nonempty strings')
    path = Path(value)
    return (root / path if not path.is_absolute() else path).resolve()


def _canonical(tool):
    try:
        return get_tool(tool).tool_id
    except ValueError:
        # Unknown source tools remain visible evidence, never masquerade as config.
        return tool


def _gpu(record):
    return {key: value for key, value in record.get('gpu', {}).items() if key != 'job_id'}


def _scope(record, plan):
    """Only compare within checkpoint/input/device/gate/timing scope, not across models."""
    backend = record['candidate']['backend']
    return {
        'model': backend['model'], 'checkpoint': backend['checkpoint'],
        'hardware': backend['hardware'], 'framework': backend['framework'],
        'workload_id': record['workload_id'],
        'workload_identity': record.get('workload_identity'),
        'workload_request_ids': record.get('workload_request_ids'),
        'runtime': record.get('runtime'),
        'precision': backend.get('precision'),
        'gpu': _gpu(record), 'quality_gate': plan.get('quality_gate'),
        'action_domain': record.get('backend_metadata', {}).get('action_domain'),
        'timing_boundary': record.get('backend_metadata', {}).get(
            'measurement_boundary', 'worker_completed_infer'),
        'harness_tree': plan.get('harness', {}).get('autotune_tree'),
    }


def _qualified(attempt, scope, has_author):
    result = attempt.get('result')
    return (
        attempt['state'] == 'validated' and result is not None
        and result['state'] == 'measured'
        and result['quality']['status'] == 'passed' and result['quality']['scope'] == scope
        and (not has_author or result.get('author_quality', {}).get('status') == 'passed'))


def _read_round(path):
    summary = _load(path)
    # Controller's plan supplies roles and immutable constructor identity. The
    # summary alone has no role field; guessing by list order is unnecessary.
    plan = _load(path.with_name('plan.json'))
    if plan.get('schema_version') not in (2, 3):
        raise ValueError(f'{path}: use an identity-bound plan v2/v3, not legacy latency notes')
    if summary.get('comparison_id') != plan.get('comparison_id'):
        raise ValueError(f'{path}: summary/plan comparison mismatch')
    attempts = summary.get('attempts')
    entries = plan.get('entries')
    if not isinstance(attempts, list) or not isinstance(entries, list) or len(entries) < 2:
        raise ValueError(f'{path}: controller attempts and reference/bookend plan required')
    by_id = {item['attempt_id']: item for item in attempts}
    if len(by_id) != len(attempts) or set(by_id) != {item['attempt_id'] for item in entries}:
        raise ValueError(f'{path}: duplicate or unbound attempts')
    controls, candidates = {}, []
    scope = summary.get('quality_scope')
    for entry in entries:
        attempt = by_id[entry['attempt_id']]
        result = attempt.get('result')
        if result is not None:
            validate_result(result)
            if (result.get('schema_version') not in (2, 3)
                    or result['candidate'] != entry['candidate']
                    or result['campaign_instance_id'] != plan['campaign_instance_id']
                    or result['comparison_id'] != plan['comparison_id']
                    or result['attempt_id'] != entry['attempt_id']
                    or result['job_id'] != attempt['job_id']):
                raise ValueError(f'{path}: result execution identity mismatch')
        if entry['role'] in ('reference', 'bookend'):
            if entry['role'] in controls:
                raise ValueError(f'{path}: duplicate reference/bookend')
            controls[entry['role']] = attempt
        elif entry['role'] == 'candidate':
            candidates.append((entry, attempt))
        else:
            raise ValueError(f'{path}: unknown controller role')
    if set(controls) != {'reference', 'bookend'}:
        raise ValueError(f'{path}: reference and bookend required')
    reference = controls['reference'].get('result')
    baseline = next(entry['candidate'] for entry in entries if entry['role'] == 'reference')
    source_qualified = True
    if plan['schema_version'] == 2:
        reference_candidate = Candidate.from_dict(baseline)
        for entry in entries:
            point = Candidate.from_dict(entry['candidate'])
            try:
                _validate_checkout_candidate(point, reference_candidate)
            except ValueError:
                source_qualified = False
                break
    has_author = 'author_baseline' in plan
    qualified = lambda attempt: _qualified(attempt, scope, has_author)
    reliable = source_qualified and summary.get('complete') is True and all(qualified(x) for x in controls.values())
    drift = base_ms = None
    if reliable:
        base_ms = median(reference['measurement']['latency_ms'])
        end_ms = median(controls['bookend']['result']['measurement']['latency_ms'])
        if base_ms <= 0:
            reliable = False
        else:
            drift = end_ms / base_ms - 1
            stored_drift = summary.get('baseline_drift_ratio')
            if stored_drift is None or abs(drift - stored_drift) > 1e-9:
                raise ValueError(f'{path}: stored bookend drift differs from measurements')
            reliable = abs(drift) <= .10
    winner = summary.get('winner')
    selected = [item['result'] for item in attempts
                if item['candidate_id'] == winner and qualified(item)]
    if winner is not None and not selected:
        raise ValueError(f'{path}: controller winner is not quality-validated')
    incumbent = selected[0] if selected else reference
    # A lower bookend of the SAME baseline is noise, not an optimization gain.
    raw_gain = (0.0 if winner == baseline['candidate_id'] else
                1 - median(incumbent['measurement']['latency_ms']) / base_ms
                ) if reliable and selected else None
    noise = max(.015, abs(drift)) if drift is not None else None
    reduction = (raw_gain if raw_gain > noise else 0.0) if raw_gain is not None else None
    if not candidates:
        reduction = raw_gain = None
    directions = defaultdict(list)
    for entry, attempt in candidates:
        tool = entry['candidate'].get('tool')
        if tool:
            directions[_canonical(tool)].append(attempt)
    direction_gains = {}
    for tool, group in directions.items():
        valid = [median(x['result']['measurement']['latency_ms']) for x in group if qualified(x)]
        # This is best qualified direction evidence, NOT another elected winner.
        # Measured quality failures establish no usable gain; launch failures do not.
        observed = any(x.get('result', {}).get('state') == 'measured'
                       for x in group if x.get('result') is not None)
        gain = max(0.0, 1 - min(valid) / base_ms) if reliable and valid else (
            0.0 if reliable and observed else None)
        direction_gains[tool] = (gain if gain > noise else 0.0) if gain is not None else None
    # Noise >= the direction stop threshold cannot establish <2% stagnation.
    convergence_ok = reliable and drift is not None and abs(drift) < .02
    identity_record = reference or {'candidate': baseline, 'workload_id': baseline['workload_id']}
    return {
        'path': str(path), 'summary': summary, 'plan': plan, 'baseline': baseline,
        'incumbent': incumbent, 'scope': _scope(identity_record, plan),
        'key': (plan['comparison_id'], plan['campaign_instance_id']),
        'candidate_count': len(candidates), 'directions': dict(directions),
        'evidence': {
            'summary': str(path), 'comparison_id': plan['comparison_id'], 'winner': winner,
            'reference_median_ms': base_ms, 'baseline_drift_ratio': drift,
            'noise_floor_ratio': noise, 'raw_latency_reduction_ratio': raw_gain,
            'latency_reduction_ratio': reduction,
            'direction_reduction_ratios': direction_gains,
            'source_comparison_qualified': source_qualified,
            'reliability': ('measured' if reliable else 'unreliable_bookends'),
            'convergence_usable': convergence_ok,
        }}


def _accounting(entries, root):
    """Read normalized scheduler observations, never forecast time from latency."""
    if not isinstance(entries, list) or not entries:
        return None, False
    jobs, pending = {}, False
    for entry in entries:
        try:
            path = _resolve(entry['evidence_ref'], root)
            job = _load(path)
            job_id = job['job_id']
            state = job['state']
            if not isinstance(job_id, str) or not job_id.strip() or not isinstance(state, str):
                raise ValueError('accounting job_id/state must be text')
            elapsed = _number(job['elapsed_seconds'], 'elapsed_seconds')
            gpus = job['gpus']
            if type(gpus) is not int or gpus < 1:
                raise ValueError('accounting gpus must be a positive integer')
            normalized = {'job_id': job_id, 'state': state, 'elapsed_seconds': elapsed,
                          'gpus': gpus, 'evidence_ref': str(path)}
            if job_id in jobs:
                raise ValueError('duplicate job accounting in one entry')
            jobs[job_id] = normalized
            pending |= state not in TERMINAL
        except (OSError, KeyError, ValueError, TypeError):
            return None, False
    return jobs, pending


def _history(path, current):
    manifest = _load(path)
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('rounds'), list):
        raise ValueError('history requires schema_version=1 and chronological rounds')
    rounds, seen, costs, unknown_cost, pending = [], set(), defaultdict(dict), False, False
    actors = Counter()
    for entry in manifest['rounds']:
        item = _read_round(_resolve(entry['summary'], path.parent))
        if item['key'] in seen:
            raise ValueError('duplicate campaign instance in history')
        seen.add(item['key'])
        actor = entry.get('actor')
        if actor not in decisions.ACTORS:
            raise ValueError('each historical round needs its real actor')
        item['evidence']['actor'] = actor
        item['comparable'] = item['scope'] == current['scope']
        rounds.append(item)
        # A model-specific history cannot silently charge another model/hardware.
        if (item['scope']['model'], item['scope']['hardware']) != (
                current['scope']['model'], current['scope']['hardware']):
            raise ValueError('history must be per-model and per-hardware')
        actors[actor] += 1
        jobs, running = _accounting(entry.get('jobs'), path.parent)
        pending |= running
        expected_jobs = {attempt['job_id'] for attempt in item['summary']['attempts']
                         if attempt.get('job_id') is not None}
        if jobs is None or not expected_jobs or not expected_jobs <= set(jobs):
            unknown_cost = True
            continue
        for tool in item['directions'] or {'profile.stage_intervals': []}:
            for job_id, job in jobs.items():
                old = costs[tool].get(job_id)
                if old is not None and old != job:
                    raise ValueError('conflicting job evidence for one method')
                costs[tool][job_id] = job
    if current['key'] not in seen:
        # Do not invent an actor or zero GPU hours for an unrecorded latest round.
        rounds.append({**current, 'comparable': True})
        unknown_cost = True
    elif not rounds or rounds[-1]['key'] != current['key']:
        raise ValueError('--summary must be the latest round in history')
    for entry in manifest.get('method_jobs', []):
        # Cold compile, calibration, failed setup and external search allocations.
        # Their cost counts even when there is no completed model campaign.
        actor = entry.get('actor')
        if actor not in decisions.ACTORS:
            raise ValueError('each preparation entry needs its real actor')
        tool = _canonical(entry['tool'])
        jobs, running = _accounting(entry.get('jobs'), path.parent)
        pending |= running
        if jobs is None:
            unknown_cost = True
            continue
        for job_id, job in jobs.items():
            old = costs[tool].get(job_id)
            if old is not None and old != job:
                raise ValueError('conflicting preparation/campaign job evidence')
            costs[tool][job_id] = job
    exhausted = {}
    for entry in manifest.get('exhausted_tools', []):
        tool = _canonical(entry['tool'])
        if not isinstance(entry.get('reason'), str) or not entry['reason'].strip():
            raise ValueError('exhausted tools require a reason')
        ref = _resolve(entry['evidence_ref'], path.parent)
        if not ref.is_file():
            raise ValueError(f'exhaustion evidence missing: {ref}')
        exhausted[tool] = {'reason': entry['reason'], 'evidence_ref': str(ref)}
    hours = {tool: math.fsum(job['gpus'] * job['elapsed_seconds'] / 3600
                             for job in jobs.values()) for tool, jobs in costs.items()}
    return rounds, hours, exhausted, unknown_cost, pending, dict(actors)


def _profile(record, current, profile_path):
    incumbent = current['incumbent']
    empty = {'candidate_id': None, 'attributed_fraction': None, 'groups': [],
             'unattributed_ms': None, 'coverage': 'none'}
    if record is None or incumbent is None:
        return empty, 'missing_profile'
    # A separate diagnostic must represent the exact current incumbent, inputs,
    # device and boundary, not an older baseline or an eager/sampler substitution.
    identity = ('candidate', 'workload_id', 'workload_identity', 'workload_request_ids',
                'runtime', 'output_domain')
    if (any(record.get(key) != incumbent.get(key) for key in identity)
            or _gpu(record) != _gpu(incumbent)
            or record.get('backend_metadata', {}).get('action_domain')
            != incumbent.get('backend_metadata', {}).get('action_domain')
            or record.get('state') != 'measured'):
        return empty, 'profile_identity_mismatch'
    try:
        value = stages.attribute(record)
    except ValueError as error:
        return {**empty, 'error': str(error)}, 'invalid_profile_accounting'
    value.update(candidate_id=record['candidate_id'], evidence_ref=profile_path)
    fraction = value['attributed_fraction']
    if fraction is None:
        return value, 'missing_profile'
    if fraction > 1 or value['unattributed_ms'] < 0:
        return value, 'invalid_profile_accounting'
    if fraction < .90 or value['coverage'] != 'full':
        return value, 'profile_coverage_below_90_percent'
    # ≥90% means separated additive wall intervals, not "full" compute coverage.
    if any(len(group['stages']) != 1 and group['ms'] > 0 for group in value['groups']):
        return value, 'unsplit_profile_groups'
    return value, None


def _stalled(rounds, tool=None):
    values = []
    for item in rounds:
        evidence = item['evidence']
        if tool is not None:
            if tool not in item['directions']:
                continue
            value = evidence['direction_reduction_ratios'][tool]
        else:
            if not item['candidate_count']:
                continue
            value = evidence['latency_reduction_ratio']
        if not item['comparable'] or not evidence['convergence_usable']:
            value = None
        values.append(value)
    count, threshold = (3, .02) if tool else (2, .03)
    return len(values) >= count and all(
        value is not None and value < threshold for value in values[-count:])


def _operator_regions(path, record, profile, model, hardware, spec_dir):
    """Measure each call site's disjoint contribution on a bound request trace."""
    if path is None:
        return {}
    evidence = _load(path)
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    if (evidence.get('schema_version') != 1
            or evidence.get('profile_sha256') != hashlib.sha256(encoded).hexdigest()):
        raise ValueError('Operator timeline does not match the current worker profile')
    if not isinstance(evidence.get('regions'), list):
        raise ValueError('Operator timeline requires a regions list')
    wall = profile['wall_ms']
    stage_ms = defaultdict(float)
    for group in profile['groups']:
        if len(group['stages']) == 1:
            stage_ms[group['stages'][0]] += group['ms']
    scopes = defaultdict(list)
    for row in evidence['regions']:
        tool = get_tool(row['tool'])
        stage, site = row['stage'], row['site']
        if not isinstance(site, str) or not site.strip():
            raise ValueError('Operator timeline requires a named call site')
        if tool.attention_requirements:
            eligible = tool.attention_sites(model, spec_dir=spec_dir, stage=stage, hardware=hardware)
            if not any(item['id'] == site and item['stage'] == stage
                       for item in eligible['eligible_sites']):
                raise ValueError(f'{site!r} is not an eligible attention site for {tool.tool_id} at {stage}')
        if stage not in stage_ms:
            raise ValueError(f'Operator timeline requires a measured enclosing stage for {stage}')
        spans = row['critical_path_spans_ms']
        if not isinstance(spans, list) or not spans:
            raise ValueError('Operator timeline requires nonempty critical-path spans')
        spans = sorted((_number(start, 'start'), _number(end, 'end', positive=True))
                       for start, end in spans)
        last_end, duration = 0.0, 0.0
        for start, end in spans:
            if not start < end <= wall:
                raise ValueError('Operator span lies outside the profiled request')
            duration += max(0.0, end - max(last_end, start))
            last_end = max(last_end, end)
        if duration > stage_ms[stage] + 1e-6:
            raise ValueError(f'Operator span union exceeds its enclosing stage {stage}')
        scopes[tool.tool_id].append((duration / wall, stage, site))
    return {tool_id: max(rows) for tool_id, rows in scopes.items()}


def _table(current, profile, operator_regions, rounds, hours, exhausted, budget, spec_dir):
    baseline = Candidate.from_dict(current['incumbent']['candidate'] if current['incumbent']
                                   else current['baseline'])
    model, hardware = baseline.backend.model, baseline.backend.hardware
    per_stage = defaultdict(float)
    for group in profile['groups']:
        if len(group['stages']) == 1:
            per_stage[group['stages'][0]] += group['ms'] / profile['wall_ms']
    table = []
    for tool in all_tools():
        matches, reason = tool.match_model(model, spec_dir=spec_dir)
        status = tool.status_for(model, spec_dir=spec_dir)
        constraint = (tool.hardware_reason(hardware, model, spec_dir=spec_dir)
                      or tool.configuration_reason(baseline.backend))
        if not matches or constraint:
            status, reason = 'blocked', reason if not matches else constraint
        row = {
            'tool_id': tool.tool_id, 'lane': tool.lane,
            'stages': list(tool.stages_for(model, spec_dir=spec_dir)),
            'status': status, 'reason': reason or tool.reason_for(model, spec_dir=spec_dir),
            'model_predicate': dict(tool.model_predicate), 'exactness': tool.exactness,
            'skills': list(tool.skills), 'gpu_hours': hours.get(tool.tool_id, 0.0),
            'remaining_gpu_hours': max(0.0, budget - hours.get(tool.tool_id, 0.0)),
            'share': None, 'stage': None, 'exclusion': None,
            'affected_share': None,
            'share_interpretation': 'measured enclosing region, NOT promised removable work',
        }
        if status == 'blocked':
            row['exclusion'] = 'blocked'
        elif tool.lane in ('baseline', 'analysis') or tool.exactness == 'n/a':
            row['exclusion'] = 'not_optimization'
        elif tool.tool_id in exhausted:
            row['exclusion'] = 'recorded_exhaustion'
            row['exhaustion_evidence'] = exhausted[tool.tool_id]
        elif row['gpu_hours'] >= budget:
            row['exclusion'] = 'method_budget_exhausted'
        elif _stalled(rounds, tool.tool_id):
            row['exclusion'] = 'three_direction_rounds_below_2_percent'
        elif tool.tool_id.startswith(('kernel.', 'quant.', 'attention.')) and tool.tool_id not in operator_regions:
            # Broad registry stage labels are not operator occurrence timings.
            # Non-additive probes and summed kernel times cannot substitute.
            row['exclusion'] = 'needs_operator_or_launch_attribution'
        else:
            mapped = set(row['stages'])
            if 'denoise.schedule' in mapped:
                mapped.add('denoise.velocity')
            available = [(share, stage) for stage, share in per_stage.items() if stage in mapped]
            if tool.tool_id in operator_regions:
                share, stage, site = operator_regions[tool.tool_id]
                available = [(share, stage)]
                row.update(affected_share=share, call_site=site,
                           share_interpretation='union of observed critical-path call-site spans')
            if available:
                row['share'], row['stage'] = max(available)
                row['ideal_region_latency_reduction_ceiling'] = row['share']
                row['ideal_region_speedup_ceiling'] = (
                    1 / (1 - row['share']) if row['share'] < 1 else None)
                if row['share'] < .05:
                    row['exclusion'] = 'share_below_5_percent'
            else:
                row['exclusion'] = 'needs_operator_or_launch_attribution'
            if row['exclusion'] is None and tool.propose is not None and tool.lane == 'config':
                try:
                    proposals = tool.propose(baseline, **dict(tool.parameters))
                except (ValueError, TypeError) as error:
                    row['exclusion'] = 'needs_explicit_method_parameters'
                    row['reason'] = str(error)
                else:
                    if not proposals:
                        row['exclusion'] = 'no_distinct_config_candidates'
        table.append(row)
    return table


def _action(action, reason, rationale, **fields):
    return {'action': action, 'reason_code': reason, 'rationale': rationale, **fields}


def recommend(summary_path, history_path, *, profile_path=None, operator_profile_path=None,
              method_budget_gpuh=6, spec_dir=None):
    """Return a reproducible recommendation; only ``main`` persists its decision."""
    budget = _number(method_budget_gpuh, 'method budget', positive=True)
    if budget > 6:
        raise ValueError('method budget cannot exceed 6 GPUh')
    summary_path, history_path = Path(summary_path).resolve(), Path(history_path).resolve()
    current = _read_round(summary_path)
    rounds, hours, exhausted, unknown_cost, pending, actors = _history(history_path, current)
    baseline = current['baseline']['backend']
    spec_dir = spec_dir or baseline.get('spec_dir')
    spec = get_spec(baseline['model'], spec_dir=spec_dir)
    if profile_path is not None:
        profile_path = str(Path(profile_path).resolve())
        try:
            record = _load(profile_path)
        except (OSError, ValueError):
            record = None
    else:
        record = current['incumbent']
        profile_path = f'{summary_path}#incumbent.stage_metrics'
    profile, gap = _profile(record, current, profile_path)
    regions = (_operator_regions(operator_profile_path, record, profile, baseline['model'],
                                 baseline['hardware'], spec_dir)
               if gap is None else {})
    table = _table(current, profile, regions, rounds, hours, exhausted, budget, spec_dir)
    eligible = [row for row in table if row['exclusion'] is None]
    # Largest MEASURED region first, then ready/exact over unimplemented/approximate.
    # This is a planning tie-break, not a forecast of which candidate will win.
    eligible.sort(key=lambda row: (-row['share'],
                                   {'ready': 0, 'needs_setup': 1, 'needs_implementation': 2}
                                   .get(row['status'], 3),
                                   row['exactness'] != 'exact', row['tool_id']))
    if pending or current['summary'].get('complete') is not True:
        chosen = _action('wait', 'job_not_terminal',
                         'Wait for the same job to end naturally; do not resubmit or add a deadline.')
    elif not current['evidence']['source_comparison_qualified']:
        chosen = _action('acquire_source_evidence', 'source_requires_plan_v3',
                         'Measure the reference, candidates and bookend with ready snapshots in a schema 3 campaign.')
    elif unknown_cost:
        chosen = _action('acquire_accounting', 'unknown_gpu_cost',
                         'Fetch terminal scheduler accounting and record the real actor for every round/setup job.')
    elif current['evidence']['reliability'] != 'measured':
        chosen = _action('acquire_round_evidence', 'unreliable_bookends',
                         'Repeat a reference/bookend campaign; no reliable win or stagnation is established.')
    elif gap:
        chosen = _action('acquire_profile', gap,
                         'Acquire disjoint current-incumbent wall attribution with >=90% coverage; '
                         'keep residual/probes separate and verify graph diagnostic parity.',
                         observation_tool='profile.stage_intervals')
    elif _stalled(rounds):
        chosen = _action('stop', 'two_rounds_below_3_percent',
                         'The latest two comparable, noise-resolved optimization rounds each reduced latency <3%.')
    elif eligible:
        first = eligible[0]
        requirements = [
            'Read tools show; preserve registered lane and all source/snapshot provenance.',
            'Open a model/stage/tool token segment before work; record the real actor.',
            'Size finite work from measured pilot/setup cost within remaining GPUh before submission; '
            'unknown duration requires a pilot, not an invented estimate.',
            'Wait for natural job termination; no walltime, process timeout or waiting deadline.',
            'A recommendation is not an accepted winner; controller quality and H2/H3/H4 still apply.',
        ]
        if first['tool_id'].startswith('compile.'):
            requirements.append(
                'Measure cold compile in a fresh empty private compiler cache; report compile/setup '
                'seconds separately from warm replay, count its full allocation, and compute break-even '
                'only from measured same-boundary positive savings.')
        chosen = _action(
            'select_tool', 'largest_measured_applicable_region',
            'Ranked by observed enclosing-region share, availability and exactness; no local speedup assumed.',
            tool=first['tool_id'], stage=first['stage'], lane=first['lane'], status=first['status'],
            share=first['share'], share_interpretation=first['share_interpretation'],
            ideal_region_latency_reduction_ceiling=first['share'],
            predicted_latency_ms=None, remaining_gpu_hours=first['remaining_gpu_hours'],
            requirements=requirements,
            **{key: first[key] for key in ('call_site', 'affected_share')
               if key in first and first[key] is not None})
    else:
        remaining = [row for row in table if row['exclusion'] not in HARD_EXCLUSIONS]
        if not remaining:
            chosen = _action('stop', 'applicable_tools_exhausted',
                             'All applicable optimization routes are blocked, evidenced exhausted, '
                             'budget-limited, stalled or already the current configuration.')
        elif all(row['exclusion'] == 'share_below_5_percent' for row in remaining):
            chosen = _action('stop', 'remaining_ceilings_below_5_percent',
                             'Every remaining measured region has an ideal latency-reduction ceiling <5%.')
        else:
            chosen = _action('acquire_profile', 'missing_method_scope',
                             'A remaining cross-cutting operator/launch/precision scope is unknown. '
                             'Acquire occurrence-level nonoverlapping attribution, not an assumed whole-stage share.',
                             observation_tool='profile.stage_intervals')
    return {
        'schema_version': 1, 'model': spec['model'], 'hardware': baseline['hardware'],
        'workload_id': current['baseline']['workload_id'], 'actor': 'agent',
        'scope': current['scope'], 'features': spec.get('features', {}),
        'inputs': {'summary': str(summary_path), 'history': str(history_path), 'profile': profile_path,
                   'operator_profile': str(operator_profile_path) if operator_profile_path else None},
        'policy': {'method_budget_gpuh': budget, 'min_stage_share': .05,
                   'minimum_profile_coverage': .90, 'direction_rounds': 3,
                   'direction_min_reduction': .02, 'model_rounds': 2, 'model_min_reduction': .03},
        'submission_policy': {'automatic_submission': False, 'natural_job_end': True,
                              'walltime': None, 'process_timeout': None, 'waiting_deadline': None,
                              'budget_admission_is_not_a_runtime_guarantee': True},
        'measured_round': current['evidence'], 'round_evidence': [x['evidence'] for x in rounds],
        'actor_attribution': actors, 'profile': profile, 'tool_table': table,
        'eligible_directions': eligible, 'chosen_action': chosen,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, help='Controller summary.json, alongside plan.json')
    parser.add_argument('--profile', help='Current incumbent worker result.json (default: embedded result)')
    parser.add_argument('--operator-profile', help='Bound critical-path call-site timeline JSON')
    parser.add_argument('--history', required=True, help='Per-model history, normalized scheduler evidence refs')
    parser.add_argument('--spec-dir', help='External ModelSpec directory; no model-name switch in this script')
    parser.add_argument('--method-budget-gpuh', type=float, default=6)
    parser.add_argument('--decision-log', required=True)
    parser.add_argument('--out', required=True, help='New report path; refuses overwrite/repeated decision')
    args = parser.parse_args(argv)
    try:
        output = Path(args.out).resolve()
        decision_path = output.with_suffix(output.suffix + '.decision.json')
        if output.exists() or decision_path.exists():
            raise ValueError('output already exists; use a new round/action path, not a duplicate decision')
        result = recommend(args.summary, args.history, profile_path=args.profile,
                           operator_profile_path=args.operator_profile,
                           method_budget_gpuh=args.method_budget_gpuh, spec_dir=args.spec_dir)
        chosen = result['chosen_action']
        # The existing decisions add CLI is the only log writer. No new schema,
        # fabricated accept/merge, actor override, or direct JSONL append here.
        payload = decisions.build_fields(
            'decision', actor='agent', summary=f'Codex outer loop: {chosen["action"]} ({chosen["reason_code"]})',
            model=result['model'], campaign=result['measured_round']['comparison_id'],
            action=('select_tool' if chosen['action'] == 'select_tool' else
                    'stop' if chosen['action'] == 'stop' else 'keep'),
            tool=chosen.get('tool'), stage=chosen.get('stage'),
            rationale=chosen['rationale'], evidence_refs=[
                 str(output), result['inputs']['summary'], result['inputs']['history'],
                 *([result['inputs']['operator_profile']] if args.operator_profile else [])])
        decisions.DecisionLog(args.decision_log).records()  # Refuse a bad log before writing artifacts.
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        with decision_path.open('x') as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        command = ['add', '--log', args.decision_log, '--json', str(decision_path)]
        with redirect_stdout(io.StringIO()):
            status = decisions.main(command)
        if status:
            raise ValueError('decisions add failed; report is not a recorded action')
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    print(json.dumps({'report': str(output), 'chosen_action': result['chosen_action'],
                      'decision_command': shlex.join(
                          [sys.executable, '-m', 'flash_rt.vla.autotune.decisions', *command])},
                     indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
