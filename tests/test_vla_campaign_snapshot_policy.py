"""Controller admission rules: source variants cannot fall back to the job checkout."""

from dataclasses import replace
import json

import pytest

from flash_rt.vla import BackendConfig
from flash_rt.vla.autotune.controller import initialize_campaign, main as controller_main
from flash_rt.vla.autotune.ledger import CampaignLedger
from flash_rt.vla.autotune.schema import Candidate
from flash_rt.vla.autotune.search import enumerate_candidates
from flash_rt.vla.autotune.tools.levers import price_levers


def baseline(**changes):
    return Candidate('base', BackendConfig(model='pi05', checkpoint='/weights', num_steps=10),
                     'frames', **changes)


def initialize(root, base, candidates):
    return initialize_campaign(root, comparison_id='r1', baseline=base, candidates=candidates,
                               workload_path='inputs.json', output_root='results')


@pytest.mark.parametrize('tool', [
    'quant.fp8',                    # patch
    'cache.pi05_cross_frame_kv',    # port
    'kernel.forgekernel',           # kernel
    'kernel.kernelagent',
    'compile.aitune',               # compiler integration uses the patch lane
    'engine.apxinf',                # baseline is also non-config
    'baseline.official_impl',
    'profile.stage_intervals',     # no blanket exception for analysis
    'analysis.lever_pricing',
    'config.custom',               # a config-looking unknown id is not evidence
    'kernel.test',                  # legacy custom provenance still needs snapshots
])
@pytest.mark.parametrize('position', ['baseline', 'candidate'])
def test_non_config_or_unknown_tool_requires_snapshots_before_any_ledger_write(tmp_path, tool, position):
    base = baseline()
    valid = replace(base, candidate_id='valid', backend=replace(base.backend, num_steps=6))
    point = replace(valid, candidate_id='unsafe', tool=tool)
    if position == 'baseline':
        base, candidates = replace(base, tool=tool), []
    else:
        candidates = [valid, point]
    root = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='snapshot_store') as rejected:
        initialize(root, base, candidates)
    assert tool in str(rejected.value)
    assert not (root / 'plan.json').exists()
    assert not (root / 'ledger.sqlite').exists()
    # The same root remains usable after fixing the proposal.
    assert initialize(root, baseline(), [valid])['schema_version'] == 2


@pytest.mark.parametrize('base_source,point_source', [
    ('a' * 40, 'b' * 40),
    ('a' * 40, None),
    (None, 'a' * 40),
    ('legacy-tree', 'different-tree'),
    ('HEAD', None),
])
@pytest.mark.parametrize('tool', [None, 'sampler.euler_steps'])
def test_source_changes_require_snapshots_even_for_config_or_implicit_provenance(
        tmp_path, base_source, point_source, tool):
    base = baseline(source_ref=base_source)
    point = replace(base, candidate_id='other', source_ref=point_source, tool=tool)
    root = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='snapshot_store') as rejected:
        initialize(root, base, [point])
    assert 'source_ref' in str(rejected.value)
    assert 'other' in str(rejected.value)
    assert not (root / 'plan.json').exists()
    assert not (root / 'ledger.sqlite').exists()


@pytest.mark.parametrize('patch_ref', ['sha256:' + 'b' * 64, 'legacy-patch'])
@pytest.mark.parametrize('position', ['candidate', 'baseline', 'both'])
@pytest.mark.parametrize('tool', [None, 'sampler.euler_steps'])
def test_any_patch_requires_snapshots_even_when_refs_match_or_tool_claims_config(
        tmp_path, patch_ref, position, tool):
    base = baseline(source_ref='a' * 40, tool=tool,
                    patch_ref=patch_ref if position in ('baseline', 'both') else None)
    point = replace(base, candidate_id='other',
                    patch_ref=patch_ref if position in ('candidate', 'both') else None)
    root = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='snapshot_store') as rejected:
        initialize(root, base, [] if position == 'baseline' else [point])
    assert 'patch_ref' in str(rejected.value)
    assert not (root / 'plan.json').exists()
    assert not (root / 'ledger.sqlite').exists()


@pytest.mark.parametrize('source_ref', [None, 'legacy-tree', 'a' * 40])
@pytest.mark.parametrize('origin', ['human', 'search', 'agent'])
def test_same_source_unpatched_legacy_config_keeps_schema_2(tmp_path, source_ref, origin):
    base = baseline(source_ref=source_ref)
    # A JSON record written before tool/stage existed remains a valid proposal.
    point = replace(base, candidate_id='other', backend=replace(base.backend, num_steps=6), origin=origin)
    point = Candidate.from_dict(json.loads(json.dumps(point.to_dict())))
    plan = initialize(tmp_path / 'campaign', base, [point])
    assert plan['schema_version'] == 2
    assert [entry['role'] for entry in plan['entries']] == ['reference', 'candidate', 'bookend']
    with CampaignLedger(tmp_path / 'campaign' / 'ledger.sqlite') as ledger:
        assert ledger.get_candidate('other') == point
        assert ledger.snapshots() == {}


@pytest.mark.parametrize('source_ref', [None, 'legacy-tree', 'a' * 40])
def test_registered_same_source_config_and_legacy_search_still_use_schema_2(tmp_path, source_ref):
    base = baseline(source_ref=source_ref, tool='sampler.euler_steps')
    points = [replace(base, candidate_id='configured', backend=replace(base.backend, num_steps=6))]
    assert initialize(tmp_path / 'config', base, points)['schema_version'] == 2
    points = enumerate_candidates(base, euler_steps=[6], epsilons=[])
    assert points and all(point.tool is None for point in points)
    assert initialize(tmp_path / 'legacy-search', base, points)['schema_version'] == 2


def test_analysis_generated_same_source_levers_follow_the_non_config_tool_rule(tmp_path):
    base = baseline()
    points = price_levers(base, {'steps': {'num_steps': 6}})
    assert points and points[0].tool == 'analysis.lever_pricing'
    assert points[0].source_ref == base.source_ref and points[0].patch_ref is None
    root = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='snapshot_store'):
        initialize(root, base, points)
    assert not (root / 'ledger.sqlite').exists()


def test_invalid_later_proposal_preserves_an_existing_ledger(tmp_path):
    root = tmp_path / 'campaign'
    root.mkdir()
    base = baseline()
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        ledger.initialize_attempts([(base, 'older', 'existing')])
        instance_id = ledger.instance_id
        previous = ledger.attempts()
    good = replace(base, candidate_id='good', tool='sampler.euler_steps')
    bad = replace(base, candidate_id='bad', source_ref='different-source')
    with pytest.raises(ValueError, match='snapshot_store'):
        initialize(root, base, [good, bad])
    assert not (root / 'plan.json').exists()
    with CampaignLedger(root / 'ledger.sqlite') as ledger:
        assert ledger.instance_id == instance_id
        assert ledger.attempts() == previous
        assert ledger.snapshots() == {}
        for identifier in ('good', 'bad'):
            with pytest.raises(KeyError):
                ledger.get_candidate(identifier)


def test_controller_cli_reports_required_snapshot_without_persisting_prefix(tmp_path, capsys):
    base = baseline()
    base_path, proposals = tmp_path / 'base.json', tmp_path / 'proposals.json'
    base_path.write_text(json.dumps(base.to_dict()))
    point = replace(base, candidate_id='patch', tool='sampler.euler_steps', patch_ref='legacy-patch')
    proposals.write_text(json.dumps([point.to_dict()]))
    root = tmp_path / 'campaign'
    code = controller_main(['init', '--root', str(root), '--comparison-id', 'r1',
                            '--baseline', str(base_path), '--proposals', str(proposals),
                            '--workload', 'inputs.json', '--output-root', 'results'])
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ''
    error = json.loads(captured.err)
    assert error['error'] == 'ValueError'
    assert 'snapshot_store' in error['message'] and '--snapshot-store' in error['message']
    assert not (root / 'plan.json').exists()
    assert not (root / 'ledger.sqlite').exists()
