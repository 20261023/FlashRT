import json

import pytest

from flash_rt.vla.autotune import decisions
from flash_rt.vla.autotune.decisions import (
    ACTORS, DEFAULT_INVALIDATION, KINDS, SCHEMA, DecisionLog, validate_record)


def _clock():
    times = iter(f'2026-11-15T10:00:{second:02d}.000000+00:00' for second in range(60))
    return lambda: next(times)


def _diagnosis(log, actor='agent'):
    return log.append('diagnosis', actor=actor, summary='Velocity dominates', model='pi05',
                      stage='denoise.velocity', symptom='VLA expert 很小却不快',
                      hypothesis='launch gaps between 18 short layers',
                      next_observation='compare eager vs graph intervals',
                      candidate_tools=['launch.cuda_graph'], confidence='medium',
                      provenance='mechanistic_inference', evidence_refs=['runs/r1/result.json'])


def test_append_writes_one_validated_line_per_record(tmp_path):
    path = tmp_path / 'logs' / 'decisions.jsonl'
    log = DecisionLog(path, clock=_clock())
    record = _diagnosis(log)
    assert record['schema'] == SCHEMA and len(record['record_id']) == 32
    assert record['recorded_at'] == '2026-11-15T10:00:00.000000+00:00'
    assert record['alternatives'] == [] and record['invalid_when'] == []
    log.append('decision', actor='human', summary='Try graphs first', stage='host.launch',
               action='select_tool', tool='launch.cuda_graph', rationale='cheapest exact lever')
    experience = log.append('experience', actor='search', summary='Graph capture helped',
                            context={'model': 'pi05', 'hardware': 'cuda_sm80', 'steps': 10},
                            outcome='kept')
    assert experience['invalidation'] == list(DEFAULT_INVALIDATION)
    log.append('note', actor='human+agent', summary='Paused for review')
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    assert [json.loads(line)['kind'] for line in lines] == ['diagnosis', 'decision', 'experience', 'note']
    assert all(validate_record(json.loads(line)) for line in lines)
    assert [item['kind'] for item in log.records()] == ['diagnosis', 'decision', 'experience', 'note']
    assert [item['actor'] for item in log.records('decision')] == ['human']


def test_actor_is_required_and_validated(tmp_path):
    log = DecisionLog(tmp_path / 'd.jsonl')
    with pytest.raises(TypeError):
        log.append('note', summary='x')
    with pytest.raises(ValueError, match='actor is required'):
        log.append('note', actor=None, summary='x')
    with pytest.raises(ValueError, match='actor must be one of'):
        log.append('note', actor='robot', summary='x')
    assert not (tmp_path / 'd.jsonl').exists()


@pytest.mark.parametrize('kind, fields, message', [
    ('bogus', {}, 'kind must be one of'),
    ('note', {'summary': ' '}, 'summary must be a nonempty'),
    ('note', {'unknown': 1}, 'Unknown note fields'),
    ('note', {'record_id': 'x'}, 'assigned by the log'),
    ('note', {'stage': 'vision'}, 'known stage id'),
    ('note', {'model': 'Pi05'}, 'model must match'),
    ('note', {'evidence_refs': 'a.json'}, 'evidence_refs must be a list'),
    ('diagnosis', {'symptom': 's', 'hypothesis': 'h', 'next_observation': 'n',
                   'confidence': 'certain', 'provenance': 'source_fact'}, 'confidence must be one of'),
    ('diagnosis', {'symptom': 's', 'hypothesis': 'h', 'next_observation': 'n',
                   'confidence': 'low', 'provenance': 'guess'}, 'provenance must be one of'),
    ('diagnosis', {'symptom': 's', 'hypothesis': 'h', 'confidence': 'low',
                   'provenance': 'source_fact'}, 'requires next_observation'),
    ('diagnosis', {'symptom': 's', 'hypothesis': 'h', 'next_observation': 'n', 'confidence': 'low',
                   'provenance': 'source_fact', 'candidate_tools': ['Bad Tool']}, 'must match'),
    ('decision', {'action': 'select_tool', 'rationale': 'r'}, 'requires tool'),
    ('decision', {'action': 'select_stage', 'rationale': 'r'}, 'requires stage'),
    ('decision', {'action': 'redirect', 'rationale': 'r'}, 'requires redirected_from'),
    ('decision', {'action': 'launch', 'rationale': 'r'}, 'action must be one of'),
    ('decision', {'action': 'keep'}, 'requires rationale'),
    ('experience', {'outcome': 'o'}, 'requires context'),
    ('experience', {'context': {'x': float('nan')}, 'outcome': 'o'}, 'finite JSON'),
    ('experience', {'context': [], 'outcome': 'o'}, 'JSON object'),
])
def test_kind_fields_are_validated(tmp_path, kind, fields, message):
    fields = dict(fields)
    summary = fields.pop('summary', 'summary')
    with pytest.raises(ValueError, match=message):
        DecisionLog(tmp_path / 'd.jsonl').append(kind, actor='agent', summary=summary, **fields)


def test_redirect_must_be_led_by_a_human(tmp_path):
    log = DecisionLog(tmp_path / 'd.jsonl', clock=_clock())
    with pytest.raises(ValueError, match='human redirect'):
        log.append('decision', actor='agent', summary='s', action='redirect', rationale='r',
                   redirected_from='agent wanted kernel.forgekernel')
    record = log.append('decision', actor='human', summary='Look at prefix instead',
                        action='redirect', rationale='prefix is 60% of wall',
                        redirected_from='agent wanted kernel.forgekernel on velocity',
                        stage='prefix.prefill')
    assert record['redirected_from'].startswith('agent wanted')


def test_refuses_to_append_after_invalid_line(tmp_path):
    path = tmp_path / 'd.jsonl'
    log = DecisionLog(path, clock=_clock())
    log.append('note', actor='human', summary='ok')
    with path.open('a') as stream:
        stream.write('{"schema": "other"}\n')
    before = path.read_bytes()
    with pytest.raises(ValueError, match=':2: invalid decision record'):
        log.append('note', actor='human', summary='again')
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        log.records()


def test_refuses_partial_last_line_and_duplicates(tmp_path):
    path = tmp_path / 'd.jsonl'
    log = DecisionLog(path, clock=_clock())
    record = log.append('note', actor='human', summary='ok')
    text = path.read_text()
    path.write_text(text + text.rstrip('\n'))
    with pytest.raises(ValueError, match='incomplete'):
        log.append('note', actor='human', summary='x')
    path.write_text(text + text)
    with pytest.raises(ValueError, match='duplicate record_id'):
        log.records()
    noncanonical = dict(record)
    del noncanonical['evidence_refs']
    with pytest.raises(ValueError, match='canonical'):
        validate_record(noncanonical)
    for key, value in (('record_id', 'X' * 32), ('recorded_at', '2026-11-15T10:00:00'),
                       ('recorded_at', '2026-11-15T10:00:00+02:00'), ('schema', 'v0')):
        with pytest.raises(ValueError):
            validate_record(dict(record, **{key: value}))


def test_summary_and_markdown_table(tmp_path):
    log = DecisionLog(tmp_path / 'd.jsonl', clock=_clock())
    _diagnosis(log)
    log.append('decision', actor='human', summary='Use | graphs', action='select_tool',
               tool='launch.cuda_graph', rationale='r', stage='host.launch')
    log.append('decision', actor='human+agent', summary='Accept\nwinner', action='accept',
               rationale='quality gate passed', candidate_ids=['base-graph'])
    summary = log.summary()
    assert summary['total'] == 3
    assert summary['by_actor'] == {'human': 1, 'agent': 1, 'human+agent': 1, 'search': 0}
    assert summary['by_kind'] == {'diagnosis': 1, 'decision': 2, 'experience': 0, 'note': 0}
    assert set(summary['by_actor_kind']) == set(ACTORS)
    assert all(set(counts) == set(KINDS) for counts in summary['by_actor_kind'].values())
    table = log.markdown_table().splitlines()
    assert table[0] == '| # | time | kind | actor | stage | summary |'
    assert table[2].startswith('| 1 | 2026-11-15 10:00:00Z | diagnosis | agent | denoise.velocity | ')
    assert table[3] == ('| 2 | 2026-11-15 10:00:01Z | decision | human | host.launch | '
                        'select_tool launch.cuda_graph: Use \\| graphs |')
    assert table[4] == ('| 3 | 2026-11-15 10:00:02Z | decision | human+agent | - | '
                        'accept: Accept winner |')
    assert len(table) == 5
    counts = log.markdown_summary().splitlines()
    assert counts[0] == '| actor | diagnosis | decision | experience | note | total |'
    assert '| human | 0 | 1 | 0 | 0 | 1 |' in counts


def test_cli_add_list_summary(tmp_path, capsys):
    path = tmp_path / 'd.jsonl'
    assert decisions.main(['add', '--log', str(path), '--kind', 'diagnosis', '--actor', 'agent',
                           '--summary', 'prefix dominates', '--stage', 'prefix.prefill',
                           '--symptom', 'TTFT 很高、生成后较快', '--hypothesis', 'long prefix',
                           '--next-observation', 'read dataflow', '--confidence', 'low',
                           '--provenance', 'measured_here', '--alternative', 'vision',
                           '--alternative', 'host', '--candidate-tool', 'prefix.vision_pool',
                           '--evidence-ref', 'r.json']) == 0
    added = json.loads(capsys.readouterr().out)
    assert added['alternatives'] == ['vision', 'host'] and added['evidence_refs'] == ['r.json']
    payload = tmp_path / 'decision.json'
    payload.write_text(json.dumps({'kind': 'decision', 'action': 'run_campaign', 'rationale': 'go',
                                   'candidate_ids': ['a', 'b'], 'campaign': 'round1'}))
    assert decisions.main(['add', '--log', str(path), '--json', str(payload),
                           '--actor', 'human', '--summary', 'run round 1']) == 0
    capsys.readouterr()
    context = tmp_path / 'context.json'
    context.write_text(json.dumps({'model': 'pi05', 'hardware': 'cuda_sm80'}))
    assert decisions.main(['add', '--log', str(path), '--kind', 'experience', '--actor', 'search',
                           '--summary', 'pool 2 kept', '--context-json', str(context),
                           '--outcome', 'kept', '--invalidation', 'hardware']) == 0
    assert json.loads(capsys.readouterr().out)['invalidation'] == ['hardware']
    assert decisions.main(['list', '--log', str(path)]) == 0
    assert [item['actor'] for item in json.loads(capsys.readouterr().out)] == ['agent', 'human', 'search']
    assert decisions.main(['list', '--log', str(path), '--kind', 'decision']) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    assert decisions.main(['list', '--log', str(path), '--markdown']) == 0
    table = capsys.readouterr().out
    assert table.startswith('| # | time | kind | actor | stage | summary |')
    assert '| human | - | run_campaign: run round 1 |' in table
    assert decisions.main(['summary', '--log', str(path)]) == 0
    assert json.loads(capsys.readouterr().out)['by_actor']['search'] == 1
    assert decisions.main(['summary', '--log', str(path), '--markdown']) == 0
    output = capsys.readouterr().out
    assert '| # | time |' in output and '| actor | diagnosis |' in output


def test_cli_rejects_invalid_input(tmp_path, capsys):
    path = tmp_path / 'd.jsonl'
    assert decisions.main(['add', '--log', str(path), '--kind', 'note', '--summary', 'x']) == 2
    assert 'actor is required' in capsys.readouterr().err
    assert not path.exists()
    assert decisions.main(['add', '--log', str(path), '--actor', 'human', '--summary', 'x']) == 2
    assert 'kind is required' in capsys.readouterr().err
    payload = tmp_path / 'p.json'
    payload.write_text(json.dumps({'kind': 'note', 'actor': 'agent', 'summary': 'x'}))
    assert decisions.main(['add', '--log', str(path), '--json', str(payload), '--actor', 'human']) == 2
    assert 'both in --json and as flags' in capsys.readouterr().err
    payload.write_text('[1]')
    assert decisions.main(['add', '--log', str(path), '--json', str(payload)]) == 2
    assert 'JSON object' in capsys.readouterr().err
    assert decisions.main(['add', '--log', str(path), '--kind', 'decision', '--actor', 'agent',
                           '--summary', 's', '--action', 'select_tool', '--rationale', 'r']) == 2
    assert 'requires tool' in capsys.readouterr().err
    assert decisions.main(['list', '--log', str(path)]) == 2
    assert 'does not exist' in capsys.readouterr().err
    path.write_text('not json\n')
    assert decisions.main(['summary', '--log', str(path)]) == 2
    assert 'invalid decision record' in capsys.readouterr().err
    assert decisions.main(['add', '--log', str(path), '--kind', 'note', '--actor', 'human',
                           '--summary', 'x']) == 2
    assert path.read_text() == 'not json\n'


def test_unicode_line_separators_do_not_break_the_log(tmp_path, capsys):
    path = tmp_path / 'd.jsonl'
    log = DecisionLog(path, clock=_clock())
    text = 'pasted\u2028from\u2029a pdf\x85end 中文'
    log.append('note', actor='agent', summary=text, text=text)
    raw = path.read_bytes()
    assert raw.count(b'\n') == 1 and '中文'.encode() in raw
    assert all(char.encode() not in raw for char in ('\u2028', '\u2029', '\x85'))
    log.append('note', actor='human', summary='again')
    assert [record['summary'] for record in log.records()] == [text, 'again']
    # A line holding the raw characters (as written before the escape) still reads back.
    record = dict(log.records()[0], record_id='0' * 31 + '1')
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    assert '\u2028' in path.read_text(encoding='utf-8')
    assert [item['summary'] for item in log.records()] == [text, 'again', text]
    assert decisions.main(['list', '--log', str(path)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3


@pytest.mark.parametrize('field', ['stage', 'kind', 'actor'])
@pytest.mark.parametrize('value', [['a'], {'a': 1}])
def test_unhashable_enum_values_are_validation_errors(tmp_path, capsys, field, value):
    path = tmp_path / 'd.jsonl'
    payload = {'kind': 'note', 'actor': 'human', 'summary': 'x', field: value}
    source = tmp_path / 'j.json'
    source.write_text(json.dumps(payload))
    assert decisions.main(['add', '--log', str(path), '--json', str(source)]) == 2
    assert 'must be' in capsys.readouterr().err and not path.exists()
    # A hand-edited stored line with the same value is reported, not a traceback.
    log = DecisionLog(path, clock=_clock())
    record = dict(log.append('note', actor='human', summary='ok'), **{field: value})
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    with pytest.raises(ValueError, match=':2: invalid decision record'):
        log.records()
    assert decisions.main(['summary', '--log', str(path)]) == 2
    assert 'invalid decision record' in capsys.readouterr().err


@pytest.mark.parametrize('action', ['accept', 'merge'])
def test_accept_and_merge_are_human_checkpoints(tmp_path, action):
    log = DecisionLog(tmp_path / 'd.jsonl', clock=_clock())
    for actor in ('agent', 'search'):
        with pytest.raises(ValueError, match=f'action {action} is a human checkpoint'):
            log.append('decision', actor=actor, summary='s', action=action, rationale='r')
    assert not (tmp_path / 'd.jsonl').exists()
    for actor in ('human', 'human+agent'):
        assert log.append('decision', actor=actor, summary='s', action=action, rationale='r')['actor'] == actor
    # Other actions stay open to the agent.
    log.append('decision', actor='agent', summary='s', action='reject', rationale='r')
