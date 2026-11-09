"""Owned graph contracts use real CPU tensor math and an injected capture seam.

These do not qualify CUDA execution; the production capture path must refuse CPU.
"""
import pytest
torch = pytest.importorskip('torch')


def driver_module():
    from flash_rt.models.fastwam import velocity_graph
    return velocity_graph


def inputs():
    return dict(latents_action=torch.ones(1, 2, 1), timestep_action=torch.tensor([.5]),
        context=torch.full((1, 3, 1), 3.), context_mask=torch.ones(1, 3, dtype=torch.bool),
        video_cache_k=[torch.full((1, 2, 1), 4.), torch.full((1, 2, 1), 6.)],
        video_cache_v=[torch.full((1, 2, 1), 5.), torch.full((1, 2, 1), 7.)],
        action_attention_mask=torch.ones(1, 2, 4, dtype=torch.bool))


def head(**data):
    return (2*data['latents_action'] + data['timestep_action'].reshape(1, 1, 1)
            + data['context'].mean() + data['context_mask'].sum()
            + sum(x.mean() for x in data['video_cache_k'])
            + sum(x.mean() for x in data['video_cache_v']) + data['action_attention_mask'].sum())


class Capture:
    def __init__(self):
        self.captures = self.replays = self.releases = 0
        self.fail_release = self.fail_replay = False
        self.static = None

    def __call__(self, fn, static):
        self.captures += 1
        self.static = static
        output = torch.empty_like(static['latents_action'])
        def replay():
            if self.fail_replay:
                raise RuntimeError('injected replay failure')
            self.replays += 1
            output.copy_(fn(**static))
        def release():
            self.releases += 1
            if self.fail_release:
                raise RuntimeError('injected reset failure')
        return replay, output, release, {'warmup_evaluations': 3, 'encoded_bodies': 1,
                                         'captured_native_linear_calls': {'a': 7, 'b': 4}}


def test_request_refreshes_same_pointers_but_evaluations_refresh_only_x_time():
    module = driver_module(); capture = Capture()
    graph = module.FastWAMVelocityGraph(head, capture=capture)
    data = inputs(); graph.begin_request()
    borrowed = graph(**data); first = borrowed.clone()
    torch.testing.assert_close(first, torch.full_like(first, 38.5), atol=0, rtol=0)
    data['latents_action'].fill_(2); data['timestep_action'].fill_(1)
    data['context'].fill_(9)  # not refreshed until the next request
    assert graph(**data).data_ptr() == borrowed.data_ptr()
    torch.testing.assert_close(borrowed, torch.full_like(borrowed, 41.), atol=0, rtol=0)
    graph.end_request(); graph.begin_request()
    torch.testing.assert_close(graph(**data), torch.full_like(first, 47.), atol=0, rtol=0)
    torch.testing.assert_close(first, torch.full_like(first, 38.5), atol=0, rtol=0)
    assert capture.captures == 1
    assert graph.snapshot()['condition_refreshes'] == 1
    assert graph.snapshot()['replays'] == 1 and graph.snapshot()['captures'] == 0
    graph.end_request(); graph.close()


def test_request_scope_rejects_missing_nested_and_ended_requests():
    graph = driver_module().FastWAMVelocityGraph(head, capture=Capture())
    with pytest.raises(RuntimeError, match='request'):
        graph(**inputs())
    graph.begin_request()
    with pytest.raises(RuntimeError, match='request'):
        graph.begin_request()
    graph.end_request()
    with pytest.raises(RuntimeError, match='request'):
        graph(**inputs())


@pytest.mark.parametrize('change', ['shape', 'stride', 'dtype', 'list', 'tree'])
def test_changed_geometry_rejects_without_replay_fallback_or_losing_original(change):
    module = driver_module(); capture = Capture()
    graph = module.FastWAMVelocityGraph(head, capture=capture)
    original = inputs(); graph.begin_request(); graph(**original); graph.end_request()
    changed = inputs()
    if change == 'shape': changed['latents_action'] = torch.ones(1, 3, 1)
    if change == 'stride': changed['context'] = torch.ones(1, 3, 2)[..., :1]
    if change == 'dtype': changed['context'] = changed['context'].double()
    if change == 'list': changed['video_cache_k'].append(torch.ones(1, 2, 1))
    if change == 'tree': changed['video_cache_k'] = tuple(changed['video_cache_k'])
    graph.begin_request()
    with pytest.raises(module.VelocityGeometryMismatch, match='geometry'):
        graph(**changed)
    assert capture.replays == 1 and capture.captures == 1
    graph.end_request(); graph.begin_request()
    torch.testing.assert_close(graph(**original), torch.full((1, 2, 1), 38.5), atol=0, rtol=0)


def test_static_layout_is_preserved_for_non_dense_inputs_and_overlap_refuses():
    module = driver_module(); capture = Capture()
    graph = module.FastWAMVelocityGraph(head, capture=capture)
    data = inputs(); data['context'] = torch.full((1, 3, 2), 3.)[..., :1]
    graph.begin_request(); graph(**data)
    assert capture.static['context'].stride() == data['context'].stride()
    other = module.FastWAMVelocityGraph(head, capture=Capture())
    overlap = inputs(); overlap['context'] = torch.ones(1, 1, 1).expand(1, 3, 1)
    other.begin_request()
    with pytest.raises(ValueError, match='overlap|layout'):
        other(**overlap)


def test_cold_body_and_logical_replay_counters_are_separate():
    graph = driver_module().FastWAMVelocityGraph(head, capture=Capture())
    graph.begin_request(); graph(**inputs()); graph(**inputs())
    report = graph.snapshot()
    assert {key: report[key] for key in ('replays', 'captures', 'condition_refreshes',
        'warmup_evaluations', 'encoded_bodies')} == dict(replays=2, captures=1,
        condition_refreshes=1, warmup_evaluations=3, encoded_bodies=1)
    assert report['captured_native_linear_calls_per_replay'] == 11
    assert report['logical_native_linear_calls_in_replays'] == 22
    graph.end_request(); graph.begin_request(); graph(**inputs())
    assert graph.snapshot()['warmup_evaluations'] == graph.snapshot()['encoded_bodies'] == 0


def test_release_failure_retains_owners_and_close_can_retry_but_infer_cannot():
    capture = Capture(); graph = driver_module().FastWAMVelocityGraph(head, capture=capture)
    graph.begin_request(); graph(**inputs()); graph.end_request()
    capture.fail_release = True
    with pytest.raises(RuntimeError, match='reset failure'):
        graph.close()
    assert graph._static is capture.static and graph._fn is head
    with pytest.raises(RuntimeError, match='clos|fail'):
        graph.begin_request()
    capture.fail_release = False; graph.close(); graph.close()
    assert capture.releases == 2 and graph._static is graph._fn is None


def test_bad_capture_output_never_publishes_a_replayable_graph_and_keeps_release():
    capture = Capture()
    def bad(fn, static):
        replay, _, release, meta = capture(fn, static)
        return replay, torch.empty(1), release, meta
    graph = driver_module().FastWAMVelocityGraph(head, capture=bad)
    graph.begin_request()
    with pytest.raises(ValueError, match='velocity'):
        graph(**inputs())
    assert capture.replays == 0
    graph.end_request()
    with pytest.raises(RuntimeError, match='fail'):
        graph.begin_request()
    graph.close(); assert capture.releases == 1


def test_replay_failure_stops_new_work_but_cleanup_is_available():
    capture = Capture(); graph = driver_module().FastWAMVelocityGraph(head, capture=capture)
    graph.begin_request(); graph(**inputs()); capture.fail_replay = True
    with pytest.raises(RuntimeError, match='replay failure'):
        graph(**inputs())
    graph.end_request()
    with pytest.raises(RuntimeError, match='fail'):
        graph.begin_request()
    graph.close(); assert capture.releases == 1


def test_capture_failure_retains_partial_cuda_owner_for_retryable_release():
    module = driver_module(); releases = []
    def fail(fn, static):
        raise module.VelocityCaptureError('injected capture failure', release=lambda: releases.append(True))
    graph = module.FastWAMVelocityGraph(head, capture=fail); graph.begin_request()
    with pytest.raises(module.VelocityCaptureError, match='capture failure'):
        graph(**inputs())
    graph.end_request(); graph.close()
    assert releases == [True]


def test_production_capture_rejects_cpu_tensors_without_initializing_cuda(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('must reject before CUDA query')
    monkeypatch.setattr(torch.cuda, 'current_stream', forbidden)
    graph = driver_module().FastWAMVelocityGraph(head); graph.begin_request()
    with pytest.raises(ValueError, match='CUDA'):
        graph(**inputs())


@pytest.mark.parametrize('failure', [None, 'body', 'end'])
def test_real_capture_orchestration_never_flushes_global_caches_and_retains_failed_owner(monkeypatch, failure):
    """Exercise production orchestration; only actual CUDA primitives are faked."""
    from contextlib import nullcontext
    module = driver_module(); events = []; calls = 0
    stream = type('Stream', (), {'synchronize': lambda self: events.append('stream_sync')})()
    class Graph:
        def capture_begin(self, *, capture_error_mode): events.append(('begin', capture_error_mode))
        def capture_end(self):
            events.append('end')
            if failure == 'end': raise RuntimeError('injected end failure')
        def replay(self): events.append('replay')
        def reset(self): events.append('reset')
    def forbidden(*args, **kwargs): pytest.fail('capture must not flush or synchronize global allocator state')
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: stream)
    monkeypatch.setattr(torch.cuda, 'default_stream', lambda device: object())
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', Graph)
    monkeypatch.setattr(torch.cuda, 'empty_cache', forbidden)
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    monkeypatch.setattr(torch._C, '_host_emptyCache', forbidden, raising=False)
    def body(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 4 and failure == 'body': raise RuntimeError('injected body failure')
        return kwargs['latents_action'] * 2
    if failure is None:
        replay, output, release, meta = module._capture_cuda(body, inputs())
        torch.testing.assert_close(output, torch.full((1, 2, 1), 2.))
        assert calls == 4 and meta['warmup_evaluations'] == 3 and meta['encoded_bodies'] == 1
        replay(); release()
        assert events == ['stream_sync', ('begin', 'global'), 'end', 'replay', 'stream_sync', 'reset']
    else:
        with pytest.raises(module.VelocityCaptureError) as error:
            module._capture_cuda(body, inputs())
        assert calls == 4 and events == ['stream_sync', ('begin', 'global'), 'end']
        error.value.release()
        assert events[-2:] == ['stream_sync', 'reset']
