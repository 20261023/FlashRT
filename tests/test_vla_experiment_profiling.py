from types import SimpleNamespace

from flash_rt.vla.autotune.profiling import profile_request


class Event:
    tick = 0
    def record(self):
        self.value = Event.tick
        Event.tick += 1
    def elapsed_time(self, other):
        return float(other.value - self.value)


def test_profile_observes_real_replays_and_restores_original_graphs():
    class Graph:
        def __init__(self):
            self.calls = 0
        def replay(self):
            self.calls += 1
    prefix, velocity = Graph(), Graph()
    pipeline = SimpleNamespace(_graph=prefix, _velocity_graphs={0: velocity})
    class Backend:
        config = SimpleNamespace(model='pi05', sampler='probeflow')
        frontend = SimpleNamespace(pipeline=pipeline)
        def prepare(self, *args, **kwargs):
            pass
        def infer(self, *args, **kwargs):
            pipeline._graph.replay()
            pipeline._velocity_graphs[0].replay()
            pipeline._velocity_graphs[0].replay()
            return {'sampler_metadata': {'nfe': 2}}
    request = SimpleNamespace(observation={}, prompt='pick', noise=None)
    result = profile_request(Backend(), request, event_factory=Event, synchronize=lambda: None)
    assert len(result['interval_ms']['prefix']) == 1
    assert len(result['interval_ms']['velocity']) == 2
    assert result['interval_sum_ms']['velocity'] == 2.0
    assert prefix.calls == 1 and velocity.calls == 2
    assert pipeline._graph is prefix
    assert pipeline._velocity_graphs[0] is velocity
    assert result['measurement_kind'] == 'diagnostic_cuda_event_intervals'
