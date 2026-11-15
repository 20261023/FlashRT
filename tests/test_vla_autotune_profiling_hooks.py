"""profile_request hooks on fake Hy/FastWAM backends (no Torch: injected events)."""

from types import SimpleNamespace

from flash_rt.vla.autotune.profiling import profile_request
from flash_rt.vla.autotune.stages import attribute


class Event:
    tick = 0

    def record(self):
        self.value = Event.tick
        Event.tick += 1

    def elapsed_time(self, other):
        return float(other.value - self.value)


class Graph:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def replay(self):
        self.log.append(self.name)


REQUEST = SimpleNamespace(observation={}, prompt='pick', noise=None)


class FakeFastWAMPipeline:
    """Mirrors FastWAMActionPipeline: infer calls _prepare_condition directly."""

    def __init__(self, nfe):
        self.nfe, self.log = nfe, []

    def prepare_condition(self, obs):
        return self._prepare_condition(obs)

    def _prepare_condition(self, obs):
        self.log.append('condition')
        return 'condition'

    def _evaluate_clock_velocity(self, state, clock, condition):
        assert condition == 'condition'
        self.log.append('velocity')
        return 0.0

    def infer(self, obs, noise=None):
        condition = self._prepare_condition(obs)
        for clock in range(self.nfe):
            self._evaluate_clock_velocity(None, clock, condition)
        return {'sampler_metadata': {'sampler': 'euler', 'nfe': self.nfe}}


def _backend(model, frontend, infer, sampler='euler'):
    class Backend:
        config = SimpleNamespace(model=model, sampler=sampler)

        def prepare(self, *args, **kwargs):
            pass

        def infer(self, observation, *, prompt=None, noise=None):
            return infer()
    Backend.frontend = frontend
    return Backend()


def test_fastwam_condition_and_velocity_recorded_when_infer_calls_private_method():
    pipe = FakeFastWAMPipeline(nfe=3)
    backend = _backend('fastwam', SimpleNamespace(pipeline=pipe), lambda: pipe.infer({}))
    result = profile_request(backend, REQUEST, event_factory=Event, synchronize=lambda: None)
    assert set(result['interval_ms']) == {'condition_vae_video_prefill', 'velocity'}
    assert len(result['interval_ms']['condition_vae_video_prefill']) == 1
    assert len(result['interval_ms']['velocity']) == 3
    assert pipe.log == ['condition', 'velocity', 'velocity', 'velocity']
    # Instance attributes are removed again: the class methods are visible.
    assert '_prepare_condition' not in vars(pipe) and '_evaluate_clock_velocity' not in vars(pipe)
    assert 'prepare_condition' not in vars(pipe)
    assert pipe._prepare_condition.__func__ is FakeFastWAMPipeline._prepare_condition
    record = {'candidate': {'backend': {'model': 'fastwam', 'sampler': 'euler'}}, 'stage_metrics': result}
    attribution = attribute(record)
    assert [group['interval_key'] for group in attribution['groups']] == [
        'condition_vae_video_prefill', 'velocity']
    assert not any('missing' in note for note in attribution['notes'])


def test_fastwam_public_prepare_condition_is_timed_once():
    pipe = FakeFastWAMPipeline(nfe=1)
    backend = _backend('fastwam', SimpleNamespace(pipeline=pipe),
                       lambda: (pipe.prepare_condition({}), {'sampler_metadata': {'nfe': 1}})[1])
    result = profile_request(backend, REQUEST, event_factory=Event, synchronize=lambda: None)
    assert len(result['interval_ms']['condition_vae_video_prefill']) == 1


def _hy_frontend(log, *, with_vit=True):
    entry = {'buf': {}, 'prefix_graph': Graph('prefix', log),
             'velocity_graphs': {1.0: Graph('v1', log), 0.5: Graph('v2', log)}}
    frontend = SimpleNamespace(_graph_cache={(64, 49): entry})
    if with_vit:
        frontend._vit_graph_cache = {(3, 1): {'graph': Graph('vit', log), 'img': 'img', 'out': 'out'}}
    return frontend, entry


def _hy_infer(frontend):
    def infer():
        if hasattr(frontend, '_vit_graph_cache'):
            frontend._vit_graph_cache[(3, 1)]['graph'].replay()
        entry = frontend._graph_cache[(64, 49)]
        entry['prefix_graph'].replay()
        for time in (1.0, 0.5):
            entry['velocity_graphs'][time].replay()
        return {'sampler_metadata': {'sampler': 'euler', 'nfe': 2}}
    return infer


def test_hy_vision_prefix_and_velocity_intervals_recorded_and_restored():
    log = []
    frontend, entry = _hy_frontend(log)
    vit, prefix = frontend._vit_graph_cache[(3, 1)]['graph'], entry['prefix_graph']
    velocities = dict(entry['velocity_graphs'])
    result = profile_request(_backend('hyvla', frontend, _hy_infer(frontend)), REQUEST,
                             event_factory=Event, synchronize=lambda: None)
    assert {key: len(value) for key, value in result['interval_ms'].items()} == {
        'vision': 1, 'prefix': 1, 'velocity': 2}
    assert log == ['vit', 'prefix', 'v1', 'v2']
    assert frontend._vit_graph_cache[(3, 1)]['graph'] is vit
    assert entry['prefix_graph'] is prefix and entry['velocity_graphs'] == velocities
    record = {'candidate': {'backend': {'model': 'hyvla', 'sampler': 'euler'}}, 'stage_metrics': result}
    attribution = attribute(record)
    assert attribution['coverage'] == 'full'
    assert attribution['groups'][0]['stages'] == ['vision.encode', 'vision.project']


def test_hy_without_vit_graph_cache_still_profiles_prefix_and_velocity():
    log = []
    frontend, _ = _hy_frontend(log, with_vit=False)
    result = profile_request(_backend('hyvla', frontend, _hy_infer(frontend)), REQUEST,
                             event_factory=Event, synchronize=lambda: None)
    assert set(result['interval_ms']) == {'prefix', 'velocity'}
    record = {'candidate': {'backend': {'model': 'hyvla', 'sampler': 'euler'}}, 'stage_metrics': result}
    attribution = attribute(record)
    assert attribution['coverage'] == 'partial'
    assert any("['vision']" in note for note in attribution['notes'])


def test_hy_attributes_restored_even_when_infer_raises():
    log = []
    frontend, entry = _hy_frontend(log)
    vit = frontend._vit_graph_cache[(3, 1)]['graph']

    def broken():
        frontend._vit_graph_cache[(3, 1)]['graph'].replay()
        raise RuntimeError('boom')
    try:
        profile_request(_backend('hyvla', frontend, broken), REQUEST, event_factory=Event,
                        synchronize=lambda: None)
    except RuntimeError:
        pass
    else:
        raise AssertionError('expected the infer error to propagate')
    assert frontend._vit_graph_cache[(3, 1)]['graph'] is vit
    assert log == ['vit']
