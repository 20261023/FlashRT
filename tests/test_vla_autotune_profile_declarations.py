"""CPU regressions for declared hooks, disjoint clocks and diagnostic captures."""

from types import SimpleNamespace
import json

import pytest

from flash_rt.vla.autotune import profiling
from flash_rt.vla.autotune.stages import attribute
from flash_rt.vla.model_spec import get_spec


class Event:
    tick = 0

    def record(self):
        self.value = Event.tick
        Event.tick += 1

    def elapsed_time(self, other):
        return float(other.value - self.value)


class Graph:
    def __init__(self, fn):
        self.fn = fn

    def replay(self, *args, **kwargs):
        return self.fn()


REQUEST = SimpleNamespace(observation={}, prompt='pick', noise=None)


def backend(model, frontend, infer):
    return SimpleNamespace(
        config=SimpleNamespace(model=model, sampler='euler', framework='torch'),
        frontend=frontend, prepare=lambda *a, **k: None,
        infer=lambda *a, **k: infer())


def declaration(adapter):
    return {'profiling': {'adapter': f'flash_rt.vla.autotune.profilers.{adapter}:install'}}


def test_declared_adapter_profiles_a_new_name_without_model_dispatch():
    pipe = SimpleNamespace(
        _prepare_condition=lambda obs: 'condition',
        _evaluate_clock_velocity=lambda *args: 0)

    def infer():
        pipe._prepare_condition({})
        pipe._evaluate_clock_velocity(None, None, None)
        return {'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}

    result = profiling.profile_request(
        backend('new-declared-name', SimpleNamespace(pipeline=pipe), infer), REQUEST,
        declaration=declaration('fastwam'), event_factory=Event)
    assert set(result['interval_ms']) == {'condition_vae_video_prefill', 'velocity'}
    assert result['interval_provenance']['velocity']['device'] == 'cuda'
    assert result['interval_provenance']['velocity']['clock'] == 'cuda_event'
    assert result['separate_from_latency_samples'] is True


def test_hy_host_regions_exclude_model_compute_and_output_wait(monkeypatch):
    log = []
    now = [0]
    monkeypatch.setattr(profiling.time, 'perf_counter_ns', lambda: now[0])

    def work(label, ns):
        log.append(label)
        now[0] += ns

    frontend = SimpleNamespace(
        _vit_graph_cache={}, _graph_cache={},
        _vit_merge=lambda *a: work('vision', 30_000_000),
        _graph_forward=lambda *a: work('prefix+velocity', 80_000_000))

    def infer():
        work('input1', 2_000_000)
        frontend._vit_merge(None)
        work('input2', 3_000_000)
        frontend._graph_forward()
        work('output', 7_000_000)
        return {'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}

    def sync():
        work('sync', 11_000_000)

    result = profiling.profile_request(
        backend('hyvla', frontend, infer), REQUEST, declaration=declaration('hy'),
        event_factory=Event, synchronize=sync)
    assert sum(result['interval_ms']['host_input']) == pytest.approx(5.0)
    assert result['interval_ms']['host_output'] == pytest.approx([7.0])
    assert result['interval_provenance']['host_input']['clock'] == 'perf_counter_ns'
    assert result['interval_provenance']['host_output']['device'] == 'cpu'
    assert '_graph_forward' in vars(frontend)


def test_fastwam_splits_real_condition_hooks_without_parent_double_count():
    pipe = SimpleNamespace(
        _initial_state=lambda noise: 'state',
        _observation_tensors=lambda obs: None,
        _evaluate_clock_velocity=lambda *a: None)
    model = SimpleNamespace(
        _encode_input_image_latents_tensor=lambda **k: None,
        video_expert=SimpleNamespace(prepare=lambda **k: None),
        mot=SimpleNamespace(prefill_video_cache_tensor=lambda **k: None))
    pipe.model = model

    def condition(obs):
        pipe._observation_tensors(obs)
        model._encode_input_image_latents_tensor(input_image=None)
        model.video_expert.prepare()
        model.mot.prefill_video_cache_tensor()

    pipe._prepare_condition = condition

    def infer():
        pipe._initial_state(None)
        pipe._prepare_condition({})
        pipe._evaluate_clock_velocity(None, None, None)
        return {'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}

    result = profiling.profile_request(
        backend('fastwam', SimpleNamespace(pipeline=pipe), infer), REQUEST,
        declaration=declaration('fastwam'), event_factory=Event)
    assert 'condition_vae_video_prefill' not in result['interval_ms']
    assert {'host_input', 'vision', 'condition_tokens', 'prefix', 'velocity'} <= set(result['interval_ms'])
    assert result['interval_ms']['vision'] == [1.0]
    assert result['interval_ms']['prefix'] == [1.0]
    assert result['interval_provenance']['condition_tokens']['stages'] == ['prefix.prefill']
    assert pipe._prepare_condition is condition


def test_attribution_rejects_additive_parent_with_children():
    record = {
        'candidate': {'backend': {'model': 'pi05', 'sampler': 'euler'}},
        'stage_metrics': {
            'measurement_kind': 'diagnostic_cuda_event_intervals', 'wall_ms': 30.0,
            'interval_ms': {'full': [20.0], 'velocity': [5.0]},
            'sampler_metadata': {'sampler': 'euler', 'nfe': 1},
        }}
    with pytest.raises(ValueError, match='overlap|double.count'):
        attribute(record)


def test_attribution_preserves_clock_and_treats_operator_probe_as_non_additive():
    metrics = {
        'measurement_kind': 'diagnostic_cuda_event_intervals', 'wall_ms': 20.0,
        'separate_from_latency_samples': True,
        'interval_ms': {'host_input': [3.0], 'vision': [4.0], 'prefix_only': [5.0],
                        'velocity': [6.0], 'host_output': [1.0], 'gemm_probe': [100.0]},
        'sampler_metadata': {'sampler': 'euler', 'nfe': 1},
        'interval_provenance': {
            'host_input': {'stages': ['host.input'], 'device': 'cpu', 'clock': 'perf_counter_ns'},
            'vision': {'stages': ['vision.encode'], 'device': 'cuda', 'clock': 'cuda_event'},
            'prefix_only': {'stages': ['vision.project', 'prefix.prefill'],
                            'device': 'cuda', 'clock': 'cuda_event'},
            'velocity': {'stages': ['denoise.velocity'], 'device': 'cuda', 'clock': 'cuda_event'},
            'host_output': {'stages': ['host.output'], 'device': 'cpu', 'clock': 'perf_counter_ns'},
            'gemm_probe': {'stages': ['kernel.gemm'], 'device': 'cuda', 'clock': 'cuda_event',
                           'scope': 'operator_probe', 'additive': False},
        },
        'expected_interval_keys': ['host_input', 'vision', 'prefix_only', 'velocity', 'host_output'],
    }
    result = attribute({'candidate': {'backend': {'model': 'pi05'}}, 'stage_metrics': metrics})
    assert result['unattributed_ms'] == pytest.approx(1.0)
    assert len(result['groups']) == 5
    assert result['groups'][0]['provenance']['device'] == 'cpu'
    assert result['operator_probes'][0]['ms'] == 100.0
    assert result['coverage'] == 'full'
    assert result['coverage_scope'] == 'declared_compute_stages'


def test_attribution_rejects_undeclared_stage_claims():
    metrics = {
        'measurement_kind': 'diagnostic_cuda_event_intervals', 'wall_ms': 20.0,
        'interval_ms': {'vision': [4.0]},
        'sampler_metadata': {'sampler': 'euler', 'nfe': 1},
        'interval_provenance': {
            'vision': {'stages': ['kernel.made_up'], 'device': 'cuda', 'clock': 'cuda_event'}},
    }
    with pytest.raises(ValueError, match='unknown stage|Unknown stage'):
        attribute({'candidate': {'backend': {'model': 'hyvla'}}, 'stage_metrics': metrics})


class CapturedGraph:
    """CPU capture driver: exercises the same source call sequence on replay."""
    def __init__(self, body):
        self.body, self.closed = body, False
        body()

    def replay(self, *args):
        assert not self.closed
        self.body()

    def reset(self):
        self.closed = True


def test_pi_euler_separate_capture_preserves_steps_inputs_and_production_graph():
    log, captures = [], []

    class Pipe:
        num_steps = 3
        input_noise_buf = SimpleNamespace(ptr=SimpleNamespace(value=123))

        def vision_encoder(self, stream=0):
            log.append('vision')

        def transformer_encoder(self, stream=0):
            log.append('prefix')

        def transformer_decoder(self, stream=0, *, _step_indices=None):
            for step in range(3) if _step_indices is None else _step_indices:
                log.append(('velocity', step))
                self.state += step + 1

        def run_pipeline(self, stream=0):
            self.vision_encoder(stream=stream)
            self.transformer_encoder(stream=stream)
            self.transformer_decoder(stream=stream)

        def forward(self, stream=0):
            self._graph.replay()
            return 123

    pipe = Pipe()
    pipe.state = 0
    pipe._graph = Graph(pipe.run_pipeline)
    original_graph = pipe._graph

    def infer():
        pipe.state = 10
        pipe.forward()
        return {'actions': [pipe.state], 'sampler_metadata': {'sampler': 'euler', 'nfe': 3}}

    def capture(body, *, stream=None):
        graph = CapturedGraph(body)
        captures.append(graph)
        return graph

    result = profiling.profile_request(
        backend('pi0', SimpleNamespace(pipeline=pipe), infer),
        SimpleNamespace(observation={}, prompt='pick', noise=[10]),
        declaration=declaration('pi'), event_factory=Event, capture_graph=capture)
    assert set(result['interval_ms']) == {
        'host_input', 'vision', 'prefix_only', 'velocity', 'host_output'}
    assert len(result['interval_ms']['velocity']) == 3
    assert log.count(('velocity', 0)) == log.count(('velocity', 1)) == log.count(('velocity', 2)) == 3
    assert pipe._graph is original_graph
    assert not {'forward', 'vision_encoder', 'transformer_encoder', 'transformer_decoder'} & vars(pipe).keys()
    assert captures[0].closed
    assert result['diagnostic_graph_recapture'] is True
    assert result['capture_parity']['status'] == 'bit_exact'


def test_pi_split_falls_back_to_full_without_supplied_noise():
    graph = Graph(lambda: None)
    pipe = SimpleNamespace(_graph=graph, run_pipeline=lambda **k: None,
                           transformer_decoder=lambda **k: None)
    result = profiling.profile_request(
        backend('pi05', SimpleNamespace(pipeline=pipe),
                lambda: (pipe._graph.replay(), {'sampler_metadata': {'sampler': 'euler', 'nfe': 1}})[1]),
        REQUEST, declaration=declaration('pi'), event_factory=Event)
    assert set(result['interval_ms']) == {'full'}
    assert not result['diagnostic_graph_recapture']
    assert any('supplied noise' in note for note in result['notes'])
    assert pipe._graph is graph


def test_dm_capture_cuts_prefix_after_vision_and_restores_graph():
    captures, log = [], []

    class Pipe:
        def vision(self, *a):
            log.append('vision')

        def prefill(self, profile):
            self.vision()
            log.append('prefix')

        def velocity(self, state, profile, step):
            log.append(('velocity', step))
            return 1

        def forward(self, profile):
            self.prefill(profile)
            profile.output = sum(self.velocity(None, profile, step) for step in range(10))

    pipe = Pipe()

    class Profile:
        def run(self, pipeline):
            self.graph.replay()

    profile = Profile()
    profile.graph = Graph(lambda: pipe.forward(profile))
    original = profile.graph
    frontend = SimpleNamespace(
        pipeline=pipe, profiles={17: profile},
        last_prepared={'input_ids': SimpleNamespace(shape=(1, 17))})

    def infer():
        profile.run(pipe)
        return {'actions': [profile.output],
                'sampler_metadata': {'sampler': 'euler', 'nfe': 10}}

    def capture(body, *, stream=None):
        graph = CapturedGraph(body)
        captures.append(graph)
        return graph

    result = profiling.profile_request(
        backend('dm05', frontend, infer),
        SimpleNamespace(observation={}, prompt='pick', noise=[1]),
        declaration=declaration('dm'), event_factory=Event, capture_graph=capture)
    assert len(result['interval_ms']['velocity']) == 10
    # Vision ends, then prefix starts; the latter does not enclose vision.
    assert result['interval_ms']['vision'] == [1.0]
    assert result['interval_ms']['prefix'] == [1.0]
    assert {'host_input', 'host_output'} <= set(result['interval_ms'])
    assert profile.graph is original and 'run' not in vars(profile)
    assert not {'vision', 'prefill', 'velocity'} & vars(pipe).keys()
    assert captures[0].closed and result['capture_parity']['status'] == 'bit_exact'


def external_spec(tmp_path, *, hook=False):
    spec = get_spec('pi05')
    spec.update(model='external-fixture-vla', aliases=[])
    spec['adapter']['hooks'] = (
        {'profile': 'flash_rt.vla.autotune.profilers.fastwam:install',
         'dataflow': 'flash_rt.vla.autotune.profilers.fastwam:describe_dataflow'}
        if hook else {})
    keys = {
        'condition_vae_video_prefill': ['host.input', 'vision.encode', 'vision.project', 'prefix.prefill'],
        'velocity': ['denoise.velocity']}
    # The generic method strategy is deliberately a new method/path, unrelated
    # to any of the five builtin model names or their private frontend names.
    if hook:
        keys['condition_vae_video_prefill'].remove('vision.project')
        spec['stages'] = [item for item in spec['stages'] if item['stage_id'] != 'vision.project']
    spec['profiling'] = {
        'keys': keys, 'expected_keys': {'euler': list(keys)},
        'strategy': {'kind': 'methods', 'methods': [
            {'path': 'pipeline.condition', 'key': 'condition_vae_video_prefill'},
            {'path': 'pipeline.field', 'key': 'velocity'}]}}
    for item in spec['stages']:
        item['profiling_keys'] = [key for key, covered in keys.items() if item['stage_id'] in covered]
    (tmp_path / 'external-fixture-vla.json').write_text(json.dumps(spec))
    return spec


@pytest.mark.parametrize('hook', [False, True])
def test_external_json_fixture_profiles_and_attributes_without_core_registration(tmp_path, hook):
    from flash_rt.vla.autotune.dataflow import describe_dataflow
    from flash_rt.vla.autotune.stages import stages_for

    external_spec(tmp_path, hook=hook)
    pipe = SimpleNamespace(
        condition=lambda obs: None, field=lambda *args: None,
        _prepare_condition=lambda obs: None,
        _evaluate_clock_velocity=lambda *args: None, action_horizon=7, action_dim=2)

    def infer():
        (pipe._prepare_condition if hook else pipe.condition)({})
        (pipe._evaluate_clock_velocity if hook else pipe.field)(None, None, None)
        return {'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}

    engine = backend('external-fixture-vla', SimpleNamespace(pipeline=pipe), infer)
    engine.config.spec_dir = str(tmp_path)
    metrics = profiling.profile_request(engine, REQUEST, event_factory=Event)
    assert metrics['interval_ms']['velocity'] == [1.0]
    record = {'candidate': {'backend': {'model': engine.config.model, 'spec_dir': str(tmp_path)}},
              'stage_metrics': metrics}
    attribution = attribute(record)
    assert attribution['coverage'] == 'full'
    assert len(attribution['groups']) == 2
    assert stages_for(engine.config.model, spec_dir=tmp_path)[0].stage_id == 'host.input'
    if hook:
        assert describe_dataflow(engine)['runtime_geometry'] == {'action_horizon': 7, 'action_dim': 2}
    else:
        assert describe_dataflow(engine) is None


def test_diagnostic_capture_parity_failure_restores_all_hooks():
    class Pipe:
        num_steps = 2
        input_noise_buf = SimpleNamespace(ptr=SimpleNamespace(value=123))

        def vision_encoder(self, stream=0):
            pass

        def transformer_encoder(self, stream=0):
            pass

        def transformer_decoder(self, stream=0, *, _step_indices=None):
            pass

        def run_pipeline(self, stream=0):
            self.vision_encoder(stream)
            self.transformer_encoder(stream)
            self.transformer_decoder(stream)

        def forward(self, stream=0):
            return 123

    pipe = Pipe()
    pipe._graph = Graph(lambda: None)
    original = pipe._graph
    calls, captures = [], []

    def infer():
        pipe.forward()
        calls.append(None)
        return {'actions': [len(calls)],
                'sampler_metadata': {'sampler': 'euler', 'nfe': 2}}

    def capture(body, *, stream=None):
        graph = CapturedGraph(body)
        captures.append(graph)
        return graph

    with pytest.raises(RuntimeError, match='changed public actions'):
        profiling.profile_request(
            backend('pi0', SimpleNamespace(pipeline=pipe), infer),
            SimpleNamespace(observation={}, prompt='pick', noise=[1]),
            declaration=declaration('pi'), event_factory=Event, capture_graph=capture)
    assert pipe._graph is original
    assert not {'forward', 'vision_encoder', 'transformer_encoder', 'transformer_decoder'} & vars(pipe).keys()
    assert captures[0].closed


def test_probe_only_is_not_request_coverage():
    metrics = {
        'measurement_kind': 'diagnostic_cuda_event_intervals', 'wall_ms': 10.0,
        'interval_ms': {'operator_gemm': [9.5]},
        'interval_provenance': {
            'operator_gemm': {'stages': ['kernel.gemm'], 'scope': 'operator_probe',
                              'additive': False, 'clock': 'cuda_event', 'device': 'cuda'}}}
    result = attribute({'candidate': {'backend': {'model': 'pi05'}}, 'stage_metrics': metrics})
    assert result['coverage'] == 'none' and result['attributed_fraction'] is None
    assert result['groups'] == []
    assert result['operator_probes'][0]['ms'] == 9.5


def test_external_graph_paths_restore_mapping_and_instance_owners(tmp_path):
    spec = external_spec(tmp_path)
    spec['profiling']['strategy'] = {
        'kind': 'graphs', 'graphs': [
            {'path': 'bundles.*.condition', 'key': 'condition_vae_video_prefill'},
            {'path': 'bundles.*.fields.*', 'key': 'velocity'}]}
    (tmp_path / 'external-fixture-vla.json').write_text(json.dumps(spec))
    log = []
    first, second = Graph(lambda: log.append('condition')), Graph(lambda: log.append('velocity'))
    entry = {'condition': first, 'fields': {0: second}}
    frontend = SimpleNamespace(bundles={'exact-shape': entry})

    def infer():
        entry['condition'].replay()
        entry['fields'][0].replay()
        return {'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}

    engine = backend(spec['model'], frontend, infer)
    engine.config.spec_dir = str(tmp_path)
    result = profiling.profile_request(engine, REQUEST, event_factory=Event)
    assert log == ['condition', 'velocity']
    assert result['interval_ms']['velocity'] == [1.0]
    assert entry['condition'] is first and entry['fields'][0] is second


def test_capture_events_from_setup_cannot_claim_request_coverage():
    class Pipe:
        num_steps = 1
        input_noise_buf = SimpleNamespace(ptr=SimpleNamespace(value=123))

        def vision_encoder(self, stream=0):
            pass

        def transformer_encoder(self, stream=0):
            pass

        def transformer_decoder(self, stream=0, *, _step_indices=None):
            pass

        def run_pipeline(self, stream=0):
            self.vision_encoder(stream)
            self.transformer_encoder(stream)
            self.transformer_decoder(stream)

        def forward(self, stream=0):
            pass

    pipe = Pipe()
    graph = pipe._graph = Graph(lambda: None)

    def capture(body, *, stream=None):
        body()
        # A broken/mismatched replay leaves all event timestamps from setup.
        return Graph(lambda: None)

    def infer():
        pipe.forward()
        return {'actions': [1], 'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}

    with pytest.raises(RuntimeError, match='event.*request|request.*event'):
        profiling.profile_request(
            backend('pi0', SimpleNamespace(pipeline=pipe), infer),
            SimpleNamespace(observation={}, prompt='pick', noise=[1]),
            declaration=declaration('pi'), event_factory=Event, capture_graph=capture)
    assert pipe._graph is graph


def test_external_regression_head_declares_new_stage_without_nfe_or_core_edits(tmp_path):
    spec = external_spec(tmp_path)
    spec['head'] = {'type': 'other', 'kind': 'regression', 'defaults': {'sampler': 'none'}}
    spec['capabilities']['samplers'] = ['none']
    spec['attention'] = []
    spec['stages'] = [{
        'stage_id': 'head.predict', 'kind': 'compute', 'title': 'Regression head',
        'seams': ['flash_rt/vla/autotune/profilers/declared.py:install'],
        'profiling_keys': ['prediction'], 'notes': ['Explicit external head declaration']}]
    spec['profiling'] = {
        'keys': {'prediction': ['head.predict']}, 'expected_keys': {'none': ['prediction']},
        'strategy': {'kind': 'methods', 'methods': [{'path': 'pipeline.predict', 'key': 'prediction'}]}}
    (tmp_path / 'external-fixture-vla.json').write_text(json.dumps(spec))
    pipe = SimpleNamespace(predict=lambda: [1])

    def infer():
        return {'actions': pipe.predict(), 'sampler_metadata': {'sampler': 'none'}}

    engine = backend(spec['model'], SimpleNamespace(pipeline=pipe), infer)
    engine.config.spec_dir, engine.config.sampler = str(tmp_path), 'none'
    metrics = profiling.profile_request(engine, REQUEST, event_factory=Event)
    record = {'candidate': {'backend': {'model': spec['model'], 'spec_dir': str(tmp_path), 'sampler': 'none'}},
              'stage_metrics': metrics}
    result = attribute(record)
    assert result['coverage'] == 'full'
    assert result['nfe'] is None and result['per_call_ms'] == {}
    assert result['groups'][0]['stages'] == ['head.predict']


@pytest.mark.parametrize('second_span, overlaps', [([3.0, 6.0], True), ([8.0, 11.0], False)])
def test_same_stage_subregions_require_disjoint_event_spans(tmp_path, second_span, overlaps):
    spec = external_spec(tmp_path)
    spec['profiling']['keys']['second_field'] = ['denoise.velocity']
    next(item for item in spec['stages'] if item['stage_id'] == 'denoise.velocity')[
        'profiling_keys'].append('second_field')
    (tmp_path / 'external-fixture-vla.json').write_text(json.dumps(spec))
    detail = {'stages': ['denoise.velocity'], 'clock': 'cuda_event', 'device': 'cuda'}
    metrics = {
        'measurement_kind': 'diagnostic_cuda_event_intervals', 'wall_ms': 20.0,
        'interval_ms': {'velocity': [6.0], 'second_field': [3.0]},
        'interval_provenance': {'velocity': detail, 'second_field': detail},
        'interval_spans_ms': {'velocity': [[2.0, 8.0]], 'second_field': [second_span]},
        'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}
    record = {'candidate': {'backend': {'model': spec['model'], 'spec_dir': str(tmp_path)}},
              'stage_metrics': metrics}
    if overlaps:
        with pytest.raises(ValueError, match='Overlapping'):
            attribute(record)
    else:
        assert attribute(record)['unattributed_ms'] == pytest.approx(11.0)


def test_nonadditive_host_overlay_is_mapped_but_never_request_coverage():
    metrics = {
        'measurement_kind': 'diagnostic_cuda_event_intervals', 'wall_ms': 25.0,
        'interval_ms': {'full': [20.0], 'host_input': [9.0]},
        'interval_sum_ms': {'full': 20.0, 'host_input': 9.0},
        'interval_provenance': {
            'full': {'stages': ['vision.encode', 'vision.project', 'prefix.prefill', 'denoise.velocity'],
                     'clock': 'cuda_event', 'device': 'cuda', 'additive': True},
            'host_input': {'stages': ['host.input'], 'clock': 'perf_counter_ns',
                           'device': 'cpu', 'scope': 'diagnostic_request', 'additive': False}},
        'interval_spans_ms': {'full': [[1.0, 21.0]], 'host_input': [[0.0, 9.0]]},
        'expected_interval_keys': ['full', 'host_input'],
        'sampler_metadata': {'sampler': 'euler', 'nfe': 10}}
    result = attribute({'candidate': {'backend': {'model': 'pi05'}}, 'stage_metrics': metrics})
    assert result['unattributed_ms'] == 5.0
    assert result['attributed_fraction'] == pytest.approx(0.8)
    assert [group['interval_key'] for group in result['groups']] == ['full']
    assert result['nonadditive_diagnostics'][0]['interval_key'] == 'host_input'
    assert 'host.input' in result['residual_stages']
    assert not any('missing' in note for note in result['notes'])
    assert result['coverage_evidence']['excluded_keys'] == ['host_input']
    assert result['coverage_evidence']['additive_keys'] == ['full']


def test_nonadditive_host_overlay_cannot_reassign_its_declared_stage():
    metrics = {
        'measurement_kind': 'diagnostic_cuda_event_intervals', 'wall_ms': 25.0,
        'interval_ms': {'host_input': [9.0]},
        'interval_provenance': {
            'host_input': {'stages': ['denoise.velocity'], 'clock': 'perf_counter_ns',
                           'device': 'cpu', 'additive': False}},
        'sampler_metadata': {'sampler': 'euler', 'nfe': 10}}
    with pytest.raises(ValueError, match='differ.*declaration'):
        attribute({'candidate': {'backend': {'model': 'pi05'}}, 'stage_metrics': metrics})


def test_generic_strategy_preserves_explicit_nonadditive_host_overlay(tmp_path):
    spec = external_spec(tmp_path)
    spec['profiling']['keys']['host_overlay'] = ['host.input']
    next(item for item in spec['stages'] if item['stage_id'] == 'host.input')[
        'profiling_keys'].append('host_overlay')
    spec['profiling']['strategy']['methods'].append({
        'path': 'pipeline.host', 'key': 'host_overlay', 'clock': 'perf_counter_ns', 'additive': False})
    (tmp_path / 'external-fixture-vla.json').write_text(json.dumps(spec))
    pipe = SimpleNamespace(condition=lambda obs: None, field=lambda *args: None, host=lambda: None)

    def infer():
        pipe.host()
        pipe.condition({})
        pipe.field()
        return {'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}

    engine = backend(spec['model'], SimpleNamespace(pipeline=pipe), infer)
    engine.config.spec_dir = str(tmp_path)
    metrics = profiling.profile_request(engine, REQUEST, event_factory=Event)
    assert metrics['interval_provenance']['host_overlay']['additive'] is False
    record = {'candidate': {'backend': {'model': spec['model'], 'spec_dir': str(tmp_path)}},
              'stage_metrics': metrics}
    assert attribute(record)['nonadditive_diagnostics'][0]['interval_key'] == 'host_overlay'


@pytest.mark.parametrize('model', ['pi0', 'pi05'])
def test_builtin_pi_euler_split_is_attributable_via_actual_json(model):
    class Pipe:
        num_steps = 2
        input_noise_buf = SimpleNamespace(ptr=SimpleNamespace(value=123))

        def vision_encoder(self, stream=0):
            pass

        def transformer_encoder(self, stream=0):
            pass

        def transformer_decoder(self, stream=0, *, _step_indices=None):
            pass

        def run_pipeline(self, stream=0):
            self.vision_encoder(stream)
            self.transformer_encoder(stream)
            self.transformer_decoder(stream)

        def forward(self, stream=0):
            self._graph.replay()

    pipe = Pipe()
    pipe._graph = Graph(pipe.run_pipeline)

    def infer():
        pipe.forward()
        return {'actions': [1], 'sampler_metadata': {'sampler': 'euler', 'nfe': 2}}

    metrics = profiling.profile_request(
        backend(model, SimpleNamespace(pipeline=pipe), infer),
        SimpleNamespace(observation={}, prompt='pick', noise=[1]),
        event_factory=Event, capture_graph=lambda body, **kwargs: CapturedGraph(body))
    result = attribute({'candidate': {'backend': {'model': model}}, 'stage_metrics': metrics})
    assert result['coverage'] == 'full'
    assert result['per_call_ms']['denoise.velocity'] == 1.0
    assert result['coverage_evidence']['missing_keys'] == []
    assert result['coverage_evidence']['span_validation'] == 'same_clock_nonoverlap_checked'
    assert result['groups'][0]['provenance']['scope'] == 'diagnostic_same_program_recapture'


def test_dm_without_noise_preserves_whole_graph_evidence_and_host_output():
    class Profile:
        def run(self, pipe):
            self.graph.replay()
    profile = Profile()
    profile.graph = Graph(lambda: None)
    original = profile.graph
    frontend = SimpleNamespace(
        pipeline=SimpleNamespace(), profiles={17: profile},
        last_prepared={'input_ids': SimpleNamespace(shape=(1, 17))})

    def infer():
        profile.run(frontend.pipeline)
        return {'actions': [1], 'sampler_metadata': {'sampler': 'euler', 'nfe': 10}}

    metrics = profiling.profile_request(backend('dm05', frontend, infer), REQUEST, event_factory=Event)
    assert set(metrics['interval_ms']) == {'full', 'host_input', 'host_output'}
    assert profile.graph is original
    result = attribute({'candidate': {'backend': {'model': 'dm05'}}, 'stage_metrics': metrics})
    assert result['coverage'] == 'full'
    assert result['per_call_ms'] == {}
    assert result['coverage_evidence']['missing_keys'] == []
    assert metrics['diagnostic_graph_recapture'] is False
    assert any('supplied noise' in note for note in result['notes'])


@pytest.mark.parametrize('compiled', [False, True])
def test_fastwam_split_and_compiler_owner_pass_declared_stage_mapping(compiled):
    pipe = SimpleNamespace(
        _initial_state=lambda noise: None, _observation_tensors=lambda obs: None,
        _evaluate_clock_velocity=lambda *args: None)
    model = SimpleNamespace(
        _encode_input_image_latents_tensor=lambda **kwargs: None,
        video_expert=SimpleNamespace(prepare=lambda: None),
        mot=SimpleNamespace(prefill_video_cache_tensor=lambda: None))
    pipe.model = model
    pipe._compiled_prefill = (lambda: None) if compiled else None

    def condition(obs):
        pipe._observation_tensors(obs)
        model._encode_input_image_latents_tensor(input_image=None)
        model.video_expert.prepare()
        (pipe._compiled_prefill or model.mot.prefill_video_cache_tensor)()

    pipe._prepare_condition = condition

    def infer():
        pipe._initial_state(None)
        pipe._prepare_condition({})
        pipe._evaluate_clock_velocity()
        return {'actions': [1], 'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}

    metrics = profiling.profile_request(
        backend('fastwam', SimpleNamespace(pipeline=pipe), infer), REQUEST, event_factory=Event)
    result = attribute({'candidate': {'backend': {'model': 'fastwam'}}, 'stage_metrics': metrics})
    assert result['coverage'] == 'full'
    assert 'condition_vae_video_prefill' not in metrics['interval_ms']
    assert len([group for group in result['groups'] if group['interval_key'] == 'prefix']) == 1
    assert not result['coverage_evidence']['missing_keys']
    assert model.mot.prefill_video_cache_tensor is not None


def test_nonadditive_compute_overlay_does_not_hide_real_missing_compute():
    metrics = {
        'measurement_kind': 'diagnostic_cuda_event_intervals', 'wall_ms': 25.0,
        'interval_ms': {'vision': [5.0], 'prefix': [8.0], 'velocity': [9.0]},
        'interval_provenance': {
            'vision': {'stages': ['vision.encode', 'vision.project'],
                       'clock': 'cuda_event', 'device': 'cuda', 'additive': False},
            'prefix': {'stages': ['prefix.prefill'], 'clock': 'cuda_event',
                       'device': 'cuda', 'additive': True},
            'velocity': {'stages': ['denoise.velocity'], 'clock': 'cuda_event',
                         'device': 'cuda', 'additive': True}},
        'sampler_metadata': {'sampler': 'euler', 'nfe': 1}}
    result = attribute({'candidate': {'backend': {'model': 'hyvla'}}, 'stage_metrics': metrics})
    assert result['coverage'] == 'partial'
    assert result['unattributed_ms'] == 8.0
    assert {'vision.encode', 'vision.project'} <= set(result['residual_stages'])
    assert result['coverage_evidence']['excluded_keys'] == ['vision']
