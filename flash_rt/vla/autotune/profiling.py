"""Separate diagnostic event intervals, never substituted for formal latency."""

from contextlib import ExitStack
import time


class _TimedGraph:
    def __init__(self, graph, stage, intervals, event_factory):
        self.graph, self.stage = graph, stage
        self.intervals, self.event_factory = intervals, event_factory

    def replay(self, *args, **kwargs):
        start, stop = self.event_factory(), self.event_factory()
        start.record()
        result = self.graph.replay(*args, **kwargs)
        stop.record()
        self.intervals.setdefault(self.stage, []).append((start, stop))
        return result

    def __getattr__(self, name):
        return getattr(self.graph, name)


def _replace_attribute(stack, owner, name, value):
    missing = object()
    original = owner.__dict__.get(name, missing)
    setattr(owner, name, value)
    def restore():
        if original is missing:
            delattr(owner, name)
        else:
            setattr(owner, name, original)
    stack.callback(restore)


def _replace_item(stack, mapping, key, value):
    original = mapping[key]
    mapping[key] = value
    stack.callback(mapping.__setitem__, key, original)


def _dm05_intervals(stack, frontend, intervals, event_factory):
    """Time the warmed fixed-E10 program, including its compiled-layer variant."""
    pipe = frontend.pipeline
    # prepare() just ran this observation. A previous request can have a
    # different prefix length and therefore a different graph and KV buffers.
    profile = frontend.profiles[frontend.last_prepared['input_ids'].shape[1]]
    captured = profile.graph is not None
    factory = event_factory
    if captured:
        import torch
        def factory():
            return torch.cuda.Event(enable_timing=True, external=True)

    pairs = {stage: [(factory(), factory()) for _ in range(count)]
             for stage, count in (('vision', 1), ('prefix', 1), ('velocity', 10))}
    if captured:
        for events in pairs.values():
            for start, end in events:
                start.record()
                end.record()
        torch.cuda.synchronize()
    vision, prefill, velocity = pipe.vision, pipe.prefill, pipe.velocity

    def timed_vision(*args):
        start, end = pairs['vision'][0]
        start.record()
        value = vision(*args)
        end.record()
        # vision (including pooling/projection) is the first call inside
        # prefill. Start the prefix-only interval here to avoid double counting.
        pairs['prefix'][0][0].record()
        return value

    def timed_prefill(*args):
        value = prefill(*args)
        pairs['prefix'][0][1].record()
        return value

    def timed_velocity(state, active_profile, step):
        start, end = pairs['velocity'][step]
        start.record()
        value = velocity(state, active_profile, step)
        end.record()
        return value

    for name, fn in (('vision', timed_vision), ('prefill', timed_prefill), ('velocity', timed_velocity)):
        _replace_attribute(stack, pipe, name, fn)
    if captured:
        # Python method wrappers cannot observe nodes inside the production
        # graph. Capture a diagnostic copy after compilation and warmup, and
        # replay that copy through the ordinary public request/host IO path.
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        stack.callback(graph.reset)
        with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
            pipe.forward(profile)
        _replace_attribute(stack, profile, 'graph', graph)
    intervals.update(pairs)


def profile_request(backend, request, *, event_factory=None, synchronize=None):
    """Replay one warmed request with event instrumentation, then restore it.

    GPU event intervals may contain stream-idle submission/dependency gaps. They
    are NOT kernel-duration sums or roofline evidence. No compute/memory bound
    conclusion is inferred here. This extra request is outside measured samples.
    """
    if event_factory is None:
        import torch
        event_factory = lambda: torch.cuda.Event(enable_timing=True)
        synchronize = torch.cuda.synchronize
    synchronize = synchronize or (lambda: None)
    backend.prepare(request.observation, prompt=request.prompt)
    frontend = backend.frontend
    intervals = {}
    with ExitStack() as stack:
        if backend.config.model == 'dm05':
            _dm05_intervals(stack, frontend, intervals, event_factory)
        elif backend.config.model in ('pi0', 'pi05'):
            pipe = frontend.pipeline
            graph = getattr(pipe, '_graph', None)
            if graph is not None:
                label = 'prefix' if backend.config.sampler == 'probeflow' else 'full'
                _replace_attribute(stack, pipe, '_graph', _TimedGraph(graph, label, intervals, event_factory))
            for key, graph in list(getattr(pipe, '_velocity_graphs', {}).items()):
                _replace_item(stack, pipe._velocity_graphs, key,
                              _TimedGraph(graph, 'velocity', intervals, event_factory))
        elif backend.config.model == 'hyvla':
            # ViT + merger graph (hyvla_thor._vit_merge replays g["graph"]).
            for entry in getattr(frontend, '_vit_graph_cache', {}).values():
                if entry.get('graph') is not None:
                    _replace_item(stack, entry, 'graph', _TimedGraph(
                        entry['graph'], 'vision', intervals, event_factory))
            for entry in frontend._graph_cache.values():
                _replace_item(stack, entry, 'prefix_graph', _TimedGraph(
                    entry['prefix_graph'], 'prefix', intervals, event_factory))
                for key, graph in list(entry['velocity_graphs'].items()):
                    _replace_item(stack, entry['velocity_graphs'], key,
                                  _TimedGraph(graph, 'velocity', intervals, event_factory))
        elif backend.config.model == 'fastwam':
            pipe = frontend.pipeline
            # infer() calls self._prepare_condition directly (the public
            # prepare_condition wraps it too), so the private method is timed.
            for name, stage in (('_prepare_condition', 'condition_vae_video_prefill'),
                                ('_evaluate_clock_velocity', 'velocity')):
                original = getattr(pipe, name)
                def timed(*args, _original=original, _stage=stage, **kwargs):
                    start, stop = event_factory(), event_factory()
                    start.record()
                    result = _original(*args, **kwargs)
                    stop.record()
                    intervals.setdefault(_stage, []).append((start, stop))
                    return result
                _replace_attribute(stack, pipe, name, timed)
        synchronize()
        started = time.perf_counter_ns()
        result = backend.infer(request.observation, prompt=request.prompt, noise=request.noise)
        synchronize()
        wall_ms = (time.perf_counter_ns() - started) / 1e6
    elapsed = {stage: [float(start.elapsed_time(stop)) for start, stop in pairs]
               for stage, pairs in intervals.items()}
    return {'measurement_kind': 'diagnostic_cuda_event_intervals',
            'separate_from_latency_samples': True, 'wall_ms': wall_ms,
            'interval_ms': elapsed,
            'interval_sum_ms': {stage: sum(values) for stage, values in elapsed.items()},
            'sampler_metadata': result.get('sampler_metadata'),
            'bound_classification': 'unknown_without_kernel_counters_or_dataflow_measurements'}
