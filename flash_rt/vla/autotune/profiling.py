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
        if backend.config.model in ('pi0', 'pi05'):
            pipe = frontend.pipeline
            graph = getattr(pipe, '_graph', None)
            if graph is not None:
                label = 'prefix' if backend.config.sampler == 'probeflow' else 'full'
                _replace_attribute(stack, pipe, '_graph', _TimedGraph(graph, label, intervals, event_factory))
            for key, graph in list(getattr(pipe, '_velocity_graphs', {}).items()):
                _replace_item(stack, pipe._velocity_graphs, key,
                              _TimedGraph(graph, 'velocity', intervals, event_factory))
        elif backend.config.model == 'hyvla':
            for entry in frontend._graph_cache.values():
                _replace_item(stack, entry, 'prefix_graph', _TimedGraph(
                    entry['prefix_graph'], 'prefix', intervals, event_factory))
                for key, graph in list(entry['velocity_graphs'].items()):
                    _replace_item(stack, entry['velocity_graphs'], key,
                                  _TimedGraph(graph, 'velocity', intervals, event_factory))
        elif backend.config.model == 'fastwam':
            pipe = frontend.pipeline
            for name, stage in (('prepare_condition', 'condition_vae_video_prefill'),
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
