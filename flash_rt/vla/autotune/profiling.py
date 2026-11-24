"""Declared, isolated request diagnostics; never a formal latency sample.

Adapters install temporary instance wrappers, not changes to model programs.
CUDA events measure stream intervals (including dependency/submission gaps);
CPU host intervals use perf_counter_ns and explicitly document their IO waits.
"""

from contextlib import ExitStack
from importlib import import_module
import time

from ..model_spec import get_spec


def _hook(reference):
    module, name = reference.split(':', 1)
    return getattr(import_module(module), name)


def _replace_attribute(stack, owner, name, value):
    missing = object()
    original = vars(owner).get(name, missing)
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


class _TimedGraph:
    def __init__(self, graph, timed):
        self.graph, self.timed = graph, timed

    def replay(self, *args, **kwargs):
        return self.timed(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.graph, name)


class DiagnosticSession:
    """One serialized request and the reversible hooks used to observe it."""

    def __init__(self, backend, request, stack, *, event_factory, synchronize,
                 capture_graph=None):
        self.backend, self.request, self.stack = backend, request, stack
        self.frontend = backend.frontend
        self.event_factory, self.synchronize = event_factory, synchronize
        self.capture_graph = capture_graph
        self.pairs, self.host_ms, self.provenance = {}, {}, {}
        self.spans = {}
        self.expected, self.notes = [], []
        self.start_host_input = False
        self._host = None
        self.recapture = False
        self.reference = None
        self.parity = None
        self.gpu_epoch = self.wall_epoch = None

    def replace(self, owner, name, value):
        _replace_attribute(self.stack, owner, name, value)

    def replace_item(self, mapping, key, value):
        _replace_item(self.stack, mapping, key, value)

    def describe(self, key, stages, boundary, *, clock='cuda_event',
                 scope='diagnostic_request', additive=True):
        if type(additive) is not bool:
            raise ValueError('Diagnostic additive flag must be bool')
        value = {
            'stages': list(stages), 'device': 'cuda' if clock == 'cuda_event' else 'cpu',
            'clock': clock, 'scope': scope, 'boundary': boundary, 'additive': additive,
        }
        if clock == 'cuda_event':
            value['semantics'] = 'stream_interval_including_submission_and_dependency_gaps'
        else:
            value['semantics'] = 'host_wall_including_local_IO_and_validation_waits'
        previous = self.provenance.setdefault(key, value)
        if previous != value:
            raise ValueError(f'Conflicting diagnostic provenance for {key!r}')

    def pair(self, key, stages, boundary, *, external=False, scope='diagnostic_request',
             additive=True):
        if self.event_factory is None:
            import torch
            factory = lambda: torch.cuda.Event(enable_timing=True, external=external)
        else:
            factory = self.event_factory
        pair = factory(), factory()
        if external:
            # Materialize event handles before capture; the replay overwrites
            # these priming timestamps before any elapsed_time is consumed.
            for event in pair:
                event.record()
        self.describe(key, stages, boundary, scope=scope, additive=additive)
        self.pairs.setdefault(key, []).append(pair)
        return pair

    def timed(self, fn, key, stages, boundary, *, pair=None, scope='diagnostic_request',
              additive=True):
        def wrapped(*args, **kwargs):
            start, stop = pair or self.pair(key, stages, boundary, scope=scope, additive=additive)
            start.record()
            result = fn(*args, **kwargs)
            stop.record()
            return result
        return wrapped

    def gpu_method(self, owner, name, key, stages, boundary, **kwargs):
        original = getattr(owner, name, None)
        if not callable(original):
            return False
        self.replace(owner, name, self.timed(original, key, stages, boundary, **kwargs))
        return True

    def graph(self, graph, key, stages, boundary, *, additive=True, scope='diagnostic_request'):
        return _TimedGraph(graph, self.timed(graph.replay, key, stages, boundary,
                                             additive=additive, scope=scope))

    def host_method(self, owner, name, key, stages, boundary, *, additive=True,
                    scope='diagnostic_request'):
        original = getattr(owner, name, None)
        if not callable(original):
            return False
        self.describe(key, stages, boundary, clock='perf_counter_ns',
                      additive=additive, scope=scope)

        def wrapped(*args, **kwargs):
            # Drain preceding work outside this interval. The interval includes
            # only this input helper's own IO/validation and completion waits.
            self.synchronize()
            started = time.perf_counter_ns()
            result = original(*args, **kwargs)
            self.synchronize()
            self._host_elapsed(key, started, time.perf_counter_ns())
            return result

        self.replace(owner, name, wrapped)
        return True

    def begin_host(self, key):
        if self._host is not None:
            raise RuntimeError('Host diagnostic intervals must not overlap')
        stages = ('host.input',) if key == 'host_input' else ('host.output',)
        boundary = ('Public request to compute entry, excluding instrumented model compute'
                    if key == 'host_input' else
                    'Completed model compute to public return, including D2H and action decoding')
        self.describe(key, stages, boundary, clock='perf_counter_ns')
        self._host = key, time.perf_counter_ns()

    def end_host(self):
        if self._host is not None:
            key, started = self._host
            self._host_elapsed(key, started, time.perf_counter_ns())
            self._host = None

    def _host_elapsed(self, key, started, stopped):
        self.host_ms.setdefault(key, []).append((stopped - started) / 1e6)
        self.spans.setdefault(key, []).append(
            [(started - self.wall_epoch) / 1e6, (stopped - self.wall_epoch) / 1e6])

    def start_request(self):
        factory = self.event_factory
        if factory is None:
            import torch
            factory = lambda: torch.cuda.Event(enable_timing=True)
        self.gpu_epoch = factory()
        self.gpu_epoch.record()
        self.synchronize()
        self.wall_epoch = time.perf_counter_ns()
        return self.wall_epoch

    def boundary(self, owner, name, *, resume_input=False, output=False):
        """Keep CPU IO regions disjoint from model compute and its drain wait."""
        original = getattr(owner, name, None)
        if not callable(original):
            return False

        def wrapped(*args, **kwargs):
            self.end_host()
            self.synchronize()
            result = original(*args, **kwargs)
            self.synchronize()
            if resume_input or output:
                self.begin_host('host_input' if resume_input else 'host_output')
            return result

        self.replace(owner, name, wrapped)
        return True

    def capture(self, body, *, stream=None):
        """Capture a private diagnostic copy after production warmup.

        The production graph, its pools and its graph handles are never reset,
        adopted or overwritten. Callers retain address-stable model inputs.
        """
        self.recapture = True
        if self.capture_graph is not None:
            graph = self.capture_graph(body, stream=stream)
            reset = getattr(graph, 'reset', None)
            if callable(reset):
                self.stack.callback(reset)
            return graph
        import torch
        stream = stream or torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.synchronize()
        graph = torch.cuda.CUDAGraph()
        self.stack.callback(graph.reset)
        with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
            body()
        torch.cuda.current_stream().wait_stream(stream)
        self.synchronize()
        return graph

    def fixed_noise(self):
        return (self.request.noise is not None
                or self.request.observation.get('noise') is not None)

    def take_reference(self):
        """Only recapture when a public output parity check is reproducible."""
        if not self.fixed_noise():
            self.notes.append('Diagnostic graph split requires supplied noise; retaining unsplit evidence')
            return False
        self.reference = self.backend.infer(
            self.request.observation, prompt=self.request.prompt, noise=self.request.noise)
        import numpy as np
        self.reference = {key: np.asarray(value).copy() for key, value in self.reference.items()
                          if key in ('actions', 'raw_actions')}
        self.synchronize()
        return True

    def check_parity(self, result):
        if self.reference is None:
            return
        import numpy as np
        compared = []
        for key in ('raw_actions', 'actions'):
            if key not in self.reference or key not in result:
                continue
            before, after = np.asarray(self.reference[key]), np.asarray(result[key])
            if (before.shape != after.shape or not np.isfinite(before).all()
                    or not np.isfinite(after).all() or not np.array_equal(before, after)):
                raise RuntimeError(f'Diagnostic capture changed public {key}; production graph restored')
            compared.append(key)
        if not compared:
            raise RuntimeError('Diagnostic capture parity requires actions or raw_actions')
        self.parity = {'status': 'bit_exact', 'fields': compared,
                       'scope': 'same_request_same_supplied_noise'}

    def elapsed(self):
        values = {}
        for key, pairs in self.pairs.items():
            values[key] = []
            for start, stop in pairs:
                begin = float(self.gpu_epoch.elapsed_time(start))
                end = float(self.gpu_epoch.elapsed_time(stop))
                if begin < 0 or end < begin:
                    raise RuntimeError(f'Diagnostic event for {key!r} did not execute during this request')
                self.spans.setdefault(key, []).append([begin, end])
                values[key].append(float(start.elapsed_time(stop)))
        values.update(self.host_ms)
        return values


def profile_request(backend, request, *, event_factory=None, synchronize=None,
                    declaration=None, capture_graph=None):
    """Observe a separate warmed request using the declared adapter hook.

    Event injection also supports non-Torch timing backends. Model discovery is
    metadata-only; a hook is imported only for this explicit diagnostic request.
    Missing hooks and missing intervals remain unknown rather than zero costs.
    """
    if synchronize is None:
        if event_factory is None:
            import torch
            synchronize = torch.cuda.synchronize
        else:
            synchronize = lambda: None
    spec = declaration if declaration is not None else get_spec(
        backend.config.model, spec_dir=getattr(backend.config, 'spec_dir', None))
    reference = spec.get('adapter', {}).get('hooks', {}).get('profile')
    reference = reference or spec.get('profiling', {}).get('adapter')
    with ExitStack() as locks:
        # VLABackend owns an RLock and serializes all public requests. Do not
        # acquire private frontend locks here: external adapters may use
        # non-reentrant locks inside infer().
        lock = getattr(backend, '_lock', None)
        if lock is not None:
            locks.enter_context(lock)
        backend.prepare(request.observation, prompt=request.prompt)
        with ExitStack() as stack:
            session = DiagnosticSession(
                backend, request, stack, event_factory=event_factory,
                synchronize=synchronize, capture_graph=capture_graph)
            session.declaration = spec
            try:
                if reference:
                    _hook(reference)(session)
                else:
                    from .profilers.declared import install
                    install(session)
                started = session.start_request()
                if session.start_host_input:
                    session.begin_host('host_input')
                result = backend.infer(
                    request.observation, prompt=request.prompt, noise=request.noise)
                session.end_host()
                synchronize()
                wall_ms = (time.perf_counter_ns() - started) / 1e6
                elapsed = session.elapsed()
                session.check_parity(result)
            finally:
                # Complete any in-flight replay before restoring wrappers and
                # freeing the private diagnostic graph, including on failure.
                synchronize()
        return {
            'measurement_kind': 'diagnostic_cuda_event_intervals',
            'separate_from_latency_samples': True, 'wall_ms': wall_ms,
            'interval_ms': elapsed,
            'interval_sum_ms': {key: sum(values) for key, values in elapsed.items()},
            'interval_provenance': {key: session.provenance[key] for key in elapsed},
            'interval_spans_ms': session.spans,
            'span_clock_domains': {
                'cuda_event': 'device_event_epoch_before_request_wall_timer',
                'perf_counter_ns': 'request_wall_timer_epoch',
            },
            'expected_interval_keys': list(session.expected),
            'sampler_metadata': result.get('sampler_metadata'),
            'diagnostic_graph_recapture': session.recapture,
            'capture_parity': session.parity, 'notes': session.notes,
            'bound_classification': 'unknown_without_kernel_counters_or_dataflow_measurements',
        }
