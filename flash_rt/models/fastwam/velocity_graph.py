"""Instance-owned, single-geometry FastWAM raw-field capture (opt-in).

The pipeline serializes access and supplies one non-default execution stream.
Inputs are copied into owned fixed-address buffers; returned velocity is borrowed
scratch, overwritten by the next replay. No sampler or integration is captured.
Importing this module does not import Torch or initialize CUDA.
"""
from __future__ import annotations


_FIELDS = frozenset(('latents_action', 'timestep_action', 'context', 'context_mask',
                     'video_cache_k', 'video_cache_v', 'action_attention_mask'))


class VelocityGeometryMismatch(ValueError):
    """A request cannot use the already captured shape/layout/device contract."""


class VelocityCaptureError(RuntimeError):
    """Failed capture retaining its partial resource for explicit cleanup."""
    def __init__(self, message, *, release):
        super().__init__(message)
        self.release = release


def _signature(value, device, *, allow_cpu):
    import torch
    if isinstance(value, dict):
        return ('dict', tuple((key, _signature(value[key], device, allow_cpu=allow_cpu))
                              for key in sorted(value)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_signature(item, device, allow_cpu=allow_cpu) for item in value))
    if not isinstance(value, torch.Tensor):
        raise TypeError('Raw velocity graph inputs must be Tensor trees, not host scalars')
    if value.device != device or (not allow_cpu and value.device.type != 'cuda'):
        raise ValueError('Raw velocity graph requires same-device CUDA tensors')
    if value.layout != torch.strided or value.numel() == 0:
        raise ValueError('Raw velocity graph requires nonempty strided tensor layouts')
    span = 1
    for stride, size in sorted((stride, size) for stride, size in zip(value.stride(), value.shape) if size > 1):
        if stride < span:
            raise ValueError('Overlapping tensor layout is unsupported by the raw velocity graph')
        span += stride*(size-1)
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


def _clone_tree(value):
    import torch
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_tree(item) for item in value)
    out = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
    out.copy_(value)
    return out


def _copy_tree(target, source):
    if isinstance(target, dict):
        for key in target:
            _copy_tree(target[key], source[key])
    elif isinstance(target, (list, tuple)):
        for left, right in zip(target, source):
            _copy_tree(left, right)
    else:
        target.copy_(source)


def _native_calls(fn):
    model = getattr(fn, '__self__', None)
    if not callable(getattr(model, 'named_modules', None)):
        return {}
    return {name: module.calls for name, module in model.named_modules()
            if hasattr(module, '_runner') and hasattr(module, 'calls')}


def _capture_cuda(fn, static):
    import torch
    device = static['latents_action'].device
    stream = torch.cuda.current_stream(device)
    if stream == torch.cuda.default_stream(device):
        raise RuntimeError('Raw velocity capture requires the pipeline non-default stream')
    graph = torch.cuda.CUDAGraph()
    def release():
        # The closure also keeps static storage and the stream alive after a
        # partially failed capture. Reset is retried explicitly if it raises.
        _keep_alive = static
        stream.synchronize()
        graph.reset()
    try:
        for _ in range(3):
            fn(**static)
        stream.synchronize()
        before = _native_calls(fn)
        # Torch2.11's convenience context performs process-wide device/host
        # cache flushes. Own the raw lifecycle on the already-current side
        # stream instead; retain GLOBAL capture safety, never relaxed mode.
        graph.capture_begin(capture_error_mode='global')
        try:
            output = fn(**static)
        finally:
            graph.capture_end()
        after = _native_calls(fn)
    except BaseException as exc:
        raise VelocityCaptureError('FastWAM raw velocity capture failed', release=release) from exc
    calls = {name: count - before[name] for name, count in after.items() if count != before[name]}
    return graph.replay, output, release, {'warmup_evaluations': 3, 'encoded_bodies': 1,
                                         'captured_native_linear_calls': calls}


class FastWAMVelocityGraph:
    """Serialized private driver; ``capture`` injects the CUDA boundary in tests."""
    def __init__(self, fn, *, capture=None):
        self._fn = fn
        self._capture = capture or _capture_cuda
        self._injected_capture = capture is not None
        self._static = self._output = self._replay = self._release = self._geometry = None
        self._active = self._condition_ready = self._failed = self._closing = self._closed = False
        self._counts = {}
        self._capture_metadata = {}
        self._total_captures = self._total_replays = 0

    def _check_live(self):
        if self._closed or self._closing or self._failed:
            raise RuntimeError('FastWAM velocity graph is closed, closing or failed; cleanup is required')

    def begin_request(self):
        self._check_live()
        if self._active:
            raise RuntimeError('A velocity graph request is already active')
        self._active, self._condition_ready = True, False
        self._counts = dict(replays=0, captures=0, condition_refreshes=0,
                            warmup_evaluations=0, encoded_bodies=0)

    def end_request(self):
        if not self._active:
            raise RuntimeError('No active velocity graph request to end')
        self._active = False
        self._condition_ready = False

    def __call__(self, **kwargs):
        self._check_live()
        if not self._active:
            raise RuntimeError('Raw velocity evaluation requires an active request')
        if set(kwargs) != _FIELDS:
            raise ValueError('Raw velocity graph requires exactly seven named Tensor arguments')
        import torch
        state = kwargs['latents_action']
        if not isinstance(state, torch.Tensor):
            raise TypeError('latents_action must be a Tensor')
        geometry = _signature(kwargs, state.device, allow_cpu=self._injected_capture)
        if self._geometry is not None and geometry != self._geometry:
            raise VelocityGeometryMismatch(f'Raw velocity graph geometry changed: expected {self._geometry}, got {geometry}')
        if self._geometry is None:
            static = _clone_tree(kwargs)
            try:
                replay, output, release, metadata = self._capture(self._fn, static)
                # Keep an unpublished resource until close even if output
                # validation fails. A failed release must never lose its owner.
                self._static, self._output, self._release = static, output, release
                if (not isinstance(output, torch.Tensor) or output.shape != state.shape
                        or output.dtype != state.dtype or output.device != state.device):
                    raise ValueError('Captured raw velocity must match latent shape/device/dtype')
                self._geometry, self._replay = geometry, replay
                self._capture_metadata = metadata
                self._counts['captures'] += 1
                self._total_captures += 1
                for key in ('warmup_evaluations', 'encoded_bodies'):
                    self._counts[key] += metadata[key]
            except BaseException as exc:
                self._failed = True
                if isinstance(exc, VelocityCaptureError):
                    self._static, self._release = static, exc.release
                raise
        try:
            if not self._condition_ready:
                _copy_tree(self._static, kwargs)
                self._counts['condition_refreshes'] += 1
                self._condition_ready = True
            else:
                for key in ('latents_action', 'timestep_action'):
                    _copy_tree(self._static[key], kwargs[key])
            self._replay()
        except BaseException:
            self._failed = True
            raise
        self._counts['replays'] += 1
        self._total_replays += 1
        return self._output

    def snapshot(self):
        calls = self._capture_metadata.get('captured_native_linear_calls', {})
        return {**self._counts, 'total_captures': self._total_captures, 'total_replays': self._total_replays,
                'captured_native_linear_calls_per_replay': sum(calls.values()),
                'logical_native_linear_calls_in_replays': sum(calls.values()) * self._counts.get('replays', 0),
                'counter_semantics': 'Logical graph work; Python forwards and CPU native enqueues are separate counters'}

    def close(self):
        if self._closed:
            return
        self._closing = True
        if self._release is not None:
            self._release()
        self._static = self._output = self._replay = self._release = self._fn = self._capture = None
        self._geometry = None
        self._active = False
        self._closed = True
