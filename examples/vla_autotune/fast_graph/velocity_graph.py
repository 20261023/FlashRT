"""Experimental fixed-geometry raw-velocity pilot, not a production frontend.

One serialized request owns condition values until completion. begin_request()
must precede EVERY request even when pointers are reused. Outputs alias graph
scratch, like the common raw-velocity contract; samplers must consume/clone them.
"""
from __future__ import annotations

import os


FIELDS = {'latents_action', 'timestep_action', 'context', 'context_mask',
          'video_cache_k', 'video_cache_v', 'action_attention_mask'}


def signature(value):
    if isinstance(value, dict):
        return ('dict', tuple((key, signature(value[key])) for key in sorted(value)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(signature(x) for x in value))
    if not all(hasattr(value, name) for name in ('shape', 'dtype', 'device', 'stride', 'copy_', 'clone')):
        raise TypeError('Velocity graph inputs must be tensor trees, never implicit host scalars')
    return (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))


def clone_tree(value):
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tree(x) for x in value)
    return value.detach().clone()


def copy_tree(target, source):
    if isinstance(target, dict):
        for key in target:
            copy_tree(target[key], source[key])
    elif isinstance(target, (list, tuple)):
        for a, b in zip(target, source):
            copy_tree(a, b)
    else:
        target.copy_(source)


def native_calls(fn):
    model = getattr(fn, '__self__', None)
    if model is None or not hasattr(model, 'named_modules'):
        return {}
    return {name: module.calls for name, module in model.named_modules()
            if hasattr(module, '_runner') and hasattr(module, 'calls')}


def capture_cuda(fn, static):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID', '').isdigit():
        raise RuntimeError('CUDA capture requires the allocated srun step')
    import torch
    device = static['latents_action'].device
    if device.type != 'cuda':
        raise RuntimeError('Real velocity capture requires CUDA tensors')
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(**static)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    before = native_calls(fn)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):  # default GLOBAL capture mode
        output = fn(**static)
    after = native_calls(fn)
    calls = {key: value - before[key] for key, value in after.items() if value != before[key]}
    def close():
        torch.cuda.synchronize(device)
        graph.reset()  # only this experiment's owned graph; never a global cache flush
    return graph.replay, output, close, {'warmup_evaluations': 3, 'captured_evaluators': 1,
                                        'captured_native_linear_calls': calls}


class VelocityGraph:
    def __init__(self, fn, *, capture=None):
        self.fn, self.capture = fn, capture or capture_cuda
        self.enabled, self.closed = True, False
        self.static = self.output = self.replay = self.release = self.geometry = None
        self.capture_metadata = {}
        self.requests = 0
        self.total_replays = 0
        self.total_captures = 0
        self.condition_ready = False
        self.request_counts = {}

    def begin_request(self):
        if self.closed:
            raise RuntimeError('Velocity graph is closed')
        self.requests += 1
        self.condition_ready = False
        self.request_counts = {'replays': 0, 'captures': 0, 'warmup_evaluations': 0,
                               'captured_evaluators': 0, 'condition_refreshes': 0}

    def __call__(self, **kwargs):
        if self.closed or not self.requests:
            raise RuntimeError('Velocity graph is closed or begin_request was not called')
        if not self.enabled:
            return self.fn(**kwargs)
        if set(kwargs) != FIELDS:
            raise ValueError('Raw velocity graph requires exactly the seven observed tensor arguments')
        geometry = signature(kwargs)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError('Velocity graph geometry changed; no stale graph or silent eager fallback')
        if self.geometry is None:
            static = clone_tree(kwargs)
            replay, output, release, meta = self.capture(self.fn, static)
            if (output.shape != kwargs['latents_action'].shape or output.dtype != kwargs['latents_action'].dtype
                    or output.device != kwargs['latents_action'].device):
                release()
                raise ValueError('Captured raw velocity does not match latent shape/dtype/device')
            self.static, self.geometry = static, geometry
            self.replay, self.output, self.release = replay, output, release
            self.capture_metadata = meta
            self.total_captures += 1
            self.request_counts['captures'] += 1
            for key in ('warmup_evaluations', 'captured_evaluators'):
                self.request_counts[key] += meta[key]
        if not self.condition_ready:
            copy_tree(self.static, kwargs)
            self.condition_ready = True
            self.request_counts['condition_refreshes'] += 1
        else:
            for key in ('latents_action', 'timestep_action'):
                copy_tree(self.static[key], kwargs[key])
        self.replay()
        self.total_replays += 1
        self.request_counts['replays'] += 1
        return self.output

    def snapshot(self):
        calls = self.capture_metadata.get('captured_native_linear_calls', {})
        return {**self.request_counts,
                'captured_native_linear_calls_per_replay': sum(calls.values()),
                'logical_native_linear_calls_in_replays': sum(calls.values()) * self.request_counts.get('replays', 0),
                'counter_semantics': 'Replays/captured body calls, NOT CPU cuBLAS API enqueue counts'}

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.release is not None:
                self.release()
        finally:
            self.static = self.output = self.replay = self.release = self.fn = None
