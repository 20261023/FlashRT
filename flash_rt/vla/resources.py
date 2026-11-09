"""Instance-owned native Pi graph cleanup, without global cache operations."""

import ctypes


def _pointer(value):
    return int(getattr(value, 'value', value) or 0)


def release_pi_frontend_graphs(frontend):
    """Release all Pi full/decoder/probe graphs before dropping their frontend.

    Pi's raw CUDAGraph wrapper does not own a destructor. Torch graphs in the
    other model families have their own RAII and are deliberately not handled
    here. Every graph/exec handle is destroyed once, even with aliased prompt
    caches. Framework-owned streams and process-global caches are not destroyed.
    """
    candidates = [getattr(frontend, 'pipeline', None),
                  getattr(frontend, '_fixed_pipeline', None)]
    candidates.extend(getattr(frontend, '_prompt_pipeline_cache', {}).values())
    candidates.extend(context.get('pipeline') for context in
                      getattr(frontend, '_batch_ctx', {}).values())
    release_pi_pipeline_graphs(candidates)


def release_pi_pipeline_graphs(candidates):
    """Also used when Pi0 replaces, rather than caches, a prompt pipeline."""
    pipelines = {id(pipe): pipe for pipe in candidates if pipe is not None}
    entries = []
    for pipe in pipelines.values():
        graphs = [getattr(pipe, '_graph', None), getattr(pipe, '_decoder_only_graph', None)]
        graphs.extend(getattr(pipe, '_velocity_graphs', {}).values())
        graphs.extend(record.cuda_graph for record in
                      getattr(pipe, '_flashrt_subgraph_captured_graphs', ()))
        for graph in graphs:
            if graph is not None and (
                    _pointer(getattr(graph, '_graph_exec', None))
                    or _pointer(getattr(graph, '_graph', None))):
                entries.append((pipe, graph))
    if not entries:
        return
    runtime = entries[0][0]._cudart
    # Use pointer-width ABI signatures, not ctypes' default C-int conversion.
    for name in ('cudaStreamSynchronize', 'cudaGraphExecDestroy', 'cudaGraphDestroy'):
        function = getattr(runtime, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int

    def check(status, operation):
        if status:
            raise RuntimeError(f'CUDA error {status}: {operation}')

    streams = {_pointer(getattr(pipe, '_graph_stream', None)) for pipe, _ in entries}
    for stream in streams:
        check(runtime.cudaStreamSynchronize(ctypes.c_void_p(stream)), 'close graph stream sync')
    seen_execs, seen_graphs = set(), set()
    for _, graph in entries:
        for attribute, name, seen in (
                ('_graph_exec', 'cudaGraphExecDestroy', seen_execs),
                ('_graph', 'cudaGraphDestroy', seen_graphs)):
            pointer = _pointer(getattr(graph, attribute, None))
            if pointer and pointer not in seen:
                check(getattr(runtime, name)(ctypes.c_void_p(pointer)), name)
                seen.add(pointer)
            handle = getattr(graph, attribute, None)
            if hasattr(handle, 'value'):
                handle.value = None
        graph._captured = False
    for pipe in pipelines.values():
        # Prevent ProbeFlow's legacy finalizer from trying to synchronize the
        # now-empty bank after the frontend's external stream has been released.
        pipe._graph = None
        if hasattr(pipe, '_decoder_only_graph'):
            pipe._decoder_only_graph = None
        if hasattr(pipe, '_velocity_graphs'):
            pipe._velocity_graphs.clear()
