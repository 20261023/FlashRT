"""Hy diagnostics through temporary wrappers; production model code is untouched."""


def _split_vision(session, entry):
    """Recapture the plain ViT/merger body with its existing input/output buffers."""
    frontend = session.frontend
    pipe = getattr(frontend, 'pipe', None)
    if (not {'graph', 'img', 'out'} <= entry.keys()
            or not callable(getattr(pipe, 'vit_forward', None))
            or not callable(getattr(pipe, 'merger_forward', None))
            or not session.take_reference()):
        return False
    scope = 'diagnostic_same_program_recapture'
    vit = session.pair('vit_encoder', ('vision.encode',), 'Hy ViT forward',
                       external=True, scope=scope)
    merger = session.pair('merger', ('vision.project',), 'Hy merger and static output copy',
                          external=True, scope=scope)

    def body():
        vit[0].record()
        features = pipe.vit_forward(entry['img'])
        vit[1].record()
        merger[0].record()
        entry['out'].copy_(pipe.merger_forward(features))
        merger[1].record()

    session.replace_item(entry, 'graph', session.capture(body))
    return True


def install(session):
    frontend = session.frontend
    cache = getattr(frontend, '_vit_graph_cache', {})
    vision_keys = ['vision']
    for entry in cache.values():
        if len(cache) == 1 and entry.get('graph') is not None and _split_vision(session, entry):
            vision_keys = ['vit_encoder', 'merger']
        elif entry.get('graph') is not None:
            session.replace_item(entry, 'graph', session.graph(
                entry['graph'], 'vision', ('vision.encode', 'vision.project'),
                '_vit_graph_cache graph replay: ViT and merger'))
    for entry in getattr(frontend, '_graph_cache', {}).values():
        if entry.get('prefix_graph') is not None:
            session.replace_item(entry, 'prefix_graph', session.graph(
                entry['prefix_graph'], 'prefix', ('prefix.prefill',),
                'prefix_graph replay including state embedding'))
        for key, graph in entry.get('velocity_graphs', {}).copy().items():
            session.replace_item(entry['velocity_graphs'], key, session.graph(
                graph, 'velocity', ('denoise.velocity',), 'One velocity_graph replay'))
    # The eager path has the same source bodies without graph-cache entries.
    if not getattr(frontend, 'use_cuda_graph', True):
        session.gpu_method(frontend, '_vit_merge', 'vision',
                           ('vision.encode', 'vision.project'), 'Eager ViT and merger')
        session.gpu_method(frontend, '_prefix_body', 'prefix',
                           ('prefix.prefill',), 'Eager prefix and state embedding')
        session.gpu_method(frontend, '_velocity_body', 'velocity',
                           ('denoise.velocity',), 'One eager action field evaluation')
    complete_io = (callable(getattr(frontend, '_vit_merge', None))
                   and callable(getattr(frontend, '_graph_forward', None)))
    if complete_io:
        session.boundary(frontend, '_vit_merge', resume_input=True)
        session.boundary(frontend, '_graph_forward', output=True)
    session.start_host_input = complete_io
    session.expected = [*vision_keys, 'prefix', 'velocity']
    if session.start_host_input:
        session.expected += ['host_input', 'host_output']
    session.notes.append('Hy host regions exclude ViT/flow compute and explicit drain waits')


def describe_dataflow(backend):
    frontend = backend.frontend
    return {
        'basis': 'runtime_graph_keys_and_source_seams', 'measurement_kind': 'static_dataflow_ledger',
        'runtime_geometry': {
            name: getattr(frontend, name) for name in
            ('chunk', 'max_action_dim', 'max_state_dim', 'n_heads', 'n_kv', 'head_dim')
            if isinstance(getattr(frontend, name, None), (int, str, bool))},
        'graph_profiles': len(getattr(frontend, '_graph_cache', {})),
        'notes': ['Prefix KV is reused across denoising evaluations within one request',
                  'No HBM traffic, kernel utilization or bottleneck classification is measured here'],
    }
