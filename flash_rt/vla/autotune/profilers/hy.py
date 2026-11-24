"""Hy diagnostics through temporary wrappers; production model code is untouched."""


def install(session):
    frontend = session.frontend
    for entry in getattr(frontend, '_vit_graph_cache', {}).values():
        if entry.get('graph') is not None:
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
    session.expected = ['vision', 'prefix', 'velocity']
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
