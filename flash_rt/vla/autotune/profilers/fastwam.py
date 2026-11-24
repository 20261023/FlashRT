"""FastWAM condition/velocity diagnostics, including compiled callable owners."""


def install(session):
    pipe = session.frontend.pipeline
    model = getattr(pipe, 'model', None)
    # All three call sites must exist before calling this a split condition.
    split = (callable(getattr(model, '_encode_input_image_latents_tensor', None))
             and callable(getattr(getattr(model, 'video_expert', None), 'prepare', None))
             and callable(getattr(pipe, '_compiled_prefill', None)
                          or getattr(getattr(model, 'mot', None), 'prefill_video_cache_tensor', None)))
    if split:
        session.host_method(pipe, '_initial_state', 'host_input', ('host.input',),
                            'FastWAM input helpers: validation, noise/context/image staging and local IO waits')
        session.host_method(pipe, '_observation_tensors', 'host_input', ('host.input',),
                            'FastWAM input helpers: validation, noise/context/image staging and local IO waits')
        session.gpu_method(model, '_encode_input_image_latents_tensor', 'vision',
                           ('vision.encode',), 'First-frame VAE encode')
        session.gpu_method(model.video_expert, 'prepare', 'condition_tokens',
                           ('prefix.prefill',), 'Video tokens, timestep and context preparation')
        if callable(getattr(pipe, '_compiled_prefill', None)):
            owner, name = pipe, '_compiled_prefill'
        else:
            owner, name = model.mot, 'prefill_video_cache_tensor'
        session.gpu_method(owner, name, 'prefix', ('prefix.prefill',),
                           'Video-expert prefill only; excludes VAE and condition token preparation')
        for owner, name, key, boundary in (
                (model, '_append_proprio_to_context', 'condition_context', 'Proprio context projection'),
                (model, '_build_mot_attention_mask', 'condition_mask', 'Condition attention mask'),
                (getattr(pipe, '_text_kv_cache', None), 'prepare', 'condition_text_kv',
                 'Optional per-request text cross-attention KV precomputation')):
            session.gpu_method(owner, name, key, ('prefix.prefill',), boundary)
        session.expected = ['host_input', 'vision', 'condition_tokens', 'prefix', 'velocity']
        session.notes.append('Condition subregions are disjoint; unwrapped condition bookkeeping remains residual')
    else:
        # Historical or incomplete adapter objects cannot claim a VAE/prefill
        # split. Preserve the aggregate evidence without inventing components.
        session.gpu_method(pipe, '_prepare_condition', 'condition_vae_video_prefill',
                           ('host.input', 'vision.encode', 'prefix.prefill'),
                           'Aggregate condition preparation; subregion hooks unavailable')
        session.expected = ['condition_vae_video_prefill', 'velocity']
        session.notes.append('Condition subregion hooks unavailable: aggregate evidence only')
    session.gpu_method(pipe, '_evaluate_clock_velocity', 'velocity',
                       ('denoise.velocity',), 'One raw velocity evaluation at the banked clock')
    # execution_metadata runs after the final finite check and before D2H. Its
    # trailing region is a real output boundary for either sampling policy.
    metadata = getattr(pipe, 'execution_metadata', None)
    if callable(metadata):
        def output_boundary(*args, **kwargs):
            value = metadata(*args, **kwargs)
            session.synchronize()
            session.begin_host('host_output')
            return value
        session.replace(pipe, 'execution_metadata', output_boundary)
        session.expected.append('host_output')
        session.notes.append('FastWAM output starts after finite-check completion; preceding routing stays residual')


def describe_dataflow(backend):
    pipe = backend.frontend.pipeline
    return {
        'basis': 'runtime_pipeline_attributes_and_source_seams',
        'measurement_kind': 'static_dataflow_ledger',
        'runtime_geometry': {
            key: getattr(pipe, key) for key in ('action_horizon', 'action_dim', 'time_scale')
            if isinstance(getattr(pipe, key, None), int)},
        'cache_text_crossattn_kv': bool(getattr(pipe, 'cache_text_crossattn_kv', False)),
        'notes': ['First-frame VAE and video KV prefill run once per request',
                  'Text KV precompute is present only when cache_text_crossattn_kv is enabled',
                  'Logical source ledger; no measured HBM traffic or kernel bound claim'],
    }
