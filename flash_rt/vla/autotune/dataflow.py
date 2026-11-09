"""Implementation-specific cost ledgers, not profiler-derived HBM measurements."""


def describe_dataflow(backend):
    config = getattr(backend, 'config', None)
    frontend = getattr(backend, 'frontend', None)
    if getattr(config, 'model', None) != 'pi05' or not hasattr(frontend, 'pipeline'):
        return None
    pipe = frontend.pipeline
    prefix = int(pipe.encoder_seq_len)
    actions = int(pipe.chunk_size)
    layers = 18 if getattr(pipe, 'prefix_export', False) else 17
    return {
        'basis': 'actual_runtime_rows_and_source_verified_pi05_bf16_geometry',
        'prefix_rows': prefix, 'action_rows': actions,
        'vision_tokens_before_pool': int(pipe.vision_seq),
        'vision_tokens_after_pool': int(pipe.vision_seq_enc),
        'prefix_ffn': {'layers': layers, 'M': prefix, 'K': 2048, 'N': 16384,
                       'linear_flops': layers * 6 * prefix * 2048 * 16384},
        'action_per_evaluation': {
            'layers': 18, 'M': actions, 'hidden': 1024,
            'q_width': 2048, 'kv_width': 256, 'ffn_width': 4096,
            'projection_and_ffn_flops': 18 * (
                2 * actions * 1024 * 2560 + 2 * actions * 2048 * 1024
                + 6 * actions * 1024 * 4096),
            'attention_product_flops': 18 * 4 * actions * (prefix + actions) * 2048,
            'logical_linear_weight_bytes': 18 * 2 * (
                1024 * 2560 + 2048 * 1024 + 3 * 1024 * 4096)},
        'notes': ['FMA=2 FLOPs; norm/activation/softmax omitted',
                  'Logical weights are not measured HBM traffic',
                  'Prefix KV reused across evaluations, not across observations',
                  'Post-ViT pooling reduces prefix rows, not the completed ViT work'],
    }
