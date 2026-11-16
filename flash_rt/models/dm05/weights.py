"""Strict Dexmal/DM05 base checkpoint mapping, without an upstream GPU model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


CHECKPOINT_REVISION = '5cd18734814abb075a9ccfd9ad6d16777b5cf10e'
MODEL_SHA256 = 'b7da77f516ebd0c0c68faed2f0cfe1c68985b7b874e49dda4292d8d286de20dd'
SOURCE_REVISION = '03f799120d425ead51d9482bb1da950b6dabade9'
VISION = 'model.vlm.model.vision_tower.vision_model'
PREFIX = 'model.vlm.model.language_model'
ACTION = 'model.action_expert'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_geometry(config):
    """Reject other checkpoints, precision profiles and Gemma configurations."""
    expected = {
        'model_type': 'dm05', 'action_dim': 32, 'chunk_size': 50,
        'vlm_config/mm_tokens_per_image': 256,
        'vlm_config/image_token_index': 262144,
        'vlm_config/vision_config/hidden_size': 1152,
        'vlm_config/vision_config/intermediate_size': 4304,
        'vlm_config/vision_config/num_hidden_layers': 27,
        'vlm_config/vision_config/num_attention_heads': 16,
        'vlm_config/vision_config/image_size': 448,
        'vlm_config/vision_config/patch_size': 14,
        'vlm_config/vision_config/layer_norm_eps': 1e-6,
    }
    for branch, width, ff in (('action_config', 1024, 4096),
                               ('vlm_config/text_config', 2560, 10240)):
        for name, value in {
            'hidden_size': width, 'intermediate_size': ff, 'num_hidden_layers': 34,
            'num_attention_heads': 8, 'num_key_value_heads': 4, 'head_dim': 256,
            'rms_norm_eps': 1e-6, 'query_pre_attn_scalar': 256,
            'sliding_window': 4096, 'hidden_activation': 'gelu_pytorch_tanh',
            'attention_bias': False, 'attn_logit_softcapping': None,
            'rope_parameters/sliding_attention/rope_theta': 10000.0,
            'rope_parameters/full_attention/rope_theta': 1000000.0,
            'rope_parameters/full_attention/factor': 8.0,
            'rope_parameters/full_attention/rope_type': 'linear',
            'rope_parameters/sliding_attention/rope_type': 'default',
            'layer_types': ['full_attention' if (i + 1) % 6 == 0 else 'sliding_attention'
                            for i in range(34)],
        }.items():
            expected[f'{branch}/{name}'] = value
    for path, value in expected.items():
        current = config
        for key in path.split('/'):
            if not isinstance(current, dict) or key not in current:
                raise ValueError(f'DM05 checkpoint is missing config {path}')
            current = current[key]
        if current != value:
            raise ValueError(f'Unsupported DM05 config {path}: {current!r}, expected {value!r}')


def required_shapes():
    shapes = {}

    def linear(name, source, target, bias=False):
        shapes[f'{name}.weight'] = (target, source)
        if bias:
            shapes[f'{name}.bias'] = (target,)

    def norm(name, width, bias=False):
        shapes[f'{name}.weight'] = (width,)
        if bias:
            shapes[f'{name}.bias'] = (width,)

    shapes[f'{VISION}.embeddings.patch_embedding.weight'] = (1152, 3, 14, 14)
    shapes[f'{VISION}.embeddings.patch_embedding.bias'] = (1152,)
    shapes[f'{VISION}.embeddings.position_embedding.weight'] = (1024, 1152)
    for i in range(27):
        base = f'{VISION}.encoder.layers.{i}'
        for name in ('layer_norm1', 'layer_norm2'):
            norm(f'{base}.{name}', 1152, True)
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            linear(f'{base}.self_attn.{name}', 1152, 1152, True)
        linear(f'{base}.mlp.fc1', 1152, 4304, True)
        linear(f'{base}.mlp.fc2', 4304, 1152, True)
    norm(f'{VISION}.post_layernorm', 1152, True)
    norm('model.vlm.model.multi_modal_projector.mm_soft_emb_norm', 1152)
    shapes['model.vlm.model.multi_modal_projector.mm_input_projection_weight'] = (1152, 2560)
    shapes[f'{PREFIX}.embed_tokens.weight'] = (262208, 2560)
    for branch, width, ff in ((PREFIX, 2560, 10240), (ACTION, 1024, 4096)):
        for i in range(34):
            base = f'{branch}.layers.{i}'
            for name in ('input_layernorm', 'post_attention_layernorm',
                         'pre_feedforward_layernorm', 'post_feedforward_layernorm'):
                norm(f'{base}.{name}', width)
            for name in ('q_norm', 'k_norm'):
                norm(f'{base}.self_attn.{name}', 256)
            for name, target in (('q_proj', 2048), ('k_proj', 1024), ('v_proj', 1024)):
                linear(f'{base}.self_attn.{name}', width, target)
            linear(f'{base}.self_attn.o_proj', 2048, width)
            linear(f'{base}.mlp.gate_proj', width, ff)
            linear(f'{base}.mlp.up_proj', width, ff)
            linear(f'{base}.mlp.down_proj', ff, width)
        norm(f'{branch}.norm', width)
    for i in range(34):
        linear(f'{ACTION}.input_time_modulators.{i}', 1024, 3072, True)
        linear(f'{ACTION}.mlp_time_modulators.{i}', 1024, 3072, True)
    linear(f'{ACTION}.final_time_modulator', 1024, 3072, True)
    linear('model.time_mlp_in', 1024, 1024, True)
    linear('model.time_mlp_out', 1024, 1024, True)
    linear('model.action_in_proj', 32, 1024, True)
    linear('model.action_out_proj', 1024, 32, True)
    return shapes


def load_weights(checkpoint):
    root = Path(checkpoint).expanduser().resolve()
    config = json.loads((root / 'config.json').read_text())
    validate_geometry(config)
    path = root / 'model.safetensors'
    actual_hash = sha256_file(path)
    if actual_hash != MODEL_SHA256:
        raise ValueError(f'DM05 requires the fixed base model SHA256 {MODEL_SHA256}; got {actual_hash}')
    shapes = required_shapes()
    weights = {}
    with safe_open(str(path), framework='pt', device='cuda') as source:
        keys = set(source.keys())
        # The tied lm_head may be omitted by safetensors, but no other extra
        # weights (adapters, training heads, etc.) may silently be ignored.
        if keys - set(shapes) - {'model.vlm.lm_head.weight'} or set(shapes) - keys:
            raise ValueError('DM05 base checkpoint tensor keys do not match the fixed architecture')
        for name, shape in shapes.items():
            tensor = source.get_tensor(name)
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise ValueError(f'{name}: expected BF16 {shape}, got {tensor.dtype} {tuple(tensor.shape)}')
            weights[name] = tensor.contiguous()
    return weights, {'checkpoint_revision': CHECKPOINT_REVISION,
                     'model_sha256': actual_hash, 'model_bytes': path.stat().st_size,
                     'config_sha256': sha256_file(root / 'config.json')}
