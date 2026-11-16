"""Opt-in DM05 tensor dataflow/Inductor schedule for the existing SM80 entry.

Only bounded, reusable layer functions are compiled. Weights, rotary tensors and
time modulations are tensor arguments, not Python layer/step specializations.
The frontend still owns the whole-policy CUDA graph and all public host IO.
There is deliberately no eager-on-compiler-error path.
"""

from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from .pipeline_cuda_sm80 import Dm05PipelineSm80, adaptive_rms, rms, rotate
from .weights import ACTION, PREFIX, VISION


def _vision_layer(hidden, weights):
    n1, b1, qkv, qb, out, ob, n2, b2, up, ub, down, db = weights
    x = F.layer_norm(hidden, (1152,), n1, b1, 1e-6)
    q, k, v = F.linear(x, qkv, qb).split(1152, dim=-1)
    q, k, v = (t.view(3, 1024, 16, 72).transpose(1, 2) for t in (q, k, v))
    x = F.scaled_dot_product_attention(q, k, v, scale=72 ** -0.5)
    x = x.transpose(1, 2).reshape(3, 1024, 1152)
    hidden = hidden + F.linear(x, out, ob)
    x = F.layer_norm(hidden, (1152,), n2, b2, 1e-6)
    x = F.gelu(F.linear(x, up, ub), approximate='tanh')
    return hidden + F.linear(x, down, db)


def _qkv(x, weight, q_norm, k_norm, cos, sin):
    q, k, v = F.linear(x, weight).split((2048, 1024, 1024), dim=-1)
    q = q.view(1, x.shape[1], 8, 256)
    k = k.view(1, x.shape[1], 4, 256)
    v = v.view(1, x.shape[1], 4, 256)
    return rotate(rms(q, q_norm), cos, sin), rotate(rms(k, k_norm), cos, sin), v


def _mlp(x, gate_up, down):
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(F.gelu(gate, approximate='tanh') * up, down)


def _prefix_layer(hidden, weights, cos, sin, mask, key_cache, value_cache):
    n1, qkv, qn, kn, out, n2, n3, gate_up, down, n4 = weights
    x = rms(hidden, n1)
    q, k, v = _qkv(x, qkv, qn, kn, cos, sin)
    key_cache[:, :hidden.shape[1]].copy_(k)
    value_cache[:, :hidden.shape[1]].copy_(v)
    # Preserve arbitrary bidirectional-image/causal-text mask support. The
    # unmasked action site below uses native GQA without these KV repeats.
    k = k.transpose(1, 2).repeat_interleave(2, dim=1)
    v = v.transpose(1, 2).repeat_interleave(2, dim=1)
    x = F.scaled_dot_product_attention(q.transpose(1, 2), k, v,
                                       attn_mask=mask, scale=0.0625)
    x = x.transpose(1, 2).reshape(1, hidden.shape[1], 2048)
    hidden = hidden + rms(F.linear(x, out), n2)
    return hidden + rms(_mlp(rms(hidden, n3), gate_up, down), n4)


def _action_layer(hidden, weights, input_modulation, mlp_modulation,
                  cos, sin, key_cache, value_cache):
    _, qkv, qn, kn, out, n2, _, gate_up, down, n4 = weights
    x, attention_gate = adaptive_rms(hidden, input_modulation)
    q, k, v = _qkv(x, qkv, qn, kn, cos, sin)
    key_cache[:, -50:].copy_(k)
    value_cache[:, -50:].copy_(v)
    x = F.scaled_dot_product_attention(
        q.transpose(1, 2), key_cache.transpose(1, 2), value_cache.transpose(1, 2),
        scale=0.0625, enable_gqa=True)
    x = x.transpose(1, 2).reshape(1, 50, 2048)
    hidden = hidden + rms(F.linear(x, out), n2) * attention_gate
    x, mlp_gate = adaptive_rms(hidden, mlp_modulation)
    return hidden + rms(_mlp(x, gate_up, down), n4) * mlp_gate


class Dm05CompiledPipelineSm80(Dm05PipelineSm80):
    """Same BF16/Euler10 equations with packed projections and compiled layers."""

    def __init__(self, checkpoint):
        start = time.perf_counter()
        super().__init__(checkpoint, native=False)
        w = self.weights

        def cat(base, names, suffix='weight'):
            return torch.cat([w[f'{base}.{name}.{suffix}'] for name in names], dim=0)

        self.vision_weights = []
        for i in range(27):
            base = f'{VISION}.encoder.layers.{i}'
            self.vision_weights.append((
                w[f'{base}.layer_norm1.weight'], w[f'{base}.layer_norm1.bias'],
                cat(f'{base}.self_attn', ('q_proj', 'k_proj', 'v_proj')),
                cat(f'{base}.self_attn', ('q_proj', 'k_proj', 'v_proj'), 'bias'),
                w[f'{base}.self_attn.out_proj.weight'], w[f'{base}.self_attn.out_proj.bias'],
                w[f'{base}.layer_norm2.weight'], w[f'{base}.layer_norm2.bias'],
                w[f'{base}.mlp.fc1.weight'], w[f'{base}.mlp.fc1.bias'],
                w[f'{base}.mlp.fc2.weight'], w[f'{base}.mlp.fc2.bias']))
        self.layer_weights = {}
        for branch in (PREFIX, ACTION):
            layers = []
            for i in range(34):
                base = f'{branch}.layers.{i}'
                layers.append((
                    w[f'{base}.input_layernorm.weight'],
                    cat(f'{base}.self_attn', ('q_proj', 'k_proj', 'v_proj')),
                    w[f'{base}.self_attn.q_norm.weight'], w[f'{base}.self_attn.k_norm.weight'],
                    w[f'{base}.self_attn.o_proj.weight'],
                    w[f'{base}.post_attention_layernorm.weight'],
                    w[f'{base}.pre_feedforward_layernorm.weight'],
                    cat(f'{base}.mlp', ('gate_proj', 'up_proj')),
                    w[f'{base}.mlp.down_proj.weight'],
                    w[f'{base}.post_feedforward_layernorm.weight']))
            self.layer_weights[branch] = layers
        # Default Inductor GEMM selection bounds cold setup; fusion is still
        # enabled. Explicitly disable compiler-owned graphs: capture has one
        # authority, Dm05ProfileSm80, after these functions have warmed.
        options = {'triton.cudagraphs': False}
        self.compiled_vision = torch.compile(_vision_layer, fullgraph=True, dynamic=False,
                                             options=options)
        self.compiled_prefix = torch.compile(_prefix_layer, fullgraph=True, dynamic=False,
                                             options=options)
        self.compiled_action = torch.compile(_action_layer, fullgraph=True, dynamic=False,
                                             options=options)
        self.preparation_s = time.perf_counter() - start
        self.prepared_lengths = set()
        self.compile_warmup = []

    def vision(self, pixels, attention):
        patch = f'{VISION}.embeddings.patch_embedding'
        hidden = F.conv2d(pixels, self.weights[f'{patch}.weight'],
                          self.weights[f'{patch}.bias'], stride=14)
        hidden = hidden.flatten(2).transpose(1, 2).contiguous()
        hidden = hidden + self.weights[f'{VISION}.embeddings.position_embedding.weight']
        for weights in self.vision_weights:
            hidden = self.compiled_vision(hidden, weights)
        hidden = self.layer_norm(hidden, f'{VISION}.post_layernorm')
        grid = hidden.transpose(1, 2).reshape(3, 1152, 32, 32)
        pooled = F.avg_pool2d(grid, 2, stride=2).flatten(2).transpose(1, 2).contiguous()
        pooled = rms(pooled, self.weights[
            'model.vlm.model.multi_modal_projector.mm_soft_emb_norm.weight'])
        return F.linear(pooled, self.weights['_projector.weight']).reshape(768, 2560)

    def prefill(self, profile):
        features = self.vision(profile.pixels, None)
        hidden = F.embedding(profile.ids, self.weights[f'{PREFIX}.embed_tokens.weight']) * self.embed_scale
        hidden = hidden.reshape(-1, 2560).index_copy(0, profile.image_rows, features)[None]
        for i, weights in enumerate(self.layer_weights[PREFIX]):
            hidden = self.compiled_prefix(
                hidden, weights, *profile.prefix_rope[(i + 1) % 6 == 0],
                profile.attention.prefix_mask, profile.keys[i], profile.values[i])

    def velocity(self, state, profile, step):
        hidden = F.linear(state, self.weights['model.action_in_proj.weight'],
                          self.weights['model.action_in_proj.bias'])
        inputs, mlps, final = self.modulations[step]
        for i, weights in enumerate(self.layer_weights[ACTION]):
            hidden = self.compiled_action(
                hidden, weights, inputs[i], mlps[i],
                *profile.suffix_rope[(i + 1) % 6 == 0], profile.keys[i], profile.values[i])
        hidden, _ = adaptive_rms(hidden, final)
        return F.linear(hidden, self.weights['model.action_out_proj.weight'],
                        self.weights['model.action_out_proj.bias'])

    def prepare(self, profile):
        """Compile and execute every region/stride variant before manual capture."""
        if profile.length not in self.prepared_lengths:
            start = time.perf_counter()
            self.forward(profile)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self.prepared_lengths.add(profile.length)
            self.compile_warmup.append({'prefix_length': profile.length,
                                        'compile_plus_first_forward_s': elapsed})

    def compilation_metadata(self):
        return {'backend': 'inductor', 'mode': 'default', 'fullgraph': True,
                'dynamic': False, 'internal_cudagraphs': False,
                'units': ['vision-layer', 'prefix-layer', 'action-layer'],
                'projection_packing': ['QKV', 'MLP-gate-up'],
                'attention': 'direct-SDPA; unmasked-action-native-GQA; masked-prefix-KV-repeat',
                'eager_fallback': False, 'preparation_s': self.preparation_s,
                'compile_warmup': list(self.compile_warmup)}
