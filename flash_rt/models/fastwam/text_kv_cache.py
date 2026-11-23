"""Request-owned ActionDiT text K/V for the real first-frame FastWAM path.

The tensor core below adapts FastWAM._denoise_action_with_video_cache,
ActionDiT.prepare, MoT.forward_action_with_video_cache_tensor and
CrossAttention.forward at upstream 7faa71108368fbb3b6885649f112af607427a2d4.
The upstream files stay read-only.

Only text_embedding and cross-attention norm_k(k(context))/v(context) move to
request preparation. Self-attention (including video K/V), query projection,
modulation, masks, attention and post-projection keep the upstream call order.
No module is patched, and no request tensor is stored on the model or adapter.

Upstream portions are distributed under the following license:

MIT License

Copyright (c) 2026 The FastWAM Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Except for third-party code and files that retain their original license
headers, this repository is released under the MIT License.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import textwrap
import tokenize

import torch
from torch import nn


# Token fingerprints ignore whitespace/comments and are independent of Python's
# AST schema. Refuse an unknown source seam rather than bypass changed semantics
# in an external source_dir that is outside FlashRT's snapshot identity.
_SOURCE_CONTRACT = {
    "FastWAM._denoise_action_with_video_cache":
        "25417e7f995ef2407cb97a611db54b1ce1950faf61404fdc46d3fe5b8c4f80db",
    "ActionDiT.prepare":
        "d4dc8b4cc66c5694a6475cb10a95d25cac570f4c065598f276bdd4fca15f8883",
    "MoT.forward_action_with_video_cache_tensor":
        "96f0f106e0c276c2187a692405b9740fbec964ad3e0d49d796bd339fc68b1bbc",
    "MoT._apply_expert_post_block_tensor":
        "46981b1ca3efc15512c9b75e4bd858f3a6edf4454bb2f2878412b8f21624d506",
    "CrossAttention.forward":
        "e0c0fa3bccf3c42b007ee76547996ee8cf2ae7eaac1e2e2cca98f25db1e21f85",
    "RMSNorm.forward":
        "267394eb4eea2337df340a54ebb66f0126e59cc035cfb9d31d33d05447b456e7",
    "RMSNorm.norm":
        "7b8c66f4ad030755443e1a281ed41e01ae9555a309e8b2fe8e8bd247d9359aba",
}


def _check_source(fn, name):
    fn = getattr(fn, "__func__", fn)
    if (not inspect.isfunction(fn) or hasattr(fn, "__wrapped__")
            or hasattr(fn, "_torchdynamo_orig_callable")):
        raise ValueError(f"cache_text_crossattn_kv requires the unwrapped pinned {name}")
    try:
        source = textwrap.dedent(inspect.getsource(fn))
        ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                   tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
        tokens = (part.string for part in tokenize.generate_tokens(io.StringIO(source).readline)
                  if part.type not in ignored)
        digest = hashlib.sha256("\0".join(tokens).encode()).hexdigest()
    except (OSError, TypeError, SyntaxError, tokenize.TokenError) as exc:
        raise ValueError(f"cache_text_crossattn_kv cannot verify source for {name}") from exc
    if digest != _SOURCE_CONTRACT[name]:
        raise ValueError(f"cache_text_crossattn_kv does not support modified {name}; use the pinned FastWAM source")
    return fn


class FastWAMTextKVCache:
    """Inference-only tensor core; cached tensors are explicit call arguments.

    Construction verifies every upstream method that this core bypasses.
    ``prepare`` must run afresh after proprio has been appended, even when the
    caller reuses the same observation object, prompt or tensor addresses.
    ``denoise`` is tensor-only and can be compiled with fullgraph=True or used
    as the manual CUDA Graph body.
    """

    def __init__(self, model, *, device):
        self.model, self.device = model, device
        self.owner = object()
        action, mot = model.action_expert, model.mot
        if (type(action).__module__ != "fastwam.models.wan22.action_dit"
                or type(action).__name__ != "ActionDiT"
                or type(mot).__module__ != "fastwam.models.wan22.mot"
                or type(mot).__name__ != "MoT"):
            raise ValueError("cache_text_crossattn_kv supports only the pinned ActionDiT/MoT tensor core")
        if (set(mot.mixtures) != {"action", "video"}
                or mot.mixtures["action"] is not action
                or mot.mixtures["video"] is not model.video_expert):
            raise ValueError("cache_text_crossattn_kv requires the real FastWAM expert/MoT aliases")
        if (getattr(action, "use_gradient_checkpointing", False)
                or getattr(mot, "mot_checkpoint_mixed_attn", False)
                or getattr(mot, "compile_training_layers", False)
                or getattr(model, "compile_training_denoise", False)):
            raise ValueError("cache_text_crossattn_kv does not support training checkpointing/compilation")
        self.num_layers = len(action.blocks)
        if self.num_layers <= 0 or mot.num_layers != self.num_layers:
            raise ValueError("cache_text_crossattn_kv requires matching nonempty ActionDiT/MoT layers")

        _check_source(model._denoise_action_with_video_cache, "FastWAM._denoise_action_with_video_cache")
        prepare = _check_source(action.prepare, "ActionDiT.prepare")
        forward = _check_source(mot.forward_action_with_video_cache_tensor,
                                "MoT.forward_action_with_video_cache_tensor")
        post = _check_source(mot._apply_expert_post_block_tensor, "MoT._apply_expert_post_block_tensor")
        self._sinusoidal_embedding = prepare.__globals__["sinusoidal_embedding_1d"]
        self._attention = forward.__globals__["flash_attention"]
        self._modulate = post.__globals__["modulate"]

        embedding = action.text_embedding
        if (type(embedding) is not nn.Sequential or len(embedding) != 3
                or type(embedding[0]) is not nn.Linear
                or type(embedding[1]) is not nn.GELU
                or embedding[1].approximate != "tanh"
                or type(embedding[2]) is not nn.Linear):
            raise ValueError("cache_text_crossattn_kv requires the pinned Linear/GELU(tanh)/Linear text embedding")
        checked = set()
        self._cross_attention = None
        for block in action.blocks:
            cross = block.cross_attn
            if (type(cross).__module__ != "fastwam.models.wan22.wan_video_dit"
                    or type(cross).__name__ != "CrossAttention"
                    or type(cross).__call__ is not nn.Module.__call__
                    or cross.hidden_dim != action.hidden_dim
                    or cross.num_heads != action.num_heads
                    or cross.attn_head_dim != action.attn_head_dim
                    or any(type(getattr(cross, name)) is not nn.Linear for name in ("q", "k", "v", "o"))):
                raise ValueError("cache_text_crossattn_kv requires ordinary pinned CrossAttention projections")
            fn = getattr(cross.forward, "__func__", cross.forward)
            if fn not in checked:
                fn = _check_source(cross.forward, "CrossAttention.forward")
                checked.add(fn)
            attention = fn.__globals__["flash_attention"]
            if self._cross_attention is not None and attention is not self._cross_attention:
                raise ValueError("cache_text_crossattn_kv requires one pinned cross-attention implementation")
            self._cross_attention = attention
            for norm in (cross.norm_q, cross.norm_k):
                if (type(norm).__module__ != "fastwam.models.wan22.wan_video_dit"
                        or type(norm).__name__ != "RMSNorm"
                        or type(norm).__call__ is not nn.Module.__call__):
                    raise ValueError("cache_text_crossattn_kv requires pinned cross-attention RMSNorm")
                for method in ("forward", "norm"):
                    fn = getattr(norm, method)
                    original = getattr(fn, "__func__", fn)
                    if original not in checked:
                        _check_source(fn, f"RMSNorm.{method}")
                        checked.add(original)
        # Hooks on cached or bypassed modules could change values on every NFE.
        # Do not silently change their frequency or skip cross-attention hooks.
        for name, module in action.named_modules():
            if getattr(module, "_compiled_call_impl", None) is not None:
                raise ValueError(f"cache_text_crossattn_kv does not support precompiled action_expert.{name}")
            if name == "text_embedding" or name.startswith("text_embedding.") or ".cross_attn" in name:
                if type(module) in (nn.Linear, nn.Sequential, nn.GELU):
                    if getattr(module.forward, "__func__", None) is not type(module).forward:
                        raise ValueError(f"cache_text_crossattn_kv requires ordinary forwards on action_expert.{name}")
                if any((module._forward_hooks, module._forward_pre_hooks, module._backward_hooks)):
                    raise ValueError(f"cache_text_crossattn_kv does not support hooks on action_expert.{name}")

    def check_runtime(self):
        """Host-only admission checks, outside every captured/compiled body."""
        if (torch.is_grad_enabled() or self.model.training
                or self.model.action_expert.training or self.model.mot.training):
            raise RuntimeError("cache_text_crossattn_kv requires eval() and torch.inference_mode()")
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError("cache_text_crossattn_kv requires the fixed model dtype without autocast")

    def prepare(self, context):
        """Own [B,L,H*Dh] normalized keys and values for exactly one condition."""
        self.check_runtime()
        action = self.model.action_expert
        embedded = action.text_embedding(context)
        keys, values = [], []
        for block in action.blocks:
            cross = block.cross_attn
            keys.append(cross.norm_k(cross.k(embedded)))
            values.append(cross.v(embedded))
        return keys, values, {
            "prefills": 1, "text_embedding_calls": 1,
            "key_projections": len(keys), "key_normalizations": len(keys),
            "value_projections": len(values),
        }

    def denoise(self, *, latents_action, timestep_action, context, context_mask,
                video_cache_k, video_cache_v, action_attention_mask, text_cache_k, text_cache_v):
        # context remains an explicit geometry input for the shared raw-field
        # interface; its values have already been projected into text_cache_*.
        # All request data used by this body is passed as tensors, never read
        # from mutable Python attributes populated during a previous request.
        if len(text_cache_k) != self.num_layers or len(text_cache_v) != self.num_layers:
            raise ValueError("FastWAM text K/V cache must contain every action layer")
        if len(video_cache_k) != self.num_layers or len(video_cache_v) != self.num_layers:
            raise ValueError("FastWAM video K/V cache must contain every action layer")
        action, mot = self.model.action_expert, self.model.mot
        t = action.time_embedding(self._sinusoidal_embedding(action.freq_dim, timestep_action))
        t_mod = action.time_projection(t).unflatten(1, (6, action.hidden_dim))
        x = action.action_encoder(latents_action)
        freqs = action.get_freqs(latents_action.shape[1])
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, latents_action.shape[1], -1).unsqueeze(1)
        for layer_idx in range(self.num_layers):
            block = action.blocks[layer_idx]
            (q, k, v, residual, gate_msa, shift_mlp, scale_mlp,
             gate_mlp, _checkpointing) = mot._build_expert_attention_io(
                expert=action, block=block, x=x, freqs=freqs, t_mod=t_mod,
            )
            mixed = self._attention(
                q=q, k=torch.cat([video_cache_k[layer_idx], k], dim=1),
                v=torch.cat([video_cache_v[layer_idx], v], dim=1),
                num_heads=mot.num_heads, ctx_mask=action_attention_mask.to(device=q.device),
            )
            x = block.gate(residual, gate_msa, block.self_attn.o(mixed))
            cross = block.cross_attn
            query = cross.norm_q(cross.q(block.norm3(x)))
            cross_output = self._cross_attention(
                q=query, k=text_cache_k[layer_idx], v=text_cache_v[layer_idx],
                num_heads=cross.num_heads, ctx_mask=context_attn_mask,
            )
            x = x + cross.o(cross_output)
            mlp_input = self._modulate(block.norm2(x), shift_mlp, scale_mlp)
            x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return action.post(x)
