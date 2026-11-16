"""DM05 attention sites: native BF16 FA2 where the mask contract permits it.

Unlike Pi0.5, DM05 has four KV heads and 1024 vision patches, so its slots
must not reuse the Pi0.5-shaped RtxFlashAttnBackend. The binding call follows
that backend's audited stride/HD96 contract and implements the same protocol.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec


class Dm05AttentionSm80(AttentionBackendBase):
    def __init__(self, prefix_length, *, native):
        spec = AttentionSpec()
        spec.add_site('vision', num_layers=27, num_q_heads=16, num_kv_heads=16,
                      head_dim=72, max_q_seq=1024, batch_axis=3)
        spec.add_site('prefix', num_layers=34, num_q_heads=8, num_kv_heads=4,
                      head_dim=256, max_q_seq=prefix_length)
        spec.add_site('suffix', num_layers=34, num_q_heads=8, num_kv_heads=4,
                      head_dim=256, max_q_seq=50, max_kv_seq=prefix_length + 50)
        super().__init__(spec)
        self.native = native
        self.prefix_mask = None
        self.slots = {}
        self.native_enqueues = 0
        if native:
            from flash_rt import flash_rt_fa2
            self.fa2 = flash_rt_fa2
            self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        for site, batch, sq, sk, hq, hk, hd in (
                ('vision', 3, 1024, 1024, 16, 16, 96 if native else 72),
                ('prefix', 1, prefix_length, prefix_length, 8, 4, 256),
                ('suffix', 1, 50, prefix_length + 50, 8, 4, 256)):
            make = lambda shape: torch.zeros(shape, dtype=torch.bfloat16, device='cuda')
            self.slots[site] = {
                'Q': make((batch, sq, hq, hd)), 'K': make((batch, sk, hk, hd)),
                'V': make((batch, sk, hk, hd)), 'O': make((batch, sq, hq, hd)),
                'lse': torch.empty(batch, hq, ((sq + 127) // 128) * 128,
                                   dtype=torch.float32, device='cuda'),
            }

    def get_slot_ptrs(self, site, layer_idx=0):
        return {key: value.data_ptr() for key, value in self.slots[site].items()}

    def run(self, site, layer_idx, q_seq, *, kv_seq=None, stream=0):
        slot = self.slots[site]
        q, k, v, out = (slot[name] for name in ('Q', 'K', 'V', 'O'))
        sk = q_seq if kv_seq is None else kv_seq
        if self.native and site != 'prefix':
            self.fa2.fwd_bf16(
                Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(), O=out.data_ptr(),
                softmax_lse=slot['lse'].data_ptr(), softmax_lse_accum=0, o_accum=0,
                batch=q.shape[0], seqlen_q=q_seq, seqlen_k=sk,
                num_heads_q=q.shape[2], num_heads_kv=k.shape[2], head_dim=q.shape[3],
                q_strides=(q.stride(0), q.stride(1), q.stride(2)),
                k_strides=(k.stride(0), k.stride(1), k.stride(2)),
                v_strides=(v.stride(0), v.stride(1), v.stride(2)),
                o_strides=(out.stride(0), out.stride(1), out.stride(2)),
                softmax_scale=72 ** -0.5 if site == 'vision' else 0.0625,
                num_sms=self.sms, stream=stream)
            self.native_enqueues += 1
        else:
            key, value = k[:, :sk].transpose(1, 2), v[:, :sk].transpose(1, 2)
            if q.shape[2] != k.shape[2]:
                key = key.repeat_interleave(2, dim=1)
                value = value.repeat_interleave(2, dim=1)
            result = F.scaled_dot_product_attention(
                q[:, :q_seq].transpose(1, 2), key, value,
                attn_mask=self.prefix_mask if site == 'prefix' else None,
                scale=72 ** -0.5 if site == 'vision' else 0.0625)
            out[:, :q_seq].copy_(result.transpose(1, 2))
        return out.data_ptr()

    def attend(self, site, layer_idx, query, key, value):
        """Tensor staging seam; the kernel dispatch itself is protocol run()."""
        slot = self.slots[site]
        width = query.shape[-1]
        slot['Q'][..., :width].copy_(query)
        slot['K'][..., :width].copy_(key)
        slot['V'][..., :width].copy_(value)
        self.run(site, layer_idx, query.shape[1], kv_seq=key.shape[1],
                 stream=torch.cuda.current_stream().cuda_stream)
        return slot['O'][..., :width].contiguous()
