"""Hy SM80 attention preparation with the active flow frontend's cast rules.

Only QKV split, rotate-half RoPE, Q/K RMSNorm and cache writes are fused.
Projection, attention, FFN and sampling still use the native BF16 pipeline.
The two entry points retain their profile's dtype/rounding boundaries; the
parallel FP32 reduction need not be bit-identical to a particular torch build.
"""

from __future__ import annotations

from importlib import import_module
import math
from numbers import Integral

import torch

from flash_rt.models.hyvla.pipeline_thor import HyVLAThorBF16Pipeline


def load_attention_prep(numerical_profile):
    """Resolve the explicitly requested module; never fall back silently."""
    if numerical_profile not in ("semantic_bf16", "official_local_fp32"):
        raise ValueError(f"Unknown Hy SM80 numerical_profile {numerical_profile!r}")
    name = (
        "hyvla_rope_qknorm_kvwrite_official_fp32"
        if numerical_profile == "official_local_fp32"
        else "hyvla_rope_qknorm_kvwrite_semantic_bf16"
    )
    try:
        module = import_module("flash_rt.flash_rt_hyvla_sm80")
        return getattr(module, name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "HyVLA fused_rope_qknorm_kvwrite=True requires flash_rt_hyvla_sm80, built with "
            "-DGPU_ARCH=80 -DFLASHRT_ENABLE_HYVLA=ON "
            "(cmake --build <build-dir> --target flash_rt_hyvla_sm80)"
        ) from exc


class HyVLASM80BF16Pipeline(HyVLAThorBF16Pipeline):
    """BF16 model pipeline with one native launch per QKV preparation."""

    def __init__(self, W, *, attention_prep=None):
        super().__init__(W)
        geometry = (W.n_heads, W.n_kv, W.head_dim)
        if geometry != (self.n_heads, self.n_kv, self.head_dim):
            raise ValueError(
                "Hy SM80 fused attention preparation requires checkpoint "
                f"(n_heads, n_kv, head_dim)=(16, 4, 128), got {geometry}"
            )
        self._local_dtype = (
            torch.float32
            if self.numerical_profile == "official_local_fp32"
            else torch.bfloat16
        )
        self._attention_prep = (
            attention_prep if attention_prep is not None
            else load_attention_prep(self.numerical_profile)
        )

    def _prepare_qkv(self, qkv, cos, sin, qk_w, kbuf, vbuf, off):
        """Write only [off:off+S]; cache heads may be compact or pre-expanded.

        Layer slices are (1,Hcache,S_tot,128), with Hcache=4 or 16.
        Requiring their actual contiguous strides avoids writing a packed
        window into a sliced/padded cache by accident. Checks use metadata
        only, so this seam is safe during CUDA graph capture.
        """
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv
        if qkv.ndim != 2 or qkv.shape[1] != (nh + 2 * nkv) * hd:
            raise ValueError(
                f"Hy SM80 fused QKV requires (S,{(nh + 2 * nkv) * hd}), "
                f"got {tuple(qkv.shape)}"
            )
        S = qkv.shape[0]
        if (kbuf.ndim != 4 or kbuf.shape[0] != 1
                or kbuf.shape[1] not in (nkv, nh) or kbuf.shape[-1] != hd
                or tuple(vbuf.shape) != tuple(kbuf.shape)):
            raise ValueError(
                "Hy SM80 fused KV requires matching (1,4|16,S_tot,128) "
                f"layer slices, got K={tuple(kbuf.shape)}, V={tuple(vbuf.shape)}"
            )
        S_tot = kbuf.shape[2]
        if (isinstance(off, bool) or not isinstance(off, Integral)
                or S < 1 or off < 0 or off + S > S_tot):
            raise ValueError(
                f"Hy SM80 fused KV has invalid window off={off}, S={S}, S_tot={S_tot}"
            )
        if not math.isfinite(self.rms_eps) or self.rms_eps <= 0:
            raise ValueError("Hy SM80 fused QK-norm requires finite eps>0")
        table_shapes = ((S, hd), (1, S, hd), (1, 1, S, hd))
        if tuple(cos.shape) not in table_shapes or tuple(sin.shape) not in table_shapes:
            raise ValueError(
                f"Hy SM80 fused RoPE requires shared per-position (S,128) "
                f"tables, got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
            )
        if len(qk_w) != 2 or any(tuple(w.shape) != (hd,) for w in qk_w):
            raise ValueError("Hy SM80 fused QK-norm requires two (128,) weights")
        inputs = (
            ("qkv", qkv, torch.bfloat16),
            ("cos", cos, self._local_dtype),
            ("sin", sin, self._local_dtype),
            ("qn_w", qk_w[0], torch.bfloat16),
            ("kn_w", qk_w[1], torch.bfloat16),
            ("kbuf", kbuf, self._local_dtype),
            ("vbuf", vbuf, torch.bfloat16),
        )
        for name, tensor, dtype in inputs:
            if (not tensor.is_cuda or tensor.device != qkv.device
                    or tensor.dtype != dtype or not tensor.is_contiguous()):
                raise ValueError(
                    f"Hy SM80 fused {name} requires contiguous {dtype} on "
                    f"{qkv.device}; got shape={tuple(tensor.shape)}, "
                    f"strides={tensor.stride()}, dtype={tensor.dtype}, "
                    f"device={tensor.device}"
                )
        # This allocation belongs to each graph's private pool, exactly like
        # the torch Q result it replaces. Never cache scratch across the
        # independent per-time velocity graphs.
        q = torch.empty(1, nh, S, hd, dtype=self._local_dtype, device=qkv.device)
        with torch.cuda.device(qkv.device):
            self._attention_prep(
                qkv.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                qk_w[0].data_ptr(), qk_w[1].data_ptr(),
                q.data_ptr(), kbuf.data_ptr(), vbuf.data_ptr(),
                S, nh, nkv, hd, S_tot, int(off), self.rms_eps,
                kbuf.shape[1] // nkv,
                torch.cuda.current_stream(qkv.device).cuda_stream,
            )
        return q


__all__ = ["HyVLASM80BF16Pipeline"]
