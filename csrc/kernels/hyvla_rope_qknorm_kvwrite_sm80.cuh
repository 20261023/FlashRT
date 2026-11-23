// Hy flow attention-preparation kernels. Inputs are BF16 projection rows
// (S,(nq+2*nkv)*128), BF16 norm weights (128,), and contiguous layer caches
// (nkv*kv_rep,S_tot,128). Q is (nq,S,128). Only [off:off+S] is written.
// All input/output buffers must be disjoint. The stream is caller-owned.
#pragma once

#include <cuda_runtime.h>

// semantic_bf16: tables/Q/K are BF16; each RoPE mul and add rounds to BF16.
// RMSNorm accumulates/multiplies weights in FP32, then rounds the output.
extern "C" int hyvla_rope_qknorm_kvwrite_semantic_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream);

// official_local_fp32: tables/Q/K stay FP32 throughout RoPE and QK-norm.
// Projection rows, norm weights, and the unrotated value cache remain BF16.
extern "C" int hyvla_rope_qknorm_kvwrite_official_fp32(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream);
