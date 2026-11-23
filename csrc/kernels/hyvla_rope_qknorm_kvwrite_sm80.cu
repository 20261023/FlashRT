// Hy's active flow semantics: rotate_half RoPE FIRST, then head-wise RMSNorm,
// then a windowed KV write. The legacy Thor fusion drops BF16 RoPE rounding;
// this additive SM80 path preserves it and also supports FP32 local Q/K.
// Grid (S,nq+2*nkv) parallelizes heads for the small denoise suffix.

#include "hyvla_rope_qknorm_kvwrite_sm80.cuh"
#include "common.cuh"

#include <cstdint>

namespace {

constexpr int kHeadDim = 128;

template <typename Local>
__device__ __forceinline__ float local_round(float value);

template <>
__device__ __forceinline__ float local_round<__nv_bfloat16>(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

template <>
__device__ __forceinline__ float local_round<float>(float value) {
    return value;
}

template <typename Local>
__device__ __forceinline__ Local local_output(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 local_output<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ float local_output<float>(float value) {
    return value;
}

template <typename Local>
__global__ void rope_qknorm_kvwrite(
    const __nv_bfloat16* __restrict__ qkv,
    const Local* __restrict__ cos,
    const Local* __restrict__ sin,
    const __nv_bfloat16* __restrict__ qn_w,
    const __nv_bfloat16* __restrict__ kn_w,
    Local* __restrict__ q_out,
    Local* __restrict__ kbuf,
    __nv_bfloat16* __restrict__ vbuf,
    int S, int nq, int nkv, int S_tot, int off, float eps, int kv_rep)
{
    const int s = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;  // one 128-thread block per position/head
    const int64_t qkv_stride = (int64_t(nq) + 2 * int64_t(nkv)) * kHeadDim;
    const int64_t input_index = int64_t(s) * qkv_stride + int64_t(h) * kHeadDim + d;

    if (h >= nq + nkv) {
        // Values never receive RoPE or norm, and retain their exact BF16 bits.
        const int kh = h - nq - nkv;
        const __nv_bfloat16 value = qkv[input_index];
        for (int r = 0; r < kv_rep; ++r) {
            const int64_t dst = (
                int64_t(kh * kv_rep + r) * S_tot + off + s) * kHeadDim + d;
            vbuf[dst] = value;
        }
        return;
    }

    const int partner = d < kHeadDim / 2 ? d + kHeadDim / 2 : d - kHeadDim / 2;
    const float x = __bfloat162float(qkv[input_index]);
    const float xp = __bfloat162float(qkv[input_index - d + partner]);
    const float cs = static_cast<float>(cos[int64_t(s) * kHeadDim + d]);
    const float sn = static_cast<float>(sin[int64_t(s) * kHeadDim + d]);
    // Each eager torch multiply materializes its own dtype before the add.
    // Explicit intrinsics prevent replacing the two products/add with FMA,
    // including in the FP32 profile. Both halves read their own table values.
    const float a = local_round<Local>(__fmul_rn(x, cs));
    const float b = local_round<Local>(__fmul_rn(d < kHeadDim / 2 ? -xp : xp, sn));
    const float rotated = local_round<Local>(__fadd_rn(a, b));

    __shared__ float partials[4];
    const float sum = block_reduce_sum(__fmul_rn(rotated, rotated), partials);
    const float inverse_rms = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.0f / kHeadDim), eps));
    const float weight = __bfloat162float(h < nq ? qn_w[d] : kn_w[d]);
    // semantic_bf16 F.rms_norm rounds after weight multiplication; the
    // official FP32 QK norm has no narrowing cast before its BF16 weight.
    const Local normalized = local_output<Local>(
        __fmul_rn(__fmul_rn(rotated, inverse_rms), weight));
    if (h < nq) {
        q_out[(int64_t(h) * S + s) * kHeadDim + d] = normalized;
    } else {
        const int kh = h - nq;
        for (int r = 0; r < kv_rep; ++r) {
            const int64_t dst = (
                int64_t(kh * kv_rep + r) * S_tot + off + s) * kHeadDim + d;
            kbuf[dst] = normalized;
        }
    }
}

template <typename Local>
int launch(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream)
{
    rope_qknorm_kvwrite<Local><<<dim3(S, nq + 2 * nkv), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkv),
        static_cast<const Local*>(cos), static_cast<const Local*>(sin),
        static_cast<const __nv_bfloat16*>(qn_w),
        static_cast<const __nv_bfloat16*>(kn_w),
        static_cast<Local*>(q_out), static_cast<Local*>(kbuf),
        static_cast<__nv_bfloat16*>(vbuf),
        S, nq, nkv, S_tot, off, eps, kv_rep);
    // A launch error is surfaced without synchronizing the caller's stream.
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int hyvla_rope_qknorm_kvwrite_semantic_bf16(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream)
{
    if (hd != kHeadDim) return static_cast<int>(cudaErrorInvalidValue);
    return launch<__nv_bfloat16>(
        qkv, cos, sin, qn_w, kn_w, q_out, kbuf, vbuf,
        S, nq, nkv, S_tot, off, eps, kv_rep, stream);
}

extern "C" int hyvla_rope_qknorm_kvwrite_official_fp32(
    const void* qkv, const void* cos, const void* sin,
    const void* qn_w, const void* kn_w,
    void* q_out, void* kbuf, void* vbuf,
    int S, int nq, int nkv, int hd, int S_tot, int off, float eps,
    int kv_rep, cudaStream_t stream)
{
    if (hd != kHeadDim) return static_cast<int>(cudaErrorInvalidValue);
    return launch<float>(
        qkv, cos, sin, qn_w, kn_w, q_out, kbuf, vbuf,
        S, nq, nkv, S_tot, off, eps, kv_rep, stream);
}
