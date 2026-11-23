// Dedicated Hy SM80 module. Its source and bindings have one CMake owner;
// enabling this fusion never exposes any legacy Thor/Orin-only entry point.
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/hyvla_rope_qknorm_kvwrite_sm80.cuh"

namespace py = pybind11;

namespace {

using Launcher = int (*)(
    const void*, const void*, const void*, const void*, const void*,
    void*, void*, void*, int, int, int, int, int, int, float, int, cudaStream_t);

void bind_attention_prep(py::module_& m, const char* name, Launcher launcher,
                         uintptr_t local_alignment, const char* doc) {
    m.def(name,
        [launcher, local_alignment, name](
            uintptr_t qkv, uintptr_t cos, uintptr_t sin,
            uintptr_t qn_w, uintptr_t kn_w,
            uintptr_t q_out, uintptr_t kbuf, uintptr_t vbuf,
            int S, int nq, int nkv, int hd, int S_tot, int off,
            float eps, int kv_rep, uintptr_t stream) {
            const std::string op(name);
            if (S <= 0 || nq <= 0 || nkv <= 0 || S_tot <= 0 || hd != 128)
                throw py::value_error(
                    op + " requires S,nq,nkv,S_tot>0 and hd==128");
            if (int64_t(nq) + 2 * int64_t(nkv) > 65535)
                throw py::value_error(op + ": nq+2*nkv exceeds the CUDA grid limit");
            if (off < 0 || S > S_tot || off > S_tot - S)
                throw py::value_error(op + ": invalid KV window off=" +
                    std::to_string(off) + ", S=" + std::to_string(S) +
                    ", S_tot=" + std::to_string(S_tot));
            if (!std::isfinite(eps) || eps <= 0)
                throw py::value_error(op + " requires finite eps>0");
            if (nq % nkv != 0 || (kv_rep != 1 && kv_rep != nq / nkv))
                throw py::value_error(
                    op + " requires nq%nkv==0 and kv_rep==1 or nq/nkv");
            const uintptr_t ptrs[] = {qkv, cos, sin, qn_w, kn_w, q_out, kbuf, vbuf};
            const uintptr_t alignments[] = {
                2, local_alignment, local_alignment, 2, 2,
                local_alignment, local_alignment, 2};
            for (int i = 0; i < 8; ++i) {
                if (ptrs[i] == 0 || ptrs[i] % alignments[i] != 0)
                    throw py::value_error(
                        op + ": pointers must be non-null and dtype-aligned");
            }
            const int status = launcher(
                reinterpret_cast<const void*>(qkv),
                reinterpret_cast<const void*>(cos),
                reinterpret_cast<const void*>(sin),
                reinterpret_cast<const void*>(qn_w),
                reinterpret_cast<const void*>(kn_w),
                reinterpret_cast<void*>(q_out),
                reinterpret_cast<void*>(kbuf),
                reinterpret_cast<void*>(vbuf),
                S, nq, nkv, hd, S_tot, off, eps, kv_rep,
                reinterpret_cast<cudaStream_t>(stream));
            if (status != static_cast<int>(cudaSuccess))
                throw std::runtime_error(op + ": CUDA launch failed: " +
                    cudaGetErrorString(static_cast<cudaError_t>(status)));
        },
        py::arg("qkv"), py::arg("cos"), py::arg("sin"),
        py::arg("qn_w"), py::arg("kn_w"),
        py::arg("q_out"), py::arg("kbuf"), py::arg("vbuf"),
        py::arg("S"), py::arg("nq"), py::arg("nkv"), py::arg("hd"),
        py::arg("S_tot"), py::arg("off"), py::arg("eps") = 1e-5f,
        py::arg("kv_rep") = 1, py::arg("stream") = 0, doc);
}

}  // namespace

PYBIND11_MODULE(flash_rt_hyvla_sm80, m) {
    m.doc() = "Hy SM80 profile-preserving RoPE + QK-norm + KV-write";
    bind_attention_prep(
        m, "hyvla_rope_qknorm_kvwrite_semantic_bf16",
        hyvla_rope_qknorm_kvwrite_semantic_bf16, 2,
        "Contiguous BF16 qkv[S,(nq+2*nkv)*128], cos/sin[S,128], "
        "qn/kn[128], Q[nq,S,128], KV[nkv*kv_rep,S_tot,128]. "
        "Preserves BF16 RoPE product/add rounding; QK-norm in FP32, BF16 out. "
        "Writes only the off:off+S KV window. Buffers must not alias.");
    bind_attention_prep(
        m, "hyvla_rope_qknorm_kvwrite_official_fp32",
        hyvla_rope_qknorm_kvwrite_official_fp32, 4,
        "Same contiguous shapes as semantic_bf16; cos/sin, Q and K are FP32. "
        "qkv, qn/kn and V remain BF16. RoPE precedes FP32 QK-norm. "
        "Writes only the off:off+S KV window. Buffers must not alias.");
}
