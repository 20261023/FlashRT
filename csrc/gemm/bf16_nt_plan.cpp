// Standalone HOST C++ pybind module. No Torch headers, custom CUDA kernel,
// transpose/copy, quantization, or changes to the legacy GemmRunner binary.
// Math/layout mirror gemm_runner.cu::bf16_nt_bias (898-956):
//   D[M,N] = A[M,K] @ B[N,K]^T + bias[N], BF16 storage, FP32 accumulation.
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace py = pybind11;

namespace {
static_assert(sizeof(std::uintptr_t) == 8, "This pointer API requires a 64-bit Python process");
static_assert(sizeof(cudaStream_t) == sizeof(std::uintptr_t), "CUDA streams must retain all pointer bits");

void check_blas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("CachedGemmRunner ") + operation +
                                 ": cuBLAS status=" + std::to_string(static_cast<int>(status)));
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CachedGemmRunner ") + operation +
                                 ": " + cudaGetErrorString(status));
    }
}

struct Plan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weight = nullptr;
    cublasLtMatrixLayout_t activation = nullptr;
    cublasLtMatrixLayout_t output = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    std::array<cublasLtMatmulHeuristicResult_t, 8> heuristics{};
    int count = 0;
    int selected = -1;

    Plan() = default;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    ~Plan() noexcept {
        // Destruction is best effort, including interpreter/CUDA teardown.
        if (preference) (void)cublasLtMatmulPreferenceDestroy(preference);
        if (output) (void)cublasLtMatrixLayoutDestroy(output);
        if (activation) (void)cublasLtMatrixLayoutDestroy(activation);
        if (weight) (void)cublasLtMatrixLayoutDestroy(weight);
        if (operation) (void)cublasLtMatmulDescDestroy(operation);
    }

    void configure(int m, int n, int k, void* bias, std::size_t workspace_bytes) {
        check_blas(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc create");
        const cublasOperation_t transposed = CUBLAS_OP_T, normal = CUBLAS_OP_N;
        check_blas(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &transposed, sizeof(transposed)), "set transA");
        check_blas(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                 &normal, sizeof(normal)), "set transB");
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        check_blas(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                 &epilogue, sizeof(epilogue)), "set epilogue");
        const cudaDataType_t bias_type = CUDA_R_16BF;
        check_blas(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                 &bias_type, sizeof(bias_type)), "set bias dtype");
        set_bias(bias);
        // Column-major views of the unchanged row-major buffers:
        // D^T[N,M] = B[N,K] @ A^T[K,M]; weight buffer itself is [N,K].
        check_blas(cublasLtMatrixLayoutCreate(&weight, CUDA_R_16BF, k, n, k), "weight layout");
        check_blas(cublasLtMatrixLayoutCreate(&activation, CUDA_R_16BF, k, m, k), "activation layout");
        check_blas(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, n, m, n), "output layout");
        check_blas(cublasLtMatmulPreferenceCreate(&preference), "preference create");
        check_blas(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "set workspace");
        // Same restriction as legacy: never round split-K reductions to BF16
        // before adding the bias. Only compute-type (FP32) reduction is legal.
        const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        check_blas(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction, sizeof(reduction)), "set reduction mask");
    }

    void set_bias(void* bias) {
        // This MUST be refreshed even for a cache hit: hundreds of different
        // Linear modules share (M,N,K) but have different bias allocations.
        check_blas(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                 &bias, sizeof(bias)), "set current bias pointer");
    }

    void query(cublasLtHandle_t handle) {
        check_blas(cublasLtMatmulAlgoGetHeuristic(handle, operation, weight, activation,
            output, output, preference, static_cast<int>(heuristics.size()), heuristics.data(), &count),
            "heuristic query");
        if (count <= 0) throw std::runtime_error("CachedGemmRunner bf16_nt_bias: no algorithm found");
    }
};

class CachedGemmRunner {
public:
    CachedGemmRunner() {
        check_cuda(cudaGetDevice(&device_), "get construction device");
        check_blas(cublasLtCreate(&handle_), "handle create");
        try {
            check_cuda(cudaMalloc(&workspace_, workspace_bytes_), "allocate 64 MiB workspace");
        } catch (...) {
            release();
            throw;
        }
    }
    CachedGemmRunner(const CachedGemmRunner&) = delete;
    CachedGemmRunner& operator=(const CachedGemmRunner&) = delete;
    ~CachedGemmRunner() noexcept { release(); }

    void bf16_nt_bias(std::uintptr_t a, std::uintptr_t b, std::uintptr_t d, std::uintptr_t bias,
                      int m, int n, int k, std::uintptr_t stream) {
        if (m <= 0 || n <= 0 || k <= 0) {
            throw std::invalid_argument("CachedGemmRunner bf16_nt_bias: M, N and K must be positive");
        }
        if (!a || !b || !d || !bias) {
            throw std::invalid_argument("CachedGemmRunner bf16_nt_bias: pointers must be non-null");
        }
        // A single (M,N,K) cache must not assume an alignment class that changes
        // across calls. Torch/CUDA base allocations satisfy 256-byte alignment.
        // Offset views are explicitly outside this narrow base-buffer contract;
        // do not fix them with hidden copies or reuse an illegally aligned algo.
        if ((a % 256) || (b % 256) || (d % 256) || (bias % 2)) {
            throw std::invalid_argument("CachedGemmRunner bf16_nt_bias: A/B/D require 256-byte base alignment; bias requires BF16 alignment");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto key = std::make_tuple(m, n, k);
        auto found = plans_.find(key);
        if (found == plans_.end()) {
            // unique_ptr protects partially configured descriptors on ANY error.
            auto fresh = std::make_unique<Plan>();
            fresh->configure(m, n, k, reinterpret_cast<void*>(bias), workspace_bytes_);
            ++heuristic_queries_;
            fresh->query(handle_);
            found = plans_.emplace(key, std::move(fresh)).first;
            ++plan_builds_;
        }
        Plan& plan = *found->second;
        plan.set_bias(reinterpret_cast<void*>(bias));
        const float alpha = 1.0f, beta = 0.0f;
        auto launch = [&](int index) {
            if (plan.heuristics[index].state != CUBLAS_STATUS_SUCCESS ||
                plan.heuristics[index].workspaceSize > workspace_bytes_) return false;
            const cublasStatus_t status = cublasLtMatmul(handle_, plan.operation,
                &alpha, reinterpret_cast<void*>(b), plan.weight,
                reinterpret_cast<void*>(a), plan.activation, &beta,
                reinterpret_cast<void*>(d), plan.output, reinterpret_cast<void*>(d), plan.output,
                &plan.heuristics[index].algo, workspace_, workspace_bytes_,
                reinterpret_cast<cudaStream_t>(stream));  // retain ALL uintptr_t bits
            if (status == CUBLAS_STATUS_SUCCESS) {
                plan.selected = index;
                ++launches_;  // successful API enqueue, not a GPU timing/readback
                return true;
            }
            // Try cached alternatives for synchronous shape/alignment rejection.
            // Do not retry a potentially partial/failed GPU execution.
            if (status != CUBLAS_STATUS_NOT_SUPPORTED && status != CUBLAS_STATUS_INVALID_VALUE &&
                status != CUBLAS_STATUS_ARCH_MISMATCH) check_blas(status, "matmul enqueue");
            return false;
        };
        const int previous = plan.selected;
        if (previous >= 0 && launch(previous)) return;
        for (int index = 0; index < plan.count; ++index) {
            if (index != previous && launch(index)) return;
        }
        throw std::runtime_error("CachedGemmRunner bf16_nt_bias: every cached heuristic rejected M=" +
            std::to_string(m) + " N=" + std::to_string(n) + " K=" + std::to_string(k));
    }

    py::dict stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        py::dict result;
        result["plan_builds"] = plan_builds_;
        result["heuristic_queries"] = heuristic_queries_;
        result["launches"] = launches_;
        return result;  // CPU counters ONLY: no CUDA call/synchronization here
    }

private:
    void release() noexcept {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        if (device_ >= 0 && (!have_previous || previous != device_)) (void)cudaSetDevice(device_);
        plans_.clear();
        if (workspace_) (void)cudaFree(workspace_);
        workspace_ = nullptr;
        if (handle_) (void)cublasLtDestroy(handle_);
        handle_ = nullptr;
        if (have_previous && previous != device_) (void)cudaSetDevice(previous);
    }

    static constexpr std::size_t workspace_bytes_ = 64ULL * 1024 * 1024;
    int device_ = -1;
    cublasLtHandle_t handle_ = nullptr;
    void* workspace_ = nullptr;
    std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans_;
    mutable std::mutex mutex_;
    std::uint64_t plan_builds_ = 0, heuristic_queries_ = 0, launches_ = 0;
};
}  // namespace

PYBIND11_MODULE(flash_rt_gemm_plan, module) {
    module.doc() = "Instance-owned cached BF16 NT+bias cuBLASLt plans; serialized ordered-stream use";
    py::class_<CachedGemmRunner>(module, "CachedGemmRunner")
        .def(py::init<>())
        .def("bf16_nt_bias", &CachedGemmRunner::bf16_nt_bias,
             py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
             py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = std::uintptr_t{0},
             py::call_guard<py::gil_scoped_release>())
        .def("stats", &CachedGemmRunner::stats);
}
