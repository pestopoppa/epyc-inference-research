// Compare the installed rocBLAS and hipBLASLt FP16 GEMM baselines on gfx90a.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void hip_ok(hipError_t status, const char * what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

void rocblas_ok(rocblas_status status, const char * what) {
    if (status != rocblas_status_success) {
        throw std::runtime_error(std::string(what) + ": status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void hipblaslt_ok(hipblasStatus_t status, const char * what) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

struct Shape {
    int64_t m;
    int64_t n;
    int64_t k;
};

struct TimedResult {
    double milliseconds;
    double tflops;
    float sample;
};

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? 0.5 * (values[middle - 1] + values[middle])
        : values[middle];
}

template <typename Launch>
TimedResult time_gemm(const Shape & shape, int warmup, int repetitions,
                      hipStream_t stream, hipEvent_t start, hipEvent_t stop,
                      Launch launch, const void * output) {
    for (int index = 0; index < warmup; ++index) {
        launch();
    }
    hip_ok(hipStreamSynchronize(stream), "warmup synchronize");

    std::vector<float> samples;
    samples.reserve(repetitions);
    for (int index = 0; index < repetitions; ++index) {
        hip_ok(hipEventRecord(start, stream), "record start");
        launch();
        hip_ok(hipEventRecord(stop, stream), "record stop");
        hip_ok(hipEventSynchronize(stop), "synchronize stop");
        float elapsed = 0.0f;
        hip_ok(hipEventElapsedTime(&elapsed, start, stop), "elapsed time");
        samples.push_back(elapsed);
    }
    const double milliseconds = median(samples);
    const double operations = 2.0 * static_cast<double>(shape.m) * shape.n * shape.k;
    __half first {};
    hip_ok(hipMemcpy(&first, output, sizeof(first), hipMemcpyDeviceToHost), "copy sample");
    return { milliseconds, operations / (milliseconds * 1.0e9), __half2float(first) };
}

void emit(const char * library, const Shape & shape, const TimedResult & result,
          int warmup, int repetitions, int algorithm_index) {
    std::printf(
        "{\"schema\":\"epyc.rocm.gemm_baseline.v1\",\"library\":\"%s\","
        "\"dtype\":\"fp16_compute_fp32\",\"m\":%lld,\"n\":%lld,\"k\":%lld,"
        "\"warmup\":%d,\"repetitions\":%d,\"algorithm_index\":%d,"
        "\"median_ms\":%.9f,\"tflops\":%.9f,\"sample\":%.9g}\n",
        library, static_cast<long long>(shape.m), static_cast<long long>(shape.n),
        static_cast<long long>(shape.k), warmup, repetitions, algorithm_index,
        result.milliseconds, result.tflops, result.sample);
}

void compare_shape(const Shape & shape, int warmup, int repetitions,
                   rocblas_handle rocblas, hipblasLtHandle_t hipblaslt,
                   hipStream_t stream, hipEvent_t start, hipEvent_t stop) {
    const size_t a_elements = static_cast<size_t>(shape.m * shape.k);
    const size_t b_elements = static_cast<size_t>(shape.k * shape.n);
    const size_t c_elements = static_cast<size_t>(shape.m * shape.n);
    __half * a = nullptr;
    __half * b = nullptr;
    __half * rocblas_c = nullptr;
    __half * hipblaslt_c = nullptr;
    hip_ok(hipMalloc(&a, a_elements * sizeof(__half)), "allocate A");
    hip_ok(hipMalloc(&b, b_elements * sizeof(__half)), "allocate B");
    hip_ok(hipMalloc(&rocblas_c, c_elements * sizeof(__half)), "allocate rocBLAS C");
    hip_ok(hipMalloc(&hipblaslt_c, c_elements * sizeof(__half)), "allocate hipBLASLt C");

    std::vector<__half> host_a(a_elements, __float2half(0.03125f));
    std::vector<__half> host_b(b_elements, __float2half(0.0625f));
    hip_ok(hipMemcpy(a, host_a.data(), host_a.size() * sizeof(__half), hipMemcpyHostToDevice),
           "copy A");
    hip_ok(hipMemcpy(b, host_b.data(), host_b.size() * sizeof(__half), hipMemcpyHostToDevice),
           "copy B");
    hip_ok(hipMemset(rocblas_c, 0, c_elements * sizeof(__half)), "clear rocBLAS C");
    hip_ok(hipMemset(hipblaslt_c, 0, c_elements * sizeof(__half)), "clear hipBLASLt C");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto rocblas_launch = [&]() {
        rocblas_ok(rocblas_gemm_ex(
            rocblas, rocblas_operation_none, rocblas_operation_none,
            shape.m, shape.n, shape.k, &alpha,
            a, rocblas_datatype_f16_r, shape.m,
            b, rocblas_datatype_f16_r, shape.k, &beta,
            rocblas_c, rocblas_datatype_f16_r, shape.m,
            rocblas_c, rocblas_datatype_f16_r, shape.m,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
            "rocblas_gemm_ex");
    };

    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t a_layout = nullptr;
    hipblasLtMatrixLayout_t b_layout = nullptr;
    hipblasLtMatrixLayout_t c_layout = nullptr;
    hipblasLtMatmulPreference_t preference = nullptr;
    hipblaslt_ok(hipblasLtMatmulDescCreate(
        &operation, HIPBLAS_COMPUTE_32F, HIP_R_32F), "create matmul descriptor");
    hipblaslt_ok(hipblasLtMatrixLayoutCreate(
        &a_layout, HIP_R_16F, shape.m, shape.k, shape.m), "create A layout");
    hipblaslt_ok(hipblasLtMatrixLayoutCreate(
        &b_layout, HIP_R_16F, shape.k, shape.n, shape.k), "create B layout");
    hipblaslt_ok(hipblasLtMatrixLayoutCreate(
        &c_layout, HIP_R_16F, shape.m, shape.n, shape.m), "create C layout");
    hipblaslt_ok(hipblasLtMatmulPreferenceCreate(&preference), "create preference");
    constexpr size_t workspace_bytes = 64U * 1024U * 1024U;
    hipblaslt_ok(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)), "set workspace preference");
    std::array<hipblasLtMatmulHeuristicResult_t, 32> heuristics {};
    int algorithm_count = 0;
    hipblaslt_ok(hipblasLtMatmulAlgoGetHeuristic(
        hipblaslt, operation, a_layout, b_layout, c_layout, c_layout,
        preference, heuristics.size(), heuristics.data(), &algorithm_count), "query heuristics");
    if (algorithm_count < 1) {
        throw std::runtime_error("hipBLASLt returned no usable heuristic");
    }
    void * workspace = nullptr;
    hip_ok(hipMalloc(&workspace, workspace_bytes), "allocate workspace");
    int selected_algorithm = -1;
    double selected_milliseconds = INFINITY;
    for (int index = 0; index < algorithm_count; ++index) {
        if (heuristics[index].state != HIPBLAS_STATUS_SUCCESS ||
                heuristics[index].workspaceSize > workspace_bytes) {
            continue;
        }
        auto probe = [&]() {
            hipblaslt_ok(hipblasLtMatmul(
                hipblaslt, operation, &alpha, a, a_layout, b, b_layout, &beta,
                hipblaslt_c, c_layout, hipblaslt_c, c_layout,
                &heuristics[index].algo, workspace, workspace_bytes, stream),
                "hipblasLtMatmul heuristic probe");
        };
        const TimedResult probe_result = time_gemm(
            shape, 1, 3, stream, start, stop, probe, hipblaslt_c);
        if (probe_result.milliseconds < selected_milliseconds) {
            selected_milliseconds = probe_result.milliseconds;
            selected_algorithm = index;
        }
    }
    if (selected_algorithm < 0) {
        throw std::runtime_error("hipBLASLt returned no runnable heuristic");
    }
    auto hipblaslt_launch = [&]() {
        hipblaslt_ok(hipblasLtMatmul(
            hipblaslt, operation, &alpha, a, a_layout, b, b_layout, &beta,
            hipblaslt_c, c_layout, hipblaslt_c, c_layout,
            &heuristics[selected_algorithm].algo,
            workspace, workspace_bytes, stream), "hipblasLtMatmul");
    };

    const TimedResult rocblas_result = time_gemm(
        shape, warmup, repetitions, stream, start, stop, rocblas_launch, rocblas_c);
    const TimedResult hipblaslt_result = time_gemm(
        shape, warmup, repetitions, stream, start, stop, hipblaslt_launch, hipblaslt_c);
    if (!std::isfinite(rocblas_result.sample) || !std::isfinite(hipblaslt_result.sample) ||
            std::abs(rocblas_result.sample - hipblaslt_result.sample) > 0.01f) {
        throw std::runtime_error("rocBLAS and hipBLASLt output samples disagree");
    }
    emit("rocblas", shape, rocblas_result, warmup, repetitions, -1);
    emit("hipblaslt", shape, hipblaslt_result, warmup, repetitions, selected_algorithm);

    hip_ok(hipFree(workspace), "free workspace");
    hipblaslt_ok(hipblasLtMatmulPreferenceDestroy(preference), "destroy preference");
    hipblaslt_ok(hipblasLtMatrixLayoutDestroy(c_layout), "destroy C layout");
    hipblaslt_ok(hipblasLtMatrixLayoutDestroy(b_layout), "destroy B layout");
    hipblaslt_ok(hipblasLtMatrixLayoutDestroy(a_layout), "destroy A layout");
    hipblaslt_ok(hipblasLtMatmulDescDestroy(operation), "destroy matmul descriptor");
    hip_ok(hipFree(hipblaslt_c), "free hipBLASLt C");
    hip_ok(hipFree(rocblas_c), "free rocBLAS C");
    hip_ok(hipFree(b), "free B");
    hip_ok(hipFree(a), "free A");
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const int repetitions = argc > 1 ? std::stoi(argv[1]) : 30;
        const int warmup = argc > 2 ? std::stoi(argv[2]) : 10;
        if (repetitions < 3 || warmup < 1) {
            throw std::runtime_error("usage: rocm_gemm_baseline_compare [repetitions>=3] [warmup>=1]");
        }
        hip_ok(hipSetDevice(0), "select device 0");
        hipDeviceProp_t properties {};
        hip_ok(hipGetDeviceProperties(&properties, 0), "read device properties");
        if (std::string(properties.gcnArchName).find("gfx90a") == std::string::npos) {
            throw std::runtime_error("this evidence runner requires gfx90a");
        }
        hipStream_t stream = nullptr;
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        rocblas_handle rocblas = nullptr;
        hipblasLtHandle_t hipblaslt = nullptr;
        hip_ok(hipStreamCreate(&stream), "create stream");
        hip_ok(hipEventCreate(&start), "create start event");
        hip_ok(hipEventCreate(&stop), "create stop event");
        rocblas_ok(rocblas_create_handle(&rocblas), "create rocBLAS handle");
        rocblas_ok(rocblas_set_stream(rocblas, stream), "set rocBLAS stream");
        hipblaslt_ok(hipblasLtCreate(&hipblaslt), "create hipBLASLt handle");

        const std::vector<Shape> shapes = {
            { 4864, 128, 896 }, { 4864, 512, 896 }, { 4864, 2048, 896 },
            { 896, 128, 4864 }, { 896, 512, 4864 }, { 896, 2048, 4864 },
            { 896, 128, 896 },  { 896, 512, 896 },  { 896, 2048, 896 },
        };
        std::printf("{\"schema\":\"epyc.rocm.gemm_baseline.meta.v1\","
                    "\"device\":\"%s\",\"arch\":\"%s\",\"shape_count\":%zu}\n",
                    properties.name, properties.gcnArchName, shapes.size());
        for (const Shape & shape : shapes) {
            compare_shape(shape, warmup, repetitions, rocblas, hipblaslt,
                          stream, start, stop);
        }

        hipblaslt_ok(hipblasLtDestroy(hipblaslt), "destroy hipBLASLt handle");
        rocblas_ok(rocblas_destroy_handle(rocblas), "destroy rocBLAS handle");
        hip_ok(hipEventDestroy(stop), "destroy stop event");
        hip_ok(hipEventDestroy(start), "destroy start event");
        hip_ok(hipStreamDestroy(stream), "destroy stream");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "rocm_gemm_baseline_compare: %s\n", error.what());
        return 1;
    }
}
