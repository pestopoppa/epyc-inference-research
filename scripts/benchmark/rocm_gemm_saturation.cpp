// A minimal, reproducible gfx90a saturation load for RVP-T0-1.
//
// This is deliberately a workload, not a profiler or a clock-control tool.  It
// performs FP32 rocBLAS GEMMs for a fixed wall-clock duration and reports the
// exact device/dimensions/iteration count as JSON.  The trusted Python sampler
// brackets this process and records power, sclk, mclk, and junction temperature
// every 250 ms; this binary never writes a sysfs control node.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

[[noreturn]] void fail(const char * what, const char * detail) {
    std::fprintf(stderr, "%s: %s\n", what, detail);
    std::exit(1);
}

void hip_ok(hipError_t status, const char * what) {
    if (status != hipSuccess) {
        fail(what, hipGetErrorString(status));
    }
}

void rocblas_ok(rocblas_status status, const char * what) {
    if (status != rocblas_status_success) {
        char detail[64];
        std::snprintf(detail, sizeof(detail), "rocblas status %d", static_cast<int>(status));
        fail(what, detail);
    }
}

int64_t positive_integer(const char * text, const char * name) {
    char * end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0) {
        fail(name, "must be a positive integer");
    }
    return static_cast<int64_t>(value);
}

int nonnegative_integer(const char * text, const char * name) {
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0) {
        fail(name, "must be a non-negative integer");
    }
    return static_cast<int>(value);
}

double positive_double(const char * text, const char * name) {
    char * end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value > 0.0)) {
        fail(name, "must be a positive number");
    }
    return value;
}

}  // namespace

int main(int argc, char ** argv) {
    double duration_s = 60.0;
    int64_t m = 8192;
    int64_t n = 8192;
    int64_t k = 8192;
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--duration-s") == 0 && i + 1 < argc) {
            duration_s = positive_double(argv[++i], "--duration-s");
        } else if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            m = positive_integer(argv[++i], "--m");
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = positive_integer(argv[++i], "--n");
        } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = positive_integer(argv[++i], "--k");
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = nonnegative_integer(argv[++i], "--device");
        } else {
            std::fprintf(stderr,
                         "usage: %s [--duration-s N] [--m N] [--n N] [--k N] "
                         "[--device ZERO_BASED]\n", argv[0]);
            return 2;
        }
    }

    hip_ok(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t properties {};
    hip_ok(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");

    const size_t a_bytes = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(float);
    const size_t b_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(float);
    const size_t c_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
    float * a = nullptr;
    float * b = nullptr;
    float * c = nullptr;
    hip_ok(hipMalloc(&a, a_bytes), "hipMalloc(A)");
    hip_ok(hipMalloc(&b, b_bytes), "hipMalloc(B)");
    hip_ok(hipMalloc(&c, c_bytes), "hipMalloc(C)");
    hip_ok(hipMemset(a, 0x01, a_bytes), "hipMemset(A)");
    hip_ok(hipMemset(b, 0x02, b_bytes), "hipMemset(B)");
    hip_ok(hipMemset(c, 0, c_bytes), "hipMemset(C)");

    rocblas_handle handle = nullptr;
    rocblas_ok(rocblas_create_handle(&handle), "rocblas_create_handle");
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto gemm = [&]() {
        rocblas_ok(rocblas_sgemm(
            handle, rocblas_operation_none, rocblas_operation_none,
            static_cast<rocblas_int>(m), static_cast<rocblas_int>(n),
            static_cast<rocblas_int>(k), &alpha, a, static_cast<rocblas_int>(m),
            b, static_cast<rocblas_int>(k), &beta, c, static_cast<rocblas_int>(m)),
            "rocblas_sgemm");
    };

    for (int i = 0; i < 3; ++i) {
        gemm();
    }
    hip_ok(hipDeviceSynchronize(), "warmup hipDeviceSynchronize");

    using clock = std::chrono::steady_clock;
    const auto started = clock::now();
    uint64_t iterations = 0;
    do {
        // A small queue amortizes host dispatch without creating an unbounded
        // backlog past the declared wall-clock window.
        for (int queued = 0; queued < 4; ++queued) {
            gemm();
            ++iterations;
        }
        hip_ok(hipDeviceSynchronize(), "measurement hipDeviceSynchronize");
    } while (std::chrono::duration<double>(clock::now() - started).count() < duration_s);
    const double elapsed_s = std::chrono::duration<double>(clock::now() - started).count();
    const long double operations = 2.0L * static_cast<long double>(m) *
                                   static_cast<long double>(n) *
                                   static_cast<long double>(k) * iterations;
    const double tflops = static_cast<double>(operations / elapsed_s / 1.0e12L);

    std::printf(
        "{\"schema\":\"epyc.rocm_gemm_saturation.v1\",\"device_index\":%d,"
        "\"device_name\":\"%s\",\"arch\":\"%s\",\"m\":%lld,\"n\":%lld,"
        "\"k\":%lld,\"iterations\":%llu,\"requested_duration_s\":%.6f,"
        "\"elapsed_s\":%.6f,\"throughput_tflops\":%.6f}\n",
        device, properties.name, properties.gcnArchName,
        static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k),
        static_cast<unsigned long long>(iterations), duration_s, elapsed_s, tflops);

    rocblas_ok(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
    hip_ok(hipFree(c), "hipFree(C)");
    hip_ok(hipFree(b), "hipFree(B)");
    hip_ok(hipFree(a), "hipFree(A)");
    return 0;
}
