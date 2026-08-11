// Self-contained gfx90a LDS probe derived from the HipKittens experiment method.
// Provenance: HazyResearch/HipKittens @ a288366e4245528f74540b3fe446637cf8345745,
// analysis/paper_experiments/phases/ds_read_b128 (MIT).  This is not a vendored
// framework component: it uses only HIP plus one ds_read_b128 instruction.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define HIP_CHECK(call) do {                                                   \
    hipError_t status_ = (call);                                               \
    if (status_ != hipSuccess) {                                               \
        std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(status_)); \
        return 2;                                                              \
    }                                                                          \
} while (0)

using floatx4_t = float __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(64, 1)
void autokernel_ds_read_b128_probe(
        uint32_t thread_a, uint32_t thread_b, uint32_t offset_a, uint32_t offset_b) {
    extern __shared__ uint32_t smem[];
    const uint32_t tid = threadIdx.x;
    if (tid == thread_a || tid == thread_b) {
        const uint32_t offset = tid == thread_a ? offset_a : offset_b;
        const uint32_t address = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(&smem[0])) + offset;
        floatx4_t data;
        asm volatile("ds_read_b128 %0, %1 offset:0"
                     : "=v"(data) : "v"(address) : "memory");
        asm volatile("" : : "v"(data) : "memory");
    }
}

static int launch(uint32_t a, uint32_t b, uint32_t offset_a, uint32_t offset_b) {
    hipLaunchKernelGGL(
        autokernel_ds_read_b128_probe, dim3(1), dim3(64), 65536, 0,
        a, b, offset_a, offset_b);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

static int parse_positive(const char *value, const char *label) {
    char *end = nullptr;
    long result = std::strtol(value, &end, 10);
    if (value[0] == '\0' || *end != '\0' || result < 1 || result > 1000000) {
        std::fprintf(stderr, "%s must be a positive integer\n", label);
        std::exit(64);
    }
    return static_cast<int>(result);
}

int main(int argc, char **argv) {
    if (argc < 2 || (std::strcmp(argv[1], "bank") != 0 &&
                     std::strcmp(argv[1], "phase") != 0)) {
        std::fprintf(stderr,
            "usage: %s bank <max-bank> <repetitions> | phase <bank-count> <repetitions>\n",
            argv[0]);
        return 64;
    }
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    std::string arch(properties.gcnArchName);
    if (arch.rfind("gfx90a", 0) != 0 || properties.warpSize != 64) {
        std::fprintf(stderr, "refusing non-gfx90a/wave64 target: arch=%s wave=%d\n",
                     properties.gcnArchName, properties.warpSize);
        return 65;
    }
    HIP_CHECK(hipFuncSetAttribute(
        reinterpret_cast<const void *>(autokernel_ds_read_b128_probe),
        hipFuncAttributeMaxDynamicSharedMemorySize, 65536));

    if (std::strcmp(argv[1], "bank") == 0) {
        if (argc != 4) return 64;
        const int max_bank = parse_positive(argv[2], "max-bank");
        const int repetitions = parse_positive(argv[3], "repetitions");
        for (int bank = 4; bank <= max_bank; ++bank) {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                int status = launch(0, 1, 0, static_cast<uint32_t>(bank * 4));
                if (status != 0) return status;
            }
        }
    } else {
        if (argc != 4) return 64;
        const int bank_count = parse_positive(argv[2], "bank-count");
        const int repetitions = parse_positive(argv[3], "repetitions");
        for (int a = 0; a < 64; ++a) {
            for (int b = a + 1; b < 64; ++b) {
                for (int repetition = 0; repetition < repetitions; ++repetition) {
                    int status = launch(
                        static_cast<uint32_t>(a), static_cast<uint32_t>(b),
                        static_cast<uint32_t>(a * bank_count * 4),
                        static_cast<uint32_t>(b * bank_count * 4));
                    if (status != 0) return status;
                }
            }
        }
    }
    return 0;
}
