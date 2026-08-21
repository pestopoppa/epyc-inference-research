#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t status = (call);                                             \
        if (status != hipSuccess) {                                             \
            std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                         hipGetErrorString(status));                            \
            return 2;                                                          \
        }                                                                      \
    } while (0)

__global__ void pc_sampling_spin(float* output, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    float value = static_cast<float>(index & 31) + 1.0f;
    for (int step = 0; step < iterations; ++step) {
        value = fmaf(value, 1.00000011920928955078125f, 0.00000095367431640625f);
    }
    output[index] = value;
}

int main() {
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx90a", 6) != 0 ||
        (properties.gcnArchName[6] != '\0' && properties.gcnArchName[6] != ':')) {
        std::fprintf(stderr, "REFUSE: expected exact gfx90a, observed %s\n",
                     properties.gcnArchName);
        return 3;
    }

    constexpr int blocks = 512;
    constexpr int threads = 256;
    constexpr int iterations = 32768;
    float* output = nullptr;
    HIP_CHECK(hipMalloc(&output, blocks * threads * sizeof(float)));
    for (int launch = 0; launch < 32; ++launch) {
        hipLaunchKernelGGL(pc_sampling_spin, dim3(blocks), dim3(threads), 0, 0,
                           output, iterations);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    std::puts("{\"status\":\"ok\",\"arch\":\"gfx90a\",\"launches\":32}");
    return 0;
}
