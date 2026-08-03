// HIP conformance harness — runs the REAL ggml device decoder ON THE MI210.
//
// WHY IT MUST RUN ON-DEVICE
//   ggml_cuda_e8m0_to_fp32 is `static __device__ __forceinline__` and its behaviour
//   is selected by a preprocessor branch on CUDART_VERSION. Reading the source tells
//   you which branch *should* be taken; only executing it on the real toolchain and
//   the real card tells you which one *is*. That distinction is the entire reason a
//   backend row is ASSERTED until a harness like this runs.
//
// READ-ONLY on the production tree: common.cuh is included, never modified, and
// nothing here links into any production artifact.
#include "common.cuh"

#include <cstdio>
#include <cstdint>

#define NCODES 9
__constant__ uint8_t d_dummy;  // keeps the translation unit honest under -Wall

__global__ void decode_e8m0(const uint8_t* codes, uint32_t* out_bits, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float f = ggml_cuda_e8m0_to_fp32(codes[i]);
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    out_bits[i] = b;
}

int main() {
    const uint8_t h_codes[NCODES] = {0, 1, 2, 126, 127, 128, 253, 254, 255};
    uint8_t*  d_codes = nullptr;
    uint32_t* d_out   = nullptr;
    uint32_t  h_out[NCODES] = {0};

    if (hipMalloc(&d_codes, sizeof(h_codes)) != hipSuccess ||
        hipMalloc(&d_out, sizeof(h_out)) != hipSuccess) {
        fprintf(stderr, "hipMalloc failed\n");
        return 2;
    }
    hipMemcpy(d_codes, h_codes, sizeof(h_codes), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(decode_e8m0, dim3(1), dim3(NCODES), 0, 0, d_codes, d_out, NCODES);
    if (hipDeviceSynchronize() != hipSuccess) {
        fprintf(stderr, "kernel launch/sync failed: %s\n", hipGetErrorString(hipGetLastError()));
        return 3;
    }
    hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost);

    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);

    printf("{\n  \"backend\": \"hip\",\n");
    printf("  \"arch\": \"%s\",\n", p.gcnArchName);
    printf("  \"source\": \"ggml/src/ggml-cuda/common.cuh (frozen production tree, included not copied)\",\n");
#if defined(CUDART_VERSION)
    printf("  \"cudart_version_defined\": true,\n  \"cudart_version\": %d,\n", CUDART_VERSION);
#else
    printf("  \"cudart_version_defined\": false,\n");
#endif
    printf("  \"e8m0_device\": [\n");
    for (int i = 0; i < NCODES; ++i) {
        printf("    {\"code\": %u, \"bits\": \"0x%08x\"}%s\n",
               h_codes[i], h_out[i], i + 1 < NCODES ? "," : "");
    }
    printf("  ]\n}\n");

    hipFree(d_codes);
    hipFree(d_out);
    return 0;
}
