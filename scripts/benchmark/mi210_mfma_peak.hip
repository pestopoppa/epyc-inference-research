// gfx90a MFMA peak-FLOPS microbenchmark.
// Register-resident operands, no memory traffic in the timed loop, so this
// measures matrix-core issue rate rather than anything about the memory system.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s at %d\n",hipGetErrorString(e),__LINE__); exit(9);} } while(0)

typedef _Float16 half4   __attribute__((ext_vector_type(4)));
typedef float    float4v __attribute__((ext_vector_type(4)));

#define UNROLL 8

__global__ __launch_bounds__(256) void mfma_peak(float* out, int iters) {
  half4 a, b;
  for (int i = 0; i < 4; ++i) {
    a[i] = (_Float16)(1.0f + (threadIdx.x & 3));
    b[i] = (_Float16)(1.0f + ((threadIdx.x >> 2) & 3));
  }
  float4v acc[UNROLL];
  for (int u = 0; u < UNROLL; ++u)
    for (int i = 0; i < 4; ++i) acc[u][i] = 0.0f;

  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      acc[u] = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc[u], 0, 0, 0);
  }

  float s = 0.0f;
  for (int u = 0; u < UNROLL; ++u)
    for (int i = 0; i < 4; ++i) s += acc[u][i];
  // never true at runtime, but the compiler cannot prove it -> the loop survives
  if (s == -1.0f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main(int argc, char** argv) {
  int blocks = (argc > 1) ? atoi(argv[1]) : 104 * 8;
  int iters  = (argc > 2) ? atoi(argv[2]) : 20000;
  int reps   = (argc > 3) ? atoi(argv[3]) : 12;
  const int TPB = 256, WAVES_PER_BLOCK = TPB / 64;

  hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p, 0));
  float* out; HIP_CHECK(hipMalloc(&out, (size_t)blocks * TPB * sizeof(float)));

  hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));

  // One v_mfma_f32_16x16x16f16 per WAVEFRONT = 2*16*16*16 = 8192 FLOP
  const double flops_per_mfma = 2.0 * 16 * 16 * 16;
  const double total_flops =
      (double)blocks * WAVES_PER_BLOCK * (double)iters * UNROLL * flops_per_mfma;

  std::vector<double> tf;
  for (int r = 0; r < reps; ++r) {
    HIP_CHECK(hipEventRecord(e0));
    hipLaunchKernelGGL(mfma_peak, dim3(blocks), dim3(TPB), 0, 0, out, iters);
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    if (r >= 2) tf.push_back(total_flops / (ms * 1e-3) / 1e12);
  }
  std::sort(tf.begin(), tf.end());
  double best = tf.back(), med = tf[tf.size() / 2];
  printf("{\"arch\":\"%s\",\"blocks\":%d,\"waves_per_block\":%d,\"iters\":%d,\"unroll\":%d,"
         "\"timed_reps\":%zu,\"best_TFLOPS\":%.2f,\"median_TFLOPS\":%.2f,\"min_TFLOPS\":%.2f}\n",
         p.gcnArchName, blocks, WAVES_PER_BLOCK, iters, UNROLL, tf.size(), best, med, tf.front());
  HIP_CHECK(hipFree(out));
  return 0;
}
