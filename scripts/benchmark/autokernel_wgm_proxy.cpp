// Diagnostic-only gfx90a grid-swizzle proxy.
//
// This is deliberately not an MMQ kernel.  It holds the kernel body constant
// and changes only the mapping from physical workgroup id to a logical M/N
// tile, exposing the L2 row-reuse half of HipKittens' WGM launch ordering on a
// single-GCD MI210.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(call) do {                                                   \
  hipError_t status = (call);                                                  \
  if (status != hipSuccess) {                                                  \
    std::fprintf(stderr, "HIP failure at %d: %s\n", __LINE__,                 \
                 hipGetErrorString(status));                                   \
    std::exit(9);                                                              \
  }                                                                            \
} while (0)

constexpr int kRows = 64;
constexpr int kCols = 64;
constexpr int kThreads = 256;

template <int WGM>
__global__ __launch_bounds__(kThreads)
void wgm_l2_proxy(const float* a, const float* b, float* output, int elements) {
  const int linear = static_cast<int>(blockIdx.x);
  int pid_m;
  int pid_n;
  if constexpr (WGM == 0) {
    pid_m = linear / kCols;
    pid_n = linear % kCols;
  } else {
    const int workgroups_per_group = WGM * kCols;
    const int group_id = linear / workgroups_per_group;
    const int first_pid_m = group_id * WGM;
    const int group_size_m = min(kRows - first_pid_m, WGM);
    const int in_group = linear % workgroups_per_group;
    pid_m = first_pid_m + (in_group % group_size_m);
    pid_n = in_group / group_size_m;
  }

  float sum = 0.0f;
  const float* a_row = a + static_cast<size_t>(pid_m) * elements;
  const float* b_row = b + static_cast<size_t>(pid_n) * elements;
  for (int index = static_cast<int>(threadIdx.x); index < elements;
       index += kThreads) {
    sum += a_row[index] * 0.75f + b_row[index] * 0.25f;
  }
  __shared__ float partial[kThreads];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width /= 2) {
    if (threadIdx.x < width) partial[threadIdx.x] += partial[threadIdx.x + width];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[pid_m * kCols + pid_n] = partial[0];
}

template <int WGM>
float launch(const float* a, const float* b, float* output, int elements,
             hipEvent_t begin, hipEvent_t end) {
  HIP_CHECK(hipEventRecord(begin));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(wgm_l2_proxy<WGM>),
                     dim3(kRows * kCols), dim3(kThreads), 0, 0,
                     a, b, output, elements);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(end));
  HIP_CHECK(hipEventSynchronize(end));
  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, begin, end));
  return elapsed_ms;
}

float dispatch(int factor, const float* a, const float* b, float* output,
               int elements, hipEvent_t begin, hipEvent_t end) {
  switch (factor) {
    case 0: return launch<0>(a, b, output, elements, begin, end);
    case 2: return launch<2>(a, b, output, elements, begin, end);
    case 4: return launch<4>(a, b, output, elements, begin, end);
    case 8: return launch<8>(a, b, output, elements, begin, end);
    case 16: return launch<16>(a, b, output, elements, begin, end);
    case 32: return launch<32>(a, b, output, elements, begin, end);
    default: std::fprintf(stderr, "unsupported WGM factor %d\n", factor); std::exit(8);
  }
}

int main(int argc, char** argv) {
  int rounds = 48;
  int elements = 131072;
  bool profile_once = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (arg == "--elements" && i + 1 < argc) elements = std::atoi(argv[++i]);
    else if (arg == "--profile-once") profile_once = true;
    else { std::fprintf(stderr, "unknown/incomplete argument: %s\n", argv[i]); return 2; }
  }
  if (rounds <= 0 || elements <= 0) return 2;

  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, 0));
  if (std::strncmp(properties.gcnArchName, "gfx90a", 6) != 0 || properties.warpSize != 64) {
    std::fprintf(stderr, "refusing non-gfx90a/wave64 device: %s wave=%d\n",
                 properties.gcnArchName, properties.warpSize);
    return 7;
  }

  const size_t row_values = static_cast<size_t>(kRows) * elements;
  std::vector<float> host_a(row_values), host_b(row_values);
  for (size_t i = 0; i < row_values; ++i) {
    host_a[i] = static_cast<float>((i * 17 + 3) % 251) / 251.0f;
    host_b[i] = static_cast<float>((i * 29 + 7) % 241) / 241.0f;
  }
  float *device_a = nullptr, *device_b = nullptr, *device_output = nullptr;
  HIP_CHECK(hipMalloc(&device_a, row_values * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_b, row_values * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_output, kRows * kCols * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_a, host_a.data(), row_values * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_b, host_b.data(), row_values * sizeof(float), hipMemcpyHostToDevice));
  hipEvent_t begin, end;
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));

  const int factors[] = {0, 2, 4, 8, 16, 32};
  std::vector<float> reference(kRows * kCols), observed(kRows * kCols);
  bool have_reference = false;
  for (int factor : factors) {
    dispatch(factor, device_a, device_b, device_output, elements, begin, end);
    HIP_CHECK(hipMemcpy(observed.data(), device_output, observed.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    if (!have_reference) { reference = observed; have_reference = true; }
    else if (std::memcmp(reference.data(), observed.data(), observed.size() * sizeof(float)) != 0) {
      std::fprintf(stderr, "logical output mismatch for WGM=%d\n", factor);
      return 6;
    }
  }

  std::printf("{\"type\":\"header\",\"arch\":\"%s\",\"wave_size\":%d,"
              "\"rows\":%d,\"cols\":%d,\"elements\":%d,\"rounds\":%d,"
              "\"correctness\":\"bit_exact\"}\n",
              properties.gcnArchName, properties.warpSize, kRows, kCols, elements,
              profile_once ? 1 : rounds);
  const int measured_rounds = profile_once ? 1 : rounds;
  for (int round = 0; round < measured_rounds; ++round) {
    // Rotation makes every factor occupy every ordinal position equally over a
    // six-round cycle, bounding monotonic thermal/clock drift.
    for (int ordinal = 0; ordinal < 6; ++ordinal) {
      const int factor = factors[(ordinal + round) % 6];
      const float elapsed_ms = dispatch(
          factor, device_a, device_b, device_output, elements, begin, end);
      std::printf("{\"type\":\"sample\",\"round\":%d,\"ordinal\":%d,"
                  "\"factor\":%d,\"elapsed_ms\":%.9g}\n",
                  round, ordinal, factor, elapsed_ms);
    }
  }
  std::fflush(stdout);

  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
  HIP_CHECK(hipFree(device_output));
  HIP_CHECK(hipFree(device_b));
  HIP_CHECK(hipFree(device_a));
  return 0;
}
