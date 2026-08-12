#include <torch/extension.h>
#include <hip/hip_runtime.h>

namespace {

__global__ void silu_kernel(const float * input, float * output, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        output[index] = value / (1.0f + expf(-value));
    }
}

} // namespace

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "SiLU HIP input must be on the ROCm device");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "SiLU HIP input must be float32");
    TORCH_CHECK(input.is_contiguous(), "SiLU HIP input must be contiguous");
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int64_t count = input.numel();
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    hipLaunchKernelGGL(silu_kernel, dim3(blocks), dim3(threads), 0, 0,
                       input.data_ptr<float>(), output.data_ptr<float>(), count);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "SiLU HIP kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &forward, "SiLU forward (HIP)");
}
