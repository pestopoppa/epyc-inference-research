#pragma once
#include "contract.hpp"
#include <hip/hip_runtime.h>
namespace exl3 {
enum class OperatorMode : uint8_t { materialized_weight_fp32_fma_v1, fused_activation_transform };
enum class Compute : uint8_t { wave64_gemv, mfma_prefill };
struct Bound {
    Plan plan{};
    void* arena=nullptr;
    Matrix* matrices=nullptr;
    DeviceRoute* routes=nullptr;
    uint32_t* counts=nullptr;
    uint32_t* order=nullptr;
    float* raw=nullptr;
    float* weights=nullptr;
    float* partial=nullptr;
    float* activations=nullptr;
    float* output_transform=nullptr;
    hipStream_t stream=nullptr;
    int device=-1;
    uint64_t identity=0;
    std::array<uint64_t,14> path_launches{};
};
// Caller owns all device allocations and their lifetimes. Bind uploads pointer tables;
// no function here performs hipMalloc/free, new/delete, or host heap allocation.
hipError_t bind(const Plan&,void* arena,size_t bytes,hipStream_t,Bound&);
hipError_t run(Bound&,const Schedule&,const float* input,float* output,
               OperatorMode,Compute);
hipError_t reconstruct(Bound&);
hipError_t probe_states(const uint32_t* tiles,unsigned k,Codebook,unsigned count,
                        uint16_t* states,float* values,hipStream_t);
hipError_t probe_codebook(const uint16_t* states,Codebook,unsigned count,float* values,hipStream_t);
hipError_t probe_mfma(const float* a,const float* b,float* c,hipStream_t);
hipError_t device_contract(int device);
} // namespace exl3
