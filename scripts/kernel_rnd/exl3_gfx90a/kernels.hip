#include "runtime.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstring>
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx90a__)
#error EXL3 requires exact gfx90a, not a generic gfx9 or wave32 target
#endif
namespace exl3 {
using Half4 = _Float16 __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
__device__ inline float half_round(float x) { return __half2float(__float2half_rn(x)); }
__device__ inline float half_bits(uint16_t u) { return __half2float(__ushort_as_half(u)); }
__device__ inline uint16_t state_at(const uint32_t* tile,unsigned k,unsigned r,unsigned c) {
    unsigned element=((c%8)*4+(r%8)/2)*8+r%2+(r/8)*2+(c/8)*4;
    unsigned pos=((element+1)*k+256*k-16)%(256*k);
    uint64_t pair=(uint64_t(tile[pos/32])<<32)|tile[(pos/32+1)%(8*k)];
    return uint16_t(pair>>(48-pos%32));
}
__device__ inline float decode(uint16_t state,Codebook cb) {
    if(cb==Codebook::mcg){uint32_t u=(uint32_t(state)*0xcbac1fedu&0x8fff8fffu)^0x3b603b60u;
        return half_round(half_bits(uint16_t(u))+half_bits(uint16_t(u>>16)));}
    uint32_t u=uint32_t(state)*0x83dcd12du;
    uint16_t sum=0x6400u+(u&255)+((u>>8)&255)+((u>>16)&255)+(u>>24);
    return half_round(fmaf(half_bits(sum),half_bits(0x1eee),half_bits(0xc931)));
}
__device__ inline float packed_value(const Matrix& m,unsigned r,unsigned c,unsigned k=0) {
    k=k?k:m.rate_x2/2;
    const auto* tile=m.packed+((size_t(r/16)*(m.padded_outputs/16)+c/16)*8*k);
    return decode(state_at(tile,k,r%16,c%16),m.codebook);
}
__global__ void codebook_probe(const uint16_t* states,Codebook cb,unsigned count,float* values) {
    unsigned i=blockIdx.x*64+threadIdx.x;if(i<count)values[i]=decode(states[i],cb);
}
__global__ void state_probe(const uint32_t* tiles,unsigned k,Codebook cb,unsigned count,uint16_t* states,float* values) {
    unsigned i=blockIdx.x*64+threadIdx.x;
    if(i<count){unsigned r=(i%256)/16,c=i%16;auto s=state_at(tiles+(i/256)*8*k,k,r,c);states[i]=s;values[i]=decode(s,cb);}
}
__global__ void decode_weights(const Matrix* matrices,float* raw,unsigned inputs,unsigned outputs) {
    auto& m=matrices[blockIdx.y];size_t i=size_t(blockIdx.x)*64+threadIdx.x;
    if(i<size_t(inputs)*outputs)raw[size_t(blockIdx.y)*inputs*outputs+i]=packed_value(m,i/outputs,i%outputs);
}
// Canonical materialization uses normalized H products accumulated in ascending FP32
// order and binary16 RNE after each transform and scale. Padding remains part of W;
// zero-padding happens on the input, before the complete output is cropped.
__device__ inline float hsign(unsigned a,unsigned b) { return (__popc(a&b)&1)?-1.f:1.f; }
__global__ void materialize_left(const Matrix* matrices,const float* raw,float* tmp,unsigned inputs,unsigned outputs) {
    unsigned e=blockIdx.y;size_t i=size_t(blockIdx.x)*64+threadIdx.x;
    if(i>=size_t(inputs)*outputs)return;auto& m=matrices[e];size_t base=size_t(e)*inputs*outputs;
    if(m.transform==Transform::none){tmp[base+i]=raw[base+i];return;}
    unsigned r=i/outputs,c=i%outputs;float sum=0;
    for(unsigned j=0;j<128;++j)sum=fmaf(hsign(r%128,j)*0.08838834764831845f,raw[base+size_t(r/128*128+j)*outputs+c],sum);
    tmp[base+i]=half_round(half_round(sum)*m.suh[r]);
}
__global__ void materialize_right(const Matrix* matrices,const float* tmp,float* out,unsigned inputs,unsigned outputs) {
    unsigned e=blockIdx.y;size_t i=size_t(blockIdx.x)*64+threadIdx.x;
    if(i>=size_t(inputs)*outputs)return;auto& m=matrices[e];size_t base=size_t(e)*inputs*outputs;
    if(m.transform==Transform::none){out[base+i]=tmp[base+i];return;}
    unsigned r=i/outputs,c=i%outputs;float sum=0;
    for(unsigned j=0;j<128;++j)sum=fmaf(tmp[base+size_t(r)*outputs+c/128*128+j],hsign(j,c%128)*0.08838834764831845f,sum);
    out[base+i]=half_round(half_round(sum)*m.svh[c]);
}
__global__ void prepare_activations(const Matrix* matrices,const DeviceRoute* routes,const float* x,float* a,
                                    unsigned inputs,unsigned padded_inputs,unsigned route_count,OperatorMode mode) {
    unsigned route=blockIdx.y,i=blockIdx.x*64+threadIdx.x;if(route>=route_count||i>=padded_inputs)return;
    auto r=routes[route];float sum=0;
    if(r.local_expert>=0){const auto& m=matrices[r.local_expert];
        if(mode==OperatorMode::fused_activation_transform&&m.transform!=Transform::none){
            for(unsigned j=0;j<128;++j){unsigned pos=i/128*128+j;
                // Do not read poisoned activation/scale tails for logical padding.
                float v=pos<inputs?x[size_t(r.token)*inputs+pos]*m.suh[pos]:0.f;
                sum=fmaf(hsign(i%128,j)*0.08838834764831845f,v,sum);}
            sum=half_round(sum);
        }else if(i<inputs)sum=x[size_t(r.token)*inputs+i];
    } a[size_t(route)*padded_inputs+i]=sum;
}
// Native packed wave64: lanes own disjoint output columns, keep ascending input
// FP32 FMA order, and therefore require no nondeterministic cross-lane reduction.
// K=0 is runtime-K; K1..K8 instantiate the grouped specialization.
template<unsigned K>
__global__ void gemv(const Matrix* matrices,const DeviceRoute* routes,const uint32_t* order,
                      unsigned first,unsigned count,int expert,const float* a,const float* weights,
                      float* out,unsigned inputs,unsigned outputs,OperatorMode mode) {
    unsigned q=blockIdx.y;if(q>=count)return;unsigned route=order?order[first+q]:first+q;
    const auto rr=routes[route];unsigned c=blockIdx.x*64+threadIdx.x;if(c>=outputs)return;
    float sum=0.f;
    if(rr.local_expert>=0&&(expert<0||rr.local_expert==expert)){
        const auto& m=matrices[rr.local_expert];const float* act=a+size_t(route)*inputs;
        for(unsigned i=0;i<inputs;++i){float w=mode==OperatorMode::fused_activation_transform?
            packed_value(m,i,c,K):weights[(size_t(rr.local_expert)*inputs+i)*outputs+c];
            sum=fmaf(w,act[i],sum);}
    }out[size_t(route)*outputs+c]=sum;
}
// CDNA2 16x16x16: lane l supplies row/column l%16 and reduction
// elements 4*(l/16)+j. Accumulator j maps to row 4*(l/16)+j, column l%16.
// One wave computes 16 route rows x 16 output columns. Packed B fragments are
// decoded directly to registers; no global unpack is used in fused mode.
__global__ void mfma(const Matrix* matrices,const DeviceRoute* routes,const uint32_t* order,
                      unsigned first,unsigned count,unsigned expert,const float* a,const float* weights,
                      float* out,unsigned inputs,unsigned outputs,OperatorMode mode) {
    unsigned lane=threadIdx.x,mrow=blockIdx.y*16+lane%16,ncol=blockIdx.x*16+lane%16;
    unsigned group=lane/16;Float4 acc={0,0,0,0};auto& mat=matrices[expert];
    for(unsigned k0=0;k0<inputs;k0+=16){Half4 av,bv;
        for(unsigned j=0;j<4;++j){unsigned ki=k0+group*4+j;
            unsigned route=mrow<count?(order?order[first+mrow]:first+mrow):0;
            av[j]=mrow<count?_Float16(a[size_t(route)*inputs+ki]):_Float16(0);
            float v=ncol<outputs?(mode==OperatorMode::fused_activation_transform?
                packed_value(mat,ki,ncol):weights[(size_t(expert)*inputs+ki)*outputs+ncol]):0.f;
            bv[j]=_Float16(v);
        }
        acc=__builtin_amdgcn_mfma_f32_16x16x16f16(av,bv,acc,0,0,0);
    }
    for(unsigned j=0;j<4;++j){unsigned row=blockIdx.y*16+group*4+j;
        if(row<count && ncol<outputs){unsigned route=order?order[first+row]:first+row;
            out[size_t(route)*outputs+ncol]=acc[j];}}
}
__global__ void finish_routes(const Matrix* matrices,const DeviceRoute* routes,const float* raw,float* partial,
                               unsigned logical_outputs,unsigned padded_outputs,unsigned count,OperatorMode mode) {
    unsigned route=blockIdx.y,c=blockIdx.x*64+threadIdx.x;if(route>=count||c>=logical_outputs)return;
    auto rr=routes[route];float value=0;
    if(rr.local_expert>=0){const auto& m=matrices[rr.local_expert];
        if(mode==OperatorMode::fused_activation_transform&&m.transform!=Transform::none){
            for(unsigned j=0;j<128;++j)value=fmaf(raw[size_t(route)*padded_outputs+c/128*128+j],
                hsign(j,c%128)*0.08838834764831845f,value);
            value*=m.svh[c];
        }else value=raw[size_t(route)*padded_outputs+c];
        if(m.bias)value+=m.bias[c];value*=rr.weight;
    }partial[size_t(route)*logical_outputs+c]=value;
}
__global__ void gather_fixed(const float* partial,float* output,unsigned topk,unsigned outputs,unsigned tokens) {
    unsigned c=blockIdx.x*64+threadIdx.x,t=blockIdx.y;if(c>=outputs||t>=tokens)return;
    float sum=0;for(unsigned k=0;k<topk;++k)sum+=partial[(size_t(t)*topk+k)*outputs+c];
    output[size_t(t)*outputs+c]=sum;
}
__global__ void gather_atomic(const float* partial,const DeviceRoute* routes,float* output,unsigned outputs,unsigned count) {
    unsigned c=blockIdx.x*64+threadIdx.x,r=blockIdx.y;if(c<outputs&&r<count&&routes[r].local_expert>=0)
        atomicAdd(output+size_t(routes[r].token)*outputs+c,partial[size_t(r)*outputs+c]);
}
__global__ void mfma_probe(const float* a,const float* b,float* c) {
    unsigned lane=threadIdx.x;Half4 av,bv;Float4 cv={0,0,0,0};
    for(unsigned j=0;j<4;++j){av[j]=_Float16(a[(lane%16)*16+(lane/16)*4+j]);bv[j]=_Float16(b[((lane/16)*4+j)*16+lane%16]);}
    cv=__builtin_amdgcn_mfma_f32_16x16x16f16(av,bv,cv,0,0,0);
    for(unsigned j=0;j<4;++j)c[((lane/16)*4+j)*16+lane%16]=cv[j];
}
hipError_t device_contract(int device) {
    hipDeviceProp_t p{};auto err=hipGetDeviceProperties(&p,device);if(err!=hipSuccess)return err;
    // A feature suffix is legal but the architecture identity must be exact.
    if(std::strncmp(p.gcnArchName,"gfx90a",6)|| (p.gcnArchName[6]&&p.gcnArchName[6]!=':')||p.warpSize!=64||
       !std::strstr(p.gcnArchName,":sramecc+")||!std::strstr(p.gcnArchName,":xnack-"))
        return hipErrorNoBinaryForGpu;
    int runtime=0;err=hipRuntimeGetVersion(&runtime);if(err!=hipSuccess)return err;
    if(runtime/10000000!=6 || (runtime/100000)%100!=2)return hipErrorNotSupported;
    return hipSuccess;
}
static uint64_t bound_identity(const Bound& b) {
    uint64_t h=identity_hash(b.plan);auto feed=[&](uint64_t v){h^=v;h*=1099511628211ULL;};
    feed(uintptr_t(b.arena));feed(uintptr_t(b.matrices));feed(uintptr_t(b.routes));feed(uintptr_t(b.counts));feed(uintptr_t(b.order));
    feed(uintptr_t(b.raw));feed(uintptr_t(b.weights));feed(uintptr_t(b.partial));feed(uintptr_t(b.activations));feed(uintptr_t(b.output_transform));
    feed(uintptr_t(b.stream));feed(b.device);return h;
}
static hipError_t pointer_extent(const void* ptr,size_t bytes,int device) {
    if(!ptr)return hipErrorInvalidDevicePointer;
    hipPointerAttribute_t attr{};auto err=hipPointerGetAttributes(&attr,ptr);if(err!=hipSuccess)return err;
    if(attr.device!=device||attr.type!=hipMemoryTypeDevice)return hipErrorInvalidDevicePointer;
    void* base=nullptr;size_t available=0;err=hipMemGetAddressRange(&base,&available,const_cast<void*>(ptr));
    if(err!=hipSuccess)return err;
    size_t offset=uintptr_t(ptr)-uintptr_t(base);
    return offset<=available && bytes<=available-offset?hipSuccess:hipErrorInvalidDevicePointer;
}
hipError_t reconstruct(Bound& b) {
    if(!b.plan.config.materialized_reference||b.identity!=bound_identity(b))return hipErrorInvalidValue;
    hipError_t err=hipSuccess;int device=-1;
    if((err=hipGetDevice(&device))!=hipSuccess)return err;
    if(device!=b.device)return hipErrorInvalidDevice;
    if((err=pointer_extent(b.arena,b.plan.arena_bytes,device))!=hipSuccess)return err;
    const auto& c=b.plan.config;dim3 grid((size_t(c.padded_inputs)*c.padded_outputs+63)/64,c.experts);
    hipLaunchKernelGGL(decode_weights,grid,dim3(64),0,b.stream,b.matrices,b.raw,c.padded_inputs,c.padded_outputs);
    if((err=hipGetLastError())!=hipSuccess)return err;
    hipLaunchKernelGGL(materialize_left,grid,dim3(64),0,b.stream,b.matrices,b.raw,b.weights,c.padded_inputs,c.padded_outputs);
    if((err=hipGetLastError())!=hipSuccess)return err;
    hipLaunchKernelGGL(materialize_right,grid,dim3(64),0,b.stream,b.matrices,b.weights,b.raw,c.padded_inputs,c.padded_outputs);
    if((err=hipGetLastError())!=hipSuccess)return err;
    // raw now carries fully reconstructed weights; scratch names retain fixed addresses.
    return hipGetLastError();
}
hipError_t bind(const Plan& p,void* arena,size_t bytes,hipStream_t stream,Bound& out) {
    if(!arena || uintptr_t(arena)%256 || bytes<p.arena_bytes || p.identity!=identity_hash(p))return hipErrorInvalidValue;
    int device=-1;auto err=hipGetDevice(&device);if(err!=hipSuccess)return err;
    err=device_contract(device);if(err!=hipSuccess)return err;
    if((err=pointer_extent(arena,bytes,device))!=hipSuccess)return err;
    for(unsigned i=0;i<p.config.experts;++i){const auto& m=p.matrices[i];
        if((err=pointer_extent(m.packed,m.packed_bytes,device))!=hipSuccess)return err;
        if(m.suh&&(err=pointer_extent(m.suh,m.padded_inputs*sizeof(float),device))!=hipSuccess)return err;
        if(m.svh&&(err=pointer_extent(m.svh,m.padded_outputs*sizeof(float),device))!=hipSuccess)return err;
        if(m.bias&&(err=pointer_extent(m.bias,m.outputs*sizeof(float),device))!=hipSuccess)return err;
    }
    Bound b{};b.plan=p;b.arena=arena;b.stream=stream;b.device=device;
    auto* base=static_cast<uint8_t*>(arena);
    b.matrices=reinterpret_cast<Matrix*>(base+p.matrices_offset);b.routes=reinterpret_cast<DeviceRoute*>(base+p.routes_offset);
    b.counts=reinterpret_cast<uint32_t*>(base+p.counts_offset);b.order=reinterpret_cast<uint32_t*>(base+p.order_offset);
    b.raw=reinterpret_cast<float*>(base+p.raw_offset);b.weights=reinterpret_cast<float*>(base+p.transformed_offset);
    b.partial=reinterpret_cast<float*>(base+p.partial_offset);b.activations=reinterpret_cast<float*>(base+p.activations_offset);
    b.output_transform=reinterpret_cast<float*>(base+p.output_transform_offset);
    err=hipMemcpyAsync(b.matrices,p.matrices.data(),p.config.experts*sizeof(Matrix),hipMemcpyHostToDevice,stream);
    if(err!=hipSuccess)return err;
    b.identity=bound_identity(b);
    if(p.config.materialized_reference){err=reconstruct(b);if(err!=hipSuccess)return err;}
    out=b;return hipSuccess;
}
hipError_t run(Bound& b,const Schedule& s,const float* input,float* output,OperatorMode mode,Compute compute) {
    const auto& c=b.plan.config;
    int device=-1;auto err=hipGetDevice(&device);if(err!=hipSuccess)return err;
    if(device!=b.device || b.identity!=bound_identity(b) || s.tokens>c.tokens || s.size!=s.tokens*c.topk ||
       s.size>c.routes || !input || !output || (mode!=OperatorMode::fused_activation_transform&&mode!=OperatorMode::materialized_weight_fp32_fma_v1)||
       (compute!=Compute::wave64_gemv&&compute!=Compute::mfma_prefill)|| (compute==Compute::mfma_prefill&&s.tokens>c.prefill_rows))return hipErrorInvalidValue;
    if(mode==OperatorMode::materialized_weight_fp32_fma_v1&&!c.materialized_reference)return hipErrorInvalidValue;
    if(!s.size)return hipSuccess;
    hipStreamCaptureStatus capture{};err=hipStreamIsCapturing(b.stream,&capture);if(err!=hipSuccess)return err;
    // Pageable route uploads and host-dependent routing are not graph safe. Fail closed.
    if(capture!=hipStreamCaptureStatusNone)return hipErrorStreamCaptureUnsupported;

    size_t input_bytes=size_t(s.tokens)*c.inputs*sizeof(float),output_bytes=size_t(s.tokens)*c.outputs*sizeof(float);
    if((err=pointer_extent(input,input_bytes,device))!=hipSuccess)return err;
    if((err=pointer_extent(output,output_bytes,device))!=hipSuccess)return err;
    auto overlaps=[](const void* a,size_t an,const void* d,size_t dn){return uintptr_t(a)<uintptr_t(d)+dn&&uintptr_t(d)<uintptr_t(a)+an;};
    if(overlaps(input,input_bytes,output,output_bytes)||overlaps(input,input_bytes,b.arena,b.plan.arena_bytes)||
       overlaps(output,output_bytes,b.arena,b.plan.arena_bytes))return hipErrorInvalidValue;
    // Validate a caller-provided Schedule too: never trust a forged/stale host record.
    for(unsigned i=0;i<s.size;++i){auto r=s.routes[i];
        if(r.local_expert < -1 || r.local_expert>=int(c.experts))return hipErrorInvalidValue;
        // Nonlocal identity is intentionally absent in DeviceRoute; counts/order still checked below.
        if(r.token!=i/c.topk||r.routing_k!=i%c.topk||!std::isfinite(r.weight))return hipErrorInvalidValue;
    }
    for(unsigned e=0;e<c.experts;++e){const auto& m=b.plan.matrices[e];
        if(overlaps(output,output_bytes,m.packed,m.packed_bytes)||
           (m.suh&&overlaps(output,output_bytes,m.suh,m.padded_inputs*sizeof(float)))||
           (m.svh&&overlaps(output,output_bytes,m.svh,m.padded_outputs*sizeof(float)))||
           (m.bias&&overlaps(output,output_bytes,m.bias,m.outputs*sizeof(float))))return hipErrorInvalidValue;
    }
    unsigned total=0;std::array<bool,max_routes> seen{};
    for(unsigned e=0;e<=c.experts;++e){if(s.counts[e]>s.size-total)return hipErrorInvalidValue;
        for(unsigned j=0;j<s.counts[e];++j){unsigned ri=s.order[total+j];if(ri>=s.size||seen[ri])return hipErrorInvalidValue;
            seen[ri]=true;auto le=s.routes[ri].local_expert;
            if((le<0?c.experts:unsigned(le))!=e)return hipErrorInvalidValue;}
        total+=s.counts[e];}
    if(total!=s.size)return hipErrorInvalidValue;
    unsigned active=0;
    for(unsigned e=0;e<c.experts;++e){unsigned bucket=0;
        for(unsigned j=0;j<c.experts;++j)if(s.counts[j]&&b.plan.matrices[j].rate_x2==b.plan.matrices[e].rate_x2&&b.plan.matrices[j].codebook==b.plan.matrices[e].codebook)++bucket;
        const auto& m=b.plan.matrices[e];auto path=capability(m.rate_x2,m.codebook,m.layout,c.policy,c.features,s.tokens,s.counts[e],bucket);
        if(path!=s.paths[e]||!executable(path))return hipErrorInvalidValue;if(s.counts[e])++active;}
    if((c.policy==Policy::grouped?active:1)>c.scheduler_slices)return hipErrorInvalidValue;
    if((err=hipMemcpyAsync(b.routes,s.routes.data(),s.size*sizeof(DeviceRoute),hipMemcpyHostToDevice,b.stream))!=hipSuccess)return err;
    if((err=hipMemcpyAsync(b.order,s.order.data(),s.size*sizeof(uint32_t),hipMemcpyHostToDevice,b.stream))!=hipSuccess)return err;
    if((err=hipMemcpyAsync(b.counts,s.counts.data(),(c.experts+1)*sizeof(uint32_t),hipMemcpyHostToDevice,b.stream))!=hipSuccess)return err;
    hipLaunchKernelGGL(prepare_activations,dim3((c.padded_inputs+63)/64,s.size),dim3(64),0,b.stream,b.matrices,b.routes,input,b.activations,c.inputs,c.padded_inputs,s.size,mode);
    if((err=hipGetLastError())!=hipSuccess)return err;
    auto launch=[&](unsigned k,unsigned first,unsigned count,int expert,const uint32_t* order)->hipError_t{
        dim3 grid((c.padded_outputs+63)/64,count);
#define LAUNCH(K) hipLaunchKernelGGL(gemv<K>,grid,dim3(64),0,b.stream,b.matrices,b.routes,order,first,count,expert,b.activations,b.raw,b.output_transform,c.padded_inputs,c.padded_outputs,mode)
        switch(k){case 1:LAUNCH(1);break;case 2:LAUNCH(2);break;case 3:LAUNCH(3);break;case 4:LAUNCH(4);break;
            case 5:LAUNCH(5);break;case 6:LAUNCH(6);break;case 7:LAUNCH(7);break;case 8:LAUNCH(8);break;default:LAUNCH(0);}
#undef LAUNCH
        return hipGetLastError();
    };
    bool batch_fallback=c.policy==Policy::grouped&&(!c.features.bc||!c.features.extension);
    if(compute==Compute::wave64_gemv&&(c.policy==Policy::unified||batch_fallback)){
        if((err=launch(0,0,s.size,-1,nullptr))!=hipSuccess)return err;
        ++b.path_launches[unsigned(batch_fallback?Path::batched_fallback:Path::unified)];}
    else {unsigned first=0;
        for(unsigned e=0;e<c.experts;++e){unsigned count=s.counts[e];
            if(count){if(compute==Compute::mfma_prefill){
                hipLaunchKernelGGL(mfma,dim3((c.padded_outputs+15)/16,(count+15)/16),dim3(64),0,b.stream,b.matrices,b.routes,b.order,first,count,e,b.activations,b.raw,b.output_transform,c.padded_inputs,c.padded_outputs,mode);
                if((err=hipGetLastError())!=hipSuccess)return err;}
                else if((err=launch(s.paths[e]==Path::grouped?b.plan.matrices[e].rate_x2/2:0,first,count,e,b.order))!=hipSuccess)return err;
                ++b.path_launches[unsigned(s.paths[e])];}
            first+=count;}
        // Explicit nonlocal rows are zeroed without dereferencing matrices[E].
        if(s.counts[c.experts]&&(err=launch(0,first,s.counts[c.experts],-1,b.order))!=hipSuccess)return err;
    }
    hipLaunchKernelGGL(finish_routes,dim3((c.outputs+63)/64,s.size),dim3(64),0,b.stream,b.matrices,b.routes,b.output_transform,b.partial,c.outputs,c.padded_outputs,s.size,mode);
    if((err=hipGetLastError())!=hipSuccess)return err;
    if(c.reduction==Reduction::deterministic){
        hipLaunchKernelGGL(gather_fixed,dim3((c.outputs+63)/64,s.tokens),dim3(64),0,b.stream,b.partial,output,c.topk,c.outputs,s.tokens);
        if((err=hipGetLastError())!=hipSuccess)return err;}
    else {
        if((err=hipMemsetAsync(output,0,size_t(s.tokens)*c.outputs*sizeof(float),b.stream))!=hipSuccess)return err;
        hipLaunchKernelGGL(gather_atomic,dim3((c.outputs+63)/64,s.size),dim3(64),0,b.stream,b.partial,b.routes,output,c.outputs,s.size);
    if((err=hipGetLastError())!=hipSuccess)return err;
    }return hipGetLastError();
}
hipError_t probe_states(const uint32_t* t,unsigned k,Codebook cb,unsigned n,uint16_t* s,float* v,hipStream_t stream){
    if(!k||k>8||!t||!s||!v||(cb!=Codebook::mcg&&cb!=Codebook::mul1))return hipErrorInvalidValue;
    if(!n)return hipSuccess;
    hipLaunchKernelGGL(state_probe,dim3((n+63)/64),dim3(64),0,stream,t,k,cb,n,s,v);return hipGetLastError();
}
hipError_t probe_codebook(const uint16_t* states,Codebook cb,unsigned count,float* values,hipStream_t stream) {
    if(!states||!values||(cb!=Codebook::mcg&&cb!=Codebook::mul1))return hipErrorInvalidValue;
    if(!count)return hipSuccess;
    hipLaunchKernelGGL(codebook_probe,dim3((count+63)/64),dim3(64),0,stream,states,cb,count,values);return hipGetLastError();
}
hipError_t probe_mfma(const float* a,const float* b,float* c,hipStream_t stream){
    if(!a||!b||!c)return hipErrorInvalidValue;
    hipLaunchKernelGGL(mfma_probe,dim3(1),dim3(64),0,stream,a,b,c);return hipGetLastError();
}
} // namespace exl3
