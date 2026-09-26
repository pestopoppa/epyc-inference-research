#include "runtime.hpp"
#include <hip/hip_fp16.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
using namespace exl3;
static unsigned assertions=0;
static std::array<uint64_t,14> total_path_launches{};
#define HIP(call) do { auto err=(call);if(err!=hipSuccess)throw std::runtime_error(std::string(#call)+": "+hipGetErrorString(err)); } while(0)
void check(bool b,const std::string& what){++assertions;if(!b)throw std::runtime_error(what);}
float hround(float x){return float(_Float16(x));}
float hbits(uint16_t bits){_Float16 h;std::memcpy(&h,&bits,2);return float(h);}
float oracle_cb(uint16_t state,Codebook cb){
    if(cb==Codebook::mcg){uint32_t x=uint32_t(state)*0xcbac1fedu;x=(x&0x8fff8fffu)^0x3b603b60u;
        return hround(hbits(uint16_t(x))+hbits(uint16_t(x>>16)));}
    uint32_t x=uint32_t(state)*0x83dcd12du;unsigned sum=0;
    for(unsigned i=0;i<4;++i)sum+=(x>>(8*i))&255;
    return hround(std::fma(hbits(uint16_t(0x6400+sum)),hbits(0x1eee),hbits(0xc931)));
}
uint16_t oracle_state(const uint32_t* words,unsigned k,unsigned row,unsigned col){
    // Invert the storage permutation by enumeration, then extract one bit at a
    // time. This does not reuse the GPU formula or its two-word extraction.
    unsigned element=0;
    for(unsigned lane=0;lane<32;++lane){unsigned r0=2*(lane%4),c0=lane/4;
        unsigned rr[8]={r0,r0+1,r0+8,r0+9,r0,r0+1,r0+8,r0+9};
        unsigned cc[8]={c0,c0,c0,c0,c0+8,c0+8,c0+8,c0+8};
        for(unsigned j=0;j<8;++j)if(rr[j]==row&&cc[j]==col)element=lane*8+j;
    }
    uint16_t state=0;int start=int((element+1)*k)-16;
    for(int j=0;j<16;++j){unsigned b=unsigned((start+j+int(256*k))%int(256*k));
        state=uint16_t((state<<1)|((words[b/32]>>(31-b%32))&1));}
    return state;
}
template<class T> struct Device {
    T* ptr=nullptr;size_t count=0;
    explicit Device(size_t n):count(n){HIP(hipMalloc(&ptr,n*sizeof(T)));}
    explicit Device(const std::vector<T>& v):Device(v.size()){put(v);}
    void put(const std::vector<T>& v){check(v.size()<=count,"device capacity");HIP(hipMemcpy(ptr,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(count);HIP(hipMemcpy(v.data(),ptr,count*sizeof(T),hipMemcpyDeviceToHost));return v;}
    ~Device(){if(ptr){auto err=hipFree(ptr);if(err!=hipSuccess){std::cerr<<"hipFree failed: "<<hipGetErrorString(err)<<"\n";std::terminate();}}}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
};
Identity identity(){Identity i;for(auto* d:{&i.model,&i.source,&i.tensor}){d->fill('a');(*d)[64]=0;}std::memcpy(i.role.data(),"gate",5);return i;}
void compare(const std::vector<float>& got,const std::vector<float>& want,float abs,float rel,const std::string& label){
    check(got.size()==want.size(),label+" shape");float max_error=0;
    for(size_t i=0;i<got.size();++i){float err=std::fabs(got[i]-want[i]);max_error=std::max(max_error,err);
        if(!std::isfinite(got[i])||!std::isfinite(want[i])||err>abs+rel*std::fabs(want[i])){
            std::ostringstream msg;msg<<label<<" index="<<i<<" actual="<<got[i]<<" expected="<<want[i]<<" error="<<err;throw std::runtime_error(msg.str());}}
    ++assertions;
}
void probe_tests(){
    std::vector<uint16_t> states(65536);for(unsigned i=0;i<65536;++i)states[i]=i;
    Device<uint16_t> ds(states);Device<float> dv(states.size());
    for(auto cb:{Codebook::mcg,Codebook::mul1}){HIP(probe_codebook(ds.ptr,cb,states.size(),dv.ptr,nullptr));
        auto values=dv.get();for(unsigned i=0;i<65536;++i)check(values[i]==oracle_cb(i,cb),"exhaustive procedural binary16 decode");}
    std::mt19937 random(1729);
    for(unsigned k=1;k<=8;++k){std::vector<uint32_t> words(8*k*17);for(auto& w:words)w=random();Device<uint32_t> dt(words);
        Device<uint16_t> ss(256*17);Device<float> vv(256*17);
        for(auto cb:{Codebook::mcg,Codebook::mul1}){HIP(probe_states(dt.ptr,k,cb,256*17,ss.ptr,vv.ptr,nullptr));auto st=ss.get();auto va=vv.get();
            for(unsigned i=0;i<256*17;++i){auto expected=oracle_state(words.data()+i/256*8*k,k,(i%256)/16,i%16);
                check(st[i]==expected,"wave64 lane map/state");check(va[i]==oracle_cb(expected,cb),"wave64 decoded lane");}}}
    for(unsigned pattern=0;pattern<4;++pattern){std::vector<float> a(256),b(256),wanted(256);
        for(unsigned r=0;r<16;++r)for(unsigned c=0;c<16;++c){
            a[r*16+c]=pattern==0?float(r*16+c+1):pattern==1?float(r==c):float(int(random()%31)-15)/8;
            b[r*16+c]=pattern==0?float(r==c):pattern==1?float((r+3)%16==c):float(int(random()%31)-15)/8;}
        for(unsigned r=0;r<16;++r)for(unsigned c=0;c<16;++c)for(unsigned k=0;k<16;++k)wanted[r*16+c]=std::fma(a[r*16+k],b[k*16+c],wanted[r*16+c]);
        Device<float> da(a),db(b),dc(256);HIP(probe_mfma(da.ptr,db.ptr,dc.ptr,nullptr));compare(dc.get(),wanted,0,0,"MFMA 256-cell map");}
    std::cout<<"{\"proposition\":\"procedural_fp16_and_wave64_lane_map\",\"status\":\"pass\",\"state_cases\":131072,\"packed_cases\":69632,\"mfma_cells\":1024}\n";
}
struct HostMatrix {std::vector<uint32_t> words;std::vector<float> suh,svh,bias,weights;};
float sign(unsigned i,unsigned j){return __builtin_parity(i&j)?-1.f:1.f;}
void materialize(HostMatrix& h,const Matrix& m){
    unsigned I=m.padded_inputs,O=m.padded_outputs,k=m.rate_x2/2;h.weights.resize(size_t(I)*O);
    for(unsigned i=0;i<I;++i)for(unsigned o=0;o<O;++o){const uint32_t* t=h.words.data()+(size_t(i/16)*(O/16)+o/16)*8*k;
        h.weights[size_t(i)*O+o]=oracle_cb(oracle_state(t,k,i%16,o%16),m.codebook);}
    if(m.transform==Transform::none)return;
    std::vector<float> tmp(h.weights.size());
    for(unsigned i=0;i<I;++i)for(unsigned o=0;o<O;++o){float sum=0;
        for(unsigned j=0;j<128;++j)sum=std::fma(sign(i%128,j)*0.08838834764831845f,h.weights[size_t(i/128*128+j)*O+o],sum);
        tmp[size_t(i)*O+o]=hround(hround(sum)*h.suh[i]);}
    for(unsigned i=0;i<I;++i)for(unsigned o=0;o<O;++o){float sum=0;
        for(unsigned j=0;j<128;++j)sum=std::fma(tmp[size_t(i)*O+o/128*128+j],sign(j,o%128)*0.08838834764831845f,sum);
        h.weights[size_t(i)*O+o]=hround(hround(sum)*h.svh[o]);}
}
std::vector<float> reference(const Config& c,const std::vector<Matrix>& matrices,const std::vector<HostMatrix>& host,
                             const std::vector<Route>& routes,const std::vector<float>& x,OperatorMode mode){
    std::vector<float> y(size_t(c.tokens)*c.outputs,0);
    for(const auto& route:routes){if(route.global_expert<c.local_begin||route.global_expert>=c.local_begin+c.experts)continue;
        unsigned e=route.global_expert-c.local_begin;auto& m=matrices[e];auto& hm=host[e];
        std::vector<float> a(c.padded_inputs,0),raw(c.padded_outputs,0);
        for(unsigned i=0;i<c.inputs;++i)a[i]=x[size_t(route.token)*c.inputs+i];
        if(mode==OperatorMode::fused_activation_transform&&m.transform!=Transform::none){auto before=a;
            for(unsigned i=0;i<c.padded_inputs;++i){float sum=0;
                for(unsigned j=0;j<128;++j){unsigned pos=i/128*128+j;float v=pos<c.inputs?before[pos]*hm.suh[pos]:0.f;
                    sum=std::fma(sign(i%128,j)*0.08838834764831845f,v,sum);}a[i]=hround(sum);}}
        for(unsigned o=0;o<c.padded_outputs;++o)for(unsigned i=0;i<c.padded_inputs;++i){
            float w=hm.weights[size_t(i)*c.padded_outputs+o];
            if(mode==OperatorMode::fused_activation_transform){unsigned k=m.rate_x2/2;auto* tile=hm.words.data()+(size_t(i/16)*(c.padded_outputs/16)+o/16)*8*k;
                w=oracle_cb(oracle_state(tile,k,i%16,o%16),m.codebook);}
            raw[o]=std::fma(w,a[i],raw[o]);}
        for(unsigned o=0;o<c.outputs;++o){float value=raw[o];
            if(mode==OperatorMode::fused_activation_transform&&m.transform!=Transform::none){value=0;
                for(unsigned j=0;j<128;++j)value=std::fma(raw[o/128*128+j],sign(j,o%128)*0.08838834764831845f,value);
                value*=hm.svh[o];}
            if(!hm.bias.empty())value+=hm.bias[o];y[size_t(route.token)*c.outputs+o]+=value*route.weight;
        }
    }return y;
}
void operator_case(unsigned tokens,unsigned topk,unsigned experts,unsigned inputs,unsigned outputs,bool transformed,
                    Policy policy,Features features,Reduction reduction,unsigned routing,bool use_mfma,bool benchmark=false,bool enable_reference=true){
    Config c{};c.tokens=tokens;c.topk=topk;c.routes=tokens*topk;c.experts=experts;c.global_experts=experts+2;c.local_begin=1;
    c.inputs=inputs;c.outputs=outputs;unsigned tile=transformed?128:16;
    c.padded_inputs=(inputs+tile-1)/tile*tile;c.padded_outputs=(outputs+tile-1)/tile*tile;
    c.prefill_rows=tokens;c.scheduler_slices=experts;c.policy=policy;c.features=features;c.reduction=reduction;c.materialized_reference=enable_reference;
    std::mt19937 random(101+inputs+outputs);std::vector<Matrix> matrices(experts);std::vector<HostMatrix> host(experts);
    std::vector<std::unique_ptr<Device<uint32_t>>> packed;std::vector<std::unique_ptr<Device<float>>> scales;
    for(unsigned e=0;e<experts;++e){auto& m=matrices[e];auto& h=host[e];unsigned k=3+e%2;
        m.inputs=inputs;m.outputs=outputs;m.padded_inputs=c.padded_inputs;m.padded_outputs=c.padded_outputs;
        m.rate_x2=2*k;m.codebook=e%2?Codebook::mul1:Codebook::mcg;m.identity=identity();m.global_expert=e+c.local_begin;
        m.transform=transformed?Transform::canonical_weight_h128_v1:Transform::none;
        h.words.resize(size_t(c.padded_inputs/16)*(c.padded_outputs/16)*8*k);for(auto& w:h.words)w=random();
        packed.emplace_back(new Device<uint32_t>(h.words));m.packed=packed.back()->ptr;m.packed_bytes=h.words.size()*4;
        h.suh.resize(c.padded_inputs);h.svh.resize(c.padded_outputs);
        for(unsigned i=0;i<c.padded_inputs;++i)h.suh[i]=i%3?1.f:-1.f;
        for(unsigned i=0;i<c.padded_outputs;++i)h.svh[i]=i%3?0.5f:-0.5f;
        scales.emplace_back(new Device<float>(h.suh));m.suh=scales.back()->ptr;
        scales.emplace_back(new Device<float>(h.svh));m.svh=scales.back()->ptr;
        if(e%2){h.bias.resize(outputs);for(unsigned i=0;i<outputs;++i)h.bias[i]=float(int(i%7)-3)/32;
            scales.emplace_back(new Device<float>(h.bias));m.bias=scales.back()->ptr;}
        materialize(h,m);
    }
    std::vector<Route> routes(c.routes);std::vector<unsigned> histogram(c.global_experts,0);
    for(unsigned i=0;i<c.routes;++i){unsigned expert=routing==0?1:routing==1?1+i%experts:
        routing==2?1+random()%experts:routing==3?0:routing==4?(i%2?0:1+i%experts):1;
        routes[i]={i/topk,i%topk,expert,1.f/topk};++histogram[expert];}
    std::vector<float> x(size_t(tokens)*inputs);for(auto& a:x)a=float(int(random()%31)-15)/16;
    auto p=plan(c,matrices.data());auto s=schedule(p,routes.data(),tokens,topk,experts);
    Device<uint8_t> arena(p.arena_bytes);Device<float> dx(x),dy(size_t(tokens)*outputs+32);Bound b{};
    HIP(bind(p,arena.ptr,arena.count,nullptr,b));HIP(hipDeviceSynchronize());
    for(auto mode:{OperatorMode::materialized_weight_fp32_fma_v1,OperatorMode::fused_activation_transform}){
        if(mode==OperatorMode::materialized_weight_fp32_fma_v1&&!enable_reference)continue;
        std::vector<float> poisoned(dy.count,std::numeric_limits<float>::quiet_NaN());dy.put(poisoned);
        HIP(run(b,s,dx.ptr,dy.ptr,mode,use_mfma?Compute::mfma_prefill:Compute::wave64_gemv));
        auto got=dy.get();for(size_t i=size_t(tokens)*outputs;i<got.size();++i)check(std::isnan(got[i]),"output tail guard");
        got.resize(size_t(tokens)*outputs);auto expected=reference(c,matrices,host,routes,x,mode);
        compare(got,expected,use_mfma?0.03f:0.0001f,use_mfma?0.001f:0.00001f,"full operator");
        // Deterministic mode must repeat bit-for-bit, independent of route distribution.
        HIP(run(b,s,dx.ptr,dy.ptr,mode,use_mfma?Compute::mfma_prefill:Compute::wave64_gemv));auto again=dy.get();again.resize(got.size());
        if(reduction==Reduction::deterministic)check(std::memcmp(got.data(),again.data(),got.size()*4)==0,"fixed-order repeated output");
    }
    for(unsigned e=0;e<experts;++e)if(s.counts[e])check(b.path_launches[unsigned(s.paths[e])]>0,"named dispatch path executed");
    for(unsigned i=0;i<b.path_launches.size();++i)total_path_launches[i]+=b.path_launches[i];
    // API rejection before any output write: changed identity, capacity, bad mapping,
    // input/output aliases, undersized pointer extent and actual capture-time refusal.
    std::vector<float> guards(dy.count,123456.f);
    auto refusal=[&](Bound& binding,const Schedule& routing,const float* in,float* out){
        dy.put(guards);
        check(run(binding,routing,in,out,OperatorMode::fused_activation_transform,Compute::wave64_gemv)!=hipSuccess,"preflight refusal");
        check(dy.get()==guards,"refusal preserves every output guard");
    };
    auto bad=b;bad.plan.matrices[0].packed_bytes++;refusal(bad,s,dx.ptr,dy.ptr);
    bad=b;bad.partial++;refusal(bad,s,dx.ptr,dy.ptr);
    auto ss=s;ss.routes[0].local_expert=experts;refusal(b,ss,dx.ptr,dy.ptr);
    refusal(b,s,dy.ptr,dy.ptr);
    refusal(b,s,dx.ptr,dy.ptr+dy.count-1);
    ss=s;ss.tokens++;refusal(b,ss,dx.ptr,dy.ptr);
    // The real HIP capture state must refuse, regardless of Features.capture.
    hipStream_t capture_stream;HIP(hipStreamCreate(&capture_stream));Bound capture_bound{};
    HIP(bind(p,arena.ptr,arena.count,capture_stream,capture_bound));HIP(hipStreamSynchronize(capture_stream));
    dy.put(guards);HIP(hipStreamBeginCapture(capture_stream,hipStreamCaptureModeThreadLocal));
    auto captured=run(capture_bound,s,dx.ptr,dy.ptr,OperatorMode::fused_activation_transform,Compute::wave64_gemv);
    check(captured==hipErrorStreamCaptureUnsupported,"actual capture refusal");
    hipGraph_t graph=nullptr;HIP(hipStreamEndCapture(capture_stream,&graph));if(graph)HIP(hipGraphDestroy(graph));
    HIP(hipStreamDestroy(capture_stream));check(dy.get()==guards,"capture refusal preserves guards");
    if(benchmark){
        hipEvent_t start,stop;HIP(hipEventCreate(&start));HIP(hipEventCreate(&stop));
        std::vector<float> elapsed;
        for(unsigned rep=0;rep<7;++rep){HIP(hipEventRecord(start));
            for(unsigned i=0;i<20;++i)HIP(run(b,s,dx.ptr,dy.ptr,OperatorMode::fused_activation_transform,use_mfma?Compute::mfma_prefill:Compute::wave64_gemv));
            HIP(hipEventRecord(stop));HIP(hipEventSynchronize(stop));float ms=0;HIP(hipEventElapsedTime(&ms,start,stop));elapsed.push_back(ms/20);}
        HIP(hipEventDestroy(start));HIP(hipEventDestroy(stop));
        std::cout<<"{\"proposition\":\"routing_microbenchmark\",\"category\":\"operator_latency_observation\",\"units\":\"ms\",\"direction\":\"lower\",\"routing\":\""<<(routing==0?"concentrated":routing==1?"spread":"random")<<"\",\"policy\":\""<<(policy==Policy::unified?"unified":"grouped")<<"\",\"reduction\":\""<<(reduction==Reduction::deterministic?"deterministic":"atomic")<<"\",\"compute\":\""<<(use_mfma?"mfma":"gemv")<<"\",\"tokens\":"<<tokens<<",\"topk\":"<<topk<<",\"inputs\":"<<inputs<<",\"outputs\":"<<outputs<<",\"scratch_bytes\":"<<p.arena_bytes<<",\"route_histogram\":[";
        for(unsigned i=0;i<histogram.size();++i)std::cout<<(i?",":"")<<histogram[i];std::cout<<"],\"raw_vector\":[";
        for(unsigned i=0;i<elapsed.size();++i)std::cout<<(i?",":"")<<elapsed[i];std::cout<<"]}\n";
    }
}
void fixture_case(const std::string& path,const std::string& expected_digest,const std::string& output_path,bool host_only=false){
    std::ifstream stream(path,std::ios::binary);if(!stream)throw std::runtime_error("missing derived canonical fixture");
    char magic[8],digest[65]{};Identity ids{};uint32_t header[9];stream.read(magic,8);stream.read(digest,64);
    stream.read(ids.model.data(),64);stream.read(ids.source.data(),64);stream.read(ids.role.data(),48);
    std::memcpy(ids.tensor.data(),digest,64);
    check(digest_valid(ids.model)&&digest_valid(ids.source)&&digest_valid(ids.tensor)&&role_valid(ids.role),"fixture source/model/role identity");
    stream.read(reinterpret_cast<char*>(header),sizeof(header));
    check(std::string(magic,8)=="EXL3GPU1"&&std::string(digest)==expected_digest,"canonical fixture digest binding");
    unsigned I=header[0],O=header[1],PI=header[2],PO=header[3],K=header[4],CB=header[5],T=header[6],bias=header[7],words=header[8];
    check(I&&O&&I<=PI&&O<=PO&&PI%128==0&&PO%128==0&&size_t(PI)*PO<=1048576&&K>=1&&K<=8&&(CB==1||CB==2)&&T>0&&T<=64&&bias<=1&&words==size_t(PI)*PO*K/32,"fixture header capacity");
    auto read=[&](auto& v){stream.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(v[0]));if(!stream)throw std::runtime_error("truncated fixture");};
    HostMatrix h;h.words.resize(words);h.suh.resize(PI);h.svh.resize(PO);if(bias)h.bias.resize(O);
    read(h.words);read(h.suh);read(h.svh);read(h.bias);std::vector<float>x(size_t(T)*I),golden(size_t(T)*O);read(x);read(golden);
    check(stream.peek()==std::char_traits<char>::eof(),"fixture trailing data refusal");
    for(const auto* values:{&h.suh,&h.svh,&h.bias,&x,&golden})for(float value:*values)check(std::isfinite(value),"fixture finite values");
    Matrix m;
    m.inputs=I;m.outputs=O;m.padded_inputs=PI;m.padded_outputs=PO;m.rate_x2=2*K;m.codebook=Codebook(CB);
    m.transform=Transform::canonical_weight_h128_v1;m.packed_bytes=words*4;m.identity=ids;
    std::copy(expected_digest.begin(),expected_digest.end(),m.identity.tensor.begin());
    Config c;c.tokens=T;c.topk=1;c.routes=T;c.experts=1;c.global_experts=1;c.inputs=I;c.outputs=O;c.padded_inputs=PI;c.padded_outputs=PO;
    c.prefill_rows=T;c.scheduler_slices=1;c.materialized_reference=true;
    materialize(h,m);std::vector<Matrix> matrices{m};std::vector<HostMatrix> hosts{h};
    std::vector<Route> routes(T);for(unsigned t=0;t<T;++t)routes[t]={t,0,0,1.f};
    auto cpu=reference(c,matrices,hosts,routes,x,OperatorMode::materialized_weight_fp32_fma_v1);
    compare(cpu,golden,0,0,"independent C++ vs canonical full operator fixture");
    if(host_only){std::cout<<"{\"proposition\":\"host_canonical_fixture_materialized_weight_fp32_fma_v1\",\"artifact_sha256\":\""<<digest<<"\",\"status\":\"pass\",\"gpu_executed\":false}\n";return;}
    Device<uint32_t> packed(h.words);Device<float> suh(h.suh),svh(h.svh);std::unique_ptr<Device<float>> dbias;
    m.packed=packed.ptr;m.suh=suh.ptr;m.svh=svh.ptr;if(bias){dbias.reset(new Device<float>(h.bias));m.bias=dbias->ptr;}
    auto p=plan(c,&m);auto sched=schedule(p,routes.data(),T,1,1);Device<uint8_t> arena(p.arena_bytes);Device<float> dx(x),dy(size_t(T)*O);Bound b;
    HIP(bind(p,arena.ptr,arena.count,nullptr,b));
    for(auto compute:{Compute::wave64_gemv,Compute::mfma_prefill}){
        HIP(run(b,sched,dx.ptr,dy.ptr,OperatorMode::materialized_weight_fp32_fma_v1,compute));auto got=dy.get();
        compare(got,golden,compute==Compute::wave64_gemv?0.f:0.00001f,compute==Compute::wave64_gemv?0.f:0.0001f,"canonical GPU full operator fixture");
        auto name=compute==Compute::wave64_gemv?"gemv":"mfma";
        std::ofstream out(output_path+"."+name+".f32",std::ios::binary);out.write(reinterpret_cast<char*>(got.data()),got.size()*4);if(!out)throw std::runtime_error("cannot persist full output");
        std::cout<<"{\"proposition\":\"materialized_weight_fp32_fma_v1\",\"fixture_sha256\":\""<<digest<<"\",\"codebook\":"<<CB<<",\"K\":"<<K<<",\"compute\":\""<<name<<"\",\"status\":\"pass\"}\n";
    }
    HIP(run(b,sched,dx.ptr,dy.ptr,OperatorMode::fused_activation_transform,Compute::wave64_gemv));auto fused=dy.get();
    auto cpu_fused=reference(c,matrices,hosts,routes,x,OperatorMode::fused_activation_transform);compare(fused,cpu_fused,0,0,"native packed fused full operator");
    float error=0;for(unsigned i=0;i<fused.size();++i)error=std::max(error,std::fabs(fused[i]-golden[i]));
    std::ofstream out(output_path+".fused.f32",std::ios::binary);out.write(reinterpret_cast<char*>(fused.data()),fused.size()*4);if(!out)throw std::runtime_error("cannot persist fused output");
    std::cout<<"{\"proposition\":\"fused_activation_transform\",\"fixture_sha256\":\""<<digest<<"\",\"status\":\"pass_against_fused_reference\",\"materialized_max_abs_difference\":"<<error<<"}\n";
}
void require_claim(){
    const char* fd_text=std::getenv("EXL3_PHYSICAL_CLAIM_FD");
    if(!fd_text)throw std::runtime_error("launch through claimed_run.py; no inherited physical claim");
    int fd=std::stoi(fd_text);struct stat inherited{},canonical{};
    const char* root=std::getenv("ORCHESTRATOR_TMP_DIR");
    if(!root)root=std::getenv("ORCHESTRATOR_PATHS_TMP_DIR");
    std::string path=std::string(root?root:"/mnt/raid0/llm/tmp")+"/gpu_device.mi210_0.lock";
    int other=open(path.c_str(),O_RDWR|O_CLOEXEC);
    if(other<0||fstat(fd,&inherited)||fstat(other,&canonical)||inherited.st_dev!=canonical.st_dev||inherited.st_ino!=canonical.st_ino)
        throw std::runtime_error("claim descriptor does not identify provider lock");
    int result=flock(other,LOCK_EX|LOCK_NB);int error=errno;
    if(result==0){flock(other,LOCK_UN);close(other);throw std::runtime_error("physical claim is not held");}
    close(other);if(error!=EWOULDBLOCK&&error!=EAGAIN)throw std::runtime_error("cannot verify claim exclusion");
}
int main(int argc,char** argv){try{
    if(argc==4&&std::string(argv[1])=="--host-fixture"){fixture_case(argv[2],argv[3],"",true);return 0;}
    require_claim();
    int device=0;HIP(hipGetDevice(&device));HIP(device_contract(device));
    hipDeviceProp_t prop{};HIP(hipGetDeviceProperties(&prop,device));
    std::cout<<"{\"device\":\""<<prop.gcnArchName<<"\",\"wavefront\":"<<prop.warpSize<<"}\n";
    if(argc>1 && std::string(argv[1])=="--fixture"){
        if(argc!=5)throw std::runtime_error("--fixture TRANSPORT CANONICAL_DIGEST OUTPUT_PREFIX");
        fixture_case(argv[2],argv[3],argv[4]);
    }else if(argc>1 && std::string(argv[1])=="--microbench"){
        for(auto policy:{Policy::unified,Policy::grouped})for(unsigned routing=0;routing<3;++routing)
            for(bool mfma:{false,true})operator_case(16,4,8,128,128,false,policy,{},Reduction::deterministic,routing,mfma,true,false);
    }else{
        probe_tests();
        for(auto policy:{Policy::unified,Policy::grouped})for(unsigned tokens:{1u,3u,4u,15u,16u,17u})
            for(unsigned routing=0;routing<5;++routing)operator_case(tokens,2,4,31,33,false,policy,{},Reduction::deterministic,routing,false);
        for(bool bc:{false,true})for(bool ext:{false,true})operator_case(4,2,4,15,17,false,Policy::grouped,{bc,ext,false},Reduction::deterministic,1,false);
        for(unsigned experts:{1u,3u,4u,8u,256u})operator_case(4,1,experts,16,16,false,Policy::grouped,{},Reduction::deterministic,1,false);
        for(bool mfma:{false,true})for(unsigned tokens:{1u,15u,16u,17u})
            operator_case(tokens,2,4,127,129,true,Policy::grouped,{},Reduction::deterministic,2,mfma);
        operator_case(16,4,8,32,48,false,Policy::unified,{},Reduction::atomic,2,false);
        operator_case(32,2,8,32,48,false,Policy::grouped,{},Reduction::deterministic,1,false,false,false);
        for(unsigned p=0;p<5;++p)check(total_path_launches[p]>0,"every named policy/fallback launched");
        std::cout<<"{\"proposition\":\"dispatch_named_path_coverage\",\"launch_counts\":{";
        for(unsigned p=0;p<5;++p)std::cout<<(p?",":"")<<"\""<<path_name(Path(p))<<"\":"<<total_path_launches[p];std::cout<<"}}\n";
    }
    HIP(hipDeviceSynchronize());HIP(hipDeviceReset());
    std::cout<<"{\"proposition\":\"standalone_gfx90a_synthetic_operators\",\"status\":\"pass\",\"assertions\":"<<assertions<<",\"real_fixture_parity\":\""<<((argc>1&&std::string(argv[1])=="--fixture")?"single_fixture_pass":"not_run")<<"\",\"full_model_inference\":false}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
