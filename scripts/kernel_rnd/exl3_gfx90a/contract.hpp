#pragma once
// Standalone experimental EXL3 operator. No ggml, model loader, or serving hooks.
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace exl3 {
constexpr unsigned max_experts = 256, max_routes = 1024;
constexpr unsigned abi_version = 1;
enum class Codebook : uint8_t { mcg = 1, mul1 = 2 };
enum class Layout : uint8_t { native_input_major_v1 = 1 };
enum class Transform : uint8_t { none = 0, canonical_weight_h128_v1 = 1 };
enum class Policy : uint8_t { unified, grouped };
enum class Reduction : uint8_t { deterministic, atomic };
enum class Path : uint8_t {
    unified, grouped, batched_fallback, small_row_fallback, single_expert_fallback,
    unsupported_rate, unsupported_codebook, unsupported_layout, capacity_overflow,
    invalid_domain, capture_fallback_refused, empty
};
inline const char* path_name(Path p) {
    switch(p) {
#define EXL3_PATH(x) case Path::x: return #x
        EXL3_PATH(unified); EXL3_PATH(grouped); EXL3_PATH(batched_fallback);
        EXL3_PATH(small_row_fallback); EXL3_PATH(single_expert_fallback);
        EXL3_PATH(unsupported_rate); EXL3_PATH(unsupported_codebook);
        EXL3_PATH(unsupported_layout); EXL3_PATH(capacity_overflow);
        EXL3_PATH(invalid_domain); EXL3_PATH(capture_fallback_refused); EXL3_PATH(empty);
#undef EXL3_PATH
    }
    return "invalid";
}
struct Features { bool bc = true, extension = true, capture = false; };
// This is the only eligibility/dispatch truth table, shared by planning and running.
inline Path capability(uint16_t rate_x2, Codebook cb, Layout layout, Policy policy,
                       Features f, unsigned tokens, unsigned routed_rows, unsigned bucket_experts) {
    if (rate_x2 < 2 || rate_x2 > 16 || (rate_x2 & 1)) return Path::unsupported_rate;
    if (cb != Codebook::mcg && cb != Codebook::mul1) return Path::unsupported_codebook;
    if (layout != Layout::native_input_major_v1) return Path::unsupported_layout;
    Path p = !routed_rows ? Path::empty : policy == Policy::unified ? Path::unified :
             !f.bc || !f.extension ? Path::batched_fallback :
             tokens < 4 || routed_rows < 4 ? Path::small_row_fallback :
             bucket_experts < 4 ? Path::single_expert_fallback : Path::grouped;
    if (f.capture && p != Path::unified && p != Path::grouped && p != Path::empty)
        return Path::capture_fallback_refused;
    return p;
}
inline bool executable(Path p) { return p <= Path::single_expert_fallback || p == Path::empty; }
struct Identity {
    std::array<char,65> model{}, source{}, tensor{}; // lowercase SHA-256 strings
    std::array<char,48> role{};
};
inline bool role_valid(const std::array<char,48>& role) {
    if(role.back()!=0)return false;
    for(const char* allowed:{"q","k","v","o","gate","up","down","lm_head"})
        if(std::strcmp(role.data(),allowed)==0)return true;
    return false;
}
inline bool digest_valid(const std::array<char,65>& d) {
    if(d[64]) return false;
    for(unsigned i=0;i<64;++i) if(!((d[i]>='0'&&d[i]<='9')||(d[i]>='a'&&d[i]<='f'))) return false;
    return true;
}
struct Matrix {
    const uint32_t* packed = nullptr;
    const float* suh = nullptr;
    const float* svh = nullptr;
    const float* bias = nullptr;
    unsigned inputs=0, outputs=0, padded_inputs=0, padded_outputs=0;
    uint16_t rate_x2=0; // exact per-matrix K in half-bit units; odd v1 rates unsupported
    Codebook codebook=Codebook::mcg;
    Layout layout=Layout::native_input_major_v1;
    Transform transform=Transform::none;
    size_t packed_bytes=0;
    Identity identity{};
    unsigned global_expert=0;
};
struct Config {
    unsigned tokens=0, topk=0, routes=0, experts=0;
    unsigned global_experts=0, local_begin=0;
    unsigned inputs=0, outputs=0, padded_inputs=0, padded_outputs=0;
    unsigned prefill_rows=0, scheduler_slices=0;
    size_t max_arena_bytes=512ULL*1024*1024;
    bool materialized_reference=false;
    Policy policy=Policy::unified;
    Reduction reduction=Reduction::deterministic;
    Features features{};
};
struct Route { uint32_t token=0, routing_k=0, global_expert=0; float weight=1.f; };
struct DeviceRoute { uint32_t token, routing_k; int32_t local_expert; float weight; };
struct Plan {
    Config config{};
    std::array<Matrix,max_experts> matrices{};
    size_t arena_bytes=0, matrices_offset=0, routes_offset=0, counts_offset=0,
           order_offset=0, raw_offset=0, transformed_offset=0, partial_offset=0, activations_offset=0, output_transform_offset=0;
    uint64_t identity=0;
};
inline size_t checked_mul(size_t a,size_t b) {
    if(b && a>std::numeric_limits<size_t>::max()/b) throw std::invalid_argument("capacity_overflow");
    return a*b;
}
inline size_t region(size_t& cursor,size_t count,size_t width) {
    if(cursor>std::numeric_limits<size_t>::max()-255) throw std::invalid_argument("capacity_overflow");
    cursor=(cursor+255)&~size_t(255); auto start=cursor; auto bytes=checked_mul(count,width);
    if(bytes>std::numeric_limits<size_t>::max()-cursor) throw std::invalid_argument("capacity_overflow");
    cursor+=bytes; return start;
}
inline uint64_t identity_hash(const Plan& p) {
    // Runtime identity, not a replacement for the content-addressed source digests.
    uint64_t h=1469598103934665603ULL;
    auto feed=[&](uint64_t v){ for(int i=0;i<8;++i){h^=(v>>(8*i))&255;h*=1099511628211ULL;} };
    const auto& c=p.config;
    if(c.experts>max_experts)return 0; // fail closed before inspecting a mutated fixed array
    for(auto v:{c.tokens,c.topk,c.routes,c.experts,c.global_experts,c.local_begin,c.inputs,c.outputs,
                c.padded_inputs,c.padded_outputs,c.prefill_rows,c.scheduler_slices}) feed(v);
    feed(unsigned(c.policy));feed(unsigned(c.reduction));feed(c.features.bc);feed(c.features.extension);
    feed(c.features.capture);feed(c.max_arena_bytes);feed(c.materialized_reference);feed(p.arena_bytes);feed(abi_version);
    for(size_t v:{p.matrices_offset,p.routes_offset,p.counts_offset,p.order_offset,p.raw_offset,p.transformed_offset,p.partial_offset,p.activations_offset,p.output_transform_offset})feed(v);
    for(unsigned i=0;i<c.experts;++i){const auto& m=p.matrices[i];feed(m.rate_x2);feed(unsigned(m.codebook));
        feed(unsigned(m.layout));feed(unsigned(m.transform));feed(m.global_expert);
        feed(uintptr_t(m.packed));feed(uintptr_t(m.suh));feed(uintptr_t(m.svh));feed(uintptr_t(m.bias));
        feed(m.packed_bytes);feed(m.inputs);feed(m.outputs);feed(m.padded_inputs);feed(m.padded_outputs);
        for(const auto* d:{&m.identity.model,&m.identity.source,&m.identity.tensor}) for(char ch:*d) feed(ch);
        for(char ch:m.identity.role)feed(ch);
    } return h;
}
inline Plan plan(const Config& c,const Matrix* matrices) {
    if(!matrices || !c.tokens || !c.topk || c.topk>c.global_experts || !c.experts || c.experts>max_experts ||
       c.routes>max_routes || checked_mul(c.tokens,c.topk)>c.routes ||
       !c.inputs || !c.outputs || c.padded_inputs<c.inputs || c.padded_outputs<c.outputs ||
       c.padded_inputs%16 || c.padded_outputs%16 || c.local_begin>c.global_experts ||
       c.experts>c.global_experts-c.local_begin || !c.scheduler_slices || !c.prefill_rows ||
       c.prefill_rows>c.tokens) throw std::invalid_argument("invalid capacity or expert domain");
    if(c.policy!=Policy::unified&&c.policy!=Policy::grouped)throw std::invalid_argument("unknown policy");
    if(c.reduction!=Reduction::deterministic&&c.reduction!=Reduction::atomic)throw std::invalid_argument("unknown reduction");
    Plan p{};p.config=c;
    for(unsigned e=0;e<c.experts;++e){ const auto& m=matrices[e];
        auto cap=capability(m.rate_x2,m.codebook,m.layout,c.policy,c.features,c.tokens,c.routes,c.experts);
        if(!executable(cap))throw std::invalid_argument(path_name(cap));
        if(!m.packed || m.inputs!=c.inputs || m.outputs!=c.outputs || m.padded_inputs!=c.padded_inputs ||
           m.padded_outputs!=c.padded_outputs || m.global_expert!=c.local_begin+e ||
           !digest_valid(m.identity.model)||!digest_valid(m.identity.source)||!digest_valid(m.identity.tensor)||
           !role_valid(m.identity.role))
            throw std::invalid_argument("ambiguous matrix identity or source mapping");
        size_t expected=checked_mul(checked_mul(c.padded_inputs/16,c.padded_outputs/16),16*m.rate_x2);
        if(m.packed_bytes!=expected)throw std::invalid_argument("packed byte extent mismatch");
        if(m.transform!=Transform::none && m.transform!=Transform::canonical_weight_h128_v1)
            throw std::invalid_argument("unknown transform");
        if(m.transform==Transform::canonical_weight_h128_v1 &&
           (!m.suh||!m.svh||c.padded_inputs%128||c.padded_outputs%128))
            throw std::invalid_argument("invalid Hadamard geometry");
        p.matrices[e]=m;
    }
    size_t cursor=0;
    p.matrices_offset=region(cursor,c.experts,sizeof(Matrix));
    p.routes_offset=region(cursor,c.routes,sizeof(DeviceRoute));
    p.counts_offset=region(cursor,c.experts+1,sizeof(uint32_t)); // sentinel ONLY in counts/sort
    p.order_offset=region(cursor,c.routes,sizeof(uint32_t));
    size_t weight_values=c.materialized_reference?checked_mul(c.experts,checked_mul(c.padded_inputs,c.padded_outputs)):0;
    p.raw_offset=region(cursor,weight_values,sizeof(float));
    p.transformed_offset=region(cursor,weight_values,sizeof(float));
    p.partial_offset=region(cursor,checked_mul(c.routes,c.outputs),sizeof(float));
    p.activations_offset=region(cursor,checked_mul(c.routes,c.padded_inputs),sizeof(float));
    p.output_transform_offset=region(cursor,checked_mul(c.routes,c.padded_outputs),sizeof(float));
    if(cursor>c.max_arena_bytes)throw std::invalid_argument("arena_byte_cap");
    p.arena_bytes=cursor;p.identity=identity_hash(p);return p;
}
struct Schedule {
    std::array<DeviceRoute,max_routes> routes{};
    std::array<uint32_t,max_experts+1> counts{};
    std::array<uint32_t,max_routes> order{};
    std::array<Path,max_experts> paths{};
    unsigned size=0, tokens=0;
};
inline Schedule schedule(const Plan& p,const Route* routes,unsigned tokens,unsigned topk,unsigned slices) {
    const auto& c=p.config;
    if(tokens>c.tokens || topk!=c.topk || checked_mul(tokens,topk)>c.routes ||
       slices>c.scheduler_slices || (tokens&&!slices))throw std::invalid_argument("capacity_overflow or exhausted_scheduler_slices");
    if(tokens&&!routes)throw std::invalid_argument("missing routes");
    Schedule s{};s.size=tokens*topk;s.tokens=tokens;
    for(unsigned i=0;i<s.size;++i){const auto& r=routes[i];
        // Input order is a contract: token-major, then routing-k. Never silently sort it away.
        if(r.token!=i/topk||r.routing_k!=i%topk||r.global_expert>=c.global_experts||!std::isfinite(r.weight))
            throw std::invalid_argument("invalid route or routing-k order");
        int local=r.global_expert>=c.local_begin&&r.global_expert<c.local_begin+c.experts?
            int(r.global_expert-c.local_begin):-1;
        s.routes[i]={r.token,r.routing_k,local,r.weight};++s.counts[local<0?c.experts:unsigned(local)];
    }
    unsigned cursor=0;
    for(unsigned e=0;e<=c.experts;++e)
        for(unsigned i=0;i<s.size;++i)if((s.routes[i].local_expert<0?c.experts:unsigned(s.routes[i].local_expert))==e)s.order[cursor++]=i;
    unsigned groups=0;
    for(unsigned e=0;e<c.experts;++e)if(s.counts[e]) {
        unsigned bucket=0;
        for(unsigned j=0;j<c.experts;++j)if(s.counts[j]&&p.matrices[j].rate_x2==p.matrices[e].rate_x2&&p.matrices[j].codebook==p.matrices[e].codebook)++bucket;
        const auto& m=p.matrices[e];
        s.paths[e]=capability(m.rate_x2,m.codebook,m.layout,c.policy,c.features,tokens,s.counts[e],bucket);
        if(!executable(s.paths[e]))throw std::invalid_argument(path_name(s.paths[e]));
        ++groups;
    } else s.paths[e]=Path::empty;
    // One slice per active expert for grouped launch. Unified has exactly one nonempty slice.
    if((c.policy==Policy::grouped?groups:unsigned(s.size!=0))>slices)
        throw std::invalid_argument("exhausted_scheduler_slices");
    return s;
}
} // namespace exl3
