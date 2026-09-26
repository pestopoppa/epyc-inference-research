#include "exl3_cpu.h"
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <stdexcept>

namespace exl3 {
namespace {
constexpr float inv = 0.00676727294921875f;
float half(uint16_t u) { _Float16 h; std::memcpy(&h, &u, 2); return float(h); }
float round_half(float x) { return float((_Float16)x); }
size_t offset(const Matrix& w, size_t kt, size_t nt) {
    size_t nk = w.padded_k / 16, nn = w.padded_n / 16;
    return (w.layout == Layout::native ? kt * nn + nt :
            (nt / 8) * nk * 8 + kt * 8 + nt % 8) * 16 * w.bits;
}
unsigned position(unsigned i) {
    unsigned t = i / 8, j = i % 8;
    return ((t % 4) * 2 + (j & 1) + 8 * ((j >> 1) & 1)) * 16 + t / 4 + 8 * (j >> 2);
}
// Independent scalar bit-window reference: reads only the exact tile allocation.
uint16_t state(const uint16_t* p, unsigned bits, unsigned i) {
    unsigned s = 0, length = bits * 256;
    for (unsigned b = 0; b < 16; ++b) {
        unsigned bit = ((i + 1) * bits + length - 16 + b) % length;
        s = (s << 1) | ((p[(bit / 16) ^ 1] >> (15 - bit % 16)) & 1);
    }
    return uint16_t(s);
}
void had(float* p, size_t n) {
    constexpr float scale = 0.08838834764831845f;
    for (size_t base = 0; base < n; base += 128) {
        for (size_t d = 1; d < 128; d *= 2)
            for (size_t j = 0; j < 128; j += 2*d)
                for (size_t i = 0; i < d; ++i) {
                    float a = p[base+j+i], b = p[base+j+i+d];
                    p[base+j+i] = a+b; p[base+j+i+d] = a-b;
                }
        for (size_t i = 0; i < 128; ++i) p[base+i] *= scale;
    }
}
// Materialized contract fixes ascending-index FMA order before each fp16 round.
void had_materialized(float* p, size_t n) {
    constexpr float scale = 0.08838834764831845f;
    float block[128];
    for (size_t base=0;base<n;base+=128) {
        for (unsigned i=0;i<128;++i) {
            float acc=0;
            for(unsigned j=0;j<128;++j)
                acc=std::fma((__builtin_parity(i&j)?-scale:scale),p[base+j],acc);
            block[i]=acc;
        }
        std::copy_n(block,128,p+base);
    }
}
void check_io(const Matrix& w, const float* x, size_t rows, size_t xs, float* y, size_t ys) {
    validate(w);
    if (!x || !y || !rows || xs < w.k || ys < w.n || rows > SIZE_MAX / xs || rows > SIZE_MAX / ys)
        throw std::invalid_argument("invalid I/O extent");
    for(size_t r=0;r<rows;++r) {
        double bound=0;
        for(size_t k=0;k<w.k;++k) {
            if(!std::isfinite(x[r*xs+k]))throw std::invalid_argument("nonfinite activation");
            bound+=std::abs(double(x[r*xs+k])*double(w.suh[k]));
        }
        if(bound>double(std::numeric_limits<float>::max())/128)throw std::invalid_argument("activation transform overflow risk");
    }
}
void require(ISA isa) { if (std::fegetround()!=FE_TONEAREST)throw std::invalid_argument("RNE rounding required"); if (!supported(isa)) throw std::invalid_argument("unsupported ISA"); }
#define BW __attribute__((target("avx512f,avx512bw,avx512vl,f16c")))
#define VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,f16c")))
#define VBMI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,f16c")))
// Two logical words form a 32-bit window. Tail wrap is materialized explicitly.
BW __m512i states16(const uint16_t* p, unsigned bits, unsigned begin) {
    alignas(64) uint32_t windows[128];
    alignas(64) int indices[16], shifts[16];
    unsigned words = 16 * bits;
    for (unsigned j=0;j<words;++j) windows[j] = (uint32_t(p[j^1]) << 16) | p[((j+1)%words)^1];
    for (unsigned j=0;j<16;++j) {
        unsigned start = ((begin+j+1)*bits + 256*bits - 16) % (256*bits);
        indices[j] = start/16; shifts[j] = 16-start%16;
    }
    auto v = _mm512_i32gather_epi32(_mm512_load_si512(indices), windows, 4);
    return _mm512_and_si512(_mm512_srlv_epi32(v,_mm512_load_si512(shifts)),_mm512_set1_epi32(65535));
}
BW __m512 values16(__m512i s, Codebook cb) {
    if (cb == Codebook::mcg) {
        auto p = _mm512_xor_si512(_mm512_and_si512(_mm512_mullo_epi32(s,_mm512_set1_epi32(0xcbac1fed)),
                           _mm512_set1_epi32(0x8fff8fff)),_mm512_set1_epi32(0x3b603b60));
        auto lo = _mm512_cvtepi32_epi16(p), hi = _mm512_cvtepi32_epi16(_mm512_srli_epi32(p,16));
        auto sum = _mm512_add_ps(_mm512_cvtph_ps(lo), _mm512_cvtph_ps(hi));
        return _mm512_cvtph_ps(_mm512_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
    }
    auto p = _mm512_mullo_epi32(s,_mm512_set1_epi32(0x83dcd12d));
    auto mask = _mm512_set1_epi32(255);
    auto sum = _mm512_add_epi32(_mm512_and_si512(p,mask),_mm512_and_si512(_mm512_srli_epi32(p,8),mask));
    sum = _mm512_add_epi32(sum,_mm512_and_si512(_mm512_srli_epi32(p,16),mask));
    sum = _mm512_add_epi32(sum,_mm512_srli_epi32(p,24));
    auto h = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(_mm512_add_epi32(sum,_mm512_set1_epi32(0x6400))));
    auto v = _mm512_fmadd_ps(h,_mm512_set1_ps(inv),_mm512_set1_ps(half(0xc931)));
    return _mm512_cvtph_ps(_mm512_cvtps_ph(v,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
}
BW void vector_tile(const uint16_t* p, unsigned bits, Codebook cb, float* out) {
    alignas(64) float values[16];
    for (unsigned begin=0;begin<256;begin+=16) {
        _mm512_store_ps(values,values16(states16(p,bits,begin),cb));
        for (unsigned j=0;j<16;++j) out[position(begin+j)] = values[j];
    }
}
BW void tile_dot(const float* tile, const float* x, float* y) {
    auto acc=_mm512_loadu_ps(y);
    for(unsigned k=0;k<16;++k)
        acc=_mm512_add_ps(acc,_mm512_mul_ps(_mm512_set1_ps(x[k]),_mm512_loadu_ps(tile+k*16)));
    _mm512_storeu_ps(y,acc);
}
BW void products_bw(__m512i p, const int32_t* a, int32_t* out) {
    auto mask = _mm512_set1_epi32(255);
    auto sum = _mm512_add_epi32(_mm512_and_si512(p,mask),_mm512_and_si512(_mm512_srli_epi32(p,8),mask));
    sum = _mm512_add_epi32(sum,_mm512_and_si512(_mm512_srli_epi32(p,16),mask));
    sum = _mm512_add_epi32(sum,_mm512_srli_epi32(p,24));
    _mm512_store_si512(out,_mm512_mullo_epi32(_mm512_sub_epi32(sum,_mm512_set1_epi32(510)),_mm512_load_si512(a)));
}
VNNI void products_vnni(__m512i p, const int32_t* a, int32_t* out) {
    auto av = _mm512_load_si512(a);
    auto bytes = _mm512_mullo_epi32(_mm512_and_si512(av,_mm512_set1_epi32(255)),_mm512_set1_epi32(0x01010101));
    auto dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(),p,bytes);
    _mm512_store_si512(out,_mm512_sub_epi32(dot,_mm512_mullo_epi32(av,_mm512_set1_epi32(510))));
}
VBMI void products_vbmi(__m512i p, const int32_t* a, int32_t* out) {
    alignas(64) uint8_t idx[64];
    for (unsigned i=0;i<64;++i) idx[i] = (i/4)*4;
    auto av = _mm512_load_si512(a);
    auto bytes = _mm512_permutexvar_epi8(_mm512_load_si512(idx),av);
    auto dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(),p,bytes);
    _mm512_store_si512(out,_mm512_sub_epi32(dot,_mm512_mullo_epi32(av,_mm512_set1_epi32(510))));
}
BW void q8_tile(const uint16_t* p, unsigned bits, const int8_t* x, int64_t* y, ISA isa) {
    alignas(64) int32_t a[16], result[16];
    for (unsigned i=0;i<256;i+=16) {
        for (unsigned j=0;j<16;++j) a[j]=x[position(i+j)/16];
        auto prod = _mm512_mullo_epi32(states16(p,bits,i),_mm512_set1_epi32(0x83dcd12d));
        if (isa==ISA::vbmi) products_vbmi(prod,a,result);
        else if (isa==ISA::vnni) products_vnni(prod,a,result);
        else products_bw(prod,a,result);
        for (unsigned j=0;j<16;++j) y[position(i+j)%16]+=result[j];
    }
}
template<unsigned K> void run(const Matrix& w, const float* x, size_t rows, size_t xs, float* y, size_t ys, ISA isa, bool q8) {
    unsigned bits = K ? K : w.bits;
    for (size_t r=0;r<rows;++r) {
        std::vector<float> in(w.padded_k,0), out(w.padded_n,0);
        for(size_t k=0;k<w.k;++k) in[k]=x[r*xs+k]*w.suh[k];
        had(in.data(),w.padded_k);
        std::vector<int8_t> quant(w.padded_k);
        std::vector<int64_t> acc(w.padded_n,0);
        float scale=0;
        if(q8) {
            for(float v:in) scale=std::max(scale,std::abs(v));
            scale/=127;
            for(size_t k=0;k<w.padded_k;++k) quant[k]=scale ? int8_t(std::clamp(std::nearbyint(in[k]/scale),-127.f,127.f)) : 0;
        }
        for(size_t kt=0;kt<w.padded_k/16;++kt) for(size_t nt=0;nt<w.padded_n/16;++nt) {
            const auto* p=w.trellis.data()+offset(w,kt,nt);
            if(q8 && isa!=ISA::scalar) q8_tile(p,bits,quant.data()+kt*16,acc.data()+nt*16,isa);
            else if(q8) for(unsigned i=0;i<256;++i) {
                unsigned pos=position(i); uint32_t v=uint32_t(state(p,bits,i))*0x83dcd12dU;
                int sum=int(v&255)+int((v>>8)&255)+int((v>>16)&255)+int(v>>24)-510;
                acc[nt*16+pos%16]+=int64_t(sum)*quant[kt*16+pos/16];
            } else {
                float tile[256]; decode_tile(p,bits,w.codebook,isa,tile);
                if(isa==ISA::scalar) {
                    for(unsigned k=0;k<16;++k) for(unsigned n=0;n<16;++n) out[nt*16+n]+=in[kt*16+k]*tile[k*16+n];
                } else tile_dot(tile,in.data()+kt*16,out.data()+nt*16);
            }
        }
        if(q8) for(size_t n=0;n<w.padded_n;++n) out[n]=float(acc[n])*scale*inv;
        had(out.data(),w.padded_n);
        for(size_t n=0;n<w.n;++n) y[r*ys+n]=out[n]*w.svh[n]+(w.bias.empty()?0:w.bias[n]);
    }
}
using Run=void(*)(const Matrix&,const float*,size_t,size_t,float*,size_t,ISA,bool);
Run dispatch(unsigned k) {
    switch(k) {
#define CASE(K) case K:return run<K>;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    default:throw std::invalid_argument("K outside 1..8");
    }
}
}
bool supported(ISA isa) {
    __builtin_cpu_init();
    bool bw=__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("f16c");
    if(isa==ISA::scalar)return true;
    if(isa==ISA::avx512bw)return bw;
    bool vnni=bw&&__builtin_cpu_supports("avx512vnni");
    if(isa==ISA::vnni)return vnni;
    if(isa==ISA::vbmi)return vnni&&__builtin_cpu_supports("avx512vbmi");
    return false;
}
void validate(const Matrix& w) {
    if (std::fegetround()!=FE_TONEAREST)throw std::invalid_argument("RNE rounding required");
    if(w.bits<1||w.bits>8||!w.k||!w.n||w.k>w.padded_k||w.n>w.padded_n||w.padded_k%128||w.padded_n%128)
        throw std::invalid_argument("invalid geometry/K");
    if(w.padded_k > 1048576 || w.padded_n > 1048576) throw std::invalid_argument("extent exceeds experimental limit");
    if(w.trellis.size()!=w.padded_k*w.padded_n*w.bits/16 || w.suh.size()!=w.padded_k || w.svh.size()!=w.padded_n)
        throw std::invalid_argument("invalid payload extent");
    if(w.codebook!=Codebook::mul1&&w.codebook!=Codebook::mcg)throw std::invalid_argument("invalid codebook");
    if(w.layout!=Layout::native&&w.layout!=Layout::band8)throw std::invalid_argument("invalid layout");
    if(!w.bias.empty()&&w.bias.size()!=w.n)throw std::invalid_argument("invalid bias extent");
    for(float v:w.bias)if(!std::isfinite(v))throw std::invalid_argument("nonfinite bias");
    for(float v:w.suh)if(!std::isfinite(v))throw std::invalid_argument("nonfinite suh");
    for(float v:w.svh)if(!std::isfinite(v))throw std::invalid_argument("nonfinite svh");
}
void repack(Matrix& w, Layout layout) {
    validate(w); Matrix target=w; target.layout=layout; validate(target);
    for(size_t kt=0;kt<w.padded_k/16;++kt)for(size_t nt=0;nt<w.padded_n/16;++nt)
        std::copy_n(w.trellis.data()+offset(w,kt,nt),16*w.bits,target.trellis.data()+offset(target,kt,nt));
    w=std::move(target);
}
float codebook_value(uint16_t s,Codebook cb) {
    if(cb==Codebook::mcg){uint32_t p=(uint32_t(s)*0xcbac1fedU&0x8fff8fffU)^0x3b603b60U;return round_half(half(p&65535)+half(p>>16));}
    if(cb!=Codebook::mul1)throw std::invalid_argument("invalid codebook");
    uint32_t p=uint32_t(s)*0x83dcd12dU;
    unsigned sum=(p&255)+((p>>8)&255)+((p>>16)&255)+(p>>24);
    return round_half(std::fma(half(uint16_t(0x6400+sum)),inv,half(0xc931)));
}
void decode_tile(const uint16_t* p,unsigned bits,Codebook cb,ISA isa,float* out) {
    require(isa); if(!p||!out||bits<1||bits>8||(cb!=Codebook::mul1&&cb!=Codebook::mcg))throw std::invalid_argument("invalid tile");
    if(isa!=ISA::scalar)vector_tile(p,bits,cb,out);
    else for(unsigned i=0;i<256;++i)out[position(i)]=codebook_value(state(p,bits,i),cb);
}
void dense(const Matrix& w,const float* x,size_t rows,size_t xs,float* y,size_t ys,ISA isa) {
    if(rows>4)throw std::invalid_argument("use prefill provider for more than four rows");
    check_io(w,x,rows,xs,y,ys);require(isa);
    dispatch(w.bits)(w,x,rows,xs,y,ys,isa,false);
}
void mul1_q8(const Matrix& w,const float* x,size_t rows,size_t xs,float* y,size_t ys,ISA isa) {
    if(w.codebook!=Codebook::mul1||rows>4)throw std::invalid_argument("Q8 MUL1 requires MUL1 and 1..4 rows");
    check_io(w,x,rows,xs,y,ys);require(isa);
    dispatch(w.bits)(w,x,rows,xs,y,ys,isa,true);
}
void indexed(const std::vector<Matrix>& experts,const std::vector<Route>& routes,const float* x,size_t xrows,size_t xs,float* y,size_t ys,ISA isa,bool grouped,bool q8) {
    require(isa);
    if(!xrows||!xs||!ys||xrows>SIZE_MAX/xs||routes.size()>SIZE_MAX/ys)throw std::invalid_argument("route extent");
    for(auto r:routes) {
        if(r.expert>=experts.size()||r.input_row>=xrows)throw std::invalid_argument("expert ID/input row");
        const auto& first=experts[routes.front().expert];
        if(experts[r.expert].k!=first.k||experts[r.expert].n!=first.n)throw std::invalid_argument("mixed expert geometry");
        check_io(experts[r.expert],x+r.input_row*xs,1,xs,y,ys);
        if(q8&&experts[r.expert].codebook!=Codebook::mul1)throw std::invalid_argument("Q8 codebook"); }
    if(grouped)for(unsigned k=1;k<=8;++k) {
        Run kernel=dispatch(k);
        for(size_t i=0;i<routes.size();++i) { auto r=routes[i];auto& w=experts[r.expert];
            if(w.bits==k)kernel(w,x+r.input_row*xs,1,xs,y+i*ys,ys,isa,q8); }
    } else for(size_t i=0;i<routes.size();++i) { auto r=routes[i];run<0>(experts[r.expert],x+r.input_row*xs,1,xs,y+i*ys,ys,isa,q8); }
}
std::vector<float> reconstruct(const Matrix& w) {
    validate(w);std::vector<float> full(w.padded_k*w.padded_n),column(w.padded_k);
    for(size_t kt=0;kt<w.padded_k/16;++kt)for(size_t nt=0;nt<w.padded_n/16;++nt) {
        float tile[256];decode_tile(w.trellis.data()+offset(w,kt,nt),w.bits,w.codebook,ISA::scalar,tile);
        for(size_t k=0;k<16;++k)std::copy_n(tile+k*16,16,full.data()+(kt*16+k)*w.padded_n+nt*16);
    }
    for(size_t n=0;n<w.padded_n;++n) {
        for(size_t k=0;k<w.padded_k;++k)column[k]=full[k*w.padded_n+n];
        had_materialized(column.data(),w.padded_k);
        for(size_t k=0;k<w.padded_k;++k)full[k*w.padded_n+n]=round_half(round_half(column[k])*w.suh[k]);
    }
    std::vector<float> result(w.k*w.n);
    for(size_t k=0;k<w.k;++k) {
        float* row=full.data()+k*w.padded_n;had_materialized(row,w.padded_n);
        for(size_t n=0;n<w.n;++n)result[k*w.n+n]=round_half(round_half(row[n])*w.svh[n]);
    }
    return result;
}
void prefill(const Matrix& w,const float* x,size_t rows,size_t xs,float* y,size_t ys,Gemm provider) {
    check_io(w,x,rows,xs,y,ys);if(!provider)throw std::invalid_argument("missing GEMM provider");
    auto weights=reconstruct(w);provider(rows,w.n,w.k,x,xs,weights.data(),y,ys);
    if(!w.bias.empty())for(size_t r=0;r<rows;++r)for(size_t n=0;n<w.n;++n)y[r*ys+n]+=w.bias[n];
}
}
