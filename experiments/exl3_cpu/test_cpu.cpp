#include "exl3_cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfenv>
#include <limits>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
using namespace exl3;
namespace {
constexpr float guard=123456.25f;
size_t checks=0;
void ensure(bool b,const std::string& s){++checks;if(!b)throw std::runtime_error(s);}
template<class F> void refuses(F f){bool refused=false;try{f();}catch(const std::invalid_argument&){refused=true;}ensure(refused,"expected refusal");}
template<class T>void read(std::ifstream& f,T* p,size_t n){f.read(reinterpret_cast<char*>(p),n*sizeof(T));ensure(bool(f),"mandatory fixture truncated");}
std::mt19937 rng(271828);
Matrix matrix(unsigned bits,Codebook cb,size_t k=131,size_t n=149){
    Matrix w{k,n,(k+127)/128*128,(n+127)/128*128,bits,cb,Layout::native,{},{},{},{}};
    w.trellis.resize(w.padded_k*w.padded_n*bits/16);for(auto&v:w.trellis)v=uint16_t(rng());
    w.suh.resize(w.padded_k);for(auto&v:w.suh)v=(int(rng()%200)-100)*.00025f;
    w.svh.resize(w.padded_n);for(auto&v:w.svh)v=(int(rng()%200)-100)*.0125f;
    return w;
}
std::vector<float> input(size_t rows,size_t stride){std::vector<float>x(rows*stride,guard);for(size_t r=0;r<rows;++r)for(size_t k=0;k<stride-7;++k)x[r*stride+k]=std::sin(float(k+r*13)*.17f);return x;}
void compare(const std::vector<float>& a,const std::vector<float>& b,float tol=0){ensure(a.size()==b.size(),"size");for(size_t i=0;i<a.size();++i)ensure(std::isfinite(a[i])&&std::abs(a[i]-b[i])<=tol*(1+std::abs(b[i])),"comparison at "+std::to_string(i)+" a="+std::to_string(a[i])+" b="+std::to_string(b[i]));}
void guards(const std::vector<float>& y,size_t rows,size_t n,size_t stride){ensure(y.front()==guard&&y.back()==guard,"outer guards");for(size_t r=0;r<rows;++r)for(size_t j=n;j<stride;++j)ensure(y[1+r*stride+j]==guard,"stride guard");}
using Blas=void(*)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
Blas blas=nullptr;
void gemm(size_t m,size_t n,size_t k,const float*a,size_t lda,const float*b,float*c,size_t ldc){blas(101,111,111,int(m),int(n),int(k),1,a,int(lda),b,int(n),0,c,int(ldc));}
void reference_gemm(size_t m,size_t n,size_t k,const float*a,size_t lda,const float*b,float*c,size_t ldc){for(size_t r=0;r<m;++r)for(size_t j=0;j<n;++j){float acc=0;for(size_t i=0;i<k;++i)acc=std::fma(a[r*lda+i],b[i*n+j],acc);c[r*ldc+j]=acc;}}
void load_blas(const char* path){void* h=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());blas=reinterpret_cast<Blas>(dlsym(h,"scipy_cblas_sgemm"));if(!blas)blas=reinterpret_cast<Blas>(dlsym(h,"cblas_sgemm"));ensure(blas,"required existing CBLAS provider unavailable");}
void fixture_tests(const std::string& dir){
    for(auto cb:{Codebook::mul1,Codebook::mcg}){
        std::ifstream f(dir+(cb==Codebook::mul1?"/mul1_lut.f16":"/mcg_lut.f16"),std::ios::binary);ensure(bool(f),"mandatory LUT missing");
        std::vector<uint16_t> lut(65536);read(f,lut.data(),lut.size());
        for(unsigned i=0;i<65536;++i){_Float16 h;std::memcpy(&h,&lut[i],2);ensure(codebook_value(i,cb)==float(h),"independent LUT mismatch");}
    }
    for(std::string name:{"mul1_k3","mul1_k4","mcg_k4"}){
        std::ifstream f(dir+"/"+name+".bin",std::ios::binary);ensure(bool(f),"mandatory real fixture missing "+name);
        uint32_t hdr[4];read(f,hdr,4);ensure(hdr[0]==0x334c5845&&hdr[3]==128,"fixture header");
        auto w=matrix(hdr[1],hdr[2]?Codebook::mcg:Codebook::mul1,128,128);
        read(f,w.trellis.data(),w.trellis.size());read(f,w.suh.data(),128);read(f,w.svh.data(),128);
        std::vector<float> raw(128*128),canonical(raw.size());read(f,raw.data(),raw.size());read(f,canonical.data(),canonical.size());
        for(auto isa:{ISA::scalar,ISA::avx512bw,ISA::vnni,ISA::vbmi})if(supported(isa))
            for(size_t kt=0;kt<8;++kt)for(size_t nt=0;nt<8;++nt){float tile[256];decode_tile(w.trellis.data()+(kt*8+nt)*16*w.bits,w.bits,w.codebook,isa,tile);for(size_t i=0;i<16;++i)for(size_t j=0;j<16;++j)ensure(tile[i*16+j]==raw[(kt*16+i)*128+nt*16+j],"real raw tile parity");}
        compare(reconstruct(w),canonical,0);
        auto x=input(7,135);std::vector<float> y(7*137+2,guard),ref=y;
        reference_gemm(7,128,128,x.data(),135,canonical.data(),ref.data()+1,137);
        prefill(w,x.data(),7,135,y.data()+1,137,gemm);compare(y,ref,2e-6f);guards(y,7,128,137);
        // Fused transform is a different numeric proposition from staged materialization.
        std::vector<float> fused(128),truth(128);dense(w,x.data(),1,135,fused.data(),128,ISA::scalar);
        reference_gemm(1,128,128,x.data(),135,canonical.data(),truth.data(),128);
        double e=0,norm=0;for(size_t i=0;i<128;++i){e+=std::pow(fused[i]-truth[i],2);norm+=truth[i]*truth[i];}
        double rms=std::sqrt(e/norm);ensure(rms<.003,"fused materialized envelope");
        std::cout<<"fixture "<<name<<" fused_relative_l2="<<rms;
        if(w.codebook==Codebook::mul1){mul1_q8(w,x.data(),1,135,fused.data(),128,ISA::scalar);e=0;for(size_t i=0;i<128;++i)e+=std::pow(fused[i]-truth[i],2);rms=std::sqrt(e/norm);ensure(rms<.025,"Q8 envelope");std::cout<<" q8_relative_l2="<<rms;}
        std::cout<<"\n";
        repack(w,Layout::band8);compare(reconstruct(w),canonical,0);
    }
}
void coverage(){
    for(unsigned bits=1;bits<=8;++bits)for(auto cb:{Codebook::mul1,Codebook::mcg}) {
        auto w=matrix(bits,cb);auto original=w.trellis;
        for(auto layout:{Layout::native,Layout::band8}) {
            repack(w,layout);
            for(size_t rows=1;rows<=4;++rows) {
                size_t xs=w.k+7,ys=w.n+11;auto x=input(rows,xs);
                std::vector<float> ref(rows*ys+2,guard),qref=ref;
                dense(w,x.data(),rows,xs,ref.data()+1,ys);guards(ref,rows,w.n,ys);
                if(cb==Codebook::mul1)mul1_q8(w,x.data(),rows,xs,qref.data()+1,ys);
                for(auto isa:{ISA::avx512bw,ISA::vnni,ISA::vbmi})if(supported(isa)){
                    std::vector<float> got(rows*ys+2,guard);dense(w,x.data(),rows,xs,got.data()+1,ys,isa);compare(got,ref);guards(got,rows,w.n,ys);
                    if(cb==Codebook::mul1){mul1_q8(w,x.data(),rows,xs,got.data()+1,ys,isa);compare(got,qref);guards(got,rows,w.n,ys);}
                }
            }
        }
        repack(w,Layout::native);ensure(w.trellis==original,"repack lossless");
    }
    // Mixed K, duplicate IDs, nonmonotonic routing and expert-specific transforms.
    std::vector<Matrix> experts;for(unsigned bits=1;bits<=8;++bits)experts.push_back(matrix(bits,Codebook::mul1,17,129));
    std::vector<Route> routes{{7,2},{0,0},{3,1},{0,2},{2,0},{6,1},{1,2},{5,0},{4,1}};
    auto x=input(3,24);size_t ys=140;std::vector<float> ref(routes.size()*ys+2,guard),got=ref;
    for(bool q8:{false,true})for(auto isa:{ISA::scalar,ISA::avx512bw,ISA::vnni,ISA::vbmi})if(supported(isa)){
        for(size_t i=0;i<routes.size();++i){auto r=routes[i];if(q8)mul1_q8(experts[r.expert],x.data()+r.input_row*24,1,24,ref.data()+1+i*ys,ys,isa);else dense(experts[r.expert],x.data()+r.input_row*24,1,24,ref.data()+1+i*ys,ys,isa);}
        for(bool grouped:{false,true}){indexed(experts,routes,x.data(),3,24,got.data()+1,ys,isa,grouped,q8);compare(got,ref);guards(got,routes.size(),129,ys);}
    }
    for(size_t i=0;i<experts.size();++i) {
        experts[i].codebook=(i%2)?Codebook::mcg:Codebook::mul1;
        repack(experts[i],(i%2)?Layout::band8:Layout::native);
    }
    for(auto isa:{ISA::scalar,ISA::avx512bw,ISA::vnni,ISA::vbmi})if(supported(isa)) {
        for(size_t i=0;i<routes.size();++i){auto r=routes[i];dense(experts[r.expert],x.data()+r.input_row*24,1,24,ref.data()+1+i*ys,ys,isa);}
        for(bool grouped:{false,true}){indexed(experts,routes,x.data(),3,24,got.data()+1,ys,isa,grouped);compare(got,ref);guards(got,routes.size(),129,ys);}
    }
    auto w=matrix(4,Codebook::mul1,1,1);float z=0,y=guard;
    dense(w,&z,1,1,&y,1);ensure(y==0,"zero input");mul1_q8(w,&z,1,1,&y,1,supported(ISA::vbmi)?ISA::vbmi:ISA::scalar);ensure(y==0,"zero Q8");
    refuses([&]{dense(w,&z,5,1,&y,1);});refuses([&]{prefill(w,&z,1,1,&y,1,nullptr);});
    // Bias is after transform/crop in both fused and provider paths.
    w.bias={.375f};dense(w,&z,1,1,&y,1);ensure(y==.375f,"fused bias");
    prefill(w,&z,1,1,&y,1,gemm);ensure(y==.375f,"provider bias");
    mul1_q8(w,&z,1,1,&y,1);ensure(y==.375f,"Q8 bias");
    auto badroutes=routes;badroutes.back().input_row=3;
    std::fill(got.begin(),got.end(),guard);auto untouched=got;
    refuses([&]{indexed(experts,badroutes,x.data(),3,24,got.data()+1,ys,ISA::scalar,true);});compare(got,untouched);
    badroutes=routes;badroutes.back().expert=experts.size();
    refuses([&]{indexed(experts,badroutes,x.data(),3,24,got.data()+1,ys,ISA::scalar,false);});compare(got,untouched);
    experts.back().bias={1};
    refuses([&]{indexed(experts,routes,x.data(),3,24,got.data()+1,ys,ISA::scalar,false);});compare(got,untouched);
    experts.back().bias.clear();experts.back().k=18;
    refuses([&]{indexed(experts,routes,x.data(),3,24,got.data()+1,ys,ISA::scalar,true);});compare(got,untouched);
    refuses([&]{dense(w,&z,0,1,&y,1);});
    refuses([&]{dense(w,&z,1,0,&y,1);});
    refuses([&]{dense(w,&z,1,1,&y,0);});
    refuses([&]{dense(w,&z,1,1,&y,1,ISA(99));});
    float tile[256];uint16_t words[128]={};
    refuses([&]{decode_tile(words,4,Codebook(99),ISA::avx512bw,tile);});
    float bad=std::numeric_limits<float>::quiet_NaN();y=guard;
    refuses([&]{dense(w,&bad,1,1,&y,1);});ensure(y==guard,"nonfinite refusal preserves output");
    std::fesetround(FE_DOWNWARD);refuses([&]{dense(w,&z,1,1,&y,1);});std::fesetround(FE_TONEAREST);
    w.bits=9;refuses([&]{validate(w);});
    w.bits=0;refuses([&]{validate(w);});w.bits=4;w.trellis.pop_back();refuses([&]{validate(w);});
}
void bench(){
    const auto begin_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::cout<<"microbench_window_begin_ns="<<begin_ns<<std::endl;
    std::vector<Matrix> experts;for(unsigned b=1;b<=8;++b)experts.push_back(matrix(b,Codebook::mul1,128,128));
    std::vector<Route> routes;for(size_t i=0;i<8;++i)routes.push_back({i,0});auto x=input(1,135);std::vector<float>y(8*128);
    for(auto isa:{ISA::scalar,ISA::avx512bw,ISA::vnni,ISA::vbmi})if(supported(isa))for(bool grouped:{false,true}){
        indexed(experts,routes,x.data(),1,135,y.data(),128,isa,grouped,true);
        for(int rep=0;rep<5;++rep){auto start=std::chrono::steady_clock::now();for(int j=0;j<4;++j)indexed(experts,routes,x.data(),1,135,y.data(),128,isa,grouped,true);
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/4;
        std::cout<<"microbench candidate_observation isa="<<int(isa)<<" grouped="<<grouped<<" rep="<<rep<<" batch_us="<<us<<" checksum="<<y[0]<<"\n";}
    }
    const auto end_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::cout<<"microbench_window_end_ns="<<end_ns<<std::endl;
}
}
int main(int argc,char**argv){try{
    if(argc<3)throw std::runtime_error("usage: test_cpu FIXTURE_DIR BLAS_LIBRARY [--bench]");
    load_blas(argv[2]);for(auto isa:{ISA::scalar,ISA::avx512bw,ISA::vnni,ISA::vbmi})std::cout<<"isa "<<int(isa)<<" supported="<<supported(isa)<<"\n";
    fixture_tests(argv[1]);coverage();if(argc>3)bench();std::cout<<"PASS checks="<<checks<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
