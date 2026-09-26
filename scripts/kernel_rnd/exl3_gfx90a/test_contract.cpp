#include "contract.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
using namespace exl3;
unsigned checks=0;
void check(bool b){++checks;if(!b)throw std::runtime_error("contract assertion "+std::to_string(checks));}
template<class F> void refuses(F f){bool refused=false;try{f();}catch(const std::invalid_argument&){refused=true;}check(refused);}
Matrix matrix(unsigned e,unsigned rate=8){Matrix m{};m.packed=reinterpret_cast<const uint32_t*>(uintptr_t(4096+e*4096));
    m.inputs=15;m.outputs=17;m.padded_inputs=16;m.padded_outputs=32;m.rate_x2=rate;
    m.packed_bytes=32*rate;m.global_expert=e;
    for(auto* d:{&m.identity.model,&m.identity.source,&m.identity.tensor}){d->fill('a');(*d)[64]=0;}
    std::memcpy(m.identity.role.data(),"gate",5);return m;}
Config config(unsigned e=4,unsigned t=16){Config c{};c.tokens=t;c.topk=1;c.routes=t;c.experts=e;c.global_experts=e+2;
    c.inputs=15;c.outputs=17;c.padded_inputs=16;c.padded_outputs=32;c.prefill_rows=t;c.scheduler_slices=e;return c;}
int main(){
    unsigned cells=0;
    for(auto policy:{Policy::unified,Policy::grouped})for(bool bc:{false,true})for(bool ext:{false,true})
    for(bool capture:{false,true})for(unsigned tokens:{0u,1u,3u,4u,5u,15u,16u,17u})
    for(unsigned rows:{0u,1u,3u,4u,5u,15u,16u,17u})for(unsigned bucket:{0u,1u,3u,4u,5u}){
        Path want=!rows?Path::empty:policy==Policy::unified?Path::unified:
            !bc||!ext?Path::batched_fallback:tokens<4||rows<4?Path::small_row_fallback:
            bucket<4?Path::single_expert_fallback:Path::grouped;
        if(capture && want!=Path::empty&&want!=Path::unified&&want!=Path::grouped)want=Path::capture_fallback_refused;
        check(capability(8,Codebook::mcg,Layout::native_input_major_v1,policy,{bc,ext,capture},tokens,rows,bucket)==want);++cells;
    }
    for(unsigned rate=0;rate<=18;++rate){auto p=capability(rate,Codebook::mul1,Layout::native_input_major_v1,Policy::unified,{},4,4,4);
        check((p==Path::unsupported_rate)==(rate<2||rate>16||(rate&1)));}
    check(capability(8,Codebook(9),Layout::native_input_major_v1,Policy::unified,{},4,4,4)==Path::unsupported_codebook);
    check(capability(8,Codebook::mcg,Layout(9),Policy::unified,{},4,4,4)==Path::unsupported_layout);
    std::array<Matrix,max_experts> matrices{};for(unsigned i=0;i<max_experts;++i)matrices[i]=matrix(i,2+2*(i%8));
    auto c=config(256,17);c.policy=Policy::grouped;auto p=plan(c,matrices.data());
    std::vector<Route> routes(17);for(unsigned i=0;i<17;++i)routes[i]={i,0,i<16?i:257,1.f};
    auto s=schedule(p,routes.data(),17,1,256);check(s.counts[256]==1);check(s.routes[16].local_expert==-1);
    check(s.counts[255]==0);check(s.paths.size()==256);check(s.counts.size()==257);
    // All-local, all-nonlocal and mixed; routing order remains immutable.
    for(unsigned kind=0;kind<3;++kind){for(unsigned i=0;i<17;++i)routes[i].global_expert=kind==0?i:kind==1?257:(i%2?257:i);
        auto q=schedule(p,routes.data(),17,1,256);check(q.counts[256]==(kind==0?0u:kind==1?17u:8u));}
    refuses([&]{schedule(p,routes.data(),18,1,256);});refuses([&]{schedule(p,routes.data(),17,2,256);});
    refuses([&]{schedule(p,routes.data(),17,1,0);});refuses([&]{schedule(p,routes.data(),17,1,257);});
    routes[0].routing_k=1;refuses([&]{schedule(p,routes.data(),17,1,256);});routes[0].routing_k=0;
    routes[0].global_expert=258;refuses([&]{schedule(p,routes.data(),17,1,256);});routes[0].global_expert=0;
    routes[0].weight=std::numeric_limits<float>::quiet_NaN();refuses([&]{schedule(p,routes.data(),17,1,256);});routes[0].weight=1;
    c.max_arena_bytes=1;refuses([&]{plan(c,matrices.data());});c=config();
    matrices[0].rate_x2=7;refuses([&]{plan(c,matrices.data());});matrices[0]=matrix(0);
    matrices[0].codebook=Codebook(3);refuses([&]{plan(c,matrices.data());});matrices[0]=matrix(0);
    matrices[0].identity.source[0]='?';refuses([&]{plan(c,matrices.data());});matrices[0]=matrix(0);
    matrices[0].global_expert=1;refuses([&]{plan(c,matrices.data());});matrices[0]=matrix(0);
    matrices[0].packed_bytes--;refuses([&]{plan(c,matrices.data());});matrices[0]=matrix(0);
    p=plan(c,matrices.data());auto initial=p.identity;
    p.matrices[0].packed=reinterpret_cast<const uint32_t*>(1234);check(identity_hash(p)!=initial);p=plan(c,matrices.data());
    p.matrices[0].packed_bytes++;check(identity_hash(p)!=initial);p=plan(c,matrices.data());
    p.matrices[0].inputs++;check(identity_hash(p)!=initial);p=plan(c,matrices.data());
    p.matrices[0].bias=reinterpret_cast<const float*>(1234);check(identity_hash(p)!=initial);p=plan(c,matrices.data());
    p.partial_offset++;check(identity_hash(p)!=initial);p=plan(c,matrices.data());
    p.config.materialized_reference=true;check(identity_hash(p)!=initial);
    p.config.experts=max_experts+1;check(identity_hash(p)==0);
    for(unsigned t:{1u,3u,4u,15u,16u,17u})for(unsigned k:{1u,2u,4u}){
        auto a=config(4,t);a.topk=k;a.routes=t*k;a.policy=Policy::grouped;
        auto pp=plan(a,matrices.data());std::vector<Route> rr(t*k);
        for(unsigned i=0;i<t*k;++i)rr[i]={i/k,i%k,(i/k)%4,0.25f};
        auto qq=schedule(pp,rr.data(),t,k,4);check(qq.size==t*k);
        for(unsigned i=0;i<t*k;++i)check(qq.routes[i].routing_k==i%k);
    }
    std::cout<<"{\"checks\":"<<checks<<",\"truth_table_cells\":"<<cells<<",\"max_experts\":256,\"status\":\"pass\"}\n";
}
