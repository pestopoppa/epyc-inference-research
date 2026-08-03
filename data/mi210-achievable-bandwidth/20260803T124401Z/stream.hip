// BabelStream-class kernels on HIP. Five kernels, correctness-checked, HIP-event timed.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(9);} } while(0)

__global__ void k_copy (double*a,const double*c,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]=c[i];}
__global__ void k_mul  (double*b,const double*c,double s,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=s*c[i];}
__global__ void k_add  (double*c,const double*a,const double*b,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i];}
__global__ void k_triad(double*a,const double*b,const double*c,double s,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]=b[i]+s*c[i];}
__global__ void k_init (double*a,double*b,double*c,double ia,double ib,double ic,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){a[i]=ia;b[i]=ib;c[i]=ic;}}

static double pct(std::vector<double>&v,double p){
  std::vector<double> s=v; std::sort(s.begin(),s.end());
  double idx=p*(s.size()-1); size_t lo=(size_t)idx; size_t hi=std::min(lo+1,s.size()-1);
  double f=idx-lo; return s[lo]*(1-f)+s[hi]*f;
}

int main(int argc,char**argv){
  int size_pow = atoi(argv[1]);
  int warmup   = atoi(argv[2]);
  int iters    = atoi(argv[3]);
  int device   = atoi(argv[4]);
  size_t n = (size_t)1 << size_pow;
  size_t bytes = n*sizeof(double);

  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop,device));

  double *a,*b,*c;
  HIP_CHECK(hipMalloc(&a,bytes)); HIP_CHECK(hipMalloc(&b,bytes)); HIP_CHECK(hipMalloc(&c,bytes));

  const int TPB=256; size_t blocks=(n+TPB-1)/TPB;
  const double ia=0.1, ib=0.2, ic=0.0, scalar=0.4;

  hipEvent_t e0,e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));

  const char* names[4]={"copy","mul","add","triad"};
  // bytes moved per kernel: copy r+w=2n, mul r+w=2n, add 2r+w=3n, triad 2r+w=3n
  const double bfac[4]={2.0,2.0,3.0,3.0};
  std::vector<std::vector<double>> t(4);

  for(int rep=0; rep<warmup+iters; ++rep){
    hipLaunchKernelGGL(k_init,dim3(blocks),dim3(TPB),0,0,a,b,c,ia,ib,ic,n);
    HIP_CHECK(hipDeviceSynchronize());

    float ms;
    #define TIME(idx, launch) \
      HIP_CHECK(hipEventRecord(e0)); launch; HIP_CHECK(hipEventRecord(e1)); \
      HIP_CHECK(hipEventSynchronize(e1)); HIP_CHECK(hipEventElapsedTime(&ms,e0,e1)); \
      if(rep>=warmup) t[idx].push_back(ms);

    TIME(0, hipLaunchKernelGGL(k_copy ,dim3(blocks),dim3(TPB),0,0,c,a,n))
    TIME(1, hipLaunchKernelGGL(k_mul  ,dim3(blocks),dim3(TPB),0,0,b,c,scalar,n))
    TIME(2, hipLaunchKernelGGL(k_add  ,dim3(blocks),dim3(TPB),0,0,c,a,b,n))
    TIME(3, hipLaunchKernelGGL(k_triad,dim3(blocks),dim3(TPB),0,0,a,b,c,scalar,n))
  }

  // ---- correctness: replay the kernel sequence on the host and compare
  double ga=ia, gb=ib, gc=ic;
  gc=ga; gb=scalar*gc; gc=ga+gb; ga=gb+scalar*gc;
  std::vector<double> ha(1024), hb(1024), hc(1024);
  HIP_CHECK(hipMemcpy(ha.data(),a,1024*sizeof(double),hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hb.data(),b,1024*sizeof(double),hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(hc.data(),c,1024*sizeof(double),hipMemcpyDeviceToHost));
  double maxerr=0;
  for(int i=0;i<1024;i++){
    maxerr=std::max(maxerr,std::fabs(ha[i]-ga));
    maxerr=std::max(maxerr,std::fabs(hb[i]-gb));
    maxerr=std::max(maxerr,std::fabs(hc[i]-gc));
  }
  int correct = (maxerr < 1e-12) ? 1 : 0;

  printf("{\n");
  printf("  \"device_name\": \"%s\",\n", prop.gcnArchName);
  printf("  \"elements\": %zu,\n", n);
  printf("  \"bytes_per_array\": %zu,\n", bytes);
  printf("  \"warmup\": %d,\n  \"timed_iters\": %d,\n", warmup, iters);
  printf("  \"correctness_passed\": %s,\n", correct?"true":"false");
  printf("  \"correctness_max_abs_error\": %.3e,\n", maxerr);
  printf("  \"kernels\": {\n");
  for(int k=0;k<4;k++){
    double mn=*std::min_element(t[k].begin(),t[k].end());
    double md=pct(t[k],0.5), p20=pct(t[k],0.20), p80=pct(t[k],0.80);
    double moved=bfac[k]*(double)bytes;
    printf("    \"%s\": {", names[k]);
    printf(" \"bytes_moved\": %.0f,", moved);
    printf(" \"best_GBps\": %.2f,",   moved/(mn *1e6));   // ms -> GB/s decimal
    printf(" \"median_GBps\": %.2f,", moved/(md *1e6));
    printf(" \"p20_GBps\": %.2f,",    moved/(p80*1e6));   // slow time -> low BW
    printf(" \"p80_GBps\": %.2f,",    moved/(p20*1e6));
    printf(" \"best_GiBps\": %.2f",   moved/(mn *1e-3)/1073741824.0);
    printf(" }%s\n", k<3?",":"");
  }
  printf("  }\n}\n");

  HIP_CHECK(hipFree(a)); HIP_CHECK(hipFree(b)); HIP_CHECK(hipFree(c));
  return correct?0:8;
}
