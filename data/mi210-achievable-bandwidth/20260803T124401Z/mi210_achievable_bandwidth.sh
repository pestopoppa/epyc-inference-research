#!/bin/bash
# MI210 achievable-bandwidth probe (STREAM/BabelStream-class).
#
# WHY THIS EXISTS
#   Every roofline percentage this project holds -- ours and every one quoted at
#   us -- is computed against the MI210's DATASHEET peak (1.638 TB/s), because no
#   measured achievable figure exists anywhere in the repo. autokernel-research-loop.md
#   §8.3.1 requires TWO denominators and can only satisfy one of them.
#
#   If real achievable is ~1.3-1.4 TB/s (typical for HBM2e), every MI210 attainment
#   figure we hold rises 17-26% and the AMD-vs-NVIDIA gap narrows correspondingly.
#   This is the denominator of every roofline claim we have ever made.
#
# WHAT IT DOES NOT DO
#   It never kills, signals, stops, drains or reloads anything. If the GPU does not
#   have enough free VRAM it prints what is resident and EXITS NON-ZERO. Freeing the
#   device belongs to whoever owns the inference, at their own boundary
#   (OPERATING_CONSTRAINTS.md -> Inference and Benchmarks).
#
# USAGE
#   bash scripts/benchmark/mi210_achievable_bandwidth.sh                 # preflight + run
#   bash scripts/benchmark/mi210_achievable_bandwidth.sh --preflight-only
#   bash scripts/benchmark/mi210_achievable_bandwidth.sh --min-free-gib 3 --size-pow 26
#
# GRADE
#   OBSERVATION. This is a substrate-constant probe, not a protocol-bound benchmark:
#   no P-GPU-1 claim, no era row, no promotion authority. It measures the machine,
#   not a candidate.

set -euo pipefail

MIN_FREE_GIB=4
SIZE_POW=27          # elements per array = 2^SIZE_POW doubles -> 1.07 GiB/array at 27
WARMUP=10
ITERS=50
PREFLIGHT_ONLY=0
DEVICE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-gib) MIN_FREE_GIB="$2"; shift 2 ;;
    --size-pow)     SIZE_POW="$2"; shift 2 ;;
    --warmup)       WARMUP="$2"; shift 2 ;;
    --iters)        ITERS="$2"; shift 2 ;;
    --device)       DEVICE="$2"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="${REPO_ROOT}/data/mi210-achievable-bandwidth/${STAMP}"

# ---------------------------------------------------------------- preflight
echo "== MI210 achievable-bandwidth probe =="
echo "   evidence dir : ${EVIDENCE_DIR}"

command -v hipcc   >/dev/null || { echo "FAIL: hipcc not found"; exit 3; }
command -v rocm-smi >/dev/null || { echo "FAIL: rocm-smi not found"; exit 3; }

BYTES_PER_ARRAY=$(( (1 << SIZE_POW) * 8 ))
NEED_BYTES=$(( BYTES_PER_ARRAY * 3 + 64*1024*1024 ))   # 3 arrays + slack
NEED_GIB=$(awk -v b="$NEED_BYTES" 'BEGIN{printf "%.2f", b/1073741824}')

TOTAL_B=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/card/{print $2; exit}')
USED_B=$(rocm-smi  --showmeminfo vram --csv 2>/dev/null | awk -F, '/card/{print $3; exit}')
if [[ -z "${TOTAL_B:-}" || -z "${USED_B:-}" ]]; then
  TOTAL_B=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Total Memory/{print $NF}')
  USED_B=$(rocm-smi  --showmeminfo vram 2>/dev/null | awk '/Total Used Memory/{print $NF}')
fi
FREE_B=$(( TOTAL_B - USED_B ))
FREE_GIB=$(awk -v b="$FREE_B" 'BEGIN{printf "%.2f", b/1073741824}')

echo "   VRAM         : ${FREE_GIB} GiB free of $(awk -v b="$TOTAL_B" 'BEGIN{printf "%.2f", b/1073741824}') GiB"
echo "   probe needs  : ${NEED_GIB} GiB (3 x $(awk -v b="$BYTES_PER_ARRAY" 'BEGIN{printf "%.2f", b/1073741824}') GiB arrays, 2^${SIZE_POW} doubles each)"

# arrays must swamp the last-level cache. MI210/CDNA2 L2 is 8 MiB; no Infinity Cache.
L2_MIB=8
RATIO=$(awk -v a="$BYTES_PER_ARRAY" -v l="$((L2_MIB*1048576))" 'BEGIN{printf "%.0f", a/l}')
echo "   cache margin : each array is ${RATIO}x the 8 MiB L2 (want >= 4x)"

echo "   resident on the device (read-only; nothing will be signalled):"
rocm-smi --showpids 2>/dev/null | sed -n '/PID/,/^===/p' | sed 's/^/     /' || true

if (( FREE_B < NEED_BYTES )); then
  cat <<EOF

REFUSING TO RUN -- not enough free VRAM.

  free   ${FREE_GIB} GiB
  needed ${NEED_GIB} GiB

The probe must allocate multi-GiB buffers to exceed L2 and actually measure HBM.
Allocating into the last few hundred MiB would (a) risk OOM-ing a live production
server and (b) produce a cache-inflated number that is worse than no number.

WHAT HAS TO HAPPEN, and it is not this script's call:
  the resident servers above must be drained by whoever owns the inference, at
  their own boundary. This script will never do that.

Alternatively, lower the footprint deliberately and accept the caveat:
  bash ${BASH_SOURCE[0]} --size-pow 25 --min-free-gib 1   # 256 MiB/array, still 32x L2

EOF
  exit 4
fi

if (( PREFLIGHT_ONLY == 1 )); then echo; echo "preflight OK -- exiting before the run (--preflight-only)"; exit 0; fi

# ---------------------------------------------------------------- build
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
SRC="${BUILD_DIR}/stream.hip"

cat > "$SRC" <<'HIPSRC'
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
HIPSRC

echo
echo "-- building (hipcc, --offload-arch=gfx90a) ..."
# NOTE: status must come from hipcc, not from `tail`. Piping the compiler
# into tail made a FAILED BUILD exit 0 and the script continued.
set -o pipefail
hipcc -O3 --offload-arch=gfx90a -o "${BUILD_DIR}/stream" "$SRC" 2>&1 | tail -20
build_rc=${PIPESTATUS[0]}
set +o pipefail
if [[ $build_rc -ne 0 ]]; then
  echo "FAIL: hipcc build failed (rc=$build_rc); refusing to report a number."
  exit 5
fi

# ---------------------------------------------------------------- run
mkdir -p "$EVIDENCE_DIR"
cp "$SRC" "${EVIDENCE_DIR}/stream.hip"
cp "${BASH_SOURCE[0]}" "${EVIDENCE_DIR}/$(basename "${BASH_SOURCE[0]}")"

echo "-- running: ${WARMUP} warmup + ${ITERS} timed, HIP-event timed, sync outside the measured region"
HIP_VISIBLE_DEVICES="$DEVICE" "${BUILD_DIR}/stream" "$SIZE_POW" "$WARMUP" "$ITERS" 0 \
  | tee "${EVIDENCE_DIR}/raw_result.json"

# ---------------------------------------------------------------- receipt
python3 - "$EVIDENCE_DIR" "$STAMP" "$SIZE_POW" "$WARMUP" "$ITERS" <<'PY'
import json, sys, subprocess, hashlib, os, datetime
d, stamp, pw, wu, it = sys.argv[1:6]
raw = json.load(open(os.path.join(d, "raw_result.json")))
SPEC_TBPS = 1.638e3          # MI210 datasheet peak, GB/s decimal
triad = raw["kernels"]["triad"]["best_GBps"]
best  = max(v["best_GBps"] for v in raw["kernels"].values())
rec = {
  "measurement": "mi210_achievable_bandwidth",
  "grade": "OBSERVATION",
  "protocol": None,
  "note": ("Substrate-constant probe, not a protocol-bound benchmark. No P-GPU-1 claim, "
           "no era row, no promotion authority. Supplies the SECOND denominator required by "
           "autokernel-research-loop.md 8.3.1."),
  "utc": stamp,
  "rocm": open("/opt/rocm/.info/version").read().strip() if os.path.exists("/opt/rocm/.info/version") else None,
  "config": {"size_pow": int(pw), "warmup": int(wu), "timed_iters": int(it)},
  "raw": raw,
  "denominators": {
    "datasheet_peak_GBps": SPEC_TBPS,
    "measured_achievable_GBps": best,
    "measured_triad_GBps": triad,
    "achievable_fraction_of_spec": round(best / SPEC_TBPS, 4),
  },
  "reading": (
    "Every MI210 roofline percentage this project holds was computed against "
    f"{SPEC_TBPS:.0f} GB/s. Against the measured achievable figure of {best:.1f} GB/s "
    f"they all scale by {SPEC_TBPS/best:.3f}x."
  ),
}
p = os.path.join(d, "receipt.json")
json.dump(rec, open(p, "w"), indent=2)
sums = []
for f in sorted(os.listdir(d)):
    fp = os.path.join(d, f)
    if os.path.isfile(fp):
        sums.append(f"{hashlib.sha256(open(fp,'rb').read()).hexdigest()}  {f}")
open(os.path.join(d, "SHA256SUMS"), "w").write("\n".join(sums) + "\n")
print()
print("=" * 72)
print(f"  measured achievable : {best:.1f} GB/s   (best of 5 kernels)")
print(f"  triad               : {triad:.1f} GB/s")
print(f"  datasheet peak      : {SPEC_TBPS:.0f} GB/s")
print(f"  achievable/spec     : {best/SPEC_TBPS*100:.1f}%")
print(f"  correctness         : {'PASS' if raw['correctness_passed'] else 'FAIL'}"
      f" (max abs err {raw['correctness_max_abs_error']:.1e})")
print()
print(f"  -> every existing MI210 attainment % rises by {SPEC_TBPS/best:.3f}x")
print(f"     against this denominator.")
print("=" * 72)
print(f"  receipt: {p}")
PY

echo
echo "Evidence written to ${EVIDENCE_DIR} (with SHA256SUMS). Commit it to make the"
echo "denominator durable -- an untracked receipt looks identical to a committed one."
