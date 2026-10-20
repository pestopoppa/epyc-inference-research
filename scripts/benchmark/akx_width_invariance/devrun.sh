#!/bin/bash
set -uo pipefail
H=/mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/harness
SRC="$(cd "$(dirname "$0")" && pwd)"
TREE=/mnt/raid0/llm/llama.cpp-experimental-champref-4348de400-20261006
CPUB=/mnt/raid0/llm/kernels/builds/cpu-20261006-4348de400/bin
GPUB=/mnt/raid0/llm/kernels/builds/gpu-20261006-4348de400/bin
CASES=/mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/predicted_fail_list.jsonl
echo "== build cpu"; /usr/bin/g++ -O2 -std=c++17 -Wall -I$TREE/ggml/include $SRC/width_probe.cpp -L$CPUB -lggml -lggml-base -lggml-cpu -lpthread -Wl,-rpath,$CPUB -o $H/width_probe_cpu || exit 10
echo "== link hip (not run)"; /usr/bin/g++ -O2 -std=c++17 -I$TREE/ggml/include $SRC/width_probe.cpp -L$GPUB -lggml -lggml-base -lggml-cpu -lpthread -Wl,-rpath,$GPUB:/opt/rocm/lib -Wl,-rpath-link,/opt/rocm/lib -o $H/width_probe_hip || echo "HIP LINK FAILED"
mkdir -p $H/devruns/smoke; rm -f $H/devruns/smoke/r.jsonl
echo "== smoke"
env -u GGML_IQK_Q8_0 GGML_IQK=1 LD_LIBRARY_PATH=$CPUB timeout 900 $H/width_probe_cpu --backend cpu --cases $CASES --ids PC-1,CPU-MM-Q8_0-ALL,CPU-MMID-Q8_0-ALL,CPU-FA-F16-2048,CPU-GDN-F32,GRAPH-GDN-builder,CPU-MM-Q4_K-F640 --kmax 2 --repeats 2 --threads 16 --out $H/devruns/smoke/r.jsonl 2>&1 | grep -v ' k=' | tail -40
echo "smoke rc=${PIPESTATUS[0]}"
echo "== rocm0 abort test on CPU binary (must exit 4, no fallback)"
LD_LIBRARY_PATH=$CPUB $H/width_probe_cpu --backend rocm0 --ids PC-1 --out $H/devruns/smoke/rocm_abort.jsonl 2>&1 | tail -2; echo "rocm0-on-cpu-binary rc=${PIPESTATUS[0]}"
[ "${1:-}" = "smoke-only" ] && exit 0
echo "== dev subset via driver"
timeout 7000 /usr/bin/python3 $SRC/run_width_probe.py run --backend cpu --build-bin $CPUB --binary $H/width_probe_cpu --cases $CASES \
  --out $H/devruns/cpu_subset --threads 16 --kmax 9 --repeats 3 \
  --ids PC-1,CPU-MM-Q8_0-ALL,CPU-MM-Q4_K-ALL,CPU-MM-IQ2_XXS-ALL,CPU-MMID-IQ2_XXS-ALL,CPU-MMID-Q8_0-ALL,CPU-FA-F16-2048,CPU-FA-Q8_0-2048,CPU-GDN-F32,GRAPH-GDN-builder,CPU-MM-Q4_K-F640,CPUN-MM-Q8_0-ALL
echo "driver rc=$?"
