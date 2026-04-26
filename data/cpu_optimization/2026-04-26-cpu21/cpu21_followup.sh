#!/bin/bash
# Post-CPU21-sweep follow-up: validate the best stack on REAP-246B (the
# model with the worst sync overhead — 4.27x scaling vs 96x ideal). If
# CPU21 affinity tuning helps Coder-30B by +5-7%, does it help REAP too?

set -uo pipefail

SWEEP_DATA=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-26-cpu21
DATA=$SWEEP_DATA/followup
mkdir -p $DATA
LDLP=/mnt/raid0/llm/llama.cpp-experimental/build/bin:/opt/AMD/aocc-compiler-5.0.0/lib
BENCH=/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench
REAP=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf
QWEN36=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf

run() {
  local name=$1; shift
  local model=$1; shift
  local extra_env=$@
  echo "=== $name ==="
  sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1
  env $extra_env LD_LIBRARY_PATH=$LDLP \
    numactl --interleave=all --physcpubind=0-95 $BENCH \
    -m "$model" -t 96 -fa 1 -p 0 -n 32 -r 3 \
    > "$DATA/${name}.log" 2>&1
  tail -4 "$DATA/${name}.log" | head -2
}

echo "============ REAP-246B with affinity variants ============"
run REAP_baseline       $REAP
run REAP_spread_cores   $REAP OMP_PROC_BIND=spread  OMP_PLACES=cores
run REAP_close_threads  $REAP OMP_PROC_BIND=close   OMP_PLACES=threads
run REAP_combined_stack $REAP OMP_PROC_BIND=spread  OMP_PLACES=cores  OMP_WAIT_POLICY=active

echo "============ Qwen3.6-35B Q8_0 with affinity variants ============"
run Q8_baseline       $QWEN36
run Q8_spread_cores   $QWEN36 OMP_PROC_BIND=spread  OMP_PLACES=cores
run Q8_close_threads  $QWEN36 OMP_PROC_BIND=close   OMP_PLACES=threads
run Q8_combined_stack $QWEN36 OMP_PROC_BIND=spread  OMP_PLACES=cores  OMP_WAIT_POLICY=active

echo "============ Coder-30B combined stack (verify additivity) ============"
run Coder_combined_stack /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active

echo "============ DONE ============"
