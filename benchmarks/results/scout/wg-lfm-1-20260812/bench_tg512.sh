#!/bin/bash
# WG-LFM-1 round 4 — tg512 cross-check.
# tg128 put LFM Q4_K_M at 24.43 t/s while llama-cli decoded the same model at
# ~42.5 t/s in the same q0 cell; Q8_0 agreed between the two instruments. A
# longer generation amortises per-run fixed cost and tells us which reading is
# the model's steady-state decode rate.
set -uo pipefail
D=/workspace/tmp/wg-lfm-1
BIN=/mnt/raid0/llm/llama.cpp/build/bin/llama-bench
export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/mnt/raid0/llm/llama.cpp/build/bin:/mnt/raid0/llm/llama.cpp-dflash/build/bin:/opt/rocm/lib
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
export GGML_IQK=1 GGML_IQK_Q8_0=1
for m in /mnt/raid0/llm/models/LFM2.5-2.6B-Q4_K_M.gguf \
         /mnt/raid0/llm/models/LFM2.5-2.6B-Q8_0.gguf \
         /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf; do
  tag="$(basename "$m" .gguf)"
  echo "=== ARM tg512 $tag @ $(date -Is) ==="
  /usr/bin/time -v taskset -c 0-23 numactl --membind=0 -- \
    "$BIN" -t 24 -fa 1 -mmp 0 -m "$m" -p 0 -n 512 -r 3 -o md \
    > "${D}/bench_tg512_${tag}.md" 2> "${D}/bench_tg512_${tag}.time"
  echo "--- done $tag rc=$? ---"
done
echo "### ROUND4_DONE $(date -Is)"
