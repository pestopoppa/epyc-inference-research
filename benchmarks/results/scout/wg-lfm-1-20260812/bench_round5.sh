#!/bin/bash
# WG-LFM-1 round 5 — the arm intended as the headline table.
# Round 4 (numactl --membind=0) produced stddevs up to 51% of the mean; node 0
# had only ~1.4 GB free at the time, so a 16 GB --no-mmap allocation bound to
# node 0 forced page-cache reclaim mid-run. Round 5 keeps compute on q0
# (taskset 0-23, -t 24) but uses the CANONICAL --interleave=all memory policy,
# and raises reps to 5 so the error bar is trustworthy.
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
  echo "=== ARM r5 $tag @ $(date -Is) ==="
  /usr/bin/time -v taskset -c 0-23 numactl --interleave=all -- \
    "$BIN" -t 24 -fa 1 -mmp 0 -m "$m" -p 512 -n 512 -r 5 -o md \
    > "${D}/bench_r5_${tag}.md" 2> "${D}/bench_r5_${tag}.time"
  echo "--- done $tag rc=$? ---"
done
echo "### ROUND5_DONE $(date -Is)"
