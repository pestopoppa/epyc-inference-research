#!/bin/bash
# WG-LFM-1 scout bench — q0-scoped (cores 0-23 = NUMA node 0).
# Canonical env stack + flags imported verbatim from bench_canonical.sh's emitted
# command; ONLY the CPU-scope dimension deviates (region grant is q0, not 0-95).
# No pipes around the llama binary (feedback_pipe_hazards) — redirect only.
set -euo pipefail
ulimit -c 0

BIN=/mnt/raid0/llm/llama.cpp/build/bin/llama-bench
OUTDIR=/workspace/tmp/wg-lfm-1

export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/mnt/raid0/llm/llama.cpp/build/bin:/mnt/raid0/llm/llama.cpp-dflash/build/bin:/opt/rocm/lib
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
export GGML_IQK=1

for m in "$@"; do
  tag="$(basename "$m" .gguf)"
  echo "=== ARM $tag @ $(date -Is) ==="
  /usr/bin/time -v taskset -c 0-23 numactl --membind=0 -- \
    "$BIN" -t 24 -fa 1 -mmp 0 -m "$m" -p 512 -n 128 -r 3 -o md \
    > "${OUTDIR}/bench_${tag}.md" 2> "${OUTDIR}/bench_${tag}.time"
  echo "--- done $tag rc=$? ---"
done
echo "ALL_ARMS_DONE"
