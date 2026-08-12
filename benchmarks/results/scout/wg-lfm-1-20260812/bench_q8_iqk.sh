#!/bin/bash
# WG-LFM-1 addendum — Q8_0 arm WITH the canonical GGML_IQK_Q8_0 gate.
# canonical_recipe.py exposes --ggml-iqk-q8-0 precisely because Q8_0 rows need
# this second gate to reach the iqk path; the first Q8_0 arm ran without it and
# logged no "[iqk] ACTIVE" line, so it was NOT the top-optimized configuration.
set -uo pipefail
D=/workspace/tmp/wg-lfm-1
BIN=/mnt/raid0/llm/llama.cpp/build/bin/llama-bench
M=/mnt/raid0/llm/models/LFM2.5-2.6B-Q8_0.gguf

export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/mnt/raid0/llm/llama.cpp/build/bin:/mnt/raid0/llm/llama.cpp-dflash/build/bin:/opt/rocm/lib
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
export GGML_IQK=1 GGML_IQK_Q8_0=1

echo "=== ARM LFM2.5-2.6B-Q8_0 (GGML_IQK_Q8_0=1) @ $(date -Is) ==="
/usr/bin/time -v taskset -c 0-23 numactl --membind=0 -- \
  "$BIN" -t 24 -fa 1 -mmp 0 -m "$M" -p 512 -n 128 -r 3 -o md \
  > "${D}/bench_LFM2.5-2.6B-Q8_0-iqkq8.md" 2> "${D}/bench_LFM2.5-2.6B-Q8_0-iqkq8.time"
echo "--- done rc=$? ---"
