#!/bin/bash
# LFM2.5-2.6B-Q4_K_M through the IDENTICAL raw harness used for gemma4, so both
# sides of the comparison are measured with the same instrument and the same
# tokenizer-derived token counts. LFM has no enable_thinking toggle (its embedded
# template unconditionally prefills '<think>'), so there is one condition only.
set -euo pipefail
ulimit -c 0

BIN=/mnt/raid0/llm/llama.cpp/build/bin/llama-completion
MODEL=/mnt/raid0/llm/models/LFM2.5-2.6B-Q4_K_M.gguf
OUT=/workspace/tmp/wg-lfm-1-thinking

export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/mnt/raid0/llm/llama.cpp/build/bin:/mnt/raid0/llm/llama.cpp-dflash/build/bin:/opt/rocm/lib
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
export GGML_IQK=1

for q in 1 2 3 4 5; do
  echo "### RUN lfm q${q} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  taskset -c 0-23 numactl --membind=0 -- \
    "$BIN" -m "$MODEL" -t 24 -fa 1 --no-mmap --no-warmup -no-cnv \
    --temp 0 -s 42 -n 512 -c 8192 --verbose-prompt \
    -f "${OUT}/lfm_q${q}.pf" \
    > "${OUT}/gen_lfm_q${q}.out" 2> "${OUT}/gen_lfm_q${q}.err"
  echo "    done rc=$?"
done
echo "MEASURE_LFM_DONE"
