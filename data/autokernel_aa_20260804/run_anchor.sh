#!/bin/bash
set -euo pipefail
# Constants below are COPIED FROM the ratified codified recipe at
# epyc-inference-research/scripts/lib/canonical_recipe.py — read, not remembered.
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
BENCH=/mnt/raid0/llm/llama.cpp/build/bin/llama-bench
MODEL=/mnt/raid0/llm/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
OUT=$1
taskset -c 0-95 numactl --interleave=all "$BENCH" \
  -m "$MODEL" -t 96 -fa 1 -mmp 0 -p 512 -n 128 -r 5 -o json > "$OUT" 2> "${OUT%.json}.stderr"
echo "done -> $OUT"
