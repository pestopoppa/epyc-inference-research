#!/bin/bash
# Phase A.1 — Q6_K PPL bit-exact gate across 5 production models, 2 env states
# Bundle: 2026-05-04-q6k-default-on-validation
set -uo pipefail

DIR="/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-05-04-q6k-default-on-validation"
BIN="/mnt/raid0/llm/llama.cpp/build_libomp_pgo_use/bin/llama-perplexity"
WIKI="/mnt/raid0/llm/data/wiki.test.raw"

declare -A MODELS=(
  [coder30b_q4km]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
  [gemma4_31b_q4km]="/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf"
  [supergemma4_31b_q4km]="/mnt/raid0/llm/models/SuperGemma4-31b-abliterated.Q4_K_M.gguf"
  [qwen3next_80b_a3b_q4km]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
  [reap246b_a35b_q4km]="/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf"
)

# Order from smallest to largest so we get fast feedback
ORDER=(coder30b_q4km gemma4_31b_q4km supergemma4_31b_q4km qwen3next_80b_a3b_q4km reap246b_a35b_q4km)

cd "$DIR"
SUMMARY="$DIR/a1_ppl_summary.tsv"
echo -e "model\tenv\tppl\tstderr" > "$SUMMARY"

run_one() {
  local model_key=$1 env_val=$2
  local out="$DIR/a1-${model_key}-q6kavx${env_val}.log"
  local model_path="${MODELS[$model_key]}"

  echo "[$(date '+%H:%M:%S')] START ${model_key} GGML_Q6_K_8X8_AVX=${env_val}" | tee -a "$DIR/a1_progress.log"
  local t0=$SECONDS

  GGML_Q6_K_8X8_AVX=$env_val \
    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
    numactl --interleave=all -- taskset -c 0-95 \
    "$BIN" -m "$model_path" -t 96 -fa 1 --no-mmap \
    -f "$WIKI" --chunks 32 \
    > "$out" 2>&1
  local rc=$?

  local elapsed=$((SECONDS - t0))
  # Extract Final estimate: line — typical format: "Final estimate: PPL = 11.1146 +/- 0.62405"
  local final_line=$(grep "Final estimate" "$out" | tail -1)
  local ppl=$(echo "$final_line" | grep -oP 'PPL = \K[0-9.]+' | head -1)
  local stderr=$(echo "$final_line" | grep -oP '\+/- \K[0-9.]+' | head -1)

  echo "[$(date '+%H:%M:%S')] END   ${model_key} env=${env_val} rc=${rc} elapsed=${elapsed}s ppl=${ppl} +/- ${stderr}" | tee -a "$DIR/a1_progress.log"
  echo -e "${model_key}\t${env_val}\t${ppl:-FAIL}\t${stderr:-}" >> "$SUMMARY"
}

for model_key in "${ORDER[@]}"; do
  for env_val in 0 1; do
    run_one "$model_key" "$env_val"
  done
  echo "[$(date '+%H:%M:%S')] === ${model_key} both env states complete ===" | tee -a "$DIR/a1_progress.log"
done

echo "[$(date '+%H:%M:%S')] === Phase A.1 ALL DONE ===" | tee -a "$DIR/a1_progress.log"
echo
echo "=== SUMMARY ==="
column -t -s$'\t' "$SUMMARY"
