#!/bin/bash
# Phase A.2 — Q6_K perf gate via llama-bench tg32 r=5 across 5 production models, 2 env states
# Runs ONLY after Phase A.1 PPL bit-exact gate passes for all 5 models
set -uo pipefail

DIR="/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-05-04-q6k-default-on-validation"
BIN="/mnt/raid0/llm/llama.cpp/build_libomp_pgo_use/bin/llama-bench"

declare -A MODELS=(
  [coder30b_q4km]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
  [gemma4_31b_q4km]="/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf"
  [supergemma4_31b_q4km]="/mnt/raid0/llm/models/SuperGemma4-31b-abliterated.Q4_K_M.gguf"
  [qwen3next_80b_a3b_q4km]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
  [reap246b_a35b_q4km]="/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf"
)

ORDER=(coder30b_q4km gemma4_31b_q4km supergemma4_31b_q4km qwen3next_80b_a3b_q4km reap246b_a35b_q4km)

cd "$DIR"
SUMMARY="$DIR/a2_perf_summary.tsv"
echo -e "model\tenv\tavg_ts\tstddev_ts\tn_runs" > "$SUMMARY"

run_one() {
  local model_key=$1 env_val=$2
  local out="$DIR/a2-${model_key}-q6kavx${env_val}.json"
  local model_path="${MODELS[$model_key]}"

  echo "[$(date '+%H:%M:%S')] START ${model_key} GGML_Q6_K_8X8_AVX=${env_val}" | tee -a "$DIR/a2_progress.log"
  local t0=$SECONDS

  GGML_Q6_K_8X8_AVX=$env_val \
    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
    numactl --interleave=all -- taskset -c 0-95 \
    "$BIN" -m "$model_path" -t 96 -fa 1 --mmap 0 \
    -p 0 -n 32 -r 5 \
    -o json > "$out" 2>"$DIR/a2-${model_key}-q6kavx${env_val}.stderr"
  local rc=$?

  local elapsed=$((SECONDS - t0))
  local avg_ts=$(jq -r '.[0].avg_ts // "FAIL"' "$out" 2>/dev/null)
  local stddev_ts=$(jq -r '.[0].stddev_ts // ""' "$out" 2>/dev/null)
  local n=$(jq -r '.[0] | (.samples_ts // []) | length' "$out" 2>/dev/null)

  echo "[$(date '+%H:%M:%S')] END   ${model_key} env=${env_val} rc=${rc} elapsed=${elapsed}s avg_ts=${avg_ts} +/- ${stddev_ts} n=${n}" | tee -a "$DIR/a2_progress.log"
  echo -e "${model_key}\t${env_val}\t${avg_ts}\t${stddev_ts}\t${n}" >> "$SUMMARY"
}

for model_key in "${ORDER[@]}"; do
  for env_val in 0 1; do
    run_one "$model_key" "$env_val"
  done
  echo "[$(date '+%H:%M:%S')] === ${model_key} both env states complete ===" | tee -a "$DIR/a2_progress.log"
done

echo "[$(date '+%H:%M:%S')] === Phase A.2 ALL DONE ===" | tee -a "$DIR/a2_progress.log"
echo
echo "=== SUMMARY ==="
column -t -s$'\t' "$SUMMARY"
