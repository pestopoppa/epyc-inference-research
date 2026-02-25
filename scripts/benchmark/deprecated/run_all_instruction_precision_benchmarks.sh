#!/bin/bash
# Master Script: Run Instruction Precision Benchmarks on All Models
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
OUTPUT_DIR="/mnt/raid0/llm/tmp/instruction_precision_rubric_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$OUTPUT_DIR/benchmark_summary_$TIMESTAMP.txt"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "INSTRUCTION PRECISION MODEL BENCHMARK SUITE"
echo "Date: $(date)"
echo "=============================================="
echo ""
echo "This tests exact instruction following - critical for orchestration!"
echo ""

# All orchestration candidate models should be tested
declare -a MODELS=(
  # Dense models
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf|Qwen2.5-72B-Instruct|dense"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf|Qwen2.5-Coder-32B|dense"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf|Llama-3.1-70B|dense"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf|Llama-3-8B|dense"

  # Math models (need precise output)
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Math-7B-Instruct-GGUF/Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf|Qwen2.5-Math-7B|dense"

  # MoE models
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf|Qwen3-Next-80B-A3B|qwen3moe"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf|Qwen3-30B-A3B|qwen3moe"
  "/mnt/raid0/llm/lmstudio/models/mradermacher/Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L-i1-GGUF/Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L.i1-Q4_K_M.gguf|Qwen3-Coder-53B-A3B|qwen3moe"

  # Reasoning models
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf|DeepSeek-R1-Qwen-7B|dense"
)

declare -A MODEL_RESULTS
declare -A MODEL_SCORES

run_model() {
  local model_path="$1"
  local model_name="$2"
  local arch="$3"

  if [[ ! -f "$model_path" ]]; then
    echo "WARNING: Model not found: $model_path"
    return 1
  fi

  echo ""
  echo "=============================================="
  echo "Testing: $model_name ($arch)"
  echo "=============================================="

  "$SCRIPT_DIR/run_instruction_precision_rubric.sh" "$model_path" "$model_name" "$arch"
  return 0
}

PASSED=0
FAILED=0

for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_path model_name arch <<<"$model_spec"

  if run_model "$model_path" "$model_name" "$arch"; then
    ((PASSED++)) || true
    MODEL_RESULTS["$model_name"]="PASS"
  else
    ((FAILED++)) || true
    MODEL_RESULTS["$model_name"]="FAIL"
  fi
done

# Generate summary with scores
{
  echo "Instruction Precision Benchmark Summary"
  echo "Generated: $(date)"
  echo "=============================================="
  echo ""
  echo "SCORING KEY:"
  echo "  T1: Basic format compliance (3 tests)"
  echo "  T2: Complex constraints (4 tests)"
  echo "  T3: Adversarial compliance (4 tests)"
  echo ""
  echo "ORCHESTRATION REQUIREMENTS:"
  echo "  Workers: T1 100%, T2 75%+"
  echo "  Orchestrators: T1 100%, T2 100%, T3 75%+"
  echo ""
  echo "=============================================="
  echo "RESULTS BY MODEL"
  echo "=============================================="

  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name arch <<<"$model_spec"

    echo ""
    echo "--- $model_name ($arch) ---"

    # Count passes per tier
    for config in baseline moe4; do
      t1_pass=0
      t2_pass=0
      t3_pass=0
      t1_total=0
      t2_total=0
      t3_total=0

      for f in "$OUTPUT_DIR"/${model_name}_${config}_t1_*.txt; do
        if [[ -f "$f" ]]; then
          ((t1_total++)) || true
          grep -q "AUTO_SCORE: PASS" "$f" && ((t1_pass++)) || true
        fi
      done
      for f in "$OUTPUT_DIR"/${model_name}_${config}_t2_*.txt; do
        if [[ -f "$f" ]]; then
          ((t2_total++)) || true
          grep -q "AUTO_SCORE: PASS" "$f" && ((t2_pass++)) || true
        fi
      done
      for f in "$OUTPUT_DIR"/${model_name}_${config}_t3_*.txt; do
        if [[ -f "$f" ]]; then
          ((t3_total++)) || true
          grep -q "AUTO_SCORE: PASS" "$f" && ((t3_pass++)) || true
        fi
      done

      if [[ $t1_total -gt 0 ]]; then
        total_pass=$((t1_pass + t2_pass + t3_pass))
        total_tests=$((t1_total + t2_total + t3_total))
        echo "  $config: T1=$t1_pass/$t1_total T2=$t2_pass/$t2_total T3=$t3_pass/$t3_total (Total: $total_pass/$total_tests)"

        # Assess orchestration readiness
        if [[ $t1_pass -eq $t1_total ]] && [[ $t2_pass -ge 3 ]] && [[ $t3_pass -ge 3 ]]; then
          echo "    -> ORCHESTRATION READY"
        elif [[ $t1_pass -eq $t1_total ]] && [[ $t2_pass -ge 3 ]]; then
          echo "    -> WORKER READY"
        else
          echo "    -> NEEDS IMPROVEMENT"
        fi
      fi
    done
  done

  echo ""
  echo "=============================================="
  echo "LEADERBOARD (by total score)"
  echo "=============================================="

  # Create temp file for sorting
  tmpfile=$(mktemp)
  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name arch <<<"$model_spec"
    total=0
    for f in "$OUTPUT_DIR"/${model_name}_baseline_*.txt; do
      [[ -f "$f" ]] && grep -q "AUTO_SCORE: PASS" "$f" && ((total++)) || true
    done
    echo "$total $model_name" >>"$tmpfile"
  done
  sort -rn "$tmpfile" | while read score name; do
    echo "  $score/11 - $name"
  done
  rm "$tmpfile"

} >"$SUMMARY_FILE"

echo ""
echo "=============================================="
echo "INSTRUCTION PRECISION SUITE COMPLETE"
echo "=============================================="
cat "$SUMMARY_FILE"
