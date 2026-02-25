#!/bin/bash
# Master Script: Run Math Benchmarks on All Models
# Tests mathematical reasoning for math_worker, math_specialist, math_oracle roles
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
OUTPUT_DIR="/mnt/raid0/llm/tmp/math_rubric_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$OUTPUT_DIR/benchmark_summary_$TIMESTAMP.txt"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MATH MODEL BENCHMARK SUITE"
echo "Date: $(date)"
echo "=============================================="

# Define models to test
# Format: "path|name|arch"
declare -a MODELS=(
  # Specialized Math Models (Dense)
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Math-7B-Instruct-GGUF/Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf|Qwen2.5-Math-7B|dense"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Math-72B-Instruct-GGUF/Qwen2.5-Math-72B-Instruct-Q4_K_M.gguf|Qwen2.5-Math-72B|dense"

  # Reasoning Models (Dense) - Strong at math
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf|DeepSeek-R1-Qwen-7B|dense"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf|DeepSeek-R1-Qwen-14B|dense"
  "/mnt/raid0/llm/lmstudio/models/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf|DeepSeek-R1-Llama-8B|dense"
  "/mnt/raid0/llm/lmstudio/models/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf|DeepSeek-R1-Llama-70B|dense"

  # General Models with Math Capability
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf|Qwen2.5-72B-Instruct|dense"

  # MoE Models (baseline + moe4)
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf|Qwen3-Next-80B-A3B|qwen3moe"
)

# Track results
declare -A MODEL_RESULTS

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

  "$SCRIPT_DIR/run_math_rubric.sh" "$model_path" "$model_name" "$arch"

  return 0
}

# Run all models
PASSED=0
FAILED=0

for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_path model_name arch <<<"$model_spec"

  if run_model "$model_path" "$model_name" "$arch"; then
    ((PASSED++)) || true
    MODEL_RESULTS["$model_name"]="PASS"
  else
    ((FAILED++)) || true
    MODEL_RESULTS["$model_name"]="FAIL (not found)"
  fi
done

# Generate summary
echo ""
echo "=============================================="
echo "MATH BENCHMARK SUITE COMPLETE"
echo "=============================================="
echo "Models tested: $((PASSED + FAILED))"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""
echo "Results saved to: $OUTPUT_DIR"

# Write summary file
{
  echo "Math Benchmark Summary"
  echo "Generated: $(date)"
  echo "=============================================="
  echo ""
  echo "Models Tested:"
  for model_name in "${!MODEL_RESULTS[@]}"; do
    echo "  $model_name: ${MODEL_RESULTS[$model_name]}"
  done
  echo ""
  echo "Speed Summary by Model and Config:"
  echo ""

  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name arch <<<"$model_spec"
    echo "--- $model_name ($arch) ---"

    # Determine configs based on arch (must match MOE_EXPERTS_FULL in rubric scripts)
    case "$arch" in
      qwen3moe | qwen3next | qwen3vlmoe) configs="baseline moe2 moe4 moe6 moe8" ;;
      *) configs="baseline" ;;
    esac

    for config in $configs; do
      echo "  Config: $config"
      for f in "$OUTPUT_DIR"/${model_name}_${config}_*.txt; do
        if [[ -f "$f" ]]; then
          test_name=$(basename "$f" .txt | sed "s/${model_name}_${config}_//")
          speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
          echo "    $test_name: $speed"
        fi
      done
    done
    echo ""
  done

  # MoE comparison table
  echo "=============================================="
  echo "MoE Speedup Analysis (baseline vs moe4)"
  echo "=============================================="
  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name arch <<<"$model_spec"
    if [[ "$arch" == "qwen3moe" ]]; then
      echo ""
      echo "Model: $model_name"
      for f in "$OUTPUT_DIR"/${model_name}_baseline_*.txt; do
        if [[ -f "$f" ]]; then
          test_name=$(basename "$f" .txt | sed "s/${model_name}_baseline_//")
          baseline_speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")
          moe4_file="$OUTPUT_DIR/${model_name}_moe4_${test_name}.txt"
          if [[ -f "$moe4_file" ]]; then
            moe4_speed=$(grep "eval time" "$moe4_file" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")
            if [[ "$baseline_speed" != "0" && "$moe4_speed" != "0" ]]; then
              speedup=$(echo "scale=2; $moe4_speed / $baseline_speed" | bc 2>/dev/null || echo "N/A")
              echo "  $test_name: ${baseline_speed} -> ${moe4_speed} t/s (${speedup}x)"
            fi
          fi
        fi
      done
    fi
  done

  # Specialized vs General comparison
  echo ""
  echo "=============================================="
  echo "SPECIALIZED vs GENERAL MODEL COMPARISON"
  echo "=============================================="
  echo "(Compare Qwen2.5-Math vs Qwen2.5-Instruct on same problems)"
  echo ""
  echo "Review output quality to determine if specialized math models"
  echo "outperform general models on T2-T3 problems."
} >"$SUMMARY_FILE"

echo ""
echo "Summary written to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
