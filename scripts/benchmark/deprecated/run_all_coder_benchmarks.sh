#!/bin/bash
# Run Coder quality+speed benchmarks on all available coder models
# Tests each model with baseline + optimization configurations where applicable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODER_RUBRIC="$SCRIPT_DIR/run_coder_rubric.sh"

echo "=============================================="
echo "Coder Model Benchmark Suite"
echo "Date: $(date)"
echo "=============================================="

# Define all coder models with their architecture
# Format: model_path|architecture
declare -A CODER_MODELS=(
  # Dense models (baseline only)
  ["Qwen2.5-Coder-7B-Q4"]="/mnt/raid0/llm/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf|dense"
  ["Qwen2.5-Coder-14B-Q4"]="/mnt/raid0/llm/models/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf|dense"
  ["Qwen2.5-Coder-32B-Q4"]="/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf|dense"

  # MoE models (baseline + moe4)
  ["Qwen3-Coder-53B-A3B-Q4"]="/mnt/raid0/llm/lmstudio/models/mradermacher/Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L-i1-GGUF/Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L.i1-Q4_K_M.gguf|qwen3moe"

  # Alternative paths if models exist
  ["Qwen2.5-Coder-7B-Q4-LM"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf|dense"
  ["Qwen2.5-Coder-32B-Q4-LM"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf|dense"
)

# Order of testing (smallest to largest, prefer primary paths)
MODEL_ORDER=(
  "Qwen2.5-Coder-7B-Q4"
  "Qwen2.5-Coder-14B-Q4"
  "Qwen2.5-Coder-32B-Q4"
  "Qwen3-Coder-53B-A3B-Q4"
)

# Track which models were tested
declare -a TESTED_MODELS=()

# Run tests
for model_name in "${MODEL_ORDER[@]}"; do
  if [[ -v CODER_MODELS[$model_name] ]]; then
    IFS='|' read -r model_path arch <<<"${CODER_MODELS[$model_name]}"

    echo ""
    echo "######################################################"
    echo "# Testing: $model_name"
    echo "# Architecture: $arch"
    echo "######################################################"

    if [[ -f "$model_path" ]]; then
      "$CODER_RUBRIC" "$model_path" "$model_name" "$arch"
      TESTED_MODELS+=("$model_name")
    else
      # Try alternative path (LM Studio)
      alt_name="${model_name}-LM"
      if [[ -v CODER_MODELS[$alt_name] ]]; then
        IFS='|' read -r alt_path alt_arch <<<"${CODER_MODELS[$alt_name]}"
        if [[ -f "$alt_path" ]]; then
          echo "Using alternative path: $alt_path"
          "$CODER_RUBRIC" "$alt_path" "$model_name" "$alt_arch"
          TESTED_MODELS+=("$model_name")
        else
          echo "SKIPPED: Model not found at primary or alternative path"
          echo "  Primary: $model_path"
          echo "  Alternative: $alt_path"
        fi
      else
        echo "SKIPPED: Model not found"
        echo "  Path: $model_path"
      fi
    fi
  fi
done

echo ""
echo "=============================================="
echo "ALL CODER BENCHMARKS COMPLETE"
echo "=============================================="
echo ""
echo "Results in: /mnt/raid0/llm/tmp/coder_rubric_results/"
echo ""
echo "Models tested: ${TESTED_MODELS[*]}"
echo ""
echo "Summary of all models (all configurations):"

for model_name in "${TESTED_MODELS[@]}"; do
  if [[ -v CODER_MODELS[$model_name] ]]; then
    IFS='|' read -r _ arch <<<"${CODER_MODELS[$model_name]}"

    # Determine configs tested (must match MOE_EXPERTS_FULL in rubric scripts)
    case "$arch" in
      qwen3moe | qwen3next | qwen3vlmoe)
        configs=("baseline" "moe2" "moe4" "moe6" "moe8")
        ;;
      *)
        configs=("baseline")
        ;;
    esac

    for cfg in "${configs[@]}"; do
      echo ""
      echo "=== $model_name ($cfg) ==="
      for f in /mnt/raid0/llm/tmp/coder_rubric_results/${model_name}_${cfg}_*.txt; do
        if [[ -f "$f" ]]; then
          test_name=$(basename "$f" .txt | sed "s/${model_name}_${cfg}_//")
          speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "N/A")
          echo "  $test_name: $speed"
        fi
      done
    done
  fi
done

# Generate comparison table for MoE models
echo ""
echo "=============================================="
echo "MoE OPTIMIZATION COMPARISON"
echo "=============================================="
for model_name in "${TESTED_MODELS[@]}"; do
  if [[ -v CODER_MODELS[$model_name] ]]; then
    IFS='|' read -r _ arch <<<"${CODER_MODELS[$model_name]}"

    if [[ "$arch" == "qwen3moe" ]]; then
      echo ""
      echo "=== $model_name: baseline vs moe4 ==="
      echo "-------------------------------------------"
      printf "%-20s %-15s %-15s %-10s\n" "Test" "baseline" "moe4" "Speedup"
      echo "-------------------------------------------"

      for test in t1_q1_factorial t1_q2_reverse_words t1_q3_stack t2_q1_palindrome t2_q2_async_fetch t2_q3_lru_cache t3_q1_ip_addresses t3_q2_rate_limiter t3_q3_bug_fix t3_q4_retry_decorator; do
        base_file="/mnt/raid0/llm/tmp/coder_rubric_results/${model_name}_baseline_${test}.txt"
        moe_file="/mnt/raid0/llm/tmp/coder_rubric_results/${model_name}_moe4_${test}.txt"

        base_speed="N/A"
        moe_speed="N/A"
        speedup="N/A"

        if [[ -f "$base_file" ]]; then
          base_speed=$(grep "eval time" "$base_file" 2>/dev/null | grep -oP '\d+\.\d+' | tail -1 || echo "N/A")
        fi
        if [[ -f "$moe_file" ]]; then
          moe_speed=$(grep "eval time" "$moe_file" 2>/dev/null | grep -oP '\d+\.\d+' | tail -1 || echo "N/A")
        fi

        # Calculate speedup if both values are numeric
        if [[ "$base_speed" != "N/A" ]] && [[ "$moe_speed" != "N/A" ]]; then
          speedup=$(echo "scale=2; $moe_speed / $base_speed" | bc)
          speedup="${speedup}x"
        fi

        printf "%-20s %-15s %-15s %-10s\n" "$test" "$base_speed" "$moe_speed" "$speedup"
      done
    fi
  fi
done
