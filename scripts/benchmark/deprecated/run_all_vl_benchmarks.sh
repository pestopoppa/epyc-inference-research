#!/bin/bash
# Run VL quality+speed benchmarks on all available VL models
# Tests each model with baseline + optimization configurations where applicable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VL_RUBRIC="$SCRIPT_DIR/run_vl_rubric.sh"

echo "=============================================="
echo "VL Model Benchmark Suite"
echo "Date: $(date)"
echo "=============================================="

# Define all VL models with their mmproj files and architecture
# Format: model_path|mmproj_path|architecture
declare -A VL_MODELS=(
  # Small/Fast models (dense)
  ["Qwen3-VL-4B-Q4"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-4B-Instruct-GGUF/Qwen3-VL-4B-Instruct-Q4_K_M.gguf|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-4B-Instruct-GGUF/mmproj-Qwen3-VL-4B-Instruct-F16.gguf|dense"
  ["Qwen2.5-VL-7B-Q4"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf|dense"
  ["Qwen3-VL-8B-Q4"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q4_K_M.gguf|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3-VL-8B-Instruct-F16.gguf|dense"

  # Medium models (MoE - will test baseline + moe4)
  ["Qwen3-VL-30B-A3B-Q4"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf|qwen3vlmoe"

  # Large models (skip for now - too slow)
  # ["Qwen3-VL-235B-Thinking"]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-VL-235B-A22B-Thinking-GGUF/Qwen3-VL-235B-A22B-Thinking-Q4_K_S-00001-of-00003.gguf|/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-VL-235B-A22B-Thinking-GGUF/mmproj-F32.gguf|qwen3vlmoe"
)

# Order of testing (smallest to largest)
MODEL_ORDER=(
  "Qwen3-VL-4B-Q4"
  "Qwen2.5-VL-7B-Q4"
  "Qwen3-VL-8B-Q4"
  "Qwen3-VL-30B-A3B-Q4"
)

# Run tests
for model_name in "${MODEL_ORDER[@]}"; do
  if [[ -v VL_MODELS[$model_name] ]]; then
    IFS='|' read -r model_path mmproj_path arch <<<"${VL_MODELS[$model_name]}"

    echo ""
    echo "######################################################"
    echo "# Testing: $model_name"
    echo "# Architecture: $arch"
    echo "######################################################"

    if [[ -f "$model_path" ]] && [[ -f "$mmproj_path" ]]; then
      "$VL_RUBRIC" "$model_path" "$mmproj_path" "$model_name" "$arch"
    else
      echo "SKIPPED: Model or mmproj file not found"
      echo "  Model: $model_path"
      echo "  MMProj: $mmproj_path"
    fi
  fi
done

echo ""
echo "=============================================="
echo "ALL VL BENCHMARKS COMPLETE"
echo "=============================================="
echo ""
echo "Results in: /mnt/raid0/llm/tmp/vl_rubric_results/"
echo ""
echo "Summary of all models (all configurations):"
for model_name in "${MODEL_ORDER[@]}"; do
  if [[ -v VL_MODELS[$model_name] ]]; then
    IFS='|' read -r _ _ arch <<<"${VL_MODELS[$model_name]}"

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
      for f in /mnt/raid0/llm/tmp/vl_rubric_results/${model_name}_${cfg}_*.txt; do
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
for model_name in "${MODEL_ORDER[@]}"; do
  if [[ -v VL_MODELS[$model_name] ]]; then
    IFS='|' read -r _ _ arch <<<"${VL_MODELS[$model_name]}"

    if [[ "$arch" == "qwen3vlmoe" ]]; then
      echo ""
      echo "=== $model_name: baseline vs moe4 ==="
      echo "-------------------------------------------"
      printf "%-20s %-15s %-15s %-10s\n" "Test" "baseline" "moe4" "Speedup"
      echo "-------------------------------------------"

      for test in t1_q1_ocr t1_q2_shapes t1_q3_icon t2_q1_chart t2_q2_invoice t2_q3_code t3_q1_math t3_q2_flowchart t3_q3_diff t3_q4_puzzle; do
        base_file="/mnt/raid0/llm/tmp/vl_rubric_results/${model_name}_baseline_${test}.txt"
        moe_file="/mnt/raid0/llm/tmp/vl_rubric_results/${model_name}_moe4_${test}.txt"

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
