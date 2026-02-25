#!/bin/bash
# Run Thinking model quality+speed benchmarks on all available thinking models
# Tests each model with baseline + optimization configurations where applicable
#
# Architecture: This wrapper determines configurations using the shared library
# and calls the rubric with EACH config individually for better progress tracking.
#
# NOTE: Model paths and configs should come from the registry when available.
# This script maintains hardcoded fallbacks for models not yet in registry.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THINKING_RUBRIC="$SCRIPT_DIR/run_thinking_rubric.sh"

# Source shared libraries
if [[ -f "$SCRIPT_DIR/lib/registry_reader.sh" ]]; then
  source "$SCRIPT_DIR/lib/registry_reader.sh"
fi
if [[ -f "$SCRIPT_DIR/lib/optimization_configs.sh" ]]; then
  source "$SCRIPT_DIR/lib/optimization_configs.sh"
fi

echo "=============================================="
echo "Thinking Model Benchmark Suite"
echo "Date: $(date)"
echo "=============================================="

# Define all thinking models with their architecture
# Format: model_path|architecture
# UPDATED 2025-12-17: Fixed paths to match actual file locations
declare -A THINKING_MODELS=(
  # === Dense models (baseline + spec decode if compatible) ===
  ["Qwen3-4B-Thinking-Q8"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-4B-Thinking-2507-GGUF/Qwen3-4B-Thinking-2507-Q8_0.gguf|dense"

  # DeepSeek R1 Distill Qwen family (dense, Qwen2.5 architecture - spec decode compatible)
  ["DeepSeek-R1-Distill-Qwen-1.5B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf|dense"
  ["DeepSeek-R1-Distill-Qwen-7B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf|dense"
  ["DeepSeek-R1-Distill-Qwen-14B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf|dense"
  ["DeepSeek-R1-Distill-Qwen-32B"]="/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf|dense"

  # DeepSeek R1 Distill Llama family (dense)
  ["DeepSeek-R1-Distill-Llama-8B"]="/mnt/raid0/llm/lmstudio/models/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf|dense"

  # DeepSeek R1 0528 (Qwen3 base, dense)
  ["DeepSeek-R1-0528-Qwen3-8B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf|dense"

  # === MoE models (baseline + expert reduction) ===
  ["Qwen3-30B-A3B-Thinking-Q4"]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/Qwen3-30B-A3B-Thinking-2507-Q4_K_S.gguf|qwen3moe"
  ["Qwen3-30B-A3B-Thinking-Q8"]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf|qwen3moe"

  # === SSM+MoE hybrid (expert reduction ONLY, NO spec decode) ===
  ["Qwen3-Next-80B-A3B-Thinking"]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF/Qwen3-Next-80B-A3B-Thinking-Q4_K_S.gguf|qwen3next"
)

# Order of testing (smallest to largest, fastest first)
MODEL_ORDER=(
  "DeepSeek-R1-Distill-Qwen-1.5B"
  "Qwen3-4B-Thinking-Q8"
  "DeepSeek-R1-Distill-Qwen-7B"
  "DeepSeek-R1-Distill-Llama-8B"
  "DeepSeek-R1-0528-Qwen3-8B"
  "DeepSeek-R1-Distill-Qwen-14B"
  "Qwen3-30B-A3B-Thinking-Q4"
  "Qwen3-30B-A3B-Thinking-Q8"
  "DeepSeek-R1-Distill-Qwen-32B"
  "Qwen3-Next-80B-A3B-Thinking"
)

# Track which models were tested
declare -a TESTED_MODELS=()

# Run tests - iterate over models and configs
for model_name in "${MODEL_ORDER[@]}"; do
  if [[ -v THINKING_MODELS[$model_name] ]]; then
    IFS='|' read -r model_path arch <<<"${THINKING_MODELS[$model_name]}"

    echo ""
    echo "######################################################"
    echo "# Testing: $model_name"
    echo "# Architecture: $arch"
    echo "######################################################"

    if [[ ! -f "$model_path" ]]; then
      echo "SKIPPED: Model not found"
      echo "  Path: $model_path"
      continue
    fi

    # Use shared library to determine configs for this architecture
    if type setup_configs &>/dev/null; then
      setup_configs "$model_name" "$arch"
      echo "# Configurations: ${CONFIGS[*]}"

      # Iterate over each config and call rubric
      # Pass DRAFT_MODEL_PATH via env var so rubric doesn't need to rediscover
      for config in "${CONFIGS[@]}"; do
        echo ""
        echo ">>> Running $model_name with config: $config"
        DRAFT_MODEL="$DRAFT_MODEL_PATH" "$THINKING_RUBRIC" "$model_path" "$model_name" "$arch" "$config"
      done
    else
      # Fallback: let rubric determine configs internally
      echo "# Using rubric internal config selection"
      "$THINKING_RUBRIC" "$model_path" "$model_name" "$arch"
    fi

    TESTED_MODELS+=("$model_name")
  fi
done

echo ""
echo "=============================================="
echo "ALL THINKING BENCHMARKS COMPLETE"
echo "=============================================="
echo ""
echo "Results in: /mnt/raid0/llm/tmp/thinking_rubric_results/"
echo ""
echo "Models tested: ${TESTED_MODELS[*]}"
echo ""
echo "Summary of all models (all configurations):"

for model_name in "${TESTED_MODELS[@]}"; do
  if [[ -v THINKING_MODELS[$model_name] ]]; then
    IFS='|' read -r _ arch <<<"${THINKING_MODELS[$model_name]}"

    # Determine configs tested (must match rubric scripts - includes lookup in full mode)
    case "$arch" in
      qwen3moe | qwen3next | qwen3vlmoe)
        configs=("baseline" "moe2" "moe4" "moe6" "moe8" "lookup" "moe2_lookup" "moe4_lookup" "moe6_lookup" "moe8_lookup")
        ;;
      *)
        configs=("baseline")
        ;;
    esac

    for cfg in "${configs[@]}"; do
      echo ""
      echo "=== $model_name ($cfg) ==="
      for f in /mnt/raid0/llm/tmp/thinking_rubric_results/${model_name}_${cfg}_*.txt; do
        if [[ -f "$f" ]]; then
          test_name=$(basename "$f" .txt | sed "s/${model_name}_${cfg}_//")
          speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "N/A")
          echo "  $test_name: $speed"
        fi
      done
    done
  fi
done

# Generate comparison table for MoE models (all optimization configs)
echo ""
echo "=============================================="
echo "MoE OPTIMIZATION COMPARISON"
echo "=============================================="
for model_name in "${TESTED_MODELS[@]}"; do
  if [[ -v THINKING_MODELS[$model_name] ]]; then
    IFS='|' read -r _ arch <<<"${THINKING_MODELS[$model_name]}"

    case "$arch" in
      qwen3moe | qwen3next | qwen3vlmoe)
        echo ""
        echo "=== $model_name: Expert Reduction Sweep ==="
        echo "--------------------------------------------------------------"
        printf "%-20s %-10s %-10s %-10s %-10s %-10s\n" "Test" "baseline" "moe2" "moe4" "moe6" "moe8"
        echo "--------------------------------------------------------------"

        for test in t1_q1_algorithm t1_q2_threadsafe t2_q1_dict_reuse t2_q2_cache_bug t2_q3_api_design t3_q1_dependency t3_q2_vector_clock t3_q3_type_system t3_q4_probability; do
          declare -A speeds
          for cfg in baseline moe2 moe4 moe6 moe8; do
            result_file="/mnt/raid0/llm/tmp/thinking_rubric_results/${model_name}_${cfg}_${test}.txt"
            if [[ -f "$result_file" ]]; then
              speeds[$cfg]=$(grep "eval time" "$result_file" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "N/A")
            else
              speeds[$cfg]="N/A"
            fi
          done
          printf "%-20s %-10s %-10s %-10s %-10s %-10s\n" \
            "$test" "${speeds[baseline]}" "${speeds[moe2]}" "${speeds[moe4]}" "${speeds[moe6]}" "${speeds[moe8]}"
          unset speeds
          declare -A speeds
        done

        echo ""
        echo "=== $model_name: Expert Reduction + Lookup ==="
        echo "--------------------------------------------------------------"
        printf "%-20s %-10s %-10s %-10s %-10s %-10s\n" "Test" "lookup" "m2+lkup" "m4+lkup" "m6+lkup" "m8+lkup"
        echo "--------------------------------------------------------------"

        for test in t1_q1_algorithm t1_q2_threadsafe t2_q1_dict_reuse t2_q2_cache_bug t2_q3_api_design t3_q1_dependency t3_q2_vector_clock t3_q3_type_system t3_q4_probability; do
          declare -A speeds
          for cfg in lookup moe2_lookup moe4_lookup moe6_lookup moe8_lookup; do
            result_file="/mnt/raid0/llm/tmp/thinking_rubric_results/${model_name}_${cfg}_${test}.txt"
            if [[ -f "$result_file" ]]; then
              speeds[$cfg]=$(grep "eval time" "$result_file" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "N/A")
            else
              speeds[$cfg]="N/A"
            fi
          done
          printf "%-20s %-10s %-10s %-10s %-10s %-10s\n" \
            "$test" "${speeds[lookup]}" "${speeds[moe2_lookup]}" "${speeds[moe4_lookup]}" "${speeds[moe6_lookup]}" "${speeds[moe8_lookup]}"
          unset speeds
          declare -A speeds
        done
        ;;
    esac
  fi
done
