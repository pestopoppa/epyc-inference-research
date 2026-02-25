#!/bin/bash
# Master Script: Run Agentic/Tool-Use Benchmarks on All Models
# Tests tool-calling capability critical for orchestration participation
# CRITICAL: Models must score 4+ on T1 to participate in orchestration
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
OUTPUT_DIR="/mnt/raid0/llm/tmp/agentic_rubric_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$OUTPUT_DIR/benchmark_summary_$TIMESTAMP.txt"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "AGENTIC/TOOL-USE MODEL BENCHMARK SUITE"
echo "Date: $(date)"
echo "=============================================="
echo ""
echo "CRITICAL: This benchmark determines orchestration eligibility!"
echo "Requirement: Score 4+ on T1 (all models), 4+ on T2 (orchestrators)"
echo ""

# Define models to test - ALL models that might participate in orchestration
# Format: "path|name|arch|role"
declare -a MODELS=(
  # Dense models - General/Instruct
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf|Qwen2.5-72B-Instruct|dense|architect"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf|Llama-3.1-70B-Instruct|dense|architect"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf|Llama-3-8B-Instruct|dense|worker"

  # Dense models - Coder (must handle tool schemas)
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf|Qwen2.5-Coder-32B|dense|coder_escalation"

  # MoE models (baseline + moe4)
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf|Qwen3-Next-80B-A3B|qwen3moe|architect"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf|Qwen3-30B-A3B|qwen3moe|escalation"
  "/mnt/raid0/llm/lmstudio/models/mradermacher/Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L-i1-GGUF/Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L.i1-Q4_K_M.gguf|Qwen3-Coder-53B-A3B|qwen3moe|coder_escalation"
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

  "$SCRIPT_DIR/run_agentic_rubric.sh" "$model_path" "$model_name" "$arch"

  return 0
}

# Run all models
PASSED=0
FAILED=0

for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_path model_name arch role <<<"$model_spec"

  echo ""
  echo ">>> Model intended role: $role"

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
echo "AGENTIC BENCHMARK SUITE COMPLETE"
echo "=============================================="
echo "Models tested: $((PASSED + FAILED))"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""
echo "Results saved to: $OUTPUT_DIR"

# Write summary file
{
  echo "Agentic/Tool-Use Benchmark Summary"
  echo "Generated: $(date)"
  echo "=============================================="
  echo ""
  echo "ORCHESTRATION ELIGIBILITY REQUIREMENTS:"
  echo "  - Worker: T1 score 4+"
  echo "  - Orchestrator: T1 score 5, T2 score 4+"
  echo "  - Architect: T1-T2 score 5, T3 score 4+"
  echo ""
  echo "Models Tested:"
  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name arch role <<<"$model_spec"
    echo "  $model_name: ${MODEL_RESULTS[$model_name]:-SKIP} (intended: $role)"
  done
  echo ""
  echo "Speed Summary by Model and Config:"
  echo ""

  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name arch role <<<"$model_spec"
    echo "--- $model_name ($arch) [intended: $role] ---"

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
    IFS='|' read -r model_path model_name arch role <<<"$model_spec"
    if [[ "$arch" == "qwen3moe" ]]; then
      echo ""
      echo "Model: $model_name"
      # Find a test that exists for both configs
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

  echo ""
  echo "=============================================="
  echo "ORCHESTRATION ELIGIBILITY ASSESSMENT"
  echo "=============================================="
  echo "(Manual review required - check T1/T2/T3 output quality)"
  echo ""
  echo "Review each model's output files to score quality."
  echo "Models scoring below thresholds should be excluded from orchestration."
} >"$SUMMARY_FILE"

echo ""
echo "Summary written to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
