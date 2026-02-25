#!/bin/bash
# Master Script: Run Long Context Benchmarks on All Models
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
OUTPUT_DIR="/mnt/raid0/llm/tmp/long_context_rubric_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$OUTPUT_DIR/benchmark_summary_$TIMESTAMP.txt"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "LONG CONTEXT MODEL BENCHMARK SUITE"
echo "Date: $(date)"
echo "=============================================="

# Models with good context length support
declare -a MODELS=(
  # Large context models
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf|Qwen2.5-72B-Instruct|dense"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf|Qwen2.5-Coder-32B|dense"

  # MoE models
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf|Qwen3-Next-80B-A3B|qwen3moe"
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf|Qwen3-30B-A3B|qwen3moe"

  # Smaller models for comparison
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf|Llama-3.1-70B|dense"
)

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

  "$SCRIPT_DIR/run_long_context_rubric.sh" "$model_path" "$model_name" "$arch"
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

# Generate summary
{
  echo "Long Context Benchmark Summary"
  echo "Generated: $(date)"
  echo "=============================================="
  echo ""
  echo "Models Tested: $((PASSED + FAILED))"
  echo "Passed: $PASSED"
  echo "Failed: $FAILED"
  echo ""
  echo "Speed Summary:"
  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name arch <<<"$model_spec"
    echo "--- $model_name ---"
    for f in "$OUTPUT_DIR"/${model_name}_*.txt; do
      if [[ -f "$f" ]]; then
        test_name=$(basename "$f" .txt | sed "s/${model_name}_//")
        speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
        echo "  $test_name: $speed"
      fi
    done
  done
} >"$SUMMARY_FILE"

echo ""
echo "Summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
