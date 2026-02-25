#!/bin/bash
# =============================================================================
# DRY RUN: Test that all benchmark models can be launched reliably
# =============================================================================
# Dynamically discovers all GGUF models and tests each one
#
# Usage: ./dry_run_all_models.sh [--skip-large] [--verbose] [--pattern PATTERN]
# =============================================================================
set -uo pipefail # Note: removed -e to handle failures gracefully

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

LLAMA_COMPLETION="${LLAMA_CPP_BIN}/llama-completion"
TIMEOUT_SECONDS=60
RESULTS_FILE="${TMP_DIR}/dry_run_results.txt"
DRY_RUN_MODEL_BASE="${MODEL_BASE}"

# Parse args
SKIP_LARGE=false
VERBOSE=false
PATTERN="*.gguf"
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-large)
      SKIP_LARGE=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p /mnt/raid0/llm/tmp

# Large models to skip (substring match)
LARGE_MODELS=(
  "70B"
  "72B"
  "80B"
  "235B"
  "480B"
)

# Architecture detection based on model name
detect_arch() {
  local model_path="$1"
  local model_name
  model_name=$(basename "$model_path")

  if [[ "$model_name" == *"Qwen3-"* ]] && [[ "$model_name" == *"-A3B-"* ]] && [[ "$model_name" != *"VL-"* ]]; then
    echo "qwen3moe"
  elif [[ "$model_name" == *"Qwen3-VL-"* ]] && [[ "$model_name" == *"-A"*"B-"* ]]; then
    echo "qwen3vlmoe"
  elif [[ "$model_name" == *"Qwen3-Next-"* ]]; then
    echo "qwen3next"
  elif [[ "$model_name" == *"Mixtral"* ]]; then
    echo "mixtral"
  else
    echo "dense"
  fi
}

echo "=============================================="
echo "DRY RUN: Testing All Benchmark Models"
echo "Date: $(date)"
echo "Model base: $MODEL_BASE"
echo "=============================================="
echo ""

# Write header to results file
echo "# Dry Run Results - $(date)" >"$RESULTS_FILE"
echo "" >>"$RESULTS_FILE"

PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

# Test function
test_model() {
  local model_path="$1"
  local model_name
  model_name=$(basename "$(dirname "$model_path")")/$(basename "$model_path")
  ((TOTAL++))

  # Skip mmproj (vision projector) files
  if [[ "$model_path" == *"mmproj"* ]]; then
    [[ "$VERBOSE" == "true" ]] && echo "  SKIP (mmproj): $model_name"
    ((SKIPPED++))
    return 0
  fi

  # Skip split models (part 2+)
  if [[ "$model_path" == *"-00002-"* ]] || [[ "$model_path" == *"-00003-"* ]]; then
    [[ "$VERBOSE" == "true" ]] && echo "  SKIP (split): $model_name"
    ((SKIPPED++))
    return 0
  fi

  # Check if large model should be skipped
  if [[ "$SKIP_LARGE" == "true" ]]; then
    for large in "${LARGE_MODELS[@]}"; do
      if [[ "$model_path" == *"$large"* ]]; then
        echo "  SKIP (large): $model_name"
        echo "SKIP|LARGE|$model_name" >>"$RESULTS_FILE"
        ((SKIPPED++))
        return 0
      fi
    done
  fi

  # Detect architecture
  local arch
  arch=$(detect_arch "$model_path")

  # Prepare test prompt
  local prompt_file="/mnt/raid0/llm/tmp/dry_run_prompt.txt"
  echo "What is 2+2? Answer with just the number." >"$prompt_file"

  # Build command based on architecture
  local moe_override=""
  case "$arch" in
    qwen3moe) moe_override="--override-kv qwen3moe.expert_used_count=int:4" ;;
    qwen3vlmoe) moe_override="--override-kv qwen3vlmoe.expert_used_count=int:4" ;;
    qwen3next) moe_override="--override-kv qwen3next.expert_used_count=int:2" ;;
  esac

  # Run test (llama-completion with --no-conversation)
  local output_file="/mnt/raid0/llm/tmp/dry_run_output.txt"
  local start_time
  start_time=$(date +%s.%N)

  [[ "$VERBOSE" == "true" ]] && echo "  Testing: $model_name (arch: $arch)"

  if timeout "$TIMEOUT_SECONDS" env OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_COMPLETION" \
    -m "$model_path" \
    $moe_override \
    -t 96 -n 32 --temp 0 --no-conversation \
    -f "$prompt_file" \
    >"$output_file" 2>&1; then

    local end_time

    end_time=$(date +%s.%N)
    local duration
    duration=$(echo "$end_time - $start_time" | bc)

    # Extract speed
    local speed
    speed=$(grep "eval time" "$output_file" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

    echo "  PASS: $model_name ($speed in ${duration}s)"
    echo "PASS|$speed|$duration|$model_name" >>"$RESULTS_FILE"
    ((PASSED++))
  else
    local exit_code=$?
    local error_hint=""

    # Check for common errors
    if grep -q "invalid argument" "$output_file" 2>/dev/null; then
      error_hint="invalid_argument"
    elif grep -q "error:" "$output_file" 2>/dev/null; then
      error_hint=$(grep "error:" "$output_file" | head -1 | sed 's/.*error: //' | head -c 40)
    elif [[ $exit_code -eq 124 ]]; then
      error_hint="timeout"
    else
      error_hint="exit_$exit_code"
    fi

    echo "  FAIL: $model_name ($error_hint)"
    echo "FAIL|$error_hint|$model_name" >>"$RESULTS_FILE"

    if [[ "$VERBOSE" == "true" ]]; then
      echo "  --- Error output (last 10 lines) ---"
      tail -10 "$output_file" 2>/dev/null | sed 's/^/    /'
      echo "  -----------------------------------"
    fi

    ((FAILED++))
  fi
}

# Discover and test all models
echo "Discovering models in $MODEL_BASE..."
echo ""

# Use for loop with glob instead of find (more reliable)
shopt -s globstar nullglob
for model_path in "$MODEL_BASE"/**/*.gguf; do
  test_model "$model_path"
done
shopt -u globstar nullglob

# Summary
echo ""
echo "=============================================="
echo "DRY RUN COMPLETE"
echo "=============================================="
echo "TOTAL:   $TOTAL"
echo "PASSED:  $PASSED"
echo "FAILED:  $FAILED"
echo "SKIPPED: $SKIPPED"
echo ""
echo "Results saved to: $RESULTS_FILE"

# Show failed models
if [[ $FAILED -gt 0 ]]; then
  echo ""
  echo "FAILED MODELS:"
  grep "^FAIL" "$RESULTS_FILE" | while IFS='|' read -r status reason model; do
    echo "  - $model ($reason)"
  done
  exit 1
fi

exit 0
