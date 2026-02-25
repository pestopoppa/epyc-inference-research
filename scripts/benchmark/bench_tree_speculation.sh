#!/bin/bash
# Tree Speculation Benchmark Script
#
# Tests tree-based speculative decoding with varying parameters:
# - n_parallel (number of tree branches)
# - p_split (probability threshold for splitting)
#
# Usage: ./bench_tree_speculation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Configuration (use env.sh values with fallbacks)
LLAMA_BIN="${LLAMA_CPP_BIN}"
MODEL_DIR="${MODELS_DIR}"
TREE_LOG_DIR="${LOG_DIR}/tree_speculation"
THREADS="${THREADS:-96}"

# Models
TARGET_MODEL="${TARGET_MODEL:-${MODEL_DIR}/Qwen2.5-Coder-32B-Q4_K_M.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-${MODEL_DIR}/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf}"

# Test prompt (code generation task)
PROMPT="Write a Python function to implement quicksort with detailed comments explaining each step:"

# Benchmark parameters
N_PREDICT=256
DRAFT_MAX=16

# Tree speculation parameters to test
N_PARALLEL_VALUES=(1 2 4 8)
P_SPLIT_VALUES=(0.05 0.1 0.2 0.3)

mkdir -p "$TREE_LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${TREE_LOG_DIR}/results_${TIMESTAMP}.csv"

echo "Tree Speculation Benchmark"
echo "=========================="
echo ""
echo "Target model: $TARGET_MODEL"
echo "Draft model: $DRAFT_MODEL"
echo "Threads: $THREADS"
echo "Draft max: $DRAFT_MAX"
echo "Results: $RESULTS_FILE"
echo ""

# Write CSV header
echo "n_parallel,p_split,tokens_generated,time_sec,tokens_per_sec,accepted,drafted,acceptance_rate" >"$RESULTS_FILE"

# Baseline: no tree speculation (n_parallel=1)
echo "=== Baseline (n_parallel=1, no tree splitting) ==="
for p_split in "${P_SPLIT_VALUES[@]}"; do
  echo -n "  p_split=$p_split: "

  OUTPUT=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_BIN/llama-speculative" \
    -m "$TARGET_MODEL" \
    -md "$DRAFT_MODEL" \
    --draft-max "$DRAFT_MAX" \
    -np 1 \
    --draft-p-split "$p_split" \
    -t "$THREADS" \
    -n "$N_PREDICT" \
    -p "$PROMPT" \
    --no-display-prompt \
    2>&1)

  # Parse output for metrics
  TOKENS=$(echo "$OUTPUT" | grep -oP 'generated \K\d+(?= tokens)' || echo "0")
  TIME=$(echo "$OUTPUT" | grep -oP '\d+\.\d+(?= seconds)' | tail -1 || echo "0")
  TPS=$(echo "$OUTPUT" | grep -oP '\d+\.\d+(?= tokens per second)' || echo "0")
  ACCEPTED=$(echo "$OUTPUT" | grep -oP 'accepted: \K\d+' || echo "0")
  DRAFTED=$(echo "$OUTPUT" | grep -oP 'drafted: \K\d+' || echo "0")

  if [ "$DRAFTED" != "0" ]; then
    ACCEPT_RATE=$(echo "scale=4; $ACCEPTED / $DRAFTED" | bc)
  else
    ACCEPT_RATE="0"
  fi

  echo "$TPS t/s (accepted: $ACCEPTED/$DRAFTED = $ACCEPT_RATE)"
  echo "1,$p_split,$TOKENS,$TIME,$TPS,$ACCEPTED,$DRAFTED,$ACCEPT_RATE" >>"$RESULTS_FILE"
done

echo ""

# Tree speculation tests
for n_parallel in "${N_PARALLEL_VALUES[@]}"; do
  if [ "$n_parallel" -eq 1 ]; then
    continue # Skip baseline, already done
  fi

  echo "=== Tree speculation (n_parallel=$n_parallel) ==="

  for p_split in "${P_SPLIT_VALUES[@]}"; do
    echo -n "  p_split=$p_split: "

    OUTPUT=$(OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_BIN/llama-speculative" \
      -m "$TARGET_MODEL" \
      -md "$DRAFT_MODEL" \
      --draft-max "$DRAFT_MAX" \
      -np "$n_parallel" \
      --draft-p-split "$p_split" \
      -t "$THREADS" \
      -n "$N_PREDICT" \
      -p "$PROMPT" \
      --no-display-prompt \
      2>&1)

    TOKENS=$(echo "$OUTPUT" | grep -oP 'generated \K\d+(?= tokens)' || echo "0")
    TIME=$(echo "$OUTPUT" | grep -oP '\d+\.\d+(?= seconds)' | tail -1 || echo "0")
    TPS=$(echo "$OUTPUT" | grep -oP '\d+\.\d+(?= tokens per second)' || echo "0")
    ACCEPTED=$(echo "$OUTPUT" | grep -oP 'accepted: \K\d+' || echo "0")
    DRAFTED=$(echo "$OUTPUT" | grep -oP 'drafted: \K\d+' || echo "0")

    if [ "$DRAFTED" != "0" ]; then
      ACCEPT_RATE=$(echo "scale=4; $ACCEPTED / $DRAFTED" | bc)
    else
      ACCEPT_RATE="0"
    fi

    echo "$TPS t/s (accepted: $ACCEPTED/$DRAFTED = $ACCEPT_RATE)"
    echo "$n_parallel,$p_split,$TOKENS,$TIME,$TPS,$ACCEPTED,$DRAFTED,$ACCEPT_RATE" >>"$RESULTS_FILE"
  done
  echo ""
done

echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
cat "$RESULTS_FILE" | column -t -s,
