#!/bin/bash
# Run VL models on Twyne whitepaper figure pages for author evaluation
# Usage: ./run_twyne_figures.sh [4b|30b|both] [--all]
#
# By default, only processes pages with figures (from OCR analysis):
# Pages 3, 6, 7, 9, 13, 19, 20, 22, 25, 29, 32
# Use --all to process all 32 pages

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Paths
LLAMA_MTMD="${LLAMA_CPP_BIN}/llama-mtmd-cli"
TWYNE_PAGES="${PROJECT_ROOT}/tmp/twyne_pages"
OUTPUT_DIR="${PROJECT_ROOT}/benchmarks/results/twyne_figure_analysis"

# Model paths
MODEL_4B="${MODEL_BASE}/lmstudio-community/Qwen3-VL-4B-Instruct-GGUF/Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_4B="${MODEL_BASE}/lmstudio-community/Qwen3-VL-4B-Instruct-GGUF/mmproj-Qwen3-VL-4B-Instruct-F16.gguf"

MODEL_30B="${MODEL_BASE}/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
MMPROJ_30B="${MODEL_BASE}/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"

# Configuration
MAX_TOKENS=1024
TIMEOUT_SEC=300 # 5 minutes per figure
THREADS=96

# Pages with figures (from OCR analysis)
FIGURE_PAGES=(3 6 7 9 13 19 20 22 25 29 32)

# Standard prompt for figure analysis
PROMPT='Analyze this figure from a DeFi whitepaper. Describe:
1. What type of visualization is this? (diagram, plot, chart, etc.)
2. What are the key components or data series shown?
3. What is the main insight or takeaway?
4. Any specific values or relationships you can identify?'

# Parse arguments
RUN_MODE="${1:-both}"
RUN_ALL=false
if [[ "${2:-}" == "--all" ]] || [[ "${1:-}" == "--all" ]]; then
  RUN_ALL=true
  if [[ "${1:-}" == "--all" ]]; then
    RUN_MODE="${2:-both}"
  fi
fi

run_model() {
  local model_name="$1"
  local model_path="$2"
  local mmproj_path="$3"
  local extra_args="${4:-}"

  echo "=============================================="
  echo "Running $model_name on Twyne figure pages"
  echo "=============================================="

  # Build list of pages to process
  local pages_to_process=()
  if [[ "$RUN_ALL" == "true" ]]; then
    for i in $(seq -w 1 32); do
      pages_to_process+=("$i")
    done
    echo "Processing ALL 32 pages"
  else
    for p in "${FIGURE_PAGES[@]}"; do
      pages_to_process+=("$(printf "%03d" "$p")")
    done
    echo "Processing ${#FIGURE_PAGES[@]} figure pages: ${FIGURE_PAGES[*]}"
  fi
  echo ""

  for page_num in "${pages_to_process[@]}"; do
    page="$TWYNE_PAGES/page_${page_num}.png"
    output_file="$OUTPUT_DIR/${model_name}_page_${page_num}.txt"

    # Skip if file doesn't exist
    if [[ ! -f "$page" ]]; then
      echo "  [SKIP] Page $page_num not found"
      continue
    fi

    # Skip if already processed
    if [[ -f "$output_file" ]] && [[ -s "$output_file" ]] && ! grep -q "TIMEOUT" "$output_file"; then
      echo "  [SKIP] Page $page_num already processed"
      continue
    fi

    echo "  [RUN] Page $page_num..."

    # Run with timeout
    if timeout "$TIMEOUT_SEC" numactl --interleave=all \
      "$LLAMA_MTMD" \
      -m "$model_path" \
      --mmproj "$mmproj_path" \
      --image "$page" \
      -p "$PROMPT" \
      -n "$MAX_TOKENS" \
      -t "$THREADS" \
      --flash-attn on \
      $extra_args \
      2>&1 | tee "$output_file.tmp"; then

      # Move temp to final
      mv "$output_file.tmp" "$output_file"
      echo "  [DONE] Page $page_num saved"
    else
      echo "  [TIMEOUT] Page $page_num timed out after ${TIMEOUT_SEC}s"
      echo "TIMEOUT after ${TIMEOUT_SEC}s" >"$output_file"
    fi
  done

  echo ""
  echo "$model_name completed. Results in $OUTPUT_DIR/"
}

# Main execution
case "$RUN_MODE" in
  4b)
    run_model "4b" "$MODEL_4B" "$MMPROJ_4B"
    ;;
  30b)
    # 30B uses MoE4 expert reduction for speed
    run_model "30b_moe4" "$MODEL_30B" "$MMPROJ_30B" "--override-kv qwen3vlmoe.expert_used_count=int:4"
    ;;
  both)
    run_model "4b" "$MODEL_4B" "$MMPROJ_4B"
    run_model "30b_moe4" "$MODEL_30B" "$MMPROJ_30B" "--override-kv qwen3vlmoe.expert_used_count=int:4"
    ;;
  *)
    echo "Usage: $0 [4b|30b|both]"
    exit 1
    ;;
esac

echo "=============================================="
echo "Figure analysis complete!"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="
