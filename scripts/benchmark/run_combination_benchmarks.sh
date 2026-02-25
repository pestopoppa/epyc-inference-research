#!/bin/bash
set -x

# Run combination optimization benchmarks
# Tests combinations of orthogonal optimizations:
#
# VALID COMBINATIONS:
# - Lookup + Hard Mask (MoE): Prompt lookup drafts, hard mask reduces experts during verify
# - Draft + Hard Mask (MoE): External draft model, hard mask reduces experts during verify
#
# INVALID COMBINATIONS (commented out):
# - Lookup + Layer Skip: --n-layer-exit applies to verification, destroys quality
# - Draft + Layer Skip: Same issue - layer skip on verify produces garbage
#
# NOTE: CAS-Spec/CLaSp (self-speculative with early exit) requires specialized
# implementation where DRAFT uses early exit and VERIFY uses full layers.
# Current llama.cpp --n-layer-exit applies to the model globally, not per-phase.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

LLAMA_CPP="${LLAMA_CPP_BIN}"
BENCH_LOG_DIR="${LOG_DIR}/benchmarks"
RESULTS_CSV="$BENCH_LOG_DIR/optimization_results_20251215_045816.csv"

mkdir -p "$BENCH_LOG_DIR"

check_existing() {
  local model="$1"
  local method="$2"
  local prompt_type="$3"
  if [[ -f "$RESULTS_CSV" ]]; then
    if grep -q ",${model},${method},.*,${prompt_type},success," "$RESULTS_CSV" 2>/dev/null; then
      local speed
      speed=$(grep ",${model},${method},.*,${prompt_type},success," "$RESULTS_CSV" | tail -1 | cut -d',' -f7)
      if [[ "$speed" != "0" ]] && [[ -n "$speed" ]]; then
        echo "SKIP: $model/$method/$prompt_type already done ($speed t/s)"
        return 0
      fi
    fi
  fi
  return 1
}

# Lookup + Hard Mask (MoE models)
run_lookup_hard_mask() {
  local name="$1"
  local model_path="$2"
  local n_expert="$3"
  local prompt_type="$4"
  local prompt="$5"

  [[ -f "$model_path" ]] || return 1
  check_existing "$name" "lookup_hard_mask_${n_expert}" "$prompt_type" && return

  echo "=== Lookup + Hard Mask ($n_expert experts): $name ($prompt_type) ==="
  local tmpfile
  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  local output
  output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-lookup" \
    -m "$model_path" \
    -f "$tmpfile" \
    --draft-max 16 \
    --moe-n-expert "$n_expert" \
    -t 96 \
    -n 200 \
    --temp 0 2>&1) || true

  rm -f "$tmpfile"

  local speed

  speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
  local accept
  accept=$(echo "$output" | grep -oP 'accept:\s*[\d.]+%' | grep -oP '[\d.]+' | tail -1)
  [[ -z "$speed" ]] && speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)

  if [[ -n "$speed" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,lookup_hard_mask_${n_expert},none,$prompt_type,success,$speed,${accept:-0},0" >>"$RESULTS_CSV"
    echo "  -> $speed t/s"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,lookup_hard_mask_${n_expert},none,$prompt_type,failed,0,0,0" >>"$RESULTS_CSV"
    echo "  -> FAILED"
  fi
}

# Lookup + Layer Skip (Dense models)
run_lookup_layer_skip() {
  local name="$1"
  local model_path="$2"
  local n_layers="$3"
  local prompt_type="$4"
  local prompt="$5"

  [[ -f "$model_path" ]] || return 1
  check_existing "$name" "lookup_layer_skip_${n_layers}" "$prompt_type" && return

  echo "=== Lookup + Layer Skip ($n_layers layers): $name ($prompt_type) ==="
  local tmpfile
  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  local output
  output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-lookup" \
    -m "$model_path" \
    -f "$tmpfile" \
    --draft-max 16 \
    --n-layer-exit "$n_layers" \
    -t 96 \
    -n 200 \
    --temp 0 2>&1) || true

  rm -f "$tmpfile"

  local speed

  speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
  local accept
  accept=$(echo "$output" | grep -oP 'accept:\s*[\d.]+%' | grep -oP '[\d.]+' | tail -1)
  [[ -z "$speed" ]] && speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)

  if [[ -n "$speed" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,lookup_layer_skip_${n_layers},none,$prompt_type,success,$speed,${accept:-0},0" >>"$RESULTS_CSV"
    echo "  -> $speed t/s"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,lookup_layer_skip_${n_layers},none,$prompt_type,failed,0,0,0" >>"$RESULTS_CSV"
    echo "  -> FAILED"
  fi
}

# External Draft + Layer Skip (Dense models)
run_draft_layer_skip() {
  local name="$1"
  local model_path="$2"
  local draft_name="$3"
  local draft_path="$4"
  local n_layers="$5"
  local prompt_type="$6"
  local prompt="$7"

  [[ -f "$model_path" ]] || return 1
  [[ -f "$draft_path" ]] || return 1
  check_existing "$name" "draft_layer_skip_${n_layers}" "$prompt_type" && return

  echo "=== Draft + Layer Skip ($n_layers layers): $name + $draft_name ($prompt_type) ==="
  local tmpfile
  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  local output
  output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-speculative" \
    -m "$model_path" \
    -md "$draft_path" \
    -f "$tmpfile" \
    --draft-max 16 \
    --n-layer-exit "$n_layers" \
    -t 96 \
    -n 200 \
    --temp 0 2>&1) || true

  rm -f "$tmpfile"

  local speed

  speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
  local accept
  accept=$(echo "$output" | grep -oP 'accept:\s*[\d.]+%' | grep -oP '[\d.]+' | tail -1)
  [[ -z "$speed" ]] && speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)

  if [[ -n "$speed" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,draft_layer_skip_${n_layers},$draft_name,$prompt_type,success,$speed,${accept:-0},0" >>"$RESULTS_CSV"
    echo "  -> $speed t/s (accept: ${accept:-0}%)"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,draft_layer_skip_${n_layers},$draft_name,$prompt_type,failed,0,0,0" >>"$RESULTS_CSV"
    echo "  -> FAILED"
  fi
}

# Test prompts
PROMPT_SUMMARIZE="Summarize the following text in 3 bullet points:\n\nThe AMD EPYC 9655 Turin processor represents a significant advancement in server CPU technology. Built on the Zen 5 architecture, it features 96 cores and 192 threads, offering unprecedented parallel processing capabilities. The processor supports DDR5-5600 memory across 12 channels, providing approximately 460 GB/s of memory bandwidth. One of the key improvements in Zen 5 is the true 512-bit AVX-512 implementation, which is not double-pumped like previous generations. This makes it particularly effective for AI inference workloads that can leverage wide vector operations. The processor is manufactured on TSMC's 4nm process node, offering improved power efficiency compared to previous generations."

PROMPT_CODE="Write a Python function that implements binary search on a sorted array. Include docstring and type hints."

echo "=========================================="
echo "Running combination optimization benchmarks"
echo "=========================================="

# ============================================================
# COMBINATION 1: Lookup + Hard Mask (MoE models)
# ============================================================
echo ""
echo "=== COMBINATION 1: Lookup + Hard Mask (MoE) ==="

for prompt_type in summarize code; do
  case $prompt_type in
    summarize) prompt="$PROMPT_SUMMARIZE" ;;
    code) prompt="$PROMPT_CODE" ;;
  esac

  run_lookup_hard_mask "Qwen3-VL-30B-A3B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf" 4 "$prompt_type" "$prompt"
  run_lookup_hard_mask "Qwen3-Coder-30B-A3B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" 4 "$prompt_type" "$prompt"
  run_lookup_hard_mask "Qwen3-Next-80B-A3B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf" 4 "$prompt_type" "$prompt"
  run_lookup_hard_mask "Qwen3-235B-A22B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf" 4 "$prompt_type" "$prompt"
  run_lookup_hard_mask "GLM-4.6-355B" "/mnt/raid0/llm/lmstudio/models/unsloth/GLM-4.6-GGUF/GLM-4.6-Q4_K_S-00001-of-00005.gguf" 4 "$prompt_type" "$prompt"
  run_lookup_hard_mask "Qwen3-Coder-480B-A35B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf" 4 "$prompt_type" "$prompt"
done

# ============================================================
# COMBINATION 2: Draft + Hard Mask (MoE models with external draft)
# ============================================================
echo ""
echo "=== COMBINATION 2: Draft + Hard Mask (MoE) ==="

# Note: This combines external draft speculation with MoE expert reduction
# Both optimizations are orthogonal and should compound

DRAFT_QWEN3_17B="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf"

# Draft + Hard Mask for MoE models (using Qwen3-1.7B as draft for Qwen3 MoE targets)
run_draft_hard_mask() {
  local name="$1"
  local model_path="$2"
  local draft_name="$3"
  local draft_path="$4"
  local n_expert="$5"
  local prompt_type="$6"
  local prompt="$7"

  [[ -f "$model_path" ]] || return 1
  [[ -f "$draft_path" ]] || return 1
  check_existing "$name" "draft_hard_mask_${n_expert}" "$prompt_type" && return

  echo "=== Draft + Hard Mask ($n_expert experts): $name + $draft_name ($prompt_type) ==="
  local tmpfile
  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  local output
  output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-speculative" \
    -m "$model_path" \
    -md "$draft_path" \
    -f "$tmpfile" \
    --draft-max 16 \
    --moe-n-expert "$n_expert" \
    -t 96 \
    -n 200 \
    --temp 0 2>&1) || true

  rm -f "$tmpfile"

  local speed

  speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
  local accept
  accept=$(echo "$output" | grep -oP 'accept:\s*[\d.]+%' | grep -oP '[\d.]+' | tail -1)
  [[ -z "$speed" ]] && speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)

  if [[ -n "$speed" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,draft_hard_mask_${n_expert},$draft_name,$prompt_type,success,$speed,${accept:-0},0" >>"$RESULTS_CSV"
    echo "  -> $speed t/s (accept: ${accept:-0}%)"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,draft_hard_mask_${n_expert},$draft_name,$prompt_type,failed,0,0,0" >>"$RESULTS_CSV"
    echo "  -> FAILED"
  fi
}

for prompt_type in summarize code; do
  case $prompt_type in
    summarize) prompt="$PROMPT_SUMMARIZE" ;;
    code) prompt="$PROMPT_CODE" ;;
  esac

  # Qwen3-Coder-30B-A3B with draft + hard mask
  run_draft_hard_mask "Qwen3-Coder-30B-A3B" \
    "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" \
    "Qwen3-1.7B" "$DRAFT_QWEN3_17B" 4 "$prompt_type" "$prompt"
done

# ============================================================
# DISABLED: Layer Skip combinations (produces garbage output)
# ============================================================
# These combinations are INVALID because --n-layer-exit applies to the
# verification phase, not the draft phase. This destroys output quality.
#
# For proper CAS-Spec/CLaSp implementation, we would need:
# 1. Draft phase with early exit (fast, low quality)
# 2. Verify phase with full layers (slow, high quality)
#
# Current llama.cpp --n-layer-exit applies globally, not per-phase.
# See Quality Evaluation Results in research_report.md for details.
#
# DISABLED: run_lookup_layer_skip (Lookup + Layer Skip)
# DISABLED: run_draft_layer_skip (Draft + Layer Skip)

echo ""
echo "=========================================="
echo "Combination benchmarks complete!"
echo "Results: $RESULTS_CSV"
echo "=========================================="
