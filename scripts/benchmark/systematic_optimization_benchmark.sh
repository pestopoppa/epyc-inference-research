#!/bin/bash
# Note: NOT using set -e because bash arithmetic returns non-zero when value is 0
set -o pipefail

# Systematic Optimization Benchmark Script
# Tests all optimization techniques and combinations on all available models
# Avoids re-testing already completed combinations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

LLAMA_CPP="${LLAMA_CPP_BIN}"
RESULTS_DIR="${LOG_DIR}/benchmarks"
RESULTS_FILE="$RESULTS_DIR/systematic_optimization_$(date +%Y%m%d_%H%M%S).csv"
EXISTING_RESULTS="$RESULTS_DIR/optimization_results_20251215_045816.csv"
BENCH_TMP_DIR="${TMP_DIR}"

mkdir -p "$RESULTS_DIR" "$BENCH_TMP_DIR"

# Initialize results file
echo "timestamp,model,model_type,optimization,k_value,temp,experts,speed_tps,accept_pct,quality,notes" >"$RESULTS_FILE"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# Check if combination already tested
is_tested() {
  local model="$1"
  local opt="$2"
  local k="${3:-0}"
  local temp="${4:-0}"
  local experts="${5:-0}"

  local model_base

  model_base=$(basename "$model" .gguf)
  # Remove Q4_K_M suffix for matching
  model_base="${model_base%-Q4_K_M}"
  model_base="${model_base%-Instruct}"

  # Check in existing results (old format: timestamp,model,method,draft,prompt_type,status,...)
  if [ -f "$EXISTING_RESULTS" ]; then
    # Match model name and method
    if grep -qi "${model_base}.*,${opt}" "$EXISTING_RESULTS" 2>/dev/null; then
      return 0
    fi
    # Also check for hard_mask_N format
    if [[ "$opt" == "moe_experts" ]] && grep -qi "${model_base}.*hard_mask_${experts}" "$EXISTING_RESULTS" 2>/dev/null; then
      return 0
    fi
    # Check for lookup_hard_mask combo
    if [[ "$opt" == "moe_lookup_combo" ]] && grep -qi "${model_base}.*lookup_hard_mask_${experts}" "$EXISTING_RESULTS" 2>/dev/null; then
      return 0
    fi
  fi

  # Check in current results (new format)
  if [ -f "$RESULTS_FILE" ] && grep -qi "${model_base}.*${opt}.*${k}.*${temp}.*${experts}" "$RESULTS_FILE" 2>/dev/null; then
    return 0
  fi

  return 1
}

# Record result
record_result() {
  local model="$1"
  local model_type="$2"
  local opt="$3"
  local k="$4"
  local temp="$5"
  local experts="$6"
  local speed="$7"
  local accept="$8"
  local quality="$9"
  local notes="${10:-}"

  echo "$(date -Iseconds),$(basename "$model"),$model_type,$opt,$k,$temp,$experts,$speed,$accept,$quality,$notes" >>"$RESULTS_FILE"
}

# Run baseline benchmark
run_baseline() {
  local model="$1"
  local output
  output="$BENCH_TMP_DIR/baseline_$(basename "$model" .gguf).txt"

  timeout 180 "$LLAMA_CPP/llama-completion" \
    -m "$model" \
    -p "Write a function to calculate factorial:" \
    -t 96 \
    -n 50 \
    --temp 0 \
    >"$output" 2>&1 || true

  local speed

  speed=$(grep "eval time" "$output" | grep -oP '[\d.]+(?= tokens per second)' | tail -1)
  echo "${speed:-0}"
}

# Run speculative benchmark with K tuning
run_speculative() {
  local target="$1"
  local draft="$2"
  local k="$3"
  local temp="$4"
  local output
  output="$BENCH_TMP_DIR/spec_k${k}_t${temp}_$(basename "$target" .gguf).txt"

  timeout 300 env OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-speculative" \
    -m "$target" \
    -md "$draft" \
    -p "Write a function to calculate factorial:" \
    --draft-max "$k" \
    -t 96 \
    -n 100 \
    --temp "$temp" \
    >"$output" 2>&1 || true

  local speed

  speed=$(grep "decoded" "$output" | grep -oP '[\d.]+(?= t/s)' | tail -1)
  local accept
  accept=$(grep "accept" "$output" | grep -oP '[\d.]+(?=%)' | tail -1)
  echo "${speed:-0}:${accept:-0}"
}

# Run lookup + expert reduction combination
run_lookup_moe_combo() {
  local model="$1"
  local arch="$2"
  local experts="$3"
  local output
  output="$BENCH_TMP_DIR/lookup_moe_exp${experts}_$(basename "$model" .gguf).txt"

  timeout 180 env OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-lookup" \
    -m "$model" \
    -p "Write a Python function to sort a list:" \
    --override-kv "${arch}.expert_used_count=int:$experts" \
    --draft-max 16 \
    -t 96 \
    -n 100 \
    --temp 0 \
    >"$output" 2>&1 || true

  local speed

  speed=$(grep "decoded" "$output" | grep -oP '[\d.]+(?= t/s)' | tail -1)
  local accept
  accept=$(grep "accept" "$output" | grep -oP '[\d.]+(?=%)' | tail -1)

  # Quality check - look for valid code output
  local quality="good"
  if grep -qE "def |function |public " "$output" 2>/dev/null; then
    quality="good"
  elif grep -qE "error|invalid|garbage|\?\?\?" "$output" 2>/dev/null; then
    quality="bad"
  fi

  echo "${speed:-0}:${accept:-0}:$quality"
}

# Quality comparison function
run_quality_comparison() {
  local model="$1"
  local opt_args="$2"
  local opt_name="$3"
  local prompt="Write a Python function to check if a number is prime:"

  local baseline_out
  baseline_out="$BENCH_TMP_DIR/quality_baseline_$(basename "$model" .gguf).txt"
  local opt_out
  opt_out="$BENCH_TMP_DIR/quality_${opt_name}_$(basename "$model" .gguf).txt"

  # Run baseline
  timeout 120 "$LLAMA_CPP/llama-completion" \
    -m "$model" -p "$prompt" -t 96 -n 80 --temp 0 \
    >"$baseline_out" 2>&1 || true

  # Run optimized
  timeout 120 "$LLAMA_CPP/llama-completion" \
    -m "$model" -p "$prompt" $opt_args -t 96 -n 80 --temp 0 \
    >"$opt_out" 2>&1 || true

  # Extract code content (after "assistant" line)
  local baseline_code
  baseline_code=$(sed -n '/^assistant$/,/^>/p' "$baseline_out" | grep -v "^assistant$" | grep -v "^>")
  local opt_code
  opt_code=$(sed -n '/^assistant$/,/^>/p' "$opt_out" | grep -v "^assistant$" | grep -v "^>")

  # Quality assessment
  if [[ -z "$opt_code" ]] || echo "$opt_code" | grep -qE "error|invalid|\?\?\?|garbage"; then
    echo "bad"
  elif echo "$opt_code" | grep -qE "def |return |if "; then
    echo "good"
  else
    echo "unknown"
  fi
}

# Run MoE expert tuning
run_moe_experts() {
  local model="$1"
  local arch="$2"
  local experts="$3"
  local output
  output="$BENCH_TMP_DIR/moe_exp${experts}_$(basename "$model" .gguf).txt"

  timeout 180 "$LLAMA_CPP/llama-completion" \
    -m "$model" \
    -p "Write a function to calculate factorial:" \
    --override-kv "${arch}.expert_used_count=int:$experts" \
    -t 96 \
    -n 50 \
    --temp 0 \
    >"$output" 2>&1 || true

  local speed

  speed=$(grep "eval time" "$output" | grep -oP '[\d.]+(?= tokens per second)' | tail -1)

  # Check quality
  local quality="good"
  if grep -q "invalid\|error\|garbage" "$output" 2>/dev/null; then
    quality="bad"
  fi

  echo "${speed:-0}:$quality"
}

# Run lookup benchmark
run_lookup() {
  local model="$1"
  local task="$2"
  local output
  output="$BENCH_TMP_DIR/lookup_${task}_$(basename "$model" .gguf).txt"

  local prompt
  case "$task" in
    summarize) prompt="Summarize this text: The quick brown fox jumps over the lazy dog. This is a test sentence for summarization." ;;
    code) prompt="Write a Python function to sort a list:" ;;
    *) prompt="Hello, how are you?" ;;
  esac

  timeout 180 env OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-lookup" \
    -m "$model" \
    -p "$prompt" \
    --draft-max 16 \
    -t 96 \
    -n 50 \
    --temp 0 \
    >"$output" 2>&1 || true

  local speed

  speed=$(grep "decoded" "$output" | grep -oP '[\d.]+(?= t/s)' | tail -1)
  local accept
  accept=$(grep "accept" "$output" | grep -oP '[\d.]+(?=%)' | tail -1)
  echo "${speed:-0}:${accept:-0}"
}

# Detect model type (dense or MoE) and architecture
detect_model_type() {
  local model="$1"
  local name
  name=$(basename "$model")

  if [[ "$name" == *"A3B"* ]] || [[ "$name" == *"A22B"* ]] || [[ "$name" == *"A35B"* ]] || [[ "$name" == *"moe"* ]] || [[ "$name" == *"MoE"* ]]; then
    # Detect specific MoE architecture
    if [[ "$name" == *"Qwen3-VL"* ]]; then
      echo "moe:qwen3vlmoe"
    elif [[ "$name" == *"Qwen3-Coder"* ]] || [[ "$name" == *"Qwen3-Next"* ]]; then
      echo "moe:qwen3moe"
    elif [[ "$name" == *"GLM"* ]]; then
      echo "moe:glm4moe"
    else
      echo "moe:unknown"
    fi
  else
    echo "dense:none"
  fi
}

# Find compatible draft model
find_draft_model() {
  local target="$1"
  local name
  name=$(basename "$target")

  # Qwen models use Qwen draft
  if [[ "$name" == *"Qwen"* ]]; then
    local draft="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"
    if [ -f "$draft" ]; then
      echo "$draft"
      return
    fi
  fi

  echo ""
}

# No draft model filter - test all models for potential optimization

# Check if model baseline is in research report
is_in_research_report() {
  local model="$1"
  local model_base
  model_base=$(basename "$model" .gguf)
  model_base="${model_base%-Q4_K_M}"
  model_base="${model_base%-Instruct}"

  if grep -qi "$model_base" "${LOG_DIR}/research_report.md" 2>/dev/null; then
    return 0
  fi
  return 1
}

# Main benchmark loop
main() {
  log "=== Systematic Optimization Benchmark ==="
  log "Results: $RESULTS_FILE"

  # Discover models
  log "Discovering models..."

  declare -a MODELS=()

  # Add all Q4_K_M models
  while IFS= read -r -d '' model; do
    MODELS+=("$model")
  done < <(find "${MODEL_BASE}" -name "*Q4_K_M.gguf" -print0 2>/dev/null | head -z -n 30)

  log "Found ${#MODELS[@]} models"

  local tested=0
  local skipped=0

  for model in "${MODELS[@]}"; do
    local model_name
    model_name=$(basename "$model" .gguf)
    local type_info
    type_info=$(detect_model_type "$model")
    local model_type="${type_info%%:*}"
    local arch="${type_info##*:}"

    log ""
    log "=== Testing: $model_name ($model_type) ==="

    # 1. Baseline (skip if in research report or CSV)
    if is_in_research_report "$model" || is_tested "$model" "baseline" 0 0 0; then
      log "Baseline already documented, skipping"
      ((skipped++))
    else
      log "Running baseline..."
      local speed
      speed=$(run_baseline "$model")
      record_result "$model" "$model_type" "baseline" 0 0 0 "$speed" "N/A" "good" ""
      ((tested++))
    fi

    # 2. Lookup decoding
    for task in summarize code; do
      if ! is_tested "$model" "lookup_$task" 16 0 0; then
        log "Running lookup ($task)..."
        local result
        result=$(run_lookup "$model" "$task")
        local speed="${result%%:*}"
        local accept="${result##*:}"
        record_result "$model" "$model_type" "lookup_$task" 16 0 0 "$speed" "$accept" "good" ""
        ((tested++))
      else
        ((skipped++))
      fi
    done

    # 3. Speculative decoding (dense models only)
    if [ "$model_type" == "dense" ]; then
      local draft
      draft=$(find_draft_model "$model")
      if [ -n "$draft" ]; then
        # K tuning with temp=0
        for k in 8 12 16 24; do
          if ! is_tested "$model" "speculative" "$k" 0 0; then
            log "Running speculative K=$k temp=0..."
            local result
            result=$(run_speculative "$model" "$draft" "$k" 0)
            local speed="${result%%:*}"
            local accept="${result##*:}"
            record_result "$model" "$model_type" "speculative" "$k" 0 0 "$speed" "$accept" "good" ""
            ((tested++))
          else
            ((skipped++))
          fi
        done

        # Temperature tuning (with optimal K=12)
        for temp in 0.3 0.5 0.7; do
          if ! is_tested "$model" "speculative" 12 "$temp" 0; then
            log "Running speculative K=12 temp=$temp..."
            local result
            result=$(run_speculative "$model" "$draft" 12 "$temp")
            local speed="${result%%:*}"
            local accept="${result##*:}"
            record_result "$model" "$model_type" "speculative" 12 "$temp" 0 "$speed" "$accept" "good" ""
            ((tested++))
          else
            ((skipped++))
          fi
        done
      fi
    fi

    # 4. MoE expert tuning
    if [ "$model_type" == "moe" ] && [ "$arch" != "unknown" ]; then
      for experts in 6 4 3; do
        if ! is_tested "$model" "moe_experts" 0 0 "$experts"; then
          log "Running MoE experts=$experts..."
          local result
          result=$(run_moe_experts "$model" "$arch" "$experts")
          local speed="${result%%:*}"
          local quality="${result##*:}"
          record_result "$model" "$model_type" "moe_experts" 0 0 "$experts" "$speed" "N/A" "$quality" ""
          ((tested++))
        else
          ((skipped++))
        fi
      done

      # 5. Combination: MoE experts + lookup
      for experts in 4 3; do
        if ! is_tested "$model" "moe_lookup_combo" 16 0 "$experts"; then
          log "Running MoE experts=$experts + lookup combo..."
          local result
          result=$(run_lookup_moe_combo "$model" "$arch" "$experts")
          local speed
          speed=$(echo "$result" | cut -d: -f1)
          local accept
          accept=$(echo "$result" | cut -d: -f2)
          local quality
          quality=$(echo "$result" | cut -d: -f3)
          record_result "$model" "$model_type" "moe_lookup_combo" 16 0 "$experts" "$speed" "$accept" "$quality" ""
          ((tested++))
        else
          ((skipped++))
        fi
      done

      # 6. Quality verification for expert reduction (run once per model)
      if ! is_tested "$model" "quality_check" 0 0 4; then
        log "Running quality check for 4 experts..."
        local quality
        quality=$(run_quality_comparison "$model" "--override-kv ${arch}.expert_used_count=int:4" "exp4")
        record_result "$model" "$model_type" "quality_check" 0 0 4 0 "N/A" "$quality" "baseline_comparison"
        ((tested++))
      else
        ((skipped++))
      fi
    fi
  done

  log ""
  log "=== Benchmark Complete ==="
  log "Tests run: $tested"
  log "Skipped (already tested): $skipped"
  log "Results saved to: $RESULTS_FILE"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
