#!/bin/bash
# =============================================================================
# DYNAMIC DRAFT MODEL DISCOVERY AND OPTIMIZATION
# =============================================================================
# Automatically discovers potential draft models, matches them with compatible
# targets, and finds optimal K and temperature configurations.
#
# Features:
#   - Architecture-based compatibility matching (same tokenizer family)
#   - Quick acceptance rate validation (>30% = viable)
#   - K-value sweep: 4, 8, 16, 24, 32
#   - Temperature sweep: 0.0, 0.2, 0.4, 0.6
#   - Reports best configuration per target model
#
# Usage:
#   ./run_draft_discovery.sh [--quick] [--target MODEL] [--draft MODEL]
#
# Options:
#   --quick         Only test K=8,16 and temp=0.0,0.2 (faster)
#   --target MODEL  Only test specific target model
#   --draft MODEL   Only test specific draft model
#   --dry-run       Show what would be tested without running
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

OUTPUT_DIR="${TMP_DIR}/draft_discovery"
RESULTS_DIR="${PROJECT_ROOT}/benchmarks/results/draft_optimization"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/$TIMESTAMP"

# Binaries
LLAMA_SPEC="${LLAMA_CPP_BIN}/llama-speculative"
LLAMA_CLI="${LLAMA_CPP_BIN}/llama-cli"

# Model directories to scan
MODEL_DIRS=(
  "${MODEL_BASE}"
  "${MODELS_DIR}"
)

# Thresholds
MAX_DRAFT_SIZE_GB=5    # Max size for draft models (5GB ~ 3B Q8)
MIN_TARGET_SIZE_GB=8   # Min size for target models (8GB ~ 7B Q4)
MIN_ACCEPTANCE_RATE=25 # Minimum acceptance % to consider viable
QUICK_TEST_TOKENS=64   # Tokens for quick acceptance test
FULL_TEST_TOKENS=256   # Tokens for full benchmark

# Sweep parameters
K_VALUES_FULL=(4 8 16 24 32 48)
K_VALUES_QUICK=(8 16)

# Temperature optimization settings
TEMP_PRECISION=0.001
TEMP_SEARCH_MIN=0.0
TEMP_SEARCH_MAX=1.0
USE_TEMP_BINARY_SEARCH="${USE_TEMP_BINARY_SEARCH:-true}"

# Parse arguments
QUICK_MODE=false
TARGET_FILTER=""
DRAFT_FILTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)
      QUICK_MODE=true
      shift
      ;;
    --target)
      TARGET_FILTER="$2"
      shift 2
      ;;
    --draft)
      DRAFT_FILTER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Select sweep parameters based on mode
if [[ "$QUICK_MODE" == "true" ]]; then
  K_VALUES=("${K_VALUES_QUICK[@]}")
  TEMP_VALUES=("${TEMP_VALUES_QUICK[@]}")
else
  K_VALUES=("${K_VALUES_FULL[@]}")
  TEMP_VALUES=("${TEMP_VALUES_FULL[@]}")
fi

# Setup
mkdir -p "$RUN_DIR" "$RESULTS_DIR"
LOG_FILE="$RUN_DIR/discovery.log"

log() {
  local msg
  msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "$msg"
  echo "$msg" >>"$LOG_FILE"
}

# =============================================================================
# ARCHITECTURE COMPATIBILITY MATRIX
# =============================================================================
# Models must share the same tokenizer family for speculative decoding to work.
# This function returns a family ID for a given model name.

get_architecture_family() {
  local model_name="$1"

  # Qwen2.5 family (includes DeepSeek-R1-Distill-Qwen which uses Qwen2 tokenizer)
  if [[ "$model_name" =~ Qwen2\.5 ]] || [[ "$model_name" =~ Qwen2-5 ]]; then
    echo "qwen2.5"
    return
  fi

  # DeepSeek-R1-Distill-Qwen uses Qwen2 tokenizer
  if [[ "$model_name" =~ DeepSeek-R1-Distill-Qwen ]]; then
    echo "qwen2.5"
    return
  fi

  # Qwen3 MoE family (A3B, A22B, A35B indicators)
  if [[ "$model_name" =~ Qwen3 ]] && [[ "$model_name" =~ A[0-9]+B ]]; then
    echo "qwen3moe"
    return
  fi

  # Qwen3 dense family
  if [[ "$model_name" =~ Qwen3 ]] || [[ "$model_name" =~ qwen3 ]]; then
    echo "qwen3"
    return
  fi

  # Llama/Meta-Llama family
  if [[ "$model_name" =~ [Ll]lama ]] || [[ "$model_name" =~ Meta-Llama ]]; then
    echo "llama"
    return
  fi

  # Mistral family
  if [[ "$model_name" =~ [Mm]istral ]]; then
    echo "mistral"
    return
  fi

  # Mixtral (MoE, usually not good for spec decode as target)
  if [[ "$model_name" =~ [Mm]ixtral ]]; then
    echo "mixtral"
    return
  fi

  # Phi family
  if [[ "$model_name" =~ [Pp]hi ]]; then
    echo "phi"
    return
  fi

  # Gemma family
  if [[ "$model_name" =~ [Gg]emma ]]; then
    echo "gemma"
    return
  fi

  # Unknown
  echo "unknown"
}

# Check if two models are compatible for speculative decoding
are_compatible() {
  local draft_family="$1"
  local target_family="$2"

  # Same family = compatible
  if [[ "$draft_family" == "$target_family" ]]; then
    return 0
  fi

  # Qwen3 dense draft can work with Qwen3 MoE target (same tokenizer)
  if [[ "$draft_family" == "qwen3" ]] && [[ "$target_family" == "qwen3moe" ]]; then
    return 0
  fi

  # Not compatible
  return 1
}

# =============================================================================
# MODEL DISCOVERY
# =============================================================================

declare -A DRAFT_MODELS   # path -> name
declare -A TARGET_MODELS  # path -> name
declare -A MODEL_FAMILIES # path -> family

discover_models() {
  log "Discovering models..."

  for dir in "${MODEL_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
      continue
    fi

    while IFS= read -r -d '' model_path; do
      local filename
      filename=$(basename "$model_path")
      local size_bytes
      size_bytes=$(stat -c%s "$model_path" 2>/dev/null || echo 0)
      local size_gb
      size_gb=$(echo "scale=2; $size_bytes / 1024 / 1024 / 1024" | bc)

      # Extract model name (remove .gguf and quantization suffix)
      local model_name
      model_name=$(echo "$filename" | sed -E 's/\.(gguf|GGUF)$//' | sed -E 's/[-_](Q[0-9]_[A-Z0-9_]+|[Ff][0-9]+|BF16|bf16)$//')

      # Get architecture family
      local family
      family=$(get_architecture_family "$model_name")

      # Skip unknown architectures
      if [[ "$family" == "unknown" ]]; then
        continue
      fi

      # Categorize by size
      local is_draft
      is_draft=$(echo "$size_gb < $MAX_DRAFT_SIZE_GB" | bc -l)
      local is_target
      is_target=$(echo "$size_gb >= $MIN_TARGET_SIZE_GB" | bc -l)

      if [[ "$is_draft" == "1" ]]; then
        # Apply draft filter if specified
        if [[ -n "$DRAFT_FILTER" ]] && [[ ! "$model_path" =~ $DRAFT_FILTER ]]; then
          continue
        fi
        DRAFT_MODELS["$model_path"]="$model_name"
        MODEL_FAMILIES["$model_path"]="$family"
        log "  Found draft: $model_name ($size_gb GB) [$family]"
      fi

      if [[ "$is_target" == "1" ]]; then
        # Apply target filter if specified
        if [[ -n "$TARGET_FILTER" ]] && [[ ! "$model_path" =~ $TARGET_FILTER ]]; then
          continue
        fi
        TARGET_MODELS["$model_path"]="$model_name"
        MODEL_FAMILIES["$model_path"]="$family"
        log "  Found target: $model_name ($size_gb GB) [$family]"
      fi

    done < <(find "$dir" -name "*.gguf" -type f -print0 2>/dev/null)
  done

  log "Found ${#DRAFT_MODELS[@]} potential draft models"
  log "Found ${#TARGET_MODELS[@]} potential target models"
}

# =============================================================================
# ACCEPTANCE RATE TEST
# =============================================================================

test_acceptance_rate() {
  local target_path="$1"
  local draft_path="$2"
  local output_file="$3"

  local prompt="Write a Python function that calculates the factorial of a number recursively."

  timeout 120 OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_SPEC" \
    -m "$target_path" \
    -md "$draft_path" \
    --draft-max 8 \
    -t 96 -n "$QUICK_TEST_TOKENS" --temp 0.2 \
    -p "$prompt" \
    >"$output_file" 2>&1 || true

  # Extract acceptance rate from output
  # llama-speculative outputs: "draft acceptance rate: X.XX%"
  local acceptance
  acceptance=$(grep -oP 'draft acceptance rate: \K[\d.]+' "$output_file" 2>/dev/null | tail -1 || echo "0")

  # Also try alternative format: "accepted: X / Y (Z%)"
  if [[ "$acceptance" == "0" ]]; then
    acceptance=$(grep -oP 'accepted:.*\((\K[\d.]+)' "$output_file" 2>/dev/null | tail -1 || echo "0")
  fi

  echo "$acceptance"
}

# =============================================================================
# FULL OPTIMIZATION SWEEP WITH BINARY SEARCH TEMPERATURE
# =============================================================================

run_optimization_sweep() {
  local target_path="$1"
  local target_name="$2"
  local draft_path="$3"
  local draft_name="$4"
  local pair_dir="$5"

  log "    Running optimization sweep..."

  local best_speed=0
  local best_k=0
  local best_temp=0
  local best_acceptance=0

  local prompt="Write a comprehensive Python class that implements a binary search tree with insert, delete, search, and traversal methods. Include docstrings and type hints."

  # First, get baseline (no speculation)
  log "      Baseline test..."
  local baseline_file="$pair_dir/baseline.txt"

  timeout 180 OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CLI" \
    -m "$target_path" \
    -t 96 -n "$FULL_TEST_TOKENS" --temp 0.2 \
    -p "$prompt" \
    >"$baseline_file" 2>&1 || true

  local baseline_speed

  baseline_speed=$(grep "eval time" "$baseline_file" 2>/dev/null | grep -oP '[\d.]+(?= tokens per second)' | tail -1 || echo "0")
  log "      Baseline: ${baseline_speed} t/s"

  # Phase 1: Find best K with fixed temp=0.2
  log "      Phase 1: K-value sweep (temp=0.2)..."
  local best_k_phase1=8
  local best_speed_phase1=0

  for k in "${K_VALUES[@]}"; do
    local test_file="$pair_dir/k${k}_temp0.2.txt"

    timeout 180 OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_SPEC" \
      -m "$target_path" \
      -md "$draft_path" \
      --draft-max "$k" \
      -t 96 -n "$FULL_TEST_TOKENS" --temp 0.2 \
      -p "$prompt" \
      >"$test_file" 2>&1 || true

    local speed

    speed=$(grep "eval time" "$test_file" 2>/dev/null | grep -oP '[\d.]+(?= tokens per second)' | tail -1 || echo "0")
    log "        K=$k -> ${speed} t/s"

    if (($(echo "$speed > $best_speed_phase1" | bc -l))); then
      best_speed_phase1="$speed"
      best_k_phase1="$k"
    fi
  done

  log "      Best K from Phase 1: K=$best_k_phase1 (${best_speed_phase1} t/s)"

  # Phase 2: Binary search for optimal temperature with best K
  if [[ "$USE_TEMP_BINARY_SEARCH" == "true" ]] && [[ "$QUICK_MODE" != "true" ]]; then
    log "      Phase 2: Binary search for optimal temperature (K=$best_k_phase1)..."

    local low="$TEMP_SEARCH_MIN"
    local high="$TEMP_SEARCH_MAX"
    local iteration=0
    local max_iterations=12

    best_temp="0.2"
    best_speed="$best_speed_phase1"
    best_k="$best_k_phase1"

    # Helper function to test a temperature
    test_temp_spec() {
      local temp="$1"
      local temp_str
      temp_str=$(printf "%.4f" "$temp")
      local test_file="$pair_dir/k${best_k}_temp${temp_str}.txt"

      timeout 180 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_SPEC" \
        -m "$target_path" \
        -md "$draft_path" \
        --draft-max "$best_k" \
        -t 96 -n "$FULL_TEST_TOKENS" --temp "$temp" \
        -p "$prompt" \
        >"$test_file" 2>&1 || true

      grep "eval time" "$test_file" 2>/dev/null | grep -oP '[\d.]+(?= tokens per second)' | tail -1 || echo "0"
    }

    while (($(echo "$high - $low > $TEMP_PRECISION" | bc -l))) && [[ $iteration -lt $max_iterations ]]; do
      ((iteration++))

      local mid

      mid=$(echo "scale=6; ($low + $high) / 2" | bc)
      local q1
      q1=$(echo "scale=6; ($low + $mid) / 2" | bc)
      local q3
      q3=$(echo "scale=6; ($mid + $high) / 2" | bc)

      log "        Iteration $iteration: testing temps [$(printf '%.3f' $low), $(printf '%.3f' $q1), $(printf '%.3f' $mid), $(printf '%.3f' $q3), $(printf '%.3f' $high)]"

      local speed_low

      speed_low=$(test_temp_spec "$low")
      local speed_q1
      speed_q1=$(test_temp_spec "$q1")
      local speed_mid
      speed_mid=$(test_temp_spec "$mid")
      local speed_q3
      speed_q3=$(test_temp_spec "$q3")
      local speed_high
      speed_high=$(test_temp_spec "$high")

      # Find best
      local best_of_five="$low"
      local best_speed_five="$speed_low"

      for tv in "$q1:$speed_q1" "$mid:$speed_mid" "$q3:$speed_q3" "$high:$speed_high"; do
        local t="${tv%%:*}"
        local s="${tv##*:}"
        if (($(echo "$s > $best_speed_five" | bc -l))); then
          best_of_five="$t"
          best_speed_five="$s"
        fi
      done

      log "          Best: temp=$(printf '%.4f' $best_of_five) -> ${best_speed_five} t/s"

      if (($(echo "$best_speed_five > $best_speed" | bc -l))); then
        best_temp="$best_of_five"
        best_speed="$best_speed_five"
      fi

      # Narrow range
      if (($(echo "$best_of_five <= $q1" | bc -l))); then
        high="$mid"
      elif (($(echo "$best_of_five <= $mid" | bc -l))); then
        low=$(echo "scale=6; $low + ($mid - $low) / 4" | bc)
        high=$(echo "scale=6; $mid + ($high - $mid) / 4" | bc)
      elif (($(echo "$best_of_five <= $q3" | bc -l))); then
        low=$(echo "scale=6; $mid - ($mid - $low) / 4" | bc)
        high=$(echo "scale=6; $high - ($high - $mid) / 4" | bc)
      else
        low="$mid"
      fi
    done

    log "      Temperature search converged: temp=$(printf '%.4f' $best_temp)"
  else
    # Quick mode: just use best K with temp=0.2
    best_k="$best_k_phase1"
    best_speed="$best_speed_phase1"
    best_temp="0.2"
  fi

  # Get acceptance rate for best config
  local best_file
  best_file="$pair_dir/k${best_k}_temp$(printf '%.4f' "$best_temp").txt"
  if [[ -f "$best_file" ]]; then
    best_acceptance=$(grep -oP 'draft acceptance rate: \K[\d.]+' "$best_file" 2>/dev/null | tail -1 || echo "0")
  fi

  # Calculate speedup
  local speedup="0"
  if (($(echo "$baseline_speed > 0" | bc -l))); then
    speedup=$(echo "scale=2; $best_speed / $baseline_speed" | bc)
  fi

  log "    RESULT: K=$best_k, temp=$(printf '%.4f' $best_temp) -> ${best_speed} t/s (${speedup}x speedup)"

  # Return results
  echo "$baseline_speed|$best_speed|$best_k|$best_temp|$best_acceptance|$speedup"
}

# =============================================================================
# MAIN DISCOVERY AND OPTIMIZATION LOOP
# =============================================================================

main() {
  cat <<'EOF'

 ██████╗ ██████╗  █████╗ ███████╗████████╗
 ██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
 ██║  ██║██████╔╝███████║█████╗     ██║
 ██║  ██║██╔══██╗██╔══██║██╔══╝     ██║
 ██████╔╝██║  ██║██║  ██║██║        ██║
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝

 ██████╗ ██╗███████╗ ██████╗ ██████╗ ██╗   ██╗███████╗██████╗ ██╗   ██╗
 ██╔══██╗██║██╔════╝██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝
 ██║  ██║██║███████╗██║     ██║   ██║██║   ██║█████╗  ██████╔╝ ╚████╔╝
 ██║  ██║██║╚════██║██║     ██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗  ╚██╔╝
 ██████╔╝██║███████║╚██████╗╚██████╔╝ ╚████╔╝ ███████╗██║  ██║   ██║
 ╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝

EOF

  log "=============================================="
  log "DRAFT MODEL DISCOVERY AND OPTIMIZATION"
  log "Started: $(date)"
  log "Run ID: $TIMESTAMP"
  log "Mode: $(if $QUICK_MODE; then echo 'QUICK'; else echo 'FULL'; fi)"
  log "K values: ${K_VALUES[*]}"
  log "Temp values: ${TEMP_VALUES[*]}"
  log "=============================================="

  # Clear discovery best file for fresh run
  : >"${TMP_DIR}/draft_discovery_best.txt"

  # Discover models
  discover_models

  if [[ ${#DRAFT_MODELS[@]} -eq 0 ]]; then
    log "ERROR: No draft models found"
    exit 1
  fi

  if [[ ${#TARGET_MODELS[@]} -eq 0 ]]; then
    log "ERROR: No target models found"
    exit 1
  fi

  # Results tracking
  declare -a RESULTS
  local pairs_tested=0
  local viable_pairs=0

  # Test each compatible pair
  for target_path in "${!TARGET_MODELS[@]}"; do
    local target_name="${TARGET_MODELS[$target_path]}"
    local target_family="${MODEL_FAMILIES[$target_path]}"

    log ""
    log "=========================================="
    log "TARGET: $target_name [$target_family]"
    log "=========================================="

    local target_dir="$RUN_DIR/targets/${target_name//\//_}"
    mkdir -p "$target_dir"

    local target_best_speed=0
    local target_best_draft=""
    local target_best_k=0
    local target_best_temp=0
    local target_baseline=0

    for draft_path in "${!DRAFT_MODELS[@]}"; do
      local draft_name="${DRAFT_MODELS[$draft_path]}"
      local draft_family="${MODEL_FAMILIES[$draft_path]}"

      # Check compatibility
      if ! are_compatible "$draft_family" "$target_family"; then
        log "  Skipping $draft_name (incompatible: $draft_family vs $target_family)"
        continue
      fi

      log ""
      log "  Testing draft: $draft_name [$draft_family]"

      if [[ "$DRY_RUN" == "true" ]]; then
        log "    [DRY RUN] Would test $draft_name -> $target_name"
        continue
      fi

      local pair_dir="$target_dir/${draft_name//\//_}"
      mkdir -p "$pair_dir"

      ((pairs_tested++)) || true

      # Quick acceptance rate test
      log "    Quick acceptance test..."
      local acceptance_file="$pair_dir/acceptance_test.txt"
      local acceptance
      acceptance=$(test_acceptance_rate "$target_path" "$draft_path" "$acceptance_file")

      log "    Acceptance rate: ${acceptance}%"

      # Check if viable
      if (($(echo "$acceptance < $MIN_ACCEPTANCE_RATE" | bc -l))); then
        log "    SKIPPED: Acceptance rate below ${MIN_ACCEPTANCE_RATE}% threshold"
        continue
      fi

      ((viable_pairs++)) || true

      # Run full optimization sweep
      local sweep_result
      sweep_result=$(run_optimization_sweep "$target_path" "$target_name" "$draft_path" "$draft_name" "$pair_dir")

      IFS='|' read -r baseline best_speed best_k best_temp best_acceptance speedup <<<"$sweep_result"

      log "    RESULT: ${best_speed} t/s (${speedup}x) @ K=$best_k, temp=$best_temp"

      # Track results
      RESULTS+=("$target_name|$draft_name|$baseline|$best_speed|$best_k|$best_temp|$best_acceptance|$speedup")

      # Track best for this target
      if [[ "$target_baseline" == "0" ]]; then
        target_baseline="$baseline"
      fi

      if (($(echo "$best_speed > $target_best_speed" | bc -l))); then
        target_best_speed="$best_speed"
        target_best_draft="$draft_name"
        target_best_k="$best_k"
        target_best_temp="$best_temp"
      fi
    done

    # Summary for this target
    if [[ -n "$target_best_draft" ]]; then
      log ""
      log "  BEST FOR $target_name:"
      log "    Draft: $target_best_draft"
      log "    Config: K=$target_best_k, temp=$target_best_temp"
      log "    Speed: ${target_best_speed} t/s (baseline: ${target_baseline} t/s)"

      # Write to discovery best file for progress display
      echo "$target_name $target_best_draft $target_best_k $target_best_temp $target_best_speed" >>"${TMP_DIR}/draft_discovery_best.txt"
    fi
  done

  # =============================================================================
  # FINAL REPORT
  # =============================================================================

  log ""
  log "=============================================="
  log "DISCOVERY COMPLETE"
  log "=============================================="
  log "Pairs tested: $pairs_tested"
  log "Viable pairs: $viable_pairs"
  log ""

  # Generate summary report
  local report_file="$RUN_DIR/SUMMARY_REPORT.md"
  {
    echo "# Draft Model Discovery Report"
    echo ""
    echo "**Date:** $(date)"
    echo "**Run ID:** $TIMESTAMP"
    echo "**Mode:** $(if $QUICK_MODE; then echo 'Quick'; else echo 'Full'; fi)"
    echo ""
    echo "## Summary"
    echo ""
    echo "- Pairs tested: $pairs_tested"
    echo "- Viable pairs (>${MIN_ACCEPTANCE_RATE}% acceptance): $viable_pairs"
    echo ""
    echo "## Best Configurations"
    echo ""
    echo "| Target | Draft | Baseline | Optimized | Speedup | K | Temp | Accept% |"
    echo "|--------|-------|----------|-----------|---------|---|------|---------|"

    for result in "${RESULTS[@]}"; do
      IFS='|' read -r target draft baseline speed k temp acceptance speedup <<<"$result"
      echo "| $target | $draft | ${baseline} t/s | ${speed} t/s | ${speedup}x | $k | $temp | ${acceptance}% |"
    done

    echo ""
    echo "## Recommended Registry Updates"
    echo ""
    echo "Add to \`orchestration/model_registry.yaml\`:"
    echo ""
    echo "\`\`\`yaml"
    echo "speculative_pairs:"

    for result in "${RESULTS[@]}"; do
      IFS='|' read -r target draft baseline speed k temp acceptance speedup <<<"$result"
      if (($(echo "$speedup > 1.5" | bc -l))); then
        echo "  - target: \"$target\""
        echo "    draft: \"$draft\""
        echo "    k: $k"
        echo "    temp: $temp"
        echo "    speedup: ${speedup}x"
      fi
    done

    echo "\`\`\`"

  } >"$report_file"

  # Also save as JSON for programmatic access
  local json_file="$RUN_DIR/results.json"
  {
    echo "{"
    echo "  \"timestamp\": \"$TIMESTAMP\","
    echo "  \"mode\": \"$(if $QUICK_MODE; then echo 'quick'; else echo 'full'; fi)\","
    echo "  \"pairs_tested\": $pairs_tested,"
    echo "  \"viable_pairs\": $viable_pairs,"
    echo "  \"results\": ["

    local first=true
    for result in "${RESULTS[@]}"; do
      IFS='|' read -r target draft baseline speed k temp acceptance speedup <<<"$result"
      if [[ "$first" != "true" ]]; then
        echo ","
      fi
      first=false
      echo -n "    {\"target\": \"$target\", \"draft\": \"$draft\", \"baseline_tps\": $baseline, \"optimized_tps\": $speed, \"k\": $k, \"temp\": $temp, \"acceptance\": $acceptance, \"speedup\": $speedup}"
    done

    echo ""
    echo "  ]"
    echo "}"
  } >"$json_file"

  # Copy to permanent storage
  cp "$report_file" "$RESULTS_DIR/discovery_${TIMESTAMP}.md"
  cp "$json_file" "$RESULTS_DIR/discovery_${TIMESTAMP}.json"

  log ""
  log "Reports saved to:"
  log "  $report_file"
  log "  $json_file"
  log "  $RESULTS_DIR/discovery_${TIMESTAMP}.md"
  log ""

  # Print summary table
  cat "$report_file"
}

main "$@"
