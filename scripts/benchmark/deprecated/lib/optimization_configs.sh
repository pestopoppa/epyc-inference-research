#!/bin/bash
# =============================================================================
# SHARED OPTIMIZATION CONFIGURATION LIBRARY
# =============================================================================
# Source this file in rubric scripts to get comprehensive optimization sweeps.
#
# Usage:
#   source "$(dirname "$0")/lib/optimization_configs.sh"
#   setup_configs "$MODEL_NAME" "$MODEL_ARCH"
#   # CONFIGS array is now populated
#   # DRAFT_MODEL_PATH is set if draft available
#
# =============================================================================

# Binaries
LLAMA_COMPLETION="${LLAMA_COMPLETION:-/mnt/raid0/llm/llama.cpp/build/bin/llama-completion}"
LLAMA_SPECULATIVE="${LLAMA_SPECULATIVE:-/mnt/raid0/llm/llama.cpp/build/bin/llama-speculative}"
LLAMA_CLI="${LLAMA_CLI:-/mnt/raid0/llm/llama.cpp/build/bin/llama-cli}"

# Quick mode (set via env var)
BENCHMARK_QUICK_MODE="${BENCHMARK_QUICK_MODE:-false}"

# =============================================================================
# SWEEP PARAMETERS
# =============================================================================

# MoE expert counts
MOE_EXPERTS_FULL=(2 4 6 8)
MOE_EXPERTS_QUICK=(4)

# Speculative decoding K values
SPEC_K_FULL=(4 8 16 24)
SPEC_K_QUICK=(8 16)

# Prompt lookup n-gram sizes
LOOKUP_NGRAM_FULL=(3 4 5)
LOOKUP_NGRAM_QUICK=(4)

# Temperature values for coarse sweep (initial phase)
TEMP_COARSE=(0.0 0.3 0.6 0.9)
TEMP_QUICK=(0.2)

# Temperature optimization precision
TEMP_PRECISION=0.001
TEMP_SEARCH_MIN=0.0
TEMP_SEARCH_MAX=1.0

# Select parameters based on mode
if [[ "$BENCHMARK_QUICK_MODE" == "true" ]]; then
  MOE_EXPERTS=("${MOE_EXPERTS_QUICK[@]}")
  SPEC_K_VALUES=("${SPEC_K_QUICK[@]}")
  LOOKUP_NGRAM=("${LOOKUP_NGRAM_QUICK[@]}")
  TEMP_VALUES=("${TEMP_QUICK[@]}")
else
  MOE_EXPERTS=("${MOE_EXPERTS_FULL[@]}")
  SPEC_K_VALUES=("${SPEC_K_FULL[@]}")
  LOOKUP_NGRAM=("${LOOKUP_NGRAM_FULL[@]}")
  TEMP_VALUES=("${TEMP_FULL[@]}")
fi

# =============================================================================
# DRAFT MODEL DISCOVERY
# =============================================================================

# Known draft model locations by architecture family
declare -A DRAFT_CANDIDATES

# Qwen2.5 family (for Qwen2.5-*, DeepSeek-R1-Distill-Qwen-*)
DRAFT_CANDIDATES["qwen2.5"]="
/mnt/raid0/llm/lmstudio/models/QuantFactory/Qwen2.5-0.5B-GGUF/Qwen2.5-0.5B.Q8_0.gguf
/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-Instruct-GGUF/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf
/mnt/raid0/llm/lmstudio/models/tensorblock/Qwen2.5-Math-1.5B-Instruct-GGUF/Qwen2.5-Math-1.5B-Instruct-Q6_K.gguf
/mnt/raid0/llm/lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf
/mnt/raid0/llm/lmstudio/models/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf
"

# Qwen3 dense family
DRAFT_CANDIDATES["qwen3"]="
/mnt/raid0/llm/lmstudio/models/mradermacher/Co-rewarding-II-Qwen3-1.7B-Base-MATH-GGUF/Co-rewarding-II-Qwen3-1.7B-Base-MATH.Q8_0.gguf
"

# Llama family
DRAFT_CANDIDATES["llama"]="
/mnt/raid0/llm/lmstudio/models/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf
"

# Get architecture family from model name
get_architecture_family() {
  local model_name="$1"

  # Qwen2.5 family (includes DeepSeek-R1-Distill-Qwen)
  if [[ "$model_name" =~ Qwen2\.5 ]] || [[ "$model_name" =~ Qwen2-5 ]]; then
    echo "qwen2.5"
    return
  fi
  if [[ "$model_name" =~ DeepSeek-R1-Distill-Qwen ]]; then
    echo "qwen2.5"
    return
  fi

  # Qwen3 MoE (has A3B, A22B, etc.)
  if [[ "$model_name" =~ Qwen3 ]] && [[ "$model_name" =~ A[0-9]+B ]]; then
    echo "qwen3moe"
    return
  fi

  # Qwen3 dense
  if [[ "$model_name" =~ Qwen3 ]] || [[ "$model_name" =~ qwen3 ]]; then
    echo "qwen3"
    return
  fi

  # Llama
  if [[ "$model_name" =~ [Ll]lama ]]; then
    echo "llama"
    return
  fi

  echo "unknown"
}

# Discover compatible draft model for a target
discover_draft_model() {
  local model_name="$1"

  # Check if explicitly provided via env var
  if [[ -n "${DRAFT_MODEL:-}" ]] && [[ -f "$DRAFT_MODEL" ]]; then
    echo "$DRAFT_MODEL"
    return
  fi

  # Get family
  local family
  family=$(get_architecture_family "$model_name")
  if [[ "$family" == "unknown" ]] || [[ -z "${DRAFT_CANDIDATES[$family]:-}" ]]; then
    echo ""
    return
  fi

  # Find first available draft
  while IFS= read -r draft; do
    draft=$(echo "$draft" | xargs) # trim whitespace
    [[ -z "$draft" ]] && continue
    if [[ -f "$draft" ]]; then
      echo "$draft"
      return
    fi
  done <<<"${DRAFT_CANDIDATES[$family]}"

  echo ""
}

# =============================================================================
# CONFIGURATION SETUP
# =============================================================================

# Global variables set by setup_configs
declare -a CONFIGS
DRAFT_MODEL_PATH=""

# Setup configurations based on model architecture
setup_configs() {
  local model_name="$1"
  local model_arch="$2"

  CONFIGS=("baseline")
  DRAFT_MODEL_PATH=""

  case "$model_arch" in
    dense)
      # Try to find compatible draft for spec decode
      DRAFT_MODEL_PATH=$(discover_draft_model "$model_name")
      if [[ -n "$DRAFT_MODEL_PATH" ]]; then
        for k in "${SPEC_K_VALUES[@]}"; do
          CONFIGS+=("spec_k${k}")
        done
        echo "[CONFIG] Found draft: $(basename "$DRAFT_MODEL_PATH")"
        echo "[CONFIG] Will test spec_k: ${SPEC_K_VALUES[*]}"
      fi
      ;;

    qwen3moe | qwen3vlmoe | mixtral | deepseek2)
      for exp in "${MOE_EXPERTS[@]}"; do
        CONFIGS+=("moe${exp}")
      done
      echo "[CONFIG] MoE model - will test expert counts: ${MOE_EXPERTS[*]}"
      ;;

    qwen3next)
      # SSM - MoE reduction works but NO speculative decoding
      for exp in "${MOE_EXPERTS[@]}"; do
        CONFIGS+=("moe${exp}")
      done
      echo "[CONFIG] SSM model - MoE sweep only (spec decode incompatible)"
      ;;

    *)
      echo "[CONFIG] Unknown arch '$model_arch' - baseline only"
      ;;
  esac

  echo "[CONFIG] Configurations: ${CONFIGS[*]}"
}

# =============================================================================
# OVERRIDE FLAG GENERATORS
# =============================================================================

# Get MoE override flags
get_moe_override() {
  local config="$1"
  local arch="$2"

  if [[ "$config" == "baseline" ]] || [[ "$config" =~ ^spec_ ]] || [[ "$config" =~ ^lookup_ ]]; then
    echo ""
    return
  fi

  local expert_count="${config#moe}"

  case "$arch" in
    qwen3moe | qwen3next)
      echo "--override-kv qwen3moe.expert_used_count=int:$expert_count"
      ;;
    qwen3vlmoe)
      echo "--override-kv qwen3vlmoe.expert_used_count=int:$expert_count"
      ;;
    mixtral)
      echo "--override-kv mixtral.expert_used_count=int:$expert_count"
      ;;
    deepseek2)
      echo "--override-kv deepseek2.expert_used_count=int:$expert_count"
      ;;
    *)
      echo ""
      ;;
  esac
}

# Check config type
is_spec_config() {
  [[ "$1" =~ ^spec_k[0-9]+$ ]]
}

is_moe_config() {
  [[ "$1" =~ ^moe[0-9]+$ ]]
}

is_lookup_config() {
  [[ "$1" =~ ^lookup_n[0-9]+$ ]]
}

# Get K value from spec config
get_spec_k() {
  echo "${1#spec_k}"
}

# Get n-gram size from lookup config
get_lookup_ngram() {
  echo "${1#lookup_n}"
}

# =============================================================================
# TEST RUNNER
# =============================================================================

# Run a single test with appropriate optimization
run_optimized_test() {
  local model_path="$1"
  local model_arch="$2"
  local test_name="$3"
  local prompt="$4"
  local config="$5"
  local output_file="$6"
  local timeout_sec="${7:-180}"
  local max_tokens="${8:-512}"
  local temperature="${9:-0.6}"

  # Write prompt to temp file
  local prompt_file="/tmp/claude/rubric_prompt_$$.txt"
  echo "$prompt" >"$prompt_file"

  local exit_code=0

  if is_spec_config "$config"; then
    # Speculative decoding
    local k_value
    k_value=$(get_spec_k "$config")
    timeout "$timeout_sec" OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_SPECULATIVE" \
      -m "$model_path" \
      -md "$DRAFT_MODEL_PATH" \
      --draft-max "$k_value" \
      -t 96 -n "$max_tokens" --temp "$temperature" \
      -f "$prompt_file" \
      >"$output_file" 2>&1 || exit_code=$?

  elif is_lookup_config "$config"; then
    # Prompt lookup
    local ngram
    ngram=$(get_lookup_ngram "$config")
    timeout "$timeout_sec" OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_CLI" \
      -m "$model_path" \
      --lookup-ngram-min "$ngram" \
      -t 96 -n "$max_tokens" --temp "$temperature" \
      -f "$prompt_file" \
      >"$output_file" 2>&1 || exit_code=$?

  else
    # Baseline or MoE
    local moe_override
    moe_override=$(get_moe_override "$config" "$model_arch")
    timeout "$timeout_sec" OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_COMPLETION" \
      -m "$model_path" \
      -t 96 -n "$max_tokens" --temp "$temperature" \
      $moe_override \
      -f "$prompt_file" \
      >"$output_file" 2>&1 || exit_code=$?
  fi

  rm -f "$prompt_file"
  return $exit_code
}

# Extract speed from output file
extract_speed() {
  local output_file="$1"
  grep "eval time" "$output_file" 2>/dev/null | grep -oP '[\d.]+(?= tokens per second)' | tail -1 || echo "0"
}

# Extract acceptance rate (for spec decode)
extract_acceptance() {
  local output_file="$1"
  grep -oP 'draft acceptance rate: \K[\d.]+' "$output_file" 2>/dev/null | tail -1 || echo ""
}

# =============================================================================
# DIVIDE-AND-CONQUER TEMPERATURE OPTIMIZATION
# =============================================================================
# Binary search to find optimal temperature within precision threshold (0.001)
#
# Algorithm:
#   1. Test 3 points: low, mid, high
#   2. Find best performing point
#   3. Narrow range around best point
#   4. Repeat until range < TEMP_PRECISION
#
# Usage:
#   result=$(optimize_temperature "$model" "$arch" "$config" "$prompt" "$output_dir")
#   # Returns: "optimal_temp|best_speed|iterations"

optimize_temperature() {
  local model_path="$1"
  local model_arch="$2"
  local base_config="$3" # baseline, moe4, spec_k8, etc.
  local prompt="$4"
  local output_dir="$5"
  local max_tokens="${6:-256}"

  local low="$TEMP_SEARCH_MIN"
  local high="$TEMP_SEARCH_MAX"
  local precision="$TEMP_PRECISION"
  local iteration=0
  local max_iterations=15 # Safety limit

  mkdir -p "$output_dir/temp_search"

  # Function to test a temperature and return speed
  test_temp() {
    local temp="$1"
    local temp_str
    temp_str=$(printf "%.4f" "$temp")
    local test_file="$output_dir/temp_search/temp_${temp_str}.txt"

    # Write prompt to file
    echo "$prompt" >"/tmp/claude/temp_opt_prompt_$$.txt"

    if is_spec_config "$base_config"; then
      local k_value
      k_value=$(get_spec_k "$base_config")
      timeout 120 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_SPECULATIVE" \
        -m "$model_path" \
        -md "$DRAFT_MODEL_PATH" \
        --draft-max "$k_value" \
        -t 96 -n "$max_tokens" --temp "$temp" \
        -f "/tmp/claude/temp_opt_prompt_$$.txt" \
        >"$test_file" 2>&1 || true
    else
      local moe_override
      moe_override=$(get_moe_override "$base_config" "$model_arch")
      timeout 120 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_COMPLETION" \
        -m "$model_path" \
        -t 96 -n "$max_tokens" --temp "$temp" \
        $moe_override \
        -f "/tmp/claude/temp_opt_prompt_$$.txt" \
        >"$test_file" 2>&1 || true
    fi

    rm -f "/tmp/claude/temp_opt_prompt_$$.txt"
    extract_speed "$test_file"
  }

  echo "[TEMP_OPT] Starting binary search for optimal temperature" >&2
  echo "[TEMP_OPT] Range: [$low, $high], precision: $precision" >&2

  local best_temp="0.2"
  local best_speed="0"

  while (($(echo "$high - $low > $precision" | bc -l))) && [[ $iteration -lt $max_iterations ]]; do
    ((iteration++))

    local mid

    mid=$(echo "scale=6; ($low + $high) / 2" | bc)
    local q1
    q1=$(echo "scale=6; ($low + $mid) / 2" | bc)
    local q3
    q3=$(echo "scale=6; ($mid + $high) / 2" | bc)

    echo "[TEMP_OPT] Iteration $iteration: testing temps [$low, $q1, $mid, $q3, $high]" >&2

    # Test all 5 points (can parallelize in future)
    local speed_low
    speed_low=$(test_temp "$low")
    local speed_q1
    speed_q1=$(test_temp "$q1")
    local speed_mid
    speed_mid=$(test_temp "$mid")
    local speed_q3
    speed_q3=$(test_temp "$q3")
    local speed_high
    speed_high=$(test_temp "$high")

    echo "[TEMP_OPT]   Speeds: low=$speed_low, q1=$speed_q1, mid=$speed_mid, q3=$speed_q3, high=$speed_high" >&2

    # Find best performing point
    local best_of_five="$low"
    local best_speed_five="$speed_low"

    for temp_val in "$q1:$speed_q1" "$mid:$speed_mid" "$q3:$speed_q3" "$high:$speed_high"; do
      local t="${temp_val%%:*}"
      local s="${temp_val##*:}"
      if (($(echo "$s > $best_speed_five" | bc -l))); then
        best_of_five="$t"
        best_speed_five="$s"
      fi
    done

    echo "[TEMP_OPT]   Best this iteration: temp=$best_of_five, speed=$best_speed_five" >&2

    # Update global best
    if (($(echo "$best_speed_five > $best_speed" | bc -l))); then
      best_temp="$best_of_five"
      best_speed="$best_speed_five"
    fi

    # Narrow the search range based on where best was found
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

    echo "[TEMP_OPT]   New range: [$low, $high]" >&2
  done

  echo "[TEMP_OPT] Converged after $iteration iterations" >&2
  echo "[TEMP_OPT] Optimal: temp=$(printf '%.4f' $best_temp), speed=$best_speed t/s" >&2

  # Return result
  printf "%.4f|%s|%d" "$best_temp" "$best_speed" "$iteration"
}

# Quick temperature optimization (fewer iterations, coarser precision)
optimize_temperature_quick() {
  local model_path="$1"
  local model_arch="$2"
  local base_config="$3"
  local prompt="$4"
  local output_dir="$5"

  # Just test the coarse grid and return best
  mkdir -p "$output_dir/temp_search"

  local best_temp="0.2"
  local best_speed="0"

  echo "[TEMP_OPT_QUICK] Testing temperatures: ${TEMP_COARSE[*]}" >&2

  for temp in "${TEMP_COARSE[@]}"; do
    local test_file="$output_dir/temp_search/temp_${temp}.txt"
    echo "$prompt" >"/tmp/claude/temp_opt_prompt_$$.txt"

    if is_spec_config "$base_config"; then
      local k_value
      k_value=$(get_spec_k "$base_config")
      timeout 120 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_SPECULATIVE" \
        -m "$model_path" \
        -md "$DRAFT_MODEL_PATH" \
        --draft-max "$k_value" \
        -t 96 -n 256 --temp "$temp" \
        -f "/tmp/claude/temp_opt_prompt_$$.txt" \
        >"$test_file" 2>&1 || true
    else
      local moe_override
      moe_override=$(get_moe_override "$base_config" "$model_arch")
      timeout 120 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_COMPLETION" \
        -m "$model_path" \
        -t 96 -n 256 --temp "$temp" \
        $moe_override \
        -f "/tmp/claude/temp_opt_prompt_$$.txt" \
        >"$test_file" 2>&1 || true
    fi

    rm -f "/tmp/claude/temp_opt_prompt_$$.txt"

    local speed

    speed=$(extract_speed "$test_file")
    echo "[TEMP_OPT_QUICK]   temp=$temp -> $speed t/s" >&2

    if (($(echo "$speed > $best_speed" | bc -l))); then
      best_temp="$temp"
      best_speed="$speed"
    fi
  done

  echo "[TEMP_OPT_QUICK] Best: temp=$best_temp, speed=$best_speed t/s" >&2
  printf "%.4f|%s|%d" "$best_temp" "$best_speed" "${#TEMP_COARSE[@]}"
}

echo "[LIB] Loaded optimization_configs.sh (QUICK_MODE=$BENCHMARK_QUICK_MODE)"
