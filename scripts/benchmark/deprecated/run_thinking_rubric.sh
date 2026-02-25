#!/bin/bash
# =============================================================================
# Thinking Model Quality Rubric Test Script
# =============================================================================
# Runs all T1/T2/T3 questions and captures results with timing
#
# Supports COMPREHENSIVE optimization testing:
#   - MoE Expert Reduction: Sweeps 2, 4, 6, 8 experts
#   - Speculative Decoding: Sweeps K=4, 8, 16, 24 with auto-discovered drafts
#   - Prompt Lookup: Tests n-gram lookup for copy-heavy scenarios
#
# IMPORTANT: This script reads launch configurations from the registry when
# available. Models should be onboarded via /new-model which evaluates the
# correct launch flags and stores them in the registry.
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source shared libraries
if [[ -f "$SCRIPT_DIR/lib/registry_reader.sh" ]]; then
  source "$SCRIPT_DIR/lib/registry_reader.sh"
fi
if [[ -f "$SCRIPT_DIR/lib/optimization_configs.sh" ]]; then
  source "$SCRIPT_DIR/lib/optimization_configs.sh"
fi

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}" # dense, qwen3moe, qwen3next, mixtral, etc.
CONFIG_PARAM="${4:-}"    # Optional: specific config to run (baseline, moe4, spec_k8, lookup, etc.)
OUTPUT_DIR="/mnt/raid0/llm/tmp/thinking_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
LLAMA_SPECULATIVE="/mnt/raid0/llm/llama.cpp/build/bin/llama-speculative"
TIMEOUT=180

# Quick mode (set via env var for overnight runs that need speed)
QUICK_MODE="${BENCHMARK_QUICK_MODE:-false}"

# Try to get launch flags from registry, fallback to --no-conversation
LAUNCH_FLAGS="${LAUNCH_FLAGS:-}"
if [[ -z "$LAUNCH_FLAGS" ]] && type get_launch_flags &>/dev/null; then
  LAUNCH_FLAGS=$(get_launch_flags "$MODEL_NAME" 2>/dev/null || echo "--no-conversation")
fi
[[ -z "$LAUNCH_FLAGS" ]] && LAUNCH_FLAGS="--no-conversation"

# Quirk: Don't pipe output directly - causes "error: invalid argument:"
# Quirk: Model uses <think>...</think> blocks for reasoning
# Quirk: Auto-enables conversation mode (has chat template) - use flags from registry

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model.gguf> <model_name> [arch] [config]"
  echo ""
  echo "Architecture types:"
  echo "  dense       - Standard dense model (tests baseline + spec decode if draft available)"
  echo "  qwen3moe    - Qwen3-MoE model (tests baseline + expert sweep 2,4,6,8)"
  echo "  qwen3next   - Qwen3-Next SSM model (baseline + expert sweep, NO spec decode)"
  echo "  mixtral     - Mixtral MoE model (baseline + expert sweep)"
  echo ""
  echo "Config parameter (optional - if omitted, tests all configs for architecture):"
  echo "  baseline    - No optimization"
  echo "  moe2-8      - Expert reduction (moe2, moe4, moe6, moe8)"
  echo "  spec_k4-24  - Speculative decoding (spec_k4, spec_k8, spec_k16, spec_k24)"
  echo "  lookup      - Prompt lookup decoding"
  echo "  moe4_lookup - Expert reduction + lookup combined"
  echo ""
  echo "Environment variables:"
  echo "  BENCHMARK_QUICK_MODE=true   - Only test baseline + best known config"
  echo "  DRAFT_MODEL=/path/to.gguf   - Specify draft model for spec decode"
  echo ""
  echo "Examples:"
  echo "  $0 /path/to/Qwen3-4B-Thinking.gguf Qwen3-4B-Thinking dense"
  echo "  $0 /path/to/Qwen3-30B-A3B-Thinking.gguf Qwen3-30B-Thinking qwen3moe"
  echo "  $0 /path/to/model.gguf ModelName qwen3moe moe4  # Run only moe4 config"
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "/mnt/raid0/llm/tmp"

# =============================================================================
# CONFIGURATION SETUP
# =============================================================================
# Configuration sweeps (MOE_EXPERTS, SPEC_K_VALUES, etc.) are defined in the
# shared library lib/optimization_configs.sh - do not duplicate here.
#
# DRAFT_MODEL env var: set by caller if using spec decode configs
# This rubric does NOT discover draft models - that belongs in config selection layer
DRAFT_MODEL_PATH="${DRAFT_MODEL:-}"

# Lookup n-gram minimum for prompt lookup decoding
LOOKUP_NGRAM_MIN="${LOOKUP_NGRAM_MIN:-3}"

# Determine configurations to test based on architecture
declare -a CONFIGS

# Check if specific config was requested via 4th parameter
if [[ -n "$CONFIG_PARAM" ]]; then
  # Single config mode - run only the specified config
  CONFIGS=("$CONFIG_PARAM")

  # For spec decode, verify draft model was provided via DRAFT_MODEL env var
  if [[ "$CONFIG_PARAM" =~ ^spec_k ]]; then
    if [[ -z "$DRAFT_MODEL_PATH" ]] || [[ ! -f "$DRAFT_MODEL_PATH" ]]; then
      echo "ERROR: Config $CONFIG_PARAM requires DRAFT_MODEL env var to be set"
      echo "       Call with: DRAFT_MODEL=/path/to/draft.gguf $0 ..."
      exit 1
    fi
    echo "Using draft model: $DRAFT_MODEL_PATH"
  fi

  echo "[SINGLE CONFIG MODE] Running only: $CONFIG_PARAM"
else
  # Multi-config mode - use shared library to determine configs
  # Note: The preferred approach is to use the wrapper which iterates configs
  # and calls this rubric with single config mode (4th param)
  if type setup_configs &>/dev/null; then
    # Use shared library for config determination
    setup_configs "$MODEL_NAME" "$MODEL_ARCH"
    # DRAFT_MODEL_PATH is set by setup_configs if draft was found
  else
    # Fallback: baseline only
    echo "WARNING: optimization_configs.sh not loaded - using baseline only"
    CONFIGS=("baseline")
  fi
fi # End of if/else for CONFIG_PARAM

echo "=============================================="
echo "Thinking Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Model file: $MODEL"
echo "Architecture: $MODEL_ARCH"
echo "Configurations to test: ${CONFIGS[*]}"
echo "Date: $(date)"
echo "=============================================="

# =============================================================================
# CONFIG-SPECIFIC COMMAND BUILDERS
# =============================================================================

# Get MoE override flags based on config and arch
get_moe_override() {
  local config="$1"
  local arch="$2"

  if [[ "$config" == "baseline" ]] || [[ "$config" =~ ^spec_ ]]; then
    echo ""
    return
  fi

  local expert_count="${config#moe}" # Extract number from moe2, moe4, moe6, moe8

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

# Check if config is speculative decoding
is_spec_config() {
  [[ "$1" =~ ^spec_k[0-9]+$ ]]
}

# Get K value from spec config
get_spec_k() {
  local config="$1"
  echo "${config#spec_k}"
}

# Check if config uses lookup decoding
is_lookup_config() {
  [[ "$1" == "lookup" ]] || [[ "$1" =~ _lookup$ ]]
}

# =============================================================================
# DRAFT MODEL COMPATIBILITY VALIDATION
# =============================================================================
# Validates that the draft model is compatible with the target model BEFORE
# running speculative decoding tests. This prevents wasting time on tests
# that will silently fall back to regular decoding.
#
# Returns: 0 if compatible, 1 if incompatible
# Sets: DRAFT_VALIDATION_ERROR with error message if incompatible

DRAFT_VALIDATED=""
DRAFT_VALIDATION_ERROR=""

validate_draft_compatibility() {
  local target_model="$1"
  local draft_model="$2"

  # Skip if already validated this session
  if [[ -n "$DRAFT_VALIDATED" ]]; then
    [[ "$DRAFT_VALIDATED" == "pass" ]] && return 0 || return 1
  fi

  echo ""
  echo "[DRAFT VALIDATION] Testing compatibility..."
  echo "  Target: $(basename "$target_model")"
  echo "  Draft:  $(basename "$draft_model")"

  local validation_output="/mnt/raid0/llm/tmp/draft_validation_$$.txt"

  # Run minimal speculative decode test (just 2 tokens to check compatibility)
  timeout 60 env OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_SPECULATIVE" \
    -m "$target_model" \
    -md "$draft_model" \
    --draft-max 4 \
    -t 96 -n 2 --temp 0 \
    -p "Hello" \
    >"$validation_output" 2>&1 || true

  # Check for known incompatibility errors
  if grep -q "draft model special tokens must match" "$validation_output"; then
    DRAFT_VALIDATION_ERROR="Special tokens mismatch between target and draft model"
    DRAFT_VALIDATED="fail"
    echo "[DRAFT VALIDATION] FAILED: $DRAFT_VALIDATION_ERROR"
    rm -f "$validation_output"
    return 1
  fi

  if grep -q "draft model vocab size" "$validation_output"; then
    DRAFT_VALIDATION_ERROR="Vocabulary size mismatch between target and draft model"
    DRAFT_VALIDATED="fail"
    echo "[DRAFT VALIDATION] FAILED: $DRAFT_VALIDATION_ERROR"
    rm -f "$validation_output"
    return 1
  fi

  if grep -q "incompatible" "$validation_output"; then
    DRAFT_VALIDATION_ERROR="Draft model incompatible (generic error)"
    DRAFT_VALIDATED="fail"
    echo "[DRAFT VALIDATION] FAILED: $DRAFT_VALIDATION_ERROR"
    rm -f "$validation_output"
    return 1
  fi

  # Check if we got any output at all (model loaded successfully)
  if ! grep -q "eval time" "$validation_output"; then
    DRAFT_VALIDATION_ERROR="Speculative decode test produced no timing output"
    DRAFT_VALIDATED="fail"
    echo "[DRAFT VALIDATION] FAILED: $DRAFT_VALIDATION_ERROR"
    rm -f "$validation_output"
    return 1
  fi

  # Extract acceptance rate from validation run
  local test_acceptance
  test_acceptance=$(grep -oP 'accept\s+=\s+\K[\d.]+' "$validation_output" 2>/dev/null | tail -1 || echo "")

  DRAFT_VALIDATED="pass"
  echo "[DRAFT VALIDATION] PASSED"
  if [[ -n "$test_acceptance" ]]; then
    echo "  Initial acceptance rate: ${test_acceptance}%"
  fi

  rm -f "$validation_output"
  return 0
}

# Check if config is MoE + lookup combination (e.g., moe4_lookup)
is_moe_lookup_config() {
  [[ "$1" =~ ^moe[0-9]+_lookup$ ]]
}

# Get MoE expert count from combo config (moe4_lookup -> 4)
get_moe_from_lookup_combo() {
  local config="$1"
  local moe_part="${config%_lookup}" # Remove _lookup suffix
  echo "${moe_part#moe}"             # Remove moe prefix
}

# Function to run a test and extract timing
run_test() {
  local test_name="$1"
  local prompt="$2"
  local config="$3"
  local moe_override="$4"
  local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

  echo ""
  echo "--- Running $test_name ($config) ---"

  # SKIP if result already exists with valid timing data
  if [[ "${BENCHMARK_SKIP_EXISTING:-true}" == "true" ]] && [[ -f "$output_file" ]]; then
    if grep -q "tokens per second" "$output_file" 2>/dev/null; then
      local existing_speed
      existing_speed=$(grep "eval time" "$output_file" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1)
      echo "SKIPPED (exists): $output_file"
      echo "  Previous speed: ${existing_speed:-N/A} t/s"
      return 0
    fi
  fi

  # DRY RUN: Skip actual model invocation but iterate through all tests
  if [[ "${BENCHMARK_DRY_RUN:-false}" == "true" ]]; then
    echo "[DRY RUN] Would run: $MODEL_NAME | $config | $test_name"
    # Write placeholder output so progress tracking has data to parse
    cat >"$output_file" <<'DRYRUN'
DRY_RUN: test placeholder
llama_print_timings:        eval time =    1000.00 ms /   100 tokens (   10.00 ms per token,   0.00 tokens per second)
DRYRUN
    echo "Speed: 0.00 tokens per second (dry run)"
    return 0
  fi

  # Write prompt to temp file (avoids shell escaping issues)
  echo "$prompt" >"/mnt/raid0/llm/tmp/rubric_prompt.txt"

  # Choose execution method based on config type
  # Note: --no-conversation disables auto-enabled conversation mode with chat templates
  if is_spec_config "$config"; then
    # Speculative decoding (llama-speculative doesn't support --no-conversation)
    local k_value
    k_value=$(get_spec_k "$config")
    timeout "$TIMEOUT" env OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_SPECULATIVE" \
      -m "$MODEL" \
      -md "$DRAFT_MODEL_PATH" \
      --draft-max "$k_value" \
      -t 96 -n 512 --temp 0.6 \
      -f "/mnt/raid0/llm/tmp/rubric_prompt.txt" \
      >"$output_file" 2>&1 || true
  elif is_lookup_config "$config"; then
    # Lookup decoding (prompt lookup optimization)
    local lookup_flags="--lookup-ngram-min $LOOKUP_NGRAM_MIN"
    local extra_moe_override=""

    # Check if this is a combo config (moe + lookup)
    if is_moe_lookup_config "$config"; then
      local exp_count
      exp_count=$(get_moe_from_lookup_combo "$config")
      # Get MoE override for the expert count
      case "$MODEL_ARCH" in
        qwen3moe | qwen3next) extra_moe_override="--override-kv qwen3moe.expert_used_count=int:$exp_count" ;;
        qwen3vlmoe) extra_moe_override="--override-kv qwen3vlmoe.expert_used_count=int:$exp_count" ;;
        mixtral) extra_moe_override="--override-kv mixtral.expert_used_count=int:$exp_count" ;;
      esac
    fi

    timeout "$TIMEOUT" env OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_COMPLETION" \
      -m "$MODEL" \
      -t 96 -n 512 --temp 0.6 $LAUNCH_FLAGS \
      $lookup_flags $extra_moe_override \
      -f "/mnt/raid0/llm/tmp/rubric_prompt.txt" \
      >"$output_file" 2>&1 || true
  else
    # Standard or MoE reduced - use launch flags from registry
    timeout "$TIMEOUT" env OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_COMPLETION" \
      -m "$MODEL" \
      -t 96 -n 512 --temp 0.6 $LAUNCH_FLAGS \
      $moe_override \
      -f "/mnt/raid0/llm/tmp/rubric_prompt.txt" \
      >"$output_file" 2>&1 || true
  fi

  # Extract timing
  local speed
  speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

  # For spec decode, also extract acceptance rate
  local acceptance=""
  if is_spec_config "$config"; then
    # Format is: "accept    = 29.688%"
    acceptance=$(grep -oP 'accept\s+=\s+\K[\d.]+' "$output_file" 2>/dev/null | tail -1 || echo "")
    if [[ -n "$acceptance" ]]; then
      echo "Speed: $speed (acceptance: ${acceptance}%)"
    else
      echo "Speed: $speed"
    fi
  else
    echo "Speed: $speed"
  fi

  echo "Output saved to: $output_file"

  # Show the answer (after </think> if present)
  echo "--- Answer ---"
  if grep -q "</think>" "$output_file"; then
    sed -n '/<\/think>/,/EOF by user/p' "$output_file" | head -20
  else
    tail -30 "$output_file" | head -20
  fi
}

# =============================================================================
# DRAFT VALIDATION (run once before any spec configs)
# =============================================================================
# Check if we have spec configs to run and validate draft compatibility upfront
SPEC_CONFIGS_VALID=true
if [[ -n "$DRAFT_MODEL_PATH" ]] && [[ -f "$DRAFT_MODEL_PATH" ]]; then
  # Check if any config in CONFIGS is a spec config
  has_spec_config=false
  for c in "${CONFIGS[@]}"; do
    if is_spec_config "$c"; then
      has_spec_config=true
      break
    fi
  done

  if [[ "$has_spec_config" == "true" ]]; then
    if ! validate_draft_compatibility "$MODEL" "$DRAFT_MODEL_PATH"; then
      echo ""
      echo "WARNING: Draft model validation FAILED"
      echo "  Error: $DRAFT_VALIDATION_ERROR"
      echo "  Speculative decoding configs will be SKIPPED"
      echo ""
      SPEC_CONFIGS_VALID=false
    fi
  fi
fi

# Run tests for each configuration
for CONFIG in "${CONFIGS[@]}"; do
  # Skip spec configs if draft validation failed
  if is_spec_config "$CONFIG" && [[ "$SPEC_CONFIGS_VALID" != "true" ]]; then
    echo ""
    echo "##############################################"
    echo "# Configuration: $CONFIG - SKIPPED (draft incompatible)"
    echo "# Reason: $DRAFT_VALIDATION_ERROR"
    echo "##############################################"
    # Log the skip
    COMPLETION_LOG="/mnt/raid0/llm/tmp/benchmark_completions.log"
    printf "%-10s %-20s %-8s %2d %8s %-30s\n" "thinking" "${MODEL_NAME:0:20}" "$CONFIG" "0" "SKIPPED" "draft_incompatible" >>"$COMPLETION_LOG"
    continue
  fi

  MOE_OVERRIDE=$(get_moe_override "$CONFIG" "$MODEL_ARCH")

  echo ""
  echo "##############################################"
  echo "# Configuration: $CONFIG"
  [[ -n "$MOE_OVERRIDE" ]] && echo "# Override: $MOE_OVERRIDE"
  echo "##############################################"

  # T1: Baseline questions
  echo ""
  echo "========== TIER 1 (Baseline) =========="

  run_test "t1_q1_algorithm" "Sort 10 mostly-sorted items. Quicksort or insertion sort? One sentence." "$CONFIG" "$MOE_OVERRIDE"

  run_test "t1_q2_threadsafe" "Is self.count += 1 thread-safe in Python? One sentence." "$CONFIG" "$MOE_OVERRIDE"

  # T2: Medium-Hard questions
  echo ""
  echo "========== TIER 2 (Medium-Hard) =========="

  run_test "t2_q1_dict_reuse" "Python function called 1000x/sec creates a 1KB dict each call.
Better to: A) Pre-allocate global dict and clear() each time, or B) Create new dict each time?
Explain memory and performance implications in 2-3 sentences." "$CONFIG" "$MOE_OVERRIDE"

  run_test "t2_q2_cache_bug" "Find the bug in this cache:

cache = {}
lock = threading.Lock()

def get_cached(key, compute_fn):
    if key in cache:
        return cache[key]
    with lock:
        result = compute_fn()
        cache[key] = result
        return result

One paragraph explanation." "$CONFIG" "$MOE_OVERRIDE"

  run_test "t2_q3_api_design" "For a library function that might fail, which return style is best?
A) return (success: bool, data, error_msg)
B) return {\"success\": bool, \"data\": ..., \"error\": ...}
C) raise Exception on error, return data on success
Brief reasoning for a Python library consumed by external users." "$CONFIG" "$MOE_OVERRIDE"

  # T3: Very Hard questions
  echo ""
  echo "========== TIER 3 (Very Hard) =========="

  run_test "t3_q1_dependency" "Service startup constraints:
- A must start before B
- B must start before C or D (either one)
- C and D cannot start simultaneously (resource conflict)
- E requires both C AND D to be running

What is the minimum number of sequential startup phases?
List the phases." "$CONFIG" "$MOE_OVERRIDE"

  run_test "t3_q2_vector_clock" "Vector clock merge:
P1 has clock [2,0,1], P2 has clock [1,2,0], P3 has clock [0,0,2].
1) P1 sends message to P2. What is P2's clock after receiving?
2) P2 then sends to P3. What is P3's clock after receiving?
Show work." "$CONFIG" "$MOE_OVERRIDE"

  run_test "t3_q3_type_system" "Consider this function:

from typing import Sequence, TypeVar
T = TypeVar('T')

def first_or_default(items: Sequence[T], default: T) -> T:
    return items[0] if items else default

What happens with: first_or_default([], None)?
Is the type signature correct? What is the subtle issue?" "$CONFIG" "$MOE_OVERRIDE"

  run_test "t3_q4_probability" "Load balancer randomly routes to 3 servers with equal probability.
Server latencies: S1=10ms, S2=50ms, S3=100ms.
What is the expected MEDIAN latency over many requests?" "$CONFIG" "$MOE_OVERRIDE"

  # Configuration Summary
  echo ""
  echo "=============================================="
  echo "CONFIG $CONFIG COMPLETE"
  echo "=============================================="
  echo "Speed summary for $MODEL_NAME ($CONFIG):"

  # Calculate max speed, question count, and acceptance rate for this config
  max_speed=0
  q_count=0
  total_acceptance=0
  acceptance_count=0

  for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_*.txt; do
    if [[ -f "$f" ]]; then
      ((q_count++)) || true
      test_name=$(basename "$f" .txt | sed "s/${MODEL_NAME}_${CONFIG}_//")
      speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")

      # For spec decode, also extract acceptance rate
      if is_spec_config "$CONFIG"; then
        acceptance=$(grep -oP 'accept\s+=\s+\K[\d.]+' "$f" 2>/dev/null | tail -1 || echo "")
        if [[ -n "$acceptance" ]]; then
          echo "  $test_name: $speed t/s (accept: ${acceptance}%)"
          total_acceptance=$(echo "$total_acceptance + $acceptance" | bc -l 2>/dev/null || echo "$total_acceptance")
          ((acceptance_count++)) || true
        else
          echo "  $test_name: $speed t/s (accept: N/A)"
        fi
      else
        echo "  $test_name: $speed t/s"
      fi

      # Track max speed
      if [[ -n "$speed" ]] && [[ "$speed" != "N/A" ]] && [[ "$speed" != "0" ]]; then
        if (($(echo "$speed > $max_speed" | bc -l 2>/dev/null || echo 0))); then
          max_speed="$speed"
        fi
      fi
    fi
  done

  # Calculate average acceptance rate for spec configs
  avg_acceptance=""
  if [[ "$acceptance_count" -gt 0 ]]; then
    avg_acceptance=$(echo "scale=2; $total_acceptance / $acceptance_count" | bc -l 2>/dev/null || echo "")
    echo ""
    echo "  Average acceptance rate: ${avg_acceptance}%"
  fi

  # Write completion entry to shared log (for progress display)
  COMPLETION_LOG="/mnt/raid0/llm/tmp/benchmark_completions.log"
  # Add discovery info for spec_k configs: draft_model,K=N,T=X,A=acceptance%
  discovery_info="-"
  if [[ "$CONFIG" =~ ^spec_k([0-9]+) ]]; then
    k_val="${BASH_REMATCH[1]}"
    draft_name=$(basename "${DRAFT_MODEL_PATH:-unknown}" .gguf | cut -c1-15)
    if [[ -n "$avg_acceptance" ]]; then
      discovery_info="${draft_name},K=${k_val},T=0.6,A=${avg_acceptance}%"
    else
      discovery_info="${draft_name},K=${k_val},T=0.6,A=N/A"
    fi
  fi
  printf "%-10s %-20s %-8s %2d %8s %-30s\n" "thinking" "${MODEL_NAME:0:20}" "$CONFIG" "$q_count" "${max_speed}" "$discovery_info" >>"$COMPLETION_LOG"
done

# Final Summary
echo ""
echo "=============================================="
echo "THINKING RUBRIC TEST COMPLETE"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
