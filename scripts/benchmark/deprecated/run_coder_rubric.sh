#!/bin/bash
# Coder Model Quality Rubric Test Script
# Runs all T1/T2/T3 coding questions and captures results with timing
#
# Single Config Mode: Pass config as 4th param, DRAFT_MODEL via env var
# Multi Config Mode: Omit 4th param, uses shared lib to determine configs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}" # dense, qwen3moe, qwen3vlmoe, etc.
CONFIG_PARAM="${4:-}"    # Optional: specific config to run (single config mode)
OUTPUT_DIR="/mnt/raid0/llm/tmp/coder_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
LLAMA_SPECULATIVE="/mnt/raid0/llm/llama.cpp/build/bin/llama-speculative"
LLAMA_CLI="/mnt/raid0/llm/llama.cpp/build/bin/llama-cli"
TIMEOUT=180

# Source shared libraries
if [[ -f "$SCRIPT_DIR/lib/optimization_configs.sh" ]]; then
  source "$SCRIPT_DIR/lib/optimization_configs.sh"
fi

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model.gguf> <model_name> [arch] [config]"
  echo ""
  echo "Architecture types:"
  echo "  dense       - Standard dense model"
  echo "  qwen3moe    - Qwen3-MoE model"
  echo "  qwen3vlmoe  - Qwen3-VL-MoE model"
  echo ""
  echo "Single config mode: Specify config as 4th param + DRAFT_MODEL env var"
  echo "Multi config mode: Omit 4th param, let shared lib determine configs"
  echo ""
  echo "Examples:"
  echo "  $0 /path/to/model.gguf ModelName dense baseline"
  echo "  DRAFT_MODEL=/path/to/draft.gguf $0 /path/to/model.gguf ModelName dense spec_k16"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# DRAFT_MODEL env var: set by caller if using spec decode configs
DRAFT_MODEL_PATH="${DRAFT_MODEL:-}"

# Determine configurations to test
declare -a CONFIGS
if [[ -n "$CONFIG_PARAM" ]]; then
  # Single config mode - caller specifies exactly which config
  CONFIGS=("$CONFIG_PARAM")

  # Validate spec decode configs have draft model
  if [[ "$CONFIG_PARAM" =~ ^spec_k ]]; then
    if [[ -z "$DRAFT_MODEL_PATH" ]] || [[ ! -f "$DRAFT_MODEL_PATH" ]]; then
      echo "ERROR: Config $CONFIG_PARAM requires DRAFT_MODEL env var to be set"
      exit 1
    fi
  fi
  echo "[SINGLE CONFIG MODE] Running only: $CONFIG_PARAM"
else
  # Multi-config mode - use shared library if available
  if type setup_configs &>/dev/null; then
    setup_configs "$MODEL_NAME" "$MODEL_ARCH"
    # DRAFT_MODEL_PATH is set by setup_configs if draft found
  else
    # Fallback if shared lib not available
    CONFIGS=("baseline")
    case "$MODEL_ARCH" in
      qwen3moe | qwen3vlmoe)
        for exp in 2 4 6 8; do
          CONFIGS+=("moe${exp}")
        done
        ;;
    esac
  fi
fi

echo "=============================================="
echo "Coder Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Model file: $MODEL"
echo "Architecture: $MODEL_ARCH"
echo "Configurations to test: ${CONFIGS[*]}"
echo "Date: $(date)"
echo "=============================================="

# Function to get MoE override based on config and arch
get_moe_override() {
  local config="$1"
  local arch="$2"

  if [[ "$config" == "baseline" ]]; then
    echo ""
    return
  fi

  local expert_count="${config#moe}" # Extract number from moe4, moe6, etc.

  case "$arch" in
    qwen3moe)
      echo "--override-kv qwen3moe.expert_used_count=int:$expert_count"
      ;;
    qwen3vlmoe)
      echo "--override-kv qwen3vlmoe.expert_used_count=int:$expert_count"
      ;;
    *)
      echo ""
      ;;
  esac
}

# Function to run a coding test and extract timing
run_coder_test() {
  local test_name="$1"
  local prompt="$2"
  local config="$3"
  local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

  echo ""
  echo "--- Running $test_name ($config) ---"

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

  # Write prompt to temp file
  echo "$prompt" >"/mnt/raid0/llm/tmp/coder_prompt.txt"

  # Run model based on config type
  if [[ "$config" =~ ^spec_k([0-9]+) ]]; then
    # Speculative decoding
    local k_value="${BASH_REMATCH[1]}"
    timeout "$TIMEOUT" OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_SPECULATIVE" \
      -m "$MODEL" \
      -md "$DRAFT_MODEL_PATH" \
      --draft-max "$k_value" \
      -t 96 -n 1024 --temp 0.3 \
      -f "/mnt/raid0/llm/tmp/coder_prompt.txt" \
      >"$output_file" 2>&1 || true

  elif [[ "$config" =~ ^lookup ]] || [[ "$config" =~ _lookup$ ]]; then
    # Prompt lookup (with or without MoE)
    local moe_override=""
    if [[ "$config" =~ ^moe([0-9]+)_lookup ]]; then
      local exp="${BASH_REMATCH[1]}"
      moe_override=$(get_moe_override "moe$exp" "$MODEL_ARCH")
    fi
    timeout "$TIMEOUT" OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_CLI" \
      -m "$MODEL" \
      --lookup-ngram-min 3 \
      -t 96 -n 1024 --temp 0.3 \
      $moe_override \
      -f "/mnt/raid0/llm/tmp/coder_prompt.txt" \
      >"$output_file" 2>&1 || true

  else
    # Baseline or MoE only
    local moe_override
    moe_override=$(get_moe_override "$config" "$MODEL_ARCH")
    timeout "$TIMEOUT" OMP_NUM_THREADS=1 numactl --interleave=all \
      "$LLAMA_COMPLETION" \
      -m "$MODEL" \
      -t 96 -n 1024 --temp 0.3 \
      $moe_override \
      -f "/mnt/raid0/llm/tmp/coder_prompt.txt" \
      >"$output_file" 2>&1 || true
  fi

  # Extract timing
  local speed
  speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

  echo "Speed: $speed"
  echo "Output saved to: $output_file"

  # Show the code output (filter out llama.cpp noise)
  echo "--- Code Output ---"
  grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:\|^==\|^>" "$output_file" | tail -40 | head -30
}

# Run tests for each configuration
for CONFIG in "${CONFIGS[@]}"; do
  echo ""
  echo "##############################################"
  echo "# Configuration: $CONFIG"
  if [[ "$CONFIG" =~ ^moe ]]; then
    echo "# Override: $(get_moe_override "$CONFIG" "$MODEL_ARCH")"
  elif [[ "$CONFIG" =~ ^spec_k ]]; then
    echo "# Draft: $(basename "$DRAFT_MODEL_PATH")"
  fi
  echo "##############################################"

  # T1: Baseline questions
  echo ""
  echo "========== TIER 1 (Baseline) =========="

  run_coder_test "t1_q1_factorial" \
    "Write a Python function that returns the factorial of a number n. Include type hints.
Just the function, no explanation needed." "$CONFIG"

  run_coder_test "t1_q2_reverse_words" \
    "Write a function that reverses each word in a string but keeps word order.
Example: 'hello world' -> 'olleh dlrow'
Just the function, no explanation needed." "$CONFIG"

  run_coder_test "t1_q3_stack" \
    "Write a Python class for a Stack with push, pop, peek, and is_empty methods.
Just the class, no explanation needed." "$CONFIG"

  # T2: Medium-Hard questions
  echo ""
  echo "========== TIER 2 (Medium-Hard) =========="

  run_coder_test "t2_q1_palindrome" \
    "Write a function to find the longest palindromic substring in a string.
Return the first one if there are multiple of the same length.
Handle edge cases. Just the function." "$CONFIG"

  run_coder_test "t2_q2_async_fetch" \
    "Write a Python async function that fetches multiple URLs concurrently.
Return a dict mapping URL to status code. Handle 5s timeout per request.
Use aiohttp. Just the function." "$CONFIG"

  run_coder_test "t2_q3_lru_cache" \
    "Implement an LRU cache class with get(key) and put(key, value) methods.
Both should be O(1). Capacity is passed to __init__.
Just the class." "$CONFIG"

  # T3: Hard questions
  echo ""
  echo "========== TIER 3 (Hard) =========="

  run_coder_test "t3_q1_ip_addresses" \
    "Write a function that finds all valid IP addresses from a string of digits.
Example: '25525511135' -> ['255.255.11.135', '255.255.111.35']
Just the function." "$CONFIG"

  run_coder_test "t3_q2_rate_limiter" \
    "Write a thread-safe rate limiter class using the token bucket algorithm.
Allow N requests per minute. Include allow_request() -> bool and reset().
Just the class." "$CONFIG"

  run_coder_test "t3_q3_bug_fix" \
    "Fix the bug in this merge intervals code:
\`\`\`python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= merged[-1][1]:
            merged[-1][1] = intervals[i][1]
        else:
            merged.append(intervals[i])
    return merged
\`\`\`
Explain the bug and provide the fixed code." "$CONFIG"

  run_coder_test "t3_q4_retry_decorator" \
    "Write a Python decorator @retry(max_attempts=3, delay=1.0, backoff=2.0)
that retries a function on exception with exponential backoff.
Should work with both sync and async functions.
Just the decorator code." "$CONFIG"

  # Configuration Summary
  echo ""
  echo "=============================================="
  echo "CONFIG $CONFIG COMPLETE"
  echo "=============================================="
  echo "Speed summary for $MODEL_NAME ($CONFIG):"

  max_speed=0 q_count=0
  for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_*.txt; do
    if [[ -f "$f" ]]; then
      ((q_count++)) || true
      test_name=$(basename "$f" .txt | sed "s/${MODEL_NAME}_${CONFIG}_//")
      speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")
      echo "  $test_name: $speed"
      [[ -n "$speed" ]] && (($(echo "$speed > $max_speed" | bc -l 2>/dev/null || echo 0))) && max_speed="$speed"
    fi
  done

  # Add discovery info for spec_k configs: draft_model,K=N,T=X
  discovery_info="-"
  if [[ "$CONFIG" =~ ^spec_k([0-9]+) ]]; then
    k_val="${BASH_REMATCH[1]}"
    draft_name=$(basename "${DRAFT_MODEL_PATH:-unknown}" .gguf | cut -c1-15)
    discovery_info="${draft_name},K=${k_val},T=0.3"
  fi
  printf "%-10s %-20s %-8s %2d %8s %-30s\n" "coder" "${MODEL_NAME:0:20}" "$CONFIG" "$q_count" "${max_speed}" "$discovery_info" >>/mnt/raid0/llm/tmp/benchmark_completions.log
done

# Final Summary
echo ""
echo "=============================================="
echo "CODER RUBRIC TEST COMPLETE"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
