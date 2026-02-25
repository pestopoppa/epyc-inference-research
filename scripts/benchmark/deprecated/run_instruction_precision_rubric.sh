#!/bin/bash
# Instruction Precision Model Quality Rubric Test Script
# Tests exact instruction following - format compliance, constraints
# Includes automated pass/fail scoring where possible
#
# Single Config Mode: Pass config as 4th param
# Multi Config Mode: Omit 4th param, uses shared lib to determine configs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}"
CONFIG_PARAM="${4:-}"
OUTPUT_DIR="/mnt/raid0/llm/tmp/instruction_precision_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=60

# Source shared libraries
if [[ -f "$SCRIPT_DIR/lib/optimization_configs.sh" ]]; then
  source "$SCRIPT_DIR/lib/optimization_configs.sh"
fi

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model.gguf> <model_name> [arch] [config]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Determine configurations
declare -a CONFIGS
if [[ -n "$CONFIG_PARAM" ]]; then
  CONFIGS=("$CONFIG_PARAM")
  echo "[SINGLE CONFIG MODE] Running only: $CONFIG_PARAM"
else
  if type setup_configs &>/dev/null; then
    setup_configs "$MODEL_NAME" "$MODEL_ARCH"
  else
    CONFIGS=("baseline")
    case "$MODEL_ARCH" in
      qwen3moe | qwen3next)
        for exp in 2 4 6 8; do
          CONFIGS+=("moe${exp}")
        done
        ;;
    esac
  fi
fi

echo "=============================================="
echo "Instruction Precision Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Architecture: $MODEL_ARCH"
echo "Date: $(date)"
echo "=============================================="

# Auto-scoring functions
score_json_only() {
  local response="$1"
  # Remove leading/trailing whitespace and check if it's valid JSON starting with {
  local cleaned
  cleaned=$(echo "$response" | grep -v "^llama\|^load\|^print_info\|^common\|^sampler" | tr -d '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if echo "$cleaned" | python3 -c "import sys,json; json.loads(sys.stdin.read())" 2>/dev/null; then
    if [[ "$cleaned" =~ ^\{ ]]; then
      echo "PASS"
    else
      echo "FAIL (JSON not at start)"
    fi
  else
    echo "FAIL (invalid JSON)"
  fi
}

score_count_items() {
  local response="$1"
  local expected="$2"
  local count
  count=$(echo "$response" | grep -v "^llama\|^load\|^$" | grep -cE "^[0-9]+\.|^-|^•|^[A-Z]" || echo "0")
  # Also try counting non-empty lines as items
  local line_count
  line_count=$(echo "$response" | grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^$" | grep -cE ".+" || echo "0")
  if [[ $count -eq $expected ]] || [[ $line_count -eq $expected ]]; then
    echo "PASS ($count or $line_count items)"
  else
    echo "FAIL (found $count/$line_count, expected $expected)"
  fi
}

score_exact_match() {
  local response="$1"
  local expected="$2"
  local cleaned
  cleaned=$(echo "$response" | grep -v "^llama\|^load\|^print_info\|^common\|^sampler" | tr -d '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  # Check against comma-separated list of acceptable answers
  IFS=',' read -ra ACCEPTABLE <<<"$expected"
  for ans in "${ACCEPTABLE[@]}"; do
    if [[ "$cleaned" == "$ans" ]]; then
      echo "PASS"
      return
    fi
  done
  echo "FAIL (got: '$cleaned')"
}

score_forbidden_words() {
  local response="$1"
  local forbidden="$2"
  local cleaned
  cleaned=$(echo "$response" | grep -v "^llama\|^load\|^print_info" | tr '[:upper:]' '[:lower:]')
  IFS=',' read -ra WORDS <<<"$forbidden"
  for word in "${WORDS[@]}"; do
    if echo "$cleaned" | grep -qi "\b$word\b"; then
      echo "FAIL (contains '$word')"
      return
    fi
  done
  echo "PASS"
}

score_word_count() {
  local response="$1"
  local min="$2"
  local max="$3"
  local cleaned
  cleaned=$(echo "$response" | grep -v "^llama\|^load\|^print_info\|^common\|^sampler")
  local count
  count=$(echo "$cleaned" | wc -w)
  if [[ $count -ge $min ]] && [[ $count -le $max ]]; then
    echo "PASS ($count words)"
  else
    echo "FAIL ($count words, expected $min-$max)"
  fi
}

run_test() {
  local test_name="$1"
  local prompt="$2"
  local config="$3"
  local score_func="${4:-}"
  local score_args="${5:-}"
  local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

  echo ""
  echo "--- Running $test_name ($config) ---"

  # DRY RUN: Skip actual model invocation but iterate through all tests
  if [[ "${BENCHMARK_DRY_RUN:-false}" == "true" ]]; then
    echo "[DRY RUN] Would run: $MODEL_NAME | $config | $test_name"
    cat >"$output_file" <<'DRYRUN'
DRY_RUN: test placeholder
llama_print_timings:        eval time =    1000.00 ms /   100 tokens (   10.00 ms per token,   0.00 tokens per second)
DRYRUN
    echo "Speed: 0.00 tokens per second (dry run)"
    return 0
  fi

  # Compute MoE override if needed
  local moe_override=""
  if [[ "$config" =~ ^moe([0-9]+) ]]; then
    if type get_moe_override &>/dev/null; then
      moe_override=$(get_moe_override "$config" "$MODEL_ARCH")
    else
      local exp="${BASH_REMATCH[1]}"
      case "$MODEL_ARCH" in
        qwen3moe | qwen3next) moe_override="--override-kv qwen3moe.expert_used_count=int:$exp" ;;
      esac
    fi
  fi

  echo "$prompt" >"/mnt/raid0/llm/tmp/ip_prompt.txt"

  timeout "$TIMEOUT" OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_COMPLETION" \
    -m "$MODEL" \
    -t 96 -n 256 --temp 0.1 \
    $moe_override \
    -f "/mnt/raid0/llm/tmp/ip_prompt.txt" \
    >"$output_file" 2>&1 || true

  local speed

  speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
  local response
  response=$(grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:" "$output_file" | head -20)

  echo "Speed: $speed"
  echo "--- Response ---"
  echo "$response" | head -10

  # Auto-score if function provided
  if [[ -n "$score_func" ]]; then
    local score_result
    case "$score_func" in
      json_only)
        score_result=$(score_json_only "$response")
        ;;
      count_items)
        score_result=$(score_count_items "$response" "$score_args")
        ;;
      exact_match)
        score_result=$(score_exact_match "$response" "$score_args")
        ;;
      forbidden_words)
        score_result=$(score_forbidden_words "$response" "$score_args")
        ;;
      word_count)
        IFS=':' read -r min max <<<"$score_args"
        score_result=$(score_word_count "$response" "$min" "$max")
        ;;
      *)
        score_result="MANUAL"
        ;;
    esac
    echo "--- Auto-Score: $score_result ---"
    echo "AUTO_SCORE: $score_result" >>"$output_file"
  fi
}

# Track scores
declare -A SCORES

for CONFIG in "${CONFIGS[@]}"; do
  echo ""
  echo "##############################################"
  echo "# Configuration: $CONFIG"
  if [[ "$CONFIG" =~ ^moe ]] && type get_moe_override &>/dev/null; then
    echo "# Override: $(get_moe_override "$CONFIG" "$MODEL_ARCH")"
  fi
  echo "##############################################"

  # T1: Basic Format Compliance
  echo ""
  echo "========== TIER 1 (Basic Format) =========="

  run_test "t1_q1_json_only" \
    "Convert this to JSON: Name is Alice, age is 30, city is Boston.
Output ONLY the JSON. No explanation, no markdown, no extra text." \
    "$CONFIG" "json_only"

  run_test "t1_q2_exact_count" \
    "List exactly 3 benefits of exercise. No more, no less." \
    "$CONFIG" "count_items" "3"

  run_test "t1_q3_single_word" \
    "Is 17 a prime number? Answer with only 'yes' or 'no', nothing else." \
    "$CONFIG" "exact_match" "yes,Yes,YES"

  # T2: Complex Constraints
  echo ""
  echo "========== TIER 2 (Complex Constraints) =========="

  run_test "t2_q1_negative_instruction" \
    "Explain photosynthesis in 2 sentences. Do NOT mention sunlight, sun, or solar." \
    "$CONFIG" "forbidden_words" "sunlight,sun,solar"

  run_test "t2_q2_word_limit" \
    "Describe machine learning in exactly 20-25 words. Count carefully." \
    "$CONFIG" "word_count" "20:25"

  run_test "t2_q3_structured_format" \
    "List 3 programming languages with their release year.
Format EXACTLY as: LANGUAGE (YEAR)
One per line, no bullets, no numbers, no extra text." \
    "$CONFIG" "count_items" "3"

  run_test "t2_q4_multiple_constraints" \
    "Name 5 countries in Europe.
- Exactly 5 countries
- Alphabetical order
- One per line
- No additional text or punctuation" \
    "$CONFIG" "count_items" "5"

  # T3: Adversarial Compliance
  echo ""
  echo "========== TIER 3 (Adversarial) =========="

  run_test "t3_q1_resist_elaboration" \
    "What is 2+2? Reply with only the number, nothing else." \
    "$CONFIG" "exact_match" "4"

  run_test "t3_q2_maintain_format" \
    "Analyze this error and respond in EXACTLY this format:
ERROR_TYPE: <type>
ROOT_CAUSE: <one sentence>
FIX: <one sentence>

No other text. Error: TypeError: Cannot read property 'map' of undefined" \
    "$CONFIG"

  run_test "t3_q3_empty_handling" \
    "Extract all email addresses from this text: 'Hello world, nice day!'
If none found, output exactly: NONE
Do not explain or apologize." \
    "$CONFIG" "exact_match" "NONE"

  run_test "t3_q4_conflicting_constraints" \
    "Write one sentence about databases that:
- Is under 12 words
- Starts with 'A'
- Ends with 'data'
- Does not use 'store' or 'storage'" \
    "$CONFIG" "word_count" "1:12"

  # Score Summary
  echo ""
  echo "=============================================="
  echo "CONFIG $CONFIG - AUTO-SCORE SUMMARY"
  echo "=============================================="

  t1_pass=0
  t2_pass=0
  t3_pass=0

  for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_t1_*.txt; do
    [[ -f "$f" ]] && grep -q "AUTO_SCORE: PASS" "$f" && ((t1_pass++)) || true
  done
  for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_t2_*.txt; do
    [[ -f "$f" ]] && grep -q "AUTO_SCORE: PASS" "$f" && ((t2_pass++)) || true
  done
  for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_t3_*.txt; do
    [[ -f "$f" ]] && grep -q "AUTO_SCORE: PASS" "$f" && ((t3_pass++)) || true
  done

  echo "T1 (Basic): $t1_pass/3 passed"
  echo "T2 (Complex): $t2_pass/4 passed"
  echo "T3 (Adversarial): $t3_pass/4 passed"
  total_pass=$((t1_pass + t2_pass + t3_pass))
  echo "Total: $total_pass/11"

  max_speed=0 q_count=0
  for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_*.txt; do
    if [[ -f "$f" ]]; then
      ((q_count++)) || true
      speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")
      [[ -n "$speed" ]] && (($(echo "$speed > $max_speed" | bc -l 2>/dev/null || echo 0))) && max_speed="$speed"
    fi
  done

  # Add discovery info for spec_k configs: draft_model,K=N,T=X
  discovery_info="-"
  if [[ "$CONFIG" =~ ^spec_k([0-9]+) ]]; then
    k_val="${BASH_REMATCH[1]}"
    draft_name=$(basename "${DRAFT_MODEL_PATH:-unknown}" .gguf | cut -c1-15)
    discovery_info="${draft_name},K=${k_val},T=0.1"
  fi
  printf "%-10s %-20s %-8s %2d %8s %-30s\n" "instprec" "${MODEL_NAME:0:20}" "$CONFIG" "$q_count" "${max_speed}" "$discovery_info" >>/mnt/raid0/llm/tmp/benchmark_completions.log
done

echo ""
echo "=============================================="
echo "INSTRUCTION PRECISION RUBRIC COMPLETE"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
