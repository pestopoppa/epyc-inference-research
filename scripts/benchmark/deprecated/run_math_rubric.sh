#!/bin/bash
# Math Model Quality Rubric Test Script
# Runs all T1/T2/T3 mathematical reasoning questions
#
# Single Config Mode: Pass config as 4th param
# Multi Config Mode: Omit 4th param, uses shared lib to determine configs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}" # dense, qwen3moe
CONFIG_PARAM="${4:-}"    # Optional: specific config to run (single config mode)
OUTPUT_DIR="/mnt/raid0/llm/tmp/math_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=180 # Math problems may need more time for reasoning

# Source shared libraries
if [[ -f "$SCRIPT_DIR/lib/optimization_configs.sh" ]]; then
  source "$SCRIPT_DIR/lib/optimization_configs.sh"
fi

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model.gguf> <model_name> [arch] [config]"
  echo ""
  echo "Architecture types:"
  echo "  dense       - Standard dense model (no MoE optimization)"
  echo "  qwen3moe    - Qwen3-MoE model (tests baseline + MoE reduction)"
  echo ""
  echo "Single config mode: Specify config as 4th param"
  echo "Multi config mode: Omit 4th param, let shared lib determine configs"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Determine configurations to test
declare -a CONFIGS
if [[ -n "$CONFIG_PARAM" ]]; then
  # Single config mode
  CONFIGS=("$CONFIG_PARAM")
  echo "[SINGLE CONFIG MODE] Running only: $CONFIG_PARAM"
else
  # Multi-config mode - use shared library if available
  if type setup_configs &>/dev/null; then
    setup_configs "$MODEL_NAME" "$MODEL_ARCH"
  else
    # Fallback
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
echo "Math Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Model file: $MODEL"
echo "Architecture: $MODEL_ARCH"
echo "Configurations to test: ${CONFIGS[*]}"
echo "Date: $(date)"
echo "=============================================="

# Function to run a test and extract timing
run_test() {
  local test_name="$1"
  local prompt="$2"
  local config="$3"
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

  # Write prompt to temp file
  echo "$prompt" >"/mnt/raid0/llm/tmp/math_prompt.txt"

  # Run model and capture output (longer timeout for math)
  timeout "$TIMEOUT" OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_COMPLETION" \
    -m "$MODEL" \
    -t 96 -n 1024 --temp 0.1 \
    $moe_override \
    -f "/mnt/raid0/llm/tmp/math_prompt.txt" \
    >"$output_file" 2>&1 || true

  # Extract timing
  local speed
  speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

  echo "Speed: $speed"
  echo "Output saved to: $output_file"

  # Show the answer
  echo "--- Answer ---"
  grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:" "$output_file" | tail -30 | head -25
}

# Run tests for each configuration
for CONFIG in "${CONFIGS[@]}"; do
  echo ""
  echo "##############################################"
  echo "# Configuration: $CONFIG"
  if [[ "$CONFIG" =~ ^moe ]] && type get_moe_override &>/dev/null; then
    echo "# Override: $(get_moe_override "$CONFIG" "$MODEL_ARCH")"
  fi
  echo "##############################################"

  # T1: Baseline questions
  echo ""
  echo "========== TIER 1 (Baseline) =========="

  run_test "t1_q1_arithmetic" \
    "Calculate: 847 × 23 + 156 ÷ 4 - 89

Show your work step by step, then give the final answer." "$CONFIG"

  run_test "t1_q2_algebra" \
    "Solve for x: 3x + 7 = 22

Show each step." "$CONFIG"

  run_test "t1_q3_conversion" \
    "Convert 2.5 kilometers to:
1. meters
2. centimeters
3. miles (use 1 mile = 1.609 km)

Show the conversion calculation for each." "$CONFIG"

  # T2: Medium-Hard questions
  echo ""
  echo "========== TIER 2 (Medium-Hard) =========="

  run_test "t2_q1_word_problem" \
    "A store offers a 20% discount on all items. After the discount, a 8% sales tax is applied. If an item originally costs \$150:

1. What is the price after discount?
2. What is the final price after tax?
3. What percentage of the original price is the final price?

Show your work for each part." "$CONFIG"

  run_test "t2_q2_system_equations" \
    "Solve the system of equations:
2x + 3y = 13
4x - y = 5

Find the values of x and y. Show all steps." "$CONFIG"

  run_test "t2_q3_probability" \
    "A bag contains 5 red balls, 3 blue balls, and 2 green balls.
If you draw 2 balls without replacement:

1. What is the probability both are red?
2. What is the probability of getting one red and one blue (in any order)?

Show your probability calculations." "$CONFIG"

  # T3: Hard questions
  echo ""
  echo "========== TIER 3 (Hard) =========="

  run_test "t3_q1_optimization" \
    "A farmer has 200 meters of fencing to enclose a rectangular field that borders a river (no fencing needed on the river side).

What dimensions maximize the enclosed area? What is the maximum area?

Set up the optimization problem and solve it." "$CONFIG"

  run_test "t3_q2_proof" \
    "Prove that the sum of the first n positive integers is n(n+1)/2.

Use mathematical induction. Show:
1. Base case
2. Inductive hypothesis
3. Inductive step" "$CONFIG"

  run_test "t3_q3_calculus" \
    "Evaluate the definite integral:
∫₀² (3x² + 2x - 1) dx

Show your work:
1. Find the antiderivative
2. Evaluate at the bounds
3. Calculate the final answer" "$CONFIG"

  run_test "t3_q4_statistics" \
    "Given the dataset: 12, 15, 18, 22, 25, 28, 30, 35

Calculate:
1. Mean
2. Median
3. Standard deviation (population)
4. Is there any outlier using the 1.5×IQR rule?

Show all calculations." "$CONFIG"

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
    discovery_info="${draft_name},K=${k_val},T=0.1"
  fi
  printf "%-10s %-20s %-8s %2d %8s %-30s\n" "math" "${MODEL_NAME:0:20}" "$CONFIG" "$q_count" "${max_speed}" "$discovery_info" >>/mnt/raid0/llm/tmp/benchmark_completions.log
done

# Final Summary
echo ""
echo "=============================================="
echo "MATH RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
