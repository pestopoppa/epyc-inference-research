#!/bin/bash
# General/Instruct Model Quality Rubric Test Script
# Runs all T1/T2/T3 general instruction-following questions
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
OUTPUT_DIR="/mnt/raid0/llm/tmp/general_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=120

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
echo "General/Instruct Model Quality Rubric"
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
  echo "$prompt" >"/mnt/raid0/llm/tmp/general_prompt.txt"

  # Run model and capture output
  timeout "$TIMEOUT" OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_COMPLETION" \
    -m "$MODEL" \
    -t 96 -n 512 --temp 0.3 \
    $moe_override \
    -f "/mnt/raid0/llm/tmp/general_prompt.txt" \
    >"$output_file" 2>&1 || true

  # Extract timing
  local speed
  speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

  echo "Speed: $speed"
  echo "Output saved to: $output_file"

  # Show the answer
  echo "--- Answer ---"
  grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:" "$output_file" | tail -25 | head -20
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

  run_test "t1_q1_reformat" \
    "Convert this to a bullet list:
The meeting covered three topics. First, we discussed the Q3 budget which is \$50K over.
Second, we reviewed hiring plans for two engineers. Third, we set the launch date for March 15." "$CONFIG"

  run_test "t1_q2_summarize" \
    "Summarize in one sentence:
The new caching layer reduced API latency from 200ms to 45ms, improving user experience
significantly. However, it increased memory usage by 2GB per server, requiring us to
upgrade our instance types from m5.large to m5.xlarge, adding \$500/month to our AWS bill." "$CONFIG"

  run_test "t1_q3_extract" \
    "Extract all email addresses from this text:
Contact John at john.doe@company.com for sales inquiries. For support, reach out to
support@company.com or help-desk@company.com. Press inquiries: press@external.org" "$CONFIG"

  # T2: Medium-Hard questions
  echo ""
  echo "========== TIER 2 (Medium-Hard) =========="

  run_test "t2_q1_json" \
    "Parse this into JSON with fields: name, role, department, start_date
'Sarah Chen joined as Senior Engineer in the Platform team on 2024-03-15.'
Output only the JSON, no explanation." "$CONFIG"

  run_test "t2_q2_multistep" \
    "Process this list:
1. Remove duplicates (case-insensitive)
2. Sort alphabetically
3. Number each item
4. Add a count at the end

Items: banana, Apple, cherry, BANANA, apple, Date, cherry" "$CONFIG"

  run_test "t2_q3_compare" \
    "Compare these two approaches in 2-3 sentences:
Approach A: Microservices - Each feature is a separate service with its own database.
Pros: Independent scaling, isolated failures. Cons: Network overhead, data consistency challenges.

Approach B: Monolith - Single application with shared database.
Pros: Simple deployment, easy data joins. Cons: Scaling limitations, coupled codebase." "$CONFIG"

  # T3: Hard questions
  echo ""
  echo "========== TIER 3 (Hard) =========="

  run_test "t3_q1_synthesis" \
    "Synthesize these three perspectives into a unified recommendation:

Engineering: 'We need 3 months to refactor the auth system properly. Rushing will create tech debt.'
Product: 'Customers are churning due to login issues. We need a fix in 2 weeks.'
Finance: 'Q4 budget is tight. Any solution over \$20K needs board approval.'

Provide a concrete recommendation in 3-4 sentences." "$CONFIG"

  run_test "t3_q2_transform" \
    "Transform this flat data into nested YAML grouped by department:

employees:
- name: Alice, dept: Engineering, level: Senior
- name: Bob, dept: Sales, level: Junior
- name: Carol, dept: Engineering, level: Junior
- name: Dave, dept: Sales, level: Senior" "$CONFIG"

  run_test "t3_q3_schedule" \
    "Schedule these meetings given constraints:
- Team sync (60min): Must include Alice, Bob, Carol
- 1:1 Alice-Dave (30min)
- 1:1 Bob-Dave (30min)
- Dave only available 9-11am and 2-4pm
- Alice unavailable 10-11am
- No back-to-back meetings for anyone

Available slots: 9am, 9:30am, 10am, 10:30am, 11am, 2pm, 2:30pm, 3pm, 3:30pm

Output a valid schedule or explain why impossible." "$CONFIG"

  run_test "t3_q4_inconsistency" \
    "These 3 documents describe the same system. Find inconsistencies:

Doc A: 'The API accepts POST requests with JSON body. Rate limit is 100 req/min.'
Doc B: 'Send data via POST with form-encoded body. Rate limit is 100 requests per minute.'
Doc C: 'API endpoint accepts POST. JSON payload required. Rate limited to 1000 req/hour.'

List all inconsistencies found." "$CONFIG"

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
  printf "%-10s %-20s %-8s %2d %8s %-30s\n" "general" "${MODEL_NAME:0:20}" "$CONFIG" "$q_count" "${max_speed}" "$discovery_info" >>/mnt/raid0/llm/tmp/benchmark_completions.log
done

# Final Summary
echo ""
echo "=============================================="
echo "GENERAL RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
