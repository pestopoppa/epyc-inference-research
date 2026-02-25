#!/bin/bash
# VL Model Quality Rubric Test Script
# Runs all T1/T2/T3 visual questions and captures results with timing
#
# Single Config Mode: Pass config as 5th param
# Multi Config Mode: Omit 5th param, uses shared lib to determine configs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
MODEL="${1:-}"
MMPROJ="${2:-}"
MODEL_NAME="${3:-unknown}"
MODEL_ARCH="${4:-dense}" # dense, qwen3vlmoe
CONFIG_PARAM="${5:-}"    # Optional: specific config to run (single config mode)
OUTPUT_DIR="/mnt/raid0/llm/tmp/vl_rubric_results"
IMAGE_DIR="/mnt/raid0/llm/claude/test_images/vl_rubric"
LLAMA_MTMD="/mnt/raid0/llm/llama.cpp/build/bin/llama-mtmd-cli"
TIMEOUT=120

# Source shared libraries
if [[ -f "$SCRIPT_DIR/lib/optimization_configs.sh" ]]; then
  source "$SCRIPT_DIR/lib/optimization_configs.sh"
fi

if [[ -z "$MODEL" ]] || [[ -z "$MMPROJ" ]]; then
  echo "Usage: $0 <model.gguf> <mmproj.gguf> <model_name> [arch] [config]"
  echo ""
  echo "Architecture types:"
  echo "  dense       - Standard dense model (no MoE optimization)"
  echo "  qwen3vlmoe  - Qwen3-VL-MoE model (tests baseline + MoE reduction)"
  echo ""
  echo "Single config mode: Specify config as 5th param"
  echo "Multi config mode: Omit 5th param, let shared lib determine configs"
  echo ""
  echo "Examples:"
  echo "  $0 /path/to/Qwen2.5-VL-7B.gguf /path/to/mmproj.gguf Qwen2.5-VL-7B dense baseline"
  echo "  $0 /path/to/Qwen3-VL-30B-A3B.gguf /path/to/mmproj.gguf Qwen3-VL-30B qwen3vlmoe moe4"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Determine configurations to test
declare -a CONFIGS
if [[ -n "$CONFIG_PARAM" ]]; then
  # Single config mode - caller specifies exactly which config
  CONFIGS=("$CONFIG_PARAM")
  echo "[SINGLE CONFIG MODE] Running only: $CONFIG_PARAM"
else
  # Multi-config mode - use shared library if available
  if type setup_configs &>/dev/null; then
    setup_configs "$MODEL_NAME" "$MODEL_ARCH"
  else
    # Fallback if shared lib not available
    CONFIGS=("baseline")
    case "$MODEL_ARCH" in
      qwen3vlmoe)
        for exp in 2 4 6 8; do
          CONFIGS+=("moe${exp}")
        done
        ;;
    esac
  fi
fi

echo "=============================================="
echo "VL Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Model file: $MODEL"
echo "MMProj: $MMPROJ"
echo "Architecture: $MODEL_ARCH"
echo "Configurations to test: ${CONFIGS[*]}"
echo "Date: $(date)"
echo "=============================================="

# Function to run a VL test and extract timing
run_vl_test() {
  local test_name="$1"
  local image="$2"
  local prompt="$3"
  local config="$4"
  local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

  echo ""
  echo "--- Running $test_name ($config) ---"
  echo "Image: $image"
  echo "Prompt: $prompt"

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

  # Compute MoE override if needed
  local moe_override=""
  if [[ "$config" =~ ^moe([0-9]+) ]]; then
    local exp="${BASH_REMATCH[1]}"
    if type get_moe_override &>/dev/null; then
      moe_override=$(get_moe_override "$config" "$MODEL_ARCH")
    else
      # Fallback
      case "$MODEL_ARCH" in
        qwen3vlmoe)
          moe_override="--override-kv qwen3vlmoe.expert_used_count=int:$exp"
          ;;
      esac
    fi
  fi

  # Run model and capture output
  timeout "$TIMEOUT" "$LLAMA_MTMD" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --image "$image" \
    -t 96 -n 256 --temp 0.3 \
    $moe_override \
    -p "$prompt" \
    >"$output_file" 2>&1 || true

  # Extract timing
  local speed
  speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

  echo "Speed: $speed"
  echo "Output saved to: $output_file"

  # Show the answer (last 20 lines before timing info)
  echo "--- Answer ---"
  grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:\|^==" "$output_file" | tail -20 | head -15
}

# Verify images exist
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "ERROR: Test images not found at $IMAGE_DIR"
  echo "Run: python scripts/benchmark/generate_vl_test_images.py"
  exit 1
fi

# Run tests for each configuration
for CONFIG in "${CONFIGS[@]}"; do
  echo ""
  echo "##############################################"
  echo "# Configuration: $CONFIG"
  if [[ "$CONFIG" =~ ^moe ]]; then
    if type get_moe_override &>/dev/null; then
      echo "# Override: $(get_moe_override "$CONFIG" "$MODEL_ARCH")"
    fi
  fi
  echo "##############################################"

  # T1: Baseline questions
  echo ""
  echo "========== TIER 1 (Baseline) =========="

  run_vl_test "t1_q1_ocr" \
    "$IMAGE_DIR/text_simple.png" \
    "What text is shown in this image? Just give the exact text." \
    "$CONFIG"

  run_vl_test "t1_q2_shapes" \
    "$IMAGE_DIR/shapes_basic.png" \
    "How many shapes are in this image? List them with colors." \
    "$CONFIG"

  run_vl_test "t1_q3_icon" \
    "$IMAGE_DIR/icon_folder.png" \
    "What does this icon represent?" \
    "$CONFIG"

  # T2: Medium-Hard questions
  echo ""
  echo "========== TIER 2 (Medium-Hard) =========="

  run_vl_test "t2_q1_chart" \
    "$IMAGE_DIR/chart_bar.png" \
    "What is the value of bar B? Which bar has the highest value?" \
    "$CONFIG"

  run_vl_test "t2_q2_invoice" \
    "$IMAGE_DIR/doc_invoice.png" \
    "Extract the total amount from this invoice." \
    "$CONFIG"

  run_vl_test "t2_q3_code" \
    "$IMAGE_DIR/code_python.png" \
    "What does this code do? Is there a bug?" \
    "$CONFIG"

  # T3: Hard questions
  echo ""
  echo "========== TIER 3 (Hard) =========="

  run_vl_test "t3_q1_math" \
    "$IMAGE_DIR/math_equation.png" \
    "Solve the equation shown in the image. Show your work." \
    "$CONFIG"

  run_vl_test "t3_q2_flowchart" \
    "$IMAGE_DIR/diagram_flowchart.png" \
    "Trace the path if input > 10 and flag = true. What path do you take?" \
    "$CONFIG"

  run_vl_test "t3_q3_diff" \
    "$IMAGE_DIR/diff_images.png" \
    "Find all differences between Image A and Image B." \
    "$CONFIG"

  run_vl_test "t3_q4_puzzle" \
    "$IMAGE_DIR/puzzle_grid.png" \
    "What shape should go in the empty cell marked with '?'? Explain the pattern." \
    "$CONFIG"

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
  printf "%-10s %-20s %-8s %2d %8s %-30s\n" "vl" "${MODEL_NAME:0:20}" "$CONFIG" "$q_count" "${max_speed}" "$discovery_info" >>/mnt/raid0/llm/tmp/benchmark_completions.log
done

# Final Summary
echo ""
echo "=============================================="
echo "VL RUBRIC TEST COMPLETE"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
