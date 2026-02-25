#!/bin/bash
# =============================================================================
# ADD MODEL TO BENCHMARK QUEUE
# =============================================================================
# Validates a model with a health check, then adds to the benchmark queue
# for the overnight script to pick up dynamically.
#
# Usage:
#   ./add_model_to_benchmark.sh /path/to/model.gguf ModelName arch
#
# Arguments:
#   model_path  - Full path to the GGUF model file
#   model_name  - Short name for results (e.g., "Qwen3-Coder-30B")
#   arch        - Architecture: dense, qwen3moe, qwen3next, glm4moe, etc.
#
# Examples:
#   ./add_model_to_benchmark.sh /path/to/model.gguf Qwen3-30B qwen3moe
#   ./add_model_to_benchmark.sh /path/to/model.gguf Llama-70B dense
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Configuration (using env vars from env.sh)
QUEUE_FILE="${LLM_ROOT}/tmp/benchmark_queue.txt"
QUEUE_LOCK="${LLM_ROOT}/tmp/benchmark_queue.lock"
LLAMA_COMPLETION="${LLAMA_CPP_BIN}/llama-completion"
HEALTH_CHECK_TIMEOUT=15
HEALTH_CHECK_PROMPT="Hello, please respond with a single word."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
  echo "Usage: $0 <model_path> <model_name> <arch> [moe_key]"
  echo ""
  echo "Arguments:"
  echo "  model_path  - Full path to the GGUF model file"
  echo "  model_name  - Short name for results (e.g., 'Qwen3-Coder-30B')"
  echo "  arch        - Architecture type:"
  echo "                  dense     - Standard dense model (no MoE override)"
  echo "                  qwen3moe  - Qwen3 MoE models"
  echo "                  qwen3next - Qwen3-Next SSM+MoE hybrid"
  echo "                  qwen2moe  - Qwen2 MoE models"
  echo "                  glm4moe   - GLM-4 MoE models"
  echo "                  deepseek2 - DeepSeek v2/v3 MoE models"
  echo "                  mixtral   - Mixtral MoE models"
  echo "                  custom    - Custom MoE (requires moe_key argument)"
  echo "  moe_key     - (Optional) Custom MoE override key for unknown architectures"
  echo "                  e.g., 'deepseek2.expert_used_count' or 'mixtral.num_experts_per_tok'"
  echo ""
  echo "Examples:"
  echo "  $0 /path/to/Qwen3-30B.gguf Qwen3-30B qwen3moe"
  echo "  $0 /path/to/Llama-70B.gguf Llama-70B dense"
  echo "  $0 /path/to/DeepSeek-R1-Distill-14B.gguf DeepSeek-R1-14B dense"
  echo "  $0 /path/to/NewMoE.gguf NewMoE custom newarch.expert_used_count"
  echo ""
  echo "To find the MoE key for an unknown model, check the GGUF metadata:"
  echo "  llama-cli -m model.gguf --verbose 2>&1 | grep -i expert"
  exit 1
}

log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Get MoE override key based on architecture
get_moe_override_key() {
  local arch="$1"
  local custom_key="${2:-}"
  case "$arch" in
    qwen3moe)
      echo "qwen3moe.expert_used_count"
      ;;
    qwen3next)
      echo "qwen3next.expert_used_count"
      ;;
    glm4moe)
      echo "glm4moe.expert_used_count"
      ;;
    qwen2moe)
      echo "qwen2moe.expert_used_count"
      ;;
    deepseek2)
      echo "deepseek2.expert_used_count"
      ;;
    mixtral)
      echo "mixtral.num_experts_per_tok"
      ;;
    custom)
      echo "$custom_key"
      ;;
    dense | *)
      echo ""
      ;;
  esac
}

# Build the health check command
build_health_check_command() {
  local model_path="$1"
  local arch="$2"
  local moe_key="$3"

  local cmd="timeout $HEALTH_CHECK_TIMEOUT env OMP_NUM_THREADS=1 numactl --interleave=all"
  cmd+=" $LLAMA_COMPLETION"
  cmd+=" -m \"$model_path\""
  cmd+=" -t 96 -n 32 --temp 0"
  cmd+=" -p \"$HEALTH_CHECK_PROMPT\""

  # Add MoE override for MoE architectures
  if [[ -n "$moe_key" ]]; then
    cmd+=" --override-kv ${moe_key}=int:4"
  fi

  echo "$cmd"
}

# Run health check
run_health_check() {
  local model_path="$1"
  local arch="$2"
  local moe_key="$3"

  local cmd

  cmd=$(build_health_check_command "$model_path" "$arch" "$moe_key")

  log_info "Running health check..."
  log_info "Command: $cmd"
  echo ""

  local output_file

  output_file=$(mktemp)
  local start_time
  start_time=$(date +%s)

  # Run the command and capture output
  if eval "$cmd" >"$output_file" 2>&1; then
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))

    # Check for actual output (not just llama.cpp logs)
    local response
    response=$(grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^system_info\|^main:" "$output_file" | head -5)

    if [[ -n "$response" ]]; then
      log_info "Health check PASSED (${duration}s)"
      echo "Response preview:"
      echo "$response" | head -3
      rm -f "$output_file"
      return 0
    else
      log_error "Health check produced no output"
      cat "$output_file"
      rm -f "$output_file"
      return 1
    fi
  else
    local exit_code=$?
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))

    if [[ $exit_code -eq 124 ]]; then
      log_error "Health check TIMEOUT after ${HEALTH_CHECK_TIMEOUT}s"
      log_error "Model may have interactive mode issues or be too slow"
    else
      log_error "Health check FAILED (exit code: $exit_code, ${duration}s)"
    fi

    echo "Output:"
    cat "$output_file" | tail -20
    rm -f "$output_file"
    return 1
  fi
}

# Add model to queue
add_to_queue() {
  local model_path="$1"
  local model_name="$2"
  local arch="$3"
  local moe_key="$4"
  local timestamp
  timestamp=$(date +%Y%m%d_%H%M%S)

  mkdir -p "$(dirname "$QUEUE_FILE")"

  # Use flock for atomic append
  (
    flock -x 200
    echo "${model_path}|${model_name}|${arch}|${moe_key}|${timestamp}" >>"$QUEUE_FILE"
  ) 200>"$QUEUE_LOCK"

  log_info "Added to queue: $QUEUE_FILE"
}

# Check if model is already in queue
check_already_queued() {
  local model_path="$1"

  if [[ -f "$QUEUE_FILE" ]]; then
    if grep -q "^${model_path}|" "$QUEUE_FILE" 2>/dev/null; then
      return 0 # Already queued
    fi
  fi
  return 1 # Not queued
}

# =============================================================================
# MAIN
# =============================================================================

if [[ $# -lt 3 ]]; then
  usage
fi

MODEL_PATH="$1"
MODEL_NAME="$2"
ARCH="$3"
CUSTOM_MOE_KEY="${4:-}"

echo "=============================================="
echo "ADD MODEL TO BENCHMARK QUEUE"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Path: $MODEL_PATH"
echo "Architecture: $ARCH"
if [[ -n "$CUSTOM_MOE_KEY" ]]; then
  echo "Custom MoE Key: $CUSTOM_MOE_KEY"
fi
echo "=============================================="
echo ""

# Validate model path
if [[ ! -f "$MODEL_PATH" ]]; then
  log_error "Model file not found: $MODEL_PATH"
  exit 1
fi

# Validate architecture
case "$ARCH" in
  dense | qwen3moe | qwen3next | glm4moe | qwen2moe | deepseek2 | mixtral) ;;
  custom)
    if [[ -z "$CUSTOM_MOE_KEY" ]]; then
      log_error "Architecture 'custom' requires a moe_key argument"
      log_error "Example: $0 model.gguf ModelName custom myarch.expert_used_count"
      exit 1
    fi
    ;;
  *)
    log_warn "Unknown architecture: $ARCH"
    log_warn "Supported: dense, qwen3moe, qwen3next, glm4moe, qwen2moe, deepseek2, mixtral, custom"
    log_warn "Proceeding with 'dense' behavior (no MoE override)"
    log_warn "If this is an MoE model, use: $0 model.gguf Name custom ARCH.expert_used_count"
    ;;
esac

# Check if already queued
if check_already_queued "$MODEL_PATH"; then
  log_warn "Model already in queue: $MODEL_PATH"
  echo "Queue contents:"
  cat "$QUEUE_FILE"
  exit 0
fi

# Get MoE override key
MOE_KEY=$(get_moe_override_key "$ARCH" "$CUSTOM_MOE_KEY")

if [[ -n "$MOE_KEY" ]]; then
  log_info "MoE architecture detected, will use: --override-kv ${MOE_KEY}=int:4"
else
  log_info "Dense architecture, no MoE override needed"
fi

echo ""

# Run health check
if run_health_check "$MODEL_PATH" "$ARCH" "$MOE_KEY"; then
  echo ""
  add_to_queue "$MODEL_PATH" "$MODEL_NAME" "$ARCH" "$MOE_KEY"
  echo ""
  log_info "SUCCESS! Model will be picked up by overnight benchmark."
  echo ""
  echo "Queue contents:"
  cat "$QUEUE_FILE"
else
  echo ""
  log_error "FAILED! Model not added to queue."
  log_error "Please check the model manually and resolve any issues."
  echo ""
  echo "Common issues:"
  echo "  - Model needs different MoE override key"
  echo "  - Model has interactive mode quirks"
  echo "  - Model file is corrupted or incompatible"
  echo ""
  echo "Debug command:"
  echo "  $(build_health_check_command "$MODEL_PATH" "$ARCH" "$MOE_KEY")"
  exit 1
fi
