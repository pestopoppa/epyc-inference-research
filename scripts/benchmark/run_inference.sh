#!/bin/bash
# run_inference.sh — Optimized inference wrapper for EPYC 9655
# Usage:
#   ./run_inference.sh                    # Interactive mode with default model
#   ./run_inference.sh "Your prompt"      # Single prompt
#   ./run_inference.sh -m /path/model     # Specify model
#   ./run_inference.sh --spec             # Enable speculative decoding

set -euo pipefail

# Source environment and logging libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"
# shellcheck source=../utils/agent_log.sh
source "${SCRIPT_DIR}/../utils/agent_log.sh"

# Default configuration (use variables from env.sh)
MODEL_MAIN="${MODELS_DIR}/DeepSeek-R1-32B-Q4_K_M.gguf"
MODEL_DRAFT="${MODELS_DIR}/Qwen2.5-0.5B-Draft-Q4_K_M.gguf"
LLAMA_CLI="${LLAMA_CPP_BIN}/llama-cli"
THREADS=96
CONTEXT=32768
SPECULATIVE=0
SPEC_TOKENS=8
PROMPT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m | --model)
      MODEL_MAIN="$2"
      shift 2
      ;;
    -d | --draft)
      MODEL_DRAFT="$2"
      shift 2
      ;;
    -t | --threads)
      THREADS="$2"
      shift 2
      ;;
    -c | --context)
      CONTEXT="$2"
      shift 2
      ;;
    --spec | --speculative)
      SPECULATIVE=1
      shift
      ;;
    --spec-tokens)
      SPEC_TOKENS="$2"
      SPECULATIVE=1
      shift 2
      ;;
    -h | --help)
      echo "Usage: $0 [options] [prompt]"
      echo ""
      echo "Options:"
      echo "  -m, --model PATH       Main model path (default: DeepSeek-R1-32B)"
      echo "  -d, --draft PATH       Draft model for speculation"
      echo "  -t, --threads N        Thread count (default: 96)"
      echo "  -c, --context N        Context size (default: 32768)"
      echo "  --spec                 Enable speculative decoding"
      echo "  --spec-tokens N        Speculative tokens (default: 8)"
      echo "  -h, --help             Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 \"Explain quantum computing\""
      echo "  $0 --spec \"Write a poem about AI\""
      echo "  $0 -m /path/to/model.gguf -t 48"
      exit 0
      ;;
    *)
      PROMPT="$1"
      shift
      ;;
  esac
done

# Validate
if [[ ! -f "$LLAMA_CLI" ]]; then
  echo "ERROR: llama-cli not found at $LLAMA_CLI"
  echo "Build llama.cpp first."
  exit 1
fi

if [[ ! -f "$MODEL_MAIN" ]]; then
  echo "ERROR: Model not found at $MODEL_MAIN"
  exit 1
fi

if [[ $SPECULATIVE -eq 1 && ! -f "$MODEL_DRAFT" ]]; then
  echo "WARNING: Draft model not found at $MODEL_DRAFT"
  echo "Falling back to standard inference."
  SPECULATIVE=0
fi

# Critical environment variables
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Build command
CMD="numactl --interleave=all $LLAMA_CLI"
CMD+=" -m $MODEL_MAIN"
CMD+=" -t $THREADS"
CMD+=" -c $CONTEXT"
CMD+=" --mlock"
CMD+=" --color"

if [[ $SPECULATIVE -eq 1 ]]; then
  CMD+=" --draft $MODEL_DRAFT"
  CMD+=" --speculative $SPEC_TOKENS"
  echo "Mode: Speculative decoding (draft: $(basename $MODEL_DRAFT), tokens: $SPEC_TOKENS)"
else
  echo "Mode: Standard inference"
fi

echo "Model: $(basename $MODEL_MAIN)"
echo "Threads: $THREADS"
echo "Context: $CONTEXT"
echo "---"

# Log the inference configuration
agent_task_start "LLM Inference" "Running inference with optimized settings"
agent_observe "model" "$MODEL_MAIN"
agent_observe "threads" "$THREADS"
agent_observe "context" "$CONTEXT"
agent_observe "speculative" "$SPECULATIVE"
if [[ $SPECULATIVE -eq 1 ]]; then
  agent_observe "draft_model" "$MODEL_DRAFT"
  agent_observe "spec_tokens" "$SPEC_TOKENS"
fi

# Run
if [[ -n "$PROMPT" ]]; then
  # Single prompt mode
  agent_observe "mode" "single_prompt"
  exec $CMD -p "$PROMPT" -n 512
else
  # Interactive mode
  agent_observe "mode" "interactive"
  exec $CMD --interactive-first
fi
