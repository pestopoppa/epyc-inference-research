#!/bin/bash
# Start Whisper transcription server with EPYC 9655 optimizations
#
# Usage:
#   ./start_whisper_server.sh [--port PORT] [--model MODEL]
#
# Default:
#   Port: 9000
#   Model: large-v3-turbo (809M params, 6x faster than large-v3)

set -euo pipefail

# Script directory and env setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Defaults
PORT="${WHISPER_PORT:-9000}"
MODEL="${WHISPER_MODEL:-large-v3-turbo}"
THREADS="${WHISPER_THREADS:-64}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--port PORT] [--model MODEL] [--threads THREADS]"
      echo ""
      echo "Options:"
      echo "  --port     Server port (default: 9000)"
      echo "  --model    Whisper model (default: large-v3-turbo)"
      echo "  --threads  CPU threads (default: 64)"
      echo ""
      echo "Environment variables:"
      echo "  WHISPER_PORT, WHISPER_MODEL, WHISPER_THREADS"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# EPYC 9655 optimization: use 64 threads (not 192 - hyperthreading hurts here)
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

# Environment variables already set by env.sh:
# HF_HOME, TRANSFORMERS_CACHE, TMPDIR

echo "=============================================="
echo "Whisper Transcription Server"
echo "=============================================="
echo "Model:    $MODEL"
echo "Port:     $PORT"
echo "Threads:  $THREADS"
echo "HF Cache: $HF_HOME"
echo "=============================================="

# Check if port is already in use
if lsof -i ":$PORT" >/dev/null 2>&1; then
  echo "ERROR: Port $PORT is already in use"
  lsof -i ":$PORT"
  exit 1
fi

# Activate pace-env if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # Try common locations for the virtual environment
  for venv_path in \
    "${LLM_ROOT}/pace-env/bin/activate" \
    "$HOME/pace-env/bin/activate" \
    "${PROJECT_ROOT}/../pace-env/bin/activate"; do
    if [[ -f "$venv_path" ]]; then
      echo "Activating venv: $venv_path"
      source "$venv_path"
      break
    fi
  done
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "WARNING: Virtual environment not found, using system Python"
  fi
fi

# Check faster-whisper is installed
if ! python -c "import faster_whisper" 2>/dev/null; then
  echo "ERROR: faster-whisper not installed"
  echo "Run: pip install faster-whisper uvicorn fastapi python-multipart"
  exit 1
fi

# Run with NUMA interleaving for optimal memory bandwidth
echo "Starting server..."
exec numactl --interleave=all \
  python "$SCRIPT_DIR/whisper_server.py" \
  --port "$PORT" \
  --model "$MODEL"
