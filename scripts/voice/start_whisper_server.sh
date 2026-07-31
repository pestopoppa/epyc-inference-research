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
if [[ -f "${SCRIPT_DIR}/../lib/env.sh" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  # shellcheck source=../lib/env.sh
  source "${SCRIPT_DIR}/../lib/env.sh"
elif [[ -f "${SCRIPT_DIR}/../../lib/env.sh" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  # shellcheck source=../../lib/env.sh
  source "${SCRIPT_DIR}/../../lib/env.sh"
else
  echo "ERROR: Could not locate scripts/lib/env.sh from $SCRIPT_DIR"
  exit 1
fi

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

# STT runs LOCALLY. The model is passed by name, so huggingface_hub would try to
# reach huggingface.co on a cold start and this role fails *silently* (it is in
# OPTIONAL_AUXILIARY_ROLES). Weights are already in the local cache; pin offline
# so a network outage cannot take speech down. Set WHISPER_ALLOW_DOWNLOAD=1
# deliberately, and only when fetching a model we do not yet have.
if [[ "${WHISPER_ALLOW_DOWNLOAD:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

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

# Resolve Python. Prefer explicit WHISPER_PYTHON, then an active venv, then
# persistent venv locations that survive reboot.
PYTHON_BIN="${WHISPER_PYTHON:-python}"
if [[ -n "${WHISPER_PYTHON:-}" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: WHISPER_PYTHON is not executable: $PYTHON_BIN"
    exit 1
  fi
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  venv_candidates=()
  if [[ -n "${WHISPER_VENV:-}" ]]; then
    venv_candidates+=("${WHISPER_VENV}/bin/activate")
  fi
  venv_candidates+=(
    "${LLM_ROOT}/pace-env/bin/activate"
    "$HOME/pace-env/bin/activate"
    "${REPO_ROOT}/.venv/bin/activate"
    "${PROJECT_ROOT}/../pace-env/bin/activate"
  )

  for venv_path in "${venv_candidates[@]}"; do
    if [[ -f "$venv_path" ]]; then
      echo "Activating venv: $venv_path"
      # shellcheck disable=SC1090
      source "$venv_path"
      PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
      break
    fi
  done
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "WARNING: Virtual environment not found, using system Python"
  fi
fi

# Check faster-whisper is installed
if ! "$PYTHON_BIN" -c "import faster_whisper" 2>/dev/null; then
  echo "ERROR: faster-whisper not installed"
  echo "Run:"
  echo "  python3 -m venv ${LLM_ROOT}/pace-env"
  echo "  ${LLM_ROOT}/pace-env/bin/python -m pip install faster-whisper uvicorn fastapi python-multipart"
  exit 1
fi

echo "Starting server..."
server_cmd=("$PYTHON_BIN" "$SCRIPT_DIR/whisper_server.py" --port "$PORT" --model "$MODEL")
if command -v numactl >/dev/null 2>&1; then
  # Run with NUMA interleaving for optimal memory bandwidth.
  exec numactl --interleave=all "${server_cmd[@]}"
else
  echo "WARNING: numactl not found; starting without NUMA interleaving"
  exec "${server_cmd[@]}"
fi
