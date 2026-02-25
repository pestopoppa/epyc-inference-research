#!/bin/bash
# =============================================================================
# Environment Library for Shell Scripts
# =============================================================================
#
# Source this file at the top of any script that needs path variables:
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/../lib/env.sh" 2>/dev/null || source "${SCRIPT_DIR}/../../scripts/lib/env.sh"
#
# Or if you know the project root:
#
#   source /path/to/project/scripts/lib/env.sh
#
# This provides:
#   - All ORCHESTRATOR_PATHS_* variables
#   - Convenience aliases (LLM_ROOT, PROJECT_ROOT, etc.)
#   - HuggingFace and cache directory exports
#
# =============================================================================

# Determine script location to find project root
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_ENV_SH_DIR}/../.." && pwd)"

# =============================================================================
# Load .env file if present
# =============================================================================

if [[ -f "${_PROJECT_ROOT}/.env" ]]; then
  # Export variables from .env (skip comments and empty lines)
  set -a
  # shellcheck disable=SC1091
  source "${_PROJECT_ROOT}/.env" 2>/dev/null || true
  set +a
fi

# =============================================================================
# Base Paths (with defaults)
# =============================================================================

# LLM root - all LLM-related files (models, cache, llama.cpp)
export ORCHESTRATOR_PATHS_LLM_ROOT="${ORCHESTRATOR_PATHS_LLM_ROOT:-/mnt/raid0/llm}"
export LLM_ROOT="${ORCHESTRATOR_PATHS_LLM_ROOT}"

# Project root - this repository
export ORCHESTRATOR_PATHS_PROJECT_ROOT="${ORCHESTRATOR_PATHS_PROJECT_ROOT:-${LLM_ROOT}/claude}"
export PROJECT_ROOT="${ORCHESTRATOR_PATHS_PROJECT_ROOT}"

# =============================================================================
# Derived Paths
# =============================================================================

# Model directories
export ORCHESTRATOR_PATHS_MODELS_DIR="${ORCHESTRATOR_PATHS_MODELS_DIR:-${LLM_ROOT}/models}"
export MODELS_DIR="${ORCHESTRATOR_PATHS_MODELS_DIR}"

export ORCHESTRATOR_PATHS_MODEL_BASE="${ORCHESTRATOR_PATHS_MODEL_BASE:-${LLM_ROOT}/lmstudio/models}"
export MODEL_BASE="${ORCHESTRATOR_PATHS_MODEL_BASE}"

# llama.cpp binaries
export ORCHESTRATOR_PATHS_LLAMA_CPP_BIN="${ORCHESTRATOR_PATHS_LLAMA_CPP_BIN:-${LLM_ROOT}/llama.cpp/build/bin}"
export LLAMA_CPP_BIN="${ORCHESTRATOR_PATHS_LLAMA_CPP_BIN}"

export ORCHESTRATOR_PATHS_LLAMA_SERVER="${ORCHESTRATOR_PATHS_LLAMA_SERVER:-${LLAMA_CPP_BIN}/llama-server}"
export LLAMA_SERVER="${ORCHESTRATOR_PATHS_LLAMA_SERVER}"

export ORCHESTRATOR_PATHS_LLAMA_MTMD="${ORCHESTRATOR_PATHS_LLAMA_MTMD:-${LLAMA_CPP_BIN}/llama-mtmd-cli}"
export LLAMA_MTMD="${ORCHESTRATOR_PATHS_LLAMA_MTMD}"

# Cache and temp directories
export ORCHESTRATOR_PATHS_CACHE_DIR="${ORCHESTRATOR_PATHS_CACHE_DIR:-${LLM_ROOT}/cache}"
export CACHE_DIR="${ORCHESTRATOR_PATHS_CACHE_DIR}"

export ORCHESTRATOR_PATHS_TMP_DIR="${ORCHESTRATOR_PATHS_TMP_DIR:-${LLM_ROOT}/tmp}"
export TMP_DIR="${ORCHESTRATOR_PATHS_TMP_DIR}"

export ORCHESTRATOR_PATHS_DRAFT_CACHE="${ORCHESTRATOR_PATHS_DRAFT_CACHE:-${CACHE_DIR}/drafts}"
export DRAFT_CACHE="${ORCHESTRATOR_PATHS_DRAFT_CACHE}"

# Project-specific paths
export ORCHESTRATOR_PATHS_LOG_DIR="${ORCHESTRATOR_PATHS_LOG_DIR:-${PROJECT_ROOT}/logs}"
export LOG_DIR="${ORCHESTRATOR_PATHS_LOG_DIR}"

export ORCHESTRATOR_PATHS_REGISTRY_PATH="${ORCHESTRATOR_PATHS_REGISTRY_PATH:-${PROJECT_ROOT}/orchestration/model_registry.yaml}"
export REGISTRY_PATH="${ORCHESTRATOR_PATHS_REGISTRY_PATH}"

export ORCHESTRATOR_PATHS_SESSIONS_DIR="${ORCHESTRATOR_PATHS_SESSIONS_DIR:-${PROJECT_ROOT}/orchestration/repl_memory/sessions}"
export SESSIONS_DIR="${ORCHESTRATOR_PATHS_SESSIONS_DIR}"

# Vision paths
export ORCHESTRATOR_PATHS_VISION_DIR="${ORCHESTRATOR_PATHS_VISION_DIR:-${LLM_ROOT}/vision}"
export VISION_DIR="${ORCHESTRATOR_PATHS_VISION_DIR}"

# Services paths
export ORCHESTRATOR_PATHS_ARCHIVE_EXTRACT="${ORCHESTRATOR_PATHS_ARCHIVE_EXTRACT:-${TMP_DIR}/archives}"
export ARCHIVE_EXTRACT_DIR="${ORCHESTRATOR_PATHS_ARCHIVE_EXTRACT}"

export ORCHESTRATOR_PATHS_PDF_ROUTER_TEMP="${ORCHESTRATOR_PATHS_PDF_ROUTER_TEMP:-${TMP_DIR}/pdf_router}"
export PDF_ROUTER_TEMP="${ORCHESTRATOR_PATHS_PDF_ROUTER_TEMP}"

# Path security prefix (empty to disable check)
export ORCHESTRATOR_PATHS_RAID_PREFIX="${ORCHESTRATOR_PATHS_RAID_PREFIX:-/mnt/raid0/}"

# =============================================================================
# HuggingFace & Cache Directories
# =============================================================================

export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_DIR}/pip}"

# XDG directories (redirect to project)
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${PROJECT_ROOT}/share}"
export XDG_STATE_HOME="${XDG_STATE_HOME:-${PROJECT_ROOT}/state}"

# Temp directory (critical: avoid root filesystem)
export TMPDIR="${TMPDIR:-${TMP_DIR}}"

# =============================================================================
# Convenience Functions
# =============================================================================

# Check if a path is under the required prefix (security check)
check_path_prefix() {
  local path="$1"
  local prefix="${ORCHESTRATOR_PATHS_RAID_PREFIX}"

  # If prefix is empty, skip check
  if [[ -z "$prefix" ]]; then
    return 0
  fi

  if [[ "$path" == "$prefix"* ]]; then
    return 0
  else
    echo "ERROR: Path '$path' is not under required prefix '$prefix'" >&2
    return 1
  fi
}

# Get absolute path to a binary in llama.cpp
llama_bin() {
  local bin_name="$1"
  echo "${LLAMA_CPP_BIN}/${bin_name}"
}

# Get absolute path to a model
model_path() {
  local relative_path="$1"
  echo "${MODEL_BASE}/${relative_path}"
}

# Ensure a directory exists (under raid prefix)
ensure_dir() {
  local dir="$1"
  if check_path_prefix "$dir"; then
    mkdir -p "$dir"
  fi
}

# =============================================================================
# Cleanup temporary variables
# =============================================================================

unset _ENV_SH_DIR
unset _PROJECT_ROOT
