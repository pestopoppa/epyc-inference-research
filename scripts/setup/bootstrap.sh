#!/bin/bash
# Bootstrap script for orchestration system setup
#
# This script:
# 1. Verifies prerequisites (Python, system tools)
# 2. Creates directory structure
# 3. Sets up environment configuration
# 4. Installs Python dependencies
# 5. Verifies llama.cpp binaries
# 6. Runs verification gates
#
# Usage:
#   ./scripts/setup/bootstrap.sh              # Full setup
#   ./scripts/setup/bootstrap.sh --create-dirs  # Only create directories
#   ./scripts/setup/bootstrap.sh --check-only   # Only check prerequisites

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source environment library for path variables (if exists)
# shellcheck source=../lib/env.sh
if [[ -f "${SCRIPT_DIR}/../lib/env.sh" ]]; then
  source "${SCRIPT_DIR}/../lib/env.sh"
fi

# Default LLM root (can be overridden by env.sh or .env)
DEFAULT_LLM_ROOT="${LLM_ROOT:-/mnt/raid0/llm}"

log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
  local cmd=$1
  local pkg=${2:-$1}
  if command -v "$cmd" &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} $cmd"
    return 0
  else
    echo -e "  ${RED}✗${NC} $cmd (install: $pkg)"
    return 1
  fi
}

check_prerequisites() {
  log_info "Checking prerequisites..."
  local missing=0

  echo "Required:"
  check_command python3 "python3" || ((missing++))
  if command -v uv &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} uv"
  elif command -v pip3 &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} pip3"
  elif command -v pip &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} pip"
  else
    echo -e "  ${RED}✗${NC} uv or pip (install: uv or python3-pip)"
    ((missing++))
  fi
  check_command git "git" || ((missing++))
  check_command cmake "cmake" || ((missing++))

  echo ""
  echo "Recommended:"
  check_command numactl "apt install numactl" || log_warn "numactl not found - needed for NUMA systems"
  check_command lsof "apt install lsof" || log_warn "lsof not found - needed for port checking"

  echo ""
  echo "Required for make gates (bootstrap can run without these):"
  check_command shellcheck "apt install shellcheck" || true
  check_command shfmt "go install mvdan.cc/sh/v3/cmd/shfmt@latest" || true
  check_command markdownlint "npm install -g markdownlint-cli" || true

  if [[ $missing -gt 0 ]]; then
    log_error "Missing $missing required tool(s). Please install them first."
    return 1
  fi

  # Check Python version
  local py_version
  py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [[ "$(echo "$py_version >= 3.11" | bc -l 2>/dev/null || echo 0)" -eq 1 ]] ||
    python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo -e "  ${GREEN}✓${NC} Python $py_version (>= 3.11)"
  else
    log_error "Python 3.11+ required, found $py_version"
    return 1
  fi

  return 0
}

setup_environment() {
  log_info "Setting up environment..."

  local env_file="$PROJECT_ROOT/.env"
  local env_example="$PROJECT_ROOT/.env.example"

  if [[ -f "$env_file" ]]; then
    log_info "Found existing .env file"
    # Source it to get LLM_ROOT
    set -a
    # shellcheck disable=SC1090
    source "$env_file" 2>/dev/null || true
    set +a
  elif [[ -f "$env_example" ]]; then
    log_info "Creating .env from .env.example"
    cp "$env_example" "$env_file"

    echo ""
    log_warn "Please edit .env to set your paths:"
    echo "  ORCHESTRATOR_PATHS_LLM_ROOT - Root directory for LLM files"
    echo "  ORCHESTRATOR_PATHS_PROJECT_ROOT - This repository location"
    echo ""
    echo "Current defaults assume: $DEFAULT_LLM_ROOT"
    echo ""
    read -p "Press Enter to continue with defaults, or Ctrl+C to edit .env first... "

    set -a
    # shellcheck disable=SC1090
    source "$env_file" 2>/dev/null || true
    set +a
  else
    log_warn "No .env.example found, using default paths"
  fi

  # Set defaults if not in environment
  export ORCHESTRATOR_PATHS_LLM_ROOT="${ORCHESTRATOR_PATHS_LLM_ROOT:-$DEFAULT_LLM_ROOT}"
  export ORCHESTRATOR_PATHS_PROJECT_ROOT="${ORCHESTRATOR_PATHS_PROJECT_ROOT:-$PROJECT_ROOT}"

  log_info "Using LLM_ROOT: $ORCHESTRATOR_PATHS_LLM_ROOT"
  log_info "Using PROJECT_ROOT: $ORCHESTRATOR_PATHS_PROJECT_ROOT"
}

create_directories() {
  log_info "Creating directory structure..."

  local llm_root="${ORCHESTRATOR_PATHS_LLM_ROOT:-$DEFAULT_LLM_ROOT}"

  local dirs=(
    "$llm_root/models"
    "$llm_root/lmstudio/models"
    "$llm_root/cache/huggingface"
    "$llm_root/cache/pip"
    "$llm_root/cache/prefixes"
    "$llm_root/cache/drafts"
    "$llm_root/tmp"
    "$llm_root/tmp/archives"
    "$llm_root/tmp/pdf_router"
    "$llm_root/tmp/claude/artifacts"
    "$llm_root/vision"
  )

  for dir in "${dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
      mkdir -p "$dir"
      echo -e "  ${GREEN}+${NC} $dir"
    else
      echo -e "  ${GREEN}✓${NC} $dir (exists)"
    fi
  done

  # Project-specific directories
  local project_root="${ORCHESTRATOR_PATHS_PROJECT_ROOT:-$PROJECT_ROOT}"
  local project_dirs=(
    "$project_root/logs"
    "$project_root/cache"
    "$project_root/share"
    "$project_root/state"
    "$project_root/orchestration/repl_memory/sessions"
  )

  for dir in "${project_dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
      mkdir -p "$dir"
      echo -e "  ${GREEN}+${NC} $dir"
    else
      echo -e "  ${GREEN}✓${NC} $dir (exists)"
    fi
  done
}

install_python_deps() {
  log_info "Installing Python dependencies..."

  cd "$PROJECT_ROOT"

  # Check for uv first
  if command -v uv &>/dev/null; then
    log_info "Using uv for installation..."
    uv sync
  elif [[ -f "pyproject.toml" ]]; then
    log_info "Using pip for installation..."
    pip install -e ".[dev]"
  elif [[ -f "requirements.txt" ]]; then
    log_info "Using pip with requirements.txt..."
    pip install -r requirements.txt
  else
    log_warn "No dependency file found (pyproject.toml or requirements.txt)"
    log_info "Installing minimal dependencies..."
    pip install pyyaml pydantic pydantic-settings
  fi
}

check_llama_cpp() {
  log_info "Checking llama.cpp..."

  local llm_root="${ORCHESTRATOR_PATHS_LLM_ROOT:-$DEFAULT_LLM_ROOT}"
  local llama_dir="$llm_root/llama.cpp"
  local bin_dir="$llama_dir/build/bin"

  if [[ ! -d "$llama_dir" ]]; then
    log_warn "llama.cpp not found at $llama_dir"
    echo ""
    echo "To set up llama.cpp:"
    echo "  git clone https://github.com/pestopoppa/llama.cpp.git $llama_dir"
    echo "  cd $llama_dir"
    echo "  git checkout production-consolidated"
    echo "  cmake -B build -DLLAMA_AVX512=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build -j\$(nproc)"
    echo ""
    return 1
  fi

  # Check branch
  local current_branch
  current_branch=$(cd "$llama_dir" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
  if [[ "$current_branch" != "production-consolidated" ]]; then
    log_warn "llama.cpp is on branch '$current_branch', expected 'production-consolidated'"
  else
    echo -e "  ${GREEN}✓${NC} llama.cpp branch: production-consolidated"
  fi

  # Check binaries
  local required_bins=("llama-server" "llama-cli" "llama-speculative")
  local missing_bins=0

  for bin in "${required_bins[@]}"; do
    if [[ -x "$bin_dir/$bin" ]]; then
      echo -e "  ${GREEN}✓${NC} $bin"
    else
      echo -e "  ${RED}✗${NC} $bin not found"
      ((missing_bins++))
    fi
  done

  if [[ $missing_bins -gt 0 ]]; then
    log_warn "$missing_bins binary(ies) missing. Build llama.cpp:"
    echo "  cd $llama_dir && cmake --build build -j\$(nproc)"
    return 1
  fi

  return 0
}

run_gates() {
  log_info "Running verification gates..."

  cd "$PROJECT_ROOT"

  if [[ -f "Makefile" ]] && grep -q "^gates:" Makefile; then
    make gates
  else
    log_warn "No Makefile with gates target found"

    # Run basic checks
    log_info "Running basic Python import check..."
    python3 -c "from src.config import get_config; print('Config loaded:', get_config().paths.project_root)"
  fi
}

show_next_steps() {
  echo ""
  log_info "Setup complete!"
  echo ""
  echo "Next steps:"
  echo "  1. Download models:"
  echo "     python scripts/setup/download_models.py --tier hot"
  echo ""
  echo "  2. Start development server:"
  echo "     python scripts/server/orchestrator_stack.py start --dev"
  echo ""
  echo "  3. Or start production server (after downloading models):"
  echo "     python scripts/server/orchestrator_stack.py start --hot-only"
  echo ""
  echo "See docs/SETUP.md for detailed instructions."
}

main() {
  local check_only=false
  local create_dirs_only=false

  while [[ $# -gt 0 ]]; do
    case $1 in
      --check-only)
        check_only=true
        shift
        ;;
      --create-dirs)
        create_dirs_only=true
        shift
        ;;
      -h | --help)
        echo "Usage: $0 [--check-only] [--create-dirs]"
        echo ""
        echo "Options:"
        echo "  --check-only    Only check prerequisites"
        echo "  --create-dirs   Only create directory structure"
        exit 0
        ;;
      *)
        log_error "Unknown option: $1"
        exit 1
        ;;
    esac
  done

  echo "=== Orchestration System Bootstrap ==="
  echo ""

  if [[ "$check_only" == true ]]; then
    check_prerequisites
    exit $?
  fi

  setup_environment

  if [[ "$create_dirs_only" == true ]]; then
    create_directories
    exit $?
  fi

  check_prerequisites || exit 1
  create_directories
  install_python_deps
  check_llama_cpp || log_warn "llama.cpp setup incomplete - see above"
  run_gates || log_warn "Some gates failed - review output above"

  show_next_steps
}

main "$@"
