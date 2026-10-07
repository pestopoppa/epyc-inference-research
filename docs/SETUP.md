# Setup Guide

Complete setup instructions for the AMD EPYC 9655 Inference Optimization project.

## Prerequisites

### Required

- **Python 3.11+**
- **git**
- **cmake** (for building llama.cpp)
- **C++ compiler** with AVX-512 support (GCC 11+ or Clang 15+)

### Recommended

- **numactl** — required for NUMA memory interleaving (`apt install numactl`)
- **lsof** — used for port checking (`apt install lsof`)

### Required for `make gates`

- **shellcheck** — shell script linting (`apt install shellcheck`)
- **shfmt** — shell format checking (`go install mvdan.cc/sh/v3/cmd/shfmt@latest`); `make shfmt` applies fixes
- **markdownlint** — markdown linting (`npm install -g markdownlint-cli`)

Missing linters fail the local gates. NextPLAID reindexing is a separate service operation:
run `make nextplaid-reindex` when NextPLAID is available on `:8088`.

## Quick Setup

```bash
# 1. Clone and configure
git clone <repo-url> && cd claude
cp .env.example .env   # Edit paths for your system

# 2. Install Python dependencies
pip install -e ".[dev]"   # or: uv sync

# 3. Verify setup
make validate-paths && make gates
```

Or use the bootstrap script for a guided setup:

```bash
./scripts/setup/bootstrap.sh
```

The bootstrap script will:

1. Verify prerequisites (Python, system tools)
2. Create the directory structure on `/mnt/raid0/`
3. Set up environment configuration from `.env.example`
4. Install Python dependencies
5. Verify llama.cpp binaries
6. Run verification gates

Use `--check-only` to just check prerequisites, or `--create-dirs` to only create directories.

## Environment Configuration

All paths are configured via environment variables. Key variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORCHESTRATOR_PATHS_LLM_ROOT` | `/mnt/raid0/llm` | Root directory for all LLM files |
| `ORCHESTRATOR_PATHS_PROJECT_ROOT` | (repo location) | This repository |
| `ORCHESTRATOR_PATHS_MODEL_BASE` | `${LLM_ROOT}/lmstudio/models` | GGUF model files |
| `ORCHESTRATOR_PATHS_LLAMA_CPP_BIN` | Unset; resolve the CPU backend through `kernels/production/cpu` | Optional explicit binary-directory override for diagnostics or isolated development |
| `HF_HOME` | `/mnt/raid0/llm/cache/huggingface` | HuggingFace cache |
| `TMPDIR` | `/mnt/raid0/llm/tmp` | Temporary files |

> **Critical**: All files must reside on `/mnt/raid0/`. The root filesystem is a 120GB SSD — writing large files there causes disk exhaustion. See [CLAUDE.md](../CLAUDE.md) for the full path policy.

## Building an experimental llama.cpp candidate

Production kernels are frozen and served from the kernel store at `/mnt/raid0/llm/kernels/production/<backend>`; do not build, modify, or benchmark in the production source tree. Kernel work starts by refreshing the `llama.cpp-experimental` worktree from the current production tip, then building and validating the complete candidate there against production on CPU and GPU. A validated candidate is promoted only as a new production version. Follow the EPYC root [Experimental Kernel Workflow](https://github.com/pestopoppa/epyc-root/blob/main/AGENTS.md#experimental-kernel-workflow--production-kernel-immutability).

Run the following commands only in the prepared `/mnt/raid0/llm/llama.cpp-experimental` tree. Each command holds the shared host's CPU 0–95 build claim while it runs:

```bash
/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 0-95 --role build -- \
  bash -c 'cd /mnt/raid0/llm/llama.cpp-experimental && cmake -B build -DLLAMA_AVX512=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"$(nproc)"'

/workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 0-95 --role build -- \
  bash -c 'cd /mnt/raid0/llm/llama.cpp-experimental && test -x build/bin/llama-server && test -x build/bin/llama-cli && LD_LIBRARY_PATH="$PWD/build/bin:${LD_LIBRARY_PATH:-}" ./build/bin/llama-cli --version'
```

The version example explicitly selects the experimental build's library directory; prove actual linkage with `epyc-inference-research/scripts/utils/verify_ggml_linkage.sh` before candidate comparisons or promotion evidence.

## Downloading Models

Models are authored in the canonical master at
`epyc-inference-research/orchestration/model_registry.yaml`; the orchestrator's
`orchestration/model_registry.yaml` is generated lean runtime output. Download by tier:

```bash
# HOT tier (~40GB, always resident) — minimum for development
python scripts/setup/download_models.py --tier hot

# WARM tier (~430GB) — full production stack
python scripts/setup/download_models.py --tier warm

# All models
python scripts/setup/download_models.py --tier all
```

See [MODEL_MANIFEST.md](MODEL_MANIFEST.md) for the role-based model configuration and substitution guide.

## Starting the Server Stack

```bash
# Development mode (0.5B draft model only, minimal RAM)
python3 scripts/server/orchestrator_stack.py start --dev

# HOT tier only (~40GB RAM)
python3 scripts/server/orchestrator_stack.py start --hot-only

# Check status
python3 scripts/server/orchestrator_stack.py status

# Stop all servers
python3 scripts/server/orchestrator_stack.py stop --all
```

## Verification

After setup, run the full gate suite:

```bash
make gates
```

This runs schema validation, shellcheck, formatting, and linting in order.

To validate just paths:

```bash
make validate-paths
```

## Container Setup

Docker and Nix targets are defined in the Makefile for reproducible environments:

```bash
# Docker
make docker-build && make docker-run    # Production API
make docker-dev                          # Development shell

# Nix
make nix-develop                         # Development shell
```

> Note: Container support requires Dockerfile/docker-compose.yml and flake.nix files, which are planned but not yet included in the repository.

## Testing

```bash
# Run all tests (uses -n 8 by default, safe for this machine)
pytest tests/

# Conservative parallelism
pytest tests/ -n 4
```

> **WARNING**: Never use `pytest -n auto` on this 192-thread machine. It spawns ~192 workers that exhaust 1.13TB RAM. See [CLAUDE.md](../CLAUDE.md) for details.

## Next Steps

- Read the [Getting Started guide](guides/getting-started.md) for project orientation
- Browse the [Chapter Index](chapters/INDEX.md) for research documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system internals
- See [Command Reference](reference/commands/QUICK_REFERENCE.md) for inference commands
