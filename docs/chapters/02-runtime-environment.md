# Chapter 02: Runtime Environment & Configuration

## Introduction

The orchestration system's runtime environment provides a hierarchical configuration system with feature flags, environment-based overrides, and safety guardrails. This chapter documents the environment setup, configuration architecture, and runtime tuning parameters that control system behavior.

The configuration system uses **pydantic-settings** for type-safe environment variable parsing with fallback to manual parsing when unavailable. Feature flags enable modular system components to be toggled independently for testing and production deployment.

## Feature Flag System

Ten independent feature flags let us turn orchestration modules on and off without touching code. In test mode everything defaults to off for isolation; in production everything flips on. The flags live in `src/features.py` and are validated at startup — if you enable `scripts` without `tools`, for example, initialization will complain.

<details>
<summary>Core feature flags table</summary>

| Flag | Environment Variable | Purpose | Dependencies |
|------|---------------------|---------|--------------|
| `memrl` | `ORCHESTRATOR_MEMRL` | Memory-based RL (TaskEmbedder, QScorer) | numpy, sqlite3, sentence-transformers |
| `tools` | `ORCHESTRATOR_TOOLS` | TOOL() function in REPL | None |
| `scripts` | `ORCHESTRATOR_SCRIPTS` | SCRIPT() function for prepared scripts | `tools` |
| `streaming` | `ORCHESTRATOR_STREAMING` | SSE /chat/stream endpoint | None |
| `openai_compat` | `ORCHESTRATOR_OPENAI_COMPAT` | OpenAI-compatible /v1/* endpoints | None |
| `repl` | `ORCHESTRATOR_REPL` | Python REPL execution environment | None |
| `caching` | `ORCHESTRATOR_CACHING` | LLM response caching | None |
| `restricted_python` | `ORCHESTRATOR_RESTRICTED_PYTHON` | Use RestrictedPython sandbox | RestrictedPython>=7.0 |
| `generation_monitor` | `ORCHESTRATOR_GENERATION_MONITOR` | Early failure detection (Phase 6) | None |
| `mock_mode` | `ORCHESTRATOR_MOCK_MODE` | Mock responses (test safety) | None |

</details>

<details>
<summary>Usage and validation patterns</summary>

<details>
<summary>Code: checking and validating feature flags</summary>

```python
from src.features import features

# Check if a feature is enabled
if features().memrl:
    from orchestration.repl_memory import TaskEmbedder
    embedder = TaskEmbedder()

# Get feature summary
enabled = features().enabled_features()
# Returns: ['repl', 'tools', 'caching', 'mock_mode']
```

```python
from src.features import get_features

features = get_features(production=True)
errors = features.validate()

if errors:
    # ['scripts feature requires tools feature']
    raise RuntimeError(f"Invalid configuration: {errors}")
```

</details>
</details>

## Hierarchical Configuration

Configuration is nested — LLM settings, escalation settings, REPL settings, and server settings each live in their own sub-config. Environment variables use double-underscore nesting (`ORCHESTRATOR_LLM__OUTPUT_CAP=4096`) to target specific leaves. The whole thing is backed by pydantic-settings so every value is type-checked and has sensible defaults.

<details>
<summary>Configuration hierarchy and environment variables</summary>

<details>
<summary>Code: environment variable examples</summary>

```bash
# Top-level settings
ORCHESTRATOR_MOCK_MODE=0
ORCHESTRATOR_DEBUG=1

# Nested LLM settings
ORCHESTRATOR_LLM__OUTPUT_CAP=4096
ORCHESTRATOR_LLM__BATCH_PARALLELISM=8
ORCHESTRATOR_LLM__CALL_TIMEOUT=300

# Nested escalation settings
ORCHESTRATOR_ESCALATION__MAX_RETRIES=3
ORCHESTRATOR_ESCALATION__MAX_ESCALATIONS=2

# Nested REPL settings
ORCHESTRATOR_REPL__MAX_OUTPUT_LEN=10000
ORCHESTRATOR_REPL__TIMEOUT_SECONDS=30
```

</details>

### Configuration Hierarchy

```
OrchestratorConfig
├── mock_mode: bool = True
├── debug: bool = False
├── llm: LLMConfig
│   ├── output_cap: int = 8192 (spill-to-file threshold)
│   ├── batch_parallelism: int = 4
│   ├── call_timeout: int = 120
│   ├── max_recursion_depth: int = 5
│   ├── default_prompt_rate: float = 0.50
│   └── default_completion_rate: float = 1.50
├── escalation: EscalationConfig
│   ├── max_retries: int = 2
│   ├── max_escalations: int = 2
│   └── optional_gates: frozenset = {"typecheck", "integration", "shellcheck"}
├── repl: REPLConfig
│   ├── max_output_len: int = 10000
│   ├── timeout_seconds: int = 30
│   ├── forbidden_modules: frozenset = {"os", "sys", "subprocess", ...}
│   └── forbidden_builtins: frozenset = {"eval", "exec", "open", ...}
├── server: ServerConfig
│   ├── default_url: str = "http://localhost:8080"
│   ├── timeout: int = 300
│   ├── num_slots: int = 4
│   ├── connect_timeout: int = 5
│   ├── retry_count: int = 3
│   └── retry_backoff: float = 0.5
├── monitor: MonitorConfig
│   ├── entropy_threshold: float = 2.5
│   ├── repetition_window: int = 50
│   ├── repetition_threshold: float = 0.3
│   └── min_tokens_before_abort: int = 20
├── paths: PathsConfig
│   ├── models_dir: Path = /mnt/raid0/llm/models
│   ├── cache_dir: Path = /mnt/raid0/llm/cache
│   ├── tmp_dir: Path = /mnt/raid0/llm/tmp
│   └── registry_path: Path = .../model_registry.yaml
└── features: FeaturesConfig
    ├── memrl: bool = False
    ├── tools: bool = False
    └── ... (10 feature flags)
```

<details>
<summary>Code: loading configuration at runtime</summary>

```python
from src.config import get_config

# Load from environment (cached)
config = get_config()

# Access nested settings
max_output = config.llm.output_cap
max_retries = config.escalation.max_retries
repl_timeout = config.repl.timeout_seconds

# Reset cache if env changes
from src.config import reset_config
reset_config()
```

</details>
</details>

## Environment Variables

Every cache, temp file, and data directory is redirected to the RAID array via environment variables. This is the project's most important safety measure — the 120GB OS drive has crashed before when large files landed on root. The path checks are enforced by hooks, but the variables should be set in every session regardless.

<details>
<summary>Required environment variables</summary>

<details>
<summary>Code: cache and path redirections</summary>

```bash
# HuggingFace/Transformers caches
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/raid0/llm/cache/huggingface
export HF_DATASETS_CACHE=/mnt/raid0/llm/cache/huggingface/datasets

# Python package cache
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip

# System temporary files
export TMPDIR=/mnt/raid0/llm/tmp

# XDG Base Directory Specification
export XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache
export XDG_DATA_HOME=/mnt/raid0/llm/claude/share
export XDG_STATE_HOME=/mnt/raid0/llm/claude/state
```

</details>

### Path Verification (Mandatory)

Before any file write operation:

```bash
# Verify path starts with /mnt/raid0/
[[ "$TARGET_PATH" == /mnt/raid0/* ]] || { echo "ERROR: Path not on RAID!"; exit 1; }
```

**Forbidden paths**: `/home/`, `/tmp/` (except via bind mount), `/var/`, `~/.cache/`, any path not starting with `/mnt/raid0/`.

</details>

## OMP & NUMA Runtime Tuning

Every inference command needs the same three-setting prefix: single OMP thread (llama.cpp manages its own parallelism), NUMA interleaving across all 12 memory channels, and 96 physical cores. Getting any of these wrong can halve throughput or worse.

<details>
<summary>Thread and NUMA configuration</summary>

<details>
<summary>Code: standard inference prefix</summary>

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m model.gguf -t 96 -p "prompt"
```

</details>

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `OMP_NUM_THREADS` | 1 | Disable OpenMP parallelism (llama.cpp handles threading) |
| `-t` | 96 | Use all physical cores (96-core EPYC 9655) |
| `--interleave=all` | Required | Interleave memory across 12 DDR5 channels (~460 GB/s) |

### NUMA Architecture

The EPYC 9655 has **12 memory channels** (1.13TB DDR5-5600 ECC). Using `--interleave=all` ensures memory bandwidth is maximized across all channels rather than binding to a single NUMA node.

**Do NOT use** `numactl --cpunodebind=0` or similar node-specific bindings for large models - this restricts memory bandwidth to 2-4 channels and cuts throughput by 60-75%.

</details>

## Python Environment

The project runs in a dedicated virtual environment called `pace-env`, managed with **uv** for fast installs. Everything from FastAPI to sentence-transformers lives here.

<details>
<summary>Environment setup and packages</summary>

<details>
<summary>Code: activation and installation</summary>

```bash
# Environment name
pace-env

# Activation
source /mnt/raid0/llm/pace-env/bin/activate

# Key packages
# - FastAPI + uvicorn (API server)
# - pydantic + pydantic-settings (config/validation)
# - httpx (async HTTP client)
# - numpy, sentence-transformers (MemRL)
# - RestrictedPython (REPL sandbox)
```

```bash
# Create environment with uv
uv venv /mnt/raid0/llm/pace-env

# Install dependencies
uv pip install -r /mnt/raid0/llm/claude/requirements.txt
```

</details>
</details>

## Session Initialization

Every session starts with an initialization script that verifies the environment, discovers available models, and checks branch safety. Skipping this step risks running benchmarks on the wrong llama.cpp branch or missing models that have been added since the last session.

<details>
<summary>Initialization steps and verification</summary>

<details>
<summary>Code: session startup commands</summary>

```bash
# Set environment variables
source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
agent_session_start "Session purpose"

# Discover models and verify llama.cpp branch
bash /mnt/raid0/llm/claude/scripts/session/session_init.sh
```

</details>

The initialization script:
1. Checks llama.cpp is on `production-consolidated` branch
2. Scans `/mnt/raid0/llm/models/` for GGUF files
3. Validates model registry against discovered models
4. Checks free memory (100GB minimum for tests)
5. Verifies environment variables point to `/mnt/raid0/`

</details>

<details>
<summary>References</summary>

- `src/features.py` - Feature flag system implementation
- `src/config.py` - Hierarchical configuration with pydantic-settings
- `scripts/session/session_init.sh` - Environment initialization
- `scripts/utils/agent_log.sh` - Session logging utilities

</details>

---

*Previous: [Chapter 01: Hardware System](01-hardware-system.md)* | *Next: [Chapter 03: llama.cpp Toolchain](03-llama-cpp-toolchain.md)*
