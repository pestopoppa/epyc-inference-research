# Command Quick Reference

Copy-paste ready commands for common operations.

## Inference Commands

### Track 1: External Draft (Speculative Decoding)

```bash
# Code generation (11x speedup)
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 -t 96 -p "Your prompt"
```

**Parameters**:
- `-m`: Target (large) model
- `-md`: Draft (small) model
- `--draft-max`: K value (tokens to draft per iteration)
- `-t 96`: Use all physical cores

### Track 2: MoE Expert Reduction

```bash
# MoE model (+21-52% speedup)
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen3-235B-A22B-Q4_K_M.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96 -p "Your prompt"
```

**Override keys by family**:
- Qwen3 MoE: `qwen3moe.expert_used_count`
- Qwen3-Next: `qwen3next.expert_used_count`
- GLM-4: `glm4.expert_used_count`

### Track 8: Prompt Lookup

```bash
# Summarization (12.7x speedup)
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  --lookup-ngram-min 3 \
  -t 96 -f prompt_with_source_material.txt
```

### Corpus Sidecar (Phase 2B, llama.cpp-experimental)

```bash
# Build with corpus sidecar support
cd /mnt/raid0/llm/llama.cpp-experimental
cmake -B build -DLLAMA_CORPUS_SIDECAR=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 96

# Server with corpus sidecar speculation
numactl --interleave=all ./build/bin/llama-server \
  -m /path/to/model.gguf \
  --spec-type corpus-sidecar \
  --corpus-path /mnt/raid0/llm/cache/corpus/v3_sharded \
  --corpus-refresh 64 --corpus-snippets 8 \
  -t 96 -c 16384 --port 9081
```

### SSM Model (Expert Reduction ONLY)

```bash
# ⛔ NO speculation with Qwen3-Next!
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Q4_K_M.gguf \
  --override-kv qwen3next.expert_used_count=int:3 \
  -t 96 -p "Your prompt"
```

## Session Management

### Session Startup

```bash
# 1. Set environment
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TMPDIR=/mnt/raid0/llm/tmp
export XDG_CACHE_HOME=/mnt/raid0/llm/epyc-orchestrator/cache

# 2. Initialize logging
source /mnt/raid0/llm/epyc-orchestrator/scripts/utils/agent_log.sh
agent_session_start "Session purpose"

# 3. Discover models
bash /mnt/raid0/llm/epyc-orchestrator/scripts/session/session_init.sh

# 4. Load context
head -100 /mnt/raid0/llm/epyc-orchestrator/logs/research_report.md
```

### After Work Completed

```bash
# Run all gates
cd /mnt/raid0/llm/epyc-orchestrator && make gates
```

## Benchmarking

### Seed Specialist Routing (3-way, debug TUI)

```bash
# Default profile is infra-stable (recommended)
python3 scripts/benchmark/seed_specialist_routing.py \
  --3way --suites simpleqa --sample-size 50 --seed 123 --debug
```

```bash
# Override profile explicitly (same as default)
python3 scripts/benchmark/seed_specialist_routing.py \
  --profile infra-stable \
  --3way --suites simpleqa --sample-size 50 --seed 123 --debug
```

```bash
# Baseline profile (for A/B comparison against infra-stable)
python3 scripts/benchmark/seed_specialist_routing.py \
  --profile baseline \
  --3way --suites simpleqa --sample-size 50 --seed 123 --debug
```

Notes:
- `infra-stable` defaults: `ORCHESTRATOR_DEFERRED_TOOL_RESULTS=1`, lock timeouts 45s, `ORCHESTRATOR_UVICORN_WORKERS=1`, cooldown 2.0s.
- `--timeout` and `--cooldown` override profile defaults when provided.

### Run Full Suite

```bash
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all
```

### Run Specific Suite

```bash
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite thinking
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite coder
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite instruction_precision
```

### Compare Results

```bash
# List all runs
./scripts/benchmark/compare_results.sh --list-runs

# Compare two runs
./scripts/benchmark/compare_results.sh --baseline RUN_ID --current RUN_ID
```

## Validation

### Validate TaskIR

```bash
python3 orchestration/validate_ir.py task orchestration/last_task_ir.json
```

### Run Gates

```bash
make gates    # Full gate check
make lint     # Lint only
make test     # Tests only
```

## Logging

### Start Session Logging

```bash
source /mnt/raid0/llm/epyc-orchestrator/scripts/utils/agent_log.sh
agent_session_start "Description"
```

### Log Task

```bash
agent_task_start "Task description" "Reasoning"
# ... do work ...
agent_task_end "Task description" "success"
```

### Analyze Logs

```bash
/mnt/raid0/llm/epyc-orchestrator/scripts/utils/agent_log_analyze.sh --summary
/mnt/raid0/llm/epyc-orchestrator/scripts/utils/agent_log_analyze.sh --loops
/mnt/raid0/llm/epyc-orchestrator/scripts/utils/agent_log_analyze.sh --errors
```

## Model Testing

### Test New Model Launch

```bash
# 1. Basic test
/mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m NEW_MODEL.gguf -p "Hello" -n 10

# 2. Check for quirks (interactive mode, output format)

# 3. Add to the canonical epyc-inference-research/orchestration/model_registry.yaml master
#    with quirks documented; the orchestrator model_registry.yaml is generated lean output.
```

### Find Override Key

```bash
# List model metadata
/mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m MODEL.gguf --verbose 2>&1 | grep expert
```

## Environment Variables

```bash
# Required for all sessions
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/raid0/llm/cache/huggingface
export HF_DATASETS_CACHE=/mnt/raid0/llm/cache/huggingface/datasets
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip
export TMPDIR=/mnt/raid0/llm/tmp
export XDG_CACHE_HOME=/mnt/raid0/llm/epyc-orchestrator/cache
export XDG_DATA_HOME=/mnt/raid0/llm/epyc-orchestrator/share
export XDG_STATE_HOME=/mnt/raid0/llm/epyc-orchestrator/state

# Critical for inference
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

---

*See [MODELS.md](../models/MODELS.md) for model configurations.*
*See [QUIRKS.md](../models/QUIRKS.md) for known issues.*
