# AMD EPYC 9655 "Turin" Inference Optimization Project

## â›”â›”â›” ABSOLUTE RULE: NO ROOT FILESYSTEM WRITES â›”â›”â›”

**ALL LLM-related files MUST reside on `/mnt/raid0/` â€” NEVER on root (`/`).**

**THIS IS NON-NEGOTIABLE.** The root filesystem is a 120GB SSD. Writing large files there causes:
- System instability and crashes
- Paging storms that freeze the machine
- Disk exhaustion that corrupts the OS

### Path Verification (MANDATORY before any file operation)

```bash
# The path MUST start with /mnt/raid0/
[[ "$TARGET_PATH" == /mnt/raid0/* ]] || { echo "ERROR: Path not on RAID!"; exit 1; }
```

### Allowed vs Forbidden Paths

| âœ… ALLOWED (RAID Array) | âŒ FORBIDDEN (Root FS) |
|-------------------------|------------------------|
| `/mnt/raid0/llm/` | `/home/` (except symlinks) |
| `/mnt/raid0/llm/claude/` | `/tmp/` (except via bind mount) |
| `/mnt/raid0/llm/claude/logs/` | `/var/` |
| `/mnt/raid0/llm/cache/` | `~/.cache/` |
| `/mnt/raid0/llm/models/` | `~/.local/` |
| `/mnt/raid0/llm/tmp/` | Any path not starting with `/mnt/raid0/` |

### /tmp/claude Bind Mount

The `/tmp/claude` directory is bind-mounted to `/mnt/raid0/llm/tmp/claude`:
```bash
# Verify the bind mount is active
mount | grep /tmp/claude
# Should show: /dev/md127 on /tmp/claude type ext4 ...

# If missing, recreate:
sudo mkdir -p /tmp/claude
sudo mount --bind /mnt/raid0/llm/tmp/claude /tmp/claude
```

### Environment Variables (MUST be set in every session)

```bash
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/raid0/llm/cache/huggingface
export HF_DATASETS_CACHE=/mnt/raid0/llm/cache/huggingface/datasets
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip
export TMPDIR=/mnt/raid0/llm/tmp
export XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache
export XDG_DATA_HOME=/mnt/raid0/llm/claude/share
export XDG_STATE_HOME=/mnt/raid0/llm/claude/state
```

---

## System Identity

- **Host**: Beelzebub
- **User**: daniele
- **Working Directory**: `/mnt/raid0/llm/`
- **Python Environment**: `pace-env`

---

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" â€” 96 cores, 192 threads (Zen 5) |
| RAM | 1.13 TB DDR5-5600 ECC, 12 channels (~460 GB/s) |
| Storage | 2Ã— Solidigm P44 Pro 2TB NVMe RAID0 (models), 120GB SSD (OS) |
| Architecture | Zen 5 with true 512-bit AVX-512 (not double-pumped) |

---

## Current Research Status (December 2025)

### âœ… PRODUCTION (Working Now)

| Track | Method | Speedup | Command |
|-------|--------|---------|---------|
| **Track 1** | External Draft Model | **5.9x** | `llama-speculative -m TARGET -md DRAFT --draft-max 16` |
| **Track 2** | MoE Soft Mask | **21-48%** | `--override-kv ARCH.expert_used_count=int:4` |

### ðŸ†• IMPLEMENT THIS WEEK (Compounds with Production)

| Track | Method | Expected Gain | Effort | Resources |
|-------|--------|---------------|--------|-----------|
| **Track 8** | Prompt Lookup | +50-100% (grounded) | 1 hour | [GitHub](https://github.com/apoorvumang/prompt-lookup-decoding) |
| **Track 6** | SuffixDecoding | +100-200% (agentic) | 1 day | [Paper](https://suffix-decoding.github.io/) |

### ðŸ”„ IMPLEMENT IF NEEDED (Alternatives)

| Track | Method | When to Use | Resources |
|-------|--------|-------------|-----------|
| **Track 7** | CAS-Spec | No compatible draft available | [arXiv](https://arxiv.org/abs/2510.26843) |
| **Track 9** | CLaSp/SWIFT | Quick fallback | [arXiv](https://arxiv.org/abs/2505.24196) |

### â›” DEPRECATED

| Track | Method | Reason |
|-------|--------|--------|
| **Track 3** | EAGLE-1 | 0% acceptance after 20+ hours debugging |
| **Track 4** | Medusa | Training required |
| **Track 5** | SSM Speculation | Architecture incompatible |

---

## Combined Optimization Stack

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    DRAFT TOKEN SOURCE                            â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  1. Prompt Lookup (Track 8) â† TRY FIRST (zero cost)             â”‚
â”‚  2. SuffixDecoding (Track 6) â† TRY SECOND (agentic patterns)    â”‚
â”‚  3. External Draft Model (Track 1) â† FALLBACK (5.9x proven)     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              +
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚              TARGET MODEL OPTIMIZATION (Orthogonal)              â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Track 2: MoE Soft Mask (+21-48% on MoE models)                 â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Expected: 8-15x on grounded/agentic, 6-7x general
```

---

## SESSION STARTUP (MANDATORY)

### 1. Initialize Logging
```bash
source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
agent_session_start "Session purpose description"
```

### 2. Discover Models
```bash
bash /mnt/raid0/llm/claude/scripts/session/session_init.sh
```

### 3. Check Untested Models
```bash
cat /mnt/raid0/llm/claude/logs/untested_models.txt
```

### 4. Load Research Context
```bash
head -100 /mnt/raid0/llm/claude/logs/research_report.md
```

---

## Directory Structure

```
/mnt/raid0/llm/
├── llama.cpp/                    # Main inference engine
│   └── build/                    # CMake build directory
├── hf/                           # HuggingFace format models
├── models/                       # GGUF converted models
├── lmstudio/                     # LM Studio models
├── LOGS -> claude/logs/          # Symlink for backwards compatibility
├── cache/                        # HF/pip caches
├── tmp/                          # Temporary files (TMPDIR)
│   └── claude/                   # Bind-mounted to /tmp/claude
└── claude/                       # Project documentation & scripts
    ├── CLAUDE.md                 # This file
    ├── README.md                 # Project overview & quick start
    ├── OPENING_PROMPT.md         # Opening prompt template
    ├── logs/                     # Benchmark and runtime logs
    │   ├── research_report.md    # Main results document
    │   ├── agent_audit.log       # Agent action log
    │   ├── benchmarks/           # Benchmark CSV results
    │   └── model_inventory.json
    ├── docs/                     # Consolidated documentation
    │   ├── model-routing.md      # Model routing strategy
    │   └── research-writer.md    # Research writer agent guide
    ├── agents/                   # Specialized agent definitions
    ├── research/                 # Research documents
    │   └── speculative_decoding_research.md
    ├── cache/                    # XDG_CACHE_HOME
    ├── share/                    # XDG_DATA_HOME
    ├── state/                    # XDG_STATE_HOME
    ├── backups/                  # Reorganization backups
    └── scripts/                  # All scripts organized here
        ├── benchmark/            # bench_zen5.sh, run_inference.sh, record_test.sh
        ├── session/              # session_init.sh, claude_safe_start.sh, health_check.sh
        ├── system/               # system_audit.sh, reorganize_project.sh
        ├── utils/                # agent_log.sh, agent_log_analyze.sh
        └── legacy/               # Archived scripts from root
```

---

## Scripts Location

Scripts are organized in `/mnt/raid0/llm/claude/scripts/`:

| Category | Path | Contents |
|----------|------|----------|
| Benchmark | `scripts/benchmark/` | `bench_zen5.sh`, `run_inference.sh`, `record_test.sh` |
| Session | `scripts/session/` | `session_init.sh`, `claude_safe_start.sh`, `health_check.sh`, `monitor_storage.sh`, `emergency_cleanup.sh` |
| System | `scripts/system/` | `system_audit.sh`, `reorganize_project.sh` |
| Utils | `scripts/utils/` | `agent_log.sh`, `agent_log_analyze.sh` |
| Legacy | `scripts/legacy/` | Archived scripts (build_llama.sh, llm_cleanup.sh, etc.) |

---

## Quick Reference Commands

### Track 1: External Draft (5.9x proven)
```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  --draft-max 16 -t 96 -p "prompt"
```

### Track 2: MoE Soft Mask (21-48% proven)
```bash
./llama-cli -m Qwen3-VL-30B-A3B-Q4_K_M.gguf \
  --override-kv qwen3vlmoe.expert_used_count=int:4 \
  -t 96 -p "prompt"
```

### Baseline Benchmark
```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  ./llama-bench -m MODEL.gguf -t 96 -p 512 -n 128
```

---

## Key Papers & Resources

### NeurIPS 2025 (Priority Reading)
| Paper | Use Case | Link |
|-------|----------|------|
| SuffixDecoding | Agentic 10x | https://suffix-decoding.github.io/ |
| CAS-Spec | Self-draft 2.3x | https://arxiv.org/abs/2510.26843 |

### Implementation Resources
| Resource | Link |
|----------|------|
| Prompt Lookup | https://github.com/apoorvumang/prompt-lookup-decoding |
| vLLM Spec Decode | https://blog.vllm.ai/2024/10/17/spec-decode.html |
| Spec Decode Papers | https://github.com/hemingkx/SpeculativeDecodingPapers |
| CLaSp/SWIFT | https://arxiv.org/abs/2505.24196 |
| Kangaroo | https://github.com/Equationliu/Kangaroo |

### Deprecated (Reference Only)
| Paper | Reason |
|-------|--------|
| EAGLE (https://github.com/SafeAILab/EAGLE) | 0% acceptance, blocked |

---

## Model Routing Strategy

This project uses **tiered model allocation** based on task complexity, latency sensitivity, and reasoning depth. Claude Code should automatically route prompts according to these rules.

### Tier 1: OPUS 4.5 (Deep Research & Complex Coding)
**When to use:** Heavy lifting requiring extended reasoning, architectural decisions, debugging complex issues

**Route to Opus 4.5 when prompt contains:**
- Deep research requests: "analyze", "investigate", "research", "design", "architecture", "strategy", "novel approach"
- Complex debugging: "debug", "trace", "error analysis", "root cause", "investigation"
- High-level planning: "plan", "strategy", "roadmap", "framework", "design", "proposal"
- High-stakes decisions: "critical", "security", "safety", "risk assessment", "trade-off analysis"
- Coding tasks: "implement", "write", "code", "refactor", "modify", "integrate", "develop" (only if complex/novel)

**Example tasks:**
- "Design a new speculative decoding approach for SSM models"
- "Debug the EAGLE-1 acceptance rate bottleneck and propose fixes"
- "Analyze architectural compatibility between draft and target models"
- "Implement Track 7 CAS-Spec with layer skipping and INT8 quantization"

---

### Tier 2: SONNET 4.5 (Default, Parallel Research, Web Search)
**When to use:** General queries, parallel questions, web/repo research, synthesis of information for Opus

**Route to Sonnet 4.5 when prompt contains:**
- General questions: "how", "what", "explain", "describe", "summarize" (non-deep research)
- Parallel queries: Multiple questions in one prompt, "also", "meanwhile", "in parallel"
- Web/repo search: "find", "search", "look for", "fetch", "check", "browse"
- Information synthesis: "compile", "gather", "compare", "contrast", "list", "overview"
- Follow-up clarifications: Short questions needing quick answers before deep work
- Shallow analysis: Quick comparisons, metrics lookups, status checks

**Example tasks:**
- "Find the latest NeurIPS 2025 papers on speculative decoding"
- "Search GitHub for llama.cpp speculative decoding implementations"
- "Compare acceptance rates across K=8,16,24 and summarize findings"
- "What are the current bottlenecks in Track 1 deployment?"
- "Fetch the vLLM documentation on spec-decode and summarize integration steps"

**Handoff to Opus:** After gathering info, say "→ Opus: elaborate on [topic]" for deep analysis

---

### Tier 3: HAIKU 4.5 (Repetitive Benchmarking & Output Collection)
**When to use:** Known-working commands, repetitive measurements, result aggregation, log analysis

**Route to Haiku 4.5 when prompt contains:**
- Repetitive benchmarking: "benchmark", "run test", "measure", "test with", "compare K values"
- Output collection: "read logs", "extract results", "collect metrics", "parse output"
- Data aggregation: "summarize table", "compile results", "aggregate", "tabulate"
- Log analysis: "analyze logs", "check status", "review results", "what happened"
- Simple status checks: "is [model] ready", "do we have [file]", "check if [condition]"
- Known-working inference: Running pre-validated commands with different parameters

**Example tasks:**
- "Run bench_zen5.sh on Qwen2.5-Coder-32B and collect metrics"
- "Execute: `llama-speculative -m TARGET.gguf -md DRAFT.gguf --draft 16 -t 96`"
- "Parse the last 50 lines of research_report.md and extract speedup numbers"
- "Benchmark K=8,16,24 on the code generation prompt and create comparison table"
- "Check if /mnt/raid0/llm/models/DeepSeek-R1-32B-Q4_K_M.gguf exists"

**Pre-requisite:** Command must be known to work (tested by Opus first)

---

### Decision Tree for Model Routing

```
User Prompt:
    ↓
[Is this a novel design/complex debugging/high-stakes decision?]
    ├─ YES → OPUS 4.5
    └─ NO ↓
       [Is this information gathering / web search / parallel questions?]
           ├─ YES → SONNET 4.5
           └─ NO ↓
              [Is this a repetitive benchmark / output collection / log parsing?]
                  ├─ YES → HAIKU 4.5
                  └─ NO ↓
                     [Is this a follow-up to Opus work?]
                         ├─ YES → SONNET 4.5
                         └─ NO → DEFAULT: SONNET 4.5
```

---

### Model Tier Comparison

| Dimension | OPUS 4.5 | SONNET 4.5 | HAIKU 4.5 |
|-----------|----------|-----------|-----------|
| **Reasoning Depth** | ⭐⭐⭐⭐⭐ Expert | ⭐⭐⭐⭐ Strong | ⭐⭐ Sufficient |
| **Speed** | Slow | Medium | ⭐⭐⭐ Fast |
| **Cost** | Highest | Medium | ⭐⭐⭐ Lowest |
| **Best For** | Novel problems, debugging | Research & synthesis | Repetition |
| **When to Use** | Complex only | Default + research | Routine execution |

---

### Workflow Examples

**Pattern 1: Research → Design → Execution**
```
User: "Find latest speculative decoding papers"
→ SONNET: Web search, compile list

User: "Design how to implement these for CPU"
→ OPUS: Deep architectural analysis

User: "Run the benchmark"
→ HAIKU: Execute known commands
```

**Pattern 2: Parallel Questions → Deep Dive**
```
User: "What's speedup? Compatible models? Next steps?"
→ SONNET: Answer all three (parallel)

User: "Why is compatibility so low?"
→ OPUS: Root cause debugging
```

**Pattern 3: Routine Benchmarking**
```
User: "Test K=8,16,24 and summarize"
→ HAIKU: Run 3 benchmarks, create table
```

---

## Specialized Agents

| Agent | Invoke | Expertise | Typical Model |
|-------|--------|-----------|---------------|
| System Administrator | `@sysadmin` | CPU governor, NUMA, hugepages | SONNET (routine) / OPUS (debugging) |
| Build Engineer | `@build-engineer` | CMake, compiler flags | SONNET (standard) / OPUS (novel) |
| Benchmark Analyst | `@benchmark-analyst` | Results interpretation | HAIKU (execute) / SONNET (analyze) |
| Safety Reviewer | `@safety-reviewer` | Risk assessment | OPUS (high-stakes) |
| Model Engineer | `@model-engineer` | GGUF conversion, quantization | HAIKU (routine) / OPUS (complex) |
| Research Engineer | `@research-engineer` | C++ modifications, Track 7 | OPUS (implementation) / SONNET (research) |

Agent definitions: `/mnt/raid0/llm/claude/agents/`

**Rule:** Invoke agents as needed, but let the model tier (Opus/Sonnet/Haiku) be determined by the task type in the routing strategy above.

---

## MANDATORY: Append-Only Agent Logging

### Log Location
- **Audit log**: `/mnt/raid0/llm/claude/logs/agent_audit.log`

### Using the Logging Library
```bash
source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
```

### Required Pattern
```bash
agent_task_start "Description" "Reasoning"
# ... do work ...
agent_task_end "Description" "success|failure"
```

### Log Analysis
```bash
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --summary
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --loops
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --errors
```

### Loop Prevention
1. Max 3 retries on failures
2. If stuck: log blocker, document attempts, STOP
3. Always verify success via exit codes

---

## Code Style

- Use `#!/bin/bash` with `set -euo pipefail`
- **ALWAYS log all actions**
- All files on `/mnt/raid0/`
- Prefix inference: `OMP_NUM_THREADS=1 numactl --interleave=all`

---

## ⚠️ Benchmarking Pitfalls

### Interactive Mode Hangs
**CRITICAL**: `llama-cli` can hang waiting for user input if not configured correctly.

**ALWAYS use these flags when benchmarking:**
```bash
llama-cli -m MODEL.gguf -f prompt.txt -n 128 \
    --no-display-prompt \  # Don't echo prompt back
    --simple-io \          # Disable interactive features
    --no-warmup \          # Skip warmup (for timing accuracy)
    --temp 0               # Deterministic output
```

**Never use:**
- `-i` or `--interactive` in automated scripts
- `-p "prompt"` without `--no-display-prompt` (can still trigger interactive mode on some models)
- Pipes without proper EOF handling

**If a benchmark hangs:**
1. Check for interactive mode prompts (expecting user input)
2. Verify timeout is set: `timeout 300 llama-cli ...`
3. Kill stuck processes: `pkill -f llama-cli`
