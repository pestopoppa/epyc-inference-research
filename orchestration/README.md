# Orchestration

This folder contains the schemas, validators, configuration, and **self-management procedures** for the hierarchical local-agent orchestration system.

## Contents

| File | Purpose |
|------|---------|
| `task_ir.schema.json` | JSON Schema for TaskIR (emitted by Front Door) |
| `architecture_ir.schema.json` | JSON Schema for ArchitectureIR (emitted by Tier-B3 Architect) |
| `validate_ir.py` | Python validator for both schemas |
| `model_registry.yaml` | Deterministic model → role mapping with acceleration configs |
| `procedure.schema.json` | JSON Schema for procedure YAML validation |
| `procedure_registry.py` | Procedure loader, validator, and executor |
| `procedure_scheduler.py` | Background job scheduler with pause/resume |
| `procedures/` | YAML procedure definitions (11 procedures) |
| `procedures/state/` | Procedure execution state files |
| `checkpoints/` | Server config checkpoints for hot-swap |
| `patches/` | Patch queue for approval workflow |

## Quick Start

### 1. Validate a TaskIR file

```bash
# Validate a file
python orchestration/validate_ir.py task orchestration/last_task_ir.json

# Validate from stdin (e.g., Front Door output)
echo '{"task_id": "abc", ...}' | python orchestration/validate_ir.py task -
```

### 2. Validate an ArchitectureIR file

```bash
python orchestration/validate_ir.py arch architecture/architecture_ir.json
```

### 3. Run all gates

```bash
make gates
# or
just gates
```

## Workflow

1. **Front Door** receives user input and emits `TaskIR` JSON
2. Save output to `orchestration/last_task_ir.json` (gitignored)
3. **Dispatcher** reads `TaskIR` and routes to specialists/workers
4. Workers emit artifacts (code, docs, diffs)
5. Run `make gates` to validate
6. On failure: return gate report to producing agent (once)
7. On second failure: escalate one tier

## TaskIR Schema

Required fields:
- `task_id`: Unique identifier (UUID recommended)
- `task_type`: `chat | doc | code | ingest | manage`
- `priority`: `interactive | batch`
- `objective`: What success looks like
- `inputs`: Array of `{type, value}` objects
- `constraints`: Hard requirements
- `assumptions`: Decisions made when request was ambiguous
- `agents`: Which agent roles are needed
- `plan.steps`: Ordered steps with `{id, actor, action, outputs}`
- `gates`: Which verification gates to run
- `definition_of_done`: Human-readable success criteria
- `escalation`: When and how to escalate failures

See `task_ir.schema.json` for full specification.

## ArchitectureIR Schema

Emitted by Tier-B3 Architect for system design decisions. Includes:
- Goals and non-goals
- Global invariants
- Repository layout with ownership
- Module definitions with public APIs
- Contracts (OpenAPI, JSON Schema, etc.)
- Cross-cutting concerns (logging, errors, config, security)
- Acceptance criteria (tests, benchmarks)
- Architecture Decision Records (ADRs)

See `architecture_ir.schema.json` for full specification.

## Model Registry

`model_registry.yaml` maps agent roles to specific models:

```yaml
roles:
  frontdoor:
    model: Qwen3.5-35B-A3B-Q4_K_M
    acceleration:
      type: moe_expert_reduction
      experts: 6
      lookup: true
    numa_instances: 4
    numa_threads_per_instance: 48

  architect_general:
    model: Qwen3.5-122B-A10B-Q4_K_M
    acceleration:
      type: moe_spec_lookup
      experts: 8
      draft: Qwen3.5-0.8B-Q8_0
      k: 8
      lookup: true
    numa_instances: 1
    numa_threads_per_instance: 96

  coder_escalation:
    model: Qwen2.5-Coder-32B-Instruct
    acceleration:
      type: speculative_decoding
      draft: Qwen2.5-Coder-0.5B-Instruct
      k: 24
```

The dispatcher should read this file and **never improvise** model selection.

## Production Server Topology (Updated 2026-03-19)

Launch the production stack using the Python orchestrator:

```bash
# Full production stack (~701GB, 62% of 1130GB RAM)
python scripts/server/orchestrator_stack.py start

# HOT tier only (~701GB)
python scripts/server/orchestrator_stack.py start --hot-only

# Development mode (single 0.5B model, fast startup)
python scripts/server/orchestrator_stack.py start --dev

# Check status
python scripts/server/orchestrator_stack.py status

# Stop all
python scripts/server/orchestrator_stack.py stop --all
```

### HOT Tier (Always Resident, ~701GB) — NUMA-Optimized (2026-03-19)

| Port(s) | Role | Model | NUMA | Acceleration | Speed |
|---------|------|-------|------|--------------|-------|
| 8080,8180,8280,8380 | frontdoor (4x) | Qwen3.5-35B-A3B Q4_K_M | 4x48t quarters | MoE6+lookup, mlock | ~19.6 t/s/inst |
| 8081,8181,8281,8381 | coder_escalation (4x) | Qwen2.5-Coder-32B f16 + 0.5B draft | 4x48t quarters | spec K=24 + lookup | ~6.6 t/s/inst |
| 8082 | worker_general, worker_math | Qwen2.5-7B-Instruct-f16 + 0.5B draft | Q0A pinned | spec K=24 | 46 t/s |
| 8083 | architect_general | Qwen3.5-122B-A10B Q4_K_M + 0.8B draft | Node 0, 96t | MoE8+spec+lookup | 12.6 t/s |
| 8084 | architect_general | Qwen3-Coder-480B-A35B Q4_K_M | Node 0, 96t | tree dm=48 ps=0.05 | 3.82 t/s |
| 8085 | ingest_long_context | Qwen3-Next-80B-A3B Q4_K_M | Node 0, 96t | None (SSM), mlock | ~12 t/s |
| 8086 | worker_vision | Qwen2.5-VL-7B Q4_K_M + mmproj | Q0B pinned | None (VL) | ~15 t/s |
| 8087 | vision_escalation | Qwen3-VL-30B-A3B Q4_K_M + mmproj | Node 1, 96t | MoE4 | ~10 t/s |
| 8090-8095 | embedder (6x) | BGE-large-en-v1.5-F16 | unpinned | probe-first | — |

### WARM Tier (Burst Capacity, ~1GB)

| Port | Role | Model | Slots | Speed |
|------|------|-------|-------|-------|
| 8102 | worker_fast | Qwen2.5-Coder-1.5B-Q4_K_M | 4 | 60 t/s |

### Services

| Port | Service | Purpose |
|------|---------|---------|
| 8000 | orchestrator API (uvicorn) | FastAPI entrypoint |
| 9001 | document_formalizer (LightOnOCR-2-1B) | PDF OCR, figure extraction |

### CLI Tools (On-Demand, No Port)

| Tool | Model | Purpose |
|------|-------|---------|
| tool_formalizer | xLAM-2-1B-Q4_K_M | Tool call formalization |
| math_formalizer | MathSmith-8B-Q4_K_M | Math problem formalization |
| pdf_router | pdftotext + PyMuPDF | Fast-path PDF extraction |

### Escalation Chains

```
Code: coder_escalation (30B) → coder_escalation (32B) → architect_general (122B)
General: frontdoor (35B) → architect_general (122B)
Vision: worker_vision (7B, port 8086) → vision_escalation (30B, port 8087)
```

### Direct-Answer Mode (2026-01-29)

Short, simple prompts bypass the REPL Python-code wrapper entirely. The REPL adds ~900 tokens of overhead and forces the model to generate Python code + call `FINAL(answer)`, which destroys quality on instruction-precision tasks (same model: 11/11 without REPL, 2/11 through REPL).

**When direct mode activates:**
- Prompt has no file/tool operation keywords (`read the file`, `list files`, `grep for`, etc.)
- Context is < 20K characters (no long-context pipeline needed)

**What happens in direct mode:**
1. User prompt (+ optional context) sent directly to model via `primitives.llm_call()`
2. No REPL wrapper, no tool environment, no FINAL() requirement
3. Response returned as `ChatResponse` with `turns=1` (the MemRL answer review gate was removed by RI-18c)

**Implementation**: `_should_use_direct_mode()` in `src/api/routes/chat.py`

### Vision Pipeline (2026-01-29)

Vision requests are routed directly to VL servers using multimodal chat completions:

```
Image + Prompt
     │
     ▼
┌──────────────────────────────────────┐
│  worker_vision (port 8086)            │
│  Qwen2.5-VL-7B, ~15 t/s              │
│  /v1/chat/completions + image_url     │
└──────────────────────────────────────┘
     │ [failure]
     ▼
┌──────────────────────────────────────┐
│  vision_escalation (port 8087)        │
│  Qwen3-VL-30B-A3B, ~10 t/s           │
│  /v1/chat/completions + image_url     │
└──────────────────────────────────────┘
     │ [failure]
     ▼
  Legacy vision pipeline (last resort)
```

Images are sent as base64 data URIs in the multimodal payload format. MIME type auto-detected from header bytes (JPEG/PNG/WebP).

### Model Self-Routing (2026-01-29)

Models can now make informed routing decisions using MemRL intelligence. This supplements the Python control flow escalation with model-initiated routing.

**Available Routing Tools (REPL):**

| Tool | Purpose |
|------|---------|
| `my_role()` | Self-awareness: role, tier, capabilities, delegation targets |
| `route_advice(task)` | MemRL recommendation: Q-values, similar tasks, confidence |
| `delegate(prompt, role, reason)` | Tracked delegation with outcome logging |
| `escalate(reason, target_role)` | Request escalation (specific target or next-in-chain) |
| `recall(query)` | Episodic memory search with Q-values |

**How It Works:**

1. **Turn 0**: Routing context injected into prompt (MemRL Q-values for similar tasks)
2. **During execution**: Model calls `route_advice()` / `my_role()` to assess
3. **Model decision**: Calls `escalate()` or `delegate()` → sets artifacts
4. **After execute**: `chat.py` checks artifacts → honors routing request
5. **Learning**: Delegation outcomes logged to MemRL → Q-values updated

**Tier Guard:**
- Tier A (frontdoor): Can delegate to workers + coder_escalation
- Tier B (specialists): Can delegate to workers
- Tier C (workers): **Cannot delegate** — use deterministic tools only

**Artifact Protocol:**

```python
# Model calls escalate() → sets artifacts
repl.artifacts["_escalation_requested"] = True
repl.artifacts["_escalation_target"] = "coder_escalation"  # optional
repl.artifacts["_escalation_reason"] = "Task requires code generation"

# Model calls delegate() → records in artifacts
repl.artifacts["_delegations"] = [{
    "from_role": "frontdoor",
    "to_role": "worker_general",
    "reason": "File-level task",
    "success": True,
    "elapsed_sec": 2.3,
}]
```

### Two-Stage Context Pipeline (2026-01-29)

ALL long-context requests (>20K chars) now use a two-stage pipeline instead of REPL exploration:

```
Phase 1: Worker Parallel Digest           Phase 2: Frontdoor Synthesis
worker_general (7B, 44 t/s)               frontdoor (30B, 18 t/s)
┌───────────────────────────┐            ┌─────────────────────────────┐
│ Context → N chunks (~4K)  │            │ Worker digests + question   │
│ Each chunk → worker       │  ───────►  │ Synthesize final answer     │
│ Produces structured digest│   TOON     │ Report exact findings       │
│ Parallel execution        │  encoded   │ No REPL code generation     │
└───────────────────────────┘            └─────────────────────────────┘
```

### MemRL Quality Review Gate (2026-01-29, removed 2026-10-03)

Removed by RI-18c. RI-18's forced-review counterfactual (579 items) fired its
pre-registered DROP: the production gate (Q < 0.6) never fired, and reviewing every
answer netted −38.3 per 100 (12 fixed, 234 broken). See
`docs/chapters/07-memrl-system.md` → *MemRL Quality Review Gate*.

### Split Pipeline (Exploration → Summarization)

```
worker_general (7B, 46 t/s)          worker_summarize (32B, 95 t/s)
┌─────────────────────────┐         ┌─────────────────────────────┐
│ • Crawl directories     │         │ • Synthesize findings       │
│ • Grep for patterns     │ ──────► │ • Create executive summary  │
│ • Extract code snippets │  TOON   │ • Answer comprehension Qs   │
│ • Collect raw results   │ encoded │ • Document understanding    │
└─────────────────────────┘         └─────────────────────────────┘
```

### Role Aliases (2026-01-29)

Models may generate natural-language role names. These are resolved automatically:

| Model generates | Maps to |
|----------------|---------|
| `researcher_agent` | `worker_general` |
| `coder_agent` | `coder_escalation` |
| `reviewer_agent` | `architect_general` |
| `math_agent` | `worker_math` |
| `vision_agent` | `worker_vision` |
| `summarizer_agent` | `worker_summarize` |

### Benchmark CLI Metrics (2026-01-29)

Both benchmark scripts now print per-prompt and per-suite metrics inline:

- **`compare_orchestrator_direct.py`**: Each prompt line shows latency (ms), tokens/sec, speedup, quality, turns, routed role
- **`run_orchestrator_benchmark.py`**: Per-suite mini-summary after each suite (quality %, avg latency, avg t/s) + aggregate Phase 2 totals

Both scripts support `--restart-api` to restart the uvicorn API (port 8000) before running. This does NOT restart the llama-server backends (8080-8090).

### API Import Safety Tests (2026-01-29)

11 tests in `tests/unit/test_api_imports.py` guard against import/signature mismatches between the canonical `src.prompt_builders` and the deprecated `src.api.services.orchestrator` wrapper. Run after any changes to prompt builder function signatures:

```bash
pytest tests/unit/test_api_imports.py -v
```

## Files to Gitignore

Add to your `.gitignore`:

```
# Transient IR files
orchestration/last_task_ir.json
orchestration/last_architecture_ir.json

# Gate reports
orchestration/gate_report_*.json
```

## Tool Compliance Testing

Models must use REPL tools instead of Python imports. The sandbox blocks dangerous operations, so models that use `import os`, `pathlib`, etc. will fail.

### Run Compliance Tests

```bash
# Mock mode (no live models needed)
pytest tests/integration/test_model_tool_compliance.py -v

# Live model tests (requires orchestrator running)
pytest tests/integration/test_model_tool_compliance.py -v --run-live-models
```

### Benchmark Prompts

Tool compliance benchmark prompts are in `benchmarks/prompts/v1/tool_compliance.yaml`:
- **Tier 1**: Basic tool usage (list_dir, peek, grep)
- **Tier 2**: Combined tools and file metadata
- **Tier 3**: Document processing, LLM delegation, shell commands

### Tool Mapping (REPL → Forbidden Python)

| REPL Tool | Forbidden Alternative |
|-----------|----------------------|
| `list_dir(path)` | `os.listdir()`, `pathlib.Path().iterdir()` |
| `peek(n, file_path)` | `open().read()`, `Path().read_text()` |
| `grep(pattern)` | `re.findall()`, subprocess |
| `file_info(path)` | `os.stat()`, `Path().stat()` |
| `run_shell(cmd)` | `subprocess.run()`, `os.system()` |

See `docs/reference/models/QUIRKS.md` for model-specific compliance notes.

## Procedure Registry (Self-Management)

The Procedure Registry enables deterministic self-management operations with ~350 tokens per operation (vs 3000-5000 for manual execution).

### Available Procedures (11 total)

| Procedure | Category | Purpose |
|-----------|----------|---------|
| `benchmark_new_model` | benchmark | Run benchmark suite on new GGUF model |
| `check_draft_compatibility` | benchmark | Validate draft-target pairing for spec decode |
| `add_model_to_registry` | registry | Add new model entry with all fields |
| `update_registry_performance` | registry | Update t/s, speedup after benchmarks |
| `add_model_quirks` | registry | Document discovered model quirks |
| `deprecate_model` | registry | Mark deprecated (manual delete only) |
| `run_quality_gates` | codebase | Run full gate suite (lint, tests, etc.) |
| `create_handoff` | codebase | Generate handoff documents |
| `prepare_finetuning_dataset` | finetuning | Prepare/split datasets |
| `run_finetuning` | finetuning | Execute LoRA/QLoRA training |
| `evaluate_finetuned_model` | finetuning | Post-training evaluation |

### Using Procedures

```python
from orchestration.procedure_registry import ProcedureRegistry

# Initialize registry
registry = ProcedureRegistry()

# List all procedures
procedures = registry.list_procedures()

# List by category
benchmark_procs = registry.list_procedures(category="benchmark")

# Execute a procedure
result = registry.execute(
    "benchmark_new_model",
    model_path="/mnt/raid0/llm/models/NewModel.gguf",
    model_name="NewModel"
)
```

### REPL Tools for Procedures

| Tool | Purpose |
|------|---------|
| `run_procedure(id, **params)` | Execute procedure |
| `list_procedures(category=None)` | List available procedures |
| `get_procedure_status(id)` | Check execution status |
| `checkpoint_create(name)` | Save server configs |
| `checkpoint_restore(id)` | Restore from checkpoint |
| `prepare_patch(files, desc)` | Generate diff for approval |
| `list_patches(status)` | List pending/approved/rejected |
| `apply_approved_patch(name)` | Apply after owner approval |
| `reject_patch(name, reason)` | Reject with reason |

### Patch Approval Workflow

All registry modifications generate patches for owner approval:

```
orchestration/patches/
├── pending/     # Awaiting approval
├── approved/    # Applied patches (audit trail)
└── rejected/    # Rejected with reason
```

### Running Tests

```bash
# Procedure registry tests (25 tests)
python -m pytest tests/unit/test_procedure_registry.py -v
```

---

## REPL Memory (Episodic Learning)

The `repl_memory/` directory contains the MemRL episodic memory system for learning from REPL tool usage patterns.

### Files

| File | Purpose |
|------|---------|
| `repl_memory/seed_examples.json` | 56 canonical REPL tool usage examples |
| `repl_memory/seed_loader.py` | Script to load seeds into episodic memory |
| `repl_memory/episodic_store.py` | SQLite + FAISS/numpy memory storage |
| `repl_memory/faiss_store.py` | FAISS embedding store + NumPy fallback |
| `repl_memory/embedder.py` | Task embedding via BGE-large (1024-dim) |
| `repl_memory/retriever.py` | Two-phase retrieval + hybrid router |
| `repl_memory/q_scorer.py` | Async Q-value update agent |
| `repl_memory/progress_logger.py` | Structured JSONL logging |
| `repl_memory/staged_scorer.py` | PARL-inspired staged reward shaping (λ annealing + exploration bonus) |

### Embedding Backend

**FAISS (default)**: O(log n) search using `IndexFlatIP` with L2 normalization
- Storage: `embeddings.faiss` + `id_map.npy`
- Performance: ~2ms for 500K entries

**NumPy (fallback)**: O(n) brute-force search for migration/rollback
- Storage: `embeddings.npy` (memory-mapped)
- Performance: ~70ms for 500K entries

```python
# FAISS backend (default)
store = EpisodicStore(db_path="/path/to/data", use_faiss=True)

# NumPy backend (fallback)
store = EpisodicStore(db_path="/path/to/data", use_faiss=False)
```

### Performance Optimizations (2026-01-27)

The episodic memory system has been optimized for low-latency operation:

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Embedding generation | 50-200ms | 2-5ms | **40x faster** |
| Storage (FAISS persist) | 10-50ms blocking | ~0ms (async) | Non-blocking |
| Graph penalty lookup | 5-20ms | <1ms | TTL cache |
| id_map lookup | O(n) | O(1) | Dict-based |

**Embedding Server (HOT Tier)**

The embedding server runs on port 8090 as part of the HOT tier:

```bash
# Start orchestrator stack (includes embedding server)
python scripts/server/orchestrator_stack.py start --hot-only
```

Fallback chain: HTTP server (2-5ms) → subprocess (50-200ms) → hash fallback (<1ms)

**Write-Behind Persistence**

FAISS index persists asynchronously every 10 seconds:

```python
# Standard use (non-blocking, batched writes)
store = EpisodicStore(flush_interval=10.0)
store.store(embedding, action, action_type, context)

# ACID-critical use (synchronous flush)
store.store_immediate(embedding, action, action_type, context)
```

**Graph Penalty Caching**

Graph lookups for failure penalties and hypothesis confidence use TTL caching:

```python
from orchestration.repl_memory import GraphEnhancedRetriever

retriever = GraphEnhancedRetriever(
    cache_ttl=60,       # 60 second TTL
    cache_maxsize=500   # Max 500 cached entries
)
```

### Seeding Episodic Memory

```bash
# First-time setup (load 56 canonical examples)
python orchestration/repl_memory/seed_loader.py

# Force reload (clear + reload)
python orchestration/repl_memory/seed_loader.py --force
```

### Database Maintenance

The episodic store can accumulate entries from validation script runs. To check for contamination:

```bash
# Check entry counts by date
python3 -c "
import sqlite3
db = sqlite3.connect('orchestration/repl_memory/sessions/episodic.db')
for row in db.execute('SELECT DATE(created_at) as d, COUNT(*) FROM memories GROUP BY d ORDER BY d'):
    print(f'  {row[0]}: {row[1]}')
"

# Backups are at *.bak files (episodic.db.bak, embeddings.faiss.bak, id_map.npy.bak)
```

**2026-01-31 cleanup**: Surgically removed 6,506 validation-run entries (Jan 30-31) while preserving 2,714 original seed entries (Jan 28). FAISS index rebuilt from 9,181 → 2,714 embeddings (-70% file size).

### Category Coverage

| Category | Count | Tools |
|----------|-------|-------|
| filesystem | 8 | `list_dir`, `file_info`, `peek` |
| procedure | 8 | `run_procedure`, `list_procedures` |
| document | 6 | `ocr_document`, `extract_figure` |
| complex | 7 | Multi-step with `llm_call` |
| shell | 5 | git, ls, find |
| simple | 4 | Direct calculations |
| search | 3 | `grep` patterns |
| vision | 3 | `analyze_figure` |
| web | 3 | `web_fetch` |
| artifacts | 3 | Store/retrieve values |
| memory | 2 | `recall` past tasks |
| escalation | 2 | `escalate` to architect |
| parallel | 2 | `llm_batch` operations |

### Checking Memory Stats

```bash
python3 -c "from orchestration.repl_memory import EpisodicStore; print(EpisodicStore().get_stats())"
```

---

## Document Preprocessing Services

### PDF Router

The PDF Router (`src/services/pdf_router.py`) intelligently routes PDF processing between fast text extraction and OCR:

```
PDF Input
    ↓
[pdftotext probe] → Quick text extraction (~100ms)
    ↓
[Quality check] → Entropy, garbage ratio, word length
    │
    ├─ HIGH quality (born-digital) → pdftotext + PyMuPDF figures
    └─ LOW quality (scanned) → LightOnOCR (OCR fallback)
```

**Usage:**

```python
from src.services.pdf_router import extract_pdf

result = extract_pdf("/path/to/document.pdf", extract_figures=True)
print(f"Method: {result.method}")  # "pdftotext" or "lightonocr"
print(f"Text: {len(result.text)} chars")
print(f"Figures: {len(result.figures)} with bounding boxes")
print(f"Quality: {result.quality_score:.2f}")
```

**Quality Assessment Thresholds:**
- `MIN_ENTROPY = 3.5` - Shannon entropy for readable text
- `MAX_GARBAGE_RATIO = 0.15` - Non-printable character ratio
- `MIN_WORD_LENGTH_AVG = 2.5` - Average word length

### Prompt Compression (DISABLED)

LLMLingua-2 extractive compression was tested but **disabled due to quality regression**.

**Findings (2026-01-27):**
- Extractive compression produces choppy, fragmentary text
- Downstream LLMs hallucinate to fill semantic gaps
- 140s vs 74s (slower, not faster)
- Prompt leakage, fake citations, typos in output

**Recommendation:** Wait for Cmprsr (abstractive compression) weights.

See `handoffs/active/cmprsr_prompt_compression.md` for details.

### TOON Format Encoding (Opt-in)

TOON (Token-Oriented Object Notation) provides 55% token reduction for structured tool outputs.

**Findings (2026-01-27):**

| Use Case | Token Reduction | Status |
|----------|----------------|--------|
| File listings | **64.6%** | ADOPT |
| OCR sections | **55.3%** | ADOPT |
| Escalation context | **42.3%** | ADOPT |
| Grep hits | -18.6% (worse) | REJECT |

**Key insight:** TOON excels for uniform arrays (file listings) but fails for grouped data (grep hits) where Markdown is more compact.

**Usage:**

```python
from src.repl_environment import REPLEnvironment, REPLConfig

config = REPLConfig(use_toon_encoding=True)  # Opt-in
repl = REPLEnvironment(context="...", config=config)
# _list_dir() now returns TOON for 3+ file directories
```

**Install:**

```bash
uv pip install "hierarchical-orchestrator[toon]"
```

See `research/TOON_EVALUATION.md` for full evaluation report.

---

## Dependencies

All dependencies are declared in `pyproject.toml`. Install with:

```bash
pip install -e .           # Core dependencies
pip install -e ".[dev]"    # + testing/linting tools
pip install -e ".[datasets]"  # + HuggingFace datasets for benchmark sampling
```

See `pyproject.toml` for optional dependency groups: `toon`, `graph`, `sandbox`, `tuning`, `ui`.
