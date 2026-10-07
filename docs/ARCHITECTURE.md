# Orchestration System Architecture

**Version**: 2.7 (Seeding Slot Progress Telemetry)
**Last Updated**: 2026-02-09

> **This is the living technical reference** — updated continuously as the system evolves.
> For narrative explanations of each subsystem, see the [Chapter Index](chapters/INDEX.md).

### Recent Updates (2026-02-09)

- **Live Slot Progress in Seeding**: 3-way forced calls now poll backend `/slots` during inference and emit `[slot-progress]` logs (task id + decoded/remain counters) for near-real-time visibility.
- **INFRA Token Estimate**: when API returns `0 tok` on timeout/disconnect, seeding now stores and logs `tokens_generated_estimate` from slot counters (`0 tok, est N tok`) to distinguish true no-generation from accounting loss.
- **Reward Telemetry Extension**: reward context now includes `tokens_generated_estimate`, `tokens_generated_effective`, `backend_task_id`, and `slot_progress_source`.
- **Architect Token Caps Reverted**: delegated architect path no longer forces `n_tokens` caps; generation limits remain controlled by runtime/server settings and seeding timeout policy.

### Recent Updates (2026-02-08)

- **Forced-role Integrity**: Eval/seed forced runs now preserve `force_role` invariants (no quality-escalation role hopping), while still allowing delegation when explicitly enabled.
- **Vision Forced-route Fix**: Forced VL roles are preserved for image workflows; document workflows still use frontdoor synthesis where intended.
- **3-way Seeding Hardening**: Adaptive per-call timeouts + heavy-port precheck reduce long blocked waits on stalled heavy ports.
- **Telemetry Consistency**: `tools_used`, `tools_called`, `tool_timings` are normalized together in seeding caller output.
- **Delegation Runtime Fix**: `ExecutionResult` dataclass fields are now used correctly in delegation paths, avoiding `'ExecutionResult' object has no attribute 'get'` failures.

### Recent Updates (2026-02-05)

- **Research Context Tracker**: DAG-based tracking of REPL tool invocations with semantic cross-refs. See [Ch03](chapters/03-repl-environment.md).

### Recent Updates (2026-02-04)

- **Routing Facade**: Unified escalation decisions (rules-authoritative, learned advisory). FailureRouter deprecated.
- **Delegation Telemetry**: Added `delegation_events`, `tools_success`, `delegation_success` to ChatResponse.

## Table of Contents

1. [System Overview](#system-overview) — *see also [Ch02: Orchestration](chapters/02-orchestration-architecture.md)*
2. [Request Flow](#request-flow) — *see also [Ch04: Server Stack](chapters/04-production-server-stack.md)*
3. [Module Responsibilities](#module-responsibilities) — *see also [Ch03: REPL](chapters/03-repl-environment.md)*
4. [Escalation System](#escalation-system) — *see also [Ch10: Escalation](chapters/10-escalation-and-routing.md)*
5. [Feature Flag System](#feature-flag-system) — *see also [Ch01: Runtime](chapters/01-runtime-environment.md)*
6. [Security Model](#security-model) — *see also [Ch14: Security](chapters/14-security-and-monitoring.md)*
7. [Adding New Features](#adding-new-features) — *see also [Ch13: Tool Registry](chapters/13-tool-registry.md)*
8. [Common Pitfalls](#common-pitfalls)
9. [Testing Guidelines](#testing-guidelines) — *see also Benchmarking Framework (epyc-inference-research)*

---

## System Overview

The orchestration system implements a hierarchical local-agent workflow for production LLM inference. The core philosophy:

> *One model thinks. Many models work. Tools decide who is right.*

### Architecture Diagram

```
                    ┌──────────────────────────────────────┐
                    │           HTTP Request               │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │            api.py                    │
                    │   (FastAPI routes & request handling)│
                    └──────────────────┬───────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
┌─────────────────┐        ┌──────────────────┐        ┌─────────────────┐
│  llm_primitives │        │  routing_facade  │        │    executor     │
│  (LLM calls)    │        │ (rules+learned)  │        │  (task exec)    │
└────────┬────────┘        └───────────┬──────┘        └─────────────┬───┘
         │                             │                             │
         │                             ▼                             │
         │                ┌──────────────────┐                       │
         │                │   escalation.py  │◄──────────────────────┘
         │                │ (unified policy) │
         │                └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            Backend Layer                            │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐  │
│  │ LlamaServer │          │  Caching    │          │   Mock      │  │
│  │  Backend    │          │  Backend    │          │  Backend    │  │
│  │  (httpx)    │          │             │          │             │  │
│  └─────────────┘          └─────────────┘          └─────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Tiers

| Tier | Role | Purpose | Memory Residency |
|------|------|---------|------------------|
| **A** | Frontdoor + Coder Primary | Intent classification, routing, simple code | HOT (always resident) |
| **B** | Specialists | Coder escalation, ingest, architects, vision escalation | HOT (always resident) |
| **C** | Workers | Explore, summarize, vision, math, fast burst | HOT (explore, vision) / WARM (fast workers) |
| **D** | Embedder + Draft | Embedding server, speculative decoding draft (co-loaded) | HOT |

**Note**: As of 2026-01, all tiers except WARM fast workers (~1.5B) are always resident. The HOT tier uses ~535GB (47% of 1130GB RAM). See [Chapter 04](chapters/04-production-server-stack.md) for the full server topology.

---

## Request Flow

### 1. HTTP Request Handling (`api.py`)

```
Request → /chat endpoint
         ↓
    Parse ChatRequest (Pydantic)
         ↓
    Increment active_requests (thread-safe)
         ↓
    Generate task_id for tracking
         ↓
    Check mock_mode vs real_mode
         ↓
    Route to _handle_chat()
```

### 2. Chat Handler Loop

```python
# Simplified flow (src/api/routes/chat.py)
for turn in range(max_turns):
    # 1. Build prompt for Root LM (canonical source: src/prompt_builders.py)
    prompt = build_root_lm_prompt(state, user_prompt, last_output, turn,
                                  routing_context=routing_context)

    # 2. Run inference (mock or real)
    if real_mode:
        response = llm_primitives.generate(role="frontdoor", prompt=prompt)
    else:
        response = mock_response(turn)

    # 3. Check for FINAL signal (with output capture)
    if "FINAL(" in response:
        answer = _resolve_answer(result)  # detects stub args, uses captured stdout
        return answer

    # 4. Execute REPL code if present
    if contains_code_block(response):
        result = repl.execute(code)
        last_output = result.output

    # 5. Check model-initiated routing (escalate/delegate artifacts)
    if repl.artifacts.get("_escalation_requested"):
        next_role = repl.artifacts.get("_escalation_target") or escalation_chain[role]

    # 6. Check error-based escalation
    elif should_escalate(result):
        next_role = routing_facade.decide(context)
```

### 3. Escalation Flow

```
Failure detected
      ↓
Create EscalationContext
      ↓
escalation.decide(context)
      ↓
┌─────────────────────────────────────────┐
│  Decision Tree:                         │
│  1. Early abort? → ESCALATE immediately │
│  2. Format error? → RETRY only          │
│  3. Retries left? → RETRY               │
│  4. Can escalate? → ESCALATE            │
│  5. At top? → FAIL                      │
└─────────────────────────────────────────┘
      ↓
Return EscalationDecision
      ↓
Execute decision (retry/escalate/fail)
```

---

## Module Responsibilities

### Core Modules

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `src/api/` | HTTP endpoints (modular structure) | All |
| `src/llm_primitives.py` | LLM call abstraction | backends |
| `src/routing_facade.py` | Unified escalation decisions (rules + learned) | escalation, MemRL |
| `src/failure_router.py` | Deprecated — thin wrapper around routing_facade | roles |
| `src/executor.py` | Task execution engine | escalation |
| `src/repl_environment/` | Sandboxed Python REPL (modular) | - |
| `src/research_context.py` | REPL tool result DAG tracking | repl_memory (optional) |
| `src/repl_environment/code_search.py` | Multi-vector code/doc retrieval (NextPLAID) | next-plaid-client |
| `src/registry_loader.py` | Model registry YAML | - |

### API Module Structure

```
src/api/
├── __init__.py          # FastAPI app factory, lifespan management
├── state.py             # Thread-safe AppState, dependency providers
├── models/
│   ├── requests.py      # ChatRequest, GateRequest
│   ├── responses.py     # ChatResponse (+ routing telemetry), HealthResponse, GatesResponse
│   ├── openai.py        # OpenAI-compatible models
│   └── sessions.py      # Session management models
├── services/
│   ├── orchestrator.py  # Root LM logic (delegates to prompt_builders)
│   └── memrl.py         # MemRL lazy loading, Q-scoring, background tasks
└── routes/
    ├── health.py        # /health endpoint
    ├── chat.py          # /chat, /chat/stream endpoints
    ├── gates.py         # /gates endpoints
    ├── stats.py         # /stats endpoints
    ├── openai_compat.py # /v1/* OpenAI-compatible endpoints
    └── sessions.py      # /sessions/* endpoints
```

### ChatResponse Fields

The `ChatResponse` model (`src/api/models/responses.py`) includes routing telemetry:

| Field | Type | Description |
|-------|------|-------------|
| `answer` | str | Final answer (with output capture — stub FINAL args resolved) |
| `turns` | int | REPL execution turns |
| `tokens_used` | int | Total tokens consumed |
| `elapsed_seconds` | float | Wall-clock time |
| `routed_to` | str | Primary role that handled the request |
| `role_history` | list[str] | Full role chain if escalated |
| `routing_strategy` | str | How routing was decided: `learned` / `rules` / `default` |
| `tokens_generated` | int | Tokens generated by inference backends |
| `delegation_events` | list | Canonical delegation telemetry (who delegated to whom) |
| `tools_success` | bool\|None | Whether tool outputs contributed to final answer |
| `delegation_success` | bool\|None | Whether delegation contributed to final answer |

### Import Architecture Note

**Canonical prompt builders live in `src/prompt_builders.py`.** The deprecated wrapper at `src/api/services/orchestrator.py` delegates to prompt_builders — always import from the canonical source in new code. The wrapper exists only for backward compatibility. Safety tests in `tests/unit/test_api_imports.py` enforce this.

### Infrastructure Modules

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `src/features.py` | Feature flag system | - |
| `src/roles.py` | Role enum definitions | - |
| `src/escalation.py` | Unified escalation policy | roles |
| `src/config.py` | Centralized configuration | - |
| `src/prompt_builders.py` | Unified prompt construction (canonical source) | roles, routing_facade |
| `src/backends/protocol.py` | Backend interface protocols | - |

### Backend Layer

The backend layer uses `httpx.Client` with connection pooling for HTTP communication with llama-server instances:

```python
httpx.Client(
    base_url=config.base_url,
    limits=httpx.Limits(
        max_connections=20,           # Total connections
        max_keepalive_connections=10, # Persistent idle
        keepalive_expiry=60.0,        # Seconds
    ),
)
```

**Performance**: ~6x latency reduction for subsequent requests (3ms → 0.5ms).

| Backend | Purpose | HTTP Client |
|---------|---------|-------------|
| `LlamaServerBackend` | Production inference via llama-server | httpx (pooled) |
| `CachingBackend` | Prefix caching wrapper | Wraps LlamaServerBackend |
| `MockBackend` | Testing without real inference | None |

### Module Dependency Graph

```
features.py ←── api.py
                  │
                  ├──→ llm_primitives.py ──→ backends/*
                  │
                  ├──→ routing_facade.py ──→ escalation.py, failure_router.py
                  │                              ↑
                  ├──→ executor.py ──────────────┤
                  │                              │
                  ├──→ escalation.py ────────────┘
                  │
                  └──→ repl_environment.py
```

---

## Escalation System

### Single Source of Truth

**Use `src/routing_facade.py` for all escalation decisions.**  
`src/escalation.py` remains the authoritative rule source; learned escalation is advisory.

The legacy `failure_router.py` is preserved for backwards compatibility but new code should use the routing facade.

### Escalation Chains

```
Worker Chain:    worker_general → coder_escalation → architect_general → FAIL
Coder Chain:     coder_escalation → architect_general → FAIL
Ingest Chain:    ingest_long_context → architect_general → FAIL
Frontdoor Chain: frontdoor → coder_escalation → architect_general → FAIL
```

### Error Categories

| Category | Behavior | Example |
|----------|----------|---------|
| `CODE` | Standard retry → escalate | Type error, test failure |
| `LOGIC` | Standard retry → escalate | Wrong output |
| `FORMAT` | Retry only, never escalate | Formatting issue |
| `SCHEMA` | Retry only, never escalate | JSON schema violation |
| `TIMEOUT` | Skip if optional gate | Gate timeout |
| `EARLY_ABORT` | Immediate escalation | Model showed failure signs |

### Usage Example

```python
from src.escalation import EscalationPolicy, EscalationContext, ErrorCategory
from src.roles import Role

policy = EscalationPolicy()
context = EscalationContext(
    current_role=Role.CODER_PRIMARY,
    failure_count=2,
    error_category=ErrorCategory.CODE,
    error_message="Tests failed",
)

decision = policy.decide(context)
if decision.should_escalate:
    print(f"Escalating to {decision.target_role}")
elif decision.should_retry:
    print(f"Retrying, {decision.retries_remaining} left")
```

---

## Feature Flag System

### Philosophy

1. Core orchestration works with ALL features disabled
2. Features are opt-in by default in tests
3. Each feature can be toggled independently
4. Dependencies are documented and validated

### Available Features

| Feature | Description | Default (Test) | Default (Prod) |
|---------|-------------|----------------|----------------|
| `memrl` | Memory-based RL (Q-scoring, learned routing) | Off | On |
| `tools` | Tool registry for REPL | Off | On |
| `scripts` | Script registry (requires tools) | Off | On |
| `streaming` | SSE streaming endpoints | Off | On |
| `openai_compat` | OpenAI-compatible API | Off | On |
| `repl` | REPL execution (includes legacy `react` alias paths) | On | On |
| `caching` | Response caching | Off | On |

### Environment Variables

```bash
# Enable specific features
export ORCHESTRATOR_MEMRL=1
export ORCHESTRATOR_TOOLS=1
export ORCHESTRATOR_STREAMING=1

# Check enabled features
python3 -c "from src.features import features; print(features().enabled_features())"
```

### Usage in Code

```python
from src.features import features

if features().memrl:
    from orchestration.repl_memory import TaskEmbedder
    embedder = TaskEmbedder()
else:
    embedder = None

if features().tools:
    from src.tool_registry import ToolRegistry
    registry = ToolRegistry()
```

---

## Security Model

### REPL Sandboxing

The REPL uses AST-based security validation (not regex) to prevent sandbox escapes.

**Blocked Operations:**
- All imports of dangerous modules (os, sys, subprocess, etc.)
- Dangerous built-ins (eval, exec, open, getattr, etc.)
- Dunder attribute access (__class__, __globals__, etc.)
- Subscript access to dunder attributes (obj["__class__"])

**Why AST?**
Regex validation is bypassable:
```python
# Regex misses this:
getattr(__builtins__, '__im' + 'port__')('os')

# AST catches it because it sees the getattr() call
```

### REPL Permission Architecture

The sandbox has two layers:

1. **User Code Layer** (AST-validated)
   - Code generated by the LLM
   - Subject to `FORBIDDEN_CALLS`, `FORBIDDEN_MODULES`, `FORBIDDEN_ATTRS`
   - Cannot use `open()`, `import`, `exec()`, etc.

2. **Trusted Tool Layer** (unrestricted)
   - Methods like `_ocr_document()`, `_peek()`, `_list_dir()`
   - Can use any Python functionality internally
   - Must validate paths against `ALLOWED_FILE_PATHS`
   - Registered in `_build_globals()` for user access

```
┌─────────────────────────────────────────────────────────────┐
│  User Code (LLM-generated)                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  result = ocr_document('/path/to/file.pdf')          │  │
│  │  info = file_info('/path/to/file')                   │  │
│  │  FINAL(result)                                        │  │
│  └───────────────────────────────────────────────────────┘  │
│              │                                              │
│              │ (calls trusted tools)                        │
│              ▼                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Trusted Tools (_ocr_document, _file_info, etc.)     │  │
│  │  - Can use open(), subprocess, requests              │  │
│  │  - MUST validate paths against ALLOWED_FILE_PATHS    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Adding New Trusted Tools

To add a tool that needs filesystem/network access:

1. **Add the method to `REPLEnvironment`** (prefix with `_`):
   ```python
   def _my_tool(self, path: str) -> str:
       """Tool description for documentation."""
       # ALWAYS validate paths first
       is_valid, error = self._validate_file_path(path)
       if not is_valid:
           return f"[ERROR: {error}]"

       # Now safe to use filesystem
       with open(path, 'r') as f:
           return f.read()
   ```

2. **Register in `_build_globals()`**:
   ```python
   globals_dict = {
       # ... existing ...
       "my_tool": self._my_tool,
   }
   ```

3. **Document in `build_root_lm_prompt()`** (src/prompt_builders.py):
   ```python
   "- `my_tool(path)`: Description of what it does",
   ```

### Research Context Tracker

The REPL includes a Research Context Tracker (`src/research_context.py`) that builds a DAG of tool invocations:

```
## Research Context
Progress: 1 analyzed, 2 pending, 0 stale

[+] G1: grep(pattern='error')
    -> Found 3 matches in log...
  [?] P1: peek(n=100)
      -> Error at line 42...
      refs: G1
```

**Features:**
- **Auto IDs**: G1 (grep), P1 (peek), L1 (llm_call), T1 (TOOL), D1 (list_dir)
- **Parent tracking**: Sequential calls auto-link to previous node
- **Cross-refs**: String regex + semantic similarity (cosine > 0.7 via BGE embeddings)
- **State injection**: Rendered tree included in `get_state()` when >= 3 nodes

**Configuration:**
```python
ResearchContext(
    use_semantic=True,        # Semantic refs ON by default
    semantic_threshold=0.7,   # Similarity threshold
)
```

**Integration:** Automatically hooked into `peek()`, `grep()`, `list_dir()`, `TOOL()`. Serialized in checkpoint/restore.

### Configuring File Path Restrictions

Allowed paths are in `REPLEnvironment.ALLOWED_FILE_PATHS`:

```python
ALLOWED_FILE_PATHS = [
    "/mnt/raid0/llm/",  # RAID array (production data)
    "/tmp/",            # Temporary files
]
```

**To add new allowed paths:**

1. Edit `ALLOWED_FILE_PATHS` in `src/repl_environment.py`
2. Be extremely careful - path validation uses `os.path.realpath()` to resolve symlinks
3. Consider security implications (can models read sensitive files?)

### Permission Tiers (Future)

For filesystem management use cases, consider implementing permission tiers:

| Tier | Read | Write | Delete | Use Case |
|------|------|-------|--------|----------|
| `readonly` | ✓ | ✗ | ✗ | Default, safe exploration |
| `write_safe` | ✓ | ✓ (allowed paths) | ✗ | Code generation, file creation |
| `admin` | ✓ | ✓ | ✓ (with confirmation) | Filesystem management |

This would require:
1. Adding a `permission_tier` parameter to `REPLEnvironment`
2. Checking tier in each tool method
3. Adding write/delete tools gated by tier

### Thread Safety

Statistics counters in `AppState` are protected by `threading.Lock`:
- `increment_request()` - thread-safe
- `increment_active()` / `decrement_active()` - thread-safe
- `get_stats()` - returns consistent snapshot

---

## Adding New Features

### Step-by-Step Guide

1. **Add feature flag** (`src/features.py`):
   ```python
   @dataclass
   class Features:
       # ... existing ...
       my_feature: bool = False  # Add with description
   ```

2. **Add environment variable** (in `get_features()`):
   ```python
   "my_feature": _env_bool("MY_FEATURE", defaults["my_feature"]),
   ```

3. **Add optional import** (if module is heavy):
   ```python
   # In api.py or relevant module
   MyModule: type | None = None

   def _load_optional_imports():
       if f.my_feature:
           from src.my_module import MyModule as _MM
           MyModule = _MM
   ```

4. **Guard feature code**:
   ```python
   if features().my_feature and MyModule:
       instance = MyModule()
   else:
       instance = None
   ```

5. **Add tests for both states**:
   ```python
   def test_my_feature_enabled():
       set_features(Features(my_feature=True))
       # test enabled behavior

   def test_my_feature_disabled():
       set_features(Features(my_feature=False))
       # test disabled behavior
   ```

6. **Update documentation**:
   - Add to `features.py` docstring
   - Add to this document's feature table
   - Add to CLAUDE.md if relevant to agents

### Adding New Roles

1. **Add to Role enum** (`src/roles.py`):
   ```python
   MY_NEW_ROLE = "my_new_role"
   """Description of what this role does."""
   ```

2. **Add tier mapping**:
   ```python
   _TIER_MAP: dict[Role, Tier] = {
       # ...
       Role.MY_NEW_ROLE: Tier.B,
   }
   ```

3. **Add escalation mapping** (if applicable):
   ```python
   _ESCALATION_MAP: dict[Role, Role] = {
       # ...
       Role.MY_NEW_ROLE: Role.ARCHITECT_GENERAL,
   }
   ```

4. **Add to the master model registry** (`epyc-inference-research/orchestration/model_registry.yaml`); the local `orchestration/model_registry.yaml` is a generated runtime view, so edit the master:
   ```yaml
   roles:
     my_new_role:
       tier: B
       model:
         name: "Model Name"
         path: "path/to/model.gguf"
       # ...
   ```

---

## Common Pitfalls

### Pitfalls Found During Refactoring

| Issue | Impact | Prevention |
|-------|--------|------------|
| Silent exception swallowing | Errors disappear, debugging nightmare | Always log with `logger.warning(..., exc_info=True)` |
| Global mutable state | Race conditions in concurrent requests | Use thread-safe methods, consider request-scoped state |
| Magic role strings | Typos cause silent bugs | Use `Role` enum exclusively |
| Duplicate escalation logic | Inconsistent behavior, maintenance burden | Use `src/escalation.py` only |
| Regex-based security | Bypassable with string tricks | Use AST-based validation |
| Hardcoded paths | Fails in different environments | Use config/environment variables |

### Memory Safety

**NEVER use `pytest -n auto` on high-core machines!**

This machine has 192 threads. Parallel pytest spawns ~192 workers, each loading models → OOM crash.

Safe commands:
```bash
pytest tests/               # Sequential (safe)
pytest tests/ -n 4          # Limited parallelism (safe)
# pytest tests/ -n auto     # DANGEROUS!
```

### Root Filesystem Protection

**ALL files must be on `/mnt/raid0/` - NEVER on root `/`.**

The root filesystem is a 120GB SSD. Writing large files causes system instability.

```bash
# Always verify paths
[[ "$PATH" == /mnt/raid0/* ]] || echo "ERROR: Not on RAID!"
```

---

## Testing Guidelines

### Test Organization

```
tests/
├── unit/                  # Unit tests (no I/O, fast)
│   ├── test_api.py
│   ├── test_escalation.py
│   ├── test_features.py
│   └── test_roles.py
├── integration/           # Integration tests (with I/O)
│   └── test_full_flow.py
└── conftest.py           # Shared fixtures, memory guards
```

### Testing Features

```python
from src.features import set_features, reset_features, Features

@pytest.fixture(autouse=True)
def reset_feature_flags():
    """Reset feature flags between tests."""
    yield
    reset_features()

def test_with_memrl_enabled():
    set_features(Features(memrl=True))
    # test behavior with MemRL

def test_with_memrl_disabled():
    set_features(Features(memrl=False))
    # test behavior without MemRL
```

### Testing Escalation

```python
from src.escalation import EscalationPolicy, EscalationContext, EscalationAction
from src.roles import Role

def test_worker_escalates_to_coder():
    policy = EscalationPolicy()
    context = EscalationContext(
        current_role=Role.WORKER_GENERAL,
        failure_count=3,  # Exceeds max_retries
    )
    decision = policy.decide(context)
    assert decision.action == EscalationAction.ESCALATE
    assert decision.target_role == Role.CODER_PRIMARY
```

---

## Appendix: Module Quick Reference

### Imports for Common Tasks

```python
# Feature flags
from src.features import features, get_features, set_features, Features

# Roles
from src.roles import Role, Tier, get_tier, get_escalation_chain

# Escalation
from src.escalation import (
    EscalationPolicy,
    EscalationContext,
    EscalationDecision,
    EscalationAction,
    ErrorCategory,
)

# Configuration
from src.config import get_config, OrchestratorConfigData, LLMConfig

# Prompt building
from src.prompt_builders import (
    PromptBuilder,
    build_root_lm_prompt,
    build_escalation_prompt,
    extract_code_from_response,
    classify_error,
)

# Backend protocols
from src.backends import (
    LLMBackend,
    StreamingBackend,
    CachingBackend,
    InferenceRequest,
    InferenceResult,
)

# HTTP backend with connection pooling
from src.backends.llama_server import LlamaServerBackend, ServerConfig

# LLM calls
from src.llm_primitives import LLMPrimitives, LLMPrimitivesConfig

# REPL
from src.repl_environment import REPLEnvironment, REPLConfig

# Research Context Tracker
from src.research_context import ResearchContext, ResearchNode
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ORCHESTRATOR_MEMRL` | Enable MemRL | 0 (test), 1 (prod) |
| `ORCHESTRATOR_TOOLS` | Enable tool registry | 0 (test), 1 (prod) |
| `ORCHESTRATOR_SCRIPTS` | Enable script registry | 0 (test), 1 (prod) |
| `ORCHESTRATOR_MOCK_MODE` | Force mock mode | 1 (test), 0 (prod) |
| `HF_HOME` | HuggingFace cache | /mnt/raid0/llm/cache/huggingface |
| `TMPDIR` | Temporary files | /mnt/raid0/llm/tmp |

---

## Routing Decision Stack (2026-02 update)

The runtime routing path now follows this order:

1. Heuristic priors (`src/api/routes/chat_routing.py`) produce soft role probabilities.
2. MemRL posterior evidence (`orchestration/repl_memory/retriever.py`) ranks candidates using robust Q-confidence and expected cost.
3. Risk-controlled confidence gate applies the effective threshold (raw or calibrated + conformal margin).
4. Escalation policy handles abstain/retry/escalate with conditional schema escalation (`src/escalation.py`, `src/graph/helpers.py`).

## Telemetry Contract (2026-02 update)

Routing/completion telemetry now includes:

- Decision provenance: `decision_source`, `was_forced`
- Confidence/cost: `similarity_topk`, `q_topk`, `q_robust_confidence`, `expected_cost_s`
- Delegation lineage: `producer_role`, `delegation_lineage`, `final_answer_role`
- Optional teacher metrics: `pass_teacher`, `pass_chosen`, `regret`, `speedup_vs_teacher`

These fields are emitted via `orchestration/repl_memory/progress_logger.py` and consumed by replay/diagnostics paths.

## Workspace State

Graph execution carries a bounded shared workspace object on `TaskState.workspace_state` (`src/graph/state.py`) with proposal->selection->broadcast updates in `src/graph/helpers.py`.
This provides a lightweight blackboard for objective/commitment/open-question continuity across escalation/delegation turns while keeping workspace growth bounded.
