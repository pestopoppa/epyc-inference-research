# Orchestrator Implementation & Optimization Research

**Goal**: Flesh out the detailed implementation plan for the hierarchical local-agent orchestrator.

**Status**: Research/Planning Phase

**Last Updated**: 2026-01-04

---

## Context Summary

### Existing Documentation

| Document | Purpose | Lines |
|----------|---------|-------|
| `research/Hierarchical_Orchestration_Methodology.md` | Main methodology + operational spec | ~1260 |
| `research/Hierarchical_Orchestration_Refinement.md` | Feedback & improvements | ~440 |
| `orchestration/model_registry.yaml` | Role → model mapping | ~1500 |
| `orchestration/task_ir.schema.json` | TaskIR JSON schema | ~250 |
| `orchestration/architecture_ir.schema.json` | ArchitectureIR JSON schema | ~340 |

### Existing Implementation

| File | Purpose | Status |
|------|---------|--------|
| `src/dispatcher.py` | Routes TaskIR to models | Functional |
| `src/registry_loader.py` | Loads model_registry.yaml | Functional |
| `src/executor.py` | Executes step plans | Stub |
| `src/context_manager.py` | Manages inter-step context | Stub |
| `src/model_server.py` | Inference request handling | Stub |

### Key Architecture Decisions (Already Made)

1. **Hierarchical, not swarm** - Strong models set trajectory, cheap workers execute
2. **Contracts first** - APIs/schemas constrain expansion
3. **Gates decide correctness** - No debate-driven convergence
4. **Mixed acceleration** - Spec decode for dense, MoE reduction for MoE, nothing for SSM

### Best Models (from Benchmarks)

| Role | Model | Speed | Method |
|------|-------|-------|--------|
| frontdoor | Qwen3-Coder-30B-A3B | 45.3 t/s | MoE 4 experts |
| coder_primary | Qwen3-Coder-30B-A3B | 45.3 t/s | MoE 4 experts |
| ingest_long_context | Qwen3-Next-80B-A3B | 11.6 t/s | MoE 2 experts (NO SPEC) |
| architect | Qwen3-235B-A22B | 6.75 t/s | MoE 4 experts |
| worker_general | Meta-Llama-3-8B | 25+ t/s | Prompt lookup |
| draft | Qwen2.5-Coder-0.5B | 85+ t/s | (for dense targets) |

---

## Key Open Questions

### 1. Model Server Architecture

**Current gap**: No implementation for loading/unloading models.

**Options**:
- A) Multiple llama-server instances (one per hot model)
- B) Single orchestrator process with subprocess llama-cli calls
- C) Custom server using llama.cpp C API bindings
- D) External server (vLLM, TGI) - not optimal for CPU

**Questions to answer**:
- How to handle warm → hot promotion?
- How to share KV cache between requests?
- How to manage memory residency policy?

### 2. Execution Strategy

**Current gap**: Dispatcher produces plan but doesn't execute it.

**Options**:
- A) Sequential execution (simple, no parallelism)
- B) Async execution with parallel groups
- C) Process pool for workers

**Questions to answer**:
- How to handle step dependencies?
- How to pass artifacts between steps?
- How to implement timeout handling?

### 3. Front Door Integration

**Current gap**: No voice/interactive interface.

**Options**:
- A) CLI-first (stdin/stdout JSON)
- B) HTTP API (FastAPI)
- C) Voice pipeline (Whisper → LLM → TTS)

**Questions to answer**:
- Where does the front door model run?
- How to handle streaming responses?
- How to maintain conversation context?

### 4. Gate Implementation

**Current gap**: Gates defined in schema but not automated.

**Current Makefile gates**:
- schema, format, lint, typecheck, unit, integration

**Questions to answer**:
- How to run gates after each step vs end-of-task?
- How to feed gate failures back to producing agent?
- How to handle escalation chain?

---

## Implementation Phases (Updated with RLM Integration)

### Phase 1: REPL Environment Foundation

**Goal**: Implement the RLM-style Python REPL environment for context-as-object handling.

**Tasks**:
1. Create sandboxed Python REPL with context variable injection
2. Implement `llm_call()` and `llm_batch()` primitives for sub-LM spawning
3. Add output capping (8192 chars) to prevent context inflation
4. Implement `FINAL(answer)` completion signaling

**Files to create/modify**:
- `src/repl_environment.py` - Sandboxed Python REPL
- `src/llm_primitives.py` - `llm_call()`, `llm_batch()` functions
- `src/context_store.py` - Context-as-variable storage

**Key insight from RLM**: The REPL is the unified execution environment. All context manipulation happens here, not in LLM context windows.

### Phase 2: Model Server Foundation

**Goal**: Reliable model loading/inference for Root LM and Sub-LMs.

**Tasks**:
1. Design model server abstraction (interface for load/unload/infer)
2. Implement llama-server based backend (one per hot model)
3. Implement memory residency policy (hot/warm/cold)
4. Add health checks and auto-recovery
5. **NEW**: Add prefix caching for repeated context slices

**Files to create/modify**:
- `src/model_server.py` - Main abstraction
- `src/backends/llama_server.py` - llama-server backend
- `src/memory_manager.py` - Residency policy

**RLM consideration**: Sub-LMs need fast cold start for parallel spawning via `llm_batch()`.

### Phase 3: Execution Engine (RLM-Style)

**Goal**: Execute Root LM → Sub-LM recursive calls with proper parallelism.

**Tasks**:
1. Implement async step execution with depth tracking (depth=0 root, depth=1 subs)
2. Add `llm_batch()` parallel execution (spawn multiple sub-LMs)
3. Implement artifact persistence via REPL variables
4. Add timeout (120s per REPL call) and retry logic
5. **NEW**: Implement emergent strategy support (peek, grep, partition+map)

**Files to create/modify**:
- `src/executor.py` - Full implementation with RLM patterns
- `src/workspace.py` - REPL-integrated artifact storage
- `src/context_manager.py` - Inter-step context via REPL variables

**RLM patterns to implement**:
```python
# Root LM can write code like:
chunk_size = len(context) // 4
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
summaries = llm_batch([f"Summarize: {c}" for c in chunks])
```

### Phase 4: Gate Integration

**Goal**: Automated quality gates with failure routing.

**Tasks**:
1. Implement gate runner (calls `make gates`)
2. Parse gate output for failures
3. Implement failure → agent routing (first fail → same agent, second → escalate)
4. Implement escalation chain (worker → specialist → architect)

**Files to create/modify**:
- `src/gate_runner.py` - Gate execution
- `src/failure_router.py` - Routing logic

### Phase 5: Front Door (Root LM Interface)

**Goal**: Interactive entry point that operates as RLM root.

**Tasks**:
1. Implement CLI interface (stdin/stdout)
2. **NEW**: Inject system prompt explaining REPL usage
3. Add HTTP API option (FastAPI)
4. Implement conversation context (stored in REPL)
5. (Optional) Voice pipeline

**Files to create/modify**:
- `src/cli.py` - CLI entry point
- `src/api.py` - HTTP API (FastAPI)
- `src/conversation.py` - Context tracking via REPL
- `src/prompts/root_lm_system.txt` - System prompt for REPL usage

**RLM front door prompt pattern**:
```
You are the Root LM orchestrator. You have access to a Python REPL with:
- `context`: A string variable containing the full user input
- `llm_call(prompt, context_slice)`: Call a sub-LM with narrowed context
- `llm_batch(prompts)`: Call multiple sub-LMs in parallel
- `FINAL(answer)`: Signal completion with final answer

You NEVER see the full context directly. Use Python to peek, grep, and partition.
```

### Phase 6: Optimization

**Goal**: Production-ready performance with RLM efficiency gains.

**Tasks**:
1. Profile and optimize hot paths (especially `llm_batch()`)
2. Implement model preloading for common sub-LM patterns
3. Add prefix caching for repeated context slices
4. **NEW**: Measure token efficiency (target: 60-80% reduction in root LM tokens)
5. Add metrics/telemetry
6. Stress testing with 1M+ token inputs

---

## Detailed Architecture Specification

### Component Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR PROCESS                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐     ┌─────────────────────────────────────────────────┐ │
│  │   CLI/API    │────▶│              REPL ENVIRONMENT                   │ │
│  │  (FastAPI)   │     │  ┌─────────────────────────────────────────────┐│ │
│  └──────────────┘     │  │  context = "..."  # User input as string    ││ │
│                       │  │  history = []     # Conversation history    ││ │
│                       │  │  artifacts = {}   # Step outputs            ││ │
│                       │  └─────────────────────────────────────────────┘│ │
│                       │                                                  │ │
│                       │  Built-in Functions:                            │ │
│                       │  ┌─────────────────────────────────────────────┐│ │
│                       │  │ llm_call(prompt, ctx_slice, role="worker")  ││ │
│                       │  │ llm_batch(prompts, role="worker")           ││ │
│                       │  │ FINAL(answer) / FINAL_VAR(var_name)         ││ │
│                       │  │ peek(n=500)  # First n chars of context     ││ │
│                       │  │ grep(pattern) # Regex search in context     ││ │
│                       │  └─────────────────────────────────────────────┘│ │
│                       └────────────────────────┬────────────────────────┘ │
│                                                │                          │
│                                                ▼                          │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │                        MODEL SERVER POOL                              ││
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        ││
│  │  │ Root LM    │ │ Coder      │ │ Worker x2  │ │ Ingest     │        ││
│  │  │ (Tier A)   │ │ (Tier B)   │ │ (Tier C)   │ │ (Tier B)   │        ││
│  │  │ Port 8080  │ │ Port 8081  │ │ Port 8082-3│ │ Port 8084  │        ││
│  │  │ HOT        │ │ HOT        │ │ HOT        │ │ WARM       │        ││
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘        ││
│  └──────────────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────────────┘
```

### Interface Definitions

#### 1. REPL Environment API

```python
class REPLEnvironment:
    """Sandboxed Python REPL with LLM primitives."""

    def __init__(self, context: str, model_pool: ModelServerPool):
        self.globals = {
            "context": context,           # The full input (never sent to LLM)
            "history": [],                # Conversation turns
            "artifacts": {},              # Named outputs from steps

            # Built-in functions
            "llm_call": self._llm_call,
            "llm_batch": self._llm_batch,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "peek": self._peek,
            "grep": self._grep,
        }
        self.model_pool = model_pool
        self.output_cap = 8192           # Max chars per sub-LM output
        self.timeout = 120               # Seconds per REPL cell

    def execute(self, code: str) -> tuple[str, bool]:
        """Execute Python code, return (output, is_final)."""
        ...

    def _llm_call(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker"
    ) -> str:
        """Call a sub-LM with optional context slice."""
        full_prompt = f"{prompt}\n\nContext:\n{context_slice}" if context_slice else prompt
        result = self.model_pool.infer(role, full_prompt)
        return result[:self.output_cap]  # Cap output

    def _llm_batch(
        self,
        prompts: list[str],
        role: str = "worker"
    ) -> list[str]:
        """Call multiple sub-LMs in parallel."""
        results = self.model_pool.infer_batch(role, prompts)
        return [r[:self.output_cap] for r in results]
```

#### 2. Model Server Pool API

```python
@dataclass
class ModelConfig:
    role: str                    # "frontdoor", "coder", "worker", etc.
    model_path: str              # Path to GGUF
    port: int                    # llama-server port
    residency: str               # "hot", "warm", "cold"
    acceleration: dict           # MoE experts, spec decode config, etc.

class ModelServerPool:
    """Manages multiple llama-server instances."""

    def __init__(self, configs: list[ModelConfig]):
        self.servers: dict[str, LlamaServer] = {}
        self.role_to_server: dict[str, str] = {}

    async def start_hot_servers(self):
        """Start all HOT residency servers at boot."""
        ...

    async def promote_to_hot(self, role: str):
        """Load a WARM model into memory."""
        ...

    async def infer(self, role: str, prompt: str) -> str:
        """Single inference call to a role."""
        server = self._get_server(role)
        return await server.complete(prompt)

    async def infer_batch(self, role: str, prompts: list[str]) -> list[str]:
        """Parallel inference calls (for llm_batch)."""
        server = self._get_server(role)
        return await asyncio.gather(*[
            server.complete(p) for p in prompts
        ])
```

#### 3. Root LM Execution Loop

```python
class Orchestrator:
    """Main execution loop for RLM-style orchestration."""

    def __init__(self, model_pool: ModelServerPool):
        self.model_pool = model_pool

    async def run(self, user_input: str) -> str:
        """Process user input through RLM architecture."""

        # 1. Create REPL environment with context
        repl = REPLEnvironment(context=user_input, model_pool=self.model_pool)

        # 2. Get Root LM system prompt
        system_prompt = self._get_root_lm_prompt()

        # 3. Root LM loop (depth=0)
        max_turns = 10
        for turn in range(max_turns):
            # Root LM receives: system prompt + REPL state + query
            root_prompt = f"{system_prompt}\n\nREPL State:\n{repl.get_state()}\n\nUser Query: {user_input}"

            # Root LM generates Python code
            code = await self.model_pool.infer("frontdoor", root_prompt)

            # Execute code in REPL
            output, is_final = repl.execute(code)

            if is_final:
                return output

        return "ERROR: Max turns exceeded without FINAL()"

    def _get_root_lm_prompt(self) -> str:
        return """You are the Root LM orchestrator. You have access to a Python REPL.

Variables available:
- `context`: Full user input as string (DO NOT print this directly)
- `history`: List of previous conversation turns
- `artifacts`: Dict of named outputs from previous steps

Functions available:
- `peek(n=500)`: Return first n chars of context (for inspection)
- `grep(pattern)`: Search context with regex, return matches
- `llm_call(prompt, ctx_slice="", role="worker")`: Call sub-LM
- `llm_batch(prompts, role="worker")`: Call multiple sub-LMs in parallel
- `FINAL(answer)`: Signal completion with final answer
- `FINAL_VAR(var_name)`: Signal completion, return contents of variable

Strategy:
1. Use peek() to understand context structure
2. Use grep() to find relevant sections
3. Use llm_batch() to process sections in parallel
4. Combine results and call FINAL()

Write Python code. Do NOT output prose."""
```

#### 4. Message Flow Example

```
User: "Summarize all Python files in this codebase"
      [context = 500KB of concatenated .py files]

Turn 1 - Root LM generates:
    files = context.split("### FILE: ")
    print(f"Found {len(files)} files")

Output: "Found 47 files"

Turn 2 - Root LM generates:
    summaries = llm_batch(
        [f"Summarize this Python file:\n{f[:4000]}" for f in files[:10]],
        role="worker"
    )
    artifacts["batch1"] = summaries

Output: [list of 10 summaries, each capped at 8192 chars]

Turn 3 - Root LM generates:
    # Continue with next batch...
    summaries2 = llm_batch(...)
    artifacts["batch2"] = summaries2

...

Turn N - Root LM generates:
    all_summaries = artifacts["batch1"] + artifacts["batch2"] + ...
    final_summary = llm_call(
        "Combine these file summaries into a cohesive codebase overview:",
        "\n\n".join(all_summaries),
        role="coder"
    )
    FINAL(final_summary)
```

### Role → Model Mapping (Production)

| Role | Model | Port | Residency | Acceleration | Max Concurrent |
|------|-------|------|-----------|--------------|----------------|
| `frontdoor` (Root LM) | Qwen3-Coder-30B-A3B | 8080 | HOT | MoE 4 experts | 1 |
| `coder` | Qwen3-Coder-30B-A3B | 8081 | HOT | MoE 4 experts | 2 |
| `worker` | Meta-Llama-3-8B | 8082-8085 | HOT | Prompt lookup | 4 |
| `ingest` | Qwen3-Next-80B-A3B | 8086 | WARM | MoE 2 experts | 1 |
| `architect` | Qwen3-235B-A22B | 8087 | WARM | MoE 4 experts | 1 |
| `math` | Qwen2.5-Math-7B | 8088 | WARM | Spec decode | 1 |

**Memory Budget** (1.13 TB available):
- HOT models: ~100GB (30B×2 + 8B×4)
- WARM models (mmap'd): ~300GB (80B + 235B + 7B)
- Headroom for KV cache: ~700GB

---

## Design Decisions Needed

### Decision 1: Model Server Pattern

| Option | Pros | Cons |
|--------|------|------|
| A) Multiple llama-server | Simple, isolated | Port management, memory overhead |
| B) Subprocess llama-cli | No server, simple | Cold start per request |
| C) C API bindings | Fastest, shared KV | Complex, binding maintenance |
| D) External server | Feature-rich | Not CPU-optimized |

**Recommendation**: Start with A (multiple llama-server), migrate to C for production.

### Decision 2: Workspace/Artifact Storage

| Option | Pros | Cons |
|--------|------|------|
| A) Filesystem (temp dirs) | Simple, debuggable | Cleanup complexity |
| B) In-memory dict | Fast, no cleanup | Lost on crash |
| C) SQLite | Persistent, queryable | Overhead |

**Recommendation**: A (filesystem) with structured paths.

### Decision 3: Front Door Model Residency

| Option | Pros | Cons |
|--------|------|------|
| A) Always hot (pinned) | Instant response | Memory use |
| B) Hot on first request | Saves memory | Cold start |

**Recommendation**: A (always hot) - front door is the user-facing latency.

---

## Files to Reference

| Purpose | Path |
|---------|------|
| Main methodology | `/mnt/raid0/llm/claude/research/Hierarchical_Orchestration_Methodology.md` |
| Refinement feedback | `/mnt/raid0/llm/claude/research/Hierarchical _Orchestration_Refinement.md` |
| Model registry | `/mnt/raid0/llm/claude/orchestration/model_registry.yaml` |
| TaskIR schema | `/mnt/raid0/llm/claude/orchestration/task_ir.schema.json` |
| Dispatcher (current) | `/mnt/raid0/llm/claude/src/dispatcher.py` |
| Benchmark results | `/mnt/raid0/llm/claude/research/RESULTS_SUMMARY.md` |
| Progress tracking | `/mnt/raid0/llm/claude/orchestration/progress/` |

---

## Next Steps

1. **Decide on model server architecture** (Question 1 above)
2. **Prototype Phase 1** (model server foundation)
3. **Iterate on design** as we hit implementation challenges

---

## Research Questions to Investigate

1. How does llama-server handle concurrent requests?
2. Can we share KV cache between requests with same prefix?
3. What's the cold start time for 45GB model (Qwen3-Next)?
4. How to detect when model output is garbage (quality check)?
5. Should workers be pooled or spawned on demand?

---

## NEW: Recursive Language Models (RLMs) Integration

**Paper**: [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab - MIT CSAIL, Dec 2025)

### Core Insight

> "Most people (mis)understand RLMs to be about LLMs invoking themselves. The deeper insight is LLMs *interacting with their own prompts as objects*."

RLMs treat long prompts as **external environment objects** within a Python REPL, allowing the LLM to:
- Programmatically examine, partition, and grep through context
- Spawn sub-LM instances for parallel processing
- Recursively call itself with transformed context subsets
- Process inputs **100x beyond context window** (tested up to 10M+ tokens)

### How RLMs Map to Our Orchestrator

| RLM Concept | Our Orchestrator Equivalent | Integration Opportunity |
|-------------|---------------------------|------------------------|
| Root LM (depth=0) | Tier A Front Door | Front door as orchestrator, never sees full context |
| Sub-LMs (depth=1) | Tier B/C Specialists/Workers | Workers receive narrowed context windows |
| REPL Environment | Workspace/Artifact Storage | Python REPL as unified execution environment |
| Context as Variable | Long-context ingestion | Qwen3-Next processes context, stores as variable |
| `llm_batch()` | Parallel workers | Spawn parallel Tier C workers |
| `FINAL(answer)` | Gate completion | Explicit completion signal |

### Key RLM Patterns to Adopt

1. **Context-as-Object Pattern**
   - Store large inputs in a Python variable, not LLM context
   - Root LM writes code to query/transform the context
   - Sub-LMs receive only relevant slices

2. **Emergent Decomposition Strategies** (no explicit training needed)
   - **Peeking**: Sample initial context structure
   - **Grepping**: Regex to filter relevant entries
   - **Partition + Map**: Chunk context, spawn parallel sub-LMs
   - **Summarization**: Condense subsets for parent analysis

3. **Tool Access Hierarchy**
   - Main RLM: Python REPL only (prevents context inflation)
   - Sub-LMs: Full tool access (web search, file ops, etc.)
   - This matches our Tier A (routing) vs Tier B/C (execution) split

4. **Output Capping**
   - Sub-LM outputs capped at 8192 chars (prevents context rot)
   - Main model sees compressed results only

### Proposed Integration: RLM-Enhanced Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Input (potentially millions of tokens)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  REPL Environment (Python)                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  context = "<entire user input as string>"              ││
│  │  # Root LM never sees this directly                     ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier A: Front Door (Root LM)                               │
│  - Receives: query + system prompt (how to use REPL)        │
│  - Emits: Python code to peek/grep/partition context        │
│  - Spawns: Sub-LM calls via llm_batch() → Tier B/C          │
│  - Model: Qwen3-Coder-30B-A3B (45 t/s, MoE 4)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Tier B: Coder │ │ Tier B: Ingest│ │ Tier C: Worker│
│ (depth=1)     │ │ (depth=1)     │ │ (depth=1)     │
│ Receives:     │ │ Receives:     │ │ Receives:     │
│ - Sliced ctx  │ │ - Chunk to    │ │ - Single file │
│ - Specific    │ │   summarize   │ │   to modify   │
│   task        │ │               │ │               │
│ Returns:      │ │ Returns:      │ │ Returns:      │
│ - Code diff   │ │ - Summary     │ │ - File diff   │
│ (capped)      │ │ (capped)      │ │ (capped)      │
└───────────────┘ └───────────────┘ └───────────────┘
```

### RLM Performance Expectations

From Prime Intellect benchmarks with GPT-5-mini:

| Task Type | RLM Improvement | Applicability to Us |
|-----------|-----------------|---------------------|
| Web research (DeepDive) | +47% | Codebase exploration |
| Long-context (Oolong) | +23% | Document ingestion |
| Math problems | -31% | Avoid for math tasks |
| Complex data handling | +18% | Refactoring, code review |

**Token efficiency**: 60-80% reduction in main model tokens through strategic sub-LM delegation.

### Implementation Considerations

**What RLMs solve for us**:
- Long-context handling without Qwen3-Next-80B's 11.6 t/s ceiling
- Context rot prevention (sub-LMs handle verbose outputs)
- Natural parallel decomposition via `llm_batch()`
- Drop-in compatibility (looks like normal LLM from outside)

**What we still need**:
- Async execution (not in reference impl)
- Prefix caching (for repeated context)
- Cost/runtime controls (deterministic execution)
- Training for optimal decomposition strategies

### Next Steps for RLM Integration

1. **Prototype**: Implement REPL environment with context-as-variable
2. **Test**: Compare RLM approach vs direct Qwen3-Next on long docs
3. **Benchmark**: Measure token efficiency and quality on our workloads
4. **Iterate**: Add async execution and caching

### References

- [arXiv Paper](https://arxiv.org/abs/2512.24601)
- [Author's Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Prime Intellect RLMEnv](https://www.primeintellect.ai/blog/rlm)
- [GitHub: recursive-llm](https://github.com/ysz/recursive-llm)

---

## Detailed Action Plan with Expectations & Mitigations

### Overview

This section provides a rigorous execution plan for each implementation phase, including:
- **Deliverables**: Concrete artifacts produced
- **Success Criteria**: How we know the phase is complete
- **Expected Challenges**: Known risks and unknowns
- **Mitigation Strategies**: How to handle failures
- **Dependencies**: What must be in place first
- **Go/No-Go Criteria**: Decision points for proceeding

---

### Phase 1: REPL Environment Foundation

#### Deliverables

| Artifact | Description | Acceptance Test |
|----------|-------------|-----------------|
| `src/repl_environment.py` | Sandboxed Python REPL | Executes `print(len(context))` without exposing context to caller |
| `src/llm_primitives.py` | `llm_call()`, `llm_batch()` stubs | Returns mock responses, tracks call count |
| `tests/test_repl_environment.py` | Unit tests | 100% coverage of REPL execution paths |
| `tests/test_sandboxing.py` | Security tests | Blocks `os.system()`, `subprocess`, `__import__` |

#### Success Criteria

1. REPL executes arbitrary Python code with `context` variable injected
2. `peek(n)` returns first n characters of context
3. `grep(pattern)` returns regex matches from context
4. `FINAL(answer)` signals completion and returns answer
5. Output capping enforced at 8192 chars
6. Timeout enforced at 120s per execution
7. Sandbox blocks dangerous operations (file I/O, network, subprocess)

#### Expected Challenges

| Challenge | Likelihood | Impact | Description |
|-----------|------------|--------|-------------|
| Sandbox escapes | Medium | High | Python's `exec()` is notoriously hard to sandbox |
| Memory exhaustion | Medium | Medium | Malicious code could allocate unlimited memory |
| Infinite loops | High | Low | Code could run forever without explicit loop limits |
| Import hijacking | Medium | High | Code could import modules that bypass restrictions |

#### Mitigation Strategies

| Challenge | Primary Mitigation | Fallback |
|-----------|-------------------|----------|
| Sandbox escapes | Use RestrictedPython or PyPy sandbox | Run REPL in subprocess with resource limits |
| Memory exhaustion | Set `resource.setrlimit(RLIMIT_AS)` | Kill subprocess on memory threshold |
| Infinite loops | Enforce 120s timeout via `signal.alarm()` | Run in subprocess with `timeout` |
| Import hijacking | Whitelist allowed modules (`re`, `json`, `math`) | Disable `__import__` entirely |

#### Dependencies

- None (foundational phase)

#### Go/No-Go Criteria

- **GO**: All acceptance tests pass, sandbox blocks all test escapes
- **NO-GO**: Sandbox escape discovered that cannot be mitigated → switch to subprocess isolation

---

### Phase 2: Model Server Foundation

#### Deliverables

| Artifact | Description | Acceptance Test |
|----------|-------------|-----------------|
| `src/model_server.py` | Abstract ModelServer interface | Defines `load()`, `unload()`, `infer()`, `health()` |
| `src/backends/llama_server.py` | llama-server HTTP client | Connects to running llama-server, returns completions |
| `src/memory_manager.py` | HOT/WARM/COLD residency policy | Tracks memory usage, enforces limits |
| `config/model_servers.yaml` | Server configuration | Port assignments, model paths, residency |
| `scripts/start_servers.sh` | Server launch script | Starts all HOT servers with correct flags |

#### Success Criteria

1. Can start llama-server for Qwen3-Coder-30B-A3B with MoE 4 experts
2. Can make inference request and receive valid response
3. Can start/stop servers programmatically
4. Health check detects server failures within 5s
5. Memory tracking accurate within 5% of actual usage
6. WARM → HOT promotion completes in <30s for 8B model

#### Expected Challenges

| Challenge | Likelihood | Impact | Description |
|-----------|------------|--------|-------------|
| Port conflicts | Low | Low | Other services using ports 8080-8088 |
| Server crashes | Medium | High | llama-server segfaults or hangs |
| Memory pressure | Medium | High | Multiple HOT models exhaust RAM |
| Startup latency | High | Medium | 30B model takes 60s+ to load |
| API incompatibility | Low | Medium | llama-server API changes between versions |

#### Mitigation Strategies

| Challenge | Primary Mitigation | Fallback |
|-----------|-------------------|----------|
| Port conflicts | Use high ports (18080+) or Unix sockets | Dynamic port allocation |
| Server crashes | Health check + auto-restart with backoff | Fall back to subprocess llama-cli |
| Memory pressure | Conservative HOT budget (80% of available) | Aggressive WARM eviction |
| Startup latency | Preload at orchestrator boot | Lazy load on first request (accept latency) |
| API incompatibility | Pin llama.cpp version, test on upgrade | Abstract interface allows backend swap |

#### Dependencies

- llama.cpp built with server support
- Models available in GGUF format
- Sufficient RAM for HOT models (~100GB)

#### Go/No-Go Criteria

- **GO**: Can reliably start/stop servers, make inference requests with <1% failure rate
- **NO-GO**: Server instability >5% → investigate llama-server bugs, consider vLLM fallback

---

### Phase 3: Execution Engine (RLM-Style)

#### Deliverables

| Artifact | Description | Acceptance Test |
|----------|-------------|-----------------|
| `src/executor.py` | Full RLM execution loop | Runs Root LM → Sub-LM calls, returns final answer |
| `src/workspace.py` | Artifact storage | Persists/retrieves step outputs |
| `src/context_manager.py` | Inter-step context | REPL variables persist across turns |
| `tests/test_executor_integration.py` | E2E tests | Full task execution with mock LLMs |

#### Success Criteria

1. Root LM receives system prompt + REPL state
2. Root LM-generated code executes in REPL
3. `llm_call()` routes to correct model server
4. `llm_batch()` executes calls in parallel
5. Artifacts persist across turns via `artifacts[]` dict
6. `FINAL(answer)` terminates loop and returns answer
7. Max 10 turns enforced
8. Per-turn timeout of 120s enforced

#### Expected Challenges

| Challenge | Likelihood | Impact | Description |
|-----------|------------|--------|-------------|
| Root LM generates invalid Python | High | Medium | Syntax errors, undefined variables |
| Root LM never calls FINAL() | Medium | Medium | Infinite loop through turns |
| Sub-LM returns garbage | Medium | High | Output doesn't match expected format |
| Parallel execution races | Medium | Medium | `llm_batch()` results arrive out of order |
| Context state corruption | Low | High | REPL state becomes inconsistent |

#### Mitigation Strategies

| Challenge | Primary Mitigation | Fallback |
|-----------|-------------------|----------|
| Invalid Python | Catch SyntaxError, return error to Root LM | Retry with simplified prompt |
| Never calls FINAL() | Max turns limit (10), detect stuck patterns | Force FINAL() with best-effort answer |
| Sub-LM garbage | Output validation, retry once | Return error string, let Root LM handle |
| Parallel races | Use `asyncio.gather()` with ordered results | Sequential fallback mode |
| State corruption | Immutable snapshots before each turn | Reset REPL on corruption detection |

#### Dependencies

- Phase 1 (REPL Environment) complete
- Phase 2 (Model Server) complete

#### Go/No-Go Criteria

- **GO**: E2E test passes with real models, <10% failure rate on diverse prompts
- **NO-GO**: >30% failure rate → add more guardrails, improve Root LM prompt

---

### Phase 4: Gate Integration

#### Deliverables

| Artifact | Description | Acceptance Test |
|----------|-------------|-----------------|
| `src/gate_runner.py` | Runs `make gates` | Captures exit code, parses output |
| `src/failure_router.py` | Routes failures to agents | First fail → same agent, second → escalate |
| `config/gates.yaml` | Gate configuration | Which gates run when, failure thresholds |

#### Success Criteria

1. Gates run automatically after code-producing steps
2. Gate failures captured with structured error info
3. Failure routed back to producing agent with context
4. Second failure triggers escalation
5. Escalation chain: worker → specialist → architect
6. Gate timeout of 60s enforced

#### Expected Challenges

| Challenge | Likelihood | Impact | Description |
|-----------|------------|--------|-------------|
| Gate flakiness | Medium | Medium | Tests pass/fail non-deterministically |
| Slow gates | High | Medium | Full test suite takes >5 min |
| Cascading failures | Medium | High | One failure causes many downstream failures |
| Unhelpful error messages | High | Medium | Agent can't understand what to fix |

#### Mitigation Strategies

| Challenge | Primary Mitigation | Fallback |
|-----------|-------------------|----------|
| Gate flakiness | Retry flaky gates 2x | Mark known-flaky gates as warnings |
| Slow gates | Run fast gates first, slow gates async | Skip slow gates for interactive tasks |
| Cascading failures | Stop on first failure per category | Batch failures, deduplicate |
| Unhelpful errors | Parse output, extract actionable info | Include raw output in context |

#### Dependencies

- Phase 3 (Execution Engine) complete
- Existing Makefile gates working

#### Go/No-Go Criteria

- **GO**: Gates catch real errors, agents can fix >50% of failures on first retry
- **NO-GO**: Agent success rate <20% → improve error parsing, add examples to prompts

---

### Phase 5: Front Door (Root LM Interface)

#### Deliverables

| Artifact | Description | Acceptance Test |
|----------|-------------|-----------------|
| `src/cli.py` | CLI entry point | `echo "prompt" | python cli.py` returns answer |
| `src/api.py` | HTTP API (FastAPI) | POST /chat returns streaming response |
| `src/conversation.py` | Conversation state | Multi-turn context preserved |
| `src/prompts/root_lm_system.txt` | System prompt | Explains REPL usage to Root LM |

#### Success Criteria

1. CLI accepts stdin prompt, prints response to stdout
2. HTTP API accepts JSON, returns streaming response
3. Conversation history maintained across turns
4. System prompt injection works for Root LM
5. Response latency <5s to first token (HOT model)
6. Graceful shutdown on Ctrl+C

#### Expected Challenges

| Challenge | Likelihood | Impact | Description |
|-----------|------------|--------|-------------|
| Streaming complexity | Medium | Medium | HTTP streaming requires careful buffering |
| Conversation state size | Medium | Low | Long conversations exceed context |
| Root LM prompt tuning | High | High | Model doesn't follow REPL instructions |
| Latency spikes | Medium | Medium | Cold model causes >30s latency |

#### Mitigation Strategies

| Challenge | Primary Mitigation | Fallback |
|-----------|-------------------|----------|
| Streaming complexity | Use SSE (Server-Sent Events) | Return full response (no streaming) |
| Conversation size | Summarize old turns, cap at 10 turns | Reset conversation, warn user |
| Root LM prompt tuning | Iterative prompt engineering, few-shot examples | Fine-tune model on REPL usage |
| Latency spikes | Keep frontdoor HOT always | Show "loading..." message |

#### Dependencies

- Phase 3 (Execution Engine) complete
- Phase 4 (Gate Integration) complete (for code tasks)

#### Go/No-Go Criteria

- **GO**: Interactive session feels responsive (<10s latency), Root LM follows instructions >80%
- **NO-GO**: Root LM compliance <50% → invest in prompt engineering or fine-tuning

---

### Phase 6: Optimization

#### Deliverables

| Artifact | Description | Acceptance Test |
|----------|-------------|-----------------|
| `benchmarks/rlm_efficiency.py` | Token efficiency benchmark | Measures Root LM vs Sub-LM token usage |
| `src/prefix_cache.py` | Prefix caching for repeated context | Cache hit rate >50% on similar prompts |
| `docs/telemetry.md` | Metrics documentation | Lists all tracked metrics |
| `config/prometheus.yaml` | Prometheus config | Metrics exposed at /metrics |

#### Success Criteria

1. Token efficiency: 60-80% reduction in Root LM tokens
2. Latency p50 <5s, p99 <30s for typical tasks
3. Throughput: >10 concurrent requests without degradation
4. Memory stable under sustained load (no leaks)
5. Prefix cache hit rate >50% for code tasks
6. 1M+ token inputs handled without OOM

#### Expected Challenges

| Challenge | Likelihood | Impact | Description |
|-----------|------------|--------|-------------|
| Token efficiency below target | Medium | High | RLM overhead negates benefits |
| Memory leaks | Medium | High | Long-running process grows unbounded |
| Prefix cache misses | High | Medium | Cache key too specific |
| Large input OOM | Medium | High | 1M tokens exceeds memory even with RLM |

#### Mitigation Strategies

| Challenge | Primary Mitigation | Fallback |
|-----------|-------------------|----------|
| Token efficiency | Tune batch sizes, reduce REPL state in prompt | Accept lower efficiency, still better than baseline |
| Memory leaks | Profile with memray, fix leaks | Periodic worker restart |
| Prefix cache misses | Use semantic hashing, not exact match | Disable caching, accept latency |
| Large input OOM | Stream-process context in chunks | Hard limit at 500K tokens |

#### Dependencies

- All previous phases complete
- Production workload available for benchmarking

#### Go/No-Go Criteria

- **GO**: Meets latency/throughput targets, stable under load
- **NO-GO**: Critical performance issues → profile and optimize specific bottlenecks

---

## Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|----|------|------------|--------|------------|-------|
| R1 | Python sandbox escape | Medium | Critical | RestrictedPython + subprocess isolation | Phase 1 |
| R2 | llama-server instability | Medium | High | Health checks + auto-restart | Phase 2 |
| R3 | Root LM doesn't follow REPL instructions | High | High | Prompt engineering + few-shot examples | Phase 5 |
| R4 | Token efficiency below 50% | Medium | Medium | Accept overhead, still viable | Phase 6 |
| R5 | Memory exhaustion with large inputs | Medium | High | Streaming + hard limits | Phase 6 |
| R6 | Gate flakiness causes false failures | Medium | Medium | Retry logic + known-flaky list | Phase 4 |
| R7 | Parallel execution deadlocks | Low | High | Timeouts + sequential fallback | Phase 3 |
| R8 | Model output quality degradation | Medium | High | Quality gates + escalation | Phase 4 |

---

## Validation Checkpoints

### Checkpoint 1: Proof of Concept (After Phase 2)

**Objective**: Demonstrate basic RLM loop works with real models.

**Test Scenario**:
```
Input: "What is 2 + 2?"
Expected: Root LM generates `FINAL("4")` or equivalent
```

**Success Criteria**:
- Root LM receives prompt
- REPL executes generated code
- Answer returned correctly
- Round-trip latency <30s

### Checkpoint 2: Parallel Execution (After Phase 3)

**Objective**: Demonstrate `llm_batch()` parallelism works.

**Test Scenario**:
```
Input: Context with 4 distinct sections
Task: Summarize each section
Expected: 4 parallel sub-LM calls, combined result
```

**Success Criteria**:
- 4 sub-LM calls execute in parallel
- Total latency < 2x single call latency
- Results correctly aggregated

### Checkpoint 3: Long Context (After Phase 3)

**Objective**: Demonstrate RLM handles context beyond model window.

**Test Scenario**:
```
Input: 100KB document (exceeds 32K context)
Task: Find specific information deep in document
Expected: Root LM greps, finds, returns answer
```

**Success Criteria**:
- Context stored as variable, not in LLM prompt
- Root LM uses `grep()` or chunking strategy
- Correct answer found
- Root LM context usage <8K tokens

### Checkpoint 4: Production Readiness (After Phase 6)

**Objective**: System handles real workloads reliably.

**Test Scenario**:
```
Load: 10 concurrent requests over 1 hour
Tasks: Mix of code, summarization, Q&A
```

**Success Criteria**:
- 95% success rate
- p99 latency <60s
- No memory growth >10%
- No unhandled exceptions

---

## Decision Log

Track key decisions made during implementation.

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-01-04 | Use multiple llama-server (Option A) | Simplest to implement, isolation | Subprocess (B), C bindings (C) |
| 2026-01-04 | Filesystem artifacts (Option A) | Debuggable, simple | In-memory (B), SQLite (C) |
| 2026-01-04 | Front door always HOT | User-facing latency critical | Load on demand (B) |
| TBD | Sandbox approach | To be decided after Phase 1 investigation | RestrictedPython vs subprocess |

---

## Rollback Procedures

### Phase 1 Rollback
If REPL sandbox is fundamentally broken:
- Switch to subprocess execution with resource limits
- Accept 50-100ms overhead per execution
- Proceed with Phase 2

### Phase 2 Rollback
If llama-server is too unstable:
- Fall back to subprocess llama-cli calls
- Accept cold-start latency (5-10s per call)
- Consider vLLM as alternative backend

### Phase 3 Rollback
If RLM execution loop fails too often:
- Revert to direct model calls (no REPL intermediary)
- Accept limited context handling
- Revisit RLM after model improvements

### Full System Rollback
If orchestrator fundamentally doesn't work:
- Document learnings
- Fall back to Claude Code for interactive work
- Use local models only for batch processing
