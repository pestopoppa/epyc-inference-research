# Chapter 11: REPL Environment & Sandboxing

## Introduction

The REPL (Read-Eval-Print Loop) environment provides sandboxed Python execution for orchestrator agents. It enables agents to explore large contexts without transmitting them through the LLM, reducing token costs by orders of magnitude. Built-in functions (`peek`, `grep`, `FINAL`) manipulate context locally, while AST-based security prevents sandbox escapes.

This architecture implements RLM-style (Retrieval-augmented Language Model) orchestration where the large document remains in local memory and the agent uses code to extract relevant portions.

## Security Architecture

The REPL uses AST-based validation to catch sandbox escape attempts that regex would miss -- things like string concatenation tricks or dunder attribute access. There are two execution backends: a fast built-in AST validator and an optional RestrictedPython layer from the Zope/Plone ecosystem.

<details>
<summary>AST validation and dual-layer sandboxing details</summary>

### AST-Based Validation

The `ASTSecurityVisitor` class analyzes parsed syntax trees before execution, making it immune to string concatenation tricks:

<details>
<summary>Code: ASTSecurityVisitor forbidden lists</summary>

```python
class ASTSecurityVisitor(ast.NodeVisitor):
    FORBIDDEN_MODULES = frozenset({
        "os", "sys", "subprocess", "socket", "shutil", "pathlib",
        "tempfile", "multiprocessing", "threading", "ctypes", "pickle",
        "importlib", "builtins", "code", "codeop", "runpy", "pkgutil",
    })

    FORBIDDEN_CALLS = frozenset({
        "__import__", "eval", "exec", "compile", "open",
        "getattr", "setattr", "delattr", "hasattr",
        "globals", "locals", "vars", "dir",
    })

    FORBIDDEN_ATTRS = frozenset({
        "__class__", "__bases__", "__subclasses__", "__mro__",
        "__dict__", "__globals__", "__locals__", "__code__",
        "__builtins__", "__closure__",
    })
```

</details>

**Why AST over Regex**: String patterns like `getattr(__builtins__, '__im' + 'port__')('os')` bypass regex checks but are caught during AST analysis.

### Dual-Layer Sandboxing

The system offers two execution backends:

| Backend | Security | Performance | Dependencies |
|---------|----------|-------------|--------------|
| **Custom AST** | AST validation + timeout | Fast | Built-in only |
| **RestrictedPython** | `compile_restricted` + guards | Medium | RestrictedPython >= 7.0 |

RestrictedPython (optional) provides battle-tested sandbox used by Zope/Plone, with `PrintCollector` for stdout capture and guarded attribute access.

</details>

## Built-In Functions

Agents interact with documents through a small set of free primitives: `peek` for previewing, `grep` for searching, and `FINAL` for returning answers. On top of those, NextPLAID-backed code and doc search provide multi-vector retrieval, and archive/web functions handle external content.

<details>
<summary>Full function reference tables and tool call examples</summary>

### Context Exploration

| Function | Purpose | Cost | Use Case |
|----------|---------|------|----------|
| `peek(n)` | First n chars | Free | Preview document structure |
| `grep(pattern)` | Regex search | Free | Find specific content |
| `FINAL(answer)` | Terminate with result | Free | Return final answer |

### Code & Document Retrieval (NextPLAID)

| Function | Purpose | Example |
|----------|---------|---------|
| `code_search(query)` | Multi-vector code search | Find function definitions, patterns |
| `doc_search(query)` | Multi-vector doc search | Find relevant documentation sections |

Uses ColBERT token-level matching via NextPLAID (:8088, LateOn-Code 130M, 128-dim). AST-chunked code units (functions, classes, methods with signatures) instead of naive character splits. Complements `recall()` (episodic memory) — `code_search` finds actual source code, `recall` finds past routing decisions. ColGrep CLI (`colgrep`) provides agent-facing hybrid search over the same codebase index.

### Extended Functions (Archive/Web)

| Function | Purpose | Example |
|----------|---------|---------|
| `archive_open(path)` | Extract ZIP/TAR/PDF | Process research papers |
| `archive_file(name)` | Get specific file | Read extracted document |
| `archive_search(query)` | Search across archive | Find all references |
| `web_fetch(url)` | Fetch HTTP content | Download documentation |

### Tool Calls from REPL

<details>
<summary>Code: Calling external tools via tool_call()</summary>

```python
# Call external tools via tool_call()
result = tool_call("math_simplify", {"expression": "x^2 + 2x + 1"})
# Returns: {"simplified": "(x+1)^2", "steps": [...]}
```

</details>

Agents can invoke the tool registry (41 deterministic tools) from within REPL code.

</details>

## Execution Model

The REPL enforces resource limits (10-minute timeout via SIGALRM, output capping at 8KB), spills large outputs to disk with rolling LLM-generated summaries, and separates trusted built-in functions from sandboxed user code.

<details>
<summary>Resource limits, output spilling, and trust layers</summary>

### Resource Limits

<details>
<summary>Code: REPLConfig dataclass</summary>

```python
@dataclass
class REPLConfig:
    timeout_seconds: int = 600  # 10 min for document processing
    output_cap: int = 8192      # Spill-to-file threshold
    spill_dir: str = "/mnt/raid0/llm/tmp/repl_output"
    max_grep_results: int = 100 # Prevent DoS via grep
    require_exploration_before_final: bool = False  # Force peek/grep
    min_exploration_calls: int = 1  # Minimum calls before FINAL
```

</details>

## Structured Mode Chaining (Phase 2a)

Structured mode now uses AST-based tool-call discovery for gate decisions. This removes regex false positives from comments and string literals.

When multiple tools are present in one turn:

- All-read-only calls can still dispatch in parallel.
- Non-read-only calls are allowed only if chain-eligible:
  - registry tools opt in via `allowed_callers: ["chain"]`
  - selected REPL built-ins (`run_shell`, `run_python_code`, `file_write_safe`, `log_append`, `benchmark_run`) are chain-enabled
  - `TOOL` / `CALL` are allowed only when the wrapped registry tool is chain-enabled
- Routing primitives (`delegate`, `escalate`) are always rejected in chained turns.

During execution, each chained tool call carries chain metadata into the registry invocation log (`caller_type`, `chain_id`, `chain_index`) for response diagnostics and replay.

`dep` mode controls:

- `ORCHESTRATOR_TOOL_CHAIN_MODE=dep`: dependency-wave execution in structured mode.
- `ORCHESTRATOR_TOOL_CHAIN_PARALLEL_MUTATIONS=1`: enables parallel mutation waves for a conservative safe set (`run_shell`, `file_write_safe`, `log_append`, `benchmark_run`).
- Unsupported/ambiguous code shapes automatically fall back to sequential execution (`mode_used=seq`, `fallback_to_seq=true` in diagnostics).

**Timeout Enforcement**: UNIX `SIGALRM` signal terminates runaway executions (600s default for document ingestion, 120s for general use).

### Output Spill-to-File

When execution output exceeds `output_cap` (8192 chars), the full output is written to `{spill_dir}/{session_id}/turn_{N}.txt` and a summary replaces the raw output in the REPL feedback loop.

**Rolling summary pattern**: A worker model (Qwen2.5-7B) maintains a compressed scratchpad across turns. On first spill, it summarizes the tail of the output. On subsequent spills, it receives the previous summary + new tail and produces an updated summary — preserving key findings while dropping superseded details. Max 10 lines, 512 tokens.

**Fallback**: When `llm_primitives` is unavailable (unit tests), a static head (15 lines) + tail (5 lines) summary is used.

**On-demand access**: The summary footer includes `peek(spill_path)` and `grep(spill_path, pattern)` instructions so the model can pull specific details from the full output.

`PromptConfig.max_output_preview` is set to 1500 chars to accommodate the summary + header + footer without further truncation by the prompt builder.

### Trusted vs User Code Layers

| Layer | Execution Context | Restrictions |
|-------|-------------------|--------------|
| **Trusted** | Built-in functions (`peek`, `grep`, `archive_open`) | Full system access |
| **User** | Agent-generated code | AST validation, no file I/O |

Built-in functions execute in the trusted layer with access to file system (for archives) and network (for `web_fetch`), but user code is sandboxed.

</details>

## Exploration Logging

The `ExplorationLog` classifies agent strategies (scan, search, delegated, mixed) and computes token efficiency. These logs feed into MemRL's episodic memory so the system can learn which exploration approaches work best for different task types.

<details>
<summary>Strategy classification and MemRL integration</summary>

### Strategy Classification

The `ExplorationLog` tracks which primitives the agent used:

<details>
<summary>Code: Strategy type definitions</summary>

```python
strategy_types = {
    "scan": peek() > grep(),      # Sequential scanning
    "search": grep() > peek(),    # Targeted searching
    "delegated": llm_call() > 0,  # Sub-agent delegation
    "mixed": Multiple strategies
}
```

</details>

**Token Efficiency**: Calculated as `result_tokens / exploration_tokens`. Higher is better—indicates effective exploration with minimal LLM calls.

### MemRL Integration

Exploration logs feed into episodic memory for Q-learning:
- **Phase 1**: Log exploration strategy (scan/search/delegated)
- **Phase 2**: Compute token efficiency
- **Phase 3**: Update Q-values based on final outcome
- **Phase 4**: Retrieve similar explorations for future tasks

</details>

## Research Context Tracker

The Research Context Tracker builds a DAG of tool invocations within a REPL session. Each call gets a prefixed ID (like G1 for grep, P1 for peek), and the system detects cross-references between results both by string matching and semantic similarity. Once three or more nodes exist, a rendered tree is injected into the agent's context.

<details>
<summary>Node IDs, cross-references, context injection, and configuration</summary>

### Node ID Assignment

Each tool invocation receives a prefixed, auto-incrementing ID:

| Prefix | Tool | Example |
|--------|------|---------|
| G | grep | G1, G2, G3 |
| P | peek | P1, P2 |
| L | llm_call | L1 |
| T | TOOL | T1 |
| D | list_dir | D1 |
| W | web_fetch | W1 |
| R | recall | R1 |
| CS | code_search | CS1 |
| DS | doc_search | DS1 |

### Cross-Reference Detection

Two mechanisms detect when results reference each other:

1. **String References**: Regex detects explicit mentions like "see G1", "based on P2"
2. **Semantic References**: Cosine similarity > 0.7 using BGE embeddings (optional, ON by default)

### Context Injection

When >= 3 nodes exist, `get_state()` includes a rendered tree:

<details>
<summary>Code: Example rendered research context tree</summary>

```
## Research Context
Progress: 1 analyzed, 2 pending, 0 stale

[+] G1: grep(pattern='error')
    -> Found 3 matches in log file...
  [?] P1: peek(n=100)
      -> Error at line 42: connection timeout...
      refs: G1
```

</details>

Status markers: `[+]` analyzed, `[?]` pending, `[~]` stale

### Integration

- **Automatic tracking**: `peek()`, `grep()`, `list_dir()`, `TOOL()` auto-tracked
- **Parent inference**: Sequential calls use previous node as parent
- **Checkpoint support**: Serialized via `to_dict()`/`from_dict()`
- **Reset**: Cleared by `repl.reset()`

### Configuration

<details>
<summary>Code: ResearchContext constructor</summary>

```python
ResearchContext(
    use_semantic=True,        # Enable semantic cross-refs (default ON)
    semantic_threshold=0.7,   # Similarity threshold for refs
    embedder=None,            # Auto-initializes from BGE pool
)
```

</details>

</details>

---

## Performance Characteristics

The REPL approach delivers massive token savings compared to feeding full documents into the LLM. With peek/grep alone you get around 100x reduction; add TOON encoding for structured data and you get another 55% on top of that.

<details>
<summary>Token reduction benchmarks</summary>

### Token Reduction

| Approach | Tokens | Speedup | Use Case |
|----------|--------|---------|----------|
| **Full context to LLM** | 50,000 | 1x | Baseline (avoid) |
| **REPL with peek/grep** | 500 | 100x | Document QA |
| **REPL with archive tools** | 2,000 | 25x | Multi-file analysis |
| **REPL with TOON encoding** | 890 | 56x | Structured data (55.6% reduction) |

**TOON Encoding**: Enabled by default (`use_toon_encoding=True`). Reduces tokens by ~55% on structured tool outputs with 41.8% latency improvement (TTFT benchmark).

</details>

## Security Considerations

The sandbox blocks import bypasses, dunder escapes, string subscript tricks, and direct eval/exec calls via AST analysis. Known limitations include CPU-bound DoS (loops run until timeout), unbounded memory allocation, and regex DoS in grep patterns -- production deployments should layer cgroups on top.

<details>
<summary>Attack surface and known limitations</summary>

### Attack Surface

1. **Import Bypass**: Blocked by AST analysis of `Import` and `ImportFrom` nodes
2. **Dunder Escapes**: `obj.__class__.__bases__[0]` caught by `visit_Attribute`
3. **String Subscript**: `obj['__globals__']` caught by `visit_Subscript`
4. **Eval/Exec**: Direct calls blocked in `FORBIDDEN_CALLS`

### Known Limitations

- **CPU DoS**: Infinite loops terminate at timeout, but consume CPU until then
- **Memory Exhaustion**: Large string concatenations not limited (rely on OS)
- **Regex DoS**: Complex patterns in `grep()` can cause slowdown

**Mitigation**: Production deployments should use cgroups for hard resource limits.

</details>

## References

<details>
<summary>Implementation, security frameworks, and related approaches</summary>

### Implementation

1. `src/repl_environment/`: Modular REPL environment (environment.py, context.py, file_exploration.py, state.py, etc.)
2. `src/research_context.py`: Research Context Tracker (~270 lines)
3. `src/restricted_executor.py`: RestrictedPython backend (425 lines)
4. Python AST module documentation: https://docs.python.org/3/library/ast.html

### Security Frameworks

4. RestrictedPython project: https://github.com/zopefoundation/RestrictedPython
5. Plone CMS security model (uses RestrictedPython): https://plone.org/security

### Related Approaches

6. Jupyter Notebook sandboxing: https://jupyter-notebook.readthedocs.io/en/stable/security.html
7. PyPy sandboxing (deprecated): https://doc.pypy.org/en/latest/sandbox.html

</details>

## Unicode Sanitizer (2026-02-09)

Models frequently copy Unicode characters from question text into generated code -- for example, a chemistry question with "contact angle of 47deg" leads to `theta = 47°` which causes a SyntaxError. The `sanitize_code_unicode()` function strips or replaces about 25 common Unicode characters before execution using a single-pass compiled regex.

<details>
<summary>Sanitizer behavior and replacements</summary>

The `sanitize_code_unicode()` function in `src/repl_environment/unicode_sanitizer.py` runs before all three execution paths (`execute()`, `_execute_structured()`, `_run_python_code()`). It replaces ~25 common Unicode characters with ASCII equivalents via a single-pass compiled regex.

Key replacements: `°`→stripped, `×`→`*`, `−`→`-`, `²`→`**2`, curly quotes→straight quotes, non-breaking/zero-width spaces→stripped.

Fast path: `code.isascii()` returns immediately (zero overhead for clean code).

</details>

## Parallel Read-Only Tool Dispatch (2026-02-17)

AST-based two-pass dispatch enables parallel execution of independent read-only REPL calls via `ThreadPoolExecutor`. Parse code → extract independent read-only calls → if all safe, dispatch in parallel; otherwise fall through to sequential `exec()`.

**Core files**: `src/repl_environment/parallel_dispatch.py` (`_ParallelCall`, `_extract_parallel_calls()`, `execute_parallel_calls()`), `environment.py` (`_state_lock`, `_READ_ONLY_REPL_TOOLS` frozenset).

**Design decisions**:
- Conservative fallback: `_extract_parallel_calls()` returns `None` on any ambiguity → sequential `exec()`
- Coarse locking: single `_state_lock` on `REPLEnvironment`, held only for counter increments (nanoseconds); I/O stays outside lock
- Thread-safe helpers: `_increment_exploration()` in `file_exploration.py`, lock guards in `routing.py` and `code_search.py`

**Impact**: 2-4x speedup on multi-tool turns. Feature flag: `parallel_tools=True` (on by default). 22+ unit tests in `tests/unit/test_repl_parallel_dispatch.py`.

---

*Previous: [Chapter 10: Orchestration Architecture](10-orchestration-architecture.md)* | *Next: [Chapter 12: Production Server Stack](12-production-server-stack.md)*
