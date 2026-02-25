# Chapter 23: Security & Monitoring

## Introduction

The orchestration system implements defense-in-depth through three independent security mechanisms: **AST-based code validation** (prevents sandbox escape in the REPL), **generation monitoring** (detects degenerate model output via entropy and repetition tracking), and **path validation** (confines all file operations to the RAID array). Together these form a layered security model where the REPL sandbox prevents dangerous code execution, the generation monitor prevents wasted compute on failed generations, and path validation prevents root filesystem exhaustion.

This chapter covers each mechanism, their thresholds, and how they interact with the escalation system.

## AST-Based Code Validation

The REPL sandbox uses Python's `ast` module to inspect syntax trees before any LLM-generated code runs. This makes it immune to string concatenation tricks that defeat regex-based filters. A two-layer permission model lets trusted built-in tools call `open()` and `subprocess` internally while blocking those same operations in user-facing code, and an optional RestrictedPython layer adds guarded attribute access on top.

<details>
<summary>Architecture and forbidden operations</summary>

### Architecture

The primary REPL sandbox (`src/repl_environment.py`) uses Python's `ast` module to analyze syntax trees before execution, making it immune to string concatenation obfuscation that defeats regex-based approaches:

```python
# Regex approach (DEFEATED by):
getattr(__builtins__, '__im' + 'port__')('os')

# AST approach (CATCHES this):
# ast.Call node with ast.Attribute -> validates target name
```

### Forbidden Operations

**14 Forbidden Module Imports:**
```python
FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "tempfile", "multiprocessing", "threading", "ctypes", "pickle",
    "importlib", "builtins", "code", "codeop", "runpy", "pkgutil",
})
```

**13 Forbidden Function Calls:**
```python
FORBIDDEN_CALLS = frozenset({
    "__import__", "eval", "exec", "compile", "open",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir",
    "input", "breakpoint", "memoryview",
})
```

**20 Forbidden Dunder Attributes:**
```python
FORBIDDEN_ATTRS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__dict__", "__globals__", "__locals__", "__code__",
    "__builtins__", "__closure__", "__func__", "__self__",
    "__module__", "__qualname__", "__annotations__",
    "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
})
```

The `ASTSecurityVisitor` checks for direct calls (`eval(...)`), attribute access (`obj.__class__`), attribute calls (`obj.__class__()`), and subscript bypass attempts (`obj['__class__']`).

### Two-Layer Permission Model

```
+---------------------------------------------+
|  User Code (LLM-generated)                  |
|  +---------------------------------------+  |
|  | result = ocr_document('/path')        |  |  <- AST-validated
|  | info = file_info('/path')             |  |     Forbidden ops blocked
|  | FINAL(result)                         |  |
|  +---------------------------------------+  |
|              | (calls)                      |
|              v                              |
|  +---------------------------------------+  |
|  | Trusted Tools (_ocr_document, etc.)   |  |  <- Unrestricted internally
|  | Can use open(), subprocess, requests  |  |     Validates paths only
|  +---------------------------------------+  |
+---------------------------------------------+
```

LLM-generated code runs in the restricted layer where dangerous operations are blocked. Trusted built-in tools (prefixed with `_`) run unrestricted but validate file paths against the whitelist.

### RestrictedPython (Optional Second Layer)

An optional second sandbox layer (`src/restricted_executor.py`) uses the battle-tested `RestrictedPython` library:

- **Guarded attribute access** via `_getattr_`: blocks all `_`-prefixed attributes
- **Guarded subscript access** via `_getitem_`: blocks `_`-prefixed keys
- **Print collection** via `PrintCollector`: captures all stdout
- **Timeout** via `signal.SIGALRM`: default 120 seconds

<details>
<summary>Code: forbidden sets and AST visitor</summary>

```python
FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "tempfile", "multiprocessing", "threading", "ctypes", "pickle",
    "importlib", "builtins", "code", "codeop", "runpy", "pkgutil",
})

FORBIDDEN_CALLS = frozenset({
    "__import__", "eval", "exec", "compile", "open",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir",
    "input", "breakpoint", "memoryview",
})

FORBIDDEN_ATTRS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__dict__", "__globals__", "__locals__", "__code__",
    "__builtins__", "__closure__", "__func__", "__self__",
    "__module__", "__qualname__", "__annotations__",
    "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
})
```

</details>
</details>

## Generation Monitoring

The `GenerationMonitor` tracks token-by-token health during model output, detecting repetition loops, entropy collapse, and perplexity spikes. When output goes degenerate, it triggers early abort to prevent wasted compute. Each agent tier gets its own thresholds -- architects are allowed more variation while coders face strict repetition limits -- and a weighted combined score prevents false positives from any single noisy signal.

<details>
<summary>Metrics, thresholds, and abort logic</summary>

### Monitored Metrics

| Metric | Computation | What It Detects |
|--------|-------------|-----------------|
| **Shannon entropy** | `-sum(p * log(p))` from logits | Model uncertainty/randomness |
| **Repetition ratio** | Fraction of repeated n-grams | Output loops |
| **Perplexity** | `exp(entropy)`, rolling window | Sustained confusion |
| **Perplexity trend** | Slope over window | Deteriorating generation |
| **Entropy spikes** | Single-token entropy jump | Sudden uncertainty |

### Abort Triggers

```python
class AbortReason(str, Enum):
    NONE = "none"
    HIGH_ENTROPY = "high_entropy"           # Sustained high entropy
    ENTROPY_SPIKE = "entropy_spike"         # Single large jump
    HIGH_REPETITION = "high_repetition"     # Excessive repeating patterns
    RISING_PERPLEXITY = "rising_perplexity" # Trend of increasing perplexity
    RUNAWAY_LENGTH = "runaway_length"       # Exceeds expected length
    COMBINED_SIGNALS = "combined_signals"   # Weighted combination
```

### Tier-Specific Thresholds

Higher-tier models are allowed more tolerance (architects produce inherently more varied output):

| Setting | Worker | Coder | Architect | Ingest |
|---------|--------|-------|-----------|--------|
| `entropy_threshold` | 4.5 | 5.0 | 6.0 | 5.5 |
| `entropy_spike_threshold` | 2.5 | 3.0 | 4.0 | 3.5 |
| `repetition_threshold` | 0.3 | 0.2 | 0.4 | 0.3 |
| `min_tokens_before_abort` | 50 | 100 | 200 | 100 |

The coder tier has the strictest repetition threshold (0.2) because repeated code is always wrong, while the architect tier is most tolerant (0.4) because design discussions naturally revisit concepts.

### Task-Specific Overrides

| Setting | Reasoning | Code | General |
|---------|-----------|------|---------|
| `entropy_threshold` | 4.5 | 4.0 | 4.0 |
| `min_tokens_before_abort` | 30 | 100 | 50 |
| `ngram_size` | 3 | 4 | 3 |
| `perplexity_window` | 15 | 20 | 20 |

Reasoning tasks use a shorter abort window (30 tokens) because failed reasoning is detectable early. Code tasks use 4-gram detection because code patterns repeat in 4-token sequences (e.g., `if err != nil`).

### Combined Failure Probability

The monitor computes a weighted failure score (0.0-1.0):

```python
weights = {
    "entropy": 0.35,      # Most predictive signal
    "spike": 0.25,
    "repetition": 0.25,
    "trend": 0.15,        # Rising/spiking perplexity
}

failure_prob = sum(w * normalize(signal) for w, signal in zip(weights, signals))
# Abort when failure_prob > 0.7 (combined_threshold)
```

Each signal is normalized to 0-1 relative to its threshold. The combined approach prevents false positives from any single noisy signal.

<details>
<summary>Code: MonitorConfig defaults</summary>

```python
@dataclass
class MonitorConfig:
    min_tokens_before_abort: int = 50
    entropy_threshold: float = 4.0
    entropy_spike_threshold: float = 2.0
    repetition_threshold: float = 0.3
    perplexity_window: int = 20
    max_length_multiplier: float = 2.0
    entropy_sustained_count: int = 10
    ngram_size: int = 3
    combined_threshold: float = 0.7
```

</details>

### Integration with Escalation

When the generation monitor aborts, it produces an `EARLY_ABORT` error category:

```python
class ErrorCategory(str, Enum):
    CODE = "code"              # Standard retry -> escalate
    LOGIC = "logic"            # Standard retry -> escalate
    FORMAT = "format"          # Retry only (never escalate)
    SCHEMA = "schema"          # Retry only (never escalate)
    TIMEOUT = "timeout"        # Skip if optional gate
    EARLY_ABORT = "early_abort" # Immediate escalation
```

`EARLY_ABORT` triggers **immediate escalation** -- no retry -- because the abort indicates the current model cannot produce valid output for this input. Retrying would produce the same degenerate output.

</details>

## Path Validation

All file operations go through a whitelist check that resolves symlinks with `realpath()` before comparison, defeating path traversal attacks. This is the primary defense against exhausting the 120GB root SSD, complemented by environment variable redirects and a bind mount for `/tmp/claude`.

<details>
<summary>Whitelist enforcement and root filesystem protection</summary>

### Whitelist Enforcement

All file operations validate paths against a strict whitelist:

```python
ALLOWED_FILE_PATHS = ["/mnt/raid0/llm/", "/tmp/"]

def _validate_file_path(path: str) -> tuple[bool, str | None]:
    resolved = os.path.realpath(path)  # Resolve symlinks
    for allowed in ALLOWED_FILE_PATHS:
        if resolved.startswith(allowed):
            return True, None
    return False, f"Path not in allowed locations"
```

**Why `realpath()`**: Resolves symlinks to their physical location, defeating attacks like:
```bash
ln -s /etc/passwd /mnt/raid0/llm/link
# realpath("/mnt/raid0/llm/link") -> "/etc/passwd" -> BLOCKED
```

### Root Filesystem Protection

The 120GB root SSD can be exhausted by unconstrained writes. Path validation is the primary defense, complemented by:

- Environment variables redirecting all caches to `/mnt/raid0/`
- Bind mount for `/tmp/claude` -> `/mnt/raid0/llm/tmp/claude`
- Storage monitoring scripts (see [Chapter 4](04-storage-and-safety.md))

<details>
<summary>Code: path validation function</summary>

```python
ALLOWED_FILE_PATHS = ["/mnt/raid0/llm/", "/tmp/"]

def _validate_file_path(path: str) -> tuple[bool, str | None]:
    resolved = os.path.realpath(path)  # Resolve symlinks
    for allowed in ALLOWED_FILE_PATHS:
        if resolved.startswith(allowed):
            return True, None
    return False, f"Path not in allowed locations"
```

</details>
</details>

## Execution Flow

When LLM-generated code enters the REPL, it passes through six stages: AST validation, timeout setup, restricted execution, generation monitoring (if the code calls an LLM tool), output capping, and finally result or error handling. An `EARLY_ABORT` from the generation monitor triggers immediate escalation, while a `REPLSecurityError` is logged and blocked outright.

<details>
<summary>End-to-end security flow</summary>

Complete security flow for REPL code execution:

```
LLM generates code
        |
        v
[1. AST Security Validation]
  - Parse AST
  - Check forbidden modules, calls, attrs
  - Reject or pass
        |
        v
[2. Timeout Setup]
  - signal.SIGALRM set
  - Default: 120 seconds
        |
        v
[3. Restricted Execution]
  - exec() in sandboxed globals
  - Trusted tools available (path-validated)
  - stdout/stderr captured
        |
        v
[4. Generation Monitor] (if LLM tool called)
  - Track entropy, repetition, perplexity
  - Abort if combined_threshold > 0.7
        |
        v
[5. Output Capping]
  - Max 100,000 chars (REPL)
  - Max 8,192 chars (RestrictedPython)
  - Truncation with marker
        |
        v
[6. Result or Error]
  - ExecutionResult with stdout, is_final, error
  - EARLY_ABORT -> immediate escalation
  - REPLSecurityError -> logged + blocked
```

</details>

## REPL Configuration

The REPL has five tunable limits that cap execution time, output size, and tool usage per session.

<details>
<summary>Config: REPL limits</summary>

```python
@dataclass
class REPLConfig:
    timeout_seconds: int = 120           # Execution timeout
    output_cap: int = 100_000            # Max output characters
    max_grep_results: int = 100          # Max grep matches returned
    max_exploration_calls: int = 50      # Max tool invocations per session
    max_file_read: int = 50_000          # Max file read size (chars)
```

</details>

## References

<details>
<summary>Project files and related chapters</summary>

### Project Files

- Generation monitor: `src/generation_monitor.py`
- REPL environment: `src/repl_environment.py`
- Restricted executor: `src/restricted_executor.py`
- Tool registry permissions: `src/tool_registry.py`
- Safety reviewer agent: `agents/safety-reviewer.md`

### Related Chapters

1. [Chapter 11: REPL Environment](11-repl-environment.md) -- full REPL architecture and built-in tools
2. [Chapter 18: Escalation & Routing](18-escalation-and-routing.md) -- how EARLY_ABORT triggers escalation
3. [Chapter 22: Tool Registry & Agent Roles](22-tool-registry.md) -- permission model and role definitions
4. [Chapter 4: Storage & Safety](04-storage-and-safety.md) -- root filesystem protection

</details>

---

*Previous: [Chapter 22: Tool Registry & Agent Roles](22-tool-registry.md)* | *Next: [Back to Index](INDEX.md)*
