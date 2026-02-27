# Chapter 22: Tool Registry & Permission Model

## Introduction

The orchestration system defines **40+ callable tools** across 8 categories with role-scoped permissions. Tools are declared in YAML (`orchestration/tool_registry.yaml`) and enforced at runtime by `src/tool_registry.py`. Each local orchestrator role (frontdoor, coder, architect, worker, etc.) receives scoped tool access via allow/deny lists.

This chapter covers the tool inventory, permission model, and invocation patterns.

## Tool Registry

The registry organizes over 40 tools into eight categories, each declared in `orchestration/tool_registry.yaml`. Every tool has a name, description, and a permission scope that determines which roles can invoke it. The categories span web access, data transformation, code execution, math (both Python and native C++), system I/O, archive handling, and LLM-powered operations.

<details>
<summary>Complete tool inventory by category</summary>

### Tool Categories (40+ Tools)

All tools are defined in `orchestration/tool_registry.yaml`:

#### Web Tools (4)

| Tool | Description | Permission |
|------|-------------|------------|
| `http_get` | Fetch content via HTTP GET | `network` |
| `http_post` | Send data via HTTP POST | `network` |
| `web_search` | DuckDuckGo search (no API key) | `network` |
| `fetch_wikipedia` | Wikipedia article summary | `network` |

#### Data Tools (9)

| Tool | Description | Permission |
|------|-------------|------------|
| `json_query` | JMESPath queries on JSON | none |
| `csv_to_json` | CSV to JSON array conversion | none |
| `json_to_csv` | JSON array to CSV conversion | none |
| `sql_query` | In-memory SQLite queries | none |
| `plot_braille` | Terminal braille character plots (C++) | `compute` |
| `plot_function` | Math function plotting (sin, cos, exp) | `compute` |
| `histogram` | Histogram from data (C++) | `compute` |
| `plot_sixel` | High-resolution sixel graphics (C++) | `compute` |
| `render_math` | LaTeX to Unicode/ASCII rendering (C++) | `compute` |

#### Code Tools (4)

| Tool | Description | Permission |
|------|-------------|------------|
| `python_eval` | Safe Python expression evaluation | `compute` |
| `run_shell` | Sandboxed shell command execution | `shell` |
| `git_status` | Git repository status | `filesystem` |
| `lint_python` | Python linting with ruff | `compute` |

#### Math Tools — Python (4)

| Tool | Description | Permission |
|------|-------------|------------|
| `calculate` | Math expression evaluation (numpy) | `compute` |
| `statistics` | Dataset statistics (mean, std, min, max, median) | `compute` |
| `monte_carlo` | Monte Carlo simulation | `compute` |
| `symbolic_solve` | Symbolic equation solving (SymPy) | `compute` |

#### Math Tools — C++ Native (8)

All use the compiled `llama-math-tools` binary with Eigen and Boost:

| Tool | Description | Library |
|------|-------------|---------|
| `matrix_solve` | Solve Ax=b with QR decomposition | Eigen |
| `matrix_eigenvalues` | Eigenvalues/eigenvectors | Eigen |
| `matrix_svd` | Singular Value Decomposition | Eigen |
| `solve_ode` | ODE solver (RK45) | Boost.Odeint |
| `optimize` | Nelder-Mead minimization | Custom |
| `monte_carlo_native` | OpenMP Monte Carlo | Custom |
| `mcmc` | Metropolis-Hastings MCMC sampler | Custom |
| `bayesopt` | Bayesian optimization (GP) | Custom |

#### System Tools (3)

| Tool | Description | Permission |
|------|-------------|------------|
| `read_file` | File contents (max 1MB) | `filesystem` |
| `write_file` | Write/append to file | `filesystem` |
| `list_directory` | Directory listing with glob | `filesystem` |

#### Archive Tools (4)

| Tool | Description | Permission |
|------|-------------|------------|
| `archive_open` | Open archive manifest (zip, tar, 7z) | `filesystem` |
| `archive_extract` | Extract files from archive | `filesystem` |
| `archive_file` | Get specific file from archive | `filesystem` |
| `archive_search` | Search archive contents | `filesystem` |

#### LLM Tools (3)

| Tool | Description | Permission |
|------|-------------|------------|
| `embed_text` | Generate text embedding | `compute` |
| `similarity_search` | Find similar items by embedding | `compute` |
| `classify_text` | Classify text into categories | `compute` |

</details>

## Side-Effect Declaration (February 2026)

Tools can declare their side effects and whether they are destructive, which lets the graph reason about tool safety without actually executing anything. A `SideEffect` enum captures six categories (local execution, LLM calls, file modification, network access, system state changes, and read-only), while each `Tool` dataclass carries a `side_effects` list and a `destructive` boolean. Destructive tools require explicit approval before running.

<details>
<summary>Side-effect enum, dataclass fields, and YAML declaration</summary>

### SideEffect Enum

<details>
<summary>Code: SideEffect enum definition</summary>

```python
class SideEffect(str, Enum):
    LOCAL_EXEC = "local_exec"        # Executes code locally
    CALLS_LLM = "calls_llm"          # Makes LLM API call
    MODIFIES_FILES = "modifies_files" # Writes to filesystem
    NETWORK_ACCESS = "network_access" # Makes network requests
    SYSTEM_STATE = "system_state"     # Modifies system state
    READ_ONLY = "read_only"           # No side effects
```

</details>

### Tool Dataclass Fields

<details>
<summary>Code: Tool dataclass with side-effect fields</summary>

```python
@dataclass
class Tool:
    name: str
    description: str
    category: ToolCategory
    parameters: dict
    # ... existing fields ...
    side_effects: list[str] = field(default_factory=list)  # SideEffect values
    destructive: bool = False  # Requires approval when True
```

</details>

### YAML Declaration

<details>
<summary>Config: side-effect declaration in tool_registry.yaml</summary>

```yaml
tools:
  - name: run_shell
    description: Sandboxed shell command execution
    category: code
    side_effects: [local_exec, modifies_files, system_state]
    destructive: true
```

</details>

Parsed by `load_from_yaml()`. Listed in `list_tools()` output only when non-empty.

Feature flag: `side_effect_tracking`.

</details>

## Structured Tool Output (February 2026)

`ToolOutput` provides a structured envelope for tool results with dual output modes (human-readable text and machine-parseable dict). When the `structured_tool_output` feature flag is enabled, `invoke()` wraps raw results in a `ToolOutput` with declared side effects. If `side_effect_tracking` is also on, destructive tools return a pending-approval status instead of executing immediately.

<details>
<summary>ToolOutput dataclass and behavior details</summary>

### ToolOutput Dataclass

<details>
<summary>Code: ToolOutput dataclass definition</summary>

```python
@dataclass
class ToolOutput:
    protocol_version: int = 1
    ok: bool = True
    status: str = "success"          # "success" | "error" | "pending_approval"
    output: Any = None
    side_effects_declared: list[str] = field(default_factory=list)
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_human(self) -> str: ...     # Human-readable text
    def to_machine(self) -> dict: ...  # Machine-parseable dict
```

</details>

## Programmatic Chaining Controls (February 2026, Phase 2a)

The registry now carries per-tool caller policy for structured-mode chaining:

- `Tool.allowed_callers`: additive caller policy (default `["direct"]`).
- `ToolRegistry.get_chainable_tools()`: returns tools that opt in with `"chain"`.
- `ToolInvocation` audit enrichment:
  - `caller_type` (`direct` or `chain`)
  - `chain_id` (shared across one multi-tool chain)
  - `chain_index` (step order within that chain)

Runtime behavior:

- Structured-mode multi-tool turns are allowed only when non-read-only calls are chain-eligible.
- Routing tools (`delegate`, `escalate`) remain blocked from chaining.
- Chat responses expose grouped chain diagnostics via `tool_chains`, including execution metadata:
  - `mode_requested`, `mode_used`, `fallback_to_seq`
  - `parallel_mutations_enabled`, `waves`, `steps`

### Behavior

- When `structured_tool_output` enabled: `invoke()` wraps raw results in `ToolOutput` with `ok=True`, includes `side_effects_declared` from tool definition.
- When `side_effect_tracking` also enabled: destructive tools return `ToolOutput(status="pending_approval", requires_approval=True)` instead of executing.
- Errors wrapped as `ToolOutput(ok=False, status="error")` instead of raising.
- `ToolOutput` slots into existing `ToolInvocation.result` field (type `Any`).

Feature flags: `structured_tool_output`, `side_effect_tracking`.

</details>

## Cascading Tool Policy (February 2026)

When the `cascading_tool_policy` feature flag is enabled, the flat `ToolPermissions` model is replaced by a layered policy chain: **Global → Role → Task → Delegation**. Each layer can only narrow (never expand) the allowed tool set, and deny always wins.

<details>
<summary>PolicyLayer, TOOL_GROUPS, and chain resolution</summary>

### Policy Chain

Implemented in `src/tool_policy.py`:

- `PolicyLayer(name, allow, deny)` — frozen dataclass. Allow intersects, deny removes.
- `TOOL_GROUPS` — group: prefixes expand to tool sets (`group:read`, `group:write`, `group:code`, `group:data`, `group:web`, `group:math`, `group:all`).
- `resolve_policy_chain(layers, all_tools)` — iterates layers, narrowing the allowed set.
- `permissions_to_policy(name, perms, all_tools)` — backward-compat adapter from `ToolPermissions`.

### Context-Aware Access

`ToolRegistry.can_use_tool()` now accepts an optional `context` dict:

```python
registry.can_use_tool("worker", "write_file", context={"read_only": True})  # False
registry.can_use_tool("frontdoor", "web_fetch", context={"no_web": True})    # False
```

Task-level constraints are injected as additional policy layers at the end of the chain.

### Layer Sources

| Layer | Source | Example |
|-------|--------|---------|
| Global | `registry.add_global_policy()` | Deny `raw_exec` for all roles |
| Role | `registry.add_role_policy()` or adapted from `ToolPermissions` | Workers get `group:read` only |
| Task | `context` parameter | `read_only=True` denies `group:write` |
| Delegation | Runtime injection | Architect narrows worker access |

Feature flag: `cascading_tool_policy` (default: False).

</details>

## Permission Model

Permissions control which roles can invoke which tools. There are four permission types (network, filesystem, shell, compute), and enforcement follows a deny-first strategy: if a tool appears on the `forbidden_tools` list, it is blocked regardless of category. Filesystem tools also validate paths against a whitelist and resolve symlinks to prevent escape attempts.

<details>
<summary>Permission types, enforcement logic, and path validation</summary>

### Permission Types

<details>
<summary>Config: permission type definitions</summary>

```yaml
permissions:
  network:
    description: Can make network requests
    requires_approval: false
  filesystem:
    description: Can read/write files
    requires_approval: true
    allowed_paths: ["/mnt/raid0/llm/", "/tmp/"]
  shell:
    description: Can execute shell commands
    requires_approval: true
  compute:
    description: Can execute computation
    requires_approval: false
```

</details>

### Enforcement

Implemented in `src/tool_registry.py`:

<details>
<summary>Code: ToolPermissions class and can_use_tool logic</summary>

```python
class ToolPermissions:
    web_access: bool
    allowed_categories: list[ToolCategory]
    allowed_tools: list[str]       # Explicit allow list
    forbidden_tools: list[str]     # Explicit deny list

def can_use_tool(self, tool: Tool) -> bool:
    # 1. Check forbidden list (deny wins)
    # 2. Check explicit allow list
    # 3. Check category + web_access flag
```

</details>

The deny list takes priority: a tool on the `forbidden_tools` list is blocked even if its category is otherwise allowed.

### Path Validation

All filesystem tools validate paths against the whitelist:

<details>
<summary>Code: path validation with symlink resolution</summary>

```python
ALLOWED_FILE_PATHS = ["/mnt/raid0/llm/", "/tmp/"]

def _validate_file_path(path: str) -> bool:
    resolved = os.path.realpath(path)  # Resolve symlinks
    return any(resolved.startswith(p) for p in ALLOWED_FILE_PATHS)
```

</details>

Uses `os.path.realpath()` to defeat symlink-based escape attempts.

</details>

## Tool Invocation Pattern

This is the standard way to check permissions and invoke a tool at runtime. Load the registry from YAML, gate on `can_use_tool`, then call `invoke` with the role and arguments.

<details>
<summary>Code: basic tool invocation example</summary>

```python
from src.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.load_from_yaml("orchestration/tool_registry.yaml")

if registry.can_use_tool("frontdoor", "fetch_docs"):
    result = registry.invoke("fetch_docs", role="frontdoor", url="...")
```

</details>

## Plugin Architecture

Beyond the static YAML registry, tools can be loaded dynamically through a plugin system managed by `src/tool_loader.py`. Each plugin lives in its own directory under `src/tools/` with a `manifest.json` declaring metadata, tool definitions, dependencies, and a settings schema. The loader supports hot reload so you can update plugins without restarting the server.

<details>
<summary>Plugin manifest, discovery, hot reload, and current plugins</summary>

### Manifest Schema

<details>
<summary>Config: example manifest.json for a plugin</summary>

```json
{
  "schema_version": "1.0",
  "name": "canvas_tools",
  "version": "1.0.0",
  "description": "Canvas tools for reasoning visualization",
  "enabled": true,
  "dependencies": ["kuzu"],
  "settings_schema": {
    "canvas_directory": {"type": "string", "default": "/mnt/raid0/llm/epyc-orchestrator/logs/canvases"}
  },
  "tools": [
    {
      "name": "export_reasoning_canvas",
      "description": "Export hypothesis/failure graphs to JSON Canvas",
      "module": "src.tools.canvas_tools",
      "function": "export_reasoning_canvas",
      "category": "data",
      "parameters": {
        "graph_type": {"type": "string", "required": false}
      }
    }
  ]
}
```

</details>

### Plugin Discovery

<details>
<summary>Code: discovering and listing plugins</summary>

```python
from src.tool_loader import ToolPluginLoader

loader = ToolPluginLoader()
count = loader.discover_plugins(Path("src/tools"))  # Scans for manifest.json files

tools = loader.list_tools(enabled_only=True)
# Returns: [{"name": "export_reasoning_canvas", "plugin": "canvas_tools", ...}, ...]
```

</details>

### Hot Reload

Plugins can be reloaded without restarting the server:

<details>
<summary>Code: hot-reloading changed plugins</summary>

```python
changed = loader.check_for_changes()  # Returns list of modified plugins
count = loader.reload_changed()        # Reloads them
```

</details>

Via MCP: `reload_plugins()` tool.

### Current Plugins

| Plugin | Tools | Description |
|--------|-------|-------------|
| `web` | `fetch_docs`, `web_search` | Web content retrieval |
| `file` | `read_file`, `list_dir` | File system operations |
| `code` | `run_tests`, `lint_code` | Code quality tools |
| `data` | `json_parse` | Data transformation |
| `canvas_tools` | `export_reasoning_canvas`, `import_canvas_edits`, `list_canvases` | JSON Canvas integration |

### Per-Tool Settings

User-specific settings stored in `src/tool_settings/{plugin_name}.json` (gitignored):

<details>
<summary>Config: per-tool settings override example</summary>

```json
{
  "enabled": true,
  "tool_overrides": {
    "export_reasoning_canvas": {"enabled": false}
  },
  "custom_config": {"canvas_directory": "/custom/path"}
}
```

</details>

</details>

## References

<details>
<summary>Project files and related chapters</summary>

### Project Files

- Tool definitions: `orchestration/tool_registry.yaml`
- Python implementation: `src/tool_registry.py`
- Plugin loader: `src/tool_loader.py`
- Plugin manifests: `src/tools/*/manifest.json`

### Related Chapters

1. [Chapter 10: Orchestration Architecture](10-orchestration-architecture.md) — TaskIR and agent tiers
2. [Chapter 18: Escalation & Routing](18-escalation-and-routing.md) — how tools route between agents
3. [Chapter 23: Security & Monitoring](23-security-and-monitoring.md) — runtime security enforcement

</details>

---

*Previous: [Chapter 21: Benchmarking Framework](21-benchmarking-framework.md)* | *Next: [Chapter 23: Security & Monitoring](23-security-and-monitoring.md)*
