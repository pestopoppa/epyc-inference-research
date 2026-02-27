# Chapter 12: Production Server Stack

## Introduction

The production server stack runs 9 llama-server instances plus 2 auxiliary services, organized into HOT/WARM/COLD memory tiers. The HOT tier (~535GB, 47% of 1.13TB RAM) stays resident for immediate availability. Worker pools provide heterogeneous parallelism with different models optimized for specific task types.

Managed by `orchestrator_stack.py`, the system provides graceful start/stop, health monitoring, and granular component reload without full restart.

## Server Topology

The stack spans three tiers of servers, each mapped to a port range. The HOT tier holds the models you interact with most — the frontdoor, coders, architects, and embedders — all pinned in RAM so there is zero cold-start penalty. Auxiliary services handle retrieval and OCR on their own ports.

<details>
<summary>Server port assignments and tier breakdown</summary>

### HOT Tier (Always Resident)

| Port | Roles | Model | Acceleration | Speed | RAM |
|------|-------|-------|--------------|-------|-----|
| 8080 | frontdoor | Qwen3-Coder-30B-A3B Q4_K_M | MoE6 | 18 t/s | ~18GB |
| 8081 | coder_escalation, worker_summarize | Qwen2.5-Coder-32B Q4_K_M + 0.5B draft | Spec K=24 + lookup | 39 t/s | ~22GB |
| 8082 | worker_explore, worker_math | Qwen2.5-7B-Instruct f16 + 0.5B draft | Spec K=24 + lookup | 44 t/s | ~14GB |
| 8083 | architect_general | Qwen3-235B-A22B Q4_K_M | MoE4 | 6.75 t/s | ~140GB |
| 8084 | architect_coding | Qwen3-Coder-480B-A35B Q4_K_M | MoE3 | 10.3 t/s | ~280GB |
| 8085 | ingest_long_context | Qwen3-Next-80B-A3B Q4_K_M | MoE4 (NO SPEC!) | 6.3 t/s | ~45GB |
| 8086 | worker_vision | Qwen2.5-VL-7B Q4_K_M + mmproj | None (VL) | ~15 t/s | ~8GB |
| 8087 | vision_escalation | Qwen3-VL-30B-A3B Q4_K_M + mmproj | MoE4 | ~10 t/s | ~20GB |
| 8090-8095 | embedder (6x) | BGE-large-en-v1.5 F16 | probe-first | — | ~4GB |

**Total HOT RAM**: ~538GB (48% of 1130GB), leaving ~592GB for KV cache and OS.

### Auxiliary Services

| Port | Service | Model | Purpose |
|------|---------|-------|---------|
| 8000 | orchestrator API | uvicorn | FastAPI HTTP entrypoint |
| 8088 | nextplaid-code | LateOn-Code (130M, ONNX INT8) | Multi-vector code retrieval (AST-chunked) |
| 8089 | nextplaid-docs | answerai-colbert-small-v1 (ONNX INT8) | Multi-vector doc retrieval |
| 9001 | document_formalizer | LightOnOCR-2-1B | PDF OCR, figure extraction |

### WARM Tier (Load on Demand)

| Port | Role | Model | Purpose |
|------|------|-------|---------|
| 8102 | worker_fast_1 | Qwen2.5-Coder-1.5B Q4_K_M | Burst capacity |
| 8112 | worker_fast_2 | Qwen2.5-Coder-1.5B Q4_K_M | Burst capacity |

**Idle Timeout**: 300 seconds (5 minutes). Automatically shut down if unused.

</details>

## Memory Architecture

About half the system RAM is pinned to HOT-tier models so they never get evicted. The remaining half is split between dynamic KV cache (which grows with concurrent requests) and OS buffers. Larger models like the 235B and 480B architects dominate the budget, but keeping them resident avoids 30-90 second reload penalties that would wreck interactive latency.

<details>
<summary>Tier allocation and model load times</summary>

### Tier Allocation

<details>
<summary>Data: RAM budget breakdown</summary>

```
Total RAM: 1130GB
├── HOT Tier: 535GB (47%) - Always resident
│   ├── Frontdoor: 18GB
│   ├── Coder escalation: 22GB
│   ├── Worker pool: 14GB
│   ├── Architects: 420GB (235B + 480B models)
│   ├── Ingest: 45GB
│   ├── Vision: 28GB
│   ├── Embedder: 1GB
│   └── NextPLAID (2x): ~1.4GB (code: 1.2GB LateOn-Code 130M + docs: 0.2GB colbert-small)
├── KV Cache: ~460GB (41%) - Dynamic allocation
└── OS + Buffers: ~135GB (12%)
```

</details>

**Design Principle**: Keep specialists resident to avoid cold-start latency (15-45s model load). Only WARM tier workers are evicted.

### Model Load Times

| Model Size | Load Time | Strategy |
|------------|-----------|----------|
| 0.5B-1.5B | 2-5s | WARM tier (acceptable cold start) |
| 7B-32B | 10-20s | HOT tier (avoid reload) |
| 80B-235B | 30-60s | HOT tier (critical) |
| 480B | 60-90s | HOT tier (always resident) |

**Optimization**: Parallel tensor repack (`production-consolidated` branch) reduces load time by 2.2x vs sequential.

</details>

## Worker Pool Architecture

Workers are not one-size-fits-all. Different models handle different task types, and the pool expands on demand when concurrent load spikes. The original 7B coder worker was removed after benchmarks proved the 32B coder-escalation endpoint was both faster and higher quality.

<details>
<summary>Worker routing, pool config, and expansion strategy</summary>

### Heterogeneous Parallelism

The worker pool uses different models for different task types:

<details>
<summary>Code: Worker pool model mapping</summary>

```python
WORKER_POOL_MODELS = {
    "explore": "/mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf",
    "fast": "/mnt/raid0/llm/lmstudio/models/.../Qwen2.5-Coder-1.5B.Q4_K_M.gguf",
}

class WorkerTier(Enum):
    HOT = "hot"    # Always resident
    WARM = "warm"  # Load on demand
```

</details>

### Task Routing

| Task Type | Worker | Model | Rationale |
|-----------|--------|-------|-----------|
| explore, summarize, understand | explore (8082) | 7B Instruct + spec decode | Quality for comprehension |
| code_impl, refactor, test_gen | worker_coder (semantic) → fast pool (8102) | Parallel coding bursts with low latency |
| boilerplate, transform | fast_1/fast_2 (8102/8112) | 1.5B WARM | High throughput, simple tasks |

**worker_coder Active (legacy `worker_code` alias retained)**: Coding-worker semantics route to the fast pool on port 8102 for parallel subtask bursts; specialist `coder_escalation` remains available for heavier synthesis/debug tasks.

### Expansion Strategy

<details>
<summary>Code: Worker pool expansion config</summary>

```python
@dataclass
class WorkerPoolConfig:
    expansion_threshold: int = 4  # Concurrent tasks to trigger WARM expansion
    warm_timeout_seconds: int = 300  # 5 min idle before shutdown
```

</details>

When concurrent load exceeds 4 tasks, WARM workers spin up. After 5 minutes idle, they shut down to free RAM.

</details>

## CLI Operations

You manage the whole stack through `orchestrator_stack.py`. It supports dev mode for quick iteration with a single tiny model, production hot-only for the full resident tier, and granular reload so you can swap one component without bouncing everything.

<details>
<summary>Start, stop, reload, and state persistence commands</summary>

### Start Commands

<details>
<summary>Code: orchestrator_stack.py usage</summary>

```bash
# Development mode (single 0.5B model, fast startup)
python3 scripts/server/orchestrator_stack.py start --dev

# Production HOT tier only (~535GB RAM)
python3 scripts/server/orchestrator_stack.py start --hot-only

# Production with specific WARM tier models
python3 scripts/server/orchestrator_stack.py start --include-warm architect_general

# Check status
python3 scripts/server/orchestrator_stack.py status

# Stop all
python3 scripts/server/orchestrator_stack.py stop --all

# Reload specific component after code changes
python3 scripts/server/orchestrator_stack.py reload orchestrator
```

</details>

### State Persistence

<details>
<summary>Config: orchestrator_state.json schema</summary>

```json
// /mnt/raid0/llm/epyc-orchestrator/logs/orchestrator_state.json
{
  "server_8080": {
    "role": "frontdoor",
    "pid": 12345,
    "port": 8080,
    "started_at": "2026-01-28T10:30:00",
    "model_path": "/mnt/raid0/llm/models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf",
    "log_file": "/mnt/raid0/llm/epyc-orchestrator/logs/llama-server-8080.log"
  }
}
```

</details>

State enables graceful shutdown and status queries without querying each server.

</details>

## Health Monitoring

Every server exposes a `/health` endpoint that the stack polls during startup and ongoing operation. The startup sequence is deliberately sequential with cooldown gaps between large models so mmap has time to settle. Vision servers get extra-long timeouts because they load both a main model and a multimodal projector.

<details>
<summary>Liveness check implementation and status output</summary>

### Liveness Checks

<details>
<summary>Code: Health polling loop</summary>

```python
def wait_for_health(port: int, timeout: int = 120) -> bool:
    """Wait for server health endpoint."""
    url = f"http://localhost:{port}/health"
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    return False
```

</details>

**Startup Sequence**: Servers start sequentially with 5s cooldown between large models (allow mmap to settle). Vision servers get 90-120s timeout (mmproj + main model).

### Status Output

<details>
<summary>Data: Example status table</summary>

```
COMPONENT                 PORT     PID        STATUS     MODEL
--------------------------------------------------------------------------------
frontdoor                 8080     12345      healthy    Qwen3-Coder-30B-A3B-Q4_K_M
coder_escalation          8081     12346      healthy    Qwen2.5-Coder-32B-Q4_K_M
architect_general         8083     12348      healthy    Qwen3-235B-A22B-Q4_K_M
orchestrator              8000     12350      healthy    uvicorn
```

</details>

</details>

## Initialization Hooks

After all servers are healthy, the stack initializes MemRL databases, seeds the REPL with examples, warms up the embedding pool, and registers the 41 deterministic tools. This runs automatically — you do not need to trigger it manually.

<details>
<summary>MemRL and tool registry init sequence</summary>

### MemRL and Tool Registry

<details>
<summary>Code: init_memrl_and_tools()</summary>

```python
def init_memrl_and_tools() -> bool:
    """Initialize MemRL databases and tool registry."""
    # [6] REPL seed examples
    seed_loader.init()

    # Warm up embedding model with test query
    requests.post("http://localhost:8090/embedding",
                  json={"content": "test embedding warmup"})

    # [7] Tool registry (41 deterministic tools)
    from orchestration.tools.executor import get_executor
    executor = get_executor()
    tools = executor.list_tools()
    # categories: math (8), symbolic (6), numerical (4), format (7), ...

    return True
```

</details>

Called automatically after server startup. Ensures episodic memory and tools are ready.

</details>

## Checkpoint Hooks

Self-management procedures can create/restore checkpoints:

<details>
<summary>Code: Checkpoint create and restore</summary>

```python
checkpoint_create("before_model_update", include_state=True)
# ... make changes ...
checkpoint_restore("before_model_update_20260128_103000")
```

</details>

Stored in `/mnt/raid0/llm/epyc-orchestrator/orchestration/checkpoints/`.

## References

<details>
<summary>Implementation, architecture patterns, and related systems</summary>

### Implementation

1. `scripts/server/orchestrator_stack.py`: Stack launcher (1330 lines)
2. `src/services/worker_pool.py`: Worker pool manager (732 lines)
3. `src/registry_loader.py`: Model registry and role resolution

### Architecture Patterns

4. Netflix's Eureka service discovery: https://github.com/Netflix/eureka
5. Kubernetes liveness/readiness probes: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/

### Related Systems

6. Ray Serve (model serving framework): https://docs.ray.io/en/latest/serve/
7. BentoML (ML serving): https://docs.bentoml.org/

</details>

## Concurrent Inference Sweep (2026-02-19)

Benchmarked optimal `-np`/concurrency per model tier using `scripts/benchmark/concurrent_inference_sweep.py` (asyncio + httpx.AsyncClient, 2 warmup + 5 measured batches, incremental CSV output).

**Results**:
| Role | Port | Recommended `-np` | Rationale |
|------|------|--------------------|-----------|
| frontdoor (30B MoE) | 8080 | **2** (was 1) | +121% aggregate TPS, p95 multiplier 1.33 |
| coder (32B dense) | 8081 | 1 (keep) | c=2 rejected: p95 multiplier 1.98 |
| worker (7B) | 8082 | 1 (keep) | c=2+ rejected: p95 multiplier ≥1.505 |
| fast_worker (1.5B) | 8102 | — | Port unavailable during sweep |

**Action taken**: Removed `frontdoor` from `SERIAL_ROLES` in `orchestrator_stack.py` so it starts with `-np 2`.

### SERIAL_ROLES

`SERIAL_ROLES` in `orchestrator_stack.py` forces `-np 1` for roles where concurrent slot contention degrades latency: `coder_escalation`, `worker_summarize`, `architect_general`, `architect_coding`, `ingest_long_context`.

---

*Previous: [Chapter 11: REPL Environment](11-repl-environment.md)* | *Next: [Chapter 13: Data Processing Pipelines](13-data-processing-pipelines.md)*
