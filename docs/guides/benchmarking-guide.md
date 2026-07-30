# Benchmarking Guide

How to verify a model on this system and feed its measurements into the production stack.

> **Repository map**: This guide is for `epyc-inference-research`. Production server / orchestrator code lives in `epyc-orchestrator`. The custom llama.cpp fork lives at `/mnt/raid0/llm/llama.cpp` (binaries: `build/bin/llama-bench`, `llama-server`, `llama-cli`, etc.). All three repos sit on the same host under `/mnt/raid0/llm/`.

## Before You Start

1. Ensure the GGUF is downloaded to `/mnt/raid0/llm/models/` (or `/mnt/raid0/llm/lmstudio/models/` per `runtime_defaults.model_base_path` in the master registry).
2. Verify llama.cpp is built and on the production branch: `bash /workspace/repos/epyc-root/scripts/session/verify_llama_cpp.sh`.
3. Skim [QUIRKS.md](../reference/models/QUIRKS.md) for known issues with the target architecture (gemma-4 idle-spin, Qwen3.5 hybrid lookup segfaults, BOS-comma vocab transplants, etc.).
4. **Confirm nothing else is benchmarking**. Per project policy (`feedback_no_concurrent_inference`), never launch llama-bench / cli / server / perplexity on the EPYC host without explicit approval — a parallel agent's measurements will be silently poisoned.
5. **Check host health before trusting any number**. After ≥1 week uptime the host needs a reboot; under 1 week run `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`, then re-warm the model with `numactl --interleave=all` (a non-NUMA-aware re-read pins the whole file to one node and halves throughput).

## Two Tracks

| Track | Purpose | Entry point |
|-------|---------|-------------|
| **Speed verification** | Confirm raw decode throughput on a fresh GGUF or kernel change | `llama-bench` (standalone) |
| **Quality / routing seeding** | Score model on the production suites, populate MemRL Q-values, gate orchestration roles | `seed_specialist_routing.py` against a running orchestrator stack |

Use speed verification when you only need a t/s number. Use seeding when you intend to deploy the model into a routing role.

## Track 1: Speed Verification

Standalone `llama-bench` is the only inference path you should run manually for speed checks (per `feedback_speed_verify_via_llama_bench`). `run_benchmark.py` is gated to user-only execution.

### Canonical Baseline (CPU)

The agreed-on canonical baseline is:

```bash
taskset -c 0-95 \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-bench \
  -m /mnt/raid0/llm/models/<MODEL>.gguf \
  -t 96 \
  -fa 1 \
  -p 0 \
  -n 128
```

Rules:

- **No** `--numa distribute`, **no** OMP env vars, **no** numactl wrapper. Plain `taskset -c 0-95 -t 96 -fa 1`. Aggregate reference bandwidth on the EPYC 9655: ~460 GB/s.
- **`-fa 1` is not optional.** `llama-bench` defaults to `-fa 0` (perplexity tooling defaults to `-fa auto`). Forgetting `-fa 1` silently costs ~8–10% on CPU decode.
- Run `pgrep -a llama` first to confirm no zombie processes are running. Concurrent inference contaminates all parties' numbers.

### OMP Env Stack (when needed)

Some scenarios — production worker-pool launches, gemma-4 MTP, post-reboot warm-up — require the full OMP stack or the model runs 3–4× degraded. The canonical post-reboot working set is:

```bash
KMP_BLOCKTIME=10 \
OMP_PROC_BIND=spread \
OMP_PLACES=cores \
OMP_WAIT_POLICY=active \
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-bench ...
```

`KMP_BLOCKTIME=10` is the workaround for the AOCC libomp idle-spin issue (`omp_pause_resource` is ignored). Without it, gemma-4 idle cores stay at ~95% utilization and decode regresses ~78% on the frontdoor port.

### When to Use Full-Machine vs Aggregate

> **Corrected 2026-07-30.** This section was headed "Single-NUMA-Node vs Aggregate" and described
> the 96-thread arm as "96t-single-NUMA-node". That is a misnomer: `taskset -c 0-95` is **all 96
> physical cores across all four NPS4 nodes**. A NUMA node on this host is 24 physical cores
> (`node0 = 0-23,96-119`, `node1 = 24-47,120-143`, `node2 = 48-71,144-167`,
> `node3 = 72-95,168-191`) — i.e. exactly one quarter. The `stack_numa.py` names
> `NUMA_NODE0 = "0-47,96-143"` / `NUMA_NODE1 = "48-95,144-191"` are NPS2-era and each span **two**
> nodes; only `NUMA_Q*` is node-aligned.

The orchestrator runs frontdoor as 4×48t NUMA-quarter instances (`numa_ports: [8080, 8180, 8280, 8380]`). Two distinct operating points exist:

- **Single-model, single-instance (192t or 96t)**: useful for measuring per-instance peak; e.g. Qwen3-Coder-30B-A3B reaches ~49 t/s at 96 threads on the whole machine (`taskset -c 0-95`). Canonical placement pairs this with `numactl --interleave=all`.
- **Concurrent split (4×48t or 32×6t)**: the production deployment mode; aggregate throughput is 1.4–1.6× the sum of independent 48t runs at the same model. **Note (observation-grade, 2026-07-30):** at *matched total concurrency* a single full-machine instance out-aggregates four quarters at every measured rung (T=4 79.7 vs 52.9, T=8 105.1 vs 81.0, T=16 131.0 vs 108.4, T=32 145.9 vs 143.8 tok/s). Protocol `P-BENCH-PLACEMENT-1` ([numa-placement-measurement-protocol](../protocols/numa-placement-measurement-protocol.md)) has a MEASUREMENT.md registry entry that is **STAGED, not applied**, so this may not gate a keep/revert/deploy/promote decision.

Index benchmarks by **model name + quantization**, never by orchestrator role (`feedback_model_not_role_indexing`). Role reassignment otherwise destroys historical data.

## Track 2: Quality / Routing Seeding

Production benchmarking runs the model behind the full orchestrator stack and scores it against the curated YAML suites (deterministic) plus the HuggingFace-backed adapters described in [Chapter 06](../chapters/06-benchmarking-framework.md) and [Chapter 07](../chapters/07-benchmark-suite-construction.md).

### Step 1: Launch the Stack

The stack manager lives in `epyc-orchestrator`:

```bash
python3 /mnt/raid0/llm/epyc-orchestrator/scripts/server/orchestrator_stack.py start --hot-only
```

`--hot-only` brings up frontdoor, escalation, worker_general, architect_general (now HOT after the 2026-05 swap to Qwen3.5-122B-A10B), worker_vision, and supporting embed/OCR services. Drop the flag to add WARM (architect_coding, ingest_long_context).

**Critical for Qwen3.x routes**: any role backed by Qwen3.6 (frontdoor) or Qwen3.5-122B (architect_general) MUST pass `chat_template_kwargs.enable_thinking=false`. Defaulting to thinking on causes degenerate `<think>` loops with empty content (+33pp accuracy when forced off on frontdoor empirically). Qwen3-Next-80B (ingest) is the exception — needs thinking ON.

Roles backed by the same GGUF MUST share **one** llama-server process; routing happens in software via the role→port map (`feedback_same_model_roles_share_server`).

### Step 2: Smoke-Test the API

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "frontdoor", "prompt": "Hello", "max_tokens": 10}'
```

If the API responds but a backend probe (e.g. on :8071 or :8084) reports `unhealthy`, the role-port may still be loading — wait for `/health` to clear before benchmarking. Per `project_stack_consolidation_2026_05`, do not read backend_probes failures on consolidated ports as "still loading" if the parent process is up.

### Step 3: Run Seeding

The 3-way MemRL seeding pipeline:

```bash
python3 /mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/seed_specialist_routing.py \
  --3way \
  --continuous
```

Results land incrementally under `/mnt/raid0/llm/epyc-orchestrator/benchmarks/results/orchestrator/` (incremental persistence is mandatory — `feedback_incremental_persistence`).

### Step 4: Run a Specific Suite or Comparison

```bash
# Single-suite orchestrator-vs-direct comparison
python3 /mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/compare_orchestrator_direct.py \
  --suite thinking --use-baseline

# Tier-balanced sampling
python3 /mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/compare_orchestrator_direct.py \
  --debug --suite all --stratify-tiers
```

`--restart-api` rebounces only the uvicorn API on :8000 (handy after Python code changes; does not touch the llama-server backends).

### Step 5: Document Results

1. Add to the master benchmark results file: [`../reference/benchmarks/RESULTS.md`](../reference/benchmarks/RESULTS.md).
2. Update `orchestration/model_registry.yaml` for the relevant role with measured `throughput`, any new `acceleration` parameters, and a dated note. The lean orchestrator registry is compiled from this master at stack-launch time — do not hand-edit it.

## Common Issues

### Model Hangs (CLI)

Use these flags to prevent interactive mode in `llama-cli`:

```bash
llama-cli -m MODEL.gguf -f prompt.txt -n 128 \
    --no-display-prompt \
    --simple-io \
    --no-warmup \
    --temp 0
```

NEVER pipe `llama-cli` output through `grep`/`tail`/`head` — redirect to a file then `cat` (`feedback_never_pipe_llama_output`).

### Low Acceptance Rate (Speculative Decoding)

- Confirm draft–target vocabulary compatibility (`/draft-compat`).
- Current production sweep values live in `model_registry.yaml`. For example, `coder_escalation` uses `draft_max: 32, p_split: 0.05` (tree, sweep-verified 2026-03-21), while `worker` (Qwen3-Coder-30B-A3B) uses `draft_max: 8, p_split: 0` (linear, tree net-negative at 48t).
- Tree spec is HARMFUL on Qwen3-Coder-480B (`-19%`) — use linear only.

### Garbage Output with MoE

Too few experts. Current production frontdoor (Qwen3.5-35B) uses `moe6`, reduced from the GGUF default of 8. Reducing further on Qwen3.5 hybrids triggers `--lookup` segfault (PR #13194), so lookup is disabled on that route.

For pure MoE models (Qwen3-Coder-30B-A3B, Qwen3-Coder-480B, REAP-25B): MoE4–MoE6 is typically safe. Below MoE3 quality usually collapses. Verify empirically with a thinking + coder suite pass before deploying.

### gemma-4 MTP Wedge

If gemma-4 MTP crashes on `GGML_ASSERT(S>0)` in `iqk_fa_templates.h` and the process becomes unkillable (listener closed, 0 conns, ~289 threads spinning ~80 cores), SIGKILL is required (`feedback_gemma4_mtp_fa_assert_wedge`). Detect via no-listen + no-conn + stale-log + high-CPU.

### Multi-day Throughput Regression

If a previously-fast configuration now benches slow, suspect host throttling before suspecting your change. Tiered fix (`feedback_host_throttle_check`):

- Uptime ≤1 week: `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`, then re-warm with `numactl --interleave=all`.
- Uptime ≥1 week: reboot. drop_caches is insufficient (confirmed at 6d 18h).

After drop_caches, **always** re-warm with `numactl --interleave=all` (or restart the server). A non-NUMA-aware re-read pins the entire file to one node and halves t/s.

## The Benchmark Suites

The current suite portfolio is documented in detail in [Chapter 06](../chapters/06-benchmarking-framework.md). Quick summary:

| Suite | Tests | Key For |
|-------|-------|---------|
| Thinking | Reasoning, logic | Oracle / architect roles |
| Coder | Code generation | Coder / escalation roles |
| Math | Mathematical proofs | Math workers |
| General | Instruction following | General workers |
| Agentic | Tool calling | Orchestrator / frontdoor |
| VL | Vision-language | Vision workers |
| Long Context | 4K–50K retrieval | Ingestion |
| Instruction Precision | Format compliance | **Critical for orchestration** — <70% disqualifies a model from frontdoor / dispatcher |
| Web Research | Multi-source web search | frontdoor, any tool-using role |
| Skill Transfer | Cross-domain skill transfer | SkillBank validation |
| Mode Advantage | Routing-signal–producing tasks | MemRL reward shaping |

Plus 15+ HuggingFace-backed adapters (MMLU, GSM8K, IFEval, GAIA, CRUXEval, BigCodeBench, PHYBench, PhysReason, …). See [Chapter 07](../chapters/07-benchmark-suite-construction.md) for the adapter list and scoring contracts.

## Results Location

```
# In epyc-orchestrator (runtime artifacts)
benchmarks/results/orchestrator/
├── comparison_{suite}_{timestamp}.json
└── run_{timestamp}.json

# In epyc-inference-research (curated benchmark data + indices)
benchmarks/results/
├── runs/
├── reviews/
└── index.jsonl
```

## See Also

- [Chapter 06: Benchmarking Framework](../chapters/06-benchmarking-framework.md) — Suites, scoring methodology, hardening history.
- [Chapter 07: Benchmark Suite Construction](../chapters/07-benchmark-suite-construction.md) — How to build new suites; scoring methods reference.
- [Chapter 08: Cost-Aware Rewards](../chapters/08-cost-aware-rewards.md) — How benchmark throughput feeds the routing reward function.
- [Master Benchmark Results](../reference/benchmarks/RESULTS.md) — Per-model scores and speeds.
- [Model Manifest](../MODEL_MANIFEST.md) — Current per-role model assignments and ports.
