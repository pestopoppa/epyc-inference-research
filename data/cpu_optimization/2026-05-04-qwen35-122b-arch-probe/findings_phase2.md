# Qwen3.5-122B-A10B Probe B — Phase 2 Wiring Revalidation (2026-05-04)

## Summary

Production's `2× --numa distribute -t 96` wiring (registry: 4.3 t/s/instance, 8.6 t/s
aggregate) is suboptimal in **both** dimensions vs canonical-recipe alternatives:

- **Latency-optimal**: 1× canonical 96t + c2 = **12.19 t/s** (+184% per-request vs production)
- **Throughput-optimal**: 4× per-NUMA-node 24t + c2 = **16.86 t/s aggregate** (+96% vs production)

The choice depends on query concurrency. For architect_general (slots=1 serial per instance),
**1× canonical is the immediate win**. 4× per-NUMA-node becomes optimal only if slot count
scales beyond 1 or if batch/eval throughput becomes the priority.

## Method

Phase 1 found c2 (`GGML_NUMA_REPACK_INTERLEAVE=0`) wins +1.28% at 96t single-instance
canonical. Phase 2 tests whether the win preserves at half-machine binding and what the
aggregate looks like under concurrent multi-instance.

All Phase 2 runs use the c2 env (`GGML_NUMA_REPACK_INTERLEAVE=0`) plus the canonical OMP
env stack (`OMP_PROC_BIND=spread`, `OMP_PLACES=cores`, `OMP_WAIT_POLICY=active`).

NPS4 topology: each NUMA node has 24 physical cores. 4 nodes × 24 cores = 96 total.

## Results

### w1a — 1× single-NUMA-node-bound (isolated)

```
numactl --cpunodebind=0 --membind=0 -- llama-bench -t 24 -fa 1 --mmap 0
```

**4.207 ± 0.011 t/s** (σ 0.27% — extremely tight)

Half-cores (24/96), single-node BW (~115/460 GB/s) → ~25% of full-machine canonical.
Confirms per-node ceiling: BW-bound at one node's 3 DRAM channels.

### w1b — 2× concurrent per-NUMA-node

```
node 0: --cpunodebind=0 --membind=0 -t 24
node 1: --cpunodebind=1 --membind=1 -t 24
(launched concurrently)
```

| Instance | avg t/s | σ % |
|---|---|---|
| node 0 | 4.194 ± 0.022 | 0.51% |
| node 1 | 4.271 ± 0.004 | 0.10% |
| **aggregate** | | **8.47** |

Linear scaling vs w1a (8.47 ≈ 2 × 4.21). **Confirms production wiring 8.6 t/s aggregate**
(within 1.5% noise of registry 4.30 × 2). The 2026-03-29 wiring is structurally similar to
w1b's per-node binding effect.

### w2 — 4× concurrent per-NUMA-node

```
node 0..3: --cpunodebind=N --membind=N -t 24
(all 4 launched concurrently)
```

| Instance | avg t/s | σ % |
|---|---|---|
| node 0 | 4.152 ± 0.071 | 1.70% |
| node 1 | 4.249 ± 0.008 | 0.18% |
| node 2 | 4.242 ± 0.017 | 0.40% |
| node 3 | 4.220 ± 0.006 | 0.13% |
| **aggregate** | | **16.86** |

**Linear 4× scaling.** Per-instance hits the per-NUMA-node BW ceiling (~4.22 t/s) regardless
of how many other nodes are running concurrent instances. No cross-node contention because
each instance is fully BW-saturated within its own node.

### w3 (single-instance canonical 96t + c2) — already done in Phase 1

```
numactl --interleave=all -- taskset -c 0-95 llama-bench -t 96 -fa 1 --mmap 0
```

**12.19 ± 0.05 t/s**

## Operating-point comparison

| Workload regime | Optimal wiring | t/s | Δ vs production |
|---|---|---|---|
| Single user, 1 request at a time | w3 (1× canonical 96t + c2) | 12.19/req | **+184% per-req** |
| 2 concurrent requests | w3 sequential (2 × 1/12.19s/tok) OR w1b parallel (2 × 4.23) | 12.19 vs 8.47 agg | w3 still wins |
| 4 concurrent requests | w2 (4× per-node 24t + c2) | 16.86 agg | **+96% aggregate** |
| Production (current 2× --numa distribute) | n/a | 8.60 agg | baseline (suboptimal both) |

For architect_general (slots=1, serial per instance), the operating regime is single-user
(rarely 2+ concurrent reasoning tasks). **w3 is the immediate win**: drop from 2 instances
to 1 instance, gain +184% per-request latency.

If the orchestrator scales architect_general to high-concurrency batch eval workloads,
w2 (4× per-node) becomes optimal, +96% aggregate over production.

## Recommendation

### Phase 1 of wiring change — IMMEDIATE (single line, low-risk)

Switch architect_general from production `2× --numa distribute -t 96` to **1× canonical 96t +
c2 env**. Expected: **+184% per-request latency** (4.3 → 12.19 t/s), at the cost of dropping
from 2 concurrent instances to 1. Acceptable for the single-user-style query pattern.

### Phase 2 of wiring change — CONDITIONAL (multi-instance)

Only if architect_general workload shifts to high concurrency (eval batches, multi-user
scenarios). In that regime, switch to **4× per-NUMA-node 24t + c2 env**. Expected: **+96%
aggregate** (8.6 → 16.86 t/s) at the cost of slightly worse per-request latency (4.3 → 4.22,
within noise).

### NOT recommended

- Stay on production `2× --numa distribute`: leaves both per-request AND aggregate on the
  table. There is no workload where this is the optimal wiring.
- Don't go to 4× without measuring: if the orchestrator can't actually saturate 4 concurrent
  architect_general requests, the per-instance throughput drops are pure loss.

## Implementation deltas

### orchestrator stack registry (per-role wiring change, separate handoff)

This requires changing `numa_instances`, `numa_ports`, and the launch invocation per the
orchestrator stack registry pattern. NOT in this session's scope.

Specifically the architect_general entry's:
```yaml
numa_instances: 2  # 2×96t cross-NUMA (2026-03-29)
numa_ports: [8083, 8183]
throughput: 4.3    # t/s per instance at 96t
```
becomes (Phase 1 latency-optimal):
```yaml
numa_instances: 1  # 1×96t canonical + c2 mbind off (2026-05-04)
numa_ports: [8083]
throughput: 12.19  # t/s single instance, c2 env
```

### per-role env block (this session, already applied)

`_ROLE_ENV_BLOCKS["architect_general"]` in orchestrator_stack.py already populated with
`GGML_NUMA_REPACK_INTERLEAVE=0`. This is the "free win" — it applies whether wiring is 1×, 2×,
or 4×.

## Bundle contents

- `c0_default_v5.json`, `c1_cpu1_stack.json`, `c2_mbind_off.json`, `c3_cpu1_plus_mbind.json`
  — Phase 1 single-instance probe results
- `w1a_node0.json` — Phase 2 single-instance node-pinned
- `w1b_node0.json`, `w1b_node1.json` — Phase 2 2× concurrent
- `w2_node0.json`, `w2_node1.json`, `w2_node2.json`, `w2_node3.json` — Phase 2 4× concurrent
- `phase2_summary.tsv`, `phase2_progress.log` — Phase 2 raw timeline
- `findings.md` — Phase 1 writeup
- `findings_phase2.md` — this file
