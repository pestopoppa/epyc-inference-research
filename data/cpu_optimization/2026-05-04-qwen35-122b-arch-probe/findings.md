# Qwen3.5-122B-A10B Arch-Class Probe B — Findings (2026-05-04)

## Closes the architect_general slot in v5 deployment draft

Per `handoffs/active/qwen35-122b-a10b-arch-class-probe.md`. Probe B methodology, n=5 reps,
canonical recipe (numactl --interleave=all + taskset 0-95 + OMP env stack + --mmap 0 + -fa 1).

## Tripwire

```
Coder-30B Q4_K_M tg32 r=5: 47.86 ± 0.36 t/s (canonical band 47-49 t/s) ✅
```

## Phase 1 — Single-instance c0/c1/c2/c3 probe (96t)

| Config | Env block | avg t/s | σ t/s | σ % | Δ vs c0 |
|---|---|---|---|---|---|
| **c0** default v5 | (none) | 12.041 | 0.037 | 0.31% | baseline |
| **c1** CPU1 stack | `GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1` | 12.065 | 0.024 | 0.20% | +0.21% |
| **c2** mbind off | `GGML_NUMA_REPACK_INTERLEAVE=0` | **12.195** | 0.051 | 0.42% | **+1.28%** |
| **c3** c1+c2 combined | (both) | 12.048 | 0.082 | 0.68% | +0.06% |

**Decision gates from handoff:**
- "any config ≥ +5% with σ ≤ 1%" → none
- "all within ±2% under tight Probe B" → ✅ all configs within ±1.3%, σ all ≤ 1%

**Winner: c2 (`GGML_NUMA_REPACK_INTERLEAVE=0`)**, +1.28% with z-score ~3 (clearly above noise).

## Arch-class assignment

This places Qwen3.5-122B-A10B Q4_K_M in the **MoE Q4 BW-bound, mbind-sensitive** sub-class.
Notably:
- Does NOT fit the Coder-30B-A3B "MoE Q4 sync-bound" class (c1 fails to deliver +1.8%)
- Does NOT fit the REAP-246B "MoE Q4 DRAM-bound, mbind-tolerant" class (c2 helps, doesn't hurt)
- Closest analogue is the Q8 frontdoor (Qwen3.6-35B-A3B Q8) class where mbind-off was +6%

The auto-mbind(MPOL_INTERLEAVE) applied to the CPU_REPACK buffer (per Q8 8x8 NUMA fix
Session 15) appears to mildly hurt this model. With mbind disabled, weights spread via
`numactl --interleave=all` first-touch (still distributed but via the kernel's
default-interleave path, not the explicit per-buffer mbind).

## The MUCH bigger finding — production wiring underperforms by ~2.8×

Per orchestrator stack registry (line 416-422), production runs:
- `architect_general`: 2× `numa distribute` -t 96 cross-NUMA, throughput **4.3 t/s/instance**

Canonical single-instance (96t, c2 winning env) measured here: **12.19 t/s**.

That's a **+184% per-instance speedup** sitting unused. The 2026-03-29 cross-NUMA wiring
predates the v5 audit + auto-mbind landings + canonical recipe stabilization. It needs
revisiting.

| Wiring | Per-instance | Concurrent capacity | Aggregate |
|---|---|---|---|
| Production now: 2× `--numa distribute` -t 96 | 4.3 t/s | 2 | 8.6 t/s |
| Canonical 1× -t 96 + c2 (this measurement) | 12.19 t/s | 1 | 12.19 t/s |
| Canonical 2× per-NUMA-node-pinned (untested w1) | ? | 2 | ? |

For the architect_general role (slots=1, serial per instance, single-user-style queries),
**per-instance latency dominates** — switching to canonical 1×96t + c2 would be a strict
~2.8× latency win at the cost of dropping from 2 concurrent slots to 1.

## Phase 2 — Bonus wiring revalidation (w0..w3) — DEFERRED

The handoff's optional w0..w3 probes test concurrent NUMA wiring patterns. Skipping for
this session because:

1. RAM headroom for w2 (4× 70 GB = 280 GB) is tight; needs explicit `numactl -H` audit
   and concurrent process coordination
2. Concurrent runs introduce timing-correlation issues per `feedback_no_concurrent_inference`
3. The single-instance c0..c3 result is already actionable for the v5 deployment draft

Recommended next-session probe sequence (~1.5 h):
- w1 (`2× --cpunodebind=N --membind=N -t 24`) — tests if 122B benefits from per-node
  binding similar to the 30B-A3B 96t-single-NUMA-node operating point
- w3 = already done (canonical 1×96t + c2)

## Recommended actions

### v5 deployment draft update (do this session)

Move `architect_general` from `todo_or_undecided` to `roles:` with:
```yaml
architect_general:
  arch_class: moe_q4_bw_bound_mbind_sensitive
  model: Qwen3.5-122B-A10B
  quant: Q4_K_M
  binary_path: build_libomp_pgo_use/bin/   # PGO universal
  env:
    GGML_NUMA_REPACK_INTERLEAVE: 0  # +1.28% canonical, σ ~0.4%, z~3
  expected_throughput:
    tg32_canonical_96t: ~12.2 t/s    # single-instance, c2 env
  source_bundle: data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe/
```

### Orchestrator integration (do this session)

Add `architect_general` entry to `_ROLE_ENV_BLOCKS` in orchestrator_stack.py so production
launches pick up the c2 env block automatically. Note the production wiring switch
(2× → 1× or to per-node pinning) is a separate decision tracked under the bonus probe
follow-up.

### Production wiring follow-up (separate handoff)

The 2.8× per-instance gap is a wiring problem, not an env-block problem. Open a new handoff
to evaluate w1 (per-NUMA-node binding for 2 concurrent instances) and decide:
- Stay 2-instance for concurrent capacity (need w1 number to compare)
- Switch to 1-instance canonical for ~+184% per-instance latency

Cross-reference: this finding parallels the 30B-A3B 96t-single-NUMA-node win
(`project_96t_single_node_operating_point.md`, +26% vs production 24t worker_explore
that was also unused).
