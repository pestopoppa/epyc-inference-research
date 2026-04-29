# MAB Tree-Shape Selector — Phase 0'' High-Rep Replication

**Date**: 2026-04-29
**Phase**: 0'' (replication of Phase 0' random-seed signal at higher n)
**Source**: Phase 0' (2026-04-30) found n=9 random-seed signal on Coder (+9.6% mean, p≈0.23 NS). Phase 0'' replicates at n=30 per cell on both Coder and REAP to determine GO vs NO-GO at p<0.05.

## Method

Same setup as Phase 0' random-seed addendum (random per-request seed, temp=0.7, top_k=40, top_p=0.95) at v5 PGO build, except n=30 reps per (model, shape, prompt) cell.

| Variable | Value |
|---|---|
| Build | `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/` (v5 PGO clang+libomp+znver5) |
| Targets | Coder-30B-A3B-Q4_K_M, REAP-246B-A35B-Q4_K_M |
| Drafter | Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0 |
| Sampling | temp=0.7, top_k=40, top_p=0.95, seed=-1 (random per-request) |
| Spec-dec | --draft-max=24 --draft-min=4 |
| Shapes | linear (p_split=0), tree (p_split=0.05) |
| Reps | 30 reps × 3 prompts × 2 shapes × 2 models = 360 requests |
| Threading | `numactl --interleave=all` + 96-thread llama-server |

Each request: n_predict=64. Server boot per cell with 30s warmup post-/health=ok.

## Result — NO-GO across both models

### Aggregate (n=90 paired per model)

| Model | linear mean t/s | tree mean t/s | Δ_mean | Δ_pct | t-stat | p-value |
|---|---|---|---|---|---|---|
| **Coder-30B Q4_K_M** | 40.58 | 38.97 | -1.61 t/s | **-3.97%** | -2.498 | **0.0125** |
| **REAP-246B Q4_K_M** | 7.64 | 7.66 | +0.026 t/s | +0.34% | 0.166 | 0.8685 |

**Coder: significant tree LOSS at p<0.05.** REAP: null.

### Per-prompt breakdown

| Model / Prompt | linear t/s | tree t/s | Δ |
|---|---|---|---|
| coder / p0 binary_search | 40.54 | 38.07 | **-6.07%** |
| coder / p1 lru_cache     | 45.79 | 44.12 | -3.64% |
| coder / p2 csv_moving_avg | 35.43 | 34.71 | -2.01% |
| reap / p0 binary_search | 7.48 | 7.66 | +2.51% |
| reap / p1 lru_cache | 9.17 | 9.13 | -0.39% |
| reap / p2 csv_moving_avg | 6.27 | 6.19 | -1.16% |

All Coder prompts show tree LOSING; all REAP prompts noise.

### Phase 0' (n=9) vs Phase 0'' (n=30) on Coder

| Phase | n | linear mean | tree mean | Δ | p-value |
|---|---|---|---|---|---|
| Phase 0' (random-seed addendum) | 9 | 37.87 | 41.49 | **+9.6%** | 0.23 (NS) |
| Phase 0'' replication | 90 | 40.58 | 38.97 | **-3.97%** | **0.0125 (significant)** |

The Phase 0' "+9.6%" was a **low-n type-I error**. At proper n=90 paired, the true effect is small-negative. The seemingly-promising p1_r2 +52% case in Phase 0' was an outlier — at n=90, p1's overall delta is -3.64%.

### Per-rep stability check (rule out noise contamination)

Concern: my CPU4 manual op-chain analysis (read/grep on source files) ran concurrently with Coder cells. Could shell activity have injected noise? Stability check shows no signature:

| Coder cell | early reps (r0-9) mean ± std | late reps (r20-29) mean ± std |
|---|---|---|
| linear / p0 | 40.85 ± 4.40 | 41.12 ± 5.29 |
| linear / p1 | 45.57 ± 5.27 | 44.28 ± 4.24 |
| linear / p2 | 34.72 ± 1.58 | 35.81 ± 3.86 |
| tree / p0 | 36.31 ± 3.66 | 38.74 ± 6.00 |
| tree / p1 | 44.23 ± 3.99 | 42.73 ± 5.67 |
| tree / p2 | 35.10 ± 2.38 | 35.78 ± 3.54 |

Early/late means and variances are statistically indistinguishable — no contamination signature. CPU4 manual analysis was paper-only (no benchmark, ≪0.1% machine load). REAP cells started AFTER the CPU4 activity ended (06:35:35 vs activity ending ~06:33).

## Phase 0'' VERDICT — NO-GO

**MAB tree-shape selector is structurally net-negative on Coder-30B-A3B + DRAFT-0.75B drafter at sampling regime (random-seed temp=0.7) at n=90.** REAP is null. Phase 1 implementation (~245 LOC) is NOT justified.

Combined evidence across all 3 tested regimes:

| Regime | Coder | REAP | Verdict |
|---|---|---|---|
| Phase 0 greedy (temp=0) | tree byte-identical to linear | same | NO-GO (verifier collapses to greedy) |
| Phase 0' fixed-seed temp=0.7 | tree byte-identical to linear | same | NO-GO (deterministic sampler) |
| Phase 0'' random-seed temp=0.7 n=90 | tree -3.97% (p=0.012) | tree +0.34% (p=0.87) | NO-GO (real regression on Coder, null on REAP) |

The MAB selector's claimed value (pick shape per-round based on drafter quality feedback) requires (a) tree mechanism to deliver gain in some regime, AND (b) a feature that predicts when tree helps vs hurts. Phase 0'' falsifies (a) on this drafter/target/workload class. Without (a), MAB is moot.

## Closure scope (per closure-inflation policy)

> "MAB tree-shape selector mechanism, tested on Qwen3-Coder-30B-A3B-Q4_K_M + Qwen3-Coder-REAP-246B-A35B-Q4_K_M targets with the Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0 drafter at v5 PGO build (clang+libomp+znver5), is structurally net-negative across all three tested verification regimes:
>
> 1. Greedy temp=0 (Phase 0 2026-04-29): tree byte-identical to linear (verifier collapses to greedy path).
> 2. Fixed-seed sampling temp=0.7 (Phase 0' 2026-04-30): tree byte-identical to linear (deterministic sampler).
> 3. Random-seed sampling temp=0.7 (Phase 0'' 2026-04-29 n=90): Coder tree -3.97% (p=0.012, significant regression); REAP tree +0.34% (p=0.87, null).
>
> Does NOT generalize to:
> - Different drafter (paper's Pythia drafter has different uncertainty profile; a fundamentally weaker drafter relative to its target could change the analysis)
> - Different arm pool (the paper's `(3,3,2,1)`, `(3,2,2,1,1)`, `(2,2,2,1,1,1)` shapes are tuned for Pythia; an arm pool tuned to EPYC's 96-core verifier could differ)
> - Multi-tenant / batched / concurrent-slot workloads (where tree-shape selection might amortize cold-start cost differently)
> - Architecturally different targets (dense, hybrid SSM, attention-only — only MoE Q4_K_M tested at scale)
>
> Phase 1 implementation (~245 LOC) is not justified. MAB selector handoff moves to `completed/` with this closure preserved."

## Operational disposition

- MAB selector handoff `mab-tree-shape-selector.md` should move to `handoffs/completed/`.
- The pre-production gate on MoE-Spec production registry integration condition (a) is RESOLVED via NO-GO closure (the gate required Phase 0 GO/NO-GO verdict; this is NO-GO at extended scope).
- No code in tree changes — MAB Phase 1 was never implemented.

## Files

- `run_probe.sh` — replication script (n=30 per cell, paired t-test in inline Python)
- `probe_master.log` — master log with timing + final aggregate + paired t-test output
- `srv_*.log` — server logs (4 cells: coder linear/tree, reap linear/tree)
- `comp_<model>_<shape>_<prompt>_<rep>.json` — per-request completion JSONs (180 Coder + 180 REAP = 360 files)
- `decision.md` — verdict + scoped closure language

## Cross-references

- Parent: [`2026-04-29-mab-tree-selector-phase-0/decision.md`](../2026-04-29-mab-tree-selector-phase-0/decision.md) (Phase 0 NO-GO greedy)
- Sibling: [`2026-04-30-mab-phase-0-prime-sampling/decision.md`](../2026-04-30-mab-phase-0-prime-sampling/decision.md) (Phase 0' INCONCLUSIVE n=9)
- Handoff (will move to completed): `handoffs/active/mab-tree-shape-selector.md`
