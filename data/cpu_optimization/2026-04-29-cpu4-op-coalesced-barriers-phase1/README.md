# CPU4 Op-Coalesced Barriers — Phase 1 (NO-GO via test)

**Date**: 2026-04-29
**Phase**: 1 (prototype + measurement)
**Source**: Phase 0 GO verdict ([`../2026-04-29-cpu4-op-coalesced-barriers-phase0/decision.md`](../2026-04-29-cpu4-op-coalesced-barriers-phase0/decision.md)).
**Build**: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/` after applying Phase 1 patch to `ggml/src/ggml-cpu/ggml-cpu.c`.

## Implementation summary

- ~80 LOC patch in `ggml-cpu.c` (compute-loop op-iteration), env-gated `GGML_BARRIER_COALESCE=1` (default off).
- For each adjacent op pair (N, N+1): skip the between-op barrier when (a) next op's `src[]` does NOT contain cur node (no read-after-write dep), AND (b) both ops are in the safe allowlist.
- **Initial allowlist (REVISED post-smoke)**: `RMS_NORM`, `NORM`, `ROPE`, `MUL`, `ADD`, `SCALE`, `UNARY`, `GLU`.
- **MUL_MAT and MUL_MAT_ID EXCLUDED** — discovered via smoke test that their shared `params->wdata` buffer (used for src1 quantization before the internal barrier at `ggml_compute_forward_mul_mat:1487`) creates a wdata-write race when coalesced. Smoke test at COALESCE=1 with MUL_MAT in allowlist produced garbled output ("Failed to parse input at pos 0: GD Sw\n…\nCompatibility几..."). With MUL_MAT excluded, smoke test passes bit-exact.

## Phase 0 estimate vs Phase 1 measurement

**Phase 0 estimate**: 24-29% per-token barrier-count reduction (Q/K/V MUL_MAT coalescing + Q-norm/K-norm + RoPE-Q/RoPE-K).

**Phase 1 reality after wdata-aware allowlist correction**: Q/K/V MUL_MAT coalescing dropped due to wdata race. Per-layer skippable in attention block: 1 (ROPE-Q → RMS_NORM-K is the only IND pair under safe allowlist). MoE FFN block: 0 skippable. Per-layer ~1/21 = 4.7%. Per-token ~48/1012 = 4.7%. **WAY below the original Phase 0 24-29% estimate.**

Phase 0 was empirically wrong about MUL_MAT. The wdata-shared-buffer hazard wasn't checked in Phase 0 static analysis.

## Correctness gate — PASSED

PPL chunk-3 on WikiText-2:

| Model | COALESCE=0 | COALESCE=1 | Verdict |
|---|---|---|---|
| Coder-30B Q4_K_M | 9.8567 ± 1.23745 | 9.8567 ± 1.23745 | ✅ BIT-EXACT |
| REAP-246B Q4_K_M | 9.3042 ± 0.99072 | 9.3042 ± 0.99072 | ✅ BIT-EXACT |

The op-coalescing path produces bit-identical model outputs. No silent reordering bugs.

## Throughput gate — NOT MET

### First-pass tg64 5-rep canonical (`taskset -c 0-95 -t 96 -fa 1`)

| Model | COALESCE=0 mean ± std | COALESCE=1 mean ± std | Δ_pct | CV |
|---|---|---|---|---|
| Coder-30B Q4_K_M | 18.83 ± 6.32 | 22.69 ± 6.90 | +20.5% | 33% |
| Next-80B Q4_K_M  | 14.76 ± 5.21 | 13.85 ± 4.40 | -6.2% | 30% |
| REAP-246B Q4_K_M |  4.29 ± 0.05 |  4.19 ± 0.02 | -2.3% | <1% (clean) |

Coder's nominal +20.5% has CV=33% and SEM≈2.7 t/s; the delta (3.86 t/s) is < 1.5 SEM → not statistically significant at n=5.

### Replication on Coder — 3 alternated trials × 5 reps (n=15 per config)

| Trial | c0 t/s | c1 t/s | Δ |
|---|---|---|---|
| 1 | 27.93 ± 6.45 | 22.95 ± 8.46 | -17.8% |
| 2 | 20.61 ± 7.09 | 15.02 ± 4.62 | -27.1% |
| 3 | 24.11 ± 5.64 | 20.34 ± 7.01 | -15.6% |
| **Aggregate** | **24.22 ± 3.66** | **19.44 ± 4.04** | **-19.7%** |

The first-pass +20.5% was measurement variance — replication shows the actual direction is consistently NEGATIVE on Coder.

(Caveat: c0 was run before c1 in each trial. Some thermal/cache drift is plausible. But the consistent direction across 3 trials and the clean REAP signal suggest the effect is real, not just thermal artifact.)

### Aggregate verdict

| Model | n_total | Direction | Magnitude |
|---|---|---|---|
| Coder-30B | 20 (5 + 15) | tree LOSES | -10 to -20% (high noise) |
| Next-80B | 5 | tree LOSES (noisy) | -6.2% (NS) |
| REAP-246B | 5 | tree LOSES (clean) | -2.3% |

**No model shows ≥+5%. All 3 sync-bound Q4_K_M models show negative or null direction. Phase 1 GATE NOT MET.**

## Why the mechanism doesn't deliver

Several compounding factors:

1. **Phase 0 over-estimated coalescing potential by 5×**: Q/K/V MUL_MAT coalescing dropped due to wdata race. Real coalescable per-layer drops from 5 to ~1.

2. **Skipped barriers are CHEAP barriers**: ROPE→RMS_NORM is cheap (RMS_NORM is fast, ROPE is fast — both are tiny ops). Skipping a cheap barrier saves little wall-clock.

3. **Per-thread overhead from the dependency check**: even though the check is fast (10 pointer comparisons per node), it runs in EVERY thread for every node iteration. At 1000 ops × 96 threads × 64 tokens = 6.1M iterations × 10 comparisons = 60M ops. At ~1 ns each that's ~60 ms of work distributed across threads. Comparable to or larger than the barrier savings.

4. **Memory ordering / cache effects**: skipping barriers lets threads desync across op boundaries. Threads that race ahead may pollute caches that other threads then fight to evict, causing more cache misses than the savings.

## Phase 1 closure (per closure-inflation policy)

> "CPU4 Phase 1 op-coalesced barriers (`GGML_BARRIER_COALESCE=1`, env-gated default-off, ~80 LOC in ggml-cpu.c) at HEAD `d45126db5` on `feature/cpu-ep-inter-process` is structurally net-negative or null on the 3 sync-bound Q4_K_M models (Coder-30B-A3B at -10 to -20% with high variance, Next-80B at -6.2% noisy, REAP-246B at -2.3% clean) under v5 PGO build. PPL bit-exact verified.
>
> Phase 0's 24-29% estimated reduction was empirically WRONG: it included MUL_MAT/MUL_MAT_ID in the coalescable allowlist without checking the shared `params->wdata` buffer hazard. With the wdata-aware safe allowlist (excludes MUL_MAT/MUL_MAT_ID), the achievable per-token barrier-count reduction drops to ~5% (1 skippable barrier per layer in attention block under the conservative independence rule). The mechanism's gain ceiling is structurally bounded below the gate threshold.
>
> Does NOT generalize to: (a) architectures with different op chains (dense, hybrid SSM, attention-only) where coalesce-eligible adjacent op pairs may differ in count or cost class; (b) future ggml graph rewrites that introduce more parallel branches (e.g., a refactor that fuses Q/K/V into separate non-wdata-sharing graph paths); (c) prefill workloads where op chains may have different shape; (d) different barrier implementations (e.g., a custom ggml_barrier with lower overhead, where the skip-savings might exceed the per-iteration check overhead).
>
> Operational disposition: code in tree disabled-by-default (matches slot-promotion treatment). The other 5 untested CPU4 deferred avenues (token-to-expert rebalance, hybrid static+dynamic spillover, cross-CCD work migration, MoE quant layout rebalance) and the wdata-aware MUL_MAT coalescing variant remain open in the design note."

## Operational disposition

- **Patch stays in tree disabled-by-default** (`GGML_BARRIER_COALESCE=1` env required to enable; default off). Mirrors slot-promotion dispatcher v1 treatment.
- **Phase 0/Phase 1 mismatch documented**: Phase 0 manual analysis missed the wdata-shared-buffer hazard. Future Phase 0 analyses MUST check buffer-sharing constraints in the allowlist, not just dependency-graph independence.
- **CPU4 design note** ([`cpu4-deferred-avenues-design-note.md`](../../../../../workspace/handoffs/active/cpu4-deferred-avenues-design-note.md)) updated to mark design #7 as TESTED-AND-FAILED, with the closure scope above.
- **Other 5 designs in note still open**: token-to-expert rebalance, hybrid static+dynamic spillover, cross-CCD work migration, MoE quant layout rebalance, AND a new variant (wdata-aware MUL_MAT coalescing) — none have been tested.

## Files

- `run_smoke.sh` / `smoke_master.log` / `smoke2_master.log` — bit-exact smoke (initial garbled @ MUL_MAT in allowlist; passing after exclusion)
- `run_full.sh` / `full_master.log` — first-pass PPL + tg64 5-rep on 3 models × 2 configs
- `run_coder_replicate.sh` / `replicate_master.log` — Coder 3 alternated trials × 5 reps
- `srv_smoke_*.log`, `comp_smoke_*.json` — per-server smoke artifacts
- `ppl_*.log` — perplexity logs (Coder + REAP × 0/1)
- `bench_*.log` — llama-bench logs (3 models × 2 configs × first-pass + 3 trials × 2 configs replication)
- `decision.md` — Phase 1 NO-GO verdict + closure scope
