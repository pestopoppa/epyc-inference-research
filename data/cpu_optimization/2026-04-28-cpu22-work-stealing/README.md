# CPU22 — Phase 3 Work-Stealing Prototype + Validation (artifact bundle)

**Track**: CPU22 — Dynamic MoE Load Balancing ([handoff](../../../../../workspace/handoffs/active/cpu-dynamic-moe-load-balancing.md))
**Run date**: 2026-04-28
**Purpose**: Phase 3 of closure-inflation remediation plan. CPU22 was originally "closed by inference" (CPU24 found sync share = 15%, so dynamic balancing's ceiling ~7%) WITHOUT running the binding empirical gate (≥10% on 2 sync-bound Q4_K_M models, no crash, PPL bit-exact). Phase 1 of remediation reverted that closure to ACTIVE pending prototype run. This bundle delivers the prototype + binding-gate measurement.

## Verdict

**CLOSED via test — gate FAILED**. Work-stealing prototype is mathematically correct (PPL bit-exact at 12 chunks on Coder-30B Q4_K_M) but throughput is neutral-to-slightly-negative across all 3 sync-bound MoE models tested. Closes the track honestly via empirical measurement (replacing the prior closure-by-inference).

## Prototype design

Single new commit on `feature/cpu-ep-inter-process` (forthcoming) adds an env-gated work-stealing path to `ggml_compute_forward_mul_mat_id` in `ggml/src/ggml-cpu/ggml-cpu.c`:

- **Env flag**: `GGML_EP_WORK_STEALING=1` (default 0 = off, falls through to existing per-expert sequential chunked path).
- **Mechanism**: at op-entry, ith==0 builds a global tile array (one tile per chunk × per non-empty expert), encoded as int64 = `(cur_a<<32)|(ith1<<16)|ith0`. After the existing barrier, all threads pull tiles from a single atomic counter (`atomic_fetch_add` on `s_ws_next`). When the counter exceeds `s_ws_total`, threads break out.
- **Tile storage**: 512 KB static buffer (256 experts × 256 chunks max = 65536 tiles × 8B). Comfortably fits Coder-30B (128 experts × ~96 chunks = 12K tiles), Next-80B (128 experts × ~80 chunks = 10K tiles), REAP-246B (similar).
- **Excluded paths** (work-stealing falls through to existing path when active): EP master/worker drone (`ep_inter`), per-CCD sharding (`ep_active`), master-parker mode. Single-instance, non-CCD-sharded MoE only.

Compatibility:
- Existing per-expert atomic counters (`atomic_current_chunk[]`) remain in the wdata workspace and are unused in the work-stealing path. No conflict.
- The `ggml_compute_forward_mul_mat_id_one_chunk` kernel function is reused unchanged. Tile-to-chunk math is recomputed per tile pull (`nchunk0`, `nchunk1`, `dr0`, `dr1`) since per-expert dimensions vary.

## PPL bit-exactness — REQUIRED gate

Coder-30B Q4_K_M wiki.test.raw chunks 1-12, fa=1, --no-mmap, taskset 0-95 + numactl --interleave=all:

| Path | PPL chunks 1-12 |
|---|---|
| env=0 (existing per-expert chunked) | **11.1146 ± 0.62405** |
| env=1 (work-stealing) | **11.1146 ± 0.62405** |

**Bit-exact byte-for-byte across all 12 chunks.** The work-stealing path's tile dispatch is mathematically equivalent to the existing path's per-expert chunked dispatch — both cover the same (cur_a, chunk) coordinate space without overlap or gaps.

## Throughput gate — FAILED

Proper canonical: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active taskset -c 0-95 numactl --interleave=all -t 96 -fa 1 -p 0 -n 64 -mmp 0 -r 5`.

| Model | Class | env=0 (existing) | env=1 (work-stealing) | Δ | Verdict |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M | sync-bound MoE / hybrid SSM-Dense-ish | 53.12 ± 0.10 | 51.89 ± 0.07 | **−2.3%** | regression |
| Qwen3-Next-80B-A3B Q4_K_M | sync-bound MoE | 23.36 ± 0.03 | 23.29 ± 0.07 | **−0.3%** (within noise) | neutral |
| Qwen3-Coder-REAP-246B-A35B Q4_K_M | DRAM-bound large MoE | 6.64 ± 0.01 | 6.59 ± 0.02 | **−0.8%** (within noise) | neutral |

**Gate threshold**: ≥10% on at least 2 of the 3 sync-bound models tested. **NOT MET** (zero models at +10%; all three are negative or within noise).

3-rep vs 5-rep gotcha (documented for posterity): the initial 3-rep Next-80B measurement showed env=1 = 22.65 vs env=0 = 21.31 (+6.3%), which would have been a positive signal. Re-running at 5 reps converged both paths to ~23.3 (Δ -0.3%, neutral). The 3-rep result was a measurement artifact — likely cache-warmup state divergence between runs. **Lesson**: 3 reps is insufficient for tight gates on this hardware; ≥5 reps required for sub-5% deltas.

## Why the prototype doesn't deliver

CPU24's perf-record finding bounds the gain: 15% sync share on REAP-246B, so the theoretical ceiling for sync-recovery work-stealing is ≈7-15%. The empirical reality is **the existing per-expert chunked path already has chunk-level work-stealing within each expert** (atomic_fetch_add on a per-expert counter). Each thread independently iterates through experts and claims chunks via atomics; threads progress through the expert list at their own rate. There is no per-expert barrier — slow threads on hard experts don't block fast threads.

The prototype's contribution is **inter-expert work-stealing**: threads pull from a global tile pool regardless of their per-thread expert-iteration position. In principle this helps when one expert is much heavier than others (its chunks last longer than other experts' chunks combined), but in practice:

1. **Chunk size already adapts to per-expert load** (lines 2113-2137 of the original code adjust `nchunk0`/`nchunk1` based on `cne1`).
2. **Atomic contention overhead** on the single global counter scales with thread count. At 96 threads pulling from one atomic, contention adds ~30 ns/op × N tiles, which exceeds the imbalance-recovery gain on workloads that already balance well.
3. **Tile-decode overhead** (parsing the encoded int64 + recomputing per-expert dimensions per tile) adds CPU cycles per tile, vs the existing path's once-per-expert dimension compute.

The Coder-30B regression (-2.3%) is consistent with these overhead components dominating over what minimal imbalance-recovery gain is available on this model.

## Stability — informal observation

5-rep runs at 96 threads completed without crash, deadlock, or PPL drift on all 3 models (Coder-30B Q4_K_M, Next-80B Q4_K_M, REAP-246B Q4_K_M). No 5-minute sustained-run stress test was performed (low marginal value given the negative throughput result), but the implementation pattern (single barrier + atomic counter, no per-thread state) is well-trodden and should not introduce stability issues distinct from the existing path.

## Closure scope

**Closed via test**: Phase 3 of remediation. The prototype was implemented, validated PPL bit-exact, and benched on 3 sync-bound MoE models. Gate FAILED (negative or within noise on all 3). The earlier closure-by-inference (CPU24 sync ceiling) was directionally correct but not gate-meeting; this empirical run is the definitive close.

**Code disposition**: the work-stealing path is preserved in the codebase compile-time — env-gated default-OFF — for any future hardware where (a) the atomic-contention overhead is lower (e.g., faster atomics or different topology), or (b) workloads emerge with severely imbalanced expert loads where the existing chunk-level work-stealing is insufficient. Strip if v5 audit prefers a smaller surface; otherwise leave as documented dead-code-by-default.

## Files

| File | Purpose |
|---|---|
| `coder30b_ppl_env0.log`, `coder30b_ppl_env1.log` | PPL bit-exact validation |
| `coder30b_env{0,1}_tg.log` | Coder-30B 5-rep throughput |
| `next80b_env{0,1}_tg.log` | Next-80B 3-rep (initial; +6.3% artifact) |
| `next80b_env{0,1}_tg_5rep.log` | Next-80B 5-rep (converged to neutral) |
| `reap246b_env{0,1}_tg.log` | REAP-246B 3-rep |
| `reap246b_env{0,1}_tg_5rep.log` | REAP-246B 5-rep |
| `system-state.txt`, `process-pre.txt`, `process-post.txt`, `ld_debug.log` | CPU20 protocol files |
| `results.csv` | tabulated env=0 vs env=1 across 3 models |
| `decision.md` | binding-gate verdict + closure |
