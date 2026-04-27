# CPU21 — Phase B Chunks 8/16 + Cross-Model Verification (artifact bundle)

**Track**: CPU21 — OpenMP Runtime And Scheduling Matrix ([handoff](../../../../../workspace/handoffs/active/cpu-openmp-runtime-scheduling-matrix.md))
**Run date**: 2026-04-28
**Purpose**: Phase 2.1 of closure-inflation remediation plan. Original CPU21 sweep (`2026-04-26-cpu21/`) ran chunks 1+4 only across static/dynamic/guided. This bundle adds chunks 8+16 across the same three policies, plus cross-model verification of any new findings on REAP-246B Q4_K_M (sync-bound) and Qwen3.6-35B Q8_0 (BW-bound).

## Scope

Run on top of the CPU21-best stack baseline (`OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`); the variable swept is `OMP_SCHEDULE`.

| Variable | Value |
|---|---|
| Affinity | `OMP_PROC_BIND=spread OMP_PLACES=cores` (CPU21-best) |
| Wait policy | `OMP_WAIT_POLICY=active` (CPU21-best) |
| Schedule × chunk | static/dynamic/guided × 8/16 (this bundle) |
| Wrapper | `taskset -c 0-95 numactl --interleave=all -t 96 -fa 1 -mmp 0 -r 3` |
| Binary | `/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench` (libgomp; HEAD `29a69599a`, default-flags build) |

## Verdict

**Partial completion of Phase 2.1**:
- ✅ Chunks 8/16 run on Coder-30B Q4_K_M across static/dynamic/guided
- ✅ Cross-model verification of the best chunk-16 finding on REAP-246B Q4_K_M and Qwen3.6-35B Q8_0
- ❌ libomp comparison NOT YET RUN — clang-20 toolchain install was sandbox-blocked. Surfacing to user for explicit authorization.

### Finding: `OMP_SCHEDULE=guided,16` is a model-specific +3.6% on Coder-30B Q4_K_M

Coder-30B Q4_K_M (hybrid SSM-Dense / sync-bound MoE proxy) shows a real, statistically-significant gain with `guided,16`:

| Config | tg32 t/s | Δ vs CPU21-best baseline |
|---|---|---|
| baseline (no OMP_SCHEDULE) | 48.28 ± 0.11 | reference |
| `OMP_SCHEDULE=static,8` | 48.07 ± 0.22 | -0.4% |
| `OMP_SCHEDULE=static,16` | 47.45 ± 0.21 | -1.7% |
| `OMP_SCHEDULE=dynamic,8` | 46.89 ± 0.34 | -2.9% |
| `OMP_SCHEDULE=dynamic,16` | 47.88 ± 0.06 | -0.8% |
| `OMP_SCHEDULE=guided,8` | 48.34 ± 0.09 | +0.1% (noise) |
| **`OMP_SCHEDULE=guided,16`** | **49.53 ± 0.16 (3-rep) / 50.01 ± 0.38 (5-rep verify)** | **+2.6% / +3.6%** |

Statistical significance check (5-rep verify run): 50.01 - 48.28 = 1.73 t/s gap; combined std ≈ 0.49; **3.5σ separation** — real signal.

### Cross-model verification (chunk-16 specifically)

| Model | Quant | Class | baseline | guided,16 | Δ |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | Q4_K_M | sync-bound MoE / Hybrid SSM-Dense-ish | 48.28 ± 0.11 | 50.01 ± 0.38 | **+3.6%** |
| Qwen3-Coder-REAP-246B-A35B | Q4_K_M | DRAM-bound large MoE | 6.29 ± 0.12 | 6.30 ± 0.07 | +0.16% (noise) |
| Qwen3.6-35B-A3B | Q8_0 | BW-bound frontdoor MoE | 23.69 ± 0.09 | 23.54 ± 0.02 | -0.6% (noise) |

**Interpretation**: `guided,16` is **NOT a universal win**. It helps Coder-30B Q4_K_M specifically — likely because Coder-30B has the smaller per-thread row-shard size (30B/A3B at Q4 has thinner work tiles per thread than REAP-246B/A35B or Qwen3.6-35B Q8_0), and `guided,16` gives the libgomp scheduler enough granularity to balance dynamic per-token variation. On the larger MoE (REAP) and the BW-bound (Q8) classes, the static partitioning of the matmul rows is already efficient and `guided,16` adds overhead without payoff.

**Deployable recommendation**: keep `OMP_SCHEDULE` UNSET in the production canonical (default static is fine universally). For Coder-30B-A3B-Instruct workloads specifically, the orchestrator could opt-in `OMP_SCHEDULE=guided,16` for a +3.6% lift. But this is a per-role optimization, not a default.

## libomp comparison — DEFERRED

The original handoff promised libgomp **and** libomp evaluation. The host has `libomp.so.5` runtime installed (via `libomp5-20`) but no clang compiler. The existing experimental build is GCC+libgomp. To get a libomp build, we need either:

1. `apt install clang-20` and rebuild (sandbox-blocked: requires user authorization for system-package install).
2. GCC linker flag manipulation (`-Xlinker -l:libomp.so.5` instead of libgomp). Fragile; risks symbol conflicts with the binary's libgomp linkage.
3. Skip libomp; document the gap.

Sandbox blocked option 1 (`sudo apt-get install clang-20` rejected). Surfaced to user for authorization.

## Files

| File | Purpose |
|---|---|
| `B_baseline_spread_cores_active.log` | apples-to-apples baseline for chunk runs |
| `B_static_chunk{8,16}.log`, `B_dynamic_chunk{8,16}.log`, `B_guided_chunk{8,16}.log` | per-config runs on Coder-30B |
| `B_guided_chunk16_verify.log` | 5-rep statistical-significance check on the +3.6% finding |
| `Q8_baseline_*.log`, `Q8_guided_chunk16.log` | Qwen3.6-35B Q8 cross-model verification |
| `REAP_baseline_*.log`, `REAP_guided_chunk16.log` | REAP-246B cross-model verification |
| `system-state.txt`, `process-pre.txt`, `process-post.txt`, `ld_debug.log` | CPU20 protocol files |
| `results.csv` | tabulated results |
| `decision.md` | verdict + scope + libomp deferral |
