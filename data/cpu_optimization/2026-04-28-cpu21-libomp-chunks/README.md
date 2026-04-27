# CPU21 — Phase B Chunks 8/16 + libomp Comparison + Cross-Model Verification (artifact bundle)

**Track**: CPU21 — OpenMP Runtime And Scheduling Matrix ([handoff](../../../../../workspace/handoffs/active/cpu-openmp-runtime-scheduling-matrix.md))
**Run date**: 2026-04-28
**Purpose**: Phase 2.1 of closure-inflation remediation plan — completes the CPU21 promised matrix. Original CPU21 sweep (`2026-04-26-cpu21/`) ran libgomp + chunks 1+4 only. This bundle adds:
- chunks 8+16 across static/dynamic/guided under libgomp (with apples-to-apples baseline)
- a **clang+libomp** build (`build_libomp/` in llama.cpp-experimental, after authorized `apt install clang-20`)
- libomp baseline + chunks 8/16 under libomp on Coder-30B Q4_K_M
- cross-model verification on REAP-246B Q4_K_M and Qwen3.6-35B Q8_0 under both runtimes
- PPL sanity check on libomp build

## Headline findings

### **libomp delivers +6.4% on Coder-30B Q4_K_M** (apples-to-apples vs libgomp at -march=znver5)

This is a major new finding. Earlier CPU21 framing said "libgomp's defaults are already near-optimal"; that conclusion stands when comparing within libgomp, but switching the OpenMP runtime to libomp adds a substantial gain on the sync-bound MoE class.

| Build | Compiler + Runtime | -march | Coder-30B tg32 (5-rep) |
|---|---|---|---|
| `build/` | gcc + libgomp | (none) | 48.28 ± 0.11 |
| `build_znver5/` | gcc + libgomp | znver5 | 50.06 ± 0.05 |
| `build_libomp/` | **clang + libomp** | znver5 | **53.28 ± 0.11** |

Decomposition:
- gcc → -march=znver5: +3.7% (auto-vectorization gain)
- libgomp → libomp: **+6.4%** (runtime change at fixed -march)
- Combined `OMP_SCHEDULE=guided,16` on top of libomp: 53.28 → 53.94 = +1.2% additional (small, within libomp variance)

### Win is MOSTLY model-specific

| Model | Quant | Class | libgomp+znver5 | libomp+znver5 | Δ (runtime change) |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | Q4_K_M | sync-bound MoE / hybrid SSM-Dense-ish | 50.06 ± 0.05 | **53.28 ± 0.11** | **+6.4%** |
| Qwen3-Coder-REAP-246B-A35B | Q4_K_M | DRAM-bound large MoE | 6.52 ± 0.02 | 6.47 ± 0.03 | -0.8% (within noise) |
| Qwen3.6-35B-A3B | Q8_0 | BW-bound frontdoor MoE | 23.82 ± 0.07 | 24.00 ± 0.04 | +0.8% (within noise) |

Coder-30B-A3B benefits substantially from libomp's runtime; REAP-246B and Qwen3.6-35B Q8 are neutral. Likely mechanism: thinner per-thread row-shard tiles (3.3B activated at A3B) benefit from libomp's lower-overhead barrier and task scheduling. Larger MoE (REAP-A35B) and BW-bound classes (Q8 frontdoor) saturate on memory bandwidth before the runtime overhead matters.

### libomp Phase B chunk sweep (Coder-30B)

| Config | tg32 t/s | Δ vs libomp baseline |
|---|---|---|
| baseline (no OMP_SCHEDULE) | 53.28 ± 0.11 | reference |
| static,8 | 52.83 ± 0.24 | -0.8% |
| static,16 | 53.52 ± 0.20 | +0.5% |
| dynamic,8 | 52.71 ± 0.39 | -1.1% |
| dynamic,16 | 53.81 ± 0.35 | +1.0% |
| guided,8 | 53.64 ± 0.45 | +0.7% |
| guided,16 | **53.94 ± 0.13** | **+1.2%** |

Under libomp, schedule policy delta is small (within 1.2% — much narrower than the libgomp +3.6% guided,16 finding). libomp's default scheduling is already closer to its libomp+guided,16 ceiling than libgomp's default was to its own ceiling.

### libgomp Phase B chunk sweep (Coder-30B, this bundle, on `build/` no-march)

| Config | tg32 t/s | Δ vs build/ baseline (48.28) |
|---|---|---|
| baseline | 48.28 ± 0.11 | reference |
| static,8 | 48.07 ± 0.22 | -0.4% |
| static,16 | 47.45 ± 0.21 | -1.7% |
| dynamic,8 | 46.89 ± 0.34 | -2.9% |
| dynamic,16 | 47.88 ± 0.06 | -0.8% |
| guided,8 | 48.34 ± 0.09 | +0.1% (within noise) |
| guided,16 | 49.53 ± 0.16 (3-rep) / 50.01 ± 0.38 (5-rep verify) | +2.6% / +3.6% |

**Note**: the libgomp Phase B runs above were on `build/` (no -march). The apples-to-apples libgomp+znver5 reference is 50.06 — so `guided,16` brings libgomp+`build/` to ~50.01 = nearly matches the libgomp+znver5 baseline. libomp WITHOUT guided,16 is **already +6.4% over libgomp+znver5+guided,16-equivalent**.

### PPL bit-exactness check

Coder-30B Q4_K_M, chunks 1-12 of WikiText-2:
- libomp+znver5 (build_libomp): PPL = **11.1146 ± 0.62405** (chunk1 = 7.5697, chunk12 = 11.1146)
- libgomp+znver5 (build_znver5): PPL = 11.1215 ± 0.62430 (chunk1 = 7.4537, chunk12 = 11.1215)

Δ = 0.0069 at chunk12. This is **clang vs gcc compiler codegen drift** in floating-point intermediates — same kind of fp-rounding-noise we saw earlier with the `-march=znver5` codegen drift. Not a quality regression; both PPLs are "correct" within the compiler's fp ordering.

Notably the libomp+znver5 PPL (11.1146) byte-matches the original gcc-no-march `build/` PPL (11.1146) exactly. Compiler determinism within a build is bit-exact (re-running same build produces byte-identical output).

## Deployable runtime profile (REVISED)

The CPU21-best universal stack remains: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`.

**New deployable opt-in for Coder-30B-A3B-Instruct workloads (build-time decision)**:
- Compile against libomp (clang-20 + `-march=znver5`): **+6.4%** on Coder-30B specifically.
- Optionally add `OMP_SCHEDULE=guided,16` for +1.2% additional under libomp (or +3.6% if stuck on libgomp).

The libomp build path is a v5 cherry-pick / build-tooling decision. See `decision.md` (sibling file in this directory) for the two deployment options (universal libomp vs per-role variants) and the recommendation.

## Files

| File | Purpose |
|---|---|
| `B_baseline_spread_cores_active.log` | apples-to-apples libgomp+`build/` baseline for chunk runs |
| `B_static_chunk{8,16}.log`, `B_dynamic_chunk{8,16}.log`, `B_guided_chunk{8,16}.log` | libgomp Phase B per-config (under `build/`) |
| `B_guided_chunk16_verify.log` | 5-rep statistical-significance check on guided,16 under libgomp |
| `Q8_baseline_*.log`, `Q8_guided_chunk16.log` | Qwen3.6-35B Q8 cross-model verification (libgomp+`build/`) |
| `REAP_baseline_*.log`, `REAP_guided_chunk16.log` | REAP-246B cross-model verification (libgomp+`build/`) |
| `libgomp_znver5_baseline_coder30b.log`, `libgomp_znver5_baseline_reap.log`, `libgomp_znver5_baseline_q8.log` | apples-to-apples libgomp+znver5 baselines |
| `libomp_baseline_coder30b.log` | libomp+znver5 baseline, 5-rep |
| `libomp_baseline_reap.log`, `libomp_baseline_q8.log` | libomp cross-model |
| `libomp_B_*.log` | libomp Phase B chunk sweep |
| `libomp_ppl_coder30b_chunks12.log` | libomp PPL sanity check |
| `system-state.txt`, `process-pre.txt`, `process-post.txt`, `ld_debug.log` | CPU20 protocol files |
| `results.csv` | tabulated results |
| `decision.md` | verdict + deployable recommendation + libomp v5 implications |
