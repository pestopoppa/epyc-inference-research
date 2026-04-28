# CPU11 — LLVM PGO on libomp build (artifact bundle)

**Track**: CPU11 PGO/LTO ([cpu-inference-optimization-index.md](../../../../../workspace/handoffs/active/cpu-inference-optimization-index.md))
**Run date**: 2026-04-28
**Purpose**: Empirical follow-up to Phase 2.1's libomp finding. Phase 2.1 showed clang+libomp delivers a substantial runtime win on the sync-bound MoE class; PGO is the natural next compiler-level lever. This bundle measures profile-guided optimization on top of the libomp build across all four production model classes (sync-bound MoE, BW-bound MoE, DRAM-bound MoE, dense/hybrid).

## Headline finding

**PGO is universally positive. Largest gain on the BW-bound Q8 frontdoor (+6.6%); smallest on DRAM-bound REAP-246B (+1.3%). All four classes statistically significant.**

| Model | Quant | Class | libomp baseline | + PGO | Δ |
|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | Q4_K_M | sync-bound MoE | 56.84 ± 0.19 | **58.65 ± 0.24** | **+3.2%** |
| Qwen3.6-35B-A3B | Q8_0 | BW-bound frontdoor MoE | 25.40 ± 0.05 | **27.08 ± 0.03** | **+6.6%** |
| Qwen3-Coder-REAP-246B-A35B | Q4_K_M | DRAM-bound large MoE | 6.82 ± 0.03 | **6.91 ± 0.02** | **+1.3%** |
| Qwen3.5-27B | Q8_0 | dense/hybrid SSM-Dense | 5.49 ± 0.01 | **5.62 ± 0.01** | **+2.4%** |

All measurements: `taskset -c 0-95 -t 96 -fa 1 -mmp 0`, 5-rep llama-bench tg32. PGO build = `build_libomp_pgo_use/` (`-fprofile-instr-use=merged.profdata`).

## Method

PGO pipeline:
1. **Instrumented build** (`build_libomp_pgo_gen/`): clang-20 + libomp + `-march=znver5` + `-fprofile-instr-generate=%p.profraw`, linked against `libclang_rt.profile-x86_64.a` (`libclang-rt-20-dev`).
2. **Training run** (profile collection): tg32 on Coder-30B Q4_K_M chunks 1-4 of WikiText-2, ~6K tokens of decode through the full kernel path. Output: 96 `*.profraw` files (one per worker thread) + 1 main thread profraw.
3. **Profile merge**: `llvm-profdata-20 merge -output=merged.profdata *.profraw` → 31 MB merged profile.
4. **PGO use build** (`build_libomp_pgo_use/`): clang-20 + libomp + `-march=znver5` + `-fprofile-instr-use=merged.profdata`. Build hash `0bc793637`.
5. **Measurement**: same llama-bench harness, 5-rep at proper canonical, all 4 production model classes.
6. **PPL bit-exactness gate**: WikiText-2 chunks 1-12 on Coder-30B Q4_K_M to verify PGO doesn't change fp ordering.

## PPL bit-exactness

Coder-30B Q4_K_M chunks 1-12: PGO build PPL = **11.1146 ± 0.62405** — byte-identical to the libomp baseline (which is itself bit-exact within compiler determinism). PGO does not change floating-point operation order; the optimizer is allowed to reorder branches, inline more aggressively, and adjust register allocation, but does not introduce reassociation. Quality preserved.

## Why is PGO universal-positive

Distinct from libomp (which only helped sync-bound Coder-30B): the libomp finding was a *runtime change* that hits the OpenMP barrier path (only a meaningful overhead on thinner per-thread tiles). PGO is a *codegen change* applied to the whole CPU backend, including:
- `mul_mat_id` dispatcher (better branch prediction for the per-expert routing)
- Q4_K / Q8_0 dot-product inner loops (better register allocation, layout)
- ggml_compute_forward_* function preludes (cold/hot split, smaller hot-path icache footprint)
- libomp's own thread-pool barrier (the `--mllvm -no-enable-noundef-analysis` profile reaches into clang/libomp itself)

These wins compound across all model classes; they're orthogonal to the BW-saturation regime that bounds the runtime ceiling.

## Compounding stack (Coder-30B Q4_K_M tg32)

| Build | t/s | vs prior step | vs gcc+libgomp+no-march |
|---|---|---|---|
| gcc + libgomp + no-march (original `build/`) | 48.28 | reference | reference |
| gcc + libgomp + `-march=znver5` | 50.06 | +3.7% (codegen) | +3.7% |
| clang + libomp + `-march=znver5` | 56.84 | +13.5% (runtime + codegen) | +17.7% |
| clang + libomp + `-march=znver5` + PGO | 58.65 | +3.2% (PGO codegen) | +21.5% |

PGO compounds cleanly on top of libomp. Q8 frontdoor gains the most; REAP-246B gains least (already DRAM-saturated, less codegen headroom).

## Deployable runtime profile (REVISED)

The CPU21-best universal stack stays: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`.

**v5 cherry-pick recommendation**: ship a **clang+libomp+znver5+PGO** build as the production binary. Universal positive on all 4 model classes, PPL bit-exact, no runtime config changes required.

Build environment dependency:
- `clang-20` (~150 MB)
- `libclang-rt-20-dev` (PGO instrumentation runtime)
- `llvm-20` (`llvm-profdata-20` for merge step)
- ~30-min profile-and-rebuild cycle

## Files

| File | Purpose |
|---|---|
| `coder30b_libomp_baseline.log`, `q8_libomp_baseline.log`, `reap_libomp_baseline.log`, `dense_libomp_baseline.log` | libomp baselines (5-rep, build_libomp/) |
| `coder30b_pgo_use.log`, `q8_pgo_use.log`, `reap_pgo_use.log`, `dense_pgo_use.log` | PGO use builds (5-rep, build_libomp_pgo_use/) |
| `coder30b_pgo_ppl.log` | PPL chunks 1-12 bit-exactness gate |
| `system-state.txt`, `process-pre.txt`, `process-post.txt`, `ld_debug.log` | CPU20 protocol files |
| `results.csv` | tabulated results |
| `decision.md` | verdict + v5 cherry-pick recommendation |
