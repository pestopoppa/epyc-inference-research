# CPU11 PGO — Decision

**Verdict**: **DEPLOY for v5.** PGO is universally positive across all 4 production model classes (+1.3% to +6.6%, all statistically significant), PPL bit-exact, no runtime configuration changes. Recommended cherry-pick as the production build path replacing the current gcc+libgomp build.

## Headline result

| Model | Class | libomp baseline | + PGO | Δ |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B Q4_K_M | sync-bound MoE | 56.84 ± 0.19 | **58.65 ± 0.24** | **+3.2%** |
| Qwen3.6-35B-A3B Q8_0 | BW-bound frontdoor MoE | 25.40 ± 0.05 | **27.08 ± 0.03** | **+6.6%** |
| Qwen3-Coder-REAP-246B-A35B Q4_K_M | DRAM-bound large MoE | 6.82 ± 0.03 | **6.91 ± 0.02** | **+1.3%** |
| Qwen3.5-27B Q8_0 | dense/hybrid SSM-Dense | 5.49 ± 0.01 | **5.62 ± 0.01** | **+2.4%** |

All 5-rep `taskset -c 0-95 -t 96 -fa 1 -mmp 0` llama-bench tg32 at proper canonical.

## Compounding vs prior compiler-level wins

PGO is ORTHOGONAL to the libomp finding (which was a *runtime* change). PGO is a *codegen* change. They compound cleanly:

| Build | Coder-30B tg32 | Δ vs prior |
|---|---|---|
| gcc + libgomp + no-march | 48.28 | reference |
| + `-march=znver5` | 50.06 | +3.7% codegen |
| + libomp runtime | 56.84 | +13.5% runtime |
| + PGO codegen | 58.65 | +3.2% codegen |

**Total compounded gain over original v4 production binary: +21.5% on Coder-30B Q4_K_M, +14.0% on Q8 frontdoor, +9.9% on REAP-246B, +20.9% on dense 27B.**

## Quality

PPL bit-exact on Coder-30B Q4_K_M wiki.test chunks 1-12: 11.1146 ± 0.62405, byte-identical to the libomp pre-PGO build. PGO does not introduce reassociation; only branch layout, inlining, register allocation. No quality regression.

## Why it works on every class (unlike libomp)

- libomp helped only Coder-30B because it targeted the OpenMP barrier path (thin per-thread tiles).
- PGO improves the entire CPU backend hot path: `mul_mat_id` dispatcher, Q4_K / Q8_0 dot loops, ggml function preludes, libomp's own runtime. These wins are not bound by per-thread BW like the runtime barrier was, so all model classes see uplift.
- BW-bound Q8 frontdoor benefits the *most* — when memory is the bottleneck, every saved cycle in the hot inner loop translates more directly to throughput.

## v5 cherry-pick implications

Build environment additions (one-time):
- `apt install clang-20 libclang-rt-20-dev llvm-20`

Build pipeline becomes a 2-stage process:
1. `cmake -B build_libomp_pgo_gen -DCMAKE_C_FLAGS="-march=znver5 -fprofile-instr-generate=%p.profraw" -DCMAKE_CXX_FLAGS="-march=znver5 -fprofile-instr-generate=%p.profraw"  -DCMAKE_C_COMPILER=/usr/bin/clang-20 -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 -DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON`
2. Run training: `taskset -c 0-95 llama-bench -t 96 -fa 1 -mmp 0 -p 0 -n 32` on representative model (Coder-30B Q4_K_M chunks 1-4 worked well).
3. `llvm-profdata-20 merge -output=merged.profdata build_libomp_pgo_gen/*.profraw`
4. `cmake -B build_libomp_pgo_use -DCMAKE_C_FLAGS="-march=znver5 -fprofile-instr-use=$PWD/merged.profdata" -DCMAKE_CXX_FLAGS="-march=znver5 -fprofile-instr-use=$PWD/merged.profdata"  ...same compiler flags...`
5. Ship `build_libomp_pgo_use/bin/llama-server` as the production binary.

Total: ~30-min profile-and-rebuild cycle. Profile is portable across ~all production hardware (same SKU); needs re-collection only if the codebase changes substantially or new hot models are added.

## Closure scope

**Closed**:
- PGO uplift across 4 production model classes (this bundle)
- PPL bit-exactness verified
- Compounding with libomp confirmed (orthogonal optimizations)

**NOT closed (tracked separately)**:
- LTO without PGO (CPU11 includes LTO but quick LTO-only runs were within noise; LTO's main benefit on this codebase is enabling cross-TU inlining that PGO already triggers — leaving as not-pursued unless v5 audit needs it)
- BOLT post-link optimization (CPU12 — see sibling bundle `2026-04-28-cpu12-bolt/`)

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 2.1 followup. CPU11 was queued for v5+1; this bundle moves it to "executed, results landed" status.
