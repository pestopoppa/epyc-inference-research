# CPU21 Phase 2.1 — Decision (REVISED post-libomp completion)

**Verdict**: **MAJOR FINDING — libomp +6.4% on Coder-30B Q4_K_M (apples-to-apples vs libgomp at -march=znver5).** Phase 2.1 fully complete: chunks 8/16 swept under both runtimes, libomp build added (clang-20 toolchain installed), cross-model verified, PPL sanity-checked.

## Headline result

Switching the OpenMP runtime from libgomp to libomp delivers a real, statistically-significant win on the sync-bound MoE class:

| Build | Coder-30B Q4_K_M tg32 (5-rep) | Δ vs libgomp+znver5 |
|---|---|---|
| gcc + libgomp + -march=znver5 | 50.06 ± 0.05 | reference |
| **clang + libomp + -march=znver5** | **53.28 ± 0.11** | **+6.4%** |

The win is **NOT** universal:
- Qwen3.6-35B Q8_0: 23.82 → 24.00 (+0.8%, within noise)
- REAP-246B Q4_K_M: 6.52 → 6.47 (-0.8%, within noise)

Likely mechanism: Coder-30B-A3B has thinner per-thread row-shard tiles (3.3B activated params); libomp's lower-overhead barrier and task scheduling amplify on this class. Larger MoE (REAP-A35B) and BW-bound classes (Q8 frontdoor) saturate on memory bandwidth before the runtime overhead matters.

## Schedule policy under libomp

`OMP_SCHEDULE=guided,16` adds a small +1.2% on top of libomp baseline (53.28 → 53.94), much less than the +3.6% it gave on libgomp. libomp's default scheduling is closer to optimal than libgomp's was.

## PPL bit-exactness

Coder-30B Q4_K_M, chunks 1-12: libomp PPL = 11.1146 ± 0.62405 vs libgomp+znver5 11.1215 ± 0.62430. Δ = 0.0069 — pure clang vs gcc fp-codegen drift, not a quality regression. PPL is deterministic within a build (re-running same build produces byte-identical output). Quality-equivalent.

## Deployable runtime profile (REVISED)

The universal CPU21-best stack stays: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active`.

**New per-role opt-in for Coder-30B-A3B-Instruct workloads**:
- **Compile against libomp** (clang-20 + `-march=znver5`): +6.4% on Coder-30B specifically.
- Optionally add `OMP_SCHEDULE=guided,16` for +1.2% additional under libomp (or +3.6% if stuck on libgomp).

This becomes a v5 cherry-pick / build-tooling decision. Two paths:
1. **Universal libomp build for v5**: ship a single libomp-built llama-server. Coder-30B gains +6.4%; other production models neutral. Requires clang-20 in the build environment (~150 MB).
2. **Per-role build variants**: libomp-built for Coder-30B role, libgomp-built (existing) for other roles. More complex orchestration but maximizes per-role performance.

Recommendation: **option 1 (universal libomp)**, since the other models are neutral (not negative). Simpler v5 audit story: one binary, libomp-built, +6.4% on Coder-30B, neutral elsewhere. Validate at v5 audit time on the full lineup.

## Closure scope

**Closed**:
- Phase A affinity matrix (libgomp, existing 2026-04-26-cpu21/) — `spread+cores+active` is universal +3-8% deployable stack.
- Phase B chunks 1/4 (libgomp, existing 2026-04-26-cpu21/) — within noise.
- Phase B chunks 8/16 (libgomp, this bundle) — guided,16 is +3.6% on Coder-30B (model-specific) under libgomp.
- Phase B chunks 8/16 (libomp, this bundle) — guided,16 is +1.2% on Coder-30B; libomp's defaults are closer to optimal.
- Phase C wait policy (libgomp, existing) — passive trap; active +0.5%.
- libgomp vs libomp comparison (this bundle, post-clang-20-install) — libomp is +6.4% on Coder-30B Q4_K_M, neutral on REAP-246B Q4_K_M and Qwen3.6-35B Q8_0.
- PPL sanity check on libomp build — bit-exact within compiler determinism (clang vs gcc fp-codegen drift = 0.0069 PPL on chunks 1-12).

**NOT closed**:
- Phase 2.6 dense/hybrid (Qwen3.5/3.6-27B) coverage — separate Phase 2.6 task. Quick libomp sanity on dense will fall out of that.
- Full Phase A affinity matrix under libomp (only baseline + chunks tested) — small expected delta given libomp's defaults are already strong; can be added if v5 audit needs it.

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 2.1 (this bundle, COMPLETE) and Phase 2.6 (dense sanity, separate).

## v5 cherry-pick implications

If v5 ships a libomp build:
- Build environment dependency: `clang-20` package (~150 MB).
- Build flags: `CC=/usr/bin/clang-20 CXX=/usr/bin/clang++-20 CFLAGS="-march=znver5" CXXFLAGS="-march=znver5"` then `cmake .. -DGGML_LLAMAFILE=ON -DGGML_OPENMP=ON`.
- Linker: `libomp.so.5` (LLVM OpenMP runtime, already installed via `libomp5-20` runtime package).
- Performance delta vs current production v4 (gcc+libgomp): expected +6-10% on Coder-30B-A3B-Instruct (combining `-march=znver5` codegen +3.7% and libomp runtime +6.4%); neutral elsewhere.
