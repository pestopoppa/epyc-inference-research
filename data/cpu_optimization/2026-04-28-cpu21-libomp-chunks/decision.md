# CPU21 Phase 2.1 — Decision

**Verdict**: **PARTIAL** (chunks 8/16 swept + cross-model verified; libomp comparison BLOCKED on toolchain install authorization).

## What was decided

### Chunk 8/16 sweep on Coder-30B Q4_K_M

`OMP_SCHEDULE=guided,16` is a **statistically-significant +3.6% on Coder-30B Q4_K_M** (50.01 ± 0.38 vs 48.28 ± 0.11 baseline; 3.5σ separation in 5-rep verification). The win is **model-specific**:

- Coder-30B Q4_K_M: **+3.6%** (real signal)
- Qwen3.6-35B Q8_0: -0.6% (within noise)
- REAP-246B Q4_K_M: +0.16% (within noise)

**Likely mechanism**: Coder-30B-A3B has thinner per-thread row-shard work tiles than the larger MoE classes (A3B = 3.3B activated params, vs A35B for the 35B/246B class). `guided,16` gives libgomp's scheduler enough granularity to balance per-token routing variation, without the overhead becoming prohibitive on smaller-active-param models. The larger MoE and BW-bound classes don't benefit because their static partition is already adequately balanced.

### Deployable stack (revised)

The CPU21-best universal stack remains: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active` (no `OMP_SCHEDULE`). This is what `cpu-benchmark-rigor-and-revalidation.md` P3 already prescribes. **Do NOT default `OMP_SCHEDULE=guided,16` system-wide** — it regresses Qwen3.6-35B Q8 by -0.6%.

**Per-role opt-in**: For Coder-30B-A3B-Instruct workloads specifically, the orchestrator MAY add `OMP_SCHEDULE=guided,16` to the role's env vars for a +3.6% lift. This would be a Tier 3 orchestrator-config change (currently deferred indefinitely per user direction post-NUMA_MIRROR closure).

## What was NOT decided (libomp gap)

The original handoff promised libgomp **and** libomp evaluation. This bundle ran libgomp only.

- libomp.so.5 runtime IS installed on the host (`/lib/x86_64-linux-gnu/libomp.so.5` from `libomp5-20`).
- LD_PRELOAD substitution from libgomp to libomp at runtime FAILS catastrophically: smoke test produced 0.35 t/s (vs 47-48 baseline) — symbol conflicts between libgomp (linked into the binary at build time) and libomp (LD_PRELOAD'd). Mixing two OpenMP runtimes is not a working strategy.
- A clean libomp build requires either (a) `clang-20` compiler installed (`apt install clang-20`), or (b) GCC linker manipulation flags (fragile; risks symbol conflicts).
- **Sandbox blocked the apt install** (system-package modification needs user authorization). Surfacing for explicit decision.

If the user authorizes the install, Phase 2.1-libomp can complete in ~2-3 hours (build_libomp/ dir + replicate Phase A/B/C subset under libomp + comparative summary). If the user prefers to skip, Phase 2.1 closes with the partial scope: "libgomp affinity + chunks {1,4,8,16} matrix complete; libomp comparison explicitly DEFERRED — `apt install clang-20` not authorized this session".

## Closure scope

**Closed for libgomp**:
- Phase A affinity matrix (existing 2026-04-26-cpu21/) — `spread+cores+active` is universal +3-8% deployable stack.
- Phase B chunks 1/4 (existing 2026-04-26-cpu21/) — within noise.
- Phase B chunks 8/16 (this bundle) — `guided,16` is model-specific +3.6% on Coder-30B Q4_K_M; per-role opt-in candidate, not default.
- Phase C wait policy (existing 2026-04-26-cpu21/) — `passive` is a deployment trap; `active` is +0.5%.

**NOT closed**: libomp comparison (deferred pending toolchain authorization).

**NOT closed**: Phase 2.6 dense/hybrid sanity check on the affinity stack (covered by Phase 2.6 separately).

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 2.1 (this bundle) and Phase 2.6 (dense sanity).
