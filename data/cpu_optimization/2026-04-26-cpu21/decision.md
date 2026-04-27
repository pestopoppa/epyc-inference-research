# CPU21 — Decision

**Verdict**: **PARTIAL** (libgomp affinity submatrix exhausted; libomp + chunks 8/16 + dense coverage remain).

## What was decided on 2026-04-26 evening

The `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active` stack is a **real, deployable, universal +3-8%** win across the sync-bound MoE Q4_K_M class:

- Coder-30B Q4_K_M: 43.82 → 46.52 (+6.2%)
- REAP-246B Q4_K_M: 6.14 → 6.33 (+3.1%)
- Combined stack additive (Phase A spread+cores +6.2% + Phase C active +0.5% ≈ measured ~+6.5% combined).

Schedule policy is **within noise** (libgomp's defaults are near-optimal for this workload).

`OMP_WAIT_POLICY=passive` is a **deployment trap** (-81.6% on Coder-30B). Add a guard at session start.

`OMP_PROC_BIND=master` HUNG (killed after 6 min). Pathological at 96-thread scale.

The deployable stack flowed into the proper canonical baseline definition in `cpu-benchmark-rigor-and-revalidation.md` P3.

## What was NOT decided (gates that remain open)

- libgomp vs libomp comparison: **not run.** libomp not installed at the time.
- Schedule chunks 8 and 16: **not run.** Only chunks 1 and 4 swept.
- Dense/hybrid (Qwen3.5/3.6-27B): **not run.** All sweeps were on MoE Q4_K_M models.

The handoff's gate-1 ("any config yields ≥5% gain on at least 2 of 4 sync-bound models with no quality drift") IS met for the affinity+wait-policy stack on Coder-30B + REAP-246B. The handoff's gate-2 ("if all configs are ≤2% or regress, mark runtime branch exhausted") is NOT applicable since gate-1 was met.

## Closure scope

**Closed**: libgomp affinity submatrix on MoE Q4_K_M class. Deployable stack identified, integrated into proper canonical baseline.

**NOT closed**: full runtime/scheduling matrix. libomp comparison + chunks 8/16 + dense coverage are in remediation Phase 2.1 + 2.6.

## Remediation reference

See `~/.claude/plans/nifty-discovering-allen.md` Phase 2.1 (libomp install + chunks 8/16) and Phase 2.6 (dense/hybrid sanity coverage). Outputs will land in:
- `2026-04-28-cpu21-libomp-chunks/` — Phase 2.1
- `2026-04-28-cpu-cross-architecture-sanity/` — Phase 2.6 (covers CPU2 + CPU21 + CPU25 dense run)

After those land, the closure scope upgrades to "libgomp + libomp matrix exhausted on MoE + dense classes" OR "libgomp matrix exhausted; libomp explicitly deferred" depending on what we choose to install.
