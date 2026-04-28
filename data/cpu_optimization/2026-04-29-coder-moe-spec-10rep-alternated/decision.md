# Coder-30B MoE-Spec 10-rep alternated — Decision: NOT DEPLOYABLE

## Verdict

**NOT DEPLOYABLE** on v5 PGO. Closure scope: this run plus all earlier evidence (gcc +7.3%, PGO single-B -43%, PGO mixed-B parity / +84% / +27% / -9%) confirms Coder MoE-Spec result is fundamentally noise-dominated on this system at this build.

## Aggregate

- Mean B=0 across 5 trials: 180.16 t/s (range 154-216, 40% spread)
- Mean B=64 across 5 trials: 163.81 t/s (range 91-200, 117% spread)
- Per-trial Δ: -1.5%, +17.9%, -23.6%, +18.9%, -48.8%
- Mean Δ: -9% with ±27% trial-to-trial std
- Trial 5 B=64=91 is outlier (~half normal)

## Why noise-dominated on Coder vs robust on REAP

- Coder mask-construction overhead (~6 ggml ops/layer × 48 layers) is ~5-10% of per-token compute
- REAP same overhead is ~1-2% of per-token compute (heavier per-token work)
- MoE-Spec union-shrinkage savings dominate cleanly on REAP, compete with overhead on Coder
- System noise (megasync cycling, page-cache shifts) amplifies the marginal signal on Coder

## Production decision

- **Coder-30B**: keep `LLAMA_ARG_MOE_SPEC_BUDGET=0` (off). NO MoE-Spec env override in registry.
- **REAP-246B**: deployable at `LLAMA_ARG_MOE_SPEC_BUDGET=40` (+13-15% pp32 robust).
- Q8 frontdoor (Qwen3.6-35B-A3B Q8): pending Phase 3 #3 measurement.

This matches the existing `decision_v5_FINAL.md` recommendation. Phase 3 #2 confirms via 10-rep alternated method that the recommendation is correct.

## Closure-inflation policy compliance

> "Coder-30B MoE-Spec result is fundamentally noise-dominated under v5 PGO build with current system noise floor (megasync cycling at ~100% on 1 core; trial-to-trial PGO baseline range 154-216 = 40% spread). 10-rep alternated method does not extract a consistent signal direction; per-trial deltas span -49% to +19%. Does NOT generalize to 'Coder MoE-Spec is dead on all hardware/builds' — different system noise floor (e.g., bare-metal isolated machine) or workload-shape (longer prompts, multi-tenant) could produce different results. The structural finding (mask overhead/savings ratio is marginal on smaller models) is well-established and unlikely to flip."
