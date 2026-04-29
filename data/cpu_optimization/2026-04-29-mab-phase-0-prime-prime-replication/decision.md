# MAB Phase 0'' DECISION — NO-GO across both targets

## Verdict

**NO-GO.** MAB tree-shape selector is structurally net-negative on Qwen3-Coder-30B-A3B-Q4_K_M + DRAFT-0.75B drafter and null on REAP-246B-A35B-Q4_K_M, both at v5 PGO build under random-seed sampling regime (temp=0.7) at n=90 paired.

## Headline

| Model | n_paired | Δ_pct | p-value |
|---|---|---|---|
| Coder-30B Q4_K_M | 90 | **-3.97%** | **0.0125 (significant LOSS)** |
| REAP-246B Q4_K_M | 90 | +0.34% | 0.8685 (null) |

## Combined evidence across regimes

| Regime | Coder | REAP | Verdict |
|---|---|---|---|
| Phase 0 greedy (temp=0) | byte-identical to linear | same | NO-GO |
| Phase 0' fixed-seed (temp=0.7) | byte-identical | same | NO-GO |
| Phase 0'' random-seed (temp=0.7, n=90) | -3.97% (p=0.012) | +0.34% (p=0.87) | NO-GO |

Phase 0' n=9 finding ("+9.6%" on Coder) was a low-n type-I error. At n=90 the true direction is small-negative on Coder and null on REAP.

## Operational decision

- MAB selector mechanism falsified across the tested drafter/target/workload class.
- Phase 1 implementation (~245 LOC) NOT justified — would only land an env-gated knob with no production pull.
- MAB selector handoff `mab-tree-shape-selector.md` moves to `handoffs/completed/`.
- Pre-production gate on MoE-Spec production registry integration condition (a) "MAB Phase 0 falsification probe completes with explicit GO or NO-GO" is RESOLVED via NO-GO at extended scope.

## Closure scope (per closure-inflation policy)

> "MAB tree-shape selector mechanism, tested on Qwen3-Coder-30B-A3B-Q4_K_M + Qwen3-Coder-REAP-246B-A35B-Q4_K_M targets with the Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0 drafter at v5 PGO build (clang+libomp+znver5), is structurally net-negative or null across all three tested verification regimes (greedy / fixed-seed sampling / random-seed sampling). At n=90 paired, Coder shows tree -3.97% (p=0.012); REAP shows tree +0.34% (p=0.87). Phase 1 implementation is not justified.
>
> Does NOT generalize to:
> - Different drafter (e.g., a fundamentally weaker drafter relative to its target — Pythia drafter has different uncertainty profile)
> - Different arm pool (paper-shapes are tuned for Pythia; shapes tuned to EPYC's 96-core verifier could differ — though we've now tested 2 regimes without signal)
> - Multi-tenant / batched / concurrent-slot workloads
> - Architecturally different targets (dense, hybrid SSM — only MoE Q4_K_M tested at scale)"

## Cross-references

- Parent: [`2026-04-29-mab-tree-selector-phase-0/`](../2026-04-29-mab-tree-selector-phase-0/) (Phase 0 NO-GO greedy)
- Sibling: [`2026-04-30-mab-phase-0-prime-sampling/`](../2026-04-30-mab-phase-0-prime-sampling/) (Phase 0' INCONCLUSIVE n=9)
- Handoff: `handoffs/active/mab-tree-shape-selector.md` → moving to `handoffs/completed/`
