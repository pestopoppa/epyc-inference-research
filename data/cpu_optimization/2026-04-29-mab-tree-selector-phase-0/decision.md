# MAB Tree-Shape Selector — Phase 0 Decision: NO-GO

## Verdict

**NO-GO** with narrow closure scope. Phase 1 (MAB selector implementation) deferred indefinitely.

## Headline measurements (5-rep proper canonical, v5 PGO, megasync noise floor)

### Coder-30B Q4_K_M end-to-end (3-prompt × 2-rep, rep0 lost to server warmup)

| Shape | t/s mean ± std | accept% | Δ vs linear |
|---|---|---|---|
| linear (p_split=0) | 33.14 ± 4.52 | 64.1 ± 6.4 | reference |
| tree (p_split=0.05) | 27.17 ± 12.77 | 64.1 ± 6.4 | **-18% (rep1: -39%, rep2: parity)** |

### REAP-246B Q4_K_M end-to-end

| Shape | t/s mean ± std | accept% | Δ vs linear |
|---|---|---|---|
| linear (p_split=0) | 7.69 ± 0.37 | 58.0 ± 2.6 | reference |
| tree (p_split=0.05) | 7.80 ± 0.27 | 58.0 ± 2.6 | +1.4% (within noise band) |

### Forward-pass baseline (no spec-dec)

- Coder pp32: 201.14 ± 5.99 t/s (megasync floor; clean baseline morning was ~379)
- REAP pp32: 46.57 ± 1.14 t/s

## Critical finding: tree at temperature=0 produces bit-identical output to linear

Direct comparison of `comp_coder_linear_rep1.json` vs `comp_coder_tree_rep1.json`:
- Both: `predicted_n=256, draft_n=230, draft_n_accepted=158`
- Both: identical `content` first 100 chars (verified byte-by-byte)
- linear: `predicted_ms=8550`
- tree: `predicted_ms=14116` (+65% wall-clock for ZERO output benefit)

**Root cause**: at temp=0 greedy decoding, the verifier accepts the highest-probability path through the tree. Non-greedy branches consume drafter+verify compute but are always rejected by the verifier. Tree branching is structurally wasted work at temp=0.

## Closure scope (narrow, per closure-inflation policy)

> "DySpec heap-spec tree at `--draft-p-split=0.05` with greedy decoding (temperature=0) produces bit-identical outputs to `--draft-p-split=0` linear baseline (verifier collapses tree to greedy path) while adding wasted draft+verify work on non-greedy branches. End-to-end on v5 PGO build with megasync noise floor: Coder-30B Q4_K_M -18% mean (high variance ±48% CV), REAP-246B Q4_K_M +1.4% within noise band. The MAB selector over the paper's arm pool `(3,3,2,1)`, `(3,2,2,1,1)`, `(2,2,2,1,1,1)` cannot recover headroom that is structurally absent at temp=0 — selecting different shapes does not help when the verifier discards all non-greedy paths regardless of shape."

**Does NOT generalize to**:
- Higher-temperature sampling (paper used temp=0 too but on different stack; their +13.7% claim may not generalize across hardware/decoder topologies)
- Different arm pools (e.g., depth-1 wide-K shapes for top-K fallback)
- Multi-tenant/concurrent workloads
- Sampling-decoding regimes (production currently uses temp=0)

## Confirmation of existing production wisdom

This NO-GO confirms the registry's existing config:
- `model_registry.yaml:378` for Coder-30B: `p_split: 0   # linear only, tree is net-negative at 48t (sweep-verified)`
- `model_registry.yaml:447` for REAP-246B: `p_split: 0   # linear only — tree harmful at all ps values, sweep-verified 2026-03-26`

The 2026-03-21/26 sweeps were correct under greedy-temp inference; v5 PGO confirms.

## Reopen criteria (NOT current Phase 1, but documented for future)

- Production workload shifts to temp>0 sampling
- A different MAB arm pool optimized for greedy-temp inference is proposed (e.g., 1-deep wide-K)
- Pythia-6.9B-class falsification on our hardware to verify whether paper's Pythia +13.7% is fully reproducible (separating "paper measurement methodology" from "tree-spec mechanism")

## Pre-prod gate release

Per `moe-spec-cpu-spec-dec-integration.md` pre-prod gate blockquote: this NO-GO verdict is a **written GO/NO-GO** (binary), satisfying gate condition (a). With Workstream B verdict written (GO with revised scope) satisfying gate condition (b), the gate **RELEASES** for production registry integration.
