# FG-1 fine-grain replay — six-arm SWE40 sealed artifacts (2026-07-27)

Deterministic replay/transform only — zero inference. Sources: `final-4arm-v4-tail-replay-20260727T080703Z` (A3/A4/A1/Laguna), `expanded-six-arm-v4-tail-replay-20260727` (A3-tc/A3-ff), `fable-swe-tail-sealed-20260727T094334Z` (FF MTP trio). Machine-readable: `fg1_results.json`. Observation-grade fine-grain read per MEASUREMENT_POLICY *Deterministic replay before regeneration*.

## Headline findings

1. **TC's "empty patches" are think-truncations, not declines.** 15 of 16 empty-patch failures hit the 3072-token cap inside the reasoning channel (`finish_reason=length`, response 0 chars, median reasoning 8,670 chars); the 16th is a single `skipped_missing_path` apply failure. TC as-configured is the **most expensive** arm: 5,021 tokens/solved vs A3's 1,506 (3.3×), median completion 2,406 vs 424. The token-efficiency thesis is not falsified — it was never exercised: the model spends its budget thinking. **Remediation candidates (FG-3): larger cap and/or no-think template** — rescue ceiling is 15 instances (18 → up to 33 if precision-on-attempts ~0.75 held).
2. **FF is the actual token-efficiency winner.** Same-harness trio: FF-non-MTP median 237 completion tokens vs stock 397 (−40%) at −1 solve (19 vs 20); across the authority table, FF tokens/solved 1,083 vs A3 1,507 (−28%), quality statistically tied (McNemar +2/−6, p=0.29). FF-MTP is leaner still (median 233, total 18,310) but its LCB weakness (19 vs 25/28) stands.
3. **Laguna SWE-route specialist is DEAD.** Laguna's 17 solves are a strict subset of A3∪A4 (unique-vs-A3∪A4 = 0; unique-vs-all-five = 0). A3 dominates it one-sidedly: +6/−0 discordant, exact p=0.031 — the only significant pair at n=40. Routing SWE tasks to Laguna adds zero coverage.
4. **The Laguna speed argument inverts (FG-4).** The coding-specialist read compared Laguna (~40 tok/s) against "A4 registry baseline 24.3 tok/s" — that row is dated 2026-05-04 (pre-v6/v8 kernels, MTP-blind). In the same sealed capture window: **A4 median 94.5 tok/s** (p10 81.8, p90 114.8) vs Laguna 44.6, A3 52.7, A1 55.2. Laguna is ~2× *slower* than the incumbent, not faster. Registry `performance` row for the 35B needs a protocol-cited refresh (24.3 → stale).
5. **Discriminating hard core**: 14/40 instances unsolved by all six arms (list in `fg1_results.json → unsolved_by_all_six`) — the natural focused-test set for future architect benches. A3 keeps 3 unique solves; TC has 1 (`scikit-learn-11310`).

## Tables

Pairwise solve overlap (diagonal = resolved):

|        | A3 | A3-ff | A3-tc | Laguna | A1 | A4 |
|--------|----|-------|-------|--------|----|----|
| A3     | 23 | 17    | 15    | 17     | 13 | 13 |
| A3-ff  |    | 19    | 16    | 15     | 14 | 12 |
| A3-tc  |    |       | 18    | 12     | 12 | 10 |
| Laguna |    |       |       | 17     | 12 | 12 |
| A1     |    |       |       |        | 15 | 10 |
| A4     |    |       |       |        |    | 13 |

Token economics (all 40 rows/arm):

| Arm | resolved | empty | truncated | median compl. tok | tokens/solved | median decode tok/s |
|-----|----------|-------|-----------|-------------------|---------------|---------------------|
| A3 | 23 | 5 | 5 | 424 | 1,506 | 52.7 |
| A3-ff | 19 | 5 | 2 | 237 | 1,083 | — (trio 28.9) |
| A3-tc | 18 | 16 | 15 | 2,406 | 5,021 | — (capture 30.5-class) |
| Laguna | 17 | 11 | 9 | 291 | 2,468 | 44.6 |
| A1 | 15 | 6 | 2 | 323 | 1,578 | 55.2 |
| A4 | 13 | 14 | 4 | 355 | 2,618 | 94.5 |

McNemar discordants: TC-vs-A3 +3/−8 (p=.23); FF-vs-A3 +2/−6 (p=.29); Laguna-vs-A3 +0/−6 (**p=.031**); Laguna-vs-A4 +5/−1 (p=.22).

## Consequences filed

- FG-3 is now concrete: TC re-run at wider cap / no-think — cheap 40-instance validation, post-campaign.
- FG-2 gains prior: Laguna also truncates on SWE (9 rows) — chronic verbosity, same remediation family.
- Laguna's remaining case: L-Q4 quant axis + non-coding suites (FG-5) + nothing else.
- A4 registry perf row stale → protocol-cited re-measure task.
