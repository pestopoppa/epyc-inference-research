# FG-1 fine-grain replay - six-arm SWE40 sealed artifacts (2026-07-27)

Deterministic replay/transform only - zero inference. Sources: `final-4arm-v4-tail-replay-20260727T080703Z` (A3/A4/A1/Laguna), `expanded-six-arm-v4-tail-replay-20260727` (A3-tc/A3-ff), and `fable-swe-tail-sealed-20260727T094334Z` (FF MTP trio). `fg1_results.json` contains every sealed input path and SHA-256. Observation-grade fine-grain read per MEASUREMENT_POLICY deterministic-replay rule.

## Headline findings

1. **TC empty patches are cap truncations during thinking-mode generation.** 15 of 16 empty-patch failures ended at the 3072-token cap; 12 have zero response chars and 3 have partial response text. The remaining failure is a single `skipped_missing_path` apply failure. TC's median reasoning text is 8670 chars. The asymmetric thinking configuration is confounded, so its 5021.1 tokens/solved versus A3's 1506.5 is diagnostic only, not comparative token-efficiency authority.
2. **FF is the banked token-efficiency leader.** Same-harness FF-non-MTP median completion tokens are 237.5 versus stock 397.5; authority-table tokens/solved are 1082.6 versus A3 1506.5. Quality is statistically tied (McNemar +2/-6, p=0.29). FF-MTP is leaner still (233.5, total 18310) but LCB-weak (19 versus 25/28).
3. **Laguna SWE-route specialist is dead.** Its 17 solves are a strict subset of A3 union A4 (unique=0); A3 dominates +6/-0, exact p=0.031.
4. **The Laguna speed argument inverts (FG-4).** In the sealed capture telemetry, A4 median decode is 94.5 tok/s (p10 81.8, p90 114.8; empirical higher quantile, zero-based ceil(q * (n - 1))) versus Laguna 44.6, A3 52.7, and A1 55.2. This remains observation-grade telemetry, not a registry replacement.
5. **Discriminating hard core:** 14/40 instances are unsolved by all six arms: django__django-10999, django__django-11087, django__django-11138, django__django-11141, django__django-11149, django__django-11211, django__django-11265, django__django-11333, django__django-11400, django__django-11433, matplotlib__matplotlib-14623, matplotlib__matplotlib-20676, sphinx-doc__sphinx-10435, sympy__sympy-11618. A3 keeps 3 unique solves; TC has 1.

## Tables

Pairwise solve overlap (diagonal = resolved):

|        | A3 | A3-ff | A3-tc | Laguna | A1 | A4 |
|--------|----|-------|-------|--------|----|----|
| A3 | 23 | 17 | 15 | 17 | 13 | 13 |
| A3-ff |  | 19 | 16 | 15 | 14 | 12 |
| A3-tc |  |  | 18 | 12 | 12 | 10 |
| Laguna |  |  |  | 17 | 12 | 12 |
| A1 |  |  |  |  | 15 | 10 |
| A4 |  |  |  |  |  | 13 |

Token economics (all 40 rows/arm):

| Arm | resolved | empty | truncated | median compl. tok | tokens/solved | median decode tok/s |
|-----|----------|-------|-----------|-------------------|---------------|---------------------|
| A3 | 23 | 5 | 5 | 424.5 | 1506.5 | 52.7 |
| A3-ff | 19 | 5 | 2 | 237.5 | 1082.6 | - (trio 28.9) |
| A3-tc | 18 | 16 | 15 | 2406.5 | 5021.1 | - (capture 28.5) |
| Laguna | 17 | 11 | 9 | 291.0 | 2468.0 | 44.6 |
| A1 | 15 | 6 | 2 | 323.5 | 1577.8 | 55.2 |
| A4 | 13 | 14 | 4 | 355.5 | 2618.2 | 94.5 |

McNemar discordants: TC-vs-A3 +3/-8 (p=0.23); FF-vs-A3 +2/-6 (p=0.29); Laguna-vs-A3 +0/-6 (p=0.031); Laguna-vs-A4 +5/-1 (p=0.22).

## Consequences filed

- FG-3 remains a clean no-think validation; the confounded TC economics cannot rank candidacy.
- FG-2 retains Laguna's SWE truncation prior, but FG-1 plus FG-4 eliminate a SWE routing case.
- Laguna's remaining case is the L-Q4 quant axis plus non-coding suites (FG-5).
- A4's registry performance row still needs a protocol-cited refresh.

## Boundaries

- No role or lineup decision follows from this replay. The equal-effort token-efficiency instrument remains open.
