# SWE-bench Verified — PRELIMINARY run, 2026-07-24 · **SUPERSEDED**

> **These are NOT the ratified numbers.** The authority for arm A3 is
> `orchestration/model_registry.yaml` → `swe_verified_pct: 57.5`,
> `swe_verified_raw: "23/40"` ("frozen 4-arm authority arm A3"). Cite that, not
> these files.

## Why they are kept

They sat **untracked in the repository root** until 2026-08-02, where an
untracked file is indistinguishable from a committed one to anyone reading the
tree — and these disagree with the ratified figure. Archived rather than deleted
so the discrepancy is explained once instead of being rediscovered.

## The discrepancy

| | this run (2026-07-24) | ratified authority |
|---|---|---|
| resolved | **21**/40 (52.5%) | **23**/40 (57.5%) |
| empty patch | 8 | 5 |

Different runs of the same arm, not a transcription error: the empty-patch count
moves with the resolved count (8 vs 5), which is what a re-run with different
generation outcomes looks like. The ratified run is the later, sealed one.

## Contents (all `schema_version: 2`, `total_instances: 500`, 40 submitted)

| file | arm | resolved | unresolved | empty patch | completed |
|---|---|---|---|---|---|
| `A1_122b_iq2.swe-eval-A1.json` | Qwen3.5-122B-A10B IQ2 | 15 | — | 6 | — |
| `A3_27b_dense.swe-eval-A3.json` | Qwen3.6-27B dense | 21 | 11 | 8 | 32 |
| `A4_35b_a3b.swe-eval-A4.json` | Qwen3.6-35B-A3B | 14 | — | 13 | — |

`total_instances: 500` is the full SWE-bench Verified set; only **40** were
submitted, so every rate here is out of 40 and is **not** comparable to a
published full-set SWE-bench Verified number.

## What they still show

Even as a superseded run, the ordering is consistent with every other instrument:
**27B (21) > 122B-IQ2 (15) > 35B-A3B (14)** on an execution-verified benchmark —
the same direction the published benchmarks give, and the opposite of the
canonical judge suite's null result (which is at 74% ceiling saturation; see
`data/judge_suite_headtohead_20260802/`).

Verify: `sha256sum -c SHA256SUMS`
