# Judge-suite head-to-head — architect_general (27B) vs frontdoor (35B-A3B)

**Date:** 2026-08-02 · **Campaign:** `judge_suite_headtohead_20260802`

## Headline

On the canonical judge suite the two models are **statistically indistinguishable**,
and the suite **cannot resolve them** — which is the finding that matters.

| | architect_general (Qwen3.6-27B Q8_0) | frontdoor (Qwen3.6-35B-A3B Q8_0) |
|---|---|---|
| paired total (0–3) | **180 / 204 (88.2%)** | **179 / 204 (87.7%)** |
| per-question wins | 9 | 4 |
| ties | \multicolumn{2}{c}{55} |

Exact sign test on the 13 discordant pairs: **p = 0.267**, not significant at α = 0.05.

## Why the null result is about the instrument, not the models

**50 of 68 questions (74%) scored a perfect 3 for BOTH arms** and carry zero
discriminating information.

| suite | both-perfect | informative |
|---|---|---|
| general | 10/10 (100%) | none |
| thinking | 9/10 (90%) | almost none |
| math | 8/9 (89%) | almost none |
| coder | 7/9 (78%) | little |
| instruction_precision | 7/11 (64%) | some |
| agentic | 6/10 (60%) | some |
| tool_compliance | 3/9 (33%) | the only real signal |

Published benchmarks separate the same two models cleanly on **8 of 8** axes
(27B ahead everywhere; Terminal-Bench +7.8, SWE-bench +3.8, LiveCodeBench +3.5).
The suite is at ceiling; the published benchmarks are not. See
`epyc-orchestrator/orchestration/public_benchmarks.yaml`.

## Protocol

Both arms captured under an **identical, verified** configuration:

* same 70 questions (identical `(suite, question_id)` sets — checked, not assumed)
* identical per-suite `max_tokens` (agentic/coder/general/math/thinking 4096,
  tool_compliance 2048, instruction_precision 512), multiplier 1
* `temperature` suite-declared, `seed=42`, `enable_thinking=false`
* frontdoor held to the 27B's **8192-token per-slot budget** via `--ctx-budget 8192`
  even though its slot offers 16384, so neither arm got a larger budget
* the same 9 `long_context` questions blocked in both arms (prompt + max_tokens
  exceeds one slot)

Serving shape was **probed from `/props`**, not assumed: the harness previously
carried a hardcoded shape pinned to `architect_general/:8083`, so pointing it at
another server planned against the wrong slot while talking to a different model.

### Judging

Two independent judgements are preserved here:

1. **Claude-as-Judge** (`arm_architect_general_27b/judge_out/*.csv`) — the anchor
   protocol. 27B scored 165/183 (90.2%).
2. **Local same-judge head-to-head** (`headtohead_local_judge/`) — architect_critic
   (Qwen3.5-122B-A10B, :8074), neither arm, scoring BOTH captures under one rubric
   with per-question arm order shuffled on a fixed seed. This is the number quoted
   above.

The two judges **disagree on direction** (Claude had frontdoor ahead 92.9% to
90.2%; the local judge has the 27B ahead by 0.5 pp) — further evidence there is no
signal here to recover.

## Known defects in the instrument (recorded, not corrected in this data)

* **Contaminated rubric.** `rubric_system_prompt` embeds calibration examples that
  name specific `question_id`s together with scores from other models — including
  `math/t3_q2_combinatorics` as a *score-1* exemplar and `coder/t1_q1_algorithm` as
  *score-0*. The judge is primed on question identity before reading the answer.
  Preserved verbatim because it is the anchor's instrument and both arms receive it
  identically, so it cannot bias the A-vs-B contrast; it does bias absolute level.
* **Two false suite questions, since fixed** (suite v2, 2026-08-02, not reflected in
  this capture):
  - `math/t3_q2_combinatorics` asserted `sum (-1)^k C(n,k) C(2n-k,n) = C(n,⌊n/2⌋)`.
    The LHS is identically **1**; it agrees only at n=1. The 27B *proved this* and
    was scored 2/3 for not constructing an involution for a false identity.
  - `coder/t1_q1_algorithm` claimed TWO bugs in code containing ONE (verified
    exhaustively: 57,915 cases, 0 failures with only `left=mid+1` applied).
  Both answer keys documented the defect and shipped anyway.
* **3 of 140 judgements** returned JSON without an integer score and were counted
  **ineligible** — excluded from both numerator and denominator, never averaged
  away: `coder/t3_q3_algorithmic_hardness` (both arms), `math/t3_q1_analysis`
  (frontdoor).

## Layout

```
arm_architect_general_27b/    27B capture + Claude-as-Judge CSVs + report.json
arm_frontdoor_35b_a3b/        frontdoor capture at the matched 8192 budget
headtohead_local_judge/       same-judge scoring of BOTH arms + summary.json
run_judge_suite.py            capture harness (probes shape; never launches servers)
judge_local.py                same-judge scorer
SHA256SUMS                    integrity manifest
```

Verify: `sha256sum -c SHA256SUMS`
