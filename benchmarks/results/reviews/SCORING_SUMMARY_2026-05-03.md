# Claude-as-Judge Benchmark Scoring Summary — 2026-05-03

**Run Date:** 2026-05-02 (sweep launched after preflight gate cleared)
**Scoring Date:** 2026-05-03
**Run ID:** `20260430_151713`
**Benchmark Suite:** Hardened V2 (6 suites: agentic, coder, general, instruction_precision, math, thinking)
**Methodology:** Claude-as-Judge per `/benchmark` skill — 0-3 rubric per question, scored in-context by Claude (one Claude subagent per config, 11 in parallel). CSVs at `benchmarks/results/reviews/may2_run/`.

## Rubric (0-3)

- **3** — Correct, complete, well-structured
- **2** — Mostly correct but incomplete (truncation) OR minor errors
- **1** — Partially relevant; wrong approach or major errors
- **0** — Wrong, hallucination, entirely-in-`<think>`, empty, or off-topic

## Overall ranking

| Rank | Config | Total | % | Pass (≥2) | #Q | Avg t/s | Notes |
|------|--------|-------|---|-----------|----|---------|-------|
| 1 | MiniMax-M2.7 Q8_0 | 24/24 | **100.0%** | 8/8 (100%) | 8 | 10.7 | **only `general` ran**; not directly comparable |
| 2 | Nemotron-Nano-9B-v2 Q8 | 80/81 | **98.8%** | 27/27 (100%) | 27 | 13.3 | 3 suites; one minor agentic ding |
| 3 | Qwen3.6-27B Q8_0 (dense) | 123/126 | **97.6%** | 41/42 (98%) | 42 | 4.5 | full-precision dense reference |
| 4 | Qwen3.6-27B Q4_K_M (dense) | 141/147 | **95.9%** | 48/49 (98%) | 49 | 6.7 | strong; only 1 sub-pass |
| 5 | Qwen3.6-35B-A3B Q8_0 (baseline) | 164/177 | **92.7%** | 55/59 (93%) | 59 | 23.6 | strong reasoning + format |
| 6 | SuperGemma-4 31B Q4_K_M | 168/183 | **91.8%** | 56/61 (92%) | 61 | 6.9 | rounded, no zeros |
| 7 | Qwen3-4B-Instruct-2507 Q8 | 79/90 | **87.8%** | 26/30 (87%) | 30 | 27.8 | 3 suites; agentic schema-discipline gap |
| 8 | Qwen3-Coder-30B-A3B Q4_K_M | 153/183 | **83.6%** | 53/61 (87%) | 61 | 43.4 | proof + constraint weakness |
| 9 | Ring-mini-linear-2.0 Q4_K_M | 63/90 | **70.0%** | 22/30 (73%) | 30 | **77.8** | 4× max-tokens cap on constraint search |
| 10 | Qwen3.6-35B-A3B Q8_0 MoE=4 | 18/183 | **9.8%** | 3/61 (5%) | 61 | 18.2 | **catastrophic** — MoE reduction broke generation |
| 11 | Qwen3.6-35B-A3B Q8_0 MoE=6 | 8/183 | **4.4%** | 2/61 (3%) | 61 | 18.5 | **catastrophic** — even worse than MoE=4 |

## Per-suite breakdown

```
Model                                    agentic         coder         general          instr_prec        math       thinking
MiniMax-M2.7 Q8_0                          —             —          24/8  (100%)         —              —             —
Nemotron-Nano-9B-v2 Q8               29/10  (97%)        —          27/9  (100%)         —              —        24/8 (100%)
Qwen3.6-27B Q8_0 (dense)             23/8   (96%)   15/5 (100%)     21/7  (100%)    28/10 (93%)    15/5 (100%)  21/7 (100%)
Qwen3.6-27B Q4_K_M (dense)           28/10  (93%)   18/6 (100%)     21/7  (100%)    27/10 (90%)    24/8 (100%)  23/8 ( 96%)
Qwen3.6-35B-A3B Q8_0 (baseline)      29/10  (97%)   27/10 (90%)     27/9  (100%)    25/11 (76%)    27/9 (100%)  29/10 (97%)
SuperGemma-4 31B Q4_K_M              28/10  (93%)   27/10 (90%)     28/10 ( 93%)    29/11 (88%)    28/10 (93%)  28/10 (93%)
Qwen3-4B-Instruct-2507 Q8            22/10  (73%)        —          30/10 (100%)         —              —        27/10 (90%)
Qwen3-Coder-30B-A3B Q4_K_M           26/10  (87%)   23/10 (77%)     29/10 ( 97%)    23/11 (70%)    27/10 (90%)  25/10 (83%)
Ring-mini-linear-2.0 Q4_K_M          21/10  (70%)        —          23/10 ( 77%)         —              —        19/10 (63%)
Qwen3.6-35B-A3B Q8_0 MoE=4            2/10  ( 7%)    1/10 ( 3%)      6/10 ( 20%)     6/11 (18%)     0/10 ( 0%)   3/10 (10%)
Qwen3.6-35B-A3B Q8_0 MoE=6            1/10  ( 3%)    0/10 ( 0%)      1/10 (  3%)     6/11 (18%)     0/10 ( 0%)   0/10 ( 0%)
```

## Headline findings

### 1. Top of the table: Qwen3.6-27B (97.6% / 95.9%) and Nemotron-Nano-9B (98.8%)
The **dense Qwen3.6-27B** scores higher than the **MoE Qwen3.6-35B-A3B** on this suite — even at Q4_K_M (95.9% vs 92.7%). Q8 ekes out an extra ~1.7pp over Q4. **Nemotron-Nano-9B-v2 Q8** scored 80/81 across only 3 suites (skipped the long ones); per-question quality is stellar but it didn't run the full suite — not directly comparable.

### 2. ★ Qwen3.6-35B-A3B MoE expert reduction is broken (10% / 4% pass rate)
- `qwen36_q8_0_moe4`: 28 of 61 questions trapped in `<think>` block (no `</think>`); 17 produced "post-think token salad" — mojibake (Chinese/Russian/Korean/Vietnamese tokens), runaway repetition (`itecture itecture itecture...`, `googleapis.com googleapis.com...`), broken JSON.
- `qwen36_q8_0_moe6`: even worse — 30 think-trapped, 28 mojibake'd. Math and thinking suites scored **literally 0/30**.
- Only ~3 questions earn full credit per config — and those are trivial single-token outputs ("4", "NONE").
- **Hypothesis**: Qwen3.6's hybrid SSM+softmax routing breaks under expert pruning at this aggression. Earlier registry note (`feedback_omp_env_stack_required.md`-adjacent) noted MoE expert reduction is generally fine on Qwen3.6 baseline, but reducing to 4-6 experts at Q8 with the new kernel may have crossed a threshold. **Action**: stop running MoE expert reduction on Qwen3.6 at <8 experts. Revisit only with re-quantization or a different MoE schedule.

### 3. ★ Ring-mini-linear-2.0 (Lightning Attention port) — 70% with 4× max-tokens-cap on constraint-search
First production data on the new architecture (commits `33b60b925` + `c9626faf8` from 2026-04-30).
- Throughput: **77.8 t/s** — by far the fastest in the table (highest tokens/sec of any model)
- 4 of 30 responses are empty (`completion_tokens = 4096`, no visible output): all 4 are constraint-search/enumeration/Fermi problems where a reasoning model would think long. The harness either strips unfinished think blocks or stops on max-tokens before any visible answer. Clear fix: bump `max_tokens` to 8192 or 16384 for next run on this model.
- The 26 responses that DID complete are coherent: AIME-style math correct, formal logic correct, strong strategic prose. **No long-context coherence loss / no recurrence drift detected.** The arch works.
- One real logic error in `thinking/t3_q2_causal_inference` (self-contradiction inside one response) — worth watching across more samples.
- Persona/branding leakage on 2 responses ("As developed by the Ant Group's Bailing team..."). Cosmetic; not a quality issue.
- **Action**: re-run Ring-mini with `max_tokens=8192` and `--reasoning auto` (default) to recover the 4 capped questions; expected pass rate jumps to ~85%+.

### 4. Qwen3-Coder-30B-A3B (frontdoor production model) — 83.6%
Standout strengths: standard coding recipes, math computation, strong general/synthesis (29/30).
Weaknesses:
- **Rigorous proofs** (combinatorics involution, randomized-select expected comparisons, Ω(n) adversary) — model wanders, hand-waves, doesn't complete.
- **Strict-format compliance** (instruction_precision: 23/33, only 70% pass) — fails self-referential word-count, cascading length rules.
- **Prompt re-reading** — `thinking/t3_q3_reasoning_trap` misreads "test set T" as "training set" and answers the wrong question.

Note: 83.6% is below previous frontdoor scores (~90% in earlier runs). Possible explanations: this is a non-thinking model on a tier-3-heavy suite that may have shifted harder; or one of the suites has tighter graders. Next run should re-test to confirm vs noise.

### 5. MiniMax-M2.7 only ran the `general` suite (8 questions)
Scored 100% (8/8) on what ran. This is a real model improvement vs the broken-MoE pattern from April scoring (which had 81K-char training-data leakage and empty responses). Clean responses, no leakage, no `<think>`-only failures.
**Action**: Re-launch MiniMax-M2.7 with full 6-suite sweep — the harness either timed out or skipped the longer suites. Worth investigating why only `general` ran.

### 6. Qwen3.6-35B-A3B baseline (92.7%) is solid but `instruction_precision` weak
- 5 of the 11 instruction_precision items lost points. Cascading-constraint and self-referential-word-count items are systematic blind spots for ~30B-class models.
- All other suites ≥90%.

## Comparison to earlier April scoring

The previous review CSVs at `benchmarks/results/reviews/qwen36_q8_0_baseline.csv` (dated 2026-04-20) were generated against the **broken pre-reboot harness** (50% throughput regression, AOCC libomp, missing taskset). They include known issues like training-data leakage on minimax_m27 and runaway repetition. **The May 2 run's results supersede them** — different binary, different launcher, different freq state. Recommend deleting or archiving the April CSVs to avoid future confusion. New baselines should be cited from `may2_run/`.

## Recommended actions

| Priority | Action | Reason |
|----------|--------|--------|
| HIGH | Stop running Qwen3.6-35B-A3B MoE expert reduction at <8 experts | catastrophic breakdown (10%/4%); not the harness's fault |
| HIGH | Re-run Ring-mini-linear-2.0 with `max_tokens=8192` | 4 max-tokens-cap zeros are recoverable; expected +50pp on those questions |
| MED | Re-launch MiniMax-M2.7 full 6-suite sweep | only general ran (8q); insufficient for evaluation |
| MED | Re-test Qwen3-Coder-30B-A3B baseline (run 2) | 83.6% vs prior ~90%; confirm vs noise or genuine regression |
| LOW | Archive April scoring CSVs (`*.csv` at `reviews/` root from 2026-04-19/20) | Generated against broken harness, superseded by `may2_run/` |
| LOW | Investigate Ring-mini persona-leakage cosmetic issue | "As developed by Ant Group's Bailing team..." in 2 responses; SFT bleed-through |

## File map

- Per-config CSVs: `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/reviews/may2_run/<config>.csv`
- Result JSONs (source): `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/runs/20260430_151713/<config>.json`
- This summary: `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/reviews/SCORING_SUMMARY_2026-05-03.md`
