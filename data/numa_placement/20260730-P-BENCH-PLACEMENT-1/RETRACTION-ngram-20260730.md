# RETRACTION — the ngram-mod speculative result (2026-07-30)

**Status: RETRACTED, same day, before any recipe changed.**

This directory contains raw evidence for two claims committed earlier today:

| commit | claim |
|---|---|
| `f36483cd` | "ngram-mod composed with draft-mtp is **2.52×** on the 35B at depth" |
| `bd1f086e` | "ngram wins where the drafter is weak — confirmed on realistic text" (**2.80×**) |

**Both are withdrawn.** The measured effect was a harness artifact, not a property
of `ngram-mod`. The raw logs and result files are deliberately left in place,
unedited — they are a correct record of what the harness produced. Only the
interpretation was wrong, and this file supersedes it.

## Mechanism

Each cell launched **one server** and sent the **same prompt** r times at
`temperature=0.3, seed=42`:

1. Run 1 generates answer X on a cold context.
2. The server retains X in the slot's KV cache.
3. Run 2 sends the identical prompt. It hits the prompt cache
   (`prompt eval time = ... / 4 tokens`) and — critically — `ngram-mod` drafts by
   matching text **already present in the context**, where X is now sitting.
4. The model drafts its own previous answer verbatim. Mean accepted draft length
   goes `3.58 → 15.88` tokens; acceptance reaches `1.000`.

The reported statistic was the median of (cold, warm). At r=2 the median *is* the
mean, so a single inflated run moved the published number by half its error.

## The control was already in the data

Sorting all 68 cell logs by `run2 ÷ run1` separates them cleanly:

* **Every cell inflated above 1.15× is an `ngram` arm.** Worst: 5.12×.
* **All 25 `draft-mtp`-only and `none` cells sit flat between 0.92× and 1.08×.**

`draft-mtp` drafts from model weights, so a warm context cannot help it. It was a
free, built-in control the whole time.

```
Qwen3.6-27B Q8_0, GPU, 16.5k ctx     run1     run2   mean draft len   published
  draft-mtp   (control)             39.20    39.42    3.58 -> 3.58      39.31  sound
  ngram + draft-mtp                 39.61   105.21    3.58 -> 15.88     72.41  INFLATED
```

Run 1 of the two arms is 39.20 vs 39.61 — statistically identical. `ngram-mod`
contributed nothing.

## Corrected result — run 1 only

| model · quant | device / depth | draft-mtp | + ngram-mod | real gain |
|---|---|---|---|---|
| Qwen3.6-35B-A3B Q8_0 | CPU full @14.1k | 24.86 | 24.85 | **−0.0 %** |
| gemma-4-26B-A4B Q4_K_M | CPU full @16.5k | 27.01 | 27.06 | +0.2 % |
| Qwen3.5-122B-A10B Q4_K_M | CPU full @14.1k | 15.76 | 16.03 | +1.7 % |
| Qwen3.6-27B Q8_0 | GPU @2k…32.4k | 47.79…31.97 | 47.80…31.98 | 0.0 % … +1.0 % |
| Qwen3.5-122B UD-IQ2_M | GPU @16.5k | 38.48 | 31.77 | **−17.4 %** |

Across 16 model × depth × device cells the gain spans **−17.4 % to +2.7 %,
centred on zero**. `ngram-mod` *alone* is much worse than MTP nearly everywhere
(122B on CPU: 9.38 vs 15.76 tok/s).

**No role should enable `ngram-mod` on this evidence.** The `acceleration.spec_type`
field change that was queued off the original finding is cancelled.

## What the prompt screening missed

The prompts were screened for degeneracy and passed: real repo source and docs at
6–17 % repeated 5-grams, versus the 99.7 % of an earlier synthetic-filler attempt
that was correctly discarded. The screening was **necessary but not sufficient** —
the contamination came from the *generation*, not the prompt. A clean prompt
cannot protect a context-reading drafter from copying the model's own prior answer.

## Amendments to P-BENCH-PLACEMENT-1

Any arm whose drafter reads from context (`ngram-mod` today; anything
retrieval- or context-matching in future) must satisfy all four:

1. **No repeated-prompt replication against a live server.** Restart the server
   per rep, erase the slot KV between reps, or give each rep a distinct prompt.
2. **Include a non-context drafter as a control arm.** Its flatness across reps is
   the diagnostic; its absence is why this took a day to find.
3. **Report n, min and max — never a bare median.** Where spread *was* printed, the
   artifact was visible on sight: the VL-7B ngram cell reads
   `min=7.42 max=19.01` against a `none` arm of 7.31.
4. **Treat `accept = 1.000` as a bug report, not a result.** Perfect acceptance
   means the continuation was already in the context.

Cross-ref: `handoffs/active/speculative-decoding-mtp-refresh.md` (NG1–NG5 filed
against the retracted finding — re-scope before executing).
