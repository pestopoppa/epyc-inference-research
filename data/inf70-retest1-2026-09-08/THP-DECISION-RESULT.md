# CHAMP-2 THP shim — DECISION-GRADE RESULT: **LIKELY IMPROVEMENT, KEEP**

Pre-registered plan `PREREG-THP-DECISION.md`, frozen 14:08:05Z (sha256 `337200315c6b…`) **before the
lock**. Boundary implemented in `tools/decide.py`, unit-tested before the run (6/6 calls, 5/6 does
not, 9/10 calls, 8/10 does not). Region held 14:10:22Z–14:52:54Z.

## The call

**Early-stop boundary fired at LOOK 1: 6/6 pairs ON-faster.** Exact two-sided alpha of the complete
two-look plan = **0.0430**, by enumeration of all 2^10 sign sequences with nested stopping handled
(a union bound would have said 0.078 — over budget, and wrong).

| pair | order | OFF t/s | ON t/s | effect | sign |
|---:|---|---:|---:|---:|---|
| 1 | OFF first | 26.5964 | 27.8845 | **+4.843%** | ON faster |
| 2 | ON first | 26.4504 | 27.9362 | **+5.617%** | ON faster |
| 3 | OFF first | 26.1430 | 27.6942 | **+5.934%** | ON faster |
| 4 | ON first | 26.5689 | 28.0738 | **+5.664%** | ON faster |
| 5 | OFF first | 27.8001 | 27.8892 | **+0.321%** | ON faster |
| 6 | ON first | 27.6036 | 27.7577 | **+0.558%** | ON faster |

Order-balanced 3 OFF-first / 3 ON-first at the look, so direction is not confounded with position.
Every session passed both contention screens and the fail-closed `THP_enabled` assertion
(**1 in all six OFF launches, 0 in all six ON launches**). No pair was dropped or replaced.

## The magnitude is NOT resolved — and the reason is the interesting part

Effects split cleanly: **+4.8 to +5.9% in pairs 1–4, +0.3 to +0.6% in pairs 5–6.** Median +5.23%,
range +0.32% to +5.93%. This design tested direction and did not size for magnitude; no number is
claimed.

But the pattern is not noise, and it is the finding:

| | mean t/s | **sd** | range |
|---|---:|---:|---:|
| **OFF** (shim off, champion default) | 26.8604 | **2.510%** | **6.17%** |
| **ON** (shim active) | 27.8726 | **0.481%** | **1.36%** |

**Variance ratio OFF/ON = 25.3x.** The ON launches sit in a 1.4% band; the OFF launches span 6.2%.
**The pairs with the largest effect are exactly those where the OFF launch was slow — the ON launch
never was.** The shim is not simply adding throughput; it appears to be **removing a downside tail
in launch-to-launch variability**, pinning the champion near the top of its own range.

That is mechanistically coherent for a `PR_SET_THP_DISABLE`: transparent hugepage backing and
khugepaged collapse are exactly the kind of per-launch, page-state-dependent variation that would
produce a slow launch sometimes and not others. Stated as a **hypothesis consistent with the data,
not a demonstrated mechanism** — n=6 per side, and variance ratios are noisy at that n.

## This bears directly on the champion-divergence flag

`CHAMPION-LAUNCH-TABLE.md` records the champion configuration measured across **9 launches spanning
3.7 hours with a between-launch sd of 5.081% and a 12.55% range** — and **every one of those
launches was shim OFF**. The present result says shim-OFF launches carry ~5x the launch-to-launch
variance of shim-ON ones.

**So a plausible partial explanation for the champion's instability — including the unexplained +9%
step between 12:20 and 12:33 — is that we have been characterising the champion in its
high-variance configuration.** This is a lead, not a conclusion: the 9 launches were not paired,
and this test's n is small. It is the first cheap thing to check if the divergence is pursued.

If it holds, it also lowers the cost of a final characterisation substantially: at the ON sd of
0.481%, a +/-1% champion headline needs **~1 launch** rather than the ~100 the OFF sd implies.
**That number must not be quoted until it is confirmed at launch granularity** — it is an
extrapolation from six launches, and quoting it now would be the same unit-and-precision error this
campaign has already made three times.

## Interpretation of "cannot tell", which did not occur

Had this returned CANNOT TELL, the report would have distinguished a **weak effect** from
**instrument instability** — the two have different follow-ups. It did not arise: 6/6 is maximal
consistency, and it fired at the first look.

## What was NOT done

* No magnitude estimate, by design.
* No correctness gate specific to shim-ON vs shim-OFF output (the shim changes page backing, not
  arithmetic; every champion-state comparison across the campaign was 24/24 byte-identical).
* No prior data reused — neither the earlier 4v4 THP block nor the characterisation `CP` arms.
  Excluded for three independent reasons recorded in `PREREG-THP-DECISION.md` s5.
