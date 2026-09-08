# INF-70 — FINAL CHARACTERISATION: champion `ef81196d5` under the canonical adopted recipe

Plan pre-registered in `PREREG-FINAL.md`, frozen **15:05:15Z** (sha256 `1d8f4ddc…`) **before the
lock**. Region held 15:05:39Z–16:12:47Z. 18 launches, **all 24/24 rows complete**, none dropped.

**Conditions carried by every number below**: hot harness, 24-prompt production mix, token-weighted
decode, **unit = LAUNCH (one arm per launch)**, precision = **between-launch**, both contention
screens live and `screened()` applied, GPU loop down and nothing of the other session running,
baseline **`ef81196d5`**, champion knob state set explicitly, **shim ON verified per launch**.
No hot absolute is compared against a cold one.

---

## 1. HEADLINE — sign claims with bounded magnitudes

> **Under the canonical adopted recipe, the champion is faster than pristine by AT LEAST 117%
> (plain) and AT LEAST 81% (served-MTP)** — bounds are the lower ends of 95% bootstrap intervals
> over launches, not point estimates.
>
> **Served-MTP is faster than plain on the champion by AT LEAST 54%.**

| configuration | n launches | **central t/s** | between-launch sd | **95% CI on the mean** |
|---|---:|---:|---:|---|
| **champion plain, shim ON** | 6 | **27.893** | **0.609%** | ±0.487%  [27.758, 28.029] |
| **champion MTP, shim ON** | 6 | **43.281** | **0.356%** | ±0.285%  [43.157, 43.404] |
| pristine plain | 3 | 12.762 | 0.360% | ±0.408%  [12.710, 12.814] |
| pristine MTP | 3 | 23.709 | 0.926% | ±1.048%  [23.461, 23.957] |

| ratio | value | 95% CI | bounded claim |
|---|---:|---|---|
| champion / pristine, **plain** | 2.1857x | [2.1730, 2.1974] | **at least +117.3%** |
| champion / pristine, **MTP** | 1.8255x | [1.8081, 1.8399] | **at least +80.8%** |
| champion **MTP / plain** | 1.5516x | [1.5439, 1.5598] | **at least +54.4%** |

**MTP draft acceptance: 82.1%**, identical on champion and pristine (`draft_n` 3267 per launch).
Acceptance is a property of the draft head and the prompt set, not of the CPU levers.

## 2. THE RATIO IS RECIPE-TO-RECIPE, NOT KNOB-CONTROLLED

Established **before** the run, by inspecting the binaries rather than assuming:

```
pristine bin-p (10221 c51e4dabf):  GGML_NOHUGEPAGE_PROCESS -> 0 occurrences
                                   GGML_NOHUGEPAGE         -> 0 occurrences   (marker: NONE)
champion bin-r1 (10303):           both present; marker INF70_CHAMPION3_PROCESS_THP_DISABLE=...
```

**Neither THP knob exists in pristine, so the same shim state is impossible.** The ratio is
therefore **champion under its canonical adopted recipe versus pristine as it shipped** — a
recipe-to-recipe comparison, and it cannot be made a knob-controlled contrast.

What *is* controlled: same harness, same window, **adjacent interleaved launches**
(`CP PP CP PP CP PP CP CP CP`), same prompt set, same server flags, same host state. Those are the
conditions the ratio rests on.

**A corollary worth recording: no champion-vs-pristine ratio this campaign has ever quoted was
knob-controlled either.** The THP difference was inside all of them, unlabelled.

## 3. THE ADOPTED RECIPE BOUGHT PRECISION AS WELL AS THROUGHPUT

This is the strongest claim in the report and the paired design supports it directly.

| | launches | between-launch sd | range |
|---|---:|---:|---:|
| **BEFORE** — shim OFF, the **now-retired** configuration (`CHAMPION-LAUNCH-TABLE.md`) | 9 | **5.081%** | **12.55%** |
| **AFTER** — shim ON, the adopted recipe (this run) | 6 | **0.609%** | **1.79%** |

**sd ratio 8.3x; variance ratio ~70x.** The nine-launch spread table was measured **entirely in the
configuration we have just retired**, and it should be read as the *before* picture — not as a
standing property of the champion.

Independently corroborated by the paired decision test, where the shim was the only thing varying
and the OFF/ON variance ratio was **25.3x** with 6/6 pairs ON-faster.

**What this costs in launches** — the operator should know what precision they are buying:

| target 95% CI | shim ON (adopted) | shim OFF (retired) |
|---|---:|---:|
| ±1.0% | 2 launches | 100 |
| ±0.5% | **6 launches** | 397 |
| ±0.25% | 23 launches | 1587 |

**A ±0.5% champion headline now costs ~23 minutes. Before adoption it would have cost ~25 hours.**

## 4. The shim-ON sd was VERIFIED, not assumed

The plan sized from the decision test's **0.481%** and required the figure be checked as it ran.
Observed here: **0.609%** over 6 launches (5 dof) for plain, **0.356%** for MTP. Same order,
slightly above the reference for plain and below it for MTP.

**Verdict: the ON sd holds.** The delivered CI (±0.487%) is fractionally wider than the ±0.385%
the plan projected, and the headline precision above is quoted at the **observed** value, not the
projected one. n=6 gives a coarse variance estimate and is labelled as a check, not a precise sd.

## 5. Per-launch record

| launch | config | t/s | screens |
|---|---|---:|---|
| fcp1..fcp6 | champion plain, shim ON | 27.984, 27.813, 27.936, 27.909, 28.110, 27.609 | all CLEAN |
| fpp1..fpp3 | pristine plain | 12.809, 12.717, 12.761 | all CLEAN |
| fcm1..fcm6 | champion MTP, shim ON | 43.467, 43.077, 43.386, 43.129, 43.246, 43.379 | fcm2 FLAGGED, rest CLEAN |
| fpm1..fpm3 | pristine MTP | 23.529, 23.954, 23.644 | all CLEAN |

`THP_enabled` read **0 on every champion launch** ("shim VERIFIED active") and **1 on every pristine
launch** ("shim correctly absent"), under the fail-closed assertion that aborts on a mismatch.
Pristine additionally asserted **knob page absent by design** — it would have aborted had a
pre-knob-page binary mapped one.

**A completeness gate was applied**: every arm must carry 24 rows and an `ARM_DONE` marker. All 18
passed. This gate exists because a partial arm is a different token mix, not a comparable one — an
interim reading of an in-flight MTP arm looked like a +7% outlier (`draft_n` 2814 vs 3267) and would
have entered the sd had it been taken at face value.

## 6. What this does NOT resolve

**`CHAMPION-DIVERGENCE.md` stays open.** The plain ratio here is **2.1857x** against the campaign's
standing **1.7151x**. Two conditions differ at once — shim state and harness/window — so this run
narrows the gap's *causes* without closing it. Specifically: pristine reproduces closely across both
(12.762 here vs 12.366 standing, +3.2%), while the champion does not. Adopting the shim explains
part of the champion's movement and its instability; it does not explain all of it, and I am not
asserting that it does.

**Not measured here**: a correctness gate specific to shim ON vs OFF (the shim changes page backing,
not arithmetic; every champion-state comparison across the campaign was 24/24 byte-identical), and
any GPU-surface number.
