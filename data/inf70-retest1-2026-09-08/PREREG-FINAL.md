# RETEST-1 — FINAL MEASUREMENT: champion `ef81196d5` + shim ON (canonical adopted recipe)

**Frozen at 2026-09-08T15:05:15Z, BEFORE the lock. Author: RETEST-1.**

The campaign's last measurement. **Characterisation, not a lever test** — no A/B, no knob under
test. Results land in `CHAMPION-FINAL.md`; this document is empty of them.

## 1. The configuration being characterised

**Champion = `ef81196d5` run under the canonical recipe, which now includes
`GGML_NOHUGEPAGE_PROCESS=1`** (operator-adopted following the paired decision test, 6/6, alpha
0.0430, `FOLD-RECORD-THP.md`).

Two knob layers, set explicitly on every champion launch, neither by default:

* **shim, at launch:** `GGML_NOHUGEPAGE_PROCESS=1` -> verified per launch by the fail-closed
  assertion, **`THP_enabled` must read 0**; a launch that disagrees aborts the run.
* **champion knob state, per arm:** `GGML_SOLO_YIELD_ROWCOL=0, GGML_SCALE_SPLIT=0,
  GGML_TINY_SOLO_CLAMP=1, GGML_GET_ROWS_SOLO=0`. `bin-r1`'s **compiled defaults are NOT the
  champion's**; using them would apply the −2.136% FIX-1 regression to the headline.

## 2. THE RATIO IS RECIPE-TO-RECIPE, NOT KNOB-CONTROLLED — established before the run

The coordinator's condition was "pristine measured under the same harness and same shim state, or
the ratio is not a ratio. State which." **Answer: the same shim state is IMPOSSIBLE**, and this was
checked rather than assumed:

```
pristine bin-p (10221 c51e4dabf):  GGML_NOHUGEPAGE_PROCESS  -> 0 occurrences, all objects
                                   GGML_NOHUGEPAGE          -> 0 occurrences
                                   INF70_..._THP_DISABLE marker -> NONE
champion bin-r1 (10303):           GGML_NOHUGEPAGE_PROCESS  -> present (cpu + common)
                                   marker: INF70_CHAMPION3_PROCESS_THP_DISABLE=DEFAULT_OFF;
                                           OPT_IN=GGML_NOHUGEPAGE_PROCESS=1;MASTER_OFF=GGML_NOHUGEPAGE=0
```

**Neither THP knob exists in pristine.** So the ratio is labelled, in the report and on every quote:

> **champion under its canonical adopted recipe, versus pristine as it shipped** — a
> **recipe-to-recipe** comparison. It is **not** a knob-controlled contrast and cannot be made one.

This also means **no champion-vs-pristine ratio this campaign has ever quoted was knob-controlled**;
the THP difference was inside all of them. Same harness, same window, adjacent launches — those
conditions ARE met and are what the ratio rests on.

Pristine launches set **no** shim env; their assertion expects `THP_enabled=1` ("shim correctly
absent"). Requesting the shim on pristine would abort by design and would be dishonest labelling.

## 3. Unit — LAUNCH, and the precision reported is between-launch

**One arm per launch.** Justified by measurement: between/within variance ratio 10.84; arms inside a
launch reduce only the 4.4% within term. The reported precision is **the standard error of the mean
across launches**, never within-session tightness.

> A 0.48-0.66% within-session figure describes the **session**, not the champion. Today's shim-OFF
> data shows the champion's own launch-to-launch spread was **12.55%**. That unit error has bitten
> three times; this is the report where it would do the most damage.

## 4. Sizing — from the SHIM-ON sd, and verified as it runs

Shim-ON between-launch sd measured in the decision test (6 launches, 1 arm each): **0.481%**.
Shim-OFF, for contrast: **2.510%**.

| target 95% CI | n launches, shim ON | n launches, shim OFF | ratio |
|---|---:|---:|---:|
| +/-1.00% | **1** | 25 | 25x |
| +/-0.50% | **4** | 97 | 24x |
| +/-0.25% | **15** | 388 | 26x |

**Chosen: n=6 champion launches per instrument** -> CI **+/-0.385%** if the ON sd holds.

**The ON sd is VERIFIED, not assumed.** The observed between-launch sd over the 6 champion launches
is computed and reported. If it is materially larger than 0.481%, **that is a finding about the
shim** — that six paired launches did not generalise — and is reported as such, with the headline
precision widened to the observed value. n=6 gives 5 dof for that estimate; it is a check, not a
precise variance estimate, and will be labelled that way.

## 5. Plan

| block | launches | order |
|---|---|---|
| plain | 6 champion + 3 pristine | `CP1 PP1 CP2 PP2 CP3 PP3 CP4 CP5 CP6` |
| MTP | 6 champion + 3 pristine | `CM1 PM1 CM2 PM2 CM3 PM3 CM4 CM5 CM6` |

Pristine launches are **interleaved** so each ratio is formed from temporally adjacent launches,
keeping slow host state common-mode. Champion launches are spread across the whole window so the
between-launch sd is estimated over a realistic span, not a 10-minute cluster.

MTP recipe: `-md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 4
--spec-draft-p-min 0.5`. **Draft acceptance is reported** per configuration.

Measured launch costs (setup + one arm + teardown): champion plain 216 s, pristine plain 368 s,
champion MTP 173 s, pristine MTP 241 s. **Total ~4161 s = 1.16 h**, one region hold.

## 6. Claim grammar and standing rules

* Headline is a **SIGN claim with a BOUNDED magnitude** — "faster, by at least X" from the interval
  — never a bare point estimate.
* Every number carries: hot harness · n · **unit (launch)** · contention model and per-arm screen
  verdicts · host state · baseline `ef81196d5` · shim state.
* Both screens live; **`screened()` for every statistic**; a dropped launch is excluded before any
  statistic; a configuration with <2 usable launches is reported **not measured**, never a value.
* Placement gate, ggml linkage assertion, `THP_enabled` fail-closed assertion, per launch.
* **No hot absolute compared against a cold one.**
* **Halt, report, WAIT** on any failure. Nothing is chained after this run.

## 7. The BEFORE picture, to be presented as such

`CHAMPION-LAUNCH-TABLE.md` (9 launches, between-launch sd **5.081%** quiet-host, range **12.55%**)
was measured **entirely shim OFF** — the configuration just retired. If the ON sd holds, that table
is the **before** picture, and the pair of them is evidence that **the adopted recipe change bought
precision as well as throughput**. That is a stronger and better-supported claim than the +5.23%
median, because the paired design supports it directly and the median is not sized for.
