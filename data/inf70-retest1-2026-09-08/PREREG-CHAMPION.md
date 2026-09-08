# RETEST-1 — CLOSING MEASUREMENT: champion `ef81196d5` headline characterisation

**Frozen at 2026-09-08T13:14:21Z, BEFORE the lock. Author: RETEST-1.**

**This is CHARACTERISATION, not a lever test.** There is no A/B, no knob under test, no treatment.
One configuration is measured cleanly so the number the campaign quotes is the number the champion
produces. Results sections are empty here; they land in `CHAMPION-RESULT.md`.

## 0. THP is CANCELLED and filed

CHAMP-2 stands exactly as reported: **+3.458%, p = 0.143, NON-CLAIM, unresolved-not-refuted**,
~11 sessions/side to settle. The shim keeps its current default. **No default is flipped on an
unresolved result.** No THP work is registered or run.

## 1. THE TRAP, stated first because it would silently corrupt the headline

`bin-r1`'s **compiled defaults are the RETEST-1 branch defaults, not the champion's**:

```
ggml_cpu_knobs_cur = { ... /*scale_split=*/1, /*tiny_solo_clamp=*/0,
                           /*solo_yield_rowcol=*/1, /*get_rows_solo=*/0, ... }
```

`GGML_SCALE_SPLIT=1` and `GGML_SOLO_YIELD_ROWCOL=1` are ON by default in that binary. **The champion
`ef81196d5` contains neither knob — the code does not exist there.** Measuring `bin-r1` at its bare
defaults would measure FIX-1+FIX-3 ON, which this campaign just established is a **−2.136%
regression**, and would therefore **understate the champion by ~2.1%**.

**Every champion arm sets the champion knob state EXPLICITLY:**

```
GGML_SOLO_YIELD_ROWCOL=0,GGML_SCALE_SPLIT=0,GGML_TINY_SOLO_CLAMP=1,GGML_GET_ROWS_SOLO=0
```

This state is **proven equivalent to the champion**, not assumed: it produced **24/24 byte-identical
output against `bin-h1`** (build 10242, champion-3 `eae02f2dc`), whose ggml-cpu source tree is the
**identical git tree object** (`040d43aa…`) to `ef81196d5`'s. The knob readback in the server log is
checked per arm.

## 2. Configurations

| id | binary | build | knobs | speculative decoding | arms |
|---|---|---|---|---|---:|
| `CP` champion plain | `bin-r1` | 10303 `2516c9807` | champion state (above) | none | 5 |
| `PP` pristine plain | `champion3/bin-p` | 10221 `c51e4dabf` | none exist (`NO_KNOBS=1`) | none | 3 |
| `PM` pristine MTP | `champion3/bin-p` | 10221 | none exist | MTP | 3 |
| `CM` champion MTP | `bin-r1` | 10303 | champion state | MTP | 5 |

MTP recipe, as codified in this campaign: `-md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.5`.

**Run order `CP → PP → PM → CM`**, so each cross-binary ratio is formed from **adjacent** sessions
(plain: CP/PP; MTP: PM/CM), keeping slow host drift common-mode across the pairing.

## 3. Units — stated per number, because they differ within one report

* **Within a configuration** (central value, precision): the unit is the **ARM**. All arms of a
  configuration sit in one hot process, so the within-session floor (~0.5%) applies.
* **Champion-vs-pristine RATIO**: the unit is the **SESSION** — the two binaries cannot share a
  process, so the ratio necessarily crosses a session boundary and carries the **between-session sd
  of 2.793%** measured today. That is ~6% of a ~1.5x effect, so the ratio is robust; it is stated
  anyway, because a ratio quoted without its unit is how this campaign got into trouble.

## 4. Sizing — for a trustworthy central value, NOT a hypothesis test

n=5 champion arms per configuration and n=3 pristine arms. This is **not** powered against a small
difference and makes no attempt to be: there is no difference under test. n=5 gives a
standard-error on the central value of roughly `sd/sqrt(5)` ≈ **0.2%** at the measured within-session
sd of ~0.5%, which is finer than any figure this campaign quotes. Pristine gets 3 arms because its
role is a ratio denominator ~50% away, not a precise value, and its arms cost ~1.5x more wall-clock.

Estimated wall clock ~50 min, one region hold.

## 5. Screens and gates — unchanged from the campaign

* Both screens live per arm (`foreign_load.py` CPU screen; `host_load.py` host screen), thresholds
  as frozen in `PREREGISTRATION.md` s6 and `AMENDMENT-1.md`.
* **`screened()` everywhere** — no statistic over an arm its own screen dropped; no verdict on zero
  cases.
* A dropped arm is excluded before any statistic. If a configuration retains **fewer than 2** arms,
  that configuration is reported as **not measured**, never as a value.
* Per-session `THP_enabled` readback with the fail-closed assertion (all sessions shim OFF, expect 1).
* `NO_KNOBS=1` for pristine, which **aborts if the pristine binary unexpectedly maps a knob page** —
  fail-closed in both directions, so the wrong binary cannot be measured under the right label.
* Placement gate (<=15% deviation across NUMA nodes) and ggml linkage assertion per session.

## 6. Claim grammar — the campaign ruling applies

The headline is a **SIGN claim with a BOUNDED magnitude**, never a bare point estimate. Champion
versus pristine is reported as *"faster, by at least X"* with the bound taken from the interval, not
as a decimal ratio quoted alone. Central values carry n, unit, and their screen verdicts.

Every number carries: hot harness · n · unit (arm|session) · contention model and per-arm screen
verdicts · host state (GPU loop down, nothing of the other session running) · baseline = champion
`ef81196d5`. **No hot absolute is compared against a cold one.**

## 7. Stop conditions

* Any session failing its linkage, placement, knob-page or THP assertion -> **halt, report, WAIT**.
  Not halt-amend-rerun.
* Any configuration reduced below 2 usable arms -> reported as not measured; no re-run without
  operator approval.
* This is the campaign's **last measurement**. Nothing is chained after it.
