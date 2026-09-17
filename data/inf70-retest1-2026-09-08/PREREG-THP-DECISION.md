# RETEST-1 — DECISION-GRADE THP TEST: pre-registered plan

**Frozen at 2026-09-08T14:08:05Z, BEFORE the lock. Author: RETEST-1.**

**The deliverable is a CALL, not an estimate**: *likely improvement* / *likely not* / *cannot tell*,
with a keep-or-no-keep recommendation attached. No magnitude is sized for; the final champion is
characterised separately, so a keep's exact size does not need resolving here.

## 1. Estimand and unit

Estimand: the **SIGN** of the effect of the CHAMP-2 whole-process THP shim
(`GGML_NOHUGEPAGE_PROCESS=1`, a `prctl(PR_SET_THP_DISABLE)`) on token-weighted decode, on champion
`ef81196d5`, hot harness, quiet host.

**Unit = one SESSION (one process launch).** The shim is taken before the 92 GB allocation, so a
process is the replicate. **One arm per session**, justified by measurement rather than assertion:
today's variance decomposition gives **between-session var 0.4742 vs within-session var 0.0438, a
ratio of 10.84** — at 2 arms/session the within term is only **4.4%** of session-mean variance.
Going 2 arms -> 1 arm raises the session-mean sd from 2.793% to 2.854%, **+2.2%**, while halving
session cost. Arms within a session buy no power against a process-scoped knob. **The coordinator's
premise is confirmed by my own arithmetic.**

## 2. Design

**Paired, adjacent, order-balanced.** 10 pairs; each pair is two back-to-back launches, one ON and
one OFF. Order alternates so direction is never confounded with position — **5 pairs OFF-first and
5 pairs ON-first**, fixed in advance:

```
P01 OFF,ON   P02 ON,OFF   P03 OFF,ON   P04 ON,OFF   P05 OFF,ON
P06 ON,OFF   P07 OFF,ON   P08 ON,OFF   P09 OFF,ON   P10 ON,OFF
```

The first look (6 pairs) is itself balanced 3/3.

## 3. Statistic and stopping rule — FIXED HERE, and there is no third look

Paired **sign test** on `d_i = ON_i − OFF_i`. A sign test is used deliberately: it never estimates a
mean, so it is **robust to the 2.793% between-session sd** that makes a magnitude expensive. That is
exactly why it is right for a direction question and wrong for a magnitude one.

| look | after | CALL if | exact two-sided alpha contribution |
|---|---|---|---|
| 1 | **6 valid pairs** | **6/6 agree** in one direction | — |
| 2 | **10 valid pairs** | **>= 9/10 agree** in one direction | — |
| — | otherwise | **CANNOT TELL** | — |

**Exact overall two-sided alpha = 0.0430**, computed by enumerating all 2^10 sign sequences under
H0 with the nested stopping handled correctly (not a union bound). Power: **0.925** if the shim wins
9 pairs in 10 on average, **0.765** at 9-in-10 -> 0.90, **0.586** at 0.85.

**There is no third look and no pair is added because the sign has not settled.** Adding pairs after
seeing a mixed result is optional stopping and manufactures significance; it is forbidden here.

**Dropped pairs are REPLACED, and this does not inflate alpha**: the drop rule is the contention
screen, which is a function of foreign load and is **blind to the measured rate**. Replacement is
therefore not outcome-dependent. Hard booking cap **14 pairs launched (28 sessions)**; if 10 valid
pairs are not obtained within that, the verdict is **CANNOT TELL**.

## 4. Verdicts, decided now

| outcome | verdict | recommendation |
|---|---|---|
| >= 9/10 (or 6/6 early) with **ON faster** | **LIKELY IMPROVEMENT** | **KEEP** — stage as a lane branch off `ef81196d5` carrying the knob's default state and its unit-correct record, for the GPU session to fold |
| >= 9/10 (or 6/6 early) with **OFF faster** | **LIKELY NOT an improvement** | **NO KEEP** — champion stays final as is |
| anything else at the cap | **CANNOT TELL** | **NO KEEP** — no default is flipped on an undetermined result; the open question stays filed as it is today |

**"Cannot tell" is a complete and acceptable verdict**, stated before the data exists so that the
cap cannot quietly become optional stopping.

## 5. Prior data is EXCLUDED — start clean

Neither the earlier 4v4 THP block nor the characterisation `CP` arms are carried in. Three
independent reasons, any one disqualifying:

1. **They motivated the hypothesis.** Reusing the data that generated a hypothesis to test it is the
   classic error, and the design changed after seeing them.
2. **Structure differs.** Both blocks are **2 or 5 arms per session**; this design is **1 arm per
   session**. Five arms from one launch are ONE session-level observation, not five.
3. **Selection.** Deciding which to keep, or how to fold a multi-arm session mean into a one-arm
   design, is judgement exercised after seeing values. Instruction was explicit: if inclusion needs
   any such judgement, exclude them all.

## 6. Carried forward, unchanged

* Per-session **`THP_enabled` readback** from `/proc/<pid>/status` with the **fail-closed assertion**
  (`session2.sh`): a session whose requested shim state does not match the observed one **aborts**.
  1 = THP allowed (shim off), 0 = prctl in force (shim on). This is why a null here is informative.
* Both screens live per arm; thresholds as frozen in `PREREGISTRATION.md` s6 and `AMENDMENT-1.md`.
* **`screened()` everywhere** — no statistic over an arm its own screen dropped, no verdict on zero
  cases. **A dropped session takes its pair with it.**
* Champion knob state set **explicitly** on every arm
  (`GGML_SOLO_YIELD_ROWCOL=0, GGML_SCALE_SPLIT=0, GGML_TINY_SOLO_CLAMP=1, GGML_GET_ROWS_SOLO=0`) —
  `bin-r1`'s compiled defaults are NOT the champion's, and using them would silently apply the
  −2.136% FIX-1 regression to both sides.
* Placement gate and ggml linkage assertion per session.
* **Halt, report, WAIT** on any failure. Not halt-amend-rerun.

## 7. Booking

| | pairs | sessions | wall clock |
|---|---:|---:|---:|
| cap | 10 | 20 | **1.27 h** |
| expected, if the effect is real (q=0.90) | 7.85 | 15.7 | **0.99 h** |
| hard cap incl. replacements | 14 | 28 | 1.77 h |

Session cost 228 s measured: 7 s eviction + 44 s model load + ~2 s placement/knob verification
+ ~165 s arm + ~10 s teardown. **The 44 s load is irreducible and bounds any design.**
