# CHAMP-2 whole-process THP shim — result

**Verdict: NON-CLAIM. Observed +3.458%, permutation p = 0.143, 95% CI [0.9962, 1.0632].**
Unit = one process launch (session). n = 4 OFF vs 4 ON, `C(8,4)=70` exhaustive.
Conditions: hot harness, quiet host (every arm CLEAN on both screens, `excess_cores` negative
throughout, announced GPU lane 0.0%), baseline = champion `ef81196d5`, binary `bin-r1`
(10303/`2516c9807`), 2 arms per session, sessions alternating OFF/ON in one region hold
11:40:09Z–12:32:23Z.

| session | shim | `THP_enabled` | mean t/s | arms | within-session spread |
|---|---|---:|---:|---|---:|
| S11_OFF | off | 1 | 24.3820 | 24.5846, 24.1793 | 1.662% |
| S12_ON | **on** | **0** | **24.2698** | 24.2212, 24.3185 | 0.401% |
| S13_OFF | off | 1 | 24.8187 | 24.8535, 24.7839 | 0.281% |
| S14_ON | **on** | **0** | 26.0888 | 26.0160, 26.1616 | 0.558% |
| S15_OFF | off | 1 | 24.6616 | 24.9717, 24.3515 | 2.515% |
| S16_ON | **on** | **0** | 26.2218 | 26.2801, 26.1634 | 0.445% |
| S17_OFF | off | 1 | 25.2876 | 25.2502, 25.3250 | 0.296% |
| S18_ON | **on** | **0** | 25.9979 | 26.1532, 25.8425 | 0.445% |

## The switch is PROVEN, so a null here would be informative

`GGML_NOHUGEPAGE_PROCESS` is a `prctl(PR_SET_THP_DISABLE)` with no log line. The discriminator is
**`THP_enabled` in `/proc/<pid>/status`**: 1 = THP allowed, 0 = the prctl is in force. It read
**1 in all four OFF sessions and 0 in all four ON sessions**, captured per session into
`runs/thp_proof.txt`, with a **fail-closed assertion** in `session2.sh` that aborts a session whose
requested state does not match the observed one. No arm here can be silently unswitched.

`AnonHugePages` is **not** usable as the discriminator and was not used: it reads ~0.06% of Rss
immediately after load and grows to ~6% within minutes on the *same* process, so the one-shot
reading taken at load time (which HARNESS-1's arm record captured) describes nothing about the arm
window.

## Why this is a NON-CLAIM despite a clean-looking pattern

**3 of the 4 ON sessions exceed EVERY OFF session** (26.09, 26.22, 26.00 vs an OFF maximum of
25.29), and the alternation tracks the knob rather than time: `24.82 → 26.09 → 24.66 → 26.22`.
The single exception is **S12_ON at 24.2698, the FIRST ON session**.

Dropping S12_ON gives perfect separation and `p = 0.029`. **There is no pre-registered basis to drop
it** — it passed both contention screens, its within-session spread is 0.401% (among the tightest in
the block), and its `THP_enabled=0` is confirmed. Removing it would be selecting the arm that
produces the answer, which is precisely the failure this campaign's pre-registration exists to
prevent. **It stays in, and the result is a NON-CLAIM.**

This was tested and held. A hypothesis was raised mid-analysis that S12_ON might be the only
*uncontaminated* ON session (and therefore the one to keep rather than the one to question), which
would have justified dropping the other three. It rested on a premise — a 12:05Z host step change —
that the operator then ruled a non-event (see below). **The refusal to drop S12_ON was correct
before that ruling and is more correct after it.** The sentence that matters is the one that costs
us the result: *dropping the single contrary session would have produced p = 0.029, and it was not
dropped.*

An order effect (first ON session after a cold OFF session) is a *hypothesis* that would explain it.
It is untested, and it is recorded as a hypothesis, not a reason.

## The structural finding: the unit is the session, and sessions are 13x noisier than arms

| scale | value |
|---|---:|
| **within-session** arm-pair spread (median of 8 sessions) | **0.501%** |
| **between-session** pooled within-group sd | **2.793%** |
| between-session *total* sd (inflated by the effect itself) | 3.160% |

**A process-scoped knob faces a floor ~13x coarser than an arm-scoped one on the same quiet host.**
This is why THP is expensive and FIX-1 is cheap, and it was predictable before the block was spent.

Required n per side (80% power, α=0.05, two-sided, on the pooled within-group sd of 2.793%):

| target effect | n sessions/side | wall time (~6.5 min/session) |
|---|---:|---:|
| the observed **+3.46%** | **11** | ~2.4 h |
| the previously reported **+1.0%** solo | 123 | ~27 h |
| the pooled **+0.16%** | 4780 | ~35 days |

**Caveat that must travel with these numbers**: they are *session*-unit figures. The 0.171%
within-session arm floor from the Q3–Q6 calibration does **not** transfer to them — a process-scoped
knob cannot borrow an arm-scoped floor. Quoting the arm floor against a session-unit question is
exactly how the 0.80% figure misled this campaign.

## What this does to the standing hypothesis

The standing hypothesis was that `GGML_VEC_Q8K` and `GGML_QSPLIT` removed the `wdata` traffic the
shim depended on, i.e. that CHAMP-2 is now **mechanism-refuted**. **This block does not support
that.** The point estimate is **+3.46%, positive and larger than the +1.0% solo figure it was
supposed to have lost**, on a champion that contains both of those levers. That is the opposite
direction from mechanism-refutation.

**It is not evidence that the shim works, either** — p = 0.143 and the CI includes 1.0. The correct
statement is: **the shim is unresolved at n=4/side, the direction is positive, and the mechanism-
refutation hypothesis is not supported by these data.** Resolving it costs ~11 sessions/side ≈ 2.4 h
of exclusive host time, which is affordable and is the one THP experiment worth booking.

## Known non-confound: the orchestrator API stop at 12:05Z

The orchestrator API (uvicorn :8000 + 6 workers, pid 3961116) was stopped by the operator at
**12:05Z, inside this block** — between `S14_ON` (12:00:06–12:06:30) and `S15_OFF`
(12:06:30–12:13:09). A future reader comparing these session timestamps against the host log will
find a service stopping mid-block; this note exists so they do not have to re-derive whether it
mattered.

**It did not. Operator ruling: the API was effectively IDLE, so it contributed no contention, and
the 12:05Z stop is a non-event.** That is the operator's knowledge of their own infrastructure and
it is the basis for this note. **The result is not stratified, no session is dropped, and the
headline is unchanged.**

**Free empirical corroboration**, from samplers that were already running (no re-run, no new
tooling):

* **Host sampler — total busy cores, the authoritative instrument for a host-wide step: FLAT
  across the entire block.** Median busy cores per arm ranged **47.89 to 48.26** (my server is 48
  threads), i.e. excess load between −0.28 and +0.26 cores, with **no discontinuity at 12:05Z**:
  `TO3 47.89, TO4 47.89 | TN3 48.18, TN4 48.03 | TO5 47.98, TO6 48.06 | TN5 47.72, TN6 48.26`.
  Six idle workers freed no measurable CPU, exactly as the ruling states.
* **Foreign %CPU sampler**: S13 `40.2, 40.2` → S14 `70.2, 45.1` → S15 `44.7, 61.1` → S16
  `25.2, 35.3`. There is **no step at the 12:05 boundary** — S14 and S15 straddle it and are the
  two *highest* readings in that stretch. A modest baseline decline does appear later, around
  **12:14** (S16 onward, ~40% → ~25%, i.e. **~0.15 of one core**), which does not align with 12:05
  and is the magnitude of desktop noise; the named processes throughout are `claude`, `htop`,
  `tmux`, `gnome-shell`, `session_bus_coo`, none of them uvicorn.

Reported as measurement, without conclusion drawn from it: the ruling stands on the operator's
knowledge either way.
