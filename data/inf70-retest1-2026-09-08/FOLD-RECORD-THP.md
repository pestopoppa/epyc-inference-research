# FOLD RECORD — CHAMP-2 whole-process THP shim: **KEEP**

**Verdict: LIKELY IMPROVEMENT — KEEP.** Pre-registered paired sign test, early-stop boundary fired
at the first look: **6/6 pairs ON-faster**, exact two-sided alpha of the whole two-look plan
**0.0430**. Plan frozen in `PREREG-THP-DECISION.md` (14:08:05Z, sha256 `337200315c6b…`) before the
lock; boundary implemented in `tools/decide.py` and unit-tested before the run.

---

## THE THREE THINGS, STATED EXPLICITLY

### 1. Branch @ commit

**Measured on:** `inf70/retest1-fix1` @ **`2516c9807`** (build **10303**), which is
`ef81196d5` + the HARNESS-1 knob page + the SYNC-17/SYNC-18 knobs.

**The champion itself is `ef81196d5`.** The measurement binary is *not* proposed for fold — it is an
instrument. The shim's own code is already present in `ef81196d5` (`common/common.cpp`), so
**no new branch and no rebuild is required to adopt this**. See §Delivery.

### 2. The knob and its default state

| | |
|---|---|
| knob | **`GGML_NOHUGEPAGE_PROCESS=1`** |
| mechanism | `prctl(PR_SET_THP_DISABLE)`, taken **before** the 92 GB model allocation |
| default state **in `ef81196d5` today** | **OFF (opt-in)** — marker string: `INF70_CHAMPION3_PROCESS_THP_DISABLE=DEFAULT_OFF;OPT_IN=GGML_NOHUGEPAGE_PROCESS=1` |
| gated by | requires `GGML_NOHUGEPAGE` on (its own default) — `common_thp_env_on("GGML_NOHUGEPAGE") && common_thp_env_opt_in("GGML_NOHUGEPAGE_PROCESS")` |
| default state being recommended | **ON** |
| **NOT this knob** | `GGML_NOHUGEPAGE` (a `madvise` on the model buffer, already ON, 86.4% of the champion). **Never conflate the two.** They are different knobs with different scopes. |

### 3. UNIT — the one people get wrong

> **The knob is set at LAUNCH. Its unit is the SESSION (one process launch), not the arm.**

Everything about this keep is session-unit and must be recorded as such:

| quantity | unit | value |
|---|---|---|
| replication unit | **session (process launch)** | one launch = one observation |
| evidence | **6 paired launches** (12 sessions) | 6/6 same direction |
| floor / precision | **between-launch** | 2.510% (OFF), 0.481% (ON) |
| **NOT applicable** | ~~arm-level floor 0.171–0.501%~~ | **does not transfer** |

**Why this warning exists:** applying the arm floor to this session-unit question earlier today gave
"4 sessions/side" where the correct answer was **4,780** — a **1200-fold** error. A fold record that
carries an arm-unit floor for a launch-unit knob reproduces exactly that mistake. The shim cannot be
switched between arms in a live process; a "per-arm" number for it is meaningless by construction.

**Verification method, also session-unit:** `THP_enabled` in `/proc/<pid>/status` — **1 = THP
allowed (shim off), 0 = prctl in force (shim on)** — read once per launch, with a fail-closed
assertion that aborts a session whose requested state does not match the observed one. All 12
sessions passed. `AnonHugePages` is **not** a valid discriminator (0.06% of Rss at load, ~6% minutes
later on the same process).

---

## Delivery — the lowest-risk route needs no fold at all

**Recommended: set `GGML_NOHUGEPAGE_PROCESS=1` in the launcher.** This is exactly how the evidence
was collected (per-session env), requires **no code change, no rebuild, no new binary, and no
re-validation of the champion**, and is trivially reversible. The champion binary stays
`ef81196d5`, bit-identical.

**Alternative, if a code default is wanted:** a lane branch off `ef81196d5` flipping the opt-in
default in `common/common.cpp` to on. That is a behaviour change to the shipped default and
**would need its own build, its own `THP_enabled` verification, and a fresh correctness gate** —
none of which this campaign performed. The measured evidence is for the env route; the code route is
equivalent only if the code path is identical, which is likely but was **not** tested.

## Correctness

The shim changes page backing, not arithmetic. Across the whole campaign, **every champion-state
comparison was byte-identical, 24/24 per pairing**, including cross-binary (bin-r1 champion-state vs
bin-h1). No correctness gate was run specifically on shim-ON vs shim-OFF output, and none is claimed.

## What is NOT claimed

**The magnitude is not resolved and this design did not size for it.** Pair effects were
`+4.843, +5.617, +5.934, +5.664, +0.321, +0.558` % — median **+5.23%**, range **+0.32% to +5.93%**.
The direction is called; the size is not. Anyone quoting a number for this keep must measure it,
and must do so at **launch** granularity.
