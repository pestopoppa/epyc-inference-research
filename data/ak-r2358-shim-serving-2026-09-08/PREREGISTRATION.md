# PRE-REGISTRATION — R23-58: does the whole-process THP shim improve the GPU **serving floor**?

**Status: REGISTERED, UNRUN.** Written 2026-09-08 while another session holds the host. Nothing in
this directory has been executed. Freeze this file (`sha256sum PREREGISTRATION.md > PREREGISTRATION.sha256`)
before the first launch; a plan edited after data exists is not a plan.

| | |
|---|---|
| id | **R23-58** |
| champion under test | `ef81196d5` (GPU tip `bff30cebe` + CPU champion-3 `9c4f73e29`) |
| knob | **`GGML_NOHUGEPAGE_PROCESS=1`** — `prctl(PR_SET_THP_DISABLE)` in an `__attribute__((constructor))` in `common/common.cpp`, default **OFF / opt-in** |
| **NOT this knob** | `GGML_NOHUGEPAGE` (the `madvise(MADV_NOHUGEPAGE)` inside `ggml_aligned_malloc`, default ON). Different scope, different knob. Never conflate them. `GGML_NOHUGEPAGE` is left at its default in **both** arms and is never set by this design. |
| surface | `llama-server` under the canonical GPU serving recipe `qwen3.8-27b-q8-gpu-dflash2-np4` |
| recipe file | `/mnt/raid0/llm/worktrees/mains/ak-rebuild-research/artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4.json` (`cpu_list` pinned `184-191`) |
| metric | `aggregate_tok_s` (sum of per-slot `predicted_per_second` over `np=4`) |
| **unit** | **the SESSION — one `llama-server` launch.** Not the arm, not the request. |
| standing floor | **4.581 % p95 at n=10**, `/mnt/raid0/llm/autokernel/loop-memory/serving-floor.qwen3.8-27b-q8-gpu-dflash2-np4.json`, calibrated 2026-09-08T10:21:45Z with the loop DOWN |
| build | **`/mnt/raid0/llm/tmp/build-fold-ef81196d5`** — FOLD-2 candidate build of the champion, HIP/gfx90a, 2026-09-08 09:25, from `fold-ef81196d5-src` @ `ef81196d5` (clean). No rebuild needed; see §7. |
| harness | `autokernel.loop.serving` at the U3 enabling change (`b5f58b74`, research lane) — `Recipe.env`, `Recipe.env_readback`, `Recipe.with_env`, `Recipe.recipe_hash`, `serving.verify_env_readback`, `serving._spread` |

---

## 1. The question, as two separable claims

### Claim (i) — LEVEL. Does the shim change the serving throughput?

> H0(i): median `aggregate_tok_s` is the same with the shim ON and OFF.

**This is the claim that may well be null, and a null here is not a failure of the experiment.**
On this recipe `-ngl 99` puts the 29 GB of weights *and* the KV cache in VRAM. Whatever the shim
touches on this path is host-side glibc-heap state — the CPU backend's graph work buffer, the slot
and sampler structures, the HTTP layer — not the large model allocation that motivated it on the CPU
decode surface. The mechanism that produced +5.23 % on CPU decode has no obvious analogue here, and
this design does not assume one.

There is direct evidence that the shim's sign is **configuration-dependent**: champion-3's own
leave-one-out (recorded in the `common/common.cpp` header comment at `ef81196d5`, lines 60–125) found
the win *inverted* to −1.48 % once `GGML_VEC_Q8K` and `GGML_QSPLIT` shipped, which is exactly why
the shipped default is OFF. A GPU path is a much larger configuration change than that. **Expect
null on (i); do not read a null as an instrument failure.**

### Claim (ii) — DISPERSION. Does the shim reduce the between-launch spread?

> H0(ii): the between-launch dispersion of `aggregate_tok_s` is the same with the shim ON and OFF.

**This is the claim that matters for the campaign.** Every GPU keep is gated by the serving floor,
and the floor *is* the between-launch dispersion: `serving.calibrate_floor` sets `floor_pct` to
`_spread(runs)["p95_dev_pct"]`, the p95 of `|run/median − 1|` over A/A launches. Today that floor is
**4.581 %**, and it is the binding constraint on the whole GPU program — a 6-keep bundle worth
**+5.958 %** on the `tg128` bench resolved to **−2.176 %, `decisive=false`** on serving at n=10 pairs
(`/mnt/raid0/llm/autokernel/loop-memory/serving/bundle-bff30cebee0d.json`). The campaign is not short
of candidate keeps. It is short of **resolution**.

On CPU decode the shim's headline effect was not a location shift at all but a **compressed downside
tail**: between-launch sd **2.510 % OFF vs 0.481 % ON**, an sd ratio of ≈5.2 (variance ratio ≈27),
with ON never running slow and OFF sometimes doing so
(`/mnt/raid0/llm/tmp/inf70/agents/retest1/CHAMPION-LAUNCH-TABLE.md`,
`/mnt/raid0/llm/tmp/inf70/agents/retest1/FOLD-RECORD-THP.md`). A means-only comparison would have
missed that entirely — which is the second of today's two expensively-learned cautions, and is now
written into `serving._spread`'s own docstring.

> **A null on (i) with a real effect on (ii) is a SUCCESS, not a mixed result.**
> A shim that leaves throughput exactly where it is and halves the floor is worth more to this
> campaign than a shim that adds 2 % to throughput and leaves the floor alone, because the first one
> lets every future keep be *decided* and the second one does not.

### Why the campaign-relevant threshold is about 3×

Baseline between-launch dispersion on this recipe, from two independent existing samples:

| set | n | mean tok/s | sd | CV |
|---|---:|---:|---:|---:|
| floor A/A (`serving-floor…json`) | 10 | 160.703 | 5.324 | **3.313 %** |
| bundle anchor arm (`bundle-bff30cebee0d.json`) | 10 | 165.386 | 5.360 | **3.241 %** |
| bundle candidate arm (same file) | 10 | 162.418 | 7.321 | 4.508 % |

The two champion-build sets agree to 0.07 pp, so **sd(ln x) ≈ 0.033** is a solid prior for the OFF
arm. The floor derived from that spread is 4.581 %. A **2×** sd reduction takes the floor to roughly
2.3 % — which would still not have made the −2.176 % bundle decisive. A **3×** reduction takes it to
roughly 1.5 %, which **would** have. So 3× is the smallest reduction that changes a decision the
campaign has actually faced, and 2× is the conservative design point this plan sizes for.

---

## 2. Design — two interleaved A/A calibrations, one per arm

**One binary. Two arms differing only in one environment variable. Each arm gets its own full A/A
calibration, and the two arms' launches are interleaved in time and order-balanced. The unit is the
session.**

| | OFF arm (control) | ON arm (treatment) |
|---|---|---|
| binary | the same HIP build of `ef81196d5` | the same HIP build of `ef81196d5` |
| recipe | `base.with_env(GGML_NOHUGEPAGE_PROCESS=None, name=…+"+thp-shim-off")` | `base.with_env(GGML_NOHUGEPAGE_PROCESS="1", name=…+"+thp-shim-on")` |
| `GGML_NOHUGEPAGE_PROCESS` | **absent from the environment** | `1` |
| `GGML_NOHUGEPAGE` | not set (default: on) | not set (default: on) |
| declared `env_readback` | `THP_enabled` must read **`1`** | `THP_enabled` must read **`0`** |

Both arms are built from ONE recipe file by `Recipe.with_env(...)`, so no JSON is edited and no field
can drift between them. `with_env` appends the override to the recipe **name**, and `recipe_hash`
covers `env` and `env_readback`, so each arm is a distinct, content-addressed measured condition that
cannot be silently judged against the other arm's floor.

### 2.1 Why the paired A/B shape was rejected

The obvious shape — `serving.compare(recipe, anchor_build, candidate_build, pairs=N)` — is wrong here
for two independent reasons, and the second is the important one.

1. **It cannot express this experiment at all.** `compare` varies the **BUILD** from a single recipe:
   both arms are launched from the same `recipe`, so they necessarily share one `env` and one
   `recipe_hash`. A same-build env A/B has no way in. Faking it by mutating `os.environ` between
   launches would put the knob outside the recipe, outside `recipe_hash`, and outside the declarative
   readback — i.e. outside everything that makes the arm provable.
2. **Its decision statistic is a location contrast, which is exactly the statistic that would miss
   the effect being hunted.** `compare` reduces each arm to a median and reports
   `median(candidate)/median(anchor) − 1`. The CPU effect was a *compressed downside tail* with the
   median barely a proxy for it; a means-only comparator returns "no effect" on a real one, because
   the arm that removes the bad launches looks identical to the arm that keeps them once the tail is
   averaged away. Using a location comparator to hunt a dispersion effect is the second of today's two
   expensive cautions, committed on purpose.

**An A/A per arm measures dispersion directly, and in the floor's own grammar.** `calibrate_floor`
already exists to answer exactly the question "how much does this condition's throughput move between
launches", and its `spread` block is computed by the same `_spread` function that defines `floor_pct`.
So the result of this run *is* two floors, side by side, with no translation step between the
statistic tested and the number the keep gate reads.

### 2.2 Interleaving and order balance — why the arms are not run as two blocks

`serving.calibrate_floor(recipe, build, samples=N)` runs its N launches consecutively. Calling it
twice — all OFF, then all ON — would confound the arm with host drift, and that drift is not small:
the CPU session measured between-launch sd at **2.793 %** within one contiguous ~52-minute block but
**5.081 %** across ~3.7 hours, and recorded an unexplained +9 % step in the middle of a working day
that nothing in its logs accounted for (`CHAMPION-LAUNCH-TABLE.md`). Two consecutive blocks would put
that entire term on whichever arm ran second.

So the runner performs the same measurement `calibrate_floor` performs, launch for launch, but on an
**interleaved, order-balanced schedule**:

```
couple k = two adjacent launches, nothing else between them
    k odd  -> (OFF, ON)
    k even -> (ON, OFF)
```

After N couples each arm has N launches, spread identically across the window, with each arm going
first in exactly half the couples. Each arm's dispersion is then computed with **`serving._spread`
itself** — not a local reimplementation — so the numbers are produced by the same code path as
`calibrate_floor`'s `floor_pct` and cannot drift from it. The runner emits one
`epyc.autokernel.serving_floor.v1`-shaped record per arm, marked `interleaved: true` so nobody
mistakes it for a block calibration.

### 2.3 The median estimate comes from the same launches

Claim (i) costs **no extra launches**. The couples are adjacent pairs by construction, so the sign of
`on − off` within each couple is available directly, and claim (i)'s sign test is computed from the
same 2N launches at zero marginal cost.

### 2.4 Why the arm cannot be switched inside a live process

Two independent reasons, both verified in the source at `ef81196d5`:

1. The shim is `__attribute__((constructor)) static void common_thp_disable_process(void)` in
   `common/common.cpp` (line ~117). It reads its environment and calls `prctl` **before `main()`**.
   Its value is fixed at `exec` time. There is no runtime path to it.
2. Even if there were, `PR_SET_THP_DISABLE` governs **future faults**. By the time a server is
   serving, its heap and its work buffers are already faulted in with whatever backing they got.
   Flipping the flag mid-process would change the label without changing the pages.

Therefore **one launch = one observation**, and any per-arm or per-request "floor" for this knob is
meaningless by construction. This is the first of today's two expensive cautions, stated so it cannot
be repeated: substituting the arm-unit floor into a session-unit sizing calculation earlier today
turned an answer of 4,780 sessions into "4 sessions" — a **1200-fold** error
(`FOLD-RECORD-THP.md` §3). **No arm-unit number enters any calculation in this document.**

---

## 3. Stop rule and looks — fixed in advance, with the arithmetic

### 3.1 Claim (i), LEVEL: the two-look sign boundary, exact two-sided α = 0.0430

Reused verbatim from the CPU session's registered and unit-tested boundary
(`PREREG-THP-DECISION.md`, implemented in `/mnt/raid0/llm/tmp/inf70/agents/retest1/tools/decide.py`).
Reusing an already-frozen rule means there is no new arithmetic to get wrong, and it makes the GPU
verdict directly comparable to the CPU one.

```
LOOK 1  after  6 valid couples : CALL iff 6/6 couples agree in direction
LOOK 2  after 10 valid couples : CALL iff >= 9/10 couples agree in direction
otherwise                      : CANNOT TELL
```

Exact two-sided α under H0 (each couple's sign is ±1 with probability ½, independent):

```
LOOK 1 fires        : 2 * (1/2)^6                        = 2/64      = 0.031250
LOOK 2 fires, given LOOK 1 did not:
    #up in {9,10} or {0,1} over 10 couples               = 2*(1+10)/1024 = 22/1024
    minus sequences already stopped at LOOK 1:
        #up=10 -> first 6 all up  -> stopped             (1 sequence)
        #up=9  -> first 6 all up iff the single down
                  lies in positions 7..10                (4 of 10 sequences)
    surviving one-sided sequences = (10 - 4) = 6
    two-sided                                            = 2*6/1024  = 0.011719
                                                           ----------------------
                                                alpha = 0.042969  ~= 0.0430
```

**A sign-test call is a DIRECTION, not a MAGNITUDE.** No percentage may be quoted as "the effect of
the shim on GPU serving throughput" on the strength of this boundary. Sizing for a magnitude is a
separate, much more expensive experiment and is explicitly out of scope (§6).

### 3.2 Claim (ii), DISPERSION: one test, at the final n, no interim looks

**Primary estimator — the ratio of `p95_dev_pct`, the floor's own statistic:**

```
rho_p95 = spread(OFF)["p95_dev_pct"] / spread(ON)["p95_dev_pct"]      (rho > 1 means ON is tighter)
```

`p95_dev_pct` is computed by `serving._spread`, which is the same function `calibrate_floor` uses to
produce `floor_pct`. **This is why it is primary: the tested statistic and the gate the campaign
actually runs on cannot drift apart.** A test on some other dispersion measure would prove something
adjacent to, but not identical to, the number that decides keeps.

**Reported alongside — the sd ratio:**

```
rho_sd = sd( ln x_OFF ) / sd( ln x_ON )
```

The sd is the better-behaved estimator in the abstract — it uses every observation, whereas
`p95_dev_pct` at n = 24 is `devs[round(0.95*23)] = devs[22]`, the second-largest of 24 deviations,
an extreme order statistic. **That objection is real but it is a sampling-distribution objection, and
this design does not rely on a sampling distribution.** The inference is a permutation test, whose
null is exact for *any* statistic by construction. The only cost of the coarser statistic is power,
and that cost is measured, not waved away (§3.3). Both ratios are reported; **`rho_p95` decides.**

**Inference — exact within-couple label permutation.** Each arm's runs are first divided by that
arm's own median, so a location difference is removed exactly and cannot leak into a scale test.
The ON/OFF label is then swapped within couples — preserving the pairing, and therefore preserving
drift — and the statistic recomputed. The two-sided p-value is the fraction of permutations with
`|ln rho|` at least as large as observed. Registered in advance: **exhaustive enumeration of all 2^n
assignments when n ≤ 18; otherwise 200,000 random within-couple swaps under seed 2358.** The seed is
part of this registration.

**α = 0.05, two-sided. One test, once, at the final n. No interim looks on claim (ii)** — an
unregistered peek at a variance ratio is exactly how a 3.3 % baseline becomes a "finding".

### 3.3 Sample size

Closed-form sizing on the sd ratio, two-sided F(n−1, n−1), α = 0.05, n = launches **per arm**:

| true sd ratio | power 0.80 | power 0.90 | total launches @0.80 |
|---|---:|---:|---:|
| 1.5× | 50 | 66 | 100 |
| **2×** | **19** | 24 | **38** |
| 3× (campaign-relevant threshold) | 9 | 11 | 18 |
| **5×** (what CPU measured: sd 2.510→0.481) | **5** | 6 | **10** |
| **25× read as an sd ratio** | **3** | 3 | **6** |
| 25× read as a *variance* ratio (= sd 5×) | 5 | 6 | 10 |

*(The CPU figure is quoted as "a 25.3× variance reduction". In sd terms that is 5.03×. Both readings
are tabulated so the two can never be confused. `25×` as an **sd** ratio would be far beyond anything
observed on either surface and is included only for completeness.)*

The **primary** statistic is coarser, so it is sized by simulation rather than by that table.
Permutation-test power at α = 0.05, sd(ln) = 0.033 lognormal, 300 datasets × 400 permutations —
a coarse estimate, quoted to two decimals only:

| n per arm | statistic | ρ=1 (size) | ρ=2 | ρ=3 | ρ=5 |
|---:|---|---:|---:|---:|---:|
| 20 | `p95_dev` (primary) | 0.06 | 0.51 | 0.93 | 1.00 |
| 20 | `sd` (reported) | 0.04 | 0.72 | 0.99 | 1.00 |
| **24** | **`p95_dev` (primary)** | — | **0.69** | **0.97** | **1.00** |
| 30 | `p95_dev` (primary) | — | 0.75 | 0.99 | 1.00 |

Both statistics hold their size at ρ=1, which is the check that matters for a permutation test.

### 3.4 Per-launch cost — derived, and labelled by whose measurement it is

**OURS (this recipe, derived from files in `loop-memory`).** Two independent derivations:

1. **Artifact mtimes of contiguous serving runs.**
   * the clean 10-sample floor ran between the contaminated floor it replaced
     (`…CONTAMINATED-20260908T1005.json.bak`, mtime 10:08:08.49Z) and its own write (10:21:45.51Z) →
     817 s / 10 launches = **81.7 s per launch**;
   * the 10-pair bundle ran between the floor's write (10:21:45.51Z) and its own
     (`serving/bundle-bff30cebee0d.json`, 10:46:23.09Z) → 1478 s / 20 launches = **73.9 s per launch**.
2. **The recipe's own parameters.** `n_predict=384`, `np=4`, and a warmup round that is discarded:
   `2 × (4 × 384) / 161.08 aggregate tok/s = 19.1 s` of request time per launch. The residual
   55–63 s is boot to `/health`, the 29,047,086,048-byte target plus the 2,056,414,752-byte drafter
   loading to VRAM, and teardown.

> **Adopted planning figure: 75 s per launch, 82 s conservative.** The two derivations agree, and the
> arithmetic explains the number rather than merely fitting it. These are **our** numbers, for the
> GPU recipe.

**THEIRS (the CPU session, measured directly).** 228 s per launch = 7 s evict + 44 s model load +
~2 s placement and knob verification + 165 s arm + 10 s teardown. The 44 s load is the structural
floor of any session-unit design and does not shrink. Our arm is 19 s rather than 165 s because the
GPU recipe's workload is 4 concurrent 384-token completions at ~161 aggregate tok/s, not their
24-prompt production mix; the model load is comparable because both move ~30 GB.

**Wall clock, priced at all three figures:**

| plan | launches | @75 s (ours) | @82 s (ours, conservative) | @228 s (theirs, upper bound) |
|---|---:|---:|---:|---:|
| 9 per arm (ρ=3, sd, 0.80) | 18 | 22.5 min | 24.6 min | 1.14 h |
| 19 per arm (ρ=2, sd, 0.80) | 38 | 47.5 min | 51.9 min | 2.41 h |
| 20 per arm | 40 | 50.0 min | 54.7 min | 2.53 h |
| **recommended: 24 per arm** | **48** | **60.0 min** | **65.6 min** | **3.04 h** |
| recommended + full replacement budget (28 couples) | 56 | 70.0 min | 76.5 min | 3.55 h |

### 3.5 The plan this pre-registration adopts

> **n = 24 valid couples (24 launches per arm, 48 launches). α = 0.05 two-sided. Up to 28 couples
> launched — a 4-couple replacement budget. Expected wall clock ≈ 60 min, worst case ≈ 70 min at our
> derived 75 s per launch (65–77 min at the conservative 82 s); ≈ 3.0–3.6 h if the launch cost turns
> out to match the CPU session's 228 s shape.**

Why 24:

* it puts the **primary** statistic at 0.97 power against the campaign-relevant 3× and 0.69 against
  the conservative 2× — the largest n whose cost still fits one clean hour;
* the reported sd ratio reaches 0.90 power at 2× (closed form), so the two statistics bracket the
  design point rather than both being marginal;
* it is even, so the order balancing is exact — 12 couples `(OFF, ON)`, 12 couples `(ON, OFF)`;
* it comfortably exceeds claim (i)'s 10-couple second look, so the level boundary is fully evaluated
  inside the same run at no extra cost.

**Claim (i)'s looks never terminate the run.** The sign boundary is *evaluated* at 6 and 10 valid
couples exactly as registered and its verdict is **frozen at whichever look first calls** — α is a
property of the decision rule, not of whether the machine stops — but data collection continues to
24 couples because claim (ii) needs them. Stopping at claim (i)'s early look would truncate claim
(ii) to n = 6, where its power against ρ=2 is 0.27. **The run's stop rule is 24 valid couples, or 28
couples launched, whichever comes first.** The runner refuses to launch couple 29.

**Replacement.** A couple containing an invalid launch (§5) is **replaced, not added**: it is dropped
and re-run. Replacement is driven only by admissibility screens that are blind to the measured rate,
so it does not inflate α. Budget: 4 replacements. A fifth invalid couple ends the run as
`INADMISSIBLE — the window was not clean enough to measure in`.

---

## 4. Pre-declared verdicts — exhaustive and mutually exclusive

Throughput verdict **T** ∈ {`T+` ON faster (CALL_ON), `T−` OFF faster (CALL_OFF), `T0` cannot tell}.
Dispersion verdict **D** ∈ {`D+` ON tighter (`rho_p95 > 1`, p < 0.05), `D−` ON wider
(`rho_p95 < 1`, p < 0.05), `D0` no detectable difference}. All nine cells, decided now:

| | **D+** (ON tighter) | **D0** (no dispersion effect) | **D−** (ON wider) |
|---|---|---|---|
| **T+** | **ADOPT — strongest form.** Set `env: {"GGML_NOHUGEPAGE_PROCESS": "1"}` in the GPU serving recipe, recalibrate the serving floor under ON, publish both results. Quote no throughput magnitude. | **ADOPT as a launcher default** (zero-cost, reversible, no rebuild). **Publish NO magnitude** — a sign test gives direction only. Floor unchanged; the campaign gains nothing in resolution. | **REJECT.** A faster arm that is noisier raises the bar every future keep must clear. Record; report `D−` to the CPU side. |
| **T0** | **SUCCESS. ADOPT.** *This is the target outcome.* Null level, real resolution gain. Recalibrate the serving floor under ON and re-run the 6-keep bundle against the new floor. | **NULL. Do not adopt. Close R23-58.** Report as a **bounded** null: at n=24 we exclude sd ratios ≥ 3 at ≈0.97 power and ≥ 2 at ≈0.69 on the primary statistic. An unbounded "no effect" is not a permitted phrasing. | **REJECT.** Record; report `D−` to the CPU side. |
| **T−** | **The hard cell. Decided here, not afterwards.** See the rule below. | **REJECT.** Direction is against and nothing is bought. Record and close. | **REJECT**, unambiguously — worse on both axes. Report `D−` to the CPU side. |

### The `T− / D+` rule — "dispersion reduced but throughput lower"

This is a real possibility on a GPU path and it must not be adjudicated after the fact. Both
quantities are percentages of the same median throughput, so they are directly comparable:

```
    level_cost   = 100 * (1 - median(ON) / median(OFF))                    [% throughput given up]
    floor_saving = spread(OFF)["p95_dev_pct"] - spread(ON)["p95_dev_pct"]  [% resolution gained]
```

1. **Hard veto first.** If `level_cost >= 4.581` (the current floor), **DO NOT ADOPT** in any form.
   A change that costs more level than the instrument can currently resolve is not a measurement
   improvement, it is a regression wearing one.
2. Otherwise, **ADOPT for serving iff `floor_saving > level_cost`** — the shim is worth its price
   only when it buys strictly more decision resolution than it costs in throughput.
3. Otherwise (`level_cost < 4.581` but `floor_saving <= level_cost`): **DO NOT ADOPT for production
   serving. ADOPT AS A MEASUREMENT INSTRUMENT ONLY** — future A/B work may run both arms with the
   shim ON to get a tighter comparison, while the champion continues to be *served* with it OFF, and
   every headline number stays an OFF-arm number. This instrument/production split is a legitimate
   and valuable outcome, and it is registered **now** precisely so that choosing it later cannot look
   like rationalisation.

### Reporting a `D−` to the CPU side

Any `D−` result must be filed back to INF-70: it is evidence the shim's effect is
configuration-dependent in the dispersion dimension as well as the level dimension — which
champion-3's own leave-one-out already demonstrated for level. It does not by itself overturn the CPU
adoption, which was measured on its own configuration; it bounds the claim's transferability.

### The controls-disagree verdict

Registered as a stopping condition, not a verdict: see §5.4. If the two independent controls disagree,
**that disagreement is itself the finding, the run stops, and no T or D verdict is issued.**

---

## 5. What would make the result inadmissible

The whole run is inadmissible unless every item below holds. Items marked *(couple)* invalidate one
couple, which is replaced within the 4-couple budget; items marked *(run)* end the run.

### 5.1 Host not quiet *(couple, and run at start)*

* **Refuse to start** if `os.getloadavg()[0] > 8.0`.
* Record `loadavg` before and after **every** launch. A launch whose pre-launch 1-minute load exceeds
  **12.0** is invalid *(couple)*. The threshold allows for the previous launch's 8 pinned threads
  still decaying out of the 1-minute average, plus 4 of slack; it is not a licence for a foreign
  tenant.
* The recipe pins to `184-191`, but that does **not** isolate it: every logical CPU on this host
  shares a physical core with 0-95, and pinned GPU host threads have been measured degrading the CPU
  floor 9× (0.80 → 7.2 %). The coupling runs both ways. A busy host is not a measurable host.
* The autokernel loop must be DOWN, as it was for the 4.581 % floor.

### 5.2 Any concurrent llama process *(run)*

Checked with the loop's own name-blind instruments. **Never `pkill`/`pgrep` on a name pattern on this
host** — any name pattern is a wildcard over other sessions' processes.

* `claim.hold()` — the `flock` on `/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock`, held for the entire
  window and re-verified at close. **A claim is acquired, never observed** (invariant 5): reading
  `rocm-smi` and concluding the device is free is a TOCTOU race. `claim.hold` refuses rather than
  queueing, and raises if the claim did not survive the window — in which case every measurement
  inside it is discarded.
* `residency.kfd_processes()` must read **0** before the run starts and **exactly 1** (our own server)
  at every in-launch sample. Any second KFD process ends the run.
* `residency.vram_bytes()` must be below `residency.RESIDENT_FLOOR_BYTES` (1 GiB) before the run
  starts.

### 5.3 Recipe or binary identity mismatch *(run)*

* `recipe.recipe_hash` is recorded on **every** launch record, per arm. Any change between launches of
  the same arm is inadmissible. The two arms' hashes must differ (they do, by construction: `env` and
  `env_readback` are inside the digest).
* The build's identity — sha256 of `bin/llama-server` and of `bin/libllama-common.so.*` — is recorded
  on every launch. One binary, unchanged, for the whole run.
* The build must be a HIP build of `ef81196d5` with the `gfx90a-house-v1` recipe, and its
  `provenance.json` must say so.
* **`cpu_list` must be `184-191`.** The 4.581 % floor was calibrated under that pin; an unpinned run
  is a different measured condition (R23-49).
* **Note on the standing floor.** Adding `env_readback` and an `env` override changes `recipe_hash`,
  so *neither* arm's hash equals the one the standing 4.581 % floor was calibrated under — and that
  floor was measured on `bff30cebe`, not `ef81196d5`. **This run never compares ON against 4.581 %.**
  Its only internal contrast is OFF vs ON, measured in the same window. The OFF arm's own
  `p95_dev_pct` is reported against 4.581 % as a **window-comparability check** — descriptive, not
  gating: a wild divergence there says this window is not like the fold window and the result should
  be read with that in mind.

### 5.4 An arm whose env did not actually take effect *(run — FATAL)*

> **"I set the knob" is not evidence the knob took effect**, and the closely-related standing rule is
> that *a null from a knob that is not in the binary is not evidence about the knob* (CLAUDE.md,
> Process Management; origin INF-70 C9, where a `libggml-cpu.so` two days older than its own fix cost
> two agents two days of diagnosing a defect that did not exist).

**Control 0 — the knob is in the artifact that will actually be LOADED.** Before the first launch the
runner resolves, via `ldd` under `Recipe.server_env(build_dir)` — the exact environment
`serving._measure_once` launches with — which `libllama-common.so` the loader will map, and scans
**that file** for the marker string
`INF70_CHAMPION3_PROCESS_THP_DISABLE=DEFAULT_OFF;OPT_IN=GGML_NOHUGEPAGE_PROCESS=1`. If it is absent,
the run **refuses to start**.

Three properties of that check, each load-bearing:

* **It scans the library, not the executable.** `common/common.cpp` compiles into
  `libllama-common.so`; `llama-server` is an 18 KB stub that links it. Scanning the executable
  returns a *false negative* on a build that fully contains the shim — see the methodological note
  in §6.
* **It scans the LINKED library, not a glob match.** `champ2/build-hip/bin` alone holds five
  vintages of `libllama-common.so.0.0.*` from five different dates, so a glob-any-match check could
  pass on a stale sibling the loader would never load — the three-ggml-generations hazard in another
  form. A library that resolves from **outside** the build directory is refused outright.
* **It carries a NEGATIVE CONTROL.** The same scan is run against a build known to predate the fold
  (`/mnt/raid0/llm/tmp/build-cor-445e93a8`, resolving `libllama-common.so.0.0.10194`) and **must
  reject it**. A scan that cannot fail says nothing about the scan that passed. If that tree is ever
  reclaimed the check fails until `--negative-control` names another pre-fold build.

The sha256 of the resolved library is recorded on every launch, so the artifact that carried the shim
is pinned in the run record — not merely a build directory that contained one somewhere.

**Control 1 — the primary positive control, and it is NOT this runner's code.** It lives in
`serving.py` as a declarative `Recipe.env_readback`, enforced by `serving.verify_env_readback` inside
`_measure_once`, **after the health check and before the warmup round**, so the arm is proven on the
live process before a single measured token. Both arms declare one entry:

```json
{"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS", "expect": {"1": "0", "unset": "1"}}
```

One declaration asserts **both** directions:

| env of this launch | required `/proc/<pid>/status` `THP_enabled` | on mismatch |
|---|---|---|
| `GGML_NOHUGEPAGE_PROCESS=1` | `0` (`PR_SET_THP_DISABLE` in force) | `EnvReadbackFailed` — the knob did not take |
| variable absent | `1` (THP allowed) | `EnvReadbackFailed` — **the control arm is secretly the treatment** |

It is fail-closed in every mode: a mismatch raises, a field the kernel does not expose raises, and an
unreadable `/proc` raises. And an env state the declaration does not cover is **refused at
construction** — so a one-sided declaration makes the control arm impossible to *build* rather than
silently unchecked. **The asymmetry is the whole point:** verifying only that ON is on lets a
mislabelled CONTROL through, and a control that is secretly the treatment is unrecoverable after the
fact — the experiment would report a null and the null would be an artefact.

**This runner does not re-implement any of that.** It declares the check in both recipes, asserts at
preflight that each arm's `readback_expectations()` is non-empty and resolves to the expected value,
and installs a *delegating* wrapper around `serving.verify_env_readback` that records, per launch,
that the check **fired**, on which pid, and what it observed. A launch in which the wrapper never
fired is FATAL: an unfired control is an unchecked arm.

Every reading is appended, with pid and the launch's env, to **`thp_proof.txt`** alongside the
results, so the proof survives independently of the log and of the JSON.

**Control 2 — the second, independent control: `AnonHugePages`.** Not in `serving.py`, so this runner
does implement it. Per launch it records `{thp_enabled, rss_kb, anon_huge_kb, anon_huge_pct}` from
`/proc/<pid>/status` and `/proc/<pid>/smaps_rollup` at two fixed points — at first `/health` (the
moment the readback fires, using the pid the wrapper hands it) and at the last successful poll before
teardown. The flag proves the `prctl` took; the AnonHuge fraction proves it did to the allocation what
it was meant to do (expected ≈0 % of RSS when ON, because `prctl` also blocks khugepaged from
collapsing pages later).

> **Registered now: if the two controls ever disagree, that disagreement is itself the finding and
> the run stops.** Concretely: an ON-arm launch with `thp_enabled == 0` **and**
> `anon_huge_pct >= 1.0 %` at the pre-teardown sample stops the run and is reported as a controls
> conflict — no T or D verdict is issued. The gate is one-directional by construction: the OFF arm's
> `AnonHugePages` is recorded but **not** gated, because a low reading there means only that
> khugepaged had not yet collapsed anything, which is not a contradiction. This matters: the CPU
> session measured `AnonHugePages` at 0.06 % of Rss at load and ~6 % minutes later on the *same*
> process, which is exactly why it is not the primary discriminator and why the sampling points are
> fixed in advance rather than chosen later.

### 5.5 GPU residency not proven *(couple)*

**"I invoked the HIP build" is not evidence of a HIP run, and `ldd` cannot prove one** — llama.cpp
*dlopens* `libggml-hip.so`, so the executable shows zero HIP linkage either way, while
`/etc/environment` puts the CPU build early in `LD_LIBRARY_PATH` and three ggml generations live on
this host. Residency is sampled **during** each launch with `residency.Sampler`; a launch is invalid
unless `proof["resident"]` is true (peak VRAM ≥ 1 GiB) and `proof["peak_kfd_processes"] == 1`.
A launch with `sclk_max − sclk_min > 50 MHz` is invalid *(couple)* — a measurement taken across a
governor transition is partly a measurement of the governor.

**Stated limitation.** `serving._measure_once` builds its environment from `Recipe.server_env`, which
pins `LD_LIBRARY_PATH` to the build's own `bin` but — unlike `bench`, `gates` and `hotspots` — does
**not** go through `residency.loader_env`, which additionally unsets `HSA_OVERRIDE_GFX_VERSION` and
appends `/opt/rocm/lib`. The serving launch is therefore the one path on which a stray
`HSA_OVERRIDE_GFX_VERSION` in the host environment would reach the process. This was left as-is
deliberately at the U3 change, and this design does not alter it. **The runner records the host's
value of `HSA_OVERRIDE_GFX_VERSION` on every launch** (expected: unset), so if it ever turns out to
matter the run is still interpretable after the fact rather than silently ambiguous.

### 5.6 Degenerate or crashed launches *(couple)*

`serving.ServerDied` — a server that exits during load, fails to reach `/health` inside the boot
timeout, or returns fewer than `n_predict/2` tokens on any slot — invalidates its couple. More than 4
such events end the run.

---

## 6. What this experiment cannot answer

* **Whether the effect transfers to any other model, quant, `np`, context or pin.** This is exactly
  one recipe: Qwen3.8-27B Q8_0 with the DFlash2 drafter, `np=4`, `ctx=16384`, `-b/-ub 2048`,
  `-ngl 99`, `-fa on`, `f16` KV, `--no-kv-unified`, pinned to `184-191`. The CPU side has already
  demonstrated that this knob's *sign* is configuration-dependent; transfer must be measured, never
  assumed.
* **Whether it interacts with HIP graph capture.** The rescued HIP-graph patch (R21-5) is not in this
  binary. If it lands, this result does not carry across it and R23-58 must be re-run.
* **The mechanism.** On `-ngl 99` the weights and KV live in VRAM. Any effect here is on host-side
  glibc-heap state, and this design does not instrument which structure. A positive result would be a
  measured fact without an explanation, and must be reported as one.
* **Any throughput magnitude.** The level boundary is a sign test. Direction only. Sizing for a
  magnitude at this recipe's 3.3 % between-launch CV would need roughly 100 launches for ±1 % and
  ~400 for ±0.5 % — the CPU session's own arithmetic, and the reason nobody should quote a number off
  this run.
* **Stability of any recalibrated floor over days.** This is one contiguous ~1-hour window. The CPU
  session measured between-launch sd at 2.793 % within a 52-minute block and 5.081 % across 3.7 hours;
  a floor measured in an hour is an hour's floor.
* **Correctness.** The shim changes page backing, not arithmetic — placement only, by construction.
  No correctness gate is run here and none is claimed.
* **Anything about production.** The fleet serves the frozen `production-consolidated-v9` kernel, not
  `ef81196d5`. Nothing in this run is a statement about production serving.

### Methodological note — absence from one artifact is not absence from the build

Recorded because it nearly cost an unnecessary rebuild, which on this host would have contended with
another session's live measurement.

The standing rule is that **a null from a knob that is not in the binary is not evidence about the
knob** (INF-70 C9). Its corollary, which is what went wrong here, is that **you must check the
artifact that actually carries the code, and you must prove the check discriminates.** The shim's
marker lives in `libllama-common.so`, because that is where `common/common.cpp` compiles to;
`llama-server` on these builds is an 18 KB stub. A scan of the executable therefore reports "absent"
on a build that fully contains the shim, and the conclusion drawn from it — *no usable build exists,
rebuild the champion* — was wrong.

Two corrections follow, and both are now in Control 0: **resolve the artifact the loader will
actually map** (`ldd`, refusing anything outside the build directory — a glob is not enough when one
`bin/` holds five vintages of the same library), and **run the check against a build known not to
contain the code**, so a pass is evidence rather than a tautology. A verification step that has never
been shown to fail has not been shown to work.

---

## 7. The build this runs on

**`/mnt/raid0/llm/tmp/build-fold-ef81196d5`** — the FOLD-2 candidate build of the current champion.
**No rebuild is required.**

| property | value | how it is established |
|---|---|---|
| source tree | `/mnt/raid0/llm/tmp/fold-ef81196d5-src` | `CMakeCache.txt` → `CMAKE_HOME_DIRECTORY` |
| commit | `ef81196d5bdd4190b46dff4ae7eecc333a46c8ce`, working tree **clean** | `git -C <src> rev-parse HEAD` + `status --porcelain` |
| built | 2026-09-08 09:25, HIP | build mtimes |
| build flags | `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx90a`, `GGML_HIP_ROCWMMA_FATTN=ON`, `GGML_NATIVE=ON` — the four `gfx90a-house-v1` divergence-free flags | `CMakeCache.txt` |
| shim artifact | `bin/libllama-common.so.0.0.10301`, carrying the full marker `…DEFAULT_OFF;OPT_IN=GGML_NOHUGEPAGE_PROCESS=1;MASTER_OFF=GGML_NOHUGEPAGE=0` | Control 0, sha256 recorded per launch |
| linkage | `llama-server` resolves `libllama-common.so.0` to that file, **inside the build directory** | `ldd` under `Recipe.server_env` |
| negative control | `/mnt/raid0/llm/tmp/build-cor-445e93a8` → `libllama-common.so.0.0.10194`, **zero** marker hits | the same scan, which therefore discriminates |

This build carries no `provenance.json`, so provenance is established **structurally** rather than by
assertion: CMakeCache names the source tree, the source tree's HEAD is the champion commit, the tree
is clean, and the four house flags match. That chain is stronger than a `provenance.json` claim, and
the runner accepts either. **Residual, stated rather than hidden:** a source tree can move *after* a
build, so HEAD describes the build only as well as the tree's mtime allows.

Two builds that are **not** to be used, recorded so nobody reaches for them:
`/mnt/raid0/llm/autokernel/loop-memory/anchor-gen-021` (`bff30cebee0d`, pre-fold) and
`/mnt/raid0/llm/tmp/champ2/build-hip` (built 2026-09-01, pre-fold). Both genuinely lack the shim.

---

## 8. Artefacts this run must produce

Written incrementally, never only at the end:

| file | content |
|---|---|
| `launches.jsonl` | one JSON object per launch, appended and fsynced as it completes: couple index, arm, position-within-couple, `aggregate_tok_s`, `recipe_hash`, `recipe_env`, `recipe_describe`, build digests, readback expectations + observed + fired, both AnonHugePages samples, `residency.Sampler.proof`, host state before and after (`load1`, KFD count, VRAM, sclk, `HSA_OVERRIDE_GFX_VERSION`), wall-clock |
| `thp_proof.txt` | one line per launch: `<sess> pid=<pid> THP_enabled=<0\|1> sessenv=[...]` — the positive control, surviving independently of the JSON |
| `state.json` | the running tally: valid couples, signs, claim (i) look status, replacements used |
| `arm-off.json`, `arm-on.json` | `epyc.autokernel.serving_floor.v1`-shaped per-arm records with the full `spread` block from `serving._spread`, marked `interleaved: true` |
| `VERDICT.json` / `VERDICT.md` | written once, at the stop rule: both verdicts, the §4 cell, the registered action, the achieved power |

The runner **must not** write to `/mnt/raid0/llm/autokernel/loop-memory/` in any form, must not
promote, fold, or advance any anchor, and must not touch the standing floor file. Recalibrating the
serving floor under the ON arm — the follow-on that a `D+` verdict calls for — is a **separate**,
separately-authorised run.

---

*Registered by the R23-58 planning session, 2026-09-08. No data existed when this was written.*
