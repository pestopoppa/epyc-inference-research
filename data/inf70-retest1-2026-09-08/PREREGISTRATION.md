# INF-70 RETEST-1 — PRE-REGISTERED ANALYSIS PLAN

**Written and frozen BEFORE any arm ran.** Results sections are deliberately empty here; they
live in `REPORT.md`. Any deviation from this document must be recorded in `REPORT.md` as a
labelled deviation with its reason.

**Frozen at 2026-09-08T09:30:11Z (epoch 1788859811). Host: Beelzebub. Author: RETEST-1.**

---

## 0. Scope

Re-measure levers that were **unresolved against the instrument**, not refuted by mechanism, on a
harness whose A/A floor is ~4x tighter. Explicitly OUT of scope, mechanism-refuted, will not be
tested: `inf10-gemv-fusion`, SYNC-20 yield floor, SYNC-17 FIX-3 column split.

Levers, in priority order: **CHAMP-2 whole-process THP shim**, **FIX-1 (`GGML_SCALE_SPLIT`)**,
**SYNC-18 (small gathers single-task)**, then SYNC-16 and SYNC-13 only if the floor holds and the
block allows.

## 1. Baseline and binary

Baseline is the fold candidate **`ef81196d5`** (`inf70/fold-candidate-20260908`).

**Established before any measurement**: `git diff --name-only 9c4f73e29 ef81196d5` returns
**ten files, all under `ggml/src/ggml-cuda/` or `ggml/src/ggml-hip/`, and nothing else.** A
CPU-only build (`-DGGML_CUDA=OFF -DGGML_HIP=OFF`) of the fold candidate is therefore compiled from
source **identical** to a CPU-only build of champion-3 `9c4f73e29`.

Consequently the baseline binary is **`agents/harness1/bin-h1`**, version `10242 (eae02f2dc)` =
champion-3 plus HARNESS-1's runtime knob page, CPU-only. This is a *derivation*, not an assumption:
if the file list above is ever shown to be incomplete, every number in this campaign is void.

The knob page is measurement infrastructure, not a lever: HARNESS-1 measured its refresh at
199.3 ns/graph vs 201.2 ns without (min of 7, below noise, negative in sign) and 24/24 byte-identical
output against the shipped champion-3 binary.

## 2. Metric

Token-weighted decode rate, **`sum(pred_n) / sum(pred_ms)`** over the 24-prompt production mix
(`e3-alpha/prompts.json`, 8 coding / 8 reasoning / 8 general), `pred_n >= 16` floor, greedy,
`max_tokens=200`, `cache_prompt=false`, `enable_thinking=false`. Prefill (`pp`) is recorded as a
**secondary diagnostic only**: decode-only movement vs decode+prefill movement discriminates a
memory/page-state effect from host drift.

## 3. Unit of observation — this is the core of the plan

**Per-prompt paired statistics are pseudo-replication and are NOT used for inference.** The 24
prompts inside one arm share one quiet-or-noisy moment; they are ONE observation. Per-prompt win
counts may be reported as descriptive colour and carry **no p-value**.

| lever | knob is consumed at | unit of observation | why |
|---|---|---|---|
| FIX-1 `GGML_SCALE_SPLIT` | graph execution | **one ARM** | switchable inside a hot process |
| SYNC-18 | graph execution | **one ARM** | switchable inside a hot process |
| SYNC-16, SYNC-13 | graph execution | **one ARM** | switchable inside a hot process |
| **CHAMP-2 THP shim** | `prctl(PR_SET_THP_DISABLE)` **before the 92 GB allocation** | **one SESSION (one server process)** | process-scoped; all arms in a session share one THP state and are NOT independent replicates of it |

The THP row is a pre-registered demotion of the brief's priority-1 lever: it is **not**
hot-switchable, so its replicate is a process launch, and its achievable n in one block is far
smaller than the other levers'. A session's value is the **mean of its arms**; the arms reduce
within-session noise but do not add THP replicates.

## 4. Primary statistic

Two-sided **exact/Monte-Carlo permutation test on arm (or session) labels**, statistic = difference
of group means of the token-weighted rate. Exhaustive when C(n,k) <= 200000, else 200000 random
relabellings. Reported: observed ratio B/A, permutation p, and a **percentile bootstrap 95% CI on
the ratio** resampling whole arms (sessions for THP).

**A result is a CLAIM only if p < 0.05 AND the 95% CI excludes 1.0.** Otherwise it is a
**NON-CLAIM**, reported as "unresolved at n=..., detectable effect at this floor was +/-X%".
A non-claim is a complete result and will not be softened into a direction.

## 5. Design — alternation and adjacency

Arms alternate `A, B, A, B, ...` inside one hot session (METH-2: A/A comparisons must be
**adjacent and back-to-back**; non-adjacent A arms measure host drift, not measurement error).
THP alternates at session granularity: `OFF, ON, OFF, ON, ...` inside a single region hold, so
slow host drift is common-mode across the pairing.

Arm order within a session is fixed and written down before the session starts.

## 6. Contention screen — threshold pre-registered HERE, before any data

Instrument: `/workspace/repos/epyc-inference-research/scripts/utils/foreign_load.py`, sampling
**live `/proc/<pid>/stat` utime+stime deltas** every 10 s **during** the arm, with bench-core SMT
siblings expanded from `/sys/.../thread_siblings_list`. `ps %CPU` is a lifetime average, cannot see
a burst, and is NOT used. (HARNESS-1's `cores_sampler.sh` used `ps -eo pcpu` and truncated at
`head -12`; it is replaced, and that replacement is a deviation from "reuse the harness" made
deliberately and recorded here.)

Foreign load is split into two accounts:

* **ANNOUNCED lane** — processes whose `Cpus_allowed_list` is a subset of `184-191`, the GPU
  session's committed pin. This is a *standing condition of the whole campaign*, present in every
  arm, therefore common-mode across the alternation. Recorded, never a drop reason.
* **OTHER** — everything else whose allowed set intersects the bench logical CPUs.

Pre-registered rules, decided now:

| condition on OTHER foreign %CPU (100 = one full core) | verdict |
|---|---|
| `max < 200` | CLEAN — arm used |
| `200 <= max < 400` | FLAGGED — arm used, flagged in the report, sensitivity re-run of the primary test excluding flagged arms is also reported |
| `max >= 400` | **DROPPED** — arm excluded before any statistic is computed |

400% = four fully-busy cores. Rationale: HARNESS-1's one ruined arm carried a 3218% peak from a
`cc1plus` build; unpinned low-intensity noise moved nothing. The discriminator is the **peak, not
the median**, so the rule is written on `max`.

**A dropped arm takes its pairings with it**: the alternation is re-paired after drops, and any A
arm left without an adjacent partner is reported as unused. Median AND max are printed per arm
regardless of verdict.

If `>= 30%` of arms in a session are DROPPED, the session is void and re-run rather than analysed.

## 7. A/A calibration — HARD GATE, evaluated before any lever arm

Five contiguous A arms, identical knobs, in one hot session on the baseline binary. Statistic:
**p95 of `|a-b| / mean` over all 10 unordered pairs**, the same estimator HARNESS-1 and autokernel
report, so the numbers are comparable.

| pair p95 | verdict |
|---|---|
| `<= 1.20%` | **PASS** — floor reproduced; proceed to lever arms |
| `1.20% < p95 <= 2.00%` | **DEGRADED** — proceed ONLY after recomputing n from the observed sd, and report every lever against the degraded floor |
| `> 2.00%` | **STOP** — report the floor and halt. No lever arms. |

The reference is HARNESS-1's **0.80%** hot floor; 1.20% is that plus a 50% margin.

**A STOP is a publishable result, not a failure**: it is direct evidence for OP-41 on what the
shared-host contention floor costs, and it will be reported as such.

## 8. Power — computed from the A/A sd BEFORE lever arms are spent

With arm-level sd `s` (in %) estimated from the A/A block, the two-sided permutation test on
`n` arms per side detects, at 80% power and alpha=0.05, approximately

    delta_min ~= 2.8 * s * sqrt(2/n)

This is computed and **written into the report before the lever arms run**, and each lever's
section states the effect size that was detectable at the n actually afforded. Priority for a
lever whose expected effect is far below `delta_min` at any affordable n is dropped rather than
spent — spending arms on an undetectable effect is how this campaign produced its unresolved rows.

**MTP preference**: where a lever permits it, arms use the MTP block, whose sd is ~2.8x smaller
than plain (0.61% vs 1.71%) and therefore needs ~8x fewer arms.

## 9. Correctness gate — before any speed number is quoted

**Bit-identity at PRODUCTION prompt length.** Full 256-token greedy streams at ~40, ~90 and ~200
prompt tokens, **plain AND MTP**, sha256 per stream, lever ON vs lever OFF. Short prompts gate
nothing (a P0 was missed that way).

Note the direction of evidence, learned by HARNESS-1 getting it backwards: for a lever that is
**bit-identical by design**, matching shas **cannot** prove the knob switched. Switching is proven
separately by (a) the server's own `ggml knobs: seq=` readback incrementing once per arm, and
(b) the knob page appearing in `/proc/<pid>/maps`. A lever that is *supposed* to change output
must show differing shas, and failure to differ is a failed switch, not a passed gate.

For SYNC-18 specifically, before any arm is spent: **prove the knob reaches the dispatch decision**
(the value is known-inert at the kernel in at least one prior report). If it does not reach, the
lever is reported as UNTESTABLE-AS-BUILT with the required patch described, and no arms are spent.

## 10. Conditions recorded with EVERY number

Every quoted figure carries, or it is not quotable:

* hot harness, arm count `n`, and the contention verdict per arm;
* **"GPU loop down/draining, autokernel serving pinned 184-191"**;
* baseline = fold candidate `ef81196d5` (CPU build == champion-3, per section 1);
* build id asserted on **object files**, not the linked `.so` (the linker is not reproducible on
  this host; the compiler is).

**Never compare a hot absolute against a cold absolute.** The same binary reads **+4.36%** hot vs
cold with byte-identical output. Both arms of every comparison come from the same harness, and
cross-harness absolutes are not compared even informally.

## 11. Stop conditions

* A/A gate fails per section 7 -> stop, report.
* A lever's correctness gate fails -> that lever is reported as a correctness failure; no speed
  number for it is quoted at all.
* The block ends -> stop at the arm boundary and report what is resolved. **Levers not reached are
  reported as not reached, never as unresolved** — those are different statements.
