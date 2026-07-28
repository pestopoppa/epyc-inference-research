# P2-5a — Shed-trade measurement spec (DESIGN ONLY; not runnable yet)

**Status**: DESIGN. **Nothing here has been run.** No inference was performed producing this
document, and the protocol it proposes is **not yet ratified** — see §8.
**Filed**: 2026-07-28 by `claude-gpu-lane`.
**Owning handoff**: [`epyc-root/handoffs/active/gpu-serving-tie-in-program.md`](../../../epyc-root/handoffs/active/gpu-serving-tie-in-program.md) task **P2-5a**.
**Motivating analysis**: `epyc-orchestrator/docs/gpu-shadow-lane.md` §3.4 (D1 admission class 3).
**Governing constitution**: `epyc-root/MEASUREMENT.md` (P-BENCH-1/2/3, P-GPU-1, P-SPEED-OBJ).

---

## 1. The question, stated so it can be answered

D1's admission class 3 ("shed batch") moves `worker_general`-class batched work to the GPU lane
when the CPU is under stress. §3.4 established that this is **partially self-defeating**: the lane
is not a pure GPU resource — its 8 host threads occupy SMT siblings 184-191, whose physical cores
88-95 are atomic region **`q3`**. Shedding therefore consumes CPU in the region most likely to be
contended precisely when CPU is stressed.

**The decision question:** *does shedding increase total useful work per unit time, under the CPU
stress conditions that would trigger it?*

### 1.1 A correction to §3.4's framing, made before it can mislead

§3.4 phrases the trade as *(GPU throughput gained) − (q3 CPU throughput lost)* in **t/s**. That
framing is convenient prose and a poor metric, for two reasons:

1. **t/s is not commensurable across the two sides.** The CPU side runs `worker_general`
   (gemma4-26B-A4B MoE); the GPU tenant is a 27B dense. A token from one is not a token from the
   other — different tokenizers, different verbosity, different work per token. Summing or
   differencing their t/s produces a number with no referent.
2. **`P-SPEED-OBJ` already settled the axis**: `task_rate` (questions / eval-wall-hour), with t/s
   retained as *host-health telemetry only*. Differencing t/s here would contradict the standing
   objective.

**Primary metric is therefore `task_rate`** — tasks completed per eval-wall-hour, summed across
both devices, on one fixed task corpus. t/s is recorded per side as telemetry and diagnostic, never
as the decision quantity.

### 1.2 Measure the net directly, do not reconstruct it

The naive design measures GPU gain and CPU loss separately and subtracts. That inherits both
measurements' noise (CV ≈ 9.1% each) into the difference and, worse, measures the two halves under
conditions that do not co-occur — which is the entire phenomenon under study.

**The net is measured directly** as a paired comparison of two whole-system configurations running
the same corpus in the same wall window (§3). The separate halves are retained as *diagnostics that
explain the sign*, not as the inputs the answer is computed from.

### 1.3 This measurement does not require building class 3

Deliberate: the batch split is driven by the **harness** (eval-path fan-out with forced role
targets), not by an admission controller. So the measurement that justifies class 3 does not
presuppose class 3. Building the feature first and measuring after is how a negative result becomes
a feature reversal instead of a decision.

---

## 2. Inputs (all must exist and be pinned before a run)

| Input | Source | Pinning requirement |
|---|---|---|
| Task corpus | The `worker_general`-class batch corpus, drawn once and frozen | Manifest with sha256; identical across every arm and rep |
| CPU fleet lineup | `orchestrator_stack.py status` + runtime-facts manifest | Terminal lineup, declared; must be the production shape, not a bench shape |
| GPU lane tenant | `orchestration/gpu_shadow_lane_tenancy.yaml` | tenant id, path, bytes, sha256, launch mode, `draft_n_max` |
| Lane serving shape | `serving_shape` block, ceiling-validated by `load_serving_shape()` | np × per-slot ctx; must be inside the np_ceiling table (P2-3d enforces) |
| Kernel | production-consolidated-v8 `67a433bf4`, binary `10107` | **Production-named kernel required** (P-GPU-1 provenance rule) |
| q3 co-tenant set | `gpu_shadow_lane_stage0.py recert` | The generated set, not a hand-list: frontdoor 8380, worker_general 8382 + 8072, ingest 8485, vision_escalation 8087, architect_general 8083 |
| Contention matrix | `orchestration/contention_matrix.yaml` | Certified fresh for the current topology hash |
| Stress definition | §4.2 | The stress level is an INPUT, not an emergent property |

---

## 3. Arms

Four arms. A0 and A2 answer the question; A1 and A3 explain the answer. All arms run the same
frozen corpus.

| Arm | CPU fleet | GPU lane | What it isolates |
|---|---|---|---|
| **A0 — CPU-only under stress** | full stress load, all tasks on CPU | **absent** (not launched; q3 free of the lane) | Baseline total `task_rate`. The status quo class 3 would replace |
| **A1 — lane resident, idle** | full stress load, all tasks on CPU | **resident, serving nothing** | **The residency tax.** Does merely holding 8 host threads on q3 cost CPU throughput, before any work is shed? The control most likely to be skipped, and the one that separates "shedding is bad" from "the lane's mere presence is bad" |
| **A2 — shed active** | stress load minus the shed fraction *f* | resident, serving the shed fraction *f* | Total `task_rate` under shedding. **A2 − A0 is the answer** |
| **A3 — GPU reference, CPU quiesced** | quiesced (declared) | resident, serving the shed fraction | The un-contended GPU ceiling, so the GPU-side contention tax (A2's GPU half vs A3) is visible rather than assumed |

**Primary quantity:** `net_task_rate = task_rate_total(A2) − task_rate_total(A0)`, higher-better,
where `task_rate_total` counts tasks completed on **both** devices in the same wall window.

**Diagnostics (each explains a possible sign):**
- `residency_tax = task_rate(A1) − task_rate(A0)` — expected ≤ 0. If this alone explains a negative
  net, the finding is "the lane must not be resident during CPU stress", which is a *different and
  cheaper* conclusion than "shedding does not work".
- `gpu_contention_tax = task_rate_gpu(A2) − task_rate_gpu(A3)` — how much the co-resident CPU load
  costs the GPU side.
- `cpu_displacement = task_rate_cpu(A2) − task_rate_cpu(A0)` — expected < 0 by construction (fewer
  tasks *and* fewer effective cores); reported so the net's composition is legible.

### 3.1 Shed fraction is swept, not assumed

*f* ∈ {0.25, 0.50, 1.00} at minimum. **The optimum may be interior** — a small shed may pay for
itself while a full shed does not, because the residency tax is paid once but the CPU displacement
grows with *f*. Reporting a single *f* would risk concluding "shedding does not work" from one
badly-chosen operating point. *f* = 0 is A0 by definition.

---

## 4. Controls and confounders

### 4.1 Ordering

Arms are **interleaved and order-randomized within each rep block**, never run as
A0×n → A2×n. Thermal drift and page-cache state both trend over a session; a blocked design
aliases that trend onto the arm effect.

### 4.2 Stress is defined, not observed

"Under CPU stress" must be an operating point, not a vibe. Stress level = a fixed concurrent-request
depth against the CPU fleet, chosen so the q3 quarter is saturated in A0 and declared in the run
header. **At least two stress levels** (e.g. saturating and 0.5× saturating): the sign of the trade
may depend on it, and class 3's premise is specifically the stressed regime. A trade measured only
at saturation cannot be generalized downward.

### 4.3 Live affinity, not intent

`affinity_preflight.py` must verify the **live** masks of every instance and of the lane's host
threads. The topology hash certifies intent, not reality — a run whose affinity was never verified
live is an observation regardless of its other rigor.

### 4.4 Memory and page-cache state

The GPU tenant load streams ~27 GiB off raid0. Page-cache pressure and post-`drop_caches` NUMA
eviction (a re-read pins one node) are real confounders on the CPU half. Required: host-health tier
per P-BENCH-1, `drop_caches` + **NUMA-interleave re-warm** (never a bare re-read), and lane
pre-warm completed *before* the measurement window opens in every arm that has a resident lane.

### 4.5 SMT non-linearity

The lane's 8 threads are SMT siblings of physical cores 88-95. SMT contention is not linear in
thread count, so the residency tax cannot be extrapolated from a 4-thread or 16-thread measurement.
Host-thread count is **fixed at 8** — the shape every v8 np×context grid measured. Varying it would
invalidate the ceiling table the lane's serving shape rests on.

### 4.6 Claims are acquired, not observed

The run holds `q3` via `epyc-orchestrator/scripts/region-lock` and the GPU device claim for its
whole duration. Observing that the lane looks free is TOCTOU and is not exclusion (BUS_PROTOCOL
rule 7).

### 4.7 Eval path only

All traffic goes through the eval-path fan-out with forced role targets — never live `/chat`.

---

## 5. Metrics and claim grammar

Every decision-gating number carries `(metric, protocol-id, n/reps, date, attestation ref)`.

| Metric | Direction | Grammar (illustrative shape only — **no values exist yet**) |
|---|---|---|
| `net_task_rate` | higher-better | `shed net +N tasks/eval-wall-h at f=0.5, stress=sat [P-SHED-1, n=10, YYYY-MM-DD, attest <ref>]` |
| `residency_tax` | higher-better (≤0 expected) | `lane residency tax −N tasks/h [P-SHED-1, n=10, YYYY-MM-DD, attest <ref>]` |
| `gpu_contention_tax` | higher-better (≤0 expected) | as above |
| per-stream p50/p95 latency | lower-better | required by P-BENCH-3 for any batched-slot claim; reported per side |
| decode t/s per side | higher-better | **telemetry only** — never a decision row (§1.1) |

Report **median and MAD** (P-BENCH-1 / P-GPU-1 result grammar). Metric direction is stated on every
row; the standing table's rate noise is **CV ≈ 9.1%**.

### 5.1 Reps and MDE, pre-registered

- Per the P-BENCH-1 rule: **n ≥ 5** for effects ≥5%, **n ≥ 10** for effects ≤2%. Given rate noise
  CV ≈ 9.1%, a plausible single-digit-percent net requires **n ≥ 10 paired blocks**.
- The **MDE is computed and published with the result**, not after seeing it.
- Rate claims go through the improvement / non-inferiority e-process per P-SPEED-OBJ —
  **never a single trial**.

### 5.2 The null result is pre-registered as informative

If `|net_task_rate|` falls below the MDE, the verdict is **"no detectable trade"**, and that is a
decision, not a failed experiment: class 3 would add an admission class, a flag, a telemetry
surface and a failure mode for no measurable gain. Writing this down in advance is what stops a
null from being re-litigated as "needs more data" once someone has already built the feature.

### 5.3 Pre-registered decision rule

| Outcome | Verdict |
|---|---|
| `net > 0`, e-process confirms, gain exceeds the operator's complexity threshold | Class 3 may be built. The measured (f, stress) region becomes its **validated envelope** — outside it, refuse |
| `net > 0` but within MDE, or e-process inconclusive | **Do not build.** Re-measure only if a consumer decision depends on it |
| `net ≤ 0` at every swept *f* | **Close class 3 permanently.** Record in D1 as measurement-closed |
| `net ≤ 0` explained wholly by `residency_tax` | Narrower finding: **the lane must not be resident during CPU stress**. Class 3 stays closed; lane residency policy gains a stress-aware rule |

---

## 6. What makes this decision-grade vs observation-grade

**Observation-grade** (informs design, gates nothing) if *any* of:
- P-SHED-1 is not yet ratified into `MEASUREMENT.md` (§8) — **this is today's state**;
- run on any non-production-named kernel (P-GPU-1 provenance rule);
- any P-GPU-1 mandatory field missing (hardware state before AND after, host interference
  declaration, binary/model identity, run recipe, result grammar, attestation);
- affinity not verified live (§4.3), or contention matrix not certified fresh;
- reps below the §5.1 floor, or a single-trial rate claim;
- the CPU fleet was not the terminal production lineup.

**Decision-grade** requires **all** of: ratified protocol; production kernel v8; every P-GPU-1 field
present for the GPU half and every P-BENCH-2 requirement for the CPU half; live-affinity
attestation; fresh contention certification; n ≥ 10 paired blocks with published MDE; e-process
verdict; and an attestation ref. **No partial upgrades** — P-GPU-1's retro-certification rule
applies unchanged.

---

## 7. Gating — why this cannot run yet

**This measurement is not runnable today, and the blocks are not merely procedural.**

| # | Gate | State as of 2026-07-28 |
|---|---|---|
| G1 | **`q3` must be free** | **BLOCKED.** `region-lock status` shows `q3` **HELD** by `bench-e8-quality` (`e8-v5-r2-cadencefix-20260728T160917Z`) — deadline-bearing E8 work; re-verified still held at 2026-07-28T17:0xZ. The lane's host threads need exactly that region. Reclaim is quiesce-and-drain at the holder's own boundary, **never forced** (fabric axiom 4) |
| G2 | **Host-health tier satisfied** | Measured 2026-07-28: uptime **4d 3h** (booted 2026-07-24T13:51Z), so P-BENCH-1's **≤1wk tier** applies today — `drop_caches` + NUMA-interleave re-warm, *not* a reboot. That tier lapses **~2026-07-31**, after which P-BENCH-1 requires a reboot. The program schedules this work **post-reboot** regardless (Phase 1), which also satisfies the tier by construction. Host reboots are **operator-only**. Practical consequence: do not attempt to squeeze this into the pre-07-31 window — the campaign is multi-hour (§9) and the window is already spoken for by deadline-bearing E8/E5 |
| G3 | **Lane activated** | Requires the operator-gated Steps 0–7 (`docs/gpu-shadow-lane.md` §7). The registry is FROZEN (D3); activation is a registry change plus the choreography, and it needs G1 as a precondition |
| G4 | **P-SHED-1 ratified** | `MEASUREMENT.md` is a **human-amendment-only trust boundary**. Until the operator amends it, every number this spec produces is observation-grade regardless of execution quality |
| G5 | **Operator run grant + quiet window** | Benchmarks run only via codified recipes with operator approval; no AutoPilot or EvalTower batch in flight |
| G6 | **Region claims acquired** | `q3` + GPU device claims held for the run's duration (§4.6) |

G1 and G3 are ordered: the lane cannot be activated while q3 is held, so **G1 precedes G3**. G4 is
independent and can be resolved at any time, including before the hardware is free.

---

## 8. P-SHED-1 is a PROPOSAL, not a protocol

There is **no existing protocol for a cross-device contention trade**. P-BENCH-2 covers CPU
multi-instance aggregate, P-BENCH-3 covers batched slot decode, P-GPU-1 covers GPU throughput —
none covers *"work moved between devices, where the mover consumes the resource it is relieving"*.
That gap is real and this spec is shaped to fill it.

**It cannot fill it unilaterally.** `MEASUREMENT.md` is human-amendment-only; agents read it and
never write it. This document therefore **proposes** `P-SHED-1` as a composite protocol
(P-BENCH-2 for the CPU half + P-GPU-1 for the GPU half + the paired whole-system design of §3, on
the P-SPEED-OBJ `task_rate` axis), and asks the operator to ratify it as an appended amendment if
they judge the design sound. Ratification is a decision package for the operator, not an action
this session takes.

Until then: this spec is a design artifact. Any run performed under it yields observations.

---

## 9. Estimated cost (for scheduling, not a commitment)

4 arms × 3 shed fractions (A2 only) × 2 stress levels × 10 paired blocks is **not** the shape —
A0/A1/A3 do not multiply by *f*. Roughly: A0 and A1 at 2 stress levels, A2 at 3 *f* × 2 stress,
A3 at 3 *f*, each × 10 reps ≈ **130 measurement blocks**, plus lane launch/teardown per residency
change (~1–2 min each, D6's drained relaunch). Block length is set by the frozen corpus. This is a
multi-hour quiet-window campaign, not an afternoon — which is itself an argument for resolving G4
(ratification) *before* spending the window, so the output is decision-grade the first time.

---

## 10. Open questions for the operator

1. **Complexity threshold** — what net gain justifies building an admission class, a flag, a
   telemetry surface and a failure mode? Without a number, a marginal positive becomes a debate.
2. **Stress levels** — is "saturating + 0.5× saturating" the right pair, or does the real trigger
   regime sit elsewhere?
3. **Ratify P-SHED-1?** (§8) — or fold this under an existing protocol the operator considers
   adequate, accepting that no current protocol covers the cross-device case.
