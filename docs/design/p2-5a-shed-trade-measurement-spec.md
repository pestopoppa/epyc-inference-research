# P2-5a — Shed-trade measurement spec (DESIGN ONLY; not runnable yet)

**Status**: DESIGN. **Nothing here has been run.** No inference was performed producing this
document. `P-SHED-1` was **RATIFIED by the operator 2026-07-28** (§8); registration against
`MEASUREMENT.md` is a human-only path and is the operator's to execute — this session has not
touched that file. Two decision-rule inputs remain **UNRESOLVED** (§10).
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
- P-SHED-1's registration has not yet been executed against `MEASUREMENT.md` by the operator
  (the design is ratified, §8);
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
| G4 | **P-SHED-1 ratified** | **CLEARED 2026-07-28** — operator ratified (§8). Registration against `MEASUREMENT.md` is a human-only path and is being packaged for the operator; this session has not touched that file. Note the two §10 values are still unresolved: they do not block the measurement, they block turning it into a verdict |
| G5 | **Operator run grant + quiet window** | Benchmarks run only via codified recipes with operator approval; no AutoPilot or EvalTower batch in flight |
| G6 | **Region claims acquired** | `q3` + GPU device claims held for the run's duration (§4.6) |

G1 and G3 are ordered: the lane cannot be activated while q3 is held, so **G1 precedes G3**. G4 is
independent and can be resolved at any time, including before the hardware is free.

---

## 8. P-SHED-1 — RATIFIED by the operator 2026-07-28

There is **no existing protocol for a cross-device contention trade**. P-BENCH-2 covers CPU
multi-instance aggregate, P-BENCH-3 covers batched slot decode, P-GPU-1 covers GPU throughput —
none covers *"work moved between devices, where the mover consumes the resource it is relieving"*.
That gap was real; `P-SHED-1` fills it as a composite protocol (P-BENCH-2 for the CPU half +
P-GPU-1 for the GPU half + the paired whole-system design of §3, on the P-SPEED-OBJ `task_rate`
axis).

**Status: the operator has RATIFIED P-SHED-1** (program task P2-5b, 2026-07-28). A pre-validated
ratification command is being packaged for the operator to execute against `MEASUREMENT.md`.

**Registration remains a human-only path.** `MEASUREMENT.md` is a human-amendment-only trust
boundary: agents read it and never write it. This session has NOT edited it and will not. Ratifying
the design and registering the protocol text are two different acts — the first has happened, the
second is the operator's to execute.

**Consequence for run planning.** Ratification removes the *protocol* obstacle (previously G4), so a
run executed under this spec can be decision-grade — provided every other §6 condition holds. It
does not remove any hardware gate: §7's G1/G2/G3/G5/G6 stand unchanged.

**Two inputs this protocol needs are still UNRESOLVED — see §10.** They are not blockers on the
protocol's validity; they are values the *decision rule* consumes. Running without them produces a
sound measurement that cannot be turned into a build/don't-build verdict.

---

## 9. Estimated cost (for scheduling, not a commitment)

4 arms × 3 shed fractions (A2 only) × 2 stress levels × 10 paired blocks is **not** the shape —
A0/A1/A3 do not multiply by *f*. Roughly: A0 and A1 at 2 stress levels, A2 at 3 *f* × 2 stress,
A3 at 3 *f*, each × 10 reps ≈ **130 measurement blocks**, plus lane launch/teardown per residency
change (~1–2 min each, D6's drained relaunch). Block length is set by the frozen corpus. This is a
multi-hour quiet-window campaign, not an afternoon — which is itself an argument for resolving G4
(ratification) *before* spending the window, so the output is decision-grade the first time.

---

## 10. UNRESOLVED inputs — recommended values below are PROPOSALS, not settled

> **Status, stated plainly:** P-SHED-1 is ratified (§8), but these two inputs were **not separately
> answered**. The values below are this session's *recommendations*, recorded so the run can be
> planned against something concrete and so the operator has a specific proposal to accept or
> reject. **They are NOT settled and must not be cited as decided.** A run may proceed without
> them — the measurement is sound either way — but the §5.3 decision rule cannot be evaluated
> until §10.1 is fixed, and §10.2 determines which operating points the run must cover, so
> deferring it past run-planning would mean re-running.

### 10.1 Complexity threshold — PROPOSED, UNRESOLVED

*What net gain justifies an admission class, a flag, a telemetry surface and a failure mode?*

**Proposed: build class 3 only if ALL THREE hold.**

1. **Statistically resolvable** — `net_task_rate` ≥ **+10%** of A0's total. Rationale: standing rate
   noise is CV ≈ 9.1%, so a smaller gain is not reliably observable *in production* even if the
   controlled measurement resolves it. A mechanism whose benefit cannot be seen once deployed
   cannot be operated, tuned, or defended later.
2. **Not an artifact of one operating point** — the gain holds at **both** stress levels (§10.2),
   not only the saturating one. A trade that exists only at a single knife-edge is a trigger
   condition too narrow to justify permanent mechanism.
3. **Time-weighted benefit is material** — see the gap below.

**A gap this session found while proposing the threshold, and did not have data to close.** The
decision is not only *"is the net positive under stress"* but *"how much of production wall time is
actually under qualifying stress"*. Class 3 is dormant outside stress, so expected benefit is
`net_gain × stress_duty_cycle`. A +15% gain during stress that occurs 3% of the time is a ~0.5%
overall improvement bought with a permanent admission class — almost certainly not worth it.
**The stress duty cycle is measured nowhere today.** Proposed handling: derive it from existing
production telemetry (a read-only analysis, no inference, no window) *before* the run is scheduled,
and require **stress duty cycle ≥ 15%** as the third condition. If the duty cycle turns out to be
small, that closes class 3 **without spending the multi-hour window at all** — which would be the
cheapest possible resolution and is the reason to do it first.

### 10.2 Stress levels — PROPOSED, UNRESOLVED

*Is "saturating + 0.5× saturating" the right pair, or does the trigger regime sit elsewhere?*

**Proposed: keep the two-level bracket, add a conditional third level, and define "saturating"
operationally rather than by assertion.**

- **Define saturation, do not assume it.** "Saturating" = the concurrent-request depth at which the
  CPU fleet's `task_rate` stops rising (the knee). This must come from a short **calibration sweep
  that is part of the run** (it is inference, so it inherits every §7 gate) — not from a guessed
  request depth. A stress level asserted rather than located would make every arm's "under stress"
  label unfalsifiable.
- **Why two levels bracket the answer.** At 0.5× saturating there is spare CPU capacity, so
  shedding is *expected* to lose: it takes q3 cores from a system that is not starved. At 1.0× it
  may win. The pair therefore brackets the sign change rather than measuring one point and
  generalising.
- **The crossover is the actually-useful number.** If the sign flips between the two levels, then
  the crossover **is class 3's admission trigger** — the stress level above which shedding pays.
  Proposed: add **0.75× saturating** *only if* the sign flips, so a third level is paid for only
  when it buys the trigger threshold. If the sign does not flip, the bracket already answers the
  question and the third level is waste.

### 10.3 What is now settled vs open

| Item | State |
|---|---|
| P-SHED-1 as a protocol | **RATIFIED** (§8); registration is the operator's human-only execution |
| Measurement design (arms, controls, metric, reps, pre-registration) | Settled by this spec |
| Complexity threshold (§10.1) | **UNRESOLVED** — proposal above; also surfaces the unmeasured stress duty cycle |
| Stress levels (§10.2) | **UNRESOLVED** — proposal above |
| Run execution | **BLOCKED** on §7 G1/G2/G3/G5/G6 (q3, host tier, activation, grant, claims) |
