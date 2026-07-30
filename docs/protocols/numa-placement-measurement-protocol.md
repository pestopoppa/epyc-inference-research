# P-BENCH-PLACEMENT-1 — NUMA placement and concurrency measurement protocol

**Protocol id**: `P-BENCH-PLACEMENT-1`
**Status**: ✅ **RATIFIED 2026-07-30.** Registered in `/workspace/MEASUREMENT.md` §2 and given
normative text in the CPU annex `/workspace/measurement/protocols/bench-cpu.md`, applied by the
operator at epyc-root commit `07b7dcab`. A run conforming to this document **is decision-grade
within its scope** and may gate keep / revert / deploy / promote — subject to this protocol's own
gates, which are not waived by ratification: a run missing measured locality, or failing the
`np=1` anchor gate, remains observation-grade or VOID respectively.

> **Superseded status text, retained so the change is auditable:** this file previously read
> *"STAGED — not yet ratified … a run conforming to this document is observation-grade by
> construction"*. That was true when written and became stale the same day. Corrected 2026-07-30
> after the contradiction was caught during authoring of `MRG-1` — the constitution said ratified
> while this file still said staged, so anyone starting from the protocol file would have wrongly
> concluded no placement number could gate a decision. **`MEASUREMENT.md` is the authority; where
> this file disagrees with it, it is wrong.**

Appendix A below is likewise historical: it was the *proposed* registry entry and has since been
applied. It is retained as the authoring record, not as a pending action.
**Created**: 2026-07-30
**Supersedes**: nothing. **Extends**: `P-BENCH-1` (single-instance decode), `P-BENCH-2`
(multi-instance aggregate), `P-BENCH-3` (batched slot decode) — none of which constrain *memory
placement*, which is why the 2026-07-30 defect set was reachable.
**Owning handoff**: `/workspace/handoffs/active/numa-placement-defect-20260730.md`
(narrative + root cause). **This file is the protocol**: it exists so the same five defects cannot
recur, not to retell them.

---

## 0. Why this protocol exists, and what it binds

On 2026-07-30 a placement audit found that **27 of 31** cells of a completed batched-decode
campaign were confounded, and that two live production roles were serving at roughly one third of
the throughput the same hardware delivers under the canonical recipe. Every one of the five causes
was **invisible to the instruments already in use**: the affinity preflight measured the defect and
reported it as `not required`; the cpuset-shape criterion passed six cells that ran with **0.00**
of their weights local; a warm page cache made a corrected re-test reproduce the *uncorrected*
number.

The design rule of this protocol follows from that: **a placement measurement is not valid because
the launch command looked right. It is valid because the realized placement was measured and
recorded.** Intent is not evidence. Every gate below is a read of realized state.

This protocol binds any run that varies, or depends on, **CPU affinity, NUMA memory policy, mmap
mode, instance count, or slot concurrency** and produces a throughput number. That includes every
placement sweep, every `-np` ladder, every quarter/half/full fleet comparison, and every
"production as-wired vs corrected" claim.

### 0.1 Instrument identity (fixed for this era)

| Field | Value |
|---|---|
| Kernel | `production-consolidated-v8` @ `67a433bf45a8a091d83b4ea0b32ff0735fd51800` |
| `llama-server --version` | `10107` |
| Era stamp | `production-consolidated-v8` (`E5-cpu-kernel` lineage; see `instrument_eras.yaml`) |
| Host topology | **NPS4**, 4 nodes, 96 physical cores + 96 SMT siblings |
| Reference measurements in this document | **2026-07-30** |
| OMP env stack | `canonical_recipe.CANONICAL_OMP_ENV` + `LLVM20_LIBDIR` on `LD_LIBRARY_PATH` — never hand-typed |

**NPS4 node map** (authoritative; identical to `affinity_preflight.NODE_CPUSETS` and to the
`NUMA_Q*` constants in `stack_numa.py`):

```
node0 = 0-23,96-119     node1 = 24-47,120-143
node2 = 48-71,144-167   node3 = 72-95,168-191
```

`0-95` are the 96 physical cores; `96-191` are their SMT siblings (core *i* pairs with thread
*i+96*). Each node owns 24 physical cores plus those 24 siblings.

> **`stack_numa.NUMA_NODE0` / `NUMA_NODE1` are NOT nodes.** The names are NPS2-era artefacts
> predating the 2026-04-24 NPS4 reboot. On this host `NUMA_NODE0 = "0-47,96-143"` spans
> **node0+node1** and `NUMA_NODE1 = "48-95,144-191"` spans **node2+node3**. Read the cpuset
> string against the map above; never trust a constant's name. `NUMA_Q0A/Q0B/Q1A/Q1B` **are**
> exactly the four NPS4 nodes and are correctly named. `NUMA_FULL = "0-95"` is all four nodes.

### 0.2 Units contract — non-negotiable

Every throughput number produced under this protocol is **tok/s** and MUST carry all three
qualifiers. A number missing any of them is not a result, it is a note.

| Qualifier | Allowed values | Meaning |
|---|---|---|
| `aggregation` | `per-stream` \| `aggregate` | `per-stream` = one client stream's decode rate. `aggregate` = summed across **all** concurrent streams of the arm (state the stream count `T`). At `T = 1` the two coincide and MUST still be labelled. |
| `spec_dec` | `on` \| `off` | Speculative decoding / MTP self-draft state. A spec-dec-on number and a spec-dec-off number are **different metrics** and must never be ratioed. |
| `metric_source` | `llama_bench_tg` \| `server_predicted_ms` | `llama-bench` `tg<N>` row, or llama.cpp's own per-request `timings.predicted_n` / `timings.predicted_ms`. **Wall-clock rates are forbidden as decode rates** (see F4). |

Direction: **higher-better** for every tok/s in this protocol. `local_fraction` is
higher-better. `achieved/nominal concurrency` is higher-better.

Additionally required on any `ingest_long_context`-class (long-context) decode number: the
**prompt token count**. The same role measures 9–12 tok/s at 26k-token prompts and 15–25 tok/s at
short context; those figures were never in conflict — the context length was simply never carried
alongside the number.

---

## 1. The five failure modes this protocol exists to prevent

Each mode below is stated as: **what it is** → **how it silently corrupts the result** →
**the mandatory check that catches it**. A run that does not execute all five checks is
non-conforming, regardless of how careful its launch commands were.

| id | Failure mode | Silent corruption | Mandatory check |
|---|---|---|---|
| **F1** | Straddling cpuset with no NUMA policy | Weights first-touch onto one node; half the thread team reads cross-node forever | §1.1 — cpuset→node classification + declared policy |
| **F2** | Warm-cache no-op | `--interleave` binds at first touch only; a warm re-test measures the OLD placement | §1.2 — `drop_caches` per arm + post-load locality read |
| **F3** | Shared mmap defeats per-instance `--membind` | Pages placed once by whoever faults first; later instances inherit it regardless of their own policy | §1.3 — `--no-mmap` for declared-per-instance placement, or measured proof of the shared placement |
| **F4** | Wall-clock reported as a decode rate | `tokens / wall_seconds` includes load, prefill, queueing and idle | §1.4 — decode rate from `predicted_n` / `predicted_ms` only |
| **F5** | Rungs that never reach nominal concurrency | A fixed prompt batch drains before saturating; the rung measures a draining queue | §1.5 — achieved concurrency measured, reported, and floored |

### 1.1 F1 — Straddling cpuset with no NUMA memory policy

**What it is.** A cpuset that spans two or more NPS4 nodes, launched with no `numactl` memory
policy. `0-47,96-143` spans **node0 and node1**. `48-95,144-191` spans **node2 and node3**.
`0-95` spans **all four**. With no policy in force, Linux first-touch places every weight page on
whichever node the loading thread happened to run on, so a 96-thread team reads roughly half its
weights across the interconnect on every token.

**How it silently corrupts.** Nothing fails. The server starts, affinity is exactly as configured,
the topology hash matches, and the throughput number is simply low — by a factor, not a percent.
Because the cpuset *is* contiguous and *is* half the machine, it reads as a deliberate placement
to anyone who does not expand it against the NPS4 map.

**Measured cost** — `Qwen3.6-35B-A3B-Q8_0`, single `llama-server` instance, `-np 1`,
`--no-mmap --mlock`, `-t 96`, **spec-dec OFF**, per-stream (= aggregate at `T = 1`),
`metric_source: server_predicted_ms`, 2026-07-30:

| arm | cpuset | policy | cache | tok/s (per-stream, spec-dec off) |
|---|---|---|---|---:|
| production as-wired | `0-47,96-143` (node0+node1) | none | warm | **9.99** |
| corrected | `0-95` (all four nodes) | `--interleave=all` | cold | **23.71** |

**Ratio 2.4×.** This is a *placement* effect, not a kernel, quant, MTP or meter effect: the
single-NPS4-node instrument-validation arm reproduced its registry-documented value to within 1%,
and v7 and v8 measure identically on this model.

**Mandatory check (F1-CHK).** For every instance of every arm, before the run is accepted:

1. Expand the instance's `taskset -c` cpuset string to a set of logical CPU ids.
2. Map each id to its node using §0.1 (`affinity_preflight._expected_nodes` implements exactly
   this).
3. Record `n_nodes` and the node set.
4. **If `n_nodes > 1`, the launch command MUST carry an explicit `numactl` memory policy.** A
   multi-node cpuset with `numactl_policy: none` is a **hard reject** — the arm is void, not
   "worth noting".
5. If `n_nodes == 1`, the arm MUST declare either `--membind=<that node>` or an explicit
   `--interleave` over a stated node list. "The cpuset is one node so first-touch will be local"
   is **not** a policy — F3 defeats it whenever the GGUF is shared.

Record the policy verbatim from the realized command line, not from config. `stack_numa._numa_prefix`
emits `numactl --<policy> --` ahead of `taskset` when `numactl_policy` /
`numactl_policy_instances` is set; the E5 harness's `instance_launch_prefix` emits
`numactl --interleave=all` only for `numactl_policy == "interleave=all"`. Either way, the
attested field is the argv that actually ran.

### 1.2 F2 — Warm-cache no-op

**What it is.** `numactl --interleave` / `--membind` set a *policy on the mapping*; the policy is
applied **at first touch**. If the model's pages are already resident in the page cache from a
previous arm, no fault occurs, no policy is applied, and the process reads whatever placement the
earlier arm produced.

**How it silently corrupts.** This is the mode that makes the defect survive investigation. Add
`--interleave=all` to a bad arm, re-run it warm, and it returns the *bad* number — which reads as
"we tested the fix and it did not help", closing the investigation on a false negative. In the
2026-07-30 matrix the warm straddle+interleave arm landed on the unpolicied figure rather than the
cold-interleave figure, exactly as this predicts.

**Mandatory check (F2-CHK).**

1. **Before every placement arm**, synchronously: `sync`, then privileged
   `echo 3 > /proc/sys/vm/drop_caches`. A failed cache action invalidates the arm (same
   fail-closed rule as `P-BENCH-4`). This is per **arm**, not per campaign, and not per model.
2. Re-warm, if the arm's design calls for a warm-up, only **through the arm's own placement
   policy** — never a bare `cat`/`dd`/plain re-read, which pins the whole file to one node
   (`feedback_drop_caches_numa_eviction`).
3. Record `cache_state ∈ {cold, warm}` as a required per-arm field.
4. **An arm declared `cache_state: warm` (e.g. "production exactly as found") MUST be accompanied
   by the same arm at `cache_state: cold`, and both MUST be reported.** A warm arm may never be
   the sole basis of a ratio, in either direction.
5. **Record a post-load locality reading for every arm** (see F3-CHK / §4). Dropping caches is the
   intervention; the locality read is the *proof the intervention took effect*. Without it, F2 is
   undetected by construction.

### 1.3 F3 — Shared mmap defeats per-instance `--membind`

**What it is.** llama.cpp `mmap`s the GGUF by default. Those pages live in the **shared page
cache**. They are placed **once**, by whichever process faults them in first, and every later
process that maps the same file maps the **same physical pages** — inheriting that placement
**regardless of its own `--membind`**. With `kernel.numa_balancing = 0` (standing policy, enforced
by `canonical_recipe.validate_host_environment`) nothing migrates them afterwards.

**How it silently corrupts.** A four-quarter fleet can be perfectly node-aligned by cpuset, pass
live-affinity verification, and still read **0%** of its weights locally. In the E5 W1 run a single
straddling instance loaded first and first-touched 100% of pages onto node0; all eleven later cells
in that run inherited it, including the "node-aligned" quarter cells, which therefore ran at
`local_fraction = 0.00` — strictly *worse* than the straddling cells they were meant to be
compared against. The cpuset-shape criterion alone would have declared them clean.

**Measured** — 4 quarter instances, each `--membind`-ed to its own node, `Qwen3.6-35B-A3B-MTP-Q8_0`,
`np=1` per instance, **MTP spec-dec ON**, 2026-07-30:

| arm | q0 (own N0) | q1 (own N1) | q2 (own N2) | q3 (own N3) | host RAM |
|---|---:|---:|---:|---:|---:|
| mmap (production default) | 25.6% local | 25.6% | 24.2% | 26.9% | 30 GB |
| `--no-mmap` | **100%** | **100%** | **100%** | **100%** | 171 GB |

| arm | fleet decode, **aggregate across 4 streams**, spec-dec on |
|---|---:|
| mmap | **40.91 tok/s** |
| `--no-mmap` | **52.13 tok/s** (**+27%**) |

RAM cost of `--no-mmap` for this fleet: **+141 GB** (30 → 171 GB) — each instance takes a private
copy instead of sharing one.

> **Corollary — under shared mmap, fleet throughput is NONDETERMINISTIC ACROSS REBOOTS.** Placement
> is decided by whichever instance faults the pages first, so fleet throughput depends on instance
> **START ORDER**. Two observations of the same model in the same week landed on opposite extremes:
> a sequential run where one straddling instance placed 100% of pages on node0 and every later cell
> inherited it, versus a simultaneous four-instance load where each grabbed ~25% and all four ended
> symmetric-but-not-local. **Neither is a configuration anybody chose.** A shared-mmap fleet number
> without a recorded start order and a measured per-instance `pages_by_node` is an accident, not a
> measurement, and MUST NOT be compared across runs.

**Mandatory check (F3-CHK).**

1. **Every arm records `mmap` ∈ {`mmap`, `--no-mmap`} per instance**, read from the realized
   `/proc/<pid>/cmdline`, not from config.
2. **Any arm whose intent is per-instance placement (`--membind`, node-local quarters) MUST run
   `--no-mmap`.** A `--membind` under shared mmap is a declaration with no effect and is a hard
   reject as a *placement* arm. (It remains a legitimate arm if and only if its declared intent is
   "measure the production sharing model" — arm `A3` in §3 — in which case the inherited placement
   must be measured and reported, and the instance **start order** recorded.)
3. **Measure realized placement per instance** from `/proc/<pid>/numa_maps` and record
   `pages_by_node` + `local_fraction`. Use `affinity_preflight.py`'s reader
   (`_summarize_numa_maps`), which sums per-node page counts over mapping lines carrying
   `file=<...>.gguf` for mmap roles and anonymous placement for `--no-mmap` roles.
4. **Arm the gate.** `affinity_preflight.py` computes
   `required = no_mmap and len(expected_nodes) == 1`, so under mmap it *observes and reports but
   never fails*. That is precisely how six cells at `local_fraction = 0.0` passed preflight. Under
   this protocol the gate is armed for **every** arm: run with `--require-memory-locality`, and
   treat a below-threshold `local_fraction` on a single-node instance as a **failure**, mmap or not.
   Threshold: `LOCALITY_THRESHOLD` — **TBD** (the tool default is `0.85`; the 2026-07-30 salvage
   audit used `≥ 0.99`; the binding value must be pre-registered before the run).
5. For an interleaved multi-node arm the acceptance criterion is **evenness**, not locality: the
   per-node page shares must each be within `INTERLEAVE_TOLERANCE` (**TBD**) of `1 / n_nodes`. The
   2026-07-30 interleaved control measured 25.00% per node across four nodes; a first-touch arm
   measured 100% on a single node. These two signatures are unambiguous and are the primary proof.
6. **`live_memory_placement_verified: true` means the placement was OBSERVED, not that it was
   CORRECT.** Never read that field as a pass. Read `local_fraction`.

### 1.4 F4 — Wall-clock reported as a decode rate

**What it is.** Computing `total_predicted_tokens / wall_seconds` and calling it a decode rate.
`wall_seconds` spans the entire cell — model load, warm-up, prefill, queueing and idle gaps — so
the quantity is a *wall-clock* rate and is not comparable to llama.cpp's `predicted_per_second`
or to a `llama-bench` `tg` row.

**How it silently corrupts.** It reads systematically **low**, and it reads low by a
*configuration-dependent* amount (largest where load and prefill are the biggest share of the
window). So it does not merely offset a comparison, it tilts it.

**Measured impact of this defect alone.** Comparing like with like — a **system-wide** decode rate
(`sum(predicted_n)` ÷ the union of all `[first_token, end]` intervals, i.e. the same numerator and
the same *kind* of denominator, minus load/prefill/idle) against the old wall-clock field, over 31
cells: **median 1.05× (≈ 5% understatement), worst case 1.22× (22%, at `T = 1`)**. Small — and it
must never again be conflated with a decode rate.

> **Do not read `per-slot ÷ wall-clock` as the bug magnitude.** The per-slot denominator
> `sum(predicted_ms)` sums **overlapping** slot intervals, so at `T = 32` it counts up to 32
> slot-seconds per elapsed second; that literal ratio (median `0.09×`) measures the denominator
> mismatch, not the defect. Replacing one metric error with another is the failure this note
> exists to prevent.

**Mandatory check (F4-CHK).**

1. The decode rate is computed **only** from llama.cpp's own per-request `timings` block —
   `predicted_n` and `predicted_ms` — or from a `llama-bench` `tg<N>` row. No other denominator is
   a decode rate.
2. Report **both** derived rates, each explicitly labelled:
   - `per_stream_decode_tps` = `sum(predicted_n) / sum(predicted_ms/1000)` — the token-weighted
     mean **per-slot** rate.
   - `system_decode_tps` = `sum(predicted_n) / |union of all [first_token_s, end_s] intervals|` —
     the **aggregate-across-streams** rate. This is the only quantity comparable to a wall-clock
     field or to `llama-bench` tg.
3. A wall-clock rate may still be recorded, but **only** under a name that says so
   (`aggregate_wallclock_tps`), and it may never appear in a decision row as a decode rate.
4. Exclusion rule, applied identically to numerator and denominator: skip a successful request
   whose `timings` block is missing, whose `predicted_n`/`predicted_ms` are absent, non-numeric,
   boolean, or `<= 0`. **Report the skip audit** (counts by reason) with every arm; a silent
   exclusion is a thinned aggregate.

### 1.5 F5 — Rungs that never reach nominal concurrency

**What it is.** A `-np N` ladder driven by a **fixed** prompt batch. The high-`N` rungs exhaust
the batch before all slots are occupied, so the machine spends much of the window partly idle
while the rung is labelled with its nominal slot count.

**How it silently corrupts.** The rung is reported as `T = 32` but never ran 32 concurrent
streams, so every batching-efficiency conclusion drawn from it describes a **draining queue**, not
a saturated machine. This is independent of F1–F4: correcting placement does not fix it.

**Measured**, 43 pinned prompts per cell, 2026-07-30 re-derivation:

| nominal `T` | cells | mean achieved concurrency | % of nominal |
|---:|---:|---:|---:|
| 1 | 3 | 1.0 | **100%** |
| 4 | 5 | 3.4 | 86% |
| 8 | 7 | 6.4 | 80% |
| 16 | 8 | 12.4 | 77% |
| 32 | 8 | 14.9 | **47%** |

**Mandatory check (F5-CHK).**

1. Compute achieved concurrency per rung as `mean_concurrency = sum(predicted_ms) / union_window`
   — the average number of slots actually decoding at once — and report it alongside nominal `T`
   as `achieved / nominal` **and** as a percentage.
2. **Pre-register a floor** `ACHIEVED_CONCURRENCY_FLOOR` (**TBD**) before the run starts. Any rung
   whose achieved/nominal falls below the floor is **rejected**: it is not reported as a `T = N`
   result and it may not enter any batching-efficiency conclusion. It may be reported as a
   separately-labelled under-saturated observation.
3. Size the driver to the concurrency: enough prompts, or a **closed-loop arrival process** that
   holds occupancy at `T` for the whole measurement window. A fixed batch sized for `T = 1` cannot
   measure `T = 32`.
4. The floor and the driver design are **part of the instrument**. Changing either is a new
   instrument version, not a tuning.

---

## 2. Mandatory anchor gate

**Rule.** Every placement or concurrency sweep MUST **first** measure `np = 1` on the arm under
test and compare it against a **recorded production anchor** for that model. **If the `np = 1`
measurement falls outside the anchor band, the entire run is VOID and MUST NOT be reported** — not
as a claim, not as an observation, not as a "directionally interesting" table. Fix the instrument,
then re-run.

The anchor gate is a **cross-instrument** check. It is the only cheap defence against a defect that
is invisible inside the harness, because it compares the harness against a completely different
production path measuring the same physical thing.

**Ordering.** The `np = 1` anchor cell runs **first** in each arm, on a freshly loaded server,
after that arm's `drop_caches`. Running it last measures a machine that has been warmed and
possibly re-placed by the rest of the ladder.

### 2.1 Worked example — the frontdoor anchor

| quantity | value |
|---|---|
| Anchor instrument | AutoPilot live production traffic, `median_request_tps` |
| Anchor value | **median 35.7 tok/s** (per-stream, spec-dec as production = MTP on), `n = 154` |
| Anchor band | **35–40 tok/s** |
| Corrected full-machine placement, `np = 1` (`0-95` + `--interleave=all`) | **38.72 tok/s** per-stream (= aggregate at `T = 1`), MTP spec-dec **on** — **inside the band** |
| Anchor reproduction, six sequential requests, freshly loaded server, winning placement | 39.35 / 36.68 / 36.62 / 36.57 / 36.40 / 36.09 tok/s per-stream |
| Original E5 grid (no anchor gate) | ≈ **7.8–16 tok/s** — far outside the band, on the same model, on the same host |

The original grid had **no such gate**. Had one existed, the campaign would have halted at its
first cell instead of producing 31 cells of which 27 were confounded. The disagreement between the
production anchor and the harness *was the signal*; nothing in the harness could see it.

Note also what the anchor tells you about scope: the AutoPilot anchor is **not** affected by the
production wiring defect, because that traffic lands on a placement which reproduces the corrected
full-machine number. An anchor drawn from the defective path would have confirmed the defect.
**Choose the anchor from a path that is independent of the thing under test.**

### 2.2 Anchor requirements

An anchor is admissible only if it states, and the run matches on, all of:

- **model + quant identity** (path + SHA-256);
- **spec-dec state** (`on`/`off`) — an anchor measured with MTP on cannot gate a spec-dec-off arm;
- **aggregation** (`per-stream` at `np = 1`);
- **`n`** (sample count behind the anchor) and the **band** (explicit low/high, not a point);
- **era** — same kernel era as the run (`production-consolidated-v8` here);
- **prompt-length regime**, for long-context roles.

Per-model anchors and bands other than frontdoor's: **TBD** (see §7).

If no admissible anchor exists for a model, the run may proceed **only** if it *establishes* one:
run the `np = 1` cell to `P-BENCH-1` rep discipline, record it as the candidate anchor with its
band, and label the whole campaign observation-grade until the anchor is independently reproduced.

---

## 3. The placement arm set

Any placement/concurrency campaign MUST cover **all five** arms below, per model. Fewer arms is
not a smaller campaign, it is an uninterpretable one: `A0→A1→A2` is a two-factor decomposition
(policy × cpuset) and dropping the bridge cell `A1` makes the two factors inseparable.

| arm | shape | cpuset | memory policy | mmap | what it isolates |
|---|---|---|---|---|---|
| **A0** | production **as-wired** | exactly what `stack_numa.NUMA_CONFIG` emits for the role today | exactly what `_numa_prefix` emits today (often **none**) | production default | The live operating point. The reference the corrected arms must beat. Isolates nothing on its own — it is the thing being explained. Run at both `cache_state: warm` (production as found) and `cache_state: cold` per F2-CHK. |
| **A1** | **same cpuset**, correct policy | identical to A0 | explicit `--interleave` over exactly A0's node set (e.g. `--interleave=0,1` for `0-47,96-143`) | as A0 | **The memory-policy effect at fixed cpuset.** This is the bridge cell: `A1 − A0` is the cost of F1's missing policy; `A2 − A1` is the cost of the cpuset shape. Without A1 the two are confounded and the campaign can only say "the corrected recipe is faster", not why. |
| **A2** | **full machine** | `canonical_recipe.CANONICAL_PREFIX` cpuset (`0-95`, all four nodes) | `--interleave=all` (already in `CANONICAL_PREFIX`) | as A0 | **The cpuset-shape effect at fixed, correct policy** — and the canonical-recipe ceiling for a single instance. This shape *was absent from the original E5 grid* and won for every model tested on 2026-07-30. |
| **A3** | **N-instance fleet, shared mmap** | N node-aligned cpusets (`NUMA_Q0A/Q0B/Q1A/Q1B` for N=4; the two straddling halves for N=2 — declare which) | per-instance `--membind=<own node>` | **mmap** (production default) | **The production sharing model, and the F3 confound itself.** This arm measures what the fleet actually does today. Its per-instance `local_fraction` will *not* match its declared `--membind`; that mismatch is the result. **Instance start order MUST be recorded** — under shared mmap it is a determinant of the number (see F3 corollary). |
| **A4** | **N-instance fleet, `--no-mmap`** | identical to A3 | per-instance `--membind=<own node>` | **`--no-mmap`** | **The declared-placement fleet** — the only arm in which each instance's placement is what it declares (`local_fraction = 1.00`). `A4 − A3` is the cost of page sharing. Record the **host RAM cost**: `--no-mmap` gives each instance a private copy (+141 GB for the 2026-07-30 35B quad, 30 → 171 GB). |

**Arm-set rules.**

1. `N` for A3/A4 is fixed per campaign and stated in the run header (4 quarters and/or 2 halves).
   A3 and A4 MUST use the **identical** cpusets, thread counts and `np`, differing **only** in
   mmap mode. Any other difference makes `A4 − A3` uninterpretable.
2. Arms are **interleaved / order-randomized** across replicate blocks, never blocked as
   `A0×n → A2×n` (thermal and page-cache drift would alias onto the arm effect — the `P-SHED-1`
   rule, which applies verbatim here).
3. Every arm gets its own `drop_caches` (F2-CHK) — including between A3 and A4, whose whole
   distinction is page-cache behaviour.
4. **Thread count is an arm attribute, not a constant.** SMT oversubscription is a real, measured
   effect on this host (a half cpuset contains 48 physical cores plus 48 siblings; `-t 96` on it
   is oversubscription). If the campaign varies `-t`, it varies it **within** an arm as a declared
   sub-arm, never silently across arms.
5. The whole set runs on **one** model at a time, under a held `region-lock` for exactly the
   physical footprint used (`bench_canonical.sh` acquires the same per-region flocks the
   orchestrator dispatch path uses; fail closed if it cannot).

---

## 4. Required per-arm evidence

Every arm — and for multi-instance arms, every **instance** — records all of the following. These
are not reporting conveniences; **an arm missing any field is non-conforming and cannot be
reported under this protocol id.**

### 4.1 Placement fields (per instance)

| field | source | notes |
|---|---|---|
| `cpuset` | realized `taskset -c` argv | verbatim string |
| `cpuset_nodes` | expansion vs §0.1 map | derived; record `n_nodes` |
| `threads` | realized `-t` | state physical vs SMT relationship to `cpuset` |
| `numactl_policy` | realized argv | `none` is a value and, on a multi-node cpuset, a **reject** |
| `mmap_mode` | `/proc/<pid>/cmdline` | `mmap` \| `--no-mmap` |
| `drop_caches` | run log | `yes`/`no` + `cache_state ∈ {cold, warm}` |
| `pages_by_node` | `/proc/<pid>/numa_maps` via `affinity_preflight._summarize_numa_maps` | per-node page counts, **measured live on the running server** |
| `local_fraction` | derived | pages on the instance's expected node(s) ÷ total |
| `live_affinity_verified` | `affinity_preflight.py` thread-union check | union of **all** threads' `Cpus_allowed_list`, exact-match to `cpuset` |
| `instance_start_order` | launcher | **required for A3**; recommended everywhere |

### 4.2 Result fields (per arm)

| field | source | notes |
|---|---|---|
| `nominal_T` | `np × n_instances` | |
| `achieved_concurrency` | `sum(predicted_ms) / union_window` | report as absolute **and** `% of nominal` (F5-CHK) |
| `per_stream_decode_tps` | `predicted_n` / `predicted_ms` | label `per-stream` |
| `system_decode_tps` | tokens ÷ union window | label `aggregate`, state `T` |
| `spec_dec` | launch config | `on`/`off`; a mixed-`spec_dec` comparison is void |
| `reps` | run design | `P-BENCH-1` rule: **≥ 5** for effects ≥ 5%, **≥ 10** for effects ≤ 2%; report **median + MAD** |
| `skip_audit` | replay | counts by reason (no `timings` block / missing field / bad type / `<= 0`) |
| `prompt_tokens` | request set | required for long-context roles; recommended always |

### 4.3 Identity and host fields (per run)

Kernel branch + commit + `llama-server --version`; binary and shared-library paths + SHA-256;
`LD_LIBRARY_PATH`; complete effective environment (must satisfy
`canonical_recipe.assert_canonical_env`); model path + size + SHA-256, quant, context, KV quant;
complete argv per instance; host-health attestation per
`canonical_recipe.validate_host_environment` (THP enabled + defrag `always`, governor
`performance`, `kernel.numa_balancing = 0`, `perf_event_paranoid` if perf-wrapped); `region-lock`
holder and region set; process witness (no foreign llama-family process overlapping the declared
cpusets); date and era stamp.

### 4.4 The grading rule

> **A run missing measured locality (`pages_by_node` / `local_fraction`) is
> OBSERVATION-GRADE AT BEST and can NEVER gate a decision.**

This is the load-bearing sentence of the protocol. It is stated as an absolute because every
weaker formulation was already available on 2026-07-30 and did not hold: the launch commands were
inspected, the cpusets were node-aligned, live affinity was verified, the preflight artifact said
`live_memory_placement_verified: true` — and six cells still ran at `local_fraction = 0.00`. The
only field that would have caught it is the one this rule makes mandatory.

Corollary rules, same force:

- **A cpuset-shape criterion alone is insufficient.** On the E5 corpus it scored 18 clean / 13
  confounded; the evidence-based criterion (cpuset **and** measured locality) scored **4 clean /
  27 confounded**. Shape is a screen, not a gate.
- **A per-instance timing gradient is corroboration, never primary proof.** At `np = 1` decode
  rates did fall monotonically with NUMA distance from the placed node (1.58× and 3.15×
  local:far on two models; flat on the interleaved control). But the gradient **washes out** at
  higher `np` as all instances contend for one memory controller — in one cell the remote instance
  was even faster. Cite the `numa_maps` measurement as the proof; cite the gradient as support.

---

## 5. Claim grammar

This repository's rule, restated verbatim from `MEASUREMENT.md` §"The one rule" and
`MEASUREMENT_POLICY.md` → *The claim rule*:

> **A decision-gating number = `(metric, protocol-id, n/reps, date, attestation ref)`.**
> A number without a protocol citation is an **observation**: usable for hypothesis formation,
> never for keep / revert / deploy / promote / buy / close.

### 5.1 The id and why it fits

**`P-BENCH-PLACEMENT-1`.**

The registry in `MEASUREMENT.md` §1 uses `P-<FAMILY>[-<AXIS>]-<version>`. The `P-BENCH-*` family
covers throughput instruments: `P-BENCH-1` (single-instance decode), `P-BENCH-2` (multi-instance
aggregate), `P-BENCH-3` (batched slot decode), `P-BENCH-4` (server-native spec-dec). When a new
**axis** of the same family was ratified on 2026-07-24 it took the axis-token form
**`P-BENCH-PREFILL-1`** — same family, named axis, version `1`. This protocol governs a new axis of
the same family (memory **placement** and its interaction with concurrency), so it takes the same
form: family `BENCH`, axis `PLACEMENT`, version `1`. It is not `P-BENCH-5`, because the numeric
slots denote serving *shapes* within the family while the axis token denotes the *measured
quantity*; placement is a quantity, not a shape.

### 5.2 Grammar for a claim under this protocol

A row MUST carry the units contract (§0.2) inline:

```
<value> tok/s <per-stream|aggregate(T=<n>)>, spec-dec <on|off>, arm <A0..A4>,
local_fraction <value> [P-BENCH-PLACEMENT-1, n=<reps>, YYYY-MM-DD, attest <ref>]
```

Examples:

- ✅ `A2 full-machine decode 23.71 tok/s per-stream, spec-dec off, local_fraction 1.00 [P-BENCH-PLACEMENT-1, n=TBD, 2026-07-30, attest TBD]`
- ✅ `A4 fleet decode 52.13 tok/s aggregate(T=4), spec-dec on, local_fraction 1.00 [P-BENCH-PLACEMENT-1, n=TBD, 2026-07-30, attest TBD]`
- ❌ `full machine is 2.4× faster` — no aggregation qualifier, no spec-dec state, no locality, no protocol, no reps.
- ❌ `quarter fleet 40.91 tok/s` — aggregation unstated, and under shared mmap this number is start-order dependent (F3 corollary), so it is not reproducible without `pages_by_node` + start order.

### 5.3 Standing status of the 2026-07-30 numbers

Every number in this document is **observation-grade**: `n = 1–2` reps, no attestation reference,
produced before this protocol was ratified. They are the *input* to a decision-grade re-run and
MUST NOT gate a wiring change, a lineup change, or a promotion. Per `MEASUREMENT.md` §5 they are
**demoted-to-prior** by default and may be **retro-certified** only if a field-by-field audit shows
every §4 field is present — and no partial upgrades (`P-GPU-1` rule, applied here unchanged).

This protocol is **prospective**: no pre-amendment placement artifact may be retro-certified under
it merely because its numbers look right.

---

## 6. Reviewer checklist

Run this down before accepting any placement number. Each line is a **read of recorded evidence**;
none of them is satisfied by inspecting a launch command.

- [ ] **Era + kernel stamped?** `production-consolidated-v8` @ `67a433bf4`, binary `10107` (or the
      then-current production-named kernel). Experimental-kernel numbers are observations, full stop.
- [ ] **Units complete on every row?** tok/s, `per-stream` vs `aggregate` (+ `T`), spec-dec
      `on`/`off`. Long-context rows also carry prompt token count.
- [ ] **Every cpuset expanded against the NPS4 map** — not read off a constant's name. Any
      `n_nodes > 1` cpuset with `numactl_policy: none` → **REJECT**.
- [ ] **`drop_caches` recorded per arm**, with `cache_state`. Any `warm` arm has a `cold`
      companion reported next to it.
- [ ] **`pages_by_node` + `local_fraction` present for EVERY instance**, measured from
      `/proc/<pid>/numa_maps` on the live server. Missing → observation-grade, cannot gate.
- [ ] **`local_fraction` actually checked against the threshold**, not just printed.
      `live_memory_placement_verified: true` is NOT a pass — it means "observed".
- [ ] **mmap mode recorded per instance.** Any `--membind` arm under shared mmap → **REJECT** as a
      placement arm. Any shared-mmap fleet arm without recorded **instance start order** → reject
      as non-reproducible.
- [ ] **Decode rate derived from `predicted_n`/`predicted_ms`** (or a `llama-bench` `tg` row).
      Any `tokens / wall_seconds` presented as a decode rate → **REJECT**. Skip audit reported.
- [ ] **Achieved concurrency reported per rung** as `achieved / nominal` and `%`. Any rung below
      the pre-registered floor is excluded from batching conclusions.
- [ ] **`np = 1` anchor gate ran FIRST and landed inside the recorded band.** Outside band → the
      whole run is VOID; do not report it.
- [ ] **All five arms A0–A4 present** for the model, with A1 (the bridge cell) actually run —
      otherwise policy and cpuset effects are confounded.
- [ ] **A3 and A4 differ ONLY in mmap mode.** Any other delta → `A4 − A3` is uninterpretable.
- [ ] **Arms interleaved / order-randomized**, not blocked.
- [ ] **Reps meet the `P-BENCH-1` rule** (≥5 for ≥5% effects, ≥10 for ≤2%), median + MAD reported.
- [ ] **`region-lock` held for the exact physical footprint**, for the whole run; process witness
      shows no foreign llama-family overlap.
- [ ] **Host-health attestation present** (THP, governor, `numa_balancing = 0`, uptime tier).
- [ ] **Claim rows carry `(metric, protocol-id, n/reps, date, attestation ref)`.** No attestation
      → observation.
- [ ] **Timing gradient, if cited, is labelled corroboration** — the `numa_maps` read is the proof.

---

## 7. Values still required

The following are referenced above as binding parameters but were not supplied and are **not
invented here**. Each must be fixed — and pre-registered — before a conforming run starts.

| # | Value | Where used | Notes / candidate sources |
|---|---|---|---|
| 1 | `LOCALITY_THRESHOLD` — minimum `local_fraction` for a single-node instance | F3-CHK step 4; §6 | `affinity_preflight.py --memory-locality-threshold` defaults to `0.85`; the 2026-07-30 salvage audit used `≥ 0.99`. Needs one binding value. |
| 2 | `INTERLEAVE_TOLERANCE` — allowed deviation from `1/n_nodes` per node on an interleaved arm | F3-CHK step 5 | 2026-07-30 interleaved control measured 25.00% per node across 4 nodes. |
| 3 | `ACHIEVED_CONCURRENCY_FLOOR` — minimum achieved/nominal for a rung to be reportable | F5-CHK step 2; §6 | The operator's requirement names a threshold; the value is unset. |
| 4 | Prompt-set size / closed-loop arrival design that holds occupancy at `T` | F5-CHK step 3 | 43 pinned prompts demonstrably cannot hold 32 slots. |
| 5 | Anchor value + band + `n` for **`ingest_long_context`** (Qwen3-Next-80B-A3B), per prompt-length regime | §2.2 | Only the frontdoor anchor (35.7, n=154, band 35–40) exists. |
| 6 | Anchor value + band + `n` for **`worker_general`** (gemma4-26B-A4B Q4_K_M MTP) | §2.2 | — |
| 7 | Anchor value + band + `n` for **`architect_general`** and any other role entering a placement campaign | §2.2 | — |
| 8 | Reps `n` behind each 2026-07-30 reference number in §1 and §2 | §5.2 examples | Currently `n = 1–2`; the exact per-figure rep count is not recorded uniformly. |
| 9 | Attestation reference for each 2026-07-30 figure | §5.2, §5.3 | The 2026-07-30 artifacts are **scratch** (`/mnt/raid0/llm/tmp/`), not promoted into `data/`. No figure currently carries an attestation ref. |
| 10 | Independent replicate of the full-machine `A2` figure | §3, §5.3 | The full-machine number rests on one invocation; the script arm labelled as its repeat actually repeats the straddle+interleave arm. |
| 11 | Ratification date + receipt for the `MEASUREMENT.md` amendment (Appendix A) | Status block | Human-only write; blocks decision-grade use of this id. |

---

## Appendix A — proposed `MEASUREMENT.md` registry entry (STAGED, operator-apply)

> **STATUS: STAGED for human review.** Written here by the authoring session and **NOT applied** —
> the measurement trust boundary (`MEASUREMENT.md` §4) is human-amendment-only. The operator
> appends this to `MEASUREMENT.md` by hand after auditing this document. Present it batched with
> other queued boundary items, and only while compute is saturated with other work
> (`MEASUREMENT_POLICY.md` → *Consolidated apply-time ratification*).

```
## P-BENCH-PLACEMENT-1 — NUMA placement and concurrency (STAGED)

Scope: any decision-gating throughput number that varies or depends on CPU affinity, NUMA
memory policy, mmap mode, instance count, or slot concurrency. Direction: higher-better
(tok/s). Composite: P-BENCH-1 governs the single-instance decode arm, P-BENCH-2 the
multi-instance aggregate, P-BENCH-3 any batched-slot rung; this protocol governs PLACEMENT
and its interaction with concurrency, which none of them constrain.

Full contract: epyc-inference-research docs/protocols/numa-placement-measurement-protocol.md.

Mandatory: (1) cpuset expanded against the live NPS4 node map, multi-node cpuset with no
numactl policy = reject; (2) drop_caches per arm with cache_state recorded, warm arms paired
with cold; (3) measured per-instance pages_by_node + local_fraction from /proc/<pid>/numa_maps,
gate ARMED regardless of mmap mode, --membind under shared mmap rejected as a placement arm,
shared-mmap fleet arms record instance start order; (4) decode rate from predicted_n /
predicted_ms only — wall-clock rates are never decode rates — reporting per-stream and
system-wide separately with a skip audit; (5) achieved concurrency measured per rung and
floored. Arms A0-A4 (production as-wired / same cpuset + correct interleave / full machine +
interleave=all / N-instance fleet shared mmap / N-instance fleet --no-mmap), interleaved, all
five required. Anchor gate: np=1 measured FIRST against a recorded production anchor for that
model; outside band = run VOID. Reps per P-BENCH-1; median + MAD.

A run missing measured locality is observation-grade at best and can never gate a decision.

Prospective: no pre-amendment placement artifact may be retro-certified under this protocol.
Claim grammar: `<value> tok/s <per-stream|aggregate(T=n)>, spec-dec <on|off>, arm <A0..A4>,
local_fraction <value> [P-BENCH-PLACEMENT-1, n=<reps>, YYYY-MM-DD, attest <ref>]`.
```

---

## Appendix B — source index

| Path | Role |
|---|---|
| `/workspace/MEASUREMENT.md` | Instrument constitution — protocol registry, claim grammar, retroactivity |
| `/workspace/agents/shared/MEASUREMENT_POLICY.md` | Agent digest — claim rule, era handling, apply-time ratification |
| `scripts/lib/canonical_recipe.py` | Codified recipe: `CANONICAL_PREFIX`, `CANONICAL_BENCH_FLAGS_LLAMA_BENCH`, `CANONICAL_OMP_ENV`, `LLVM20_LIBDIR`, `validate_canonical_env`, `validate_host_environment` |
| `scripts/benchmark/bench_canonical.sh` | The only sanctioned `llama-bench` entry point; acquires `region-lock` for the pinned cpu list, fail-closed |
| `scripts/benchmark/server_numa_np_sweep.py` | Placement/`np` sweep harness; `instance_launch_prefix` builds `taskset` + optional `numactl` |
| `epyc-orchestrator/scripts/server/stack_numa.py` | `NUMA_Q0A/Q0B/Q1A/Q1B` (the four NPS4 nodes), `NUMA_NODE0`/`NUMA_NODE1` (NPS2-era names, **straddling**), `NUMA_FULL`, `NUMA_CONFIG`, `_numa_prefix` |
| `epyc-orchestrator/scripts/server/affinity_preflight.py` | `NODE_CPUSETS`, `_thread_union` (live affinity), `_summarize_numa_maps` (`pages_by_node`, `local_fraction`), the `required` predicate that must be armed |
| `/workspace/handoffs/active/numa-placement-defect-20260730.md` | Root-cause narrative and task list (T1–T8) |
| `/mnt/raid0/llm/tmp/e5_rederived.md` | Offline re-derivation of all 31 cells — per-slot / system-wide rates, achieved concurrency, salvage verdicts. Zero inference |
