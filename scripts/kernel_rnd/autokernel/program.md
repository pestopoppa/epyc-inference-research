# autokernel — program

This is the human control surface for the kernel research loop. It is a **procedure manual**,
not a research strategy: it tells you how to run a round, where the results go, and what you
may not touch. **It deliberately contains no search strategy and no hypothesis backlog** — the
one exception is the inbox at the bottom, which the operator appends to and the loop reads.
Finding the win is your job, not this file's.

This file carries no authority. It cannot license a freeze, a cutover, or a production reload.

---

## Setup

Work with the operator to:

1. **Agree a campaign id** — date-based, e.g. `ak-aug05`. It is validated by
   `execution.worktree.validate_campaign_id` and it namespaces the worktree, the journal and
   the evidence root. A campaign id is never reused.
2. **Read the in-scope material.** All of it, before the first round:
   - `execution/README.md` §0 (what exists), §1 (preflight), §6 (the honest list of what is
     still open). §6 decides whether today is a campaign or a plumbing session.
   - `data/autokernel_aa_20260804/README.md` — historical v8 A/A evidence behind the paired
     design. It is a regression fixture, not v9 ranking authority.
   - `data/autokernel_aa_20260805_rest_recovery/README.md` — the claimed six-run follow-up that
     tested the earlier decode drift across a 180-second idle boundary.
   - The hardware section of this file.
3. **Verify the frozen trees**, exactly as `execution/README.md` §1.1 prints them. If
   `/mnt/raid0/llm/llama.cpp` is not at `0db32c06e3e` on `production-consolidated-v9`, stop:
   something else moved production and every anchor you are about to take is wrong.
   Separately verify the reviewed measurement overlay at `b05a1618c3a` on
   `experimental-v9-autokernel-t1-hardening-final`; candidate worktrees start from that exact
   direct child of production so the hardened evaluator is present without patching serving.
4. **Verify the claim is acquirable and the host is quiet** — §1.2 and §1.3. Two A/A runs on
   2026-08-04 were destroyed by a legitimate co-tenant because we held no claim. The co-tenant
   did nothing wrong; nothing told it the host was in use.
5. **Confirm and go.**

The entrypoint:

```bash
cd /mnt/raid0/llm/epyc-inference-research
python3 -m scripts.kernel_rnd.autokernel.campaign --help
python3 -m scripts.kernel_rnd.autokernel.campaign --campaign-id ak-aug05 --candidate-id akc-0001 --model /mnt/raid0/llm/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
python3 -m scripts.kernel_rnd.autokernel.campaign --campaign-id ak-aug05 --candidate-id akc-0001 --candidate /path/to/candidate.patch --proposal-manifest /path/to/proposal-v4.json --calibration-bundle /path/to/current-v9-control-bundle --physical-envelope /path/to/physical-envelope.json --model /mnt/raid0/llm/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --journal-root /mnt/raid0/llm/autokernel/campaigns/ak-aug05 --execute --i-hold-the-host
```

`--model` is required and has no default — the cell must dispatch the path you changed, and a
default would let a candidate be measured on a model that never touches it. The path above is
the A/A's own model (`Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`, MoE, so it exercises
`MUL_MAT_ID`), which is what makes the drift bound comparable.
`--journal-root` is also mandatory on the executing path: every completed benchmark attempt is
fsynced there before it may be pooled, so a declared extension round cannot be re-run after its
result is observed. Use a new durable campaign directory; never put it under `/tmp` or a checkout.
`--proposal-manifest` is mandatory too. It must be a valid current-schema proposal record for the same campaign;
the driver fsyncs it before preflight, claim acquisition, candidate mutation, or build. A v2 record
remains readable history but cannot drive a new run because it has no representation/demand frame.
Current proposal-v4 records also carry `external_numbers` (an empty list when none are used); every
entry must include source revision and matching per-quant, same-basis roofline normalization.
`--calibration-bundle` is mandatory for execution and must be the accepted live-control bundle for
the exact production commit, measurement-instrument commit and recipe. The driver rejects the v8
bundle in `data/autokernel_controls_3pct_20260805/` on v9; those 3%/12-block values are historical
regression evidence, not constants to copy. The bundle supplies the contribution floor and B_min.
`--physical-envelope` (or `--ranked-units`) is also mandatory and binds the physical speed-of-light
guard to the exact measurement unit.
The CPU IQK known-win diagnostic is a parameter proposal, not a source patch: set
`change_class` to `parameter` and declare `change.parameter_surface` as
`{"candidate":{"ggml_iqk":"1"},"anchor":{"ggml_iqk":"0"}}`. The campaign
driver projects only that recipe-registered arm-local variant; it rejects identical
arms and every other key.

**The entrypoint is inert unless you ask for a run.** `--dry-run` is the parser's default, not a
flag you remember: without `--execute` it acquires nothing, spawns nothing and builds nothing —
it composes every step, prints the exact argv and env that WOULD be spawned, and emits no speed
number. `--execute` additionally requires the host attestation, proposal, current calibration and
physical envelope spelled out above, and refuses with exit 2 before host work without them. That is
the default because this is a shared host and the failure it prevents
is real: on 2026-08-04 two of six A/A runs were destroyed by a legitimate co-tenant. A flagless
invocation must never be the thing that starts a benchmark.

`--help` is authoritative. If this file and `--help` disagree, `--help` is right and this file
is the bug — fix it in the same session you notice it. `test_program_md.py` parses the commands
above with the real parser so the two cannot drift silently.

---

## What you CAN do

- **Edit kernel sources inside the campaign worktree** created by
  `worktree.create_campaign_worktree`. Everything under `ggml/` is fair game: layouts, repack
  paths, dispatch predicates, threading, fusion, build flags, new kernels, new backends.
- **Choose recipe *parameters*** from the registered parameter space (`evaluator.recipes`) —
  model, phase, batch, context, thread count for a declared non-canonical cell.
- **Carry a lineage across rounds** when the change is a sequence rather than one mutation —
  each step is still its own candidate with its own anchor, its own gates and its own verdict.
  A lineage buys you patience, not a relaxed rule; nothing here waives a gate.

## What you CANNOT do

- **Never modify, build in, rebase, commit to, or switch the branch of a frozen production
  tree** — `/mnt/raid0/llm/llama.cpp`, `/mnt/raid0/llm/whisper.cpp`, `/mnt/raid0/llm/qwentts.cpp`.
  `worktree.ProductionTreeViolation` and `t0_provider.ProductionTreeRefusal` enforce this; do
  not go around them. We version past production, we never patch it in place.
- **Never a name-pattern process operation.** No `pkill`, no `pgrep`, no matching on a command
  line. Kill only PIDs you captured yourself, and verify they are dead. On 2026-07-31 a
  name-pattern kill took out another agent's `llama-server` twice, and `earlyoom` — whose argv
  necessarily contains the names it guards.
- **Never run inference outside a held claim.** `t0_provider.require_claim` and
  `microbench.ClaimNotHeld` refuse. Acquire with
  `execution.cpu_region_claim.acquire_cpu_region_claim` (CPU) or
  `resource.device_claim.acquire_device_claim` (GPU). **Never unlink a lock file to clear one** —
  the flock IS the fact. If a crashed round left a claim behind, `acquire_cpu_region_claim`
  reclaims it itself once the payload is provably abandoned past `stale_grace_s`; orchestrator-
  shaped debris it cannot date is refused, and the sanctioned cleanup for that is the
  orchestrator's own `region-lock sweep`, never a second repo guessing. An expired claim is
  never stolen — nothing here preempts a live holder.
- **Never edit the evaluator or the recipe constants.** `evaluator/` and
  `scripts/lib/canonical_recipe.py` are read-only to the loop. **The loop may not tune its own
  instrument** — an accept rule the searcher can widen is not an accept rule.
- **Never construct a `Verdict` by hand.** `api.compute_verdict` is the only constructor and
  the mint token is checked.
- **Never look at a delta and then decide how much more to run.** N is `--blocks`, fixed on the
  `CampaignSpec` before the first block; `campaign.decide` REFUSES any other count, so a longer
  run is not admissible input rather than discouraged input. Do not enable optional stopping in
  `evaluator.statistics`; import it for calibration constants only.
- **No host reboot, no privileged cache action outside the sanctioned path, no new
  dependencies.**

---

## The accept rule is in code, and this file will not restate it

`karpathy/autoresearch`'s program.md contradicts itself here — "an improvement of ~0 but much
simpler code? Keep" against "if val_bpb is equal or worse, git reset". Prose cannot hold a
decision rule. The rule the driver applies is `campaign.decide`, and it is thirty lines:

```
campaign.decide(pairs, t0=..., blocks_precommitted=N, drift_bound=...,
                contribution_floor=..., calibration_evidence_ref=...) -> AcceptDecision
    RAISES AcceptRuleMisuse if T0 did not all-PASS      (no speed number exists for a
                                                         wrong kernel — not a penalised one)
    RAISES AcceptRuleMisuse if len(pairs) != N          (no optional stopping, by refusal)
    KEEP iff  min(delta) > 0  AND  median(relative) > contribution_floor
    otherwise REVERT, with the reason on the decision
```

You read `AcceptDecision.keep` and you journal `.reason`. You do not compare two throughput
numbers and form your own opinion, and you do not put a threshold in a proposal.

Three consequences of the measured A/A, so you understand why the rule is shaped this way:

- Between-run spread is **3.70% on pp512 and 4.32% on tg128** (CV 1.62% / 1.88%). A single-run
  strict-`<` is a coin flip with a decimal point on this host.
- Decode declined **monotonically** across the original four consecutive runs. The claimed
  six-run follow-up did not reproduce that trajectory: decode rose before the 180-second rest,
  the first post-rest run was 3.29% colder, and the second recovered to 0.50% below the pre-rest
  run. That rules out adding an inter-arm rest, but it does not make sequential A/B safe:
  day-to-day absolute throughput also shifted sharply. **Interleaved paired blocks are
  mandatory**, not preferred, and
  `Pair.order` carries which arm ran first so the pairing is checkable after the fact. The
  separate anchor-movement gate exists for the same reason: `min(delta) > 0` alone is satisfied
  by systematic drift, which is precisely what the A/A found.
- The candidate contribution floor and B_min come only from the current identity-bound calibration
  bundle. The v8 3% / B_min=12 result is historical and is rejected on v9. `drift_bound` remains a
  separate adjacent-anchor movement control; it is not reused as the candidate threshold.

**The rule is deliberately conservative.** Whether a +2% effect is resolvable is answered by the
current campaign's predeclared floor and MDE, never by importing the historical v8 answer. The
remedy for an unresolved effect is more information or a different declared campaign, never a
post-result looser rule.

**Simplicity is not an exception to any of this.** A change that deletes code and lands inside
the contribution floor is a REVERT under the rule — every parity result is. If you believe a
simplification is worth keeping anyway, that is a proposal to the operator with the decision's
own numbers attached, and it is **never** spelled as a keep the loop makes for itself. A
deletion that also deletes the computation is the exact failure this gate exists for and it
arrives wearing parity: `min(delta) > 0` does not protect you there, T0 does.

**There is a second verdict layer and it is not reconciled with the above.** `evaluator/api.py`
computes an `api.Verdict` (`status` × `effect_resolution` → `speed_rank_admissible`); the driver
does not call it, and `campaign.decide` does not consult it. Two live consequences you must know
before campaign #1: (1) `status == pass` is unreachable today — four T0 surfaces are
structurally `COULD_NOT_CHECK`, every class demotes to `inconclusive`, so `speed_rank_admissible`
is `False` for every candidate; and (2) a correctness FAIL and a parity resolution coexist on one
`Verdict` (`status=fail`, `effect_resolution=below_noise_floor`), so anything reading
`effect_resolution` without reading `status` accepts a kernel that broke `MUL_MAT_ID`. **Which
layer is authoritative is the operator's call, and it is the first thing to settle.** Do not
settle it by softening `_ON_GATE_COULD_NOT_CHECK` — that is tuning the instrument to make your
own result pass, and it is the one edit this file forbids outright.

---

## Output format

The driver prints a header, then the terminal state, then every release. `--json` adds the whole
result document. The five states are the whole vocabulary:

```
state: dry_run_composed      composed and printed; nothing was executed
state: preflight_refused     the host or the claim said no; no candidate was harmed
state: t0_failed             correctness failed; NO speed number was computed
state: decided               KEEP or REVERT, with .reason carrying the numbers
state: error                 the run raised; releases and the tree proof still ran
```

**`t0_failed` is a distinct terminal state on purpose, and it is the one to understand.** It is
not "failed with a bad speed number" — no blocks are planned at all, and `decide()` raises if
something calls it anyway. That is two locks on the same door, because the ordering is the whole
property and one lock is a convention.

**`exit 0` means the campaign terminated cleanly and production is byte-identical. It does not
mean the candidate won.** A REVERT is a successful campaign. Read the release lines and the
production-tree line before you read the decision: a run that decided KEEP but printed
`NOT RELEASED` on a claim is a worse outcome than a clean REVERT.

`state: t0_failed` is the thing to read first. A correctness failure makes the speed number
irrelevant — it is not a slower win, it is not a win.

**Throughput is reward-hackable and this is the difference that matters.** Deleting the
computation is the fastest kernel there is. `autoresearch` does not have this problem — its
`val_bpb` gets *worse* if you delete work. Ours does not, so correctness is a gate rather than
a metric: `t0_provider` produces the evidence, `correctness.py` scores it. The predecessor
harness tested `MUL_MAT` only, and a kernel that broke `MUL_MAT_ID` — MoE dispatch, run on
every token in production — passed it cleanly.

---

## Logging

There is no `results.tsv`. The record is `journal.Journal`: append-only, fsynced, sharded,
with rebuilt views asserted consistent at bootstrap. AutoPilot lost 232 trials and ~16 days of
compute to a restart that came up empty with nothing objecting.

- **Bank each round before starting the next.** Each persisted unit is a drain point.
- **Evidence goes under `data/<campaign_id>/`**, created by
  `storage.ensure_campaign_evidence_root` with `SHA256SUMS` and a README. **Never
  `/mnt/raid0/llm/tmp/`** — the 2026-07-04 async-prefetch win, the one real result this project
  ever produced, was written there and that directory no longer exists. `storage.is_scratch_path`
  refuses a citation into it.
- **One record shape.** `schemas.py` owns it. Do not invent a field.

---

## The loop

Your half is steps 1–4. `campaign.run_campaign` owns 5–10 and its order is the deliverable —
read `run_campaign`, not this list, if the two ever disagree.

```
LOOP FOREVER:
 -- yours, and none of it needs a claim ------------------------------------
 1. discover      profile, wall-share map, roofline utilisation, the do-not-repeat
                  ledger, and the open hypotheses in the inbox below.
 2. pick a target placement/launch -> dispatcher -> autotune -> layout/repack ->
                  fusion -> scheduling -> new kernel -> architecture. Skipping a
                  cheaper layer needs an evidence receipt, not an intuition.
 3. propose       one conceptual mutation, its declared surface files, its declared
                  symbol deltas, and its falsifier.
 4. criticise it  BEFORE spending a claim: refuse repeats of receipted negatives
                  and claims above the roofline ceiling. Land the change in the
                  worktree; --execute has nothing to apply otherwise.
 -- the driver's, and it is one invocation ---------------------------------
 5. preflight     host + claim. Refusing here costs nothing.
 6. claim -> worktree -> apply -> build
 7. T0            correctness. A failure RETURNS: no blocks are planned, no speed
                  number exists, `decide()` refuses to be called.
 8. N alternating paired blocks, N pre-committed on the spec
 9. decide        KEEP or REVERT; keep_or_revert acts on the worktree
10. finally:      release every held resource in reverse order, then prove the
                  production trees are byte-identical — on EVERY path, including
                  the one that raised.
 -- yours again -------------------------------------------------------------
11. bank the event, update the search state, resolve any inbox hypothesis this
    round settled, then go to 1.
```

**Step 10 runs from a `finally`, and that is the point.** A driver that dies holding a claim is
how the next session finds the host locked by a corpse, and a driver that skips the
production-tree proof on the failing path skips it exactly when it mattered. If
`prove_production_unchanged` ever comes back `FAIL`, or a release line says `NOT RELEASED`, that
outranks everything else you did today — stop and read it before you queue another candidate.

**Steps 1–4 are the whole reason a busy host is not a blocked host.** They need no claim. When
the machine belongs to someone else, this is where you live.

**Crashes**: a build failure or a tool crash is a result. Fix it if it is a typo; log it and
move on if the idea is broken. Do not spend three rounds resurrecting one candidate.

**Rule out the test method before believing a result.** A 26% decode collapse on 2026-08-04
was another session's stack starting during the probe, not our kernel. Reported as drift, it
would have justified almost any amount of machinery.

---

## NEVER STOP — and the five times you do

Once the loop has begun, do **not** pause to ask whether to continue. Do not ask "is this a
good stopping point?". The operator may be asleep and expects to wake up to results. If you
run out of ideas, think harder: re-read the profile, combine near-misses, read the papers the
oracle registry points at, try a more radical change.

You stop for exactly five things:

- **`DISK_PRESSURE`** — free space under the floor. `storage.py` computes it and owns the
  constant; it is a real stop because the next thing to fail is the journal.
- **`BUDGET_STOP`** — the campaign's declared compute or storage budget is spent. Also
  `storage.py`. Declared up front, like N.
- **A blocked instrument** — the driver cannot produce a decision at all (not "produced an
  unwelcome one"). `state: error`, or `AcceptRuleMisuse` on a well-formed run, is a stop. A
  REVERT is not: a REVERT is the loop working.
- **Plateau** — a declared number of consecutive rounds with no KEEP and no new information.
  Say the number before the campaign starts, or it is not a plateau, it is discouragement.
- **The claim is not acquirable.** This is not a stop, it is a *lane change*: you may not bench
  without the claim, but steps 1–4 need no claim at all. Discover, propose and criticise until
  the host is yours, then retry. Benching anyway is the one failure that makes every number in
  the campaign worthless.

Everything else — a failed candidate, a refuted hypothesis, an inconclusive round, an empty
frontier — is the loop working.

---

## What is known about making THIS hardware fast

Domain knowledge, not procedure. It is here because it is measured, and because a wrong hint
is worse than none — if you contradict one of these, you need a measurement, not an argument.

**EPYC 9655 (Zen 5 / Turin), 96 cores / 192 threads, 12 CCDs, NPS4, ~1.1 TB DDR5**

- **CPU decode is bandwidth-bound.** 27B Q8_0 decode saturates at ~26% of the ~460 GB/s
  roofline. A compute-side ukernel that wins +31.8% at 1 thread (AVX-512BW 8×8 Q8_0 GEMV) is
  worth +1–3% at production thread counts, because both paths converge on the same ceiling.
  **Compute micro-optimisation pays on prefill, not decode.**
- **Canonical CPU cell**: `taskset -c 0-95 numactl --interleave=all llama-bench -t 96 -fa 1
  -mmp 0`, with `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active
  OMP_DYNAMIC=false GGML_IQK=1`. Read from `scripts/lib/canonical_recipe.py` via
  `evaluator.recipes` — never typed from memory. Production once drifted off `interleave=all`
  on a 1.7% warm A/B and the front door ended up at 46% of canonical.
- `-fa 1` is **always explicit**: `llama-bench` defaults to `0` and the swing is 8–10%.
- **iqk (`GGML_IQK=1`) is a prefill win**: +42.7% GLM-IQ2, +39.0% Qwen-Next-IQ2, +33.0%
  Hy3-IQ1, decode neutral. It covers K/legacy quants plus IQ2/IQ3 and IQ4_XS. **IQ1 is still
  stubbed**, and Q2_K/Q3_K are deliberately off iqk (the Hy3 corruption lesson).
- **NUMA first-touch is the repeat offender.** A repack buffer allocated without `mbind`
  first-touches node 0; 26 GB of weights on one node behind 96 threads cost **2.8×**. Any new
  buffer type on this host must interleave. Note also that `mmap` *shares* NUMA placement
  across instances — only `--no-mmap` plus membind is node-local.
- **Zen 5 instruction reality**: `VPMADDUBSW` runs 2/cycle, `VPDPBUSD` 1/cycle. Porting a
  quantized GEMV to VNNI **loses** here — measured −1.3% on Q4_K, and a second independent
  −3.6%/+1.7% data point on Q8_0. Do not assume the modern instruction wins.
- **The x86 repack dispatcher has real holes**: Q5_K, Q6_K and Q8_0 are NEON-gated and fall
  through to `nullptr`. Naively enabling the generic path regresses **−66% to −71%** — the
  plumbing works, the hand-written x86 kernels do not exist. That is the gap, and it needs
  intrinsics, not a dispatcher flip.
- **Operating points**: 96t single-instance ≈ 49 t/s; 48t NPS4 ≈ 46.6; splitting into 4×48t or
  32×6t gives +44–58% *aggregate*. The optimum topology is model-size dependent — do not carry
  one model's answer to another.
- The April 2026 single-instance lever list (CCD pools, `GGML_NUMA_WEIGHTS`, barrier ordering,
  MUL_MAT+ADD fusion) was worked to exhaustion. That is scoped to **that list**, not a general
  claim that CPU work is finished.

**MI210 (gfx90a, CDNA2), 64 GB HBM2e, ROCm 6.2**

- **HIP only. Vulkan is architecturally impossible** — no ICD from anyone supports the
  compute-only MI200 family. Do not re-attempt it.
- **GPU host threads pin to `184-191`** — the MI210's node-3 SMT siblings. `88-95` is the
  superseded pinning and is wrong. Read it via `recipes.gpu_host_cpu_list()`, which parses the
  codified launcher so a correction there reaches here.
- **Bandwidth**: achievable **1433.3 GB/s** measured, 87.5% of the 1638 GB/s datasheet peak.
  Quote which denominator you used or the utilisation is not a number, and keep cross-vendor
  comparison spec-to-spec.
- **The ladder gap is the target, not the fp16 rung.** fp16 decode reaches ~62% of roofline,
  Q8_0 ~47%, Q4_K ~33% — so the quantized shortfall is an MMQ **dequant** artifact on CDNA2,
  not general kernel immaturity. Production does not serve fp16, so a win on that rung alone
  lands where no role is served.
- **Flash attention does not help MI210 decode** (`-fa 0` beat `-fa 1` on tg); it helps
  prefill. Default MMQ beats forced rocBLAS at batch 1.

**Measuring on this host**

- **Normalise to roofline utilisation** (`measured_tps / (achievable_bw / bytes_per_token)`),
  and for MoE count **active-expert bytes**, not model size. Raw speedups do not transfer.
  Utilisation is a routing input and a headroom bound — never a gate.
- **The recipe's own frequency check is wrong as written.** `FREQ_BOOST_MIN_CORES=80` above
  2.5 GHz **fails on a healthy idle host** — an idle EPYC parks its cores; this host read 16
  above 2.5 GHz idle and 117 under load. A preflight copying the THRESHOLD without copying the
  LOAD aborts on a good machine, and the first thing anyone then does is switch the check off.
  `campaign.check_boost_under_load` is the fixed form and it has **three** outcomes, not two:
  not under load → `COULD_NOT_CHECK` (never PASS — an unevaluated throttle check is not a
  passing one, and this box has silently sat at −60% for days); under load and boosting → PASS;
  under load and not → FAIL. The guard still bites; that is why it was fixed and not deleted.
- Verify the host is not throttled before a long run: this box has silently sat at −60% for
  days.
- Post-`drop_caches` re-reads pin one NUMA node. Model load order matters; load sequentially.

---

## Hypothesis inbox

**Append one line to the bottom of this file. That is the entire protocol.** No schema, no
ticket, no approval. The loop reads this section at step 1 and an entry here is ranked with
everything else, **never promoted because of who wrote it**. It faces step 4's critic unchanged
and it is subject to the do-not-repeat ledger (`schemas.py` owns that field); if it repeats a
receipted negative, the critic says so.

A falsifier is welcome and not required. With one, the loop can close the hypothesis; without
one, it will propose against it and close it when the evidence allows.

The loop **never deletes a line here.** When a round settles one, it appends the outcome and
the event id to that same line: `→ REFUTED ev-…`, `→ CONFIRMED ev-…`, `→ INCONCLUSIVE ev-…`.
An unmarked line is still open and gets re-surfaced every planning round — that is what stops
a question feeling "already tried" without a receipt.

```
- try a hand-written AVX-512BW 8x8 GEMV for Q6_K; the generic repack path is why it regressed
- Q4_K decode on gfx90a is dequant-bound — a fused dequant+GEMV should close half the ladder gap
- fuse the elementwise/norm cluster; if a wall-share map puts it under 20% I am wrong
```

<!-- OPERATOR: append below this line. One line each. -->
