# Model Registration Runbook — the gate every orchestration-stack model must pass

**Id**: `MRG-1` (Model Registration Gate, version 1)
**Created**: 2026-07-30
**Status**: BINDING as a process gate. It introduces **no new metric and no new protocol id** —
see §0.2 — so it does not touch the measurement trust boundary and needs no `MEASUREMENT.md`
amendment. Every figure it demands is produced under an already-ratified `P-BENCH-*` id.
**Composes**: [`numa-placement-measurement-protocol.md`](numa-placement-measurement-protocol.md)
(`P-BENCH-PLACEMENT-1`) — Step 4 delegates to it wholesale and restates none of it.
**Governed by**: [`/workspace/MEASUREMENT.md`](/workspace/MEASUREMENT.md) (constitution) and its
CPU annex [`/workspace/measurement/protocols/bench-cpu.md`](/workspace/measurement/protocols/bench-cpu.md)
(normative `P-BENCH-*` text). Where this runbook and those disagree, **they win**.

---

## 0. What this is, and what it is not

### 0.1 The rule

> **A model that has not completed Steps 0–9 of this runbook is NOT REGISTERED.
> An unregistered model MUST NOT be given a production role in the orchestration stack.**

Registration is a **gate**, not a guide. It is executed once per
`(model file, quantisation, kernel era)` triple and is re-executed whenever any of the three
changes (§9.4). Its output is a directory of raw evidence plus a signed-off `REGISTRATION.md`
whose step table has **no empty verdicts** — that table is the artifact a reviewer reads, and an
unexecuted step is an empty row, which is visible. Skipping is therefore a *documented refusal*,
never a silent omission.

### 0.2 What it is not

It is not a measurement protocol. It defines no metric, no instrument, no noise model and no
retroactivity rule. It is a **checklist over existing ratified protocols**, and every number it
requires carries the protocol id of whichever `P-BENCH-*` produced it. Consequences:

- It cannot be cited *instead of* a protocol id. `[MRG-1]` is not a claim citation.
- It cannot upgrade a figure's grade. If `P-BENCH-PLACEMENT-1` says a run missing measured
  locality is observation-grade, completing this runbook does not make it decision-grade.
- Amending it is ordinary engineering review, not a measurement-boundary amendment
  (`MEASUREMENT.md` §5).

### 0.3 Metric scoping — read this before writing any number

`MEASUREMENT.md` §1 is binding and is **not restated here**. The one consequence you must carry:
every figure produced under this runbook is an **instrument-level** figure, so its metric is
**tok/s** under a `P-BENCH-*` id, and it is fully decision-grade in that scope. `task_rate` is the
AutoPilot objective axis and never appears in a registration table. Do not convert between them.

### 0.4 Units contract

`P-BENCH-PLACEMENT-1` §0.2 defines the three mandatory qualifiers (`aggregation`, `spec_dec`,
`metric_source`). They apply to **every** number in a registration record, including Steps 1, 6, 7
and 8 which are outside that protocol's own scope. Restated only as an obligation, not a
definition: a tok/s figure in a registration record without all three qualifiers is not a result.

Two additional registration-specific obligations:

1. **Quantisation travels with the model name.** Every table row, every headline, every chart
   label reads `<model> <quant>` — `gemma-4-26B-A4B-it-ORIG Q4_K_M`, never `gemma`. Two quants of
   one model are two registrations and two sets of numbers; a bare model name is ambiguous by
   construction and has already produced one wrong-file comparison (see the `modelref_results.txt`
   caveat in `data/numa_placement/20260730-P-BENCH-PLACEMENT-1/README.md`).
2. **`instances`, `np per instance` and `T = instances × np` are three separate columns.** A row
   that reports only "np=4" is unreadable — four slots on one instance and one slot on each of
   four instances are different machines.

### 0.5 Instrument identity

Fixed per era; do not hand-type it. Take kernel branch/commit, `llama-server --version`, era
stamp, host topology and the OMP env stack from `P-BENCH-PLACEMENT-1` §0.1 and from
`scripts/lib/canonical_recipe.py` (`CANONICAL_PREFIX`, `CANONICAL_BENCH_FLAGS_LLAMA_BENCH`,
`CANONICAL_OMP_ENV`, `LLVM20_LIBDIR`). The only sanctioned `llama-bench` entry point remains
`scripts/benchmark/bench_canonical.sh`, which composes `validate_canonical_env()`,
`validate_host_environment()` and `assert_binary_resolves_correctly()` and acquires the
`region-lock` for the pinned cpu list, fail-closed. **Never reconstruct the recipe from memory**
(`feedback_use_codified_recipes_not_memory`).

---

## 1. The six failure modes this runbook makes structurally impossible

Each is stated as **what happened** → **why it was invisible** → **the check that catches it** →
**the step that owns that check**. A registration that has not executed all six checks is
incomplete regardless of how many numbers it contains.

`P-BENCH-PLACEMENT-1` §1 already defines five failure modes `F1–F5`. The mapping is given so the
two documents compose rather than duplicate: where a row says "= F*n*", the mechanism, the
measured cost and the mandatory check all live **there**, and this runbook adds only the
registration-level obligation to file the evidence.

| id | Failure mode | Owning step | Relation to `P-BENCH-PLACEMENT-1` |
|---|---|---|---|
| **R1** | Baseline numbers published as headlines | Step 2 | New here in *registration* form; the measurement obligation is annex-B gate 6 |
| **R2** | Placement never validated | Step 4 | = **F1** + **F3** — delegated entirely |
| **R3** | A metric that was not what its name said | all steps | = **F4** — delegated entirely |
| **R4** | A warm-cache A/B that could not detect the defect | Step 4 | = **F2** — delegated entirely |
| **R5** | No anchor | Step 3 | = `P-BENCH-PLACEMENT-1` §2 — delegated entirely |
| **R6** | Untested axes presented as unknown-but-fine | Steps 5, 6, 7, 8 | New here; no existing protocol requires axis coverage |

### R1 — Baseline numbers published as headlines

**What happened.** Three of the four production CPU roles were reported with speculative decoding
**OFF**. Those figures were published as the roles' throughput. Understatement ran **1.51×–2.12×**.
The 122B architect role read **11.04 tok/s** where production, on its own registry recipe, does
**24.00 tok/s** (per-stream, spec-dec on) — so the published number was less than half the truth
and pointed every downstream sizing decision the wrong way.

**Why it was invisible.** A spec-dec-off number is not wrong, it is *a different metric*. Nothing
in the artifact said which metric it was, and nothing required the two to be distinguishable. The
harness ran, the server started, the number was plausible.

**The check (R1-CHK).** Every registration figure is produced on the role's **full registry
acceleration recipe**, and every such figure MUST report **draft acceptance** alongside the rate,
as positive proof speculation actually engaged. Zero or absent acceptance on a role whose registry
`acceleration.type` is `speculative_decoding` is a **hard FAIL of the step**, not a footnote —
the arm did not run the recipe it claims. A spec-dec-off baseline is permitted **only** in a
section explicitly headed *Addendum — baselines*, only to isolate a single variable, and it may
never appear in a headline, a summary table, a dashboard tile, or a comparison against another
role. Owning step: **Step 2**. Normative source: annex B, `P-BENCH-PLACEMENT-1` gate 6.

### R2 — Placement never validated

**What happened.** Two production roles ran for months on cpusets straddling two NPS4 nodes with
no memory policy, at roughly half the achievable rate.

**Why it was invisible.** The cpuset was contiguous and was exactly half the machine, so it read
as deliberate. Live affinity verified. The preflight's locality gate computed
`required = no_mmap and len(expected_nodes) == 1` and therefore *observed and reported but never
failed*.

**The check (R2-CHK).** Registration MUST include **measured per-instance `local_fraction`**, read
from `/proc/<pid>/numa_maps` on the **live, loaded** server, filed in the registration record as
raw output. Intent is not evidence. Full mechanism, thresholds and the arming rule:
`P-BENCH-PLACEMENT-1` §1.1 and §1.3. Owning step: **Step 4**.

### R3 — A metric that was not what its name said

**What happened.** Throughput was computed as `tokens / wall_seconds` and reported as a decode
rate.

**Why it was invisible.** It reads low by a *configuration-dependent* amount, so it does not
offset a comparison, it tilts it.

**The check (R3-CHK).** Decode rate comes from `predicted_n` / `predicted_ms` (llama.cpp's own
per-request `timings`, i.e. the `eval time` line) or from a `llama-bench` `tg<N>` row. Nothing
else is a decode rate, in **any** step of this runbook. A wall-clock rate may be recorded only
under a name that says so. Full rule, both required derived rates, and the skip audit:
`P-BENCH-PLACEMENT-1` §1.4. Owning step: **all**; the reviewer checks it per table.

### R4 — A warm-cache A/B that could not detect the defect

**What happened.** A corrected arm was re-run warm and returned the *uncorrected* number, which
read as "we tested the fix and it did not help".

**Why it was invisible.** `numactl --interleave` and `--membind` bind at **FIRST TOUCH only**. If
the pages are already in the page cache, no fault occurs, no policy applies, and the process
inherits the previous arm's placement.

**The check (R4-CHK).** `sync` + privileged `drop_caches` before **every** placement arm — per
arm, not per campaign — with `cache_state ∈ {cold, warm}` recorded, warm arms always paired with
a cold companion, and re-warm only through the arm's own policy (never a bare re-read; see
`feedback_drop_caches_numa_eviction`). Full rule: `P-BENCH-PLACEMENT-1` §1.2. Owning step:
**Step 4**.

### R5 — No anchor

**What happened.** Nothing compared `np=1` against a known production rate, so an entire grid ran
at roughly half speed, unnoticed, to completion.

**Why it was invisible.** Every internal consistency check passed. The disagreement existed only
*between* the harness and production, and nothing looked across that boundary.

**The check (R5-CHK).** The `np=1` cell on the canonical placement runs **FIRST**, on a freshly
loaded server, after that arm's `drop_caches`, and is compared against a recorded production
anchor for that model+quant+spec-dec state. **Outside the band ⇒ the run is VOID**: stop, do not
proceed to any later step, do not report the run as a claim, an observation, or a "directionally
interesting" table. Fix the instrument and re-run. Anchor admissibility (six required fields) and
the establish-a-new-anchor path: `P-BENCH-PLACEMENT-1` §2. Owning step: **Step 3**.

### R6 — Untested axes presented as unknown-but-fine

**What happened.** Three axes had never been measured on production recipes and were treated as
settled: the **context-length curve**, **slot width**, and the **interaction with the GPU lane**.
Two of the three later turned out to carry large effects — long prompts starve a quarter to a
third of streams under batching, and a full-machine instance loses about a third of its throughput
to a co-resident GPU lane.

**Why it was invisible.** An unmeasured axis produces no artifact, so it leaves no trace in any
review. Absence of a warning read as absence of a problem.

**The check (R6-CHK).** Registration MUST produce a verdict on **every** axis in Steps 5–8. An
axis that genuinely does not apply is recorded as **`N/A` with a written reason** in the
`REGISTRATION.md` step table — never left blank, never omitted. `N/A` is a claim the reviewer can
reject. Owning steps: **5, 6, 7, 8**.

---

## 2. The registration checklist

Execute in order. Steps 0–4 are strictly sequential; Steps 5–8 may be reordered among themselves
but each requires Steps 0–4 to have passed. Every step names: **RUN**, **RECORD**, **PASS/FAIL**.

Throughout: hold a `region-lock` for exactly the physical footprint in use, for the whole run, and
record the holder and region set (`P-BENCH-PLACEMENT-1` §4.3). Host-health attestation via
`canonical_recipe.validate_host_environment()` is required per run, not per campaign — those
values reset at boot.

---

### Step 0 — Identity

**RUN.** Read the GGUF header only. Use `ggufmeta.py`
(`data/numa_placement/20260730-P-BENCH-PLACEMENT-1/ggufmeta.py`) — a header-only key/value reader
that loads no tensors and no model, so this step costs nothing and requires no lock.

```
python3 ggufmeta.py <gguf-path>
```

**RECORD.**

| field | source | note |
|---|---|---|
| `path` | filesystem | **Canonical root `/mnt/raid0/llm/models/`.** See path rule below. |
| `sha256` **or** `size_bytes` + `mtime` | filesystem | sha256 preferred; size+mtime acceptable for multi-hundred-GB files, but then both fields are mandatory |
| `quant` | filename / GGUF | e.g. `Q4_K_M`, `Q8_0`, `UD-IQ2_M` |
| `architecture` | `general.architecture` | e.g. `qwen35moe`, `gemma4`, `ssm_moe_hybrid`, `moe_hybrid` |
| `block_count` | `<arch>.block_count` | total layers |
| `kv_layers` | derived | layers that actually hold KV — see hybrid rule below |
| `attention.head_count_kv` | `<arch>.attention.head_count_kv` | may be a **per-layer array** |
| `key_length` / `value_length` | `<arch>.attention.key_length` / `.value_length` | if absent, `key_length = embedding_length / head_count` and `value_length = key_length` |
| `context_length` (trained) | `<arch>.context_length` | the trained window, not the served `-c` |
| `split_count` | filesystem | for sharded GGUFs, register the **first shard** path |

**Path rule.** All weights live under the single consolidated root `/mnt/raid0/llm/models/` (flat
GGUFs plus absorbed `<publisher>/<repo>/` trees). The former root
`/mnt/raid0/llm/lmstudio/models/` is now a **per-publisher symlink farm**: old absolute paths
resolve permanently and those symlinks must not be deleted, but **register the canonical root
path**, and do not add new files under the old root. Reference:
`docs/reference/models/REGISTRY_STANDARDS.md` → *Storage Root*. A registration whose recorded path
is under the legacy root FAILS this step — it is still resolvable, but it is not the canonical
identity and it silently diverges from `model_base_path`.

**Hybrid / SSM rule.** `attention.head_count_kv` is a scalar for dense and standard-MoE
architectures and a **per-layer array** for some hybrid and SSM models, where `0` means that layer
holds no KV. When it is an array, `kv_layers = count of non-zero entries` and
`n_head_kv = max(array)`; using `block_count` as `kv_layers` then overstates KV by the ratio of
total to attention layers. `ggufmeta.py` implements the rule; do not re-derive it by hand, and do
not decide the layer count by architecture name — read the header.

**The header outranks the registry.** Where a registry `model:` block carries a geometry field
(`attention_layers`, `ctx_max`, `size_gb`) that disagrees with the GGUF header, the **header is
authoritative** and the disagreement is recorded in the registration note as a registry defect to
be repaired separately. This is not hypothetical: `ingest_long_context`'s registry entry records
`attention_layers: 32` while its Q4_K_M GGUF header reports `block_count = 48` with a scalar
`head_count_kv`, i.e. KV on all 48 layers — a 1.5× difference in every capacity figure downstream.
Step 1 must be computed from the header.

**Quantisation rule.** From this point on, `<model> <quant>` is one atomic label. Any table,
headline or summary naming the model without the quant FAILS review (§0.4).

**PASS/FAIL.** PASS iff every field above is populated from the file itself (not from the
registry, not from a filename convention, not from memory), the path is under the canonical root,
and the recorded digest identifies the exact bytes that Steps 2–8 will serve. Any field read from
a secondary source ⇒ FAIL.

---

### Step 1 — Capacity arithmetic (no inference)

Pure arithmetic. No server, no lock, no inference. Run it **before** committing machine time,
because it is what tells you which shapes in Step 5 are even legal.

**RUN.** `maxctx.py` (`data/numa_placement/20260730-P-BENCH-PLACEMENT-1/maxctx.py`), extended with
this model's Step-0 geometry. Reuse its constants by name — `NODE_FREE_GIB`, `OVERHEAD_GIB`,
`SHAPES`, `KVQ` — do not retype the numbers.

**The arithmetic.**

```
KV_bytes_per_token = kv_layers × n_head_kv × ( key_length × bytes(k_quant)
                                             + value_length × bytes(v_quant) )

max_ctx(shape)     = min( trained_context,
                          (shape_RAM − weights − overhead) / KV_bytes_per_token )
```

`bytes(·)` per element comes from `KVQ` in `maxctx.py`; the KV quant pair is **the role's
configured pair**, not a default — `worker_general` runs `k=q8_0 v=q8_0`, `architect_general` runs
`k=q4_0 v=f16`. A wrong quant pair silently changes the answer by up to ~2×.

**`shape_RAM` is what ONE instance can reach.** This is the part that has been got wrong:

| shape | policy | reach | failure behaviour |
|---|---|---|---|
| quarter | `--membind=<node>` | **its own node only** (`NODE_FREE_GIB`, measured ≈263 GiB free of 283 GiB) | **FAILS — it does not spill.** Exceeding the node's free memory is an allocation failure, not a slowdown |
| half | `--interleave=<node pair>` | two nodes | — |
| full | `--interleave=all` | all four nodes | — |

**`--no-mmap` multiplies the weights, not the KV.** Under `--no-mmap` each instance holds a
**private copy of the WHOLE model** — a quarter instance holds the entire model, not a quarter of
it. Count weights **per instance**, `N` times over for an `N`-instance fleet. This is the
difference between a fleet that fits and one that does not, and the RAM bill is large enough to be
the deciding factor on its own (see `no_mmap_budget.md` in the placement attestation directory for
the per-role figures).

**RECORD.**

- `KV_bytes_per_token` (state it in KiB/token) and the k/v quant pair it assumes.
- `max_ctx` for **each shape the role will actually run** in Step 5, each labelled with its
  binding constraint: `trained-ctx` (headroom to spare) or `RAM`.
- **Per-node resident total for the intended lineup**: for each NPS4 node, the sum of
  `weights + KV(at the served -c) + overhead` for every instance that will be resident on it,
  including instances belonging to *other* roles in the intended lineup.
- The served `-c` and the per-request context, if the role divides `-c` across slots.

**PASS/FAIL.** PASS iff **every** node's projected resident total is **strictly under**
`NODE_FREE_GIB`. FAIL otherwise — and a FAIL here is not "try it and see": a `--membind` quarter
over budget fails to allocate. Record the margin, not just the verdict.

> Empirically, for the currently deployed models this step passes with large headroom and the
> binding constraint is the **trained context**, not RAM. That is a finding, not an assumption:
> a new model with a wider KV geometry (more `kv_layers`, more `n_head_kv`, `f16` KV) can invert
> it, which is exactly why the arithmetic is a mandatory step rather than a rule of thumb.

---

### Step 2 — Production recipe

This is the step that makes **R1** impossible.

**RUN.** Extract the role's `acceleration` block verbatim from
`epyc-orchestrator/orchestration/model_registry.yaml` and reproduce it exactly on the measurement
server. Paste the extracted YAML into the registration record so the reviewer can diff intent
against argv.

**The three traps.** These are not stylistic; each one silently produces a spec-dec-off number
that looks like a valid measurement.

1. **Self-draft roles must pass NO `-md`.** `frontdoor` and `architect_general` use `draft-mtp`
   **self-draft**: the registry's `draft_model` resolves to the *same real path* as `model.path`,
   and the launcher suppresses `-md` for same-realpath drafts (`_same_real_model_path` in
   `epyc-orchestrator/scripts/server/orchestrator_stack.py`, `stack_commands.py`, and
   `src/inference/model_server.py`). Passing `-md <the same file>` is not "being explicit" — it
   loads a second copy and is not the production code path. Measure with
   `--spec-type draft-mtp --spec-draft-n-max <draft_max> --device-draft none` and no `-md`.
2. **`worker_general` uses a SEPARATE draft model.** Its `acceleration` names a distinct
   `draft_role` resolving to a different GGUF (the `assistant-v6-Q8_0` file), plus `draft_max`,
   `draft_p_min`, `threads_draft` and `ubatch`. It therefore **does** take `-md <draft path>`, and
   all four of those parameters are part of the recipe. This role had never been measured with its
   draft model at all: speculation had been disabled to dodge the gemma4 self-MTP
   `ASSERT(S>0)` wedge — but production does not use self-MTP for gemma, it uses a separate draft,
   which is a different code path and does not wedge. Disabling a feature to avoid a bug in a
   configuration production does not run is how a 1.51× understatement gets published.
3. **`ingest_long_context` has `acceleration: {type: none}` and spec-off genuinely IS its
   recipe.** Its registry entry additionally carries
   `constraints.forbid: [speculative_decoding, eagle]` with the reason that an SSM's recurrent
   state cannot fork for an external draft. For this role, and only for roles with an explicit
   `type: none`, a spec-dec-off number **is** the production number — and it must still be
   labelled `spec-dec off` in every table, because the label is what distinguishes it from R1.

**RECORD.** Per arm: complete argv; the extracted `acceleration` YAML; `spec_type`; `draft_max`;
draft model path or the explicit note "self-draft, `-md` suppressed"; `threads_draft`; `ubatch`;
KV quant pair; and — mandatory — **`draft acceptance`**, read from the server's own
`draft acceptance = <x>` log line, reported as mean over the measured requests alongside the rate.

**PASS/FAIL.**

- Role's registry `acceleration.type` is `speculative_decoding` ⇒ **PASS requires a non-zero,
  reported draft acceptance on every measured arm.** Absent or zero acceptance ⇒ **FAIL**: the
  recipe did not engage and no figure from that arm may be reported as a production rate.
- Role's registry `acceleration.type` is `none` ⇒ PASS requires the recorded absence of
  speculation arguments **and** an explicit `spec-dec off` label on every figure.
- Any figure from this step that appears outside a section headed *Addendum — baselines* while
  carrying `spec-dec off` on a `speculative_decoding` role ⇒ **FAIL**.

**Reference implementation.** `data/numa_placement/20260730-P-BENCH-PLACEMENT-1/prodopt.sh` runs
exactly this shape for all four current roles, including the acceptance parse, and its committed
results (`prodopt_results.txt`, n=3, observation-grade) are the worked example of the R1 delta.

---

### Step 3 — Anchor gate

**RUN.** Measure `np=1` on the canonical placement, **FIRST**, on a freshly loaded server, after
that arm's `drop_caches`, using the Step-2 production recipe. Compare against the recorded
production anchor for this model+quant.

Anchor admissibility (model+quant identity with digest; spec-dec state; aggregation; `n`; explicit
low/high **band**, not a point; era; prompt-length regime for long-context roles) is defined in
`P-BENCH-PLACEMENT-1` §2.2 and is not restated. Take the anchor from a path **independent of the
thing under test** — an anchor drawn from the defective path confirms the defect.

**New model with no anchor.** `P-BENCH-PLACEMENT-1` §2.2 permits the run to *establish* one: run
the `np=1` cell to `P-BENCH-1` rep discipline, record it as the candidate anchor with its band, and
label the campaign observation-grade. **Registration adds one requirement on top**: the candidate
anchor must be **independently reproduced** — a separate invocation, on a separate day or after a
reboot, landing inside the proposed band — before the model may be given a production role. A
single-invocation anchor is a self-fulfilling gate.

**RECORD.** Anchor source instrument and value; band; `n` behind the anchor; the measured `np=1`
figure with all three units qualifiers; in-band verdict; for a newly established anchor, both
invocations.

**PASS/FAIL.** In band ⇒ PASS, proceed. **Outside band ⇒ the entire run is VOID.** Stop. Do not
run Steps 4–8. Do not report the `np=1` figure. Fix the instrument, then restart from Step 3.

---

### Step 4 — Placement

**RUN.** The full arm set `A0–A4` per `P-BENCH-PLACEMENT-1` §3, interleaved / order-randomized
across replicate blocks, executing all five mandatory checks `F1-CHK`–`F5-CHK` from its §1 and
recording every field in its §4.

**This runbook restates none of that.** The five gates, the arm definitions, the bridge-cell rule
(`A1` is not optional — drop it and policy is confounded with cpuset), the `A3`/`A4` identity
requirement, the drop-caches discipline, the grading rule, and the `LOCALITY_THRESHOLD` /
`INTERLEAVE_TOLERANCE` / `ACHIEVED_CONCURRENCY_FLOOR` pre-registration obligation all live there
and are cited, not copied. Read that document before running this step.

**What registration adds, and only this:**

1. **File it.** The measured per-instance `pages_by_node` and `local_fraction` — raw
   `/proc/<pid>/numa_maps` output, or the `numaloc.py` summary beside the raw capture — go **into
   the registration directory**, not into a scratch path. A locality reading that was computed and
   discarded is not evidence.
2. **Every arm runs the Step-2 production recipe** (annex B, `P-BENCH-PLACEMENT-1` gate 6). A
   placement campaign on spec-dec-off arms measures placement correctly and registers the model
   wrongly.

**RECORD.** Per instance: `cpuset`, `cpuset_nodes` + `n_nodes`, `threads` (state its physical
vs SMT relationship to the cpuset), `numactl_policy` from realized argv, `mmap_mode` from
`/proc/<pid>/cmdline`, `drop_caches` + `cache_state`, `pages_by_node`, `local_fraction`,
`live_affinity_verified`, `instance_start_order`. Per arm: the §4.2 result fields.

**PASS/FAIL.** Per `P-BENCH-PLACEMENT-1` — including its hard rejects (multi-node cpuset with no
policy; `--membind` under shared mmap as a placement arm; shared-mmap fleet arm without recorded
start order; wall-clock presented as a decode rate; rung below the pre-registered concurrency
floor). Registration adds one: **a step with no filed per-instance `local_fraction` FAILS**, and
the model is not registered.

> `live_memory_placement_verified: true` means the placement was **observed**, not that it was
> correct. Read `local_fraction`.

---

### Step 5 — Shape × concurrency

**RUN.** Sweep **every shape the role can legally run** (per Step 1) × `-np`.

Shape definitions are fixed and are the ones in
`data/numa_placement/20260730-P-BENCH-PLACEMENT-1/shapes_prodopt.sh`:

| shape | instances | cpuset(s) | policy | `-t` |
|---|---|---|---|---|
| full | 1 | `0-95` | `--interleave=all` | 96 |
| half | 2 | `0-47,96-143` / `48-95,144-191` | `--interleave=0,1` / `--interleave=2,3` | 48 |
| quarter | 4 | the four NPS4 node cpusets | `--membind=0..3` | 24 |

All instances `--no-mmap`, so each owns node-local weights (else Step 4's `F3` reject applies).
`-np` ladder: at minimum `{1, 2, 4}`, extended upward until either the aggregate stops rising or
Step 1's capacity arithmetic forbids the next rung. Every cell runs the Step-2 production recipe
and reports acceptance.

**RECORD — five columns, always, never conflated:**

| `instances` | `np per instance` | `T = instances × np` | per-stream tok/s | aggregate tok/s |
|---|---|---|---|---|

plus `spec_dec`, `metric_source`, `accept`, `reps`, and the per-cell `local_fraction` from Step 4's
discipline.

**The comparability rule.** **Only equal-`T` rows are comparable.** One full instance at `np=4` and
four quarter instances at `np=1` are both `T=4` and may be compared; one full instance at `np=4`
and two half instances at `np=4` are `T=4` and `T=8` and may not. **A single conflated number is
not an acceptable output of this step** — "the model does X tok/s" answers no question a lineup
decision asks.

**Interpretive note (not a rule).** The winning shape is **model-dependent**, and it has already
reversed between architectures on this host at the same `T`. Do not carry another model's shape
verdict into this one, and do not assume small-active-parameter MoEs favour quartering — measured,
the opposite has held. See `shapes_prodopt_results.txt` and `gapfill_results.txt`
(observation-grade) for the current per-role picture.

**PASS/FAIL.** PASS iff: every legal shape × every ladder rung has a cell or a written `N/A`
reason; every cell carries all five columns plus acceptance; no cross-`T` comparison appears in
the record; and every rung's achieved concurrency clears the pre-registered
`ACHIEVED_CONCURRENCY_FLOOR` (`P-BENCH-PLACEMENT-1` `F5-CHK`) or is excluded and labelled
under-saturated.

---

### Step 6 — Context curve

The axis that had never been measured on production recipes.

**RUN.** Decode **and** prefill at a stated ladder of prompt lengths, on the Step-2 production
recipe, at the shape the role serves.

**Standing minimum ladder**: **~0.5k, ~8k, ~32k** prompt tokens, generated by
`data/numa_placement/20260730-P-BENCH-PLACEMENT-1/mkprompts.py`. Roles whose production context
exceeds 32k add rungs up to their served `-c` (per-role rungs: **TBD**, §5). Record the **actual
tokenised prompt length** reported by the server, not the target — the two differ, and the
difference matters at the top of the ladder.

**`-c` allocation and context OCCUPANCY are different costs.** Do not conflate them:

- **Allocation** (`-c N`) reserves KV for `N` tokens **up front**, and where the server divides
  `-c` across slots, the per-request window is `-c / slots`. This cost is paid at load time whether
  or not any request is long, and it is what Step 1's arithmetic budgets.
- **Occupancy** is how many tokens a given request actually holds. This is what moves the decode
  rate — attention cost grows with the resident sequence.

A table that varies `-c` and reports a rate has measured allocation. A table that varies the prompt
at fixed `-c` has measured occupancy. **Say which.** Both are legitimate; silently mixing them
produces a curve that means nothing.

**RECORD.** Per rung: `prompt_tokens` (measured), prefill tok/s, decode tok/s (per-stream), `-c`,
slots, per-request window, `spec_dec`, **draft acceptance at that rung**, `metric_source`, reps.
Acceptance is per-point because it is not constant across context — it is the mechanism by which a
long-context decode rate can fall faster than attention alone explains.

**PASS/FAIL.** PASS iff every ladder rung has decode **and** prefill, both from
`predicted_n`/`predicted_ms` (never wall-clock), with measured prompt tokens and per-point
acceptance, and the record states explicitly whether allocation or occupancy was varied. Any rung
missing acceptance on a speculating role ⇒ FAIL (R1-CHK applies per point, not per campaign).

---

### Step 7 — Slot width

The axis where the median lies.

**RUN.** Sweep `-np` at the role's realistic prompt lengths — at minimum one short-prompt and one
long-prompt (≥8k) arm at each `-np` rung, so the comparison is **within-arm**. Reference
implementation: `data/numa_placement/20260730-P-BENCH-PLACEMENT-1/slotcheck.sh`.

**RECORD — the distribution, not the median.** Per cell: `p10`, `p50`, aggregate tok/s, the
per-stream `min`, the count and percentage of **starved** streams, and the spread (`p50/p10`).

**Why p10 is mandatory.** At 8k prompts on gemma the median looked healthy while **25–29% of
streams fell below a quarter of the cell median** — a spread of 25× to 44×. Aggregate throughput
concealed it completely and the median concealed it almost completely. The mechanism is
prefill/decode interference: a stream whose decode is repeatedly preempted by other streams'
prefill effectively starves. Only an order statistic in the lower tail sees this.

**Pre-register the starvation threshold.** The definition used in the reference implementation is
*a stream decoding below 25% of that cell's median*. The **binding PASS threshold for
registration — the maximum tolerable starved fraction, and the minimum tolerable `p10` as a
fraction of `p50` — is TBD** (§5) and MUST be pre-registered before the sweep, not chosen after
seeing the distribution.

**RECORD also**: `-np` semantics for the record — `-np` is a **ceiling on concurrent slots, not a
fixed cost**; measured decode rate tracks the number of slots *active at that instant*, so
survivors speed up as other streams finish. But slot width is **not free**: `-c` is divided across
slots and KV for all slots is allocated up front, so a wide `-np` costs memory and per-request
context whether or not the slots are busy (Step 1 and Step 6 both consume this).

**PASS/FAIL.** PASS iff every cell reports `p10` alongside `p50` and aggregate, the starved
fraction is computed against a **pre-registered** threshold, and the observed starved fraction is
within it. A cell reported with a median only ⇒ **FAIL**. Verdict is per prompt-length arm: it is
expected and normal for a role to PASS at short prompts and FAIL at long, and that split verdict
**is** the serving-policy answer (raise slots for short-prompt roles, hold them at 1 for
long-context roles) — record it as such.

---

### Step 8 — Co-residency

**RUN.** If the role can be co-resident with an active GPU lane, measure it against one. If it
cannot, record `N/A` with the reason.

**The topology fact that decides the answer.** The GPU lane pins its host threads to **logical
184-191**. Core *i* pairs with thread *i+96*, so those fold to **PHYSICAL cores 88-95**, which sit
inside `node3 = 72-95,168-191` — **region q3**. Therefore:

| CPU shape | cpuset | shares cores with the GPU lane? |
|---|---|---|
| full | `0-95` | **YES** — `88-95` ⊂ `0-95` |
| half (low) | `0-47,96-143` | **NO** |
| half (high) | `48-95,144-191` | **YES** |
| quarter q3 | `72-95,168-191` | **YES** |
| quarters q0/q1/q2 | — | **NO** |

**Measured** (operator-supplied, attestation **TBD** §5): under a bandwidth-generating GPU-lane
proxy, a **full instance lost 34%**; the **low half lost nothing**. That is not a small correction
— it is larger than the gap between shapes in Step 5, so a shape decision taken without this axis
can be reversed by it.

**RECORD.** Whether the role is co-resident-capable and why; the lane proxy's definition (what
generates the bandwidth, and its own placement); the CPU shape's cpuset and its intersection with
`88-95`; per-stream and aggregate tok/s **with the lane idle** and **with the lane active**,
same arm, same recipe, same cache discipline; the loss as a percentage; `local_fraction` under
both conditions.

**PASS/FAIL.** PASS iff the role is measured against an active lane in **every** shape it may be
deployed in, or each unmeasured shape carries a written `N/A` reason. A shape whose cpuset
intersects `88-95` and which was measured **only** with the lane idle is **not registered for
that shape** — it may not be deployed there. This is the check that converts "we never tested it"
from an invisible gap into a deployment-blocking empty row.

---

### Step 9 — Registration record

**Where.** One directory per registration:

```
data/model_registration/<YYYYMMDD>-<model-slug>-<quant>/
```

following the convention already established by
`data/numa_placement/20260730-P-BENCH-PLACEMENT-1/`: **every raw log lives beside the script that
produced it**, so every number is re-derivable from the directory alone, with no reference to a
scratch path and no reliance on anyone's shell history. Scratch paths under `/mnt/raid0/llm/tmp/`
are working space; a registration that cites one has not been filed.

**What must be persisted.**

| artifact | content |
|---|---|
| `REGISTRATION.md` | The step table (below), identity block, and every headline figure in claim grammar |
| `<step>.sh` | The exact script that produced each step's numbers, committed, not reconstructed |
| `<step>_results.txt` | That script's raw output |
| `<step>_<arm>.log` | Raw `llama-server` / `llama-bench` stderr per arm — the `eval time` and `draft acceptance` lines are the primary record |
| `numa_maps/` | Raw per-instance `/proc/<pid>/numa_maps` captures (Step 4) |
| `identity.json` | Step 0 fields + kernel branch/commit/`--version` + binary and library SHA-256s + `ldd` + complete effective environment |
| `capacity.txt` | Step 1 arithmetic output, including the per-node lineup projection |
| `acceleration.yaml` | The role's `acceleration` block, copied verbatim from the registry at run time |
| `host_health.json` | `validate_host_environment()` output + uptime tier + `region-lock` holder and region set + process witness |

**The step table** — this is the mechanism that makes a skipped step visible. It goes at the top of
`REGISTRATION.md` and every row must carry a verdict:

| # | Step | Verdict | Artifact | Note |
|---|---|---|---|---|
| 0 | Identity | PASS / FAIL | `identity.json` | |
| 1 | Capacity arithmetic | PASS / FAIL | `capacity.txt` | margin per node |
| 2 | Production recipe | PASS / FAIL | `acceleration.yaml`, `prodopt_results.txt` | acceptance per arm |
| 3 | Anchor gate | PASS / **VOID** | `anchor_results.txt` | band + source |
| 4 | Placement | PASS / FAIL | `placement_results.txt`, `numa_maps/` | min `local_fraction` |
| 5 | Shape × concurrency | PASS / FAIL / **N/A + reason** | `shapes_results.txt` | |
| 6 | Context curve | PASS / FAIL / **N/A + reason** | `ctx_results.txt` | allocation vs occupancy |
| 7 | Slot width | PASS / FAIL / **N/A + reason** | `slots_results.txt` | threshold, p10, starved % |
| 8 | Co-residency | PASS / FAIL / **N/A + reason** | `coresidency_results.txt` | per shape |

**An empty verdict cell means NOT REGISTERED.** `N/A` is a verdict and requires a written reason a
reviewer can reject; blank is not.

**Claim grammar every figure must carry.** Per `MEASUREMENT.md` §3, a claim is
`(metric, protocol-id, n/reps, date, attestation ref)`. In a registration record, with the
`P-BENCH-PLACEMENT-1` §5.2 units contract inline:

```
<model> <quant> <role> <axis> <value> tok/s <per-stream|aggregate(T=<n>)>,
spec-dec <on|off>[, accept <a>], <shape/arm>, local_fraction <f>
[<P-BENCH-*>, n=<reps>, YYYY-MM-DD, attest <path>]
```

- ✅ `Qwen3.5-122B-A10B Q4_K_M architect_general decode 24.00 tok/s per-stream, spec-dec on (draft-mtp n_max 4), accept 0.831, arm A2, local_fraction TBD [P-BENCH-PLACEMENT-1, n=3, 2026-07-30, attest data/numa_placement/20260730-P-BENCH-PLACEMENT-1/prodopt_results.txt]` — fields complete; rep adequacy for a *decision* is still judged by the owning protocol's reps rule.
- ❌ `architect does 11.04 tok/s` — no quant, no aggregation, no spec-dec state (and it is the R1 baseline).
- ❌ `gemma 256.77 tok/s` — no quant, no `T`, aggregation unstated, and cross-`T` incomparable.

**Reps.** Registration figures follow the `P-BENCH-1` rule (annex B): **≥5 reps for effects ≥5%,
≥10 for effects ≤2%**, report **median + MAD**. The committed 2026-07-30 `n=3` reference figures
cited throughout this runbook are **observation-grade** and are worked examples of *method*, not
registration-grade values.

**§9.4 — Re-registration triggers.** A registration is valid for exactly one
`(model file, quantisation, kernel era)` triple. Re-run this runbook, in full, when **any** of the
following changes:

1. the GGUF bytes (digest change), including a re-quantisation or a re-download;
2. the quantisation, including the KV quant pair;
3. the kernel era (`instrument_eras.yaml`) — e.g. a v8 → v9 promotion;
4. the role's registry `acceleration` block;
5. the host topology (an NPS reboot invalidates every cpuset in the record);
6. the shape or lineup the role is deployed in, where Step 5/8 has no cell for the new shape.

Items 4–6 permit a **partial** re-run of the affected steps only if Steps 0 and 3 are re-executed
and pass unchanged; items 1–3 always require the full runbook.

---

## 3. REVIEWER CHECKLIST

One page. Run down before accepting any registration. Each line is a read of **recorded evidence**;
none of them is satisfied by reading a launch command or a summary.

**Identity and record**

- [ ] Model named as `<model> <quant>` in **every** table, headline and summary. No bare names.
- [ ] GGUF path under `/mnt/raid0/llm/models/`, with sha256 (or size **and** mtime).
- [ ] Geometry read from the GGUF header, not the registry. Hybrid/SSM: `kv_layers` from the
      non-zero entries of the `head_count_kv` array, not `block_count`.
- [ ] Era + kernel stamped: branch, commit, `llama-server --version`, binary/library SHA-256s.
- [ ] Every raw log is in the registration directory, beside its script. No `/mnt/raid0/llm/tmp/`
      citations.
- [ ] **Step table has no empty verdicts.** Every `N/A` carries a written, rejectable reason.

**Recipe (R1)**

- [ ] Each arm's argv matches the registry `acceleration` block, which is filed verbatim.
- [ ] Self-draft roles pass **no `-md`**. Separate-draft roles pass `-md` plus `draft_max`,
      `draft_p_min`, `threads_draft`, `ubatch`.
- [ ] **Non-zero draft acceptance reported for every arm of every speculating role.** Missing or
      zero ⇒ reject the figure.
- [ ] Any `spec-dec off` figure on a speculating role appears **only** under *Addendum —
      baselines*, never in a headline, summary or cross-role comparison.

**Anchor (R5)**

- [ ] `np=1` ran **FIRST**, freshly loaded, after `drop_caches`.
- [ ] Anchor states model+quant, spec-dec state, aggregation, `n`, explicit band, era, and
      prompt-length regime — and comes from a path independent of the thing under test.
- [ ] In band. Outside band ⇒ the run is VOID and must not be reported at all.
- [ ] New anchor ⇒ independently reproduced on a separate invocation/day.

**Placement (R2, R4)**

- [ ] Every cpuset expanded against the NPS4 map, not read off a constant's name. Any `n_nodes > 1`
      with `numactl_policy: none` ⇒ **REJECT**.
- [ ] `drop_caches` recorded **per arm** with `cache_state`; every warm arm has a cold companion.
- [ ] **`pages_by_node` + `local_fraction` filed for EVERY instance**, from live `/proc/<pid>/numa_maps`.
      Missing ⇒ not registered.
- [ ] `local_fraction` **checked against the pre-registered threshold**, not merely printed.
      `live_memory_placement_verified: true` is not a pass.
- [ ] mmap mode per instance from `/proc/<pid>/cmdline`. `--membind` under shared mmap ⇒ reject as
      a placement arm. Shared-mmap fleet without recorded start order ⇒ reject as non-reproducible.
- [ ] All five arms `A0–A4` present, `A1` actually run, `A3`/`A4` differing **only** in mmap mode,
      arms interleaved not blocked.

**Metric (R3)**

- [ ] Every decode rate from `predicted_n`/`predicted_ms` or a `llama-bench` `tg` row. Any
      `tokens / wall_seconds` presented as a decode rate ⇒ **REJECT**. Skip audit reported.
- [ ] Every tok/s carries `per-stream` vs `aggregate(T=n)` **and** `spec-dec on/off`. Long-context
      rows also carry measured prompt tokens.
- [ ] Reps meet the `P-BENCH-1` rule; median + MAD reported.

**Axes (R6)**

- [ ] Step 1 per-node lineup projection present, and **under** `NODE_FREE_GIB` on every node, with
      the margin stated. `--no-mmap` weights counted **per instance**.
- [ ] Step 5: `instances`, `np`, `T`, per-stream, aggregate as five columns. **No cross-`T`
      comparison anywhere.** Every legal shape covered or `N/A` with reason.
- [ ] Step 5: achieved concurrency per rung against nominal; sub-floor rungs excluded and labelled.
- [ ] Step 6: decode **and** prefill at each ladder rung, measured prompt tokens, **acceptance per
      point**, and an explicit statement of whether `-c` allocation or context occupancy was varied.
- [ ] Step 7: **`p10` reported, not just the median**; starved fraction computed against a
      **pre-registered** threshold; verdict given per prompt-length arm.
- [ ] Step 8: co-residency measured for every deployable shape, or `N/A` with reason. Any shape
      whose cpuset intersects **physical 88-95** measured only with the lane idle ⇒ **not
      registered for that shape**.

**Sign-off**

- [ ] `region-lock` held for the exact physical footprint for the whole run; process witness shows
      no foreign llama-family overlap.
- [ ] Host-health attestation present (THP enabled + defrag, governor, `numa_balancing = 0`,
      `perf_event_paranoid` if perf-wrapped, uptime tier).
- [ ] Every headline figure carries `(metric, protocol-id, n/reps, date, attestation ref)`.
- [ ] Re-registration triggers (§9.4) recorded so the record's expiry conditions are explicit.

---

## 4. Composition map — what comes from where

| Concern | Authority | This runbook's role |
|---|---|---|
| Claim grammar, metric scoping, retroactivity, governance | `MEASUREMENT.md` | cites |
| `P-BENCH-*` normative text, reps rules, promotion decision rules | `measurement/protocols/bench-cpu.md` (annex B) | cites |
| Placement/concurrency measurement: `F1–F5`, arms `A0–A4`, anchor gate, per-arm evidence, grading | `numa-placement-measurement-protocol.md` | delegates Step 4 and Step 3 wholesale |
| Canonical recipe constants, env stack, binary/library validation, host health | `scripts/lib/canonical_recipe.py`, `scripts/benchmark/bench_canonical.sh` | imports by name |
| Model paths, storage root, registry field format | `docs/reference/models/REGISTRY_STANDARDS.md` | cites |
| Role acceleration recipes | `epyc-orchestrator/orchestration/model_registry.yaml` | reads at run time; files verbatim |
| Capacity arithmetic + GGUF header reader | `data/numa_placement/20260730-P-BENCH-PLACEMENT-1/{maxctx,ggufmeta}.py` | reuses |
| **Which measurements a model owes before it may hold a production role** | **this runbook** | **owns** |

The last row is the whole of the runbook's original content. Everything above it is a pointer.

---

## 5. Values still required (TBD)

Referenced above as binding parameters, not supplied, and deliberately **not invented here**. Each
must be fixed and pre-registered before a conforming registration starts.

| # | Value | Step | Notes / candidate source |
|---|---|---|---|
| 1 | `LOCALITY_THRESHOLD` — minimum `local_fraction` for a single-node instance | 4 | Inherited TBD, `P-BENCH-PLACEMENT-1` §7.1. Tool default `0.85`; the 2026-07-30 salvage audit used `≥0.99`. One binding value needed. |
| 2 | `INTERLEAVE_TOLERANCE` — allowed deviation from `1/n_nodes` per node | 4 | Inherited TBD, `P-BENCH-PLACEMENT-1` §7.2. |
| 3 | `ACHIEVED_CONCURRENCY_FLOOR` — minimum achieved/nominal for a reportable rung | 4, 5 | Inherited TBD, `P-BENCH-PLACEMENT-1` §7.3. |
| 4 | Prompt-set size / closed-loop arrival design that holds occupancy at `T` | 5 | Inherited TBD, `P-BENCH-PLACEMENT-1` §7.4. |
| 5 | Anchor value + band + `n` for `worker_general`, `architect_general`, `ingest_long_context` | 3 | Only the `frontdoor` anchor exists (median 35.7 tok/s, band 35–40, `n=154`). The 2026-07-30 `prodopt_results.txt` medians are **candidates at `n=3`, observation-grade**, and none has an independent reproduction. |
| 6 | Anchor **band width policy** — how wide a band is admissible, as an absolute or a % of the anchor | 3 | The frontdoor band is stated as a pair, with no rule for deriving one. A new model cannot establish an anchor without it. |
| 7 | `STARVATION_THRESHOLD` — the fraction of the cell median below which a stream counts as starved | 7 | Reference implementation used 25% of cell median as a *descriptive* label. The binding value is unset. |
| 8 | `MAX_STARVED_FRACTION` and `MIN_P10_OVER_P50` — the PASS conditions for Step 7 | 7 | The operator's requirement names a stated threshold and a p10 figure; neither value is fixed. Measured context: 25–29% starved at 8k on gemma. |
| 9 | Context ladder rungs above 32k, per role | 6 | Standing minimum is 0.5k / 8k / 32k. Roles serving wider windows need stated rungs. |
| 10 | GPU-lane proxy **fidelity** — a real ROCm llama-server rather than a synthetic proxy | 8 | ~~not specified in a committed script~~ **CLOSED as to specification**: both proxies are committed — `data/numa_placement/20260730-P-BENCH-PLACEMENT-1/gpuoverlap.sh` (spin, SMT-pressure only) and `gpuoverlap2.sh` (DRAM streaming). They deliberately BRACKET rather than predict: spin gave 0% on both shapes, bandwidth gave −34% full / 0% half. What remains open is fidelity — no ROCm build exists on disk (`build-v8-hip/bin/llama-server` is a 17 KB stub), so the real figure has never been taken. |
| 11 | ~~Attestation reference for the co-residency figures~~ **CLOSED** | 8 | `data/numa_placement/20260730-P-BENCH-PLACEMENT-1/gpuoverlap_results.txt` and `gpuoverlap2_results.txt`, committed 2026-07-30 at `7ace20a7`, each beside its script. |
| 12 | Registration sign-off authority — who accepts a completed `REGISTRATION.md`, and whether a second reviewer is required | 9 | The reviewer checklist exists; the role that runs it does not. |
| 13 | `local_fraction` for the 2026-07-30 `prodopt`/`shapes_prodopt` reference figures | 2, 5 | Those runs used the canonical placement and `--no-mmap` but did not capture `numa_maps`, so the field is unfilled in the worked examples above. |

### Open defects found while writing this runbook, noted not fixed here

**1. Protocol status block is stale.** `MEASUREMENT.md` §2 and annex B both list
`P-BENCH-PLACEMENT-1` as **✅ ratified 2026-07-30**, while the header of
`numa-placement-measurement-protocol.md` still reads **STAGED — not yet ratified**. The
constitution wins, so the protocol is ratified and this runbook treats it as such; the protocol
document's own status block needs a human-authored correction. Until it is corrected, a reader who
starts from the protocol file will incorrectly conclude that no placement number can gate a
decision.

**2. Registry geometry disagrees with the GGUF header.** `ingest_long_context` records
`attention_layers: 32`; the Q4_K_M GGUF header reports `block_count = 48` with a scalar
`head_count_kv = 2`, i.e. KV on all 48 layers. Every capacity figure derived from the registry
field is understated by 1.5×. Step 0's rule (header outranks registry) prevents the error
propagating into new registrations, but the registry row itself still needs repair, and any
existing sizing decision that used it should be re-derived.
