# `autokernel` — the AutoKernel research-loop substrate

Owning design: [`/workspace/handoffs/active/autokernel-research-loop.md`](/workspace/handoffs/active/autokernel-research-loop.md)
(the `epyc-root` handoff). Section references below (`§5.8`, `invariant 7`, …)
are to that document.

AutoKernel is the campaign/control plane that proposes, builds, and measures
experimental kernel candidates against a frozen production anchor. **Its
release-side job ends at a *release package*.** A human executes every freeze
and cutover (§1.3, invariant 5); nothing in this package carries freeze or
cutover authority, and `schemas.find_authority_flavoured_keys()` refuses any
auth-flavoured key in a machine-authored record so the absence is *checked*, not
merely intended.

Nothing here writes to a production kernel tree, starts a process, runs a
benchmark, or runs inference.

---

## What is implemented (AK1 + AK2 + AK3 partial + AK4 + AK5/AK6 + AK8/AK9)

| Module | What it owns |
|---|---|
| `schemas.py` | The seven §7 record contracts (the evaluation event in **two live versions** — `v2` readable forever, `v3` current), canonical JSON/content hashing, the `PASS`/`FAIL`/`COULD_NOT_CHECK` `Check` type, and the record-level checkers (`check_anchor_binding`, `check_scope_denominator_admits_gate`, `check_metric_commensurability`). **Single source of truth: every other module is written against these and must not invent a record shape.** |
| `journal.py` | The append-only, fsynced, **sharded** primary record. Shard ordering, torn-tail repair, cursors, archiving, supersession (record-scope and retrieval-scope), tombstones, preflight attestations, derived views, and `check_view_consistency`. |
| `storage.py` | §3.7 durability classes, §5.8 retention classes and rule-bound tombstoned expiry, per-campaign quota and `DISK_PRESSURE`, the `data/<campaign>/` evidence root with `SHA256SUMS` + README, and `verify_durability`. |
| `resource/device_claim.py` | The cross-process **exclusive GPU device claim** (§2.6) — `flock(LOCK_EX)` on a never-unlinked lock file, PID+start-time+boot-id liveness, journaled crash reclamation, quiesce-and-drain revocation, and the claim receipt id that lands in every evaluation event. |
| `resource/preflight.py` | The **one** audited read-only inference preflight (§3.5): claim witness as the target instrument, an opt-in name-pattern enumerator as the labelled interim one, and an AST self-audit proving the module cannot deliver a signal. |
| `resource/claim_witness.py` | The seam between the three above: a conforming `GpuClaimReader` over the device claim, and resolution of an evaluation event's opaque `resource_claim_receipt` back to the claim that produced it. |
| `evaluator/api.py` | **AK3.** The typed evaluator interface and the only place a verdict is COMPUTED. Owns the protocol's eight preconditions, twelve void conditions, fourteen search-grade conjuncts, the record grammar, the tier state machine, and `E_PROCESS_CONSTRUCTION_IDS` — the bundle's construction registry-of-record. |
| `evaluator/integrity.py` | **AK3.** §8.5.1 source integrity: an ELF `.dynsym`/`.symtab` reader, an Itanium-ABI demangler, symbol/arity/registration/dispatch-predicate diffs, clean-build-from-snapshot provenance, unified-diff parsing against the change-class envelope, the mechanically derived `core_header` risk tier, the §10.6 complexity ceiling, repair-from-clean-parent, and the gates that bind all of it to the request's anchor, artifact and derived surface. |
| `evaluator/surface.py` | **AK3.** §6.4 affected-surface derivation from the build system's OWN dependency information (make/ninja depfiles, cmake link lines), the dispatch-trace stage, three-stage reconciliation (`derived ⊇ traced`), the actor's declaration as a SCORED prediction rather than a scope input, and §3.2 normalized-binary comparison. |
| `evaluator/correctness.py` | **AK3.** The seventeen T0 gates: backend-op units, exact-reference comparison, unseen boundary shapes, surface reconciliation, the no-fallback proof, state/rollback/teardown, sanitizers (ASAN/UBSAN, no core dumps), output coherence vs the anchor, determinism class, binary+linkage identity, and anti-reward-hacking. `CoherenceVerdict` is computed and cannot be stamped. |
| `evaluator/statistics.py` | **AK3.** The calibration block in its normative solve order — `φ`, `B_min`, the α budgets and their thresholds, the anchor-gate band — the anytime-valid e-process (two bundle-fixed constructions), the pre-committed stopping rule with bounded extension, the MDE, order control, the selection/confirmation split, and the reducer that produces a conforming `api.EffectEstimate`. |
| `evaluator/controls.py` | **AK3.** The five controls as hashed data (definitions AND predicates), the A/A cadence scheduler, the historical-win-replay declared contract with its normative unavailable branch and operator escalation, and the projection into `api.WindowAttestations`. |
| `evaluator/recipes.py` | **AK3.** The codified recipe constructors — every measurement argv is emitted by one, carrying its constructor id and content hash — for `test-backend-ops`, `test-quantize-perf` and `llama-bench`, CPU and GPU. |
| `controller/state_machine.py` | **AK4.** §8.1's explicit machine: 17 live states, 13 stop states, the declared edge table, journal-then-act transitions, the operator-control latch (invariant 19), §8.2 BOOTSTRAP with its consistency assertion and deliberate-rebase escape, §8.9 anchor re-verification, and §8.10's stop-evidence table. **Owns disposition — no model output decides a transition.** |
| `controller/guards.py` | **AK4.** The fifteen §8.10 guards and their precedence, the directive vocabulary, the operator decision package (§18 item 7), the accept-side control's three statuses, and `dispose()` — the reduction that turns a round's guard verdicts into one disposition. A STOP is validated against `check_stop_evidence` AT CONSTRUCTION. |
| `controller/context.py` | **AK4.** §6.1's planner and critic briefs: fifteen sections, every item CITED to a journal event, a bounded render, §8.3.1 roofline utilisation with both denominators, the §19.2 do-not-repeat surface, the §6.5 oracle registry, and the quarantine block imported content is rendered inside. Refuses to leak the confirmation stratum or planner narrative. |
| `controller/hypotheses.py` | **AK4.** §8.4.0 operator hypotheses: the operator-facing store, the append-only hypothesis ledger, still-open tracking, the resolution record, and `check_do_not_repeat`. Every origin enters at `design_prior` and **the gate cannot see who stated the hypothesis** (AK-D38). |
| `controller/planner.py` | **AK4.** The provider-agnostic planner adapter: prompt bundles, the quarantine fence, the response contract, `assemble_proposal` (the controller-owned fields a model may not write), `ReplayProvider` for invariant 11, and cost attribution. |
| `controller/critic.py` | **AK4.** §6.3's pre-run critic (ten questions, deterministic gates, revisions the critic owns and gates it cannot waive) and §8.8's post-run critic (classification reconciled against the raw gates, the durable lesson, the next experiment). Severity only ever goes UP. |
| `controller/selection.py` | **AK4.** §8.3's cost hierarchy and its receipted skips, §8.4's twenty-two rejection codes and the journaled skip, the §19.2 ledger match, the fingerprint blacklist, §8.4.1's HARVEST/EXPLORE phase decision with a derived yield calibration, and the arm-budget partition. |
| `controller/composition.py` | **AK4.** §8.9 champion maintenance: the frontier and its mechanism-diversity floor, lineage proposal, `compose_champion` (which re-measures the COMBINED candidate and cites no member evidence), the re-anchor plan, and the ANCHOR_MOVED supersession sweep that kills comparisons while preserving source and correctness. |
| `controller/oracles.py` | **AK4.** The §6.5 oracle registry, ONCE, at both granularities — the design's table row and the individual tree a port names — so the compiler renders ids the critic can gate on. |
| `controller/fingerprint.py` | **AK4.** The one identity a filtered proposal is journaled under (§8.4). Prose-free by construction, because a blacklist a reworder can walk around is not a blacklist. |
| `release/plan.py` | **AK5.** §10.1's release-plan compiler: the join from source tree → backends → stable production paths → live roles → distinct models/quants/contexts/KV/speculation/concurrency/placement/co-residency at the production-optimal recipe, plus the §3.2 per-backend unchanged test consumed as a RECORD (never re-implemented), §10.5 incumbent evidence, §10.6 diff-complexity ceilings, and the linkage requirement per tree. Every narrowing leaves a receipt; a role it cannot plan becomes an `UnplannableRole` rather than disappearing. |
| `release/readiness.py` | **AK5.** The §9.7 T2 readiness ESTIMATOR — an advisory signal, `is_trigger = False` by class. Per-backend, per-phase, each phase under its own protocol; matrix coverage, capacity deltas, mechanism confirmation, the §1.6 phase-trade exception, and the `+25%/+20%` reference comparison **reported and never branched on**. `composite_readiness()` and `freeze_eligibility()` RAISE: a scalar folding two protocols cannot gate, and readiness is not eligibility. |
| `release/t3.py` | **AK5.** The §10.2 release gate — nine phases in order (identity preflight incl. §3.2, build+linkage, backend correctness, the performance matrix, quality, stability, capacity/utility incl. the §1.6 objective, the transaction **dry run**, the seal), the §10.4 waiver verifier (hash-pinned, predicate-checked, scope resolved from the operator's own document), §9.1 rerun idempotence and cooldown, the seven-component sealed bundle, and PASS / FAIL / PASS_WITH_WAIVER. `T3Runner` implements AK4's `ReleaseGateRunner` seam. `calibration_request()` replays the preserved v8 and speech freezes as dry runs. |
| `release/packager.py` | **AK6.** The §11.2 release PACKAGE a human executes: AK7's operator freeze request, `seal_champion`'s six refusals, the trusted-evaluator seam, the next version/branch/tag, the §10.5 archive + rollback plan, the drafted era rows and AutoPilot rebaseline note, the statically pre-validated operator command sequence, the §11.3 cutover ASK as a `session_bus` message, the §11.5 watch window, and the operator decision package. Twelve §11.2 "may not" doors that raise. |
| `adapters/serving_runtime.py` | **AK8.** §13.5 — the backend that does **not** travel the kernel-freeze path. Owns the §11.6 three-gate stack-change package (registry guard → linkage/identity → live-vs-intended config), `task_rate` as the only admissible objective, and `refuse_kernel_freeze()`, which raises rather than degrading to a freeze-shaped package with empty kernel fields. |
| `adapters/whisper_stt.py` | **AK9.** §13.3 — the STT backend: its frozen tree and binary inventory, the ggml-linkage verifier contract, the WER/RTF corpus and exclusion accounting, op-shape coverage, `assess_complexity` traced from the DIFF rather than a declared label, and `release_gate_readiness()`, which returns COULD_NOT_CHECK — never PASS — while `P-STT-*` remains a draft. |
| `adapters/qwentts_tts.py` | **AK9.** §13.4 — the TTS backend: pipeline stages, stage attribution, numerical-safety and roundtrip-WER checks (measured THROUGH the frozen production STT binary, never through the champion), corpus identity, and the same draft-protocol release refusal. |

### AK3 — what the evaluator delivers

It replaces `scripts/kernel_rnd/kernel_eval.sh`, which is fenced off and exits 2.
That script's three defects are the three structural properties here:

1. **It stamped a literal** (`"status":"OK"` in a `printf`). A `Verdict` is
   constructible only by `compute_verdict()`, and `Verdict.__post_init__`
   re-derives `status`, `effect_resolution`, `speed_rank_admissible`,
   `integrity_flags` and `derivation` from the evidence on the same object,
   raising `VerdictTampering` on any disagreement.
2. **It reported coherence with no anchor** (`COH="coherent"` for any non-empty
   generation). An absent, incomplete or mutated anchor is a `VoidFinding`, the
   verdict is `INVALID`, and every gate declaring `requires_anchor=True` has its
   PASS demoted to `COULD_NOT_CHECK` before the status is derived.
3. **It let speed be reported beside a correctness problem.** `rank_key()` raises
   `SpeedRankUnavailable` rather than returning a penalised rank, and
   `rank_candidates()` returns `(ranked, unrankable)` so a withheld rank is a
   journaled reason, never a shorter list.

Governing instrument:
[`/workspace/measurement/protocols/kernel-research.md`](/workspace/measurement/protocols/kernel-research.md)
(Annex K, **P-AK-SEARCH-1**, RATIFIED 2026-08-03). It emits a **search record,
not a claim**, and T3/T4 are refused by name — they are release instruments
outside the protocol's scope, owned by AK5 through the `ReleaseTierEvaluator`
seam.

`evaluator/test_conformance.py` walks that protocol's own sections and asserts
one test per obligation. Its coverage test fails if a registered obligation has
no claiming test, if a test claims an unregistered obligation, or if a claiming
test asserts nothing — so deleting a test is not a way to make it pass.
Obligations whose subject is a real measurement are registered in `SEAM_ONLY`
with the reason they cannot be asserted directly: a **declared** state, never a
silent skip.

### AK4 — what the controller delivers

AK1–AK3 built the substrate; **AK4 is the half that WALKS the loop.** It is also
the half where the project's two most expensive AutoPilot scars live, and each is
answered structurally rather than by a check somebody remembers to call:

- **A control that was requested but never verified.** AutoPilot's pause was a
  silent no-op for months because the run state was cached in memory and written
  back over the operator's change. Here the latch is a FILE, re-read from disk at
  the top of every iteration under the journal write lock; no object holds a
  `ControlLatch` as attribute state; `ControllerStateMachine.__slots__` has no
  slot for one, so a future edit that tries to cache it fails at runtime; and
  `audit_no_cached_control_state()` proves it from the object rather than from a
  comment. A halt survives restart, and an ack without its latch — the
  crash-between-ack-and-latch window — is a HARD failure rather than "no control
  pending".
- **A restart that came up empty with nothing objecting.** 232 trials and ~16
  days of compute vanished because a rebuilt derived view disagreed with the
  record and nothing refused to start. §8.2 step 10 refuses, and the deliberate
  rebase is an explicit escape that lands its reason in the journal, so an
  intentional wipe is never indistinguishable from the loss.

The authority rule the whole plane rests on: **the LLM proposes and interprets; a
deterministic controller disposes every gate and stop condition.** `check_stop_
evidence` takes no `origin` parameter and `check_do_not_repeat` takes no author,
so there is no argument along which trust could be extended; a critic disposition
can only make a proposal's fate WORSE; and an operator's stop request meets the
same evidence table as an internally generated one. `test_ak4_conformance.py`
asserts this per obligation — the five things P-AK-SEARCH-1 authorizes, the nine
it denies, its eight preconditions, and §4's twenty invariants, one test each.
Obligations that cannot be checked without taking a measurement are registered in
that file's `SEAM_ONLY` table with the reason and the owner, never skipped: a
skipped test is silent, a registered seam is a list an auditor can count.

### Guarantees the package as a whole makes

1. **Nothing is lost by crashing.** Every append is fsynced before it returns; a
   torn tail is discarded loudly with its own `TORN_APPEND_DISCARDED` event;
   `read_all()` raises on any unreadable line rather than returning a partial
   history that looks exactly like a complete one. `test_integration` rebuilds
   the whole campaign from the journal alone and asserts the digest matches.
2. **Inability to evaluate is a THIRD outcome.** Every checker returns
   `PASS` / `FAIL` / `COULD_NOT_CHECK`. `PreflightResult.__bool__` raises, so
   `if preflight(...)` is a stack trace rather than a silent misreading.
3. **Evidence is never evicted; only the `expirable` class expires**, and only
   with a tombstone in the journal *before* the bytes go.
4. **A live claim holder is never stolen from.** `unknown` liveness raises and
   journals a defect; revocation is quiesce-and-drain, never a signal.
5. **The record and the retrieval view are different objects.** `read_all()`
   keeps a withdrawn belief and its prose forever; `retrieve()` withholds it, and
   citing it back in raises (§5.5 items 6/7, invariant 20).
6. **Explicit failure over silent fallback, everywhere.** There is no no-op
   journal, no default sink, no fail-open reader; a missing dependency or an
   unreadable file raises.

---

### AK5/AK6/AK8/AK9 — what the release plane delivers

**The cardinal rule: AutoKernel never freezes and never cuts over.** It produces a
release PACKAGE that a human executes. A kernel freeze crosses four human-only
trust boundaries (`MEASUREMENT.md:140-142` — the freeze, the era-registry rows,
the AutoPilot baseline apply, and the pinned human-only path list), so there is no
such authority here to hold, delegate, or flag. Every one of the twelve §11.2 "may
not" capabilities is a function whose entire body is a `raise`, and
`packager.audit_refusal_doors_raise_unconditionally()` walks the list and FAILs on
a door that grew an `if`.

That rule is *proved*, not documented, in four independent ways:

- each module's own AST self-audit, anchored to its own module identity so the
  guarantee is not obtainable by handing the auditor a different string;
- `release/test_release_integration.py::TestNoProductionWritePathsAnywhere`,
  which parses **every** module under `release/` and `adapters/` — enumerated from
  the filesystem, so a module added tomorrow is covered by default — and proves
  no write/spawn/signal call exists in any non-test module and that no call
  anywhere names a production branch, a stable kernel path, `instrument_eras.yaml`
  or `autopilot_baseline.yaml`. It parses calls rather than grepping text,
  because both directories name all four targets in nearly every docstring;
- `t3.TransactionPlan` refuses construction with `executed=True`, and the §10.2
  phase-8 transaction is a DRY RUN by type;
- `schemas.find_authority_flavoured_keys()` over the sealed bundle and over the
  package's own record, enforced at construction.

The release path is not the same for every backend, and that asymmetry is
load-bearing (AK-D9/AK-D23): `llama_cpu`/`llama_gpu`/`whisper_stt`/`qwentts_tts`
release through a kernel freeze; `serving_runtime` travels the §11.6 three-gate
stack-change path and is refused **by name at all four release-plane doors** —
`plan.ReleaseTarget`, `t3.ReleasePlanView`, `packager.OperatorFreezeRequest` and
`readiness.ObjectiveSpec`.

§10.4's calibration is wired and passing: the T3 dry run against the REAL
`artifacts/operator/ratify_v8_final_freeze_20260725.json` predicts **FAIL without
the waiver**, naming both Q8 pairs; with the waiver and a reconstructed N-1
archive it is `PASS_WITH_WAIVER` with exactly those two claims suppressed by name
and the forfeit recorded. The waiver alone never clears the integrity spine.

## Integration seams reconciled — AK5/AK6/AK8/AK9 release plane (2026-08-03)

Same pattern as AK3 and AK4, and the same lesson: `plan.py`, `readiness.py`,
`t3.py`, `packager.py` and the three adapters were each written and red-teamed
alone, each suite was green, and seven disagreements lived in the gaps. Every one
of them passed both of the involved modules' own suites first, because each module
was right *about itself*. `release/test_release_integration.py` is where they stay
fixed.

1. **A §3.2 drop verdict `plan.py` refuses and `t3.py` accepted.**
   `plan.drop_verdict_contradictions()` re-derives `may_drop_cells` from stage 1,
   stage 2, agreement, transfer scope and findings — because the boolean is a
   plain FIELD and it deletes a backend's whole matrix. `t3.UnchangedView` READ
   it, and guarded only `unchanged_outcome`. A view with `agreement=FAIL` and no
   stage 2 was accepted and dropped the backend. `unchanged_view()` accepts a
   hand-built view and a bare mapping by design, so the compiler is not the only
   door; the same conditions now hold at both, as `UnchangedView.drop_contradictions()`.
2. **§1.6's conjunction was adjudicated over standings nothing joined to the
   matrix.** `phase_performance_matrix` measures cells; `phase_capacity_utility`
   decides the whole per-phase objective from caller-supplied `PhaseStanding`
   records. Nothing connected them: a standing with `cell_ids=()`, or citing a
   correctness cell, or a diagnostic cell, or a cell of another phase, satisfied
   it alone — and deleting every prefill CELL while keeping the prefill STANDING
   left the release gate at **PASS**. A standing must now name at least one
   planned, gating, production-optimal `performance_matrix` cell of its own
   (backend, phase) that this run recorded a result for.
3. **A conjunct could be deleted rather than failed.** `readiness.py`'s red-team
   closed exactly this on the advisory signal (an objective naming only `decode`
   reached `objective_met`); the release gate had the identical hole in
   `phase_protocols`, where it decides a freeze. Both now read
   `schemas.PHASES_BY_BACKEND`, the SSOT `plan.BackendBinding` already held itself
   to.
4. **`serving_runtime` was refused at three doors and admitted at the fourth.**
   It is in `schemas.BACKENDS` but absent from both `SOURCE_TREE_BY_BACKEND` and
   `PHASES_BY_BACKEND`, so in `readiness.py` the champion-lineage check and the
   §1.6 conjunction check each degraded to a **no-op** on it while the signal
   still rendered as a kernel backend's — and the readiness line is what a freeze
   request cites.
5. **A FAILing run licensed claims.** `compute_verdict` built
   `ReleaseReceipt.claims` from every passing gating cell regardless of verdict,
   and AK6 renders `len(receipt.claims)` onto the operator's first page as *"claims
   licensed: N"* and copies the receipt into the durable package record. A release
   that did not pass licenses nothing; the cells that did pass are kept, in
   `withheld_claims`.
6. **A `CellResult` could relabel the plan's cell.** `T3Request` cross-checked
   only that the result's `cell_id` was in the plan, so every other facet was the
   measured party's to restate. Flipping `co_resident` True satisfies the §10.2
   phase 4 `llama_cpu` co-residency requirement with a run that was never
   co-resident; flipping it False deletes the only cell class that measures the
   machine the way production runs it. §12 derives scope MECHANICALLY, so the
   scope facets must now agree. `reps` is deliberately exempt — the compiler runs
   before anything is measured.
7. **The machine-actor vocabulary and the waiver verifier were in different
   modules.** AK6 guards five human-authority identity fields with
   `MACHINE_ACTOR_TOKENS`; `t3.verify_waiver` accepted any non-empty
   `authorized_by`, so a waiver attributed to `autokernel` verified as
   human-attested and turned FAIL into PASS_WITH_WAIVER. The import direction is
   packager → t3, so the check lands in the packager rather than forking the token
   set downward — a self-granted waiver cannot reach a package a human executes.

Two more, found in the same pass and outside the release plane proper:

8. **`release/test_plan.py` and `test_readiness.py` loaded a SECOND copy of
   `schemas` and `surface`.** The flat `sys.path.insert` + `from autokernel import …`
   idiom binds to a different module object than the rest of the package uses
   under `unittest discover -t .`, so every `isinstance` guard across that boundary
   fails silently — `compile_release_plan` would refuse a genuine
   `surface.BackendUnchangedResult` for being the other copy's. The README already
   forbade the idiom; both release-plane suites now use relative imports.
9. **`adapters/serving_runtime.py`'s self-audit was the one of three not anchored
   to its own module.** Its two siblings bind the audited AST to their own
   `BACKEND` and checker names; this one returned PASS for any clean text,
   including a sibling adapter's source.

One asymmetry was **pinned rather than "fixed"**: `t3.Cell` admits a `None`
`workload_phase` while `readiness.T2Cell` requires one, because correctness,
quality, stability and capacity cells are not throughput cells and have no phase
to be non-inferior in. The performance-matrix cells that DO carry one are checked
against `PHASES_BY_BACKEND` at the compiler and at the objective.

## Integration seams reconciled — AK4 controller (2026-08-03)

Six AK4 modules were built in parallel against one state machine. Each was green
on its own; a suite of individually-green modules is precisely the shape in which
a seam defect survives, because every module is consistent with itself and the
disagreement lives in the gap. The integration pass found six, and
`controller/test_loop_integration.py` is where they stay fixed.

1. **Two `proposal_fingerprint` implementations writing one journal field.** The
   planner adapter hashed `change.conceptual_change` — free prose — and the
   screener did not. Both wrote `PROPOSAL_SKIPPED.payload["fingerprint"]`, and
   `read_skip_history()` counts them in ONE dict against a threshold of two, so
   two skips of one concept counted 1 + 1: §8.4's auto-blacklist never fired and
   §8.10's degradation run was computed over a key the record did not use. Now
   one algorithm in `controller/fingerprint.py`, and it is the PROSE-FREE one —
   the planner-side test that asserted rewording minted a new fingerprint was
   asserting attempt 119 looking novel, and has been inverted.
2. **Two §6.5 oracle registries sharing ONE id out of nineteen.** The compiler
   rendered `upstream llama.cpp / ggml` into the planner brief; the critic gated
   on `llama.cpp_upstream` and rejected the citation as *"not in the declared
   registry"* — a refusal that blamed the planner for the controller's own
   disagreement. `controller/oracles.py` now holds the table once, at both
   granularities (the §6.5 row and the tree a port names), and both consumers
   derive.
3. **Two harvest-class vocabularies.** The critic had three classes and the
   compiler four, so §6.5's own FlashAttention/FlashInfer row (`conditional`)
   was inexpressible in the plane that gates it.
4. **Two hypothesis-origin vocabularies.** `hypotheses` opens at `controller` and
   `import`; `context` accepted neither, and offered a `record` origin the store
   cannot produce. A controller-opened hypothesis raised `ContextInputError` on
   its way into the very brief §8.4.0 requires it to appear in.
5. **A reserved closure word the DISPOSER did not know.** `guards` refused
   "exhausted"; `state_machine.check_stop_evidence` did not — and `stop()` and
   `dispose_stop_request()` are both public, so a stop that never met a guard
   reached the record on the word §8.10 names first. The machine now owns the
   vocabulary AND the matcher, `guards` compiles nothing of its own, and the scan
   covers the enumeration as well as the reason.
6. **Three budget units and no converter.** §7.1's manifest declares HOURS, a
   §7.2 proposal declares MINUTES, and `context.reduce_budget_ledger()`
   accumulates SECONDS. The obvious wiring makes the budget gate 60x too
   permissive, in the direction that overspends.
   `selection.budget_remaining_from_caps()` is now the only sanctioned crossing,
   and a missing cap raises rather than defaulting.

One apparent disagreement was **deliberate and has been pinned rather than
"fixed"**: `SUPERSEDED_FACT` rejects in `selection` and is advisory in
`hypotheses`, because §19.2 says *"do not execute the stale PROPOSAL; regenerate
from current source"* — it closes the proposal, not the question. Making the two
equal would close research the design keeps open, so a test asserts the asymmetry
and says why.

## Integration seams reconciled — AK3 evaluator (2026-08-03)

The seven evaluator modules were written in parallel against one interface. Each
passed its own suite and its own red-team pass; every defect below lived
*between* two of them, where each module was individually correct and the two
descriptions of the same object did not match.
`evaluator/test_integration.py` is the regression barrier for these; each row
was verified non-vacuous by reverting the fix in an isolated copy and confirming
the suite fails.

| Seam | Disagreement | Resolution |
|---|---|---|
| statistics → api → schemas | `PairedBlockReducer` set `EffectEstimate.raw_samples` to nested **tuples** (the estimate is a frozen, hashable dataclass); `schemas.canonical_json` **refuses tuples**. Every reduction the reducer actually produced raised `TypeError` out of `content_hash(event)` — the record could not be hashed, journaled, or emitted. Neither unit suite saw it: each fixture used the shape its own module wanted. | `api._canonicalizable()` converts tuple→list recursively for the event's `performance.raw_samples`, and RAISES (naming the path) on anything else `schemas` cannot represent rather than stringifying it. |
| controls → api | `ControlPanelResult.definitions_check` — the control **definitions and predicate** tamper digest — had no field in `api.WindowAttestations`, so *"any post-hoc change to … the control definitions"* (What voids a run) could not reach `check_void_conditions` at all. A campaign could rebind a control predicate and every record in the window still read as clean. | New required `WindowAttestations.control_definitions_immutable`, folded into `VOID_POST_HOC_RULE_CHANGE` and into search-grade conjunct 8, plus `controls.window_control_attestations()` — the projection that fills it, mirroring `statistics.BlockReduction.window_checks`. |
| api ↔ statistics | `CalibrationOutputs.e_process_construction_id` accepted any non-empty string while `statistics.CONSTRUCTIONS` is the bundle's registry. Three suites' fixtures named `paired_betting_supermartingale/v1`, which no reducer implements — a recorded selection nothing can reproduce. | `api.E_PROCESS_CONSTRUCTION_IDS` is the registry-of-record and `CalibrationOutputs` refuses an id outside it; `statistics.py` asserts at **import time** that its own registry has exactly those ids, so drift is an `ImportError` rather than an unreproducible record. |
| surface → integrity | `SourceIntegrityInputs.declared_surface_scope` is a caller-supplied `"full_tree"`/`"partial"` string and the core-header tier FAILs on an under-declaration — while `surface.AffectedSurface.full_tree` is the **derived** answer to the same question, and nothing compared them. A caller who derived `full_tree=True` and typed `"partial"` got a PASS. | `integrity.surface_scope_for()` projects the derived manifest into the scope string, `check_declared_surface_scope()` compares them (COULD_NOT_CHECK when unbound), and `SourceIntegrityGateRunner(derived_surfaces=…)` emits `integrity.declared_surface_scope_binding`. The binding is a declared capability — `surface_binding` says whether the runner has it, and a registered-but-missing manifest raises rather than falling back. |
| api ↔ any runner | A gate runner returning **zero** gates derived to `status: pass`: `_derive` walks the gate list and an empty list contributes nothing to worsen. `TierDispatcher` guarded the *unregistered*-tier case for exactly this reason and left the empty-return case open. | `TierDispatcher.dispatch` raises `EvaluatorNotWired` on an empty gate sequence: a tier that produced no findings because it ran nothing is not a tier that found nothing. |
| api ↔ statistics (record) | `render_search_record_grammar` prints `request.metric` and `request.metric_direction` beside `effect.value`, and nothing compared them. An estimate of a *different* quantity rendered under this cell's metric name and was fully search-grade. | `check_record_grammar_complete` FAILs on a metric or direction mismatch, citing `MEASUREMENT.md:25-26` — the grammar's head is one triple. |
| api ↔ calibration | `EffectEstimate.noise_floor` need not be the cell's calibrated `φ`, and `_resolve_effect` reads the floor **on the record**. Zeroing it turned a sub-floor estimate into a rankable `improvement`, defeating *"an estimate whose magnitude does not exceed φ MUST NOT be ranked"*. | `evaluate_search_grade` FAILs `calibration_block_accepted` when the record's floor is not the cell's calibrated `φ`. |
| api ↔ schemas ↔ journal (**a voided run could not be journaled**) | *"A voided run is journaled as `INVALID` with its reason, and is **never silently discarded**."* `evaluation_event.v2` required `anchor.binary_sha256`/`linkage_sha256` unconditionally and `Journal.append` validates before it writes — so the ANCHOR-MISSING void, the one case where there is no digest to record, could produce **no valid record at all**. `build_evaluation_event` was right to raise rather than invent a digest, and the run survived only as `durable_payload`, which is not the primary record. | `evaluation_event.v3` permits `anchor` to be **structurally absent**, and only when `status == "invalid"` **and** `integrity_flags` names an anchor void reason (`schemas.ANCHOR_VOID_REASONS`, checked against `api.VOID_REASONS` in `test_conformance`). `schemas.is_placeholder_digest` refuses `0`*64 and friends in an anchor, so the exemption cannot be taken by faking one instead. `AnchorMissing` now fires only for an anchor **claimed** and unreadable/fabricated. |
| schemas ↔ precondition 4 | Precondition 4 names the anchor by source commit **and** binary SHA-256 **and** linkage SHA-256. `evaluation_event.v2.anchor` carried two of the three, so `source_commit` rode along as an unvalidated extra key in a free-form block — present in practice, checked by nothing. | `v3.anchor.source_commit` is REQUIRED and validated by the same `_need_commit` helper as `candidate.worktree.source_commit`. |

## Integration seams reconciled — AK1 + AK2 substrate (2026-08-03)

The five substrate modules were built in parallel against the same design. Each
passed its own suite; the defects were all *between* them.
`autokernel/test_integration.py` is the regression barrier for these.

| Seam | Disagreement | Resolution |
|---|---|---|
| journal ↔ storage | `storage.tombstone_id` identifies a reclamation by (campaign, path, sha256, kind, rule); the journal's receipt view was keyed by **the content hash alone**. Two byte-identical build trees at two paths were two reclamations there and **one** slot here, and `check_view_consistency` said PASS because it recounted by the same key. | `journal.tombstone_view_key()` keys on (hash, path); `path` is now REQUIRED by the native TOMBSTONE validator; the checker adds a second, independently keyed recount over distinct `tombstone_id`s. |
| storage → journal | The sink handed the journal a record carrying `epyc.autokernel.artifact_tombstone.v1` that `validate_artifact_tombstone` had never seen — `plan_expiry` validates only the `intent` record, and the `reclaimed`/`failed` variants are built afterwards. | `JournalTombstoneSink.append` validates every record before it goes. |
| preflight → journal | `require_no_concurrent_inference` hands the attestation back on the exception "so the caller can journal it" (invariant 7) — and the journal's **closed** kind vocabulary had no kind to journal it as. The instruction was unfollowable. | New native kind `PREFLIGHT_ATTESTATION` + `Journal.append_preflight_attestation()`. Its validator mirrors `PreflightResult.__post_init__`, so the durable record cannot say less than the object it came from. |
| device_claim → preflight | `preflight` needs a `GpuClaimReader`; `device_claim` produces `ClaimReceipt`s; nothing bridged them, so **every GPU-scoped preflight was `COULD_NOT_CHECK` even with the claim held**, and preflight's refusal text still said the substrate "does not exist yet". The obvious hand-rolled bridge also passed `ClaimReceipt.holder_label: Optional[str]` into `GpuClaimWitness.holder_label: str`, rendering a FAIL finding's `whose` as the literal string `"None"`. | `claim_witness.device_claim_witness_reader()` / `gpu_claim_sources()`; `GpuClaimWitness.__post_init__` now refuses an unattributable witness. |
| device_claim ↔ schemas | `evaluation_event.resource_claim_receipt` is an opaque string, and a non-empty string is exactly what an *invented* receipt also is. `check_device_claim_held` needs `{claim_id, device_id}`; the event carries no device. Nothing downstream could resolve the binding. | `claim_witness.resolve_claim_receipt()` / `check_event_claim_receipt()` resolve the id against the claim journal, which already records the full receipt on every `claim_acquired`. |
| storage ↔ everything | `storage.py` did an unconditional import-time `sys.path.insert(0, <autokernel dir>)` and imported a **flat** `schemas`, so `storage.Check is schemas.Check` was `False` and every `isinstance(v, schemas.Check)` across the seam silently said no — and `import resource` anywhere later in the process resolved to `autokernel/resource/__init__.py` instead of the **stdlib** `resource`, the shadowing that package's own docstring forbids. | Package-relative import with a file-identity assertion, `sys.path` touched only on the genuine flat-import fallback. Asserted structurally from each module's AST, since the test files legitimately insert paths themselves. |

---

## The shared lock root, and the follow-up that belongs to `epyc-orchestrator`

`device_claim.py` writes `gpu_device.<device_id>.lock` into **the same on-disk
root** as the orchestrator's CPU region locks:

```
/mnt/raid0/llm/tmp/
├── cpu_region.frontdoor.q0.lock      # epyc-orchestrator/src/runtime/cpu_region_lock.py
├── cpu_region.GLOBAL.all.lock
├── gpu_device.mi210_0.lock           # autokernel/resource/device_claim.py
└── gpu_device.mi210_0.revoke.json
```

Resolution order is copied **exactly** from `cpu_region_lock._tmp_dir()`:
`ORCHESTRATOR_TMP_DIR`, then `ORCHESTRATOR_PATHS_TMP_DIR`, then the hard-coded
path. **There is deliberately no AutoKernel-specific override env var**: a
research-repo-only variable would let the two repositories resolve different
roots and silently stop excluding each other, which is the single failure the
whole module exists to prevent. Tests pass `lock_root=` explicitly instead.

Sharing the root is what makes an orchestrator process and a research process
exclude each other **without either importing the other's code** — the exclusion
fact is a kernel `flock`, not an agreement between two Python packages.

### Follow-up owned by whoever holds `epyc-orchestrator`

AK2's checklist requires *"extend `region_lock_cli.py` with a device verb, or add
a sibling CLI sharing its lock root; **do not fork the lock semantics**"*. That
is a change in the orchestrator repository and is **not** in scope here — this
package is the research repo's half. Three items are waiting on that side:

1. **A `--device` verb** on `region_lock_cli.py` (or a sibling CLI on the same
   root) that acquires/inspects/revokes through *these* semantics rather than
   reimplementing them.
2. **Scope or retire `src/gpu_lease.py` for cross-process use.** It is a
   `threading.Condition` lease: correct for intra-process ownership inside one
   orchestrator (`axa2_live_cutover_bundle.py:535`), and structurally unable to
   exclude a second process. Either label it intra-process-only or migrate that
   call site.
3. **An explicit mode/ownership statement for the lock root.** It is currently
   `0777` with lock files at `0o666 & ~umask`. Cross-repo exclusion depends on
   every participant being able to `O_RDWR` the file; a different uid with
   `umask 022` fails *closed* (good) but is then silently unable to participate
   (not good).

A fourth, smaller item lives on the research side and is recorded here so it is
not lost: `preflight.py` mirrors `cpu_region_lock`'s payload key set, and the
test that "pins" it is circular (it asserts a constant defined in the same test
file). Making it non-circular needs a cross-repo read and is a design call.

---

## Running the tests

Plain `unittest`; no pytest dependency, no network, no GPU, no inference. Every
suite is safe to run on the shared host at any time.

```bash
cd /mnt/raid0/llm/epyc-inference-research

# The whole package (this is the gate).
python3 -m unittest discover -s scripts/kernel_rnd/autokernel -t . -p 'test_*.py'

# Same, with the resource-leak screen the repo just fixed a class of bugs for.
python3 -W error::ResourceWarning -m unittest discover \
    -s scripts/kernel_rnd/autokernel -t . -p 'test_*.py'

# One suite.
python3 -m unittest scripts.kernel_rnd.autokernel.test_integration
python3 -m unittest scripts.kernel_rnd.autokernel.resource.test_claim_witness

# The cross-module suites — the ones that fail when two modules disagree.
python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_conformance.py
python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_integration.py
python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_ak4_conformance.py
python3 -m unittest scripts/kernel_rnd/autokernel/controller/test_loop_integration.py
python3 -m unittest scripts.kernel_rnd.autokernel.release.test_release_integration

# As a plain script.
python3 scripts/kernel_rnd/autokernel/test_integration.py
```

Expected: **4066 tests, OK (expected failures=1)**. The one `expectedFailure` is
`test_preflight.RealKernelLockEncodingTest.test_KNOWN_HOLE_unlinking_a_held_lock_
file_hides_its_live_holder` — a real, documented hole (unlinking a held lock file
hides its live holder from the `/proc/locks` witness), deliberately left visible
so the suite flips to a failure the moment someone closes it.

The package is also in `Makefile`'s `PYTEST_SMOKE`, so `make lint` and
`make test` cover it:

```bash
make lint     # ruff over PYTHON_SMOKE + PYTEST_SMOKE
make test     # uv run pytest -q $(PYTEST_SMOKE)
```

### Import convention

Import **through the package** (`from autokernel import journal`), not by putting
`autokernel/` or `autokernel/resource/` on `sys.path`. Both shortcuts load a
second copy of `schemas.py`, and the second one shadows the stdlib `resource`
module for the rest of the process. The existing per-module test files still use
the flat idiom for historical reasons; new code should not.

This is not stylistic. Under `unittest discover -t .` the package is already
imported as `scripts.kernel_rnd.autokernel`, so `from autokernel.evaluator import
surface` creates a **second module object with different classes**, and every
`isinstance` guard across that boundary returns False for an object that is
genuinely of that type. The release plane's two flat suites were converted for
exactly this reason (seam 8 above); the remaining ~23 are a live hazard for any
future cross-suite fixture reuse.

---

## What is NOT implemented

### Remaining in AK1

- **Bootstrap knowledge corpus (§19).** The event *kinds* exist and are
  validated (`LEGACY_SOURCE_DISCOVERED`, `PRIOR_ATOMIZED`, `SEED_COMPILED`, …)
  but nothing enumerates or content-hashes the historical source manifest,
  atomizes the source draft, or compiles the three derived memory products. No
  regime-matched retrieval proof for fixed planner/critic fixtures.
- **`kernel_store.py` quarantine.** Existing rows are not yet marked
  `legacy_unverified`, and `kernel_eval.sh` is not yet marked
  deprecated-and-unrunnable. Contaminated legacy rows can still reach a reader.
- **§19.3 suppression receipts** are validated on `RETRIEVAL_SUPERSEDED`, but
  nothing re-verifies them on an anchor move.
- **SQLite as a rebuildable view.** The journal is the source of truth and
  `rebuild_views()` folds it in memory; there is no SQLite projection and no
  rebuild command.
- **Failure / mechanism / do-not-repeat / context views.** `Views` covers
  campaigns, proposals, candidates, evaluations, champions, release packages,
  waivers, tombstones, the frontier and stop states. The planner-facing
  *failure*, *mechanism* and *do-not-repeat* views do not exist.
- **`Views` are record-scope, not retrieval-scope.** Slot payloads carry raw
  `narrative`, including for a `RETRIEVAL_SUPERSEDED` belief. The invariant-20
  boundary lives only on `Journal.retrieve()`; a consumer rendering views into a
  planner brief must apply `strip_narrative()` and
  `retrieval_superseded_event_ids` itself. **This is the highest-value remaining
  item in the journal** and needs a retrieval-scope view API.
- **`PREFLIGHT_ATTESTATION` is folded into no view.** It is in the record and the
  consistency checker ignores it, exactly like `OPERATOR_CONTROL_ACK` and
  `VIEW_REBASED`. AK3's evaluator is the consumer that will want the fold.
- **The §3.7 durability exposures are not cleared** — the np_context decision
  surface is still under `/mnt/raid0/llm/tmp/`, the two np_context study bundles
  are untracked, and the P2-5j protocol is not restored to git.
- **`check_evidence_durability.py` is not extended** to cover AutoKernel
  citations.
- **Validators not yet written:** stale production anchor, undeclared change
  class on a proposal (the field is validated; the *policy* is not), unbounded
  resource/storage request, missing fallback.
- **`kernel_store.py:88` file-handle warning** is not fixed, and pytest is not
  pinned in `pyproject.toml` / `uv.lock` (the Makefile still injects it with
  `--with pytest`).

### Remaining in AK2

- **Worktree managers and build layout.** Nothing creates
  `llama.cpp-ak-<campaign_id>` worktrees, namespaces `ak/<campaign_id>/…`
  branches, does pathspec-limited commits in the shared clone, or produces build
  identity receipts. Production source/build path denial exists only in
  `storage.plan_expiry` (for deletion), not for writes.
- **CPU region claim *acquisition*.** `preflight.read_region_claims` **reads**
  the region-lock namespace; nothing in this package acquires a CPU region
  claim. That is the orchestrator's `cpu_region_lock`, and invariant 9 requires
  the acquisition, not the read.
- **Co-residency policy integration.** `evaluation_event.co_residency` is
  validated; nothing decides it.
- **Owned cgroup / PID-receipt process scope with verified teardown.** Not
  started. `preflight.read_own_scope` enumerates the scope; nothing *creates*
  one.
- **Host-health / reboot-required / cache-preparation states**, the one-week
  uptime ceiling, and the §10.7 reboot decision package.
- **Session-bus registration** — roster id, heartbeat at every task boundary,
  outbox, lane declaration, revoke handling, C19/C20 visibility, and the
  re-read-instructions checkpoint.
- **`scripts/utils/agent_log.sh` wiring** and rollback-command logging.
- **C6 sandbox verification on the real host**, and its extension to *candidate
  binary execution* (the loop compiles code it authored and then runs it with
  GPU access on a shared host — §8.5.1).
- **`kernel_eval.sh`'s `gpu_idle()` is not yet deleted.** AK2's acceptance
  criterion is "deleted, not wrapped"; the replacement now exists, the deletion
  has not happened.
- **Resource starvation / drain / resume tests and campaign checkpointing.**

### Remaining in AK3

The evaluator is the machinery, not the run. Everything below is a real gap and
none of it is closed by the suites above.

- **Nothing has been executed.** No candidate has been built, no op suite has
  run, no microbench has been taken, no calibration block has been solved on real
  A/A material. Every number in every test is synthetic. AK3's own acceptance
  criterion — *"the first phase that consumes inference"* — is **not met**: the
  machinery exists, the campaign does not.
- **The runner seams are unimplemented on purpose.** `api.TierGateRunner` has
  three fixture implementations (`correctness.T0CorrectnessRunner`,
  `integrity.SourceIntegrityGateRunner`, `surface.SurfaceGateRunner`) and every
  one of them consumes evidence somebody else must produce.
  `correctness.T0EvidenceProvider` has exactly one implementation,
  `StaticEvidenceProvider`, which serves a dict. **Nothing builds a candidate,
  runs `test-backend-ops`, collects a dispatch trace, or takes a paired block.**
  `controls.ControlRunner` likewise has no implementation.
- **Two derivations of the §8.5.1 gates coexist.** `integrity.py` implements them
  over ELF tables, parsed diffs and build provenance; `correctness.py` carries
  three shallower `t0.source_integrity.*` gates over self-declared evidence
  objects. Composing both runners emits both (the ids do not collide, and
  `test_conformance` asserts that), and both are `integrity`-class so either
  failing blocks ranking — but a caller wiring ONLY the correctness runner gets
  the shallow three and nothing says the deep set never ran. Converging them is
  an AK4 decision, not a local edit.
- **`derived ⊇ traced` is unsatisfiable against a full-suite trace.** `surface.py`
  compares the whole traced op set against the candidate's derived surface, so a
  T0 trace that also dispatched unrelated ops reads as an escape. Either the
  trace must be pre-filtered to the derived surface or the symbol axes must
  compare only *changed* dispatch; nothing states or enforces either today.
  `TracedSurface` also records no denominator of what the trace covered, which
  the record grammar's `scope=` requires.
- **Precondition 6 reaches one argv surface of about six.** Only
  `correctness.SanitizerInvocation` carries an `api.RecipeReceipt`;
  `OpSuiteEvidence`, `CoherenceEvidence`, `DeterminismEvidence`,
  `BoundaryShapeEvidence`, `LinkageEvidence` and `StateSafetyEvidence` carry an
  opaque `receipt_ref: str`. **A hand-typed `test-backend-ops` argv is
  undetectable at T0.** `recipes.py` also exposes no `verify_receipt()`, so
  `api.check_preconditions` passes precondition 6 on the mere *presence* of a
  `RecipeReceipt` — a frozen dataclass of three well-shaped strings.
- **Three §8.5.1 gates cannot detect a self-report.** `BuildProvenance` and
  `DiffPolicyEvidence` have no `produced_by` field, so `build_dir_was_fresh`,
  `incremental_objects_present` and `commit_was_pathspec_limited` are taken on
  faith. Every other evidence type has the field.
- **CLOSED 2026-08-03 — precondition 4 is enforced on all three components in
  `correctness`, and every piece of evidence carrying anchor-derived material now
  names the anchor it was captured against.** Five evidence types —
  `LinkageEvidence`, `CoherenceEvidence`, `DeterminismEvidence`,
  `StaticAnalysisEvidence`, `AntiRewardHackingEvidence` — each carry
  `anchor_source_commit` + `anchor_binary_sha256` + `anchor_linkage_sha256` under
  ONE rule (`_validate_anchor_triple`): all three or none — a partially named
  anchor resolves to more than one artifact — and no placeholders
  (`schemas.is_placeholder_digest`), which is the same rule
  `evaluation_event.v3` applies to the record these feed. The linkage gate
  compares the commit as well as the two digests, so an anchor rebuilt from a
  different source at an identical digest is visible. Every consumer compares the
  capture's recorded anchor against the anchor it is handed, using
  `api.AnchorIdentity.identity_matches()` (which therefore now has callers in
  `correctness.py`), and RAISES an `EvidenceAnchorMismatch` subclass —
  `CoherenceAnchorMismatch`, `DeterminismAnchorMismatch`,
  `StaticAnalysisAnchorMismatch`, `AntiRewardHackingAnchorMismatch` — rather than
  downgrading: invariant 11 makes re-scoring saved outputs the normal path, so
  that is exactly where a capture taken against anchor A gets scored against
  anchor B. Evidence that recorded NO identity is COULD_NOT_CHECK on every
  surface, never an implicit match, and never a downgraded FAIL either.
  The live consequence is closed with it: `check_output_coherence`'s
  reconciliation of the anchor's determinism class now requires the coherence
  capture and `DeterminismEvidence` to name the SAME anchor before their
  agreement counts as corroboration — a mismatch raises, and an unrecorded
  determinism identity is COULD_NOT_CHECK, because two records agreeing does not
  establish they describe one anchor.
  **What remains here:** the recorded identity is still the producer's
  declaration — `produced_by` is checked, but nothing re-derives that a capture
  really ran against the anchor it names, so this binds an honest producer's
  replay and not a lying one's capture. `LinkageEvidence` reads its triple
  field-by-field (per-component reason texts) rather than through
  `recorded_anchor()`, so the linkage gate FAILs on a drifted anchor where the
  other four raise — the outcome differs by design, the binding does not.
  `integrity.check_evidence_binding` does bind the ELF tables, and registration
  tables carry no provenance digest at all.
- **The §8.5.1 (5) repair cap is enforced by nothing.** No gate consults a
  `RepairLedger` or `RepairPolicy`; `check_repair_from_clean_parent` accepts
  `attempt_index=99`, and `check_repair_from_clean_parent(None)` PASSes, which is
  exactly how a repair evades the clean-parent rule.
- **Control seed rotation is declared, hashed, and never enforced on the run
  path.** `derive_control_seed`, `ControlBundle.seed_for` and
  `SeedRotationSchedule.check_rotation` have no caller; `run_all` hands all five
  controls the same seed.
- **The calibration outputs are re-typable at the campaign boundary.**
  `CampaignStatistics` takes an `api.CalibrationOutputs`, not the
  `CalibrationSolve` that produced it, so a hand-built outputs object drives the
  reducer. The MDE resists (it is recomputed from the retained A/A pool); `φ`,
  `α_sel` and `α_conf` do not. `CalibrationAttempt.solve_order_recorded` is a
  dataclass *default*, so *"a conforming implementation MUST record that it did"*
  is satisfied by a literal.
- **`ControlPanelResult` has no mint token.** `api.Verdict` re-derives its own
  status and refuses a stamped one; a `ControlPanelResult` built by hand with an
  all-PASS panel yields `may_rank=True` with no control ever run.
- **Calibration recomputation at a campaign boundary and the A/A cadence driver
  are still nobody's.** `release/readiness.py` now computes the advisory signal
  and `controller/composition.py` composes a champion lineage, but nothing
  recomputes the calibration block at a boundary or drives the cadence.
- **T3 is refused HERE and implemented in AK5.** `api.admit_tier("T3")` still
  raises by name — that is the point — and `release/t3.T3Runner` is the
  `ReleaseTierEvaluator` that fills the seam. T4 remains unimplemented.

### Remaining in AK4

- **No runner.** AK4 is the plane that DECIDES; nothing yet drives a real
  campaign end to end against the real evaluator, holds a real device claim for
  a window, or calls a real model. `test_loop_integration.py` walks the whole
  loop against fakes, which proves the seams fit and proves nothing about a
  live host.
- **`stop_policy.max_consecutive_proposal_skips` has no declared home.**
  `selection.planner_health_stop_request()` requires it and rightly refuses to
  invent one, but `schemas.validate_campaign` does not name it, so a
  §7.1-conforming manifest can omit the single input `PLANNER_DEGRADED` needs
  and the loop discovers it by raising. The schema is AK1's;
  `test_ak4_conformance.TestDeclaredCampaignControls` is the record of the gap
  and FAILS the day it is closed.
- **`mechanism_class` on `composition.admit_to_frontier` is a caller parameter**
  bound to nothing but the planner-authored `change_class`, so the §8.9 diversity
  floor is decided by a label a model wrote. Closing it needs a signature change
  (resolve it from the proposal via `views`), which is a design call.
- **Several unbounded reads.** `ProposalScreener.screen` folds the whole journal
  per proposal (O(journal) per call, O(N²) per campaign) and
  `hypotheses.planner_round_block` re-surfaces every attempt of every hypothesis
  forever. `journal.read_since(reader_id)` exists and is unused. Bounding either
  needs a counted, receipted truncation — a silent slice would be exactly the
  discard §8.4 forbids.
- **The prose fields that are still primary keys.** `selection.mechanism` is
  free text and is both the ledger's key and part of the blacklist fingerprint;
  `hierarchy_layer` is self-declared, so the actor chooses how many §8.3 receipts
  it owes. Both need a campaign-declared vocabulary.
- **`context` and `planner` section vocabularies do not match** (nine shared, six
  and ten unshared). Harmless today because the compiler's bundle binds to the
  adapter directly by `manifest_sha256` rather than round-tripping through
  `ContextManifest` — asserted in `test_loop_integration` — but a caller that
  hand-builds a `ContextManifest` from compiler sections has no total mapping.

### Remaining in AK5/AK6 (the release plane)

- **No runner, and no live release has been rehearsed.** Everything below the
  gate is fixtures. Nothing has compiled a plan from the real
  `orchestration/derived/stack_priors.yaml`, sealed a real candidate, held a real
  compute window, or produced a package an operator has read. The chain is proved
  to FIT; nothing here is evidence about a real freeze.
- **`P-KERNEL-FREEZE-1` is a DRAFT.** `t3.phase_identity_preflight` raises
  `ReleaseProtocolNotRatified` for `mode="release"` under an unratified protocol,
  so **every** run this package can currently perform is a dry run. Ratification
  is human-only and is the gate on AK5 being usable at all.
- **The per-phase measurement protocols carry no ratification status.**
  `ProtocolBinding` proves the FREEZE protocol is ratified; the protocols the
  matrix cells are graded under (`P-BENCH-1`, `P-GPU-1`, `P-STT-*`, `P-TTS-*`)
  arrive as bare ids in `phase_protocols`. The adapters know — `whisper_stt` and
  `qwentts_tts` each expose `release_gate_readiness()`, which returns
  COULD_NOT_CHECK while their families are drafts — and nothing in the release
  plane calls it. Wiring it needs an adapter→gate edge `t3.py` deliberately does
  not have; the honest fix is a per-phase `ProtocolBinding` in `T3Request`, which
  is a request-shape change AK6 also depends on.
- **`t3.verify_waiver` still accepts any non-empty `authorized_by`.** The
  machine-actor refusal is enforced in the packager (seam 7), so a self-granted
  waiver cannot reach a package — but T3's own verdict still reads
  PASS_WITH_WAIVER. The correct home for `MACHINE_ACTOR_TOKENS` is `schemas.py`,
  where both planes can read it; that file is the shared SSOT and was out of scope
  for the integration pass.
- **Waivers are structurally validated, not authenticated.** No signature, no
  trust-boundary path check on `document_path`, and `WaiverBinding.pinned_sha256`
  pins the document a caller supplied rather than one read from an operator-owned
  path.
- **`_transaction_elements` coverage is satisfied by a command that NAMES an
  element**, not by proving it acts on it — a comment mentioning
  `instrument_eras.yaml` covers the era-registry element. Tightening it needs a
  verb vocabulary per element kind.
- **`RollbackPlan.incumbent_libraries` carries no backend attribution**, because
  `t3.ArchivedBuild.libraries` carries none either. On a three-ggml-generation
  host that is the field a rollback would most want attributed.
- **The `/kernel` dashboard JSON contract and the freshness/health fold** (AK6
  checklist) are not built: an HTTP surface plus a panel→producer registry.
- **`readiness.py` open items from its red-team** remain as reported:
  `check_matrix_coverage()` accepts foreign-backend cells at its public entry
  point; capacity/mechanism evidence is exempt from the lineage-ordering check and
  `MechanismConfirmation` carries no timestamp at all; coverage/co-residency/
  repetition checks count inadmissible cells and the `llama_cpu` co-residency
  requirement can be closed by a sentinel; sub-floor estimates can still be
  selected as weakest or best (an operator call — excluding them makes a phase
  measured entirely at parity report "no figure").
- **`t3.py` open items**: a phase trade's `expected_gain` is validated for
  structure and never compared to any standing; `sealed_fingerprint` hashes the
  active waiver's digest but not its coverage, so two runs whose waivers cover
  different cells share a fingerprint (fail-closed today).

### Remaining in AK8/AK9 (the adapters)

- **Both speech release-protocol families are drafts**, so `whisper_stt` and
  `qwentts_tts` are search-legal and release-blocked by design. Their phase
  vocabularies are absent from `schemas.PHASES_BY_BACKEND` for the same reason.
- **`serving_runtime`'s `kernels/production` pattern refuses the package from
  reporting the normal launch command** — the stable symlink is also the path a
  service legitimately executes, so a realistic §11.6 package cannot name its own
  serving binary. Fixing it needs a pattern that distinguishes *executes a binary
  under the path* from *mutates the path*.
- **Gate 3 never checks `argv[0]` against `binary_path`** and never ties a
  `LiveProcessFact.pid` to gate 2's observed pid.
- **`check_device_evidence(expected_lane="gpu")` accepts `Device 0: CPU`** in both
  speech adapters: it requires a `Device N: <name>` line and never checks the name
  denotes a GPU. A correct fix needs a device-name vocabulary that belongs in the
  evaluator bundle.
- **`check_exclusion_rate(aa_dispersion=…)` has a unit hazard** (fraction vs
  percentage-point, no suffix) and **`check_stage_attribution(tolerance_ms=…)` is
  unbounded** — a tolerance larger than the measurement passes.
- **One `EXPECTED_SHARED_LIBRARIES` set for the whole inventory**, so the §10.2
  phase-2 gate is unrunnable for inventory members that link a subset.

### Escalations raised by this work, for the operator

- **The 2026-07-31 ggml-linkage remediation is not live in this container.**
  `/etc/environment`, `.devcontainer/devcontainer.json` and `.devc/overrides.json`
  were all cleaned, but the running container still exports the pre-remediation
  `LD_LIBRARY_PATH`, under which all four frozen speech binaries resolve
  `libggml*.so.0` from `/mnt/raid0/llm/llama.cpp/build/bin` (ggml 0.16.0) while
  loading `libggml-hip.so.0` from their own tree. Any speech measurement taken
  from an agent shell here without an explicit per-launcher `LD_LIBRARY_PATH` is
  attributable to the wrong build. Operator action: container restart.
- **`ratify_speech_kernel_freeze_20260731.json:27-28` anchors "whisper
  large-v3-turbo f16" to `wer_pct: 2.35`**, which in
  `/mnt/raid0/llm/tmp/stt_wer_results.json` belongs to *faster-whisper
  large-v3-turbo int8 CPU 48t* (44/1870). The `whisper.cpp large-v3-turbo f16
  MI210 GPU` arm is **3.37 % (63/1870)**. A ratified receipt is corrected by a
  superseding receipt, which is human-only.

### Not started at all

**AK7** beyond its entry point: `packager.OperatorFreezeRequest` is the door and
`audit_no_clock_or_self_trigger()` proves the packager has no clock and never
constructs one of its own, but the cadence policy AK7 describes has no
implementation (deliberately — AK-D25 keeps cadence an operator policy).

### Known, documented holes in what *is* implemented

- Unlinking a held lock file hides its live holder from the `/proc/locks`
  witness — a tmp reaper over `/mnt/raid0/llm/tmp` does it, no attacker needed.
  Tracked as an `expectedFailure` in `test_preflight.py`.
- `Journal.write_lock()` re-entrancy is not thread-safe (cross-process is
  correct; two threads sharing one `Journal` both believe they hold a lock only
  one acquired).
- Liveness identity has no PID-namespace component, so two containers sharing a
  hostname with private PID namespaces would classify a *live* foreign holder as
  DEAD. Not triggerable on this host today (`/proc/self/ns/pid` is the init ns).
- Neither `storage.expire_artifact` nor the tombstone path takes any lock; two
  processes can both pass `plan_expiry` on one artifact.
- `_parse_line` rejects any unknown `kind`, so a journal written by a newer
  version is entirely unreadable by an older reader — fail-closed by design, and
  a forward-compatibility wall.
