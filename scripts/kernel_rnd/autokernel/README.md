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

## What is implemented (AK1 + AK2 + AK3, partial)

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

# The two AK3 cross-module suites — the ones that fail when two modules disagree.
python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_conformance.py
python3 -m unittest scripts/kernel_rnd/autokernel/evaluator/test_integration.py

# As a plain script.
python3 scripts/kernel_rnd/autokernel/test_integration.py
```

Expected: **1846 tests, OK (expected failures=1)**. The one `expectedFailure` is
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
- **Precondition 4 is enforced on two of its three components in `correctness`.**
  `LinkageEvidence` carries `anchor_binary_sha256` and `anchor_linkage_sha256`
  and no `anchor_source_commit`; `CoherenceEvidence` names no anchor at all, so a
  `byte_identical` PASS is against *some* anchor output.
  `api.AnchorIdentity.identity_matches()` compares all three and is never called
  from `correctness.py`. `integrity.check_evidence_binding` does bind the ELF
  tables, and registration tables carry no provenance digest at all.
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
- **Readiness, composition and the campaign lifecycle are AK4/AK6.** The
  evaluator computes per-record verdicts; nothing computes the advisory readiness
  signal, recomputes calibration at a campaign boundary, drives the A/A cadence,
  or composes a champion lineage.
- **T3/T4 are refused, not implemented.** `api.ReleaseTierEvaluator` is the seam
  AK5 fills.

### Not started at all

**AK4** (state machine, context compiler, planner/critic adapters, selection,
composition, guards). **AK5** (T2 scope and weights, readiness estimator,
release-plan compiler, T3 runner, waiver verification, the v8 dry-run).
**AK6** (readiness reporting and the operator decision surface).

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
