# `autokernel` — the AutoKernel research-loop substrate

Owning design: [`epyc-root/handoffs/active/autokernel-research-loop.md`](/workspace/handoffs/active/autokernel-research-loop.md).
Section references below (`§5.8`, `invariant 7`, …) are to that document.

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

## What is implemented (AK1 + AK2, partial)

| Module | What it owns |
|---|---|
| `schemas.py` | The seven versioned §7 record contracts, canonical JSON/content hashing, the `PASS`/`FAIL`/`COULD_NOT_CHECK` `Check` type, and the record-level checkers (`check_anchor_binding`, `check_scope_denominator_admits_gate`, `check_metric_commensurability`). **Single source of truth: every other module is written against these and must not invent a record shape.** |
| `journal.py` | The append-only, fsynced, **sharded** primary record. Shard ordering, torn-tail repair, cursors, archiving, supersession (record-scope and retrieval-scope), tombstones, preflight attestations, derived views, and `check_view_consistency`. |
| `storage.py` | §3.7 durability classes, §5.8 retention classes and rule-bound tombstoned expiry, per-campaign quota and `DISK_PRESSURE`, the `data/<campaign>/` evidence root with `SHA256SUMS` + README, and `verify_durability`. |
| `resource/device_claim.py` | The cross-process **exclusive GPU device claim** (§2.6) — `flock(LOCK_EX)` on a never-unlinked lock file, PID+start-time+boot-id liveness, journaled crash reclamation, quiesce-and-drain revocation, and the claim receipt id that lands in every evaluation event. |
| `resource/preflight.py` | The **one** audited read-only inference preflight (§3.5): claim witness as the target instrument, an opt-in name-pattern enumerator as the labelled interim one, and an AST self-audit proving the module cannot deliver a signal. |
| `resource/claim_witness.py` | The seam between the three above: a conforming `GpuClaimReader` over the device claim, and resolution of an evaluation event's opaque `resource_claim_receipt` back to the claim that produced it. |

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

## Integration seams that were reconciled (2026-08-03)

The five modules were built in parallel against the same design. Each passed its
own suite; the defects were all *between* them. `test_integration.py` is the
regression barrier for these.

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

# As a plain script.
python3 scripts/kernel_rnd/autokernel/test_integration.py
```

Expected: **571 tests, OK (expected failures=1)** (`pytest` reports it as 570 passed + 1 xfailed). The one `expectedFailure` is
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

### Not started at all

**AK3** (typed evaluator API, affected-surface derivation and trace, correctness
surfaces, ASAN path, codified microbench recipe, e-process reducer, red-team,
the four calibrated controls) — **the first phase that consumes inference**.
**AK4** (state machine, context compiler, planner/critic adapters, selection,
composition, guards). **AK5** (T2 scope and weights, readiness estimator,
release-plan compiler, T3 runner, waiver verification, the v8 dry-run).

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
