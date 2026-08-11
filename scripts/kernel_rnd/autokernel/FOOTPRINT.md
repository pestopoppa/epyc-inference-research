# AutoKernel footprint — what campaign #1 actually imports

**Generated from the import graph, and asserted against it** by
`test_campaign_footprint.TestFootprintDocumentMatchesTheTree`, on each run:
every `yes`/`no` against the walked graph, every module against a row, every row
against a module that exists, every row against a stated reason, and the three
totals against the tree. If this file drifts from the tree in any of those, the
suite goes red. It is an assertion, not a description.

The per-row LINE COUNT is the one column that is regenerated but not asserted
to the line, and that is deliberate: five sessions share this clone, asserting
it exactly produced five red suites in forty minutes on 2026-08-04 and not one
of them was a boundary violation. The totals carry a 1,000-line tolerance for
the same reason — enough to ignore a plane being edited, not enough to ignore a
plane MOVING. `--refresh` restores every mechanical column to the line.

Reachability is computed by walking the AST of every module reachable from the
declared campaign-#1 roots — following relative imports at any level,
function-level imports, parent-package `__init__` side effects and dynamic
`import_module(…)` strings, absolute and relative. It is not a grep. A dynamic
import whose module name is not a literal (an f-string, a concatenation) cannot
be followed by any static walk, so it is REPORTED rather than skipped —
`test_no_dynamic_import_on_the_campaign_path_is_unresolvable` is red while one
exists, because a walk that silently steps over it reports a clean boundary over
a module it never looked at.

---

## The number

| | non-test lines |
|---|---:|
| **ON THE CAMPAIGN PATH** | **63,758** |
| **DEFERRED** (provably unreachable) | **5,467** |
| **TOTAL** | **69,225** |

**There is no deferred half.** The compact modules deliberately off the
mutation/build path are offline analysis or pre-campaign planning surfaces: the
observe-only proposal diagnostic, prior-art compiler, substrate and lane
registries, turn-productivity reducer, and operator-invoked live-control
producer. The running campaign does import the compile-artifact veto because
that check must fire before behavioral T0. Reachability remains enforced module
by module; a new strategy or release plane is still forbidden by default.

The three figures above are regenerated from the tree and asserted row by row; no
percentage or line count is repeated anywhere else in this file, because a number
stated twice is a number that can drift in one of the two places.

### What the operator deleted, 2026-08-04

Until 2026-08-04 this table read *roughly half of this package is not on the path
from "an idea for a kernel" to "a measured number"*, over a total near 101k. The
operator acted on it: `release/`, `adapters/`, `surface/` and the AK4 strategy
plane under `controller/` were removed — about 79,600 lines including their
tests, recoverable from the tag `autokernel-preserve-20260804`. The rationale is
`epyc-root/artifacts/operator/autokernel-simplification-review.md`; the
prediction this document made — that removing them could not change what campaign
#1 does, because the walked graph reached none of them — held, and no reachability
assertion in `test_campaign_footprint.py` changed.

### The last four modules to cross, 2026-08-04

What survived the deletion under `controller/` was its MEMORY half:
`hypotheses.py` (the operator's hypothesis drop-in), `do_not_repeat.py` (the
§19.2 ledger) and `shared.py` (the six lines the two of them needed from the
plane that was removed). This document argued they were deferred *"read by
whoever PROPOSES a candidate and by nothing that measures one"*.

That was right about STRATEGY and wrong about CLAIMS. `hypotheses.py` also holds
`claim_for_hypothesis`, which documents itself as **"The ONLY route from a
hypothesis to a resource claim"** and enforces the rule that a falsifier is
optional when a question is written and mandatory before a claim is spent on it.
While it sat on the far side of this boundary it had **zero non-test callers**:
`campaign.py` called `acquire_cpu_region_claim` directly, so the gate enforced
nothing. The driver is what SPENDS the claim, so the gate belongs in the driver,
and `campaign.py --hypothesis` now acquires the region claim through it.

`do_not_repeat.py` came with it, and the check was made before admitting it
rather than after: `check_do_not_repeat(*, regime, matches)` is pure and the
driver never calls it — but `HypothesisTracker.authorize_claim(ledger=…)` has no
default and `claim_for_hypothesis` refuses a token carrying no verdict, so no
spendable token exists without a real ledger. `compile_for_tracker` builds one
from the tracker's own record, and it lives there. `controller/__init__.py`
binds both modules, so reaching one reaches both in any case.

**The prefix is still banned.** `test_campaign_footprint.CONTROLLER_ALLOWED` is a
LIST of four module names, not a prefix allowance: a fifth module dropped into
`controller/` is a boundary finding, and
`test_a_direct_import_of_controller_is_caught` plants exactly that module in a
copy of the tree and asserts it is caught.

### One correction to the earlier estimate

The deferred half was previously put at ~54k on the assumption that
`evaluator/integrity.py` and `evaluator/surface.py` were also unreachable.
**They are not.** `worktree.py` builds its build-identity receipt
from `integrity.check_clean_build_from_snapshot` and `integrity.hash_source_tree`,
`microbench.py` hashes the benchmarked binary with `integrity.sha256_file`, and
`chain.py` reads `surface.AffectedSurface` change classes. Those two modules
cannot be cut without cutting the receipt that proves a candidate was built
clean.

What *is* deferred inside them is their **gate derivation** —
`integrity.SourceIntegrityGateRunner` / `SourceIntegrityFirstRunner` and
`surface.SurfaceGateRunner`, the second of the two coexisting §8.5.1
derivations, the one that consumes evidence nothing produces (T0 today produces
nine of seventeen surfaces). Those get a **symbol fence** in
`TestTheProvenanceFence` — an allowlist frozen from the tree on 2026-08-04, so
any new name bound from either module fails the suite — rather than a
reachability ban that could only be satisfied by hollowing out `chain.py`.

A boundary you can only pass by deleting what it inspects measures the test
author, not the code.

---

## Every module

`yes` = reachable from the declared campaign-#1 roots. Reasons are a real
incident or a measured fact; "reduced rigour" is not a reason.

| module | lines | campaign #1 | reason |
|---|---:|:---:|---|
| `campaign.py` | 3,490 | yes | THE ENTRYPOINT. Before it landed, `grep -rl "__main__|argparse|def main("` over every non-test module returned nothing: 94k lines, 5,695 passing tests, and no way to start it — which is the whole reason this package has produced no result |
| `dashboard.py` | 201 | yes | the terminal result was fsynced but the only dashboard exporter had been deleted, so active AutoKernel work remained permanently absent from the operator surface; this compact projection dates itself from the journal entry and cannot make an old campaign fresh |
| `__init__.py` | 14 | yes | package docstring; `schemas` is declared here as the single source of record shape |
| `schemas.py` | 3,360 | yes | one record shape — every module is written against it and none invents its own |
| `journal.py` | 2,181 | yes | AutoPilot lost 232 trials and ~16 days when a restart came up empty and nothing objected |
| `offline_least_commitment.py` | 345 | no | AP-WM-1 observe-only archive analysis; importing an offline hypothesis diagnostic into the mutation/build path would give it accidental live authority |
| `turn_productivity.py` | 481 | no | AK-PT-1/AK-X-6 archive reducer; it consumes completed refine-turn records and may only withhold future search advancement, so campaign #1 must not give it live rank or mutation authority |
| `prior_art.py` | 597 | no | deterministic proposal-input compiler; it classifies findings before a proposal exists and has no place in the mutation/build process |
| `profile_report.py` | 542 | no | RVP-1–7 deterministic offline C4 report; it consumes completed paired traces and has no mutation, build, profiler-launch or ranking authority |
| `profile_context.py` | 287 | no | C4 hash-bound discovery/evaluator projection; it exposes diagnostic context to authoring without gaining verdict or ranking authority |
| `substrate.py` | 340 | no | validated planning facts; it reads checked-in measured/datasheet receipts before proposal construction and never joins the mutation/build path |
| `lanes.py` | 314 | no | screening declarations and rank-transfer calibration; without measured calibration campaign #1 stays on the full verified path |
| `artifact_diff.py` | 200 | yes | AK-TR-6 must veto an unconfirmed GPU claim before the behavioral T0 provider can launch |
| `storage.py` | 1,859 | yes | the 2026-07-04 async-prefetch win was written to `/mnt/raid0/llm/tmp/` and that directory no longer exists |
| `evaluator/__init__.py` | 41 | yes | docstring only — it binds no submodule, so importing `evaluator.api` does not drag the plane in |
| `evaluator/api.py` | 3,320 | yes | a `Verdict` is constructible only via `compute_verdict()`; `kernel_eval.sh` stamped `"status":"OK"` unconditionally |
| `evaluator/correctness.py` | 3,838 | yes | throughput is reward-hackable: deleting the computation is the fastest kernel there is |
| `evaluator/recipes.py` | 2,433 | yes | argv from a hashed constructor — production drifted off NUMA interleave 2026-05-24 and the front door ended up at 46% of canonical |
| `evaluator/devices.py` | 813 | yes | a GPU cell must not be satisfied by `Device 0: CPU` |
| `evaluator/controls.py` | 2,406 | yes | the A/A control plane — 2026-08-04 measured 1.62% / 1.88% between-run CV over four identical runs |
| `evaluator/baseline_honesty.py` | 129 | no | AK-BH-4 exact-surface strongest-provider selector; it rejects AUTO and cross-surface transfer before a campaign can claim an honest floor |
| `evaluator/sensitivity.py` | 333 | no | RVP-C2-7/C2-11/C5-2 standalone two-axis reducer; missing or insensitive seed/transform populations are unscoreable rather than campaign passes |
| `evaluator/oracle_integrity.py` | 145 | no | RVP-C2-8/C2-9 standalone reducers: hostile distributions and checker isolation are correctness prerequisites, not campaign-path ranking authority |
| `evaluator/historical_tasks.py` | 194 | no | RVP-C5-R/C3-2 sealed historical-task descriptor and expert-ceiling reducer; no terminal candidate means `COULD_NOT_CHECK` |
| `evaluator/statistics.py` | 3,669 | yes | **calibration constants and `median` only.** Its e-process made the gate unpassable: threshold 10 against a sign-martingale that tops out at 5.5687, at every effect size. Fenced by `TestNoOptionalStopping` |
| `evaluator/integrity.py` | 3,661 | yes | **provenance primitives only** — `sha256_file`, `hash_source_tree`, `EMPTY_TREE_SHA256`, the clean-build snapshot check. Its §8.5.1 gate runner is fenced off |
| `evaluator/surface.py` | 3,195 | yes | **change-class constants only** — `AffectedSurface`, the core/shared-header fanout classes. `SurfaceGateRunner` is fenced off |
| `execution/__init__.py` | 24 | yes | docstring only; states the deny-8 limits every executor inherits |
| `execution/worktree.py` | 2,773 | yes | no candidate exists without it: production-tip anchoring, campaign worktree, build, build-identity receipt |
| `execution/microbench.py` | 4,320 | yes | paired ALTERNATING blocks plus C6-10 ranked hard cases — each hostile unit changes the receipted recipe and contributes blocks to the same rank instead of living only in a correctness gate |
| `execution/device_sampler.py` | 410 | yes | RVP-C3-4 numeric 250 ms ROCm state producer; it brackets the exact captured benchmark-process lifetime and fails closed on missing fields, failed probes, empty traces, or cadence gaps |
| `execution/instrument_integrity.py` | 111 | yes | RVP-C6-1: a candidate binary is built from candidate-controlled source, so every live T1 invocation must re-pin its reward-bearing translation unit to the named anchor before it can emit a number |
| `execution/physical_bounds.py` | 197 | yes | RVP-C6-4 physical impossibility screen: per-shape conservative work floors and hardware peak ceilings are bound to the exact delivered unit and recipe/model/parameter frame, then every live sample is checked before ranking |
| `execution/reward_hack_scan.py` | 186 | yes | RVP-C6-6/C6-9 plus static C6-2/C6-3 detectors: protected-frame, pointer-memo, structured-shortcut, environment/timing and stream/thread findings; the named 10 planted/15 clean corpus states sensitivity/specificity/FPR, not arbitrary-program coverage |
| `execution/sandbox.py` | 587 | yes | C6 candidate boundary: Landlock write confinement, seccomp signal/network/namespace denials, non-root finite rlimits, per-invocation cgroup membership and verified empty teardown |
| `execution/t0_provider.py` | 3,676 | yes | the predecessor harness tested MUL_MAT only, so a kernel that broke MUL_MAT_ID — MoE dispatch, every token in production — passed it cleanly |
| `execution/control_runner.py` | 1,550 | yes | runs the neutral / A-A controls that the measured drift makes mandatory rather than optional |
| `execution/live_controls.py` | 839 | no | standalone, operator-invoked calibration producer for the fixed five controls; it prepares the instrument before campaign #1 and is deliberately not imported by the mutation/build entrypoint |
| `execution/cpu_region_claim.py` | 2,408 | yes | 2026-08-04: two A/A runs were destroyed by a legitimate co-tenant because the loop held no claim. Before this module a claim could be READ but never acquired |
| `execution/chain.py` | 1,928 | yes | holds the seams — four mismatches between executors and evaluator, one of them a field whose meaning INVERTS across the seam |
| `resource/__init__.py` | 28 | yes | docstring only; names the `resource`-shadows-stdlib hazard the loop must not trip |
| `resource/device_claim.py` | 1,826 | yes | §2.6's first row of substrate that exists nowhere in the project: a cross-process GPU device claim someone actually holds |
| `resource/preflight.py` | 1,788 | yes | INC-20260731: a name-pattern kill took out another agent's `llama-server` twice, and `earlyoom`, whose argv names what it guards |
| `resource/claim_witness.py` | 325 | yes | invariant 9 — idle sensing is never a claim, and the witness is what tells the two apart |
| `controller/__init__.py` | 76 | yes | binds every surviving controller module, so importing one reaches both — which is why `controller.do_not_repeat` is on the path whether or not the driver names it, and why `CONTROLLER_ALLOWED` lists this file rather than leaving the edge unexplained |
| `controller/authoring_contract.py` | 468 | no | AK-PL-1/AK-LE-4/AK-LE-5 pre-proposal adapter: fully rendered prompt leak refusal, priced never-bulk-read context, reversible compaction, and structured external numbers; it calls no model and must not gain mutation/build authority |
| `controller/reward_monitor.py` | 453 | no | C6 monitor adapter: binds campaign/candidate traces and the whole journal tree to a predeclared monitor panel, requires awareness plus reasoning visibility, and reports sensitivity/specificity/FPR without calling a model |
| `controller/hypotheses.py` | 4,493 | yes | `claim_for_hypothesis` — the falsifier-before-compute gate `campaign.py --hypothesis` acquires its region claim through. It calls itself the ONLY route from a hypothesis to a resource claim and had ZERO non-test callers until 2026-08-04, because this boundary put it on the far side of the line: the driver is what SPENDS the claim |
| `controller/do_not_repeat.py` | 2,205 | yes | the §19.2 ledger a loop needs to tell "tried and failed" from "never tried". On the path because `authorize_claim(ledger=…)` has no default and `claim_for_hypothesis` refuses a token with no verdict, so no spendable token exists without a real one — `compile_for_tracker` is it |
| `controller/shared.py` | 166 | yes | the six lines `hypotheses` and `do_not_repeat` reached into the removed plane for — `ControllerError`, `selection_block()`, `LEDGER_DIMENSIONS` and the fingerprint pair. Twenty thousand lines were pinned by six, because a concern shared by two modules had nowhere to live; reached only through them, and the campaign path names nothing in it |

---

## What the boundary test enforces

`test_campaign_footprint.py`, beside the entrypoint:

1. **`TestCampaignFootprint.test_campaign_path_does_not_reach_the_deferred_half`**
   — the walked graph reaches nothing under `controller/`, `release/`,
   `adapters/` or `surface/`. The last three are deleted, so for them it is a ban
   on their ever coming back onto the path; `controller/` is the live prefix, and
   it is the DEFERRED figure in the table above.
2. **`test_the_deferred_half_is_still_on_disk`**,
   **`test_the_deferred_half_is_a_real_share_of_the_tree`** and
   **`test_the_entrypoint_exists`** — anti-vacuity. The boundary must not start
   passing because its targets were renamed, because the deferred modules migrated
   onto the campaign path, or because the entrypoint it is drawn around was
   deleted. The last of those was a `skipUnless`, which meant `rm campaign.py`
   turned this whole file into a green no-op. The line floor in the second was
   RE-PINNED on 2026-08-04: 40,000 was calibrated against a 46k deferred half that
   the operator then deleted, and a floor no surviving file can clear is not a
   check, it is a permanent red. The derivation of the replacement is in the
   constant's own comment.
2b. **`test_no_dynamic_import_on_the_campaign_path_is_unresolvable`** — every
   check here is a statement about a graph that was WALKED, and an
   `import_module(f"{__package__}.…")` is a hole in that graph. It is REPORTED,
   never stepped over: a walk that skips what it cannot follow returns a clean
   boundary over a module nobody looked at.
3. **`TestTheProvenanceFence`** — `integrity.py` and `surface.py` are reachable
   only through a frozen allowlist of hashing / change-class names, and neither
   deferred gate runner is wired. Both the allowlist and the denylist are
   checked against the modules' own ASTs, so a typo cannot silently forbid or
   permit nothing. The fence reads four idioms, and the last two defeated its
   first version: a direct name import; `integrity.X` through a one-hop alias;
   `evaluator.integrity.X` through a CHAINED alias, which runs because
   `worktree.py` has already made `integrity` an attribute of the package; and
   `getattr(integrity, "X")`.
4. **`TestNoOptionalStopping`** — the campaign path binds no e-process name, and
   reaches no interim-look method through an object it legitimately holds
   (`CampaignStatistics.sequential_evaluation()` is exactly the hole a
   name-level check leaves open). The accept rule, `api._resolve_effect`, cannot
   import `statistics` at all, so it can READ a recorded e-value but never run
   one.
   **Scope limit, stated so it is not mistaken for cover:** this is a check on
   the import graph, so it forbids optional stopping *inside a process*. It says
   nothing by itself about `execution/README.md` §6.5 — a declared round re-run
   until it crosses — because that is optional stopping ACROSS processes. The
   durable completed-run ledger closes that separate seam; this static assertion
   remains responsible only for the in-process path. §6.5 closed 2026-08-05.
5. **`TestTheWalkerItself`** — synthetic graphs that bite-verify the walker
   (transitive edge, function-level import, `try:`-guarded import,
   parent-`__init__` side effect, two-level relative import, dynamic import by
   absolute string, by relative string, and by f-string, chained alias, and
   `getattr`), with compliant-path controls beside them that must produce no
   finding.
5b. **`TestTheTotalsCheckBites`** — the totals check against text it must reject:
   a swapped pair, a plane moving, a total restated in the prose, a missing row.
   Its own first fixture put the two halves one tolerance apart, which made a
   swap unobservable and the bite test FAIL — so it also records, as an
   assertion, the bound it really has.
6. **`TestTheBoundaryCatchesRealTreeViolations`** — the same four checks run
   against a **copy of this package with a violation planted in it**: every
   mutation caught, and compliant-path controls beside each, all silent. Toy
   fixtures verify that the walker resolves each import *form*; this verifies
   the checks fire on the real closure. The copy is why: the shared
   clone (`/workspace/repos/…` and `/mnt/raid0/llm/…` are one checkout) is never
   written to.
   **Re-pointed 2026-08-04.** Five of these plants named modules under the deleted
   planes. A probe that imports a module which does not exist plants NOTHING — the
   walk drops a name with no file, the check reports clean, and the test fails
   claiming the walker did not bite when what was missing was the target. They now
   plant against `controller/`, the one deferred prefix still on disk, which keeps
   each mechanism (direct import, function-level import, parent-`__init__` side
   effect, dynamic import by absolute and by relative string) exercised against the
   real tree. The deleted prefixes keep the one assertion still meaningful about
   them — `test_re_adding_a_deleted_plane_and_importing_it_is_caught`, which writes
   an empty stub plane into the temp COPY and proves the ban fires the moment
   something depends on it again.

## Three findings this exercise produced

**1. The improvement verdict is currently unreachable on the campaign path.**
`api._resolve_effect` returns `EFFECT_EVIDENCE_BELOW_THRESHOLD` whenever
`effect.e_value < effect.threshold`, and `EffectEstimate` requires both fields at
construction. For the CPU decode cell the calibration solves `threshold = 10`
while the sign-martingale tops out at 5.5687 over five same-sign blocks — at four
different effect magnitudes, because the construction is sign-based. So a
candidate that is genuinely faster still resolves to "evidence below threshold".
Campaign #1 must either produce its T1 result outside `_resolve_effect`, or
declare a construction whose e-value can cross. The 2026-08-04 A/A data
(between-run CV 1.62% / 1.88%) says the honest answer is the first one: a median
over paired deltas covers this noise, and the e-process was never warranted here.
`TestNoOptionalStopping` fences the mechanism; it does not decide this question.

**2. `campaign.py` does not reach the A/A control plane.** The entrypoint imports
`journal`, `schemas`, `storage`, `evaluator.{api, correctness, devices, recipes}`,
`execution.{chain, cpu_region_claim, microbench, t0_provider, worktree}` and
`resource.{claim_witness, device_claim, preflight}` — and neither
`evaluator/controls.py` nor `execution/control_runner.py`. Those
two are declared campaign-#1 roots here, and they are declared for a measured
reason: on 2026-08-04 four A/A runs of identical code showed decode declining
MONOTONICALLY (52.76 → 52.31 → 51.62 → 50.52), which is drift, not scatter, and
no number of repetitions removes it. Interleaved paired blocks handle the drift
WITHIN a comparison; the neutral control is what tells you the drift is there at
all. A campaign that never runs one cannot distinguish "this kernel is 4% faster"
from "this kernel ran first". This is a gap in the entrypoint, not in the list.

**3. `evaluator/` is reachable in full — all nine modules.** There is no thinner
evaluator slice available without breaking a real seam. If the operator wants the
campaign path smaller, `evaluator/` is where the next boundary has to be drawn,
and it is a refactor, not a deletion.

## Acting on this

To delete a deferred plane: move its prefix into `DELETED_BY_OPERATOR` in
`test_campaign_footprint.py`, remove the directory, and delete its rows here.
The reachability assertion still holds — an absent module is unreachable — and
that edit is why this was written as a test rather than as a recommendation.

**What that edit does NOT cover, measured on 2026-08-04 by doing it.** "Nothing
else changes" was too strong, and the correction generalises past this package:

* Every OTHER test that named a module in the plane is now a test with no
  subject. Three kinds turned up — a boundary bite-test whose planted violation
  can no longer be planted, a composition test importing the plane's driver, and
  an anti-vacuity control that proved a rule was not so narrow it rejected its own
  real consumers. The first is re-pointed at a surviving plane, the second is
  excised, and the third has no replacement and was removed with a note saying
  what was lost. Fixing the first two by asserting less would have been the exact
  failure this document exists to catch.
* Any calibrated constant measured against the deleted lines — here the deferred
  line floor — must be re-derived from the tree that is left, not lowered until
  the suite goes quiet.
* A plane rarely leaves cleanly. `controller/` could not: two modules the operator
  kept reached into the removed plane for six lines, which now live in
  `controller/shared.py`. Budget for a survivors' module before starting.
