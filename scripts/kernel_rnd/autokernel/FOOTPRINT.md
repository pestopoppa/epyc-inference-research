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
| **ON THE CAMPAIGN PATH** | **49,877** |
| **DEFERRED** (provably unreachable) | **51,142** |
| **TOTAL** | **101,019** |

**Roughly half of this package is not on the path from "an idea for a
kernel" to "a measured number."** The three figures above are regenerated from
the tree and asserted row by row; no percentage or line count is repeated
anywhere else in this file, because a number stated twice is a number that can
drift in one of the two places. It is not wrong; it is not reached. That is what the
boundary test makes provable, and it is what makes deleting it a one-line
decision (`DELETED_BY_OPERATOR` in `test_campaign_footprint.py`) rather than a
leap of faith.

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
| `campaign.py` | 2,195 | yes | THE ENTRYPOINT. Before it landed, `grep -rl "__main__|argparse|def main("` over every non-test module returned nothing: 94k lines, 5,695 passing tests, and no way to start it — which is the whole reason this package has produced no result |
| `__init__.py` | 14 | yes | package docstring; `schemas` is declared here as the single source of record shape |
| `schemas.py` | 2,790 | yes | one record shape — every module is written against it and none invents its own |
| `journal.py` | 2,127 | yes | AutoPilot lost 232 trials and ~16 days when a restart came up empty and nothing objected |
| `storage.py` | 1,858 | yes | the 2026-07-04 async-prefetch win was written to `/mnt/raid0/llm/tmp/` and that directory no longer exists |
| `evaluator/__init__.py` | 41 | yes | docstring only — it binds no submodule, so importing `evaluator.api` does not drag the plane in |
| `evaluator/api.py` | 3,024 | yes | a `Verdict` is constructible only via `compute_verdict()`; `kernel_eval.sh` stamped `"status":"OK"` unconditionally |
| `evaluator/correctness.py` | 3,514 | yes | throughput is reward-hackable: deleting the computation is the fastest kernel there is |
| `evaluator/recipes.py` | 2,368 | yes | argv from a hashed constructor — production drifted off NUMA interleave 2026-05-24 and the front door ended up at 46% of canonical |
| `evaluator/devices.py` | 577 | yes | a GPU cell must not be satisfied by `Device 0: CPU` |
| `evaluator/controls.py` | 2,405 | yes | the A/A control plane — 2026-08-04 measured 1.62% / 1.88% between-run CV over four identical runs |
| `evaluator/statistics.py` | 3,628 | yes | **calibration constants and `median` only.** Its e-process made the gate unpassable: threshold 10 against a sign-martingale that tops out at 5.5687, at every effect size. Fenced by `TestNoOptionalStopping` |
| `evaluator/integrity.py` | 3,581 | yes | **provenance primitives only** — `sha256_file`, `hash_source_tree`, `EMPTY_TREE_SHA256`, the clean-build snapshot check. Its §8.5.1 gate runner is fenced off |
| `evaluator/surface.py` | 3,195 | yes | **change-class constants only** — `AffectedSurface`, the core/shared-header fanout classes. `SurfaceGateRunner` is fenced off |
| `execution/__init__.py` | 24 | yes | docstring only; states the deny-8 limits every executor inherits |
| `execution/worktree.py` | 2,645 | yes | no candidate exists without it: production-tip anchoring, campaign worktree, build, build-identity receipt |
| `execution/microbench.py` | 3,193 | yes | paired ALTERNATING blocks — 2026-08-04 A/A decode declined monotonically over four runs, so candidate-then-anchor charges the second arm ~4% systematically and repetitions do not remove it |
| `execution/t0_provider.py` | 3,052 | yes | the predecessor harness tested MUL_MAT only, so a kernel that broke MUL_MAT_ID — MoE dispatch, every token in production — passed it cleanly |
| `execution/control_runner.py` | 1,546 | yes | runs the neutral / A-A controls that the measured drift makes mandatory rather than optional |
| `execution/cpu_region_claim.py` | 2,408 | yes | 2026-08-04: two A/A runs were destroyed by a legitimate co-tenant because the loop held no claim. Before this module a claim could be READ but never acquired |
| `execution/chain.py` | 1,725 | yes | holds the seams — four mismatches between executors and evaluator, one of them a field whose meaning INVERTS across the seam |
| `resource/__init__.py` | 28 | yes | docstring only; names the `resource`-shadows-stdlib hazard the loop must not trip |
| `resource/device_claim.py` | 1,826 | yes | §2.6's first row of substrate that exists nowhere in the project: a cross-process GPU device claim someone actually holds |
| `resource/preflight.py` | 1,788 | yes | INC-20260731: a name-pattern kill took out another agent's `llama-server` twice, and `earlyoom`, whose argv names what it guards |
| `resource/claim_witness.py` | 325 | yes | invariant 9 — idle sensing is never a claim, and the witness is what tells the two apart |
| `controller/__init__.py` | 71 | no | binds every controller module, so any import of this package pulls the whole plane |
| `controller/selection.py` | 2,953 | no | 22 rejection codes and zero domain knowledge about what makes an EPYC or an MI210 kernel fast |
| `controller/guards.py` | 3,670 | no | deterministic stop conditions for a loop that has never taken a step |
| `controller/context.py` | 3,167 | no | compiles context for an LLM planner; campaign #1's proposals come from the session, not from a compiled context |
| `controller/hypotheses.py` | 4,481 | no | an operator hypothesis channel with no campaign to feed it |
| `controller/critic.py` | 2,077 | no | pre/post-run critics gating proposals that no producer emits yet |
| `controller/state_machine.py` | 1,972 | no | states for a loop with no entrypoint; campaign #1 is one candidate, run once, by hand |
| `controller/composition.py` | 2,165 | no | champion-lineage maintenance before a single champion exists |
| `controller/planner.py` | 1,397 | no | research strategy in Python, replacing 114 lines of prose in autoresearch's `program.md` |
| `controller/oracles.py` | 405 | no | a §6.5 oracle registry whose two consumers are both in this deferred plane |
| `controller/do_not_repeat.py` | 2,196 | no | the §19.2 memory-update plane `hypotheses.py` says it does not own; it feeds `check_do_not_repeat()` and `selection.match_ledger()`, both of which are in this same deferred plane |
| `controller/fingerprint.py` | 144 | no | proposal identity for proposals nothing produces |
| `release/__init__.py` | 31 | no | binds `plan`; the release plane ships a champion and campaign #1 has none |
| `release/t3.py` | 6,660 | no | the T3 freeze gate — explicitly outside P-AK-SEARCH-1, and `api.admit_tier()` refuses T3/T4 by name on the search path |
| `release/readiness.py` | 4,312 | no | 4,312 lines producing a number the design forbids branching on: `is_trigger=False`, and `composite_readiness()` / `freeze_eligibility()` raise |
| `release/packager.py` | 4,276 | no | seals a package a human executes; needed to SHIP a champion, never to FIND one |
| `release/plan.py` | 2,503 | no | the release-plan compiler was committed the day BEFORE the code that can compile a candidate |
| `adapters/serving_runtime.py` | 3,646 | no | the three-gate stack-change path — a backend this kernel search is not searching |
| `adapters/whisper_stt.py` | 1,806 | no | STT backend; the speech kernels are frozen at `production-speech-v1` and campaign #1 is a llama.cpp CPU-decode cell |
| `adapters/qwentts_tts.py` | 1,935 | no | TTS backend, same freeze, same reason |
| `adapters/__init__.py` | 21 | no | docstring only, but the package it heads is backends nothing is searching |
| `surface/__init__.py` | 40 | no | imports `dashboard_contract`, so reaching ANY module under `surface/` executes the producer |
| `surface/dashboard_contract.py` | 1,214 | no | a freshness contract so that a dead loop cannot read as fresh. The loop has never been alive; make it live first |

---

## What the boundary test enforces

`test_campaign_footprint.py`, beside the entrypoint:

1. **`TestCampaignFootprint.test_campaign_path_does_not_reach_the_deferred_half`**
   — the walked graph reaches nothing under `controller/`, `release/`,
   `adapters/` or `surface/`. This is the DEFERRED figure in the table above.
2. **`test_the_deferred_half_is_still_on_disk`**,
   **`test_the_deferred_half_is_a_real_share_of_the_tree`** and
   **`test_the_entrypoint_exists`** — anti-vacuity. The boundary must not start
   passing because its targets were renamed, because 20,000 lines migrated onto
   the campaign path, or because the entrypoint it is drawn around was deleted.
   The last of those was a `skipUnless`, which meant `rm campaign.py` turned this
   whole file into a green no-op.
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
   nothing about `execution/README.md` §6.5 — a declared round re-run until it
   crosses — which is optional stopping ACROSS processes and needs a durable run
   ledger, not a static assertion. §6.5 is still OPEN and was re-reproduced on
   2026-08-04.
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
   the checks fire on the real 95k-line closure. The copy is why: the shared
   clone (`/workspace/repos/…` and `/mnt/raid0/llm/…` are one checkout) is never
   written to.

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
nothing else changes. That edit is the entire cost, and it is the reason this
was written as a test rather than as a recommendation.
