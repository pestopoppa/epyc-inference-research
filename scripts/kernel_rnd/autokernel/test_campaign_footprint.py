"""The campaign-#1 import boundary — enforced by walking the graph, not by prose.

WHY THIS FILE EXISTS
--------------------
AutoKernel was 94,083 non-test lines and 5,695 passing tests, and it had produced
NO RESULT. The reason was not that any of it was wrong; it was that most of it was
not on the path from "an idea for a kernel" to "a measured number", and nobody
could tell which half was which. `release/` was committed the day BEFORE the code
that can compile a candidate. `controller/selection.py` encoded 22 rejection
codes and zero facts about what makes an EPYC or an MI210 kernel fast.
`surface/dashboard_contract.py` was a freshness contract for a loop that has never
been alive.

This module draws the line and makes it MECHANICAL. It parses the AST of every
module reachable from the campaign roots — following relative imports,
function-level imports, parent-package `__init__` side effects and dynamic
`import_module("autokernel.…")` strings — and asserts the campaign path does not
reach the deferred half.

WHY A BOUNDARY TEST AND NOT A DELETION — AND WHAT THE DELETION THEN COST
------------------------------------------------------------------------
Deleting the deferred half is the operator's call, and this test is what makes that
call a ONE-LINE decision instead of a leap of faith: the deferred half is
PROVABLY INERT on the campaign path, so removing it cannot change what campaign
#1 does. `test_the_deferred_half_is_still_on_disk` is that one line — when the
operator deletes a plane, its prefix moves to `DELETED_BY_OPERATOR`.

On 2026-08-04 the operator made that call: `release/`, `adapters/`, `surface/`
and the AK4 strategy plane under `controller/` were removed, ~79,600 lines,
recoverable from the tag `autokernel-preserve-20260804`. The prediction held —
no reachability assertion changed — but "nothing else here changes" did not,
and the correction is worth stating because it is a general fact about
guards, not a detail of this one:

    A guard whose only targets have been deleted is not a guard that passes.
    It is a guard that can no longer be exercised.

`TestTheBoundaryCatchesRealTreeViolations` plants a violation in a copy of the
real tree and asserts it is caught. Five of those plants named modules under the
removed planes, and a probe that imports a module which does not exist plants
nothing at all — the walk drops the name and the check reports clean. So they are
re-pointed at `controller/`, which is still on disk, still a `DEFERRED` prefix and
still unreachable from the campaign path, and the DELETED prefixes get the one
assertion that is still meaningful about them:
`test_re_adding_a_deleted_plane_and_importing_it_is_caught`.

THE SAME LESSON, ONE LEVEL DOWN — 2026-08-04, later the same day
----------------------------------------------------------------
`campaign.py` gained `--hypothesis`, and with it the falsifier-before-compute
gate: the region claim is now acquired through
`hypotheses.claim_for_hypothesis`, so a question with no falsifier — or a
placeholder one — cannot reach a claim. That gate is the reason this boundary's
`controller` entry was WRONG. It read *"read by whoever PROPOSES a candidate and
by nothing that measures one"*, which is true of strategy and false of claims:
the driver is what SPENDS the claim, and while the ban stood, "the ONLY route
from a hypothesis to a resource claim" had zero non-test callers.

So the prefix is narrowed MODULE BY MODULE (`CONTROLLER_ALLOWED`), never by
prefix — a prefix allowance would silently re-admit anything added under
`controller/` later. And the plants moved again for exactly the reason above:
they had been re-pointed at `controller.hypotheses` / `controller.do_not_repeat`,
which are now named exceptions, so all five would have gone quiet with the target
absent rather than the bite. Each now imports a module PLANTED under
`controller/` in the copy — still banned, still real, still off the campaign
path — and `test_an_allow_listed_controller_module_is_not_a_finding` is the
control on the other side.

There is now no unreachable module left in this package. The boundary is not
therefore decorative: what it guards is the NEXT module, and the four checks
below are all still exercised against a real tree with a real edge in it.

It also fails LOUDLY the first time someone reintroduces a dependency on the
deferred half, which is the failure mode that produced the 94k lines in the first
place.

WHAT THIS FILE DOES *NOT* CLAIM
-------------------------------
Two modules the deferred list originally named — `evaluator/integrity.py` and
`evaluator/surface.py` — ARE reachable from the campaign path, and no honest test
can say otherwise: `worktree.py` gets its build-identity receipt from
`integrity.check_clean_build_from_snapshot` / `hash_source_tree`, `microbench.py`
hashes the benchmarked binary with `integrity.sha256_file`, and `chain.py` reads
`surface.AffectedSurface` change classes. What IS deferred is their GATE
DERIVATION — `integrity.SourceIntegrityGateRunner` and `surface.SurfaceGateRunner`,
the second of the two coexisting §8.5.1 derivations, the one that consumes
evidence nothing produces. So those two get a SYMBOL fence (an allowlist frozen
from the tree, `TestTheProvenanceFence`) rather than a reachability ban, and the
fence fails the moment a new name is bound from either module.

Reporting the weaker true statement instead of asserting the stronger false one
is the point: a boundary you can only satisfy by hollowing out `chain.py` is a
boundary that measures the test author, not the code.
"""

from __future__ import annotations

import ast
import re
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

#: This file never IMPORTS the package it guards — it only parses it. That is
#: deliberate: importing `campaign.py` to ask what it imports would execute it,
#: and the boundary must be checkable on a host where nothing may be run. It
#: therefore needs no `sys.path` surgery either, and the `sys.path.insert` that
#: used to be here was both dead and a way to change import resolution for every
#: other test sharing the process.
PKG_DIR = Path(__file__).resolve().parent
ROOT_PKG = PKG_DIR.name

#: The entrypoint this boundary is drawn around. It is a path, not an import:
#: this file must be readable and runnable whether or not the entrypoint exists
#: yet, and the moment it lands it is walked with everything else.
ENTRYPOINT_PATH = PKG_DIR / "campaign.py"
ENTRYPOINT_MODULE = f"{ROOT_PKG}.campaign"


# =============================================================================
# The declared campaign-#1 root set
# =============================================================================
#
# Each root is here because a real incident or a measured fact put it here. The
# reason strings are asserted against FOOTPRINT.md, so they cannot drift into
# decoration.

CAMPAIGN_ROOTS = {
    f"{ROOT_PKG}.artifact_diff":
        "AK-TR-6 must veto an unconfirmed GPU claim before behavioral T0 can launch",
    f"{ROOT_PKG}.candidate_record":
        "every executed candidate must be fsynced from the exact built snapshot and "
        "evaluation event identities before terminal STOP",
    f"{ROOT_PKG}.dashboard":
        "the terminal result was fsynced but the only dashboard exporter had been deleted, "
        "so active AutoKernel work remained permanently absent from the operator surface",
    f"{ROOT_PKG}.schemas":
        "one record shape; every other module is written against it",
    f"{ROOT_PKG}.journal":
        "AutoPilot lost 232 trials / ~16 days to a restart nothing objected to",
    f"{ROOT_PKG}.storage":
        "the 2026-07-04 async-prefetch win was written to /mnt/raid0/llm/tmp/ and that "
        "directory no longer exists",
    f"{ROOT_PKG}.source_candidate":
        "source-changing proposals consume one immutable embedded patch bundle through "
        "the guarded worktree mutation boundary",
    f"{ROOT_PKG}.source_prerequisite_package":
        "source candidates may rank only after archived raw sensitivity, hostile and "
        "checker CSV bytes are re-reduced and rebound to the exact live build identities",
    f"{ROOT_PKG}.evaluator.api":
        "a Verdict is constructible only via compute_verdict(); it cannot be stamped",
    f"{ROOT_PKG}.evaluator.correctness":
        "throughput is reward-hackable — deleting the computation is the fastest kernel "
        "there is",
    f"{ROOT_PKG}.evaluator.recipes":
        "argv from a hashed constructor; production drifted off NUMA interleave 2026-05-24 "
        "and the front door ended up at 46% of canonical",
    f"{ROOT_PKG}.evaluator.devices":
        "a GPU cell must not be satisfied by 'Device 0: CPU'",
    f"{ROOT_PKG}.evaluator.controls":
        "the A/A control plane — 2026-08-04 measured 1.62%/1.88% between-run CV and a "
        "MONOTONIC decode decline across four identical runs",
    f"{ROOT_PKG}.execution.worktree":
        "no candidate exists without it",
    f"{ROOT_PKG}.execution.microbench":
        "paired alternating blocks: 2026-08-04 A/A decode declined monotonically across "
        "four runs, so candidate-then-anchor charges the second arm ~4% systematically",
    f"{ROOT_PKG}.execution.physical_bounds":
        "RVP-C6-4 refuses physically impossible throughput before it can enter a rank",
    f"{ROOT_PKG}.execution.powercap_broker":
        "the v9 CPU preflight could not read root-owned 0400 package counters, while "
        "running the campaign as root correctly failed the candidate sandbox",
    f"{ROOT_PKG}.execution.t0_provider":
        "the predecessor harness tested MUL_MAT only, so a kernel that broke MUL_MAT_ID — "
        "MoE dispatch, every token in production — passed it cleanly",
    f"{ROOT_PKG}.execution.control_runner":
        "runs the neutral/A-A controls the 2026-08-04 drift makes mandatory",
    f"{ROOT_PKG}.execution.cpu_region_claim":
        "2026-08-04: two A/A runs destroyed by a legitimate co-tenant because we held no "
        "claim",
    f"{ROOT_PKG}.execution.chain":
        "holds the seams between the executors and the evaluator",
    f"{ROOT_PKG}.resource.device_claim":
        "the GPU sibling of the CPU region claim; §2.6's first row of missing substrate",
    f"{ROOT_PKG}.resource.preflight":
        "INC-20260731: a name-pattern kill took out another agent's server twice, and "
        "earlyoom, whose argv names what it guards",
    f"{ROOT_PKG}.resource.claim_witness":
        "a claim that is observed but never held is not a claim (invariant 9)",
    f"{ROOT_PKG}.controller.hypotheses":
        "`claim_for_hypothesis` calls itself the ONLY route from a hypothesis to a "
        "resource claim and had ZERO non-test callers until 2026-08-04 — a guard "
        "defined and never wired, the fifth of that shape in this package",
    f"{ROOT_PKG}.controller.do_not_repeat":
        "`authorize_claim(ledger=…)` has no default and a token with no do-not-repeat "
        "verdict is refused at the door, so the driver cannot mint a spendable one "
        "without the §19.2 ledger this module compiles",
}

#: The five callerless guards discovered before the lean-loop cut. This is a
#: source-level contract, not a prose checklist: if a caller disappears or is
#: renamed, the suite fails. The call fragment is intentionally specific enough
#: to distinguish using the guarded API from merely mentioning its symbol.
GUARD_CALLER_CONTRACT = {
    "worktree-mutating-subcommand refusal": (
        "execution/worktree.py", "if sub in self._WORKTREE_MUTATING_SUBCOMMANDS:"),
    "retry order reversal": (
        "execution/microbench.py", "statistics.OrderSchedule.derive("),
    "do-not-repeat ledger": (
        "controller/hypotheses.py", "verdict = check_do_not_repeat("),
    "per-control seed rotation": (
        "execution/control_runner.py", "plan = self.harness.seed_plan("),
    "falsifier before resource claim": (
        "campaign.py", "hypotheses.claim_for_hypothesis(spec.authorization, acquire)"),
}


# =============================================================================
# The deferred half
# =============================================================================

#: Prefixes the campaign path must not reach at all. Matching is on module
#: boundaries: "autokernel.release" matches "autokernel.release.t3" but never
#: "autokernel.release_notes".
DEFERRED = {
    f"{ROOT_PKG}.controller":
        "banned by PREFIX and opened module by module: see CONTROLLER_ALLOWED. The "
        "entry used to read 'read by whoever PROPOSES a candidate and by nothing that "
        "measures one', which was right about STRATEGY and wrong about CLAIMS — the "
        "driver is what SPENDS the claim, so the falsifier-before-compute gate belongs "
        "there, and while this entry banned the whole prefix `claim_for_hypothesis` (\"the "
        "ONLY route from a hypothesis to a resource claim\") had zero non-test callers. "
        "The prefix stays banned so that a module ADDED under `controller/` later is a "
        "finding rather than a silent re-admission",
    f"{ROOT_PKG}.release":
        "restored for AK9 speech release-plan compilation, but still needed only to SHIP "
        "a champion and deliberately unreachable from campaign #1",
    f"{ROOT_PKG}.adapters":
        "restored as pure AK9 speech declarations and release bindings; campaign #1 "
        "still searches llama_cpu and must not import them",
    f"{ROOT_PKG}.surface":
        "a freshness contract so a dead loop cannot read as fresh; the loop has never "
        "been alive",
}

#: Deferred prefixes the OPERATOR has deleted. Moving a prefix here is the whole
#: cost of acting on this boundary: the reachability assertion still holds (an
#: absent module is unreachable), and `test_the_deferred_half_is_still_on_disk`
#: stops requiring the files. Nothing else in this file changes.
DELETED_BY_OPERATOR: tuple = (
    f"{ROOT_PKG}.surface",
)

#: Removed 2026-08-04 on the operator's approval; recoverable from the tag
#: `autokernel-preserve-20260804`. The selected `release` compiler and speech
#: `adapters` were restored on 2026-08-12 for AK9 and remain DEFERRED above;
#: `surface` remains deleted.
#:
#: The edit above is the entire cost of acting on this boundary, exactly as this
#: file promised — but the removal was NOT free, and the bill is worth recording
#: because it is the refactoring lesson in miniature. `controller` could not go
#: wholesale: `hypotheses.py` and `do_not_repeat.py` are the operator's hypothesis
#: drop-in and the do-not-repeat memory, and they reached into the removed plane
#: for exactly SIX LINES — `state_machine.ControllerError` (a two-line base
#: exception), `fingerprint.selection_block()` (four lines) — plus
#: `selection.LEDGER_DIMENSIONS`, a constant describing what the LEDGER keys on
#: that had no business living in `selection.py`. Twenty thousand lines were
#: pinned by six. They now live in `controller/shared.py`, which is where a
#: concern shared by two modules belongs and where this package has never had a
#: place to put one.
#:
#: `controller` therefore does NOT appear above: four modules under it are still
#: on disk, so it stays in `DEFERRED` as a live prefix — which is also what keeps
#: this boundary from becoming decorative. `DEFERRED` and `DELETED_BY_OPERATOR`
#: are now disjoint sets with different jobs: the first names a plane the campaign
#: path must not reach and CAN still be reached (so the checks below have
#: something to bite on in the real tree), the second names prefixes that may
#: never come back onto the path at all.


# =============================================================================
# The one narrowing of a deferred prefix: an explicit, per-module allow-list
# =============================================================================
#
# 2026-08-04. `campaign.py` gained `--hypothesis`, and with it the
# falsifier-before-compute gate: the claim is acquired through
# `hypotheses.claim_for_hypothesis` so a question with no falsifier — or a
# placeholder one — cannot reach a claim. That gate had been written, documented
# as "the ONLY route from a hypothesis to a resource claim", and never called by
# anything but its own tests, because this boundary put it on the far side of the
# line. The line was in the wrong place: whoever PROPOSES a candidate is not the
# one that spends the card, and the driver is.
#
# THIS IS A LIST, NOT A PREFIX, and that is the entire design. Allowing
# `autokernel.controller.*` would re-admit anything a future session drops into
# that directory, silently and forever — the same shape as a `DEFERRED` entry
# whose targets were all deleted. Every module below is named, and every one has
# to say why the CAMPAIGN PATH (not the package, not the plane) needs it.
#
# `test_the_allow_list_is_exact` proves this is not a prefix, and
# `test_a_new_module_under_controller_is_still_caught` plants a real module under
# `controller/` in the copied tree and asserts the prefix ban still bites.
CONTROLLER_ALLOWED = {
    f"{ROOT_PKG}.controller":
        "the package __init__ itself. Importing ANY module under a package executes "
        "its `__init__`, so this row is not a choice — it is the consequence of the "
        "two rows below, stated rather than left as an unexplained edge",
    f"{ROOT_PKG}.controller.hypotheses":
        "`claim_for_hypothesis` — the gate `campaign.py` now acquires its region claim "
        "through. A falsifier is optional when the operator writes a question and "
        "MANDATORY before a claim is spent on it, and the spend happens in the driver",
    f"{ROOT_PKG}.controller.do_not_repeat":
        "NOT 'just in case', and the check was made before admitting it: "
        "`check_do_not_repeat(*, regime, matches)` is pure and the driver never calls "
        "it — but `HypothesisTracker.authorize_claim(ledger=…)` has NO default and "
        "`claim_for_hypothesis` raises `LedgerNotConsulted` on a token carrying no "
        "verdict, so no SPENDABLE token exists without a real ledger. "
        "`compile_for_tracker` builds it from the tracker's own record and lives here. "
        "It is also reached unconditionally via `controller/__init__.py`",
    f"{ROOT_PKG}.controller.shared":
        "`ControllerError`, `selection_block()` and `LEDGER_DIMENSIONS` — the six lines "
        "the two modules above were pinned to the removed AK4 plane by. Reached only "
        "through them; the campaign path names nothing in it",
}


# =============================================================================
# The symbol fence — modules that ARE reachable, with a deferred gate surface
# =============================================================================

#: `evaluator/integrity.py`: reachable for provenance and raw evidence
#: primitives only, never its §8.5.1 gate runners. A new name here is not a
#: style question: it needs a concrete campaign evidence consumer.
INTEGRITY_ALLOWED_NAMES = frozenset({
    # hashing / build identity — worktree.py's receipt and microbench.py's binary hash
    "sha256_file", "hash_source_tree", "TreeDigest", "BuildProvenance",
    "EMPTY_TREE_SHA256", "IntegrityError",
    # the anchor/symbol SEAM projections in chain.py: types and pure diff helpers,
    # none of which is a gate runner
    "DeclaredSymbolDeltas", "ElfSymbolTable", "RegistrationDiff", "RegistrationTable",
    "PatternRegistrationExtractor",
    "SourceDiff", "SymbolDiff", "KIND_DISPATCH_PREDICATE", "KIND_OP_REGISTRATION",
    "diff_registration_tables", "diff_symbol_tables", "extract_elf_symbols",
    "parse_mangled_name", "parse_unified_diff",
})

#: `evaluator/surface.py`: reachable for CHANGE-CLASS CONSTANTS only.
SURFACE_ALLOWED_NAMES = frozenset({
    "AffectedSurface", "FULL_TREE_CHANGE_CLASSES",
    "OA_CORE_HEADER_CHANGE_CLASS", "OA_SHARED_HEADER_FANOUT",
})

#: The second of the two coexisting §8.5.1 derivations. `integrity.py` implements
#: the gates over ELF tables, parsed diffs and build provenance; `correctness.py`
#: carries three shallower `t0.source_integrity.*` gates over self-declared
#: evidence objects. Both are `integrity`-class, so wiring both means either can
#: block ranking — and the deeper one consumes evidence NOTHING produces, which
#: makes it unsatisfiable rather than strict.
DEFERRED_GATE_RUNNERS = {
    f"{ROOT_PKG}.evaluator.integrity": ("SourceIntegrityGateRunner",
                                        "SourceIntegrityFirstRunner"),
    f"{ROOT_PKG}.evaluator.surface": ("SurfaceGateRunner",),
}


# =============================================================================
# The optional-stopping fence
# =============================================================================
#
# MEASURED 2026-08-04 (data/autokernel_aa_20260804/): four A/A runs of identical
# code on a quiet host gave between-run CV 1.62% (pp512) and 1.88% (tg128). A
# 1.6-1.9% CV does not justify an e-process; a median over paired deltas covers
# it. And the e-process that IS in the tree made the gate UNPASSABLE — the
# sign-martingale tops out at 5.5687 over five same-sign blocks against a
# calibrated threshold of 10, at four different effect magnitudes, because the
# construction is sign-based (README, "Remaining in AK3", item 1).
#
# `statistics.py` is therefore imported for CALIBRATION CONSTANTS ONLY. These are
# the names that would turn it back into a sequential procedure with interim
# looks, and this is the assertion that stops that creeping back in.

OPTIONAL_STOPPING_NAMES = frozenset({
    "SequentialEvaluation", "LookResult", "BlockRequest", "StopDecision",
    "run_e_process", "EProcessRun", "EProcessConstruction", "select_construction",
    "CONSTRUCTIONS", "null_boundary_for",
})

#: Reaching the machinery through an object that is legitimately bound. Without
#: this, `CampaignStatistics` — which microbench.py holds for `b_min`,
#: `threshold_for` and `order_schedule` — hands out a `SequentialEvaluation` from
#: a method call that no name-level check would ever see.
OPTIONAL_STOPPING_CALLS = frozenset({
    "sequential_evaluation", "next_block_request", "submit_block",
})

# The prospective writer evaluates the fixed, fully completed block vector once
# to populate the evaluator's required e-value field.  It cannot request a
# block, stop early, alter the campaign accept rule, or trigger an interim look.
# Keep this exception exact by module and symbol; every other binding remains a
# campaign-boundary defect.
FIXED_N_EVENT_REDUCTION = {
    f"{ROOT_PKG}.execution.control_runner": frozenset({
        "run_e_process", "select_construction",
    }),
}

STATISTICS_MODULE = f"{ROOT_PKG}.evaluator.statistics"
ACCEPT_RULE_MODULE = f"{ROOT_PKG}.evaluator.api"


# =============================================================================
# The walker
# =============================================================================


#: AST cache, keyed on (path, mtime, size) so a rewritten probe file is never a
#: stale hit. Module-level, not per-graph: `TestTheBoundaryCatchesRealTreeViolations`
#: builds a fresh `ImportGraph` over the same copied 95k-line tree for every
#: planted mutation, and re-parsing it each time was most of this file's runtime.
_TREE_CACHE: dict = {}


class BoundaryError(Exception):
    """A module could not be parsed or resolved while walking the graph."""


class ImportGraph:
    """Transitive import closure of a package, from the AST.

    Parameterised on (`pkg_dir`, `root`) rather than hard-wired to this package,
    so the walker itself can be bite-verified against synthetic trees that DO
    violate the boundary. A guard nobody has ever seen fail is a guard nobody has
    ever tested.

    Four things it follows that a grep does not:

    1. **Relative imports at any level** — `from .. import schemas` inside
       `execution/chain.py`.
    2. **Function-level imports** — `chain.py` line 819 does exactly this to
       break a cycle, so a module-level-only walk would miss a whole edge.
    3. **Parent-package `__init__` side effects** — importing
       `autokernel.surface.anything` executes `surface/__init__.py`, which
       imports `dashboard_contract`. The edge is real and invisible in the
       importing module's own source.
    4. **Dynamic imports by string** — `importlib.import_module("autokernel.…")`
       is the obvious way around an AST check, so the string literals are read
       too.
    """

    def __init__(self, pkg_dir: Path, root: str) -> None:
        self.pkg_dir = Path(pkg_dir)
        self.root = root
        self._edges: dict = {}
        #: Dynamic-import calls whose module argument this walk CANNOT resolve
        #: (an f-string, a concatenation, a variable). Recorded rather than
        #: skipped: silently dropping the one construct that defeats an AST walk
        #: is a fail-open, and it produces a success-shaped result.
        self.unresolved: list = []

    # -- module <-> path ----------------------------------------------------

    def module_path(self, module: str):
        """Path backing `module`, or None if it names something that is not a module.

        `from .evaluator import api` yields both `autokernel.evaluator` (a
        package) and `autokernel.evaluator.api` (a module); `from ..evaluator
        import integrity` also yields names like
        `autokernel.evaluator.integrity.sha256_file` when a caller imports a
        FUNCTION. Returning None for the last of those is what keeps a class name
        from being mistaken for a module.
        """
        parts = module.split(".")
        if not parts or parts[0] != self.root:
            return None
        base = self.pkg_dir.joinpath(*parts[1:])
        module_file = base.with_suffix(".py")
        if module_file.is_file():
            return module_file
        init_file = base / "__init__.py"
        if init_file.is_file():
            return init_file
        return None

    @staticmethod
    def _ancestors(module: str):
        parts = module.split(".")
        for i in range(1, len(parts)):
            yield ".".join(parts[:i])

    # -- one module's edges -------------------------------------------------

    def _resolve_from(self, node: ast.ImportFrom, module: str, is_package: bool) -> list:
        if node.level:
            parts = module.split(".")
            if not is_package:
                parts = parts[:-1]
            up = node.level - 1
            if up:
                if up > len(parts):
                    raise BoundaryError(
                        f"{module}: relative import escapes the package root "
                        f"(level={node.level})")
                parts = parts[: len(parts) - up]
            prefix = ".".join(parts)
            if node.module:
                prefix = f"{prefix}.{node.module}" if prefix else node.module
        else:
            prefix = node.module or ""
        if not prefix or prefix.split(".")[0] != self.root:
            return []
        out = [prefix]
        for alias in node.names:
            if alias.name != "*":
                out.append(f"{prefix}.{alias.name}")
        return out

    #: Functions whose FIRST argument is a module name. `module_from_spec` is
    #: deliberately absent: it takes a spec object, never a name, so listing it
    #: would report every use of it as unresolvable.
    _DYNAMIC_IMPORT_FUNCS = frozenset({"import_module", "__import__", "find_spec"})

    def _resolve_relative_string(self, spec: str, module: str, is_package: bool):
        """`import_module(".controller.guards", __package__)` — same arithmetic as
        a `from . import` at the same level. Absolute strings were already read;
        relative ones were NOT, and they are the cheaper bypass of the two."""
        level = len(spec) - len(spec.lstrip("."))
        rest = spec[level:]
        parts = module.split(".")
        if not is_package:
            parts = parts[:-1]
        up = level - 1
        if up > len(parts):
            return None
        if up:
            parts = parts[: len(parts) - up]
        prefix = ".".join(parts)
        if not prefix or prefix.split(".")[0] != self.root:
            return None
        return f"{prefix}.{rest}" if rest else prefix

    def _dynamic_imports(self, tree: ast.AST, module: str, is_package: bool) -> list:
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None)
            if name not in self._DYNAMIC_IMPORT_FUNCS:
                continue
            args = list(node.args) + [kw.value for kw in node.keywords
                                      if kw.arg in (None, "name")]
            if not args:
                continue
            target = args[0]
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                spec = target.value
                if spec.startswith("."):
                    # `import_module(name, package)` anchors on whatever string is
                    # passed as `package` — `__package__` and `__name__` differ by
                    # exactly one component and BOTH appear in the wild. Emit both
                    # readings: `closure()` drops the one that names no module, so
                    # over-emitting is the safe direction for a guard.
                    out.extend(r for r in (
                        self._resolve_relative_string(spec, module, True),
                        self._resolve_relative_string(spec, module, is_package))
                        if r is not None)
                elif spec.split(".")[0] == self.root:
                    out.append(spec)
            else:
                finding = (f"{module}:{node.lineno} calls {name}(…) with a module name this "
                           "walk cannot resolve; the boundary cannot see where it goes")
                if finding not in self.unresolved:
                    self.unresolved.append(finding)
        return out

    def imports_of(self, module: str) -> list:
        path = self.module_path(module)
        if path is None:
            raise BoundaryError(f"{module} has no module file under {self.pkg_dir}")
        tree = self.parse(module)
        is_package = path.name == "__init__.py"
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == self.root:
                        found.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                found.update(self._resolve_from(node, module, is_package))
        found.update(self._dynamic_imports(tree, module, is_package))
        return sorted(found)

    def parse(self, module: str) -> ast.AST:
        """Cached on (path, mtime, size): the checks below walk the same 95k-line
        closure three times over, and re-parsing it each time was 9s of the 11s
        this file cost the suite."""
        path = self.module_path(module)
        if path is None:
            raise BoundaryError(f"{module} has no module file under {self.pkg_dir}")
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = _TREE_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - a broken tree is a hard stop
            raise BoundaryError(f"{module}: {exc}") from exc
        _TREE_CACHE[key] = tree
        return tree

    # -- the closure --------------------------------------------------------

    def closure(self, roots) -> dict:
        """`{module: [modules it imports]}` over everything reachable from `roots`.

        Only real modules become keys. Names that resolve to nothing (a class
        imported from a module, a stdlib collision) are dropped, and a root that
        does not exist is dropped too — `test_every_declared_campaign_root_exists`
        is what makes a typo in `CAMPAIGN_ROOTS` fail as a missing root rather
        than as a silently smaller graph.
        """
        edges: dict = {}
        stack = list(roots)
        while stack:
            module = stack.pop()
            if module in edges:
                continue
            if self.module_path(module) is None:
                continue
            deps = self.imports_of(module)
            edges[module] = deps
            stack.extend(deps)
            stack.extend(a for a in self._ancestors(module) if a not in edges)
        return edges

    # -- queries ------------------------------------------------------------

    @staticmethod
    def under(module: str, prefix: str) -> bool:
        return module == prefix or module.startswith(prefix + ".")

    def importers_of(self, edges: dict, prefix: str) -> list:
        return sorted(
            importer for importer, deps in edges.items()
            if any(self.under(dep, prefix) for dep in deps)
        )

    @staticmethod
    def _dotted(node: ast.AST):
        """`evaluator.integrity.SourceIntegrityGateRunner` -> the three parts.

        None if the chain does not bottom out in a plain name (a subscript, a
        call result), because then it names nothing this walk can resolve.
        """
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name) or not parts:
            return None
        parts.append(node.id)
        parts.reverse()
        return parts

    def module_aliases(self, module: str) -> dict:
        """`{local name: the dotted thing it names}` for every in-package binding.

        `from . import evaluator` binds `evaluator` -> `autokernel.evaluator`,
        which is the whole point: `evaluator.integrity.X` is then a two-hop
        attribute chain that an alias-is-a-bare-Name check cannot see, and the
        chain WORKS at runtime because something else on the path already
        imported `evaluator.integrity`.
        """
        is_package = self.module_path(module).name == "__init__.py"
        out: dict = {}
        for node in ast.walk(self.parse(module)):
            if isinstance(node, ast.ImportFrom):
                resolved = self._resolve_from(node, module, is_package)
                if not resolved:
                    continue
                prefix = resolved[0]
                for alias in node.names:
                    if alias.name != "*":
                        out[alias.asname or alias.name] = f"{prefix}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] != self.root:
                        continue
                    if alias.asname:
                        out[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        out[head] = head
        return out

    def names_bound_from(self, edges: dict, target: str) -> dict:
        """`{importer: {name, …}}` — every attribute of `target` any module binds.

        Four idioms are read, and the last two are the ones a single-hop check
        misses:

        1. `from ..evaluator.integrity import sha256_file` — a direct name import.
        2. `from ..evaluator import integrity` then `integrity.sha256_file(…)`.
        3. `from .. import evaluator` then `evaluator.integrity.SourceIntegrityGateRunner`
           — a CHAINED alias. Runnable, because `worktree.py` has already made
           `evaluator.integrity` an attribute of the package.
        4. `getattr(integrity, "SourceIntegrityGateRunner")` — the three-character
           edit that turns an attribute into a string.

        The target module itself is excluded — a module is allowed to contain its
        own machinery; the question is who REACHES it.
        """
        used: dict = {}
        for module in edges:
            if module == target or self.module_path(module) is None:
                continue
            tree = self.parse(module)
            is_package = self.module_path(module).name == "__init__.py"
            aliases = self.module_aliases(module)
            names: set = set()
            for node in ast.walk(tree):
                # Idiom 1 — the imported NAME, not whatever it was aliased to.
                if isinstance(node, ast.ImportFrom):
                    resolved = self._resolve_from(node, module, is_package)
                    if resolved and resolved[0] == target:
                        names.update(a.name for a in node.names if a.name != "*")
                # Idioms 2 and 3 — one alias hop or several.
                elif isinstance(node, ast.Attribute):
                    self._bind(names, aliases, target, self._dotted(node))
                # Idiom 4 — the attribute spelled as a string.
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "getattr" and len(node.args) >= 2 \
                        and isinstance(node.args[1], ast.Constant) \
                        and isinstance(node.args[1].value, str):
                    obj = node.args[0]
                    chain = [obj.id] if isinstance(obj, ast.Name) else self._dotted(obj)
                    self._bind(names, aliases, target, chain, node.args[1].value)
            if names:
                used[module] = names
        return used

    @staticmethod
    def _bind(names: set, aliases: dict, target: str, chain, extra=None) -> None:
        """Record `chain` (plus a `getattr` string) if it lands inside `target`."""
        if not chain or chain[0] not in aliases:
            return
        full = ".".join([aliases[chain[0]], *chain[1:]] + ([extra] if extra else []))
        if full.startswith(target + "."):
            names.add(full[len(target) + 1:].split(".")[0])

    def attribute_uses(self, edges: dict, *, exclude=()) -> dict:
        """`{importer: {attr, …}}` — every `x.attr` reference, by attribute name.

        Name-level checks cannot see `campaign_statistics.sequential_evaluation()`.
        This can.

        REFERENCES, not just calls: `look = stats.sequential_evaluation` followed
        by `look(...)` is the same interim look one line apart, and a Call-only
        scan reads it as clean. Zero modules on the campaign path reference any
        forbidden name in either form, so widening this costs no false positive.
        """
        out: dict = {}
        for module in edges:
            if module in exclude or self.module_path(module) is None:
                continue
            names = {node.attr for node in ast.walk(self.parse(module))
                     if isinstance(node, ast.Attribute)}
            if names:
                out[module] = names
        return out


def campaign_roots() -> list:
    """The declared roots, plus the entrypoint.

    The entrypoint is UNCONDITIONAL. It was once guarded by `is_file()`, which
    meant deleting or renaming `campaign.py` narrowed this boundary to the
    declared roots and left the suite green — the same "pass by deleting what
    the guard inspects" hole this file refuses to leave open for `DEFERRED`.
    `TestCampaignFootprint.test_the_entrypoint_exists` is the loud failure now.
    """
    return [*sorted(CAMPAIGN_ROOTS), ENTRYPOINT_MODULE]


#: One walk of the real tree, shared by every class below. Four independent
#: `setUpClass` walks of a 95k-line closure cost ~20s of suite time for four
#: identical answers.
_REAL_GRAPH = ImportGraph(PKG_DIR, ROOT_PKG)
_REAL_EDGES: dict = {}


def campaign_edges() -> dict:
    if not _REAL_EDGES:
        _REAL_EDGES.update(_REAL_GRAPH.closure(campaign_roots()))
    return _REAL_EDGES


# =============================================================================
# The three checks, as functions over (graph, edges)
# =============================================================================
#
# Factored out so the SAME code that guards this package can be pointed at a
# COPY of it with a violation planted in it (`TestTheBoundaryCatchesRealTreeViolations`).
# A guard verified only against toy fixtures has been verified against toy
# fixtures.


def deferred_findings(graph: "ImportGraph", edges: dict) -> list:
    """Every deferred module the campaign path reaches, minus the named exceptions.

    The exception set is `CONTROLLER_ALLOWED`, matched by EXACT MODULE NAME. A
    membership test rather than a prefix test is the whole of the narrowing: a
    prefix allowance would re-admit every module added under `controller/` after
    today, which is the failure mode this file exists to make loud.
    """
    allowed = {name.replace(ROOT_PKG, graph.root, 1) for name in CONTROLLER_ALLOWED}
    findings = []
    for prefix, reason in sorted(DEFERRED.items()):
        prefix = prefix.replace(ROOT_PKG, graph.root, 1)
        for module in sorted(m for m in edges
                             if graph.under(m, prefix) and m not in allowed):
            findings.append(
                f"{module} is reachable from the campaign path via "
                f"{graph.importers_of(edges, module)} — deferred because: {reason}")
    return findings


def provenance_fence_findings(graph: "ImportGraph", edges: dict) -> list:
    findings = []
    for leaf, allowed in (("evaluator.integrity", INTEGRITY_ALLOWED_NAMES),
                          ("evaluator.surface", SURFACE_ALLOWED_NAMES)):
        target = f"{graph.root}.{leaf}"
        used = graph.names_bound_from(edges, target)
        bound = set().union(*used.values()) if used else set()
        for name in sorted(bound - allowed):
            findings.append(f"{target}.{name} is bound on the campaign path and is not a "
                            "provenance primitive")
    for suffix, runners in sorted(DEFERRED_GATE_RUNNERS.items()):
        target = suffix.replace(ROOT_PKG, graph.root, 1)
        used = graph.names_bound_from(edges, target)
        bound = set().union(*used.values()) if used else set()
        for runner in runners:
            if runner in bound:
                findings.append(f"{target}.{runner} is wired on the campaign path")
    return findings


def optional_stopping_findings(graph: "ImportGraph", edges: dict) -> list:
    target = f"{graph.root}.evaluator.statistics"
    findings = []
    for module, names in sorted(graph.names_bound_from(edges, target).items()):
        allowed = FIXED_N_EVENT_REDUCTION.get(module, frozenset())
        for name in sorted((names & OPTIONAL_STOPPING_NAMES) - allowed):
            findings.append(f"{module} binds {target}.{name}")
    for module, names in sorted(graph.attribute_uses(edges, exclude=(target,)).items()):
        for name in sorted(names & OPTIONAL_STOPPING_CALLS):
            findings.append(f"{module} calls .{name}(…) — that is an interim look")
    return findings


def unresolved_import_findings(graph: "ImportGraph", _edges: dict) -> list:
    """Dynamic imports the walk could not follow. Empty is the only passing answer.

    This is the fail-open the other three checks would otherwise sit on: a walk
    that silently skips `import_module(f"{__package__}.controller.guards")`
    reports a clean boundary over a module it never looked at.
    """
    return list(graph.unresolved)


def _write(root: Path, rel: str, source: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


# =============================================================================
# The walker, bite-verified against synthetic trees
# =============================================================================


class TestTheWalkerItself(unittest.TestCase):
    """Every case below is a graph the real boundary test must be able to catch.

    `test_a_compliant_graph_is_accepted` is the compliant-path control: a guard
    that fires on everything proves nothing, and this project has shipped one of
    those before (`integrity`'s unsatisfiable derivation).
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.pkg = self.root / "fakepkg"
        _write(self.root, "fakepkg/__init__.py", "")
        _write(self.root, "fakepkg/denied/__init__.py", "")
        _write(self.root, "fakepkg/denied/thing.py", "VALUE = 1\n")
        self.graph = ImportGraph(self.pkg, "fakepkg")

    def reached(self, *roots) -> set:
        edges = self.graph.closure(list(roots))
        return {m for m in edges if self.graph.under(m, "fakepkg.denied")}

    def test_a_compliant_graph_is_accepted(self):
        """CONTROL: a clean chain must produce no finding."""
        _write(self.root, "fakepkg/entry.py", """
            from __future__ import annotations
            from . import helper
        """)
        _write(self.root, "fakepkg/helper.py", "VALUE = 2\n")
        self.assertEqual(self.reached("fakepkg.entry"), set())

    def test_a_direct_import_of_a_denied_module_is_caught(self):
        _write(self.root, "fakepkg/entry.py", "from .denied import thing\n")
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_a_transitive_import_is_caught(self):
        """entry -> helper -> denied. The edge nobody reviewing entry.py sees."""
        _write(self.root, "fakepkg/entry.py", "from . import helper\n")
        _write(self.root, "fakepkg/helper.py", "from .denied import thing\n")
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_a_function_level_import_is_caught(self):
        """`chain.py` uses this idiom at line 819 to break a cycle."""
        _write(self.root, "fakepkg/entry.py", """
            def later():
                from .denied import thing  # noqa: PLC0415
                return thing.VALUE
        """)
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_an_import_inside_a_try_block_is_caught(self):
        _write(self.root, "fakepkg/entry.py", """
            try:
                from .denied import thing
            except ImportError:
                thing = None
        """)
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_a_parent_package_init_side_effect_is_caught(self):
        """`surface/__init__.py` imports `dashboard_contract`; importing ANY module
        under `surface` therefore executes it. The importing module's own source
        shows nothing."""
        _write(self.root, "fakepkg/sub/__init__.py", "from ..denied import thing\n")
        _write(self.root, "fakepkg/sub/leaf.py", "VALUE = 3\n")
        _write(self.root, "fakepkg/entry.py", "from .sub import leaf\n")
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_a_two_level_relative_import_is_resolved(self):
        _write(self.root, "fakepkg/sub/__init__.py", "")
        _write(self.root, "fakepkg/sub/leaf.py", "from ..denied import thing\n")
        _write(self.root, "fakepkg/entry.py", "from .sub import leaf\n")
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_a_dynamic_import_by_string_is_caught(self):
        """The obvious way around an AST check."""
        _write(self.root, "fakepkg/entry.py", """
            import importlib

            def later():
                return importlib.import_module("fakepkg.denied.thing")
        """)
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_a_class_name_is_not_mistaken_for_a_module(self):
        """`from .helper import Thing` must not invent `fakepkg.helper.Thing`."""
        _write(self.root, "fakepkg/helper.py", "class Thing:\n    pass\n")
        _write(self.root, "fakepkg/entry.py", "from .helper import Thing\n")
        edges = self.graph.closure(["fakepkg.entry"])
        self.assertIn("fakepkg.helper", edges)
        self.assertNotIn("fakepkg.helper.Thing", edges)

    def test_a_similarly_named_sibling_is_not_a_false_positive(self):
        """CONTROL: prefix matching is on module boundaries, not on characters."""
        _write(self.root, "fakepkg/denied_notreally.py", "VALUE = 4\n")
        _write(self.root, "fakepkg/entry.py", "from . import denied_notreally\n")
        self.assertEqual(self.reached("fakepkg.entry"), set())

    def test_names_bound_from_reads_both_import_idioms(self):
        _write(self.root, "fakepkg/target.py", "def alpha():\n    pass\n\n\ndef beta():\n    pass\n")
        _write(self.root, "fakepkg/via_alias.py", """
            from . import target
            X = target.alpha
        """)
        _write(self.root, "fakepkg/via_name.py", "from .target import beta\n")
        _write(self.root, "fakepkg/entry.py", "from . import via_alias, via_name\n")
        edges = self.graph.closure(["fakepkg.entry"])
        used = self.graph.names_bound_from(edges, "fakepkg.target")
        self.assertEqual(used.get("fakepkg.via_alias"), {"alpha"})
        self.assertEqual(used.get("fakepkg.via_name"), {"beta"})

    def test_attribute_uses_sees_a_method_reached_through_an_object(self):
        _write(self.root, "fakepkg/entry.py", """
            def go(bundle):
                return bundle.sequential_evaluation(candidate_id="x")
        """)
        edges = self.graph.closure(["fakepkg.entry"])
        calls = self.graph.attribute_uses(edges)
        self.assertIn("sequential_evaluation", calls.get("fakepkg.entry", set()))

    def test_attribute_uses_is_quiet_on_a_compliant_module(self):
        """CONTROL."""
        _write(self.root, "fakepkg/entry.py", """
            def go(bundle):
                return bundle.threshold_for("selection")
        """)
        edges = self.graph.closure(["fakepkg.entry"])
        calls = self.graph.attribute_uses(edges)
        self.assertEqual(
            calls.get("fakepkg.entry", set()) & OPTIONAL_STOPPING_CALLS, set())

    # -- the four ways a SYMBOL fence is reached without a bare alias ---------

    def _target_names(self, target: str = "fakepkg.sub.target") -> set:
        edges = self.graph.closure(["fakepkg.entry"])
        used = self.graph.names_bound_from(edges, target)
        return set().union(*used.values()) if used else set()

    def _sub_target(self) -> None:
        _write(self.root, "fakepkg/sub/__init__.py", "")
        _write(self.root, "fakepkg/sub/target.py", "class Forbidden:\n    pass\n\n\ndef ok():\n    pass\n")

    def test_a_chained_module_alias_is_seen(self):
        """`from . import sub` then `sub.target.Forbidden` — two hops, no bare alias.

        Runnable: something else on the path has already imported `sub.target`,
        which is what makes it an attribute of `sub`.
        """
        self._sub_target()
        _write(self.root, "fakepkg/entry.py", """
            from . import sub
            from .sub import target  # what makes the attribute exist at runtime

            def go():
                return sub.target.Forbidden
        """)
        self.assertIn("Forbidden", self._target_names())

    def test_a_getattr_by_string_is_seen(self):
        """The three-character edit that turns an attribute into a string."""
        self._sub_target()
        _write(self.root, "fakepkg/entry.py", """
            from .sub import target

            def go():
                return getattr(target, "Forbidden")
        """)
        self.assertIn("Forbidden", self._target_names())

    def test_a_getattr_through_a_chained_alias_is_seen(self):
        self._sub_target()
        _write(self.root, "fakepkg/entry.py", """
            from . import sub
            from .sub import target  # noqa: F401

            def go():
                return getattr(sub.target, "Forbidden")
        """)
        self.assertIn("Forbidden", self._target_names())

    def test_a_permitted_name_through_a_chained_alias_is_not_a_finding(self):
        """CONTROL: chaining is not itself the violation — the NAME is."""
        self._sub_target()
        _write(self.root, "fakepkg/entry.py", """
            from . import sub
            from .sub import target  # noqa: F401

            def go():
                return sub.target.ok()
        """)
        bound = self._target_names()
        self.assertIn("ok", bound)
        self.assertNotIn("Forbidden", bound)

    # -- dynamic imports the walk must either follow or REPORT ----------------

    def test_a_relative_dynamic_import_string_is_caught(self):
        _write(self.root, "fakepkg/entry.py", """
            import importlib

            def later():
                return importlib.import_module(".denied.thing", __package__)
        """)
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_a_relative_dynamic_import_anchored_on_dunder_name_is_caught(self):
        """`__name__` and `__package__` differ by one component; both are used."""
        _write(self.root, "fakepkg/entry.py", """
            import importlib

            def later():
                return importlib.import_module("..denied.thing", __name__)
        """)
        self.assertIn("fakepkg.denied.thing", self.reached("fakepkg.entry"))

    def test_a_non_literal_dynamic_import_is_reported_not_skipped(self):
        """FAIL-CLOSED: an f-string the walk cannot follow must not read as clean."""
        _write(self.root, "fakepkg/entry.py", """
            import importlib

            def later():
                return importlib.import_module(f"{__package__}.denied.thing")
        """)
        self.assertEqual(self.reached("fakepkg.entry"), set(),
                         "an f-string is not resolvable; it must surface as unresolved")
        self.assertTrue(any("import_module" in u for u in self.graph.unresolved),
                        f"unreported: {self.graph.unresolved}")

    def test_a_literal_dynamic_import_reports_nothing_unresolved(self):
        """CONTROL: a guard that flags every `import_module(…)` flags nothing."""
        _write(self.root, "fakepkg/entry.py", """
            import importlib

            def later():
                return importlib.import_module("fakepkg.denied.thing")
        """)
        self.reached("fakepkg.entry")
        self.assertEqual(self.graph.unresolved, [])

    def test_an_out_of_package_dynamic_import_reports_nothing_unresolved(self):
        """CONTROL: `recipes.py` imports `scripts.lib.canonical_recipe` this way."""
        _write(self.root, "fakepkg/entry.py", """
            import importlib

            def later():
                return importlib.import_module("scripts.lib.canonical_recipe")
        """)
        self.reached("fakepkg.entry")
        self.assertEqual(self.graph.unresolved, [])


# =============================================================================
# The boundary itself
# =============================================================================


class TestCampaignFootprint(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _REAL_GRAPH
        cls.edges = campaign_edges()

    def test_every_declared_campaign_root_exists(self):
        """A renamed root must fail here, not shrink the graph in silence."""
        missing = [m for m in CAMPAIGN_ROOTS if self.graph.module_path(m) is None]
        self.assertEqual(missing, [], f"declared campaign roots with no module file: {missing}")

    def test_the_declared_roots_are_all_in_the_walked_graph(self):
        for module in CAMPAIGN_ROOTS:
            self.assertIn(module, self.edges)

    def test_campaign_path_does_not_reach_the_deferred_half(self):
        """THE assertion. Roughly half the package, provably inert on this path."""
        findings = deferred_findings(self.graph, self.edges)
        self.assertEqual(findings, [], "\n".join(findings))

    def test_the_deferred_half_is_still_on_disk(self):
        """Anti-vacuity: this boundary must not pass because its targets vanished.

        A guard you can satisfy by deleting what it inspects measures nothing.
        When the operator DOES delete a plane, its prefix moves to
        `DELETED_BY_OPERATOR` — that edit is the whole cost of acting on this
        boundary, and it is deliberately one line.
        """
        for prefix in sorted(DEFERRED):
            if prefix in DELETED_BY_OPERATOR:
                self.assertIsNone(
                    self.graph.module_path(prefix),
                    f"{prefix} is listed as deleted by the operator but is still on disk")
                continue
            self.assertIsNotNone(
                self.graph.module_path(prefix),
                f"{prefix} is neither on disk nor listed in DELETED_BY_OPERATOR; if it was "
                "deleted, say so there — if it was RENAMED, this boundary is now a no-op")

    #: RE-PINNED 2026-08-04, against the tree the operator's deletion left behind.
    #:
    #: The bound is not a taste and it is not a round number carried over from the
    #: 46k era — 40,000 was calibrated against a deferred half that no longer
    #: exists, and after the deletion it could only be satisfied by files that are
    #: gone. What is left under a live `DEFERRED` prefix is `controller/`:
    #: `hypotheses.py` (~4,500 non-test lines) and `do_not_repeat.py` (~2,200),
    #: plus `__init__.py` and `shared.py`, measured at 6,913 lines on this date.
    #:
    #: The derivation, so the number can be re-derived rather than trusted: the
    #: property being defended is "the ALLOW-LISTED plane is still a plane, where
    #: the allow-list says it is". The cheapest way to break it is to move the
    #: SMALLER of the two modules out from under the prefix, which would leave
    #: 6,913 - 2,205 = 4,708. So the bound sits above that and below the tree, at
    #: 5,000: either module moving fails this check, and ~1,900 lines of ordinary
    #: editing churn — by any of the sessions that share this clone — does not.
    #:
    #: SECOND CORRECTION, same day: the assertion below used to be called "the
    #: deferred half is a real share of the tree", and after `--hypothesis` was
    #: wired that name is false — every module under the one live `DEFERRED`
    #: prefix is now named in `CONTROLLER_ALLOWED`, so the deferred half is empty.
    #: The MEASUREMENT is unchanged and still bites for its original reason (a
    #: module leaving `controller/` collapses the count); only the claim it makes
    #: has been corrected to the one it can support.
    CONTROLLER_PLANE_FLOOR = 5_000

    def test_the_allow_listed_plane_is_a_real_share_of_the_tree(self):
        """Anti-vacuity for the ALLOW-LIST, which is now the exception that matters.

        If a refactor moved `hypotheses.py` out of `controller/` into a module the
        campaign path already imports, every reachability assertion above would
        still pass — the module would simply be somewhere no prefix bans — and the
        one narrowing this boundary grants would have quietly become a hole.
        """
        lines = 0
        for prefix in sorted(DEFERRED):
            if prefix in DELETED_BY_OPERATOR:
                continue
            sub = PKG_DIR.joinpath(*prefix.split(".")[1:])
            for path in sub.rglob("*.py"):
                if not path.name.startswith("test_"):
                    lines += len(path.read_text(encoding="utf-8").splitlines())
        self.assertGreater(lines, self.CONTROLLER_PLANE_FLOOR,
                           f"the live deferred prefix is under "
                           f"{self.CONTROLLER_PLANE_FLOOR:,} lines; either it was deleted "
                           "(say so in DELETED_BY_OPERATOR) or a module moved out from "
                           "under the allow-list")

    # -- the allow-list itself ---------------------------------------------

    def test_every_allow_listed_module_is_real(self):
        """An allow-list naming a module that does not exist grants nothing and
        hides the fact that the thing it was granted for has moved."""
        missing = [m for m in sorted(CONTROLLER_ALLOWED)
                   if self.graph.module_path(m) is None]
        self.assertEqual(missing, [],
                         f"CONTROLLER_ALLOWED names modules with no file: {missing}")

    def test_every_allow_listed_module_sits_under_a_deferred_prefix(self):
        """The allow-list is an EXCEPTION to a ban. A row naming something no ban
        covers is not an exception; it is a comment that reads like a rule."""
        stray = [m for m in sorted(CONTROLLER_ALLOWED)
                 if not any(self.graph.under(m, p) for p in DEFERRED)]
        self.assertEqual(stray, [], f"CONTROLLER_ALLOWED rows outside DEFERRED: {stray}")

    def test_every_allow_listed_module_states_why_the_campaign_path_needs_it(self):
        thin = [m for m, why in sorted(CONTROLLER_ALLOWED.items()) if len(why) < 40]
        self.assertEqual(thin, [], f"allow-list rows with no argument: {thin}")

    def test_the_allow_list_is_exact_and_not_a_prefix(self):
        """THE BITE, stated as an assertion rather than as a comment.

        A sibling module under the same package must NOT be allowed by the fact
        that its neighbours are. This is checked against the allow-list's own
        matching rule, so it fails the moment someone rewrites the membership
        test as `startswith`.
        """
        allowed = set(CONTROLLER_ALLOWED)
        for module in sorted(allowed):
            sibling = module + ".revenant"
            self.assertNotIn(sibling, allowed,
                             f"{sibling} is allowed by prefix; the list must be exact")
        edges = dict(self.edges)
        edges[f"{ROOT_PKG}.controller.revenant"] = []
        self.assertTrue(deferred_findings(self.graph, edges),
                        "a module under an allow-listed package produced no finding; "
                        "the allow-list has become a prefix allowance")

    def test_the_entrypoint_exists(self):
        """The boundary is drawn AROUND an entrypoint; without one it guards nothing.

        This was a `skipUnless` on every entrypoint assertion below, which meant
        `rm campaign.py` turned the boundary into a green no-op.
        """
        self.assertTrue(ENTRYPOINT_PATH.is_file(),
                        f"{ENTRYPOINT_PATH} is gone; 94k lines with no way to start it is "
                        "the exact state this boundary exists to end")

    def test_no_dynamic_import_on_the_campaign_path_is_unresolvable(self):
        """FAIL-CLOSED: every check above is a statement about a graph that was WALKED.

        A dynamic import the walk cannot follow is a hole in that graph, and
        skipping it silently would let all three checks report clean over a
        module nobody looked at.
        """
        campaign_edges()
        self.assertEqual(self.graph.unresolved, [], "\n".join(self.graph.unresolved))

    def test_the_entrypoint_stays_inside_the_declared_closure(self):
        """campaign.py may only reach what the declared roots already reach."""
        declared = set(ImportGraph(PKG_DIR, ROOT_PKG).closure(sorted(CAMPAIGN_ROOTS)))
        entry = set(ImportGraph(PKG_DIR, ROOT_PKG).closure([ENTRYPOINT_MODULE]))
        extra = sorted(entry - declared - {ENTRYPOINT_MODULE})
        self.assertEqual(
            extra, [],
            f"{ENTRYPOINT_MODULE} reaches modules outside the declared campaign-#1 set: "
            f"{extra}. Either the module belongs in CAMPAIGN_ROOTS with a real incident or "
            "a measured fact as its reason, or campaign #1 does not need it.")


# =============================================================================
# The symbol fence over the two reachable §8.5.1 modules
# =============================================================================


class TestTheProvenanceFence(unittest.TestCase):
    """`integrity.py` and `surface.py` are reachable, and only as primitives."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _REAL_GRAPH
        cls.edges = campaign_edges()

    def _bound(self, target: str) -> set:
        used = self.graph.names_bound_from(self.edges, target)
        return set().union(*used.values()) if used else set()

    def test_the_fenced_modules_really_are_reachable(self):
        """Anti-vacuity: an allowlist over an unreachable module says nothing."""
        for target in DEFERRED_GATE_RUNNERS:
            self.assertIn(target, self.edges,
                          f"{target} is no longer on the campaign path — if that is real, "
                          "move it into DEFERRED and delete this fence")

    def test_only_provenance_primitives_are_bound_from_integrity(self):
        target = f"{ROOT_PKG}.evaluator.integrity"
        extra = sorted(self._bound(target) - INTEGRITY_ALLOWED_NAMES)
        self.assertEqual(
            extra, [],
            f"new names bound from {target} on the campaign path: {extra}. This module is "
            "reachable for hashing and build identity only; its §8.5.1 gate derivation is "
            "the deferred half, and T0 today produces nine of seventeen surfaces, so those "
            "gates cannot be satisfied.")

    def test_only_change_class_constants_are_bound_from_surface(self):
        target = f"{ROOT_PKG}.evaluator.surface"
        extra = sorted(self._bound(target) - SURFACE_ALLOWED_NAMES)
        self.assertEqual(extra, [], f"new names bound from {target}: {extra}")

    def test_no_deferred_gate_runner_is_reachable(self):
        for target, runners in sorted(DEFERRED_GATE_RUNNERS.items()):
            bound = self._bound(target)
            for runner in runners:
                self.assertNotIn(
                    runner, bound,
                    f"{target}.{runner} is wired on the campaign path. Two derivations of "
                    "the §8.5.1 gates coexist and this is the one that consumes evidence "
                    "nothing produces.")

    def test_the_forbidden_gate_runners_are_real_names(self):
        """Anti-vacuity: a denylist of typos forbids nothing."""
        for target, runners in sorted(DEFERRED_GATE_RUNNERS.items()):
            defined = {n.name for n in ast.walk(self.graph.parse(target))
                       if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            for runner in runners:
                self.assertIn(runner, defined,
                              f"{runner} is not defined in {target}; this denylist entry is "
                              "a typo and forbids nothing")

    def test_the_allowlists_are_real_names(self):
        """Anti-vacuity, the other direction: an allowlist of typos is dead weight."""
        for target, allowed in (
                (f"{ROOT_PKG}.evaluator.integrity", INTEGRITY_ALLOWED_NAMES),
                (f"{ROOT_PKG}.evaluator.surface", SURFACE_ALLOWED_NAMES)):
            tree = self.graph.parse(target)
            defined = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Assign):
                    defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
            missing = sorted(allowed - defined)
            self.assertEqual(missing, [], f"{target}: allowlist names that do not exist: "
                                          f"{missing}")


# =============================================================================
# Optional stopping
# =============================================================================


class TestNoOptionalStopping(unittest.TestCase):
    """`statistics.py` is on the campaign path for CALIBRATION CONSTANTS ONLY.

    The e-process in that module made the gate unpassable: threshold 10 against a
    sign-martingale that tops out at 5.5687 over five same-sign blocks, at every
    effect magnitude, because the construction is sign-based. The fix's
    authorization was self-certifying. And the 2026-08-04 A/A data says the
    procedure was never warranted here in the first place — between-run CV
    1.62% (pp512) / 1.88% (tg128); a median over paired deltas covers that.

    So: no interim looks, no adaptive block count, no early stop. The block count
    is fixed by the plan before the first block is run.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _REAL_GRAPH
        cls.edges = campaign_edges()

    def test_statistics_is_on_the_path_at_all(self):
        """Anti-vacuity: these assertions must be about a module that is imported."""
        self.assertIn(STATISTICS_MODULE, self.edges)

    def test_no_optional_stopping_name_is_bound_on_the_campaign_path(self):
        used = self.graph.names_bound_from(self.edges, STATISTICS_MODULE)
        findings = []
        for module, names in sorted(used.items()):
            allowed = FIXED_N_EVENT_REDUCTION.get(module, frozenset())
            for name in sorted((names & OPTIONAL_STOPPING_NAMES) - allowed):
                findings.append(f"{module} binds {STATISTICS_MODULE}.{name}")
        self.assertEqual(
            findings, [],
            "\n".join(findings) + "\n\nThe campaign path may hold statistics.py for "
            "calibration constants, block plumbing and `median`. Constructing an e-process "
            "or a sequential evaluation is optional stopping, and the sign-martingale "
            "cannot cross its own calibrated threshold.")

    def test_the_e_process_cannot_be_reached_through_a_bound_object(self):
        """`CampaignStatistics.sequential_evaluation()` is the hole a name check leaves."""
        calls = self.graph.attribute_uses(self.edges, exclude=(STATISTICS_MODULE,))
        findings = []
        for module, names in sorted(calls.items()):
            for name in sorted(names & OPTIONAL_STOPPING_CALLS):
                findings.append(f"{module} calls .{name}(…) — that is an interim look")
        self.assertEqual(findings, [], "\n".join(findings))

    def test_the_accept_rule_cannot_run_an_e_process(self):
        """`api._resolve_effect` is the accept rule, and `api` cannot reach statistics.

        The e-value is a RECORDED FIELD of `EffectEstimate` that the rule reads;
        it is not a computation the rule can trigger. That is what makes the
        `e_value < threshold` branch auditable from the record instead of being
        re-derived at decision time under whatever construction happened to be
        bound.
        """
        accept_edges = ImportGraph(PKG_DIR, ROOT_PKG).closure([ACCEPT_RULE_MODULE])
        self.assertNotIn(STATISTICS_MODULE, accept_edges,
                         f"{ACCEPT_RULE_MODULE} now imports {STATISTICS_MODULE}; the accept "
                         "rule can run its own e-process")
        tree = self.graph.parse(ACCEPT_RULE_MODULE)
        resolve = next((n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == "_resolve_effect"), None)
        self.assertIsNotNone(resolve, "api._resolve_effect is gone; the accept rule moved and "
                                      "this assertion no longer covers it")
        called = {n.func.attr if isinstance(n.func, ast.Attribute) else
                  getattr(n.func, "id", None)
                  for n in ast.walk(resolve) if isinstance(n, ast.Call)}
        self.assertEqual(called & (OPTIONAL_STOPPING_NAMES | OPTIONAL_STOPPING_CALLS), set(),
                         "the accept rule itself calls into the sequential machinery")

    def test_the_forbidden_optional_stopping_names_are_real(self):
        """Anti-vacuity: every forbidden name must exist in statistics.py."""
        tree = self.graph.parse(STATISTICS_MODULE)
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
        missing = sorted(OPTIONAL_STOPPING_NAMES - defined)
        self.assertEqual(missing, [], f"names this test forbids but statistics.py does not "
                                      f"define: {missing}")

    def test_the_forbidden_look_methods_are_real(self):
        tree = self.graph.parse(STATISTICS_MODULE)
        methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        missing = sorted(OPTIONAL_STOPPING_CALLS - methods)
        self.assertEqual(missing, [], f"look methods this test forbids but statistics.py does "
                                      f"not define: {missing}")


# =============================================================================
# The same three checks, run against a COPY of this package with a violation in it
# =============================================================================


class TestTheBoundaryCatchesRealTreeViolations(unittest.TestCase):
    """Mutation verification against the REAL modules, not a toy fixture.

    `TestTheWalkerItself` proves the walker resolves each import FORM. This
    proves the three checks fire when a violation is planted in a tree that
    contains the actual 94k lines — the same closure, the same
    `__init__` side effects, the same alias names.

    The package is COPIED to a temp dir first. The shared clone is never
    written to: `/workspace/repos/<name>` and `/mnt/raid0/llm/<name>` are one
    clone, and a parallel session can be mid-read of any file here.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.copy_dir = Path(cls._tmp.name) / ROOT_PKG
        shutil.copytree(
            PKG_DIR, cls.copy_dir,
            ignore=shutil.ignore_patterns("__pycache__", "test_*.py", "*.md", "data"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _findings(self, source: str, check) -> list:
        planted = self.copy_dir / "_boundary_probe.py"
        planted.write_text(textwrap.dedent(source), encoding="utf-8")
        try:
            graph = ImportGraph(self.copy_dir, ROOT_PKG)
            roots = [*campaign_roots(), f"{ROOT_PKG}._boundary_probe"]
            return check(graph, graph.closure(roots))
        finally:
            planted.unlink(missing_ok=True)

    def assert_caught(self, source: str, check, needle: str) -> None:
        findings = self._findings(source, check)
        self.assertTrue(findings, f"planted violation was NOT caught: {needle}")
        self.assertTrue(any(needle in f for f in findings),
                        f"caught something else than {needle!r}: {findings}")

    # -- planting a module under a live DEFERRED prefix ---------------------

    #: The module every deferred plant below imports. It does not exist in the
    #: real tree, it is written into the COPY, and it is deliberately NOT in
    #: `CONTROLLER_ALLOWED` — which is exactly the case the prefix ban is kept
    #: for now that the four real controller modules are named exceptions.
    REVENANT = f"{ROOT_PKG}.controller.revenant"

    def plant_controller_module(self) -> None:
        """Write `controller/revenant.py` (and a subpackage) into the COPY.

        Two shapes, because two different mechanisms are tested through them:
        `revenant.py` is a plain module a probe can import directly, and
        `revenant_pkg/` is a package whose `__init__` imports a submodule — the
        invisible edge, the one no reading of the probe's own source reveals.
        """
        sub = self.copy_dir / "controller"
        (sub / "revenant.py").write_text("VALUE = 1\n", encoding="utf-8")
        pkg = sub / "revenant_pkg"
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").write_text(
            "from . import hidden\n\n__all__ = ['hidden']\n", encoding="utf-8")
        (pkg / "hidden.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.addCleanup(shutil.rmtree, pkg, True)
        self.addCleanup((sub / "revenant.py").unlink, True)

    # -- the deferred half --------------------------------------------------
    #
    # RE-POINTED 2026-08-04. Every planted violation below used to name a module
    # under `release/`, `adapters/` or `surface/`. Those are gone, and a probe
    # that imports a module which does not exist plants NOTHING: `closure()`
    # drops a name with no file, so `deferred_findings` returned [] and all five
    # of these tests failed with "planted violation was NOT caught" — the walker's
    # bite reported as absent when what was absent was the target.
    #
    # RE-POINTED AGAIN, same day, for the same reason one level down. They were
    # pointed at `controller.hypotheses` / `controller.do_not_repeat`, and those
    # two are now NAMED EXCEPTIONS (`CONTROLLER_ALLOWED`) because `campaign.py`
    # acquires its claim through the falsifier gate. A plant that names an
    # allow-listed module is not a violation, so all five would have gone quiet —
    # the same "the target is what is absent" failure, arrived at from the other
    # direction. Each now imports a module PLANTED under `controller/` in the
    # copy: still a live `DEFERRED` prefix, still really on disk, still really
    # absent from the campaign closure, and — the point of the allow-list being a
    # list — still banned. Every mechanism each test was written for is intact.
    #
    # The other reading — "assert that re-adding a DELETED module is caught" — is
    # not dropped; it is `test_re_adding_a_deleted_plane_and_importing_it_is_caught`
    # below, which is the one thing the deleted prefixes can still be tested for.

    def test_a_direct_import_of_controller_is_caught(self):
        self.plant_controller_module()
        self.assert_caught("from .controller import revenant\n",
                           deferred_findings, self.REVENANT)

    def test_an_allow_listed_controller_module_is_not_a_finding(self):
        """CONTROL, and the one that stops the four plants passing vacuously.

        `controller.hypotheses` is the module `campaign.py` imports for the
        falsifier gate. If it produced a finding, the boundary and the driver
        would be in permanent disagreement and someone would eventually silence
        the wrong one of the two.
        """
        self.assertEqual(
            self._findings("from .controller import hypotheses\n", deferred_findings), [])

    def test_re_adding_a_deleted_plane_and_importing_it_is_caught(self):
        """The DELETED prefixes, and the only honest assertion left about them.

        Every prefix in `DELETED_BY_OPERATOR` is also in `DEFERRED`, which is a pair
        of claims that would otherwise go
        unchecked forever: an absent module is trivially unreachable, so nothing
        distinguishes "the ban holds" from "the ban is dead text". What CAN be
        checked is the case the ban is for — someone puts the plane back and
        depends on it — and that is what is planted here, in the temp COPY.

        Writing the module into the copy is not restoring it: the real tree is
        never written to (`test_the_probe_leaves_nothing_behind`), and the file is
        an empty stub, not the 17,782 lines that were removed.
        """
        for prefix in DELETED_BY_OPERATOR:
            plane = prefix.split(".")[-1]
            sub = self.copy_dir / plane
            sub.mkdir(exist_ok=True)
            (sub / "__init__.py").write_text("", encoding="utf-8")
            (sub / "revenant.py").write_text("VALUE = 1\n", encoding="utf-8")
            try:
                with self.subTest(plane=plane):
                    self.assert_caught(f"from .{plane} import revenant\n",
                                       deferred_findings, f"{prefix}.revenant")
            finally:
                shutil.rmtree(sub)

    def test_importing_one_package_pulls_what_its_dunder_init_binds(self):
        """The invisible edge: a package `__init__` importing a submodule.

        The probe names `controller.revenant_pkg` and nothing else. The ONLY
        route from it to `revenant_pkg.hidden` is the parent package's `__init__`
        running as a side effect of the import — the form `surface/__init__.py`
        used to demonstrate, and which no reading of the probe's own source shows.
        (`controller/__init__.py` binds `hypotheses` and `do_not_repeat` the same
        way; both are now named exceptions, so the mechanism is demonstrated on a
        planted package instead of on them.)
        """
        self.plant_controller_module()
        self.assert_caught("from .controller import revenant_pkg\n",
                           deferred_findings, f"{self.REVENANT}_pkg.hidden")

    def test_a_function_level_import_of_controller_is_caught(self):
        self.plant_controller_module()
        self.assert_caught("""
            def later():
                from .controller import revenant
                return revenant
        """, deferred_findings, self.REVENANT)

    def test_a_dynamic_import_string_into_controller_is_caught(self):
        self.plant_controller_module()
        self.assert_caught("""
            import importlib

            def later():
                return importlib.import_module("autokernel.controller.revenant")
        """, deferred_findings, self.REVENANT)

    def test_a_compliant_module_produces_no_deferred_finding(self):
        """CONTROL."""
        self.assertEqual(
            self._findings("from . import schemas\nfrom .evaluator import api\n",
                           deferred_findings), [])

    # -- the provenance fence ----------------------------------------------

    def test_wiring_the_source_integrity_gate_runner_is_caught(self):
        self.assert_caught("""
            from .evaluator import integrity

            def runner():
                return integrity.SourceIntegrityGateRunner
        """, provenance_fence_findings, "SourceIntegrityGateRunner")

    def test_wiring_the_surface_gate_runner_is_caught(self):
        self.assert_caught("from .evaluator.surface import SurfaceGateRunner\n",
                           provenance_fence_findings, "SurfaceGateRunner")

    def test_a_new_integrity_name_beyond_the_allowlist_is_caught(self):
        self.assert_caught("""
            from .evaluator import integrity

            def go(x):
                return integrity.check_symbol_preservation(x)
        """, provenance_fence_findings, "check_symbol_preservation")

    def test_a_provenance_primitive_produces_no_fence_finding(self):
        """CONTROL: hashing a file is what this module is on the path FOR."""
        self.assertEqual(
            self._findings("""
                from .evaluator import integrity

                def go(p):
                    return integrity.sha256_file(p)
            """, provenance_fence_findings), [])

    # -- optional stopping --------------------------------------------------

    def test_binding_sequential_evaluation_is_caught(self):
        self.assert_caught("from .evaluator.statistics import SequentialEvaluation\n",
                           optional_stopping_findings, "SequentialEvaluation")

    def test_calling_run_e_process_through_the_module_alias_is_caught(self):
        self.assert_caught("""
            from .evaluator import statistics

            def go(x):
                return statistics.run_e_process(x, construction=None,
                                                hypothesis="improvement", margin=0.0,
                                                threshold=10.0)
        """, optional_stopping_findings, "run_e_process")

    def test_an_interim_look_through_a_bound_object_is_caught(self):
        """The hole a name-level check leaves: `CampaignStatistics` is legitimately held."""
        self.assert_caught("""
            def go(campaign_statistics):
                return campaign_statistics.sequential_evaluation(
                    candidate_id="c", stratum="selection",
                    metric_direction="higher_better")
        """, optional_stopping_findings, "sequential_evaluation")

    # -- the four bypasses a single-hop alias check leaves open ---------------
    #
    # Each of these was MISSED by the first version of this file and is runnable
    # against the real tree: `worktree.py` already imports `evaluator.integrity`
    # and `microbench.py` already imports `evaluator.statistics`, so the chained
    # attribute resolves at runtime without the bypassing module importing either.

    def test_a_chained_alias_into_the_gate_runner_is_caught(self):
        self.assert_caught("""
            from . import evaluator

            def runner():
                return evaluator.integrity.SourceIntegrityGateRunner
        """, provenance_fence_findings, "SourceIntegrityGateRunner")

    def test_a_chained_alias_into_the_e_process_is_caught(self):
        self.assert_caught("""
            from . import evaluator

            def go(x):
                return evaluator.statistics.run_e_process(x)
        """, optional_stopping_findings, "run_e_process")

    def test_getattr_by_string_into_the_gate_runner_is_caught(self):
        self.assert_caught("""
            from .evaluator import integrity

            def runner():
                return getattr(integrity, "SourceIntegrityGateRunner")
        """, provenance_fence_findings, "SourceIntegrityGateRunner")

    def test_a_chained_alias_to_a_primitive_is_not_a_finding(self):
        """CONTROL: the chain is not the violation, the NAME is."""
        self.assertEqual(
            self._findings("""
                from . import evaluator

                def go(p):
                    return evaluator.integrity.sha256_file(p)
            """, provenance_fence_findings), [])

    # -- dynamic imports ------------------------------------------------------

    def test_a_relative_dynamic_import_into_controller_is_caught(self):
        self.plant_controller_module()
        self.assert_caught("""
            import importlib

            def later():
                return importlib.import_module(".controller.revenant", __package__)
        """, deferred_findings, self.REVENANT)

    def test_an_fstring_dynamic_import_is_reported_unresolved(self):
        """FAIL-CLOSED: what the walk cannot follow must not read as a clean pass.

        The named module has to EXIST *and be banned* for the first half to say
        anything: against `controller.guards`, which was deleted, "no deferred
        finding" is what a walk that resolved the f-string perfectly would also
        report, so the assertion held for the wrong reason. The same is now true
        of `controller.hypotheses`, which is an allow-listed exception. Against
        the PLANTED `controller.revenant` — which
        `test_a_dynamic_import_string_into_controller_is_caught` proves IS caught
        when the same string is a literal — silence here is a statement about the
        f-string and nothing else.
        """
        self.plant_controller_module()
        self.assertEqual(
            self._findings("""
                import importlib

                def later():
                    return importlib.import_module(f"{__package__}.controller.revenant")
            """, deferred_findings), [],
            "an f-string names no module statically; it must surface as unresolved, "
            "not as a deferred finding")
        self.assert_caught("""
            import importlib

            def later():
                return importlib.import_module(f"{__package__}.controller.revenant")
        """, unresolved_import_findings, "import_module")

    def test_the_real_campaign_path_has_no_unresolved_dynamic_import(self):
        """CONTROL and anti-vacuity in one: the check above must be silent here."""
        self.assertEqual(self._findings("from . import schemas\n",
                                        unresolved_import_findings), [])

    def test_an_interim_look_bound_before_it_is_called_is_caught(self):
        """`look = obj.sequential_evaluation` then `look(...)` — one line apart.

        A Call-only scan reads this as clean, which is the whole trick.
        """
        self.assert_caught("""
            def go(campaign_statistics):
                look = campaign_statistics.sequential_evaluation
                return look(candidate_id="c", stratum="selection",
                            metric_direction="higher_better")
        """, optional_stopping_findings, "sequential_evaluation")

    def test_the_calibration_use_produces_no_stopping_finding(self):
        """CONTROL: `median` over paired deltas is the rule the A/A data supports."""
        self.assertEqual(
            self._findings("""
                from .evaluator import statistics

                def go(values):
                    return statistics.median(values)
            """, optional_stopping_findings), [])

    def test_the_probe_leaves_nothing_behind(self):
        self._findings("from . import schemas\n", deferred_findings)
        self.assertFalse((self.copy_dir / "_boundary_probe.py").exists())
        self.assertFalse((PKG_DIR / "_boundary_probe.py").exists(),
                         "the probe was written into the REAL package")


# =============================================================================
# FOOTPRINT.md is generated from this graph, so it cannot drift
# =============================================================================

FOOTPRINT_PATH = PKG_DIR / "FOOTPRINT.md"

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([0-9,]+)\s*\|\s*(yes|no)\s*\|\s*(.+?)\s*\|\s*$")


def parse_footprint(text: str) -> dict:
    rows = {}
    for line in text.splitlines():
        match = _ROW.match(line)
        if match:
            path, lines, imported, reason = match.groups()
            rows[path] = (int(lines.replace(",", "")), imported == "yes", reason)
    return rows


#: The three rows `refresh_footprint` regenerates. Stripped from the text before
#: the restatement check below, because "stated once" is a rule about the PROSE,
#: not about the table: two total rows may legitimately carry the same figure —
#: they do whenever the deferred half is empty — and neither of them can drift,
#: since both are rewritten from the same walk.
_TOTAL_ROW_LABELS = ("ON THE CAMPAIGN PATH", "DEFERRED", "TOTAL")


def _outside_the_total_rows(text: str) -> str:
    for label in _TOTAL_ROW_LABELS:
        text = re.sub(rf"^\|\s*\*\*{re.escape(label)}\*\*.*$", "", text,
                      flags=re.MULTILINE)
    return text


#: SCOPE LIMIT of the restatement rule, stated rather than left to be discovered.
#:
#: A three-digit or smaller total has no distinctive digits, and no textual rule
#: can tell a restatement of it from an unrelated literal. `0` — the DEFERRED
#: total since `--hypothesis` put the controller plane on the campaign path —
#: occurs inside `T0`, `/mnt/raid0/llm`, `t0_provider.py` and `Device 0: CPU`,
#: six times in the real document, none of them a figure. So the restatement rule
#: applies only to totals of 1,000 and up, which every real footprint total is.
#:
#: What is NOT weakened: the VALUE check above is exact at every magnitude and is
#: what actually says "a plane moved". This limit is on the secondary rule about
#: the same figure appearing twice, and it is asserted in both directions by
#: `TestTheTotalsCheckBites`.
RESTATEMENT_CHECK_FLOOR = 1_000


def total_findings(text: str, imported: int, deferred: int, tolerance: int) -> list:
    """The three headline totals, each checked in its OWN row and restated nowhere.

    CORRECTED 2026-08-04, when the deferred half went to zero. `text.count(best)`
    was a SUBSTRING count over the whole document, and it has two defects that
    only surfaced once the figures stopped being three distinct five-digit
    numbers: a deferred total of `0` matched inside every date and every line
    count (80 hits), and `ON THE CAMPAIGN PATH == TOTAL` — true whenever nothing
    is deferred — reported the table's own other row as a drifting copy. Both are
    the check failing on its own arithmetic rather than on a document defect. The
    rule it was written for is unchanged and is now stated directly: a total may
    not be restated OUTSIDE the rows that regenerate, and the match is on the
    whole number rather than on a run of digits inside a longer one.
    """
    findings = []
    prose = _outside_the_total_rows(text)
    for label, value in (("ON THE CAMPAIGN PATH", imported), ("DEFERRED", deferred),
                         ("TOTAL", imported + deferred)):
        row = re.search(rf"^\|\s*\*\*{re.escape(label)}\*\*.*$", text, re.MULTILINE)
        if row is None:
            findings.append(f"FOOTPRINT.md has no **{label}** total row")
            continue
        stated = re.findall(r"[0-9][0-9,]*", row.group(0))
        if not stated:
            findings.append(f"the **{label}** row states no number")
            continue
        best = min(stated, key=lambda s: abs(int(s.replace(",", "")) - value))
        if abs(int(best.replace(",", "")) - value) > tolerance:
            findings.append(f"the **{label}** row says {stated}, the tree says {value:,} — "
                            "that is a plane MOVING, not a plane being edited")
            continue
        if value < RESTATEMENT_CHECK_FLOOR:
            continue
        elsewhere = len(re.findall(rf"(?<![0-9,]){re.escape(best)}(?![0-9,])", prose))
        if elsewhere:
            findings.append(f"{best} is stated {elsewhere} time(s) outside the "
                            f"**{label}** row; only the row regenerates, so the other "
                            "copies drift silently")
    return findings


class TestTheTotalsCheckBites(unittest.TestCase):
    """The totals check, verified against text it MUST reject.

    Deleting the per-line assertion is only safe if what replaced it still
    catches the thing that mattered: a plane moving across the boundary, and the
    two figures being swapped.
    """

    #: Deliberately far apart. The first version of this fixture used 49,000 and
    #: 50,000 — one tolerance apart — and `test_a_swapped_pair_is_rejected` FAILED,
    #: because a swap of two near-equal numbers is not a numeric difference. A
    #: fixture that makes the signal under test unobservable passes a broken
    #: implementation; `test_the_bound_on_swap_detection_is_stated` records the
    #: real limit instead of hiding it.
    TABLE = ("| **ON THE CAMPAIGN PATH** | **30,000** |\n"
             "| **DEFERRED** (provably unreachable) | **70,000** |\n"
             "| **TOTAL** | **100,000** |\n")

    def test_a_correct_table_is_accepted(self):
        """CONTROL."""
        self.assertEqual(total_findings(self.TABLE, 30_000, 70_000, 1_000), [])

    def test_normal_editing_churn_is_accepted(self):
        """CONTROL: the whole reason the per-line assertion was removed."""
        self.assertEqual(total_findings(self.TABLE, 30_400, 70_300, 1_000), [])

    def test_a_swapped_pair_is_rejected(self):
        findings = total_findings(self.TABLE, 70_000, 30_000, 1_000)
        self.assertTrue(findings, "swapping the two figures passed")

    def test_a_plane_moving_across_the_boundary_is_rejected(self):
        """20,000 lines migrating is exactly what the tolerance must not absorb."""
        findings = total_findings(self.TABLE, 50_000, 50_000, 1_000)
        self.assertTrue(findings)

    def test_the_bound_on_swap_detection_is_stated(self):
        """SCOPE LIMIT, asserted so it cannot be mistaken for cover.

        When the two halves are within one tolerance of each other — which is the
        case in the tree TODAY — no numeric check can tell them apart, and this
        one does not pretend to. It is also the case where it does not matter: the
        operator's decision is "delete roughly half", and both readings say that.
        The reachability FLAGS, which are exact and per row, are what say which
        half is which.
        """
        near = ("| **ON THE CAMPAIGN PATH** | **49,700** |\n"
                "| **DEFERRED** (provably unreachable) | **50,100** |\n"
                "| **TOTAL** | **99,800** |\n")
        self.assertEqual(total_findings(near, 50_100, 49_700, 1_000), [])

    def test_a_total_restated_in_the_prose_is_rejected(self):
        text = self.TABLE + "\nThe deferred half is 70,000 lines.\n"
        findings = total_findings(text, 30_000, 70_000, 1_000)
        self.assertTrue(any("70,000 is stated 1 time(s) outside" in f for f in findings),
                        findings)

    def test_two_rows_carrying_the_same_figure_are_accepted(self):
        """CONTROL for the 2026-08-04 correction, and it is the tree TODAY.

        When nothing is deferred, ON THE CAMPAIGN PATH and TOTAL are the same
        number by arithmetic. Reporting that as a drifting copy would be the
        check failing on its own construction — and the fix for it, deleting the
        restatement rule, would have thrown away the thing it does catch (below).
        """
        table = ("| **ON THE CAMPAIGN PATH** | **57,410** |\n"
                 "| **DEFERRED** (provably unreachable) | **0** |\n"
                 "| **TOTAL** | **57,410** |\n")
        self.assertEqual(total_findings(table, 57_410, 0, 1_000), [])

    def test_the_restatement_rule_does_not_apply_below_its_floor(self):
        """SCOPE LIMIT, asserted so it cannot be mistaken for cover.

        `0` — the DEFERRED total since `--hypothesis` put the controller plane on
        the campaign path — is a substring of `T0`, `raid0`, `t0_provider.py` and
        `Device 0: CPU`, six times over in the real document and not one of them
        a figure. `RESTATEMENT_CHECK_FLOOR` is where the rule stops applying, and
        this is the assertion that the limit is real rather than incidental.

        What is NOT given up is checked one test down: the VALUE of that same row
        is still compared against the tree, exactly, at every magnitude.
        """
        table = ("| **ON THE CAMPAIGN PATH** | **57,410** |\n"
                 "| **DEFERRED** (provably unreachable) | **0** |\n"
                 "| **TOTAL** | **57,410** |\n"
                 "\nT0 runs first; the trees live under /mnt/raid0/llm.\n")
        self.assertEqual(total_findings(table, 57_410, 0, 1_000), [])

    def test_a_zero_deferred_total_is_still_checked_for_its_value(self):
        """CONTROL for the limit above: the row is exempt from ONE rule, not two."""
        table = ("| **ON THE CAMPAIGN PATH** | **57,410** |\n"
                 "| **DEFERRED** (provably unreachable) | **0** |\n"
                 "| **TOTAL** | **57,410** |\n")
        findings = total_findings(table, 50_000, 7_410, 1_000)
        self.assertTrue(any("DEFERRED" in f for f in findings), findings)

    def test_a_four_digit_total_is_still_restatement_checked(self):
        """The floor is a floor, not an off switch."""
        table = ("| **ON THE CAMPAIGN PATH** | **30,000** |\n"
                 "| **DEFERRED** (provably unreachable) | **1,200** |\n"
                 "| **TOTAL** | **31,200** |\n"
                 "\nThe deferred half is 1,200 lines.\n")
        findings = total_findings(table, 30_000, 1_200, 1_000)
        self.assertTrue(any("1,200 is stated 1 time(s) outside" in f for f in findings),
                        findings)

    def test_a_missing_row_is_rejected(self):
        findings = total_findings("| **TOTAL** | **100,000** |\n", 30_000, 70_000, 1_000)
        self.assertTrue(any("ON THE CAMPAIGN PATH" in f for f in findings), findings)


class TestFootprintDocumentMatchesTheTree(unittest.TestCase):
    """The table is an assertion, not a description.

    A footprint document that drifts from the tree is worse than none: it is the
    same failure as a dashboard that renders clean over a dead producer.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _REAL_GRAPH
        cls.edges = campaign_edges()
        cls.rows = parse_footprint(FOOTPRINT_PATH.read_text(encoding="utf-8")) \
            if FOOTPRINT_PATH.is_file() else {}

    def test_footprint_exists(self):
        self.assertTrue(FOOTPRINT_PATH.is_file(), f"{FOOTPRINT_PATH} is missing")

    def test_every_non_test_module_has_a_row(self):
        on_disk = {str(p.relative_to(PKG_DIR)) for p in PKG_DIR.rglob("*.py")
                   if not p.name.startswith("test_")}
        missing = sorted(on_disk - set(self.rows))
        self.assertEqual(missing, [], f"modules with no FOOTPRINT.md row: {missing}")

    def test_no_row_names_a_module_that_is_gone(self):
        extra = sorted(p for p in self.rows if not (PKG_DIR / p).is_file())
        self.assertEqual(extra, [], f"FOOTPRINT.md rows for files that do not exist: {extra}")

    #: How to fix any failure in this class. The mechanical columns regenerate;
    #: the reason column never does.
    REFRESH = ("python3 scripts/kernel_rnd/autokernel/test_campaign_footprint.py --refresh "
               "(from the repo root)")

    #: Line counts drift by one edit to one module by any of the sessions sharing
    #: this clone, and there is no boundary question they answer that the
    #: reachability flags do not. Asserting them to the LINE produced five red
    #: suites in forty minutes on 2026-08-04, none of them a boundary violation —
    #: and a guard that cries wolf is how `release/` got committed the day before
    #: the code that can compile a candidate. So: the split is asserted (below) to
    #: the nearest thousand, which still catches a plane MOVING and ignores a
    #: plane being edited. `--refresh` restores the table to the line.
    TOTAL_TOLERANCE = 1_000

    def test_every_reachability_flag_matches_the_walked_graph(self):
        wrong = []
        for rel, (_, claimed, _) in sorted(self.rows.items()):
            module = f"{ROOT_PKG}." + rel[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            actual = module in self.edges
            if actual != claimed:
                wrong.append(f"{rel}: table says imported={claimed}, graph says {actual}")
        self.assertEqual(wrong, [], "\n".join(wrong))

    def test_every_row_carries_a_reason(self):
        empty = sorted(p for p, (_, _, reason) in self.rows.items() if len(reason) < 20)
        self.assertEqual(empty, [], f"FOOTPRINT.md rows with no real reason: {empty}")

    def test_the_totals_are_the_tree(self):
        """The split must be a number the operator can act on.

        Measured against the TREE, not against the sum of the table's own rows —
        summing the rows would have let the whole table drift together and still
        add up. Two things this catches that no per-row check does:

        * each total is matched IN ITS OWN ROW. `assertIn(f"{value:,}", text)` was
          satisfied by the number appearing anywhere in the file, so SWAPPING the
          campaign-path and deferred figures passed.
        * each total may appear only once in the whole file. The prose used to
          restate `46,567` and `48.6%`; only the row regenerates, so the copies
          drifted silently — a dashboard rendering clean over a dead producer.
        """
        text = FOOTPRINT_PATH.read_text(encoding="utf-8")
        imported = deferred = 0
        for path in PKG_DIR.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            rel = str(path.relative_to(PKG_DIR))
            module = f"{ROOT_PKG}." + rel[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            count = len(path.read_text(encoding="utf-8").splitlines())
            if module in self.edges:
                imported += count
            else:
                deferred += count
        findings = total_findings(text, imported, deferred, self.TOTAL_TOLERANCE)
        self.assertEqual(findings, [], "\n".join(findings) + f"\n\nfix: {self.REFRESH}")


def refresh_footprint() -> str:
    """Rewrite the MECHANICAL columns of FOOTPRINT.md in place. `--refresh`.

    Line counts, `yes`/`no` flags and the three totals are facts and are
    regenerated. **Rows are not created and reasons are never written.** A module
    with no row is REPORTED by name and left without one, so
    `test_every_non_test_module_has_a_row` stays red until a human adds the row
    and the incident or measured fact that puts the module where it is. The flag
    can refresh arithmetic; it cannot invent a justification, and it will not
    quiet the failure that asks for one.
    """
    graph = ImportGraph(PKG_DIR, ROOT_PKG)
    edges = graph.closure(campaign_roots())
    text = FOOTPRINT_PATH.read_text(encoding="utf-8")
    rows = parse_footprint(text)

    def module_of(rel: str) -> str:
        module = f"{ROOT_PKG}." + rel[:-3].replace("/", ".")
        return module[: -len(".__init__")] if module.endswith(".__init__") else module

    on_disk = sorted(str(p.relative_to(PKG_DIR)) for p in PKG_DIR.rglob("*.py")
                     if not p.name.startswith("test_"))
    lines_of = {rel: len((PKG_DIR / rel).read_text(encoding="utf-8").splitlines())
                for rel in on_disk}
    reached = {rel: module_of(rel) in edges for rel in on_disk}

    out = []
    for line in text.splitlines():
        match = _ROW.match(line)
        if match and match.group(1) in lines_of:
            rel = match.group(1)
            out.append(f"| `{rel}` | {lines_of[rel]:,} | "
                       f"{'yes' if reached[rel] else 'no'} | {match.group(4)} |")
        else:
            out.append(line)
    text = "\n".join(out) + ("\n" if text.endswith("\n") else "")

    imported = sum(n for rel, n in lines_of.items() if reached[rel])
    deferred = sum(n for rel, n in lines_of.items() if not reached[rel])
    for label, value in (("ON THE CAMPAIGN PATH", imported),
                         ("DEFERRED", deferred),
                         ("TOTAL", imported + deferred)):
        text = re.sub(rf"(\*\*{re.escape(label)}\*\*[^|]*\| \*\*)[0-9,]+(\*\*)",
                      rf"\g<1>{value:,}\g<2>", text)
    FOOTPRINT_PATH.write_text(text, encoding="utf-8")

    missing = [rel for rel in on_disk if rel not in rows]
    return (f"refreshed {FOOTPRINT_PATH.name}: {imported:,} on the campaign path, "
            f"{deferred:,} deferred, {imported + deferred:,} total"
            + (f"\nMODULES WITH NO ROW (write the reason yourself): {missing}"
               if missing else ""))


class DeclaredGuardCallerAuditTest(unittest.TestCase):
    """A declared guard without a real non-test caller is a failing build."""

    def test_every_previously_callerless_guard_has_its_required_live_caller(self):
        for guard, (relative_path, call_fragment) in GUARD_CALLER_CONTRACT.items():
            with self.subTest(guard=guard):
                source = (PKG_DIR / relative_path).read_text(encoding="utf-8")
                code = "\n".join(
                    line for line in source.splitlines()
                    if not line.lstrip().startswith("#"))
                self.assertIn(
                    call_fragment, code,
                    f"{guard} is declared but its required non-test caller disappeared")

    def test_guard_contract_names_exactly_the_five_discovered_instances(self):
        self.assertEqual(len(GUARD_CALLER_CONTRACT), 5)
        self.assertEqual(len(set(GUARD_CALLER_CONTRACT.values())), 5)


if __name__ == "__main__":  # pragma: no cover
    if "--refresh" in sys.argv:
        print(refresh_footprint())
    else:
        unittest.main()
