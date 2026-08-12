#!/usr/bin/env python3
"""packager.py — the AK6 release packager (§11.2, §11.3, §11.5).

WHY THIS MODULE EXISTS
----------------------
AK5 answers *"is this champion releasable?"* — `plan.py` derives the matrix,
`t3.py` adjudicates it, `readiness.py` reports the advisory signal. None of them
answers the question the operator actually has to act on: **what exactly do I
type, in what order, and what happens if it goes wrong?**

The tempting shape for that answer is a broker — something that holds the verdict
and performs the freeze. That shape was withdrawn on 2026-08-02 (AK-D6), and the
audit that withdrew it is the reason this module looks the way it does: an
automatic freeze crosses **four** human-only trust boundaries, not one
(`MEASUREMENT.md:140-142`, design §1.3):

  1. the freeze/cutover itself;
  2. the era-registry rows (`orchestration/instrument_eras.yaml`, separately pinned
     in `human_only_paths.yaml:35-37`) — the v8 cutover wrote three;
  3. the AutoPilot baseline apply (`orchestration/autopilot_baseline.yaml`,
     `human_only_paths.yaml:38-40`), which the E8 precedent opens as a
     **fail-closed hold** until an operator-ratified reseed; and
  4. the pinned human-only path list itself, which is branch-pattern-scoped and
     therefore matches `production-consolidated-v9` the moment that branch exists.

**THE CARDINAL RULE: AutoKernel never freezes and never cuts over.** This module
produces a PACKAGE that a human executes. There is no authority here to hold, to
delegate, or to flag — which is why the refusals below are *functions that raise*
and *properties proved from this module's own AST*, not sentences in a docstring.
`P-AK-SEARCH-1` denial 5 says the same thing from the search side: a readiness
signal is not a freeze trigger.

WHAT THIS MODULE MAY DO (§11.2, first sentence)
-----------------------------------------------
Seal the champion; run T3 **through the trusted evaluator**; assemble the verdict
bundle; compute the next version and the full transaction plan; compute the
rollback plan and verify the archive target; DRAFT the era-registry row and the
AutoPilot rebaseline note; pre-validate every operator command end-to-end
(`MEASUREMENT.md:138-145` requires exactly this); and present a four-part decision
package (`OPERATING_CONSTRAINTS.md:69-78`).

WHAT IT MAY NOT DO, MADE STRUCTURAL (§11.2, second sentence)
------------------------------------------------------------
Edit source; rebuild the candidate outside the sealed build; change protocols,
thresholds or scope; waive failed evidence; touch any production branch, symlink,
era registry or baseline file; or EXECUTE any command it drafted.

Each of those is a **named function in `REFUSED_CAPABILITIES` that raises
unconditionally**, and `audit_refusal_doors_raise_unconditionally()` proves from
the AST that each still does. A greppable refusal is worth more than a prohibition
nobody can find: a future caller looking for "how do I execute this" lands on the
door and reads why it is nailed shut, rather than writing the tenth private
`subprocess.run`.

Three further properties are proved rather than asserted:

  * `audit_no_write_or_process_paths()` — no write, spawn or signal call exists.
    It denies `open` (the one call that takes a mode), the pathlib mutators —
    including `.replace()`, which IS the move-a-stable-kernel-symlink primitive —
    and `getattr(x, "<denied>")`, which routes around attribute matching.
  * `audit_no_clock_or_self_trigger()` — **the packager has no clock.** Every
    timestamp is an input. A module that can read the wall clock can decide that
    it is time to freeze; a module that cannot must be handed an
    `OperatorFreezeRequest`, and it can never mint one, because it never
    constructs one anywhere in its own source (AK7).
  * `audit_verdict_is_delegated()` — the packager never calls `t3.run_t3`,
    `t3.compute_verdict` or a phase runner. The verdict arrives through the
    injected `ReleaseTierEvaluator` seam, so "the evaluator that graded this"
    is a recorded fact rather than an implementation detail (invariant 4:
    actor, evaluator and packager are distinct authority domains).

CUTOVER IS A REQUEST, NEVER AN ACTION (§11.3)
----------------------------------------------
`OPERATING_CONSTRAINTS.md:41` — a reload *"must be executed BY THAT SESSION, at a
moment it chooses… route the request via coordinator-agent to the owning
session"*. So the package carries a `CutoverRequest`: a bus message record with
`needs_routing_to` naming the inference owner and `action_required` set, which the
owning session schedules at its own boundary. It names no time, and this module
has no transport — `send_cutover_request()` raises. An autonomous restart would be
precisely the preemption that rule was written for (INC-20260728-reload-preemption).

THE POST-CUTOVER WATCH WINDOW IS A DECLARED ARTIFACT (§11.5)
------------------------------------------------------------
T4 answers *"did the cutover work"*. The watch window answers *"was the cutover
right"*, and it is the last automatic safety net in the path now that the operator
performs the cutover. It is assembled here, **before the data is seen**, because a
band chosen after seeing the data is not a band:

  * duration — 7 days **or** a declared minimum volume per affected role,
    whichever is LATER ("a window that expires on a quiet weekend has observed
    nothing");
  * the six-signal table of §11.5, each with its own alarm direction, derived from
    the signal id rather than declared beside it;
  * era-labelled comparison against the **incumbent era's recorded distribution**
    (`MEASUREMENT.md:233`), never a remembered number;
  * bands fixed at package assembly and hashed, so an evaluation compared against
    different bands is refused rather than reported;
  * a hash-bound activation manifest naming each affected role's intended backend,
    binary and linkage identity, so T4 cannot choose its expected identity after
    seeing what the cutover actually left running;
  * an owner who is whoever executed the cutover; and
  * an explicit close-with-verdict step — *"an unclosed window is an open
    question, not a pass"*.

It produces a **RECOMMENDATION, never a claim**: production telemetry is
observational and uncontrolled, so a signal outside its band raises a decision
package and reverts nothing.

AK7 PREPARATION
---------------
`OperatorFreezeRequest` is the entry point the operator invokes. Executing the
freeze it asks for needs the operator **and a real compute window** — a full T3
matrix is a benchmarking program, not a validation pass — and is therefore out of
scope for any autonomous run. See `AK7_SCOPE_NOTE`.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md` §1.3, §1.5,
§1.6, §3.2, §10.4–§10.6, §11.1–§11.6, §14 phases AK6/AK7, §15.4, invariants 2–5.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .. import schemas, storage
from ..evaluator import integrity
from . import t3

__all__ = [
    # identity
    "MODULE_ID", "PACKAGE_SCHEMA", "FREEZE_REQUEST_SCHEMA", "WATCH_WINDOW_SCHEMA",
    "CUTOVER_MESSAGE_SCHEMA", "RECORD_CLASS", "EXECUTED_BY", "PACKAGE_NOTICE",
    "AK7_SCOPE_NOTE",
    # vocabularies
    "PACKAGE_STATES", "STATE_READY", "STATE_INCOMPLETE", "STATE_BLOCKED",
    "OPERATOR_AUTHORITY", "MACHINE_ACTOR_TOKENS", "REFUSED_CAPABILITIES",
    "HUMAN_ONLY_TARGET_PATTERNS", "ERA_ROW_KINDS", "REQUIRED_WATCH_SIGNALS",
    "WATCH_SIGNAL_ALARM_RULES", "WATCH_SIGNAL_SOURCES", "WATCH_RECOMMENDATIONS",
    "WATCH_CLOSE_VERDICTS", "WATCH_WINDOW_OUTPUT_CLASS",
    "DEFAULT_WATCH_WINDOW_DAYS", "REQUIRED_TRADEOFF_AXES", "TRUSTED_EVALUATOR_TIER",
    "SIGNAL_THROUGHPUT", "SIGNAL_LATENCY", "SIGNAL_ERROR_RATES", "SIGNAL_MEMORY",
    "SIGNAL_QUALITY", "SIGNAL_SUPERVISOR", "ALARM_REGRESSION", "ALARM_ANY_INCREASE",
    "ALARM_DRIFT_TO_FLOOR", "ALARM_ANY_OCCURRENCE", "WATCH_CONTINUE",
    "WATCH_RAISE_DECISION_PACKAGE", "WATCH_INCOMPLETE_EVIDENCE",
    "WATCH_CLOSE_NO_REGRESSION", "WATCH_STATE_OPEN", "WATCH_STATE_CLOSEABLE",
    "WATCH_STATE_CLOSED", "CUTOVER_ASK", "ERA_EFFECTIVE_FROM",
    "ERA_ROW_KIND_KERNEL", "ERA_ROW_KIND_AUTOPILOT_SPEED", "ERA_ROW_KIND_UMBRELLA",
    # errors
    "PackagerError", "PackagerInputError", "SelfTriggerRefused",
    "FreezeExecutionRefused", "CutoverExecutionRefused", "ProductionWriteRefused",
    "SealRefused", "EvaluatorNotTrusted", "VersionCollision", "RollbackIncomplete",
    "IncumbentModificationRefused", "BandsNotFixedBeforeData", "WatchWindowOpen",
    "StateNotDerived",
    # the refusal doors
    "execute_freeze", "schedule_cutover", "send_cutover_request",
    "execute_operator_command", "write_production_branch", "move_stable_kernel_path",
    "apply_era_registry_row", "apply_autopilot_baseline", "edit_candidate_source",
    "rebuild_candidate", "amend_protocol_or_threshold", "waive_failed_evidence",
    # AK7 entry point
    "ComputeWindow", "OperatorFreezeRequest",
    # sealing and evaluation
    "SealedRelease", "seal_champion", "TrustedEvaluation", "run_release_evaluation",
    # transaction, version, rollback
    "NextVersion", "compute_next_version", "build_transaction_plan",
    "RollbackPlan", "verify_archive_target", "build_rollback_plan",
    # drafts
    "EraRowDraft", "draft_era_registry_row", "draft_autopilot_rebaseline_note",
    # operator commands
    "OperatorCommand", "CommandSequenceReview", "validate_command_sequence",
    "ELEMENT_VERBS", "era_row_registry_path",
    # cutover request
    "CutoverRequest", "build_cutover_request",
    # watch window
    "WatchSignalBand", "WatchWindowCloseStep", "WatchWindow", "WatchObservation",
    "WatchWindowProgress", "WatchSignalStanding", "WatchWindowRecommendation",
    "WatchWindowClosure", "watch_window_close_condition", "evaluate_watch_window",
    "close_watch_window", "default_watch_bands",
    # decision package
    "DecisionOption", "DecisionRecommendation", "DecisionPackage",
    "build_decision_package", "render_decision_package",
    # the package
    "PackageFinding", "LinkageSummary", "derive_linkage_summary", "ReleasePackage",
    "assemble_release_package", "render_first_page",
    # self-audits
    "audit_no_write_or_process_paths", "audit_refusal_doors_raise_unconditionally",
    "audit_no_clock_or_self_trigger", "audit_verdict_is_delegated",
]


# =============================================================================
# Identity
# =============================================================================

#: Versioned, because a package names the packager that assembled it and a package
#: assembled by a different one is a different artifact (schemas.py CONVENTIONS).
MODULE_ID = "autokernel.release.packager/v2"

#: The package IS `schemas.SCHEMA_RELEASE_PACKAGE`; this module does not invent a
#: second shape for the same record. `schemas.validate_release_package()` is the
#: contract, and `test_packager.py` asserts a READY package satisfies it.
PACKAGE_SCHEMA = schemas.SCHEMA_RELEASE_PACKAGE

FREEZE_REQUEST_SCHEMA = "epyc.autokernel.freeze_request.v1"
WATCH_WINDOW_SCHEMA = "epyc.autokernel.post_cutover_watch_window.v2"

#: The bus message envelope, verbatim from `coordination/session-bus/`. The cutover
#: request is an ORDINARY bus message; inventing a private envelope for it would
#: put it outside `session_bus.py validate` and outside the routing rules that make
#: `needs_routing_to` / `action_required` structural rather than prose.
CUTOVER_MESSAGE_SCHEMA = "session_bus.msg.v1"

#: Annex K requires every instrument to state the class of record it emits. A
#: release package is a HANDOFF: it licenses nothing on its own and every write it
#: describes is performed by a human.
RECORD_CLASS = ("RELEASE PACKAGE — a handoff for operator execution. It contains no "
                "production write and no authority claim.")

EXECUTED_BY = "operator"

PACKAGE_NOTICE = (
    "AutoKernel assembled this package and executes none of it. The freeze, the "
    "era-registry rows, the AutoPilot rebaseline apply and the cutover are human-only "
    "writes (MEASUREMENT.md:140-142); the cutover is additionally scheduled by whoever "
    "owns the inference, at a boundary that session chooses (OPERATING_CONSTRAINTS.md:41)."
)

#: AK7, stated where a reader looking for the freeze entry point will find it.
AK7_SCOPE_NOTE = (
    "Executing a freeze needs TWO things this module cannot supply: an operator, and a "
    "real compute window. The operator half is a trust boundary — four human-only writes "
    "(MEASUREMENT.md:140-142). The compute half is arithmetic: a T3 release matrix is a "
    "benchmarking program under P-BENCH-1 / P-BENCH-PREFILL-1 / P-GPU-1 release reps, "
    "holding a CPU region or an exclusive device claim for its whole window, on a host "
    "whose uptime tier still qualifies (§10.7 caps an unattended campaign at roughly one "
    "week of uptime). Neither half is available inside an autonomous run, so a freeze is "
    "OUT OF SCOPE for one. An autonomous run may prepare a package and stop at "
    f"{'RELEASE_PACKAGE_READY'!r}; a human takes it from there."
)

TRUSTED_EVALUATOR_TIER = t3.TIER

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PRODUCTION_BRANCH_RE = re.compile(r"^production-(consolidated|speech)-v(\d+)$")

#: §3.3: *"the loop's terminal success state is `RELEASE_PACKAGE_READY`, not
#: `FREEZE_ELIGIBLE`"*. The three states are the usual three outcomes: a package
#: that could not be checked is not a blocked one and is certainly not a ready one.
STATE_READY = "RELEASE_PACKAGE_READY"
STATE_INCOMPLETE = "RELEASE_PACKAGE_INCOMPLETE"
STATE_BLOCKED = "RELEASE_PACKAGE_BLOCKED"
PACKAGE_STATES = (STATE_READY, STATE_INCOMPLETE, STATE_BLOCKED)


# =============================================================================
# Errors — every one is a refusal about MATERIAL or AUTHORITY. A finding about
# the candidate's evidence comes back as a `PackageFinding`, never as an
# exception: raising on a regression would delete the record of the regression.
# =============================================================================

class PackagerError(Exception):
    """Base for every refusal this module raises."""


class PackagerInputError(PackagerError):
    """The material handed in cannot be packaged, and no verdict follows from it."""


class SelfTriggerRefused(PackagerError):
    """Something other than an operator tried to open a freeze request (AK7)."""


class FreezeExecutionRefused(PackagerError):
    """A freeze was asked of the packager. There is no such capability here."""


class CutoverExecutionRefused(PackagerError):
    """A cutover was asked of the packager (§11.3). The package carries a REQUEST."""


class ProductionWriteRefused(PackagerError):
    """A production branch, symlink, era registry or baseline write was reached for."""


class SealRefused(PackagerError):
    """The champion cannot be sealed as presented (§11.1 "sealed release candidate")."""


class EvaluatorNotTrusted(PackagerError):
    """The object offered as the release evaluator is not one (invariant 4)."""


class VersionCollision(PackagerError):
    """The computed next version is already taken, or the incumbent is not the tip."""


class RollbackIncomplete(PackagerError):
    """§10.5: the incumbent is archived, not merely rebuildable. Nothing to fall back to."""


class IncumbentModificationRefused(PackagerError):
    """Something in the package would modify the incumbent (invariant 3, §10.5)."""


class BandsNotFixedBeforeData(PackagerError):
    """§11.5: bands are set at assembly, before the window opens and the data is seen."""


class WatchWindowOpen(PackagerError):
    """The window has not met its close condition; closing it now would invent a pass."""


class StateNotDerived(PackagerError):
    """A package's state was stamped rather than derived from its own findings."""


# =============================================================================
# Small helpers. Local by house style — `plan.py`, `t3.py` and `readiness.py`
# each own theirs, so a module can be audited without reading a sibling's
# private surface. `_timestamp` is deliberately NOT the obvious spelling: it builds
# an aware datetime with `datetime.combine` rather than `.replace(tzinfo=…)`,
# because `audit_no_write_or_process_paths` denies `.replace()` (it is
# `Path.replace`, the symlink-move primitive) and an AST audit cannot tell the two
# apart. A guard satisfiable only by exempting its own call sites is not a guard,
# so the benign homograph is rewritten instead. `.astimezone` is NOT the
# substitute: on a naive datetime it assumes local time, which would silently
# reinterpret every timestamp this module reads.
# =============================================================================

_SEVERITY = {schemas.PASS: 0, schemas.COULD_NOT_CHECK: 1, schemas.FAIL: 2}


def _worst(checks: Iterable[schemas.Check]) -> schemas.Check:
    """FAIL beats COULD_NOT_CHECK beats PASS, carrying every reason forward."""
    worst = schemas.PASS
    reasons: list = []
    for check in checks:
        if not isinstance(check, schemas.Check):
            raise PackagerInputError(
                f"expected a schemas.Check, got {type(check).__name__}")
        if _SEVERITY[check.outcome] > _SEVERITY[worst]:
            worst = check.outcome
        reasons.extend(check.reasons)
    return schemas.Check(worst, tuple(reasons))


def _fail(*reasons: str) -> schemas.Check:
    return schemas.Check(schemas.FAIL, tuple(reasons))


def _cnc(*reasons: str) -> schemas.Check:
    return schemas.Check(schemas.COULD_NOT_CHECK, tuple(reasons))


def _check_dict(check: schemas.Check) -> dict:
    return {"outcome": check.outcome, "reasons": list(check.reasons)}


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PackagerInputError(f"{label}: required, a non-empty string")
    return value


def _opt_text(value: Any, label: str) -> Optional[str]:
    return None if value is None else _text(value, label)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PackagerInputError(f"{label}: required, a bool")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PackagerInputError(f"{label}: required, a positive int")
    return value


def _sha256(value: Any, label: str) -> str:
    _text(value, label)
    # `fullmatch`, not `match(r"^…$")`: `$` also matches before a trailing newline,
    # so a digest read off `sha256sum` without `.strip()` would clear both this
    # check and the placeholder check below (`"0"*64 + "\n"`).
    if not _SHA256_RE.fullmatch(value):
        raise PackagerInputError(f"{label}: {value!r} is not a lowercase sha256 digest")
    if schemas.is_placeholder_digest(value):
        raise PackagerInputError(
            f"{label}: {value!r} is the digest of no bytes at all. It is well-formed and "
            "it means the artifact was never read.")
    return value


def _commit(value: Any, label: str) -> str:
    _text(value, label)
    if not _COMMIT_RE.fullmatch(value):
        raise PackagerInputError(f"{label}: {value!r} is not a full 40-hex commit id")
    if schemas.is_placeholder_digest(value):
        raise PackagerInputError(
            f"{label}: {value!r} names no commit any git command resolves")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    _text(value, label)
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise PackagerInputError(f"{label}: {value!r} is not an ISO-8601 timestamp ({exc})")
    if parsed.tzinfo:
        return parsed
    return datetime.combine(parsed.date(), parsed.time(), tzinfo=timezone.utc)


def _str_tuple(value: Any, label: str, *, non_empty: bool = True) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise PackagerInputError(f"{label}: required, a list/tuple of strings")
    out = tuple(value)
    for item in out:
        _text(item, f"{label}[]")
    if non_empty and not out:
        raise PackagerInputError(f"{label}: must not be empty")
    return out


def _typed_tuple(value: Any, label: str, klass: type, *, non_empty: bool = False) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise PackagerInputError(f"{label}: required, a list/tuple of {klass.__name__}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, klass):
            raise PackagerInputError(
                f"{label}: expected {klass.__name__}, got {type(item).__name__}")
    if non_empty and not out:
        raise PackagerInputError(f"{label}: must not be empty")
    return out


def _mapping(value: Any, label: str, *, non_empty: bool = False) -> Mapping:
    if not isinstance(value, Mapping):
        raise PackagerInputError(f"{label}: required, a mapping")
    if non_empty and not value:
        raise PackagerInputError(f"{label}: must not be empty")
    return value


def _production_version_number(branch: str) -> Optional[int]:
    match = _PRODUCTION_BRANCH_RE.fullmatch(branch)
    return int(match.group(2)) if match else None


def _under_production_tree(path: str) -> bool:
    """True when a path resolves inside one of the FROZEN production trees.

    Sourced from `storage.production_tree_forms()` rather than a literal list, so a
    fourth production tree is declared in one place (invariant 3).
    """
    resolved = path.rstrip("/") + "/"
    for root in storage.production_tree_forms():
        root = root.rstrip("/") + "/"
        if resolved == root or resolved.startswith(root):
            return True
    return False


# =============================================================================
# The refusal doors — §11.2's "may not" list, as code that raises
# =============================================================================

#: Every capability §11.2 denies the packager, mapped to the function that denies
#: it. The map is the SSOT: `audit_refusal_doors_raise_unconditionally()` walks it
#: and proves from the AST that each named function still does nothing but raise,
#: so removing a `raise` fails a test rather than quietly restoring a capability.
#:
#: Naming them at all is deliberate. `serving_runtime.refuse_kernel_freeze()` and
#: `readiness.freeze_eligibility()` established the pattern: a greppable refusal
#: that explains itself is worth more than a prohibition a future caller cannot
#: find, because the caller who cannot find it writes the capability instead.
REFUSED_CAPABILITIES = {
    "edit_source": "edit_candidate_source",
    "rebuild_outside_the_sealed_build": "rebuild_candidate",
    "change_protocols_thresholds_or_scope": "amend_protocol_or_threshold",
    "waive_failed_evidence": "waive_failed_evidence",
    "write_a_production_branch": "write_production_branch",
    "move_a_stable_kernel_symlink": "move_stable_kernel_path",
    "write_an_era_registry_row": "apply_era_registry_row",
    "apply_an_autopilot_baseline": "apply_autopilot_baseline",
    "execute_a_drafted_command": "execute_operator_command",
    "perform_the_freeze": "execute_freeze",
    "perform_the_cutover": "schedule_cutover",
    "send_the_cutover_request": "send_cutover_request",
}


def execute_freeze(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. A freeze crosses four human-only trust boundaries."""
    raise FreezeExecutionRefused(
        "AutoKernel does not freeze. A kernel freeze crosses FOUR human-only trust "
        "boundaries (MEASUREMENT.md:140-142): the freeze itself, the era-registry rows, "
        "the AutoPilot baseline apply, and the pinned human-only path list, which is "
        "branch-pattern-scoped and matches the new production branch the moment it "
        "exists. There is no authority here to hold or to delegate (AK-D6, invariant 5). "
        "The package this module assembles is executed by a human."
    )


def schedule_cutover(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. Cutover is scheduled by whoever owns the inference (§11.3)."""
    raise CutoverExecutionRefused(
        "AutoKernel does not schedule or perform a cutover. OPERATING_CONSTRAINTS.md:41: "
        "a reload 'must be executed BY THAT SESSION, at a moment it chooses; it is never "
        "forced upon that session's workflow from outside'. The package carries a cutover "
        "REQUEST routed on the bus to the inference owner. An autonomous restart is the "
        "preemption that rule exists to prevent (INC-20260728-reload-preemption)."
    )


def send_cutover_request(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. This module has no transport, and it owns no outbox."""
    raise CutoverExecutionRefused(
        "This module writes nothing, including a bus message. `build_cutover_request()` "
        "returns a RECORD; the session that holds the roster id appends it to ITS OWN "
        "outbox (BUS_PROTOCOL rule 1: no file ever has two writers, and an agent may only "
        "address the files of whoever it claims to be)."
    )


def execute_operator_command(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. §11.2: the packager may not execute any command it drafted."""
    raise ProductionWriteRefused(
        "§11.2: the packager may not 'execute any command it drafted'. Pre-validation "
        "here is STATIC — shape, target scope, coverage of the transaction, and a "
        "supplied validation receipt. A packager that ran its own commands to validate "
        "them would have performed the transaction in order to check it."
    )


def write_production_branch(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. Invariant 3: frozen means immutable."""
    raise ProductionWriteRefused(
        "No write to a production-named branch. Invariant 3: no actor builds in or "
        "modifies a production tree, and `human_only_paths.yaml:42-49` covers 'any commit "
        "landing on a frozen production kernel branch'. We version PAST production."
    )


def move_stable_kernel_path(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. Repointing a stable kernel path is the cutover itself."""
    raise ProductionWriteRefused(
        "No repointing of a stable kernel path (/mnt/raid0/llm/kernels/production/*). "
        "That link IS the cutover: the moment it moves, every launcher resolves a "
        "different binary. It is drafted in the transaction's symlink diff and moved by "
        "the operator."
    )


def apply_era_registry_row(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. Era rows are human-only and separately pinned."""
    raise ProductionWriteRefused(
        "No era-registry write. `orchestration/instrument_eras.yaml` is human-only "
        "(MEASUREMENT.md:140-142) and separately pinned (human_only_paths.yaml:35-37). "
        "The package carries a DRAFT row (§1.3 item 2); a freeze whose era row is "
        "unwritten produces evidence nobody can interpret, which is why the draft exists "
        "and why AutoKernel does not write it."
    )


def apply_autopilot_baseline(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. Baseline applies are human-only (E8 precedent)."""
    raise ProductionWriteRefused(
        "No AutoPilot baseline apply. `orchestration/autopilot_baseline.yaml` is "
        "human-only (human_only_paths.yaml:38-40). The E8 precedent is explicit: the "
        "cutover opens a fail-closed rebaseline HOLD until an operator-ratified reseed "
        "writes fresh values and windows (instrument_eras.yaml:166-172). The package "
        "carries the note; the operator ratifies the reseed."
    )


def edit_candidate_source(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. §11.2: the packager may not edit source."""
    raise ProductionWriteRefused(
        "§11.2: the packager may not edit source. The sealed candidate is immutable by "
        "definition (§11.1); a packager that could edit it could change what T3 graded "
        "after T3 graded it."
    )


def rebuild_candidate(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. §11.2: no rebuild outside the sealed build."""
    raise ProductionWriteRefused(
        "§11.2: the packager may not rebuild the candidate outside the sealed build. "
        "Invariant 2: release evidence is produced by the same full candidate that is "
        "frozen — no promotion-time reconciliation. A rebuilt candidate is a different "
        "candidate, and §10.5 says so from the other side: rebuilding a commit under a "
        "drifted toolchain does not reproduce its binary."
    )


def amend_protocol_or_threshold(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. Invariant 17 / P-AK-SEARCH-1 denial 6."""
    raise ProductionWriteRefused(
        "§11.2: the packager may not change protocols, thresholds or scope. These are "
        "read-only for automated processes (MEASUREMENT.md:119-120, invariant 17), and "
        "scope is mechanically derived (invariant 18) precisely so the measured party "
        "cannot set it. A discovered gap is RECORDED and blocks release eligibility; it "
        "is never patched."
    )


def waive_failed_evidence(*_args: Any, **_kwargs: Any) -> None:
    """Always raises. §10.4: a waiver is human-authored; the gate only verifies it."""
    raise ProductionWriteRefused(
        "§11.2: the packager may not waive failed evidence. A waiver is a human-authored "
        "`epyc.autokernel.operator_waiver.v1` document stored under the trust-boundary "
        "path set and hash-pinned into the T3 bundle. The evaluator verifies its hash and "
        "its predicate and never judges its merits (§10.4); the packager does not even do "
        "that much — it reports which waivers T3 verified."
    )


# =============================================================================
# AK7 — the freeze-request entry point the operator invokes
# =============================================================================

OPERATOR_AUTHORITY = "operator"

#: Tokens that betray a machine actor in an identity field. A freeze request whose
#: requester is the loop is the loop triggering itself, which is the one thing AK7
#: must be unable to express.
#:
#: OWNED BY `schemas.py`, not by this module, and re-exported here so existing
#: importers keep working. It lived here first, one layer ABOVE the gate that
#: needed it: `t3.verify_waiver` accepted any non-empty `authorized_by`, so a
#: waiver attributed to `autokernel` verified as human-attested and T3's own
#: verdict read PASS_WITH_WAIVER — this module's refusal only stopped it from
#: reaching a package. Both planes now read one vocabulary.
MACHINE_ACTOR_TOKENS = schemas.MACHINE_ACTOR_TOKENS


def _machine_actor_tokens(identity: str) -> tuple:
    return schemas.machine_actor_tokens(identity)


def _require_human_actor(identity: str, label: str, *, error: type = SelfTriggerRefused):
    found = _machine_actor_tokens(identity)
    if found:
        raise error(
            f"{label}: {identity!r} names a machine actor ({', '.join(found)}). "
            "Freeze and cutover are human-only writes (MEASUREMENT.md:140-142); an "
            "automated identity in this field is the loop authorising itself.")
    return identity


@dataclass(frozen=True)
class ComputeWindow:
    """The real compute window a freeze needs, declared by whoever owns it.

    Not decoration. A T3 release matrix runs release reps under each phase's own
    protocol while holding a CPU region or an exclusive device claim for the whole
    window (`P-AK-SEARCH-1` precondition 1 states the shape for search; a release
    gate does not get to hold less). §10.7 additionally caps the host: an uptime
    tier violation invalidates the arm, and a reboot is operator authority. A
    package that names no window is asking for a freeze nobody has budgeted.
    """

    window_id: str
    owner: str
    opens_at: str
    closes_at: str
    purpose: str

    def __post_init__(self) -> None:
        _text(self.window_id, "ComputeWindow.window_id")
        _require_human_actor(_text(self.owner, "ComputeWindow.owner"),
                             "ComputeWindow.owner", error=PackagerInputError)
        opens = _timestamp(self.opens_at, "ComputeWindow.opens_at")
        closes = _timestamp(self.closes_at, "ComputeWindow.closes_at")
        if closes <= opens:
            raise PackagerInputError(
                "ComputeWindow: closes_at must be after opens_at; a window of zero or "
                "negative length is a window nobody can run a matrix in")
        _text(self.purpose, "ComputeWindow.purpose")

    @property
    def hours(self) -> float:
        return (_timestamp(self.closes_at, "closes_at")
                - _timestamp(self.opens_at, "opens_at")).total_seconds() / 3600.0

    def to_dict(self) -> dict:
        return {"window_id": self.window_id, "owner": self.owner,
                "opens_at": self.opens_at, "closes_at": self.closes_at,
                "purpose": self.purpose, "hours": self.hours}


@dataclass(frozen=True)
class OperatorFreezeRequest:
    """AK7's entry point: the operator asks for a freeze; the loop cannot ask itself.

    **This module never constructs one.** That is the self-trigger defence and it is
    checkable: `audit_no_clock_or_self_trigger()` parses this module and FAILs if
    `OperatorFreezeRequest(...)` appears anywhere in it, and separately if any clock
    call appears. A module with no clock cannot decide that it is time to freeze,
    and a module that cannot mint this object cannot answer its own request. The
    only way one exists is for a caller outside this module to build it.

    `readiness_signal_ref` is CONTEXT, not a cause. A readiness signal is advisory
    and explicitly not a trigger (AK-D3, `P-AK-SEARCH-1` denial 5), so there is no
    `triggered_by` field for it to occupy — the request records what the operator
    was looking at, and the operator remains the reason.
    """

    request_id: str
    campaign_id: str
    source_tree: str
    requested_by: str
    requested_at: str
    authority: str
    compute_window: ComputeWindow
    reason: str
    readiness_signal_ref: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.request_id, "OperatorFreezeRequest.request_id")
        if not self.request_id.startswith("akfr-"):
            raise PackagerInputError(
                "OperatorFreezeRequest.request_id: must start with 'akfr-'")
        _text(self.campaign_id, "OperatorFreezeRequest.campaign_id")
        if not self.campaign_id.startswith("ak-"):
            raise PackagerInputError(
                "OperatorFreezeRequest.campaign_id: must start with 'ak-'")
        if self.source_tree not in schemas.SOURCE_TREES:
            raise PackagerInputError(
                f"OperatorFreezeRequest.source_tree: {self.source_tree!r} is not one of "
                f"{sorted(schemas.SOURCE_TREES)}. `serving_runtime` has no source tree "
                "and releases through the §11.6 three-gate stack-change path, which this "
                "packager does not travel (AK-D9, AK-D23).")
        if self.authority != OPERATOR_AUTHORITY:
            raise SelfTriggerRefused(
                f"OperatorFreezeRequest.authority: {self.authority!r} is not "
                f"{OPERATOR_AUTHORITY!r}. A freeze request carries operator authority or "
                "it is not a freeze request (invariant 5, MEASUREMENT.md:140-142).")
        _require_human_actor(_text(self.requested_by, "OperatorFreezeRequest.requested_by"),
                             "OperatorFreezeRequest.requested_by")
        requested = _timestamp(self.requested_at, "OperatorFreezeRequest.requested_at")
        if not isinstance(self.compute_window, ComputeWindow):
            raise PackagerInputError(
                "OperatorFreezeRequest.compute_window: required, a ComputeWindow. "
                + AK7_SCOPE_NOTE)
        if _timestamp(self.compute_window.closes_at, "closes_at") <= requested:
            raise PackagerInputError(
                "OperatorFreezeRequest: the declared compute window closes at or before "
                "the request was made, so there is no window in which to run the matrix")
        _text(self.reason, "OperatorFreezeRequest.reason")
        _opt_text(self.readiness_signal_ref, "OperatorFreezeRequest.readiness_signal_ref")

    def to_dict(self) -> dict:
        return {
            "schema": FREEZE_REQUEST_SCHEMA,
            "request_id": self.request_id,
            "campaign_id": self.campaign_id,
            "source_tree": self.source_tree,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "authority": self.authority,
            "compute_window": self.compute_window.to_dict(),
            "reason": self.reason,
            "readiness_signal_ref": self.readiness_signal_ref,
            "scope_note": AK7_SCOPE_NOTE,
        }


# =============================================================================
# Sealing the champion (§11.1 "sealed release candidate")
# =============================================================================

@dataclass(frozen=True)
class SealedRelease:
    """The champion, sealed: `t3.SealedCandidate` plus the build receipt and when.

    `t3.SealedCandidate` already binds the §3.2 identity set. What it does not
    carry is the build receipt digest the package schema requires, or the moment
    the seal closed — both of which belong to the packager rather than to the gate,
    because the gate adjudicates a seal it is handed and this module is what hands
    it over.
    """

    champion_id: str
    candidate: t3.SealedCandidate
    build_receipt_sha256: str
    seal_inputs_ref: str
    sealed_at: str

    def __post_init__(self) -> None:
        _text(self.champion_id, "SealedRelease.champion_id")
        if not isinstance(self.candidate, t3.SealedCandidate):
            raise PackagerInputError("SealedRelease.candidate: must be a t3.SealedCandidate")
        _sha256(self.build_receipt_sha256, "SealedRelease.build_receipt_sha256")
        _text(self.seal_inputs_ref, "SealedRelease.seal_inputs_ref")
        _timestamp(self.sealed_at, "SealedRelease.sealed_at")

    @property
    def backends(self) -> tuple:
        return tuple(sorted(self.candidate.binary_sha256))

    def rollup_binary_sha256(self) -> str:
        """One digest over the per-backend binary map.

        The package schema's `sealed_candidate.binary_sha256` is a single digest, and
        a source tree serves up to two binaries (§1.5). A roll-up is the only honest
        single value — so the per-backend map rides beside it in the same block under
        `binary_sha256_by_backend`, and nothing is collapsed away.
        """
        return schemas.content_hash(dict(self.candidate.binary_sha256))

    def rollup_linkage_sha256(self) -> str:
        return schemas.content_hash(dict(self.candidate.linkage_sha256))

    def to_dict(self) -> dict:
        return {
            "champion_id": self.champion_id,
            "candidate_id": self.candidate.candidate_id,
            "seal_sha256": self.candidate.seal_sha256,
            "binary_sha256": self.rollup_binary_sha256(),
            "binary_sha256_by_backend": dict(self.candidate.binary_sha256),
            "linkage_sha256": self.rollup_linkage_sha256(),
            "linkage_sha256_by_backend": dict(self.candidate.linkage_sha256),
            "build_receipt_sha256": self.build_receipt_sha256,
            "seal_inputs_ref": self.seal_inputs_ref,
            "sealed_at": self.sealed_at,
            "source_tree": self.candidate.source_tree,
            "candidate_branch": self.candidate.candidate_branch,
            "candidate_commit": self.candidate.candidate_commit,
            "production_base_commit": self.candidate.production_base_commit,
            "evaluator_bundle_sha256": self.candidate.evaluator_bundle_sha256,
            "scope_manifest_sha256": self.candidate.scope_manifest_sha256,
            "evidence_tree_sha256": self.candidate.evidence_tree_sha256,
            "build_dirs": dict(self.candidate.build_dirs),
            "overlay_present": self.candidate.overlay_present,
            "tree_clean": self.candidate.tree_clean,
            "ancestry_clean": self.candidate.ancestry_clean,
        }


def seal_champion(*, champion_id: str, candidate: t3.SealedCandidate,
                  build_receipt_sha256: str, seal_inputs_ref: str, sealed_at: str,
                  pinned_evaluator_bundle_sha256: str, incumbent_branch: str,
                  incumbent_commit: str) -> SealedRelease:
    """Seal the champion, or refuse and say which of the six ways it failed.

    AK6's checklist names six refusals and every one of them is here, because each
    is a way a package could look complete while being unreleasable:

      * **missing hashes** — a backend the tree serves with no binary, linkage or
        build directory. Without them there is nothing to install and nothing to
        reproduce at rollback;
      * **evaluator drift** — the seal's evaluator bundle is not the campaign's
        pinned one. `P-AK-SEARCH-1` precondition 5 voids every record in a window
        on exactly this drift, and invariant 17 stops release eligibility for a
        lineage whose judge moved;
      * **dirty ancestry / dirty tree / missing overlay** — `bench-cpu.md:38-44`
        defines candidate release identity as a *clean committed tree whose binary
        reports that commit*, and the promotion checklist requires the agent-file
        overlay baked in so the new production tree ships freeze-aware agent files;
      * **incumbent modification** — a build directory inside a frozen production
        tree. Invariant 3, and the reason we version past production;
      * **a base that is not the incumbent tip** — invariant 1. A seal anchored on
        a commit production has already moved off is measuring against a
        denominator that no longer exists (AK-D22, `ANCHOR_MOVED`);
      * **a production-named candidate branch** — already refused by
        `t3.SealedCandidate`, restated here only because this is where a caller
        looks for the list.
    """
    if not isinstance(candidate, t3.SealedCandidate):
        raise PackagerInputError("seal_champion: candidate must be a t3.SealedCandidate")
    _text(incumbent_branch, "seal_champion: incumbent_branch")
    _commit(incumbent_commit, "seal_champion: incumbent_commit")
    _sha256(pinned_evaluator_bundle_sha256, "seal_champion: pinned_evaluator_bundle_sha256")

    if candidate.evaluator_bundle_sha256 != pinned_evaluator_bundle_sha256:
        raise SealRefused(
            f"evaluator drift: the seal names bundle "
            f"{candidate.evaluator_bundle_sha256[:12]} while the campaign pins "
            f"{pinned_evaluator_bundle_sha256[:12]}. The evaluator that scored this "
            "lineage is not the evaluator of record; every record in the affected "
            "window is void (P-AK-SEARCH-1 precondition 5, invariant 17).")

    missing_state = [name for name in ("overlay_present", "tree_clean", "ancestry_clean")
                     if getattr(candidate, name) is not True]
    if missing_state:
        raise SealRefused(
            f"the candidate is not sealable: {missing_state} are not established. "
            "bench-cpu.md:38-44 defines release identity as a clean committed tree whose "
            "binary reports that commit, and the promotion checklist requires the "
            "agent-file overlay baked into the candidate.")

    if candidate.production_base_commit != incumbent_commit:
        raise SealRefused(
            f"the seal is anchored on {candidate.production_base_commit[:12]} but the "
            f"incumbent tip is {incumbent_commit[:12]}. Invariant 1: every campaign is "
            "anchored on the CURRENT production tip; a seal against a moved anchor has a "
            "denominator that no longer exists (AK-D22).")

    served = sorted(b for b, tree in schemas.SOURCE_TREE_BY_BACKEND.items()
                    if tree == candidate.source_tree)
    gaps: list = []
    for backend in served:
        for label, mapping in (("binary_sha256", candidate.binary_sha256),
                               ("linkage_sha256", candidate.linkage_sha256),
                               ("build_dirs", candidate.build_dirs)):
            if backend not in mapping:
                gaps.append(f"{backend}.{label}")
    if gaps:
        raise SealRefused(
            f"missing hashes: {gaps}. §1.5 — freeze scope is the union of backends the "
            f"tree serves ({served}); a backend with no recorded binary, linkage or build "
            "directory has nothing to install and nothing to reproduce at rollback. "
            "Narrowing has exactly one route, the §3.2 unchanged test, and it leaves a "
            "transfer receipt rather than a gap.")

    inside = sorted(f"{b}={p}" for b, p in candidate.build_dirs.items()
                    if _under_production_tree(p))
    if inside:
        raise IncumbentModificationRefused(
            f"the candidate was built inside a FROZEN production tree: {inside}. "
            "Invariant 3: no actor builds in or modifies a production tree. All kernel "
            "work happens on llama.cpp-experimental branches (CLAUDE.md).")

    if _production_version_number(incumbent_branch) is None:
        raise SealRefused(
            f"incumbent_branch {incumbent_branch!r} is not a production version branch, "
            "so no successor version can be computed from it")

    return SealedRelease(champion_id=champion_id, candidate=candidate,
                         build_receipt_sha256=build_receipt_sha256,
                         seal_inputs_ref=seal_inputs_ref, sealed_at=sealed_at)


# =============================================================================
# Running T3 through the TRUSTED evaluator (§11.2, invariant 4)
# =============================================================================

@dataclass(frozen=True)
class TrustedEvaluation:
    """One T3 run, plus the cross-checks that it graded THIS request.

    `check` is not a re-grading of the candidate — that would be the packager
    marking its own homework, and `audit_verdict_is_delegated()` proves it does not
    happen. It is a check on the SEAM: same request fingerprint, same run id, same
    mode, a bundle that rehashes to its own payload. An evaluator that returned a
    verdict for a different request is not this run's evaluator, however green the
    verdict looks.
    """

    result: t3.T3Result
    evaluator_class: str
    evaluator_tier: str
    request_fingerprint: str
    check: schemas.Check

    def __post_init__(self) -> None:
        if not isinstance(self.result, t3.T3Result):
            raise PackagerInputError("TrustedEvaluation.result: must be a t3.T3Result")
        _text(self.evaluator_class, "TrustedEvaluation.evaluator_class")
        _text(self.evaluator_tier, "TrustedEvaluation.evaluator_tier")
        _text(self.request_fingerprint, "TrustedEvaluation.request_fingerprint")
        if not isinstance(self.check, schemas.Check):
            raise PackagerInputError("TrustedEvaluation.check: must be a schemas.Check")

    @property
    def verdict(self) -> str:
        return self.result.verdict

    @property
    def bundle_sha256(self) -> Optional[str]:
        return None if self.result.bundle is None else self.result.bundle.bundle_sha256

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "bundle_sha256": self.bundle_sha256,
            "run_id": self.result.run_id,
            "mode": self.result.mode,
            "fingerprint": self.request_fingerprint,
            "evaluator_class": self.evaluator_class,
            "evaluator_tier": self.evaluator_tier,
            "seam_check": _check_dict(self.check),
            "phase_results": {p.phase_id: p.to_dict() for p in self.result.phase_results},
            "verdict_computation": self.result.verdict_computation.to_dict(),
            "requires_human_code_review": self.result.requires_human_code_review,
            "first_page_notice": self.result.first_page_notice,
            "rerun": self.result.rerun.to_dict(),
        }


def run_release_evaluation(request: t3.T3Request, *, evaluator: Any) -> TrustedEvaluation:
    """Run T3 by DELEGATING to the injected release-tier evaluator.

    The seam is `evaluator.api.ReleaseTierEvaluator`: `tier == "T3"` plus
    `evaluate_release(request)`. The packager does not call `t3.run_t3` directly and
    `audit_verdict_is_delegated()` proves it — not because the function is wrong,
    but because invariant 4 makes actor, evaluator and packager distinct authority
    domains, and a packager that reached into the gate's internals would be one
    domain wearing two hats. Going through the seam also records WHICH evaluator
    ran, which is a fact the package has to carry.
    """
    if not isinstance(request, t3.T3Request):
        raise PackagerInputError("run_release_evaluation: request must be a t3.T3Request")
    tier = getattr(evaluator, "tier", None)
    if tier != TRUSTED_EVALUATOR_TIER:
        raise EvaluatorNotTrusted(
            f"the object offered as the release evaluator declares tier {tier!r}, not "
            f"{TRUSTED_EVALUATOR_TIER!r}. `evaluator.api.admit_tier()` refuses T3 by name "
            "and points at the release-tier seam; anything else grading a release would "
            "be producing a release-shaped record under a search protocol.")
    evaluate = getattr(evaluator, "evaluate_release", None)
    if not callable(evaluate):
        raise EvaluatorNotTrusted(
            "the release evaluator has no callable `evaluate_release`; the "
            "ReleaseTierEvaluator seam is unimplemented and nothing has graded this "
            "candidate")

    expected = request.fingerprint()
    result = evaluate(request)
    if not isinstance(result, t3.T3Result):
        raise EvaluatorNotTrusted(
            f"the release evaluator returned {type(result).__name__}, not a t3.T3Result. "
            "A verdict this module had to interpret is a verdict this module computed.")

    reasons: list = []
    if result.fingerprint != expected:
        reasons.append(
            f"the evaluator graded fingerprint {result.fingerprint[:12]} while this "
            f"request is {expected[:12]}: the verdict belongs to a different sealed "
            "candidate, plan, protocol or waiver set")
    if result.run_id != request.run_id:
        reasons.append(f"run id {result.run_id!r} is not the requested {request.run_id!r}")
    if result.mode != request.mode:
        reasons.append(f"mode {result.mode!r} is not the requested {request.mode!r}")
    if result.bundle is None:
        if result.verdict != "FAIL":
            reasons.append(
                f"the seal did not close but the verdict is {result.verdict!r}; a release "
                "whose evidence nobody can rehash is not a passing release")
    else:
        payload = result.bundle.payload
        recomputed = schemas.content_hash(payload)
        if recomputed != result.bundle.bundle_sha256:
            reasons.append(
                f"the bundle digest {result.bundle.bundle_sha256[:12]} is not the hash of "
                f"its own payload ({recomputed[:12]})")
        if payload.get("sealed_candidate", {}).get("candidate_id") != \
                request.sealed.candidate_id:
            reasons.append(
                "the sealed bundle names a different candidate than the request did")
    check = _fail(*reasons) if reasons else schemas.Check(schemas.PASS)
    return TrustedEvaluation(
        result=result, evaluator_class=type(evaluator).__name__,
        evaluator_tier=str(tier), request_fingerprint=expected, check=check)


# =============================================================================
# The next version (§10.2 phase 8: "exact next version, branch/tag")
# =============================================================================

@dataclass(frozen=True)
class NextVersion:
    """The successor version, derived from the incumbent branch and refused on collision."""

    family: str
    incumbent_branch: str
    incumbent_version_number: int
    next_version_number: int
    next_branch: str
    next_tag: str
    era_prefix: str

    def to_dict(self) -> dict:
        return {"family": self.family, "incumbent_branch": self.incumbent_branch,
                "incumbent_version_number": self.incumbent_version_number,
                "next_version_number": self.next_version_number,
                "next_branch": self.next_branch, "next_tag": self.next_tag,
                "era_prefix": self.era_prefix}


def compute_next_version(*, incumbent_branch: str,
                         existing_branches: Sequence[str],
                         existing_tags: Sequence[str] = ()) -> NextVersion:
    """`production-consolidated-v8` + the branch list ⇒ `…-v9`, or a refusal.

    Two refusals, and the second is the one that matters. A **reused version name**
    is the obvious collision. The subtle one is an incumbent that is not the TIP of
    its own series: if a `-v10` exists while the caller believes `-v8` is current,
    the successor computed from v8 is v9, which is behind reality. The version
    number is also the era number (`E8-cpu-kernel`, `instrument_eras.yaml:140-172`),
    so a wrong successor writes an era row that misorders the evidence timeline —
    and `MEASUREMENT.md:233` requires every number to be era-labelled.
    """
    _text(incumbent_branch, "compute_next_version: incumbent_branch")
    branches = frozenset(_str_tuple(existing_branches,
                                    "compute_next_version: existing_branches",
                                    non_empty=False))
    tags = frozenset(_str_tuple(existing_tags, "compute_next_version: existing_tags",
                                non_empty=False))
    match = _PRODUCTION_BRANCH_RE.fullmatch(incumbent_branch)
    if match is None:
        raise PackagerInputError(
            f"compute_next_version: {incumbent_branch!r} is not a production version "
            "branch (production-consolidated-vN / production-speech-vN), so it has no "
            "computable successor")
    family, current = match.group(1), int(match.group(2))

    # TAGS COUNT for staleness, not only for the name collision below. A
    # `production-consolidated-v10` tag with no branch of that name still says the
    # series moved past v8 — the version number is the era number, and an era row
    # written for E9 after E10 exists misorders the evidence timeline whichever kind
    # of ref recorded the newer version.
    higher = sorted(
        b for b in (branches | tags)
        if (m := _PRODUCTION_BRANCH_RE.fullmatch(b)) is not None
        and m.group(1) == family and int(m.group(2)) > current)
    if higher:
        raise VersionCollision(
            f"{incumbent_branch!r} is not the tip of its series: {higher} already exist "
            "as a branch or a tag. "
            "The successor computed from a stale incumbent is behind production, and the "
            "version number is the era number, so the era row would misorder the "
            "evidence timeline (MEASUREMENT.md:233).")

    next_number = current + 1
    next_branch = f"production-{family}-v{next_number}"
    next_tag = next_branch
    taken = sorted({next_branch} & branches | {next_tag} & tags)
    if taken:
        raise VersionCollision(
            f"version name(s) {taken} already exist. A reused production version name "
            "makes two different kernels indistinguishable in every receipt, era row and "
            "rollback anchor that names one.")
    return NextVersion(
        family=family, incumbent_branch=incumbent_branch,
        incumbent_version_number=current, next_version_number=next_number,
        next_branch=next_branch, next_tag=next_tag, era_prefix=f"E{next_number}")


# =============================================================================
# Rollback (§10.5) — the incumbent is ARCHIVED, not merely rebuildable
# =============================================================================

@dataclass(frozen=True)
class RollbackPlan:
    """Where production goes back to, and the proof that it can.

    §10.5 is the whole reason this is not just a branch name: *"Rebuilding an old
    commit under a drifted toolchain does not reproduce that binary."* The v8
    quality gate compared against a PRESERVED binary at
    `/mnt/raid0/llm/llama.cpp-v7-build-backup-6ad45fa3ff/cpu-bin/llama-server`, and
    `/mnt/raid0/llm/kernels/archive/` is empty. A rollback plan naming a commit and
    hoping is not a rollback plan.
    """

    rollback_branch: str
    rollback_head: str
    incumbent_archive_path: str
    #: ((backend, path, sha256), …) — every backend the tree serves.
    incumbent_binaries: tuple
    #: ((backends, path, sha256), …), attributed. The attribution is NOT minted
    #: here — `t3.ArchivedBuild.libraries` carries it and this field transports it
    #: unchanged. That direction is the whole point: a rollback plan that invented an
    #: attribution would put a fact in the operator's package that nothing measured,
    #: and on a three-ggml-generation host that fact is exactly the one a rollback
    #: most needs to be true.
    incumbent_libraries: tuple
    #: ((link_path, restore_target), …) — putting the stable paths back.
    stable_path_restore: tuple
    archive_check: schemas.Check
    verified_at: str
    #: TRISTATE, and `None` is the default. §11.5 requires the rollback anchor to
    #: stay live and verified for the whole watch window; a `bool` defaulting to
    #: `True` answered that requirement for every caller who never thought about it,
    #: which is "we did not check" wearing "we checked". `None` is the third state and
    #: `assemble_release_package` reports it as COULD_NOT_CHECK.
    anchor_live: Optional[bool] = None

    def __post_init__(self) -> None:
        _text(self.rollback_branch, "RollbackPlan.rollback_branch")
        _commit(self.rollback_head, "RollbackPlan.rollback_head")
        _text(self.incumbent_archive_path, "RollbackPlan.incumbent_archive_path")
        triples: list = []
        for i, entry in enumerate(self.incumbent_binaries or ()):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise PackagerInputError(
                    f"RollbackPlan.incumbent_binaries[{i}]: expected "
                    "(backend, path, sha256)")
            backend = _text(entry[0], f"RollbackPlan.incumbent_binaries[{i}].backend")
            if backend not in schemas.BACKENDS:
                raise PackagerInputError(
                    f"RollbackPlan.incumbent_binaries[{i}]: {backend!r} is not a known "
                    "backend")
            triples.append((
                backend, _text(entry[1], f"RollbackPlan.incumbent_binaries[{i}].path"),
                _sha256(entry[2], f"RollbackPlan.incumbent_binaries[{i}].sha256")))
        object.__setattr__(self, "incumbent_binaries", tuple(triples))
        libraries: list = []
        for i, entry in enumerate(self.incumbent_libraries or ()):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise PackagerInputError(
                    f"RollbackPlan.incumbent_libraries[{i}]: expected "
                    "(backends, path, sha256). The unattributed (path, sha256) shape is "
                    "what this field used to carry, and it is the shape that cannot say "
                    "which of three ggml generations a preserved library belongs to.")
            backends = entry[0]
            if isinstance(backends, str):
                raise PackagerInputError(
                    f"RollbackPlan.incumbent_libraries[{i}].backends: a single string is "
                    "not a backend set")
            backends = _str_tuple(
                backends, f"RollbackPlan.incumbent_libraries[{i}].backends")
            for backend in backends:
                if backend not in schemas.BACKENDS:
                    raise PackagerInputError(
                        f"RollbackPlan.incumbent_libraries[{i}].backends: {backend!r} is "
                        "not a known backend")
            libraries.append((
                tuple(sorted(set(backends))),
                _text(entry[1], f"RollbackPlan.incumbent_libraries[{i}].path"),
                _sha256(entry[2], f"RollbackPlan.incumbent_libraries[{i}].sha256")))
        object.__setattr__(self, "incumbent_libraries", tuple(libraries))
        if not self.incumbent_binaries:
            raise RollbackIncomplete(
                "RollbackPlan.incumbent_binaries is empty: there is no archived binary to "
                "fall back to (§10.5). A commit id is not a rollback target.")
        # The attribution has to hold HERE, not only where the plan is compiled.
        # `verify_archive_target()` FAILs a backend with no attributed library, but
        # that is one door: `assemble_release_package()` takes a `RollbackPlan`
        # object, and `archive_check` is a FIELD on it. A hand-built plan carrying
        # `incumbent_libraries=()` and `archive_check=Check(PASS)` reaches the
        # operator's package with the attribution requirement never asked — the
        # `unchanged_view()` lesson (README seam 1) in a second place. Deleting the
        # binary instead of supplying the library is not an escape: that drops the
        # backend out of `RollbackPlan.backends`, which `assemble_release_package`
        # already reports as ROLLBACK_MISSING_BACKEND against the sealed set.
        attributed = {b for backends, _p, _d in self.incumbent_libraries
                      for b in backends}
        unattributed = sorted(
            {b for b, _p, _d in self.incumbent_binaries} - attributed)
        if unattributed:
            raise RollbackIncomplete(
                f"RollbackPlan: {unattributed} would be rolled back to an archived "
                f"binary with no attributed library (the plan attributes libraries to "
                f"{sorted(attributed)}). §10.5 archives built binaries AND linked "
                "libraries; on a three-ggml-generation host an unattributed rollback "
                "resolves the binary against whatever is on the path at rollback time, "
                "which is the 2026-07-31 linkage incident with a longer fuse.")
        pairs: list = []
        for i, entry in enumerate(self.stable_path_restore or ()):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise PackagerInputError(
                    f"RollbackPlan.stable_path_restore[{i}]: expected "
                    "(link_path, restore_target)")
            pairs.append((_text(entry[0], f"RollbackPlan.stable_path_restore[{i}].link"),
                          _text(entry[1], f"RollbackPlan.stable_path_restore[{i}].target")))
        if not pairs:
            raise RollbackIncomplete(
                "RollbackPlan.stable_path_restore is empty: the plan does not say where "
                "the stable kernel paths point after a rollback, so a rollback would "
                "restore the branch and leave every launcher on the new binary")
        object.__setattr__(self, "stable_path_restore", tuple(pairs))
        if not isinstance(self.archive_check, schemas.Check):
            raise PackagerInputError("RollbackPlan.archive_check: must be a schemas.Check")
        _timestamp(self.verified_at, "RollbackPlan.verified_at")
        if self.anchor_live is not None:
            _bool(self.anchor_live, "RollbackPlan.anchor_live")

    @property
    def backends(self) -> tuple:
        return tuple(sorted({b for b, _p, _d in self.incumbent_binaries}))

    def rollup_binary_sha256(self) -> str:
        """One digest over the archived binary set — same roll-up rule as the seal."""
        return schemas.content_hash(
            {backend: digest for backend, _path, digest in self.incumbent_binaries})

    def to_dict(self) -> dict:
        return {
            "rollback_branch": self.rollback_branch,
            "rollback_head": self.rollback_head,
            "incumbent_archive_path": self.incumbent_archive_path,
            "incumbent_binary_sha256": self.rollup_binary_sha256(),
            "incumbent_binaries": [
                {"backend": b, "path": p, "sha256": d}
                for b, p, d in self.incumbent_binaries],
            "incumbent_libraries": [
                {"backends": list(b), "path": p, "sha256": d}
                for b, p, d in self.incumbent_libraries],
            "stable_path_restore": [list(pair) for pair in self.stable_path_restore],
            "archive_check": _check_dict(self.archive_check),
            "verified_at": self.verified_at,
            "anchor_live": self.anchor_live,
            "executed_by": EXECUTED_BY,
        }


def verify_archive_target(archive: t3.IncumbentArchive, *, backends: Sequence[str],
                          incumbent_branch: str, incumbent_commit: str,
                          expected_binary_sha256: Mapping[str, str]) -> schemas.Check:
    """Is the thing we would roll back to actually there, and is it the incumbent?

    Four ways this fails, all seen in the wild or named in the design:

      * **no N−1 entry** for the incumbent branch/commit — nothing archived;
      * **`rebuilt=True`** — §10.5 in one word. A rebuild of the incumbent commit is
        a different binary; archiving it and calling it the incumbent is the defect
        the clause exists to name;
      * **a hash mismatch** against the incumbent's recorded binary — the archive
        holds *something*, and it is not what production is running;
      * **a scratch archive root** — `MEASUREMENT.md:146-156` forbids scratch paths
        as the citation of record, and a rollback anchor one `tmp` sweep away from
        vanishing is worse than an absent one, because it reads as present.

    Returns COULD_NOT_CHECK — never PASS — when a backend's expected digest was not
    supplied. An unchecked archive is not a verified one.
    """
    if not isinstance(archive, t3.IncumbentArchive):
        raise PackagerInputError("verify_archive_target: archive must be a t3.IncumbentArchive")
    wanted = _str_tuple(backends, "verify_archive_target: backends")
    _text(incumbent_branch, "verify_archive_target: incumbent_branch")
    _commit(incumbent_commit, "verify_archive_target: incumbent_commit")
    _mapping(expected_binary_sha256, "verify_archive_target: expected_binary_sha256")

    entry = archive.entry(t3.ARCHIVE_GENERATION_N1)
    if entry is None:
        return _fail(
            f"the incumbent archive has no {t3.ARCHIVE_GENERATION_N1} entry. §10.5: the "
            "freeze transaction archives the incumbent's built binaries and linked "
            "libraries, because rebuilding an old commit under a drifted toolchain does "
            "not reproduce that binary.")

    reasons: list = []
    unknown: list = []
    if entry.branch != incumbent_branch:
        reasons.append(f"the archived branch is {entry.branch!r}, not the incumbent "
                       f"{incumbent_branch!r}")
    if entry.commit != incumbent_commit:
        reasons.append(f"the archived head is {entry.commit[:12]}, not the incumbent "
                       f"{incumbent_commit[:12]}")
    if entry.rebuilt:
        reasons.append(
            "the archived build is marked `rebuilt`. §10.5: rebuilding an old commit "
            "under a drifted toolchain does not reproduce that binary, so a rebuild is "
            "not the incumbent and cannot be the rollback target.")
    if storage.is_scratch_path(entry.archive_root):
        reasons.append(
            f"the archive root {entry.archive_root!r} is a scratch path. Evidence of "
            "record may not live there (MEASUREMENT.md:146-156), and a rollback anchor "
            "one sweep away from deletion reads as present while being absent.")

    archived = {digest for _path, digest in entry.binaries}
    for backend in wanted:
        expected = expected_binary_sha256.get(backend)
        if expected is None:
            unknown.append(
                f"{backend}: no incumbent binary digest was supplied, so the archive "
                "cannot be checked against what production is actually running")
            continue
        if expected not in archived:
            reasons.append(
                f"{backend}: the incumbent binary {expected[:12]} is not among the "
                f"archived binaries {sorted(d[:12] for d in archived)}")

    # What the backend attribution on `ArchivedBuild.libraries` is FOR. §10.5
    # archives "built binaries AND linked libraries" because a preserved binary
    # whose libraries were not preserved with it resolves against whatever is on the
    # path at rollback time. Until the attribution existed this was uncheckable: an
    # archive with one library and four backends looked identical to an archive with
    # one library per backend. Now the hole has a name.
    for backend in wanted:
        if not entry.libraries_for(backend):
            reasons.append(
                f"{backend}: the archive attributes no preserved library to this "
                f"backend (it attributes to {list(entry.attributed_backends)}). §10.5 "
                "archives binaries AND linked libraries; a preserved binary whose "
                "libraries were not preserved with it resolves against whatever is on "
                "the path at rollback time, which is the 2026-07-31 ggml-linkage "
                "incident with a longer fuse.")

    entry_checks = [c for c in archive.check() if isinstance(c, schemas.Check)]
    parts = [_fail(*reasons)] if reasons else []
    if unknown:
        parts.append(_cnc(*unknown))
    parts.extend(entry_checks)
    return _worst(parts) if parts else schemas.Check(schemas.PASS)


def build_rollback_plan(*, archive: t3.IncumbentArchive, backends: Sequence[str],
                        incumbent_branch: str, incumbent_commit: str,
                        expected_binary_sha256: Mapping[str, str],
                        stable_path_restore: Sequence[Sequence[str]],
                        verified_at: str,
                        anchor_live: Optional[bool] = None) -> RollbackPlan:
    """Assemble the rollback plan from the archive, or refuse as incomplete.

    The refusal is structural and separate from the verdict: a plan that cannot name
    an archived binary per backend, or cannot say where the stable paths point
    afterwards, is INCOMPLETE and raises. A plan that is complete but does not verify
    is a plan with a FAILing `archive_check`, which blocks the package with a reason
    the operator can act on. Those are different problems and they must not collapse
    into one.
    """
    wanted = _str_tuple(backends, "build_rollback_plan: backends")
    check = verify_archive_target(
        archive, backends=wanted, incumbent_branch=incumbent_branch,
        incumbent_commit=incumbent_commit, expected_binary_sha256=expected_binary_sha256)
    entry = archive.entry(t3.ARCHIVE_GENERATION_N1)
    if entry is None:
        raise RollbackIncomplete(
            f"no {t3.ARCHIVE_GENERATION_N1} archive entry: there is nothing to roll back "
            "to (§10.5, and /mnt/raid0/llm/kernels/archive/ was empty as of 2026-08-02)")

    by_digest = {digest: path for path, digest in entry.binaries}
    binaries: list = []
    gaps: list = []
    for backend in wanted:
        expected = expected_binary_sha256.get(backend)
        path = by_digest.get(expected) if expected else None
        if path is None:
            gaps.append(backend)
            continue
        binaries.append((backend, path, expected))
    if gaps:
        raise RollbackIncomplete(
            f"no archived binary resolves for {sorted(gaps)}. A rollback that restores "
            "the branch but not the binaries leaves production on a kernel nobody "
            "archived (§10.5).")

    # Copied through verbatim. `t3.ArchivedBuild` is the source of the attribution
    # and this is a transport, not a producer: anything computed here would be a
    # fact about the packager rather than about the archive.
    libraries = tuple((backends, path, digest)
                      for backends, path, digest in entry.libraries)
    return RollbackPlan(
        rollback_branch=entry.branch, rollback_head=entry.commit,
        incumbent_archive_path=entry.archive_root,
        incumbent_binaries=tuple(binaries), incumbent_libraries=libraries,
        stable_path_restore=tuple(tuple(p) for p in stable_path_restore),
        archive_check=check, verified_at=verified_at, anchor_live=anchor_live)


# =============================================================================
# The transaction plan (§10.2 phase 8) — a DRY RUN, drafted for the operator
# =============================================================================

def build_transaction_plan(*, version: NextVersion, install_path: str,
                           stable_path_moves: Sequence[Sequence[str]],
                           service_impact: Sequence[str],
                           era_actions: Sequence[Mapping],
                           receipt_paths: Sequence[str],
                           rollback: RollbackPlan) -> t3.TransactionPlan:
    """Compute the exact transaction, as the dry run `t3.TransactionPlan` already is.

    This does not re-implement `t3.TransactionPlan`; it computes its fields and lets
    that class enforce its own rules (`executed=True` is refused there, and every
    era action must carry `draft=True`). What is added here is the packager's own
    cross-check: the transaction's rollback anchor must be the rollback PLAN's
    anchor, and the new version must not be the thing we would roll back to.

    Note what is NOT refused: `next_target` paths inside a production tree. That is
    where a new production build legitimately lives, and refusing it would be the
    "gate that cannot express its own output" defect — the operator's normal action
    would become unstateable. The defence against a production write is that this
    module performs none, not that it refuses to describe one.
    """
    if not isinstance(version, NextVersion):
        raise PackagerInputError("build_transaction_plan: version must be a NextVersion")
    if not isinstance(rollback, RollbackPlan):
        raise PackagerInputError("build_transaction_plan: rollback must be a RollbackPlan")
    if rollback.rollback_branch == version.next_branch:
        raise PackagerInputError(
            f"the rollback anchor and the new version are both {version.next_branch!r}; "
            "a transaction whose fallback is its own target has no fallback")
    moves: list = []
    for i, entry in enumerate(stable_path_moves or ()):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise PackagerInputError(
                f"build_transaction_plan: stable_path_moves[{i}] must be "
                "(link_path, current_target, next_target)")
        moves.append(tuple(_text(v, f"stable_path_moves[{i}]") for v in entry))
    if not moves:
        raise PackagerInputError(
            "build_transaction_plan: no stable-path move was computed. A kernel freeze "
            "that repoints nothing installs nothing; §1.5's four stable paths are how "
            "every launcher resolves a binary.")
    restore = {link for link, _target in rollback.stable_path_restore}
    uncovered = sorted({link for link, _c, _n in moves} - restore)
    if uncovered:
        raise RollbackIncomplete(
            f"the transaction moves {uncovered} but the rollback plan does not restore "
            "them. Every path the cutover repoints is a path the rollback must put back.")
    return t3.TransactionPlan(
        next_branch=version.next_branch,
        next_version_number=version.next_version_number,
        next_tag=version.next_tag,
        install_path=_text(install_path, "build_transaction_plan: install_path"),
        symlink_diff=tuple(moves),
        service_impact=tuple(service_impact or ()),
        era_actions=tuple(era_actions or ()),
        receipt_paths=tuple(receipt_paths or ()),
        rollback_branch=rollback.rollback_branch,
        rollback_head=rollback.rollback_head,
    )


# =============================================================================
# Drafts for operator execution (§1.3 items 2 and 3, §11.4)
# =============================================================================

#: The v8 cutover wrote THREE era rows — `E8-cpu-kernel`, `E8-autopilot-speed`,
#: `E8` (`instrument_eras.yaml:140-172`). The kinds are named here so a draft that
#: covers one of them is INCOMPLETE by construction rather than by review: a freeze
#: whose era row is unwritten produces evidence nobody can interpret (§1.3 item 2),
#: and `MEASUREMENT.md:233` requires every number to be era-labelled.
ERA_ROW_KIND_KERNEL = "kernel"
ERA_ROW_KIND_AUTOPILOT_SPEED = "autopilot_speed"
ERA_ROW_KIND_UMBRELLA = "umbrella"
ERA_ROW_KINDS = (ERA_ROW_KIND_KERNEL, ERA_ROW_KIND_AUTOPILOT_SPEED,
                 ERA_ROW_KIND_UMBRELLA)

#: The era row's effective moment is the operator's cutover, which has not
#: happened. A literal timestamp in a draft would be this module predicting when a
#: human will act — and every number stamped with it would carry that prediction.
ERA_EFFECTIVE_FROM = "at_operator_cutover"


@dataclass(frozen=True)
class EraRowDraft:
    """One DRAFT era-registry row. A draft, in a package, for a human to write."""

    era_id: str
    kind: str
    subject: str
    backends: tuple
    supersedes: Optional[str]
    note: str
    effective_from: str = ERA_EFFECTIVE_FROM

    def __post_init__(self) -> None:
        _text(self.era_id, "EraRowDraft.era_id")
        if self.kind not in ERA_ROW_KINDS:
            raise PackagerInputError(
                f"EraRowDraft.kind: {self.kind!r} is not one of {list(ERA_ROW_KINDS)}")
        _text(self.subject, "EraRowDraft.subject")
        object.__setattr__(self, "backends", _str_tuple(
            self.backends, "EraRowDraft.backends", non_empty=False))
        for backend in self.backends:
            if backend not in schemas.BACKENDS:
                raise PackagerInputError(
                    f"EraRowDraft.backends: {backend!r} is not a known backend")
        _opt_text(self.supersedes, "EraRowDraft.supersedes")
        _text(self.note, "EraRowDraft.note")
        if self.effective_from != ERA_EFFECTIVE_FROM:
            raise PackagerInputError(
                f"EraRowDraft.effective_from must be {ERA_EFFECTIVE_FROM!r}, not "
                f"{self.effective_from!r}. The era boundary is the operator's cutover; "
                "a literal timestamp here is this module predicting when a human will "
                "act, and every number labelled with that era would inherit the guess.")

    def to_dict(self) -> dict:
        return {"era_id": self.era_id, "kind": self.kind, "subject": self.subject,
                "backends": list(self.backends), "supersedes": self.supersedes,
                "note": self.note, "effective_from": self.effective_from,
                "draft": True, "written_by": EXECUTED_BY}


def draft_era_registry_row(*, rows: Sequence[EraRowDraft], version: NextVersion,
                           registry_path: str, incumbent_era: str,
                           drafted_at: str) -> dict:
    """Assemble the DRAFT era-registry block. It is a record, never a write.

    Structural refusals only — an era id that does not carry the successor
    version's prefix is a wiring error (the version number IS the era number), and
    a row set that is not a set of `EraRowDraft` is not a draft. Whether the set
    COVERS all three §1.3 kinds is a finding on the package, not a raise, because a
    package blocked with "the autopilot-speed era row is missing" is more use to an
    operator than an exception that produced no package at all.
    """
    drafts = _typed_tuple(rows, "draft_era_registry_row: rows", EraRowDraft,
                          non_empty=True)
    if not isinstance(version, NextVersion):
        raise PackagerInputError("draft_era_registry_row: version must be a NextVersion")
    _text(registry_path, "draft_era_registry_row: registry_path")
    _text(incumbent_era, "draft_era_registry_row: incumbent_era")
    _timestamp(drafted_at, "draft_era_registry_row: drafted_at")

    wrong = sorted(d.era_id for d in drafts if not d.era_id.startswith(version.era_prefix))
    if wrong:
        raise PackagerInputError(
            f"era ids {wrong} do not carry the successor prefix {version.era_prefix!r}. "
            "The production version number is the era number (the v8 cutover wrote "
            "E8-cpu-kernel, E8-autopilot-speed and E8); a mismatched prefix misorders "
            "every era-labelled number that follows.")
    seen = [d.era_id for d in drafts]
    if len(set(seen)) != len(seen):
        raise PackagerInputError(f"duplicate era ids in the draft: {sorted(seen)}")

    return {
        "draft": True,
        "written_by": EXECUTED_BY,
        "registry_path": registry_path,
        "human_only_path": True,
        "human_only_reference": "human_only_paths.yaml:35-37; MEASUREMENT.md:140-142",
        "predecessor_era": incumbent_era,
        "era_prefix": version.era_prefix,
        "kinds_present": sorted({d.kind for d in drafts}),
        "kinds_required": list(ERA_ROW_KINDS),
        "rows": [d.to_dict() for d in drafts],
        "drafted_at": drafted_at,
        "drafted_by": MODULE_ID,
        "notice": ("DRAFT. AutoKernel does not write the era registry; the operator "
                   "does. A freeze whose era row is unwritten produces evidence nobody "
                   "can interpret (§1.3 item 2, MEASUREMENT.md:233)."),
    }


def draft_autopilot_rebaseline_note(*, era_id: str, baseline_path: str,
                                    affected_roles: Sequence[str], hold_reason: str,
                                    drafted_at: str) -> str:
    """The DRAFT AutoPilot rebaseline note, following the E8 precedent exactly.

    E8 is explicit (`instrument_eras.yaml:166-172`): the cutover *"opens a
    fail-closed E8 AutoPilot rebaseline hold … until an **operator-ratified** E8
    quality-baseline reseed writes fresh values and windows"*. Fail-closed is the
    load-bearing word: the hold is the default state, not something someone
    remembers to set, because a kernel change moves the speed era even when model
    quality is identical (§11.4).
    """
    _text(era_id, "draft_autopilot_rebaseline_note: era_id")
    _text(baseline_path, "draft_autopilot_rebaseline_note: baseline_path")
    roles = _str_tuple(affected_roles,
                       "draft_autopilot_rebaseline_note: affected_roles")
    _text(hold_reason, "draft_autopilot_rebaseline_note: hold_reason")
    _timestamp(drafted_at, "draft_autopilot_rebaseline_note: drafted_at")
    return "\n".join([
        f"DRAFT AutoPilot rebaseline note for era {era_id} — drafted {drafted_at} by "
        f"{MODULE_ID}.",
        "",
        f"AutoKernel does not apply this. `{baseline_path}` is a human-only write "
        "(MEASUREMENT.md:140-142, human_only_paths.yaml:38-40) and the reseed is "
        "operator-ratified.",
        "",
        f"At cutover, open a FAIL-CLOSED {era_id} rebaseline hold covering "
        f"{', '.join(roles)}. Following the E8 precedent "
        f"(instrument_eras.yaml:166-172), the hold stays closed until an "
        f"operator-ratified {era_id} quality-baseline reseed writes fresh values and "
        "windows.",
        "",
        f"Why the hold is needed: {hold_reason}",
        "",
        "A new kernel changes orchestrator speed priors and AutoPilot's speed era even "
        "when model quality is identical (§11.4). Throughput priors measured under the "
        "predecessor era are not comparable across the boundary "
        "(MEASUREMENT.md:83-84, :233), so they are re-derived rather than carried.",
    ])


# =============================================================================
# The pre-validated operator command sequence (MEASUREMENT.md:138-145)
# =============================================================================

#: A command matching one of these is a human-only write. The list is a
#: CLASSIFIER, not a refusal: the operator's normal freeze commands necessarily
#: name every one of these targets, and a packager that refused to *describe* them
#: could not state its own output — the defect class already found once in the
#: `serving_runtime` adapter, in the opposite direction. What the classification
#: buys is that a command touching a trust boundary must carry a rollback and must
#: be marked as the operator's, and that neither can be declared away, because both
#: are derived from the command text rather than from a flag beside it.
HUMAN_ONLY_TARGET_PATTERNS = (
    (re.compile(r"production-(consolidated|speech)-v\d+"),
     "names a frozen production kernel branch "
     "(human_only_paths.yaml:42-49, §1.3 item 4)"),
    (re.compile(r"kernels/production"),
     "repoints or writes a stable production kernel path (invariant 3)"),
    (re.compile(r"instrument_eras\.ya?ml"),
     "writes an era-registry row (MEASUREMENT.md:140-142, human_only_paths.yaml:35-37)"),
    (re.compile(r"autopilot_baseline\.ya?ml"),
     "applies an AutoPilot baseline (MEASUREMENT.md:140-142, human_only_paths.yaml:38-40)"),
    (re.compile(r"human_only_paths"),
     "amends the pinned human-only path list (§1.3 item 4)"),
    (re.compile(r"\breboot\b"),
     "reboots the host (MEASUREMENT.md:140-142, operator authority)"),
)


def _human_only_reasons(text: str) -> tuple:
    return tuple(reason for pattern, reason in HUMAN_ONLY_TARGET_PATTERNS
                 if pattern.search(text))


@dataclass(frozen=True)
class OperatorCommand:
    """One step the OPERATOR runs. `executed_by` has exactly one legal value.

    `human_only` is a derived PROPERTY, not a field. Invariant 18's shape — declared
    equals traced — applied to the one place it would otherwise be cheapest to lie:
    a command that repoints `/mnt/raid0/llm/kernels/production/cpu` while declaring
    `human_only=False` would be a trust-boundary write wearing an ordinary label.
    """

    step: int
    command: str
    purpose: str
    expected_effect: str
    target_paths: tuple
    validation_receipt: str
    validation_method: str
    validated: bool
    rollback_command: Optional[str] = None
    executed_by: str = EXECUTED_BY

    def __post_init__(self) -> None:
        _positive_int(self.step, "OperatorCommand.step")
        _text(self.command, "OperatorCommand.command")
        _text(self.purpose, "OperatorCommand.purpose")
        _text(self.expected_effect, "OperatorCommand.expected_effect")
        object.__setattr__(self, "target_paths", _str_tuple(
            self.target_paths, "OperatorCommand.target_paths", non_empty=False))
        _text(self.validation_method, "OperatorCommand.validation_method")
        _bool(self.validated, "OperatorCommand.validated")
        if self.validated:
            _text(self.validation_receipt, "OperatorCommand.validation_receipt")
        _opt_text(self.rollback_command, "OperatorCommand.rollback_command")
        if self.executed_by != EXECUTED_BY:
            raise ProductionWriteRefused(
                f"OperatorCommand.executed_by: {self.executed_by!r} is not "
                f"{EXECUTED_BY!r}. §11.2 — the packager may not execute any command it "
                "drafted, so a command in this package that anything else executes is a "
                "contradiction in the record itself.")

    @property
    def scanned_text(self) -> str:
        return " ".join((self.command,) + tuple(self.target_paths))

    @property
    def human_only_reasons(self) -> tuple:
        return _human_only_reasons(self.scanned_text)

    @property
    def human_only(self) -> bool:
        return bool(self.human_only_reasons)

    def to_dict(self) -> dict:
        return {"step": self.step, "command": self.command, "purpose": self.purpose,
                "expected_effect": self.expected_effect,
                "target_paths": list(self.target_paths),
                "validation_receipt": self.validation_receipt,
                "validation_method": self.validation_method,
                "validated": self.validated,
                "rollback_command": self.rollback_command,
                "executed_by": self.executed_by,
                "human_only": self.human_only,
                "human_only_reasons": list(self.human_only_reasons)}


@dataclass(frozen=True)
class CommandSequenceReview:
    """What static pre-validation concluded, and what it could not conclude."""

    check: schemas.Check
    validated_commands: tuple
    unvalidated_commands: tuple
    findings: tuple
    covered_elements: tuple
    uncovered_elements: tuple

    def to_dict(self) -> dict:
        return {"check": _check_dict(self.check),
                "validated_step_count": len(self.validated_commands),
                "unvalidated_steps": [c.step for c in self.unvalidated_commands],
                "findings": list(self.findings),
                "covered_elements": list(self.covered_elements),
                "uncovered_elements": list(self.uncovered_elements)}


def era_row_registry_path(era_row: Mapping) -> str:
    """The era registry this row writes, or `""` when the row does not name one.

    Split out and made public because ONE reader of this key decides whether the
    era registry is in the coverage denominator at all, and the caller has to be
    able to ask the same question the enumerator asked. `draft_era_registry_row()`
    always sets it; `validate_command_sequence()` and `assemble_release_package()`
    accept a hand-built mapping, and there the key is simply absent-able.
    """
    value = era_row.get("registry_path")
    return value if isinstance(value, str) and value else ""


def _transaction_elements(transaction: t3.TransactionPlan, rollback: RollbackPlan,
                          era_row: Mapping) -> tuple:
    """Every thing the transaction says will happen, as a coverage denominator.

    Derived from the transaction itself rather than listed by hand, so a transaction
    that grows an element grows the requirement. `MEASUREMENT.md:138-145` asks for
    every operator command to be pre-validated end to end; the other half of "end to
    end" is that every step of the transaction has a command, which is what an
    uncovered element names.
    """
    elements: list = [
        ("branch", transaction.next_branch),
        ("tag", transaction.next_tag),
        ("install_path", transaction.install_path),
        ("archive", rollback.incumbent_archive_path),
    ]
    elements.extend(("stable_path", link) for link, _c, _n in transaction.symlink_diff)
    # A missing `registry_path` SHRINKS this denominator rather than failing it, which
    # is why `validate_command_sequence` refuses the era row that omits it before it
    # ever gets here. Keep the guard there, not a raise here: this function's whole
    # job is to enumerate, and an enumerator that raises cannot report.
    registry_path = era_row_registry_path(era_row)
    if registry_path:
        elements.append(("era_registry", registry_path))
    elements.extend(("receipt_path", path) for path in transaction.receipt_paths)
    return tuple(elements)


#: What a command must be seen DOING to an element of each kind. Coverage used to
#: ask only that some validated command's text CONTAIN the element's value, pooled
#: across the whole sequence — which a COMMENT satisfies, and which `$EDITOR
#: something_else  # remember to update instrument_eras.yaml` satisfied for the
#: era-registry element. Naming a thing is not acting on it.
#:
#: Both classes of verb are legitimate, because the coverage finding says
#: "performs OR VERIFIES it": `ln` repoints a stable path and `readlink` proves
#: where it points, and a sequence that verifies the archive without touching it is
#: the correct sequence, not an uncovered one.
#:
#: `$EDITOR` is a verb here, and deliberately. A human-only registry write is
#: performed by opening the file in an editor — that is the sanctioned idiom for
#: `instrument_eras.yaml` and `autopilot_baseline.yaml` (`MEASUREMENT.md:140-142`),
#: and a vocabulary that refused it would forbid the only compliant way to do the
#: thing it is checking for.
ELEMENT_VERBS = {
    "branch": ("git", "branch", "checkout", "switch", "tag", "rev-parse", "show-ref",
               "merge-base", "log", "for-each-ref"),
    "tag": ("git", "tag", "describe", "rev-parse", "show-ref", "for-each-ref"),
    "install_path": ("install", "cp", "rsync", "mkdir", "ln", "readlink", "ls", "stat",
                     "test", "find", "sha256sum", "du"),
    "archive": ("cp", "rsync", "tar", "mkdir", "sha256sum", "ls", "stat", "test",
                "find", "du"),
    "stable_path": ("ln", "readlink", "mv", "ls", "stat", "test", "find"),
    "era_registry": ("$EDITOR", "${EDITOR}", "EDITOR", "vim", "vi", "nano", "yq",
                     "python3", "diff", "grep", "sha256sum", "cat", "test", "ls"),
    "autopilot_baseline": ("$EDITOR", "${EDITOR}", "EDITOR", "vim", "vi", "nano", "yq",
                           "python3", "diff", "grep", "sha256sum", "cat", "test", "ls"),
    "receipt_path": ("cp", "rsync", "mkdir", "tee", "sha256sum", "ls", "stat", "test",
                     "find", "cat", "python3", "git"),
}

#: A `#` that starts a token runs to end of line. Erring toward OVER-stripping is
#: the fail-CLOSED direction: text wrongly treated as a comment can only make an
#: element look uncovered, never covered.
_SHELL_COMMENT_RE = re.compile(r"(?m)(?:^|\s)#.*$")
_WORD_RE = re.compile(r"[A-Za-z0-9_$.{}-]+")

#: A quoted span is DATA, not a verb. `#` is not the only way to write prose on a
#: command line: `echo "reminder: $EDITOR instrument_eras.yaml"` names the era
#: registry and puts `$EDITOR` in the token set while running `echo`, which is the
#: comment hole with a different quoting character. Unterminated quotes are left
#: alone deliberately — the pattern requires a closing quote, so a stray `"` cannot
#: swallow the rest of a real command and turn a genuine verb into prose.
_QUOTED_SPAN_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


def _executable_text(command: str) -> str:
    """The part of a command line that actually runs."""
    return _SHELL_COMMENT_RE.sub(" ", command)


def _verb_text(command: str) -> str:
    """The part of a command line a VERB may be read out of: unquoted, uncommented.

    Deliberately narrower than `_executable_text()`, and used for exactly one of the
    two halves of `_acts_on`. A quoted span may still NAME the element — `python3 -c
    "…open('orchestration/instrument_eras.yaml')…"` is a real way to edit the
    registry and its verb (`python3`) is in command position outside the quotes — so
    only the verb lookup is narrowed. Narrowing both would forbid that idiom, which
    is the failure mode this vocabulary has already had once.
    """
    return _QUOTED_SPAN_RE.sub(" ", _executable_text(command))


def _acts_on(command: "OperatorCommand", kind: str, value: str) -> bool:
    """Does THIS ONE command both name the element and carry a verb for its kind?

    Three tightenings over the old `value in "\\n".join(every command)`:

      1. **one command**, not the pooled text of all of them — element named in step
         2 and verb present in step 9 is not a step that does the thing;
      2. **comments removed** from the command before it is scanned;
      3. **a verb from the element's own kind**, so a command that mentions the era
         registry while doing something else does not cover it; and
      4. **the verb read only out of the UNQUOTED text**, because stripping `#` and
         nothing else left the same hole under a different quoting character:
         `echo "reminder: $EDITOR orchestration/instrument_eras.yaml"` named the
         element and put `$EDITOR` in the token set while running `echo`, and the
         whole sequence reviewed as covered.

    The element's value may be named in the executable text — quoted spans included,
    because a path inside `python3 -c "…"` really is the path that command edits — or
    in `target_paths`. `target_paths` is where a command declares what it touches, and
    a declared target with no verb is the same silence the old check could not see.
    """
    verbs = ELEMENT_VERBS.get(kind)
    if not verbs:
        # An element kind with no declared vocabulary is NOT auto-covered. A kind
        # added to `_transaction_elements` without a vocabulary here would otherwise
        # be a free pass, which is the failure mode this table exists to remove.
        return False
    executable = _executable_text(command.command)
    named = value in executable or any(value in path for path in command.target_paths)
    if not named:
        return False
    words = set(_WORD_RE.findall(_verb_text(command.command)))
    return any(verb in words for verb in verbs)


def _within_surface(path: str, surface: str) -> bool:
    """Containment at a path-component boundary, in ONE direction.

    The tempting second direction — `surface.startswith(path)`, so a command may
    target a PARENT of a transaction element — makes the check defeatable by
    breadth: a command targeting `/` is then a prefix of every surface and lands
    inside all of them. Every legitimate parent a command needs (`install_path`,
    the archive root, the receipt directory) is already a declared element in its
    own right, so the direction is not needed and the hole is not worth it.

    Component boundaries matter for the same reason they do in `plan.py`:
    `/mnt/raid0/llm/llama.cpp` is a string prefix of
    `/mnt/raid0/llm/llama.cpp-experimental`, which is a different tree.
    """
    return path == surface or path.startswith(surface.rstrip("/") + "/")


def validate_command_sequence(commands: Sequence[OperatorCommand], *,
                              transaction: t3.TransactionPlan, rollback: RollbackPlan,
                              era_row: Mapping,
                              autopilot_baseline_path: str) -> CommandSequenceReview:
    """Pre-validate the sequence end to end — STATICALLY, because running it is the
    transaction.

    Six properties, each of which has a way of being wrong that looks fine:

      1. **contiguous ordering from 1** — a sequence with a gap is a sequence
         somebody edited, and the operator would run the surviving steps in order
         without noticing the missing one;
      2. **every step carries a validation receipt and a method** — an unvalidated
         command is filed separately and blocks the package rather than riding in
         the array as though it had been checked (`schemas` refuses `validated:
         false` in the sequence for exactly this reason);
      3. **every human-only step carries a rollback command** — a trust-boundary
         write with no stated way back is how a cutover becomes irreversible in
         practice while looking reversible on paper;
      4. **coverage** — every element of the transaction is ACTED ON by some
         command. A transaction step with no command is a step the operator has to
         invent at 3am. "Acted on" is `ELEMENT_VERBS`, not a substring: this check
         used to pool every command's text and ask whether the element's value
         appeared anywhere in it, which a comment in an unrelated step satisfied;
      5. **containment** — every command's target paths lie inside the transaction's
         declared surface. A command reaching outside it is scope this package did
         not derive (invariant 18, §12); and
      6. **the AutoPilot baseline is addressed** — §11.4/E8 make the rebaseline part
         of the transaction's consequences, so it is an element of the coverage
         denominator and is held to property 4 exactly like the rest.
    """
    cmds = _typed_tuple(commands, "validate_command_sequence: commands", OperatorCommand,
                        non_empty=True)
    if not isinstance(transaction, t3.TransactionPlan):
        raise PackagerInputError(
            "validate_command_sequence: transaction must be a t3.TransactionPlan")
    if not isinstance(rollback, RollbackPlan):
        raise PackagerInputError(
            "validate_command_sequence: rollback must be a RollbackPlan")
    _mapping(era_row, "validate_command_sequence: era_row", non_empty=True)
    _text(autopilot_baseline_path, "validate_command_sequence: autopilot_baseline_path")

    findings: list = []
    steps = [c.step for c in cmds]
    if sorted(steps) != list(range(1, len(cmds) + 1)):
        findings.append(
            f"COMMAND_SEQUENCE_NOT_CONTIGUOUS: steps {sorted(steps)} are not 1..{len(cmds)}; "
            "a gap is an edited sequence, and the operator runs what is in front of them")

    validated = tuple(c for c in cmds if c.validated)
    unvalidated = tuple(c for c in cmds if not c.validated)
    for command in unvalidated:
        findings.append(
            f"COMMAND_NOT_PRE_VALIDATED: step {command.step} ({command.command!r}) carries "
            "no completed pre-validation. MEASUREMENT.md:138-145 requires every operator "
            "command to be pre-validated end to end before it is handed over.")
    for command in validated:
        if command.human_only and not command.rollback_command:
            findings.append(
                f"HUMAN_ONLY_COMMAND_WITHOUT_ROLLBACK: step {command.step} "
                f"({'; '.join(command.human_only_reasons)}) states no way back")

    # The era-registry element is the ONE element of the denominator that is
    # conditional on the input, and a conditional conjunct is a conjunct that can be
    # satisfied by deleting it. `_transaction_elements` appends `era_registry` only
    # when the era row names a `registry_path`, so an era row with the key absent or
    # blank removed the era registry from the coverage requirement entirely and the
    # sequence passed with no step writing the row at all — the human-only write at
    # `MEASUREMENT.md:140-142` silently dropped out of the package. The absence is
    # therefore the finding, and the strengthened verb vocabulary is not reachable
    # around it. §11.2/E8 makes the era row part of every freeze transaction; a
    # freeze that genuinely writes no era row is not a case this drops quietly, it is
    # a case whose row must say which registry it is a row IN.
    if not era_row_registry_path(era_row):
        findings.append(
            "ERA_ROW_NAMES_NO_REGISTRY_PATH: the drafted era row does not name a "
            "`registry_path`, so there is no era-registry element for any command to "
            "be held against and the coverage check silently stops asking about the "
            "human-only registry write (MEASUREMENT.md:140-142). A conjunct that "
            "disappears when its subject is omitted is not a conjunct.")
    elements = _transaction_elements(transaction, rollback, era_row)
    # §11.4/E8 makes the rebaseline a consequence of the cutover, so it is an
    # element of the transaction and is held to the same "acts on it" bar as every
    # other one. It was previously checked by substring alone, which is the hole
    # this whole vocabulary closes — leaving one element on the old rule would have
    # left the hole under a different name.
    covered_elements = elements + (("autopilot_baseline", autopilot_baseline_path),)
    covered: list = []
    uncovered: list = []
    for kind, value in covered_elements:
        acting = [c.step for c in validated if value and _acts_on(c, kind, value)]
        if acting:
            covered.append(f"{kind}:{value}")
        else:
            uncovered.append(f"{kind}:{value}")
            naming = [c.step for c in validated
                      if value and (value in c.scanned_text)]
            named_but_inert = (
                f" Step(s) {naming} NAME it without any {kind} verb "
                f"{sorted(ELEMENT_VERBS.get(kind, ()))} in the part of the line that "
                "runs; naming a thing is not acting on it." if naming else "")
            findings.append(
                f"TRANSACTION_STEP_UNCOMMANDED: the transaction names {kind} {value!r} and "
                "no pre-validated command performs or verifies it." + named_but_inert)

    declared_surface = tuple(value for _kind, value in elements) + (
        autopilot_baseline_path, transaction.install_path)
    for command in cmds:
        outside = sorted(
            path for path in command.target_paths
            if not any(_within_surface(path, surface)
                       for surface in declared_surface if surface))
        if outside:
            findings.append(
                f"COMMAND_OUTSIDE_TRANSACTION_SCOPE: step {command.step} targets {outside}, "
                "which the derived transaction does not contain. Scope is mechanically "
                "derived (invariant 18); a command reaching outside it widens the "
                "transaction after the plan was compiled.")

    check = _fail(*findings) if findings else schemas.Check(schemas.PASS)
    return CommandSequenceReview(
        check=check, validated_commands=validated, unvalidated_commands=unvalidated,
        findings=tuple(findings), covered_elements=tuple(covered),
        uncovered_elements=tuple(uncovered))


# =============================================================================
# The cutover REQUEST (§11.3) — routed on the bus, never an action
# =============================================================================

#: What the message asks for, as a closed token rather than prose. The bus's own
#: 2026-07-29 lesson is that routing intent living in payload prose gets truncated
#: away; the same applies to the *verb*. This one says "schedule it yourself", and
#: there is no vocabulary member that says "restart now".
CUTOVER_ASK = "schedule_at_your_own_boundary"


@dataclass(frozen=True)
class CutoverRequest:
    """A bus message record asking the inference owner to schedule the cutover.

    `OPERATING_CONSTRAINTS.md:41`: a reload *"must be executed BY THAT SESSION, at a
    moment it chooses; it is never forced upon that session's workflow from
    outside"*. So this record names **no time**. There is no `scheduled_at` field
    for one to occupy, and `CUTOVER_ASK` is the only ask it can carry.

    `needs_routing_to` and `action_required` are top-level message fields rather
    than payload prose, because prose routing intent has already been truncated
    away once on this bus (2026-07-29, two missed messages). `needs_routing_to`
    must be non-empty for the same reason `session_bus.py append` refuses
    `action_required` on a broadcast: *"intent with no addressee is the failure
    shape itself"*.

    This module has no transport. `send_cutover_request()` raises; the session that
    owns the roster id appends this record to ITS OWN outbox (BUS_PROTOCOL rule 1).
    """

    message_id: str
    from_agent: str
    to_agent: str
    needs_routing_to: tuple
    task_id: str
    created_at: str
    package_id: str
    next_branch: str
    service_impact: tuple
    rollback_summary: str
    ask: str = CUTOVER_ASK
    action_required: bool = True

    def __post_init__(self) -> None:
        _text(self.message_id, "CutoverRequest.message_id")
        _text(self.from_agent, "CutoverRequest.from_agent")
        _text(self.to_agent, "CutoverRequest.to_agent")
        object.__setattr__(self, "needs_routing_to", _str_tuple(
            self.needs_routing_to, "CutoverRequest.needs_routing_to"))
        if self.to_agent == "*" and not self.needs_routing_to:
            raise PackagerInputError(
                "CutoverRequest: a broadcast with no `needs_routing_to` is intent with no "
                "addressee (BUS_PROTOCOL, routing intent is structural)")
        _text(self.task_id, "CutoverRequest.task_id")
        _timestamp(self.created_at, "CutoverRequest.created_at")
        _text(self.package_id, "CutoverRequest.package_id")
        _text(self.next_branch, "CutoverRequest.next_branch")
        object.__setattr__(self, "service_impact", _str_tuple(
            self.service_impact, "CutoverRequest.service_impact", non_empty=False))
        _text(self.rollback_summary, "CutoverRequest.rollback_summary")
        if self.ask != CUTOVER_ASK:
            raise CutoverExecutionRefused(
                f"CutoverRequest.ask: {self.ask!r} is not {CUTOVER_ASK!r}. The package "
                "asks the inference owner to schedule the cutover at a boundary that "
                "session chooses; anything more imperative is the preemption "
                "OPERATING_CONSTRAINTS.md:41 exists to prevent.")
        _bool(self.action_required, "CutoverRequest.action_required")

    def to_bus_message(self) -> dict:
        """The `session_bus.msg.v1` envelope, ready for the owning session to append."""
        return {
            "schema_version": CUTOVER_MESSAGE_SCHEMA,
            "id": self.message_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "kind": "request",
            "needs_routing_to": list(self.needs_routing_to),
            "action_required": self.action_required,
            "task_id": self.task_id,
            "ts": self.created_at,
            "payload": {
                "ask": self.ask,
                "package_id": self.package_id,
                "next_branch": self.next_branch,
                "service_impact": list(self.service_impact),
                "rollback_summary": self.rollback_summary,
                "scheduling_rule": (
                    "OPERATING_CONSTRAINTS.md:41 — the reload is executed BY the session "
                    "that owns the inference, at a moment it chooses. This request names "
                    "no time and blocks nothing (BUS_PROTOCOL rule 2)."),
                "delivered_by": ("the owning session appends this record to its own "
                                 "outbox; AutoKernel writes no bus file"),
            },
        }

    def to_dict(self) -> dict:
        return {"message": self.to_bus_message(), "sent": False,
                "notice": ("A REQUEST, not an action. AutoKernel neither schedules nor "
                           "performs a cutover (§11.3).")}


def build_cutover_request(*, message_id: str, from_agent: str, to_agent: str,
                          needs_routing_to: Sequence[str], task_id: str,
                          created_at: str, package_id: str,
                          transaction: t3.TransactionPlan,
                          rollback: RollbackPlan) -> CutoverRequest:
    """Derive the cutover request from the transaction it is about."""
    if not isinstance(transaction, t3.TransactionPlan):
        raise PackagerInputError(
            "build_cutover_request: transaction must be a t3.TransactionPlan")
    if not isinstance(rollback, RollbackPlan):
        raise PackagerInputError("build_cutover_request: rollback must be a RollbackPlan")
    summary = (f"rollback to {rollback.rollback_branch}@{rollback.rollback_head[:12]} from "
               f"the archived incumbent at {rollback.incumbent_archive_path} "
               f"(archive check: {rollback.archive_check.outcome})")
    return CutoverRequest(
        message_id=message_id, from_agent=from_agent, to_agent=to_agent,
        needs_routing_to=tuple(needs_routing_to), task_id=task_id,
        created_at=created_at, package_id=package_id,
        next_branch=transaction.next_branch,
        service_impact=tuple(transaction.service_impact),
        rollback_summary=summary)


# =============================================================================
# §11.5 — the post-cutover watch window, declared BEFORE the data is seen
# =============================================================================

#: §11.5's default duration. Named as a default the package must still STATE — a
#: window whose duration nobody declared is a window nobody agreed to.
DEFAULT_WATCH_WINDOW_DAYS = 7

WATCH_WINDOW_OUTPUT_CLASS = (
    "RECOMMENDATION — NOT A CLAIM. Production telemetry is observational and "
    "uncontrolled; the window uses the standing noise reference and MDE rather than "
    "pretending to protocol grade (§11.5, MEASUREMENT.md:9-11)."
)

#: The six rows of §11.5's signal table, verbatim in content and order.
SIGNAL_THROUGHPUT = "decode_and_prefill_throughput_at_production_recipes"
SIGNAL_LATENCY = "per_request_latency_p50_p95"
SIGNAL_ERROR_RATES = "error_timeout_and_fallback_rates"
SIGNAL_MEMORY = "memory_growth_and_vram_ram_headroom"
SIGNAL_QUALITY = "quality_proxies_on_production_traffic"
SIGNAL_SUPERVISOR = "crash_restart_and_stale_process_events"

REQUIRED_WATCH_SIGNALS = (
    SIGNAL_THROUGHPUT, SIGNAL_LATENCY, SIGNAL_ERROR_RATES, SIGNAL_MEMORY,
    SIGNAL_QUALITY, SIGNAL_SUPERVISOR,
)

#: The alarm direction is a property of the SIGNAL, per §11.5's third column — so
#: it is looked up here rather than declared beside each band. A band that could
#: declare its own direction could declare the harmless one.
ALARM_REGRESSION = "regression_is_the_alarm"
ALARM_ANY_INCREASE = "any_increase_is_the_alarm"
ALARM_DRIFT_TO_FLOOR = "drift_toward_the_residency_floor_is_the_alarm"
ALARM_ANY_OCCURRENCE = "any_occurrence_is_the_alarm"

WATCH_SIGNAL_ALARM_RULES = {
    SIGNAL_THROUGHPUT: ALARM_REGRESSION,
    SIGNAL_LATENCY: ALARM_REGRESSION,
    SIGNAL_ERROR_RATES: ALARM_ANY_INCREASE,
    SIGNAL_MEMORY: ALARM_DRIFT_TO_FLOOR,
    SIGNAL_QUALITY: ALARM_REGRESSION,
    SIGNAL_SUPERVISOR: ALARM_ANY_OCCURRENCE,
}

#: §11.5's second column. Recorded so the window names WHERE each number comes
#: from; a signal whose source is unstated cannot be collected by anyone else.
WATCH_SIGNAL_SOURCES = {
    SIGNAL_THROUGHPUT: "orchestrator/serving telemetry",
    SIGNAL_LATENCY: "serving telemetry",
    SIGNAL_ERROR_RATES: "server logs and backend receipts",
    SIGNAL_MEMORY: "host and device sampling",
    SIGNAL_QUALITY: "evidence plane",
    SIGNAL_SUPERVISOR: "supervisor",
}

#: Deliberately closed, and deliberately containing no action on production. §11.5:
#: *"A signal outside its band raises a decision package; it does not itself revert
#: anything."* There is no `revert`, no `rollback` and no `restart` member, so the
#: window cannot recommend one however far outside a band a signal lands.
WATCH_CONTINUE = "continue_watching"
WATCH_RAISE_DECISION_PACKAGE = "raise_decision_package"
WATCH_INCOMPLETE_EVIDENCE = "incomplete_evidence"
WATCH_CLOSE_NO_REGRESSION = "close_with_no_regression_observed"
WATCH_RECOMMENDATIONS = (WATCH_CONTINUE, WATCH_RAISE_DECISION_PACKAGE,
                         WATCH_INCOMPLETE_EVIDENCE, WATCH_CLOSE_NO_REGRESSION)

#: The verdicts the explicit close step may record. `inconclusive` is a first-class
#: member: a window that observed too little says so rather than passing by default.
WATCH_CLOSE_VERDICTS = ("no_regression_observed", "regression_observed", "inconclusive")

WATCH_STATE_OPEN = "OPEN"
WATCH_STATE_CLOSEABLE = "CLOSEABLE"
WATCH_STATE_CLOSED = "CLOSED"


@dataclass(frozen=True)
class WatchSignalBand:
    """One signal's band, fixed at assembly from the INCUMBENT ERA's distribution.

    `basis_ref` and `noise_reference_ref` are required, not decorative:
    `MEASUREMENT.md:233` requires era-labelled comparison, and §11.5 says the
    pre-cutover baseline *"is the incumbent era's recorded distribution, not a
    remembered number"*. A band with no basis is a number somebody chose.

    A band must have at least one edge on the side its alarm rule watches. A
    "regression is the alarm" signal with no lower bound cannot alarm.
    """

    signal_id: str
    unit: str
    basis_ref: str
    noise_reference_ref: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    mde: Optional[float] = None
    roles: tuple = ()

    def __post_init__(self) -> None:
        if self.signal_id not in REQUIRED_WATCH_SIGNALS:
            raise PackagerInputError(
                f"WatchSignalBand.signal_id: {self.signal_id!r} is not one of §11.5's six "
                f"signals {list(REQUIRED_WATCH_SIGNALS)}")
        _text(self.unit, "WatchSignalBand.unit")
        _text(self.basis_ref, "WatchSignalBand.basis_ref")
        _text(self.noise_reference_ref, "WatchSignalBand.noise_reference_ref")
        for name in ("lower", "upper", "mde"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))):
                raise PackagerInputError(f"WatchSignalBand.{name}: must be a number or None")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise PackagerInputError(
                f"WatchSignalBand({self.signal_id}): lower {self.lower} exceeds upper "
                f"{self.upper}; nothing can be inside that band")
        object.__setattr__(self, "roles", _str_tuple(
            self.roles, "WatchSignalBand.roles", non_empty=False))
        rule = self.alarm_rule
        if rule in (ALARM_REGRESSION, ALARM_DRIFT_TO_FLOOR) and self.lower is None:
            raise PackagerInputError(
                f"WatchSignalBand({self.signal_id}): {rule} needs a lower edge, or the "
                "signal cannot alarm in the direction §11.5 says it alarms in")
        if rule in (ALARM_ANY_INCREASE, ALARM_ANY_OCCURRENCE) and self.upper is None:
            raise PackagerInputError(
                f"WatchSignalBand({self.signal_id}): {rule} needs an upper edge")

    @property
    def alarm_rule(self) -> str:
        return WATCH_SIGNAL_ALARM_RULES[self.signal_id]

    @property
    def source(self) -> str:
        return WATCH_SIGNAL_SOURCES[self.signal_id]

    def standing_for(self, value: Optional[float]) -> schemas.Check:
        """Inside the band, outside it, or — the third outcome — not evaluable."""
        if value is None:
            return _cnc(f"{self.signal_id}: no value was observed")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _cnc(f"{self.signal_id}: observed value {value!r} is not a number")
        reasons: list = []
        if self.lower is not None and value < self.lower:
            reasons.append(f"{self.signal_id}: {value} is below the band's lower edge "
                           f"{self.lower} {self.unit} ({self.alarm_rule})")
        if self.upper is not None and value > self.upper:
            reasons.append(f"{self.signal_id}: {value} is above the band's upper edge "
                           f"{self.upper} {self.unit} ({self.alarm_rule})")
        return _fail(*reasons) if reasons else schemas.Check(schemas.PASS)

    def to_dict(self) -> dict:
        return {"signal_id": self.signal_id, "source": self.source,
                "alarm_rule": self.alarm_rule, "unit": self.unit,
                "lower": self.lower, "upper": self.upper, "mde": self.mde,
                "basis_ref": self.basis_ref,
                "noise_reference_ref": self.noise_reference_ref,
                "roles": list(self.roles)}


def default_watch_bands(*, basis_ref_by_signal: Mapping[str, str],
                        noise_reference_ref: str,
                        edges: Mapping[str, Mapping[str, Any]]) -> tuple:
    """Build the six bands from declared per-signal edges. No edge is invented here.

    There is deliberately no fallback edge: a band this module chose would be a
    release threshold nobody declared, and §11.5 puts band-setting at package
    assembly time *from the incumbent era's observed distribution*, which is data
    this module does not hold.
    """
    _mapping(basis_ref_by_signal, "default_watch_bands: basis_ref_by_signal",
             non_empty=True)
    _text(noise_reference_ref, "default_watch_bands: noise_reference_ref")
    _mapping(edges, "default_watch_bands: edges", non_empty=True)
    bands: list = []
    for signal_id in REQUIRED_WATCH_SIGNALS:
        spec = edges.get(signal_id)
        if not isinstance(spec, Mapping):
            raise PackagerInputError(
                f"default_watch_bands: no edge spec for {signal_id!r}. §11.5's table has "
                "six rows and a window missing one is blind in that direction.")
        basis = basis_ref_by_signal.get(signal_id)
        if not isinstance(basis, str) or not basis.strip():
            raise PackagerInputError(
                f"default_watch_bands: {signal_id!r} has no incumbent-era basis ref; the "
                "pre-cutover baseline is the incumbent era's recorded distribution, not a "
                "remembered number (§11.5, MEASUREMENT.md:233)")
        bands.append(WatchSignalBand(
            signal_id=signal_id, unit=_text(spec.get("unit"), f"{signal_id}.unit"),
            basis_ref=basis, noise_reference_ref=noise_reference_ref,
            lower=spec.get("lower"), upper=spec.get("upper"), mde=spec.get("mde"),
            roles=tuple(spec.get("roles", ()))))
    return tuple(bands)


@dataclass(frozen=True)
class WatchWindowCloseStep:
    """§11.5: *"Closing the window is an explicit action that records the verdict —
    an unclosed window is an open question, not a pass."*"""

    owner: str
    action: str = "close_with_verdict"
    verdict_required: bool = True
    verdict_vocabulary: tuple = WATCH_CLOSE_VERDICTS
    unclosed_state: str = "OPEN_QUESTION"

    def __post_init__(self) -> None:
        _require_human_actor(_text(self.owner, "WatchWindowCloseStep.owner"),
                             "WatchWindowCloseStep.owner", error=PackagerInputError)
        _text(self.action, "WatchWindowCloseStep.action")
        if self.verdict_required is not True:
            raise PackagerInputError(
                "WatchWindowCloseStep.verdict_required must be True: a close that records "
                "no verdict is an expiry, and §11.5 requires the verdict")
        object.__setattr__(self, "verdict_vocabulary", _str_tuple(
            self.verdict_vocabulary, "WatchWindowCloseStep.verdict_vocabulary"))
        unknown = sorted(set(self.verdict_vocabulary) - set(WATCH_CLOSE_VERDICTS))
        if unknown:
            raise PackagerInputError(
                f"WatchWindowCloseStep.verdict_vocabulary: {unknown} are not among "
                f"{list(WATCH_CLOSE_VERDICTS)}")

    def to_dict(self) -> dict:
        return {"owner": self.owner, "action": self.action,
                "verdict_required": self.verdict_required,
                "verdict_vocabulary": list(self.verdict_vocabulary),
                "unclosed_state": self.unclosed_state}


@dataclass(frozen=True)
class WatchWindow:
    """The whole §11.5 artifact, assembled before the window opens.

    Every field here is a commitment made in advance. The three that matter most:

      * **`bands_fixed_at` must not be after `opens_at`.** §11.5: bands are *"named
        in the package so they are fixed before the window opens rather than chosen
        after seeing the data"*. A band set later is a band chosen against the data
        it is meant to judge, and `bands_sha256()` is what makes that checkable at
        evaluation time rather than a matter of trust.
      * **`min_volume_by_role` covers every affected role.** *"A window that expires
        on a quiet weekend has observed nothing."* Duration alone is not exposure.
      * **`owner` is human.** §11.5: the window *"is owned by whoever executed the
        cutover; AutoKernel computes and reports"*.
    """

    window_id: str
    package_id: str
    owner: str
    incumbent_era: str
    candidate_era: str
    affected_roles: tuple
    min_duration_days: int
    min_volume_by_role: Mapping[str, int]
    bands: tuple
    bands_fixed_at: str
    opens_at: str
    close_step: WatchWindowCloseStep
    rollback_anchor_ref: str
    activation_manifest_ref: str
    activation_manifest_sha256: str
    comparison_method: str = "era_labelled_vs_incumbent_era_recorded_distribution"

    def __post_init__(self) -> None:
        _text(self.window_id, "WatchWindow.window_id")
        _text(self.package_id, "WatchWindow.package_id")
        _require_human_actor(_text(self.owner, "WatchWindow.owner"), "WatchWindow.owner",
                             error=PackagerInputError)
        _text(self.incumbent_era, "WatchWindow.incumbent_era")
        _text(self.candidate_era, "WatchWindow.candidate_era")
        if self.incumbent_era == self.candidate_era:
            raise PackagerInputError(
                "WatchWindow: the incumbent and candidate eras are the same label, so the "
                "comparison is not era-labelled and MEASUREMENT.md:233 is unsatisfied")
        object.__setattr__(self, "affected_roles", _str_tuple(
            self.affected_roles, "WatchWindow.affected_roles"))
        _positive_int(self.min_duration_days, "WatchWindow.min_duration_days")
        if self.min_duration_days < DEFAULT_WATCH_WINDOW_DAYS:
            # §11.5's duration is `later_of(7 days, per-role volume)`. `later_of` has
            # no branch in which the answer is shorter than seven days, so a declared
            # duration below the floor is not a stricter reading of the rule — it is
            # the one number in the window that could be set to make the window close
            # sooner, and it is the number nothing else checks.
            raise PackagerInputError(
                f"WatchWindow.min_duration_days: {self.min_duration_days} is below "
                f"§11.5's {DEFAULT_WATCH_WINDOW_DAYS}-day floor. The rule is later_of("
                f"{DEFAULT_WATCH_WINDOW_DAYS} days, declared per-role volume); a shorter "
                "window is a window that closes before the last automatic safety net in "
                "the path has seen a weekday cycle.")
        _mapping(self.min_volume_by_role, "WatchWindow.min_volume_by_role", non_empty=True)
        missing = sorted(set(self.affected_roles) - set(self.min_volume_by_role))
        if missing:
            raise PackagerInputError(
                f"WatchWindow.min_volume_by_role: no declared minimum volume for {missing}. "
                "§11.5 closes on 7 days OR a declared minimum volume per affected role, "
                "whichever is LATER — a window that expires on a quiet weekend has "
                "observed nothing.")
        for role, volume in self.min_volume_by_role.items():
            _positive_int(volume, f"WatchWindow.min_volume_by_role[{role!r}]")
        object.__setattr__(self, "bands", _typed_tuple(
            self.bands, "WatchWindow.bands", WatchSignalBand, non_empty=True))
        covered = [b.signal_id for b in self.bands]
        if len(set(covered)) != len(covered):
            raise PackagerInputError(
                f"WatchWindow.bands: duplicate signals {sorted(covered)}")
        absent = sorted(set(REQUIRED_WATCH_SIGNALS) - set(covered))
        if absent:
            raise PackagerInputError(
                f"WatchWindow.bands: §11.5's signal table has six rows and {absent} have "
                "no band. An unbanded signal is a direction the last automatic safety net "
                "in the path cannot see.")
        fixed = _timestamp(self.bands_fixed_at, "WatchWindow.bands_fixed_at")
        opens = _timestamp(self.opens_at, "WatchWindow.opens_at")
        if fixed > opens:
            raise BandsNotFixedBeforeData(
                f"WatchWindow: bands were fixed at {self.bands_fixed_at}, after the window "
                f"opens at {self.opens_at}. §11.5 fixes bands at package assembly, before "
                "the data is seen; a band chosen afterwards is a band chosen against its "
                "own evidence.")
        if not isinstance(self.close_step, WatchWindowCloseStep):
            raise PackagerInputError(
                "WatchWindow.close_step: required, a WatchWindowCloseStep. An unclosed "
                "window is an open question, not a pass (§11.5).")
        _text(self.rollback_anchor_ref, "WatchWindow.rollback_anchor_ref")
        _text(self.activation_manifest_ref, "WatchWindow.activation_manifest_ref")
        _sha256(self.activation_manifest_sha256, "WatchWindow.activation_manifest_sha256")
        _text(self.comparison_method, "WatchWindow.comparison_method")

    def band_for(self, signal_id: str) -> WatchSignalBand:
        for band in self.bands:
            if band.signal_id == signal_id:
                return band
        raise KeyError(signal_id)

    def bands_sha256(self) -> str:
        """The digest an evaluation must present to prove it used THESE bands."""
        return schemas.content_hash(
            {"bands": [b.to_dict() for b in sorted(self.bands, key=lambda b: b.signal_id)],
             "fixed_at": self.bands_fixed_at})

    def to_dict(self) -> dict:
        return {
            "schema": WATCH_WINDOW_SCHEMA,
            "window_id": self.window_id,
            "package_id": self.package_id,
            "owner": self.owner,
            "output_class": WATCH_WINDOW_OUTPUT_CLASS,
            "incumbent_era": self.incumbent_era,
            "candidate_era": self.candidate_era,
            "comparison_method": self.comparison_method,
            "affected_roles": list(self.affected_roles),
            "duration_rule": ("later_of(min_duration_days, min_volume_by_role); §11.5 "
                              f"default is {DEFAULT_WATCH_WINDOW_DAYS} days"),
            "min_duration_days": self.min_duration_days,
            "min_volume_by_role": dict(self.min_volume_by_role),
            "signals": [b.to_dict() for b in self.bands],
            "bands_fixed_at": self.bands_fixed_at,
            "bands_sha256": self.bands_sha256(),
            "opens_at": self.opens_at,
            "close_step": self.close_step.to_dict(),
            "rollback_anchor_ref": self.rollback_anchor_ref,
            "activation_manifest_ref": self.activation_manifest_ref,
            "activation_manifest_sha256": self.activation_manifest_sha256,
            "rollback_anchor_rule": ("the rollback anchor stays live and verified for the "
                                     "whole window (§11.5)"),
            "computed_by": MODULE_ID,
        }


@dataclass(frozen=True)
class WatchObservation:
    """One signal's observed value, era-labelled and pointing at its raw samples."""

    signal_id: str
    value: Optional[float]
    observed_at: str
    era_label: str
    samples_ref: str

    def __post_init__(self) -> None:
        if self.signal_id not in REQUIRED_WATCH_SIGNALS:
            raise PackagerInputError(
                f"WatchObservation.signal_id: {self.signal_id!r} is not one of §11.5's six")
        if self.value is not None and (isinstance(self.value, bool)
                                       or not isinstance(self.value, (int, float))):
            raise PackagerInputError("WatchObservation.value: must be a number or None")
        _timestamp(self.observed_at, "WatchObservation.observed_at")
        _text(self.era_label, "WatchObservation.era_label")
        _text(self.samples_ref, "WatchObservation.samples_ref")

    def to_dict(self) -> dict:
        return {"signal_id": self.signal_id, "value": self.value,
                "observed_at": self.observed_at, "era_label": self.era_label,
                "samples_ref": self.samples_ref}


@dataclass(frozen=True)
class WatchWindowProgress:
    """What the window has actually seen so far, and against which bands."""

    now: str
    volume_by_role: Mapping[str, int]
    bands_sha256: str
    observations: tuple = ()

    def __post_init__(self) -> None:
        _timestamp(self.now, "WatchWindowProgress.now")
        _mapping(self.volume_by_role, "WatchWindowProgress.volume_by_role")
        for role, volume in self.volume_by_role.items():
            if not isinstance(volume, int) or isinstance(volume, bool) or volume < 0:
                raise PackagerInputError(
                    f"WatchWindowProgress.volume_by_role[{role!r}]: a non-negative int")
        _sha256(self.bands_sha256, "WatchWindowProgress.bands_sha256")
        object.__setattr__(self, "observations", _typed_tuple(
            self.observations, "WatchWindowProgress.observations", WatchObservation))

    def to_dict(self) -> dict:
        return {"now": self.now, "volume_by_role": dict(self.volume_by_role),
                "bands_sha256": self.bands_sha256,
                "observations": [o.to_dict() for o in self.observations]}


@dataclass(frozen=True)
class WatchSignalStanding:
    signal_id: str
    check: schemas.Check
    observed: Optional[float]
    band: Mapping

    def to_dict(self) -> dict:
        return {"signal_id": self.signal_id, "check": _check_dict(self.check),
                "observed": self.observed, "band": dict(self.band)}


@dataclass(frozen=True)
class WatchWindowRecommendation:
    """A RECOMMENDATION. It reverts nothing and it licenses no claim."""

    window_id: str
    state: str
    recommendation: str
    duration_met: bool
    volume_met: bool
    elapsed_days: float
    standings: tuple
    alarms: tuple
    unevaluable: tuple
    reasons: tuple

    def __post_init__(self) -> None:
        if self.recommendation not in WATCH_RECOMMENDATIONS:
            raise PackagerInputError(
                f"WatchWindowRecommendation.recommendation: {self.recommendation!r} is not "
                f"one of {list(WATCH_RECOMMENDATIONS)}. The vocabulary is closed and "
                "contains no action on production: a signal outside its band raises a "
                "decision package, it does not revert anything (§11.5).")

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "record_class": WATCH_WINDOW_OUTPUT_CLASS,
            "state": self.state,
            "recommendation": self.recommendation,
            "duration_met": self.duration_met,
            "volume_met": self.volume_met,
            "elapsed_days": self.elapsed_days,
            "standings": [s.to_dict() for s in self.standings],
            "alarms": list(self.alarms),
            "unevaluable": list(self.unevaluable),
            "reasons": list(self.reasons),
        }


def watch_window_close_condition(window: WatchWindow,
                                 progress: WatchWindowProgress) -> dict:
    """§11.5's *"whichever is later"*, as arithmetic rather than a reading.

    Both conditions must hold. `later` is the operative word: duration OR volume
    would let a busy first day close a seven-day window, and duration alone lets a
    quiet week close one that saw no traffic.
    """
    if not isinstance(window, WatchWindow):
        raise PackagerInputError("watch_window_close_condition: window must be a WatchWindow")
    if not isinstance(progress, WatchWindowProgress):
        raise PackagerInputError(
            "watch_window_close_condition: progress must be a WatchWindowProgress")
    opens = _timestamp(window.opens_at, "window.opens_at")
    now = _timestamp(progress.now, "progress.now")
    elapsed_days = (now - opens).total_seconds() / 86400.0
    duration_met = elapsed_days >= window.min_duration_days
    short: list = []
    for role in window.affected_roles:
        required = window.min_volume_by_role[role]
        served = progress.volume_by_role.get(role, 0)
        if served < required:
            short.append(f"{role}: {served}/{required}")
    volume_met = not short
    reasons: list = []
    if not duration_met:
        reasons.append(
            f"{elapsed_days:.2f} of {window.min_duration_days} declared days elapsed")
    if short:
        reasons.append(f"declared minimum volume not yet served — {'; '.join(short)}")
    return {"elapsed_days": elapsed_days, "duration_met": duration_met,
            "volume_met": volume_met, "closeable": duration_met and volume_met,
            "reasons": tuple(reasons)}


def evaluate_watch_window(window: WatchWindow,
                          progress: WatchWindowProgress) -> WatchWindowRecommendation:
    """Compare the observations against the bands fixed at assembly.

    Refuses outright when `progress.bands_sha256` is not the window's own digest.
    That is not pedantry: it is the only way to tell "we compared against the bands
    we committed to" from "we compared against bands that moved", and the second is
    indistinguishable from the first in the output.

    Three outcomes per signal, and the middle one is load-bearing: a signal with no
    observation is `COULD_NOT_CHECK`, which yields `incomplete_evidence` — never
    `close_with_no_regression_observed`. A watch window that reports "no regression"
    because it collected nothing is the AutoPilot failure this project already paid
    for: every dashboard green while the producer was dead.
    """
    if not isinstance(window, WatchWindow):
        raise PackagerInputError("evaluate_watch_window: window must be a WatchWindow")
    if not isinstance(progress, WatchWindowProgress):
        raise PackagerInputError(
            "evaluate_watch_window: progress must be a WatchWindowProgress")
    expected = window.bands_sha256()
    if progress.bands_sha256 != expected:
        raise BandsNotFixedBeforeData(
            f"the observations were compared against bands {progress.bands_sha256[:12]}, "
            f"but this window fixed {expected[:12]} at assembly. §11.5 fixes the bands "
            "before the window opens; an evaluation against different bands is a band "
            "chosen after seeing the data, and it is indistinguishable from a valid one "
            "in the output.")

    # EVERY observation of a signal, not the first one seen. `setdefault` kept the
    # first and discarded the rest, so a second sample four times outside the band
    # was silently dropped and the window recommended
    # `close_with_no_regression_observed` over an observed regression — the
    # last-wins/first-wins collapse this package has already had three times, in the
    # one place where the discarded record is an ALARM. Worst standing governs:
    # §11.5 says a signal outside its band raises a decision package, and one
    # excursion is an excursion however many samples were inside.
    by_signal: dict = {}
    for observation in progress.observations:
        if observation.era_label != window.candidate_era:
            continue
        by_signal.setdefault(observation.signal_id, []).append(observation)

    standings: list = []
    alarms: list = []
    unevaluable: list = []
    reasons: list = []
    for signal_id in REQUIRED_WATCH_SIGNALS:
        band = window.band_for(signal_id)
        observed = by_signal.get(signal_id) or ()
        checks = [band.standing_for(o.value) for o in observed]
        check = _worst(checks) if checks else _cnc(
            f"{signal_id}: no observation labelled era {window.candidate_era!r} was "
            "recorded; an unobserved signal is not a quiet one")
        value = None
        for observation, standing in zip(observed, checks):
            if standing.outcome == check.outcome:
                value = observation.value
                break
        if len(observed) > 1:
            check = schemas.Check(check.outcome, check.reasons + (
                f"{signal_id}: {len(observed)} observations were folded and the worst "
                "standing governs; a later sample cannot be answered by an earlier one",))
        standings.append(WatchSignalStanding(
            signal_id=signal_id, check=check, observed=value, band=band.to_dict()))
        if check.outcome == schemas.FAIL:
            alarms.append(signal_id)
            reasons.extend(check.reasons)
        elif check.outcome == schemas.COULD_NOT_CHECK:
            unevaluable.append(signal_id)
            reasons.extend(check.reasons)

    condition = watch_window_close_condition(window, progress)
    reasons.extend(condition["reasons"])
    if alarms:
        recommendation = WATCH_RAISE_DECISION_PACKAGE
    elif unevaluable:
        recommendation = WATCH_INCOMPLETE_EVIDENCE
    elif condition["closeable"]:
        recommendation = WATCH_CLOSE_NO_REGRESSION
    else:
        recommendation = WATCH_CONTINUE
    state = WATCH_STATE_CLOSEABLE if condition["closeable"] else WATCH_STATE_OPEN
    return WatchWindowRecommendation(
        window_id=window.window_id, state=state, recommendation=recommendation,
        duration_met=condition["duration_met"], volume_met=condition["volume_met"],
        elapsed_days=condition["elapsed_days"], standings=tuple(standings),
        alarms=tuple(alarms), unevaluable=tuple(unevaluable), reasons=tuple(reasons))


@dataclass(frozen=True)
class WatchWindowClosure:
    """The explicit close, with its verdict and the human who recorded it."""

    window_id: str
    verdict: str
    closed_by: str
    closed_at: str
    recommendation: WatchWindowRecommendation

    def to_dict(self) -> dict:
        return {"window_id": self.window_id, "state": WATCH_STATE_CLOSED,
                "verdict": self.verdict, "closed_by": self.closed_by,
                "closed_at": self.closed_at,
                "recommendation_at_close": self.recommendation.to_dict(),
                "record_class": WATCH_WINDOW_OUTPUT_CLASS}


def close_watch_window(window: WatchWindow, progress: WatchWindowProgress, *,
                       verdict: str, closed_by: str,
                       closed_at: str) -> WatchWindowClosure:
    """Close the window with a verdict. Refuses to close one that is still open.

    §11.5 makes closing an explicit action that records a verdict. Two refusals
    keep it explicit: the close condition must be met (otherwise the closure would
    manufacture a pass out of an unfinished window), and `closed_by` must be a
    human — AutoKernel computes and reports, and the window belongs to whoever
    executed the cutover.
    """
    recommendation = evaluate_watch_window(window, progress)
    if recommendation.state != WATCH_STATE_CLOSEABLE:
        raise WatchWindowOpen(
            f"the window has not met its close condition: {'; '.join(recommendation.reasons)}. "
            "Closing it now would record a verdict over a window that has not observed what "
            "it committed to observe.")
    if verdict not in window.close_step.verdict_vocabulary:
        raise PackagerInputError(
            f"close_watch_window: verdict {verdict!r} is not among "
            f"{list(window.close_step.verdict_vocabulary)}")
    _require_human_actor(_text(closed_by, "close_watch_window: closed_by"),
                         "close_watch_window: closed_by", error=PackagerInputError)
    _timestamp(closed_at, "close_watch_window: closed_at")
    if verdict == "no_regression_observed" and recommendation.unevaluable:
        raise PackagerInputError(
            f"close_watch_window: {verdict!r} cannot be recorded while "
            f"{list(recommendation.unevaluable)} were never evaluated. "
            "'We did not look' is not 'we looked and saw nothing'.")
    if verdict == "no_regression_observed" and recommendation.alarms:
        # The symmetric half, and the one that was missing. "We did not look" was
        # refused while "we looked and it alarmed" was accepted: a window whose
        # throughput signal sat outside its band could be closed
        # `no_regression_observed`, and the closure is the record of record — the
        # recommendation it contradicts is a field inside it that nobody re-reads.
        raise PackagerInputError(
            f"close_watch_window: {verdict!r} cannot be recorded while "
            f"{list(recommendation.alarms)} stand outside their bands. §11.5: a signal "
            "outside its band raises a decision package. The verdict for that window is "
            "`regression_observed` or `inconclusive`; a closure that contradicts its own "
            "recommendation is how the observation stops existing.")
    return WatchWindowClosure(window_id=window.window_id, verdict=verdict,
                              closed_by=closed_by, closed_at=closed_at,
                              recommendation=recommendation)


# =============================================================================
# The four-part decision package (OPERATING_CONSTRAINTS.md:69-78)
# =============================================================================

#: Every axis the contract names. A tradeoff block missing one is the shape of an
#: option presented favourably — reversibility is the one that goes missing, and it
#: is the one that matters for a freeze.
REQUIRED_TRADEOFF_AXES = ("cost", "risk", "time", "quality", "reversibility")


@dataclass(frozen=True)
class DecisionOption:
    option_id: str
    label: str
    entails: str
    tradeoffs: Mapping
    supporting_data: tuple = ()

    def __post_init__(self) -> None:
        _text(self.option_id, "DecisionOption.option_id")
        _text(self.label, "DecisionOption.label")
        _text(self.entails, "DecisionOption.entails")
        _mapping(self.tradeoffs, "DecisionOption.tradeoffs", non_empty=True)
        missing = [axis for axis in REQUIRED_TRADEOFF_AXES if not self.tradeoffs.get(axis)]
        if missing:
            raise PackagerInputError(
                f"DecisionOption({self.option_id}).tradeoffs is missing {missing}. "
                "OPERATING_CONSTRAINTS.md:69-78 names cost, risk, time, quality and "
                "reversibility; an option presented without one of them is an option "
                "presented favourably.")
        for axis in REQUIRED_TRADEOFF_AXES:
            _text(self.tradeoffs[axis], f"DecisionOption({self.option_id}).tradeoffs.{axis}")
        object.__setattr__(self, "supporting_data", _str_tuple(
            self.supporting_data, "DecisionOption.supporting_data", non_empty=False))

    def to_dict(self) -> dict:
        return {"option_id": self.option_id, "label": self.label,
                "entails": self.entails,
                "tradeoffs": {axis: self.tradeoffs[axis] for axis in REQUIRED_TRADEOFF_AXES},
                "supporting_data": list(self.supporting_data)}


@dataclass(frozen=True)
class DecisionRecommendation:
    option_id: str
    why: str
    tie_breaker: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.option_id, "DecisionRecommendation.option_id")
        _text(self.why, "DecisionRecommendation.why")
        _opt_text(self.tie_breaker, "DecisionRecommendation.tie_breaker")

    def to_dict(self) -> dict:
        return {"option_id": self.option_id, "why": self.why,
                "tie_breaker": self.tie_breaker}


@dataclass(frozen=True)
class DecisionPackage:
    """Context, options, recommendation, default — and never an open question.

    *"Never escalate a decision with an open-ended question."* A trailing `?` in the
    context or the default is exactly that escalation wearing a decision package's
    formatting, so it is refused rather than rendered.
    """

    context: str
    options: tuple
    recommendation: DecisionRecommendation
    default_outcome: str

    def __post_init__(self) -> None:
        _text(self.context, "DecisionPackage.context")
        object.__setattr__(self, "options", _typed_tuple(
            self.options, "DecisionPackage.options", DecisionOption, non_empty=True))
        if not 2 <= len(self.options) <= 4:
            raise PackagerInputError(
                f"DecisionPackage.options: {len(self.options)} options. The contract asks "
                "for 2–4 concrete choices; one is an instruction and five is a survey.")
        ids = [o.option_id for o in self.options]
        if len(set(ids)) != len(ids):
            raise PackagerInputError(f"DecisionPackage.options: duplicate ids {sorted(ids)}")
        if not isinstance(self.recommendation, DecisionRecommendation):
            raise PackagerInputError(
                "DecisionPackage.recommendation: required, a DecisionRecommendation")
        if self.recommendation.option_id not in ids:
            raise PackagerInputError(
                f"DecisionPackage.recommendation names {self.recommendation.option_id!r}, "
                f"which is not among the options {ids}")
        _text(self.default_outcome, "DecisionPackage.default_outcome")
        for label, value in (("context", self.context),
                             ("default_outcome", self.default_outcome)):
            if value.strip().endswith("?"):
                raise PackagerInputError(
                    f"DecisionPackage.{label} ends in a question. Never escalate a "
                    "decision with an open-ended question "
                    "(OPERATING_CONSTRAINTS.md:69-78); state the fork and the options.")

    @property
    def ordered_options(self) -> tuple:
        """Recommended first, as the delivery contract requires."""
        chosen = [o for o in self.options if o.option_id == self.recommendation.option_id]
        return tuple(chosen + [o for o in self.options
                               if o.option_id != self.recommendation.option_id])

    def to_dict(self) -> dict:
        return {"context": self.context,
                "options": [o.to_dict() for o in self.ordered_options],
                "recommendation": self.recommendation.to_dict(),
                "default_outcome": self.default_outcome}


def _blockers_are_cell_scoped(evaluation: TrustedEvaluation) -> bool:
    """True when every T3 objection is a CELL, not the integrity spine.

    §10.4's waiver covers cells: v8's excluded two model/shape pairs from a matrix.
    It did not waive whether the binary linked correctly. So the "grant a scoped
    waiver" option is offered only when a waiver could in principle apply, and is
    withheld — rather than offered and then refused — when the objection is a phase
    blocker.
    """
    computation = evaluation.result.verdict_computation
    return bool(computation.failed_cells or computation.unevaluable_cells) \
        and not computation.blocking_reasons


def build_decision_package(*, package_id: str, state: str,
                           freeze_request: OperatorFreezeRequest,
                           evaluation: TrustedEvaluation, version: NextVersion,
                           transaction: t3.TransactionPlan, rollback: RollbackPlan,
                           watch_window: WatchWindow, cutover_request: CutoverRequest,
                           findings: Sequence["PackageFinding"]) -> DecisionPackage:
    """Derive the four parts from the package's own facts.

    Deterministic on purpose. A decision package composed by a narrating LLM is the
    same failure invariant 14 forbids for readiness — the numbers would come from
    prose rather than from records — so the options, the recommendation and the
    default are all computed from the verdict, the findings and the transaction.

    Every option set contains a "do not freeze" member and none contains an option
    AutoKernel performs: the whole vocabulary is things the operator does.
    """
    if state not in PACKAGE_STATES:
        raise PackagerInputError(f"build_decision_package: unknown state {state!r}")
    blocking = [f for f in findings if f.gating and f.outcome != schemas.PASS]
    blocker_lines = tuple(f"{f.code}: {f.detail}" for f in blocking)
    window = freeze_request.compute_window
    rollback_line = (f"{rollback.rollback_branch}@{rollback.rollback_head[:12]} from the "
                     f"archived incumbent at {rollback.incumbent_archive_path} "
                     f"(archive check {rollback.archive_check.outcome})")
    decline = DecisionOption(
        option_id="decline",
        label=f"Decline this package and keep production on {version.incumbent_branch}",
        entails=("Nothing is executed. The champion keeps accumulating against the "
                 "current anchor and a later freeze request produces a fresh package."),
        tradeoffs={
            "cost": "zero now; the next package re-runs the T3 matrix, a full compute window",
            "risk": "none to production; the accumulated champion stays unreleased",
            "time": "immediate",
            "quality": "production keeps the incumbent kernel's measured behaviour",
            "reversibility": "n/a — nothing changes",
        },
        supporting_data=(f"incumbent {version.incumbent_branch} "
                         f"(era {watch_window.incumbent_era})",))

    if state == STATE_READY:
        options = [
            DecisionOption(
                option_id="execute-in-window",
                label=(f"Execute the freeze to {version.next_branch} in compute window "
                       f"{window.window_id}"),
                entails=(f"You run the pre-validated command sequence in order inside "
                         f"{window.window_id} ({window.opens_at} → {window.closes_at}, "
                         f"owner {window.owner}), write the drafted era rows, ratify the "
                         f"AutoPilot rebaseline, then route the cutover request to "
                         f"{list(cutover_request.needs_routing_to)} so the inference owner "
                         "schedules the reload at its own boundary."),
                tradeoffs={
                    "cost": f"one compute window of {window.hours:.1f}h, owned by {window.owner}",
                    "risk": ("four human-only writes, all drafted and pre-validated; T3 "
                             f"returned {evaluation.verdict} with no blocking finding"),
                    "time": f"{window.hours:.1f}h plus the {watch_window.min_duration_days}-day watch window",
                    "quality": (f"claims licensed: "
                                f"{len(evaluation.result.receipt.claims)}; suppressed by "
                                f"waiver: {len(evaluation.result.receipt.suppressed_claims)}"),
                    "reversibility": f"rollback to {rollback_line}",
                },
                supporting_data=(f"bundle {evaluation.bundle_sha256 or 'unsealed'}",
                                 f"fingerprint {evaluation.request_fingerprint[:12]}")),
            DecisionOption(
                option_id="defer-to-a-later-window",
                label="Defer: keep the package and execute in a later window",
                entails=("Nothing is executed now. The package stays valid while the "
                         "anchor holds; if production moves, the seal is re-anchored and "
                         "T3 is re-run (invariant 1, AK-D22 ANCHOR_MOVED)."),
                tradeoffs={
                    "cost": "zero now; a re-run if the anchor moves",
                    "risk": "champion drift grows and the highest-risk code stays unexercised",
                    "time": "deferred",
                    "quality": "unchanged — production keeps the incumbent kernel",
                    "reversibility": "n/a — nothing changes",
                },
                supporting_data=(f"seal anchored on "
                                 f"{transaction.rollback_head or 'unknown'}[:12]",)),
            decline,
        ]
        recommendation = DecisionRecommendation(
            option_id="execute-in-window",
            why=("T3 returned "
                 f"{evaluation.verdict} on the sealed candidate, the rollback anchor "
                 f"verified ({rollback.archive_check.outcome}), every operator command is "
                 "pre-validated and the watch window's bands are fixed. The remaining "
                 "cost is the declared compute window, and deferring re-buys that cost "
                 "later while the champion drifts further from the anchor."))
        default = (f"No freeze occurs. Production stays on {version.incumbent_branch}, the "
                   "cutover request is not routed, and this package remains a record the "
                   "next freeze request supersedes.")
    elif state == STATE_BLOCKED:
        options = [
            DecisionOption(
                option_id="fix-the-blockers",
                label="Do not freeze; clear the named blockers and re-run T3",
                entails=("The blockers are addressed at their source and a fresh T3 run "
                         "produces a new package. The rerun guard's cooldown applies to an "
                         "unchanged fingerprint (a re-run that changed nothing is refused, "
                         "not re-graded)."),
                tradeoffs={
                    "cost": "another compute window once the blockers are cleared",
                    "risk": "none to production",
                    "time": "as long as the blockers take",
                    "quality": "the release keeps the evidence standard it is failing now",
                    "reversibility": "n/a — nothing changes",
                },
                supporting_data=blocker_lines or ("no blocker detail recorded",)),
        ]
        if _blockers_are_cell_scoped(evaluation):
            options.append(DecisionOption(
                option_id="scoped-operator-waiver",
                label="Author a scoped operator waiver for the named cells and re-run T3",
                entails=("You author an `epyc.autokernel.operator_waiver.v1` document "
                         "under the trust-boundary path set naming the exact cells, the "
                         "reason and the forfeited claims; it is hash-pinned into the "
                         "bundle and T3 re-runs. v8 shipped this way, on "
                         "`promotion_decision: false` plus WAIVE-Q8."),
                tradeoffs={
                    "cost": "the waiver document plus one T3 re-run",
                    "risk": ("the release makes NO claim for the waived cells — a waived "
                             "cell forfeits its claim rather than weakening it"),
                    "time": "shorter than fixing the cells",
                    "quality": (f"forfeits the claims on "
                                f"{list(evaluation.result.verdict_computation.failed_cells)}"
                                f" / "
                                f"{list(evaluation.result.verdict_computation.unevaluable_cells)}"),
                    "reversibility": f"rollback to {rollback_line}",
                },
                supporting_data=("AutoKernel cannot author or grant this: a waiver is "
                                 "human-authored and the gate only verifies its hash and "
                                 "predicate (§10.4)",)))
        options.append(decline)
        recommendation = DecisionRecommendation(
            option_id="fix-the-blockers",
            why=("T3 returned FAIL and the package is blocked. Correctness, integrity and "
                 "identity are lexicographically prior to speed (invariant 6), so no "
                 "throughput result on this candidate compensates for what is failing."),
            tie_breaker=("whether every blocker is cell-scoped: a waiver covers cells, "
                         "never the integrity spine"))
        default = (f"No freeze occurs. Production stays on {version.incumbent_branch} and "
                   "the blockers remain open on the record.")
    else:
        options = [
            DecisionOption(
                option_id="supply-the-missing-evidence",
                label="Supply what could not be checked and re-run T3",
                entails=("The COULD_NOT_CHECK items are resolved — a receipt read, a "
                         "digest supplied, an archive verified — and T3 re-runs on the "
                         "same seal."),
                tradeoffs={
                    "cost": "the missing evidence plus one T3 re-run",
                    "risk": "none to production",
                    "time": "usually shorter than a full re-measurement",
                    "quality": "restores the evidence standard the package is short of",
                    "reversibility": "n/a — nothing changes",
                },
                supporting_data=blocker_lines or ("no unevaluable detail recorded",)),
            DecisionOption(
                option_id="proceed-on-a-declared-forfeit",
                label="Proceed with a waiver that names what was never checked",
                entails=("You author a waiver naming the unchecked items and the claims "
                         "they forfeit. The release then makes no claim in those areas, "
                         "and says so in its receipt."),
                tradeoffs={
                    "cost": "the waiver document plus one T3 re-run",
                    "risk": ("higher than it looks: 'we could not tell' is not 'it is "
                             "fine', and the forfeited claims are exactly the ones nobody "
                             "measured"),
                    "time": "short",
                    "quality": "forfeits every claim resting on the unchecked evidence",
                    "reversibility": f"rollback to {rollback_line}",
                },
                supporting_data=("a waiver is human-authored; AutoKernel verifies, never "
                                 "grants (§10.4)",)),
            decline,
        ]
        recommendation = DecisionRecommendation(
            option_id="supply-the-missing-evidence",
            why=("The package is incomplete rather than failing: nothing says the "
                 "candidate is worse, and nothing says it is not. Resolving the "
                 "unevaluable items is cheaper than forfeiting the claims that rest on "
                 "them."))
        default = (f"No freeze occurs. Production stays on {version.incumbent_branch} and "
                   "the unevaluable items stay open.")

    context = (
        f"Package {package_id} answers freeze request {freeze_request.request_id} "
        f"({freeze_request.requested_by}, {freeze_request.requested_at}) for source tree "
        f"{freeze_request.source_tree}. T3 returned {evaluation.verdict}; the package "
        f"state is {state}. AutoKernel cannot proceed past this point: the freeze, the "
        "era rows, the AutoPilot rebaseline and the cutover are human-only writes "
        "(MEASUREMENT.md:140-142), and the cutover is additionally scheduled by whoever "
        "owns the inference.")
    return DecisionPackage(context=context, options=tuple(options),
                           recommendation=recommendation, default_outcome=default)


def render_decision_package(package: DecisionPackage, *, title: str,
                            first_page_notice: Optional[str] = None) -> str:
    """Render the four parts as compact markdown, recommended option FIRST.

    §10.6: above the blast-radius ceiling the package *"is marked
    `REQUIRES_HUMAN_CODE_REVIEW` and says so on its first page"*. The notice is
    emitted immediately under the title, before the context, so that "first page"
    means the first thing read rather than a field somewhere in a JSON blob.
    """
    if not isinstance(package, DecisionPackage):
        raise PackagerInputError("render_decision_package: package must be a DecisionPackage")
    lines = [f"# {_text(title, 'render_decision_package: title')}"]
    if first_page_notice:
        lines.extend(["", f"> **{first_page_notice}**"])
    lines.extend(["", "## 1. Context", "", package.context, "", "## 2. Options", ""])
    for index, option in enumerate(package.ordered_options):
        marker = " **(Recommended)**" if option.option_id == package.recommendation.option_id \
            else ""
        lines.append(f"### Option {chr(ord('A') + index)} — {option.label}{marker}")
        lines.extend(["", f"- **Entails**: {option.entails}"])
        for axis in REQUIRED_TRADEOFF_AXES:
            lines.append(f"- **{axis.capitalize()}**: {option.tradeoffs[axis]}")
        for datum in option.supporting_data:
            lines.append(f"- *Supporting*: {datum}")
        lines.append("")
    lines.extend(["## 3. Recommendation", "",
                  f"**{package.recommendation.option_id}** — {package.recommendation.why}"])
    if package.recommendation.tie_breaker:
        lines.extend(["", f"Tie-breaker: {package.recommendation.tie_breaker}"])
    lines.extend(["", "## 4. Default", "", package.default_outcome, "", "---", "",
                  PACKAGE_NOTICE])
    return "\n".join(lines)


# =============================================================================
# The package
# =============================================================================

@dataclass(frozen=True)
class PackageFinding:
    """One thing wrong, or unverifiable, with the package as assembled.

    Findings are how evidence problems are recorded. Exceptions are how MATERIAL
    and AUTHORITY problems are refused. The split matters: raising on a failed cell
    would delete the record of the failure, and a package that cannot be assembled
    tells the operator less than a blocked one that names why.
    """

    code: str
    detail: str
    outcome: str
    gating: bool = True

    def __post_init__(self) -> None:
        _text(self.code, "PackageFinding.code")
        _text(self.detail, "PackageFinding.detail")
        if self.outcome not in (schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK):
            raise PackagerInputError(
                f"PackageFinding.outcome: {self.outcome!r} is not a Check outcome")
        _bool(self.gating, "PackageFinding.gating")

    def to_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail, "outcome": self.outcome,
                "gating": self.gating}


@dataclass(frozen=True)
class LinkageSummary:
    """The linkage answer, DERIVED from T3's own build/linkage phase.

    Not re-asserted here. The package schema refuses a non-PASS linkage on a
    passing verdict — *"a binary that inherits another tree's ggml runs silently
    wrong"* — so the value has to come from the phase that actually read the
    verifier receipts, or the refusal is checking this module's opinion.
    """

    status: str
    receipt: str
    per_backend: Mapping = field(default_factory=dict)
    reasons: tuple = ()

    def __post_init__(self) -> None:
        if self.status not in (schemas.PASS, schemas.FAIL, schemas.COULD_NOT_CHECK):
            raise PackagerInputError(f"LinkageSummary.status: {self.status!r}")
        _text(self.receipt, "LinkageSummary.receipt")

    def to_dict(self) -> dict:
        return {"status": self.status, "receipt": self.receipt,
                "per_backend": dict(self.per_backend), "reasons": list(self.reasons)}


def derive_linkage_summary(evaluation: TrustedEvaluation) -> LinkageSummary:
    """Read the linkage answer out of T3's build/linkage phase result.

    A PASSing phase that verified NOTHING yields COULD_NOT_CHECK rather than PASS:
    every backend's cells can be dropped by the §3.2 unchanged test, in which case
    the candidate's linkage was never checked because there was nothing new to
    check. That is an honest third outcome, and it stops the guarantee from being
    obtainable by deleting what it inspects.
    """
    if not isinstance(evaluation, TrustedEvaluation):
        raise PackagerInputError(
            "derive_linkage_summary: evaluation must be a TrustedEvaluation")
    phase = evaluation.result.phase(t3.PHASE_BUILD_LINKAGE)
    per_backend = {
        key.split(".", 1)[1]: value.get("outcome")
        for key, value in phase.detail.items()
        if key.startswith("linkage.") and isinstance(value, Mapping)}
    receipt = (f"{t3.PHASE_BUILD_LINKAGE} phase of T3 run {evaluation.result.run_id} "
               f"via {t3.LINKAGE_VERIFIER_RELPATH}")
    if phase.check.outcome == schemas.PASS and not per_backend:
        return LinkageSummary(
            status=schemas.COULD_NOT_CHECK, receipt=receipt, per_backend={},
            reasons=("the build/linkage phase passed without verifying any backend — "
                     "every backend's cells were dropped, so no candidate linkage was "
                     "proven and the incumbent's stands",))
    return LinkageSummary(status=phase.check.outcome, receipt=receipt,
                          per_backend=per_backend, reasons=tuple(phase.check.reasons))


def _derive_package_state(findings: Sequence[PackageFinding]) -> str:
    """FAIL ⇒ BLOCKED, else COULD_NOT_CHECK ⇒ INCOMPLETE, else READY.

    The one derivation. `ReleasePackage.__post_init__` re-runs it and refuses a
    disagreement in either direction, so a state cannot be stamped onto a package
    whose own findings say otherwise — the hole `readiness.ReadinessSignal` had
    until it was closed the same way.
    """
    gating = [f for f in findings if f.gating]
    if any(f.outcome == schemas.FAIL for f in gating):
        return STATE_BLOCKED
    if any(f.outcome == schemas.COULD_NOT_CHECK for f in gating):
        return STATE_INCOMPLETE
    return STATE_READY


@dataclass(frozen=True)
class ReleasePackage:
    """§7.6 — what AutoKernel hands the operator, and nothing more.

    *"It contains no production write and no authority claim."* The second half is
    enforced rather than intended: `__post_init__` runs
    `schemas.find_authority_flavoured_keys()` over this package's own rendered
    record and refuses any hit, so a freeze/cutover/promotion flag added later
    fails construction instead of quietly becoming load-bearing.
    """

    package_id: str
    campaign_id: str
    source_tree: str
    created_at: str
    freeze_request: OperatorFreezeRequest
    sealed: SealedRelease
    evaluation: TrustedEvaluation
    version: NextVersion
    transaction: t3.TransactionPlan
    rollback: RollbackPlan
    era_row_draft: Mapping
    rebaseline_note: str
    linkage: LinkageSummary
    command_review: CommandSequenceReview
    watch_window: WatchWindow
    cutover_request: CutoverRequest
    decision_package: DecisionPackage
    release_plan: Mapping
    active_waivers: tuple
    waiver_bindings: tuple
    change_classes: tuple
    diff_complexity: Mapping
    findings: tuple
    state: str
    readiness_signal: Optional[Mapping] = None

    def __post_init__(self) -> None:
        _text(self.package_id, "ReleasePackage.package_id")
        if not self.package_id.startswith("akr-"):
            raise PackagerInputError("ReleasePackage.package_id: must start with 'akr-'")
        _text(self.campaign_id, "ReleasePackage.campaign_id")
        if not self.campaign_id.startswith("ak-"):
            raise PackagerInputError("ReleasePackage.campaign_id: must start with 'ak-'")
        if self.source_tree not in schemas.SOURCE_TREES:
            raise PackagerInputError(
                f"ReleasePackage.source_tree: {self.source_tree!r} is not one of "
                f"{sorted(schemas.SOURCE_TREES)}")
        _timestamp(self.created_at, "ReleasePackage.created_at")
        object.__setattr__(self, "findings", _typed_tuple(
            self.findings, "ReleasePackage.findings", PackageFinding))
        if self.state not in PACKAGE_STATES:
            raise PackagerInputError(
                f"ReleasePackage.state: {self.state!r} is not one of {list(PACKAGE_STATES)}")
        derived = _derive_package_state(self.findings)
        if derived != self.state:
            raise StateNotDerived(
                f"ReleasePackage.state is {self.state!r} but its own findings yield "
                f"{derived!r}. A state that can be stamped independently of the evidence "
                "is a state that can be wrong in the direction nobody notices — and the "
                f"direction that matters here is {STATE_READY!r}.")
        _text(self.rebaseline_note, "ReleasePackage.rebaseline_note")
        _mapping(self.era_row_draft, "ReleasePackage.era_row_draft", non_empty=True)
        _mapping(self.release_plan, "ReleasePackage.release_plan", non_empty=True)
        _mapping(self.diff_complexity, "ReleasePackage.diff_complexity", non_empty=True)
        object.__setattr__(self, "change_classes", _str_tuple(
            self.change_classes, "ReleasePackage.change_classes", non_empty=False))
        authority = schemas.find_authority_flavoured_keys(self.to_dict())
        if authority:
            raise ProductionWriteRefused(
                f"ReleasePackage carries authority-flavoured keys {authority}. §1.3: a "
                "freeze crosses four human-only trust boundaries, so there is no such "
                "authority for a machine-authored record to declare (§7.6: the package "
                "'contains no production write and no authority claim').")

    # -- derived properties ---------------------------------------------------

    @property
    def blocking_findings(self) -> tuple:
        return tuple(f for f in self.findings if f.gating and f.outcome != schemas.PASS)

    @property
    def requires_human_code_review(self) -> bool:
        """§10.6, derived — a declared `False` cannot suppress a traced reason.

        Three independent sources, OR-ed: T3's own per-backend ceiling assessment,
        a `core_header` change class (a KIND of change, not a size band, AK-D30),
        and a diff that touches shared ggml core.

        The third source is UNSTATED-is-yes. `.get(…) is True` read a missing key as
        "no", so a diff nobody classified cleared the §10.6 marker by omission — and
        "we did not measure the blast radius" is not "the blast radius was small"
        (`t3._requires_human_code_review` says the same for a missing per-backend
        assessment).
        """
        return bool(
            self.evaluation.result.requires_human_code_review
            or "core_header" in self.change_classes
            or not isinstance(self.diff_complexity.get("touches_shared_core"), bool)
            or self.diff_complexity.get("touches_shared_core") is True)

    @property
    def first_page_notice(self) -> Optional[str]:
        if not self.requires_human_code_review:
            return None
        reasons = [self.evaluation.result.first_page_notice] \
            if self.evaluation.result.first_page_notice else []
        if "core_header" in self.change_classes:
            reasons.append("a member change class is `core_header`, which reaches every "
                           "op in both the CPU and GPU builds (AK-D30, §8.5.1)")
        if not isinstance(self.diff_complexity.get("touches_shared_core"), bool):
            reasons.append("`diff_complexity.touches_shared_core` was never stated, so "
                           "no assessment says this diff stays out of shared ggml core "
                           "(§10.6)")
        elif self.diff_complexity.get("touches_shared_core") is True:
            reasons.append("the diff touches shared ggml core (§10.6)")
        return f"{integrity.REQUIRES_HUMAN_CODE_REVIEW} — " + "; ".join(reasons)

    def to_dict(self) -> dict:
        """The `epyc.autokernel.release_package.v1` record.

        `operator_command_sequence` carries only PRE-VALIDATED commands, because
        `schemas.validate_release_package` refuses `validated: false` there and a
        package must not smuggle an unchecked command into the array that says
        everything in it was checked. Nothing is hidden: the unvalidated ones are in
        `command_review.unvalidated_steps` and each has produced a blocking finding.
        """
        return {
            "schema": PACKAGE_SCHEMA,
            "record_class": RECORD_CLASS,
            "package_id": self.package_id,
            "campaign_id": self.campaign_id,
            "source_tree": self.source_tree,
            "created_at": self.created_at,
            "packaged_by": MODULE_ID,
            "state": self.state,
            "terminal_success_state": STATE_READY,
            "executed_by": EXECUTED_BY,
            "notice": PACKAGE_NOTICE,
            "refusals": sorted(REFUSED_CAPABILITIES),
            "freeze_request": self.freeze_request.to_dict(),
            "sealed_candidate": self.sealed.to_dict(),
            "t3_verdict": {
                "verdict": self.evaluation.verdict,
                "bundle_sha256": self.evaluation.bundle_sha256,
                "phase_results": {p.phase_id: p.to_dict()
                                  for p in self.evaluation.result.phase_results},
                "evaluation": self.evaluation.to_dict(),
                "receipt": self.evaluation.result.receipt.to_dict(),
            },
            "active_waivers": [dict(w) for w in self.active_waivers],
            "waiver_bindings": [dict(w) for w in self.waiver_bindings],
            "release_plan": dict(self.release_plan),
            "transaction_plan": self.transaction.to_dict(),
            "next_version": self.version.to_dict(),
            "rollback_plan": self.rollback.to_dict(),
            "draft_era_registry_row": dict(self.era_row_draft),
            "draft_autopilot_rebaseline_note": self.rebaseline_note,
            "linkage_verification": self.linkage.to_dict(),
            "operator_command_sequence": [
                c.to_dict() for c in self.command_review.validated_commands],
            "command_review": self.command_review.to_dict(),
            "watch_window": self.watch_window.to_dict(),
            "cutover_request": self.cutover_request.to_dict(),
            "decision_package": self.decision_package.to_dict(),
            "change_classes": list(self.change_classes),
            "diff_complexity": dict(self.diff_complexity),
            "requires_human_code_review": self.requires_human_code_review,
            "first_page_notice": self.first_page_notice,
            "findings": [f.to_dict() for f in self.findings],
            "blocking_finding_codes": sorted({f.code for f in self.blocking_findings}),
            "readiness_signal": None if self.readiness_signal is None
            else dict(self.readiness_signal),
        }

    def sha256(self) -> str:
        return schemas.content_hash(self.to_dict())

    def schema_violations(self) -> list:
        """What `schemas.validate_release_package` says about this package's record."""
        return schemas.validate_release_package(self.to_dict())


def assemble_release_package(*, package_id: str, created_at: str,
                             freeze_request: OperatorFreezeRequest,
                             sealed: SealedRelease, evaluation: TrustedEvaluation,
                             version: NextVersion, transaction: t3.TransactionPlan,
                             rollback: RollbackPlan, era_row_draft: Mapping,
                             rebaseline_note: str,
                             commands: Sequence[OperatorCommand],
                             watch_window: WatchWindow,
                             cutover_request: CutoverRequest,
                             autopilot_baseline_path: str,
                             change_classes: Sequence[str],
                             diff_complexity: Mapping,
                             waivers: Sequence[t3.WaiverBinding] = (),
                             release_plan: Optional[Mapping] = None,
                             readiness_signal: Optional[Mapping] = None) -> ReleasePackage:
    """Assemble the package, deriving every finding and then the state.

    The cross-checks below are the point of this function. Each pair of inputs is
    individually well-formed and could still describe two different releases: a
    transaction for v9 beside a version record for v10; a watch window belonging to
    another package; a T3 verdict for a different seal. None of those is detectable
    inside any one object, and all of them produce a package that reads correctly.

    `release_plan` is DERIVED from the sealed bundle when there is one, because the
    plan that matters is the plan T3 graded. A supplied plan is accepted only when
    the seal did not close, and that fact is itself recorded.
    """
    for label, value, klass in (
            ("freeze_request", freeze_request, OperatorFreezeRequest),
            ("sealed", sealed, SealedRelease),
            ("evaluation", evaluation, TrustedEvaluation),
            ("version", version, NextVersion),
            ("transaction", transaction, t3.TransactionPlan),
            ("rollback", rollback, RollbackPlan),
            ("watch_window", watch_window, WatchWindow),
            ("cutover_request", cutover_request, CutoverRequest)):
        if not isinstance(value, klass):
            raise PackagerInputError(
                f"assemble_release_package: {label} must be a {klass.__name__}")
    _text(package_id, "assemble_release_package: package_id")
    _timestamp(created_at, "assemble_release_package: created_at")
    _mapping(era_row_draft, "assemble_release_package: era_row_draft", non_empty=True)
    _mapping(diff_complexity, "assemble_release_package: diff_complexity", non_empty=True)
    bindings = _typed_tuple(waivers, "assemble_release_package: waivers",
                            t3.WaiverBinding)

    findings: list = []

    def note(code: str, detail: str, outcome: str = schemas.FAIL, gating: bool = True):
        findings.append(PackageFinding(code=code, detail=detail, outcome=outcome,
                                       gating=gating))

    # -- the seam: did this evaluator grade THIS material? --------------------
    if evaluation.check.outcome != schemas.PASS:
        note("EVALUATION_SEAM_UNSOUND", "; ".join(evaluation.check.reasons)
             or "the trusted-evaluator seam did not verify",
             outcome=evaluation.check.outcome)
    if evaluation.result.verdict == "FAIL":
        computation = evaluation.result.verdict_computation
        note("T3_VERDICT_FAIL",
             f"T3 returned FAIL — failed cells {list(computation.failed_cells)}, "
             f"unevaluable cells {list(computation.unevaluable_cells)}, blocking reasons "
             f"{list(computation.blocking_reasons)}")
    elif evaluation.result.verdict not in schemas.T3_VERDICTS:
        note("T3_VERDICT_UNKNOWN", f"verdict {evaluation.result.verdict!r} is not one of "
             f"{sorted(schemas.T3_VERDICTS)}")
    if evaluation.result.bundle is None:
        note("RELEASE_BUNDLE_UNSEALED",
             "the T3 seal did not close, so there is no bundle whose evidence anyone can "
             "rehash (§10.2 phase 9)")

    # -- identity: one candidate, one tree, one campaign ----------------------
    if freeze_request.source_tree != sealed.candidate.source_tree:
        note("FREEZE_REQUEST_TREE_MISMATCH",
             f"the request names {freeze_request.source_tree!r} and the seal names "
             f"{sealed.candidate.source_tree!r}; freezes are per source tree (§1.5)")
    if evaluation.result.bundle is not None:
        # "A T3 verdict for a different seal" is named in this function's own
        # docstring as one of the things it exists to catch, and it was the one
        # cross-check nobody wrote: `sealed` and `evaluation` arrive as separate
        # arguments, each internally consistent, and the package carried candidate A's
        # identity beside candidate B's PASS. The bundle is the only place the graded
        # seal is recorded, so it is what the package's own seal is compared against.
        graded_seal = evaluation.result.bundle.payload.get("sealed_candidate")
        graded_seal = graded_seal if isinstance(graded_seal, Mapping) else {}
        for field_name, ours in (("candidate_id", sealed.candidate.candidate_id),
                                 ("seal_sha256", sealed.candidate.seal_sha256)):
            theirs = graded_seal.get(field_name)
            if theirs != ours:
                note("SEALED_CANDIDATE_NOT_THE_GRADED_ONE",
                     f"the package seals {field_name} {ours!r} and the sealed bundle was "
                     f"graded over {theirs!r}. The verdict in this package belongs to a "
                     "different candidate, and nothing inside either object could say so "
                     "(invariant 2: release evidence is produced by the same full "
                     "candidate that is frozen)")
        graded_campaign = evaluation.result.bundle.payload.get("campaign_id")
        if graded_campaign != freeze_request.campaign_id:
            note("CAMPAIGN_MISMATCH",
                 f"the freeze request names campaign {freeze_request.campaign_id!r} and "
                 f"the sealed bundle was graded under {graded_campaign!r}. A package "
                 "labelled one campaign over another campaign's evidence puts two "
                 "calibrations, two anchors and two control sets under one heading "
                 "(P-AK-SEARCH-1 denial 4)")
    if version.incumbent_branch != rollback.rollback_branch:
        note("ROLLBACK_NOT_THE_INCUMBENT",
             f"the rollback anchor is {rollback.rollback_branch!r} but the incumbent is "
             f"{version.incumbent_branch!r}; a rollback to something other than what "
             "production is running now is not a rollback")
    if transaction.next_branch != version.next_branch or \
            transaction.next_version_number != version.next_version_number:
        note("TRANSACTION_VERSION_MISMATCH",
             f"the transaction targets {transaction.next_branch!r}/"
             f"{transaction.next_version_number} while the computed successor is "
             f"{version.next_branch!r}/{version.next_version_number}")
    if transaction.rollback_branch != rollback.rollback_branch or \
            transaction.rollback_head != rollback.rollback_head:
        note("TRANSACTION_ROLLBACK_MISMATCH",
             "the transaction's rollback anchor is not the rollback plan's anchor")
    if watch_window.package_id != package_id:
        note("WATCH_WINDOW_FOREIGN",
             f"the watch window belongs to package {watch_window.package_id!r}")
    if cutover_request.package_id != package_id:
        note("CUTOVER_REQUEST_FOREIGN",
             f"the cutover request belongs to package {cutover_request.package_id!r}")
    if cutover_request.next_branch != transaction.next_branch:
        note("CUTOVER_REQUEST_VERSION_MISMATCH",
             f"the cutover request names {cutover_request.next_branch!r}, the transaction "
             f"{transaction.next_branch!r}")

    # -- rollback and archive (§10.5) ----------------------------------------
    if rollback.archive_check.outcome != schemas.PASS:
        note("ROLLBACK_ARCHIVE_UNVERIFIED", "; ".join(rollback.archive_check.reasons)
             or "the incumbent archive did not verify",
             outcome=rollback.archive_check.outcome)
    if rollback.anchor_live is None:
        note("ROLLBACK_ANCHOR_LIVENESS_UNSTATED",
             "the rollback plan does not say whether the anchor is live. §11.5 requires it "
             "to stay live and verified for the whole watch window, and the field used to "
             "default to True — so a plan nobody checked answered the requirement by "
             "construction. Unstated is COULD_NOT_CHECK, not yes",
             outcome=schemas.COULD_NOT_CHECK)
    elif not rollback.anchor_live:
        note("ROLLBACK_ANCHOR_NOT_LIVE",
             "§11.5 requires the rollback anchor to stay live and verified for the whole "
             "watch window; it is recorded as not live")
    missing_backends = sorted(set(sealed.backends) - set(rollback.backends))
    if missing_backends:
        note("ROLLBACK_BACKEND_UNCOVERED",
             f"no archived incumbent binary for {missing_backends}, which the seal builds")

    # -- era rows and the rebaseline note (§1.3 items 2 and 3) ----------------
    # TRACED from the rows, never read off `kinds_present`. The declared key is a
    # summary the caller writes, and this function accepts a hand-built mapping: an
    # era block carrying `kinds_present: [kernel, autopilot_speed, umbrella]` and
    # `rows: []` reached RELEASE_PACKAGE_READY with no era row in it at all. That is
    # the same declared-vs-traced hole `OperatorCommand.human_only` closes one layer
    # down, in the place where the missing thing is a human-only write.
    rows = era_row_draft.get("rows")
    rows = tuple(rows) if isinstance(rows, (list, tuple)) else ()
    kinds_present = {row.get("kind") for row in rows if isinstance(row, Mapping)}
    declared_kinds = set(era_row_draft.get("kinds_present") or ())
    overdeclared = sorted(declared_kinds - kinds_present)
    if overdeclared:
        note("ERA_ROW_KINDS_DECLARED_NOT_DRAFTED",
             f"the era block declares {overdeclared} in `kinds_present` and carries no "
             "row of that kind. The operator writes the ROWS; a summary key is not a "
             "draft, and a freeze whose era row is unwritten produces evidence nobody "
             "can interpret (§1.3 item 2, MEASUREMENT.md:233)")
    missing_kinds = [kind for kind in ERA_ROW_KINDS if kind not in kinds_present]
    if missing_kinds:
        note("ERA_ROW_KIND_MISSING",
             f"the draft era block has no {missing_kinds} row. The v8 cutover wrote three "
             "(E8-cpu-kernel, E8-autopilot-speed, E8); a freeze whose era row is unwritten "
             "produces evidence nobody can interpret (§1.3 item 2, MEASUREMENT.md:233)")
    if era_row_draft.get("draft") is not True or \
            era_row_draft.get("written_by") != EXECUTED_BY:
        note("ERA_ROW_NOT_A_DRAFT",
             "the era block does not declare itself a draft written by the operator")
    if not isinstance(diff_complexity.get("touches_shared_core"), bool):
        note("DIFF_COMPLEXITY_SHARED_CORE_UNSTATED",
             "`diff_complexity` does not state `touches_shared_core` as a bool, so the "
             "§10.6 shared-core question was answered by a missing key. An unassessed "
             "diff has not cleared a blast-radius ceiling; the package is INCOMPLETE "
             "rather than clear", outcome=schemas.COULD_NOT_CHECK)
    if autopilot_baseline_path not in rebaseline_note:
        note("REBASELINE_NOTE_NAMES_NO_BASELINE",
             f"the rebaseline note never names {autopilot_baseline_path!r}, so it does not "
             "say which human-only file the operator must reseed")

    # -- linkage, derived from T3's own phase ---------------------------------
    linkage = derive_linkage_summary(evaluation)
    if linkage.status != schemas.PASS:
        note("LINKAGE_NOT_PROVEN", "; ".join(linkage.reasons)
             or f"linkage status {linkage.status}", outcome=linkage.status)

    # -- the operator command sequence ---------------------------------------
    review = validate_command_sequence(
        commands, transaction=transaction, rollback=rollback, era_row=era_row_draft,
        autopilot_baseline_path=autopilot_baseline_path)
    for detail in review.findings:
        note(detail.split(":", 1)[0], detail)
    if not review.validated_commands:
        note("OPERATOR_COMMAND_SEQUENCE_EMPTY",
             "no command survived pre-validation, so the package hands the operator "
             "nothing executable (MEASUREMENT.md:138-145)")

    # -- waivers (§10.4): pinned here, verified there -------------------------
    verified_ids = {v.waiver_id for v in evaluation.result.products.waiver_verifications
                    if v.verified}
    used_ids = {w.get("waiver_id") for w in evaluation.result.verdict_computation.waived_cells}
    # A binding map is LAST-WINS, and the losing binding vanishes from every check
    # below it: two `WaiverBinding`s with one id, the machine-attributed one first,
    # and `WAIVER_SELF_GRANTED` was never raised because the scan only ever saw the
    # survivor. The package's waiver findings were order-dependent. Duplicates are
    # therefore a finding in their own right — two documents claiming to be one
    # waiver is a contradiction about which bytes T3 pinned — and every scan below
    # that can be defeated by dedup now walks `bindings`, not the map.
    duplicate_ids = sorted({b.waiver_id for b in bindings
                            if [x.waiver_id for x in bindings].count(b.waiver_id) > 1})
    if duplicate_ids:
        note("WAIVER_BINDING_DUPLICATE",
             f"waiver id(s) {duplicate_ids} are pinned more than once with different "
             "documents or coverage. A map keyed on the id keeps one and silently drops "
             "the other, so the package would report the surviving document's hash, "
             "coverage and attribution for a waiver T3 may have verified from the other "
             "(§10.4)")
    bound = {b.waiver_id: b for b in bindings}
    active = tuple(
        {"waiver_id": waiver_id, "sha256": bound[waiver_id].pinned_sha256,
         "document_path": bound[waiver_id].document_path,
         "covers_cell_ids": list(bound[waiver_id].covers_cell_ids),
         # WAS THE DOCUMENT READ, and what did it hash to when it was? The package is
         # what a human executes, so it must state which of its waivers is a file
         # somebody opened and which is a quotation. `t3.waiver_read_violations` and
         # not the `was_read` property: a property is overridable by a subclass, and
         # the package is a durable record — the one place a lie survives the run.
         "read": not t3.waiver_read_violations(bound[waiver_id]),
         "observed_sha256": bound[waiver_id].observed_sha256}
        for waiver_id in sorted(used_ids & set(bound)))
    for waiver_id in sorted(used_ids - set(bound)):
        note("WAIVER_SUPPRESSED_BUT_UNPINNED",
             f"waiver {waiver_id!r} suppressed a cell in the T3 run but is not pinned into "
             "this package, so the package cannot say which document did it (§10.4)")
    for waiver_id in sorted(set(bound) - verified_ids):
        note("WAIVER_PINNED_BUT_UNVERIFIED",
             f"waiver {waiver_id!r} is pinned into the package and T3 did not verify it; "
             "an unverified waiver suppresses nothing and must not read as active",
             outcome=schemas.COULD_NOT_CHECK)
    # "T3 refused it" and "nobody read it" are two different states and they collapsed
    # into one finding above, so the package could not tell an operator which had
    # happened. A quotation is not a weaker attestation, it is a DIFFERENT object:
    # `t3.WaiverBinding` carries a document, a path and a digest that are three
    # independent assertions by the party being gated, and only
    # `t3.waiver_binding_from_path` produces one whose bytes were read.
    for binding in sorted(bindings, key=lambda b: (b.waiver_id, b.pinned_sha256)):
        if not t3.waiver_read_violations(binding):
            continue
        note("WAIVER_PINNED_UNREAD",
             f"waiver {binding.waiver_id!r} ({binding.pinned_sha256[:12]}) is pinned "
             f"from {binding.document_path!r} as a QUOTATION: nothing read that file, "
             "so its document, its path and its digest are all assertions by the party "
             "being gated. §10.4's evaluator verifies a waiver's hash, and a hash over "
             "bytes the caller handed over verifies nothing "
             "(t3.waiver_binding_from_path)",
             outcome=schemas.COULD_NOT_CHECK)
    # §10.4 waivers are HUMAN-only. The vocabulary that decides whether an identity
    # is a machine now lives in `schemas.py`, so `t3.verify_waiver` refuses a
    # self-granted waiver at the gate and this stays as the OUTER layer: the package
    # is what a human executes, and a self-granted exception must not reach one even
    # if a caller reached the packager with a verdict T3 never produced.
    for binding in sorted(bindings, key=lambda b: (b.waiver_id, b.pinned_sha256)):
        for field_name, attributed, found in schemas.machine_attributions(
                binding.document):
            note("WAIVER_SELF_GRANTED",
                 f"waiver {binding.waiver_id!r} ({binding.pinned_sha256[:12]}) names "
                 f"{attributed!r} in {field_name!r}, a "
                 f"machine actor ({', '.join(found)}). §10.4 waivers are "
                 "human-only (MEASUREMENT.md:140-142); a waiver the loop "
                 "attributed to itself is the loop excusing its own failing cell, "
                 f"and this one suppresses "
                 f"{list(binding.covers_cell_ids)}.")
    if evaluation.result.verdict == "PASS_WITH_WAIVER" and not active:
        note("PASS_WITH_WAIVER_PINS_NOTHING",
             "the verdict is PASS_WITH_WAIVER and the package pins no waiver")

    # -- the release plan: the plan T3 graded, not one supplied beside it -----
    if evaluation.result.bundle is not None:
        graded_plan = evaluation.result.bundle.payload.get("release_plan")
        if not isinstance(graded_plan, Mapping) or not graded_plan:
            note("RELEASE_PLAN_ABSENT_FROM_BUNDLE",
                 "the sealed bundle carries no release plan, so the package cannot state "
                 "the scope that was graded")
            plan_block = dict(release_plan or {"unavailable": True})
        else:
            plan_block = dict(graded_plan)
            if release_plan is not None and \
                    schemas.content_hash(dict(release_plan)) != \
                    schemas.content_hash(plan_block):
                note("RELEASE_PLAN_NOT_THE_GRADED_PLAN",
                     "the supplied release plan is not the plan the sealed bundle graded; "
                     "scope is mechanically derived and the graded plan is the one that "
                     "counts (invariant 18)")
    else:
        plan_block = dict(release_plan or {"unavailable": True})
        note("RELEASE_PLAN_UNSEALED",
             "there is no sealed bundle, so the release plan in this package is not the "
             "one a bundle digest binds", outcome=schemas.COULD_NOT_CHECK, gating=False)

    state = _derive_package_state(findings)
    decision = build_decision_package(
        package_id=package_id, state=state, freeze_request=freeze_request,
        evaluation=evaluation, version=version, transaction=transaction,
        rollback=rollback, watch_window=watch_window, cutover_request=cutover_request,
        findings=tuple(findings))
    return ReleasePackage(
        package_id=package_id, campaign_id=freeze_request.campaign_id,
        source_tree=sealed.candidate.source_tree, created_at=created_at,
        freeze_request=freeze_request, sealed=sealed, evaluation=evaluation,
        version=version, transaction=transaction, rollback=rollback,
        era_row_draft=dict(era_row_draft), rebaseline_note=rebaseline_note,
        linkage=linkage, command_review=review, watch_window=watch_window,
        cutover_request=cutover_request, decision_package=decision,
        release_plan=plan_block, active_waivers=active,
        waiver_bindings=tuple(b.to_dict() for b in bindings),
        change_classes=tuple(change_classes or ()), diff_complexity=dict(diff_complexity),
        findings=tuple(findings), state=state, readiness_signal=readiness_signal)


def render_first_page(package: ReleasePackage) -> str:
    """The operator's first page: the §10.6 notice, then the four-part decision."""
    if not isinstance(package, ReleasePackage):
        raise PackagerInputError("render_first_page: package must be a ReleasePackage")
    title = (f"Release package {package.package_id} — {package.state} — "
             f"{package.version.incumbent_branch} → {package.version.next_branch}")
    return render_decision_package(package.decision_package, title=title,
                                   first_page_notice=package.first_page_notice)


# =============================================================================
# Self-audits — the cardinal rule, proved from this module's own source
# =============================================================================

#: Same shape as `t3._FORBIDDEN_IMPORTS`. `pathlib` is deliberately absent from
#: both: reading is not writing, and the subject of this audit is the ability to
#: MUTATE the host. `time` IS here, and it is doing double duty — it is a process
#: verb and it is a clock (see `audit_no_clock_or_self_trigger`).
_FORBIDDEN_IMPORTS = frozenset({
    "os", "subprocess", "shutil", "signal", "socket", "ctypes", "multiprocessing",
    "tempfile", "sqlite3", "urllib", "http", "requests", "pty", "fcntl", "resource",
    "shlex", "asyncio", "threading", "time", "io", "sched", "importlib", "pickle",
    "runpy", "posix",
})
_FORBIDDEN_CALL_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__",
                                   "input"})
#: Attribute calls. `pathlib` is allowed for READING, so every mutating pathlib
#: method has to be named or the allowance is a hole: `Path(p).open("w")` is
#: `open(p, "w")` written as an attribute, `Path(new).replace(link)` IS the
#: move-a-stable-kernel-symlink primitive, and `hardlink_to` is `symlink_to` under
#: another name. All four were reachable in a sibling module until they were named.
_FORBIDDEN_CALL_ATTRS = frozenset({
    "write_text", "write_bytes", "write", "writelines", "truncate", "flush",
    "open", "mkdir", "makedirs", "unlink", "rmtree", "rmdir", "remove", "rename",
    "renames", "replace", "chmod", "chown", "lchmod", "mknod", "symlink", "symlink_to",
    "link", "link_to", "hardlink_to", "touch", "copy", "copy2", "copyfile", "copytree",
    "move", "system", "popen", "run", "call", "check_call", "check_output", "Popen",
    "spawn", "fork", "kill", "killpg", "terminate", "send_signal", "sleep",
    "import_module", "dump", "startfile", "execv", "execve", "execvp", "posix_spawn",
})
#: Clock reads. A module that can ask what time it is can decide that it is time to
#: freeze; a module that cannot must be told, by an operator, in a request it did
#: not build. `fromisoformat`, `combine` and `strftime` are NOT here — parsing and
#: rendering a timestamp somebody handed you is not reading a clock.
_FORBIDDEN_CLOCK_ATTRS = frozenset({
    "now", "utcnow", "today", "monotonic", "perf_counter", "time_ns",
    "fromtimestamp", "utcfromtimestamp", "process_time",
})

#: Receivers that can actually HAND OUT a clock. `.now` is the field name on
#: `WatchWindowProgress`, so `progress.now` and `self.now` are ordinary data reads
#: and flagging them would forbid this module's own compliant idiom — the defect
#: `_timestamp` already exists to avoid on the write side. A clock ALIAS is
#: `datetime.now` bound without calling it, and what distinguishes it is the
#: receiver, not the attribute name.
_CLOCK_BEARING_RECEIVERS = frozenset({"datetime", "date", "time", "clock"})


def _call_func_ids(tree: ast.AST) -> frozenset:
    """Identities of the nodes standing in CALL position anywhere in `tree`."""
    return frozenset(id(node.func) for node in ast.walk(tree)
                     if isinstance(node, ast.Call))


def _unauditable_dispatch(tree: ast.AST) -> list:
    """Calls whose callee is neither a name nor an attribute.

    `builtins.__dict__["open"](p, "w")` IS `open(p, "w")` written as a subscript,
    and `getattr(m, verb)(...)` is the same move through a call. A name-based
    denylist cannot read either, so the construct itself is the finding: an audit
    that silently returns PASS over dispatch it cannot follow is asserting a
    property it never checked.
    """
    findings: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and not isinstance(node.func,
                                                         (ast.Name, ast.Attribute)):
            findings.append(
                f"line {node.lineno}: calls a {type(node.func).__name__} expression. "
                "A callee this audit cannot name is a callee it cannot deny — "
                "`builtins.__dict__['open'](p, 'w')` is `open(p, 'w')` with different "
                "punctuation.")
    return findings


def _receiver_roots(node: ast.AST) -> set:
    """Every name in an attribute chain's receiver, e.g. `a.b.c` ⇒ {'a', 'b'}."""
    roots: set = set()
    while isinstance(node, ast.Attribute):
        roots.add(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        roots.add(node.id)
    return roots


def _literal_loop_bindings(tree: ast.AST) -> dict:
    """`{name: {constant, …} | {None}}` for names bound by iterating a literal.

    `getattr(x, name)` inside `for name in ("overlay_present", "tree_clean", …)` is
    a spelled-out attribute read: every value the name can take is in the source, so
    the denylist can read it. `getattr(m, "sys" + "tem")` is not. Resolving the loop
    is what lets the audit refuse the second WITHOUT forbidding the first — this
    module uses the first idiom twice, and a guard that bans its own compliant
    spelling is a guard that gets exempted rather than obeyed.

    A name whose binding cannot be resolved to constants maps to `{None}`, which is
    the unauditable answer and never the safe one.
    """
    out: dict = {}
    for node in ast.walk(tree):
        pairs: list = []
        if isinstance(node, ast.For):
            pairs = [(node.target, node.iter)]
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp,
                               ast.DictComp)):
            pairs = [(gen.target, gen.iter) for gen in node.generators]
        for target, iterable in pairs:
            if not isinstance(target, ast.Name):
                continue
            values: set = {None}
            if isinstance(iterable, (ast.Tuple, ast.List, ast.Set)) and iterable.elts:
                literals = {element.value for element in iterable.elts
                            if isinstance(element, ast.Constant)
                            and isinstance(element.value, str)}
                if len(literals) == len(iterable.elts):
                    values = literals
            out.setdefault(target.id, set()).update(values)
    return out


def _assigned_values(tree: ast.AST) -> list:
    """Every expression bound to a name by an assignment, with its line.

    Aliasing is how a denylist keyed on the CALL site is walked around: the call
    `sink(x)` says nothing, and `sink = Path(p).write_text` two lines above is where
    the capability was actually acquired. This yields the right-hand sides so the
    audits can look at what was bound rather than only at what was called.
    """
    out: list = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)) or \
                (isinstance(node, ast.AnnAssign) and node.value is not None):
            out.append(node.value)
    return out


def _module_source(source: Optional[str]) -> tuple:
    """`(source, check_or_None)` — an unreadable module is not an audited one."""
    if source is not None:
        return (source, None)
    try:
        return (Path(__file__).read_text(encoding="utf-8"), None)
    except OSError as exc:
        return (None, _cnc(f"could not read {__file__}: {exc}"))


def _parse_audited(source: Optional[str]) -> tuple:
    """`(tree, check_or_None)`, binding the audit to THIS module.

    An empty string parses, contains no forbidden construct, and would otherwise
    return PASS — the check certifying its own absence. So a PASS is issued only for
    source that defines `MODULE_ID = "autokernel.release.packager/v1"`. A FAIL is
    still returned unbound: a forbidden construct is a finding about the text
    whoever wrote it.
    """
    text, refusal = _module_source(source)
    if refusal is not None:
        return (None, refusal)
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return (None, _cnc(f"could not parse module: {exc}"))
    return (tree, None)


def _defines_this_module(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODULE_ID" and \
                        isinstance(node.value, ast.Constant) and \
                        node.value.value == MODULE_ID:
                    return True
    return False


def audit_no_write_or_process_paths(source: Optional[str] = None) -> schemas.Check:
    """Prove from the AST that the packager cannot write, spawn, or signal.

    This is the cardinal rule with a parser attached. `test_packager.py` asserts
    PASS, which turns invariant 5 into a regression barrier rather than an
    intention: a future edit that reaches for `subprocess` fails a test in the same
    commit.
    """
    tree, refusal = _parse_audited(source)
    if refusal is not None:
        return refusal

    findings: list = _unauditable_dispatch(tree)
    called = _call_func_ids(tree)
    loop_bindings = _literal_loop_bindings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    findings.append(f"line {node.lineno}: imports {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root in _FORBIDDEN_IMPORTS:
                findings.append(f"line {node.lineno}: imports from {node.module!r}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
                findings.append(f"line {node.lineno}: calls {func.id}()")
            elif isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CALL_ATTRS:
                findings.append(f"line {node.lineno}: calls .{func.attr}()")
            elif isinstance(func, ast.Name) and func.id in ("getattr", "setattr"):
                # A DYNAMIC attribute name is the same route with the evidence
                # removed: `getattr(m, "sys" + "tem")` reads no differently to this
                # audit than `getattr(m, "to_dict")`, so the unreadable form is the
                # finding rather than a silent pass.
                named = node.args[1] if len(node.args) >= 2 else None
                reachable: set = set()
                if named is None:
                    reachable = set()
                elif isinstance(named, ast.Constant) and isinstance(named.value, str):
                    reachable = {named.value}
                elif isinstance(named, ast.Name):
                    reachable = set(loop_bindings.get(named.id) or {None})
                else:
                    reachable = {None}
                if None in reachable:
                    findings.append(
                        f"line {node.lineno}: {func.id}() is given a computed attribute "
                        "name this audit cannot resolve to constants, so it cannot say "
                        "which attribute is reached. Name it, or iterate a literal.")
                for candidate_attr in sorted(a for a in reachable if a is not None):
                    if candidate_attr in _FORBIDDEN_CALL_ATTRS or \
                            candidate_attr in _FORBIDDEN_CALL_NAMES:
                        findings.append(
                            f"line {node.lineno}: reaches {candidate_attr!r} through "
                            f"{func.id}(), which routes around the attribute denylist")
        elif isinstance(node, ast.Name) and id(node) not in called and \
                isinstance(node.ctx, ast.Load) and node.id in _FORBIDDEN_CALL_NAMES:
            # BINDING a write primitive is acquiring it. `w = open` followed by
            # `w(path, "w")` puts nothing forbidden in call position, and the audit
            # returned PASS over exactly that until the reference itself counted.
            findings.append(
                f"line {node.lineno}: binds {node.id} without calling it; a write verb "
                "bound to a name is a write verb, and the call site that uses it is "
                "unreadable to a call-position denylist")
        elif isinstance(node, ast.Attribute) and id(node) not in called and \
                node.attr in _FORBIDDEN_CALL_ATTRS:
            findings.append(
                f"line {node.lineno}: references .{node.attr} without calling it; a "
                "mutating method bound to a name is the same capability one indirection "
                "away")
    if findings:
        return _fail(*findings)
    if not _defines_this_module(tree):
        return _cnc(
            f"the audited source does not define MODULE_ID == {MODULE_ID!r}, so a clean "
            "result says nothing about this module — an empty string is also clean")
    return schemas.Check(schemas.PASS)


def audit_no_clock_or_self_trigger(source: Optional[str] = None) -> schemas.Check:
    """Prove the packager has no clock and cannot mint its own freeze request (AK7).

    Two properties, one audit, because they are the same guarantee from two sides:

      * **no clock** — every timestamp in this module is a parameter. A module that
        reads the wall clock can conclude that a quarterly cadence has elapsed and
        that it is therefore time to freeze; AK-D25 makes cadence an operator
        POLICY, not a loop parameter, and this is what keeps it one.
      * **no `OperatorFreezeRequest(...)` construction anywhere in this module** —
        the request is the operator's artifact. If this module could build one it
        could answer its own, and every downstream refusal would be guarding a door
        with the key taped to it.
    """
    tree, refusal = _parse_audited(source)
    if refusal is not None:
        return refusal

    findings: list = _unauditable_dispatch(tree)
    called = _call_func_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CLOCK_ATTRS:
                findings.append(
                    f"line {node.lineno}: reads a clock via .{func.attr}(); every "
                    "timestamp here must be an input, or the module can decide it is "
                    "time to freeze")
            if isinstance(func, ast.Name) and func.id == "OperatorFreezeRequest":
                findings.append(
                    f"line {node.lineno}: constructs OperatorFreezeRequest. The freeze "
                    "request is the operator's artifact; a module that can mint one can "
                    "trigger itself (AK7).")
            if isinstance(func, ast.Attribute) and func.attr == "OperatorFreezeRequest":
                findings.append(
                    f"line {node.lineno}: constructs OperatorFreezeRequest through an "
                    "attribute, which is the same mint by another route")
        elif isinstance(node, ast.Attribute) and id(node) not in called and \
                node.attr in _FORBIDDEN_CLOCK_ATTRS and \
                (_receiver_roots(node.value) & _CLOCK_BEARING_RECEIVERS):
            # `clock = datetime.now` then `clock()`: nothing forbidden is ever in
            # call position and the audit passed. Receiver-qualified, so this
            # module's own `progress.now` / `self.now` field reads — which are data,
            # not clocks — are untouched.
            findings.append(
                f"line {node.lineno}: binds .{node.attr} off a clock-bearing receiver "
                "without calling it. A clock held under another name is still a clock, "
                "and the call that uses it reads as an ordinary local.")
    for value in _assigned_values(tree):
        name = value.attr if isinstance(value, ast.Attribute) else \
            (value.id if isinstance(value, ast.Name) else None)
        if name == "OperatorFreezeRequest":
            findings.append(
                f"line {value.lineno}: binds OperatorFreezeRequest to another name. The "
                "mint refusal is about the CAPABILITY, not the spelling: an alias "
                "called later is the same self-trigger (AK7).")
    if findings:
        return _fail(*findings)
    if not _defines_this_module(tree):
        return _cnc(
            f"the audited source does not define MODULE_ID == {MODULE_ID!r}; a clean "
            "result over foreign source proves nothing about this module")
    return schemas.Check(schemas.PASS)


def audit_verdict_is_delegated(source: Optional[str] = None) -> schemas.Check:
    """Prove the packager never computes a T3 verdict itself (invariant 4).

    The verdict arrives through `run_release_evaluation`'s injected
    `ReleaseTierEvaluator`. Calling `t3.run_t3`, `t3.compute_verdict` or a phase
    runner directly would work — and would collapse two authority domains into one
    process that both produces and grades, which is exactly what §1.4 separates.
    """
    tree, refusal = _parse_audited(source)
    if refusal is not None:
        return refusal

    denied = {"run_t3", "compute_verdict", "phase_seal", "phase_identity_preflight",
              "phase_build_linkage", "phase_backend_correctness",
              "phase_performance_matrix", "phase_quality", "phase_stability",
              "phase_capacity_utility", "phase_transaction_dry_run"}
    findings: list = _unauditable_dispatch(tree)
    called = _call_func_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and id(node) not in called:
            # Same aliasing route as the other two audits: `grade = t3.run_t3`
            # followed by `grade(request)` puts nothing denied in call position.
            referenced = node.attr if isinstance(node, ast.Attribute) else node.id
            if referenced in denied:
                findings.append(
                    f"line {node.lineno}: binds {referenced} without calling it; the "
                    "gate reached through an alias is the gate, and invariant 4 is "
                    "about the authority domain rather than the spelling")
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else \
            (func.id if isinstance(func, ast.Name) else None)
        if name in denied:
            findings.append(
                f"line {node.lineno}: calls {name}() directly. The release verdict is "
                "produced by the trusted evaluator through the ReleaseTierEvaluator seam; "
                "a packager that ran the gate would be grading material it assembled "
                "(invariant 4, §1.4).")
    if findings:
        return _fail(*findings)
    if not _defines_this_module(tree):
        return _cnc(
            f"the audited source does not define MODULE_ID == {MODULE_ID!r}; a clean "
            "result over foreign source proves nothing about this module")
    return schemas.Check(schemas.PASS)


def audit_refusal_doors_raise_unconditionally(source: Optional[str] = None) -> schemas.Check:
    """Prove every §11.2 "may not" is a function whose whole body is a `raise`.

    Walks `REFUSED_CAPABILITIES` — the SSOT — rather than a second list here, so a
    capability added to the map without a door fails, and a door softened into
    something that can return fails. "Unconditionally" is the load-bearing word: a
    door with an `if` in it is a door with a key, and the reviewer who added the
    condition will have had a good reason at the time.
    """
    tree, refusal = _parse_audited(source)
    if refusal is not None:
        return refusal
    # Binding is checked FIRST here, unlike the other three. Their FAIL is a finding
    # about whatever text they were handed — a `subprocess` import is forbidden
    # wherever it appears. This audit's subject is THIS module's own structure, so
    # "no function named execute_freeze" in foreign source is a statement about the
    # foreign source, not a violation, and reporting it as FAIL would make the audit
    # loud in the wrong place while still telling you nothing about this module.
    if not _defines_this_module(tree):
        return _cnc(
            f"the audited source does not define MODULE_ID == {MODULE_ID!r}, so its "
            "missing doors say nothing about this module's refusals")

    # MODULE LEVEL only (`tree.body`, not `ast.walk`). A door defined inside another
    # function satisfies a walk and binds no module attribute at all: `packager.
    # execute_freeze` would not exist, and the audit would report the door as present
    # while the name a caller reaches for is missing.
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    doors = set(REFUSED_CAPABILITIES.values())
    findings: list = []
    # A door is the NAME, not the `def`. `execute_freeze = lambda *a, **k: None`
    # after a compliant definition leaves the AST full of correct-looking raises and
    # the module exporting a function that returns None. Any rebinding of a door name
    # — assignment, `import … as`, loop variable, `with … as` — is the finding.
    for node in ast.walk(tree):
        rebound: list = []
        if isinstance(node, ast.Assign):
            rebound = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            rebound = [node.target] if isinstance(node.target, ast.Name) else []
        elif isinstance(node, ast.For):
            rebound = [node.target] if isinstance(node.target, ast.Name) else []
        elif isinstance(node, ast.withitem):
            rebound = [node.optional_vars] if isinstance(node.optional_vars, ast.Name) \
                else []
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            rebound = [ast.Name(id=(alias.asname or alias.name.split(".")[0]),
                                 lineno=node.lineno)
                       for alias in node.names]
        for target in rebound:
            if target.id in doors:
                findings.append(
                    f"line {getattr(target, 'lineno', node.lineno)}: rebinds "
                    f"{target.id}(), the door refusing "
                    f"{sorted(c for c, n in REFUSED_CAPABILITIES.items() if n == target.id)}. "
                    "A refusal that a later statement can replace is a refusal for as "
                    "long as nobody replaces it.")
    for capability, name in sorted(REFUSED_CAPABILITIES.items()):
        node = functions.get(name)
        if node is None:
            findings.append(
                f"{capability}: no function named {name}() exists to refuse it")
            continue
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and \
                isinstance(body[0].value, ast.Constant) and \
                isinstance(body[0].value.value, str):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Raise):
            findings.append(
                f"{capability}: {name}() does not consist of a single unconditional "
                f"raise (its body is {[type(s).__name__ for s in body]}). A refusal with "
                "a branch in it is a capability with a precondition.")
    if findings:
        return _fail(*findings)
    return schemas.Check(schemas.PASS)
