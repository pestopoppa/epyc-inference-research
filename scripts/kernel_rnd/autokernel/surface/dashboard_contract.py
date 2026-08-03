#!/usr/bin/env python3
"""dashboard_contract.py — the AK6 `/kernel` contract v2 PRODUCER.

WHAT THIS MODULE IS FOR
-----------------------
`schemas.validate_kernel_dashboard_v2` says what a valid operator-surface document
looks like. Nothing produced one. The hub's `/kernel` panel has been reading
`KERNEL_DASHBOARD_JSON` — a gitignored scratch path that does not exist — and
rendering clean, which is the absence-tolerance scar in its purest form:

> Today's `/kernel` page is **absence-tolerant over a missing directory** — it
> renders clean when its producer is dead, which is the exact shape of AutoPilot
> dying at trial 1302 and staying dead ~23 HOURS with every dashboard green.

This module is that missing producer. It derives every field from the module that
OWNS the fact, writes exactly one file, and cannot make a dead loop look alive.

THREE DESIGN RULES, ALL STRUCTURAL
----------------------------------
1. **DERIVE, never restate.** Every number in the document is read off the object
   whose module computed it: `standing` off `ReadinessSignal.standing`, free bytes
   off `StorageState.free_bytes`, package state off `ReleasePackage.state`. A
   figure this module recomputed would be a second source of truth and would
   drift; `test_dashboard_contract.py` proves the copy by handing the producer a
   deliberately self-contradictory signal and asserting the OWNER's value comes
   out. The one thing this module does compute is the summary over its own
   sections, which nothing else owns — and even that is recomputed by the
   validator so it cannot be stamped independently of the evidence.

2. **Absence is a VALUE, not an omission.** A caller may not pass `None` for a
   section. It passes the real object or an explicit `Unreported(reason=...)`, and
   `Unreported` is what a dead producer looks like: the section still appears, with
   `status: "not_reported"` and the reason. `_require_input` refuses `None`
   precisely so a forgotten input cannot become a clean panel.

3. **Liveness is derived from the LOOP's records, never from this process.**
   `produced_at` is `schemas.dashboard_liveness_timestamp(...)`, the newest
   timestamp among the OBSERVED sections whose `as_of` comes from a journaled
   record. Live host readings — free disk, held device claims — are measured by
   this module itself and are excluded, because a surface process that is merely
   alive must not be able to manufacture freshness for a controller that is dead.
   A no-op re-export therefore cannot read as fresh, and a fully dead loop yields
   `produced_at: null`, which every consumer classifies as `missing`. That is the
   property `server.py` already reaches for ("from semantic run timestamps, not
   file mtime"); this module makes it derivable rather than hoped for.

THE WRITE
---------
This is the only module in `autokernel` outside the journal/storage/claim/controller
planes that writes anything, and it writes ONE file. The write is bounded three
ways, all of which refuse rather than warn:

  * the destination is checked against `packager.HUMAN_ONLY_TARGET_PATTERNS` — the
    SSOT for "production branch, stable kernel symlink, era registry, AutoPilot
    baseline, human-only path list" — BEFORE and AFTER symlink resolution, so a
    symlink into production is refused too;
  * the destination is refused if `storage.is_scratch_path` classifies it as
    scratch (durability, `MEASUREMENT.md:146-156`), if it lands in a frozen
    production tree (`storage.PRODUCTION_TREES`, invariant 3), or if it is inside
    any git working tree — a file rewritten on every export does not belong in a
    checkout several sessions share;
  * the payload must validate under `schemas.validate_kernel_dashboard_v2` before
    a byte is written, so a malformed document is never the thing on disk, and the
    bytes are written in full and re-measured before the rename, so a short write
    can never replace a good document with a truncated one.

It runs no process, builds nothing, benchmarks nothing, and holds no freeze or
cutover authority.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md`, phase AK6.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .. import schemas, storage
from ..controller import state_machine as sm
from ..release import packager, readiness as readiness_mod

__all__ = [
    # identity and configuration
    "MODULE_ID", "CONTRACT_SCHEMA", "DEFAULT_EXPORT_ROOT", "DEFAULT_EXPORT_FILENAME",
    "DEFAULT_EXPORT_PATH", "OBSERVATION_NOTICE",
    # errors
    "SurfaceError", "ContractInputError", "ExportDestinationRefused",
    "ContractInvalid",
    # inputs
    "Unreported", "ControllerObservation", "HeadroomObservation",
    "ClaimsObservation", "BlockingCondition", "BlockingObservation",
    "ContractInputs",
    # derivation
    "observe_controller", "derive_blocking_conditions", "build_contract",
    # export
    "ExportReceipt", "export_contract", "assert_exportable_destination",
]

#: Versioned: a contract produced by a different reducer is a different contract.
MODULE_ID = "autokernel.surface.dashboard_contract/v2"

CONTRACT_SCHEMA = schemas.SCHEMA_KERNEL_DASHBOARD_V2

# =============================================================================
# The durable output path
# =============================================================================
#
# WHY THIS PATH AND NOT THE OLD ONE. The hub's default was
# `/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_dashboard.json`. Three separate
# defects in one string: `/mnt/raid0/llm/tmp` is the FIRST entry of
# `storage.EPHEMERAL_ROOTS`, so the file was one `tmp` sweep from vanishing and
# the sweep leaves no event behind; the directory does not exist on this host at
# all; and it sits inside a build scratch tree owned by nobody.
#
# WHY NOT THE REPOSITORY. Durable does not mean tracked. The 2026-08-03 operator
# ruling gitignores heavy campaign output on purpose, and this file is a DERIVED
# VIEW of the journal — regenerable from records that are themselves the evidence
# of record — so carrying it in git would add a churning artifact that is never
# the citation for anything. It is also rewritten on every export, which is the
# worst possible shape for a tracked file in a repository several sessions share.
#
# WHY THIS ONE. `/mnt/raid0` is the 3.7 TB array that survives reboots and holds
# every other piece of AutoKernel's on-disk state; `/mnt/raid0/llm/autokernel/` is
# outside every checkout, so nothing here can ride into a parallel session's
# commit; the path is a fixed, well-known constant so the hub needs no environment
# variable to find it; and `assert_exportable_destination` proves it is neither
# scratch, nor a production tree, nor any human-only target — checked against the
# SSOT rather than asserted here in prose.
DEFAULT_EXPORT_ROOT = "/mnt/raid0/llm/autokernel/surface"
DEFAULT_EXPORT_FILENAME = "kernel_dashboard.json"
DEFAULT_EXPORT_PATH = os.path.join(DEFAULT_EXPORT_ROOT, DEFAULT_EXPORT_FILENAME)

OBSERVATION_NOTICE = (
    "Every figure here is an OBSERVATION (MEASUREMENT.md) — it never gates a "
    "keep/revert/deploy/promote decision, and AutoKernel holds no freeze or "
    "cutover authority. `produced_at` is derived from the loop's own journaled "
    "record timestamps, never from this file's mtime: when it is null or old, the "
    "producer is not reporting, and an empty panel means EXACTLY that."
)

#: The controller's successful terminus. It is a member of `sm.STOP_STATES`
#: because the loop has stopped advancing, but it is NOT a blocking condition —
#: reporting the finish line as a blocker is how a surface teaches an operator to
#: ignore its blockers.
_NON_BLOCKING_STOP_STATES = frozenset({sm.RELEASE_PACKAGE_READY})


# =============================================================================
# Errors — each is a refusal, never a degraded export
# =============================================================================

class SurfaceError(Exception):
    """Base class for operator-surface failures."""


class ContractInputError(SurfaceError):
    """An input is missing, mistyped, or silently absent."""


class ExportDestinationRefused(SurfaceError):
    """The destination is a human-only target, a scratch path, or a production tree."""


class ContractInvalid(SurfaceError):
    """The derived document does not validate under its own schema."""


# =============================================================================
# Inputs — absence is passed EXPLICITLY or not at all
# =============================================================================

@dataclass(frozen=True)
class Unreported:
    """What a section looks like when its owner did not report.

    This type exists so that "the producer is dead" has a spelling. The
    alternative — `None`, or an omitted key — is how the `/kernel` panel came to
    render clean over a missing directory: nothing distinguished "no blocking
    conditions" from "nobody computed blocking conditions".

    `reason` is mandatory and must be non-empty. An unexplained absence is the
    same dead panel with a new label on it.
    """

    reason: str
    refused: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ContractInputError(
                "Unreported.reason: must be a non-empty explanation — an absence "
                "with no reason is indistinguishable from a healthy empty panel")
        if not isinstance(self.refused, bool):
            raise ContractInputError("Unreported.refused: must be a boolean")

    @property
    def status(self) -> str:
        return schemas.SECTION_REFUSED if self.refused else schemas.SECTION_NOT_REPORTED


@dataclass(frozen=True)
class ControllerObservation:
    """The campaign phase, as the controller itself reports it.

    Built by `observe_controller` from a live `ControllerStateMachine`, so the
    state, the sequence number and the last transition all come from the object
    that owns them. Nothing here re-derives a phase from journal contents.
    """

    campaign_id: str
    state: str
    seq: int
    stopped: bool
    last_transition: Any

    def __post_init__(self) -> None:
        _text(self.campaign_id, "ControllerObservation.campaign_id")
        if self.state not in sm.STATES:
            raise ContractInputError(
                f"ControllerObservation.state: {self.state!r} is not a controller "
                f"state; the vocabulary is owned by controller.state_machine.STATES")
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0:
            raise ContractInputError("ControllerObservation.seq: must be a non-negative int")
        if not isinstance(self.stopped, bool):
            raise ContractInputError("ControllerObservation.stopped: must be a boolean")
        if not isinstance(self.last_transition, sm.Transition):
            raise ContractInputError(
                "ControllerObservation.last_transition: must be a "
                "controller.state_machine.Transition — the surface reports the "
                "transition the ledger recorded, never a summary of it")


@dataclass(frozen=True)
class HeadroomObservation:
    """Storage and budget headroom, as `storage.py` and the budget ledger report it.

    A LIVE host reading, which is why `DASHBOARD_LIVENESS_SECTIONS` excludes this
    section from `produced_at`: free disk measured now says nothing about whether
    the loop is still running, and letting it say so would rebuild the scar one
    layer up.
    """

    storage_state: Any
    quota_state: Any
    budget: Any = None
    measured_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.storage_state, storage.StorageState):
            raise ContractInputError(
                "HeadroomObservation.storage_state: must be a storage.StorageState")
        if not isinstance(self.quota_state, storage.QuotaState):
            raise ContractInputError(
                "HeadroomObservation.quota_state: must be a storage.QuotaState")


@dataclass(frozen=True)
class ClaimsObservation:
    """The resource claims held, as `resource/device_claim.py` receipts.

    Also a live reading, and also excluded from liveness for the same reason: a
    claim file that is still on disk proves a lock exists, not that anything is
    using it.
    """

    receipts: tuple = ()
    observed_at: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipts", tuple(self.receipts))
        for receipt in self.receipts:
            for attr in ("claim_id", "device_id", "state", "campaign_id", "acquired_at"):
                if not hasattr(receipt, attr):
                    raise ContractInputError(
                        f"ClaimsObservation.receipts: {receipt!r} is not a device-claim "
                        f"receipt (missing {attr!r})")


@dataclass(frozen=True)
class BlockingCondition:
    """One open blocking condition, named by the vocabulary of its OWNER.

    `kind` is never invented here: it is `sm.EVALUATOR_COVERAGE_GAP`,
    `sm.ANCHOR_MOVED`, a `readiness.BLOCK_*` constant, a
    `PhaseTradeAssessment.STATUS_*` constant upper-cased, `storage.DISK_PRESSURE`
    or `packager.STATE_BLOCKED`. `derive_blocking_conditions` binds each one to
    the owning constant, and the test suite asserts the emitted set is a subset of
    the owners' vocabularies — which is what stops this module from growing a
    private taxonomy that drifts from the code that actually blocks.
    """

    kind: str
    origin: str
    detail: str
    since: Optional[str] = None
    owner: Optional[str] = None
    deadline: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.kind, "BlockingCondition.kind")
        _text(self.detail, "BlockingCondition.detail")
        if self.origin not in schemas.DASHBOARD_BLOCKING_ORIGINS:
            raise ContractInputError(
                f"BlockingCondition.origin: {self.origin!r} is not one of "
                f"{sorted(schemas.DASHBOARD_BLOCKING_ORIGINS)}")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "origin": self.origin, "detail": self.detail,
                "since": self.since, "owner": self.owner, "deadline": self.deadline}


@dataclass(frozen=True)
class BlockingObservation:
    """The open blocking conditions, and the newest RECORD time among them.

    `as_of` IS NOT AN INPUT. It is derived here, in `__post_init__`, from the
    `since` of the conditions themselves — each of which was copied off the
    record that established the block — and an explicitly supplied value that
    disagrees is refused rather than kept.

    The reason is a defect this class shipped with. `as_of` used to be whatever
    the caller passed to `derive_blocking_conditions`, and this section counted
    toward `produced_at`; so an exporter that passed its own wall clock produced
    a document whose controller, champion, readiness and package records were all
    a month old, with `produced_at: now` and `degraded: false`. Every other
    liveness timestamp in the contract is a journaled record's; this one was the
    hole, and a hole in the freshness envelope is the whole scar.

    `unreported_owners` names the owners `derive_blocking_conditions` could not
    read. Without it, "no open blocking conditions" and "nobody told me about the
    blocking conditions" render identically — the panel-level spelling of exactly
    the failure this contract exists to make impossible.
    """

    conditions: tuple = ()
    as_of: Optional[str] = None
    unreported_owners: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "unreported_owners", tuple(self.unreported_owners))
        for condition in self.conditions:
            if not isinstance(condition, BlockingCondition):
                raise ContractInputError(
                    "BlockingObservation.conditions: every entry must be a "
                    f"BlockingCondition, got {type(condition).__name__}")
        for owner in self.unreported_owners:
            _text(owner, "BlockingObservation.unreported_owners")
        derived = _newest_condition_timestamp(self.conditions)
        if self.as_of is not None and self.as_of != derived:
            raise ContractInputError(
                f"BlockingObservation.as_of: {self.as_of!r} is not the newest record "
                f"time among the conditions themselves ({derived!r}). This field is "
                "DERIVED, never supplied: a caller that can stamp it can date a "
                "derived section from the exporter's own clock, and a section dated "
                "by the exporter is a live surface process vouching for a dead loop.")
        object.__setattr__(self, "as_of", derived)


@dataclass(frozen=True)
class ContractInputs:
    """Everything the contract is derived from, one field per section.

    Every section field is EITHER the owning module's object OR an `Unreported`.
    `None` is refused (`_require_input`), because a forgotten input must not be
    able to produce a clean panel — that is the whole failure this contract
    exists to make impossible.
    """

    campaign_id: str
    controller: Any
    champion: Any
    readiness: Any
    headroom: Any
    blocking: Any
    claims: Any
    release_package: Any
    exported_at: str = ""

    def __post_init__(self) -> None:
        _text(self.campaign_id, "ContractInputs.campaign_id")
        if not self.campaign_id.startswith("ak-"):
            raise ContractInputError(
                f"ContractInputs.campaign_id: {self.campaign_id!r} must start with 'ak-'")
        _require_input(self.controller, "controller", ControllerObservation)
        _require_input(self.champion, "champion", Mapping)
        _require_input(self.readiness, "readiness", readiness_mod.ReadinessReport)
        _require_input(self.headroom, "headroom", HeadroomObservation)
        _require_input(self.blocking, "blocking", BlockingObservation)
        _require_input(self.claims, "claims", ClaimsObservation)
        _require_input(self.release_package, "release_package", packager.ReleasePackage)


def _newest_condition_timestamp(conditions: Sequence[Any]) -> Optional[str]:
    """The newest `since` among `conditions`, or None when none carries one.

    Ordered with `schemas._parse_ts` — the SAME parser
    `schemas.dashboard_liveness_timestamp` uses — deliberately: two timestamp
    orderings in one contract is two answers to "which of these is newer", and the
    one that would win is whichever the consumer happened to call.

    A condition with no `since` (an evaluator coverage gap carries an owner and a
    deadline, not a record time) contributes nothing rather than defaulting to now.

    NOTE for anyone tempted to put `blocking_conditions` back in
    `schemas.DASHBOARD_LIVENESS_SECTIONS`: one condition here — disk pressure —
    dates itself from `HeadroomObservation.measured_at`, which is a LIVE host
    reading taken by the exporter. It is honest as the time the pressure was
    observed and dishonest as evidence the loop is alive, which is precisely why
    this section does not establish liveness.
    """
    newest_raw = None
    newest_dt = None
    for condition in conditions:
        parsed = schemas._parse_ts(getattr(condition, "since", None))
        if parsed is None:
            continue
        if newest_dt is None or parsed > newest_dt:
            newest_dt, newest_raw = parsed, condition.since
    return newest_raw


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractInputError(f"{what}: must be a non-empty string")
    return value


def _require_input(value: Any, name: str, expected: Any) -> None:
    """Refuse `None` and refuse a wrong type; accept `Unreported`.

    THE BITE OF THIS FUNCTION is the `None` branch. Python's natural spelling for
    "I have nothing for this section" is `None`, and every absence-tolerant panel
    in this project was built out of exactly that reflex. Making it a hard input
    error forces the caller to write the reason down, and the reason is what the
    operator reads when the panel is empty.
    """
    if isinstance(value, Unreported):
        return
    if value is None:
        raise ContractInputError(
            f"ContractInputs.{name}: None is not an input. Pass the owning module's "
            f"object, or Unreported(reason=...) saying why nobody reported. A "
            f"section that can be silently absent renders as a clean panel over a "
            f"dead producer.")
    if not isinstance(value, expected):
        raise ContractInputError(
            f"ContractInputs.{name}: expected {getattr(expected, '__name__', expected)} "
            f"or Unreported, got {type(value).__name__}")


# =============================================================================
# Observation — built by ASKING the owning module, never by re-deriving
# =============================================================================

def observe_controller(machine: Any, *, campaign_id: str) -> ControllerObservation:
    """Read the campaign phase off a live `ControllerStateMachine`.

    Every field comes from the machine's own accessors — `state`, `seq`,
    `is_stopped()`, and the last entry of its transition ledger. This module does
    not replay the journal to decide what phase the campaign is in: the controller
    owns that, and a second opinion computed here would be the AutoPilot
    derived-view scar (a rebuilt view disagreeing with the record, with nothing
    objecting).
    """
    read = machine.ledger.read()
    transitions = getattr(read, "transitions", ())
    if not transitions:
        raise ContractInputError(
            "observe_controller: the transition ledger is empty, so there is no "
            "recorded phase to report. Pass Unreported(reason=...) instead of "
            "reporting a phase the ledger does not support.")
    return ControllerObservation(
        campaign_id=campaign_id,
        state=machine.state,
        seq=machine.seq,
        stopped=machine.is_stopped(),
        last_transition=transitions[-1])


def derive_blocking_conditions(
    *,
    controller: Any,
    readiness_report: Any,
    release_package: Any,
    headroom: Any = None,
    champion: Any = None,
    coverage_gaps: Sequence[Any] = (),
) -> BlockingObservation:
    """Collect the open blocking conditions from the modules that own them.

    THERE IS NO `as_of` PARAMETER. The observation dates itself from the records
    its conditions came from (`BlockingObservation.__post_init__`); an exporter
    that could date this section could date a derived panel from its own clock.

    Seven owners, seven vocabularies, zero new names:

      * `controller.state_machine` — the stop state itself, minus
        `RELEASE_PACKAGE_READY`, which is the successful terminus and not a block;
      * `release.readiness` — every `ReadinessSignal.blockers` entry, which is
        already a `BLOCK_*` constant (`ANCHOR_MOVED`, `COVERAGE_GAP`,
        `PHASE_TRADE_DECISION_REQUIRED`, …);
      * `release.readiness.PhaseTradeAssessment` — the phase-trade exception,
        reported when the assessment says the operator must decide, named by the
        assessment's OWN status constant;
      * `controller.context.CoverageGap` — an evaluator coverage gap, named by
        `sm.EVALUATOR_COVERAGE_GAP` and carrying the gap's owner and deadline;
      * `storage` — disk pressure, named by `storage.DISK_PRESSURE`;
      * `release.packager` — a blocked package, named by `packager.STATE_BLOCKED`,
        with each gating finding's own code;
      * the CHAMPION RECORD — its own `blocking_conditions` entries, which are
        machine names by convention across this package
        (`composition.BLOCKING_REANCHOR_REMEASURE`, `EVALUATOR_COVERAGE_GAP`,
        `T2_INTERACTION_FAILED`). Added after an adversarial pass produced a
        document whose champion was held by `EVALUATOR_COVERAGE_GAP` while the
        panel an operator reads to find out what is wrong said `open: []`.

    Every owner that did NOT report is named in `unreported_owners`, because an
    empty `open` list computed from silence is the same clean panel over a dead
    producer at one level down.

    Ordering is stable (owner order, then the owner's own order) so two exports of
    the same state produce byte-identical documents — a diff that changes when
    nothing changed teaches an operator to stop reading diffs.
    """
    out: list = []
    silent: list = []

    if not isinstance(controller, ControllerObservation):
        silent.append("controller")
    if not isinstance(readiness_report, readiness_mod.ReadinessReport):
        silent.append("readiness")
    if not isinstance(release_package, packager.ReleasePackage):
        silent.append("release_package")
    if not isinstance(headroom, HeadroomObservation):
        silent.append("headroom")
    if not isinstance(champion, Mapping):
        silent.append("champion")

    if isinstance(controller, ControllerObservation) and controller.stopped:
        if controller.state not in _NON_BLOCKING_STOP_STATES:
            out.append(BlockingCondition(
                kind=controller.state, origin="controller_stop",
                detail=controller.last_transition.reason,
                since=controller.last_transition.at))

    if isinstance(readiness_report, readiness_mod.ReadinessReport):
        for signal in readiness_report.signals:
            for blocker in signal.blockers:
                out.append(BlockingCondition(
                    kind=blocker, origin="readiness",
                    detail=f"{signal.backend}: readiness blocker from "
                           f"{signal.reducer_id}",
                    since=signal.computed_at, owner=signal.backend))
            trade = signal.phase_trade
            if getattr(trade, "operator_decision_required", False):
                out.append(BlockingCondition(
                    kind=str(trade.status).upper(), origin="phase_trade",
                    detail=f"{signal.backend}: phase trade on "
                           f"{trade.regressing_phase!r} requires an operator "
                           f"decision ({'; '.join(trade.reasons) or 'no reason given'})",
                    since=signal.computed_at, owner=signal.backend))

    for gap in coverage_gaps or ():
        out.append(BlockingCondition(
            kind=sm.EVALUATOR_COVERAGE_GAP, origin="evaluator_coverage",
            detail=f"{gap.missing_class} blocks {gap.blocked_lineage}",
            owner=gap.owner, deadline=gap.deadline))

    if isinstance(headroom, HeadroomObservation):
        if headroom.storage_state.state == storage.DISK_PRESSURE:
            out.append(BlockingCondition(
                kind=storage.DISK_PRESSURE, origin="storage",
                detail="; ".join(headroom.storage_state.reasons) or "disk pressure",
                since=headroom.measured_at))

    if isinstance(release_package, packager.ReleasePackage):
        if release_package.state == packager.STATE_BLOCKED:
            for finding in release_package.blocking_findings:
                out.append(BlockingCondition(
                    kind=packager.STATE_BLOCKED, origin="release_package",
                    detail=f"{finding.code}: {finding.detail}",
                    since=release_package.created_at))

    if isinstance(champion, Mapping):
        held_by = champion.get("combined_candidate_id") or "the champion lineage"
        for entry in champion.get("blocking_conditions") or ():
            text = entry if isinstance(entry, str) else repr(entry)
            named = isinstance(entry, str) and bool(schemas._BLOCKING_KIND_RE.match(entry))
            out.append(BlockingCondition(
                kind=text if named else schemas.CHAMPION_BLOCKED_UNNAMED,
                origin="champion",
                detail=f"{held_by} is held by {text}",
                since=champion.get("created_at")))

    return BlockingObservation(conditions=tuple(out), unreported_owners=tuple(silent))


# =============================================================================
# Section rendering — each section says who reported it and when
# =============================================================================

def _unreported_section(absent: Unreported) -> dict:
    return {"status": absent.status, "as_of": None, "reason": absent.reason}


def _check_dict(check: Any) -> Optional[dict]:
    """A `schemas.Check` rendered as data, keeping COULD_NOT_CHECK distinguishable.

    The third outcome is the point: a consumer that folds it into PASS or FAIL
    turns "we could not tell" into an answer, which is the same class of defect as
    a clean panel over a dead producer.
    """
    if not isinstance(check, schemas.Check):
        # NOT `return None`. Rendering an unrecognised verdict as null is a
        # fail-open default inside the one module whose whole purpose is to not
        # fail open: `null` is what an absent check looks like, and a consumer
        # reads an absent check as "nothing to report". Every `Check` field on
        # `ReadinessSignal`, `MatrixCoverage` and `PhaseStanding` is typed and
        # guarded at construction, so anything else here is a caller handing this
        # module an object it did not verify.
        raise ContractInputError(
            f"a readiness verdict rendered as {type(check).__name__} is not a "
            "schemas.Check. The surface refuses it rather than emitting null: a "
            "null verdict reads as 'nothing to report', which is the clean panel "
            "this contract exists to make impossible.")
    return {"outcome": check.outcome, "reasons": list(check.reasons)}


def _campaign_section(observation: Any) -> dict:
    if isinstance(observation, Unreported):
        return _unreported_section(observation)
    transition = observation.last_transition
    return {
        "status": schemas.SECTION_OBSERVED,
        "as_of": transition.at,
        "campaign_id": observation.campaign_id,
        "state": observation.state,
        "seq": observation.seq,
        "stopped": observation.stopped,
        "last_transition": {
            "seq": transition.seq,
            "from_state": transition.from_state,
            "to_state": transition.to_state,
            "trigger": transition.trigger,
            "reason": transition.reason,
            "at": transition.at,
            "receipt": transition.receipt,
        },
    }


def _champion_section(record: Any) -> dict:
    """Champion membership and readiness, off the champion RECORD.

    The record is `schemas.SCHEMA_CHAMPION`, which already carries membership, the
    composed candidate, the branch, the anchor commit, its own blocking conditions
    and its rendered readiness signal. It is validated before it is rendered: a
    champion record this module could not validate is one it must not present as
    a champion.
    """
    if isinstance(record, Unreported):
        return _unreported_section(record)
    violations = schemas.validate_champion(record)
    if violations:
        raise ContractInputError(
            "ContractInputs.champion: the champion record does not validate, so it "
            "may not be rendered as a champion: " + "; ".join(violations))
    created_at = record.get("created_at")
    if not created_at:
        # `validate_champion` leaves `created_at` optional, and for a journal record
        # that is fine. For the SURFACE it is not: a champion rendered without a
        # record time contributes an unfalsifiable "present" to the panel and
        # nothing to its freshness, which is precisely the shape of a clean panel
        # over a dead producer.
        raise ContractInputError(
            "ContractInputs.champion: the champion record carries no 'created_at', "
            "so the surface cannot say when the loop last wrote it. Pass "
            "Unreported(reason=...) rather than a champion with no record time.")
    readiness_block = record.get("readiness") or {}
    return {
        "status": schemas.SECTION_OBSERVED,
        "as_of": created_at,
        "combined_candidate_id": record.get("combined_candidate_id"),
        "member_candidate_ids": list(record.get("member_candidates") or ()),
        "source_tree": record.get("source_tree"),
        "branch": record.get("branch"),
        "anchor_commit": record.get("anchor_commit"),
        "affected_surface_union_sha256": record.get("affected_surface_union_sha256"),
        "storage_gb": record.get("storage_gb"),
        "blocking_conditions": list(record.get("blocking_conditions") or ()),
        "readiness": {
            "reference_signal": readiness_block.get("reference_signal"),
            "by_backend": dict(readiness_block.get("by_backend") or {}),
        },
    }


def _standing_section(report: Any) -> dict:
    """Per-backend standing, copied off each `ReadinessSignal`.

    `standing`, `blockers`, the per-phase checks and the coverage verdict are all
    read, never recomputed. `readiness.compute_readiness` is the reducer named in
    the signal (`reducer_id`); a second reduction here would be a second source of
    truth for the one number a freeze decision leans on.
    """
    if isinstance(report, Unreported):
        return _unreported_section(report)
    backends: dict = {}
    newest: Optional[str] = None
    for signal in report.signals:
        backends[signal.backend] = {
            "standing": signal.standing,
            "blockers": list(signal.blockers),
            "source_tree": signal.source_tree,
            "champion_candidate_id": signal.champion_candidate_id,
            "controls_marker": signal.controls_marker,
            "evaluator_bundle_sha256": signal.evaluator_bundle_sha256,
            "reducer_id": signal.reducer_id,
            "statistics_module_id": signal.statistics_module_id,
            "computed_at": signal.computed_at,
            "matrix_coverage": _check_dict(signal.matrix.overall),
            "improvement_backend_wide": _check_dict(signal.improvement_backend_wide),
            "improvement_per_protected_cell": _check_dict(
                signal.improvement_per_protected_cell),
            "phase_trade": {
                "status": signal.phase_trade.status,
                "regressing_phase": signal.phase_trade.regressing_phase,
                "operator_decision_required":
                    signal.phase_trade.operator_decision_required,
                "reasons": list(signal.phase_trade.reasons),
            },
            "phases": [
                {
                    "phase": phase.phase,
                    "protocol_id": phase.protocol_id,
                    "non_inferior": _check_dict(phase.non_inferior),
                    "improved": _check_dict(phase.improved),
                    "blockers": list(phase.blockers),
                }
                for phase in signal.phases
            ],
        }
        if newest is None or (signal.computed_at or "") > newest:
            newest = signal.computed_at
    return {
        "status": schemas.SECTION_OBSERVED,
        "as_of": report.computed_at or newest,
        "campaign_id": report.campaign_id,
        "backends": backends,
    }


def _headroom_section(observation: Any) -> dict:
    if isinstance(observation, Unreported):
        return _unreported_section(observation)
    state = observation.storage_state
    quota = observation.quota_state
    budget = observation.budget
    budget_block = None
    if budget is not None:
        budget_block = {
            "proposals_recorded": budget.proposals_recorded,
            "candidates_recorded": budget.candidates_recorded,
            "controller_tokens": budget.controller_tokens,
            "build_seconds": budget.build_seconds,
            "evaluator_wall_seconds": budget.evaluator_wall_seconds,
            "gpu_seconds": budget.gpu_seconds,
            "cpu_region_seconds": budget.cpu_region_seconds,
            "storage_gb": budget.storage_gb,
        }
    return {
        "status": schemas.SECTION_OBSERVED,
        # A LIVE reading: `as_of` records when the host was measured and is
        # deliberately not admitted to `produced_at` (see the module docstring).
        "as_of": observation.measured_at,
        "storage": {
            "state": state.state,
            "free_bytes": state.free_bytes,
            "total_bytes": state.total_bytes,
            "floor_bytes": state.floor_bytes,
            "pressured": state.pressured,
            "reasons": list(state.reasons),
        },
        "quota": {
            "state": quota.state,
            "used_bytes": quota.used_bytes,
            "limit_bytes": quota.limit_bytes,
            "fraction": quota.fraction,
            "exhausted": quota.exhausted,
            "reasons": list(quota.reasons),
        },
        "budget": budget_block,
    }


def _blocking_section(observation: Any) -> dict:
    """The open conditions, plus the owners that did not answer.

    A DERIVED section: `as_of` is the newest record time among the conditions
    themselves (never the exporter's clock), and this section does not establish
    liveness — `schemas.DASHBOARD_LIVENESS_SECTIONS` excludes it, because a
    section restating other sections can only inherit freshness.

    `unreported_owners` is rendered even when it is empty, so a consumer reads the
    same key every time instead of inferring health from a missing one.
    """
    if isinstance(observation, Unreported):
        return _unreported_section(observation)
    return {
        "status": schemas.SECTION_OBSERVED,
        "as_of": observation.as_of,
        "open": [condition.to_dict() for condition in observation.conditions],
        "unreported_owners": list(observation.unreported_owners),
    }


def _claims_section(observation: Any) -> dict:
    if isinstance(observation, Unreported):
        return _unreported_section(observation)
    return {
        "status": schemas.SECTION_OBSERVED,
        # Also a live reading, also excluded from `produced_at`.
        "as_of": observation.observed_at,
        "held": [
            {
                "claim_id": receipt.claim_id,
                "device_id": receipt.device_id,
                "state": receipt.state,
                "campaign_id": receipt.campaign_id,
                "purpose": receipt.purpose,
                "host": getattr(receipt, "host", None),
                "holder_pid": getattr(receipt, "holder_pid", None),
                "holder_label": getattr(receipt, "holder_label", None),
                "acquired_at": receipt.acquired_at,
                "expires_at": getattr(receipt, "expires_at", None),
                "released_at": getattr(receipt, "released_at", None),
            }
            for receipt in observation.receipts
        ],
    }


def _release_package_section(package: Any) -> dict:
    if isinstance(package, Unreported):
        return _unreported_section(package)
    return {
        "status": schemas.SECTION_OBSERVED,
        "as_of": package.created_at,
        "package_id": package.package_id,
        "campaign_id": package.campaign_id,
        "source_tree": package.source_tree,
        # `state` is the packager's own derived value; `ReleasePackage.__post_init__`
        # already refuses a state its findings do not yield, so copying it is the
        # only way to stay in agreement with the packager.
        "state": package.state,
        "requires_human_code_review": package.requires_human_code_review,
        "change_classes": list(package.change_classes),
        "blocking_findings": [
            {"code": finding.code, "detail": finding.detail,
             "outcome": finding.outcome}
            for finding in package.blocking_findings
        ],
        "executed_by": packager.EXECUTED_BY,
    }


_SECTION_BUILDERS = {
    schemas.DASHBOARD_SECTION_CAMPAIGN: ("controller", _campaign_section),
    schemas.DASHBOARD_SECTION_CHAMPION: ("champion", _champion_section),
    schemas.DASHBOARD_SECTION_BACKEND_STANDING: ("readiness", _standing_section),
    schemas.DASHBOARD_SECTION_HEADROOM: ("headroom", _headroom_section),
    schemas.DASHBOARD_SECTION_BLOCKING: ("blocking", _blocking_section),
    schemas.DASHBOARD_SECTION_CLAIMS: ("claims", _claims_section),
    schemas.DASHBOARD_SECTION_RELEASE_PACKAGE: ("release_package",
                                                _release_package_section),
}


# =============================================================================
# The contract
# =============================================================================

def _condition_key(condition: BlockingCondition) -> tuple:
    return (condition.kind, condition.origin, condition.owner)


def _assert_blocking_agrees_with_its_own_inputs(inputs: ContractInputs) -> None:
    """Refuse a blocking panel that omits a block the SAME document reports.

    THE DEFECT THIS CLOSES, found by attacking the built module: `blocking` was
    whatever the caller handed over, checked against nothing. So a document could
    carry `backend_standing.llama_cpu.standing = objective_not_met` with
    `blockers: ['ANCHOR_MOVED']`, a `campaign` section STOPPED on `ANCHOR_MOVED`,
    and a champion held by `EVALUATOR_COVERAGE_GAP` — while
    `blocking_conditions.open` was `[]` and `degraded` was `false`, and the
    document validated. The one panel an operator reads to answer "is anything
    wrong?" rendered clean over three blocks the rest of the same file spelled
    out.

    The fix is the one the packager and the validator already use against exactly
    this class of hole: RECOMPUTE and compare, instead of trusting a summary that
    can be stamped independently of its evidence.

    CONTAINMENT, not equality, and the asymmetry is load-bearing: evaluator
    coverage gaps come from `controller.context`, which `ContractInputs` does not
    carry, so a caller must be able to report MORE than this function can rebuild.
    It may never report less.
    """
    reported = inputs.blocking
    if isinstance(reported, Unreported):
        # An absent panel is honest — it renders as absence and counts toward
        # `degraded`. It is the SILENTLY EMPTY one that lies.
        return
    rebuilt = derive_blocking_conditions(
        controller=inputs.controller, readiness_report=inputs.readiness,
        release_package=inputs.release_package, headroom=inputs.headroom,
        champion=inputs.champion)
    present = {_condition_key(condition) for condition in reported.conditions}
    missing = [condition for condition in rebuilt.conditions
               if _condition_key(condition) not in present]
    if missing:
        raise ContractInputError(
            "ContractInputs.blocking: the blocking panel omits "
            f"{len(missing)} condition(s) this document's OWN sections report — "
            + "; ".join(f"{c.kind} (from {c.origin})" for c in missing)
            + ". Derive the panel with derive_blocking_conditions() from the same "
            "objects, or pass Unreported(reason=...). A panel that reads clear "
            "while the sections beside it read blocked is the absence-tolerance "
            "scar with the producer alive.")


def build_contract(inputs: ContractInputs) -> dict:
    """Derive the v2 contract document. Pure: no I/O, no clock, no host access.

    The document's own summary fields — `produced_at`, `generated_at`, `degraded`,
    `unreported_sections` — are computed by the SAME functions the validator uses
    to check them (`schemas.dashboard_liveness_timestamp`,
    `schemas.dashboard_unreported_sections`). That is deliberate: a producer with
    its own copy of the rule can drift from the checker, and the direction it
    drifts is always "looks healthier than it is".
    """
    if not isinstance(inputs, ContractInputs):
        raise ContractInputError(
            f"build_contract: expected ContractInputs, got {type(inputs).__name__}")
    exported_at = _text(inputs.exported_at, "ContractInputs.exported_at")
    _assert_blocking_agrees_with_its_own_inputs(inputs)

    sections: dict = {}
    for name in schemas.DASHBOARD_SECTIONS:
        attr, builder = _SECTION_BUILDERS[name]
        sections[name] = builder(getattr(inputs, attr))

    produced_at = schemas.dashboard_liveness_timestamp(sections)
    unreported = schemas.dashboard_unreported_sections(sections)

    # Run identity: a consumer comparing two exports can tell whether the LOOP
    # advanced without stat()ing anything — same campaign and same controller
    # sequence means it did not. When the controller did not report there is no
    # producing run to name, and `run` is null rather than invented.
    controller = inputs.controller
    run = None
    if isinstance(controller, ControllerObservation):
        run = {
            "campaign_id": controller.campaign_id,
            "controller_seq": controller.seq,
            "controller_state": controller.state,
            "ledger_receipt": controller.last_transition.receipt,
        }

    document = {
        "schema": CONTRACT_SCHEMA,
        "contract_version": 2,
        "campaign_id": inputs.campaign_id,
        "produced_at": produced_at,
        # The v1 spelling of the SAME semantic value, so the deployed hub reader
        # (which looks for `generated_at`) classifies a v2 file correctly instead
        # of falling through to "missing" and rendering an empty-but-clean panel.
        "generated_at": produced_at,
        "exported_at": exported_at,
        "producer": {"module_id": MODULE_ID, "run": run},
        "sections": sections,
        "degraded": bool(unreported),
        "unreported_sections": unreported,
        "observation_notice": OBSERVATION_NOTICE,
    }

    violations = schemas.validate_kernel_dashboard_v2(document)
    if violations:
        raise ContractInvalid(
            "build_contract produced a document that does not validate under "
            f"{CONTRACT_SCHEMA}: " + "; ".join(violations))
    return document


# =============================================================================
# The one write
# =============================================================================

def _human_only_reasons(text: str) -> list:
    """Reasons `text` names a human-only target — SSOT is the packager's table."""
    return [reason for pattern, reason in packager.HUMAN_ONLY_TARGET_PATTERNS
            if pattern.search(text)]


def _enclosing_checkout(path: str) -> Optional[str]:
    """The nearest ancestor directory that is a git working tree, or None.

    `.git` is tested with `exists()`, not `is_dir()`: a worktree added with
    `git worktree add` carries a `.git` FILE, and this repository's own identity
    rule (`/workspace/repos/<name>` symlinked into `/mnt/raid0/llm/<name>`) makes
    linked worktrees and shared clones the normal case rather than an exotic one.
    """
    for ancestor in pathlib.Path(path).parents:
        if (ancestor / ".git").exists():
            return str(ancestor)
    return None


def assert_exportable_destination(path: Any) -> str:
    """Return the absolute destination, or REFUSE it.

    Five refusals, in the order that closes the loopholes rather than the order
    that reads nicely:

      1. the literal path names a human-only target
         (`packager.HUMAN_ONLY_TARGET_PATTERNS` — production branch, stable kernel
         path, era registry, AutoPilot baseline, the human-only path list itself);
      2. the SYMLINK-RESOLVED path names one. Checking only the literal string is
         how a guard gets bypassed by `ln -s`, and this repository's own
         working-tree identity rule (`/workspace/repos/<name>` is a symlink into
         `/mnt/raid0/llm/<name>`) means symlinked roots are the normal case here,
         not an exotic one;
      3. the resolved path is inside a frozen production tree
         (`storage.PRODUCTION_TREES` via `storage.production_tree_forms`,
         invariant 3);
      4. the resolved path is scratch (`storage.is_scratch_path`) — the defect in
         the path this module replaces, where the export lived one `tmp` sweep
         from vanishing with no event behind it;
      5. the resolved path is inside a GIT WORKING TREE. The chosen destination is
         durable *because* it "is outside every checkout, so nothing here can ride
         into a parallel session's commit" — and until this check existed that was
         a sentence in a comment, not a property: an adversarial pass pointed the
         one writer at `epyc-inference-research/data/` and at
         `/workspace/handoffs/active/` and both were accepted. Both are shared
         clones (CLAUDE.md, working-tree identity), a file rewritten on every
         export is the worst possible shape for a tracked path, and this
         repository has already been bitten by staged files riding into another
         session's commit. A guarantee stated in prose is not a guarantee.

    A directory, or a name that is not `.json`, is refused too: this writer emits
    one JSON document and has no business being pointed at anything else.
    """
    if isinstance(path, os.PathLike):
        raw = os.fspath(path)
    elif isinstance(path, str):
        raw = path
    else:
        raise ExportDestinationRefused(
            f"export destination: expected a path, got {type(path).__name__}")
    if not raw.strip():
        raise ExportDestinationRefused("export destination: must not be empty")

    absolute = os.path.abspath(os.path.expanduser(raw))
    resolved = str(pathlib.Path(absolute).resolve())

    for candidate, how in ((raw, "as written"), (absolute, "absolute"),
                           (resolved, "symlink-resolved")):
        reasons = _human_only_reasons(candidate)
        if reasons:
            raise ExportDestinationRefused(
                f"export destination {candidate!r} ({how}) {reasons[0]}. AutoKernel "
                "writes one derived view of its own journal and nothing else; a "
                "human-only target is not a place this module may name, let alone "
                "write.")

    for form in storage.production_tree_forms():
        if resolved == form.rstrip("/") or resolved.startswith(form.rstrip("/") + "/"):
            raise ExportDestinationRefused(
                f"export destination {resolved!r} is inside the frozen production "
                f"tree {form!r} (invariant 3: no actor modifies a production tree)")

    if storage.is_scratch_path(resolved):
        raise ExportDestinationRefused(
            f"export destination {resolved!r} resolves under a scratch root "
            f"({', '.join(storage.EPHEMERAL_ROOTS)}). The panel this file feeds was "
            "already pointed at a scratch path once and rendered clean over the "
            "hole; a durable surface may not live one sweep from gone.")

    checkout = _enclosing_checkout(resolved)
    if checkout is not None:
        raise ExportDestinationRefused(
            f"export destination {resolved!r} is inside the git working tree "
            f"{checkout!r}. This module rewrites its output on every export and "
            "every checkout on this host is shared with other live sessions, so a "
            "destination in one is a churning file that rides into somebody else's "
            "commit. The derived view lives on durable disk outside every "
            f"checkout ({DEFAULT_EXPORT_ROOT}); the journal it is derived from is "
            "the evidence of record.")

    if not resolved.endswith(".json"):
        raise ExportDestinationRefused(
            f"export destination {resolved!r} is not a .json file; this module "
            "writes one JSON document and nothing else")
    if os.path.isdir(absolute):
        raise ExportDestinationRefused(
            f"export destination {absolute!r} is a directory")
    return absolute


@dataclass(frozen=True)
class ExportReceipt:
    """What was written, where, and what it says about liveness."""

    path: str
    bytes_written: int
    sha256: str
    produced_at: Optional[str]
    exported_at: str
    degraded: bool
    unreported_sections: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "unreported_sections",
                           tuple(self.unreported_sections))


def export_contract(document: Mapping, *, path: Any = None) -> ExportReceipt:
    """Validate `document` and write it to ONE durable path, atomically.

    Validation happens BEFORE any byte is written, so the file on disk is never a
    document the schema would reject: a consumer that meets a malformed export has
    to guess, and the guess that comes naturally — "treat it as empty" — is the
    absence-tolerant failure again.

    The write is atomic (temporary file in the destination directory, then
    `os.replace`), so a reader never sees a half-written contract and a crashed
    exporter leaves the previous, honest document in place rather than a truncated
    one that looks like a dead loop.
    """
    if not isinstance(document, Mapping):
        raise ContractInvalid(
            f"export_contract: expected a mapping, got {type(document).__name__}")
    violations = schemas.validate_kernel_dashboard_v2(document)
    if violations:
        raise ContractInvalid(
            "export_contract refuses an invalid document: " + "; ".join(violations))

    destination = assert_exportable_destination(
        DEFAULT_EXPORT_PATH if path is None else path)
    payload = schemas.canonical_bytes(dict(document))
    digest = schemas.content_hash(dict(document))

    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = os.path.join(parent, f".{os.path.basename(destination)}.{os.getpid()}.tmp")
    try:
        handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            # `os.write` MAY WRITE FEWER BYTES THAN IT IS GIVEN, and its return
            # value was being discarded. An adversarial pass made it short and the
            # result was the worst available outcome: a truncated document
            # installed atomically OVER the previous honest one, with a receipt
            # reporting the full length and the full document's hash. Loop until
            # the payload is out, then verify the file's own size against it —
            # the check that survives even if this loop is one day rewritten.
            written = 0
            while written < len(payload):
                count = os.write(handle, payload[written:])
                if count <= 0:
                    raise ContractInvalid(
                        f"export_contract: wrote {written} of {len(payload)} bytes "
                        f"to {temporary!r} and the write stopped making progress. "
                        "Nothing is installed: a truncated contract replacing a "
                        "good one is worse than a stale good one, because it "
                        "parses as nothing and a consumer reads nothing as empty.")
                written += count
            os.fsync(handle)
            actual = os.fstat(handle).st_size
        finally:
            os.close(handle)
        if actual != len(payload):
            raise ContractInvalid(
                f"export_contract: {temporary!r} holds {actual} bytes but the "
                f"document is {len(payload)}. The partial file is discarded and "
                "the previous export is left in place.")
        os.replace(temporary, destination)
    except BaseException:
        # A temporary left in the durable directory outlives the crash that made
        # it, and the next reader finds a dotfile nobody owns.
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    # The rename itself has to survive a host crash, or "atomic" only means
    # "atomic until the power goes". Directory fsync is the second half of the
    # atomic-write recipe and it is the half that gets left out.
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)

    return ExportReceipt(
        path=destination,
        bytes_written=len(payload),
        sha256=digest,
        produced_at=document.get("produced_at"),
        exported_at=document.get("exported_at"),
        degraded=bool(document.get("degraded")),
        unreported_sections=tuple(document.get("unreported_sections") or ()))
