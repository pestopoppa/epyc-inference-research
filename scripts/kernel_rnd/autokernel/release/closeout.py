#!/usr/bin/env python3
"""Operator-triggered AutoKernel dry-run closeout.

This is an integration seam, not another controller.  It runs an injected lean
sequencer, resolves the schema-bound composed champion from its journal, records
the operator's request, and then invokes the existing readiness, T3 and package
planes.  Its terminal success is a validated ``RELEASE_PACKAGE_READY`` record.

The module deliberately lives under :mod:`autokernel.release` and is not imported
by that package's initializer.  Campaign #1 therefore cannot reach it.  It owns no
build, benchmark, inference, process, clock or production-write capability; all
candidate and release material is caller supplied, and the only mutation here is
append-only AutoKernel journal evidence.

Synthetic material is labelled ``architecture_regression_fixture`` in both the
durable request and package.  Such a package proves wiring and recovery semantics
only.  It is never empirical evidence and cannot authorize production action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

from .. import journal, schemas
from ..controller import champion, sequencer
from . import packager, readiness, t3

MODULE_ID = "autokernel.release.closeout/v1"

EVIDENCE_ARCHITECTURE_FIXTURE = "architecture_regression_fixture"
EVIDENCE_OPERATOR_SUPPLIED = "operator_supplied_dry_run_evidence"
EVIDENCE_CLASSES = frozenset({
    EVIDENCE_ARCHITECTURE_FIXTURE,
    EVIDENCE_OPERATOR_SUPPLIED,
})

STATE_READY = packager.STATE_READY
STATE_RESOURCE_PREEMPTED = "RESOURCE_PREEMPTED"
STATE_TAMPER_REFUSED = "TAMPER_REFUSED"
STATE_FAILED = "FAILED"
TERMINAL_FAILURE_STATES = frozenset({
    STATE_RESOURCE_PREEMPTED, STATE_TAMPER_REFUSED, STATE_FAILED,
})


class CloseoutError(Exception):
    """Base for closeout material, identity and lifecycle refusals."""


class CloseoutUnavailable(CloseoutError):
    """No active composed champion is available for an operator request."""


class CloseoutTampered(CloseoutError):
    """Two supposedly identical lifecycle objects disagree on identity or bytes."""


class CloseoutNotReady(CloseoutError):
    """The dry-run package is valid but its own evidence does not derive READY."""


class ResourcePreempted(CloseoutError):
    """The injected compiler lost a predeclared resource before it produced material."""


@dataclass(frozen=True)
class PackageAssemblyInputs:
    """Every package input except the T3 verdict and readiness signal.

    Those two are deliberately absent: :class:`OperatorCloseout` obtains them by
    calling the current release evaluator and readiness reducer seam, so an
    injected compiler cannot smuggle a precomputed verdict into the package.
    """

    package_id: str
    created_at: str
    freeze_request: packager.OperatorFreezeRequest
    sealed: packager.SealedRelease
    version: packager.NextVersion
    transaction: t3.TransactionPlan
    rollback: packager.RollbackPlan
    era_row_draft: Mapping[str, Any]
    rebaseline_note: str
    commands: Sequence[packager.OperatorCommand]
    watch_window: packager.WatchWindow
    cutover_request: packager.CutoverRequest
    autopilot_baseline_path: str
    change_classes: Sequence[str]
    diff_complexity: Mapping[str, Any]
    waivers: Sequence[t3.WaiverBinding] = ()
    release_plan: Optional[Mapping[str, Any]] = None

    def assemble(self, *, evaluation: packager.TrustedEvaluation,
                 readiness_report: readiness.ReadinessReport
                 ) -> packager.ReleasePackage:
        return packager.assemble_release_package(
            package_id=self.package_id,
            created_at=self.created_at,
            freeze_request=self.freeze_request,
            sealed=self.sealed,
            evaluation=evaluation,
            version=self.version,
            transaction=self.transaction,
            rollback=self.rollback,
            era_row_draft=self.era_row_draft,
            rebaseline_note=self.rebaseline_note,
            commands=self.commands,
            watch_window=self.watch_window,
            cutover_request=self.cutover_request,
            autopilot_baseline_path=self.autopilot_baseline_path,
            change_classes=self.change_classes,
            diff_complexity=self.diff_complexity,
            waivers=self.waivers,
            release_plan=self.release_plan,
            readiness_signal=readiness_report.to_dict(),
        )


@dataclass(frozen=True)
class CompiledReleaseMaterial:
    """Deterministic, caller-supplied inputs to the existing release plane."""

    readiness_report: readiness.ReadinessReport
    t3_request: t3.T3Request
    package: PackageAssemblyInputs


class ReleaseMaterialCompiler(Protocol):
    def compile(self, *, freeze_request: packager.OperatorFreezeRequest,
                state: champion.SourceTreeState,
                snapshot: champion.JournalSnapshot,
                champion_event: journal.JournalEntry) -> CompiledReleaseMaterial: ...


class OperatorRequestSupplier(Protocol):
    """The external human-authority seam; this module never mints a request."""

    def request(self, *, state: champion.SourceTreeState,
                snapshot: champion.JournalSnapshot,
                champion_event: journal.JournalEntry
                ) -> packager.OperatorFreezeRequest: ...


@dataclass(frozen=True)
class CloseoutResult:
    state: str
    search: sequencer.LoopResult
    champion_event_id: str
    request_event_id: str
    request_sha256: str
    package_event_id: Optional[str] = None
    package: Optional[Mapping[str, Any]] = None
    terminal_event_id: Optional[str] = None
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.state == STATE_READY

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.operator_closeout_result.v1",
            "state": self.state,
            "ready": self.ready,
            "search": self.search.to_dict(),
            "champion_event_id": self.champion_event_id,
            "request_event_id": self.request_event_id,
            "request_sha256": self.request_sha256,
            "package_event_id": self.package_event_id,
            "package": None if self.package is None else dict(self.package),
            "terminal_event_id": self.terminal_event_id,
            "detail": self.detail,
        }


def _latest_champion_event(snapshot: champion.JournalSnapshot,
                           source_tree: str) -> journal.JournalEntry:
    matches = [entry for entry in snapshot.entries
               if entry.kind == journal.KIND_CHAMPION_UPDATED
               and entry.payload.get("source_tree") == source_tree
               and entry.payload.get("status") in {"active", "reanchored"}]
    if not matches:
        raise CloseoutUnavailable(
            f"{source_tree}: no active schema-bound composed champion")
    return max(matches, key=lambda entry: entry.seq)


def _request_payload(*, freeze_request: packager.OperatorFreezeRequest,
                     champion_event: journal.JournalEntry,
                     state: champion.SourceTreeState,
                     evidence_class: str) -> dict:
    if evidence_class not in EVIDENCE_CLASSES:
        raise CloseoutTampered(
            f"evidence_class {evidence_class!r} is not one of "
            f"{sorted(EVIDENCE_CLASSES)}")
    body = {
        "freeze_request": freeze_request.to_dict(),
        "champion_event_id": champion_event.event_id,
        "combined_candidate_id": state.composed_champion,
        "evidence_class": evidence_class,
    }
    return {
        "request_id": freeze_request.request_id,
        "request_sha256": schemas.content_hash(body),
        "source_tree": freeze_request.source_tree,
        "requested_by": freeze_request.requested_by,
        "requested_at": freeze_request.requested_at,
        **body,
    }


def _existing_package(snapshot: champion.JournalSnapshot,
                      request_sha256: str) -> Optional[journal.JournalEntry]:
    matches = [entry for entry in snapshot.entries
               if entry.kind == journal.KIND_RELEASE_PACKAGE_PREPARED
               and (entry.payload.get("operator_dry_run") or {}).get(
                   "request_sha256") == request_sha256]
    if len({schemas.canonical_json(entry.payload) for entry in matches}) > 1:
        raise CloseoutTampered(
            "one operator dry-run request resolves to contradictory package bytes")
    return max(matches, key=lambda entry: entry.seq) if matches else None


def _existing_terminal(snapshot: champion.JournalSnapshot,
                       request_sha256: str) -> Optional[journal.JournalEntry]:
    matches = [entry for entry in snapshot.entries
               if entry.kind == journal.KIND_OPERATOR_RELEASE_DRY_RUN_TERMINATED
               and entry.payload.get("request_sha256") == request_sha256]
    return max(matches, key=lambda entry: entry.seq) if matches else None


def _validate_material(material: CompiledReleaseMaterial, *,
                       freeze_request: packager.OperatorFreezeRequest,
                       state: champion.SourceTreeState,
                       champion_event: journal.JournalEntry) -> None:
    if not isinstance(material, CompiledReleaseMaterial):
        raise CloseoutTampered(
            "release material compiler must return CompiledReleaseMaterial")
    report = material.readiness_report
    if not isinstance(report, readiness.ReadinessReport) or not report.signals:
        raise CloseoutTampered("readiness material is not a non-empty ReadinessReport")
    if report.campaign_id != freeze_request.campaign_id:
        raise CloseoutTampered("readiness report belongs to another campaign")
    for signal in report.signals:
        if signal.source_tree != state.source_tree:
            raise CloseoutTampered("readiness signal belongs to another source tree")
        if signal.champion_candidate_id != state.composed_champion:
            raise CloseoutTampered("readiness signal belongs to another champion")
        if signal.anchor.source_commit != state.incumbent.commit:
            raise CloseoutTampered("readiness signal uses another production denominator")
        evaluator = champion_event.payload.get("evaluator") or {}
        if signal.evaluator_bundle_sha256 != evaluator.get("bundle_sha256"):
            raise CloseoutTampered("readiness signal uses another evaluator bundle")

    request = material.t3_request
    if not isinstance(request, t3.T3Request):
        raise CloseoutTampered("compiled T3 input is not a T3Request")
    if request.mode != t3.MODE_DRY_RUN:
        raise CloseoutTampered(
            "operator closeout accepts dry-run T3 only; release mode remains a "
            "separate operator action and unratified protocols refuse it")
    if request.campaign_id != freeze_request.campaign_id:
        raise CloseoutTampered("T3 request belongs to another campaign")
    if request.sealed.candidate_id != state.composed_champion:
        raise CloseoutTampered("T3 request seals another candidate")
    if request.sealed.source_tree != state.source_tree:
        raise CloseoutTampered("T3 request seals another source tree")
    if request.sealed.production_base_commit != state.incumbent.commit:
        raise CloseoutTampered("T3 request uses another production denominator")
    evaluator = champion_event.payload.get("evaluator") or {}
    if request.sealed.evaluator_bundle_sha256 != evaluator.get("bundle_sha256"):
        raise CloseoutTampered("T3 request uses another evaluator bundle")

    assembly = material.package
    if not isinstance(assembly, PackageAssemblyInputs):
        raise CloseoutTampered("compiled package input is not PackageAssemblyInputs")
    if schemas.canonical_json(assembly.freeze_request.to_dict()) != \
            schemas.canonical_json(freeze_request.to_dict()):
        raise CloseoutTampered("package carries another operator freeze request")
    if assembly.sealed.candidate != request.sealed:
        raise CloseoutTampered("package seal and T3 seal disagree")


def _package_payload(package: packager.ReleasePackage, *,
                     request_event: journal.JournalEntry,
                     champion_event: journal.JournalEntry,
                     state: champion.SourceTreeState,
                     evidence_class: str) -> dict:
    payload = package.to_dict()
    payload["production_anchor"] = state.incumbent.to_dict()
    payload["sealed_candidate"] = dict(payload["sealed_candidate"])
    payload["sealed_candidate"]["member_candidates"] = list(
        state.champion_members)
    payload["operator_dry_run"] = {
        "module_id": MODULE_ID,
        "request_event_id": request_event.event_id,
        "request_sha256": request_event.payload["request_sha256"],
        "champion_event_id": champion_event.event_id,
        "evidence_class": evidence_class,
        "empirical_claim": False if evidence_class == EVIDENCE_ARCHITECTURE_FIXTURE
        else None,
        "assembled_package_sha256": package.sha256(),
    }
    violations = schemas.validate_release_package(payload)
    if violations:
        raise CloseoutTampered(
            "enriched package is schema-invalid: " + "; ".join(violations))
    if payload.get("state") != STATE_READY:
        raise CloseoutNotReady(
            f"dry-run closeout terminates only at {STATE_READY}, got "
            f"{payload.get('state')!r}")
    return payload


def _terminal_payload(*, state: str, request_sha256: str,
                      request_event_id: str, source_tree: str,
                      exc: Exception) -> dict:
    if state not in TERMINAL_FAILURE_STATES:
        raise ValueError(f"unknown terminal state {state!r}")
    body = {
        "request_sha256": request_sha256,
        "request_event_id": request_event_id,
        "source_tree": source_tree,
        "state": state,
        "failure_class": type(exc).__name__,
        "failure_detail": str(exc) or type(exc).__name__,
    }
    return {"terminal_sha256": schemas.content_hash(body), **body}


class OperatorCloseout:
    """Drive one explicit operator request to a validated dry-run package."""

    def __init__(self, *, book: journal.Journal, loop: sequencer.Sequencer,
                 compiler: ReleaseMaterialCompiler,
                 request_supplier: OperatorRequestSupplier,
                 source_tree: str,
                 evidence_class: str):
        if not isinstance(book, journal.Journal):
            raise TypeError("book must be a journal.Journal")
        if not isinstance(loop, sequencer.Sequencer):
            raise TypeError("loop must be a sequencer.Sequencer")
        if loop.book is not book:
            raise CloseoutTampered("sequencer and closeout must share one journal")
        if not callable(getattr(compiler, "compile", None)):
            raise TypeError("compiler must implement compile()")
        if not callable(getattr(request_supplier, "request", None)):
            raise TypeError("request_supplier must implement request()")
        if source_tree not in schemas.SOURCE_TREES:
            raise ValueError(f"unknown source_tree {source_tree!r}")
        if evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"evidence_class must be one of {sorted(EVIDENCE_CLASSES)}")
        self.book = book
        self.loop = loop
        self.compiler = compiler
        self.request_supplier = request_supplier
        self.source_tree = source_tree
        self.evidence_class = evidence_class

    def run(self) -> CloseoutResult:
        search = self.loop.run()
        snapshot = champion.read_validated_snapshot(self.book)
        state = champion.project_source_tree(
            snapshot,
            self.loop.anchor_provider.current_anchor(self.source_tree))
        champion_event = _latest_champion_event(snapshot, state.source_tree)
        if state.active_champion is None or state.composed_champion is None:
            raise CloseoutUnavailable("the projected champion is not active")
        combined = state.candidates.get(state.composed_champion)
        if combined is None:
            raise CloseoutTampered("active champion has no schema-bound candidate")
        freeze_request = self.request_supplier.request(
            state=state, snapshot=snapshot, champion_event=champion_event)
        if not isinstance(freeze_request, packager.OperatorFreezeRequest):
            raise CloseoutTampered(
                "operator request supplier did not return OperatorFreezeRequest")
        if combined.campaign.get("campaign_id") != freeze_request.campaign_id:
            raise CloseoutTampered(
                "operator request campaign is not the composed candidate campaign")
        if freeze_request.source_tree != state.source_tree:
            raise CloseoutTampered("operator request names another source tree")

        request_payload = _request_payload(
            freeze_request=freeze_request,
            champion_event=champion_event,
            state=state,
            evidence_class=self.evidence_class)
        try:
            request_event = champion.append_idempotent(
                self.book, journal.KIND_OPERATOR_RELEASE_DRY_RUN_REQUESTED,
                request_payload)
        except Exception as exc:
            raise CloseoutTampered(
                f"operator request identity was rewritten: {exc}") from exc

        snapshot = champion.read_validated_snapshot(self.book)
        existing_package = _existing_package(snapshot, request_payload["request_sha256"])
        if existing_package is not None:
            violations = schemas.validate_release_package(existing_package.payload)
            if violations or existing_package.payload.get("state") != STATE_READY:
                raise CloseoutTampered(
                    "recovered package is not a valid ready package: "
                    + "; ".join(violations))
            return CloseoutResult(
                STATE_READY, search, champion_event.event_id, request_event.event_id,
                request_payload["request_sha256"],
                package_event_id=existing_package.event_id,
                package=existing_package.payload,
                detail="recovered already-fsynced ready package")
        existing_terminal = _existing_terminal(
            snapshot, request_payload["request_sha256"])
        if existing_terminal is not None:
            return CloseoutResult(
                existing_terminal.payload["state"], search,
                champion_event.event_id, request_event.event_id,
                request_payload["request_sha256"],
                terminal_event_id=existing_terminal.event_id,
                detail=existing_terminal.payload["failure_detail"])

        try:
            material = self.compiler.compile(
                freeze_request=freeze_request,
                state=state,
                snapshot=snapshot,
                champion_event=champion_event)
            _validate_material(
                material, freeze_request=freeze_request,
                state=state, champion_event=champion_event)
            evaluation = packager.run_release_evaluation(
                material.t3_request, evaluator=t3.T3Runner())
            package = material.package.assemble(
                evaluation=evaluation,
                readiness_report=material.readiness_report)
            payload = _package_payload(
                package, request_event=request_event,
                champion_event=champion_event, state=state,
                evidence_class=self.evidence_class)
            package_event = champion.append_idempotent(
                self.book, journal.KIND_RELEASE_PACKAGE_PREPARED, payload)
        except ResourcePreempted as exc:
            terminal_state = STATE_RESOURCE_PREEMPTED
            failure_exc = exc
        except CloseoutTampered as exc:
            terminal_state = STATE_TAMPER_REFUSED
            failure_exc = exc
        except CloseoutNotReady as exc:
            terminal_state = STATE_FAILED
            failure_exc = exc
        except Exception as exc:
            terminal_state = STATE_FAILED
            failure_exc = exc
        else:
            return CloseoutResult(
                STATE_READY, search, champion_event.event_id, request_event.event_id,
                request_payload["request_sha256"],
                package_event_id=package_event.event_id, package=payload,
                detail="validated dry-run package fsynced; no production action taken")

        terminal = champion.append_idempotent(
            self.book, journal.KIND_OPERATOR_RELEASE_DRY_RUN_TERMINATED,
            _terminal_payload(
                state=terminal_state,
                request_sha256=request_payload["request_sha256"],
                request_event_id=request_event.event_id,
                source_tree=state.source_tree,
                exc=failure_exc))
        return CloseoutResult(
            terminal_state, search, champion_event.event_id,
            request_event.event_id, request_payload["request_sha256"],
            terminal_event_id=terminal.event_id,
            detail=terminal.payload["failure_detail"])


__all__ = [
    "MODULE_ID", "EVIDENCE_ARCHITECTURE_FIXTURE", "EVIDENCE_OPERATOR_SUPPLIED",
    "EVIDENCE_CLASSES", "STATE_READY", "STATE_RESOURCE_PREEMPTED",
    "STATE_TAMPER_REFUSED", "STATE_FAILED", "TERMINAL_FAILURE_STATES",
    "CloseoutError", "CloseoutUnavailable", "CloseoutTampered", "CloseoutNotReady",
    "ResourcePreempted", "PackageAssemblyInputs", "CompiledReleaseMaterial",
    "ReleaseMaterialCompiler", "OperatorRequestSupplier", "CloseoutResult",
    "OperatorCloseout",
]
