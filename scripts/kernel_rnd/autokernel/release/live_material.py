"""Fail-closed live champion to operator dry-run closeout material adapter.

The expensive readiness and T3 observations remain external trusted receipts.
This adapter binds their typed products to the exact current append-only journal
and composed champion, derives the release seal from those bytes, and rebinds a
prevalidated dry-run package template.  It has no clock, process, inference,
build, mutation, transport, freeze, or cutover authority.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

from .. import journal, schemas
from ..controller import champion
from . import closeout, packager, readiness, t3

SCHEMA = "epyc.autokernel.live_release_material_receipt.v1"
AUTHORITY = "operator_triggered_dry_run_only"


class LiveMaterialError(closeout.CloseoutTampered):
    pass


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(child) for child in value)
    return value


def _hash(value: Any) -> str:
    return schemas.content_hash(_jsonable(value))


def bind_sealed_candidate(*, template: t3.SealedCandidate,
                          state: champion.SourceTreeState,
                          candidate: champion.CandidateSnapshot,
                          champion_event: journal.JournalEntry,
                          overlay_receipt_sha256: str) -> t3.SealedCandidate:
    """Bind a producer's full-build seal to exact AutoKernel journal evidence.

    A search campaign may exercise only one backend while a releasable kernel set
    contains several.  Consequently this function preserves additional measured
    backend artifacts from ``template``, but requires the campaign backend to be
    byte-for-byte identical to the journaled artifact.  The returned seal is the
    value for which readiness/T3/package observations must be produced; the live
    compiler never retrofits observations made for a different seal.
    """
    schemas.require.sha256(overlay_receipt_sha256, "overlay_receipt_sha256",
                           error=LiveMaterialError)
    if state.composed_champion != candidate.candidate_id:
        raise LiveMaterialError("sealed candidate is not the composed champion")
    record = candidate.record
    campaign = candidate.campaign
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LiveMaterialError("composed champion carries no built artifacts")
    backend = campaign["backend"]
    expected_identity = {
        "candidate_id": state.composed_champion,
        "source_tree": state.source_tree,
        "candidate_branch": record["worktree"]["branch"],
        "production_base_commit": state.incumbent.commit,
        "candidate_commit": record["worktree"]["source_commit"],
        "evaluator_bundle_sha256": champion_event.payload["evaluator"][
            "bundle_sha256"],
        "scope_manifest_sha256": campaign["scope"][
            "derived_role_manifest_sha256"],
    }
    mismatched = sorted(
        name for name, expected in expected_identity.items()
        if getattr(template, name) != expected)
    if mismatched:
        raise LiveMaterialError(
            "full-build seal differs from journal identity: " + ", ".join(mismatched))
    expected_backend = {
        "binary_sha256": artifacts.get("binary_sha256"),
        "linkage_sha256": artifacts.get("linkage_sha256"),
        "build_dirs": record["build"]["build_dir"],
    }
    for name in ("binary_sha256", "linkage_sha256"):
        schemas.require.sha256(expected_backend[name], f"candidate.artifacts.{name}",
                               error=LiveMaterialError)
    actual_backend = {
        "binary_sha256": template.binary_sha256.get(backend),
        "linkage_sha256": template.linkage_sha256.get(backend),
        "build_dirs": template.build_dirs.get(backend),
    }
    if actual_backend != expected_backend:
        raise LiveMaterialError(
            f"full-build seal does not preserve journaled {backend} artifact")
    if not (template.overlay_present and template.tree_clean
            and template.ancestry_clean):
        raise LiveMaterialError(
            "full-build seal lacks overlay, clean-tree, or clean-ancestry proof")
    evidence_tree = schemas.content_hash({
        "candidate_event_id": candidate.record_event_id,
        "candidate_record_sha256": _hash(candidate.record),
        "evaluation_sha256_by_id": {
            event["event_id"]: _hash(event) for event in candidate.evaluations},
        "champion_event_id": champion_event.event_id,
        "champion_event_sha256": _hash(champion_event.payload),
        "overlay_receipt_sha256": overlay_receipt_sha256,
    })
    unsealed = dataclasses.replace(
        template, evidence_tree_sha256=evidence_tree,
        seal_sha256=schemas.content_hash({"pending": "live-evidence-binding"}))
    seal_sha256 = schemas.content_hash({
        "schema": "epyc.autokernel.live_seal_binding.v1",
        "candidate": {
            key: value for key, value in _jsonable(unsealed).items()
            if key != "seal_sha256"},
    })
    return dataclasses.replace(unsealed, seal_sha256=seal_sha256)


@dataclass(frozen=True)
class LiveMaterialReceipt:
    """Immutable hashes over every external typed input and actual journal view."""

    champion_event_id: str
    champion_event_sha256: str
    candidate_event_id: str
    candidate_record_sha256: str
    evaluation_sha256_by_id: Mapping[str, str]
    readiness_report_sha256: str
    t3_template_sha256: str
    package_template_sha256: str
    overlay_receipt_sha256: str
    overlay_present: bool
    sealed_at: str
    receipt_sha256: str
    schema: str = SCHEMA
    authority: str = AUTHORITY

    def body(self) -> dict:
        return {
            "schema": self.schema, "authority": self.authority,
            "champion_event_id": self.champion_event_id,
            "champion_event_sha256": self.champion_event_sha256,
            "candidate_event_id": self.candidate_event_id,
            "candidate_record_sha256": self.candidate_record_sha256,
            "evaluation_sha256_by_id": dict(self.evaluation_sha256_by_id),
            "readiness_report_sha256": self.readiness_report_sha256,
            "t3_template_sha256": self.t3_template_sha256,
            "package_template_sha256": self.package_template_sha256,
            "overlay_receipt_sha256": self.overlay_receipt_sha256,
            "overlay_present": self.overlay_present,
            "sealed_at": self.sealed_at,
        }

    def validate(self) -> None:
        if self.schema != SCHEMA or self.authority != AUTHORITY:
            raise LiveMaterialError("live material receipt schema/authority differs")
        for name in (
            "champion_event_sha256", "candidate_record_sha256",
            "readiness_report_sha256", "t3_template_sha256",
            "package_template_sha256", "overlay_receipt_sha256", "receipt_sha256",
        ):
            schemas.require.sha256(getattr(self, name), name, error=LiveMaterialError)
        if not self.overlay_present:
            raise LiveMaterialError("candidate overlay is not attested present")
        if self.receipt_sha256 != schemas.content_hash(self.body()):
            raise LiveMaterialError("receipt_sha256 does not bind live material inputs")


def make_receipt(*, champion_event: journal.JournalEntry,
                 candidate: champion.CandidateSnapshot,
                 readiness_report: readiness.ReadinessReport,
                 t3_template: t3.T3Request,
                 package_template: closeout.PackageAssemblyInputs,
                 overlay_receipt_sha256: str, overlay_present: bool,
                 sealed_at: str) -> LiveMaterialReceipt:
    body = {
        "schema": SCHEMA, "authority": AUTHORITY,
        "champion_event_id": champion_event.event_id,
        "champion_event_sha256": _hash(champion_event.payload),
        "candidate_event_id": candidate.record_event_id,
        "candidate_record_sha256": _hash(candidate.record),
        "evaluation_sha256_by_id": {
            event["event_id"]: _hash(event) for event in candidate.evaluations},
        "readiness_report_sha256": _hash(readiness_report),
        "t3_template_sha256": _hash(t3_template),
        "package_template_sha256": _hash(package_template),
        "overlay_receipt_sha256": overlay_receipt_sha256,
        "overlay_present": overlay_present, "sealed_at": sealed_at,
    }
    return LiveMaterialReceipt(
        **{key: body[key] for key in body if key not in {"schema", "authority"}},
        receipt_sha256=schemas.content_hash(body))


class JournalReleaseMaterialCompiler:
    """Rebind trusted release observations to the current composed champion."""

    def __init__(self, *, readiness_report: readiness.ReadinessReport,
                 t3_template: t3.T3Request,
                 package_template: closeout.PackageAssemblyInputs,
                 receipt: LiveMaterialReceipt):
        self.readiness_report = readiness_report
        self.t3_template = t3_template
        self.package_template = package_template
        self.receipt = receipt

    def compile(self, *, freeze_request: packager.OperatorFreezeRequest,
                state: champion.SourceTreeState,
                snapshot: champion.JournalSnapshot,
                champion_event: journal.JournalEntry) -> closeout.CompiledReleaseMaterial:
        self.receipt.validate()
        if state.composed_champion is None:
            raise LiveMaterialError("no composed champion exists")
        combined = state.candidates.get(state.composed_champion)
        if combined is None:
            raise LiveMaterialError("composed champion candidate is absent")
        checks = {
            "champion event id": champion_event.event_id == self.receipt.champion_event_id,
            "champion bytes": _hash(champion_event.payload)
                              == self.receipt.champion_event_sha256,
            "candidate event id": combined.record_event_id
                                  == self.receipt.candidate_event_id,
            "candidate bytes": _hash(combined.record)
                               == self.receipt.candidate_record_sha256,
            "evaluations": {
                event["event_id"]: _hash(event) for event in combined.evaluations
            } == dict(self.receipt.evaluation_sha256_by_id),
            "readiness bytes": _hash(self.readiness_report)
                               == self.receipt.readiness_report_sha256,
            "T3 template bytes": _hash(self.t3_template)
                                 == self.receipt.t3_template_sha256,
            "package template bytes": _hash(self.package_template)
                                     == self.receipt.package_template_sha256,
        }
        failed = sorted(name for name, ok in checks.items() if not ok)
        if failed:
            raise LiveMaterialError("live material receipt mismatch: " + ", ".join(failed))
        record = combined.record
        sealed = bind_sealed_candidate(
            template=self.t3_template.sealed, state=state, candidate=combined,
            champion_event=champion_event,
            overlay_receipt_sha256=self.receipt.overlay_receipt_sha256)
        if sealed != self.t3_template.sealed:
            raise LiveMaterialError(
                "T3 observations were produced for a different evidence seal")
        if self.t3_template.campaign_id != freeze_request.campaign_id:
            raise LiveMaterialError("T3 observations belong to another campaign")
        if self.package_template.sealed.candidate != sealed:
            raise LiveMaterialError("package template was produced for another seal")
        request = self.t3_template
        sealed_release = dataclasses.replace(
            self.package_template.sealed,
            champion_id=f"akch-{state.composed_champion}",
            build_receipt_sha256=schemas.content_hash(record["build"]),
            seal_inputs_ref=f"sha256:{self.receipt.receipt_sha256}",
            sealed_at=self.receipt.sealed_at)
        package = dataclasses.replace(
            self.package_template, freeze_request=freeze_request,
            sealed=sealed_release)
        material = closeout.CompiledReleaseMaterial(
            self.readiness_report, request, package)
        closeout._validate_material(
            material, freeze_request=freeze_request, state=state,
            champion_event=champion_event)
        return material


__all__ = [
    "SCHEMA", "AUTHORITY", "LiveMaterialError", "LiveMaterialReceipt",
    "bind_sealed_candidate", "make_receipt", "JournalReleaseMaterialCompiler",
]
