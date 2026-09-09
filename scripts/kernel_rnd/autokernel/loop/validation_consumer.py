"""Consume native serving evidence into immutable candidate validation batches.

This module is deliberately a consumer, not a grader.  It validates the existing
native carrier and calibration contracts before entering the controller's candidate
transaction boundary.  Real advancement additionally needs a registered semantic
authority both here and in :class:`CandidateTransactions`; an identifier alone never
supplies that capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .. import schemas
from . import candidate_manifest as cm
from . import candidate_transactions as ct
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import native_capture_control as nc


class ValidationConsumerError(RuntimeError):
    """Evidence or candidate state cannot safely make the requested transition."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _batch_identity(value: cm.ValidationBatch) -> tuple[Any, ...]:
    return (value.batch_id, value.candidate_manifest_digest,
            value.comparator_manifest_digest, value.row_set_digest,
            value.expected_validated_predecessor, value.launch_integration_tip,
            value.accounted_keeps_since_gate, value.accounted_threshold_generation,
            value.required_loo_keep_ids, value.required_row_ids,
            tuple(item.row_id for item in value.rows))


def _build_dso_set_digest(build: cm.BuildIdentity) -> str:
    """Canonical planned-serving loader-name plus content-digest identity."""
    return schemas.content_hash([
        {"load_name": Path(item.path).name, "sha256": item.sha256}
        for item in build.dsos])


@dataclass(frozen=True)
class SemanticDecision:
    """Typed result returned by a registered owning-protocol adapter."""

    source_grade: str
    trace_grade: str
    intended_use: str
    permitted: bool
    reasons: tuple[str, ...] = ()


SemanticEvaluator = Callable[[ep.ExperimentPlan, Mapping[str, Any]], SemanticDecision]


@dataclass(frozen=True)
class RegisteredSemanticAuthority:
    """An installed capability, not a declared verifier name.

    ``fixture_only`` makes test authority visible in results and prevents callers from
    mistaking a hermetic transaction-wiring proof for production claim authority.
    """

    authority_id: str
    transaction_verifier_id: str
    evaluate: SemanticEvaluator
    fixture_only: bool = False
    claim_grade_verifier: Any = None

    def __post_init__(self) -> None:
        if not self.authority_id or not self.transaction_verifier_id \
                or not callable(self.evaluate):
            raise ValidationConsumerError("semantic authority must be an installed capability")


@dataclass(frozen=True)
class RowDebt:
    row_id: str
    required: bool
    status: str
    prerequisites: tuple[str, ...]


@dataclass(frozen=True)
class BatchAssembly:
    batch: cm.ValidationBatch
    row_set: cm.RequiredRowSet
    loo_plans: Mapping[str, cm.LOOPlan]
    debt: tuple[RowDebt, ...]
    trigger: str


@dataclass(frozen=True)
class NativeRowEvidence:
    row_id: str
    anchor_measurement_id: str
    anchor_payload: Mapping[str, Any]
    candidate_measurement_id: str
    candidate_payload: Mapping[str, Any]
    calibration: ep.CalibrationReceipt


@dataclass(frozen=True)
class HistoricalReceiptRowEvidence:
    """Historical canonical receipts, not a live native-capture capability."""

    row_id: str
    claim_grade_pair: Any

    def __post_init__(self):
        from .validation_claim_receipt import ClaimGradeReceiptPairReference
        if (not isinstance(self.row_id, str) or not self.row_id
                or type(self.claim_grade_pair) is not ClaimGradeReceiptPairReference):
            raise ValidationConsumerError("historical row requires a concrete receipt pair")
        object.__setattr__(self, "claim_grade_pair",
                           ClaimGradeReceiptPairReference.from_dict(self.claim_grade_pair.to_dict()))


class ValidationConsumer:
    """Bridge real native captures to the existing durable candidate transactions."""

    def __init__(self, *, transactions: ct.CandidateTransactions,
                 native_validator: nc.NativeCaptureValidator,
                 evidence_store: mc.ArtifactStore,
                 semantic_authorities: Mapping[str, RegisteredSemanticAuthority] | None = None,
                 registered_estimators: Mapping[str, Callable[[ep.CalibrationReceipt], float]] | None = None,
                 calibration_rule_id: str | None = None,
                 calibration_rule: Callable[[ep.CalibrationReceipt, ep.ExperimentPlan], bool | str] | None = None) -> None:
        if not isinstance(transactions, ct.CandidateTransactions):
            raise ValidationConsumerError("transactions must be CandidateTransactions")
        if not isinstance(native_validator, nc.NativeCaptureValidator):
            raise ValidationConsumerError("native validator must be NativeCaptureValidator")
        if not isinstance(evidence_store, mc.ArtifactStore):
            raise ValidationConsumerError("evidence store must be ArtifactStore")
        self.transactions = transactions
        self.native_validator = native_validator
        self.evidence_store = evidence_store
        self.semantic_authorities = dict(semantic_authorities or {})
        self.registered_estimators = dict(registered_estimators or {})
        self.calibration_rule_id = calibration_rule_id
        self.calibration_rule = calibration_rule
        self.calibration_cache = ep.CalibrationCache()

    def assemble_due_batch(self, *, request_id: str, candidate: cm.CandidateManifest,
                           comparator: cm.CandidateManifest,
                           row_set: cm.RequiredRowSet) -> BatchAssembly:
        """Freeze the current due tip and persist the batch through the real transaction API."""
        snapshot = self.transactions.inspect()
        state = cm.CandidateState.from_dict(snapshot["state"])
        candidate, comparator = candidate.validated(), comparator.validated()
        row_set = cm.RequiredRowSet.from_dict(row_set.to_dict())
        if not state.gate_due:
            raise ValidationConsumerError("validation gate is not due")
        if state.integration_tip != candidate.manifest_digest:
            raise ValidationConsumerError("candidate is not the current integration tip")
        trigger = ("both" if state.keeps_since_gate >= 4
                   and state.threshold_generation > state.covered_threshold_generation
                   else "cadence" if state.keeps_since_gate >= 4 else "gain")
        batch_id = "validation-" + _digest({
            "candidate": candidate.manifest_digest,
            "comparator": comparator.manifest_digest,
            "rows": row_set.row_set_digest,
            "predecessor": state.validated_candidate,
            "keeps": state.keeps_since_gate,
            "generation": state.threshold_generation,
        })[:24]
        batch = cm.ValidationBatch(
            batch_id, candidate.manifest_digest, comparator.manifest_digest,
            row_set.row_set_digest, state.validated_candidate, state.integration_tip,
            state.keeps_since_gate, state.threshold_generation,
            tuple(item.keep_id for item in candidate.keeps),
            tuple(item.row_id for item in row_set.rows if item.required),
            tuple(cm.ValidationRowState(item.row_id, "pending", None, None)
                  for item in row_set.rows)).validated()
        loo_plans = {item.keep_id: cm.plan_loo(candidate, item.keep_id)
                     for item in candidate.keeps}
        debt = tuple(self._initial_debt(row, loo_plans) for row in row_set.rows)
        self.transactions.start_batch(request_id=request_id, batch=batch, row_set=row_set,
                                      candidate=candidate, comparator=comparator)
        return BatchAssembly(batch, row_set, loo_plans, debt, trigger)

    def _initial_debt(self, row: cm.ValidationRow,
                      loo_plans: Mapping[str, cm.LOOPlan]) -> RowDebt:
        missing = ["native_anchor_measurement", "native_candidate_measurement",
                   "applicable_calibration", "shared_claim_grade",
                   "owning_protocol_intended_use_authority",
                   "instrument_identity_unknown"]
        blocked_loo = sorted(key for key, value in loo_plans.items()
                             if value.status != "planned")
        missing.extend(f"identifiable_loo:{key}" for key in blocked_loo)
        return RowDebt(row.row_id, row.required, "pending", tuple(missing))

    def record_native_row(self, *, request_id: str, assembly: BatchAssembly,
                          candidate: cm.CandidateManifest,
                          comparator: cm.CandidateManifest,
                          evidence: NativeRowEvidence | HistoricalReceiptRowEvidence,
                          authority_id: str | None = None) -> cm.ValidationRowState:
        """Verify, seal, and journal one exact row; all evidence I/O precedes locking."""
        # Pin caller-carried objects to the cached authoritative projection before
        # reading native artifacts. CandidateTransactions.record_row intentionally
        # accepts only a batch id and row set, so this consumer owns the wider join.
        candidate, comparator = candidate.validated(), comparator.validated()
        frozen = self._active_batch(assembly.batch.batch_id)
        supplied_batch = assembly.batch.validated()
        if _batch_identity(frozen) != _batch_identity(supplied_batch):
            raise ValidationConsumerError("assembly batch differs from active frozen batch")
        if (assembly.row_set.row_set_digest != frozen.row_set_digest
                or candidate.manifest_digest != frozen.candidate_manifest_digest
                or comparator.manifest_digest != frozen.comparator_manifest_digest):
            raise ValidationConsumerError(
                "assembly candidate/comparator/row-set differs from active frozen batch")
        rows = {item.row_id: item for item in assembly.row_set.rows}
        row = rows.get(evidence.row_id)
        if row is None:
            raise ValidationConsumerError("evidence row is outside the frozen row set")
        if type(evidence) is HistoricalReceiptRowEvidence:
            body = self._historical_body(assembly.batch, row, assembly.row_set,
                candidate, comparator, evidence.claim_grade_pair, authority_id)
            stored = self.evidence_store.write("candidate-validation-row", body)
            receipt = cm.RowReceipt.from_dict({"schema": cm.RECEIPT_SCHEMA,
                "batch_id": assembly.batch.batch_id,
                "candidate_manifest_digest": candidate.manifest_digest,
                "comparator_manifest_digest": comparator.manifest_digest,
                "row_set_digest": assembly.row_set.row_set_digest, "row_id": row.row_id,
                "native_evidence_ref": stored.locator, "native_evidence_digest": stored.sha256,
                "intended_use": "validate", "use_disposition": "policy_undefined"})
            decision = body["semantic_decision"]
            status = "prerequisite_missing" if decision["source_grade"] == "Unavailable" else "inconclusive"
            state = cm.ValidationRowState(row.row_id, status, "; ".join(body["prerequisites"]), receipt)
            self.transactions.record_row(request_id=request_id, batch_id=assembly.batch.batch_id,
                                         row_set=assembly.row_set, row_state=state)
            return state
        anchor = self.native_validator.validate(
            evidence.anchor_measurement_id, evidence.anchor_payload)
        candidate_capture = self.native_validator.validate(
            evidence.candidate_measurement_id, evidence.candidate_payload)
        anchor_payload, candidate_payload = anchor.payload(), candidate_capture.payload()
        anchor_carrier, candidate_carrier = anchor_payload["carrier"], candidate_payload["carrier"]
        if anchor_carrier["arm"] != "anchor" or candidate_carrier["arm"] != "candidate":
            raise ValidationConsumerError("native evidence does not contain exact anchor/candidate arms")
        if anchor_carrier["plan"] != candidate_carrier["plan"]:
            raise ValidationConsumerError("native evidence arms use different frozen plans")
        if (anchor_carrier["lineage_id"] != candidate_carrier["lineage_id"]
                or anchor_carrier["comparison_identities"]
                != candidate_carrier["comparison_identities"]):
            raise ValidationConsumerError(
                "native evidence arms use different comparison lineage/identity")
        plan = ep.ExperimentPlan.from_dict(anchor_carrier["plan"])
        prerequisites = self._binding_debt(row, plan, candidate, comparator)
        if anchor.status != "measurement" or candidate_capture.status != "measurement":
            prerequisites.append("complete_native_measurement_pair")
        calibration = ep.calibration_applicability(
            evidence.calibration, plan,
            registered_estimators=self.registered_estimators,
            registered_rule_id=self.calibration_rule_id,
            applicability_rule=self.calibration_rule, cache=self.calibration_cache)
        if calibration.status != "applicable":
            prerequisites.extend(f"calibration:{reason}" for reason in calibration.reasons)

        # Exercise the existing policy guard. It intentionally remains undefined for
        # claim-gating plans until an owner-specific semantic adapter resolves the use.
        structural_use = _structural_use(plan, candidate_carrier)
        if structural_use.status == "refused":
            prerequisites.extend(f"structural_use:{reason}" for reason in structural_use.reasons)

        authority = self.semantic_authorities.get(authority_id or "")
        decision = None
        if authority is None:
            prerequisites.append("registered_semantic_authority")
        else:
            decision = authority.evaluate(plan, {
                "anchor": anchor_payload, "candidate": candidate_payload,
                "row": row.to_dict(), "calibration": evidence.calibration.to_dict(),
                "structural_use": structural_use.to_dict(),
            })
            if not isinstance(decision, SemanticDecision):
                raise ValidationConsumerError("semantic adapter returned an untyped decision")
            if (decision.intended_use != "validate_production" or not decision.permitted
                    or decision.source_grade != "Witnessed"
                    or decision.trace_grade != "Attested"):
                prerequisites.extend(decision.reasons or ("semantic authority refused",))
        # v1 carries an instrument label, not a captured loaded-instrument identity.
        # Only visibly fixture-scoped wiring may proceed through this known absence.
        if authority is None or not authority.fixture_only:
            prerequisites.append("instrument_identity_unknown")
        if prerequisites:
            status = "prerequisite_missing"
            state = cm.ValidationRowState(row.row_id, status,
                                          "; ".join(sorted(set(prerequisites))), None)
        else:
            bundle = {
                "schema": "epyc.autokernel.validation_native_pair.v1",
                "batch": assembly.batch.to_dict(), "row": row.to_dict(),
                "candidate": candidate.to_dict(),
                "comparator": comparator.to_dict(),
                "row_set": assembly.row_set.to_dict(),
                "plan_digest": plan.digest,
                "native": {
                    "anchor": self._native_closure(anchor, anchor_payload),
                    "candidate": self._native_closure(
                        candidate_capture, candidate_payload),
                },
                "calibration": {
                    "receipt": evidence.calibration.to_dict(),
                    "digest": evidence.calibration.digest,
                    "plan_reference": plan.calibration_ref,
                },
                "semantic_authority_id": authority.authority_id,
                "semantic_decision": decision.__dict__,
                "fixture_only_authority": authority.fixture_only,
            }
            stored = self.evidence_store.write("candidate-validation-row", bundle)
            receipt = cm.RowReceipt.from_dict({
                "schema": cm.RECEIPT_SCHEMA, "batch_id": assembly.batch.batch_id,
                "candidate_manifest_digest": assembly.batch.candidate_manifest_digest,
                "comparator_manifest_digest": assembly.batch.comparator_manifest_digest,
                "row_set_digest": assembly.batch.row_set_digest, "row_id": row.row_id,
                "native_evidence_ref": stored.locator,
                "native_evidence_digest": stored.sha256,
                "intended_use": "validate", "use_disposition": "permitted"})
            state = cm.ValidationRowState(row.row_id, "passed", None, receipt)
        self.transactions.record_row(request_id=request_id, batch_id=assembly.batch.batch_id,
                                     row_set=assembly.row_set, row_state=state)
        return state

    @staticmethod
    def _native_closure(capture: nc.ValidatedNativeCapture,
                        payload: Mapping[str, Any]) -> dict[str, Any]:
        carrier = payload["carrier"]
        return {
            "measurement_id": capture.measurement_id,
            "payload_digest": capture.payload_digest,
            "payload": dict(payload),
            "carrier_artifact": dict(payload["artifact"]),
            "raw_artifacts": [
                {"artifact_digest": item["document"]["artifact_digest"],
                 "stored": dict(item["stored"])}
                for item in carrier["raw_artifacts"]],
        }

    def reopen_row(self, receipt: cm.RowReceipt) -> Mapping[str, Any]:
        """Reopen and revalidate one sealed row without journal/history scanning."""
        receipt = cm.RowReceipt.from_dict(receipt.to_dict())
        locator = receipt.native_evidence_ref
        if Path(locator).name != locator:
            raise ValidationConsumerError("sealed row locator is not a store leaf")
        try:
            bundle = self.evidence_store.read(locator, receipt.native_evidence_digest)
            verified = self.evidence_store.verify("candidate-validation-row", bundle)
        except (ValueError, mc.CaptureError, mc.SecureRuntimeError) as exc:
            raise ValidationConsumerError("sealed row artifact cannot be reopened") from exc
        if verified.locator != locator or verified.sha256 != receipt.native_evidence_digest:
            raise ValidationConsumerError("sealed row locator/digest differs from receipt")
        return self._validate_reopened_bundle(bundle, receipt)

    def verify_row_receipt(self, receipt: cm.RowReceipt, row: cm.ValidationRow,
                           candidate: cm.CandidateManifest,
                           comparator: cm.CandidateManifest) -> Mapping[str, Any]:
        """Public replay verifier for the exact typed candidate transaction join."""
        row = cm.ValidationRow.from_dict(row.to_dict())
        candidate, comparator = candidate.validated(), comparator.validated()
        bundle = self.reopen_row(receipt)
        if (cm.ValidationRow.from_dict(bundle["row"]) != row
                or cm.CandidateManifest.from_dict(bundle["candidate"]) != candidate
                or cm.CandidateManifest.from_dict(bundle["comparator"]) != comparator):
            raise ValidationConsumerError(
                "replayed row/candidate/comparator differs from verifier arguments")
        return bundle

    def _validate_reopened_bundle(self, bundle: Mapping[str, Any],
                                  receipt: cm.RowReceipt) -> Mapping[str, Any]:
        if isinstance(bundle, Mapping) and bundle.get("schema") == "epyc.autokernel.validation_native_pair.v3":
            return self._reopen_historical_row(bundle, receipt)
        fields = {"schema", "batch", "row", "candidate", "comparator", "row_set",
                  "plan_digest", "native", "calibration", "semantic_authority_id",
                  "semantic_decision", "fixture_only_authority"}
        if (not isinstance(bundle, Mapping) or set(bundle) != fields
                or bundle.get("schema") != "epyc.autokernel.validation_native_pair.v1"):
            raise ValidationConsumerError("sealed row has unsupported or open schema")
        decision = bundle.get("semantic_decision")
        if (not isinstance(bundle.get("semantic_authority_id"), str)
                or not bundle["semantic_authority_id"]
                or type(bundle.get("fixture_only_authority")) is not bool
                or not isinstance(decision, Mapping)
                or set(decision) != {"source_grade", "trace_grade", "intended_use",
                                     "permitted", "reasons"}
                or not all(isinstance(decision.get(field), str) and decision[field]
                           for field in ("source_grade", "trace_grade", "intended_use"))
                or type(decision.get("permitted")) is not bool
                or not isinstance(decision.get("reasons"), (list, tuple))
                or not all(isinstance(reason, str) and reason
                           for reason in decision.get("reasons", ()))
                or decision.get("source_grade") != "Witnessed"
                or decision.get("trace_grade") != "Attested"
                or decision.get("intended_use") != "validate_production"
                or decision.get("permitted") is not True
                or bundle.get("fixture_only_authority") is not True):
            raise ValidationConsumerError("sealed semantic decision is malformed")
        try:
            batch = cm.ValidationBatch.from_dict(bundle["batch"])
            row = cm.ValidationRow.from_dict(bundle["row"])
            candidate = cm.CandidateManifest.from_dict(bundle["candidate"])
            comparator = cm.CandidateManifest.from_dict(bundle["comparator"])
            row_set = cm.RequiredRowSet.from_dict(bundle["row_set"])
            cm._validate_batch_obligations(batch, row_set, candidate, comparator)
        except (KeyError, TypeError, ValueError, cm.CandidateError,
                cm.TransitionError) as exc:
            raise ValidationConsumerError("sealed candidate/batch closure is invalid") from exc
        rows = {item.row_id: item for item in row_set.rows}
        if (rows.get(row.row_id) != row
                or receipt.batch_id != batch.batch_id
                or receipt.row_id != row.row_id
                or receipt.candidate_manifest_digest != candidate.manifest_digest
                or receipt.comparator_manifest_digest != comparator.manifest_digest
                or receipt.row_set_digest != row_set.row_set_digest
                or receipt.intended_use != "validate"
                or receipt.use_disposition != "permitted"
                or not batch.matches_receipt(receipt)
                or batch.candidate_manifest_digest != candidate.manifest_digest
                or batch.comparator_manifest_digest != comparator.manifest_digest
                or batch.row_set_digest != row_set.row_set_digest):
            raise ValidationConsumerError("receipt identity differs from sealed frozen closure")
        native = bundle.get("native")
        calibration = bundle.get("calibration")
        if not isinstance(native, Mapping) or set(native) != {"anchor", "candidate"} \
                or not isinstance(calibration, Mapping) \
                or set(calibration) != {"receipt", "digest", "plan_reference"}:
            raise ValidationConsumerError("sealed row closure is malformed")
        restored: dict[str, tuple[nc.ValidatedNativeCapture, Mapping[str, Any]]] = {}
        for arm in ("anchor", "candidate"):
            item = native[arm]
            if not isinstance(item, Mapping) or set(item) != {
                    "measurement_id", "payload_digest", "payload",
                    "carrier_artifact", "raw_artifacts"}:
                raise ValidationConsumerError("sealed native arm closure is malformed")
            try:
                validated = self.native_validator.validate(
                    item.get("measurement_id"), item.get("payload"))
            except (mc.CaptureError, TypeError, ValueError) as exc:
                raise ValidationConsumerError(
                    f"sealed {arm} native artifacts cannot be revalidated") from exc
            payload = validated.payload()
            expected_raw = [
                {"artifact_digest": row["document"]["artifact_digest"],
                 "stored": dict(row["stored"])}
                for row in payload["carrier"]["raw_artifacts"]]
            if (validated.payload_digest != item.get("payload_digest")
                    or mc._plain(payload["artifact"])
                    != mc._plain(item.get("carrier_artifact"))
                    or mc._plain(expected_raw)
                    != mc._plain(item.get("raw_artifacts"))):
                raise ValidationConsumerError("sealed native arm closure changed")
            restored[arm] = (validated, payload)
        anchor_carrier = restored["anchor"][1]["carrier"]
        candidate_carrier = restored["candidate"][1]["carrier"]
        if (anchor_carrier["arm"] != "anchor"
                or candidate_carrier["arm"] != "candidate"
                or anchor_carrier["plan"] != candidate_carrier["plan"]
                or anchor_carrier["lineage_id"] != candidate_carrier["lineage_id"]
                or anchor_carrier["comparison_identities"]
                != candidate_carrier["comparison_identities"]):
            raise ValidationConsumerError("sealed native pair identity is inconsistent")
        restored_plan = ep.ExperimentPlan.from_dict(anchor_carrier["plan"])
        if restored_plan.digest != bundle.get("plan_digest"):
            raise ValidationConsumerError("sealed plan digest is inconsistent")
        if any(capture.status != "measurement" for capture, _payload in restored.values()):
            raise ValidationConsumerError(
                "sealed native pair is diagnostic rather than complete measurements")
        structural_use = _structural_use(restored_plan, candidate_carrier)
        if structural_use.status == "refused":
            raise ValidationConsumerError(
                "sealed native pair is structurally ineligible: "
                + "; ".join(structural_use.reasons))
        binding_debt = self._binding_debt(row, restored_plan, candidate, comparator)
        if binding_debt:
            raise ValidationConsumerError(
                "sealed row semantics are inconsistent: " + "; ".join(binding_debt))
        try:
            restored_calibration = ep.CalibrationReceipt.from_dict(
                mc._plain(calibration.get("receipt")))
        except (TypeError, ValueError) as exc:
            raise ValidationConsumerError("sealed calibration cannot be revalidated") from exc
        if (restored_calibration.digest != calibration.get("digest")
                or calibration.get("plan_reference") != restored_calibration.digest
                or restored_plan.calibration_ref != calibration.get("plan_reference")):
            raise ValidationConsumerError("sealed calibration closure changed")
        disposition = ep.calibration_applicability(
            restored_calibration, restored_plan,
            registered_estimators=self.registered_estimators,
            registered_rule_id=self.calibration_rule_id,
            applicability_rule=self.calibration_rule, cache=self.calibration_cache)
        if disposition.status != "applicable":
            raise ValidationConsumerError("sealed calibration is no longer applicable")
        return bundle

    def _historical_registration(self, authority_id):
        from .validation_semantic_adapter import RegisteredClaimGradeVerifier, RegisteredServingSemanticEvaluator
        authority = self.semantic_authorities.get(authority_id or "")
        if (type(authority) is not RegisteredSemanticAuthority or authority.fixture_only
                or type(authority.claim_grade_verifier) is not RegisteredClaimGradeVerifier
                or type(authority.evaluate) is not RegisteredServingSemanticEvaluator
                or authority.evaluate.claim_grade_verifier is not authority.claim_grade_verifier
                or authority.evaluate.adapter.projection is not authority.claim_grade_verifier.projection):
            raise ValidationConsumerError("historical row requires the exact installed semantic registration")
        return authority

    def _historical_body(self, batch, row, row_set, candidate, comparator, pair, authority_id):
        """Consume historical facts; this path cannot pass or advance a row."""
        from . import observation_binding as ob
        authority = self._historical_registration(authority_id)
        actual = authority.claim_grade_verifier.reopen_pair(pair)
        debt = self._binding_debt(row, actual.plan, candidate, comparator)
        if debt:
            raise ValidationConsumerError("historical receipt/validation row identity differs: " + "; ".join(debt))
        decision = authority.evaluate(actual.plan, {"claim_grade_pair": pair})
        if (type(decision) is not SemanticDecision or decision.permitted
                or decision.intended_use != "validate_production" or not decision.reasons):
            raise ValidationConsumerError("historical receipt cannot authorize validation")
        body = {"schema": "epyc.autokernel.validation_native_pair.v3",
            "batch": batch.to_dict(), "row": row.to_dict(), "candidate": candidate.to_dict(),
            "comparator": comparator.to_dict(), "row_set": row_set.to_dict(),
            "plan_digest": actual.plan.digest, "claim_grade_pair": pair.to_dict(),
            "semantic_authority_id": authority.authority_id,
            "semantic_decision": ob._plain(decision.__dict__),
            "source_identity": ob._plain(actual.anchor["source_identity"]),
            "final_view_digest": actual.view.view_digest,
            "prerequisites": sorted(set(decision.reasons))}
        return body

    def _reopen_historical_row(self, bundle, receipt):
        from .validation_claim_receipt import ClaimGradeReceiptPairReference
        fields = {"schema", "batch", "row", "candidate", "comparator", "row_set", "plan_digest",
            "claim_grade_pair", "semantic_authority_id", "semantic_decision", "source_identity",
            "final_view_digest", "prerequisites"}
        if set(bundle) != fields:
            raise ValidationConsumerError("historical row bundle has missing/unknown fields")
        try:
            batch = cm.ValidationBatch.from_dict(bundle["batch"])
            row = cm.ValidationRow.from_dict(bundle["row"])
            candidate = cm.CandidateManifest.from_dict(bundle["candidate"])
            comparator = cm.CandidateManifest.from_dict(bundle["comparator"])
            row_set = cm.RequiredRowSet.from_dict(bundle["row_set"])
            pair = ClaimGradeReceiptPairReference.from_dict(bundle["claim_grade_pair"])
            cm._validate_batch_obligations(batch, row_set, candidate, comparator)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValidationConsumerError("historical row original frozen closure is malformed") from exc
        if (not batch.matches_receipt(receipt) or receipt.row_id != row.row_id
                or receipt.intended_use != "validate" or receipt.use_disposition != "policy_undefined"
                or row not in row_set.rows or batch.row_set_digest != row_set.row_set_digest
                or receipt.candidate_manifest_digest != candidate.manifest_digest
                or receipt.comparator_manifest_digest != comparator.manifest_digest):
            raise ValidationConsumerError("historical row receipt differs from original binding")
        expected = self._historical_body(batch, row, row_set, candidate, comparator,
                                         pair, bundle["semantic_authority_id"])
        if mc._plain(bundle) != expected:
            raise ValidationConsumerError("historical row differs from complete receipt replay")
        return bundle

    @staticmethod
    def _binding_debt(row: cm.ValidationRow, plan: ep.ExperimentPlan,
                      candidate: cm.CandidateManifest,
                      comparator: cm.CandidateManifest) -> list[str]:
        candidate_targets = {item.target_revision_digest: item for item in candidate.targets}
        target = candidate_targets.get(row.target_revision_digest)
        controls = {item.target_revision_digest: item for item in comparator.targets}
        control = controls.get(row.control_target_revision_digest)
        if target is None or control is None:
            return ["candidate_or_comparator_target_missing"]
        ca, an = dict(plan.candidate_identity), dict(plan.anchor_identity)
        candidate_builds = {item.execution_digest: item for item in candidate.builds}
        comparator_builds = {item.execution_digest: item for item in comparator.builds}
        candidate_build = candidate_builds.get(row.candidate_build_digest)
        comparator_build = comparator_builds.get(row.control_build_digest)
        if candidate_build is None or comparator_build is None:
            return ["candidate_or_comparator_build_missing"]
        expected = (
            plan.instrument_class == "serving", plan.category == row.category,
            plan.protocol_ref == row.protocol_id,
            ca.get("backend") == row.backend, an.get("backend") == row.backend,
            ca.get("resolved_execution_digest") == row.candidate_recipe_digest,
            an.get("resolved_execution_digest") == row.control_recipe_digest,
            ca.get("model_digest") == row.model_digest == an.get("model_digest"),
            ca.get("drafter_digest") == row.drafter_digest == an.get("drafter_digest"),
            ca.get("workload_digest") == row.workload_digest == an.get("workload_digest"),
            ca.get("executable_digest") == candidate_build.executable.sha256,
            an.get("executable_digest") == comparator_build.executable.sha256,
            ca.get("dso_set_digest") == _build_dso_set_digest(candidate_build),
            an.get("dso_set_digest") == _build_dso_set_digest(comparator_build),
        )
        return [] if all(expected) else ["frozen_row_plan_or_measured_artifact_identity_mismatch"]

    def complete(self, *, request_id: str, assembly: BatchAssembly) -> cm.ValidationBatch:
        batch = self._active_batch(assembly.batch.batch_id)
        self.transactions.complete_batch(request_id=request_id, batch=batch)
        return batch

    def advance(self, *, request_id: str, assembly: BatchAssembly,
                candidate: cm.CandidateManifest, comparator: cm.CandidateManifest,
                authority_id: str, loo_results: Mapping[str, cm.LOOResult]) -> Mapping[str, Any]:
        authority = self.semantic_authorities.get(authority_id)
        if authority is None:
            raise cm.TrustedVerificationRequired("registered semantic authority is not connected")
        snapshot = self.transactions.inspect()
        state = cm.CandidateState.from_dict(snapshot["state"])
        if state.integration_tip != assembly.batch.launch_integration_tip:
            raise ValidationConsumerError("integration tip/generation changed before advancement")
        completed = next((item for item in state.completed_batches
                          if item.batch_id == assembly.batch.batch_id), None)
        if completed is None:
            raise ValidationConsumerError("batch is not completed")
        return self.transactions.advance_validated(
            request_id=request_id, verifier_id=authority.transaction_verifier_id,
            batch=completed, row_set=assembly.row_set, candidate=candidate,
            comparator=comparator, loo_plans=assembly.loo_plans,
            loo_results=loo_results)

    def _active_batch(self, batch_id: str) -> cm.ValidationBatch:
        state = cm.CandidateState.from_dict(self.transactions.inspect()["state"])
        batch = next((item for item in state.active_batches if item.batch_id == batch_id), None)
        if batch is None:
            raise ValidationConsumerError("batch is not active")
        return batch


def plan_view(carrier: Mapping[str, Any]) -> Mapping[str, Any]:
    value = carrier.get("admissible_view")
    if not isinstance(value, Mapping):
        raise ValidationConsumerError("carrier has no admissible unit view")
    return value


def _structural_use(plan: ep.ExperimentPlan,
                    carrier: Mapping[str, Any]) -> ep.UseDisposition:
    value = plan_view(carrier)
    view = ep.AdmissibleUnitView(
        **{key: item for key, item in value.items() if key != "selected_rows"},
        selected_rows=tuple(ep.RawUnit.from_dict(item)
                            for item in value["selected_rows"]))
    return ep.eligibility(
        plan, view, plan.intended_use, current_epoch=plan.epoch)


__all__ = ["BatchAssembly", "NativeRowEvidence", "HistoricalReceiptRowEvidence", "RegisteredSemanticAuthority",
           "RowDebt", "SemanticDecision", "ValidationConsumer", "ValidationConsumerError"]
