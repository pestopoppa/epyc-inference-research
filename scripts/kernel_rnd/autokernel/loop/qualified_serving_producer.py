"""Producer-side qualified serving evidence and the live installed authority (AKU-08m2).

The producer seals, beside each canonical claim-grade receipt pair, one content-addressed
``qualified-serving-evidence`` record naming the ORIGINAL direct-runtime-admission control
reference (checked to reopen and be admitted through ``RuntimeAdmission.reopen_admission``
at emission) and the original ``CalibrationReceipt``. ``reopen_row_evidence`` rebuilds the
exact ``QualifiedReceiptRowEvidence`` from that record, and ``install_live_authority`` is the
live registered authority whose evaluator decides v5 rows through ``RegisteredServingDecision``
with the production control owner ``runtime_admission_control``.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from . import experiment_plan as ep
from . import measurement_capture as mc
from . import validation_consumer as vc
from . import validation_semantic_adapter as sa
from .validation_claim_receipt import ClaimGradeReceiptPairReference

SCHEMA = "epyc.autokernel.qualified_serving_evidence.v1"
NAMESPACE = "qualified-serving-evidence"
OWNER_ID = "runtime-admission-serving-decision/v1"


class QualifiedProducerRefused(RuntimeError):
    pass


def control_reference(admission: Any, reference: Mapping[str, Any]) -> dict[str, str]:
    """The original direct-runtime-admission reference, only if it reopens as admitted now."""
    if not isinstance(reference, Mapping) or not {"locator", "sha256"} <= set(reference):
        raise QualifiedProducerRefused("control reference must name a stored direct-runtime-admission")
    ref = {"locator": reference["locator"], "sha256": reference["sha256"]}
    try:
        _pair, admitted = admission.reopen_admission(dict(ref, verified=True))
    except Exception as exc:  # Moved, rewritten or non-rederiving controls never emit.
        raise QualifiedProducerRefused(f"direct runtime admission does not reopen: {exc}") from exc
    if admitted is not True:
        raise QualifiedProducerRefused("direct runtime admission was not admitted")
    return ref


class QualifiedPairProducer:
    """Emits receipt pair + control reference + calibration receipt as one sealed record."""

    def __init__(self, *, evidence_store: mc.ArtifactStore, runtime_admission: Any,
                 adapter: sa.ValidationSemanticAdapter | None = None,
                 source_store: mc.ArtifactStore | None = None,
                 receipt_store: mc.ArtifactStore | None = None) -> None:
        if not callable(getattr(runtime_admission, "reopen_admission", None)):
            raise QualifiedProducerRefused("producer requires the direct runtime admission owner")
        self.evidence_store, self.runtime_admission = evidence_store, runtime_admission
        self.adapter, self.source_store, self.receipt_store = adapter, source_store, receipt_store

    def produce_pair(self, *, anchor_source: Mapping[str, str], candidate_source: Mapping[str, str],
                     anchor_measurement_id: str, candidate_measurement_id: str,
                     admissible_view_digest: str) -> ClaimGradeReceiptPairReference:
        """Seal both canonical receipts from exact v2 Journal sources (registered projector)."""
        if self.adapter is None or self.source_store is None or self.receipt_store is None:
            raise QualifiedProducerRefused("canonical receipt production needs the installed adapter/stores")
        anchor, candidate = (self.adapter.produce_receipt(
            source_store=self.source_store, source_locator=src["locator"], source_sha256=src["sha256"],
            receipt_store=self.receipt_store) for src in (anchor_source, candidate_source))
        return ClaimGradeReceiptPairReference(anchor, candidate, anchor_measurement_id,
                                              candidate_measurement_id, admissible_view_digest)

    def emit(self, pair: ClaimGradeReceiptPairReference, *, admission_reference: Mapping[str, Any],
             calibration: ep.CalibrationReceipt) -> dict[str, str]:
        if type(pair) is not ClaimGradeReceiptPairReference or type(calibration) is not ep.CalibrationReceipt:
            raise QualifiedProducerRefused("qualified evidence needs a typed receipt pair and calibration receipt")
        body = {"schema": SCHEMA, "pair": pair.to_dict(), "calibration": calibration.to_dict(),
                "control_reference": control_reference(self.runtime_admission, admission_reference)}
        stored = self.evidence_store.write(NAMESPACE, body)
        return {"locator": stored.locator, "sha256": stored.sha256}


def reopen_row_evidence(store: mc.ArtifactStore, reference: Mapping[str, str],
                        row_id: str) -> vc.QualifiedReceiptRowEvidence:
    """Rebuild the exact row evidence the producer sealed (never re-derived from config)."""
    body = mc._plain(store.read(reference["locator"], reference["sha256"]))
    if (not isinstance(body, Mapping) or set(body) != {"schema", "pair", "calibration", "control_reference"}
            or body["schema"] != SCHEMA or store.verify(NAMESPACE, body).sha256 != reference["sha256"]):
        raise QualifiedProducerRefused("qualified serving evidence record differs")
    return vc.QualifiedReceiptRowEvidence(
        row_id, ClaimGradeReceiptPairReference.from_dict(body["pair"]),
        calibration=ep.CalibrationReceipt.from_dict(body["calibration"]),
        control_reference=dict(body["control_reference"]))


def install_live_authority(adapter: sa.ValidationSemanticAdapter, transaction_verifier_id: str, *,
                           source_store: mc.ArtifactStore, receipt_store: mc.ArtifactStore,
                           runtime_admission: Any, estimators: Mapping[str, Callable[..., Any]],
                           calibration_rule_id: str, calibration_rule: Callable[..., Any],
                           native_validator: Any = None):
    """The live registered authority: same evaluator, production calibration/control owners."""
    if not callable(getattr(runtime_admission, "reopen_admission", None)):
        raise QualifiedProducerRefused("live authority requires the direct runtime admission owner")
    decision = sa.RegisteredServingDecision(
        OWNER_ID, estimators, calibration_rule_id, calibration_rule,
        sa.runtime_admission_control(runtime_admission), fixture_only=False)
    return adapter.registered_authority(transaction_verifier_id, source_store=source_store,
                                        receipt_store=receipt_store, native_validator=native_validator,
                                        serving_decision=decision)
