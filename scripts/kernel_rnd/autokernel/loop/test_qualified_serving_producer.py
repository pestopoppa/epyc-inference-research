"""AKU-08m2: producer emits control refs + CalibrationReceipts beside its receipt pairs; the live
installed authority decides v5 rows through RegisteredServingDecision.

FIXTURES (labelled): the measurement pair is the hermetic relabel used by AKU-08m1, and the
direct runtime admission owner is a fixture with RuntimeAdmission.reopen_admission's exact
StoredArtifact reference contract over a real ArtifactStore record.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from . import campaign_control as control
from . import candidate_transactions as ct
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import observation_binding as ob
from . import qualified_serving_producer as Q
from . import validation_consumer as vc
from . import validation_semantic_adapter as sa
from .measurement_capture import StoredArtifact
from .test_campaign_control import _resolved
from .test_candidate_manifest import _state
from .test_feed_root_projection import actual_final as actual_final  # noqa: F401 (fixture)
from .test_validation_final_receipt import _qualified_service, real as real  # noqa: F401 (fixture)


class FixtureRuntimeAdmission:
    """FIXTURE direct runtime admission owner (production: RuntimeAdmission)."""

    def __init__(self, store, plan, admitted=True):
        self.store, self.admitted = store, admitted
        self.digests = (dict(plan.anchor_identity)["resolved_execution_digest"],
                        dict(plan.candidate_identity)["resolved_execution_digest"])

    def reopen_admission(self, reference):
        artifact = StoredArtifact(**reference)  # same contract as runtime_admission._read
        body = ob._plain(self.store.read(artifact.locator, artifact.sha256))
        if self.store.verify("direct-runtime-admission", body) != artifact:
            raise RuntimeError("original direct-runtime-admission namespace differs")
        pair = SimpleNamespace(anchor=SimpleNamespace(execution_digest=self.digests[0]),
                               candidate=SimpleNamespace(execution_digest=self.digests[1]))
        return pair, self.admitted and body["admitted"]


@contextmanager
def _live(real, tmp_path, s, runtime, *, fresh=True):
    controller = control.CampaignController(_resolved(), tmp_path / "live-service")
    controller.__enter__()
    source, receipts = mc.ArtifactStore(real["source"].root), mc.ArtifactStore(real["receipts"].root)
    store = mc.ArtifactStore(tmp_path / "live-rows")
    try:
        manager = ct.CandidateTransactions(controller, git_backend=s.git)
        adapter = sa.ValidationSemanticAdapter(sa.PinnedRootProjection(real["root"], source_closure=real["closure"]))
        authority = Q.install_live_authority(
            adapter, "current-semantic/v2", source_store=source, receipt_store=receipts,
            runtime_admission=runtime, estimators={"median.v1": lambda r: r.value},
            calibration_rule_id="fixture-calibration/v1", calibration_rule=lambda _r, _p: True)
        assert type(authority.evaluate.serving_decision) is sa.RegisteredServingDecision
        validator = nc.NativeCaptureValidator(binding=nc.NativeCaptureBinding(
            controller.resolved.campaign_id, controller.config_digest,
            controller.config_generation, "unused-live-consumer", 1), store=source)
        consumer = vc.ValidationConsumer(transactions=manager, native_validator=validator,
            evidence_store=store, semantic_authorities={authority.authority_id: authority})
        assembly = None
        if fresh:
            manager.initialize(request_id="initialize", state=_state(s.comparator), manifest=s.comparator)
            manager.integrate(request_id="integrate", previous=s.comparator, candidate=s.candidate,
                              threshold_signal=True)
            assembly = consumer.assemble_due_batch(request_id="assemble", candidate=s.candidate,
                                                   comparator=s.comparator, row_set=s.rows)

        def record(evidence):
            return consumer.record_native_row(request_id="qualified", assembly=assembly,
                candidate=s.candidate, comparator=s.comparator, authority_id=authority.authority_id,
                evidence=evidence)
        yield SimpleNamespace(consumer=consumer, manager=manager, record=record)
    finally:
        store.close()
        receipts.close()
        source.close()
        controller.close()


def test_producer_emits_control_and_calibration_and_live_authority_passes_v5(real, tmp_path, monkeypatch):
    with _qualified_service(real, tmp_path, monkeypatch) as s:
        runtime_store = mc.ArtifactStore(tmp_path / "runtime")
        evidence_store = mc.ArtifactStore(tmp_path / "producer-evidence")
        try:
            admission = runtime_store.write("direct-runtime-admission", {"admitted": True}).to_dict()
            runtime = FixtureRuntimeAdmission(runtime_store, s.plan)
            producer = Q.QualifiedPairProducer(evidence_store=evidence_store, runtime_admission=runtime)
            sealed = producer.emit(real["pair"], admission_reference=admission, calibration=s.calibration)
            evidence = Q.reopen_row_evidence(evidence_store, sealed, s.production.row_id)
            assert evidence.claim_grade_pair == real["pair"] and evidence.calibration == s.calibration
            assert evidence.control_reference == {k: admission[k] for k in ("locator", "sha256")}
            with _live(real, tmp_path, s, runtime) as h:
                state = h.record(evidence)
                assert (state.status, state.receipt.use_disposition) == ("passed", "permitted")
                bundle = ob._plain(h.consumer.reopen_row(state.receipt))
                assert bundle["schema"] == vc.QUALIFIED_V5 and bundle["prerequisites"] == []
                facts = bundle["serving_decision"]
                assert facts["fixture_only"] is False and facts["control"]["admitted"] is True
                receipt = state.receipt
            with _live(real, tmp_path, s, runtime, fresh=False) as h:  # restart replays identically
                assert ob._plain(h.consumer.reopen_row(receipt)) == bundle
            # The live control owner re-decides: a no-longer-admitted original refuses replay.
            with _live(real, tmp_path, s, FixtureRuntimeAdmission(runtime_store, s.plan, admitted=False),
                       fresh=False) as h:
                with pytest.raises(vc.ValidationConsumerError):
                    h.consumer.reopen_row(receipt)
        finally:
            evidence_store.close()
            runtime_store.close()


def test_producer_refuses_unadmitted_or_unreopenable_controls_and_tampered_records(real, tmp_path, monkeypatch):
    with _qualified_service(real, tmp_path, monkeypatch) as s:
        runtime_store = mc.ArtifactStore(tmp_path / "runtime")
        evidence_store = mc.ArtifactStore(tmp_path / "producer-evidence")
        try:
            refused = runtime_store.write("direct-runtime-admission", {"admitted": False}).to_dict()
            producer = Q.QualifiedPairProducer(evidence_store=evidence_store,
                                               runtime_admission=FixtureRuntimeAdmission(runtime_store, s.plan))
            with pytest.raises(Q.QualifiedProducerRefused, match="not admitted"):
                producer.emit(real["pair"], admission_reference=refused, calibration=s.calibration)
            with pytest.raises(Q.QualifiedProducerRefused, match="does not reopen"):
                producer.emit(real["pair"], admission_reference={"locator": "missing.json", "sha256": "0" * 64},
                              calibration=s.calibration)
            with pytest.raises(Q.QualifiedProducerRefused, match="typed"):
                producer.emit(real["pair"].to_dict(), admission_reference=refused, calibration=s.calibration)
            with pytest.raises(Q.QualifiedProducerRefused, match="admission owner"):
                Q.QualifiedPairProducer(evidence_store=evidence_store, runtime_admission=object())
            other = evidence_store.write(Q.NAMESPACE, {"schema": "other"}).to_dict()
            with pytest.raises(Q.QualifiedProducerRefused, match="differs"):
                Q.reopen_row_evidence(evidence_store, other, s.production.row_id)
        finally:
            evidence_store.close()
            runtime_store.close()


def test_production_control_owner_accepts_the_sealed_two_field_reference(tmp_path):
    store = mc.ArtifactStore(tmp_path / "runtime")
    try:
        ref = store.write("direct-runtime-admission", {"admitted": True}).to_dict()
        plan = SimpleNamespace(anchor_identity={"resolved_execution_digest": "a" * 64},
                               candidate_identity={"resolved_execution_digest": "c" * 64})
        admission = sa.runtime_admission_control(FixtureRuntimeAdmission(store, plan))(
            {"locator": ref["locator"], "sha256": ref["sha256"]})
        assert admission == sa.ControlAdmission(True, "a" * 64, "c" * 64)
    finally:
        store.close()
