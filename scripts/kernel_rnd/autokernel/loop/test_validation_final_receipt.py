"""Actual original final-v3 semantic receipts: diagnostic, immutable and replayable."""
from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from . import experiment_plan as ep
from . import candidate_manifest as cm
from . import candidate_transactions as ct
from . import campaign_control as control
from . import feed_runtime as feed
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import native_parent_receipt_replay as replay
from . import observation_binding as ob
from . import resolved_recipe as rr
from . import validation_consumer as vc
from . import validation_claim_receipt as cr
from . import validation_projection_source as current
from . import validation_semantic_adapter as sa
from .test_evidence_feed import _root_repo
from .test_feed_root_projection import actual_final as actual_final
from .test_campaign_control import _resolved
from .test_candidate_manifest import _state
from .test_candidate_transactions import FakeGitBackend
from .test_validation_consumer import _candidate_chain, _row


@pytest.fixture(scope="module")
def real(actual_final, tmp_path_factory):
    controller, _, native = actual_final
    root = _root_repo()
    source = mc.ArtifactStore(controller / "unified-native-artifacts")
    receipts = mc.ArtifactStore(tmp_path_factory.mktemp("current-semantic-receipts"))
    events = {event["payload"]["carrier"]["arm"]: event for event in native
              if event["payload"]["schema"] == current.CAPTURE_SCHEMA}
    assert set(events) == {"anchor", "candidate"}
    instrument = events["anchor"]["payload"]["carrier"]["loaded_instrument"]["artifact"]
    original = mc._plain(source.read(instrument["locator"], instrument["sha256"]))
    expected = original["used_constants"]["producer_source_closure"]
    pins = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in feed.ROOT_SOURCES}
    closure = sa.ProjectionSourceClosure("80f4f60b5de46825b0e2488bced4a085bd8f58ad", pins, expected)
    projection = sa.PinnedRootProjection(root, source_closure=closure)
    refs, bodies = {}, {}
    for arm, event in events.items():
        artifact = source.write("original-semantic-source-event", event)
        refs[arm] = projection.produce_receipt(source_store=source, source_locator=artifact.locator,
            source_sha256=artifact.sha256, receipt_store=receipts)
        bodies[arm] = projection.reopen_receipt(reference=refs[arm], source_store=source,
                                              receipt_store=receipts)
    view = events["anchor"]["payload"]["carrier"]["admissible_view"]
    pair = cr.ClaimGradeReceiptPairReference(refs["anchor"], refs["candidate"],
        events["anchor"]["record_id"], events["candidate"]["record_id"], view["view_digest"])
    try:
        yield {"root": root, "source": source, "receipts": receipts, "closure": closure,
            "projection": projection, "pair": pair, "bodies": bodies, "events": events,
            "controller": controller}
    finally:
        source.close()
        receipts.close()


def test_actual_current_receipt_preserves_original_provenance_and_no_tuple(real):
    for body in real["bodies"].values():
        assert body["schema"] == current.RECEIPT_SCHEMA
        assert set(body["projection"]) == {"status", "reason"}
        assert body["projection"]["status"] == "diagnostic"
        assert body["authority_scope"] == "final_pinned_source"  # Source proof, not science.
        assert body["native_provenance"]["producer_source_closure"] == real["closure"].expected_native_producer_source
        assert body["source_identity"] == real["projection"]._installed_source_identity
        with pytest.raises(TypeError):
            body["projection"]["reason"] = "invented"
    verifier = sa.RegisteredClaimGradeVerifier(real["projection"], real["source"], real["receipts"])
    actual = verifier.reopen_pair(real["pair"])
    assert not actual.view.complete and actual.view.selected_rows == ()
    with pytest.raises(sa.SemanticAdapterError, match="qualified measurement"):
        verifier.verify(actual.plan, actual.view, real["pair"])


def test_foreign_plan_diagnostic_refuses_before_unavailable_result(real):
    plan = ep.ExperimentPlan.from_dict(real["events"]["anchor"]["payload"]["carrier"]["plan"])
    foreign = ep.ExperimentPlan.from_dict(plan.to_dict() | {"plan_id": "foreign-plan"})
    with pytest.raises(sa.SemanticAdapterError, match="different plan"):
        sa.ValidationSemanticAdapter(real["projection"]).evaluate_receipt_pair(foreign,
            anchor=real["pair"].anchor, candidate=real["pair"].candidate,
            source_store=real["source"], receipt_store=real["receipts"])


def test_two_concrete_installations_cannot_cross_register(real):
    other = sa.PinnedRootProjection(real["root"], source_closure=real["closure"])
    verifier = sa.RegisteredClaimGradeVerifier(other, real["source"], real["receipts"])
    with pytest.raises(sa.SemanticAdapterError, match="concrete installed"):
        sa.RegisteredServingSemanticEvaluator(sa.ValidationSemanticAdapter(real["projection"]), verifier)


def test_loaded_receipt_function_change_cannot_relabel_installation(real, monkeypatch):
    monkeypatch.setattr(current, "native_sources_bound", lambda *_: True)
    with pytest.raises(ValueError, match="implementation/configuration changed"):
        current.source_identity(real["projection"])


@pytest.mark.parametrize("field", ["finalizer", "selected_scientific_adapter"])
def test_missing_original_final_source_never_is_final_pinned(real, field):
    provenance = ob._plain(real["bodies"]["anchor"]["native_provenance"])
    provenance["parent_final_source_identity"].pop(field)
    assert not current.native_sources_bound(provenance, real["closure"].expected_native_producer_source)


def test_incomplete_matching_final_source_still_is_not_final_pinned(real):
    provenance = ob._plain(real["bodies"]["anchor"]["native_provenance"])
    expected = ob._plain(real["closure"].expected_native_producer_source)
    selected = expected["scientific_adapters"]["correctness"]
    final = selected["owning_source_pins"]["final_trial_source"]
    final["owner"][0]["identity"]["implementation_status"] = "unproven"
    provenance["producer_source_closure"] = expected
    provenance["parent_final_source_identity"] = {"finalizer": final,
                                                   "selected_scientific_adapter": selected}
    assert not current.native_sources_bound(provenance, expected)


@pytest.mark.parametrize("mutation", ["tuple", "grade", "unknown", "id", "instrument", "view"])
def test_resealed_receipt_mutations_cannot_be_replayed(real, mutation):
    body = ob._plain(real["bodies"]["anchor"])
    if mutation == "tuple":
        body["projection"]["claim_tuple"] = {}
    elif mutation == "grade":
        body["projection"]["source_grade"] = "Witnessed"
    elif mutation == "unknown":
        body["unknown"] = True
    elif mutation == "id":
        body["native_binding"]["measurement_id"] = "e" * 64
    elif mutation == "instrument":
        body["native_binding"]["instrument_identity_sha256"] = "e" * 64
    else:
        body["native_binding"]["final_view_digest"] = "e" * 64
    body.pop("receipt_id")
    body["receipt_id"] = "claim-grade-" + sa.schemas.content_hash(body)[:24]
    artifact = real["receipts"].write("canonical-claim-grade-receipt", body)
    reference = cr.ClaimGradeReceiptReference(body["receipt_id"], artifact.locator, artifact.sha256)
    with pytest.raises(sa.SemanticAdapterError):
        real["projection"].reopen_receipt(reference=reference,
            source_store=real["source"], receipt_store=real["receipts"])


def test_fresh_projection_replays_without_live_native_authority(real, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("historical replay attempted to recreate live native authority")
    monkeypatch.setattr(nc.NativeCaptureValidator, "prevalidate", forbidden)
    monkeypatch.setattr(nc.NativeCaptureValidator, "validate", forbidden)
    monkeypatch.setattr(replay.IssuedNativeEvidenceRegistry, "__init__", forbidden)
    projection = sa.PinnedRootProjection(real["root"], source_closure=real["closure"])
    for arm, reference in (("anchor", real["pair"].anchor), ("candidate", real["pair"].candidate)):
        assert projection.reopen_receipt(reference=reference, source_store=real["source"],
            receipt_store=real["receipts"]) == real["bodies"][arm]


def test_actual_validation_consumer_records_and_restarts_without_live_issuance(real, tmp_path, monkeypatch):
    """Real transaction WAL, actual native artifacts; candidate metadata is hermetic.

    The original run is CANDIDATE, not OPTIMUM: retain it only as an optional
    diagnostic seed row. The actual required production row remains pending.
    """
    plan = ep.ExperimentPlan.from_dict(real["events"]["anchor"]["payload"]["carrier"]["plan"])
    library = real["controller"].parent / "candidate/build/bin/libggml-base.so"
    dso = rr.ArtifactDigest.from_dict({"schema": rr.ARTIFACT_SCHEMA, "role": "dso",
        "path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}, role="dso")
    comparator, chain = _candidate_chain(plan, 1, dsos={"anchor": (dso,), "candidate": (dso,)})
    candidate = chain[-1]
    production, _ = _row(plan, candidate, comparator)
    diagnostic = cm.ValidationRow.from_dict(production.to_dict() | {
        "row_id": "original-candidate-diagnostic", "row_kind": "seed", "required": False,
        "category": plan.category,
        "instrument_digest": dict(plan.anchor_identity)["instrument_identity_sha256"]})
    rows = cm.RequiredRowSet.from_dict(cm.RequiredRowSet("actual-final-v3", (diagnostic, production)).to_dict())
    assert plan.category == "CANDIDATE" and production.category == "OPTIMUM"
    assert vc.ValidationConsumer._binding_debt(diagnostic, plan, candidate, comparator) == []
    git = FakeGitBackend()
    validators = []
    recorded = original_bundle = original_snapshot = None

    def forbidden(*_args, **_kwargs):
        pytest.fail("historical consumer attempted live validation or original issuer restoration")

    monkeypatch.setattr(nc.NativeCaptureValidator, "validate", forbidden)
    monkeypatch.setattr(nc.NativeCaptureValidator, "prevalidate", forbidden)
    monkeypatch.setattr(replay.IssuedNativeEvidenceRegistry, "__init__", forbidden)
    for restarted in (False, True):
        controller = control.CampaignController(_resolved(), tmp_path / "consumer-service")
        controller.__enter__()
        source = mc.ArtifactStore(real["source"].root)
        receipts = mc.ArtifactStore(real["receipts"].root)
        evidence_store = mc.ArtifactStore(tmp_path / "consumer-evidence")
        try:
            manager = ct.CandidateTransactions(controller, git_backend=git)
            projection = sa.PinnedRootProjection(real["root"], source_closure=real["closure"])
            adapter = sa.ValidationSemanticAdapter(projection)
            authority = adapter.registered_authority(transaction_verifier_id="current-semantic/v2",
                source_store=source, receipt_store=receipts)
            # A new unused live validator satisfies the existing constructor; it
            # has neither a current fence nor any original issuance registry.
            validator = nc.NativeCaptureValidator(binding=nc.NativeCaptureBinding(
                controller.resolved.campaign_id, controller.config_digest,
                controller.config_generation, "unused-historical-consumer", 1), store=source)
            validators.append(validator)
            assert validator.fence_provider is validator.parent_receipt_replayer is None
            consumer = vc.ValidationConsumer(transactions=manager, native_validator=validator,
                evidence_store=evidence_store, semantic_authorities={authority.authority_id: authority})
            if not restarted:
                manager.initialize(request_id="initialize", state=_state(comparator), manifest=comparator)
                manager.integrate(request_id="integrate", previous=comparator,
                    candidate=candidate, threshold_signal=True)
                assembly = consumer.assemble_due_batch(request_id="assemble", candidate=candidate,
                    comparator=comparator, row_set=rows)
                # The same native plan cannot be laundered into the OPTIMUM row.
                with pytest.raises(vc.ValidationConsumerError, match="identity differs"):
                    consumer.record_native_row(request_id="wrong-category", assembly=assembly,
                        candidate=candidate, comparator=comparator,
                        evidence=vc.HistoricalReceiptRowEvidence(production.row_id, real["pair"]),
                        authority_id=authority.authority_id)
                recorded = consumer.record_native_row(request_id="diagnostic", assembly=assembly,
                    candidate=candidate, comparator=comparator,
                    evidence=vc.HistoricalReceiptRowEvidence(diagnostic.row_id, real["pair"]),
                    authority_id=authority.authority_id)
                assert recorded.status == "prerequisite_missing"
                assert recorded.receipt.use_disposition == "policy_undefined"
                original_bundle = consumer.reopen_row(recorded.receipt)
                original_snapshot = manager.inspect()
                assert original_snapshot["state"]["validated_candidate"] is None
                states = {row["row_id"]: row for row in original_snapshot["state"]["active_batches"][0]["rows"]}
                assert states[production.row_id]["status"] == "pending"
            else:
                assert validators[0] is not validator
                assert manager.inspect() == original_snapshot
            count = len(controller._journal.read_all())
            for _ in range(2):
                reopened = consumer.verify_row_receipt(recorded.receipt, diagnostic, candidate, comparator)
                assert reopened == original_bundle
                assert reopened["semantic_decision"]["source_grade"] == "Unavailable"
                assert reopened["semantic_decision"]["permitted"] is False
                assert reopened["source_identity"] == real["bodies"]["anchor"]["source_identity"]
                assert reopened["claim_grade_pair"] == real["pair"].to_dict()
            assert len(controller._journal.read_all()) == count
            assert manager.inspect()["state"]["validated_candidate"] is None
            changed = ob._plain(original_bundle)
            changed["semantic_decision"]["permitted"] = True
            stored = evidence_store.write("candidate-validation-row", changed)
            forged = replace(recorded.receipt, native_evidence_ref=stored.locator,
                             native_evidence_digest=stored.sha256)
            with pytest.raises(vc.ValidationConsumerError, match="complete receipt replay"):
                consumer.reopen_row(forged)
        finally:
            evidence_store.close()
            receipts.close()
            source.close()
            controller.close()
