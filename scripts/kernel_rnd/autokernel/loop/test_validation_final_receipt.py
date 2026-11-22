"""Actual original final-v3 semantic receipts: diagnostic, immutable and replayable."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from types import SimpleNamespace

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
    # Final-receipt semantics own the seven-file v2 closure independently of
    # the feed's current v3 profile projection closure.
    pins = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in feed.ROOT_SOURCES_V2}
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
                # v4: the warranted diagnostic pair is reprojected and stops at qualification.
                assert reopened["schema"] == vc.HISTORICAL_V4
                assert list(reopened["eligibility"]["receipt_codes"]) == ["claim_grade_receipt_unqualified"]
                assert reopened["eligibility"]["disposition"]["status"] == "refused"
                assert "eligibility:claim_grade_receipt_unqualified" in list(reopened["prerequisites"])
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


def test_reopened_pair_rederives_matching_producer_provenance_warrant(real):
    actual = sa.RegisteredClaimGradeVerifier(real["projection"], real["source"],
                                             real["receipts"]).reopen_pair(real["pair"])
    assert actual.producer_provenance == current.ProducerProvenance(True, ())


def test_receipt_scope_cannot_disagree_with_reopened_provenance(real, monkeypatch):
    original = current.ProducerProvenance
    monkeypatch.setattr(current, "ProducerProvenance", lambda *_args: original(False, ()))
    with pytest.raises(sa.SemanticAdapterError, match="authority scope differs"):
        sa.RegisteredClaimGradeVerifier(real["projection"], real["source"],
                                        real["receipts"]).reopen_pair(real["pair"])


@pytest.mark.parametrize("change,code", [("bytes", "producer_implementation_differs"),
                                         ("slot", "configured_verifier_slot_differs")])
def test_installed_closure_change_reopens_typed_compatibility_only(real, tmp_path, monkeypatch, change, code):
    expected = ob._plain(real["closure"].expected_native_producer_source)
    first = expected["measurement_capture"]["callables"][0]["identity"]
    if change == "bytes":
        first["implementation_sha256"] = "e" * 64
    else:
        expected["native_validator"]["observation_verifiers"]["purpose"] = dict(first)
    closure = sa.ProjectionSourceClosure(real["closure"].root_commit,
                                         dict(real["closure"].root_source_sha256), expected)
    projection = sa.PinnedRootProjection(real["root"], source_closure=closure)
    receipts = mc.ArtifactStore(tmp_path / "receipts")
    try:
        refs = {}
        for arm, ref in (("anchor", real["pair"].anchor), ("candidate", real["pair"].candidate)):
            event = real["receipts"].read(ref.locator, ref.sha256)["source_event"]
            refs[arm] = projection.produce_receipt(source_store=real["source"],
                source_locator=event["locator"], source_sha256=event["sha256"], receipt_store=receipts)
        pair = replace(real["pair"], anchor=refs["anchor"], candidate=refs["candidate"])
        verifier = sa.RegisteredClaimGradeVerifier(projection, real["source"], receipts)
        actual = verifier.reopen_pair(pair)
        assert {actual.anchor["authority_scope"], actual.candidate["authority_scope"]} == {"compatibility_only"}
        assert not actual.producer_provenance.warranted and code in actual.producer_provenance.reasons
        # A measurement-shaped pair (hermetic relabel of the reopened bodies) still
        # never becomes eligible: typed Unavailable decision, typed verify refusal.
        def measured(body):
            return ob._plain(body) | {"projection": {"status": "measurement",
                "source_grade": "Witnessed", "trace_grade": "Attested"}}
        shaped = replace(actual, anchor=measured(actual.anchor), candidate=measured(actual.candidate),
                         view=replace(actual.view, complete=True))
        monkeypatch.setattr(sa.RegisteredClaimGradeVerifier, "reopen_pair", lambda _self, _pair: shaped)
        decision = sa.ValidationSemanticAdapter(projection).evaluate(
            actual.plan, {"claim_grade_pair": pair}, claim_grade_verifier=verifier)
        assert (decision.source_grade, decision.permitted) == ("Unavailable", False)
        assert code in decision.reasons
        # The relabelled view is hermetic, so bypass only the structural view re-check.
        monkeypatch.setattr(ep, "_validated_view", lambda _plan, view: view)
        with pytest.raises(sa.ProducerProvenanceRefused) as refused:
            verifier.verify(actual.plan, shaped.view, pair)
        assert code in refused.value.reasons
    finally:
        receipts.close()


# AKU-08c: typed receipt eligibility discharges only ep.eligibility's missing-adapter refusal.
GRADER_MISSING = "shared registered ClaimTuple grader adapter is not integrated"


def _refused(code):
    return f"registered claim-grade receipt refused: {code}"


def _eligibility(plan, view, verifier, pair, use=None):
    typed = sa.RegisteredClaimGradeEligibility(verifier, pair)
    return ep.eligibility(plan, view, use or plan.intended_use, current_epoch=plan.epoch,
                          registered_claim_grade=typed)


def _complete(**overrides):
    """Hermetic structural plan/view (fixture): the reopened pair is relabelled onto it."""
    from .test_experiment_plan import complete_rows, plan_dict
    plan = ep.ExperimentPlan.from_dict(plan_dict(**({"intended_use": "validate", "n": 1,
                                                     "calibration_ref": "cal-1"} | overrides)))
    return plan, ep.admissible_units(plan, complete_rows(plan))


def _shaped(real, plan, view, warrant):
    def measured(body):
        return ob._freeze(ob._plain(body) | {"projection": {"status": "measurement",
            "source_grade": "Witnessed", "trace_grade": "Attested"}})
    return sa.ReopenedClaimReceiptPair(plan, view, measured(real["bodies"]["anchor"]),
        measured(real["bodies"]["candidate"]), {}, {}, warrant)


def test_actual_warranted_pair_reprojects_at_eligibility_and_stops_at_qualification(real):
    plan = ep.ExperimentPlan.from_dict(real["events"]["anchor"]["payload"]["carrier"]["plan"])
    view = sa.RegisteredClaimGradeVerifier(real["projection"], real["source"], real["receipts"]
                                           ).reopen_pair(real["pair"]).view
    verifier = sa.RegisteredClaimGradeVerifier(real["projection"], real["source"], real["receipts"])
    typed = sa.RegisteredClaimGradeEligibility(verifier, real["pair"])
    # Provenance is warranted, so the typed code is the diagnostic (no ClaimTuple) gate.
    assert typed.discharge(plan, view) == ("claim_grade_receipt_unqualified",)
    result = ep.eligibility(plan, view, "validate_production", current_epoch=plan.epoch,
                            registered_claim_grade=typed)
    assert result.status == "refused"


@pytest.mark.parametrize("use,status,reasons", [
    ("validate", "permitted", ()),
    ("validate_production", "policy_undefined",
     ("deployment use requires the owning qualified serving decision",))])
def test_warranted_measurement_receipt_discharges_only_the_missing_adapter_gate(
        real, monkeypatch, use, status, reasons):
    plan, view = _complete(intended_use=use, instrument="serving", category="OPTIMUM")
    assert ep.eligibility(plan, view, use, current_epoch=plan.epoch).reasons == (GRADER_MISSING,)
    shaped = _shaped(real, plan, view, current.ProducerProvenance(True, ()))
    monkeypatch.setattr(sa.RegisteredClaimGradeVerifier, "reopen_pair", lambda _self, _pair: shaped)
    verifier = sa.RegisteredClaimGradeVerifier(real["projection"], real["source"], real["receipts"])
    result = _eligibility(plan, view, verifier, real["pair"])
    assert (result.status, result.reasons) == (status, reasons)
    # Actor-supplied grades remain inert data, byte-for-byte legacy behaviour.
    for actor in ("PASS", {"grade": "Witnessed"}, shaped):
        legacy = ep.eligibility(plan, view, use, current_epoch=plan.epoch, registered_claim_grade=actor)
        assert legacy.reasons == (GRADER_MISSING,)


@pytest.mark.parametrize("case,codes", [
    ("unwarranted", ("producer_provenance_unwarranted", "producer_source_closure_absent")),
    ("unqualified", ("claim_grade_receipt_unqualified",)),
    ("foreign_plan", ("claim_grade_receipt_unverified",)),
    ("unregistered", ("claim_grade_receipt_unregistered",))])
def test_unwarranted_or_unbound_receipts_keep_refusing_with_typed_codes(real, monkeypatch, case, codes):
    plan, view = _complete()
    warrant = current.ProducerProvenance(case != "unwarranted", () if case != "unwarranted"
                                         else ("producer_source_closure_absent",))
    shaped = _shaped(real, plan, view, warrant)
    if case == "unqualified":
        shaped = replace(shaped, anchor=real["bodies"]["anchor"])
    elif case == "foreign_plan":
        other, other_view = _complete(intended_use="bank")
        shaped = replace(shaped, plan=other, view=other_view)
    monkeypatch.setattr(sa.RegisteredClaimGradeVerifier, "reopen_pair", lambda _self, _pair: shaped)
    verifier = (object() if case == "unregistered" else
                sa.RegisteredClaimGradeVerifier(real["projection"], real["source"], real["receipts"]))
    result = _eligibility(plan, view, verifier, real["pair"])
    assert result.status == "refused"
    assert result.reasons == tuple(_refused(code) for code in codes) + (GRADER_MISSING,)


def test_receipts_from_a_different_projector_pin_refuse_typed(real, tmp_path):
    expected = ob._plain(real["closure"].expected_native_producer_source)
    expected["measurement_capture"]["callables"][0]["identity"]["implementation_sha256"] = "e" * 64
    closure = sa.ProjectionSourceClosure(real["closure"].root_commit,
                                         dict(real["closure"].root_source_sha256), expected)
    other = sa.PinnedRootProjection(real["root"], source_closure=closure)
    receipts = mc.ArtifactStore(tmp_path / "other-pin-receipts")
    try:
        refs = {}
        for arm, ref in (("anchor", real["pair"].anchor), ("candidate", real["pair"].candidate)):
            event = real["receipts"].read(ref.locator, ref.sha256)["source_event"]
            refs[arm] = other.produce_receipt(source_store=real["source"],
                source_locator=event["locator"], source_sha256=event["sha256"], receipt_store=receipts)
        pair = replace(real["pair"], anchor=refs["anchor"], candidate=refs["candidate"])
        plan = ep.ExperimentPlan.from_dict(real["events"]["anchor"]["payload"]["carrier"]["plan"])
        mine = sa.RegisteredClaimGradeVerifier(other, real["source"], receipts)
        view = mine.reopen_pair(pair).view
        # Its own installation: compatibility-only, typed provenance refusal.
        assert sa.RegisteredClaimGradeEligibility(mine, pair).discharge(plan, view) == (
            "producer_provenance_unwarranted", "producer_implementation_differs")
        # The installed (published) projector pin refuses the foreign receipt identity.
        installed = sa.RegisteredClaimGradeVerifier(real["projection"], real["source"], receipts)
        assert sa.RegisteredClaimGradeEligibility(installed, pair).discharge(plan, view) == (
            "claim_grade_receipt_unverified",)
    finally:
        receipts.close()


def test_receipts_are_immutable_and_tampering_refuses_at_eligibility(real, tmp_path):
    root = tmp_path / "immutable-receipts"
    receipts = mc.ArtifactStore(root)
    try:
        event = real["receipts"].read(real["pair"].anchor.locator, real["pair"].anchor.sha256)["source_event"]
        kwargs = dict(source_store=real["source"], source_locator=event["locator"],
                      source_sha256=event["sha256"], receipt_store=receipts)
        first = real["projection"].produce_receipt(**kwargs)
        assert real["projection"].produce_receipt(**kwargs) == first == real["pair"].anchor
        cand = real["receipts"].read(real["pair"].candidate.locator, real["pair"].candidate.sha256)
        second = real["projection"].produce_receipt(**(kwargs | {
            "source_locator": cand["source_event"]["locator"], "source_sha256": cand["source_event"]["sha256"]}))
        plan = ep.ExperimentPlan.from_dict(real["events"]["anchor"]["payload"]["carrier"]["plan"])
        verifier = sa.RegisteredClaimGradeVerifier(real["projection"], real["source"], receipts)
        view = verifier.reopen_pair(real["pair"]).view
        path = root / first.locator
        raw = path.read_bytes()
        path.chmod(0o600)
        path.write_bytes(raw.replace(b'"diagnostic"', b'"measurement"', 1))
        # The content-addressed store refuses to rewrite the name with different bytes.
        with pytest.raises(mc.CaptureError, match="different bytes"):
            real["projection"].produce_receipt(**kwargs)
        assert sa.RegisteredClaimGradeEligibility(verifier, replace(real["pair"], candidate=second)
                                                  ).discharge(plan, view) == ("claim_grade_receipt_unverified",)
    finally:
        receipts.close()


def test_eligibility_reprojection_is_deterministic_across_hash_seeds(real):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    script = """import json, sys
from pathlib import Path
from autokernel.loop import experiment_plan as ep, measurement_capture as mc
from autokernel.loop import validation_claim_receipt as cr, validation_semantic_adapter as sa
from autokernel import schemas
args = json.loads(sys.argv[1])
closure = sa.ProjectionSourceClosure(args["commit"], args["pins"], args["expected"])
projection = sa.PinnedRootProjection(Path(args["root"]), source_closure=closure)
source, receipts = mc.ArtifactStore(Path(args["source"])), mc.ArtifactStore(Path(args["receipts"]))
pair = cr.ClaimGradeReceiptPairReference.from_dict(args["pair"])
verifier = sa.RegisteredClaimGradeVerifier(projection, source, receipts)
actual = verifier.reopen_pair(pair)
codes = sa.RegisteredClaimGradeEligibility(verifier, pair).discharge(actual.plan, actual.view)
print(json.dumps({"bodies": [schemas.content_hash(mc._plain(b)) for b in (actual.anchor, actual.candidate)],
                  "codes": list(codes), "warranted": actual.producer_provenance.warranted}))
"""
    args = json.dumps({"commit": real["closure"].root_commit, "root": str(real["root"]),
        "pins": dict(real["closure"].root_source_sha256),
        "expected": ob._plain(real["closure"].expected_native_producer_source),
        "source": str(real["source"].root), "receipts": str(real["receipts"].root),
        "pair": real["pair"].to_dict()})
    package_root = str(Path(sa.__file__).resolve().parents[2])
    rows = [json.loads(subprocess.check_output([sys.executable, "-c", script, args], text=True,
        timeout=120, env={**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": package_root}))
        for seed in (1, 977)]
    expected = [sa.schemas.content_hash(ob._plain(real["bodies"][arm])) for arm in ("anchor", "candidate")]
    assert rows == [{"bodies": expected, "codes": ["claim_grade_receipt_unqualified"], "warranted": True}] * 2



# AKU-08c second child: the historical consumer records typed receipt eligibility in v4.
@contextmanager
def _historical_consumer(real, tmp_path, projection=None, receipts_root=None):
    """Real transaction WAL and actual receipts; candidate metadata is hermetic (as above)."""
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
    rows = cm.RequiredRowSet.from_dict(cm.RequiredRowSet("actual-final-v4", (diagnostic, production)).to_dict())
    controller = control.CampaignController(_resolved(), tmp_path / "consumer-service")
    controller.__enter__()
    source = mc.ArtifactStore(real["source"].root)
    receipts = mc.ArtifactStore(receipts_root or real["receipts"].root)
    store = mc.ArtifactStore(tmp_path / "consumer-evidence")
    try:
        manager = ct.CandidateTransactions(controller, git_backend=FakeGitBackend())
        projection = projection or sa.PinnedRootProjection(real["root"], source_closure=real["closure"])
        authority = sa.ValidationSemanticAdapter(projection).registered_authority(
            transaction_verifier_id="current-semantic/v2", source_store=source, receipt_store=receipts)
        validator = nc.NativeCaptureValidator(binding=nc.NativeCaptureBinding(
            controller.resolved.campaign_id, controller.config_digest,
            controller.config_generation, "unused-historical-consumer", 1), store=source)
        consumer = vc.ValidationConsumer(transactions=manager, native_validator=validator,
            evidence_store=store, semantic_authorities={authority.authority_id: authority})
        manager.initialize(request_id="initialize", state=_state(comparator), manifest=comparator)
        manager.integrate(request_id="integrate", previous=comparator, candidate=candidate, threshold_signal=True)
        assembly = consumer.assemble_due_batch(request_id="assemble", candidate=candidate,
            comparator=comparator, row_set=rows)

        def record(pair):
            return consumer.record_native_row(request_id="diagnostic", assembly=assembly,
                candidate=candidate, comparator=comparator, authority_id=authority.authority_id,
                evidence=vc.HistoricalReceiptRowEvidence(diagnostic.row_id, pair))
        yield SimpleNamespace(consumer=consumer, assembly=assembly, candidate=candidate,
            comparator=comparator, diagnostic=diagnostic, authority=authority, store=store, record=record)
    finally:
        store.close()
        receipts.close()
        source.close()
        controller.close()


def _rewrite(store, receipt, body):
    stored = store.write("candidate-validation-row", body)
    return replace(receipt, native_evidence_ref=stored.locator, native_evidence_digest=stored.sha256), stored


def test_v3_historical_rows_reopen_unchanged_and_v3_reader_refuses_v4(real, tmp_path):
    with _historical_consumer(real, tmp_path) as h:
        recorded = h.record(real["pair"])
        v4 = ob._plain(h.consumer.reopen_row(recorded.receipt))
        v3 = h.consumer._historical_body(h.assembly.batch, h.diagnostic, h.assembly.row_set,
            h.candidate, h.comparator, real["pair"], h.authority.authority_id, vc.HISTORICAL_V3)
        # v4 is additive: every v3 field except schema/prerequisites is identical.
        assert set(v4) - set(v3) == {"eligibility"} and "eligibility" not in v3
        common = set(v3) - {"schema", "prerequisites"}
        assert {key: v4[key] for key in common} == {key: v3[key] for key in common}
        assert v3["schema"] == vc.HISTORICAL_V3
        assert v3["prerequisites"] == sorted(set(v3["semantic_decision"]["reasons"]))
        legacy, stored = _rewrite(h.store, recorded.receipt, v3)
        raw = (h.store.root / stored.locator).read_bytes()
        for _ in range(2):
            assert ob._plain(h.consumer.reopen_row(legacy)) == v3
        assert (h.store.root / stored.locator).read_bytes() == raw
        # A v3 reader (exact v3 field set) refuses a v4 row, and v4 refuses losing its disposition.
        for body in (v4 | {"schema": vc.HISTORICAL_V3},
                     {key: value for key, value in v4.items() if key != "eligibility"}):
            forged, _ = _rewrite(h.store, recorded.receipt, body)
            with pytest.raises(vc.ValidationConsumerError, match="missing/unknown fields"):
                h.consumer.reopen_row(forged)
        changed = ob._plain(v4)
        changed["eligibility"]["receipt_codes"] = []
        forged, _ = _rewrite(h.store, recorded.receipt, changed)
        with pytest.raises(vc.ValidationConsumerError, match="complete receipt replay"):
            h.consumer.reopen_row(forged)


def test_consumer_records_unwarranted_receipts_with_typed_eligibility_codes(real, tmp_path):
    expected = ob._plain(real["closure"].expected_native_producer_source)
    expected["measurement_capture"]["callables"][0]["identity"]["implementation_sha256"] = "e" * 64
    closure = sa.ProjectionSourceClosure(real["closure"].root_commit,
                                         dict(real["closure"].root_source_sha256), expected)
    projection = sa.PinnedRootProjection(real["root"], source_closure=closure)
    receipts_root = tmp_path / "unwarranted-receipts"
    receipts = mc.ArtifactStore(receipts_root)
    try:
        refs = {}
        for arm, ref in (("anchor", real["pair"].anchor), ("candidate", real["pair"].candidate)):
            event = real["receipts"].read(ref.locator, ref.sha256)["source_event"]
            refs[arm] = projection.produce_receipt(source_store=real["source"],
                source_locator=event["locator"], source_sha256=event["sha256"], receipt_store=receipts)
    finally:
        receipts.close()
    pair = replace(real["pair"], anchor=refs["anchor"], candidate=refs["candidate"])
    with _historical_consumer(real, tmp_path, projection, receipts_root) as h:
        state = h.record(pair)
        bundle = h.consumer.reopen_row(state.receipt)
        assert bundle["schema"] == vc.HISTORICAL_V4
        assert list(bundle["eligibility"]["receipt_codes"]) == [
            "producer_provenance_unwarranted", "producer_implementation_differs"]
        assert state.status == "prerequisite_missing"
        assert "eligibility:producer_provenance_unwarranted" in state.reason
        assert state.receipt.use_disposition == "policy_undefined"


def test_registered_eligibility_gates_receipt_backed_findings(real, monkeypatch):
    from .test_scoped_evidence import claim_dict
    plan = ep.ExperimentPlan.from_dict(real["events"]["anchor"]["payload"]["carrier"]["plan"])
    verifier = sa.RegisteredClaimGradeVerifier(real["projection"], real["source"], real["receipts"])
    # FIXTURE comparison receipt over the actual pair: no producer journals the real one yet.
    receipt = feed.RegisteredComparisonReceipt(
        "fixture-comparison", hashlib.sha256(b"fixture-comparison").hexdigest(), real["pair"],
        feed.evidence.ClaimKey.from_dict(claim_dict()), "positive", 3.0, "strict_search",
        plan.intended_use, True)
    row = SimpleNamespace(payload={"measurement_id": real["pair"].candidate_measurement_id},
                          event_id="candidate-event")
    codes = feed.registered_claim_grade_eligibility(verifier)
    # The actual warranted pair is diagnostic: reprojected, typed refusal, no Finding.
    assert codes(receipt) == ("claim_grade_receipt_unqualified",)
    assert feed.registered_claim_grade_eligibility(object())(receipt) == ("claim_grade_receipt_unregistered",)
    projector = feed.ReceiptBackedFindingProjector({row.payload["measurement_id"]: receipt}.get,
                                                   codes, epoch="epoch-1")
    assert projector(None, ("Witnessed", "Attested"), row, 7) is None
    assert projector.refusals == {"candidate-event": ("claim_grade_receipt_unqualified",)}
    # Hermetic warranted measurement relabel: eligible for its own non-deployment use only.
    for use, expected in (("validate_production", ("comparison_use_policy_undefined",)), ("validate", ())):
        other, view = _complete(intended_use=use, instrument="serving", category="OPTIMUM")
        shaped = _shaped(real, other, view, current.ProducerProvenance(True, ()))
        monkeypatch.setattr(sa.RegisteredClaimGradeVerifier, "reopen_pair", lambda _self, _pair: shaped)
        assert codes(replace(receipt, intended_use=use)) == expected
    projector = feed.ReceiptBackedFindingProjector(
        {row.payload["measurement_id"]: replace(receipt, intended_use="validate")}.get, codes, epoch="epoch-1")
    finding = projector(None, ("Witnessed", "Attested"), row, 7)
    assert (finding.frontier, finding.authority_reference) == (7, "fixture:fixture-comparison")
    assert finding.intended_use_disposition["disposition"] == "exploration_only"
