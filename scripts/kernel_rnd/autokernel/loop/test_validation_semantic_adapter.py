"""Hermetic tests for the pinned semantic-owner integration."""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import py_compile
import runpy
import shutil
import sys
import time
from types import ModuleType

import pytest

from .. import schemas
from ..release.test_readiness import green_signal
from . import candidate_manifest as cm
from . import measurement_capture as mc
from . import validation_consumer as vc
from . import validation_claim_receipt as cr
from . import validation_loo_evidence as le
from . import validation_objective_decision as od
from . import validation_semantic_adapter as sa
from .test_validation_consumer import _consumer, _evidence


ROOT = Path("/mnt/raid0/llm/worktrees/mains/autokernel-unified-20260908")
TEAM2_ROOT = Path("/mnt/raid0/llm/worktrees/mains/autokernel-consumers-root-20260909")
TEAM2_PIN = sa.ProjectionSourcePin(
    "519acd08c017d4011c74ccc5e56491bce181351b",
    sa.CLAIM_TUPLE_SHA256,
    "3b5d7882d3007c661129096dc38af18a569c149c37dca36b5b2f6f428fbc2151",
    sa.ARM_ADAPTER_ID,
    "04cacacc8576048ff18a96e2c332ca2e2ea59bfdbfcc0acc451c1939f0ad3123",
    "00a880b3fa12dcd1cd2b06d4b9e30959ee14c38441cc5f28ae706d8e2ee68839")


def _alter_decision(store, reference, *, binding=None, native=None):
    body = json.loads(json.dumps(mc._plain(
        store.read(reference.locator, reference.sha256))))
    if binding:
        body["candidate_binding"].update(binding)
    if native:
        body["native_binding"].update(native)
    body.pop("decision_id")
    body["decision_id"] = "objective-" + schemas.content_hash(body)[:24]
    artifact = store.write("validation-objective-decision", body)
    return od.ObjectiveDecisionReference(
        body["decision_id"], body["objective_digest"], body["disposition"],
        artifact.locator, artifact.sha256)


def _assembly_and_receipt(tmp_path, *, fixture_authority=True):
    values = _consumer(tmp_path, authority=True, keep_count=1, cadence=0,
                       fixture_authority=fixture_authority)
    controller, consumer, candidate, comparator, row, row_set, plan, captures, calibration, \
        _capture_store, evidence_store = values
    assembly = consumer.assemble_due_batch(
        request_id="start", candidate=candidate, comparator=comparator, row_set=row_set)
    state = consumer.record_native_row(
        request_id="row", assembly=assembly, candidate=candidate,
        comparator=comparator, evidence=_evidence(row, captures, calibration),
        authority_id="fixture")
    assert state.receipt is not None
    return values, assembly, state.receipt, evidence_store


def test_pinned_root_uses_registered_projector_and_refuses_older_v1_shape(tmp_path):
    values, _assembly, receipt, _store = _assembly_and_receipt(tmp_path)
    controller, consumer, candidate, comparator, row, _row_set, _plan, captures, \
        _calibration, capture_store, _evidence_store = values
    try:
        projection = sa.PinnedRootProjection(ROOT)
        with pytest.raises(sa.SemanticAdapterError, match="refused the native source"):
            projection.grade_v1(
                captures["anchor"][1], receipt_locator=receipt.native_evidence_ref,
                receipt_sha256=receipt.native_evidence_digest,
                corpus_root=capture_store.root)
        assert projection.native_v2_available
        assert projection.source_pin == sa.ProjectionSourcePin(
            sa.FINAL_V2_ROOT_COMMIT, sa.FINAL_V2_CLAIM_TUPLE_SHA256,
            sa.FINAL_V2_ARM_PROJECTOR_SHA256, sa.FINAL_V2_ADAPTER_ID,
            sa.FINAL_V2_MEASUREMENT_CAPTURE_SHA256,
            sa.FINAL_V2_OBSERVATION_BINDING_SHA256)
        reopened = consumer.verify_row_receipt(receipt, row, candidate, comparator)
        assert reopened["row"]["row_id"] == row.row_id
        with pytest.raises(vc.ValidationConsumerError):
            consumer.verify_row_receipt(receipt, row, comparator, candidate)
    finally:
        controller.__exit__(None, None, None)


def test_pin_refuses_changed_projector_bytes(tmp_path):
    root = tmp_path / "root"
    (root / "scripts/vidya/adapters").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/vidya/claim_tuple.py", root / "scripts/vidya/claim_tuple.py")
    target = root / "scripts/vidya/adapters/autokernel_unified_arm.py"
    shutil.copy2(ROOT / "scripts/vidya/adapters/autokernel_unified_arm.py", target)
    target.write_bytes(target.read_bytes() + b"\n# changed\n")
    with pytest.raises(sa.SemanticAdapterError, match="not the pinned version"):
        sa.PinnedRootProjection(root)


def test_loader_executes_verified_source_not_stale_timestamp_pyc(tmp_path):
    source = tmp_path / "pinned_source.py"
    source.write_text("VALUE = 'old'\n")
    before = source.stat()
    py_compile.compile(str(source), doraise=True)
    source.write_text("VALUE = 'new'\n")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert sa._digest(source) == hashlib.sha256(b"VALUE = 'new'\n").hexdigest()
    loaded = sa._load("_aku_stale_pyc_regression", source)
    assert loaded.VALUE == "new"


def test_loader_failure_restores_existing_module(tmp_path):
    source = tmp_path / "broken.py"
    source.write_text("this is not valid python !!!\n")
    name = "_aku_failed_load_cleanup"
    prior = ModuleType(name)
    sys.modules[name] = prior
    try:
        with pytest.raises(SyntaxError):
            sa._load(name, source)
        assert sys.modules[name] is prior
    finally:
        sys.modules.pop(name, None)


def test_loader_base_exception_restores_existing_module(tmp_path):
    source = tmp_path / "exits.py"
    source.write_text("raise SystemExit(17)\n")
    name = "_aku_base_exception_cleanup"
    prior = ModuleType(name)
    sys.modules[name] = prior
    try:
        with pytest.raises(SystemExit, match="17"):
            sa._load(name, source)
        assert sys.modules[name] is prior
    finally:
        sys.modules.pop(name, None)


def test_source_replacement_during_capture_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "moving.py"
    replacement = tmp_path / "replacement.py"
    source.write_text("VALUE = 'first'\n")
    replacement.write_text("VALUE = 'other'\n")
    original_read = sa.os.read
    swapped = False

    def replacing_read(descriptor, size):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.replace(replacement, source)
        return original_read(descriptor, size)

    monkeypatch.setattr(sa.os, "read", replacing_read)
    with pytest.raises(sa.SemanticAdapterError, match="changed while being read"):
        sa._source_bytes(source)


def test_source_fifo_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "not-source.py"
    os.mkfifo(fifo)
    started = time.monotonic()
    with pytest.raises(sa.SemanticAdapterError, match="bounded regular file"):
        sa._source_bytes(fifo)
    assert time.monotonic() - started < 0.5


def test_concurrent_projector_construction_cannot_cross_wire_registration():
    def construct(_index):
        projection = sa.PinnedRootProjection(ROOT)
        return projection.projector.__globals__["ClaimTuple"], \
            projection.claim_tuple.ClaimTuple

    with ThreadPoolExecutor(max_workers=8) as pool:
        pairs = tuple(pool.map(construct, range(24)))
    assert all(projected is registered for projected, registered in pairs)


def test_real_adapter_keeps_candidate_row_unavailable(tmp_path):
    values = _consumer(tmp_path, authority=False, keep_count=1, cadence=0)
    controller, consumer, candidate, comparator, row, row_set, _plan, captures, calibration, \
        _capture_store, _evidence_store = values
    try:
        adapter = sa.ValidationSemanticAdapter(sa.PinnedRootProjection(ROOT))
        consumer.semantic_authorities["real"] = adapter.registered_authority("owner-v1")
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=row_set)
        state = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="real")
        assert state.status == "prerequisite_missing"
        assert "native-v2" in (state.reason or "")
        assert "instrument_identity_unknown" in (state.reason or "")
    finally:
        controller.__exit__(None, None, None)


def test_moving_team2_v2_receipt_reprojects_but_is_compatibility_only(tmp_path):
    fixture_module = runpy.run_path(
        str(TEAM2_ROOT / "tests/vidya/test_autokernel_unified_arm.py"))
    source_store = mc.ArtifactStore(tmp_path / "v2-source")
    receipt_store = mc.ArtifactStore(tmp_path / "v2-receipts")
    carrier = fixture_module["v2_carrier_fixture"](source_store.root)
    carrier_artifact = source_store.write("team2-v2-carrier", carrier)
    event = {"journal_schema": "epyc.autokernel.journal_entry.v1",
             "event_id": "event-v2", "seq": 1,
             "kind": "PLANNED_SERVING_ARM_CAPTURED",
             "campaign_id": "campaign-1", "record_id": carrier["measurement_id"],
             "written_at": "2026-09-09T00:00:02Z",
             "payload": {"schema": "epyc.autokernel.unified_arm_capture.v2",
                 "measurement_id": carrier["measurement_id"], "carrier": carrier,
                 "artifact": carrier_artifact.to_dict()}}
    event_artifact = source_store.write("team2-v2-journal-event", event)
    projection = sa.PinnedRootProjection(TEAM2_ROOT, source_pin=TEAM2_PIN)
    reference = projection.produce_receipt(
        source_store=source_store, source_locator=event_artifact.locator,
        source_sha256=event_artifact.sha256, receipt_store=receipt_store)
    body = projection.reopen_receipt(
        reference=reference, source_store=source_store, receipt_store=receipt_store)
    assert body["authority_scope"] == "compatibility_only"

    assert body["source_identity"]["measurement_capture_source_sha256"] \
        == TEAM2_PIN.measurement_capture_source_sha256
    assert body["source_identity"]["observation_binding_source_sha256"] \
        == TEAM2_PIN.observation_binding_source_sha256
    assert (body["projection"]["source_grade"],
            body["projection"]["trace_grade"]) == ("Witnessed", "Attested")
    assert body["native_binding"]["instrument_identity_sha256"] \
        == carrier["loaded_instrument"]["identity_sha256"]
    with pytest.raises(sa.SemanticAdapterError, match="pair identity"):
        sa.ValidationSemanticAdapter(projection).reopen_receipt_pair(
            anchor=reference, candidate=reference, source_store=source_store,
            receipt_store=receipt_store)
    bad = cr.ClaimGradeReceiptReference(
        reference.receipt_id, reference.locator,
        schemas.content_hash({"replacement": "receipt"}))
    with pytest.raises(sa.SemanticAdapterError, match="cannot be reopened"):
        projection.reopen_receipt(
            reference=bad, source_store=source_store, receipt_store=receipt_store)
    changed = json.loads(json.dumps(mc._plain(body)))
    changed["source_identity"]["measurement_capture_producer_id"] = \
        "epyc.autokernel.measurement_capture/other"
    changed.pop("receipt_id")
    changed["receipt_id"] = "claim-grade-" + schemas.content_hash(changed)[:24]
    changed_artifact = receipt_store.write("canonical-claim-grade-receipt", changed)
    changed_ref = cr.ClaimGradeReceiptReference(
        changed["receipt_id"], changed_artifact.locator, changed_artifact.sha256)
    with pytest.raises(sa.SemanticAdapterError, match="source identity changed"):
        projection.reopen_receipt(
            reference=changed_ref, source_store=source_store,
            receipt_store=receipt_store)
    malformed_artifact = source_store.write(
        "team2-v2-journal-event", {"payload": None})
    with pytest.raises(sa.SemanticAdapterError, match="event is malformed"):
        projection.produce_receipt(
            source_store=source_store, source_locator=malformed_artifact.locator,
            source_sha256=malformed_artifact.sha256, receipt_store=receipt_store)


def test_final_verifier_pins_do_not_retroactively_attest_producer_source(
        tmp_path, monkeypatch):
    fixture_module = runpy.run_path(
        str(ROOT / "tests/vidya/test_autokernel_unified_arm.py"))
    source_store = mc.ArtifactStore(tmp_path / "source")
    receipt_store = mc.ArtifactStore(tmp_path / "receipts")
    carrier = fixture_module["v2_carrier_fixture"](source_store.root)
    carrier_artifact = source_store.write("native-v2-carrier", carrier)
    event = {"journal_schema": "epyc.autokernel.journal_entry.v1",
             "event_id": "event-final-v2", "seq": 1,
             "kind": "PLANNED_SERVING_ARM_CAPTURED",
             "campaign_id": "campaign-1", "record_id": carrier["measurement_id"],
             "written_at": "2026-09-09T00:00:02Z",
             "payload": {"schema": "epyc.autokernel.unified_arm_capture.v2",
                 "measurement_id": carrier["measurement_id"], "carrier": carrier,
                 "artifact": carrier_artifact.to_dict()}}
    event_artifact = source_store.write("native-v2-journal-event", event)
    projection = sa.PinnedRootProjection(ROOT)
    assert projection.native_v2_available
    reference = projection.produce_receipt(
        source_store=source_store, source_locator=event_artifact.locator,
        source_sha256=event_artifact.sha256, receipt_store=receipt_store)
    body = projection.reopen_receipt(
        reference=reference, source_store=source_store, receipt_store=receipt_store)
    assert body["authority_scope"] == "compatibility_only"
    candidate_body = json.loads(json.dumps(body))
    candidate_body["native_binding"]["arm"] = "candidate"
    semantic = sa.ValidationSemanticAdapter(projection)
    monkeypatch.setattr(
        semantic, "reopen_receipt_pair",
        lambda **_kwargs: (body, candidate_body))
    decision = semantic.evaluate_receipt_pair(
        sa.ep.ExperimentPlan.from_dict(carrier["plan"]), anchor=reference,
        candidate=reference, source_store=source_store,
        receipt_store=receipt_store)
    assert not decision.permitted
    assert "native producer source identity is compatibility-only" in decision.reasons
    wrong_schema = json.loads(json.dumps(event))
    wrong_schema["payload"]["carrier"]["lifecycle_observations"][0]["schema"] = \
        "epyc.autokernel.lifecycle_observation_reference.other"
    wrong_schema_artifact = source_store.write("team2-v2-journal-event", wrong_schema)
    with pytest.raises(sa.SemanticAdapterError, match="producer/schema identity"):
        projection.produce_receipt(
            source_store=source_store, source_locator=wrong_schema_artifact.locator,
            source_sha256=wrong_schema_artifact.sha256, receipt_store=receipt_store)


def test_actual_readiness_result_and_exact_loo_index_round_trip(tmp_path):
    values, assembly, receipt, store = _assembly_and_receipt(tmp_path)
    controller, _consumer_obj, candidate, comparator, old_row, _old_set, plan, captures, \
        calibration, _capture_store, _evidence_store = values
    try:
        signal = green_signal()
        row_data = old_row.to_dict()
        row_data.update(
            objective_digest=schemas.content_hash(signal.objective.to_dict()),
            protocol_id=next(iter(signal.objective.protocol_by_phase.values())),
            instrument_digest=schemas.content_hash({"fixture": "prospective-v2-instrument"}))
        row = cm.ValidationRow.from_dict(row_data)
        row_set = cm.RequiredRowSet.from_dict(
            cm.RequiredRowSet("semantic-v1", (row,)).to_dict())
        frozen = replace(
            assembly.batch, row_set_digest=row_set.row_set_digest,
            required_row_ids=(row.row_id,),
            rows=(cm.ValidationRowState(row.row_id, "pending", None, None),)).validated()
        anchor_id, anchor_payload = captures["anchor"]
        candidate_id, candidate_payload = captures["candidate"]
        native = od.NativePairBinding(
            plan.digest, anchor_id,
            anchor_payload["carrier"]["carrier_digest"], candidate_id,
            candidate_payload["carrier"]["carrier_digest"], calibration.digest,
            row.instrument_digest, anchor_payload["artifact"]["locator"],
            anchor_payload["artifact"]["sha256"])
        decision_ref = od.seal_readiness_decision(
            store=store, signal=signal, batch=frozen, row=row, row_set=row_set,
            candidate=candidate, comparator=comparator, native=native)
        decision = od.reopen_readiness_decision(store=store, reference=decision_ref)
        assert decision["source_standing"] == "objective_met"
        assert decision_ref.disposition == "policy_undefined"

        loo_plan = cm.plan_loo(candidate, candidate.keeps[0].keep_id)
        sealed_bundle = dict(store.read(
            receipt.native_evidence_ref, receipt.native_evidence_digest))
        sealed_bundle.update(batch=frozen.to_dict(), row=row.to_dict(),
                             row_set=row_set.to_dict())
        semantic_row = store.write("candidate-validation-row", sealed_bundle)
        semantic_receipt = cm.RowReceipt.from_dict({
            "schema": cm.RECEIPT_SCHEMA, "batch_id": frozen.batch_id,
            "candidate_manifest_digest": candidate.manifest_digest,
            "comparator_manifest_digest": comparator.manifest_digest,
            "row_set_digest": row_set.row_set_digest, "row_id": row.row_id,
            "native_evidence_ref": semantic_row.locator,
            "native_evidence_digest": semantic_row.sha256,
            "intended_use": "validate", "use_disposition": "permitted"})
        pointer = le.RowEvidencePointer(semantic_receipt)
        body_digest = le.evidence_digest(
            plan=loo_plan, candidate=candidate, row_set=row_set,
            row_evidence=(pointer,), objective_decisions=(decision_ref,))
        result = cm.LOOResult.from_dict({
            "schema": cm.LOO_RESULT_SCHEMA, "plan_digest": loo_plan.plan_digest,
            "candidate_manifest_digest": candidate.manifest_digest,
            "keep_id": loo_plan.keep_id,
            "derived_manifest_digest": loo_plan.derived_manifest_digest,
            "row_set_digest": row_set.row_set_digest,
            "receipt_digests": [semantic_row.sha256],
            "disposition": "supports_keep", "evidence_digest": body_digest,
            "deletion_authorized": False})
        loo_ref = le.seal_loo_evidence(
            store=store, result=result, plan=loo_plan, candidate=candidate,
            row_set=row_set, row_evidence=(pointer,), objective_decisions=(decision_ref,))
        exact = le.LOOEvidenceIndex((loo_ref,)).exact(body_digest)
        assert le.reopen_loo_evidence(store=store, reference=exact, result=result)["plan"] \
            == loo_plan.to_dict()
        forged_pointer = le.RowEvidencePointer(replace(
            semantic_receipt,
            row_set_digest=schemas.content_hash({"different": "row-set"})))
        with pytest.raises(le.LOOEvidenceError, match="semantic binding"):
            le.seal_loo_evidence(
                store=store, result=result, plan=loo_plan, candidate=candidate,
                row_set=row_set, row_evidence=(forged_pointer,),
                objective_decisions=(decision_ref,))
        foreign_batch = _alter_decision(
            store, decision_ref, binding={"batch_id": "another-validation-batch"})
        foreign_comparator = _alter_decision(
            store, decision_ref,
            binding={"comparator_manifest_digest":
                     schemas.content_hash({"different": "comparator"})})
        foreign_native = _alter_decision(
            store, decision_ref,
            native={"anchor_measurement_id":
                    schemas.content_hash({"different": "measurement"})})
        for wrong, reason in (
                (foreign_batch, "batch/comparator"),
                (foreign_comparator, "batch/comparator"),
                (foreign_native, "native binding")):
            with pytest.raises(le.LOOEvidenceError, match=reason):
                le.seal_loo_evidence(
                    store=store, result=result, plan=loo_plan, candidate=candidate,
                    row_set=row_set, row_evidence=(pointer,),
                    objective_decisions=(wrong,))
        with pytest.raises(le.LOOEvidenceError):
            le.LOOEvidenceIndex(()).exact(body_digest)
        with pytest.raises(le.LOOEvidenceError, match="cannot be reopened"):
            le.reopen_loo_evidence(
                store=store,
                reference=replace(loo_ref, locator="missing-loo-evidence.json"),
                result=result)
        with pytest.raises(od.ObjectiveDecisionError, match="cannot be reopened"):
            od.reopen_readiness_decision(
                store=store,
                reference=replace(decision_ref, locator="missing-objective.json"))
    finally:
        controller.__exit__(None, None, None)
