"""Hermetic integration tests for the candidate validation consumer."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from .. import schemas
from . import candidate_manifest as cm
from . import candidate_transactions as ct
from . import campaign_control as control
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import planned_serving as ps
from . import validation_consumer as vc
from .test_campaign_control import _resolved
from .test_candidate_manifest import _manifest, _state
from .test_candidate_transactions import FakeGitBackend
from .test_experiment_plan import receipt_dict
from .test_native_capture_control import _binding, _fence, _Provider
from .test_measurement_capture import _context
from .test_planned_serving import _measure, _plan, _prompts, _recipes


def _calibration() -> ep.CalibrationReceipt:
    value = receipt_dict(unit="process")
    value["metric"] = "aggregate_tok_s"
    return ep.CalibrationReceipt.from_dict(value)


def _produce(controller, tmp_path, calibration, *, lineage="lineage-1",
             capture_store=None, partial=False, plan_updates=None):
    at, ct_recipe, anchor, candidate = _recipes()
    helper = replace(anchor.dsos[0], path="/build/bin/libhelper.so", sha256="e" * 64)
    def with_helper(recipe):
        provisional = replace(
            recipe, dsos=recipe.dsos + (helper,), snapshot_digest="0" * 64,
            execution_digest="0" * 64)
        return replace(
            provisional, snapshot_digest=schemas.content_hash(provisional._snapshot_dict()),
            execution_digest=schemas.content_hash(provisional._normalized_execution_dict()))

    anchor = with_helper(anchor)
    candidate = with_helper(candidate)
    plan_row = _plan(at, ct_recipe, anchor, candidate).to_dict()
    plan_row.update({
        "campaign_id": controller.resolved.campaign_id,
        "target_revision": "production-target-v1",
        "category": "OPTIMUM", "phase": "release",
        "record_class": "registered_claim", "intended_use": "validate_production",
        "calibration_ref": calibration.digest, "unit": "process",
    })
    plan_row.update(plan_updates or {})
    plan = ep.ExperimentPlan.from_dict(plan_row)
    context_row = _context(at, ct_recipe, anchor, candidate).to_dict()
    context_row.update({
        "campaign_id": controller.resolved.campaign_id,
        "config_digest": controller.config_digest,
        "config_generation": controller.config_generation,
        "supervisor_incarnation": controller.supervisor_incarnation,
        "lineage_id": lineage,
    })
    context = mc.CaptureContext.from_dict(context_row)
    capture_store = capture_store or mc.ArtifactStore(tmp_path / "native")
    payloads = {}
    sink = mc.NativeMeasurementSink(
        context=context, store=capture_store,
        capture_transaction=lambda measurement_id, payload:
        payloads.setdefault(measurement_id, payload) or {"record_id": measurement_id})
    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct_recipe,
        anchor_recipe=anchor, candidate_recipe=candidate,
        prompts=_prompts(at), stage_provider=_Provider(context), artifact_sink=sink,
        lineage_id=context.lineage_id, clock=lambda: 1.0,
        wall_clock=lambda: next(ticks), measure=_measure([], partial=partial))
    validator = nc.NativeCaptureValidator(
        binding=_binding(context), store=capture_store,
        fence_provider=lambda _measurement_id, actual: _fence(actual))
    by_arm = {payload["carrier"]["arm"]: (measurement_id, payload)
              for measurement_id, payload in payloads.items()}
    return plan, by_arm, validator, capture_store, {
        "anchor": anchor.dsos, "candidate": candidate.dsos}


def _matching_manifest(plan, *, candidate, keep_count=1, target_backend="cpu",
                       arm_dsos=None, dso_mutation=None):
    raw = _manifest(manifest_id="candidate" if candidate else "comparator")
    identity = dict(plan.candidate_identity if candidate else plan.anchor_identity)
    raw["builds"][0]["executable"]["sha256"] = identity["executable_digest"]
    raw["builds"][0]["dsos"] = [item.to_dict() for item in arm_dsos]
    if dso_mutation == "missing":
        raw["builds"][0]["dsos"] = raw["builds"][0]["dsos"][:-1]
    elif dso_mutation == "changed":
        raw["builds"][0]["dsos"][0]["sha256"] = "f" * 64
    elif dso_mutation == "extra":
        extra = dict(raw["builds"][0]["dsos"][0])
        extra.update(path="/fixture/libextra.so", sha256="e" * 64)
        raw["builds"][0]["dsos"].append(extra)
    build = cm.BuildIdentity.from_dict(raw["builds"][0])
    target = raw["targets"][0]
    target.update({
        "backend": target_backend, "build_execution_digest": build.execution_digest,
        "resolved_recipe_execution_digest": identity["resolved_execution_digest"],
        "resolved_recipe_snapshot_digest": identity["resolved_snapshot_digest"],
        "model_digest": identity["model_digest"], "drafter_digest": identity["drafter_digest"],
        "workload_digest": identity["workload_digest"],
    })
    raw["keeps"] = []
    if candidate:
        prior = "7" * 64
        for index in range(keep_count):
            current = (identity["resolved_execution_digest"] if index == keep_count - 1
                       else f"{index + 1:x}" * 64)
            raw["keeps"].append({
                "schema": cm.KEEP_SCHEMA, "request_id": f"keep-request-{index}",
                "keep_id": f"keep-{index}", "parent_manifest_digest": "0" * 64,
                "kind": "runtime", "changes": [{"schema": cm.CHANGE_SCHEMA,
                    "field_id": "runtime:recipe", "previous_digest": prior,
                    "current_digest": current}], "affected_scopes": [identity["backend"]],
                "dependencies": []})
            prior = current
    return cm.CandidateManifest.from_dict(raw)


def _candidate_chain(plan, count, *, target_backend="cpu", dsos, dso_mutation=None):
    base = _matching_manifest(
        plan, candidate=False, target_backend=target_backend,
        arm_dsos=dsos["anchor"], dso_mutation=dso_mutation)
    chain, previous = [], base
    final_recipe = dict(plan.candidate_identity)["resolved_execution_digest"]
    for index in range(count):
        current = final_recipe if index == count - 1 else f"{index + 1:x}" * 64
        current_snapshot = (dict(plan.candidate_identity)["resolved_snapshot_digest"]
                            if index == count - 1 else f"{index + 5:x}" * 64)
        raw = previous.to_dict()
        raw["manifest_id"] = f"candidate-{index}"
        raw["parent_manifest_digest"] = previous.manifest_digest
        raw["targets"][0]["resolved_recipe_execution_digest"] = current
        raw["targets"][0]["resolved_recipe_snapshot_digest"] = current_snapshot
        raw["keeps"].append({
            "schema": cm.KEEP_SCHEMA, "request_id": f"keep-request-{index}",
            "keep_id": f"keep-{index}", "parent_manifest_digest": previous.manifest_digest,
            "kind": "runtime", "changes": [
                {"schema": cm.CHANGE_SCHEMA, "field_id": "runtime:recipe",
                 "previous_digest": previous.targets[0].resolved_recipe_execution_digest,
                 "current_digest": current},
                {"schema": cm.CHANGE_SCHEMA, "field_id": "runtime:snapshot",
                 "previous_digest": previous.targets[0].resolved_recipe_snapshot_digest,
                 "current_digest": current_snapshot}], "affected_scopes": ["cpu"],
            "dependencies": []})
        previous = cm.CandidateManifest.from_dict(raw)
        chain.append(previous)
    return base, chain


def _row(plan, candidate, comparator, *, backend="cpu"):
    ca, an = candidate.targets[0], comparator.targets[0]
    row = cm.ValidationRow.from_dict({
        "schema": cm.ROW_SCHEMA, "row_id": f"production-{backend}",
        "row_kind": "production", "required": True,
        "target_revision_digest": ca.target_revision_digest,
        "control_target_revision_digest": an.target_revision_digest, "backend": backend,
        "candidate_build_digest": ca.build_execution_digest,
        "control_build_digest": an.build_execution_digest, "model_digest": ca.model_digest,
        "drafter_digest": ca.drafter_digest, "category": "OPTIMUM",
        "control_recipe_digest": an.resolved_recipe_execution_digest,
        "candidate_recipe_digest": ca.resolved_recipe_execution_digest,
        "instrument_digest": "c" * 64, "protocol_id": plan.protocol_ref,
        "objective_digest": "d" * 64, "workload_digest": ca.workload_digest,
        "exact_candidate_required": True})
    direct = cm.RequiredRowSet("production-v1", (row,))
    return row, cm.RequiredRowSet.from_dict(direct.to_dict())


def _consumer(tmp_path, *, authority=False, keep_count=1, cadence=0,
              target_backend="cpu", fixture_authority=True, dso_mutation=None):
    service = tmp_path / "service"
    controller = control.CampaignController(_resolved(), service)
    controller.__enter__()
    calibration = _calibration()
    plan, captures, validator, capture_store, dsos = _produce(
        controller, tmp_path, calibration)
    comparator, chain = _candidate_chain(
        plan, keep_count, target_backend=target_backend, dsos=dsos,
        dso_mutation=dso_mutation)
    candidate = chain[-1]
    row, row_set = _row(plan, candidate, comparator)
    verifier_id = "fixture-owner/v1"
    verifiers = {verifier_id: (lambda *_args: True, lambda *_args: True)} if authority else {}
    manager = ct.CandidateTransactions(controller, git_backend=FakeGitBackend(),
                                       verifiers=verifiers)
    manager.initialize(request_id="init", state=_state(comparator), manifest=comparator)
    for index, item in enumerate(chain):
        manager.integrate(
            request_id=f"integrate-{index}", previous=comparator if index == 0 else chain[index - 1],
            candidate=item, threshold_signal=(cadence < 4 and index == len(chain) - 1))
    authorities = {}
    if authority:
        authorities["fixture"] = vc.RegisteredSemanticAuthority(
            "fixture-owner/v1", verifier_id,
            lambda _plan, _evidence: vc.SemanticDecision(
                "Witnessed", "Attested", "validate_production", True),
            fixture_only=fixture_authority)
    evidence_store = mc.ArtifactStore(tmp_path / "validation-evidence")
    consumer = vc.ValidationConsumer(
        transactions=manager, native_validator=validator, evidence_store=evidence_store,
        semantic_authorities=authorities,
        registered_estimators={"median.v1": lambda receipt: receipt.value},
        calibration_rule_id="fixture-calibration/v1",
        calibration_rule=lambda _receipt, _plan: True)
    return (controller, consumer, candidate, comparator, row, row_set, plan,
            captures, calibration, capture_store, evidence_store)


def _evidence(row, captures, calibration):
    anchor_id, anchor = captures["anchor"]
    candidate_id, candidate = captures["candidate"]
    return vc.NativeRowEvidence(row.row_id, anchor_id, anchor, candidate_id, candidate,
                                calibration)


def _reseal(store, receipt, bundle):
    stored = store.write("candidate-validation-row", bundle)
    return replace(
        receipt, native_evidence_ref=stored.locator,
        native_evidence_digest=stored.sha256)


def _native_closures(consumer, captures):
    result = {}
    for arm in ("anchor", "candidate"):
        measurement_id, payload = captures[arm]
        result[arm] = consumer._native_closure(
            consumer.native_validator.validate(measurement_id, payload), payload)
    return result


def _append_runtime(previous):
    raw = previous.to_dict()
    raw["manifest_id"] = "newer-tip"
    raw["parent_manifest_digest"] = previous.manifest_digest
    old_recipe = raw["targets"][0]["resolved_recipe_execution_digest"]
    old_snapshot = raw["targets"][0]["resolved_recipe_snapshot_digest"]
    raw["targets"][0]["resolved_recipe_execution_digest"] = "e" * 64
    raw["targets"][0]["resolved_recipe_snapshot_digest"] = "f" * 64
    raw["keeps"].append({
        "schema": cm.KEEP_SCHEMA, "request_id": "newer-request", "keep_id": "newer",
        "parent_manifest_digest": previous.manifest_digest, "kind": "runtime",
        "changes": [
            {"schema": cm.CHANGE_SCHEMA, "field_id": "runtime:recipe",
             "previous_digest": old_recipe, "current_digest": "e" * 64},
            {"schema": cm.CHANGE_SCHEMA, "field_id": "runtime:snapshot",
             "previous_digest": old_snapshot, "current_digest": "f" * 64}],
        "affected_scopes": ["cpu"], "dependencies": []})
    return cm.CandidateManifest.from_dict(raw)


def test_default_authority_unavailable_records_exact_debt_and_real_transaction(tmp_path):
    values = _consumer(tmp_path, authority=False, keep_count=4, cadence=4)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        retried = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        assert retried.batch == assembly.batch
        assert assembly.trigger == "cadence"
        assert any("identifiable_loo" in item for item in assembly.debt[0].prerequisites)
        state = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration))
        assert state.status == "prerequisite_missing"
        assert "registered_semantic_authority" in state.reason
        consumer.complete(request_id="complete", assembly=assembly)
        snapshot = consumer.transactions.inspect()["state"]
        assert snapshot["keeps_since_gate"] == 4  # refused-before-start is not attempted
        assert assembly.batch.batch_id in snapshot["validation_debt"]
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


def test_fixture_owner_exercises_record_complete_and_advance_with_native_artifact(tmp_path):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        recorded = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        assert recorded.status == "passed" and recorded.receipt is not None
        assert consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture") == recorded
        captures.clear()  # the durable closure, not caller memory, must be sufficient
        reopened = consumer.reopen_row(recorded.receipt)
        assert reopened["native"]["anchor"]["measurement_id"]
        assert reopened["native"]["candidate"]["raw_artifacts"]
        assert reopened["calibration"]["plan_reference"] == calibration.digest
        consumer.complete(request_id="complete", assembly=assembly)
        plan = assembly.loo_plans["keep-0"]
        result = cm.LOOResult.from_dict({
            "schema": cm.LOO_RESULT_SCHEMA, "plan_digest": plan.plan_digest,
            "candidate_manifest_digest": candidate.manifest_digest, "keep_id": "keep-0",
            "derived_manifest_digest": plan.derived_manifest_digest,
            "row_set_digest": rows.row_set_digest,
            "receipt_digests": [recorded.receipt.native_evidence_digest],
            "disposition": "neutral", "evidence_digest": "2" * 64,
            "deletion_authorized": False})
        advanced = consumer.advance(
            request_id="advance", assembly=assembly, candidate=candidate,
            comparator=comparator, authority_id="fixture", loo_results={"keep-0": result})
        assert advanced["validated_candidate"] == candidate.manifest_digest
        assert advanced["validation_debt"] == []
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


@pytest.mark.parametrize("field,value", [
    ("batch_id", "different-batch"),
    ("row_id", "different-row"),
    ("candidate_manifest_digest", "9" * 64),
    ("comparator_manifest_digest", "8" * 64),
    ("row_set_digest", "7" * 64),
])
def test_reopen_binds_every_receipt_identity_field(tmp_path, field, value):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        recorded = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        assert consumer.reopen_row(recorded.receipt)
        with pytest.raises(vc.ValidationConsumerError):
            consumer.reopen_row(replace(recorded.receipt, **{field: value}))
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


@pytest.mark.parametrize("damage", [
    "batch", "row_semantics", "plan_digest", "native_arm", "native_lineage",
    "calibration", "semantic_decision",
])
def test_reopen_closed_validator_refuses_forged_bundle_fields(tmp_path, damage):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        recorded = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        bundle = json.loads(json.dumps(mc._plain(consumer.reopen_row(recorded.receipt))))
        if damage == "batch":
            bundle["batch"]["batch_id"] = "forged-batch"
        elif damage == "row_semantics":
            bundle["row"]["protocol_id"] = "forged-protocol/v1"
        elif damage == "plan_digest":
            bundle["plan_digest"] = "9" * 64
        elif damage == "native_arm":
            bundle["native"]["anchor"]["payload"]["carrier"]["arm"] = "candidate"
        elif damage == "native_lineage":
            bundle["native"]["candidate"]["payload"]["carrier"]["lineage_id"] = "other"
        elif damage == "calibration":
            bundle["calibration"]["plan_reference"] = "8" * 64
        else:
            bundle["semantic_decision"]["permitted"] = False
        forged = _reseal(stores[1], recorded.receipt, bundle)
        with pytest.raises(vc.ValidationConsumerError):
            consumer.reopen_row(forged)
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


def test_reopen_refuses_valid_diagnostic_native_pair(tmp_path):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        recorded = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        bundle = json.loads(json.dumps(mc._plain(consumer.reopen_row(recorded.receipt))))
        diagnostic_plan, diagnostic, _, _, _ = _produce(
            controller, tmp_path, calibration, capture_store=stores[0],
            plan_updates={"unit": "session"})
        bundle["native"] = _native_closures(consumer, diagnostic)
        bundle["plan_digest"] = diagnostic_plan.digest
        forged = _reseal(stores[1], recorded.receipt, bundle)
        with pytest.raises(vc.ValidationConsumerError, match="diagnostic"):
            consumer.reopen_row(forged)
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


def test_reopen_refuses_structurally_ineligible_measurement_pair(tmp_path):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        recorded = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        bundle = json.loads(json.dumps(mc._plain(consumer.reopen_row(recorded.receipt))))
        refused_plan, refused, _, _, _ = _produce(
            controller, tmp_path, calibration, capture_store=stores[0],
            plan_updates={"record_class": "observation", "phase": "observation"})
        bundle["native"] = _native_closures(consumer, refused)
        bundle["plan_digest"] = refused_plan.digest
        forged = _reseal(stores[1], recorded.receipt, bundle)
        with pytest.raises(vc.ValidationConsumerError, match="structurally ineligible"):
            consumer.reopen_row(forged)
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


@pytest.mark.parametrize("mutation", ["missing", "changed", "extra"])
def test_full_dso_set_mismatch_is_structural_debt_even_with_callback(tmp_path, mutation):
    values = _consumer(tmp_path, authority=True, dso_mutation=mutation)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        state = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        assert state.status == "prerequisite_missing"
        assert "measured_artifact_identity_mismatch" in state.reason
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


def test_nonfixture_callback_cannot_fill_unknown_instrument_identity(tmp_path):
    values = _consumer(tmp_path, authority=True, fixture_authority=False)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        state = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        assert state.status == "prerequisite_missing"
        assert "instrument_identity_unknown" in state.reason
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


@pytest.mark.parametrize("damage", [
    "missing_bundle", "replaced_bundle", "missing_carrier", "replaced_raw"])
def test_reopen_refuses_missing_or_replaced_native_refs(tmp_path, damage):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        recorded = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        bundle = consumer.reopen_row(recorded.receipt)
        anchor = bundle["native"]["anchor"]
        if damage == "missing_bundle":
            damaged = stores[1].root / recorded.receipt.native_evidence_ref
            damaged.unlink()
        elif damage == "replaced_bundle":
            damaged = stores[1].root / recorded.receipt.native_evidence_ref
            damaged.write_bytes(b"{}")
        elif damage == "missing_carrier":
            damaged = stores[0].root / anchor["carrier_artifact"]["locator"]
            damaged.unlink()
        else:
            damaged = stores[0].root / anchor["raw_artifacts"][0]["stored"]["locator"]
            damaged.write_bytes(b"{}")
        with pytest.raises(vc.ValidationConsumerError,
                           match="cannot be (reopened|revalidated)"):
            consumer.reopen_row(recorded.receipt)
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


@pytest.mark.parametrize("damage", ["closed", "replaced"])
def test_reopen_refuses_closed_or_replaced_evidence_store(tmp_path, damage):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        recorded = consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        if damage == "closed":
            stores[1].close()
        else:
            stores[1].root.rename(tmp_path / "original-validation-evidence")
            stores[1].root.mkdir(mode=0o700)
        with pytest.raises(vc.ValidationConsumerError, match="cannot be reopened"):
            consumer.reopen_row(recorded.receipt)
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


def test_new_integration_tip_after_completed_batch_refuses_advancement(tmp_path):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        consumer.record_native_row(
            request_id="row", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(row, captures, calibration),
            authority_id="fixture")
        consumer.complete(request_id="complete", assembly=assembly)
        newer = _append_runtime(candidate)
        consumer.transactions.integrate(
            request_id="newer", previous=candidate, candidate=newer)
        with pytest.raises(vc.ValidationConsumerError, match="changed before advancement"):
            consumer.advance(
                request_id="advance", assembly=assembly, candidate=candidate,
                comparator=comparator, authority_id="fixture", loo_results={})
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


def test_caller_cannot_swap_manifest_or_forge_active_assembly(tmp_path):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        alternate = _append_runtime(candidate)
        with pytest.raises(vc.ValidationConsumerError, match="candidate/comparator/row-set"):
            consumer.record_native_row(
                request_id="alternate", assembly=assembly, candidate=alternate,
                comparator=comparator, evidence=_evidence(row, captures, calibration),
                authority_id="fixture")
        forged_batch = replace(assembly.batch, candidate_manifest_digest="0" * 64)
        forged = replace(assembly, batch=forged_batch)
        with pytest.raises(vc.ValidationConsumerError, match="differs from active"):
            consumer.record_native_row(
                request_id="forged", assembly=forged, candidate=candidate,
                comparator=comparator, evidence=_evidence(row, captures, calibration),
                authority_id="fixture")
        assert all(item.status == "pending" for item in
                   cm.CandidateState.from_dict(
                       consumer.transactions.inspect()["state"]).active_batches[0].rows)
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


def test_separately_valid_lineages_cannot_form_one_row_pair(tmp_path):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        _, other, _, _, _ = _produce(
            controller, tmp_path, calibration, lineage="lineage-2",
            capture_store=stores[0])
        anchor_id, anchor = captures["anchor"]
        candidate_id, candidate_payload = other["candidate"]
        mixed = vc.NativeRowEvidence(
            row.row_id, anchor_id, anchor, candidate_id, candidate_payload, calibration)
        with pytest.raises(vc.ValidationConsumerError, match="comparison lineage"):
            consumer.record_native_row(
                request_id="mixed", assembly=assembly, candidate=candidate,
                comparator=comparator, evidence=mixed, authority_id="fixture")
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


def test_required_cpu_pass_gpu_missing_and_optional_seed_pending(tmp_path):
    values = _consumer(tmp_path, authority=True, target_backend="both")
    controller, consumer, candidate, comparator, cpu, _, _, captures, calibration, *stores = values
    gpu = replace(cpu, row_id="production-gpu", backend="gpu")
    seed = replace(cpu, row_id="optional-seed", row_kind="seed", required=False,
                   category="CANDIDATE")
    direct = cm.RequiredRowSet(
        "cpu-gpu-seed-v1", tuple(sorted((cpu, gpu, seed), key=lambda item: item.row_id)))
    rows = cm.RequiredRowSet.from_dict(direct.to_dict())
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        passed = consumer.record_native_row(
            request_id="cpu", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(cpu, captures, calibration),
            authority_id="fixture")
        assert passed.status == "passed"
        with pytest.raises(cm.TransitionError, match="pending/running required"):
            consumer.complete(request_id="too-soon", assembly=assembly)
        missing_gpu = consumer.record_native_row(
            request_id="gpu", assembly=assembly, candidate=candidate,
            comparator=comparator, evidence=_evidence(gpu, captures, calibration),
            authority_id="fixture")
        assert missing_gpu.status == "prerequisite_missing"
        completed = consumer.complete(request_id="complete", assembly=assembly)
        assert {item.row_id: item.status for item in completed.rows} == {
            "production-cpu": "passed", "production-gpu": "prerequisite_missing",
            "optional-seed": "pending"}
        with pytest.raises(cm.TransitionError, match="production-gpu.*not passed"):
            consumer.advance(
                request_id="advance", assembly=assembly, candidate=candidate,
                comparator=comparator, authority_id="fixture", loo_results={})
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()


@pytest.mark.parametrize("mutation", ["unit", "instrument", "backend"])
def test_wrong_calibration_instrument_or_backend_cannot_pass(tmp_path, mutation):
    values = _consumer(tmp_path, authority=True)
    controller, consumer, candidate, comparator, row, rows, _, captures, calibration, *stores = values
    try:
        assembly = consumer.assemble_due_batch(
            request_id="start", candidate=candidate, comparator=comparator, row_set=rows)
        if mutation == "unit":
            raw = calibration.to_dict()
            raw["unit"] = "session"
            calibration = ep.CalibrationReceipt.from_dict(raw)
        elif mutation == "instrument":
            bad = replace(row, instrument_digest="f" * 64)
            direct = cm.RequiredRowSet("bad", (bad,))
            rows = cm.RequiredRowSet.from_dict(direct.to_dict())
            # The frozen row set itself cannot be swapped at record time.
        else:
            bad = replace(row, backend="gpu")
            direct = cm.RequiredRowSet("bad", (bad,))
            rows = cm.RequiredRowSet.from_dict(direct.to_dict())
        if mutation == "unit":
            state = consumer.record_native_row(
                request_id="row", assembly=assembly, candidate=candidate, comparator=comparator,
                evidence=_evidence(row, captures, calibration), authority_id="fixture")
            assert state.status == "prerequisite_missing" and "calibration" in state.reason
        else:
            with pytest.raises(cm.TransitionError, match="row set does not match"):
                consumer.transactions.record_row(
                    request_id="wrong", batch_id=assembly.batch.batch_id, row_set=rows,
                    row_state=cm.ValidationRowState(bad.row_id, "failed", mutation, None))
    finally:
        stores[0].close()
        stores[1].close()
        controller.close()
