"""Selected transport only: no A2 reservation/event/nomination authority."""
from __future__ import annotations

from dataclasses import replace
import copy
import time

import pytest

from . import discovery_screen as discovery, experiment_plan as ep
from . import planned_unit_selection as selected, unified_worker as worker
from . import observation_binding as ob, serving
from . import native_capture_control as capture_control, measurement_capture as capture
from .test_unified_worker import _prepared, _start, Authority
from .test_driver_execution import (_as_observed_v2,
    _run_real_controller_child_v2_capture_and_restart)
from .test_discovery_screen import _pair, _plan


def _six_unit_transport(prepared, order=3, attempt_id="fixture-attempt-0"):
    """Predeclare six fixture units before execution; only observation is synthetic."""
    row = prepared.plan.to_dict()
    row.update(phase="discovery", record_class="discovery_screen", intended_use="nominate",
               protocol_ref=discovery.A2_PROTOCOL, protocol_status="ratified",
               stopping={"kind": "fixed_n", "n_per_arm": 3, "paired": False},
               required_witnesses=sorted(discovery.MANDATORY_WITNESSES))
    units = []
    for arm in ("anchor", "candidate"):
        original = next(unit.to_dict() for unit in prepared.plan.expected_units if unit.arm == arm)
        for index in range(3):
            unit = original | {"unit_id": f"selected-{arm}-{index}",
                "process_id": f"selected-process-{len(units)}", "order_index": len(units),
                "pair_id": None}
            units.append(unit)
    row["expected_units"] = units
    plan = ep.ExperimentPlan.from_dict(row)
    scope = selected.SelectedPlanUnitRange.from_plan(plan, order, order + 1)
    dispatch = selected.SelectedUnitDispatch(plan.digest, plan.target_revision, scope, attempt_id)
    body = prepared.body() | {"schema": worker.PREPARED_SCHEMA_V4,
        "plan": plan.to_dict(), "dispatch": dispatch.to_dict(),
        "capture_context_base": worker._plain(prepared.capture_context_base) | {
            "protocol_id": plan.protocol_ref, "protocol_status": plan.protocol_status}}
    return worker.PreparedPlannedServingStage.from_dict(
        body | {"prepared_digest": worker._digest(body)})


def test_range_is_full_plan_absolute_membership_and_immutable():
    plan = _plan(_pair())
    scope = selected.SelectedPlanUnitRange.from_plan(plan, 3, 4)
    assert scope.plan_digest == plan.digest
    assert scope.units(plan) == (plan.expected_units[3],)
    assert scope.units(plan)[0].order_index == 3
    assert scope.units(plan)[0].arm == "candidate"
    raw = scope.to_dict()
    raw["unit_ids"][0] = "caller-mutated"
    assert scope.unit_ids == (plan.expected_units[3].unit_id,)
    dispatch = selected.SelectedUnitDispatch(plan.digest, plan.target_revision, scope, "attempt-0")
    retry = replace(dispatch, attempt_id="attempt-1")
    assert dispatch.lineage_id != retry.lineage_id
    assert dispatch.selection == retry.selection
    assert dispatch.to_dict()["execution_authorized"] is False


@pytest.mark.parametrize("mutation", ["order", "foreign", "spec", "duplicate", "extra", "boolean"])
def test_rehashed_foreign_reordered_or_malformed_range_refuses(mutation):
    plan = _plan(_pair())
    row = selected.SelectedPlanUnitRange.from_plan(plan, 0, 2).to_dict()
    if mutation == "order":
        row["unit_ids"].reverse()
    elif mutation == "foreign":
        row["plan_digest"] = "a" * 64
    elif mutation == "spec":
        row["unit_specs_digest"] = "b" * 64
    elif mutation == "duplicate":
        row["unit_ids"][1] = row["unit_ids"][0]
    elif mutation == "extra":
        row["authority"] = True
    else:
        row["start_order"] = False
    row["selection_digest"] = selected._digest({key: value for key, value in row.items()
                                                if key != "selection_digest"})
    with pytest.raises(selected.SelectionRefused):
        selected.SelectedPlanUnitRange.from_dict(row).units(plan)


def test_units_returns_the_canonical_plan_that_was_checked():
    plan = _plan(_pair())
    original = plan.to_dict()
    scope = selected.SelectedPlanUnitRange.from_plan(plan, 3, 4)
    object.__setattr__(plan, "to_dict", lambda: copy.deepcopy(original))
    object.__setattr__(plan, "expected_units", tuple(reversed(plan.expected_units)))
    units = scope.units(plan)
    assert units[0] == ep.ExperimentPlan.from_dict(original).expected_units[3]
    assert units[0] != plan.expected_units[3]


def test_selected_dispatch_cannot_claim_authority_or_hide_multiple_units():
    plan = _plan(_pair())
    with pytest.raises(selected.SelectionRefused, match="exactly one"):
        selected.SelectedUnitDispatch(plan.digest, plan.target_revision,
            selected.SelectedPlanUnitRange.from_plan(plan, 0, 3), "attempt")
    row = selected.SelectedUnitDispatch(plan.digest, plan.target_revision,
        selected.SelectedPlanUnitRange.from_plan(plan, 0, 1), "attempt").to_dict()
    row["execution_authorized"] = True
    row["dispatch_digest"] = selected._digest({key: value for key, value in row.items()
                                               if key != "dispatch_digest"})
    with pytest.raises(selected.SelectionRefused, match="authority"):
        selected.SelectedUnitDispatch.from_dict(row)


def test_prepared_selected_cursor_request_and_legacy_schema_refusals(tmp_path):
    original, _ = _as_observed_v2(_prepared(tmp_path), serving._measure_once)
    prepared = _six_unit_transport(original)
    assert prepared.native_observed and prepared.unit_bounds == (3, 4)
    for schema in (worker.PREPARED_SCHEMA, worker.PREPARED_SCHEMA_V2, worker.PREPARED_SCHEMA_V3):
        row = prepared.body() | {"schema": schema}
        with pytest.raises(worker.WorkerBridgeRefused):
            worker.PreparedPlannedServingStage.from_dict(row | {"prepared_digest": worker._digest(row)})
    row = prepared.body() | {"previous": {}}
    with pytest.raises(worker.WorkerBridgeRefused):
        worker.PreparedPlannedServingStage.from_dict(row | {"prepared_digest": worker._digest(row)})
    dispatch = prepared.selected_dispatch
    start = _start(prepared)
    row = start.body() | {"request_id": dispatch.request_id, "lineage_id": dispatch.lineage_id,
                          "stage_id": dispatch.stage_id}
    start = worker.WorkerStart.from_dict(row | {"start_digest": worker._digest(row)})
    authority = Authority()
    provider = worker.OwnedWorkerStageProvider(prepared=prepared, start=start,
        authority=authority, membership_probe=lambda _: None, clock=lambda: 1.0)
    with pytest.raises(worker.WorkerBridgeRefused, match="out of order"):
        provider.admit(prepared.plan.digest, prepared.plan.expected_units[0], worker.ps.STAGES)
    unit = prepared.selected_range.units(prepared.plan)[0]
    provider.admit(prepared.plan.digest, unit, worker.ps.STAGES)
    assert authority.units == [(4, unit.unit_id, None)]
    parent = worker.PlannedWorkerInvocation.open(prepared, worker.ParentUnitEvidenceAuthority())
    try:
        request = parent.stage_request(request_id=dispatch.request_id,
            lineage_id=dispatch.lineage_id, stage_id=dispatch.stage_id, control_revision=1)
        parent.validate_request(request)
        with pytest.raises(worker.WorkerBridgeRefused, match="selected unit"):
            parent.validate_request(replace(request, lineage_id="other-unit"))
        parent.start = start
        payload = {"schema": worker.UNIT_REQUEST_SCHEMA, "nonce": start.nonce,
            "sequence": 4, "plan_digest": prepared.plan.digest, "lineage_id": start.lineage_id,
            "unit": unit.to_dict(), "prior_completion_digest": None}
        assert parent.validate_unit_request(payload | {"request_digest": worker._digest(payload)})[0] == 4
        parent._next = 4
        with pytest.raises(worker.WorkerBridgeRefused, match="order"):
            parent.validate_unit_request(payload | {"request_digest": worker._digest(payload)})
    finally:
        parent.close()


def test_loaded_transport_default_changes_are_pinned_without_stripping_configuration(monkeypatch):
    identity = selected.loaded_source_identity()
    assert all(row["explicit_configuration_digest"] for row in identity["callables"])
    changed = dict(worker.run_prepared_stage.__kwdefaults__)
    changed["clock"] = lambda: 1.0
    monkeypatch.setattr(worker.run_prepared_stage, "__kwdefaults__", changed)
    assert selected.loaded_source_identity() != identity
    changed["clock"] = iter(())  # nonserializable default cannot claim a stable pin
    monkeypatch.setattr(worker.run_prepared_stage, "__kwdefaults__", changed)
    with pytest.raises(selected.SelectionRefused, match="unproven"):
        selected.loaded_source_identity()


def test_source_guard_rejects_callable_object_with_borrowed_python_code(monkeypatch):
    original = worker.run_prepared_stage
    class NonPythonCallable:
        __module__ = original.__module__
        __qualname__ = original.__qualname__
        __code__ = original.__code__
        def __call__(self, *args, **kwargs):
            pytest.fail("non-Python source probe must never execute")
    monkeypatch.setattr(worker, "run_prepared_stage", NonPythonCallable())
    with pytest.raises(selected.SelectionRefused, match="actual Python function"):
        selected.loaded_source_identity()


def test_direct_runner_rejects_multiple_selected_units_before_admission(tmp_path):
    original, _ = _as_observed_v2(_prepared(tmp_path), serving._measure_once)
    prepared = _six_unit_transport(original)
    class ForbiddenProvider:
        def admit(self, *_args):
            pytest.fail("multi-unit transport reached admission")
    def forbidden(*_args, **_kwargs):
        pytest.fail("multi-unit transport reached capture or measurement")
    with pytest.raises(worker.ps.PlannedServingError, match="exactly one"):
        worker.ps.run_planned_comparison(prepared.plan,
            anchor_template=prepared.runtime_pair.anchor.template,
            candidate_template=prepared.runtime_pair.candidate.template,
            anchor_recipe=prepared.runtime_pair.anchor,
            candidate_recipe=prepared.runtime_pair.candidate, prompts=prepared.prompts,
            stage_provider=ForbiddenProvider(), artifact_sink=forbidden,
            lineage_id="refused-before-admission", measure=forbidden,
            observation_session_factory=object(),
            selected_range=selected.SelectedPlanUnitRange.from_plan(prepared.plan, 0, 2))


@pytest.mark.parametrize("order", [0, 3, 5])
def test_actual_native_child_runs_only_selected_original_unit(tmp_path, monkeypatch, order):
    from . import native_parent_service as service, native_parent_receipt_replay as replay
    from . import lifecycle_observation as lo
    from pathlib import Path
    producers = []
    scopes = []
    replayer = replay.NativeParentReceiptReplayer()
    injected = pytest.MonkeyPatch()
    original_init = capture_control.NativeCaptureValidator.__init__

    def validator_init(self, *args, **kwargs):
        kwargs["parent_receipt_replayer"] = replayer
        original_init(self, *args, **kwargs)

    def producer_factory(authority, prepared, lifecycle, configuration):
        root = tmp_path / "fixture-probe"
        probe = lo.FilesystemProbe(proc_root=root / "proc", sysfs_cpu_root=root / "cpu",
            boot_id_path=root / "boot", cgroup_root=root / "cgroup")
        registry = replay.IssuedNativeEvidenceRegistry(artifact_root=prepared.artifact_root,
                                                      max_units=1)
        scope = replayer.using(registry)
        scope.__enter__()
        scopes.append(scope)
        producer = service.NativeParentEvidenceService(authority, prepared, lifecycle,
            configuration, registry=registry, runtime_probe=probe)
        producers.append(producer)
        return producer

    def inspect_result(*, prepared, start, terminal, fence, reference, result):
        # Hostile, self-consistent result bytes: original actual parent fence is
        # retained. Rehashed envelopes are not positive terminal/held authority.
        store = capture.ArtifactStore(prepared.artifact_root)
        try:
            for mutation in ("order", "process", "arm", "prompt", "scientific-completeness"):
                body = result.to_dict()
                body.pop("result_digest")
                raw = body["run"]["raw_units"][0]
                if mutation == "order":
                    raw["observed_order_index"] = (raw["observed_order_index"] + 1) % 6
                elif mutation == "process":
                    raw["process_id"] = "foreign-original-process"
                elif mutation == "arm":
                    raw["arm"] = "candidate" if raw["arm"] == "anchor" else "anchor"
                elif mutation == "prompt":
                    raw["prompt_ids"] = ["foreign-original-prompt"]
                else:
                    body["run"]["execution_complete"] = True
                    body["run"]["admissible_view"]["complete"] = True
                hostile = worker.PlannedWorkerResult.from_dict(
                    body | {"result_digest": worker._digest(body)})
                artifact = store.write(
                    f"planned-worker-result:{prepared.prepared_digest}:{start.nonce}",
                    hostile.to_dict())
                hostile_ref = replace(reference, result_digest=hostile.result_digest,
                    result_locator=artifact.locator, result_sha256=artifact.sha256)
                hostile_terminal = replace(terminal, result_digest=worker._digest(hostile_ref.to_dict()))
                with pytest.raises(worker.WorkerBridgeRefused,
                                   match="original absolute unit|scientific completeness"):
                    worker.reopen_deferred_result(hostile_ref, prepared=prepared, start=start,
                                                  terminal=hostile_terminal, fence=fence)
        finally:
            store.close()

    injected.setattr(capture_control.NativeCaptureValidator, "__init__", validator_init)
    try:
        prepared, result, pids = _run_real_controller_child_v2_capture_and_restart(
            tmp_path, monkeypatch, producer_type=producer_factory,
            prepare_transport=lambda original: _six_unit_transport(original, order),
            inspect_result=inspect_result)
    finally:
        for scope in reversed(scopes):
            scope.__exit__(None, None, None)
        injected.undo()
    assert len(pids) == 1  # actual spawned descendant; helper proves it is dead
    scope = prepared.selected_range
    unit = scope.units(prepared.plan)[0]
    assert len(prepared.plan.expected_units) == 6
    assert result.body["schema"] == worker.RESULT_SCHEMA_V3
    assert result.body["plan_digest"] == prepared.plan.digest
    assert result.body["completed_unit_ids"] == (unit.unit_id,)
    raw = result.body["run"]["raw_units"][0]
    assert (raw["unit_id"], raw["process_id"], raw["observed_order_index"]) == (
        unit.unit_id, unit.process_id, order)
    assert result.body["run"]["execution_complete"] is False
    assert result.body["run"]["selected_range_complete"] is True
    assert result.body["run"]["admissible_view"]["complete"] is False
    assert len(producers[0]._unit_producers) == 1
    captured = result.body["captures"][0]
    carrier = worker._plain(captured["payload"]["carrier"])
    assert carrier["plan"] == prepared.plan.to_dict()
    assert carrier["lineage_id"] == prepared.selected_dispatch.lineage_id
    assert carrier["status"] == "diagnostic"
    assert raw["witnesses"]["inference_exclusion"] == {"status": "unknown", "ref": None}
    assert raw["witnesses"]["frequency_power_envelope"] == {"status": "unknown", "ref": None}
    capture_identity = {"producer": capture.PRODUCER_ID_V2,
        "plan_digest": prepared.plan.digest, "lineage_id": prepared.selected_dispatch.lineage_id,
        "arm": unit.arm, "capture_schema": capture.CAPTURE_SCHEMA_V2,
        "instrument_identity_sha256": prepared.plan.loaded_instrument["identity_sha256"]}
    assert capture.schemas.content_hash(capture_identity) == result.body["captures"][0]["measurement_id"]
    next_scope = selected.SelectedPlanUnitRange.from_plan(prepared.plan, 4, 5)
    next_dispatch = selected.SelectedUnitDispatch(prepared.plan.digest,
        prepared.plan.target_revision, next_scope, prepared.selected_dispatch.attempt_id)
    if order != 4:
        assert capture.schemas.content_hash(capture_identity | {
            "lineage_id": next_dispatch.lineage_id}) != result.body["captures"][0]["measurement_id"]
    store = capture.ArtifactStore(prepared.artifact_root)
    validator = capture_control.NativeCaptureValidator(binding=capture_control.NativeCaptureBinding(
        prepared.plan.campaign_id, prepared.capture_context_base["config_digest"],
        prepared.capture_context_base["config_generation"],
        prepared.capture_context_base["supervisor_id"],
        prepared.capture_context_base["supervisor_incarnation"]), store=store,
        fence_provider=lambda *_args: None)
    try:
        for mutation in ("legacy-schema", "foreign-range"):
            changed = copy.deepcopy(carrier)
            artifact = next(item["document"] for item in changed["raw_artifacts"]
                            if item["document"]["kind"] == "completed_attempt")
            artifact.pop("artifact_digest")
            if mutation == "legacy-schema":
                artifact["schema"] = worker.ps.ARTIFACT_SCHEMA_V2
            else:
                artifact["selected_range"] = next_scope.to_dict()
            artifact["artifact_digest"] = capture.schemas.content_hash(artifact)
            with pytest.raises(capture_control.NativeCaptureRefused,
                               match="shape is not closed|range/membership"):
                validator._verify_artifacts(result.body["captures"][0]["measurement_id"],
                    changed, worker._plain(result.body["captures"][0]["artifact"]))
    finally:
        store.close()
    for schema in (worker.RESULT_SCHEMA, worker.RESULT_SCHEMA_V2):
        forged = result.to_dict() | {"schema": schema}
        forged.pop("result_digest")
        with pytest.raises(worker.WorkerBridgeRefused):
            worker.PlannedWorkerResult.from_dict(forged | {"result_digest": worker._digest(forged)})
    for mutation in ("completion", "range", "overlap"):
        forged = result.to_dict()
        forged.pop("result_digest")
        if mutation == "completion":
            forged["run"]["selected_range_complete"] = False
        elif mutation == "range":
            forged["selected_range"] = next_scope.to_dict()
            forged["run"]["selected_range"] = next_scope.to_dict()
        else:
            forged["completed_unit_ids"] *= 2
        with pytest.raises(worker.WorkerBridgeRefused):
            worker.PlannedWorkerResult.from_dict(forged | {"result_digest": worker._digest(forged)})
    run = result.body["run"]
    incomplete = worker.ps.PlannedServingRun(worker.ps.RUN_SCHEMA_V3,
        prepared.plan.digest, prepared.prompts.digest, prepared.selected_dispatch.lineage_id,
        prepared.plan.anchor_identity, prepared.plan.candidate_identity, (),
        ep.admissible_units(prepared.plan, ()), "policy_undefined", False, "unit stopped",
        selected_range=scope.to_dict())
    assert run["selected_range_complete"] and not incomplete.to_dict()["selected_range_complete"]
    assert Path(prepared.artifact_root).is_dir()


def test_source_identity_still_matches_native_clock_and_full_closure():
    identity = ob.loaded_planned_serving_identity(measurement_callable=serving._measure_once,
        fence_clock=time.monotonic, serving_timer=time.time)
    assert identity["configuration_complete"]
    assert identity["used_constants"]["selected_unit_transport"]["execution_authorized"] is False
