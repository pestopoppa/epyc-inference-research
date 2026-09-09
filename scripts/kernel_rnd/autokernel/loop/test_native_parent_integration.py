"""Native parent integration: actual tiny child/socket; hermetic resource/proc fixtures."""
from __future__ import annotations

from dataclasses import replace
import queue
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from . import campaign_control as cc
from . import experiment_plan as ep
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import native_parent_evidence as npe
from . import native_parent_receipt_replay as replay
from . import native_parent_service as service
from . import observation_binding as ob
from . import scheduling
from . import unified_worker as uw
from . import worker_lifecycle as wl
from .. import journal as journal_module
from .test_native_parent_evidence import _case, _result
from .test_unified_worker import _prepared, _start
from .test_driver_execution import _run_real_controller_child_v2_capture_and_restart


@pytest.fixture
def issued(tmp_path):
    case = _case(tmp_path)
    result, _receipt = _result(case)
    registry = replay.IssuedNativeEvidenceRegistry(
        artifact_root=case["store"].root, max_units=1)
    registry.record(producer=case["producer"], request=case["request"], result=result)
    context = case["context"]
    attempt = {"kind": "completed_attempt", "unit_id": context.unit_id,
        "stage_witnesses": {name: item.to_dict() for name, item
                            in result.completion.stage_witnesses.items()},
        "terminal": result.completion.terminal,
        "provider_recorded_screen": result.completion.recorded_screen,
        "recorded_screen": result.completion.recorded_screen}
    carrier = {"plan": context.plan.to_dict(), "capture_context": {
        "worker_id": context.fence.worker_id, "worker_incarnation": context.fence.worker_incarnation},
        "raw_artifacts": [{"document": ob._plain(case["native"])}, {"document": attempt}]}
    try:
        yield case, registry, carrier, result
    finally:
        case["store"].close()


def test_original_issued_receipt_reopens_and_replays_without_resampling(issued):
    case, registry, carrier, result = issued
    replayer = replay.NativeParentReceiptReplayer()
    with replayer.using(registry):
        replayer.replay(carrier, store=case["store"])
        replayer.replay(carrier, store=case["store"])
    assert case["producer"]._result is result
    assert len(case["producer"]._readbacks) == 1


@pytest.mark.parametrize("mutation", ["status", "reference", "added_witness",
    "missing_witness", "screen", "unit", "worker", "native"])
def test_child_mutation_cannot_reissue_parent_witness(issued, mutation):
    case, registry, carrier, _ = issued
    attempt = carrier["raw_artifacts"][1]["document"]
    if mutation == "status":
        attempt["stage_witnesses"]["purpose"] = {"status": "pass", "ref": "child:purpose"}
    elif mutation == "reference":
        attempt["stage_witnesses"]["identity"]["ref"] = "child:replacement"
    elif mutation == "added_witness":
        attempt["stage_witnesses"]["invented"] = {"status": "pass", "ref": "child:invented"}
    elif mutation == "missing_witness":
        del attempt["stage_witnesses"]["placement"]
    elif mutation == "screen":
        attempt["provider_recorded_screen"] = "clean"
    elif mutation == "unit":
        carrier["raw_artifacts"][0]["document"]["unit_id"] = "other"
    elif mutation == "worker":
        carrier["capture_context"]["worker_incarnation"] += 1
    else:
        carrier["raw_artifacts"][0]["document"]["value"] += 1
    with replay.NativeParentReceiptReplayer().using(registry) as replayer:
        with pytest.raises((replay.ParentReceiptRefused, npe.NativeEvidenceRefused)):
            replayer.replay(carrier, store=case["store"])


def test_artifact_presence_cannot_reconstruct_lost_parent_issuance(issued):
    case, _registry, carrier, _ = issued
    empty = replay.IssuedNativeEvidenceRegistry(artifact_root=case["store"].root, max_units=1)
    with replay.NativeParentReceiptReplayer().using(empty) as replayer:
        with pytest.raises(replay.ParentReceiptRefused, match="issuance"):
            replayer.replay(carrier, store=case["store"])


def test_replay_reopens_original_readback_bytes(issued):
    case, registry, carrier, result = issued
    body = case["store"].read(result.receipt.locator, result.receipt.sha256)
    raw = body["runtime_readbacks"][0]
    path = case["store"].root / raw["locator"]
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    try:
        with replay.NativeParentReceiptReplayer().using(registry) as replayer:
            with pytest.raises(mc.CaptureError, match="digest"):
                replayer.replay(carrier, store=case["store"])
    finally:
        path.write_bytes(original)


def test_parent_registry_rejects_unissued_result_object(issued):
    case, registry, _carrier, result = issued
    copied = npe.NativeUnitEvidenceResult(result.completion, result.receipt, result.receipt_digest)
    with pytest.raises(replay.ParentReceiptRefused, match="actual evaluated"):
        registry.record(producer=case["producer"], request=case["request"], result=copied)


def _assert_no_notice(authority):
    with pytest.raises(queue.Empty):
        authority.next_notice(timeout=0)


def test_reopen_omitted_and_explicit_empty_verifiers_are_identical(issued):
    case, _registry, _carrier, _result_value = issued
    context = case["context"]
    reference = ob.LifecycleObservationReference.from_dict(case["native"]["lifecycle_observation"])
    args = {"store": case["store"], "instrument": context.binding.instrument,
        "expected": {"unit_id": context.unit_id, "process_generation_id": context.unit.process_id,
            "fence_id": context.fence.fence_id, "active_claim_ref": context.binding.active_claim_ref,
            "container_id": context.binding.container_id,
            "capture_context": ob._plain(context.binding.worker_binding)}}
    assert ob.validate_reopened_observation(reference, **args) == ob.validate_reopened_observation(
        reference, **args, verifiers=ob.ParentObservationVerifiers())


@pytest.fixture
def phase_invocation(tmp_path, issued):
    case, _registry, _carrier, _result_value = issued
    context = case["context"]
    prepared = _prepared(tmp_path)
    authority = uw.ParentUnitEvidenceAuthority(max_records=4)
    invocation = uw.PlannedWorkerInvocation(prepared, authority)
    invocation.prepared = SimpleNamespace(schema=uw.PREPARED_SCHEMA_V2, plan=context.plan)
    invocation.start = _start(prepared)
    invocation._next = context.unit.order_index
    invocation._active = (context.unit.order_index + 1, context.fence)
    invocation._active_observation_binding = context.binding
    invocation._active_observation_target = {"binding_ref": wl._digest(ob._plain(context.descendant_event))}
    body = {"schema": uw.OBSERVATION_PHASE_REQUEST_SCHEMA, "nonce": invocation.start.nonce,
        "sequence": context.unit.order_index + 1, "unit_id": context.unit_id,
        "process_generation_id": context.unit.process_id, "fence_id": context.fence.fence_id,
        "binding_digest": context.binding.to_dict()["binding_digest"],
        "descendant_binding_ref": invocation._active_observation_target["binding_ref"],
        "phase": "health", "boundary_monotonic_s": time.monotonic()}
    try:
        yield invocation, authority, body | {"request_digest": uw._digest(body)}, case
    finally:
        invocation.close()


@pytest.mark.parametrize("field,value", [("schema", "v1"), ("nonce", "other"),
    ("sequence", True), ("unit_id", "other"), ("process_generation_id", "other"),
    ("fence_id", "other"), ("binding_digest", "b" * 64),
    ("descendant_binding_ref", "b" * 64), ("phase", "measurement"),
    ("boundary_monotonic_s", True), ("extra", "not-allowed")])
def test_phase_packet_exact_join_mutation_refused(phase_invocation, field, value):
    invocation, authority, packet, _case_value = phase_invocation
    packet[field] = value
    packet["request_digest"] = uw._digest({k: v for k, v in packet.items() if k != "request_digest"})
    with pytest.raises(uw.WorkerBridgeRefused):
        invocation.handle_observation_phase(packet)
    _assert_no_notice(authority)


def test_phase_duplicate_notice_reuses_one_cache_result_and_rejects_conflict(phase_invocation):
    invocation, authority, packet, _case_value = phase_invocation
    invocation.handle_observation_phase(packet)
    notice = authority.next_notice(timeout=0)
    invocation.handle_observation_phase(packet)
    _assert_no_notice(authority)
    authority.publish_observation_phase(notice["key"], {"outcome": "unavailable"})
    invocation.poll_evidence()
    invocation.handle_observation_phase(packet)
    _assert_no_notice(authority)
    assert invocation._pending_observation_phase is None
    packet["boundary_monotonic_s"] += .001
    packet["request_digest"] = uw._digest({k: v for k, v in packet.items() if k != "request_digest"})
    with pytest.raises(uw.WorkerBridgeRefused, match="retry conflicts"):
        invocation.handle_observation_phase(packet)
    with pytest.raises(uw.WorkerBridgeRefused, match="not a transport"):
        authority.publish_observation_phase(notice["key"], {"outcome": "pass"})


@pytest.mark.parametrize("mutation", ["inline", "missing_artifact", "extra_artifact", "digest", "sequence", "schema"])
def test_v2_completion_is_artifact_only_and_exact(phase_invocation, mutation):
    invocation, authority, _packet, case = phase_invocation
    packet = uw._artifact_completion_request(start=invocation.start,
        sequence=case["context"].unit.order_index + 1, fence=case["context"].fence,
        native_observation=case["request"]["native_observation"])
    if mutation == "inline":
        packet["observation"] = {"success": True}
    elif mutation == "missing_artifact":
        packet.pop("native_observation")
    elif mutation == "extra_artifact":
        packet["native_observation"]["inline"] = True
    elif mutation == "sequence":
        packet["sequence"] = True
    elif mutation == "schema":
        packet["schema"] = uw.UNIT_COMPLETION_REQUEST_SCHEMA
    packet["request_digest"] = ("0" * 64 if mutation == "digest" else
        uw._digest({k: v for k, v in packet.items() if k != "request_digest"}))
    with pytest.raises(uw.WorkerBridgeRefused):
        invocation.handle_completion(packet)
    _assert_no_notice(authority)


def test_completion_request_cannot_overtake_pending_health_notice(phase_invocation):
    invocation, _authority, packet, case = phase_invocation
    invocation.handle_observation_phase(packet)
    completion = uw._artifact_completion_request(start=invocation.start,
        sequence=case["context"].unit.order_index + 1, fence=case["context"].fence,
        native_observation=case["request"]["native_observation"])
    with pytest.raises(uw.WorkerBridgeRefused, match="no active unit"):
        invocation.handle_completion(completion)


def test_serialized_socket_exchanges_keep_responses_with_their_requests(tmp_path):
    prepared = _prepared(tmp_path)
    parent, child = socket.socketpair()
    authority = uw.InheritedUnitAuthority(child, start=replace(
        _start(prepared), provider_deadline=time.monotonic() + 5))
    results = {}
    errors = []
    def exchange(index):
        try:
            results[index] = authority._exchange({"index": index})["index"]
        except BaseException as exc:
            errors.append(exc)
    workers = [threading.Thread(target=exchange, args=(index,)) for index in range(2)]
    try:
        for worker in workers:
            worker.start()
        for _ in workers:
            request = uw.read_bounded_message(parent, deadline=time.monotonic() + 2)
            uw.write_bounded_message(parent, request, deadline=time.monotonic() + 2)
        for worker in workers:
            worker.join(timeout=2)
        assert not any(worker.is_alive() for worker in workers)
        assert not errors and results == {0: 0, 1: 1}
    finally:
        authority.close()
        parent.close()


def test_contended_exchange_cannot_extend_original_provider_deadline(tmp_path):
    prepared = _prepared(tmp_path)
    parent, child = socket.socketpair()
    authority = uw.InheritedUnitAuthority(child, start=replace(
        _start(prepared), provider_deadline=time.monotonic() + 5))
    authority._exchange_lock.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(uw.WorkerBridgeRefused, match="deadline expired"):
            authority._exchange({"bounded": True}, deadline=started + .02)
        assert authority.closed and time.monotonic() - started < .5
    finally:
        authority._exchange_lock.release()
        authority.close()
        parent.close()


def test_actual_tiny_child_artifact_phase_and_parent_replay_then_restart_duplicate(tmp_path, monkeypatch):
    """No fixture scorer: the actual parent producer is the only result issuer."""
    replayer = replay.NativeParentReceiptReplayer()
    registry_scopes = []
    services = []
    controllers = []
    injected = pytest.MonkeyPatch()
    original_validator_init = nc.NativeCaptureValidator.__init__
    original_enter = cc.CampaignController.__enter__

    def validator_init(self, *args, **kwargs):
        kwargs["parent_receipt_replayer"] = replayer
        original_validator_init(self, *args, **kwargs)

    def enter(self):
        result = original_enter(self)
        controllers.append(self)
        return result

    def producer_factory(authority, prepared, lifecycle, configuration):
        registry = replay.IssuedNativeEvidenceRegistry(
            artifact_root=prepared.artifact_root, max_units=len(prepared.plan.expected_units))
        scope = replayer.using(registry)
        scope.__enter__()
        registry_scopes.append(scope)
        root = tmp_path / "fixture-probe"
        probe = lo.FilesystemProbe(proc_root=root / "proc", sysfs_cpu_root=root / "cpu",
            boot_id_path=root / "boot", cgroup_root=root / "cgroup")
        producer = service.NativeParentEvidenceService(authority, prepared, lifecycle,
            configuration, registry=registry, runtime_probe=probe)
        services.append(producer)
        return producer

    injected.setattr(nc.NativeCaptureValidator, "__init__", validator_init)
    injected.setattr(cc.CampaignController, "__enter__", enter)
    try:
        _run_real_controller_child_v2_capture_and_restart(
            tmp_path, monkeypatch, producer_type=producer_factory)
        assert len(services) == 1 and services[0].stopped
        producer = services[0]
        assert len(producer._unit_producers) == len(producer.prepared.plan.expected_units)
        assert len(producer._phase_results) == len(producer.prepared.plan.expected_units)
        for actual in producer._unit_producers.values():
            result = actual._result
            assert result.completion.recorded_screen == "flagged_but_retained"
            assert result.completion.stage_witnesses["identity"].status == "pass"
            assert result.completion.stage_witnesses["request_completeness"].status == "pass"
            assert result.completion.stage_witnesses["correctness"] == ep.Witness("unknown", None)
            assert result.completion.stage_witnesses["contention"] == ep.Witness("unknown", None)
            assert len(actual._readbacks) == 1
        original = controllers[0]
        rows = journal_module.Journal(str(original.store / "journal")).read_all()
        native = [row for row in rows if row.kind == journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED]
        assert len(native) == 2 and all(row.payload["carrier"]["status"] == "diagnostic" for row in native)
        config = original._scheduler_engine.config
        engine = scheduling.SchedulerEngine(config, scheduling.initial_state(config, original.resolved.campaign_id))
        restarted = cc.CampaignController(original.resolved, original.store, snapshot_version=3,
            scheduler_engine=engine, readiness_check=lambda: (True, None),
            lifecycle_provider=original._lifecycle_provider)
        with restarted:
            assert restarted._native_validator is None
            before = len(restarted._journal.read_all())
            with restarted.native_capture_callback() as capture:
                for row in native:
                    assert capture(row.record_id, row.payload).record_id == row.record_id
            assert len(restarted._journal.read_all()) == before
    finally:
        for scope in reversed(registry_scopes):
            scope.__exit__(None, None, None)
        injected.undo()
