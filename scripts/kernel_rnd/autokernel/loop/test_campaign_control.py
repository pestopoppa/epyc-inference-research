"""Hermetic campaign control, replay, lock, and admission-fence tests."""
from __future__ import annotations

from dataclasses import replace
import json
import os
import threading

import pytest

from .. import journal as journal_module
from . import campaign, campaign_control as control
from .test_campaign import _manifest, _registry, _target


def _resolved(campaign_id="unified-ak-test"):
    raw = _manifest(production=[_target("prod")])
    raw["campaign_id"] = campaign_id
    return campaign.resolve_manifest(campaign.CampaignManifest.from_dict(raw),
                                     registry_snapshot=_registry())


def _command(resolved, request_id, operation, revision, generation=1):
    digest = control.command_digest(operation=operation, payload={},
                                    campaign_id=resolved.campaign_id,
                                    config_generation=generation)
    return {"schema": control.COMMAND_SCHEMA, "campaign_id": resolved.campaign_id,
            "config_generation": generation, "request_id": request_id,
            "operation": operation, "payload": {}, "payload_digest": digest,
            "expected_control_revision": revision}


def test_controller_lock_is_exclusive_and_lock_inode_is_retained(tmp_path):
    store = tmp_path / "service"
    first = control.CampaignController(_resolved(), store)
    first.__enter__()
    inode = (store / ".supervisor.lock").stat().st_ino
    try:
        with pytest.raises(control.ControlRefused, match="another campaign"):
            control.CampaignController(_resolved(), store).__enter__()
    finally:
        first.close()
    assert (store / ".supervisor.lock").stat().st_ino == inode


def test_double_enter_same_instance_refuses_without_reacquiring(tmp_path):
    controller = control.CampaignController(_resolved(), tmp_path / "service")
    controller.__enter__()
    fd = controller._lease_fd
    try:
        with pytest.raises(control.ControlRefused, match="single-lifetime"):
            controller.__enter__()
        assert controller._lease_fd == fd
    finally:
        controller.close()


def test_symlink_lock_is_refused(tmp_path):
    store = tmp_path / "service"
    store.mkdir(mode=0o700)
    (store / "target").write_text("")
    (store / ".supervisor.lock").symlink_to("target")
    with pytest.raises(control.ControlRefused, match="symlink"):
        control.CampaignController(_resolved(), store).__enter__()


def test_initial_pause_and_restart_increment_numeric_stream_identity(tmp_path):
    store = tmp_path / "service"
    with control.CampaignController(_resolved(), store) as first:
        one = first.publish_snapshot()
        two = first.publish_snapshot()
        assert one["observed_state"] == one["desired_state"] == "paused"
        assert one["active_worker"] is None
        assert two["sequence"] == one["sequence"] + 1
        assert two["last_scientific_result_at"] is None
    with control.CampaignController(_resolved(), store) as second:
        three = second.publish_snapshot()
        assert three["stream_epoch"] == one["stream_epoch"] + 1
        assert three["supervisor_incarnation"] == one["supervisor_incarnation"] + 1
        assert three["sequence"] == 1


def test_resume_waits_for_injected_prerequisite_and_duplicate_is_idempotent(tmp_path):
    store = tmp_path / "service"
    with control.CampaignController(_resolved(), store) as controller:
        command = _command(controller.resolved, "resume-1", "resume", 0)
        result = controller.apply_command(command)
        duplicate = dict(command, expected_control_revision=999)
        assert controller.apply_command(duplicate) == result
        assert result["accepted"] is True and result["completed"] is False
        assert result["observed_state"] == "waiting_prerequisite"
        assert controller.control_revision == 1
    with control.CampaignController(_resolved(), store) as replayed:
        assert replayed.command_results["resume-1"]["completed"] is False
        assert replayed.observed_state == "waiting_prerequisite"


def test_drain_is_durable_terminal_and_old_resume_cannot_erase_it(tmp_path):
    store = tmp_path / "service"
    resolved = _resolved()
    with control.CampaignController(resolved, store) as controller:
        drained = controller.apply_command(_command(resolved, "drain-1", "drain", 0))
        assert drained["completed"] is True
    with control.CampaignController(resolved, store) as restarted:
        assert restarted.desired_state == restarted.observed_state == "drained"
        with pytest.raises(control.ControlRefused, match="stale control"):
            restarted.apply_command(_command(resolved, "old-resume", "resume", 0))
        with pytest.raises(control.ControlRefused, match="terminal"):
            restarted.apply_command(_command(resolved, "new-resume", "resume", 1))


def test_duplicate_changed_semantics_and_new_stale_revision_refuse(tmp_path):
    resolved = _resolved()
    with control.CampaignController(resolved, tmp_path / "service") as controller:
        controller.apply_command(_command(resolved, "same", "pause", 0))
        with pytest.raises(control.ControlRefused, match="different semantics"):
            controller.apply_command(_command(resolved, "same", "resume", 1))
        with pytest.raises(control.ControlRefused, match="current=1"):
            controller.apply_command(_command(resolved, "new", "pause", 0))


def test_different_generation_or_campaign_store_refuses(tmp_path):
    store = tmp_path / "service"
    resolved = _resolved()
    with control.CampaignController(resolved, store):
        pass
    with pytest.raises(control.ControlRefused, match="different campaign/config"):
        control.CampaignController(resolved, store, config_generation=2).__enter__()
    changed = replace(resolved, campaign_id="different")
    with pytest.raises(control.ControlRefused, match="another campaign identity"):
        control.CampaignController(changed, store).__enter__()


def test_journal_failure_happens_before_ack_or_state_change(tmp_path, monkeypatch):
    resolved = _resolved()
    with control.CampaignController(resolved, tmp_path / "service") as controller:
        assert controller._journal is not None
        monkeypatch.setattr(controller._journal, "append",
                            lambda *_a, **_k: (_ for _ in ()).throw(OSError("fsync")))
        with pytest.raises(OSError, match="fsync"):
            controller.apply_command(_command(resolved, "pause", "pause", 0))
        assert controller.control_revision == 0
        with pytest.raises(control.ControlRefused, match="poisoned"):
            controller.apply_command(_command(resolved, "retry", "pause", 0))
    monkeypatch.undo()
    with control.CampaignController(resolved, tmp_path / "service") as replayed:
        assert replayed.control_revision == 0
        assert "pause" not in replayed.command_results


def test_uncertain_failure_after_durable_write_requires_replay(tmp_path, monkeypatch):
    resolved = _resolved()
    store = tmp_path / "service"
    with control.CampaignController(resolved, store) as controller:
        assert controller._journal is not None
        real_append = controller._journal.append

        def append_then_fault(*args, **kwargs):
            real_append(*args, **kwargs)
            raise OSError("return path lost after fsync")

        monkeypatch.setattr(controller._journal, "append", append_then_fault)
        with pytest.raises(OSError, match="after fsync"):
            controller.apply_command(_command(resolved, "uncertain", "pause", 0))
        with pytest.raises(control.ControlRefused, match="poisoned"):
            controller.publish_snapshot()
    monkeypatch.undo()
    with control.CampaignController(resolved, store) as replayed:
        assert replayed.control_revision == 1
        assert replayed.command_results["uncertain"]["accepted"] is True


def test_projection_failure_replays_durable_acceptance(tmp_path, monkeypatch):
    resolved = _resolved()
    store = tmp_path / "service"
    with control.CampaignController(resolved, store) as controller:
        monkeypatch.setattr(control.status, "write_json",
                            lambda *_a, **_k: (_ for _ in ()).throw(OSError("projection")))
        with pytest.raises(OSError, match="projection"):
            controller.apply_command(_command(resolved, "pause", "pause", 0))
    monkeypatch.undo()
    with control.CampaignController(resolved, store) as replayed:
        assert replayed.control_revision == 1
        assert replayed.command_results["pause"]["accepted"] is True
        retry = _command(resolved, "pause", "pause", 999)
        assert replayed.apply_command(retry) == replayed.command_results["pause"]


def test_corrupt_history_fails_closed(tmp_path):
    store = tmp_path / "service"
    with control.CampaignController(_resolved(), store):
        pass
    with (store / "journal" / "events.jsonl").open("ab") as stream:
        stream.write(b"{not-json}\n")
        stream.flush()
        os.fsync(stream.fileno())
    with pytest.raises(Exception, match="unreadable line"):
        control.CampaignController(_resolved(), store).__enter__()


def test_unsupported_supervisor_event_schema_fails_closed(tmp_path):
    store = tmp_path / "service"
    with control.CampaignController(_resolved(), store):
        pass
    path = store / "journal" / "events.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["payload"]["schema"] = "epyc.autokernel.campaign_supervisor_event.v99"
    rows[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="invalid campaign supervisor history"):
        control.CampaignController(_resolved(), store).__enter__()


def test_foreign_campaign_journal_without_supervisor_event_is_not_claimed(tmp_path):
    store = tmp_path / "service"
    store.mkdir(mode=0o700)
    foreign = journal_module.Journal(str(store / "journal"), campaign_id="foreign")
    foreign.initialize()
    foreign.append_control_ack(control="pause", control_id="foreign-1",
                               received_at="2026-09-09T00:00:00Z",
                               disposition="accepted")
    with pytest.raises(control.ControlRefused, match="another campaign identity"):
        control.CampaignController(_resolved(), store).__enter__()


def test_supervisor_event_unknown_data_and_nonmonotonic_start_refuse(tmp_path):
    for field, value, match in (("extra", True, "START requires exactly"),
                                ("stream_epoch", 9, "monotonic replay")):
        store = tmp_path / field
        with control.CampaignController(_resolved(), store):
            pass
        path = store / "journal" / "events.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        if field == "extra":
            row["payload"]["data"][field] = value
        else:
            row["payload"][field] = value
        path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
                        encoding="utf-8")
        with pytest.raises(Exception, match=match):
            control.CampaignController(_resolved(), store).__enter__()


def test_valid_control_event_before_first_start_is_semantically_refused(tmp_path):
    resolved = _resolved()
    source_store = tmp_path / "source"
    with control.CampaignController(resolved, source_store) as source:
        source.apply_command(_command(resolved, "pause-source", "pause", 0))
        assert source._journal is not None
        payload = dict(source._journal.read_all()[-1].payload)
    target_store = tmp_path / "target"
    target_store.mkdir(mode=0o700)
    target = journal_module.Journal(str(target_store / "journal"),
                                    campaign_id=resolved.campaign_id)
    target.initialize()
    target.append(journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, payload)
    with pytest.raises(Exception, match="monotonic replay"):
        control.CampaignController(resolved, target_store).__enter__()


def test_replay_refuses_valid_resume_event_after_terminal_drain(tmp_path):
    resolved = _resolved()
    drained_store = tmp_path / "drained"
    with control.CampaignController(resolved, drained_store) as drained:
        drained.apply_command(_command(resolved, "drain", "drain", 0))
    source_store = tmp_path / "source-resume"
    with control.CampaignController(resolved, source_store,
                                    readiness_check=lambda: (False, "waiting")) as source:
        source.apply_command(_command(resolved, "pause-first", "pause", 0))
        source.apply_command(_command(resolved, "resume-after", "resume", 1))
        assert source._journal is not None
        resume_payload = dict(source._journal.read_all()[-1].payload)
    journal = journal_module.Journal(str(drained_store / "journal"),
                                     campaign_id=resolved.campaign_id)
    journal.append(journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, resume_payload)
    with pytest.raises(Exception, match="terminal drain"):
        control.CampaignController(resolved, drained_store).__enter__()


@pytest.mark.parametrize("override,reason", [
    ({"desired_state": "paused"}, "admissions_closed"),
    ({"control_revision": 1}, "stale_control_revision"),
    ({"supervisor_incarnation": 1}, "stale_supervisor_incarnation"),
    ({"grant": control.TrustedGrant("g", 1, 100, revoked=True)}, "grant_revoked"),
    ({"grant": control.TrustedGrant("g", 1, 100, renewal_ok=False)}, "renewal_failed"),
    ({"grant_identity": "old"}, "stale_grant"),
    ({"grant_generation": 0}, "invalid_expected_grant_generation"),
    ({"grant": control.TrustedGrant("g", 1, 14)}, "insufficient"),
    ({"dependency_check": lambda: None}, "dependencies_unknown"),
    ({"dependency_check": lambda: (_ for _ in ()).throw(RuntimeError("fault"))},
     "callback_failed"),
    ({"dependency_check": lambda: False}, "dependencies_unsatisfied"),
])
def test_stage_admission_denials(override, reason):
    args = {"desired_state": "running", "current_control_revision": 2,
            "control_revision": 2, "current_supervisor_incarnation": 3,
            "supervisor_incarnation": 3,
            "grant": control.TrustedGrant("g", 1, 100), "now": 10,
            "grant_identity": "g", "grant_generation": 1,
            "max_stage_seconds": 3, "teardown_seconds": 2,
            "dependency_check": lambda: True}
    args.update(override)
    decision = control.may_start_stage(**args)
    assert decision.allowed is False and reason in decision.reason


def test_stage_admission_requires_stage_plus_teardown_and_allows_exact_fit():
    decision = control.may_start_stage(
        desired_state="running", current_control_revision=2, control_revision=2,
        current_supervisor_incarnation=3, supervisor_incarnation=3,
        grant=control.TrustedGrant("grant", 1, 15), now=10,
        grant_identity="grant", grant_generation=1,
        max_stage_seconds=3, teardown_seconds=2, dependency_check=lambda: True)
    assert decision == control.AdmissionDecision(True, "admitted")


def test_complete_resolved_identity_not_request_digest_binds_store(tmp_path):
    manifest = campaign.CampaignManifest.from_dict(
        _manifest(production=[_target("prod")]))
    first = campaign.resolve_manifest(manifest, registry_snapshot=_registry())
    moved_registry = _registry()
    moved_registry["model"]["model-a"]["sha256"] = "f" * 64
    moved = campaign.resolve_manifest(manifest, registry_snapshot=moved_registry)
    assert moved.manifest_digest == first.manifest_digest
    assert control.resolved_config_digest(moved) != control.resolved_config_digest(first)
    store = tmp_path / "service"
    with control.CampaignController(first, store):
        pass
    with pytest.raises(control.ControlRefused, match="different campaign/config"):
        control.CampaignController(moved, store).__enter__()


def test_direct_noncanonical_resolved_campaign_refuses_at_boundary(tmp_path):
    resolved = _resolved()
    malformed = replace(resolved, targets=list(resolved.targets))
    with pytest.raises(control.ControlRefused, match="canonical normalized"):
        control.CampaignController(malformed, tmp_path / "service")


def test_closed_instance_cannot_publish_command_or_admit_against_new_owner(tmp_path):
    resolved = _resolved()
    store = tmp_path / "service"
    old = control.CampaignController(resolved, store)
    old.__enter__()
    old.close()
    assert old._journal is None and old._lease_fd is None
    with control.CampaignController(resolved, store) as current:
        for operation in (
                lambda: old.apply_command(_command(resolved, "old", "resume", 0)),
                old.snapshot, old.publish_snapshot,
                lambda: old.command_results, lambda: old.producer_build,
                lambda: old.may_start_stage(
                    control_revision=0, supervisor_incarnation=1,
                    grant=control.TrustedGrant("g", 1, 100), grant_identity="g",
                    grant_generation=1, now=0, max_stage_seconds=1,
                    teardown_seconds=0, dependency_check=lambda: True)):
            with pytest.raises(control.ControlRefused, match="closed"):
                operation()
        assert current.control_revision == 0


def test_close_serializes_with_inflight_command(tmp_path):
    resolved = _resolved()
    entered = threading.Event()
    release = threading.Event()

    def readiness():
        entered.set()
        assert release.wait(2)
        return False, "not authorized"

    controller = control.CampaignController(resolved, tmp_path / "service",
                                            readiness_check=readiness)
    controller.__enter__()
    results = []
    worker = threading.Thread(target=lambda: results.append(controller.apply_command(
        _command(resolved, "resume-race", "resume", 0))))
    closer = threading.Thread(target=controller.close)
    worker.start()
    assert entered.wait(1)
    closer.start()
    assert closer.is_alive()
    release.set()
    worker.join(2)
    closer.join(2)
    assert not worker.is_alive() and not closer.is_alive()
    assert results[0]["accepted"] is True
    with pytest.raises(control.ControlRefused, match="closed"):
        controller.snapshot()


def test_snapshots_are_deep_copies_and_do_not_rescan_journal(tmp_path, monkeypatch):
    resolved = _resolved()
    with control.CampaignController(resolved, tmp_path / "service") as controller:
        controller.apply_command(_command(resolved, "pause-copy", "pause", 0))
        assert controller._journal is not None
        monkeypatch.setattr(controller._journal, "read_all",
                            lambda: (_ for _ in ()).throw(AssertionError("rescanned")))
        first = controller.publish_snapshot()
        first["command_results"][0]["desired_state"] = "corrupted"
        first["producer_build"]["sha256"] = "0" * 64
        second = controller.publish_snapshot()
        assert second["command_results"][0]["desired_state"] == "paused"
        assert second["producer_build"]["sha256"] != "0" * 64
        assert second["journal_cursor"] == first["journal_cursor"]
        assert second["producer_build"]["identity_basis"] == (
            "loaded_callable_bytecode_and_constants_sha256")


def test_producer_build_manifest_includes_accessors_and_marks_generated_methods():
    first = control._loaded_producer_build_identity()
    second = control._loaded_producer_build_identity()
    assert first == second
    symbols = first["included_symbols"]
    assert ("property_fget:CampaignController.command_results.fget" in symbols)
    assert ("property_fget:CampaignController.producer_build.fget" in symbols)
    assert "generated_method:TrustedGrant.__init__" in symbols
    assert first["scope"] == (
        "campaign_control_callable_bytecode_and_selected_constants")


def test_malformed_operation_and_untyped_grant_refuse_without_typeerror(tmp_path):
    resolved = _resolved()
    row = _command(resolved, "bad", "pause", 0)
    row["operation"] = []
    with pytest.raises(control.ControlRefused, match="operation"):
        control.validate_command(row)
    decision = control.may_start_stage(
        desired_state="running", current_control_revision=0, control_revision=0,
        current_supervisor_incarnation=1, supervisor_incarnation=1,
        grant={"identity": "fake"}, grant_identity="fake", grant_generation=1,
        now=0, max_stage_seconds=1, teardown_seconds=0,
        dependency_check=lambda: True)
    assert decision == control.AdmissionDecision(False, "malformed_untrusted_grant")


def test_native_event_validator_types_malformed_operation_without_typeerror(tmp_path):
    resolved = _resolved()
    with control.CampaignController(resolved, tmp_path / "service") as controller:
        controller.apply_command(_command(resolved, "valid", "pause", 0))
        assert controller._journal is not None
        event = controller._journal.read_all()[-1]
        payload = json.loads(json.dumps(event.payload))
        payload["data"]["command"]["operation"] = []
        with pytest.raises(ValueError, match="operation"):
            controller._journal.append(journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT,
                                       payload)


def test_consumer_fence_rejects_bool_revision_and_nonbool_grant_flags():
    base = {"desired_state": "running", "current_control_revision": 1,
            "control_revision": 1, "current_supervisor_incarnation": 1,
            "supervisor_incarnation": 1,
            "grant": control.TrustedGrant("g", 1, 10), "grant_identity": "g",
            "grant_generation": 1, "now": 0, "max_stage_seconds": 1,
            "teardown_seconds": 0, "dependency_check": lambda: True}
    for key, value in (("current_control_revision", True), ("control_revision", True),
                       ("current_supervisor_incarnation", True),
                       ("supervisor_incarnation", "1")):
        args = dict(base, **{key: value})
        assert control.may_start_stage(**args).reason == "invalid_fence_revision"
    for grant in (control.TrustedGrant("g", 1, 10, revoked=0),
                  control.TrustedGrant("g", 1, 10, renewal_ok=1)):
        assert control.may_start_stage(**dict(base, grant=grant)).reason == (
            "invalid_grant_state")


def test_native_validator_returns_violations_for_arbitrary_nested_json(tmp_path):
    resolved = _resolved()
    with control.CampaignController(resolved, tmp_path / "service") as controller:
        controller.apply_command(_command(resolved, "valid-nested", "pause", 0))
        assert controller._journal is not None
        entries = controller._journal.read_all()
        start = json.loads(json.dumps(entries[0].payload))
        accepted = json.loads(json.dumps(entries[-1].payload))
    cases = []
    event_list = json.loads(json.dumps(start))
    event_list["event"] = []
    cases.append(event_list)
    desired_object = json.loads(json.dumps(start))
    desired_object["data"]["desired_state"] = {}
    cases.append(desired_object)
    operation_list = json.loads(json.dumps(accepted))
    operation_list["data"]["command"]["operation"] = []
    cases.append(operation_list)
    payload_number = json.loads(json.dumps(accepted))
    payload_number["data"]["command"]["payload"] = 5
    cases.append(payload_number)
    for payload in cases:
        violations = journal_module._validate_native_payload(
            journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, payload)
        assert violations
    assert journal_module._validate_native_payload(
        journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, []) == [
            "payload: required mapping"]


def test_native_validator_rejects_structural_but_semantically_invalid_transition(
        tmp_path):
    resolved = _resolved()
    with control.CampaignController(resolved, tmp_path / "service") as controller:
        controller.apply_command(_command(resolved, "pause-state", "pause", 0))
        assert controller._journal is not None
        payload = json.loads(json.dumps(controller._journal.read_all()[-1].payload))
    payload["data"]["observed_state"] = "running"
    payload["data"]["result"]["observed_state"] = "running"
    violations = journal_module._validate_native_payload(
        journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, payload)
    assert "data.observed_state: does not match operation" in violations


def test_native_resume_validator_returns_violations_for_unhashable_states_and_payload(
        tmp_path):
    resolved = _resolved()
    with control.CampaignController(resolved, tmp_path / "service") as controller:
        controller.apply_command(_command(resolved, "resume-state", "resume", 0))
        assert controller._journal is not None
        accepted = json.loads(json.dumps(controller._journal.read_all()[-1].payload))
    for malformed in ([], {}):
        payload = json.loads(json.dumps(accepted))
        payload["data"]["observed_state"] = malformed
        payload["data"]["result"]["observed_state"] = malformed
        violations = journal_module._validate_native_payload(
            journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, payload)
        assert "data.observed_state: does not match operation" in violations
        assert "data.observed_state: invalid" in violations
    payload = json.loads(json.dumps(accepted))
    payload["data"]["command"]["payload"] = {"invalid": float("nan")}
    violations = journal_module._validate_native_payload(
        journal_module.KIND_CAMPAIGN_SUPERVISOR_EVENT, payload)
    assert "data.command.payload: must be empty mapping" in violations


def test_named_lock_replacement_fences_old_and_new_controller(tmp_path):
    resolved = _resolved()
    store = tmp_path / "service"
    old = control.CampaignController(resolved, store)
    old.__enter__()
    lock = store / ".supervisor.lock"
    lock.unlink()
    lock.write_text("", encoding="utf-8")
    lock.chmod(0o600)
    try:
        with pytest.raises(control.ControlRefused, match="supervisor lock identity changed"):
            old.snapshot()
        with pytest.raises(Exception, match="lock identity changed"):
            control.CampaignController(resolved, store).__enter__()
    finally:
        old.close()


def test_symlinked_journal_root_and_critical_leaf_refuse(tmp_path):
    resolved = _resolved()
    store = tmp_path / "symlink-root"
    store.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store / "journal").symlink_to(outside, target_is_directory=True)
    with pytest.raises(control.ControlRefused, match="journal root"):
        control.CampaignController(resolved, store).__enter__()

    live_store = tmp_path / "live"
    controller = control.CampaignController(resolved, live_store)
    controller.__enter__()
    write_lock = live_store / "journal" / journal_module.LOCK_NAME
    write_lock.unlink()
    write_lock.symlink_to(outside / "fake-lock")
    try:
        with pytest.raises(control.ControlRefused, match="critical file"):
            controller.snapshot()
    finally:
        controller.close()

    event_store = tmp_path / "event-link"
    with control.CampaignController(resolved, event_store):
        pass
    event_file = event_store / "journal" / "events.jsonl"
    event_file.unlink()
    event_file.symlink_to(outside / "fake-events")
    with pytest.raises(control.ControlRefused, match="critical file"):
        control.CampaignController(resolved, event_store).__enter__()


def test_live_controller_refuses_replaced_journal_directory(tmp_path):
    store = tmp_path / "service"
    controller = control.CampaignController(_resolved(), store)
    controller.__enter__()
    journal_root = store / "journal"
    journal_root.rename(store / "displaced-journal")
    journal_root.mkdir(mode=0o700)
    try:
        with pytest.raises(control.ControlRefused, match="journal root identity changed"):
            controller.snapshot()
    finally:
        controller.close()
