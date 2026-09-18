"""Focused command-v2 identity, replay, compatibility, and shutdown tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from .. import journal
from . import campaign_command_v2 as command_v2
from . import campaign_control as control
from .test_campaign_control import _resolved


def _command(controller, request_id="fenced", operation="pause"):
    row = {
        "schema": command_v2.COMMAND_SCHEMA,
        "campaign_id": controller.resolved.campaign_id,
        "config_generation": controller.config_generation,
        "config_digest": controller.config_digest,
        "supervisor_incarnation": controller.supervisor_incarnation,
        "request_id": request_id, "operation": operation, "payload": {},
        "expected_control_revision": controller.control_revision,
    }
    row["payload_digest"] = command_v2.command_digest(
        operation=operation, payload={}, campaign_id=row["campaign_id"],
        config_generation=row["config_generation"], config_digest=row["config_digest"],
        supervisor_incarnation=row["supervisor_incarnation"], request_id=request_id,
        expected_control_revision=row["expected_control_revision"])
    return row


def test_false_incarnation_and_config_refuse_before_any_mutation(tmp_path):
    with control.CampaignController(
            _resolved(), tmp_path / "store", snapshot_version=2) as controller:
        before = (controller.control_revision, controller.desired_state,
                  controller._journal_cursor)
        for field, value in (("supervisor_incarnation", 999),
                             ("config_digest", "f" * 64)):
            row = _command(controller, request_id=f"bad-{field}")
            row[field] = value
            row["payload_digest"] = command_v2.command_digest(
                operation=row["operation"], payload={}, campaign_id=row["campaign_id"],
                config_generation=row["config_generation"],
                config_digest=row["config_digest"],
                supervisor_incarnation=row["supervisor_incarnation"],
                request_id=row["request_id"],
                expected_control_revision=row["expected_control_revision"])
            with pytest.raises(control.ControlRefused, match="identity"):
                controller.apply_command(row)
            assert (controller.control_revision, controller.desired_state,
                    controller._journal_cursor) == before


def test_digest_binds_request_id_schema_and_expected_revision(tmp_path):
    with control.CampaignController(
            _resolved(), tmp_path / "store", snapshot_version=2) as controller:
        first = _command(controller, request_id="digest-a")
        changed_id = _command(controller, request_id="digest-b")
        changed_revision = dict(first, expected_control_revision=1)
        changed_revision["payload_digest"] = command_v2.command_digest(
            operation=first["operation"], payload={}, campaign_id=first["campaign_id"],
            config_generation=first["config_generation"],
            config_digest=first["config_digest"],
            supervisor_incarnation=first["supervisor_incarnation"],
            request_id=first["request_id"], expected_control_revision=1)
        assert first["payload_digest"] != changed_id["payload_digest"]
        assert first["payload_digest"] != changed_revision["payload_digest"]
        assert command_v2.validate_command(changed_revision) == changed_revision


def test_exact_lost_ack_retry_survives_restart_but_changed_request_refuses(tmp_path):
    store = tmp_path / "store"
    command = None
    with control.CampaignController(_resolved(), store, snapshot_version=2) as first:
        command = _command(first)
        result = first.apply_command(command)
        assert first.apply_command(command) == result
        assert first.control_revision == 1
    assert command is not None
    with control.CampaignController(_resolved(), store, snapshot_version=2) as restarted:
        assert restarted.supervisor_incarnation == 2
        assert restarted.apply_command(command) == result
        assert restarted.control_revision == 1
        changed = dict(command, expected_control_revision=9)
        changed["payload_digest"] = command_v2.command_digest(
            operation=changed["operation"], payload={},
            campaign_id=changed["campaign_id"],
            config_generation=changed["config_generation"],
            config_digest=changed["config_digest"],
            supervisor_incarnation=changed["supervisor_incarnation"],
            request_id=changed["request_id"], expected_control_revision=9)
        with pytest.raises(control.ControlRefused, match="different semantics"):
            restarted.apply_command(changed)


def test_failure_before_command_append_replays_no_invented_acceptance(tmp_path, monkeypatch):
    store = tmp_path / "store"
    controller = control.CampaignController(_resolved(), store, snapshot_version=2)
    controller.__enter__()
    command = _command(controller, request_id="before-append")
    assert controller._journal is not None
    append = controller._journal.append

    def fail_before_append(*_args, **_kwargs):
        raise OSError("fixture pre-append failure")

    monkeypatch.setattr(controller._journal, "append", fail_before_append)
    with pytest.raises(OSError, match="pre-append"):
        controller.apply_command(command)
    monkeypatch.setattr(controller._journal, "append", append)
    controller.close()
    with control.CampaignController(_resolved(), store, snapshot_version=2) as restarted:
        assert "before-append" not in restarted.command_results


def test_failure_after_durable_acceptance_replays_exact_lost_ack(tmp_path, monkeypatch):
    store = tmp_path / "store"
    controller = control.CampaignController(_resolved(), store, snapshot_version=2)
    controller.__enter__()
    command = _command(controller, request_id="after-append")
    publish = controller._publish_snapshot_locked

    def fail_after_append():
        raise OSError("fixture post-append failure")

    monkeypatch.setattr(controller, "_publish_snapshot_locked", fail_after_append)
    with pytest.raises(OSError, match="post-append"):
        controller.apply_command(command)
    monkeypatch.setattr(controller, "_publish_snapshot_locked", publish)
    accepted = controller.command_results["after-append"]
    controller.close()
    with control.CampaignController(_resolved(), store, snapshot_version=2) as restarted:
        assert restarted.apply_command(command) == accepted
        assert restarted.control_revision == 1


def test_repeated_signal_drain_request_is_one_durable_command(tmp_path):
    with control.CampaignController(
            _resolved(), tmp_path / "store", snapshot_version=2) as controller:
        first = controller.request_shutdown_drain()
        second = controller.request_shutdown_drain()
        assert first == second
        assert controller.control_revision == 1
        assert len(controller.command_results) == 1
        assert controller.await_shutdown_drain(time.monotonic() + 0.1) == first


@pytest.mark.parametrize("snapshot_version", [1, 2])
def test_shutdown_reuses_completed_operator_drain_across_repeat_and_restart(
        tmp_path, snapshot_version):
    store = tmp_path / "store"
    with control.CampaignController(
            _resolved(), store, snapshot_version=snapshot_version) as controller:
        if snapshot_version == 2:
            command = _command(controller, "operator-drain", "drain")
        else:
            command = {
                "schema": control.COMMAND_SCHEMA,
                "campaign_id": controller.resolved.campaign_id,
                "config_generation": controller.config_generation,
                "request_id": "operator-drain", "operation": "drain", "payload": {},
                "expected_control_revision": controller.control_revision,
            }
            command["payload_digest"] = control.command_digest(
                operation="drain", payload={}, campaign_id=command["campaign_id"],
                config_generation=command["config_generation"])
        prior = controller.apply_command(command)
        assert controller.request_shutdown_drain() == prior
        assert controller.request_shutdown_drain() == prior
        assert len(controller.command_results) == 1
    with control.CampaignController(
            _resolved(), store, snapshot_version=snapshot_version) as restarted:
        assert restarted.request_shutdown_drain() == prior
        assert restarted.await_shutdown_drain(time.monotonic() + 0.1) == prior
        assert len(restarted.command_results) == 1


def test_shutdown_reuses_pending_operator_drain_until_actual_settlement(tmp_path):
    with control.CampaignController(
            _resolved(), tmp_path / "store", snapshot_version=2) as controller:
        controller._worker_run_active = True
        prior = controller.apply_command(_command(controller, "operator-pending", "drain"))
        assert prior["completed"] is False
        assert controller.request_shutdown_drain() == prior
        assert len(controller.command_results) == 1
        with controller._shutdown_condition:
            controller._worker_run_active = False
            controller._settle_v2_commands_locked()
        completed = controller.await_shutdown_drain(time.monotonic() + 0.1)
        assert completed["request_id"] == "operator-pending"
        assert completed["completed"] is True


def test_shutdown_wait_is_condition_signaled_not_sleep_polled(tmp_path, monkeypatch):
    with control.CampaignController(
            _resolved(), tmp_path / "store", snapshot_version=2) as controller:
        controller._worker_run_active = True
        controller.request_shutdown_drain()
        monkeypatch.setattr(time, "sleep", lambda _seconds: pytest.fail("polled sleep"))
        outcome = []

        def waiter():
            outcome.append(controller.await_shutdown_drain(time.monotonic() + 1))

        thread = threading.Thread(target=waiter)
        thread.start()
        with controller._shutdown_condition:
            controller._worker_run_active = False
            controller._settle_v2_commands_locked()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert outcome[0]["observed_state"] == "drained"


def test_v1_behavior_remains_explicit_and_old_validator_refuses_v2(tmp_path):
    with control.CampaignController(
            _resolved(), tmp_path / "store", snapshot_version=2) as controller:
        fenced = _command(controller)
        with pytest.raises(control.ControlRefused):
            control.validate_command(fenced)
        legacy = {
            "schema": control.COMMAND_SCHEMA,
            "campaign_id": controller.resolved.campaign_id,
            "config_generation": 1, "request_id": "legacy", "operation": "pause",
            "payload": {}, "expected_control_revision": 0,
        }
        legacy["payload_digest"] = control.command_digest(
            operation="pause", payload={}, campaign_id=legacy["campaign_id"],
            config_generation=1)
        assert controller.apply_command(legacy)["completed"] is True


def test_closed_old_kind_vocabulary_refuses_new_journal_without_mutation(tmp_path):
    store = tmp_path / "journal"
    log = journal.Journal(str(store), campaign_id="campaign")
    log.initialize()
    payload = {"not": "used"}
    # Exercise the actual byte parser with the previous closed vocabulary.
    current = journal.KINDS
    envelope = {
        "journal_schema": journal.JOURNAL_ENTRY_SCHEMA, "event_id": "event",
        "seq": 1, "kind": journal.KIND_CAMPAIGN_COMMAND_V3,
        "campaign_id": "campaign", "record_id": None,
        "written_at": "2026-09-09T00:00:00Z", "payload": payload,
    }
    raw = (json.dumps(envelope, sort_keys=True) + "\n").encode()
    before = raw
    try:
        journal.KINDS = frozenset(current - {journal.KIND_CAMPAIGN_COMMAND_V3})
        _entry, defect = journal._parse_line(raw, 1, 1)
        assert defect is not None and "malformed envelope" in defect.reason
    finally:
        journal.KINDS = current
    assert raw == before


@pytest.mark.parametrize("snapshot_version", [1, 2])
def test_campaign_service_sigterm_durably_drains_and_exits_owned_child(
        tmp_path, snapshot_version):
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(_resolved().to_dict()), encoding="utf-8")
    store = tmp_path / "store"
    env = os.environ.copy()
    env["AUTOKERNEL_CONTROL_TOKEN"] = "fixture-token"
    process = subprocess.Popen([
        sys.executable, "-m", "scripts.kernel_rnd.autokernel.loop.campaign_service",
        "--resolved-campaign", str(resolved), "--store", str(store),
        "--snapshot-version", str(snapshot_version), "--listen", "127.0.0.1:0",
        "--shutdown-deadline", "1",
    ], cwd=Path(__file__).parents[4], env=env)
    try:
        deadline = time.monotonic() + 3
        snapshot = store / control.SNAPSHOT_FILE
        while not snapshot.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=3) == 0
        assert process.poll() is not None
        body = json.loads(snapshot.read_text(encoding="utf-8"))
        assert body["desired_state"] == body["observed_state"] == "drained"
        assert any(row["request_id"].startswith("service-sigterm-drain:")
                   for row in body["command_results"])
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
