"""Real authenticated HTTP → original serial owner → captured tiny children."""
import copy
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import pytest

from . import serial_control as sc, serial_run as sr
from .test_serial_run import _inputs, _scheduled


def _until(read, predicate=lambda value: bool(value)):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            value = read()
            if predicate(value):
                return value
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(.01)
    raise AssertionError("original serial condition did not complete")


def _http(endpoint, request=None, *, token="fixture-token", origin="http://localhost:8100"):
    req = urllib.request.Request(endpoint + ("/snapshot" if request is None else "/commands"),
        data=None if request is None else json.dumps(request).encode(),
        headers={"Authorization": "Bearer " + token, "Origin": origin,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as response:
        return json.load(response)


def _command(snapshot, operation, request_id):
    return {"schema": sc.COMMAND_SCHEMA, "config_digest": snapshot["config_digest"],
            "owner_id": snapshot["owner_id"], "request_id": request_id, "operation": operation,
            "expected_revision": snapshot["revision"], "expected_batch": snapshot["batch_number"],
            "expected_target": snapshot["active_target"]}


def test_authenticated_pause_finishes_batch_resume_drain_settles_original_child(tmp_path, monkeypatch):
    root, argv = _inputs(tmp_path, monkeypatch, mode="wait", rounds=3)
    argv = _scheduled(tmp_path, argv)
    child = tmp_path / "child.py"
    child.write_text(child.read_text().replace(
        "while not stopped[0] and time.monotonic() < until:",
        "while not stopped[0] and not (out / 'release').exists() and time.monotonic() < until:"))
    monkeypatch.setenv("AUTOKERNEL_CONTROL_TOKEN", "fixture-token")
    results, errors = {}, []

    def operator():
        try:
            first = root / "batches/batch-000000"
            _until(lambda: (first / "ready").exists())
            report = _until(lambda: json.loads((root / "loop-status.json").read_text()),
                            lambda row: row.get("serial_control", {}).get("active_target") == "cpu")
            endpoint = report["serial_control"]["endpoint"]
            for token, origin, code in (("wrong", "http://localhost:8100", 401),
                                         ("fixture-token", "http://wrong", 403)):
                with pytest.raises(urllib.error.HTTPError) as error:
                    _http(endpoint, token=token, origin=origin)
                assert error.value.code == code
            initial = _http(endpoint)
            request = _command(initial, "pause", "pause-1")
            ack = _http(endpoint, request)
            assert not ack["completed"] and ack["outcome"] == "pending"
            assert _http(endpoint, request) == ack  # Lost ACK retry, no second revision.
            assert not (root / "STOP").exists()
            assert not (first / "loop-continuation.json").exists()
            (first / "release").touch()
            paused = _until(lambda: _http(endpoint), lambda row: row["observed_state"] == "paused")
            assert paused["commands"][-1]["result"]["completed"]
            assert not (root / "batches/batch-000001").exists()
            state = json.loads((root / "serial-state.json").read_text())
            assert state["next_batch"] == 1 and state["active"] is None
            assert state["last_results"]["0"]  # Original result/accounting before pause.
            assert state["scheduler_state"]["campaign_attempts"] == 1
            assert state["scheduler_state"]["campaign_charged_seconds"] == 3.0
            with pytest.raises(urllib.error.HTTPError) as error:
                _http(endpoint, _command(initial, "drain", "stale"))
            assert error.value.code == 400
            resume = _http(endpoint, _command(paused, "resume", "resume-2"))
            assert resume["completed"]
            second = root / "batches/batch-000001"
            _until(lambda: (second / "ready").exists())
            running = _until(lambda: _http(endpoint),
                             lambda row: row["batch_number"] == 1 and row["active_target"] is not None)
            results["drain"] = _http(endpoint, _command(running, "drain", "drain-3"))
            assert not results["drain"]["completed"]
        except BaseException as exc:
            errors.append(exc)
            (root / "STOP").touch()

    thread = threading.Thread(target=operator)
    thread.start()
    try:
        assert sr.main([*argv, "--control-listen", "127.0.0.1:0",
                        "--control-origin", "http://localhost:8100"]) == 0
    finally:
        thread.join(20)
    assert not thread.is_alive() and not errors, errors
    state = json.loads((root / "serial-state.json").read_text())
    terminal = json.loads((root / "loop-status.json").read_text())["serial_control"]
    assert state["next_batch"] == 2 and state["active"] is None
    assert state["control"]["revision"] == 3
    assert terminal["observed_state"] == "drained"
    assert terminal["commands"][-1]["result"]["outcome"] == "completed"
    assert terminal["commands"][-1]["result"]["completed"]
    assert len((root / "seen.jsonl").read_text().splitlines()) == 2
    second = json.loads((root / "batches/batch-000001/loop-continuation.json").read_text())
    assert second["terminal"] == "stopped"
    assert state["scheduler_state"]["campaign_attempts"] == 2
    assert state["scheduler_state"]["campaign_charged_seconds"] == 6.0
    assert len(state["scheduler_state"]["accounted_receipts"]) == 2
    with pytest.raises(urllib.error.URLError):
        _http(terminal["endpoint"])


def _queued(owner, operation="pause", request_id="p"):
    request = _command(owner.publish_snapshot(), operation, request_id)
    item = {"request": request, "done": threading.Event()}
    owner._queue.put_nowait(item)
    return item


def test_terminal_failure_never_completes_drain_and_last_batch_pause_superseded():
    for operation, failed, expected, completed in (
            ("drain", True, "failed", False), ("pause", False, "superseded", True)):
        state, saved, stops = {"next_batch": 0}, [], []
        owner = sc.SerialControl(state, lambda: saved.append(copy.deepcopy(state)),
                                 lambda *_: stops.append(True), config_digest="c" * 64)
        owner.pump({"selected_id": "cpu"})
        item = _queued(owner, operation)
        owner.pump({"selected_id": "cpu"})
        assert item["done"].is_set() and saved
        final = owner.pump(terminal=True, stopped=operation == "drain", failed=failed)
        result = final["commands"][-1]["result"]
        assert result["outcome"] == expected and result["completed"] is completed
        assert final["observed_state"] == ("failed" if failed else "complete")


def test_failed_save_cannot_ack_or_signal_and_moved_target_refuses():
    state = {"next_batch": 0}
    stops = []
    def failed_save():
        raise OSError("fixture write failure")
    owner = sc.SerialControl(state, failed_save, lambda *_: stops.append(True),
                             config_digest="c" * 64)
    owner.pump({"selected_id": "cpu"})
    item = _queued(owner, "drain")
    with pytest.raises(OSError, match="fixture"):
        owner.pump({"selected_id": "cpu"})
    assert not item["done"].is_set() and not stops
    other = sc.SerialControl({"next_batch": 0}, lambda: None, lambda *_: None,
                             config_digest="c" * 64)
    other.pump({"selected_id": "cpu"})
    moved = _queued(other)
    other.pump({"selected_id": "gpu"})
    assert "batch/target changed" in moved["error"]
    assert other.state["control"]["revision"] == 0


def test_retained_pause_without_listener_is_actionable_not_auto_resume(tmp_path, monkeypatch):
    root, argv = _inputs(tmp_path, monkeypatch, rounds=1)
    targets = [sr._target_args(Path(argv[i + 1])) for i, value in enumerate(argv)
               if value == "--target-args"]
    config = sr._digest({"targets": targets, "batch_iterations": 1, "rounds": 1})
    (root / "serial-state.json").write_text(json.dumps({
        "schema": sr.SERIAL_SCHEMA, "config_digest": config, "next_batch": 0,
        "active": None, "last_results": {}, "source_results": {}, "failed_targets": {},
        "control": {"revision": 1, "desired_state": "paused", "commands": []}}))
    with pytest.raises(sr.SerialRefused, match="retained serial pause requires --control-listen"):
        sr.main(argv)
    assert not (root / "seen.jsonl").exists()
    assert json.loads((root / "serial-state.json").read_text())["control"]["desired_state"] == "paused"


def test_loopback_token_and_dry_run_preserve_existing_boundary(tmp_path, monkeypatch):
    root, argv = _inputs(tmp_path, monkeypatch)
    monkeypatch.setenv("AUTOKERNEL_CONTROL_TOKEN", "fixture-token")
    with pytest.raises(SystemExit):
        sr.main([*argv, "--control-listen", "0.0.0.0:8911"])
    monkeypatch.setattr(sr.run if hasattr(sr, "run") else __import__(
        sr.__package__ + ".run", fromlist=["main"]), "main", lambda _argv: 0)
    assert sr.main([*argv, "--control-listen", "127.0.0.1:0", "--dry-run"]) == 0
    assert not (root / "serial-state.json").exists()
