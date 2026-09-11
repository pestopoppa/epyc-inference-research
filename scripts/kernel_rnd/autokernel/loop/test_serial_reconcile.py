"""Recover original completed tiny children, never relaunch a possibly live owner."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from . import serial_run as sr, worker_lifecycle
from .test_serial_run import _inputs


class RouterCrash(BaseException):
    pass


def _full_result(continuation, status="bench_failed", reason="setup failed"):
    attempt = {"status": status, "turn_recorded_at": "2026-09-10T00:00:00Z"}
    if reason is not None:
        attempt["reason"] = reason
    return {
        "schema": "epyc.autokernel.loop_run.v1",
        "epoch": "e" * 64, "anchor_commit": "a" * 40,
        "surface": "serving:fixture", "pairs": 5, "noise_floor_pct": 1.0,
        "elapsed_s": 1.0, "workers": 1, "iterations": [attempt],
        "phase_seconds": {"setup": 1.0}, "phase_seconds_are_lane_seconds": True,
        "pool": {"workers": 1, "wall_seconds": 1.0, "tail_seconds": 1.0,
                 "tail_fraction": 1.0, "superseded": 0},
        "continuation": continuation, "target": continuation["selected_target"],
        "runtime_preparation": {"status": "fixture"},
        "launch_snapshot": "f" * 64, "floor_request_digest": "d" * 64,
    }


def _completed_before_router_save(tmp_path, monkeypatch, *, mode="good"):
    root, argv = _inputs(tmp_path, monkeypatch, mode=mode, rounds=1)
    original = sr.status.write_json
    def crash(directory, name, body, **kwargs):
        if name == "serial-state.json" and body.get("next_batch") == 1 and body.get("active") is None:
            raise RouterCrash("after actual child terminal, before router commits progress")
        return original(directory, name, body, **kwargs)
    with monkeypatch.context() as local:
        local.setattr(sr.status, "write_json", crash)
        with pytest.raises(RouterCrash):
            sr.main(argv)
    state_path = root / "serial-state.json"
    state = json.loads(state_path.read_text())
    assert state["next_batch"] == 0 and state["active"]["pid"] is not None
    assert state["active"]["process_identity"]["pid"] == state["active"]["pid"]
    with pytest.raises(FileNotFoundError):
        os.stat(f"/proc/{state['active']['pid']}")
    result = root / "batches/batch-000000/loop-continuation.json"
    assert sr.load_completed(result)[0]["terminal"] in {"complete", "stopped"}
    return root, argv, state_path, state


@pytest.mark.parametrize("legacy", [False, True])
def test_original_completed_child_reconciles_once_without_replay(tmp_path, monkeypatch, legacy):
    root, argv, state_path, state = _completed_before_router_save(tmp_path, monkeypatch)
    original_bytes = (root / "batches/batch-000000/loop-continuation.json").read_bytes()
    if legacy:
        state["active"].pop("process_identity")
        state_path.write_text(json.dumps(state))
    assert sr.main(argv) == 0
    final = json.loads(state_path.read_text())
    assert final["next_batch"] == 2 and final["active"] is None
    assert final["last_reconciliation"]["target_index"] == 0
    assert final["last_reconciliation"]["process_terminal_basis"] == "original_pid_absent; exit_status_unavailable"
    seen = [json.loads(row) for row in (root / "seen.jsonl").read_text().splitlines()]
    assert len(seen) == 2 and len({sr.option(row["argv"], "--out") for row in seen}) == 2
    assert (root / "batches/batch-000000/loop-continuation.json").read_bytes() == original_bytes
    assert sr.main(argv) == 0
    assert len((root / "seen.jsonl").read_text().splitlines()) == 2


def test_original_stopped_completion_keeps_stop_across_reconciliation(tmp_path, monkeypatch):
    root, argv, state_path, _ = _completed_before_router_save(tmp_path, monkeypatch, mode="stop")
    assert sr.main(argv) == 0
    assert (root / "STOP").exists()
    assert json.loads(state_path.read_text())["last_reconciliation"]["terminal"] == "stopped"
    assert len((root / "seen.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_actually_live_child_is_not_adopted_signaled_or_relaunched(tmp_path, monkeypatch, legacy):
    root, argv, state_path, state = _completed_before_router_save(tmp_path, monkeypatch)
    # A live owned fixture child: deliberately substitute its identity into the
    # retained active row. Even a valid completed artifact cannot bypass liveness.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        state["active"]["pid"] = child.pid
        if legacy:
            state["active"].pop("process_identity")
        else:
            state["active"]["process_identity"] = worker_lifecycle.process_identity(child.pid).to_dict()
        before = json.dumps(state)
        state_path.write_text(before)
        with pytest.raises(sr.SerialRefused, match="still (live|present)"):
            sr.main(argv)
        assert child.poll() is None
        assert state_path.read_text() == before
        assert len((root / "seen.jsonl").read_text().splitlines()) == 1
    finally:
        child.terminate()
        child.wait(timeout=5)
        assert child.poll() is not None


@pytest.mark.parametrize("field", ["batch_dir", "input_argv_sha256", "selected_id", "store", "pid", "unknown"])
def test_changed_original_active_identity_cannot_reconcile(tmp_path, monkeypatch, field):
    root, argv, state_path, state = _completed_before_router_save(tmp_path, monkeypatch)
    state["active"][field] = None if field == "pid" else "changed"
    before = json.dumps(state)
    state_path.write_text(before)
    with pytest.raises(sr.SerialRefused):
        sr.main(argv)
    assert state_path.read_text() == before
    assert len((root / "seen.jsonl").read_text().splitlines()) == 1


def test_missing_terminal_artifact_retains_original_active_and_logs(tmp_path, monkeypatch):
    root, argv, state_path, _ = _completed_before_router_save(tmp_path, monkeypatch)
    before = state_path.read_bytes()
    result = root / "batches/batch-000000/loop-continuation.json"
    continuation = json.loads(result.read_text())
    malformed = _full_result(continuation, status="measured_null", reason=None)
    malformed.pop("epoch")  # A plausible but incomplete result is not recoverable.
    (result.parent / "loop-run.json").write_text(json.dumps(malformed))
    result.rename(result.with_suffix(".retained-for-test"))
    with pytest.raises(sr.SerialRefused, match="durable full result is malformed"):
        sr.main(argv)
    assert state_path.read_bytes() == before
    assert (root / "batches/batch-000000/stdout.log").is_file()
    assert (root / "batches/batch-000000/stderr.log").is_file()


def test_restart_recovers_only_matching_embedded_failure_observation(tmp_path, monkeypatch):
    root, argv, state_path, _state = _completed_before_router_save(tmp_path, monkeypatch)
    continuation_path = root / "batches/batch-000000/loop-continuation.json"
    continuation = json.loads(continuation_path.read_text())
    continuation["outcome_counts"] = {"bench_failed": 1}
    full_path = continuation_path.parent / "loop-run.json"
    full_path.write_text(json.dumps(_full_result(continuation)))
    continuation_path.unlink()

    assert sr.main(argv) == 0
    recovered, _sha = sr.load_completed(continuation_path)
    assert recovered["outcome_counts"] == {"bench_failed": 1}
    assert recovered["recovered_result_sha256"] == sr._json(full_path)[1]
    saved = json.loads(state_path.read_text())
    assert saved["last_reconciliation"]["result"]["path"] == str(continuation_path)
    assert json.loads(full_path.read_text())["iterations"][0]["reason"] == "setup failed"
    changed = json.loads(full_path.read_text())
    changed["elapsed_s"] = 2.0
    full_path.write_text(json.dumps(changed))
    with pytest.raises(sr.SerialRefused, match="full-result digest differs"):
        sr.load_completed(continuation_path)


def test_restart_refuses_forged_embedded_receipt_without_losing_it_or_advancing(
        tmp_path, monkeypatch):
    root, argv, state_path, _state = _completed_before_router_save(tmp_path, monkeypatch)
    before = state_path.read_bytes()
    continuation_path = root / "batches/batch-000000/loop-continuation.json"
    continuation = json.loads(continuation_path.read_text())
    continuation["input_argv"].append("--forged")
    continuation["input_argv_sha256"] = sr._digest(continuation["input_argv"])
    full_path = continuation_path.parent / "loop-run.json"
    full_path.write_text(json.dumps(
        _full_result(continuation, status="measured_null", reason=None)))
    original_full = full_path.read_bytes()
    continuation_path.unlink()

    with pytest.raises(sr.SerialRefused, match="different actual child arguments"):
        sr.main(argv)
    assert state_path.read_bytes() == before
    assert full_path.read_bytes() == original_full
    assert not continuation_path.exists()
