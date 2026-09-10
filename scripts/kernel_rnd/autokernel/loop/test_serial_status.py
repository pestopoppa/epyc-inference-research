"""Original writers and tiny owned loop fixtures, not hardware measurements."""
import json
import os
from unittest import mock

import pytest

from . import serial_run, status, test_serial_run


def test_optional_fields_do_not_change_ordinary_status_shape(tmp_path):
    status.write(tmp_path, state="running", epoch="e", campaign_id="ak-loop",
                 anchor_commit="a", surface="tg128", pairs=5, noise_floor_pct=None)
    body = status.read(tmp_path)
    assert "batch" not in body and "routing" not in body
    status.write(tmp_path, state="running", epoch="e", campaign_id="ak-loop",
                 anchor_commit="a", surface="tg128", pairs=5, noise_floor_pct=None,
                 batch={"output_dir": str(tmp_path / "out"), "pid": os.getpid()})
    assert status.read(tmp_path)["batch"] == {
        "output_dir": str(tmp_path / "out"), "pid": os.getpid()}


@pytest.mark.parametrize("mode,rounds", [("good", 1), ("fail", 0), ("stop", 0)])
def test_actual_router_reports_configuration_cursor_and_terminal_facts(tmp_path, monkeypatch, mode, rounds):
    root, argv = test_serial_run._inputs(tmp_path, monkeypatch, mode=mode, rounds=rounds)
    assert serial_run.main(argv) == (1 if mode == "fail" else 0)
    body = status.read(root)
    state = json.loads((root / "serial-state.json").read_text())
    routing = body["routing"]
    assert routing["target_count"] == state["target_count"] == 2
    assert routing["batch_iterations"] == state["batch_iterations"] == 1
    assert routing["rounds"] == state["rounds"] == rounds
    assert routing["next_batch"] == state["next_batch"]
    assert routing["stop_requested"] is (mode == "stop")
    assert len(routing["failed_targets"]) == (2 if mode == "fail" else 0)
    assert body["iterations_done"] == 0  # Routing did not claim child iterations.
    assert "batch" not in body


def test_original_run_publisher_retains_actual_output_and_pid():
    with mock.patch.object(status, "write", wraps=status.write) as written:
        test_serial_run.test_existing_run_emits_actual_current_and_cor_then_resumes_without_anchor_rebuild()
    batches = [call.kwargs["batch"] for call in written.call_args_list if "batch" in call.kwargs]
    assert batches and len({row["output_dir"] for row in batches}) == 2
    assert all(row["pid"] == os.getpid() for row in batches)
    assert all(set(row) == {"output_dir", "pid"} for row in batches)
