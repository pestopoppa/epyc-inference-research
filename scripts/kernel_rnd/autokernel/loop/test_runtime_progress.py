"""Existing runtime/startup publishers; no hardware or new measurement grade."""
import json
from unittest import mock

from . import runtime_admission as admission, runtime_calibration as calibration, status
from .test_existing_cpu_run import test_existing_main_cpu_five_iterations_preserves_canonical_champion as cpu_run
from .test_runtime_calibration import test_actual_calibration_original_prefix_reopens_without_relaunch as original_calibration
from .test_runtime_calibration import original_claim
from .test_runtime_admission import installed_fixture


def test_actual_launch_checkpoints_publish_counts_and_callback_fault_is_nonfatal(tmp_path, monkeypatch):
    rows = []
    original_init = calibration.DirectCalibration.__init__

    def report(row):
        rows.append(json.loads(json.dumps(row)))
        if row["completed_launches"] == 1:
            raise OSError("synthetic dashboard publication failure after durable launch")

    def observed_init(self, **kwargs):
        original_init(self, **kwargs, on_progress=report)

    monkeypatch.setattr(calibration.DirectCalibration, "__init__", observed_init)
    # Actual tiny HTTP launches, original prefix interruption/reopen, retained
    # solve and source refusal tests. Host-health observations are synthetic.
    original_calibration(tmp_path, monkeypatch)
    assert rows[0]["completed_launches"] == 0
    assert rows[-1]["completed_launches"] == 8
    assert all(row["launch_limit"] == 8 and not row["limit_is_upper_bound"] for row in rows)
    assert all(row["phase"] == "calibration" for row in rows)
    assert any(row["pending"] and row["completed_launches"] == 0 for row in rows)
    assert any(row["completed_launches"] == 2 and row["pending"] is None for row in rows)
    assert all(len(row["frame_digest"]) == 64 and row["observed_at"] for row in rows)


def test_actual_main_publishes_selected_recipe_without_heartbeat_redating(monkeypatch):
    original_init, original_write = admission.RuntimeAdmission.__init__, status.write
    rows = []
    stamp = "2026-09-10T00:00:00Z"

    def observed_init(self, **kwargs):
        original_init(self, **kwargs)
        self._progress({"observed_at": stamp}, operation="fixture_preparation")

    def recorded_write(*args, **kwargs):
        row = kwargs.get("runtime_preparation")
        if row is not None:
            rows.append(json.loads(json.dumps(row)))
        return original_write(*args, **kwargs)

    monkeypatch.setattr(admission.RuntimeAdmission, "__init__", observed_init)
    monkeypatch.setattr(status, "write", recorded_write)
    with mock.patch.object(admission.RuntimeAdmission, "calibration",
                           side_effect=AssertionError("source-only path must not calibrate runtime")):
        cpu_run(False)
    assert rows
    observed = [row for row in rows if row["progress"] is not None]
    assert len(observed) > 2
    assert {row["progress"]["observed_at"] for row in observed} == {stamp}
    assert all("threads=" in row["selected_recipe"] and "cpu=0-95" in row["selected_recipe"]
               for row in observed)
    assert all(row["calibration_launches"] in (None, 800) for row in observed)
    assert all(row["selected_recipe_reference"] is None for row in rows)


def test_control_and_candidate_windows_report_without_changing_admission(tmp_path, monkeypatch):
    pair, store, _counter, make_owner = installed_fixture(tmp_path, monkeypatch)
    updates = []
    try:
        with original_claim(tmp_path / "private.lock") as holder:
            owner = make_owner(holder)
            def report(row):
                updates.append(row)
                # Deliberately fail every callback, including after completed
                # launches; execution/admission must not depend on the dashboard.
                raise OSError("synthetic disconnected dashboard")
            owner._on_progress = report
            result = owner.compare(pair)
            assert result["decisive"] is True
            assert owner.retain(result, pair.anchor) == pair.candidate
        operations = {row["operation"] for row in updates}
        assert {"historical_control", "candidate_selection", "candidate_confirmation", "admitted"} <= operations
        windows = [row for row in updates if row.get("phase") == "pair_window"]
        assert windows and any(row["pending"] is not None for row in windows)
        assert any(row["completed_launches"] > 0 and row["pending"] is None for row in windows)
        assert all(row["completed_launches"] <= row["launch_limit"] for row in windows)
    finally:
        store.close()
