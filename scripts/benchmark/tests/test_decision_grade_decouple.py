"""NUMA-DECOUPLE-1 (INF-45): trimmed-window emptiness no longer gates a cell's
decision_grade, but is recorded as `trimmed_window_empty` + `caveats`, and a
model-level verdict is demoted to decision_grade_caveated by it."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server_numa_np_sweep as sns  # noqa: E402
from test_server_numa_np_sweep import _fake_cell_cache_reset, load_cell  # noqa: E402


def _summ(tmp_path):
    cell = load_cell(tmp_path)
    records = [
        sns.StreamRequestRecord(
            cell_id=cell.cell_id, qid=f"q{i}", suite="s", request_index=i,
            stream_id=i, instance_port=19080, success=True, start_s=0.0,
            first_token_s=0.1, end_s=1.0, ttft_ms=100.0, latency_ms=1000.0,
            predicted_tokens=1, prompt_tokens=1, predicted_tps=1.0,
        )
        for i in range(4)
    ]
    return sns.summarize_cell(
        cell=cell, records=records, wall_s=2.0, env={"GGML_IQK": "1"},
        instance_pids={19080: 1}, affinity={"live_affinity_verified": True},
        run_overrides_active=False, host_warnings=[], throttle_check={"warnings": []},
        cache_reset=_fake_cell_cache_reset(cell),
    )


def test_empty_trimmed_window_decouples_but_is_recorded(tmp_path):
    row = _summ(tmp_path)
    assert row["decision_grade"] is True
    assert row["decision_grade_blockers"] == []
    assert row["trimmed_window_empty"] is True
    assert row["trimmed_window_ready"] is False
    assert row["tasks_per_hour_trimmed"] == 0.0
    assert len(row["caveats"]) == 1 and row["caveats"][0].startswith("empty_trimmed_window")


def test_apply_basis_caveat_demotes_on_empty_trimmed_member():
    entry = {"status": "decision_grade", "trimmed_window_empty_cells": ["c1"]}
    out = sns._apply_basis_caveat(entry)
    assert out["status"] == "decision_grade_caveated"
    assert "empty trimmed window" in out["basis_note"] and "c1" in out["basis_note"]


def test_apply_basis_caveat_untouched_without_empty_members():
    entry = {"status": "decision_grade", "trimmed_window_empty_cells": []}
    assert sns._apply_basis_caveat(entry)["status"] == "decision_grade"
