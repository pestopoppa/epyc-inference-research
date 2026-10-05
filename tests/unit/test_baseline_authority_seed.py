"""Tests for baseline authority seed event preparation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import baseline_authority_seed as seed_mod  # noqa: E402


def _state() -> dict[str, Any]:
    return {
        "trial_counter": 12,
        "baseline_state": {
            "baselines_by_tier": {
                "1": 1.8,
                "2": 1.6,
            },
            "cost": 0.5,
        },
    }


def _trial(trial_id: int = 11) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "quality": 1.7,
    }


def test_build_seed_event_makes_empty_ledger_cutover_ready() -> None:
    result = seed_mod.build_baseline_seed_event(_state(), [_trial()])

    assert result.status == "ready"
    assert result.before == {
        "status": "no_events",
        "event_count": 0,
        "valid_snapshot_count": 0,
        "cutover_ready": False,
        "cutover_blockers": [
            "no baseline promotion events; YAML remains cold-start seed"
        ],
        "warnings": [],
    }
    assert result.after is not None
    assert result.after["status"] == "match"
    assert result.after["cutover_ready"] is True
    assert result.event is not None
    assert result.event["type"] == "baseline_promotion"
    assert result.event["policy_version"] == "baseline-state-seed-v1"
    assert result.event["source_trial_id"] == 11
    assert result.event["tier"] == 2
    assert result.event["new_quality"] == 1.6
    assert result.event["baseline_state"] == _state()["baseline_state"]
    assert result.event["proof"]["seeded_from_state_baseline"] is True


def test_build_seed_event_uses_requested_tier() -> None:
    result = seed_mod.build_baseline_seed_event(_state(), [_trial()], tier=1)

    assert result.status == "ready"
    assert result.event is not None
    assert result.event["tier"] == 1
    assert result.event["new_quality"] == 1.8


def test_build_seed_event_blocks_existing_uncutover_ledger() -> None:
    existing = {
        "type": "baseline_promotion",
        "source_trial_id": 7,
        "tier": 1,
        "previous_quality": None,
        "new_quality": 1.9,
        "baseline_state": {"baselines_by_tier": {"1": 1.9}},
    }

    result = seed_mod.build_baseline_seed_event(_state(), [existing])

    assert result.status == "existing_ledger_blocked"
    assert result.warning == "existing baseline promotion ledger is not cutover-ready"
    assert result.event is None


def test_build_seed_event_reports_already_aligned() -> None:
    baseline = _state()["baseline_state"]
    existing = {
        "type": "baseline_promotion",
        "source_trial_id": 7,
        "tier": 2,
        "previous_quality": None,
        "new_quality": 1.6,
        "baseline_state": baseline,
    }

    result = seed_mod.build_baseline_seed_event(_state(), [existing])

    assert result.status == "already_aligned"
    assert result.event is None


def test_append_refuses_when_autopilot_running(monkeypatch, tmp_path: Path) -> None:
    result = seed_mod.build_baseline_seed_event(_state(), [_trial()])
    journal_path = tmp_path / "autopilot_journal.jsonl"
    journal_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(seed_mod, "_autopilot_running_pids", lambda: [123, 456])

    written = seed_mod.append_baseline_seed_event(journal_path, result)

    assert written.status == "live_autopilot_running"
    assert written.live_autopilot_pids == [123, 456]
    assert journal_path.read_text(encoding="utf-8") == ""


def test_append_writes_event_when_no_autopilot_running(monkeypatch, tmp_path: Path) -> None:
    result = seed_mod.build_baseline_seed_event(_state(), [_trial()])
    journal_path = tmp_path / "autopilot_journal.jsonl"
    journal_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(seed_mod, "_autopilot_running_pids", lambda: [])

    written = seed_mod.append_baseline_seed_event(journal_path, result)
    rows = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]

    assert written.status == "written"
    assert len(rows) == 1
    assert rows[0]["policy_version"] == "baseline-state-seed-v1"


def test_autopilot_process_scan_uses_script_position_and_exact_start_token(
    monkeypatch, tmp_path: Path
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()

    def process(pid: int, *argv: str) -> None:
        process_dir = proc_root / str(pid)
        process_dir.mkdir()
        (process_dir / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")

    process(
        101,
        "python",
        "-u",
        "-B",
        "/opt/epyc/scripts/autopilot/autopilot.py",
        "start",
        "--max-trials",
        "1",
    )
    process(102, "/opt/epyc/scripts/autopilot/autopilot.py", "start")
    process(103, "python", "/opt/epyc/autopilot.py", "startup")
    process(104, "python", "/opt/epyc/autopilot.py.bak", "start")
    process(105, "echo", "autopilot.py", "start")
    process(106, "python", "-c", "pass", "autopilot.py", "start")
    process(107, "python", "-c", "autopilot.py start")
    process(
        108,
        "python",
        "/opt/epyc/scripts/autopilot/baseline_authority_seed.py",
        "--append",
    )
    process(109, "python", "/opt/epyc/scripts/autopilot/autopilot.py", "start")
    monkeypatch.setattr(seed_mod.os, "getpid", lambda: 109)

    assert seed_mod._autopilot_running_pids(proc_root) == [101, 102]


def test_autopilot_process_scan_skips_raced_and_malformed_proc_entries(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    missing = proc_root / "202"
    missing.mkdir()
    malformed = proc_root / "203"
    malformed.mkdir()
    (malformed / "cmdline").write_bytes(b"python /opt/epyc/autopilot.py start")
    (proc_root / "204").mkdir()
    (proc_root / "204" / "cmdline").write_bytes(b"python\0-c\0pass\0autopilot.py\0start\0")
    (proc_root / "not-a-pid").mkdir()

    assert seed_mod._autopilot_running_pids(proc_root) == []


@pytest.mark.parametrize("failure_scope", ["process_table", "process_cmdline"])
def test_append_refuses_when_process_observation_is_unknown(
    failure_scope: str, monkeypatch, tmp_path: Path
) -> None:
    result = seed_mod.build_baseline_seed_event(_state(), [_trial()])
    journal_path = tmp_path / "autopilot_journal.jsonl"
    journal_path.write_text("", encoding="utf-8")
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    cmdline_path = proc_root / "123" / "cmdline"
    cmdline_path.parent.mkdir()
    cmdline_path.write_bytes(b"python\0/opt/epyc/autopilot.py\0start\0")

    if failure_scope == "process_table":
        original_iterdir = Path.iterdir

        def fail_proc_enumeration(path: Path):
            if path == proc_root:
                raise PermissionError("fixture process table unavailable")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fail_proc_enumeration)
    else:
        original_read_bytes = Path.read_bytes

        def fail_cmdline_read(path: Path) -> bytes:
            if path == cmdline_path:
                raise PermissionError("fixture cmdline unavailable")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", fail_cmdline_read)

    observer = seed_mod._autopilot_running_pids
    monkeypatch.setattr(
        seed_mod,
        "_autopilot_running_pids",
        lambda: observer(proc_root),
    )
    written = seed_mod.append_baseline_seed_event(journal_path, result)

    assert written.status == "process_observation_unknown"
    assert "refusing baseline seed append" in written.warning
    assert journal_path.read_text(encoding="utf-8") == ""


def test_cli_trial_counter_mismatch_returns_two(tmp_path: Path, capsys) -> None:
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    journal_path.write_text("", encoding="utf-8")

    rc = seed_mod.main(
        [
            "--state",
            str(state_path),
            "--journal",
            str(journal_path),
            "--append",
            "--expect-trial-counter",
            "99",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert out["status"] == "trial_counter_mismatch"


def test_cli_journal_max_trial_mismatch_returns_two(tmp_path: Path, capsys) -> None:
    state_path = tmp_path / "autopilot_state.json"
    journal_path = tmp_path / "autopilot_journal.jsonl"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    journal_path.write_text(json.dumps(_trial(11)) + "\n", encoding="utf-8")

    rc = seed_mod.main(
        [
            "--state",
            str(state_path),
            "--journal",
            str(journal_path),
            "--append",
            "--expect-journal-max-trial-id",
            "10",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert out["status"] == "journal_max_trial_id_mismatch"
    assert "expected journal max trial_id 10" in out["warning"]
