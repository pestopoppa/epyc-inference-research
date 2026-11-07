"""keep_timing: per-stage keep-path observability (payload + read-only report)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from . import keep_timing as kt, run


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 1.5
        return self.t


def _timer():
    return kt.KeepTimer(head="h" * 40, clock=_Clock(), wall=lambda: "2026-11-07T00:00:00Z")


def _keep_payload():
    t = _timer()
    with t.stage("anchor_clean_build", commit="c" * 40) as rec:
        rec["build_dir"] = "/store/anchor-gen-3"
    with t.stage("anchor_guard"):
        with t.stage("anchor_guard_build", commit="c" * 40, build_dir="/scratch"):
            pass
        with t.stage("anchor_guard_digest", build_dir="/scratch"):
            pass
        t.skip("anchor_guard_aa", "previous A/A passed 12 min ago")
    for name in ("headline", "reprofile", "prune_anchor_generations"):
        with t.stage(name):
            pass
    return t.to_dict()


def test_payload_has_every_stage_with_end_after_start():
    p = _keep_payload()
    assert p["schema"] == kt.SCHEMA
    names = [s["stage"] for s in p["stages"]]
    for want in ("anchor_clean_build", "anchor_guard_build", "anchor_guard_digest",
                 "anchor_guard_aa", "anchor_guard", "headline", "reprofile"):
        assert want in names
    for s in p["stages"]:
        assert s["end_monotonic"] >= s["start_monotonic"]
        assert s["started_at"] and s["ended_at"]
        assert s["duration_s"] >= 0


def test_skipped_aa_carries_reason():
    aa = [s for s in _keep_payload()["stages"] if s["stage"] == "anchor_guard_aa"]
    assert aa and aa[0]["skipped"] is True and "12 min" in aa[0]["reason"]


def test_raising_stage_is_recorded_and_reraised():
    t = _timer()
    with pytest.raises(ValueError):
        with t.stage("headline"):
            raise ValueError("boom")
    assert t.stages[0]["error"].startswith("ValueError")


def test_duplicate_build_flag():
    p = _keep_payload()
    assert p["duplicate_builds"] and p["duplicate_builds"][0]["builds"] == 2
    t = _timer()
    with t.stage("anchor_clean_build", commit="a" * 40):
        pass
    with t.stage("anchor_guard_build", commit="b" * 40):
        pass
    assert t.to_dict()["duplicate_builds"] == []


def _store(tmp_path: Path) -> Path:
    db = sqlite3.connect(tmp_path / "experiments.db")
    db.execute("CREATE TABLE experiments (attempt_id TEXT, recorded_at TEXT, "
               "status TEXT, payload TEXT)")
    rows = [("1", "a", "kept", json.dumps({"status": "kept", kt.FIELD: _keep_payload()})),
            ("2", "b", "kept", json.dumps({"status": "kept"})),
            ("3", "c", "anchor_verified", json.dumps({kt.FIELD: _keep_payload()}))]
    db.executemany("INSERT INTO experiments VALUES (?,?,?,?)", rows)
    db.commit()
    db.close()
    return tmp_path


def test_report_on_fixture_store(tmp_path, capsys):
    report = kt.summarize(kt.read_store(_store(tmp_path)))
    assert report["keeps_timed"] == 1 and report["keeps_untimed"] == 1
    assert report["stages"]["anchor_clean_build"]["n"] == 1
    assert report["stages"]["anchor_guard_aa"]["skipped"] == 1
    assert len(report["duplicate_builds"]) == 1
    assert kt.main([str(tmp_path)]) == 0
    assert "DUPLICATE BUILD" in capsys.readouterr().out


def test_run_wires_every_keep_stage():
    src = Path(run.__file__).read_text()
    for name in ("anchor_clean_build", "anchor_guard_build", "anchor_guard_heal_build",
                 "anchor_guard_digest", "anchor_guard_aa", "headline", "reprofile",
                 "prune_anchor_generations", "accumulate_after_keep", "codegen_summary"):
        assert f'"{name}"' in src
    assert "attempt[keep_timing.FIELD]" in src
