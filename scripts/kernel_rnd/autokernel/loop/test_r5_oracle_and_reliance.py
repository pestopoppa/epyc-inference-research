"""r5 (2026-11-11) regressions: the oracle gap must name its missing part, and the planner
prompt must tell the planner to declare reliance on claims that shaped the proposal."""
from pathlib import Path

from . import actors, cpu_route_witness as crw


def test_missing_gdb_is_named(tmp_path):
    (tmp_path / "bin").mkdir()
    for name in ("test-backend-ops", "libggml-cpu.so.0"):
        (tmp_path / "bin" / name).write_text("x")
    missing = crw.missing_oracle_parts(tmp_path, gdb="")
    assert len(missing) == (1 if crw.SCRIPT.is_file() else 2)
    assert missing[0].startswith("gdb")


def test_all_parts_present_reports_nothing(tmp_path):
    (tmp_path / "bin").mkdir()
    for name in ("test-backend-ops", "libggml-cpu.so.0"):
        (tmp_path / "bin" / name).write_text("x")
    assert crw.SCRIPT.is_file()
    assert crw.missing_oracle_parts(tmp_path, gdb="/usr/bin/gdb") == []


def test_missing_binary_and_dso_are_named(tmp_path):
    missing = crw.missing_oracle_parts(tmp_path, gdb="/usr/bin/gdb")
    assert any(m.endswith("test-backend-ops") for m in missing)
    assert any(m.endswith("libggml-cpu.so.0") for m in missing)


def test_engagement_reason_names_missing_part(tmp_path, monkeypatch):
    monkeypatch.setattr(crw.shutil, "which", lambda _n: None)
    monkeypatch.setattr(crw.platform, "machine", lambda: "x86_64")
    result = crw._engagement(tmp_path, object(), resolved_recipe=None)
    assert result.status == "unavailable" and "gdb" in result.reason


def test_prompt_asks_for_reliance_on_shaping_claims():
    task = actors._HYPOTHESIS_TASK
    assert "builds on, differs from, or avoids repeating" in task
    assert "answer [] only if no shown claim bears on it" in task
