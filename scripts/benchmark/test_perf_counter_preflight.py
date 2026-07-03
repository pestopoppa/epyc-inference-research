#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess

import pytest

import perf_counter_preflight as p


def test_print_event_csv(capsys):
    assert p.main(["--print-event-csv"]) == 0
    out = capsys.readouterr().out.strip()
    assert "fp_ops_retired_by_type.vector_mac" in out
    assert "task-clock" in out
    assert "\n" not in out


def test_missing_perf_blocks(monkeypatch):
    monkeypatch.setattr(p.shutil, "which", lambda name: None)

    report = p.build_report()

    assert report["status"] == "blocked"
    assert report["perf"]["found"] is False
    assert "Install or expose" in report["recommendation"]


def test_perf_list_missing_canonical_event_blocks(monkeypatch):
    monkeypatch.setattr(p.shutil, "which", lambda name: "/usr/bin/perf")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="  cycles\n  instructions\n  task-clock\n",
            stderr="",
        )

    monkeypatch.setattr(p.subprocess, "run", fake_run)

    report = p.build_report()

    assert report["status"] == "blocked"
    assert "cycles" in report["events"]["present"]
    assert "fp_ops_retired_by_type.vector_mac" in report["events"]["missing"]


def test_all_events_visible_with_probe_passes(monkeypatch):
    monkeypatch.setattr(p.shutil, "which", lambda name: "/usr/bin/perf")
    events_text = "\n".join(f"  {event}" for event in p.CANONICAL_PERF_EVENTS)

    def fake_run(args, **kwargs):
        if args[1:3] == ["list", "--no-desc"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=events_text, stderr="")
        if args[1:3] == ["stat", "-x,"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="1,cycles\n")
        raise AssertionError(args)

    monkeypatch.setattr(p.subprocess, "run", fake_run)

    report = p.build_report(probe=True, probe_duration_s=0.01)

    assert report["status"] == "ok"
    assert report["events"]["missing"] == []
    assert report["probe"]["ok"] is True


def test_writes_json_and_markdown(monkeypatch, tmp_path):
    monkeypatch.setattr(p.shutil, "which", lambda name: None)
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    rc = p.main(["--output-json", str(output_json), "--output-md", str(output_md)])

    assert rc == 0
    assert json.loads(output_json.read_text())["schema"] == p.SCHEMA
    assert "AMD Perf Counter Preflight" in output_md.read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
