from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "candidate_eval_gate.py"


def run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_candidate_gate_plan_lists_default_steps() -> None:
    result = run_gate("--json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "plan"
    assert report["ok"] is True
    assert [step["name"] for step in report["steps"]] == [
        "docs-check",
        "analysis-check",
        "security-check",
        "health",
        "test",
    ]
    assert {step["status"] for step in report["steps"]} == {"planned"}


def test_candidate_gate_rejects_unknown_step() -> None:
    result = run_gate("--steps", "docs-check,missing")

    assert result.returncode != 0
    assert "unknown gate step(s): missing" in result.stderr


def test_candidate_gate_can_select_subset() -> None:
    result = run_gate("--steps", "security-check,docs-check", "--json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert [step["name"] for step in report["steps"]] == [
        "docs-check",
        "security-check",
    ]
