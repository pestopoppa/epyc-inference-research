from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import architect_bench_analyze as analyze


def _row(qid: str, *, correct: bool = True, speed: float = 10.0) -> dict:
    return {"suite": "toy", "id": qid, "seed": 42,
            "correct": correct, "decode_tok_s": speed}


def test_reports_no_faster_fraction_among_correct_outputs_only():
    candidate = [_row(f"q{i}", speed=8.0 if i < 9 else 10.0 if i < 15 else 12.0)
                 for i in range(31)]
    candidate.extend([_row("wrong-1", correct=False, speed=1.0),
                      _row("wrong-2", correct=False, speed=50.0)])
    baseline = [_row(f"q{i}", speed=10.0) for i in range(31)]

    got = analyze.slow_crr(candidate, baseline)

    assert got["status"] == "reported"
    assert got["solved"] == 31
    assert got["no_faster"] == 15
    assert got["fraction"] == 15 / 31


def test_existing_thirty_correct_output_floor_suppresses_small_denominator():
    got = analyze.slow_crr([_row(f"q{i}") for i in range(29)],
                           [_row(f"q{i}") for i in range(29)])
    assert got["status"] == "below_minimum"
    assert got["solved"] == 29
    assert got["fraction"] is None
    assert got["no_faster"] is None


def test_missing_baseline_pair_makes_conditional_unavailable():
    got = analyze.slow_crr([_row("q1")], [])
    assert got["status"] == "incomplete"
    assert got["solved"] == 1
    assert got["unmatched"] == 1
    assert got["fraction"] is None
    assert "unmatched=1" in got["reason"]


def test_missing_or_invalid_speed_does_not_shrink_denominator():
    candidate = [_row(f"q{i}") for i in range(30)]
    baseline = [_row(f"q{i}") for i in range(30)]
    baseline[4].pop("decode_tok_s")
    got = analyze.slow_crr(candidate, baseline)
    assert got["status"] == "incomplete"
    assert got["solved"] == 30
    assert got["invalid_baseline_speed"] == 1
    assert got["fraction"] is None
    assert "invalid_baseline_speed=1" in got["reason"]


def test_duplicate_question_seed_identity_is_refused():
    candidate = [_row("q1"), _row("q1")]
    baseline = [_row("q1")]
    got = analyze.slow_crr(candidate, baseline)
    assert got["status"] == "incomplete"
    assert "duplicate_identity=1" in got["reason"]


def test_duplicate_baseline_identity_is_refused():
    got = analyze.slow_crr([_row("q1")], [_row("q1"), _row("q1")])
    assert got["status"] == "incomplete"
    assert got["solved"] == 1
    assert got["duplicate_baseline_identity"] == 1
    assert got["fraction"] is None


def test_candidate_correctness_must_be_strict_boolean():
    candidate = [_row("q1")]
    candidate[0]["correct"] = 1
    got = analyze.slow_crr(candidate, [_row("q1")])
    assert got["status"] == "incomplete"
    assert "correctness=1" in got["reason"]


def test_missing_candidate_speed_is_counted_and_denominator_is_preserved():
    candidate = [_row(f"q{i}") for i in range(30)]
    candidate[7].pop("decode_tok_s")
    got = analyze.slow_crr(candidate, [_row(f"q{i}") for i in range(30)])
    assert got["status"] == "incomplete"
    assert got["solved"] == 30
    assert got["invalid_candidate_speed"] == 1
    assert got["fraction"] is None
    assert "invalid_candidate_speed=1" in got["reason"]


def test_cli_integrates_diagnostic_against_explicit_baseline(tmp_path: Path):
    runs = tmp_path / "runs"
    rows_by_arm = {"baseline": [], "candidate": []}
    for i in range(30):
        rows_by_arm["baseline"].append(_row(f"q{i}", speed=10.0))
        rows_by_arm["candidate"].append(_row(f"q{i}", speed=8.0 if i < 12 else 11.0))
    for arm, rows in rows_by_arm.items():
        folder = runs / "toy" / arm
        folder.mkdir(parents=True)
        (folder / "per_question.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    script = Path(analyze.__file__)
    proc = subprocess.run(
        [sys.executable, str(script), str(runs), "--slow-crr-baseline", "baseline"],
        check=True, capture_output=True, text=True,
    )
    assert "candidate: 12/30 = 40.0%" in proc.stdout
    assert "minimum 30" in proc.stdout


def _write_cli_rows(runs: Path, rows_by_arm: dict[str, list[dict]]) -> None:
    for arm, rows in rows_by_arm.items():
        folder = runs / "toy" / arm
        folder.mkdir(parents=True)
        (folder / "per_question.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run_cli(runs: Path) -> str:
    script = Path(analyze.__file__)
    proc = subprocess.run(
        [sys.executable, str(script), str(runs), "--slow-crr-baseline", "baseline"],
        check=True, capture_output=True, text=True,
    )
    return proc.stdout


def test_cli_keeps_unmatched_correct_candidate_in_full_denominator(tmp_path: Path):
    runs = tmp_path / "runs"
    rows_by_arm = {
        "baseline": [_row(f"q{i}", speed=10.0) for i in range(30)],
        "candidate": [_row(f"q{i}", speed=8.0 if i < 15 else 12.0)
                      for i in range(30)] + [_row("candidate-only", speed=9.0)],
        "third": [_row(f"q{i}") for i in range(30)],
    }
    _write_cli_rows(runs, rows_by_arm)

    output = _run_cli(runs)

    assert "candidate: unavailable" in output
    assert "unmatched=1" in output
    assert "31 correct outputs" in output
    assert "candidate: 1/30" not in output


def test_cli_third_arm_missing_rows_do_not_shrink_candidate_baseline_scope(
        tmp_path: Path):
    runs = tmp_path / "runs"
    rows_by_arm = {
        "baseline": [_row(f"q{i}", speed=10.0) for i in range(30)],
        "candidate": [_row(f"q{i}", speed=8.0 if i < 12 else 11.0)
                      for i in range(30)],
        "third": [_row(f"q{i}") for i in range(8)],
    }
    _write_cli_rows(runs, rows_by_arm)

    output = _run_cli(runs)

    assert "candidate: 12/30 = 40.0%" in output
