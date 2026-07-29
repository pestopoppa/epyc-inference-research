"""Tests for the E5 W2 capture-smoke verdict.

The checker's whole job is to REFUSE a broken capture, so the tests are built
around the real failure it exists to catch — the W0 Gemma shape: HTTP 200,
predicted_n > 0, `response_text: ""`, and no `reasoning_text` field at all
(the old parser read only `content`/`delta.content` while gemma emitted
`reasoning_content`, so the answer channel captured nothing and the reasoning
was never persisted anywhere).

Re-attributed 2026-07-29 (research 5d6a17f2): that parser bug was real and is
fixed, but the token budget was spent in the reasoning channel because the
harness emitted no `--reasoning` flag and gemma4 defaults to reasoning ON. The
checker detects the shape; `--reasoning off` is what prevents it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "e5_w2_capture_smoke_check.py"
_spec = importlib.util.spec_from_file_location("e5_w2_capture_smoke_check", MODULE_PATH)
assert _spec and _spec.loader
smoke = importlib.util.module_from_spec(_spec)
sys.modules["e5_w2_capture_smoke_check"] = smoke
_spec.loader.exec_module(smoke)


def write_run(tmp: Path, responses: list[dict], cells: list[dict] | None = None) -> Path:
    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "responses.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in responses),
        encoding="utf-8",
    )
    (run / "cells.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in (cells or [])),
        encoding="utf-8",
    )
    return run


def good_row(qid: str) -> dict:
    # Both channels genuinely populated — the fixture must not quietly remove
    # the signal the checker is testing for.
    return {
        "cell_id": "gemma4_26b_a4b_q4km_mtp-C1-np8-capturesmoke",
        "qid": qid,
        "response_text": "The answer is 42.",
        "reasoning_text": "Let me work through this step by step.",
        "http_status": 200,
        "timings": {"predicted_n": 200},
    }


def w0_broken_row(qid: str) -> dict:
    """The exact historic W0 Gemma row shape."""
    return {
        "cell_id": "gemma4_26b_a4b_q4km_mtp-C1-np1-scout",
        "qid": qid,
        "response_text": "",
        "http_status": 200,
        "timings": {"predicted_n": 64},
    }


def test_capture_accepts_populated_answer_and_reasoning_channels():
    with tempfile.TemporaryDirectory() as tmp:
        run = write_run(Path(tmp), [good_row(f"q{i}") for i in range(43)])
        failures, stats = smoke.check_capture(run)
    assert failures == []
    assert stats["generated"] == 43
    assert stats["empty_answer_text"] == 0
    assert stats["rows_with_reasoning_text"] == 43


def test_capture_refuses_the_w0_gemma_failure_shape():
    with tempfile.TemporaryDirectory() as tmp:
        run = write_run(Path(tmp), [w0_broken_row(f"q{i}") for i in range(43)])
        failures, stats = smoke.check_capture(run)
    assert stats["empty_answer_text"] == 43
    assert stats["rows_with_reasoning_text"] == 0
    joined = " | ".join(failures)
    assert "lack a reasoning_text field" in joined
    assert "EMPTY answer text" in joined


def test_capture_refuses_reasoning_only_output():
    # Reasoning IS captured (post-fix field present) but the answer channel is
    # still empty — the failure that would otherwise look "fixed" because the
    # new field exists.
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for i in range(43):
            row = good_row(f"q{i}")
            row["response_text"] = "   "  # whitespace is not an answer
            rows.append(row)
        run = write_run(Path(tmp), rows)
        failures, stats = smoke.check_capture(run)
    assert stats["empty_answer_text"] == 43
    assert any("EMPTY answer text" in f for f in failures)


def test_capture_ignores_rows_that_generated_nothing():
    # predicted_n == 0 means no tokens were produced, so an empty answer is not
    # a capture defect — the fail-close is conditioned on generation.
    with tempfile.TemporaryDirectory() as tmp:
        row = good_row("q0")
        row["response_text"] = ""
        row["timings"] = {"predicted_n": 0}
        run = write_run(Path(tmp), [row])
        failures, stats = smoke.check_capture(run)
    assert stats["generated"] == 0
    assert not any("EMPTY answer text" in f for f in failures)


def test_capture_surfaces_the_harness_fail_close_blocker():
    with tempfile.TemporaryDirectory() as tmp:
        run = write_run(
            Path(tmp),
            [good_row("q0")],
            cells=[
                {
                    "cell_id": "gemma4_26b_a4b_q4km_mtp-C1-np8-capturesmoke",
                    "decision_grade_blockers": [
                        "response_capture_failure: 7 generated response(s) "
                        "lacked answer-text SSE deltas"
                    ],
                }
            ],
        )
        failures, stats = smoke.check_capture(run)
    assert stats["capture_fail_close_cells"] == 1
    assert any("response_capture_failure" in f for f in failures)


def test_empty_ledger_is_a_failure_not_a_pass():
    # A run that captured nothing must never soft-pass by vacuous truth.
    with tempfile.TemporaryDirectory() as tmp:
        run = write_run(Path(tmp), [])
        failures, _ = smoke.check_capture(run)
    assert failures and "nothing was captured" in failures[0]


def test_scorer_verdict_uses_parse_ok_budget():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        run.mkdir()
        rows = [
            {"cell_id": "cellA", "qid": f"q{i}", "parse_ok": i >= 2}
            for i in range(43)
        ]
        (run / "offline_scores.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
        )

        class _Done:
            returncode = 0
            stdout = ""

        original = smoke.subprocess.run
        smoke.subprocess.run = lambda *a, **k: _Done()  # type: ignore[assignment]
        try:
            # exactly at budget (2 failures) -> pass
            failures, stats = smoke.check_scorer(run, "python3")
            assert failures == []
            assert stats["parse_ok_by_cell"]["cellA"] == {"total": 43, "parse_ok": 41}

            # one over budget -> fail
            rows[2]["parse_ok"] = False
            (run / "offline_scores.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                encoding="utf-8",
            )
            failures, _ = smoke.check_scorer(run, "python3")
            assert failures and "parse failures" in failures[0]
        finally:
            smoke.subprocess.run = original  # type: ignore[assignment]


def test_scorer_failure_is_not_swallowed():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        run.mkdir()

        class _Failed:
            returncode = 2
            stdout = "boom"

        original = smoke.subprocess.run
        smoke.subprocess.run = lambda *a, **k: _Failed()  # type: ignore[assignment]
        try:
            failures, _ = smoke.check_scorer(run, "python3")
        finally:
            smoke.subprocess.run = original  # type: ignore[assignment]
    assert failures and "exited 2" in failures[0]
