"""Incremental EvalTower question-result sidecar persistence."""

from __future__ import annotations

import errno
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "autopilot"))

import eval_tower  # noqa: E402
from eval_tower import EvalTower, QuestionResult  # noqa: E402


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sidecar_path(root: Path, trial_id: int) -> Path:
    return root / f"trial_{trial_id}" / "question_results.jsonl"


def test_eval_batch_persists_serial_question_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    tower.set_trial_context(101)

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt="SECRET_PROMPT",
            expected="SECRET_EXPECTED",
            answer="SECRET_RAW_ANSWER",
            correct=bool(q["correct"]),
            tokens_generated=7,
            elapsed_s=0.25,
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    results = tower._eval_batch(
        [{"id": "q1", "correct": True}, {"id": "q2", "correct": False}],
        client=object(),  # type: ignore[arg-type]
        label="T1",
    )

    rows = _read_rows(_sidecar_path(tmp_path, 101))
    question_rows = [row for row in rows if row["row_type"] == "question_result"]
    assert [row["ordinal"] for row in question_rows] == [0, 1]
    assert {row["trial_id"] for row in question_rows} == {101}
    assert {row["label"] for row in question_rows} == {"T1"}
    assert {row["requested_n"] for row in question_rows} == {2}
    assert len({row["eval_batch_id"] for row in question_rows}) == 1
    assert question_rows[0]["result"]["question_id"] == "q1"
    assert question_rows[0]["result"]["correct"] is True
    assert question_rows[1]["result"]["correct"] is False


def test_eval_batch_persists_concurrent_question_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "3")
    tower = EvalTower()
    tower.set_trial_context(202)

    # workers>1 drives the pipelined generation/scoring pools, so the fake
    # replaces the GENERATION phase and hands scoring a ready ``final_result``.
    def fake_generate(q: dict, client: object) -> "eval_tower._GenOutcome":
        time.sleep(float(q.get("delay", 0.0)))
        return eval_tower._GenOutcome(
            gen_ended_at_s=time.time(),
            final_result=QuestionResult(
                question_id=str(q["id"]),
                suite="unit",
                prompt=str(q["id"]),
                expected="ok",
                answer="ok",
                correct=True,
            ),
        )

    monkeypatch.setattr(tower, "_generate_question", fake_generate)

    results = tower._eval_batch(
        [
            {"id": "q1", "delay": 0.02},
            {"id": "q2", "delay": 0.0},
            {"id": "q3", "delay": 0.01},
        ],
        client=object(),  # type: ignore[arg-type]
        label="T2",
    )

    rows = _read_rows(_sidecar_path(tmp_path, 202))
    question_rows = [row for row in rows if row["row_type"] == "question_result"]
    assert sorted(row["ordinal"] for row in question_rows) == [0, 1, 2]
    assert {row["result"]["question_id"] for row in question_rows} == {"q1", "q2", "q3"}
    assert len({row["eval_batch_id"] for row in question_rows}) == 1
    assert results.sidecar_persistence["result_rows_fsync_returned_n"] == 3
    assert results.sidecar_persistence["archive_complete"] is True


def test_eval_batch_sidecar_persists_answer_but_not_prompt_or_expected(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # 2026-07-22 operator directive: the JSONL sidecar now persists the FULL raw
    # answer at the row level so arms are re-scorable/resumable from the artifact
    # alone. `prompt`/`expected` stay excluded (reconstructable from the dataset),
    # and the COMPACT `result` block is unchanged (answer_hash only).
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    tower.set_trial_context(303)

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id="q-secret",
            suite="unit",
            prompt="DO_NOT_WRITE_PROMPT",
            expected="DO_NOT_WRITE_EXPECTED",
            answer="PERSIST_THIS_ANSWER",
            correct=True,
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    tower._eval_batch([{"id": "q-secret"}], client=object(), label="leak")  # type: ignore[arg-type]

    text = _sidecar_path(tmp_path, 303).read_text(encoding="utf-8")
    assert "DO_NOT_WRITE_PROMPT" not in text
    assert "DO_NOT_WRITE_EXPECTED" not in text
    # Raw answer IS persisted now (row-level), for re-scoring / resume.
    assert "PERSIST_THIS_ANSWER" in text
    row = [
        r
        for r in _read_rows(_sidecar_path(tmp_path, 303))
        if r["row_type"] == "question_result"
    ][0]
    assert row["answer"] == "PERSIST_THIS_ANSWER"
    # The compact result block stays compact — answer_hash only, no raw fields.
    assert "answer_hash" in row["result"]
    assert "prompt" not in row["result"]
    assert "expected" not in row["result"]
    assert "answer" not in row["result"]


def test_eval_batch_writes_complete_marker_on_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    tower.set_trial_context(404)

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt=str(q["id"]),
            expected="ok",
            answer="ok",
            correct=True,
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)

    results = tower._eval_batch(
        [{"id": "q1"}, {"id": "q2"}], client=object(), label="complete"
    )  # type: ignore[arg-type]

    rows = _read_rows(_sidecar_path(tmp_path, 404))
    assert rows[0]["row_type"] == "batch_start"
    assert rows[-1]["row_type"] == "batch_complete"
    assert rows[-1]["complete"] is True
    assert rows[-1]["completed_n"] == 2
    assert rows[-1]["recovery_contract"] == "complete_marker_required"
    assert results.sidecar_persistence["archive_complete"] is True
    assert results.sidecar_persistence["archive_status"] == "complete_by_writer_returns"
    assert results.sidecar_persistence["result_rows_fsync_returned_n"] == 2


def test_writer_initialization_failure_is_reported_without_changing_results(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt="prompt",
            expected="ok",
            answer="ok" if q["correct"] else "wrong",
            correct=bool(q["correct"]),
        )

    def fail_writer_init(*_args, **_kwargs) -> None:
        raise OSError(errno.EACCES, "private path detail")

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)
    monkeypatch.setattr(eval_tower._EvalQuestionJsonlWriter, "__init__", fail_writer_init)

    results = tower._eval_batch(
        [{"id": "q1", "correct": True}, {"id": "q2", "correct": False}],
        client=object(),  # type: ignore[arg-type]
        label="init-failure",
    )

    assert isinstance(results, list)
    assert [row.correct for row in results] == [True, False]
    assert [row.disposition for row in results] == ["scored", "scored"]
    status = results.sidecar_persistence
    assert status["writer_initialized"] is False
    assert status["completed_n"] == 2
    assert status["result_rows_attempted_n"] == 0
    assert status["result_rows_not_attempted_n"] == 2
    assert status["result_rows_fsync_returned_n"] == 0
    assert status["completion_marker_attempted"] is False
    assert status["archive_complete"] is False
    assert status["failure_reasons"] == [
        {
            "operation": "initialize",
            "error_type": "PermissionError",
            "error_code": "EACCES",
        }
    ]
    assert "private path detail" not in json.dumps(status)
    aggregate = tower._aggregate(results, tier=1)
    assert aggregate.quality == 0.5
    assert aggregate.details["question_sidecar_persistence"] == status


def test_result_append_failure_keeps_completion_marker_but_marks_archive_incomplete(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    tower.set_trial_context(505)
    original_append_result = eval_tower._EvalQuestionJsonlWriter.append_result

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt="prompt",
            expected="ok",
            answer="ok" if q["correct"] else "wrong",
            correct=bool(q["correct"]),
        )

    def fail_first_result(self, *, ordinal: int, **kwargs) -> None:
        if ordinal == 0:
            raise OSError(errno.EIO, "/unreported/private/path")
        original_append_result(self, ordinal=ordinal, **kwargs)

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)
    monkeypatch.setattr(eval_tower._EvalQuestionJsonlWriter, "append_result", fail_first_result)

    results = tower._eval_batch(
        [{"id": "q1", "correct": True}, {"id": "q2", "correct": False}],
        client=object(),  # type: ignore[arg-type]
        label="append-failure",
    )
    rows = _read_rows(_sidecar_path(tmp_path, 505))
    status = results.sidecar_persistence

    assert [row.correct for row in results] == [True, False]
    assert [row.disposition for row in results] == ["scored", "scored"]
    assert [row["row_type"] for row in rows] == ["batch_start", "question_result", "batch_complete"]
    assert rows[-1]["completed_n"] == 2
    assert status["result_rows_attempted_n"] == 2
    assert status["result_rows_not_attempted_n"] == 0
    assert status["result_rows_fsync_returned_n"] == 1
    assert status["result_rows_failed_n"] == 1
    assert status["completion_marker_fsync_returned"] is True
    assert status["archive_complete"] is False
    assert status["archive_status"] == "incomplete"
    assert status["failure_reasons"] == [
        {
            "operation": "question_result",
            "error_type": "OSError",
            "error_code": "EIO",
        }
    ]
    assert "/unreported/private/path" not in json.dumps(status)
    aggregate = tower._aggregate(results, tier=1)
    assert aggregate.quality == 0.5
    assert aggregate.details["quality_denominator"] == 2
    assert aggregate.details["question_sidecar_persistence"]["archive_complete"] is False


def test_completion_marker_failure_is_reported_in_role_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()
    tower.set_trial_context(606)

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt="prompt",
            expected="ok",
            answer="ok",
            correct=True,
            confidence=0.75,
            confidence_source="completion_probabilities_geomean",
        )

    def fail_complete(self, *, completed_n: int, elapsed_s: float) -> None:
        raise OSError(errno.ENOSPC, "private completion detail")

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)
    monkeypatch.setattr(
        tower,
        "_load_verifier_suite_questions",
        lambda suite, split, **_kwargs: [
            {"id": "q1", "suite": suite, "prompt": "prompt", "expected": "ok"}
        ],
    )
    monkeypatch.setattr(eval_tower._EvalQuestionJsonlWriter, "append_complete", fail_complete)

    report = tower.eval_calibration("unit", "test", roles=["worker_general"], n=1)
    role_status = report["per_role"]["worker_general"]["question_sidecar_persistence"]

    assert role_status["batch_start_fsync_returned"] is True
    assert role_status["result_rows_fsync_returned_n"] == 1
    assert role_status["completion_marker_attempted"] is True
    assert role_status["completion_marker_fsync_returned"] is False
    assert role_status["archive_status"] == "incomplete"
    assert role_status["failure_reasons"] == [
        {
            "operation": "batch_complete",
            "error_type": "OSError",
            "error_code": "ENOSPC",
        }
    ]
    assert "private completion detail" not in json.dumps(role_status)


def test_partition_filtered_aggregate_keeps_full_batch_capture_status(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AUTOPILOT_EVAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTOPILOT_EVAL_CONCURRENCY", "1")
    tower = EvalTower()

    def fake_eval_question(q: dict, client: object) -> QuestionResult:
        return QuestionResult(
            question_id=str(q["id"]),
            suite="unit",
            prompt="prompt",
            expected="ok",
            answer="ok" if q["correct"] else "wrong",
            correct=bool(q["correct"]),
            eval_partition=str(q["eval_partition"]),
        )

    monkeypatch.setattr(tower, "_eval_question", fake_eval_question)
    results = tower._eval_batch(
        [
            {"id": "core", "correct": True, "eval_partition": "core"},
            {"id": "audit", "correct": False, "eval_partition": "audit"},
        ],
        client=object(),  # type: ignore[arg-type]
        label="partition-filter",
    )

    aggregate = tower._aggregate_decision_partitions(
        results,
        tier=1,
        excluded_partitions={"audit"},
    )

    assert aggregate.quality == 1.0
    assert aggregate.details["quality_denominator"] == 1
    assert aggregate.details["question_sidecar_persistence"] == results.sidecar_persistence
    assert aggregate.details["question_sidecar_persistence"]["completed_n"] == 2
    assert aggregate.details["question_sidecar_persistence"]["result_rows_fsync_returned_n"] == 2


def test_empty_or_missing_batch_status_is_never_reported_complete() -> None:
    tower = EvalTower()
    empty_batch = tower._eval_batch([], client=object(), label="empty")  # type: ignore[arg-type]
    empty_aggregate = tower._aggregate(empty_batch, tier=1)
    legacy_aggregate = tower._aggregate([], tier=1)

    assert empty_batch.sidecar_persistence["archive_status"] == "not_applicable"
    assert empty_batch.sidecar_persistence["archive_complete"] is None
    assert (
        empty_aggregate.details["question_sidecar_persistence"]["archive_status"]
        == "not_applicable"
    )
    assert "question_sidecar_persistence" not in legacy_aggregate.details
