"""Deterministic fixtures for the offline ContextBench scorer; no dataset or checkout I/O."""
from __future__ import annotations

import json

import pytest

from src.contextbench_score import (
    DISPOSITION_SCHEMA,
    ScoringInputError,
    normalize_relative_path,
    score_contextbench_bytes,
    score_contextbench_records,
)


def _candidate(path, *, total_lines=3, ranges=((2, 2),), full=7000, slices=2500, codemap=500):
    return {
        "path": path,
        "priority": 1.0,
        "cost_full": full,
        "cost_slices": slices,
        "cost_codemap": codemap,
        "desired_mode": "full",
        "line_ranges": [list(pair) for pair in ranges],
        "total_lines": total_lines,
    }


def _prediction(task, arm, files=(), spans=(), candidates=(), status="ok"):
    return {
        "task_id": task,
        "arm": arm,
        "status": status,
        "pred_files": list(files),
        "pred_spans": list(spans),
        "pack_candidates": list(candidates),
    }


def _inputs():
    tasks = [
        {"instance_id": "small", "gold_context": [
            {"file": ".config/settings.py", "start_line": 1, "end_line": 2},
            {"file": "./.config//settings.py", "start_line": 2, "end_line": 3},
        ]},
        {"instance_id": "large", "gold_context": [
            {"file": "src/a.py", "start_line": 3, "end_line": 4},
            {"file": "src/b.py", "start_line": 1, "end_line": 5},
        ]},
        {"instance_id": "empty-gold", "gold_context": []},
        {"instance_id": "scratch", "gold_context": [
            {"file": "fixtures/scratch.txt", "start_line": 1, "end_line": 2},
        ]},
        {"instance_id": "unresolved", "gold_context": [
            {"file": "../outside.py", "start_line": 1, "end_line": 2},
        ]},
        {"instance_id": "safe-unresolved", "gold_context": [
            {"file": "unindexed/generated.py", "start_line": 7, "end_line": 9},
        ]},
    ]
    dispositions = {
        "schema": DISPOSITION_SCHEMA,
        "tasks": [
            {"task_id": "small", "disposition": "include", "evidence_paths": []},
            {"task_id": "large", "disposition": "include", "evidence_paths": []},
            {"task_id": "empty-gold", "disposition": "exclude_empty_gold", "evidence_paths": []},
            {"task_id": "scratch", "disposition": "exclude_scratch_gold",
             "evidence_paths": ["fixtures/scratch.txt"]},
            {"task_id": "unresolved", "disposition": "exclude_unresolvable_gold", "evidence_paths": []},
            {"task_id": "safe-unresolved", "disposition": "exclude_unresolvable_gold",
             "evidence_paths": ["unindexed/generated.py"]},
        ],
    }
    candidate = _candidate(".config/settings.py")
    predictions = [
        _prediction("small", "found", ["./.config/settings.py", ".config/settings.py"], [
            {"path": ".config/settings.py", "start_line": 1, "end_line": 2},
            {"path": "./.config/settings.py", "start_line": 2, "end_line": 3},
        ], [candidate]),
        _prediction("large", "found", ["src/a.py"], [
            {"path": "src/a.py", "start_line": 3, "end_line": 4},
        ]),
        _prediction("small", "empty", status="no_context_extracted"),
        _prediction("large", "empty", status="checkout_failed"),
    ]
    return tasks, predictions, dispositions


def _score():
    tasks, predictions, dispositions = _inputs()
    return score_contextbench_records(tasks, predictions, dispositions,
                                      declared_arms=["found", "empty"])


def _scope(result, arm, scope, budget=None):
    arms = {row["arm"]: row for row in result["arms"]}
    return next(row for row in arms[arm]["scopes"]
                if row["scope"] == scope and row["budget_tokens"] == budget)


def test_safe_posix_path_normalization_preserves_dotfiles_and_refuses_escape():
    assert normalize_relative_path("./a//b.py") == "a/b.py"
    assert normalize_relative_path(".config/settings.py") == ".config/settings.py"
    for unsafe in ("../secret", "a/../secret", "/absolute/path", "C:/repo/a.py", "a\\b.py", "a\x00b"):
        with pytest.raises(ScoringInputError):
            normalize_relative_path(unsafe)


def test_duplicate_json_members_in_every_input_layer_are_refused():
    valid_task = b'{"instance_id":"t","gold_context":[{"file":"a.py","start_line":1,"end_line":1}]}\n'
    valid_prediction = b'{"task_id":"t","arm":"a","status":"ok","pred_files":["a.py"],"pred_spans":[],"pack_candidates":[]}\n'
    valid_dispositions = b'{"schema":"epyc.contextbench_dispositions.v1","tasks":[{"task_id":"t","disposition":"include","evidence_paths":[]}]}'
    cases = [
        (b'{"instance_id":"t","instance_id":"t","gold_context":[]}\n',
         valid_prediction, valid_dispositions),
        (json.dumps({"instance_id": "t", "gold_context":
                     '[{"file":"a.py","file":"b.py","start_line":1,"end_line":1}]'}).encode() + b"\n",
         valid_prediction, valid_dispositions),
        (valid_task,
         b'{"task_id":"t","task_id":"t","arm":"a","status":"ok","pred_files":[],"pred_spans":[],"pack_candidates":[]}\n',
         valid_dispositions),
        (valid_task, valid_prediction,
         b'{"schema":"epyc.contextbench_dispositions.v1","schema":"epyc.contextbench_dispositions.v1","tasks":[]}'),
    ]
    for task_bytes, prediction_bytes, disposition_bytes in cases:
        with pytest.raises(ScoringInputError, match="duplicate"):
            score_contextbench_bytes(task_bytes, prediction_bytes, disposition_bytes,
                                     declared_arms=["a"])


def test_scoring_deduplicates_overlapping_inclusive_spans_and_macro_averages_tasks():
    result = _score()
    found = _scope(result, "found", "discovery")
    assert result["eligible_task_ids"] == ["large", "small"]
    assert result["eligible_task_count"] == 2
    # Per-task file recall is 1.0 for small and 0.5 for large, not pooled 2/3.
    assert found["metrics"]["file_recall"] == {"value": 0.75, "direction": "higher_better", "unit": "fraction"}
    small = next(row for row in found["per_task"] if row["task_id"] == "small")
    assert small["counts"]["line"] == {"tp": 3, "fp": 0, "fn": 0}
    assert small["metrics"]["line_f1"] == 1.0
    assert {row["reason"] for row in result["excluded"]} == {
        "exclude_empty_gold", "exclude_scratch_gold", "exclude_unresolvable_gold"
    }


def test_empty_predictions_and_failures_are_zero_and_remain_in_macro_denominator():
    empty = _scope(_score(), "empty", "discovery")
    assert empty["metrics"]["file_precision"]["value"] == 0.0
    assert empty["metrics"]["file_recall"]["value"] == 0.0
    assert empty["metrics"]["line_f1"]["value"] == 0.0
    assert [row["status"] for row in empty["per_task"]] == ["checkout_failed", "no_context_extracted"]


def test_pack_scopes_are_deterministic_and_map_full_slice_codemap_to_spans():
    result = _score()
    assert result == _score()
    packed_2k = _scope(result, "found", "packed", 2000)
    packed_4k = _scope(result, "found", "packed", 4000)
    packed_8k = _scope(result, "found", "packed", 8000)
    # The large candidate does not fit any mode; for the small candidate the fixed packer
    # selects codemap at 2k, slices at 4k, and full content at 8k.
    assert packed_2k["metrics"]["line_recall"]["value"] == 0.0
    assert packed_4k["metrics"]["line_recall"]["value"] == pytest.approx(1 / 6)
    assert packed_8k["metrics"]["line_recall"]["value"] == pytest.approx(0.5)
    assert result["budget_cost_basis"] == "caller_supplied_candidate_cost_fields"
    assert "no tokenizer or cost estimator runs here" in result["budget_note"]


def test_error_row_partial_outputs_are_preserved_but_score_zero_in_denominator():
    tasks, predictions, dispositions = _inputs()
    row = next(item for item in predictions if item["task_id"] == "small" and item["arm"] == "empty")
    row["pred_files"] = [".config/settings.py"]
    row["pred_spans"] = [{"path": ".config/settings.py", "start_line": 1, "end_line": 2}]
    row["pack_candidates"] = [_candidate(".config/settings.py")]
    result = score_contextbench_records(tasks, predictions, dispositions,
                                        declared_arms=["found", "empty"])
    for scope_name, budget in (("discovery", None), ("packed", 2000),
                               ("packed", 4000), ("packed", 8000)):
        scope = _scope(result, "empty", scope_name, budget)
        assert [item["status"] for item in scope["per_task"]] == [
            "checkout_failed", "no_context_extracted"
        ]
        assert scope["metrics"]["file_recall"]["value"] == 0.0
        assert scope["metrics"]["line_recall"]["value"] == 0.0
        small = next(item for item in scope["per_task"] if item["task_id"] == "small")
        assert small["counts"]["file"]["tp"] == small["counts"]["line"]["tp"] == 0


def test_error_row_malformed_partial_output_is_still_refused():
    tasks, predictions, dispositions = _inputs()
    row = next(item for item in predictions if item["task_id"] == "small" and item["arm"] == "empty")
    row["pred_files"] = [".config/settings.py"]
    row["pred_spans"] = [{"path": "elsewhere.py", "start_line": 1, "end_line": 2}]
    with pytest.raises(ScoringInputError, match="span names an unlisted file"):
        score_contextbench_records(tasks, predictions, dispositions, declared_arms=["found", "empty"])


def test_missing_prediction_or_duplicate_task_arm_is_refused():
    tasks, predictions, dispositions = _inputs()
    with pytest.raises(ScoringInputError, match="coverage mismatch"):
        score_contextbench_records(tasks, predictions[:-1], dispositions,
                                   declared_arms=["found", "empty"])
    with pytest.raises(ScoringInputError, match="duplicate prediction"):
        score_contextbench_records(tasks, predictions + [predictions[0]], dispositions,
                                   declared_arms=["found", "empty"])


def test_safe_unresolvable_disposition_requires_exact_caller_audited_gold_paths():
    tasks, predictions, dispositions = _inputs()
    disposition = next(row for row in dispositions["tasks"]
                       if row["task_id"] == "safe-unresolved")
    disposition["evidence_paths"] = []
    with pytest.raises(ScoringInputError, match="safe unresolvable exclusion requires distinct evidence"):
        score_contextbench_records(tasks, predictions, dispositions,
                                   declared_arms=["found", "empty"])


def test_disposition_manifest_must_cover_exact_rows_and_match_empty_gold():
    tasks, predictions, dispositions = _inputs()
    dispositions["tasks"].pop()
    with pytest.raises(ScoringInputError, match="lacks an explicit disposition"):
        score_contextbench_records(tasks, predictions, dispositions, declared_arms=["found", "empty"])
    tasks, predictions, dispositions = _inputs()
    dispositions["tasks"][2]["disposition"] = "include"
    with pytest.raises(ScoringInputError, match="include requires nonempty safe gold"):
        score_contextbench_records(tasks, predictions, dispositions, declared_arms=["found", "empty"])


def test_candidate_rows_reject_duplicate_paths_invalid_ranges_and_bad_costs():
    tasks, predictions, dispositions = _inputs()
    small = next(item for item in predictions if item["task_id"] == "small" and item["arm"] == "found")
    small["pack_candidates"] = [_candidate(".config/settings.py"), _candidate(".config/settings.py")]
    with pytest.raises(ScoringInputError, match="duplicate candidate path"):
        score_contextbench_records(tasks, predictions, dispositions, declared_arms=["found", "empty"])
    tasks, predictions, dispositions = _inputs()
    small = next(item for item in predictions if item["task_id"] == "small" and item["arm"] == "found")
    small["pack_candidates"][0]["line_ranges"] = [[0, 2]]
    with pytest.raises(ScoringInputError, match="inclusive 1-based"):
        score_contextbench_records(tasks, predictions, dispositions, declared_arms=["found", "empty"])


def test_byte_entrypoint_refuses_blank_malformed_and_duplicate_rows():
    tasks, predictions, dispositions = _inputs()
    task_bytes = b"\n".join(json.dumps(row).encode() for row in tasks) + b"\n"
    pred_bytes = b"\n".join(json.dumps(row).encode() for row in predictions) + b"\n"
    disposition_bytes = json.dumps(dispositions).encode()
    result = score_contextbench_bytes(task_bytes, pred_bytes, disposition_bytes,
                                      declared_arms=["found", "empty"])
    assert result["schema"] == "epyc.contextbench_discovery_score.v1"
    with pytest.raises(ScoringInputError, match="empty row"):
        score_contextbench_bytes(task_bytes + b"\n", pred_bytes, disposition_bytes,
                                 declared_arms=["found", "empty"])
    duplicate_task_bytes = task_bytes + json.dumps(tasks[0]).encode() + b"\n"
    with pytest.raises(ScoringInputError, match="duplicate task"):
        score_contextbench_bytes(duplicate_task_bytes, pred_bytes, disposition_bytes,
                                 declared_arms=["found", "empty"])
