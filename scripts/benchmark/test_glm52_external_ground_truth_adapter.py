#!/usr/bin/env python3
"""No-inference tests for glm52_external_ground_truth_adapter.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "glm52_external_ground_truth_adapter.py"
_SPEC = importlib.util.spec_from_file_location("glm52_external_ground_truth_adapter", _MODULE_PATH)
adapter = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_external_ground_truth_adapter"] = adapter
_SPEC.loader.exec_module(adapter)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_judgebench_normalizes_ab_labels(tmp_path):
    path = tmp_path / "judgebench.jsonl"
    _write_jsonl(
        path,
        [
            {
                "pair_id": "p1",
                "source": "suite",
                "question": "q",
                "response_A": "better",
                "response_B": "worse",
                "label": "A>B",
            },
            {
                "pair_id": "p2",
                "source": "suite",
                "question": "q",
                "response_A": "worse",
                "response_B": "better",
                "label": "B>A",
            },
        ],
    )

    rows = adapter.load_judgebench(path, suite="gpt")

    assert [row.gold_label for row in rows] == ["A", "B"]
    assert rows[0].source_benchmark == "judgebench"
    assert rows[0].provenance["scoring_method"] == "exact_match"
    assert rows[0].gold_instrument_version.startswith("local-file-sha256:")


def test_llmbar_normalizes_numeric_labels(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            [
                {"input": "q1", "output_1": "a", "output_2": "b", "label": 1},
                {"input": "q2", "output_1": "a", "output_2": "b", "label": 2},
            ]
        ),
        encoding="utf-8",
    )

    rows = adapter.load_llmbar(path, suite="natural")

    assert [row.gold_label for row in rows] == ["A", "B"]
    assert rows[1].source_suite == "natural"


def test_rewardbench_records_emit_chosen_over_rejected():
    rows = adapter.load_rewardbench_records(
        [{"id": "r1", "prompt": "q", "chosen": "good", "rejected": "bad", "subset": "chat"}],
        suite="filtered",
        version="local-file-sha256:test",
    )

    assert len(rows) == 1
    assert rows[0].gold_label == "A"
    assert rows[0].candidate == "good"
    assert rows[0].candidate_b == "bad"
    assert rows[0].source_suite == "chat"


def test_rewardbench2_expands_multi_rejected_and_skips_ties():
    class ArrayLike:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    rows = adapter.load_rewardbench2_records(
        [
            {
                "id": "rb2",
                "prompt": "q",
                "chosen": ArrayLike(["good1", "good2"]),
                "rejected": ArrayLike(["bad1", "bad2"]),
                "subset": "hard",
            },
            {"id": "tie", "prompt": "q", "chosen": ["a"], "rejected": ["b"], "tie": True},
            {"id": "tie-subset", "prompt": "q", "chosen": ["a"], "rejected": ["b"], "subset": "Ties"},
        ],
        suite="test",
        version="local-file-sha256:test",
    )

    assert len(rows) == 4
    assert {row.source_row_id for row in rows} == {"rb2:0:0", "rb2:0:1", "rb2:1:0", "rb2:1:1"}
    assert all(row.gold_label == "A" for row in rows)


def test_judgelm_derives_higher_score_and_skips_equal(tmp_path):
    path = tmp_path / "judgelm.jsonl"
    _write_jsonl(
        path,
        [
            {
                "question_id": 1,
                "question_body": "q",
                "answer1_body": "a1",
                "answer2_body": "a2",
                "score": [{"rougeLsum": 0.2}, {"rougeLsum": 0.4}],
            },
            {
                "question_id": 2,
                "question_body": "q",
                "answer1_body": "a1",
                "answer2_body": "a2",
                "score": [{"rougeLsum": 0.4}, {"rougeLsum": 0.4}],
            },
            {
                "question_id": 3,
                "question_body": "q",
                "answer1_body": "",
                "answer2_body": "a2",
                "score": [{"rougeLsum": 0.6}, {"rougeLsum": 0.4}],
            },
        ],
    )

    rows = adapter.load_judgelm(path, suite="val", score_key="rougeLsum")

    assert len(rows) == 1
    assert rows[0].gold_label == "B"
    assert rows[0].provenance["score_b"] == 0.4


def test_pairwise_scoring_handles_correct_wrong_and_parse_failure():
    assert adapter.score_pairwise_text('{"decision":"A","confidence":0.8}', "A")["correct"] is True
    assert adapter.score_pairwise_text('{"decision":"B","confidence":0.8}', "A")["correct"] is False
    failed = adapter.score_pairwise_text('{"decision":"C"}', "A")
    assert failed["decision"] == "parse_error"
    assert failed["parse_failure"]["reason"] == "schema_invalid"


def test_balanced_selection_is_deterministic():
    rows = [
        adapter.PairwiseRow(str(i), "pairwise", "q", "a", "b", label, "s", "v", "bench", "suite", str(i), {})
        for i, label in enumerate(["A", "A", "B", "B", "B"])
    ]

    selected1 = adapter.select_balanced_rows(rows, n=4, seed_key="seed")
    selected2 = adapter.select_balanced_rows(rows, n=4, seed_key="seed")

    assert [row.row_id for row in selected1] == [row.row_id for row in selected2]
    assert [row.gold_label for row in selected1].count("A") == 2
    assert [row.gold_label for row in selected1].count("B") == 2


def test_prompt_budget_truncates_and_stays_under_limit():
    row = adapter.PairwiseRow(
        "r",
        "pairwise",
        "task " * 4000,
        "candidate a " * 4000,
        "candidate b " * 4000,
        "A",
        "s",
        "v",
        "bench",
        "suite",
        "r",
        {},
    )

    fit = adapter.fit_pairwise_prompt_to_budget(
        row,
        context_length=1024,
        max_completion_tokens=64,
        prompt_context_guard_tokens=64,
        max_field_chars=20000,
        token_counter=lambda text: len(text.split()),
    )

    assert fit["prompt_token_count"] <= fit["prompt_token_max"]
    assert any(attempt["candidate_a_truncated"] for attempt in fit["prompt_fit_attempts"])
    assert "CANDIDATE A" in fit["prompt"]
    assert "CANDIDATE B" in fit["prompt"]


def test_cli_dry_run_writes_plan_and_selected_rows(tmp_path):
    data = tmp_path / "judgebench.jsonl"
    _write_jsonl(
        data,
        [
            {"pair_id": "a", "question": "q", "response_A": "a", "response_B": "b", "label": "A>B"},
            {"pair_id": "b", "question": "q", "response_A": "a", "response_B": "b", "label": "B>A"},
        ],
    )
    plan = tmp_path / "plan.json"
    rows_out = tmp_path / "rows.jsonl"

    rc = adapter.main(
        [
            "--dataset",
            "judgebench",
            "--path",
            str(data),
            "--suite",
            "gpt",
            "--n",
            "2",
            "--out-plan",
            str(plan),
            "--out-rows-jsonl",
            str(rows_out),
        ]
    )

    assert rc == 0
    plan_data = json.loads(plan.read_text())
    assert plan_data["schema"] == adapter.SCHEMA
    assert plan_data["mode"] == "dry-run"
    assert plan_data["execution_allowed"] is True
    written = [json.loads(line) for line in rows_out.read_text().splitlines()]
    assert len(written) == 2
    assert {row["gold_label"] for row in written} == {"A", "B"}
