from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import e5_w0_offline_score as scorer


class FakePrimitives:
    scorer_path = Path(__file__)
    anomaly_path = Path(__file__)
    anomaly_config_path = Path(__file__)

    @staticmethod
    def score_answer(answer, expected, method, config):
        return False

    @staticmethod
    def extract_multiple_choice_letter(text):
        return "A" if text == "A" else None

    @staticmethod
    def extract_multiple_choice_text_index(text, choices):
        return choices.index(text) if text in choices else None

    @staticmethod
    def extract_code_block(text, language):
        return "x" if text.startswith("def ") else None

    @staticmethod
    def detect_repetition_loop(text):
        return text == "loop"

    @staticmethod
    def repetition_loop_threshold():
        return 0.4


def test_parse_contract_only_removes_completed_think_blocks():
    primitives = FakePrimitives()
    assert scorer.strip_completed_think_blocks("<think>x</think> answer") == "answer"
    assert scorer.strip_completed_think_blocks("<think>cut") == "<think>cut"
    assert not scorer.parse_ok_for_response("<think>x</think>", "exact_match", {}, primitives)
    assert scorer.parse_ok_for_response("A", "multiple_choice", {}, primitives)
    assert scorer.parse_ok_for_response("second", "multiple_choice", {"choices": ["first", "second"]}, primitives)
    assert not scorer.parse_ok_for_response("unfinished prose", "multiple_choice", {}, primitives)
    assert scorer.parse_ok_for_response("def f(): pass", "code_execution", {}, primitives)
    assert not scorer.parse_ok_for_response("prose", "code_execution", {}, primitives)
    assert scorer.parse_ok_for_response("anything", "substring", {}, primitives)


def test_score_run_writes_exact_contract_and_provenance(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "selected_prompts.jsonl").write_text('{"qid":"q1","prompt":"prompt"}\n')
    (run / "cells.jsonl").write_text('{"cell_id":"cell1"}\n')
    (run / "responses.jsonl").write_text(
        '{"cell_id":"cell1","qid":"q1","http_status":200,"response_text":"loop"}\n'
    )
    pool = {"q1": [{"id": "q1", "prompt": "prompt", "expected": "x", "scoring_method": "substring", "scoring_config": {}}]}
    provenance = scorer.score_run(run, pool, FakePrimitives())
    rows = [json.loads(line) for line in (run / "offline_scores.jsonl").read_text().splitlines()]
    assert rows == [{"cell_id": "cell1", "qid": "q1", "parse_ok": True, "repetition_loop": True}]
    assert provenance["coverage"] == {"cells": 1, "qids": 1, "expected_rows": 1}


def test_score_run_fails_closed_on_missing_coverage(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "selected_prompts.jsonl").write_text('{"qid":"q1","prompt":"prompt"}\n')
    (run / "cells.jsonl").write_text('{"cell_id":"cell1"}\n')
    (run / "responses.jsonl").write_text("")
    pool = {"q1": [{"id": "q1", "prompt": "prompt", "expected": "x", "scoring_method": "substring", "scoring_config": {}}]}
    with pytest.raises(ValueError, match="coverage mismatch"):
        scorer.score_run(run, pool, FakePrimitives())


def test_score_run_fails_closed_on_non_200_and_scorer_exception(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "selected_prompts.jsonl").write_text('{"qid":"q1","prompt":"prompt"}\n')
    (run / "cells.jsonl").write_text('{"cell_id":"cell1"}\n')
    (run / "responses.jsonl").write_text(
        '{"cell_id":"cell1","qid":"q1","http_status":500,"response_text":"answer"}\n'
    )
    pool = {"q1": [{"id": "q1", "prompt": "prompt", "expected": "x", "scoring_method": "substring", "scoring_config": {}}]}
    with pytest.raises(ValueError, match="non-200"):
        scorer.score_run(run, pool, FakePrimitives())

    (run / "responses.jsonl").write_text(
        '{"cell_id":"cell1","qid":"q1","http_status":200,"response_text":"answer"}\n'
    )

    class RaisingPrimitives(FakePrimitives):
        @staticmethod
        def score_answer(answer, expected, method, config):
            raise RuntimeError("governed scorer failed")

    with pytest.raises(RuntimeError, match="governed scorer failed"):
        scorer.score_run(run, pool, RaisingPrimitives())
