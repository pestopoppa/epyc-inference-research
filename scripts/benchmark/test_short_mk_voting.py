#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import short_mk_voting as smk


def test_extract_vote_key_multiple_choice_takes_last_explicit_answer():
    question = {"scoring_method": "multiple_choice"}
    answer = "I considered B.\nAnswer: C"
    assert smk.extract_vote_key(answer, question) == "C"


def test_extract_vote_key_multiple_choice_uses_canonical_final_line_fallback():
    question = {"scoring_method": "multiple_choice"}
    answer = "I considered B while reasoning.\nC"
    assert smk.extract_vote_key(answer, question) == "C"


def test_extract_vote_key_exact_match_uses_boxed_answer():
    question = {"scoring_method": "exact_match", "scoring_config": {}}
    assert smk.extract_vote_key("Therefore \\boxed{42}.", question) == "42"


def test_majority_vote_prefers_majority_answer():
    question = {"expected": "C", "scoring_method": "multiple_choice", "scoring_config": {}}
    completions = [
        smk.Completion(index=0, text="Answer: B", completion_tokens=6, elapsed_seconds=1.0),
        smk.Completion(index=1, text="Answer: C", completion_tokens=8, elapsed_seconds=1.1),
        smk.Completion(index=2, text="Answer: C", completion_tokens=7, elapsed_seconds=1.2),
    ]
    vote = smk.majority_vote(question, completions)
    assert vote.vote_key == "C"
    assert vote.count == 2
    assert vote.correct is True


def test_majority_vote_tie_breaks_on_shortest_completion():
    question = {"expected": "A", "scoring_method": "multiple_choice", "scoring_config": {}}
    completions = [
        smk.Completion(index=0, text="Answer: B", completion_tokens=20, elapsed_seconds=1.0),
        smk.Completion(index=1, text="Answer: A", completion_tokens=5, elapsed_seconds=1.1),
    ]
    vote = smk.majority_vote(question, completions)
    assert vote.vote_key == "A"
    assert vote.completion_tokens == 5


def test_build_questions_samples_from_pool(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    header = {"__pool_metadata__": True, "generated_at": "2026-06-19T00:00:00+00:00"}
    rows = [
        {"id": "gpqa-1", "suite": "gpqa", "prompt": "p1", "expected": "A", "scoring_method": "multiple_choice"},
        {"id": "math-1", "suite": "math", "prompt": "p2", "expected": "2", "scoring_method": "exact_match"},
    ]
    pool_path.write_text("\n".join(json.dumps(item) for item in [header, *rows]) + "\n")

    questions = smk.build_questions(pool_path, ["gpqa", "math"], sample_per_suite=1, seed=1)
    assert [item["suite"] for item in questions] == ["gpqa", "math"]


def test_main_dry_run_writes_sample_plan(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    output = tmp_path / "dry-run.json"
    header = {"__pool_metadata__": True, "generated_at": "2026-06-19T00:00:00+00:00"}
    row = {"id": "gpqa-1", "suite": "gpqa", "prompt": "p1", "expected": "A", "scoring_method": "multiple_choice"}
    pool_path.write_text(json.dumps(header) + "\n" + json.dumps(row) + "\n")

    rc = smk.main([
        "--suites",
        "gpqa",
        "--sample-per-suite",
        "1",
        "--pool",
        str(pool_path),
        "--model-port",
        "8070",
        "--output",
        str(output),
        "--dry-run",
    ])
    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "dry_run"
    assert payload["question_count"] == 1
