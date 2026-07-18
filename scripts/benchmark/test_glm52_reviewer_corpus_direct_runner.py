#!/usr/bin/env python3
"""Inference-free tests for glm52_reviewer_corpus_direct_runner.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "glm52_reviewer_corpus_direct_runner.py"
_SPEC = importlib.util.spec_from_file_location("glm52_reviewer_corpus_direct_runner", _MODULE_PATH)
runner = importlib.util.module_from_spec(_SPEC)
sys.modules["glm52_reviewer_corpus_direct_runner"] = runner
_SPEC.loader.exec_module(runner)


def _row(row_id: str, *, label: str, candidate: str = "patch", domain: str = "code") -> dict:
    return {
        "row_id": row_id,
        "corpus_id": "nearmiss-v1",
        "domain": domain,
        "gold_label": label,
        "gold_confidence": "multi_oracle",
        "gold_source": "synthetic",
        "gold_instrument_version": "v1",
        "task": "Fix the bug.",
        "candidate": candidate,
    }


def test_iter_judgeable_rows_filters_candidate_gold_and_domain(tmp_path):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        _row("a", label="accept"),
        _row("b", label="reject"),
        _row("c", label="reject", candidate=""),
        _row("d", label="reject", domain="general"),
        {**_row("e", label="reject"), "gold_confidence": "observation"},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    got = list(
        runner.iter_judgeable_rows(
            corpus,
            domain="code",
            gold_confidence={"multi_oracle"},
        )
    )
    assert [row.row_id for row in got] == ["a", "b"]


def test_select_balanced_rows_prefers_accept_reject_balance():
    rows = [
        runner.CorpusRow("a1", _row("a1", label="accept")),
        runner.CorpusRow("a2", _row("a2", label="accept")),
        runner.CorpusRow("r1", _row("r1", label="reject")),
        runner.CorpusRow("r2", _row("r2", label="reject")),
    ]
    selected = runner.select_balanced_rows(rows, n=4, seed_key="seed")
    labels = [row.raw["gold_label"] for row in selected]
    assert labels.count("accept") == 2
    assert labels.count("reject") == 2
    assert [row.row_id for row in selected] == [
        row.row_id for row in runner.select_balanced_rows(rows, n=4, seed_key="seed")
    ]


def test_fit_prompt_to_budget_truncates_candidate_until_token_budget_fits():
    row = _row("r1", label="reject", candidate="x " * 4000)
    prompt_info = runner.fit_prompt_to_budget(
        row,
        context_length=1024,
        max_completion_tokens=128,
        prompt_context_guard_tokens=128,
        max_field_chars=5000,
        token_counter=lambda prompt: len(prompt.split()),
    )
    assert prompt_info["prompt_token_count"] <= prompt_info["prompt_token_max"]
    assert any(attempt["candidate_truncated"] for attempt in prompt_info["prompt_fit_attempts"])


def test_parse_review_decision_text_accepts_schema_valid_json():
    obj, failure = runner.parse_review_decision_text(
        '{"decision":"approve","confidence":0.91,"blocking":{"tripwire":false}}'
    )
    assert failure is None
    assert obj["decision"] == "approve"


def test_ledger_row_for_parse_error_marks_parse_error():
    row = runner.CorpusRow("r1", _row("r1", label="reject"))
    ledger = runner.ledger_row_for_result(
        row,
        result={"latency_ms": 12.0, "usage": {}, "artifacts": {"response": "resp.json"}},
        seed=42,
        rubric_version="rv",
        era="era",
    )
    assert ledger["decision"] == "parse_error"
    assert ledger["reviewer_model_quant"] == "glm_52_ud_iq2m"
    assert ledger["candidate_id"] == "r1"
    assert ledger["event_source_path"] == "resp.json"


def test_runtime_processes_excludes_current_runner(monkeypatch):
    monkeypatch.setattr(
        runner.smoke,
        "pgrep",
        lambda pattern: [
            {"pid": runner.os.getpid(), "command": "python glm52_reviewer_corpus_direct_runner.py"},
            {"pid": 12345, "command": "llama-server -m glm.gguf"},
        ],
    )
    assert runner.runtime_processes("glm52|llama-server") == [
        {"pid": 12345, "command": "llama-server -m glm.gguf"}
    ]


def test_build_plan_records_selected_rows_without_inference(tmp_path, monkeypatch):
    corpus = tmp_path / "rows.jsonl"
    rows = [_row("a", label="accept"), _row("r", label="reject")]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = runner.parse_args(
        [
            "--corpus",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--n",
            "2",
        ]
    )
    monkeypatch.setattr(runner.base, "resolve_binary", lambda path: Path("/tmp/llama-server"))
    monkeypatch.setattr(runner.base, "resolve_library_path", lambda binary, library_path: Path("/tmp"))
    monkeypatch.setattr(
        runner.base,
        "collect_inventory",
        lambda model_dir: {
            "status": "ready",
            "primary_shard": "/tmp/glm.gguf",
            "refusal_reasons": [],
        },
    )
    monkeypatch.setattr(runner, "server_extra_args", lambda: ["--reasoning", "off"])
    monkeypatch.setattr(runner.smoke, "pgrep", lambda pattern: [])
    plan = runner.build_plan(args)
    assert plan["execution_allowed"] is True
    assert plan["corpus"]["n_selected"] == 2
    assert set(plan["corpus"]["selected_label_counts"]) == {"accept", "reject"}
    assert plan["request"]["rubric_version"] == runner.DEFAULT_RUBRIC_VERSION
