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
        "source_benchmark": "seeded-mutation",
        "source_suite": "debugbench",
        "provenance": {"scoring_method": "substring"},
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


def test_iter_judgeable_rows_filters_source_representation(tmp_path):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        _row("dbg", label="accept"),
        {**_row("crux", label="accept"), "source_suite": "cruxeval", "provenance": {"scoring_method": "exact_match"}},
        {**_row("ccrab", label="reject"), "source_benchmark": "c-crab", "source_suite": "python", "provenance": {}},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    got = list(
        runner.iter_judgeable_rows(
            corpus,
            domain="code",
            gold_confidence={"multi_oracle"},
            source_suites={"cruxeval"},
            source_benchmarks={"seeded-mutation"},
            scoring_methods={"exact_match"},
        )
    )
    assert [row.row_id for row in got] == ["crux"]


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


def test_summarize_row_set_records_representation_counts():
    rows = [
        runner.CorpusRow("a", _row("a", label="accept", candidate="abc")),
        runner.CorpusRow(
            "r",
            {**_row("r", label="reject", candidate="abcd"), "source_suite": "cruxeval", "provenance": {"scoring_method": "exact_match"}},
        ),
    ]
    summary = runner.summarize_row_set(rows)
    assert summary["label_counts"] == {"accept": 1, "reject": 1}
    assert summary["representation_counts"] == {
        "seeded-mutation|debugbench|substring": 1,
        "seeded-mutation|cruxeval|exact_match": 1,
    }
    assert summary["candidate_payload_scope_counts"] == {"answer_fragment": 2}
    assert summary["candidate_chars"] == {"min": 3, "p50": 4, "max": 4}


def test_candidate_payload_scope_marks_non_scorer_rows_as_full_candidate():
    row = {**_row("patch", label="reject"), "provenance": {"candidate_is": "patch_to_review"}}
    assert runner.candidate_payload_scope(row) == "full_candidate"


def test_answer_fragment_refusal_requires_explicit_override():
    rows = [runner.CorpusRow("a", _row("a", label="accept", candidate="abc"))]
    reasons = runner.answer_fragment_refusal_reasons(rows, allow_answer_fragment_review=False)
    assert reasons
    assert "answer-fragment" in reasons[0]
    assert runner.answer_fragment_refusal_reasons(rows, allow_answer_fragment_review=True) == []


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


def test_fit_prompt_to_budget_keeps_long_patch_when_budget_fits():
    candidate = "x " * 8000
    row = _row("r1", label="accept", candidate=candidate)
    prompt_info = runner.fit_prompt_to_budget(
        row,
        context_length=12000,
        max_completion_tokens=256,
        prompt_context_guard_tokens=256,
        max_field_chars=runner.DEFAULT_MAX_FIELD_CHARS,
        token_counter=lambda prompt: len(prompt.split()),
    )
    assert prompt_info["prompt_token_count"] <= prompt_info["prompt_token_max"]
    assert prompt_info["truncation"]["candidate_truncated"] is False
    assert candidate in prompt_info["prompt"]


def test_parse_review_decision_text_accepts_schema_valid_json():
    obj, failure = runner.parse_review_decision_text(
        '{"decision":"approve","confidence":0.91,"blocking":{"tripwire":false},'
        '"evidence":{"basis":"diff implements the requested behavior","risk":"no blocker found"}}'
    )
    assert failure is None
    assert obj["decision"] == "approve"


def test_binary_schema_allows_only_approve_reject():
    schema = runner.binary_review_decision_response_schema()
    assert schema["properties"]["decision"]["enum"] == ["approve", "reject"]
    assert "evidence" in schema["required"]
    assert schema["properties"]["evidence"]["required"] == ["basis", "risk"]
    assert schema["properties"]["evidence"]["properties"]["basis"]["maxLength"] == 180


def test_parse_review_decision_text_rejects_shared_review_actions():
    obj, failure = runner.parse_review_decision_text(
        '{"decision":"request_changes","confidence":0.91,"blocking":{"tripwire":true},'
        '"evidence":{"basis":"bad hunk","risk":"wrong field"}}'
    )
    assert obj is None
    assert failure is not None
    assert failure["reason"] == "schema_invalid"


def test_parse_review_decision_text_requires_evidence():
    obj, failure = runner.parse_review_decision_text(
        '{"decision":"approve","confidence":0.91,"blocking":{"tripwire":false}}'
    )
    assert obj is None
    assert failure is not None
    assert "$.evidence" in "\n".join(failure["errors"])


def test_generic_prompt_header_is_task_grounded_not_strict_by_default():
    assert "strict reviewer" not in runner.GENERIC_PROMPT_HEADER.lower()
    assert "speculative concerns" in runner.GENERIC_PROMPT_HEADER
    assert "substantially satisfies" in runner.GENERIC_PROMPT_HEADER


def test_patch_diff_prompt_requires_negative_evidence_scrutiny():
    row = {
        **_row("ccrab", label="reject", candidate="diff --git a/x b/x\n+broken"),
        "source_benchmark": "c-crab",
        "source_suite": "python",
        "provenance": {"candidate_is": "patch_to_review"},
    }
    prompt, meta = runner.build_review_prompt(row, max_field_chars=1000)
    assert meta["review_mode"] == "patch_diff_strict"
    assert "Start from reject" in prompt
    assert "changed test/assertion that would fail without the fix" in prompt
    assert "nearby/pass-only behavior" in prompt
    assert "helper/API changes are not tied to the rule path" in prompt
    assert "misspelled or likely undefined identifiers" in prompt
    assert "evidence.basis" in prompt
    assert "under 20 words each" in prompt


def test_build_review_prompt_includes_oracle_note_only_when_present():
    row = _row("ordinary", label="reject")
    plain_prompt, plain_meta = runner.build_review_prompt(row, max_field_chars=1000)
    noted_prompt, noted_meta = runner.build_review_prompt(
        {**row, "oracle_note": "The added test must reproduce the reported dbt raw-space failure."},
        max_field_chars=1000,
    )

    assert "CURATED REVIEW CONSTRAINT" not in plain_prompt
    assert plain_meta["oracle_note_present"] is False
    assert "CURATED REVIEW CONSTRAINT" in noted_prompt
    assert "dbt raw-space failure" in noted_prompt
    assert noted_meta["oracle_note_present"] is True


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
    rows = [
        {**_row("a", label="accept"), "provenance": {}},
        {**_row("r", label="reject"), "provenance": {}},
    ]
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
    assert plan["corpus"]["selection_mode"] == "balanced_seeded"


def test_read_row_ids_file_ignores_blank_lines_and_comments(tmp_path):
    row_ids_file = tmp_path / "rows.txt"
    row_ids_file.write_text(
        "\n"
        "# reviewed controls\n"
        "row-a\n"
        "row-b  # keep this one\n"
        "   \n"
        "row-a\n",
        encoding="utf-8",
    )

    assert runner.read_row_ids_file(row_ids_file) == ["row-a", "row-b", "row-a"]


def test_read_oracle_notes_file_requires_row_id_mapping(tmp_path):
    notes_file = tmp_path / "notes.json"
    notes_file.write_text(json.dumps({"row-a": "Check the exact failing path."}), encoding="utf-8")

    assert runner.read_oracle_notes_file(notes_file) == {"row-a": "Check the exact failing path."}


def test_load_oracle_notes_refuses_conflicting_duplicates(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"row-a": "first"}), encoding="utf-8")
    second.write_text(json.dumps({"row-a": "second"}), encoding="utf-8")

    try:
        runner.load_oracle_notes([first, second])
    except ValueError as exc:
        assert "conflicting oracle notes" in str(exc)
    else:
        raise AssertionError("expected conflicting oracle notes to fail")


def test_requested_row_ids_dedupes_files_and_cli(tmp_path):
    row_ids_file = tmp_path / "rows.txt"
    row_ids_file.write_text("row-a\nrow-b\n", encoding="utf-8")
    args = runner.parse_args(
        [
            "--row-ids-file",
            str(row_ids_file),
            "--row-id",
            "row-b",
            "--row-id",
            "row-c",
        ]
    )

    assert runner.requested_row_ids(args) == ["row-a", "row-b", "row-c"]


def test_build_plan_uses_explicit_row_ids_in_order(tmp_path, monkeypatch):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        {**_row("accept-a", label="accept"), "provenance": {}},
        {**_row("reject-a", label="reject"), "provenance": {}},
        {**_row("accept-b", label="accept"), "provenance": {}},
        {**_row("reject-b", label="reject"), "provenance": {}},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = runner.parse_args(
        [
            "--corpus",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--n",
            "99",
            "--row-id",
            "reject-b",
            "--row-id",
            "accept-a",
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
    assert plan["corpus"]["selection_mode"] == "explicit_row_ids"
    assert plan["corpus"]["n_requested"] == 2
    assert plan["corpus"]["selected_row_ids"] == ["reject-b", "accept-a"]
    assert plan["corpus"]["explicit_row_ids"] == ["reject-b", "accept-a"]


def test_build_plan_records_selected_oracle_notes(tmp_path, monkeypatch):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        {**_row("accept-a", label="accept"), "provenance": {}},
        {**_row("reject-a", label="reject"), "provenance": {}},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    notes_file = tmp_path / "notes.json"
    notes_file.write_text(json.dumps({"reject-a": "Check the exact patched failing path."}), encoding="utf-8")
    args = runner.parse_args(
        [
            "--corpus",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--row-id",
            "reject-a",
            "--row-id",
            "accept-a",
            "--oracle-notes-file",
            str(notes_file),
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
    assert plan["review_hints"]["selected_oracle_note_row_ids"] == ["reject-a"]
    assert plan["review_hints"]["oracle_notes_by_row_id"] == {
        "reject-a": "Check the exact patched failing path."
    }


def test_build_plan_refuses_missing_explicit_row_id(tmp_path, monkeypatch):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        {**_row("accept-a", label="accept"), "provenance": {}},
        {**_row("reject-a", label="reject"), "provenance": {}},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = runner.parse_args(
        [
            "--corpus",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--row-id",
            "accept-a",
            "--row-id",
            "missing-row",
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

    assert plan["execution_allowed"] is False
    assert plan["corpus"]["missing_explicit_row_ids"] == ["missing-row"]
    assert any("explicit row ids not found" in reason for reason in plan["refusal_reasons"])


def test_build_plan_refuses_mixed_representation_by_default(tmp_path, monkeypatch):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        _row("a", label="accept"),
        {**_row("r", label="reject"), "source_suite": "cruxeval", "provenance": {"scoring_method": "exact_match"}},
    ]
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
    assert plan["execution_allowed"] is False
    assert any("mix source_suite/scoring" in reason for reason in plan["refusal_reasons"])


def test_build_plan_refuses_answer_fragment_without_override(tmp_path, monkeypatch):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        {**_row("a", label="accept"), "source_suite": "cruxeval", "provenance": {"scoring_method": "exact_match"}},
        {**_row("r", label="reject"), "source_suite": "cruxeval", "provenance": {"scoring_method": "exact_match"}},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = runner.parse_args(
        [
            "--corpus",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--n",
            "2",
            "--source-suite",
            "cruxeval",
            "--provenance-scoring-method",
            "exact_match",
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
    assert plan["execution_allowed"] is False
    assert any("answer-fragment" in reason for reason in plan["refusal_reasons"])


def test_build_plan_allows_explicit_representation_filter_with_fragment_override(tmp_path, monkeypatch):
    corpus = tmp_path / "rows.jsonl"
    rows = [
        {**_row("a", label="accept"), "source_suite": "cruxeval", "provenance": {"scoring_method": "exact_match"}},
        {**_row("r", label="reject"), "source_suite": "cruxeval", "provenance": {"scoring_method": "exact_match"}},
        _row("other", label="reject"),
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = runner.parse_args(
        [
            "--corpus",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--n",
            "2",
            "--source-suite",
            "cruxeval",
            "--provenance-scoring-method",
            "exact_match",
            "--allow-answer-fragment-review",
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
    assert plan["corpus"]["selected_summary"]["representation_counts"] == {
        "seeded-mutation|cruxeval|exact_match": 2
    }
