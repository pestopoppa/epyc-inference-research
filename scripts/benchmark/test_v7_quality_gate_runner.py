#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import v7_quality_gate_runner as runner


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_query_server_defaults_to_chat_completions(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode())
        seen["timeout"] = timeout
        return _FakeResponse({"choices": [{"message": {"content": "C"}}]})

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    got = runner.query_server("http://127.0.0.1:18072", "prompt")

    assert got == "C"
    assert seen["url"] == "http://127.0.0.1:18072/v1/chat/completions"
    assert seen["payload"]["messages"] == [{"role": "user", "content": "prompt"}]
    assert seen["payload"]["stream"] is False
    assert seen["timeout"] == runner.REQUEST_TIMEOUT_S


def test_query_server_keeps_completion_mode(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):  # noqa: ANN001, ARG001
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode())
        return _FakeResponse({"choices": [{"text": "D"}]})

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    got = runner.query_server(
        "http://127.0.0.1:18072",
        "prompt",
        endpoint="completion",
    )

    assert got == "D"
    assert seen["url"] == "http://127.0.0.1:18072/v1/completions"
    assert seen["payload"]["prompt"] == "prompt"
    assert seen["payload"]["logprobs"] == 0


def test_extract_letter_answer_prefers_explicit_answer():
    assert runner.extract_letter_answer("I think the answer is C.") == "C"
    assert runner.extract_letter_answer("C.") == "C"
    assert runner.extract_letter_answer("I think C is likely") == ""


def test_text_fingerprint_reports_utf8_bytes_not_only_characters():
    text = "caf\u00e9"
    assert runner.text_fingerprint(text) == {
        "chars": 4,
        "utf8_bytes": 5,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def test_extract_letter_answer_verbose_bare_letter_no_penalty():
    """Regression: a verbose arm that reasons then puts a bare letter on the
    final line HAS answered. The 2026-07 architect-bench artifact was that
    A4 (verbose) leaked 15% of gpqa to false parse-failures on exactly this
    shape while terse A1 scored 0% failures -- a systematic bias against
    models that show their work. Must parse; must NOT bias by verbosity."""
    verbose = (
        "Let me work through the equilibrium. [PO4^3-] approx 6.2e-7 M.\n"
        "This matches option D.\n\nD"
    )
    assert runner.extract_letter_answer(verbose) == "D"
    # bold / parenthesised final-line variants a CoT model emits
    assert runner.extract_letter_answer("...so the product is the ether.\n\n**B**") == "B"
    assert runner.extract_letter_answer("reasoning...\n(A)") == "A"
    # a genuinely truncated derivation (no answer) must still fail to parse,
    # so truncations are not silently credited.
    assert runner.extract_letter_answer(
        "Step 1: balance the redox couple. Step 2: the half reaction for"
    ) == ""


def test_score_response_handles_aime_numeric_exact_match():
    q = {
        "scoring_method": "exact_match",
        "scoring_config": {
            "extract_pattern": r"(\d+)\s*$",
            "normalize_numeric": True,
        },
    }

    assert runner.score_response("After solving, the final answer is 070", "70", q)
    assert runner.score_response("Final answer: 7", "007", q)
    assert not runner.score_response("Final answer: 71", "70", q)


def test_score_response_keeps_multiple_choice_path():
    q = {"scoring_method": "multiple_choice", "scoring_config": {}}

    assert runner.score_response("The answer is D.", "D", q)
    assert not runner.score_response("The answer is C.", "D", q)


def test_run_suite_does_not_fold_or_resume_other_suite_rows(tmp_path, monkeypatch):
    """A shared JSONL must not let MMLU draws satisfy GPQA work."""
    questions = [{"id": "shared-id", "prompt": "prompt", "expected": "C", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    calls = []

    def fake_query(*args, **kwargs):  # noqa: ANN001
        calls.append((args, kwargs))
        return {
            "text": "C", "reasoning": "", "finish_reason": "stop",
            "completion_tokens": 1, "prompt_tokens": 1,
            "decode_tok_s": 1.0, "error": "",
        }

    monkeypatch.setattr(runner, "query_server_meta", fake_query)
    rows = tmp_path / "shared.jsonl"
    rows.write_text(json.dumps({
        "suite": "mmlu_pro", "id": "shared-id", "seed": 42,
        "tier": "1", "correct": True,
    }) + "\n")

    with rows.open("a") as handle:
        result = runner.run_suite(
            "gpqa", "http://unused", n=1, seed=42, per_question_out=handle,
        )

    assert len(calls) == 1
    assert result["n"] == 1
    assert result["correct"] == 1
    assert result["n_questions"] == 1
    assert [json.loads(line)["suite"] for line in rows.read_text().splitlines()] == [
        "mmlu_pro", "gpqa",
    ]


def test_run_suite_same_suite_resume_is_idempotent(tmp_path, monkeypatch):
    questions = [{"id": "q1", "prompt": "prompt", "expected": "C", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)

    def should_not_query(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("a completed same-suite draw must not be re-queried")

    monkeypatch.setattr(runner, "query_server_meta", should_not_query)
    rows = tmp_path / "same-suite.jsonl"
    row = {
        "suite": "gpqa", "id": "q1", "seed": 42, "tier": "1",
        "correct": True, "empty_response": False, "truncated": False,
        "capture_schema_version": runner.CAPTURE_SCHEMA_VERSION,
        "runner_source_sha256": runner.RUNNER_SOURCE_SHA256,
        "prompt": "prompt", "expected": "C", "response": "C", "reasoning": "",
    }
    row["prompt_fingerprint"] = runner.text_fingerprint(row["prompt"])
    row["response_fingerprint"] = runner.text_fingerprint(row["response"])
    row["reasoning_fingerprint"] = runner.text_fingerprint(row["reasoning"])
    rows.write_text(json.dumps(row) + "\n")

    with rows.open("a") as handle:
        result = runner.run_suite(
            "gpqa", "http://unused", n=1, seed=42, per_question_out=handle,
        )

    assert result["n"] == 1
    assert result["correct"] == 1
    assert result["n_questions"] == 1
    assert rows.read_text().splitlines() == [json.dumps(row)]


def test_run_suite_persists_and_scores_full_search_replace_response(tmp_path, monkeypatch):
    response = "<<<<<<< SEARCH\n" + ("x" * 5000) + "\n=======\ny\n>>>>>>> REPLACE file.py"
    questions = [{"id": "swe-1", "prompt": "prompt", "expected": "unused", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    seen = []
    monkeypatch.setattr(runner, "score_response", lambda text, *_: seen.append(text) or True)
    monkeypatch.setattr(
        runner,
        "query_server_meta",
        lambda *args, **kwargs: {
            "text": response, "reasoning": "", "finish_reason": "stop",
            "completion_tokens": 1, "prompt_tokens": 1, "decode_tok_s": 1.0, "error": "",
        },
    )
    rows = tmp_path / "swe.jsonl"
    with rows.open("a") as handle:
        runner.run_suite("swebench_oracle", "http://unused", n=1, seed=42, per_question_out=handle)

    assert seen == [response]
    row = json.loads(rows.read_text())
    assert row["response"] == response
    assert row["response_fingerprint"] == {
        "chars": len(response),
        "utf8_bytes": len(response.encode("utf-8")),
        "sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
    }
    assert row["prompt"] == "prompt"
    assert row["prompt_fingerprint"] == runner.text_fingerprint("prompt")


def test_swe_capture_never_tail_slices_a_valid_early_block(tmp_path, monkeypatch):
    """Regression for the historical ``response[-4000:]`` artifact loss.

    The only valid block appears before more than 4,000 trailing characters.
    A tail slice would silently erase the block before conversion.
    """
    block = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE file.py"
    response = block + "\n" + ("trailing analysis " * 400)
    assert len(response) > 4_000
    questions = [{"id": "early-block", "prompt": "prompt", "expected": "unused", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "score_response", lambda *_: False)
    monkeypatch.setattr(
        runner,
        "query_server_meta",
        lambda *args, **kwargs: {
            "text": response, "reasoning": "", "finish_reason": "stop",
            "completion_tokens": 10, "prompt_tokens": 3, "decode_tok_s": 1.0, "error": "",
        },
    )
    rows = tmp_path / "early-block.jsonl"
    with rows.open("a") as handle:
        result = runner.run_suite("swebench_oracle", "http://unused", n=1, seed=42,
                                  per_question_out=handle)

    row = json.loads(rows.read_text())
    assert row["response"] == response
    assert row["response"].startswith(block)
    assert row["swe_search_replace"]["parseable_block_count"] == 1
    assert row["capture_schema_version"] == runner.CAPTURE_SCHEMA_VERSION
    assert row["runner_source_sha256"] == runner.RUNNER_SOURCE_SHA256
    assert result["capture"]["swebench_search_replace"]["score_status"] == (
        "terminal_no_prompt_contract_candidate"
    )


def test_swe_capture_persists_reasoning_and_strict_contract_diagnostics(tmp_path, monkeypatch):
    response = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE file.py"
    reasoning = "Need preserve this reasoning payload: cafe"
    questions = [{"id": "swe-1", "prompt": "prompt", "expected": "unused", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "score_response", lambda *_: False)
    monkeypatch.setattr(
        runner,
        "query_server_meta",
        lambda *args, **kwargs: {
            "text": response, "reasoning": reasoning, "finish_reason": "stop",
            "completion_tokens": 2, "prompt_tokens": 3, "decode_tok_s": 1.0, "error": "",
        },
    )
    rows = tmp_path / "swe.jsonl"
    with rows.open("a") as handle:
        result = runner.run_suite("swebench_oracle", "http://unused", n=1, seed=42,
                                  per_question_out=handle)

    row = json.loads(rows.read_text())
    assert row["reasoning"] == reasoning
    assert row["reasoning_fingerprint"] == runner.text_fingerprint(reasoning)
    assert row["swe_search_replace"] == {
        "marker_counts": {"search": 1, "divider": 1, "replace": 1},
        "parseable_block_count": 1,
        "has_markers": True,
        "malformed_contract": False,
        "state": "strict_ready",
        "converter_ready": True,
        "score_provisional": False,
    }
    summary = result["capture"]["swebench_search_replace"]
    assert summary["new_rows"] == 1
    assert summary["state_counts"] == {
        "strict_ready": 1,
        "prompt_contract_candidate": 0,
        "model_truncation_no_patch": 0,
        "model_truncation_partial_patch": 0,
        "request_error": 0,
    }
    assert summary["converter_contract_ready"] is True
    assert summary["score_status"] == "terminal_no_prompt_contract_candidate"


def test_swe_capture_warns_and_marks_malformed_contract(tmp_path, monkeypatch, capsys):
    response = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> file.py"
    questions = [{"id": "swe-malformed", "prompt": "prompt", "expected": "unused", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "score_response", lambda *_: False)
    monkeypatch.setattr(
        runner,
        "query_server_meta",
        lambda *args, **kwargs: {
            "text": response, "reasoning": "", "finish_reason": "stop",
            "completion_tokens": 2, "prompt_tokens": 3, "decode_tok_s": 1.0, "error": "",
        },
    )
    rows = tmp_path / "swe-malformed.jsonl"
    with rows.open("a") as handle:
        result = runner.run_suite("swebench_oracle", "http://unused", n=1, seed=42,
                                  per_question_out=handle)

    row = json.loads(rows.read_text())
    assert row["swe_search_replace"]["parseable_block_count"] == 0
    assert row["swe_search_replace"]["malformed_contract"] is True
    assert row["swe_search_replace"]["state"] == "prompt_contract_candidate"
    assert result["capture"]["swebench_search_replace"]["converter_contract_ready"] is False
    assert result["capture"]["swebench_search_replace"]["score_status"] == "provisional_prompt_contract"
    assert "WARNING swebench_oracle capture not converter-ready id=swe-malformed" in capsys.readouterr().err


def test_swe_zero_marker_stop_and_length_are_not_converter_ready(tmp_path, monkeypatch, capsys):
    questions = [
        {"id": "stop", "prompt": "stop", "expected": "unused", "tier": 1},
        {"id": "length", "prompt": "length", "expected": "unused", "tier": 1},
    ]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "score_response", lambda *_: False)
    replies = iter([
        {"text": "analysis only", "reasoning": "", "finish_reason": "stop",
         "completion_tokens": 2, "prompt_tokens": 3, "decode_tok_s": 1.0, "error": ""},
        {"text": "still analyzing", "reasoning": "", "finish_reason": "length",
         "completion_tokens": 3072, "prompt_tokens": 3, "decode_tok_s": 1.0, "error": ""},
    ])
    monkeypatch.setattr(runner, "query_server_meta", lambda *args, **kwargs: next(replies))
    rows = tmp_path / "swe-zero-marker.jsonl"
    with rows.open("a") as handle:
        result = runner.run_suite("swebench_oracle", "http://unused", n=2, seed=42,
                                  per_question_out=handle)

    written = [json.loads(line) for line in rows.read_text().splitlines()]
    assert [row["swe_search_replace"]["state"] for row in written] == [
        "prompt_contract_candidate", "model_truncation_no_patch",
    ]
    summary = result["capture"]["swebench_search_replace"]
    assert summary["converter_contract_ready"] is False
    assert summary["score_status"] == "provisional_prompt_contract"
    warnings = capsys.readouterr().err
    assert "id=stop state=prompt_contract_candidate anomalies=zero_strict_blocks" in warnings
    assert "id=length state=model_truncation_no_patch anomalies=length_cap,zero_strict_blocks" in warnings


def test_swe_request_error_keeps_summary_provisional(tmp_path, monkeypatch, capsys):
    questions = [{"id": "request-error", "prompt": "prompt", "expected": "unused", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "score_response", lambda *_: False)
    monkeypatch.setattr(
        runner,
        "query_server_meta",
        lambda *args, **kwargs: {
            "text": "", "reasoning": "", "finish_reason": "request_error",
            "completion_tokens": 0, "prompt_tokens": 0, "decode_tok_s": 0.0,
            "error": "socket closed",
        },
    )
    rows = tmp_path / "request-error.jsonl"
    with rows.open("a") as handle:
        result = runner.run_suite("swebench_oracle", "http://unused", n=1, seed=42,
                                  per_question_out=handle)

    row = json.loads(rows.read_text())
    assert row["swe_search_replace"]["state"] == "request_error"
    assert row["swe_search_replace"]["score_provisional"] is True
    summary = result["capture"]["swebench_search_replace"]
    assert summary["state_counts"]["request_error"] == 1
    assert summary["score_status"] == "provisional_request_error"
    assert "anomalies=request_error,zero_strict_blocks" in capsys.readouterr().err


def test_live_capture_status_is_atomic_and_fail_closed_only_for_transport(tmp_path, monkeypatch):
    questions = [
        {"id": "length", "prompt": "length", "expected": "unused", "tier": 1},
        {"id": "error", "prompt": "error", "expected": "unused", "tier": 1},
    ]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "score_response", lambda *_: False)
    replies = iter([
        {"text": "analysis", "reasoning": "", "finish_reason": "length",
         "completion_tokens": 3072, "prompt_tokens": 3, "decode_tok_s": 1.0, "error": ""},
        {"text": "", "reasoning": "", "finish_reason": "request_error",
         "completion_tokens": 0, "prompt_tokens": 0, "decode_tok_s": 0.0, "error": "socket"},
    ])
    monkeypatch.setattr(runner, "query_server_meta", lambda *args, **kwargs: next(replies))
    rows = tmp_path / "pq.jsonl"
    status = tmp_path / "pq.live-status.json"
    with rows.open("a") as handle:
        runner.run_suite(
            "swebench_oracle", "http://unused", n=2, seed=42,
            per_question_out=handle, live_status_out=status,
        )

    published = json.loads(status.read_text())
    assert published["completed_draws"] == 2
    assert published["expected_draws"] == 2
    assert published["complete"] is True
    assert published["length_cap_rows"] == 1
    assert published["request_error_rows"] == 1
    assert published["swebench_search_replace"]["zero_strict_block_rows"] == 2
    assert published["swebench_search_replace"]["partial_strict_block_rows"] == 0
    assert published["provisional"] is True
    assert published["artifact_integrity_fail_closed"] is True
    assert not list(tmp_path.glob(".pq.live-status.json.*.tmp"))


def test_live_capture_status_keeps_length_only_model_outcome_terminal(tmp_path, monkeypatch):
    questions = [{"id": "length", "prompt": "length", "expected": "unused", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "score_response", lambda *_: False)
    monkeypatch.setattr(
        runner,
        "query_server_meta",
        lambda *args, **kwargs: {
            "text": "unfinished", "reasoning": "", "finish_reason": "length",
            "completion_tokens": 3072, "prompt_tokens": 3, "decode_tok_s": 1.0, "error": "",
        },
    )
    rows = tmp_path / "length.jsonl"
    status = tmp_path / "length.live-status.json"
    with rows.open("a") as handle:
        runner.run_suite(
            "swebench_oracle", "http://unused", n=1, seed=42,
            per_question_out=handle, live_status_out=status,
        )

    published = json.loads(status.read_text())
    assert published["length_cap_rows"] == 1
    assert published["request_error_rows"] == 0
    assert published["provisional"] is False
    assert published["artifact_integrity_fail_closed"] is False
    assert published["swebench_search_replace"]["score_status"] == (
        "terminal_no_prompt_contract_candidate"
    )


def test_swe_resume_requeries_legacy_v3_rows_without_full_prompt(tmp_path, monkeypatch):
    questions = [{"id": "legacy", "prompt": "prompt", "expected": "unused", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "query_server_meta", lambda *args, **kwargs: {
        "text": "fresh", "reasoning": "", "finish_reason": "stop",
        "completion_tokens": 1, "prompt_tokens": 1, "decode_tok_s": 1.0, "error": "",
    })
    rows = tmp_path / "swe-legacy.jsonl"
    rows.write_text(json.dumps({
        "suite": "swebench_oracle", "id": "legacy", "seed": 42, "tier": "1",
        "correct": False, "empty_response": False, "truncated": False,
        "finish_reason": "stop", "request_error": "", "response": "no patch here",
    }) + "\n")
    with rows.open("a") as handle:
        result = runner.run_suite("swebench_oracle", "http://unused", n=1, seed=42,
                                  per_question_out=handle)

    summary = result["capture"]["swebench_search_replace"]
    assert summary["resumed_rows"] == 0
    assert summary["state_counts"]["prompt_contract_candidate"] == 1
    assert summary["summary_complete"] is True
    assert summary["converter_contract_ready"] is False
    assert summary["score_status"] == "provisional_prompt_contract"


def test_swe_resume_requeries_source_or_fingerprint_mismatch(tmp_path, monkeypatch):
    questions = [{"id": "provenance", "prompt": "prompt", "expected": "unused", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "query_server_meta", lambda *args, **kwargs: {
        "text": "fresh", "reasoning": "", "finish_reason": "stop",
        "completion_tokens": 1, "prompt_tokens": 1, "decode_tok_s": 1.0, "error": "",
    })
    rows = tmp_path / "source-mismatch.jsonl"
    rows.write_text(json.dumps({
        "suite": "swebench_oracle", "id": "provenance", "seed": 42, "tier": "1",
        "correct": False, "empty_response": False, "truncated": False,
        "finish_reason": "stop", "request_error": "", "response": "",
        "capture_schema_version": runner.CAPTURE_SCHEMA_VERSION,
        "runner_source_sha256": "not-the-current-runner",
        "swe_search_replace": {
            "has_markers": True, "parseable_block_count": 1, "malformed_contract": False,
            "state": "strict_ready",
        },
    }) + "\n")
    with rows.open("a") as handle:
        result = runner.run_suite("swebench_oracle", "http://unused", n=1, seed=42,
                                  per_question_out=handle)

    summary = result["capture"]["swebench_search_replace"]
    assert summary["resumed_rows"] == 0
    assert summary["summary_complete"] is True


def test_resume_quarantines_tampered_row_before_writing_one_fresh_draw(tmp_path, monkeypatch):
    questions = [{"id": "q1", "prompt": "current prompt", "expected": "C", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "query_server_meta", lambda *args, **kwargs: {
        "text": "C", "reasoning": "", "finish_reason": "stop",
        "completion_tokens": 1, "prompt_tokens": 1, "decode_tok_s": 1.0, "error": "",
    })
    rows = tmp_path / "capture.jsonl"
    rows.write_text(json.dumps({
        "suite": "gpqa", "id": "q1", "seed": 42, "expected": "C",
        "capture_schema_version": "v7_quality_gate_capture.v3", "prompt": "old prompt",
        "response": "C", "reasoning": "",
    }) + "\n")

    with rows.open("a") as handle:
        runner.run_suite("gpqa", "http://unused", n=1, seed=42, per_question_out=handle)

    retained = [json.loads(line) for line in rows.read_text().splitlines()]
    assert len(retained) == 1
    assert retained[0]["capture_schema_version"] == runner.CAPTURE_SCHEMA_VERSION
    rejected = [json.loads(line) for line in Path(f"{rows}.rejected.jsonl").read_text().splitlines()]
    assert rejected[0]["reason"] == "resume_validation_failed"
    assert rejected[0]["row"]["prompt"] == "old prompt"


def test_atomic_compaction_rebinds_actual_append_handle_and_quarantines_malformed(tmp_path):
    capture = tmp_path / "capture.jsonl"
    capture.write_text("keep\n{malformed\n")
    with capture.open("a") as handle:
        runner.replace_capture_contents(handle, ["keep"])
        handle.write("fresh\n")
        handle.flush()

    assert capture.read_text() == "keep\nfresh\n"


def test_resume_requeries_transport_failure(tmp_path, monkeypatch):
    questions = [{"id": "q1", "prompt": "prompt", "expected": "C", "tier": 1}]
    monkeypatch.setattr(runner, "load_questions", lambda *args, **kwargs: questions)
    monkeypatch.setattr(runner, "query_server_meta", lambda *args, **kwargs: {
        "text": "C", "reasoning": "", "finish_reason": "stop",
        "completion_tokens": 1, "prompt_tokens": 1, "decode_tok_s": 1.0, "error": "",
    })
    row = {
        "suite": "gpqa", "id": "q1", "seed": 42, "expected": "C", "tier": "1",
        "capture_schema_version": runner.CAPTURE_SCHEMA_VERSION,
        "runner_source_sha256": runner.RUNNER_SOURCE_SHA256,
        "prompt": "prompt", "response": "", "reasoning": "",
        "finish_reason": "request_error", "request_error": "timeout",
    }
    for field in ("prompt", "response", "reasoning"):
        row[f"{field}_fingerprint"] = runner.text_fingerprint(row[field])
    capture = tmp_path / "capture.jsonl"
    capture.write_text(json.dumps(row) + "\n{not-json}\n")

    with capture.open("a") as handle:
        runner.run_suite("gpqa", "http://unused", n=1, seed=42, per_question_out=handle)

    retained = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(retained) == 1
    rejected = [json.loads(line) for line in Path(f"{capture}.rejected.jsonl").read_text().splitlines()]
    assert {item["reason"] for item in rejected} == {"resume_validation_failed", "malformed_json"}


def test_main_pins_capture_schema_and_runner_source_hash(tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    monkeypatch.setattr(runner, "wait_for_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "run_suite",
        lambda *args, **kwargs: {"suite": "swebench_oracle", "n": 0, "correct": 0, "accuracy": 0.0},
    )
    monkeypatch.setattr(sys, "argv", [
        "v7_quality_gate_runner.py", "--output", str(out), "--suites", "swebench_oracle",
    ])

    assert runner.main() == 0
    meta = json.loads(out.read_text())["meta"]
    assert meta["capture_schema_version"] == runner.CAPTURE_SCHEMA_VERSION
    assert meta["runner_source_sha256"] == runner.RUNNER_SOURCE_SHA256
