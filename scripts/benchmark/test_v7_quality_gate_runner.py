#!/usr/bin/env python3
from __future__ import annotations

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
    }
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
    assert json.loads(rows.read_text()) ["response"] == response
