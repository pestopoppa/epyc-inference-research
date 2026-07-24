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
