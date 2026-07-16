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
    assert seen["timeout"] == 120


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
