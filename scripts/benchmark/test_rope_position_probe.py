#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import rope_position_probe as rope


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_call_chat_completion_disables_thinking_and_reads_content(monkeypatch):
    seen = {}

    def fake_post(url, *, json, timeout):
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        return _FakeResponse({"choices": [{"message": {"content": "2"}}]})

    monkeypatch.setattr(rope.requests, "post", fake_post)

    got = rope._call_chat_completion(
        "prompt",
        "127.0.0.1",
        8070,
        "/v1/chat/completions",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert got == "2"
    assert seen["url"] == "http://127.0.0.1:8070/v1/chat/completions"
    assert seen["json"]["messages"] == [{"role": "user", "content": "prompt"}]
    assert seen["json"]["max_tokens"] == 4
    assert seen["json"]["temperature"] == 0.0
    assert seen["json"]["stream"] is False
    assert seen["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["timeout"] == 120


def test_call_chat_completion_falls_back_to_reasoning_content(monkeypatch):
    def fake_post(url, *, json, timeout):  # noqa: ARG001
        return _FakeResponse({"choices": [{"message": {"content": "", "reasoning_content": "3"}}]})

    monkeypatch.setattr(rope.requests, "post", fake_post)

    assert (
        rope._call_chat_completion(
            "prompt",
            "127.0.0.1",
            8070,
            "/v1/chat/completions",
        )
        == "3"
    )


def test_parse_args_defaults_endpoint_by_api(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["rope_position_probe.py", "--api", "chat", "--context-length", "4096"],
    )

    args = rope._parse_args()

    assert args.endpoint == "/v1/chat/completions"
    assert args.chat_template_kwargs == {"enable_thinking": False}
