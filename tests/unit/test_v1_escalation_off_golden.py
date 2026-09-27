#!/usr/bin/env python3
"""TE-1 (UFH-13) — without an explicit x_escalation, client tool mode stays byte-identical.

Every case runs twice: flag ``v1_escalation`` off, and flag ON with the key
absent (escalation is opt-in per request). Both must match the same golden.

The fixture was captured from the route BEFORE ``v1_escalation`` existed
(origin/main b020a1a8). The default REPL/direct modes are pinned by
``test_openai_compat_default_golden.py``; this file pins the client tool mode,
which is the mode OpenCode (and therefore the UFH-13 thesis experiment) uses.

Each case records the exact ``chat_completion_call`` arguments, every
``llm_call`` (there must be none), and the response body with only the
per-request volatile fields normalised.

Regenerate ONLY when a flag-off change is intended:
``TE1_REGEN_GOLDEN=1 pytest tests/unit/test_v1_escalation_off_golden.py``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.state import get_state, reset_state
from src.features import reset_features

GOLDEN = Path(__file__).parent / "fixtures" / "v1_escalation_off_golden.json"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

_MCQ = (
    "Which gas makes up most of Earth's atmosphere?\n"
    "A) Oxygen\nB) Nitrogen\nC) Argon\nD) Carbon dioxide\n"
    "Answer with the letter."
)

_ANSWER = "The answer is B) Nitrogen, which is about 78 percent of the atmosphere by volume."

_TOOL_CALL = {
    "id": "call_abc",
    "type": "function",
    "function": {"name": "read", "arguments": '{"path": "README.md"}'},
}

_BASE = {
    "model": "orchestrator",
    "messages": [{"role": "user", "content": _MCQ}],
    "tools": _TOOLS,
    "x_tool_mode": "client",
    "x_session_id": "ses_te1_golden",
    "x_show_routing": True,
    "temperature": 0,
    "seed": 42,
}

CASES: dict[str, dict[str, Any]] = {
    "client_nonstream_answer": {**_BASE},
    "client_stream_answer_include_usage": {
        **_BASE,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    "client_nonstream_tool_call": {**_BASE, "x_user_id": "user_te1"},
    "client_nonstream_answer_no_routing": {
        key: value for key, value in _BASE.items() if key != "x_show_routing"
    },
}

_BACKEND_USAGE = {"prompt_tokens": 91, "completion_tokens": 17, "cached_tokens": 64}


@pytest.fixture(params=["flag_off", "flag_on_key_absent"])
def client(monkeypatch, request):
    """Flag off, and flag ON with no x_escalation (opt-in): both pinned to one golden."""
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    if request.param == "flag_off":
        monkeypatch.delenv("ORCHESTRATOR_V1_ESCALATION", raising=False)
    else:
        monkeypatch.setenv("ORCHESTRATOR_V1_ESCALATION", "1")
    reset_features()
    reset_state()
    get_state()
    with TestClient(app, raise_server_exceptions=False) as c:
        state = get_state()
        if state.registry is None:
            state.registry = MagicMock()
        yield c
    reset_features()


def _install(monkeypatch, name: str) -> MagicMock:
    primitives = MagicMock()
    primitives.total_tokens_generated = 17
    if name == "client_nonstream_tool_call":
        result = {"content": "", "tool_calls": [_TOOL_CALL], "finish_reason": "tool_calls"}
    else:
        result = {"content": _ANSWER, "tool_calls": [], "finish_reason": "stop"}
    primitives.chat_completion_call.return_value = {**result, "usage": dict(_BACKEND_USAGE)}
    primitives.llm_call.return_value = "UNEXPECTED llm_call"

    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    return primitives


def _normalise_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {"id", "created"}:
                out[k] = f"<{k}>"
            elif k == "elapsed_seconds":
                out[k] = "<elapsed>"
            else:
                out[k] = _normalise_obj(v)
        return out
    if isinstance(obj, list):
        return [_normalise_obj(v) for v in obj]
    return obj


def _normalise_sse(text: str) -> str:
    text = re.sub(r'"id": "chatcmpl-[0-9a-f]+"', '"id": "<id>"', text)
    text = re.sub(r'"created": \d+', '"created": "<created>"', text)
    return re.sub(r'"elapsed_seconds": [0-9.e-]+', '"elapsed_seconds": "<elapsed>"', text)


def _calls(mock: MagicMock) -> list[dict[str, Any]]:
    return [
        {
            "args": json.loads(json.dumps(list(c.args), default=str)),
            "kwargs": json.loads(json.dumps(dict(sorted(c.kwargs.items())), default=str)),
        }
        for c in mock.call_args_list
    ]


def capture(client, monkeypatch, name: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = CASES[name] if body is None else body
    primitives = _install(monkeypatch, name)
    r = client.post("/v1/chat/completions", json=body)
    record: dict[str, Any] = {
        "status": r.status_code,
        "chat_completion_calls": _calls(primitives.chat_completion_call),
        "llm_calls": _calls(primitives.llm_call),
    }
    if body.get("stream"):
        record["sse"] = _normalise_sse(r.text)
    else:
        record["json"] = _normalise_obj(r.json())
    return record


@pytest.mark.parametrize("name", sorted(CASES))
def test_client_mode_with_flag_off_matches_pre_te1_golden(client, monkeypatch, name):
    got = capture(client, monkeypatch, name)
    if os.environ.get("TE1_REGEN_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        data[name] = got
        GOLDEN.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        pytest.skip("golden regenerated")
    expected = json.loads(GOLDEN.read_text())[name]
    assert got["status"] == 200
    assert got["llm_calls"] == []
    assert got == expected
