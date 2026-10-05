#!/usr/bin/env python3
"""HS-4 P0.1(b) — the default /v1 tool mode stays byte-identical.

The golden fixture was captured from the route BEFORE client-executed tool mode
existed (origin/main 09bdb998). Each case records two things:

* the exact arguments the route passed to ``LLMPrimitives.llm_call`` (the
  prompt is where the REPL bridge rewrites client tools into ``CALL()`` text),
* the exact response body (JSON, or the raw SSE stream), with only the
  per-request volatile fields (``id``, ``created``, ``elapsed_seconds``)
  normalised.

Regenerate ONLY when a default-mode change is intended:
``HS4_REGEN_GOLDEN=1 pytest tests/unit/test_openai_compat_default_golden.py``.
The ``*_include_usage`` cases were added by HS-4 P0.4; the pre-existing cases
were not regenerated. 2026-09-28 (intended): the ``direct_*`` cases' ``llm_call``
now carries ``skip_suffix=True`` — the x_disable_repl direct call follows /chat's
direct-stage prompt contract (tests/unit/test_openai_direct_prompt_contract.py);
the response bytes did not change. HS-OD-6 (2026-10-05) intentionally changes the
three direct-with-tools cases: they now return a 422 before inference because the
disabled REPL cannot execute the tool instructions.
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

GOLDEN = Path(__file__).parent / "fixtures" / "openai_compat_default_golden.json"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]

_HISTORY = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "open a.txt"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "alpha"},
    {"role": "user", "content": "what did it say?"},
]

CASES: dict[str, dict[str, Any]] = {
    "direct_nonstream_tools": {
        "model": "frontdoor",
        "messages": _HISTORY,
        "tools": _TOOLS,
        "tool_choice": "auto",
        "x_disable_repl": True,
        "x_show_routing": True,
        "x_max_escalation": "B1",
    },
    "direct_stream_tools": {
        "model": "frontdoor",
        "messages": _HISTORY,
        "tools": _TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        "x_disable_repl": True,
        "x_show_routing": True,
        "stream": True,
    },
    "direct_nonstream_plain": {
        "model": "orchestrator",
        "messages": [{"role": "user", "content": "hi"}],
        "x_disable_repl": True,
    },
    "repl_nonstream_tools": {
        "model": "orchestrator",
        "messages": _HISTORY,
        "tools": _TOOLS,
        "x_show_routing": True,
    },
    "repl_stream_tools": {
        "model": "orchestrator",
        "messages": _HISTORY,
        "tools": _TOOLS,
        "x_show_routing": True,
        "stream": True,
    },
    # HS-4 P0.4 (added cases, captured after the fact): stream_options.include_usage
    # appends one usage chunk; every chunk before it is identical to the cases above.
    "direct_stream_tools_include_usage": {
        "model": "frontdoor",
        "messages": _HISTORY,
        "tools": _TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        "x_disable_repl": True,
        "x_show_routing": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
    "repl_stream_tools_include_usage": {
        "model": "orchestrator",
        "messages": _HISTORY,
        "tools": _TOOLS,
        "x_show_routing": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    },
}


@pytest.fixture(params=["v1_escalation_off", "v1_escalation_on_key_absent"])
def client(monkeypatch, request):
    # TE-1: escalation is opt-in per request, so turning the flag on must not move
    # these bytes for a request that sends no x_escalation.
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    if request.param == "v1_escalation_off":
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


def _install(monkeypatch, answer: str) -> MagicMock:
    primitives = MagicMock()
    primitives.llm_call.return_value = answer
    primitives.total_tokens_generated = 7

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


def _capture(client, monkeypatch, name: str) -> dict[str, Any]:
    body = CASES[name]
    # A bare expression is auto-wrapped into FINAL() by the REPL path; the direct
    # path returns it verbatim.
    answer = '"the answer"' if name.startswith("repl") else "the answer"
    primitives = _install(monkeypatch, answer)
    r = client.post("/v1/chat/completions", json=body)
    calls = [
        {"args": list(c.args), "kwargs": dict(sorted(c.kwargs.items()))}
        for c in primitives.llm_call.call_args_list
    ]
    record: dict[str, Any] = {"status": r.status_code, "llm_calls": calls}
    if r.status_code >= 400:
        record["json"] = _normalise_obj(r.json())
    elif body.get("stream"):
        record["sse"] = _normalise_sse(r.text)
    else:
        record["json"] = _normalise_obj(r.json())
    return record


@pytest.mark.parametrize("name", sorted(CASES))
def test_default_mode_matches_pre_client_mode_golden(client, monkeypatch, name):
    got = _capture(client, monkeypatch, name)
    if os.environ.get("HS4_REGEN_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        data[name] = got
        GOLDEN.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        pytest.skip("golden regenerated")
    expected = json.loads(GOLDEN.read_text())[name]
    expected_status = 422 if name in {
        "direct_nonstream_tools",
        "direct_stream_tools",
        "direct_stream_tools_include_usage",
    } else 200
    assert got["status"] == expected_status
    assert got == expected
