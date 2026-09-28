#!/usr/bin/env python3
"""The /v1 ``x_disable_repl`` direct call follows /chat's direct-stage prompt contract.

Origin (2026-09-28 serving proofs, ARCHSWAP-20260927): ``x_force_role`` +
``x_disable_repl`` to the :8083 27B (architect_critic / coder_escalation /
ingest_long_context — thinking-on, so NOT in ``chat_completions_roles()`` and
served over ``/completion``) sent the bare question with the role's
``system_prompt_suffix`` appended and no chat template. The model continued the
text: architect_critic echoed a JSON template, coder_escalation repeated the
suffix's last line, and the ``<think>`` block arrived inline in ``content``.

Contract now:
* a ``/completion`` role gets the orchestrator-side chat template
  (``apply_chat_template_for_role``), a ``/v1/chat/completions`` role stays bare;
* ``skip_suffix=True`` (no suffix after the user's text), as /chat's direct stage;
* a leading closed ``<think>`` block is split into ``message.reasoning_content``
  (stream: one ``reasoning_content`` delta), and the field is omitted otherwise.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.models.openai import OpenAIMessage
from src.api.routes import openai_compat
from src.api.state import get_state, reset_state
from src.features import reset_features

_CHATML = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def _registry(model_name: str = "Qwen3.8-27B-Q8_0") -> MagicMock:
    registry = MagicMock()
    registry.get_role.return_value.model.name = model_name
    return registry


# ── helpers ────────────────────────────────────────────────────────────────


def test_completion_role_gets_chat_template(monkeypatch):
    monkeypatch.setattr(
        "src.chat_completions_roles.chat_completions_roles", lambda: {"frontdoor"}
    )
    out = openai_compat._direct_call_prompt("what is 17*23?", "architect_critic", _registry())
    assert out == _CHATML.format(user="what is 17*23?")


def test_chat_completions_role_stays_bare(monkeypatch):
    # llama-server --jinja templates it; pre-templating would double-template.
    monkeypatch.setattr(
        "src.chat_completions_roles.chat_completions_roles", lambda: {"frontdoor"}
    )
    assert openai_compat._direct_call_prompt("hi", "frontdoor", _registry()) == "hi"


def test_role_enum_is_resolved_by_value(monkeypatch):
    from src.roles import Role

    monkeypatch.setattr(
        "src.chat_completions_roles.chat_completions_roles", lambda: {"frontdoor"}
    )
    assert openai_compat._direct_call_prompt("hi", Role.FRONTDOOR, _registry()) == "hi"


@pytest.mark.parametrize(
    ("text", "reasoning", "answer"),
    [
        ("<think>\nr1\n</think>\n\nThe answer.", "r1", "The answer."),
        ("  <think>a\nb</think>ok", "a\nb", "ok"),
        ("plain answer", None, "plain answer"),
        # Unclosed (budget ran out inside the block): nothing to split.
        ("<think>still thinking", None, "<think>still thinking"),
        # Not leading: an answer that talks about tags is not reasoning.
        ("Use <think>x</think> tags.", None, "Use <think>x</think> tags."),
        ("", None, ""),
    ],
)
def test_split_leading_reasoning(text, reasoning, answer):
    assert openai_compat._split_leading_reasoning(text) == (reasoning, answer)


def test_message_omits_absent_reasoning_content():
    assert "reasoning_content" not in OpenAIMessage(role="assistant", content="x").model_dump()
    dumped = OpenAIMessage(role="assistant", content="x", reasoning_content="r").model_dump()
    assert dumped["reasoning_content"] == "r"


# ── route ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    monkeypatch.delenv("ORCHESTRATOR_V1_ESCALATION", raising=False)
    reset_features()
    reset_state()
    get_state()
    monkeypatch.setattr(
        "src.chat_completions_roles.chat_completions_roles", lambda: {"frontdoor"}
    )
    monkeypatch.setattr(
        openai_compat, "_servable_role_names", lambda: {"frontdoor", "architect_critic"}
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        get_state().registry = _registry()
        yield c
    reset_features()


def _install(monkeypatch, answer: str) -> MagicMock:
    primitives = MagicMock()
    primitives.llm_call.return_value = answer
    primitives.total_tokens_generated = 5
    primitives.total_prompt_tokens_reported = 0

    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    return primitives


def _body(**kw):
    return {
        "model": "orchestrator",
        "messages": [{"role": "user", "content": "In one sentence: what is 17*23?"}],
        "max_tokens": 64,
        "x_force_role": "architect_critic",
        "x_disable_repl": True,
        **kw,
    }


def test_forced_completion_role_is_templated_without_suffix(client, monkeypatch):
    primitives = _install(monkeypatch, "<think>\n17*23 = 391\n</think>\n\n17 x 23 = 391.")
    r = client.post("/v1/chat/completions", json=_body())
    assert r.status_code == 200, r.text
    call = primitives.llm_call.call_args
    assert call.args[0] == _CHATML.format(user="In one sentence: what is 17*23?")
    assert call.kwargs["skip_suffix"] is True
    message = r.json()["choices"][0]["message"]
    assert message["content"] == "17 x 23 = 391."
    assert message["reasoning_content"] == "17*23 = 391"


def test_no_reasoning_key_when_answer_has_no_think_block(client, monkeypatch):
    _install(monkeypatch, "391")
    r = client.post("/v1/chat/completions", json=_body())
    message = r.json()["choices"][0]["message"]
    assert message["content"] == "391"
    assert "reasoning_content" not in message


def test_stream_emits_reasoning_delta_before_content(client, monkeypatch):
    primitives = _install(monkeypatch, "<think>r</think>ok")
    r = client.post("/v1/chat/completions", json=_body(stream=True))
    assert r.status_code == 200
    assert primitives.llm_call.call_args.kwargs["skip_suffix"] is True
    chunks = [
        json.loads(line[len("data: "):])
        for line in r.text.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    assert deltas[0] == {"role": "assistant", "reasoning_content": "r"}
    assert "".join(d.get("content") or "" for d in deltas) == "ok"
    assert "role" not in deltas[1]
