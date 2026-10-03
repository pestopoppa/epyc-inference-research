"""RI-23 (OP-69 (a), 2026-09-29): thinking-on roles on the /v1/chat/completions lane.

Flag ``thinking_roles_chat_lane`` (default OFF):

* OFF — routing, payloads, template skip, inference meta and /v1 responses are unchanged:
  the thinking roles stay on ``/completion`` with no ``chat_template_kwargs``, and
  ``thinking_off()`` binds nothing.
* ON — thinking-on roles (live stack priors: ``--jinja`` AND ``enable_thinking is True``)
  go to ``/v1/chat/completions`` per request with their registry ``chat_template_kwargs``;
  ``message.reasoning_content`` (or streamed ``delta.reasoning_content``) is captured on the
  ``InferenceResult``, recorded in the primitives' inference meta and surfaced on /v1.

(The answer review verdict's RI-23 call-shape and RI-22 status tests were removed with
the verdict itself by RI-18c.)

Everything is mocked; no network.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src import chat_completions_roles as cc
from src.features import get_features, reset_features, set_features

THINKING_ROLES = {"architect_critic", "coder_escalation", "ingest_long_context"}
REGISTRY_CTK = {"enable_thinking": True, "reasoning_effort": "medium"}


@pytest.fixture(autouse=True)
def _clean_features():
    reset_features()
    cc._thinking_roles_cache = None
    yield
    reset_features()
    cc._thinking_roles_cache = None


@pytest.fixture
def flag_on():
    set_features(get_features(override={"thinking_roles_chat_lane": True}))
    yield


@pytest.fixture
def thinking_roles(monkeypatch):
    monkeypatch.setattr(cc, "thinking_chat_lane_roles", lambda: set(THINKING_ROLES))


@pytest.fixture
def role_config():
    from src.registry_loader import (
        AccelerationConfig,
        MemoryConfig,
        ModelConfig,
        PerformanceMetrics,
        RoleConfig,
    )

    return RoleConfig(
        name="architect_critic",
        tier="A",
        description="test",
        model=ModelConfig(name="m", path="m.gguf", quant="Q8_0", size_gb=1.0),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


def _backend(use_chat_completions: bool = False):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig

    return LlamaServerBackend(
        config=ServerConfig(base_url="http://test:8083", use_chat_completions=use_chat_completions)
    )


def _completion_response(content: str = "done") -> Mock:
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "content": content,
        "tokens_predicted": 3,
        "tokens_evaluated": 10,
        "stop_type": "eos",
        "timings": {"prompt_ms": 1.0, "predicted_ms": 1.0, "predicted_per_second": 30.0},
    }
    return response


def _chat_response(content: str, reasoning: str | None) -> Mock:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
        "timings": {"prompt_ms": 5.0, "predicted_ms": 5.0, "predicted_per_second": 30.0},
    }
    return response


def _infer(backend, role_config, request, response):
    captured: dict[str, Any] = {}

    def _post(path, json=None, timeout=None):
        captured["path"] = path
        captured["payload"] = json
        return response

    with patch.object(backend.client, "post", side_effect=_post), patch(
        "src.registry.registry_loader.chat_template_kwargs_for_role",
        return_value=REGISTRY_CTK,
    ):
        result = backend.infer(role_config, request)
    return captured, result


# ── role sets ────────────────────────────────────────────────────────────────


def test_flag_defaults_off():
    assert get_features().thinking_roles_chat_lane is False
    assert get_features(production=True).thinking_roles_chat_lane is False


def test_flag_off_role_sets_unchanged(monkeypatch, thinking_roles):
    monkeypatch.setattr(cc, "_live_chat_completions_roles", lambda: {"frontdoor"})
    monkeypatch.delenv(cc.ENV_VAR, raising=False)
    assert cc.chat_completions_roles() == cc.static_chat_completions_roles() == {"frontdoor"}
    assert cc.thinking_chat_lane_role("architect_critic") is False


def test_flag_on_adds_thinking_roles_to_template_skip_only(monkeypatch, flag_on, thinking_roles):
    monkeypatch.setattr(cc, "_live_chat_completions_roles", lambda: {"frontdoor"})
    monkeypatch.delenv(cc.ENV_VAR, raising=False)
    assert cc.chat_completions_roles() == {"frontdoor"} | THINKING_ROLES
    # The startup-baked set never includes them (per-request admission instead).
    assert cc.static_chat_completions_roles() == {"frontdoor"}
    assert cc.thinking_chat_lane_role("architect_critic") is True
    assert cc.thinking_chat_lane_role("frontdoor") is False
    assert cc.thinking_chat_lane_role(None) is False


def test_thinking_roles_derive_from_jinja_and_enable_thinking(monkeypatch):
    def rec(jinja, thinking):
        return {
            "serving": {"launch": {"runtime": {"flags": {"jinja": jinja}}}},
            "acceleration": {"enable_thinking": thinking},
        }

    monkeypatch.setattr(
        cc,
        "live_stack_role_records",
        lambda: {
            "architect_critic": rec(True, True),
            "coder_escalation": rec(True, True),
            "frontdoor": rec(True, False),
            # Qwen3-VL Instruct, no --jinja, no thinking toggle: never admitted.
            "worker_vision": rec(False, None),
            "odd": rec(None, True),
        },
    )
    assert cc.thinking_chat_lane_roles() == {"architect_critic", "coder_escalation"}


def test_thinking_off_binds_only_with_flag_on(flag_on):
    assert cc.current_chat_template_kwargs_override() is None
    with cc.thinking_off():
        assert cc.current_chat_template_kwargs_override() == {"enable_thinking": False}
    assert cc.current_chat_template_kwargs_override() is None


def test_thinking_off_is_a_noop_with_flag_off():
    with cc.thinking_off():
        assert cc.current_chat_template_kwargs_override() is None


# ── backend: lane, kwargs, reasoning ─────────────────────────────────────────


def test_flag_off_thinking_role_stays_on_completion(role_config, thinking_roles):
    from src.model_server import InferenceRequest

    backend = _backend(False)
    request = InferenceRequest(role="architect_critic", prompt="hello", n_tokens=16)
    captured, result = _infer(backend, role_config, request, _completion_response("done"))

    assert captured["path"] == "/completion"
    assert captured["payload"] == backend._build_payload(role_config, request)
    assert "chat_template_kwargs" not in captured["payload"]
    assert result.output == "done"
    assert result.reasoning_content is None


def test_flag_on_thinking_role_goes_to_chat_lane_with_registry_kwargs(
    role_config, flag_on, thinking_roles
):
    from src.model_server import InferenceRequest

    backend = _backend(False)  # a /completion-baked backend: admitted per request
    request = InferenceRequest(role="architect_critic", prompt="what is 2+2?", n_tokens=64)
    captured, result = _infer(
        backend, role_config, request, _chat_response("4", "2+2 is 4.")
    )

    assert captured["path"] == "/v1/chat/completions"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "what is 2+2?"}]
    assert captured["payload"]["chat_template_kwargs"] == REGISTRY_CTK
    assert result.output == "4"
    assert result.reasoning_content == "2+2 is 4."
    assert "reasoning_content" not in result.to_dict()


def test_per_call_override_merges_over_registry_kwargs(role_config, flag_on, thinking_roles):
    from src.model_server import InferenceRequest

    registry_ctk = dict(REGISTRY_CTK)
    backend = _backend(False)
    request = InferenceRequest(
        role="architect_critic",
        prompt="verdict?",
        n_tokens=80,
        chat_template_kwargs={"enable_thinking": False},
    )
    captured: dict[str, Any] = {}

    def _post(path, json=None, timeout=None):
        captured["payload"] = json
        return _chat_response("OK", None)

    with patch.object(backend.client, "post", side_effect=_post), patch(
        "src.registry.registry_loader.chat_template_kwargs_for_role",
        return_value=registry_ctk,
    ):
        result = backend.infer(role_config, request)

    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "reasoning_effort": "medium",
    }
    assert registry_ctk == REGISTRY_CTK  # cached registry dict never mutated
    assert result.reasoning_content is None


def test_override_field_survives_prefix_cache_replace():
    from dataclasses import replace

    from src.model_server import InferenceRequest

    req = InferenceRequest(role="r", prompt="p", chat_template_kwargs={"enable_thinking": False})
    assert replace(req, slot_id=2).chat_template_kwargs == {"enable_thinking": False}
    assert InferenceRequest(role="r", prompt="p").chat_template_kwargs is None


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return b""


def _sse(delta: dict[str, Any], finish: str | None = None) -> str:
    return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]})


def test_stream_collects_reasoning_deltas_without_feeding_on_chunk(
    role_config, flag_on, thinking_roles
):
    from src.model_server import InferenceRequest

    backend = _backend(False)
    request = InferenceRequest(role="coder_escalation", prompt="write add()", n_tokens=256)
    lines = [
        _sse({"role": "assistant", "reasoning_content": "Think about "}),
        _sse({"reasoning_content": "add."}),
        _sse({"content": "def add(a, b):"}),
        _sse({"content": " return a + b"}, "stop"),
        "data: [DONE]",
    ]
    seen: list[str] = []
    captured: dict[str, Any] = {}

    def _stream(method, path, json=None, timeout=None):
        captured["path"] = path
        captured["payload"] = json
        return _FakeStream(lines)

    with patch.object(backend.client, "stream", side_effect=_stream), patch(
        "src.registry.registry_loader.chat_template_kwargs_for_role",
        return_value=REGISTRY_CTK,
    ):
        result = backend.infer_stream_text(role_config, request, on_chunk=seen.append)

    assert captured["path"] == "/v1/chat/completions"
    assert captured["payload"]["chat_template_kwargs"] == REGISTRY_CTK
    assert seen == ["def add(a, b):", " return a + b"]
    assert result.output == "def add(a, b): return a + b"
    assert result.reasoning_content == "Think about add."


# ── primitives: override plumbing and meta ───────────────────────────────────


class _FakeBackend:
    def __init__(self, reasoning: str | None):
        self.reasoning = reasoning
        self.requests: list[Any] = []

    def infer(self, role_config, request):
        from src.model_server import InferenceResult

        self.requests.append(request)
        return InferenceResult(
            role="architect_critic",
            output="OK",
            tokens_generated=2,
            generation_speed=10.0,
            elapsed_time=0.1,
            success=True,
            completion_reason="stop",
            reasoning_content=self.reasoning,
        )


def _primitives(backend):
    from src.llm_primitives import LLMPrimitives

    primitives = LLMPrimitives(mock_mode=False, server_urls={})
    primitives._backends = {"architect_critic": backend}
    return primitives


def test_primitives_flag_off_no_override_no_reasoning_meta():
    backend = _FakeBackend("some reasoning")
    primitives = _primitives(backend)
    with cc.thinking_off():
        primitives.llm_call("p", role="architect_critic", n_tokens=8, skip_suffix=True)
    assert backend.requests[0].chat_template_kwargs is None
    assert "reasoning_content" not in (primitives.get_last_inference_meta() or {})


def test_primitives_flag_on_carries_override_and_records_reasoning(flag_on):
    backend = _FakeBackend("thought it through")
    primitives = _primitives(backend)
    with cc.thinking_off():
        primitives.llm_call("p", role="architect_critic", n_tokens=8, skip_suffix=True)
    assert backend.requests[0].chat_template_kwargs == {"enable_thinking": False}
    primitives.llm_call("q", role="architect_critic", n_tokens=8, skip_suffix=True)
    assert backend.requests[1].chat_template_kwargs is None  # unbound after the block
    assert primitives.get_last_inference_meta()["reasoning_content"] == "thought it through"


# ── other thinking-off callers ───────────────────────────────────────────────


def test_review_service_calls_run_thinking_off(flag_on):
    from src.proactive_delegation.review_service import ArchitectReviewService

    seen: list[Any] = []
    primitives = MagicMock()
    primitives.llm_call.side_effect = lambda *a, **k: (
        seen.append(cc.current_chat_template_kwargs_override()) or '{"d":"approve","s":0.9}'
    )
    service = ArchitectReviewService(primitives)
    service.review_plan(objective="o", task_type="chat", plan_steps=[{"id": "S1", "actor": "x"}])
    assert seen and all(s == {"enable_thinking": False} for s in seen)


# ── /v1 surfacing ────────────────────────────────────────────────────────────


def test_last_call_reasoning_reads_meta_only():
    from src.api.routes import openai_compat

    assert openai_compat._last_call_reasoning(MagicMock()) is None  # MagicMock meta: not a dict
    p = MagicMock()
    p.get_last_inference_meta.return_value = {"reasoning_content": "r"}
    assert openai_compat._last_call_reasoning(p) == "r"
    p.get_last_inference_meta.return_value = {"reasoning_content": "  "}
    assert openai_compat._last_call_reasoning(p) is None
    p.get_last_inference_meta.side_effect = RuntimeError
    assert openai_compat._last_call_reasoning(p) is None


@pytest.fixture
def v1_client(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import app
    from src.api.routes import openai_compat
    from src.api.state import get_state, reset_state

    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    monkeypatch.delenv("ORCHESTRATOR_V1_ESCALATION", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_THINKING_ROLES_CHAT_LANE", "1")
    reset_features()
    reset_state()
    get_state()
    monkeypatch.setattr(cc, "_live_chat_completions_roles", lambda: {"frontdoor"})
    monkeypatch.setattr(cc, "thinking_chat_lane_roles", lambda: set(THINKING_ROLES))
    monkeypatch.setattr(
        openai_compat, "_servable_role_names", lambda: {"frontdoor", "architect_critic"}
    )
    registry = MagicMock()
    registry.get_role.return_value.model.name = "Qwen3.8-27B-Q8_0"
    with TestClient(app, raise_server_exceptions=False) as c:
        get_state().registry = registry
        yield c
    reset_features()


def _install(monkeypatch, answer: str, reasoning: str | None) -> MagicMock:
    import src.llm_primitives as llm_primitives_module

    primitives = MagicMock()
    primitives.llm_call.return_value = answer
    primitives.total_tokens_generated = 5
    primitives.total_prompt_tokens_reported = 0
    primitives.get_last_inference_meta.return_value = (
        {"reasoning_content": reasoning} if reasoning is not None else {}
    )
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


def test_v1_direct_flag_on_prompt_is_bare_and_reasoning_surfaced(v1_client, monkeypatch):
    primitives = _install(monkeypatch, "17 x 23 = 391.", "17*23 = 391")
    r = v1_client.post("/v1/chat/completions", json=_body())
    assert r.status_code == 200, r.text
    # On the chat lane the server templates: the orchestrator sends the bare turn.
    assert primitives.llm_call.call_args.args[0] == "In one sentence: what is 17*23?"
    message = r.json()["choices"][0]["message"]
    assert message["content"] == "17 x 23 = 391."
    assert message["reasoning_content"] == "17*23 = 391"


def test_v1_direct_flag_on_stream_emits_reasoning_delta(v1_client, monkeypatch):
    _install(monkeypatch, "ok", "r")
    r = v1_client.post("/v1/chat/completions", json=_body(stream=True))
    assert r.status_code == 200
    chunks = [
        json.loads(line[len("data: "):])
        for line in r.text.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    assert deltas[0] == {"role": "assistant", "reasoning_content": "r"}
    assert "".join(d.get("content") or "" for d in deltas) == "ok"


def test_v1_repl_bridge_surfaces_final_turn_reasoning(v1_client, monkeypatch):
    _install(monkeypatch, 'FINAL("391")', "multiply")
    r = v1_client.post("/v1/chat/completions", json=_body(x_disable_repl=False))
    assert r.status_code == 200, r.text
    message = r.json()["choices"][0]["message"]
    assert message["content"] == "391"
    assert message["reasoning_content"] == "multiply"
