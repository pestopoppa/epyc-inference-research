#!/usr/bin/env python3
"""TE-1 (UFH-13) — /v1 escalation through /chat's own post-answer hooks.

Offline only: the route runs against a fake ``LLMPrimitives`` whose calls resolve
their server from the ``server_urls`` the ROUTE passes in (the registry-derived
``get_config().server_urls``), with fixed llama-server-style timings per role.
Quality escalation runs the real ``stages._quality_escalate`` with the quality
detector forced. (The review gate, the other post-answer hook these tests drove, was
removed with /chat's by RI-18c; its tests went with it.) The flag-off byte-identity pin is
``test_v1_escalation_off_golden.py`` (client mode) plus
``test_openai_compat_default_golden.py`` (REPL/direct modes).

Assertions name ROLES, never ports: the pending role swap moves architect_general
(:8083 today) to Flash-Next while coder_escalation stays on the 27B.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.state import get_state, reset_state
from src.config import get_config
from src.features import features, reset_features

MCQ = (
    "Which gas makes up most of Earth's atmosphere?\n"
    "A) Oxygen\nB) Nitrogen\nC) Argon\nD) Carbon dioxide\nAnswer with the letter."
)
FRONTDOOR_ANSWER = "The answer is A) Oxygen, because it is what we breathe in every single breath."
QUALITY_ANSWER = "B) Nitrogen."
READ_TOOL = {
    "type": "function",
    "function": {"name": "read", "parameters": {"type": "object", "properties": {}}},
}
TOOL_CALL = {
    "id": "call_abc",
    "type": "function",
    "function": {"name": "read", "arguments": "{}"},
}

# (prompt_ms, gen_ms, tokens, prompt_tokens) per role — what llama-server's
# timings/usage would report for one call.
TIMINGS = {
    "frontdoor": (40.0, 160.0, 17, 91),
    "architect_general": (1250.0, 3750.5, 12, 300),
    "architect_critic": (1250.0, 3750.5, 12, 300),
    "coder_escalation": (900.0, 2100.0, 40, 280),
    "worker_general": (100.0, 400.0, 30, 150),
}


class FakePrimitives:
    """The seam /v1 calls, with LLMPrimitives' per-request accumulators."""

    def __init__(
        self,
        *,
        server_urls: dict[str, str],
        registry: Any,
        answers: dict[str, Any],
        client_result: dict[str, Any] | None = None,
    ) -> None:
        self.server_urls = dict(server_urls)
        self.registry = registry
        self.answers = answers
        self.client_result = client_result
        self.calls: list[dict[str, Any]] = []
        self.total_calls = 0
        self.total_tokens_generated = 0
        self.total_prompt_eval_ms = 0.0
        self.total_generation_ms = 0.0
        self.total_prompt_tokens_reported = 0
        self._trace: dict[str, Any] = {}
        self._last: dict[str, Any] | None = None

    def set_request_trace_keys(self, keys):
        self._trace = {k: v for k, v in (keys or {}).items() if v is not None}

    def get_request_trace_keys(self):
        return dict(self._trace)

    def get_last_inference_meta(self):
        return self._last

    def _account(self, kind: str, role: str, prompt: Any, kwargs: dict[str, Any]) -> None:
        self.total_calls += 1
        prompt_ms, gen_ms, tokens, prompt_tokens = TIMINGS.get(role, (0.0, 0.0, 0, 0))
        self.total_prompt_eval_ms += prompt_ms
        self.total_generation_ms += gen_ms
        self.total_tokens_generated += tokens
        self.total_prompt_tokens_reported += prompt_tokens
        self._last = {"role": role, "completion_reason": "stop"}
        self.calls.append(
            {
                "kind": kind,
                "role": role,
                "url": self.server_urls.get(role),
                "trace": dict(self._trace),
                "prompt": prompt,
                "kwargs": kwargs,
            }
        )

    def llm_call(self, prompt, role="worker", n_tokens=None, **kwargs):
        role = str(role)
        self._account("llm", role, prompt, {"n_tokens": n_tokens, **kwargs})
        answer = self.answers.get(role, "")
        return answer(prompt) if callable(answer) else answer

    def chat_completion_call(
        self, messages, role, tools=None, tool_choice=None, n_tokens=None, **kw
    ):
        role = str(role)
        self._account("chat", role, messages, {"n_tokens": n_tokens, **kw})
        prompt_ms, gen_ms, tokens, prompt_tokens = TIMINGS[role]
        return {
            **self.client_result,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens},
        }

    def consultant_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["url"] == self.server_urls["architect_general"]]


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Flag ON, tap on (events to tmp)."""
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    monkeypatch.setenv("ORCHESTRATOR_V1_ESCALATION", "1")
    monkeypatch.setenv("ORCHESTRATOR_V1_CLIENT_SESSION_GUARD", "0")
    monkeypatch.setenv("INFERENCE_TAP_FILE", str(tmp_path / "tap.log"))
    monkeypatch.setenv("INFERENCE_TAP_EVENTS_FILE", str(tmp_path / "events.jsonl"))
    reset_features()
    reset_state()
    get_state()
    with TestClient(app, raise_server_exceptions=False) as client:
        state = get_state()
        if state.registry is None:
            state.registry = MagicMock()
        yield SimpleNamespace(client=client, state=state, tmp=tmp_path, monkeypatch=monkeypatch)
    reset_features()


def _install(env, *, client_result=None, answers=None) -> dict[str, Any]:
    holder: dict[str, Any] = {}
    default_answers = {
        "architect_general": QUALITY_ANSWER,
        "coder_escalation": QUALITY_ANSWER,
        "frontdoor": '"the answer"',
    }
    if answers:
        default_answers.update(answers)

    def _factory(**kwargs):
        holder["kwargs"] = kwargs
        holder["fake"] = FakePrimitives(
            server_urls=kwargs["server_urls"],
            registry=kwargs.get("registry"),
            answers=default_answers,
            client_result=client_result
            or {"content": FRONTDOOR_ANSWER, "tool_calls": [], "finish_reason": "stop"},
        )
        return holder["fake"]

    import src.llm_primitives as llm_primitives_module

    env.monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", _factory)
    return holder


def _body(**overrides) -> dict[str, Any]:
    body = {
        "model": "orchestrator",
        "messages": [{"role": "user", "content": MCQ}],
        "tools": [READ_TOOL],
        "x_tool_mode": "client",
        "x_session_id": "ses_te1",
        "x_show_routing": True,
    }
    body.update(overrides)
    return body


def _tap_events(env) -> list[dict[str, Any]]:
    path = env.tmp / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _force_quality_issue(env) -> None:
    env.monkeypatch.setenv("ORCHESTRATOR_GENERATION_MONITOR", "1")
    reset_features()
    import src.api.routes.chat_pipeline.stages as stages

    env.monkeypatch.setattr(stages, "_detect_output_quality_issue", lambda answer: "repetition")


# ── flag and key contract ────────────────────────────────────────────────────


def test_flag_is_default_off_in_test_and_prod(monkeypatch):
    from src.features import _FEATURE_REGISTRY

    monkeypatch.delenv("ORCHESTRATOR_V1_ESCALATION", raising=False)
    spec = next(s for s in _FEATURE_REGISTRY if s.name == "v1_escalation")
    assert (spec.default_test, spec.default_prod) == (False, False)
    reset_features()
    assert features().v1_escalation is False


def test_bad_x_escalation_value_is_422(env):
    _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="architect_critic"))
    assert r.status_code == 422
    assert "x_escalation" in r.text


# ── A2: flag on, consultant pinned, quality escalation fires ────────────────


def test_a2_quality_escalation_to_architect_general_with_exact_telemetry(env):
    _force_quality_issue(env)
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="architect_general"))
    assert r.status_code == 200, r.text
    fake = holder["fake"]
    data = r.json()

    # /chat's direct-stage chain: frontdoor answer -> quality escalation to the pinned
    # consultant (the review gate after it was removed by RI-18c).
    assert [c["role"] for c in fake.calls] == ["frontdoor", "architect_general"]
    assert data["choices"][0]["message"]["content"] == QUALITY_ANSWER
    assert data["choices"][0]["finish_reason"] == "stop"

    receipt = data["x_orchestrator_metadata"]["escalation"]
    assert receipt["enabled"] is True and receipt["fired"] is True
    assert receipt["target"] == "pinned" and receipt["target_role"] == "architect_general"
    assert receipt["from_role"] == "frontdoor"
    assert receipt["final_answer_role"] == "architect_general"
    (step,) = receipt["steps"]
    assert (step["trigger"], step["from_role"], step["to_role"]) == (
        "quality_escalation",
        "frontdoor",
        "architect_general",
    )
    assert step["consultant"] is True and step["outcome"] == "adopted"
    assert (step["prompt_ms"], step["gen_ms"]) == (1250.0, 3750.5)
    assert step["device_seconds"] == pytest.approx(5.0005)
    assert (step["tokens"], step["prompt_tokens"], step["calls"]) == (12, 300, 1)
    assert step["server_url"] == get_config().server_urls.as_dict()["architect_general"]
    assert receipt["consultant_device_seconds"] == pytest.approx(5.0005)
    assert receipt["request_device_seconds"] == pytest.approx((40 + 160 + 1250 + 3750.5) / 1000)

    # The escalation call's own tap section is tagged; the request's keys come back after.
    escalation_call = fake.calls[1]
    assert escalation_call["trace"]["escalation_trigger"] == "quality_escalation"
    assert escalation_call["trace"]["escalation_to_role"] == "architect_general"
    assert escalation_call["trace"]["x_escalation"] == "architect_general"
    assert "escalation_trigger" not in fake.get_request_trace_keys()

    # usage: the frontdoor call plus the escalation call's server counts.
    assert data["usage"]["completion_tokens"] == 17 + 12
    assert data["usage"]["prompt_tokens"] == 91 + 300

    events = [e for e in _tap_events(env) if e["event"] == "v1_escalation"]
    assert len(events) == 1
    event = events[0]
    assert "request_id" not in event
    assert event["request_keys"]["x_session_id"] == "ses_te1"
    assert event["request_keys"]["x_escalation"] == "architect_general"
    assert event["consultant_device_seconds"] == pytest.approx(5.0005)
    assert event["steps"][0]["to_role"] == "architect_general"
    # RI-18c: no review_gate tap event any more.
    assert not [e for e in _tap_events(env) if e.get("event") == "review_gate"]


def test_a2_pinned_quality_escalation_targets_architect_general_not_coder(env):
    _force_quality_issue(env)
    holder = _install(env, answers={"architect_general": QUALITY_ANSWER})
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="architect_general"))
    assert r.status_code == 200, r.text
    fake = holder["fake"]
    assert [c["role"] for c in fake.calls] == ["frontdoor", "architect_general"]
    assert "coder_escalation" not in [c["role"] for c in fake.calls]
    assert fake.calls[1]["kwargs"]["n_tokens"] == 2048  # /chat's own quality-escalation call
    receipt = r.json()["x_orchestrator_metadata"]["escalation"]
    (step,) = receipt["steps"]
    assert (step["trigger"], step["to_role"], step["outcome"]) == (
        "quality_escalation",
        "architect_general",
        "adopted",
    )
    assert step["consultant"] is True
    assert receipt["final_answer_role"] == "architect_general"
    assert r.json()["choices"][0]["message"]["content"] == QUALITY_ANSWER


def test_auto_keeps_chat_targets_quality_escalation_goes_to_coder_escalation(env):
    _force_quality_issue(env)
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="auto"))
    assert r.status_code == 200, r.text
    fake = holder["fake"]
    assert [c["role"] for c in fake.calls] == ["frontdoor", "coder_escalation"]
    receipt = r.json()["x_orchestrator_metadata"]["escalation"]
    assert receipt["target"] == "chat_default" and receipt["target_role"] is None
    (step,) = receipt["steps"]
    assert step["to_role"] == "coder_escalation"
    urls = get_config().server_urls.as_dict()
    # Consultant = architect_general's server. coder_escalation counts only while it
    # shares that process (true before the role swap, false after it).
    assert step["consultant"] is (urls["coder_escalation"] == urls["architect_general"])


def test_flag_on_and_key_absent_is_off_opt_in(env):
    """Opt-in: the flag alone never escalates unkeyed traffic, and records nothing."""
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body())
    assert r.status_code == 200
    meta = r.json()["x_orchestrator_metadata"]
    assert "escalation" not in meta
    assert "x_escalation" not in meta["request_keys"]
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor"]
    assert r.json()["choices"][0]["message"]["content"] == FRONTDOOR_ANSWER
    assert not [e for e in _tap_events(env) if e["event"] == "v1_escalation"]


def test_explicit_auto_without_a_quality_issue_makes_no_escalation_call(env):
    """RI-18c: with the review gate gone, a clean answer under ``auto`` is served as is."""
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="auto"))
    receipt = r.json()["x_orchestrator_metadata"]["escalation"]
    assert receipt["requested"] == "auto" and receipt["enabled"] is True
    assert receipt["target"] == "chat_default"
    assert receipt["fired"] is False and receipt["steps"] == []
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor"]
    assert r.json()["choices"][0]["message"]["content"] == FRONTDOOR_ANSWER


def test_no_quality_issue_means_no_escalation_call(env):
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="architect_general"))
    receipt = r.json()["x_orchestrator_metadata"]["escalation"]
    assert receipt["enabled"] is True and receipt["fired"] is False
    assert receipt["steps"] == [] and receipt["consultant_device_seconds"] == 0
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor"]
    assert r.json()["choices"][0]["message"]["content"] == FRONTDOOR_ANSWER


def test_tool_call_turn_is_never_escalated(env):
    holder = _install(
        env, client_result={"content": "", "tool_calls": [TOOL_CALL], "finish_reason": "tool_calls"}
    )
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="architect_general"))
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor"]
    assert r.json()["x_orchestrator_metadata"]["escalation"]["fired"] is False


def test_streaming_client_mode_streams_the_escalated_answer_and_usage(env):
    _force_quality_issue(env)
    holder = _install(env)
    r = env.client.post(
        "/v1/chat/completions",
        json=_body(
            x_escalation="architect_general", stream=True, stream_options={"include_usage": True}
        ),
    )
    assert r.status_code == 200
    events = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"])
    assert content == QUALITY_ANSWER
    final = next(e for e in events if e["choices"] and e["choices"][0]["finish_reason"])
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["x_orchestrator_metadata"]["escalation"][
        "consultant_device_seconds"
    ] == pytest.approx(5.0005)
    usage = events[-1]["usage"]
    assert usage["completion_tokens"] == 17 + 12
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor", "architect_general"]


# ── default REPL bridge and x_disable_repl ──────────────────────────────────


def test_repl_mode_final_answer_gets_no_escalation_hook(env):
    """/chat's REPL stage ran only the review gate; RI-18c removed it, so a FINAL answer
    of the REPL bridge is served unescalated (and /chat's REPL stage has no quality
    escalation, so a forced quality issue changes nothing)."""
    _force_quality_issue(env)
    long_final = '"' + FRONTDOOR_ANSWER + '"'
    holder = _install(env, answers={"frontdoor": long_final})
    body = _body(x_escalation="architect_general")
    body.pop("x_tool_mode")
    body.pop("tools")
    r = env.client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor"]
    receipt = r.json()["x_orchestrator_metadata"]["escalation"]
    assert receipt["fired"] is False and receipt["steps"] == []
    assert r.json()["choices"][0]["message"]["content"] == FRONTDOOR_ANSWER


def test_disable_repl_gets_the_direct_chain(env):
    _force_quality_issue(env)
    holder = _install(
        env, answers={"frontdoor": FRONTDOOR_ANSWER, "architect_general": QUALITY_ANSWER}
    )
    body = _body(x_escalation="architect_general", x_disable_repl=True)
    body.pop("x_tool_mode")
    body.pop("tools")
    r = env.client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    # quality escalation adopts architect_general's answer, exactly as /chat's direct
    # stage does (no review gate follows it since RI-18c).
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor", "architect_general"]
    assert r.json()["choices"][0]["message"]["content"] == QUALITY_ANSWER


# ── the switch: every way escalation must NOT happen ─────────────────────────


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"x_escalation": "off"}, "x_escalation_off"),
        (
            {"x_escalation": "architect_general", "x_force_role": "architect_general"},
            "role_override",
        ),
        (
            {"x_escalation": "architect_general", "x_orchestrator_role": "architect_general"},
            "role_override",
        ),
        ({"x_escalation": "auto", "x_force_model": "frontdoor"}, "role_override"),
        ({"x_escalation": "auto", "model": "worker_general"}, "not_frontdoor"),
    ],
)
def test_disabled_paths_make_no_escalation_call(env, overrides, reason):
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(**overrides))
    assert r.status_code == 200, r.text
    fake = holder["fake"]
    assert len(fake.calls) == 1 and fake.calls[0]["kind"] == "chat"
    receipt = r.json()["x_orchestrator_metadata"]["escalation"]
    assert receipt["enabled"] is False and receipt["disabled_reason"] == reason
    assert receipt["fired"] is False and receipt["steps"] == []
    assert r.json()["choices"][0]["message"]["content"] == FRONTDOOR_ANSWER


def test_flag_off_with_key_sent_records_flag_off_and_never_escalates(env):
    env.monkeypatch.setenv("ORCHESTRATOR_V1_ESCALATION", "0")
    reset_features()
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="architect_general"))
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor"]
    meta = r.json()["x_orchestrator_metadata"]
    assert meta["escalation"]["disabled_reason"] == "flag_off"
    assert meta["request_keys"]["x_escalation"] == "architect_general"
    events = [e for e in _tap_events(env) if e["event"] == "v1_escalation"]
    assert events and events[0]["enabled"] is False


def test_a1_off_serves_exactly_what_the_flag_off_route_serves(env):
    """x_escalation=off with the flag ON == flag OFF, except the recorded key/receipt."""
    served = {}
    for label, flag, extra in (("flag_off", "0", {}), ("a1", "1", {"x_escalation": "off"})):
        env.monkeypatch.setenv("ORCHESTRATOR_V1_ESCALATION", flag)
        reset_features()
        holder = _install(env)
        for stream in (False, True):
            r = env.client.post("/v1/chat/completions", json=_body(stream=stream, **extra))
            text = r.text
            if stream:
                chunks = [
                    json.loads(line[6:])
                    for line in text.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
            else:
                chunks = [r.json()]
            for chunk in chunks:
                chunk.pop("id", None)
                chunk.pop("created", None)
                meta = chunk.get("x_orchestrator_metadata")
                if meta:
                    meta.pop("elapsed_seconds", None)
                    meta.pop("escalation", None)
                    meta.get("request_keys", {}).pop("x_escalation", None)
            served[(label, stream)] = (
                chunks,
                [{k: v for k, v in c.items() if k != "trace"} for c in holder["fake"].calls],
            )
    for stream in (False, True):
        assert served[("flag_off", stream)] == served[("a1", stream)]


def test_hook_failure_is_recorded_and_serves_the_unescalated_answer(env):
    import src.api.routes.chat_pipeline.stages as stages

    def _boom(*_a, **_k):
        raise RuntimeError("detector exploded")

    env.monkeypatch.setattr(stages, "_quality_escalate", _boom)
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="architect_general"))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == FRONTDOOR_ANSWER
    receipt = r.json()["x_orchestrator_metadata"]["escalation"]
    assert receipt["error"] == "RuntimeError: detector exploded"
    assert receipt["final_answer_role"] == "frontdoor"
    assert [c["role"] for c in holder["fake"].calls] == ["frontdoor"]


# ── TE-2: the escalation reaches architect_general's server ─────────────────


def test_te2_escalation_reaches_the_registry_resolved_architect_general_server(env):
    """Zero inference. The route builds primitives from the registry-derived
    server_urls; A2's consultant calls land on whatever server architect_general
    resolves to, and on no other role's server unless it IS that server."""
    _force_quality_issue(env)
    holder = _install(env)
    r = env.client.post("/v1/chat/completions", json=_body(x_escalation="architect_general"))
    assert r.status_code == 200
    registry_urls = get_config().server_urls.as_dict()
    assert holder["kwargs"]["server_urls"] == registry_urls
    fake = holder["fake"]
    escalated = [
        c
        for c in fake.calls
        if c["trace"].get("escalation_trigger") == "quality_escalation"
    ]
    assert escalated and all(c["role"] == "architect_general" for c in escalated)
    assert all(c["url"] == registry_urls["architect_general"] for c in escalated)
    assert fake.consultant_calls() == escalated
    receipt = r.json()["x_orchestrator_metadata"]["escalation"]
    assert receipt["consultant_role"] == "architect_general"
    assert receipt["consultant_url"] == registry_urls["architect_general"]
    from src.llm_primitives.backend import _url_str_ports

    assert receipt["consultant_ports"] == _url_str_ports(registry_urls["architect_general"])
    # Escalation left frontdoor's server.
    assert registry_urls["architect_general"] != registry_urls["frontdoor"]


# ── units ────────────────────────────────────────────────────────────────────


def test_emit_request_event_writes_only_when_the_tap_is_on(monkeypatch, tmp_path):
    from src.runtime import inference_tap

    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("INFERENCE_TAP_EVENTS_FILE", str(events))
    monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
    monkeypatch.setattr(inference_tap, "_read_sentinel", lambda: "")
    assert inference_tap.emit_request_event("v1_escalation", chat_id="c1") is False
    assert not events.exists()
    monkeypatch.setenv("INFERENCE_TAP_FILE", str(tmp_path / "tap.log"))
    assert inference_tap.emit_request_event("v1_escalation", chat_id="c1", request_id="x") is True
    (event,) = [json.loads(line) for line in events.read_text().splitlines()]
    assert event["event"] == "v1_escalation" and event["chat_id"] == "c1"
    assert "request_id" not in event


def test_quality_escalate_default_target_is_unchanged_for_chat(monkeypatch):
    import src.api.routes.chat_pipeline.stages as stages
    from src.roles import Role

    monkeypatch.setenv("ORCHESTRATOR_GENERATION_MONITOR", "1")
    reset_features()
    monkeypatch.setattr(stages, "_detect_output_quality_issue", lambda answer: "repetition")
    primitives = MagicMock()
    primitives.llm_call.return_value = "better"
    answer, role = stages._quality_escalate("bad bad bad", "p", primitives, Role.FRONTDOOR)
    assert (answer, role) == ("better", Role.CODER_ESCALATION)
    assert primitives.llm_call.call_args.kwargs["role"] == "coder_escalation"
    reset_features()


def test_plan_is_none_whenever_no_key_was_sent():
    from src.api.routes.v1_escalation import plan_v1_escalation

    for flag_on in (False, True):
        assert (
            plan_v1_escalation(
                flag_on=flag_on,
                requested=None,
                role="frontdoor",
                role_override=False,
                image_input=False,
            )
            is None
        )
    plan = plan_v1_escalation(
        flag_on=True, requested="auto", role="frontdoor", role_override=False, image_input=True
    )
    assert plan.enabled is False and plan.disabled_reason == "image_input"
