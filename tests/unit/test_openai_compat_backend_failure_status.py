#!/usr/bin/env python3
"""HS-OD-2 — backend failures must not be returned as assistant content.

`/v1/chat/completions` used to catch **every** exception into
`response_text = f"[ERROR] Backend failed: {e}"` and fall through to a normal
`OpenAIChatResponse`. A downstream harness then saw a successful completion whose
text merely began with `[ERROR]`: retry logic, error metrics and eval scorers all
treated it as a model answer, so every eval fan-out through `:8000` scored
backend outages as low-quality generations.

These tests pin the seam in both directions — a failure must surface as a real
error status (or a terminal SSE `error` event, since a stream cannot retract its
200), and the ordinary success path must still be a clean 200 with the model's
text. The second half matters as much as the first: a guard that only asserts the
failure would pass just as happily if the route started erroring on everything.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.state import get_state, reset_state
from src.features import reset_features
from src.scheduling.contention_gate import ContentionDenied


def _install_primitives(monkeypatch, *, llm_call):
    """Force real mode and hand the route a primitives double."""
    primitives = MagicMock()
    primitives.llm_call.side_effect = llm_call
    primitives.total_tokens_generated = 7

    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(
        llm_primitives_module, "LLMPrimitives", lambda **_kwargs: primitives
    )
    return primitives


@pytest.fixture
def client(monkeypatch):
    # `use_real_mode` is `registry is not None and not features().mock_mode`, and
    # mock_mode defaults True "for safety" — without both halves the route serves
    # a [MOCK] completion and every assertion below tests nothing.
    # NB the registry's env_var is the SUFFIX — the real variable is
    # ORCHESTRATOR_MOCK_MODE. Plain MOCK_MODE is silently ignored and the route
    # keeps serving [MOCK] completions, which is a green-looking no-op test.
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    reset_features()
    reset_state()
    get_state()
    with TestClient(app, raise_server_exceptions=False) as c:
        state = get_state()
        if state.registry is None:
            state.registry = MagicMock()
        yield c
    reset_features()


def _body(**overrides):
    body = {
        "model": "frontdoor",
        "messages": [{"role": "user", "content": "hello"}],
        "x_disable_repl": True,
    }
    body.update(overrides)
    return body


def test_backend_exception_is_502_not_a_200_completion(client, monkeypatch):
    """The core HS-OD-2 assertion."""
    _install_primitives(
        monkeypatch, llm_call=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("upstream died"))
    )

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 502, (
        f"backend failure returned {r.status_code}; a non-error status here is the "
        "fail-open shape HS-OD-2 exists to close"
    )
    payload = r.json()
    text = json.dumps(payload)
    assert "upstream died" in text
    # The failure must NOT be dressed as a completion.
    assert "choices" not in payload or not payload.get("choices")


def test_contention_denied_keeps_its_own_503(client, monkeypatch):
    """ContentionDenied has a dedicated handler; it must reach it."""
    _install_primitives(
        monkeypatch,
        llm_call=lambda *a, **k: (_ for _ in ()).throw(ContentionDenied("region busy")),
    )

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 503, (
        "ContentionDenied must reach its app-level handler; swallowing it turns "
        "documented back-pressure into a model answer"
    )
    assert "region busy" in json.dumps(r.json())


def test_success_path_is_still_a_clean_200(client, monkeypatch):
    """The compliant path — the half a failure-only guard would not catch."""
    _install_primitives(monkeypatch, llm_call=lambda *a, **k: "the real answer")

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 200
    payload = r.json()
    assert payload["choices"][0]["message"]["content"] == "the real answer"
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_streaming_backend_failure_emits_a_terminal_error_event(client, monkeypatch):
    """A stream cannot retract its 200, so the event body carries the signal."""
    _install_primitives(
        monkeypatch, llm_call=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("upstream died"))
    )

    r = client.post("/v1/chat/completions", json=_body(stream=True))
    events = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]")
    ]

    assert events, "stream produced no events"
    errors = [e for e in events if "error" in e]
    assert errors, f"streamed failure carried no error object: {events}"
    assert errors[0]["error"]["code"] == 502
    assert errors[0]["choices"][0]["finish_reason"] == "error"

    # The failure must never arrive as assistant content, and the stream must not
    # claim it stopped normally — that pairing is exactly what made an outage
    # indistinguishable from a low-quality generation.
    for event in events:
        for choice in event.get("choices", []):
            assert "upstream died" not in str(choice.get("delta", {}))
            assert choice.get("finish_reason") != "stop"


def test_streaming_success_path_still_streams_content(client, monkeypatch):
    """Compliant streaming path stays intact."""
    _install_primitives(monkeypatch, llm_call=lambda *a, **k: "hi")

    r = client.post("/v1/chat/completions", json=_body(stream=True))
    events = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]")
    ]

    assert not any("error" in e for e in events)
    streamed = "".join(
        c.get("delta", {}).get("content", "")
        for e in events
        for c in e.get("choices", [])
    )
    assert streamed == "hi"
    assert any(
        c.get("finish_reason") == "stop" for e in events for c in e.get("choices", [])
    )


def test_inband_connection_failure_is_502_not_a_200_completion(client, monkeypatch):
    """The raise-side is only half of HS-OD-2. LLMPrimitives.llm_call does NOT
    raise on backend failure: its fail-open contract (primitives.py) returns
    an in-band ``[ERROR: ...]`` string at start-of-answer for transport
    failures, timeouts and upstream non-200s. Without a guard on the RESULT,
    a connection failure would still reach the client as assistant content
    with HTTP 200 — a success wearing an outage's clothes.
    """
    _install_primitives(
        monkeypatch,
        llm_call=lambda *a, **k: (
            "[ERROR: ConnectError: connection refused for http://localhost:9999]"
        ),
    )

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 502, (
        f"in-band connection failure returned {r.status_code}; a non-error status "
        "here is the fail-open shape HS-OD-2 exists to close"
    )
    payload = r.json()
    text = json.dumps(payload)
    assert "connection refused" in text
    assert "choices" not in payload or not payload.get("choices")


def test_inband_http_5xx_from_upstream_is_502(client, monkeypatch):
    """An upstream non-200 is the same failure class: llm_call returns it as
    an in-band ``[ERROR: ...]`` string, which must map to 502 like the raised
    transport failures."""
    _install_primitives(
        monkeypatch,
        llm_call=lambda *a, **k: (
            "[ERROR: Server returned 503 Service Unavailable from http://localhost:9999]"
        ),
    )

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 502
    text = json.dumps(r.json())
    assert "503" in text
    assert "Backend failed" in text


def test_inband_backend_failure_in_repl_mode_is_502(client, monkeypatch):
    """REPL path: the raw llm_call result IS the failure sentinel, not a
    generation. auto_wrap_final would otherwise coerce it into
    ``FINAL("[ERROR: ...]")`` and the REPL would return it as the model's
    final answer."""
    _install_primitives(
        monkeypatch,
        llm_call=lambda *a, **k: (
            "[ERROR: Backend unavailable (circuit open): http://localhost:9999]"
        ),
    )

    r = client.post("/v1/chat/completions", json=_body(x_disable_repl=False))

    assert r.status_code == 502, (
        f"REPL-mode in-band failure returned {r.status_code}; the error string "
        "must not execute as a FINAL(...) answer"
    )
    text = json.dumps(r.json())
    assert "circuit open" in text
    assert "choices" not in r.json() or not r.json().get("choices")


def test_streaming_inband_backend_failure_emits_terminal_error_event(client, monkeypatch):
    """Streaming: an in-band failure must emit the terminal SSE error event
    and stop — never stream the error text as assistant content and close
    with finish_reason "stop"."""
    _install_primitives(
        monkeypatch,
        llm_call=lambda *a, **k: (
            "[ERROR: Backend unavailable (circuit open): http://localhost:9999]"
        ),
    )

    r = client.post("/v1/chat/completions", json=_body(stream=True))
    events = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]")
    ]

    assert events, "stream produced no events"
    errors = [e for e in events if "error" in e]
    assert errors, f"streamed in-band failure carried no error object: {events}"
    assert errors[0]["error"]["code"] == 502
    assert errors[0]["choices"][0]["finish_reason"] == "error"

    for event in events:
        for choice in event.get("choices", []):
            assert "circuit open" not in str(choice.get("delta", {}))
            assert choice.get("finish_reason") != "stop"


def test_mid_answer_error_quote_is_not_a_failure(client, monkeypatch):
    """The in-band anchor is the START-of-answer prefix, never a loose
    substring: a model legitimately discussing "[ERROR:" mid-answer must
    still be a clean 200 (measurement_guards convention)."""
    _install_primitives(
        monkeypatch,
        llm_call=lambda *a, **k: (
            "Backend failures are emitted as '[ERROR: ...]' at start-of-answer, "
            "but this sentence is a real model answer."
        ),
    )

    r = client.post("/v1/chat/completions", json=_body())

    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content.startswith("Backend failures are emitted")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("disable_repl", [False, True])
def test_typed_admission_denial_is_retryable_before_headers(client, monkeypatch, stream, disable_repl):
    from src.exceptions import AdmissionDenied, AdmissionDeniedText

    denial = AdmissionDenied("[ERROR: admission] Backend queue full for http://localhost:9999")
    primitives = _install_primitives(
        monkeypatch, llm_call=lambda *a, **k: AdmissionDeniedText(denial)
    )
    response = client.post("/v1/chat/completions", json=_body(
        stream=stream, x_disable_repl=disable_repl))
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert response.headers["retry-after-ms"] == "5000"
    assert "Backend queue full" in json.dumps(response.json())
    assert primitives.llm_call.call_count == 1


@pytest.mark.parametrize("stream", [False, True])
def test_plain_model_admission_marker_cannot_manufacture_backpressure(client, monkeypatch, stream):
    _install_primitives(monkeypatch, llm_call=lambda *a, **k:
                        "[ERROR: [ERROR: admission] Backend queue full for http://localhost:9999]")
    response = client.post("/v1/chat/completions", json=_body(stream=stream))
    assert response.status_code == (200 if stream else 502)
    assert "retry-after" not in response.headers
    if stream:
        event = next(json.loads(line[6:]) for line in response.text.splitlines()
                     if line.startswith("data: ") and "error" in line)
        assert event["error"]["code"] == 502


def test_streaming_typed_contention_denial_has_retry_headers(client, monkeypatch):
    _install_primitives(monkeypatch, llm_call=lambda *a, **k:
                        (_ for _ in ()).throw(ContentionDenied("region busy")))
    response = client.post("/v1/chat/completions", json=_body(stream=True))
    assert response.status_code == 503
    assert response.headers["retry-after-ms"] == "5000"


def test_residual_sse_denial_text_is_retryable():
    from src.api.routes.openai_compat import _sse_error_event
    event = json.loads(_sse_error_event(chat_id="c", created=0, model="m", message="region busy",
                                     error_type="contention_denied", status_code=503)[6:])
    assert "503 service unavailable" in event["error"]["message"]


def test_primitive_keeps_admission_type_and_legacy_text(monkeypatch):
    from src.exceptions import AdmissionDenied, AdmissionDeniedText
    from src.llm_primitives import LLMPrimitives
    primitives = LLMPrimitives(mock_mode=True)
    primitives.mock_mode = False
    denial = AdmissionDenied("[ERROR: admission] Backend queue full for http://localhost:9999")
    monkeypatch.setattr(primitives, "_real_call", lambda *a, **k:
                        (_ for _ in ()).throw(denial))
    result = primitives.llm_call("hello", skip_suffix=True)
    assert isinstance(result, str) and isinstance(result, AdmissionDeniedText)
    assert result == f"[ERROR: {denial}]" and result.error is denial
    assert primitives.call_log[-1].error == str(denial)
    assert primitives._recursion_depth == 0
    assert not isinstance(denial, RuntimeError)


@pytest.mark.parametrize("answer", ["unique replay", ""])
def test_successful_stream_replays_primed_chunk_once(client, monkeypatch, answer):
    primitives = _install_primitives(monkeypatch, llm_call=lambda *a, **k: answer)
    response = client.post("/v1/chat/completions", json=_body(stream=True))
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ") and line[6:] != "[DONE]"]
    choices = [choice for event in events for choice in event.get("choices", [])]
    assert "".join(choice.get("delta", {}).get("content", "") for choice in choices) == answer
    assert sum(choice.get("finish_reason") == "stop" for choice in choices) == 1
    assert response.text.count("data: [DONE]") == 1
    assert primitives.llm_call.call_count == 1
