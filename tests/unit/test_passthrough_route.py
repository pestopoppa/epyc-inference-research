"""UFH14-B6 — /v1/passthrough/{role}/... raw OpenAI passthrough behind the gate.

Offline and inference-free: the "llama-server" is an in-thread fake HTTP server
on 127.0.0.1 that speaks the v10 wire shapes (SSE chunks, ``timings`` on the
final chunk / ``response.completed``), the context limit is injected and the
shared KV pool gate is a fresh ``SharedKVPoolAdmission`` with no /slots reads.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.routes import passthrough as pt
from src.backends.context_limits import ContextLimit
from src.scheduling import kv_pool_admission as kpa
from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

TIMINGS = {
    "cache_n": 12, "prompt_n": 40, "prompt_ms": 61.5, "prompt_per_second": 650.4,
    "predicted_n": 3, "predicted_ms": 90.0, "predicted_per_second": 33.3,
}
TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}
TOOL_CALL = {
    "id": "call_1", "type": "function",
    "function": {"name": "read", "arguments": "{\"path\": \"README.md\"}"},
}


def _sse(obj) -> bytes:
    return b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"


class FakeLlamaServer:
    """Minimal llama-server stand-in. ``hold`` pauses a stream after its first event."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, bytes]] = []
        self.hold = threading.Event()
        self.hold.set()
        self.error_status: int | None = None
        self.client_gone = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):  # quiet
                pass

            def do_GET(self):
                payload = json.dumps({"object": "list", "data": [{"id": "fake.gguf"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                outer.requests.append((self.path, raw))
                body = json.loads(raw)
                if outer.error_status:
                    err = {"error": {"code": 400, "type": "exceed_context_size_error",
                                     "message": "request (9 tokens) exceeds the available "
                                                "context size (8 tokens)"}}
                    self.send_response(outer.error_status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(err).encode())
                    return
                if not body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "id": "chatcmpl-x", "object": "chat.completion",
                        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [TOOL_CALL]}}],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 3,
                                  "total_tokens": 43},
                        "timings": TIMINGS,
                    }).encode())
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if self.path == "/v1/responses":
                    events = [
                        b"event: response.created\n" + _sse({"type": "response.created"}),
                        b"event: response.output_text.delta\n"
                        + _sse({"type": "response.output_text.delta", "delta": "hi"}),
                        b"event: response.completed\n" + _sse({
                            "type": "response.completed",
                            "response": {"object": "response", "usage": {
                                "input_tokens": 40, "output_tokens": 3, "total_tokens": 43}},
                            "timings": TIMINGS,
                        }),
                    ]
                else:
                    events = [
                        _sse({"choices": [{"index": 0, "delta": {"role": "assistant",
                                                                 "content": "he"}}]}),
                        _sse({"choices": [{"index": 0, "delta": {"content": "llo"}}]}),
                        _sse({"choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
                              "timings": TIMINGS}),
                        _sse(b"[DONE]"),
                    ]
                try:
                    for i, event in enumerate(events):
                        self.wfile.write(event)
                        self.wfile.flush()
                        if i == 0:
                            outer.hold.wait(timeout=10)
                except (BrokenPipeError, ConnectionResetError):
                    outer.client_gone = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.hold.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    srv = FakeLlamaServer()
    yield srv
    srv.close()


@pytest.fixture
def pool(monkeypatch):
    fresh = SharedKVPoolAdmission(occupancy=lambda url: None)
    monkeypatch.setattr(kpa, "_shared_pool_admission", fresh)
    for name in (kpa.KV_POOL_LONG_PREFILL_ENV, kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV,
                 kpa.KV_POOL_WAIT_ENV, kpa.KV_POOL_MAX_QUEUED_ENV):
        monkeypatch.delenv(name, raising=False)
    return fresh


@pytest.fixture
def limit_holder():
    return {}


@pytest.fixture(autouse=True)
def wiring(monkeypatch, server, pool, limit_holder, tmp_path):
    monkeypatch.setattr(pt, "_server_urls", lambda: {
        "architect_critic": server.url, "fleet_role": "http://a:1,http://b:2",
    })
    limit_holder["limit"] = ContextLimit(
        url=server.url, per_request_n_ctx=196608, total_slots=4, kv_unified=True,
        source="registry",
    )
    monkeypatch.setattr(pt, "_context_limit", lambda call: limit_holder["limit"])
    log = tmp_path / "serving_calls.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", str(log))
    monkeypatch.delenv(pt.PASSTHROUGH_ENV, raising=False)
    monkeypatch.delenv(pt.PASSTHROUGH_ROLES_ENV, raising=False)
    return log


def _records(log) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as c:
        yield c


def _state():
    return SimpleNamespace(health_tracker=None, admission=None, registry=None)


def _request(path: str, raw: bytes, host: str = "127.0.0.1"):
    from starlette.requests import Request

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    scope = {
        "type": "http", "method": "POST", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": [(b"content-type", b"application/json")],
        "client": (host, 40000), "server": ("127.0.0.1", 8000), "scheme": "http",
        "http_version": "1.1", "root_path": "",
    }
    return Request(scope, receive)


# ── streaming ────────────────────────────────────────────────────────────────


async def test_stream_is_unbuffered_and_hands_on_the_lease_at_first_chunk(
    server, pool, wiring, monkeypatch
):
    monkeypatch.setenv(kpa.KV_POOL_LONG_PREFILL_ENV, "10")  # this prompt is "long"
    monkeypatch.setenv(kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV, "0")  # lease never expires
    server.hold.clear()  # upstream sends event 1, then waits
    raw = json.dumps({
        "model": "anything", "stream": True,
        "messages": [{"role": "user", "content": "x" * 400}],
    }).encode()
    resp = await pt._passthrough(
        "chat/completions", "architect_critic",
        _request("/v1/passthrough/architect_critic/chat/completions", raw), _state(),
    )
    assert resp.status_code == 200
    assert resp.media_type.startswith("text/event-stream")
    it = resp.body_iterator
    first = await asyncio.wait_for(it.__anext__(), timeout=5)
    # The first event arrived while upstream is still holding the rest: unbuffered.
    assert b'"content": "he"' in first
    assert server.hold.is_set() is False
    await asyncio.sleep(0.1)
    # Long-prefill lease handed on at the first chunk; the token reservation stays.
    assert pool.long_prefill_holder(server.url) is None
    assert pool.in_flight_tokens(server.url) > 0
    server.hold.set()
    rest = b""
    async for chunk in it:
        rest += chunk
    assert b'"timings"' in rest and b"[DONE]" in rest
    # Reservation returned at the end of the stream.
    assert pool.in_flight_tokens(server.url) == 0
    # Upstream received the client's bytes verbatim.
    assert server.requests == [("/v1/chat/completions", raw)]

    (rec,) = _records(wiring)
    assert rec["caller"]["source"] == "passthrough"
    assert rec["method"] == "passthrough"
    assert rec["role"] == "architect_critic"
    assert rec["dispatched"] is True
    assert rec["outcome"] == "ok"
    assert rec["timings_source"] == "server"
    assert rec["timings"]["prompt_n"] == 40 and rec["timings"]["predicted_n"] == 3
    assert rec["notes"]["endpoint"] == "/v1/chat/completions"
    assert rec["notes"]["stream"] is True
    assert rec["passthrough"]["long_prefill"] is True
    assert rec["passthrough"]["sse_events"] == 4
    assert "pre_dispatch_wait_ms" in rec["queue"]


async def test_client_disconnect_mid_stream_is_recorded_as_cancelled(server, pool, wiring):
    server.hold.clear()
    raw = json.dumps({"stream": True, "messages": [{"role": "user", "content": "hi"}]}).encode()
    resp = await pt._passthrough(
        "chat/completions", "architect_critic",
        _request("/v1/passthrough/architect_critic/chat/completions", raw), _state(),
    )
    it = resp.body_iterator
    await asyncio.wait_for(it.__anext__(), timeout=5)
    await it.aclose()  # the client went away
    # Upstream is still holding its next event: closing the upstream response must
    # end the worker on its own (no further chunk needed to notice the cancel).
    deadline = time.time() + 5
    while not _records(wiring) and time.time() < deadline:
        await asyncio.sleep(0.05)
    assert server.hold.is_set() is False
    server.hold.set()
    # ... and the server sees the client gone (llama-server stops the slot).
    deadline = time.time() + 5
    while not server.client_gone and time.time() < deadline:
        await asyncio.sleep(0.05)
    assert server.client_gone
    (rec,) = _records(wiring)
    assert rec["outcome"] == "cancelled"
    assert pool.in_flight_tokens(server.url) == 0


def test_responses_stream_forwarded_natively_with_timings(client, server, wiring):
    raw = json.dumps({"model": "m", "stream": True, "input": "hello",
                      "tools": [{"type": "function", "name": "read", "parameters": {}}]})
    r = client.post("/v1/passthrough/architect_critic/responses", content=raw,
                    headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert r.headers["x-orchestrator-passthrough"] == "architect_critic"
    assert "event: response.completed" in r.text
    completed = [line for line in r.text.splitlines() if '"response.completed"' in line][0]
    assert json.loads(completed[len("data: "):])["timings"] == TIMINGS
    assert server.requests == [("/v1/responses", raw.encode())]
    (rec,) = _records(wiring)
    assert rec["caller"]["source"] == "passthrough"
    assert rec["notes"]["endpoint"] == "/v1/responses"
    assert rec["timings"]["prompt_ms"] == 61.5
    assert rec["notes"]["usage"]["input_tokens"] == 40


# ── tools ────────────────────────────────────────────────────────────────────


def test_tools_round_trip_verbatim(client, server, wiring):
    body = {
        "model": "gpt-whatever",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "read the readme"},
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_1", "content": "# README"},
        ],
        "tools": [TOOL],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "stop": ["</done>"],
        "temperature": 0.2,
        "max_tokens": 64000,  # no clamp, no 32768 rule in passthrough
    }
    raw = json.dumps(body)
    r = client.post("/v1/passthrough/architect_critic/chat/completions", content=raw,
                    headers={"content-type": "application/json"})
    assert r.status_code == 200
    out = r.json()
    assert out["choices"][0]["message"]["tool_calls"] == [TOOL_CALL]
    assert out["timings"] == TIMINGS
    assert server.requests == [("/v1/chat/completions", raw.encode())]
    (rec,) = _records(wiring)
    assert rec["passthrough"]["stream"] is False
    assert rec["passthrough"]["new_tokens_budget"] == 64000
    assert rec["timings"]["cache_n"] == 12
    assert rec["notes"]["usage"]["total_tokens"] == 43


def test_upstream_error_forwarded_verbatim(client, server, wiring):
    server.error_status = 400
    r = client.post("/v1/passthrough/architect_critic/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "exceed_context_size_error"
    (rec,) = _records(wiring)
    assert rec["outcome"] == "failed"
    assert rec["passthrough"]["http_status"] == 400


# ── the gate ─────────────────────────────────────────────────────────────────


def test_admission_queues_until_the_pool_frees(client, server, pool, limit_holder):
    limit_holder["limit"] = ContextLimit(
        url=server.url, per_request_n_ctx=8192, total_slots=4, kv_unified=True,
        source="registry",
    )
    # An in-flight request already holds the whole pool.
    held = pool.acquire(server.url, 8150, 8192, max_new_tokens=0)
    assert held is not None
    result = {}

    def call():
        result["r"] = client.post(
            "/v1/passthrough/architect_critic/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000},
        )

    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.6)
    assert server.requests == []  # queued, never dispatched
    assert pool.queued(server.url) == 1
    pool.release(server.url, held)
    t.join(timeout=10)
    assert result["r"].status_code == 200
    assert len(server.requests) == 1


def test_admission_wait_budget_ends_as_typed_pool_exhausted(
    client, server, pool, limit_holder, wiring, monkeypatch
):
    monkeypatch.setenv(kpa.KV_POOL_WAIT_ENV, "0.3")
    limit_holder["limit"] = ContextLimit(
        url=server.url, per_request_n_ctx=8192, total_slots=4, kv_unified=True,
        source="registry",
    )
    held = pool.acquire(server.url, 8150, 8192, max_new_tokens=0)
    try:
        r = client.post("/v1/passthrough/architect_critic/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}]})
    finally:
        pool.release(server.url, held)
    assert r.status_code == 503
    assert r.headers["retry-after"]
    assert r.json()["context_overflow"]["kind"] == "pool_exhausted"
    assert server.requests == []
    (rec,) = _records(wiring)
    assert rec["dispatched"] is False
    assert rec["outcome"] == "context_overflow"
    assert rec["caller"]["source"] == "passthrough"


def test_request_semaphore_full_is_503(client, server, monkeypatch):
    from src.api.admission import AdmissionController
    from src.api.state import get_state

    controller = AdmissionController(limits={server.url: 1})
    assert controller.acquire(server.url)
    monkeypatch.setattr(get_state(), "admission", controller)
    r = client.post("/v1/passthrough/architect_critic/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert r.json()["error"] == "admission_queue_full"
    assert server.requests == []


def test_binding_per_request_cap_refuses_request_too_large(
    client, server, limit_holder, wiring
):
    # Unified -c 196608 above the model's trained 32768: the server would accept.
    limit_holder["limit"] = ContextLimit(
        url=server.url, per_request_n_ctx=32768, total_slots=4, kv_unified=True,
        source="registry", server_n_ctx=196608, request_cap=32768,
    )
    r = client.post("/v1/passthrough/architect_critic/responses",
                    json={"input": "y" * (4 * 33000), "stream": True})
    assert r.status_code == 413
    overflow = r.json()["context_overflow"]
    assert overflow["kind"] == "request_too_large"
    assert server.requests == []
    (rec,) = _records(wiring)
    assert rec["dispatched"] is False
    assert rec["outcome"] == "context_overflow"
    assert rec["error"]["kind"] == "request_too_large"


def test_prompt_below_cap_is_forwarded(client, server, limit_holder):
    limit_holder["limit"] = ContextLimit(
        url=server.url, per_request_n_ctx=32768, total_slots=4, kv_unified=True,
        source="registry", server_n_ctx=196608, request_cap=32768,
    )
    r = client.post("/v1/passthrough/architect_critic/chat/completions",
                    json={"messages": [{"role": "user", "content": "y" * 4000}]})
    assert r.status_code == 200
    assert len(server.requests) == 1


# ── safety and resolution ────────────────────────────────────────────────────


def test_non_localhost_is_refused(server):
    with TestClient(app, raise_server_exceptions=False) as remote:  # host "testclient"
        r = remote.post("/v1/passthrough/architect_critic/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 403
    assert server.requests == []


def test_unknown_role_404_fleet_422_allow_list_403(client, server, monkeypatch):
    msg = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/passthrough/nope/chat/completions", json=msg).status_code == 404
    assert client.post("/v1/passthrough/fleet_role/chat/completions", json=msg).status_code == 422
    monkeypatch.setenv(pt.PASSTHROUGH_ROLES_ENV, "frontdoor")
    assert client.post(
        "/v1/passthrough/architect_critic/chat/completions", json=msg
    ).status_code == 403
    monkeypatch.setenv(pt.PASSTHROUGH_ROLES_ENV, "")
    monkeypatch.setenv(pt.PASSTHROUGH_ENV, "0")
    assert client.post(
        "/v1/passthrough/architect_critic/chat/completions", json=msg
    ).status_code == 404
    assert server.requests == []


def test_models_forwarded(client, server):
    r = client.get("/v1/passthrough/architect_critic/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "fake.gguf"


def test_existing_routes_unchanged():
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/v1/chat/completions" in paths
    assert "/v1/responses" not in paths  # not grafted onto the orchestrated surface
    assert "/v1/passthrough/{role}/responses" in paths


# ── estimates ────────────────────────────────────────────────────────────────


def test_prompt_text_counts_rendered_strings_not_structure_or_images():
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "abc"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
            {"role": "assistant", "tool_calls": [TOOL_CALL]},
        ],
        "tools": [TOOL],
    }
    text = pt.prompt_text(body)
    assert "abc" in text and "README.md" in text and '"read"' in text
    assert "base64" not in text and "call_1" not in text
    assert pt.prompt_text({"instructions": "sys", "input": [
        {"role": "user", "content": "q"},
        {"type": "function_call_output", "call_id": "c", "output": "out"},
    ]}) == "sys\nq\nout"
    assert pt.requested_new_tokens({"max_output_tokens": 77}) == 77
    assert pt.requested_new_tokens({}) > 0
