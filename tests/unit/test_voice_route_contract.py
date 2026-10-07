"""Off-host contract controls for the session-backed voice SSE route."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import dep_app_state, dep_session_store
from src.api.routes.voice import router


class _LeaseManager:
    def __init__(self):
        self.released = []
        self.release_observations = []
        self.worker_finished = None
        self.held = False
        self.reacquire_refused = False

    def acquire(self, session_id, *, ttl_s, wait_s, label):
        if self.held:
            from src.session.lease import SessionLeaseHeld

            holder = SimpleNamespace(
                owner_pid=1, owner_host="synthetic", fencing_token=17,
                expires_at=time.time() + 60,
            )
            self.reacquire_refused = True
            raise SessionLeaseHeld(session_id, holder)
        self.held = True
        return SimpleNamespace(session_id=session_id, fencing_token=17, ttl_s=ttl_s)

    def heartbeat(self, lease):
        return lease

    def release(self, lease):
        self.released.append(lease.fencing_token)
        self.held = False
        if self.worker_finished is not None:
            self.release_observations.append(self.worker_finished.is_set())


class _Store:
    def __init__(self):
        self.leases = _LeaseManager()
        self.rows = []

    def get_session(self, session_id):
        return SimpleNamespace(id=session_id)

    def get_conversation_summary(self, session_id):
        return None

    def get_messages(self, session_id, *, limit):
        return []

    def append_message(self, session_id, turn_id, role, text, **kwargs):
        self.rows.append((role, text, kwargs))
        return SimpleNamespace(id=len(self.rows))


class _Primitives:
    def __init__(self, **kwargs):
        pass

    @contextmanager
    def request_context(self, **kwargs):
        assert isinstance(kwargs.get("deadline_s"), float)
        yield

    def llm_call(self, prompt, *, role, n_tokens, skip_suffix, on_chunk):
        on_chunk("Hello ")
        on_chunk("there.")
        return "Hello there."


def _client(monkeypatch, store, *, inference=_Primitives, enabled=True):
    from src import config, llm_primitives
    import src.features as feature_module

    monkeypatch.setattr(
        "src.chat_completions_roles.chat_completions_roles", lambda: {"frontdoor"}
    )
    monkeypatch.setattr(
        config, "get_config",
        lambda: SimpleNamespace(server_urls=SimpleNamespace(as_dict=lambda: {})),
    )
    monkeypatch.setattr(llm_primitives, "LLMPrimitives", inference)
    monkeypatch.setattr(
        feature_module, "features", lambda: SimpleNamespace(voice_turn=enabled)
    )
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.dependency_overrides[dep_app_state] = lambda: SimpleNamespace(
        registry=object(), health_tracker=object(), admission=None
    )
    app.dependency_overrides[dep_session_store] = lambda: store
    return TestClient(app)


def test_voice_turn_feature_defaults_off_in_both_profiles():
    from src.features import Features, _FEATURE_REGISTRY

    spec = next(item for item in _FEATURE_REGISTRY if item.name == "voice_turn")
    assert spec.default_test is False
    assert spec.default_prod is False
    assert Features().voice_turn is False


def test_voice_route_is_disabled_by_default_until_explicit_enable(monkeypatch):
    store = _Store()
    client = _client(monkeypatch, store, enabled=False)

    response = client.post("/v1/voice/turn", json={
        "session_id": "session-a", "user_request": "Say hello"
    })

    assert response.status_code == 404
    assert store.rows == []
    assert store.leases.released == []


def test_voice_route_streams_native_chunks_and_persists_only_completed_answer(monkeypatch):
    store = _Store()
    client = _client(monkeypatch, store)

    response = client.post("/v1/voice/turn", json={
        "session_id": "session-a", "user_request": "Say hello", "cancel_token": "opaque"
    })

    assert response.status_code == 200
    assert 'event: answer.delta\ndata: {"turn_id":' in response.text
    assert '"text": "Hello "' in response.text
    assert '"text": "there."' in response.text
    assert 'event: done\n' in response.text
    assert '"cancel_token": "opaque"' in response.text
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert events
    assert len({event["turn_id"] for event in events}) == 1
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert [row[0] for row in store.rows] == ["user", "assistant"]
    assert store.rows[-1][1] == "Hello there."
    assert store.rows[-1][2]["spoken_text"] == "Hello there."
    assert store.rows[-1][2]["fencing_token"] == 17
    assert store.leases.released == [17]


def test_voice_response_verbatim_preserves_literal_values_before_emitting(monkeypatch):
    class CapturingPrimitives(_Primitives):
        prompt = ""

        def llm_call(self, prompt, *, role, n_tokens, skip_suffix, on_chunk):
            type(self).prompt = prompt
            on_chunk("The release is ")
            on_chunk("v10.2.")
            return "The release is v10.2."

    store = _Store()
    response = _client(monkeypatch, store, inference=CapturingPrimitives).post(
        "/v1/voice/turn", json={
            "session_id": "session-a", "user_request": "State the release",
            "response_mode": "verbatim", "must_preserve": ["v10.2"],
        }
    )

    assert response.status_code == 200
    assert '"text": "The release is v10.2."' in response.text
    assert 'event: preserve\n' in response.text
    assert '"values": ["v10.2"]' in response.text
    assert 'event: done\n' in response.text
    assert '"v10.2"' in CapturingPrimitives.prompt
    assert "verbatim" in CapturingPrimitives.prompt
    assert store.rows[-1][1] == "The release is v10.2."
    assert store.rows[-1][2]["spoken_text"] == "The release is v10.2."


def test_display_goal_emits_a_nonspoken_payload_and_persists_it_separately(monkeypatch):
    class DisplayPrimitives(_Primitives):
        def llm_call(self, prompt, *, role, n_tokens, skip_suffix, on_chunk):
            on_chunk("```sh\nrun --flag\n```\n")
            return "```sh\nrun --flag\n```\n"

    store = _Store()
    response = _client(monkeypatch, store, inference=DisplayPrimitives).post(
        "/v1/voice/turn", json={
            "session_id": "session-a", "user_request": "Show the command",
            "response_goal": "display",
        }
    )

    assert response.status_code == 200
    assert "event: display\ndata: " in response.text
    assert '"payload": {"text": "```sh\\nrun --flag\\n```\\n"}' in response.text
    assert "event: answer.delta" not in response.text
    assert '"spoken_text": null' in response.text or store.rows[-1][2]["spoken_text"] is None
    assert store.rows[-1][2]["display"] == {"text": "```sh\nrun --flag\n```\n"}


def test_missing_protected_value_fails_closed_without_emitting_or_persisting_answer(monkeypatch):
    class OmittingPrimitives(_Primitives):
        def llm_call(self, prompt, *, role, n_tokens, skip_suffix, on_chunk):
            on_chunk("The release is ready.")
            return "The release is ready."

    store = _Store()
    response = _client(monkeypatch, store, inference=OmittingPrimitives).post(
        "/v1/voice/turn", json={
            "session_id": "session-a", "user_request": "State the release",
            "must_preserve": ["v10.2"],
        }
    )

    assert response.status_code == 200
    assert "event: error\ndata: " in response.text
    assert "event: answer.delta" not in response.text
    assert "event: preserve" not in response.text
    assert "event: done" not in response.text
    assert [row[0] for row in store.rows] == ["user"]


def test_voice_route_refuses_header_identity_mismatch_before_write(monkeypatch):
    store = _Store()
    client = _client(monkeypatch, store)

    response = client.post(
        "/v1/voice/turn",
        headers={"x-session-id": "session-b"},
        json={"session_id": "session-a", "user_request": "Hello"},
    )

    assert response.status_code == 400
    assert store.rows == []
    assert store.leases.released == []


def test_voice_route_does_not_persist_partial_answer_on_generation_failure(monkeypatch):
    class FailingPrimitives(_Primitives):
        def llm_call(self, prompt, *, role, n_tokens, skip_suffix, on_chunk):
            on_chunk("partial")
            raise RuntimeError("synthetic backend failure")

    store = _Store()
    client = _client(monkeypatch, store, inference=FailingPrimitives)

    response = client.post("/v1/voice/turn", json={
        "session_id": "session-a", "user_request": "Say hello"
    })

    assert response.status_code == 200
    assert '"text": "partial"' in response.text
    assert 'event: error\n' in response.text
    assert 'event: done\n' not in response.text
    assert [row[0] for row in store.rows] == ["user"]
    assert store.leases.released == [17]


def test_voice_timeout_uses_timeout_event_and_deadline(monkeypatch):
    worker_finished = threading.Event()

    class TimedOutPrimitives(_Primitives):
        def llm_call(self, prompt, *, role, n_tokens, skip_suffix, on_chunk):
            on_chunk("partial")
            worker_finished.set()
            raise TimeoutError("synthetic request deadline")

    store = _Store()
    store.leases.worker_finished = worker_finished
    response = _client(monkeypatch, store, inference=TimedOutPrimitives).post(
        "/v1/voice/turn", json={"session_id": "session-a", "user_request": "Hello"}
    )

    assert response.status_code == 200
    assert '"text": "partial"' in response.text
    assert '"code": "timeout"' in response.text
    assert '"message": "request deadline exceeded"' in response.text
    assert store.leases.release_observations == [True]
    assert store.leases.released == [17]


def test_voice_setup_timeout_returns_http_504_before_stream_headers(monkeypatch):
    from src.api.routes import voice as voice_module

    ticks = iter((100.0, 102.0))
    monkeypatch.setattr(voice_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    monkeypatch.setattr("src.api.routes.chat_utils.role_timeout_for", lambda _role: 1)
    store = _Store()

    response = _client(monkeypatch, store).post(
        "/v1/voice/turn", json={"session_id": "session-a", "user_request": "Hello"}
    )

    assert response.status_code == 504
    assert store.leases.released == [17]
    assert [row[0] for row in store.rows] == ["user"]


def test_thread_chunk_close_waits_for_underlying_worker_quiescence():
    from src.api.routes.voice import _thread_chunks

    async def scenario():
        release_worker = threading.Event()
        worker_finished = threading.Event()
        cancelled = threading.Event()

        async def connected_request():
            return False

        request = SimpleNamespace(is_disconnected=connected_request)

        def call(on_chunk):
            on_chunk("first")
            release_worker.wait(timeout=2)
            worker_finished.set()
            return "first"

        stream = _thread_chunks(
            call, cancel_event=cancelled, request=request,
            deadline_s=asyncio.get_running_loop().time() + 1,
        )
        assert await anext(stream) == ("chunk", "first")
        closing = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0.02)
        assert not closing.done()
        assert not worker_finished.is_set()
        assert cancelled.is_set()
        closing.cancel()
        await asyncio.sleep(0.01)
        closing.cancel()
        await asyncio.sleep(0.01)
        assert not closing.done()
        release_worker.set()
        try:
            await closing
        except asyncio.CancelledError:
            pass
        assert worker_finished.is_set()

    asyncio.run(scenario())


def test_thread_chunk_deadline_expiry_after_chunk_cancels_and_drains_worker():
    from src.api.routes.voice import _thread_chunks

    async def scenario():
        release_worker = threading.Event()
        worker_finished = threading.Event()
        cancelled = threading.Event()

        async def connected_request():
            return False

        def call(on_chunk):
            on_chunk("partial")
            release_worker.wait(timeout=2)
            worker_finished.set()
            return "partial"

        stream = _thread_chunks(
            call, cancel_event=cancelled,
            request=SimpleNamespace(is_disconnected=connected_request),
            deadline_s=asyncio.get_running_loop().time() + 0.05,
        )
        assert await anext(stream) == ("chunk", "partial")
        timeout_read = asyncio.create_task(anext(stream))
        kind, _ = await timeout_read
        assert kind == "timeout"
        assert cancelled.is_set()
        assert not worker_finished.is_set()
        closing = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0.02)
        assert not closing.done()
        release_worker.set()
        await closing
        assert worker_finished.is_set()

    asyncio.run(scenario())


def test_thread_chunk_disconnect_cancels_and_drains_worker():
    from src.api.routes.voice import _thread_chunks

    async def scenario():
        release_worker = threading.Event()
        worker_finished = threading.Event()
        disconnected = threading.Event()
        cancelled = threading.Event()

        async def is_disconnected():
            return disconnected.is_set()

        def call(_on_chunk):
            release_worker.wait(timeout=2)
            worker_finished.set()
            return "never streamed"

        stream = _thread_chunks(
            call, cancel_event=cancelled,
            request=SimpleNamespace(is_disconnected=is_disconnected),
            deadline_s=asyncio.get_running_loop().time() + 1,
        )
        read = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.02)
        disconnected.set()
        await asyncio.sleep(0.15)
        assert cancelled.is_set()
        assert not worker_finished.is_set()
        assert not read.done()
        release_worker.set()
        try:
            await read
        except asyncio.CancelledError:
            pass
        assert worker_finished.is_set()

    asyncio.run(scenario())


def test_consumer_exception_keeps_session_lease_until_worker_exits(monkeypatch):
    worker_finished = threading.Event()

    store = _Store()
    store.leases.worker_finished = worker_finished

    class SlowPrefixPrimitivesWithDrain(_Primitives):
        def llm_call(self, prompt, *, role, n_tokens, skip_suffix, on_chunk):
            on_chunk(" " * 4097)
            try:
                store.leases.acquire(
                    "session-a", ttl_s=60, wait_s=0, label="synthetic-second-owner"
                )
            except Exception:
                pass
            time.sleep(0.05)
            worker_finished.set()
            return "answer"

    response = _client(monkeypatch, store, inference=SlowPrefixPrimitivesWithDrain).post(
        "/v1/voice/turn", json={"session_id": "session-a", "user_request": "Hello"}
    )

    assert response.status_code == 200
    assert 'event: error\n' in response.text
    assert '"code": "generation_failed"' in response.text
    assert store.leases.reacquire_refused
    assert store.leases.release_observations == [True]
    assert store.leases.released == [17]
