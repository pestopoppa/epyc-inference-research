"""RTG-58 P2 — the prefix index fed by the real serving paths (flag on).

Offline and inference-free, against in-process fakes speaking the v10 wire:

* the passthrough chat route ``POST /v1/passthrough/{role}/chat/completions``
  (TestClient → gate → fake llama-server → serving record → index), and
* the primitives path the ``/chat`` pipeline uses (``LLMPrimitives.llm_call`` →
  ``_real_call`` → ``CachingBackend`` → ``LlamaServerBackend`` → SSE
  ``/completion`` over ``httpx.MockTransport``), including the ``idle`` pin.

``/slots`` is the fake's slot table, read through the context-limit resolver's
``pool_occupancy`` exactly as production reads it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from src.backends.context_limits import (
    ContextLimit,
    ContextLimitResolver,
    PoolOccupancy,
    SlotState,
    set_context_limit_resolver,
)
from src.inference import prefix_index as pi
from src.scheduling import kv_pool_admission as kpa
from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

URL = "http://localhost:8083"


class Slots:
    """A mutable fake ``/slots`` table."""

    def __init__(self, n: int = 4) -> None:
        self.rows = {i: dict(busy=False, n=0, task=None) for i in range(n)}

    def set(self, sid: int, *, busy: bool = False, n: int = 0, task: int | None = None) -> None:
        self.rows[sid] = dict(busy=busy, n=n, task=task)

    def __call__(self, url: str) -> PoolOccupancy:
        return PoolOccupancy(url=url, slots=tuple(
            SlotState(slot_id=i, n_ctx=98304, is_processing=r["busy"], n_prompt_tokens=r["n"],
                      n_remain=None, n_decoded=1, n_prompt_tokens_processed=0,
                      id_task=r["task"])
            for i, r in self.rows.items()))


@pytest.fixture(autouse=True)
def flag_on(monkeypatch, tmp_path):
    from src.runtime import long_prefill_lease

    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.BLOCK_CHARS_ENV, "256")
    monkeypatch.setattr(long_prefill_lease, "lease_dir", lambda: tmp_path)
    for name in (pi.FORK_ENV, pi.PIN_ENV, pi.PIN_MIN_TOKENS_ENV):
        monkeypatch.delenv(name, raising=False)
    pi.reset_indexes()
    yield
    pi.reset_indexes()


def _doc(tag: str, lines: int) -> str:
    return "".join(f"{tag} line {i}: shared context for every turn\n" for i in range(lines))


def _records(path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ── the passthrough chat route end to end ──────────────────────────────────────


def test_passthrough_route_feeds_the_index_and_binds_from_slots(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from src.api import app
    from src.api.routes import passthrough as pt
    from tests.unit.test_passthrough_route import TIMINGS, FakeLlamaServer

    srv = FakeLlamaServer()
    try:
        monkeypatch.setattr(kpa, "_shared_pool_admission",
                            SharedKVPoolAdmission(occupancy=lambda url: None, cross_process=False))
        monkeypatch.setattr(pt, "_server_urls", lambda: {"architect_critic": srv.url})
        monkeypatch.setattr(pt, "_context_limit", lambda call: ContextLimit(
            url=srv.url, per_request_n_ctx=196608, total_slots=4, kv_unified=True,
            source="registry"))
        log = tmp_path / "pt.jsonl"
        monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", str(log))
        monkeypatch.delenv(pt.PASSTHROUGH_ENV, raising=False)
        monkeypatch.delenv(pt.PASSTHROUGH_ROLES_ENV, raising=False)
        doc = _doc("R", 1000)
        first = [{"role": "system", "content": doc}, {"role": "user", "content": "q1"}]
        raw = json.dumps({"messages": first}).encode()
        with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as c:
            r = c.post("/v1/passthrough/architect_critic/chat/completions", content=raw,
                       headers={"Content-Type": "application/json"})
            assert r.status_code == 200
        assert [body for _p, body in srv.requests] == [raw]  # bytes forwarded untouched
        idx = pi.peek_index(srv.url)
        assert idx is not None and idx.status()["entries"] == {"pending": 1}
        assert idx.stats["observed"] == 1
        # v10's OAI body carries no id_slot: the slot is inferred from /slots, where
        # the call left prompt_n + cache_n + predicted_n tokens.
        slots = Slots()
        slots.set(2, n=TIMINGS["prompt_n"] + TIMINGS["cache_n"] + TIMINGS["predicted_n"], task=11)
        idx.reconcile(slots(srv.url))
        assert idx.stats["bound_inferred"] == 1
        follow_up = pt._wire_payload("chat/completions", {"messages": first + [
            {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]})
        from src.backends import serving_calls

        m = idx.lookup(serving_calls._prompt_text_for_fingerprint(follow_up))
        assert (m.source, m.slot_id) == ("slot_idle", 2) and m.matched_chars > 30_000
        # the gate wiring (this branch) records its prediction; the first call had none
        assert _records(log)[0]["kv_admission"]["prefix_index"]["match"]["source"] is None
    finally:
        srv.close()


# ── the /chat pipeline's backend half: llm_call → /completion, with the idle pin ─


def _primitives(monkeypatch, tmp_path, slots: Slots):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.inference.prefix_cache import CachingBackend, PrefixRouter
    from src.llm_primitives import LLMPrimitives

    monkeypatch.setattr(kpa, "_shared_pool_admission",
                        SharedKVPoolAdmission(occupancy=slots, cross_process=False))
    resolver = ContextLimitResolver(
        live=False, registry_facts=lambda: {8083: {"context_tokens": 393216, "slots": 4,
                                                   "kv_unified": True}},
        role_urls=lambda: {})
    monkeypatch.setattr(resolver, "pool_occupancy", slots)
    set_context_limit_resolver(resolver)
    log = tmp_path / "sc.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", str(log))
    payloads: list[dict] = []
    task = {"n": 20}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        task["n"] += 1
        prompt_tokens = len(body["prompt"]) // 4
        slot = body.get("id_slot", 1)
        slots.set(slot, n=prompt_tokens + 5, task=task["n"])
        final = {"content": "", "tokens_predicted": 5, "tokens_evaluated": prompt_tokens,
                 "id_slot": slot, "stop": True, "stop_type": "eos",
                 "timings": {"prompt_n": prompt_tokens, "cache_n": 0, "predicted_n": 5}}
        sse = (f"data: {json.dumps({'content': 'ok', 'stop': False})}\n\n"
               f"data: {json.dumps(final)}\n\n")
        return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})

    lsb = LlamaServerBackend(ServerConfig(base_url=URL, num_slots=4))
    lsb.client = httpx.Client(base_url=URL, transport=httpx.MockTransport(handler))
    tracker = MagicMock()
    tracker.is_available.return_value = True
    prims = LLMPrimitives(mock_mode=False, server_urls={"coder_escalation": URL},
                          health_tracker=tracker)
    prims._backends["coder_escalation"] = CachingBackend(lsb, PrefixRouter(num_slots=4))
    return prims, payloads, log


def test_llm_call_feeds_the_index_with_the_servers_slot(monkeypatch, tmp_path):
    slots = Slots()
    prims, payloads, log = _primitives(monkeypatch, tmp_path, slots)
    trunk = _doc("W", 300)
    try:
        assert prims.llm_call(trunk + "first", role="coder_escalation", n_tokens=8) == "ok"
    finally:
        set_context_limit_resolver(None)
    idx = pi.peek_index(URL)
    assert idx is not None and idx.status()["entries"] == {"pending": 1}
    idx.reconcile(slots(URL))
    st = idx.status()
    assert st["slots"]["1"]["origin"] == "exact" and st["slots"]["1"]["id_task"] == 21
    assert all("id_slot" not in p for p in payloads)  # pin policy defaults to off


def test_idle_pin_sends_id_slot_only_for_a_verified_idle_slot(monkeypatch, tmp_path):
    monkeypatch.setenv(pi.PIN_ENV, "idle")
    slots = Slots()
    prims, payloads, log = _primitives(monkeypatch, tmp_path, slots)
    trunk = _doc("V", 300)  # ~13k chars, ~3.3k tokens: above the 2048-token pin floor
    try:
        prims.llm_call(trunk + "first", role="coder_escalation", n_tokens=8)
        # the server then routes the slot elsewhere: slot 1 idle, still holding call 1
        prims.llm_call(trunk + "second", role="coder_escalation", n_tokens=8)
        slots.set(1, busy=True, n=9000, task=99)  # a foreign task took slot 1
        prims.llm_call(trunk + "third", role="coder_escalation", n_tokens=8)
    finally:
        set_context_limit_resolver(None)
    assert "id_slot" not in payloads[0]           # nothing known yet
    assert payloads[1].get("id_slot") == 1        # verified idle slot with the trunk
    assert "id_slot" not in payloads[2]           # slot 1 busy with foreign content: no pin
    idx = pi.peek_index(URL)
    assert idx.stats["pins"] == 1 and "1" not in idx.status()["slots"]


def test_pin_policy_ignores_chat_lane_and_explicit_slots(monkeypatch):
    from src.inference.prefix_cache import CachingBackend, PrefixRouter

    monkeypatch.setenv(pi.PIN_ENV, "idle")
    monkeypatch.setenv(pi.PIN_MIN_TOKENS_ENV, "100")
    slots = Slots()
    trunk = _doc("P", 400)
    idx = pi.get_index(URL)
    idx.observe_served(trunk, slot_id=3, prompt_tokens=4000, generated_tokens=0)
    slots.set(3, n=4000, task=7)
    idx.reconcile(slots(URL))
    inner = MagicMock()
    inner.config = SimpleNamespace(base_url=URL, use_chat_completions=True)
    cb = CachingBackend(inner, PrefixRouter(num_slots=4))

    def req(slot_id=None, chat_payload=None):
        return SimpleNamespace(prompt=trunk + "next", slot_id=slot_id, role="coder",
                               stop_sequences=[], chat_payload=chat_payload)

    cb.infer(MagicMock(), req())
    assert inner.infer.call_args[0][1].slot_id is None          # chat lane: never
    inner.config.use_chat_completions = False
    cb.infer(MagicMock(), req(slot_id=0))
    assert inner.infer.call_args[0][1].slot_id == 0             # caller's slot wins
    cb.infer(MagicMock(), req(chat_payload={"messages": []}))
    assert inner.infer.call_args[0][1].slot_id is None          # tool payloads: never
    cb.infer(MagicMock(), req())
    assert inner.infer.call_args[0][1].slot_id == 3
