"""RTG-58 P2 — the prefix index wired into admission and the serving path.

Offline and inference-free. Two end-to-end wiring paths are exercised against
in-process fakes that speak the v10 wire shapes:

* the passthrough chat route ``POST /v1/passthrough/{role}/chat/completions``
  (TestClient → gate → upstream → serving record → index), and
* the primitives path the ``/chat`` pipeline uses (``LLMPrimitives.llm_call`` →
  ``_real_call`` → KV pool gate → ``CachingBackend`` → ``LlamaServerBackend`` →
  ``/completion`` over ``httpx.MockTransport``).

``/slots`` is the fake's own slot table, read through the pool's injected
occupancy function exactly as production reads it through the resolver.
"""

from __future__ import annotations

import json
import threading
import time
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
        self.rows = {i: dict(busy=False, n=0, task=None, decoded=1) for i in range(n)}

    def set(self, sid: int, *, busy: bool = False, n: int = 0, task: int | None = None,
            decoded: int = 1) -> None:
        self.rows[sid] = dict(busy=busy, n=n, task=task, decoded=decoded)

    def __call__(self, url: str) -> PoolOccupancy:
        return PoolOccupancy(url=url, slots=tuple(
            SlotState(slot_id=i, n_ctx=98304, is_processing=r["busy"], n_prompt_tokens=r["n"],
                      n_remain=None, n_decoded=r["decoded"], n_prompt_tokens_processed=0,
                      id_task=r["task"])
            for i, r in self.rows.items()))


@pytest.fixture(autouse=True)
def flag_on(monkeypatch, tmp_path):
    from src.runtime import long_prefill_lease

    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.BLOCK_CHARS_ENV, "256")
    monkeypatch.setattr(long_prefill_lease, "lease_dir", lambda: tmp_path)
    for name in (pi.FORK_ENV, pi.PIN_ENV, pi.LPM_ENV, pi.LPM_MIN_TOKENS_ENV,
                 pi.TRUNK_MIN_TOKENS_ENV, pi.TRUNK_HOLD_S_ENV, kpa.KV_POOL_LONG_PREFILL_ENV,
                 kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV, kpa.KV_POOL_WAIT_ENV):
        monkeypatch.delenv(name, raising=False)
    pi.reset_indexes()
    yield
    pi.reset_indexes()


def _doc(tag: str, lines: int) -> str:
    return "".join(f"{tag} line {i}: shared context for every turn\n" for i in range(lines))


def _seed_slot(text: str, slot: int, slots: Slots, *, tokens: int = 4000, task: int = 7):
    idx = pi.get_index(URL)
    idx.observe_served(text, slot_id=slot, prompt_tokens=tokens, generated_tokens=0)
    slots.set(slot, n=tokens, task=task)
    idx.reconcile(slots(URL))
    return idx


# ── admission decisions ───────────────────────────────────────────────────────


def test_credit_from_a_verified_idle_slot_sizes_the_long_prefill_rule(monkeypatch):
    monkeypatch.setenv(kpa.KV_POOL_LONG_PREFILL_ENV, "8000")
    monkeypatch.setenv(kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV, "100")
    slots = Slots()
    trunk = _doc("T", 900)  # ~40k chars
    _seed_slot(trunk, 2, slots)
    pool = SharedKVPoolAdmission(occupancy=slots, history=SimpleNamespace(
        lookup=lambda url, ladder: (None, "no_history")), cross_process=False)
    prompt = trunk + "new question"
    t = pool.acquire(URL, len(prompt) // 3, 393216, prompt_text=None, prefix_key=prompt,
                     timeout_s=0)
    info = pool.admission_record(t)
    assert info["cache_credit_source"] == "prefix_index"
    assert info["long_prefill"] is False and info["cache_credited"] is True
    pidx = info["prefix_index"]
    assert pidx["match"]["source"] == "slot_idle" and pidx["predicted_slot"] == 2
    assert pidx["predicted_cache_tokens"] > 3000
    status = pool.get_status()[URL]["prefix_index"]
    assert status["entries"] == {"slot": 1, "inflight": 1}
    pool.release(URL, t)
    assert pi.get_index(URL).status()["entries"] == {"slot": 1}


def test_lpm_admits_a_reusable_prefix_ahead_of_a_head_that_does_not_fit(monkeypatch):
    monkeypatch.setenv(pi.LPM_MIN_TOKENS_ENV, "100")
    slots = Slots()
    trunk = _doc("L", 400)
    _seed_slot(trunk, 1, slots)
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False)
    held = pool.acquire(URL, 700, 1000, timeout_s=0)
    out: dict[str, int | None] = {}
    cold = threading.Thread(target=lambda: out.setdefault("cold", pool.acquire(
        URL, 400, 1000, prefix_key=_doc("C", 400), timeout_s=5, poll_s=0.01)))
    cold.start()
    time.sleep(0.2)  # the cold request is the queue head and does not fit
    hot = pool.acquire(URL, 50, 1000, prefix_key=trunk + "q", timeout_s=2, poll_s=0.01)
    assert hot is not None and "cold" not in out
    info = pool.admission_record(hot)["prefix_index"]
    assert info["verdict"] == "lpm" and info["lpm_score"] > 0
    assert pi.get_index(URL).stats["lpm_passes"] == 1
    pool.release(URL, held)
    cold.join(timeout=5)
    assert out["cold"] is not None
    pool.release(URL, hot)
    pool.release(URL, out["cold"])


def test_lpm_pass_happens_and_each_waiter_is_passed_at_most_once(monkeypatch):
    monkeypatch.setenv(pi.LPM_MIN_TOKENS_ENV, "100")
    slots = Slots()
    trunk = _doc("M", 400)
    _seed_slot(trunk, 1, slots)
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False)
    with pool._cond:  # a head waiter that cannot fit, queued ahead
        pool._queue[URL] = [100]
        pool._waiting[100] = (900, False)
        pool._lpm_score[100] = 0
        pool._inflight[URL] = {99: 200}
    first = pool._admissible(URL, 101, [100, 101], 50, False, 1000, None)
    assert first == "wait"  # 101 has no score yet
    pool._lpm_score[101] = 2000
    assert pool._admissible(URL, 101, [100, 101], 50, False, 1000, None) == "lpm"
    assert pool._lpm_skips[100] == 1
    pool._lpm_score[102] = 3000
    assert pool._admissible(URL, 102, [100, 102], 50, False, 1000, None) == "wait"  # budget spent


#: A P1 server's ``/props.slot_fork`` with the fork on: fork features follow it.
KV_CAPS = {"min_tokens": 0, "mode": "kv", "checkpoint_at": False}


def test_trunk_hold_waits_for_the_siblings_prefill_with_fork_on(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    slots = Slots()
    slots.set(0, busy=True, n=12000, task=1, decoded=0)  # the owner is in prefill
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False,
                                 fork_caps=lambda url: KV_CAPS)
    trunk = _doc("F", 1200)
    owner = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "child A", timeout_s=0)
    assert owner is not None
    got: dict[str, int | None] = {}
    child = threading.Thread(target=lambda: got.setdefault("b", pool.acquire(
        URL, 13000, 393216, prefix_key=trunk + "child B", timeout_s=10, poll_s=0.01)))
    child.start()
    time.sleep(0.3)
    assert "b" not in got  # held behind the trunk
    assert pi.get_index(URL).stats["trunk_holds"] == 1
    pool.prefill_done(URL, owner)  # first chunk: the trunk is prefilled
    child.join(timeout=5)
    info = pool.admission_record(got["b"])["prefix_index"]
    assert info["match"]["source"] == "inflight"
    assert info["fork_plan"]["source"] == "inflight"
    assert info["fork_plan"]["junction_chars"] == len(trunk) + len("child ")
    assert info["fork_credit_tokens"] > 0
    # the reservation is credited with the shared trunk
    assert pool._inflight[URL][got["b"]] < pool._inflight[URL][owner]
    pool.release(URL, got["b"])
    pool.release(URL, owner)


def test_no_trunk_hold_without_fork(monkeypatch):
    slots = Slots()
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False)
    trunk = _doc("N", 1200)
    a = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    b = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=0)
    assert a is not None and b is not None
    info = pool.admission_record(b)["prefix_index"]
    assert info["fork"] is False and info["fork_credit_tokens"] == 0
    assert "fork_plan" not in info
    assert info["unique_cells"]["shareable_tokens_est"] > 0  # what P1 would save


def test_trunk_hold_times_out(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    monkeypatch.setenv(pi.TRUNK_HOLD_S_ENV, "0.2")
    slots = Slots()
    slots.set(0, busy=True, n=12000, task=1, decoded=0)
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False,
                                 fork_caps=lambda url: KV_CAPS)
    trunk = _doc("O", 1200)
    owner = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    t0 = time.perf_counter()
    b = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=5, poll_s=0.01)
    assert b is not None and time.perf_counter() - t0 < 3
    assert pool.admission_record(b)["prefix_index"]["trunk_hold_timeout"] is True
    assert pi.get_index(URL).stats["trunk_hold_timeouts"] == 1
    pool.release(URL, b)
    pool.release(URL, owner)


# ── wiring 1: the passthrough chat route end to end ────────────────────────────


@pytest.fixture
def passthrough_env(monkeypatch, tmp_path):
    from tests.unit.test_passthrough_route import FakeLlamaServer
    from src.api.routes import passthrough as pt

    srv = FakeLlamaServer()
    slots = Slots()
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False)
    monkeypatch.setattr(kpa, "_shared_pool_admission", pool)
    monkeypatch.setattr(pt, "_server_urls", lambda: {"architect_critic": srv.url})
    monkeypatch.setattr(pt, "_context_limit", lambda call: ContextLimit(
        url=srv.url, per_request_n_ctx=196608, total_slots=4, kv_unified=True,
        source="registry"))
    log = tmp_path / "serving_calls.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", str(log))
    monkeypatch.delenv(pt.PASSTHROUGH_ENV, raising=False)
    monkeypatch.delenv(pt.PASSTHROUGH_ROLES_ENV, raising=False)
    yield SimpleNamespace(srv=srv, slots=slots, pool=pool, log=log)
    srv.close()


def test_passthrough_route_observes_binds_and_predicts(passthrough_env):
    from fastapi.testclient import TestClient

    from src.api import app

    env = passthrough_env
    doc = _doc("R", 1000)
    first = [{"role": "system", "content": doc}, {"role": "user", "content": "q1"}]
    with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as c:
        r = c.post("/v1/passthrough/architect_critic/chat/completions", json={"messages": first})
        assert r.status_code == 200
        idx = pi.peek_index(env.srv.url)
        assert idx is not None and idx.status()["entries"] == {"pending": 1}
        # The server left the call in slot 2: prompt_n + cache_n + predicted_n tokens.
        env.slots.set(2, n=40 + 12 + 3, task=11)
        r = c.post("/v1/passthrough/architect_critic/chat/completions", json={
            "messages": first + [{"role": "assistant", "content": "a1"},
                                 {"role": "user", "content": "q2"}]})
        assert r.status_code == 200
    rec1, rec2 = [json.loads(x) for x in env.log.read_text().splitlines() if x.strip()]
    assert rec1["kv_admission"]["prefix_index"]["match"]["source"] is None
    pidx = rec2["kv_admission"]["prefix_index"]
    assert pidx["match"]["source"] == "slot_idle" and pidx["predicted_slot"] == 2
    assert idx.stats["bound_inferred"] == 1
    assert idx.stats["predictions"] == 2  # both records compared with timings.cache_n
    # the forwarded body is the client's own bytes (the index never edits a request)
    assert all(b"id_slot" not in raw for _p, raw in env.srv.requests)


# ── wiring 2: the /chat pipeline's backend half, llm_call → /completion ─────────


def test_llm_call_through_caching_backend_feeds_and_uses_the_index(monkeypatch, tmp_path):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.inference.prefix_cache import CachingBackend, PrefixRouter
    from src.llm_primitives import LLMPrimitives

    slots = Slots()
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False)
    monkeypatch.setattr(kpa, "_shared_pool_admission", pool)
    set_context_limit_resolver(ContextLimitResolver(
        live=False, registry_facts=lambda: {8083: {"context_tokens": 393216, "slots": 4,
                                                   "kv_unified": True}},
        role_urls=lambda: {}))
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", str(tmp_path / "sc.jsonl"))
    payloads: list[dict] = []
    task = {"n": 20}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        task["n"] += 1
        prompt_tokens = len(body["prompt"]) // 4
        slot = body.get("id_slot", 1)
        slots.set(slot, n=prompt_tokens + 5, task=task["n"])
        final = {
            "content": "" if body.get("stream") else "ok", "tokens_predicted": 5,
            "tokens_evaluated": prompt_tokens, "id_slot": slot, "stop": True,
            "stop_type": "eos",
            "timings": {"prompt_n": prompt_tokens, "cache_n": 0, "predicted_n": 5,
                        "prompt_ms": 1.0, "predicted_ms": 1.0, "predicted_per_second": 5.0},
        }
        if body.get("stream"):  # the primitives lane streams /completion (SSE)
            sse = (f"data: {json.dumps({'content': 'ok', 'stop': False})}\n\n"
                   f"data: {json.dumps(final)}\n\n")
            return httpx.Response(200, text=sse, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json=final)

    lsb = LlamaServerBackend(ServerConfig(base_url=URL, num_slots=4))
    lsb.client = httpx.Client(base_url=URL, transport=httpx.MockTransport(handler))
    tracker = MagicMock()
    tracker.is_available.return_value = True
    prims = LLMPrimitives(mock_mode=False, server_urls={"coder_escalation": URL},
                          health_tracker=tracker)
    prims._backends["coder_escalation"] = CachingBackend(lsb, PrefixRouter(num_slots=4))
    trunk = _doc("W", 300)
    try:
        assert prims.llm_call(trunk + "first", role="coder_escalation", n_tokens=8) == "ok"
        idx = pi.peek_index(URL)
        assert idx is not None
        assert prims.llm_call(trunk + "second", role="coder_escalation", n_tokens=8) == "ok"
    finally:
        set_context_limit_resolver(None)
    assert idx.stats["bound_exact"] == 1          # call 1, bound at call 2's admission
    idx.reconcile(slots(URL))                     # call 2 left slot 1 with new content
    assert idx.status()["slots"]["1"]["origin"] == "exact" and idx.stats["bound_exact"] == 2
    assert idx.stats["matches"].get("slot_idle") == 1  # the second call found the first
    records = [json.loads(x) for x in (tmp_path / "sc.jsonl").read_text().splitlines()]
    second = [r for r in records if r.get("dispatched")][-1]
    assert second["kv_admission"]["prefix_index"]["predicted_slot"] == 1
    # default pin policy is off: the payload carries no id_slot
    assert all("id_slot" not in p for p in payloads)
