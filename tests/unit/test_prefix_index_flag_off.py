"""RTG-58 P2 — with ``ORCHESTRATOR_PREFIX_INDEX`` off, behaviour is unchanged.

The proof is structural and end to end:

* constructing a ``PrefixIndex`` raises in every test here, so any flag-off path
  that reached the index would fail loudly;
* the callers do not even compute a prefix key (``{}`` / ``None``);
* the admission record and the gate status carry exactly the pre-RTG-58 keys;
* the wire payload equals what ``LlamaServerBackend._build_payload`` builds for
  the same request (no ``id_slot``, nothing added), the passthrough forwards the
  client's raw bytes, and the serving records mention no prefix index;
* no ledger file is written.
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

# The admission record and gate status keys as of origin/main f8c9c0a3 (pre-RTG-58).
LEGACY_RECORD_KEYS = {
    "prompt_tokens_est", "cache_credit_source", "cache_credited_tokens",
    "cache_credit_prefix_tokens_est", "cache_credit_matched_chars",
    "cache_credit_unavailable", "prefill_tokens_est", "long_prefill", "cache_credited",
    "long_prefill_threshold", "queue_wait_ms",
}
# Present when the KVU-15c history credited the call (legacy behaviour).
LEGACY_HISTORY_KEYS = {"cache_credit_history_age_s", "cache_credit_history_tokens_since"}
LEGACY_STATUS_KEYS = {
    "reserved_tokens", "in_flight", "queued", "new_token_ratio", "long_prefill_threshold",
    "long_prefill_lease_held", "long_prefill_lease_held_here",
    "long_prefill_lease_cross_process", "queued_long_prefills", "admission_stats",
}
LEGACY_STAT_KEYS = {
    "admitted", "queued_admissions", "queue_wait_s_total", "queue_wait_s_max",
    "long_prefill_admitted", "long_prefill_waits", "long_prefill_wait_s_total",
    "short_passed_long", "abandoned", "cache_credited", "cache_credit_fp_history",
    "cache_credit_slots_text", "cache_credit_unavailable",
}


@pytest.fixture(autouse=True)
def flag_off(monkeypatch, tmp_path):
    from src.runtime import long_prefill_lease

    monkeypatch.delenv(pi.FLAG_ENV, raising=False)
    # Sub-flags ON: they must be inert while the master flag is off.
    monkeypatch.setenv(pi.FORK_ENV, "1")
    monkeypatch.setenv(pi.PIN_ENV, "idle")
    monkeypatch.setattr(long_prefill_lease, "lease_dir", lambda: tmp_path)

    # The hooks swallow every exception by contract (they must never fail a
    # request), so raising alone would prove nothing: record the attempt too
    # and assert on it after the test.
    attempts: list[tuple] = []

    def _boom(*_a, **_k):
        attempts.append(_a[1:])
        raise AssertionError("a PrefixIndex was constructed with the flag off")

    monkeypatch.setattr(pi.PrefixIndex, "__init__", _boom)
    pi.reset_indexes()
    yield tmp_path
    pi.reset_indexes()
    assert not attempts, f"PrefixIndex constructed with the flag off: {attempts}"
    assert not list(tmp_path.glob(f"{pi.FILE_PREFIX}*")), "ledger written with the flag off"


def _occ(url: str) -> PoolOccupancy:
    return PoolOccupancy(url=url, slots=(
        SlotState(slot_id=0, n_ctx=98304, is_processing=False, n_prompt_tokens=500,
                  n_remain=None, n_decoded=1, id_task=3),
        SlotState(slot_id=1, n_ctx=98304, is_processing=False, n_prompt_tokens=0,
                  n_remain=None, n_decoded=None, id_task=None),
    ))


def test_flags_and_callers_compute_nothing():
    from src.api.routes import passthrough as pt
    from src.api.routes.chat_pipeline import scout_stage
    from src.llm_primitives.inference import _prefix_index_kwargs

    assert not pi.enabled() and not pi.fork_enabled() and not pi.lpm_enabled()
    assert pi.pin_policy() == "off"
    req = SimpleNamespace(prompt="x" * 5000, chat_payload=None)
    assert _prefix_index_kwargs(req) == {}
    assert pt.wire_prefix_key("chat/completions", {"messages": [{"role": "user"}]}) is None
    assert scout_stage._scout_prefix_key([{"role": "user", "content": "hi"}]) == {}


def test_admission_record_and_status_are_the_legacy_shape():
    pool = SharedKVPoolAdmission(occupancy=_occ, cross_process=False, history=SimpleNamespace(
        lookup=lambda url, ladder: (None, "no_history")))
    # Even a caller that passes a key cannot switch the index on.
    a = pool.acquire(URL, 1000, 393216, prefix_key="k" * 9000, timeout_s=0)
    b = pool.acquire(URL, 1000, 393216, timeout_s=0)
    assert set(pool.admission_record(a)) == LEGACY_RECORD_KEYS
    assert set(pool.admission_record(b)) == LEGACY_RECORD_KEYS
    pool.prefill_done(URL, a)
    status = pool.get_status()[URL]
    assert set(status) == LEGACY_STATUS_KEYS
    assert set(status["admission_stats"]) == LEGACY_STAT_KEYS
    assert pool._lpm_score == {} and pool._trunk_held == {} and pool._trunk_since == {}
    pool.release(URL, a)
    pool.release(URL, b)
    assert pi.peek_index(URL) is None
    assert pool._fork_source == {}


def test_review_fix_surfaces_are_inert_with_the_flag_off(monkeypatch):
    """KPF-27e / review fixes: the server-fork fields change nothing with the
    flag off — no /props read, no unique-cell projection, no checkpoint_at."""
    import json

    import httpx

    from src.api.routes.chat_pipeline import scout_stage as S

    reads: list[str] = []
    occ = PoolOccupancy(url=URL, kv_pool={"size": 393216, "used": 61000, "shared": 60000}, slots=(
        SlotState(slot_id=0, n_ctx=1, is_processing=True, n_prompt_tokens=60000, n_remain=None,
                  kv_private=0, kv_shared=60000),
        SlotState(slot_id=1, n_ctx=1, is_processing=True, n_prompt_tokens=61000, n_remain=None,
                  kv_private=1000, kv_shared=60000)))
    pool = SharedKVPoolAdmission(occupancy=lambda url: occ, cross_process=False,
                                 fork_caps=lambda url: reads.append(url) or None)
    assert pool._observed(URL)[0] == 121000  # legacy sum, not the unique count
    t = pool.acquire(URL, 1000, 393216, prefix_key="k" * 9000, timeout_s=0)
    assert t is not None and reads == []
    pool.release(URL, t)
    caps_pool = SimpleNamespace(fork_caps=lambda url: {"min_tokens": 1, "mode": "checkpoint",
                                                      "checkpoint_at": True})
    msgs = [{"role": "system", "content": "s" * 90000}, {"role": "user", "content": "t"}]
    assert S._scout_checkpoint_at(caps_pool, URL, msgs) is None
    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    inner = S.ChatCompletionsTransport("http://x", client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    gate = S.PoolGatedTransport(inner, url=URL, limit=SimpleNamespace(pool_tokens=393216),
                                pool=SharedKVPoolAdmission(occupancy=lambda url: None,
                                                           cross_process=False))
    gate.complete(msgs, max_tokens=8, should_stop=lambda: False, timeout_s=5)
    assert set(seen["body"]) == {"messages", "max_tokens", "stream", "temperature",
                                 "cache_prompt", "stream_options", "chat_template_kwargs"}


def test_primitives_lane_payload_is_unchanged(monkeypatch, tmp_path):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.inference.prefix_cache import CachingBackend, PrefixRouter
    from src.llm_primitives import LLMPrimitives

    pool = SharedKVPoolAdmission(occupancy=_occ, cross_process=False)
    monkeypatch.setattr(kpa, "_shared_pool_admission", pool)
    set_context_limit_resolver(ContextLimitResolver(
        live=False, registry_facts=lambda: {8083: {"context_tokens": 393216, "slots": 4,
                                                   "kv_unified": True}},
        role_urls=lambda: {}))
    log = tmp_path / "sc.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", str(log))
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        final = {"content": "", "tokens_predicted": 2, "tokens_evaluated": 10, "id_slot": 0,
                 "stop": True, "stop_type": "eos",
                 "timings": {"prompt_n": 10, "cache_n": 0, "predicted_n": 2}}
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
    sent: list = []
    real_stream = lsb.infer_stream_text

    def spy(role_config, request, on_chunk=None):
        sent.append((role_config, request))
        return real_stream(role_config, request, on_chunk=on_chunk)

    monkeypatch.setattr(lsb, "infer_stream_text", spy)
    try:
        prompt = "shared trunk line\n" * 400 + "question"
        assert prims.llm_call(prompt, role="coder_escalation", n_tokens=8) == "ok"
        assert prims.llm_call(prompt + " again", role="coder_escalation", n_tokens=8) == "ok"
    finally:
        set_context_limit_resolver(None)
    assert len(payloads) == 2
    for (role_config, request), wire in zip(sent, payloads):
        assert request.slot_id is None
        expected = lsb._build_payload(role_config, request)
        expected["stream"] = True
        assert wire == expected
    records = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert records and "prefix_index" not in json.dumps(records)
    for rec in records:
        if "kv_admission" in rec:
            assert LEGACY_RECORD_KEYS <= set(rec["kv_admission"]) <= (
                LEGACY_RECORD_KEYS | LEGACY_HISTORY_KEYS)


def test_passthrough_forwards_raw_bytes_and_records_no_index(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from src.api import app
    from src.api.routes import passthrough as pt
    from tests.unit.test_passthrough_route import FakeLlamaServer

    srv = FakeLlamaServer()
    try:
        pool = SharedKVPoolAdmission(occupancy=_occ, cross_process=False)
        monkeypatch.setattr(kpa, "_shared_pool_admission", pool)
        monkeypatch.setattr(pt, "_server_urls", lambda: {"architect_critic": srv.url})
        monkeypatch.setattr(pt, "_context_limit", lambda call: ContextLimit(
            url=srv.url, per_request_n_ctx=196608, total_slots=4, kv_unified=True,
            source="registry"))
        log = tmp_path / "pt.jsonl"
        monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", str(log))
        monkeypatch.delenv(pt.PASSTHROUGH_ENV, raising=False)
        monkeypatch.delenv(pt.PASSTHROUGH_ROLES_ENV, raising=False)
        raw = json.dumps({"messages": [{"role": "system", "content": "doc " * 3000},
                                       {"role": "user", "content": "q"}]}).encode()
        with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as c:
            for _ in range(2):
                r = c.post("/v1/passthrough/architect_critic/chat/completions", content=raw,
                           headers={"Content-Type": "application/json"})
                assert r.status_code == 200
        assert [body for _p, body in srv.requests] == [raw, raw]
        records = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        assert len(records) == 2 and "prefix_index" not in json.dumps(records)
        assert all(LEGACY_RECORD_KEYS <= set(r["kv_admission"])
                   <= LEGACY_RECORD_KEYS | LEGACY_HISTORY_KEYS for r in records)
    finally:
        srv.close()
