""":8083 KV-pool decision step 1 (2026-10-03): one long prefill at a time per server,
scouts admitted by tokens through the shared-pool gate, and a config-derived
per-request context cap.

Offline and inference-free: occupancy, transports and backends are fakes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.backends.context_limits import (
    ContextLimit,
    ContextLimitResolver,
    PoolOccupancy,
    SlotState,
    limit_from_registry,
    parse_props,
    parse_slots,
    registry_facts_by_port,
    set_context_limit_resolver,
)
from src.exceptions import ContextOverflowError
from src.scheduling import kv_pool_admission as kpa
from src.scheduling.kv_pool_admission import KVPoolQueueFull, SharedKVPoolAdmission

URL = "http://localhost:8083"
POOL = 196608


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (kpa.KV_POOL_LONG_PREFILL_ENV, kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV,
                 kpa.KV_POOL_LONG_PREFILL_OBSERVE_ENV, kpa.KV_POOL_MAX_QUEUED_ENV,
                 kpa.KV_POOL_RATIO_INIT_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fresh_pool(monkeypatch):
    pool = SharedKVPoolAdmission(occupancy=lambda url: None)
    monkeypatch.setattr(kpa, "_shared_pool_admission", pool)
    return pool


def _wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


# ── the one-long-prefill rule ─────────────────────────────────────────────────────────────


class TestLongPrefillRule:
    def test_default_threshold_is_16k_and_zero_disables(self, monkeypatch):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        assert pool.long_prefill_threshold() == 16384
        assert pool.is_long_prefill(16384) and not pool.is_long_prefill(16383)
        monkeypatch.setenv(kpa.KV_POOL_LONG_PREFILL_ENV, "0")
        assert not pool.is_long_prefill(500_000)
        first = pool.acquire(URL, 40_000, POOL, timeout_s=0)
        assert first is not None
        assert pool.acquire(URL, 40_000, POOL, timeout_s=0) is not None, "rule off: tokens only"

    def test_second_long_waits_while_first_prefills_then_times_out(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        first = pool.acquire(URL, 40_000, POOL, timeout_s=0)
        assert first is not None and pool.long_prefill_holder(URL) == first
        t0 = time.perf_counter()
        # Fits by tokens (80k of 196k) but another long prefill is in flight.
        assert pool.acquire(URL, 40_000, POOL, timeout_s=0.05, poll_s=0.01) is None
        assert time.perf_counter() - t0 < 2.0
        stats = pool.get_status()[URL]["admission_stats"]
        assert stats["abandoned"] == 1
        # A request's own deadline bounds the wait the same way.
        assert pool.acquire(URL, 40_000, POOL, deadline_s=time.perf_counter() + 0.05,
                            poll_s=0.01) is None
        # Cancellation ends the wait too.
        assert pool.acquire(URL, 40_000, POOL, timeout_s=5, cancel_check=lambda: True,
                            poll_s=0.01) is None

    def test_prefill_done_hands_the_lease_on(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        first = pool.acquire(URL, 40_000, POOL, timeout_s=0)
        got = []
        th = threading.Thread(target=lambda: got.append(
            pool.acquire(URL, 30_000, POOL, timeout_s=5, poll_s=0.01)))
        th.start()
        assert _wait_until(lambda: pool.queued(URL) == 1)
        time.sleep(0.05)
        assert not got, "must wait while the first long request is in prefill"
        pool.prefill_done(URL, first)          # first output chunk seen
        th.join(timeout=5)
        assert got and got[0] is not None
        assert pool.long_prefill_holder(URL) == got[0]
        # Both now decode concurrently: the rule never serialises decode.
        assert pool.get_status()[URL]["in_flight"] == 2
        st = pool.get_status()[URL]["admission_stats"]
        assert st["long_prefill_waits"] == 1 and st["long_prefill_wait_s_total"] > 0
        assert st["queued_admissions"] == 1 and st["long_prefill_admitted"] == 2

    def test_release_also_frees_the_lease(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        first = pool.acquire(URL, 40_000, POOL, timeout_s=0)
        pool.release(URL, first)
        assert pool.long_prefill_holder(URL) is None
        assert pool.acquire(URL, 40_000, POOL, timeout_s=0) is not None

    def test_prefill_done_for_another_ticket_is_harmless(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        first = pool.acquire(URL, 40_000, POOL, timeout_s=0)
        pool.prefill_done(URL, first + 999)
        pool.prefill_done(URL, None)
        assert pool.long_prefill_holder(URL) == first

    def test_lease_expires_at_the_prefill_rate_floor(self, monkeypatch):
        now = [1000.0]
        pool = SharedKVPoolAdmission(occupancy=lambda u: None, clock=lambda: now[0])
        monkeypatch.setenv(kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV, "250")
        first = pool.acquire(URL, 50_000, POOL, timeout_s=0)   # 50k / 250 = 200 s
        assert first is not None
        now[0] += 199.0
        assert pool.long_prefill_holder(URL) == first
        assert pool.acquire(URL, 20_000, POOL, timeout_s=0) is None
        now[0] += 1.5
        assert pool.long_prefill_holder(URL) is None, "a batch caller never reports prefill"
        assert pool.acquire(URL, 20_000, POOL, timeout_s=0) is not None

    def test_floor_zero_holds_until_prefill_done_or_release(self, monkeypatch):
        now = [0.0]
        monkeypatch.setenv(kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV, "0")
        pool = SharedKVPoolAdmission(occupancy=lambda u: None, clock=lambda: now[0])
        first = pool.acquire(URL, 50_000, POOL, timeout_s=0)
        now[0] += 1e6
        assert pool.long_prefill_holder(URL) == first

    def test_short_requests_are_not_held_by_the_lease(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        pool.acquire(URL, 40_000, POOL, timeout_s=0)
        assert pool.acquire(URL, 2_000, POOL, timeout_s=0) is not None

    def test_short_request_passes_a_long_one_held_only_by_the_lease(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        first = pool.acquire(URL, 40_000, POOL, timeout_s=0)
        got = []
        th = threading.Thread(target=lambda: got.append(
            pool.acquire(URL, 30_000, POOL, timeout_s=5, poll_s=0.01)))
        th.start()
        assert _wait_until(lambda: pool.queued(URL) == 1)
        short = pool.acquire(URL, 2_000, POOL, timeout_s=1, poll_s=0.01)
        assert short is not None, "short traffic keeps moving past a lease-held long request"
        assert pool.get_status()[URL]["admission_stats"]["short_passed_long"] == 1
        pool.release(URL, first)
        th.join(timeout=5)
        assert got and got[0] is not None

    def test_short_cannot_pass_if_it_would_eat_the_waiting_long_reservation(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        # 100k in flight (long, lease held); a 90k long waits; pool 196,608.
        first = pool.acquire(URL, 100_000, POOL, timeout_s=0)
        th = threading.Thread(target=lambda: pool.acquire(URL, 90_000, POOL, timeout_s=0.5,
                                                          poll_s=0.01))
        th.start()
        assert _wait_until(lambda: pool.queued(URL) == 1)
        # 10k fits beside the 100k alone, but not beside 100k + the queued 90k.
        assert pool.acquire(URL, 10_000, POOL, timeout_s=0.05, poll_s=0.01) is None
        th.join(timeout=5)
        pool.release(URL, first)

    def test_short_never_passes_a_long_one_waiting_for_TOKENS(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        first = pool.acquire(URL, 150_000, POOL, timeout_s=0)
        pool.prefill_done(URL, first)       # its prefill is over: decode holds the tokens
        assert pool.long_prefill_holder(URL) is None
        th = threading.Thread(target=lambda: pool.acquire(URL, 100_000, POOL, timeout_s=0.5,
                                                          poll_s=0.01))
        th.start()
        assert _wait_until(lambda: pool.queued(URL) == 1)
        assert pool.acquire(URL, 1_000, POOL, timeout_s=0.05, poll_s=0.01) is None, "FCFS"
        th.join(timeout=5)
        pool.release(URL, first)

    def test_rule_is_per_server(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        pool.acquire(URL, 40_000, POOL, timeout_s=0)
        assert pool.acquire("http://localhost:8070", 40_000, POOL, timeout_s=0) is not None

    def test_queue_bound_still_applies_to_long_waiters(self, monkeypatch):
        monkeypatch.setenv(kpa.KV_POOL_MAX_QUEUED_ENV, "1")
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        pool.acquire(URL, 40_000, POOL, timeout_s=0)
        th = threading.Thread(target=lambda: pool.acquire(URL, 40_000, POOL, timeout_s=0.5,
                                                          poll_s=0.01))
        th.start()
        assert _wait_until(lambda: pool.queued(URL) == 1)
        with pytest.raises(KVPoolQueueFull):
            pool.acquire(URL, 40_000, POOL, timeout_s=0)
        th.join(timeout=5)


class TestLongPrefillConcurrency:
    def test_many_concurrent_long_requests_prefill_one_at_a_time_but_decode_together(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        lock = threading.Lock()
        state = {"prefilling": 0, "peak_prefill": 0, "admitted": 0, "peak_admitted": 0}
        errors: list[BaseException] = []

        def request():
            try:
                t = pool.acquire(URL, 30_000, POOL, timeout_s=10, poll_s=0.002)
                assert t is not None
                with lock:
                    state["prefilling"] += 1
                    state["admitted"] += 1
                    state["peak_prefill"] = max(state["peak_prefill"], state["prefilling"])
                    state["peak_admitted"] = max(state["peak_admitted"], state["admitted"])
                time.sleep(0.03)                       # prefill
                with lock:
                    state["prefilling"] -= 1
                pool.prefill_done(URL, t)              # first chunk
                time.sleep(0.15)                       # decode
                with lock:
                    state["admitted"] -= 1
                pool.release(URL, t)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=request) for _ in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=20)
        assert not errors
        assert state["peak_prefill"] == 1, "never two long prefills at once on one server"
        assert state["peak_admitted"] >= 2, "decode of long requests still overlaps"
        st = pool.get_status()[URL]["admission_stats"]
        assert st["admitted"] == 5 and st["long_prefill_admitted"] == 5
        assert st["long_prefill_waits"] >= 1

    def test_mixed_traffic_short_requests_flow_during_a_long_prefill(self):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        first = pool.acquire(URL, 60_000, POOL, timeout_s=0)
        long_waiter = []
        th = threading.Thread(target=lambda: long_waiter.append(
            pool.acquire(URL, 60_000, POOL, timeout_s=5, poll_s=0.005)))
        th.start()
        assert _wait_until(lambda: pool.queued(URL) == 1)
        shorts = []
        sth = [threading.Thread(target=lambda: shorts.append(
            pool.acquire(URL, 1_500, POOL, timeout_s=2, poll_s=0.005))) for _ in range(3)]
        for t in sth:
            t.start()
        for t in sth:
            t.join(timeout=5)
        assert len(shorts) == 3 and all(s is not None for s in shorts)
        assert not long_waiter, "the second long prefill still waits"
        pool.prefill_done(URL, first)
        th.join(timeout=5)
        assert long_waiter and long_waiter[0] is not None


class TestObservedLongPrefill:
    @staticmethod
    def _occ(*, decoded, processed, processing=True):
        return PoolOccupancy(url=URL, slots=(
            SlotState(slot_id=0, n_ctx=POOL, is_processing=processing, n_prompt_tokens=processed,
                      n_remain=None, n_decoded=decoded, n_prompt_tokens_processed=processed),
            SlotState(slot_id=1, n_ctx=POOL, is_processing=False, n_prompt_tokens=0, n_remain=None),
        ))

    def test_parse_slots_reads_prefill_progress(self):
        body = [{"id": 0, "n_ctx": POOL, "is_processing": True, "n_prompt_tokens": 9000,
                 "n_prompt_tokens_processed": 8000, "n_prompt_tokens_cache": 1000,
                 "next_token": [{"n_remain": 512, "n_decoded": 0}]},
                {"id": 1, "n_ctx": POOL, "is_processing": False}]
        occ = parse_slots(URL, body)
        assert occ.slots[0].prefilling and occ.slots[0].n_prompt_tokens_processed == 8000
        assert not occ.slots[1].prefilling and occ.slots[1].n_decoded is None
        assert occ.long_prefills(4096) == 1 and occ.long_prefills(8001) == 0
        assert occ.long_prefills(0) == 0

    def test_an_observed_long_prefill_from_another_client_holds_long_requests(self):
        occ = self._occ(decoded=0, processed=12_000)       # someone else, mid-prefill
        pool = SharedKVPoolAdmission(occupancy=lambda u: occ)
        assert pool.acquire(URL, 40_000, POOL, timeout_s=0.05, poll_s=0.01) is None
        assert pool.acquire(URL, 2_000, POOL, timeout_s=0) is not None, "short is not held"

    def test_decoding_or_short_observed_prefills_do_not_hold(self):
        for occ in (self._occ(decoded=5, processed=50_000),   # decoding now
                    self._occ(decoded=0, processed=1_000)):   # prefill still short
            pool = SharedKVPoolAdmission(occupancy=lambda u, o=occ: o)
            ticket = pool.acquire(URL, 40_000, POOL, timeout_s=0)
            assert ticket is not None
            # Each instance stands for a worker and the lease is host-wide
            # (KVU-15a): let it go before the next "worker" asks.
            pool.release(URL, ticket)

    def test_observation_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv(kpa.KV_POOL_LONG_PREFILL_OBSERVE_ENV, "0")
        occ = self._occ(decoded=0, processed=50_000)
        pool = SharedKVPoolAdmission(occupancy=lambda u: occ)
        assert pool.acquire(URL, 40_000, POOL, timeout_s=0) is not None


# ── scouts admit by tokens ────────────────────────────────────────────────────────────────


class _Resolver:
    """Free slots on /slots and a shared-pool (or not) ContextLimit for the URL."""

    def __init__(self, *, pool_tokens=POOL, shared=True, total=4):
        self.total = total
        self.limit = ContextLimit(URL, pool_tokens, total, shared, "registry")

    def pool_occupancy(self, url):
        return PoolOccupancy(url=url, slots=tuple(
            SlotState(slot_id=i, n_ctx=self.limit.per_request_n_ctx, is_processing=False,
                      n_prompt_tokens=0, n_remain=None) for i in range(self.total)))

    def limit_for_url(self, url):
        return self.limit


class _SpyPool(SharedKVPoolAdmission):
    def __init__(self):
        super().__init__(occupancy=lambda u: None)
        self.acquired: list[tuple[str, int, int, int]] = []
        self.released: list[int] = []

    def acquire(self, url, tokens, pool_tokens, **kw):
        self.acquired.append((url, int(tokens), int(pool_tokens), int(kw.get("max_new_tokens", 0))))
        return super().acquire(url, tokens, pool_tokens, **kw)

    def release(self, url, ticket, *, success=True):
        self.released.append(ticket)
        return super().release(url, ticket, success=success)


def _scout_lane(tmp_path):
    from src.repl_environment import task_root as TR

    root = tmp_path / "lane"
    (root / "src").mkdir(parents=True)
    (root / "src" / "k.c").write_text("void hot(void) {\n  int x = 0;\n}\n")
    return TR.TaskScope(root=str(root.resolve()), edit_mode=TR.EDIT_MODE_NONE)


class _Transport:
    name = "fake"

    def __init__(self, *, pool=None, delay_s=0.0):
        self.lock = threading.Lock()
        self.calls = 0
        self.pool = pool
        self.delay_s = delay_s
        self.reserved_seen: list[int] = []

    def complete(self, messages, *, max_tokens, should_stop, timeout_s):
        from src.api.routes.chat_pipeline import scout_stage as S

        with self.lock:
            self.calls += 1
            if self.pool is not None:
                self.reserved_seen.append(self.pool.in_flight_tokens(URL))
        if self.delay_s:
            time.sleep(self.delay_s)
        return S.CompletionResult("SUMMARY\nhot at src/k.c:1", prompt_tokens=10, completion_tokens=5)


def _spec(n):
    return {"enabled": True, "max": 4, "max_turns": 2, "summary_tokens": 400, "budget_s": 10.0,
            "reserve_slots": 1,
            "targets": [{"file": "src/k.c", "label": f"T-{i}", "share": 0.1} for i in range(n)]}


class TestScoutsAdmitByTokens:
    def test_every_scout_call_reserves_its_estimated_prompt_on_the_gate(self, tmp_path):
        from src.api.routes.chat_pipeline import scout_stage as S

        pool = _SpyPool()
        transport = _Transport(pool=pool)
        stage = asyncio.run(S.run_scouts(_spec(2), scope=_scout_lane(tmp_path), role="architect_critic",
                                         url=URL, transport=transport, resolver=_Resolver(),
                                         pool_admission=pool))
        assert stage.report["completed"] == 2
        assert stage.report["pool_gate"]["gated"] is True
        assert stage.report["pool_gate"]["admitted_calls"] == 2 == len(pool.acquired)
        for url, tokens, pool_tokens, max_new in pool.acquired:
            assert url == URL and pool_tokens == POOL
            assert tokens > 50, "token estimate of the real messages, not a slot count"
            assert max_new == 400 + 256
        assert all(seen > 0 for seen in transport.reserved_seen), "reserved during the call"
        assert len(pool.released) == 2 and pool.in_flight_tokens(URL) == 0

    def test_scouts_wait_for_tokens_not_slots(self, tmp_path):
        """Slots are free, but the pool is nearly full: scouts queue on tokens and
        time out instead of oversubscribing it."""
        from src.api.routes.chat_pipeline import scout_stage as S

        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        held = pool.acquire(URL, POOL - 500, POOL, timeout_s=0)   # someone holds ~all of it
        transport = _Transport()
        spec = _spec(2)
        spec["budget_s"] = 0.6
        stage = asyncio.run(S.run_scouts(spec, scope=_scout_lane(tmp_path), role="architect_critic",
                                         url=URL, transport=transport, resolver=_Resolver(),
                                         pool_admission=pool))
        assert stage.report["cap"]["cap"] == 2, "the slot cap alone would have run both"
        assert transport.calls == 0, "never dispatched while the pool is full"
        assert all(r.status in ("timeout", "cancelled") for r in stage.results)
        pool.release(URL, held)

    def test_token_gate_serialises_scouts_that_do_not_fit_together(self, tmp_path):
        from src.api.routes.chat_pipeline import scout_stage as S

        # Pool fits ONE scout reservation (prompt + 656 decode) but not two.
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        transport = _Transport(pool=pool, delay_s=0.1)
        stage = asyncio.run(S.run_scouts(_spec(3), scope=_scout_lane(tmp_path), role="architect_critic",
                                         url=URL, transport=transport,
                                         resolver=_Resolver(pool_tokens=1500), pool_admission=pool))
        assert stage.report["completed"] == 3
        assert stage.report["max_inflight_calls"] == 1, "gate waits are not in-flight calls"

    def test_non_shared_pool_is_not_gated(self, tmp_path):
        from src.api.routes.chat_pipeline import scout_stage as S

        pool = _SpyPool()
        stage = asyncio.run(S.run_scouts(_spec(1), scope=_scout_lane(tmp_path), role="frontdoor",
                                         url=URL, transport=_Transport(),
                                         resolver=_Resolver(shared=False), pool_admission=pool))
        assert stage.report["completed"] == 1 and stage.report["pool_gate"]["gated"] is False
        assert pool.acquired == []

    def test_default_pool_is_the_process_gate(self, tmp_path, fresh_pool):
        from src.api.routes.chat_pipeline import scout_stage as S

        stage = asyncio.run(S.run_scouts(_spec(1), scope=_scout_lane(tmp_path), role="architect_critic",
                                         url=URL, transport=_Transport(), resolver=_Resolver()))
        assert stage.report["pool_gate"]["admitted_calls"] == 1
        assert fresh_pool.get_status()[URL]["admission_stats"]["admitted"] == 1


# ── per-request context cap from config ───────────────────────────────────────────────────


class TestPerRequestCap:
    def test_today_196608_is_below_the_model_cap(self):
        lim = parse_props(URL, {"default_generation_settings": {"n_ctx": 196608}, "total_slots": 4},
                          {"context_tokens": 196608, "slots": 4, "kv_unified": True, "ctx_max": 262144})
        assert lim.per_request_n_ctx == 196608 and lim.pool_tokens == 196608
        assert not lim.cap_binding

    def test_after_relaunch_at_393216_the_cap_is_262144_and_the_pool_stays_whole(self):
        lim = parse_props(URL, {"default_generation_settings": {"n_ctx": 393216}, "total_slots": 4},
                          {"context_tokens": 393216, "slots": 4, "kv_unified": True, "ctx_max": 262144})
        assert lim.kv_unified and lim.shared_pool
        assert lim.per_request_n_ctx == 262144 and lim.cap_binding
        assert lim.pool_tokens == 393216 and lim.slot_n_ctx == 393216
        reg = limit_from_registry(URL, {"context_tokens": 393216, "slots": 4, "kv_unified": True,
                                        "ctx_max": 262144})
        assert (reg.per_request_n_ctx, reg.pool_tokens) == (262144, 393216)
        assert lim.to_dict()["request_cap"] == 262144

    def test_no_declared_cap_leaves_the_server_limit(self):
        reg = limit_from_registry(URL, {"context_tokens": 393216, "slots": 4, "kv_unified": True})
        assert reg.per_request_n_ctx == 393216 and reg.request_cap is None

    def test_observe_keeps_the_cap(self):
        r = ContextLimitResolver(live=False, role_urls=lambda: {}, registry_facts=lambda: {
            8083: {"context_tokens": 393216, "slots": 4, "kv_unified": True, "ctx_max": 262144}})
        assert r.limit_for_url(URL).per_request_n_ctx == 262144
        r.observe(URL, n_ctx=300_000)
        lim = r.limit_for_url(URL)
        assert lim.per_request_n_ctx == 262144 and lim.slot_n_ctx == 300_000

    def test_the_cap_comes_from_the_compiled_stack_priors(self):
        from pathlib import Path

        import yaml

        facts = registry_facts_by_port()
        assert facts, "compiled stack priors present in the tree"
        assert facts[8083]["ctx_max"] == 262144
        # The pool is whatever the lean declares for the :8083 host — recomputed, not pinned
        # (was the literal 196608; STACKCHG-KVPOOL-20261003 raised it to 393216).
        lean = yaml.safe_load((Path(__file__).resolve().parents[2]
                               / "orchestration" / "model_registry.yaml").read_text())
        declared = lean["server_mode"]["architect_critic"]["serving_shape"]["n_ctx"]
        assert facts[8083]["context_tokens"] == declared

    def _prims(self, monkeypatch, facts):
        from src.llm_primitives import LLMPrimitives

        monkeypatch.setenv("ORCHESTRATOR_CTX_POOL_BACKOFF_S", "0")
        monkeypatch.setenv("ORCHESTRATOR_CONTEXT_OVERFLOW_ROLES", "")
        set_context_limit_resolver(ContextLimitResolver(live=False, registry_facts=lambda: facts,
                                                        role_urls=lambda: {}))
        tracker = MagicMock()
        tracker.is_available.return_value = True
        return LLMPrimitives(mock_mode=False, server_urls={"architect_critic": URL},
                             health_tracker=tracker)

    def test_inference_refuses_a_prompt_over_a_binding_cap_before_dispatch(self, monkeypatch, fresh_pool):
        prims = self._prims(monkeypatch, {8083: {"context_tokens": 393216, "slots": 4,
                                                 "kv_unified": True, "ctx_max": 262144}})
        backend = MagicMock(spec=[])
        backend.infer = MagicMock(side_effect=AssertionError("must not reach the server"))
        prims._backends["architect_critic"] = backend
        with pytest.raises(ContextOverflowError) as ei:
            prims._real_call("x" * (4 * 270_000), "architect_critic", n_tokens=100)
        assert ei.value.kind == "request_too_large" and ei.value.source == "request_cap"
        assert ei.value.n_ctx == 262144
        assert fresh_pool.get_status() == {}, "refused before the pool gate"

    def test_no_refusal_when_the_server_limit_binds(self, monkeypatch, fresh_pool):
        from src.model_server import InferenceResult

        prims = self._prims(monkeypatch, {8083: {"context_tokens": 196608, "slots": 4,
                                                 "kv_unified": True, "ctx_max": 262144}})
        backend = MagicMock(spec=[])
        backend.infer = MagicMock(return_value=InferenceResult(
            role="r", output="ok", tokens_generated=1, generation_speed=1.0, elapsed_time=0.1,
            success=True))
        prims._backends["architect_critic"] = backend
        assert prims._real_call("x" * 400, "architect_critic", n_tokens=100) == "ok"


# ── inference hands the lease on at the first streamed chunk ─────────────────────────────


class TestInferencePrefillDone:
    def test_first_stream_chunk_releases_the_long_prefill_lease(self, monkeypatch, fresh_pool):
        from src.llm_primitives import LLMPrimitives
        from src.model_server import InferenceResult

        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_PREFILL_FLOOR_TPS", "0")   # no expiry
        set_context_limit_resolver(ContextLimitResolver(live=False, role_urls=lambda: {},
            registry_facts=lambda: {8083: {"context_tokens": 196608, "slots": 4,
                                           "kv_unified": True, "ctx_max": 262144}}))
        tracker = MagicMock()
        tracker.is_available.return_value = True
        prims = LLMPrimitives(mock_mode=False, server_urls={"architect_critic": URL},
                              health_tracker=tracker)
        seen = SimpleNamespace(holder_before=None, holder_after=None)

        def stream(role_config, request, on_chunk):
            seen.holder_before = fresh_pool.long_prefill_holder(URL)
            on_chunk("first")
            seen.holder_after = fresh_pool.long_prefill_holder(URL)
            return InferenceResult(role="r", output="first", tokens_generated=1,
                                   generation_speed=1.0, elapsed_time=0.1, success=True)

        backend = MagicMock(spec=[])
        backend.infer_stream_text = MagicMock(side_effect=stream)
        prims._backends["architect_critic"] = backend
        assert prims._real_call("x" * (3 * 20_000), "architect_critic", n_tokens=100) == "first"
        assert seen.holder_before is not None, "a 20k-token prompt took the long-prefill lease"
        assert seen.holder_after is None, "the first chunk handed it on"
        assert fresh_pool.in_flight_tokens(URL) == 0
