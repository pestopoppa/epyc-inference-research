"""KVU-15c: the cached-prefix credit is re-based on orchestrator-side prefix
history (fingerprints of what this server already served), not on /slots text,
which llama-server v10 returns only under LLAMA_SERVER_SLOTS_DEBUG.

Offline and inference-free: occupancy is faked, the history is a fresh
``PrefixHistory`` with injected clock / --cache-ram / launch id, and its
host-wide file lives in the per-test hermetic lease dir.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.backends import serving_calls
from src.backends.context_limits import (
    ContextLimitResolver,
    PoolOccupancy,
    SlotState,
    estimate_tokens_conservative,
    set_context_limit_resolver,
)
from src.scheduling import kv_pool_admission as kpa
from src.scheduling import prefix_history as ph
from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

URL = "http://localhost:8083"
OTHER_URL = "http://localhost:8070"
POOL = 196608

SYSTEM = "<|im_start|>system\nYou are a careful coding agent.<|im_end|>\n"
HISTORY = SYSTEM + "".join(
    f"<|im_start|>user\nturn {i}: " + ("lorem ipsum dolor sit amet " * 200) + "<|im_end|>\n"
    for i in range(24))                     # ~130k chars, ~43k conservative tokens
NEW_TURN = "<|im_start|>user\nnext: please continue.<|im_end|>\n<|im_start|>assistant\n"
PROMPT = HISTORY + NEW_TURN


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (kpa.KV_POOL_LONG_PREFILL_ENV, kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV,
                 kpa.KV_POOL_LONG_PREFILL_OBSERVE_ENV, kpa.KV_POOL_MAX_QUEUED_ENV,
                 kpa.KV_POOL_RATIO_INIT_ENV, kpa.KV_POOL_CROSS_PROCESS_LEASE_ENV,
                 kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV, ph.HISTORY_ENV, ph.HOST_WIDE_ENV,
                 ph.WINDOW_S_ENV, ph.BYTES_PER_TOKEN_ENV, ph.MAX_ENTRIES_ENV):
        monkeypatch.delenv(name, raising=False)


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _history(*, clock=None, cache_ram=65536, launch="L1", host_wide=False, idle_residency=None):
    launch_ids = {"id": launch}
    hist = ph.PrefixHistory(
        clock=clock or Clock(),
        cache_ram_mib=lambda url: cache_ram,
        launch_id=lambda url: launch_ids["id"],
        host_wide=host_wide,
        # Injected like cache_ram, so these cases never read the tree's compiled
        # priors (STACKCHG-8083BATCH-20261004 added the idle-slot purge bound).
        idle_residency=lambda url: idle_residency,
    )
    hist.launch_ids = launch_ids  # test handle
    return hist


def _ladder(text):
    return serving_calls.prefix_ladder_for_text(text)


def _lease_held_elsewhere():
    holder = SharedKVPoolAdmission(occupancy=lambda u: None)
    assert holder.acquire(URL, 40_000, POOL, timeout_s=0) is not None
    return holder


WHOLE = estimate_tokens_conservative(PROMPT)


# ── the ladder ─────────────────────────────────────────────────────────────────────────────


class TestLadder:
    def test_ladder_agrees_with_the_record_fingerprint_at_the_record_depths(self):
        req = SimpleNamespace(prompt=PROMPT, chat_payload=None)
        fp = serving_calls.prefix_fingerprints(req)
        ladder = serving_calls.prefix_ladder(req)
        assert ladder["chars"] == fp["chars"] == len(PROMPT)
        for key, value in fp.items():
            if key != "chars":
                assert ladder["fp"][int(key[1:])] == value

    def test_ladder_is_dense_and_never_overshoots(self):
        depths = serving_calls.PREFIX_LADDER_DEPTHS_CHARS
        assert all(b / a <= 1.26 for a, b in zip(depths, depths[1:]))
        ladder = _ladder(PROMPT)
        assert max(ladder["fp"]) <= len(PROMPT)
        assert _ladder("") is None and _ladder("short") == {"chars": 5, "fp": {}}

    def test_chat_payload_ladder_is_the_json_of_tools_then_messages(self):
        payload = {"tools": [], "messages": [{"role": "user", "content": "x" * 5000}]}
        a = serving_calls.prefix_ladder(SimpleNamespace(chat_payload=payload, prompt="ignored"))
        text = serving_calls._prompt_text_for_fingerprint(SimpleNamespace(chat_payload=payload))
        assert a == _ladder(text)


# ── credit from history ───────────────────────────────────────────────────────────────────


class TestHistoryCredit:
    def test_a_request_extending_a_served_prompt_is_credited_and_not_held(self):
        hist = _history()
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000, generated_tokens=500)
        assert WHOLE >= 16384, "the whole prompt alone is a long prefill"
        _lease_held_elsewhere()
        no_slots = SharedKVPoolAdmission(occupancy=lambda u: None, history=hist)
        ticket = no_slots.acquire(URL, WHOLE, POOL, timeout_s=0, prefix_ladder=_ladder(PROMPT))
        assert ticket is not None, "only the new turn is prefill: admitted past the lease"
        assert no_slots.long_prefill_holder(URL) is None
        info = no_slots.admission_record(ticket)
        assert info["cache_credit_source"] == "fp_history"
        assert info["cache_credited"] is True and info["long_prefill"] is False
        matched = info["cache_credit_matched_chars"]
        assert 0.8 * len(HISTORY) <= matched <= len(HISTORY)
        # suffix at max(1/3, measured tokens/char) + the default margin, capped at whole
        rate = max(1 / 3, 40_000 / len(HISTORY))
        expected = min(WHOLE, math.ceil((len(PROMPT) - matched) * rate) + 4096)
        assert info["prefill_tokens_est"] == expected
        assert info["cache_credited_tokens"] == WHOLE - expected
        assert info["cache_credit_prefix_tokens_est"] == int(matched * 40_000 / len(HISTORY))
        assert info["cache_credit_unavailable"] is None
        stats = no_slots.get_status()[URL]["admission_stats"]
        assert stats["cache_credit_fp_history"] == 1 and stats["cache_credited"] == 1
        # KV reservation is never credited, and the record goes with the ticket.
        assert no_slots.in_flight_tokens(URL) >= WHOLE
        no_slots.release(URL, ticket)
        assert no_slots.admission_record(ticket) is None

    def test_no_credit_across_ports(self):
        hist = _history()
        hist.remember(OTHER_URL, _ladder(HISTORY), prompt_tokens=40_000)
        _lease_held_elsewhere()
        pool = SharedKVPoolAdmission(occupancy=lambda u: None, history=hist)
        assert pool.acquire(URL, WHOLE, POOL, timeout_s=0.05, poll_s=0.01,
                            prefix_ladder=_ladder(PROMPT)) is None
        match, reason = hist.lookup(URL, _ladder(PROMPT))
        assert match is None and reason == "no_history"
        assert hist.lookup(OTHER_URL, _ladder(PROMPT))[0] is not None
        # Loopback spellings are one server.
        assert hist.lookup("http://127.0.0.1:8070", _ladder(PROMPT))[0] is not None

    def test_a_different_prompt_is_not_credited(self):
        hist = _history()
        hist.remember(URL, _ladder("x" * 200_000), prompt_tokens=50_000)
        match, reason = hist.lookup(URL, _ladder(PROMPT))
        assert match is None and reason == "no_match"

    def test_longest_match_wins_across_entries(self):
        hist = _history()
        hist.remember(URL, _ladder(HISTORY[:20_000]), prompt_tokens=6_000)
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        hist.remember(URL, _ladder(SYSTEM + "other" * 2000), prompt_tokens=3_000)
        match, _ = hist.lookup(URL, _ladder(PROMPT))
        assert match.matched_chars > 100_000 and match.entry_prompt_tokens == 40_000


class TestExpiry:
    def test_time_window(self, monkeypatch):
        clock = Clock()
        hist = _history(clock=clock)
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        clock.t += 1799
        assert hist.lookup(URL, _ladder(PROMPT))[0] is not None
        clock.t += 2
        assert hist.lookup(URL, _ladder(PROMPT)) == (None, "no_match")
        monkeypatch.setenv(ph.WINDOW_S_ENV, "3600")
        assert hist.lookup(URL, _ladder(PROMPT))[0] is not None

    def test_cache_ram_volume_evicts_older_prefixes(self):
        # 8192 MiB at 65536 B/token = 131072 tokens of cache.
        hist = _history(cache_ram=8192)
        assert hist.capacity_tokens(URL) == 131072
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        hist.remember(URL, _ladder("a" * 100_000), prompt_tokens=60_000)
        assert hist.lookup(URL, _ladder(PROMPT))[0] is not None, "60k + its own 40k still fit"
        hist.remember(URL, _ladder("b" * 100_000), prompt_tokens=40_000)
        assert hist.lookup(URL, _ladder(PROMPT)) == (None, "no_match"), "evicted by volume"

    def test_a_prefix_larger_than_the_cache_never_counts(self):
        hist = _history(cache_ram=1024)  # 16384 tokens
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        assert hist.lookup(URL, _ladder(PROMPT))[0] is None

    def test_cache_ram_zero_means_no_credit(self):
        hist = _history(cache_ram=0)
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        assert hist.lookup(URL, _ladder(PROMPT)) == (None, "cache_ram_off")

    def test_undeclared_cache_ram_uses_the_server_default(self):
        hist = _history(cache_ram=None)
        assert hist.capacity_tokens(URL) == 8192 * 1024 * 1024 // 65536

    def test_a_server_relaunch_empties_the_history(self):
        hist = _history()
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        assert hist.lookup(URL, _ladder(PROMPT))[0] is not None
        hist.launch_ids["id"] = "L2"
        assert hist.lookup(URL, _ladder(PROMPT)) == (None, "no_match")

    def test_cache_ram_is_read_from_the_registry_per_port(self):
        resolver = ContextLimitResolver(live=False, role_urls=lambda: {}, registry_facts=lambda: {
            8083: {"context_tokens": POOL, "slots": 4, "kv_unified": True,
                   "cache_ram_mib": 65536},
            8086: {"context_tokens": 8192, "slots": 2, "kv_unified": True,
                   "cache_ram_mib": 0},
        })
        set_context_limit_resolver(resolver)
        hist = ph.PrefixHistory(launch_id=lambda u: None, host_wide=False)
        assert hist.capacity_tokens(URL) == 65536 * 1024 * 1024 // 65536
        assert hist.capacity_tokens("http://localhost:8086") == 0
        assert hist.capacity_tokens("http://localhost:9999") == 131072  # undeclared


class TestFallback:
    @pytest.mark.parametrize("case", ["no_ladder", "credit_disabled", "history_off"])
    def test_whole_prompt_sizing_when_no_credit_is_reliable(self, case, monkeypatch):
        hist = _history()
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        ladder = None if case == "no_ladder" else _ladder(PROMPT)
        if case == "credit_disabled":
            monkeypatch.setenv(kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV, "-1")
        if case == "history_off":
            monkeypatch.setenv(ph.HISTORY_ENV, "0")
        _lease_held_elsewhere()
        pool = SharedKVPoolAdmission(occupancy=lambda u: None, history=hist)
        assert pool.acquire(URL, WHOLE, POOL, timeout_s=0.05, poll_s=0.01,
                            prefix_ladder=ladder) is None, f"{case}: still a long prefill"
        est, facts = pool.history_prefill_estimate(URL, WHOLE, ladder)
        assert est == WHOLE and facts["reason"] == {
            "no_ladder": "no_fingerprint", "credit_disabled": "disabled",
            "history_off": "disabled"}[case]

    def test_unavailable_credit_is_recorded_with_slots_no_prompt(self):
        # Production /slots (debug off): idle slots carry no prompt text.
        occ = PoolOccupancy(url=URL, slots=(
            SlotState(slot_id=0, n_ctx=POOL, is_processing=False, n_prompt_tokens=0,
                      n_remain=None),))
        pool = SharedKVPoolAdmission(occupancy=lambda u: occ, history=_history())
        ticket = pool.acquire(URL, 1000, POOL, timeout_s=0, prompt_text=PROMPT,
                              prefix_ladder=_ladder(PROMPT))
        info = pool.admission_record(ticket)
        assert info["cache_credit_source"] is None and info["cache_credited_tokens"] == 0
        assert info["cache_credit_unavailable"] == {"fp_history": "no_history",
                                                    "slots_text": "slots_no_prompt"}
        assert pool.get_status()[URL]["admission_stats"]["cache_credit_unavailable"] == 1

    def test_slots_text_is_still_a_second_source(self):
        occ = PoolOccupancy(url=URL, slots=(SlotState(
            slot_id=0, n_ctx=POOL, is_processing=False, n_prompt_tokens=len(HISTORY) // 4,
            n_remain=None, prompt_text=HISTORY),))
        _lease_held_elsewhere()
        pool = SharedKVPoolAdmission(occupancy=lambda u: occ, history=_history())
        ticket = pool.acquire(URL, WHOLE, POOL, timeout_s=0, prompt_text=PROMPT,
                              prefix_ladder=_ladder(PROMPT))
        assert ticket is not None
        info = pool.admission_record(ticket)
        assert info["cache_credit_source"] == "slots_text"
        assert info["cache_credit_matched_chars"] == len(HISTORY)
        assert info["cache_credited_tokens"] == WHOLE - (-(-len(NEW_TURN) // 3) + 4096)

    def test_the_larger_credit_wins_when_both_sources_match(self):
        hist = _history()
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        occ = PoolOccupancy(url=URL, slots=(SlotState(
            slot_id=0, n_ctx=POOL, is_processing=False, n_prompt_tokens=10,
            n_remain=None, prompt_text=SYSTEM),))
        pool = SharedKVPoolAdmission(occupancy=lambda u: occ, history=hist)
        ticket = pool.acquire(URL, WHOLE, POOL, timeout_s=0, prompt_text=PROMPT,
                              prefix_ladder=_ladder(PROMPT))
        assert pool.admission_record(ticket)["cache_credit_source"] == "fp_history"


# ── host-wide history and the record feed ─────────────────────────────────────────────────


class TestHostWideAndFeed:
    def test_another_worker_sees_the_served_prefix_through_the_file(self):
        clock = Clock()
        worker_a = _history(clock=clock, host_wide=True)
        worker_b = _history(clock=clock, host_wide=True)
        worker_a.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        match, _ = worker_b.lookup(URL, _ladder(PROMPT))
        assert match is not None and match.entry_prompt_tokens == 40_000
        state = json.loads(worker_a.path(URL).read_text())
        assert state["schema"] == "epyc.orchestrator.kv_prefix_history.v1"
        assert len(state["entries"]) == 1 and state["cum"] == 40_000

    def test_the_file_is_bounded(self, monkeypatch):
        monkeypatch.setenv(ph.MAX_ENTRIES_ENV, "3")
        hist = _history(host_wide=True)
        for i in range(6):
            hist.remember(URL, _ladder(f"{i}" * 3000), prompt_tokens=1000)
        assert len(json.loads(hist.path(URL).read_text())["entries"]) == 3

    def _record(self, **over):
        rec = {"dispatched": True, "outcome": "ok",
               "server": {"base_url": URL, "port": 8083},
               "timings": {"prompt_n": 1000, "cache_n": 39_000, "predicted_n": 200}}
        rec.update(over)
        return rec

    def test_remember_record_uses_the_server_measured_prompt(self):
        hist = _history()
        ph.set_prefix_history(hist)
        ph.remember_record(self._record(), _ladder(HISTORY))
        match, _ = hist.lookup(URL, _ladder(PROMPT))
        assert match.entry_prompt_tokens == 40_000

    @pytest.mark.parametrize("over", [{"dispatched": False}, {"outcome": "timeout"},
                                      {"outcome": "context_overflow"}, {"server": {}}])
    def test_only_served_calls_are_remembered(self, over):
        hist = _history()
        ph.set_prefix_history(hist)
        ph.remember_record(self._record(**over), _ladder(HISTORY))
        assert hist.lookup(URL, _ladder(PROMPT)) == (None, "no_history")

    def test_recorded_backend_call_feeds_the_history(self, tmp_path, monkeypatch):
        monkeypatch.setenv(serving_calls.LOG_ENV, str(tmp_path / "calls.jsonl"))
        hist = _history()
        ph.set_prefix_history(hist)

        class Backend:
            config = SimpleNamespace(base_url=URL)

            @serving_calls.recorded_call("infer")
            def infer(self, role_config, request):
                serving_calls.note_timings({"prompt_n": 100, "cache_n": 39_900,
                                            "predicted_n": 5})
                return SimpleNamespace(success=True, output="ok")

        Backend().infer(None, SimpleNamespace(prompt=HISTORY, chat_payload=None, role="r"))
        match, _ = hist.lookup(URL, _ladder(PROMPT))
        assert match is not None and match.entry_prompt_tokens == 40_000


# ── the serving record fields ─────────────────────────────────────────────────────────────


class TestRecordFields:
    def test_staged_kv_admission_becomes_its_own_record_block(self):
        info = {"cache_credit_source": "fp_history", "cache_credited_tokens": 30_000,
                "cache_credit_prefix_tokens_est": 38_000}
        staged = {"role": "r", "port": 8083, "_ts0": 1.0, "kv_admission": info}
        rec = serving_calls.build_record(method="infer", role_config=None, request=None,
                                         base_url=URL, ts_start=2.0, ts_end=3.0,
                                         staged=staged)
        assert rec["kv_admission"] == info
        assert "kv_admission" not in rec["caller"] and rec["caller"]["port"] == 8083

    def test_inference_stages_the_admission_decision(self, monkeypatch):
        from src.llm_primitives import LLMPrimitives
        from src.model_server import InferenceResult

        set_context_limit_resolver(ContextLimitResolver(live=False, role_urls=lambda: {},
            registry_facts=lambda: {8083: {"context_tokens": POOL, "slots": 4,
                                           "kv_unified": True, "ctx_max": 262144}}))
        hist = _history()
        hist.remember(URL, _ladder(HISTORY), prompt_tokens=40_000)
        pool = SharedKVPoolAdmission(occupancy=lambda u: None, history=hist)
        monkeypatch.setattr(kpa, "_shared_pool_admission", pool)
        staged: list = []
        real = serving_calls.annotate_staged

        def spy(**fields):
            staged.append(fields)
            return real(**fields)

        monkeypatch.setattr(serving_calls, "annotate_staged", spy)
        tracker = MagicMock()
        tracker.is_available.return_value = True
        prims = LLMPrimitives(mock_mode=False, server_urls={"architect_critic": URL},
                              health_tracker=tracker)
        backend = MagicMock(spec=[])
        backend.infer_stream_text = MagicMock(side_effect=lambda rc, req, on_chunk: (
            on_chunk("ok"), InferenceResult(role="r", output="ok", tokens_generated=1,
                                            generation_speed=1.0, elapsed_time=0.1,
                                            success=True))[1])
        prims._backends["architect_critic"] = backend
        assert prims._real_call(PROMPT, "architect_critic", n_tokens=100) == "ok"
        (info,) = [f["kv_admission"] for f in staged if "kv_admission" in f]
        assert info["cache_credit_source"] == "fp_history"
        assert info["cache_credited"] is True and info["cache_credited_tokens"] > 0
