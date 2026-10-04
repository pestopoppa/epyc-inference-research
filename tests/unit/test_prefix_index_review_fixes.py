"""RTG-58 P2 branch B: the stack owner's review findings (D1-D6, KPF-26) and the
KPF-27e mapping of the RTG-58 P1 server interface
(``/mnt/raid0/llm/tmp/kpf-p1-20261004/INTERFACE.md``).

The reviewer's own failing tests are ``test_review_p2b_defects.py``; these pin
the rest of each fix. Everything here is offline: fake ``/slots`` tables,
``httpx.MockTransport``, injected ``/props`` readers. No server, no inference.
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
    ContextLimitResolver,
    PoolOccupancy,
    SlotCheckpoint,
    SlotState,
    parse_slots,
)
from src.inference import prefix_index as pi
from src.inference.prefix_index import PrefixIndex
from src.scheduling import kv_pool_admission as kpa
from src.scheduling.kv_pool_admission import KVPoolQueueFull, SharedKVPoolAdmission

URL = "http://localhost:8083"
BLOCK = 64


class Slots:
    """A mutable fake ``/slots`` table (optionally with the KPF-27e fields)."""

    def __init__(self, n: int = 4) -> None:
        self.rows = {i: dict(busy=False, n=0, task=None, decoded=1) for i in range(n)}
        self.up = True
        self.kv_pool: dict[str, int] | None = None

    def set(self, sid: int, *, busy: bool = False, n: int = 0, task: int | None = None,
            decoded: int = 1, **extra) -> None:
        self.rows[sid] = dict(busy=busy, n=n, task=task, decoded=decoded, **extra)

    def __call__(self, url: str) -> PoolOccupancy | None:
        if not self.up:
            return None
        return PoolOccupancy(url=url, kv_pool=self.kv_pool, slots=tuple(
            SlotState(slot_id=i, n_ctx=98304, is_processing=r["busy"], n_prompt_tokens=r["n"],
                      n_remain=None, n_decoded=r["decoded"], n_prompt_tokens_processed=0,
                      id_task=r["task"], content_epoch=r.get("epoch"),
                      prefix_hash=r.get("hash"), kv_private=r.get("private"),
                      kv_shared=r.get("shared"), checkpoints=r.get("checkpoints", ()))
            for i, r in self.rows.items()))


@pytest.fixture(autouse=True)
def flag_on(monkeypatch, tmp_path):
    from src.runtime import long_prefill_lease

    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.BLOCK_CHARS_ENV, "256")
    monkeypatch.setattr(long_prefill_lease, "lease_dir", lambda: tmp_path)
    for name in (pi.FORK_ENV, pi.PIN_ENV, pi.LPM_ENV, pi.LPM_MIN_TOKENS_ENV,
                 pi.TRUNK_MIN_TOKENS_ENV, pi.TRUNK_HOLD_S_ENV, pi.TRUNK_GRACE_S_ENV,
                 pi.CREDIT_FRESH_S_ENV, kpa.KV_POOL_LONG_PREFILL_ENV,
                 kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV, kpa.KV_POOL_WAIT_ENV,
                 kpa.KV_POOL_MAX_QUEUED_ENV):
        monkeypatch.delenv(name, raising=False)
    pi.reset_indexes()
    yield
    pi.reset_indexes()


def _doc(tag: str, lines: int) -> str:
    return "".join(f"{tag} line {i}: shared context for every turn\n" for i in range(lines))


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def idx(tmp_path):
    clock = Clock()
    index = PrefixIndex(URL, clock=clock, wall=clock, launch_id=lambda url: "L1",
                        directory=lambda: tmp_path, block=BLOCK)
    index.clock = clock  # type: ignore[attr-defined]
    return index


def _text(tag: str, blocks: int) -> str:
    return "".join(f"{tag}{i:05d}".ljust(BLOCK, ".") for i in range(blocks))


#: A P1 server's ``/props.slot_fork`` with the fork on (``kv`` mode, no floor).
KV_CAPS = {"min_tokens": 0, "mode": "kv", "checkpoint_at": False}


def _fork_pool(slots: Slots, **kw) -> SharedKVPoolAdmission:
    # Fork features need the server to report its fork (FORK=auto, or 1).
    kw.setdefault("fork_caps", lambda url: KV_CAPS)
    return SharedKVPoolAdmission(occupancy=slots, cross_process=False, **kw)


# ── D1: trunk-first engages for a simultaneous fan-out ──────────────────────────


def test_d1_hold_ends_by_slots_once_the_owner_was_seen_prefilling(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    monkeypatch.setenv(pi.TRUNK_GRACE_S_ENV, "0.2")  # one "TTL" = 0.1 s here
    slots = Slots()
    pool = _fork_pool(slots)
    trunk = _doc("S", 1200)
    owner = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    got: dict[str, int | None] = {}
    t = threading.Thread(target=lambda: got.setdefault("b", pool.acquire(
        URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=10, poll_s=0.01)))
    slots.set(0, busy=True, n=12000, task=1, decoded=0)  # the owner reaches the server
    t.start()
    time.sleep(0.15)
    assert "b" not in got  # held (and, past one TTL, seen prefilling)
    slots.set(0, busy=True, n=12000, task=1, decoded=3)  # first token: prefill over
    t.join(timeout=5)
    assert got["b"] is not None
    assert pi.get_index(URL).stats["trunk_holds"] == 1
    pool.release(URL, got["b"])
    pool.release(URL, owner)


# ── D2: the scout gate hands the prefill on, and feeds the index ────────────────


def _sse(*objs) -> bytes:
    return ("".join(f"data: {json.dumps(o)}\n\n" for o in objs) + "data: [DONE]\n\n").encode()


def test_d2_transport_fires_the_first_chunk_hook_once_and_returns_timings():
    from src.api.routes.chat_pipeline import scout_stage as S

    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_sse(
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]},
            {"choices": [], "timings": {"prompt_n": 5, "cache_n": 7, "id_slot": 2}}))

    hits = []
    t = S.ChatCompletionsTransport("http://x", client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = t.complete([{"role": "user", "content": "q"}], max_tokens=8,
                     should_stop=lambda: False, timeout_s=5, on_first_chunk=lambda: hits.append(1))
    assert hits == [1]
    assert out.timings == {"prompt_n": 5, "cache_n": 7, "id_slot": 2}
    assert "checkpoint_at" not in seen["body"]  # never sent unless asked


def test_d2_pool_gate_passes_the_hook_and_reports_prefill_done_once():
    from src.api.routes.chat_pipeline.scout_stage import CompletionResult, PoolGatedTransport

    pool = MagicMock()
    pool.acquire.return_value = 3

    class Inner:
        gate_kwargs = True

        def complete(self, messages, *, max_tokens, should_stop, timeout_s,
                     on_first_chunk=None, checkpoint_at=None):
            on_first_chunk()
            assert pool.prefill_done.call_count == 1  # at first output, not at the end
            assert checkpoint_at is None  # flag-on fork path only
            return CompletionResult("x", 1, 1, "stop")

    gate = PoolGatedTransport(Inner(), url=URL, limit=SimpleNamespace(pool_tokens=1000), pool=pool)
    gate.complete([{"role": "user", "content": "q"}], max_tokens=4,
                  should_stop=lambda: False, timeout_s=5)
    assert pool.prefill_done.call_count == 1
    pool.release.assert_called_once()


def test_d2_scout_trunk_reaches_the_index_with_its_server_slot(monkeypatch):
    from src.api.routes.chat_pipeline.scout_stage import (
        CompletionResult,
        PoolGatedTransport,
        _scout_prefix_key,
    )

    messages = [{"role": "system", "content": _doc("Y", 200)}, {"role": "user", "content": "t1"}]
    pool = _fork_pool(Slots())
    inner = MagicMock()
    inner.complete.return_value = CompletionResult(
        "x", 10, 2, "stop", timings={"prompt_n": 900, "cache_n": 0, "predicted_n": 2,
                                     "id_slot": 1, "id_task": 41})
    gate = PoolGatedTransport(inner, url=URL, limit=SimpleNamespace(pool_tokens=393216), pool=pool)
    gate.complete(messages, max_tokens=8, should_stop=lambda: False, timeout_s=30)
    idx = pi.get_index(URL)
    assert idx.stats["observed"] == 1 and idx.stats["predictions"] == 1
    entry = idx._entries["pending:slot1"]
    assert entry.slot_id == 1 and entry.id_task == 41 and entry.origin == "exact"
    assert entry.chars == len(_scout_prefix_key(messages)["prefix_key"])


def test_d2_scout_checkpoint_at_only_for_a_checkpoint_mode_fork_server(monkeypatch):
    from src.api.routes.chat_pipeline.scout_stage import _scout_checkpoint_at

    long_sys = [{"role": "system", "content": _doc("Z", 400)}, {"role": "user", "content": "t"}]
    caps = {"min_tokens": 1024, "mode": "checkpoint", "checkpoint_at": True}
    pool = SimpleNamespace(fork_caps=lambda url: caps)
    assert _scout_checkpoint_at(pool, URL, long_sys) is None  # FORK not requested
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    assert _scout_checkpoint_at(pool, URL, long_sys) == [{"message": 0, "at": "end"}]
    short = [{"role": "system", "content": "tiny"}, {"role": "user", "content": "t"}]
    assert _scout_checkpoint_at(pool, URL, short) is None  # below the fork floor
    for other in ({**caps, "mode": "kv"}, {**caps, "checkpoint_at": False}, None):
        assert _scout_checkpoint_at(SimpleNamespace(fork_caps=lambda url, c=other: c),
                                    URL, long_sys) is None
    monkeypatch.setenv(pi.FLAG_ENV, "0")
    assert _scout_checkpoint_at(pool, URL, long_sys) is None


# ── D3: fork credit follows the shared cells ────────────────────────────────────


def test_d3_two_dependents_share_with_the_heir_after_the_source_leaves(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    pool = _fork_pool(Slots())
    trunk = _doc("H", 1200)
    a = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    pool.prefill_done(URL, a)
    b = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=0)
    c = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "C", timeout_s=0)
    credit_b = pool.admission_record(b)["prefix_index"]["fork_credit_tokens"]
    credit_c = pool.admission_record(c)["prefix_index"]["fork_credit_tokens"]
    assert credit_b > 0 and credit_c > 0
    full = pool.reservation_tokens(URL, 13000, 0)
    pool.release(URL, a)
    # The trunk is live once (in the heir), never zero times and never twice.
    assert pool.in_flight_tokens(URL) == 2 * full - min(credit_b, credit_c)
    heir = b if pool._inflight[URL][b] == full else c
    other = c if heir == b else b
    assert pool._fork_source[other][1] == heir
    pool.release(URL, heir)
    assert pool._inflight[URL][other] == full  # the last holder owns the whole trunk
    assert pi.get_index(URL).stats["fork_credit_reattributed"] > 0
    pool.release(URL, other)
    assert pool.in_flight_tokens(URL) == 0 and not pool._fork_source


def test_d3_a_dependent_released_first_leaves_no_bookkeeping(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    pool = _fork_pool(Slots())
    trunk = _doc("I", 1200)
    a = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    pool.prefill_done(URL, a)
    b = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=0)
    pool.release(URL, b)
    assert not pool._fork_source
    pool.release(URL, a)
    assert pool.in_flight_tokens(URL) == 0


def test_d3_busy_slot_fork_credit_needs_slots_up(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    slots = Slots()
    pool = _fork_pool(slots)
    trunk = _doc("J", 600)
    idx = pi.get_index(URL)
    idx.observe_served(trunk, slot_id=1, prompt_tokens=7000, generated_tokens=0)
    slots.set(1, n=7000, task=5)
    idx.reconcile(slots(URL))
    slots.set(1, busy=True, n=7100, task=5, decoded=4)  # same task, decoding on: busy source
    t = pool.acquire(URL, 9000, 393216, prefix_key=trunk + "q", timeout_s=0)
    info = pool.admission_record(t)["prefix_index"]
    assert info["match"]["source"] == "slot_busy" and info["fork_credit_tokens"] > 0
    pool.release(URL, t)
    slots.up = False  # /slots gone: nothing would count the busy slot's cells
    v = pool._prefix_index_view(URL, 99, idx, trunk + "q", 9000, 9000, 9000, {}, first=False,
                                slots_up=False)[2]
    assert v["fork_credit_tokens"] == 0


def test_d3_unmeasured_prefix_estimate_is_3_6_chars_per_token():
    e = pi.Entry(id="inflight:1", kind="inflight", hashes=("a",), chars=36000)
    assert e.tokens_for_chars(36000) == 10000


# ── D4: hash once per ticket, unique cells outside the gate lock ────────────────


def test_d4_the_key_is_hashed_once_per_ticket_however_long_it_waits(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    pool = _fork_pool(Slots())
    trunk = _doc("K", 1200)
    owner = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    calls = {"n": 0}
    real = pi.block_hashes

    def counting(text, block=None):
        if text == trunk + "B":
            calls["n"] += 1
        return real(text, block)

    monkeypatch.setattr(pi, "block_hashes", counting)
    unique_calls = {"locked": 0}
    real_unique = pi.PrefixIndex.unique_cells

    def unique(self):
        if pool._cond._is_owned():  # type: ignore[attr-defined]
            unique_calls["locked"] += 1
        return real_unique(self)

    monkeypatch.setattr(pi.PrefixIndex, "unique_cells", unique)
    b = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=0.3, poll_s=0.01)
    assert b is None and calls["n"] == 1  # ~30 polls, one hash
    pool.prefill_done(URL, owner)
    c = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=1, poll_s=0.01)
    assert c is not None and unique_calls["locked"] == 0
    assert pool.admission_record(c)["prefix_index"]["unique_cells"]["entries"] >= 2
    pool.release(URL, c)
    pool.release(URL, owner)


# ── D5: an index credit keeps the legacy sources' reasons ───────────────────────


def test_d5_index_credit_keeps_the_history_reason(monkeypatch):
    slots = Slots()
    pool = _fork_pool(slots)
    trunk = _doc("D", 900)
    idx = pi.get_index(URL)
    idx.observe_served(trunk, slot_id=2, prompt_tokens=9000, generated_tokens=0)
    slots.set(2, n=9000, task=7)
    idx.reconcile(slots(URL))
    t = pool.acquire(URL, 20000, 393216, prefix_key=trunk + "next", timeout_s=0)
    rec = pool.admission_record(t)
    assert rec["cache_credit_source"] == "prefix_index"
    assert rec["cache_credit_unavailable"] is None
    assert set(rec["cache_credit_history_reason"]) == {"fp_history", "slots_text"}
    pool.release(URL, t)


# ── D6: an unverified slot entry is no credit source ────────────────────────────


def test_d6_ledger_adopted_entry_is_credited_only_after_this_worker_verifies_it(tmp_path):
    clock = Clock()
    kw = dict(clock=clock, wall=clock, launch_id=lambda url: "L1",
              directory=lambda: tmp_path, block=BLOCK, host_wide_ledger=True)
    writer, reader = PrefixIndex(URL, **kw), PrefixIndex(URL, **kw)
    text = _text("V", 30)
    writer.observe_served(text, slot_id=1, prompt_tokens=3000, generated_tokens=0)
    occ = PoolOccupancy(url=URL, slots=(SlotState(slot_id=1, n_ctx=8192, is_processing=False,
                                                  n_prompt_tokens=3000, n_remain=None,
                                                  n_decoded=1, id_task=9),))
    writer.reconcile(occ)
    reader.reconcile(None)  # /slots unavailable here: adopts the ledger entry, cannot verify
    assert reader._entries["slot:1"].verified_at is None
    assert reader.lookup(text + "x").source is None
    reader.reconcile(occ)
    assert reader.lookup(text + "x").source == "slot_idle"
    clock.t += pi.DEFAULT_CREDIT_FRESH_S + 1  # /slots silent since: stale again
    assert reader.lookup(text + "x").source is None


# ── KPF-26: a fan-out wave held behind one trunk does not fill the queue ────────


def test_kpf26_trunk_held_siblings_do_not_count_toward_the_queue_cap(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    monkeypatch.setenv(kpa.KV_POOL_MAX_QUEUED_ENV, "2")
    pool = _fork_pool(Slots())
    trunk = _doc("W", 1200)
    owner = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    results: list = []

    def sib(i):
        try:
            results.append(pool.acquire(URL, 13000, 393216, prefix_key=trunk + f"S{i}",
                                        timeout_s=10, poll_s=0.01))
        except KVPoolQueueFull as exc:
            results.append(exc)

    threads = [threading.Thread(target=sib, args=(i,)) for i in range(5)]
    for th in threads:
        th.start()
        time.sleep(0.05)  # each is trunk-held before the next arrives
    assert pool.queued(URL) == 5  # > max_queued=2, none refused
    pool.prefill_done(URL, owner)
    for th in threads:
        th.join(timeout=10)
    assert len(results) == 5 and all(isinstance(r, int) for r in results)
    for r in results:
        pool.release(URL, r)
    pool.release(URL, owner)


def test_kpf26_queue_cap_is_unchanged_without_trunk_holds(monkeypatch):
    pool = _fork_pool(Slots())
    assert pool._queue_full([1, 2], 2) and not pool._queue_full([1], 2)
    pool._trunk_held.update({1: True, 2: True})
    assert not pool._queue_full([1, 2, 3], 2)
    assert pool._queue_full(list(range(1, 9)), 2)  # TRUNK_QUEUE_FACTOR x cap


# ── KPF-27e: the server interface ───────────────────────────────────────────────


def test_parse_slots_reads_the_fork_fields_and_defaults_without_them():
    body = [
        {"id": 0, "n_ctx": 393216, "is_processing": True, "n_prompt_tokens": 61234,
         "id_task": 3, "content_epoch": 7, "prefix_hash": "9c1f0e5ab3d2c417",
         "kv_cells": {"private": 1234, "shared": 60000},
         "kv_pool": {"size": 393216, "used": 125000, "shared": 60000},
         "checkpoints": [{"n_tokens": 60000, "pinned": True, "prefix_hash": "41b0d7c2e9a85f13"},
                         {"bogus": 1}]},
        {"id": 1, "n_ctx": 393216, "is_processing": False, "n_prompt_tokens": 10},
    ]
    occ = parse_slots(URL, body)
    s0, s1 = occ.slots
    assert (s0.content_epoch, s0.prefix_hash, s0.kv_private, s0.kv_shared) == (
        7, "9c1f0e5ab3d2c417", 1234, 60000)
    assert s0.checkpoints == (SlotCheckpoint(60000, True, "41b0d7c2e9a85f13"),)
    assert occ.kv_pool == {"size": 393216, "used": 125000, "shared": 60000}
    assert (s1.content_epoch, s1.prefix_hash, s1.kv_private, s1.checkpoints) == (None, None, None, ())
    v10 = parse_slots(URL, [{"id": 0, "is_processing": True, "n_prompt_tokens": 5}])
    assert v10.kv_pool is None and v10 == PoolOccupancy(url=URL, slots=v10.slots)


def test_projected_unique_tokens_counts_forked_cells_once():
    def slot(i, n, private, shared):
        return SlotState(slot_id=i, n_ctx=1, is_processing=True, n_prompt_tokens=n, n_remain=100,
                         kv_private=private, kv_shared=shared)

    occ = PoolOccupancy(url=URL, slots=(slot(0, 60000, 0, 60000), slot(1, 61000, 1000, 60000)),
                        kv_pool={"size": 393216, "used": 61000, "shared": 60000})
    assert occ.projected_tokens(1.0) == 121200
    assert occ.projected_unique_tokens(1.0) == 61200  # 1000 private + 60000 shared + decode
    plain = PoolOccupancy(url=URL, slots=occ.slots)
    assert plain.projected_unique_tokens(1.0) == plain.projected_tokens(1.0)


def test_gate_counts_unique_cells_only_with_the_flag_on(monkeypatch):
    slots = Slots()
    slots.kv_pool = {"size": 393216, "used": 61000, "shared": 60000}
    slots.set(0, busy=True, n=60000, private=0, shared=60000)
    slots.set(1, busy=True, n=61000, private=1000, shared=60000)
    pool = _fork_pool(slots)
    assert pool._observed(URL)[0] == 61000
    monkeypatch.setenv(pi.FLAG_ENV, "0")
    assert pool._observed(URL)[0] == 121000


def test_epoch_binding_keeps_a_prefix_across_an_appending_task_and_drops_on_a_bump(idx):
    text = _text("E", 30)
    idx.observe_served(text, slot_id=2, prompt_tokens=3000, generated_tokens=0)

    def occ(**kw):
        row = dict(slot_id=2, n_ctx=8192, is_processing=False, n_prompt_tokens=3000,
                   n_remain=None, n_decoded=1, id_task=7, content_epoch=4)
        row.update(kw)
        return PoolOccupancy(url=URL, slots=(SlotState(**row),))

    idx.reconcile(occ())
    assert idx._entries["slot:2"].content_epoch == 4
    idx.reconcile(occ(id_task=8, is_processing=True, n_prompt_tokens=3500))  # extends ours
    assert "slot:2" in idx._entries and idx._entries["slot:2"].busy
    idx.reconcile(occ(id_task=8, n_prompt_tokens=3500, content_epoch=5))  # truncated/forked into
    assert "slot:2" not in idx._entries
    assert idx.stats["stale_drops"]["epoch_changed"] == 1


def test_prefix_hash_and_checkpoint_hash_changes_drop_a_bound_slot(idx):
    text = _text("H", 30)

    def occ(h="aaaa", cps=(SlotCheckpoint(2000, True, "c1"),), n=3000):
        return PoolOccupancy(url=URL, slots=(SlotState(
            slot_id=1, n_ctx=8192, is_processing=False, n_prompt_tokens=n, n_remain=None,
            n_decoded=1, id_task=7, content_epoch=1, prefix_hash=h, checkpoints=cps),))

    idx.observe_served(text, slot_id=1, prompt_tokens=3000, generated_tokens=0)
    idx.reconcile(occ())
    idx.reconcile(occ())
    assert "slot:1" in idx._entries
    idx.reconcile(occ(h="bbbb"))
    assert idx.stats["stale_drops"]["prefix_hash_changed"] == 1
    idx.observe_served(text, slot_id=1, prompt_tokens=3000, generated_tokens=0)
    idx.reconcile(occ())
    idx.reconcile(occ(h="zzzz", n=3100, cps=(SlotCheckpoint(2000, True, "c2"),)))
    assert idx.stats["stale_drops"]["prefix_hash_changed"] == 2


def test_timings_id_slot_and_id_task_bind_the_chat_lane_exactly(idx, monkeypatch):
    monkeypatch.setattr(pi, "get_index", lambda url: idx)
    text = _text("C", 30)
    pi.observe_record({"dispatched": True, "outcome": "ok", "server": {"base_url": URL},
                       "notes": {"timings": {"prompt_n": 2900, "cache_n": 100, "predicted_n": 5,
                                             "id_slot": 3, "id_task": 77}}},
                      text, key_kind="approx")
    e = idx._entries["pending:slot3"]
    assert (e.slot_id, e.id_task, e.origin) == (3, 77, "exact")
    two_idle = PoolOccupancy(url=URL, slots=tuple(
        SlotState(slot_id=i, n_ctx=8192, is_processing=False, n_prompt_tokens=3005,
                  n_remain=None, n_decoded=1, id_task=t) for i, t in ((2, 76), (3, 77))))
    idx.reconcile(two_idle)  # two slots hold the same count: only the task decides
    assert idx._entries["slot:3"].origin == "exact" and idx.stats["bound_by_task"] == 1


def test_task_mismatch_never_binds(idx):
    text = _text("M", 30)
    idx.observe_served(text, slot_id=1, prompt_tokens=3000, generated_tokens=0, id_task=5)
    idx.reconcile(PoolOccupancy(url=URL, slots=(SlotState(
        slot_id=1, n_ctx=8192, is_processing=False, n_prompt_tokens=3000, n_remain=None,
        n_decoded=1, id_task=6),)))
    assert "slot:1" not in idx._entries


def test_fork_caps_and_the_fork_switch(monkeypatch):
    caps = pi.fork_caps_from_props(
        {"slot_fork": {"min_tokens": 2048, "mode": "checkpoint", "checkpoint_at": True}})
    assert caps == {"min_tokens": 2048, "mode": "checkpoint", "checkpoint_at": True}
    assert pi.fork_caps_from_props({"n_ctx": 1}) is None
    none_mode = pi.fork_caps_from_props({"slot_fork": {"min_tokens": 2048, "mode": "none"}})
    assert pi.fork_enabled(caps) is False  # FORK unset
    monkeypatch.setenv(pi.FORK_ENV, "1")
    # Fable re-review: FORK=1 with no server fork (caps None: every v10 server)
    # refuses rather than over-admitting by the trunk size.
    assert not pi.fork_enabled(None) and pi.fork_enabled(caps) and not pi.fork_enabled(none_mode)
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    assert pi.fork_enabled(caps) and not pi.fork_enabled(None) and not pi.fork_enabled(none_mode)
    monkeypatch.setenv(pi.FLAG_ENV, "0")
    assert not pi.fork_enabled(caps) and not pi.fork_requested()


def test_resolver_reads_slot_fork_from_props_and_caches_it():
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return {"n_ctx": 393216, "slot_fork": {"min_tokens": 1024, "mode": "kv",
                                               "checkpoint_at": True}}

    r = ContextLimitResolver(fetch_props=fetch, live=True)
    assert r.fork_caps(URL)["mode"] == "kv"
    assert r.fork_caps(URL)["min_tokens"] == 1024 and len(calls) == 1
    assert ContextLimitResolver(fetch_props=fetch, live=False).fork_caps(URL) is None


def test_gate_takes_mode_and_floor_from_the_server(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    caps = {"min_tokens": 50000, "mode": "checkpoint", "checkpoint_at": True}
    pool = _fork_pool(Slots(), fork_caps=lambda url: caps)
    trunk = _doc("P", 1200)  # ~15k tokens: below the server's 50k floor
    a = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    b = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=0)
    assert b is not None  # no hold: the server would not fork a trunk this short
    info = pool.admission_record(b)["prefix_index"]
    assert info["fork"] is True and info["server_fork"]["mode"] == "checkpoint"
    pool.release(URL, b)
    pool.release(URL, a)
    pool2 = _fork_pool(Slots(), fork_caps=lambda url: {**caps, "mode": "none"})
    c = pool2.acquire(URL, 13000, 393216, prefix_key=trunk + "C", timeout_s=0)
    assert pool2.admission_record(c)["prefix_index"]["fork"] is False


def test_checkpoint_mode_cuts_a_busy_source_to_its_checkpoints(idx):
    text = _text("B", 40)
    idx.observe_served(text, slot_id=1, prompt_tokens=4000, generated_tokens=0)

    def occ(busy, cps):
        return PoolOccupancy(url=URL, slots=(SlotState(
            slot_id=1, n_ctx=8192, is_processing=busy, n_prompt_tokens=4000 if not busy else 4100,
            n_remain=None, n_decoded=1, id_task=7, content_epoch=2, checkpoints=cps),))

    idx.reconcile(occ(False, ()))
    idx.reconcile(occ(True, (SlotCheckpoint(1500, True, "x"), SlotCheckpoint(3900, False, "y"))))
    key = text + "suffix"
    full = idx.lookup(key, fork=True)
    assert full.source == "slot_busy" and full.tokens_est == 4000
    cut = idx.lookup(key, fork=True, fork_mode="checkpoint")
    assert cut.tokens_est == 3900
    idx.reconcile(occ(True, (SlotCheckpoint(5000, False, "z"),)))
    assert idx.lookup(key, fork=True, fork_mode="checkpoint").tokens_est == 0


def test_fnv1a64_matches_the_reference_vectors():
    assert pi.fnv1a64_tokens([]) == "cbf29ce484222325"
    # FNV-1a 64 of the bytes 01 00 00 00 (token id 1, int32 little-endian).
    h = 0xCBF29CE484222325
    for b in (1, 0, 0, 0):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    assert pi.fnv1a64_tokens([1]) == f"{h:016x}" and len(pi.fnv1a64_tokens([1, 2])) == 16


def test_serving_record_keeps_the_fork_timings():
    from src.backends import serving_calls

    for k in ("n_fork_tokens", "fork_src_slot", "fork_src_kind", "id_slot", "id_task"):
        assert k in serving_calls.TIMING_KEYS


# ── Fable re-review notes (084921bb): FORK=auto, busy-source re-inflation,
#    per-owner prefill, prompt_progress ───────────────────────────────────────


def test_rr1_fork_1_without_server_caps_warns_once_and_stays_off(monkeypatch, caplog):
    monkeypatch.setenv(pi.FORK_ENV, "1")
    with caplog.at_level("WARNING", logger="src.inference.prefix_index"):
        assert pi.fork_enabled(None) is False
        assert pi.fork_enabled(None) is False
    warned = [r for r in caplog.records if "slot_fork" in r.getMessage()]
    assert len(warned) == 1  # once per process, not once per gate poll
    assert pi.fork_enabled(KV_CAPS) is True  # a server that reports its fork


def test_rr1_fork_auto_follows_the_server_and_never_warns(monkeypatch, caplog):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    with caplog.at_level("WARNING", logger="src.inference.prefix_index"):
        assert pi.fork_enabled(None) is False
        assert pi.fork_enabled({**KV_CAPS, "mode": "none"}) is False
        assert pi.fork_enabled(KV_CAPS) is True
        assert pi.fork_enabled({**KV_CAPS, "mode": "checkpoint"}) is True
    assert not [r for r in caplog.records if "slot_fork" in r.getMessage()]
    assert pi.fork_requested()


@pytest.mark.parametrize("mode", ["1", "auto"])
def test_rr1_v10_server_gets_no_trunk_hold_and_no_fork_credit(monkeypatch, mode):
    """Every v10 server: no ``/props.slot_fork``. The sibling of a prefilled
    trunk reserves its FULL size and is never held (no real sharing there)."""
    monkeypatch.setenv(pi.FORK_ENV, mode)
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    slots = Slots()
    slots.set(0, busy=True, n=12000, task=1, decoded=0)  # something is prefilling
    pool = _fork_pool(slots, fork_caps=lambda url: None)
    trunk = _doc("V", 1200)
    a = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    b = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=0)
    assert a is not None and b is not None  # not held behind A's trunk
    info = pool.admission_record(b)["prefix_index"]
    assert info["fork"] is False and info["fork_credit_tokens"] == 0
    assert "fork_plan" not in info
    full = pool.reservation_tokens(URL, 13000, 0)
    assert pool.in_flight_tokens(URL) == 2 * full
    assert pi.get_index(URL).stats["trunk_holds"] == 0
    pool.release(URL, b)
    pool.release(URL, a)


def _busy_source(monkeypatch):
    """A fork credit taken against slot 1, busy with task 5 (a foreign request
    the gate holds no ticket for)."""
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    slots = Slots()
    pool = _fork_pool(slots)
    trunk = _doc("R", 600)
    idx = pi.get_index(URL)
    idx.observe_served(trunk, slot_id=1, prompt_tokens=7000, generated_tokens=0)
    slots.set(1, n=7000, task=5)
    idx.reconcile(slots(URL))
    slots.set(1, busy=True, n=7100, task=5, decoded=4)  # same task, decoding on: busy source
    t = pool.acquire(URL, 9000, 393216, prefix_key=trunk + "q", timeout_s=0)
    info = pool.admission_record(t)["prefix_index"]
    assert info["match"]["source"] == "slot_busy" and info["fork_credit_tokens"] > 0
    full = pool.reservation_tokens(URL, 9000, 0)
    assert pool._inflight[URL][t] == full - info["fork_credit_tokens"]
    return slots, pool, idx, t, full, info["fork_credit_tokens"]


def _poll(pool):
    """Any other admission poll reads /slots first (``_observed``)."""
    other = pool.acquire(URL, 10, 393216, timeout_s=0)
    pool.release(URL, other)


def test_rr2_busy_source_still_busy_keeps_the_credit(monkeypatch):
    slots, pool, idx, t, full, credit = _busy_source(monkeypatch)
    slots.set(1, busy=True, n=7200, task=5, decoded=9)
    _poll(pool)
    assert pool._inflight[URL][t] == full - credit
    assert idx.stats["fork_credit_reinflated"] == 0
    pool.release(URL, t)
    assert not pool._busy_fork


@pytest.mark.parametrize("change", ["idle", "other_task", "missing", "slots_down"])
def test_rr2_busy_source_that_stops_being_counted_reinflates_the_dependant(monkeypatch, change):
    """``projected_tokens`` skips idle slots: once the busy source goes idle (its
    cells live on, shared with the dependant), runs another task, vanishes, or
    /slots is unreadable, nothing counts the shared cells unless the dependant
    reserves them again."""
    slots, pool, idx, t, full, credit = _busy_source(monkeypatch)
    if change == "idle":
        slots.set(1, busy=False, n=7300, task=5, decoded=1)
    elif change == "other_task":
        slots.set(1, busy=True, n=400, task=6, decoded=0)
    elif change == "missing":
        del slots.rows[1]
    else:
        slots.up = False
    _poll(pool)
    assert pool._inflight[URL][t] == full
    assert idx.stats["fork_credit_reinflated"] == credit
    assert pool.admission_record(t)["prefix_index"]["fork_credit_reinflated"] == credit
    assert not pool._busy_fork
    _poll(pool)  # one-way: never re-inflated twice
    assert pool._inflight[URL][t] == full
    pool.release(URL, t)
    assert pool.in_flight_tokens(URL) == 0


def test_rr2_flag_off_never_tracks_busy_fork(monkeypatch):
    monkeypatch.delenv(pi.FLAG_ENV, raising=False)
    pool = _fork_pool(Slots())
    t = pool.acquire(URL, 9000, 393216, timeout_s=0)
    assert not pool._busy_fork
    pool.release(URL, t)


def _inflight_entry(idx, ticket, *, pre_tasks, age):
    idx.begin(ticket, _text(f"T{ticket}", 4))
    e = idx._entries[f"inflight:{ticket}"]
    e.pre_tasks = pre_tasks
    e.ts = idx._clock() - age
    return e


def _occ(*rows):
    return PoolOccupancy(url=URL, slots=tuple(
        SlotState(slot_id=sid, n_ctx=8192, is_processing=busy, n_prompt_tokens=100,
                  n_remain=None, n_decoded=dec, n_prompt_tokens_processed=0, id_task=task)
        for sid, busy, task, dec in rows))


def test_rr3_inflight_binds_to_the_one_slot_that_started_a_task_since_admission(idx):
    pre = {0: None, 1: None, 2: 9}
    e = _inflight_entry(idx, 1, pre_tasks=pre, age=5.0)
    # slot 2 was already busy with task 9 before admission: not a candidate
    idx.reconcile(_occ((0, True, 11, 0), (1, False, None, 1), (2, True, 9, 0)))
    assert e.slot_id == 0 and idx.stats["inflight_bound"] == 1


def test_rr3_ambiguous_or_too_fresh_never_binds(idx):
    pre = {0: None, 1: None}
    a = _inflight_entry(idx, 1, pre_tasks=pre, age=5.0)
    b = _inflight_entry(idx, 2, pre_tasks=pre, age=5.0)
    idx.reconcile(_occ((0, True, 11, 0), (1, True, 12, 0)))
    assert a.slot_id is None and b.slot_id is None  # two entries, two new slots
    idx.end(1)
    idx.end(2)
    c = _inflight_entry(idx, 3, pre_tasks=pre, age=0.0)  # read may predate the dispatch
    idx.reconcile(_occ((0, True, 11, 0), (1, False, None, 1)))
    assert c.slot_id is None


def test_rr3_bound_owner_is_seen_and_finished_by_its_own_slot_only(idx):
    owner = _inflight_entry(idx, 1, pre_tasks={}, age=5.0)
    owner.slot_id = 0
    idx._note_prefilling(idx._clock(), 1, {2})  # a FOREIGN slot is prefilling
    assert owner.seen_prefilling is False
    idx._note_prefilling(idx._clock(), 2, {0, 2})
    assert owner.seen_prefilling is True
    now = idx._clock()
    assert not idx._prefill_over(owner, now, 2, {0, 2})
    # its own slot is decoding; the foreign slot still prefills: the trunk is done
    assert idx._prefill_over(owner, now, 1, {2})
    # an unbound sibling ignores the prefilling slot that is bound to the owner
    other = _inflight_entry(idx, 2, pre_tasks=None, age=5.0)
    idx._note_prefilling(now, 1, {0})
    assert other.seen_prefilling is False


def test_rr3_trunk_hold_ends_when_the_owners_slot_decodes_despite_foreign_prefill(monkeypatch):
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    monkeypatch.setenv(pi.TRUNK_GRACE_S_ENV, "0.2")  # one "TTL" = 0.1 s here
    slots = Slots()
    slots.set(2, busy=True, n=30000, task=9, decoded=0)  # a long foreign prefill
    pool = _fork_pool(slots)
    trunk = _doc("U", 1200)
    owner = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    got: dict[str, int | None] = {}
    t = threading.Thread(target=lambda: got.setdefault("b", pool.acquire(
        URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=10, poll_s=0.01)))
    slots.set(0, busy=True, n=12000, task=1, decoded=0)  # the owner reaches slot 0
    t.start()
    time.sleep(0.25)
    assert "b" not in got
    idx = pi.get_index(URL)
    assert idx._entries[f"inflight:{owner}"].slot_id == 0
    slots.set(0, busy=True, n=12000, task=1, decoded=3)  # owner decodes; slot 2 still prefills
    t.join(timeout=5)
    assert got["b"] is not None  # server-wide "prefilling == 0" would have held it
    pool.release(URL, got["b"])
    pool.release(URL, owner)


def test_rr4_prompt_progress_chunks_do_not_fire_the_first_chunk_hook():
    from src.api.routes.chat_pipeline import scout_stage as S

    lines_read: list[int] = []
    hook_at: list[int] = []

    def handler(request):
        return httpx.Response(200, content=_sse(
            {"choices": [{"delta": {}}], "prompt_progress": {"processed": 512, "total": 4096}},
            {"choices": [{"delta": {}}], "prompt_progress": {"processed": 4096, "total": 4096}},
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]}))

    t = S.ChatCompletionsTransport("http://x", client=httpx.Client(transport=httpx.MockTransport(handler)))

    def should_stop():
        lines_read.append(1)  # called once per streamed line, before it is parsed
        return False

    out = t.complete([{"role": "user", "content": "q"}], max_tokens=8,
                     should_stop=should_stop, timeout_s=5,
                     on_first_chunk=lambda: hook_at.append(len(lines_read)))
    # SSE lines: data, blank, data, blank, data(content "a") -> the 5th line,
    # not the first prompt_progress chunk (line 1) still inside the prefill.
    assert hook_at == [5] and out.text == "ab"
