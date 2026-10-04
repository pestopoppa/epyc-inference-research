"""Review of feat/rtg58-p2-prefix-index-gate (B, rebased on 99348f54) — the
stack owner's defect tests (D1-D3 and the v2 flag-off D2 lease proof), landed as
regression tests. Fork tests run with ``FORK=auto`` and a server that reports its
fork (``/props.slot_fork``): without one, fork features stay off by design.

Each test states the scenario it reproduces; every one is flag-ON (the flag-off
path is proven elsewhere).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.backends.context_limits import PoolOccupancy, SlotState
from src.inference import prefix_index as pi
from src.scheduling import kv_pool_admission as kpa
from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

URL = "http://localhost:8083"


class Slots:
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


#: A P1 server's ``/props.slot_fork`` with the fork on (``kv`` mode, no floor).
KV_CAPS = {"min_tokens": 0, "mode": "kv", "checkpoint_at": False}


def _doc(tag: str, lines: int) -> str:
    return "".join(f"{tag} line {i}: shared context for every turn\n" for i in range(lines))


# ── D1: trunk-first never engages for a simultaneous fan-out ─────────────────────


def test_trunk_hold_is_defeated_by_slots_lag_right_after_the_owner_is_admitted(monkeypatch):
    """FORK on. Owner A is admitted; sibling B arrives milliseconds later (a scout
    fan-out). /slots is a cached read (1.5 s TTL) taken BEFORE A reached the
    server, so it shows zero prefilling slots. ``_prefill_over`` treats
    ``server_prefilling == 0`` as "the owner's prefill is over" and B is admitted
    at once: trunk-first does not fire in exactly the case it exists for. The
    existing test pre-sets a busy slot (decoded=0) before A is admitted, which
    hides this."""
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    slots = Slots()  # all idle: the server has not seen A yet
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False,
                                 fork_caps=lambda url: KV_CAPS)
    trunk = _doc("F", 1200)
    owner = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "child A", timeout_s=0)
    assert owner is not None
    child = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "child B", timeout_s=0.5,
                         poll_s=0.01)
    try:
        # Expected: B is held behind A's trunk until A's prefill is seen to end.
        assert pi.get_index(URL).stats["trunk_holds"] == 1
        assert child is None
    finally:
        pool.release(URL, child)
        pool.release(URL, owner)


# ── D2: the scout pool gate never hands the prefill on ───────────────────────────


def test_scout_pool_gate_hands_the_prefill_on_at_first_output():
    """``PoolGatedTransport`` streams its upstream but never calls
    ``pool.prefill_done``. With B, a trunk hold on a scout fan-out can then only
    end by ``/slots`` showing zero prefilling slots (false on a busy :8083), the
    floor-rate deadline (prompt/250 tok/s: ~52 s for a 13k trunk), or
    TRUNK_HOLD_S (120 s). The same gap already holds the long-prefill lease
    through a scout's whole decode."""
    from src.api.routes.chat_pipeline.scout_stage import CompletionResult, PoolGatedTransport

    pool = MagicMock()
    pool.acquire.return_value = 7
    inner = MagicMock()
    inner.complete.return_value = CompletionResult("answer", 10, 2, "stop")
    gate = PoolGatedTransport(inner, url=URL, limit=SimpleNamespace(pool_tokens=393216),
                              pool=pool)
    gate.complete([{"role": "user", "content": "q"}], max_tokens=8,
                  should_stop=lambda: False, timeout_s=30)
    pool.release.assert_called_once()
    assert pool.prefill_done.called, "scout gate never reports first output"


# ── D3: fork credit leaks the shared cells out of the gate's own accounting ─────


def test_fork_credit_survives_the_sources_release_and_under_counts_the_pool(monkeypatch):
    """FORK on. A (13k) is admitted and prefilled; B forks from A's trunk and
    reserves want - shared. A finishes (or fails) and releases; B is still in
    flight and the shared cells still live (B references them), but no ticket
    reserves them any more: ``in_flight_tokens`` under-counts by the credit, so a
    third request is admitted against cells that are occupied. ``/slots`` covers
    this only while it is reachable (``_fits`` = max(own, seen))."""
    monkeypatch.setenv(pi.FORK_ENV, "auto")
    monkeypatch.setenv(pi.TRUNK_MIN_TOKENS_ENV, "1000")
    slots = Slots()
    pool = SharedKVPoolAdmission(occupancy=slots, cross_process=False,
                                 fork_caps=lambda url: KV_CAPS)
    trunk = _doc("G", 1200)
    a = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "A", timeout_s=0)
    pool.prefill_done(URL, a)  # A's trunk is prefilled: a legal fork source
    b = pool.acquire(URL, 13000, 393216, prefix_key=trunk + "B", timeout_s=0)
    info = pool.admission_record(b)["prefix_index"]
    assert info["fork_credit_tokens"] > 0 and info["fork_plan"]["source"] == "inflight"
    want_b_full = pool.reservation_tokens(URL, 13000, 0)
    pool.release(URL, a)
    try:
        # B alone now holds trunk + its own suffix: the pool owes at least B's
        # full reservation.
        assert pool.in_flight_tokens(URL) >= want_b_full
    finally:
        pool.release(URL, b)


# ── v2 (084921bb) flag-OFF proof for the D2 lease hand-off on the scout lane ───


def _lease_pool(monkeypatch):
    monkeypatch.delenv(pi.FLAG_ENV, raising=False)  # flag OFF: the production path
    monkeypatch.setenv(kpa.KV_POOL_LONG_PREFILL_ENV, "100")
    pool = SharedKVPoolAdmission(occupancy=lambda url: None, cross_process=False,
                                 history=SimpleNamespace(lookup=lambda u, l: (None, "no_history")))
    drops: list[str] = []
    real_drop = pool._drop_lease

    def counted_drop(url):
        # _take_lease drops defensively BEFORE recording a lease; count only a
        # drop that ends a lease that exists.
        if url in pool._prefill_lease:
            drops.append(url)
        real_drop(url)

    monkeypatch.setattr(pool, "_drop_lease", counted_drop)
    return pool, drops


def test_v2_flag_off_scout_lease_handed_on_at_first_chunk_exactly_once(monkeypatch):
    from src.api.routes.chat_pipeline.scout_stage import CompletionResult, PoolGatedTransport

    pool, drops = _lease_pool(monkeypatch)
    seen: dict[str, bool] = {}

    class Inner:
        gate_kwargs = True

        def complete(self, messages, *, max_tokens, should_stop, timeout_s,
                     on_first_chunk=None, checkpoint_at=None):
            seen["held_at_dispatch"] = pool.long_prefill_lease_busy(URL)
            on_first_chunk()           # the server's first choices chunk (n_decoded == 1)
            seen["held_after_first"] = pool.long_prefill_lease_busy(URL)
            seen["drops_after_first"] = len(drops)
            on_first_chunk()           # a second call must be a no-op
            return CompletionResult("x", 1, 1, "stop")

    gate = PoolGatedTransport(Inner(), url=URL, limit=SimpleNamespace(pool_tokens=393216), pool=pool)
    gate.complete([{"role": "user", "content": "q" * 3000}], max_tokens=4,
                  should_stop=lambda: False, timeout_s=5)
    assert seen == {"held_at_dispatch": True, "held_after_first": False, "drops_after_first": 1}
    assert drops == [URL]                 # release() did not drop it a second time
    assert pool.in_flight_tokens(URL) == 0 and not pool.long_prefill_lease_busy(URL)
    assert pi.peek_index(URL) is None     # flag off: no index was created


def test_v2_flag_off_scout_exception_after_first_chunk_still_releases_once(monkeypatch):
    from src.api.routes.chat_pipeline.scout_stage import PoolGatedTransport

    pool, drops = _lease_pool(monkeypatch)

    class Inner:
        gate_kwargs = True

        def complete(self, messages, *, max_tokens, should_stop, timeout_s,
                     on_first_chunk=None, checkpoint_at=None):
            on_first_chunk()
            raise RuntimeError("upstream died mid-stream")

    gate = PoolGatedTransport(Inner(), url=URL, limit=SimpleNamespace(pool_tokens=393216), pool=pool)
    with pytest.raises(RuntimeError):
        gate.complete([{"role": "user", "content": "q" * 3000}], max_tokens=4,
                      should_stop=lambda: False, timeout_s=5)
    assert drops == [URL] and pool.in_flight_tokens(URL) == 0
    assert not pool.long_prefill_lease_busy(URL)


def test_v2_flag_off_scout_without_hook_support_hands_on_at_return(monkeypatch):
    from src.api.routes.chat_pipeline.scout_stage import CompletionResult, PoolGatedTransport

    pool, drops = _lease_pool(monkeypatch)
    inner = MagicMock(spec=["complete"])  # no gate_kwargs: the pre-v2 transport shape
    inner.complete.return_value = CompletionResult("x", 1, 1, "stop")
    gate = PoolGatedTransport(inner, url=URL, limit=SimpleNamespace(pool_tokens=393216), pool=pool)
    gate.complete([{"role": "user", "content": "q" * 3000}], max_tokens=4,
                  should_stop=lambda: False, timeout_s=5)
    assert "on_first_chunk" not in inner.complete.call_args.kwargs
    assert drops == [URL] and pool.in_flight_tokens(URL) == 0
