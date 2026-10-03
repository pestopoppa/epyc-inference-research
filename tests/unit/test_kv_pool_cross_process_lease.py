"""KVU-15a: the long-prefill lease is host-wide (one flock per server, across the
API's uvicorn workers) and sized on NEW tokens (a cached prefix is credited).

Offline and inference-free: occupancy is faked and "workers" are either separate
SharedKVPoolAdmission instances or real child Python processes that admit
against a private lease directory. No server is contacted.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.backends.context_limits import (
    ContextLimitResolver,
    PoolOccupancy,
    SlotState,
    common_prefix_chars,
    estimate_tokens_conservative,
    parse_slots,
    set_context_limit_resolver,
)
from src.runtime import long_prefill_lease as lpl
from src.scheduling import kv_pool_admission as kpa
from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

URL = "http://localhost:8083"
POOL = 196608
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (kpa.KV_POOL_LONG_PREFILL_ENV, kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV,
                 kpa.KV_POOL_LONG_PREFILL_OBSERVE_ENV, kpa.KV_POOL_MAX_QUEUED_ENV,
                 kpa.KV_POOL_RATIO_INIT_ENV, kpa.KV_POOL_CROSS_PROCESS_LEASE_ENV,
                 kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV):
        monkeypatch.delenv(name, raising=False)


def _wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


# ── the primitive ─────────────────────────────────────────────────────────────────────────


class TestLeasePrimitive:
    def test_key_is_per_server_and_folds_loopback(self, _hermetic_long_prefill_lease):
        assert lpl.server_key("http://localhost:8083") == "local_8083"
        assert lpl.server_key("http://127.0.0.1:8083/") == "local_8083"
        assert lpl.server_key("localhost:8083") == "local_8083"
        assert lpl.server_key("http://10.0.0.5:8083") == "10.0.0.5_8083"
        assert lpl.server_key("http://localhost:8070") != lpl.server_key(URL)
        assert lpl.lease_path(URL) == _hermetic_long_prefill_lease / "kv_prefill_lease.local_8083.lock"

    def test_exclusive_even_within_one_process_and_released_explicitly(self):
        first = lpl.try_acquire(URL, payload={"ticket": 7})
        assert first is not None and lpl.is_held(URL)
        assert lpl.try_acquire("http://127.0.0.1:8083") is None, "same server, other spelling"
        assert lpl.try_acquire("http://localhost:8070") is not None, "other server is independent"
        info = lpl.holder_info(URL)
        assert info["ticket"] == 7 and info["pid"] == os.getpid()
        first.release()
        first.release()  # idempotent
        assert not lpl.is_held(URL) and lpl.holder_info(URL) is None
        assert lpl.try_acquire(URL) is not None

    def test_never_held_server_is_free(self):
        assert not lpl.is_held("http://localhost:9999")


# ── two workers in one process (two admission instances = two uvicorn workers) ───────────


class TestTwoWorkersOneHost:
    def test_second_worker_waits_for_the_first_workers_long_prefill(self):
        w1 = SharedKVPoolAdmission(occupancy=lambda u: None)
        w2 = SharedKVPoolAdmission(occupancy=lambda u: None)
        t1 = w1.acquire(URL, 40_000, POOL, timeout_s=0)
        assert t1 is not None
        # w2 has no reservation and sees no /slots: before KVU-15a it admitted.
        assert w2.acquire(URL, 40_000, POOL, timeout_s=0.05, poll_s=0.01) is None
        assert w2.long_prefill_lease_busy(URL) and w2.long_prefill_holder(URL) is None
        assert w2.get_status()[URL]["long_prefill_lease_held"] is True
        assert w2.get_status()[URL]["long_prefill_lease_held_here"] is False
        assert w2.acquire(URL, 2_000, POOL, timeout_s=0) is not None, "short is never held"
        got = []
        th = threading.Thread(target=lambda: got.append(
            w2.acquire(URL, 40_000, POOL, timeout_s=5, poll_s=0.01)))
        th.start()
        time.sleep(0.05)
        assert not got
        w1.prefill_done(URL, t1)            # first chunk on worker 1
        th.join(timeout=5)
        assert got and got[0] is not None and w2.long_prefill_holder(URL) == got[0]

    def test_release_and_cancel_paths_free_the_host_lease(self):
        w1 = SharedKVPoolAdmission(occupancy=lambda u: None)
        t1 = w1.acquire(URL, 40_000, POOL, timeout_s=0)
        w1.release(URL, t1, success=False)  # error / cancel / timeout: caller's finally
        assert not lpl.is_held(URL)

    def test_rate_floor_expiry_frees_the_host_lease_without_other_traffic(self, monkeypatch):
        # 40,000 tokens at 200,000 tok/s = 0.2 s; nothing else calls into w1.
        monkeypatch.setenv(kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV, "200000")
        w1 = SharedKVPoolAdmission(occupancy=lambda u: None)
        assert w1.acquire(URL, 40_000, POOL, timeout_s=0) is not None
        assert lpl.is_held(URL)
        assert _wait_until(lambda: not lpl.is_held(URL), timeout=5.0), "timer released it"
        w2 = SharedKVPoolAdmission(occupancy=lambda u: None)
        assert w2.acquire(URL, 40_000, POOL, timeout_s=0) is not None

    def test_env_off_restores_the_per_process_lease(self, monkeypatch):
        monkeypatch.setenv(kpa.KV_POOL_CROSS_PROCESS_LEASE_ENV, "0")
        w1 = SharedKVPoolAdmission(occupancy=lambda u: None)
        w2 = SharedKVPoolAdmission(occupancy=lambda u: None)
        assert w1.acquire(URL, 40_000, POOL, timeout_s=0) is not None
        assert not lpl.is_held(URL)
        assert w2.acquire(URL, 40_000, POOL, timeout_s=0) is not None


# ── real processes ────────────────────────────────────────────────────────────────────────

_REPLAY_CHILD = textwrap.dedent("""
    import os, sys, threading, time
    sys.path.insert(0, os.environ["KVU15A_REPO"])
    from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

    url, log, go = sys.argv[1], sys.argv[2], sys.argv[3]
    threads, rounds = int(sys.argv[4]), int(sys.argv[5])
    prefill_s, decode_s = float(sys.argv[6]), float(sys.argv[7])
    pool = SharedKVPoolAdmission(occupancy=lambda u: None)   # no /slots: lease only
    while not os.path.exists(go):
        time.sleep(0.002)

    def note(kind):
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, f"{kind} {time.time():.6f} {os.getpid()}\\n".encode())
        finally:
            os.close(fd)

    def worker():
        for _ in range(rounds):
            ticket = pool.acquire(url, 40_000, 196608, timeout_s=60, poll_s=0.01)
            assert ticket is not None
            note("start")
            time.sleep(prefill_s)            # the long prefill
            note("end")
            pool.prefill_done(url, ticket)   # first chunk
            time.sleep(decode_s)             # decode runs concurrently
            pool.release(url, ticket)

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts: t.start()
    for t in ts: t.join()
""")


def _peak_overlap(lines: list[str]) -> int:
    events = []
    for line in lines:
        kind, ts, _pid = line.split()
        events.append((float(ts), 0 if kind == "end" else 1))
    peak = cur = 0
    for _ts, delta in sorted(events):
        cur += 1 if delta == 1 else -1
        peak = max(peak, cur)
    return peak


def _replay(tmp_path: Path, lease_dir: Path, *, cross_process: bool, procs=2, threads=2,
            rounds=3) -> list[str]:
    log, go = tmp_path / "events.log", tmp_path / "go"
    env = dict(os.environ, KVU15A_REPO=str(REPO_ROOT), ORCHESTRATOR_TMP_DIR=str(lease_dir),
               PYTHONPATH=str(REPO_ROOT))
    env[kpa.KV_POOL_CROSS_PROCESS_LEASE_ENV] = "1" if cross_process else "0"
    children = [subprocess.Popen(
        [sys.executable, "-c", _REPLAY_CHILD, URL, str(log), str(go), str(threads),
         str(rounds), "0.08", "0.05"], env=env, cwd=str(REPO_ROOT))
        for _ in range(procs)]
    try:
        time.sleep(0.5)        # let both import and reach the start line
        go.write_text("go")
        for child in children:
            assert child.wait(timeout=120) == 0
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    return log.read_text().splitlines()


class TestTwoProcessReplay:
    def test_peak_one_concurrent_long_prefill_per_server_across_processes(self, tmp_path):
        lease_dir = tmp_path / "leases"
        lines = _replay(tmp_path, lease_dir, cross_process=True)
        assert len(lines) == 2 * 2 * 2 * 3, "every prefill started and ended"
        assert len({line.split()[2] for line in lines}) == 2, "both processes prefilled"
        assert _peak_overlap(lines) == 1

    def test_control_per_process_lease_overlaps(self, tmp_path):
        """The replay is sensitive: with the old per-process lease two workers
        DO prefill at once, so the peak-1 result above is not vacuous."""
        lines = _replay(tmp_path, tmp_path / "leases", cross_process=False)
        assert _peak_overlap(lines) >= 2


_DYING_CHILD = textwrap.dedent("""
    import os, signal, sys, time
    sys.path.insert(0, os.environ["KVU15A_REPO"])
    from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

    url, held, go = sys.argv[1], sys.argv[2], sys.argv[3]
    pool = SharedKVPoolAdmission(occupancy=lambda u: None)
    ticket = pool.acquire(url, 40_000, 196608, timeout_s=10)   # lease expiry ~160 s away
    assert ticket is not None
    open(held, "w").write(str(os.getpid()))
    while not os.path.exists(go):
        time.sleep(0.005)
    os.kill(os.getpid(), signal.SIGKILL)   # a crashed worker: no finally, no release
""")


class TestWorkerDeath:
    def test_a_killed_worker_releases_the_lease(self, tmp_path, _hermetic_long_prefill_lease):
        held, go = tmp_path / "held", tmp_path / "go"
        env = dict(os.environ, KVU15A_REPO=str(REPO_ROOT), PYTHONPATH=str(REPO_ROOT),
                   ORCHESTRATOR_TMP_DIR=str(_hermetic_long_prefill_lease))
        child = subprocess.Popen([sys.executable, "-c", _DYING_CHILD, URL, str(held), str(go)],
                                 env=env, cwd=str(REPO_ROOT))
        try:
            assert _wait_until(held.exists, timeout=60)
            assert lpl.is_held(URL)
            assert lpl.holder_info(URL)["pid"] == child.pid
            survivor = SharedKVPoolAdmission(occupancy=lambda u: None)
            assert survivor.acquire(URL, 40_000, POOL, timeout_s=0.05, poll_s=0.01) is None
            go.write_text("die")
            assert child.wait(timeout=30) == -signal.SIGKILL
            assert not lpl.is_held(URL), "the kernel dropped the dead worker's flock"
            assert survivor.acquire(URL, 40_000, POOL, timeout_s=0) is not None
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)


# ── new tokens, not the whole prompt ──────────────────────────────────────────────────────

SYSTEM = "<|im_start|>system\nYou are a careful coding agent.<|im_end|>\n"
HISTORY = SYSTEM + "".join(
    f"<|im_start|>user\nturn {i}: " + ("lorem ipsum dolor sit amet " * 200) + "<|im_end|>\n"
    for i in range(24))                     # ~130k chars, ~43k conservative tokens
NEW_TURN = "<|im_start|>user\nnext: please continue.<|im_end|>\n<|im_start|>assistant\n"


def _occ(*slots):
    return PoolOccupancy(url=URL, slots=tuple(slots))


def _slot(slot_id, text, *, processing=False):
    return SlotState(slot_id=slot_id, n_ctx=POOL, is_processing=processing,
                     n_prompt_tokens=len(text) // 4, n_remain=None,
                     n_decoded=None if not processing else 10, prompt_text=text)


class TestCachedPrefixCredit:
    def test_parse_slots_keeps_the_slot_prompt(self):
        body = [{"id": 0, "is_processing": False, "n_prompt_tokens": 9, "prompt": "<s>hello"},
                {"id": 1, "is_processing": False}]
        occ = parse_slots(URL, body)
        assert occ.slots[0].prompt_text == "<s>hello" and occ.slots[1].prompt_text is None

    def test_common_prefix_tolerates_a_leading_bos(self):
        assert common_prefix_chars(HISTORY + NEW_TURN, "<s>" + HISTORY) == len(HISTORY)
        assert common_prefix_chars(HISTORY, "x" * 500 + HISTORY) == 0, "head too far in"
        assert common_prefix_chars("abc" * 100, "") == 0

    def test_a_request_whose_prefix_is_cached_is_not_held(self):
        prompt = HISTORY + NEW_TURN
        whole = estimate_tokens_conservative(prompt)
        assert whole >= 16384, "the whole prompt alone would be a long prefill"
        holder = SharedKVPoolAdmission(occupancy=lambda u: None)
        assert holder.acquire(URL, 40_000, POOL, timeout_s=0) is not None   # lease taken
        occ = _occ(_slot(0, HISTORY), _slot(1, "unrelated" * 50))
        pool = SharedKVPoolAdmission(occupancy=lambda u: occ)
        # Without the text the old whole-prompt rule holds it ...
        assert pool.acquire(URL, whole, POOL, timeout_s=0.05, poll_s=0.01) is None
        # ... with it, only the new turn (+ margin) is prefill: admitted at once.
        ticket = pool.acquire(URL, whole, POOL, timeout_s=0, prompt_text=prompt)
        assert ticket is not None
        assert pool.long_prefill_holder(URL) is None, "a cached request takes no lease"
        assert pool.get_status()[URL]["admission_stats"]["cache_credited"] == 1
        # The KV reservation is NOT credited: cached cells still occupy the pool.
        assert pool.in_flight_tokens(URL) >= whole

    def test_estimate_is_suffix_plus_margin_capped_at_whole(self, monkeypatch):
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        prompt = HISTORY + NEW_TURN
        whole = estimate_tokens_conservative(prompt)
        est = pool.prefill_tokens_estimate(whole, prompt, len(HISTORY))
        assert est == -(-len(NEW_TURN) // 3) + 4096
        assert pool.prefill_tokens_estimate(whole, prompt, 0) == whole
        assert pool.prefill_tokens_estimate(whole, None, len(HISTORY)) == whole
        assert pool.prefill_tokens_estimate(1000, prompt, len(HISTORY)) == 1000
        monkeypatch.setenv(kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV, "-1")
        assert pool.prefill_tokens_estimate(whole, prompt, len(HISTORY)) == whole

    @pytest.mark.parametrize("case", ["busy_slot", "system_prompt_only", "no_slots",
                                      "credit_disabled"])
    def test_no_reliable_cache_falls_back_to_the_whole_prompt(self, case, monkeypatch):
        prompt = HISTORY + NEW_TURN
        whole = estimate_tokens_conservative(prompt)
        occ = {
            "busy_slot": _occ(_slot(0, HISTORY, processing=True)),
            "system_prompt_only": _occ(_slot(0, SYSTEM + "<|im_start|>user\nsomething else")),
            "no_slots": None,
            "credit_disabled": _occ(_slot(0, HISTORY)),
        }[case]
        if case == "credit_disabled":
            monkeypatch.setenv(kpa.KV_POOL_CACHE_CREDIT_MARGIN_ENV, "-1")
        holder = SharedKVPoolAdmission(occupancy=lambda u: None)
        assert holder.acquire(URL, 40_000, POOL, timeout_s=0) is not None
        pool = SharedKVPoolAdmission(occupancy=lambda u: occ)
        assert pool.acquire(URL, whole, POOL, timeout_s=0.05, poll_s=0.01,
                            prompt_text=prompt) is None, f"{case}: still a long prefill"


# ── inference wiring ──────────────────────────────────────────────────────────────────────


class TestInferenceWiring:
    def _prims(self):
        from src.llm_primitives import LLMPrimitives

        set_context_limit_resolver(ContextLimitResolver(live=False, role_urls=lambda: {},
            registry_facts=lambda: {8083: {"context_tokens": 196608, "slots": 4,
                                           "kv_unified": True, "ctx_max": 262144}}))
        tracker = MagicMock()
        tracker.is_available.return_value = True
        return LLMPrimitives(mock_mode=False, server_urls={"architect_critic": URL},
                             health_tracker=tracker)

    def test_dispatch_holds_the_host_lease_until_the_first_chunk_and_passes_the_text(
            self, monkeypatch):
        from src.model_server import InferenceResult

        monkeypatch.setenv(kpa.KV_POOL_PREFILL_FLOOR_TPS_ENV, "0")
        pool = SharedKVPoolAdmission(occupancy=lambda u: None)
        monkeypatch.setattr(kpa, "_shared_pool_admission", pool)
        seen = SimpleNamespace(text=None, held_before=None, held_after=None)
        real_acquire = pool.acquire

        def spy(*a, **kw):
            seen.text = kw.get("prompt_text")
            return real_acquire(*a, **kw)

        monkeypatch.setattr(pool, "acquire", spy)
        prims = self._prims()

        def stream(role_config, request, on_chunk):
            seen.held_before = lpl.is_held(URL)
            on_chunk("first")
            seen.held_after = lpl.is_held(URL)
            return InferenceResult(role="r", output="first", tokens_generated=1,
                                   generation_speed=1.0, elapsed_time=0.1, success=True)

        backend = MagicMock(spec=[])
        backend.infer_stream_text = MagicMock(side_effect=stream)
        prims._backends["architect_critic"] = backend
        prompt = "x" * (3 * 20_000)
        assert prims._real_call(prompt, "architect_critic", n_tokens=100) == "first"
        assert seen.text == prompt
        assert seen.held_before is True, "host-wide lease held during the prefill"
        assert seen.held_after is False, "first chunk released it host-wide"
        assert not lpl.is_held(URL)
