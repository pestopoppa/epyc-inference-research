"""INF-78 OAB-3 (R2): quiescent_after suppression + the trailing-work witness.

Offline and inference-free. The acceptance test injects a real CPU-burning Q-scorer behind
the production launch site (``api.services.memrl.score_completed_task``, the fire-and-forget
MemRL q-scoring submit), lets ``chat()`` return, and witnesses this process's utime+stime
AFTER the reply:
  * without ``quiescent_after`` the scoring outlives the reply and the witness FAILS;
  * with ``quiescent_after=True`` the scoring is never started and the witness PASSES.
The burn is ~1.2 core-seconds on one thread, once. (Until UFH14-B4e, 2026-10-05, the vehicle
was the architect prewarm launch site, ``graph.helpers._maybe_prewarm_architect``; the
prewarmer was deleted, so the test now rides a launch site that still exists.)

Run: .venv/bin/python -m pytest tests/unit/test_oab3_quiescence_witness.py -q
"""
from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.runtime import quiescence as Q
from src.runtime import trailing_work_witness as W


@pytest.fixture(autouse=True)
def _isolated_quiescence(monkeypatch, tmp_path):
    base = tmp_path / "quiescence"
    monkeypatch.setenv(Q.HOLD_DIR_ENV, str(base))
    Q.clear()
    yield base
    Q.clear()
    import shutil

    shutil.rmtree(base, ignore_errors=True)


# ── carrier ──────────────────────────────────────────────────────────────────


def test_flag_absent_is_a_noop(_isolated_quiescence):
    assert Q.begin(False) is None
    assert Q.suppress("typed_decisions_shadow") is False
    assert Q.suppress_scoring("t1") is False
    assert not (_isolated_quiescence / "holds").exists()
    assert Q.quiet_active() is False


def test_carrier_records_and_holds(monkeypatch):
    monkeypatch.setenv(Q.HOLD_AFTER_ENV, "30")
    carrier = Q.begin(True, request_id="req-1", budget_s=100)
    assert Q.suppress("typed_decisions_shadow") and Q.suppress("typed_decisions_shadow")
    assert Q.suppress_scoring("task-9")
    snap = carrier.snapshot()
    assert snap["suppressed"] == ["typed_decisions_shadow", "memrl_q_scoring"]
    assert snap["suppressed_count"] == 3 and snap["hold_after_s"] == 30.0
    assert Q.is_excluded("task-9") and not Q.is_excluded("task-10")
    now = time.time()
    assert Q.quiet_active(now)                      # held for the whole request budget
    assert Q.quiet_active(now + 125)                # 100 s budget + 30 s after
    Q.finish(carrier)                               # the reply: shrink to now + 30 s
    assert Q.quiet_active(now + 10)
    assert not Q.quiet_active(time.time() + 31)     # expired (and pruned)
    assert not Q.quiet_active()


# ── launch sites ─────────────────────────────────────────────────────────────


def test_memrl_scoring_suppressed_and_excluded():
    from src.api.services import memrl

    state = SimpleNamespace(q_scorer=object(), q_scorer_enabled=True)
    pool = MagicMock()
    with patch.object(memrl, "_get_score_pool", return_value=pool):
        memrl.score_completed_task(state, "task-a")
        assert pool.submit.call_count == 1          # production: scored in the background
        Q.begin(True, request_id="r")
        memrl.score_completed_task(state, "task-b")
        assert pool.submit.call_count == 1          # quiescent: never submitted
    assert Q.is_excluded("task-b") and not Q.is_excluded("task-a")


@pytest.mark.asyncio
async def test_idle_scoring_loop_respects_the_hold(monkeypatch):
    from src.api.services import memrl

    calls: list[dict] = []

    class Scorer:
        def score_pending_tasks(self, should_stop=None, skip_task=None):
            calls.append({"stop": should_stop, "skip": skip_task})
            return {"tasks_processed": 0}

    ticks = {"n": 0}

    async def fast_sleep(_s):
        ticks["n"] += 1
        if ticks["n"] > 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(memrl.asyncio, "sleep", fast_sleep)
    monkeypatch.setenv("ORCHESTRATOR_MEMRL_BACKGROUND", "1")
    state = SimpleNamespace(active_requests=0, q_scorer=Scorer(), q_scorer_enabled=True,
                            progress_logger=None)
    carrier = Q.begin(True, request_id="hold-me")
    Q.clear()  # the hold is cross-process; the loop never sees a carrier
    await memrl.background_cleanup(state)
    assert calls == []                              # held: no scoring tick ran
    Q.finish(carrier)
    monkeypatch.setenv(Q.HOLD_AFTER_ENV, "0")
    Q.finish(carrier)
    # The hold deadline is persisted as f"{t:.3f}" (quiescence._write_hold), so a 0 s
    # hold can round UP by <=0.5 ms, and the patched loop re-checks within microseconds.
    # Let wall time pass the rounded deadline so the hold is genuinely expired
    # (was ~1-in-3 flaky without this). time.sleep is real; asyncio.sleep is patched.
    time.sleep(0.005)
    assert not Q.quiet_active()
    ticks["n"] = 0
    await memrl.background_cleanup(state)
    assert len(calls) == 2
    assert calls[0]["stop"] is Q.quiet_active and calls[0]["skip"] is Q.is_excluded


def test_score_pending_tasks_stops_and_skips():
    from orchestration.repl_memory.q_scorer import QScorer

    scored: list[str] = []
    fake = SimpleNamespace(
        _last_score_time=None,
        config=SimpleNamespace(min_score_interval_seconds=300, batch_size=3),
        reader=SimpleNamespace(get_unscored_tasks=lambda: ["x1", "t1", "t2", "t3", "t4"]),
        _score_task=lambda tid: scored.append(tid) or {},
        logger=SimpleNamespace(flush=lambda: None),
    )
    out = QScorer.score_pending_tasks(fake, skip_task=lambda t: t.startswith("x"))
    assert scored == ["t1", "t2", "t3"] and out["tasks_processed"] == 3
    scored.clear()
    fake._last_score_time = None
    stop_after = iter([False, True])
    out = QScorer.score_pending_tasks(fake, should_stop=lambda: next(stop_after))
    assert scored == ["x1"] and out.get("stopped_early") is True


# ── witness (unit) ───────────────────────────────────────────────────────────


def _ledger(entries, t):
    return W.Ledger(read_at_monotonic_s=t, clock_ticks_per_s=100, entries=entries)


def test_verdict_rules():
    before = _ledger({1: {"label": "api", "cpu_ticks": 100, "starttime_ticks": 5},
                      2: {"label": "llama-server:8083", "cpu_ticks": 0, "starttime_ticks": 7}}, 0.0)
    idle = _ledger({1: {"label": "api", "cpu_ticks": 140, "starttime_ticks": 5},
                    2: {"label": "llama-server:8083", "cpu_ticks": 3, "starttime_ticks": 7}}, 60.0)
    v = W.verdict(before, idle)
    assert v["quiescent"] and v["total_core_s"] == pytest.approx(0.43)
    busy = _ledger({1: {"label": "api", "cpu_ticks": 151, "starttime_ticks": 5},
                    2: {"label": "llama-server:8083", "cpu_ticks": 0, "starttime_ticks": 7}}, 60.0)
    v = W.verdict(before, busy)
    assert not v["quiescent"] and v["violations"][0]["reason"] == "cpu_work"
    reused = _ledger({1: {"label": "api", "cpu_ticks": 100, "starttime_ticks": 99},
                      2: {"label": "", "vanished": True},
                      3: {"label": "api-child", "cpu_ticks": 10, "starttime_ticks": 200}}, 60.0)
    reasons = {x["reason"] for x in W.verdict(before, reused)["violations"]}
    assert reasons == {"pid_reused_mid_window", "vanished_mid_window", "appeared_mid_window"}


def test_read_process_cpu_parses_comm_with_parens(tmp_path):
    (tmp_path / "42").mkdir()
    fields = ["S", "1"] + ["0"] * 9 + ["250", "50"] + ["0"] * 6 + ["777"] + ["0"] * 10
    (tmp_path / "42" / "stat").write_text("42 (weird ) (name) " + " ".join(fields))
    assert W.read_process_cpu(42, proc_root=str(tmp_path)) == (300, 777)
    assert W.read_process_cpu(43, proc_root=str(tmp_path)) is None


def test_discovery_finds_api_tree_and_configured_llama_servers(tmp_path):
    def proc(pid, ppid, argv):
        d = tmp_path / str(pid)
        d.mkdir()
        fields = ["S", str(ppid)] + ["0"] * 17 + ["1"] + ["0"] * 10
        (d / "stat").write_text(f"{pid} (x) " + " ".join(fields))
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")

    proc(10, 1, ["python", "-m", "uvicorn", "src.api:app", "--workers", "6"])
    proc(11, 10, ["python", "-c", "from multiprocessing.spawn import spawn_main"])
    proc(12, 11, ["git", "log"])
    proc(20, 1, ["/opt/llama-server", "-m", "x.gguf", "--port", "8083"])
    proc(21, 1, ["/opt/llama-server", "-m", "y.gguf", "--port", "9999"])  # not ours
    proc(30, 1, ["bash"])
    found = W.discover_orchestrator_pids(ports={8083}, proc_root=str(tmp_path))
    assert found == {10: "api", 11: "api-child", 12: "api-child", 20: "llama-server:8083"}


# ── acceptance: injected trailing scoring trips the witness; quiescent_after suppresses it ──


class _FakeHttpRequest:
    async def is_disconnected(self) -> bool:
        return False


def _burn(seconds: float) -> None:
    end = time.thread_time() + seconds
    x = 0
    while time.thread_time() < end:
        x += 1


class _BurningScorer:
    """Stands in for the MemRL Q-scorer: ``_score_task`` burns ~1.2 core-s in the q-scorer
    pool thread, i.e. work the orchestrator process keeps doing after /chat has replied."""

    def _score_task(self, task_id, mode_context=None):
        time.sleep(0.05)  # starts after the handler has returned
        _burn(1.2)


async def _run_chat_then_witness(quiescent_after: bool) -> tuple[dict, object]:
    from src.api.models import ChatRequest, ChatResponse
    from src.api.routes.chat import chat
    from src.api.services.memrl import score_completed_task

    app_state = SimpleNamespace(
        q_scorer=_BurningScorer(), q_scorer_enabled=True, progress_logger=None
    )

    async def fake_handle_chat(*_args, **_kwargs):
        # The real pipeline reaches this launch site when a turn completes.
        score_completed_task(app_state, "oab3-witness-task")
        return ChatResponse(answer="ok", turns=1, elapsed_seconds=0.01, mock_mode=True)

    request = ChatRequest(prompt="t", quiescent_after=quiescent_after)
    with patch("src.api.routes.chat._handle_chat", new=fake_handle_chat):
        response = await chat(request, _FakeHttpRequest(), MagicMock())
        # ── the reply has been produced; witness this process from here ──
        verdict = await asyncio.to_thread(
            W.witness, {os.getpid(): "orchestrator(test)"}, window_s=2.0, threshold_core_s=0.5
        )
        await asyncio.sleep(0)  # let any finished trailing task settle
    return verdict, response


@pytest.mark.asyncio
async def test_injected_trailing_work_trips_the_witness():
    verdict, response = await _run_chat_then_witness(quiescent_after=False)
    assert verdict["quiescent"] is False, verdict
    assert verdict["violations"][0]["reason"] == "cpu_work"
    assert verdict["violations"][0]["cpu_core_seconds"] > 0.5
    assert response.quiescence is None


@pytest.mark.asyncio
async def test_quiescent_after_suppresses_the_trailing_work():
    verdict, response = await _run_chat_then_witness(quiescent_after=True)
    assert verdict["quiescent"] is True, verdict
    assert response.quiescence["suppressed"] == ["memrl_q_scoring"]
    assert Q.is_excluded("oab3-witness-task")  # never scored later by the idle loop either
    assert Q.current() is None  # never outlives the request
