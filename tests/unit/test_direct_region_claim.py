"""ARCHSWAP 2026-09-27 item A-3: the OAB-8 scouts call a llama-server directly, bypassing
``LLMPrimitives``. A CPU-resident target must still be claimed with the SAME
``cpu_region_lock`` claim the normal call path takes:

  * the scouts take ONE claim per stage, run concurrently inside it, and get status
    ``error`` when it cannot be had;
  * a GPU target (no CPU regions) takes no claim and is unchanged;
  * with the per-region flag off (legacy mode) nothing takes a claim, and the legacy
    ``inference_lock`` is never used.

(The escalation prewarmer, the second direct caller, was deleted in UFH14-B4e.)

Offline: the topology is synthetic, lock files live in ``tmp_path``, model calls go to
fakes, and nothing reaches a real port.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.runtime.cpu_region_lock as crl
import src.runtime.instance_topology as topo
from src.api.routes.chat_pipeline import scout_stage as S
from src.repl_environment import task_root as TR
from src.runtime import direct_region_claim as drc

CPU_PORT = 18074
GPU_PORT = 18083
FULL = frozenset({"q0", "q1", "q2", "q3"})
NUMA = {
    "architect_general": {"instances": [("0-95", CPU_PORT)]},
    "architect_critic": {"instances": [("184-191", GPU_PORT)]},
}


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """Per-region locks ON, synthetic topology, every lock / telemetry file in tmp_path."""
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "1")
    monkeypatch.delenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", raising=False)
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_LIVE_TELEMETRY_DIR", str(tmp_path / "telemetry"))
    monkeypatch.setenv("ORCHESTRATOR_INFERENCE_LOCK_POLL_MS", "10")
    real_lookup = _real_topology_lookup
    monkeypatch.setattr(
        topo, "topology_instance_for_port", lambda port, numa_config=None: real_lookup(port, NUMA)
    )
    monkeypatch.setattr(topo, "get_instance_regions", lambda: topo.build_instance_regions(NUMA))


_real_topology_lookup = topo.topology_instance_for_port


class RecordingClaim:
    """Stands in for ``cpu_region_lock_for_instance``; records calls and hold state."""

    def __init__(self, *, fail: Exception | None = None):
        self.fail = fail
        self.calls: list[tuple[str, int, dict]] = []
        self.held = 0
        self.max_held = 0
        self.lock = threading.Lock()

    def __call__(self, role, idx, **kwargs):
        self.calls.append((role, idx, kwargs))

        @contextlib.contextmanager
        def _cm():
            if self.fail is not None:
                raise self.fail
            with self.lock:
                self.held += 1
                self.max_held = max(self.max_held, self.held)
            try:
                yield {}
            finally:
                with self.lock:
                    self.held -= 1

        return _cm()


def _no_legacy_lock(monkeypatch):
    import src.runtime.inference_lock as legacy

    def _boom(*a, **k):
        raise AssertionError("the legacy inference_lock must never be used by a direct caller")

    monkeypatch.setattr(legacy, "inference_lock", _boom)


# ── resolution ────────────────────────────────────────────────────────────────────────────


def test_resolution_follows_the_normal_path():
    cpu = drc.resolve_claim_target("architect_general", f"http://localhost:{CPU_PORT}")
    assert cpu is not None and cpu.lock_role == "architect_general" and cpu.instance_idx == 0
    assert cpu.regions == FULL and cpu.port == CPU_PORT
    # GPU host lane (HT-only cpus) owns no CPU region: no claim.
    assert drc.resolve_claim_target("architect_critic", f"http://localhost:{GPU_PORT}") is None
    # Unknown port: the normal path's fallback is (role, 0).
    fallback = drc.resolve_claim_target("architect_general", "http://localhost:1")
    assert fallback is not None and fallback.lock_role == "architect_general"


def test_legacy_mode_takes_no_claim(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PER_REGION_LOCKS", "0")
    assert drc.resolve_claim_target("architect_general", f"http://localhost:{CPU_PORT}") is None


# ── scouts ────────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def lane():
    base = Path("/mnt/raid0/llm/tmp") / f"test_a3_scouts_{uuid.uuid4().hex[:10]}"
    root = base / "lane"
    (root / "src").mkdir(parents=True)
    (root / "src" / "k.c").write_text("void hot(void) {\n  int x = 0;\n}\n")
    TR.clear_request_scope()
    yield SimpleNamespace(scope=TR.TaskScope(root=str(root.resolve()), edit_mode=TR.EDIT_MODE_NONE))
    TR.clear_request_scope()
    shutil.rmtree(base, ignore_errors=True)


class _Resolver:
    def pool_occupancy(self, url):
        from src.backends.context_limits import PoolOccupancy, SlotState

        slots = tuple(
            SlotState(slot_id=i, n_ctx=8192, is_processing=False, n_prompt_tokens=0, n_remain=None)
            for i in range(4)
        )
        return PoolOccupancy(url=url, slots=slots)


class _Transport:
    """Every scout summarises on its first call; records whether the claim was held."""

    name = "fake"

    def __init__(self, claim: RecordingClaim | None, barrier: threading.Barrier | None = None):
        self.claim, self.barrier = claim, barrier
        self.held_at_call: list[int] = []
        self.lock = threading.Lock()

    def complete(self, messages, *, max_tokens, should_stop, timeout_s):
        with self.lock:
            self.held_at_call.append(self.claim.held if self.claim is not None else -1)
        if self.barrier is not None:
            self.barrier.wait()
        return S.CompletionResult(
            "SUMMARY\nsrc/k.c:1 is hot", prompt_tokens=10, completion_tokens=5
        )


def _targets(n):
    return [{"file": "src/k.c", "label": f"T-{i}"} for i in range(n)]


def _scout_spec(n):
    return {
        "enabled": True,
        "targets": _targets(n),
        "max": 4,
        "max_turns": 2,
        "summary_tokens": 200,
        "budget_s": 10.0,
        "reserve_slots": 1,
    }


def test_scouts_take_one_claim_and_stay_concurrent_inside_it(lane, monkeypatch):
    _no_legacy_lock(monkeypatch)
    claim = RecordingClaim()
    monkeypatch.setattr(crl, "cpu_region_lock_for_instance", claim)
    transport = _Transport(claim, barrier=threading.Barrier(3, timeout=10))
    deadline = time.perf_counter() + 60
    stage = asyncio.run(
        S.run_scouts(
            _scout_spec(3),
            scope=lane.scope,
            role="architect_general",
            url=f"http://127.0.0.1:{CPU_PORT}",
            transport=transport,
            resolver=_Resolver(),
            request_deadline_s=deadline,
        )
    )
    rep = stage.report
    assert rep["completed"] == 3 and rep["max_inflight_calls"] == 3, "still concurrent"
    assert len(claim.calls) == 1, "ONE claim for the whole stage, not one per call"
    role, idx, kwargs = claim.calls[0]
    assert (role, idx) == ("architect_general", 0)
    assert kwargs["deadline_s"] == deadline and callable(kwargs["cancel_check"])
    assert transport.held_at_call == [1, 1, 1], "every scout call ran inside the claim"
    assert claim.held == 0, "released when the stage returned"
    assert rep["region_claim"]["status"] == "held" and rep["region_claim"]["released"] is True
    assert rep["region_claim"]["target"]["regions"] == sorted(FULL)


def test_scouts_wait_for_the_real_claim_then_run(lane, monkeypatch):
    """Real lock files: an outside holder delays the scouts until it releases."""
    transport = _Transport(None)
    release_at = {}

    def _holder(ready: threading.Event):
        with crl.cpu_region_lock_for_instance(
            "architect_general", 0, timeout_s=5.0, request_tag="autokernel-window"
        ):
            ready.set()
            time.sleep(0.3)
            release_at["t"] = time.perf_counter()

    ready = threading.Event()
    th = threading.Thread(target=_holder, args=(ready,))
    th.start()
    assert ready.wait(5)
    stage = asyncio.run(
        S.run_scouts(
            _scout_spec(2),
            scope=lane.scope,
            role="architect_general",
            url=f"http://127.0.0.1:{CPU_PORT}",
            transport=transport,
            resolver=_Resolver(),
            request_deadline_s=time.perf_counter() + 30,
        )
    )
    th.join(5)
    rep = stage.report
    assert rep["completed"] == 2
    assert rep["region_claim"]["status"] == "held" and rep["region_claim"]["wait_s"] >= 0.1


def test_contended_scout_claim_errors_the_scouts_and_the_stage_returns(lane, monkeypatch):
    claim = RecordingClaim(fail=crl.CpuRegionLockTimeout("region lock deadline exceeded"))
    monkeypatch.setattr(crl, "cpu_region_lock_for_instance", claim)
    transport = _Transport(claim)
    stage = asyncio.run(
        S.run_scouts(
            _scout_spec(2),
            scope=lane.scope,
            role="architect_general",
            url=f"http://127.0.0.1:{CPU_PORT}",
            transport=transport,
            resolver=_Resolver(),
            request_deadline_s=time.perf_counter() + 30,
        )
    )
    rep = stage.report
    assert transport.held_at_call == [], "no scout call without the claim"
    assert [r.status for r in stage.results] == ["error", "error"]
    assert all("region claim timeout" in (r.error or "") for r in stage.results)
    assert rep["region_claim"]["status"] == "timeout" and stage.block == ""


def test_gpu_target_scouts_are_unchanged(lane, monkeypatch):
    claim = RecordingClaim()
    monkeypatch.setattr(crl, "cpu_region_lock_for_instance", claim)
    transport = _Transport(claim)
    stage = asyncio.run(
        S.run_scouts(
            _scout_spec(2),
            scope=lane.scope,
            role="architect_critic",
            url=f"http://127.0.0.1:{GPU_PORT}",
            transport=transport,
            resolver=_Resolver(),
        )
    )
    assert stage.report["completed"] == 2 and claim.calls == []
    assert stage.report["region_claim"]["status"] == "none"
    assert transport.held_at_call == [0, 0]


def test_cancelled_request_releases_a_claim_it_was_waiting_for(lane, monkeypatch):
    """Cancelling the request mid-wait stops the acquire; no lock outlives the handler."""
    transport = _Transport(None)
    hold = threading.Event()
    ready = threading.Event()

    def _holder():
        with crl.cpu_region_lock_for_instance("architect_general", 0, timeout_s=5.0):
            ready.set()
            hold.wait(5)

    th = threading.Thread(target=_holder)
    th.start()
    assert ready.wait(5)

    async def _run():
        task = asyncio.ensure_future(
            S.run_scouts(
                _scout_spec(2),
                scope=lane.scope,
                role="architect_general",
                url=f"http://127.0.0.1:{CPU_PORT}",
                transport=transport,
                resolver=_Resolver(),
                request_deadline_s=time.perf_counter() + 30,
            )
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    hold.set()
    th.join(5)
    assert transport.held_at_call == []
    # The regions are free again: a fresh claim succeeds at once.
    with crl.cpu_region_lock_for_instance("architect_general", 0, timeout_s=0.5):
        pass
