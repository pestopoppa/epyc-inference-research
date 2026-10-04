"""CPU windows: the claim yields during actor phases and comes back before any CPU step.

The region-lock provider is a FAKE orchestrator module (the `test_direct_held_claim`
pattern) that takes real temporary kernel flocks, so the original `HeldCpuClaim`
observes real locks and its retained interval is checked by the scheduler's own
component validator. A "peer" is a second open file description flocking the same
files: it genuinely contends. No physical region, orchestrator process, bus file or
model is touched.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import fcntl
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType
from unittest import mock

import pytest

from . import claim, cpu_window, loop, pipeline, pool, serial_scheduling


class CpuRegionLockTimeout(RuntimeError):
    """Same class NAME as the orchestrator's; `claim.region_lock_busy` matches on it."""


class FakeProvider:
    def __init__(self, tmp_path: Path) -> None:
        self.paths = {"global": tmp_path / "cpu_region.GLOBAL.q0.lock",
                      "role": tmp_path / "cpu_region.autokernel-cpu.q0.lock"}
        self.events: list[str] = []

    @contextmanager
    def cpu_region_lock(self, role, regions, *, timeout_s=None, cancel_check=None,
                        request_tag=None, **_kwargs):
        handles = [self.paths[name].open("a") for name in ("global", "role")]
        entered = False
        try:
            deadline = time.monotonic() + (timeout_s or 0)
            for handle in handles:
                while True:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if cancel_check is not None and cancel_check():
                            raise CpuRegionLockTimeout("cancelled") from None
                        if time.monotonic() >= deadline:
                            raise CpuRegionLockTimeout("timeout") from None
                        time.sleep(0.005)
            entered = True
            self.events.append("enter")
            yield {"q0": self.paths["role"]}
        finally:
            for handle in handles:
                handle.close()
            if entered:
                self.events.append("exit")


class Peer:
    """Another holder of the same regions (a second open file description)."""

    def __init__(self, provider: FakeProvider) -> None:
        self.paths = provider.paths
        self.handles = []

    def hold(self) -> None:
        self.handles = [self.paths[name].open("a") for name in ("global", "role")]
        for handle in self.handles:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)   # raises if not free

    def release(self) -> None:
        for handle in self.handles:
            handle.close()
        self.handles = []


@pytest.fixture
def provider(tmp_path, monkeypatch):
    fake = FakeProvider(tmp_path)
    modules = {name: ModuleType(name) for name in (
        "src.runtime.cpu_region_lock", "src.runtime.instance_topology",
        "src.runtime.region_lock_cli")}
    modules["src.runtime.cpu_region_lock"].cpu_region_lock = fake.cpu_region_lock
    modules["src.runtime.cpu_region_lock"].global_region_lock_path = \
        lambda region: fake.paths["global"]
    modules["src.runtime.cpu_region_lock"].CpuRegionLockTimeout = CpuRegionLockTimeout
    modules["src.runtime.instance_topology"].cpu_list_to_regions = lambda cpus: frozenset({"q0"})
    modules["src.runtime.instance_topology"].ATOMIC_REGIONS = ("q0",)
    modules["src.runtime.region_lock_cli"]._preflight = lambda **kwargs: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    monkeypatch.setattr(claim, "_ensure_orchestrator_importable", lambda: None)
    return fake


def _log(tmp_path: Path, rows) -> Path:
    path = tmp_path / "actor-replies" / "actor-calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"schema": "epyc.autokernel.actor_call.v1", "role": role, "wall_s": wall})
             for role, wall in rows]
    lines.append(json.dumps({"schema": "epyc.autokernel.actor_call_metrics.v1",
                             "role": "planner", "wall_s": 99999}))   # a sibling row: ignored
    path.write_text("\n".join(lines) + "\n")
    return path


class FakeBus:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        return mock.Mock(returncode=0, stderr="")


def _window(tmp_path, *, now=None, stop=None, bus=None, ttl_s=120.0, wait_bound_s=0.1,
            ledger=True, log_rows=None):
    clock = (lambda: now[0]) if now is not None else time.time
    publisher = cpu_window.BusPublisher("autokernel", root=tmp_path / "root",
                                        runner=bus or FakeBus(), synchronous=True)
    estimator = cpu_window.PhaseEstimator(
        _log(tmp_path, log_rows if log_rows is not None else [
            ("planner", 600), ("planner", 900), ("planner", 5000),
            ("critic", 100), ("critic", 200), ("critic", 300),
            ("author", 1200)]),
        planner_budget_s=2700, author_budget_s=7200, critic_timeout_s=7200)
    return cpu_window.CpuWindow(
        campaign="ak-test", path=tmp_path / "global" / "cpu-window.json",
        campaign_path=tmp_path / "store" / cpu_window.CAMPAIGN_NAME,
        ledger_dir=(tmp_path / "batch") if ledger else None, estimator=estimator,
        wait_bound_s=wait_bound_s, poll_s=0.02, should_stop=stop or (lambda: False),
        bus=publisher, reserved={"cpus": "88-95", "compile": "92-95", "op_test": "88-95",
                                 "source": "affinity_tail"},
        heartbeat_s=None, ttl_s=ttl_s, clock=clock, log=lambda _text: None)


def _read(tmp_path):
    return json.loads((tmp_path / "global" / "cpu-window.json").read_text())


def _events(tmp_path):
    path = tmp_path / "global" / "cpu-window.events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---- the lease ----------------------------------------------------------------------

def test_yield_and_reacquire_keep_the_original_evidence_valid(provider):
    peer = Peer(provider)
    with claim.hold_cpu("0") as receipt:
        lease = claim.yield_lease(receipt)
        assert lease is not None and lease.held and receipt.observe()["status"] == "held"
        assert lease.release(reason="actor:planner") is True
        assert not lease.held and receipt.observe()["status"] == "lost"
        peer.hold()             # the regions are genuinely free while yielded
        peer.release()
        segment = lease.reacquire(reason="build_and_measure")
        assert lease.held and lease.generation == 2 and segment["acquire_reason"] == "build_and_measure"
        assert receipt.observe()["status"] == "held"
    assert provider.events == ["enter", "exit", "enter", "exit"]
    interval = receipt.retained_interval()
    # The scheduler's own validator: open AND close are same-owner observations.
    serial_scheduling._component(interval)
    ledger = lease.ledger()
    assert ledger["continuous"] is False and ledger["generation"] == 2
    first, second = ledger["segments"]
    assert first["release_reason"] == "actor:planner" and first["released_at"] is not None
    assert second["acquired_at"] >= first["released_at"]


def test_reacquire_waits_on_a_held_peer_past_the_bound_and_never_preempts(provider):
    peer = Peer(provider)
    bounds, waits = [], []
    with claim.hold_cpu("0") as receipt:
        lease = claim.yield_lease(receipt)
        lease.release(reason="actor:author")
        peer.hold()
        done = threading.Event()

        def take_back():
            lease.reacquire(reason="build_and_measure", wait_bound_s=0.05, poll_s=0.02,
                            on_bound=bounds.append, on_wait=waits.append)
            done.set()

        thread = threading.Thread(target=take_back)
        thread.start()
        assert _wait_until(lambda: bounds)          # past the bound: logged ...
        time.sleep(0.1)
        assert not done.is_set() and not lease.held  # ... and still waiting, peer intact
        assert len(bounds) == 1 and peer.handles
        peer.release()
        thread.join(5)
        assert done.is_set() and lease.held
        assert lease.segments[-1]["wait_exceeded_bound"] is True
        assert lease.segments[-1]["waited_s"] >= 0.05
    serial_scheduling._component(receipt.retained_interval())


def test_stop_while_waiting_refuses_with_the_claim_released_and_close_still_observes(provider):
    peer = Peer(provider)
    with claim.hold_cpu("0") as receipt:
        lease = claim.yield_lease(receipt)
        lease.release(reason="actor:planner")
        peer.hold()
        with pytest.raises(claim.ClaimRefused, match="stop requested"):
            lease.reacquire(reason="build", should_stop=lambda: True, poll_s=0.02)
        assert not lease.held
        peer.release()
    # The claim's own close re-acquired for a real close observation.
    assert receipt._closed["status"] == "held"
    serial_scheduling._component(receipt.retained_interval())


def test_an_unused_lease_is_the_historical_claim_byte_for_byte(provider):
    with claim.hold_cpu("0") as receipt:
        assert set(receipt) == {"device_id", "cpu_list", "regions", "lock_paths", "pid"}
        serialized = json.dumps(receipt, sort_keys=True)
    assert provider.events == ["enter", "exit"]
    assert json.loads(serialized) == dict(receipt)
    assert claim.yield_lease(receipt).ledger()["continuous"] is True
    serial_scheduling._component(receipt.retained_interval())
    assert claim.yield_lease({"device_id": "cpu"}) is None


# ---- the window ---------------------------------------------------------------------

def test_window_opens_on_actor_phases_closes_on_critic2_and_reacquires_in_the_tail(
        provider, tmp_path):
    now = [1_800_000_000.0]
    bus = FakeBus()
    window = _window(tmp_path, now=now, bus=bus)
    peer = Peer(provider)
    with claim.hold_cpu("0") as receipt:
        assert window.bind(receipt) is True
        lease = window.lease
        assert _read(tmp_path)["state"] == "closed" and _read(tmp_path)["loop_holds_claim"]

        window.note_step("lane0", "proposing a hypothesis")
        body = _read(tmp_path)
        assert body["state"] == "open" and not body["loop_holds_claim"] and not lease.held
        assert body["phase"] == "planner" and body["opened_at"] is not None
        # planner min(median 900, budget 2700) + critic 200 + author 1200 + critic 200
        assert body["est_close_at"] == cpu_window._iso(now[0] + 900 + 200 + 1200 + 200)
        assert body["est_close_basis"]["per_phase"]["planner"]["samples"] == 3
        assert body["cpus_reserved_by_loop"] == "88-95" and "ak-check" in body["contention"]
        peer.hold()                    # a peer fits a measurement into the window
        peer.release()

        now[0] += 600
        window.note_step("lane0", "critic pass 1: reviewing the hypothesis")
        assert _read(tmp_path)["phase"] == "critic1" and _read(tmp_path)["state"] == "open"
        opened_at = _read(tmp_path)["opened_at"]
        window.note_step("lane0", "authoring the patch")
        assert _read(tmp_path)["opened_at"] == opened_at      # one window across phases

        now[0] += 1000
        window.note_step("lane0", "critic pass 2: reviewing the diff")
        body = _read(tmp_path)
        assert body["state"] == "closing" and body["closing_reason"] == "critic2_started"
        assert body["est_close_at"] == cpu_window._iso(now[0] + 200)   # first CPU step ETA
        assert not lease.held

        window.note_step("lane0", "building and gating")
        with window.wrap_fence(None)():
            assert lease.held
            body = _read(tmp_path)
            assert body["state"] == "closed" and body["loop_holds_claim"] and body["wait"] is None
        assert lease.held                         # held until the next actor phase
        window.note_step("lane0", "proposing a hypothesis")
        assert not lease.held and _read(tmp_path)["state"] == "open"
        window.teardown()
        assert lease.held and _read(tmp_path)["state"] == "closed"
    window.finalize()
    body = _read(tmp_path)
    assert body["state"] == "closed" and body["loop_holds_claim"] is False
    assert body["closing_reason"] == "loop_exit" and body["phase"] == "exited"
    serial_scheduling._component(receipt.retained_interval())
    states = [event["state"] for event in _events(tmp_path)]
    # closing (critic2) -> closing (re-acquire started) -> closed (held), and the same
    # pair at teardown; the last closed is the exit.
    assert states == ["closed", "open", "open", "open", "closing", "closing", "closed",
                      "open", "closing", "closed", "closed"]
    ledger = json.loads((tmp_path / "batch" / cpu_window.LEDGER_NAME).read_text())
    assert ledger["claim"]["generation"] == 3 and not ledger["claim"]["continuous"]
    assert (tmp_path / "store" / cpu_window.CAMPAIGN_NAME).is_file()
    # Bus: FYI broadcasts from the loop's OWN outbox, one per event.
    assert len(bus.calls) == len(states)
    argv = bus.calls[1]
    assert argv[argv.index("--agent") + 1] == "autokernel"
    assert argv[argv.index("--target") + 1] == "outbox"
    message = json.loads(argv[argv.index("--json") + 1])
    assert message["to"] == "*" and message["kind"] == "status"
    assert "action_required" not in message
    assert message["payload"]["event"] == cpu_window.BUS_EVENT
    assert message["payload"]["state"] == "open" and message["payload"]["campaign"] == "ak-test"


def test_tail_waits_while_a_peer_holds_logs_past_the_bound_then_closes(provider, tmp_path):
    window = _window(tmp_path, wait_bound_s=0.05)
    peer = Peer(provider)
    with claim.hold_cpu("0") as receipt:
        window.bind(receipt)
        window.note_step("lane0", "authoring the patch")
        peer.hold()
        inside = []

        def tail():
            with window.wrap_fence(None)():
                inside.append(window.lease.held)

        thread = threading.Thread(target=tail)
        thread.start()
        assert _wait_until(lambda: (_read(tmp_path).get("wait") or {}).get("exceeded"))
        body = _read(tmp_path)
        assert body["state"] == "closing" and not body["loop_holds_claim"] and not inside
        peer.release()
        thread.join(5)
        assert inside == [True] and _read(tmp_path)["state"] == "closed"
    window.finalize()


def test_a_lane_in_the_tail_keeps_the_claim_for_every_other_lane(provider, tmp_path):
    window = _window(tmp_path)
    with claim.hold_cpu("0") as receipt:
        window.bind(receipt)
        with window.wrap_fence(None)():
            window.note_step("lane1", "proposing a hypothesis")
            assert window.lease.held and _read(tmp_path)["state"] == "closed"
        window.note_step("lane1", "authoring the patch")
        assert not window.lease.held and _read(tmp_path)["state"] == "open"
        window.teardown()
    window.finalize()


def test_stop_keeps_the_claim_released_and_refuses_the_tail_until_teardown(provider, tmp_path):
    stop = {"asked": False}
    window = _window(tmp_path, stop=lambda: stop["asked"])
    peer = Peer(provider)
    with claim.hold_cpu("0") as receipt:
        window.bind(receipt)
        window.note_step("lane0", "proposing a hypothesis")
        stop["asked"] = True                          # stop/pause asked mid-actor phase
        window.note_step("lane0", "critic pass 1: reviewing the hypothesis")
        assert not window.lease.held                  # nothing re-takes it on its own
        peer.hold()
        with pytest.raises(loop.TailRefused, match="stop requested"):
            with window.wrap_fence(None)():
                pytest.fail("measured without the claim")
        body = _read(tmp_path)
        assert not window.lease.held and body["state"] == "closing" \
            and not body["loop_holds_claim"]
        peer.release()
        window.teardown()                             # the claim's close observation
        assert window.lease.held
    window.finalize()
    assert _read(tmp_path)["state"] == "closed"
    serial_scheduling._component(receipt.retained_interval())


def test_a_crash_while_yielded_goes_stale_and_an_exception_leaves_it_closed(provider, tmp_path):
    window = _window(tmp_path, ttl_s=30.0)
    with claim.hold_cpu("0") as receipt:
        window.bind(receipt)
        window.note_step("lane0", "proposing a hypothesis")
        # A hard kill here runs nothing more: the file says open, but only until
        # its heartbeat expires.
        path = tmp_path / "global" / "cpu-window.json"
        assert cpu_window.read_window(path)["effective_state"] == "open"
        assert cpu_window.read_window(path, now=time.time() + 31)["effective_state"] == "stale"
        window.teardown()
    # The next batch's bind overwrites the stale file at once.
    later = _window(tmp_path)
    with claim.hold_cpu("0") as receipt:
        later.bind(receipt)
        assert cpu_window.read_window(path)["effective_state"] == "closed"
        later.teardown()
    later.finalize()

    # A Python-level failure while yielded unwinds exactly as run.py registers it.
    failing = _window(tmp_path)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with ExitStack() as ownership:
            ownership.callback(failing.finalize)
            receipt = ownership.enter_context(claim.hold_cpu("0"))
            failing.bind(receipt)
            ownership.callback(failing.teardown)
            failing.note_step("lane0", "authoring the patch")
            assert _read(tmp_path)["state"] == "open"
            raise Boom()
    body = cpu_window.read_window(path)
    assert body["state"] == "closed" and body["loop_holds_claim"] is False
    assert body["effective_state"] == "stale"   # no live owner: the heartbeat is expired
    serial_scheduling._component(receipt.retained_interval())


def test_initial_acquisition_waits_for_a_peer_instead_of_refusing_the_batch(provider, tmp_path):
    window = _window(tmp_path)
    peer = Peer(provider)
    peer.hold()
    got = []
    with ExitStack() as ownership:
        thread = threading.Thread(target=lambda: got.append(window.acquire_initial(
            lambda: ownership.enter_context(claim.hold_cpu("0")))))
        thread.start()
        assert _wait_until(lambda: (tmp_path / "global" / "cpu-window.json").exists()
                           and _read(tmp_path)["closing_reason"] == "startup_acquire")
        assert _read(tmp_path)["state"] == "closing" and not got
        peer.release()
        thread.join(10)
        assert got and window.bind(got[0])
        assert _read(tmp_path)["wait"] is None and _read(tmp_path)["loop_holds_claim"]
    window.finalize()


def test_initial_acquisition_still_raises_anything_but_a_busy_region(tmp_path):
    window = _window(tmp_path)
    with pytest.raises(claim.ClaimRefused, match="preflight"):
        window.acquire_initial(lambda: (_ for _ in ()).throw(claim.ClaimRefused("preflight")))
    stopping = _window(tmp_path, stop=lambda: True)
    with pytest.raises(claim.ClaimRefused, match="stop requested"):
        stopping.acquire_initial(lambda: (_ for _ in ()).throw(CpuRegionLockTimeout("busy")))


def test_a_claim_without_a_lease_publishes_nothing_and_changes_nothing(tmp_path):
    window = _window(tmp_path)
    assert window.bind({"device_id": "cpu", "regions": ["q0"]}) is False
    window.note_step("lane0", "proposing a hypothesis")
    with window.wrap_fence(None)():
        pass
    window.teardown()
    window.finalize()
    assert not (tmp_path / "global").exists() and not (tmp_path / "store").exists()
    assert not (tmp_path / "batch").exists()


def test_phase_estimates_use_recent_medians_capped_by_budget(tmp_path):
    estimator = cpu_window.PhaseEstimator(
        _log(tmp_path, [("planner", 9000), ("planner", 8000), ("critic", 50)]),
        planner_budget_s=2700, author_budget_s=3000, critic_timeout_s=7200)
    seconds, basis = estimator.estimate(("planner", "critic1", "author", "critic2"))
    assert basis["planner"]["seconds"] == 2700            # median 8500 capped
    assert basis["critic1"]["median_wall_s"] == 50
    assert basis["author"]["samples"] == 0 and basis["author"]["seconds"] == 3000
    assert seconds == 2700 + 50 + 3000 + 50
    missing = cpu_window.PhaseEstimator(tmp_path / "absent.jsonl", planner_budget_s=1,
                                        author_budget_s=2, critic_timeout_s=3)
    assert missing.estimate(("critic2",))[0] == 3


def test_classification_covers_every_actor_label_and_nothing_else():
    assert cpu_window.classify("proposing a hypothesis") == "planner"
    assert cpu_window.classify("critic pass 1: reviewing the hypothesis (resuming x)") == "critic1"
    assert cpu_window.classify("critic pass 1: empty reply, retrying once") == "critic1"
    assert cpu_window.classify("authoring the patch") == "author"
    assert cpu_window.classify("critic pass 2: reviewing the diff") == "critic2"
    for label in ("critic pass 2: carried verdict (resumed at build)", "building and gating",
                  "measuring A/B on the device", "restoring the retained patch (no author call)",
                  "authoring report derived from the lane diff",
                  "prevalidated runtime option: deterministic checks, no critic call"):
        assert cpu_window.classify(label) is None


def test_a_refused_bus_append_disables_the_bus_and_never_raises(tmp_path):
    def refused(argv, **_kw):
        return mock.Mock(returncode=2, stderr="session_bus: not a roster id 'autokernel'")
    logs = []
    bus = cpu_window.BusPublisher("autokernel", runner=refused, synchronous=True,
                                  log=logs.append)
    bus.emit({"event": "x"})
    bus.emit({"event": "y"})
    assert bus.disabled_reason and len(logs) == 1 and "roster" in logs[0]
    assert cpu_window.BusPublisher(None).disabled_reason == "no bus agent configured"


# ---- wiring -------------------------------------------------------------------------

def test_pool_drive_without_a_window_is_unchanged_and_with_one_is_wired(tmp_path):
    captured = []

    def fake_run_pool(**kwargs):
        captured.append(kwargs)
        kwargs["on_step"]("lane0", "proposing a hypothesis")
        return []

    workers = [pipeline.Worker("lane0", tmp_path / "w0", tmp_path / "b0")]
    common = dict(workers=workers, make_planner=None, make_critic=None, build_context=dict,
                  make_gate=None, make_measure=None, record=lambda _o: None, iterations=1,
                  reset=lambda _w: "0" * 40, commit=lambda *_a: "0" * 40)
    with mock.patch.object(pool.pipeline, "run_pool", fake_run_pool):
        pool.drive(**common)
        window = mock.Mock()
        window.wrap_fence.return_value = "wrapped-fence"
        pool.drive(**common, cpu_window=window)
    assert captured[0]["tail"]._fence is None
    assert captured[1]["tail"]._fence == "wrapped-fence"
    window.wrap_fence.assert_called_once_with(None)
    window.note_step.assert_called_once_with("lane0", "proposing a hypothesis")


def test_run_parser_defaults_on_and_serial_common_args_admit_the_knobs(tmp_path):
    from . import run, serial_roster, serial_run as sr
    from .test_serial_roster import _inputs
    source = Path(run.__file__).read_text(encoding="utf-8")
    assert '"--cpu-window-yield", choices=("on", "off"), default="on"' in source
    _resolved, _owners, argv = _inputs(tmp_path, backends=("cpu",))
    common = tmp_path / "common.json"
    common.write_text(json.dumps(["--cpu-window-yield", "off",
                                  "--cpu-window-wait-bound-s", "600",
                                  "--cpu-window-bus-agent=off"]))
    targets, _skipped, _cpus = serial_roster.build_targets(
        Path(sr.option(argv, "--resolved-campaign")), Path(sr.option(argv, "--owned-targets")),
        target_root=Path(sr.option(argv, "--state-dir")) / "targets", common_path=common)
    assert sr.option(targets[0], "--cpu-window-yield") == "off"
    assert sr.option(targets[0], "--cpu-window-wait-bound-s") == "600"


@pytest.mark.parametrize("knob", ["on", "off"])
def test_actual_main_measures_only_on_a_held_claim_and_off_is_the_batch_long_claim(
        provider, tmp_path, knob):
    """The installed five-iteration CPU lifecycle (`test_existing_cpu_run`), with the
    real `hold_cpu` over the fake provider nested inside the fixture's own claim double.
    Every compile, oracle and serving measurement records whether the lease was held."""
    from . import gates, run, serving, test_existing_cpu_run as cpu_fixture
    genuine_main, real_hold, real_drive = run.main, claim.hold_cpu, pool.drive
    seen, steps, drives = {}, [], []
    window_path = tmp_path / "win" / "cpu-window.json"

    def wrapper(argv):
        # Reached as the fixture's "real main" (it captured `run.main` = this).
        fixture_hold = claim.hold_cpu        # the fixture's double, installed by its cpu_main

        @contextmanager
        def nested(cpu_list, **kwargs):
            # A CPU run asks for gpu-quiet SHARED under the default quiet window; the
            # fake provider here has no gpu-quiet owner (test_gpu_quiet_measurement_window
            # covers that path), so the lease is exercised on the regions alone.
            seen["hold_kwargs"] = kwargs
            with fixture_hold(cpu_list):
                with real_hold(cpu_list) as receipt:
                    seen["receipt"] = receipt
                    yield receipt

        def recorded(name, original):
            def call(*args, **kwargs):
                steps.append((name, claim.yield_lease(seen["receipt"]).held))
                return original(*args, **kwargs)
            return call

        def drive(*args, **kwargs):
            drives.append("cpu_window" in kwargs)
            return real_drive(*args, **kwargs)

        guard = (mock.patch.object(cpu_window, "CpuWindow", side_effect=AssertionError(
                     "window constructed with the knob off"))
                 if knob == "off" else mock.patch.object(cpu_window, "HEARTBEAT_S", 30.0))
        with mock.patch.object(claim, "hold_cpu", nested), \
                mock.patch.object(gates, "compiles", recorded("compile", gates.compiles)), \
                mock.patch.object(gates, "op_correctness",
                                  recorded("oracle", gates.op_correctness)), \
                mock.patch.object(serving, "_measure_once",
                                  recorded("measure", serving._measure_once)), \
                mock.patch.object(pool, "drive", drive), guard:
            return genuine_main([*argv, "--cpu-window-yield", knob,
                                 "--cpu-window-path", str(window_path),
                                 "--cpu-window-bus-agent", "off"])

    with mock.patch.object(run, "main", wrapper):
        cpu_fixture.test_existing_main_cpu_five_iterations_preserves_canonical_champion(False)
    assert steps and all(held for _name, held in steps), steps
    assert {name for name, _held in steps} >= {"compile", "oracle", "measure"}
    assert seen["hold_kwargs"] == {"gpu_quiet": True}   # the default quiet window
    lease = claim.yield_lease(seen["receipt"])
    serial_scheduling._component(seen["receipt"].retained_interval())
    if knob == "off":
        assert drives == [False] and lease.generation == 1
        assert provider.events == ["enter", "exit"] and not window_path.parent.exists()
        return
    assert drives == [True] and lease.generation > 1      # it yielded and came back
    body = cpu_window.read_window(window_path)
    assert body["state"] == "closed" and body["loop_holds_claim"] is False
    events = [json.loads(line) for line in
              window_path.with_name("cpu-window.events.jsonl").read_text().splitlines()]
    assert "open" in {event["state"] for event in events}
    assert provider.events.count("enter") == lease.generation
