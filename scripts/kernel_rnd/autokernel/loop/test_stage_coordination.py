"""CTC-2 part 2: parent server + child client + run.py stage scope, fake children."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from . import stage_admission as sa
from . import stage_coordination as sc
from . import stage_footprint as sf

ADAPTER = sf.ExecutionAdapter(sf.ADAPTER_CPU_SERVING, None, None, None, None)


def fp(target, regions, devices=(), cpus=frozenset({0})):
    return sf.StageFootprint("measure", target, "a", ADAPTER, None, "/b", None, None,
                             frozenset(cpus), frozenset(cpus), tuple(regions), None,
                             tuple(devices), None)


def connect(server, target, pid):
    parent, child = server.spawn_pair()
    server.attach(parent, sa.ChildIdentity(pid, pid, target))
    return sc.ChildClient(child)


def scope(client, target, regions, devices=(), log=None):
    log = [] if log is None else log

    @contextmanager
    def hold_cpu(cpus):
        log.append(("cpu+", target, cpus))
        try:
            yield ("cpu", cpus)
        finally:
            log.append(("cpu-", target))

    @contextmanager
    def hold_device():
        log.append(("dev+", target))
        try:
            yield ("dev",)
        finally:
            log.append(("dev-", target))
    return sc.StageScope(client, runtime=fp(target, regions, devices), build=None,
                         hold_cpu=hold_cpu, hold_device=hold_device), log


def test_conflicting_children_exclude_through_the_parent():
    server = sc.ParentServer()
    a, log = scope(connect(server, "A", 11), "A", ["q0"])
    b, _ = scope(connect(server, "B", 12), "B", ["q0"], log=log)
    inside, order = threading.Event(), []

    def run_b():
        with b.stage("measure"):
            order.append("B")

    with a.stage("measure") as held:
        assert held == [("cpu", "0")]
        t = threading.Thread(target=run_b)
        t.start()
        time.sleep(0.2)
        order.append("A-end")
        assert "B" not in order
    t.join(5)
    assert order == ["A-end", "B"]
    assert log.index(("cpu-", "A")) < log.index(("cpu+", "B", "0"))


def test_compatible_children_overlap():
    server = sc.ParentServer()
    a, _ = scope(connect(server, "A", 21), "A", ["q0"])
    b, _ = scope(connect(server, "B", 22), "B", ["q3"], ["mi210_0"])
    with a.stage("measure"):
        done = threading.Event()
        threading.Thread(target=lambda: (b.stage("measure").__enter__(), done.set())).start()
        assert done.wait(5)


def test_order_is_admission_then_cpu_then_device_and_nested_calls_reuse_the_lease():
    server = sc.ParentServer()
    s, log = scope(connect(server, "G", 31), "G", ["q3"], ["mi210_0"])
    with s.stage("measure"):
        with s.stage("profile") as inner:
            assert len(inner) == 2
    assert log == [("cpu+", "G", "0"), ("dev+", "G"), ("dev-", "G"), ("cpu-", "G")]
    assert [w["kind"] for w in s.windows] == ["measure"]


def test_device_failure_releases_the_cpu_claim_and_the_admission():
    server = sc.ParentServer()
    client = connect(server, "G", 41)
    log = []

    @contextmanager
    def hold_cpu(cpus):
        log.append("cpu+")
        try:
            yield "cpu"
        finally:
            log.append("cpu-")

    def hold_device():
        raise TimeoutError("device busy")
    s = sc.StageScope(client, runtime=fp("G", ["q3"], ["mi210_0"]), build=None,
                      hold_cpu=hold_cpu, hold_device=hold_device)
    with pytest.raises(TimeoutError):
        with s.stage("measure"):
            pass
    assert log == ["cpu+", "cpu-"]
    deadline = time.time() + 5
    while server.admission.status() and time.time() < deadline:
        time.sleep(0.01)
    assert server.admission.status() == []


def test_wrapped_callbacks_run_inside_a_stage():
    server = sc.ParentServer()
    s, log = scope(connect(server, "A", 51), "A", ["q1"])
    gate_for = s.wrap_factory("build", lambda worker: (lambda h, p: s.held()))
    assert gate_for(0)("h", []) == [("cpu", "0")]
    assert s.wrap("profile", lambda: log[-1])() == ("cpu+", "A", "0")


def test_child_exit_cancels_its_entries():
    server = sc.ParentServer()
    parent, child = server.spawn_pair()
    server.attach(parent, sa.ChildIdentity(61, 61, "A"))
    client = sc.ChildClient(child)
    with client.admitted(fp("A", ["q0"])):
        assert server.admission.status()[0]["state"] == "running"
        child.close()
    deadline = time.time() + 5
    while server.admission.status() and time.time() < deadline:
        time.sleep(0.01)
    assert server.admission.status() == []


def test_spoofed_target_is_refused_by_the_parent():
    server = sc.ParentServer()
    client = connect(server, "A", 71)
    with pytest.raises(sc.CoordinationRefused, match="captured target"):
        with client.admitted(fp("B", ["q0"])):
            pass


def test_default_mode_has_no_client_and_popen_is_unchanged():
    assert sc.client_from_env({}) is None
    assert sc.popen_kwargs(None, None, (5,)) == (None, (5,), None, None)


def test_run_long_claim_owners_and_window_modes_are_refused():
    base = dict(validate_source_loo=False, cpu_held_intervals_v2="off", gpu_window="off")
    for override, runtime, match in (({}, True, "runtime admission"),
                                     ({"cpu_held_intervals_v2": "on"}, False, "CTC-4"),
                                     ({"gpu_window": "required"}, False, "window"),
                                     ({}, False, "enrolled target")):
        with pytest.raises(sc.CoordinationRefused, match=match):
            sc.install(None, args=SimpleNamespace(**{**base, **override}), target=None,
                       launch=None, resources=None, hold_cpu=None, hold_device=None,
                       runtime_enabled=runtime)


def test_source_loo_is_no_longer_refused_and_gets_the_stage_held_claims():
    args = SimpleNamespace(validate_source_loo=True, cpu_held_intervals_v2="off", gpu_window="off")
    with pytest.raises(sc.CoordinationRefused, match="enrolled target"):   # passes the LOO check, stops later
        sc.install(None, args=args, target=None, launch=None, resources=None, hold_cpu=None,
                   hold_device=None, runtime_enabled=False)
    server = sc.ParentServer()
    s, log = scope(connect(server, "G", 21), "G", ["q3"], ["mi210_0"])
    with pytest.raises(sc.CoordinationRefused, match="no stage"):
        s.claims()
    with s.stage("build"):
        cpu, gpu = s.claims()
        assert cpu == ("cpu", "0") and gpu == ("dev",)
        with s.stage("build"):                       # nested reuses the same lease
            assert s.claims() == (cpu, gpu)
    assert ("cpu-", "G") in log and ("dev-", "G") in log
    with pytest.raises(sc.CoordinationRefused):      # released claims are never handed out afterwards
        s.claims()
    c, clog = scope(connect(server, "C", 22), "C", ["q0"])
    with c.stage("build"):
        assert c.claims()[1] is None                 # CPU-only stage: no device claim


@pytest.mark.parametrize('surface', ['cpu', 'gpu'])
def test_budgeted_stage_counts_nested_lease_once(tmp_path, surface):
    from ..controller import run_budget

    now = [0.0]
    budget = run_budget.RunBudget(run_budget.Ceilings(cpu_s=10, gpu_s=10), tmp_path,
                                  clock=lambda: now[0], monotonic=lambda: now[0])
    @contextmanager
    def owned(*args):
        yield object()
    class Client:
        @contextmanager
        def admitted(self, footprint):
            yield
    cpu = lambda cpus: (run_budget.budgeted_claim(owned(), budget, 'cpu')
                        if surface == 'cpu' else owned())
    device = (lambda: run_budget.budgeted_claim(owned(), budget, 'gpu')) if surface == 'gpu' else None
    target = fp('A', ['q0'], ['gpu'] if surface == 'gpu' else ())
    stage = sc.StageScope(Client(), runtime=target, build=None, hold_cpu=cpu, hold_device=device)
    with stage.stage('measure'):
        now[0] += 2
        with stage.stage('profile'):
            now[0] += 3
    assert budget.snapshot()['arm_seconds_claimed'][surface] == 5
