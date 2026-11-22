"""CTC-2 §9 'Exclusion and fairness' with fake children at the stage boundary."""
from __future__ import annotations

import threading

import pytest

from . import stage_admission as sa
from . import stage_footprint as sf

ADAPTER = sf.ExecutionAdapter(sf.ADAPTER_CPU_SERVING, None, None, None, None)


def fp(target, regions, devices=(), kind="measure"):
    return sf.StageFootprint(kind, target, "a", ADAPTER, None, None, None, None,
                             frozenset(), frozenset(), tuple(regions), None, tuple(devices),
                             None)


def child(target, pid):
    return sa.ChildIdentity(pid, pid * 10, target)


@pytest.fixture
def adm():
    a = sa.Admission()
    for t, p in (("A", 1), ("B", 2), ("C", 3)):
        a.register(child(t, p))
    return a


def test_conflicting_stages_exclude_and_compatible_ones_overlap(adm):
    ra = adm.request(child("A", 1), "a", fp("A", ["q0"]))
    rb = adm.request(child("B", 2), "a", fp("B", ["q0", "q1"]))
    rc = adm.request(child("C", 3), "a", fp("C", ["q3"], ["mi210_0"]))
    assert ra.granted and not rb.granted and rc.granted
    adm.release(ra)
    assert rb.granted


def test_fifo_oldest_waiter_is_not_starved_by_newer_conflicting_work(adm):
    run = adm.request(child("A", 1), "a", fp("A", ["q0"]))
    oldest = adm.request(child("B", 2), "a", fp("B", ["q0", "q1"]))
    # C is compatible with the running stage but conflicts with the older waiter.
    jumper = adm.request(child("C", 3), "a", fp("C", ["q1"]))
    assert not oldest.granted and not jumper.granted
    adm.release(run)
    assert oldest.granted and not jumper.granted
    adm.release(oldest)
    assert jumper.granted


def test_partial_claim_failure_releases_what_was_taken():
    released = []

    class Held:
        def __init__(self, name):
            self.name = name

        def release(self):
            released.append(self.name)

    def device(_name):
        raise TimeoutError("device busy")

    with pytest.raises(TimeoutError):
        sa.acquire(fp("A", ["q0", "q1"], ["mi210_0"]), cpu_region=Held, device=device)
    assert released == ["q1", "q0"]


def test_stage_context_yields_held_claims_and_frees_admission(adm):
    released = []

    class Held:
        def __init__(self, name):
            self.name = name

        def release(self):
            released.append(self.name)

    with sa.stage(adm, child("A", 1), "a", fp("A", ["q0"], ["mi210_0"]),
                  cpu_region=Held, device=Held) as held:
        assert [h.name for h in held] == ["q0", "mi210_0"]
        assert adm.status()[0]["state"] == "running"
    assert released == ["mi210_0", "q0"] and adm.status() == []


def test_waiting_child_is_granted_from_another_thread(adm):
    hold = adm.request(child("A", 1), "a", fp("A", ["q2"]))
    seen = []

    def fake_child():
        with sa.stage(adm, child("B", 2), "a", fp("B", ["q2"]),
                      cpu_region=lambda r: type("H", (), {"release": lambda s: None})(),
                      device=lambda d: None, timeout_s=5):
            seen.append("ran")

    t = threading.Thread(target=fake_child)
    t.start()
    assert adm.status()[-1]["state"] == "waiting_resource" or seen
    adm.release(hold)
    t.join(5)
    assert seen == ["ran"]


def test_requests_are_bound_to_the_captured_child_identity(adm):
    with pytest.raises(sa.AdmissionRefused, match="unknown child"):
        adm.request(sa.ChildIdentity(1, 999, "A"), "a", fp("A", ["q0"]))
    with pytest.raises(sa.AdmissionRefused, match="captured target"):
        adm.request(child("A", 1), "a", fp("B", ["q0"]))
    adm.request(child("A", 1), "a", fp("A", ["q0"]))
    with pytest.raises(sa.AdmissionRefused, match="outstanding"):
        adm.request(child("A", 1), "a", fp("A", ["q1"]))


def test_cancel_removes_a_crashed_childs_entries_and_reschedules(adm):
    adm.request(child("A", 1), "a", fp("A", ["q0"]))
    waiter = adm.request(child("B", 2), "a", fp("B", ["q0"]))
    assert adm.cancel_child(child("A", 1)) == 1
    assert waiter.granted


def test_control_messages_are_closed():
    assert sa.decode(sa.encode("grant", seq=1))["seq"] == 1
    with pytest.raises(ValueError):
        sa.encode("launch")
    with pytest.raises(sa.AdmissionRefused):
        sa.decode(b'{"kind": "exec"}')
