"""MI210 exclusion applies to CPU measurement windows, not authoring/builds."""
import fcntl
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from . import claim, loop, run


def _launch(cpu_list, backend="cpu"):
    return SimpleNamespace(backend=backend, template=SimpleNamespace(cpu_list=cpu_list))


def test_full_q3_cpu_measurement_holds_gpu_item_lock(tmp_path, monkeypatch):
    lock = tmp_path / "mi210.lock"
    monkeypatch.setattr(claim, "DEVICE_LOCK", lock)
    with run._q3_cpu_gpu_quiet_window(
            _launch("0-95"), on_wait=lambda: pytest.fail("unexpected wait"),
            should_stop=lambda: False):
        with lock.open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_reduced_cpu_window_does_not_claim_gpu(tmp_path, monkeypatch):
    lock = tmp_path / "mi210.lock"
    monkeypatch.setattr(claim, "DEVICE_LOCK", lock)
    with lock.open("a") as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with run._q3_cpu_gpu_quiet_window(
                _launch("0-47"), on_wait=lambda: pytest.fail("unexpected wait"),
                should_stop=lambda: False):
            pass


def test_busy_gpu_item_defers_q3_measurement_before_body(tmp_path, monkeypatch):
    lock = tmp_path / "mi210.lock"
    monkeypatch.setattr(claim, "DEVICE_LOCK", lock)
    notices = []
    measured = False
    with lock.open("a") as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(loop.TailRefused, match="before q3 CPU measurement"):
            with run._q3_cpu_gpu_quiet_window(
                    _launch("0-95"), on_wait=lambda: notices.append("waiting"),
                    should_stop=lambda: bool(notices)):
                measured = True
    assert notices == ["waiting"]
    assert not measured


def test_invalid_arm_reschedule_reacquires_measurement_window(monkeypatch):
    events = []

    @contextmanager
    def window():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    invalid = loop.MeasurementInvalid("invalid arm", {"arm": "candidate"})
    invalid.reschedule = lambda: {"continued": True}
    monkeypatch.setattr(run, "ServingComparison", lambda row, scope: (row, scope))

    def initial():
        raise invalid

    with pytest.raises(loop.MeasurementInvalid) as caught:
        run._serving_comparison(initial, "fixture", measurement_window=window)
    assert events == ["enter", "exit"]
    assert caught.value.reschedule() == ({"continued": True}, "fixture")
    assert events == ["enter", "exit", "enter", "exit"]
