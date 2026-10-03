"""MI210 exclusion is OPT-IN (`--cpu-measurement-gpu-quiet q3`) and, when on, applies to
CPU measurement windows only, not authoring/builds. Default: a CPU run takes no GPU claim."""
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
            should_stop=lambda: False, policy=run.CPU_MEASUREMENT_GPU_QUIET_Q3):
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
                should_stop=lambda: False, policy=run.CPU_MEASUREMENT_GPU_QUIET_Q3):
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
                    should_stop=lambda: bool(notices), policy=run.CPU_MEASUREMENT_GPU_QUIET_Q3):
                measured = True
    assert notices == ["waiting"]
    assert not measured


def test_default_cpu_measurement_takes_no_gpu_claim(tmp_path, monkeypatch):
    """Regression (2026-10-03): the DS41 CPU-only loop held mi210_0 through every
    measurement window and blocked a GPU window. By default a CPU run -- even one
    whose CPU list covers q3 -- never opens, waits on, or holds the MI210 flock."""
    lock = tmp_path / "mi210.lock"
    monkeypatch.setattr(claim, "DEVICE_LOCK", lock)
    with lock.open("a") as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a GPU item holds it
        for policy in ({}, {"policy": run.CPU_MEASUREMENT_GPU_QUIET_OFF}):
            measured = []
            with run._q3_cpu_gpu_quiet_window(
                    _launch("0-95"), on_wait=lambda: pytest.fail("CPU run waited on the GPU"),
                    should_stop=lambda: pytest.fail("CPU run polled the GPU lock"), **policy):
                measured.append(True)
            assert measured == [True]
    lock.unlink()
    with run._q3_cpu_gpu_quiet_window(
            _launch("0-95"), on_wait=lambda: None, should_stop=lambda: False):
        assert not lock.exists(), "default CPU window must not even create the GPU lock"
    assert run.CPU_MEASUREMENT_GPU_QUIET_POLICIES[0] == run.CPU_MEASUREMENT_GPU_QUIET_OFF


def test_unknown_gpu_quiet_policy_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(claim, "DEVICE_LOCK", tmp_path / "mi210.lock")
    with pytest.raises(ValueError, match="quiet policy"):
        with run._q3_cpu_gpu_quiet_window(
                _launch("0-95"), on_wait=lambda: None, should_stop=lambda: False,
                policy="on"):
            pass


def test_cli_declares_gpu_quiet_policy_default_off(capsys):
    with pytest.raises(SystemExit) as exited:
        run.main(["--help"])
    assert exited.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--cpu-measurement-gpu-quiet {off,q3}" in text
    assert "(default: off)" in text.split("--cpu-measurement-gpu-quiet {off,q3}")[-1][:900]


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
