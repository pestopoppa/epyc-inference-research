"""The CPU/GPU measurement quiet window. DEFAULT ON since 2026-10-04 (operator ruling:
GPU benches and CPU measurements never overlap): a q3 CPU measurement holds the MI210
flock, a GPU measurement holds the q3 CPU region claim -- each for the measurement body
only, never authoring/builds. `--cpu-measurement-gpu-quiet off` is the explicit opt-out."""
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


def test_off_cpu_measurement_takes_no_gpu_claim(tmp_path, monkeypatch):
    """Explicit opt-out (2026-10-03 regression kept): under "off" a CPU run -- even one
    whose CPU list covers q3 -- never opens, waits on, or holds the MI210 flock."""
    lock = tmp_path / "mi210.lock"
    monkeypatch.setattr(claim, "DEVICE_LOCK", lock)
    with lock.open("a") as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a GPU item holds it
        measured = []
        with run._q3_cpu_gpu_quiet_window(
                _launch("0-95"), on_wait=lambda: pytest.fail("CPU run waited on the GPU"),
                should_stop=lambda: pytest.fail("CPU run polled the GPU lock"),
                policy=run.CPU_MEASUREMENT_GPU_QUIET_OFF):
            measured.append(True)
        assert measured == [True]
    lock.unlink()
    with run._q3_cpu_gpu_quiet_window(
            _launch("0-95"), on_wait=lambda: None, should_stop=lambda: False,
            policy=run.CPU_MEASUREMENT_GPU_QUIET_OFF):
        assert not lock.exists(), "an opted-out CPU window must not even create the GPU lock"


def test_default_policy_is_q3_and_holds_the_flock(tmp_path, monkeypatch):
    """2026-10-04 flip: the default IS the quiet window."""
    assert run.CPU_MEASUREMENT_GPU_QUIET_DEFAULT == run.CPU_MEASUREMENT_GPU_QUIET_Q3
    lock = tmp_path / "mi210.lock"
    monkeypatch.setattr(claim, "DEVICE_LOCK", lock)
    with run._q3_cpu_gpu_quiet_window(
            _launch("0-95"), on_wait=lambda: pytest.fail("unexpected wait"),
            should_stop=lambda: False):
        with lock.open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_q3_cpu_measurement_does_not_wait_on_a_q3_honouring_gpu_holder(tmp_path, monkeypatch):
    """No deadlock: an AutoKernel GPU run holds mi210_0 for its life and takes q3 per
    measurement; a CPU run already holding q3 must not wait on its flock."""
    lock = tmp_path / "mi210.lock"
    monkeypatch.setattr(claim, "DEVICE_LOCK", lock)
    with lock.open("a") as gpu_run:
        fcntl.flock(gpu_run, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with claim.q3_honouring_marker():
            measured = []
            with run._q3_cpu_gpu_quiet_window(
                    _launch("0-95"), on_wait=lambda: pytest.fail("waited on a q3-honouring holder"),
                    should_stop=lambda: False):
                measured.append(True)
            assert measured == [True]
        assert not claim.q3_honouring_marker_path().exists()


def test_a_stale_honouring_marker_does_not_exempt_another_holder(tmp_path, monkeypatch):
    lock = tmp_path / "mi210.lock"
    monkeypatch.setattr(claim, "DEVICE_LOCK", lock)
    claim.q3_honouring_marker_path().write_text(
        '{"pid": 999999999, "start_ticks": 1}', encoding="utf-8")
    notices = []
    with lock.open("a") as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(loop.TailRefused):
            with run._q3_cpu_gpu_quiet_window(
                    _launch("0-95"), on_wait=lambda: notices.append("waiting"),
                    should_stop=lambda: bool(notices)):
                pytest.fail("measured under a foreign GPU holder")


def _busy():
    class CpuRegionLockTimeout(RuntimeError):
        pass
    return CpuRegionLockTimeout("q3 busy")


def test_gpu_measurement_holds_q3_for_its_body_only():
    events = []

    @contextmanager
    def hold():
        events.append("acquire")
        yield {"regions": ["q3"]}
        events.append("release")

    with run._gpu_q3_measurement_window(True, on_wait=lambda: pytest.fail("wait"),
                                        should_stop=lambda: False, hold=hold):
        events.append("measure")
    assert events == ["acquire", "measure", "release"]


def test_gpu_measurement_waits_on_busy_q3_then_refuses_on_stop():
    notices = []

    def hold():
        raise _busy()

    with pytest.raises(loop.TailRefused, match="q3 CPU claim"):
        with run._gpu_q3_measurement_window(True, on_wait=lambda: notices.append("w"),
                                            should_stop=lambda: bool(notices), hold=hold):
            pytest.fail("measured without q3")
    assert notices == ["w"]


def test_gpu_measurement_retries_busy_q3_until_granted(monkeypatch):
    monkeypatch.setattr(run.time, "sleep", lambda _s: None)
    attempts = []

    @contextmanager
    def granted():
        yield {}

    def hold():
        attempts.append(1)
        if len(attempts) < 3:
            raise _busy()
        return granted()

    with run._gpu_q3_measurement_window(True, on_wait=lambda: None,
                                        should_stop=lambda: False, hold=hold):
        pass
    assert len(attempts) == 3


def test_gpu_measurement_window_noops():
    def hold():
        pytest.fail("q3 claimed")
    for kwargs in ({"policy": run.CPU_MEASUREMENT_GPU_QUIET_OFF}, {"already_held": True}):
        with run._gpu_q3_measurement_window(True, on_wait=lambda: None,
                                            should_stop=lambda: False, hold=hold, **kwargs):
            pass
    with run._gpu_q3_measurement_window(False, on_wait=lambda: None,
                                        should_stop=lambda: False, hold=hold):
        pass


def test_gpu_measurement_propagates_a_non_busy_refusal():
    def hold():
        raise claim.ClaimRefused("preflight failed")
    with pytest.raises(claim.ClaimRefused, match="preflight"):
        with run._gpu_q3_measurement_window(True, on_wait=lambda: None,
                                            should_stop=lambda: False, hold=hold):
            pass


def test_gpu_q3_claim_is_named_by_physical_cores():
    """REGION-SIBLING-1: 184-191 maps to no orchestrator region today; 88-95 is q3."""
    assert claim.GPU_Q3_MEASUREMENT_CPU_LIST == "88-95"


def test_nested_measurement_window_enters_the_quiet_window_once():
    depth = run.threading.local()
    entered = []

    @contextmanager
    def window():
        entered.append("in")
        yield
        entered.append("out")

    with run._reentrant(depth, window()):
        with run._reentrant(depth, window()):
            pass
    assert entered == ["in", "out"]


def test_unknown_gpu_quiet_policy_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(claim, "DEVICE_LOCK", tmp_path / "mi210.lock")
    with pytest.raises(ValueError, match="quiet policy"):
        with run._q3_cpu_gpu_quiet_window(
                _launch("0-95"), on_wait=lambda: None, should_stop=lambda: False,
                policy="on"):
            pass


def test_cli_declares_gpu_quiet_policy_default_q3(capsys):
    with pytest.raises(SystemExit) as exited:
        run.main(["--help"])
    assert exited.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "--cpu-measurement-gpu-quiet {off,q3}" in text
    assert "(default: q3)" in text.split("--cpu-measurement-gpu-quiet {off,q3}")[-1][:900]


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


def test_q3_preflight_refuses_without_the_cross_role_mutex_and_run_probes_it_at_startup(monkeypatch):
    """The GPU half of the quiet window is proven at STARTUP (claim.q3_measurement_preflight
    behind parser.error), not discovered at the first measurement after a build."""
    from pathlib import Path
    import re
    from . import claim
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "0")
    with pytest.raises(claim.ClaimRefused, match="cross-role region mutex"):
        claim.q3_measurement_preflight()
    body = (Path(__file__).parent / "run.py").read_text(encoding="utf-8")
    assert re.search(r"if \(direct_launch and not cpu_launch\n\s+and args\.cpu_measurement_gpu_quiet "
                     r"!= CPU_MEASUREMENT_GPU_QUIET_OFF\):\n(?:.*\n){1,6}?\s+claim\.q3_measurement_preflight\(\)"
                     r"\n\s+except \(claim\.ClaimRefused, ImportError, OSError\) as exc:\n\s+parser\.error",
                     body)
