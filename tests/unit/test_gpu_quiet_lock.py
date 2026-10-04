"""gpu-quiet: the host-wide reader/writer flock between GPU benches and CPU measurements.

Every test redirects the lock directory into its own tmp dir (ORCHESTRATOR_TMP_DIR);
none touches the real /mnt/raid0/llm/tmp, /run or /tmp lock files.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.runtime import gpu_quiet_lock as gq
from src.runtime.cpu_region_lock import CpuRegionLockTimeout, region_lock_path

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _lock_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TMP_DIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_INFERENCE_LOCK_POLL_MS", "10")
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    yield tmp_path


def _try(mode: int) -> bool:
    """Probe the gpu-quiet file from a fresh open file description."""
    with open(gq.gpu_quiet_lock_path(), "a+b") as fh:
        try:
            fcntl.flock(fh.fileno(), mode | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return True


def test_lock_file_sits_beside_the_region_locks_outside_their_namespace(tmp_path):
    assert gq.gpu_quiet_lock_path() == tmp_path / "gpu_quiet.lock"
    assert gq.gpu_quiet_lock_path().parent == region_lock_path("bench", "q0").parent
    # Region sweeps/status glob `cpu_region.*.*.lock`; gpu-quiet must never match.
    assert not list(tmp_path.glob("cpu_region.*.*.lock"))
    with gq.gpu_quiet_lock("exclusive", role="t"):
        assert not list(tmp_path.glob("cpu_region.*"))


def test_exclusive_excludes_everyone_and_releases():
    with gq.gpu_quiet_lock("exclusive", role="bench-gpu"):
        assert not _try(fcntl.LOCK_SH)
        assert not _try(fcntl.LOCK_EX)
    assert _try(fcntl.LOCK_EX)


def test_shared_holders_coexist_and_block_exclusive():
    with gq.gpu_quiet_lock("shared", role="cpu-a"):
        with gq.gpu_quiet_lock("shared", role="cpu-b", timeout_s=0.5):
            assert _try(fcntl.LOCK_SH)
            assert not _try(fcntl.LOCK_EX)
            with pytest.raises(CpuRegionLockTimeout):
                with gq.gpu_quiet_lock("exclusive", role="bench-gpu", timeout_s=0.1):
                    pytest.fail("exclusive granted over shared holders")


def test_shared_waits_for_an_exclusive_holder_then_times_out():
    with gq.gpu_quiet_lock("exclusive", role="bench-gpu"):
        started = time.monotonic()
        with pytest.raises(CpuRegionLockTimeout):
            with gq.gpu_quiet_lock("shared", role="cpu", timeout_s=0.2):
                pytest.fail("shared granted over an exclusive holder")
        assert time.monotonic() - started >= 0.15


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="gpu-quiet mode"):
        with gq.gpu_quiet_lock("q3", role="t"):
            pass


def test_holder_records_name_role_tag_pid_and_vanish_on_release():
    with gq.gpu_quiet_lock("shared", role="cpu-a", request_tag="tag-a"):
        with gq.gpu_quiet_lock("shared", role="cpu-b", request_tag="tag-b"):
            state = gq.holders()
            assert state["held"] is True and state["mode"] == "shared"
            got = sorted((h["role"], h["request_tag"], h["pid"], h["mode"])
                         for h in state["holders"])
            assert got == [("cpu-a", "tag-a", os.getpid(), "shared"),
                           ("cpu-b", "tag-b", os.getpid(), "shared")]
    state = gq.holders()
    assert state == {"path": str(gq.gpu_quiet_lock_path()), "held": False, "mode": None,
                     "holders": []}
    assert not list(gq.gpu_quiet_holders_dir().glob("*.json"))


def test_exclusive_mode_is_reported():
    with gq.gpu_quiet_lock("exclusive", role="bench-gpu", request_tag="g"):
        state = gq.holders()
        assert state["mode"] == "exclusive"
        assert [h["role"] for h in state["holders"]] == ["bench-gpu"]


def test_a_stale_record_is_not_a_holder_and_is_swept_on_next_acquire():
    directory = gq.gpu_quiet_holders_dir()
    directory.mkdir(parents=True)
    stale = directory / "999999999-dead.json"
    stale.write_text(json.dumps({"pid": 999999999, "role": "ghost", "mode": "exclusive"}))
    gq.gpu_quiet_lock_path().touch()
    assert gq.holders()["holders"] == []
    assert gq.holders()["held"] is False
    with gq.gpu_quiet_lock("shared", role="cpu"):
        assert not stale.exists()


def test_a_live_record_without_the_flock_is_not_reported():
    """Realized-first: the flock is the fact, a record alone proves nothing."""
    directory = gq.gpu_quiet_holders_dir()
    directory.mkdir(parents=True)
    (directory / f"{os.getpid()}-x.json").write_text(
        json.dumps({"pid": os.getpid(), "role": "liar", "mode": "exclusive"}))
    gq.gpu_quiet_lock_path().touch()
    assert gq.holders()["holders"] == []


def test_combined_acquire_takes_gpu_quiet_first_then_regions(monkeypatch):
    order = []
    real_quiet = gq.gpu_quiet_lock

    @contextmanager
    def quiet(*a, **k):
        with real_quiet(*a, **k) as record:
            order.append("gpu-quiet")
            yield record
        order.append("gpu-quiet released")

    @contextmanager
    def regions(role, regs, **k):
        assert not _try(fcntl.LOCK_EX), "regions taken without gpu-quiet held"
        order.append("regions")
        yield {r: Path(r) for r in regs}
        order.append("regions released")

    monkeypatch.setattr(gq, "gpu_quiet_lock", quiet)
    monkeypatch.setattr(gq, "cpu_region_lock", regions)
    with gq.gpu_quiet_then_regions("cpu", {"q0"}, gpu_quiet="shared") as grant:
        assert set(grant["regions"]) == {"q0"} and grant["gpu_quiet"]["mode"] == "shared"
    assert order == ["gpu-quiet", "regions", "regions released", "gpu-quiet released"]


def test_combined_acquire_backs_off_gpu_quiet_while_regions_are_busy(monkeypatch):
    """Never sit on gpu-quiet behind a region holder: between attempts it is free."""
    monkeypatch.setattr(gq, "_BACKOFF_S", 0.0)
    attempts = []
    free_between = []

    @contextmanager
    def regions(role, regs, **k):
        attempts.append(k["timeout_s"])
        if len(attempts) < 3:
            raise CpuRegionLockTimeout("busy")
        yield {r: Path(r) for r in regs}

    real_sleep = time.sleep

    def sleep(s):
        free_between.append(_try(fcntl.LOCK_EX))
        real_sleep(0)

    monkeypatch.setattr(gq, "cpu_region_lock", regions)
    monkeypatch.setattr(gq.time, "sleep", sleep)
    with gq.gpu_quiet_then_regions("cpu", {"q1"}, gpu_quiet="exclusive",
                                   region_attempt_s=0.25):
        assert not _try(fcntl.LOCK_SH)
    assert attempts == [0.25, 0.25, 0.25]
    assert free_between == [True, True]


def test_combined_acquire_times_out_with_the_region_timeout(monkeypatch):
    monkeypatch.setattr(gq, "_BACKOFF_S", 0.01)

    @contextmanager
    def regions(role, regs, **k):
        time.sleep(min(0.05, k["timeout_s"]))
        raise CpuRegionLockTimeout("busy")
        yield  # pragma: no cover

    monkeypatch.setattr(gq, "cpu_region_lock", regions)
    with pytest.raises(CpuRegionLockTimeout):
        with gq.gpu_quiet_then_regions("cpu", {"q1"}, gpu_quiet="shared", timeout_s=0.3):
            pytest.fail("granted")
    assert _try(fcntl.LOCK_EX), "gpu-quiet leaked after a timeout"


def test_combined_acquire_without_a_mode_is_the_plain_region_claim(monkeypatch):
    calls = []

    @contextmanager
    def regions(role, regs, **k):
        calls.append((role, set(regs), k["timeout_s"]))
        yield {}

    monkeypatch.setattr(gq, "cpu_region_lock", regions)
    monkeypatch.setattr(gq, "gpu_quiet_lock", lambda *a, **k: pytest.fail("gpu-quiet taken"))
    with gq.gpu_quiet_then_regions("bench", {"q2"}, gpu_quiet=None, timeout_s=7.0) as g:
        assert g == {"regions": {}, "gpu_quiet": None}
    assert calls == [("bench", {"q2"}, 7.0)]


def test_gpu_quiet_alone_takes_no_region(monkeypatch):
    monkeypatch.setattr(gq, "cpu_region_lock", lambda *a, **k: pytest.fail("region taken"))
    with gq.gpu_quiet_then_regions("bench-gpu", (), gpu_quiet="exclusive") as grant:
        assert grant["regions"] == {}
        assert not _try(fcntl.LOCK_SH)


def test_real_region_claim_under_shared_gpu_quiet_releases_both():
    with gq.gpu_quiet_then_regions("cpu-measure", {"q0"}, gpu_quiet="shared",
                                   timeout_s=2.0) as grant:
        assert set(grant["regions"]) == {"q0"}
        assert not _try(fcntl.LOCK_EX)
    assert _try(fcntl.LOCK_EX)


def test_serving_region_claims_never_touch_gpu_quiet():
    """The orchestrator's per-call claims ignore gpu-quiet: a GPU bench holding it
    exclusive must not delay a serving region claim at all."""
    from src.runtime.cpu_region_lock import cpu_region_lock

    with gq.gpu_quiet_lock("exclusive", role="bench-gpu"):
        with cpu_region_lock("frontdoor", {"q2", "q3"}, instance_idx=2, timeout_s=0.5) as held:
            assert set(held) == {"q2", "q3"}
    assert not gq.gpu_quiet_lock_path().exists() or _try(fcntl.LOCK_EX)


# ------------------------------------------------------------------ the CLI, end to end

def _cli(args, tmp_path, timeout=30):
    env = os.environ.copy()
    env["ORCHESTRATOR_TMP_DIR"] = str(tmp_path)
    env["ORCHESTRATOR_INFERENCE_LOCK_POLL_MS"] = "10"
    return subprocess.run(
        [sys.executable, "-m", "src.runtime.region_lock_cli", *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout,
        check=False)


_CHILD_PROBE = (
    "import fcntl,sys\n"
    "fh=open(sys.argv[1],'a+b')\n"
    "try:\n"
    "    fcntl.flock(fh.fileno(), fcntl.LOCK_SH|fcntl.LOCK_NB)\n"
    "except OSError:\n"
    "    sys.exit(7)\n"
    "sys.exit(0)\n"
)


def test_cli_gpu_quiet_exclusive_alone_holds_it_for_the_child(tmp_path):
    lock = tmp_path / "gpu_quiet.lock"
    result = _cli(["run", "--gpu-quiet", "exclusive", "--role", "bench-gpu", "--tag", "g1",
                   "--", sys.executable, "-c", _CHILD_PROBE, str(lock)], tmp_path)
    assert result.returncode == 7, result.stderr  # the child could not take it shared
    assert "held gpu-quiet exclusive" in result.stderr
    assert not list(tmp_path.glob("cpu_region.*")), "a gpu-quiet-only run took a region"
    with open(lock, "a+b") as fh:  # released when the child exited
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_cli_shared_with_regions_takes_both(tmp_path):
    result = _cli(["run", "--regions", "q0", "--gpu-quiet", "shared", "--no-preflight",
                   "--", sys.executable, "-c", _CHILD_PROBE, str(tmp_path / "gpu_quiet.lock")],
                  tmp_path)
    assert result.returncode == 0, result.stderr  # shared: the child can share it too
    assert "regions ['q0']" in result.stderr


def test_cli_times_out_with_exit_75(tmp_path):
    with gq.gpu_quiet_lock("exclusive", role="bench-gpu"):
        result = _cli(["run", "--gpu-quiet", "shared", "--timeout-s", "0.3",
                       "--", sys.executable, "-c", "pass"], tmp_path)
    assert result.returncode == 75, result.stderr
    assert "TIMEOUT acquiring gpu-quiet shared" in result.stderr


def test_cli_run_needs_some_target(tmp_path):
    result = _cli(["run", "--", sys.executable, "-c", "pass"], tmp_path)
    assert result.returncode == 64
    assert "--gpu-quiet" in result.stderr


def test_cli_rejects_an_unknown_gpu_quiet_mode(tmp_path):
    result = _cli(["run", "--gpu-quiet", "q3", "--", "true"], tmp_path)
    assert result.returncode == 2


def test_cli_status_shows_gpu_quiet_holders(tmp_path):
    with gq.gpu_quiet_lock("exclusive", role="bench-gpu", request_tag="gpu-tag"):
        table = _cli(["status"], tmp_path)
        assert table.returncode == 0, table.stderr
        line = [row for row in table.stdout.splitlines() if row.startswith("gpu-quiet")]
        assert line and "EXCL" in line[0] and "bench-gpu" in line[0] \
            and f"pid {os.getpid()}" in line[0] and "gpu-tag" in line[0]
        both = _cli(["status", "--json", "--gpu-quiet"], tmp_path)
        body = json.loads(both.stdout)
        assert [row["region"] for row in body["regions"]] == ["q0", "q1", "q2", "q3"]
        assert body["gpu_quiet"]["mode"] == "exclusive"
        assert body["gpu_quiet"]["holders"][0]["request_tag"] == "gpu-tag"
        plain = json.loads(_cli(["status", "--json"], tmp_path).stdout)
        assert isinstance(plain, list) and [r["region"] for r in plain] == ["q0", "q1", "q2", "q3"]
    idle = _cli(["status"], tmp_path)
    assert [row for row in idle.stdout.splitlines() if row.startswith("gpu-quiet")] == [
        "gpu-quiet  free   "]
