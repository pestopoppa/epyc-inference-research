"""DS41-W8d: RAM admission before DS41-class server starts. No server is ever launched."""
import contextlib
import inspect
import os
import re
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from . import ab_server_pair, capacity_probe, model_identity, ram_admit, serving

@pytest.fixture(autouse=True)
def private_lease(tmp_path, monkeypatch):
    directory = tmp_path / "lease"
    directory.mkdir(mode=0o700)
    path = directory / "lease.lock"
    path.touch(mode=0o600)
    monkeypatch.setattr(ram_admit, "DS41_LEASE_PATH", str(path))
    monkeypatch.setattr(ram_admit, "_trusted_ancestors", lambda parent: os.open(parent, os.O_RDONLY | os.O_DIRECTORY))


REAL_TRUSTED_ANCESTORS = ram_admit._trusted_ancestors
DS41 = ram_admit.DS41_SHARD
OTHER = "/mnt/raid0/llm/models/Qwen/Qwen3.6-35B-A3B-Q4_K_M.gguf"


def meminfo(tmp_path, avail_gib):
    p = tmp_path / "meminfo"
    p.write_text(f"MemTotal: 1\nMemAvailable: {avail_gib * 1048576 + 5} kB\n")
    return p


def fake_proc(tmp_path, procs, self_pid=100):
    """procs: {pid: (cmdline_args, maps_text, ppid)}; maps None = unreadable."""
    root = tmp_path / "proc"
    root.mkdir()
    for pid, (args, maps, ppid) in procs.items():
        d = root / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in args) + b"\0")
        (d / "stat").write_text(f"{pid} (x) S {ppid} 0 0\n")
        if maps is not None:
            (d / "maps").write_text(maps)
    return root


def test_margin_and_integer_floor(tmp_path):
    ok = ram_admit.check(739, meminfo=meminfo(tmp_path, 803))
    assert ok.admit and ok.avail_gib == 803 and ok.need_gib == 739
    bad = ram_admit.check(739, meminfo=meminfo(tmp_path, 802))
    assert not bad.admit and "margin" in bad.reason


def test_ds41_mode_holders_via_cmdline_maps_basename_and_unreadable(tmp_path):
    base = os.path.basename(DS41)
    root = fake_proc(tmp_path, {
        100: (["pytest", DS41], "", 1),                      # self: skipped even naming the model
        200: (["llama-server", "-m", DS41], "", 1),          # cmdline full path
        201: (["x"], f"7f00 r--p {base}\n", 1),              # maps basename
        202: (["unrelated"], "libc.so\n", 1),
        203: (["x"], None, 1),                               # maps unreadable -> not a hit
    })
    (root / "204").mkdir()                                   # no cmdline: unreadable
    r = ram_admit.check(1, ds41=True, meminfo=meminfo(tmp_path, 900), proc_root=root, self_pid=100)
    assert not r.admit
    assert sorted((h["pid"], h["via"]) for h in r.holders) == [(200, "cmdline"), (201, "maps")]
    assert r.unreadable == 1
    clear = ram_admit.check(1, ds41=False, meminfo=meminfo(tmp_path, 900), proc_root=root, self_pid=100)
    assert clear.admit and not clear.holders


def test_ancestors_of_self_are_skipped(tmp_path):
    root = fake_proc(tmp_path, {50: (["bash", "run.sh", DS41], "", 1), 100: (["py"], "", 50)})
    r = ram_admit.check(1, ds41=True, meminfo=meminfo(tmp_path, 900), proc_root=root, self_pid=100)
    assert r.admit


def test_is_ds41_argv():
    assert ram_admit.is_ds41_argv(["taskset", "-c", "1", "llama-server", "-m", DS41])
    assert ram_admit.is_ds41_argv(["llama-server", f"--model={DS41}"])
    assert not ram_admit.is_ds41_argv(["llama-server", "-m", OTHER])
    assert not ram_admit.is_ds41_argv(["llama-server", "-p", DS41])


def test_guard_launch_noop_for_non_ds41_and_refuses_ds41():
    with mock.patch.object(ram_admit, "check") as chk:
        assert ram_admit.guard_launch(["llama-server", "-m", OTHER]) is None
        chk.assert_not_called()
        chk.return_value = ram_admit.AdmitResult(False, "nope", 10, 739)
        with pytest.raises(ram_admit.RamAdmissionRefused, match="nope"):
            ram_admit.guard_launch(["llama-server", "-m", DS41])
        chk.assert_called_once_with(739, ds41=True)


# ---- choke points: admission strictly BEFORE Popen, on every server-start path ----

REFUSE = ram_admit.AdmitResult(False, "refused for test", 10, 739)
ADMIT = ram_admit.AdmitResult(True, "ok", 900, 739)


class Stop(Exception):
    pass


def order_log(result):
    log = []
    def check(*a, **k):
        log.append("admit")
        return result
    def popen(*a, **k):
        log.append("popen")
        raise Stop
    return log, check, popen


def measure_once(model):
    rec = serving.Recipe(name="t", model=model, device="none", ngl=0)
    return lambda: serving._measure_once(rec, Path("/nonexistent/build"), 59999, boot_timeout_s=2)


@contextlib.contextmanager
def serving_stubs(popen):
    with mock.patch.object(serving, "require_free_port"), \
         mock.patch.object(serving.kernel_coverage, "open_launch_sink", return_value=None), \
         mock.patch.object(serving.residency, "Sampler"), \
         mock.patch.object(serving.residency, "CpuLifecycleSampler"), \
         mock.patch.object(serving.procguard, "measurement_watch", return_value=None), \
         mock.patch.object(serving.subprocess, "Popen", side_effect=popen):
        yield


def test_measure_once_ds41_refused_before_popen(monkeypatch):
    log, check, popen = order_log(REFUSE)
    monkeypatch.setattr(ram_admit, "check", check)
    with serving_stubs(popen), pytest.raises(serving.ServerDied, match="RAM admission refused"):
        measure_once(DS41)()
    assert log == ["admit"]


def test_measure_once_ds41_admitted_then_popen(monkeypatch):
    log, check, popen = order_log(ADMIT)
    monkeypatch.setattr(ram_admit, "check", check)
    with serving_stubs(popen), pytest.raises(Stop):
        measure_once(DS41)()
    assert log == ["admit", "popen"]


def test_measure_once_non_ds41_never_checks(monkeypatch):
    log, check, popen = order_log(REFUSE)
    monkeypatch.setattr(ram_admit, "check", check)
    with serving_stubs(popen), pytest.raises(Stop):
        measure_once(OTHER)()
    assert log == ["popen"]


def test_model_identity_serve_refuses_before_popen(monkeypatch):
    log, check, popen = order_log(REFUSE)
    monkeypatch.setattr(ram_admit, "check", check)
    recipe = mock.Mock(argv=["llama-server", "-m", DS41], launch_env={})
    monkeypatch.setattr(model_identity.subprocess, "Popen", popen)
    with pytest.raises(ram_admit.RamAdmissionRefused):
        model_identity.serve(recipe, [])
    assert log == ["admit"]
    log.clear()
    recipe = mock.Mock(argv=["llama-server", "-m", OTHER], launch_env={})
    with pytest.raises(Stop):
        model_identity.serve(recipe, [])
    assert log == ["popen"]


def test_capacity_probe_refuses_before_popen(monkeypatch):
    log, check, popen = order_log(REFUSE)
    monkeypatch.setattr(ram_admit, "check", check)
    kw = dict(env={}, port=1, sequence=[], backend="cpu", popen=popen,
              sampler_factory=lambda: contextlib.nullcontext(mock.Mock()))
    with pytest.raises(ram_admit.RamAdmissionRefused):
        capacity_probe.run(argv=["llama-server", "-m", DS41], **kw)
    assert log == ["admit"]
    log.clear()
    with pytest.raises(Stop):
        capacity_probe.run(argv=["llama-server", "-m", OTHER], **kw)
    assert log == ["popen"]


def test_server_pair_refuses_before_popen(monkeypatch, tmp_path):
    log, check, popen = order_log(REFUSE)
    monkeypatch.setattr(ram_admit, "check", check)
    monkeypatch.setattr(ab_server_pair.subprocess, "Popen", popen)
    monkeypatch.setattr(ab_server_pair, "port_is_free", lambda p: True)
    spec = mock.Mock()
    pair = ab_server_pair.ServerPair.__new__(ab_server_pair.ServerPair)
    pair.specs = {"A": spec}
    pair.log_dir, pair.procs, pair.proof = tmp_path, {}, {}
    monkeypatch.setattr(ab_server_pair, "argv_for", lambda s: ["llama-server", "-m", DS41])
    monkeypatch.setattr(ab_server_pair, "env_for", lambda s: {})
    with pytest.raises(ram_admit.RamAdmissionRefused):
        try:
            pair.start()
        except Exception as exc:  # start() may wrap; unwrap to the cause
            raise exc.__cause__ or exc.__context__ or exc
    assert log == ["admit"]


def test_every_popen_launch_site_is_guarded():
    """Each server-start function uses the lifetime-admission launch helper."""
    for fn in (serving._measure_once, model_identity.serve, capacity_probe.run,
               ab_server_pair.ServerPair.start):
        src = inspect.getsource(fn)
        assert "ram_admit.popen(" in src


def test_non_ds41_exact_passthrough():
    spawn = mock.Mock(return_value=object())
    argv = ["server", "-m", OTHER]
    result = ram_admit.popen(argv, popen=spawn, env={"X": "y"}, start_new_session=True)
    assert result is spawn.return_value
    spawn.assert_called_once_with(argv, env={"X": "y"}, start_new_session=True)

def test_missing_or_untrusted_inode_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(ram_admit, "DS41_LEASE_PATH", str(tmp_path / "missing"))
    with pytest.raises(ram_admit.RamAdmissionRefused, match="lease unavailable"):
        ram_admit.popen(["server", "-m", DS41], popen=mock.Mock())

def test_lease_closes_on_check_and_spawn_failure(monkeypatch):
    argv = ["server", "-m", DS41]
    monkeypatch.setattr(ram_admit, "check", lambda *a, **k: REFUSE)
    with pytest.raises(ram_admit.RamAdmissionRefused):
        ram_admit.popen(argv, popen=mock.Mock())
    monkeypatch.setattr(ram_admit, "check", lambda *a, **k: ADMIT)
    with pytest.raises(Stop):
        ram_admit.popen(argv, popen=mock.Mock(side_effect=Stop))
    fd = ram_admit._lease_fd(ram_admit.DS41_LEASE_PATH)
    os.close(fd)


def test_insecure_lease_permissions_refused(monkeypatch):
    os.chmod(ram_admit.DS41_LEASE_PATH, 0o666)
    with pytest.raises(ram_admit.RamAdmissionRefused, match="lease unavailable"):
        ram_admit.popen(["server", "-m", DS41], popen=mock.Mock())


def test_exec_failure_releases_lease(monkeypatch):
    monkeypatch.setattr(ram_admit, "check", lambda *a, **k: ADMIT)
    monkeypatch.setattr(ram_admit.os, "execvpe", mock.Mock(side_effect=FileNotFoundError))
    with pytest.raises(FileNotFoundError):
        ram_admit.exec_ds41(["server", "-m", DS41])
    fd = ram_admit._lease_fd(ram_admit.DS41_LEASE_PATH)
    os.close(fd)


def test_ancestor_walk_rejects_writable_and_missing_paths(tmp_path):
    # The test root under the host's writable tmp is itself intentionally refused.
    with pytest.raises(PermissionError):
        REAL_TRUSTED_ANCESTORS(str(tmp_path))
    fd = REAL_TRUSTED_ANCESTORS("/usr")
    os.close(fd)
    with pytest.raises(OSError):
        REAL_TRUSTED_ANCESTORS("/usr/not-a-real-lease-directory")
