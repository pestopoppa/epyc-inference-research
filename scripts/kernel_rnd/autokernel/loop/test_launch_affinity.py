"""Loader-side deterministic pinning (2026-10-22). Synthetic /proc only; nothing is launched."""

from __future__ import annotations

import dataclasses
import errno
from pathlib import Path

import pytest

from . import launch_affinity as la
from . import serving

LANE = "184-191"


def _task(proc: Path, pid: int, tid: int, mask: str) -> None:
    d = proc / str(pid) / "task" / str(tid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "status").write_text(f"Name:\tllama-server\nCpus_allowed_list:\t{mask}\n")
    (d / "comm").write_text("llama-server\n")


def _proc(tmp_path: Path, pid: int, masks: dict[int, str], ppid: int) -> Path:
    proc = tmp_path / "proc"
    for tid, mask in masks.items():
        _task(proc, pid, tid, mask)
    (proc / str(pid) / "stat").write_text(f"{pid} (llama-server) S {ppid} 1 1 0\n")
    return proc


def _setter(proc: Path, pid: int, calls: list):
    def setter(tid, cpus):
        calls.append(tid)
        _task(proc, pid, tid, la.format_cpu_list(cpus))
    return setter


def test_the_finding_is_repinned_and_submasks_kept(tmp_path):
    masks = {100 + i: LANE for i in range(196)}
    masks[107] = "0-191"
    masks[400] = "186"
    proc = _proc(tmp_path, 100, masks, ppid=55)
    assert [e["tid"] for e in la.escapes(100, LANE, proc)] == [107]
    calls: list = []
    res = la.pin_own_child(100, LANE, proc_root=proc, setter=_setter(proc, 100, calls),
                           launcher_pid=55)
    assert res is not None and res.ok and calls == [107] and res.tasks_seen == 197
    assert la.escapes(100, LANE, proc) == []
    assert dict((t, la.format_cpu_list(m)) for t, _c, m in la.task_masks(100, proc))[400] == "186"


def test_not_our_child_is_never_touched(tmp_path):
    proc = _proc(tmp_path, 100, {100: LANE, 101: "0-191"}, ppid=1)
    calls: list = []
    assert la.pin_own_child(100, LANE, proc_root=proc, setter=_setter(proc, 100, calls),
                            launcher_pid=55) is None
    assert calls == []


def test_eperm_leaves_an_escape_and_fails(tmp_path):
    proc = _proc(tmp_path, 100, {100: LANE, 101: "0-191"}, ppid=55)

    def setter(tid, cpus):
        raise OSError(errno.EPERM, "nope")

    res = la.pin_all_tasks(100, LANE, proc_root=proc, setter=setter)
    assert not res.ok and res.errors and res.escapes_after[0]["tid"] == 101


def test_vanished_pid_is_not_ok(tmp_path):
    assert not la.pin_all_tasks(9, LANE, proc_root=tmp_path, setter=lambda *a: None).ok


@pytest.mark.parametrize("bad", ["", "x", "3-1", "1-"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        la.parse_cpu_list(bad)


def test_format_round_trip():
    assert la.format_cpu_list(la.parse_cpu_list("0-3,7,96-99")) == "0-3,7,96-99"


def test_loader_env_is_unchanged_and_the_repin_is_wired():
    """Layer 1 is deliberately NOT in the AK loader (identity-bearing env; package D2)."""
    import inspect

    assert la.ROCR_AFFINITY_ENV_KEY not in serving.LOADER_OWNED_ENV
    src = inspect.getsource(serving._measure_once)
    assert "launch_affinity.pin_own_child(srv.pid, recipe.cpu_list)" in src
    assert src.index("pin_own_child") < src.index('one(i, "measurement")')
    assert 'affinity_pin["escapes_end"]' in src


def test_every_canonical_gpu_serving_recipe_pins():
    root = Path(__file__).resolve().parents[4] / "artifacts" / "serving-recipes"
    paths = sorted(root.rglob("*.json"))
    assert paths, f"no serving recipes under {root}"
    checked = 0
    for path in paths:
        try:
            recipe = serving.Recipe.load(path)
        except Exception:
            continue  # non-recipe JSON under the tree
        if recipe.device == "none":
            continue
        assert recipe.cpu_list, f"{path.name}: GPU recipe without a cpu_list"
        la.parse_cpu_list(recipe.cpu_list)
        checked += 1
    assert checked >= 1


# --- GPS-PIN-2 (2026-10-23): the driver API one-off research GPU drivers use ---------------

def test_prefix_cpuset_and_layer1_env():
    env = {la.ROCR_AFFINITY_ENV_KEY: "1"}
    assert la.prefix_cpuset(["taskset", "-c", LANE, "numactl", "x"]) == LANE
    assert la.apply_pinned_launch_env(env, ["taskset", "-c", LANE, "llama-server"])
    assert env[la.ROCR_AFFINITY_ENV_KEY] == "0"  # an inherited "1" would re-open the escape
    bare: dict = {}
    assert not la.apply_pinned_launch_env(bare, ["llama-server"]) and bare == {}
    with pytest.raises(la.LaunchAffinityError):
        la.prefix_cpuset(["taskset", "-p", "1"])


def test_enforce_repins_records_and_refuses(tmp_path):
    masks = {100 + i: LANE for i in range(196)}
    masks[107] = "0-191"
    proc = _proc(tmp_path, 100, masks, ppid=55)
    (proc / "100" / "environ").write_bytes(b"A=1\0HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0\0")
    calls: list = []
    rec = la.enforce_pinned_child(100, LANE, proc_root=proc, setter=_setter(proc, 100, calls),
                                  launcher_pid=55)
    assert rec["ok"] and calls == [107] and rec["census"] == {LANE: 196}
    assert rec["env_knob"] == "0"
    with pytest.raises(la.LaunchAffinityError, match="not a child"):
        la.enforce_pinned_child(100, LANE, proc_root=proc, setter=_setter(proc, 100, []),
                                launcher_pid=56)

    def eperm(tid, cpus):
        raise OSError(errno.EPERM, "nope")

    _task(proc, 100, 300, "0-191")
    with pytest.raises(la.LaunchAffinityError, match="escape"):
        la.enforce_pinned_child(100, LANE, proc_root=proc, setter=eperm, launcher_pid=55)


def test_pin_watch_repins_while_running_and_refuses_when_it_cannot(tmp_path):
    proc = _proc(tmp_path, 100, {100: LANE, 101: "0-191"}, ppid=55)
    calls: list = []
    w = la.PinWatch(100, LANE, proc_root=proc, setter=_setter(proc, 100, calls), launcher_pid=55)
    w.poll()
    _task(proc, 100, 102, "0-191")  # a thread created mid-run
    w.poll()
    assert calls == [101, 102] and w.record["ok"] and w.record["polls"] == 2
    assert w.record["census"] == {LANE: 3}

    stranger = la.PinWatch(100, LANE, proc_root=proc, setter=_setter(proc, 100, []), launcher_pid=9)
    with stranger:
        pass
    assert not stranger.record["ok"] and stranger.record["polls"] == 0

    def eperm(tid, cpus):
        raise OSError(errno.EPERM, "nope")

    _task(proc, 100, 103, "0-191")
    bad = la.PinWatch(100, LANE, proc_root=proc, setter=eperm, launcher_pid=55)
    bad.poll()
    assert not bad.record["ok"] and bad.record["escapes_after"]


def test_pin_watch_thread_polls_and_stops(tmp_path):
    proc = _proc(tmp_path, 100, {100: LANE}, ppid=55)
    with la.PinWatch(100, LANE, interval_s=0.01, proc_root=proc, setter=lambda *a: None,
                     launcher_pid=55) as w:
        import time
        deadline = time.monotonic() + 5
        while w.polls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert w.record["ok"] and w._thread is not None and not w._thread.is_alive()


def test_cli_enforce_and_check(tmp_path, capsys):
    import json
    proc = _proc(tmp_path, 100, {100: LANE, 101: LANE}, ppid=55)
    assert la.main(["--check", "100", LANE, "--json", "--proc-root", str(proc)]) == 0
    assert json.loads(capsys.readouterr().out)["census"] == {LANE: 2}
    # --enforce refuses a pid the caller did not spawn (launcher 56 != ppid 55)
    assert la.main(["--enforce", "100", LANE, "--launcher-pid", "56", "--json",
                    "--proc-root", str(proc)]) == 1
    assert "not a child" in json.loads(capsys.readouterr().out)["refused"]
