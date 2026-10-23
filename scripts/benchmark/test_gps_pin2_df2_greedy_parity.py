"""GPS-PIN-2: df2_greedy_parity pins through launch_affinity. Synthetic /proc; no server."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("df2_gp", HERE / "df2_greedy_parity.py")
df2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(df2)
la = df2._load_launch_affinity()
LANE = "184-191"


def _proc(tmp: Path, pid: int, masks: dict, ppid: int) -> Path:
    proc = tmp / "proc"
    for tid, m in masks.items():
        d = proc / str(pid) / "task" / str(tid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "status").write_text(f"Cpus_allowed_list:\t{m}\n")
        (d / "comm").write_text("llama-server\n")
    (proc / str(pid) / "stat").write_text(f"{pid} (llama-server) S {ppid} 1 1 0\n")
    return proc


class FakeProc:
    pid = 4242
    returncode = None
    def poll(self): return self.returncode
    def send_signal(self, s): self.returncode = 0
    def wait(self, timeout=None): return 0
    def kill(self): self.returncode = -9


def _wire(monkeypatch, tmp, masks, seen_env, setter_ok=True):
    proc = _proc(tmp, 4242, masks, ppid=77)
    def setter(tid, cpus):
        if not setter_ok:
            raise OSError(1, "EPERM")
        d = proc / "4242" / "task" / str(tid)
        (d / "status").write_text(f"Cpus_allowed_list:\t{la.format_cpu_list(cpus)}\n")
    real = la.enforce_pinned_child
    class Shim:
        def __getattr__(self, n): return getattr(la, n)
        @staticmethod
        def enforce_pinned_child(pid, cpus, label=""):
            return real(pid, cpus, label=label, proc_root=proc, setter=setter, launcher_pid=77)
    monkeypatch.setattr(df2, "_load_launch_affinity", lambda: Shim())
    def popen(argv, **kw):
        seen_env.update(kw["env"]); return FakeProc()
    monkeypatch.setattr(df2.subprocess, "Popen", popen)
    monkeypatch.setattr(df2, "wait_ready", lambda p, l: None)
    monkeypatch.setattr(df2, "read_vram", lambda: df2.VRAM_RESIDENT_FLOOR + 1)
    monkeypatch.setattr(df2, "ask", lambda p, n: {"content": "x", "tokens": [1], "timings": {}})


def test_pinned_arm_sets_knob_repins_and_records(monkeypatch, tmp_path):
    env: dict = {}
    _wire(monkeypatch, tmp_path, {4242: LANE, 4243: "0-191"}, env)
    r = df2.run_arm(tmp_path / "b" / "bin", tmp_path / "out", "baseline", [{"id": "q", "prompt": "p"}], 4, 128, LANE)
    assert env["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] == "0"
    assert r["affinity"]["ok"] and len(r["affinity"]["repinned"]) == 1
    assert json.loads((tmp_path / "out/baseline/affinity.json").read_text())["ok"]


def test_unpinned_arm_untouched(monkeypatch, tmp_path):
    env: dict = {}
    _wire(monkeypatch, tmp_path, {4242: "0-191"}, env)
    monkeypatch.delenv("HSA_OVERRIDE_CPU_AFFINITY_DEBUG", raising=False)
    r = df2.run_arm(tmp_path / "b" / "bin", tmp_path / "out", "baseline", [], 4, 128, None)
    assert "HSA_OVERRIDE_CPU_AFFINITY_DEBUG" not in env and r["affinity"] is None


def test_residual_escape_refuses_the_arm(monkeypatch, tmp_path):
    env: dict = {}
    _wire(monkeypatch, tmp_path, {4242: LANE, 4243: "0-191"}, env, setter_ok=False)
    with pytest.raises(df2.ArmRefused, match="escape"):
        df2.run_arm(tmp_path / "b" / "bin", tmp_path / "out", "baseline", [], 4, 128, LANE)
