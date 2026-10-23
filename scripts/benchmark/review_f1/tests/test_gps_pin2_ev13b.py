"""GPS-PIN-2: ev13b Server pins through launch_affinity. Synthetic /proc; no server."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT / "scripts/kernel_rnd"))
sys.path.insert(0, str(WT / "scripts/benchmark"))
sys.path.insert(0, str(WT / "scripts/benchmark/review_f1"))
import ev13b_run as ev  # noqa: E402
from autokernel.loop import launch_affinity as la  # noqa: E402

LANE = "184-191"


class FakeProc:
    pid = 900
    returncode = None
    def poll(self): return self.returncode
    def send_signal(self, s): self.returncode = 0
    def wait(self, timeout=None): return 0
    def kill(self): self.returncode = -9


class FakeResp:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return b'{"model_path": "m", "chat_template": ""}'


def _proc(tmp, masks):
    proc = tmp / "proc"
    for tid, m in masks.items():
        d = proc / "900" / "task" / str(tid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "status").write_text(f"Cpus_allowed_list:\t{m}\n")
        (d / "comm").write_text("llama-server\n")
    (proc / "900/stat").write_text("900 (llama-server) S 77 1 1 0\n")
    return proc


def _wire(monkeypatch, tmp, masks, env_seen, setter_ok=True):
    proc = _proc(tmp, masks)
    def setter(tid, cpus):
        if not setter_ok:
            raise OSError(1, "EPERM")
        (proc / "900/task" / str(tid) / "status").write_text(f"Cpus_allowed_list:\t{la.format_cpu_list(cpus)}\n")
    real = la.enforce_pinned_child
    monkeypatch.setattr(ev.la, "enforce_pinned_child",
                        lambda pid, cpus, label="": real(pid, cpus, label=label, proc_root=proc,
                                                         setter=setter, launcher_pid=77))
    monkeypatch.setattr(ev.subprocess, "Popen", lambda argv, **kw: (env_seen.update(kw["env"]), FakeProc())[1])
    monkeypatch.setattr(ev.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    monkeypatch.setattr(ev.residency, "vram_bytes", lambda: 64 * 2**30)
    class S:
        proof = {}
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(ev.residency, "Sampler", S)


def test_judge_launch_sets_knob_repins_records(monkeypatch, tmp_path):
    env: dict = {}
    _wire(monkeypatch, tmp_path, {900: LANE, 901: "0-191"}, env)
    srv = ev.Server(ev.judge_argv("/m.gguf", []), 1, tmp_path / "s.log")
    with srv:
        pass
    assert env["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] == "0"
    assert srv.record["affinity"]["ok"] and len(srv.record["affinity"]["repinned"]) == 1


def test_residual_escape_refuses_and_tears_down(monkeypatch, tmp_path):
    env: dict = {}
    _wire(monkeypatch, tmp_path, {900: LANE, 901: "0-191"}, env, setter_ok=False)
    srv = ev.Server(ev.judge_argv("/m.gguf", []), 1, tmp_path / "s.log")
    with pytest.raises(la.LaunchAffinityError):
        with srv:
            pass
    assert srv.proc.returncode == 0  # SIGTERM path ran (own pid only)
    assert srv.record["affinity"] is None
