"""GPS-PIN-2: OCC-1 reader affinity enforcement. Synthetic /proc; no server."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.benchmark.occ1 import run_occ1

LANE = "184-191"
SH = Path(run_occ1.__file__).with_name("launch_reader.sh")
la = run_occ1._load_launch_affinity()


def _proc(tmp, pid, masks, ppid):
    proc = tmp / "proc"
    for tid, m in masks.items():
        d = proc / str(pid) / "task" / str(tid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "status").write_text(f"Cpus_allowed_list:\t{m}\n")
        (d / "comm").write_text("llama-server\n")
    (proc / str(pid) / "stat").write_text(f"{pid} (llama-server) S {ppid} 1 1 0\n")
    return proc


class NS:
    def __init__(self, **kw): self.__dict__.update(kw)


def _la(proc, ok=True):
    def setter(tid, cpus):
        if not ok:
            raise OSError(1, "EPERM")
        (proc / "900/task" / str(tid) / "status").write_text(f"Cpus_allowed_list:\t{la.format_cpu_list(cpus)}\n")
    class Shim:
        LaunchAffinityError = la.LaunchAffinityError
        parent_pid = staticmethod(lambda pid: la.parent_pid(pid, proc))
        @staticmethod
        def enforce_pinned_child(pid, cpus, label="", launcher_pid=None):
            return la.enforce_pinned_child(pid, cpus, label=label, proc_root=proc, setter=setter,
                                           launcher_pid=launcher_pid)
    return Shim()


def test_repin_then_record(tmp_path):
    proc = _proc(tmp_path, 900, {900: LANE, 901: "0-191"}, ppid=55)
    rec = run_occ1.enforce_reader_affinity(NS(server_pid=900), tmp_path, la=_la(proc))
    assert rec["ok"] and len(rec["repinned"]) == 1
    assert json.loads((tmp_path / "affinity.json").read_text())["ok"]


def test_residual_escape_refuses_run(tmp_path):
    proc = _proc(tmp_path, 900, {900: LANE, 901: "0-191"}, ppid=55)
    with pytest.raises(SystemExit, match="affinity refused"):
        run_occ1.enforce_reader_affinity(NS(server_pid=900), tmp_path, la=_la(proc, ok=False))
    assert json.loads((tmp_path / "affinity.json").read_text())["ok"] is False


def test_no_server_pid_is_recorded_as_unverified(tmp_path):
    rec = run_occ1.enforce_reader_affinity(NS(server_pid=None), tmp_path, la=_la(tmp_path))
    assert rec["ok"] is None and "NOT verified" in rec["skipped"]


def test_launcher_exports_knob_before_exec_taskset():
    text = SH.read_text()
    assert subprocess.run(["bash", "-n", str(SH)]).returncode == 0
    assert text.index("export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0") < text.index("exec taskset")
