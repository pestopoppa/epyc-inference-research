"""GPS-PIN-2: the pinned llama-bench drivers set the ROCr knob, re-pin, and refuse on escape.

Synthetic /proc and a fake Popen only -- no bench is ever launched.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LANE = "184-191"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"gpspin2_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODS = [_load(n) for n in ("champion_anchor_validation", "mmq_mfma_recheck",
                           "boundary_20260901_helper")]


class FakeProc:
    pid = 4242
    returncode = 0

    def __init__(self, argv, **kw):
        FakeProc.env = kw["env"]
        FakeProc.argv = argv

    def communicate(self, timeout=None):
        return '[{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0}]', ""


def _watch_factory(mod, tmp_path, masks):
    proc = tmp_path / "proc"
    for tid, mask in masks.items():
        d = proc / "4242" / "task" / str(tid)
        d.mkdir(parents=True)
        (d / "status").write_text(f"Cpus_allowed_list:\t{mask}\n")
        (d / "comm").write_text("llama-bench\n")
    (proc / "4242" / "stat").write_text("4242 (llama-bench) S 77 1 1 0\n")
    real = mod.la.PinWatch

    def setter(tid, cpus):
        if setter.fail:
            return
        (proc / "4242" / "task" / str(tid) / "status").write_text(
            f"Cpus_allowed_list:\t{mod.la.format_cpu_list(cpus)}\n")
    setter.fail = False

    def factory(pid, cpus):
        return real(pid, cpus, interval_s=0.01, proc_root=proc, setter=setter, launcher_pid=77)
    return factory, setter


@pytest.mark.parametrize("mod", MODS, ids=lambda m: m.__name__)
def test_knob_and_repin_then_record(mod, monkeypatch, tmp_path):
    factory, _ = _watch_factory(mod, tmp_path, {1: LANE, 2: "0-191"})
    monkeypatch.setattr(mod.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(mod.la, "PinWatch", factory)
    env = {"A": "1", "HSA_OVERRIDE_CPU_AFFINITY_DEBUG": "1"}
    sink: list = []
    done = mod.run_pinned(["taskset", "-c", LANE, "x"], env, 5, "t", sink)
    assert FakeProc.env["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] == "0"
    assert done.returncode == 0 and done.stdout.startswith("[")
    rec = sink[0]
    assert rec["ok"] and rec["repinned"] and rec["label"] == "t"


@pytest.mark.parametrize("mod", MODS, ids=lambda m: m.__name__)
def test_residual_escape_refuses(mod, monkeypatch, tmp_path):
    factory, setter = _watch_factory(mod, tmp_path, {1: LANE, 2: "0-191"})
    setter.fail = True  # the re-pin does not stick
    monkeypatch.setattr(mod.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(mod.la, "PinWatch", factory)
    sink: list = []
    with pytest.raises(RuntimeError, match="pin REFUSED"):
        mod.run_pinned(["taskset", "-c", LANE, "x"], {}, 5, "t", sink)
    assert sink and not sink[0]["ok"] and sink[0]["escapes_after"]


@pytest.mark.parametrize("mod", MODS, ids=lambda m: m.__name__)
def test_not_our_child_refuses(mod, monkeypatch, tmp_path):
    factory, _ = _watch_factory(mod, tmp_path, {1: LANE})
    monkeypatch.setattr(mod.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(mod.la, "PinWatch",
                        factory)
    (tmp_path / "proc" / "4242" / "stat").write_text("4242 (x) S 1 1 1 0\n")
    with pytest.raises(RuntimeError, match="pin REFUSED"):
        mod.run_pinned(["taskset", "-c", LANE, "x"], {}, 5, "t", [])
