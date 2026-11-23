"""AKU-22: foreign-load sampler wired into residency. Synthetic proc/sysfs only, no compute."""
from pathlib import Path
import time

from . import foreign_residency as fr
from . import residency


def proc(root, pid, cpus, ticks=0):
    d = Path(root) / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    f = ["0"] * 50
    f[0], f[11] = "S", str(ticks)
    (d / "stat").write_text(f"{pid} (w) " + " ".join(f))
    (d / "status").write_text(f"Cpus_allowed_list:\t{cpus}\n")


def topo(root, n=4):
    for c in range(2 * n):
        d = Path(root) / f"cpu{c}" / "topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(f"{c % n},{c % n + n}\n")


def make(tmp_path, bound=None, pid=10):
    sysfs, procfs = tmp_path / "sys", tmp_path / "proc"
    topo(sysfs)
    proc(procfs, pid, "0-3")
    path = str(sysfs / "cpu{}" / "topology" / "thread_siblings_list")
    obs = fr.ForeignLoadObserver(pid, bench_cpus="0-3", bound_pct=bound, proc=str(procfs),
                                 sibling_path=path)
    return obs, procfs


def test_reuses_shared_module_and_expands_siblings(tmp_path):
    obs, _ = make(tmp_path)
    assert obs._fl.__file__.endswith("scripts/utils/foreign_load.py")
    assert obs.bench_logical == set(range(8))  # 4-7 are siblings of bench cores 0-3


def test_sibling_pinned_burst_counts_and_uses_live_deltas(tmp_path):
    obs, procfs = make(tmp_path, bound=50.0)
    proc(procfs, 20, "4-7", ticks=0)           # pinned ONLY to siblings of the bench cores
    prev, t0 = obs._fl._snapshot(str(procfs)), time.monotonic() - 1.0
    proc(procfs, 20, "4-7", ticks=obs._fl.HZ * 3)   # burned ~3 CPU-s in the ~1s window
    obs.sample(prev, t0)
    s = obs.summary
    assert s["samples"] == 1 and s["peak_foreign_pct"] > 100
    assert s["peak"]["top"][0]["pid"] == 20 and s["samples_over_bound"] == 1
    assert s["clean"] is False and s["method"].startswith("proc_stat_tick_deltas")


def test_own_process_and_disjoint_pinning_are_not_foreign(tmp_path):
    obs, procfs = make(tmp_path, bound=1.0)
    proc(procfs, 30, "8-9")                     # outside the bench's physical cores
    prev, t0 = obs._fl._snapshot(str(procfs)), time.monotonic() - 1.0
    proc(procfs, 10, "0-3", ticks=obs._fl.HZ * 5)   # the watched server itself
    proc(procfs, 30, "8-9", ticks=obs._fl.HZ * 5)
    obs.sample(prev, t0)
    s = obs.summary
    assert s["peak_foreign_pct"] == 0 and s["clean"] is True


def test_zero_samples_is_not_clean(tmp_path):
    obs, _ = make(tmp_path, bound=5.0)
    s = obs.summary
    assert s["samples"] == 0 and s["clean"] is None and s["median_foreign_pct"] is None


def test_sampler_proof_carries_foreign_load_and_never_raises(monkeypatch):
    monkeypatch.setattr(residency, "vram_bytes", lambda: -1)
    monkeypatch.setattr(residency, "kfd_processes", lambda: 0)
    monkeypatch.setattr(residency, "sclk_mhz", lambda: 0)
    monkeypatch.setattr(residency, "CpuLifecycleSampler",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no cpu observer")))
    import os
    sampler = residency.Sampler(interval=0.05)
    sampler.set_foreign_bound(25.0)
    sampler.watch_pid(os.getpid())
    with sampler:
        time.sleep(0.15)
    fl = sampler.proof["foreign_load"]
    assert fl["schema"] == fr.SCHEMA and fl["bound_pct"] == 25.0 and fl["pid"] == os.getpid()


def test_missing_shared_module_is_recorded_not_silent(monkeypatch):
    def boom(*a, **k):
        raise ImportError("gone")
    monkeypatch.setattr(fr, "load_sampler_module", boom)
    monkeypatch.setattr(residency, "CpuLifecycleSampler", lambda *a, **k: None)
    sampler = residency.Sampler()
    sampler.watch_pid(1)
    proof = sampler.proof
    assert "foreign_load" not in proof and "ImportError" in proof["foreign_load_error"]
