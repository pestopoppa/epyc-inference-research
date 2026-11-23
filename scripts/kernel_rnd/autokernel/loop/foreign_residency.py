"""AKU-22: the SMT-correct foreign-load sampler, wired into residency.

Reuses `scripts/utils/foreign_load.py` (promoted from INF-70's scratch `foreign.py`) rather
than rebuilding it: live `/proc/<pid>/stat` tick deltas (never `ps %CPU`) and sibling
expansion of the bench region from sysfs (never assumed). `residency.Sampler.watch_pid`
starts one observer per watched process; its `summary` rides in `Sampler.proof` under
`foreign_load`.

Observation only. It records foreign load and whether a declared bound was exceeded; it
never terminates anything and gates nothing here -- the consumer decides what a breach means.
"""
from __future__ import annotations

import importlib.util
import statistics
import threading
import time
from pathlib import Path

SCHEMA = "epyc.autokernel.foreign_load_summary.v1"
_FOREIGN_LOAD = Path(__file__).resolve().parents[3] / "utils" / "foreign_load.py"
_MAX_SAMPLES = 4096
_module = None


def load_sampler_module(path: Path = _FOREIGN_LOAD):
    """Import the shared sampler by path; raises if it is absent (callers record that)."""
    global _module
    if path == _FOREIGN_LOAD and _module is not None:
        return _module
    spec = importlib.util.spec_from_file_location("epyc_foreign_load", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load foreign-load sampler from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if path == _FOREIGN_LOAD:
        _module = module
    return module


class ForeignLoadObserver:
    """Samples foreign CPU load on the bench region's physical cores while a pid runs.

    `bound_pct` is the declared foreign-load bound (percent of ONE core, as the sampler
    reports); a sample above it counts toward `samples_over_bound`. None = unbounded.
    """

    def __init__(self, pid: int, *, interval: float = 2.0, bench_cpus: str = "0-95",
                 bound_pct: float | None = None, module=None, proc: str = "/proc",
                 sibling_path: str | None = None) -> None:
        if interval <= 0 or (bound_pct is not None and bound_pct < 0):
            raise ValueError("foreign-load interval/bound out of range")
        self._fl = module or load_sampler_module()
        self.pid, self.interval, self.bound_pct = int(pid), interval, bound_pct
        self.bench_cpus, self._proc = bench_cpus, proc
        kwargs = {} if sibling_path is None else {"sibling_path": sibling_path}
        self.bench_logical = self._fl.bench_logical_cpus(
            self._fl.parse_cpu_list(bench_cpus), **kwargs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pct: list[float] = []
        self._peak: dict | None = None
        self._over = 0
        self._errors: list[str] = []

    def sample(self, prev, tprev):
        """One tick; returns the new (snapshot, time). Exposed so tests drive it directly."""
        now = time.monotonic()
        rec, cur = self._fl.sample_once(prev, now - tprev, self.bench_logical,
                                        self._fl._own_pids(self.pid, self._proc),
                                        proc=self._proc)
        with self._lock:
            if len(self._pct) < _MAX_SAMPLES:
                self._pct.append(rec["foreign_pct"])
                if self.bound_pct is not None and rec["foreign_pct"] > self.bound_pct:
                    self._over += 1
                if self._peak is None or rec["foreign_pct"] > self._peak["foreign_pct"]:
                    self._peak = {"foreign_pct": rec["foreign_pct"], "top": rec["top"][:3]}
        return cur, now

    def _loop(self) -> None:
        try:
            prev, tprev = self._fl._snapshot(self._proc), time.monotonic()
        except Exception as exc:  # noqa: BLE001 -- recorded, never raised into the run
            self._errors.append(f"{type(exc).__name__}: {exc}"[:256])
            return
        while not self._stop.wait(self.interval):
            try:
                prev, tprev = self.sample(prev, tprev)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    if len(self._errors) < 8:
                        self._errors.append(f"{type(exc).__name__}: {exc}"[:256])

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="foreign-load", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def summary(self) -> dict:
        with self._lock:
            pct, peak, over, errors = list(self._pct), self._peak, self._over, list(self._errors)
        return {"schema": SCHEMA, "pid": self.pid, "bench_cpus": self.bench_cpus,
                "bench_logical_cpus": len(self.bench_logical),
                "method": "proc_stat_tick_deltas_smt_sibling_expanded",
                "interval_s": self.interval, "samples": len(pct),
                "median_foreign_pct": statistics.median(pct) if pct else None,
                "peak_foreign_pct": max(pct) if pct else None, "peak": peak,
                "bound_pct": self.bound_pct, "samples_over_bound": over if self.bound_pct is not None else None,
                "errors": errors,
                # Zero samples is an instrument that never ran, not a quiet host.
                "clean": None if not pct else (over == 0 if self.bound_pct is not None else None)}
