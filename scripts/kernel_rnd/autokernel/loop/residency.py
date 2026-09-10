#!/usr/bin/env python3
"""Prove the run actually happened on the GPU.

"I invoked the HIP build" is not evidence of a HIP run, and `ldd` cannot supply it:
llama.cpp **dlopens** `libggml-hip.so`, so the executable shows zero HIP linkage
either way, while `/etc/environment` puts the CPU build early in `LD_LIBRARY_PATH`.
Three ggml generations live on this host and a binary that inherits another tree's
ggml runs silently wrong.

So residency is sampled DURING the run. A sample taken afterwards proves nothing --
llama-bench frees its allocation on the way out, which is exactly why a post-hoc
reading of 0% VRAM is the NORMAL result and not evidence of a CPU run.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import os
import re
import statistics
import threading
import time

VRAM_SYSFS = Path("/sys/class/drm/card2/device/mem_info_vram_used")
KFD_PROC = Path("/sys/class/kfd/kfd/proc")
#: The ACHIEVED core clock, in Hz. Not `pp_dpm_sclk`.
#:
#: The first version of this read `pp_dpm_sclk` and took the starred DPM level. Under
#: `power_dpm_force_performance_level = high` that file stars BOTH levels:
#:
#:     0: 1700Mhz *
#:     1: 1700Mhz *
#:
#: so the reader returned 1700 unconditionally, `min == max` by construction, and
#: `clock_stable` was True on every run forever -- a check that cannot fail is not a
#: check. It also could not see a droop BELOW the cap, and `pp_features` shows
#: `APCC_DFLL` enabled (the DFLL droops on current spikes) plus DS_SOCCLK / DS_FCLK /
#: DS_LCLK still dynamic under `high`. hwmon reports what the clock actually did.
SCLK_HWMON = sorted(Path("/sys/class/drm/card2/device/hwmon").glob("hwmon*/freq1_input"))
#: A model resident on the device moves VRAM well past this. Below it, "resident" is
#: not proven -- which is a refusal, not a warning.
RESIDENT_FLOOR_BYTES = 1 << 30


def vram_bytes() -> int:
    try:
        return int(VRAM_SYSFS.read_text().strip())
    except (OSError, ValueError):
        return -1


def sclk_mhz() -> int:
    """The achieved core clock in MHz; 0 if unreadable.

    Reads hwmon rather than the DPM table, so a droop below the pinned cap is visible.
    """
    for path in SCLK_HWMON:
        try:
            return int(path.read_text().strip()) // 1_000_000
        except (OSError, ValueError):
            continue
    return 0


def kfd_processes() -> int:
    try:
        return len(list(KFD_PROC.iterdir()))
    except OSError:
        return -1


class Sampler:
    """Peak VRAM and KFD process count over one process lifetime."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_vram = 0
        self.peak_kfd = 0
        self.min_sclk = 0
        self.max_sclk = 0
        self.samples = 0
        #: Every SUCCESSFUL VRAM read, in order. Kept because a peak alone cannot
        #: distinguish "the device held the model throughout" from "one spike"; and
        #: because its LENGTH is the only thing that separates a window that read zero
        #: from a window whose sysfs node could not be read at all. `vram_bytes()`
        #: returns -1 on a failed read and -1 never enters this list, so
        #: `len(vram_readings) == 0` with `samples > 0` means UNSAMPLEABLE, not idle.
        self.vram_readings: list[int] = []

    def _loop(self) -> None:
        while not self._stop.is_set():
            vram = vram_bytes()
            if vram >= 0:
                self.vram_readings.append(vram)
                self.peak_vram = max(self.peak_vram, vram)
            self.peak_kfd = max(self.peak_kfd, kfd_processes())
            clock = sclk_mhz()
            if clock:
                self.max_sclk = max(self.max_sclk, clock)
                self.min_sclk = clock if not self.min_sclk else min(self.min_sclk, clock)
            self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "Sampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def proof(self) -> dict:
        # Snapshot: the sampling thread may still be appending, and `statistics.median`
        # sorts a copy either way.
        readings = list(self.vram_readings)
        return {
            "peak_vram_bytes": self.peak_vram,
            # The peak answers "did it ever get there"; the median answers "did it STAY
            # there". A window whose peak clears the floor on one sample out of forty is
            # not the same evidence as one that held the model for the whole run.
            "median_vram_bytes": int(statistics.median(readings)) if readings else 0,
            # Successful reads, NOT loop iterations. Zero here with a non-zero `samples`
            # is an unreadable instrument, which is a different fact from a zero reading.
            "vram_reads": len(readings),
            "peak_kfd_processes": self.peak_kfd,
            "sclk_min_mhz": self.min_sclk,
            "sclk_max_mhz": self.max_sclk,
            # A measurement taken across a clock change is not a measurement of the
            # kernel; it is partly a measurement of the governor.
            "clock_stable": bool(self.min_sclk) and self.min_sclk == self.max_sclk,
            "samples": self.samples,
            "resident": self.peak_vram >= RESIDENT_FLOOR_BYTES,
        }


class CpuLifecycleSampler:
    """Bounded factual procfs observations, NOT a placement/contention warrant.

    Allowed NUMA nodes are permissions, not actual page placement. Each task is
    identified by TID/start ticks; the process must retain its attached identity.
    Phase-boundary-crossing reads, missing reads and exhausted budgets stay visible.
    No campaign, container, lock or scientific identity is manufactured here.
    """

    def __init__(self, *, proc_root: Path = Path("/proc"), interval: float = 1.0,
                 max_samples: int = 8192, max_tasks: int = 512,
                 max_bytes: int = 32 << 20, max_sample_s: float = 0.1,
                 max_processes: int = 2048):
        for value, limit in ((max_samples, 8192), (max_tasks, 512),
                             (max_bytes, 32 << 20), (max_processes, 2048)):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError("CPU observation count/byte budget is out of bounds")
        if not all(math.isfinite(value) and 0 < value <= 60
                   for value in (interval, max_sample_s)):
            raise ValueError("CPU observation timing budget is out of bounds")
        self.proc_root, self.interval = proc_root, interval
        self.max_samples, self.max_tasks = max_samples, max_tasks
        self.max_bytes, self.max_sample_s = max_bytes, max_sample_s
        self.max_processes = max_processes
        # Only the preceding bounded census, not another accumulated history.
        self._previous_processes = {}
        self._previous_census_at = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._target = None
        self._phase = "setup"
        self._generation = 0
        self._samples = []
        self._markers = []
        self._errors = []
        self._bytes = 0
        self._last_sample = None
        self._finished = False
        self._truncated = False
        self._boot_id = None

    @staticmethod
    def _text(path: Path) -> str:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            raw = os.read(fd, 16385)
            if len(raw) > 16384:
                raise ValueError("proc read byte bound exceeded")
            return raw.decode("ascii")
        finally:
            os.close(fd)

    def _identity(self, root: Path, pid: int) -> int:
        text = self._text(root / "stat")
        prefix, separator, tail = text.rpartition(") ")
        if not separator or prefix.split(" (", 1)[0] != str(pid):
            raise ValueError("proc stat PID differs")
        start_ticks = int(tail.split()[19])
        if start_ticks < 0:
            raise ValueError("negative process start ticks")
        return start_ticks

    def _task(self, root: Path, pid: int) -> dict:
        before = self._identity(root, pid)
        fields = {}
        for line in self._text(root / "status").splitlines():
            key, _, value = line.partition(":")
            if key in {"Cpus_allowed_list", "Mems_allowed_list"}:
                value = value.strip()
                if key in fields or not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
                    raise ValueError("malformed or duplicate allowed-list field")
                previous = -1
                for part in value.split(","):
                    bounds = [int(number) for number in part.split("-")]
                    if bounds[0] <= previous or bounds[-1] < bounds[0]:
                        raise ValueError("unordered allowed-list field")
                    previous = bounds[-1]
                fields[key] = value
        if len(fields) != 2:
            raise ValueError("missing CPU/NUMA allowed-list fields")
        if self._identity(root, pid) != before:
            raise ValueError("PID/TID identity changed during read")
        return {"id": pid, "start_ticks": before, **fields}

    def _task_ids(self, root: Path) -> list[int]:
        ids = []
        with os.scandir(root / "task") as entries:
            for entry in entries:
                if not entry.name.isdecimal():
                    raise ValueError("unexpected task directory entry")
                ids.append(int(entry.name))
                if len(ids) > self.max_tasks:
                    raise ValueError("task count bound exceeded")
        return sorted(ids)

    def _host(self) -> dict:
        """Original counters, not a host-health verdict or a quiet-host threshold."""
        from .lifecycle_observation import _key_values, _psi
        row = {"cpu_ticks": None, "memory_kib": None, "swap_pages": None,
               "memory_psi": None, "errors": []}
        readers = {
            "memory_kib": lambda: _key_values(self._text(self.proc_root / "meminfo"),
                ("MemAvailable", "SwapFree", "SwapTotal"), "meminfo"),
            "swap_pages": lambda: _key_values(self._text(self.proc_root / "vmstat"),
                ("pswpin", "pswpout"), "vmstat"),
            "memory_psi": lambda: _psi(self._text(self.proc_root / "pressure/memory")),
        }
        try:
            # Only the aggregate first line is required; /proc/stat's per-CPU tail
            # may exceed the read bound on this host and is not part of this fact.
            fd = os.open(self.proc_root / "stat", os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            try:
                raw = os.read(fd, 16384)
            finally:
                os.close(fd)
            if b"\n" not in raw:
                raise ValueError("aggregate CPU stat line is incomplete")
            fields = raw.split(b"\n", 1)[0].decode("ascii").split()
            values = [int(value) for value in fields[1:]]
            if fields[0] != "cpu" or len(values) != 10 or min(values) < 0:
                raise ValueError("aggregate CPU stat fields differ")
            row["cpu_ticks"] = dict(zip(("user", "nice", "system", "idle", "iowait",
                "irq", "softirq", "steal", "guest", "guest_nice"), values))
        except Exception as exc:
            row["errors"].append(f"cpu_ticks: {type(exc).__name__}: {exc}"[:256])
        for key, read in readers.items():
            try:
                value = read()
                if key == "memory_psi" and any(
                        not math.isfinite(number) or number < 0
                        for counters in value.values() for number in counters.values()):
                    raise ValueError("PSI contains negative/nonfinite counters")
                row[key] = value
            except Exception as exc:
                row["errors"].append(f"{key}: {type(exc).__name__}: {exc}"[:256])
        return row

    def _non_target_activity(self, target, deadline: float) -> dict:
        """Bounded identity-stable CPU deltas. Names are facts, never a classifier.

        A non-target process may be our parent, an ordinary build, an idle server,
        or somebody else's work. Neither CPU ticks nor an executable name establishes
        competing model inference, held-region overlap, or permission to signal it.
        """
        from .lifecycle_observation import parse_proc_stat
        started = time.monotonic()
        row = {"started_monotonic_s": started, "ended_monotonic_s": None,
               "previous_started_monotonic_s": self._previous_census_at,
               "entries_seen": 0, "processes_read": 0, "incomplete": False,
               "errors": [], "active_intervals": [], "classification": "unproven"}
        current = {}
        def error(exc):
            row["incomplete"] = True
            if len(row["errors"]) < 8:
                row["errors"].append(str(exc)[:256])
        try:
            with os.scandir(self.proc_root) as entries:
                for entry in entries:
                    row["entries_seen"] += 1
                    if row["entries_seen"] > self.max_processes + 256:
                        raise ValueError("proc directory entry bound exceeded")
                    if time.monotonic() > deadline or self._stop.is_set():
                        raise ValueError("process census time bound exceeded or observer closing")
                    if not entry.name.isdecimal():
                        continue
                    pid = int(entry.name)
                    if target is not None and pid == target["pid"]:
                        continue
                    if row["processes_read"] >= self.max_processes:
                        raise ValueError("process census count bound exceeded")
                    row["processes_read"] += 1
                    try:
                        root = self.proc_root / entry.name
                        before = parse_proc_stat(self._text(root / "stat"), pid)
                        allowed = self._task(root, pid)
                        after = parse_proc_stat(self._text(root / "stat"), pid)
                        if (before["start_ticks"] != after["start_ticks"]
                                or allowed["start_ticks"] != after["start_ticks"]
                                or after["cpu_ticks"] < before["cpu_ticks"]):
                            raise ValueError(f"PID {pid}: identity/counter changed during read")
                        observed = {**after, "Cpus_allowed_list": allowed["Cpus_allowed_list"]}
                        current[pid] = observed
                        previous = self._previous_processes.get(pid)
                        if previous is not None:
                            if previous["start_ticks"] != after["start_ticks"]:
                                error(f"PID {pid}: reused between samples")
                            elif after["cpu_ticks"] < previous["cpu_ticks"]:
                                error(f"PID {pid}: counter regressed between samples")
                            elif after["cpu_ticks"] > previous["cpu_ticks"]:
                                row["active_intervals"].append({"before": previous,
                                    "after": observed,
                                    "cpu_tick_delta": after["cpu_ticks"] - previous["cpu_ticks"]})
                    except Exception as exc:
                        error(f"PID {pid}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            error(exc)
        row["ended_monotonic_s"] = time.monotonic()
        if row["ended_monotonic_s"] > deadline:
            error("process census exceeded sample deadline")
        self._previous_processes = current
        self._previous_census_at = started
        return row

    def note_hook_failure(self, method: str, exc: Exception) -> None:
        with self._lock:
            if len(self._errors) < 16:
                self._errors.append(f"{method}: {type(exc).__name__}: {exc}"[:256])

    def phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            self._generation += 1
            if len(self._markers) < 32:
                self._markers.append({"phase": phase, "monotonic_s": time.monotonic(),
                                      "wall_s": time.time()})
            else:
                self._truncated = True
        self._wake.set()

    def checkpoint(self, marker: str) -> None:
        self.phase(marker)

    def start(self, phase: str = "setup") -> None:
        self.phase(phase)
        try:
            self._boot_id = self._text(self.proc_root / "sys/kernel/random/boot_id").strip()
        except Exception as exc:
            self.note_hook_failure("boot_id", exc)
        self._thread = threading.Thread(target=self._loop, name="cpu-lifecycle-observer", daemon=True)
        self._thread.start()

    def attach_target(self, pid: int) -> None:
        # Bind immediately after the owning Popen, never to a later reused PID.
        start_ticks = self._identity(self.proc_root / str(pid), pid)
        with self._lock:
            if self._target is not None:
                raise ValueError("CPU observer target already attached")
            self._target = {"pid": pid, "start_ticks": start_ticks}
        self._wake.set()

    def _sample(self) -> None:
        started = time.monotonic()
        with self._lock:
            if self._stop.is_set():
                return
            if len(self._samples) >= self.max_samples:
                self._truncated = True
                self._stop.set()
                return
            target = None if self._target is None else dict(self._target)
            phase, generation = self._phase, self._generation
        row = {"started_monotonic_s": started, "wall_s": time.time(), "phase": phase,
               "process": None, "tasks": [], "error": None, "identity_mismatch": None,
               "host": self._host(), "non_target_activity": None}
        try:
            if target is None:
                raise ValueError("target not attached yet; host facts retained")
            root = self.proc_root / str(target["pid"])
            observed_start = self._identity(root, target["pid"])
            if observed_start != target["start_ticks"]:
                row["identity_mismatch"] = {"expected_start_ticks": target["start_ticks"],
                                            "observed_start_ticks": observed_start}
                raise ValueError("attached process identity changed")
            row["process"] = self._task(root, target["pid"])
            ids = self._task_ids(root)
            if not ids:
                raise ValueError("empty task census")
            for tid in ids:
                if self._stop.is_set():
                    raise ValueError("observer closed during sample")
                if time.monotonic() - started > self.max_sample_s:
                    raise ValueError("sample duration bound exceeded")
                row["tasks"].append(self._task(root / "task" / str(tid), tid))
            if self._task_ids(root) != ids:
                raise ValueError("task census changed during read")
            observed_start = self._identity(root, target["pid"])
            if observed_start != target["start_ticks"]:
                row["identity_mismatch"] = {"expected_start_ticks": target["start_ticks"],
                                            "observed_start_ticks": observed_start}
                raise ValueError("attached process identity changed during sample")
            if time.monotonic() - started > self.max_sample_s:
                raise ValueError("sample duration bound exceeded")
            if self._stop.is_set():
                raise ValueError("observer closed during sample")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:256]
        row["non_target_activity"] = self._non_target_activity(target, started + self.max_sample_s)
        ended = time.monotonic()
        with self._lock:
            row.update(ended_monotonic_s=ended, phase_at_end=self._phase,
                       crosses_phase_boundary=generation != self._generation,
                       gap_before_s=None if self._last_sample is None else started - self._last_sample)
            size = len(json.dumps(row, separators=(",", ":")).encode("utf-8"))
            if self._bytes + size > self.max_bytes:
                self._truncated = True
                self._stop.set()
                return
            self._samples.append(row)
            self._bytes += size
            self._last_sample = started

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._wake.clear()
                self._sample()
                self._wake.wait(self.interval)
        except Exception as exc:
            self.note_hook_failure("sample", exc)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.phase("closed")
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    @property
    def observation(self) -> dict:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            status = "unavailable"
            if any(row["process"] is not None or row["tasks"] for row in self._samples):
                status = "observed" if any(row["error"] is None for row in self._samples) else "partial"
            # A detached snapshot: no background mutation of already archived rows.
            return json.loads(json.dumps({
                "schema": "epyc.autokernel.cpu_lifecycle_facts.v1",
                "status": status,
                "scope": "process_and_task_allowed_lists_not_numa_page_placement",
                "clock_domain": "serving_process_monotonic", "boot_id": self._boot_id,
                "target": self._target, "markers": self._markers, "samples": self._samples,
                "errors": self._errors, "truncated": self._truncated,
                "shutdown_resolved": not alive, "interval_s": self.interval,
                "bounds": {"samples": self.max_samples, "tasks_per_sample": self.max_tasks,
                           "processes_per_sample": self.max_processes,
                           "sample_bytes": self.max_bytes, "read_bytes": 16384,
                           "sample_duration_s": self.max_sample_s},
                "cpu_placement": "unproven", "contention": "unproven",
                "host_noise_policy": "ordinary_load_and_PSI_are_diagnostic_not_blockers",
                "foreign_inference_classification": "not_available"}))


def cpu_lifecycle_invalidity(facts: dict, cpu_list: str | None) -> list[dict]:
    """Only observed contradictions; missing facts do not become a clean warrant.

    Setup/exec may precede taskset, and teardown may legitimately lose the process.
    No pressure, load, process name, missing read or inferred inference is a veto.
    """
    allowed = set()
    if cpu_list is not None:
        for part in cpu_list.split(","):
            numbers = [int(value) for value in part.split("-")]
            allowed.update(range(numbers[0], numbers[-1] + 1))
    failures = []
    previous_outside = {}
    for index, row in enumerate(facts.get("samples", ())):
        if row.get("phase") not in {"health", "warmup", "measurement", "measurement_end"}:
            continue
        if row.get("crosses_phase_boundary"):
            previous_outside = {}
            continue
        if row.get("identity_mismatch") is not None:
            failures.append({"condition": "original_target_identity_changed", "sample_index": index,
                             **row["identity_mismatch"]})
        if row.get("error") is not None or not allowed:
            previous_outside = {}
            continue
        outside = {}
        for task in [row.get("process"), *row.get("tasks", ())]:
            if task is None:
                continue
            for part in task["Cpus_allowed_list"].split(","):
                numbers = [int(value) for value in part.split("-")]
                if any(cpu not in allowed for cpu in range(numbers[0], numbers[-1] + 1)):
                    key = (task["id"], task["start_ticks"])
                    outside[key] = task
                    break
        # The process and its leader task are the same identity, not two samples.
        for key, task in outside.items():
            previous = previous_outside.get(key)
            if previous is not None and previous[1] <= row["started_monotonic_s"]:
                failures.append({"condition": "task_affinity_outside_original_recipe",
                    "first_sample_index": previous[0], "sample_index": index,
                    "observations": 2, "task": task, "expected_cpu_list": cpu_list})
        previous_outside = {key: (index, row["ended_monotonic_s"]) for key in outside}
    return failures


def loader_env(binary: Path) -> dict[str, str]:
    """`LD_LIBRARY_PATH` that pins this build's own ggml.

    The binary's own directory FIRST. Inheriting the host default puts another
    tree's ggml ahead of it -- the three-generations hazard -- and the resulting run
    is wrong in a way no exit code reports.
    """
    import os
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:/opt/rocm/lib"
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    return env


__all__ = ["KFD_PROC", "RESIDENT_FLOOR_BYTES", "Sampler", "VRAM_SYSFS",
           "kfd_processes", "loader_env", "sclk_mhz", "vram_bytes"]
