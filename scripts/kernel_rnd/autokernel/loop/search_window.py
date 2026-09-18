"""Bounded same-attempt window observations; no grant or invented control panel.

Raw observations are retained before any owning predicate runs. A transport ACK,
an empty namespace, a post-close sample, or this module's receipt is not a search
verdict. The result owner must still bind original calibration/control/T0 material.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import fnmatch
import hashlib
import math
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Mapping

from ..execution import microbench
from ..resource import preflight
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import observation_binding as ob

CONFIG_SCHEMA = "epyc.autokernel.search_window_configuration.v1"
RECEIPT_SCHEMA = "epyc.autokernel.search_window_observation.v1"


class SearchWindowRefused(RuntimeError):
    pass


def _positive(value: Any, label: str, *, integer: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 or (integer and type(value) is not int)):
        raise SearchWindowRefused(f"{label} must be finite and positive")


@dataclass(frozen=True)
class WindowLimits:
    cadence_s: float = 0.1
    max_sample_duration_s: float = 0.08
    max_gap_s: float = 0.5
    max_boundary_duration_s: float = 2.0
    max_samples: int = 256
    max_read_bytes: int = 256 * 1024
    max_processes: int = 4096
    max_directory_entries: int = 8192
    max_snapshot_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive(value, name, integer=not name.endswith("_s"))
        if self.max_sample_duration_s > self.max_gap_s or self.cadence_s > self.max_gap_s:
            raise SearchWindowRefused("sample duration/cadence exceeds the declared gap")


@dataclass(frozen=True)
class InstalledSearchWindowConfiguration:
    claim_root: str
    proc_root: str
    sysfs_cpu_root: str
    storage_root: str
    cpu_regions: tuple[str, ...]
    host_policy: microbench.HostStatePolicy
    storage_floor_bytes_free: int
    expected_source_digest: str
    limits: WindowLimits = WindowLimits()
    # Original owning outputs are separate preparation dependencies. Their
    # locators alone never confer calibration or control authority.
    calibration_material_ref: str | None = None
    control_material_ref: str | None = None
    package_energy_paths: tuple[tuple[int, str, str], ...] = ()
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA or type(self.limits) is not WindowLimits:
            raise SearchWindowRefused("unsupported window configuration")
        for name in ("claim_root", "proc_root", "sysfs_cpu_root", "storage_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
                raise SearchWindowRefused(f"{name} requires an absolute explicit path")
        if (not isinstance(self.cpu_regions, tuple) or not self.cpu_regions
                or len(set(self.cpu_regions)) != len(self.cpu_regions)
                or any(not isinstance(item, str) or not item for item in self.cpu_regions)):
            raise SearchWindowRefused("an exact nonempty CPU-region scope is required")
        if type(self.host_policy) is not microbench.HostStatePolicy:
            raise SearchWindowRefused("host policy requires the existing concrete type")
        for name in ("min_frequency_ratio", "max_load_per_core"):
            _positive(getattr(self.host_policy, name), f"host policy {name}")
        for name in ("require_frequency", "require_load", "require_package_power"):
            if type(getattr(self.host_policy, name)) is not bool:
                raise SearchWindowRefused(f"host policy {name} requires a boolean")
        _positive(self.storage_floor_bytes_free, "storage floor", integer=True)
        ob._sha(self.expected_source_digest, "window source digest")
        for package, energy, maximum in self.package_energy_paths:
            if type(package) is not int or package < 0 or not all(
                    isinstance(path, str) and Path(path).is_absolute() for path in (energy, maximum)):
                raise SearchWindowRefused("package energy paths require explicit original identities")
        if len({row[0] for row in self.package_energy_paths}) != len(self.package_energy_paths):
            raise SearchWindowRefused("package energy identity occurs more than once")
        if not isinstance(self.package_energy_paths, tuple) or any(
                not isinstance(row, tuple) for row in self.package_energy_paths):
            raise SearchWindowRefused("package source configuration must be immutable tuples")
        for name in ("calibration_material_ref", "control_material_ref"):
            value = getattr(self, name)
            if value is not None:
                ob._text(value, name)

    @classmethod
    def from_dict(cls, value: Any) -> "InstalledSearchWindowConfiguration":
        if not isinstance(value, Mapping) or set(value) != {field.name for field in fields(cls)}:
            raise SearchWindowRefused("window configuration has missing or unknown fields")
        row = ob._plain(value)
        for name, kind in (("limits", WindowLimits), ("host_policy", microbench.HostStatePolicy)):
            supplied = row[name]
            if not isinstance(supplied, dict) or set(supplied) != {field.name for field in fields(kind)}:
                raise SearchWindowRefused(f"window {name} has missing or unknown fields")
            try:
                row[name] = kind(**supplied)
            except (TypeError, ValueError) as exc:
                raise SearchWindowRefused(f"invalid window {name}") from exc
        if not isinstance(row["cpu_regions"], list) or not isinstance(row["package_energy_paths"], list):
            raise SearchWindowRefused("window regions/package paths must be explicit arrays")
        if any(not isinstance(item, list) or len(item) != 3 for item in row["package_energy_paths"]):
            raise SearchWindowRefused("window package source must have exact three-field rows")
        row["cpu_regions"] = tuple(row["cpu_regions"])
        row["package_energy_paths"] = tuple(tuple(item) for item in row["package_energy_paths"])
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return ob._plain(asdict(self))

    @property
    def digest(self) -> str:
        return lo._digest(self.to_dict())

    def preparation_requirements(self) -> tuple[str, ...]:
        requirements = []
        if self.calibration_material_ref is None:
            requirements.append("original_serving_cell_calibration_material")
        if self.control_material_ref is None:
            requirements.append("original_serving_cell_executed_control_material")
        if self.host_policy.nominal_khz is None:
            requirements.append("cell_healthy_nominal_frequency")
        return tuple(requirements)


def _metadata(info: os.stat_result) -> dict[str, int]:
    return {name: getattr(info, name) for name in (
        "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")}


def read_raw_file(path: Path, maximum: int) -> dict[str, Any]:
    """Read bounded EOF, including procfs size-zero files; never open a FIFO."""
    started = time.monotonic()
    row: dict[str, Any] = {"path": str(path), "started": started, "ended": None,
        "before": None, "after": None, "data_base64": None, "sha256": None, "error": None}
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            row["before"] = _metadata(before)
            if not stat.S_ISREG(before.st_mode):
                raise SearchWindowRefused("observation source is not a regular file")
            if before.st_size > maximum:
                raise SearchWindowRefused("observation byte bound exceeded")
            chunks, size = [], 0
            while True:
                chunk = os.read(fd, min(64 * 1024, maximum + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > maximum:
                    raise SearchWindowRefused("observation byte bound exceeded")
            after = os.fstat(fd)
            row["after"] = _metadata(after)
        finally:
            os.close(fd)
        named = os.stat(path, follow_symlinks=False)
        if ((before.st_dev, before.st_ino, before.st_mode)
                != (after.st_dev, after.st_ino, after.st_mode)
                or (after.st_dev, after.st_ino, after.st_mode)
                != (named.st_dev, named.st_ino, named.st_mode)):
            raise SearchWindowRefused("observation source identity changed while reading")
        raw = b"".join(chunks)
        row.update(data_base64=base64.b64encode(raw).decode("ascii"),
                   sha256=hashlib.sha256(raw).hexdigest())
    except (OSError, SearchWindowRefused) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["ended"] = time.monotonic()
    return row


def _raw(row: Mapping[str, Any] | None) -> bytes | None:
    if row is None or row["error"] is not None:
        return None
    raw = base64.b64decode(row["data_base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise SearchWindowRefused("captured bytes differ from original digest")
    return raw


def _text(row: Mapping[str, Any] | None) -> str | None:
    raw = _raw(row)
    return None if raw is None else raw.decode("utf-8", "replace")


def _cgroup(text: str | None) -> str | None:
    if text is None:
        return None
    lines = text.splitlines()
    return next((line[3:] for line in lines if line.startswith("0::")), lines[0] if lines else None)


class BoundedWindowSnapshot:
    """One original snapshot, advanced one bounded operation at a time."""

    def __init__(self, config: InstalledSearchWindowConfiguration, cpus: tuple[int, ...],
                 *, census: bool, marker: str) -> None:
        self.config, self.cpus, self.marker = config, cpus, marker
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.deadline = self.started + (config.limits.max_boundary_duration_s if census
                                       else config.limits.max_sample_duration_s)
        self.ended: float | None = None
        self.ended_at: str | None = None
        self.files: dict[str, Any] = {}
        self.errors: list[str] = []
        self.directories: dict[str, Any] = {}
        self.storage: dict[str, Any] | None = None
        self._bytes = 0
        self._jobs: deque[tuple[str, str, Path]] = deque()
        proc, sysfs = Path(config.proc_root), Path(config.sysfs_cpu_root)
        for name in ("loadavg", "uptime"):
            self._jobs.append(("file", name, proc / name))
        for cpu in cpus:
            for field in ("cpufreq/scaling_cur_freq", "topology/physical_package_id"):
                self._jobs.append(("file", f"cpu/{cpu}/{field}", sysfs / f"cpu{cpu}" / field))
        for field in ("cpuinfo_min_freq", "cpuinfo_max_freq"):
            self._jobs.append(("file", field, sysfs / f"cpu{cpus[0]}" / "cpufreq" / field))
        for package, energy, maximum in config.package_energy_paths:
            self._jobs.append(("file", f"energy/{package}", Path(energy)))
            self._jobs.append(("file", f"energy_max/{package}", Path(maximum)))
        if census:
            self._jobs.extend((("directory", "processes", proc),
                ("directory", "claims", Path(config.claim_root)),
                ("file", "locks", proc / "locks"),
                ("storage", "storage", Path(config.storage_root))))

    @property
    def complete(self) -> bool:
        return not self._jobs and self.ended is not None

    def cancel(self, reason: str) -> None:
        if reason not in self.errors:
            self.errors.append(reason)
        self._jobs.clear()
        self.ended = time.monotonic()
        self.ended_at = datetime.now(timezone.utc).isoformat()

    def step(self) -> None:
        if not self._jobs:
            return
        if time.monotonic() >= self.deadline:
            self.cancel("snapshot duration exceeded")
            return
        kind, key, path = self._jobs.popleft()
        try:
            if kind == "file":
                row = read_raw_file(path, self.config.limits.max_read_bytes)
                raw = _raw(row)
                self._bytes += 0 if raw is None else len(raw)
                self.files[key] = row
                if self._bytes > self.config.limits.max_snapshot_bytes:
                    self.cancel("snapshot byte budget exhausted")
            elif kind == "storage":
                before = time.monotonic()
                value = os.statvfs(path)
                self.storage = {"path": str(path), "started": before,
                    "ended": time.monotonic(), "free_bytes": value.f_bavail * value.f_frsize,
                    "floor_bytes": self.config.storage_floor_bytes_free}
            else:
                entries = lo._bounded_entries(path, self.config.limits.max_directory_entries, key)
                names = sorted(item.name for item in entries)
                self.directories[key] = {"path": str(path), "names": names}
                if key == "processes":
                    selected = sorted((int(name) for name in names if name.isdigit()))
                    if len(selected) > self.config.limits.max_processes:
                        raise SearchWindowRefused("process census bound exhausted")
                    for pid in selected:
                        for field in ("stat", "cmdline", "cgroup"):
                            self._jobs.append(("file", f"pid/{pid}/{field}", path / str(pid) / field))
                else:
                    for name in names:
                        if fnmatch.fnmatchcase(name, preflight._LOCK_GLOB):
                            self._jobs.append(("file", f"claim/{name}", path / name))
        except (OSError, lo.ObservationError, SearchWindowRefused) as exc:
            self.errors.append(f"{key}: {type(exc).__name__}: {exc}")
        if time.monotonic() > self.deadline:
            self.cancel("snapshot duration exceeded")
        if not self._jobs:
            self.ended = time.monotonic()
            self.ended_at = datetime.now(timezone.utc).isoformat()

    def body(self) -> dict[str, Any]:
        return {"marker": self.marker, "started": self.started, "ended": self.ended,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "files": self.files, "directories": self.directories, "storage": self.storage,
            "errors": self.errors, "complete": self.complete}


def host_state(snapshot: Mapping[str, Any], cpus: tuple[int, ...]) -> microbench.HostState:
    files = snapshot["files"]
    unreadable: list[str] = []
    def integer(key: str) -> int | None:
        value = _text(files.get(key))
        try:
            return int(value.strip()) if value is not None else None
        except ValueError:
            return None
    frequencies, packages, energies = [], [], []
    for cpu in cpus:
        frequency = integer(f"cpu/{cpu}/cpufreq/scaling_cur_freq")
        if frequency is None:
            unreadable.append(f"cpu{cpu}: scaling_cur_freq unreadable")
        else:
            frequencies.append((cpu, frequency))
        package = integer(f"cpu/{cpu}/topology/physical_package_id")
        if package is None:
            unreadable.append(f"cpu{cpu}: physical_package_id unreadable")
        else:
            packages.append((cpu, package))
    for package in sorted({item[1] for item in packages}):
        energy, maximum = integer(f"energy/{package}"), integer(f"energy_max/{package}")
        if energy is not None and maximum is not None:
            energies.append((package, energy, maximum, files[f"energy/{package}"]["path"]))
    def number(key: str) -> float | None:
        value = _text(files.get(key))
        try:
            parsed = float(value.split()[0]) if value else None
            return parsed if parsed is not None and math.isfinite(parsed) and parsed >= 0 else None
        except (ValueError, IndexError):
            return None
    return microbench.HostState(observed_at=snapshot["ended_at"],
        cpu_list=",".join(str(cpu) for cpu in cpus), khz_by_cpu=tuple(frequencies),
        driver_min_khz=integer("cpuinfo_min_freq"), driver_max_khz=integer("cpuinfo_max_freq"),
        load1=number("loadavg"), source="original-paths-in-sealed-window-snapshot",
        unreadable=tuple(unreadable), uptime_s=number("uptime"),
        monotonic_s=snapshot["ended"], package_by_cpu=tuple(packages),
        package_energy_uj=tuple(energies))


def captured_preflight(snapshot: Mapping[str, Any], *, self_pid: int,
                       scope: preflight.PreflightScope) -> preflight.PreflightResult:
    """Original predicates over exact original dev/inode and process bytes."""
    files, directories = snapshot["files"], snapshot["directories"]
    process_rows = {}
    for name in directories.get("processes", {}).get("names", ()):
        if name.isdigit():
            row = files.get(f"pid/{int(name)}/stat")
            process_rows[int(name)] = (_text(row), None if row is None else row["error"])
    try:
        owned = preflight.reduce_owned_scope(self_pid=self_pid, ancestor_stats=process_rows,
            pid_stats=process_rows, cgroup=_cgroup(_text(files.get(f"pid/{self_pid}/cgroup"))))
        ownership_error = None
    except preflight.PreflightUnavailable as exc:
        owned, ownership_error = None, str(exc)
    claims, descriptions = [], {}
    region_error = None
    names = directories.get("claims", {}).get("names")
    lock_text = _text(files.get("locks"))
    if snapshot["errors"] or not snapshot["complete"]:
        region_error = "captured claim census is incomplete: " + "; ".join(snapshot["errors"])
    elif names is None or not any(fnmatch.fnmatchcase(name, preflight._LOCK_GLOB) for name in names):
        region_error = "captured region namespace is missing or empty"
    elif lock_text is None:
        region_error = "original /proc/locks capture is unavailable"
    else:
        locks = preflight.parse_proc_locks(lock_text)
        for name in names:
            if not fnmatch.fnmatchcase(name, preflight._LOCK_GLOB):
                continue
            row = files.get(f"claim/{name}")
            if row is None or row["error"] is not None:
                region_error = "one or more original claim files is unreadable"
                break
            stem = name[len(preflight._LOCK_PREFIX):-len(preflight._LOCK_SUFFIX)]
            role, _, region = stem.rpartition(".")
            if not role or not region:
                continue
            info = row["before"]
            key = (f"{os.major(info['st_dev']):02x}:{os.minor(info['st_dev']):02x}", info["st_ino"])
            holders = locks.get(key, preflight.LockHolders())
            claims.append(preflight.parse_region_claim(role=role, region=region,
                lock_path=row["path"], holders=holders, raw=_text(row)))
            for pid in holders.holder_pids:
                raw, error = process_rows.get(pid, (None, None))
                description: dict[str, Any] = {"pid": pid}
                if error is not None:
                    description["unreadable"] = error
                elif raw is None:
                    description["vanished"] = True
                else:
                    parsed = preflight._parse_stat(raw)
                    if parsed is not None:
                        for field in ("comm", "ppid", "starttime_ticks"):
                            description[field] = parsed[field]
                    argv = _raw(files.get(f"pid/{pid}/cmdline"))
                    if argv:
                        first = argv.split(b"\0")[0].decode("utf-8", "replace")
                        description.update(argv0=first, argv0_basename=os.path.basename(first))
                    cgroup = _cgroup(_text(files.get(f"pid/{pid}/cgroup")))
                    if cgroup:
                        description["cgroup"] = cgroup
                descriptions[pid] = description
        if not claims and region_error is None:
            region_error = "captured region namespace naming contract is unparseable"
    material = preflight.CapturedClaimWitness(owned, ownership_error=ownership_error,
        region_claims=tuple(claims), region_error=region_error,
        holder_descriptions=descriptions)
    # No GPU claim is invented from a device's free-form purpose or CPU flock.
    # The exact installed GPU claim witness remains an explicit prerequisite.
    return preflight.reduce_claim_witness(scope, material, observed_at=snapshot["ended_at"])


def lifecycle_window_facts(observation: Mapping[str, Any], markers: Mapping[str, Any],
                           samples: Any, limits: WindowLimits) -> Mapping[str, Any]:
    """Join original during-work observations, not endpoint absence or argv purpose.

    This is a factual coverage reduction, never a no-contention/purpose verdict.
    The original lifecycle interval reducer owns CPU/census/memory semantics.
    """
    observation = lo.validate_observation(ob._plain(observation))
    intervals, interval_issues = lo._intervals(observation["samples"],
        set(observation["held_claim"]["physical_cpus"] or ()), observation["gap_limit_s"])
    if intervals != observation["intervals"]:
        raise SearchWindowRefused("lifecycle intervals do not rederive from original samples")
    reasons: list[str] = []
    for phase in ("health", "warmup", "measurement", "teardown"):
        actual = [row["monotonic_s"] for row in observation["phase_boundaries"]
                  if row["phase"] == phase]
        packet = markers.get(phase)
        if packet is None or actual != [packet["boundary_monotonic_s"]]:
            reasons.append(f"{phase} parent/lifecycle marker join unavailable")
    end = markers.get("measurement_end")
    checkpoints = [row["marker_monotonic_s"] for row in observation["samples"]
                   if row["phase"] == "measurement" and row["marker_label"] == "measurement_end"]
    if end is None or checkpoints != [end["boundary_monotonic_s"]]:
        reasons.append("measurement_end original checkpoint join unavailable")
    phases = {}
    for phase, next_marker in (("warmup", "measurement"), ("measurement", "measurement_end")):
        first, last = markers.get(phase), markers.get(next_marker)
        labelled = [row for row in intervals if row["phase"] == phase]
        rows = []
        phase_reasons = []
        if first is None or last is None:
            phase_reasons.append("phase interval markers unavailable")
            inside = []
        else:
            start, finish = first["boundary_monotonic_s"], last["boundary_monotonic_s"]
            rows = [row for row in labelled if start <= row["start_monotonic_s"]
                    <= row["end_monotonic_s"] <= finish]
            inside = [row for row in observation["samples"] if row["phase"] == phase
                and start <= row["read_started_monotonic_s"] <= row["read_ended_monotonic_s"] <= finish]
            if finish <= start or len(inside) < 2:
                phase_reasons.append("fewer than two actual during-phase census observations")
            if inside and (inside[0]["read_started_monotonic_s"] - start > observation["gap_limit_s"]
                    or finish - inside[-1]["read_ended_monotonic_s"] > observation["gap_limit_s"]):
                phase_reasons.append("uncovered phase endpoint gap")
        if not rows or any(row["status"] != "observed" for row in rows):
            phase_reasons.append("original lifecycle interval coverage unknown")
        if any(row["status"] != "observed" or row["probe_over_budget"] for row in inside):
            phase_reasons.append("original during-phase probe unknown or over budget")
        overlaps = [row for interval in rows for row in interval["potential_foreign_overlap"]]
        phases[phase] = {"coverage": "unknown" if phase_reasons else "observed",
            "reasons": phase_reasons, "during_sample_count": len(inside),
            "intervals": rows, "potential_foreign_overlap": overlaps,
            "boundary_crossing_intervals": [row for row in labelled if row not in rows],
            "purpose": "unknown; owning classification required"}
    begin = markers.get("measurement")
    host = [] if begin is None or end is None else [ob._plain(row) for row in samples
        if begin["boundary_monotonic_s"] <= row["started"] <= row["ended"] <= end["boundary_monotonic_s"]
        and row["complete"] and not row["errors"]]
    host_reasons = []
    if len(host) < 2:
        host_reasons.append("fewer than two complete during-measurement host samples")
    if begin is not None and end is not None and host:
        gaps = [host[0]["started"] - begin["boundary_monotonic_s"],
                end["boundary_monotonic_s"] - host[-1]["ended"],
                *(right["started"] - left["ended"] for left, right in zip(host, host[1:]))]
        if any(gap < 0 or gap > limits.max_gap_s for gap in gaps):
            host_reasons.append("host sample coverage gap")
    overall = (not reasons and not host_reasons
               and all(row["coverage"] == "observed" for row in phases.values()))
    return ob._freeze({"coverage": "observed" if overall else "unknown",
        "marker_join": "unknown" if reasons else "observed",
        "marker_join_reasons": reasons, "phases": phases,
        "original_interval_issues": interval_issues,
        "host_measurement_coverage": "unknown" if host_reasons else "observed",
        "host_measurement_reasons": host_reasons, "host_during_sample_count": len(host),
        "no_concurrent_work": "unknown; coverage is not a scientific clean-window verdict"})


@dataclass(frozen=True)
class SearchWindowReceipt:
    artifact: mc.StoredArtifact
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact.to_dict(), "digest": self.digest}


class SameAttemptWindowOwner:
    """One actual parent thread, original attempt identities, no replay recapture."""

    def __init__(self, *, config: InstalledSearchWindowConfiguration, prepared: Any,
                 lifecycle: Any, store: mc.ArtifactStore) -> None:
        from . import unified_worker as uw
        from . import worker_lifecycle as wl
        if (type(config) is not InstalledSearchWindowConfiguration
                or type(prepared) is not uw.PreparedPlannedServingStage
                or not prepared.native_observed
                or not isinstance(lifecycle, wl.WorkerLifecycle)):
            raise SearchWindowRefused("window owner needs actual v2 parent lifecycle")
        if config.expected_source_digest != source_digest():
            raise SearchWindowRefused("loaded window source differs from installed pin")
        reference = ob.LoadedInstrumentReference.from_dict(ob._plain(prepared.plan.loaded_instrument))
        instrument = ob._plain(store.read(reference.artifact.locator, reference.artifact.sha256))
        if instrument["used_constants"].get("search_window_configuration") != config.to_dict():
            raise SearchWindowRefused("window configuration was not pinned before plan issuance")
        if instrument["used_constants"].get("search_window_source") != ob._plain(source_identity()):
            raise SearchWindowRefused("window reducer/collector source was not prospectively pinned")
        self.config, self.prepared, self.lifecycle, self.store = config, prepared, lifecycle, store
        self._thread: threading.Thread | None = None
        self._units: dict[str, dict[str, Any]] = {}
        self._current: str | None = None
        self._pending: BoundedWindowSnapshot | None = None
        self._next_sample = 0.0
        self._receipts: dict[str, SearchWindowReceipt] = {}

    def _owned_thread(self) -> None:
        actual = threading.current_thread()
        if self._thread is None:
            self._thread = actual
        elif actual is not self._thread:
            raise SearchWindowRefused("window collector moved off its actual owning thread")

    def _boundary(self, row: Mapping[str, Any], marker: str) -> Mapping[str, Any]:
        snapshot = BoundedWindowSnapshot(self.config, tuple(row["binding"].held_claim["logical_cpus"]),
                                        census=True, marker=marker)
        deadline = min(row["start"].provider_deadline,
            time.monotonic() + self.config.limits.max_boundary_duration_s)
        while not snapshot.complete:
            if time.monotonic() >= deadline:
                snapshot.cancel("boundary observation deadline exceeded")
                break
            snapshot.step()
            if time.monotonic() > deadline:
                snapshot.cancel("boundary observation deadline exceeded")
                break
        return ob._freeze(ob._plain(snapshot.body()))

    def open_unit(self, *, start: Any, binding: ob.ObservationUnitBinding,
                  claim: Mapping[str, Any]) -> None:
        self._owned_thread()
        original = {"start": start.to_dict(), "binding": binding.to_dict(), "claim": ob._plain(claim)}
        prior = self._units.get(binding.unit_id)
        if prior is not None:
            if ob._plain(prior["original"]) != original:
                raise SearchWindowRefused("window open retry changed original identity")
            return
        if self._current is not None:
            raise SearchWindowRefused("prior window has not closed")
        if binding.unit_id not in {unit.unit_id for unit in self.prepared.plan.expected_units}:
            raise SearchWindowRefused("window unit is absent from original plan")
        if len(self._units) >= len(self.prepared.plan.expected_units):
            raise SearchWindowRefused("window unit retention bound exhausted")
        row: dict[str, Any] = {"start": start, "binding": binding,
            "original": ob._freeze(original), "markers": {}, "samples": [], "gaps": [],
            "closed": False, "target": None}
        row["open"] = self._boundary(row, "open")
        self._units[binding.unit_id] = row
        self._current = binding.unit_id

    def marker(self, *, context: Any, packet: Mapping[str, Any]) -> None:
        self._owned_thread()
        row = self._units.get(context.unit_id)
        if row is None or row["binding"] != context.binding or row["start"].nonce != context.nonce:
            raise SearchWindowRefused("marker lacks its original parent window")
        marker = packet["phase"]
        prior = row["markers"].get(marker)
        frozen = ob._freeze(ob._plain(packet))
        if prior is not None:
            if prior != frozen:
                raise SearchWindowRefused("window marker retry differs")
            return
        if row["closed"]:
            raise SearchWindowRefused("window is already closed")
        row["markers"][marker] = frozen
        row["target"] = ob._freeze(ob._plain(context.descendant_event))
        if marker == "measurement":
            self._next_sample = time.monotonic() + self.config.limits.cadence_s
        if marker in ("measurement_end", "teardown"):
            if self._pending is not None:
                self._pending.cancel("measurement closed during sample capture")
                row["samples"].append(ob._freeze(ob._plain(self._pending.body())))
                self._pending = None
        if marker == "teardown":
            row["close"] = self._boundary(row, "close")
            try:
                row["close_claim"] = self.lifecycle.describe_active_observation_claim(
                    start=row["start"], unit_id=context.unit_id,
                    process_generation_id=context.unit.process_id,
                    deadline=context.fence.valid_until)
                row["close_claim_error"] = None
            except Exception as exc:
                row["close_claim"], row["close_claim_error"] = None, f"{type(exc).__name__}: {exc}"
            row["closed"] = True
            self._current = None

    def poll_delay(self) -> float:
        if self._current is None:
            return 0.05
        row = self._units[self._current]
        if "measurement" not in row["markers"] or "measurement_end" in row["markers"]:
            return 0.05
        return 0.0 if self._pending is not None else min(0.05, max(0.0, self._next_sample - time.monotonic()))

    def poll(self) -> None:
        self._owned_thread()
        if self._current is None:
            return
        row = self._units[self._current]
        if "measurement" not in row["markers"] or "measurement_end" in row["markers"]:
            return
        now = time.monotonic()
        if self._pending is None:
            if now < self._next_sample:
                return
            if len(row["samples"]) >= self.config.limits.max_samples:
                if "sample count bound exhausted" not in row["gaps"]:
                    row["gaps"].append("sample count bound exhausted")
                self._next_sample = now + self.config.limits.cadence_s
                return
            if now - self._next_sample > self.config.limits.max_gap_s:
                row["gaps"].append("scheduled measurement sample was late")
            self._pending = BoundedWindowSnapshot(self.config,
                tuple(row["binding"].held_claim["logical_cpus"]), census=False, marker="measurement")
        pending = self._pending
        if now - pending.started > self.config.limits.max_sample_duration_s:
            pending.cancel("sample duration exceeded")
        else:
            pending.step()
            if time.monotonic() - pending.started > self.config.limits.max_sample_duration_s:
                pending.cancel("sample duration exceeded")
        if pending.complete:
            row["samples"].append(ob._freeze(ob._plain(pending.body())))
            self._pending = None
            self._next_sample = time.monotonic() + self.config.limits.cadence_s

    def seal(self, *, context: Any, native: Mapping[str, Any],
             parent_result: Any) -> SearchWindowReceipt:
        self._owned_thread()
        from .native_parent_evidence import NativeUnitEvidenceResult
        if type(parent_result) is not NativeUnitEvidenceResult:
            raise SearchWindowRefused("window completion requires the actual parent factual result")
        parent_body = ob._plain(self.store.read(
            parent_result.receipt.locator, parent_result.receipt.sha256))
        if (lo._digest(parent_body) != parent_result.receipt_digest
                or parent_body["identity"] != context.identity
                or parent_body["lifecycle_observation"] != ob._plain(native["lifecycle_observation"])):
            raise SearchWindowRefused("parent factual result differs from original window")
        prior = self._receipts.get(context.unit_id)
        if prior is not None:
            body = self.store.read(prior.artifact.locator, prior.artifact.sha256)
            if body["native_artifact_digest"] != native["artifact_digest"]:
                raise SearchWindowRefused("window seal retry changes native result")
            return prior
        row = self._units.get(context.unit_id)
        if row is None:
            raise SearchWindowRefused("native completion lacks its original window")
        facts = {key: ob._plain(row.get(key)) for key in (
            "original", "markers", "open", "close", "close_claim", "close_claim_error",
            "samples", "gaps", "closed", "target")}
        reference = ob.LifecycleObservationReference.from_dict(ob._plain(native["lifecycle_observation"]))
        ob.validate_reopened_observation(reference, store=self.store,
            expected={"unit_id": context.unit_id, "process_generation_id": context.unit.process_id,
                "fence_id": context.fence.fence_id, "active_claim_ref": context.binding.active_claim_ref,
                "container_id": context.binding.container_id,
                "capture_context": ob._plain(context.binding.worker_binding)},
            instrument=context.binding.instrument)
        observation = self.store.read(reference.artifact.locator, reference.artifact.sha256)
        lifecycle_facts = lifecycle_window_facts(observation, row["markers"], row["samples"], self.config.limits)
        body = {"schema": RECEIPT_SCHEMA, "configuration": self.config.to_dict(),
            "source": ob._plain(source_identity()), "plan_digest": self.prepared.plan.digest,
            "native_artifact_digest": native["artifact_digest"], "identity": context.identity,
            "parent_factual_receipt": parent_result.receipt.to_dict(),
            "lifecycle_observation": reference.to_dict(), "lifecycle_facts": ob._plain(lifecycle_facts),
            "facts": facts, "preparation_requirements": list(self.config.preparation_requirements()),
            "scientific_authority": "none; owning search/control/gate material required"}
        digest = lo._digest(body)
        reference = self.store.write(f"search-window:{digest}", body)
        receipt = SearchWindowReceipt(reference, digest)
        self._receipts[context.unit_id] = receipt
        return receipt

    def reopen(self, receipt: SearchWindowReceipt, *, context: Any,
               native: Mapping[str, Any]) -> Mapping[str, Any]:
        if type(receipt) is not SearchWindowReceipt or self._receipts.get(context.unit_id) != receipt:
            raise SearchWindowRefused("window receipt was not issued by this actual attempt owner")
        # Completion shuts the service's descriptor-bound store. Reopen the
        # exact original root for replay; issuance remains this owner's token.
        store = mc.ArtifactStore(self.prepared.artifact_root)
        try:
            body = ob._plain(store.read(receipt.artifact.locator, receipt.artifact.sha256))
        finally:
            store.close()
        if (lo._digest(body) != receipt.digest or body["identity"] != context.identity
                or body["native_artifact_digest"] != native["artifact_digest"]):
            raise SearchWindowRefused("window receipt identity differs")
        return ob._freeze(ob._plain(body))


def source_identity() -> Mapping[str, Any]:
    functions = (read_raw_file, _raw, _text, _cgroup, host_state, captured_preflight,
        lifecycle_window_facts, lo._intervals, lo.validate_observation,
        ob.validate_reopened_observation,
        _metadata, InstalledSearchWindowConfiguration.__post_init__,
        InstalledSearchWindowConfiguration.from_dict, WindowLimits.__post_init__,
        BoundedWindowSnapshot.__init__, BoundedWindowSnapshot.step, BoundedWindowSnapshot.cancel,
        SameAttemptWindowOwner.__init__, SameAttemptWindowOwner._owned_thread,
        SameAttemptWindowOwner._boundary, SameAttemptWindowOwner.poll_delay,
        SameAttemptWindowOwner.open_unit, SameAttemptWindowOwner.marker,
        SameAttemptWindowOwner.poll, SameAttemptWindowOwner.seal, SameAttemptWindowOwner.reopen,
        preflight.reduce_claim_witness, preflight.reduce_owned_scope,
        preflight.parse_proc_locks, preflight.parse_region_claim,
        preflight._parse_stat, preflight.combine_verdicts,
        microbench.HostStatePolicy.frequency_verdict, microbench.HostStatePolicy.check_load,
        microbench.derive_package_power_attestation)
    return ob._freeze({"schema": "epyc.autokernel.search_window_source.v1",
        "callables": [lo.callable_identity(function) for function in functions]})


def source_digest() -> str:
    return lo._digest(ob._plain(source_identity()))
