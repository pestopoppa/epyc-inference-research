"""Fail-closed ROCm device-state sampling over an exact measurement window.

The evaluator already parses and grades numeric device-state samples.  This
module is the producer that was missing: it samples the numeric amdgpu hwmon
counters exposed beneath ROCm SMI on an absolute 250 ms schedule while the
benchmark process launched by the trusted runner is alive.  It never searches
for a process and never signals one; the caller owns the measurement-process
handle and brackets this sampler around that handle's lifetime.  The logical
command retained in the receipt names the ROCm SMI fields being sampled, while
``source`` records the actual counter transport.

Sampling failures are material.  A missing field, a failed snapshot read, an
empty window, or a cadence gap greater than twice the declared interval refuses
the receipt instead of silently returning a partial trace.  The direct hwmon
transport is required on this host because both a fresh ``rocm-smi`` process
and persistent ``librocm_smi64`` reads stall for about one second under a
saturating GEMM and therefore cannot truthfully provide 250 ms samples.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import threading
import time
from ctypes import byref, c_int64, c_uint32
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..evaluator import devices


MODULE_ID = "autokernel.execution.device_sampler/v1"
DEFAULT_ROCM_SMI = "/opt/rocm/bin/rocm-smi"
DEFAULT_INTERVAL_S = 0.250
ROCM_SMI_BINDINGS_DIR = "/opt/rocm/libexec/rocm_smi"
MI210_NOMINAL_SCLK_MHZ = 1700.0
MI210_MIN_SCLK_RATIO = 0.90


class DeviceSamplingError(RuntimeError):
    """The required in-window device trace could not be produced."""


@dataclass(frozen=True)
class SnapshotResult:
    """One numeric snapshot result, injectable for deterministic tests."""

    argv: tuple
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TimedDeviceStateSample:
    """A parsed state sample and its position inside the measured window."""

    offset_s: float
    sample: devices.DeviceStateSample

    def __post_init__(self) -> None:
        if isinstance(self.offset_s, bool) or not isinstance(self.offset_s, (int, float)) \
                or self.offset_s < 0:
            raise ValueError("device sample offset_s must be non-negative")
        if not isinstance(self.sample, devices.DeviceStateSample):
            raise TypeError("device sample must be DeviceStateSample")

    def to_dict(self) -> dict:
        return {"offset_s": float(self.offset_s), **self.sample.to_dict()}


@dataclass(frozen=True)
class DeviceSamplingReceipt:
    """The complete numeric trace for one exact benchmark-process lifetime."""

    sampler_id: str
    device_id: str
    source: str
    started_at: str
    ended_at: str
    interval_s: float
    duration_s: float
    command: tuple
    samples: tuple

    def __post_init__(self) -> None:
        for name in ("sampler_id", "device_id", "source", "started_at", "ended_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"device sampling receipt {name} must be non-empty")
        if self.interval_s <= 0 or self.duration_s < 0:
            raise ValueError("device sampling interval must be positive and duration non-negative")
        if not self.command or not all(isinstance(token, str) and token for token in self.command):
            raise ValueError("device sampling command must be a non-empty argv tuple")
        if not self.samples or not all(
                isinstance(row, TimedDeviceStateSample) for row in self.samples):
            raise DeviceSamplingError(
                "the measurement window contains no parsed device-state samples")
        offsets = tuple(row.offset_s for row in self.samples)
        if offsets != tuple(sorted(offsets)):
            raise DeviceSamplingError("device-state sample offsets are not monotonic")
        # One slow/missed call may not be hidden by reporting the requested
        # interval.  Twice the interval is a deliberately explicit admission
        # ceiling; exceeding it makes this a partial trace, not a 250 ms trace.
        gaps = tuple(b - a for a, b in zip(offsets, offsets[1:]))
        if gaps and max(gaps) > 2.0 * self.interval_s:
            raise DeviceSamplingError(
                f"device-state sampling cadence gap {max(gaps):.6f}s exceeds the "
                f"declared ceiling {2.0 * self.interval_s:.6f}s")

    @property
    def max_gap_s(self) -> float:
        offsets = tuple(row.offset_s for row in self.samples)
        return max((b - a for a, b in zip(offsets, offsets[1:])), default=0.0)

    def to_dict(self) -> dict:
        payload = {
            "schema": "epyc.autokernel.device_sampling_receipt.v1",
            "sampler_id": self.sampler_id,
            "device_id": self.device_id,
            "source": self.source,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "interval_s": float(self.interval_s),
            "duration_s": float(self.duration_s),
            "command": list(self.command),
            "sample_count": len(self.samples),
            "max_gap_s": self.max_gap_s,
            "samples": [row.to_dict() for row in self.samples],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    def device_state(self, *, nominal_sclk_mhz: float,
                     min_sclk_ratio: float) -> devices.DeviceState:
        payload = self.to_dict()
        return devices.DeviceState(
            device_id=self.device_id,
            source=self.source,
            nominal_sclk_mhz=nominal_sclk_mhz,
            min_sclk_ratio=min_sclk_ratio,
            samples=tuple(row.sample for row in self.samples),
            receipt_ref=f"sha256:{payload['sha256']}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_snapshot(argv: Sequence[str], timeout_s: float) -> SnapshotResult:
    """Run only the named read-only probe and capture its small text response."""
    try:
        completed = subprocess.run(
            list(argv), stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DeviceSamplingError(f"rocm-smi snapshot failed: {exc}") from exc
    return SnapshotResult(
        argv=tuple(str(token) for token in argv), returncode=completed.returncode,
        stdout=completed.stdout, stderr=completed.stderr)


class RocmSmiLibrarySnapshotRunner:
    """Persistent ``librocm_smi64`` reader, avoiding one CLI startup per sample.

    ``rocm-smi`` itself uses these installed bindings.  Keeping the library
    initialized across the window is necessary on this host: a fresh CLI under
    GEMM load took about one second, so a process-per-sample implementation
    correctly failed its own 250 ms cadence gate.
    """

    source = "rocm-smi-lib/numeric/v1"

    def __init__(self, *, bindings_dir: str = ROCM_SMI_BINDINGS_DIR) -> None:
        if bindings_dir not in sys.path:
            sys.path.insert(0, bindings_dir)
        try:
            self._bindings = importlib.import_module("rsmiBindings")
            self._library = self._bindings.initRsmiBindings(silent=True)
        except (ImportError, OSError) as exc:
            raise DeviceSamplingError(
                f"could not initialize installed ROCm SMI bindings: {exc}") from exc
        status = self._library.rsmi_init(0)
        if status != self._bindings.rsmi_status_t.RSMI_STATUS_SUCCESS:
            raise DeviceSamplingError(f"rsmi_init failed with status {status}")

    def _frequency_mhz(self, device_index: int, clock_type: int) -> float:
        frequencies = self._bindings.rsmi_frequencies_t()
        status = self._library.rsmi_dev_gpu_clk_freq_get(
            c_uint32(device_index), clock_type, byref(frequencies))
        if status != self._bindings.rsmi_status_t.RSMI_STATUS_SUCCESS:
            raise DeviceSamplingError(
                f"rsmi_dev_gpu_clk_freq_get({clock_type}) failed with status {status}")
        if frequencies.current >= frequencies.num_supported:
            raise DeviceSamplingError(
                f"ROCm SMI returned clock index {frequencies.current} outside "
                f"{frequencies.num_supported} supported levels")
        return frequencies.frequency[frequencies.current] / 1_000_000.0

    def __call__(self, argv: Sequence[str], timeout_s: float) -> SnapshotResult:
        del timeout_s  # Direct library calls carry no subprocess timeout.
        try:
            device_index = int(argv[argv.index("-d") + 1])
        except (ValueError, IndexError) as exc:
            raise DeviceSamplingError("ROCm SMI logical argv has no device index") from exc
        sclk_mhz = self._frequency_mhz(
            device_index, self._bindings.rsmi_clk_type_t.RSMI_CLK_TYPE_SYS)
        mclk_mhz = self._frequency_mhz(
            device_index, self._bindings.rsmi_clk_type_t.RSMI_CLK_TYPE_MEM)
        power = c_int64(0)
        power_type = self._bindings.rsmi_power_type_t()
        status = self._library.rsmi_dev_power_get(
            c_uint32(device_index), byref(power), byref(power_type))
        if status != self._bindings.rsmi_status_t.RSMI_STATUS_SUCCESS:
            raise DeviceSamplingError(f"rsmi_dev_power_get failed with status {status}")
        temperature = c_int64(0)
        status = self._library.rsmi_dev_temp_metric_get(
            c_uint32(device_index),
            self._bindings.rsmi_temperature_type_t.RSMI_TEMP_TYPE_JUNCTION,
            self._bindings.rsmi_temperature_metric_t.RSMI_TEMP_CURRENT,
            byref(temperature))
        if status != self._bindings.rsmi_status_t.RSMI_STATUS_SUCCESS:
            raise DeviceSamplingError(
                f"rsmi_dev_temp_metric_get(junction) failed with status {status}")
        text = (
            f"GPU[{device_index}] : Temperature (Sensor junction) (C): "
            f"{temperature.value / 1000.0}\n"
            f"GPU[{device_index}] : mclk clock level: 0: ({mclk_mhz}Mhz)\n"
            f"GPU[{device_index}] : sclk clock level: 0: ({sclk_mhz}Mhz)\n"
            f"GPU[{device_index}] : Average Graphics Package Power (W): "
            f"{power.value / 1_000_000.0}\n")
        return SnapshotResult(tuple(str(token) for token in argv), 0, text, "")


class AmdgpuHwmonSnapshotRunner:
    """Read the kernel counters behind ROCm SMI without per-read SMI stalls."""

    source = "amdgpu-hwmon/numeric-250ms/v1"

    def __init__(self, *, sysfs_root: str = "/sys/class/drm") -> None:
        candidates = []
        for card in sorted(Path(sysfs_root).glob("card[0-9]*")):
            for hwmon in sorted((card / "device" / "hwmon").glob("hwmon*")):
                name = hwmon / "name"
                try:
                    if name.read_text(encoding="utf-8").strip() == "amdgpu":
                        candidates.append(hwmon)
                except OSError:
                    continue
        if not candidates:
            raise DeviceSamplingError("no readable amdgpu hwmon device exists")
        self._devices = tuple(candidates)

    @staticmethod
    def _number(path: Path) -> int:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise DeviceSamplingError(f"could not read numeric hwmon field {path}: {exc}") from exc

    def __call__(self, argv: Sequence[str], timeout_s: float) -> SnapshotResult:
        del timeout_s
        try:
            device_index = int(argv[argv.index("-d") + 1])
            root = self._devices[device_index]
        except (ValueError, IndexError) as exc:
            raise DeviceSamplingError(
                "logical ROCm device index does not map to an amdgpu hwmon device") from exc
        sclk_mhz = self._number(root / "freq1_input") / 1_000_000.0
        mclk_mhz = self._number(root / "freq2_input") / 1_000_000.0
        power_w = self._number(root / "power1_average") / 1_000_000.0
        temperature_c = self._number(root / "temp2_input") / 1_000.0
        text = (
            f"GPU[{device_index}] : Temperature (Sensor junction) (C): {temperature_c}\n"
            f"GPU[{device_index}] : mclk clock level: 0: ({mclk_mhz}Mhz)\n"
            f"GPU[{device_index}] : sclk clock level: 0: ({sclk_mhz}Mhz)\n"
            f"GPU[{device_index}] : Average Graphics Package Power (W): {power_w}\n")
        return SnapshotResult(tuple(str(token) for token in argv), 0, text, "")


class RocmSmiSamplingSession:
    """One start/stop sampling session; not reusable after ``stop``."""

    def __init__(self, *, command: Sequence[str], interval_s: float,
                 command_timeout_s: float, device_id: str,
                 runner: Callable[[Sequence[str], float], SnapshotResult],
                 monotonic: Callable[[], float], now: Callable[[], str],
                 source: str) -> None:
        self._command = tuple(str(token) for token in command)
        self._interval_s = float(interval_s)
        self._command_timeout_s = float(command_timeout_s)
        self._device_id = device_id
        self._runner = runner
        self._source = source
        self._monotonic = monotonic
        self._now = now
        self._stop = threading.Event()
        self._samples: list[TimedDeviceStateSample] = []
        self._errors: list[str] = []
        self._started_mono: Optional[float] = None
        self._started_at: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "RocmSmiSamplingSession":
        if self._thread is not None:
            raise DeviceSamplingError("device sampling session was already started")
        self._started_mono = self._monotonic()
        self._started_at = self._now()
        self._thread = threading.Thread(
            target=self._sample_loop, name="autokernel-device-sampler", daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        assert self._started_mono is not None
        deadline = self._started_mono
        while not self._stop.is_set():
            delay = deadline - self._monotonic()
            if delay > 0 and self._stop.wait(delay):
                break
            observed = self._monotonic()
            try:
                result = self._runner(self._command, self._command_timeout_s)
                if not isinstance(result, SnapshotResult):
                    raise TypeError("device snapshot runner must return SnapshotResult")
                if result.returncode != 0:
                    raise DeviceSamplingError(
                        f"rocm-smi exited {result.returncode}: {result.stderr[-400:]!r}")
                sample = devices.parse_rocm_smi_snapshot(
                    clocks_text=result.stdout, power_text=result.stdout,
                    temperature_text=result.stdout, under_measurement_load=True)
                self._samples.append(TimedDeviceStateSample(
                    offset_s=observed - self._started_mono, sample=sample))
            except Exception as exc:  # retained and raised by the trusted caller
                self._errors.append(f"{type(exc).__name__}: {exc}")
                self._stop.set()
                break
            # Absolute deadlines prevent probe runtime from accumulating as
            # schedule drift.  A slow probe will instead create a visible gap
            # that DeviceSamplingReceipt refuses.
            deadline += self._interval_s
            while deadline <= self._monotonic():
                deadline += self._interval_s

    def stop(self) -> DeviceSamplingReceipt:
        if self._thread is None or self._started_mono is None or self._started_at is None:
            raise DeviceSamplingError("device sampling session was not started")
        # Capture the caller's boundary BEFORE waiting for an in-flight
        # rocm-smi subprocess to return.  Thread-drain time is not measurement
        # time; using the post-join clock made a 1.1 s window claim ~2 s on this
        # host when the final read was slow.
        ended_mono = self._monotonic()
        ended_at = self._now()
        self._stop.set()
        self._thread.join(timeout=max(2.0, self._command_timeout_s + 1.0))
        if self._thread.is_alive():
            raise DeviceSamplingError("device sampling thread did not stop")
        if self._errors:
            raise DeviceSamplingError(
                "device-state sampling failed closed: " + "; ".join(self._errors))
        return DeviceSamplingReceipt(
            sampler_id=MODULE_ID, device_id=self._device_id,
            source=self._source, started_at=self._started_at,
            ended_at=ended_at, interval_s=self._interval_s,
            duration_s=ended_mono - self._started_mono, command=self._command,
            samples=tuple(self._samples))


class RocmSmiSampler:
    """Factory for exact-lifetime ROCm sampling sessions."""

    def __init__(self, *, rocm_smi: str = DEFAULT_ROCM_SMI, device_index: int = 0,
                 interval_s: float = DEFAULT_INTERVAL_S,
                 command_timeout_s: float = 2.0,
                 runner: Optional[Callable[[Sequence[str], float], SnapshotResult]] = None,
                 source: Optional[str] = None,
                 monotonic: Callable[[], float] = time.monotonic,
                 now: Callable[[], str] = _utc_now) -> None:
        if not isinstance(device_index, int) or isinstance(device_index, bool) \
                or device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if interval_s <= 0 or command_timeout_s <= 0:
            raise ValueError("sampling interval and command timeout must be positive")
        if not isinstance(rocm_smi, str) or not rocm_smi.startswith("/"):
            raise ValueError("rocm_smi must be an absolute executable path")
        self._command = (
            rocm_smi, "-d", str(device_index), "--showclocks", "--showpower", "--showtemp")
        self._interval_s = float(interval_s)
        self._command_timeout_s = float(command_timeout_s)
        self._device_id = f"ROCm{device_index}"
        if runner is None:
            runner = AmdgpuHwmonSnapshotRunner()
            source = source or runner.source
        self._runner = runner
        self._source = source or "rocm-smi/injected-numeric/v1"
        self._monotonic = monotonic
        self._now = now

    def start(self) -> RocmSmiSamplingSession:
        return RocmSmiSamplingSession(
            command=self._command, interval_s=self._interval_s,
            command_timeout_s=self._command_timeout_s, device_id=self._device_id,
            runner=self._runner, monotonic=self._monotonic, now=self._now,
            source=self._source).start()
