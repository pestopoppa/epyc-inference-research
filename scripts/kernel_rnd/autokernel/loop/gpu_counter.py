"""Observe ROCm GFX activity during an owned GPU claim; never gate a run."""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import threading
import time

LIBRARY = Path("/opt/rocm-6.2.0/lib/librocm_smi64.so.7")
DEVICE = Path("/sys/class/drm/card2/device")
SCHEMA = "epyc.autokernel.gfx_counter_trace.v1"


class _Counter(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("value", ctypes.c_uint64)]


class _Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
                ("patch", ctypes.c_uint32), ("build", ctypes.c_char_p)]


def _bdf_number(text):
    domain, bus, rest = text.split(":")
    device, function = rest.split(".")
    return ((int(domain, 16) << 32) | (int(bus, 16) << 8)
            | (int(device, 16) << 3) | int(function, 16))


class RocmReader:
    """ROCm 6.2 ABI, selected by claim device and verified PCI identity."""
    def __init__(self, device_id):
        if device_id != "mi210_0":
            raise ValueError(f"unmapped claimed device: {device_id}")
        self.path = LIBRARY.resolve(strict=True)
        info = self.path.stat()
        self.file_identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = self.path.stat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != self.file_identity:
            raise ValueError("ROCm library changed while hashing")
        self.sha256 = digest.hexdigest()
        self.bdf = DEVICE.resolve(strict=True).name
        if self.bdf != "0000:43:00.0":
            raise ValueError("claimed mi210_0 PCI identity changed")
        expected = _bdf_number(self.bdf)
        self.lib = ctypes.CDLL(str(self.path))
        signatures = {
            "rsmi_init": [ctypes.c_uint64], "rsmi_shut_down": [],
            "rsmi_num_monitor_devices": [ctypes.POINTER(ctypes.c_uint32)],
            "rsmi_dev_pci_id_get": [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)],
            "rsmi_utilization_count_get": [ctypes.c_uint32, ctypes.POINTER(_Counter),
                                             ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)],
            "rsmi_version_get": [ctypes.POINTER(_Version)],
        }
        for name, arguments in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = arguments, ctypes.c_int
        self._call("rsmi_init", 0)
        try:
            version = _Version()
            self._call("rsmi_version_get", ctypes.byref(version))
            self.version = {"major": version.major, "minor": version.minor,
                            "patch": version.patch,
                            "build": version.build.decode(errors="replace") if version.build else None}
            count = ctypes.c_uint32()
            self._call("rsmi_num_monitor_devices", ctypes.byref(count))
            matches = []
            for index in range(count.value):
                bdf = ctypes.c_uint64()
                self._call("rsmi_dev_pci_id_get", index, ctypes.byref(bdf))
                if bdf.value == expected:
                    matches.append(index)
            if len(matches) != 1:
                raise ValueError("claimed GPU PCI identity did not map uniquely to ROCm")
            self.index, self.expected = matches[0], expected
        except BaseException:
            self.lib.rsmi_shut_down()
            raise

    def _call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name}: RSMI status {code}")

    def read(self):
        info = self.path.stat()
        if LIBRARY.resolve(strict=True) != self.path or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != self.file_identity:
            raise ValueError("ROCm library identity changed")
        if DEVICE.resolve(strict=True).name != self.bdf:
            raise ValueError("claimed GPU sysfs identity changed")
        bdf = ctypes.c_uint64()
        self._call("rsmi_dev_pci_id_get", self.index, ctypes.byref(bdf))
        if bdf.value != self.expected:
            raise ValueError("ROCm GPU identity changed")
        counter, timestamp = _Counter(0, 0), ctypes.c_uint64()
        before = time.monotonic_ns()
        self._call("rsmi_utilization_count_get", self.index, ctypes.byref(counter), 1,
                   ctypes.byref(timestamp))
        after = time.monotonic_ns()
        return {"host_before_ns": before, "host_after_ns": after,
                "counter_timestamp_ns": timestamp.value, "counter": counter.value}

    def identity(self):
        return {"bdf": self.bdf, "index": self.index, "library": str(self.path),
                "library_file_identity": list(self.file_identity), "library_sha256": self.sha256,
                "version": self.version,
                "counter_type": "RSMI_COARSE_GRAIN_GFX_ACTIVITY"}

    def close(self):
        self._call("rsmi_shut_down")


def reduce_trace(samples, errors):
    """A counter delta is weighted GFX activity, not a binary busy duration.

    Native and host elapsed spans must agree within their actual call brackets. A
    refusal is not proof of different clock rates, nor is a pass proof that the
    firmware clocks remain equivalent across resets or between observations.
    """
    result = {"observed_window": None, "approx_gfx_activity_weighted_seconds": None,
              "unavailable_reason": None,
              "limitation": "device-wide GFX activity; resets between reads cannot be excluded"}
    def unavailable(reason):
        result["unavailable_reason"] = reason
        return result
    if errors:
        return unavailable("collector reported an error")
    if len(samples) < 2:
        return unavailable("fewer than two counter samples")
    keys = ("host_before_ns", "host_after_ns", "counter_timestamp_ns", "counter")
    if any(not isinstance(row, dict) or any(
            not isinstance(row.get(k), int) or isinstance(row.get(k), bool)
            or row[k] < (0 if k == "counter" else 1)
            for k in keys) for row in samples):
        return unavailable("invalid counter or nonpositive timestamp")
    if any(row["host_before_ns"] > row["host_after_ns"] for row in samples):
        return unavailable("reversed host timestamp bracket")
    for a, b in zip(samples, samples[1:]):
        if a["host_after_ns"] > b["host_before_ns"]:
            return unavailable("overlapping or decreasing host timestamp brackets")
        native = b["counter_timestamp_ns"] - a["counter_timestamp_ns"]
        if native <= 0 or b["counter"] < a["counter"]:
            return unavailable("counter or native timestamp decreased or stopped")
        lower = b["host_before_ns"] - a["host_after_ns"]
        upper = b["host_after_ns"] - a["host_before_ns"]
        if not lower <= native <= upper:
            return unavailable("native and host timestamp spans are inconsistent")
        if (b["counter"] - a["counter"]) * 10000 > native:
            return unavailable("counter delta exceeds 100 percent per millisecond")
    first, last = samples[0], samples[-1]
    result["observed_window"] = {"start": dict(first), "end": dict(last),
                                 "native_timestamp_origin": "unspecified"}
    result["approx_gfx_activity_weighted_seconds"] = (last["counter"] - first["counter"]) / 100000.0
    return result


class Collector:
    """Cumulative firmware counter is continuous; read at claim and phase boundaries."""
    def __init__(self, device_id, *, reader_factory=RocmReader):
        self.device_id, self.reader_factory = device_id, reader_factory
        self._lock = threading.Lock()
        self._samples, self._errors = [], []
        self._identity, self._reader = None, None
        self._closed = False
        self._acquired_observed_ns = time.monotonic_ns()
        self._release_observed_ns = None

    def start(self):
        with self._lock:
            try:
                self._reader = self.reader_factory(self.device_id)
                self._identity = self._reader.identity()
                self._samples.append({**self._reader.read(), "phase": "claim_start"})
            except Exception as exc:
                self._errors.append(f"{type(exc).__name__}: {exc}")
        return self

    def sample(self, phase="phase"):
        with self._lock:
            if self._closed or self._reader is None:
                return
            try:
                self._samples.append({**self._reader.read(), "phase": str(phase)[:256]})
            except Exception as exc:
                self._errors.append(f"{type(exc).__name__}: {exc}")

    def stop(self):
        with self._lock:
            if self._closed:
                return self._snapshot_locked()
            if self._reader is not None:
                try:
                    self._samples.append({**self._reader.read(), "phase": "claim_end"})
                except Exception as exc:
                    self._errors.append(f"{type(exc).__name__}: {exc}")
                try:
                    self._reader.close()
                except Exception as exc:
                    self._errors.append(f"{type(exc).__name__}: {exc}")
            self._release_observed_ns = time.monotonic_ns()
            self._closed = True
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self):
        samples, errors = list(self._samples), list(self._errors)
        identity, closed = self._identity, self._closed
        return {"schema": SCHEMA, "claim_device_id": self.device_id,
                "identity": identity, "samples": samples, "errors": errors,
                "closed": closed, "claim_acquired_observed_ns": self._acquired_observed_ns,
                "claim_release_observed_ns": self._release_observed_ns,
                "full_claim_coverage": None,
                **reduce_trace(samples, errors)}


def attach(ownership, device_id, persist, *, collector_factory=Collector):
    """Attach after acquisition; ExitStack stops/persists before claim release."""
    collector = collector_factory(device_id)
    def finish():
        try:
            persist(collector.stop())
        except Exception as exc:
            print(f"GPU counter observation unavailable: {type(exc).__name__}: {exc}")
    ownership.callback(finish)
    try:
        collector.start()
    except Exception as exc:
        print(f"GPU counter observation unavailable: {type(exc).__name__}: {exc}")
    return collector
