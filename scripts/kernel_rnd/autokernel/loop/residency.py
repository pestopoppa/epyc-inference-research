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
import threading

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

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_vram = max(self.peak_vram, vram_bytes())
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
        return {
            "peak_vram_bytes": self.peak_vram,
            "peak_kfd_processes": self.peak_kfd,
            "sclk_min_mhz": self.min_sclk,
            "sclk_max_mhz": self.max_sclk,
            # A measurement taken across a clock change is not a measurement of the
            # kernel; it is partly a measurement of the governor.
            "clock_stable": bool(self.min_sclk) and self.min_sclk == self.max_sclk,
            "samples": self.samples,
            "resident": self.peak_vram >= RESIDENT_FLOOR_BYTES,
        }


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
