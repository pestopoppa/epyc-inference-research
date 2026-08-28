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
#: A model resident on the device moves VRAM well past this. Below it, "resident" is
#: not proven -- which is a refusal, not a warning.
RESIDENT_FLOOR_BYTES = 1 << 30


def vram_bytes() -> int:
    try:
        return int(VRAM_SYSFS.read_text().strip())
    except (OSError, ValueError):
        return -1


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
        self.samples = 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_vram = max(self.peak_vram, vram_bytes())
            self.peak_kfd = max(self.peak_kfd, kfd_processes())
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
           "kfd_processes", "loader_env", "vram_bytes"]
