"""Static MI210 KFD/VRAM sampler for governed source proof commands."""
from __future__ import annotations

import time
from pathlib import Path

from . import gpu_source_evidence as evidence


class ResidencySamplerError(RuntimeError):
    pass


class Mi210ResidencySampler:
    """Sample only during a captured child lifetime; foreign KFD is a refusal."""
    DEVICE_ID = "mi210_0"
    KFD = Path("/sys/class/kfd/kfd/proc")
    VRAM = Path("/sys/class/drm/card2/device/mem_info_vram_used")

    def __init__(self, *, kfd_root: Path = KFD, vram_path: Path = VRAM,
                 proc_root: Path = Path("/proc")) -> None:
        self.kfd_root, self.vram_path, self.proc_root = kfd_root, vram_path, proc_root

    def _parent(self, pid: int) -> int | None:
        try:
            # /proc/<pid>/stat's parent is field 4 after the final ')' — names
            # can contain spaces/parentheses, so split only after that delimiter.
            raw = (self.proc_root / str(pid) / "stat").read_text(encoding="utf-8")
            return int(raw.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            return None

    def _belongs(self, pid: int, child: int) -> bool:
        current = pid
        seen: set[int] = set()
        while current not in seen and current > 0:
            if current == child:
                return True
            seen.add(current)
            parent = self._parent(current)
            if parent is None:
                return False
            current = parent
        return False

    def __call__(self, child_pid: int) -> evidence.GpuResidencySample:
        try:
            pids = tuple(sorted(int(path.name) for path in self.kfd_root.iterdir()
                                if path.name.isdecimal()))
            vram = int(self.vram_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise ResidencySamplerError("MI210 KFD/VRAM sampling is unavailable") from exc
        owned = tuple(pid for pid in pids if self._belongs(pid, child_pid))
        foreign = tuple(pid for pid in pids if pid not in owned)
        if foreign:
            raise ResidencySamplerError(f"foreign KFD process overlaps governed source proof: {foreign}")
        if not owned:
            # GpuResidencySample deliberately requires a non-empty PID tuple.
            # Preserve that shape for a pre-dispatch poll without inventing a
            # KFD witness: zero VRAM makes this sample incapable of satisfying
            # the evidence residency predicate.  The executor keeps polling
            # while the child is alive and only a later owned KFD PID can prove
            # residency.
            return evidence.GpuResidencySample(
                observed_monotonic_ns=time.monotonic_ns(), device_id=self.DEVICE_ID,
                kfd_pids=(child_pid,), vram_bytes=0)
        return evidence.GpuResidencySample(observed_monotonic_ns=time.monotonic_ns(),
                                           device_id=self.DEVICE_ID, kfd_pids=owned,
                                           vram_bytes=vram)
