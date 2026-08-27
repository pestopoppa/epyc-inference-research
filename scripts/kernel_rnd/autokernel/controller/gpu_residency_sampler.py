"""Static MI210 KFD/VRAM sampler for governed source proof commands."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from . import gpu_source_evidence as evidence


class ResidencySamplerError(RuntimeError):
    pass


class GpuContentionTimeout(ResidencySamplerError):
    """The GPU did not become free of FOREIGN processes within the bound.

    Distinct from a bare ResidencySamplerError so a caller can classify a
    contention wait-out separately from an unreadable /sys.  A process inside
    the campaign's OWN process tree (a sibling leg draining) is never foreign
    and never counted here — the 2026-08-27 audit found the controller's own
    child (pid 964901) flagged as foreign, crashing the deployment.
    """


class Mi210ResidencySampler:
    """Sample only during a captured child lifetime; foreign KFD is a refusal.

    ``owner_root_pid`` names the campaign's own process root (the controller).
    Any KFD process whose ancestry reaches it is OURS — a sibling leg draining
    is not "foreign".  When it is ``None`` the classifier falls back to the
    original "descendant of the sampled child only" rule, so existing callers
    and tests are unchanged.
    """
    DEVICE_ID = "mi210_0"
    KFD = Path("/sys/class/kfd/kfd/proc")
    VRAM = Path("/sys/class/drm/card2/device/mem_info_vram_used")

    def __init__(self, *, kfd_root: Path = KFD, vram_path: Path = VRAM,
                 proc_root: Path = Path("/proc"),
                 owner_root_pid: int | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        if owner_root_pid is not None and (isinstance(owner_root_pid, bool)
                                           or owner_root_pid < 1):
            raise ResidencySamplerError("owner_root_pid must be a positive pid or None")
        self.kfd_root, self.vram_path, self.proc_root = kfd_root, vram_path, proc_root
        self.owner_root_pid = owner_root_pid
        self._sleep, self._monotonic = sleep, monotonic

    def _parent(self, pid: int) -> int | None:
        try:
            # /proc/<pid>/stat's parent is field 4 after the final ')' — names
            # can contain spaces/parentheses, so split only after that delimiter.
            raw = (self.proc_root / str(pid) / "stat").read_text(encoding="utf-8")
            return int(raw.rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            return None

    def _reaches(self, pid: int, target: int) -> bool:
        """True if walking pid's parent chain reaches ``target``."""
        current = pid
        seen: set[int] = set()
        while current not in seen and current > 0:
            if current == target:
                return True
            seen.add(current)
            parent = self._parent(current)
            if parent is None:
                return False
            current = parent
        return False

    def _belongs(self, pid: int, child: int) -> bool:
        return self._reaches(pid, child)

    def _is_ours(self, pid: int, child_pid: int | None) -> bool:
        """A KFD pid is ours if it descends from the sampled child OR, when an
        owner root is known, from the campaign's own process tree."""
        if child_pid is not None and self._belongs(pid, child_pid):
            return True
        if self.owner_root_pid is not None and self._reaches(pid, self.owner_root_pid):
            return True
        return False

    def _read_kfd_pids(self) -> tuple[int, ...]:
        return tuple(sorted(int(path.name) for path in self.kfd_root.iterdir()
                            if path.name.isdecimal()))

    def wait_until_clear(self, *, timeout_s: float = 120.0,
                         poll_s: float = 1.0) -> tuple[int, ...]:
        """Block (bounded) until no FOREIGN KFD process holds the GPU.

        Returns the owned KFD pids once the device is free of foreign work, so
        a timed leg never starts on a contended GPU (the common cause of the
        v27 'foreign KFD' crashes, which used to abort the whole deployment).
        Raises GpuContentionTimeout if foreign work persists past the bound.
        """
        if timeout_s <= 0 or poll_s <= 0:
            raise ResidencySamplerError("timeout_s and poll_s must be positive")
        deadline = self._monotonic() + timeout_s
        while True:
            try:
                pids = self._read_kfd_pids()
            except (OSError, ValueError) as exc:
                raise ResidencySamplerError("MI210 KFD sampling is unavailable") from exc
            foreign = tuple(pid for pid in pids if not self._is_ours(pid, None))
            if not foreign:
                return tuple(pid for pid in pids if self._is_ours(pid, None))
            if self._monotonic() >= deadline:
                raise GpuContentionTimeout(
                    f"GPU still held by foreign KFD process(es) after {timeout_s:.0f}s: {foreign}")
            self._sleep(poll_s)

    def __call__(self, child_pid: int) -> evidence.GpuResidencySample:
        try:
            pids = self._read_kfd_pids()
            vram = int(self.vram_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise ResidencySamplerError("MI210 KFD/VRAM sampling is unavailable") from exc
        owned = tuple(pid for pid in pids if self._belongs(pid, child_pid))
        # A sibling leg in our OWN process tree is not foreign — it is not
        # attributed to this leg's residency, but it must not crash the proof.
        foreign = tuple(pid for pid in pids
                        if pid not in owned and not self._is_ours(pid, child_pid))
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
                                           vram_bytes=vram, launcher_pid=child_pid)
