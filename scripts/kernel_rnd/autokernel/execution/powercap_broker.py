"""Narrow read-only access to root-owned package-energy counters.

The benchmark remains a non-root Landlock/seccomp child. Only a captured,
networkless container reads the powercap mount, and it exposes no socket or
writable path to candidate code. Every lifecycle command names the exact
container id returned by ``docker run``; there is no process-name search.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

__all__ = [
    "DEFAULT_IMAGE_ID", "POWER_CAP_ROOT", "PowercapBroker",
    "PowercapBrokerError", "PowercapBrokerReceipt",
]

DEFAULT_IMAGE_ID = (
    "sha256:3a2e92b4133d06d1287f96ec47bacd743717b377f4b9df6be1e3af626c35dbb0"
)
POWER_CAP_ROOT = Path("/sys/devices/virtual/powercap")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_READ_SCRIPT = r"""
find /powercap -type f -name name -print | sort | while IFS= read -r name_file; do
    name=$(cat "$name_file")
    case "$name" in
        package-*)
            domain=${name_file%/name}
            package=${name#package-}
            energy=$(cat "$domain/energy_uj")
            maximum=$(cat "$domain/max_energy_range_uj")
            printf '%s\t%s\t%s\t%s\n' "$package" "$energy" "$maximum" "$domain/energy_uj"
            ;;
    esac
done
""".strip()


class PowercapBrokerError(RuntimeError):
    """The privileged reader could not prove a complete exact read."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PowercapBrokerReceipt:
    image_id: str
    container_id: str
    powercap_root: str
    started_at: str
    read_script_sha256: str

    def to_dict(self) -> dict:
        return {
            "schema": "epyc.autokernel.powercap_broker_receipt.v1",
            "image_id": self.image_id,
            "container_id": self.container_id,
            "powercap_root": self.powercap_root,
            "started_at": self.started_at,
            "read_script_sha256": self.read_script_sha256,
        }


class PowercapBroker:
    """A lazily-started, read-only Docker powercap reader.

    Container startup never occurs inside a measured block: callers keep one
    broker alive across preflight and all open/close host-state snapshots. A
    read uses ``docker exec`` against the captured id and returns only integer
    energy counters for the package ids derived from the claimed CPU list.
    """

    def __init__(self, *, image_id: str = DEFAULT_IMAGE_ID,
                 powercap_root: Path | str = POWER_CAP_ROOT,
                 runner: Callable[..., Any] = subprocess.run) -> None:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError("image_id must be a full sha256 image id")
        self._image_id = image_id
        self._powercap_root = Path(powercap_root)
        self._runner = runner
        self._container_id: Optional[str] = None
        self._receipt: Optional[PowercapBrokerReceipt] = None

    @property
    def active(self) -> bool:
        return self._container_id is not None

    @property
    def container_id(self) -> Optional[str]:
        return self._container_id

    def _run(self, argv: Sequence[str]) -> Any:
        return self._runner(
            tuple(argv), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)

    @staticmethod
    def _detail(proc: Any) -> str:
        return (str(getattr(proc, "stderr", "")).strip()
                or str(getattr(proc, "stdout", "")).strip()
                or f"exit {getattr(proc, 'returncode', '?')}")

    def start(self) -> PowercapBrokerReceipt:
        if self._receipt is not None:
            return self._receipt
        if not self._powercap_root.is_dir():
            raise PowercapBrokerError(
                f"powercap root {self._powercap_root} is unavailable")

        image = self._run(("docker", "image", "inspect", "--format", "{{.Id}}",
                           self._image_id))
        if image.returncode != 0 or image.stdout.strip() != self._image_id:
            raise PowercapBrokerError(
                f"pinned broker image is unavailable: {self._detail(image)}")

        proc = self._run((
            "docker", "run", "--detach", "--rm", "--network", "none",
            "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "16",
            "--mount",
            f"type=bind,src={self._powercap_root},dst=/powercap,readonly",
            self._image_id, "sh", "-ceu", "while :; do sleep 3600; done",
        ))
        container_id = proc.stdout.strip()
        if proc.returncode != 0 or not _CONTAINER_ID.fullmatch(container_id):
            raise PowercapBrokerError(
                f"could not start the powercap reader: {self._detail(proc)}")
        self._container_id = container_id
        try:
            state = self._run(("docker", "container", "inspect", "--format",
                               "{{.State.Running}} {{.Image}}", container_id))
            expected = f"true {self._image_id}"
            if state.returncode != 0 or state.stdout.strip() != expected:
                raise PowercapBrokerError(
                    f"broker identity/running-state mismatch: {self._detail(state)}")
        except BaseException:
            self.close()
            raise

        self._receipt = PowercapBrokerReceipt(
            image_id=self._image_id, container_id=container_id,
            powercap_root=str(self._powercap_root), started_at=_utc_now(),
            read_script_sha256=hashlib.sha256(_READ_SCRIPT.encode()).hexdigest())
        return self._receipt

    def receipt(self) -> PowercapBrokerReceipt:
        return self.start()

    def read_package_energy(self, *, packages: Sequence[int]) -> tuple:
        requested = tuple(sorted(set(int(package) for package in packages)))
        if not requested:
            return ()
        receipt = self.start()
        proc = self._run(("docker", "exec", receipt.container_id,
                          "sh", "-ceu", _READ_SCRIPT))
        if proc.returncode != 0:
            raise PowercapBrokerError(
                f"powercap read failed: {self._detail(proc)}")

        found: dict[int, tuple[int, int, str]] = {}
        for line in proc.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) != 4:
                raise PowercapBrokerError(f"malformed powercap row: {line!r}")
            try:
                package, energy, maximum = map(int, fields[:3])
            except ValueError as exc:
                raise PowercapBrokerError(
                    f"non-integer powercap row: {line!r}") from exc
            if package in found or energy < 0 or maximum <= 0:
                raise PowercapBrokerError(f"invalid powercap row: {line!r}")
            found[package] = (energy, maximum, fields[3])

        missing = [package for package in requested if package not in found]
        if missing:
            raise PowercapBrokerError(
                f"broker returned no counter for package(s) {missing}")
        return tuple(
            (package, found[package][0], found[package][1],
             f"docker:{receipt.container_id}@{receipt.image_id}:"
             f"script={receipt.read_script_sha256}:{found[package][2]}")
            for package in requested
        )

    def read_host_state(self, **kwargs: Any) -> Any:
        from . import microbench  # local import avoids a module cycle

        return microbench.read_host_state(
            package_energy_reader=self.read_package_energy, **kwargs)

    def _inspect_running(self, container_id: str) -> Optional[bool]:
        proc = self._run(("docker", "container", "inspect", "--format",
                          "{{.State.Running}}", container_id))
        if proc.returncode == 0:
            value = proc.stdout.strip()
            if value not in ("true", "false"):
                raise PowercapBrokerError(
                    f"unexpected broker state: {value!r}")
            return value == "true"
        detail = self._detail(proc).lower()
        if "no such" in detail or "not found" in detail:
            return None
        raise PowercapBrokerError(
            f"could not verify broker teardown: {self._detail(proc)}")

    def close(self) -> None:
        container_id = self._container_id
        if container_id is None:
            return
        stop = self._run(("docker", "stop", "--time", "5", container_id))
        running = self._inspect_running(container_id)
        if running:
            killed = self._run(("docker", "kill", container_id))
            if killed.returncode != 0:
                raise PowercapBrokerError(
                    f"broker SIGKILL escalation failed: {self._detail(killed)}")
            running = self._inspect_running(container_id)
        if running:
            raise PowercapBrokerError(
                f"captured broker container {container_id} is still running")
        if running is False:
            removed = self._run(("docker", "rm", container_id))
            if removed.returncode != 0:
                raise PowercapBrokerError(
                    f"stopped broker could not be removed: {self._detail(removed)}")
            if self._inspect_running(container_id) is not None:
                raise PowercapBrokerError(
                    f"captured broker container {container_id} still exists")
        if stop.returncode != 0 and running is not None:
            raise PowercapBrokerError(
                f"broker stop failed: {self._detail(stop)}")
        self._container_id = None
        self._receipt = None

    def __enter__(self) -> "PowercapBroker":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
