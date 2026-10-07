"""Fresh serving child ownership of its local HTTP listening socket.

Linux process/socket observations only. Checks bracket request timers; they do
not authenticate responses or claim an adversarial endpoint guarantee.
"""
from __future__ import annotations

import os
from pathlib import Path
import socket


class ServerOwnershipRefused(RuntimeError):
    """The declared endpoint is not owned by the fresh original child."""


def require_free_port(port: int) -> None:
    """Refuse an existing local listener before launching another server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # Match the real server so ordinary active-close TIME_WAIT is reusable.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ServerOwnershipRefused(f"serving port {port} is already occupied") from exc


class ServerPortOwner:
    def __init__(self, child, port: int):
        self.child, self.port = child, port
        self.root = Path("/proc") / str(child.pid)
        self.start_ticks = self._identity()

    def _identity(self) -> int:
        if self.child.poll() is not None:
            raise ServerOwnershipRefused("fresh serving child already exited")
        try:
            tail = (self.root / "stat").read_text().rpartition(") ")[2].split()
            if tail[0] == "Z":
                raise ServerOwnershipRefused("fresh serving child is a zombie")
            return int(tail[19])
        except (OSError, ValueError, IndexError) as exc:
            raise ServerOwnershipRefused("fresh serving child identity unavailable") from exc

    def listener_ready(self) -> bool:
        """False while our child starts; refuse any other listener or changed child."""
        if self._identity() != self.start_ticks:
            raise ServerOwnershipRefused("fresh serving child generation changed")
        inodes = set()
        try:
            for table, addresses in (
                ("tcp", {"0100007F", "00000000"}),
                ("tcp6", {"00000000000000000000000000000000",
                          "0000000000000000FFFF00000100007F"}),
            ):
                for line in (self.root / "net" / table).read_text().splitlines()[1:]:
                    fields = line.split()
                    address, port = fields[1].split(":")
                    if fields[3] == "0A" and int(port, 16) == self.port and address in addresses:
                        inodes.add(fields[9])
            owned = set()
            for fd in (self.root / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue  # A closed descriptor is absent, never an ownership witness.
                if target.startswith("socket:["):
                    owned.add(target[8:-1])
        except (OSError, ValueError, IndexError) as exc:
            raise ServerOwnershipRefused("fresh serving listener ownership unavailable") from exc
        if self._identity() != self.start_ticks:
            raise ServerOwnershipRefused("fresh serving child changed during socket observation")
        if inodes - owned:
            raise ServerOwnershipRefused(f"serving port {self.port} belongs to another process")
        return bool(inodes)

    def require_listener(self) -> None:
        if not self.listener_ready():
            raise ServerOwnershipRefused("fresh serving child has no owned listener")
