"""Filesystem and process containment primitives for discovery_supervisor.

This module deliberately exposes only fd-relative operations.  Pathnames are
used to select an authority root once; all security decisions are made against
the opened object and every later leaf operation is relative to that pinned fd.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import time
from typing import Any, Mapping


class SecureRuntimeError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def object_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "dev": info.st_dev,
        "ino": info.st_ino,
        "uid": info.st_uid,
        "nlink": info.st_nlink,
        "mode": stat.S_IMODE(info.st_mode),
        "size": info.st_size,
    }


def directory_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "dev": info.st_dev,
        "ino": info.st_ino,
        "uid": info.st_uid,
        "nlink": info.st_nlink,
        "mode": stat.S_IMODE(info.st_mode),
    }


def read_stable_fd(
    fd: int, *, limit: int, require_owned: bool = False, require_single_link: bool = False
) -> tuple[bytes, dict[str, int]]:
    before = os.fstat(fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size > limit
        or (require_owned and before.st_uid != os.getuid())
        or (require_single_link and before.st_nlink != 1)
    ):
        raise SecureRuntimeError("opened object violates its regular-file identity")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise SecureRuntimeError("opened object exceeds its byte ceiling")
    after = os.fstat(fd)
    stable = (
        "st_dev",
        "st_ino",
        "st_uid",
        "st_nlink",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(after, key) for key in stable) or total != after.st_size:
        raise SecureRuntimeError("opened object changed while being read")
    return b"".join(chunks), object_identity(after)


def open_stable(path: Path, *, limit: int = 64 * 1024 * 1024) -> tuple[int, bytes, dict[str, int]]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        raw, identity = read_stable_fd(fd, limit=limit)
        return fd, raw, identity
    except Exception:
        os.close(fd)
        raise


@dataclass
class RuntimeRoot:
    path: Path
    fd: int
    identity: dict[str, int]

    @classmethod
    def create_or_open(cls, path: Path) -> "RuntimeRoot":
        if not path.is_absolute() or ".." in path.parts:
            raise SecureRuntimeError("runtime root must be an absolute lexical path")
        old_umask = os.umask(0o077)
        try:
            path.mkdir(parents=True, mode=0o700, exist_ok=True)
        finally:
            os.umask(old_umask)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        info = os.fstat(fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            os.close(fd)
            raise SecureRuntimeError("runtime root must be an owned mode-0700 directory")
        return cls(path.absolute(), fd, directory_identity(info))

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def verify(self, expected: Mapping[str, Any] | None = None) -> None:
        current = directory_identity(os.fstat(self.fd))
        current_path = directory_identity(os.stat(self.path, follow_symlinks=False))
        if current != self.identity or current_path != self.identity:
            raise SecureRuntimeError("runtime root object identity changed")
        if expected is not None and dict(expected) != current:
            raise SecureRuntimeError("runtime root differs from launch specification")

    def open_leaf(self, name: str, flags: int, mode: int = 0o600) -> int:
        if not name or "/" in name or name in {".", ".."}:
            raise SecureRuntimeError("state leaf is not a single safe component")
        return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=self.fd)

    def read_bytes(self, name: str, *, limit: int = 64 * 1024 * 1024, mode: int = 0o600) -> bytes:
        fd = self.open_leaf(name, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            raw, identity = read_stable_fd(fd, limit=limit)
            if identity["mode"] != mode or identity["uid"] != os.getuid() or identity["nlink"] != 1:
                raise SecureRuntimeError("private state mode is invalid")
            return raw
        finally:
            os.close(fd)

    def exists(self, name: str) -> bool:
        try:
            fd = self.open_leaf(name, os.O_RDONLY)
        except FileNotFoundError:
            return False
        else:
            os.close(fd)
            return True

    def atomic_bytes(self, name: str, raw: bytes) -> None:
        temporary = f".{name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
        fd = self.open_leaf(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise SecureRuntimeError("atomic state write made no progress")
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        os.fsync(self.fd)
        self.verify()

    def open_append(self, name: str) -> int:
        created = False
        try:
            fd = self.open_leaf(name, os.O_RDWR | os.O_APPEND)
        except FileNotFoundError:
            fd = self.open_leaf(name, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        info = os.fstat(fd)
        identity = object_identity(info)
        if (
            identity["uid"] != os.getuid()
            or identity["nlink"] != 1
            or identity["mode"] != 0o600
            or not stat.S_ISREG(info.st_mode)
        ):
            os.close(fd)
            raise SecureRuntimeError("append target lost private object identity")
        if created:
            os.fsync(self.fd)
        return fd


def open_beneath(root_fd: int, relative: str, flags: int = os.O_RDONLY) -> int:
    """Open a no-symlink descendant with component-by-component openat."""
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in {"", ".", ".."} for p in parts):
        raise SecureRuntimeError("beneath path is not a safe relative path")
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current
            )
            os.close(current)
            current = next_fd
        return os.open(parts[-1], flags | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
    finally:
        os.close(current)


class OwnedCgroup:
    """One exact cgroup v2 subtree; never addresses a process group number."""

    def __init__(self, name: str, *, base: Path = Path("/sys/fs/cgroup")) -> None:
        if not name.startswith("epyc-autokernel-") or "/" in name:
            raise SecureRuntimeError("invalid AutoKernel cgroup name")
        self.base = base
        self.name = name
        self.path = base / name
        self.dir_fd: int | None = None

    def create(self) -> None:
        base_fd = os.open(self.base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                os.mkdir(self.name, 0o700, dir_fd=base_fd)
            except FileExistsError:
                pass
            self.dir_fd = os.open(
                self.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=base_fd
            )
        finally:
            os.close(base_fd)
        info = os.fstat(self.dir_fd)
        if info.st_uid != os.getuid():
            raise SecureRuntimeError("controller cgroup is not owned by this uid")
        if self.pids():
            raise SecureRuntimeError("controller cgroup was not empty at acquisition")

    def _leaf(self, name: str, flags: int) -> int:
        if self.dir_fd is None:
            raise SecureRuntimeError("controller cgroup is not open")
        return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.dir_fd)

    def _read(self, name: str) -> str:
        fd = self._leaf(name, os.O_RDONLY)
        try:
            chunks = []
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks).decode("ascii")
        finally:
            os.close(fd)

    def _write(self, name: str, value: bytes) -> None:
        fd = self._leaf(name, os.O_WRONLY)
        try:
            os.write(fd, value)
        finally:
            os.close(fd)

    def pids(self) -> tuple[int, ...]:
        return tuple(int(row) for row in self._read("cgroup.procs").split() if row)

    def identity(self) -> dict[str, int | str]:
        if self.dir_fd is None:
            raise SecureRuntimeError("controller cgroup is not open")
        return {"path": str(self.path), **directory_identity(os.fstat(self.dir_fd))}

    def populated(self) -> bool:
        fields = dict(
            line.split(maxsplit=1) for line in self._read("cgroup.events").splitlines())
        if fields.get("populated") not in {"0", "1"}:
            raise SecureRuntimeError("controller cgroup populated state is malformed")
        return fields["populated"] == "1"

    def add(self, pid: int) -> None:
        self._write("cgroup.procs", f"{pid}\n".encode("ascii"))
        if pid not in self.pids():
            raise SecureRuntimeError("child did not enter its dedicated cgroup")

    def signal_all(self, signum: int, identities: Mapping[int, int]) -> bool:
        sent = False
        for pid in self.pids():
            expected_ticks = identities.get(pid)
            if expected_ticks is None:
                continue
            try:
                fd = os.pidfd_open(pid, 0)
            except ProcessLookupError:
                continue
            try:
                raw = Path(f"/proc/{pid}/stat").read_bytes()
                close = raw.rfind(b")")
                fields = raw[close + 1 :].split()
                if close < 0 or len(fields) < 20 or int(fields[19]) != expected_ticks:
                    raise SecureRuntimeError("cgroup member PID identity changed")
                signal.pidfd_send_signal(fd, signum)
                sent = True
            finally:
                os.close(fd)
        return sent

    def kill(self) -> None:
        self._write("cgroup.kill", b"1\n")

    def wait_empty(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.populated():
                return True
            time.sleep(0.02)
        return not self.populated()

    def close_and_remove(self) -> None:
        if self.dir_fd is not None:
            if self.populated():
                raise SecureRuntimeError("refusing to remove populated controller cgroup")
            os.close(self.dir_fd)
            self.dir_fd = None
        descendants = sorted(
            (path for path in self.path.rglob("*")
             if path.is_dir() and not path.is_symlink()),
            key=lambda path: len(path.parts), reverse=True)
        for descendant in descendants:
            info = descendant.stat(follow_symlinks=False)
            if info.st_uid != os.getuid():
                raise SecureRuntimeError("controller descendant cgroup owner changed")
            descendant.rmdir()
        os.rmdir(self.path)
