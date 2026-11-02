#!/usr/bin/env python3
"""Building blocks for KVU-19a-2 source and runtime custody.

Sealed memfds prevent later writes to the copied archive/patch bytes. Runtime
sampling binds mapped ggml files and executable to PID birth and digests.
Neither operation alone proves compiler input custody or GPU residency.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path

# Linux fcntl.h ABI; some Python builds omit these names.
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)

def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sealed_copy(path: Path, expected_sha256: str) -> int:
    """Return an immutable memfd containing exactly the expected source bytes."""
    fd = os.memfd_create("kvu19a2-source", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        h = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                h.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("memfd write failed")
                    view = view[written:]
        if h.hexdigest() != expected_sha256:
            raise ValueError("source digest mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, F_ADD_SEALS, SEAL_WRITE | SEAL_GROW | SEAL_SHRINK | SEAL_SEAL)
        return fd
    except BaseException:
        os.close(fd)
        raise


def process_birth(proc: Path) -> int:
    # comm is parenthesized and may itself contain spaces and parentheses.
    stat = (proc / "stat").read_text()
    tail = stat[stat.rindex(")") + 2:].split()
    return int(tail[19])  # stat field 22; tail starts at field 3


def ggml_maps(text: str) -> dict[str, tuple[str, int]]:
    result = {}
    for line in text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        path = parts[5]
        if "libggml" not in Path(path.removesuffix(" (deleted)")).name:
            continue
        if not path.startswith("/") or path.endswith(" (deleted)"):
            raise ValueError("unverifiable mapped ggml path")
        identity = (parts[3], int(parts[4]))
        if path in result and result[path] != identity:
            raise ValueError("conflicting map identities")
        result[path] = identity
    return result


def sample_process(pid: int, expected_birth: int, expected_exe_sha256: str,
                   expected_libraries: dict[str, str], proc_root: Path = Path("/proc")) -> dict:
    """Fail closed if executable, every mapped ggml object, or PID identity differs."""
    if not expected_libraries or not any("libggml-hip" in Path(p).name for p in expected_libraries):
        raise ValueError("expected HIP library missing")
    proc = proc_root / str(pid)
    if process_birth(proc) != expected_birth:
        raise ValueError("PID birth mismatch")
    before = ggml_maps((proc / "maps").read_text())
    if set(before) != set(expected_libraries):
        raise ValueError("mapped ggml coverage mismatch")
    if sha_file(proc / "exe") != expected_exe_sha256:
        raise ValueError("executable digest mismatch")
    for path, (dev, inode) in before.items():
        p = Path(path)
        st = p.stat()
        major, minor = (int(x, 16) for x in dev.split(":"))
        if (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino) != (major, minor, inode):
            raise ValueError("mapped file identity mismatch")
        if sha_file(p) != expected_libraries[path]:
            raise ValueError("mapped file digest mismatch")
        after = p.stat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns):
            raise ValueError("mapped file changed during sampling")
    if process_birth(proc) != expected_birth or ggml_maps((proc / "maps").read_text()) != before:
        raise ValueError("process or mappings changed during sampling")
    return {"pid": pid, "birth_ticks": expected_birth, "executable_sha256": expected_exe_sha256,
            "mapped_ggml": dict(expected_libraries),
            "scope": "one in-window process sample; not GPU residency or build provenance"}
