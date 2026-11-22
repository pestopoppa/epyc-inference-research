"""Source build log verification, split verbatim from source_build_execution.py (code-budget split)."""
from __future__ import annotations
import hashlib
import os
import stat
from typing import Any, Mapping


class SourceBuildExecutionRefused(RuntimeError):
    pass


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise SourceBuildExecutionRefused(f"{label} must be lowercase SHA-256")
    return value


def _verify_log(path: str, expected_sha: str, expected_identity: Any,
                maximum: int) -> None:
    """Reopen the exact evaluator-owned log named by the admitted request."""
    _sha(expected_sha, "build log digest")
    fields = {"device", "inode", "uid", "mode", "nlink", "size", "mtime_ns", "ctime_ns"}
    if not isinstance(expected_identity, Mapping) or set(expected_identity) != fields:
        raise SourceBuildExecutionRefused("build log identity fields differ")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SourceBuildExecutionRefused("build log is unavailable") from exc
    try:
        before = os.fstat(fd)
        identity = {
            "device": before.st_dev, "inode": before.st_ino, "uid": before.st_uid,
            "mode": stat.S_IMODE(before.st_mode), "nlink": before.st_nlink,
            "size": before.st_size, "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
        if (identity != dict(expected_identity) or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1 or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 1 or before.st_size > maximum):
            raise SourceBuildExecutionRefused("build log differs from its owned identity")
        digest, remaining = hashlib.sha256(), before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise SourceBuildExecutionRefused("build log ended before its owned size")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        linked = os.stat(path, follow_symlinks=False)
        def current(row):
            return {"device": row.st_dev, "inode": row.st_ino, "uid": row.st_uid,
                    "mode": stat.S_IMODE(row.st_mode), "nlink": row.st_nlink,
                    "size": row.st_size, "mtime_ns": row.st_mtime_ns,
                    "ctime_ns": row.st_ctime_ns}
        if current(after) != identity or current(linked) != identity:
            raise SourceBuildExecutionRefused("build log changed during parent verification")
        if digest.hexdigest() != expected_sha:
            raise SourceBuildExecutionRefused("build log bytes differ from its receipt")
    except OSError as exc:
        raise SourceBuildExecutionRefused("build log verification failed") from exc
    finally:
        os.close(fd)
