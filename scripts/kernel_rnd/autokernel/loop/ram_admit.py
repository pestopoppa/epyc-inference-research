"""RAM admission for DS41-class model-server starts (DS41-W8d).

Mirrors /mnt/raid0/llm/tmp/compute-flags/ram_admit.sh; switch to the tracked script once
OP-RAMADMIT-D9 lands (intended tracked path: root
`scripts/coordination/ram_admit.sh`). Stdlib only:
the observer reads /proc and NEVER signals processes or matches them for killing
(INC-20260731-broad-process-pattern-kills).

Admit iff
  * MemAvailable_GiB - need_GiB >= MARGIN_GIB (integer-floor GiB, like the script; earlyoom
    fires at 40 GiB available, so the margin is 64), and
  * for DS41-class launches, no live readable process maps (`/proc/<pid>/maps`) or names
    (`/proc/<pid>/cmdline`) the DS41-1017 shard, by full path or basename. Own pid and own
    ancestors are skipped (a launcher wrapper's argv necessarily names the model).

`guard_launch(argv)` is a read-only observation: it is a no-op for non-DS41 argv and raises
`RamAdmissionRefused` for a refused DS41-class launch. It never touches argv or env.
"""
from __future__ import annotations

import os
import fcntl
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

MARGIN_GIB = 64
DS41_NEED_GIB = 739
DS41_SHARD = ("/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/"
              "DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf")
_MODEL_FLAGS = ("-m", "--model", "-md", "--model-draft")


class RamAdmissionRefused(RuntimeError):
    """A DS41-class server start was refused by RAM admission (not a candidate result)."""

    def __init__(self, result: "AdmitResult"):
        super().__init__(f"RAM admission refused: {result.reason}")
        self.result = result


@dataclass(frozen=True)
class AdmitResult:
    admit: bool
    reason: str
    avail_gib: int
    need_gib: int
    holders: list = field(default_factory=list)  # [{"pid", "via", "artifact"}]
    unreadable: int = 0
    scanned: int = 0


def mem_available_gib(meminfo: str | os.PathLike = "/proc/meminfo") -> int:
    for line in Path(meminfo).read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1048576
    raise RuntimeError(f"MemAvailable not found in {meminfo}")


def _ancestors(proc_root: Path, pid: int) -> set[int]:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            stat = (proc_root / str(pid) / "stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])  # ppid follows state
        except (OSError, ValueError, IndexError):
            break
    return seen


def find_holders(artifacts: Sequence[str], *, proc_root: str | os.PathLike = "/proc",
                 self_pid: int | None = None) -> tuple[list[dict], int, int]:
    """(holders, unreadable, scanned): live processes naming/mapping any artifact."""
    root = Path(proc_root)
    skip = _ancestors(root, self_pid if self_pid is not None else os.getpid())
    skip.add(self_pid if self_pid is not None else os.getpid())
    holders: list[dict] = []
    unreadable = scanned = 0
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.name.isdigit() or int(entry.name) in skip:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            unreadable += 1
            continue
        scanned += 1
        try:
            maps = (entry / "maps").read_text(errors="replace")
        except OSError:
            maps = ""
        for art in artifacts:
            base = os.path.basename(art)
            via = ("cmdline" if (art in cmd or base in cmd)
                   else "maps" if (maps and (art in maps or base in maps)) else None)
            if via:
                holders.append({"pid": int(entry.name), "via": via, "artifact": base})
    return holders, unreadable, scanned


def check(need_gib: int, *, ds41: bool = False, meminfo: str | os.PathLike = "/proc/meminfo",
          proc_root: str | os.PathLike = "/proc", self_pid: int | None = None,
          margin_gib: int = MARGIN_GIB) -> AdmitResult:
    avail = mem_available_gib(meminfo)
    if avail - need_gib < margin_gib:
        return AdmitResult(False, f"MemAvailable {avail} GiB - need {need_gib} GiB < margin "
                                  f"{margin_gib} GiB", avail, need_gib)
    holders: list[dict] = []
    unreadable = scanned = 0
    if ds41:
        holders, unreadable, scanned = find_holders([DS41_SHARD], proc_root=proc_root,
                                                    self_pid=self_pid)
        if holders:
            who = ", ".join(f"pid={h['pid']} via={h['via']}" for h in holders[:5])
            return AdmitResult(False, f"DS41-class server live: {who}", avail, need_gib,
                               holders, unreadable, scanned)
    return AdmitResult(True, f"MemAvailable {avail} GiB, need {need_gib} GiB"
                             f"{', ds41 clear' if ds41 else ''}", avail, need_gib, holders,
                       unreadable, scanned)


def is_ds41_argv(argv: Sequence[object]) -> bool:
    """True iff the launch's model (-m/--model/-md) is a DeepSeek-V4.1 GGUF."""
    toks = [str(a) for a in argv]
    vals: list[str] = []
    for i, t in enumerate(toks):
        if t in _MODEL_FLAGS and i + 1 < len(toks):
            vals.append(toks[i + 1])
        elif t.startswith("--model="):
            vals.append(t.split("=", 1)[1])
    return any(os.path.basename(v).lower().startswith("deepseek-v4.1") for v in vals)


def guard_launch(argv: Sequence[object], **kw) -> AdmitResult | None:
    """Pre-Popen admission. None (no check) for non-DS41 argv; raises on refusal."""
    if not is_ds41_argv(argv):
        return None
    result = check(DS41_NEED_GIB, ds41=True, **kw)
    if not result.admit:
        raise RamAdmissionRefused(result)
    return result


# PROPOSED shared inode; the owner must provision this exact private directory and
# file before enabling any participating launcher. Never unlink/replace either.
DS41_LEASE_PATH = "/mnt/raid0/llm/ds41-admission/lease.lock"

def _trusted_ancestors(parent):
    if not os.path.isabs(parent):
        raise PermissionError("lease path must be absolute")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in parent.split("/"):
            if not part:
                continue
            if part in (".", ".."):
                raise PermissionError("invalid lease path")
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            st = os.fstat(fd)
            if st.st_uid not in (0, os.geteuid()) or st.st_mode & 0o022:
                raise PermissionError("lease ancestor must be trusted and not group/world writable")
        return fd
    except BaseException:
        os.close(fd)
        raise

def _lease_fd(path):
    parent, name = os.path.split(os.fspath(path))
    directory = _trusted_ancestors(parent)
    try:
        ds = os.fstat(directory)
        if ds.st_uid != os.geteuid() or stat.S_IMODE(ds.st_mode) != 0o700:
            raise PermissionError("lease directory must be owned by this UID and mode 0700")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        try:
            st = os.fstat(fd)
            current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            parent_now = os.stat(parent, follow_symlinks=False)
            if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid()
                    or stat.S_IMODE(st.st_mode) != 0o600 or st.st_nlink != 1
                    or (st.st_dev, st.st_ino) != (current.st_dev, current.st_ino)
                    or (ds.st_dev, ds.st_ino) != (parent_now.st_dev, parent_now.st_ino)):
                raise PermissionError("untrusted or replaced lease inode")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BaseException:
            os.close(fd)
            raise
    finally:
        os.close(directory)

def popen(argv, *, popen=None, **kwargs):
    """Start DS41 under a lifetime lease; non-DS41 calls pass through unchanged.

    The child must retain the passed FD through all exec wrappers and until exit.
    The inherited open description retains the flock even if the parent dies.
    """
    spawn = subprocess.Popen if popen is None else popen
    if not is_ds41_argv(argv):
        return spawn(argv, **kwargs)
    if "pass_fds" in kwargs:
        raise ValueError("caller must not supply pass_fds to DS41 admission")
    try:
        fd = _lease_fd(DS41_LEASE_PATH)
    except OSError as exc:
        raise RamAdmissionRefused(AdmitResult(False, f"DS41 lease unavailable: {exc}", 0, DS41_NEED_GIB)) from exc
    try:
        guard_launch(argv)
        return spawn(argv, pass_fds=(fd,), **kwargs)
    finally:
        os.close(fd)  # Never unlock the shared open description.


def exec_ds41(argv):
    """CLI model-exec choke point for a DS41 shell launcher; replaces this PID."""
    if not is_ds41_argv(argv):
        raise ValueError("DS41 exec requires a recognized model argv")
    try:
        fd = _lease_fd(DS41_LEASE_PATH)
    except OSError as exc:
        raise RamAdmissionRefused(AdmitResult(False, f"DS41 lease unavailable: {exc}", 0, DS41_NEED_GIB)) from exc
    try:
        guard_launch(argv)
        os.set_inheritable(fd, True)
        os.execvpe(argv[0], argv, os.environ)
    finally:
        os.close(fd)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        sys.exit("usage: ram_admit.py -- command [args ...]")
    try:
        exec_ds41(sys.argv[2:])
    except (RamAdmissionRefused, OSError, ValueError) as exc:
        print(f"refuse: {exc}", file=sys.stderr)
        sys.exit(4)
