#!/usr/bin/env python3
"""ONE scratch-resource registry for the AutoKernel loop (operator directive 2026-09-26).

Disk management is codified at the FLOW level, not re-implemented per feature. Every
scratch directory, detached worktree and scratch file the loop creates is allocated
through a scope, carries an ownership marker, and is journalled; the scope releases it
on exit (normal return, exception, KeyboardInterrupt, or the loop's SIGTERM stop path,
which unwinds through the same `with` blocks). Whatever a hard kill leaves behind is
collected by `sweep()` at the next run / batch / iteration start -- but only when it is
MARKED and its owner is provably gone. Unmarked paths are never touched.

    reg = ScratchRegistry(state_dir / "scratch", owner={"campaign": ..., "state_dir": ...,
                          "run_id": ..., "pid": os.getpid()}, min_free_bytes=50 * GB)
    with reg.scope("run", name=run_id) as run:
        reg.sweep()
        with reg.scope("iteration", name="it-7") as it:
            wt = it.worktree(repo, base_commit, "author-a")      # git worktree add --detach
            build = it.dir("ak-check-build", "author-a")
            if not reg.ensure_free(20 * GB):
                ...degrade (best-of N -> 1, op-test -> compile-only)...

Markers: a DIRECTORY carries `<dir>/.ak-scratch-owner`; a FILE or WORKTREE carries a
sidecar `<path>.ak-scratch-owner` (a marker inside a worktree would be an untracked file
an actor's `git add -A` could commit). Journal: `<root>/scratch-journal.jsonl`.

NEVER `git worktree prune` OR `git gc` (see pool.py: on 2026-08-12 that destroyed five
live lanes). A worktree whose directory is gone has its ONE admin entry removed, and
only after its `gitdir` file is verified to point at the registered path. `_git`
refuses both subcommands; `test_scratch.py` greps the package for them.

Stdlib only.
"""
from __future__ import annotations

import fcntl
import itertools
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = "epyc.autokernel.scratch.v1"
MARKER = ".ak-scratch-owner"
SIDECAR_SUFFIX = ".ak-scratch-owner"
JOURNAL = "scratch-journal.jsonl"
LEVELS = ("run", "batch", "iteration", "call")
KEEP_MODES = ("none", "failed", "all")
GB = 10 ** 9
DEFAULT_MIN_FREE_GB = 50
DEFAULT_MIN_FREE_BYTES = DEFAULT_MIN_FREE_GB * GB
RESOURCES = ("dir", "worktree", "file")

# The ban, as data. `_git` refuses any argv that contains one of these subcommand
# sequences; the assertion below keeps the table from being emptied by an edit.
_FORBIDDEN_GIT = (("worktree", "prune"), ("gc",))
assert _FORBIDDEN_GIT and all(_FORBIDDEN_GIT), "scratch: the prune/gc ban must stay armed"

_SAFE = re.compile(r"[^A-Za-z0-9._@+-]+")


class ScratchError(RuntimeError):
    """A registry operation could not be completed."""


class ScratchRefused(ScratchError):
    """The registry refused to touch a path it cannot prove it owns."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe(name: str) -> str:
    out = _SAFE.sub("-", str(name)).strip("-.")
    if not out:
        raise ValueError(f"scratch: unusable name {name!r}")
    return out[:120]


def _pid_start(pid: int) -> str | None:
    """Kernel start time of `pid` (field 22 of /proc/<pid>/stat), or None."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    try:
        return raw[raw.rindex(")") + 2:].split()[19]
    except (ValueError, IndexError):
        return None


def pid_alive(pid: Any, start: str | None = None) -> bool:
    """True when `pid` exists (and, when `start` was recorded, is the SAME process --
    a recycled pid with a different start time is dead for our purposes)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if start is not None:
        now = _pid_start(pid)
        if now is not None and now != start:
            return False
    return True


def _git(*argv: str, cwd: Path | str | None = None,
         timeout: float = 300) -> subprocess.CompletedProcess:
    for banned in _FORBIDDEN_GIT:
        n = len(banned)
        if any(tuple(argv[i:i + n]) == banned for i in range(len(argv) - n + 1)):
            raise ScratchRefused(f"scratch: `git {' '.join(banned)}` is banned in the loop")
    return subprocess.run(["git", *argv], cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def _tree_bytes(path: Path) -> int:
    """Apparent size of a path (symlinks not followed). Best effort, never raises."""
    total = 0
    try:
        st = path.lstat()
    except OSError:
        return 0
    if not path.is_dir() or path.is_symlink():
        return st.st_size
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _marker_path(resource: str, path: Path) -> Path:
    if resource == "dir":
        return path / MARKER
    return path.with_name(path.name + SIDECAR_SUFFIX)


def read_marker(resource: str, path: Path) -> dict | None:
    try:
        data = json.loads(_marker_path(resource, Path(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == SCHEMA else None


class Scope:
    """One `with registry.scope(...)` block. Allocators live here; everything allocated
    is released when the block exits (innermost scope first)."""

    def __init__(self, registry: "ScratchRegistry", level: str, name: str,
                 parent: "Scope | None") -> None:
        self.registry = registry
        self.level = level
        self.name = name
        self.parent = parent
        self.id = f"{level}-{next(registry._seq):06d}-{_safe(name)}"
        self.children: list[Scope] = []
        self.resources: list[dict] = []
        self.failed = False
        self.closed = False
        self._tmp: Path | None = None

    # -- lifecycle --------------------------------------------------------------------
    def __enter__(self) -> "Scope":
        self.registry._stack().append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        stack = self.registry._stack()
        if self in stack:
            stack.remove(self)
        self.close(failed=exc_type is not None)
        return False

    def mark_failed(self) -> None:
        """Flag this scope failed without raising (e.g. an iteration that returned a
        failure outcome) so `--scratch-keep failed` retains it."""
        self.failed = True

    def close(self, *, failed: bool = False) -> None:
        self.registry._close(self, failed=failed or self.failed)

    def scope(self, level: str, name: str) -> "Scope":
        """An explicit child scope (thread-independent nesting)."""
        return self.registry.scope(level, name=name, parent=self)

    @property
    def info(self) -> dict:
        return {"level": self.level, "name": self.name, "id": self.id}

    # -- allocators -------------------------------------------------------------------
    def dir(self, kind: str, name: str, *, at: Path | str | None = None) -> Path:
        """A fresh, marked scratch directory (`<root>/<kind>/<name>` unless `at`)."""
        return self.registry._allocate(self, "dir", kind, name, at=at)

    def tmpdir(self, name: str = "") -> Path:
        """This scope's system-temp directory (`<root>/tmp/<scope id>`), allocated once
        per scope and released with it."""
        if self._tmp is None:
            self._tmp = self.dir("tmp", self.id + (f"-{_safe(name)}" if name else ""))
        return self._tmp

    def tmp_env(self, base: dict | None = None) -> dict:
        """A subprocess environment whose TMPDIR/TMP/TEMP point at `tmpdir()`, so stray
        tempfiles from actors, compilers and ak-check land in a scoped, released dir."""
        env = dict(os.environ if base is None else base)
        tmp = str(self.tmpdir())
        env.update(TMPDIR=tmp, TMP=tmp, TEMP=tmp)
        return env

    def file(self, kind: str, name: str, *, at: Path | str | None = None) -> Path:
        """A marked scratch file path (parent created, file NOT created); released by
        unlink. Its marker is the sidecar `<path>.ak-scratch-owner`."""
        return self.registry._allocate(self, "file", kind, name, at=at)

    def worktree(self, repo: Path | str, base_commit: str, name: str, *,
                 at: Path | str | None = None) -> Path:
        """`git worktree add --detach <path> <base_commit>` in `repo`, marked by a
        sidecar; released by `git worktree remove --force` (dirty trees included)."""
        return self.registry._allocate(self, "worktree", "worktrees", name, at=at,
                                       repo=Path(repo), base_commit=base_commit)


class ScratchRegistry:
    """The per-run registry. `root` lives under the campaign state dir."""

    def __init__(self, root: Path | str, owner: dict, min_free_bytes: int =
                 DEFAULT_MIN_FREE_BYTES, *, keep: str = "none") -> None:
        if keep not in KEEP_MODES:
            raise ValueError(f"scratch: keep must be one of {KEEP_MODES}, not {keep!r}")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        owner = dict(owner or {})
        owner.setdefault("pid", os.getpid())
        owner.setdefault("run_id", uuid.uuid4().hex[:12])
        owner.setdefault("pid_start", _pid_start(int(owner["pid"])))
        owner = {k: (str(v) if isinstance(v, Path) else v) for k, v in owner.items()}
        self.owner = owner
        self.run_id = str(owner["run_id"])
        self.min_free_bytes = int(min_free_bytes)
        self.keep = keep
        self.instance = uuid.uuid4().hex
        self.journal_path = self.root / JOURNAL
        self._seq = itertools.count(1)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._active: dict[str, Scope] = {}
        self._stats = {"allocated": 0, "released": 0, "retained": 0,
                       "release_failures": 0, "bytes_freed": 0, "bytes_retained": 0,
                       "sweep_removed": 0, "sweep_bytes_freed": 0, "sweep_skipped": 0,
                       "sweeps": 0, "guard_checks": 0, "guard_refusals": 0}
        self._live: dict[str, dict] = {}

    # -- scopes -----------------------------------------------------------------------
    def _stack(self) -> list[Scope]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        return stack

    def current(self) -> Scope | None:
        """The innermost open scope on THIS thread."""
        stack = self._stack()
        return stack[-1] if stack else None

    def scope(self, level: str, name: str = "", *, parent: Scope | None = None) -> Scope:
        """`with registry.scope("iteration", name="it-7") as s:`. `parent` defaults to
        the innermost open scope on this thread."""
        if level not in LEVELS:
            raise ValueError(f"scratch: scope level must be one of {LEVELS}, not {level!r}")
        parent = parent if parent is not None else self.current()
        scope = Scope(self, level, name or level, parent)
        with self._lock:
            self._active[scope.id] = scope
            if parent is not None:
                parent.children.append(scope)
        return scope

    def _close(self, scope: Scope, *, failed: bool) -> None:
        with self._lock:
            if scope.closed:
                return
            scope.closed = True
        # Innermost first: any child still open (a thread-leaked or non-`with` scope)
        # is closed before this scope's own resources go.
        for child in reversed(list(scope.children)):
            child.close(failed=failed)
        retain = self.keep == "all" or (self.keep == "failed" and failed)
        for res in reversed(list(scope.resources)):
            if retain:
                self._retain(res, scope, failed=failed)
            else:
                self._release(res, reason="scope-exit")
        with self._lock:
            self._active.pop(scope.id, None)
        self._journal({"event": "scope_close", "scope": scope.info, "failed": failed,
                       "retained": retain, "resources": len(scope.resources)})

    # -- allocation -------------------------------------------------------------------
    def _default_path(self, resource: str, kind: str, name: str) -> Path:
        return self.root / _safe(kind) / _safe(name)

    def _allocate(self, scope: Scope, resource: str, kind: str, name: str, *,
                  at: Path | str | None = None, repo: Path | None = None,
                  base_commit: str | None = None) -> Path:
        if scope.closed:
            raise ScratchError(f"scratch: scope {scope.id} is closed")
        path = Path(at).absolute() if at is not None else self._default_path(resource, kind, name)
        self._guard_path(path)
        if os.path.lexists(path) or os.path.lexists(_marker_path(resource, path)):
            self._reclaim_stale(resource, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = {"schema": SCHEMA, "owner": self.owner, "scope": scope.info,
                  "kind": kind, "resource": resource, "created_at": _now(),
                  "pid": self.owner["pid"], "path": str(path),
                  "registry_root": str(self.root), "instance": self.instance,
                  "retained": False}
        if resource == "dir":
            path.mkdir()
            self._write_marker(resource, path, marker)
        elif resource == "file":
            self._write_marker(resource, path, marker)
        else:
            assert repo is not None and base_commit
            marker["repo"] = str(Path(repo).resolve())
            marker["base_commit"] = base_commit
            # Marker FIRST: a crash between the two leaves a marked, collectible path.
            self._write_marker(resource, path, marker)
            done = _git("-C", str(repo), "worktree", "add", "--detach", str(path),
                        base_commit)
            if done.returncode != 0:
                _marker_path(resource, path).unlink(missing_ok=True)
                raise ScratchError(f"scratch: git worktree add failed: "
                                   f"{(done.stderr or done.stdout).strip()[-400:]}")
            admin = self._admin_dir_of(path)
            if admin is not None:
                marker["admin_dir"] = str(admin)
                self._write_marker(resource, path, marker)
        with self._lock:
            scope.resources.append(marker)
            self._live[str(path)] = marker
            self._stats["allocated"] += 1
        self._journal({"event": "allocate", "resource": resource, "kind": kind,
                       "path": str(path), "scope": scope.info,
                       **({"repo": marker["repo"], "base_commit": base_commit}
                          if resource == "worktree" else {})})
        return path

    def _guard_path(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved == Path("/") or resolved == self.root or resolved in self.root.parents:
            raise ScratchRefused(f"scratch: refusing to allocate at {path}")

    def _reclaim_stale(self, resource: str, path: Path) -> None:
        """A name collision: reclaim it only when it is ours-to-collect; else refuse."""
        marker = read_marker(resource, path)
        why = self._collectible(marker) if marker else None
        if marker is None or why is None or marker.get("resource") != resource:
            raise FileExistsError(f"scratch: {path} exists and is not a collectible "
                                  f"registry resource")
        self._release(marker, reason=f"reclaim:{why}", sweep=True)
        if os.path.lexists(path):
            raise FileExistsError(f"scratch: could not reclaim {path}")

    def _write_marker(self, resource: str, path: Path, marker: dict) -> None:
        target = _marker_path(resource, path)
        tmp = target.with_name(target.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, target)

    # -- release ----------------------------------------------------------------------
    def _retain(self, res: dict, scope: Scope, *, failed: bool) -> None:
        path = Path(res["path"])
        res = dict(res, retained=True, retained_at=_now(), retained_failed=failed)
        try:
            self._write_marker(res["resource"], path, res)
        except OSError:
            pass
        size = _tree_bytes(path)
        with self._lock:
            self._live.pop(str(path), None)
            self._stats["retained"] += 1
            self._stats["bytes_retained"] += size
        self._journal({"event": "retain", "resource": res["resource"], "path": str(path),
                       "scope": scope.info, "bytes": size, "keep": self.keep})

    def _release(self, res: dict, *, reason: str, sweep: bool = False) -> bool:
        path = Path(res["path"])
        resource = res["resource"]
        size = _tree_bytes(path)
        try:
            if resource == "dir":
                self._release_dir(path)
            elif resource == "file":
                path.unlink(missing_ok=True)
                _marker_path("file", path).unlink(missing_ok=True)
            else:
                self.remove_worktree(res.get("repo"), path)
        except (OSError, ScratchError, subprocess.SubprocessError) as exc:
            with self._lock:
                self._stats["release_failures"] += 1
            self._journal({"event": "release_failed", "resource": resource,
                           "path": str(path), "reason": reason,
                           "error": f"{type(exc).__name__}: {exc}"[:500]})
            return False
        with self._lock:
            self._live.pop(str(path), None)
            if sweep:
                self._stats["sweep_removed"] += 1
                self._stats["sweep_bytes_freed"] += size
            else:
                self._stats["released"] += 1
            self._stats["bytes_freed"] += size
        self._journal({"event": "sweep_remove" if sweep else "release",
                       "resource": resource, "path": str(path), "reason": reason,
                       "bytes": size})
        return True

    def _release_dir(self, path: Path) -> None:
        if not os.path.lexists(path):
            return
        marker = read_marker("dir", path)
        if marker is None or marker.get("registry_root") != str(self.root):
            raise ScratchRefused(f"scratch: {path} carries no marker of this registry")
        self._guard_path(path)
        shutil.rmtree(path)

    @staticmethod
    def _admin_dir_of(path: Path) -> Path | None:
        try:
            text = (Path(path) / ".git").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not text.startswith("gitdir:"):
            return None
        admin = Path(text[len("gitdir:"):].strip())
        if not admin.is_absolute():
            admin = (Path(path) / admin)
        return admin.resolve()

    def remove_worktree(self, repo: Path | str | None, path: Path | str) -> None:
        """Remove ONE registry-owned worktree. Refuses anything without this registry's
        sidecar marker. `git worktree remove --force` when the directory exists; when it
        is already gone, the single admin entry recorded in the marker is removed after
        its `gitdir` is verified to point at `path` -- never `prune`."""
        path = Path(path).absolute()
        marker = read_marker("worktree", path)
        if (marker is None or marker.get("resource") != "worktree"
                or marker.get("registry_root") != str(self.root)
                or Path(marker.get("path", "")) != path):
            raise ScratchRefused(f"scratch: {path} is not a marked worktree of this registry")
        repo = Path(repo or marker.get("repo") or "")
        if os.path.lexists(path):
            done = _git("-C", str(repo), "worktree", "remove", "--force", str(path))
            if done.returncode != 0 and os.path.lexists(path):
                raise ScratchError(f"scratch: git worktree remove failed for {path}: "
                                   f"{(done.stderr or done.stdout).strip()[-400:]}")
        if not os.path.lexists(path):
            self._remove_admin_entry(marker, path)
        _marker_path("worktree", path).unlink(missing_ok=True)

    def _remove_admin_entry(self, marker: dict, path: Path) -> None:
        admin = marker.get("admin_dir")
        if not admin:
            return
        admin = Path(admin)
        if not admin.is_dir():
            return
        if admin.parent.name != "worktrees":
            raise ScratchRefused(f"scratch: {admin} is not a worktree admin entry")
        try:
            gitdir = (admin / "gitdir").read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ScratchRefused(f"scratch: {admin} has no readable gitdir") from exc
        want = {str(path / ".git"), str((path / ".git").resolve())}
        pointed = Path(gitdir) if Path(gitdir).is_absolute() else admin / gitdir
        if gitdir not in want and str(pointed.resolve()) not in want:
            raise ScratchRefused(f"scratch: admin entry {admin} points at {gitdir}, not "
                                 f"{path / '.git'}; refusing")
        shutil.rmtree(admin)
        self._journal({"event": "admin_entry_removed", "admin_dir": str(admin),
                       "path": str(path)})

    # -- sweep ------------------------------------------------------------------------
    def _collectible(self, marker: dict | None) -> str | None:
        """Why a marked resource may be collected now, or None to leave it."""
        if not marker or marker.get("registry_root") != str(self.root):
            return None
        owner = marker.get("owner") or {}
        pid, start = owner.get("pid", marker.get("pid")), owner.get("pid_start")
        alive = pid_alive(pid, start)
        same_run = str(owner.get("run_id")) == self.run_id
        if marker.get("retained"):
            if alive and not same_run and pid != os.getpid():
                return None  # another live run's debugging scratch
            return "retained-keep-none" if self.keep == "none" else None
        if not alive:
            return "owner-dead"
        if not same_run:
            # A live pid under another run id: a concurrent run keeps its scratch; the
            # same process under an old run id is a finished earlier run.
            return "stale-run-same-process" if pid == os.getpid() else None
        if marker.get("instance") == self.instance:
            scope_id = (marker.get("scope") or {}).get("id")
            with self._lock:
                if scope_id not in self._active:
                    return "orphan-scope"
        return None

    def _candidates(self) -> list[tuple[str, Path]]:
        seen: dict[str, str] = {}
        try:
            kinds = [p for p in self.root.iterdir() if p.is_dir() and not p.is_symlink()]
        except OSError:
            kinds = []
        for kind in kinds:
            try:
                entries = list(kind.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.name.endswith(SIDECAR_SUFFIX):
                    target = entry.with_name(entry.name[:-len(SIDECAR_SUFFIX)])
                    m = read_marker("file", target)
                    if m and m.get("resource") in ("file", "worktree"):
                        seen[str(target)] = m["resource"]
                elif entry.is_dir() and not entry.is_symlink() and (entry / MARKER).is_file():
                    seen[str(entry)] = "dir"
        # Resources allocated `at=` elsewhere are found through the journal.
        for path, resource in self._journal_live().items():
            seen.setdefault(path, resource)
        return sorted((res, Path(p)) for p, res in seen.items())

    def _journal_live(self) -> dict[str, str]:
        live: dict[str, str] = {}
        try:
            lines = self.journal_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return live
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            path = row.get("path")
            if not path:
                continue
            if row.get("event") == "allocate":
                live[path] = row.get("resource", "dir")
            elif row.get("event") in ("release", "sweep_remove"):
                live.pop(path, None)
        return live

    def sweep(self) -> dict:
        """Collect marked resources whose owner is gone (dead pid, or a finished run in
        this process), retained scratch when keep=none, and this registry's own orphans.
        Never touches an unmarked path. Journals every action."""
        removed, skipped, failed = [], 0, []
        for resource, path in self._candidates():
            marker = read_marker(resource, path)
            if marker is None:
                continue  # unmarked (or marker gone): never ours to touch
            why = self._collectible(marker)
            if why is None:
                skipped += 1
                continue
            if self._release(marker, reason=f"sweep:{why}", sweep=True):
                removed.append({"path": str(path), "resource": resource, "reason": why})
            else:
                failed.append(str(path))
        with self._lock:
            self._stats["sweeps"] += 1
            self._stats["sweep_skipped"] += skipped
        result = {"removed": removed, "skipped": skipped, "failed": failed}
        self._journal({"event": "sweep", "removed": len(removed), "skipped": skipped,
                       "failed": len(failed)})
        return result

    # -- disk guard -------------------------------------------------------------------
    def free_bytes(self) -> int:
        return shutil.disk_usage(self.root).free

    def ensure_free(self, bytes_needed: int = 0) -> bool:
        """True when `bytes_needed` fits while leaving `min_free_bytes` free. False means
        the caller MUST degrade (best-of N->1, op-test->compile-only)."""
        free = self.free_bytes()
        ok = free - int(bytes_needed) >= self.min_free_bytes
        with self._lock:
            self._stats["guard_checks"] += 1
            if not ok:
                self._stats["guard_refusals"] += 1
        if not ok:
            self._journal({"event": "guard_refused", "free_bytes": free,
                           "bytes_needed": int(bytes_needed),
                           "min_free_bytes": self.min_free_bytes})
        return ok

    # -- stats / journal --------------------------------------------------------------
    def stats(self, *, measure_live: bool = False) -> dict:
        """Counters for metrics rows and loop-status. `bytes_live` walks the live
        resources, so it is only computed when asked."""
        with self._lock:
            out = dict(self._stats)
            live = list(self._live)
        out["live"] = len(live)
        out["root"] = str(self.root)
        out["keep"] = self.keep
        out["min_free_bytes"] = self.min_free_bytes
        if measure_live:
            out["bytes_live"] = sum(_tree_bytes(Path(p)) for p in live)
            out["bytes_allocated"] = out["bytes_freed"] + out["bytes_live"] + out["bytes_retained"]
        return out

    def _journal(self, row: dict) -> None:
        row = {"ts": _now(), "run_id": self.run_id, "pid": self.owner.get("pid"), **row}
        line = json.dumps(row, sort_keys=True, default=str) + "\n"
        try:
            with open(self.journal_path, "a", encoding="utf-8") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    fh.write(line)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            pass  # the journal is evidence of cleanup, never a reason to fail it


# -- CLI knobs --------------------------------------------------------------------------
def add_arguments(parser) -> None:
    """`--scratch-min-free-gb` and `--scratch-keep` for run.py."""
    parser.add_argument("--scratch-min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
                        help="free-space floor (GB) the scratch guard keeps; below it "
                             "callers degrade (best-of N->1, op-test->compile-only)")
    parser.add_argument("--scratch-keep", choices=KEEP_MODES, default="none",
                        help="retain iteration scratch for debugging: none (default), "
                             "failed, or all; retained scratch stays marked and the next "
                             "sweep under keep=none collects it")


def from_args(args, *, root: Path | str, owner: dict) -> ScratchRegistry:
    return ScratchRegistry(root, owner,
                           int(float(getattr(args, "scratch_min_free_gb",
                                             DEFAULT_MIN_FREE_GB)) * GB),
                           keep=getattr(args, "scratch_keep", "none") or "none")


__all__ = ["DEFAULT_MIN_FREE_BYTES", "DEFAULT_MIN_FREE_GB", "GB", "JOURNAL", "KEEP_MODES",
           "LEVELS", "MARKER", "SIDECAR_SUFFIX", "Scope", "ScratchError", "ScratchRefused",
           "ScratchRegistry", "add_arguments", "from_args", "pid_alive", "read_marker"]
