"""Bounded reclamation of reproducible build caches from retired serial states.

This is intentionally not a general tmp cleaner.  It recognizes only build roots
named by a validated AutoKernel continuation, never treats a dirty Git worktree as
evidence that generated build output is unique, and never removes receipts, logs,
patches, source worktrees, stores, anchors, or the current state's cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Any, Sequence


PLAN_SCHEMA = "epyc.autokernel.serial_build_retention_plan.v1"
RETRY_SCHEMA = "epyc.autokernel.serial_build_retention_retry.v1"
RETRY_FILENAME = "build-retention-retry.json"
DEFAULT_TRIGGER_FREE_BYTES = 400 * 1024 ** 3
DEFAULT_TARGET_FREE_BYTES = 500 * 1024 ** 3
DEFAULT_RECENT_STATE_CACHES = 1
DEFAULT_MAX_BUILD_DIRS = 8


class BuildRetentionRefused(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BuildCache:
    state_root: Path
    build_root: Path
    source_worktree: Path
    source_commit: str
    recipe_digest: str
    bytes: int
    state_mtime_ns: int


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _directory_bytes(path: Path) -> int:
    total = 0
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs
                   if not (Path(base) / name).is_symlink()]
        for name in files:
            item = Path(base) / name
            try:
                if not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    return total


def _option(argv: Sequence[str], name: str) -> str | None:
    values = []
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif item.startswith(name + "="):
            values.append(item[len(name) + 1:])
    return values[-1] if values else None


def _load_json(path: Path, *, maximum: int = 8 * 1024 * 1024) -> Any:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or not 0 < info.st_size <= maximum):
        raise BuildRetentionRefused(f"unbounded or non-regular record: {path}")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise BuildRetentionRefused(f"record changed while read: {path}")
    return json.loads(raw)


def _write_retry(root: Path, cache: BuildCache) -> None:
    path = root / RETRY_FILENAME
    temporary = root / f".{RETRY_FILENAME}.{os.getpid()}.{time.time_ns()}"
    body = {"schema": RETRY_SCHEMA, "build_root": str(cache.build_root),
            "source_commit": cache.source_commit,
            "recipe_digest": cache.recipe_digest}
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(body, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _open_inactive_state(root: Path) -> tuple[dict[str, Any], int, int] | None:
    """Open and return a settled state plus its still-held lock descriptor."""
    lock_path = root / "serial.lock"
    state_path = root / "serial-state.json"
    if not lock_path.is_file() or not state_path.is_file():
        return None
    descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        state = _load_json(state_path)
        # A dead child with an uncleared active record is intentionally retained:
        # the ordinary supervisor still owes crash reconciliation.
        if (not isinstance(state, dict)
                or state.get("schema") != "epyc.autokernel.serial_run.v1"
                or state.get("active") is not None):
            os.close(descriptor)
            return None
        return state, state_path.stat().st_mtime_ns, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _inactive_state(root: Path) -> tuple[dict[str, Any], int] | None:
    opened = _open_inactive_state(root)
    if opened is None:
        return None
    state, mtime_ns, descriptor = opened
    os.close(descriptor)
    return state, mtime_ns


def _git_commit_exists(worktree: Path, commit: str) -> bool:
    if (len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)):
        return False
    import subprocess
    done = subprocess.run(
        ["git", "-C", str(worktree), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    return done.returncode == 0


def _cache_from_state(root: Path, state: dict[str, Any], mtime_ns: int) -> BuildCache | None:
    references = state.get("last_results")
    if not isinstance(references, dict) or not references:
        return None
    continuations = []
    for reference in references.values():
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            return None
        path = Path(reference["path"])
        try:
            path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            return None
        try:
            info = path.lstat()
        except OSError:
            return None
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or not 0 < info.st_size <= 8 * 1024 * 1024):
            return None
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            return None
        continuations.append(json.loads(raw))
    build_roots, worktrees, commits, recipe_inputs = set(), set(), set(), []
    protected = set()
    for row in continuations:
        argv = row.get("input_argv")
        anchor = row.get("current_anchor")
        cor = row.get("cor_anchor")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            return None
        build = _option(argv, "--worker-build-root")
        worktree = _option(argv, "--worktree")
        if not build or not worktree or not isinstance(anchor, dict):
            return None
        build_roots.add(str(Path(build).resolve()))
        worktrees.add(str(Path(worktree).resolve()))
        commits.add(anchor.get("commit"))
        for item in (anchor, cor):
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                protected.add(str(Path(item["path"]).resolve()))
        # This binds the rebuild recipe without copying large generated output.
        recipe_inputs.append({
            "stable_argv": [item for item in argv
                            if item not in {"--resume-run", "--out", "--iterations"}],
            "config_digest": state.get("config_digest"),
        })
    if len(build_roots) != 1 or len(worktrees) != 1 or len(commits) != 1:
        return None
    build_root = Path(next(iter(build_roots)))
    worktree = Path(next(iter(worktrees)))
    commit = next(iter(commits))
    target_root = root / "targets"
    if not target_root.is_dir() or target_root.is_symlink():
        return None
    try:
        build_root.relative_to(target_root.resolve())
    except ValueError:
        return None
    if (not build_root.is_dir() or build_root.is_symlink()
            or any(Path(item) == build_root or build_root in Path(item).parents
                   for item in protected)
            or not _git_commit_exists(worktree, commit)):
        return None
    recipe_digest = _digest(recipe_inputs)
    lanes = [item for item in build_root.iterdir() if item.is_dir() and not item.is_symlink()]
    generated = bool(lanes) and all((lane / "CMakeCache.txt").is_file() for lane in lanes)
    retry = None
    try:
        retry = _load_json(root / RETRY_FILENAME, maximum=4096)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    retry_matches = retry == {
        "schema": RETRY_SCHEMA, "build_root": str(build_root),
        "source_commit": commit, "recipe_digest": recipe_digest}
    if not generated and not retry_matches:
        return None
    return BuildCache(root, build_root, worktree, commit,
                      recipe_digest, _directory_bytes(build_root), mtime_ns)


def plan(parent: Path, current: Path, *, free_bytes: int | None = None,
         trigger_free_bytes: int = DEFAULT_TRIGGER_FREE_BYTES,
         target_free_bytes: int = DEFAULT_TARGET_FREE_BYTES,
         recent_state_caches: int = DEFAULT_RECENT_STATE_CACHES,
         max_build_dirs: int = DEFAULT_MAX_BUILD_DIRS) -> dict[str, Any]:
    """Return an exact bounded plan. No bytes are removed here."""
    if (trigger_free_bytes < 0 or target_free_bytes < trigger_free_bytes
            or recent_state_caches < 0 or max_build_dirs < 1):
        raise BuildRetentionRefused("invalid build-retention policy")
    parent, current = parent.resolve(), current.resolve()
    free = shutil.disk_usage(parent).free if free_bytes is None else free_bytes
    caches = []
    for root in sorted(parent.iterdir()):
        if root.resolve() == current or not root.is_dir() or root.is_symlink():
            continue
        inactive = _inactive_state(root)
        if inactive is None:
            continue
        cache = _cache_from_state(root, *inactive)
        if cache is not None:
            caches.append(cache)
    caches.sort(key=lambda item: (item.state_mtime_ns, str(item.state_root)), reverse=True)
    recent = caches[:recent_state_caches]
    eligible = list(reversed(caches[recent_state_caches:]))
    selected = []
    projected = free
    if free < trigger_free_bytes:
        for cache in eligible:
            if len(selected) >= max_build_dirs or projected >= target_free_bytes:
                break
            selected.append(cache)
            projected += cache.bytes
    body = {
        "schema": PLAN_SCHEMA,
        "created_at_ns": time.time_ns(),
        "parent": str(parent),
        "current_state": str(current),
        "free_bytes_before": free,
        "trigger_free_bytes": trigger_free_bytes,
        "target_free_bytes": target_free_bytes,
        "recent_state_caches": recent_state_caches,
        "max_build_dirs": max_build_dirs,
        "protected_recent": [str(item.build_root) for item in recent],
        "selected": [{"state_root": str(item.state_root),
                      "build_root": str(item.build_root),
                      "source_worktree": str(item.source_worktree),
                      "source_commit": item.source_commit,
                      "recipe_digest": item.recipe_digest,
                      "bytes": item.bytes,
                      "generated_output_dirty_status_ignored": True}
                     for item in selected],
        "projected_free_bytes": projected,
    }
    return body | {"plan_digest": _digest(body)}


def execute(plan_body: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Revalidate every selected root, quarantine by rename, then remove it."""
    body = dict(plan_body)
    supplied = body.pop("plan_digest", None)
    if body.get("schema") != PLAN_SCHEMA or supplied != _digest(body):
        raise BuildRetentionRefused("retention plan digest differs")
    removed, skipped = [], []
    for row in body["selected"]:
        root, build = Path(row["state_root"]), Path(row["build_root"])
        try:
            opened = _open_inactive_state(root)
            if opened is None:
                skipped.append({"build_root": str(build),
                                "reason": "identity_or_lease_changed"})
                continue
            state, mtime_ns, descriptor = opened
            try:
                current = _cache_from_state(root, state, mtime_ns)
                if (current is None or current.build_root != build
                        or current.source_commit != row["source_commit"]
                        or current.recipe_digest != row["recipe_digest"]):
                    skipped.append({"build_root": str(build),
                                    "reason": "identity_or_lease_changed"})
                    continue
                if dry_run:
                    skipped.append({"build_root": str(build), "reason": "dry_run"})
                    continue
                quarantine = build.with_name(
                    f".{build.name}.retention-{os.getpid()}-{time.time_ns()}")
                before = build.stat(follow_symlinks=False)
                if not stat.S_ISDIR(before.st_mode):
                    raise OSError("build root is no longer a directory")
                os.rename(build, quarantine)
                moved = quarantine.stat(follow_symlinks=False)
                if ((before.st_dev, before.st_ino) != (moved.st_dev, moved.st_ino)
                        or not stat.S_ISDIR(moved.st_mode)):
                    raise OSError("quarantined build identity changed")
                try:
                    shutil.rmtree(quarantine)
                except OSError:
                    # The continuation still names ``build``. Restore that exact
                    # path so the next startup can revalidate and retry instead of
                    # stranding an invisible .retention-* directory forever.
                    if quarantine.exists():
                        if build.exists():
                            raise OSError(
                                "both quarantined and canonical build roots exist")
                        os.rename(quarantine, build)
                        _write_retry(root, current)
                        raise
                (root / RETRY_FILENAME).unlink(missing_ok=True)
            finally:
                os.close(descriptor)
        except OSError as exc:
            skipped.append({"build_root": str(build),
                            "reason": f"{type(exc).__name__}: {exc}"})
            continue
        removed.append({"build_root": str(build), "bytes": row["bytes"]})
    return {"schema": "epyc.autokernel.serial_build_retention_result.v1",
            "plan_digest": supplied, "dry_run": dry_run,
            "removed": removed, "skipped": skipped,
            "reclaimed_bytes": sum(item["bytes"] for item in removed),
            "free_bytes_after": shutil.disk_usage(Path(body["parent"])).free}


__all__ = ["BuildRetentionRefused", "DEFAULT_MAX_BUILD_DIRS",
           "DEFAULT_RECENT_STATE_CACHES", "DEFAULT_TARGET_FREE_BYTES",
           "DEFAULT_TRIGGER_FREE_BYTES", "PLAN_SCHEMA", "RETRY_SCHEMA", "execute", "plan"]
