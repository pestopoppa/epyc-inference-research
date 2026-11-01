"""Shared durable IO for actor, candidate and panel custody.

Callers own admission, disposition semantics and release decisions. These helpers
only capture original bytes, write durable records and transfer owned artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, row: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(row, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_path(path.parent)
    sync_path(path.parent.parent)
    sync_path(path.parent.parent.parent)
    return path


def write_record(store: Path, kind: str, row: dict) -> Path:
    stamp = hashlib.sha256(f"{kind}:{time.time_ns()}".encode()).hexdigest()
    return write_json(Path(store) / "patches" / f"{kind}-{stamp}.json", row)


def copy_reply_bundle(source: Path, target: Path, mirror: Path) -> str | None:
    if source.is_symlink():
        raise RuntimeError("panel reply directory is a symlink")
    if not source.exists():
        return None
    if any(path.is_symlink() for path in source.rglob("*")):
        raise RuntimeError("panel reply directory contains a symlink")
    shutil.copytree(source, target)
    for original in source.iterdir():
        if original.is_file() and original.suffix in {".stdout", ".stderr"}:
            destination = mirror / original.name
            if destination.exists():
                if destination.read_bytes() != original.read_bytes():
                    raise RuntimeError("panel reply filename collision")
            else:
                with original.open("rb") as incoming, destination.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
    for path in target.rglob("*"):
        if path.is_file():
            sync_path(path)
    directories = sorted((p for p in target.rglob("*") if p.is_dir()),
                         key=lambda p: len(p.parts), reverse=True)
    for directory in [*directories, target, mirror]:
        sync_path(directory)
    return str(target.resolve())


def _check_rows(data: bytes, *, tail_only: bool = False) -> None:
    if not data:
        return
    if not data.endswith(b"\n"):
        raise OSError("call log has an unterminated tail")
    lines = data.splitlines()
    if tail_only:
        lines = lines[-1:]
    try:
        if any(not isinstance(json.loads(line), dict) for line in lines):
            raise ValueError("call row is not an object")
    except (ValueError, UnicodeError) as exc:
        raise OSError("call log has a corrupt row") from exc


def append_rows(destination: Path, text: str) -> None:
    """Append complete rows durably; roll back a failed append while holding custody.

    Callers serialize writers to a destination. Existing malformed tails are never
    repaired here: their provenance must be reconciled before accepting another row.
    """
    _check_rows(text.encode("utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fence = destination.with_name(destination.name + ".append-inflight")
    # Acquire before reading the offset: concurrent writers cannot roll back each
    # other's rows. An incomplete transaction remains fenced for reconciliation.
    with fence.open("x", encoding="utf-8") as marker:
        try:
            original = destination.read_bytes()
        except FileNotFoundError:
            original = b""
        _check_rows(original, tail_only=True)
        offset = len(original)
        marker.write(str(offset) + "\n")
        marker.flush()
        os.fsync(marker.fileno())
    sync_path(destination.parent)
    try:
        with open(destination, "a", encoding="utf-8") as stream:
            if stream.write(text) != len(text):
                raise OSError("short call log append")
            stream.flush()
            os.fsync(stream.fileno())
        sync_path(destination.parent)
    except Exception as exc:
        try:
            with open(destination, "r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
            sync_path(destination.parent)
            fence.unlink()
            sync_path(destination.parent)
        except Exception as rollback:
            raise OSError(f"call log append failed; rollback failed: {rollback}") from exc
        raise
    fence.unlink()
    sync_path(destination.parent)



def transfer_tail(source: Path, destination: Path, offset: int, text: str) -> None:
    append_rows(destination, text)
    with open(source, "r+b") as stream:
        stream.truncate(offset)
        stream.flush()
        os.fsync(stream.fileno())


def release_scope_paths(scope, workspace: Path, extra) -> bool:
    release = getattr(scope, "release", None)
    if release is None:
        return False
    try:
        release(workspace)
    except Exception:
        return False
    for path in extra:
        try:
            release(path)
        except Exception:
            pass
    return True


def capture_from_base(repo: Path, base: str, capture, git_env) -> tuple[bytes, list[str]]:
    """Exact base-to-current source patch, including accepted untracked source."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              env=git_env(), timeout=600, check=True).stdout
    head = git("rev-parse", "HEAD").decode().strip()
    def diff(ref):
        return git("diff", "--binary", "--full-index", "--no-ext-diff",
                   "--no-textconv", ref, "--")
    dirty = capture(repo)
    tracked_head, tracked_base = diff(head), diff(base)
    if dirty is None:
        if tracked_head:
            raise RuntimeError("dirty source changed during capture")
        additions, extra = [], b""
    else:
        captured_head, patch, additions = dirty
        if captured_head != head or not patch.startswith(tracked_head):
            raise RuntimeError("source changed during capture")
        extra = patch[len(tracked_head):]
    if (git("rev-parse", "HEAD").decode().strip() != head
            or diff(base) != tracked_base or capture(repo) != dirty):
        raise RuntimeError("source changed during capture")
    return tracked_base + extra, additions
