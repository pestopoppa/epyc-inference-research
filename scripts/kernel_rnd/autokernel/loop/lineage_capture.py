"""Default-off prospective source capture for offline lineage diagnostics.

This module is deliberately not imported by the live loop. A future approved
producer may call ``capture`` at the existing ``keep_the_diff`` boundary, after
the authored diff exists and before the lane is reset. The caller must supply
its original attempt/journal identity; this module never guesses one from a
mechanism name, lane, path, or current git state after the attempt.

The retained ``patches/<mechanism>.<lane>.patch`` files overwrite repeat
attempts and old archive rows do not bind their bytes. They cannot be used as
historical lineage evidence. This helper records ORIGINAL patch and full source
bytes prospectively, with content addresses and an immutable capture receipt.
It does not build, evaluate, score, or change any policy or gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping


SCHEMA = "epyc.autokernel.lineage_source_capture.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _immutable(path: Path, raw: bytes) -> None:
    """Create once, fsync, and refuse a changed same-name artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError(f"immutable lineage artifact conflicts: {path}")
        return
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # A failed first write must not be mistaken for a complete receipt.
        path.unlink(missing_ok=True)
        raise


def capture(store: Path, *, capture_id: str, parent_id: str | None,
            base_commit: str, patch_bytes: bytes | None,
            source_files: Mapping[str, bytes]) -> dict:
    """Seal the original authored patch and exact post-authoring source files.

    ``source_files`` must contain the complete analysis scope, not just a
    post-hoc chosen hot function. Callers must freeze its file list before any
    score is known. A distinct attempt receives a distinct ``capture_id`` even
    if its patch bytes repeat; outcome joins require that ID in the native row.
    """
    if not _ID.fullmatch(capture_id) or (parent_id is not None and not _ID.fullmatch(parent_id)):
        raise ValueError("invalid capture or parent identity")
    if not _COMMIT.fullmatch(base_commit):
        raise ValueError("base_commit must be a full source commit")
    if parent_id is None:
        if patch_bytes is not None:
            raise ValueError("root source capture must have no patch")
    elif not isinstance(patch_bytes, bytes) or not patch_bytes:
        raise ValueError("child authored patch bytes required")
    if not source_files:
        raise ValueError("complete source file mapping required")
    files = []
    for name, raw in sorted(source_files.items()):
        if not isinstance(name, str):
            raise ValueError("source file path must be text")
        path = PurePosixPath(name)
        if (name == "." or not path.parts or path.is_absolute() or ".." in path.parts
                or name != str(path) or not isinstance(raw, bytes)):
            raise ValueError(f"unsafe source file entry: {name!r}")
        raw.decode("utf-8")
        files.append((name, raw, _sha(raw)))
    patch_sha = _sha(patch_bytes) if patch_bytes is not None else None
    source_text = "".join(f"// FILE {name}\n{raw.decode('utf-8')}\n"
                          for name, raw, _ in files).encode("utf-8")
    source_sha = _sha(source_text)
    root = Path(store) / "lineage"
    if patch_sha is not None:
        _immutable(root / "blobs" / patch_sha[:2] / f"{patch_sha}.patch", patch_bytes)
    _immutable(root / "blobs" / source_sha[:2] / f"{source_sha}.txt", source_text)
    receipt = {"schema": SCHEMA, "capture_id": capture_id, "parent_id": parent_id,
               "base_commit": base_commit, "patch_sha256": patch_sha,
               "solution_sha256": source_sha,
               "files": [{"path": name, "sha256": digest} for name, _, digest in files],
               "authority": "offline_lineage_source_only"}
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _immutable(root / "captures" / f"{capture_id}.json", encoded)
    return receipt


__all__ = ["SCHEMA", "capture"]
