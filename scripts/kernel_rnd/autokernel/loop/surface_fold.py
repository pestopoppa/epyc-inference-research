"""Bounded original source-keep receipts; no fold or promotion authority."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping

from . import archive


KEEP_RECEIPT_SCHEMA = "epyc.autokernel.experimental_source_keep.v1"
ASSEMBLY_SCHEMA = "epyc.autokernel.shared_source_fold.v1"
MAX_RECEIPT_BYTES = 256 * 1024
MAX_KEEP_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_PATCH_BYTES = 16 * 1024 * 1024


class FoldRefused(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(char not in "0123456789abcdef" for char in value):
        raise FoldRefused(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FoldRefused(f"{label} must be nonempty text")
    return value


def _git(repo: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, timeout=600, env=env)
    if done.returncode:
        raise FoldRefused(f"git {args[0]} refused: {done.stderr.strip()}")
    return done.stdout.strip()


def _commit(repo: Path, value: Any, label: str) -> str:
    value = _text(value, label)
    actual = _git(repo, "rev-parse", "--verify", f"{value}^{{commit}}")
    if actual != value:
        raise FoldRefused(f"{label} is not a full unambiguous commit")
    return value


@dataclass(frozen=True)
class ExperimentalKeepReceipt:
    surface: str
    mechanism_id: str
    request_id: str
    repo: str
    branch: str
    parent_commit: str
    kept_commit: str
    patch_path: str
    patch_sha256: str
    patch_metadata_path: str
    patch_metadata_sha256: str
    selected_target: Mapping[str, Any]
    launch_snapshot_digest: str
    floor_request_digest: str
    floor_unit: str
    comparison: Mapping[str, Any]
    comparison_digest: str

    @classmethod
    def from_dict(cls, value: Any) -> "ExperimentalKeepReceipt":
        if not isinstance(value, dict):
            raise FoldRefused("keep receipt must be an object")
        keys = {"schema", "surface", "mechanism_id", "request_id", "repo", "branch",
                "parent_commit", "kept_commit", "patch_path", "patch_sha256",
                "patch_metadata_path", "patch_metadata_sha256",
                "selected_target", "launch_snapshot_digest", "floor_request_digest",
                "floor_unit", "comparison", "comparison_digest"}
        if set(value) != keys or value.get("schema") != KEEP_RECEIPT_SCHEMA:
            raise FoldRefused("keep receipt has an unknown schema or field set")
        repo, patch, metadata = (Path(_text(value[key], key))
                                 for key in ("repo", "patch_path", "patch_metadata_path"))
        if not all(path.is_absolute() for path in (repo, patch, metadata)):
            raise FoldRefused("keep receipt paths must be absolute")
        target, comparison = value["selected_target"], value["comparison"]
        if not isinstance(target, dict) or not isinstance(comparison, dict):
            raise FoldRefused("keep target and comparison must be objects")
        if _digest(comparison) != _sha(value["comparison_digest"], "comparison digest"):
            raise FoldRefused("comparison digest differs")
        unit = _text(value["floor_unit"], "floor unit")
        if unit not in {"arm", "session", "process"}:
            raise FoldRefused("floor unit is unknown")
        return cls(_text(value["surface"], "surface"),
                   _text(value["mechanism_id"], "mechanism id"),
                   _text(value["request_id"], "request id"), str(repo.resolve()),
                   _text(value["branch"], "branch"),
                   _text(value["parent_commit"], "parent commit"),
                   _text(value["kept_commit"], "kept commit"), str(patch.resolve()),
                   _sha(value["patch_sha256"], "patch digest"), str(metadata.resolve()),
                   _sha(value["patch_metadata_sha256"], "patch metadata digest"),
                   dict(target),
                   _sha(value["launch_snapshot_digest"], "launch snapshot"),
                   _sha(value["floor_request_digest"], "floor request"), unit,
                   dict(comparison), _sha(value["comparison_digest"], "comparison digest"))

    @property
    def keep_id(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema": KEEP_RECEIPT_SCHEMA, "surface": self.surface,
                "mechanism_id": self.mechanism_id, "request_id": self.request_id,
                "repo": self.repo, "branch": self.branch,
                "parent_commit": self.parent_commit, "kept_commit": self.kept_commit,
                "patch_path": self.patch_path, "patch_sha256": self.patch_sha256,
                "patch_metadata_path": self.patch_metadata_path,
                "patch_metadata_sha256": self.patch_metadata_sha256,
                "selected_target": dict(self.selected_target),
                "launch_snapshot_digest": self.launch_snapshot_digest,
                "floor_request_digest": self.floor_request_digest,
                "floor_unit": self.floor_unit, "comparison": dict(self.comparison),
                "comparison_digest": self.comparison_digest}

    def validated(self) -> "ExperimentalKeepReceipt":
        return ExperimentalKeepReceipt.from_dict(self.to_dict())


def bounded_regular_bytes(path: Path, limit: int) -> bytes:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise FoldRefused(f"artifact path is not an absolute closed path: {path}")
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    fd = None
    try:
        for component in path.parts[1:-1]:
            next_parent = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent)
            os.close(parent)
            parent = next_parent
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > limit:
            raise FoldRefused(f"artifact is not a bounded regular file: {path}")
        raw = os.read(fd, limit + 1)
        after = os.fstat(fd)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)
    if len(raw) != before.st_size or (before.st_dev, before.st_ino, before.st_size,
                                      before.st_mtime_ns) != (after.st_dev, after.st_ino,
                                                              after.st_size, after.st_mtime_ns):
        raise FoldRefused(f"artifact changed while read: {path}")
    return raw


def validate_original(receipt: ExperimentalKeepReceipt) -> ExperimentalKeepReceipt:
    receipt = receipt.validated()
    repo = Path(receipt.repo)
    parent = _commit(repo, receipt.parent_commit, "parent commit")
    kept = _commit(repo, receipt.kept_commit, "kept commit")
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", parent, kept],
                      timeout=600).returncode:
        raise FoldRefused("kept commit is not descended from its original parent")
    patch = bounded_regular_bytes(Path(receipt.patch_path), MAX_PATCH_BYTES)
    if hashlib.sha256(patch).hexdigest() != receipt.patch_sha256:
        raise FoldRefused("retained patch digest differs")
    metadata_raw = bounded_regular_bytes(Path(receipt.patch_metadata_path), MAX_RECEIPT_BYTES)
    if hashlib.sha256(metadata_raw).hexdigest() != receipt.patch_metadata_sha256:
        raise FoldRefused("retained patch metadata digest differs")
    try:
        metadata = json.loads(metadata_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoldRefused("retained patch metadata is invalid JSON") from exc
    expected_meta = {"schema", "original_head", "worktree", "lane", "mechanism_id",
                     "patch_file", "patch_sha256", "untracked_source_paths", "scope"}
    metadata_repo = Path(str(metadata.get("worktree"))).resolve() if isinstance(metadata, dict) else None
    same_source = False
    if metadata_repo is not None:
        try:
            same_source = (Path(_git(repo, "rev-parse", "--path-format=absolute",
                                     "--git-common-dir")) ==
                           Path(_git(metadata_repo, "rev-parse", "--path-format=absolute",
                                     "--git-common-dir")))
        except FoldRefused:
            same_source = False
    if not isinstance(metadata, dict) or set(metadata) != expected_meta \
            or metadata.get("schema") != "epyc.autokernel.source_patch_archive.v1" \
            or metadata.get("original_head") != parent \
            or not same_source \
            or metadata.get("mechanism_id") != receipt.mechanism_id \
            or metadata.get("patch_file") != Path(receipt.patch_path).name \
            or metadata.get("patch_sha256") != receipt.patch_sha256 \
            or metadata.get("scope") != "source_only_not_execution_evidence":
        raise FoldRefused("retained patch metadata differs from original keep")
    # Bind the original retained patch to the complete kept tree.  An ancestry
    # check alone would permit unrelated edits in the kept commit, and parsing
    # stdout/metadata cannot grant source authority.  A private temporary index
    # applies the exact bytes without touching the repository worktree or refs.
    with tempfile.TemporaryDirectory(prefix="autokernel-fold-index-") as temporary:
        index = Path(temporary) / "index"
        exact_patch = Path(temporary) / "original.patch"
        exact_patch.write_bytes(patch if patch.endswith(b"\n") else patch + b"\n")
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(index)
        _git(repo, "read-tree", parent, env=env)
        _git(repo, "apply", "--cached", "--binary", "--recount", str(exact_patch), env=env)
        if _git(repo, "write-tree", env=env) != _git(repo, "rev-parse", f"{kept}^{{tree}}"):
            raise FoldRefused("retained patch does not produce the complete kept tree")
    return receipt


def retain_receipt(root: Path, receipt: ExperimentalKeepReceipt) -> Path:
    receipt = validate_original(receipt)
    path = Path(root) / "fold-receipts" / f"{receipt.keep_id}.json"
    raw = canonical_bytes(receipt.to_dict()) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    archive._retain_bytes(path, raw)
    return path


def receipt_reference(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    raw = bounded_regular_bytes(path, MAX_KEEP_RECEIPT_BYTES)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def reopen_reference(value: Any) -> ExperimentalKeepReceipt:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise FoldRefused("keep reference has an unknown field set")
    path = Path(_text(value["path"], "keep reference path"))
    if not path.is_absolute():
        raise FoldRefused("keep reference path must be absolute")
    raw = bounded_regular_bytes(path, MAX_KEEP_RECEIPT_BYTES)
    if hashlib.sha256(raw).hexdigest() != _sha(value["sha256"], "keep reference digest"):
        raise FoldRefused("keep reference digest differs")
    try:
        return ExperimentalKeepReceipt.from_dict(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoldRefused("keep reference is invalid JSON") from exc


__all__ = ["ExperimentalKeepReceipt", "FoldRefused", "KEEP_RECEIPT_SCHEMA",
           "MAX_PATCH_BYTES", "MAX_RECEIPT_BYTES", "MAX_KEEP_RECEIPT_BYTES",
           "bounded_regular_bytes",
           "canonical_bytes", "receipt_reference", "reopen_reference",
           "retain_receipt", "validate_original"]
