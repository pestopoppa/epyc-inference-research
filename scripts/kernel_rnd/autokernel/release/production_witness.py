#!/usr/bin/env python3
"""Zero-production-write witness for the AKU-13 end-to-end campaign.

AKU-13 asks for a campaign that stops at a validated package with zero production
writes.  The packager refuses every production-write capability in code
(``packager.REFUSED_CAPABILITIES``), but a refusal door proves nothing about what
the surrounding campaign did on disk.  This module is the independent, black-box
check: it records a stat-level manifest of the protected trees before the run and
compares it after, so "zero writes" is an observed fact about the filesystem and
not a property inferred from our own code.

The witness is read-only.  It performs ``lstat`` and ``scandir`` only (symlinks are
never followed), hashes nothing unless a caller opts in for small files, and owns
no build, benchmark, inference, process or clock capability.  Pointing it at the
real frozen trees costs one directory walk; the unit tests use disposable trees.

Usage by a live campaign owner (the live run itself stays blocked behind the
OP-AK-HOLD window grant)::

    PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.release.production_witness \
      snapshot --out /mnt/raid0/llm/autokernel/witness/<campaign-id>.before.json
    ... run the campaign to its validated package ...
    PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.release.production_witness \
      verify --before /mnt/raid0/llm/autokernel/witness/<campaign-id>.before.json \
      --out /mnt/raid0/llm/autokernel/witness/<campaign-id>.verdict.json

``verify`` exits 0 only when the manifests are identical.  A snapshot or verdict
that covers zero entries is refused, so an empty or mistyped root cannot pass
vacuously.

The module is not imported by ``autokernel.release.__init__``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

MODULE_ID = "autokernel.release.production_witness/v1"
SNAPSHOT_SCHEMA = "epyc.autokernel.production_tree_snapshot.v1"
VERDICT_SCHEMA = "epyc.autokernel.production_write_witness.v1"

KIND_FILE = "file"
KIND_DIR = "dir"
KIND_LINK = "link"
KIND_OTHER = "other"
KIND_MISSING = "missing"


class WitnessError(Exception):
    """Base for witness refusals."""


class WitnessUnavailable(WitnessError):
    """The witness cannot give a non-vacuous answer (no roots, missing or empty root)."""


class ProductionWritesDetected(WitnessError):
    """The protected trees differ between the before and after snapshots."""

    def __init__(self, diff: "TreeDiff") -> None:
        self.diff = diff
        super().__init__(
            "production tree changed: "
            f"{len(diff.added)} added, {len(diff.removed)} removed, "
            f"{len(diff.modified)} modified")


def default_protected_roots() -> tuple:
    """The frozen production trees, from the single declaration in ``storage``."""
    from .. import storage
    return tuple(storage.PRODUCTION_TREES)


def _entry(path: str, st: os.stat_result, hash_max_bytes: int) -> list:
    mode = st.st_mode
    if stat.S_ISLNK(mode):
        return [KIND_LINK, os.readlink(path)]
    if stat.S_ISDIR(mode):
        return [KIND_DIR, stat.S_IMODE(mode)]
    if stat.S_ISREG(mode):
        digest = None
        if 0 < hash_max_bytes and st.st_size <= hash_max_bytes:
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    h.update(block)
            digest = h.hexdigest()
        return [KIND_FILE, st.st_size, st.st_mtime_ns, stat.S_IMODE(mode), digest]
    return [KIND_OTHER, stat.S_IFMT(mode)]


def _walk(root: str, hash_max_bytes: int, out: dict) -> None:
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError:
        out[root] = [KIND_MISSING]
        return
    out[root] = _entry(root, root_stat, hash_max_bytes)
    if not stat.S_ISDIR(root_stat.st_mode):
        return
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                children = sorted(it, key=lambda e: e.name)
        except (FileNotFoundError, NotADirectoryError):
            continue
        for child in children:
            path = os.path.join(current, child.name)
            try:
                st = child.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            out[path] = _entry(path, st, hash_max_bytes)
            if stat.S_ISDIR(st.st_mode):
                stack.append(path)


@dataclass(frozen=True)
class TreeSnapshot:
    roots: tuple
    hash_max_bytes: int
    entries: Mapping[str, list]

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.entries)).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "module": MODULE_ID,
            "roots": list(self.roots),
            "hash_max_bytes": self.hash_max_bytes,
            "entry_count": len(self.entries),
            "digest": self.digest,
            "entries": {k: self.entries[k] for k in sorted(self.entries)},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TreeSnapshot":
        if data.get("schema") != SNAPSHOT_SCHEMA:
            raise WitnessError(f"not a {SNAPSHOT_SCHEMA} document")
        snap = cls(tuple(data["roots"]), int(data["hash_max_bytes"]),
                   dict(data["entries"]))
        if snap.digest != data.get("digest") or len(snap.entries) != data.get("entry_count"):
            raise WitnessError("snapshot digest or entry count does not match its entries")
        return snap


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def take_snapshot(roots: Iterable[str], *, hash_max_bytes: int = 0,
                  require_existing: bool = True) -> TreeSnapshot:
    """Record a stat-level manifest of every root (symlinks are never followed)."""
    resolved = tuple(os.path.abspath(str(r)) for r in roots)
    if not resolved:
        raise WitnessUnavailable("no protected roots were given; a zero-write claim would be vacuous")
    entries: dict = {}
    for root in resolved:
        _walk(root, hash_max_bytes, entries)
    if require_existing:
        missing = [r for r in resolved if entries.get(r, [KIND_MISSING])[0] == KIND_MISSING]
        if missing:
            raise WitnessUnavailable(f"protected root does not exist: {missing}")
        for root in resolved:
            if not any(p == root or p.startswith(root.rstrip("/") + "/")
                       for p in entries if p != root) and entries[root][0] == KIND_DIR:
                raise WitnessUnavailable(f"protected root is an empty directory: {root}")
    return TreeSnapshot(resolved, hash_max_bytes, entries)


@dataclass(frozen=True)
class TreeDiff:
    added: tuple
    removed: tuple
    modified: tuple

    @property
    def clean(self) -> bool:
        return not (self.added or self.removed or self.modified)

    def to_dict(self) -> dict:
        return {"added": list(self.added), "removed": list(self.removed),
                "modified": list(self.modified)}


def diff_snapshots(before: TreeSnapshot, after: TreeSnapshot) -> TreeDiff:
    if before.roots != after.roots or before.hash_max_bytes != after.hash_max_bytes:
        raise WitnessError("snapshots were taken over different roots or hash settings")
    b, a = before.entries, after.entries
    added = tuple(sorted(set(a) - set(b)))
    removed = tuple(sorted(set(b) - set(a)))
    modified = tuple(sorted(p for p in set(a) & set(b) if a[p] != b[p]))
    return TreeDiff(added, removed, modified)


def verdict_for(before: TreeSnapshot, after: TreeSnapshot) -> dict:
    diff = diff_snapshots(before, after)
    return {
        "schema": VERDICT_SCHEMA,
        "module": MODULE_ID,
        "ok": diff.clean,
        "roots": list(before.roots),
        "entry_count": len(before.entries),
        "before_digest": before.digest,
        "after_digest": after.digest,
        "diff": diff.to_dict(),
    }


def assert_zero_writes(before: TreeSnapshot, after: TreeSnapshot) -> dict:
    verdict = verdict_for(before, after)
    if not verdict["ok"]:
        raise ProductionWritesDetected(diff_snapshots(before, after))
    return verdict


class ZeroWriteWitness:
    """Context manager: snapshot on entry, compare on exit.

    ``verdict`` is set on exit.  When the body raised, the original exception
    propagates and is never masked by a witness failure.
    """

    def __init__(self, roots: Optional[Sequence[str]] = None, *,
                 hash_max_bytes: int = 0, raise_on_write: bool = True) -> None:
        self.roots = tuple(roots) if roots is not None else default_protected_roots()
        self.hash_max_bytes = hash_max_bytes
        self.raise_on_write = raise_on_write
        self.before: Optional[TreeSnapshot] = None
        self.after: Optional[TreeSnapshot] = None
        self.verdict: Optional[dict] = None

    def __enter__(self) -> "ZeroWriteWitness":
        self.before = take_snapshot(self.roots, hash_max_bytes=self.hash_max_bytes)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.before is not None
        self.after = take_snapshot(self.roots, hash_max_bytes=self.hash_max_bytes,
                                   require_existing=False)
        self.verdict = verdict_for(self.before, self.after)
        if exc_type is None and self.raise_on_write and not self.verdict["ok"]:
            raise ProductionWritesDetected(diff_snapshots(self.before, self.after))
        return False


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="write a before-manifest of the protected roots")
    snap.add_argument("--root", action="append", help="protected root (default: frozen production trees)")
    snap.add_argument("--hash-max-bytes", type=int, default=0)
    snap.add_argument("--out", required=True)
    ver = sub.add_parser("verify", help="compare the roots to a before-manifest; exit 0 only if identical")
    ver.add_argument("--before", required=True)
    ver.add_argument("--out")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "snapshot":
            roots = args.root or default_protected_roots()
            snap = take_snapshot(roots, hash_max_bytes=args.hash_max_bytes)
            _write_json(Path(args.out), snap.to_dict())
            print(f"snapshot {snap.digest} entries={len(snap.entries)}")
            return 0
        before = TreeSnapshot.from_dict(json.loads(Path(args.before).read_text()))
        after = take_snapshot(before.roots, hash_max_bytes=before.hash_max_bytes,
                              require_existing=False)
        verdict = verdict_for(before, after)
        if args.out:
            _write_json(Path(args.out), verdict)
        print(json.dumps({k: verdict[k] for k in ("ok", "before_digest", "after_digest")}))
        return 0 if verdict["ok"] else 3
    except WitnessError as exc:
        print(f"production_witness: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
