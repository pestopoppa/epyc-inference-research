"""Kernel-store-first build lookup for AutoKernel source-change candidates (AK-SETUP-1).

Before compiling a candidate the harness derives a content-addressed `store_key` from the
inputs that decide the bytes (patch sha, base and instrument authority, cmake defines,
toolchain, environment, required targets) and looks it up under
`/mnt/raid0/llm/kernels/builds/ak-<key16>/BUILD-OK.json`.

A hit is VERIFIED, never assumed: the entry must name the same key, the pointed-at build
dir must still exist, and every recorded binary digest must match the file on disk. Any
failure is a MISS (a stale entry costs a recompile, never a campaign failure). A hit is
materialised into the planned build dir by hard link (same filesystem) and the recorded
BuildResult dict is replayed; a miss builds normally and then publishes a pointer entry.

Not in the key: paths, `parallelism.jobs` (AK-SETUP-2 accepts a -j change only with an
identical digest), the deployment/supervisor identities and `proposal_sha256`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import hashlib
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .. import schemas

DEFAULT_STORE = Path("/mnt/raid0/llm/kernels/builds")
ENTRY_SCHEMA = "epyc.autokernel.candidate_build_store_entry.v1"
KEY_SCHEMA = "epyc.autokernel.candidate_store_key.v1"
_KEY_FIELDS = ("patch_sha256", "production_base_authority", "instrument_authority",
               "selected_gpu_base_blobs", "cmake_defines", "build_type", "required_targets",
               "build_environment", "toolchain")
_DIGEST_FILES = ("llama-bench", "test-backend-ops")


def store_key(contract: Mapping[str, Any]) -> str:
    missing = [k for k in _KEY_FIELDS if k not in contract]
    if missing:
        raise ValueError(f"build contract lacks store-key fields: {missing}")
    return schemas.content_hash({"schema": KEY_SCHEMA, **{k: contract[k] for k in _KEY_FIELDS}})


def entry_dir(store_root: Path, key: str) -> Path:
    return Path(store_root) / f"ak-{key[:16]}"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_digests(build_dir: Path, targets: Sequence[str] = _DIGEST_FILES) -> dict[str, str]:
    out = {}
    for name in targets:
        p = Path(build_dir) / "bin" / name
        if not p.is_file() or p.is_symlink():
            raise FileNotFoundError(str(p))
        out[name] = _sha(p)
    return out


def lookup(store_root: Path, key: str) -> dict[str, Any] | None:
    """Verified entry for `key`, or None. Never raises on a bad entry."""
    try:
        entry = json.loads((entry_dir(store_root, key) / "BUILD-OK.json").read_text("utf-8"))
        if (not isinstance(entry, Mapping) or entry.get("schema") != ENTRY_SCHEMA
                or entry.get("ak_store_key") != key):
            return None
        build_dir = Path(entry["build_dir"])
        digests = entry["binary_digests"]
        if not isinstance(digests, Mapping) or not digests or not isinstance(entry["result"], Mapping):
            return None
        for name, want in digests.items():
            p = build_dir / "bin" / name
            if not p.is_file() or p.is_symlink() or _sha(p) != want:
                return None
        return dict(entry)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def publish(store_root: Path, key: str, *, build_dir: Path, result: Mapping[str, Any],
            build_seconds: float | None = None) -> Path | None:
    """Record a pointer entry for a freshly built candidate. Best-effort: OSError -> None."""
    try:
        d = entry_dir(store_root, key)
        d.mkdir(parents=True, exist_ok=True)
        body = {"schema": ENTRY_SCHEMA, "ak_store_key": key, "kind": "ak-candidate",
                "build_dir": str(Path(build_dir).resolve()), "result": dict(result),
                "binary_digests": binary_digests(build_dir), "build_seconds": build_seconds}
        text = json.dumps(body, sort_keys=True, indent=1, allow_nan=False) + "\n"
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".BUILD-OK.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, d / "BUILD-OK.json")
        return d / "BUILD-OK.json"
    except (OSError, FileNotFoundError, TypeError, ValueError):
        return None


def materialize(entry: Mapping[str, Any], dest: Path) -> bool:
    """Hard-link the stored build into `dest` and re-verify digests there. False -> build fresh."""
    src, dest = Path(entry["build_dir"]), Path(dest)
    if dest.exists():
        return False  # never overlay a planned dir that already holds something
    try:
        shutil.copytree(src, dest, copy_function=os.link, symlinks=True)
        for name, want in entry["binary_digests"].items():
            p = dest / "bin" / name
            if not p.is_file() or _sha(p) != want:
                raise OSError("digest mismatch after materialise")
        return True
    except (OSError, shutil.Error):
        shutil.rmtree(dest, ignore_errors=True)
        return False
