#!/usr/bin/env python3
"""Derive a sealed :class:`champion.AnchorIdentity` by MEASURING a real tree.

`champion.py` is deliberately pure: it "does not choose research ideas, edit a
tree, build, benchmark, launch a process, or prepare a release". Deriving an
anchor requires reading binaries off disk and asking the loader what it actually
binds, so that work lives here and hands `champion.py` an already-sealed anchor.

WHY THIS EXISTS
---------------
The operator's standing requirement (2026-08-27) is that there is ALWAYS an
aggregate production candidate ready for promotion gate testing. That means a
champion must exist from the first moment of a campaign, seeded from production,
so AutoKernel screens against an accumulating aggregate instead of re-deriving
deltas against a fixed anchor forever.

`champion.AnchorIdentity` refuses construction unless every artifact carries a
measured `binary_sha256` and `linkage_sha256`, and unless the artifact set covers
exactly the backends the source tree declares. That refusal is correct and is the
reason a champion cannot be conjured: Champion₀ must cite the real frozen
production binaries. This module measures them rather than asserting them.

`linkage_sha256` uses the definition already fixed by
`execution/t0_provider.linkage_digest`: a digest of the RESOLVED `(soname, path)`
table, sorted -- "the identity of what the loader actually bound, not of the
binary and not of the env string". Two builds with identical binaries that
resolve different libraries have different linkage, which is the whole point.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

from .. import schemas
from .champion import AnchorArtifact, AnchorIdentity, ChampionError

__all__ = ["AnchorMeasurementError", "BACKEND_BUILD_DIRS",
           "binary_sha256", "linkage_sha256", "production_anchor"]


class AnchorMeasurementError(ChampionError):
    """A tree could not be measured into a sealed anchor."""


#: Backend -> build subdirectory, for the llama.cpp source tree. `llama_cpu` and
#: `llama_gpu` are exactly the backends `schemas.SOURCE_TREE_BY_BACKEND` maps to
#: `llama.cpp`, and `AnchorIdentity` requires the set to match exactly.
BACKEND_BUILD_DIRS: Mapping[str, str] = {
    "llama_cpu": "build",
    "llama_gpu": "build-hip",
}

_LDD_TIMEOUT_S = 120.0


def binary_sha256(path: Path) -> str:
    """SHA-256 of a regular file, read in chunks (these binaries are large)."""
    if path.is_symlink() or not path.is_file():
        raise AnchorMeasurementError(f"anchor binary must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linkage_sha256(binary: Path, library_path: Path) -> str:
    """Digest of the RESOLVED (soname, path) table, per t0_provider.linkage_digest.

    Deliberately NOT a hash of raw `ldd` output: that text carries ASLR addresses
    and would differ on every invocation. Only successfully resolved rows count; an
    unresolved row is refused rather than silently hashed as absent, because
    "cannot resolve" and "resolves elsewhere" must not produce the same identity.
    """
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{library_path}:{env.get('LD_LIBRARY_PATH', '')}"
    try:
        completed = subprocess.run(
            ("ldd", str(binary)), env=env, text=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=_LDD_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise AnchorMeasurementError(f"ldd timed out on {binary}") from exc
    except subprocess.CalledProcessError as exc:
        raise AnchorMeasurementError(f"ldd failed on {binary}: {exc.output}") from exc

    rows: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if "=>" not in line:
            continue
        soname, _, rest = line.partition("=>")
        resolved = rest.strip().split(" (")[0].strip()
        if not resolved or resolved == "not found":
            raise AnchorMeasurementError(
                f"{binary} has an unresolved shared object: {soname.strip()!r}")
        rows.append((soname.strip(), resolved))
    if not rows:
        raise AnchorMeasurementError(f"ldd resolved no libraries for {binary}")
    ordered = tuple(sorted(set(rows)))
    return hashlib.sha256(json.dumps(ordered, sort_keys=True).encode("utf-8")).hexdigest()


def production_anchor(tree_root: Path, *, branch: str, commit: str,
                      source_tree: str = "llama.cpp", tool: str = "llama-server",
                      expected_binary_sha256: Mapping[str, str] | None = None
                      ) -> AnchorIdentity:
    """Measure `tree_root` into the sealed anchor Champion₀ is seeded from.

    `expected_binary_sha256` (backend -> digest) is the pin from
    `scripts/session/verify_llama_cpp.sh`. When supplied, a mismatch is refused:
    seeding a champion off a tree that is not the ratified production build would
    silently re-anchor every future comparison. Supply it.
    """
    root = Path(tree_root)
    if not root.is_dir():
        raise AnchorMeasurementError(f"anchor tree root is not a directory: {root}")
    if source_tree not in schemas.SOURCE_TREES:
        raise AnchorMeasurementError(f"unknown source tree {source_tree!r}")

    required = {backend for backend, tree in schemas.SOURCE_TREE_BY_BACKEND.items()
                if tree == source_tree}
    if set(BACKEND_BUILD_DIRS) != required:
        raise AnchorMeasurementError(
            f"build-dir map covers {sorted(BACKEND_BUILD_DIRS)}, "
            f"{source_tree} requires {sorted(required)}")

    artifacts: list[AnchorArtifact] = []
    for backend in sorted(BACKEND_BUILD_DIRS):
        binary = root / BACKEND_BUILD_DIRS[backend] / "bin" / tool
        measured = binary_sha256(binary)
        expected = (expected_binary_sha256 or {}).get(backend)
        if expected is not None and measured != expected:
            raise AnchorMeasurementError(
                f"{backend} {tool} is {measured}, ratified production pin is {expected} — "
                "refusing to seed a champion off an unratified build")
        artifacts.append(AnchorArtifact(
            backend=backend, tool=tool, binary_sha256=measured,
            linkage_sha256=linkage_sha256(binary, binary.parent)))

    return AnchorIdentity(source_tree=source_tree, branch=branch, commit=commit,
                          artifacts=tuple(sorted(artifacts)), sealed=True)
