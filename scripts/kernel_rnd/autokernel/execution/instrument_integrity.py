"""RVP-C6-1 measurement-source pinning at the candidate/anchor seam.

AutoKernel is allowed to edit kernels, not its reward instrument.  The binaries
are built from the candidate tree, so a binary digest alone cannot distinguish a
kernel change from a candidate that also weakened ``llama-bench`` or
``test-backend-ops``.  This module hashes the measurement translation units in
both source roots and requires byte identity with the named anchor immediately
before a measured invocation.

The anchor tree is the authority rather than a hash copied into this repository:
the hardened T1 instrument intentionally advances on an experimental branch
while the serving production tree remains frozen.  A campaign therefore names
the reviewed instrument anchor it was built from, and every candidate must carry
those exact bytes.  Missing and unreadable files fail; deletion can never satisfy
the comparison vacuously.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .. import schemas

__all__ = ["TRANSLATION_UNITS", "compare_to_anchor", "compare_manifest_to_anchor"]


TRANSLATION_UNITS = {
    "llama-bench": ("tools/llama-bench/llama-bench.cpp",),
    "test-backend-ops": ("tests/test-backend-ops.cpp",),
    "test-quantize-perf": ("tests/test-quantize-perf.cpp",),
}


def _digest(root: str, relative: str) -> tuple[str | None, str | None]:
    path = Path(root, relative)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"{path}: {type(exc).__name__}: {exc}"
    return hashlib.sha256(data).hexdigest(), None


def compare_to_anchor(*, tool: str, candidate_root: str,
                      anchor_root: str) -> schemas.Check:
    """Require every reward-bearing source file to equal the anchor byte-for-byte."""
    units = TRANSLATION_UNITS.get(tool)
    if units is None:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"no measurement-source manifest is registered for tool {tool!r}",))
    reasons: list[str] = []
    matched: list[str] = []
    for relative in units:
        anchor_hash, anchor_error = _digest(anchor_root, relative)
        candidate_hash, candidate_error = _digest(candidate_root, relative)
        if anchor_error:
            reasons.append(f"anchor instrument source is unreadable: {anchor_error}")
        if candidate_error:
            reasons.append(f"candidate instrument source is unreadable: {candidate_error}")
        if anchor_hash is not None and candidate_hash is not None:
            if anchor_hash != candidate_hash:
                reasons.append(
                    f"candidate {relative} sha256={candidate_hash} differs from anchor "
                    f"sha256={anchor_hash}; a proposal may not edit its reward instrument")
            else:
                matched.append(f"{relative}@{anchor_hash}")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS, (
        "measurement translation units are byte-identical to the named anchor: "
        + ", ".join(matched),))


def compare_manifest_to_anchor(*, candidate_root: str, anchor_root: str,
                               tools: tuple[str, ...] | None = None) -> schemas.Check:
    """Pin the complete registered reward-instrument source manifest.

    RVP-C6-1 names all three translation units, not merely the tool used by the
    current tier.  Checking the complete manifest at every live launch keeps a
    proposal from weakening T0 while T1 happens to be running (or vice versa),
    and de-duplicates a path if a future pair of tools share one source file.
    """
    selected = tuple(TRANSLATION_UNITS) if tools is None else tuple(tools)
    unknown = tuple(sorted(set(selected) - set(TRANSLATION_UNITS)))
    if unknown:
        return schemas.Check(schemas.COULD_NOT_CHECK, (
            f"no measurement-source manifest is registered for tools {unknown!r}",))
    relatives = tuple(dict.fromkeys(
        relative for tool in selected for relative in TRANSLATION_UNITS[tool]
    ))
    reasons: list[str] = []
    matched: list[str] = []
    for relative in relatives:
        anchor_hash, anchor_error = _digest(anchor_root, relative)
        candidate_hash, candidate_error = _digest(candidate_root, relative)
        if anchor_error:
            reasons.append(f"anchor instrument source is unreadable: {anchor_error}")
        if candidate_error:
            reasons.append(f"candidate instrument source is unreadable: {candidate_error}")
        if anchor_hash is not None and candidate_hash is not None:
            if anchor_hash != candidate_hash:
                reasons.append(
                    f"candidate {relative} sha256={candidate_hash} differs from anchor "
                    f"sha256={anchor_hash}; a proposal may not edit its reward instrument")
            else:
                matched.append(f"{relative}@{anchor_hash}")
    if reasons:
        return schemas.Check(schemas.FAIL, tuple(reasons))
    return schemas.Check(schemas.PASS, (
        "complete measurement-source manifest is byte-identical to the named anchor: "
        + ", ".join(matched),))
