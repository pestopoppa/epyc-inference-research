#!/usr/bin/env python3
"""Device-ISA identity check (AK-ISA-IDENT-1): do two HIP builds carry the same GPU code?

Zero-GPU, read-only, no builds. For each HIP library (``libggml-hip.so*`` with a
``.hip_fatbin``) or bare gfx90a code object, every embedded gfx90a code object is reduced to a
*normalized* fingerprint: its exported symbols (name, value, size, binding/type) plus the
bytes of every section that carries code or data (``.text``, ``.rodata`` kernel descriptors,
``.data*``), and nothing that can differ for non-device reasons (``.comment``, ``.note*``,
``.debug*``, string/symbol/hash tables, ``.dynamic``, ``.llvm_addrsig``, ``__hip_cuid_*`` symbols).

The library fingerprint is the SHA-256 of the SORTED per-object fingerprints, so object
order in the fatbin (link order) does not matter. ``compare`` returns:

* ``identical``      -- same multiset of objects: GPU re-measurement of the candidate is
                        redundant, so the candidate may be skipped (eligible);
* ``differs``        -- the multisets differ;
* ``indeterminate``  -- either side unreadable, compressed, empty or parse-failed.

``indeterminate`` is ALWAYS treated as differing: ``eligible_to_skip`` is True only for
``identical``. The check proves equality of the compiled device code only; it says nothing
about host code, runtime knobs or launch parameters, so a candidate that changes those is not
covered (the caller must have restricted the diff to device-compiled source).

Usage::

    device_isa_identity.py compare ANCHOR_LIB CANDIDATE_LIB
    device_isa_identity.py survey [--builds /mnt/raid0/llm/kernels/builds] [--json OUT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gfx90a_isa_audit as _audit  # noqa: E402

SCHEMA = "epyc.autokernel.device_isa_identity.v1"
# Sections whose contents are not device behaviour (paths, toolchain strings, notes, tables).
_EXCLUDED_PREFIXES = (".debug", ".comment", ".note", ".gnu", ".llvm_addrsig", ".strtab",
                      ".shstrtab", ".symtab", ".dynstr", ".dynsym", ".hash", ".rela", ".rel",
                      ".dynamic", ".relro_padding")
# Per-build identifier symbols (``__hip_cuid_<hash>``) vary between builds with identical code.
_VOLATILE_SYMBOL_PREFIXES = ("__hip_cuid_",)
SHT_NULL, SHT_NOBITS = 0, 8


def object_fingerprint(co: bytes) -> str:
    """Normalized fingerprint of ONE gfx90a code object (raises on malformed ELF)."""
    secs = _audit.elf_sections(co)
    parts: list[tuple] = []
    for s in secs:
        if s.type == SHT_NULL or s.name.startswith(_EXCLUDED_PREFIXES) or not s.name:
            continue
        if s.type == SHT_NOBITS:
            parts.append((s.name, s.size, ""))
        else:
            parts.append((s.name, s.size, hashlib.sha256(co[s.offset:s.offset + s.size]).hexdigest()))
    symbols = sorted((name, value, size, info) for name, value, size, info, _shndx
                     in _audit.elf_symbols(co, secs)
                     if name and not name.startswith(_VOLATILE_SYMBOL_PREFIXES))
    payload = json.dumps({"sections": sorted(parts), "symbols": symbols}, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def fingerprint(path: str | Path) -> dict[str, Any]:
    """Library/code-object fingerprint; ``status`` is ``ok`` or ``indeterminate`` + reason."""
    path = Path(path)
    out: dict[str, Any] = {"schema": SCHEMA, "path": str(path), "status": "indeterminate",
                           "reason": None, "objects": 0, "fingerprint": None, "object_fingerprints": []}
    try:
        objects = _audit.load_code_objects(path)
    except (OSError, _audit.AuditError, struct.error, ValueError) as exc:
        out["reason"] = f"unreadable: {type(exc).__name__}: {exc}"
        return out
    if not objects:
        out["reason"] = "no gfx90a code objects"
        return out
    try:
        prints = sorted(object_fingerprint(co) for co in objects)
    except (_audit.AuditError, struct.error, ValueError, IndexError) as exc:
        out["reason"] = f"malformed code object: {type(exc).__name__}: {exc}"
        return out
    out.update(status="ok", objects=len(prints), object_fingerprints=prints,
               fingerprint=hashlib.sha256("\n".join(prints).encode()).hexdigest())
    return out


def compare_fingerprints(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if a.get("status") != "ok" or b.get("status") != "ok":
        return {"verdict": "indeterminate", "eligible_to_skip": False,
                "reason": a.get("reason") or b.get("reason") or "fingerprint not ok"}
    if a["fingerprint"] == b["fingerprint"]:
        return {"verdict": "identical", "eligible_to_skip": True, "objects_differing": 0}
    ca, cb = Counter(a["object_fingerprints"]), Counter(b["object_fingerprints"])
    return {"verdict": "differs", "eligible_to_skip": False,
            "objects_differing": sum(((ca - cb) + (cb - ca)).values()),
            "objects_a": a["objects"], "objects_b": b["objects"]}


def compare(anchor: str | Path, candidate: str | Path) -> dict[str, Any]:
    return compare_fingerprints(fingerprint(anchor), fingerprint(candidate))


def _hip_lib(build: Path) -> Path | None:
    for name in ("libggml-hip.so.0.16.0", "libggml-hip.so"):
        p = build / "bin" / name
        if p.is_file():
            return p.resolve()
    hits = sorted((build / "bin").glob("libggml-hip.so*")) if (build / "bin").is_dir() else []
    return hits[0].resolve() if hits else None


def survey_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ident = sum(r["verdict"] == "identical" for r in rows)
    indet = sum(r["verdict"] == "indeterminate" for r in rows)
    return {"pairs": n, "identical": ident, "differs": n - ident - indet, "indeterminate": indet,
            "identical_fraction": (ident / n) if n else None}


def survey(builds_root: str | Path) -> dict[str, Any]:
    """Pair each stored GPU candidate build with its parent build (by BUILD-IDENTITY
    ``source_sha`` / ``parent``) and compare device ISA. Builds without a stored parent
    build are not paired (counted in ``unpaired``)."""
    root = Path(builds_root)
    ids: dict[str, dict] = {}
    for d in sorted(root.glob("gpu-*")):
        p = d / "BUILD-IDENTITY.json"
        try:
            ids[d.name] = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    by_sha = {j.get("source_sha"): n for n, j in ids.items() if j.get("source_sha")}
    cache: dict[str, dict] = {}

    def fp(name: str) -> dict:
        if name not in cache:
            lib = _hip_lib(root / name)
            cache[name] = fingerprint(lib) if lib else {
                "status": "indeterminate", "reason": "no libggml-hip in build"}
        return cache[name]

    rows, unpaired = [], 0
    for name, j in ids.items():
        parent = by_sha.get(j.get("parent")) if j.get("parent") else None
        if parent is None or parent == name:
            unpaired += 1
            continue
        res = compare_fingerprints(fp(parent), fp(name))
        rows.append({"candidate": name, "anchor": parent, **res})
    return {"schema": SCHEMA, "builds": len(ids), "unpaired": unpaired, "rows": rows,
            "summary": survey_summary(rows)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("anchor", type=Path)
    c.add_argument("candidate", type=Path)
    s = sub.add_parser("survey")
    s.add_argument("--builds", type=Path, default=Path("/mnt/raid0/llm/kernels/builds"))
    s.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        res = compare(args.anchor, args.candidate)
        print(json.dumps(res, indent=1))
        return 0 if res["verdict"] == "identical" else 1
    doc = survey(args.builds)
    if args.json:
        args.json.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(doc["summary"] | {"builds": doc["builds"], "unpaired": doc["unpaired"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
