"""AKX-ALL-29.3: recognise a GPU candidate build that changed no kernel code.

A candidate whose gfx90a code-object BODIES are byte-identical to the incumbent's
cannot differ in speed except by noise, so timing it can only manufacture a false
keep or a wasted window. `body_digest` hashes the executable content of every gfx90a
code object in a HIP library (the `.text`, `.rodata` -- kernel descriptors -- and
`.data` sections; not the notes, comment or build-id, which a rebuild moves). It
reuses the ELF/offload-bundle reader of `scripts/kernel_rnd/gfx90a_isa_audit.py`.

Correct when the hash is unavailable: any failure to read either library yields
None, `identical_bodies` then returns None, and the caller times the candidate
exactly as before. Only a positive match is ever acted on, and the outcome is a
"no_op_build" record carrying both digests -- never a keep.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

SCHEMA = "epyc.autokernel.noop_build.v1"
BODY_SECTIONS = (".text", ".rodata", ".data")
_AUDIT = Path(__file__).resolve().parents[2] / "gfx90a_isa_audit.py"


def _audit_module():
    spec = importlib.util.spec_from_file_location("_akx_gfx90a_isa_audit", _AUDIT)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules.setdefault(spec.name, module)   # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def code_object_body_digest(blob: bytes, audit=None) -> str:
    audit = audit or _audit_module()
    sections = audit.elf_sections(blob)
    h = hashlib.sha256()
    for name in BODY_SECTIONS:
        data = audit.section_bytes(blob, sections, name)
        h.update(name.encode() + b"\0" + str(-1 if data is None else len(data)).encode()
                 + b"\0" + (data or b""))
    return h.hexdigest()


def body_digest(path: Path | str) -> dict[str, Any] | None:
    """Digest of all gfx90a code-object bodies in a HIP library; None if unavailable."""
    try:
        audit = _audit_module()
        blobs = audit.load_code_objects(Path(path))
        if not blobs:
            return None
        parts = sorted(code_object_body_digest(blob, audit) for blob in blobs)
    except Exception:       # noqa: BLE001 -- unavailable means "time it", never "skip it"
        return None
    return {"digest": hashlib.sha256("\n".join(parts).encode()).hexdigest(),
            "n_code_objects": len(parts)}


def find_hip_library(build_dir: Path | str) -> Path | None:
    bin_dir = Path(build_dir) / "bin"
    for name in ("libggml-hip.so", "libggml-hip.so.0"):
        if (bin_dir / name).is_file():
            return bin_dir / name
    found = sorted(bin_dir.glob("libggml-hip.so.*")) if bin_dir.is_dir() else []
    return found[0] if found else None


def identical_bodies(incumbent: Path | str | None,
                     candidate: Path | str | None) -> dict[str, Any] | None:
    """The no-op-build record when both libraries hash and match; otherwise None."""
    if incumbent is None or candidate is None:
        return None
    inc, cand = body_digest(incumbent), body_digest(candidate)
    if inc is None or cand is None or inc["digest"] != cand["digest"]:
        return None
    return {"schema": SCHEMA, "section_scope": list(BODY_SECTIONS),
            "incumbent_body_sha256": inc["digest"], "candidate_body_sha256": cand["digest"],
            "n_code_objects": inc["n_code_objects"],
            "incumbent_library": str(incumbent), "candidate_library": str(candidate)}
