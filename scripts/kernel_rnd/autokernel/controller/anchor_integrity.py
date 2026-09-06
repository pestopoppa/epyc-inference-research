#!/usr/bin/env python3
"""Code-section digests for the anchor guard's hash pre-check (R22-3).

WHY. Builds of one commit are DETERMINISTIC on this host: the R21-10 probe built
`ce1df3aa` twice into different directories and the two `libggml-hip.so` files
differed by exactly ONE byte, in `.dynstr` — the RUNPATH string, which encodes the
build directory's own path. Every section that carries code (`.text`,
`.hip_fatbin`, `.rodata`) was bit-identical. So "is the promoted anchor the same
code as a fresh champion build" is answerable BY HASH, deterministically, before a
single benchmark pair is spent — and the answer changes what an above-floor A/A
means:

  * digests IDENTICAL → the anchor provably IS the champion, and an above-floor
    A/A reading indicts the MEASUREMENT SESSION, not the anchor. Run 21 aborted a
    healthy run on a 4.2σ instrument excursion (+1.765% against a pooled A/A σ of
    0.417%) because the guard had only the A/A to reason from. The guard was right
    by its own rules and wrong about the world.
  * digests DIFFER → run 18's fault class (the binary in the anchor slot is not
    the champion), proven without 20 pairs of device time.

WHAT IS HASHED. The sections named in `CODE_SECTIONS`, in file order, each
prefixed by its name and length so section boundaries cannot alias. Everything
else is EXCLUDED on purpose: `.dynstr`/`.dynamic`/RUNPATH legitimately differ by
build-path length, `.rela.*` and the build-id follow them, and the section header
table shifts with any of it. An include-list, not an exclude-list, so a novel
section added by a future toolchain cannot silently join the digest and turn
determinism flaky.

This module lives in `controller/` because it is a LIBRARY the loop calls (like
`build_recipe` and `experiments`), not loop control flow. The decision — abort,
heal, or continue — stays in `loop/anchor.py`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

#: The sections that carry the code a benchmark executes. `.text` is host code,
#: `.hip_fatbin` the device kernels, `.rodata` the constant pools both read.
CODE_SECTIONS = (".text", ".hip_fatbin", ".rodata")

#: The library the digest is taken over: the tree's GPU kernels all live here,
#: and it is the artifact the R21-10 determinism probe was run on.
LIBRARY = Path("bin") / "libggml-hip.so"


def section_spans(elf_path: Path | str) -> list[tuple[str, int, int]]:
    """`(name, offset, size)` per section, parsed from `readelf -S -W`.

    Adapted from the R21-10 probe (`artifacts/r2110-anchor-guard-abort/
    r2110-secdiff.py`), which classified the double-build diff byte-by-byte with
    this same parse. Returns [] when readelf refuses the file — a truncated or
    non-ELF artifact must read as "no digest", never as a digest of garbage.
    """
    try:
        out = subprocess.run(["readelf", "-S", "-W", str(elf_path)],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    spans = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        parts = line.split("]", 1)[1].split()
        if len(parts) < 6 or parts[1] in ("NULL", "Type"):
            continue
        try:
            name, off, size = parts[0], int(parts[3], 16), int(parts[4], 16)
        except ValueError:
            continue
        spans.append((name, off, size))
    return spans


def code_digest(elf_path: Path | str) -> str | None:
    """SHA-256 over the CODE sections of one ELF, or None when it cannot be taken.

    None — never a digest of the wrong thing — when the file is missing, readelf
    cannot parse it, or NONE of `CODE_SECTIONS` is present (an ELF with no code
    sections is not the artifact this question is about). The caller treats None
    as "hash unavailable, fall back to the A/A alone", so this function must
    never launder an error into a stable-looking value.
    """
    path = Path(elf_path)
    spans = [(name, off, size) for name, off, size in section_spans(path)
             if name in CODE_SECTIONS]
    if not spans:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for name, off, size in sorted(spans, key=lambda s: s[1]):
                # Name + length prefix: two layouts whose section bytes happen to
                # concatenate identically must not collide.
                digest.update(f"{name}:{size}:".encode())
                handle.seek(off)
                remaining = size
                while remaining > 0:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        return None  # truncated mid-section: not a digest
                    digest.update(chunk)
                    remaining -= len(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def build_digest(build_dir: Path | str) -> str | None:
    """The code digest of a BUILD DIRECTORY's kernel library, or None.

    This is the callable `loop/run.py` injects into `anchor.verify` — keyed off
    the build directory because that is the unit the guard compares (the promoted
    anchor slot vs the scratch build).
    """
    return code_digest(Path(build_dir) / LIBRARY)


#: Objects that are part of the LIBRARY (the champion kernel + the llama library it serves),
#: as opposed to the executables. Executables are target-set dependent -- PROMOTION_TARGETS
#: builds llama-cli/llama-server/mtmd that the guard's narrow build does not -- so they must
#: never enter the identity digest; the library code must.
_IDENTITY_EXCLUDE = ("tools/", "examples/", "tests/", "pocs/")


def object_digest(build_dir: Path | str) -> str | None:
    """The DETERMINISTIC identity of a build: one hash over the sha256 of every library
    object. This is what the anchor guard should compare, and why (2026-09-06, "resolve
    this once and for all"):

    Across every digest mismatch investigated, the COMPILER was byte-reproducible -- 0 of
    379 objects differed -- while the LINKER was not: `libggml-hip.so`'s host .text/.rodata
    layout varied across links of IDENTICAL objects (four distinct .so digests for one
    commit). Hashing the linked .so therefore aborted every keep on link noise that changes
    nothing about the kernel. Hashing the objects keys the guard on the layer that is
    deterministic AND that proves champion identity: a stale anchor (run 18's fault class,
    built from the wrong commit) has a DIFFERENT object for the mutated file, so it is still
    caught -- more precisely than before, because the diff names the file.

    Executables are excluded (`_IDENTITY_EXCLUDE`): they are target-set dependent and not
    the champion. None when the build dir holds no library objects."""
    root = Path(build_dir)
    rows = []
    for o in sorted(root.rglob("*.o")):
        rel = str(o.relative_to(root))
        if rel.startswith(_IDENTITY_EXCLUDE):
            continue
        rows.append(f"{rel}:{hashlib.sha256(o.read_bytes()).hexdigest()}")
    if not rows:
        return None
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def object_manifest(build_dir: Path | str) -> dict[str, str]:
    """Map each compiled object under `build_dir` to its sha256. The inputs the linker
    folds into LIBRARY -- so a digest mismatch can be localized to the compiler or the
    linker instead of only named (R23-45)."""
    root = Path(build_dir)
    return {str(o.relative_to(root)): hashlib.sha256(o.read_bytes()).hexdigest()
            for o in sorted(root.rglob("*.o"))}


def object_diff(anchor_build: Path | str, scratch_build: Path | str) -> dict:
    """Localize an anchor-guard digest mismatch. `linker_only` is True when the two
    builds share every object BYTE-FOR-BYTE yet LIBRARY still differs -- i.e. the
    compiler was deterministic and the LINKER is the non-deterministic step (the
    2026-09-04 host .text/.rodata layout drift). Otherwise `objects_differing` names the
    objects the compiler produced differently, which is a different bug entirely."""
    a, b = object_manifest(anchor_build), object_manifest(scratch_build)
    common = sorted(set(a) & set(b))
    differ = [k for k in common if a[k] != b[k]]
    return {"n_objects": len(common), "n_differing": len(differ),
            "objects_differing": differ[:20],
            "only_in_anchor": sorted(set(a) - set(b))[:20],
            "only_in_scratch": sorted(set(b) - set(a))[:20],
            # no COMMON object differs => whatever differs is the link (the executables in
            # only_in_* are target-set extras, not LIBRARY inputs)
            "linker_only": not differ}


__all__ = ["CODE_SECTIONS", "LIBRARY", "build_digest", "object_diff", "object_digest", "object_manifest", "code_digest",
           "section_spans"]
