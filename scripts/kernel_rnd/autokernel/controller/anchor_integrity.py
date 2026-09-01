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


__all__ = ["CODE_SECTIONS", "LIBRARY", "build_digest", "code_digest",
           "section_spans"]
