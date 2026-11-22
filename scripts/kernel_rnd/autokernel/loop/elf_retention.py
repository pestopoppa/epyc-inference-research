"""AKU-10f2: retain what a candidate build's ELF actually loads, read outside the mutex.

The candidate manifest declares DSOs, but the loader follows the executable's own
DT_NEEDED / DT_RUNPATH (or DT_RPATH). A DSO or RUNPATH directory the build needs but
did not declare must still be retained. This module parses the ELF64 dynamic section
with bounded seeks (no subprocess, no whole-file read) and resolves NEEDED names only
inside the binary's own RUNPATH directories; names resolved elsewhere (system libc,
ld.so cache) are not build artifacts and are not retained here.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import struct

PT_LOAD, PT_DYNAMIC = 1, 2
DT_NULL, DT_NEEDED, DT_STRTAB, DT_STRSZ, DT_RPATH, DT_RUNPATH = 0, 1, 5, 10, 15, 29
MAX_PHDRS, MAX_DYN, MAX_STR = 256, 4096, 1 << 20


class ElfUnreadable(ValueError):
    pass


def read_dynamic(path: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (DT_NEEDED names, RUNPATH-or-RPATH entries) of one ELF64 little-endian file."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
            if len(head) < 64 or head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
                raise ElfUnreadable(f"not an ELF64 little-endian file: {path}")
            phoff, = struct.unpack_from("<Q", head, 32)
            phentsize, phnum = struct.unpack_from("<HH", head, 54)
            if phnum > MAX_PHDRS or phentsize < 56:
                raise ElfUnreadable(f"ELF program headers out of bounds: {path}")
            handle.seek(phoff)
            table = handle.read(phentsize * phnum)
            loads, dynamic = [], None
            for index in range(phnum):
                p_type, _f, off, vaddr, _pa, filesz, _m, _a = struct.unpack_from(
                    "<IIQQQQQQ", table, index * phentsize)
                if p_type == PT_LOAD:
                    loads.append((vaddr, off, filesz))
                elif p_type == PT_DYNAMIC:
                    dynamic = (off, min(filesz, MAX_DYN * 16))
            if dynamic is None:
                return (), ()  # static executable: loads nothing
            handle.seek(dynamic[0])
            raw = handle.read(dynamic[1])
            entries = [struct.unpack_from("<qQ", raw, i) for i in range(0, len(raw) - 15, 16)]
            tags: dict[int, list[int]] = {}
            for tag, value in entries:
                if tag == DT_NULL:
                    break
                tags.setdefault(tag, []).append(value)
            strtab = next((off + tags[DT_STRTAB][0] - vaddr for vaddr, off, size in loads
                           if DT_STRTAB in tags and vaddr <= tags[DT_STRTAB][0] < vaddr + size), None)
            if strtab is None:
                raise ElfUnreadable(f"ELF dynamic string table is unmapped: {path}")
            handle.seek(strtab)
            strings = handle.read(min(tags.get(DT_STRSZ, [MAX_STR])[0], MAX_STR))
    except OSError as exc:
        raise ElfUnreadable(f"ELF unreadable: {path}: {exc}") from exc

    def text(offset: int) -> str:
        end = strings.find(b"\0", offset)
        if offset >= len(strings) or end < 0:
            raise ElfUnreadable(f"ELF dynamic string out of bounds: {path}")
        return strings[offset:end].decode("utf-8", "surrogateescape")

    needed = tuple(text(value) for value in tags.get(DT_NEEDED, ()))
    paths = tags.get(DT_RUNPATH) or tags.get(DT_RPATH) or ()
    runpath = tuple(item for value in paths for item in text(value).split(":") if item)
    return needed, runpath


def resolve(executable: str, needed, runpath) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Existing RUNPATH directories ($ORIGIN expanded) and NEEDED files found inside them."""
    origin = str(Path(executable).parent)
    dirs = []
    for entry in runpath:
        directory = os.path.normpath(entry.replace("${ORIGIN}", origin).replace("$ORIGIN", origin))
        if os.path.isabs(directory) and os.path.isdir(directory) and directory not in dirs:
            dirs.append(directory)
    found = []
    for name in needed:
        hit = next((os.path.join(d, name) for d in dirs if os.path.isfile(os.path.join(d, name))), None)
        if hit is not None and hit not in found:
            found.append(hit)
    return tuple(dirs), tuple(found)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extend_view(manifests, *, build_id_of, directory_id, dso_id, make_artifact):
    """Yield (executable_artifact_id, extra CatalogArtifacts, extra dep ids) or uncertainty.

    Called by build_native_view outside the controller mutex. A present but unparsable
    executable yields an uncertain scope (fail closed); an absent path is left to the
    existing identity verification, which already refuses it.
    """
    extras, deps, uncertain = [], {}, []
    for manifest in manifests:
        for build in manifest.builds:
            path = build.executable.path
            if not os.path.isfile(path):
                continue
            try:
                dirs, files = resolve(path, *read_dynamic(path))
            except ElfUnreadable:
                uncertain.append(f"elf_dependencies_unreadable:{build.build_id}")
                continue
            declared = {item.path for item in build.dsos}
            ids = []
            for directory in dirs:
                extras.append(make_artifact(directory_id(directory), "shared_dso_dir", directory, None))
                ids.append(directory_id(directory))
            for file in files:
                if file in declared:
                    continue
                artifact_id = dso_id(file, sha256_file(file))
                extras.append(make_artifact(artifact_id, "candidate_artifact", file, sha256_file(file)))
                ids.append(artifact_id)
            if ids:
                key = build_id_of(build.executable)
                deps[key] = sorted({*deps.get(key, ()), *ids})
    return extras, deps, uncertain
