#!/usr/bin/env python3
"""Replay recipe-derived HIP CU IDs with pinned inputs; no expected-symbol input."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(
    manifest: Path,
    manifest_sha: str,
    source_root: Path,
    output_root: Path,
    argv: list[str],
    cwd: Path | None = None,
) -> tuple[list[str], dict]:
    if file_sha(manifest) != manifest_sha:
        raise ValueError("CUID manifest changed")
    m = json.loads(manifest.read_bytes())
    compiler = Path(argv[0]) if argv else Path("/")
    if (
        str(compiler) != m["compiler"]
        or file_sha(compiler.resolve(strict=True)) != m["compiler_sha256"]
    ):
        raise ValueError("compiler identity changed")
    for name, expected in m["config_files"].items():
        if file_sha(Path(name)) != expected:
            raise ValueError("compiler configuration changed")
    ids = [r["cuid"] for r in m["tu"].values()]
    if len(set(ids)) != len(ids) or any(
        not x or any(c not in "0123456789abcdef" for c in x) for x in ids
    ):
        raise ValueError("TU IDs must be nonempty distinct hex")
    if any(
        a.startswith(("-cuid=", "-fuse-cuid=", "-fhip-cuid=")) or a.startswith("@")
        for a in argv[1:]
    ):
        raise ValueError("unreviewed CUID override/response file")
    if any(
        a.startswith(("-Xclang", "-Xarch_", "--config", "--no-default-config", "-cc1"))
        for a in argv[1:]
    ):
        raise ValueError("unreviewed driver forwarding/configuration override")
    here = (cwd or Path.cwd()).resolve()
    sources = []
    outputs = []
    dependency_files = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-o", "-x", "-I", "-D", "-isystem", "-include", "-MF", "-MT", "-MQ"):
            if i + 1 >= len(argv):
                raise ValueError("missing option value")
            if a == "-o":
                outputs.append(argv[i + 1])
            elif a == "-MF":
                dependency_files.append(argv[i + 1])
            i += 2
            continue
        if a.startswith("-MF"):
            dependency_files.append(a[3:])
            i += 1
            continue
        if not a.startswith("-"):
            sources.append(a)
        i += 1
    if len(sources) != 1 or len(outputs) != 1:
        raise ValueError("exactly one explicit source/output required")
    root = source_root.resolve(strict=True)
    source = (here / sources[0]).resolve(strict=True)
    try:
        key = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("TU outside private source root") from exc
    if key not in m["tu"]:
        raise ValueError("unreviewed TU")
    row = m["tu"][key]
    if file_sha(source) != row["source_sha256"]:
        raise ValueError("TU bytes differ from retained recipe")
    output = (here / outputs[0]).resolve()
    allowed = output_root.resolve(strict=True)
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("output outside private build root")
    for name in dependency_files:
        dependency = (here / name).resolve()
        if (
            not name
            or name == "-"
            or dependency == allowed
            or not dependency.is_relative_to(allowed)
        ):
            raise ValueError("dependency output outside private build root")
    # -cuid is a literal per-TU driver input, NOT the __hip_cuid suffix (CodeGen hashes it again).
    command = [argv[0], "-cuid=" + row["cuid"], *argv[1:]]
    return command, {
        "TU": key,
        "driver_cuid": row["cuid"],
        "manifest_sha256": manifest_sha,
        "source_sha256": row["source_sha256"],
        "output": str(output),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    command, proof = prepare(a.manifest, a.manifest_sha256, a.source_root, a.output_root, cmd)
    print("HIP_CUID_REPLAY " + json.dumps(proof, sort_keys=True), flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
