"""Bounded, non-authoritative codegen evidence for a retained AutoKernel build.

This is an observation artifact, not a correctness or performance gate. In
particular, an MI210 build is not a CUDA build: PTX/SASS/CUBIN fields are never
inferred from an AMD binary, and register spills/occupancy need separate
compiler or profiler evidence that this collector does not have.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import tempfile
import time
from typing import Any


SCHEMA = "epyc.autokernel.codegen_summary.v1"
MAX_OBJECTS = 8
MAX_OBJECT_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_SCAN_FILES = 2048
TIMEOUT_S = 8.0
LLVM_OBJDUMP = Path("/opt/rocm/llvm/bin/llvm-objdump")
_INSTRUCTION = re.compile(r"^\s*[0-9a-f]+:\s+([a-z][a-z0-9_.]*)\b", re.I)


def _disassemble(path: Path) -> tuple[str | None, str]:
    """Read at most MAX_OUTPUT_BYTES; kill a noisy or stalled tool."""
    if not LLVM_OBJDUMP.is_file():
        return None, "llvm-objdump unavailable"
    proc = subprocess.Popen(
        (str(LLVM_OBJDUMP), "--disassemble", "--no-show-raw-insn", str(path)),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + TIMEOUT_S
    try:
        assert proc.stdout is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "disassembly timeout"
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                return None, "disassembly timeout"
            chunk = os.read(proc.stdout.fileno(), min(65536, MAX_OUTPUT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_OUTPUT_BYTES:
                return None, "disassembly output exceeds bound"
        if proc.wait(timeout=max(0.1, deadline - time.monotonic())) != 0:
            return None, "llvm-objdump failed"
        return b"".join(chunks).decode("utf-8", "replace"), "ok"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _objects(build_dir: Path) -> tuple[list[Path], str | None]:
    found: list[Path] = []
    visited = 0
    for root, dirs, files in os.walk(build_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(root) / d).is_symlink())
        visited += len(dirs)
        if visited > MAX_SCAN_FILES:
            return found, "build-tree directory scan exceeds bound"
        for name in sorted(files):
            visited += 1
            if visited > MAX_SCAN_FILES:
                return found, "build-tree file scan exceeds bound"
            if not name.endswith((".hsaco", ".co")):
                continue
            path = Path(root) / name
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_size > MAX_OBJECT_BYTES:
                continue
            found.append(path)
            if len(found) >= MAX_OBJECTS:
                return found, "code-object count reaches bound"
    return found, None


def summarize_codegen(backend: str, build_dir: str | Path) -> dict[str, Any]:
    """Return a bounded summary; unavailable evidence is explicit, never fabricated."""
    result: dict[str, Any] = {
        "schema": SCHEMA, "backend": backend, "authority": "diagnostic_only",
        "ptx_sass_cubin": "unavailable: non-CUDA backend",
        "register_spills": None, "occupancy": None,
        "vectorization": None, "instruction_mix": None,
        "objects": [], "status": "unavailable", "reason": None,
    }
    if backend == "llama_cpu":
        result["reason"] = "CPU machine-code analysis is not implemented"
        return result
    if backend != "llama_gpu":
        result["reason"] = "unsupported backend"
        return result
    root = Path(build_dir)
    if not root.is_dir():
        result["reason"] = "build directory unavailable"
        return result
    try:
        objects, limit_reason = _objects(root)
        if not objects:
            result["reason"] = limit_reason or (
                "no standalone AMD code object; embedded HIP fatbin extraction unavailable")
            return result
        totals = {"scalar": 0, "vector": 0, "matrix": 0, "memory": 0, "other": 0}
        for path in objects:
            with path.open("rb") as stream:
                raw = stream.read(MAX_OBJECT_BYTES + 1)
            if len(raw) > MAX_OBJECT_BYTES:
                result["objects"].append({
                    "relative_path": str(path.relative_to(root)),
                    "disassembly_status": "object grew beyond bound"})
                continue
            row: dict[str, Any] = {
                "relative_path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            disassembly, reason = _disassemble(path)
            row["disassembly_status"] = reason
            if disassembly is not None:
                counts = dict.fromkeys(totals, 0)
                for line in disassembly.splitlines():
                    match = _INSTRUCTION.match(line)
                    if not match:
                        continue
                    op = match.group(1)
                    kind = ("matrix" if "mfma" in op or "wmma" in op else
                            "memory" if any(term in op for term in
                                            ("load", "store", "buffer", "flat_", "ds_")) else
                            "vector" if op.startswith("v_") else
                            "scalar" if op.startswith("s_") else "other")
                    counts[kind] += 1
                    totals[kind] += 1
                row["instruction_mix"] = counts
            result["objects"].append(row)
        result["instruction_mix"] = totals if any(totals.values()) else None
        result["status"] = "partial" if result["instruction_mix"] else "unavailable"
        result["reason"] = limit_reason or (
            "spills, occupancy and vectorization require separate verified evidence")
    except (OSError, subprocess.SubprocessError) as exc:
        result["status"] = "unavailable"
        result["reason"] = f"codegen inspection failed: {type(exc).__name__}"
    return result


def retain_summary(store: Path, champion_head: str, *, backend: str,
                   build_dir: Path) -> dict[str, Any]:
    """Write one fsynced, content-stable sidecar for a committed variant.

    The caller also embeds the returned object in its attempt row. It should
    treat diagnostic collection failure as non-gating after a champion commit.
    """
    if re.fullmatch(r"[0-9a-f]{40}", champion_head) is None:
        raise ValueError("champion head must be a full SHA-1 commit id")
    summary = summarize_codegen(backend, build_dir)
    summary["champion_head"] = champion_head
    directory = store / "codegen"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{champion_head}.json"
    payload = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("existing codegen sidecar differs for champion head")
        return summary
    descriptor, temporary = tempfile.mkstemp(prefix=f".{champion_head}.", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return summary
