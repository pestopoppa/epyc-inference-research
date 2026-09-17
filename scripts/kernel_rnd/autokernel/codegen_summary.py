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
from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA = "epyc.autokernel.codegen_summary.v1"
MAX_OBJECTS = 8
MAX_OBJECT_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_SCAN_FILES = 2048
MAX_CMAKE_CACHE_BYTES = 2 * 1024 * 1024
TIMEOUT_S = 8.0
MAX_TOTAL_S = 12.0
LLVM_OBJDUMP = Path("/opt/rocm/llvm/bin/llvm-objdump")
CPU_OBJDUMP = Path("/usr/bin/objdump")
CPU_LIBRARY = "libggml-cpu.so"
# Exported wrappers relevant to the current CPU GDN / quant-dot search. This
# is a diagnostic sample, not a claim that every inlined helper was inspected.
CPU_SYMBOLS = (
    "ggml_compute_forward_gated_delta_net",
    "ggml_vec_dot_q4_K_q8_K",
    "ggml_vec_dot_q5_K_q8_K",
    "ggml_vec_dot_q6_K_q8_K",
)
_INSTRUCTION = re.compile(r"^\s*[0-9a-f]+:\s+([a-z][a-z0-9_.]*)\b", re.I)
_CPU_SYMBOL_HEADER = re.compile(r"^\s*[0-9a-f]+\s+<([^>]+)>:\s*$", re.I | re.M)


def _disassemble(path: Path, *, timeout_s: float = TIMEOUT_S) -> tuple[str | None, str]:
    """Read at most MAX_OUTPUT_BYTES; kill a noisy or stalled tool."""
    if not LLVM_OBJDUMP.is_file():
        return None, "llvm-objdump unavailable"
    proc = subprocess.Popen(
        (str(LLVM_OBJDUMP), "--disassemble", "--no-show-raw-insn", str(path)),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + timeout_s
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
        if proc.stdout is not None:
            proc.stdout.close()


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


def _cpu_library(build_dir: Path) -> tuple[Path | None, str]:
    """Resolve the bounded installed ggml CPU DSO, never an ambient library."""
    try:
        root = build_dir.resolve(strict=True)
        library = (root / "bin" / CPU_LIBRARY).resolve(strict=True)
        library.relative_to(root / "bin")
        if not library.is_file():
            return None, "candidate CPU library is not a regular file"
        if library.stat().st_size > MAX_OBJECT_BYTES:
            return None, "candidate CPU library exceeds inspection bound"
        return library, "ok"
    except (OSError, ValueError):
        return None, "candidate CPU library missing or outside build bin"


def _disassemble_cpu(path: Path, symbol: str, *, timeout_s: float) -> tuple[str | None, str]:
    """Disassemble one allowlisted CPU symbol with a wall/output bound."""
    if symbol not in CPU_SYMBOLS:
        return None, "CPU symbol is not allowlisted"
    if not CPU_OBJDUMP.is_file():
        return None, "CPU objdump unavailable"
    proc = subprocess.Popen(
        (str(CPU_OBJDUMP), "--disassemble=" + symbol,
         "--no-show-raw-insn", str(path)),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + timeout_s
    try:
        assert proc.stdout is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "CPU disassembly timeout"
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                return None, "CPU disassembly timeout"
            chunk = os.read(proc.stdout.fileno(), min(65536, MAX_OUTPUT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_OUTPUT_BYTES:
                return None, "CPU disassembly output exceeds bound"
        if proc.wait(timeout=max(0.1, deadline - time.monotonic())) != 0:
            return None, "CPU objdump failed"
        output = b"".join(chunks).decode("utf-8", "replace")
        if symbol not in _CPU_SYMBOL_HEADER.findall(output):
            return None, "CPU symbol absent from disassembly"
        return output, "ok"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


def _cpu_summary(build_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    library, reason = _cpu_library(build_dir)
    if library is None:
        result["reason"] = reason
        return result
    with library.open("rb") as stream:
        raw = stream.read(MAX_OBJECT_BYTES + 1)
    if len(raw) > MAX_OBJECT_BYTES:
        result["reason"] = "candidate CPU library grew beyond inspection bound"
        return result
    row: dict[str, Any] = {
        "relative_path": str(library.relative_to(build_dir.resolve())),
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "disassembly_status": "unavailable", "symbols": []}
    totals = {"scalar": 0, "vector": 0, "matrix": 0, "memory": 0, "other": 0}
    deadline = time.monotonic() + MAX_TOTAL_S
    failures = []
    for symbol in CPU_SYMBOLS:
        remaining = deadline - time.monotonic()
        disassembly, status = (
            _disassemble_cpu(library, symbol, timeout_s=min(TIMEOUT_S, remaining))
            if remaining > 0 else (None, "total collector time budget exhausted"))
        if disassembly is None:
            failures.append(f"{symbol}: {status}")
            continue
        counts = dict.fromkeys(totals, 0)
        in_symbol = False
        for line in disassembly.splitlines():
            header = _CPU_SYMBOL_HEADER.match(line)
            if header:
                in_symbol = header.group(1) == symbol
                continue
            if not in_symbol:
                continue
            match = _INSTRUCTION.match(line)
            if not match:
                continue
            op = match.group(1)
            kind = "memory" if "(%" in line or "[" in line else \
                   "vector" if op.startswith("v") else "scalar"
            counts[kind] += 1
            totals[kind] += 1
        if any(counts.values()):
            row["symbols"].append({"name": symbol, "instruction_mix": counts})
        else:
            failures.append(f"{symbol}: no parsed instructions")
    if row["symbols"]:
        row["instruction_mix"] = totals
        row["disassembly_status"] = "ok"
        result["instruction_mix"] = totals
        result["status"] = "partial"
    result["objects"] = [row]
    result["reason"] = "; ".join(failures) if failures else (
        "spills, occupancy and vectorization require separate verified evidence")
    return result


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
        return _cpu_summary(Path(build_dir), result)
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
        deadline = time.monotonic() + MAX_TOTAL_S
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
            remaining = deadline - time.monotonic()
            disassembly, reason = (
                _disassemble(path, timeout_s=min(TIMEOUT_S, remaining))
                if remaining > 0 else (None, "total collector time budget exhausted"))
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


def _toolchain_identity(build_dir: Path) -> dict[str, Any]:
    """Capture bounded compiler declarations from this build, not ambient PATH."""
    objdump_stat = None
    try:
        stat = LLVM_OBJDUMP.stat()
        objdump_stat = {"path": str(LLVM_OBJDUMP.resolve()),
                        "device": stat.st_dev, "inode": stat.st_ino,
                        "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        pass
    cache = build_dir / "CMakeCache.txt"
    try:
        with cache.open("rb") as stream:
            raw = stream.read(MAX_CMAKE_CACHE_BYTES + 1)
    except OSError:
        return {"status": "unavailable", "reason": "CMakeCache.txt unavailable",
                "objdump_stat": objdump_stat}
    if len(raw) > MAX_CMAKE_CACHE_BYTES:
        return {"status": "unavailable", "reason": "CMakeCache.txt exceeds bound",
                "objdump_stat": objdump_stat}
    wanted = ("CMAKE_CXX_COMPILER", "CMAKE_C_COMPILER", "CMAKE_HIP_COMPILER",
              "CMAKE_CUDA_COMPILER", "CMAKE_CXX_COMPILER_ID",
              "CMAKE_CXX_COMPILER_VERSION")
    declarations = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        if "=" not in line or line.startswith(("#", "//")):
            continue
        key, value = line.split("=", 1)
        name = key.split(":", 1)[0]
        if name in wanted:
            declarations[name] = value
    compiler_path = declarations.get("CMAKE_HIP_COMPILER") or declarations.get(
        "CMAKE_CXX_COMPILER")
    compiler_stat = None
    if compiler_path:
        try:
            compiler = Path(compiler_path).resolve(strict=True)
            stat = compiler.stat()
            if compiler.is_file():
                compiler_stat = {"path": str(compiler), "device": stat.st_dev,
                                 "inode": stat.st_ino, "bytes": stat.st_size,
                                 "mtime_ns": stat.st_mtime_ns}
        except OSError:
            pass
    return {"status": "captured" if declarations else "unavailable",
            "cmake_cache_sha256": hashlib.sha256(raw).hexdigest(),
            "declarations": declarations, "compiler_stat": compiler_stat,
            "objdump_stat": objdump_stat,
            "identity_limit": "paths/stat/cache digest; compiler binary content not hashed"}


def _belief_claim_tuple(summary: Mapping[str, Any], *, attempt_identity: str,
                        source_tree_oid: str, observed_at: str) -> dict[str, Any]:
    """Producer-authored ClaimTuple-shaped observation; no local grading rule."""
    available = summary["status"] == "partial" and summary["instruction_mix"] is not None
    extra = {
        "authority": "diagnostic_only", "not_throughput_or_correctness": True,
        "not_occupancy_evidence": True,
        "attempt_identity": attempt_identity,
        "retained_source_commit": summary["champion_head"],
        "retained_source_tree_oid": source_tree_oid,
        "backend": summary["backend"], "toolchain": summary["toolchain"],
        "build_frame_sha256": summary["build_frame_sha256"],
        "summary_core_sha256": summary["summary_core_sha256"],
        "code_objects": [{"relative_path": row["relative_path"],
                          "sha256": row["sha256"]}
                         for row in summary["objects"] if "sha256" in row],
        "instruction_mix": summary["instruction_mix"],
        "unavailable_fields": [name for name in
                               ("register_spills", "occupancy", "vectorization")
                               if summary[name] is None],
        "ptx_sass_cubin": summary["ptx_sass_cubin"],
    }
    if summary["backend"] == "llama_cpu":
        extra["disassembled_symbols"] = [
            symbol["name"] for row in summary["objects"]
            for symbol in row.get("symbols", [])]
    return {
        "measurement_id": "ak-codegen:" + attempt_identity + ":" + summary["build_frame_sha256"],
        "metric": "codegen_disassembly_availability",
        "value": int(available), "unit": "availability indicator",
        "metric_direction": "higher_better",  # evidence coverage, never kernel performance
        "category": "CANDIDATE",
        "claim": ("Bounded native code-object disassembly was available for this retained build"
                  if available else
                  "Bounded native code-object disassembly was unavailable for this retained build"),
        "date": observed_at,
        "protocol_id": "",  # diagnostic observation; no decision-grade protocol
        "reps": 1, "reps_basis": "one retained build; not benchmark repetitions",
        "attestation_locator": summary["artifact_ref"],
        # The summary and tuple live in one file; a full-file self-hash is
        # impossible. The reader re-derives the core digest below instead.
        "attestation_sha256": "", "attestation_verified": None,
        "source_class": "measurement",
        "extra": extra,
    }


def retain_summary(store: Path, champion_head: str, *, backend: str,
                   build_dir: Path, recipe: Mapping[str, Any] | None = None,
                   attempt_identity: str | None = None,
                   source_tree_oid: str | None = None) -> dict[str, Any]:
    """Write one fsynced, content-stable sidecar for a committed variant.

    The caller also embeds the returned object in its attempt row. It should
    treat diagnostic collection failure as non-gating after a champion commit.
    """
    if re.fullmatch(r"[0-9a-f]{40}", champion_head) is None:
        raise ValueError("champion head must be a full SHA-1 commit id")
    summary = summarize_codegen(backend, build_dir)
    summary["champion_head"] = champion_head
    if (attempt_identity is None) != (source_tree_oid is None):
        raise ValueError("attempt identity and source tree must be supplied together")
    if source_tree_oid is not None and re.fullmatch(r"[0-9a-f]{40}", source_tree_oid) is None:
        raise ValueError("source tree must be a full Git tree OID")
    if attempt_identity is not None and not attempt_identity.strip():
        raise ValueError("attempt identity must be nonempty")
    summary["toolchain"] = _toolchain_identity(Path(build_dir))
    if backend == "llama_cpu":
        try:
            stat = CPU_OBJDUMP.stat()
            summary["toolchain"]["cpu_objdump_stat"] = {
                "path": str(CPU_OBJDUMP.resolve()), "device": stat.st_dev,
                "inode": stat.st_ino, "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
        except OSError:
            summary["toolchain"]["cpu_objdump_stat"] = None
    binary = Path(build_dir) / "bin" / "llama-bench"
    try:
        stat = binary.stat()
        binary_stat = {"device": stat.st_dev, "inode": stat.st_ino,
                       "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        binary_stat = None
    # This is a build-frame identity, not a content hash of an embedded fatbin.
    # The bounded standalone object hashes are already in the summary; a missing
    # binary/object remains visible and cannot silently become codegen evidence.
    frame = {"backend": backend, "build_dir": str(Path(build_dir).resolve()),
             "recipe": dict(recipe or {}), "binary_stat": binary_stat,
             "toolchain": summary["toolchain"],
             "object_sha256s": [row["sha256"] for row in summary["objects"]
                                if "sha256" in row]}
    frame_digest = hashlib.sha256(json.dumps(
        frame, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary["build_frame"] = frame
    summary["build_frame_sha256"] = frame_digest
    directory = store / "codegen"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{champion_head}.{backend}.{frame_digest}.json"
    summary["artifact_ref"] = f"codegen/{name}"
    if attempt_identity is not None:
        summary["attempt_identity"] = attempt_identity
        summary["source_tree_oid"] = source_tree_oid
        # This digest covers every native evidence field and the exact build
        # frame, but not the tuple that cites it. The strict reader must check
        # it before projecting the tuple; no reader-side tuple reconstruction.
        summary["summary_core_sha256"] = hashlib.sha256(json.dumps(
            summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        summary["belief_claim_tuple"] = _belief_claim_tuple(
            summary, attempt_identity=attempt_identity,
            source_tree_oid=source_tree_oid,
            observed_at=datetime.now(timezone.utc).isoformat())
    destination = directory / name
    payload = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{champion_head}.", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # create-only: a concurrent same-head writer may win, but cannot be
            # overwritten between an exists check and a rename.
            os.link(temporary, destination)
        except FileExistsError:
            previous = destination.read_bytes()
            if previous != payload:
                try:
                    existing = json.loads(previous)
                    core = {key: value for key, value in existing.items()
                            if key not in {"summary_core_sha256", "belief_claim_tuple"}}
                    valid_existing = (summary.get("summary_core_sha256")
                        and existing.get("summary_core_sha256") == summary["summary_core_sha256"]
                        and hashlib.sha256(json.dumps(
                            core, sort_keys=True, separators=(",", ":")).encode()
                            ).hexdigest() == summary["summary_core_sha256"])
                except (ValueError, TypeError, AttributeError):
                    valid_existing = False
                if not valid_existing:
                    raise ValueError("existing codegen sidecar differs for build frame")
                return existing
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return summary
