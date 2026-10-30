"""Prospective, append-only capture journal for workload census producers.

A journal is evidence of what its caller recorded, not proof of a successful
census.  The caller must append start before launch and terminal after reaping;
failures are terminal events too.  Existing census records are never recaptured.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

SCHEMA = "epyc.autokernel.census_capture_event.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"dev": st.st_dev, "ino": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def append_event(path: Path, event: Mapping[str, Any]) -> str:
    """Persist one sealed event; fail if an existing journal is malformed.

    Each event chains to the preceding content hash. Owner-only permissions
    limit access to a newly created journal;
    the caller must serialize all later appends for this run.
    """
    path = Path(path)
    data = dict(event)
    if "schema" in data or "previous_sha256" in data or "content_sha256" in data:
        raise ValueError("reserved capture field")
    previous = ""
    if path.exists():
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            seal = row.pop("content_sha256")
            if row.get("schema") != SCHEMA or row.get("previous_sha256") != previous:
                raise ValueError("invalid capture chain")
            if hashlib.sha256(canonical(row)).hexdigest() != seal:
                raise ValueError("invalid capture seal")
            previous = seal
    data.update(schema=SCHEMA, previous_sha256=previous)
    seal = hashlib.sha256(canonical(data)).hexdigest()
    data["content_sha256"] = seal
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "ab") as stream:
        stream.write(canonical(data) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return seal


def _refs(paths: Mapping[str, str]) -> dict[str, dict[str, str]]:
    if not paths or any(not key or not value for key, value in paths.items()):
        raise ValueError("nonempty named file references required")
    return {key: {"path": str(Path(value).resolve(strict=True)),
                  "sha256": file_sha256(Path(value))} for key, value in paths.items()}


def _retain(path: Path, evidence: Path) -> dict[str, str]:
    """Copy small write-time evidence, recording the resolved origin separately."""
    origin = path.resolve(strict=True)
    before = file_identity(origin)
    with origin.open("rb") as stream:
        raw = stream.read((64 << 20) + 1)
    if len(raw) > 64 << 20 or file_identity(origin) != before:
        raise ValueError(f"input too large or changed while captured: {origin}")
    digest = hashlib.sha256(raw).hexdigest()
    evidence.mkdir(parents=True, exist_ok=True, mode=0o700)
    dest = evidence / digest
    try:
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        if file_sha256(dest) != digest:
            raise ValueError(f"retained evidence differs: {dest}")
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return {"origin_path": str(origin), "path": str(dest.resolve()), "sha256": digest}


def _retain_manifest_files(ref: Mapping[str, str], evidence: Path) -> dict[str, dict[str, str]]:
    """Preserve manifest bytes and separately retain every source they name."""
    result = {}
    for line in Path(ref["path"]).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or not Path(match[2]).is_absolute() or match[2] in result:
            raise ValueError("malformed or duplicate native source manifest entry")
        item = _retain(Path(match[2]), evidence)
        if item["sha256"] != match[1] or item["origin_path"] != match[2]:
            raise ValueError(f"manifest source changed: {match[2]}")
        result[match[2]] = item
    if not result:
        raise ValueError("empty native source manifest")
    return result


def begin(path: Path, *, attempt_id: str, started_ns: int, argv: list[str],
          env: Mapping[str, str], shape: Mapping[str, Any], grant: str,
          deadline: str, kernel_commit: str, inputs: Mapping[str, str]) -> str:
    """Write before Popen; inputs include binary, model, recipe, binding and sources."""
    required = {"binary", "model", "recipe", "binding", "producer", "driver", "grant",
                "capture_module", "residency", "workload_contract", "build_receipt_binding", "build_complete",
                "build_log", "elf_manifest", "build_job", "build_sources"}
    if Path(path).exists():
        raise ValueError("capture path reused")
    if (not attempt_id or started_ns <= 0 or not argv or not env or not grant or
            not deadline or not kernel_commit or not required <= set(inputs) or not shape):
        append_event(path, {"kind": "refusal", "attempt_id": attempt_id,
                            "started_ns": started_ns, "reason": "incomplete prelaunch capture"})
        raise ValueError("incomplete or reused prelaunch capture")
    try:
        evidence = Path(path).parent / "evidence"
        references = {key: (_refs({key: value})[key] if key in {"binary", "model"}
                            else _retain(Path(value), evidence)) for key, value in inputs.items()}
        if Path(references["grant"]["path"]).read_bytes() != grant.encode():
            raise ValueError("grant changed between read and capture")
        manifests = {key: _retain_manifest_files(references[key], evidence)
                     for key in ("build_sources", "source_manifest") if key in references}
    except (OSError, ValueError) as exc:
        append_event(path, {"kind": "refusal", "attempt_id": attempt_id,
                            "started_ns": started_ns, "reason": str(exc)})
        raise
    return append_event(path, {"kind": "start", "attempt_id": attempt_id,
        "started_ns": started_ns, "argv": argv, "env": dict(env), "shape": dict(shape),
        "grant": grant, "grant_sha256": hashlib.sha256(grant.encode()).hexdigest(),
        "deadline": deadline, "kernel_commit": kernel_commit, "inputs": references,
        "manifest_sources": manifests})


def finish(path: Path, *, attempt_id: str, ended_ns: int, rc: int | None,
           error: str, samples: list[dict[str, Any]], log: str | None,
           loaded_dsos: Mapping[str, str], spawned_ns: int | None = None,
           loaded_before: Mapping[str, Mapping[str, int]] | None = None) -> str:
    """Write after reap, including partial/refused attempts; never invent a result."""
    lines = Path(path).read_bytes().splitlines()
    if len(lines) != 1:
        raise ValueError("attempt must have exactly one start")
    start = json.loads(lines[0])
    if start.get("kind") != "start" or start.get("attempt_id") != attempt_id:
        raise ValueError("start does not match attempt")
    if ended_ns < start["started_ns"] or (rc is None and not error):
        raise ValueError("terminal outcome or time missing")
    evidence_error = ""
    log_ref = None
    try:
        log_ref = _refs({"log": log})["log"] if log else None
        dsos = {}
        for name, dso_path in loaded_dsos.items():
            before = dict((loaded_before or {})[dso_path])
            path_obj = Path(dso_path)
            if file_identity(path_obj) != before:
                raise ValueError(f"loaded DSO changed before hashing: {dso_path}")
            ref = _refs({name: dso_path})[name]
            after = file_identity(path_obj)
            if before != after:
                raise ValueError(f"loaded DSO changed while hashing: {dso_path}")
            dsos[name] = {**ref, "mapped_identity": before, "hashed_identity": after}
    except (OSError, ValueError, KeyError) as exc:
        dsos = {}
        evidence_error = f"terminal evidence unavailable: {exc}"
        error = (error + "; " if error else "") + evidence_error
    seal = append_event(path, {"kind": "terminal", "attempt_id": attempt_id,
        "ended_ns": ended_ns, "spawned_ns": spawned_ns, "rc": rc, "error": error, "samples": samples,
        "log": log_ref, "loaded_dsos": dsos})
    if evidence_error:
        raise ValueError(evidence_error)
    return seal


def seal_record(path: Path, *, record: Path, journals: list[Path]) -> str:
    """Bind a native record and its already completed attempt journals."""
    if Path(path).exists() or not journals:
        raise ValueError("new sidecar and attempts required")
    refs = _refs({f"attempt_{i}": str(p) for i, p in enumerate(journals)})
    for p in journals:
        events = [json.loads(x) for x in Path(p).read_bytes().splitlines()]
        if len(events) != 2 or events[0].get("kind") != "start" or events[1].get("kind") != "terminal":
            raise ValueError("attempt not terminal")
    return append_event(path, {"kind": "record", "record": _refs({"record": str(record)})["record"],
                               "attempts": refs})
