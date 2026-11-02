"""Fail-closed per-cell HPR exposure reservation; no inference or policy verdict.

The sealed inventory must be prepared from native evidence by the owner. A
different or missing corpus/selection is UNKNOWN, never presumed unexposed.
Reserve a complete cell before its first launch; a crash leaves it reserved.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reserve(ledger: Path, expected_ledger_sha256: str, claims_dir: Path, *,
            corpus_sha256: str, selection_sha256: str, model: str, unit: str,
            arm: str, wave: int, attempt_id: str) -> Path:
    ledger_bytes = ledger.read_bytes()
    if sha(ledger_bytes) != expected_ledger_sha256:
        raise ValueError("exposure ledger digest mismatch")
    doc = json.loads(ledger_bytes)
    if doc.get("schema") != "epyc.hpr.exposure-ledger.v1":
        raise ValueError("exposure ledger schema")
    expected = {"corpus_sha256": corpus_sha256, "selection_sha256": selection_sha256,
                "model": model, "unit": unit}
    if doc.get("cohort") != expected:
        raise ValueError("cohort mismatch: disjointness is unknown")
    if (arm not in ("dense", "coarse") or type(wave) is not int or wave not in (0, 1, 2)
            or not isinstance(attempt_id, str) or not attempt_id.strip()):
        raise ValueError("invalid cell or attempt")
    rows = doc.get("cells")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("complete six-cell exposure inventory required")
    table = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("arm") not in ("dense", "coarse") or type(row.get("wave")) is not int or row["wave"] not in (0, 1, 2):
            raise ValueError("invalid inventory cell")
        key = (row["arm"], row["wave"])
        if key in table or row.get("state") not in ("unexposed", "exposed_complete", "exposed_unknown"):
            raise ValueError("duplicate or invalid inventory state")
        if not isinstance(row.get("evidence"), str) or not row["evidence"].strip():
            raise ValueError("cell evidence missing")
        table[key] = row
    if set(table) != {(a, w) for a in ("dense", "coarse") for w in range(3)}:
        raise ValueError("incomplete exposure inventory")
    if table[(arm, wave)]["state"] != "unexposed":
        raise ValueError("cell already exposed or unknown: refusing redraw")
    if not claims_dir.is_dir() or claims_dir.is_symlink():
        raise ValueError("existing non-symlink reservation directory required")
    # Binary or port changes do not make the same cohort/cell new.
    token = sha(json.dumps({**expected, "arm": arm, "wave": wave}, sort_keys=True,
                           separators=(",", ":")).encode())
    path = claims_dir / (token + ".json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        payload = {"schema": "epyc.hpr.exposure-reservation.v1", "cohort": expected,
                   "arm": arm, "wave": wave, "attempt_id": attempt_id,
                   "ledger_sha256": expected_ledger_sha256,
                   "state": "reserved_before_launch_may_be_exposed"}
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        dirfd = os.open(claims_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    except BaseException:
        # Never remove a reservation after a partial write; uncertainty stays exposed.
        raise
    return path
