"""Finalization record for the native actor-wire matrix (VB-AK-NATIVE-WIRE-WRITE, producer side).

``actor-wire-pass.json`` exists only on PASS and ``matrix-results.json`` carries no schema or hash, so a
FAIL or aborted run left nothing a reader could bind. ``finalization`` wraps the matrix driver and, on
EVERY exit (PASS, FAIL, abort, exception), writes ``actor-wire-final.json`` into the evidence directory
(the reader resolves ``<seat-digest>.json`` next to the record, so that is where it can verify them).
Shape and checks mirror root ``scripts/vidya/adapters/autokernel_actor_wire.validate_final``.
Lives outside ``loop/`` so the frozen loop code-line ceilings are untouched.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from .loop import actor_wire_gate as gate

FINAL_SCHEMA = "epyc.autokernel.actor_wire_final.v1"
FINAL_NAME = "actor-wire-final.json"
MODULE = "autokernel.loop.actor_wire_test"
REPO = "epyc-inference-research"


def seal(record: dict) -> dict:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return {**body, "record_sha256": gate.digest(body)}


def producer_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out or "unknown"


def _seat_cleanup(evidence: Path, key: str) -> dict:
    """Owned-call cleanup from the seat's own diagnostics; missing evidence is never verified."""
    try:
        receipts = json.loads((evidence / f"{key}.json").read_text())["diagnostics"]["owned_call_cleanup"]
    except (OSError, ValueError, KeyError, TypeError):
        receipts = []
    ok = bool(receipts) and all(isinstance(r, dict) and r.get("cleanup_verified") is True
                                and r.get("census_verified") is True and not r.get("survivors")
                                for r in receipts)
    return {"receipts": len(receipts), "all_verified": ok}


def build(request: dict, outcomes: list, results: dict, evidence: Path, abort: str | None,
          cleanup_uncertain: bool) -> dict:
    seats = request["seats"]
    keys = [gate.digest(s) for s in seats]
    attempted = [o["seat"] for o in outcomes]
    complete = len(outcomes) == len(seats) and abort is None
    ok = complete and all(o["status"] == "PASS" for o in outcomes)
    shas = {}
    for key in attempted:
        path = evidence / f"{key}.json"
        shas[key] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
    return seal({
        "schema": FINAL_SCHEMA, "finalized_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": {"repo": REPO, "commit": producer_commit(), "module": MODULE},
        "request": {k: request[k] for k in ("schema", "seats", "actor_config_digest", "suite")},
        "status": "PASS" if ok else "FAIL", "outcomes": [dict(o) for o in outcomes],
        "attempted": len(outcomes), "expected": len(seats), "abort": abort,
        "unvisited": keys[len(outcomes):],
        "results": {k: v for k, v in results.items() if k in attempted},
        "evidence": shas,
        "cleanup": {"cleanup_uncertain": cleanup_uncertain,
                    "seats": {k: _seat_cleanup(evidence, k) for k in attempted}}})


@contextmanager
def finalization(request, evidence_dir, results, outcomes):
    """Write the record at the end of every run. ``results``/``outcomes`` are the driver's live containers."""
    state = {"abort": None, "uncertain": False}
    try:
        yield
    except BaseException as exc:
        # The driver's own end-of-matrix "seats failed" AssertionError is a FAIL, not an abort.
        matrix_fail = (isinstance(exc, AssertionError) and len(outcomes) == len(request["seats"])
                       and any(o["status"] == "FAIL" for o in outcomes))
        if not matrix_fail:
            state["abort"] = f"{type(exc).__name__}: {exc}"
            state["uncertain"] = type(exc).__name__ == "CleanupUncertain"
        raise
    finally:
        if evidence_dir is not None:
            directory = Path(evidence_dir)
            gate.write(directory / FINAL_NAME, build(request, outcomes, results, directory,
                                                     state["abort"], state["uncertain"]))
