"""OAB-29: bind native-client wire evidence to the effective actor invocation.

This is an admission record, not a signature or a measurement grade. The supported
producer is ``python -m autokernel.loop.actor_wire_test --request ... --record ...``.
A missing client, skipped check, or unsupported adapter cannot produce a pass.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "epyc.autokernel.actor_wire.v1"
CHECKS = ("native_provider_reply", "large_reply_byte_exact", "large_prompt_on_wire",
          "role_options_on_wire", "readonly_denial_continues", "actor_timeout",
          "rc1_salvage", "signal_transient", "template_echo_rejected")
EVIDENCE_KINDS = {"native_provider": CHECKS[:6], "mocked_process_controls": CHECKS[6:]}
SOURCES = ("actor_wire_gate.py", "actor_wire_test.py", "test_actor_wire_gate.py",
           "actors.py", "actor_opencode_config.py", "actor_serving.py",
           "actor_passthrough.py", "actor_metrics.py", "procguard.py", "run.py",
           "test_promotion_targets.py")


class WireGateError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def suite_identity() -> dict:
    root = Path(__file__).resolve().parent
    return {name: file_digest(root / name) for name in SOURCES}


def describe(role, backend, seat, *, timeout_s, launch_config=None) -> dict:
    """Seat fields, CLI identity and launch context are all part of admission.

    No credentials or ambient provider configuration are serialized. Installed
    client bytes (not just a version label) must match when a record is consumed.
    """
    path = Path(backend.binary).resolve()
    return {"role": role, "backend": dataclasses.asdict(backend),
            "client_sha256": file_digest(path) if path.is_file() else None,
            "seat": dataclasses.asdict(seat), "timeout_s": timeout_s,
            "read_only": role == "critic", "launch_config": launch_config or {}}


def normalize(seats) -> list:
    unique = {digest(seat): seat for seat in seats}
    return [unique[key] for key in sorted(unique)]


def request(seats) -> dict:
    seats = normalize(seats)
    return {"schema": SCHEMA, "seats": seats, "actor_config_digest": digest(seats),
            "suite": suite_identity()}


def read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise WireGateError("wire record must be a regular JSON file under 1 MiB")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WireGateError("wire record must be an object")
    return value


def validate_request(row):
    seats = row.get("seats")
    if (row.get("schema") != SCHEMA or not isinstance(seats, list) or not seats
            or seats != normalize(seats) or row.get("actor_config_digest") != digest(seats)
            or row.get("suite") != suite_identity()):
        raise WireGateError("stale or malformed actor wire request")
    for seat in seats:
        backend = seat["backend"]
        if backend["kind"] not in ("codex", "claude", "opencode"):
            raise WireGateError(f"untested adapter {backend['kind']!r}; explicit override required")
        path = Path(backend["binary"]).resolve()
        if (not path.is_file() or not os.access(path, os.X_OK)
                or not seat.get("client_sha256")
                or file_digest(path) != seat["client_sha256"]):
            raise WireGateError(f"missing or changed installed actor CLI: {path}")


def passed(row, results) -> dict:
    validate_request(row)
    expected = {digest(seat) for seat in row["seats"]}
    if set(results) != expected:
        raise WireGateError("wire evidence does not cover all effective seats")
    for value in results.values():
        if value.get("checks") != {key: True for key in CHECKS}:
            raise WireGateError("incomplete wire checks cannot produce PASS")
        if value.get("evidence_kinds") != {key: list(items) for key, items in EVIDENCE_KINDS.items()}:
            raise WireGateError("native provider evidence and mocked controls must be distinguished")
        if not value.get("requests_sha256"):
            raise WireGateError("native provider requests were not captured")
    return {**row, "status": "PASS", "results": results}


def bind(seats, record=None, override=None):
    wanted = request(seats)
    if record is not None and override is not None:
        raise WireGateError("record and override reason are mutually exclusive")
    if override is not None:
        reason = override.strip()
        if not reason:
            raise WireGateError("untested-seat override requires a nonempty reason")
        return {**wanted, "status": "OVERRIDDEN", "override_reason": reason}
    if record is None:
        raise WireGateError("no passing actor wire record; run actor_wire_test with the "
                            "written request or supply --allow-untested-actor-seats REASON")
    validate_request(wanted)
    for key in ("schema", "actor_config_digest", "seats", "suite"):
        if record.get(key) != wanted[key]:
            raise WireGateError(f"actor wire record {key} mismatch")
    if record.get("status") != "PASS":
        raise WireGateError("actor wire record did not pass")
    passed(wanted, record.get("results", {}))
    return record


def write(path: Path, row):
    """Evidence writer; fsync before replacing the admission manifest."""
    from . import status
    status.write_json(path.parent, path.name, row, prefix=".actor-wire-")


def admit(seats, *, store, out, record_path=None, override=None):
    """Always write a runnable request; persist the receipt before any actor call."""
    wanted = request(seats)
    path = Path(store) / "actor-wire-request.json"
    write(path, wanted)
    try:
        receipt = bind(seats, read(record_path) if record_path else None, override)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise WireGateError(f"{exc}; request: {path}") from exc
    # This is an initial loop-run manifest, subsequently enriched by run.py's final
    # writer. An interrupted run still retains the override and its exact reason.
    target = Path(out) if out is not None else Path(store)
    write(target / "loop-run.json", {"schema": "epyc.autokernel.loop_run.v1",
                                    "state": "actor_admitted", "actor_wire_gate": receipt})
    return receipt
