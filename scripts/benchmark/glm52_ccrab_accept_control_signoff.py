#!/usr/bin/env python3
"""Summarize manual GLM C-CRAB accept-control signoff without inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


AUDIT_PACKET_SCHEMA = "glm52_ccrab_accept_control_audit_packet.v1"
REPORT_SCHEMA = "glm52_ccrab_accept_control_signoff.v1"
REVIEWED_STATUS = "reviewed"
UNREVIEWED_STATUS = "unreviewed"
HARD_ACCEPT_DECISION = "hard_accept"
REJECT_OR_AMBIGUOUS_DECISION = "reject_or_ambiguous"
VALID_DECISIONS = {HARD_ACCEPT_DECISION, REJECT_OR_AMBIGUOUS_DECISION}
VALID_STATUSES = {REVIEWED_STATUS, UNREVIEWED_STATUS}


class ValidationError(ValueError):
    """Raised when an audit packet cannot be safely summarized."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("audit packet must be a JSON object")
    return data


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_row_ids_file(path: Path) -> list[str]:
    row_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        row_id = stripped.split("#", 1)[0].strip()
        if row_id:
            row_ids.append(row_id)
    return row_ids


def duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _require_string(value: Any, *, field: str, row_id: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        prefix = f"row {row_id}: " if row_id else ""
        raise ValidationError(f"{prefix}{field} must be a non-empty string")
    return value


def _optional_note(value: Any, *, row_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"row {row_id}: signoff.notes must be a string or null")
    return value


def _require_note(value: Any, *, row_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"row {row_id}: signoff.notes must be a non-empty string")
    return value.strip()


def validate_packet(packet: dict[str, Any]) -> list[dict[str, Any]]:
    if packet.get("schema") != AUDIT_PACKET_SCHEMA:
        raise ValidationError(f"schema must be {AUDIT_PACKET_SCHEMA}")
    rows = packet.get("rows")
    if not isinstance(rows, list):
        raise ValidationError("rows must be a list")
    packet_selected_n = packet.get("selected_n")
    if packet_selected_n is not None and packet_selected_n != len(rows):
        raise ValidationError("selected_n must match len(rows) when present")
    return rows


def validate_row(row: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(row, dict):
        raise ValidationError("each row must be a JSON object")
    row_id = _require_string(row.get("row_id"), field="row_id")
    signoff = row.get("signoff")
    if not isinstance(signoff, dict):
        raise ValidationError(f"row {row_id}: signoff must be an object")

    missing = {"status", "reviewer", "reviewed_at", "decision", "notes"} - set(signoff)
    if missing:
        raise ValidationError(f"row {row_id}: signoff missing fields: {', '.join(sorted(missing))}")

    status = signoff.get("status")
    if status not in VALID_STATUSES:
        raise ValidationError(f"row {row_id}: unknown signoff.status {status!r}")

    decision = signoff.get("decision")
    if status == UNREVIEWED_STATUS:
        if decision is not None:
            raise ValidationError(f"row {row_id}: unreviewed rows must have null decision")
        _optional_note(signoff.get("notes"), row_id=row_id)
        return row_id, signoff

    if decision not in VALID_DECISIONS:
        raise ValidationError(f"row {row_id}: unknown signoff.decision {decision!r}")
    _require_string(signoff.get("reviewer"), field="signoff.reviewer", row_id=row_id)
    _require_string(signoff.get("reviewed_at"), field="signoff.reviewed_at", row_id=row_id)
    _require_note(signoff.get("notes"), row_id=row_id)
    return row_id, signoff


def build_report(
    packet: dict[str, Any],
    *,
    min_hard_accepts: int,
    allow_unreviewed: bool,
    expected_row_ids: list[str] | None = None,
) -> dict[str, Any]:
    if min_hard_accepts < 0:
        raise ValidationError("min_hard_accepts must be >= 0")
    if expected_row_ids is not None and duplicate_values(expected_row_ids):
        raise ValidationError("expected row ids file contains duplicate row ids")

    rows = validate_packet(packet)
    accepted_row_ids: list[str] = []
    rejected_row_ids: list[str] = []
    unreviewed_row_ids: list[str] = []
    oracle_notes: dict[str, dict[str, Any]] = {}

    for row in rows:
        row_id, signoff = validate_row(row)
        if row_id in accepted_row_ids or row_id in rejected_row_ids or row_id in unreviewed_row_ids:
            raise ValidationError(f"duplicate row_id in packet: {row_id}")
        status = signoff["status"]
        decision = signoff["decision"]
        if status == UNREVIEWED_STATUS:
            unreviewed_row_ids.append(row_id)
        elif decision == HARD_ACCEPT_DECISION:
            accepted_row_ids.append(row_id)
            oracle_notes[row_id] = {
                "notes": signoff.get("notes"),
                "reviewer": signoff.get("reviewer"),
                "reviewed_at": signoff.get("reviewed_at"),
            }
        elif decision == REJECT_OR_AMBIGUOUS_DECISION:
            rejected_row_ids.append(row_id)
        else:
            raise AssertionError(f"validated unexpected decision: {decision!r}")

    hard_accept_n = len(accepted_row_ids)
    rejected_or_ambiguous_n = len(rejected_row_ids)
    unreviewed_n = len(unreviewed_row_ids)
    if expected_row_ids is None:
        accepted_row_ids_match_expected = None
        missing_expected_accepted_row_ids: list[str] = []
        unexpected_accepted_row_ids: list[str] = []
    else:
        accepted_row_ids_match_expected = accepted_row_ids == expected_row_ids
        expected_set = set(expected_row_ids)
        accepted_set = set(accepted_row_ids)
        missing_expected_accepted_row_ids = [row_id for row_id in expected_row_ids if row_id not in accepted_set]
        unexpected_accepted_row_ids = [row_id for row_id in accepted_row_ids if row_id not in expected_set]
    expected_gate = accepted_row_ids_match_expected is not False
    decision_grade = (
        hard_accept_n >= min_hard_accepts
        and rejected_or_ambiguous_n == 0
        and (allow_unreviewed or unreviewed_n == 0)
        and expected_gate
    )
    return {
        "schema": REPORT_SCHEMA,
        "audit_packet_schema": AUDIT_PACKET_SCHEMA,
        "selected_n": len(rows),
        "min_hard_accepts": min_hard_accepts,
        "allow_unreviewed": allow_unreviewed,
        "hard_accept_n": hard_accept_n,
        "rejected_or_ambiguous_n": rejected_or_ambiguous_n,
        "unreviewed_n": unreviewed_n,
        "decision_grade": decision_grade,
        "expected_row_ids_n": len(expected_row_ids) if expected_row_ids is not None else None,
        "accepted_row_ids_match_expected": accepted_row_ids_match_expected,
        "missing_expected_accepted_row_ids": missing_expected_accepted_row_ids,
        "unexpected_accepted_row_ids": unexpected_accepted_row_ids,
        "accepted_row_ids": accepted_row_ids,
        "rejected_row_ids": rejected_row_ids,
        "unreviewed_row_ids": unreviewed_row_ids,
        "oracle_notes": oracle_notes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_packet", type=Path)
    parser.add_argument("--min-hard-accepts", type=int, default=24)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument(
        "--expected-row-ids",
        type=Path,
        help="Optional row-id file that accepted hard controls must match exactly for decision_grade=true.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--row-ids-out", type=Path)
    parser.add_argument("--oracle-notes-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        packet = read_json(args.audit_packet)
        expected_row_ids = read_row_ids_file(args.expected_row_ids) if args.expected_row_ids else None
        report = build_report(
            packet,
            min_hard_accepts=args.min_hard_accepts,
            allow_unreviewed=args.allow_unreviewed,
            expected_row_ids=expected_row_ids,
        )
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        write_text(args.json_out, report_text)
    else:
        print(report_text, end="")
    if args.row_ids_out:
        accepted_text = "\n".join(report["accepted_row_ids"])
        write_text(args.row_ids_out, f"{accepted_text}\n" if accepted_text else "")
    if args.oracle_notes_out:
        write_text(args.oracle_notes_out, json.dumps(report["oracle_notes"], indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
