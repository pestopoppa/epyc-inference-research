#!/usr/bin/env python3
"""Create/apply a GLM C-CRAB accept-control review sheet without inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import glm52_ccrab_accept_control_signoff as signoff_mod


CSV_FIELDS = [
    "row_id",
    "instance_id",
    "repo",
    "pull_number",
    "candidate_chars",
    "candidate_redacted_long_digit_runs",
    "machine_recommendation",
    "machine_reason",
    "format_concerns",
    "decision",
    "reviewer",
    "reviewed_at",
    "notes",
]

UNREVIEWED_DECISIONS = {"", "unreviewed"}


class ReviewSheetError(ValueError):
    """Raised when review-sheet input cannot be safely applied."""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewSheetError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReviewSheetError(f"{path}: expected a JSON object")
    return data


def _packet_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return signoff_mod.validate_packet(packet)
    except signoff_mod.ValidationError as exc:
        raise ReviewSheetError(str(exc)) from exc


def _row_ids(rows: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ReviewSheetError("all packet rows must have a non-empty row_id")
        ids.append(row_id)
    duplicates = signoff_mod.duplicate_values(ids)
    if duplicates:
        raise ReviewSheetError(f"duplicate row ids in packet: {', '.join(duplicates)}")
    return ids


def _load_recommendations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = _read_json(path)
    recommendations = data.get("recommendations")
    if not isinstance(recommendations, list):
        raise ReviewSheetError("machine recommendations must contain a recommendations list")
    by_id: dict[str, dict[str, Any]] = {}
    for item in recommendations:
        if not isinstance(item, dict):
            raise ReviewSheetError("machine recommendation rows must be objects")
        row_id = item.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise ReviewSheetError("machine recommendation rows must have row_id")
        if row_id in by_id:
            raise ReviewSheetError(f"duplicate recommendation for row_id {row_id}")
        by_id[row_id] = item
    return by_id


def _format_concerns(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


def build_review_rows(
    packet: dict[str, Any],
    *,
    machine_recommendations: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    rows = _packet_rows(packet)
    review_rows: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row["row_id"])
        recommendation = machine_recommendations.get(row_id, {})
        review_rows.append(
            {
                "row_id": row_id,
                "instance_id": str(row.get("instance_id") or recommendation.get("instance_id") or ""),
                "repo": str(row.get("repo") or recommendation.get("repo") or ""),
                "pull_number": str(row.get("pull_number") or recommendation.get("pull_number") or ""),
                "candidate_chars": str(row.get("candidate_chars") or ""),
                "candidate_redacted_long_digit_runs": str(
                    bool(row.get("candidate_redacted_long_digit_runs"))
                ).lower(),
                "machine_recommendation": str(recommendation.get("recommendation") or ""),
                "machine_reason": str(recommendation.get("reason") or ""),
                "format_concerns": _format_concerns(recommendation.get("format_concerns")),
                "decision": "",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            }
        )
    return review_rows


def write_review_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"row_id", "decision", "reviewer", "reviewed_at", "notes"} - set(reader.fieldnames or [])
        if missing:
            raise ReviewSheetError(f"review CSV missing columns: {', '.join(sorted(missing))}")
        rows = [{key: (value or "") for key, value in row.items()} for row in reader]
    if not rows:
        raise ReviewSheetError("review CSV has no rows")
    return rows


def _normalized_decision(value: str, *, row_id: str) -> str | None:
    decision = value.strip()
    if decision in UNREVIEWED_DECISIONS:
        return None
    if decision not in signoff_mod.VALID_DECISIONS:
        raise ReviewSheetError(
            f"row {row_id}: decision must be hard_accept, reject_or_ambiguous, or blank"
        )
    return decision


def apply_review_csv(
    packet: dict[str, Any],
    review_rows: list[dict[str, str]],
    *,
    default_reviewer: str | None,
    default_reviewed_at: str | None,
) -> dict[str, Any]:
    packet_copy = deepcopy(packet)
    packet_rows = _packet_rows(packet_copy)
    expected_ids = _row_ids(packet_rows)
    csv_ids = [row.get("row_id", "").strip() for row in review_rows]
    duplicates = signoff_mod.duplicate_values(csv_ids)
    if duplicates:
        raise ReviewSheetError(f"duplicate row ids in review CSV: {', '.join(duplicates)}")
    if csv_ids != expected_ids:
        missing = [row_id for row_id in expected_ids if row_id not in set(csv_ids)]
        unexpected = [row_id for row_id in csv_ids if row_id not in set(expected_ids)]
        raise ReviewSheetError(
            "review CSV row ids must exactly match packet order"
            f"; missing={missing}; unexpected={unexpected}"
        )

    now = default_reviewed_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    for packet_row, review_row in zip(packet_rows, review_rows, strict=True):
        row_id = str(packet_row["row_id"])
        decision = _normalized_decision(review_row.get("decision", ""), row_id=row_id)
        if decision is None:
            packet_row["signoff"] = {
                "status": signoff_mod.UNREVIEWED_STATUS,
                "reviewer": None,
                "reviewed_at": None,
                "decision": None,
                "notes": review_row.get("notes") or None,
            }
            continue
        reviewer = (review_row.get("reviewer") or default_reviewer or "").strip()
        reviewed_at = (review_row.get("reviewed_at") or now).strip()
        notes = (review_row.get("notes") or "").strip()
        if not reviewer:
            raise ReviewSheetError(f"row {row_id}: reviewer is required for reviewed decisions")
        if not reviewed_at:
            raise ReviewSheetError(f"row {row_id}: reviewed_at is required for reviewed decisions")
        if not notes:
            raise ReviewSheetError(f"row {row_id}: notes are required for reviewed decisions")
        packet_row["signoff"] = {
            "status": signoff_mod.REVIEWED_STATUS,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "decision": decision,
            "notes": notes,
        }
    return packet_copy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_packet", type=Path)
    parser.add_argument("--machine-recommendations", type=Path)
    parser.add_argument("--review-csv-out", type=Path)
    parser.add_argument("--apply-review-csv", type=Path)
    parser.add_argument("--signed-packet-out", type=Path)
    parser.add_argument("--default-reviewer")
    parser.add_argument("--default-reviewed-at")
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        packet = _read_json(args.audit_packet)
        _packet_rows(packet)
        summary: dict[str, Any] = {
            "schema": "glm52_ccrab_accept_control_review_sheet.v1",
            "audit_packet": str(args.audit_packet),
            "review_csv_written": None,
            "signed_packet_written": None,
            "note": (
                "This helper emits/applies explicit review decisions only; "
                "machine recommendations are advisory and are never converted to signoff decisions."
            ),
        }
        if args.review_csv_out:
            recommendation_map = _load_recommendations(args.machine_recommendations)
            review_rows = build_review_rows(packet, machine_recommendations=recommendation_map)
            write_review_csv(review_rows, args.review_csv_out)
            summary["review_csv_written"] = str(args.review_csv_out)
            summary["review_row_n"] = len(review_rows)
        if args.apply_review_csv or args.signed_packet_out:
            if not args.apply_review_csv or not args.signed_packet_out:
                raise ReviewSheetError("--apply-review-csv and --signed-packet-out must be provided together")
            review_rows = _load_review_csv(args.apply_review_csv)
            signed_packet = apply_review_csv(
                packet,
                review_rows,
                default_reviewer=args.default_reviewer,
                default_reviewed_at=args.default_reviewed_at,
            )
            report = signoff_mod.build_report(
                signed_packet,
                min_hard_accepts=24,
                allow_unreviewed=False,
            )
            _write_text(args.signed_packet_out, json.dumps(signed_packet, indent=2, sort_keys=True) + "\n")
            summary["signed_packet_written"] = str(args.signed_packet_out)
            summary["signed_packet_report"] = report
        if args.summary_out:
            _write_text(args.summary_out, json.dumps(summary, indent=2, sort_keys=True) + "\n")
        elif not args.review_csv_out and not args.signed_packet_out:
            print(json.dumps(summary, indent=2, sort_keys=True))
    except (ReviewSheetError, signoff_mod.ValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
