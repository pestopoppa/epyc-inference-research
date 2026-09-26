#!/usr/bin/env python3
"""Validate and replay the deterministic CJ-15 local judge-cascade fixture.

This module uses only frozen JSON records and response strings. It never
constructs a model client or performs inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "typed_decisions"
    / "cj15_judge_cascade_v1"
)


class FixtureError(ValueError):
    """Raised when fixture bytes or conformance expectations do not match."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise FixtureError(f"{path.name}: final newline is required")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise FixtureError(f"{path.name}:{line_no}: blank rows are forbidden")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FixtureError(f"{path.name}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise FixtureError(f"{path.name}:{line_no}: row must be a JSON object")
        rows.append(row)
    return rows


def _parse_response(raw: str | None, candidate_ids: set[str]) -> dict[str, Any]:
    if raw is None:
        return {"status": "invalid", "choice_id": None, "confidence": None}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"status": "invalid", "choice_id": None, "confidence": None}
    if not isinstance(value, dict):
        return {"status": "invalid", "choice_id": None, "confidence": None}
    if value.get("status") == "abstain":
        return {"status": "abstain", "choice_id": None, "confidence": None}
    choice_id = value.get("choice_id")
    confidence = value.get("confidence")
    if (
        value.get("status") != "decision"
        or not isinstance(choice_id, str)
        or choice_id not in candidate_ids
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        return {"status": "invalid", "choice_id": None, "confidence": None}
    return {
        "status": "selected",
        "choice_id": choice_id,
        "confidence": float(confidence),
    }


def replay_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen primary/ fallback contract to one stress row."""
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FixtureError(f"{row.get('row_id')}: candidates must be a nonempty list")
    candidate_ids = {
        candidate.get("candidate_id")
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    if (
        len(candidate_ids) != len(candidates)
        or not all(isinstance(candidate_id, str) for candidate_id in candidate_ids)
    ):
        raise FixtureError(f"{row.get('row_id')}: candidate IDs must be unique strings")
    primary = _parse_response(row.get("primary_raw"), candidate_ids)
    fallback_reason: str | None = None
    if primary["status"] == "selected":
        chosen = primary
        via = "primary"
        fallback_status = None
    else:
        fallback_reason = primary["status"]
        fallback = _parse_response(row.get("fallback_raw"), candidate_ids)
        fallback_status = fallback["status"]
        chosen = fallback
        via = "fallback" if fallback["status"] == "selected" else "none"
    return {
        "status": "selected" if chosen["status"] == "selected" else "abstain",
        "selected_id": chosen["choice_id"],
        "via": via,
        "fallback_reason": fallback_reason,
        "primary_status": primary["status"],
        "fallback_status": fallback_status,
    }


def validate_fixture(root: Path = FIXTURE_DIR) -> dict[str, Any]:
    """Validate digests, frozen splits, source rows, and every expected output."""
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = _sha256(manifest_bytes)
    sidecar = (root / "manifest.sha256").read_text(encoding="ascii").strip()
    if sidecar != manifest_digest:
        raise FixtureError("manifest.sha256 does not match manifest.json")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "cj15.judge-cascade-fixture.v1":
        raise FixtureError("unsupported fixture schema_version")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise FixtureError("manifest.files must be an object")
    for name, expected_digest in files.items():
        actual = _sha256((root / name).read_bytes())
        if actual != expected_digest:
            raise FixtureError(f"digest mismatch for {name}")

    source_rows = _read_jsonl(root / "source_records.jsonl")
    stress_rows = _read_jsonl(root / "stress_rows.jsonl")
    source_ids = [row.get("row_id") for row in source_rows]
    stress_ids = [row.get("row_id") for row in stress_rows]
    if len(source_ids) != len(set(source_ids)) or len(stress_ids) != len(set(stress_ids)):
        raise FixtureError("row IDs must be unique within each file")
    split = manifest.get("split", {})
    if source_ids != split.get("calibration_reference_ids"):
        raise FixtureError("source rows differ from frozen calibration/reference IDs")
    if stress_ids != split.get("evaluation_ids"):
        raise FixtureError("stress rows differ from frozen evaluation IDs")
    expected_sources = manifest.get("source_expected_per_row", {})
    if set(expected_sources) != set(source_ids):
        raise FixtureError("manifest source_expected_per_row must cover every source row")
    for row in source_rows:
        row_id = row["row_id"]
        if row.get("observed_outputs") != expected_sources[row_id]:
            raise FixtureError(f"{row_id}: historical source outputs differ from manifest")
        criteria = row.get("criteria")
        labels = row.get("expected_criterion_verdicts")
        if (
            not isinstance(criteria, list)
            or not criteria
            or not isinstance(labels, list)
            or len(criteria) != len(labels)
        ):
            raise FixtureError(f"{row_id}: source criteria and labels must be nonempty and aligned")
        if not all(isinstance(label, bool) for label in labels):
            raise FixtureError(f"{row_id}: source criterion labels must be booleans")
        expected_overall = "pass" if all(labels) else "fail"
        if row.get("expected_overall") != expected_overall:
            raise FixtureError(f"{row_id}: source overall label disagrees with criterion labels")

    expected_by_id = manifest.get("expected_per_row", {})
    if set(expected_by_id) != set(stress_ids):
        raise FixtureError("manifest expected_per_row must cover every stress row")
    totals = {
        "rows": len(stress_rows),
        "correct": 0,
        "abstentions": 0,
        "invalid_primary": 0,
        "invalid_fallback": 0,
        "fallbacks": 0,
    }
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in stress_rows:
        row_id = row["row_id"]
        if row.get("split") != "evaluation":
            raise FixtureError(f"{row_id}: stress row is outside the evaluation split")
        pairs.setdefault(row.get("pair_id"), []).append(row)
        actual = replay_row(row)
        if actual != row.get("expected_output") or actual != expected_by_id[row_id]:
            raise FixtureError(f"{row_id}: replay output differs from frozen expectation")
        if actual["status"] == "abstain":
            totals["abstentions"] += 1
        elif actual["selected_id"] == row.get("expected_winner_id"):
            totals["correct"] += 1
        if actual["primary_status"] == "invalid":
            totals["invalid_primary"] += 1
        if actual["fallback_status"] == "invalid":
            totals["invalid_fallback"] += 1
        if actual["fallback_status"] is not None:
            totals["fallbacks"] += 1
    for pair_id, pair_rows in pairs.items():
        if len(pair_rows) != 2 or {row.get("order") for row in pair_rows} != {"forward", "reverse"}:
            raise FixtureError(f"{pair_id}: each scenario must include both option orders")
        if len({row.get("expected_winner_id") for row in pair_rows}) != 1:
            raise FixtureError(f"{pair_id}: gold winner must be invariant to option order")
        candidate_views = [
            {
                candidate["candidate_id"]: (
                    candidate.get("text"),
                    candidate.get("display_name"),
                )
                for candidate in row["candidates"]
            }
            for row in pair_rows
        ]
        if candidate_views[0] != candidate_views[1]:
            # The deliberate name-swap pair must preserve content while
            # exchanging only the displayed A/B names.
            first = {key: value[0] for key, value in candidate_views[0].items()}
            second = {key: value[0] for key, value in candidate_views[1].items()}
            names_1 = {key: value[1] for key, value in candidate_views[0].items()}
            names_2 = {key: value[1] for key, value in candidate_views[1].items()}
            is_name_swap = any(row.get("rubric_name_swapped") for row in pair_rows)
            if not is_name_swap or first != second or names_1 == names_2:
                raise FixtureError(f"{pair_id}: candidate content changed across option orders")
    # The denominator is every frozen row, including malformed and abstaining
    # decisions; it is never reduced to only resolved selections.
    totals["denominator"] = len(stress_rows)
    totals["accuracy_over_all_rows"] = (
        totals["correct"] / len(stress_rows) if stress_rows else 0.0
    )
    return {
        "fixture": manifest["fixture_id"],
        "manifest_sha256": manifest_digest,
        "source_rows": len(source_rows),
        "stress": totals,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args()
    try:
        result = validate_fixture(args.fixture_dir)
    except (FixtureError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"CJ-15 fixture invalid: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
