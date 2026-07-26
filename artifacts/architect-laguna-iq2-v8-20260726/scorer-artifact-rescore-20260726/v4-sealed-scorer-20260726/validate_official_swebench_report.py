#!/usr/bin/env python3
"""Validate a sealed selected-instance SWE-bench report without scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL_ID = "laguna-swe-v4-selected-instance-postscore.v1"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique_ids(report: dict, key: str) -> list[str]:
    value = report.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of string IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{key} contains duplicate IDs")
    return value


def _require_count(report: dict, key: str, values: list[str]) -> None:
    count = report.get(key)
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"{key} must be an integer")
    if count != len(values):
        raise ValueError(f"{key} does not match its ID list")


def validate_selected_report(
    report: dict,
    requested_ids: list[str],
    expected_empty_ids: list[str],
) -> dict:
    if report.get("schema_version") != 2:
        raise ValueError("official report schema_version must be 2")
    expected = set(requested_ids)
    if len(requested_ids) != 40 or len(expected) != 40:
        raise ValueError("sealed package must carry exactly 40 unique requested IDs")
    if len(expected_empty_ids) != len(set(expected_empty_ids)):
        raise ValueError("sealed predictions contain duplicate empty-patch IDs")
    expected_empty = set(expected_empty_ids)
    if not expected_empty <= expected:
        raise ValueError("sealed empty-patch IDs escape the requested denominator")

    submitted = _unique_ids(report, "submitted_ids")
    completed = _unique_ids(report, "completed_ids")
    empty = _unique_ids(report, "empty_patch_ids")
    resolved = _unique_ids(report, "resolved_ids")
    unresolved = _unique_ids(report, "unresolved_ids")
    errors = _unique_ids(report, "error_ids")
    incomplete = _unique_ids(report, "incomplete_ids")
    numeric_pairs = {
        "submitted_instances": submitted,
        "completed_instances": completed,
        "empty_patch_instances": empty,
        "resolved_instances": resolved,
        "unresolved_instances": unresolved,
        "error_instances": errors,
    }
    for field, values in numeric_pairs.items():
        _require_count(report, field, values)
    if "incomplete_instances" in report:
        _require_count(report, "incomplete_instances", incomplete)

    submitted_set = set(submitted)
    completed_set = set(completed)
    empty_set = set(empty)
    resolved_set = set(resolved)
    unresolved_set = set(unresolved)
    incomplete_set = set(incomplete)
    if report["submitted_instances"] != 40 or submitted_set != expected:
        raise ValueError("submitted report IDs do not exactly bind the selected 40")
    if report["error_instances"] != 0 or errors:
        raise ValueError("official harness reported selected-instance errors")
    if completed_set & empty_set or completed_set | empty_set != submitted_set:
        raise ValueError("completed and empty-patch IDs do not partition submitted IDs")
    if empty_set != expected_empty:
        raise ValueError("reported empty-patch IDs do not match sealed predictions")
    if resolved_set & unresolved_set or resolved_set | unresolved_set != completed_set:
        raise ValueError("resolved and unresolved IDs do not partition completed IDs")
    if incomplete_set & submitted_set:
        raise ValueError("incomplete IDs include selected submitted instances")

    total = report.get("total_instances")
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("total_instances must be an integer")
    if total != len(submitted_set) + len(incomplete_set):
        raise ValueError("total_instances does not equal submitted plus unsubmitted incomplete IDs")
    return {
        "status": "VALID",
        "protocol_id": PROTOCOL_ID,
        "denominator": 40,
        "completed_instances": len(completed),
        "empty_patch_instances": len(empty),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "error_instances": 0,
        "unsubmitted_incomplete_instances": len(incomplete),
        "percent_resolved_over_40": 100.0 * len(resolved) / 40,
    }


def load_package_contract(package: Path) -> tuple[dict, list[str], dict]:
    manifest_path = package / "sealed_package_manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != "epyc.laguna-swe-v4-sealed-package.v1":
        raise ValueError("unexpected sealed package manifest schema")
    if manifest.get("status") != "READY_FOR_OFFICIAL_SCORING":
        raise ValueError("sealed package is not ready for official scoring")
    validator_path = Path(__file__).resolve()
    validator_sha = sha256_bytes(validator_path.read_bytes())
    if validator_sha != manifest.get("postscore_validator_sha256"):
        raise ValueError("post-score validator SHA does not match package manifest")
    predictions_path = package / "predictions_v4.json"
    predictions_raw = predictions_path.read_bytes()
    predictions_sha = sha256_bytes(predictions_raw)
    if predictions_sha != manifest.get("predictions_sha256"):
        raise ValueError("sealed predictions SHA does not match package manifest")
    predictions = json.loads(predictions_raw)
    if not isinstance(predictions, list) or len(predictions) != 40:
        raise ValueError("sealed predictions must contain exactly 40 rows")
    prediction_ids = [row.get("instance_id") for row in predictions]
    if any(not isinstance(item, str) for item in prediction_ids):
        raise ValueError("sealed prediction instance IDs must be strings")
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("sealed predictions contain duplicate instance IDs")
    requested = manifest.get("requested_ids")
    if prediction_ids != requested:
        raise ValueError("sealed prediction order does not match package requested IDs")
    empty = [
        row["instance_id"]
        for row in predictions
        if row.get("model_patch") in ("", None)
    ]

    argv_path = package / "official_swebench_argv.json"
    argv_raw = argv_path.read_bytes()
    if sha256_bytes(argv_raw) != manifest.get("official_argv_sha256"):
        raise ValueError("official scorer argv SHA does not match package manifest")
    argv_plan = json.loads(argv_raw)
    if argv_plan.get("requested_ids") != requested:
        raise ValueError("official scorer argv requested IDs drifted from package manifest")
    if argv_plan.get("predictions_sha256") != predictions_sha:
        raise ValueError("official scorer argv predictions SHA drifted from package manifest")
    bindings = {
        "protocol_id": PROTOCOL_ID,
        "validator_path": str(validator_path),
        "validator_sha256": validator_sha,
        "package_manifest_path": str(manifest_path),
        "package_manifest_sha256": sha256_bytes(manifest_raw),
        "capture_pq_sha256": manifest.get("capture_pq_sha256"),
        "converter_sha256": manifest.get("converter_sha256"),
        "diagnostics_summary_sha256": manifest.get("diagnostics_summary_sha256"),
        "official_argv_sha256": manifest.get("official_argv_sha256"),
        "predictions_path": str(predictions_path),
        "predictions_sha256": predictions_sha,
    }
    return manifest, empty, bindings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    manifest, expected_empty, bindings = load_package_contract(args.package)
    report_raw = args.report.read_bytes()
    report_sha = sha256_bytes(report_raw)
    if args.expected_report_sha256 and report_sha != args.expected_report_sha256:
        raise ValueError("official report SHA drifted from expected binding")
    result = validate_selected_report(
        json.loads(report_raw),
        manifest["requested_ids"],
        expected_empty,
    )
    result["bindings"] = {
        **bindings,
        "report_path": str(args.report),
        "report_sha256": report_sha,
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
