#!/usr/bin/env python3
"""Validate the immutable powered-SWE candidate manifest against a gold report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_unique_string_list(report: dict, key: str) -> list[str]:
    values = report.get(key)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"gold report {key} must be a list of strings")
    if len(values) != len(set(values)):
        raise ValueError(f"gold report {key} contains duplicate IDs")
    return values


def require_count(report: dict, key: str, expected: int) -> None:
    value = report.get(key)
    if value != expected:
        raise ValueError(f"gold report {key}={value!r}, expected {expected}")


def validate(manifest: dict, report: dict) -> tuple[list[str], dict]:
    candidates = manifest.get("candidate_ids")
    target = manifest.get("gold_validated_target_count")
    if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
        raise ValueError("manifest candidate_ids must be a list of strings")
    if len(candidates) != manifest.get("candidate_count") or len(set(candidates)) != len(candidates):
        raise ValueError("manifest candidate count or uniqueness contract failed")
    if not isinstance(target, int) or target < 1 or target > len(candidates):
        raise ValueError("manifest gold_validated_target_count is invalid")

    if report.get("schema_version") != 2:
        raise ValueError("gold report must use schema_version=2")
    submitted = require_unique_string_list(report, "submitted_ids")
    completed = require_unique_string_list(report, "completed_ids")
    resolved = require_unique_string_list(report, "resolved_ids")
    unresolved = require_unique_string_list(report, "unresolved_ids")
    empty = require_unique_string_list(report, "empty_patch_ids")
    incomplete = require_unique_string_list(report, "incomplete_ids")
    errors = require_unique_string_list(report, "error_ids")
    candidate_set = set(candidates)
    if set(submitted) != candidate_set:
        missing = sorted(candidate_set - set(submitted))
        unexpected = sorted(set(submitted) - candidate_set)
        raise ValueError(f"gold report submitted_ids must exactly match candidates; missing={missing[:3]} unexpected={unexpected[:3]}")
    if incomplete or errors:
        raise ValueError("gold report must be terminal with no incomplete or error IDs")
    if set(resolved) & set(unresolved) or set(completed) != set(resolved) | set(unresolved):
        raise ValueError("gold report completed/resolved/unresolved partition is inconsistent")
    if (set(completed) & set(empty)) or set(completed) | set(empty) != candidate_set:
        raise ValueError("gold report completed and empty-patch IDs must partition submitted candidates")
    for key, values in (
        ("total_instances", candidates),
        ("submitted_instances", submitted),
        ("completed_instances", completed),
        ("resolved_instances", resolved),
        ("unresolved_instances", unresolved),
        ("empty_patch_instances", empty),
        ("error_instances", errors),
    ):
        require_count(report, key, len(values))

    resolved_set = set(resolved)
    if not resolved_set <= candidate_set:
        unexpected = sorted(resolved_set - candidate_set)
        raise ValueError(f"gold report contains non-candidate resolved IDs: {unexpected[:3]}")

    accepted = [item for item in candidates if item in resolved_set][:target]
    if len(accepted) < target:
        raise ValueError(f"only {len(accepted)} of required {target} candidate IDs passed gold validation")

    summary = {
        "schema": "powered_gold_acceptance.v1",
        "accepted_count": len(accepted),
        "candidate_count": len(candidates),
        "target_count": target,
        "resolved_candidate_count": len(resolved_set),
        "accepted_ids_sha256": hashlib.sha256(("\n".join(accepted) + "\n").encode()).hexdigest(),
        "report_counts": {
            key: report.get(key)
            for key in ("total_instances", "submitted_instances", "completed_instances", "resolved_instances", "unresolved_instances", "empty_patch_instances", "error_instances")
        },
    }
    return accepted, summary


def reject_inconsistent_existing(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"refusing to overwrite inconsistent existing output: {path}")


def write_atomic_or_verify(path: Path, content: str) -> None:
    reject_inconsistent_existing(path, content)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def preflight_outputs(outputs: tuple[tuple[Path, str], ...]) -> None:
    for path, content in outputs:
        reject_inconsistent_existing(path, content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold-report", type=Path, required=True)
    parser.add_argument("--accepted-ids-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    report = load_json(args.gold_report)
    accepted, summary = validate(manifest, report)
    summary["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    summary["gold_report_sha256"] = hashlib.sha256(args.gold_report.read_bytes()).hexdigest()
    accepted_content = "\n".join(accepted) + "\n"
    summary_content = json.dumps(summary, indent=2) + "\n"
    # Validate both destinations before either independent atomic replacement.
    outputs = ((args.accepted_ids_out, accepted_content), (args.summary_out, summary_content))
    preflight_outputs(outputs)
    write_atomic_or_verify(args.accepted_ids_out, accepted_content)
    write_atomic_or_verify(args.summary_out, summary_content)


if __name__ == "__main__":
    main()
