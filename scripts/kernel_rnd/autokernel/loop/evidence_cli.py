#!/usr/bin/env python3
"""Offline inspection CLI for the scoped evidence projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from . import scoped_evidence as evidence
from . import status


QUERY_SCHEMA = "epyc.autokernel.evidence_query.v1"
OUTPUT_SCHEMA = "epyc.autokernel.evidence_inspection.v1"
QUERY_FIELDS = {"schema", "scope", "claim_key", "intended_use", "current_epoch",
                "limit", "projection_available"}


def _load(path: Path) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise evidence.EvidenceValidationError(
                    f"{path}: duplicate JSON object key {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise evidence.EvidenceValidationError(f"{path}: {exc}") from exc


def inspect_files(findings_path: Path, invalidations_path: Path,
                  query_path: Path) -> dict[str, Any]:
    finding_objects = _load(findings_path)
    invalidation_objects = _load(invalidations_path)
    query = _load(query_path)
    if not isinstance(finding_objects, list):
        raise evidence.EvidenceValidationError("findings JSON: expected array")
    if not isinstance(invalidation_objects, list):
        raise evidence.EvidenceValidationError("invalidations JSON: expected array")
    if not isinstance(query, Mapping):
        raise evidence.EvidenceValidationError("query JSON: expected object")
    missing, extra = QUERY_FIELDS - set(query), set(query) - QUERY_FIELDS
    if missing or extra:
        raise evidence.EvidenceValidationError(
            "query JSON fields: "
            + (f"missing {sorted(missing)} " if missing else "")
            + (f"unknown {sorted(extra)}" if extra else ""))
    if query["schema"] != QUERY_SCHEMA:
        raise evidence.EvidenceValidationError(
            f"query schema unsupported: {query['schema']!r}")
    if not isinstance(query["projection_available"], bool):
        raise evidence.EvidenceValidationError("query projection_available: expected boolean")
    findings = tuple(evidence.Finding.from_dict(row) for row in finding_objects)
    claim = evidence.ClaimKey.from_dict(query["claim_key"])
    index = evidence.EvidenceIndex(
        findings, invalidation_objects, current_epoch=query["current_epoch"],
        projection_available=query["projection_available"])
    result = index.retrieve(query["scope"], claim, query["intended_use"], query["limit"])
    return {"schema": OUTPUT_SCHEMA, "retrieval": result.to_dict(),
            "projection": index.to_dict(), "execution_authorized": False,
            "trusted_certificate_adapters_connected": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect supplied scoped evidence without executing or grading it")
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--invalidations", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        output = inspect_files(args.findings, args.invalidations, args.query)
        if args.out is not None:
            status.write_json(args.out.parent, args.out.name, output,
                              prefix=".scoped-evidence-")
    except (evidence.EvidenceValidationError, TypeError, ValueError) as exc:
        print(f"scoped-evidence validation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
