"""Offline-only validator and summary for candidate manifests and batches."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import candidate_manifest as cm
from . import status


SUMMARY_SCHEMA = "epyc.autokernel.candidate_dry_summary.v1"


def _read(path: Path, parser, label: str):
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise cm.CandidateError(f"cannot load {label} {path}: {exc}") from exc
    return parser(value)


def build_summary(manifest: cm.CandidateManifest, *, row_set: cm.RequiredRowSet | None = None,
                  batch: cm.ValidationBatch | None = None,
                  state: cm.CandidateState | None = None) -> dict[str, Any]:
    manifest = manifest.validated()
    prerequisites: list[str] = []
    row_counts: Counter[str] = Counter()
    if row_set is not None:
        row_set = cm.RequiredRowSet.from_dict(row_set.to_dict())
    if batch is not None:
        batch = batch.validated()
        row_counts.update(item.status for item in batch.rows)
        if batch.candidate_manifest_digest != manifest.manifest_digest:
            prerequisites.append("batch_candidate_manifest_mismatch")
        if row_set is None:
            prerequisites.append("required_row_set_missing")
        elif batch.row_set_digest != row_set.row_set_digest:
            prerequisites.append("batch_row_set_mismatch")
        required_ids = ({item.row_id for item in row_set.rows if item.required}
                        if row_set else set())
        state_by_id = {item.row_id: item for item in batch.rows}
        if any(state_by_id.get(row_id) is None
               or state_by_id[row_id].status != "passed" for row_id in required_ids):
            prerequisites.append("required_rows_incomplete")
        # JSON status is historical data, never the trusted validator callback.
        prerequisites.append("trusted_registered_verifier_not_connected")
    if state is not None:
        state = state.validated()
        if state.production_ref_digest != manifest.production_ref_digest:
            prerequisites.append("production_ref_mismatch")
    return {
        "schema": SUMMARY_SCHEMA,
        "mode": "offline_validation_only",
        "execution_authorized": False,
        "production_promotion_authorized": False,
        "validated_pointer_eligible": False,
        "manifest": {"manifest_id": manifest.manifest_id,
                     "manifest_digest": manifest.manifest_digest,
                     "parent_manifest_digest": manifest.parent_manifest_digest,
                     "production_ref_digest": manifest.production_ref_digest,
                     "build_execution_digests": [item.execution_digest
                                                 for item in manifest.builds],
                     "keeps": len(manifest.keeps), "targets": len(manifest.targets)},
        "row_set_digest": row_set.row_set_digest if row_set else None,
        "batch": (None if batch is None else {
            "batch_id": batch.batch_id,
            "candidate_manifest_digest": batch.candidate_manifest_digest,
            "comparator_manifest_digest": batch.comparator_manifest_digest,
            "row_set_digest": batch.row_set_digest,
            "row_dispositions": dict(sorted(row_counts.items()))}),
        "state": (None if state is None else {
            "integration_tip": state.integration_tip,
            "validated_candidate": state.validated_candidate,
            "keeps_since_gate": state.keeps_since_gate,
            "threshold_generation": state.threshold_generation,
            "covered_threshold_generation": state.covered_threshold_generation,
            "gate_due": state.gate_due,
            "validation_debt": list(state.validation_debt)}),
        "prerequisites": sorted(set(prerequisites)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate candidate-manifest records offline; never execute or promote")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--row-set", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = _read(args.manifest, cm.CandidateManifest.from_dict, "candidate manifest")
        row_set = (_read(args.row_set, cm.RequiredRowSet.from_dict, "row set")
                   if args.row_set else None)
        batch = (_read(args.batch, cm.ValidationBatch.from_dict, "validation batch")
                 if args.batch else None)
        state = (_read(args.state, cm.CandidateState.from_dict, "candidate state")
                 if args.state else None)
        body = build_summary(manifest, row_set=row_set, batch=batch, state=state)
        if args.out:
            status.write_json(args.out.parent, args.out.name, body, prefix=".candidate-")
        else:
            json.dump(body, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
    except (cm.CandidateError, OSError) as exc:
        print(f"candidate dry validation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SUMMARY_SCHEMA", "build_summary", "main"]
