"""Offline-only validator and summary for candidate manifests and batches."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import candidate_manifest as cm
from . import candidate_transactions as ct
from .campaign import ResolvedCampaign
from .campaign_control import CampaignController, ControlRefused
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


def _mutation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persist an explicit candidate transaction; never execute or promote")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--resolved-campaign", required=True, type=Path)
        child.add_argument("--store", required=True, type=Path)
        child.add_argument("--config-generation", type=int, default=1)
        child.add_argument("--request-id", required=True)
        child.add_argument("--repo", action="append", default=[], metavar="ID=PATH",
                           help="exact repository root for every manifest source")
        return child

    init = command("init")
    init.add_argument("--manifest", required=True, type=Path)
    init.add_argument("--state", required=True, type=Path)
    integrate = command("integrate")
    integrate.add_argument("--previous", required=True, type=Path)
    integrate.add_argument("--manifest", required=True, type=Path)
    start = command("start-batch")
    start.add_argument("--batch", required=True, type=Path)
    start.add_argument("--row-set", required=True, type=Path)
    start.add_argument("--manifest", required=True, type=Path)
    start.add_argument("--comparator", required=True, type=Path)
    row = command("record-row")
    row.add_argument("--batch-id", required=True)
    row.add_argument("--row-set", required=True, type=Path)
    row.add_argument("--row-state", required=True, type=Path)
    complete = command("complete")
    complete.add_argument("--batch", required=True, type=Path)
    return parser


def _resolved(path: Path) -> ResolvedCampaign:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ct.CandidateTransactionError(
            f"cannot load resolved campaign {path}: {exc}") from exc
    if isinstance(value, dict) and "resolved_campaign" in value:
        value = value["resolved_campaign"]
    try:
        return ResolvedCampaign.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ct.CandidateTransactionError(f"invalid resolved campaign: {exc}") from exc


def _repos(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        repo_id, separator, path = value.partition("=")
        if not separator or not repo_id or not path or repo_id in result:
            raise ct.CandidateTransactionError(
                "--repo must be unique non-empty ID=PATH")
        result[repo_id] = Path(path)
    return result


def _mutate(argv: Sequence[str]) -> int:
    args = _mutation_parser().parse_args(argv)
    resolved = _resolved(args.resolved_campaign)
    if args.config_generation < 1:
        raise ct.CandidateTransactionError("config generation must be positive")
    repos = _repos(args.repo)
    backend = ct.GitCandidateBackend(repos, campaign_id=resolved.campaign_id)

    # Parse all caller-controlled records, and verify source/repository bindings,
    # before acquiring a store whose controller entry appends a START event.
    if args.operation == "init":
        operation_values = {
            "state": _read(args.state, cm.CandidateState.from_dict, "candidate state"),
            "manifest": _read(args.manifest, cm.CandidateManifest.from_dict,
                              "candidate manifest"),
        }
        backend.plan(args.request_id, operation_values["manifest"].sources)
    elif args.operation == "integrate":
        operation_values = {
            "previous": _read(args.previous, cm.CandidateManifest.from_dict,
                              "previous candidate manifest"),
            "candidate": _read(args.manifest, cm.CandidateManifest.from_dict,
                               "candidate manifest"),
        }
        backend.plan(args.request_id, operation_values["candidate"].sources)
    elif args.operation == "start-batch":
        operation_values = {
            "batch": _read(args.batch, cm.ValidationBatch.from_dict,
                           "validation batch"),
            "row_set": _read(args.row_set, cm.RequiredRowSet.from_dict, "row set"),
            "candidate": _read(args.manifest, cm.CandidateManifest.from_dict,
                               "candidate manifest"),
            "comparator": _read(args.comparator, cm.CandidateManifest.from_dict,
                                "comparator manifest"),
        }
    elif args.operation == "record-row":
        operation_values = {
            "batch_id": args.batch_id,
            "row_set": _read(args.row_set, cm.RequiredRowSet.from_dict, "row set"),
            "row_state": _read(args.row_state, cm.ValidationRowState.from_dict,
                               "row state"),
        }
    else:
        operation_values = {
            "batch": _read(args.batch, cm.ValidationBatch.from_dict,
                           "validation batch"),
        }
    controller = CampaignController(
        resolved, args.store, config_generation=args.config_generation)
    controller.__enter__()
    try:
        transactions = ct.CandidateTransactions(controller, git_backend=backend)
        if args.operation == "init":
            result = transactions.initialize(
                request_id=args.request_id, **operation_values)
        elif args.operation == "integrate":
            result = transactions.integrate(
                request_id=args.request_id, **operation_values)
        elif args.operation == "start-batch":
            result = transactions.start_batch(
                request_id=args.request_id, **operation_values)
        elif args.operation == "record-row":
            result = transactions.record_row(
                request_id=args.request_id, **operation_values)
        else:
            result = transactions.complete_batch(
                request_id=args.request_id, **operation_values)
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    finally:
        controller.close()


def main(argv: Sequence[str] | None = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    try:
        if values and values[0] in {
                "init", "integrate", "start-batch", "record-row", "complete"}:
            return _mutate(values)
        args = _parser().parse_args(values)
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
    except (cm.CandidateError, cm.TransitionError, ct.CandidateTransactionError,
            ControlRefused, OSError) as exc:
        print(f"candidate operation refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SUMMARY_SCHEMA", "build_summary", "main"]
