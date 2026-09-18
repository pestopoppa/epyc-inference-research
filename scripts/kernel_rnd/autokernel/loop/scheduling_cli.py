#!/usr/bin/env python3
"""Offline inspection CLI for bounded scheduler selection and accounting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import scheduling as scheduler
from . import status
from .campaign import ResolvedCampaign

OUTPUT_SCHEMA = "epyc.autokernel.scheduler_inspection.v1"
RECEIPT_INPUT_SCHEMA = "epyc.autokernel.scheduler_receipt_input.v1"


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise scheduler.SchedulingRefused(f"{path}: {exc}") from exc


def _resolved(path: Path) -> ResolvedCampaign:
    value = _json(path)
    if isinstance(value, dict) and "resolved_campaign" in value:
        value = value["resolved_campaign"]
    try:
        return ResolvedCampaign.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise scheduler.SchedulingRefused(f"resolved campaign is invalid: {exc}") from exc


def _scheduler_identity(campaign: ResolvedCampaign) -> str:
    return f"{campaign.campaign_id}:{scheduler.digest(campaign.to_dict())}"


def _target_identity(target) -> str:
    return "target-group:" + scheduler.digest({
        "target_ids": list(target.target_ids),
        "revision": target.revision,
        "workload_signature": target.workload_signature,
    })


def _validate_campaign_inputs(campaign: ResolvedCampaign,
                              proposals: tuple[scheduler.StageProposal, ...]
                              ) -> tuple[scheduler.StageProposal, ...]:
    roster = {}
    for target in campaign.targets:
        for target_id in target.target_ids:
            roster[f"{target_id}@{target.revision}"] = target
    normalized = []
    for proposal in proposals:
        target = roster.get(proposal.target_revision)
        if target is None:
            raise scheduler.SchedulingRefused(
                f"proposal {proposal.proposal_id} references unknown target revision")
        backend = target.execution.backend
        if proposal.backend != backend and backend != "both":
            raise scheduler.SchedulingRefused(
                f"proposal {proposal.proposal_id} backend differs from resolved target")
        if proposal.alias_identity != target.workload_signature:
            raise scheduler.SchedulingRefused(
                f"proposal {proposal.proposal_id} alias identity is not the resolved workload")
        if target.status != "ready":
            if (proposal.stage_class not in {"prerequisite", "build"}
                    or proposal.production_frontier or proposal.seed_id is not None):
                raise scheduler.SchedulingRefused(
                    f"proposal {proposal.proposal_id} target is not ready; prerequisite work required")
        if proposal.production_frontier:
            frontier_target = roster.get(proposal.frontier_id)
            if "production" not in target.enrolled_as or frontier_target is not target:
                raise scheduler.SchedulingRefused(
                    f"proposal {proposal.proposal_id} is not a resolved production frontier")
        if proposal.seed_id is not None and "seed" not in target.enrolled_as:
            raise scheduler.SchedulingRefused(
                f"proposal {proposal.proposal_id} is not a resolved seed")
        canonical = _target_identity(target)
        row = proposal.to_dict()
        row["target_revision"] = canonical
        if proposal.production_frontier:
            row["frontier_id"] = canonical
        normalized.append(scheduler.StageProposal.from_dict(row))
    return tuple(normalized)


def inspect_files(*, campaign_path: Path, config_path: Path, proposals_path: Path,
                  now: float, state_path: Path | None = None,
                  receipts_path: Path | None = None,
                  outages_path: Path | None = None) -> dict:
    campaign = _resolved(campaign_path)
    config = scheduler.SchedulerConfig.from_dict(_json(config_path))
    proposals_raw = _json(proposals_path)
    if not isinstance(proposals_raw, list):
        raise scheduler.SchedulingRefused("proposals JSON must be an array")
    proposals = tuple(scheduler.StageProposal.from_dict(row) for row in proposals_raw)
    proposals = _validate_campaign_inputs(campaign, proposals)
    scheduler_id = _scheduler_identity(campaign)
    state = (scheduler.initial_state(config, scheduler_id)
             if state_path is None else scheduler.SchedulerState.from_dict(_json(state_path)))
    if state.scheduler_id != scheduler_id:
        raise scheduler.SchedulingRefused(
            "scheduler state belongs to a different resolved campaign snapshot")
    receipts_raw = [] if receipts_path is None else _json(receipts_path)
    outages_raw = [] if outages_path is None else _json(outages_path)
    if not isinstance(receipts_raw, list) or not isinstance(outages_raw, list):
        raise scheduler.SchedulingRefused("receipts and outages JSON must be arrays")
    outages = tuple(scheduler.Outage.from_dict(row) for row in outages_raw)
    engine = scheduler.SchedulerEngine(config, state)
    selection = engine.select_stage(proposals, now=now, outages=outages)
    if len(receipts_raw) > 1:
        raise scheduler.SchedulingRefused(
            "offline transition accepts at most one receipt for its selected opportunity")
    if receipts_raw:
        row = receipts_raw[0]
        if not isinstance(row, dict) or set(row) != {"schema", "receipt", "outcome"} \
                or row.get("schema") != RECEIPT_INPUT_SCHEMA:
            raise scheduler.SchedulingRefused("receipt input has unsupported fields/schema")
        receipt = scheduler.HeldClaimReceipt.from_dict(row["receipt"])
        declared = {proposal.proposal_id: proposal for proposal in proposals}
        if receipt.proposal_id not in declared:
            raise scheduler.SchedulingRefused("receipt references an undeclared proposal")
        engine.account_stage(selection, receipt, outcome=row["outcome"])
    next_state = engine.export_state()
    return {"schema": OUTPUT_SCHEMA, "campaign_id": campaign.campaign_id,
            "scheduler_config_digest": config.digest,
            "configuration_status": "provisional_not_statistical_policy",
            "state": next_state.to_dict(), "selection": selection.to_dict(),
            "actual_accounting": engine.accounting_view().to_dict(),
            "execution_authorized": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect deterministic scheduler state without allocating resources")
    parser.add_argument("--resolved-campaign", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--outages", type=Path)
    parser.add_argument("--now", type=float, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        output = inspect_files(
            campaign_path=args.resolved_campaign, config_path=args.config,
            proposals_path=args.proposals, now=args.now, state_path=args.state,
            receipts_path=args.receipts, outages_path=args.outages)
        if args.out is not None:
            status.write_json(args.out.parent, args.out.name, output,
                              prefix=".scheduler-inspection-")
    except (scheduler.SchedulingRefused, TypeError, ValueError) as exc:
        print(f"scheduler inspection refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
