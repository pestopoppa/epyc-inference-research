#!/usr/bin/env python3
"""No-inference AutoKernel evidence-path dress rehearsal.

This CLI validates the prospective IQK/control inputs and emits a field-level
producer manifest.  It never opens a model, builds a kernel, benchmarks, claims
resources, mutates a journal, or grants live authority.  Completed real journals
are still required before the receipt projector and AP-WM-1 may run.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import campaign
from . import least_commitment_capture as capture
from . import least_commitment_receipts as receipts
from . import offline_least_commitment as offline
from . import schemas
from .controller import champion, completed_campaign_adapter, sequencer
from .release import closeout, live_material, packager, readiness, t3

SCHEMA = "epyc.autokernel.evidence_path_rehearsal.v1"
AUTHORITY = "architecture_regression_fixture"


class RehearsalError(ValueError):
    pass


def _load(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise RehearsalError(f"{path}: expected a JSON object")
    return raw


def verify_campaign_json_contract(model: str) -> dict[str, Any]:
    """Exercise the current campaign automation surface without inference."""
    machine, trace = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(trace):
        code = campaign.main(["--model", model, "--json"], out=machine)
    if code != 0:
        raise RehearsalError(f"campaign --json dry run exited {code}")
    try:
        payload = json.loads(machine.getvalue())
    except json.JSONDecodeError as exc:
        raise RehearsalError(
            "campaign --json did not emit exactly one JSON document") from exc
    if payload.get("executed") is not False \
            or payload.get("state") != "dry_run_composed":
        raise RehearsalError("campaign --json unexpectedly executed or did not compose")
    if "DRY RUN" not in trace.getvalue():
        raise RehearsalError("campaign --json trace was not routed to stderr")
    return {
        "stdout": "exactly_one_json_document",
        "trace": "stderr",
        "state": payload["state"],
        "executed": payload["executed"],
    }


def producer_manifest(*, intervention_proposal: Mapping[str, Any],
                      intervention_plan: capture.CapturePlan,
                      control_proposal: Mapping[str, Any],
                      control_plan: capture.CapturePlan) -> dict[str, Any]:
    """Return and validate the complete two-branch producer inventory."""
    fields: dict[str, str] = {
        "proposal": "campaign.HostOps.record_proposal",
        "evaluation_events": "campaign.HostOps._evaluation_events",
        "candidate_record": "campaign.HostOps.prepare_durable_records",
        "terminal_decided": "campaign._finish -> HostOps.journal",
        "matched_control_proposal": "least_commitment_capture.make_iqk_control_proposal",
        "matched_control_join": "least_commitment_receipts.assemble_plan",
        "diagnostic_receipts": "least_commitment_receipts.project",
        "outcome_receipts": "least_commitment_receipts.project",
        "matched_one_factor_receipt": "least_commitment_receipts.project",
        "archive": "least_commitment_archive_builder.build_archive",
        "ap_wm_report": "offline_least_commitment.evaluate_archive",
        "sequencer_admission": "controller.completed_campaign_adapter.project",
        "champion": "controller.champion.promote_composition",
        "release_material": "release.live_material.JournalReleaseMaterialCompiler",
        "readiness": "release.live_material -> release.readiness",
        "t3": "release.closeout.OperatorCloseout -> release.t3",
        "package": "release.closeout.OperatorCloseout -> release.packager",
    }
    for name in capture.DIAGNOSTICS:
        fields[f"diagnostic.{name}"] = (
            "least_commitment_capture.CapturePlan.diagnostics -> "
            "campaign.HostOps.prepare_durable_records")
    for name, reducer in capture.OUTCOME_REDUCERS.items():
        fields[f"outcome.{name}"] = (
            f"least_commitment_capture.materialize:{reducer}")
    for proposal, plan, role in (
        (intervention_proposal, intervention_plan, "intervention"),
        (control_proposal, control_plan, "control"),
    ):
        if plan.role != role:
            raise RehearsalError(f"{role} capture plan has role={plan.role!r}")
        if plan.raw["proposal_id"] != proposal["proposal_id"]:
            raise RehearsalError(f"{role} plan/proposal identity differs")
    if intervention_plan.raw["matched_control_proposal_id"] \
            != control_proposal["proposal_id"]:
        raise RehearsalError("intervention does not name the generated control proposal")
    if intervention_plan.raw["candidate_frame_id"] != control_plan.raw["candidate_frame_id"]:
        raise RehearsalError("control and intervention candidate frames differ")
    if intervention_proposal["representation_contract"]["frame_sha256"] \
            != control_proposal["representation_contract"]["frame_sha256"]:
        raise RehearsalError("control generation changed the representation frame")
    changed = sorted(
        key for key in intervention_plan.raw["factors"]
        if intervention_plan.raw["factors"].get(key) != control_plan.raw["factors"].get(key))
    if changed != [intervention_plan.raw["changed_factor"]]:
        raise RehearsalError(f"pair changes {changed}, expected one declared factor")
    # Import/callability checks make this a live wiring inventory, not prose that
    # can survive deletion of a downstream implementation.
    callables = (
        receipts.assemble_plan, receipts.project, offline.evaluate_archive,
        completed_campaign_adapter.project, sequencer._validate_campaign_run,
        champion.promote_composition,
        live_material.bind_sealed_candidate,
        live_material.JournalReleaseMaterialCompiler.compile,
        readiness.evaluate_t2_trigger, t3.run_t3, packager.assemble_release_package,
        closeout.OperatorCloseout.run,
    )
    if not all(callable(value) for value in callables):
        raise RehearsalError("a declared downstream producer is not callable")
    return {
        "schema": SCHEMA, "authority": AUTHORITY,
        "inference_started": False, "live_authority": False,
        "intervention_proposal_id": intervention_proposal["proposal_id"],
        "control_proposal_id": control_proposal["proposal_id"],
        "representation_frame_sha256": intervention_proposal[
            "representation_contract"]["frame_sha256"],
        "candidate_frame_id": intervention_plan.raw["candidate_frame_id"],
        "sole_changed_factor": changed[0],
        "field_producers": dict(sorted(fields.items())),
        "post_measurement_commands": [
            "python3 -m scripts.kernel_rnd.autokernel.least_commitment_receipts "
            "projection-plan.json --output-dir /absolute/new/output",
            "python3 -m scripts.kernel_rnd.autokernel.offline_least_commitment "
            "/absolute/new/output/archive.json --output ap-wm-report.json",
        ],
        "real_evidence_required": [
            "two distinct clean DECIDED journals",
            "T0 and T1 evaluation events for each candidate",
            "PASS production immutability and released resource claims",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--intervention-proposal", type=Path, required=True)
    parser.add_argument("--intervention-capture-plan", type=Path, required=True)
    parser.add_argument("--control-campaign-id", required=True)
    parser.add_argument("--control-proposal-id", required=True)
    parser.add_argument("--control-candidate-id", required=True)
    parser.add_argument("--control-capture-plan", type=Path, required=True)
    parser.add_argument("--control-proposal-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    intervention = _load(args.intervention_proposal)
    violations = schemas.validate_proposal_v3(intervention)
    if violations:
        raise RehearsalError("invalid intervention proposal: " + "; ".join(violations))
    control = capture.make_iqk_control_proposal(
        intervention, campaign_id=args.control_campaign_id,
        proposal_id=args.control_proposal_id)
    intervention_plan = capture.load(
        args.intervention_capture_plan, proposal=intervention,
        campaign_id=intervention["campaign_id"],
        candidate_id=_load(args.intervention_capture_plan)["candidate_id"])
    control_plan = capture.load(
        args.control_capture_plan, proposal=control,
        campaign_id=args.control_campaign_id, candidate_id=args.control_candidate_id)
    report = producer_manifest(
        intervention_proposal=intervention, intervention_plan=intervention_plan,
        control_proposal=control, control_plan=control_plan)
    report["campaign_cli_contract"] = verify_campaign_json_contract(args.model)
    args.control_proposal_output.write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
