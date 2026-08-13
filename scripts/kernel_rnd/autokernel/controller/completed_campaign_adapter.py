"""Project one clean live campaign journal into the lean sequencer contract.

This is an offline identity/banking join.  It launches nothing and writes
nothing.  A kept candidate becomes ``banked`` only when its journaled T0/T1
events themselves contain passing dispatch and mechanism gates and the measured
throughput clears both the accepted floor and MDE.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .. import least_commitment_receipts, schemas
from . import sequencer


class CompletedCampaignAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectedCampaign:
    envelope: sequencer.ProposalEnvelope
    run: sequencer.CampaignRun


def _required_passing_gate(event: Mapping[str, Any], block: str, gate_id: str) -> str:
    values = event.get(block)
    if not isinstance(values, Mapping):
        raise CompletedCampaignAdapterError(f"{event.get('event_id')}: {block} is absent")
    if values.get(gate_id) != schemas.PASS:
        raise CompletedCampaignAdapterError(
            f"{event.get('event_id')}: required {block}.{gate_id} is not PASS")
    return gate_id


def _require_speed_rank_admissible(event: Mapping[str, Any]) -> None:
    """Require the final T1's own verdict-bearing rank admission.

    A campaign's accept rule is deliberately simpler than the evaluator: it
    decides whether every precommitted paired block favoured the candidate and
    cleared the declared contribution floor.  The evaluator can subsequently
    withhold the speed rank when the *closed* T1 window exposes a void,
    incomplete discipline, or an effect below MDE.  Banking the former while
    ignoring the latter would turn an unrankable final evaluation into a
    durable frontier candidate.

    This value lives in the emitted event's search-discipline block, which is
    the event-bound projection of ``Verdict.speed_rank_admissible``.  Missing
    is refused rather than treated as the historical default: a legacy record
    cannot prove a rank it did not journal.
    """
    performance = event.get("performance")
    discipline = (performance.get("search_discipline")
                  if isinstance(performance, Mapping) else None)
    admitted = (discipline.get("speed_rank_admissible")
                if isinstance(discipline, Mapping) else None)
    if admitted is not True:
        raise CompletedCampaignAdapterError(
            f"{event.get('event_id')}: final T1 evaluation is not "
            "speed-rank-admissible")


def project(*, campaign_record: Mapping[str, Any], journal_root: str,
            campaign_id: str, proposal_id: str,
            completion_event_id: str) -> ProjectedCampaign:
    row = {
        "journal_root": journal_root, "campaign_id": campaign_id,
        "proposal_id": proposal_id, "completion_event_id": completion_event_id,
        "matched_control_id": None,
    }
    evidence = least_commitment_receipts._load_completed_evidence(row)
    envelope = sequencer.ProposalEnvelope(
        campaign=copy.deepcopy(campaign_record), proposal=copy.deepcopy(evidence.proposal))
    events = tuple(evidence.evaluations[event_id]
                   for event_id in evidence.candidate["evaluation_event_ids"])
    candidate = copy.deepcopy(evidence.candidate)
    decision = evidence.result.get("decision")
    keep = isinstance(decision, Mapping) and decision.get("keep") is True
    if keep:
        t0 = next((event for event in events
                   if event.get("tier") == "T0" and event.get("status") == "pass"), None)
        t1 = next((event for event in events
                   if event.get("tier") == "T1" and event.get("status") == "pass"), None)
        if t0 is None or t1 is None:
            raise CompletedCampaignAdapterError("banking requires passing T0 and T1 events")
        _require_speed_rank_admissible(t1)
        dispatch_gate = _required_passing_gate(
            t0, "stability", "no_fallback_dispatch_trace")
        mechanism_gate = _required_passing_gate(
            t1, "mechanism", "t1.parameter_intervention_explained")
        calibration = evidence.result.get("spec", {}).get("calibration")
        if not isinstance(calibration, Mapping):
            raise CompletedCampaignAdapterError("banking requires terminal calibration")
        observed = decision.get("median_relative")
        floor = calibration.get("contribution_floor")
        mde = calibration.get("mde")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in (observed, floor, mde)):
            raise CompletedCampaignAdapterError("banking throughput values are incomplete")
        if float(observed) < max(float(floor), float(mde)):
            raise CompletedCampaignAdapterError("kept throughput does not clear floor and MDE")
        candidate["status"] = "banked"
        candidate["champion_status"] = "frontier"
        candidate["banking_verdict"] = {
            "disposition": "banked",
            "t0": {"all_pass_event_id": t0["event_id"]},
            "sentinels": {"required_all_pass_event_ids": [t0["event_id"]]},
            "real_path_dispatch": {
                "resolution": "confirmed", "event_id": t0["event_id"],
                "gate_id": dispatch_gate,
            },
            "mechanism": {
                "resolution": "explained", "event_id": t1["event_id"],
                "gate_id": mechanism_gate,
            },
            "qualifying_axis": {
                "axis": "throughput", "evaluation_event_id": t1["event_id"],
                "resolution": "above_floor", "observed_effect": float(observed),
                "calibrated_floor": float(floor),
                "minimum_detectable_effect": float(mde),
                "non_dominated": None, "non_dominated_check_ref": None,
            },
        }
    else:
        candidate["status"] = "rejected"
        candidate["champion_status"] = "none"
        candidate.pop("banking_verdict", None)
    run = sequencer.CampaignRun((candidate,), events)
    sequencer._validate_campaign_run(run, envelope)
    return ProjectedCampaign(envelope, run)
