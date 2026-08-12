#!/usr/bin/env python3
"""Prospective AK-WM-2 capture contract for a live campaign.

The offline AP-WM-1 evaluator must not guess values after seeing a result.  This
module therefore validates a hash-bound plan before the campaign takes a claim,
then reduces only three already-declared outcome functions over the completed
measurement.  It has no selector, champion, release, process, or inference API.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import schemas

SCHEMA = "epyc.autokernel.least_commitment_capture_plan.v1"
BLOCK_SCHEMA = "epyc.autokernel.least_commitment_capture.v1"
ROLES = frozenset({"control", "intervention"})
DIAGNOSTICS = (
    "unsupported_scope_width", "compatible_future_mass", "k_rho",
    "information_gain", "novelty", "raw_impurity", "weighted_minority",
)
OUTCOME_REDUCERS = {
    "heldout_regime_transfer": "decision.median_relative",
    "falsifier_resolution": "decision.median_relative_minus_contribution_floor",
    "noise_floor": "calibration.noise_floor_phi",
}
_FIELDS = frozenset({
    "schema", "capture_id", "campaign_id", "candidate_id", "proposal_id",
    "role", "matched_control_proposal_id", "candidate_frame_id", "regime",
    "surface", "intervention_id", "changed_factor", "factors", "diagnostics",
    "recodings", "diagnostic_source_receipts", "outcome_reducers",
    "capture_mode", "plan_sha256",
})


class CapturePlanError(ValueError):
    pass


def _finite(value: Any, label: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise CapturePlanError(f"{label}: expected a finite number")
    value = float(value)
    if non_negative and value < 0:
        raise CapturePlanError(f"{label}: expected a non-negative number")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise CapturePlanError(f"{label}: expected non-empty text without NUL")
    return value


def plan_sha256(raw: Mapping[str, Any]) -> str:
    return schemas.content_hash({key: raw[key] for key in sorted(raw)
                                 if key != "plan_sha256"})


@dataclass(frozen=True)
class CapturePlan:
    raw: Mapping[str, Any]

    @property
    def role(self) -> str:
        return self.raw["role"]

    @property
    def plan_sha256(self) -> str:
        return self.raw["plan_sha256"]

    def to_dict(self) -> dict:
        return json.loads(schemas.canonical_json(self.raw))


def from_mapping(raw: Any, *, proposal: Mapping[str, Any], campaign_id: str,
                 candidate_id: str) -> CapturePlan:
    """Validate a prospective plan and bind it to one exact proposal/run."""
    if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
        got = sorted(raw) if isinstance(raw, Mapping) else type(raw).__name__
        raise CapturePlanError(f"capture plan fields must be exactly {sorted(_FIELDS)}; got {got}")
    if raw.get("schema") != SCHEMA:
        raise CapturePlanError(f"schema must be {SCHEMA!r}")
    expected_hash = _text(raw.get("plan_sha256"), "plan_sha256")
    if expected_hash != plan_sha256(raw):
        raise CapturePlanError("plan_sha256 does not bind the capture plan")
    for key, expected in (("campaign_id", campaign_id),
                          ("candidate_id", candidate_id),
                          ("proposal_id", proposal.get("proposal_id"))):
        if raw.get(key) != expected:
            raise CapturePlanError(f"{key} does not match the campaign proposal")
    if raw.get("capture_mode") not in {"measured", "architecture_regression_fixture"}:
        raise CapturePlanError("capture_mode must be measured or architecture_regression_fixture")
    role = raw.get("role")
    if role not in ROLES:
        raise CapturePlanError(f"role must be one of {sorted(ROLES)}")
    control_id = raw.get("matched_control_proposal_id")
    if role == "control" and control_id is not None:
        raise CapturePlanError("control plan cannot name a matched control")
    if role == "intervention" and (not isinstance(control_id, str) or not control_id.strip()
                                   or control_id == proposal.get("proposal_id")):
        raise CapturePlanError("intervention plan requires another matched control proposal id")
    for key in ("capture_id", "candidate_frame_id", "regime", "surface",
                "intervention_id", "changed_factor"):
        _text(raw.get(key), key)
    if raw["changed_factor"] != "ggml_iqk":
        raise CapturePlanError("the live parameter path licenses only changed_factor=ggml_iqk")
    factors = raw.get("factors")
    if not isinstance(factors, Mapping) or not factors:
        raise CapturePlanError("factors must be a non-empty mapping")
    if "ggml_iqk" not in factors:
        raise CapturePlanError("factors must include ggml_iqk")
    surface = proposal.get("change", {}).get("parameter_surface", {})
    candidate_arm = surface.get("candidate", {}).get("ggml_iqk")
    if str(factors["ggml_iqk"]) != candidate_arm:
        raise CapturePlanError("factors.ggml_iqk does not match the proposal candidate arm")
    diagnostics = raw.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(DIAGNOSTICS):
        raise CapturePlanError("diagnostics must declare exactly the AP-WM-1 diagnostics")
    values = {key: _finite(diagnostics[key], f"diagnostics.{key}")
              for key in DIAGNOSTICS}
    if values["information_gain"] != float(proposal["expected_information_gain"]):
        raise CapturePlanError("diagnostics.information_gain must equal proposal expected_information_gain")
    fixture_ids = set(proposal["representation_contract"][
        "semantics_preserving_recoding_fixture_ids"])
    recodings = raw.get("recodings")
    if not isinstance(recodings, Mapping) or set(recodings) != fixture_ids:
        raise CapturePlanError("recodings must cover exactly the proposal representation fixtures")
    for fixture_id, recoded in recodings.items():
        if not isinstance(recoded, Mapping) or set(recoded) != set(DIAGNOSTICS):
            raise CapturePlanError(f"recodings.{fixture_id} must carry every diagnostic")
        for key in DIAGNOSTICS:
            _finite(recoded[key], f"recodings.{fixture_id}.{key}")
    receipts = raw.get("diagnostic_source_receipts")
    if not isinstance(receipts, Mapping) or set(receipts) != set(DIAGNOSTICS):
        raise CapturePlanError("diagnostic_source_receipts must bind every diagnostic")
    for key, receipt in receipts.items():
        if not isinstance(receipt, Mapping) or set(receipt) != {"receipt_id", "sha256"}:
            raise CapturePlanError(f"diagnostic_source_receipts.{key} must be {{receipt_id, sha256}}")
        _text(receipt.get("receipt_id"), f"diagnostic_source_receipts.{key}.receipt_id")
        schemas.require.sha256(
            receipt.get("sha256"),
            f"diagnostic_source_receipts.{key}.sha256",
            error=CapturePlanError,
        )
    if raw.get("outcome_reducers") != OUTCOME_REDUCERS:
        raise CapturePlanError("outcome_reducers must be the immutable live reducer set")
    target_regimes = set(proposal.get("target", {}).get("regimes", ()))
    if target_regimes and raw["regime"] not in target_regimes:
        raise CapturePlanError("regime is outside the proposal target vocabulary")
    return CapturePlan(json.loads(schemas.canonical_json(raw)))


def load(path: str | Path, *, proposal: Mapping[str, Any], campaign_id: str,
         candidate_id: str) -> CapturePlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_mapping(raw, proposal=proposal, campaign_id=campaign_id,
                        candidate_id=candidate_id)


def materialize(plan: CapturePlan, *, decision: Any, calibration: Any) -> dict:
    """Reduce a validated pre-run plan with the completed measured decision."""
    if decision is None or calibration is None:
        raise CapturePlanError("a measured decision and calibration are required")
    median = _finite(getattr(decision, "median_relative", None),
                     "decision.median_relative")
    floor = _finite(getattr(decision, "contribution_floor", None),
                    "decision.contribution_floor", non_negative=True)
    noise = _finite(getattr(calibration, "noise_floor_phi", None),
                    "calibration.noise_floor_phi", non_negative=True)
    raw = plan.raw
    return {
        "schema": BLOCK_SCHEMA,
        "capture_id": raw["capture_id"],
        "capture_plan_sha256": raw["plan_sha256"],
        "capture_mode": raw["capture_mode"],
        "role": raw["role"],
        "matched_control_proposal_id": raw["matched_control_proposal_id"],
        "candidate_frame_id": raw["candidate_frame_id"],
        "regime": raw["regime"], "surface": raw["surface"],
        "intervention_id": raw["intervention_id"],
        "changed_factor": raw["changed_factor"],
        "factors": dict(raw["factors"]),
        "diagnostics": dict(raw["diagnostics"]),
        "recodings": {key: dict(value) for key, value in raw["recodings"].items()},
        "diagnostic_source_receipts": dict(raw["diagnostic_source_receipts"]),
        "outcome_reducers": dict(OUTCOME_REDUCERS),
        "outcome": {
            "heldout_regime_transfer": median,
            "falsifier_resolution": median - floor,
            "noise_floor": noise,
        },
    }


def make_iqk_control_proposal(intervention: Mapping[str, Any], *, campaign_id: str,
                              proposal_id: str) -> dict:
    """Create the exact current-schema A/A control for an IQK intervention.

    The control holds both arms at the production setting.  It is not executable
    without a bound ``role=control`` capture plan, which is the campaign-side
    capability that distinguishes a predeclared control from an accidental no-op.
    """
    control = json.loads(schemas.canonical_json(intervention))
    control["campaign_id"] = _text(campaign_id, "campaign_id")
    control["proposal_id"] = _text(proposal_id, "proposal_id")
    parameter = control["change"]["parameter_surface"]
    anchor = parameter["anchor"]["ggml_iqk"]
    parameter["candidate"]["ggml_iqk"] = anchor
    control["hypothesis"] = "Matched A/A control for " + intervention["proposal_id"]
    if isinstance(control.get("narrative"), str):
        control["narrative"] = "Matched A/A control; no performance claim. " + control["narrative"]
    violations = schemas.validate_proposal(control)
    if violations:
        raise CapturePlanError("derived control proposal is invalid: " + "; ".join(violations))
    return control
