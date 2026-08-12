#!/usr/bin/env python3
"""Prospective AK-WM-2 capture contract for a live campaign.

The offline AP-WM-1 evaluator must not guess values after seeing a result.  This
module therefore validates a hash-bound plan before the campaign takes a claim,
then reduces only three already-declared outcome functions over the completed
measurement.  It has no selector, champion, release, process, or inference API.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import schemas

SCHEMA = "epyc.autokernel.least_commitment_capture_plan.v2"
BLOCK_SCHEMA = "epyc.autokernel.least_commitment_capture.v2"
SOURCE_SCHEMA = "epyc.autokernel.least_commitment_diagnostic_source.v1"
HELDOUT_SCHEMA = "epyc.autokernel.least_commitment_heldout_outcome.v1"
FALSIFIER_SCHEMA = "epyc.autokernel.least_commitment_falsifier.v1"
ROLES = frozenset({"control", "intervention"})
DIAGNOSTICS = (
    "unsupported_scope_width", "compatible_future_mass", "k_rho",
    "information_gain", "novelty", "raw_impurity", "weighted_minority",
)
OUTCOME_REDUCERS = {
    "heldout_regime_transfer": "heldout_outcome_receipt.relative_effect",
    "falsifier_resolution": "role_specific_falsifier.v1",
    "noise_floor": "calibration.noise_floor_phi",
}
_FIELDS = frozenset({
    "schema", "capture_id", "campaign_id", "candidate_id", "proposal_id",
    "matched_experiment_id",
    "role", "matched_control_proposal_id", "candidate_frame_id", "regime",
    "surface", "intervention_id", "changed_factor", "factors", "diagnostics",
    "recodings", "diagnostic_source_receipts", "heldout_outcome_receipt",
    "outcome_reducers",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_quotient(cells: Any, label: str) -> dict[str, float]:
    """Reduce one predeclared representation quotient into AP-WM diagnostics."""
    if not isinstance(cells, list) or not cells:
        raise CapturePlanError(f"{label}: expected at least one diagnostic cell")
    unsupported = raw_impurity = compatible = minority = k_rho = 0.0
    demand_total = 0.0
    seen: set[str] = set()
    for index, cell in enumerate(cells):
        prefix = f"{label}[{index}]"
        if not isinstance(cell, Mapping) or set(cell) != {
                "cell_id", "demand_weight", "supported", "compatible",
                "report_mass", "regret_margin"}:
            raise CapturePlanError(f"{prefix}: malformed diagnostic cell")
        cell_id = _text(cell.get("cell_id"), f"{prefix}.cell_id")
        if cell_id in seen:
            raise CapturePlanError(f"{prefix}.cell_id: duplicate {cell_id!r}")
        seen.add(cell_id)
        demand = _finite(cell.get("demand_weight"), f"{prefix}.demand_weight",
                         non_negative=True)
        margin = _finite(cell.get("regret_margin"), f"{prefix}.regret_margin",
                         non_negative=True)
        if not isinstance(cell.get("supported"), bool) \
                or not isinstance(cell.get("compatible"), bool):
            raise CapturePlanError(f"{prefix}: supported/compatible must be booleans")
        reports = cell.get("report_mass")
        if not isinstance(reports, Mapping) or not reports:
            raise CapturePlanError(f"{prefix}.report_mass: expected non-empty mapping")
        masses = [_finite(value, f"{prefix}.report_mass.{key}", non_negative=True)
                  for key, value in reports.items()]
        if not math.isclose(sum(masses), 1.0, abs_tol=1e-12):
            raise CapturePlanError(f"{prefix}.report_mass: values must sum to one")
        cell_minority = 1.0 - max(masses)
        demand_total += demand
        unsupported += 0.0 if cell["supported"] else 1.0
        compatible += demand if cell["compatible"] else 0.0
        raw_impurity += 1.0 if sum(value > 0.0 for value in masses) > 1 else 0.0
        minority += demand * cell_minority
        k_rho += demand * margin * cell_minority
    if not math.isclose(demand_total, 1.0, abs_tol=1e-12):
        raise CapturePlanError(f"{label}: demand weights must sum to one")
    return {
        "unsupported_scope_width": unsupported,
        "compatible_future_mass": compatible,
        "k_rho": k_rho,
        "raw_impurity": raw_impurity,
        "weighted_minority": minority,
    }


def derive_diagnostics(source: Any, *, proposal: Mapping[str, Any],
                       candidate_frame_id: str) -> tuple[dict, dict]:
    """Derive diagnostics from a hash-bound prospective quotient source."""
    required = {
        "schema", "authority", "receipt_id", "proposal_sha256",
        "representation_frame_sha256", "candidate_frame_id",
        "do_not_repeat_match_ids", "quotients",
    }
    if not isinstance(source, Mapping) or set(source) != required:
        raise CapturePlanError("diagnostic source fields differ from the closed schema")
    if source.get("schema") != SOURCE_SCHEMA \
            or source.get("authority") != "prospective_observe_only":
        raise CapturePlanError("diagnostic source schema/authority differs")
    _text(source.get("receipt_id"), "diagnostic source receipt_id")
    if source.get("proposal_sha256") != schemas.content_hash(proposal):
        raise CapturePlanError("diagnostic source proposal_sha256 differs")
    frame = proposal["representation_contract"]["frame_sha256"]
    if source.get("representation_frame_sha256") != frame:
        raise CapturePlanError("diagnostic source representation frame differs")
    if source.get("candidate_frame_id") != candidate_frame_id:
        raise CapturePlanError("diagnostic source candidate frame differs")
    matches = source.get("do_not_repeat_match_ids")
    if not isinstance(matches, list) or any(
            not isinstance(value, str) or not value.strip() for value in matches):
        raise CapturePlanError("do_not_repeat_match_ids must be a list of ids")
    if len(matches) != len(set(matches)):
        raise CapturePlanError("do_not_repeat_match_ids contains duplicates")
    fixture_ids = set(proposal["representation_contract"][
        "semantics_preserving_recoding_fixture_ids"])
    quotients = source.get("quotients")
    if not isinstance(quotients, Mapping) \
            or set(quotients) != fixture_ids | {"canonical"}:
        raise CapturePlanError(
            "diagnostic source quotients must cover canonical plus every recoding")
    def complete(values: dict[str, float]) -> dict[str, float]:
        return {
            **values,
            "information_gain": float(proposal["expected_information_gain"]),
            "novelty": 1.0 if not matches else 0.0,
        }
    diagnostics = complete(_derive_quotient(quotients["canonical"], "quotients.canonical"))
    recodings = {
        fixture_id: complete(_derive_quotient(
            quotients[fixture_id], f"quotients.{fixture_id}"))
        for fixture_id in sorted(fixture_ids)
    }
    return diagnostics, recodings


def source_binding(path: str | Path) -> dict:
    """Return the exact plan binding for one diagnostic source file."""
    source_path = Path(path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return {
        "path": str(source_path),
        "receipt_id": _text(source.get("receipt_id"), "diagnostic source receipt_id"),
        "sha256": _sha256_file(source_path),
    }


@dataclass(frozen=True)
class CapturePlan:
    raw: Mapping[str, Any]
    heldout_outcome: Mapping[str, Any]
    diagnostic_semantics_sha256: str

    @property
    def role(self) -> str:
        return self.raw["role"]

    @property
    def plan_sha256(self) -> str:
        return self.raw["plan_sha256"]

    def to_dict(self) -> dict:
        return json.loads(schemas.canonical_json(self.raw))

def _bound_json_receipt(binding: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != {
            "path", "receipt_id", "sha256"}:
        raise CapturePlanError(f"{label} must be {{path, receipt_id, sha256}}")
    path = Path(_text(binding.get("path"), f"{label}.path"))
    if not path.is_absolute():
        raise CapturePlanError(f"{label}.path must be absolute")
    receipt_id = _text(binding.get("receipt_id"), f"{label}.receipt_id")
    schemas.require.sha256(
        binding.get("sha256"), f"{label}.sha256", error=CapturePlanError)
    try:
        observed = _sha256_file(path)
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapturePlanError(f"{label}: cannot read source: {exc}") from exc
    if observed != binding["sha256"]:
        raise CapturePlanError(f"{label}: source SHA-256 differs")
    if not isinstance(source, Mapping) or source.get("receipt_id") != receipt_id:
        raise CapturePlanError(f"{label}: receipt identity differs")
    return source


def _validate_heldout_outcome(
        source: Mapping[str, Any], *, proposal: Mapping[str, Any],
        candidate_frame_id: str, target_regimes: set[str]) -> dict[str, Any]:
    required = {
        "schema", "authority", "receipt_id", "proposal_id", "proposal_sha256",
        "candidate_frame_id", "regime", "surface", "metric", "metric_direction",
        "relative_effect", "measurement_record_sha256", "capture_mode",
    }
    if set(source) != required:
        raise CapturePlanError("heldout outcome fields differ from the closed schema")
    if source.get("schema") != HELDOUT_SCHEMA \
            or source.get("authority") != "observe_only_measurement" \
            or source.get("capture_mode") != "measured":
        raise CapturePlanError("heldout outcome schema/authority/capture_mode differs")
    if source.get("proposal_id") != proposal.get("proposal_id") \
            or source.get("proposal_sha256") != schemas.content_hash(proposal):
        raise CapturePlanError("heldout outcome proposal binding differs")
    if source.get("candidate_frame_id") != candidate_frame_id:
        raise CapturePlanError("heldout outcome candidate frame differs")
    regime = _text(source.get("regime"), "heldout outcome regime")
    if regime in target_regimes:
        raise CapturePlanError(
            "heldout outcome regime must be outside the proposal target regimes")
    for key in ("surface", "metric"):
        _text(source.get(key), f"heldout outcome {key}")
    if source.get("metric_direction") not in {"higher", "lower"}:
        raise CapturePlanError("heldout outcome metric_direction must be higher or lower")
    schemas.require.sha256(
        source.get("measurement_record_sha256"),
        "heldout outcome measurement_record_sha256", error=CapturePlanError)
    return {
        **dict(source),
        "relative_effect": _finite(
            source.get("relative_effect"), "heldout outcome relative_effect"),
    }


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
    matched_experiment_id = _text(
        raw.get("matched_experiment_id"), "matched_experiment_id")
    if not matched_experiment_id.startswith("akm-"):
        raise CapturePlanError("matched_experiment_id must start with 'akm-'")
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
    source_records: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for key, receipt in receipts.items():
        label = f"diagnostic_source_receipts.{key}"
        source = _bound_json_receipt(receipt, label=label)
        source_records[(receipt["path"], receipt["receipt_id"], receipt["sha256"])] = source
    if len(source_records) != 1:
        raise CapturePlanError(
            "all diagnostics must come from one mechanically reduced source receipt")
    source = next(iter(source_records.values()))
    derived, derived_recodings = derive_diagnostics(
        source, proposal=proposal, candidate_frame_id=raw["candidate_frame_id"])
    if diagnostics != derived:
        raise CapturePlanError("diagnostics differ from the bound source derivation")
    if recodings != derived_recodings:
        raise CapturePlanError("recodings differ from the bound source derivation")
    if raw.get("outcome_reducers") != OUTCOME_REDUCERS:
        raise CapturePlanError("outcome_reducers must be the immutable live reducer set")
    target_regimes = set(proposal.get("target", {}).get("regimes", ()))
    if target_regimes and raw["regime"] not in target_regimes:
        raise CapturePlanError("regime is outside the proposal target vocabulary")
    heldout_source = _bound_json_receipt(
        raw.get("heldout_outcome_receipt"), label="heldout_outcome_receipt")
    heldout = _validate_heldout_outcome(
        heldout_source, proposal=proposal,
        candidate_frame_id=raw["candidate_frame_id"],
        target_regimes=target_regimes)
    return CapturePlan(
        json.loads(schemas.canonical_json(raw)),
        json.loads(schemas.canonical_json(heldout)),
        schemas.content_hash({
            "do_not_repeat_match_ids": source["do_not_repeat_match_ids"],
            "quotients": source["quotients"],
        }))


def bind_executed_factor_frame(
    plan: CapturePlan, *, matched_experiment_id: str,
    factors: Mapping[str, Any],
) -> None:
    """Fail closed unless the prospective plan names every executed axis.

    ``CampaignSpec`` derives ``factors`` from the actual recipe, model bytes,
    seeds, topology, calibration, envelope, provider and registered parameter.
    The plan may bind those bytes prospectively, but it cannot choose the
    vocabulary by omission.
    """
    if plan.raw.get("matched_experiment_id") != matched_experiment_id:
        raise CapturePlanError("matched experiment identity differs at execution")
    if not isinstance(factors, Mapping) or not factors:
        raise CapturePlanError("executed factor frame is empty")
    expected = json.loads(schemas.canonical_json(factors))
    if plan.raw.get("factors") != expected:
        missing = sorted(set(expected) - set(plan.raw.get("factors", {})))
        extra = sorted(set(plan.raw.get("factors", {})) - set(expected))
        changed = sorted(
            key for key in set(expected) & set(plan.raw.get("factors", {}))
            if plan.raw["factors"][key] != expected[key])
        raise CapturePlanError(
            "capture factors differ from mechanically executed frame: "
            f"missing={missing}, extra={extra}, changed={changed}")


def load(path: str | Path, *, proposal: Mapping[str, Any], campaign_id: str,
         candidate_id: str) -> CapturePlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_mapping(raw, proposal=proposal, campaign_id=campaign_id,
                        candidate_id=candidate_id)


def materialize(plan: CapturePlan, *, decision: Any, calibration: Any,
                executed_factors: Mapping[str, Any]) -> dict:
    """Reduce a validated pre-run plan with the completed measured decision."""
    if decision is None or calibration is None:
        raise CapturePlanError("a measured decision and calibration are required")
    median = _finite(getattr(decision, "median_relative", None),
                     "decision.median_relative")
    floor = _finite(getattr(decision, "contribution_floor", None),
                    "decision.contribution_floor", non_negative=True)
    noise = _finite(getattr(calibration, "noise_floor_phi", None),
                    "calibration.noise_floor_phi", non_negative=True)
    keep = getattr(decision, "keep", None)
    if not isinstance(keep, bool):
        raise CapturePlanError("decision.keep must be a boolean")
    raw = plan.raw
    bind_executed_factor_frame(
        plan, matched_experiment_id=raw["matched_experiment_id"],
        factors=executed_factors)
    if raw["role"] == "control":
        decision_triggered = keep
        noise_triggered = abs(median) > noise
        falsifier_resolution = noise - abs(median)
        trigger_rule = "decision.keep OR abs(decision.median_relative) > noise_floor"
    else:
        decision_triggered = not keep
        noise_triggered = False
        falsifier_resolution = median - floor
        trigger_rule = "NOT decision.keep"
    falsifier = {
        "schema": FALSIFIER_SCHEMA,
        "role": raw["role"],
        "trigger_rule": trigger_rule,
        "triggered": decision_triggered or noise_triggered,
        "predicates": {
            "keep_decision": keep,
            "absolute_effect": abs(median),
            "noise_floor": noise,
            "decision_triggered": decision_triggered,
            "noise_exceeded": noise_triggered,
        },
    }
    return {
        "schema": BLOCK_SCHEMA,
        "capture_id": raw["capture_id"],
        "capture_plan_sha256": raw["plan_sha256"],
        "capture_mode": raw["capture_mode"],
        "matched_experiment_id": raw["matched_experiment_id"],
        "role": raw["role"],
        "matched_control_proposal_id": raw["matched_control_proposal_id"],
        "candidate_frame_id": raw["candidate_frame_id"],
        "regime": raw["regime"], "surface": raw["surface"],
        "intervention_id": raw["intervention_id"],
        "changed_factor": raw["changed_factor"],
        "factors": json.loads(schemas.canonical_json(executed_factors)),
        "diagnostics": dict(raw["diagnostics"]),
        "recodings": {key: dict(value) for key, value in raw["recodings"].items()},
        "diagnostic_source_receipts": dict(raw["diagnostic_source_receipts"]),
        "diagnostic_semantics_sha256": plan.diagnostic_semantics_sha256,
        "heldout_outcome_receipt": dict(raw["heldout_outcome_receipt"]),
        "heldout_outcome": dict(plan.heldout_outcome),
        "outcome_reducers": dict(OUTCOME_REDUCERS),
        "falsifier": falsifier,
        "outcome": {
            "heldout_regime_transfer": plan.heldout_outcome["relative_effect"],
            "falsifier_resolution": falsifier_resolution,
            "noise_floor": noise,
        },
    }


def require_independent_control_diagnostics(
        intervention: CapturePlan, control: CapturePlan) -> None:
    """Refuse a control whose diagnostic semantics were copied from its intervention."""
    if intervention.role != "intervention" or control.role != "control":
        raise CapturePlanError("diagnostic independence requires intervention/control roles")
    intervention_binding = next(iter(
        intervention.raw["diagnostic_source_receipts"].values()))
    control_binding = next(iter(control.raw["diagnostic_source_receipts"].values()))
    if (intervention_binding["receipt_id"] == control_binding["receipt_id"]
            or intervention_binding["sha256"] == control_binding["sha256"]):
        raise CapturePlanError(
            "control diagnostic source must be independently bound")
    if intervention.diagnostic_semantics_sha256 == control.diagnostic_semantics_sha256:
        raise CapturePlanError(
            "control diagnostic semantics are identical to the intervention source")


def make_iqk_control_proposal(intervention: Mapping[str, Any], *, campaign_id: str,
                              proposal_id: str) -> dict:
    """Create the exact current-schema A/A control for an IQK intervention.

    The control holds both arms at the production setting.  This function creates
    only the proposal shell: it intentionally does not derive or copy a diagnostic
    source.  It is not executable
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
