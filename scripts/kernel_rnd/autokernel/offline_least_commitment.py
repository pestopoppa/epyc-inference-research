#!/usr/bin/env python3
"""Immutable, observe-only AP-WM-1 comparison for completed proposal archives.

This module deliberately has no selector or promotion API. It compares diagnostics
inside one predeclared representation/demand frame and emits an offline report. A
caller cannot ask it to mutate fitness, archive admission, champion state, or T2/T3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

try:
    from . import schemas
except ImportError:  # direct script execution
    import schemas


ARCHIVE_SCHEMA = "epyc.autokernel.least_commitment_archive.v1"
REPORT_SCHEMA = "epyc.autokernel.least_commitment_report.v1"
PROTOCOL_ID = "AP-WM-1/offline-v1"
AUTHORITY = "observe_only"
REAL_REPORT_PROVENANCE_SCHEMA = "epyc.autokernel.least_commitment_report_provenance.v1"
PROJECTION_SCHEMA = "epyc.autokernel.least_commitment_receipt_projection_result.v1"
BUILD_SCHEMA = "epyc.autokernel.least_commitment_archive_build.v1"

DIAGNOSTICS = (
    "unsupported_scope_width",
    "compatible_future_mass",
    "k_rho",
    "information_gain",
    "novelty",
    "raw_impurity",
    "weighted_minority",
)
NEW_DIAGNOSTICS = DIAGNOSTICS[:3]
BASELINE_DIAGNOSTICS = DIAGNOSTICS[3:]
DIRECTIONS = frozenset({"higher", "lower"})


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(archive: Any) -> list[str]:
    """Return all protocol violations; an empty list means evaluation is allowed."""
    errors: list[str] = []
    if not isinstance(archive, Mapping):
        return ["archive: expected a mapping"]
    if archive.get("schema") != ARCHIVE_SCHEMA:
        errors.append(f"schema: expected {ARCHIVE_SCHEMA!r}")
    if archive.get("protocol_id") != PROTOCOL_ID:
        errors.append(f"protocol_id: expected {PROTOCOL_ID!r}")
    if archive.get("authority") != AUTHORITY:
        errors.append("authority: must be 'observe_only'")
    for key in ("archive_id", "created_at", "candidate_frame_id"):
        if not isinstance(archive.get(key), str) or not archive[key].strip():
            errors.append(f"{key}: required non-empty string")
    directions = archive.get("diagnostic_directions")
    if not isinstance(directions, Mapping):
        errors.append("diagnostic_directions: required mapping")
    else:
        if set(directions) != set(DIAGNOSTICS):
            errors.append("diagnostic_directions: must declare exactly the protocol diagnostics")
        for key, value in directions.items():
            if value not in DIRECTIONS:
                errors.append(f"diagnostic_directions.{key}: expected 'higher' or 'lower'")
    weights = archive.get("outcome_weights")
    if not isinstance(weights, Mapping):
        errors.append("outcome_weights: required mapping")
    else:
        if set(weights) != {"heldout_regime_transfer", "falsifier_resolution"}:
            errors.append("outcome_weights: must declare transfer and falsifier weights")
        elif not all(_number(value) and value >= 0 for value in weights.values()):
            errors.append("outcome_weights: values must be finite non-negative numbers")
        elif not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
            errors.append("outcome_weights: values must sum to 1")

    rows = archive.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows: matched completed-proposal archive must not be empty")
        return errors
    by_id: dict[str, Mapping[str, Any]] = {}
    frames: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"rows[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix}: expected mapping")
            continue
        for key in (
            "proposal_id",
            "completion_event_id",
            "candidate_frame_id",
            "regime",
            "surface",
            "intervention_id",
            "changed_factor",
        ):
            if not isinstance(row.get(key), str) or not row[key].strip():
                errors.append(f"{prefix}.{key}: required non-empty string")
        proposal_id = row.get("proposal_id")
        if isinstance(proposal_id, str):
            if proposal_id in by_id:
                errors.append(f"{prefix}.proposal_id: duplicate {proposal_id!r}")
            by_id[proposal_id] = row
        if row.get("candidate_frame_id") != archive.get("candidate_frame_id"):
            errors.append(f"{prefix}.candidate_frame_id: cross-frame row is not comparable")
        contract = row.get("representation_contract")
        contract_errors = schemas.validate_representation_contract(
            contract, f"{prefix}.representation_contract."
        )
        errors.extend(contract_errors)
        if isinstance(contract, Mapping):
            frame = contract.get("frame_sha256")
            if isinstance(frame, str):
                frames.add(frame)
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(DIAGNOSTICS):
            errors.append(f"{prefix}.diagnostics: must declare exactly the protocol diagnostics")
        elif not all(_number(value) for value in diagnostics.values()):
            errors.append(f"{prefix}.diagnostics: values must be finite numbers")
        outcome = row.get("outcome")
        if not isinstance(outcome, Mapping):
            errors.append(f"{prefix}.outcome: required mapping")
        else:
            for key in ("heldout_regime_transfer", "falsifier_resolution", "noise_floor"):
                if not _number(outcome.get(key)) or (key == "noise_floor" and outcome[key] < 0):
                    errors.append(f"{prefix}.outcome.{key}: invalid finite number")
        recodings = row.get("recodings")
        fixture_ids = (
            contract.get("semantics_preserving_recoding_fixture_ids", [])
            if isinstance(contract, Mapping)
            else []
        )
        if not isinstance(recodings, Mapping) or set(recodings) != set(fixture_ids):
            errors.append(f"{prefix}.recodings: must cover every declared recoding fixture")
        elif any(
            not isinstance(values, Mapping)
            or set(values) != set(DIAGNOSTICS)
            or not all(_number(value) for value in values.values())
            for values in recodings.values()
        ):
            errors.append(f"{prefix}.recodings: each fixture must carry all finite diagnostics")
    if len(frames) > 1:
        errors.append("rows: representation/demand frames differ; ordering is not comparable")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        control_id = row.get("matched_control_id")
        if control_id is None:
            continue
        if not isinstance(control_id, str) or control_id not in by_id:
            errors.append(f"rows[{index}].matched_control_id: does not resolve")
        elif control_id == row.get("proposal_id"):
            errors.append(f"rows[{index}].matched_control_id: cannot reference itself")
        else:
            control = by_id[control_id]
            for key in ("candidate_frame_id", "regime", "surface"):
                if row.get(key) != control.get(key):
                    errors.append(f"rows[{index}].matched_control_id: {key} is not matched")
    if not any(isinstance(row, Mapping) and row.get("matched_control_id") for row in rows):
        errors.append(
            "rows: at least one completed one-factor intervention/control pair is required"
        )
    return errors


def validate_real_report_provenance(
        archive: Mapping[str, Any], projection: Any) -> list[str]:
    """Validate the strict builder/projector chain required for a real report."""
    errors: list[str] = []
    build = archive.get("build_provenance")
    if not isinstance(build, Mapping) or build.get("schema") != BUILD_SCHEMA:
        errors.append("build_provenance: strict archive builder provenance is required")
    if not isinstance(projection, Mapping):
        return errors + ["projection_provenance: required mapping"]
    if projection.get("schema") != PROJECTION_SCHEMA:
        errors.append(f"projection_provenance.schema: expected {PROJECTION_SCHEMA!r}")
    if projection.get("authority") != "observe_only_journal_projection":
        errors.append("projection_provenance.authority: journal projection required")
    archive_ref = projection.get("archive")
    if not isinstance(archive_ref, Mapping) or set(archive_ref) != {"path", "sha256"}:
        errors.append("projection_provenance.archive: expected {path, sha256}")
    if projection.get("archive_sha256") != schemas.content_hash(archive):
        errors.append("projection_provenance.archive_sha256: archive content differs")
    emitted = projection.get("emitted_receipts")
    if not isinstance(emitted, list) or not emitted:
        errors.append("projection_provenance.emitted_receipts: required non-empty list")
    rows = archive.get("rows")
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            sources = row.get("source_receipts") if isinstance(row, Mapping) else None
            if not isinstance(sources, Mapping):
                errors.append(f"rows[{index}].source_receipts: strict provenance required")
                continue
            for key in ("proposal_event_id", "proposal_sha256",
                        "campaign_result_sha256", "hypothesis_binding",
                        "diagnostic", "outcome"):
                if not sources.get(key):
                    errors.append(f"rows[{index}].source_receipts.{key}: required")
            for key in ("diagnostic", "outcome", "matched_intervention"):
                binding = sources.get(key)
                if binding is None and key == "matched_intervention" \
                        and row.get("matched_control_id") is None:
                    continue
                if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
                    errors.append(
                        f"rows[{index}].source_receipts.{key}: expected {{path, sha256}}")
                    continue
                path = Path(str(binding.get("path", "")))
                if not path.is_absolute() or not path.is_file():
                    errors.append(
                        f"rows[{index}].source_receipts.{key}: bound file is absent")
                elif binding.get("sha256") != _file_sha256(path):
                    errors.append(
                        f"rows[{index}].source_receipts.{key}: file SHA-256 differs")
    return errors


def _tau(x: list[float], y: list[float]) -> float | None:
    concordant = discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            product = (x[i] - x[j]) * (y[i] - y[j])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denominator = concordant + discordant
    return None if denominator == 0 else (concordant - discordant) / denominator


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _signed_delta(value: float, control: float, direction: str) -> float:
    delta = value - control
    return delta if direction == "higher" else -delta


def evaluate_archive(
        archive: Mapping[str, Any], *, projection: Mapping[str, Any] | None = None,
        real_label: bool = False) -> dict[str, Any]:
    """Evaluate a valid archive and return an immutable observe-only report."""
    errors = validate_archive(archive)
    if errors:
        raise ValueError("invalid AP-WM-1 archive:\n- " + "\n- ".join(errors))
    if real_label:
        provenance_errors = validate_real_report_provenance(archive, projection)
        if provenance_errors:
            raise ValueError(
                "invalid AP-WM-1 real provenance:\n- "
                + "\n- ".join(provenance_errors))
    rows = archive["rows"]
    by_id = {row["proposal_id"]: row for row in rows}
    weights = archive["outcome_weights"]
    directions = archive["diagnostic_directions"]
    pairs: list[dict[str, Any]] = []
    for row in rows:
        control_id = row.get("matched_control_id")
        if not control_id:
            continue
        control = by_id[control_id]
        outcome_delta = sum(
            weights[key] * (row["outcome"][key] - control["outcome"][key]) for key in weights
        )
        noise_floor = max(row["outcome"]["noise_floor"], control["outcome"]["noise_floor"])
        diagnostic_deltas = {
            key: _signed_delta(row["diagnostics"][key], control["diagnostics"][key], direction)
            for key, direction in directions.items()
        }
        recoding_deltas = {
            fixture_id: {
                key: _signed_delta(values[key], control["recodings"][fixture_id][key], direction)
                for key, direction in directions.items()
            }
            for fixture_id, values in row["recodings"].items()
        }
        pairs.append(
            {
                "proposal_id": row["proposal_id"],
                "control_id": control_id,
                "regime": row["regime"],
                "surface": row["surface"],
                "outcome_delta": outcome_delta,
                "noise_floor": noise_floor,
                "intervention_noise_floor": row["outcome"]["noise_floor"],
                "control_noise_floor": control["outcome"]["noise_floor"],
                "effective": abs(outcome_delta) > noise_floor,
                "diagnostic_deltas": diagnostic_deltas,
                "recoding_deltas": recoding_deltas,
            }
        )

    def summarize(group_pairs: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for metric in DIAGNOSTICS:
            effective = [pair for pair in group_pairs if pair["effective"]]
            predicted = [pair["diagnostic_deltas"][metric] for pair in effective]
            observed = [pair["outcome_delta"] for pair in effective]
            sign_errors = [
                abs(obs) if pred == 0 or pred * obs < 0 else 0.0
                for pred, obs in zip(predicted, observed)
            ]
            non_tied = [(pred, obs) for pred, obs in zip(predicted, observed) if pred != 0]
            correct = sum(1 for pred, obs in non_tied if pred * obs > 0)
            fixture_taus = []
            fixture_ids = sorted(
                {fixture_id for pair in effective for fixture_id in pair["recoding_deltas"]}
            )
            for fixture_id in fixture_ids:
                recoded = [pair["recoding_deltas"][fixture_id][metric] for pair in effective]
                fixture_taus.append(
                    {"fixture_id": fixture_id, "kendall_tau": _tau(predicted, recoded)}
                )
            result[metric] = {
                "kendall_direction": _tau(predicted, observed),
                "conditional_predictive_value": (correct / len(non_tied) if non_tied else None),
                "mean_sign_error": mean(sign_errors) if sign_errors else None,
                "p90_sign_error": _percentile(sign_errors, 0.90),
                "worst_sign_error": max(sign_errors) if sign_errors else None,
                "effective_pairs": len(effective),
                "total_pairs": len(group_pairs),
                "noise_floor_exclusions": len(group_pairs) - len(effective),
                "recoding_stability": fixture_taus,
            }
        return result

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_regime_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_surface_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["regime"], pair["surface"])].append(pair)
        by_regime_groups[pair["regime"]].append(pair)
        by_surface_groups[pair["surface"]].append(pair)
    overall = summarize(pairs)
    by_regime_surface = {
        f"{regime}::{surface}": summarize(group_pairs)
        for (regime, surface), group_pairs in sorted(grouped.items())
    }
    baseline_cpv = max(
        (overall[name]["conditional_predictive_value"] or 0.0) for name in BASELINE_DIAGNOSTICS
    )
    independent = []
    for name in NEW_DIAGNOSTICS:
        stats = overall[name]
        stable = all(item["kendall_tau"] == 1.0 for item in stats["recoding_stability"])
        if (
            stable
            and stats["effective_pairs"] >= 5
            and (stats["conditional_predictive_value"] or 0.0) >= baseline_cpv + 0.05
        ):
            independent.append(name)
    recommendation = (
        "offline_independent_signal_observed_retain_observe_only"
        if independent
        else "retain_simpler_baseline"
    )
    effective_pair_count = sum(1 for pair in pairs if pair["effective"])
    minimum_effective_pairs = 5
    matched_validation = {
        "status": "PASS",
        "validated_pair_count": len(pairs),
        "all_controls_resolved": all(pair["control_id"] in by_id for pair in pairs),
        "candidate_frame_id": archive["candidate_frame_id"],
        "representation_frame_sha256": rows[0]["representation_contract"]["frame_sha256"],
    }
    power_status = (
        "adequately_powered"
        if effective_pair_count >= minimum_effective_pairs
        else "underpowered"
    )
    return {
        "schema": REPORT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "authority": AUTHORITY,
        "archive_id": archive["archive_id"],
        "archive_sha256": schemas.content_hash(archive),
        "candidate_frame_id": archive["candidate_frame_id"],
        "representation_frame_sha256": rows[0]["representation_contract"]["frame_sha256"],
        "diagnostic_directions": directions,
        "outcome_weights": weights,
        "pair_count": len(pairs),
        "pair_noise_floors": [{
            "proposal_id": pair["proposal_id"],
            "control_id": pair["control_id"],
            "intervention_noise_floor": pair["intervention_noise_floor"],
            "control_noise_floor": pair["control_noise_floor"],
            "effective_noise_floor": pair["noise_floor"],
            "effective": pair["effective"],
        } for pair in pairs],
        "matched_validation": matched_validation,
        "power": {
            "status": power_status,
            "effective_pair_count": effective_pair_count,
            "minimum_effective_pairs": minimum_effective_pairs,
        },
        "overall": overall,
        "by_regime": {
            name: summarize(group_pairs) for name, group_pairs in sorted(by_regime_groups.items())
        },
        "by_surface": {
            name: summarize(group_pairs) for name, group_pairs in sorted(by_surface_groups.items())
        },
        "by_regime_surface": by_regime_surface,
        "independent_new_diagnostics": independent,
        "recommendation": (
            "underpowered_retain_observe_only"
            if power_status == "underpowered" else recommendation),
        "evidence_label": "real" if real_label else "fixture_or_unlabelled",
        "report_provenance": ({
            "schema": REAL_REPORT_PROVENANCE_SCHEMA,
            "projection_result_sha256": schemas.content_hash(projection),
            "archive_build_schema": BUILD_SCHEMA,
        } if real_label else None),
        "live_authority": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--projection-result", type=Path, required=True,
        help="strict journal projector result required for a real-labelled report")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    projection = json.loads(args.projection_result.read_text(encoding="utf-8"))
    archive_ref = projection.get("archive") if isinstance(projection, Mapping) else None
    if not isinstance(archive_ref, Mapping) \
            or Path(str(archive_ref.get("path", ""))).resolve() != args.archive.resolve() \
            or archive_ref.get("sha256") != _file_sha256(args.archive):
        raise ValueError(
            "projection result does not hash-bind the exact archive input path")
    report = evaluate_archive(archive, projection=projection, real_label=True)
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
