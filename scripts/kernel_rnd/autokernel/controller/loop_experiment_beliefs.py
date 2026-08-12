#!/usr/bin/env python3
"""Prospective belief projection for a complete AK-LE planner reduction.

This writer deliberately wraps, rather than changes, the reducer whose exact
bytes are pinned by an already-running planner panel.  It invokes that reducer
from the original manifest, panel, and prefilter contract, then adds one
self-hashed observation row for each predeclared search-persistence metric in
each planner cell.  It does not rank cells, select a champion, or acquire any
campaign or release authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from . import loop_experiment_prefilter as prefilter
from . import loop_experiment_runner as runner
from . import loop_experiments as experiments


PRODUCER_ID = "autokernel.controller.loop_experiment_beliefs/v1"
PRODUCER_REF = (
    "git://epyc-inference-research/scripts/kernel_rnd/autokernel/controller/"
    "loop_experiment_beliefs.py"
)
AUTHORITY = "observe_only_no_campaign_ranking_champion_or_release_authority"
REPS_BASIS = "scored:one complete hash-bound AK-LE planner cell"
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_METRICS = (
    (
        "novel_nonduplicate_count",
        "ak_le_planner_novel_nonduplicate_count",
        "count",
        "higher_better",
    ),
    (
        "prefilter_survival_count",
        "ak_le_planner_prefilter_survival_count",
        "count",
        "higher_better",
    ),
    (
        "already_optimized_termination_count",
        "ak_le_planner_already_optimized_termination",
        "indicator",
        "lower_better",
    ),
    (
        "elapsed_wall_seconds",
        "ak_le_planner_elapsed_wall_seconds",
        "seconds",
        "lower_better",
    ),
)


class BeliefProjectionError(ValueError):
    """The reduction cannot support an exact prospective belief projection."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise BeliefProjectionError(f"{label} must be a lowercase SHA-256")
    return value


def _producer_identity() -> dict[str, str]:
    return {
        "producer_id": PRODUCER_ID,
        "ref": PRODUCER_REF,
        "sha256": _file_sha(Path(__file__).resolve()),
    }


def _prediction_map(manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    embedded = manifest.get("experiment_contract")
    if not isinstance(embedded, Mapping):
        raise BeliefProjectionError("manifest lacks its embedded experiment contract")
    predictions = embedded.get("direction_predictions")
    if not isinstance(predictions, list):
        raise BeliefProjectionError("experiment direction predictions are missing")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {
            "model_id", "quant_id", "direction", "rationale"
        }:
            raise BeliefProjectionError("experiment direction prediction is malformed")
        key = (row["model_id"], row["quant_id"])
        if key in result:
            raise BeliefProjectionError("experiment direction prediction is duplicated")
        result[key] = dict(row)
    return result


def _measurement_rows(
    reduction: Mapping[str, Any], *, manifest: Mapping[str, Any],
    prefilter_contract: Mapping[str, Any], producer: Mapping[str, str],
) -> list[dict[str, Any]]:
    receipt = reduction.get("planner_receipt")
    evidence_rows = reduction.get("prefilter_evidence")
    if not isinstance(receipt, Mapping) or receipt.get("schema") != \
            experiments.PLANNER_RECEIPT_SCHEMA:
        raise BeliefProjectionError("nested planner receipt schema is unsupported")
    if not isinstance(evidence_rows, list):
        raise BeliefProjectionError("reduction lacks per-cell prefilter evidence")
    search_rows = receipt.get("search_persistence_observations")
    if not isinstance(search_rows, list) or len(search_rows) != len(evidence_rows):
        raise BeliefProjectionError("planner and prefilter cell counts differ")
    if receipt.get("authority") != experiments.AUTHORITY:
        raise BeliefProjectionError("planner receipt requests unsupported authority")
    constraints = receipt.get("constraints")
    if not isinstance(constraints, Mapping) or any(
        constraints.get(name) is not False for name in (
            "campaign_1_authority", "ranking_authority", "champion_authority",
            "release_authority", "controller_ab_authority",
        )
    ):
        raise BeliefProjectionError("planner receipt requests forbidden authority")

    reducer_producer = prefilter_contract.get("producer")
    if not isinstance(reducer_producer, dict) or set(reducer_producer) != {"ref", "sha256"}:
        raise BeliefProjectionError("prefilter reducer producer identity is malformed")
    _sha(reducer_producer.get("sha256"), "prefilter reducer producer SHA-256")
    predictions = _prediction_map(manifest)
    by_cell = {}
    for evidence in evidence_rows:
        if not isinstance(evidence, dict):
            raise BeliefProjectionError("prefilter evidence row is malformed")
        cell_id = evidence.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in by_cell:
            raise BeliefProjectionError("prefilter evidence cell identity is invalid")
        unsigned_evidence = dict(evidence)
        claimed_evidence = unsigned_evidence.pop("evidence_sha256", None)
        if _sha(claimed_evidence, "prefilter evidence SHA-256") != _digest(unsigned_evidence):
            raise BeliefProjectionError("prefilter evidence SHA-256 does not verify")
        by_cell[cell_id] = evidence

    rows: list[dict[str, Any]] = []
    for search in search_rows:
        if not isinstance(search, dict):
            raise BeliefProjectionError("search-persistence observation is malformed")
        cell_id = search.get("cell_id")
        evidence = by_cell.get(cell_id)
        if evidence is None or search.get("evidence_sha256") != evidence["evidence_sha256"]:
            raise BeliefProjectionError("search cell does not bind its prefilter evidence")
        decisions = evidence.get("decisions")
        if not isinstance(decisions, list):
            raise BeliefProjectionError("prefilter decision evidence is malformed")
        survived = sum(row.get("survived_prefilter") is True for row in decisions
                       if isinstance(row, dict))
        if (search.get("prefilter_survival_count") != survived
                or search.get("novel_nonduplicate_count") != survived):
            raise BeliefProjectionError(
                "search novelty/survival counts do not rederive from prefilter evidence")
        termination_count = int(search.get("termination") == "already_optimized")
        if search.get("already_optimized_termination_count") != termination_count:
            raise BeliefProjectionError(
                "already-optimized indicator does not rederive from termination")
        elapsed = search.get("elapsed_wall_seconds")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed <= 0):
            raise BeliefProjectionError("planner elapsed wall time is invalid")
        arm = {
            name: search.get(name) for name in (
                "cell_id", "model_id", "quant_id", "effort", "target_context_mode"
            )
        }
        if any(not isinstance(value, str) or not value for value in arm.values()):
            raise BeliefProjectionError("planner arm identity is incomplete")
        prediction = predictions.get((arm["model_id"], arm["quant_id"]))
        if prediction is None:
            raise BeliefProjectionError("planner arm lacks its predeclared direction")
        search_sha = _digest(search)
        for native_field, metric, unit, direction in _METRICS:
            value = search[native_field]
            row: dict[str, Any] = {
                "measurement_id": f"ak_le_{cell_id}_{native_field}",
                "metric": metric,
                "value": value,
                "unit": unit,
                "metric_direction": direction,
                "category": "BASELINE",
                "reps": 1,
                "reps_basis": REPS_BASIS,
                "claim": (
                    f"AK-LE planner cell {cell_id} observed {native_field}={value}"
                ),
                "extra": {
                    "measurement_role": "search_persistence_observation",
                    "native_field": native_field,
                    "metric_interpretation": (
                        "observed_cost_lower_is_better; not evidence that shorter search "
                        "is more persistent"
                        if native_field == "elapsed_wall_seconds"
                        else "predeclared search-persistence outcome"
                    ),
                    **arm,
                    "predeclared_direction": prediction["direction"],
                    "predeclared_direction_rationale": prediction["rationale"],
                    "scored_cell_basis": {
                        "reduction_schema": prefilter.REDUCTION_SCHEMA,
                        "planner_receipt_schema": experiments.PLANNER_RECEIPT_SCHEMA,
                        "planner_receipt_sha256": receipt["receipt_sha256"],
                        "search_persistence_observation_sha256": search_sha,
                    },
                    "manifest_sha256": reduction["manifest_sha256"],
                    "panel_sha256": reduction["panel_sha256"],
                    "prefilter_contract_sha256": reduction[
                        "prefilter_contract_sha256"],
                    "prefilter_evidence_sha256": evidence["evidence_sha256"],
                    "raw_observation_sha256": evidence["raw_observation_sha256"],
                    "projection_producer": dict(producer),
                    "prefilter_reducer_producer": dict(reducer_producer),
                    "authority": AUTHORITY,
                    "observation_only": True,
                    "campaign_1_authority": False,
                    "ranking_authority": False,
                    "champion_authority": False,
                    "release_authority": False,
                },
            }
            row["measurement_sha256"] = _digest(row)
            rows.append(row)
    return rows


def reduce_with_beliefs(
    *, manifest: Mapping[str, Any], panel: Mapping[str, Any],
    prefilter_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-run the pinned reducer and add write-side per-cell belief rows."""
    validated_manifest = runner.validate_manifest(manifest)
    reduction = prefilter.reduce_planner_panel(
        manifest=validated_manifest, panel=panel,
        prefilter_contract=prefilter_contract,
    )
    if "belief_measurements" in reduction:
        raise BeliefProjectionError("belief projection is write-once")
    result = copy.deepcopy(reduction)
    result["belief_measurements"] = _measurement_rows(
        result, manifest=validated_manifest,
        prefilter_contract=prefilter_contract, producer=_producer_identity(),
    )
    result.pop("reduction_sha256", None)
    result["reduction_sha256"] = _digest(result)
    return result


def validate_reduction_with_beliefs(
    reduction: Mapping[str, Any], *, manifest: Mapping[str, Any],
    panel: Mapping[str, Any], prefilter_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-derive the whole finalized reduction from its three exact sources."""
    if not isinstance(reduction, Mapping):
        raise BeliefProjectionError("belief reduction must be an object")
    payload = copy.deepcopy(dict(reduction))
    claimed = payload.pop("reduction_sha256", None)
    if _sha(claimed, "reduction SHA-256") != _digest(payload):
        raise BeliefProjectionError("belief reduction SHA-256 does not verify")
    rows = payload.get("belief_measurements")
    if not isinstance(rows, list) or not rows:
        raise BeliefProjectionError("belief reduction has no measurement rows")
    for row in rows:
        if not isinstance(row, dict):
            raise BeliefProjectionError("belief measurement must be an object")
        unsigned = dict(row)
        row_sha = unsigned.pop("measurement_sha256", None)
        if _sha(row_sha, "measurement SHA-256") != _digest(unsigned):
            raise BeliefProjectionError("belief measurement SHA-256 does not verify")
    expected = reduce_with_beliefs(
        manifest=manifest, panel=panel, prefilter_contract=prefilter_contract)
    if dict(reduction) != expected:
        raise BeliefProjectionError(
            "belief reduction does not exactly rederive from manifest/panel/prefilter evidence")
    return copy.deepcopy(dict(reduction))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise BeliefProjectionError(f"{label} must be an existing absolute regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BeliefProjectionError(f"{label} contains malformed JSON") from exc
    if not isinstance(payload, dict):
        raise BeliefProjectionError(f"{label} must contain one JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise BeliefProjectionError("output must be a new absolute file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(
                dict(payload), indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--prefilter-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    panel_path = Path(args.panel)
    filter_path = Path(args.prefilter_contract)
    manifest = _load_json(manifest_path, "manifest")
    panel = _load_json(panel_path, "panel")
    contract = _load_json(filter_path, "prefilter contract")
    validated_manifest = runner.validate_manifest(manifest)
    validated_panel = prefilter.validate_panel(panel, manifest=validated_manifest)
    prefilter.verify_panel_evidence_files(validated_panel, panel_path)
    if _file_sha(filter_path) != validated_manifest["prefilter"]["sha256"]:
        raise BeliefProjectionError(
            "execution manifest does not pin the exact prefilter contract file bytes")
    reduction = reduce_with_beliefs(
        manifest=validated_manifest, panel=validated_panel,
        prefilter_contract=contract,
    )
    validate_reduction_with_beliefs(
        reduction, manifest=validated_manifest, panel=validated_panel,
        prefilter_contract=contract,
    )
    _atomic_json(Path(args.output), reduction)
    print(json.dumps(reduction, sort_keys=True))
    return 0


__all__ = [
    "AUTHORITY", "BeliefProjectionError", "PRODUCER_ID", "PRODUCER_REF",
    "REPS_BASIS", "reduce_with_beliefs", "validate_reduction_with_beliefs",
]


if __name__ == "__main__":
    raise SystemExit(main())
