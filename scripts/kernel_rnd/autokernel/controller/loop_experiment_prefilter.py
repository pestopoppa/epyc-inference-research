#!/usr/bin/env python3
"""Deterministic external structural prefilter and AK-LE-1/2 reducer.

The prefilter makes no semantic quality judgment.  It admits structurally valid
model output (already enforced by ``loop_experiment_runner``), rejects exact
normalized fingerprints present in the predeclared prior set, and rejects every
repeated occurrence after the first within one planner cell.  It never asks a
model or operator to label a result.

A raw panel may be reduced only when its execution manifest pins the exact bytes
of a versioned prefilter contract produced by this module.  A pin to executable
source alone is insufficient: source does not bind the prior set, duplicate
scope, or algorithm.  That distinction intentionally leaves panels captured
under an under-specified pin as raw evidence rather than retroactively choosing
their filter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from . import loop_experiment_runner as runner
from . import loop_experiments as experiments


PREFILTER_SCHEMA = "epyc.autokernel.planner_structural_prefilter_contract.v1"
EVIDENCE_SCHEMA = "epyc.autokernel.planner_structural_prefilter_evidence.v1"
REDUCTION_SCHEMA = "epyc.autokernel.loop_experiment_planner_reduction.v1"
REFUSAL_SCHEMA = "epyc.autokernel.loop_experiment_planner_reduction_refusal.v1"
ALGORITHM = "normalized_fingerprint_prior_and_per_cell_first_occurrence.v1"
AUTHORITY = "observe_only_no_campaign_ranking_champion_or_release_authority"
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class PrefilterError(ValueError):
    """Reduction input is mutable, incomplete, under-specified, or unbound."""


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise PrefilterError(f"{label} must be a lowercase SHA-256")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise PrefilterError("output must be a new absolute file")
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


def _atomic_canonical(path: Path, payload: Mapping[str, Any]) -> None:
    """Write bytes whose SHA is the mapping's canonical digest."""
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise PrefilterError("output must be a new absolute file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical(dict(payload)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _producer_pin() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "ref": (
            "git://epyc-inference-research/scripts/kernel_rnd/autokernel/"
            "controller/loop_experiment_prefilter.py"),
        "sha256": _file_sha(path),
    }


def compile_prefilter_contract(
    *, prior_hypothesis_sha256: Sequence[str],
) -> dict[str, Any]:
    """Compile the smallest runnable structural prefilter for a future panel."""
    prior = tuple(_sha(value, "prior hypothesis SHA-256")
                  for value in prior_hypothesis_sha256)
    if prior != tuple(sorted(set(prior))):
        raise PrefilterError(
            "prior hypothesis SHA-256 values must be sorted and duplicate-free")
    payload: dict[str, Any] = {
        "schema": PREFILTER_SCHEMA,
        "authority": AUTHORITY,
        "algorithm": ALGORITHM,
        "producer": _producer_pin(),
        "prior_hypothesis_sha256": list(prior),
        "semantics": {
            "fingerprint": (
                "sha256(canonical JSON of casefolded whitespace-normalized "
                "mechanism,target_surface,falsifiable_counter,predicted_direction)"),
            "admissibility": (
                "runner-v1 strict four-field non-empty structural observation"),
            "prior_match": "exact fingerprint equality",
            "duplicate_scope": "within_cell",
            "duplicate_policy": "first_occurrence_survives",
            "cross_cell_duplicates": "retained_for_matched_arm_independence",
            "semantic_quality_label": "not_performed",
            "campaign_do_not_repeat_gate": "not_replaced_or_invoked",
        },
        "constraints": {
            "model_label_requested": False,
            "operator_label_requested": False,
            "campaign_1_authority": False,
            "ranking_authority": False,
            "champion_authority": False,
            "release_authority": False,
        },
    }
    payload["contract_sha256"] = _digest(payload)
    return payload


def compile_bound_planner_manifest(
    contract: experiments.ExperimentContract, *,
    prefilter_contract_path: str | Path,
    context: Any,
    target_lines: Mapping[str, str],
    model_pins: Sequence[runner.ModelCellPin],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Compile a panel only after independently persisted filter bytes verify.

    The prefilter contract must already exist; this function never creates or
    rewrites it.  Thus the experiment's :class:`ArtifactPin` necessarily precedes
    execution-manifest compilation rather than being retrofitted after capture.
    """
    if not isinstance(contract, experiments.ExperimentContract):
        raise TypeError("contract must be an ExperimentContract")
    path = Path(prefilter_contract_path)
    if (not path.is_absolute() or not path.is_file() or path.is_symlink()
            or path.resolve() != path):
        raise PrefilterError(
            "prefilter contract must be an existing canonical absolute file")
    if contract.prefilter.ref != str(path) or contract.prefilter.sha256 != _file_sha(path):
        raise PrefilterError(
            "experiment contract does not pin the persisted prefilter contract bytes")
    payload = _load_json(path, "prefilter contract")
    validated = validate_prefilter_contract(
        payload,
        expected_prior_hypothesis_sha256=contract.prior_hypothesis_sha256)
    if validated["producer"] != _producer_pin():
        raise PrefilterError(
            "current reducer source identity differs from the execution pin")
    return runner.compile_planner_manifest(
        contract, context=context, target_lines=target_lines,
        model_pins=model_pins, timeout_seconds=timeout_seconds)


def validate_prefilter_contract(
    contract: Mapping[str, Any], *, expected_prior_hypothesis_sha256: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise PrefilterError("prefilter contract must be an object")
    payload = dict(contract)
    claimed = payload.pop("contract_sha256", None)
    _sha(claimed, "prefilter contract SHA-256")
    if _digest(payload) != claimed:
        raise PrefilterError("prefilter contract SHA-256 does not verify")
    if set(payload) != {
            "schema", "authority", "algorithm", "producer",
            "prior_hypothesis_sha256", "semantics", "constraints"}:
        raise PrefilterError("prefilter contract has unknown or missing fields")
    if (payload.get("schema") != PREFILTER_SCHEMA
            or payload.get("authority") != AUTHORITY
            or payload.get("algorithm") != ALGORITHM):
        raise PrefilterError("prefilter contract is not the runnable structural v1 contract")
    producer = payload.get("producer")
    if not isinstance(producer, dict) or set(producer) != {"ref", "sha256"}:
        raise PrefilterError("prefilter producer pin is malformed")
    _sha(producer.get("sha256"), "prefilter producer SHA-256")
    if producer.get("ref") != _producer_pin()["ref"]:
        raise PrefilterError("prefilter producer reference is not structural v1")
    expected = tuple(expected_prior_hypothesis_sha256)
    actual = payload.get("prior_hypothesis_sha256")
    if (not isinstance(actual, list) or any(
            not isinstance(value, str) or not _SHA_RE.fullmatch(value)
            for value in actual)
            or tuple(actual) != tuple(sorted(set(actual)))):
        raise PrefilterError("prefilter prior set is malformed")
    if tuple(actual) != expected:
        raise PrefilterError("prefilter prior set differs from the experiment contract")
    if payload.get("semantics") != compile_prefilter_contract(
            prior_hypothesis_sha256=expected)["semantics"]:
        raise PrefilterError("prefilter structural semantics drifted")
    constraints = payload.get("constraints")
    if (not isinstance(constraints, dict)
            or constraints != compile_prefilter_contract(
                prior_hypothesis_sha256=expected)["constraints"]):
        raise PrefilterError("prefilter requests forbidden authority")
    payload["contract_sha256"] = claimed
    return payload


def validate_panel(
    panel: Mapping[str, Any], *, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a complete raw panel against its exact execution manifest."""
    if not isinstance(panel, Mapping):
        raise PrefilterError("planner panel must be an object")
    payload = dict(panel)
    claimed = payload.pop("panel_sha256", None)
    _sha(claimed, "panel SHA-256")
    if _digest(payload) != claimed:
        raise PrefilterError("panel SHA-256 does not verify")
    if set(payload) != {
            "schema", "status", "authority", "experiment_id",
            "experiment_contract_sha256", "manifest_sha256", "capture_mode",
            "observations", "constraints", "next_required_step"}:
        raise PrefilterError("panel has unknown or missing fields")
    if (payload.get("schema") != runner.PANEL_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("authority") != runner.AUTHORITY
            or payload.get("capture_mode") != "measured_model_output"):
        raise PrefilterError("panel is not a complete measured planner panel")
    if (payload.get("manifest_sha256") != manifest.get("manifest_sha256")
            or payload.get("experiment_contract_sha256") != manifest.get(
                "experiment_contract_sha256")
            or payload.get("experiment_id") != manifest.get("experiment_id")):
        raise PrefilterError("panel identity differs from its execution manifest")
    constraints = payload.get("constraints")
    if (not isinstance(constraints, dict)
            or constraints.get("external_prefilter_applied") is not False
            or constraints.get("scaffold_observations_present") is not False
            or any(constraints.get(key) is not False for key in (
                "campaign_1_authority", "ranking_authority", "champion_authority",
                "release_authority"))):
        raise PrefilterError("panel already claims filtering, scaffolds, or authority")
    rows = payload.get("observations")
    if not isinstance(rows, list):
        raise PrefilterError("panel observations must be an array")
    expected_ids = [cell["cell_id"] for cell in manifest["cells"]]
    if [row.get("cell_id") if isinstance(row, dict) else None for row in rows] != expected_ids:
        raise PrefilterError("panel observations differ from manifest cell order")
    required = {
        "argv", "cell_id", "cli_executable_sha256", "effort",
        "elapsed_wall_seconds", "finished_at", "model_id", "observation",
        "observation_sha256", "prompt_sha256", "provider", "quant_id",
        "result_sha256", "returncode", "started_at", "status", "stderr_sha256",
        "stdout_sha256", "timed_out",
    }
    for cell, row in zip(manifest["cells"], rows):
        if not isinstance(row, dict) or set(row) != required:
            raise PrefilterError("panel observation has unknown or missing fields")
        if any(row.get(key) != cell.get(key) for key in (
                "cell_id", "provider", "model_id", "quant_id", "effort")):
            raise PrefilterError("panel observation cell identity drifted")
        if row.get("status") != "parsed" or row.get("returncode") != 0 \
                or row.get("timed_out") is not False:
            raise PrefilterError("panel observation is not a successful parsed capture")
        elapsed = row.get("elapsed_wall_seconds")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed <= 0):
            raise PrefilterError("panel elapsed wall time is invalid")
        for name in ("observation_sha256", "prompt_sha256", "result_sha256",
                     "stderr_sha256", "stdout_sha256", "cli_executable_sha256"):
            _sha(row.get(name), name)
        raw = row.get("observation")
        if not isinstance(raw, dict):
            raise PrefilterError("panel raw observation must be an object")
        # Reuse the execution boundary's exact parser, including strict field shape.
        trusted = runner.parse_raw_observation(
            json.dumps({key: value for key, value in raw.items() if key != "cell_id"}),
            provider=row["provider"], expected_cell_id=row["cell_id"])
        if trusted != raw:
            raise PrefilterError("panel embedded raw observation drifted")
    payload["panel_sha256"] = claimed
    return payload


def verify_panel_evidence_files(panel: Mapping[str, Any], panel_path: Path) -> None:
    """Verify each embedded observation against the runner's sealed file bytes."""
    root = panel_path.parent
    for ordinal, row in enumerate(panel["observations"], 1):
        path = root / f"{ordinal:04d}-{row['cell_id']}" / "observation.json"
        if not path.is_file() or path.is_symlink():
            raise PrefilterError(f"sealed observation file is missing: {path}")
        if _file_sha(path) != row["observation_sha256"]:
            raise PrefilterError(f"sealed observation file SHA-256 drifted: {path}")
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PrefilterError(f"sealed observation file is malformed: {path}") from exc
        if observed != row["observation"]:
            raise PrefilterError(f"sealed observation content drifted: {path}")


def _prefilter_cell(
    row: Mapping[str, Any], *, prior: frozenset[str], contract_sha256: str,
) -> tuple[experiments.PlannerObservation, dict[str, Any]]:
    seen: set[str] = set()
    decisions: list[dict[str, Any]] = []
    survived: list[bool] = []
    for ordinal, raw in enumerate(row["observation"]["hypotheses"]):
        hypothesis = experiments.HypothesisObservation(
            mechanism=raw["mechanism"], target_surface=raw["target_surface"],
            falsifiable_counter=raw["falsifiable_counter"],
            predicted_direction=raw["predicted_direction"],
            survived_prefilter=False)
        fingerprint = hypothesis.fingerprint
        if fingerprint in prior:
            decision = "rejected_exact_prior"
            passed = False
        elif fingerprint in seen:
            decision = "rejected_duplicate_in_cell"
            passed = False
        else:
            decision = "survived_structural_prefilter"
            passed = True
        decision_row = {
            "hypothesis_ordinal": ordinal,
            "hypothesis_fingerprint_sha256": fingerprint,
            "decision": decision,
            "survived_prefilter": passed,
        }
        decision_row["decision_sha256"] = _digest({
            "cell_id": row["cell_id"],
            "raw_observation_sha256": row["observation_sha256"],
            **decision_row,
        })
        decisions.append(decision_row)
        survived.append(passed)
        seen.add(fingerprint)
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "algorithm": ALGORITHM,
        "prefilter_contract_sha256": contract_sha256,
        "cell_id": row["cell_id"],
        "raw_observation_sha256": row["observation_sha256"],
        "decisions": decisions,
    }
    evidence["evidence_sha256"] = _digest(evidence)
    observation = runner.materialize_planner_observation(
        row["observation"], survived_prefilter=survived,
        elapsed_wall_seconds=row["elapsed_wall_seconds"],
        evidence_sha256=evidence["evidence_sha256"], provider=row["provider"])
    return observation, evidence


def reduce_planner_panel(
    *, manifest: Mapping[str, Any], panel: Mapping[str, Any],
    prefilter_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pinned filter and emit an authority-free planner-only receipt."""
    manifest_payload = runner.validate_manifest(manifest)
    panel_payload = validate_panel(panel, manifest=manifest_payload)
    embedded = manifest_payload["experiment_contract"]
    prior = tuple(embedded["prior_hypothesis_sha256"])
    filter_payload = validate_prefilter_contract(
        prefilter_contract, expected_prior_hypothesis_sha256=prior)
    if filter_payload["producer"] != _producer_pin():
        raise PrefilterError(
            "current reducer source identity differs from the execution pin")
    pinned = manifest_payload["prefilter"]
    # The API cannot observe file bytes, so it binds the contract's canonical bytes.
    # The CLI additionally checks the exact on-disk artifact bytes before calling us.
    if pinned.get("sha256") != _digest(filter_payload):
        raise PrefilterError(
            "execution manifest does not pin this runnable prefilter contract")

    observations: list[experiments.PlannerObservation] = []
    evidence: list[dict[str, Any]] = []
    for row in panel_payload["observations"]:
        materialized, bound = _prefilter_cell(
            row, prior=frozenset(prior),
            contract_sha256=filter_payload["contract_sha256"])
        observations.append(materialized)
        evidence.append(bound)

    # Reconstruct the immutable experiment object only through the already strict
    # manifest fields needed by the reducer.  Keeping this conversion local avoids a
    # permissive general "from arbitrary dict" parser in the experiment contract.
    contract = _contract_from_manifest(embedded)
    receipt = experiments.reduce_planner_receipt(
        contract, planner_observations=observations, capture_mode="measured")
    reduction: dict[str, Any] = {
        "schema": REDUCTION_SCHEMA,
        "authority": AUTHORITY,
        "manifest_sha256": manifest_payload["manifest_sha256"],
        "panel_sha256": panel_payload["panel_sha256"],
        "prefilter_contract_sha256": filter_payload["contract_sha256"],
        "prefilter_evidence": evidence,
        "planner_receipt": receipt,
        "constraints": {
            "raw_panel_mutated": False,
            "scaffold_observations_fabricated": False,
            "campaign_1_authority": False,
            "ranking_authority": False,
            "champion_authority": False,
            "release_authority": False,
        },
    }
    reduction["reduction_sha256"] = _digest(reduction)
    return reduction


def refuse_under_specified_panel(
    *, manifest: Mapping[str, Any], panel: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit a durable refusal for a valid panel lacking a runnable filter pin."""
    manifest_payload = runner.validate_manifest(manifest)
    panel_payload = validate_panel(panel, manifest=manifest_payload)
    prefilter = manifest_payload["prefilter"]
    ref = prefilter.get("ref") if isinstance(prefilter, dict) else None
    if not isinstance(ref, str):
        raise PrefilterError("manifest prefilter pin is malformed")
    reasons = [
        {
            "code": "runnable_prefilter_contract_not_pinned",
            "detail": (
                "the execution manifest pins source bytes rather than an independently "
                "persisted versioned prefilter contract binding algorithm and inputs"),
        },
        {
            "code": "prefilter_algorithm_and_inputs_not_bound",
            "detail": (
                "a source-file pin does not declare whether the experiment filter is "
                "exact-prior matching, within-cell deduplication, a do-not-repeat "
                "lookup, or another reviewed algorithm, and binds none of their inputs"),
        },
        {
            "code": "pinned_source_not_invocable_from_raw_observation",
            "detail": (
                "the pinned do_not_repeat.py API consumes a CompiledLedger plus regime "
                "and structural-target mapping, while the raw panel supplies only four "
                "planner hypothesis fields; therefore that source cannot be applied to "
                "this panel without inventing unbound inputs"),
        },
    ]
    receipt: dict[str, Any] = {
        "schema": REFUSAL_SCHEMA,
        "status": "refused",
        "authority": AUTHORITY,
        "experiment_id": manifest_payload["experiment_id"],
        "manifest_sha256": manifest_payload["manifest_sha256"],
        "panel_sha256": panel_payload["panel_sha256"],
        "prefilter_pin": dict(prefilter),
        "raw_hypothesis_count": sum(
            len(row["observation"]["hypotheses"])
            for row in panel_payload["observations"]),
        "reasons": reasons,
        "disposition": "raw_panel_retained_no_reduction_or_empirical_claim",
        "next_run_requirement": (
            "persist and pin epyc.autokernel.planner_structural_prefilter_contract.v1 "
            "before execution-manifest compilation; its experiment-only exact-prior/"
            "within-cell structural filter does not invoke, weaken, or replace the "
            "campaign do_not_repeat claim gate"),
        "constraints": {
            "prefilter_decisions_emitted": False,
            "planner_receipt_emitted": False,
            "raw_panel_mutated": False,
            "campaign_1_authority": False,
            "ranking_authority": False,
            "champion_authority": False,
            "release_authority": False,
        },
    }
    receipt["refusal_sha256"] = _digest(receipt)
    return receipt


def _contract_from_manifest(payload: Mapping[str, Any]) -> experiments.ExperimentContract:
    fixed = payload["fixed"]
    selected = fixed["selected_task"]
    return experiments.ExperimentContract(
        experiment_id=payload["experiment_id"],
        fixed=experiments.FixedPromptFrame(
            champion=experiments.ArtifactPin(**fixed["champion"]),
            retrieval_context_sha256=fixed["retrieval_context_sha256"],
            propose_prompt=fixed["propose_prompt"],
            propose_prompt_sha256=fixed["propose_prompt_sha256"],
            selected_task=experiments.SelectedTaskArtifact(**selected)),
        planner_arms=tuple(experiments.PlannerArm(**row)
                           for row in payload["planner_arms"]),
        predictions=tuple(experiments.DirectionPrediction(**row)
                          for row in payload["direction_predictions"]),
        scaffold_arms=tuple(experiments.ScaffoldArm(
            cell_id=row["cell_id"], model_id=row["model_id"],
            quant_id=row["quant_id"], effort=row["effort"],
            scaffold=row["scaffold"], roles=tuple(
                experiments.RoleBudget(
                    role=role["role"], wall_seconds=role["wall_seconds"],
                    instruction=role["instruction"],
                    instruction_sha256=role["instruction_sha256"])
                for role in row["roles"])) for row in payload["scaffold_arms"]),
        prior_hypothesis_sha256=tuple(payload["prior_hypothesis_sha256"]),
        prefilter=experiments.ArtifactPin(**payload["prefilter"]),
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PrefilterError(f"{label} must be an existing absolute regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PrefilterError(f"{label} contains malformed JSON") from exc
    if not isinstance(payload, dict):
        raise PrefilterError(f"{label} must contain one JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser(
        "compile", help="compile a structural prefilter contract for a future run")
    compile_parser.add_argument("--output", required=True)
    compile_parser.add_argument(
        "--prior-hypothesis-sha256", action="append", default=[])
    reduce_parser = subparsers.add_parser(
        "reduce", help="reduce a complete panel under its pinned prefilter")
    reduce_parser.add_argument("--manifest", required=True)
    reduce_parser.add_argument("--panel", required=True)
    reduce_parser.add_argument("--prefilter-contract", required=True)
    reduce_parser.add_argument("--output", required=True)
    refuse_parser = subparsers.add_parser(
        "refuse", help="seal refusal when a raw panel lacks a runnable filter pin")
    refuse_parser.add_argument("--manifest", required=True)
    refuse_parser.add_argument("--panel", required=True)
    refuse_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "compile":
        output = Path(args.output)
        contract = compile_prefilter_contract(
            prior_hypothesis_sha256=args.prior_hypothesis_sha256)
        # Canonical bytes make the ArtifactPin usable by both file and mapping APIs:
        # sha256(file bytes) == sha256(canonical contract mapping).
        _atomic_canonical(output, contract)
        print(json.dumps({
            "prefilter_contract": contract,
            "artifact_pin": {"ref": str(output), "sha256": _file_sha(output)},
        }, sort_keys=True))
        return 0

    manifest_path = Path(args.manifest)
    panel_path = Path(args.panel)
    manifest = _load_json(manifest_path, "manifest")
    panel = _load_json(panel_path, "panel")
    validated_manifest = runner.validate_manifest(manifest)
    validated_panel = validate_panel(panel, manifest=validated_manifest)
    verify_panel_evidence_files(validated_panel, panel_path)
    if args.command == "refuse":
        refusal = refuse_under_specified_panel(
            manifest=validated_manifest, panel=validated_panel)
        _atomic_json(Path(args.output), refusal)
        print(json.dumps(refusal, sort_keys=True))
        return 0

    filter_path = Path(args.prefilter_contract)
    prefilter = _load_json(filter_path, "prefilter contract")
    if _file_sha(filter_path) != validated_manifest["prefilter"]["sha256"]:
        raise PrefilterError(
            "execution manifest does not pin the exact prefilter contract file bytes")
    reduction = reduce_planner_panel(
        manifest=validated_manifest, panel=validated_panel,
        prefilter_contract=prefilter)
    _atomic_json(Path(args.output), reduction)
    print(json.dumps(reduction, sort_keys=True))
    return 0


__all__ = [
    "ALGORITHM", "AUTHORITY", "EVIDENCE_SCHEMA", "PREFILTER_SCHEMA",
    "REDUCTION_SCHEMA", "REFUSAL_SCHEMA", "PrefilterError",
    "compile_bound_planner_manifest", "compile_prefilter_contract",
    "reduce_planner_panel", "refuse_under_specified_panel", "validate_panel",
    "validate_prefilter_contract", "verify_panel_evidence_files",
]


if __name__ == "__main__":
    raise SystemExit(main())
