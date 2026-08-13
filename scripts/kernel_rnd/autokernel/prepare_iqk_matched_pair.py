#!/usr/bin/env python3
"""Deterministically prepare a matched CPU-IQK intervention/control pair.

The input manifest names committed/prospective source receipts and two real,
proposal-bound held-out measurement receipts.  The producer derives the A/A
control, shared randomization seeds, physical frame, complete one-factor frame,
and v2 capture plans.  It creates two new output directories atomically and
never runs inference, builds, claims resources, or mutates a campaign journal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from . import campaign, least_commitment_capture as capture
from . import least_commitment_heldout, schemas
from .execution import physical_bounds


LEGACY_SCHEMA = "epyc.autokernel.iqk_matched_pair_preparation.v1"
SCHEMA = "epyc.autokernel.iqk_matched_pair_preparation.v2"
RESULT_SCHEMA = "epyc.autokernel.iqk_matched_pair_preparation_result.v2"
PREFILL_RECIPE_ID = "t1b.llama_cpu.llama_bench_prefill.v1"
DECODE_RECIPE_ID = "t1b.llama_cpu.llama_bench_decode.v1"
CANONICAL_FRAMES = {
    PREFILL_RECIPE_ID: {"n_prompt": 512, "shape": "pp512"},
    DECODE_RECIPE_ID: {"n_gen": 128, "shape": "tg128"},
}
HYPOTHESIS_STORE_FILENAME = "hypotheses.json"
HYPOTHESIS_STORE_SCHEMA = "epyc.autokernel.operator_hypotheses.v1"
HYPOTHESIS_FALSIFIER = (
    "The accepted paired run fails a required integrity gate or its median "
    "relative prefill gain does not exceed the predeclared 3% contribution floor."
)
DECODE_HYPOTHESIS_FALSIFIER = (
    "The accepted paired run fails a required integrity gate or its median "
    "relative decode gain does not exceed the predeclared 3% contribution floor."
)
AA_CONTROL_FALSIFIER = (
    "The accepted A/A control exceeds the predeclared drift bound or fails "
    "the required control-integrity gates."
)

# The live A/A calibration and the ranked CPU-IQK pair are one comparison
# frame.  r3 established that five repetitions cannot resolve the predeclared
# 3% contribution floor inside the 20-block ceiling, while the same instrument
# at one repetition had already supplied usable A/A evidence.  Do not allow a
# copied manifest to silently select a different repetition regime.
IQK_MATCHED_PAIR_REPS = campaign.IQK_MATCHED_PAIR_REPS


class PreparationError(ValueError):
    pass


def _measurement_frame(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Load one closed canonical recipe frame, retaining v1 prefill inputs.

    A v1 manifest predates recipe selection and therefore means exactly the
    historical pp512 cell.  V2 makes the recipe and its work dimension
    explicit.  In particular, decode cannot inherit ``n_prompt`` or a prefill
    calibration merely because both recipes use the same llama-bench binary.
    """
    if raw.get("schema") == LEGACY_SCHEMA:
        return {"recipe_id": PREFILL_RECIPE_ID, "n_prompt": 512,
                "shape": "pp512"}
    frame = raw.get("measurement_frame")
    if not isinstance(frame, Mapping):
        raise PreparationError("measurement_frame must be an object")
    recipe_id = frame.get("recipe_id")
    canonical = CANONICAL_FRAMES.get(recipe_id)
    if canonical is None:
        raise PreparationError(
            "measurement_frame.recipe_id must name the canonical CPU prefill "
            "or decode recipe")
    expected = {"recipe_id", next(key for key in canonical if key != "shape")}
    if set(frame) != expected:
        raise PreparationError(
            f"measurement_frame fields for {recipe_id} must be exactly "
            f"{sorted(expected)}")
    token_key = next(key for key in canonical if key != "shape")
    if frame[token_key] != canonical[token_key]:
        raise PreparationError(
            f"measurement_frame.{token_key} must select the canonical "
            f"{canonical['shape']} cell ({canonical[token_key]})")
    return {"recipe_id": recipe_id, token_key: canonical[token_key],
            "shape": canonical["shape"]}


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{label}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{label}: expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644)
    try:
        written = 0
        while written < len(body):
            count = os.write(descriptor, body[written:])
            if count <= 0:
                raise OSError(f"short write while publishing {path}")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(path: Path, label: str) -> dict[str, str]:
    if not path.is_dir() or path.is_symlink():
        raise PreparationError(f"{label}: expected a non-symlink directory")
    files: dict[str, str] = {}
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise PreparationError(f"{label}: symlink input is refused: {child}")
        if child.is_file():
            files[str(child.relative_to(path))] = _sha256(child)
    if not files:
        raise PreparationError(f"{label}: input directory contains no files")
    return files


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise PreparationError(f"{label}: expected non-empty text without NUL")
    return value


def _path(value: Any, label: str, *, file: bool = False) -> Path:
    path = Path(_text(value, label))
    if not path.is_absolute():
        raise PreparationError(f"{label}: path must be absolute")
    if file and (path.is_symlink() or not path.is_file()):
        raise PreparationError(f"{label}: expected an existing non-symlink file")
    return path


def _branch(raw: Any, label: str) -> dict[str, Any]:
    required = {
        "campaign_id", "candidate_id", "capture_id", "intervention_id",
        "diagnostic_source", "heldout_outcome", "evidence_stage", "output_dir",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise PreparationError(f"{label}: fields must be exactly {sorted(required)}")
    result = {key: _text(raw[key], f"{label}.{key}") for key in (
        "campaign_id", "candidate_id", "capture_id", "intervention_id")}
    result["diagnostic_source"] = _path(
        raw["diagnostic_source"], f"{label}.diagnostic_source", file=True)
    result["evidence_stage"] = _text(raw["evidence_stage"], f"{label}.evidence_stage")
    if result["evidence_stage"] not in capture.EVIDENCE_STAGES:
        raise PreparationError(
            f"{label}.evidence_stage must be one of {sorted(capture.EVIDENCE_STAGES)}")
    heldout = raw["heldout_outcome"]
    if result["evidence_stage"] == "bootstrap":
        if heldout is not None:
            raise PreparationError(f"{label}.bootstrap cannot name heldout_outcome")
        result["heldout_outcome"] = None
    else:
        result["heldout_outcome"] = _path(
            heldout, f"{label}.heldout_outcome", file=True)
    result["output_dir"] = _path(raw["output_dir"], f"{label}.output_dir")
    return result


def _matched_envelope(
        template: physical_bounds.PhysicalEnvelope, *,
        spec: campaign.CampaignSpec, matched_experiment_id: str,
        calibration_path: Path, template_path: Path,
        calibration_cell_conversion: bool,
) -> tuple[physical_bounds.PhysicalEnvelope, dict[str, Any]]:
    """Convert a verified calibration-cell envelope to one campaign unit.

    Calibration labels its raw control cell (for example
    ``model.gguf:pp512:aa_calibration``), while CampaignSpec requires its
    canonical ranked-unit identity (``recipe_id:/absolute/model``).  Only those
    two identity fields are converted.  All physical facts remain byte-value
    equal and the result carries a self-hashed conversion receipt.
    """
    raw = template.to_dict()
    source = dict(raw)
    if not calibration_cell_conversion and raw["shape_id"] != spec.measurement_unit_id:
        raise PreparationError(
            "legacy physical envelope shape_id must already equal the canonical "
            "campaign measurement unit")
    raw["shape_id"] = spec.measurement_unit_id
    raw["measurement_frame_sha256"] = physical_bounds.measurement_frame_sha256(
        spec.recipe_id, spec.bench_params_for(matched_experiment_id))
    converted = physical_bounds.PhysicalEnvelope.from_mapping(raw)
    invariant = dict(raw)
    invariant.pop("shape_id")
    invariant.pop("measurement_frame_sha256")
    receipt: dict[str, Any] = {
        "schema": "epyc.autokernel.physical_envelope_conversion.v1",
        "calibration_bundle": str(calibration_path),
        "source_template": str(template_path),
        "source_file_sha256": _sha256(template_path),
        "source_envelope_sha256": schemas.content_hash(source),
        "source_shape_id": source["shape_id"],
        "source_measurement_frame_sha256": source["measurement_frame_sha256"],
        "destination_shape_id": converted.shape_id,
        "destination_measurement_frame_sha256": converted.measurement_frame_sha256,
        "recipe_id": spec.recipe_id,
        "matched_experiment_id": matched_experiment_id,
        "invariant_physical_facts_sha256": schemas.content_hash(invariant),
        "converted_fields": [
            name for name in ("measurement_frame_sha256", "shape_id")
            if source[name] != raw[name]],
    }
    receipt["receipt_sha256"] = schemas.content_hash(receipt)
    return converted, receipt


def _validate_calibration_envelope(
        calibration_path: Path, template: physical_bounds.PhysicalEnvelope,
        measurement_frame: Mapping[str, Any], model: str) -> None:
    """Require v2 physical facts from the selected calibration's exact cell.

    The matched schedule changes the measurement-frame hash, but it does not
    license changing the work derivation, delivered unit, physical ceilings or
    shape.  Without this join, a copied pp512 envelope could be relabelled as
    decode by replacing only its frame digest.
    """
    declaration = _load(
        calibration_path / "campaign_declaration.json",
        "calibration campaign declaration")
    envelopes = declaration.get("physical_envelopes")
    committed = envelopes.get("aa_calibration") if isinstance(envelopes, Mapping) else None
    if not isinstance(committed, Mapping):
        raise PreparationError(
            "v2 calibration declaration lacks the committed-cell physical envelope")
    try:
        expected = physical_bounds.PhysicalEnvelope.from_mapping(committed)
    except (TypeError, ValueError) as exc:
        raise PreparationError(
            f"calibration committed-cell physical envelope is invalid: {exc}") from exc
    canonical_source_shape = (
        f"{Path(model).name}:{measurement_frame['shape']}:aa_calibration")
    if expected.shape_id != canonical_source_shape:
        raise PreparationError(
            "calibration committed-cell physical envelope has a noncanonical "
            f"shape_id: {expected.shape_id!r} != {canonical_source_shape!r}")
    selected = template.to_dict()
    declared = expected.to_dict()
    selected.pop("measurement_frame_sha256", None)
    declared.pop("measurement_frame_sha256", None)
    if selected != declared:
        raise PreparationError(
            f"physical envelope template does not match the calibration's exact "
            f"{measurement_frame['shape']} committed cell")


def _rebind_provider_reference(
        proposal: dict[str, Any], calibration_path: Path) -> None:
    """Bind the proposal provider to the accepted calibration instrument.

    Proposal templates are routinely copied between campaign eras.  Provider
    identity is therefore never trusted from that template: all four
    load-bearing hashes are taken from the calibration anchor, and a missing
    anchor field refuses preparation.
    """
    source_path = calibration_path / "runtime-source-label.json"
    source = _load(source_path, "calibration runtime source label")
    required = ("measurement_instrument_commit", "measurement_binary_sha256",
                "measurement_linkage_sha256", "measurement_toolchain_manifest_sha256")
    missing = [key for key in required if not source.get(key)]
    if missing:
        raise PreparationError(
            "calibration anchor lacks provider evidence: " + ", ".join(missing))
    if source["measurement_instrument_commit"] != campaign.MEASUREMENT_COMMIT:
        raise PreparationError("calibration anchor measurement commit is not current")
    provider = proposal.get("provider_reference")
    if not isinstance(provider, dict):
        raise PreparationError("proposal provider_reference must be an object")
    provider.update({
        "source_ref": campaign.MEASUREMENT_REPO,
        "source_commit": campaign.MEASUREMENT_COMMIT,
        "artifact_sha256": source["measurement_binary_sha256"],
        "linkage_manifest_sha256": source["measurement_linkage_sha256"],
        "toolchain_manifest_sha256": source["measurement_toolchain_manifest_sha256"],
    })


def _copy_bound_receipt(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copyfile(source, destination)
    return capture.source_binding(destination)


def _build_plan(
        *, proposal: Mapping[str, Any], branch: Mapping[str, Any], role: str,
        matched_experiment_id: str, matched_control_proposal_id: str | None,
        factors: Mapping[str, Any], staging: Path, published: Path,
) -> dict[str, Any]:
    diagnostic_path = staging / "least-commitment-diagnostic-source.json"
    heldout_path = staging / "least-commitment-heldout-outcome.json"
    _copy_bound_receipt(Path(branch["diagnostic_source"]), diagnostic_path)
    source = _load(diagnostic_path, f"{role} diagnostic source")
    expected_frame = least_commitment_heldout.candidate_frame_id(
        least_commitment_heldout.candidate_frame_from_factors(factors, proposal))
    # The source receipt owns quotient semantics, while the candidate frame is
    # mechanically derived from this exact campaign.  Rebind it in the private
    # transaction so stale preparation artifacts cannot select a frame by hand.
    source["candidate_frame_id"] = expected_frame
    source["proposal_sha256"] = schemas.content_hash(proposal)
    _write_json(diagnostic_path, source)
    diagnostic_binding = capture.source_binding(diagnostic_path)
    heldout_binding = None
    if branch["evidence_stage"] == "heldout_bound":
        assert branch["heldout_outcome"] is not None
        heldout_binding = _copy_bound_receipt(
            Path(branch["heldout_outcome"]), heldout_path)
    diagnostics, recodings = capture.derive_diagnostics(
        source, proposal=proposal,
        candidate_frame_id=expected_frame)
    raw: dict[str, Any] = {
        "schema": capture.SCHEMA,
        "capture_id": branch["capture_id"],
        "campaign_id": branch["campaign_id"],
        "candidate_id": branch["candidate_id"],
        "proposal_id": proposal["proposal_id"],
        "matched_experiment_id": matched_experiment_id,
        "role": role,
        "matched_control_proposal_id": matched_control_proposal_id,
        "candidate_frame_id": expected_frame,
        "regime": proposal["target"]["regimes"][0],
        "surface": proposal["target"]["ops"][0],
        "intervention_id": branch["intervention_id"],
        "changed_factor": "ggml_iqk",
        "factors": json.loads(schemas.canonical_json(factors)),
        "diagnostics": diagnostics,
        "recodings": recodings,
        "diagnostic_source_receipts": {
            name: dict(diagnostic_binding) for name in capture.DIAGNOSTICS},
        "evidence_stage": branch["evidence_stage"],
        "heldout_outcome_receipt": heldout_binding,
        "outcome_reducers": dict(
            capture.BOOTSTRAP_OUTCOME_REDUCERS
            if branch["evidence_stage"] == "bootstrap" else capture.OUTCOME_REDUCERS),
        "capture_mode": "measured",
    }
    raw["plan_sha256"] = capture.plan_sha256(raw)
    plan = capture.from_mapping(
        raw, proposal=proposal, campaign_id=str(branch["campaign_id"]),
        candidate_id=str(branch["candidate_id"]))
    # The source files are validated while they are private in staging, but the
    # durable plan must name their post-rename paths.  Rebind only the paths;
    # receipt identities and byte hashes remain unchanged.  A second complete
    # validation runs after both directories are published and rolls them back
    # together on any discrepancy.
    durable = copy.deepcopy(plan.raw)
    for binding in durable["diagnostic_source_receipts"].values():
        binding["path"] = str(
            published / "least-commitment-diagnostic-source.json")
    if durable["heldout_outcome_receipt"] is not None:
        durable["heldout_outcome_receipt"]["path"] = str(
            published / "least-commitment-heldout-outcome.json")
    durable["plan_sha256"] = capture.plan_sha256(durable)
    _write_json(staging / "least-commitment-capture-plan.json", durable)
    return durable


def _base_spec(
        *, proposal: Mapping[str, Any], branch: Mapping[str, Any], model: str,
        calibration: campaign.LeanCalibration, blocks: int, reps: int,
        measurement_frame: Mapping[str, Any],
) -> campaign.CampaignSpec:
    recipe_id = str(measurement_frame["recipe_id"])
    return campaign.CampaignSpec(
        campaign_id=str(branch["campaign_id"]),
        candidate_id=str(branch["candidate_id"]),
        candidate_ref="registered:ggml_iqk", model=model,
        backend=campaign.BACKEND_CPU,
        recipe_id=recipe_id,
        n_prompt=int(measurement_frame.get("n_prompt", 512)),
        n_gen=int(measurement_frame.get("n_gen", 128)),
        blocks=blocks, reps=reps, proposal=proposal,
        calibration=calibration)


def _hypothesis_store(*, proposal: Mapping[str, Any], candidate_id: str,
                      role: str, measurement_frame: Mapping[str, Any]) -> dict[str, Any]:
    """Build the campaign-local operator store required by ``--hypothesis``.

    Pair preparation owns fresh candidate identities, so the ordinal in the
    candidate is the only stable way to bind the corresponding operator entry.
    The statement remains proposal-owned; preparation supplies only the durable
    falsifier and the regime used by the v9 IQK campaign.
    """
    if role not in {"intervention", "control"}:
        raise PreparationError(f"unsupported hypothesis binding role: {role}")
    suffix = candidate_id.rsplit("-", 1)[-1]
    stem = "aa-control" if role == "control" else "known-real"
    hypothesis_id = f"akh-iqk-v9-{stem}-{suffix}"
    return {
        "schema": HYPOTHESIS_STORE_SCHEMA,
        "hypotheses": [{
            "author": "operator",
            "created_at": "2026-08-12T00:00:00+00:00",
            "falsifier": (
                AA_CONTROL_FALSIFIER if role == "control" else
                DECODE_HYPOTHESIS_FALSIFIER
                if measurement_frame["recipe_id"] == DECODE_RECIPE_ID else
                HYPOTHESIS_FALSIFIER),
            "hypothesis_id": hypothesis_id,
            "regime": {
                "backend": "llama_cpu",
                "model": "Qwen2.5-Coder-0.5B-Q4_K_M",
                "recipe_id": measurement_frame["recipe_id"],
                "shape": measurement_frame["shape"],
            },
            "statement": proposal["hypothesis"],
        }],
    }


def prepare(raw: Mapping[str, Any]) -> dict[str, Any]:
    common_fields = {
        "schema", "matched_experiment_id", "model", "calibration_bundle",
        "physical_envelope_template", "intervention_proposal",
        "intervention_campaign_id", "intervention_proposal_id",
        "control_proposal_id", "intervention", "control", "blocks", "reps",
    }
    schema = raw.get("schema")
    required = (common_fields if schema == LEGACY_SCHEMA
                else common_fields | {"measurement_frame"})
    if schema not in {LEGACY_SCHEMA, SCHEMA} or set(raw) != required:
        raise PreparationError(
            f"manifest fields/schema must be legacy {sorted(common_fields)} or "
            f"v2 {sorted(common_fields | {'measurement_frame'})}")
    measurement_frame = _measurement_frame(raw)
    matched_id = _text(raw["matched_experiment_id"], "matched_experiment_id")
    if not matched_id.startswith("akm-"):
        raise PreparationError("matched_experiment_id must start with 'akm-'")
    model = str(_path(raw["model"], "model", file=True))
    calibration_path = _path(raw["calibration_bundle"], "calibration_bundle")
    template_path = _path(
        raw["physical_envelope_template"], "physical_envelope_template", file=True)
    proposal_path = _path(
        raw["intervention_proposal"], "intervention_proposal", file=True)
    intervention = _branch(raw["intervention"], "intervention")
    control_branch = _branch(raw["control"], "control")
    if intervention["output_dir"] == control_branch["output_dir"]:
        raise PreparationError("intervention and control output directories must differ")
    if Path(intervention["output_dir"]).parent != \
            Path(control_branch["output_dir"]).parent:
        raise PreparationError(
            "intervention and control outputs require one parent for atomic publication")
    for branch in (intervention, control_branch):
        target = Path(branch["output_dir"])
        if target.exists() or target.is_symlink() or not target.parent.is_dir():
            raise PreparationError(
                f"output directory must be new under an existing parent: {target}")
    blocks, reps = raw["blocks"], raw["reps"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (blocks, reps)):
        raise PreparationError("blocks and reps must be positive integers")
    if reps != IQK_MATCHED_PAIR_REPS:
        raise PreparationError(
            "CPU-IQK matched pairs require "
            f"reps={IQK_MATCHED_PAIR_REPS}; got reps={reps}")
    proposal = _load(proposal_path, "intervention proposal")
    # The source proposal is an immutable semantic template. Each real campaign
    # gets fresh identities before every receipt is derived and bound.
    proposal["campaign_id"] = _text(
        raw["intervention_campaign_id"], "intervention_campaign_id")
    proposal["proposal_id"] = _text(
        raw["intervention_proposal_id"], "intervention_proposal_id")
    calibration = campaign.load_calibration_bundle(calibration_path)
    if calibration.recipe_id != measurement_frame["recipe_id"]:
        raise PreparationError(
            f"calibration recipe {calibration.recipe_id!r} does not match "
            f"measurement frame {measurement_frame['recipe_id']!r}")
    expected_regime = "decode" if measurement_frame["recipe_id"] == DECODE_RECIPE_ID \
        else "prefill"
    target = proposal.get("target")
    if not isinstance(target, Mapping) or target.get("regimes") != [expected_regime]:
        raise PreparationError(
            f"intervention proposal must target exactly the {expected_regime!r} regime")
    if measurement_frame["recipe_id"] == DECODE_RECIPE_ID:
        if target.get("shapes") != [measurement_frame["shape"]]:
            raise PreparationError(
                "decode intervention proposal must target exactly shape 'tg128'")
        if any(branch["evidence_stage"] != "bootstrap"
               for branch in (intervention, control_branch)):
            raise PreparationError(
                "the distinct-regime decode pair must use bootstrap evidence; "
                "it produces held-out receipts for a later target campaign")
    # Preparation is the last entirely non-executing boundary before these
    # immutable inputs become campaign roots.  Do not publish a pair which the
    # campaign's accepted cell-local calibration will later refuse: that would
    # leave apparently ready artifacts which can never be measured.
    if not calibration.b_min_blocks <= blocks <= calibration.max_blocks:
        raise PreparationError(
            f"blocks={blocks} is outside the accepted calibration range "
            f"[{calibration.b_min_blocks}, {calibration.max_blocks}] from "
            f"{calibration.evidence_ref}")
    _rebind_provider_reference(proposal, calibration_path)
    violations = schemas.validate_proposal(proposal)
    if violations:
        raise PreparationError("invalid intervention proposal: " + "; ".join(violations))
    if proposal.get("campaign_id") != intervention["campaign_id"]:
        raise PreparationError("intervention proposal campaign_id differs")
    control_proposal_id = _text(raw["control_proposal_id"], "control_proposal_id")
    control = capture.make_iqk_control_proposal(
        proposal, campaign_id=str(control_branch["campaign_id"]),
        proposal_id=control_proposal_id)
    template = physical_bounds.PhysicalEnvelope.from_mapping(
        _load(template_path, "physical envelope template"))
    if schema == SCHEMA:
        _validate_calibration_envelope(
            calibration_path, template, measurement_frame, model)
    # The A/A control is intentionally uninstantiable without its typed control
    # plan. Derive the common frame once from the intervention, then change the
    # one licensed factor before constructing both final, fully governed specs.
    base = _base_spec(
        proposal=proposal, branch=intervention, model=model,
        calibration=calibration, blocks=blocks, reps=reps,
        measurement_frame=measurement_frame)
    envelope, envelope_conversion = _matched_envelope(
        template, spec=base, matched_experiment_id=matched_id,
        calibration_path=calibration_path, template_path=template_path,
        calibration_cell_conversion=schema == SCHEMA)
    intervention_factors = base.matched_factor_frame_for(
        matched_id, physical_envelope=envelope)
    control_factors = copy.deepcopy(intervention_factors)
    control_factors["ggml_iqk"] = "0"
    envelopes = {"intervention": envelope, "control": envelope}
    factors = {
        "intervention": intervention_factors, "control": control_factors}
    changed = [key for key in factors["intervention"]
               if factors["intervention"][key] != factors["control"][key]]
    if changed != ["ggml_iqk"]:
        raise PreparationError(
            f"derived pair must differ only on ggml_iqk; changed={changed}")

    staging_parent = Path(intervention["output_dir"]).parent
    staging_root = Path(tempfile.mkdtemp(prefix=".ak-iqk-pair-", dir=staging_parent))
    published: list[Path] = []
    try:
        staged = {name: staging_root / name for name in ("intervention", "control")}
        destinations = {
            "intervention": Path(intervention["output_dir"]),
            "control": Path(control_branch["output_dir"]),
        }
        for path in staged.values():
            path.mkdir(mode=0o700)
        _write_json(staged["intervention"] / "proposal-v4.json", proposal)
        _write_json(staged["control"] / "proposal-v4.json", control)
        # Both pair roots are independently consumable campaign roots.  Keep a
        # local copy of the exact operator store so --hypothesis never depends
        # on an external path that preparation did not publish.
        for name, selected_proposal, branch in (
                ("intervention", proposal, intervention),
                ("control", control, control_branch)):
            _write_json(
                staged[name] / HYPOTHESIS_STORE_FILENAME,
                _hypothesis_store(
                    proposal=selected_proposal,
                    candidate_id=str(branch["candidate_id"]), role=name,
                    measurement_frame=measurement_frame))
        plans = {
            "intervention": _build_plan(
                proposal=proposal, branch=intervention, role="intervention",
                matched_experiment_id=matched_id,
                matched_control_proposal_id=control_proposal_id,
                factors=factors["intervention"], staging=staged["intervention"],
                published=destinations["intervention"]),
            "control": _build_plan(
                proposal=control, branch=control_branch, role="control",
                matched_experiment_id=matched_id,
                matched_control_proposal_id=None,
                factors=factors["control"], staging=staged["control"],
                published=destinations["control"]),
        }
        for name in ("intervention", "control"):
            _write_json(staged[name] / "physical-envelope.json",
                        envelopes[name].to_dict())
        for name in ("intervention", "control"):
            os.replace(staged[name], destinations[name])
            published.append(destinations[name])
        # Validate the durable absolute bindings after publication.  The two
        # output directories are transaction-owned and are both removed by the
        # exception path if either final campaign fails admission.
        parsed_plans: dict[str, capture.CapturePlan] = {}
        for name in ("intervention", "control"):
            branch = intervention if name == "intervention" else control_branch
            selected_proposal = proposal if name == "intervention" else control
            parsed_plans[name] = capture.from_mapping(
                plans[name], proposal=selected_proposal,
                campaign_id=str(branch["campaign_id"]),
                candidate_id=str(branch["candidate_id"]))
            final = campaign.CampaignSpec(
                campaign_id=str(branch["campaign_id"]),
                candidate_id=str(branch["candidate_id"]),
                candidate_ref="registered:ggml_iqk", model=model,
                backend=campaign.BACKEND_CPU,
                recipe_id=str(measurement_frame["recipe_id"]),
                n_prompt=int(measurement_frame.get("n_prompt", 512)),
                n_gen=int(measurement_frame.get("n_gen", 128)),
                blocks=blocks, reps=reps, proposal=selected_proposal,
                calibration=calibration,
                least_commitment_plan=parsed_plans[name],
                matched_experiment_id=matched_id,
                physical_envelope=envelopes[name])
            if final.matched_factor_frame != factors[name]:
                raise PreparationError(f"{name}: final campaign frame drifted")
        result = {
            "schema": RESULT_SCHEMA,
            "matched_experiment_id": matched_id,
            "measurement_frame": dict(measurement_frame),
            "sole_changed_factor": "ggml_iqk",
            "physical_envelope_conversion": envelope_conversion,
            "input_manifest_sha256": schemas.content_hash(raw),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "input_sources": {
                "model": _sha256(Path(model)),
                "calibration_bundle": _tree_hashes(
                    calibration_path, "calibration_bundle"),
                "physical_envelope_template": _sha256(template_path),
                "intervention_proposal": _sha256(proposal_path),
                "intervention_diagnostic_source": _sha256(
                    Path(intervention["diagnostic_source"])),
                "intervention_heldout_outcome": (
                    None if intervention["heldout_outcome"] is None else _sha256(
                        Path(intervention["heldout_outcome"]))),
                "control_diagnostic_source": _sha256(
                    Path(control_branch["diagnostic_source"])),
                "control_heldout_outcome": (
                    None if control_branch["heldout_outcome"] is None else _sha256(
                        Path(control_branch["heldout_outcome"]))),
            },
            "outputs": {name: {
                "path": str(path),
                "files": {child.name: _sha256(child)
                          for child in sorted(path.iterdir()) if child.is_file()},
            } for name, path in destinations.items()},
            "inference_started": False,
            "campaign_executed": False,
        }
        return {**result, "result_sha256": schemas.content_hash(result)}
    except BaseException:
        for path in published:
            if path.exists() and path.parent == staging_parent:
                shutil.rmtree(path)
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.result.is_absolute() or args.result.exists() \
            or not args.result.parent.is_dir():
        raise PreparationError("--result must be a new absolute path")
    result = prepare(_load(args.manifest.resolve(), "preparation manifest"))
    _write_json_exclusive(args.result, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
