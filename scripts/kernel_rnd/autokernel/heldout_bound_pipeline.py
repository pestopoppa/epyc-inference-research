#!/usr/bin/env python3
"""Atomically bridge a completed decode pair to heldout-bound prefill + AP-WM.

``prepare`` refuses before creating either campaign directory unless both
distinct-regime decode journals are clean and can produce proposal-bound held-
out receipts.  It then delegates the atomic pair publication to
``prepare_iqk_matched_pair`` and emits exact dry-run/execute commands plus a
step-6 archive template.  ``archive`` discovers the two immutable DECIDED
terminals, projects their receipts, builds the archive, and writes a real-
labelled observe-only AP-WM report.  Neither mode runs inference.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import least_commitment_heldout as heldout
from . import least_commitment_receipts as receipts
from . import offline_least_commitment as offline
from . import prepare_iqk_matched_pair as pair
from . import schemas
from . import journal
from .least_commitment_capture import make_iqk_control_proposal


SCHEMA = "epyc.autokernel.heldout_bound_pipeline.v1"
RESULT_SCHEMA = "epyc.autokernel.heldout_bound_pipeline_result.v1"
ARCHIVE_RESULT_SCHEMA = "epyc.autokernel.heldout_bound_archive_result.v1"
_FIELDS = frozenset({
    "schema", "pair_manifest", "fixed_proposals", "heldout_measurements",
    "nominal_khz", "archive",
})
_MEASUREMENT_FIELDS = frozenset({"receipt_id", "measurement"})
_FIXED_PROPOSAL_FIELDS = frozenset({"path", "sha256"})
_ARCHIVE_FIELDS = frozenset({
    "archive_id", "created_at", "diagnostic_directions", "outcome_weights",
    "output_dir", "report_output",
})


class HeldoutPipelineError(ValueError):
    pass


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise HeldoutPipelineError(f"{label}: expected non-empty text without NUL")
    return value


def _absolute(value: Any, label: str) -> Path:
    path = Path(_text(value, label))
    if not path.is_absolute():
        raise HeldoutPipelineError(f"{label}: must be absolute")
    return path


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeldoutPipelineError(f"{label}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HeldoutPipelineError(f"{label}: expected a JSON object")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
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


def _target_proposals(raw_pair: Mapping[str, Any]) -> tuple[dict, dict]:
    proposal_path = _absolute(
        raw_pair.get("intervention_proposal"), "pair_manifest.intervention_proposal")
    proposal = _load(proposal_path, "intervention proposal")
    proposal["campaign_id"] = _text(
        raw_pair.get("intervention_campaign_id"), "intervention_campaign_id")
    proposal["proposal_id"] = _text(
        raw_pair.get("intervention_proposal_id"), "intervention_proposal_id")
    calibration = _absolute(raw_pair.get("calibration_bundle"), "calibration_bundle")
    pair._rebind_provider_reference(proposal, calibration)
    violations = schemas.validate_proposal(proposal)
    if violations:
        raise HeldoutPipelineError(
            "invalid fresh intervention proposal: " + "; ".join(violations))
    branch = raw_pair.get("control")
    if not isinstance(branch, Mapping):
        raise HeldoutPipelineError("pair_manifest.control is absent")
    control = make_iqk_control_proposal(
        proposal,
        campaign_id=_text(branch.get("campaign_id"), "control.campaign_id"),
        proposal_id=_text(raw_pair.get("control_proposal_id"), "control_proposal_id"),
    )
    return proposal, control


def _fixed_proposals(raw: Any, derived: tuple[dict, dict]) -> tuple[dict, dict]:
    if not isinstance(raw, Mapping) or set(raw) != {"intervention", "control"}:
        raise HeldoutPipelineError("fixed_proposals must contain both roles")
    loaded = []
    for role, expected in zip(("intervention", "control"), derived):
        binding = raw[role]
        if not isinstance(binding, Mapping) or set(binding) != _FIXED_PROPOSAL_FIELDS:
            raise HeldoutPipelineError(
                f"fixed_proposals.{role} fields must be exactly "
                f"{sorted(_FIXED_PROPOSAL_FIELDS)}")
        path = _absolute(binding["path"], f"fixed_proposals.{role}.path")
        if path.is_symlink() or not path.is_file():
            raise HeldoutPipelineError(
                f"fixed_proposals.{role}.path must be an existing non-symlink file")
        observed = pair._sha256(path)
        if observed != binding["sha256"]:
            raise HeldoutPipelineError(f"fixed_proposals.{role}: file SHA-256 differs")
        proposal = _load(path, f"fixed {role} proposal")
        if proposal != expected:
            raise HeldoutPipelineError(
                f"fixed_proposals.{role}: record differs from the exact pair derivation")
        loaded.append(proposal)
    return loaded[0], loaded[1]


def _campaign_command(root: Path, raw_pair: Mapping[str, Any], *, execute: bool) -> list[str]:
    proposal = _load(root / "proposal-v4.json", "prepared proposal")
    store = _load(root / pair.HYPOTHESIS_STORE_FILENAME, "hypothesis store")
    hypotheses = store.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 1:
        raise HeldoutPipelineError("prepared hypothesis store is not singular")
    branch = (raw_pair["intervention"]
              if proposal["proposal_id"] == raw_pair["intervention_proposal_id"]
              else raw_pair["control"])
    command = [
        "python3", "-m", "scripts.kernel_rnd.autokernel.campaign",
        "--campaign-id", proposal["campaign_id"],
        "--candidate-id", branch["candidate_id"],
        "--candidate", "registered:ggml_iqk",
        "--proposal-manifest", str(root / "proposal-v4.json"),
        "--least-commitment-capture-plan",
        str(root / "least-commitment-capture-plan.json"),
        "--matched-experiment-id", raw_pair["matched_experiment_id"],
        "--calibration-bundle", raw_pair["calibration_bundle"],
        "--physical-envelope", str(root / "physical-envelope.json"),
        "--backend", "llama_cpu", "--recipe", pair.PREFILL_RECIPE_ID,
        "--model", raw_pair["model"], "--blocks", str(raw_pair["blocks"]),
        "--reps", str(raw_pair["reps"]),
        "--nominal-khz", str(raw_pair["_nominal_khz"]),
        "--journal-root", str(root),
        "--hypothesis", hypotheses[0]["hypothesis_id"],
        "--hypothesis-store", str(root / pair.HYPOTHESIS_STORE_FILENAME),
        "--json",
    ]
    if execute:
        command.extend(("--execute", "--i-hold-the-host"))
    return command


def prepare(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _FIELDS or raw.get("schema") != SCHEMA:
        raise HeldoutPipelineError(f"pipeline fields/schema must be exactly {sorted(_FIELDS)}")
    raw_pair = raw.get("pair_manifest")
    if not isinstance(raw_pair, Mapping):
        raise HeldoutPipelineError("pair_manifest must be an object")
    raw_pair = copy.deepcopy(dict(raw_pair))
    if raw_pair.get("schema") != pair.SCHEMA \
            or raw_pair.get("measurement_frame", {}).get("recipe_id") != pair.PREFILL_RECIPE_ID:
        raise HeldoutPipelineError("step 5 requires the v2 canonical prefill pair manifest")
    for role in ("intervention", "control"):
        branch = raw_pair.get(role)
        if not isinstance(branch, Mapping) \
                or branch.get("evidence_stage") != "heldout_bound" \
                or branch.get("heldout_outcome") is not None:
            raise HeldoutPipelineError(
                f"{role}: template must reserve a null heldout_bound receipt")
        output = _absolute(branch.get("output_dir"), f"{role}.output_dir")
        if output.exists():
            raise HeldoutPipelineError(f"{role}: campaign directory already exists")
    nominal = raw.get("nominal_khz")
    if isinstance(nominal, bool) or not isinstance(nominal, int) or nominal <= 0:
        raise HeldoutPipelineError("nominal_khz must be a positive integer")
    measurements = raw.get("heldout_measurements")
    if not isinstance(measurements, Mapping) or set(measurements) != {
            "intervention", "control"}:
        raise HeldoutPipelineError("heldout_measurements must contain both roles")
    targets = dict(zip(
        ("intervention", "control"),
        _fixed_proposals(raw.get("fixed_proposals"), _target_proposals(raw_pair))))
    # Pair preparation consumes the exact fixed intervention record. It will
    # independently derive and compare the fixed control again during final
    # plan validation.
    raw_pair["intervention_proposal"] = raw["fixed_proposals"]["intervention"]["path"]
    projected: dict[str, dict] = {}
    for role in ("intervention", "control"):
        item = measurements[role]
        if not isinstance(item, Mapping) or set(item) != _MEASUREMENT_FIELDS:
            raise HeldoutPipelineError(
                f"heldout_measurements.{role} fields must be exactly "
                f"{sorted(_MEASUREMENT_FIELDS)}")
        projected[role] = heldout.project(
            receipt_id=_text(item["receipt_id"], f"{role}.receipt_id"),
            target_proposal=targets[role], measurement=item["measurement"])

    # Only after BOTH decode journals project cleanly may pair.prepare create
    # either final campaign directory.
    parent = Path(raw_pair["intervention"]["output_dir"]).parent
    staging = Path(tempfile.mkdtemp(prefix=".heldout-pair-inputs-", dir=parent))
    try:
        for role in ("intervention", "control"):
            receipt_path = staging / f"{role}-heldout.json"
            receipt_path.write_text(
                json.dumps(projected[role], sort_keys=True), encoding="utf-8")
            raw_pair[role]["heldout_outcome"] = str(receipt_path)
        pair_result = pair.prepare(raw_pair)
        for role in ("intervention", "control"):
            output = Path(pair_result["outputs"][role]["path"])
            if _load(output / "proposal-v4.json", f"published {role} proposal") \
                    != targets[role]:
                for published in pair_result["outputs"].values():
                    shutil.rmtree(Path(published["path"]), ignore_errors=True)
                raise HeldoutPipelineError(
                    f"published {role} proposal differs from fixed proposal")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    raw_pair["_nominal_khz"] = nominal
    roots = {role: Path(pair_result["outputs"][role]["path"])
             for role in ("intervention", "control")}
    archive = raw.get("archive")
    if not isinstance(archive, Mapping) or set(archive) != _ARCHIVE_FIELDS:
        raise HeldoutPipelineError(f"archive fields must be exactly {sorted(_ARCHIVE_FIELDS)}")
    archive_template = {
        **copy.deepcopy(dict(archive)),
        "rows": [{
            "journal_root": str(roots[role]),
            "campaign_id": raw_pair[role]["campaign_id"],
            "proposal_id": (raw_pair["intervention_proposal_id"] if role == "intervention"
                            else raw_pair["control_proposal_id"]),
            "completion_event_id": None,
            "matched_control_id": (raw_pair["control_proposal_id"]
                                   if role == "intervention" else None),
        } for role in ("control", "intervention")],
    }
    result = {
        "schema": RESULT_SCHEMA, "inference_started": False,
        "campaign_executed": False,
        "pair_result": pair_result,
        "heldout_receipts": {role: {
            "path": str(roots[role] / "least-commitment-heldout-outcome.json"),
            "sha256": pair._sha256(
                roots[role] / "least-commitment-heldout-outcome.json"),
        } for role in ("intervention", "control")},
        "commands": {role: {
            "dry_run": _campaign_command(roots[role], raw_pair, execute=False),
            "execute": _campaign_command(roots[role], raw_pair, execute=True),
        } for role in ("intervention", "control")},
        "archive_template": archive_template,
    }
    return {**result, "result_sha256": schemas.content_hash(result)}


def _terminal(row: Mapping[str, Any]) -> str:
    try:
        entries = journal.Journal(
            row["journal_root"], campaign_id=row["campaign_id"]).read_all()
    except (OSError, ValueError, journal.JournalError) as exc:
        raise HeldoutPipelineError(
            f"{row['campaign_id']}: no DECIDED terminal: {exc}") from exc
    terminals = [entry for entry in entries if entry.kind == journal.KIND_STOP_STATE
                 and entry.payload.get("state") == "decided"]
    if len(terminals) != 1:
        raise HeldoutPipelineError(
            f"{row['campaign_id']}: expected one DECIDED terminal, got {len(terminals)}")
    return terminals[0].event_id


def archive(preparation: Mapping[str, Any]) -> dict[str, Any]:
    if preparation.get("schema") != RESULT_SCHEMA:
        raise HeldoutPipelineError("archive input is not a heldout pipeline result")
    expected = preparation.get("result_sha256")
    unsigned = {key: value for key, value in preparation.items() if key != "result_sha256"}
    if expected != schemas.content_hash(unsigned):
        raise HeldoutPipelineError("pipeline result_sha256 differs")
    template = copy.deepcopy(preparation["archive_template"])
    rows = template.pop("rows")
    for row in rows:
        row["completion_event_id"] = _terminal(row)
    output = _absolute(template.pop("output_dir"), "archive.output_dir")
    report_path = _absolute(template.pop("report_output"), "archive.report_output")
    if output.exists() or report_path.exists() or not output.parent.is_dir() \
            or not report_path.parent.is_dir():
        raise HeldoutPipelineError("archive/report outputs must be new under existing parents")
    plan = receipts.assemble_plan(
        archive_id=template["archive_id"], created_at=template["created_at"],
        diagnostic_directions=template["diagnostic_directions"],
        outcome_weights=template["outcome_weights"], completed_rows=rows)
    projected = receipts.project(plan, output)
    archive_record = _load(output / "archive.json", "projected archive")
    report = offline.evaluate_archive(
        archive_record, projection=projected, real_label=True)
    _write_exclusive(report_path, report)
    result = {
        "schema": ARCHIVE_RESULT_SCHEMA, "live_authority": False,
        "projection": projected,
        "report": {"path": str(report_path), "sha256": pair._sha256(report_path)},
        "completion_rows": rows,
    }
    return {**result, "result_sha256": schemas.content_hash(result)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("manifest", type=Path)
    prepare_parser.add_argument("--result", type=Path, required=True)
    archive_parser = sub.add_parser("archive")
    archive_parser.add_argument("preparation_result", type=Path)
    archive_parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.result.is_absolute() or args.result.exists() \
            or not args.result.parent.is_dir():
        raise HeldoutPipelineError("--result must be a new absolute path")
    if args.command == "prepare":
        value = prepare(_load(args.manifest.resolve(), "pipeline manifest"))
    else:
        value = archive(_load(args.preparation_result.resolve(), "pipeline result"))
    _write_exclusive(args.result, value)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
