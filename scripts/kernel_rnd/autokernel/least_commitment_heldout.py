#!/usr/bin/env python3
"""Project and revalidate genuine out-of-regime AK-WM observations.

The receipt produced here never accepts an empirical scalar from its caller.
It joins a target proposal to a distinct, clean completed campaign, resolves
``decision.median_relative`` from the journaled terminal result, and derives a
candidate-frame identity from immutable execution fields.  Capture-plan
validation repeats the same join; a copied hash or hand-entered effect is not
evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import journal, schemas
from .controller import hypotheses
from .evaluator import recipes


SCHEMA = "epyc.autokernel.least_commitment_heldout_outcome.v2"
AUTHORITY = "observe_only_journal_projection"
CAPTURE_MODE = "measured"
_RECEIPT_FIELDS = frozenset({
    "schema", "authority", "receipt_id", "proposal_id", "proposal_sha256",
    "candidate_frame_id", "candidate_frame", "regime", "surface", "metric",
    "metric_direction", "relative_effect", "measurement_record", "capture_mode",
})
_MEASUREMENT_FIELDS = frozenset({
    "journal_root", "campaign_id", "proposal_id", "completion_event_id",
    "journal_event_id", "record", "pointer", "record_sha256", "value_sha256",
})
_FRAME_FIELDS = (
    "candidate_ref", "backend", "model_sha256", "cpu_list", "devices",
    "device_names", "device_index", "n_gpu_layers", "production_commit",
    "measurement_commit", "provider_reference", "changed_factor",
    "anchor_parameter_surface",
)


class HeldoutProjectionError(ValueError):
    """The held-out source is not a compatible real completed measurement."""


@dataclass(frozen=True)
class CompletedMeasurement:
    campaign_id: str
    proposal_id: str
    completion_event_id: str
    proposal: Mapping[str, Any]
    result: Mapping[str, Any]
    candidate: Mapping[str, Any]
    event_ids: Mapping[str, str]


def _reject_nonreal(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if value.get("capture_mode") in {"fixture", "synthetic", "dry_run"} \
                or value.get("synthetic") is True or value.get("fixture") is True \
                or value.get("dry_run") is True:
            raise HeldoutProjectionError(f"{label}: explicit non-real marker")
        for key, child in value.items():
            _reject_nonreal(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonreal(child, f"{label}[{index}]")
    elif isinstance(value, str) and value.strip().casefold() in {
            "fixture", "synthetic", "dry_run", "dry-run"}:
        raise HeldoutProjectionError(f"{label}: explicit non-real marker")


def _validate_hypothesis(root: Path, campaign_id: str,
                         proposal: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    binding = spec.get("hypothesis")
    if not isinstance(binding, Mapping) or binding.get("bound") is not True:
        raise HeldoutProjectionError(
            "held-out measurement campaign is exploratory or unbound")
    hypothesis_id = _text(binding.get("hypothesis_id"), "hypothesis_id")
    authorization = binding.get("authorization")
    if not isinstance(authorization, Mapping):
        raise HeldoutProjectionError("held-out hypothesis authorization is absent")
    try:
        parsed = hypotheses.ClaimAuthorization.from_dict(authorization)
    except (TypeError, ValueError) as exc:
        raise HeldoutProjectionError(
            f"held-out hypothesis authorization is invalid: {exc}") from exc
    if parsed.hypothesis_id != hypothesis_id or parsed.campaign_id != campaign_id:
        raise HeldoutProjectionError("held-out hypothesis authorization identity differs")
    ledger_path = root / hypotheses.LEDGER_FILENAME
    ledger = hypotheses.HypothesisLedger(str(ledger_path)).read()
    if ledger.discarded_tail_bytes:
        raise HeldoutProjectionError("held-out hypothesis ledger has a torn tail")
    opened = [event for event in ledger.events
              if event.kind == hypotheses.EVENT_OPENED
              and event.hypothesis_id == hypothesis_id]
    authorized = [event for event in ledger.events
                  if event.kind == hypotheses.EVENT_CLAIM_AUTHORIZED
                  and event.hypothesis_id == hypothesis_id
                  and event.seq == parsed.ledger_seq]
    statement = (opened[0].payload.get("hypothesis", {}).get("statement")
                 if len(opened) == 1 else None)
    if statement != proposal.get("hypothesis") or len(authorized) != 1 \
            or authorized[0].payload.get("authorization") != dict(authorization):
        raise HeldoutProjectionError(
            "held-out hypothesis statement/authorization does not resolve exactly")


def _load_completed_measurement(row: Mapping[str, Any]) -> CompletedMeasurement:
    normalized = _row(row)
    root = Path(normalized["journal_root"])
    if not root.is_absolute():
        raise HeldoutProjectionError("measurement.journal_root must be absolute")
    campaign_id = normalized["campaign_id"]
    proposal_id = normalized["proposal_id"]
    completion_id = normalized["completion_event_id"]
    book = journal.Journal(str(root), campaign_id=campaign_id)
    report = book.scan()
    if report.defects:
        raise HeldoutProjectionError(
            "completed measurement journal is corrupt: "
            + "; ".join(str(item) for item in report.defects))
    if report.torn_tail is not None:
        raise HeldoutProjectionError(
            "completed measurement journal has an unacknowledged torn tail")
    entries = list(report.entries)
    views = journal.rebuild_views(entries)
    consistency = journal.check_view_consistency(entries, views)
    if consistency.outcome != schemas.PASS:
        raise HeldoutProjectionError(
            "completed measurement journal views are inconsistent: "
            + "; ".join(consistency.reasons))
    proposals = [entry for entry in entries
                 if entry.kind == journal.KIND_PROPOSAL_RECORDED
                 and entry.record_id == proposal_id]
    if len(proposals) != 1:
        raise HeldoutProjectionError(
            f"measurement proposal resolves {len(proposals)} times")
    proposal = proposals[0].payload
    if proposal.get("schema") not in {
            schemas.SCHEMA_PROPOSAL_V3, schemas.SCHEMA_PROPOSAL_V4}:
        raise HeldoutProjectionError("measurement proposal is not proposal-v3/v4")
    violations = schemas.validate_record(proposal)
    if violations:
        raise HeldoutProjectionError(
            "measurement proposal is invalid: " + "; ".join(violations))
    terminals = [entry for entry in entries
                 if entry.kind == journal.KIND_STOP_STATE
                 and entry.event_id == completion_id]
    if len(terminals) != 1:
        raise HeldoutProjectionError(
            "measurement completion event does not resolve exactly once")
    payload = terminals[0].payload
    result = payload.get("result") if isinstance(payload, Mapping) else None
    spec = result.get("spec") if isinstance(result, Mapping) else None
    result_proposal = spec.get("proposal") if isinstance(spec, Mapping) else None
    clean = {
        "state_decided": payload.get("state") == "decided"
                         and result.get("state") == "decided"
                         if isinstance(result, Mapping) else False,
        "campaign": isinstance(result, Mapping)
                    and result.get("campaign_id") == campaign_id,
        "proposal": isinstance(result_proposal, Mapping)
                    and result_proposal.get("proposal_id") == proposal_id,
        "executed": isinstance(result, Mapping) and result.get("executed") is True,
        "ok": isinstance(result, Mapping) and result.get("ok") is True,
        "decision": isinstance(result, Mapping)
                    and isinstance(result.get("decision"), Mapping),
        "production": isinstance(result, Mapping)
                      and isinstance(result.get("production_unchanged"), Mapping)
                      and result["production_unchanged"].get("outcome") == schemas.PASS,
        "releases": isinstance(result, Mapping)
                    and isinstance(result.get("releases"), list)
                    and bool(result["releases"])
                    and all(isinstance(item, Mapping)
                            and item.get("released") is True
                            for item in result["releases"]),
        "pairs": isinstance(result, Mapping)
                 and isinstance(result.get("pairs"), list) and bool(result["pairs"]),
    }
    failed = sorted(key for key, value in clean.items() if not value)
    if failed:
        raise HeldoutProjectionError(
            f"held-out terminal campaign is incomplete: {failed}")
    assert isinstance(result, Mapping) and isinstance(spec, Mapping)
    _validate_hypothesis(root, campaign_id, proposal, spec)
    terminal_seq = terminals[0].seq
    candidate_id = result.get("candidate_id")
    candidates = [entry for entry in entries
                  if entry.kind == journal.KIND_CANDIDATE_RECORDED
                  and entry.seq < terminal_seq
                  and entry.payload.get("candidate_id") == candidate_id
                  and entry.payload.get("proposal_id") == proposal_id
                  and entry.payload.get("campaign_id") == campaign_id]
    if len(candidates) != 1:
        raise HeldoutProjectionError(
            "held-out candidate does not resolve exactly once before terminal")
    declared = candidates[0].payload.get("evaluation_event_ids")
    if not isinstance(declared, list) or not declared or len(declared) != len(set(declared)):
        raise HeldoutProjectionError("held-out candidate evaluation ids are absent/repeated")
    evaluations = [entry for entry in entries
                   if entry.kind == journal.KIND_EVALUATION_EVENT
                   and entry.seq < candidates[0].seq
                   and entry.record_id in set(declared)]
    if len(evaluations) != len(declared) \
            or not any(entry.payload.get("tier") == "T1" for entry in evaluations):
        raise HeldoutProjectionError(
            "held-out candidate lacks exact journaled T1 evidence")
    for entry in evaluations:
        if entry.payload.get("campaign_id") != campaign_id \
                or entry.payload.get("candidate_id") != candidate_id:
            raise HeldoutProjectionError("held-out evaluation identity differs")
        violations = schemas.validate_evaluation_event(entry.payload)
        if violations:
            raise HeldoutProjectionError(
                "held-out evaluation is invalid: " + "; ".join(violations))
    for label, value in (("proposal", proposal), ("result", result),
                         ("candidate", candidates[0].payload),
                         ("evaluations", [entry.payload for entry in evaluations])):
        _reject_nonreal(value, label)
    return CompletedMeasurement(
        campaign_id=campaign_id, proposal_id=proposal_id,
        completion_event_id=completion_id, proposal=proposal, result=result,
        candidate=candidates[0].payload,
        event_ids={"proposal": proposals[0].event_id,
                   "result": terminals[0].event_id,
                   "candidate": candidates[0].event_id},
    )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise HeldoutProjectionError(f"{label}: expected non-empty text without NUL")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise HeldoutProjectionError(f"{label}: expected a finite number")
    return float(value)


def _parameter_surface(proposal: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    change = proposal.get("change")
    surface = change.get("parameter_surface") if isinstance(change, Mapping) else None
    if proposal.get("change_class") != "parameter" or not isinstance(surface, Mapping):
        raise HeldoutProjectionError(f"{label}: expected a parameter proposal")
    candidate, anchor = surface.get("candidate"), surface.get("anchor")
    if not isinstance(candidate, Mapping) or not isinstance(anchor, Mapping) \
            or set(candidate) != {"ggml_iqk"} or set(anchor) != {"ggml_iqk"}:
        raise HeldoutProjectionError(
            f"{label}: held-out IQK projection requires only ggml_iqk")
    return surface


def candidate_frame_from_factors(
        factors: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the cross-regime frame from a prospective matched campaign."""
    surface = _parameter_surface(proposal, "target proposal")
    required = set(_FRAME_FIELDS) - {
        "provider_reference", "changed_factor", "anchor_parameter_surface"}
    missing = sorted(required - set(factors))
    if missing:
        raise HeldoutProjectionError(
            f"target matched factors omit candidate-frame fields: {missing}")
    provider = proposal.get("provider_reference")
    if not isinstance(provider, Mapping) \
            or provider.get("target_backend") != factors.get("backend"):
        raise HeldoutProjectionError(
            "target proposal provider backend differs from matched factors")
    frame = {key: factors[key] for key in _FRAME_FIELDS
             if key not in {"provider_reference", "changed_factor",
                            "anchor_parameter_surface"}}
    frame.update({
        "provider_reference": proposal.get("provider_reference"),
        "changed_factor": "ggml_iqk",
        "anchor_parameter_surface": dict(surface["anchor"]),
    })
    return json.loads(schemas.canonical_json(frame))


def _candidate_frame_from_evidence(
        evidence: CompletedMeasurement) -> dict[str, Any]:
    spec = evidence.result.get("spec")
    if not isinstance(spec, Mapping):
        raise HeldoutProjectionError("completed result has no campaign spec")
    surface = _parameter_surface(evidence.proposal, "measurement proposal")
    required = set(_FRAME_FIELDS) - {
        "provider_reference", "changed_factor", "anchor_parameter_surface"}
    missing = sorted(required - set(spec))
    if missing:
        raise HeldoutProjectionError(
            "completed campaign predates held-out frame provenance; missing spec "
            f"fields: {missing}")
    provider = evidence.proposal.get("provider_reference")
    if not isinstance(provider, Mapping) \
            or provider.get("target_backend") != spec.get("backend"):
        raise HeldoutProjectionError(
            "measurement proposal provider backend differs from completed spec")
    frame = {key: spec[key] for key in _FRAME_FIELDS
             if key not in {"provider_reference", "changed_factor",
                            "anchor_parameter_surface"}}
    frame.update({
        "provider_reference": evidence.proposal.get("provider_reference"),
        "changed_factor": "ggml_iqk",
        "anchor_parameter_surface": dict(surface["anchor"]),
    })
    return json.loads(schemas.canonical_json(frame))


def candidate_frame_id(frame: Mapping[str, Any]) -> str:
    return "akcf-" + schemas.content_hash(frame)


def _row(raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"journal_root", "campaign_id", "proposal_id", "completion_event_id"}
    if not isinstance(raw, Mapping) or set(raw) != allowed:
        raise HeldoutProjectionError(
            f"measurement row fields must be exactly {sorted(allowed)}")
    return {key: _text(raw[key], f"measurement.{key}") for key in sorted(allowed)}


def _measurement(
        row: Mapping[str, Any], *, target_proposal: Mapping[str, Any],
) -> tuple[CompletedMeasurement, float, dict[str, Any], str, str, str, str]:
    normalized = _row(row)
    try:
        evidence = _load_completed_measurement(normalized)
    except (ValueError, OSError, journal.JournalError) as exc:
        raise HeldoutProjectionError(f"completed held-out campaign is invalid: {exc}") from exc
    if _parameter_surface(evidence.proposal, "measurement proposal") \
            != _parameter_surface(target_proposal, "target proposal"):
        raise HeldoutProjectionError(
            "held-out measurement parameter surface differs from target proposal")
    effect = _number(
        evidence.result["decision"].get("median_relative"),
        "held-out decision.median_relative")
    regimes = evidence.proposal.get("target", {}).get("regimes")
    ops = evidence.proposal.get("target", {}).get("ops")
    if not isinstance(regimes, list) or len(regimes) != 1:
        raise HeldoutProjectionError(
            "held-out proposal must name exactly one measured regime")
    if not isinstance(ops, list) or len(ops) != 1:
        raise HeldoutProjectionError(
            "held-out proposal must name exactly one measured surface")
    regime = _text(regimes[0], "held-out regime")
    surface = _text(ops[0], "held-out surface")
    target_regimes = set(target_proposal.get("target", {}).get("regimes") or ())
    if regime in target_regimes:
        raise HeldoutProjectionError(
            "held-out measurement regime is inside the target proposal")
    recipe = recipes.get_recipe(evidence.result["spec"]["recipe_id"])
    frame = _candidate_frame_from_evidence(evidence)
    return evidence, effect, frame, regime, surface, recipe.metric, recipe.metric_direction


def project(*, receipt_id: str, target_proposal: Mapping[str, Any],
            measurement: Mapping[str, Any]) -> dict[str, Any]:
    violations = schemas.validate_proposal(target_proposal)
    if violations:
        raise HeldoutProjectionError(
            "target proposal is invalid: " + "; ".join(violations))
    evidence, effect, frame, regime, surface, metric, direction = _measurement(
        measurement, target_proposal=target_proposal)
    result = {
        "schema": SCHEMA, "authority": AUTHORITY,
        "receipt_id": _text(receipt_id, "receipt_id"),
        "proposal_id": target_proposal["proposal_id"],
        "proposal_sha256": schemas.content_hash(target_proposal),
        "candidate_frame_id": candidate_frame_id(frame),
        "candidate_frame": frame,
        "regime": regime, "surface": surface,
        "metric": metric, "metric_direction": direction,
        "relative_effect": effect,
        "measurement_record": {
            **_row(measurement),
            "journal_event_id": evidence.event_ids["result"],
            "record": "result", "pointer": "/decision/median_relative",
            "record_sha256": schemas.content_hash(evidence.result),
            "value_sha256": schemas.content_hash(effect),
        },
        "capture_mode": CAPTURE_MODE,
    }
    return json.loads(schemas.canonical_json(result))


def validate(
        source: Mapping[str, Any], *, target_proposal: Mapping[str, Any],
        expected_candidate_frame_id: str,
        expected_candidate_frame: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(source, Mapping) or set(source) != _RECEIPT_FIELDS:
        raise HeldoutProjectionError(
            "held-out outcome fields differ from the closed schema")
    if source.get("schema") != SCHEMA or source.get("authority") != AUTHORITY \
            or source.get("capture_mode") != CAPTURE_MODE:
        raise HeldoutProjectionError("held-out schema/authority/capture mode differs")
    if source.get("proposal_id") != target_proposal.get("proposal_id") \
            or source.get("proposal_sha256") != schemas.content_hash(target_proposal):
        raise HeldoutProjectionError("held-out target proposal binding differs")
    _text(source.get("receipt_id"), "held-out receipt_id")
    measurement = source.get("measurement_record")
    if not isinstance(measurement, Mapping) or set(measurement) != _MEASUREMENT_FIELDS:
        raise HeldoutProjectionError(
            "held-out measurement_record fields differ from the closed schema")
    projected = project(
        receipt_id=str(source["receipt_id"]), target_proposal=target_proposal,
        measurement={key: measurement[key] for key in (
            "journal_root", "campaign_id", "proposal_id", "completion_event_id")},
    )
    if projected != dict(source):
        raise HeldoutProjectionError(
            "held-out receipt differs from a fresh projection of its completed journal")
    if source.get("candidate_frame_id") != expected_candidate_frame_id:
        raise HeldoutProjectionError("held-out candidate frame identity differs")
    if expected_candidate_frame is not None:
        expected = json.loads(schemas.canonical_json(expected_candidate_frame))
        if source.get("candidate_frame") != expected \
                or candidate_frame_id(expected) != expected_candidate_frame_id:
            raise HeldoutProjectionError(
                "held-out measurement is not the prospective campaign candidate frame")
    return dict(projected)


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeldoutProjectionError(f"{label}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HeldoutProjectionError(f"{label}: expected a JSON object")
    return value


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = _load(args.manifest.resolve(), "projection manifest")
    required = {"receipt_id", "target_proposal", "measurement"}
    if set(manifest) != required:
        raise HeldoutProjectionError(
            f"manifest fields must be exactly {sorted(required)}")
    output = args.output.resolve()
    if not args.output.is_absolute() or args.output.exists() \
            or not output.parent.is_dir():
        raise HeldoutProjectionError("--output must be a new absolute path")
    target_path = Path(_text(manifest["target_proposal"], "target_proposal"))
    if not target_path.is_absolute() or target_path.is_symlink() \
            or not target_path.is_file():
        raise HeldoutProjectionError(
            "target_proposal must be an existing absolute non-symlink file")
    receipt = project(
        receipt_id=manifest["receipt_id"],
        target_proposal=_load(target_path, "target proposal"),
        measurement=manifest["measurement"],
    )
    _write_json_exclusive(output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
