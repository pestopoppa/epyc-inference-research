#!/usr/bin/env python3
"""Project AK-WM-2 receipts from immutable completed-campaign journal bytes.

This module is intentionally an offline, read-only bridge.  A projection plan
contains JSON-pointer *bindings*, never empirical scalar values.  Every emitted
diagnostic, outcome, frame, and intervention value is copied from a validated
proposal, candidate, evaluation, or DECIDED terminal record in the named
append-only journal.  The receipt retains the source record hash, journal event
id, pointer, and value hash so the existing archive builder can verify the
result without trusting this process's working memory.

The current campaign record does not yet carry all AP-WM-1 diagnostic and
held-out-outcome values.  That is a refusal, not an invitation to synthesize
them.  This projector becomes runnable as soon as those values are journaled by
a reviewed producer; it cannot manufacture a first archive from the IQK speed
result alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import journal, least_commitment_archive_builder as builder
    from . import offline_least_commitment as protocol, schemas
    from .evaluator import recipes
except ImportError:  # direct script execution
    import journal
    import least_commitment_archive_builder as builder
    import offline_least_commitment as protocol
    import schemas
    from evaluator import recipes


PLAN_SCHEMA = "epyc.autokernel.least_commitment_receipt_projection.v1"
PROJECTION_SCHEMA = "epyc.autokernel.least_commitment_receipt_projection_result.v1"
AUTHORITY = "observe_only_journal_projection"
_PLAN_FIELDS = frozenset({
    "schema", "archive_id", "created_at", "candidate_frame_id_binding",
    "diagnostic_directions", "outcome_weights", "rows", "plan_sha256",
})
_ROW_FIELDS = frozenset({
    "journal_root", "campaign_id", "proposal_id", "completion_event_id",
    "candidate_frame_id_binding", "regime_binding", "surface_binding",
    "intervention_id_binding", "changed_factor_binding", "factor_bindings",
    "diagnostic_bindings", "recoding_bindings", "outcome_bindings",
    "matched_control_id",
})
_BINDING_FIELDS = frozenset({"record", "pointer", "record_sha256"})
_OUTCOMES = frozenset({
    "heldout_regime_transfer", "falsifier_resolution", "noise_floor",
})


class ReceiptProjectionError(ValueError):
    """The requested projection is incomplete, synthetic, or not one-factor."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _need_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ReceiptProjectionError(f"{label}: required non-empty text without NUL")
    return value


def _need_number(value: Any, label: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ReceiptProjectionError(f"{label}: required finite number")
    if non_negative and value < 0:
        raise ReceiptProjectionError(f"{label}: must be non-negative")
    return float(value)


def _reject_nonreal(value: Any, label: str) -> None:
    """Reject explicit fixture/synthetic/dry-run markers without prose heuristics."""
    if isinstance(value, Mapping):
        capture = value.get("capture_mode")
        if capture in {"fixture", "synthetic", "dry_run"}:
            raise ReceiptProjectionError(
                f"{label}.capture_mode={capture!r} is not real campaign evidence")
        if value.get("synthetic") is True or value.get("fixture") is True \
                or value.get("dry_run") is True:
            raise ReceiptProjectionError(f"{label}: explicit non-real marker is true")
        for key, child in value.items():
            _reject_nonreal(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonreal(child, f"{label}[{index}]")
    elif isinstance(value, str):
        marker = value.strip().casefold()
        if marker in {"fixture", "synthetic", "dry_run", "dry-run"} \
                or marker.startswith(("fixture://", "fixture/", "synthetic://")):
            raise ReceiptProjectionError(
                f"{label}: explicit non-real marker {value!r}")


def _decode_pointer_token(token: str) -> str:
    out = ""
    index = 0
    while index < len(token):
        if token[index] != "~":
            out += token[index]
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in "01":
            raise ReceiptProjectionError(f"invalid JSON pointer escape in {token!r}")
        out += "/" if token[index + 1] == "1" else "~"
        index += 2
    return out


def _encode_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _resolve_pointer(document: Any, pointer: str, label: str) -> Any:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ReceiptProjectionError(f"{label}.pointer: expected RFC 6901 pointer")
    current = document
    if not pointer:
        return current
    for raw in pointer[1:].split("/"):
        token = _decode_pointer_token(raw)
        if isinstance(current, Mapping):
            if token not in current:
                raise ReceiptProjectionError(
                    f"{label}.pointer: {pointer!r} is absent at {token!r}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise ReceiptProjectionError(
                    f"{label}.pointer: {token!r} is not a canonical array index")
            index = int(token)
            if index >= len(current):
                raise ReceiptProjectionError(
                    f"{label}.pointer: array index {index} is out of bounds")
            current = current[index]
        else:
            raise ReceiptProjectionError(
                f"{label}.pointer: cannot descend through {type(current).__name__}")
    return current


@dataclass(frozen=True)
class CompletedEvidence:
    campaign_id: str
    proposal_id: str
    completion_event_id: str
    proposal: Mapping[str, Any]
    result: Mapping[str, Any]
    candidate: Mapping[str, Any]
    evaluations: Mapping[str, Mapping[str, Any]]
    event_ids: Mapping[str, str]

    def records(self) -> dict[str, Mapping[str, Any]]:
        return {
            "proposal": self.proposal,
            "result": self.result,
            "candidate": self.candidate,
            **{f"evaluation:{key}": value for key, value in self.evaluations.items()},
        }


def _load_completed_evidence(row: Mapping[str, Any]) -> CompletedEvidence:
    root = Path(_need_text(row.get("journal_root"), "row.journal_root"))
    if not root.is_absolute():
        raise ReceiptProjectionError("row.journal_root must be absolute")
    campaign_id = _need_text(row.get("campaign_id"), "row.campaign_id")
    proposal_id = _need_text(row.get("proposal_id"), "row.proposal_id")
    completion_id = _need_text(
        row.get("completion_event_id"), "row.completion_event_id")

    # Reuse the existing builder's clean terminal admission.  This enforces a
    # proposal-v3 join, DECIDED state, executed/ok result, immutable production,
    # nonempty pairs, and released resources before projection begins.
    completed = builder._completed_proposal(row)
    book = journal.Journal(str(root), campaign_id=campaign_id)
    entries = book.read_all()
    views = journal.rebuild_views(entries)
    consistency = journal.check_view_consistency(entries, views)
    if consistency.outcome != schemas.PASS:
        raise ReceiptProjectionError(
            "journal views are not consistent: " + "; ".join(consistency.reasons))
    terminal_entries = [
        entry for entry in entries
        if entry.kind == journal.KIND_STOP_STATE and entry.event_id == completion_id
    ]
    if len(terminal_entries) != 1:
        raise ReceiptProjectionError("completion event does not resolve exactly once")
    terminal_seq = terminal_entries[0].seq

    candidate_id = completed.result.get("candidate_id")
    candidates = [
        entry for entry in entries
        if entry.kind == journal.KIND_CANDIDATE_RECORDED
        and entry.seq < terminal_seq
        and entry.payload.get("candidate_id") == candidate_id
        and entry.payload.get("proposal_id") == proposal_id
        and entry.payload.get("campaign_id") == campaign_id
    ]
    if len(candidates) != 1:
        raise ReceiptProjectionError(
            f"{campaign_id}/{proposal_id}: expected one candidate record before "
            f"terminal, got {len(candidates)}")
    candidate_entry = candidates[0]
    declared_events = candidate_entry.payload.get("evaluation_event_ids")
    if not isinstance(declared_events, list) or not declared_events:
        raise ReceiptProjectionError("candidate record names no evaluation events")
    if len(declared_events) != len(set(declared_events)):
        raise ReceiptProjectionError("candidate record repeats an evaluation event id")
    evaluations: dict[str, Mapping[str, Any]] = {}
    evaluation_event_ids: dict[str, str] = {}
    for event_id in declared_events:
        matches = [
            entry for entry in entries
            if entry.kind == journal.KIND_EVALUATION_EVENT
            and entry.seq < candidate_entry.seq and entry.record_id == event_id
        ]
        if len(matches) != 1:
            raise ReceiptProjectionError(
                f"candidate evaluation {event_id!r} does not resolve exactly once before it")
        event = matches[0]
        payload = event.payload
        if payload.get("campaign_id") != campaign_id \
                or payload.get("candidate_id") != candidate_id:
            raise ReceiptProjectionError(
                f"evaluation {event_id!r} is bound to another campaign/candidate")
        violations = schemas.validate_evaluation_event(payload)
        if violations:
            raise ReceiptProjectionError(
                f"evaluation {event_id!r} is invalid: {'; '.join(violations)}")
        evaluations[event_id] = payload
        evaluation_event_ids[event_id] = event.event_id
    if not any(event.get("tier") == "T1" for event in evaluations.values()):
        raise ReceiptProjectionError("completed proposal has no journaled T1 evidence")

    for label, value in (
        ("proposal", completed.proposal), ("result", completed.result),
        ("candidate", candidate_entry.payload), ("evaluations", evaluations),
    ):
        _reject_nonreal(value, label)
    return CompletedEvidence(
        campaign_id=campaign_id, proposal_id=proposal_id,
        completion_event_id=completion_id, proposal=completed.proposal,
        result=completed.result, candidate=candidate_entry.payload,
        evaluations=evaluations,
        event_ids={
            "proposal": completed.proposal_event_id,
            "result": completion_id,
            "candidate": candidate_entry.event_id,
            **{f"evaluation:{key}": value
               for key, value in evaluation_event_ids.items()},
        },
    )


def _binding(binding: Any, evidence: CompletedEvidence, label: str) -> tuple[Any, dict]:
    if not isinstance(binding, Mapping) or set(binding) != _BINDING_FIELDS:
        raise ReceiptProjectionError(
            f"{label}: binding fields must be exactly {sorted(_BINDING_FIELDS)}")
    record = _need_text(binding.get("record"), f"{label}.record")
    pointer = binding.get("pointer")
    records = evidence.records()
    if record not in records:
        raise ReceiptProjectionError(
            f"{label}.record: {record!r} is not one of {sorted(records)}")
    value = _resolve_pointer(records[record], pointer, label)
    record_sha256 = _need_text(
        binding.get("record_sha256"), f"{label}.record_sha256")
    observed_record_sha256 = _content_hash(records[record])
    if record_sha256 != observed_record_sha256:
        raise ReceiptProjectionError(
            f"{label}.record_sha256: {record_sha256} != journal record "
            f"{observed_record_sha256}")
    provenance = {
        "journal_event_id": evidence.event_ids[record],
        "record": record,
        "record_sha256": observed_record_sha256,
        "pointer": pointer,
        "value_sha256": _content_hash(value),
    }
    return value, provenance


def _mapping_bindings(bindings: Any, expected: set[str], evidence: CompletedEvidence,
                      label: str, *, numeric: bool) -> tuple[dict, dict]:
    if not isinstance(bindings, Mapping) or set(bindings) != expected:
        raise ReceiptProjectionError(
            f"{label}: must bind exactly {sorted(expected)}")
    values: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for key in sorted(expected):
        value, source = _binding(bindings[key], evidence, f"{label}.{key}")
        values[key] = _need_number(
            value, f"{label}.{key}", non_negative=key == "noise_floor") \
            if numeric else value
        provenance[key] = source
    return values, provenance


def _scalar(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ReceiptProjectionError(f"{label}: factor value must be finite")
        return value
    raise ReceiptProjectionError(f"{label}: factor value must be a JSON scalar")


def _project_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
        got = sorted(row) if isinstance(row, Mapping) else type(row).__name__
        raise ReceiptProjectionError(
            f"row fields must be exactly {sorted(_ROW_FIELDS)}; got {got}")
    evidence = _load_completed_evidence(row)
    contract = evidence.proposal["representation_contract"]
    frame = contract["frame_sha256"]
    demand = contract["empirical_demand"]["weights_sha256"]

    scalar_names = (
        "candidate_frame_id", "regime", "surface", "intervention_id",
        "changed_factor",
    )
    scalar_values: dict[str, str] = {}
    scalar_sources: dict[str, dict] = {}
    for name in scalar_names:
        value, source = _binding(
            row[f"{name}_binding"], evidence, f"row.{name}_binding")
        scalar_values[name] = _need_text(value, f"row.{name}")
        scalar_sources[name] = source

    factor_bindings = row.get("factor_bindings")
    if not isinstance(factor_bindings, Mapping) or not factor_bindings:
        raise ReceiptProjectionError("row.factor_bindings must be a non-empty mapping")
    factors: dict[str, Any] = {}
    factor_sources: dict[str, dict] = {}
    for name, binding in sorted(factor_bindings.items()):
        _need_text(name, "row.factor_bindings key")
        value, source = _binding(binding, evidence, f"row.factor_bindings.{name}")
        factors[name] = _scalar(value, f"row.factor_bindings.{name}")
        factor_sources[name] = source
    if scalar_values["changed_factor"] not in factors:
        raise ReceiptProjectionError(
            "row.changed_factor is absent from the journal-bound factor mapping")

    diagnostics, diagnostic_sources = _mapping_bindings(
        row.get("diagnostic_bindings"), set(protocol.DIAGNOSTICS), evidence,
        "row.diagnostic_bindings", numeric=True)
    outcomes, outcome_sources = _mapping_bindings(
        row.get("outcome_bindings"), set(_OUTCOMES), evidence,
        "row.outcome_bindings", numeric=True)
    recoding_bindings = row.get("recoding_bindings")
    fixture_ids = set(contract["semantics_preserving_recoding_fixture_ids"])
    if not isinstance(recoding_bindings, Mapping) or set(recoding_bindings) != fixture_ids:
        raise ReceiptProjectionError(
            "row.recoding_bindings must cover exactly the representation fixtures")
    recodings: dict[str, dict] = {}
    recoding_sources: dict[str, dict] = {}
    for fixture_id in sorted(fixture_ids):
        values, sources = _mapping_bindings(
            recoding_bindings[fixture_id], set(protocol.DIAGNOSTICS), evidence,
            f"row.recoding_bindings.{fixture_id}", numeric=True)
        recodings[fixture_id] = values
        recoding_sources[fixture_id] = sources

    recipe = recipes.get_recipe(evidence.result["spec"]["recipe_id"])
    common = {
        "proposal_id": evidence.proposal_id,
        "proposal_sha256": _content_hash(evidence.proposal),
        "completion_event_id": evidence.completion_event_id,
        "campaign_result_sha256": _content_hash(evidence.result),
        "representation_frame_sha256": frame,
        "empirical_demand_weights_sha256": demand,
    }
    diagnostic = {
        "schema": builder.DIAGNOSTIC_SCHEMA,
        "proposal_id": common["proposal_id"],
        "proposal_sha256": common["proposal_sha256"],
        "representation_frame_sha256": frame,
        "empirical_demand_weights_sha256": demand,
        "diagnostics": diagnostics,
        "recodings": recodings,
        "capture_mode": "measured",
        "authority": AUTHORITY,
        "source_provenance": {
            "diagnostics": diagnostic_sources, "recodings": recoding_sources,
        },
    }
    outcome = {
        "schema": builder.OUTCOME_SCHEMA,
        **common,
        "candidate_frame_id": scalar_values["candidate_frame_id"],
        "metric": recipe.metric, "metric_direction": recipe.metric_direction,
        "regime": scalar_values["regime"], "surface": scalar_values["surface"],
        "intervention_id": scalar_values["intervention_id"],
        "changed_factor": scalar_values["changed_factor"],
        "outcome": outcomes,
        "capture_mode": "measured", "authority": AUTHORITY,
        "source_provenance": {
            **scalar_sources, "outcome": outcome_sources,
        },
    }
    return {
        "evidence": evidence, "diagnostic": diagnostic, "outcome": outcome,
        "factors": factors, "factor_sources": factor_sources,
        "matched_control_id": row.get("matched_control_id"),
    }


def _plan_hash(plan: Mapping[str, Any]) -> str:
    return _content_hash({key: plan[key] for key in sorted(plan)
                          if key != "plan_sha256"})


def assemble_plan(*, archive_id: str, created_at: str,
                  diagnostic_directions: Mapping[str, str],
                  outcome_weights: Mapping[str, float],
                  completed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compile projection bindings from completed campaign journals.

    A caller supplies only completed-campaign identities and an optional matched
    control id.  Every JSON pointer and record digest is derived here from the
    validated candidate record, eliminating the hand-authored binding gap that
    previously made the synthetic test fixture the only working producer.
    """
    if len(completed_rows) < 2:
        raise ReceiptProjectionError("completed_rows requires at least two campaigns")
    compiled = []
    for index, row in enumerate(completed_rows):
        allowed = {"journal_root", "campaign_id", "proposal_id",
                   "completion_event_id", "matched_control_id"}
        if not isinstance(row, Mapping) or set(row) != allowed:
            raise ReceiptProjectionError(
                f"completed_rows[{index}] fields must be exactly {sorted(allowed)}")
        evidence = _load_completed_evidence(row)
        block = evidence.candidate.get("derived_verdicts", {}).get("least_commitment")
        if not isinstance(block, Mapping) or block.get("schema") \
                != "epyc.autokernel.least_commitment_capture.v1":
            raise ReceiptProjectionError(
                f"{evidence.proposal_id}: candidate has no live least-commitment capture")
        if block.get("capture_mode") != "measured":
            raise ReceiptProjectionError(
                f"{evidence.proposal_id}: capture_mode is not measured")
        candidate_sha = _content_hash(evidence.candidate)

        def binding(pointer: str) -> dict[str, str]:
            return {"record": "candidate", "pointer": pointer,
                    "record_sha256": candidate_sha}

        prefix = "/derived_verdicts/least_commitment"
        fixture_ids = evidence.proposal["representation_contract"][
            "semantics_preserving_recoding_fixture_ids"]
        factor_names = sorted(block.get("factors") or {})
        if not factor_names:
            raise ReceiptProjectionError(
                f"{evidence.proposal_id}: least-commitment factors are absent")
        compiled.append({
            "journal_root": row["journal_root"],
            "campaign_id": row["campaign_id"],
            "proposal_id": row["proposal_id"],
            "completion_event_id": row["completion_event_id"],
            "candidate_frame_id_binding": binding(f"{prefix}/candidate_frame_id"),
            "regime_binding": binding(f"{prefix}/regime"),
            "surface_binding": binding(f"{prefix}/surface"),
            "intervention_id_binding": binding(f"{prefix}/intervention_id"),
            "changed_factor_binding": binding(f"{prefix}/changed_factor"),
            "factor_bindings": {
                key: binding(f"{prefix}/factors/{_encode_pointer_token(key)}")
                for key in factor_names},
            "diagnostic_bindings": {
                key: binding(f"{prefix}/diagnostics/{key}")
                for key in protocol.DIAGNOSTICS},
            "recoding_bindings": {
                fixture_id: {
                    key: binding(
                        f"{prefix}/recodings/{_encode_pointer_token(fixture_id)}/{key}")
                    for key in protocol.DIAGNOSTICS}
                for fixture_id in fixture_ids},
            "outcome_bindings": {
                key: binding(f"{prefix}/outcome/{key}") for key in _OUTCOMES},
            "matched_control_id": row["matched_control_id"],
        })
    plan = {
        "schema": PLAN_SCHEMA,
        "archive_id": _need_text(archive_id, "archive_id"),
        "created_at": _need_text(created_at, "created_at"),
        "candidate_frame_id_binding": dict(compiled[0]["candidate_frame_id_binding"]),
        "diagnostic_directions": dict(diagnostic_directions),
        "outcome_weights": dict(outcome_weights),
        "rows": compiled,
    }
    plan["plan_sha256"] = _plan_hash(plan)
    # Validate all fields and joins now, without publishing receipt files.
    if set(plan["diagnostic_directions"]) != set(protocol.DIAGNOSTICS):
        raise ReceiptProjectionError("diagnostic_directions are incomplete")
    if set(plan["outcome_weights"]) != {
            "heldout_regime_transfer", "falsifier_resolution"}:
        raise ReceiptProjectionError("outcome_weights are incomplete")
    return plan


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    body = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
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


def _published_paths(value: Any, staging: Path, published: Path) -> Any:
    """Replace staging path strings in an already validated builder result."""
    if isinstance(value, Mapping):
        return {key: _published_paths(child, staging, published)
                for key, child in value.items()}
    if isinstance(value, list):
        return [_published_paths(child, staging, published) for child in value]
    if isinstance(value, str):
        prefix = str(staging) + os.sep
        if value.startswith(prefix):
            return str(published) + os.sep + value[len(prefix):]
    return value


def project(plan: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write three receipt families plus an existing-builder manifest."""
    if not isinstance(plan, Mapping) or set(plan) != _PLAN_FIELDS:
        got = sorted(plan) if isinstance(plan, Mapping) else type(plan).__name__
        raise ReceiptProjectionError(
            f"plan fields must be exactly {sorted(_PLAN_FIELDS)}; got {got}")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ReceiptProjectionError(f"schema must be {PLAN_SCHEMA!r}")
    expected_hash = _need_text(plan.get("plan_sha256"), "plan.plan_sha256")
    if _plan_hash(plan) != expected_hash:
        raise ReceiptProjectionError("plan.plan_sha256 does not bind the plan")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ReceiptProjectionError("plan.rows requires at least two completed campaigns")
    directions = plan.get("diagnostic_directions")
    if not isinstance(directions, Mapping) or set(directions) != set(protocol.DIAGNOSTICS) \
            or any(value not in protocol.DIRECTIONS for value in directions.values()):
        raise ReceiptProjectionError("diagnostic_directions are incomplete or invalid")
    weights = plan.get("outcome_weights")
    if not isinstance(weights, Mapping) \
            or set(weights) != {"heldout_regime_transfer", "falsifier_resolution"}:
        raise ReceiptProjectionError("outcome_weights are incomplete")
    numeric_weights = [_need_number(value, f"outcome_weights.{key}", non_negative=True)
                       for key, value in weights.items()]
    if not math.isclose(sum(numeric_weights), 1.0, abs_tol=1e-12):
        raise ReceiptProjectionError("outcome_weights must sum to 1")

    projected = [_project_row(row) for row in rows]
    campaign_ids = {item["evidence"].campaign_id for item in projected}
    if len(campaign_ids) < 2:
        raise ReceiptProjectionError(
            "plan.rows requires at least two distinct clean completed campaigns")
    by_id = {item["evidence"].proposal_id: item for item in projected}
    if len(by_id) != len(projected):
        raise ReceiptProjectionError("plan.rows repeats a proposal_id")
    archive_frame_value = None
    archive_frame_source = None
    first = projected[0]["evidence"]
    archive_frame_value, archive_frame_source = _binding(
        plan["candidate_frame_id_binding"], first,
        "plan.candidate_frame_id_binding")
    archive_frame_value = _need_text(archive_frame_value, "plan candidate frame")

    match_receipts: dict[str, dict] = {}
    matched_count = 0
    for item in projected:
        control_id = item["matched_control_id"]
        if control_id is None:
            continue
        matched_count += 1
        proposal_id = item["evidence"].proposal_id
        if not isinstance(control_id, str) or control_id not in by_id \
                or control_id == proposal_id:
            raise ReceiptProjectionError(
                f"{proposal_id}: matched_control_id does not resolve to another row")
        control = by_id[control_id]
        intervention_outcome = item["outcome"]
        control_outcome = control["outcome"]
        for key in ("candidate_frame_id", "metric", "regime", "surface"):
            if intervention_outcome[key] != control_outcome[key]:
                raise ReceiptProjectionError(
                    f"{proposal_id}: matched control differs on {key}")
        if set(item["factors"]) != set(control["factors"]):
            raise ReceiptProjectionError(
                f"{proposal_id}: matched factor vocabularies differ")
        changed = sorted(key for key in item["factors"]
                         if item["factors"][key] != control["factors"][key])
        if len(changed) != 1:
            raise ReceiptProjectionError(
                f"{proposal_id}: matched pair changes {len(changed)} factors, expected exactly one")
        if changed[0] != intervention_outcome["changed_factor"]:
            raise ReceiptProjectionError(
                f"{proposal_id}: changed_factor {intervention_outcome['changed_factor']!r} "
                f"does not equal the sole derived factor {changed[0]!r}")
        match_receipts[proposal_id] = {
            "schema": builder.MATCH_SCHEMA,
            "intervention_proposal_id": proposal_id,
            "control_proposal_id": control_id,
            "intervention_completion_event_id": item["evidence"].completion_event_id,
            "control_completion_event_id": control["evidence"].completion_event_id,
            "candidate_frame_id": intervention_outcome["candidate_frame_id"],
            "regime": intervention_outcome["regime"],
            "surface": intervention_outcome["surface"],
            "changed_factor": changed[0], "one_factor": True,
            "capture_mode": "measured", "authority": AUTHORITY,
            "source_provenance": {
                "intervention_factors": item["factor_sources"],
                "control_factors": control["factor_sources"],
            },
        }
    if matched_count == 0:
        raise ReceiptProjectionError("plan.rows contains no matched intervention")
    if any(item["outcome"]["candidate_frame_id"] != archive_frame_value
           for item in projected):
        raise ReceiptProjectionError("row candidate frames differ from the archive frame")

    requested_output = Path(output_dir)
    if not requested_output.is_absolute():
        raise ReceiptProjectionError("output_dir must be absolute")
    output_dir = requested_output.resolve()
    if not output_dir.name or output_dir.exists():
        raise ReceiptProjectionError("output_dir must name a new directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    build_rows = []
    emitted = []
    try:
        for item in projected:
            proposal_id = item["evidence"].proposal_id
            row_dir = staging / proposal_id
            row_dir.mkdir()
            diagnostic_path = row_dir / "diagnostics.json"
            outcome_path = row_dir / "outcome.json"
            _write_json_exclusive(diagnostic_path, item["diagnostic"])
            _write_json_exclusive(outcome_path, item["outcome"])
            build_row = {
                "journal_root": str(Path(next(
                    row["journal_root"] for row in rows
                    if row["proposal_id"] == proposal_id))),
                "campaign_id": item["evidence"].campaign_id,
                "proposal_id": proposal_id,
                "completion_event_id": item["evidence"].completion_event_id,
                "diagnostic_receipt": {
                    "path": str(diagnostic_path),
                    "sha256": _file_sha256(diagnostic_path)},
                "outcome_receipt": {
                    "path": str(outcome_path),
                    "sha256": _file_sha256(outcome_path)},
            }
            if proposal_id in match_receipts:
                match_path = row_dir / "matched-intervention.json"
                _write_json_exclusive(match_path, match_receipts[proposal_id])
                build_row["matched_control_id"] = item["matched_control_id"]
                build_row["matched_intervention_receipt"] = {
                    "path": str(match_path), "sha256": _file_sha256(match_path)}
                emitted.append(str(match_path))
            build_rows.append(build_row)
            emitted.extend((str(diagnostic_path), str(outcome_path)))

        first_contract = first.proposal["representation_contract"]
        first_frame = first_contract["frame_sha256"]
        first_demand = first_contract["empirical_demand"]["weights_sha256"]
        if any(
            item["evidence"].proposal["representation_contract"]["frame_sha256"]
            != first_frame
            or item["evidence"].proposal["representation_contract"][
                "empirical_demand"]["weights_sha256"] != first_demand
            for item in projected
        ):
            raise ReceiptProjectionError(
                "rows use different representation or empirical-demand frames")
        metric_direction = recipes.get_recipe(
            first.result["spec"]["recipe_id"]).metric_direction
        if any(recipes.get_recipe(item["evidence"].result["spec"]["recipe_id"])
               .metric_direction != metric_direction for item in projected):
            raise ReceiptProjectionError("rows use different metric directions")
        validation_manifest = {
            "schema": builder.BUILD_SCHEMA,
            "archive_id": _need_text(plan.get("archive_id"), "plan.archive_id"),
            "created_at": _need_text(plan.get("created_at"), "plan.created_at"),
            "candidate_frame_id": archive_frame_value,
            "representation_frame_sha256": first_contract["frame_sha256"],
            "empirical_demand_weights_sha256": first_contract["empirical_demand"][
                "weights_sha256"],
            "metric_direction": metric_direction,
            "diagnostic_directions": dict(directions),
            "outcome_weights": dict(weights),
            "rows": build_rows,
        }
        # Nothing at output_dir exists until the unchanged builder accepts the
        # complete private sibling tree.  The published manifest differs only
        # by the mechanically substituted directory prefix.
        archive = builder.build_archive(validation_manifest)
        manifest = _published_paths(validation_manifest, staging, output_dir)
        published_archive = _published_paths(archive, staging, output_dir)
        manifest_path = staging / "archive-build-manifest.json"
        archive_path = staging / "archive.json"
        _write_json_exclusive(manifest_path, manifest)
        _write_json_exclusive(archive_path, published_archive)
        result = {
            "schema": PROJECTION_SCHEMA, "authority": AUTHORITY,
            "plan_sha256": expected_hash,
            "candidate_frame_source": archive_frame_source,
            "archive_build_manifest": {
                "path": str(output_dir / manifest_path.name),
                "sha256": _file_sha256(manifest_path)},
            "archive": {
                "path": str(output_dir / archive_path.name),
                "sha256": _file_sha256(archive_path)},
            "archive_sha256": _content_hash(published_archive),
            "emitted_receipts": sorted(
                str(_published_paths(path, staging, output_dir)) for path in emitted),
        }
        _write_json_exclusive(staging / "projection-result.json", result)
        if output_dir.exists():
            raise ReceiptProjectionError("output_dir appeared during publication")
        os.rename(staging, output_dir)
        return result
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, nargs="?")
    parser.add_argument(
        "--assemble-completed", type=Path,
        help="completed-campaign identity manifest; derives every projection binding",
    )
    parser.add_argument(
        "--plan-output", type=Path,
        help="write the plan derived by --assemble-completed before projection",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.assemble_completed is not None:
        if args.plan is not None or args.plan_output is None:
            parser.error("--assemble-completed requires --plan-output and no positional plan")
        source = json.loads(args.assemble_completed.read_text(encoding="utf-8"))
        if not isinstance(source, Mapping):
            raise ReceiptProjectionError("completed manifest must be a JSON object")
        completed_rows = source.get("rows")
        if not isinstance(completed_rows, list):
            raise ReceiptProjectionError("completed manifest rows must be a list")
        raw = assemble_plan(
            archive_id=source.get("archive_id"), created_at=source.get("created_at"),
            diagnostic_directions=source.get("diagnostic_directions"),
            outcome_weights=source.get("outcome_weights"),
            completed_rows=completed_rows,
        )
        args.plan_output.write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.output_dir is None:
            print(json.dumps({"plan": str(args.plan_output),
                              "plan_sha256": raw["plan_sha256"]}, sort_keys=True))
            return 0
    else:
        if args.plan is None:
            parser.error("a positional plan or --assemble-completed is required")
        if args.output_dir is None:
            parser.error("--output-dir is required when projecting")
        raw = json.loads(args.plan.read_text(encoding="utf-8"))
    result = project(raw, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
