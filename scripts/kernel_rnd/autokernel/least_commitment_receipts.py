#!/usr/bin/env python3
"""Project AK-WM-2 receipts from immutable completed-campaign journal bytes.

This module is intentionally an offline, read-only bridge.  A projection plan
contains JSON-pointer *bindings*, never empirical scalar values.  Every emitted
diagnostic, outcome, frame, and intervention value is copied from a validated
proposal, candidate, evaluation, or DECIDED terminal record in the named
append-only journal.  The receipt retains the source record hash, journal event
id, pointer, and value hash so the existing archive builder can verify the
result without trusting this process's working memory.

A control that terminally refused during preflight may be replaced without
rerunning its already-completed intervention.  The optional ``control_retry_of``
identity is compiled into a typed lineage receipt only when the original
journal proves that no claim, T0, microbenchmark, pair, or decision occurred and
the replacement preserves the exact prospective control contract.  The
intervention must still name the original control proposal; the lineage is the
only admitted alias to the fresh completed proposal.

The current campaign record does not yet carry all AP-WM-1 diagnostic and
held-out-outcome values.  That is a refusal, not an invitation to synthesize
them.  This projector becomes runnable as soon as those values are journaled by
a reviewed producer; it cannot manufacture a first archive from the IQK speed
result alone.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import journal, least_commitment_archive_builder as builder
    from . import least_commitment_capture as capture
    from . import offline_least_commitment as protocol, schemas
    from .evaluator import recipes
except ImportError:  # direct script execution
    import journal
    import least_commitment_archive_builder as builder
    import least_commitment_capture as capture
    import offline_least_commitment as protocol
    import schemas
    from evaluator import recipes


PLAN_SCHEMA = "epyc.autokernel.least_commitment_receipt_projection.v1"
PROJECTION_SCHEMA = "epyc.autokernel.least_commitment_receipt_projection_result.v1"
AUTHORITY = "observe_only_journal_projection"
CONTROL_RETRY_SCHEMA = "epyc.autokernel.preflight_control_retry_lineage.v1"
_PLAN_FIELDS = frozenset({
    "schema", "archive_id", "created_at", "candidate_frame_id_binding",
    "diagnostic_directions", "outcome_weights", "rows", "plan_sha256",
})
_ROW_FIELDS = frozenset({
    "journal_root", "campaign_id", "proposal_id", "completion_event_id",
    "candidate_frame_id_binding", "regime_binding", "surface_binding",
    "intervention_id_binding", "changed_factor_binding",
    "matched_experiment_id_binding", "factor_bindings",
    "diagnostic_bindings", "recoding_bindings", "outcome_bindings",
    "matched_control_id",
})
_RETRY_INPUT_FIELDS = frozenset({
    "schema", "journal_root", "campaign_id", "proposal_id",
    "completion_event_id",
})
_RETRY_RECEIPT_FIELDS = frozenset({
    "schema", "authority", "original", "replacement", "semantic_contract_sha256",
    "claim_journal_observation",
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
    journal_root: Path
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
    # proposal-v3/v4 join, DECIDED state, executed/ok result, immutable production,
    # nonempty pairs, and released resources before projection begins.
    completed = builder._completed_proposal(row)
    book = journal.Journal(str(root), campaign_id=campaign_id)
    report = book.scan()
    if report.torn_tail is not None:
        raise ReceiptProjectionError(
            "completed evidence journal has an unacknowledged torn tail")
    entries = list(report.entries)
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
        journal_root=root,
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


def _read_json_bound(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if not path.is_absolute():
        raise ReceiptProjectionError(f"{label}: path must be absolute")
    try:
        observed = _file_sha256(path)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptProjectionError(f"{label}: cannot read bound JSON: {exc}") from exc
    if observed != expected_sha256:
        raise ReceiptProjectionError(
            f"{label}: SHA-256 {observed} != journal-bound {expected_sha256}")
    if not isinstance(value, Mapping):
        raise ReceiptProjectionError(f"{label}: expected a JSON object")
    return value


def _control_proposal_semantics(proposal: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    """Strip only retry identities from an exact generated A/A control proposal."""
    value = copy.deepcopy(dict(proposal))
    surface = value.get("change", {}).get("parameter_surface")
    if not isinstance(surface, Mapping) or surface.get("candidate") != surface.get("anchor"):
        raise ReceiptProjectionError(f"{label}: proposal is not an exact A/A control")
    hypothesis = value.get("hypothesis")
    if not isinstance(hypothesis, str) or re.fullmatch(
            r"Matched A/A control for [A-Za-z0-9][A-Za-z0-9._:-]*", hypothesis) is None:
        raise ReceiptProjectionError(
            f"{label}: control hypothesis is not the generated matched A/A form")
    value["hypothesis"] = "Matched A/A control for <retry-source>"
    value.pop("campaign_id", None)
    value.pop("proposal_id", None)
    return value


def _diagnostic_source_semantics(binding: Any, *, proposal: Mapping[str, Any],
                                 label: str) -> str:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "receipt_id", "sha256"}:
        raise ReceiptProjectionError(f"{label}: malformed diagnostic source binding")
    path = Path(_need_text(binding.get("path"), f"{label}.path"))
    expected_sha = _need_text(binding.get("sha256"), f"{label}.sha256")
    source = _read_json_bound(path, expected_sha, label)
    if source.get("proposal_sha256") != _content_hash(proposal):
        raise ReceiptProjectionError(f"{label}: diagnostic source names another proposal")
    if source.get("receipt_id") != binding.get("receipt_id"):
        raise ReceiptProjectionError(f"{label}: diagnostic receipt identity differs")
    semantic = copy.deepcopy(dict(source))
    semantic.pop("proposal_sha256", None)
    semantic.pop("receipt_id", None)
    return _content_hash(semantic)


def _control_capture_semantics(*, root: Path, proposal: Mapping[str, Any],
                               result: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    spec = result.get("spec")
    if not isinstance(spec, Mapping):
        raise ReceiptProjectionError(f"{label}: terminal result has no spec")
    expected_sha = _need_text(
        spec.get("least_commitment_capture_plan_sha256"),
        f"{label}.least_commitment_capture_plan_sha256")
    path = (root / "least-commitment-capture-plan.json").resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptProjectionError(
            f"{label}: cannot read capture plan: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ReceiptProjectionError(f"{label}: capture plan is not a JSON object")
    if raw.get("schema") != capture.SCHEMA or capture.plan_sha256(raw) != expected_sha:
        raise ReceiptProjectionError(f"{label}: capture plan schema/hash is invalid")
    if raw.get("campaign_id") != result.get("campaign_id") \
            or raw.get("proposal_id") != proposal.get("proposal_id") \
            or raw.get("candidate_id") != result.get("candidate_id"):
        raise ReceiptProjectionError(f"{label}: capture plan identity differs from its journal")
    if raw.get("role") != "control" or raw.get("matched_control_proposal_id") is not None:
        raise ReceiptProjectionError(f"{label}: capture plan is not an unpaired control arm")
    if raw.get("capture_mode") != "measured" or raw.get("evidence_stage") != "bootstrap" \
            or raw.get("heldout_outcome_receipt") is not None:
        raise ReceiptProjectionError(
            f"{label}: preflight retry lineage is limited to measured bootstrap controls")
    try:
        capture.from_mapping(
            raw, proposal=proposal, campaign_id=result["campaign_id"],
            candidate_id=result["candidate_id"])
    except capture.CapturePlanError as exc:
        raise ReceiptProjectionError(f"{label}: invalid capture plan: {exc}") from exc

    value = copy.deepcopy(dict(raw))
    for key in ("capture_id", "campaign_id", "candidate_id", "proposal_id", "plan_sha256"):
        value.pop(key, None)
    factors = value.get("factors")
    if not isinstance(factors, Mapping):
        raise ReceiptProjectionError(f"{label}: factors are absent")
    factors = copy.deepcopy(dict(factors))
    if factors.get("backend") != "llama_cpu" or factors.get("devices") != []:
        raise ReceiptProjectionError(
            f"{label}: preflight control retry is limited to CPU-only campaigns")
    # The pre-r44 prefill producer omitted this key when its governed value was
    # one.  The live runner used one pair per block; spelling the same default
    # explicitly is an identity migration, not an empirical change.
    factors.setdefault("fresh_pairs_per_block", 1)
    if factors["fresh_pairs_per_block"] != 1:
        raise ReceiptProjectionError(
            f"{label}: retry changes fresh_pairs_per_block from the governed default")
    value["factors"] = factors
    bindings = value.get("diagnostic_source_receipts")
    if not isinstance(bindings, Mapping) or set(bindings) != set(capture.DIAGNOSTICS):
        raise ReceiptProjectionError(f"{label}: diagnostic source bindings are incomplete")
    value["diagnostic_source_receipts"] = {
        name: _diagnostic_source_semantics(
            bindings[name], proposal=proposal,
            label=f"{label}.diagnostic_source_receipts.{name}")
        for name in sorted(capture.DIAGNOSTICS)
    }
    return {
        "proposal": _control_proposal_semantics(proposal, label),
        "capture_plan": value,
    }


def _scan_claim_absence(path_text: Any, campaign_id: str, label: str) -> dict[str, Any]:
    path = Path(_need_text(path_text, f"{label}.claim_journal_path"))
    if not path.is_absolute():
        raise ReceiptProjectionError(f"{label}: claim journal path must be absolute")
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise ReceiptProjectionError(f"{label}: cannot read claim journal: {exc}") from exc
    matches = []
    for line_no, line in enumerate(body.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReceiptProjectionError(
                f"{label}: malformed claim journal line {line_no}: {exc}") from exc
        receipt = row.get("detail", {}).get("receipt") if isinstance(row, Mapping) else None
        if isinstance(receipt, Mapping) and receipt.get("campaign_id") == campaign_id:
            matches.append(row.get("record_id"))
    if matches:
        raise ReceiptProjectionError(
            f"{label}: original preflight-refused campaign acquired a resource claim")
    return {
        "path": str(path), "scanned_bytes": len(body),
        "scanned_sha256": hashlib.sha256(body).hexdigest(),
        "matching_claim_record_ids": [],
    }


def _validate_claim_observation(observation: Any, campaign_id: str, label: str) -> None:
    if not isinstance(observation, Mapping) or set(observation) != {
            "path", "scanned_bytes", "scanned_sha256", "matching_claim_record_ids"}:
        raise ReceiptProjectionError(f"{label}: malformed claim-journal observation")
    path = Path(_need_text(observation.get("path"), f"{label}.path"))
    scanned_bytes = observation.get("scanned_bytes")
    if isinstance(scanned_bytes, bool) or not isinstance(scanned_bytes, int) \
            or scanned_bytes < 0:
        raise ReceiptProjectionError(f"{label}.scanned_bytes is invalid")
    if observation.get("matching_claim_record_ids") != []:
        raise ReceiptProjectionError(f"{label}: observation does not prove claim absence")
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise ReceiptProjectionError(f"{label}: cannot reread claim journal: {exc}") from exc
    if len(body) < scanned_bytes or hashlib.sha256(body[:scanned_bytes]).hexdigest() \
            != observation.get("scanned_sha256"):
        raise ReceiptProjectionError(f"{label}: observed claim-journal prefix changed")
    # New unrelated appends do not stale a lineage receipt, but a later record
    # that names the supposedly claim-free original campaign invalidates it.
    for line_no, line in enumerate(body.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReceiptProjectionError(
                f"{label}: malformed claim journal line {line_no}: {exc}") from exc
        receipt = row.get("detail", {}).get("receipt") if isinstance(row, Mapping) else None
        if isinstance(receipt, Mapping) and receipt.get("campaign_id") == campaign_id:
            raise ReceiptProjectionError(
                f"{label}: original preflight-refused campaign acquired a resource claim")


def _compile_control_retry_lineage(raw: Any, replacement: CompletedEvidence) -> dict[str, Any]:
    """Bind one no-compute preflight refusal to an exact completed control retry."""
    if not isinstance(raw, Mapping) or set(raw) != _RETRY_INPUT_FIELDS \
            or raw.get("schema") != CONTROL_RETRY_SCHEMA:
        raise ReceiptProjectionError(
            f"control_retry_of fields/schema must be exactly {sorted(_RETRY_INPUT_FIELDS)}")
    root = Path(_need_text(raw.get("journal_root"), "control_retry_of.journal_root"))
    if not root.is_absolute():
        raise ReceiptProjectionError("control_retry_of.journal_root must be absolute")
    campaign_id = _need_text(raw.get("campaign_id"), "control_retry_of.campaign_id")
    proposal_id = _need_text(raw.get("proposal_id"), "control_retry_of.proposal_id")
    completion_id = _need_text(
        raw.get("completion_event_id"), "control_retry_of.completion_event_id")
    if campaign_id == replacement.campaign_id or proposal_id == replacement.proposal_id:
        raise ReceiptProjectionError("control retry must use fresh campaign and proposal ids")
    book = journal.Journal(str(root), campaign_id=campaign_id)
    report = book.scan()
    if report.torn_tail is not None:
        raise ReceiptProjectionError("original retry journal has an unacknowledged torn tail")
    entries = list(report.entries)
    if len(entries) != 2 or [entry.kind for entry in entries] != [
            journal.KIND_PROPOSAL_RECORDED, journal.KIND_STOP_STATE]:
        raise ReceiptProjectionError(
            "original retry journal must contain only proposal and preflight refusal; "
            "compute or claim-boundary activity was recorded")
    consistency = journal.check_view_consistency(entries, journal.rebuild_views(entries))
    if consistency.outcome != schemas.PASS:
        raise ReceiptProjectionError("original retry journal views are inconsistent")
    proposal_entry, terminal_entry = entries
    if proposal_entry.record_id != proposal_id or terminal_entry.event_id != completion_id:
        raise ReceiptProjectionError("original retry journal identity does not resolve exactly")
    terminal = terminal_entry.payload
    result = terminal.get("result") if isinstance(terminal, Mapping) else None
    if not isinstance(result, Mapping):
        raise ReceiptProjectionError("original retry terminal carries no result")
    no_compute = {
        "terminal_state": terminal.get("state") == result.get("state") == "preflight_refused",
        "campaign": result.get("campaign_id") == campaign_id,
        "executed_command": result.get("executed") is True,
        "no_pairs": result.get("pairs") == [],
        "no_steps": result.get("steps") == [],
        "no_releases": result.get("releases") == [],
        "no_t0": result.get("t0") is None,
        "no_decision": result.get("decision") is None,
        "preflight_failed": isinstance(result.get("preflight"), Mapping)
                            and result["preflight"].get("outcome") == schemas.FAIL,
        "production": isinstance(result.get("production_unchanged"), Mapping)
                      and result["production_unchanged"].get("outcome") == schemas.PASS,
    }
    failed = sorted(key for key, passed in no_compute.items() if not passed)
    if failed:
        raise ReceiptProjectionError(
            f"original retry terminal is not a no-compute preflight refusal: {failed}")
    original_semantics = _control_capture_semantics(
        root=root, proposal=proposal_entry.payload, result=result, label="original control")
    replacement_semantics = _control_capture_semantics(
        root=replacement.journal_root, proposal=replacement.proposal,
        result=replacement.result, label="replacement control")
    if original_semantics != replacement_semantics:
        raise ReceiptProjectionError(
            "control retry changes source/frame/factor/schedule/control semantics")
    replacement_entries = journal.Journal(
        str(replacement.journal_root), campaign_id=replacement.campaign_id).read_all()
    replacement_proposals = [
        entry for entry in replacement_entries
        if entry.kind == journal.KIND_PROPOSAL_RECORDED
        and entry.record_id == replacement.proposal_id]
    if len(replacement_proposals) != 1:
        raise ReceiptProjectionError("replacement proposal event does not resolve exactly")
    try:
        original_finished = datetime.fromisoformat(
            terminal_entry.written_at.replace("Z", "+00:00"))
        retry_started = datetime.fromisoformat(
            replacement_proposals[0].written_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReceiptProjectionError("retry journal timestamps are invalid") from exc
    if original_finished > retry_started:
        raise ReceiptProjectionError("replacement predates the original preflight refusal")
    claim_observation = _scan_claim_absence(
        original_semantics["capture_plan"]["factors"].get("claim_journal_path"),
        campaign_id, "original control")
    semantic_sha = _content_hash(original_semantics)
    return {
        "schema": CONTROL_RETRY_SCHEMA,
        "authority": AUTHORITY,
        "original": {
            "journal_root": str(root), "campaign_id": campaign_id,
            "proposal_id": proposal_id,
            "proposal_event_id": proposal_entry.event_id,
            "proposal_sha256": _content_hash(proposal_entry.payload),
            "completion_event_id": completion_id,
            "terminal_sha256": _content_hash(terminal),
            "capture_plan_sha256": result["spec"][
                "least_commitment_capture_plan_sha256"],
        },
        "replacement": {
            "journal_root": str(replacement.journal_root),
            "campaign_id": replacement.campaign_id,
            "proposal_id": replacement.proposal_id,
            "proposal_sha256": _content_hash(replacement.proposal),
            "completion_event_id": replacement.completion_event_id,
            "campaign_result_sha256": _content_hash(replacement.result),
            "capture_plan_sha256": replacement.result["spec"][
                "least_commitment_capture_plan_sha256"],
        },
        "semantic_contract_sha256": semantic_sha,
        "claim_journal_observation": claim_observation,
    }


def _validate_control_retry_lineage(receipt: Any, replacement: CompletedEvidence) -> dict:
    if not isinstance(receipt, Mapping) or set(receipt) != _RETRY_RECEIPT_FIELDS \
            or receipt.get("schema") != CONTROL_RETRY_SCHEMA \
            or receipt.get("authority") != AUTHORITY:
        raise ReceiptProjectionError("control_retry_lineage receipt fields/schema differ")
    original = receipt.get("original")
    if not isinstance(original, Mapping):
        raise ReceiptProjectionError("control_retry_lineage.original is absent")
    source = {key: original.get(key) for key in _RETRY_INPUT_FIELDS if key != "schema"}
    source["schema"] = CONTROL_RETRY_SCHEMA
    observed = _compile_control_retry_lineage(source, replacement)
    _validate_claim_observation(
        receipt.get("claim_journal_observation"), original.get("campaign_id"),
        "control_retry_lineage.claim_journal_observation")
    # The global claim journal is append-only and shared.  Its sealed prefix is
    # the proof; unrelated records appended after assembly do not stale it.
    observed["claim_journal_observation"] = receipt.get("claim_journal_observation")
    if observed != receipt:
        raise ReceiptProjectionError("control_retry_lineage receipt is stale or tampered")
    return observed


def _require_planned_control_join(*, intervention_block: Mapping[str, Any],
                                  control_block: Mapping[str, Any],
                                  intervention_proposal_id: str,
                                  control_proposal_id: str,
                                  retry_lineage: Mapping[str, Any] | None) -> None:
    if intervention_block.get("role") != "intervention" \
            or control_block.get("role") != "control" \
            or control_block.get("matched_control_proposal_id") is not None:
        raise ReceiptProjectionError(
            f"{intervention_proposal_id}: matched rows are not "
            "intervention/control capture roles")
    planned_control_id = intervention_block.get("matched_control_proposal_id")
    expected_control_id = (retry_lineage["original"]["proposal_id"]
                           if retry_lineage is not None else control_proposal_id)
    if planned_control_id != expected_control_id:
        raise ReceiptProjectionError(
            f"{intervention_proposal_id}: intervention capture planned control "
            f"{planned_control_id!r}, not {expected_control_id!r}")


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


def _factor_value(value: Any, label: str) -> Any:
    """Admit canonical JSON factors while refusing non-finite numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ReceiptProjectionError(f"{label}: factor value must be finite")
    if isinstance(value, Mapping):
        return {str(key): _factor_value(child, f"{label}.{key}")
                for key, child in value.items()}
    if isinstance(value, list):
        return [_factor_value(child, f"{label}[{index}]")
                for index, child in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ReceiptProjectionError(f"{label}: factor value is not canonical JSON")


def _project_row(row: Mapping[str, Any]) -> dict[str, Any]:
    row_fields = set(row) if isinstance(row, Mapping) else set()
    if not isinstance(row, Mapping) or row_fields not in (
            set(_ROW_FIELDS), set(_ROW_FIELDS) | {"control_retry_lineage"}):
        got = sorted(row) if isinstance(row, Mapping) else type(row).__name__
        raise ReceiptProjectionError(
            f"row fields must be {sorted(_ROW_FIELDS)} with only the optional "
            f"control_retry_lineage extension; got {got}")
    evidence = _load_completed_evidence(row)
    contract = evidence.proposal["representation_contract"]
    frame = contract["frame_sha256"]
    demand = contract["empirical_demand"]["weights_sha256"]

    scalar_names = (
        "candidate_frame_id", "regime", "surface", "intervention_id",
        "changed_factor", "matched_experiment_id",
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
        factors[name] = _factor_value(value, f"row.factor_bindings.{name}")
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
        "matched_experiment_id": scalar_values["matched_experiment_id"],
        "outcome": outcomes,
        "capture_mode": "measured", "authority": AUTHORITY,
        "source_provenance": {
            **scalar_sources, "outcome": outcome_sources,
        },
    }
    retry_lineage = None
    if "control_retry_lineage" in row:
        retry_lineage = _validate_control_retry_lineage(
            row["control_retry_lineage"], evidence)
    return {
        "evidence": evidence, "diagnostic": diagnostic, "outcome": outcome,
        "factors": factors, "factor_sources": factor_sources,
        "diagnostic_semantics_sha256": _need_text(
            evidence.candidate["derived_verdicts"]["least_commitment"].get(
                "diagnostic_semantics_sha256"),
            "least_commitment.diagnostic_semantics_sha256"),
        "matched_control_id": row.get("matched_control_id"),
        "control_retry_lineage": retry_lineage,
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
        if not isinstance(row, Mapping) or set(row) not in (
                allowed, allowed | {"control_retry_of"}):
            raise ReceiptProjectionError(
                f"completed_rows[{index}] fields must be {sorted(allowed)} with only "
                "the optional control_retry_of extension")
        evidence = _load_completed_evidence(row)
        block = evidence.candidate.get("derived_verdicts", {}).get("least_commitment")
        if not isinstance(block, Mapping) or block.get("schema") \
                != "epyc.autokernel.least_commitment_capture.v2":
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
        compiled_row = {
            "journal_root": row["journal_root"],
            "campaign_id": row["campaign_id"],
            "proposal_id": row["proposal_id"],
            "completion_event_id": row["completion_event_id"],
            "candidate_frame_id_binding": binding(f"{prefix}/candidate_frame_id"),
            "regime_binding": binding(f"{prefix}/regime"),
            "surface_binding": binding(f"{prefix}/surface"),
            "intervention_id_binding": binding(f"{prefix}/intervention_id"),
            "changed_factor_binding": binding(f"{prefix}/changed_factor"),
            "matched_experiment_id_binding": binding(
                f"{prefix}/matched_experiment_id"),
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
        }
        if "control_retry_of" in row:
            if row["matched_control_id"] is not None:
                raise ReceiptProjectionError(
                    f"{evidence.proposal_id}: only a control row may carry control_retry_of")
            compiled_row["control_retry_lineage"] = _compile_control_retry_lineage(
                row["control_retry_of"], evidence)
        compiled.append(compiled_row)
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
        intervention_block = item["evidence"].candidate[
            "derived_verdicts"]["least_commitment"]
        control_block = control["evidence"].candidate[
            "derived_verdicts"]["least_commitment"]
        retry_lineage = control["control_retry_lineage"]
        _require_planned_control_join(
            intervention_block=intervention_block, control_block=control_block,
            intervention_proposal_id=proposal_id,
            control_proposal_id=control_id, retry_lineage=retry_lineage)
        if item["diagnostic_semantics_sha256"] \
                == control["diagnostic_semantics_sha256"]:
            raise ReceiptProjectionError(
                f"{proposal_id}: control diagnostic semantics equal intervention semantics")
        intervention_outcome = item["outcome"]
        control_outcome = control["outcome"]
        for key in ("candidate_frame_id", "metric", "regime", "surface"):
            if intervention_outcome[key] != control_outcome[key]:
                raise ReceiptProjectionError(
                    f"{proposal_id}: matched control differs on {key}")
        if (intervention_outcome.get("matched_experiment_id")
                != control_outcome.get("matched_experiment_id")):
            raise ReceiptProjectionError(
                f"{proposal_id}: matched experiment identity differs")
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
            "matched_experiment_id": intervention_outcome[
                "matched_experiment_id"],
            "capture_mode": "measured", "authority": AUTHORITY,
            "control_retry_lineage": retry_lineage,
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
