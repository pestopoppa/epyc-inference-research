"""AKU-06l pure records: unified-driver child accounting and operation settlement v2.

Accounting evidence only, never execution or scientific authority.  Legacy kinds and
schemas are untouched; old readers refuse these (unknown kind / different schema)."""
from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from . import scheduling

CHILD_ACCOUNTED_SCHEMA = "epyc.autokernel.unified_driver_child_accounted.v1"
SETTLED_V2_SCHEMA = "epyc.autokernel.unified_driver_settled.v2"
SETTLEMENT_REQUEST_V2_SCHEMA = "epyc.autokernel.unified_driver_settlement_request.v2"
OPERATIONS = ("source_build", "actor_preparation")
ROLES = scheduling.OPERATION_ROLES
DISPOSITIONS = ("accepted", "diagnostic")
SETTLEMENT_OUTCOMES = ("failed", "invalid", "prerequisite")

_SHA = frozenset("0123456789abcdef")
_CHILD_FIELDS = frozenset({
    "schema", "campaign_id", "config_generation", "config_digest", "supervisor_incarnation",
    "catalog_id", "transition_id", "selection_digest", "operation", "role", "ordinal",
    "binding_digest", "request_id", "plan_digest", "lineage_id", "stage_id", "terminal",
    "terminal_digest", "receipt", "receipt_digest", "disposition", "predecessor", "digest"})
_TERMINAL_FIELDS = frozenset({
    "worker_id", "worker_generation", "request_id", "plan_digest", "lineage_id", "stage_id",
    "grant_id", "grant_generation", "container_id", "return_code", "result_digest",
    "accepted", "reason"})
_SETTLED_V2_FIELDS = frozenset({
    "schema", "catalog_id", "campaign_id", "config_generation", "config_digest",
    "supervisor_incarnation", "selection", "prior_projection_digest",
    "after_projection_digest", "transition_id", "outcome", "terminal_refs", "children",
    "denial_ref"})


class OperationAccountingRefused(ValueError):
    """A child/settlement v2 record is malformed or internally inconsistent."""


def _digest(value: Any) -> str:
    return scheduling.digest(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _SHA for c in value):
        raise OperationAccountingRefused(f"{label} must be a lowercase SHA-256")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OperationAccountingRefused(f"{label} must be a positive integer")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OperationAccountingRefused(f"{label} must be non-empty text")
    return value


def child_digest(body: Mapping[str, Any]) -> str:
    return _digest({key: value for key, value in body.items() if key != "digest"})


def make_child_record(*, campaign_id: str, config_generation: int, config_digest: str,
                      supervisor_incarnation: int, catalog_id: str, transition_id: str,
                      selection_digest: str, operation: str, role: str,
                      binding_digest: str, terminal: Mapping[str, Any],
                      receipt: Mapping[str, Any], disposition: str,
                      predecessor: str | None) -> dict[str, Any]:
    """Build one closed child accounting record (digest included)."""
    body = {
        "schema": CHILD_ACCOUNTED_SCHEMA, "campaign_id": campaign_id,
        "config_generation": config_generation, "config_digest": config_digest,
        "supervisor_incarnation": supervisor_incarnation, "catalog_id": catalog_id,
        "transition_id": transition_id, "selection_digest": selection_digest,
        "operation": operation, "role": role,
        "ordinal": ROLES.index(role) if role in ROLES else -1,
        "binding_digest": binding_digest, "request_id": terminal.get("request_id"),
        "plan_digest": terminal.get("plan_digest"), "lineage_id": terminal.get("lineage_id"),
        "stage_id": terminal.get("stage_id"), "terminal": dict(terminal),
        "terminal_digest": _digest(dict(terminal)), "receipt": dict(receipt),
        "receipt_digest": _digest(dict(receipt)), "disposition": disposition,
        "predecessor": predecessor}
    body["digest"] = child_digest(body)
    return validate_child_record(body)


def validate_child_record(value: Any) -> dict[str, Any]:
    """Closed-shape and self-consistency validation; unknown fields/roles refuse."""
    if not isinstance(value, Mapping) or set(value) != _CHILD_FIELDS:
        raise OperationAccountingRefused("child accounting record has missing/unknown fields")
    row = copy.deepcopy(dict(value))
    if row["schema"] != CHILD_ACCOUNTED_SCHEMA:
        raise OperationAccountingRefused("child accounting record schema is unsupported")
    _text(row["campaign_id"], "campaign_id")
    _positive(row["config_generation"], "config_generation")
    _positive(row["supervisor_incarnation"], "supervisor_incarnation")
    for name in ("config_digest", "catalog_id", "transition_id", "selection_digest",
                 "binding_digest", "plan_digest", "terminal_digest", "receipt_digest",
                 "digest"):
        _sha(row[name], name)
    if row["operation"] not in OPERATIONS:
        raise OperationAccountingRefused("child accounting operation is unsupported")
    if row["role"] not in ROLES or row["ordinal"] != ROLES.index(row["role"]):
        raise OperationAccountingRefused("child accounting role/ordinal is invalid")
    if row["disposition"] not in DISPOSITIONS:
        raise OperationAccountingRefused("child accounting disposition is invalid")
    for name in ("request_id", "lineage_id", "stage_id"):
        _text(row[name], name)
    if row["ordinal"] == 0:
        if row["predecessor"] is not None:
            raise OperationAccountingRefused("first child cannot name a predecessor")
    else:
        _sha(row["predecessor"], "predecessor")
    terminal = row["terminal"]
    if not isinstance(terminal, Mapping) or set(terminal) != _TERMINAL_FIELDS:
        raise OperationAccountingRefused("child terminal identity is not closed")
    if (any(terminal[name] != row[name] for name in (
            "request_id", "plan_digest", "lineage_id", "stage_id"))
            or terminal["lineage_id"] != row["lineage_id"]
            or _digest(dict(terminal)) != row["terminal_digest"]):
        raise OperationAccountingRefused("child terminal differs from its record identity")
    try:
        receipt = scheduling.HeldClaimReceipt.from_dict(row["receipt"])
    except (scheduling.SchedulingRefused, TypeError, ValueError) as exc:
        raise OperationAccountingRefused(f"child receipt is malformed: {exc}") from exc
    if (receipt.ownership_generation != terminal["worker_generation"]
            or receipt.allocation_generation != terminal["grant_generation"]):
        raise OperationAccountingRefused("child receipt generations differ from its terminal")
    row["receipt"] = receipt.to_dict()
    if _digest(row["receipt"]) != row["receipt_digest"]:
        raise OperationAccountingRefused("child receipt digest differs")
    if child_digest(row) != row["digest"]:
        raise OperationAccountingRefused("child accounting content digest differs")
    return row


def validate_child_chain(children: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """An ordered planner/critic/build prefix with exact predecessors and no reuse."""
    rows = [validate_child_record(item) for item in children]
    seen_requests: set[str] = set()
    seen_receipts: set[str] = set()
    previous: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        if row["ordinal"] != index:
            raise OperationAccountingRefused("child records are not an ordered role prefix")
        if row["request_id"] in seen_requests:
            raise OperationAccountingRefused("child request is reused by another role")
        receipt_id = row["receipt"]["receipt_id"]
        if receipt_id in seen_receipts:
            raise OperationAccountingRefused("one receipt cannot satisfy two child roles")
        seen_requests.add(row["request_id"])
        seen_receipts.add(receipt_id)
        if previous is not None:
            if (row["predecessor"] != previous["digest"]
                    or any(row[name] != previous[name] for name in (
                        "campaign_id", "config_generation", "config_digest",
                        "supervisor_incarnation", "catalog_id", "transition_id",
                        "selection_digest", "operation", "binding_digest"))):
                raise OperationAccountingRefused("child chain breaks predecessor/identity")
        previous = row
    return rows


def operation_children(rows: Sequence[Mapping[str, Any]]) -> tuple[scheduling.OperationChild, ...]:
    """Derive the original receipts (never a synthetic aggregate) from child records."""
    return tuple(scheduling.OperationChild(
        row["role"], row["selection_digest"],
        scheduling.HeldClaimReceipt.from_dict(row["receipt"]))
        for row in validate_child_chain(rows))


def check_outcome_against_children(outcome: str, rows: Sequence[Mapping[str, Any]],
                                   denial_ref: str | None) -> None:
    """Closed settlement rules: empty prefix = typed no-receipt cancellation."""
    if outcome not in SETTLEMENT_OUTCOMES:
        raise OperationAccountingRefused("operation settlement outcome is unsupported")
    if not rows:
        if denial_ref is None or outcome != "failed":
            raise OperationAccountingRefused(
                "an empty child prefix settles only as a failed denial cancellation")
        _sha(denial_ref, "denial_ref")
        return
    if outcome == "prerequisite":
        if (tuple(row["role"] for row in rows) != ROLES
                or any(row["disposition"] != "accepted" for row in rows)):
            raise OperationAccountingRefused(
                "prerequisite requires accepted planner, critic and build children")
        if denial_ref is not None:
            raise OperationAccountingRefused("a successful operation carries no denial")
    elif denial_ref is not None:
        _sha(denial_ref, "denial_ref")


def validate_settled_v2(payload: Any) -> list[str]:
    """Violations (empty = valid) of one UNIFIED_DRIVER_SETTLED.v2 payload."""
    out: list[str] = []
    if not isinstance(payload, Mapping) or set(payload) != _SETTLED_V2_FIELDS:
        return ["payload: unified driver v2 settlement has missing/unknown fields"]
    if payload["schema"] != SETTLED_V2_SCHEMA:
        out.append(f"schema: must be {SETTLED_V2_SCHEMA!r}")
    for name in ("catalog_id", "config_digest", "prior_projection_digest",
                 "after_projection_digest", "transition_id"):
        try:
            _sha(payload[name], name)
        except OperationAccountingRefused as exc:
            out.append(str(exc))
    if not isinstance(payload["campaign_id"], str) or not payload["campaign_id"]:
        out.append("campaign_id: required non-empty text")
    for name in ("config_generation", "supervisor_incarnation"):
        try:
            _positive(payload[name], name)
        except OperationAccountingRefused as exc:
            out.append(str(exc))
    if not isinstance(payload["selection"], Mapping):
        out.append("selection: required mapping")
    if payload["outcome"] not in SETTLEMENT_OUTCOMES:
        out.append("outcome: unsupported for an operation settlement")
    refs = payload["terminal_refs"]
    if (not isinstance(refs, list) or any(not isinstance(i, str) or not i for i in refs)
            or len(refs) != len(set(refs))):
        out.append("terminal_refs: required unique nonempty strings")
    children = payload["children"]
    if (not isinstance(children, list)
            or any(not isinstance(i, str) or len(i) != 64 or any(c not in _SHA for c in i)
                   for i in children)
            or len(children) != len(set(children)) or len(children) > len(ROLES)):
        out.append("children: required unique ordered child digests")
    denial = payload["denial_ref"]
    if denial is not None:
        try:
            _sha(denial, "denial_ref")
        except OperationAccountingRefused as exc:
            out.append(str(exc))
    if isinstance(children, list) and not children and (
            denial is None or payload["outcome"] != "failed"):
        out.append("children: an empty prefix requires a failed outcome and denial_ref")
    if isinstance(children, list) and children and payload["outcome"] == "prerequisite" \
            and (len(children) != len(ROLES) or denial is not None):
        out.append("outcome: prerequisite requires every role and no denial")
    try:
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        out.append(f"payload: non-canonical JSON value: {exc}")
    return out


__all__ = [
    "CHILD_ACCOUNTED_SCHEMA", "SETTLED_V2_SCHEMA", "SETTLEMENT_REQUEST_V2_SCHEMA",
    "OPERATIONS", "ROLES", "DISPOSITIONS", "SETTLEMENT_OUTCOMES",
    "OperationAccountingRefused", "child_digest", "make_child_record",
    "validate_child_record", "validate_child_chain", "operation_children",
    "check_outcome_against_children", "validate_settled_v2"]
