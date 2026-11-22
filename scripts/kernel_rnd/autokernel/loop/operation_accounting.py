"""AKU-06l pure records: unified-driver child accounting and operation settlement v2.

Accounting evidence only, never execution or scientific authority.  Legacy kinds and
schemas are untouched; old readers refuse these (unknown kind / different schema)."""
from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from . import scheduling

CHILD_ACCOUNTED_SCHEMA = "epyc.autokernel.unified_driver_child_accounted.v1"
# v2 = a fallback invocation (attempt >= 1) of an already accounted role; v1 stays the
# exact first-invocation record, so single-invocation chains are byte-identical.
CHILD_ACCOUNTED_SCHEMA_V2 = "epyc.autokernel.unified_driver_child_accounted.v2"
MAX_OPERATION_CHILDREN = 32
SETTLED_V2_SCHEMA = "epyc.autokernel.unified_driver_settled.v2"
CHILD_INTENT_SCHEMA = "epyc.autokernel.unified_driver_child_intent.v1"
SETTLEMENT_REQUEST_V2_SCHEMA = "epyc.autokernel.unified_driver_settlement_request.v2"
ENROLLMENT_REF_SCHEMA = "epyc.autokernel.unified_driver_enrollment_ref.v1"
INTENT_RESOLVED_SCHEMA = "epyc.autokernel.unified_driver_child_intent_resolved.v1"
# A single-stage operation (standalone profile/runtime/calibration work) has exactly one
# child, role "stage", always recorded as child v2 (attempt 0); a v1 reader refuses it.
SINGLE_STAGE_ROLE = "stage"
SINGLE_STAGE_OPERATIONS = ("runtime_comparison", "calibration_preparation",
                           "profile_preparation")
OPERATIONS = ("source_build", "actor_preparation", *SINGLE_STAGE_OPERATIONS)
ROLES = scheduling.OPERATION_ROLES
DISPOSITIONS = ("accepted", "diagnostic")
SETTLEMENT_OUTCOMES = ("failed", "invalid", "prerequisite")

_SHA = frozenset("0123456789abcdef")
_CHILD_FIELDS = frozenset({
    "schema", "campaign_id", "config_generation", "config_digest", "supervisor_incarnation",
    "catalog_id", "transition_id", "selection_digest", "operation", "role", "ordinal",
    "binding_digest", "request_id", "plan_digest", "lineage_id", "stage_id", "terminal",
    "terminal_digest", "receipt", "receipt_digest", "disposition", "predecessor", "digest"})
_CHILD_FIELDS_V2 = _CHILD_FIELDS | {"attempt"}
_INTENT_FIELDS = frozenset({
    "schema", "campaign_id", "config_generation", "config_digest", "supervisor_incarnation",
    "catalog_id", "transition_id", "selection_digest", "operation", "role",
    "binding_digest", "request_id", "plan_digest", "lineage_id", "stage_id",
    "artifact_contract_digest", "digest"})
_TERMINAL_FIELDS = frozenset({
    "worker_id", "worker_generation", "request_id", "plan_digest", "lineage_id", "stage_id",
    "grant_id", "grant_generation", "container_id", "return_code", "result_digest",
    "accepted", "reason"})
_SETTLED_V2_FIELDS = frozenset({
    "schema", "catalog_id", "campaign_id", "config_generation", "config_digest",
    "supervisor_incarnation", "selection", "prior_projection_digest",
    "after_projection_digest", "transition_id", "outcome", "terminal_refs", "children",
    "denial_ref"})
# Additive optional keys: absent keeps a legacy v2 payload byte-identical; the
# pre-AKU-06l2 v2 reader (exact field set) refuses a payload that carries one.
_SETTLED_V2_OPTIONAL = frozenset({"operation", "enrollment_ref"})
_ENROLLMENT_REF_FIELDS = frozenset({
    "schema", "enrollment_digest", "build_child_digest", "build_identity_digest"})
_RESOLUTION_FIELDS = frozenset({
    "schema", "campaign_id", "config_generation", "config_digest",
    "supervisor_incarnation", "catalog_id", "transition_id", "request_id",
    "intent_digest", "intent_incarnation", "resolution", "cost", "reason", "digest"})


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


def _ordinal(role: Any) -> int:
    return ROLES.index(role) if role in ROLES else 0 if role == SINGLE_STAGE_ROLE else -1


def single_stage(rows: Sequence[Mapping[str, Any]]) -> bool:
    """True iff ``rows`` is the (non-empty) child prefix of a single-stage operation."""
    return bool(rows) and rows[0]["role"] == SINGLE_STAGE_ROLE


def ordered_roles(operation: str, roles: Sequence[str]) -> bool:
    """The closed role order of ``operation``: one ``stage`` or a planner-first chain."""
    if operation in SINGLE_STAGE_OPERATIONS:
        return tuple(roles) == (SINGLE_STAGE_ROLE,)
    return scheduling.ordered_operation_roles(roles)


def child_digest(body: Mapping[str, Any]) -> str:
    return _digest({key: value for key, value in body.items() if key != "digest"})


def make_child_record(*, campaign_id: str, config_generation: int, config_digest: str,
                      supervisor_incarnation: int, catalog_id: str, transition_id: str,
                      selection_digest: str, operation: str, role: str,
                      binding_digest: str, terminal: Mapping[str, Any],
                      receipt: Mapping[str, Any], disposition: str,
                      predecessor: str | None, attempt: int = 0) -> dict[str, Any]:
    """Build one closed child accounting record (digest included); v2 iff attempt > 0
    or the single-stage role."""
    v2 = bool(attempt) or role == SINGLE_STAGE_ROLE
    body = {
        "schema": CHILD_ACCOUNTED_SCHEMA_V2 if v2 else CHILD_ACCOUNTED_SCHEMA,
        "campaign_id": campaign_id,
        "config_generation": config_generation, "config_digest": config_digest,
        "supervisor_incarnation": supervisor_incarnation, "catalog_id": catalog_id,
        "transition_id": transition_id, "selection_digest": selection_digest,
        "operation": operation, "role": role,
        "ordinal": _ordinal(role),
        "binding_digest": binding_digest, "request_id": terminal.get("request_id"),
        "plan_digest": terminal.get("plan_digest"), "lineage_id": terminal.get("lineage_id"),
        "stage_id": terminal.get("stage_id"), "terminal": dict(terminal),
        "terminal_digest": _digest(dict(terminal)), "receipt": dict(receipt),
        "receipt_digest": _digest(dict(receipt)), "disposition": disposition,
        "predecessor": predecessor, **({"attempt": attempt} if v2 else {})}
    body["digest"] = child_digest(body)
    return validate_child_record(body)


def validate_child_record(value: Any) -> dict[str, Any]:
    """Closed-shape and self-consistency validation; unknown fields/roles refuse."""
    v2 = isinstance(value, Mapping) and value.get("schema") == CHILD_ACCOUNTED_SCHEMA_V2
    if not isinstance(value, Mapping) or set(value) != (_CHILD_FIELDS_V2 if v2 else _CHILD_FIELDS):
        raise OperationAccountingRefused("child accounting record has missing/unknown fields")
    row = copy.deepcopy(dict(value))
    if row["schema"] not in (CHILD_ACCOUNTED_SCHEMA, CHILD_ACCOUNTED_SCHEMA_V2):
        raise OperationAccountingRefused("child accounting record schema is unsupported")
    if v2 and row["role"] != SINGLE_STAGE_ROLE:
        _positive(row["attempt"], "attempt")
    elif v2 and (row["attempt"] != 0 or isinstance(row["attempt"], bool)):
        raise OperationAccountingRefused("a single-stage child has exactly one attempt")
    _text(row["campaign_id"], "campaign_id")
    _positive(row["config_generation"], "config_generation")
    _positive(row["supervisor_incarnation"], "supervisor_incarnation")
    for name in ("config_digest", "catalog_id", "transition_id", "selection_digest",
                 "binding_digest", "plan_digest", "terminal_digest", "receipt_digest",
                 "digest"):
        _sha(row[name], name)
    if row["operation"] not in OPERATIONS:
        raise OperationAccountingRefused("child accounting operation is unsupported")
    if _ordinal(row["role"]) < 0 or row["ordinal"] != _ordinal(row["role"]):
        raise OperationAccountingRefused("child accounting role/ordinal is invalid")
    if ((row["role"] == SINGLE_STAGE_ROLE) != (row["operation"] in SINGLE_STAGE_OPERATIONS)
            or (row["role"] == SINGLE_STAGE_ROLE and not v2)):
        raise OperationAccountingRefused("child accounting role differs from its operation")
    if row["disposition"] not in DISPOSITIONS:
        raise OperationAccountingRefused("child accounting disposition is invalid")
    for name in ("request_id", "lineage_id", "stage_id"):
        _text(row[name], name)
    if row["ordinal"] == 0 and child_attempt(row) == 0:
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


def child_attempt(row: Mapping[str, Any]) -> int:
    """The per-role invocation index (0 = first, v1; >= 1 = fallback, v2)."""
    return row.get("attempt", 0)


def make_child_intent(**fields: Any) -> dict[str, Any]:
    """One closed build-intent-before-launch record (digest included).

    Written before the child is launched; a replayed intent with no accounted child
    is accounting uncertainty, never success and never zero cost."""
    body = {"schema": CHILD_INTENT_SCHEMA, **fields}
    body["digest"] = child_digest(body)
    return validate_child_intent(body)


def validate_child_intent(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _INTENT_FIELDS \
            or value["schema"] != CHILD_INTENT_SCHEMA:
        raise OperationAccountingRefused("child intent has missing/unknown fields or schema")
    row = copy.deepcopy(dict(value))
    _text(row["campaign_id"], "campaign_id")
    _positive(row["config_generation"], "config_generation")
    _positive(row["supervisor_incarnation"], "supervisor_incarnation")
    for name in ("config_digest", "catalog_id", "transition_id", "selection_digest",
                 "binding_digest", "plan_digest", "artifact_contract_digest", "digest"):
        _sha(row[name], name)
    for name in ("request_id", "lineage_id", "stage_id"):
        _text(row[name], name)
    if row["operation"] != "source_build" or row["role"] != "build":
        raise OperationAccountingRefused("child intent is defined only for a source build")
    if child_digest(row) != row["digest"]:
        raise OperationAccountingRefused("child intent content digest differs")
    return row


def validate_child_chain(children: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """An ordered planner/critic/build sequence with exact predecessors and no reuse.

    Each role may repeat (fallback invocations, attempt 0, 1, ...) before the next role;
    no role is skipped or revisited."""
    rows = [validate_child_record(item) for item in children]
    if len(rows) > MAX_OPERATION_CHILDREN:
        raise OperationAccountingRefused("operation has too many child records")
    seen_requests: set[str] = set()
    seen_receipts: set[str] = set()
    previous: dict[str, Any] | None = None
    for row in rows:
        expected = ((0, 0) if previous is None else
                    (previous["ordinal"], child_attempt(previous) + 1)
                    if row["ordinal"] == previous["ordinal"] else (previous["ordinal"] + 1, 0))
        if (row["ordinal"], child_attempt(row)) != expected:
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
                                   denial_ref: str | None, *,
                                   enrollment_ref: Any = None, uncertain: int = 0) -> None:
    """Closed settlement rules: empty prefix = typed no-receipt cancellation.

    A single-stage child settles any v1 scheduler outcome, but a diagnostic child only
    ``failed``. An enrollment reference is extra evidence on an accepted source-build
    prerequisite, never a substitute for its build child. ``uncertain`` resolved intents
    (no accounted cost) allow only ``failed`` over a non-empty prefix."""
    if single_stage(rows):
        if (outcome not in scheduling.OUTCOMES or len(rows) != 1 or denial_ref is not None
                or enrollment_ref is not None or uncertain
                or (rows[0]["disposition"] != "accepted" and outcome != "failed")):
            raise OperationAccountingRefused(
                "a single-stage operation settles one child; diagnostic only as failed")
        return
    if outcome not in SETTLEMENT_OUTCOMES:
        raise OperationAccountingRefused("operation settlement outcome is unsupported")
    if enrollment_ref is not None:
        if outcome != "prerequisite":
            raise OperationAccountingRefused("enrollment is referenced only by a prerequisite")
        check_enrollment_ref(enrollment_ref, rows)
    if uncertain and (not rows or outcome != "failed"):
        raise OperationAccountingRefused(
            "a resolved uncertain intent settles only as failed over its accounted prefix")
    if not rows:
        if denial_ref is None or outcome != "failed":
            raise OperationAccountingRefused(
                "an empty child prefix settles only as a failed denial cancellation")
        _sha(denial_ref, "denial_ref")
        return
    if outcome == "prerequisite":
        last = {row["role"]: row for row in rows}  # each role's final invocation
        if (tuple(last) != ROLES
                or any(row["disposition"] != "accepted" for row in last.values())):
            raise OperationAccountingRefused(
                "prerequisite requires accepted planner, critic and build children")
        if denial_ref is not None:
            raise OperationAccountingRefused("a successful operation carries no denial")
    elif denial_ref is not None:
        _sha(denial_ref, "denial_ref")


def make_enrollment_ref(enrollment: Mapping[str, Any],
                        rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reference one published source-build enrollment from its accepted build child."""
    try:
        selection = scheduling.Selection.from_dict(enrollment["selection"]).digest
        ref = {"schema": ENROLLMENT_REF_SCHEMA, "enrollment_digest": _digest(dict(enrollment)),
               "build_child_digest": rows[-1]["digest"] if rows else None,
               "build_identity_digest": enrollment["build_identity_digest"]}
    except (KeyError, TypeError, scheduling.SchedulingRefused) as exc:
        raise OperationAccountingRefused(f"enrollment is malformed: {exc}") from exc
    if (not rows or enrollment.get("transition_id") != rows[-1]["transition_id"]
            or selection != rows[-1]["selection_digest"]):
        raise OperationAccountingRefused("enrollment belongs to another transition/selection")
    check_enrollment_ref(ref, rows)
    return ref


def check_enrollment_ref(value: Any, rows: Sequence[Mapping[str, Any]] | None) -> dict:
    """Closed shape; with ``rows`` it must name the final accepted source-build child."""
    if not isinstance(value, Mapping) or set(value) != _ENROLLMENT_REF_FIELDS \
            or value["schema"] != ENROLLMENT_REF_SCHEMA:
        raise OperationAccountingRefused("enrollment reference fields/schema differ")
    for name in ("enrollment_digest", "build_child_digest", "build_identity_digest"):
        _sha(value[name], name)
    if rows is not None:
        last = rows[-1] if rows else None
        if (last is None or last["operation"] != "source_build" or last["role"] != "build"
                or last["disposition"] != "accepted"
                or value["build_child_digest"] != last["digest"]):
            raise OperationAccountingRefused(
                "enrollment does not reference the final accepted build child")
    return dict(value)


def make_intent_resolution(intent: Mapping[str, Any], *, supervisor_incarnation: int,
                           reason: str) -> dict[str, Any]:
    """Typed diagnostic resolution of a replayed build intent with no accounted child.

    The spent cost stays uncertain: never success, zero cost or a relaunch."""
    intent = validate_child_intent(intent)
    body = {"schema": INTENT_RESOLVED_SCHEMA,
            **{name: intent[name] for name in (
                "campaign_id", "config_generation", "config_digest", "catalog_id",
                "transition_id", "request_id")},
            "supervisor_incarnation": supervisor_incarnation,
            "intent_digest": intent["digest"],
            "intent_incarnation": intent["supervisor_incarnation"],
            "resolution": "uncertain_diagnostic", "cost": "uncertain", "reason": reason}
    body["digest"] = child_digest(body)
    return validate_intent_resolution(body)


def validate_intent_resolution(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESOLUTION_FIELDS \
            or value["schema"] != INTENT_RESOLVED_SCHEMA:
        raise OperationAccountingRefused("intent resolution has missing/unknown fields or schema")
    row = copy.deepcopy(dict(value))
    _text(row["campaign_id"], "campaign_id")
    _text(row["request_id"], "request_id")
    for name in ("config_generation", "supervisor_incarnation", "intent_incarnation"):
        _positive(row[name], name)
    for name in ("config_digest", "catalog_id", "transition_id", "intent_digest", "digest"):
        _sha(row[name], name)
    if (row["resolution"] != "uncertain_diagnostic" or row["cost"] != "uncertain"
            or row["intent_incarnation"] >= row["supervisor_incarnation"]
            or len(_text(row["reason"], "reason")) > 4096):
        raise OperationAccountingRefused(
            "intent resolution is only an uncertain diagnostic of an earlier lifetime")
    if child_digest(row) != row["digest"]:
        raise OperationAccountingRefused("intent resolution content digest differs")
    return row


def validate_settled_v2(payload: Any) -> list[str]:
    """Violations (empty = valid) of one UNIFIED_DRIVER_SETTLED.v2 payload."""
    out: list[str] = []
    if (not isinstance(payload, Mapping) or not _SETTLED_V2_FIELDS <= set(payload)
            or set(payload) - _SETTLED_V2_FIELDS - _SETTLED_V2_OPTIONAL):
        return ["payload: unified driver v2 settlement has missing/unknown fields"]
    operation = payload.get("operation")
    if "operation" in payload and operation not in SINGLE_STAGE_OPERATIONS:
        out.append("operation: only a single-stage operation is named")
    if "enrollment_ref" in payload:
        try:
            check_enrollment_ref(payload["enrollment_ref"], None)
        except OperationAccountingRefused as exc:
            out.append(str(exc))
        if payload["outcome"] != "prerequisite" or "operation" in payload:
            out.append("enrollment_ref: only on a multi-child prerequisite")
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
    if payload["outcome"] not in (
            scheduling.OUTCOMES if "operation" in payload else SETTLEMENT_OUTCOMES):
        out.append("outcome: unsupported for an operation settlement")
    refs = payload["terminal_refs"]
    if (not isinstance(refs, list) or any(not isinstance(i, str) or not i for i in refs)
            or len(refs) != len(set(refs))):
        out.append("terminal_refs: required unique nonempty strings")
    children = payload["children"]
    if (not isinstance(children, list)
            or any(not isinstance(i, str) or len(i) != 64 or any(c not in _SHA for c in i)
                   for i in children)
            or len(children) != len(set(children))
            or len(children) > MAX_OPERATION_CHILDREN):
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
    if "operation" in payload:
        if not isinstance(children, list) or len(children) != 1 or denial is not None:
            out.append("children: a single-stage operation settles exactly one child")
    elif isinstance(children, list) and children and payload["outcome"] == "prerequisite" \
            and (len(children) < len(ROLES) or denial is not None):
        out.append("outcome: prerequisite requires every role and no denial")
    try:
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        out.append(f"payload: non-canonical JSON value: {exc}")
    return out


__all__ = [
    "ENROLLMENT_REF_SCHEMA", "INTENT_RESOLVED_SCHEMA", "SINGLE_STAGE_ROLE",
    "SINGLE_STAGE_OPERATIONS", "single_stage", "ordered_roles", "make_enrollment_ref",
    "check_enrollment_ref", "make_intent_resolution", "validate_intent_resolution",
    "CHILD_INTENT_SCHEMA", "make_child_intent", "validate_child_intent",
    "CHILD_ACCOUNTED_SCHEMA", "CHILD_ACCOUNTED_SCHEMA_V2", "MAX_OPERATION_CHILDREN",
    "child_attempt", "SETTLED_V2_SCHEMA", "SETTLEMENT_REQUEST_V2_SCHEMA",
    "OPERATIONS", "ROLES", "DISPOSITIONS", "SETTLEMENT_OUTCOMES",
    "OperationAccountingRefused", "child_digest", "make_child_record",
    "validate_child_record", "validate_child_chain", "operation_children",
    "check_outcome_against_children", "validate_settled_v2"]
