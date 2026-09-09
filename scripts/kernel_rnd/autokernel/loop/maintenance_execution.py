"""Bounded orchestration for retention maintenance ownership.

This module defines the executable state machine around ``retention_consumer``.
The controller exposes the durable exclusion transitions consumed by a backend,
but no concrete ``ControllerMaintenanceBackend``, complete fresh catalog, or real
resource provider is released yet, so default execution remains unavailable.
Expensive filesystem and provider calls are deliberately outside controller
admission transactions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import math
from typing import Any, Mapping

from . import retention_consumer as consumer


LEGACY_EVENT_SCHEMA = "epyc.autokernel.maintenance_execution_event.v1"
EVENT_SCHEMA = "epyc.autokernel.maintenance_execution_event.v2"
TOKEN_SCHEMA = "epyc.autokernel.maintenance_exclusion_token.v1"
HOLD_SCHEMA = "epyc.autokernel.maintenance_hold_receipt.v1"
ACCOUNTING_SCHEMA = "epyc.autokernel.maintenance_accounting_receipt.v1"
NO_HOLD_SCHEMA = "epyc.autokernel.maintenance_no_hold_receipt.v1"
EVENTS = frozenset({
    "INTENT", "PROVIDER_HELD", "MUTATION_REVALIDATED", "IO_COMPLETE",
    "COMPLETED", "ABORTED", "UNRESOLVED",
})
MAX_SELECTION = 64
_EVENT_FIELDS = {
    "schema", "event", "token", "hold", "cost", "accounting_receipt_digest",
    "abort_receipt", "reason", "occurred_at",
}
_LEGACY_EVENT_FIELDS = _EVENT_FIELDS - {"abort_receipt"}


class MaintenanceExecutionRefused(RuntimeError):
    """The current owner/authority binding cannot safely advance."""


def _iso(value: Any, label: str) -> str:
    value = _text(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MaintenanceExecutionRefused(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise MaintenanceExecutionRefused(f"{label} must include timezone")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MaintenanceExecutionRefused(f"{label} must be non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise MaintenanceExecutionRefused(f"{label} must be lowercase SHA-256")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ExclusionToken:
    token_id: str
    campaign_id: str
    config_digest: str
    config_generation: int
    supervisor_id: str
    supervisor_incarnation: int
    snapshot_id: str
    snapshot_generation: int
    snapshot_digest: str
    plan_digest: str
    policy_digest: str
    selected_artifact_ids: tuple[str, ...]
    admitted_at: str
    predecessor_token_digest: str | None = None
    token_digest: str = ""
    schema: str = TOKEN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TOKEN_SCHEMA:
            raise MaintenanceExecutionRefused("unsupported exclusion token schema")
        for name in ("token_id", "campaign_id", "supervisor_id", "snapshot_id",
                     "admitted_at"):
            _text(getattr(self, name), name)
        _iso(self.admitted_at, "admitted_at")
        for name in ("config_digest", "snapshot_digest", "plan_digest", "policy_digest"):
            _sha(getattr(self, name), name)
        if self.predecessor_token_digest is not None:
            _sha(self.predecessor_token_digest, "predecessor_token_digest")
        for name in ("config_generation", "supervisor_incarnation",
                     "snapshot_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MaintenanceExecutionRefused(f"{name} must be a positive integer")
        if (not self.selected_artifact_ids
                or len(self.selected_artifact_ids) > MAX_SELECTION
                or len(set(self.selected_artifact_ids)) != len(self.selected_artifact_ids)
                or any(not isinstance(item, str) or not item
                       for item in self.selected_artifact_ids)):
            raise MaintenanceExecutionRefused("token selection must contain 1..64 unique IDs")
        expected = _digest(self.body())
        if self.token_digest and self.token_digest != expected:
            raise MaintenanceExecutionRefused("exclusion token digest mismatch")
        object.__setattr__(self, "token_digest", expected)

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "token_id": self.token_id,
                "campaign_id": self.campaign_id, "config_digest": self.config_digest,
                "config_generation": self.config_generation,
                "supervisor_id": self.supervisor_id,
                "supervisor_incarnation": self.supervisor_incarnation,
                "snapshot_id": self.snapshot_id,
                "snapshot_generation": self.snapshot_generation,
                "snapshot_digest": self.snapshot_digest, "plan_digest": self.plan_digest,
                "policy_digest": self.policy_digest,
                "selected_artifact_ids": list(self.selected_artifact_ids),
                "admitted_at": self.admitted_at,
                "predecessor_token_digest": self.predecessor_token_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExclusionToken":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise MaintenanceExecutionRefused("exclusion token has missing/unknown fields")
        row = dict(value)
        row["selected_artifact_ids"] = tuple(row["selected_artifact_ids"])
        return cls(**row)

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "token_digest": self.token_digest}


@dataclass(frozen=True, slots=True)
class HoldReceipt:
    provider_id: str
    hold_id: str
    request_digest: str
    provider_generation: int
    accounting_epoch: int
    deadline: float
    current: bool
    revoked: bool
    receipt_digest: str = ""
    schema: str = HOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HOLD_SCHEMA:
            raise MaintenanceExecutionRefused("unsupported hold receipt schema")
        for name in ("provider_id", "hold_id"):
            _text(getattr(self, name), name)
        _sha(self.request_digest, "request_digest")
        for name in ("provider_generation", "accounting_epoch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MaintenanceExecutionRefused(f"{name} must be positive")
        if (isinstance(self.deadline, bool) or not isinstance(self.deadline, (int, float))
                or not math.isfinite(float(self.deadline)) or self.deadline <= 0):
            raise MaintenanceExecutionRefused("hold deadline must be finite and positive")
        if type(self.current) is not bool or type(self.revoked) is not bool:
            raise MaintenanceExecutionRefused("hold status must be boolean")
        expected = _digest(self.body())
        if self.receipt_digest and self.receipt_digest != expected:
            raise MaintenanceExecutionRefused("hold receipt digest mismatch")
        object.__setattr__(self, "receipt_digest", expected)

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "provider_id": self.provider_id,
                "hold_id": self.hold_id, "request_digest": self.request_digest,
                "provider_generation": self.provider_generation,
                "accounting_epoch": self.accounting_epoch, "deadline": self.deadline,
                "current": self.current, "revoked": self.revoked}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HoldReceipt":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise MaintenanceExecutionRefused("hold receipt has missing/unknown fields")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class NoHoldReceipt:
    provider_id: str
    request_digest: str
    provider_generation: int
    refused_at: str
    reason: str
    receipt_digest: str = ""
    schema: str = NO_HOLD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NO_HOLD_SCHEMA:
            raise MaintenanceExecutionRefused("unsupported no-hold receipt schema")
        _text(self.provider_id, "provider_id")
        _sha(self.request_digest, "request_digest")
        if (isinstance(self.provider_generation, bool)
                or not isinstance(self.provider_generation, int)
                or self.provider_generation < 1):
            raise MaintenanceExecutionRefused("provider_generation must be positive")
        _iso(self.refused_at, "refused_at")
        _text(self.reason, "reason")
        expected = _digest(self.body())
        if self.receipt_digest and self.receipt_digest != expected:
            raise MaintenanceExecutionRefused("no-hold receipt digest mismatch")
        object.__setattr__(self, "receipt_digest", expected)

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "provider_id": self.provider_id,
                "request_digest": self.request_digest,
                "provider_generation": self.provider_generation,
                "refused_at": self.refused_at, "reason": self.reason}

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NoHoldReceipt":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise MaintenanceExecutionRefused("no-hold receipt has missing/unknown fields")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class MaintenanceCost:
    artifact_count: int
    reclaimed_bytes: int
    deleted_artifact_count: int
    deleted_bytes_this_attempt: int

    @classmethod
    def from_result(cls, result: consumer.HeldRetentionResult) -> "MaintenanceCost":
        outcomes = result.outcomes
        return cls(len(outcomes), sum(item.measured_size_bytes for item in outcomes),
                   sum(item.deleted for item in outcomes),
                   sum(item.measured_size_bytes for item in outcomes if item.deleted))

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MaintenanceCost":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise MaintenanceExecutionRefused("maintenance cost has missing/unknown fields")
        result = cls(**dict(value))
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0
               for item in result.to_dict().values()):
            raise MaintenanceExecutionRefused("maintenance costs must be nonnegative integers")
        return result


@dataclass(frozen=True, slots=True)
class AccountingReceipt:
    hold_receipt_digest: str
    token_digest: str
    cost: MaintenanceCost
    disposition: str
    receipt_digest: str = ""
    schema: str = ACCOUNTING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ACCOUNTING_SCHEMA or self.disposition not in {"complete", "aborted"}:
            raise MaintenanceExecutionRefused("invalid accounting receipt")
        _sha(self.hold_receipt_digest, "hold_receipt_digest")
        _sha(self.token_digest, "token_digest")
        if not isinstance(self.cost, MaintenanceCost):
            raise TypeError("cost must be MaintenanceCost")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (self.cost.artifact_count, self.cost.reclaimed_bytes,
                             self.cost.deleted_artifact_count,
                             self.cost.deleted_bytes_this_attempt)):
            raise MaintenanceExecutionRefused("maintenance costs must be nonnegative integers")
        expected = _digest(self.body())
        if self.receipt_digest and self.receipt_digest != expected:
            raise MaintenanceExecutionRefused("accounting receipt digest mismatch")
        object.__setattr__(self, "receipt_digest", expected)

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "hold_receipt_digest": self.hold_receipt_digest,
                "token_digest": self.token_digest,
                "cost": {"artifact_count": self.cost.artifact_count,
                         "reclaimed_bytes": self.cost.reclaimed_bytes,
                         "deleted_artifact_count": self.cost.deleted_artifact_count,
                         "deleted_bytes_this_attempt": self.cost.deleted_bytes_this_attempt},
                "disposition": self.disposition}

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AccountingReceipt":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise MaintenanceExecutionRefused(
                "accounting receipt has missing/unknown fields")
        row = dict(value)
        row["cost"] = MaintenanceCost.from_dict(row["cost"])
        return cls(**row)


@dataclass(frozen=True, slots=True)
class MaintenanceState:
    phase: str | None = None
    token: ExclusionToken | None = None
    hold: HoldReceipt | None = None
    cost: MaintenanceCost | None = None
    accounting_receipt_digest: str | None = None
    abort_receipt_digest: str | None = None
    reason: str | None = None
    last_event_digest: str | None = None

    @property
    def owned(self) -> bool:
        return self.phase not in {None, "COMPLETED", "ABORTED"}


def validate_event(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one closed durable maintenance transition."""
    if not isinstance(value, Mapping):
        raise MaintenanceExecutionRefused("maintenance event has missing/unknown fields")
    row = dict(value)
    schema = row.get("schema")
    expected_fields = (_EVENT_FIELDS if schema == EVENT_SCHEMA
                       else _LEGACY_EVENT_FIELDS if schema == LEGACY_EVENT_SCHEMA
                       else None)
    if expected_fields is None or set(row) != expected_fields:
        raise MaintenanceExecutionRefused("maintenance event has missing/unknown fields")
    if row["event"] not in EVENTS:
        raise MaintenanceExecutionRefused("maintenance event schema/event is invalid")
    if schema == LEGACY_EVENT_SCHEMA:
        if row["event"] == "ABORTED":
            raise MaintenanceExecutionRefused(
                "legacy ABORTED event lacks provider settlement evidence")
    token = ExclusionToken.from_dict(row["token"])
    row["token"] = token.to_dict()
    hold = row["hold"]
    if hold is not None:
        hold = HoldReceipt.from_dict(hold)
        row["hold"] = {**hold.body(), "receipt_digest": hold.receipt_digest}
    cost = row["cost"]
    if cost is not None:
        cost = MaintenanceCost.from_dict(cost)
        row["cost"] = cost.to_dict()
    accounting = row["accounting_receipt_digest"]
    if accounting is not None:
        _sha(accounting, "accounting_receipt_digest")
    abort_receipt = row.get("abort_receipt")
    if abort_receipt is not None:
        if not isinstance(abort_receipt, Mapping):
            raise MaintenanceExecutionRefused("abort receipt must be a closed receipt")
        if abort_receipt.get("schema") == ACCOUNTING_SCHEMA:
            abort_receipt = AccountingReceipt.from_dict(abort_receipt)
        elif abort_receipt.get("schema") == NO_HOLD_SCHEMA:
            abort_receipt = NoHoldReceipt.from_dict(abort_receipt)
        else:
            raise MaintenanceExecutionRefused("abort receipt schema is invalid")
        row["abort_receipt"] = abort_receipt.to_dict()
    reason = row["reason"]
    if reason is not None:
        _text(reason, "reason")
    _iso(row["occurred_at"], "occurred_at")
    event = row["event"]
    if event == "INTENT" and any(
            item is not None for item in (hold, cost, accounting, abort_receipt, reason)):
        raise MaintenanceExecutionRefused("INTENT must not carry hold/cost/result fields")
    if event in {"PROVIDER_HELD", "MUTATION_REVALIDATED"} \
            and (hold is None or any(
                item is not None for item in (cost, accounting, abort_receipt, reason))):
        raise MaintenanceExecutionRefused(f"{event} requires only a hold receipt")
    if event == "IO_COMPLETE" and (hold is None or cost is None
                                    or accounting is not None or abort_receipt is not None
                                    or reason is not None):
        raise MaintenanceExecutionRefused("IO_COMPLETE requires hold and cost only")
    if event == "COMPLETED" and (hold is None or cost is None or accounting is None
                                 or abort_receipt is not None or reason is not None):
        raise MaintenanceExecutionRefused("COMPLETED requires hold, cost and accounting")
    if event == "COMPLETED":
        expected = AccountingReceipt(
            hold.receipt_digest, token.token_digest, cost, "complete").receipt_digest
        if accounting != expected:
            raise MaintenanceExecutionRefused(
                "COMPLETED accounting receipt digest is misbound")
    if event == "UNRESOLVED" and (
            reason is None or accounting is not None or abort_receipt is not None):
        raise MaintenanceExecutionRefused("UNRESOLVED requires a reason and no settlement")
    if event == "ABORTED":
        if reason is None or abort_receipt is None:
            raise MaintenanceExecutionRefused(
                "ABORTED requires a reason and provider-owned receipt")
        if isinstance(abort_receipt, AccountingReceipt):
            zero = MaintenanceCost(0, 0, 0, 0)
            if hold is None or cost != zero or accounting != abort_receipt.receipt_digest:
                raise MaintenanceExecutionRefused(
                    "settled ABORTED requires exact hold, zero cost and receipt digest")
            _validate_accounting(abort_receipt, hold, token, zero, "aborted")
        elif (hold is not None or cost is not None or accounting is not None):
            raise MaintenanceExecutionRefused(
                "no-hold ABORTED must carry only its refusal receipt")
        else:
            _validate_no_hold(abort_receipt, token)
    return row


def project_events(values) -> MaintenanceState:
    """Project one indexed exclusion with exact-retry and transition checks."""
    state = MaintenanceState()
    prior_row: dict[str, Any] | None = None
    allowed = {
        "INTENT": {"INTENT", "PROVIDER_HELD", "ABORTED", "UNRESOLVED"},
        "PROVIDER_HELD": {
            "INTENT", "MUTATION_REVALIDATED", "IO_COMPLETE", "ABORTED", "UNRESOLVED",
        },
        "MUTATION_REVALIDATED": {
            "INTENT", "MUTATION_REVALIDATED", "IO_COMPLETE", "ABORTED", "UNRESOLVED",
        },
        "IO_COMPLETE": {"INTENT", "COMPLETED", "UNRESOLVED"},
        "COMPLETED": {"INTENT"},
        "ABORTED": {"INTENT"},
        "UNRESOLVED": {"INTENT"},
    }
    for value in values:
        row = validate_event(value)
        digest = _digest(row)
        if prior_row is not None and row == prior_row:
            continue
        event = row["event"]
        token = ExclusionToken.from_dict(row["token"])
        if state.phase is None:
            if event != "INTENT":
                raise MaintenanceExecutionRefused("maintenance history must begin with INTENT")
            if token.predecessor_token_digest is not None:
                raise MaintenanceExecutionRefused("initial maintenance token has a predecessor")
        elif event not in allowed[state.phase]:
            raise MaintenanceExecutionRefused(
                f"maintenance transition {state.phase}->{event} is invalid")
        recovering = state.owned and event == "INTENT"
        restarting = state.phase in {"COMPLETED", "ABORTED"} and event == "INTENT"
        if recovering and token == state.token:
            raise MaintenanceExecutionRefused(
                "maintenance recovery requires a fresh chained token")
        if restarting and (token == state.token
                           or token.predecessor_token_digest is not None):
            raise MaintenanceExecutionRefused(
                "new maintenance intent must begin a fresh token chain")
        if state.owned and state.token is not None and token != state.token:
            previous = state.token
            semantic = (
                "campaign_id", "config_digest", "config_generation", "snapshot_id",
                "snapshot_generation", "snapshot_digest", "plan_digest", "policy_digest",
                "selected_artifact_ids",
            )
            if (not recovering
                    or token.predecessor_token_digest != previous.token_digest
                    or any(getattr(token, name) != getattr(previous, name)
                           for name in semantic)):
                raise MaintenanceExecutionRefused("owned maintenance token changed")
        hold = HoldReceipt.from_dict(row["hold"]) if row["hold"] is not None else None
        if hold is not None:
            _validate_hold(hold, token, previous=state.hold)
        cost = MaintenanceCost.from_dict(row["cost"]) if row["cost"] is not None else None
        if event == "COMPLETED" and cost != state.cost:
            raise MaintenanceExecutionRefused(
                "COMPLETED cost differs from IO_COMPLETE")
        starting = event == "INTENT"
        state = MaintenanceState(
            phase=event, token=token,
            hold=None if starting else (hold or state.hold),
            cost=None if starting else (cost or state.cost),
            accounting_receipt_digest=row["accounting_receipt_digest"],
            abort_receipt_digest=(row.get("abort_receipt", {}).get("receipt_digest")
                                  if row.get("abort_receipt") is not None else None),
            reason=row["reason"], last_event_digest=digest)
        prior_row = row
    return state


def make_event(event: str, token: ExclusionToken, *, occurred_at: str,
               hold: HoldReceipt | None = None, cost: MaintenanceCost | None = None,
               accounting_receipt_digest: str | None = None,
               abort_receipt: AccountingReceipt | NoHoldReceipt | None = None,
               reason: str | None = None) -> dict[str, Any]:
    return validate_event({
        "schema": EVENT_SCHEMA, "event": event, "token": token.to_dict(),
        "hold": ({**hold.body(), "receipt_digest": hold.receipt_digest}
                 if hold is not None else None),
        "cost": cost.to_dict() if cost is not None else None,
        "accounting_receipt_digest": accounting_receipt_digest,
        "abort_receipt": abort_receipt.to_dict() if abort_receipt is not None else None,
        "reason": reason, "occurred_at": occurred_at,
    })


@dataclass(frozen=True, slots=True)
class Admission:
    token: ExclusionToken
    lease: consumer.MaintenanceLease


class ControllerMaintenanceBackend:
    """Release seam for short controller-owned transactions; unavailable today."""

    def admit(self, job: consumer.RetentionJob) -> Admission:
        raise MaintenanceExecutionRefused("controller maintenance transaction is unavailable")

    def revalidate(self, token: ExclusionToken, hold: HoldReceipt) \
            -> consumer.MaintenanceLease:
        raise MaintenanceExecutionRefused("controller maintenance revalidation is unavailable")

    def append_tombstone(self, token: ExclusionToken, kind: str,
                         payload: Mapping[str, Any], campaign_id: str | None) -> Any:
        raise MaintenanceExecutionRefused("controller maintenance journal is unavailable")

    def io_complete(self, token: ExclusionToken, cost: MaintenanceCost) -> None:
        raise MaintenanceExecutionRefused("controller maintenance completion is unavailable")

    def complete(self, token: ExclusionToken, accounting: AccountingReceipt) -> None:
        raise MaintenanceExecutionRefused("controller maintenance completion is unavailable")

    def abort(self, token: ExclusionToken, reason: str,
              receipt: AccountingReceipt | NoHoldReceipt,
              hold: HoldReceipt | None = None) -> None:
        raise MaintenanceExecutionRefused("controller maintenance abort is unavailable")

    def unresolved(self, token: ExclusionToken, reason: str) -> None:
        raise MaintenanceExecutionRefused("controller maintenance recovery is unavailable")


class MaintenanceHoldProvider:
    """No existing provider implements maintenance holds; default is unavailable."""

    def acquire(self, token: ExclusionToken) -> HoldReceipt | NoHoldReceipt:
        raise MaintenanceExecutionRefused("maintenance hold provider is unavailable")

    def refresh(self, hold: HoldReceipt, token: ExclusionToken) -> HoldReceipt:
        raise MaintenanceExecutionRefused("maintenance hold provider is unavailable")

    def finish(self, hold: HoldReceipt, token: ExclusionToken, cost: MaintenanceCost,
               disposition: str) -> AccountingReceipt:
        raise MaintenanceExecutionRefused("maintenance accounting provider is unavailable")


class _JournalProxy:
    def __init__(self, backend: ControllerMaintenanceBackend,
                 provider: MaintenanceHoldProvider, token: ExclusionToken,
                 hold: HoldReceipt):
        self.backend, self.provider, self.token = backend, provider, token
        self.hold = hold
        self.intent_durable = False

    def append(self, kind: str, payload: Mapping[str, Any], *, campaign_id=None) -> Any:
        # Provider I/O first, outside the controller. Controller revalidation and
        # append are separate short transactions and never enclose hashing/removal.
        previous = self.hold
        refreshed = self.provider.refresh(previous, self.token)
        _validate_hold(refreshed, self.token, previous=previous)
        self.hold = refreshed
        self.backend.revalidate(self.token, self.hold)
        result = self.backend.append_tombstone(self.token, kind, payload, campaign_id)
        if payload.get("reclamation_state") == "intent":
            self.intent_durable = True
        return result


def _request_digest(token: ExclusionToken) -> str:
    return _digest({"campaign_id": token.campaign_id,
                    "config_digest": token.config_digest,
                    "token_digest": token.token_digest,
                    "plan_digest": token.plan_digest,
                    "policy_digest": token.policy_digest,
                    "selected_artifact_ids": list(token.selected_artifact_ids)})


def _validate_hold(hold: HoldReceipt, token: ExclusionToken,
                   *, previous: HoldReceipt | None = None) -> None:
    if not isinstance(hold, HoldReceipt):
        raise TypeError("provider returned no HoldReceipt")
    if hold.request_digest != _request_digest(token) or not hold.current or hold.revoked:
        raise MaintenanceExecutionRefused("maintenance hold is stale, revoked, or misbound")
    if previous is not None and (
            hold.provider_id != previous.provider_id
            or hold.hold_id != previous.hold_id
            or hold.provider_generation != previous.provider_generation
            or hold.accounting_epoch != previous.accounting_epoch):
        raise MaintenanceExecutionRefused("maintenance hold identity or generation changed")


def _validate_no_hold(receipt: NoHoldReceipt, token: ExclusionToken) -> None:
    if not isinstance(receipt, NoHoldReceipt):
        raise MaintenanceExecutionRefused(
            "provider returned neither a hold nor an explicit no-hold receipt")
    if receipt.request_digest != _request_digest(token):
        raise MaintenanceExecutionRefused("provider no-hold receipt is misbound")


def _validate_accounting(accounting: Any, hold: HoldReceipt, token: ExclusionToken,
                         cost: MaintenanceCost, disposition: str) -> AccountingReceipt:
    if (not isinstance(accounting, AccountingReceipt)
            or accounting.hold_receipt_digest != hold.receipt_digest
            or accounting.token_digest != token.token_digest
            or accounting.cost != cost or accounting.disposition != disposition):
        raise MaintenanceExecutionRefused(
            f"provider {disposition} accounting receipt is missing or misbound")
    return accounting


@dataclass(frozen=True, slots=True)
class MaintenanceExecutionResult:
    retention: consumer.HeldRetentionResult
    token_digest: str
    accounting: AccountingReceipt
    cost: MaintenanceCost


def execute(job: consumer.RetentionJob, *, backend: ControllerMaintenanceBackend | None = None,
            provider: MaintenanceHoldProvider | None = None, now=None) \
        -> MaintenanceExecutionResult:
    """Reserve briefly, perform I/O unlocked, then finalize briefly."""
    if not isinstance(job, consumer.RetentionJob):
        raise TypeError("job must be a native RetentionJob")
    backend = backend or ControllerMaintenanceBackend()
    provider = provider or MaintenanceHoldProvider()
    admission = backend.admit(job)
    if not isinstance(admission, Admission):
        raise TypeError("controller returned no Admission")
    token = admission.token
    if (token.snapshot_id != job.plan.snapshot_id
            or token.snapshot_generation != job.plan.snapshot_generation
            or token.snapshot_digest != job.plan.snapshot_digest
            or token.plan_digest != job.plan.plan_digest
            or token.policy_digest != job.policy_digest
            or token.selected_artifact_ids != job.selected_artifact_ids):
        backend.unresolved(token, "controller admission token is misbound to retention job")
        raise MaintenanceExecutionRefused("controller admission token is misbound to retention job")
    try:
        acquired = provider.acquire(token)
        if isinstance(acquired, NoHoldReceipt):
            _validate_no_hold(acquired, token)
        else:
            _validate_hold(acquired, token)
    except BaseException as exc:
        try:
            backend.unresolved(token, f"provider acquisition is unresolved: {exc}")
        except BaseException:
            pass
        raise
    if isinstance(acquired, NoHoldReceipt):
        try:
            backend.abort(token, f"provider explicitly refused hold: {acquired.reason}",
                          acquired)
        except BaseException as exc:
            try:
                backend.unresolved(
                    token, f"explicit no-hold settlement is unresolved: {exc}")
            except BaseException:
                pass
            raise
        raise MaintenanceExecutionRefused(
            f"maintenance hold explicitly refused: {acquired.reason}")
    hold = acquired

    proxy: _JournalProxy | None = None
    accounting_done = False
    try:
        lease = backend.revalidate(token, hold)
        proxy = _JournalProxy(backend, provider, token, hold)
        lease = replace(lease, journal=proxy)

        class HeldOwner(consumer.MaintenanceOwner):
            def held(self, operation):
                return operation(lease)

        result = consumer.execute(job, owner=HeldOwner(), now=now)
        cost = MaintenanceCost.from_result(result)
        backend.io_complete(token, cost)
        accounting = provider.finish(proxy.hold, token, cost, "complete")
        accounting = _validate_accounting(
            accounting, proxy.hold, token, cost, "complete")
        accounting_done = True
        backend.complete(token, accounting)
        return MaintenanceExecutionResult(result, token.token_digest, accounting, cost)
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        # Once a tombstone intent exists, ambiguous deletion/completion cannot
        # release the exclusion. Recovery must inspect the native Journal/bytes.
        if accounting_done or (proxy is not None and proxy.intent_durable):
            try:
                backend.unresolved(token, reason)
            except BaseException:
                pass
        else:
            # A provider hold must settle before controller exclusion release.  If
            # settlement is ambiguous, preserve the still-owned token for recovery.
            if not accounting_done:
                try:
                    aborted_cost = MaintenanceCost(0, 0, 0, 0)
                    settlement_hold = proxy.hold if proxy is not None else hold
                    aborted = provider.finish(
                        settlement_hold, token, aborted_cost, "aborted")
                    _validate_accounting(
                        aborted, settlement_hold, token, aborted_cost, "aborted")
                except BaseException as settlement_exc:
                    try:
                        backend.unresolved(
                            token,
                            f"provider accounting/release is unresolved: {settlement_exc}",
                        )
                    except BaseException:
                        pass
                else:
                    backend.abort(token, reason, aborted, settlement_hold)
        raise


__all__ = [
    "ACCOUNTING_SCHEMA", "Admission", "AccountingReceipt", "ControllerMaintenanceBackend",
    "EVENT_SCHEMA", "EVENTS", "ExclusionToken", "HOLD_SCHEMA", "HoldReceipt",
    "LEGACY_EVENT_SCHEMA",
    "MaintenanceCost", "MaintenanceExecutionRefused", "MaintenanceExecutionResult",
    "MaintenanceHoldProvider", "MaintenanceState", "NO_HOLD_SCHEMA", "NoHoldReceipt",
    "TOKEN_SCHEMA", "execute",
    "make_event", "project_events", "validate_event",
]
