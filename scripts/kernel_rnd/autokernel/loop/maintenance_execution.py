"""Bounded orchestration for retention maintenance ownership.

This module defines the executable state machine around ``retention_consumer``.
``CampaignMaintenanceBackend`` binds it to the actual controller transactions and
``BrokerHoldProvider`` is the hold/refresh/settlement interface the real OP-41
broker (AKU-11) implements. No broker is registered or constructed by default, so
default execution remains unavailable. Filesystem and provider calls stay outside
controller transactions; ``BoundedHoldProvider`` adds per-call deadlines.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import os
import threading
from typing import Any, Mapping

from . import retention_consumer as consumer


LEGACY_EVENT_SCHEMA = "epyc.autokernel.maintenance_execution_event.v1"
EVENT_SCHEMA = "epyc.autokernel.maintenance_execution_event.v2"
SETTLED_EVENT_SCHEMA = "epyc.autokernel.maintenance_execution_event.v3"
SETTLEMENT_SCHEMA = "epyc.autokernel.maintenance_hold_settlement.v1"
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


class ProviderDeadlineExceeded(MaintenanceExecutionRefused):
    """A broker call outlived its deadline; its outcome is unknown."""


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
class HoldSettlement:
    """Broker-issued exactly-once settlement of one hold, with exact held time."""

    provider_id: str
    hold_id: str
    hold_receipt_digest: str
    token_digest: str
    cost: MaintenanceCost
    disposition: str
    accounting_receipt_digest: str
    acquired_at_ms: int
    settled_at_ms: int
    held_ms: int
    refresh_count: int
    receipt_digest: str = ""
    schema: str = SETTLEMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SETTLEMENT_SCHEMA:
            raise MaintenanceExecutionRefused("unsupported hold settlement schema")
        _text(self.provider_id, "provider_id")
        _text(self.hold_id, "hold_id")
        if self.accounting().receipt_digest != self.accounting_receipt_digest:
            raise MaintenanceExecutionRefused("hold settlement accounting digest is misbound")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (self.acquired_at_ms, self.settled_at_ms, self.held_ms,
                             self.refresh_count)):
            raise MaintenanceExecutionRefused("hold settlement times must be nonnegative ints")
        if self.held_ms != self.settled_at_ms - self.acquired_at_ms:
            raise MaintenanceExecutionRefused("held time differs from the acquire/settle interval")
        expected = _digest(self.body())
        if self.receipt_digest and self.receipt_digest != expected:
            raise MaintenanceExecutionRefused("hold settlement digest mismatch")
        object.__setattr__(self, "receipt_digest", expected)

    def accounting(self) -> AccountingReceipt:
        return AccountingReceipt(self.hold_receipt_digest, self.token_digest, self.cost,
                                 self.disposition)

    def body(self) -> dict[str, Any]:
        return {name: (self.cost.to_dict() if name == "cost" else getattr(self, name))
                for name in self.__dataclass_fields__ if name != "receipt_digest"}

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HoldSettlement":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise MaintenanceExecutionRefused("hold settlement has missing/unknown fields")
        return cls(**{**value, "cost": MaintenanceCost.from_dict(value["cost"])})


def _settlement_from(value: Any) -> "HoldSettlement | NoHoldReceipt":
    schema = value.get("schema") if isinstance(value, Mapping) else None
    if schema == SETTLEMENT_SCHEMA:
        return HoldSettlement.from_dict(value)
    if schema == NO_HOLD_SCHEMA:
        return NoHoldReceipt.from_dict(value)
    raise MaintenanceExecutionRefused("hold settlement schema is invalid")


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
                       else _EVENT_FIELDS | {"settlement"} if schema == SETTLED_EVENT_SCHEMA
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
    # A chained recovery token completes with its predecessor's broker settlement.
    bound = token.token_digest
    if (row.get("settlement") or {}).get("token_digest") == token.predecessor_token_digest \
            and event == "COMPLETED" and token.predecessor_token_digest is not None:
        bound = token.predecessor_token_digest
    if event == "COMPLETED":
        expected = AccountingReceipt(
            hold.receipt_digest, bound, cost, "complete").receipt_digest
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
    if row.get("settlement") is not None:
        settlement = _settlement_from(row["settlement"])
        row["settlement"] = settlement.to_dict()
        if event == "INTENT":
            if token.predecessor_token_digest is None or (
                    isinstance(settlement, HoldSettlement)
                    and (settlement.token_digest != token.predecessor_token_digest
                         or settlement.disposition != "aborted")):
                raise MaintenanceExecutionRefused(
                    "only a recovery INTENT may carry its predecessor's aborted settlement")
        elif (event not in {"COMPLETED", "ABORTED"}
              or not isinstance(settlement, HoldSettlement) or hold is None
              or settlement.hold_receipt_digest != hold.receipt_digest
              or settlement.token_digest != bound
              or settlement.accounting_receipt_digest != accounting
              or settlement.cost != cost):
            raise MaintenanceExecutionRefused(f"{event} hold settlement is misbound")
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
        "UNRESOLVED": {"INTENT", "COMPLETED"},
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
        settled = (state.owned and event == "COMPLETED" and token != state.token
                   and row.get("settlement") is not None and row["hold"] == (
                       None if state.hold is None else
                       {**state.hold.body(), "receipt_digest": state.hold.receipt_digest}))
        prior = (_settlement_from(row["settlement"])
                 if recovering and row.get("settlement") is not None else None)
        held = state.hold
        if (isinstance(prior, NoHoldReceipt) and (
                held is not None or prior.request_digest != _request_digest(state.token))) \
                or (isinstance(prior, HoldSettlement) and held is not None
                    and (prior.provider_id, prior.hold_id) != (held.provider_id, held.hold_id)):
            raise MaintenanceExecutionRefused(
                "recovery settlement does not resolve the unresolved provider hold")
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
            if (not (recovering or settled)
                    or token.predecessor_token_digest != previous.token_digest
                    or any(getattr(token, name) != getattr(previous, name)
                           for name in semantic)):
                raise MaintenanceExecutionRefused("owned maintenance token changed")
        hold = HoldReceipt.from_dict(row["hold"]) if row["hold"] is not None else None
        if hold is not None:
            _validate_hold(hold, state.token if settled else token, previous=state.hold)
        cost = MaintenanceCost.from_dict(row["cost"]) if row["cost"] is not None else None
        if event == "COMPLETED" and cost != state.cost:
            raise MaintenanceExecutionRefused(
                "COMPLETED cost differs from IO_COMPLETE")
        if state.phase == "UNRESOLVED" and event == "COMPLETED" and not settled:
            raise MaintenanceExecutionRefused(
                "UNRESOLVED completes only with the broker's settlement of its own hold")
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
               reason: str | None = None,
               settlement: HoldSettlement | NoHoldReceipt | None = None) -> dict[str, Any]:
    """Emit v2 bytes unless a broker settlement is carried (additive v3)."""
    extra = {} if settlement is None else {"settlement": settlement.to_dict()}
    return validate_event({
        **extra, "schema": SETTLED_EVENT_SCHEMA if extra else EVENT_SCHEMA,
        "event": event, "token": token.to_dict(),
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
    lease: consumer.MaintenanceLease | None
    recovered: HoldSettlement | None = None  # broker had already settled "complete"


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


class BrokerHoldProvider(MaintenanceHoldProvider):
    """Interface the real OP-41 broker (AKU-11) implements; unavailable by default.

    ``settle`` is exactly-once per hold: repeating it with the same receipt, cost
    and disposition returns the identical settlement, anything else refuses.
    ``resolve`` settles as aborted, or proves absent, an unresolved predecessor's hold.
    """

    def settle(self, hold: HoldReceipt, token: ExclusionToken, cost: MaintenanceCost,
               disposition: str) -> HoldSettlement:
        raise MaintenanceExecutionRefused("maintenance hold broker is unavailable")

    def resolve(self, token: ExclusionToken, hold: HoldReceipt | None) \
            -> HoldSettlement | NoHoldReceipt:
        raise MaintenanceExecutionRefused("maintenance hold broker is unavailable")

    def finish(self, hold, token, cost, disposition) -> AccountingReceipt:
        return self.settle(hold, token, cost, disposition).accounting()


class BoundedHoldProvider(BrokerHoldProvider):
    """Run each broker call off the controller mutex under a wall-clock deadline.

    An overdue call raises ``ProviderDeadlineExceeded`` and is left running on its
    daemon thread; the broker's exactly-once settlement must win over it.
    """

    def __init__(self, inner: BrokerHoldProvider, *, mutex, deadline_s: float):
        if not isinstance(inner, BrokerHoldProvider) or not 0 < deadline_s <= 300:
            raise MaintenanceExecutionRefused("bounded provider needs a broker and 0..300s")
        self.inner, self.mutex, self.deadline_s = inner, mutex, deadline_s

    def _call(self, name: str, *args):
        if self.mutex._is_owned():
            raise MaintenanceExecutionRefused(f"provider {name} called under controller mutex")
        box: list = []

        def run():
            try:
                box.append((True, getattr(self.inner, name)(*args)))
            except BaseException as exc:  # relayed to the caller below
                box.append((False, exc))

        worker = threading.Thread(target=run, name=f"maintenance-{name}", daemon=True)
        worker.start()
        worker.join(self.deadline_s)
        if not box:
            raise ProviderDeadlineExceeded(f"provider {name} exceeded {self.deadline_s}s")
        if not box[0][0]:
            raise box[0][1]
        return box[0][1]

    def acquire(self, token):
        return self._call("acquire", token)

    def refresh(self, hold, token):
        return self._call("refresh", hold, token)

    def settle(self, hold, token, cost, disposition):
        return self._call("settle", hold, token, cost, disposition)

    def resolve(self, token, hold):
        return self._call("resolve", token, hold)


def native_tombstones(journal, view, selected) -> tuple[consumer.NativeTombstone, ...]:
    """Bounded replay input: journaled native tombstones of the selected artifacts."""
    from .. import journal as journal_module
    wanted = {consumer.storage.tombstone_id(
        node.expiry.campaign_id, os.path.realpath(node.path), node.expiry.sha256,
        node.expiry.expirable_kind, node.expiry.rule_id)
        for node in view.nodes if node.artifact_id in selected and node.expiry and node.path}
    return tuple(
        consumer.NativeTombstone(entry.event_id, {
            key: value for key, value in entry.payload.items()
            if key not in {"storage_class", "path"}})
        for entry in journal.read_all()
        if entry.kind == journal_module.KIND_TOMBSTONE
        and entry.payload.get("tombstone_id") in wanted)


class CampaignMaintenanceBackend(ControllerMaintenanceBackend):
    """Concrete backend: every method is one short ``CampaignController`` transaction.

    Admitting over an UNRESOLVED exclusion first has the broker resolve the
    predecessor's hold, outside the mutex, so recovery never holds two holds.
    """

    def __init__(self, controller, prepared, policy, provider=None):
        from .native_retention_catalog import PreparedCatalogView
        if not isinstance(prepared, PreparedCatalogView):
            raise TypeError("prepared must be a native PreparedCatalogView")
        self.controller, self.view, self.policy = controller, prepared.view, policy
        self.provider = provider or BrokerHoldProvider()

    def _lease(self, token: ExclusionToken) -> consumer.MaintenanceLease:
        if consumer.collect_native_snapshot(self.view).snapshot_digest != token.snapshot_digest:
            raise MaintenanceExecutionRefused("prepared native view differs from the token")
        return consumer.MaintenanceLease(
            self.view, self.policy, None, f"controller:{token.token_digest}",
            native_tombstones(self.controller._journal, self.view,
                              token.selected_artifact_ids))

    def admit(self, job):
        if type(self.provider) in {MaintenanceHoldProvider, BrokerHoldProvider}:
            raise MaintenanceExecutionRefused("maintenance hold provider is unavailable")
        prior = self.controller.maintenance_recovery_owner()
        resolution = None if prior is None else self.provider.resolve(*prior)
        if prior is not None and not isinstance(resolution, (HoldSettlement, NoHoldReceipt)):
            raise MaintenanceExecutionRefused(
                "broker could not prove the unresolved hold settled or absent")
        if isinstance(resolution, HoldSettlement) and resolution.disposition == "complete":
            # The broker already settled this exclusion: journal it, never re-run.
            token = self.controller.maintenance_recover_completed(job, resolution)
            return Admission(token, None, resolution)
        token = self.controller.maintenance_admit(job, prior_settlement=resolution)
        return Admission(token, self._lease(token))

    def revalidate(self, token, hold):
        self.controller.maintenance_revalidate(token, hold)
        return self._lease(token)

    def append_tombstone(self, token, kind, payload, campaign_id):
        return self.controller.maintenance_append_tombstone(token, kind, payload, campaign_id)

    def io_complete(self, token, cost):
        self.controller.maintenance_io_complete(token, cost)

    def complete(self, token, accounting, settlement=None):
        self.controller.maintenance_complete(token, accounting, settlement)

    def abort(self, token, reason, receipt, hold=None, settlement=None):
        self.controller.maintenance_abort(token, reason, receipt, hold, settlement)

    def unresolved(self, token, reason):
        self.controller.maintenance_unresolved(token, reason)


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


def _settle(provider, hold: HoldReceipt, token: ExclusionToken, cost: MaintenanceCost,
            disposition: str) -> tuple[AccountingReceipt, HoldSettlement | None]:
    settle = getattr(provider, "settle", None)
    if settle is None:
        return _validate_accounting(provider.finish(hold, token, cost, disposition),
                                    hold, token, cost, disposition), None
    settlement = settle(hold, token, cost, disposition)
    if not isinstance(settlement, HoldSettlement) or (
            settlement.provider_id, settlement.hold_id) != (hold.provider_id, hold.hold_id):
        raise MaintenanceExecutionRefused(f"provider {disposition} settlement is misbound")
    return _validate_accounting(settlement.accounting(), hold, token, cost,
                                disposition), settlement


def _settled_call(method, settlement, *args) -> None:
    method(*args) if settlement is None else method(*args, settlement=settlement)


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
    retention: consumer.HeldRetentionResult | None  # None: recovered, nothing re-run
    token_digest: str
    accounting: AccountingReceipt
    cost: MaintenanceCost
    recovered: bool = False


def execute(job: consumer.RetentionJob, *, backend: ControllerMaintenanceBackend | None = None,
            provider: MaintenanceHoldProvider | None = None, now=None) \
        -> MaintenanceExecutionResult:
    """Reserve briefly, perform I/O unlocked, then finalize briefly."""
    if not isinstance(job, consumer.RetentionJob):
        raise TypeError("job must be a native RetentionJob")
    backend = backend or ControllerMaintenanceBackend()
    provider = provider or getattr(backend, "provider", None) or MaintenanceHoldProvider()
    if type(backend) is not ControllerMaintenanceBackend \
            and type(provider) in {MaintenanceHoldProvider, BrokerHoldProvider}:
        raise MaintenanceExecutionRefused("maintenance hold provider is unavailable")
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
    if admission.recovered is not None:
        return MaintenanceExecutionResult(None, token.token_digest,
                                          admission.recovered.accounting(),
                                          admission.recovered.cost, True)
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
        accounting, settlement = _settle(provider, proxy.hold, token, cost, "complete")
        accounting_done = True
        _settled_call(backend.complete, settlement, token, accounting)
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
                    aborted, settled = _settle(
                        provider, settlement_hold, token, aborted_cost, "aborted")
                except BaseException as settlement_exc:
                    try:
                        backend.unresolved(
                            token,
                            f"provider accounting/release is unresolved: {settlement_exc}",
                        )
                    except BaseException:
                        pass
                else:
                    _settled_call(backend.abort, settled, token, reason, aborted,
                                  settlement_hold)
        raise


__all__ = [
    "ACCOUNTING_SCHEMA", "Admission", "AccountingReceipt", "BoundedHoldProvider",
    "BrokerHoldProvider", "CampaignMaintenanceBackend", "ControllerMaintenanceBackend",
    "EVENT_SCHEMA", "EVENTS", "ExclusionToken", "HOLD_SCHEMA", "HoldReceipt",
    "HoldSettlement", "LEGACY_EVENT_SCHEMA", "ProviderDeadlineExceeded",
    "SETTLED_EVENT_SCHEMA", "SETTLEMENT_SCHEMA", "native_tombstones",
    "MaintenanceCost", "MaintenanceExecutionRefused", "MaintenanceExecutionResult",
    "MaintenanceHoldProvider", "MaintenanceState", "NO_HOLD_SCHEMA", "NoHoldReceipt",
    "TOKEN_SCHEMA", "execute",
    "make_event", "project_events", "validate_event",
]
