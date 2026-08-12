"""Pure, release-local host/resource/storage preflight decisions.

The original release gate imported the deleted AK4 controller guard plane for
three small observations.  Restoring that dependency would also restore a
second autonomous state machine beside the lean sequencer.  This module keeps
the release boundary narrow: it reads nothing, writes nothing, starts nothing,
and returns only a three-way decision over caller-supplied receipts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .. import storage

CONTINUE = "CONTINUE"
STOP = "STOP"
COULD_NOT_EVALUATE = "COULD_NOT_EVALUATE"
HOST_UPTIME_CEILING_SECONDS = 7 * 24 * 60 * 60


class PreflightInputError(ValueError):
    """A release preflight input is malformed or self-contradictory."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise PreflightInputError(f"{label}: required, a non-empty NUL-free string")
    return value


def _timestamp(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PreflightInputError(f"{label}: invalid timestamp {text!r}") from exc
    if parsed.tzinfo is None:
        raise PreflightInputError(f"{label}: timestamp must carry a timezone")
    return parsed


@dataclass(frozen=True)
class Decision:
    outcome: str
    reason: str

    def __post_init__(self) -> None:
        if self.outcome not in (CONTINUE, STOP, COULD_NOT_EVALUATE):
            raise PreflightInputError(f"unknown preflight outcome {self.outcome!r}")
        _text(self.reason, "reason")


@dataclass(frozen=True)
class HostHealth:
    uptime_seconds: int
    observed_at: str
    receipt: str
    ceiling_seconds: int = HOST_UPTIME_CEILING_SECONDS
    observable: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.uptime_seconds, bool) or not isinstance(
                self.uptime_seconds, int) or self.uptime_seconds < 0:
            raise PreflightInputError("uptime_seconds: required, a non-negative int")
        _timestamp(self.observed_at, "observed_at")
        _text(self.receipt, "receipt")
        if isinstance(self.ceiling_seconds, bool) or not isinstance(
                self.ceiling_seconds, int) or self.ceiling_seconds <= 0:
            raise PreflightInputError("ceiling_seconds: required, a positive int")
        if self.ceiling_seconds > HOST_UPTIME_CEILING_SECONDS:
            raise PreflightInputError(
                "ceiling_seconds may be stricter than, but never looser than, "
                "the ratified seven-day ceiling")
        if not isinstance(self.observable, bool):
            raise PreflightInputError("observable: required, a bool")


@dataclass(frozen=True)
class ResourceClaimObservation:
    resource: str
    claim_kind: str
    acquired: bool
    observed_at: str
    receipt: Optional[str] = None
    held_by: Optional[str] = None
    unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.resource, "resource")
        _text(self.claim_kind, "claim_kind")
        _timestamp(self.observed_at, "observed_at")
        if not isinstance(self.acquired, bool):
            raise PreflightInputError("acquired: required, a bool")
        if self.acquired:
            _text(self.receipt, "receipt")
        else:
            _text(self.unavailable_reason, "unavailable_reason")
        if self.held_by is not None:
            _text(self.held_by, "held_by")


@dataclass(frozen=True)
class StorageObservation:
    path: str
    state: storage.StorageState
    expirable_backlog_bytes: int
    receipt: str

    def __post_init__(self) -> None:
        _text(self.path, "path")
        if not isinstance(self.state, storage.StorageState):
            raise PreflightInputError("state must be storage.StorageState")
        if isinstance(self.expirable_backlog_bytes, bool) or not isinstance(
                self.expirable_backlog_bytes, int) or self.expirable_backlog_bytes < 0:
            raise PreflightInputError(
                "expirable_backlog_bytes: required, a non-negative int")
        _text(self.receipt, "receipt")


def guard_host_uptime(host: HostHealth, *, owner: str,
                      escalation_deadline: str, now: str) -> Decision:
    if not isinstance(host, HostHealth):
        raise PreflightInputError("host must be HostHealth")
    _text(owner, "owner")
    if _timestamp(escalation_deadline, "escalation_deadline") <= _timestamp(now, "now"):
        raise PreflightInputError("escalation_deadline must be after now")
    if not host.observable:
        return Decision(COULD_NOT_EVALUATE, "host uptime could not be observed")
    if host.uptime_seconds < host.ceiling_seconds:
        return Decision(
            CONTINUE,
            f"uptime {host.uptime_seconds}s is under the {host.ceiling_seconds}s ceiling")
    return Decision(
        STOP,
        f"host uptime {host.uptime_seconds}s reached the {host.ceiling_seconds}s "
        "ceiling; no decision-grade measurement proceeds and a reboot is operator authority")


def guard_resource_available(observation: ResourceClaimObservation) -> Decision:
    if not isinstance(observation, ResourceClaimObservation):
        raise PreflightInputError("observation must be ResourceClaimObservation")
    if observation.acquired:
        return Decision(
            CONTINUE,
            f"{observation.claim_kind} claim on {observation.resource} is HELD "
            f"under receipt {observation.receipt}")
    return Decision(
        STOP,
        f"{observation.claim_kind} claim on {observation.resource} was not acquired: "
        f"{observation.unavailable_reason}. The release gate drains rather than "
        "inferring availability from idle sensing")


def guard_storage_headroom(observation: StorageObservation) -> Decision:
    if not isinstance(observation, StorageObservation):
        raise PreflightInputError("observation must be StorageObservation")
    state = observation.state
    numeric_pressure = state.free_bytes < state.floor_bytes
    if numeric_pressure != state.pressured:
        return Decision(
            COULD_NOT_EVALUATE,
            f"storage observation for {observation.path} is self-contradictory")
    if not state.pressured:
        return Decision(
            CONTINUE,
            f"free {state.free_bytes} bytes is at or above the "
            f"{state.floor_bytes}-byte floor on {observation.path}")
    reclaimable = state.free_bytes + observation.expirable_backlog_bytes
    if reclaimable >= state.floor_bytes:
        # T3 may not reclaim.  It blocks until the already-authorized expiry
        # plane has run, which is safer than the old controller REFUSE state
        # being accidentally treated as a release pass.
        return Decision(
            STOP,
            f"free {state.free_bytes} bytes is below the {state.floor_bytes}-byte "
            "floor; eligible expiry could clear it, but release evaluation cannot "
            "perform reclamation")
    return Decision(
        STOP,
        f"free {state.free_bytes} bytes plus eligible expiry "
        f"{observation.expirable_backlog_bytes} bytes does not clear the "
        f"{state.floor_bytes}-byte floor")
