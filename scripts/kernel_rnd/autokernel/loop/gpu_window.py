#!/usr/bin/env python3
"""G1 -- the MI210 window handover between AutoKernel and the stack-owned executor.

STATUS (2026-10-04, after the G1 review NACK, `tmp/g1-review-20261004/REVIEW.md`):
AutoKernel's half of the agreed contract. Nothing here sends a bus message, parks a
role, touches a server or writes the window file. The request/wait/release handshake
(`StackOwnerWindow`) stays a refusing stub until the executor ships; the payload
builders, the read-only grant check and the schedule reader are real.

Why a window at all
-------------------
Production `:8083` (Qwen3.8-27B + DFlash2, `architect_critic` and the roles sharing it)
holds most of the MI210's 64 GiB -- KVU-16h measured it at 59.77 GiB under
`speculative.n_max: 0` alternation (51.69 GiB before). An AutoKernel GPU batch cannot
co-reside, so it needs an exclusive window, and only the stack owner may stop and reload
a production server (reload-ownership rule, `agents/shared/OPERATING_CONSTRAINTS.md`).

Agreed sequence
---------------
1. **AK requests** a window as a `compute-request` in its OWN outbox, routed to the
   coordinator-daemon (BUS_PROTOCOL.md rule 11, `compute_policy.yaml`), from the
   `inference` consumer (AK campaigns run from it). The payload names task, device
   `gpu` / `mi210_0`, window {duration_s, expected_end}, est_h, release_condition,
   and the G1 fields: campaign, purpose, parked roles/ports, the 62 GiB peak ceiling
   and a `schedule_ref` that MUST resolve to an entry of the shared schedule file
   (`SCHEDULE_FILE`, `epyc.gpu_window_schedule.v1`) covering the requested window.
   `WindowRequest.to_bus_payload`. A bus message is not a credential: the executor is
   driven by the schedule + its own lease, never by the message alone.
2. **The stack-owned executor** (orchestrator) drains `:8083`, stops it, and is the ONLY
   writer of `mi210.json` (`written_by: gpu_window_executor`, a `window_id`,
   `expected_end <= since + 60 min`, `grant_state`: draining -> granted -> restoring).
   It restores production at expected_end (+ grace) even if AK is dead.
3. **AK claims, runs, releases**: it takes `mi210_0` (the device flock), and every GPU
   measurement takes the orchestrator's gpu-quiet lock EXCLUSIVE (`claim.hold_gpu_quiet_measurement`;
   never a CPU region claim around production traffic). At every GPU batch boundary
   (serial router) and every GPU measurement (run.py) `assert_window_active` re-reads the
   window file: a preempt request, a non-`granted` state, an expired end or a foreign
   writer ends AK's use of the device. The in-run residency sampler aborts the launch it
   started if VRAM passes `GPU_PEAK_CEILING_BYTES` (`residency.Sampler`, ceiling hook).
   Release = drop the device flock + append `release_message(...)` to AK's own outbox.
   **AK never writes `mi210.json`**; the executor sees release as the flock going free.
4. **The executor restores**: reload with the current compiled argv, serving proof,
   then `holder=production`.

Stack-owner constraints (workspace-ec, 2026-10-04) kept: max window 60 min unless the
operator approves (schedule `operator_approved`), one window at a time, never during a
backlog block unless scheduled, AK GPU runs at or under 62 GiB peak with in-run abort.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from .surface_validation import GPU_CAPACITY_CEILING_BYTES as GPU_PEAK_CEILING_BYTES

WINDOW_FILE = Path("/mnt/raid0/llm/tmp/gpu-window/mi210.json")
SCHEDULE_FILE = Path("/mnt/raid0/llm/tmp/gpu-window/schedule.json")
SCHEDULE_SCHEMA = "epyc.gpu_window_schedule.v1"
DEVICE_ID = "mi210_0"
EXECUTOR_WRITER = "gpu_window_executor"
GRANTED_STATE = "granted"
REQUEST_KIND = "compute-request"
RELEASE_KIND = "status"
COORDINATOR = "coordinator-daemon"
REQUESTING_CONSUMER = "inference"      # compute_policy.yaml: AK campaigns run from it
MAX_WINDOW_S = 3600                    # longer needs the operator
PARKED_ROLES = ("architect_critic",)
PARKED_PORTS = (8083,)


class WindowNotAgreed(NotImplementedError):
    """The live handshake stays refused until the stack-owned executor ships."""


class GpuWindowPreempted(RuntimeError):
    """The window no longer grants AutoKernel the device; stop using it now."""


class ScheduleInvalid(ValueError):
    """The shared schedule file is missing, malformed or breaks a schedule rule."""


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _now(now: datetime | None) -> datetime:
    return datetime.now(timezone.utc) if now is None else now


# ---------------------------------------------------------------- schedule file (f)

def validate_schedule(body: Any) -> list[str]:
    """Minimal mirror of the orchestrator validator; returns the rule violations."""
    if not isinstance(body, Mapping) or body.get("schema") != SCHEDULE_SCHEMA:
        return [f"schema must be {SCHEDULE_SCHEMA}"]
    entries = body.get("entries")
    if not isinstance(entries, list):
        return ["entries must be a list"]
    errors: list[str] = []
    seen: set[str] = set()
    spans: dict[str, list[tuple[datetime, datetime, str]]] = {}
    for n, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(f"entry {n} is not an object")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append(f"entry {n} has no id")
            continue
        if entry_id in seen:
            errors.append(f"duplicate id {entry_id}")
        seen.add(entry_id)
        start, end = _parse_ts(entry.get("start")), _parse_ts(entry.get("end"))
        if start is None or end is None or end <= start:
            errors.append(f"{entry_id}: end must be after start")
            continue
        if (end - start).total_seconds() > MAX_WINDOW_S and entry.get("operator_approved") is not True:
            errors.append(f"{entry_id}: longer than {MAX_WINDOW_S} s without operator_approved")
        spans.setdefault(str(entry.get("device_id")), []).append((start, end, entry_id))
    for device, rows in spans.items():
        rows.sort()
        for (_s1, e1, a), (s2, _e2, b) in zip(rows, rows[1:]):
            if s2 < e1:
                errors.append(f"{device}: {a} overlaps {b}")
    return errors


def read_schedule(path: Path = SCHEDULE_FILE) -> list[dict]:
    """The schedule's entries; raises ScheduleInvalid on any rule violation."""
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScheduleInvalid(f"schedule {path} unreadable: {exc}") from exc
    errors = validate_schedule(body)
    if errors:
        raise ScheduleInvalid(f"schedule {path}: " + "; ".join(errors))
    return [dict(entry) for entry in body["entries"]]


def resolve_schedule_ref(ref: str | None, *, campaign_id: str, start: datetime,
                         end: datetime, path: Path = SCHEDULE_FILE) -> dict:
    """The schedule entry `ref` names, which must cover [start, end] on mi210_0 for AK."""
    if not ref:
        raise ScheduleInvalid("a window request needs a schedule_ref (shared schedule file)")
    for entry in read_schedule(path):
        if entry["id"] != ref:
            continue
        e_start, e_end = _parse_ts(entry["start"]), _parse_ts(entry["end"])
        if entry.get("device_id") != DEVICE_ID or entry.get("holder") != "autokernel":
            raise ScheduleInvalid(f"{ref} is not an autokernel {DEVICE_ID} entry")
        if entry.get("campaign_id") != campaign_id:
            raise ScheduleInvalid(f"{ref} is scheduled for {entry.get('campaign_id')!r}, "
                                  f"not {campaign_id!r}")
        if not (e_start <= start and end <= e_end):
            raise ScheduleInvalid(f"{ref} [{entry['start']}, {entry['end']}] does not cover "
                                  f"the requested window [{start.isoformat()}, {end.isoformat()}]")
        return entry
    raise ScheduleInvalid(f"schedule_ref {ref!r} not in {path}")


# ---------------------------------------------------------------- request (bus payload)

@dataclass(frozen=True)
class WindowRequest:
    campaign_id: str
    purpose: str
    duration_s: int
    schedule_ref: str | None = None      # entry in the shared schedule file
    gpu_peak_ceiling_bytes: int = GPU_PEAK_CEILING_BYTES
    release_condition: str = ("AutoKernel releases the mi210_0 device flock at batch end, "
                              "on preempt_requested_at, or at expected_end")

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.purpose:
            raise ValueError("a window request needs a campaign and a purpose")
        if not 0 < int(self.duration_s) <= MAX_WINDOW_S:
            raise ValueError(f"window duration must be in (0, {MAX_WINDOW_S}] s; "
                             "longer needs the operator")
        if self.gpu_peak_ceiling_bytes > GPU_PEAK_CEILING_BYTES:
            raise ValueError("AK GPU runs stay at or under the 62 GiB ceiling")

    def to_bus_payload(self, *, now: datetime | None = None,
                       schedule_path: Path = SCHEDULE_FILE) -> dict[str, Any]:
        """The `compute-request` AK appends to its OWN outbox (never sent from here).

        Refuses (ScheduleInvalid) unless `schedule_ref` resolves to a schedule entry
        covering [now, now + duration]."""
        start = _now(now)
        end = start + timedelta(seconds=int(self.duration_s))
        resolve_schedule_ref(self.schedule_ref, campaign_id=self.campaign_id,
                             start=start, end=end, path=schedule_path)
        return {"kind": REQUEST_KIND, "from": REQUESTING_CONSUMER,
                "needs_routing_to": [COORDINATOR], "requires_ack": True,
                "payload": {"task": f"autokernel GPU window: {self.campaign_id} ({self.purpose})",
                            "device": "gpu", "device_id": DEVICE_ID,
                            "window": {"duration_s": int(self.duration_s),
                                       "expected_end": end.isoformat()},
                            "est_h": round(int(self.duration_s) / 3600, 3),
                            "release_condition": self.release_condition,
                            "campaign_id": self.campaign_id, "purpose": self.purpose,
                            "parked_roles": list(PARKED_ROLES),
                            "parked_ports": list(PARKED_PORTS),
                            "gpu_peak_ceiling_bytes": int(self.gpu_peak_ceiling_bytes),
                            "schedule_ref": self.schedule_ref}}


def release_message(grant: "WindowGrant", *, campaign_id: str) -> dict[str, Any]:
    """What AK appends to its own outbox AFTER dropping the device flock. The flock
    going free is the executor's signal; this message is a courtesy, not a write."""
    return {"kind": RELEASE_KIND, "from": REQUESTING_CONSUMER,
            "needs_routing_to": [COORDINATOR], "requires_ack": False,
            "payload": {"event": "gpu-window-released", "device_id": DEVICE_ID,
                        "window_id": grant.window_id, "campaign_id": campaign_id,
                        "released_at": datetime.now(timezone.utc).isoformat()}}


# ---------------------------------------------------------------- grant (read-only)

@dataclass(frozen=True)
class WindowGrant:
    holder: str | None
    expected_end: str | None
    parked_ports: tuple[int, ...]
    preempt_requested_at: str | None
    written_by: str | None = None
    window_id: str | None = None
    device_id: str | None = None
    grant_state: str | None = None
    schedule_ref: str | None = None

    def inactive_reason(self, now: datetime | None = None) -> str | None:
        """None while the executor's window grants AutoKernel the device, else why not."""
        if self.holder != "autokernel":
            return f"holder is {self.holder!r}, not autokernel"
        if self.preempt_requested_at:
            return f"preempt requested at {self.preempt_requested_at}"
        if self.written_by != EXECUTOR_WRITER:
            return f"window written by {self.written_by!r}, not {EXECUTOR_WRITER}"
        if not self.window_id:
            return "window has no window_id"
        if self.device_id != DEVICE_ID:
            return f"window is for {self.device_id!r}, not {DEVICE_ID}"
        if self.grant_state != GRANTED_STATE:
            return f"grant_state is {self.grant_state!r}, not {GRANTED_STATE}"
        end = _parse_ts(self.expected_end)
        if end is None:
            return "window has no valid expected_end"
        if _now(now) >= end:
            return f"window expired at {self.expected_end}"
        if not set(PARKED_PORTS) <= set(self.parked_ports):
            return f"parked ports {list(self.parked_ports)} do not cover {list(PARKED_PORTS)}"
        return None

    @property
    def active(self) -> bool:
        return self.inactive_reason() is None


def read_grant(path: Path = WINDOW_FILE) -> WindowGrant:
    """Read-only view of the window file. A missing/garbled file grants nothing."""
    try:
        body: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(body, Mapping):
            body = {}
    except (OSError, ValueError):
        body = {}
    return WindowGrant(holder=body.get("holder"), expected_end=body.get("expected_end"),
                       parked_ports=tuple(int(p) for p in body.get("parked_ports") or ()
                                          if str(p).isdigit()),
                       preempt_requested_at=body.get("preempt_requested_at"),
                       written_by=body.get("written_by"), window_id=body.get("window_id"),
                       device_id=body.get("device_id"), grant_state=body.get("grant_state"),
                       schedule_ref=body.get("schedule_ref"))


def assert_window_active(path: Path = WINDOW_FILE) -> WindowGrant:
    """The GPU batch / measurement boundary check. Raises GpuWindowPreempted."""
    grant = read_grant(path)
    reason = grant.inactive_reason()
    if reason is not None:
        raise GpuWindowPreempted(f"MI210 window not granted to AutoKernel ({path}): {reason}")
    return grant


class StackOwnerWindow(Protocol):
    """The handshake: request (compute-request) -> executor parks -> grant -> AK releases."""

    def request(self, request: WindowRequest) -> str: ...          # returns the corr_id

    def wait_grant(self, corr_id: str, timeout_s: float) -> WindowGrant: ...

    def release(self, grant: WindowGrant) -> None: ...  # drop device flock + outbox note


class UnagreedStackOwnerWindow:
    """Stub: every live call refuses until the stack-owned executor ships."""

    def request(self, request: WindowRequest) -> str:
        raise WindowNotAgreed("G1 handshake not live: the stack-owned executor has not shipped")

    def wait_grant(self, corr_id: str, timeout_s: float) -> WindowGrant:
        raise WindowNotAgreed("G1 handshake not live: the stack-owned executor has not shipped")

    def release(self, grant: WindowGrant) -> None:
        raise WindowNotAgreed("G1 handshake not live: the stack-owned executor has not shipped")


__all__ = ["DEVICE_ID", "GPU_PEAK_CEILING_BYTES", "GpuWindowPreempted", "MAX_WINDOW_S",
           "SCHEDULE_FILE", "SCHEDULE_SCHEMA", "ScheduleInvalid", "StackOwnerWindow",
           "UnagreedStackOwnerWindow", "WINDOW_FILE", "WindowGrant", "WindowNotAgreed",
           "WindowRequest", "assert_window_active", "read_grant", "read_schedule",
           "release_message", "resolve_schedule_ref", "validate_schedule"]
