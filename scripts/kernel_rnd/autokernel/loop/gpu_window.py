#!/usr/bin/env python3
"""G1 -- the MI210 window handover between AutoKernel and the stack-owned executor.

STATUS (2026-10-04, after the G1 review NACK, `tmp/g1-review-20261004/REVIEW.md`):
AutoKernel's half of the agreed contract. Nothing here parks a role, touches a server
or writes the window file. `BusStackOwnerWindow` is the live handshake (round-2 F5):
it appends the `compute-request` to AK's OWN outbox through `session_bus.py append`
(the single-writer CLI, schema-validated), polls AK's inbox read-only for the
coordinator-daemon's `compute-grant` / `compute-deny` with the request's `corr_id`,
verifies the grant (device, expected_end, campaign) AND the executor's window file plus
its lease (window_id, `compute_grant` binding) before anything uses the device, and
fails closed on a deny, a timeout or any mismatch. `UnagreedStackOwnerWindow` remains
the refusing stub for callers that must not open a window.

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

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from .surface_validation import GPU_CAPACITY_CEILING_BYTES as GPU_PEAK_CEILING_BYTES

WINDOW_FILE = Path("/mnt/raid0/llm/tmp/gpu-window/mi210.json")
SCHEDULE_FILE = Path("/mnt/raid0/llm/tmp/gpu-window/schedule.json")
SCHEDULE_SCHEMA = "epyc.gpu_window_schedule.v1"
DEVICE_ID = "mi210_0"
EXECUTOR_WRITER = "gpu_window_executor"
GRANTED_STATE = "granted"
REQUEST_KIND = "compute-request"
GRANT_KIND = "compute-grant"
DENY_KIND = "compute-deny"
RELEASE_KIND = "status"
#: The one live bus (session_bus.get_bus_root); tests pass a tmp root explicitly.
BUS_ROOT = Path("/workspace/coordination/session-bus")
#: Grant/window clock slack: the executor may round expected_end to the second.
GRANT_SLACK_S = 60
DEFAULT_GRANT_TIMEOUT_S = 900.0        # executor drain (<=10 min) + stop + margin
DEFAULT_GRANT_POLL_S = 5.0
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


class WindowDenied(RuntimeError):
    """The coordinator-daemon answered the compute-request with compute-deny."""


class WindowGrantTimeout(RuntimeError):
    """No verified grant (bus grant + executor window) arrived within the timeout."""


class GrantUnverified(RuntimeError):
    """A grant arrived but does not match the request, the device or the window."""


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
        # `session_bus.schema.json` compute-request: task, window (STRING),
        # device_region (string), est_h, release_condition are required; the structured
        # window rides beside it. `to` is the daemon: it has no roster row, so it cannot
        # sit in `needs_routing_to` (`append` refuses non-roster routing ids).
        return {"kind": REQUEST_KIND, "from": REQUESTING_CONSUMER,
                "to": COORDINATOR, "requires_ack": True,
                "payload": {"task": f"autokernel GPU window: {self.campaign_id} ({self.purpose})",
                            "device": "gpu", "device_id": DEVICE_ID,
                            "device_region": f"gpu/{DEVICE_ID}",
                            "window": (f"{start.isoformat()} -> {end.isoformat()} "
                                       f"({int(self.duration_s)} s)"),
                            "window_detail": {"duration_s": int(self.duration_s),
                                              "start": start.isoformat(),
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
            "to": COORDINATOR, "requires_ack": False,
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
    campaign_id: str | None = None
    #: Set only by `BusStackOwnerWindow.wait_grant`: the verified compute-grant id.
    compute_grant: str | None = None

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
                       schedule_ref=body.get("schedule_ref"),
                       campaign_id=body.get("campaign_id"))


def assert_window_active(path: Path = WINDOW_FILE) -> WindowGrant:
    """The GPU batch / measurement boundary check. Raises GpuWindowPreempted."""
    grant = read_grant(path)
    reason = grant.inactive_reason()
    if reason is not None:
        raise GpuWindowPreempted(f"MI210 window not granted to AutoKernel ({path}): {reason}")
    return grant


def abort_reason(grant: WindowGrant, path: Path = WINDOW_FILE, *,
                 now: datetime | None = None) -> str | None:
    """Mid-measurement abort test (round-2 F6): None while `grant`'s window still stands.

    Two things end a measurement in flight: the window's hard `expected_end` (the
    executor restores at end + grace even against a live holder, so AK must be off the
    device by then), and the executor taking the device back (holder / grant_state /
    window_id changed). A preempt request alone is NOT here: it is a drain, honoured at
    the next measurement or batch boundary (`assert_window_active`)."""
    moment = _now(now)
    end = _parse_ts(grant.expected_end)
    if end is None or moment >= end:
        return f"window expected_end {grant.expected_end} reached"
    current = read_grant(path)
    if current.window_id != grant.window_id:
        return f"window replaced ({grant.window_id!r} -> {current.window_id!r})"
    if current.holder != "autokernel":
        return f"holder is now {current.holder!r}"
    if current.grant_state != GRANTED_STATE:
        return f"grant_state is now {current.grant_state!r}"
    current_end = _parse_ts(current.expected_end)
    if current_end is None or moment >= current_end:
        return f"window expected_end {current.expected_end} reached"
    return None


class ExpiryWatcher:
    """Thread that calls `on_abort(reason)` ONCE when `abort_reason` fires.

    It only decides; what to stop is the caller's (its OWN launched processes, by the
    PIDs it captured -- `residency.OWN_LAUNCHES`). A watcher that cannot read the window
    file sees a missing window (holder None) and fires: unreadable is not "still mine".

    The granted expected_end is a clock fact and fires on the first poll past it. A
    revocation read from the window file must be seen on TWO consecutive polls before it
    kills a measurement (two-sample persistence, INVARIANTS.md): one torn or transient
    read is not the executor taking the device back.
    """

    def __init__(self, grant: WindowGrant, on_abort: Callable[[str], None], *,
                 path: Path = WINDOW_FILE, poll_s: float = 1.0,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.grant, self.path, self.poll_s = grant, Path(path), float(poll_s)
        self._on_abort, self._clock = on_abort, clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: str | None = None
        self.fired: str | None = None

    def check(self) -> str | None:
        if self.fired is not None:
            return self.fired
        now = _now(None if self._clock is None else self._clock())
        end = _parse_ts(self.grant.expected_end)
        if end is None or now >= end:
            reason = f"window expected_end {self.grant.expected_end} reached"
        else:
            reason = abort_reason(self.grant, self.path, now=now)
            if reason is None:
                self._pending = None
                return None
            if self._pending is None:
                self._pending = reason  # first sight: confirm on the next poll
                return None
        self.fired = reason
        self._on_abort(reason)
        return reason

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.check() is not None:
                    return
            except Exception as exc:  # noqa: BLE001 -- a broken watcher must not pass silently
                self.fired = f"expiry watcher failed: {type(exc).__name__}: {exc}"
                try:
                    self._on_abort(self.fired)
                finally:
                    return
            self._stop.wait(self.poll_s)

    def __enter__(self) -> "ExpiryWatcher":
        self._thread = threading.Thread(target=self._loop, name="gpu-window-expiry",
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


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


# ---------------------------------------------------------------- live handshake (F5)

def _root_repo() -> Path:
    return Path(os.environ.get("EPYC_ROOT_REPO", "/workspace"))


def session_bus_appender(bus_root: Path, agent: str = REQUESTING_CONSUMER,
                         python: str | None = None) -> Callable[[dict], None]:
    """Append to AK's OWN outbox through the single-writer CLI (schema + writer checks).

    Never a direct file write: `session_bus.py append --target outbox` derives the
    writer from the path and validates the row against `session_bus.schema.json`."""
    script = _root_repo() / "scripts" / "coordination" / "session_bus.py"

    def append(msg: dict) -> None:
        result = subprocess.run(
            [python or sys.executable, str(script), "--bus-root", str(bus_root), "append",
             "--agent", agent, "--target", "outbox", "--json", json.dumps(msg)],
            capture_output=True, text=True, timeout=60, check=False)
        if result.returncode != 0:
            raise GrantUnverified(f"session_bus append refused the message (rc="
                                  f"{result.returncode}): {result.stderr.strip()[-400:]}")
    return append


def _message_id(agent: str = REQUESTING_CONSUMER) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"msg-{stamp}-{secrets.randbelow(10**9)}-{agent}"


def _inbox_rows(path: Path, offset: int) -> list[dict]:
    """Complete JSON rows of `path` after byte `offset` (read-only; a torn tail line is
    left for the next poll)."""
    try:
        raw = Path(path).read_bytes()[offset:]
    except OSError:
        return []
    rows = []
    for line in raw.split(b"\n")[:-1]:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _refers_to(row: Mapping[str, Any], msg_id: str) -> bool:
    ids = row.get("corr_ids") if isinstance(row.get("corr_ids"), list) else []
    return row.get("corr_id") == msg_id or msg_id in ids


def verify_bus_grant(row: Mapping[str, Any], request_msg: Mapping[str, Any], *,
                     now: datetime | None = None) -> datetime:
    """The grant's verified expected_end; raises GrantUnverified on any mismatch.

    A grant must: be a `compute-grant` FROM the coordinator-daemon, answer THIS
    request's id, name `mi210_0`, carry a parseable expected_end in the future, no
    later than the requested end (+ slack), within MAX_WINDOW_S, for this campaign."""
    request_id = str(request_msg["id"])
    req = request_msg["payload"]
    if row.get("kind") != GRANT_KIND:
        raise GrantUnverified(f"not a {GRANT_KIND}: {row.get('kind')!r}")
    if row.get("from") != COORDINATOR:
        raise GrantUnverified(f"grant from {row.get('from')!r}, not {COORDINATOR}")
    if not _refers_to(row, request_id):
        raise GrantUnverified(f"grant corr_id {row.get('corr_id')!r} != request {request_id}")
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    detail = payload.get("window_detail") if isinstance(payload.get("window_detail"),
                                                        Mapping) else {}
    device = payload.get("device_id") or detail.get("device_id")
    if device != DEVICE_ID and DEVICE_ID not in str(payload.get("device_region") or ""):
        raise GrantUnverified(f"grant is for {device or payload.get('device_region')!r}, "
                              f"not {DEVICE_ID}")
    end = _parse_ts(payload.get("expected_end") or detail.get("expected_end"))
    if end is None:
        raise GrantUnverified("grant carries no parseable expected_end")
    moment = _now(now)
    requested_end = _parse_ts(req["window_detail"]["expected_end"])
    if end <= moment:
        raise GrantUnverified(f"grant expected_end {end.isoformat()} already passed")
    if end > requested_end + timedelta(seconds=GRANT_SLACK_S):
        raise GrantUnverified(f"grant expected_end {end.isoformat()} is past the requested "
                              f"{requested_end.isoformat()}")
    if (end - moment).total_seconds() > MAX_WINDOW_S + GRANT_SLACK_S:
        raise GrantUnverified(f"grant window longer than {MAX_WINDOW_S} s")
    campaign = payload.get("campaign_id")
    if campaign is not None and campaign != req["campaign_id"]:
        raise GrantUnverified(f"grant is for campaign {campaign!r}, not {req['campaign_id']!r}")
    return end


def lease_path(window: Path) -> Path:
    """The executor's lease file beside the window (read-only here)."""
    return Path(window).with_name(Path(window).name + ".lease.json")


def verify_window_for_grant(window: WindowGrant, *, request_msg: Mapping[str, Any],
                            grant_row: Mapping[str, Any], grant_end: datetime,
                            window_path: Path, now: datetime | None = None) -> None:
    """The executor's window is the one this grant opened, or GrantUnverified.

    Window: active for AutoKernel, same campaign / schedule_ref when it names them, and
    expected_end no later than the bus grant's (+ slack). Lease: same window_id and its
    `authority.compute_grant` names this grant (or the request) -- a window opened for
    someone else's grant is not ours."""
    reason = window.inactive_reason(now)
    if reason is not None:
        raise GrantUnverified(f"window not granted: {reason}")
    req = request_msg["payload"]
    if window.campaign_id is not None and window.campaign_id != req["campaign_id"]:
        raise GrantUnverified(f"window is for campaign {window.campaign_id!r}")
    if window.schedule_ref is not None and window.schedule_ref != req.get("schedule_ref"):
        raise GrantUnverified(f"window schedule_ref {window.schedule_ref!r} != "
                              f"{req.get('schedule_ref')!r}")
    end = _parse_ts(window.expected_end)
    if end is None or end > grant_end + timedelta(seconds=GRANT_SLACK_S):
        raise GrantUnverified(f"window expected_end {window.expected_end} is past the "
                              f"grant's {grant_end.isoformat()}")
    try:
        lease = json.loads(lease_path(window_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GrantUnverified(f"executor lease unreadable: {exc}") from exc
    if not isinstance(lease, Mapping) or lease.get("window_id") != window.window_id:
        raise GrantUnverified("executor lease does not name this window_id")
    authority = lease.get("authority") if isinstance(lease.get("authority"), Mapping) else {}
    accepted = {str(grant_row.get("id")), str(request_msg["id"])}
    if grant_row.get("relayed_src"):
        accepted.add(str(grant_row["relayed_src"]))
    if str(authority.get("compute_grant")) not in accepted:
        raise GrantUnverified(f"window opened under compute_grant "
                              f"{authority.get('compute_grant')!r}, not this grant")


class BusStackOwnerWindow:
    """The live G1 handshake: compute-request -> verified grant -> AK holds -> release.

    Fails CLOSED everywhere: a deny raises WindowDenied, no verified grant within
    `timeout_s` raises WindowGrantTimeout, a mismatching grant/window raises
    GrantUnverified. Never writes the window file or anyone else's bus file."""

    def __init__(self, *, bus_root: Path = BUS_ROOT, window_path: Path = WINDOW_FILE,
                 schedule_path: Path = SCHEDULE_FILE, agent: str = REQUESTING_CONSUMER,
                 appender: Callable[[dict], None] | None = None,
                 clock: Callable[[], datetime] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic,
                 poll_s: float = DEFAULT_GRANT_POLL_S) -> None:
        self.bus_root, self.window_path = Path(bus_root), Path(window_path)
        self.schedule_path, self.agent = Path(schedule_path), agent
        self._append = appender or session_bus_appender(self.bus_root, agent)
        self._clock, self._sleep, self._monotonic = clock, sleep, monotonic
        self.poll_s = float(poll_s)
        self._requests: dict[str, tuple[dict, int]] = {}

    def _inbox(self) -> Path:
        return self.bus_root / "inbox" / f"{self.agent}.jsonl"

    def request(self, request: WindowRequest) -> str:
        now = None if self._clock is None else self._clock()
        msg = request.to_bus_payload(now=now, schedule_path=self.schedule_path)
        msg["id"] = _message_id(self.agent)
        try:
            offset = self._inbox().stat().st_size
        except OSError:
            offset = 0
        self._append(msg)
        self._requests[msg["id"]] = (msg, offset)
        return msg["id"]

    def wait_grant(self, corr_id: str, timeout_s: float = DEFAULT_GRANT_TIMEOUT_S) -> WindowGrant:
        if corr_id not in self._requests:
            raise GrantUnverified(f"{corr_id} is not a request this handshake sent")
        request_msg, offset = self._requests[corr_id]
        deadline = self._monotonic() + float(timeout_s)
        grant_row = grant_end = None
        last = "no compute-grant yet"
        while True:
            now = None if self._clock is None else self._clock()
            if grant_row is None:
                for row in _inbox_rows(self._inbox(), offset):
                    if not _refers_to(row, corr_id):
                        continue
                    if row.get("kind") == DENY_KIND:
                        payload = row.get("payload") or {}
                        raise WindowDenied(f"compute-deny for {corr_id}: "
                                           f"{payload.get('reason')} (next: {payload.get('next')})")
                    if row.get("kind") == GRANT_KIND:
                        grant_end = verify_bus_grant(row, request_msg, now=now)
                        grant_row = row
                        break
            if grant_row is not None:
                window = read_grant(self.window_path)
                try:
                    verify_window_for_grant(window, request_msg=request_msg,
                                            grant_row=grant_row, grant_end=grant_end,
                                            window_path=self.window_path, now=now)
                except GrantUnverified as exc:
                    last = f"bus grant {grant_row.get('id')} verified; window: {exc}"
                else:
                    return replace(window, compute_grant=str(grant_row.get("id")))
                if grant_end is not None and _now(now) >= grant_end:
                    raise WindowGrantTimeout(f"grant {grant_row.get('id')} expired before "
                                             f"the executor granted the window: {last}")
            if self._monotonic() >= deadline:
                raise WindowGrantTimeout(f"no verified grant for {corr_id} within "
                                         f"{timeout_s:.0f} s: {last}")
            self._sleep(self.poll_s)

    def release(self, grant: WindowGrant, *, hold: Any = None,
                campaign_id: str | None = None) -> None:
        """Drop the device flock (the executor's signal), then a courtesy status row."""
        if hold is not None:
            hold.release()
        msg = release_message(grant, campaign_id=campaign_id or grant.campaign_id or "")
        msg["id"] = _message_id(self.agent)
        self._append(msg)


__all__ = ["BUS_ROOT", "BusStackOwnerWindow", "ExpiryWatcher", "GrantUnverified",
           "WindowDenied", "WindowGrantTimeout", "abort_reason", "lease_path",
           "session_bus_appender", "verify_bus_grant", "verify_window_for_grant",
           "DEVICE_ID", "GPU_PEAK_CEILING_BYTES", "GpuWindowPreempted", "MAX_WINDOW_S",
           "SCHEDULE_FILE", "SCHEDULE_SCHEMA", "ScheduleInvalid", "StackOwnerWindow",
           "UnagreedStackOwnerWindow", "WINDOW_FILE", "WindowGrant", "WindowNotAgreed",
           "WindowRequest", "assert_window_active", "read_grant", "read_schedule",
           "release_message", "resolve_schedule_ref", "validate_schedule"]
