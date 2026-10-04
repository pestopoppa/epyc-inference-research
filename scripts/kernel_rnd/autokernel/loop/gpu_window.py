#!/usr/bin/env python3
"""G1 DESIGN NOTE -- the MI210 window handover between AutoKernel and the stack owner.

STATUS: design + stub interface only (2026-10-04). Nothing here sends a bus message,
parks a role or touches a server. The stack owner (workspace-ec) must agree to this
shape before any handshake is implemented (AK long-context audit §4.2 G1).

Why a window at all
-------------------
Production `:8083` (Qwen3.8-27B + DFlash2, `architect_critic` and the roles sharing it)
holds 51-59 GiB of the MI210's 64 GiB. An AutoKernel GPU batch cannot co-reside, so it
needs an exclusive window, and only the stack owner may stop and reload a production
server (reload-ownership rule, `agents/shared/OPERATING_CONSTRAINTS.md`).

Sequence
--------
1. **AK requests** a window on the session bus to the stack-owner session
   (`WindowRequest.to_bus_payload`: campaign, purpose, duration, expected end, VRAM
   ceiling, schedule reference). AK never parks anything itself.
2. **The owner drains and unloads**, reusing what exists (constraint a):
   `python -m src.runtime.gpu_window park --roles architect_critic,... --ports 8083
   --holder autokernel --expected-end <t>` (parked roles answer 503 `role_parked`; the
   drain waits for in-flight completions), then `orchestrator_stack.py stop
   architect_critic`.
3. **AK claims, runs, releases**: it takes `mi210_0` (`resource/device_claim.py`), runs
   its GPU batches -- each GPU measurement also takes the q3 measurement claim
   (`claim.hold_q3_measurement`, claimed as `88-95` because `184-191` maps to no region
   until REGION-SIBLING-1 lands; serving `:8083` never needs it) -- checks
   `preempt_requested_at` at every measurement boundary, releases the claim and writes
   `holder=released`.
4. **The owner reloads**: `orchestrator_stack.py reload architect_critic`, a serving
   proof, then `gpu_window restore` (`holder=production`).

The serial router schedules GPU batches only inside a granted window
(`read_grant(...).active`) and CPU batches otherwise.

Stack-owner constraints (workspace-ec, 2026-10-04)
--------------------------------------------------
(a) Reuse `gpu_window park/restore` + `orchestrator_stack.py stop/reload`; AK requests
    via the bus, the stack owner executes.
(b) An automated, stack-owned executor is acceptable (a small daemon or an
    `orchestrator_stack` subcommand with a lease file) running ONLY that sequence:
    max window 60 min by default (longer needs the operator); auto-restore at
    expected_end + 10 min grace even if AK dies; refuse while an operator stack change
    is pending apply or bring-up; refuse if a long request (> N min remaining) is in
    flight -- drain timeout 10 min, then refuse rather than kill.
(c) Restore always uses the CURRENT compiled argv from the stack, never a cached one.
(d) One window at a time, recorded in the existing `mi210.json` window file and
    visible on the hub.
(e) AK GPU runs stay at or under 62 GiB peak, with abort-at-ceiling (like the KVU-16h
    runner). The G5 capacity dimension uses the same ceiling
    (`surface_validation.GPU_CAPACITY_CEILING_BYTES`); the in-run abort is a residency-
    sampler hook to add with the executor.
(f) Never during a backlog block unless it is scheduled in the shared schedule file
    (`WindowRequest.schedule_ref`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

WINDOW_FILE = Path("/mnt/raid0/llm/tmp/gpu-window/mi210.json")
STACK_OWNER_SESSION = "workspace-ec"
MAX_WINDOW_S = 3600            # (b) longer needs the operator
RESTORE_GRACE_S = 600          # (b) auto-restore at expected_end + 10 min
DRAIN_TIMEOUT_S = 600          # (b) then refuse, never kill
from .surface_validation import GPU_CAPACITY_CEILING_BYTES as GPU_PEAK_CEILING_BYTES  # (e)
PARKED_ROLES = ("architect_critic",)
PARKED_PORTS = (8083,)
REQUEST_KIND = "gpu-window-request"


class WindowNotAgreed(NotImplementedError):
    """The handshake is a design until the stack owner agrees to it."""


@dataclass(frozen=True)
class WindowRequest:
    campaign_id: str
    purpose: str
    duration_s: int
    schedule_ref: str | None = None      # (f) entry in the shared schedule file
    gpu_peak_ceiling_bytes: int = GPU_PEAK_CEILING_BYTES

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.purpose:
            raise ValueError("a window request needs a campaign and a purpose")
        if not 0 < int(self.duration_s) <= MAX_WINDOW_S:
            raise ValueError(f"window duration must be in (0, {MAX_WINDOW_S}] s; "
                             "longer needs the operator")
        if self.gpu_peak_ceiling_bytes > GPU_PEAK_CEILING_BYTES:
            raise ValueError("AK GPU runs stay at or under the 62 GiB ceiling")

    def to_bus_payload(self) -> dict[str, Any]:
        """The payload AK would append to its own outbox, routed to the stack owner."""
        return {"kind": REQUEST_KIND, "needs_routing_to": [STACK_OWNER_SESSION],
                "requires_ack": True,
                "payload": {"campaign_id": self.campaign_id, "purpose": self.purpose,
                            "duration_s": int(self.duration_s),
                            "parked_roles": list(PARKED_ROLES),
                            "parked_ports": list(PARKED_PORTS),
                            "gpu_peak_ceiling_bytes": int(self.gpu_peak_ceiling_bytes),
                            "schedule_ref": self.schedule_ref,
                            "restore_grace_s": RESTORE_GRACE_S,
                            "drain_timeout_s": DRAIN_TIMEOUT_S}}


@dataclass(frozen=True)
class WindowGrant:
    holder: str | None
    expected_end: str | None
    parked_ports: tuple[int, ...]
    preempt_requested_at: str | None

    @property
    def active(self) -> bool:
        """True while the window file grants AutoKernel the device and has not expired."""
        if self.holder != "autokernel" or not self.expected_end:
            return False
        try:
            end = datetime.fromisoformat(self.expected_end)
        except ValueError:
            return False
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < end and set(PARKED_PORTS) <= set(self.parked_ports)


def read_grant(path: Path = WINDOW_FILE) -> WindowGrant:
    """Read-only view of the window file (d). A missing/garbled file grants nothing."""
    try:
        body: Mapping[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        body = {}
    return WindowGrant(holder=body.get("holder"), expected_end=body.get("expected_end"),
                       parked_ports=tuple(int(p) for p in body.get("parked_ports") or ()
                                          if str(p).isdigit()),
                       preempt_requested_at=body.get("preempt_requested_at"))


class StackOwnerWindow(Protocol):
    """The handshake, once agreed: request -> owner parks -> grant -> AK releases."""

    def request(self, request: WindowRequest) -> str: ...          # returns the corr_id

    def wait_grant(self, corr_id: str, timeout_s: float) -> WindowGrant: ...

    def release(self, grant: WindowGrant) -> None: ...             # writes holder=released


class UnagreedStackOwnerWindow:
    """Stub: every call refuses until workspace-ec agrees to the sequence above."""

    def request(self, request: WindowRequest) -> str:
        raise WindowNotAgreed("G1 handshake not agreed with the stack owner (workspace-ec)")

    def wait_grant(self, corr_id: str, timeout_s: float) -> WindowGrant:
        raise WindowNotAgreed("G1 handshake not agreed with the stack owner (workspace-ec)")

    def release(self, grant: WindowGrant) -> None:
        raise WindowNotAgreed("G1 handshake not agreed with the stack owner (workspace-ec)")


__all__ = ["GPU_PEAK_CEILING_BYTES", "MAX_WINDOW_S", "StackOwnerWindow",
           "UnagreedStackOwnerWindow", "WINDOW_FILE", "WindowGrant", "WindowNotAgreed",
           "WindowRequest", "read_grant"]
