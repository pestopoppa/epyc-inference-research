#!/usr/bin/env python3
"""device_claim.py — the cross-process exclusive GPU device claim (§2.6, §14 AK2).

WHY THIS MODULE EXISTS
----------------------
Nothing in this project can currently stop two processes from using the MI210 at
the same time. §2.6 lists the cross-process GPU device claim as the first row of
"substrate that does not exist yet", and invariant 9 (§4) is unsatisfiable
without it: *"Resources are acquired, not observed. Every CPU/GPU benchmark or
profiler run holds the appropriate region/device claim. Idle sensing is never a
claim."* Two things stand in for it today, and neither is a claim:

  * **`kernel_eval.sh`'s `gpu_idle()` (`:77-82`)** shells out to `rocm-smi
    --showpids` and proceeds when nothing is running. That is idle SENSING: it is
    TOCTOU by construction, two sessions can both observe an idle card and both
    start, and it is the exact pattern §10.4 forbids. AK2's acceptance criteria
    require it to be *deleted, not wrapped* — this module is what replaces it.

  * **`epyc-orchestrator/src/gpu_lease.py`** is a `threading.Condition` lease
    (`:63-69`). A `threading.Lock` lives in one interpreter's heap. A second
    process gets its own heap, its own manager, and its own unheld lease, so it
    grants itself the GPU while the first process is mid-benchmark. It cannot
    exclude another process and therefore cannot be the device claim, no matter
    how the API reads. It remains correct for what it is — intra-process
    ownership inside one orchestrator (`axa2_live_cutover_bundle.py:535`) — and
    this module deliberately does NOT imitate it.

The physical exclusion fact here is an `fcntl.flock(LOCK_EX)` held on a file, the
same primitive and the same on-disk root as the CPU sibling
`epyc-orchestrator/src/runtime/cpu_region_lock.py`. Sharing the root is the whole
point: `gpu_device.mi210_0.lock` sits beside `cpu_region.frontdoor.q0.lock` under
`/mnt/raid0/llm/tmp/`, so an orchestrator process and a research process exclude
each other without either importing the other's code. This module is the research
repo's half; **an orchestrator-side CLI verb wrapping it (a `--device` verb on
`region_lock_cli.py`, or a sibling CLI sharing this lock root) is a follow-up
owned by whoever holds `epyc-orchestrator`** — AK2 requires it not be a fork of
the lock semantics, which is why every rule here is in one module with no
orchestrator import.

WHAT THIS MODULE GUARANTEES
---------------------------
1. **Exclusion is the kernel's, not ours.** `flock(LOCK_EX|LOCK_NB)` on a never-
   unlinked lock file. Two processes cannot both hold it, including two claims in
   the same process (flock conflicts between open file descriptions), so a
   self-deadlock surfaces as a clean timeout instead of a double-booked GPU.
2. **Liveness is PID + process start time, never a heartbeat.** The holder's
   start time comes from `/proc/<pid>/stat` field 22, plus the kernel boot id, so
   a recycled PID cannot impersonate a dead holder. A heartbeat is not a lease:
   INC-20260727-stale-heartbeat established that a heartbeat written once is a
   birth certificate, and §3.6/§2.6 note that a heartbeat-based claim means
   *nothing can revoke you*. There is no heartbeat anywhere in this file.
3. **A live holder is never stolen from.** Reclamation requires the holder to be
   *positively* determined dead. `unknown` — another host, an unreadable
   `/proc`, a corrupt payload — is a THIRD outcome and never licenses a
   takeover; it raises `DeviceClaimInconsistent` and journals a defect.
4. **Crash recovery is journaled before it happens.** A dead holder's claim is
   reclaimable after a grace period; the `claim_reclaimed` record is written
   BEFORE the new payload, so a reclaim can never occur without a record of it
   (invariant 7: all outcomes are durable).
5. **Revocation is quiesce-and-drain, never forcible** (`BUS_PROTOCOL.md:47-51`,
   fabric axiom 4). A revoker marks the claim `revoking` in a sidecar file; the
   holder sees it at *its own* boundary, acknowledges, drains, and releases. An
   ignored revocation surfaces as a `defect` record. **This module never sends a
   signal, never unlinks another process's lock file, and never calls
   `flock(LOCK_UN)` on a descriptor it did not open.**
6. **Every claim emits a receipt** (`claim_id`, device, holder pid + start time,
   `acquired_at`, purpose, campaign) whose `claim_id` is the string callers put
   in `evaluation_event.resource_claim_receipt` (`schemas.validate_evaluation_
   event`), binding a measurement to the exclusivity that produced it.

DELIBERATE DEVIATIONS FROM A LITERAL READING OF THE AK2 CHECKLIST
-----------------------------------------------------------------
* §14 AK2 says *"Acquire is atomic (`O_CREAT|O_EXCL` plus `flock`)"*. The
  `O_CREAT|O_EXCL` half is NOT implemented as lockfile creation, because
  create-and-unlink lockfiles are unsafe in exactly the way this module exists to
  prevent: process A opens the file, process B unlinks it and creates a new one,
  and now A and B hold `LOCK_EX` on two different inodes for the same path — two
  holders, no error. The lock file is created on demand with `O_CREAT` and
  **never unlinked**; atomicity comes from `flock`, which is atomic and
  auto-released by the kernel on process death. The acceptance criterion is
  "atomic and genuinely exclusive across processes", and this is the shape that
  actually delivers it.
* Device ids are VALIDATED, not sanitized. The CPU sibling rewrites `/` to `_`
  in role and region names; doing that to a device key would silently map two
  distinct device ids onto one lock file (over-exclusion) or, worse, let one
  device be addressed by two names that do not exclude each other. A bad device
  id raises.
* `timeout_s=0` means *one attempt*, not *block forever*. The CPU sibling reads
  a non-positive timeout as "no timeout"; on a shared GPU that turns a typo into
  an unbounded hang. `None` blocks forever and has to be written out.

SCOPE
-----
No inference, no benchmark, no process start/stop/signal, no production tree.
Reading `/proc/<pid>/stat` for a PID this module's own lock file names is a
targeted read of a recorded id, not a name-pattern scan — it is the audited
read-only shape §3.5 asks for, and it is the opposite of `pgrep`
(INC-20260731-broad-process-pattern-kills).

Requires Linux `/proc`. On a host without it the module raises at claim time
rather than falling back to PID-only liveness, because PID-only liveness IS the
impersonation hole this design was written to close.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from ..schemas import COULD_NOT_CHECK, FAIL, PASS, Check, canonical_json

__all__ = [
    "DEVICE_CLAIM_SCHEMA",
    "RECEIPT_SCHEMA",
    "REVOCATION_SCHEMA",
    "JOURNAL_SCHEMA",
    "DEFAULT_LOCK_ROOT",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_POLL_S",
    "DEFAULT_STALE_GRACE_S",
    "LIVE",
    "DEAD",
    "UNKNOWN",
    "DeviceClaimError",
    "DeviceClaimTimeout",
    "DeviceClaimInconsistent",
    "DeviceClaimUnreadable",
    "ClaimJournal",
    "ClaimReceipt",
    "DeviceClaim",
    "Liveness",
    "acquire_device_claim",
    "gpu_device_claim",
    "gpu_device_claims",
    "device_lock_path",
    "revocation_path",
    "assess_holder_liveness",
    "current_holder_identity",
    "request_revocation",
    "revocation_status",
    "check_revocation_compliance",
    "check_device_claim_held",
    "check_claim_expiry",
    "inspect_device_claim",
]

# =============================================================================
# Identity and defaults
# =============================================================================

DEVICE_CLAIM_SCHEMA = "epyc.autokernel.device_claim.v1"
RECEIPT_SCHEMA = "epyc.autokernel.device_claim_receipt.v1"
REVOCATION_SCHEMA = "epyc.autokernel.device_claim_revocation.v1"
JOURNAL_SCHEMA = "epyc.autokernel.device_claim_journal.v1"

# The CPU sibling's root. Resolution order is copied EXACTLY from
# `cpu_region_lock._tmp_dir()` (ORCHESTRATOR_TMP_DIR, then
# ORCHESTRATOR_PATHS_TMP_DIR, then the hard-coded path). No AutoKernel-specific
# env var exists on purpose: a research-repo-only override would let the two
# repos resolve different roots and silently stop excluding each other, which is
# the one failure this whole module is here to prevent. Tests pass `lock_root=`
# explicitly instead.
DEFAULT_LOCK_ROOT = "/mnt/raid0/llm/tmp"
_LOCK_ROOT_ENV_VARS = ("ORCHESTRATOR_TMP_DIR", "ORCHESTRATOR_PATHS_TMP_DIR")

_LOCK_NAME = "gpu_device.{device_id}.lock"
_REVOKE_NAME = "gpu_device.{device_id}.revoke.json"

DEFAULT_TIMEOUT_S = 300.0
DEFAULT_POLL_S = 0.05
# Grace before a dead holder's claim may be reclaimed, measured from the claim's
# own `acquired_at`. There is no other clock available and that is by design:
# a heartbeat would be the obvious second clock, and a heartbeat is not a lease.
DEFAULT_STALE_GRACE_S = 30.0

# A payload larger than this is corruption, not a claim.
_MAX_PAYLOAD_BYTES = 65536

_DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_CLAIM_ID_PREFIX = "akd-"
_REVOCATION_ID_PREFIX = "akv-"
_JOURNAL_ID_PREFIX = "akj-"

# Journal record kinds.
KIND_ACQUIRED = "claim_acquired"
KIND_RELEASED = "claim_released"
KIND_RECLAIMED = "claim_reclaimed"
KIND_REVOCATION_REQUESTED = "revocation_requested"
KIND_REVOCATION_ACKNOWLEDGED = "revocation_acknowledged"
KIND_REVOCATION_SATISFIED = "revocation_satisfied"
KIND_REVOCATION_DISCARDED = "revocation_discarded"
KIND_DEFECT = "defect"

# Defect classes. Each one names a state a machine must NOT resolve on its own.
DEFECT_LIVE_HOLDER_FREE_LOCK = "device_claim.live_holder_without_lock"
DEFECT_UNVERIFIABLE_CLAIM = "device_claim.unverifiable_claim"
DEFECT_REVOCATION_IGNORED = "device_claim.revocation_ignored"
# A drain that never completed because its target DIED holding the claim. It is
# still a failed drain, but it is not defiance: filing it as `revocation_ignored`
# sends a human to escalate with the owner of a process that no longer exists.
DEFECT_REVOCATION_ORPHANED = "device_claim.revocation_orphaned"

# Liveness is three-valued. `unknown` is never collapsed into either of the
# other two: an unreadable /proc is not a dead holder.
LIVE = "live"
DEAD = "dead"
UNKNOWN = "unknown"

# Claim states written into the payload.
STATE_HELD = "held"
STATE_DRAINING = "draining"


# =============================================================================
# Exceptions — every one of these is a refusal, never a degraded success
# =============================================================================

class DeviceClaimError(RuntimeError):
    """Base for every device-claim failure."""


class DeviceClaimTimeout(DeviceClaimError):
    """The device was held by someone else for the whole budget.

    This is the ordinary contention outcome and the ONLY one a caller should
    retry: the device is genuinely busy and nothing is wrong.
    """


class DeviceClaimInconsistent(DeviceClaimError):
    """The lock is free but its payload cannot be shown to be abandoned.

    Raised when the recorded holder is alive, or when its liveness cannot be
    determined at all. Taking the claim here would be a steal from a possibly
    live holder, so the module refuses and journals a defect for a human.
    """


class DeviceClaimUnreadable(DeviceClaimError):
    """A claim/revocation record exists but could not be read as a record."""


# =============================================================================
# Paths and identifiers
# =============================================================================

def _default_lock_root() -> Path:
    for name in _LOCK_ROOT_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return Path(DEFAULT_LOCK_ROOT)


def _validated_device_id(device_id: Any) -> str:
    if not isinstance(device_id, str) or not _DEVICE_ID_RE.match(device_id):
        raise ValueError(
            f"invalid device id {device_id!r}: must match {_DEVICE_ID_RE.pattern} "
            "(ids are validated, never rewritten — silently rewriting one device "
            "id into another's lock file would break exclusion, not fix a typo)"
        )
    return device_id


def _resolved_root(lock_root: Optional[os.PathLike | str]) -> Path:
    return Path(lock_root) if lock_root is not None else _default_lock_root()


def device_lock_path(device_id: str, lock_root: Optional[os.PathLike | str] = None) -> Path:
    """`{lock_root}/gpu_device.{device_id}.lock` — the exclusion fact's file.

    Shares a directory with the CPU sibling's `cpu_region.{role}.{region}.lock`
    so that both repos' claims live in one place a human can list.
    """
    return _resolved_root(lock_root) / _LOCK_NAME.format(
        device_id=_validated_device_id(device_id)
    )


def revocation_path(device_id: str, lock_root: Optional[os.PathLike | str] = None) -> Path:
    """`{lock_root}/gpu_device.{device_id}.revoke.json` — the drain request.

    A separate file on purpose. The holder owns `LOCK_EX` on the lock file for
    the whole claim, so a revoker physically cannot write into it; demanding the
    lock in order to ask for the lock would make revocation impossible.
    """
    return _resolved_root(lock_root) / _REVOKE_NAME.format(
        device_id=_validated_device_id(device_id)
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def _utc_now_iso(now: Optional[float] = None) -> str:
    moment = datetime.now(timezone.utc) if now is None else datetime.fromtimestamp(
        now, tz=timezone.utc
    )
    return moment.isoformat()


def _parse_iso(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise DeviceClaimUnreadable(f"{field_name}: expected an ISO-8601 string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DeviceClaimUnreadable(f"{field_name}: {value!r} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise DeviceClaimUnreadable(f"{field_name}: {value!r} has no timezone offset")
    return parsed


# =============================================================================
# Holder identity and liveness — PID plus start time, never a heartbeat
# =============================================================================

@dataclass(frozen=True)
class Liveness:
    """Three-valued liveness verdict. `unknown` is not a soft `dead`."""

    state: str
    reason: str

    def __post_init__(self) -> None:
        if self.state not in (LIVE, DEAD, UNKNOWN):
            raise ValueError(f"invalid liveness state: {self.state!r}")

    @property
    def reclaimable(self) -> bool:
        """True only for a positively DEAD holder. `unknown` never reclaims."""
        return self.state == DEAD


def _read_boot_id() -> str:
    """Kernel boot id — the namespace that makes a start-tick value meaningful.

    `/proc/<pid>/stat` field 22 counts clock ticks since boot, so the same value
    means different wall times across reboots. Recording the boot id turns "same
    PID, same start ticks" from a strong heuristic into an exact identity, and
    makes a lock left over from before a reboot trivially classifiable as dead.
    """
    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="ascii") as fh:
            return fh.read().strip()
    except OSError as exc:
        raise DeviceClaimError(
            "cannot read /proc/sys/kernel/random/boot_id: this module requires Linux "
            "/proc for PID+start-time liveness, and refuses to fall back to PID-only "
            "liveness (a recycled PID would then impersonate a dead holder)"
        ) from exc


def _read_proc_stat(pid: int) -> Optional[tuple[str, int]]:
    """Return `(state_char, start_ticks)` for `pid`, or None if it is gone.

    Raises OSError for anything that is not "the process does not exist" — a
    permission error must not be misread as a dead process.

    The comm field can contain spaces and parentheses, so the split is on the
    LAST ')': everything after it is field 3 onwards, making field 22
    (`starttime`) index 19.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            raw = fh.read()
    except FileNotFoundError:
        return None
    except ProcessLookupError:
        return None
    close_paren = raw.rfind(b")")
    if close_paren < 0:
        raise DeviceClaimUnreadable(f"/proc/{pid}/stat: no comm field terminator")
    fields = raw[close_paren + 1:].split()
    if len(fields) < 20:
        raise DeviceClaimUnreadable(
            f"/proc/{pid}/stat: only {len(fields)} fields after comm, need 20 for starttime"
        )
    return fields[0].decode("ascii", "replace"), int(fields[19])


def current_holder_identity(label: Optional[str] = None) -> dict:
    """Identity block for THIS process: pid, start ticks, boot id, host."""
    pid = os.getpid()
    stat = _read_proc_stat(pid)
    if stat is None:
        # Reading our own /proc entry cannot fail unless /proc is not mounted.
        raise DeviceClaimError(
            f"/proc/{pid}/stat does not exist for this process; /proc is required"
        )
    return {
        "pid": pid,
        "start_ticks": stat[1],
        "boot_id": _read_boot_id(),
        "host": socket.gethostname(),
        "label": label,
    }


def assess_holder_liveness(holder: Any) -> Liveness:
    """Is the process named by a claim payload still running?

    DEAD requires positive evidence: no `/proc` entry, a zombie, a start-tick
    mismatch (PID recycled), or a different boot id. Everything else that is not
    a match — another host, an unreadable `/proc`, a malformed record — is
    UNKNOWN, and UNKNOWN never authorises a reclaim.
    """
    if not isinstance(holder, Mapping):
        return Liveness(UNKNOWN, f"holder block is {type(holder).__name__}, not a mapping")
    pid = holder.get("pid")
    start_ticks = holder.get("start_ticks")
    boot_id = holder.get("boot_id")
    host = holder.get("host")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return Liveness(UNKNOWN, f"holder.pid is not a positive int: {pid!r}")
    if not isinstance(start_ticks, int) or isinstance(start_ticks, bool):
        return Liveness(UNKNOWN, f"holder.start_ticks is not an int: {start_ticks!r}")
    if not isinstance(boot_id, str) or not boot_id.strip():
        # NOT `dead`. A missing or malformed boot id is the absence of the
        # namespace that makes `start_ticks` mean anything, so the holder cannot
        # be shown to have died — it can only be shown to be unverifiable. The
        # `boot_id != local_boot` test below is a genuine death proof ONLY for a
        # well-formed id from a different boot; letting `None` fall into it
        # turned "we cannot check" into "positively dead" and made a LIVE holder
        # reclaimable, which is the one thing this module exists to prevent.
        return Liveness(
            UNKNOWN,
            f"holder.boot_id is not a non-empty string: {boot_id!r}; without a boot id "
            "the recorded start_ticks cannot be interpreted, and inability to evaluate "
            "is never evidence of death",
        )

    local_host = socket.gethostname()
    if host != local_host:
        # /proc only describes this machine. A claim from elsewhere is not
        # verifiable here, and an unverifiable claim is never a dead one.
        return Liveness(UNKNOWN, f"holder.host {host!r} is not this host {local_host!r}")

    local_boot = _read_boot_id()
    if boot_id != local_boot:
        return Liveness(
            DEAD,
            f"holder.boot_id {boot_id!r} predates the current boot {local_boot!r}, "
            "so its process cannot still be running",
        )

    try:
        stat = _read_proc_stat(pid)
    except PermissionError as exc:
        return Liveness(UNKNOWN, f"/proc/{pid}/stat is unreadable: {exc}")
    except DeviceClaimUnreadable as exc:
        return Liveness(UNKNOWN, str(exc))
    except OSError as exc:
        return Liveness(UNKNOWN, f"/proc/{pid}/stat could not be read: {exc}")

    if stat is None:
        return Liveness(DEAD, f"no /proc/{pid}: the holder process is gone")
    state_char, actual_ticks = stat
    if actual_ticks != start_ticks:
        return Liveness(
            DEAD,
            f"pid {pid} start_ticks {actual_ticks} != recorded {start_ticks}: the PID was "
            "recycled and the running process is a different one",
        )
    if state_char == "Z":
        # A zombie has already exited; the kernel released its flock at exit.
        # Without this branch an unreaped crash would look like a live holder
        # forever and the device would never be reclaimable.
        return Liveness(DEAD, f"pid {pid} is a zombie: it has exited and its locks are released")
    return Liveness(LIVE, f"pid {pid} is running with matching start_ticks {start_ticks}")


# =============================================================================
# Journal — a reclamation that is not recorded did not happen
# =============================================================================

class ClaimJournal:
    """Append-only JSONL sink for device-claim events.

    Deliberately tiny and local: the sharded AutoKernel event journal is a
    sibling AK1 deliverable, and this module must not fork it. What it does
    guarantee is that the records this module is *required* to emit —
    reclamations and defects — are durable before the action they describe
    becomes visible.

    There is no null/no-op journal and no default. Invariant 7 makes every
    outcome durable, and a fail-open default sink is precisely the shape that
    has poisoned this project's stores before while every component reported
    healthy: the reclaim would succeed, the record would vanish, and the next
    reader would see an unexplained owner change.
    """

    def __init__(self, path: os.PathLike | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, device_id: str, detail: Mapping[str, Any]) -> dict:
        """Append one record and fsync it. Raises on any write failure."""
        if not isinstance(kind, str) or not kind:
            raise ValueError("journal record kind must be a non-empty string")
        record = {
            "schema": JOURNAL_SCHEMA,
            "record_id": _new_id(_JOURNAL_ID_PREFIX),
            "kind": kind,
            "device_id": device_id,
            "created_at": _utc_now_iso(),
            "host": socket.gethostname(),
            "writer_pid": os.getpid(),
            "detail": dict(detail),
        }
        # canonical_json (not json.dumps) so these bytes match the encoding the
        # real journal will hash, and so a non-serialisable detail raises here
        # rather than producing an unstable repr.
        line = (canonical_json(record) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o666)
        try:
            # One O_APPEND write() of a short buffer to a local file is atomic
            # with respect to other appenders, which is what makes this safe for
            # the several processes that share a campaign journal.
            written = os.write(fd, line)
            if written != len(line):
                raise DeviceClaimError(
                    f"short journal write ({written} of {len(line)} bytes) to {self.path}"
                )
            os.fsync(fd)
        finally:
            os.close(fd)
        return record

    def read_all(self) -> list[dict]:
        """Every record, in append order. Raises on a malformed line.

        A journal that cannot be parsed is a broken journal; returning the
        readable prefix would let a caller conclude "no reclamation happened"
        from a truncated file.
        """
        if not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DeviceClaimUnreadable(
                        f"{self.path}:{lineno}: malformed journal line: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise DeviceClaimUnreadable(
                        f"{self.path}:{lineno}: journal line is not an object"
                    )
                out.append(record)
        return out


def _require_journal(journal: Any) -> Any:
    if journal is None:
        raise TypeError(
            "a device claim requires a journal: reclamations and revocation defects "
            "must be durable before they take effect (invariant 7). Pass a "
            "ClaimJournal; there is no no-op default on purpose."
        )
    if not callable(getattr(journal, "append", None)):
        raise TypeError(
            f"journal {type(journal).__name__} has no callable .append(kind, device_id, detail)"
        )
    return journal


# =============================================================================
# Receipt — the string an evaluation event cites
# =============================================================================

_RECEIPT_FIELDS = (
    "schema", "claim_id", "device_id", "lock_path", "state", "holder_pid",
    "holder_start_ticks", "holder_boot_id", "host", "holder_label", "purpose",
    "campaign_id", "acquired_at", "expires_at", "released_at", "reclaimed_from",
)


@dataclass(frozen=True)
class ClaimReceipt:
    """Immutable snapshot of one claim.

    `claim_id` is the string that goes into
    `evaluation_event.resource_claim_receipt` (schemas.py), which is how a
    measurement is bound to the exclusivity that produced it: without the
    receipt an event asserts a number with no evidence that anything else was
    kept off the device while it was taken.
    """

    claim_id: str
    device_id: str
    lock_path: str
    state: str
    holder_pid: int
    holder_start_ticks: int
    holder_boot_id: str
    host: str
    purpose: str
    campaign_id: str
    acquired_at: str
    holder_label: Optional[str] = None
    expires_at: Optional[str] = None
    released_at: Optional[str] = None
    reclaimed_from: Optional[dict] = None
    schema: str = RECEIPT_SCHEMA

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _RECEIPT_FIELDS}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ClaimReceipt":
        """Rebuild a receipt. Raises on a missing or unknown field.

        Tolerating either would let a receipt round-trip into a DIFFERENT
        receipt, and the whole point of the id is that it means one thing.
        """
        if not isinstance(obj, Mapping):
            raise TypeError(f"receipt must be a mapping, got {type(obj).__name__}")
        missing = [name for name in _RECEIPT_FIELDS if name not in obj]
        if missing:
            raise ValueError(f"receipt is missing required fields: {missing}")
        unknown = [name for name in obj if name not in _RECEIPT_FIELDS]
        if unknown:
            raise ValueError(f"receipt carries unknown fields: {sorted(unknown)}")
        if obj["schema"] != RECEIPT_SCHEMA:
            raise ValueError(f"receipt schema {obj['schema']!r} != {RECEIPT_SCHEMA!r}")
        return cls(**{name: obj[name] for name in _RECEIPT_FIELDS})


def _receipt_from_payload(payload: Mapping[str, Any], lock_path: Path,
                          released_at: Optional[str] = None) -> ClaimReceipt:
    holder = payload["holder"]
    return ClaimReceipt(
        claim_id=payload["claim_id"],
        device_id=payload["device_id"],
        lock_path=str(lock_path),
        state=payload["state"],
        holder_pid=holder["pid"],
        holder_start_ticks=holder["start_ticks"],
        holder_boot_id=holder["boot_id"],
        host=holder["host"],
        holder_label=holder.get("label"),
        purpose=payload["purpose"],
        campaign_id=payload["campaign_id"],
        acquired_at=payload["acquired_at"],
        expires_at=payload.get("expires_at"),
        released_at=released_at,
        reclaimed_from=payload.get("reclaimed_from"),
    )


# =============================================================================
# Payload I/O — always performed while holding the flock
# =============================================================================

def _write_payload(fd: int, payload: Mapping[str, Any]) -> None:
    data = (canonical_json(payload) + "\n").encode("utf-8")
    os.ftruncate(fd, 0)
    os.pwrite(fd, data, 0)
    os.fsync(fd)


def _clear_payload(fd: int) -> None:
    os.ftruncate(fd, 0)
    os.fsync(fd)


def _read_payload_fd(fd: int) -> Optional[dict]:
    """Parse the payload from an open fd. None if the file is empty (free)."""
    raw = os.pread(fd, _MAX_PAYLOAD_BYTES + 1, 0)
    if len(raw) > _MAX_PAYLOAD_BYTES:
        raise DeviceClaimUnreadable(
            f"claim payload exceeds {_MAX_PAYLOAD_BYTES} bytes; treating as corruption"
        )
    return _parse_payload_bytes(raw)


def _parse_payload_bytes(raw: bytes) -> Optional[dict]:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeviceClaimUnreadable(f"claim payload is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DeviceClaimUnreadable(
            f"claim payload is a {type(payload).__name__}, not an object"
        )
    if payload.get("schema") != DEVICE_CLAIM_SCHEMA:
        raise DeviceClaimUnreadable(
            f"claim payload schema {payload.get('schema')!r} != {DEVICE_CLAIM_SCHEMA!r}"
        )
    for key in ("claim_id", "device_id", "state", "holder", "acquired_at", "purpose",
                "campaign_id"):
        if key not in payload:
            raise DeviceClaimUnreadable(f"claim payload is missing {key!r}")
    return payload


def _read_payload_path(path: Path, *, attempts: int = 5,
                       retry_s: float = 0.02) -> Optional[dict]:
    """Read a claim payload WITHOUT taking the lock (revoker/observer path).

    The holder rewrites its payload in place (ftruncate + pwrite) while holding
    the lock, so an unlocked reader can catch a torn write. The window is a
    couple of microseconds and the write happens at most twice per claim, so a
    short bounded retry closes it. After the retries a parse failure is reported
    as unreadable — never as "no claim", which would read as "device free".
    """
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with open(path, "rb") as fh:
                raw = fh.read(_MAX_PAYLOAD_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(raw) > _MAX_PAYLOAD_BYTES:
            # The read above is bounded, so without this the oversize case is a
            # SILENT TRUNCATION: a valid-JSON prefix followed by megabytes of
            # anything parses fine here while `_read_payload_fd` rejects the same
            # file as corruption. Two readers of one file must not disagree about
            # whether it is a claim.
            raise DeviceClaimUnreadable(
                f"{path}: claim payload exceeds {_MAX_PAYLOAD_BYTES} bytes; treating as "
                "corruption (retrying cannot shrink it)"
            )
        try:
            return _parse_payload_bytes(raw)
        except DeviceClaimUnreadable as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(retry_s)
    raise DeviceClaimUnreadable(f"{path}: {last_exc}")


# =============================================================================
# flock helpers
# =============================================================================

def _try_flock_ex(fd: int) -> bool:
    """Non-blocking LOCK_EX. True on success, False if another OFD holds it."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return False
        raise


def _open_lock_fd(path: Path) -> int:
    """Open (creating on demand) the lock file.

    O_CLOEXEC matters: a candidate binary this loop launches must not inherit
    the device claim. Without it, `release()` would return while an exec'd child
    still pinned the lock through the inherited descriptor, and the claim would
    outlive the claimant invisibly.

    flock (not fcntl/lockf) is also deliberate: fcntl record locks are dropped
    when the process closes ANY descriptor to the file, so an unrelated
    open/close of the same path elsewhere in the process would silently release
    a held device claim.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)


def _unlock_and_close(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# =============================================================================
# Classification of an existing payload found under a FREE lock
# =============================================================================

_TAKE = "take"
_WAIT = "wait"
_REFUSE = "refuse"


@dataclass(frozen=True)
class _Disposition:
    action: str
    reason: str
    liveness: Optional[Liveness] = None
    previous: Optional[dict] = None


def _classify(payload: Optional[dict], *, stale_grace_s: float, now: float) -> _Disposition:
    """Decide what to do with a payload found while we hold the free lock.

    Three outcomes, and the third is the point:
      * TAKE   — no payload, or a positively dead holder past the grace period.
      * WAIT   — a dead holder still inside its grace period; retry later.
      * REFUSE — a live holder, or a holder whose liveness cannot be determined.
                 Never a takeover; the caller journals a defect and raises.
    """
    if payload is None:
        return _Disposition(_TAKE, "lock file carries no claim payload")

    liveness = assess_holder_liveness(payload.get("holder"))
    if liveness.state == LIVE:
        holder = payload.get("holder", {})
        self_note = ""
        if isinstance(holder, Mapping) and holder.get("pid") == os.getpid():
            self_note = (
                " — the payload names THIS process, so an earlier claim in this "
                "process released its lock without clearing its payload"
            )
        return _Disposition(
            _REFUSE,
            f"the lock is free but its recorded holder is alive ({liveness.reason}){self_note}. "
            "A live holder is never stolen from; if this claim really is abandoned the "
            "holder must release it, or a human must resolve the inconsistency.",
            liveness=liveness,
            previous=payload,
        )
    if liveness.state == UNKNOWN:
        return _Disposition(
            _REFUSE,
            f"the recorded holder's liveness cannot be determined ({liveness.reason}); "
            "inability to evaluate is not evidence of death, so the claim is not reclaimed",
            liveness=liveness,
            previous=payload,
        )

    acquired = _parse_iso(payload.get("acquired_at"), "acquired_at")
    age_s = now - acquired.timestamp()
    if age_s < stale_grace_s:
        return _Disposition(
            _WAIT,
            f"holder is dead ({liveness.reason}) but the claim is only {age_s:.3f}s old, "
            f"inside the {stale_grace_s:.3f}s reclaim grace period",
            liveness=liveness,
            previous=payload,
        )
    return _Disposition(
        _TAKE,
        f"holder is dead ({liveness.reason}) and the claim is {age_s:.3f}s old, past the "
        f"{stale_grace_s:.3f}s grace period",
        liveness=liveness,
        previous=payload,
    )


def _holder_summary(payload: Mapping[str, Any]) -> dict:
    holder = payload.get("holder")
    holder = dict(holder) if isinstance(holder, Mapping) else {}
    return {
        "claim_id": payload.get("claim_id"),
        "holder": holder,
        "acquired_at": payload.get("acquired_at"),
        "purpose": payload.get("purpose"),
        "campaign_id": payload.get("campaign_id"),
        "state": payload.get("state"),
    }


# =============================================================================
# The claim handle
# =============================================================================

class DeviceClaim:
    """A held, exclusive device claim. Construct via `acquire_device_claim`.

    Usable directly as a context manager (`with acquire_device_claim(...) as c`)
    and released exactly once, on the normal path and on an exception path
    alike. Release is idempotent so a `finally` block after a partial failure is
    safe.
    """

    def __init__(self, *, fd: int, lock_path: Path, revoke_path: Path,
                 payload: dict, journal: Any) -> None:
        self._fd = fd
        self._lock_path = lock_path
        self._revoke_path = revoke_path
        self._payload = payload
        self._journal = journal
        self._released = False
        self._final_receipt: Optional[ClaimReceipt] = None
        # The release records are tracked separately from the release itself:
        # dropping the lock is what makes the release real, but a release nobody
        # recorded is not durable (invariant 7), so a failed journal write must
        # stay retryable instead of being latched as done by `_released`.
        self._release_journaled = False
        self._release_context: Optional[dict] = None

    # -- identity ---------------------------------------------------------
    @property
    def claim_id(self) -> str:
        return self._payload["claim_id"]

    @property
    def device_id(self) -> str:
        return self._payload["device_id"]

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def held(self) -> bool:
        return not self._released

    def receipt(self) -> ClaimReceipt:
        """Snapshot receipt. After release it carries `released_at`."""
        if self._final_receipt is not None:
            return self._final_receipt
        return _receipt_from_payload(self._payload, self._lock_path)

    # -- revocation (quiesce and drain, holder side) -----------------------
    def revocation(self) -> Optional[dict]:
        """The revocation request addressed to THIS claim, if any.

        Call at your own task boundaries. Revocation is cooperative by design
        (`BUS_PROTOCOL.md:47-51`, fabric axiom 4): nothing preempts a running
        claim, so a holder that never asks is never interrupted — it is instead
        reported as a defect by `check_revocation_compliance`.
        """
        record = _read_revocation_file(self._revoke_path)
        if record is None:
            return None
        if record.get("claim_id") != self.claim_id:
            # Addressed to a previous holder of this device; not ours to honour.
            return None
        return record

    def acknowledge_revocation(self) -> dict:
        """Mark this claim `draining` and record the acknowledgement.

        Acknowledging does not release: the holder finishes the unit of work it
        is in and releases at its own boundary. That is the whole difference
        between a drain and a kill.
        """
        self._assert_held()
        record = self.revocation()
        if record is None:
            raise DeviceClaimError(
                f"no revocation is outstanding for claim {self.claim_id}; "
                "acknowledging a revocation that was never requested would put a "
                "false drain record in the journal"
            )
        if self._payload.get("revocation_acknowledged_at"):
            return record
        acknowledged_at = _utc_now_iso()
        self._payload = dict(self._payload)
        self._payload["state"] = STATE_DRAINING
        self._payload["revocation_acknowledged_at"] = acknowledged_at
        self._payload["revocation_id"] = record.get("revocation_id")
        _write_payload(self._fd, self._payload)
        self._journal.append(KIND_REVOCATION_ACKNOWLEDGED, self.device_id, {
            "claim_id": self.claim_id,
            "revocation_id": record.get("revocation_id"),
            "acknowledged_at": acknowledged_at,
            "requested_at": record.get("requested_at"),
        })
        return record

    # -- release ----------------------------------------------------------
    def release(self) -> ClaimReceipt:
        """Release the claim and journal it. Idempotent.

        The journal step is RETRIED by a repeat call until it succeeds. Caching
        "already released" over a failed record write would have made the
        durability step unretryable: the lock is gone, so no later actor can
        reconstruct the release, and the outcome would be permanently missing
        from the journal (invariant 7).
        """
        first_call = not self._released
        if first_call:
            released_at = _utc_now_iso()
            receipt = _receipt_from_payload(
                self._payload, self._lock_path, released_at=released_at
            )
            pending: Optional[dict] = None
            pending_error: Optional[str] = None
            try:
                pending = self.revocation()
            except DeviceClaimUnreadable as exc:
                pending_error = str(exc)

            clear_error: Optional[str] = None
            try:
                # The revocation marker is deliberately LEFT on disk. Deleting it
                # here would erase the only evidence that a drain was asked for, and
                # `check_revocation_compliance` could no longer tell "honoured" from
                # "never requested". Inheritance by the next holder is prevented
                # twice over instead: a revocation is addressed to one `claim_id`,
                # and the next acquisition discards a marker whose target is gone.
                _clear_payload(self._fd)
            except OSError as exc:
                # The lock is about to be dropped while a payload naming THIS
                # LIVE process is still on disk — the exact unresolvable state
                # `_classify` refuses to touch, which poisons the device for
                # every later claimant until this process exits. It cannot be
                # repaired from here, so it must not be silent: it is journaled
                # as a defect below and raised.
                clear_error = f"{type(exc).__name__}: {exc}"
            finally:
                # The lock is released no matter what. Holding a GPU because a
                # bookkeeping step failed is strictly worse than a missing record,
                # and the records are written immediately below.
                self._released = True
                self._final_receipt = receipt
                self._release_context = {
                    "released_at": released_at,
                    "pending": pending,
                    "pending_error": pending_error,
                    "clear_error": clear_error,
                    "acknowledged": bool(self._payload.get("revocation_acknowledged_at")),
                }
                _unlock_and_close(self._fd)

        self._journal_release()
        assert self._final_receipt is not None
        if first_call and self._release_context["clear_error"] is not None:
            raise DeviceClaimError(
                f"claim {self.claim_id} released its lock on {self._lock_path} but could "
                f"not clear its payload ({self._release_context['clear_error']}): the "
                "device now carries a claim naming a live process beside a free lock and "
                "is NOT claimable until a human truncates the file. A defect record was "
                "journaled."
            )
        return self._final_receipt

    def _journal_release(self) -> None:
        """Write the release records; a no-op once they are durable."""
        if self._release_journaled or self._release_context is None:
            return
        ctx = self._release_context
        assert self._final_receipt is not None
        if ctx["clear_error"] is not None:
            self._journal.append(KIND_DEFECT, self.device_id, {
                "defect_class": DEFECT_LIVE_HOLDER_FREE_LOCK,
                "reason": (
                    "the claim payload could not be cleared on release, so a payload "
                    "naming a live process was left beside a free lock: "
                    f"{ctx['clear_error']}"
                ),
                "claim_id": self.claim_id,
                "lock_path": str(self._lock_path),
                "observer_pid": os.getpid(),
            })
        if ctx["pending"] is not None:
            self._journal.append(KIND_REVOCATION_SATISFIED, self.device_id, {
                "claim_id": self.claim_id,
                "revocation_id": ctx["pending"].get("revocation_id"),
                "requested_at": ctx["pending"].get("requested_at"),
                "acknowledged": ctx["acknowledged"],
                "released_at": ctx["released_at"],
            })
        self._journal.append(KIND_RELEASED, self.device_id, {
            "claim_id": self.claim_id,
            "released_at": ctx["released_at"],
            "receipt": self._final_receipt.to_dict(),
            "revocation_read_error": ctx["pending_error"],
            "payload_clear_error": ctx["clear_error"],
        })
        self._release_journaled = True

    def _assert_held(self) -> None:
        if self._released:
            raise DeviceClaimError(f"claim {self.claim_id} has already been released")

    def __enter__(self) -> "DeviceClaim":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
        return None


# =============================================================================
# Acquisition
# =============================================================================

def acquire_device_claim(
    device_id: str,
    *,
    purpose: str,
    campaign_id: str,
    journal: ClaimJournal,
    holder_label: Optional[str] = None,
    timeout_s: Optional[float] = DEFAULT_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    stale_grace_s: float = DEFAULT_STALE_GRACE_S,
    max_hold_s: Optional[float] = None,
    lock_root: Optional[os.PathLike | str] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> DeviceClaim:
    """Acquire the exclusive claim on `device_id`, or raise.

    Args:
        device_id: e.g. `"mi210_0"`. Validated, never rewritten.
        purpose: why the device is being held. Required and non-empty — an
            unattributable claim is indistinguishable from a leaked one.
        campaign_id: the AutoKernel campaign (`ak-…`) or the session that owns
            this hold. Required and non-empty for the same reason.
        journal: where reclamations and releases are recorded. Required.
        timeout_s: budget in seconds. `None` blocks forever (write it out on
            purpose); `<= 0` makes exactly one attempt. NOTE this differs from
            `cpu_region_lock`, where `<= 0` blocks forever — the divergence is
            deliberate and fails in the visible direction.
        stale_grace_s: how old a dead holder's claim must be before it may be
            reclaimed, measured from its `acquired_at`.
        max_hold_s: declared maximum hold, recorded as `expires_at`. ADVISORY:
            an expired claim is never stolen, it is a reason to `request_
            revocation`.

    Raises:
        DeviceClaimTimeout: someone else held it for the whole budget.
        DeviceClaimInconsistent: the lock was free but the recorded holder is
            alive or unverifiable — a defect is journaled and nothing is taken.
    """
    device_id = _validated_device_id(device_id)
    journal = _require_journal(journal)
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("purpose must be a non-empty string")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign_id must be a non-empty string")
    if poll_s <= 0:
        raise ValueError(f"poll_s must be positive, got {poll_s!r}")
    if stale_grace_s < 0:
        raise ValueError(f"stale_grace_s must be >= 0, got {stale_grace_s!r}")
    if max_hold_s is not None and max_hold_s <= 0:
        raise ValueError(f"max_hold_s must be positive when set, got {max_hold_s!r}")

    lock_path = device_lock_path(device_id, lock_root)
    revoke_file = revocation_path(device_id, lock_root)
    holder = current_holder_identity(holder_label)

    deadline = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
    single_attempt = timeout_s is not None and timeout_s <= 0
    attempts = 0
    last_wait_reason = "device was held for the whole budget"

    while True:
        attempts += 1
        if cancel_check is not None and cancel_check():
            raise DeviceClaimTimeout(
                f"device claim on {device_id!r} cancelled before acquisition "
                f"(purpose={purpose!r})"
            )
        fd = _open_lock_fd(lock_path)
        keep_fd = False
        try:
            if _try_flock_ex(fd):
                try:
                    payload = _read_payload_fd(fd)
                    disposition = _classify(
                        payload, stale_grace_s=stale_grace_s, now=time.time()
                    )
                except DeviceClaimUnreadable as exc:
                    journal.append(KIND_DEFECT, device_id, {
                        "defect_class": DEFECT_UNVERIFIABLE_CLAIM,
                        "reason": str(exc),
                        "lock_path": str(lock_path),
                        "observer_pid": os.getpid(),
                    })
                    raise DeviceClaimInconsistent(
                        f"{lock_path}: {exc}. The claim is NOT reclaimed — a payload that "
                        "cannot be parsed cannot be shown to be abandoned. A human must "
                        "confirm no process is using the device and truncate the file."
                    ) from exc

                if disposition.action == _REFUSE:
                    journal.append(KIND_DEFECT, device_id, {
                        "defect_class": DEFECT_LIVE_HOLDER_FREE_LOCK,
                        "reason": disposition.reason,
                        "liveness": disposition.liveness.state if disposition.liveness else None,
                        "recorded_claim": _holder_summary(payload) if payload else None,
                        "observer_pid": os.getpid(),
                    })
                    raise DeviceClaimInconsistent(
                        f"device {device_id!r}: {disposition.reason}"
                    )

                if disposition.action == _TAKE:
                    reclaimed_from = None
                    if disposition.previous is not None:
                        reclaimed_from = _holder_summary(disposition.previous)
                        # Journal BEFORE the takeover: a reclamation that is not
                        # recorded must not be able to happen. If this raises,
                        # the finally block releases the lock and nothing moved.
                        journal.append(KIND_RECLAIMED, device_id, {
                            "reason": disposition.reason,
                            "liveness": (
                                disposition.liveness.state if disposition.liveness else None
                            ),
                            "liveness_reason": (
                                disposition.liveness.reason if disposition.liveness else None
                            ),
                            "stale_grace_s": stale_grace_s,
                            "reclaimed_from": reclaimed_from,
                            "reclaimed_by_pid": holder["pid"],
                        })
                    _discard_foreign_revocation(revoke_file, device_id, journal)
                    acquired_at = _utc_now_iso()
                    expires_at = None
                    if max_hold_s is not None:
                        expires_at = _utc_now_iso(time.time() + max_hold_s)
                    new_payload = {
                        "schema": DEVICE_CLAIM_SCHEMA,
                        "claim_id": _new_id(_CLAIM_ID_PREFIX),
                        "device_id": device_id,
                        "state": STATE_HELD,
                        "holder": holder,
                        "purpose": purpose,
                        "campaign_id": campaign_id,
                        "acquired_at": acquired_at,
                        "expires_at": expires_at,
                        "reclaimed_from": reclaimed_from,
                        "revocation_acknowledged_at": None,
                        "revocation_id": None,
                    }
                    _write_payload(fd, new_payload)
                    try:
                        claim = DeviceClaim(
                            fd=fd, lock_path=lock_path, revoke_path=revoke_file,
                            payload=new_payload, journal=journal,
                        )
                        journal.append(KIND_ACQUIRED, device_id, {
                            "claim_id": new_payload["claim_id"],
                            "receipt": claim.receipt().to_dict(),
                            "attempts": attempts,
                            "reclaimed": reclaimed_from is not None,
                        })
                    except BaseException:
                        # The payload is already on disk but this acquisition is
                        # failing, and the `finally` below is about to drop the
                        # lock. Clearing first is mandatory: a payload naming
                        # this LIVE process beside a free lock is exactly the
                        # unresolvable state `_classify` refuses to touch, so
                        # leaving it would poison the device for every later
                        # claimant until this process exits.
                        _clear_payload(fd)
                        raise
                    keep_fd = True
                    return claim

                last_wait_reason = disposition.reason
            else:
                last_wait_reason = "the device lock is held by another process"
        finally:
            if not keep_fd:
                _unlock_and_close(fd)

        if single_attempt or (deadline is not None and time.monotonic() >= deadline):
            raise DeviceClaimTimeout(
                f"could not acquire device {device_id!r} within "
                f"{'a single attempt' if single_attempt else f'{timeout_s}s'}: "
                f"{last_wait_reason}. Current holder: {_holder_note(lock_path)}"
            )
        time.sleep(poll_s)


def _holder_note(lock_path: Path) -> str:
    """Human-readable holder attribution for an error message ONLY.

    Best-effort is acceptable here and nowhere else in this module: this string
    never gates a decision, so an unreadable payload degrades the message rather
    than the exclusion.
    """
    try:
        payload = _read_payload_path(lock_path, attempts=1)
    except (DeviceClaimUnreadable, OSError) as exc:
        return f"(unreadable: {exc})"
    if payload is None:
        return "(no payload recorded)"
    holder = payload.get("holder", {})
    holder = holder if isinstance(holder, Mapping) else {}
    return (
        f"claim_id={payload.get('claim_id')} pid={holder.get('pid')} "
        f"campaign={payload.get('campaign_id')} purpose={payload.get('purpose')!r}"
    )


def _discard_foreign_revocation(revoke_file: Path, device_id: str, journal: Any) -> None:
    """Drop a revocation aimed at a claim that no longer exists.

    Without this, a drain order that its target never saw would be inherited by
    the next holder, which would then drain immediately for no reason. A
    revocation is always addressed to one `claim_id`.
    """
    try:
        record = _read_revocation_file(revoke_file)
    except DeviceClaimUnreadable as exc:
        journal.append(KIND_DEFECT, device_id, {
            "defect_class": DEFECT_UNVERIFIABLE_CLAIM,
            "reason": f"revocation file is unreadable: {exc}",
            "revocation_path": str(revoke_file),
        })
        raise
    if record is None:
        return
    journal.append(KIND_REVOCATION_DISCARDED, device_id, {
        "revocation_id": record.get("revocation_id"),
        "claim_id": record.get("claim_id"),
        "reason": "the targeted claim is no longer held; a new claim never inherits a "
                  "drain order addressed to its predecessor",
    })
    try:
        revoke_file.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def gpu_device_claim(device_id: str, **kwargs: Any):
    """`with gpu_device_claim("mi210_0", ...) as claim:` — released on exit."""
    claim = acquire_device_claim(device_id, **kwargs)
    try:
        yield claim
    finally:
        claim.release()


@contextmanager
def gpu_device_claims(device_ids: Iterable[str], **kwargs: Any):
    """All-or-nothing claim over several devices; LIFO release.

    Devices are acquired in sorted order so two callers requesting overlapping
    sets cannot deadlock against each other, and a partial acquisition releases
    what it already holds before propagating. Yields `{device_id: DeviceClaim}`.
    """
    ordered = sorted(set(device_ids))
    acquired: list[DeviceClaim] = []
    try:
        for device_id in ordered:
            acquired.append(acquire_device_claim(device_id, **kwargs))
        yield {claim.device_id: claim for claim in acquired}
    finally:
        # EVERY claim is released even if one release raises. A release can fail
        # (a journal write, a payload clear), and a bare loop stopped at the
        # first failure — stranding the remaining locks, which are held by THIS
        # process and which nothing else can free while it lives. The first
        # failure is re-raised after the rest are unwound.
        failures: list[BaseException] = []
        for claim in reversed(acquired):
            try:
                claim.release()
            except BaseException as exc:   # noqa: BLE001 - re-raised below
                failures.append(exc)
        if failures:
            raise failures[0]


# =============================================================================
# Revocation — requester side
# =============================================================================

def _read_revocation_file(path: Path, *, attempts: int = 5,
                          retry_s: float = 0.02) -> Optional[dict]:
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with open(path, "rb") as fh:
                raw = fh.read(_MAX_PAYLOAD_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(raw) > _MAX_PAYLOAD_BYTES:
            raise DeviceClaimUnreadable(
                f"{path}: revocation record exceeds {_MAX_PAYLOAD_BYTES} bytes; treating "
                "as corruption rather than parsing a truncated prefix of it"
            )
        text = raw.decode("utf-8", errors="replace").strip()
        if text:
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(retry_s)
                continue
            if not isinstance(record, dict):
                raise DeviceClaimUnreadable(f"{path}: revocation is not an object")
            if record.get("schema") != REVOCATION_SCHEMA:
                raise DeviceClaimUnreadable(
                    f"{path}: revocation schema {record.get('schema')!r} != "
                    f"{REVOCATION_SCHEMA!r}"
                )
            return record
        last_exc = ValueError("revocation file is empty")
        if attempt + 1 < attempts:
            time.sleep(retry_s)
    raise DeviceClaimUnreadable(f"{path}: {last_exc}")


def request_revocation(
    device_id: str,
    *,
    reason: str,
    requested_by: str,
    journal: ClaimJournal,
    drain_deadline_s: float,
    lock_root: Optional[os.PathLike | str] = None,
) -> dict:
    """Ask the current holder to drain and release. Never forcible.

    Writes a `revoking` marker addressed to the holder's `claim_id`. The holder
    reads it at ITS boundary, acknowledges, finishes the unit of work it is in,
    and releases (`BUS_PROTOCOL.md:47-51`: *"Never mid-decode, never a kill"*).
    Nothing here signals, kills, or unlocks anything.

    Raises if the device is not currently claimed: an untargeted revocation
    would be inherited by whoever claims the device next, which turns a drain
    request into a random future interruption.

    Re-asking for the SAME `claim_id` supersedes the outstanding marker but
    keeps the EARLIEST `drain_deadline_at` and the original `first_requested_at`.
    A revoker that nudges more often than its own drain bound would otherwise
    push the deadline forward indefinitely and make the ignored-revocation
    defect unreachable — the compliance check would be defeated by the act of
    asking again.
    """
    device_id = _validated_device_id(device_id)
    journal = _require_journal(journal)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if not isinstance(requested_by, str) or not requested_by.strip():
        raise ValueError("requested_by must be a non-empty string")
    if drain_deadline_s < 0:
        raise ValueError(f"drain_deadline_s must be >= 0, got {drain_deadline_s!r}")

    lock_path = device_lock_path(device_id, lock_root)
    payload = _read_payload_path(lock_path)
    if payload is None:
        raise DeviceClaimError(
            f"device {device_id!r} has no recorded claim, so there is nothing to revoke; "
            "a revocation is always addressed to one claim_id"
        )

    revoke_file = revocation_path(device_id, lock_root)
    try:
        previous = _read_revocation_file(revoke_file)
    except DeviceClaimUnreadable:
        previous = None

    requested_at_epoch = time.time()
    deadline_epoch = requested_at_epoch + drain_deadline_s
    first_requested_at = _utc_now_iso(requested_at_epoch)
    supersedes = None
    if previous is not None and previous.get("claim_id") == payload["claim_id"]:
        # Re-asking the SAME claim must never buy it more time. Without this, a
        # revoker that nudges more often than its own drain bound keeps pushing
        # `drain_deadline_at` into the future, `check_revocation_compliance` can
        # never reach FAIL, and the ignored-revocation defect becomes
        # unreachable — the check would be defeated by the act of asking again.
        # The EARLIEST outstanding deadline wins.
        supersedes = previous.get("revocation_id")
        first_requested_at = (previous.get("first_requested_at")
                              or previous.get("requested_at") or first_requested_at)
        try:
            prev_deadline = _parse_iso(previous.get("drain_deadline_at"),
                                       "drain_deadline_at")
        except DeviceClaimUnreadable:
            prev_deadline = None
        if prev_deadline is not None:
            deadline_epoch = min(deadline_epoch, prev_deadline.timestamp())

    record = {
        "schema": REVOCATION_SCHEMA,
        "revocation_id": _new_id(_REVOCATION_ID_PREFIX),
        "device_id": device_id,
        "claim_id": payload["claim_id"],
        "state": "revoking",
        "reason": reason,
        "requested_by": requested_by,
        "requested_by_pid": os.getpid(),
        "first_requested_at": first_requested_at,
        "supersedes": supersedes,
        "requested_at": _utc_now_iso(requested_at_epoch),
        "drain_deadline_s": float(drain_deadline_s),
        # The EARLIEST deadline among the outstanding requests for this claim,
        # which is why this is not simply `requested_at + drain_deadline_s`.
        "drain_deadline_at": _utc_now_iso(deadline_epoch),
        "target_holder": _holder_summary(payload),
    }

    _atomic_write_json(revoke_file, record)
    journal.append(KIND_REVOCATION_REQUESTED, device_id, {
        "revocation_id": record["revocation_id"],
        "claim_id": record["claim_id"],
        "reason": reason,
        "requested_by": requested_by,
        "drain_deadline_s": float(drain_deadline_s),
        "drain_deadline_at": record["drain_deadline_at"],
    })
    return record


def _atomic_write_json(path: Path, record: Mapping[str, Any]) -> None:
    """Write via a temp file + rename so a reader never sees a partial record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    data = (canonical_json(record) + "\n").encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o666)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def revocation_status(device_id: str,
                      lock_root: Optional[os.PathLike | str] = None) -> Optional[dict]:
    """The outstanding revocation record for `device_id`, or None."""
    return _read_revocation_file(revocation_path(device_id, lock_root))


def check_revocation_compliance(
    device_id: str,
    *,
    journal: ClaimJournal,
    lock_root: Optional[os.PathLike | str] = None,
    now: Optional[float] = None,
) -> Check:
    """Did the holder honour the outstanding revocation within its bound?

    PASS
        The revocation is satisfied — the device is now held by a different
        claim, or the targeted claim's payload is gone AND its flock is free.
        The lock probe is required: "no payload" alone was passable by
        truncating or deleting the lock file out from under a live holder, i.e.
        the check could be passed by deleting the thing it inspects.
    FAIL
        The deadline has passed and the targeted claim is still recorded as the
        holder without releasing. A `defect` record is journaled (rule 8: *"A
        revocation the holder ignores surfaces as a `defect`, never as a silent
        inconsistency"*), classed `revocation_ignored` when the holder is live
        and `revocation_orphaned` when it is dead — a holder that died did not
        ignore anything, and mislabelling it sends a human to escalate with the
        owner of a process that no longer exists. Nothing is killed.
    COULD_NOT_CHECK
        No revocation is outstanding (nothing to judge), the drain deadline has
        not elapsed yet (undecided, NOT compliant), a record is unreadable, or
        the payload is missing while the flock is still held (undecidable, and
        itself a defect a human must resolve).
    """
    device_id = _validated_device_id(device_id)
    journal = _require_journal(journal)
    moment = time.time() if now is None else now

    try:
        record = _read_revocation_file(revocation_path(device_id, lock_root))
    except DeviceClaimUnreadable as exc:
        return Check(COULD_NOT_CHECK, (f"revocation record is unreadable: {exc}",))
    if record is None:
        return Check(COULD_NOT_CHECK, (
            "no revocation is outstanding for this device; a satisfied revocation is "
            "recorded in the journal, not on disk",
        ))

    try:
        payload = _read_payload_path(device_lock_path(device_id, lock_root))
    except DeviceClaimUnreadable as exc:
        return Check(COULD_NOT_CHECK, (f"claim payload is unreadable: {exc}",))

    lock_path = device_lock_path(device_id, lock_root)
    if payload is None:
        # "No payload" has TWO causes and only one of them is compliance: the
        # holder released, or somebody truncated/deleted the lock file while the
        # holder still owns the flock. The lock root is a shared world-writable
        # directory, so the second is reachable by accident as well as on
        # purpose — and it turned this FAIL into a PASS, i.e. the check could be
        # passed by deleting the thing it inspects. Only a FREE lock proves the
        # holder let go.
        free = _probe_lock_free(lock_path)
        if free is None:
            return Check(COULD_NOT_CHECK, (
                f"device {device_id!r} carries no claim payload and its lock file "
                f"{lock_path} could not be probed, so the release cannot be confirmed",
            ))
        if not free:
            return Check(COULD_NOT_CHECK, (
                f"device {device_id!r} carries no claim payload but its flock is STILL "
                "HELD: the payload was truncated or deleted out from under a live "
                "holder. Compliance is undecidable and the missing payload is itself a "
                "defect — a human must establish who holds the device.",
            ))
        return Check(PASS, (
            f"claim {record.get('claim_id')!r} released {device_id!r} (no payload and a "
            "free lock): the revocation was honoured",
        ))

    if payload.get("claim_id") != record.get("claim_id"):
        return Check(PASS, (
            f"claim {record.get('claim_id')!r} is no longer the recorded holder of "
            f"{device_id!r}: the revocation was honoured",
        ))

    acknowledged = payload.get("revocation_acknowledged_at")
    try:
        deadline_at = _parse_iso(record.get("drain_deadline_at"), "drain_deadline_at")
    except DeviceClaimUnreadable as exc:
        return Check(COULD_NOT_CHECK, (str(exc),))

    if moment < deadline_at.timestamp():
        return Check(COULD_NOT_CHECK, (
            f"the drain deadline {record.get('drain_deadline_at')} has not elapsed; "
            "the holder is still entitled to finish the unit of work it is in, so "
            "compliance is not yet decidable"
            + (f" (acknowledged at {acknowledged})" if acknowledged else ""),
        ))

    # WHO is being accused matters. A holder that DIED holding the claim did not
    # ignore anything, and filing that as `revocation_ignored` points a human at
    # the owner of a process that no longer exists.
    liveness = assess_holder_liveness(payload.get("holder"))
    orphaned = liveness.state == DEAD
    journal.append(KIND_DEFECT, device_id, {
        "defect_class": (DEFECT_REVOCATION_ORPHANED if orphaned
                         else DEFECT_REVOCATION_IGNORED),
        "revocation_id": record.get("revocation_id"),
        "claim_id": record.get("claim_id"),
        "requested_at": record.get("requested_at"),
        "first_requested_at": record.get("first_requested_at"),
        "drain_deadline_at": record.get("drain_deadline_at"),
        "acknowledged_at": acknowledged,
        "observed_at": _utc_now_iso(moment),
        "holder_liveness": liveness.state,
        "holder_liveness_reason": liveness.reason,
        "flock_free": _probe_lock_free(lock_path),
        "note": (
            "the targeted holder is dead: the drain never completed and the claim is "
            "stale, but nothing was ignored — this is a crash-recovery item, not an "
            "escalation to a running owner."
            if orphaned else
            "quiesce-and-drain was not honoured within the declared bound; escalate "
            "to the claim's owner. This module never preempts a holder."
        ),
    })
    return Check(FAIL, (
        f"claim {record.get('claim_id')!r} still holds {device_id!r} past its drain "
        f"deadline {record.get('drain_deadline_at')}"
        + (f" (acknowledged at {acknowledged} but not released)" if acknowledged
           else " and never acknowledged the revocation")
        + (f"; its recorded holder is {liveness.state} ({liveness.reason})"
           if orphaned else ""),
    ))


# =============================================================================
# Checkers for the evaluator side
# =============================================================================

def check_device_claim_held(
    receipt: ClaimReceipt | Mapping[str, Any],
    *,
    lock_root: Optional[os.PathLike | str] = None,
) -> Check:
    """Is the claim named by `receipt` actually held right now?

    The evaluator calls this to bind a measurement to its exclusivity: an event
    citing a `resource_claim_receipt` whose claim is not held measured a device
    that anyone could have been sharing.

    PASS — the payload names this claim AND the lock is held.
    FAIL — a different claim (or none) holds the device, or the payload names
           this claim while the lock is free (the claim leaked).
    COULD_NOT_CHECK — the payload or the lock file cannot be read.
    """
    if isinstance(receipt, ClaimReceipt):
        receipt = receipt.to_dict()
    if not isinstance(receipt, Mapping):
        return Check(COULD_NOT_CHECK, (f"receipt is a {type(receipt).__name__}",))
    device_id = receipt.get("device_id")
    claim_id = receipt.get("claim_id")
    if not isinstance(device_id, str) or not isinstance(claim_id, str):
        return Check(COULD_NOT_CHECK, ("receipt lacks device_id/claim_id",))
    try:
        lock_path = device_lock_path(device_id, lock_root)
    except ValueError as exc:
        return Check(COULD_NOT_CHECK, (str(exc),))
    try:
        payload = _read_payload_path(lock_path)
    except (DeviceClaimUnreadable, OSError) as exc:
        return Check(COULD_NOT_CHECK, (f"claim payload unreadable: {exc}",))
    if payload is None:
        return Check(FAIL, (f"device {device_id!r} carries no claim payload",))
    if payload.get("claim_id") != claim_id:
        return Check(FAIL, (
            f"device {device_id!r} is recorded to claim {payload.get('claim_id')!r}, "
            f"not {claim_id!r}",
        ))
    free = _probe_lock_free(lock_path)
    if free is None:
        return Check(COULD_NOT_CHECK, (f"could not probe the lock file {lock_path}",))
    if free:
        return Check(FAIL, (
            f"claim {claim_id!r} is recorded on {device_id!r} but its flock is free: "
            "the claim leaked and nothing is excluding other processes",
        ))
    return Check(PASS, (f"claim {claim_id!r} holds {device_id!r}",))


def check_claim_expiry(
    device_id: str,
    *,
    lock_root: Optional[os.PathLike | str] = None,
    now: Optional[float] = None,
) -> Check:
    """Has the current claim outlived its declared `max_hold_s`?

    A FAIL here is a reason to call `request_revocation`, never a licence to
    reclaim: expiry is a declaration by the holder, not a fact about the holder,
    and a live process is never stolen from.
    """
    moment = time.time() if now is None else now
    try:
        payload = _read_payload_path(device_lock_path(device_id, lock_root))
    except (DeviceClaimUnreadable, OSError, ValueError) as exc:
        return Check(COULD_NOT_CHECK, (f"claim payload unreadable: {exc}",))
    if payload is None:
        return Check(COULD_NOT_CHECK, (f"device {device_id!r} carries no claim",))
    expires_at = payload.get("expires_at")
    if expires_at is None:
        return Check(COULD_NOT_CHECK, (
            f"claim {payload.get('claim_id')!r} declared no maximum hold, so expiry "
            "cannot be evaluated",
        ))
    try:
        deadline = _parse_iso(expires_at, "expires_at")
    except DeviceClaimUnreadable as exc:
        return Check(COULD_NOT_CHECK, (str(exc),))
    if moment <= deadline.timestamp():
        return Check(PASS, (f"claim {payload.get('claim_id')!r} expires at {expires_at}",))
    return Check(FAIL, (
        f"claim {payload.get('claim_id')!r} passed its declared expiry {expires_at}; "
        "request a revocation — an expired claim is still not reclaimable while its "
        "holder is alive",
    ))


def _probe_lock_free(lock_path: Path) -> Optional[bool]:
    """True if nothing holds the flock right now. ADVISORY ONLY.

    This is observation, and observation is not a claim (invariant 9). The
    answer is stale the instant it is returned; it exists for dashboards and
    for `check_device_claim_held`, never as a precondition for using a device.
    """
    try:
        fd = _open_lock_fd(lock_path)
    except OSError:
        return None
    try:
        if _try_flock_ex(fd):
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        return False
    except OSError:
        return None
    finally:
        os.close(fd)


def inspect_device_claim(device_id: str,
                         lock_root: Optional[os.PathLike | str] = None) -> dict:
    """Read-only diagnostic view of a device's claim state.

    ADVISORY: every field here is stale the moment it is returned. Nothing may
    decide to use a device from this dict — that is the `gpu_idle()` mistake
    this module replaces. Acquire the claim instead.
    """
    device_id = _validated_device_id(device_id)
    lock_path = device_lock_path(device_id, lock_root)
    out: dict = {
        "device_id": device_id,
        "lock_path": str(lock_path),
        "advisory": True,
        "observed_at": _utc_now_iso(),
    }
    try:
        payload = _read_payload_path(lock_path)
    except (DeviceClaimUnreadable, OSError) as exc:
        out["state"] = "unverifiable"
        out["error"] = str(exc)
        out["claim"] = None
        out["holder_liveness"] = UNKNOWN
        return out

    free = _probe_lock_free(lock_path)
    out["flock_free"] = free
    out["claim"] = payload
    if payload is None:
        out["state"] = "free" if free is not False else "locked_without_payload"
        out["holder_liveness"] = None
    else:
        liveness = assess_holder_liveness(payload.get("holder"))
        out["holder_liveness"] = liveness.state
        out["holder_liveness_reason"] = liveness.reason
        if free is False:
            out["state"] = (
                "revoking" if payload.get("state") == STATE_DRAINING else "held"
            )
        elif liveness.state == DEAD:
            out["state"] = "stale"
        else:
            out["state"] = "unverifiable"
    try:
        out["revocation"] = _read_revocation_file(revocation_path(device_id, lock_root))
    except DeviceClaimUnreadable as exc:
        out["revocation"] = None
        out["revocation_error"] = str(exc)
    return out
