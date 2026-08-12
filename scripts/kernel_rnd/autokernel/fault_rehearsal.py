#!/usr/bin/env python3
"""Governed real OS-process fault rehearsal for AutoKernel.

This producer closes a deliberately narrow empirical gap.  It launches only
small Python children that this invocation creates, records their exact
PID/PGID/start-time identities, and exercises four host-process properties:

* an fsynced AutoKernel journal survives a planted child-process crash and is
  replayed by a fresh ``Journal`` instance;
* a revocation request against a disposable fake-device claim is advisory and
  does not preempt the live holder;
* owned-process cleanup uses identity-checked TERM followed by KILL only when
  the TERM grace expires, and verifies that the captured PID is dead; and
* changed bytes are refused when read through a hash-bound artifact seam.

The rehearsal has no inference, benchmark, build, GPU, kernel-tree, stack,
release, freeze, or production authority.  Its claim root lives below the new
output directory; it never reads or writes the shared live claim root.

Run from the research repository root::

    PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.fault_rehearsal \
      --output-dir /mnt/raid0/llm/autokernel/rehearsals/<new-campaign-id>

``--output-dir`` must not exist.  The complete directory is published by one
same-filesystem rename only after ``receipt.json`` has been fsynced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from . import journal as journal_mod
from .resource import device_claim as device_claim_mod


RECEIPT_SCHEMA = "epyc.autokernel.host_process_fault_rehearsal.v1"
CAPTURE_MODE = "measured_host_process_rehearsal"
CAMPAIGN_PREFIX = "ak-fault-rehearsal-"
DISPOSABLE_DEVICE_ID = "autokernel_rehearsal0"
PLANTED_CRASH_EXIT_CODE = 73
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_CHILD_MODES = frozenset(
    {
        "_journal_crash_child",
        "_journal_replay_child",
        "_claim_holder_child",
    }
)
EXPECTED_LEGS = (
    "durable_journal_crash_restart_replay",
    "resource_revocation_non_preemption",
    "hash_bound_artifact_tamper_refusal",
)


class RehearsalError(RuntimeError):
    """A rehearsal invariant failed; no degraded success is available."""


class ProcessOwnershipError(RehearsalError):
    """A captured PID no longer denotes the process this producer launched."""


class TamperRefusal(RehearsalError):
    """Hash-bound artifact bytes differ from the declared identity."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        mode,
    )
    try:
        written = os.write(fd, data)
        if written != len(data):
            raise RehearsalError(f"short write to {temporary}: {written} of {len(data)} bytes")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _canonical_bytes(value) + b"\n")


def _read_proc_start_ticks(pid: int) -> Optional[int]:
    """Return Linux ``/proc`` field 22 for one exact PID, or None if gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    close_paren = raw.rfind(b")")
    if close_paren < 0:
        raise ProcessOwnershipError(f"/proc/{pid}/stat has no comm terminator")
    fields = raw[close_paren + 1 :].split()
    if len(fields) < 20:
        raise ProcessOwnershipError(
            f"/proc/{pid}/stat has {len(fields)} post-comm fields; expected >=20"
        )
    return int(fields[19])


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    if not value:
        raise RehearsalError("kernel boot id is empty")
    return value


def _git_identity(source: Path) -> dict[str, Optional[str]]:
    """Resolve HEAD from git metadata without launching an untracked process."""
    cursor = source.resolve()
    if cursor.is_file():
        cursor = cursor.parent
    while cursor != cursor.parent and not (cursor / ".git").exists():
        cursor = cursor.parent
    dot_git = cursor / ".git"
    if not dot_git.exists():
        return {"root": None, "branch": None, "commit": None}
    if dot_git.is_file():
        line = dot_git.read_text(encoding="utf-8").strip()
        if not line.startswith("gitdir: "):
            raise RehearsalError(f"unrecognised gitdir marker at {dot_git}")
        git_dir = Path(line[8:])
        if not git_dir.is_absolute():
            git_dir = (cursor / git_dir).resolve()
    else:
        git_dir = dot_git
    common_dir = git_dir
    common_marker = git_dir / "commondir"
    if common_marker.exists():
        common_dir = (git_dir / common_marker.read_text(encoding="utf-8").strip()).resolve()
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    branch = None
    if head.startswith("ref: "):
        ref = head[5:]
        branch = ref.removeprefix("refs/heads/")
        commit = None
        for root in (git_dir, common_dir):
            ref_path = root / ref
            if ref_path.exists():
                commit = ref_path.read_text(encoding="ascii").strip()
                break
        if commit is None:
            packed = common_dir / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="ascii").splitlines():
                    if line.startswith(("#", "^")) or not line.strip():
                        continue
                    candidate, candidate_ref = line.split(" ", 1)
                    if candidate_ref == ref:
                        commit = candidate
                        break
    else:
        commit = head
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RehearsalError(f"HEAD is not a full commit id: {commit!r}")
    return {"root": str(cursor), "branch": branch, "commit": commit}


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_ticks: int
    boot_id: str
    argv: tuple[str, ...]
    argv_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "start_ticks": self.start_ticks,
            "boot_id": self.boot_id,
            "argv": list(self.argv),
            "argv_sha256": self.argv_sha256,
        }


@dataclass
class OwnedProcess:
    identity: ProcessIdentity
    handle: Any
    stdout_path: Path
    stderr_path: Path


class ProcessAdapter(Protocol):
    """Injected seam used by the ownership and escalation tests."""

    def spawn(self, argv: Sequence[str], stdout_path: Path, stderr_path: Path) -> OwnedProcess: ...

    def is_alive(self, process: OwnedProcess) -> bool: ...

    def wait(self, process: OwnedProcess, timeout_s: float) -> Optional[int]: ...

    def signal_group(self, process: OwnedProcess, signal_number: int) -> None: ...

    def verify_dead(self, process: OwnedProcess) -> bool: ...


class RealProcessAdapter:
    """Linux adapter that acts only on exact processes returned by ``spawn``."""

    @staticmethod
    def _assert_identity(process: OwnedProcess) -> None:
        identity = process.identity
        current_ticks = _read_proc_start_ticks(identity.pid)
        if current_ticks is None:
            raise ProcessLookupError(identity.pid)
        if current_ticks != identity.start_ticks:
            raise ProcessOwnershipError(
                f"pid {identity.pid} start_ticks changed from {identity.start_ticks} "
                f"to {current_ticks}; refusing to signal a recycled PID"
            )
        current_pgid = os.getpgid(identity.pid)
        if current_pgid != identity.pgid or identity.pgid != identity.pid:
            raise ProcessOwnershipError(
                f"pid {identity.pid} pgid identity is {current_pgid}, captured "
                f"{identity.pgid}; a producer child must lead its private group"
            )

    def spawn(self, argv: Sequence[str], stdout_path: Path, stderr_path: Path) -> OwnedProcess:
        exact_argv = tuple(str(item) for item in argv)
        if not exact_argv or Path(exact_argv[0]).resolve() != Path(sys.executable).resolve():
            raise RehearsalError("fault rehearsal may launch only this Python executable")
        if (
            len(exact_argv) != 5
            or exact_argv[1:3] != ("-m", "autokernel.fault_rehearsal")
            or exact_argv[3] not in ALLOWED_CHILD_MODES
        ):
            raise RehearsalError(
                "fault rehearsal may launch only its three exact internal child modes"
            )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            handle = subprocess.Popen(
                exact_argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                close_fds=True,
            )
        start_ticks = _read_proc_start_ticks(handle.pid)
        if start_ticks is None:
            handle.wait()
            raise RehearsalError("child exited before its process identity could be captured")
        pgid = os.getpgid(handle.pid)
        if pgid != handle.pid:
            handle.terminate()
            handle.wait(timeout=2.0)
            raise ProcessOwnershipError(
                f"new-session child {handle.pid} did not lead its process group {pgid}"
            )
        identity = ProcessIdentity(
            pid=handle.pid,
            pgid=pgid,
            start_ticks=start_ticks,
            boot_id=_boot_id(),
            argv=exact_argv,
            argv_sha256=_sha256_bytes(_canonical_bytes(list(exact_argv))),
        )
        return OwnedProcess(identity, handle, stdout_path, stderr_path)

    def is_alive(self, process: OwnedProcess) -> bool:
        if process.handle.poll() is not None:
            return False
        self._assert_identity(process)
        return True

    def wait(self, process: OwnedProcess, timeout_s: float) -> Optional[int]:
        try:
            return process.handle.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None

    def signal_group(self, process: OwnedProcess, signal_number: int) -> None:
        self._assert_identity(process)
        os.killpg(process.identity.pgid, signal_number)

    def verify_dead(self, process: OwnedProcess) -> bool:
        if process.handle.poll() is None:
            return False
        # wait()/poll() reaps the direct child.  A missing exact /proc identity
        # is the final proof that the captured PID is dead rather than merely a
        # zombie whose flock has already disappeared.
        return _read_proc_start_ticks(process.identity.pid) is None


def terminate_owned_process(
    process: OwnedProcess,
    adapter: ProcessAdapter,
    *,
    term_grace_s: float = 1.0,
    kill_grace_s: float = 2.0,
) -> dict[str, Any]:
    """TERM one captured private group, KILL only if needed, verify death."""
    actions: list[str] = []
    if adapter.is_alive(process):
        adapter.signal_group(process, signal.SIGTERM)
        actions.append("SIGTERM")
        exit_code = adapter.wait(process, term_grace_s)
        if exit_code is None:
            adapter.signal_group(process, signal.SIGKILL)
            actions.append("SIGKILL")
            exit_code = adapter.wait(process, kill_grace_s)
    else:
        exit_code = adapter.wait(process, 0.0)
    verified_dead = exit_code is not None and adapter.verify_dead(process)
    if not verified_dead:
        raise RehearsalError(
            f"captured child pid={process.identity.pid} was not verified dead after {actions}"
        )
    return {
        "actions": actions,
        "exit_code": exit_code,
        "verified_dead": True,
        "identity": process.identity.to_dict(),
    }


def read_hash_bound_artifact(path: Path, expected_sha256: str) -> bytes:
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be lowercase hexadecimal SHA-256")
    data = path.read_bytes()
    actual = _sha256_bytes(data)
    if actual != expected_sha256:
        raise TamperRefusal(
            f"artifact {path.name} sha256 {actual} != declared {expected_sha256}; "
            "changed bytes are refused"
        )
    return data


def _child_argv(mode: str, config_path: Path) -> tuple[str, ...]:
    return (
        str(Path(sys.executable).resolve()),
        "-m",
        "autokernel.fault_rehearsal",
        f"_{mode}",
        str(config_path),
    )


def _validated_child_config(config_path: Path, *, path_fields: Sequence[str]) -> dict[str, Any]:
    """Admit child I/O only below the producer-created private staging root."""
    config_path = config_path.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RehearsalError("child config must be an object")
    rehearsal_root = Path(config.get("rehearsal_root", "")).resolve(strict=True)
    if not rehearsal_root.name.endswith(".staging"):
        raise RehearsalError("child rehearsal root is not a private staging directory")
    mode = rehearsal_root.stat().st_mode & 0o777
    if rehearsal_root.stat().st_uid != os.getuid() or mode & 0o077:
        raise RehearsalError(
            f"child rehearsal root must be owned by uid {os.getuid()} with private mode; "
            f"observed uid={rehearsal_root.stat().st_uid} mode={oct(mode)}"
        )
    if not config_path.is_relative_to(rehearsal_root):
        raise RehearsalError("child config is outside its declared rehearsal root")
    for field in path_fields:
        value = config.get(field)
        if not isinstance(value, str) or not value:
            raise RehearsalError(f"child config field {field!r} must be a path string")
        resolved = Path(value).resolve(strict=False)
        if not resolved.is_relative_to(rehearsal_root):
            raise RehearsalError(
                f"child config field {field!r} escapes the rehearsal root: {resolved}"
            )
    return config


def _await_json(
    path: Path,
    process: OwnedProcess,
    adapter: ProcessAdapter,
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise RehearsalError(f"{path} did not contain an object")
            return value
        if not adapter.is_alive(process):
            raise RehearsalError(
                f"child pid={process.identity.pid} exited before publishing {path.name}"
            )
        time.sleep(0.01)
    raise RehearsalError(f"timed out waiting for child artifact {path}")


def _process_log_evidence(process: OwnedProcess) -> dict[str, Any]:
    return {
        "stdout": {
            "path": str(process.stdout_path.name),
            "sha256": _sha256_file(process.stdout_path),
            "size_bytes": process.stdout_path.stat().st_size,
        },
        "stderr": {
            "path": str(process.stderr_path.name),
            "sha256": _sha256_file(process.stderr_path),
            "size_bytes": process.stderr_path.stat().st_size,
        },
    }


def _journal_crash_leg(root: Path, campaign_id: str, adapter: ProcessAdapter) -> dict[str, Any]:
    leg_root = root / "journal-crash-restart"
    leg_root.mkdir()
    config_path = leg_root / "child-config.json"
    marker_path = leg_root / "crash-ready.json"
    config = {
        "rehearsal_root": str(root),
        "journal_root": str(leg_root / "journal"),
        "campaign_id": campaign_id,
        "marker_path": str(marker_path),
    }
    _atomic_write_json(config_path, config)
    process = adapter.spawn(
        _child_argv("journal_crash_child", config_path),
        leg_root / "child.stdout",
        leg_root / "child.stderr",
    )
    cleanup = None
    restart_process: Optional[OwnedProcess] = None
    try:
        marker = _await_json(marker_path, process, adapter)
        exit_code = adapter.wait(process, 5.0)
        if exit_code is None:
            cleanup = terminate_owned_process(process, adapter)
            raise RehearsalError("planted crash child did not exit within 5 seconds")
        if exit_code != PLANTED_CRASH_EXIT_CODE:
            raise RehearsalError(
                f"planted crash exited {exit_code}, expected {PLANTED_CRASH_EXIT_CODE}"
            )
        if not adapter.verify_dead(process):
            raise RehearsalError("planted crash child was not verified dead")

        # Restart/replay happens in a SECOND captured OS process.  It must
        # discover the same fsynced event before it appends the restart record.
        restart_marker_path = leg_root / "restart-replayed.json"
        restart_config_path = leg_root / "restart-config.json"
        _atomic_write_json(
            restart_config_path,
            {
                **config,
                "expected_event_id": marker["event_id"],
                "expected_events_digest": marker["events_digest"],
                "restart_marker_path": str(restart_marker_path),
            },
        )
        restart_process = adapter.spawn(
            _child_argv("journal_replay_child", restart_config_path),
            leg_root / "restart.stdout",
            leg_root / "restart.stderr",
        )
        restart_marker = _await_json(restart_marker_path, restart_process, adapter)
        restart_exit = adapter.wait(restart_process, 5.0)
        if restart_exit != 0 or not adapter.verify_dead(restart_process):
            raise RehearsalError(
                f"restart/replay child exit={restart_exit} was not a verified clean exit"
            )
        restarted = journal_mod.Journal(str(leg_root / "journal"), campaign_id=campaign_id)
        restarted.initialize()
        final_events = restarted.read_all()
        if [event.seq for event in final_events] != [1, 2]:
            raise RehearsalError("restart append did not continue the durable sequence")
        return {
            "name": "durable_journal_crash_restart_replay",
            "status": "PASS",
            "crash_process": process.identity.to_dict(),
            "crash_process_exit_code": exit_code,
            "crash_process_verified_dead": True,
            "restart_process": restart_process.identity.to_dict(),
            "restart_process_exit_code": restart_exit,
            "restart_process_verified_dead": True,
            "pre_crash_event_id": marker["event_id"],
            "pre_crash_events_digest": marker["events_digest"],
            "restart_event_id": restart_marker["restart_event_id"],
            "final_event_count": len(final_events),
            "final_events_digest": journal_mod.events_digest(final_events),
            "crash_logs": _process_log_evidence(process),
            "restart_logs": _process_log_evidence(restart_process),
        }
    finally:
        if restart_process is not None and adapter.is_alive(restart_process):
            terminate_owned_process(restart_process, adapter)
        if adapter.is_alive(process):
            cleanup = terminate_owned_process(process, adapter)
        if cleanup is not None:
            # Cleanup is deliberately not used to convert a failed crash into a
            # passing crash; it only prevents a leaked owned child.
            _atomic_write_json(leg_root / "cleanup.json", cleanup)


def _resource_revocation_leg(
    root: Path, campaign_id: str, adapter: ProcessAdapter
) -> dict[str, Any]:
    leg_root = root / "resource-revocation"
    leg_root.mkdir()
    lock_root = leg_root / "disposable-claims"
    lock_root.mkdir()
    claim_journal_path = leg_root / "claim-events.jsonl"
    marker_path = leg_root / "holder-ready.json"
    ack_path = leg_root / "holder-ack.json"
    config_path = leg_root / "child-config.json"
    config = {
        "rehearsal_root": str(root),
        "lock_root": str(lock_root),
        "claim_journal": str(claim_journal_path),
        "device_id": DISPOSABLE_DEVICE_ID,
        "campaign_id": campaign_id,
        "marker_path": str(marker_path),
        "ack_path": str(ack_path),
        "self_timeout_s": 30.0,
    }
    _atomic_write_json(config_path, config)
    process = adapter.spawn(
        _child_argv("claim_holder_child", config_path),
        leg_root / "child.stdout",
        leg_root / "child.stderr",
    )
    teardown = None
    try:
        ready = _await_json(marker_path, process, adapter)
        journal = device_claim_mod.ClaimJournal(claim_journal_path)
        request = device_claim_mod.request_revocation(
            DISPOSABLE_DEVICE_ID,
            reason="fault rehearsal verifies advisory non-preemption",
            requested_by="autokernel-fault-rehearsal",
            journal=journal,
            drain_deadline_s=0.05,
            lock_root=lock_root,
        )
        ack = _await_json(ack_path, process, adapter)
        if ack.get("revocation_id") != request["revocation_id"]:
            raise RehearsalError("holder acknowledged a different revocation id")
        time.sleep(0.08)
        compliance = device_claim_mod.check_revocation_compliance(
            DISPOSABLE_DEVICE_ID,
            journal=journal,
            lock_root=lock_root,
        )
        if compliance.outcome != device_claim_mod.FAIL:
            raise RehearsalError(
                f"ignored revocation reduced to {compliance.outcome}, expected FAIL"
            )
        if not adapter.is_alive(process):
            raise RehearsalError("revocation preempted the holder process")
        payload = device_claim_mod.inspect_device_claim(DISPOSABLE_DEVICE_ID, lock_root=lock_root)
        if payload.get("state") not in ("held", "revoking") or not payload.get("claim"):
            raise RehearsalError("holder lost its claim before owned teardown")
        teardown = terminate_owned_process(process, adapter)
        if teardown["actions"] not in (["SIGTERM"], ["SIGTERM", "SIGKILL"]):
            raise RehearsalError("owned teardown did not follow TERM→optional-KILL")
        records = journal.read_all()
        defect_classes = [
            row.get("detail", {}).get("defect_class")
            for row in records
            if row.get("kind") == device_claim_mod.KIND_DEFECT
        ]
        if device_claim_mod.DEFECT_REVOCATION_IGNORED not in defect_classes:
            raise RehearsalError("ignored revocation defect was not journaled")
        return {
            "name": "resource_revocation_non_preemption",
            "status": "PASS",
            "disposable_claim_root": "resource-revocation/disposable-claims",
            "device_id": DISPOSABLE_DEVICE_ID,
            "claim_id": ready["receipt"]["claim_id"],
            "revocation_id": request["revocation_id"],
            "acknowledged_by_holder": True,
            "compliance_outcome_while_alive": compliance.outcome,
            "holder_alive_after_deadline": True,
            "defect_class": device_claim_mod.DEFECT_REVOCATION_IGNORED,
            "teardown": teardown,
            "claim_journal_sha256": _sha256_file(claim_journal_path),
            "claim_journal_record_count": len(records),
            "logs": _process_log_evidence(process),
        }
    finally:
        if adapter.is_alive(process):
            teardown = terminate_owned_process(process, adapter)
        if teardown is not None:
            _atomic_write_json(leg_root / "teardown.json", teardown)


def _tamper_leg(root: Path) -> dict[str, Any]:
    leg_root = root / "tamper-refusal"
    leg_root.mkdir()
    artifact = leg_root / "artifact.bin"
    original = b"autokernel-host-process-fault-rehearsal-anchor-v1\n"
    _atomic_write_bytes(artifact, original)
    expected = _sha256_bytes(original)
    _atomic_write_json(
        leg_root / "artifact-manifest.json",
        {"artifact": artifact.name, "sha256": expected, "size_bytes": len(original)},
    )
    if read_hash_bound_artifact(artifact, expected) != original:
        raise RehearsalError("intact hash-bound artifact did not round-trip")
    tampered = original + b"tamper\n"
    _atomic_write_bytes(artifact, tampered)
    actual = _sha256_bytes(tampered)
    try:
        read_hash_bound_artifact(artifact, expected)
    except TamperRefusal as exc:
        refusal = str(exc)
    else:
        raise RehearsalError("tampered bytes passed the hash-bound artifact seam")
    return {
        "name": "hash_bound_artifact_tamper_refusal",
        "status": "PASS",
        "artifact": "tamper-refusal/artifact.bin",
        "declared_sha256": expected,
        "tampered_sha256": actual,
        "refusal_type": TamperRefusal.__name__,
        "refusal": refusal,
    }


def _environment_identity() -> dict[str, Any]:
    producer = Path(__file__).resolve()
    executable = Path(sys.executable).resolve()
    return {
        "host": socket.gethostname(),
        "boot_id": _boot_id(),
        "platform": platform.platform(),
        "kernel_release": platform.release(),
        "python_version": platform.python_version(),
        "python_executable": str(executable),
        "python_executable_sha256": _sha256_file(executable),
        "producer_path": str(producer),
        "producer_sha256": _sha256_file(producer),
        "source_tree": _git_identity(producer),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "runner_pid": os.getpid(),
        "runner_start_ticks": _read_proc_start_ticks(os.getpid()),
    }


def _authority_boundary() -> dict[str, bool]:
    return {
        "inference": False,
        "benchmark": False,
        "build": False,
        "gpu": False,
        "kernel_tree_write": False,
        "production_write": False,
        "stack_control": False,
        "release": False,
        "freeze": False,
        "promotion": False,
    }


def validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Validate the durable envelope without trusting producer-derived fields."""
    violations: list[str] = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        violations.append(f"schema must be {RECEIPT_SCHEMA!r}")
    if receipt.get("capture_mode") != CAPTURE_MODE:
        violations.append(f"capture_mode must be {CAPTURE_MODE!r}")
    campaign_id = receipt.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.startswith(CAMPAIGN_PREFIX):
        violations.append(f"campaign_id must start with {CAMPAIGN_PREFIX!r}")
    legs = receipt.get("legs")
    if not isinstance(legs, list):
        violations.append("legs must be a list")
        legs = []
    leg_names = [leg.get("name") for leg in legs if isinstance(leg, Mapping)]
    if leg_names != list(EXPECTED_LEGS):
        violations.append(f"legs must be exactly {list(EXPECTED_LEGS)!r} in order")
    leg_statuses = [leg.get("status") for leg in legs if isinstance(leg, Mapping)]
    if len(leg_statuses) != len(legs) or any(
        status not in ("PASS", "FAIL") for status in leg_statuses
    ):
        violations.append("every leg must be an object with status PASS or FAIL")
    derived_status = (
        "PASS"
        if len(leg_statuses) == len(EXPECTED_LEGS)
        and all(status == "PASS" for status in leg_statuses)
        else "FAIL"
    )
    if receipt.get("status") != derived_status:
        violations.append(f"status must derive to {derived_status!r} from leg statuses")
    if receipt.get("live_claim_root_touched") is not False:
        violations.append("live_claim_root_touched must be false")
    authority = receipt.get("authority")
    expected_authority = _authority_boundary()
    if authority != expected_authority:
        violations.append("authority must be the exact all-false authority boundary")
    source = (receipt.get("environment") or {}).get("source_tree")
    commit = source.get("commit") if isinstance(source, Mapping) else None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        violations.append("environment.source_tree.commit must be a full commit id")
    claimed_hash = receipt.get("receipt_sha256")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    actual_hash = _sha256_bytes(_canonical_bytes(body))
    if claimed_hash != actual_hash:
        violations.append(
            f"receipt_sha256 {claimed_hash!r} does not match canonical body {actual_hash}"
        )
    return violations


def run_fault_rehearsal(
    output_dir: os.PathLike[str] | str,
    *,
    campaign_id: Optional[str] = None,
    process_adapter: Optional[ProcessAdapter] = None,
) -> dict[str, Any]:
    """Run and atomically publish one real process-only rehearsal receipt."""
    target = Path(output_dir).expanduser().resolve(strict=False)
    if not target.is_relative_to(Path("/mnt/raid0").resolve()):
        raise ValueError("fault rehearsal output must live below /mnt/raid0")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    campaign_id = (
        campaign_id or f"{CAMPAIGN_PREFIX}{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )
    if not campaign_id.startswith(CAMPAIGN_PREFIX):
        raise ValueError(f"campaign_id must start with {CAMPAIGN_PREFIX!r}")
    stage = target.parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    stage.mkdir(mode=0o700)
    adapter = process_adapter or RealProcessAdapter()
    started_at = time.time()
    legs: list[dict[str, Any]] = []
    try:
        for leg_name, producer in (
            (
                "durable_journal_crash_restart_replay",
                lambda: _journal_crash_leg(stage, campaign_id, adapter),
            ),
            (
                "resource_revocation_non_preemption",
                lambda: _resource_revocation_leg(stage, campaign_id, adapter),
            ),
            ("hash_bound_artifact_tamper_refusal", lambda: _tamper_leg(stage)),
        ):
            try:
                legs.append(producer())
            except Exception as exc:
                legs.append(
                    {
                        "name": leg_name,
                        "status": "FAIL",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        passed = all(leg.get("status") == "PASS" for leg in legs) and len(legs) == 3
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "capture_mode": CAPTURE_MODE,
            "campaign_id": campaign_id,
            "status": "PASS" if passed else "FAIL",
            "started_at_epoch_s": started_at,
            "completed_at_epoch_s": time.time(),
            "environment": _environment_identity(),
            "authority": _authority_boundary(),
            "live_claim_root_touched": False,
            "process_selection": "captured_children_only_no_name_pattern_scan",
            "legs": legs,
        }
        receipt["receipt_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
        violations = validate_receipt(receipt)
        if violations:
            raise RehearsalError("producer created an invalid receipt: " + "; ".join(violations))
        _atomic_write_json(stage / "receipt.json", receipt)
        _fsync_dir(stage)
        os.replace(stage, target)
        _fsync_dir(target.parent)
        return receipt
    except BaseException:
        # Preserve a failed staging tree for diagnosis rather than deleting
        # durable fault evidence.  It remains clearly non-final because only
        # the exact requested target is a published result.
        raise


def _journal_crash_child(config_path: Path) -> int:
    config = _validated_child_config(config_path, path_fields=("journal_root", "marker_path"))
    journal = journal_mod.Journal(config["journal_root"], campaign_id=config["campaign_id"])
    journal.initialize()
    event = journal.append(
        journal_mod.KIND_PROPOSAL_SKIPPED,
        {
            "proposal_ref": "fault-rehearsal-planted-crash",
            "reason": "durability rehearsal event fsynced before intentional child exit",
        },
    )
    events = journal.read_all()
    _atomic_write_json(
        Path(config["marker_path"]),
        {
            "event_id": event.event_id,
            "events_digest": journal_mod.events_digest(events),
            "event_count": len(events),
        },
    )
    os._exit(PLANTED_CRASH_EXIT_CODE)


def _journal_replay_child(config_path: Path) -> int:
    config = _validated_child_config(
        config_path,
        path_fields=("journal_root", "marker_path", "restart_marker_path"),
    )
    journal = journal_mod.Journal(config["journal_root"], campaign_id=config["campaign_id"])
    journal.initialize()
    replayed = journal.read_all()
    if len(replayed) != 1 or replayed[0].event_id != config["expected_event_id"]:
        raise RehearsalError("restart process did not replay the exact pre-crash event")
    if journal_mod.events_digest(replayed) != config["expected_events_digest"]:
        raise RehearsalError("restart process replayed a different pre-crash digest")
    restart_event = journal.append(
        journal_mod.KIND_STOP_STATE,
        {"state": "fault_rehearsal_restart_replayed"},
    )
    final_events = journal.read_all()
    _atomic_write_json(
        Path(config["restart_marker_path"]),
        {
            "restart_event_id": restart_event.event_id,
            "final_event_count": len(final_events),
            "final_events_digest": journal_mod.events_digest(final_events),
        },
    )
    return 0


def _claim_holder_child(config_path: Path) -> int:
    config = _validated_child_config(
        config_path,
        path_fields=("lock_root", "claim_journal", "marker_path", "ack_path"),
    )
    if config.get("device_id") != DISPOSABLE_DEVICE_ID:
        raise RehearsalError(f"child device id must be the disposable {DISPOSABLE_DEVICE_ID!r}")
    claim_journal = device_claim_mod.ClaimJournal(config["claim_journal"])
    claim = device_claim_mod.acquire_device_claim(
        config["device_id"],
        purpose="harmless process-only non-preemption rehearsal",
        campaign_id=config["campaign_id"],
        journal=claim_journal,
        lock_root=config["lock_root"],
        timeout_s=1.0,
        poll_s=0.01,
        stale_grace_s=0.0,
        holder_label="autokernel-fault-rehearsal-owned-child",
    )
    stop = False

    def _term(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _term)
    _atomic_write_json(Path(config["marker_path"]), {"receipt": claim.receipt().to_dict()})
    deadline = time.monotonic() + float(config["self_timeout_s"])
    try:
        while not stop and time.monotonic() < deadline:
            revocation = claim.revocation()
            if revocation is not None:
                if claim.receipt().state != device_claim_mod.STATE_DRAINING:
                    claim.acknowledge_revocation()
                    _atomic_write_json(
                        Path(config["ack_path"]),
                        {
                            "revocation_id": revocation["revocation_id"],
                            "acknowledged_at_epoch_s": time.time(),
                        },
                    )
                # Deliberately remain alive and hold the claim.  This is the
                # non-preemption observation; the parent later terminates only
                # this exact child process under its captured identity.
            time.sleep(0.01)
    finally:
        claim.release()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="new atomic output directory")
    parser.add_argument("--campaign-id", help=f"optional id starting with {CAMPAIGN_PREFIX}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "_journal_crash_child":
        return _journal_crash_child(Path(args[1]))
    if args and args[0] == "_journal_replay_child":
        return _journal_replay_child(Path(args[1]))
    if args and args[0] == "_claim_holder_child":
        return _claim_holder_child(Path(args[1]))
    parsed = _parser().parse_args(args)
    receipt = run_fault_rehearsal(parsed.output_dir, campaign_id=parsed.campaign_id)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
