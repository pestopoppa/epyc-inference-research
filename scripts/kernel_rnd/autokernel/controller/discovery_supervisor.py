"""Durable host-owned supervisor for AutoKernel discovery controllers.

The public launcher writes one sealed launch specification and asks a dedicated
tmux server to run this module.  tmux is the host-owned lifetime boundary: the
supervisor and controller do not remain children of the interactive agent that
requested the launch.  The supervisor itself owns exactly one controller
process group, records Linux PID identities before acting on them, and keeps a
private append-only death ledger.

Live mode deliberately accepts only the deployment factory's config-only CLI.
The hardware-free canary is a separate, bounded internal command used to prove
that the lifetime boundary survives the launching process.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


class SupervisorError(RuntimeError):
    pass


SPEC_SCHEMA = "epyc.autokernel.discovery_supervisor_spec.v1"
IDENTITY_SCHEMA = "epyc.autokernel.discovery_supervisor_identity.v1"
LEDGER_SCHEMA = "epyc.autokernel.discovery_supervisor_ledger.v1"
FACTORY_MODULE = (
    "scripts.kernel_rnd.autokernel.controller.discovery_deployment_factory"
)
SUPERVISOR_MODULE = (
    "scripts.kernel_rnd.autokernel.controller.discovery_supervisor"
)
TMUX_SOCKET_NAME = "epyc-autokernel-supervisors"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_FACTORY_PATH = Path(__file__).with_name("discovery_deployment_factory.py").resolve()
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SupervisorError(f"execution module is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_start_ticks(pid: int) -> tuple[str, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    close = raw.rfind(b")")
    fields = raw[close + 1:].split() if close >= 0 else []
    if len(fields) < 20:
        raise SupervisorError(f"/proc/{pid}/stat cannot prove process identity")
    return fields[0].decode("ascii", "replace"), int(fields[19])


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    if not value:
        raise SupervisorError("kernel boot id is empty")
    return value


def _host_identity() -> dict[str, str]:
    """Return a durable host discriminator without requiring systemd machine-id.

    This host intentionally has an empty ``/etc/machine-id``.  Prefer it when
    populated, but bind the kernel hostname when it is unavailable; the
    explicit source field prevents the two namespaces from being confused.
    """
    machine_id = Path("/etc/machine-id").read_text(encoding="ascii").strip()
    if machine_id:
        source, value = "machine-id", machine_id
    else:
        source, value = "kernel-hostname", socket.gethostname()
    if not value:
        raise SupervisorError("host identity source is empty")
    return {
        "host_id_source": source,
        "host_id_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def _process_identity(pid: int, *, pgid: int | None = None) -> dict[str, Any]:
    current = _read_start_ticks(pid)
    if current is None:
        raise SupervisorError(f"pid {pid} exited before identity capture")
    process_group = os.getpgid(pid) if pgid is None else pgid
    return {
        "pid": pid,
        "pgid": process_group,
        "start_ticks": current[1],
        "boot_id": _boot_id(),
        "host": socket.gethostname(),
        **_host_identity(),
    }


def _identity_liveness(identity: Mapping[str, Any]) -> tuple[str, str]:
    required = ("pid", "start_ticks", "boot_id", "host",
                "host_id_source", "host_id_sha256")
    if any(key not in identity for key in required):
        return "unknown", "identity is missing a required host/process field"
    if identity["host"] != socket.gethostname():
        return "unknown", "identity belongs to another hostname"
    local_host_identity = _host_identity()
    if (identity["host_id_source"] != local_host_identity["host_id_source"]
            or identity["host_id_sha256"] != local_host_identity["host_id_sha256"]):
        return "unknown", "identity belongs to another host identity namespace"
    if identity["boot_id"] != _boot_id():
        return "dead", "identity predates the current boot"
    pid = identity["pid"]
    ticks = identity["start_ticks"]
    if (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            or not isinstance(ticks, int) or isinstance(ticks, bool)):
        return "unknown", "identity PID/start ticks are malformed"
    current = _read_start_ticks(pid)
    if current is None:
        return "dead", "recorded PID no longer exists"
    if current[1] != ticks:
        return "dead", "recorded PID was recycled"
    if current[0] == "Z":
        return "dead", "recorded PID is a zombie"
    return "live", "PID, start ticks, boot id, and host identity match"


def _ensure_private_root(path: Path) -> Path:
    if not path.is_absolute():
        raise SupervisorError("runtime root must be absolute")
    resolved = path.resolve(strict=False)
    old_umask = os.umask(0o077)
    try:
        resolved.mkdir(parents=True, mode=0o700, exist_ok=True)
    finally:
        os.umask(old_umask)
    info = resolved.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SupervisorError("runtime root must be a real directory")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise SupervisorError("runtime root must be owned by this uid and mode 0700")
    return resolved


def _validate_private_stat(info: os.stat_result, path: Path, *,
                           max_bytes: int) -> None:
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise SupervisorError(
            f"private state must be an owned mode-0600 single-link regular file: {path}"
        )
    if info.st_size > max_bytes:
        raise SupervisorError(f"private state exceeds its byte ceiling: {path}")


def _require_private_file(path: Path, *, max_bytes: int = 1024 * 1024) -> os.stat_result:
    info = path.lstat()
    _validate_private_stat(info, path, max_bytes=max_bytes)
    return info


def _read_private_bytes(path: Path, *, max_bytes: int = 1024 * 1024) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _validate_private_stat(before, path, max_bytes=max_bytes)
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise SupervisorError(f"private state exceeds its byte ceiling: {path}")
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size)
                != (after.st_dev, after.st_ino, after.st_size)
                or total != after.st_size):
            raise SupervisorError(f"private state changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    _require_private_file(path, max_bytes=max(len(raw), 1))
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(value) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    raw = _read_private_bytes(path)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise SupervisorError(f"private state is not a JSON object: {path}")
    if raw != _canonical_bytes(value) + b"\n":
        raise SupervisorError(f"private state is not canonically encoded: {path}")
    return value


class DeathLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sequence = 0
        self.previous_sha256: str | None = None
        self.records: list[dict[str, Any]] = []
        if path.exists():
            raw = _read_private_bytes(path, max_bytes=64 * 1024 * 1024)
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if (not isinstance(row, dict) or set(row) != {
                        "schema", "sequence", "previous_sha256", "written_at",
                        "event", "payload", "record_sha256"}
                        or _canonical_bytes(row).decode("utf-8") != line
                        or row.get("schema") != LEDGER_SCHEMA
                        or row.get("sequence") != self.sequence + 1
                        or row.get("previous_sha256") != self.previous_sha256):
                    raise SupervisorError("death ledger hash chain is malformed")
                claimed = row.get("record_sha256")
                body = dict(row)
                body.pop("record_sha256", None)
                if claimed != _content_hash(body):
                    raise SupervisorError("death ledger record digest is invalid")
                self.sequence += 1
                self.previous_sha256 = claimed
                self.records.append(row)

    def append(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "schema": LEDGER_SCHEMA,
            "sequence": self.sequence + 1,
            "previous_sha256": self.previous_sha256,
            "written_at": _utc_now(),
            "event": event,
            "payload": dict(payload),
        }
        body["record_sha256"] = _content_hash(body)
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        try:
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
                raise SupervisorError("death ledger lost its private file identity")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            raw = _canonical_bytes(body) + b"\n"
            remaining = memoryview(raw)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise SupervisorError("death ledger append made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(self.path, 0o600)
        self.sequence += 1
        self.previous_sha256 = body["record_sha256"]
        self.records.append(body)
        return body


@dataclass(frozen=True)
class LaunchSpec:
    body: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return _content_hash(dict(self.body))

    @property
    def runtime_root(self) -> Path:
        return Path(str(self.body["runtime_root"]))

    @property
    def session_name(self) -> str:
        return f"ak-{self.sha256[:24]}"

    @classmethod
    def read(cls, path: Path) -> "LaunchSpec":
        value = _read_json(path)
        cls._validate(value)
        return cls(value)

    @staticmethod
    def _validate(value: Mapping[str, Any]) -> None:
        expected = {
            "schema", "kind", "runtime_root", "deployment", "validate_only",
            "canary", "python", "cwd", "restart_policy", "termination_policy",
            "execution_modules",
        }
        if set(value) != expected or value.get("schema") != SPEC_SCHEMA:
            raise SupervisorError("launch specification schema/keys are invalid")
        if value.get("kind") not in {"deployment", "canary"}:
            raise SupervisorError("launch specification kind is invalid")
        for key in ("runtime_root", "python", "cwd"):
            path = Path(str(value.get(key, "")))
            if not path.is_absolute() or ".." in path.parts:
                raise SupervisorError(f"launch specification {key} is not absolute")
        deployment = value.get("deployment")
        if value["kind"] == "deployment":
            if not isinstance(deployment, str) or not Path(deployment).is_absolute():
                raise SupervisorError("deployment launch lacks an absolute config path")
            if not isinstance(value.get("validate_only"), bool):
                raise SupervisorError("deployment validate_only is not boolean")
            if value.get("canary") is not None:
                raise SupervisorError("deployment launch carries canary authority")
        else:
            canary = value.get("canary")
            if deployment is not None or value.get("validate_only") is not True:
                raise SupervisorError("canary must be hardware-free validate-only")
            if (not isinstance(canary, dict) or set(canary) != {
                    "hold_seconds", "exit_code", "spawn_descendant"}):
                raise SupervisorError("canary contract is malformed")
            if (not isinstance(canary["hold_seconds"], float)
                    or not 0.2 <= canary["hold_seconds"] <= 120.0
                    or not isinstance(canary["exit_code"], int)
                    or isinstance(canary["exit_code"], bool)
                    or not 0 <= canary["exit_code"] <= 125
                    or not isinstance(canary["spawn_descendant"], bool)):
                raise SupervisorError("canary bounds are invalid")
        restart = value.get("restart_policy")
        termination = value.get("termination_policy")
        if (not isinstance(restart, dict)
                or set(restart) != {"max_restarts", "delay_seconds"}
                or not isinstance(restart["max_restarts"], int)
                or isinstance(restart["max_restarts"], bool)
                or not 0 <= restart["max_restarts"] <= 10
                or not isinstance(restart["delay_seconds"], float)
                or not 0.0 <= restart["delay_seconds"] <= 300.0):
            raise SupervisorError("restart policy is invalid")
        if value["kind"] == "deployment" and restart["max_restarts"] != 0:
            raise SupervisorError(
                "deployment restart requires a typed offline reconciliation receipt; "
                "none is implemented, so max_restarts must be zero"
            )
        if (not isinstance(termination, dict)
                or set(termination) != {"term_grace_seconds", "kill_grace_seconds"}
                or any(not isinstance(termination[key], float)
                       or not 0.1 <= termination[key] <= 60.0
                       for key in termination)):
            raise SupervisorError("termination policy is invalid")
        modules = value.get("execution_modules")
        if (not isinstance(modules, dict)
                or set(modules) != {"supervisor", "deployment_factory"}):
            raise SupervisorError("execution module binding is invalid")
        for binding in modules.values():
            if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                    or not Path(str(binding["path"])).is_absolute()
                    or _HEX64.fullmatch(str(binding["sha256"])) is None):
                raise SupervisorError("execution module identity is malformed")

    def verify_execution_modules(self) -> None:
        expected = self.body["execution_modules"]
        actual = {
            "supervisor": {
                "path": str(Path(__file__).resolve()),
                "sha256": _file_sha256(Path(__file__).resolve()),
            },
            "deployment_factory": {
                "path": str(_FACTORY_PATH),
                "sha256": _file_sha256(_FACTORY_PATH),
            },
        }
        if actual != expected:
            raise SupervisorError("supervisor/factory execution module bytes changed")
        if Path(str(self.body["cwd"])).resolve(strict=True) != _REPO_ROOT:
            raise SupervisorError("launch cwd no longer names this execution checkout")
        if (Path(str(self.body["python"])).resolve(strict=True)
                != Path(sys.executable).resolve(strict=True)):
            raise SupervisorError("launch Python no longer names this execution runtime")

    def child_argv(self) -> tuple[str, ...]:
        python = str(self.body["python"])
        if self.body["kind"] == "deployment":
            argv = [python, "-m", FACTORY_MODULE,
                    "--deployment", str(self.body["deployment"])]
            if self.body["validate_only"]:
                argv.append("--validate-only")
            return tuple(argv)
        canary = self.body["canary"]
        argv = [
            python, "-m", SUPERVISOR_MODULE, "_canary-child",
            "--hold-seconds", str(canary["hold_seconds"]),
            "--exit-code", str(canary["exit_code"]),
        ]
        if canary["spawn_descendant"]:
            argv.append("--spawn-descendant")
        return tuple(argv)


def _new_spec(*, runtime_root: Path, deployment: Path | None,
              validate_only: bool, canary: Mapping[str, Any] | None,
              max_restarts: int, restart_delay: float,
              term_grace: float, kill_grace: float) -> LaunchSpec:
    runtime = _ensure_private_root(runtime_root)
    kind = "canary" if canary is not None else "deployment"
    if kind == "deployment":
        if deployment is None:
            raise SupervisorError("deployment path is required")
        deployment_value = str(deployment.resolve(strict=True))
        if not Path(deployment_value).is_file():
            raise SupervisorError("deployment config is not a regular file")
    else:
        deployment_value = None
        validate_only = True
    body = {
        "schema": SPEC_SCHEMA,
        "kind": kind,
        "runtime_root": str(runtime),
        "deployment": deployment_value,
        "validate_only": validate_only,
        "canary": dict(canary) if canary is not None else None,
        "python": str(Path(sys.executable).resolve(strict=True)),
        "cwd": str(_REPO_ROOT),
        "restart_policy": {
            "max_restarts": max_restarts,
            "delay_seconds": float(restart_delay),
        },
        "termination_policy": {
            "term_grace_seconds": float(term_grace),
            "kill_grace_seconds": float(kill_grace),
        },
        "execution_modules": {
            "supervisor": {
                "path": str(Path(__file__).resolve()),
                "sha256": _file_sha256(Path(__file__).resolve()),
            },
            "deployment_factory": {
                "path": str(_FACTORY_PATH),
                "sha256": _file_sha256(_FACTORY_PATH),
            },
        },
    }
    LaunchSpec._validate(body)
    return LaunchSpec(body)


def _tmux(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("tmux", "-L", TMUX_SOCKET_NAME, *args),
        stdin=subprocess.DEVNULL, text=True, capture_output=True, check=check,
    )


def _tmux_has_session(session_name: str) -> bool:
    result = _tmux("has-session", "-t", f"={session_name}")
    return result.returncode == 0


def _persist_spec(path: Path, spec: LaunchSpec) -> None:
    raw = _canonical_bytes(dict(spec.body)) + b"\n"
    if path.exists():
        _require_private_file(path)
        if _read_private_bytes(path) != raw:
            raise SupervisorError(
                "runtime root is already bound to a different launch specification"
            )
        return
    _atomic_bytes(path, raw)


def _validate_process_record(value: Any, *, child: bool) -> None:
    expected = {
        "pid", "pgid", "start_ticks", "boot_id", "host",
        "host_id_source", "host_id_sha256",
    }
    if child:
        expected.add("argv_sha256")
    if not isinstance(value, dict) or set(value) != expected:
        raise SupervisorError("process identity schema is invalid")
    for key in ("pid", "pgid", "start_ticks"):
        if (not isinstance(value[key], int) or isinstance(value[key], bool)
                or value[key] <= 0):
            raise SupervisorError(f"process identity {key} is invalid")
    for key in ("boot_id", "host", "host_id_source"):
        if not isinstance(value[key], str) or not value[key]:
            raise SupervisorError(f"process identity {key} is invalid")
    if _HEX64.fullmatch(str(value["host_id_sha256"])) is None:
        raise SupervisorError("process host identity digest is invalid")
    if child and _HEX64.fullmatch(str(value["argv_sha256"])) is None:
        raise SupervisorError("child argv digest is invalid")


def _validate_identity(value: Mapping[str, Any], spec: LaunchSpec) -> None:
    expected = {
        "schema", "spec_sha256", "session_name", "tmux_socket_name", "state",
        "updated_at", "supervisor", "child", "restart_count", "exit_code",
    }
    if set(value) != expected or value.get("schema") != IDENTITY_SCHEMA:
        raise SupervisorError("supervisor identity schema/keys are invalid")
    if (value["spec_sha256"] != spec.sha256
            or value["session_name"] != spec.session_name
            or value["tmux_socket_name"] != TMUX_SOCKET_NAME):
        raise SupervisorError("supervisor identity is not bound to this launch spec")
    if value["state"] not in {"starting", "running", "stopped"}:
        raise SupervisorError("supervisor identity state is invalid")
    _validate_process_record(value["supervisor"], child=False)
    if value["child"] is not None:
        _validate_process_record(value["child"], child=True)
    if ((value["state"] == "running") != (value["child"] is not None)
            or (value["state"] != "stopped" and value["exit_code"] is not None)
            or (value["state"] == "stopped" and not isinstance(value["exit_code"], int))
            or not isinstance(value["restart_count"], int)
            or isinstance(value["restart_count"], bool)
            or value["restart_count"] < 0
            or not isinstance(value["updated_at"], str)
            or not value["updated_at"]):
        raise SupervisorError("supervisor identity state fields are inconsistent")


def _validate_ledger_records(ledger: DeathLedger, spec: LaunchSpec,
                             identity: Mapping[str, Any] | None) -> None:
    payload_keys = {
        "supervisor_started": {"spec_sha256", "session_name", "supervisor"},
        "child_started": {"restart_count", "child", "stdout", "stderr"},
        "signal_forwarded": {"signal", "child"},
        "child_exited": {"restart_count", "return_code", "cleanup_actions",
                         "stop_signal"},
        "restart_scheduled": {"restart_count", "delay_seconds",
                              "last_return_code"},
        "restarts_exhausted": {"restart_count", "max_restarts",
                               "last_return_code"},
        "supervisor_fault": {"exception_type", "message", "cleanup_actions",
                             "cleanup_error"},
        "supervisor_stopped": {"exit_code", "restart_count", "stop_signal"},
    }
    latest_start = None
    for row in ledger.records:
        event = row["event"]
        payload = row["payload"]
        if event not in payload_keys or not isinstance(payload, dict) \
                or set(payload) != payload_keys[event]:
            raise SupervisorError("death ledger event payload schema is invalid")
        if event == "supervisor_started":
            if (payload["spec_sha256"] != spec.sha256
                    or payload["session_name"] != spec.session_name):
                raise SupervisorError("death ledger is not bound to this launch spec")
            _validate_process_record(payload["supervisor"], child=False)
            latest_start = payload["supervisor"]
        elif event in {"child_started", "signal_forwarded"}:
            _validate_process_record(payload["child"], child=True)
            if event == "child_started":
                if (payload["stdout"] != str(spec.runtime_root / "controller.stdout.log")
                        or payload["stderr"]
                        != str(spec.runtime_root / "controller.stderr.log")):
                    raise SupervisorError("death ledger log paths escaped the runtime root")
    if identity is None:
        return
    if latest_start != identity["supervisor"]:
        raise SupervisorError("identity does not match the latest ledger supervisor")
    if identity["state"] == "stopped" and (
            not ledger.records or ledger.records[-1]["event"] != "supervisor_stopped"):
        raise SupervisorError("stopped identity lacks a terminal death-ledger record")


def _status_payload(runtime_root: Path) -> dict[str, Any]:
    root = _ensure_private_root(runtime_root)
    spec_path = root / "launch-spec.json"
    identity_path = root / "identity.json"
    if not spec_path.exists():
        return {"status": "absent", "runtime_root": str(root)}
    spec = LaunchSpec.read(spec_path)
    if spec.runtime_root.resolve(strict=True) != root:
        raise SupervisorError("launch spec is not self-bound to this runtime root")
    spec.verify_execution_modules()
    identity = _read_json(identity_path) if identity_path.exists() else None
    if identity is not None:
        _validate_identity(identity, spec)
    ledger_path = root / "death-ledger.jsonl"
    ledger = DeathLedger(ledger_path) if ledger_path.exists() else None
    if identity is not None and ledger is None:
        raise SupervisorError("supervisor identity exists without a death ledger")
    if ledger is not None:
        _validate_ledger_records(ledger, spec, identity)
    liveness = ("unknown", "supervisor identity has not been published")
    if isinstance(identity, dict) and isinstance(identity.get("supervisor"), dict):
        liveness = _identity_liveness(identity["supervisor"])
    return {
        "status": liveness[0],
        "reason": liveness[1],
        "runtime_root": str(root),
        "spec_sha256": spec.sha256,
        "session_name": spec.session_name,
        "tmux_session": _tmux_has_session(spec.session_name),
        "ledger_sequence": ledger.sequence if ledger is not None else 0,
        "identity": identity,
    }


def start_detached(spec: LaunchSpec, *, start_timeout: float = 10.0) -> dict[str, Any]:
    root = _ensure_private_root(spec.runtime_root)
    spec_path = root / "launch-spec.json"
    _persist_spec(spec_path, spec)
    current = _status_payload(root)
    if current["status"] == "live":
        if not current["tmux_session"]:
            raise SupervisorError("live supervisor has no matching tmux session")
        return {**current, "launch_result": "already_running"}
    if current["status"] == "unknown" and current.get("identity") is not None:
        raise SupervisorError("existing supervisor identity is not safely classifiable")
    if _tmux_has_session(spec.session_name):
        raise SupervisorError("tmux session exists without a matching live identity")
    command = (
        str(Path(sys.executable).resolve()), "-m", SUPERVISOR_MODULE,
        "_run", "--spec", str(spec_path),
    )
    result = _tmux(
        "new-session", "-d", "-s", spec.session_name,
        "-c", str(spec.body["cwd"]), "--", *command,
    )
    if result.returncode != 0:
        raise SupervisorError(f"tmux launch failed: {result.stderr.strip()}")
    deadline = time.monotonic() + start_timeout
    while time.monotonic() < deadline:
        current = _status_payload(root)
        if current["status"] == "live":
            return {**current, "launch_result": "started"}
        if not current["tmux_session"]:
            break
        time.sleep(0.05)
    raise SupervisorError("detached supervisor did not publish a live identity")


def _verify_child(identity: Mapping[str, Any]) -> None:
    state = _read_start_ticks(int(identity["pid"]))
    if state is None or state[1] != identity["start_ticks"]:
        raise SupervisorError("refusing to signal a missing or recycled child PID")
    if os.getpgid(int(identity["pid"])) != identity["pgid"]:
        raise SupervisorError("refusing to signal a child whose process group changed")
    if identity["pid"] != identity["pgid"]:
        raise SupervisorError("supervised child does not lead its private process group")


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise SupervisorError(f"owned process group {pgid} became unverifiable") from exc


def _signal_owned_group(identity: Mapping[str, Any], signum: int,
                        *, leader_may_be_dead: bool = False) -> bool:
    if not leader_may_be_dead:
        _verify_child(identity)
    try:
        os.killpg(int(identity["pgid"]), signum)
        return True
    except ProcessLookupError:
        return False


def _cleanup_group(identity: Mapping[str, Any], *, term_grace: float,
                   kill_grace: float, term_already_sent: bool) -> list[str]:
    pgid = int(identity["pgid"])
    actions: list[str] = []
    if not _group_exists(pgid):
        return actions
    if not term_already_sent:
        if _signal_owned_group(identity, signal.SIGTERM, leader_may_be_dead=True):
            actions.append("SIGTERM")
    deadline = time.monotonic() + term_grace
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _group_exists(pgid):
        if _signal_owned_group(identity, signal.SIGKILL, leader_may_be_dead=True):
            actions.append("SIGKILL")
        deadline = time.monotonic() + kill_grace
        while _group_exists(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _group_exists(pgid):
        raise SupervisorError(f"owned process group {pgid} survived SIGKILL")
    return actions


def _open_private_append(path: Path):
    descriptor = os.open(
        path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    info = os.fstat(descriptor)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
        os.close(descriptor)
        raise SupervisorError("private log lost its owned mode-0600 single-link identity")
    return os.fdopen(descriptor, "ab", buffering=0)


def _write_identity(path: Path, spec: LaunchSpec, *, state: str,
                    supervisor: Mapping[str, Any], child: Mapping[str, Any] | None,
                    restarts: int, exit_code: int | None = None) -> dict[str, Any]:
    value = {
        "schema": IDENTITY_SCHEMA,
        "spec_sha256": spec.sha256,
        "session_name": spec.session_name,
        "tmux_socket_name": TMUX_SOCKET_NAME,
        "state": state,
        "updated_at": _utc_now(),
        "supervisor": dict(supervisor),
        "child": dict(child) if child is not None else None,
        "restart_count": restarts,
        "exit_code": exit_code,
    }
    _atomic_json(path, value)
    return value


def supervise(spec_path: Path) -> int:
    spec = LaunchSpec.read(spec_path.resolve(strict=True))
    root = _ensure_private_root(spec.runtime_root)
    if spec_path.parent.resolve() != root:
        raise SupervisorError("launch specification is outside its runtime root")
    spec.verify_execution_modules()
    lock_path = root / "supervisor.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        lock_info = os.fstat(lock_fd)
        if (not stat.S_ISREG(lock_info.st_mode)
                or lock_info.st_uid != os.getuid() or lock_info.st_nlink != 1
                or stat.S_IMODE(lock_info.st_mode) != 0o600):
            raise SupervisorError(
                "singleton lock lost its owned mode-0600 single-link identity"
            )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError("another supervisor holds the singleton lock") from exc
        ledger = DeathLedger(root / "death-ledger.jsonl")
        _validate_ledger_records(ledger, spec, None)
        supervisor_identity = _process_identity(os.getpid())
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, _canonical_bytes(supervisor_identity) + b"\n")
        os.fsync(lock_fd)
        identity_path = root / "identity.json"
        ledger.append("supervisor_started", {
            "spec_sha256": spec.sha256,
            "session_name": spec.session_name,
            "supervisor": supervisor_identity,
        })
        _write_identity(identity_path, spec, state="starting",
                        supervisor=supervisor_identity, child=None, restarts=0)
        requested_signal = 0

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal requested_signal
            if requested_signal == 0:
                requested_signal = signum

        previous = {signum: signal.signal(signum, request_stop)
                    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        restarts = 0
        final_code = 0
        active_process: subprocess.Popen[bytes] | None = None
        active_identity: dict[str, Any] | None = None
        fault: Exception | None = None
        try:
            while True:
                stdout_path = root / "controller.stdout.log"
                stderr_path = root / "controller.stderr.log"
                with _open_private_append(stdout_path) as stdout_handle, \
                        _open_private_append(stderr_path) as stderr_handle:
                    process = subprocess.Popen(
                        spec.child_argv(), cwd=str(spec.body["cwd"]),
                        stdin=subprocess.DEVNULL, stdout=stdout_handle,
                        stderr=stderr_handle, start_new_session=True,
                        close_fds=True,
                    )
                child_identity = _process_identity(process.pid)
                active_process = process
                active_identity = child_identity
                if child_identity["pgid"] != process.pid:
                    process.kill()
                    process.wait(timeout=2.0)
                    raise SupervisorError("child failed to lead a private process group")
                child_identity["argv_sha256"] = _content_hash(list(spec.child_argv()))
                active_identity = child_identity
                ledger.append("child_started", {
                    "restart_count": restarts,
                    "child": child_identity,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                })
                _write_identity(identity_path, spec, state="running",
                                supervisor=supervisor_identity, child=child_identity,
                                restarts=restarts)
                forwarded = False
                while process.poll() is None and requested_signal == 0:
                    time.sleep(0.1)
                if requested_signal and process.poll() is None:
                    _signal_owned_group(child_identity, requested_signal)
                    forwarded = True
                    ledger.append("signal_forwarded", {
                        "signal": requested_signal,
                        "child": child_identity,
                    })
                    try:
                        process.wait(timeout=spec.body["termination_policy"][
                            "term_grace_seconds"])
                    except subprocess.TimeoutExpired:
                        pass
                return_code = process.poll()
                if return_code is None:
                    actions = _cleanup_group(
                        child_identity,
                        term_grace=0.0,
                        kill_grace=spec.body["termination_policy"]["kill_grace_seconds"],
                        term_already_sent=forwarded,
                    )
                    return_code = process.wait(timeout=spec.body[
                        "termination_policy"]["kill_grace_seconds"])
                else:
                    actions = _cleanup_group(
                        child_identity,
                        term_grace=spec.body["termination_policy"]["term_grace_seconds"],
                        kill_grace=spec.body["termination_policy"]["kill_grace_seconds"],
                        term_already_sent=forwarded,
                    )
                ledger.append("child_exited", {
                    "restart_count": restarts,
                    "return_code": return_code,
                    "cleanup_actions": actions,
                    "stop_signal": requested_signal or None,
                })
                active_process = None
                active_identity = None
                if requested_signal:
                    final_code = 128 + requested_signal
                    break
                if return_code == 0:
                    final_code = 0
                    break
                maximum = spec.body["restart_policy"]["max_restarts"]
                if restarts >= maximum:
                    ledger.append("restarts_exhausted", {
                        "restart_count": restarts,
                        "max_restarts": maximum,
                        "last_return_code": return_code,
                    })
                    final_code = int(return_code)
                    break
                restarts += 1
                delay = spec.body["restart_policy"]["delay_seconds"]
                ledger.append("restart_scheduled", {
                    "restart_count": restarts,
                    "delay_seconds": delay,
                    "last_return_code": return_code,
                })
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and requested_signal == 0:
                    time.sleep(min(0.1, deadline - time.monotonic()))
                if requested_signal:
                    final_code = 128 + requested_signal
                    break
        except Exception as exc:  # a fault must still own and close its child group
            fault = exc
            cleanup_actions: list[str] = []
            cleanup_error: str | None = None
            if active_process is not None and active_identity is not None:
                try:
                    if active_process.poll() is None:
                        _signal_owned_group(active_identity, signal.SIGTERM)
                        cleanup_actions.append("SIGTERM")
                    cleanup_actions.extend(_cleanup_group(
                        active_identity,
                        term_grace=spec.body["termination_policy"]["term_grace_seconds"],
                        kill_grace=spec.body["termination_policy"]["kill_grace_seconds"],
                        term_already_sent=bool(cleanup_actions),
                    ))
                    if active_process.poll() is None:
                        active_process.wait(timeout=spec.body[
                            "termination_policy"]["kill_grace_seconds"])
                except Exception as cleanup_exc:  # retain both faults durably
                    cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            final_code = 70
            ledger.append("supervisor_fault", {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "cleanup_actions": cleanup_actions,
                "cleanup_error": cleanup_error,
            })
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        ledger.append("supervisor_stopped", {
            "exit_code": final_code,
            "restart_count": restarts,
            "stop_signal": requested_signal or None,
        })
        _write_identity(identity_path, spec, state="stopped",
                        supervisor=supervisor_identity, child=None,
                        restarts=restarts, exit_code=final_code)
        if fault is not None:
            raise fault
        return final_code
    finally:
        os.close(lock_fd)


def stop_supervisor(runtime_root: Path, *, timeout: float = 15.0) -> dict[str, Any]:
    status = _status_payload(runtime_root)
    if status["status"] == "dead":
        return {**status, "stop_result": "already_stopped"}
    if status["status"] != "live":
        raise SupervisorError(f"supervisor is not safely signalable: {status['reason']}")
    identity = status["identity"]["supervisor"]
    before = _identity_liveness(identity)
    if before[0] != "live":
        raise SupervisorError("supervisor identity changed before signal")
    try:
        pidfd = os.pidfd_open(int(identity["pid"]), 0)
    except ProcessLookupError:
        return {**_status_payload(runtime_root), "stop_result": "already_stopped"}
    try:
        after_open = _identity_liveness(identity)
        if after_open[0] != "live":
            raise SupervisorError("supervisor identity changed while opening pidfd")
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
    finally:
        os.close(pidfd)
    deadline = time.monotonic() + timeout
    after = before
    while time.monotonic() < deadline:
        after = _identity_liveness(identity)
        if after[0] == "dead":
            break
        time.sleep(0.05)
    if after[0] != "dead":
        raise SupervisorError("supervisor did not stop after SIGTERM")
    return {**_status_payload(runtime_root), "stop_result": "stopped"}


def _canary_child(hold_seconds: float, exit_code: int,
                  spawn_descendant: bool) -> int:
    descendant = None
    if spawn_descendant:
        descendant = subprocess.Popen(
            (str(Path(sys.executable).resolve()), "-c",
             f"import time; time.sleep({hold_seconds + 60.0!r})"),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True,
        )
    payload = {
        "schema": "epyc.autokernel.discovery_supervisor_canary.v1",
        "pid": os.getpid(),
        "start_ticks": _read_start_ticks(os.getpid())[1],
        "descendant_pid": descendant.pid if descendant is not None else None,
        "hardware_accessed": False,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    time.sleep(hold_seconds)
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="launch a sealed deployment under tmux")
    start.add_argument("--deployment", required=True)
    start.add_argument("--runtime-root", required=True)
    start.add_argument("--validate-only", action="store_true")
    canary = sub.add_parser("canary", help="launch a hardware-free lifetime canary")
    canary.add_argument("--runtime-root", required=True)
    canary.add_argument("--hold-seconds", type=float, default=5.0)
    canary.add_argument("--exit-code", type=int, default=0)
    canary.add_argument("--spawn-descendant", action="store_true")
    start.add_argument("--max-restarts", type=int, default=0,
                       help="must remain 0 until typed offline reconciliation exists")
    canary.add_argument("--max-restarts", type=int, default=2)
    for command in (start, canary):
        command.add_argument("--restart-delay", type=float, default=2.0)
        command.add_argument("--term-grace", type=float, default=10.0)
        command.add_argument("--kill-grace", type=float, default=5.0)
    status = sub.add_parser("status")
    status.add_argument("--runtime-root", required=True)
    stop = sub.add_parser("stop")
    stop.add_argument("--runtime-root", required=True)
    stop.add_argument("--timeout", type=float, default=15.0)
    run = sub.add_parser("_run")
    run.add_argument("--spec", required=True)
    child = sub.add_parser("_canary-child")
    child.add_argument("--hold-seconds", required=True, type=float)
    child.add_argument("--exit-code", required=True, type=int)
    child.add_argument("--spawn-descendant", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "_run":
        return supervise(Path(args.spec))
    if args.command == "_canary-child":
        return _canary_child(args.hold_seconds, args.exit_code,
                             args.spawn_descendant)
    if args.command == "status":
        print(json.dumps(_status_payload(Path(args.runtime_root)), sort_keys=True))
        return 0
    if args.command == "stop":
        print(json.dumps(stop_supervisor(
            Path(args.runtime_root), timeout=args.timeout), sort_keys=True))
        return 0
    canary = None
    deployment = None
    validate_only = bool(getattr(args, "validate_only", False))
    if args.command == "canary":
        canary = {
            "hold_seconds": float(args.hold_seconds),
            "exit_code": args.exit_code,
            "spawn_descendant": args.spawn_descendant,
        }
    else:
        deployment = Path(args.deployment)
    spec = _new_spec(
        runtime_root=Path(args.runtime_root), deployment=deployment,
        validate_only=validate_only, canary=canary,
        max_restarts=args.max_restarts, restart_delay=args.restart_delay,
        term_grace=args.term_grace, kill_grace=args.kill_grace,
    )
    print(json.dumps(start_detached(spec), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SupervisorError as exc:
        print(f"discovery-supervisor: {exc}", file=sys.stderr)
        raise SystemExit(2)
