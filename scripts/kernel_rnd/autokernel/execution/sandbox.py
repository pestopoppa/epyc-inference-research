#!/usr/bin/env python3
"""Fail-closed candidate-process confinement for AutoKernel.

This module deliberately uses kernel facilities that are available to an
unprivileged process.  It does not call a container runtime and it does not
pretend that a path check is a sandbox:

* Landlock handles every filesystem write right and grants it only beneath one
  invocation-owned directory;
* seccomp denies signalling, ptrace/process-memory writes, networking, mount /
  namespace changes, kernel-module operations and BPF;
* the launcher refuses uid 0 and sets finite rlimits before ``execve``;
* every invocation joins a fresh cgroup-v2 leaf.  The parent verifies that the
  leaf is empty, uses ``cgroup.kill`` on escaped descendants when necessary,
  and removes it before reporting clean teardown.

The wrapper process is replaced by the candidate with ``execve``.  Therefore
the PID captured by the evaluator is the PID whose controls are attested and
whose cgroup is later drained; there is no untracked intermediate worker.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import json
import os
import resource
import secrets
import sys
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Mapping, Sequence


SANDBOX_ID = "autokernel.execution.sandbox/landlock-seccomp-cgroup-v1"
CGROUP_ROOT_ENV = "EPYC_AUTOKERNEL_CGROUP_ROOT"
HOST_CGROUP_ROOT = "/sys/fs/cgroup/autokernel"


class SandboxError(RuntimeError):
    """A required containment control was unavailable or contradicted itself."""


def default_cgroup_root() -> str:
    """Return the evaluator-selected cgroup parent without weakening failure.

    The host provisions one narrow, unprivileged delegation for AutoKernel.
    Older environments that deliberately made the cgroup-v2 mount root
    writable keep working; an unavailable or non-writable result is rejected
    by :class:`SandboxPolicy` before a candidate is spawned.
    """
    configured = os.environ.get(CGROUP_ROOT_ENV, "").strip()
    if configured:
        if not os.path.isabs(configured):
            raise SandboxError(f"{CGROUP_ROOT_ENV} must name an absolute path")
        return configured
    if os.path.isdir(HOST_CGROUP_ROOT):
        return HOST_CGROUP_ROOT
    return "/sys/fs/cgroup"


# x86_64 syscall numbers.  AutoKernel's production host is x86_64; refusing a
# different architecture is safer than silently installing the wrong filter.
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000
_AUDIT_ARCH_X86_64 = 0xC000003E

_BPF_LD_W_ABS = 0x20
_BPF_JMP_JEQ_K = 0x15
_BPF_RET_K = 0x06

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
_LANDLOCK_ACCESS_FS_REFER = 1 << 13
_LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14

_LANDLOCK_WRITE_V1 = (
    _LANDLOCK_ACCESS_FS_WRITE_FILE
    | _LANDLOCK_ACCESS_FS_REMOVE_DIR
    | _LANDLOCK_ACCESS_FS_REMOVE_FILE
    | _LANDLOCK_ACCESS_FS_MAKE_CHAR
    | _LANDLOCK_ACCESS_FS_MAKE_DIR
    | _LANDLOCK_ACCESS_FS_MAKE_REG
    | _LANDLOCK_ACCESS_FS_MAKE_SOCK
    | _LANDLOCK_ACCESS_FS_MAKE_FIFO
    | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | _LANDLOCK_ACCESS_FS_MAKE_SYM
)

# Default-deny would break the runtime's thread and GPU ioctl paths.  This is a
# narrow, explicit denial surface for capabilities a benchmark never needs.
# The names are carried into the receipt so the policy is auditable without
# decoding BPF bytecode.
_BLOCKED_SYSCALLS: Mapping[str, int] = {
    "kill": 62,
    "ptrace": 101,
    "rt_sigqueueinfo": 129,
    "pivot_root": 155,
    "mount": 165,
    "umount2": 166,
    "init_module": 175,
    "delete_module": 176,
    "tkill": 200,
    "tgkill": 234,
    "unshare": 272,
    "accept4": 288,
    "rt_tgsigqueueinfo": 297,
    "process_vm_writev": 311,
    "setns": 308,
    "sendmmsg": 307,
    "bpf": 321,
    "userfaultfd": 323,
    "pidfd_send_signal": 424,
    "socket": 41,
    "connect": 42,
    "accept": 43,
    "sendto": 44,
    "sendmsg": 46,
    "bind": 49,
    "listen": 50,
}


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort),
                ("filters", ctypes.POINTER(_SockFilter))]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
_LIBC.prctl.restype = ctypes.c_int


def _syscall(number: int, *args: Any) -> int:
    result = int(_LIBC.syscall(number, *args))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _prctl(option: int, arg2: Any, arg3: Any = 0) -> None:
    if _LIBC.prctl(option, arg2, arg3, 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def landlock_abi() -> int:
    """Return the kernel Landlock ABI, or raise instead of degrading."""
    try:
        abi = _syscall(_SYS_LANDLOCK_CREATE_RULESET, 0, 0,
                       _LANDLOCK_CREATE_RULESET_VERSION)
    except OSError as exc:
        raise SandboxError(f"Landlock ABI query failed: {exc}") from exc
    if abi < 1:
        raise SandboxError(f"Landlock ABI {abi} cannot enforce filesystem writes")
    return abi


def _landlock_write_rights(abi: int) -> int:
    rights = _LANDLOCK_WRITE_V1
    if abi >= 2:
        rights |= _LANDLOCK_ACCESS_FS_REFER
    if abi >= 3:
        rights |= _LANDLOCK_ACCESS_FS_TRUNCATE
    return rights


def install_landlock(writable_root: str) -> tuple[int, int]:
    """Handle every write right and grant it only under ``writable_root``."""
    root = Path(writable_root).resolve(strict=True)
    if not root.is_dir():
        raise SandboxError(f"sandbox writable root is not a directory: {root}")
    abi = landlock_abi()
    rights = _landlock_write_rights(abi)
    attr = _LandlockRulesetAttr(rights)
    try:
        ruleset_fd = _syscall(
            _SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    except OSError as exc:
        raise SandboxError(f"Landlock ruleset creation failed: {exc}") from exc
    path_fd = -1
    try:
        path_fd = os.open(root, os.O_PATH | os.O_CLOEXEC)
        path_attr = _LandlockPathBeneathAttr(rights, path_fd, 0)
        _syscall(_SYS_LANDLOCK_ADD_RULE, ruleset_fd, _LANDLOCK_RULE_PATH_BENEATH,
                 ctypes.byref(path_attr), 0)
        _prctl(_PR_SET_NO_NEW_PRIVS, 1)
        _syscall(_SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    except OSError as exc:
        raise SandboxError(f"Landlock activation failed: {exc}") from exc
    finally:
        if path_fd >= 0:
            os.close(path_fd)
        os.close(ruleset_fd)
    return abi, rights


def _statement(code: int, k: int) -> _SockFilter:
    return _SockFilter(code, 0, 0, k)


def _jump(code: int, k: int, jt: int, jf: int) -> _SockFilter:
    return _SockFilter(code, jt, jf, k)


def install_seccomp() -> str:
    """Install the candidate deny policy and return its content identity."""
    if os.uname().machine != "x86_64":
        raise SandboxError(
            f"seccomp policy is compiled for x86_64, not {os.uname().machine!r}")
    filters = [
        _statement(_BPF_LD_W_ABS, 4),
        _jump(_BPF_JMP_JEQ_K, _AUDIT_ARCH_X86_64, 1, 0),
        _statement(_BPF_RET_K, _SECCOMP_RET_KILL_PROCESS),
        _statement(_BPF_LD_W_ABS, 0),
    ]
    for number in sorted(set(_BLOCKED_SYSCALLS.values())):
        filters.extend((
            _jump(_BPF_JMP_JEQ_K, number, 0, 1),
            _statement(_BPF_RET_K, _SECCOMP_RET_ERRNO | errno.EPERM),
        ))
    filters.append(_statement(_BPF_RET_K, _SECCOMP_RET_ALLOW))
    array_type = _SockFilter * len(filters)
    array = array_type(*filters)
    program = _SockFprog(len(filters), array)
    try:
        _prctl(_PR_SET_NO_NEW_PRIVS, 1)
        _prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(program))
    except OSError as exc:
        raise SandboxError(f"seccomp activation failed: {exc}") from exc
    return _seccomp_policy_sha256()


def _seccomp_policy_sha256() -> str:
    return hashlib.sha256(json.dumps(
        sorted(_BLOCKED_SYSCALLS.items()), separators=(",", ":")
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResourceLimits:
    """Finite process-local ceilings that remain active after exec."""

    address_space_bytes: int = 2 * (1 << 40)
    file_size_bytes: int = 16 * (1 << 30)
    open_files: int = 4096
    processes: int = 32768
    cpu_time_s: int = 8 * 3600

    def __post_init__(self) -> None:
        for name in ("address_space_bytes", "file_size_bytes", "open_files",
                     "processes", "cpu_time_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int")

    def to_dict(self) -> dict:
        return {
            "address_space_bytes": self.address_space_bytes,
            "file_size_bytes": self.file_size_bytes,
            "open_files": self.open_files,
            "processes": self.processes,
            "cpu_time_s": self.cpu_time_s,
        }


def install_resource_limits(limits: ResourceLimits) -> None:
    resource.setrlimit(resource.RLIMIT_AS,
                       (limits.address_space_bytes, limits.address_space_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE,
                       (limits.file_size_bytes, limits.file_size_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE,
                       (limits.open_files, limits.open_files))
    resource.setrlimit(resource.RLIMIT_NPROC,
                       (limits.processes, limits.processes))
    resource.setrlimit(resource.RLIMIT_CPU,
                       (limits.cpu_time_s, limits.cpu_time_s))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


@dataclass(frozen=True)
class SandboxPolicy:
    writable_root: str
    cgroup_root: str = dataclass_field(default_factory=default_cgroup_root)
    limits: ResourceLimits = ResourceLimits()
    token: str = ""

    def __post_init__(self) -> None:
        root = Path(self.writable_root).resolve(strict=True)
        cgroup = Path(self.cgroup_root).resolve(strict=True)
        if not root.is_dir():
            raise SandboxError(f"writable_root is not a directory: {root}")
        if not cgroup.is_dir() or not os.access(cgroup, os.W_OK):
            raise SandboxError(f"cgroup root is not a writable directory: {cgroup}")
        if os.geteuid() == 0:
            raise SandboxError("candidate sandbox refuses a root execution identity")
        if not isinstance(self.limits, ResourceLimits):
            raise TypeError("limits must be ResourceLimits")
        token = self.token or secrets.token_hex(8)
        if not token.isalnum() or len(token) > 64:
            raise ValueError("sandbox token must be 1..64 alphanumeric characters")
        object.__setattr__(self, "writable_root", str(root))
        object.__setattr__(self, "cgroup_root", str(cgroup))
        object.__setattr__(self, "token", token)
        # Constructor is the fail-closed availability check.  No process is
        # launched merely to discover that containment is impossible.
        landlock_abi()

    def cgroup_path(self, pid: int) -> Path:
        return Path(self.cgroup_root, f"autokernel-{pid}-{self.token}")

    def encode(self, *, receipt_path: str) -> str:
        receipt = Path(receipt_path).resolve()
        writable = Path(self.writable_root)
        try:
            receipt.relative_to(writable)
        except ValueError:
            pass
        else:
            raise SandboxError(
                "sandbox receipt must be evaluator-owned outside the candidate's "
                "writable tree")
        document = {
            "writable_root": self.writable_root,
            "cgroup_root": self.cgroup_root,
            "limits": self.limits.to_dict(),
            "token": self.token,
            "receipt_path": str(receipt),
        }
        return base64.urlsafe_b64encode(json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).decode("ascii")

    def wrap(self, argv: Sequence[str], *, receipt_path: str) -> tuple[str, ...]:
        command = tuple(str(item) for item in argv)
        if not command:
            raise ValueError("sandboxed argv must be non-empty")
        return (sys.executable, str(Path(__file__).resolve()), "--policy",
                self.encode(receipt_path=receipt_path), "--", *command)


def _decode_policy(encoded: str) -> tuple[SandboxPolicy, Path]:
    try:
        raw = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
        limits = ResourceLimits(**raw["limits"])
        policy = SandboxPolicy(
            writable_root=raw["writable_root"], cgroup_root=raw["cgroup_root"],
            limits=limits, token=raw["token"])
        receipt = Path(raw["receipt_path"]).resolve()
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise SandboxError(f"invalid encoded sandbox policy: {exc}") from exc
    root = Path(policy.writable_root)
    try:
        receipt.relative_to(root)
    except ValueError:
        pass
    else:
        raise SandboxError(
            "sandbox receipt must be evaluator-owned outside the candidate's "
            "writable tree")
    if not receipt.parent.is_dir():
        raise SandboxError(f"sandbox receipt parent does not exist: {receipt.parent}")
    return policy, receipt


def _join_owned_cgroup(policy: SandboxPolicy) -> Path:
    path = policy.cgroup_path(os.getpid())
    try:
        path.mkdir(mode=0o700)
        Path(path, "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
    except OSError as exc:
        try:
            path.rmdir()
        except OSError:
            pass
        raise SandboxError(f"could not create/join owned cgroup {path}: {exc}") from exc
    return path


def _open_receipt(path: Path) -> int:
    """Open evaluator-owned evidence before Landlock, never inherited by argv.

    The candidate must not be able to rewrite the activation receipt it is
    being judged by.  The trusted wrapper opens a fresh file in the evaluator's
    sibling directory before installing Landlock, writes it only after every
    control is active, and closes the descriptor before ``execve``.
    """
    try:
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                       0o600)
    except OSError as exc:
        raise SandboxError(f"cannot create evaluator-owned sandbox receipt {path}: {exc}") \
            from exc


def _write_receipt(fd: int, document: Mapping[str, Any]) -> None:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    encoded = payload.encode("utf-8")
    offset = 0
    while offset < len(encoded):
        offset += os.write(fd, encoded[offset:])
    os.fsync(fd)


def _process_start_ticks(pid: int | None = None) -> int:
    """Linux ``/proc/<pid>/stat`` field 22, paired with PID against reuse."""
    target = os.getpid() if pid is None else int(pid)
    try:
        text = Path(f"/proc/{target}/stat").read_text(encoding="ascii")
        tail = text[text.rfind(")") + 2:].split()
        value = int(tail[19])  # tail starts at field 3; index 19 is field 22
    except (OSError, ValueError, IndexError) as exc:
        raise SandboxError(f"cannot read process start time for pid {target}: {exc}") from exc
    if value <= 0:
        raise SandboxError(f"process start time for pid {target} is invalid: {value}")
    return value


def launch(policy: SandboxPolicy, receipt_path: Path, argv: Sequence[str]) -> None:
    """Install every control, emit the activation receipt, then replace self."""
    if os.geteuid() == 0:
        raise SandboxError("candidate sandbox refuses uid 0")
    receipt_fd = _open_receipt(receipt_path)
    try:
        cgroup = _join_owned_cgroup(policy)
        install_resource_limits(policy.limits)
        abi, rights = install_landlock(policy.writable_root)
        seccomp_sha256 = install_seccomp()
        receipt = {
            "schema": "epyc.autokernel.sandbox_receipt.v1",
            "sandbox_id": SANDBOX_ID,
            "pid": os.getpid(),
            "process_start_ticks": _process_start_ticks(),
            "euid": os.geteuid(),
            "landlock_abi": abi,
            "landlock_write_rights": rights,
            "seccomp_sha256": seccomp_sha256,
            "blocked_syscalls": sorted(_BLOCKED_SYSCALLS),
            "writable_root": policy.writable_root,
            "cgroup_path": str(cgroup),
            "resource_limits": policy.limits.to_dict(),
            "activated_at_unix_ns": time.time_ns(),
            "argv_sha256": hashlib.sha256(json.dumps(
                list(argv), separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
        }
        _write_receipt(receipt_fd, receipt)
    finally:
        os.close(receipt_fd)
    os.execvpe(argv[0], list(argv), dict(os.environ))


def cleanup_cgroup(policy: SandboxPolicy, pid: int, *, timeout_s: float = 5.0) -> dict:
    """Drain and remove the invocation leaf; return a verified teardown receipt."""
    path = policy.cgroup_path(pid)
    if not path.exists():
        raise SandboxError(f"expected owned cgroup does not exist: {path}")
    killed = False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            members = Path(path, "cgroup.procs").read_text(encoding="ascii").split()
        except OSError as exc:
            raise SandboxError(f"cannot read owned cgroup membership: {exc}") from exc
        if not members:
            break
        kill_path = Path(path, "cgroup.kill")
        if kill_path.exists():
            try:
                kill_path.write_text("1", encoding="ascii")
                killed = True
            except OSError as exc:
                raise SandboxError(f"cannot kill escaped cgroup descendants: {exc}") from exc
        time.sleep(0.02)
    else:
        raise SandboxError(f"owned cgroup {path} remained populated after teardown")
    try:
        path.rmdir()
    except OSError as exc:
        raise SandboxError(f"owned cgroup {path} could not be removed: {exc}") from exc
    return {"cgroup_path": str(path), "verified_empty": True,
            "removed": not path.exists(), "descendants_killed": killed}


def read_receipt(path: str | Path) -> dict:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SandboxError(f"sandbox activation receipt is unavailable: {exc}") from exc
    required = {
        "schema", "sandbox_id", "pid", "process_start_ticks", "euid", "landlock_abi",
        "landlock_write_rights", "seccomp_sha256", "blocked_syscalls",
        "writable_root", "cgroup_path", "resource_limits", "argv_sha256",
    }
    missing = sorted(required - set(document))
    if missing:
        raise SandboxError(f"sandbox receipt is missing {missing}")
    if document["schema"] != "epyc.autokernel.sandbox_receipt.v1" \
            or document["sandbox_id"] != SANDBOX_ID:
        raise SandboxError("sandbox receipt names an unknown schema or implementation")
    if document["euid"] == 0 or document["landlock_abi"] < 1:
        raise SandboxError("sandbox receipt does not attest non-root Landlock confinement")
    if isinstance(document["process_start_ticks"], bool) \
            or not isinstance(document["process_start_ticks"], int) \
            or document["process_start_ticks"] <= 0:
        raise SandboxError("sandbox receipt carries an invalid process-start identity")
    expected = set(_BLOCKED_SYSCALLS)
    if set(document["blocked_syscalls"]) != expected:
        raise SandboxError("sandbox receipt's syscall surface does not match this evaluator")
    return document


def verify_receipt(document: Mapping[str, Any], *, policy: SandboxPolicy,
                   pid: int, argv: Sequence[str]) -> None:
    """Bind an activation receipt to the exact evaluator-owned invocation."""
    if not isinstance(policy, SandboxPolicy):
        raise TypeError("policy must be a SandboxPolicy")
    expected_argv_sha = hashlib.sha256(json.dumps(
        [str(item) for item in argv], separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    expectations = {
        "pid": pid,
        "writable_root": policy.writable_root,
        "cgroup_path": str(policy.cgroup_path(pid)),
        "resource_limits": policy.limits.to_dict(),
        "argv_sha256": expected_argv_sha,
        "seccomp_sha256": _seccomp_policy_sha256(),
    }
    for field, expected in expectations.items():
        if document.get(field) != expected:
            raise SandboxError(
                f"sandbox receipt {field} does not match this invocation: "
                f"expected {expected!r}, got {document.get(field)!r}")
    abi = document.get("landlock_abi")
    if document.get("landlock_write_rights") != _landlock_write_rights(abi):
        raise SandboxError("sandbox receipt's Landlock rights do not match its ABI")


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command after -- is required")
    try:
        policy, receipt = _decode_policy(args.policy)
        launch(policy, receipt, command)
    except (SandboxError, OSError, ValueError) as exc:
        print(f"autokernel sandbox refused: {exc}", file=sys.stderr)
        return 125
    return 127  # exec never returns


if __name__ == "__main__":
    raise SystemExit(_main())
