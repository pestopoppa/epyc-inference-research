#!/usr/bin/env python3
"""Fail-closed candidate-process confinement for AutoKernel.

This module deliberately uses kernel facilities that are available to an
unprivileged process.  It does not call a container runtime and it does not
pretend that a path check is a sandbox:

* Landlock handles every filesystem write right and grants it only beneath one
  invocation-owned directory; the controller profile additionally handles
  read/execute and grants only exact declared runtime inputs;
* seccomp denies signalling, ptrace/process-memory writes, mount / namespace
  changes, kernel-module operations and BPF.  The default profile denies all
  networking; the controller profile admits outbound INET clients (including
  client-side ``bind``), denies listen/accept and AF_UNIX creation, and inherits
  one preconnected broker;
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
import socket
import stat
import sys
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Mapping, Sequence


SANDBOX_ID = "autokernel.execution.sandbox/landlock-seccomp-cgroup-v2"
RECEIPT_SCHEMA = "epyc.autokernel.sandbox_receipt.v2"
DEFAULT_PROFILE = "candidate_default_v1"
CONTROLLER_PROFILE = "controller_outbound_client_v1"
EVALUATOR_PROFILE = "candidate_evaluator_gpu_v1"
NETWORK_DENY_ALL = "deny_all"
NETWORK_OUTBOUND_CLIENT = "outbound_client"
BROKER_FD_ENV = "EPYC_AUTOKERNEL_BROKER_FD"
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

_LANDLOCK_READ_EXECUTE = (
    _LANDLOCK_ACCESS_FS_EXECUTE
    | _LANDLOCK_ACCESS_FS_READ_FILE
    | _LANDLOCK_ACCESS_FS_READ_DIR
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
    "process_vm_readv": 310,
    "setns": 308,
    "sendmmsg": 307,
    "bpf": 321,
    "userfaultfd": 323,
    "pidfd_send_signal": 424,
    "io_uring_setup": 425,
    "io_uring_enter": 426,
    "io_uring_register": 427,
    "pidfd_getfd": 438,
    "process_madvise": 440,
    "socket": 41,
    "connect": 42,
    "accept": 43,
    "sendto": 44,
    "sendmsg": 46,
    "bind": 49,
    "listen": 50,
}

_CONTROLLER_BLOCKED_SYSCALLS: Mapping[str, int] = {
    name: number for name, number in _BLOCKED_SYSCALLS.items()
    if name not in {"socket", "connect", "sendto", "sendmsg", "sendmmsg", "bind"}
}
# Unnamed socketpair IPC has no filesystem or external peer and is required by
# Tokio's signal driver in the pinned Codex CLI.  New AF_UNIX sockets remain
# denied by the family filter below.  The pinned client also performs
# client-side bind before outbound traffic; listen/accept remain syscall-denied,
# so a bound stream cannot become a server.

_SECCOMP_DATA_NR_OFFSET = 0
_SECCOMP_DATA_ARCH_OFFSET = 4
_SECCOMP_DATA_ARG0_OFFSET = 16


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


def install_landlock(
    writable_root: str, writable_device_paths: Sequence[str] = (), *,
    restrict_reads: bool = False, readable_roots: Sequence[str] = (),
    readable_files: Sequence[str] = (),
    executable_files: Sequence[str] = (),
) -> tuple[int, int]:
    """Install write confinement and optional default-deny read/exec policy."""
    root = Path(writable_root).resolve(strict=True)
    if not root.is_dir():
        raise SandboxError(f"sandbox writable root is not a directory: {root}")
    abi = landlock_abi()
    rights = _landlock_write_rights(abi)
    if restrict_reads:
        rights |= _LANDLOCK_READ_EXECUTE
    attr = _LandlockRulesetAttr(rights)
    try:
        ruleset_fd = _syscall(
            _SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    except OSError as exc:
        raise SandboxError(f"Landlock ruleset creation failed: {exc}") from exc
    path_fds: list[int] = []
    try:
        root_fd = os.open(root, os.O_PATH | os.O_CLOEXEC)
        path_fds.append(root_fd)
        path_attr = _LandlockPathBeneathAttr(rights, root_fd, 0)
        _syscall(_SYS_LANDLOCK_ADD_RULE, ruleset_fd, _LANDLOCK_RULE_PATH_BENEATH,
                 ctypes.byref(path_attr), 0)
        for raw_path in writable_device_paths:
            device_fd = os.open(raw_path, os.O_PATH | os.O_CLOEXEC)
            path_fds.append(device_fd)
            # ROCm needs O_RDWR plus ioctl on these character devices.  It does
            # not need create, remove, rename, or truncate authority there.
            device_access = _LANDLOCK_ACCESS_FS_WRITE_FILE
            if restrict_reads:
                device_access |= _LANDLOCK_ACCESS_FS_READ_FILE
            device_attr = _LandlockPathBeneathAttr(device_access, device_fd, 0)
            _syscall(_SYS_LANDLOCK_ADD_RULE, ruleset_fd,
                     _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(device_attr), 0)
        if restrict_reads:
            for raw_path in readable_roots:
                path_fd = os.open(raw_path, os.O_PATH | os.O_CLOEXEC)
                path_fds.append(path_fd)
                read_attr = _LandlockPathBeneathAttr(
                    _LANDLOCK_READ_EXECUTE, path_fd, 0)
                _syscall(
                    _SYS_LANDLOCK_ADD_RULE, ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(read_attr), 0)
            for raw_path in readable_files:
                path_fd = os.open(raw_path, os.O_PATH | os.O_CLOEXEC)
                path_fds.append(path_fd)
                read_attr = _LandlockPathBeneathAttr(
                    _LANDLOCK_ACCESS_FS_READ_FILE, path_fd, 0)
                _syscall(
                    _SYS_LANDLOCK_ADD_RULE, ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(read_attr), 0)
            for raw_path in executable_files:
                path_fd = os.open(raw_path, os.O_PATH | os.O_CLOEXEC)
                path_fds.append(path_fd)
                execute_attr = _LandlockPathBeneathAttr(
                    _LANDLOCK_ACCESS_FS_READ_FILE | _LANDLOCK_ACCESS_FS_EXECUTE,
                    path_fd, 0)
                _syscall(
                    _SYS_LANDLOCK_ADD_RULE, ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(execute_attr), 0)
        _prctl(_PR_SET_NO_NEW_PRIVS, 1)
        _syscall(_SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    except OSError as exc:
        raise SandboxError(f"Landlock activation failed: {exc}") from exc
    finally:
        for path_fd in path_fds:
            os.close(path_fd)
        os.close(ruleset_fd)
    return abi, rights


def _statement(code: int, k: int) -> _SockFilter:
    return _SockFilter(code, 0, 0, k)


def _jump(code: int, k: int, jt: int, jf: int) -> _SockFilter:
    return _SockFilter(code, jt, jf, k)


def _network_policy(profile: str) -> tuple[Mapping[str, int], bool, str]:
    if profile in {DEFAULT_PROFILE, EVALUATOR_PROFILE}:
        return _BLOCKED_SYSCALLS, False, NETWORK_DENY_ALL
    if profile == CONTROLLER_PROFILE:
        return _CONTROLLER_BLOCKED_SYSCALLS, True, NETWORK_OUTBOUND_CLIENT
    raise SandboxError(f"unknown sandbox profile: {profile!r}")


def install_seccomp(profile: str = DEFAULT_PROFILE) -> str:
    """Install the selected deny policy and return its content identity."""
    if os.uname().machine != "x86_64":
        raise SandboxError(
            f"seccomp policy is compiled for x86_64, not {os.uname().machine!r}")
    blocked, deny_unix_socket, _network = _network_policy(profile)
    filters = [
        _statement(_BPF_LD_W_ABS, _SECCOMP_DATA_ARCH_OFFSET),
        _jump(_BPF_JMP_JEQ_K, _AUDIT_ARCH_X86_64, 1, 0),
        _statement(_BPF_RET_K, _SECCOMP_RET_KILL_PROCESS),
        _statement(_BPF_LD_W_ABS, _SECCOMP_DATA_NR_OFFSET),
    ]
    if deny_unix_socket:
        filters.extend((
            # Permit only INET/INET6 socket creation.  The broker's exact
            # AF_UNIX stream is connected by the trusted wrapper before this
            # filter and passed as one inherited descriptor.
            _jump(_BPF_JMP_JEQ_K, _BLOCKED_SYSCALLS["socket"], 0, 5),
            _statement(_BPF_LD_W_ABS, _SECCOMP_DATA_ARG0_OFFSET),
            _jump(_BPF_JMP_JEQ_K, socket.AF_INET, 2, 0),
            _jump(_BPF_JMP_JEQ_K, socket.AF_INET6, 1, 0),
            _statement(_BPF_RET_K, _SECCOMP_RET_ERRNO | errno.EPERM),
            _statement(_BPF_LD_W_ABS, _SECCOMP_DATA_NR_OFFSET),
        ))
    for number in sorted(set(blocked.values())):
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
    return _seccomp_policy_sha256(profile)


def _seccomp_policy_sha256(profile: str = DEFAULT_PROFILE) -> str:
    blocked, deny_unix_socket, network = _network_policy(profile)
    return hashlib.sha256(json.dumps(
        {
            "blocked_syscalls": sorted(blocked.items()),
            "deny_unix_socket_creation": deny_unix_socket,
            "network_profile": network,
        }, sort_keys=True, separators=(",", ":")
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
    writable_device_paths: tuple[str, ...] = ()
    profile: str = DEFAULT_PROFILE
    readable_roots: tuple[str, ...] = ()
    readable_files: tuple[str, ...] = ()
    executable_files: tuple[str, ...] = ()
    broker_socket_path: str | None = None
    broker_peer_pid: int | None = None
    broker_peer_start_ticks: int | None = None

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
        allowed_devices = {"/dev/kfd", "/dev/dri/renderD128", "/dev/null"}
        normalized_devices = tuple(str(Path(path).resolve(strict=True))
                                   for path in self.writable_device_paths)
        if len(normalized_devices) != len(set(normalized_devices)):
            raise SandboxError("writable_device_paths must be unique")
        for path in normalized_devices:
            if path not in allowed_devices or not stat.S_ISCHR(os.stat(path).st_mode):
                raise SandboxError(
                    f"writable device is not an admitted ROCm character device: {path}")
        _blocked, _deny_unix, network_profile = _network_policy(self.profile)
        normalized_roots = tuple(
            str(Path(path).resolve(strict=True)) for path in self.readable_roots)
        normalized_files = tuple(
            str(Path(path).resolve(strict=True)) for path in self.readable_files)
        normalized_executables = tuple(
            str(Path(path).resolve(strict=True)) for path in self.executable_files)
        if len(normalized_roots) != len(set(normalized_roots)):
            raise SandboxError("readable_roots must be unique")
        if len(normalized_files) != len(set(normalized_files)):
            raise SandboxError("readable_files must be unique")
        if len(normalized_executables) != len(set(normalized_executables)):
            raise SandboxError("executable_files must be unique")
        for path in normalized_roots:
            candidate = Path(path)
            if not candidate.is_dir():
                raise SandboxError(f"readable root is not a directory: {path}")
            if candidate == Path("/") or candidate.is_relative_to(Path("/dev")):
                raise SandboxError(
                    f"readable root would expose host devices: {path}")
        for path in normalized_files:
            mode = os.stat(path).st_mode
            evaluator_random = (
                self.profile == EVALUATOR_PROFILE
                and path == "/dev/urandom" and stat.S_ISCHR(mode))
            if not stat.S_ISREG(mode) and not evaluator_random:
                raise SandboxError(f"readable file is not regular: {path}")
            if path.startswith("/dev/") and not evaluator_random:
                raise SandboxError(
                    f"readable device is not the evaluator random source: {path}")
        for path in normalized_executables:
            mode = os.stat(path).st_mode
            if not stat.S_ISREG(mode) or not os.access(path, os.X_OK):
                raise SandboxError(
                    f"executable file is not an executable regular file: {path}")
        broker_path: str | None = None
        if self.broker_socket_path is not None:
            broker = Path(self.broker_socket_path).resolve(strict=True)
            if not stat.S_ISSOCK(os.stat(broker).st_mode):
                raise SandboxError(f"broker path is not a Unix socket: {broker}")
            try:
                broker.relative_to(root)
            except ValueError:
                pass
            else:
                raise SandboxError(
                    "broker socket must be outside controller-writable state")
            broker_path = str(broker)
        if self.profile == DEFAULT_PROFILE:
            if (normalized_roots or normalized_files or normalized_executables
                    or broker_path is not None):
                raise SandboxError(
                    "read allowlisting and broker sockets require controller profile")
        elif self.profile == CONTROLLER_PROFILE:
            if not normalized_roots and not normalized_files \
                    and not normalized_executables:
                raise SandboxError(
                    "controller profile requires exact readable roots or files")
            if broker_path is None:
                raise SandboxError("controller profile requires an exact broker socket")
            if (isinstance(self.broker_peer_pid, bool)
                    or not isinstance(self.broker_peer_pid, int)
                    or self.broker_peer_pid <= 1):
                raise SandboxError("controller profile requires broker_peer_pid")
            if (isinstance(self.broker_peer_start_ticks, bool)
                    or not isinstance(self.broker_peer_start_ticks, int)
                    or self.broker_peer_start_ticks <= 0):
                raise SandboxError(
                    "controller profile requires broker_peer_start_ticks")
            if any(path != "/dev/null" for path in normalized_devices):
                raise SandboxError(
                    "controller profile can admit only the null device")
            if network_profile != NETWORK_OUTBOUND_CLIENT:
                raise SandboxError("controller profile must use outbound-client network")
        elif self.profile == EVALUATOR_PROFILE:
            if not normalized_roots and not normalized_files:
                raise SandboxError(
                    "evaluator profile requires exact readable roots or files")
            if broker_path is not None:
                raise SandboxError("evaluator profile cannot inherit a broker socket")
            if self.broker_peer_pid is not None \
                    or self.broker_peer_start_ticks is not None:
                raise SandboxError("evaluator profile cannot name a broker peer")
            if (len(normalized_devices) != 3 or set(normalized_devices)
                    != {"/dev/kfd", "/dev/dri/renderD128", "/dev/null"}):
                raise SandboxError(
                    "evaluator profile requires the exact MI210 pair and /dev/null")
            if network_profile != NETWORK_DENY_ALL:
                raise SandboxError("evaluator profile must deny all networking")
        else:
            raise SandboxError(f"unknown sandbox profile: {self.profile!r}")
        token = self.token or secrets.token_hex(8)
        if not token.isalnum() or len(token) > 64:
            raise ValueError("sandbox token must be 1..64 alphanumeric characters")
        object.__setattr__(self, "writable_root", str(root))
        object.__setattr__(self, "cgroup_root", str(cgroup))
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "writable_device_paths", normalized_devices)
        object.__setattr__(self, "readable_roots", normalized_roots)
        object.__setattr__(self, "readable_files", normalized_files)
        object.__setattr__(self, "executable_files", normalized_executables)
        object.__setattr__(self, "broker_socket_path", broker_path)
        # Constructor is the fail-closed availability check.  No process is
        # launched merely to discover that containment is impossible.
        landlock_abi()

    def cgroup_path(self, pid: int) -> Path:
        return Path(self.cgroup_root, f"autokernel-{pid}-{self.token}")

    @property
    def network_profile(self) -> str:
        return _network_policy(self.profile)[2]

    @property
    def restrict_reads(self) -> bool:
        return self.profile in {CONTROLLER_PROFILE, EVALUATOR_PROFILE}

    def policy_document(self) -> dict[str, Any]:
        blocked, deny_unix, network = _network_policy(self.profile)
        return {
            "sandbox_id": SANDBOX_ID,
            "profile": self.profile,
            "writable_root": self.writable_root,
            "cgroup_root": self.cgroup_root,
            "writable_device_paths": list(self.writable_device_paths),
            "readable_roots": list(self.readable_roots),
            "readable_files": list(self.readable_files),
            "executable_files": list(self.executable_files),
            "broker_socket_path": self.broker_socket_path,
            "broker_peer_pid": self.broker_peer_pid,
            "broker_peer_start_ticks": self.broker_peer_start_ticks,
            "read_allowlist_enforced": self.restrict_reads,
            "network_profile": network,
            "blocked_syscalls": sorted(blocked),
            "deny_unix_socket_creation": deny_unix,
            "resource_limits": self.limits.to_dict(),
        }

    @property
    def policy_sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.policy_document(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

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
            "writable_device_paths": list(self.writable_device_paths),
            "profile": self.profile,
            "readable_roots": list(self.readable_roots),
            "readable_files": list(self.readable_files),
            "executable_files": list(self.executable_files),
            "broker_socket_path": self.broker_socket_path,
            "broker_peer_pid": self.broker_peer_pid,
            "broker_peer_start_ticks": self.broker_peer_start_ticks,
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
            limits=limits, token=raw["token"],
            writable_device_paths=tuple(raw.get("writable_device_paths", ())),
            profile=raw.get("profile", DEFAULT_PROFILE),
            readable_roots=tuple(raw.get("readable_roots", ())),
            readable_files=tuple(raw.get("readable_files", ())),
            executable_files=tuple(raw.get("executable_files", ())),
            broker_socket_path=raw.get("broker_socket_path"),
            broker_peer_pid=raw.get("broker_peer_pid"),
            broker_peer_start_ticks=raw.get("broker_peer_start_ticks"))
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


def _connect_broker(policy: SandboxPolicy) -> tuple[int | None, dict[str, int] | None]:
    """Connect the one admitted UDS before AF_UNIX creation is denied."""
    if policy.broker_socket_path is None:
        return None, None
    broker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        broker.connect(policy.broker_socket_path)
        credentials = broker.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        peer_pid = int.from_bytes(credentials[0:4], sys.byteorder, signed=True)
        peer_uid = int.from_bytes(credentials[4:8], sys.byteorder, signed=True)
        peer_gid = int.from_bytes(credentials[8:12], sys.byteorder, signed=True)
        peer_start_ticks = _process_start_ticks(peer_pid)
        if peer_pid != policy.broker_peer_pid \
                or peer_start_ticks != policy.broker_peer_start_ticks:
            raise SandboxError(
                "evaluation broker peer identity changed before activation")
        descriptor = broker.detach()
        os.set_inheritable(descriptor, True)
        return descriptor, {
            "pid": peer_pid, "start_ticks": peer_start_ticks,
            "uid": peer_uid, "gid": peer_gid,
        }
    except Exception as exc:
        broker.close()
        if isinstance(exc, SandboxError):
            raise
        raise SandboxError(
            f"cannot connect exact evaluation broker {policy.broker_socket_path}: {exc}") \
            from exc


def launch(policy: SandboxPolicy, receipt_path: Path, argv: Sequence[str]) -> None:
    """Install every control, emit the activation receipt, then replace self."""
    if os.geteuid() == 0:
        raise SandboxError("candidate sandbox refuses uid 0")
    receipt_fd = _open_receipt(receipt_path)
    broker_fd: int | None = None
    broker_peer: dict[str, int] | None = None
    try:
        cgroup = _join_owned_cgroup(policy)
        install_resource_limits(policy.limits)
        process_start_ticks = _process_start_ticks()
        broker_fd, broker_peer = _connect_broker(policy)
        abi, rights = install_landlock(
            policy.writable_root, policy.writable_device_paths,
            restrict_reads=policy.restrict_reads,
            readable_roots=policy.readable_roots,
            readable_files=policy.readable_files,
            executable_files=policy.executable_files)
        seccomp_sha256 = install_seccomp(policy.profile)
        blocked, deny_unix, network_profile = _network_policy(policy.profile)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "sandbox_id": SANDBOX_ID,
            "pid": os.getpid(),
            "process_start_ticks": process_start_ticks,
            "euid": os.geteuid(),
            "landlock_abi": abi,
            "landlock_write_rights": rights,
            "landlock_handled_rights": rights,
            "read_allowlist_enforced": policy.restrict_reads,
            "readable_roots": list(policy.readable_roots),
            "readable_files": list(policy.readable_files),
            "executable_files": list(policy.executable_files),
            "seccomp_sha256": seccomp_sha256,
            "blocked_syscalls": sorted(blocked),
            "profile": policy.profile,
            "network_profile": network_profile,
            "outbound_socket_families": (
                ["AF_INET", "AF_INET6"]
                if network_profile == NETWORK_OUTBOUND_CLIENT else []),
            "server_socket_operations_denied": [
                name for name in ("listen", "accept", "accept4")
                if name in blocked],
            "unix_socket_creation_denied": deny_unix,
            "broker_socket_path": policy.broker_socket_path,
            "broker_fd_inherited": broker_fd is not None,
            "broker_peer": broker_peer,
            "writable_root": policy.writable_root,
            "writable_device_paths": list(policy.writable_device_paths),
            "cgroup_path": str(cgroup),
            "resource_limits": policy.limits.to_dict(),
            "policy_sha256": policy.policy_sha256,
            "activated_at_unix_ns": time.time_ns(),
            "argv_sha256": hashlib.sha256(json.dumps(
                list(argv), separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
        }
        _write_receipt(receipt_fd, receipt)
    except BaseException:
        if broker_fd is not None:
            os.close(broker_fd)
        raise
    finally:
        os.close(receipt_fd)
    environment = dict(os.environ)
    if broker_fd is not None:
        environment[BROKER_FD_ENV] = str(broker_fd)
    try:
        os.execvpe(argv[0], list(argv), environment)
    finally:
        if broker_fd is not None:
            os.close(broker_fd)


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
        "writable_device_paths",
        "landlock_handled_rights", "read_allowlist_enforced",
        "readable_roots", "readable_files", "executable_files",
        "profile", "network_profile",
        "outbound_socket_families", "server_socket_operations_denied",
        "unix_socket_creation_denied", "broker_socket_path",
        "broker_fd_inherited", "policy_sha256",
        "broker_peer",
    }
    missing = sorted(required - set(document))
    if missing:
        raise SandboxError(f"sandbox receipt is missing {missing}")
    if document["schema"] != RECEIPT_SCHEMA \
            or document["sandbox_id"] != SANDBOX_ID:
        raise SandboxError("sandbox receipt names an unknown schema or implementation")
    if document["euid"] == 0 or document["landlock_abi"] < 1:
        raise SandboxError("sandbox receipt does not attest non-root Landlock confinement")
    if isinstance(document["process_start_ticks"], bool) \
            or not isinstance(document["process_start_ticks"], int) \
            or document["process_start_ticks"] <= 0:
        raise SandboxError("sandbox receipt carries an invalid process-start identity")
    try:
        expected = set(_network_policy(document["profile"])[0])
    except SandboxError as exc:
        raise SandboxError("sandbox receipt names an unknown profile") from exc
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
        "writable_device_paths": list(policy.writable_device_paths),
        "profile": policy.profile,
        "read_allowlist_enforced": policy.restrict_reads,
        "readable_roots": list(policy.readable_roots),
        "readable_files": list(policy.readable_files),
        "executable_files": list(policy.executable_files),
        "network_profile": policy.network_profile,
        "broker_socket_path": policy.broker_socket_path,
        "broker_fd_inherited": policy.broker_socket_path is not None,
        "policy_sha256": policy.policy_sha256,
        "cgroup_path": str(policy.cgroup_path(pid)),
        "resource_limits": policy.limits.to_dict(),
        "argv_sha256": expected_argv_sha,
        "seccomp_sha256": _seccomp_policy_sha256(policy.profile),
    }
    for field, expected in expectations.items():
        if document.get(field) != expected:
            raise SandboxError(
                f"sandbox receipt {field} does not match this invocation: "
                f"expected {expected!r}, got {document.get(field)!r}")
    abi = document.get("landlock_abi")
    expected_rights = _landlock_write_rights(abi)
    if policy.restrict_reads:
        expected_rights |= _LANDLOCK_READ_EXECUTE
    if document.get("landlock_write_rights") != expected_rights \
            or document.get("landlock_handled_rights") != expected_rights:
        raise SandboxError("sandbox receipt's Landlock rights do not match its ABI")
    blocked, deny_unix, network = _network_policy(policy.profile)
    if set(document.get("blocked_syscalls", ())) != set(blocked):
        raise SandboxError("sandbox receipt's blocked syscalls do not match policy")
    if document.get("unix_socket_creation_denied") is not deny_unix:
        raise SandboxError("sandbox receipt's Unix-socket policy does not match")
    expected_families = (["AF_INET", "AF_INET6"]
                         if network == NETWORK_OUTBOUND_CLIENT else [])
    if document.get("outbound_socket_families") != expected_families:
        raise SandboxError("sandbox receipt's outbound families do not match")
    expected_server_denials = [
        name for name in ("listen", "accept", "accept4")
        if name in blocked]
    if document.get("server_socket_operations_denied") != expected_server_denials:
        raise SandboxError("sandbox receipt's server denials do not match")
    broker_peer = document.get("broker_peer")
    if policy.broker_socket_path is None:
        if broker_peer is not None:
            raise SandboxError("default sandbox receipt unexpectedly names a broker")
    elif (not isinstance(broker_peer, Mapping)
          or broker_peer.get("pid") != policy.broker_peer_pid
          or broker_peer.get("start_ticks") != policy.broker_peer_start_ticks
          or isinstance(broker_peer.get("uid"), bool)
          or not isinstance(broker_peer.get("uid"), int)
          or isinstance(broker_peer.get("gid"), bool)
          or not isinstance(broker_peer.get("gid"), int)):
        raise SandboxError("sandbox receipt's broker peer identity does not match")


__all__ = [
    "BROKER_FD_ENV", "CGROUP_ROOT_ENV", "CONTROLLER_PROFILE",
    "EVALUATOR_PROFILE",
    "DEFAULT_PROFILE", "HOST_CGROUP_ROOT", "NETWORK_DENY_ALL",
    "NETWORK_OUTBOUND_CLIENT", "RECEIPT_SCHEMA", "ResourceLimits",
    "SANDBOX_ID", "SandboxError", "SandboxPolicy", "cleanup_cgroup",
    "default_cgroup_root", "install_landlock", "install_resource_limits",
    "install_seccomp", "landlock_abi", "launch", "read_receipt",
    "verify_receipt",
]


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
