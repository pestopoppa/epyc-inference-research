#!/usr/bin/env python3
"""Controller-only Arena sandbox integration over ``execution.sandbox`` v2.

This module is an adapter, not a second sandbox implementation.  It builds the
``CONTROLLER_PROFILE`` from exact runtime inputs, supplies the prefix accepted
by :func:`arena_adapter.launch`, captures the exec-stable controller PID, and
strictly verifies activation plus cgroup teardown.  It never constructs a GPU
candidate profile and never grants a controller a device or claim credential.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..execution import sandbox


SCHEMA = "epyc.autokernel.arena_controller_sandbox.v1"
TEARDOWN_SCHEMA = "epyc.autokernel.arena_controller_sandbox_teardown.v1"
CONTROLLER_ENVIRONMENT: Mapping[str, str] = MappingProxyType({
    "PYTHONDONTWRITEBYTECODE": "1",
    # CPython otherwise falls back to opening /dev/urandom during early hash
    # initialization after Landlock correctly denies the entire /dev tree.
    "PYTHONHASHSEED": "0",
})
_BROAD_ROOTS = {
    Path("/"), Path("/mnt"), Path("/mnt/raid0"), Path("/mnt/raid0/llm"),
    Path("/workspace"), Path("/usr"), Path("/usr/local"), Path("/opt"),
}
_PRODUCTION_ROOTS = tuple(Path(value) for value in (
    "/mnt/raid0/llm/llama.cpp", "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp",
))
_STATE_ROOTS = tuple(Path(value) for value in (
    "/mnt/raid0/llm/autokernel/campaigns",
    "/mnt/raid0/llm/autokernel/probes",
    "/mnt/raid0/llm/autokernel/surface",
))
_NETWORK_CONFIG_PATHS = tuple(Path(value) for value in (
    "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf",
    "/etc/ssl/openssl.cnf",
))


class ControllerSandboxError(RuntimeError):
    """An Arena controller isolation input or receipt is unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _exact_path(raw: str | Path, *, directory: bool) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise ControllerSandboxError(f"runtime path must be absolute: {path}")
    if path.is_symlink():
        raise ControllerSandboxError(f"runtime path must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ControllerSandboxError(f"runtime path is unavailable: {path}: {exc}") from exc
    if resolved != path:
        raise ControllerSandboxError(
            f"runtime path traverses a symlink; pass the exact real path: {path}")
    if directory and not path.is_dir():
        raise ControllerSandboxError(f"runtime root is not a directory: {path}")
    if not directory and (not path.is_file() or not stat.S_ISREG(path.stat().st_mode)):
        raise ControllerSandboxError(f"runtime file is not regular: {path}")
    return path


def _exact_socket(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink():
        raise ControllerSandboxError(
            f"broker socket must be an exact absolute non-symlink path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ControllerSandboxError(f"broker socket is unavailable: {path}: {exc}") \
            from exc
    if resolved != path or not stat.S_ISSOCK(path.stat().st_mode):
        raise ControllerSandboxError(f"broker path is not an exact Unix socket: {path}")
    return path


def _admit_path(path: Path, *, workspace: Path,
                forbidden_roots: Sequence[Path]) -> None:
    if path in _BROAD_ROOTS:
        raise ControllerSandboxError(f"runtime root is too broad: {path}")
    if path == Path("/dev") or path.is_relative_to(Path("/dev")):
        raise ControllerSandboxError(f"controller runtime cannot expose devices: {path}")
    if _overlaps(path, workspace):
        raise ControllerSandboxError(
            "controller workspace is implicit writable state, not a runtime allowlist row")
    for forbidden in (*_PRODUCTION_ROOTS, *_STATE_ROOTS, *forbidden_roots):
        if _overlaps(path, forbidden):
            raise ControllerSandboxError(
                f"controller runtime overlaps production/campaign/evidence state: {path}")


def _unique_exact(paths: Iterable[str | Path], *, directory: bool,
                  workspace: Path, forbidden_roots: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = _exact_path(raw, directory=directory)
        _admit_path(path, workspace=workspace, forbidden_roots=forbidden_roots)
        if path in seen:
            raise ControllerSandboxError(f"duplicate runtime allowlist path: {path}")
        seen.add(path)
        result.append(path)
    return tuple(result)


def _process_start_ticks(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        value = int(text[text.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError) as exc:
        raise ControllerSandboxError(
            f"cannot capture controller start time for pid {pid}: {exc}") from exc
    if value <= 0:
        raise ControllerSandboxError("controller process start time is invalid")
    return value


def _shared_libraries(executable: Path) -> tuple[Path, ...]:
    if executable.read_bytes()[:4] != b"\x7fELF":
        return ()
    result = subprocess.run(
        ("ldd", str(executable)), capture_output=True, text=True,
        check=False, timeout=15)
    if result.returncode != 0:
        raise ControllerSandboxError(
            f"cannot resolve shared-library closure for {executable}: {result.stderr.strip()}")
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        words = line.replace("=>", " ").split()
        raw = next((word for word in words if word.startswith("/")), None)
        if raw is not None:
            path = Path(raw).resolve(strict=True)
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def _elf_interpreter(executable: Path) -> Path | None:
    if executable.read_bytes()[:4] != b"\x7fELF":
        return None
    result = subprocess.run(
        ("readelf", "--program-headers", "--wide", str(executable)),
        capture_output=True, text=True, check=False, timeout=15)
    if result.returncode != 0:
        raise ControllerSandboxError(
            f"cannot resolve ELF interpreter for {executable}: "
            f"{result.stderr.strip()}")
    match = re.search(r"Requesting program interpreter:\s*([^\]]+)\]", result.stdout)
    if match is None:
        return None
    return Path(match.group(1)).resolve(strict=True)


def _shebang_executables(script: Path) -> tuple[Path, ...]:
    with script.open("rb") as handle:
        first = handle.readline(4096)
    if not first.startswith(b"#!"):
        return ()
    try:
        words = first[2:].decode("utf-8").strip().split()
    except UnicodeDecodeError as exc:
        raise ControllerSandboxError(f"invalid shebang in {script}") from exc
    if not words or not Path(words[0]).is_absolute():
        raise ControllerSandboxError(
            f"controller CLI shebang must name an absolute interpreter: {script}")
    # Shebang text may name a system-maintained symlink (not caller authority),
    # so bind the current exact target identity rather than rejecting the CLI.
    interpreter = _exact_path(
        Path(words[0]).resolve(strict=True), directory=False)
    result = [interpreter]
    if interpreter == Path("/usr/bin/env"):
        if len(words) != 2 or "/" in words[1]:
            raise ControllerSandboxError(
                f"controller CLI env shebang is not one exact program: {script}")
        resolved = shutil.which(words[1])
        if resolved is None:
            raise ControllerSandboxError(
                f"controller CLI shebang program is unavailable: {words[1]}")
        result.append(_exact_path(Path(resolved).resolve(strict=True), directory=False))
    return tuple(result)


@dataclass(frozen=True)
class RuntimeAllowlist:
    readable_roots: tuple[str, ...]
    readable_files: tuple[str, ...]
    executable_files: tuple[str, ...]
    identities: Mapping[str, str]

    @property
    def sha256(self) -> str:
        return _digest({
            "readable_roots": list(self.readable_roots),
            "readable_files": list(self.readable_files),
            "executable_files": list(self.executable_files),
            "identities": dict(self.identities),
        })


def discover_runtime_allowlist(
    *, workspace: str | Path, python_executable: str | Path,
    controller_source_roots: Sequence[str | Path],
    controller_entrypoint: str | Path,
    repository_module_roots: Sequence[str | Path],
    codex_cli: str | Path, node_executable: str | Path,
    codex_auth: str | Path, ca_files: Sequence[str | Path],
    additional_cli_executables: Sequence[str | Path] = (),
    additional_cli_read_files: Sequence[str | Path] = (),
    additional_cli_package_roots: Sequence[str | Path] = (),
    forbidden_roots: Sequence[str | Path] = (),
) -> RuntimeAllowlist:
    """Discover the exact controller runtime; never infer a parent-directory grant.

    Callers must pass real, non-symlink paths.  Python reports its own versioned
    stdlib/site roots.  Codex contributes only its package root, exact Node
    runtime, auth file and CA file(s).  Another declared CLI contributes only
    its exact executable, library closure, explicitly named read files, and
    explicitly named package roots.  No HOME or arbitrary environment prefix
    is admitted.
    """
    work = _exact_path(workspace, directory=True)
    forbidden = tuple(_exact_path(path, directory=True) for path in forbidden_roots)
    python = _exact_path(python_executable, directory=False)
    codex = _exact_path(codex_cli, directory=False)
    node = _exact_path(node_executable, directory=False)
    additional_clis = _unique_exact(
        additional_cli_executables, directory=False, workspace=work,
        forbidden_roots=forbidden)
    entrypoint = _exact_path(controller_entrypoint, directory=False)
    auth = _exact_path(codex_auth, directory=False)
    sources = _unique_exact(
        controller_source_roots, directory=True, workspace=work,
        forbidden_roots=forbidden)
    modules = _unique_exact(
        repository_module_roots, directory=True, workspace=work,
        forbidden_roots=forbidden)
    if not any(entrypoint.is_relative_to(root) for root in (*sources, *modules)):
        raise ControllerSandboxError(
            "controller entrypoint is outside the licensed/module source roots")
    probe = subprocess.run(
        (str(python), "-I", "-c",
         "import json,sysconfig; print(json.dumps({k:sysconfig.get_path(k) "
         "for k in ('stdlib','platstdlib','purelib','platlib')}))"),
        capture_output=True, text=True, check=False, timeout=15)
    if probe.returncode != 0:
        raise ControllerSandboxError(
            f"pinned Python runtime discovery failed: {probe.stderr.strip()}")
    try:
        python_roots = tuple(dict.fromkeys(json.loads(probe.stdout).values()))
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ControllerSandboxError("pinned Python emitted invalid runtime roots") from exc
    codex_package = codex.parent.parent
    additional_packages = _unique_exact(
        additional_cli_package_roots, directory=True, workspace=work,
        forbidden_roots=forbidden)
    roots = _unique_exact(
        (*sources, *modules, *python_roots, codex_package, *additional_packages),
        directory=True,
        workspace=work, forbidden_roots=forbidden)
    ca_paths = _unique_exact(
        ca_files, directory=False, workspace=work, forbidden_roots=forbidden)
    network_config = _unique_exact(
        (path.resolve(strict=True) for path in _NETWORK_CONFIG_PATHS),
        directory=False, workspace=work, forbidden_roots=forbidden)
    declared_clis = (codex, *additional_clis)
    shebang_executables = tuple(dict.fromkeys(
        executable for cli in declared_clis
        for executable in _shebang_executables(cli)))
    all_runtime_executables = tuple(dict.fromkeys(
        (python, node, *declared_clis, *shebang_executables)))
    library_closure = tuple(dict.fromkeys(
        path for executable in all_runtime_executables
        for path in _shared_libraries(executable)))
    elf_interpreters = tuple(dict.fromkeys(
        path for executable in all_runtime_executables
        if (path := _elf_interpreter(executable)) is not None
    ))
    # ELF interpreters are separately executed by the kernel.  Keep them exact
    # rather than admitting either executable's sibling directory.
    executables = _unique_exact(
        (python, node, *declared_clis, *shebang_executables, *elf_interpreters),
        directory=False, workspace=work,
        forbidden_roots=forbidden)
    extra_read_files = _unique_exact(
        additional_cli_read_files, directory=False, workspace=work,
        forbidden_roots=forbidden)
    files_raw: list[str | Path] = [
        auth, *extra_read_files, *ca_paths, *network_config]
    loader_cache = Path("/etc/ld.so.cache")
    if loader_cache.is_file() and not loader_cache.is_symlink():
        files_raw.append(loader_cache)
    for discovered in library_closure:
        if discovered not in files_raw:
            files_raw.append(discovered)
    files = _unique_exact(
        files_raw, directory=False, workspace=work, forbidden_roots=forbidden)
    # A file already covered by a root is redundant authority, not harmless
    # discoverability.  Keep its identity but remove it from the Landlock rows.
    readable_files = tuple(
        path for path in files if not any(path.is_relative_to(root) for root in roots))
    executable_files = tuple(
        path for path in executables
        if not any(path.is_relative_to(root) for root in roots))
    if len(readable_files) != len(set(readable_files)):
        raise ControllerSandboxError("runtime file allowlist is not unique")
    identities = {
        str(path): _sha256_file(path)
        for path in dict.fromkeys((
            python, node, *declared_clis, entrypoint, auth, *extra_read_files,
            *ca_paths, *network_config,
            *shebang_executables, *readable_files, *executable_files))
    }
    return RuntimeAllowlist(
        readable_roots=tuple(map(str, roots)),
        readable_files=tuple(map(str, readable_files)),
        executable_files=tuple(map(str, executable_files)),
        identities=MappingProxyType(identities))


def copy_controller_workspace(source: str | Path,
                              destination: str | Path) -> dict[str, Any]:
    """Materialize one exact regular-file task copy; symlinks/devices refuse."""
    src = _exact_path(source, directory=True)
    dst = Path(destination)
    if not dst.is_absolute() or dst.exists() or dst.is_symlink():
        raise ControllerSandboxError(
            "controller workspace destination must be a new absolute path")
    if _overlaps(src, dst.resolve()):
        raise ControllerSandboxError(
            "controller workspace source and destination must not overlap")
    manifest: dict[str, str] = {}
    for path in sorted(src.rglob("*")):
        if path.is_symlink():
            raise ControllerSandboxError(f"controller workspace contains symlink: {path}")
        if path.is_file():
            if not stat.S_ISREG(path.stat().st_mode):
                raise ControllerSandboxError(
                    f"controller workspace contains non-regular file: {path}")
            manifest[str(path.relative_to(src))] = _sha256_file(path)
        elif not path.is_dir():
            raise ControllerSandboxError(
                f"controller workspace contains unsupported node: {path}")
    shutil.copytree(src, dst, symlinks=False)
    copied = {
        str(path.relative_to(dst)): _sha256_file(path)
        for path in sorted(dst.rglob("*")) if path.is_file()
    }
    if copied != manifest:
        raise ControllerSandboxError(
            "controller workspace copy does not match its source manifest")
    return {
        "schema": SCHEMA, "source": str(src), "workspace": str(dst),
        "files": manifest, "manifest_sha256": _digest(manifest),
    }


class ControllerSandboxInvocation:
    """One exact controller launch prefix and its strict completion verifier."""

    def __init__(self, *, policy: sandbox.SandboxPolicy, receipt_path: Path,
                 expected_argv: Sequence[str], runtime: RuntimeAllowlist):
        self.policy = policy
        self.receipt_path = receipt_path
        self.expected_argv = tuple(expected_argv)
        self.runtime = runtime
        sentinel = policy.wrap(("__controller__",), receipt_path=str(receipt_path))
        self.command_prefix = sentinel[:-1]
        self._lock = threading.Lock()
        self._pid: int | None = None
        self._start_ticks: int | None = None

    @property
    def environment_overrides(self) -> Mapping[str, str]:
        """Fixed startup controls required by the exact read allowlist."""
        return CONTROLLER_ENVIRONMENT

    def process_started(self, pid: int) -> None:
        """Capture PID identity; broker registration may safely run afterwards.

        The sandbox wrapper can connect before this Popen callback.  The broker
        must queue that accepted socket until this exact PID/start-time is
        registered; ancestry or uid-based admission is forbidden.
        """
        start_ticks = _process_start_ticks(pid)
        with self._lock:
            if self._pid is not None:
                raise ControllerSandboxError("controller PID was registered twice")
            self._pid, self._start_ticks = pid, start_ticks

    @property
    def pid(self) -> int | None:
        return self._pid

    def verify_and_teardown(self, teardown_receipt: str | Path) -> dict[str, Any]:
        target = Path(teardown_receipt).resolve()
        if _overlaps(target, Path(self.policy.writable_root)):
            raise ControllerSandboxError("teardown receipt must be evaluator-owned")
        if target.exists() or not target.parent.is_dir():
            raise ControllerSandboxError("teardown receipt target must be new")
        with self._lock:
            pid, start_ticks = self._pid, self._start_ticks
        if pid is None or start_ticks is None:
            raise ControllerSandboxError("controller PID was never captured")
        activation_error: BaseException | None = None
        activation: dict[str, Any] | None = None
        try:
            activation = sandbox.read_receipt(self.receipt_path)
            sandbox.verify_receipt(
                activation, policy=self.policy, pid=pid,
                argv=self.expected_argv)
            if activation.get("process_start_ticks") != start_ticks:
                raise ControllerSandboxError(
                    "activation receipt disagrees with captured PID start time")
        except BaseException as exc:
            activation_error = exc
        teardown = sandbox.cleanup_cgroup(self.policy, pid)
        if activation_error is not None:
            raise ControllerSandboxError(
                f"controller activation verification failed after cleanup: {activation_error}") \
                from activation_error
        assert activation is not None
        payload = {
            "schema": TEARDOWN_SCHEMA, "pid": pid,
            "process_start_ticks": start_ticks,
            "policy_sha256": self.policy.policy_sha256,
            "runtime_allowlist_sha256": self.runtime.sha256,
            "activation_receipt": str(self.receipt_path),
            "activation_receipt_sha256": _sha256_file(self.receipt_path),
            "teardown": teardown,
        }
        payload["receipt_sha256"] = _digest(payload)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, (json.dumps(
                payload, sort_keys=True, separators=(",", ":")) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        return payload


def prepare_controller_sandbox(
    *, workspace: str | Path, receipt_path: str | Path,
    expected_argv: Sequence[str], runtime: RuntimeAllowlist,
    broker_socket_path: str | Path, broker_peer_pid: int,
    broker_peer_start_ticks: int, cgroup_root: str | Path | None = None,
) -> ControllerSandboxInvocation:
    """Build the exact controller profile; no environment prefix is returned."""
    work = _exact_path(workspace, directory=True)
    receipt = Path(receipt_path).resolve()
    if receipt.exists() or not receipt.parent.is_dir():
        raise ControllerSandboxError(
            "activation receipt target must be new in an existing evaluator directory")
    checked_roots = _unique_exact(
        runtime.readable_roots, directory=True, workspace=work,
        forbidden_roots=())
    checked_files = _unique_exact(
        runtime.readable_files, directory=False, workspace=work,
        forbidden_roots=())
    checked_executables = _unique_exact(
        runtime.executable_files, directory=False, workspace=work,
        forbidden_roots=())
    if (tuple(map(str, checked_roots)) != runtime.readable_roots
            or tuple(map(str, checked_files)) != runtime.readable_files
            or tuple(map(str, checked_executables)) != runtime.executable_files):
        raise ControllerSandboxError("runtime allowlist changed during admission")
    for executable_path in checked_executables:
        if not os.access(executable_path, os.X_OK):
            raise ControllerSandboxError(
                f"runtime executable is not executable: {executable_path}")
    if not expected_argv or Path(expected_argv[0]).is_symlink():
        raise ControllerSandboxError("controller argv requires an exact non-symlink executable")
    executable = _exact_path(expected_argv[0], directory=False)
    argv = (str(executable), *map(str, expected_argv[1:]))
    admitted_roots = tuple(Path(path) for path in runtime.readable_roots)
    admitted_executables = {Path(path) for path in runtime.executable_files}
    if executable not in admitted_executables and not any(
            executable.is_relative_to(root) for root in admitted_roots):
        raise ControllerSandboxError(
            "controller executable is absent from the runtime allowlist")
    kwargs: dict[str, Any] = {}
    if cgroup_root is not None:
        kwargs["cgroup_root"] = str(_exact_path(cgroup_root, directory=True))
    policy = sandbox.SandboxPolicy(
        str(work), profile=sandbox.CONTROLLER_PROFILE,
        readable_roots=runtime.readable_roots,
        readable_files=runtime.readable_files,
        executable_files=runtime.executable_files,
        broker_socket_path=str(_exact_socket(broker_socket_path)),
        broker_peer_pid=broker_peer_pid,
        broker_peer_start_ticks=broker_peer_start_ticks,
        **kwargs,
    )
    return ControllerSandboxInvocation(
        policy=policy, receipt_path=receipt,
        expected_argv=argv, runtime=runtime)


__all__ = [
    "CONTROLLER_ENVIRONMENT", "ControllerSandboxError",
    "ControllerSandboxInvocation", "RuntimeAllowlist",
    "copy_controller_workspace", "discover_runtime_allowlist",
    "prepare_controller_sandbox",
]
