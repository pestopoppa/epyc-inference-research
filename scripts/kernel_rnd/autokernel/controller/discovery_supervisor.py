"""Durable, sealed, host-owned supervisor for AutoKernel discovery controllers.

The public launcher snapshots the exact Python execution closure and canonical
deployment config into a private runtime directory.  A dedicated tmux server
executes only that read-only closure.  The supervisor pins all state authority
by directory fd, contains every controller process in one owned cgroup v2
subtree, and records an exact hash-linked lifecycle ledger.
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
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from . import discovery_supervisor_secure as secure


class SupervisorError(RuntimeError):
    pass


SPEC_SCHEMA = "epyc.autokernel.discovery_supervisor_spec.v4"
IDENTITY_SCHEMA = "epyc.autokernel.discovery_supervisor_identity.v2"
LEDGER_SCHEMA = "epyc.autokernel.discovery_supervisor_ledger.v2"
FACTORY_MODULE = "scripts.kernel_rnd.autokernel.controller.discovery_deployment_factory"
SUPERVISOR_MODULE = "scripts.kernel_rnd.autokernel.controller.discovery_supervisor"
SECURE_MODULE = "scripts.kernel_rnd.autokernel.controller.discovery_supervisor_secure"
TMUX_SOCKET_NAME = "epyc-autokernel-supervisors"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_IMMUTABLE_CLOSURE_BASE = Path("/var/lib/epyc-autokernel/execution-closures")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_STATE_LIMIT = 64 * 1024 * 1024
GRAPH_EXECUTION_MODULES = {
    "deployment_factory":
        "scripts/kernel_rnd/autokernel/controller/discovery_deployment_factory.py",
    "discovery_controller":
        "scripts/kernel_rnd/autokernel/controller/discovery_controller.py",
    "hypotheses": "scripts/kernel_rnd/autokernel/controller/hypotheses.py",
    "do_not_repeat": "scripts/kernel_rnd/autokernel/controller/do_not_repeat.py",
    "discovery_telemetry":
        "scripts/kernel_rnd/autokernel/controller/discovery_telemetry.py",
    "gpu_discovery_runner": "scripts/benchmark/run_autokernel_gpu_discovery.py",
    "gpu_source_adapter":
        "scripts/kernel_rnd/autokernel/controller/gpu_source_adapter.py",
    "discovery_static_registry":
        "scripts/kernel_rnd/autokernel/controller/discovery_static_registry.py",
    "discovery_supervisor":
        "scripts/kernel_rnd/autokernel/controller/discovery_supervisor.py",
    "discovery_supervisor_secure":
        "scripts/kernel_rnd/autokernel/controller/discovery_supervisor_secure.py",
    "discovery_deployment":
        "scripts/kernel_rnd/autokernel/controller/discovery_deployment.py",
    "gpu_load_admission":
        "scripts/kernel_rnd/autokernel/controller/gpu_load_admission.py",
    "split_runtime_verifier":
        "scripts/kernel_rnd/autokernel/controller/split_runtime_verifier.py",
    "inference_window": "scripts/kernel_rnd/autokernel/execution/inference_window.py",
    "cpu_region_claim": "scripts/kernel_rnd/autokernel/execution/cpu_region_claim.py",
    "worktree": "scripts/kernel_rnd/autokernel/execution/worktree.py",
    "source_candidate": "scripts/kernel_rnd/autokernel/source_candidate.py",
    "instrument_integrity":
        "scripts/kernel_rnd/autokernel/execution/instrument_integrity.py",
    "t0_provider": "scripts/kernel_rnd/autokernel/execution/t0_provider.py",
    "evaluator_integrity": "scripts/kernel_rnd/autokernel/evaluator/integrity.py",
    "gpu_source_evidence":
        "scripts/kernel_rnd/autokernel/controller/gpu_source_evidence.py",
    "gpu_source_proofs":
        "scripts/kernel_rnd/autokernel/controller/gpu_source_proofs.py",
    "gpu_discovery_beliefs": "scripts/benchmark/autokernel_gpu_discovery_beliefs.py",
    "device_claim": "scripts/kernel_rnd/autokernel/resource/device_claim.py",
    "device_sampler": "scripts/kernel_rnd/autokernel/execution/device_sampler.py",
    "gpu_residency_sampler":
        "scripts/kernel_rnd/autokernel/controller/gpu_residency_sampler.py",
    "codex_container_actor":
        "scripts/kernel_rnd/autokernel/controller/codex_container_actor.py",
    "claude_fable5_critic_actor":
        "scripts/kernel_rnd/autokernel/controller/claude_fable5_critic_actor.py",
    "hypothesis_portfolio": "scripts/kernel_rnd/autokernel/hypothesis_portfolio.py",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return secure.canonical_bytes(value)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stable_path(path: Path, *, limit: int = _STATE_LIMIT) -> tuple[bytes, dict[str, int]]:
    try:
        fd, raw, identity = secure.open_stable(path, limit=limit)
    except secure.SecureRuntimeError as exc:
        raise SupervisorError(str(exc)) from exc
    os.close(fd)
    return raw, identity


def _file_sha256(path: Path) -> str:
    raw, _identity = _stable_path(path)
    return hashlib.sha256(raw).hexdigest()


def _read_start_ticks(pid: int) -> tuple[str, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    close = raw.rfind(b")")
    fields = raw[close + 1 :].split() if close >= 0 else []
    if len(fields) < 20:
        raise SupervisorError(f"/proc/{pid}/stat cannot prove process identity")
    return fields[0].decode("ascii", "replace"), int(fields[19])


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    if not value:
        raise SupervisorError("kernel boot id is empty")
    return value


def _host_identity() -> dict[str, str]:
    machine = Path("/etc/machine-id").read_text(encoding="ascii").strip()
    source, value = (
        ("machine-id", machine) if machine else ("kernel-hostname", socket.gethostname())
    )
    if not value:
        raise SupervisorError("host identity source is empty")
    return {"host_id_source": source, "host_id_sha256": hashlib.sha256(value.encode()).hexdigest()}


def _process_identity(pid: int) -> dict[str, Any]:
    current = _read_start_ticks(pid)
    if current is None:
        raise SupervisorError(f"pid {pid} exited before identity capture")
    return {
        "pid": pid,
        "start_ticks": current[1],
        "boot_id": _boot_id(),
        "host": socket.gethostname(),
        **_host_identity(),
    }


def _identity_liveness(identity: Mapping[str, Any]) -> tuple[str, str]:
    required = {"pid", "start_ticks", "boot_id", "host", "host_id_source", "host_id_sha256"}
    if not required <= set(identity):
        return "unknown", "identity is missing a required host/process field"
    if identity["host"] != socket.gethostname() or any(
        identity[key] != value for key, value in _host_identity().items()
    ):
        return "unknown", "identity belongs to another host"
    if identity["boot_id"] != _boot_id():
        return "dead", "identity predates the current boot"
    pid, ticks = identity["pid"], identity["start_ticks"]
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(ticks, int)
        or isinstance(ticks, bool)
    ):
        return "unknown", "identity PID/start ticks are malformed"
    current = _read_start_ticks(pid)
    if current is None or current[1] != ticks or current[0] == "Z":
        return "dead", "recorded PID is absent, recycled, or a zombie"
    return "live", "PID, start ticks, boot id, and host identity match"


def _runtime(path: Path) -> secure.RuntimeRoot:
    try:
        return secure.RuntimeRoot.create_or_open(path)
    except secure.SecureRuntimeError as exc:
        raise SupervisorError(str(exc)) from exc


def _ensure_private_root(path: Path) -> Path:
    root = _runtime(path)
    try:
        return root.path
    finally:
        root.close()


def _atomic_json(root: secure.RuntimeRoot, name: str, value: Mapping[str, Any]) -> None:
    root.atomic_bytes(name, _canonical_bytes(value) + b"\n")


def _read_json(root: secure.RuntimeRoot, name: str) -> dict[str, Any]:
    raw = root.read_bytes(name, limit=_STATE_LIMIT)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SupervisorError(f"private state is not JSON: {name}") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise SupervisorError(f"private state is not canonical: {name}")
    return value


def _sudo(*argv: str) -> None:
    result = subprocess.run(
        ("/usr/bin/sudo", "-n", *argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SupervisorError(f"immutable closure root operation failed: {result.stderr.strip()}")


def _install_root_owned_closure(staging: Path, destination: Path) -> None:
    """Publish one generated snapshot below a non-user-writable root parent."""
    if destination.parent != _IMMUTABLE_CLOSURE_BASE or not _HEX64.fullmatch(destination.name):
        raise SupervisorError("immutable closure destination escaped its base")
    _sudo(
        "/usr/bin/install",
        "-d",
        "-o",
        "root",
        "-g",
        "root",
        "-m",
        "0755",
        str(_IMMUTABLE_CLOSURE_BASE.parent),
        str(_IMMUTABLE_CLOSURE_BASE),
    )
    for parent in (_IMMUTABLE_CLOSURE_BASE.parent, _IMMUTABLE_CLOSURE_BASE):
        info = parent.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
            raise SupervisorError("immutable closure parent is not root-owned and non-writable")
    if destination.exists():
        shutil.rmtree(staging)
        return
    _sudo("/usr/bin/chown", "-R", "--no-dereference", "root:root", str(staging))
    _sudo("/usr/bin/chmod", "-R", "a+rX,a-w", str(staging))
    result = subprocess.run(
        (
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/mv",
            "-T",
            "--",
            str(staging),
            str(destination),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        if not destination.exists():
            raise SupervisorError(f"immutable closure publication failed: {result.stderr.strip()}")
        _sudo(
            "/usr/bin/rm",
            "-rf",
            "--one-file-system",
            "--",
            str(staging),
        )


def _sealed_closure_manifest(
    closure: Path, expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Require an exact root-owned 0555/0444 tree with no extra leaves."""
    actual_files: set[str] = set()
    actual: dict[str, dict[str, Any]] = {}
    root_fd = os.open(closure, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for path in closure.rglob("*"):
            relative = path.relative_to(closure).as_posix()
            info = path.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise SupervisorError("immutable closure contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                if info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o555:
                    raise SupervisorError("immutable closure directory is not root-owned mode-0555")
                continue
            if not stat.S_ISREG(info.st_mode):
                raise SupervisorError("immutable closure contains a special file")
            actual_files.add(relative)
        if actual_files != set(expected):
            raise SupervisorError("immutable closure file set differs from manifest")
        for relative, row in expected.items():
            fd = secure.open_beneath(root_fd, relative)
            try:
                raw, identity = secure.read_stable_fd(fd, limit=_STATE_LIMIT)
            finally:
                os.close(fd)
            if (
                identity["uid"] != 0
                or identity["nlink"] != 1
                or identity["mode"] != 0o444
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
            ):
                raise SupervisorError("immutable closure file identity or bytes changed")
            actual[relative] = {
                "sha256": row["sha256"],
                "source": row["source"],
                "closure": identity,
            }
    finally:
        os.close(root_fd)
    root_identity = secure.directory_identity(os.stat(closure, follow_symlinks=False))
    if root_identity["uid"] != 0 or root_identity["mode"] != 0o555:
        raise SupervisorError("immutable closure root is not root-owned mode-0555")
    return actual


def _copy_execution_closure(root: secure.RuntimeRoot) -> dict[str, Any]:
    """Copy exact source bytes, then publish them under root-only authority."""
    closure = root.path / "execution-closure"
    if closure.exists():
        raise SupervisorError("execution closure already exists before spec creation")
    closure.mkdir(mode=0o700)
    source_root_fd = os.open(_SOURCE_SCRIPTS_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    manifest: dict[str, dict[str, Any]] = {}
    selected: list[Path] = []
    autokernel = _SOURCE_SCRIPTS_ROOT / "kernel_rnd" / "autokernel"
    for path in autokernel.rglob("*"):
        relative = path.relative_to(_SOURCE_SCRIPTS_ROOT)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise SupervisorError(f"execution source closure contains symlink: {relative}")
        if stat.S_ISREG(info.st_mode):
            selected.append(relative)
    for relative in (
        Path("__init__.py"),
        Path("kernel_rnd/__init__.py"),
        Path("benchmark/__init__.py"),
        Path("benchmark/autokernel_gpu_discovery_beliefs.py"),
        Path("benchmark/autokernel_progression.py"),
        Path("benchmark/run_autokernel_gpu_discovery.py"),
        Path("lib/__init__.py"),
        Path("lib/canonical_recipe.py"),
    ):
        if relative not in selected and (_SOURCE_SCRIPTS_ROOT / relative).exists():
            selected.append(relative)
    try:
        for relative in sorted(selected, key=str):
            try:
                fd = secure.open_beneath(source_root_fd, relative.as_posix())
                try:
                    raw, source_identity = secure.read_stable_fd(fd, limit=_STATE_LIMIT)
                finally:
                    os.close(fd)
            except secure.SecureRuntimeError as exc:
                raise SupervisorError(str(exc)) from exc
            destination = closure / "scripts" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            out = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o400,
            )
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(out, view)
                    view = view[written:]
                os.fsync(out)
            finally:
                os.close(out)
            os.chmod(destination, 0o400)
            copied, _identity = _stable_path(destination)
            if copied != raw:
                raise SupervisorError("execution closure copy differs from opened source bytes")
            manifest[f"scripts/{relative.as_posix()}"] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source": source_identity,
                "closure": None,
            }
    finally:
        os.close(source_root_fd)
    content_manifest = {relative: row["sha256"] for relative, row in manifest.items()}
    content_sha256 = _content_hash(content_manifest)
    destination = _IMMUTABLE_CLOSURE_BASE / content_sha256
    _install_root_owned_closure(closure, destination)
    sealed_manifest = _sealed_closure_manifest(destination, manifest)
    return {
        "path": str(destination),
        "content_sha256": content_sha256,
        "manifest": sealed_manifest,
        "manifest_sha256": _content_hash(sealed_manifest),
        "root_identity": secure.directory_identity(os.stat(destination, follow_symlinks=False)),
    }


def _verify_execution_closure(spec: "LaunchSpec", *, require_self: bool = False) -> None:
    closure = Path(spec.body["execution_closure"]["path"])
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise SupervisorError("sealed execution requires bytecode-disabled Python")
    if closure.parent != _IMMUTABLE_CLOSURE_BASE:
        raise SupervisorError("execution closure is outside its root-owned base")
    if (
        secure.directory_identity(os.stat(closure, follow_symlinks=False))
        != spec.body["execution_closure"]["root_identity"]
    ):
        raise SupervisorError("execution closure root object changed")
    actual = _sealed_closure_manifest(closure, spec.body["execution_closure"]["manifest"])
    if (
        actual != spec.body["execution_closure"]["manifest"]
        or _content_hash(actual) != spec.body["execution_closure"]["manifest_sha256"]
        or _content_hash({relative: row["sha256"] for relative, row in actual.items()})
        != spec.body["execution_closure"]["content_sha256"]
    ):
        raise SupervisorError("execution closure bytes or identities changed")
    if not require_self:
        return
    here = Path(__file__).resolve()
    factory = here.with_name("discovery_deployment_factory.py")
    helper = here.with_name("discovery_supervisor_secure.py")
    expected_modules = spec.body["execution_modules"]
    found = {"supervisor": here, "deployment_factory": factory, "secure_runtime": helper}
    for name, path in found.items():
        expected = expected_modules[name]
        if str(path) != expected["path"] or _file_sha256(path) != expected["sha256"]:
            raise SupervisorError("supervisor/factory execution module bytes changed")


def _canonical_config(root: secure.RuntimeRoot, deployment: Path) -> dict[str, Any]:
    raw, source_identity = _stable_path(deployment)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SupervisorError("deployment config is not JSON") from exc
    canonical = _canonical_bytes(value) + b"\n"
    semantic_sha256 = value.get("config_sha256") if isinstance(value, dict) else None
    if not isinstance(semantic_sha256, str) or _HEX64.fullmatch(semantic_sha256) is None:
        raise SupervisorError("deployment config lacks a semantic config identity")
    if _content_hash({
            key: item for key, item in value.items()
            if key != "config_sha256"}) != semantic_sha256:
        raise SupervisorError("deployment config semantic identity is invalid")
    root.atomic_bytes("deployment-config.json", canonical)
    fd = root.open_leaf("deployment-config.json", os.O_RDONLY)
    try:
        copied, identity = secure.read_stable_fd(fd, limit=_STATE_LIMIT)
    finally:
        os.close(fd)
    if copied != canonical:
        raise SupervisorError("canonical config copy changed during creation")
    return {
        "source_path": str(deployment.absolute()),
        "source_identity": source_identity,
        "runtime_leaf": "deployment-config.json",
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "semantic_sha256": semantic_sha256,
        "canonical_size": len(canonical),
        "identity": identity,
    }


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
    def read(cls, root: secure.RuntimeRoot) -> "LaunchSpec":
        value = _read_json(root, "launch-spec.json")
        cls._validate(value)
        root.verify(value["runtime_root_identity"])
        return cls(value)

    @staticmethod
    def _validate(value: Mapping[str, Any]) -> None:
        expected = {
            "schema",
            "kind",
            "runtime_root",
            "runtime_root_identity",
            "deployment_config",
            "validate_only",
            "canary",
            "python",
            "restart_policy",
            "termination_policy",
            "execution_closure",
            "execution_modules",
            "graph_execution_modules",
            "cgroup",
        }
        if set(value) != expected or value.get("schema") != SPEC_SCHEMA:
            raise SupervisorError("launch specification schema/keys are invalid")
        if value.get("kind") not in {"deployment", "canary"}:
            raise SupervisorError("launch specification kind is invalid")
        for key in ("runtime_root", "python"):
            path = Path(str(value[key]))
            if not path.is_absolute() or ".." in path.parts:
                raise SupervisorError(f"launch specification {key} is invalid")
        if not isinstance(value["runtime_root_identity"], dict):
            raise SupervisorError("runtime root binding is invalid")
        config = value["deployment_config"]
        if value["kind"] == "deployment":
            config_keys = {
                "source_path",
                "source_identity",
                "runtime_leaf",
                "canonical_sha256",
                "semantic_sha256",
                "canonical_size",
                "identity",
            }
            if (
                not isinstance(config, dict)
                or set(config) != config_keys
                or _HEX64.fullmatch(str(config["canonical_sha256"])) is None
                or _HEX64.fullmatch(str(config["semantic_sha256"])) is None
            ):
                raise SupervisorError("deployment config binding is malformed")
        elif config is not None:
            raise SupervisorError("canary carries deployment config authority")
        restart = value["restart_policy"]
        if (
            set(restart) != {"max_restarts", "delay_seconds"}
            or not isinstance(restart["max_restarts"], int)
            or not 0 <= restart["max_restarts"] <= 10
        ):
            raise SupervisorError("restart policy is invalid")
        if value["kind"] == "deployment" and restart["max_restarts"] != 0:
            raise SupervisorError("deployment restart requires a typed reconciliation receipt")
        if value["kind"] == "canary":
            canary = value["canary"]
            if not isinstance(canary, dict) or set(canary) != {
                "hold_seconds",
                "exit_code",
                "spawn_descendant",
            }:
                raise SupervisorError("canary contract is malformed")
        if set(value["execution_modules"]) != {
            "supervisor",
            "deployment_factory",
            "secure_runtime",
        }:
            raise SupervisorError("execution module binding is invalid")
        graph_modules = value["graph_execution_modules"]
        if (not isinstance(graph_modules, dict)
                or set(graph_modules) != set(GRAPH_EXECUTION_MODULES)):
            raise SupervisorError("graph execution module binding is invalid")
        for name, logical in GRAPH_EXECUTION_MODULES.items():
            row = graph_modules[name]
            if (not isinstance(row, dict)
                    or set(row) != {"logical_path", "sha256"}
                    or row["logical_path"] != logical
                    or _HEX64.fullmatch(str(row["sha256"])) is None):
                raise SupervisorError("graph execution module identity is malformed")
        if not isinstance(value["execution_closure"], dict) or set(value["execution_closure"]) != {
            "path",
            "content_sha256",
            "manifest",
            "manifest_sha256",
            "root_identity",
        }:
            raise SupervisorError("execution closure binding is invalid")
        if _HEX64.fullmatch(str(value["execution_closure"]["content_sha256"])) is None:
            raise SupervisorError("execution closure content digest is invalid")
        manifest = value["execution_closure"]["manifest"]
        if any(logical not in manifest
               or manifest[logical].get("sha256") != graph_modules[name]["sha256"]
               for name, logical in GRAPH_EXECUTION_MODULES.items()):
            raise SupervisorError("graph modules differ from execution closure manifest")

    def child_argv(
        self, config_fd: int | None = None, authority_fd: int | None = None
    ) -> tuple[str, ...]:
        python = str(self.body["python"])
        if self.body["kind"] == "deployment":
            if config_fd is None or authority_fd is None:
                raise SupervisorError("deployment child requires inherited authority fds")
            argv = [
                python,
                "-B",
                "-m",
                FACTORY_MODULE,
                "--deployment",
                self.body["deployment_config"]["source_path"],
                "--supervised-config-fd",
                str(config_fd),
                "--supervised-authority-fd",
                str(authority_fd),
                "--supervisor-runtime-root",
                str(self.runtime_root),
            ]
            if self.body["validate_only"]:
                argv.append("--validate-only")
            return tuple(argv)
        canary = self.body["canary"]
        argv = [
            python,
            "-B",
            "-m",
            SUPERVISOR_MODULE,
            "_canary-child",
            "--hold-seconds",
            str(canary["hold_seconds"]),
            "--exit-code",
            str(canary["exit_code"]),
        ]
        if canary["spawn_descendant"]:
            argv.append("--spawn-descendant")
        return tuple(argv)


def _new_spec(
    *,
    runtime_root: Path,
    deployment: Path | None,
    validate_only: bool,
    canary: Mapping[str, Any] | None,
    max_restarts: int,
    restart_delay: float,
    term_grace: float,
    kill_grace: float,
) -> LaunchSpec:
    root = _runtime(runtime_root)
    try:
        if root.exists("launch-spec.json"):
            return LaunchSpec.read(root)
        closure = _copy_execution_closure(root)
        # The staging subdirectory has been atomically moved below the
        # root-owned closure base. Pin the runtime directory's final identity.
        root.identity = secure.directory_identity(os.fstat(root.fd))
        kind = "canary" if canary is not None else "deployment"
        config = None
        if kind == "deployment":
            if deployment is None:
                raise SupervisorError("deployment path is required")
            config = _canonical_config(root, deployment)
        else:
            validate_only = True
        module_base = Path(closure["path"]) / "scripts/kernel_rnd/autokernel/controller"
        modules = {}
        for name, filename in {
            "supervisor": "discovery_supervisor.py",
            "deployment_factory": "discovery_deployment_factory.py",
            "secure_runtime": "discovery_supervisor_secure.py",
        }.items():
            path = module_base / filename
            modules[name] = {"path": str(path), "sha256": _file_sha256(path)}
        graph_modules = {
            name: {"logical_path": logical,
                   "sha256": closure["manifest"][logical]["sha256"]}
            for name, logical in GRAPH_EXECUTION_MODULES.items()
        }
        body = {
            "schema": SPEC_SCHEMA,
            "kind": kind,
            "runtime_root": str(root.path),
            "runtime_root_identity": root.identity,
            "deployment_config": config,
            "validate_only": validate_only,
            "canary": dict(canary) if canary is not None else None,
            "python": str(Path(sys.executable).resolve(strict=True)),
            "restart_policy": {"max_restarts": max_restarts, "delay_seconds": float(restart_delay)},
            "termination_policy": {
                "term_grace_seconds": float(term_grace),
                "kill_grace_seconds": float(kill_grace),
            },
            "execution_closure": closure,
            "execution_modules": modules,
            "graph_execution_modules": graph_modules,
            "cgroup": {
                "name": f"epyc-autokernel-{hashlib.sha256(str(root.path).encode()).hexdigest()[:24]}",
                "base": "/sys/fs/cgroup",
            },
        }
        LaunchSpec._validate(body)
        return LaunchSpec(body)
    finally:
        root.close()


class DeathLedger:
    """Validated and appended through the same exclusively-locked fd."""

    def __init__(self, root: secure.RuntimeRoot) -> None:
        self.root = root
        self.records: list[dict[str, Any]] = []
        self.sequence = 0
        self.previous_sha256: str | None = None
        if root.exists("death-ledger.jsonl"):
            fd = root.open_append("death-ledger.jsonl")
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                self._load_fd(fd)
            finally:
                os.close(fd)

    def _load_fd(self, fd: int) -> None:
        raw, identity = secure.read_stable_fd(fd, limit=_STATE_LIMIT)
        if identity["mode"] != 0o600:
            raise SupervisorError("death ledger mode is invalid")
        records: list[dict[str, Any]] = []
        previous = None
        for number, line in enumerate(raw.splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SupervisorError("death ledger contains a torn record") from exc
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "schema",
                    "sequence",
                    "previous_sha256",
                    "written_at",
                    "event",
                    "payload",
                    "record_sha256",
                }
                or line != _canonical_bytes(row)
                or row["schema"] != LEDGER_SCHEMA
                or row["sequence"] != number
                or row["previous_sha256"] != previous
            ):
                raise SupervisorError("death ledger hash chain is malformed")
            claimed = row["record_sha256"]
            body = dict(row)
            body.pop("record_sha256")
            if claimed != _content_hash(body):
                raise SupervisorError("death ledger record digest is invalid")
            previous = claimed
            records.append(row)
        _validate_ledger_fsm(records)
        self.records, self.sequence, self.previous_sha256 = records, len(records), previous

    def append(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        fd = self.root.open_append("death-ledger.jsonl")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._load_fd(fd)
            body = {
                "schema": LEDGER_SCHEMA,
                "sequence": self.sequence + 1,
                "previous_sha256": self.previous_sha256,
                "written_at": _utc_now(),
                "event": event,
                "payload": dict(payload),
            }
            body["record_sha256"] = _content_hash(body)
            prospective = [*self.records, body]
            _validate_ledger_fsm(prospective)
            os.lseek(fd, 0, os.SEEK_END)
            raw = _canonical_bytes(body) + b"\n"
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
            self.records, self.sequence = prospective, len(prospective)
            self.previous_sha256 = body["record_sha256"]
            return body
        finally:
            os.close(fd)


def _validate_ledger_fsm(records: Sequence[Mapping[str, Any]]) -> None:
    state = "empty"
    supervisor = None
    child = None
    restart = 0
    last_code = None
    for row in records:
        event, payload = row["event"], row["payload"]
        if not isinstance(payload, dict):
            raise SupervisorError("death ledger payload is not an object")
        if event == "supervisor_started":
            if state not in {"empty", "stopped"} or set(payload) != {
                "spec_sha256",
                "session_name",
                "supervisor",
                "tmux",
            }:
                raise SupervisorError("invalid supervisor_started transition")
            supervisor, child, restart, state = payload["supervisor"], None, 0, "ready"
        elif event == "child_started":
            if (
                state not in {"ready", "restart"}
                or child is not None
                or set(payload) != {"restart_count", "child", "stdout", "stderr", "cgroup"}
                or payload["restart_count"] != restart
            ):
                raise SupervisorError("invalid child_started transition")
            child, state = payload["child"], "running"
        elif event == "signal_forwarded":
            if (
                state != "running"
                or set(payload) != {"signal", "child"}
                or payload["child"] != child
            ):
                raise SupervisorError("invalid signal_forwarded transition")
        elif event == "child_exited":
            if (
                state != "running"
                or set(payload)
                != {"restart_count", "return_code", "cleanup_actions", "stop_signal"}
                or payload["restart_count"] != restart
            ):
                raise SupervisorError("invalid child_exited transition")
            last_code, child, state = payload["return_code"], None, "exited"
        elif event == "restart_scheduled":
            if (
                state != "exited"
                or not last_code
                or set(payload) != {"restart_count", "delay_seconds", "last_return_code"}
                or payload["restart_count"] != restart + 1
                or payload["last_return_code"] != last_code
            ):
                raise SupervisorError("invalid restart_scheduled transition")
            restart, state = payload["restart_count"], "restart"
        elif event == "restarts_exhausted":
            if (
                state != "exited"
                or not last_code
                or set(payload) != {"restart_count", "max_restarts", "last_return_code"}
                or payload["restart_count"] != restart
                or payload["last_return_code"] != last_code
            ):
                raise SupervisorError("invalid restarts_exhausted transition")
            state = "terminal"
        elif event == "supervisor_fault":
            if state in {"empty", "stopped"} or set(payload) != {
                "exception_type",
                "message",
                "cleanup_actions",
                "cleanup_error",
                "active_child_started_record_sha256",
                "cgroup",
            }:
                raise SupervisorError("invalid supervisor_fault transition")
            child, state = None, "terminal"
        elif event == "supervisor_stopped":
            if (
                state not in {"ready", "exited", "terminal", "restart"}
                or child is not None
                or set(payload) != {"exit_code", "restart_count", "stop_signal", "supervisor"}
                or payload["restart_count"] != restart
                or payload["supervisor"] != supervisor
            ):
                raise SupervisorError("invalid supervisor_stopped transition")
            state = "stopped"
        else:
            raise SupervisorError("unknown death ledger event")


def _tmux(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("tmux", "-L", TMUX_SOCKET_NAME, *args),
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        check=False,
    )


def _tmux_binding(session_name: str) -> dict[str, Any] | None:
    result = _tmux(
        "display-message",
        "-p",
        "-t",
        f"{session_name}:0.0",
        "#{session_id}\t#{pane_id}\t#{pane_pid}",
    )
    if result.returncode:
        return None
    parts = result.stdout.strip().split("\t")
    if len(parts) != 3:
        raise SupervisorError("tmux returned a malformed pane identity")
    pid = int(parts[2])
    ticks = _read_start_ticks(pid)
    if ticks is None:
        raise SupervisorError("tmux pane PID disappeared during identity capture")
    return {
        "session_id": parts[0],
        "pane_id": parts[1],
        "pane_pid": pid,
        "pane_start_ticks": ticks[1],
    }


def _tmux_has_session(session_name: str) -> bool:
    return _tmux_binding(session_name) is not None


def _validate_tmux_binding(
    binding: Mapping[str, Any], supervisor: Mapping[str, Any], session_name: str
) -> None:
    current = _tmux_binding(session_name)
    if (
        current != dict(binding)
        or current is None
        or current["pane_pid"] != supervisor["pid"]
        or current["pane_start_ticks"] != supervisor["start_ticks"]
    ):
        raise SupervisorError("tmux pane/session identity is not bound to supervisor")


def _persist_spec(root: secure.RuntimeRoot, spec: LaunchSpec) -> None:
    raw = _canonical_bytes(dict(spec.body)) + b"\n"
    if root.exists("launch-spec.json"):
        if root.read_bytes("launch-spec.json") != raw:
            raise SupervisorError("runtime root is bound to a different launch spec")
    else:
        root.atomic_bytes("launch-spec.json", raw)


def _write_identity(
    root: secure.RuntimeRoot,
    spec: LaunchSpec,
    *,
    state: str,
    supervisor: Mapping[str, Any],
    tmux: Mapping[str, Any],
    child: Mapping[str, Any] | None,
    restarts: int,
    exit_code: int | None = None,
) -> dict[str, Any]:
    value = {
        "schema": IDENTITY_SCHEMA,
        "spec_sha256": spec.sha256,
        "session_name": spec.session_name,
        "tmux_socket_name": TMUX_SOCKET_NAME,
        "state": state,
        "updated_at": _utc_now(),
        "supervisor": dict(supervisor),
        "tmux": dict(tmux),
        "child": dict(child) if child else None,
        "restart_count": restarts,
        "exit_code": exit_code,
    }
    _atomic_json(root, "identity.json", value)
    return value


def _validate_identity(value: Mapping[str, Any], spec: LaunchSpec) -> None:
    if (
        set(value)
        != {
            "schema",
            "spec_sha256",
            "session_name",
            "tmux_socket_name",
            "state",
            "updated_at",
            "supervisor",
            "tmux",
            "child",
            "restart_count",
            "exit_code",
        }
        or value["schema"] != IDENTITY_SCHEMA
    ):
        raise SupervisorError("supervisor identity schema/keys are invalid")
    if value["spec_sha256"] != spec.sha256 or value["session_name"] != spec.session_name:
        raise SupervisorError("supervisor identity is not bound to this launch spec")
    if value["state"] not in {"starting", "running", "stopped"}:
        raise SupervisorError("supervisor identity state is invalid")


def _validate_config_fd(spec: LaunchSpec, fd: int) -> bytes:
    expected = spec.body["deployment_config"]
    raw, identity = secure.read_stable_fd(fd, limit=_STATE_LIMIT)
    if (
        identity != expected["identity"]
        or len(raw) != expected["canonical_size"]
        or hashlib.sha256(raw).hexdigest() != expected["canonical_sha256"]
    ):
        raise SupervisorError("deployment config object differs from launch spec")
    try:
        value = json.loads(raw)
        if raw != _canonical_bytes(value) + b"\n":
            raise SupervisorError("deployment config bytes are not canonical")
        if (not isinstance(value, dict)
                or value.get("config_sha256") != expected["semantic_sha256"]
                or _content_hash({key: item for key, item in value.items()
                                  if key != "config_sha256"})
                != expected["semantic_sha256"]):
            raise SupervisorError("deployment config semantic identity differs from launch spec")
    except json.JSONDecodeError as exc:
        raise SupervisorError("deployment config bytes are not JSON") from exc
    return raw


def verified_supervised_config(runtime_root: Path, fd: int) -> bytes:
    """Factory-side independent verification of its inherited config object."""
    root = _runtime(runtime_root)
    try:
        spec = LaunchSpec.read(root)
        _verify_execution_closure(spec, require_self=True)
        return _validate_config_fd(spec, fd)
    finally:
        root.close()


def verified_supervised_launch(
    runtime_root: Path, config_fd: int, authority_fd: int
) -> tuple[bytes, dict[str, Any]]:
    """Factory-side proof of the spec, ledger, controller and config carrier."""
    root = _runtime(runtime_root)
    try:
        spec = LaunchSpec.read(root)
        _verify_execution_closure(spec, require_self=True)
        config = _validate_config_fd(spec, config_fd)
        raw, identity = secure.read_stable_fd(authority_fd, limit=1024 * 1024)
        if identity["mode"] != 0o600 or raw[-1:] != b"\n":
            raise SupervisorError("supervised build authority object is malformed")
        authority = json.loads(raw)
        if not isinstance(authority, dict) or raw != _canonical_bytes(authority) + b"\n":
            raise SupervisorError("supervised build authority is not canonical")
        expected_keys = {
            "schema",
            "launch_spec",
            "death_ledger",
            "spec_sha256",
            "deployment_config_canonical_sha256",
            "deployment_config_semantic_sha256",
            "supervisor",
            "controller",
            "ledger_child_started_record_sha256",
        }
        if (
            set(authority) != expected_keys
            or authority["schema"] != "epyc.autokernel.supervised_build_authority.v2"
            or authority["spec_sha256"] != spec.sha256
            or authority["deployment_config_canonical_sha256"]
            != spec.body["deployment_config"]["canonical_sha256"]
            or authority["deployment_config_semantic_sha256"]
            != spec.body["deployment_config"]["semantic_sha256"]
        ):
            raise SupervisorError("supervised build authority binding is invalid")
        spec_fd = root.open_leaf("launch-spec.json", os.O_RDONLY)
        ledger_fd = root.open_leaf("death-ledger.jsonl", os.O_RDONLY)
        try:
            spec_raw, spec_identity = secure.read_stable_fd(spec_fd, limit=_STATE_LIMIT)
            _ledger_raw, ledger_identity = secure.read_stable_fd(ledger_fd, limit=_STATE_LIMIT)
        finally:
            os.close(spec_fd)
            os.close(ledger_fd)

        def carrier(path: Path, object_: Mapping[str, int], sha: str | None = None):
            row = {
                "path": str(path),
                "device": object_["dev"],
                "inode": object_["ino"],
                "uid": object_["uid"],
                "mode": object_["mode"],
                "nlink": object_["nlink"],
            }
            if sha is not None:
                row["sha256"] = sha
            return row

        if authority["launch_spec"] != carrier(
            root.path / "launch-spec.json", spec_identity, hashlib.sha256(spec_raw).hexdigest()
        ) or authority["death_ledger"] != carrier(
            root.path / "death-ledger.jsonl", ledger_identity
        ):
            raise SupervisorError("supervised authority state object changed")
        ledger = DeathLedger(root)
        matching = [
            row
            for row in ledger.records
            if row["event"] == "child_started"
            and row["record_sha256"] == authority["ledger_child_started_record_sha256"]
        ]
        if len(matching) != 1 or matching[0]["payload"]["child"] != authority["controller"]:
            raise SupervisorError("supervised authority lacks exact child_started linkage")
        current = _process_identity(os.getpid())
        controller = authority["controller"]
        if any(controller.get(key) != value for key, value in current.items()) or controller.get(
            "pgid"
        ) != os.getpgid(0):
            raise SupervisorError("factory process differs from supervised controller identity")
        status_identity = _read_json(root, "identity.json")
        if (
            status_identity["supervisor"] != authority["supervisor"]
            or status_identity["child"] != controller
        ):
            raise SupervisorError("live identity differs from supervised build authority")
        return config, authority
    finally:
        root.close()


def _validate_imported_module_set(
    expected: Mapping[str, Mapping[str, str]],
    modules: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(modules, Mapping) or not modules:
        raise SupervisorError("imported execution module provenance is empty")
    semantic = {
        name: {"logical_path": row.get("logical_path"),
               "sha256": row.get("sha256")}
        for name, row in modules.items() if isinstance(row, Mapping)
    }
    if semantic != expected:
        raise SupervisorError(
            "imported execution module set differs from launch authority")


def verify_imported_execution_modules(
    runtime_root: Path, modules: Mapping[str, Mapping[str, Any]]
) -> None:
    """Bind imported module objects to the exact root-sealed closure manifest."""
    root = _runtime(runtime_root)
    closure_fd = -1
    try:
        spec = LaunchSpec.read(root)
        _verify_execution_closure(spec, require_self=True)
        closure = Path(spec.body["execution_closure"]["path"])
        manifest = spec.body["execution_closure"]["manifest"]
        closure_fd = os.open(closure, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        required = {
            "logical_path", "path", "sha256", "dev", "ino", "uid", "mode",
            "nlink", "size", "mtime_ns", "ctime_ns",
        }
        _validate_imported_module_set(spec.body["graph_execution_modules"], modules)
        seen: set[str] = set()
        for name, row in modules.items():
            if not isinstance(name, str) or not isinstance(row, Mapping) \
                    or set(row) != required:
                raise SupervisorError("imported execution module provenance is malformed")
            logical = row["logical_path"]
            if (not isinstance(logical, str) or logical in seen
                    or logical.startswith("/") or ".." in Path(logical).parts
                    or not logical.startswith("scripts/") or logical not in manifest):
                raise SupervisorError("imported execution module logical path is invalid")
            seen.add(logical)
            expected_path = closure / logical
            if row["path"] != str(expected_path):
                raise SupervisorError("imported execution module escaped sealed closure")
            try:
                fd = secure.open_beneath(closure_fd, logical)
                raw, identity = secure.read_stable_fd(fd, limit=_STATE_LIMIT)
                opened = os.fstat(fd)
            except secure.SecureRuntimeError as exc:
                raise SupervisorError(str(exc)) from exc
            finally:
                if "fd" in locals():
                    os.close(fd)
                    del fd
            facts = {
                "dev": identity["dev"], "ino": identity["ino"],
                "uid": identity["uid"], "mode": identity["mode"],
                "nlink": identity["nlink"], "size": identity["size"],
                "mtime_ns": opened.st_mtime_ns, "ctime_ns": opened.st_ctime_ns,
            }
            if ({key: row[key] for key in facts} != facts
                    or row["sha256"] != hashlib.sha256(raw).hexdigest()
                    or row["sha256"] != manifest[logical]["sha256"]
                    or identity != manifest[logical]["closure"]):
                raise SupervisorError(
                    "imported execution module object differs from sealed manifest")
    finally:
        if closure_fd >= 0:
            os.close(closure_fd)
        root.close()


def _status_payload(runtime_root: Path) -> dict[str, Any]:
    root = _runtime(runtime_root)
    try:
        if not root.exists("launch-spec.json"):
            return {
                "status": "absent",
                "reason": "no launch specification",
                "runtime_root": str(root.path),
                "identity": None,
            }
        spec = LaunchSpec.read(root)
        identity = _read_json(root, "identity.json") if root.exists("identity.json") else None
        if identity is None:
            return {
                "status": "dead",
                "reason": "no supervisor identity",
                "runtime_root": str(root.path),
                "spec_sha256": spec.sha256,
                "session_name": spec.session_name,
                "tmux_session": False,
                "ledger_sequence": 0,
                "identity": None,
            }
        _validate_identity(identity, spec)
        ledger = DeathLedger(root)
        liveness = _identity_liveness(identity["supervisor"])
        if liveness[0] == "live":
            _verify_execution_closure(spec)
            _validate_tmux_binding(identity["tmux"], identity["supervisor"], spec.session_name)
        return {
            "status": liveness[0],
            "reason": liveness[1],
            "runtime_root": str(root.path),
            "spec_sha256": spec.sha256,
            "session_name": spec.session_name,
            "tmux_session": _tmux_binding(spec.session_name) is not None,
            "ledger_sequence": ledger.sequence,
            "identity": identity,
        }
    finally:
        root.close()


def start_detached(spec: LaunchSpec, *, start_timeout: float = 20.0) -> dict[str, Any]:
    root = _runtime(spec.runtime_root)
    try:
        _persist_spec(root, spec)
    finally:
        root.close()
    current = _status_payload(spec.runtime_root)
    if current["status"] == "live":
        return {**current, "launch_result": "already_running"}
    if _tmux_binding(spec.session_name) is not None:
        raise SupervisorError("tmux session exists without matching live identity")
    closure = spec.body["execution_closure"]["path"]
    command = (
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={closure}",
        str(spec.body["python"]),
        "-B",
        "-m",
        SUPERVISOR_MODULE,
        "_run",
        "--runtime-root",
        str(spec.runtime_root),
    )
    result = _tmux("new-session", "-d", "-s", spec.session_name, "-c", str(closure), "--", *command)
    if result.returncode:
        raise SupervisorError(f"tmux launch failed: {result.stderr.strip()}")
    deadline = time.monotonic() + start_timeout
    while time.monotonic() < deadline:
        current = _status_payload(spec.runtime_root)
        if current["status"] == "live":
            return {**current, "launch_result": "started"}
        if _tmux_binding(spec.session_name) is None:
            break
        time.sleep(0.05)
    raise SupervisorError("detached supervisor did not publish a bound live identity")


def _member_identities(cgroup: secure.OwnedCgroup) -> dict[int, int]:
    identities = {}
    for pid in cgroup.pids():
        current = _read_start_ticks(pid)
        if current is not None:
            identities[pid] = current[1]
    return identities


def _cleanup_cgroup(
    cgroup: secure.OwnedCgroup, *, term_grace: float, kill_grace: float, term_already_sent: bool
) -> list[str]:
    actions: list[str] = []
    if cgroup.pids() and not term_already_sent:
        identities = _member_identities(cgroup)
        if cgroup.signal_all(signal.SIGTERM, identities):
            actions.append("pidfd:SIGTERM")
    if not cgroup.wait_empty(term_grace):
        cgroup.kill()
        actions.append("cgroup.kill")
    if not cgroup.wait_empty(kill_grace):
        raise SupervisorError("owned controller cgroup survived cgroup.kill")
    return actions


def _open_log(root: secure.RuntimeRoot, name: str):
    return os.fdopen(root.open_append(name), "ab", buffering=0)


def _launch_child(
    spec: LaunchSpec, root: secure.RuntimeRoot, cgroup: secure.OwnedCgroup
) -> tuple[subprocess.Popen[bytes], dict[str, Any], int, int | None]:
    config_fd = None
    pass_fds: tuple[int, ...] = ()
    if spec.body["kind"] == "deployment":
        config_fd = root.open_leaf(spec.body["deployment_config"]["runtime_leaf"], os.O_RDONLY)
        _validate_config_fd(spec, config_fd)
        os.set_inheritable(config_fd, True)
        pass_fds = (config_fd,)
    authority_fd = None
    if spec.body["kind"] == "deployment":
        authority_leaf = f"launch-authority.{os.getpid()}.{time.monotonic_ns()}.json"
        authority_fd = root.open_leaf(authority_leaf, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.set_inheritable(authority_fd, True)
        pass_fds = (*pass_fds, authority_fd)
    argv = spec.child_argv(config_fd, authority_fd)
    gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
    os.set_inheritable(gate_read, True)
    bootstrap = (
        str(spec.body["python"]),
        "-B",
        "-m",
        SUPERVISOR_MODULE,
        "_child-bootstrap",
        "--gate-fd",
        str(gate_read),
        "--",
        *argv,
    )
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(spec.body["execution_closure"]["path"]),
    }
    with (
        _open_log(root, "controller.stdout.log") as stdout,
        _open_log(root, "controller.stderr.log") as stderr,
    ):
        process = subprocess.Popen(
            bootstrap,
            cwd=spec.body["execution_closure"]["path"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            pass_fds=(*pass_fds, gate_read),
        )
    os.close(gate_read)
    try:
        cgroup.add(process.pid)
        identity = _process_identity(process.pid)
    except Exception:
        os.close(gate_write)
        if config_fd is not None:
            os.close(config_fd)
        if authority_fd is not None:
            os.close(authority_fd)
        raise
    identity["pgid"] = os.getpgid(process.pid)
    identity["argv_sha256"] = _content_hash(list(argv))
    if config_fd is not None:
        os.close(config_fd)
    return process, identity, gate_write, authority_fd


def _publish_launch_authority(
    root: secure.RuntimeRoot,
    spec: LaunchSpec,
    authority_fd: int,
    supervisor: Mapping[str, Any],
    controller: Mapping[str, Any],
    child_record: Mapping[str, Any],
) -> None:
    spec_fd = root.open_leaf("launch-spec.json", os.O_RDONLY)
    ledger_fd = root.open_leaf("death-ledger.jsonl", os.O_RDONLY)
    try:
        spec_raw, spec_identity = secure.read_stable_fd(spec_fd, limit=_STATE_LIMIT)
        _ledger_raw, ledger_identity = secure.read_stable_fd(ledger_fd, limit=_STATE_LIMIT)
    finally:
        os.close(spec_fd)
        os.close(ledger_fd)

    def carrier(path: Path, object_: Mapping[str, int], sha: str | None = None):
        row = {
            "path": str(path),
            "device": object_["dev"],
            "inode": object_["ino"],
            "uid": object_["uid"],
            "mode": object_["mode"],
            "nlink": object_["nlink"],
        }
        if sha is not None:
            row["sha256"] = sha
        return row

    value = {
        "schema": "epyc.autokernel.supervised_build_authority.v2",
        "launch_spec": carrier(
            root.path / "launch-spec.json", spec_identity, hashlib.sha256(spec_raw).hexdigest()
        ),
        "death_ledger": carrier(root.path / "death-ledger.jsonl", ledger_identity),
        "spec_sha256": spec.sha256,
        "deployment_config_canonical_sha256":
            spec.body["deployment_config"]["canonical_sha256"],
        "deployment_config_semantic_sha256":
            spec.body["deployment_config"]["semantic_sha256"],
        "supervisor": dict(supervisor),
        "controller": dict(controller),
        "ledger_child_started_record_sha256": child_record["record_sha256"],
    }
    raw = _canonical_bytes(value) + b"\n"
    os.ftruncate(authority_fd, 0)
    os.lseek(authority_fd, 0, os.SEEK_SET)
    view = memoryview(raw)
    while view:
        written = os.write(authority_fd, view)
        view = view[written:]
    os.fsync(authority_fd)
    os.lseek(authority_fd, 0, os.SEEK_SET)


def _child_bootstrap(gate_fd: int, argv: Sequence[str]) -> int:
    """Wait until the parent proves cgroup membership, then exec exact argv."""
    if gate_fd < 3 or not argv:
        raise SupervisorError("child bootstrap authority is malformed")
    token = os.read(gate_fd, 2)
    os.close(gate_fd)
    if token != b"G":
        raise SupervisorError("child bootstrap did not receive its cgroup gate")
    os.execvpe(argv[0], list(argv), os.environ)
    raise AssertionError("execvpe returned")


def supervise(runtime_root: Path) -> int:
    root = _runtime(runtime_root)
    lock_fd = -1
    cgroup = None
    try:
        spec = LaunchSpec.read(root)
        _verify_execution_closure(spec, require_self=True)
        lock_fd = root.open_leaf("supervisor.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError("another supervisor holds the singleton lock") from exc
        supervisor_identity = _process_identity(os.getpid())
        binding = _tmux_binding(spec.session_name)
        if binding is None:
            raise SupervisorError("supervisor has no owning tmux session")
        _validate_tmux_binding(binding, supervisor_identity, spec.session_name)
        ledger = DeathLedger(root)
        ledger.append(
            "supervisor_started",
            {
                "spec_sha256": spec.sha256,
                "session_name": spec.session_name,
                "supervisor": supervisor_identity,
                "tmux": binding,
            },
        )
        _write_identity(
            root,
            spec,
            state="starting",
            supervisor=supervisor_identity,
            tmux=binding,
            child=None,
            restarts=0,
        )
        requested_signal = 0

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal requested_signal
            if not requested_signal:
                requested_signal = signum

        previous = {
            sig: signal.signal(sig, request_stop)
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        }
        restarts = 0
        final_code = 0
        active = None
        active_child_record = None
        cgroup_identity = None
        try:
            while True:
                cgroup = secure.OwnedCgroup(
                    f"{spec.body['cgroup']['name']}-{supervisor_identity['pid']}-{restarts}",
                    base=Path(spec.body["cgroup"]["base"]),
                )
                cgroup.create()
                cgroup_identity = cgroup.identity()
                process, child_identity, gate_write, authority_fd = _launch_child(
                    spec, root, cgroup
                )
                active = process
                child_record = ledger.append(
                    "child_started",
                    {
                        "restart_count": restarts,
                        "child": child_identity,
                        "stdout": str(root.path / "controller.stdout.log"),
                        "stderr": str(root.path / "controller.stderr.log"),
                        "cgroup": cgroup_identity,
                    },
                )
                active_child_record = child_record["record_sha256"]
                _write_identity(
                    root,
                    spec,
                    state="running",
                    supervisor=supervisor_identity,
                    tmux=binding,
                    child=child_identity,
                    restarts=restarts,
                )
                if authority_fd is not None:
                    _publish_launch_authority(
                        root, spec, authority_fd, supervisor_identity, child_identity, child_record
                    )
                    os.close(authority_fd)
                os.write(gate_write, b"G")
                os.close(gate_write)
                while process.poll() is None and not requested_signal:
                    time.sleep(0.05)
                forwarded = False
                if requested_signal and process.poll() is None:
                    forwarded = cgroup.signal_all(requested_signal, _member_identities(cgroup))
                    ledger.append(
                        "signal_forwarded", {"signal": requested_signal, "child": child_identity}
                    )
                actions = _cleanup_cgroup(
                    cgroup,
                    term_grace=spec.body["termination_policy"]["term_grace_seconds"],
                    kill_grace=spec.body["termination_policy"]["kill_grace_seconds"],
                    term_already_sent=forwarded,
                )
                return_code = process.wait(
                    timeout=spec.body["termination_policy"]["kill_grace_seconds"]
                )
                cgroup.close_and_remove()
                if cgroup.path.exists():
                    raise SupervisorError("controller cgroup survived exact removal")
                actions.append("cgroup.remove")
                cgroup = None
                ledger.append(
                    "child_exited",
                    {
                        "restart_count": restarts,
                        "return_code": return_code,
                        "cleanup_actions": actions,
                        "stop_signal": requested_signal or None,
                    },
                )
                active = None
                active_child_record = None
                cgroup_identity = None
                if requested_signal:
                    final_code = 128 + requested_signal
                    break
                if return_code == 0:
                    final_code = 0
                    break
                maximum = spec.body["restart_policy"]["max_restarts"]
                if restarts >= maximum:
                    ledger.append(
                        "restarts_exhausted",
                        {
                            "restart_count": restarts,
                            "max_restarts": maximum,
                            "last_return_code": return_code,
                        },
                    )
                    final_code = return_code
                    break
                restarts += 1
                delay = spec.body["restart_policy"]["delay_seconds"]
                ledger.append(
                    "restart_scheduled",
                    {
                        "restart_count": restarts,
                        "delay_seconds": delay,
                        "last_return_code": return_code,
                    },
                )
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not requested_signal:
                    time.sleep(min(0.05, deadline - time.monotonic()))
        except Exception as exc:
            actions = []
            cleanup_error = None
            try:
                if cgroup is not None:
                    actions = _cleanup_cgroup(
                        cgroup,
                        term_grace=spec.body["termination_policy"]["term_grace_seconds"],
                        kill_grace=spec.body["termination_policy"]["kill_grace_seconds"],
                        term_already_sent=False,
                    )
                    cgroup.close_and_remove()
                    actions.append("cgroup.remove")
                    cgroup = None
                if active is not None and active.poll() is None:
                    active.wait(timeout=spec.body["termination_policy"]["kill_grace_seconds"])
            except Exception as cleanup_exc:
                cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            ledger.append(
                "supervisor_fault",
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "cleanup_actions": actions,
                    "cleanup_error": cleanup_error,
                    "active_child_started_record_sha256": active_child_record,
                    "cgroup": cgroup_identity,
                },
            )
            final_code = 70
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        ledger.append(
            "supervisor_stopped",
            {
                "exit_code": final_code,
                "restart_count": restarts,
                "stop_signal": requested_signal or None,
                "supervisor": supervisor_identity,
            },
        )
        _write_identity(
            root,
            spec,
            state="stopped",
            supervisor=supervisor_identity,
            tmux=binding,
            child=None,
            restarts=restarts,
            exit_code=final_code,
        )
        return final_code
    finally:
        if cgroup is not None:
            try:
                if cgroup.populated():
                    cgroup.kill()
                    cgroup.wait_empty(2.0)
                cgroup.close_and_remove()
            except (OSError, secure.SecureRuntimeError):
                pass
        if lock_fd >= 0:
            os.close(lock_fd)
        root.close()


def stop_supervisor(runtime_root: Path, *, timeout: float = 15.0) -> dict[str, Any]:
    status = _status_payload(runtime_root)
    if status["status"] == "dead":
        return {**status, "stop_result": "already_stopped"}
    if status["status"] != "live":
        raise SupervisorError(f"supervisor is not safely signalable: {status['reason']}")
    identity = status["identity"]["supervisor"]
    _validate_tmux_binding(status["identity"]["tmux"], identity, status["session_name"])
    try:
        pidfd = os.pidfd_open(identity["pid"], 0)
    except ProcessLookupError:
        return {**_status_payload(runtime_root), "stop_result": "already_stopped"}
    try:
        if _identity_liveness(identity)[0] != "live":
            raise SupervisorError("supervisor identity changed while opening pidfd")
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
    finally:
        os.close(pidfd)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _identity_liveness(identity)[0] != "dead":
        time.sleep(0.05)
    if _identity_liveness(identity)[0] != "dead":
        raise SupervisorError("supervisor did not stop after SIGTERM")
    return {**_status_payload(runtime_root), "stop_result": "stopped"}


def _canary_child(hold_seconds: float, exit_code: int, spawn_descendant: bool) -> int:
    descendant = None
    descendant_cgroup = None
    if spawn_descendant:
        descendant = subprocess.Popen(
            (
                str(Path(sys.executable).resolve()),
                "-B",
                "-c",
                f"import time; time.sleep({hold_seconds + 60.0!r})",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        unified = [line.split("::", 1)[1] for line in
                   Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
                   if "::" in line]
        if len(unified) != 1:
            raise SupervisorError("canary cannot identify its controller cgroup")
        descendant_cgroup = Path(
            "/sys/fs/cgroup", unified[0].lstrip("/"),
            f"canary-nested-{descendant.pid}")
        descendant_cgroup.mkdir(mode=0o700)
        Path(descendant_cgroup, "cgroup.procs").write_text(
            str(descendant.pid), encoding="ascii")
    print(
        json.dumps(
            {
                "schema": "epyc.autokernel.discovery_supervisor_canary.v2",
                "pid": os.getpid(),
                "start_ticks": _read_start_ticks(os.getpid())[1],
                "descendant_pid": descendant.pid if descendant else None,
                "descendant_cgroup": (str(descendant_cgroup)
                                      if descendant_cgroup is not None else None),
                "hardware_accessed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    time.sleep(hold_seconds)
    return exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--deployment", required=True)
    start.add_argument("--runtime-root", required=True)
    start.add_argument("--validate-only", action="store_true")
    canary = sub.add_parser("canary")
    canary.add_argument("--runtime-root", required=True)
    canary.add_argument("--hold-seconds", type=float, default=5.0)
    canary.add_argument("--exit-code", type=int, default=0)
    canary.add_argument("--spawn-descendant", action="store_true")
    start.add_argument("--max-restarts", type=int, default=0)
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
    run.add_argument("--runtime-root", required=True)
    child = sub.add_parser("_canary-child")
    child.add_argument("--hold-seconds", required=True, type=float)
    child.add_argument("--exit-code", required=True, type=int)
    child.add_argument("--spawn-descendant", action="store_true")
    bootstrap = sub.add_parser("_child-bootstrap")
    bootstrap.add_argument("--gate-fd", required=True, type=int)
    bootstrap.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def _require_cli_runtime() -> None:
    if (
        not sys.dont_write_bytecode
        or "-B" not in sys.orig_argv[1:]
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise SupervisorError(
            "public supervisor CLI requires python -B and PYTHONDONTWRITEBYTECODE=1"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_cli_runtime()
    if args.command == "_run":
        return supervise(Path(args.runtime_root))
    if args.command == "_canary-child":
        return _canary_child(args.hold_seconds, args.exit_code, args.spawn_descendant)
    if args.command == "_child-bootstrap":
        argv = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
        return _child_bootstrap(args.gate_fd, argv)
    if args.command == "status":
        print(json.dumps(_status_payload(Path(args.runtime_root)), sort_keys=True))
        return 0
    if args.command == "stop":
        print(
            json.dumps(
                stop_supervisor(Path(args.runtime_root), timeout=args.timeout), sort_keys=True
            )
        )
        return 0
    canary = (
        None
        if args.command == "start"
        else {
            "hold_seconds": float(args.hold_seconds),
            "exit_code": args.exit_code,
            "spawn_descendant": args.spawn_descendant,
        }
    )
    deployment = Path(args.deployment) if args.command == "start" else None
    spec = _new_spec(
        runtime_root=Path(args.runtime_root),
        deployment=deployment,
        validate_only=bool(getattr(args, "validate_only", False)),
        canary=canary,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
        term_grace=args.term_grace,
        kill_grace=args.kill_grace,
    )
    print(json.dumps(start_detached(spec), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SupervisorError, secure.SecureRuntimeError) as exc:
        print(f"discovery-supervisor: {exc}", file=sys.stderr)
        raise SystemExit(2)
