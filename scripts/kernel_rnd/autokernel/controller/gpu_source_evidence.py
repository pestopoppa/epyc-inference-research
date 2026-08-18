#!/usr/bin/env python3
"""Governed evidence producer for two-arm GPU source discovery.

The module owns evidence capture only.  It never chooses a command, shells out
by itself, or turns its receipts into promotion authority.  Callers inject the
correctness and rocprof executors; this producer binds their exact argv to
exclusive MI210 claims, file-backed output, in-window residency, source/build
identities, and deterministic reductions of the timestamp CSVs.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from .. import schemas
from ..execution import t0_provider
from ..resource import device_claim
from . import gpu_source_proofs as proofs
from . import split_runtime_verifier

AUTHORITY = "nonpromotable_candidate_only_discovery"
BORROWED_PHASE_SCHEMA = "epyc.autokernel.borrowed_device_claim_phase.v1"
CORRECTNESS_SCHEMA = "epyc.autokernel.targeted_correctness_receipt.v3"
CORRECTNESS_REFUSAL_SCHEMA = "epyc.autokernel.targeted_correctness_refusal.v1"
CORRECTNESS_PARSER_ID = "ak.t0.backend_ops_console/v1"
EXECUTION_POLICY_SCHEMA = "epyc.autokernel.gpu_source_execution_policy.v2"
ATTRIBUTION_SCHEMA = "epyc.autokernel.gpu_kernel_attribution.v2"
PAIR_SCHEMA = "epyc.autokernel.gpu_kernel_attribution_pair.v1"
SEALED_BUNDLE_SCHEMA = "epyc.autokernel.gpu_source_evidence_bundle.v1"
SHA = re.compile(r"^[0-9a-f]{64}$")
SOURCE_TREE_SCHEMA = "epyc.autokernel.source_tree_identity.v1"
ROCPROF_TIMESTAMP_OUTPUT = "{TIMESTAMP_CSV}"


class EvidenceProducerError(RuntimeError):
    """The producer refused to mint a success receipt."""


class CorrectnessParseRefusal(EvidenceProducerError):
    """The authoritative backend-op parser could not prove the targeted run."""


def _durable_refusal_reason(exc: BaseException) -> str:
    """ASCII-stable error text for the legacy proof receipt hash reducer."""
    return str(exc).encode("ascii", "backslashreplace").decode("ascii")


def _hash_file(path: Path, label: str, *, allow_empty: bool = True) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceProducerError(f"{label} must be a regular non-symlink file")
    if not allow_empty and path.stat().st_size == 0:
        raise EvidenceProducerError(f"{label} must not be empty")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise EvidenceProducerError(f"refusing to overwrite evidence artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _seal(path: Path, body: Mapping[str, Any]) -> Mapping[str, Any]:
    value = dict(body)
    value["receipt_sha256"] = schemas.content_hash(value)
    _atomic(path, value)
    return proofs.load_receipt(path, schema=str(value["schema"]))


def _hash(value: str, label: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise EvidenceProducerError(f"{label} must be a SHA-256 digest")
    return value


def _argv(value: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item or "\0" in item
                         for item in result):
        raise EvidenceProducerError(f"{label} must be exact non-empty argv")
    if any(re.search(r"[;|&`\n\r]|\$\(", item) for item in result):
        raise EvidenceProducerError(f"{label} contains forbidden shell metacharacters")
    return result


@dataclass(frozen=True)
class GpuResidencySample:
    observed_monotonic_ns: int
    device_id: str
    kfd_pids: tuple[int, ...]
    vram_bytes: int
    launcher_pid: int | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.observed_monotonic_ns, int)
                or self.observed_monotonic_ns < 0
                or not isinstance(self.device_id, str) or not self.device_id
                or not self.kfd_pids
                or any(isinstance(pid, bool) or not isinstance(pid, int) or pid < 1
                       for pid in self.kfd_pids)
                or (self.launcher_pid is not None and (isinstance(self.launcher_pid, bool)
                    or not isinstance(self.launcher_pid, int) or self.launcher_pid < 1))
                or isinstance(self.vram_bytes, bool)
                or not isinstance(self.vram_bytes, int) or self.vram_bytes < 0):
            raise EvidenceProducerError("invalid KFD/VRAM sample")


@dataclass(frozen=True)
class ExecutionCapture:
    """Executor-owned observation of exactly one child process."""

    argv: tuple[str, ...]
    exit_code: int
    child_pid: int
    started_at: str
    ended_at: str
    started_monotonic_ns: int
    ended_monotonic_ns: int
    samples: tuple[GpuResidencySample, ...]
    runtime_maps_identity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _argv(self.argv, "captured argv")
        if (isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
                or isinstance(self.child_pid, bool) or not isinstance(self.child_pid, int)
                or self.child_pid < 1
                or not isinstance(self.started_at, str) or not self.started_at
                or not isinstance(self.ended_at, str) or not self.ended_at
                or not isinstance(self.started_monotonic_ns, int)
                or not isinstance(self.ended_monotonic_ns, int)
                or self.started_monotonic_ns >= self.ended_monotonic_ns):
            raise EvidenceProducerError("invalid executor capture interval")


@dataclass(frozen=True)
class CommandInvocation:
    kind: str
    arm: str
    argv: tuple[str, ...]
    stdout_path: Path
    stderr_path: Path
    timestamp_csv_path: Path | None = None
    working_directory: Path | None = None
    environment: tuple[tuple[str, str], ...] = ()
    runtime_maps_required: bool = False
    runtime_maps_context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"correctness", "rocprof"}:
            raise EvidenceProducerError("command kind must be correctness or rocprof")
        if self.arm not in {"candidate", "anchor"}:
            raise EvidenceProducerError("command arm must be candidate or anchor")
        _argv(self.argv, "invocation argv")
        paths = (self.stdout_path, self.stderr_path) + (() if self.timestamp_csv_path is None else (self.timestamp_csv_path,))
        if any(not path.is_absolute() for path in paths):
            raise EvidenceProducerError("executor output paths must be absolute")
        if self.kind == "rocprof" and self.timestamp_csv_path is None:
            raise EvidenceProducerError("rocprof invocation requires timestamp CSV output")
        if self.kind == "correctness" and self.timestamp_csv_path is not None:
            raise EvidenceProducerError("correctness invocation may not claim a timestamp CSV")
        if (self.working_directory is None or not self.working_directory.is_absolute()
                or not self.working_directory.is_dir() or self.working_directory.is_symlink()):
            raise EvidenceProducerError("command working directory must be an absolute real directory")
        if (not self.environment
                or any(not isinstance(item, tuple) or len(item) != 2
                       or not all(isinstance(part, str) for part in item)
                       for item in self.environment)
                or len(dict(self.environment)) != len(self.environment)
                or any(not key or "\0" in key or "\0" in value
                       for key, value in self.environment)):
            raise EvidenceProducerError("command environment must be exact unique key/value pairs")
        if "LD_LIBRARY_PATH" not in dict(self.environment):
            raise EvidenceProducerError("command environment must bind LD_LIBRARY_PATH")
        if self.runtime_maps_required and (self.kind != "rocprof"
                                           or not isinstance(self.runtime_maps_context, Mapping)):
            raise EvidenceProducerError("runtime maps are only valid for a bound rocprof arm")


class CommandExecutor(Protocol):
    def __call__(self, invocation: CommandInvocation) -> ExecutionCapture: ...


@dataclass(frozen=True)
class BoundInputFile:
    role: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not self.role or not self.path.is_absolute():
            raise EvidenceProducerError("bound input requires role and absolute path")
        _hash(self.sha256, f"{self.role} SHA-256")


@dataclass(frozen=True)
class BuildIdentityFiles:
    source_identity: BoundInputFile
    binary: BoundInputFile
    hip_library: BoundInputFile
    config: BoundInputFile
    linkage: BoundInputFile


@dataclass(frozen=True)
class SharedRewardRuntimeFiles:
    """File-backed source-patch reward closure.

    Correctness intentionally uses the candidate build's diagnostic binary.
    Throughput/rocprof instead execute this one shared reward binary and select
    exactly one arm through the first loader path.
    """
    measurement_binary: BoundInputFile
    runtime_receipt: BoundInputFile
    anchor_hip_library: BoundInputFile
    candidate_hip_library: BoundInputFile


@dataclass(frozen=True)
class EvidenceIdentityFiles:
    candidate: BuildIdentityFiles
    anchor: BuildIdentityFiles
    manifest: BoundInputFile
    model: BoundInputFile
    workload: BoundInputFile
    runtime_config: BoundInputFile
    materialization: BoundInputFile
    shared_runtime: SharedRewardRuntimeFiles | None = None


RuntimeMapsSampler = Callable[[CommandInvocation, int, GpuResidencySample], Mapping[str, Any]]


class SubprocessCommandExecutor:
    """Direct-spawn executor; sampling and process construction are injectable.

    The producer still verifies the returned capture.  This implementation is
    the production seam that prevents an actor from fabricating a capture.
    Tests inject a fake executor and never spawn a profiler.
    """

    def __init__(self, *, residency_sampler: Callable[[int], GpuResidencySample],
                 runtime_maps_sampler: RuntimeMapsSampler | None = None,
                 sample_interval_s: float = .02,
                 popen: Callable[..., Any] = subprocess.Popen) -> None:
        if sample_interval_s <= 0 or not math.isfinite(sample_interval_s):
            raise EvidenceProducerError("sample interval must be finite and positive")
        self.residency_sampler = residency_sampler
        self.runtime_maps_sampler = runtime_maps_sampler
        self.sample_interval_s = sample_interval_s
        self.popen = popen

    def __call__(self, invocation: CommandInvocation) -> ExecutionCapture:
        started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        started_ns = time.monotonic_ns()
        samples: list[GpuResidencySample] = []
        runtime_maps_identity: Mapping[str, Any] | None = None
        child: Any | None = None
        with invocation.stdout_path.open("x", encoding="utf-8") as stdout, \
                invocation.stderr_path.open("x", encoding="utf-8") as stderr:
            try:
                child = self.popen(
                    list(invocation.argv), stdin=subprocess.DEVNULL, stdout=stdout,
                    stderr=stderr, env=dict(invocation.environment),
                    cwd=str(invocation.working_directory), close_fds=True)
                while child.poll() is None:
                    sample = self.residency_sampler(int(child.pid))
                    if not isinstance(sample, GpuResidencySample):
                        raise EvidenceProducerError(
                            "residency sampler returned an invalid sample")
                    samples.append(sample)
                    if (invocation.runtime_maps_required and runtime_maps_identity is None
                            and sample.vram_bytes > 0
                            and (int(child.pid) in sample.kfd_pids
                                 or sample.launcher_pid == int(child.pid))):
                        if self.runtime_maps_sampler is None:
                            raise EvidenceProducerError(
                                "shared reward invocation has no in-window maps sampler")
                        runtime_maps_identity = self.runtime_maps_sampler(
                            invocation, int(child.pid), sample)
                        if not isinstance(runtime_maps_identity, Mapping):
                            raise EvidenceProducerError("runtime maps sampler returned no typed identity")
                    time.sleep(self.sample_interval_s)
                exit_code = int(child.wait())
            except BaseException:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    except TypeError:
                        child.wait()
                    if child.poll() is None:
                        child.kill()
                        try:
                            child.wait(timeout=10)
                        except TypeError:
                            child.wait()
                    if child.poll() is None:
                        raise EvidenceProducerError(
                            "captured child remained alive after teardown")
                raise
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
        ended_ns = time.monotonic_ns()
        ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return ExecutionCapture(
            argv=invocation.argv, exit_code=exit_code, child_pid=int(child.pid),
            started_at=started_at, ended_at=ended_at,
            started_monotonic_ns=started_ns, ended_monotonic_ns=ended_ns,
            samples=tuple(samples), runtime_maps_identity=runtime_maps_identity)


@dataclass(frozen=True)
class ExactDispatch:
    signature: str
    kernel_pattern: str
    calls: int
    grid: int
    workgroup: int
    lds_bytes: int
    blocks_per_call: int

    def __post_init__(self) -> None:
        if (not self.signature or any(isinstance(value, bool) or not isinstance(value, int)
                                     for value in (self.calls, self.grid, self.workgroup,
                                                   self.lds_bytes, self.blocks_per_call))
                or self.calls < 1 or self.grid < 1 or self.workgroup < 1
                or self.lds_bytes < 0 or self.blocks_per_call < 1):
            raise EvidenceProducerError("invalid exact dispatch expectation")
        try:
            re.compile(self.kernel_pattern)
        except re.error as exc:
            raise EvidenceProducerError("invalid exact dispatch regex") from exc


@dataclass(frozen=True)
class ForbiddenDispatch:
    signature: str
    kernel_pattern: str

    def __post_init__(self) -> None:
        if not self.signature:
            raise EvidenceProducerError("forbidden dispatch signature is empty")
        try:
            re.compile(self.kernel_pattern)
        except re.error as exc:
            raise EvidenceProducerError("invalid forbidden dispatch regex") from exc


@dataclass(frozen=True)
class InvariantDispatch:
    signature: str
    kernel_pattern: str

    def __post_init__(self) -> None:
        if not self.signature:
            raise EvidenceProducerError("invariant signature is empty")
        try:
            re.compile(self.kernel_pattern)
        except re.error as exc:
            raise EvidenceProducerError("invalid invariant dispatch regex") from exc


@dataclass(frozen=True)
class DispatchContract:
    candidate_exact: tuple[ExactDispatch, ...]
    anchor_exact: tuple[ExactDispatch, ...]
    candidate_forbidden: tuple[ForbiddenDispatch, ...] = ()
    anchor_forbidden: tuple[ForbiddenDispatch, ...] = ()
    invariants: tuple[InvariantDispatch, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_exact or not self.anchor_exact:
            raise EvidenceProducerError("both arms require exact dispatch expectations")
        signatures = [item.signature for group in (
            self.candidate_exact, self.anchor_exact, self.candidate_forbidden,
            self.anchor_forbidden, self.invariants) for item in group]
        if len(signatures) != len(set(signatures)):
            raise EvidenceProducerError("dispatch expectation signatures must be globally unique")


@dataclass(frozen=True)
class GpuSourceEvidencePlan:
    campaign_id: str
    device_id: str
    manifest_sha256: str
    model_sha256: str
    workload_sha256: str
    runtime_config_sha256: str
    candidate: proofs.BuildIdentity
    anchor: proofs.BuildIdentity
    correctness_argv: tuple[str, ...]
    correctness_backend: str
    correctness_op: str
    expected_correctness_cases: int
    candidate_rocprof_argv: tuple[str, ...]
    anchor_rocprof_argv: tuple[str, ...]
    dispatch: DispatchContract
    identity_files: EvidenceIdentityFiles
    policy: BoundInputFile
    correctness_inputs: tuple[BoundInputFile, ...] = ()
    candidate_rocprof_inputs: tuple[BoundInputFile, ...] = ()
    anchor_rocprof_inputs: tuple[BoundInputFile, ...] = ()
    required_correctness_argv_paths: tuple[Path, ...] = ()
    required_candidate_rocprof_argv_paths: tuple[Path, ...] = ()
    required_anchor_rocprof_argv_paths: tuple[Path, ...] = ()
    execution_cwd: Path = Path("/")
    correctness_environment: tuple[tuple[str, str], ...] = ()
    candidate_rocprof_environment: tuple[tuple[str, str], ...] = ()
    anchor_rocprof_environment: tuple[tuple[str, str], ...] = ()
    shared_runtime: SharedRewardRuntimeFiles | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.campaign_id, str) or not self.campaign_id
                or not isinstance(self.device_id, str) or not self.device_id):
            raise EvidenceProducerError("campaign and device identities are required")
        for name in ("manifest_sha256", "model_sha256", "workload_sha256",
                     "runtime_config_sha256"):
            _hash(getattr(self, name), name)
        if not isinstance(self.candidate, proofs.BuildIdentity) or not isinstance(self.anchor, proofs.BuildIdentity):
            raise EvidenceProducerError("both build identities must be typed")
        if self.candidate == self.anchor:
            raise EvidenceProducerError("candidate and anchor build identities must differ")
        _argv(self.correctness_argv, "correctness argv")
        _argv(self.candidate_rocprof_argv, "candidate rocprof argv")
        _argv(self.anchor_rocprof_argv, "anchor rocprof argv")
        if (isinstance(self.expected_correctness_cases, bool)
                or not isinstance(self.expected_correctness_cases, int)
                or self.expected_correctness_cases < 1):
            raise EvidenceProducerError("expected correctness count must be positive")
        if (not isinstance(self.correctness_backend, str)
                or not self.correctness_backend
                or not isinstance(self.correctness_op, str)
                or not self.correctness_op):
            raise EvidenceProducerError(
                "correctness backend and operation must be explicit")
        if not isinstance(self.identity_files, EvidenceIdentityFiles):
            raise EvidenceProducerError("plan requires typed file-backed identities")
        if not isinstance(self.policy, BoundInputFile):
            raise EvidenceProducerError("plan requires a sealed adapter policy")
        if (not self.execution_cwd.is_absolute() or not self.execution_cwd.is_dir()
                or self.execution_cwd.is_symlink()
                ):
            raise EvidenceProducerError("plan requires a sealed working directory")
        for label, environment in (
            ("correctness", self.correctness_environment),
            ("candidate rocprof", self.candidate_rocprof_environment),
            ("anchor rocprof", self.anchor_rocprof_environment),
        ):
            if (not environment
                    or any(not isinstance(item, tuple) or len(item) != 2
                           or not all(isinstance(part, str) for part in item)
                           for item in environment)
                    or len(dict(environment)) != len(environment)
                    or "LD_LIBRARY_PATH" not in dict(environment)):
                raise EvidenceProducerError(
                    f"plan requires sealed {label} LD_LIBRARY_PATH environment")
        if self.shared_runtime != self.identity_files.shared_runtime:
            raise EvidenceProducerError("shared reward runtime must be carried by the typed identity files")
        for label, command, inputs in (
            ("correctness", self.correctness_argv, self.correctness_inputs),
            ("candidate rocprof", self.candidate_rocprof_argv,
             self.candidate_rocprof_inputs),
            ("anchor rocprof", self.anchor_rocprof_argv,
             self.anchor_rocprof_inputs),
        ):
            if not Path(command[0]).is_absolute():
                raise EvidenceProducerError(f"{label} executable must be an absolute path")
            if any(not isinstance(item, BoundInputFile) for item in inputs):
                raise EvidenceProducerError(f"{label} inputs must be typed bound files")
            if not any(item.role == "executable" and item.path == Path(command[0])
                       for item in inputs):
                raise EvidenceProducerError(f"{label} executable is not policy-bound")
        for label, command, inputs, binary in (
            ("candidate rocprof", self.candidate_rocprof_argv,
             self.candidate_rocprof_inputs,
             (self.shared_runtime.measurement_binary.path if self.shared_runtime
              else self.identity_files.candidate.binary.path)),
            ("anchor rocprof", self.anchor_rocprof_argv,
             self.anchor_rocprof_inputs,
             (self.shared_runtime.measurement_binary.path if self.shared_runtime
              else self.identity_files.anchor.binary.path)),
        ):
            timestamps = [item for item in inputs if item.role == "timestamp_input"]
            if len(timestamps) != 1:
                raise EvidenceProducerError(f"{label} requires one sealed timestamp input")
            expected_pair = ("-i", str(timestamps[0].path))
            if not any(tuple(command[index:index + 2]) == expected_pair
                       for index in range(len(command) - 1)):
                raise EvidenceProducerError(f"{label} does not bind rocprof -i input")
            if str(binary) not in command:
                raise EvidenceProducerError(f"{label} does not execute its bound target binary")
        for label, command, required in (
            ("correctness", self.correctness_argv,
             self.required_correctness_argv_paths),
            ("candidate rocprof", self.candidate_rocprof_argv,
             self.required_candidate_rocprof_argv_paths),
            ("anchor rocprof", self.anchor_rocprof_argv,
             self.required_anchor_rocprof_argv_paths),
        ):
            if not required or any(not path.is_absolute() or str(path) not in command
                                   for path in required):
                raise EvidenceProducerError(
                    f"{label} does not bind every required model/workload/config path")
        candidate_dir = str((self.shared_runtime.candidate_hip_library.path.parent
                             if self.shared_runtime else
                             self.identity_files.candidate.hip_library.path.parent))
        anchor_dir = str((self.shared_runtime.anchor_hip_library.path.parent
                          if self.shared_runtime else
                          self.identity_files.anchor.hip_library.path.parent))
        correctness_candidate_dir = str(self.identity_files.candidate.hip_library.path.parent)
        for label, environment, required, forbidden in (
            ("correctness", self.correctness_environment, correctness_candidate_dir,
             str(self.identity_files.anchor.hip_library.path.parent)),
            ("candidate rocprof", self.candidate_rocprof_environment,
             candidate_dir, anchor_dir),
            ("anchor rocprof", self.anchor_rocprof_environment,
             anchor_dir, candidate_dir),
        ):
            ld_paths = dict(environment)["LD_LIBRARY_PATH"].split(":")
            if not ld_paths or ld_paths[0] != required or forbidden in ld_paths:
                raise EvidenceProducerError(
                    f"{label} LD_LIBRARY_PATH does not exclusively pin its arm")
        if self.shared_runtime:
            runtime = self.shared_runtime
            if (runtime.candidate_hip_library.sha256 != self.candidate.hip_library_sha256
                    or runtime.anchor_hip_library.sha256 != self.anchor.hip_library_sha256):
                raise EvidenceProducerError("shared reward HIP arms do not bind build identities")
            if _normalized_rocprof_argv(self.candidate_rocprof_argv) != _normalized_rocprof_argv(self.anchor_rocprof_argv):
                raise EvidenceProducerError("source rocprof arms differ beyond their bound timestamp output")
            for environment, hip in ((self.candidate_rocprof_environment,
                                      runtime.candidate_hip_library),
                                     (self.anchor_rocprof_environment,
                                      runtime.anchor_hip_library)):
                ld_paths = dict(environment)["LD_LIBRARY_PATH"].split(":")
                if (len(ld_paths) < 2 or ld_paths[0] != str(hip.path.parent)
                        or ld_paths[1] != str(runtime.measurement_binary.path.parent)):
                    raise EvidenceProducerError("source rocprof arm does not use sealed split reward closure")


def _normalized_rocprof_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize only the producer-owned rocprof ``-o`` output placeholder.

    The plan cannot name a per-operation output directory.  It may use this
    exact token once; the producer substitutes a fresh contained CSV path and
    records the resulting argv.  No other arm-specific argv variation is
    admissible for source-patch reward attribution.
    """
    result = list(argv)
    if "-o" not in result:
        return tuple(result)
    index = result.index("-o")
    if index + 1 >= len(result) or result[index + 1] != ROCPROF_TIMESTAMP_OUTPUT:
        raise EvidenceProducerError("rocprof -o must use the sealed timestamp output token")
    if result.count("-o") != 1 or result.count(ROCPROF_TIMESTAMP_OUTPUT) != 1:
        raise EvidenceProducerError("rocprof command has ambiguous timestamp output authority")
    result[index + 1] = ROCPROF_TIMESTAMP_OUTPUT
    return tuple(result)


def _materialize_rocprof_argv(argv: tuple[str, ...], output: Path) -> tuple[str, ...]:
    if ROCPROF_TIMESTAMP_OUTPUT not in argv:
        return argv
    if not output.is_absolute():
        raise EvidenceProducerError("rocprof output substitution requires an absolute path")
    return tuple(str(output) if item == ROCPROF_TIMESTAMP_OUTPUT else item for item in argv)


def _receipt_rocprof_template(body: Mapping[str, Any]) -> tuple[str, ...]:
    """Recover the sole producer-owned output placeholder from a receipt."""
    try:
        argv = list(body["command_argv"])
        output = str(body["timestamp_csv_path"])
    except (KeyError, TypeError) as exc:
        raise EvidenceProducerError("attribution receipt lacks command/output binding") from exc
    if "-o" not in argv:
        return tuple(argv)
    index = argv.index("-o")
    if (argv.count("-o") != 1 or index + 1 >= len(argv)
            or argv[index + 1] != output):
        raise EvidenceProducerError("attribution receipt has unbound rocprof output argv")
    argv[index + 1] = ROCPROF_TIMESTAMP_OUTPUT
    return tuple(argv)


def _bound_reference(value: BoundInputFile) -> dict[str, Any]:
    return {"role": value.role, "path": str(value.path), "sha256": value.sha256}


def _shared_runtime_reference(value: SharedRewardRuntimeFiles | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {key: _bound_reference(getattr(value, key)) for key in (
        "measurement_binary", "runtime_receipt", "anchor_hip_library",
        "candidate_hip_library")}


def _bound_from_dict(value: Mapping[str, Any]) -> BoundInputFile:
    try:
        return BoundInputFile(role=str(value["role"]), path=Path(value["path"]),
                              sha256=str(value["sha256"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceProducerError("bound input reference is malformed") from exc


def _build_files_reference(value: BuildIdentityFiles) -> dict[str, Any]:
    return {key: _bound_reference(getattr(value, key)) for key in (
        "source_identity", "binary", "hip_library", "config", "linkage")}


def _build_files_from_dict(value: Mapping[str, Any]) -> BuildIdentityFiles:
    try:
        return BuildIdentityFiles(**{
            key: _bound_from_dict(value[key]) for key in (
                "source_identity", "binary", "hip_library", "config", "linkage")})
    except (KeyError, TypeError) as exc:
        raise EvidenceProducerError("build identity files are malformed") from exc


def _identity_files_reference(value: EvidenceIdentityFiles) -> dict[str, Any]:
    result = {
        "candidate": _build_files_reference(value.candidate),
        "anchor": _build_files_reference(value.anchor),
        **{key: _bound_reference(getattr(value, key)) for key in (
            "manifest", "model", "workload", "runtime_config", "materialization")},
    }
    if value.shared_runtime is not None:
        result["shared_runtime"] = {
            key: _bound_reference(getattr(value.shared_runtime, key)) for key in (
                "measurement_binary", "runtime_receipt", "anchor_hip_library",
                "candidate_hip_library")}
    return result


def _identity_files_from_dict(value: Mapping[str, Any]) -> EvidenceIdentityFiles:
    try:
        return EvidenceIdentityFiles(
            candidate=_build_files_from_dict(value["candidate"]),
            anchor=_build_files_from_dict(value["anchor"]),
            manifest=_bound_from_dict(value["manifest"]),
            model=_bound_from_dict(value["model"]),
            workload=_bound_from_dict(value["workload"]),
            runtime_config=_bound_from_dict(value["runtime_config"]),
            materialization=_bound_from_dict(value["materialization"]),
            shared_runtime=(None if value.get("shared_runtime") is None else
                            SharedRewardRuntimeFiles(**{
                                key: _bound_from_dict(value["shared_runtime"][key])
                                for key in ("measurement_binary", "runtime_receipt",
                                            "anchor_hip_library", "candidate_hip_library")})))
    except (KeyError, TypeError) as exc:
        raise EvidenceProducerError("evidence identity files are malformed") from exc


def _verify_bound(value: BoundInputFile) -> None:
    if _hash_file(value.path, value.role, allow_empty=False) != value.sha256:
        raise EvidenceProducerError(f"bound input {value.role} bytes changed")


def _verify_executable(command: Sequence[str], inputs: Sequence[BoundInputFile],
                       label: str) -> None:
    executable = Path(command[0])
    if (not executable.is_file() or executable.is_symlink()
            or not os.access(executable, os.X_OK)):
        raise EvidenceProducerError(f"{label} executable is not a regular executable file")
    if not any(item.role == "executable" and item.path == executable
               for item in inputs):
        raise EvidenceProducerError(f"{label} executable escaped sealed policy")


def _verify_build_files(files: BuildIdentityFiles,
                        identity: proofs.BuildIdentity, arm: str) -> None:
    expected = {
        "binary": identity.binary_sha256,
        "hip_library": identity.hip_library_sha256,
        "config": identity.config_sha256,
        "linkage": identity.linkage_sha256,
    }
    for name, digest in expected.items():
        item = getattr(files, name)
        _verify_bound(item)
        if item.sha256 != digest:
            raise EvidenceProducerError(f"{arm} {name} does not match build identity")
    _verify_source_tree_identity(files.source_identity, identity, arm)


def _verify_source_tree_identity(carrier: BoundInputFile,
                                 identity: proofs.BuildIdentity, arm: str) -> None:
    """Validate the durable carrier for a source *tree* digest.

    A tree digest is intentionally not the digest of this JSON receipt.  The
    bound-input hash protects the carrier bytes; the receipt's complete,
    self-hashed TreeDigest manifest proves the commit/tree identity after its
    worktree has been torn down.
    """
    _verify_bound(carrier)
    try:
        source = json.loads(carrier.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError(f"{arm} source tree identity is not JSON") from exc
    if not isinstance(source, Mapping):
        raise EvidenceProducerError(f"{arm} source tree identity is malformed")
    body = {key: value for key, value in source.items() if key != "receipt_sha256"}
    if source.get("receipt_sha256") != schemas.content_hash(body):
        raise EvidenceProducerError(f"{arm} source tree identity self-hash mismatch")
    if (source.get("schema") != SOURCE_TREE_SCHEMA
            or source.get("source_commit") != identity.source_commit
            or not isinstance(source.get("root_provenance"), str)
            or not Path(source["root_provenance"]).is_absolute()
            or source.get("exclusions") != [".git"]):
        raise EvidenceProducerError(f"{arm} source tree receipt provenance mismatch")
    tree = source.get("tree")
    if not isinstance(tree, Mapping) or tree.get("listing_is_complete") is not True:
        raise EvidenceProducerError(f"{arm} source tree receipt lacks complete listing")
    try:
        claimed_sha = _hash(str(tree["sha256"]), "tree SHA-256")
        file_count = tree["file_count"]
        total_bytes = tree["total_bytes"]
        entries = tree["entries"]
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceProducerError(f"{arm} source tree receipt is malformed") from exc
    if (isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 0
            or isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 0
            or not isinstance(entries, list) or len(entries) != file_count):
        raise EvidenceProducerError(f"{arm} source tree totals/listing disagree")
    normalized: list[tuple[str, str, str]] = []
    for entry in entries:
        if (not isinstance(entry, list) or len(entry) != 3
                or entry[0] not in {"100644", "100755", "120000"}
                or not isinstance(entry[1], str) or not SHA.fullmatch(entry[1])
                or not isinstance(entry[2], str) or not entry[2]
                or Path(entry[2]).is_absolute() or ".." in Path(entry[2]).parts):
            raise EvidenceProducerError(f"{arm} source tree entry is malformed")
        normalized.append((entry[0], entry[1], entry[2]))
    if normalized != sorted(normalized, key=lambda row: row[2]) or len({row[2] for row in normalized}) != len(normalized):
        raise EvidenceProducerError(f"{arm} source tree listing is not canonical")
    manifest_sha = hashlib.sha256(
        "".join(f"{mode}\t{digest}\t{path}\n" for mode, digest, path in normalized)
        .encode("utf-8")).hexdigest()
    if claimed_sha != manifest_sha or claimed_sha != identity.source_sha256:
        raise EvidenceProducerError(f"{arm} source tree SHA is not file-backed")


def _policy_payload(plan: GpuSourceEvidencePlan) -> dict[str, Any]:
    return {
        "schema": EXECUTION_POLICY_SCHEMA,
        "manifest_sha256": plan.manifest_sha256,
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "anchor_build_identity": asdict(plan.anchor),
        "correctness_argv": list(plan.correctness_argv),
        "correctness_parser_id": CORRECTNESS_PARSER_ID,
        "correctness_backend": plan.correctness_backend,
        "correctness_op": plan.correctness_op,
        "expected_correctness_cases": plan.expected_correctness_cases,
        "candidate_rocprof_argv": list(plan.candidate_rocprof_argv),
        "anchor_rocprof_argv": list(plan.anchor_rocprof_argv),
        "correctness_inputs": [_bound_reference(x) for x in plan.correctness_inputs],
        "candidate_rocprof_inputs": [_bound_reference(x) for x in plan.candidate_rocprof_inputs],
        "anchor_rocprof_inputs": [_bound_reference(x) for x in plan.anchor_rocprof_inputs],
        "required_correctness_argv_paths": [str(x) for x in plan.required_correctness_argv_paths],
        "required_candidate_rocprof_argv_paths": [str(x) for x in plan.required_candidate_rocprof_argv_paths],
        "required_anchor_rocprof_argv_paths": [str(x) for x in plan.required_anchor_rocprof_argv_paths],
        "execution_cwd": str(plan.execution_cwd),
        "correctness_environment": [list(item) for item in plan.correctness_environment],
        "candidate_rocprof_environment": [list(item) for item in plan.candidate_rocprof_environment],
        "anchor_rocprof_environment": [list(item) for item in plan.anchor_rocprof_environment],
        "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
        "dispatch": _expectations(plan),
    }


def _verify_plan_files(plan: GpuSourceEvidencePlan) -> None:
    _verify_build_files(plan.identity_files.candidate, plan.candidate, "candidate")
    _verify_build_files(plan.identity_files.anchor, plan.anchor, "anchor")
    for item, digest, label in (
        (plan.identity_files.manifest, plan.manifest_sha256, "manifest"),
        (plan.identity_files.model, plan.model_sha256, "model"),
        (plan.identity_files.workload, plan.workload_sha256, "workload"),
        (plan.identity_files.runtime_config, plan.runtime_config_sha256,
         "runtime config"),
    ):
        _verify_bound(item)
        if item.sha256 != digest:
            raise EvidenceProducerError(f"{label} file does not match declared identity")
    _verify_bound(plan.identity_files.materialization)
    try:
        materialization = json.loads(
            plan.identity_files.materialization.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError("source materialization receipt is not JSON") from exc
    # Older governed fixtures predate the canonical builder receipt.  Only the
    # new concrete-builder schema is admitted without a self hash fallback.
    if materialization.get("schema") == "epyc.autokernel.gpu_source_materialization.v1":
        if (materialization.get("receipt_sha256") != schemas.content_hash(
                {key: value for key, value in materialization.items() if key != "receipt_sha256"})):
            raise EvidenceProducerError("source materialization receipt self-hash mismatch")
    required_materialization = {
        "schema": "epyc.autokernel.gpu_source_materialization.v1",
        "manifest_sha256": plan.manifest_sha256,
        "candidate_source_commit": plan.candidate.source_commit,
        "candidate_source_sha256": plan.candidate.source_sha256,
        "patch_applied": True,
        "production_tree": False,
    }
    if any(materialization.get(key) != value
           for key, value in required_materialization.items()):
        raise EvidenceProducerError(
            "source materialization does not prove manifest-applied experimental tree")
    if plan.shared_runtime is not None:
        _verify_shared_runtime(plan.shared_runtime, plan=plan)
    for item in (plan.correctness_inputs + plan.candidate_rocprof_inputs
                 + plan.anchor_rocprof_inputs):
        _verify_bound(item)
    _verify_executable(plan.correctness_argv, plan.correctness_inputs, "correctness")
    _verify_executable(plan.candidate_rocprof_argv,
                       plan.candidate_rocprof_inputs, "candidate rocprof")
    _verify_executable(plan.anchor_rocprof_argv,
                       plan.anchor_rocprof_inputs, "anchor rocprof")
    _verify_bound(plan.policy)
    try:
        policy = json.loads(plan.policy.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError("sealed adapter policy is not JSON") from exc
    if policy != _policy_payload(plan):
        raise EvidenceProducerError("sealed adapter policy differs from execution plan")


def _verify_shared_runtime(runtime: SharedRewardRuntimeFiles,
                           *, plan: GpuSourceEvidencePlan) -> None:
    for item in (runtime.measurement_binary, runtime.runtime_receipt,
                 runtime.anchor_hip_library, runtime.candidate_hip_library):
        _verify_bound(item)
    try:
        body = json.loads(runtime.runtime_receipt.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError("shared reward runtime receipt is not JSON") from exc
    if not isinstance(body, Mapping):
        raise EvidenceProducerError("shared reward runtime receipt is malformed")
    payload = {key: value for key, value in body.items() if key != "receipt_sha256"}
    if body.get("receipt_sha256") != schemas.content_hash(payload):
        raise EvidenceProducerError("shared reward runtime receipt self-hash mismatch")
    expected = {
        "schema": "epyc.autokernel.shared_reward_runtime.v1",
        "authority": AUTHORITY,
        "promotion_claim": False,
        "measurement_binary_sha256": runtime.measurement_binary.sha256,
        "anchor_hip_sha256": runtime.anchor_hip_library.sha256,
        "candidate_hip_sha256": runtime.candidate_hip_library.sha256,
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError("shared reward runtime receipt identity mismatch")
    split = body.get("split_runtime_manifest")
    if not isinstance(split, Mapping) or split.get("schema") != split_runtime_verifier.SCHEMA:
        raise EvidenceProducerError("shared reward runtime lacks a sealed split-runtime manifest")
    try:
        root = Path(str(split["root"])).resolve(strict=True)
        verified = split_runtime_verifier.verify_split_runtime(root)
    except (KeyError, OSError, split_runtime_verifier.SplitRuntimeError) as exc:
        raise EvidenceProducerError("shared reward split runtime cannot be revalidated") from exc
    if split != verified.to_dict():
        raise EvidenceProducerError("shared reward split runtime changed after sealing")
    expected_paths = {
        "measurement": verified.reward_binary.resolve(strict=True),
        "anchor": (verified.anchor_hip_dir / "libggml-hip.so.0").resolve(strict=True),
        "candidate": (verified.candidate_hip_dir / "libggml-hip.so.0").resolve(strict=True),
    }
    actual_paths = {
        "measurement": runtime.measurement_binary.path.resolve(strict=True),
        "anchor": runtime.anchor_hip_library.path.resolve(strict=True),
        "candidate": runtime.candidate_hip_library.path.resolve(strict=True),
    }
    if actual_paths != expected_paths:
        raise EvidenceProducerError("shared reward carriers do not select verified runtime objects")
    if (runtime.anchor_hip_library.sha256 != plan.anchor.hip_library_sha256
            or runtime.candidate_hip_library.sha256 != plan.candidate.hip_library_sha256):
        raise EvidenceProducerError("shared reward runtime does not match arm build HIP identities")


def _receipt_dict(value: object, label: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()  # type: ignore[union-attr]
    if not isinstance(value, Mapping):
        raise EvidenceProducerError(f"{label} is not a claim receipt")
    return dict(value)


def _check_result_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    passed = getattr(value, "passed", None)
    if isinstance(passed, bool):
        return passed
    return (getattr(value, "outcome", None) == schemas.PASS
            or getattr(value, "status", None) == schemas.PASS)


def _default_claim_verifier(receipt: Mapping[str, Any]) -> bool:
    return _check_result_passed(device_claim.check_device_claim_held(receipt))


def _validate_claim_pair(opened: Mapping[str, Any], released: Mapping[str, Any],
                         *, plan: GpuSourceEvidencePlan) -> None:
    try:
        opened_typed = device_claim.ClaimReceipt.from_dict(opened)
    except (TypeError, ValueError) as exc:
        raise EvidenceProducerError("opened claim does not satisfy the device-claim schema") from exc
    if released.get("schema") == BORROWED_PHASE_SCHEMA:
        expected = {
            "schema": BORROWED_PHASE_SCHEMA,
            "mode": "borrowed_outer_reservation",
            "outer_claim_id": opened_typed.claim_id,
            "device_id": opened_typed.device_id,
            "campaign_id": opened_typed.campaign_id,
            "physical_release": False,
        }
        if (any(released.get(key) != value for key, value in expected.items())
                or not isinstance(released.get("phase_ended_at"), str)
                or not released["phase_ended_at"]
                or "released_at" in released):
            raise EvidenceProducerError("borrowed claim phase end is malformed")
        if opened_typed.device_id != plan.device_id or opened_typed.campaign_id != plan.campaign_id:
            raise EvidenceProducerError("borrowed claim does not bind the planned device/campaign")
        return
    try:
        released_typed = device_claim.ClaimReceipt.from_dict(released)
    except (TypeError, ValueError) as exc:
        raise EvidenceProducerError("released claim does not satisfy the device-claim schema") from exc
    if opened_typed.released_at is not None or not released_typed.released_at:
        raise EvidenceProducerError("claim release is missing or contradictory")
    comparable = ("claim_id", "device_id", "lock_path", "state", "holder_pid",
                  "holder_start_ticks", "holder_boot_id", "host", "holder_label",
                  "purpose", "campaign_id", "acquired_at", "expires_at", "reclaimed_from")
    if any(getattr(opened_typed, key) != getattr(released_typed, key) for key in comparable):
        raise EvidenceProducerError("open/release claim identities differ")
    if opened_typed.device_id != plan.device_id or opened_typed.campaign_id != plan.campaign_id:
        raise EvidenceProducerError("claim does not bind the planned device/campaign")


def _claim_boundary_fields(opened: Mapping[str, Any], ended: Mapping[str, Any],
                           residency: Mapping[str, Any]) -> dict[str, Any]:
    borrowed = residency.get("device_claim_mode") == "borrowed_outer_reservation"
    return {
        "device_claim_open": dict(opened),
        "device_claim_mode": ("borrowed_outer_reservation" if borrowed
                              else "direct_device_claim"),
        **({"device_claim_borrowed_phase_end": dict(ended)} if borrowed
           else {"device_claim_released": dict(ended)}),
    }


def _validate_claim_boundary(body: Mapping[str, Any], *, plan: GpuSourceEvidencePlan) -> None:
    mode = body.get("device_claim_mode")
    borrowed = body.get("device_claim_borrowed_phase_end")
    released = body.get("device_claim_released")
    if mode == "borrowed_outer_reservation":
        if not isinstance(borrowed, Mapping) or released is not None:
            raise EvidenceProducerError("borrowed phase cannot assert a physical claim release")
        ended = borrowed
    elif mode in (None, "direct_device_claim"):
        if not isinstance(released, Mapping) or borrowed is not None:
            raise EvidenceProducerError("direct claim lacks its physical release")
        ended = released
    else:
        raise EvidenceProducerError("device claim mode is unknown")
    _validate_claim_pair(body.get("device_claim_open", {}), ended, plan=plan)


def _validate_residency_witness(value: object, *, device_id: str, label: str) -> None:
    if not isinstance(value, Mapping):
        raise EvidenceProducerError(f"{label} lacks an in-window residency witness")
    try:
        child_pid = int(value["child_pid"])
        started = int(value["execution_started_monotonic_ns"])
        ended = int(value["execution_ended_monotonic_ns"])
        samples = value["samples"]
        claimed_count = int(value["overlap_sample_count"])
        claimed_max = int(value["max_vram_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceProducerError(f"{label} residency witness is malformed") from exc
    if (value.get("overlapped") is not True
            or value.get("claim_verified_before") is not True
            or value.get("claim_verified_after") is not True
            or child_pid < 1 or started >= ended
            or not isinstance(samples, list) or not samples):
        raise EvidenceProducerError(f"{label} lacks in-window claim/KFD/VRAM evidence")
    valid: list[Mapping[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise EvidenceProducerError(f"{label} residency sample is malformed")
        try:
            observed = int(sample["observed_monotonic_ns"])
            pids = sample["kfd_pids"]
            vram = int(sample["vram_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceProducerError(f"{label} residency sample is malformed") from exc
        if (sample.get("device_id") != device_id or not started <= observed <= ended
                or not isinstance(pids, list) or not pids or vram <= 0
                or (child_pid not in pids and sample.get("launcher_pid") != child_pid)):
            raise EvidenceProducerError(f"{label} residency sample missed child lifetime")
        valid.append(sample)
    if (claimed_count != len(valid) or claimed_max != max(int(x["vram_bytes"])
                                                          for x in valid)):
        raise EvidenceProducerError(f"{label} residency reduction mismatch")


def _residency(capture: ExecutionCapture, device_id: str) -> dict[str, Any]:
    samples = [sample for sample in capture.samples
               if capture.started_monotonic_ns <= sample.observed_monotonic_ns
               <= capture.ended_monotonic_ns and sample.device_id == device_id
               and (capture.child_pid in sample.kfd_pids
                    or sample.launcher_pid == capture.child_pid)
               and sample.vram_bytes > 0]
    if not samples:
        raise EvidenceProducerError("no KFD+nonzero-VRAM sample overlapped the child execution")
    return {
        "overlapped": True,
        "child_pid": capture.child_pid,
        "execution_started_monotonic_ns": capture.started_monotonic_ns,
        "execution_ended_monotonic_ns": capture.ended_monotonic_ns,
        "overlap_sample_count": len(samples),
        "kfd_pids": sorted({pid for sample in samples for pid in sample.kfd_pids}),
        "max_vram_bytes": max(sample.vram_bytes for sample in samples),
        "samples": [{
            **asdict(sample),
            "kfd_pids": list(sample.kfd_pids),
        } for sample in samples],
    }


def _run_claimed(
    invocation: CommandInvocation, *, plan: GpuSourceEvidencePlan,
    executor: CommandExecutor, claim_acquirer: Callable[..., Any],
    claim_verifier: Callable[[Mapping[str, Any]], object], claim_journal: Any,
    claim_timeout_s: float,
) -> tuple[ExecutionCapture, dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Re-hash every executable/input and the sealed policy immediately before
    # each claim.  A validated plan is not a lease on mutable bytes.
    _verify_plan_files(plan)
    invocation.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [invocation.stdout_path, invocation.stderr_path]
    if invocation.timestamp_csv_path is not None:
        output_paths.append(invocation.timestamp_csv_path)
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise EvidenceProducerError("executor outputs must be fresh paths")
    claim = None
    borrowed_outer = False
    opened: dict[str, Any] | None = None
    capture: ExecutionCapture | None = None
    failure: BaseException | None = None
    released: dict[str, Any] | None = None
    verified_before = verified_after = False
    try:
        claim = claim_acquirer(
            plan.device_id,
            purpose=f"AutoKernel GPU source evidence {invocation.kind}/{invocation.arm}",
            campaign_id=plan.campaign_id,
            journal=claim_journal,
            holder_label="gpu_source_evidence.py",
            timeout_s=claim_timeout_s,
            max_hold_s=3600.0,
        )
        borrowed_outer = bool(getattr(claim, "borrowed_outer_reservation", False))
        opened = _receipt_dict(claim.receipt(), "opened claim")
        verified_before = _check_result_passed(claim_verifier(opened))
        if not verified_before:
            raise EvidenceProducerError("device claim was not verifiably held before execution")
        capture = executor(invocation)
        if not isinstance(capture, ExecutionCapture) or capture.argv != invocation.argv:
            raise EvidenceProducerError("executor did not attest the exact planned argv")
        verified_after = _check_result_passed(claim_verifier(opened))
        if not verified_after:
            raise EvidenceProducerError("device claim was not verifiably held after execution")
    except BaseException as exc:
        failure = exc
    try:
        if claim is not None:
            released = _receipt_dict(claim.release(), "released claim")
    except BaseException as exc:
        failure = failure or exc
    if failure is not None:
        if isinstance(failure, EvidenceProducerError):
            raise failure
        raise EvidenceProducerError(f"{invocation.kind}/{invocation.arm} execution failed: {failure}") from failure
    if capture is None or opened is None or released is None:
        raise EvidenceProducerError("claimed execution did not produce complete evidence")
    _validate_claim_pair(opened, released, plan=plan)
    residency = _residency(capture, plan.device_id)
    residency["claim_verified_before"] = verified_before
    residency["claim_verified_after"] = verified_after
    residency["device_claim_mode"] = (
        "borrowed_outer_reservation" if borrowed_outer else "direct_device_claim")
    residency["outer_claim_id"] = opened["claim_id"] if borrowed_outer else None
    return capture, opened, released, residency


def _output_hashes(invocation: CommandInvocation) -> dict[str, Any]:
    result = {
        "stdout_path": str(invocation.stdout_path),
        "stdout_sha256": _hash_file(invocation.stdout_path, "stdout", allow_empty=False),
        "stderr_path": str(invocation.stderr_path),
        "stderr_sha256": _hash_file(invocation.stderr_path, "stderr"),
    }
    if invocation.timestamp_csv_path is not None:
        result.update({
            "timestamp_csv_path": str(invocation.timestamp_csv_path),
            "timestamp_csv_sha256": _hash_file(
                invocation.timestamp_csv_path, "timestamp CSV", allow_empty=False),
        })
    return result


@dataclass(frozen=True)
class _CorrectnessResult:
    backend: str
    operation: str
    passed_cases: int
    total_cases: int
    skipped_backends: tuple[str, ...]
    backends_passed: int
    backends_total: int
    overall: str

    @property
    def summary(self) -> str:
        return f"{self.passed_cases}/{self.total_cases} tests passed"


def _parse_correctness(stdout: str, plan: GpuSourceEvidencePlan) -> _CorrectnessResult:
    """Reduce console bytes through the T0 parser, then enforce this exact plan.

    The tool counts skipped devices as passed backends.  Therefore neither its
    final ``N/N backends passed`` line nor a regex spanning that line proves the
    selected GPU ran anything.  The authoritative T0 parser attributes cases to
    concrete backend frames, excludes unsupported cases, and reconciles its
    parse with the tool's per-backend count before this reducer accepts it.
    """
    try:
        run = t0_provider.parse_backend_ops_console(stdout)
        run.reconcile()
    except t0_provider.OutputParseError as exc:
        raise CorrectnessParseRefusal(
            f"correctness console parse refused: {exc}") from exc

    targets = tuple(row for row in run.backends
                    if row.name == plan.correctness_backend)
    if len(targets) != 1:
        raise CorrectnessParseRefusal(
            "correctness output must contain exactly one target backend frame")
    target = targets[0]
    if target.skipped:
        raise CorrectnessParseRefusal("target correctness backend was skipped")
    if target.status != "OK":
        raise CorrectnessParseRefusal("target correctness backend did not report OK")

    compared = tuple(case for case in target.cases
                     if case.status != "not_supported")
    if not compared:
        raise CorrectnessParseRefusal(
            "target correctness backend exercised zero supported cases")
    if any(case.op != plan.correctness_op for case in compared):
        raise CorrectnessParseRefusal(
            "target correctness backend exercised an unexpected operation")
    if any(not case.passed for case in compared):
        raise CorrectnessParseRefusal(
            "target correctness backend contains a failed case")
    expected = plan.expected_correctness_cases
    if (len(compared) != expected
            or target.reported_passed != expected
            or target.reported_total != expected):
        raise CorrectnessParseRefusal(
            "correctness did not pass the exact expected case count")

    others = tuple(row for row in run.backends if row is not target)
    if any(not row.skipped or row.cases for row in others):
        raise CorrectnessParseRefusal(
            "a non-target backend was exercised by targeted correctness")
    if (run.backends_total != len(run.backends)
            or run.backends_passed != run.backends_total
            or run.overall != "OK"
            or run.failing_tests):
        raise CorrectnessParseRefusal(
            "correctness backend/overall summaries do not prove a clean run")
    return _CorrectnessResult(
        backend=plan.correctness_backend,
        operation=plan.correctness_op,
        passed_cases=expected,
        total_cases=expected,
        skipped_backends=tuple(row.name for row in others),
        backends_passed=run.backends_passed,
        backends_total=run.backends_total,
        overall=run.overall)


def _produce_correctness(
    root: Path, plan: GpuSourceEvidencePlan, executor: CommandExecutor, *,
    claim_acquirer: Callable[..., Any], claim_verifier: Callable[[Mapping[str, Any]], object],
    claim_journal: Any, claim_timeout_s: float,
) -> Mapping[str, Any]:
    directory = root / "correctness"
    invocation = CommandInvocation(
        kind="correctness", arm="candidate", argv=plan.correctness_argv,
        stdout_path=(directory / "stdout.txt").resolve(),
        stderr_path=(directory / "stderr.txt").resolve(),
        working_directory=plan.execution_cwd,
        environment=plan.correctness_environment)
    capture, opened, released, residency = _run_claimed(
        invocation, plan=plan, executor=executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=claim_timeout_s)
    outputs = _output_hashes(invocation)
    try:
        parsed = _parse_correctness(
            invocation.stdout_path.read_text(encoding="utf-8"), plan)
    except CorrectnessParseRefusal as exc:
        _seal(directory / "refusal.json", {
            "schema": CORRECTNESS_REFUSAL_SCHEMA,
            "authority": AUTHORITY,
            "promotion_claim": False,
            "status": "refused",
            "classification": "output_parse_refusal",
            "error_type": type(exc).__name__,
            "reason": _durable_refusal_reason(exc),
            "campaign_id": plan.campaign_id,
            "device_id": plan.device_id,
            "manifest_sha256": plan.manifest_sha256,
            "candidate_build_identity": asdict(plan.candidate),
            "workload_sha256": plan.workload_sha256,
            "command_argv": list(plan.correctness_argv),
            "command_cwd": str(plan.execution_cwd),
            "command_environment_sha256": schemas.content_hash(
                [list(item) for item in plan.correctness_environment]),
            "exit_code": capture.exit_code,
            **outputs,
            "started_at": capture.started_at,
            "ended_at": capture.ended_at,
            "correctness_parser_id": CORRECTNESS_PARSER_ID,
            "correctness_backend": plan.correctness_backend,
            "correctness_op": plan.correctness_op,
            "expected_cases": plan.expected_correctness_cases,
            **_claim_boundary_fields(opened, released, residency),
            "residency_witness": residency,
        })
        raise
    if capture.exit_code != 0:
        raise EvidenceProducerError("targeted correctness command exited nonzero")
    body = {
        "schema": CORRECTNESS_SCHEMA,
        "authority": AUTHORITY,
        "non_promotable": True,
        "promotion_claim": False,
        "status": "complete",
        "result": "PASS",
        "campaign_id": plan.campaign_id,
        "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "identity_files": _identity_files_reference(plan.identity_files),
        "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in plan.correctness_inputs],
        "workload_sha256": plan.workload_sha256,
        "command_argv": list(plan.correctness_argv),
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in plan.correctness_environment]),
        "exit_code": capture.exit_code,
        **outputs,
        "started_at": capture.started_at,
        "ended_at": capture.ended_at,
        "summary": parsed.summary,
        "correctness_parser_id": CORRECTNESS_PARSER_ID,
        "correctness_backend": parsed.backend,
        "correctness_op": parsed.operation,
        "skipped_backends": list(parsed.skipped_backends),
        "backends_passed": parsed.backends_passed,
        "backends_total": parsed.backends_total,
        "overall": parsed.overall,
        "expected_cases": plan.expected_correctness_cases,
        "passed_cases": plan.expected_correctness_cases,
        "exact_case_ok": True,
        **_claim_boundary_fields(opened, released, residency),
        "residency_witness": residency,
    }
    return _seal(directory / "receipt.json", body)


def _integer(row: Mapping[str, str], key: str, *, minimum: int) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceProducerError(f"timestamp CSV has invalid {key}") from exc
    if value < minimum:
        raise EvidenceProducerError(f"timestamp CSV {key} is below {minimum}")
    return value


def _load_dispatches(path: Path) -> list[dict[str, Any]]:
    _hash_file(path, "timestamp CSV", allow_empty=False)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"KernelName", "grd", "wgr", "lds", "BeginNs", "EndNs"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise EvidenceProducerError("timestamp CSV lacks required rocprof-v1 columns")
        dispatches: list[dict[str, Any]] = []
        for index, row in enumerate(reader):
            if None in row:
                raise EvidenceProducerError("timestamp CSV row has surplus columns")
            kernel = row.get("KernelName")
            if not isinstance(kernel, str) or not kernel:
                raise EvidenceProducerError("timestamp CSV kernel is empty")
            grid = _integer(row, "grd", minimum=1)
            workgroup = _integer(row, "wgr", minimum=1)
            lds = _integer(row, "lds", minimum=0)
            begin = _integer(row, "BeginNs", minimum=0)
            end = _integer(row, "EndNs", minimum=1)
            if end <= begin or grid % workgroup:
                raise EvidenceProducerError("timestamp row has invalid duration or non-integral blocks")
            dispatches.append({
                "index": index, "kernel": kernel, "grid": grid,
                "workgroup": workgroup, "lds": lds,
                "blocks_per_call": grid // workgroup,
                "begin_ns": begin, "end_ns": end,
            })
    if not dispatches:
        raise EvidenceProducerError("timestamp CSV contains no dispatches")
    return dispatches


def _matching(rows: Sequence[Mapping[str, Any]], pattern: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if re.search(pattern, str(row["kernel"]))]


def _geometry_signature(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[tuple[int, int, int, int], int] = {}
    for row in rows:
        key = (int(row["grid"]), int(row["workgroup"]), int(row["lds"]),
               int(row["blocks_per_call"]))
        counts[key] = counts.get(key, 0) + 1
    return {
        "calls": len(rows),
        "geometries": [
            {"grid": key[0], "workgroup": key[1], "lds_bytes": key[2],
             "blocks_per_call": key[3], "calls": count}
            for key, count in sorted(counts.items())
        ],
    }


def _reduce_arm(
    rows: Sequence[Mapping[str, Any]], *, exact: Sequence[ExactDispatch],
    forbidden: Sequence[ForbiddenDispatch], invariants: Sequence[InvariantDispatch],
) -> dict[str, Any]:
    exact_result: dict[str, Any] = {}
    for expectation in exact:
        # One real kernel symbol can launch at several governed geometries in
        # the same workload (quantize_q8_1 is the canonical case).  Select the
        # geometry as well as the escaped name so each expected cell remains
        # exact and independently countable.
        hits = [row for row in _matching(rows, expectation.kernel_pattern)
                if (int(row["grid"]), int(row["workgroup"]), int(row["lds"]),
                    int(row["blocks_per_call"])) ==
                   (expectation.grid, expectation.workgroup,
                    expectation.lds_bytes, expectation.blocks_per_call)]
        geometry = _geometry_signature(hits)
        expected_geometry = [{
            "grid": expectation.grid, "workgroup": expectation.workgroup,
            "lds_bytes": expectation.lds_bytes,
            "blocks_per_call": expectation.blocks_per_call,
            "calls": expectation.calls,
        }]
        if geometry != {"calls": expectation.calls, "geometries": expected_geometry}:
            raise EvidenceProducerError(
                f"exact dispatch {expectation.signature} count/geometry mismatch")
        exact_result[expectation.signature] = geometry
    for pattern in {item.kernel_pattern for item in exact}:
        allowed = {(item.grid, item.workgroup, item.lds_bytes, item.blocks_per_call)
                   for item in exact if item.kernel_pattern == pattern}
        unexpected = [row for row in _matching(rows, pattern)
                      if (int(row["grid"]), int(row["workgroup"]), int(row["lds"]),
                          int(row["blocks_per_call"])) not in allowed]
        if unexpected:
            raise EvidenceProducerError(
                "exact dispatch matched an unreviewed geometry")
    forbidden_result: dict[str, int] = {}
    for expectation in forbidden:
        count = len(_matching(rows, expectation.kernel_pattern))
        forbidden_result[expectation.signature] = count
        if count:
            raise EvidenceProducerError(
                f"forbidden dispatch {expectation.signature} remains ({count} calls)")
    invariant_result: dict[str, Any] = {}
    for expectation in invariants:
        hits = _matching(rows, expectation.kernel_pattern)
        if not hits:
            raise EvidenceProducerError(
                f"invariant dispatch {expectation.signature} has no calls")
        invariant_result[expectation.signature] = _geometry_signature(hits)
    return {"exact": exact_result, "forbidden": forbidden_result,
            "invariants": invariant_result}


def _produce_attribution_arm(
    root: Path, arm: str, plan: GpuSourceEvidencePlan,
    executor: CommandExecutor, *, claim_acquirer: Callable[..., Any],
    claim_verifier: Callable[[Mapping[str, Any]], object], claim_journal: Any,
    claim_timeout_s: float,
) -> Mapping[str, Any]:
    directory = root / f"attribution-{arm}"
    argv_template = (plan.candidate_rocprof_argv if arm == "candidate"
                     else plan.anchor_rocprof_argv)
    identity = plan.candidate if arm == "candidate" else plan.anchor
    exact = plan.dispatch.candidate_exact if arm == "candidate" else plan.dispatch.anchor_exact
    forbidden = (plan.dispatch.candidate_forbidden if arm == "candidate"
                 else plan.dispatch.anchor_forbidden)
    inputs = (plan.candidate_rocprof_inputs if arm == "candidate"
              else plan.anchor_rocprof_inputs)
    output_csv = (directory / "timestamps.csv").resolve()
    argv = _materialize_rocprof_argv(argv_template, output_csv)
    invocation = CommandInvocation(
        kind="rocprof", arm=arm, argv=argv,
        stdout_path=(directory / "stdout.txt").resolve(),
        stderr_path=(directory / "stderr.txt").resolve(),
        timestamp_csv_path=output_csv,
        working_directory=plan.execution_cwd,
        environment=(plan.candidate_rocprof_environment if arm == "candidate"
                     else plan.anchor_rocprof_environment),
        runtime_maps_required=plan.shared_runtime is not None,
        runtime_maps_context=(None if plan.shared_runtime is None else {
            "arm": arm, "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
            "model": _bound_reference(plan.identity_files.model),
            "model_sha256": plan.model_sha256, "device_id": plan.device_id}))
    capture, opened, released, residency = _run_claimed(
        invocation, plan=plan, executor=executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=claim_timeout_s)
    outputs = _output_hashes(invocation)
    if capture.exit_code != 0:
        raise EvidenceProducerError(f"{arm} rocprof command exited nonzero")
    assert invocation.timestamp_csv_path is not None
    dispatches = _load_dispatches(invocation.timestamp_csv_path)
    reduction = _reduce_arm(
        dispatches, exact=exact, forbidden=forbidden,
        invariants=plan.dispatch.invariants)
    runtime_maps = _validated_runtime_maps_identity(
        capture, plan=plan, arm=arm, residency=residency)
    body = {
        "schema": ATTRIBUTION_SCHEMA,
        "authority": AUTHORITY,
        "non_promotable": True,
        "promotion_claim": False,
        "status": "complete",
        "result": "PASS",
        "arm": arm,
        "campaign_id": plan.campaign_id,
        "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "build_identity": asdict(identity),
        "identity_files": _identity_files_reference(plan.identity_files),
        "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in inputs],
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "command_argv": list(argv),
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in (
                plan.candidate_rocprof_environment if arm == "candidate"
                else plan.anchor_rocprof_environment)]),
        "exit_code": capture.exit_code,
        **outputs,
        "timestamp_reduction_sha256": schemas.content_hash(dispatches),
        "started_at": capture.started_at,
        "ended_at": capture.ended_at,
        "dispatches": dispatches,
        "exact_dispatch_signatures": reduction["exact"],
        "forbidden_dispatch_signatures": reduction["forbidden"],
        "invariant_signatures": reduction["invariants"],
        **_claim_boundary_fields(opened, released, residency),
        "residency_witness": residency,
        "runtime_maps_identity": runtime_maps,
    }
    return _seal(directory / "receipt.json", body)


def _reference(loaded: Mapping[str, Any]) -> dict[str, Any]:
    return {key: loaded[key] for key in ("path", "file_sha256", "native_sha256", "body")}


def _expectations(plan: GpuSourceEvidencePlan) -> dict[str, Any]:
    return {
        "candidate_exact": [asdict(item) for item in plan.dispatch.candidate_exact],
        "anchor_exact": [asdict(item) for item in plan.dispatch.anchor_exact],
        "candidate_forbidden": [asdict(item) for item in plan.dispatch.candidate_forbidden],
        "anchor_forbidden": [asdict(item) for item in plan.dispatch.anchor_forbidden],
        "invariants": [asdict(item) for item in plan.dispatch.invariants],
    }


def _produce_pair(
    root: Path, plan: GpuSourceEvidencePlan, candidate: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> Mapping[str, Any]:
    candidate_body, anchor_body = candidate["body"], anchor["body"]
    if candidate_body["invariant_signatures"] != anchor_body["invariant_signatures"]:
        raise EvidenceProducerError("candidate changed an invariant hot signature")
    if plan.shared_runtime is not None:
        candidate_maps = candidate_body.get("runtime_maps_identity")
        anchor_maps = anchor_body.get("runtime_maps_identity")
        if not isinstance(candidate_maps, Mapping) or not isinstance(anchor_maps, Mapping):
            raise EvidenceProducerError("shared reward attribution lacks in-window loader-map identities")
        if candidate_maps.get("runtime_manifest_sha256") != anchor_maps.get("runtime_manifest_sha256"):
            raise EvidenceProducerError("source arms did not map one shared runtime closure")
    body = {
        "schema": PAIR_SCHEMA,
        "authority": AUTHORITY,
        "non_promotable": True,
        "promotion_claim": False,
        "manifest_sha256": plan.manifest_sha256,
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "anchor_build_identity": asdict(plan.anchor),
        "identity_files": _identity_files_reference(plan.identity_files),
        "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
        "execution_policy": _bound_reference(plan.policy),
        "correctness_inputs": [_bound_reference(x) for x in plan.correctness_inputs],
        "candidate_rocprof_inputs": [_bound_reference(x) for x in plan.candidate_rocprof_inputs],
        "anchor_rocprof_inputs": [_bound_reference(x) for x in plan.anchor_rocprof_inputs],
        "required_correctness_argv_paths": [str(x) for x in plan.required_correctness_argv_paths],
        "required_candidate_rocprof_argv_paths": [str(x) for x in plan.required_candidate_rocprof_argv_paths],
        "required_anchor_rocprof_argv_paths": [str(x) for x in plan.required_anchor_rocprof_argv_paths],
        "execution_cwd": str(plan.execution_cwd),
        "correctness_environment": [list(item) for item in plan.correctness_environment],
        "candidate_rocprof_environment": [list(item) for item in plan.candidate_rocprof_environment],
        "anchor_rocprof_environment": [list(item) for item in plan.anchor_rocprof_environment],
        "expectations": _expectations(plan),
        "candidate": _reference(candidate),
        "anchor": _reference(anchor),
        "invariant_signatures": candidate_body["invariant_signatures"],
        "inverse_attribution_proved": True,
        "candidate_runtime_maps_identity": candidate_body.get("runtime_maps_identity"),
        "anchor_runtime_maps_identity": anchor_body.get("runtime_maps_identity"),
    }
    return _seal(root / "attribution-pair.json", body)


def _reload_reference(reference: Mapping[str, Any], *, schema: str) -> Mapping[str, Any]:
    if not isinstance(reference, Mapping):
        raise EvidenceProducerError("bundle reference is not an object")
    loaded = proofs.load_receipt(Path(str(reference.get("path", ""))), schema=schema)
    for key in ("file_sha256", "native_sha256", "body"):
        if loaded[key] != reference.get(key):
            raise EvidenceProducerError("referenced receipt changed after sealing")
    return loaded


def _validate_attribution_body(body: Mapping[str, Any], *, plan: GpuSourceEvidencePlan,
                               arm: str) -> None:
    identity = plan.candidate if arm == "candidate" else plan.anchor
    argv_template = (plan.candidate_rocprof_argv if arm == "candidate"
                     else plan.anchor_rocprof_argv)
    try:
        argv = _materialize_rocprof_argv(argv_template,
                                         Path(str(body["timestamp_csv_path"])))
    except (KeyError, TypeError) as exc:
        raise EvidenceProducerError(f"{arm} attribution receipt lacks timestamp output") from exc
    expected = {
        "schema": ATTRIBUTION_SCHEMA, "authority": AUTHORITY,
        "non_promotable": True, "promotion_claim": False,
        "status": "complete", "result": "PASS", "arm": arm,
        "campaign_id": plan.campaign_id, "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "build_identity": asdict(identity), "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "command_argv": list(argv), "exit_code": 0,
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in (
                plan.candidate_rocprof_environment if arm == "candidate"
                else plan.anchor_rocprof_environment)]),
        "identity_files": _identity_files_reference(plan.identity_files),
        "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in (
            plan.candidate_rocprof_inputs if arm == "candidate"
            else plan.anchor_rocprof_inputs)],
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError(f"{arm} attribution receipt identity/config mismatch")
    _validate_claim_boundary(body, plan=plan)
    for kind in ("stdout", "stderr", "timestamp_csv"):
        path = Path(str(body.get(f"{kind}_path", "")))
        if _hash_file(path, kind, allow_empty=kind != "timestamp_csv") != body.get(f"{kind}_sha256"):
            raise EvidenceProducerError(f"{arm} {kind} bytes changed")
    rows = _load_dispatches(Path(str(body["timestamp_csv_path"])))
    if rows != body.get("dispatches") or schemas.content_hash(rows) != body.get("timestamp_reduction_sha256"):
        raise EvidenceProducerError(f"{arm} timestamp reduction changed")
    exact = plan.dispatch.candidate_exact if arm == "candidate" else plan.dispatch.anchor_exact
    forbidden = (plan.dispatch.candidate_forbidden if arm == "candidate"
                 else plan.dispatch.anchor_forbidden)
    reduction = _reduce_arm(rows, exact=exact, forbidden=forbidden,
                            invariants=plan.dispatch.invariants)
    if (body.get("exact_dispatch_signatures") != reduction["exact"]
            or body.get("forbidden_dispatch_signatures") != reduction["forbidden"]
            or body.get("invariant_signatures") != reduction["invariants"]):
        raise EvidenceProducerError(f"{arm} dispatch derivation mismatch")
    _validate_residency_witness(
        body.get("residency_witness"), device_id=plan.device_id, label=arm)
    if plan.shared_runtime is not None:
        _validate_runtime_maps_receipt(body.get("runtime_maps_identity"), plan=plan, arm=arm,
                                       residency=body.get("residency_witness"))


def _validated_runtime_maps_identity(capture: ExecutionCapture, *, plan: GpuSourceEvidencePlan,
                                     arm: str, residency: Mapping[str, Any]) -> dict[str, Any] | None:
    if plan.shared_runtime is None:
        return None
    return _validate_runtime_maps_receipt(capture.runtime_maps_identity, plan=plan,
                                          arm=arm, residency=residency)


def _validate_runtime_maps_receipt(value: object, *, plan: GpuSourceEvidencePlan,
                                   arm: str, residency: object) -> dict[str, Any]:
    """Bind a verifier-produced in-window /proc/maps identity to an arm receipt."""
    if not isinstance(value, Mapping) or not isinstance(residency, Mapping):
        raise EvidenceProducerError(f"{arm} source attribution lacks a runtime maps identity")
    try:
        typed = split_runtime_verifier.HotResidencyIdentity(
            runtime_manifest_sha256=str(value["runtime_manifest_sha256"]),
            arm=str(value["arm"]), reward_binary_sha256=str(value["reward_binary_sha256"]),
            hip_library_sha256=str(value["hip_library_sha256"]),
            model_path=Path(str(value["model_path"])), model_sha256=str(value["model_sha256"]),
            device_id=str(value["device_id"]), kfd_pid=int(value["kfd_pid"]),
            boot_id=str(value["boot_id"]), process_start_ticks=int(value["process_start_ticks"]),
            mapped_local_sha256=dict(value["mapped_local_sha256"]),
            identity_sha256=str(value["identity_sha256"]))
    except (KeyError, TypeError, ValueError, split_runtime_verifier.SplitRuntimeError) as exc:
        raise EvidenceProducerError(f"{arm} runtime maps identity is malformed") from exc
    runtime = plan.shared_runtime
    assert runtime is not None
    try:
        runtime_body = json.loads(runtime.runtime_receipt.path.read_text(encoding="utf-8"))
        manifest_sha = str(runtime_body["split_runtime_manifest"]["manifest_sha256"])
        kfd_pids = {int(pid) for pid in residency["kfd_pids"]}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError(f"{arm} runtime maps context is malformed") from exc
    expected_hip = (runtime.candidate_hip_library.sha256 if arm == "candidate"
                    else runtime.anchor_hip_library.sha256)
    if (typed.runtime_manifest_sha256 != manifest_sha or typed.arm != arm
            or typed.reward_binary_sha256 != runtime.measurement_binary.sha256
            or typed.hip_library_sha256 != expected_hip
            or typed.model_path != plan.identity_files.model.path
            or typed.model_sha256 != plan.model_sha256
            or typed.device_id != plan.device_id or typed.kfd_pid not in kfd_pids):
        raise EvidenceProducerError(f"{arm} runtime maps identity does not bind the sealed arm/run")
    return typed.to_dict()


def _validate_correctness_body(body: Mapping[str, Any], plan: GpuSourceEvidencePlan) -> None:
    expected = {
        "schema": CORRECTNESS_SCHEMA, "authority": AUTHORITY,
        "non_promotable": True, "promotion_claim": False,
        "status": "complete", "result": "PASS", "campaign_id": plan.campaign_id,
        "device_id": plan.device_id, "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "workload_sha256": plan.workload_sha256,
        "command_argv": list(plan.correctness_argv), "exit_code": 0,
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in plan.correctness_environment]),
        "identity_files": _identity_files_reference(plan.identity_files),
        "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in plan.correctness_inputs],
        "correctness_parser_id": CORRECTNESS_PARSER_ID,
        "correctness_backend": plan.correctness_backend,
        "correctness_op": plan.correctness_op,
        "expected_cases": plan.expected_correctness_cases,
        "passed_cases": plan.expected_correctness_cases, "exact_case_ok": True,
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError("correctness receipt identity/config/result mismatch")
    _validate_claim_boundary(body, plan=plan)
    for kind in ("stdout", "stderr"):
        path = Path(str(body.get(f"{kind}_path", "")))
        if _hash_file(path, kind, allow_empty=kind == "stderr") != body.get(f"{kind}_sha256"):
            raise EvidenceProducerError(f"correctness {kind} bytes changed")
    parsed = _parse_correctness(
        Path(str(body["stdout_path"])).read_text(encoding="utf-8"), plan)
    if (parsed.summary != body.get("summary")
            or list(parsed.skipped_backends) != body.get("skipped_backends")
            or parsed.backends_passed != body.get("backends_passed")
            or parsed.backends_total != body.get("backends_total")
            or parsed.overall != body.get("overall")):
        raise EvidenceProducerError("correctness summary changed")
    _validate_residency_witness(
        body.get("residency_witness"), device_id=plan.device_id,
        label="correctness")


def load_gpu_source_correctness_receipt(
        path: Path, plan: GpuSourceEvidencePlan) -> Mapping[str, Any]:
    """Re-open one completed correctness phase without requiring later phases.

    This is the durable phase boundary: a crash or refusal in attribution may
    not turn a completed GPU correctness command back into an in-memory fact.
    The loader recursively rechecks the plan's immutable inputs, receipt hash,
    raw stdout/stderr bytes, typed backend-op parse, device claim and in-window
    residency before returning the receipt.
    """
    _verify_plan_files(plan)
    try:
        loaded = proofs.load_receipt(path, schema=CORRECTNESS_SCHEMA)
    except proofs.ProofError as exc:
        raise EvidenceProducerError(
            "completed correctness receipt is not durably recoverable") from exc
    _validate_correctness_body(loaded["body"], plan)
    return loaded


def load_gpu_source_correctness_refusal(
        path: Path, plan: GpuSourceEvidencePlan) -> Mapping[str, Any]:
    """Validate a durable typed refusal without converting it into a pass."""
    _verify_plan_files(plan)
    try:
        loaded = proofs.load_receipt(path, schema=CORRECTNESS_REFUSAL_SCHEMA)
    except proofs.ProofError as exc:
        raise EvidenceProducerError(
            "correctness parse refusal is not durably recoverable") from exc
    body = loaded["body"]
    expected = {
        "authority": AUTHORITY,
        "promotion_claim": False,
        "status": "refused",
        "classification": "output_parse_refusal",
        "error_type": "CorrectnessParseRefusal",
        "campaign_id": plan.campaign_id,
        "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "workload_sha256": plan.workload_sha256,
        "command_argv": list(plan.correctness_argv),
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in plan.correctness_environment]),
        "correctness_parser_id": CORRECTNESS_PARSER_ID,
        "correctness_backend": plan.correctness_backend,
        "correctness_op": plan.correctness_op,
        "expected_cases": plan.expected_correctness_cases,
    }
    if (any(body.get(key) != value for key, value in expected.items())
            or not isinstance(body.get("reason"), str)
            or not body["reason"]):
        raise EvidenceProducerError(
            "correctness parse refusal identity/classification mismatch")
    _validate_claim_boundary(body, plan=plan)
    for kind in ("stdout", "stderr"):
        path_ = Path(str(body.get(f"{kind}_path", "")))
        if _hash_file(path_, kind, allow_empty=kind == "stderr") != body.get(
                f"{kind}_sha256"):
            raise EvidenceProducerError(
                f"correctness refusal {kind} bytes changed")
    try:
        _parse_correctness(
            Path(str(body["stdout_path"])).read_text(encoding="utf-8"), plan)
    except CorrectnessParseRefusal as exc:
        if _durable_refusal_reason(exc) != body["reason"]:
            raise EvidenceProducerError(
                "correctness parse refusal reason changed") from exc
    else:
        raise EvidenceProducerError(
            "correctness parse refusal now parses as a pass")
    _validate_residency_witness(
        body.get("residency_witness"), device_id=plan.device_id,
        label="correctness refusal")
    return loaded


def _contract_from_dict(value: Mapping[str, Any]) -> DispatchContract:
    try:
        return DispatchContract(
            candidate_exact=tuple(ExactDispatch(**row) for row in value["candidate_exact"]),
            anchor_exact=tuple(ExactDispatch(**row) for row in value["anchor_exact"]),
            candidate_forbidden=tuple(ForbiddenDispatch(**row) for row in value["candidate_forbidden"]),
            anchor_forbidden=tuple(ForbiddenDispatch(**row) for row in value["anchor_forbidden"]),
            invariants=tuple(InvariantDispatch(**row) for row in value["invariants"]),
        )
    except (KeyError, TypeError) as exc:
        raise EvidenceProducerError("sealed dispatch contract is malformed") from exc


def _plan_from_receipts(correctness: Mapping[str, Any], pair: Mapping[str, Any]) -> GpuSourceEvidencePlan:
    correct_body, pair_body = correctness["body"], pair["body"]
    candidate_ref = _reload_reference(pair_body.get("candidate", {}), schema=ATTRIBUTION_SCHEMA)
    anchor_ref = _reload_reference(pair_body.get("anchor", {}), schema=ATTRIBUTION_SCHEMA)
    candidate_body, anchor_body = candidate_ref["body"], anchor_ref["body"]
    try:
        plan = GpuSourceEvidencePlan(
            campaign_id=str(correct_body["campaign_id"]),
            device_id=str(correct_body["device_id"]),
            manifest_sha256=str(pair_body["manifest_sha256"]),
            model_sha256=str(pair_body["model_sha256"]),
            workload_sha256=str(pair_body["workload_sha256"]),
            runtime_config_sha256=str(pair_body["runtime_config_sha256"]),
            candidate=proofs.BuildIdentity(**pair_body["candidate_build_identity"]),
            anchor=proofs.BuildIdentity(**pair_body["anchor_build_identity"]),
            correctness_argv=tuple(correct_body["command_argv"]),
            correctness_backend=str(correct_body["correctness_backend"]),
            correctness_op=str(correct_body["correctness_op"]),
            expected_correctness_cases=int(correct_body["expected_cases"]),
            candidate_rocprof_argv=_receipt_rocprof_template(candidate_body),
            anchor_rocprof_argv=_receipt_rocprof_template(anchor_body),
            dispatch=_contract_from_dict(pair_body["expectations"]),
            identity_files=_identity_files_from_dict(pair_body["identity_files"]),
            shared_runtime=_identity_files_from_dict(pair_body["identity_files"]).shared_runtime,
            policy=_bound_from_dict(pair_body["execution_policy"]),
            correctness_inputs=tuple(_bound_from_dict(x) for x in pair_body["correctness_inputs"]),
            candidate_rocprof_inputs=tuple(_bound_from_dict(x) for x in pair_body["candidate_rocprof_inputs"]),
            anchor_rocprof_inputs=tuple(_bound_from_dict(x) for x in pair_body["anchor_rocprof_inputs"]),
            required_correctness_argv_paths=tuple(Path(x) for x in pair_body["required_correctness_argv_paths"]),
            required_candidate_rocprof_argv_paths=tuple(Path(x) for x in pair_body["required_candidate_rocprof_argv_paths"]),
            required_anchor_rocprof_argv_paths=tuple(Path(x) for x in pair_body["required_anchor_rocprof_argv_paths"]),
            execution_cwd=Path(pair_body["execution_cwd"]),
            correctness_environment=tuple(tuple(x) for x in pair_body["correctness_environment"]),
            candidate_rocprof_environment=tuple(tuple(x) for x in pair_body["candidate_rocprof_environment"]),
            anchor_rocprof_environment=tuple(tuple(x) for x in pair_body["anchor_rocprof_environment"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceProducerError("sealed bundle cannot reconstruct its plan") from exc
    return plan


def produce_gpu_source_evidence(
    *, output_root: Path, plan: GpuSourceEvidencePlan,
    correctness_executor: CommandExecutor, rocprof_executor: CommandExecutor,
    claim_journal: Any, claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    claim_verifier: Callable[[Mapping[str, Any]], object] = _default_claim_verifier,
    claim_timeout_s: float = 300.0,
) -> proofs.GpuSourceProofBundle:
    """Execute correctness, candidate attribution, and anchor inverse attribution.

    The root must be fresh.  Every command owns an independently acquired and
    released device claim.  A failure leaves raw file-backed diagnostics but
    never produces a success bundle.
    """
    root = output_root.resolve()
    if root.exists() or output_root.is_symlink():
        raise EvidenceProducerError("output_root must be a fresh path")
    if (isinstance(claim_timeout_s, bool) or not isinstance(claim_timeout_s, (int, float))
            or not math.isfinite(claim_timeout_s) or claim_timeout_s < 0):
        raise EvidenceProducerError("claim timeout must be finite and non-negative")
    _verify_plan_files(plan)
    root.mkdir(parents=True)
    correctness = _produce_correctness(
        root, plan, correctness_executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=float(claim_timeout_s))
    candidate = _produce_attribution_arm(
        root, "candidate", plan, rocprof_executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=float(claim_timeout_s))
    anchor = _produce_attribution_arm(
        root, "anchor", plan, rocprof_executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=float(claim_timeout_s))
    pair = _produce_pair(root, plan, candidate, anchor)
    bundle = proofs.GpuSourceProofBundle.from_validated_paths(
        manifest_sha256=plan.manifest_sha256, candidate=plan.candidate,
        anchor=plan.anchor, workload_sha256=plan.workload_sha256,
        correctness=_reference(correctness), attribution=_reference(pair))
    _seal(root / "proof-bundle.json", {
        "schema": SEALED_BUNDLE_SCHEMA,
        "authority": AUTHORITY,
        "promotion_claim": False,
        "bundle": bundle.to_dict(),
    })
    # Re-read the complete graph once before returning it to the controller.
    return load_gpu_source_evidence_bundle(root / "proof-bundle.json")


def load_gpu_source_evidence_bundle(path: Path) -> proofs.GpuSourceProofBundle:
    """Re-open a sealed bundle and every file/receipt it cites."""
    wrapper = proofs.load_receipt(path, schema=SEALED_BUNDLE_SCHEMA)["body"]
    if (wrapper.get("authority") != AUTHORITY
            or wrapper.get("promotion_claim") is not False
            or not isinstance(wrapper.get("bundle"), Mapping)):
        raise EvidenceProducerError("sealed bundle crossed the discovery authority boundary")
    raw = wrapper["bundle"]
    if (raw.get("schema") != "epyc.autokernel.gpu_source_proof_bundle.v1"
            or raw.get("authority") != AUTHORITY
            or raw.get("promotion_claim") is not False):
        raise EvidenceProducerError("inner GPU source bundle schema/authority mismatch")
    correctness = _reload_reference(raw.get("correctness", {}), schema=CORRECTNESS_SCHEMA)
    pair = _reload_reference(raw.get("attribution", {}), schema=PAIR_SCHEMA)
    plan = _plan_from_receipts(correctness, pair)
    _verify_plan_files(plan)
    if (plan.manifest_sha256 != raw.get("manifest_sha256")
            or plan.workload_sha256 != raw.get("workload_sha256")
            or asdict(plan.candidate) != raw.get("candidate")
            or asdict(plan.anchor) != raw.get("anchor")):
        raise EvidenceProducerError("bundle top-level identities differ from receipts")
    _validate_correctness_body(correctness["body"], plan)
    pair_body = pair["body"]
    candidate = _reload_reference(pair_body["candidate"], schema=ATTRIBUTION_SCHEMA)
    anchor = _reload_reference(pair_body["anchor"], schema=ATTRIBUTION_SCHEMA)
    _validate_attribution_body(candidate["body"], plan=plan, arm="candidate")
    _validate_attribution_body(anchor["body"], plan=plan, arm="anchor")
    if (candidate["body"]["invariant_signatures"]
            != anchor["body"]["invariant_signatures"]
            or pair_body.get("invariant_signatures")
            != candidate["body"]["invariant_signatures"]
            or pair_body.get("inverse_attribution_proved") is not True):
        raise EvidenceProducerError("inverse attribution or invariant signature mismatch")
    bundle = proofs.GpuSourceProofBundle(
        manifest_sha256=str(raw["manifest_sha256"]), candidate=plan.candidate,
        anchor=plan.anchor, workload_sha256=str(raw["workload_sha256"]),
        correctness=_reference(correctness), attribution=_reference(pair),
        bundle_sha256=str(raw["bundle_sha256"]))
    if bundle.to_dict() != raw:
        raise EvidenceProducerError("inner bundle bytes are not the typed bundle contract")
    return bundle


__all__ = [
    "AUTHORITY", "EvidenceProducerError", "CorrectnessParseRefusal",
    "CORRECTNESS_PARSER_ID", "GpuResidencySample",
    "ExecutionCapture", "CommandInvocation", "CommandExecutor",
    "ExactDispatch", "ForbiddenDispatch", "InvariantDispatch",
    "DispatchContract", "GpuSourceEvidencePlan", "produce_gpu_source_evidence",
    "load_gpu_source_correctness_receipt", "load_gpu_source_correctness_refusal",
    "load_gpu_source_evidence_bundle",
]
