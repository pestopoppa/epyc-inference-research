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
from ..resource import device_claim
from . import gpu_source_proofs as proofs

AUTHORITY = "nonpromotable_candidate_only_discovery"
CORRECTNESS_SCHEMA = "epyc.autokernel.targeted_correctness_receipt.v2"
ATTRIBUTION_SCHEMA = "epyc.autokernel.gpu_kernel_attribution.v2"
PAIR_SCHEMA = "epyc.autokernel.gpu_kernel_attribution_pair.v1"
SEALED_BUNDLE_SCHEMA = "epyc.autokernel.gpu_source_evidence_bundle.v1"
SHA = re.compile(r"^[0-9a-f]{64}$")


class EvidenceProducerError(RuntimeError):
    """The producer refused to mint a success receipt."""


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

    def __post_init__(self) -> None:
        if (not isinstance(self.observed_monotonic_ns, int)
                or self.observed_monotonic_ns < 0
                or not isinstance(self.device_id, str) or not self.device_id
                or not self.kfd_pids
                or any(isinstance(pid, bool) or not isinstance(pid, int) or pid < 1
                       for pid in self.kfd_pids)
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
class EvidenceIdentityFiles:
    candidate: BuildIdentityFiles
    anchor: BuildIdentityFiles
    manifest: BoundInputFile
    model: BoundInputFile
    workload: BoundInputFile
    runtime_config: BoundInputFile


class SubprocessCommandExecutor:
    """Direct-spawn executor; sampling and process construction are injectable.

    The producer still verifies the returned capture.  This implementation is
    the production seam that prevents an actor from fabricating a capture.
    Tests inject a fake executor and never spawn a profiler.
    """

    def __init__(self, *, residency_sampler: Callable[[int], GpuResidencySample],
                 environment: Mapping[str, str], sample_interval_s: float = .02,
                 popen: Callable[..., Any] = subprocess.Popen) -> None:
        if sample_interval_s <= 0 or not math.isfinite(sample_interval_s):
            raise EvidenceProducerError("sample interval must be finite and positive")
        self.residency_sampler = residency_sampler
        self.environment = dict(environment)
        self.sample_interval_s = sample_interval_s
        self.popen = popen

    def __call__(self, invocation: CommandInvocation) -> ExecutionCapture:
        started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        started_ns = time.monotonic_ns()
        samples: list[GpuResidencySample] = []
        with invocation.stdout_path.open("x", encoding="utf-8") as stdout, \
                invocation.stderr_path.open("x", encoding="utf-8") as stderr:
            child = self.popen(
                list(invocation.argv), stdin=subprocess.DEVNULL, stdout=stdout,
                stderr=stderr, env=self.environment, close_fds=True)
            while child.poll() is None:
                sample = self.residency_sampler(int(child.pid))
                if not isinstance(sample, GpuResidencySample):
                    child.terminate()
                    child.wait()
                    raise EvidenceProducerError("residency sampler returned an invalid sample")
                samples.append(sample)
                time.sleep(self.sample_interval_s)
            exit_code = int(child.wait())
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
        ended_ns = time.monotonic_ns()
        ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return ExecutionCapture(
            argv=invocation.argv, exit_code=exit_code, child_pid=int(child.pid),
            started_at=started_at, ended_at=ended_at,
            started_monotonic_ns=started_ns, ended_monotonic_ns=ended_ns,
            samples=tuple(samples))


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
    correctness_summary_pattern: str
    expected_correctness_cases: int
    candidate_rocprof_argv: tuple[str, ...]
    anchor_rocprof_argv: tuple[str, ...]
    dispatch: DispatchContract
    identity_files: EvidenceIdentityFiles
    policy: BoundInputFile
    correctness_inputs: tuple[BoundInputFile, ...] = ()
    candidate_rocprof_inputs: tuple[BoundInputFile, ...] = ()
    anchor_rocprof_inputs: tuple[BoundInputFile, ...] = ()

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
        try:
            compiled = re.compile(self.correctness_summary_pattern)
        except re.error as exc:
            raise EvidenceProducerError("invalid correctness summary regex") from exc
        if not {"passed", "total"}.issubset(compiled.groupindex):
            raise EvidenceProducerError("correctness regex requires named passed and total groups")
        if not isinstance(self.identity_files, EvidenceIdentityFiles):
            raise EvidenceProducerError("plan requires typed file-backed identities")
        if not isinstance(self.policy, BoundInputFile):
            raise EvidenceProducerError("plan requires a sealed adapter policy")
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
             self.candidate_rocprof_inputs, self.identity_files.candidate.binary.path),
            ("anchor rocprof", self.anchor_rocprof_argv,
             self.anchor_rocprof_inputs, self.identity_files.anchor.binary.path),
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


def _bound_reference(value: BoundInputFile) -> dict[str, Any]:
    return {"role": value.role, "path": str(value.path), "sha256": value.sha256}


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
    return {
        "candidate": _build_files_reference(value.candidate),
        "anchor": _build_files_reference(value.anchor),
        **{key: _bound_reference(getattr(value, key)) for key in (
            "manifest", "model", "workload", "runtime_config")},
    }


def _identity_files_from_dict(value: Mapping[str, Any]) -> EvidenceIdentityFiles:
    try:
        return EvidenceIdentityFiles(
            candidate=_build_files_from_dict(value["candidate"]),
            anchor=_build_files_from_dict(value["anchor"]),
            manifest=_bound_from_dict(value["manifest"]),
            model=_bound_from_dict(value["model"]),
            workload=_bound_from_dict(value["workload"]),
            runtime_config=_bound_from_dict(value["runtime_config"]))
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
        "source_identity": identity.source_sha256,
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
    try:
        source = json.loads(files.source_identity.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError(f"{arm} source identity is not JSON") from exc
    if not isinstance(source, Mapping) or source.get("source_commit") != identity.source_commit:
        raise EvidenceProducerError(f"{arm} source commit is not file-backed")


def _policy_payload(plan: GpuSourceEvidencePlan) -> dict[str, Any]:
    return {
        "schema": "epyc.autokernel.gpu_source_execution_policy.v1",
        "manifest_sha256": plan.manifest_sha256,
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "anchor_build_identity": asdict(plan.anchor),
        "correctness_argv": list(plan.correctness_argv),
        "correctness_summary_pattern": plan.correctness_summary_pattern,
        "expected_correctness_cases": plan.expected_correctness_cases,
        "candidate_rocprof_argv": list(plan.candidate_rocprof_argv),
        "anchor_rocprof_argv": list(plan.anchor_rocprof_argv),
        "correctness_inputs": [_bound_reference(x) for x in plan.correctness_inputs],
        "candidate_rocprof_inputs": [_bound_reference(x) for x in plan.candidate_rocprof_inputs],
        "anchor_rocprof_inputs": [_bound_reference(x) for x in plan.anchor_rocprof_inputs],
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


def _receipt_dict(value: object, label: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()  # type: ignore[union-attr]
    if not isinstance(value, Mapping):
        raise EvidenceProducerError(f"{label} is not a claim receipt")
    return dict(value)


def _check_result_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return getattr(value, "status", None) == schemas.PASS


def _default_claim_verifier(receipt: Mapping[str, Any]) -> bool:
    return _check_result_passed(device_claim.check_device_claim_held(receipt))


def _validate_claim_pair(opened: Mapping[str, Any], released: Mapping[str, Any],
                         *, plan: GpuSourceEvidencePlan) -> None:
    try:
        opened_typed = device_claim.ClaimReceipt.from_dict(opened)
        released_typed = device_claim.ClaimReceipt.from_dict(released)
    except (TypeError, ValueError) as exc:
        raise EvidenceProducerError("claim receipts do not satisfy the device-claim schema") from exc
    if opened_typed.released_at is not None or not released_typed.released_at:
        raise EvidenceProducerError("claim release is missing or contradictory")
    comparable = ("claim_id", "device_id", "lock_path", "state", "holder_pid",
                  "holder_start_ticks", "holder_boot_id", "host", "holder_label",
                  "purpose", "campaign_id", "acquired_at", "expires_at", "reclaimed_from")
    if any(getattr(opened_typed, key) != getattr(released_typed, key) for key in comparable):
        raise EvidenceProducerError("open/release claim identities differ")
    if opened_typed.device_id != plan.device_id or opened_typed.campaign_id != plan.campaign_id:
        raise EvidenceProducerError("claim does not bind the planned device/campaign")


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
                or not isinstance(pids, list) or child_pid not in pids or vram <= 0):
            raise EvidenceProducerError(f"{label} residency sample missed child lifetime")
        valid.append(sample)
    if (claimed_count != len(valid) or claimed_max != max(int(x["vram_bytes"])
                                                          for x in valid)):
        raise EvidenceProducerError(f"{label} residency reduction mismatch")


def _residency(capture: ExecutionCapture, device_id: str) -> dict[str, Any]:
    samples = [sample for sample in capture.samples
               if capture.started_monotonic_ns <= sample.observed_monotonic_ns
               <= capture.ended_monotonic_ns and sample.device_id == device_id
               and capture.child_pid in sample.kfd_pids and sample.vram_bytes > 0]
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


def _parse_summary(stdout: str, plan: GpuSourceEvidencePlan) -> str:
    matches = list(re.finditer(plan.correctness_summary_pattern, stdout, re.MULTILINE))
    if len(matches) != 1:
        raise EvidenceProducerError("correctness stdout must contain exactly one summary")
    match = matches[0]
    try:
        passed, total = int(match.group("passed")), int(match.group("total"))
    except (ValueError, IndexError) as exc:
        raise EvidenceProducerError("correctness summary counts are invalid") from exc
    if passed != plan.expected_correctness_cases or total != plan.expected_correctness_cases:
        raise EvidenceProducerError("correctness did not pass the exact expected case count")
    return match.group(0)


def _produce_correctness(
    root: Path, plan: GpuSourceEvidencePlan, executor: CommandExecutor, *,
    claim_acquirer: Callable[..., Any], claim_verifier: Callable[[Mapping[str, Any]], object],
    claim_journal: Any, claim_timeout_s: float,
) -> Mapping[str, Any]:
    directory = root / "correctness"
    invocation = CommandInvocation(
        kind="correctness", arm="candidate", argv=plan.correctness_argv,
        stdout_path=(directory / "stdout.txt").resolve(),
        stderr_path=(directory / "stderr.txt").resolve())
    capture, opened, released, residency = _run_claimed(
        invocation, plan=plan, executor=executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=claim_timeout_s)
    outputs = _output_hashes(invocation)
    summary = _parse_summary(invocation.stdout_path.read_text(encoding="utf-8"), plan)
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
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in plan.correctness_inputs],
        "workload_sha256": plan.workload_sha256,
        "command_argv": list(plan.correctness_argv),
        "exit_code": capture.exit_code,
        **outputs,
        "started_at": capture.started_at,
        "ended_at": capture.ended_at,
        "summary": summary,
        "correctness_summary_pattern": plan.correctness_summary_pattern,
        "expected_cases": plan.expected_correctness_cases,
        "passed_cases": plan.expected_correctness_cases,
        "exact_case_ok": True,
        "device_claim_open": opened,
        "device_claim_released": released,
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
        hits = _matching(rows, expectation.kernel_pattern)
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
    argv = plan.candidate_rocprof_argv if arm == "candidate" else plan.anchor_rocprof_argv
    identity = plan.candidate if arm == "candidate" else plan.anchor
    exact = plan.dispatch.candidate_exact if arm == "candidate" else plan.dispatch.anchor_exact
    forbidden = (plan.dispatch.candidate_forbidden if arm == "candidate"
                 else plan.dispatch.anchor_forbidden)
    inputs = (plan.candidate_rocprof_inputs if arm == "candidate"
              else plan.anchor_rocprof_inputs)
    invocation = CommandInvocation(
        kind="rocprof", arm=arm, argv=argv,
        stdout_path=(directory / "stdout.txt").resolve(),
        stderr_path=(directory / "stderr.txt").resolve(),
        timestamp_csv_path=(directory / "timestamps.csv").resolve())
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
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in inputs],
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "command_argv": list(argv),
        "exit_code": capture.exit_code,
        **outputs,
        "timestamp_reduction_sha256": schemas.content_hash(dispatches),
        "started_at": capture.started_at,
        "ended_at": capture.ended_at,
        "dispatches": dispatches,
        "exact_dispatch_signatures": reduction["exact"],
        "forbidden_dispatch_signatures": reduction["forbidden"],
        "invariant_signatures": reduction["invariants"],
        "device_claim_open": opened,
        "device_claim_released": released,
        "residency_witness": residency,
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
        "execution_policy": _bound_reference(plan.policy),
        "correctness_inputs": [_bound_reference(x) for x in plan.correctness_inputs],
        "candidate_rocprof_inputs": [_bound_reference(x) for x in plan.candidate_rocprof_inputs],
        "anchor_rocprof_inputs": [_bound_reference(x) for x in plan.anchor_rocprof_inputs],
        "expectations": _expectations(plan),
        "candidate": _reference(candidate),
        "anchor": _reference(anchor),
        "invariant_signatures": candidate_body["invariant_signatures"],
        "inverse_attribution_proved": True,
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
    argv = plan.candidate_rocprof_argv if arm == "candidate" else plan.anchor_rocprof_argv
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
        "identity_files": _identity_files_reference(plan.identity_files),
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in (
            plan.candidate_rocprof_inputs if arm == "candidate"
            else plan.anchor_rocprof_inputs)],
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError(f"{arm} attribution receipt identity/config mismatch")
    _validate_claim_pair(body.get("device_claim_open", {}),
                         body.get("device_claim_released", {}), plan=plan)
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


def _validate_correctness_body(body: Mapping[str, Any], plan: GpuSourceEvidencePlan) -> None:
    expected = {
        "schema": CORRECTNESS_SCHEMA, "authority": AUTHORITY,
        "non_promotable": True, "promotion_claim": False,
        "status": "complete", "result": "PASS", "campaign_id": plan.campaign_id,
        "device_id": plan.device_id, "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "workload_sha256": plan.workload_sha256,
        "command_argv": list(plan.correctness_argv), "exit_code": 0,
        "identity_files": _identity_files_reference(plan.identity_files),
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in plan.correctness_inputs],
        "correctness_summary_pattern": plan.correctness_summary_pattern,
        "expected_cases": plan.expected_correctness_cases,
        "passed_cases": plan.expected_correctness_cases, "exact_case_ok": True,
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError("correctness receipt identity/config/result mismatch")
    _validate_claim_pair(body.get("device_claim_open", {}),
                         body.get("device_claim_released", {}), plan=plan)
    for kind in ("stdout", "stderr"):
        path = Path(str(body.get(f"{kind}_path", "")))
        if _hash_file(path, kind, allow_empty=kind == "stderr") != body.get(f"{kind}_sha256"):
            raise EvidenceProducerError(f"correctness {kind} bytes changed")
    if _parse_summary(Path(str(body["stdout_path"])).read_text(encoding="utf-8"), plan) != body.get("summary"):
        raise EvidenceProducerError("correctness summary changed")
    _validate_residency_witness(
        body.get("residency_witness"), device_id=plan.device_id,
        label="correctness")


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
            correctness_summary_pattern=str(correct_body["correctness_summary_pattern"]),
            expected_correctness_cases=int(correct_body["expected_cases"]),
            candidate_rocprof_argv=tuple(candidate_body["command_argv"]),
            anchor_rocprof_argv=tuple(anchor_body["command_argv"]),
            dispatch=_contract_from_dict(pair_body["expectations"]),
            identity_files=_identity_files_from_dict(pair_body["identity_files"]),
            policy=_bound_from_dict(pair_body["execution_policy"]),
            correctness_inputs=tuple(_bound_from_dict(x) for x in pair_body["correctness_inputs"]),
            candidate_rocprof_inputs=tuple(_bound_from_dict(x) for x in pair_body["candidate_rocprof_inputs"]),
            anchor_rocprof_inputs=tuple(_bound_from_dict(x) for x in pair_body["anchor_rocprof_inputs"]),
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
    "AUTHORITY", "EvidenceProducerError", "GpuResidencySample",
    "ExecutionCapture", "CommandInvocation", "CommandExecutor",
    "ExactDispatch", "ForbiddenDispatch", "InvariantDispatch",
    "DispatchContract", "GpuSourceEvidencePlan", "produce_gpu_source_evidence",
    "load_gpu_source_evidence_bundle",
]
