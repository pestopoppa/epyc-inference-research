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
import stat
import struct
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from .. import schemas
from ..evaluator import hawkeye_measurement
from ..execution import microbench, t0_provider
from ..resource import device_claim
from ... import c6_reward_integrity
from . import gpu_source_proofs as proofs
from . import split_runtime_verifier

AUTHORITY = "nonpromotable_candidate_only_discovery"
BORROWED_PHASE_SCHEMA = "epyc.autokernel.borrowed_device_claim_phase.v1"
CORRECTNESS_SCHEMA = "epyc.autokernel.targeted_correctness_receipt.v3"
C6_CORRECTNESS_SCHEMA = "epyc.autokernel.c6_correctness_receipt.v1"
C6_SIDECAR_SCHEMA = "epyc.autokernel.c6_native_operator_sidecar.v3"
C6_ORACLE_SIDECAR_SCHEMA = "epyc.autokernel.c6_native_oracle_phase.v1"
C6_INPUT_BINDING_SCHEMA = "epyc.autokernel.c6_input_binding.v1"
C6_LEG_BINDING_SCHEMA = "epyc.autokernel.c6_leg_binding.v1"
C6_MODE_COMBINED = "combined"
C6_MODE_ORACLE = "oracle"
C6_MODE_CANDIDATE = "candidate"
C6_READY_TOKEN = b"R"
C6_CONTINUE_TOKEN = b"C"
C6_SIDECAR_OUTPUT = "{C6_SIDECAR_OUTPUT}"
CORRECTNESS_REFUSAL_SCHEMA = "epyc.autokernel.targeted_correctness_refusal.v1"
CORRECTNESS_PARSER_ID = "ak.t0.backend_ops_console/v1"
EXECUTION_POLICY_SCHEMA = "epyc.autokernel.gpu_source_execution_policy.v2"
ATTRIBUTION_SCHEMA = "epyc.autokernel.gpu_kernel_attribution.v3"
ATTRIBUTION_REFUSAL_SCHEMA = "epyc.autokernel.gpu_kernel_attribution_refusal.v1"
PROFILER_ATTEMPT_SCHEMA = "epyc.autokernel.rocprofiler_transport_attempt.v2"
PROFILER_RUNTIME_SCHEMA = "epyc.autokernel.rocprofiler_runtime_closure.v1"
PAIR_SCHEMA = "epyc.autokernel.gpu_kernel_attribution_pair.v2"
PAIR_REFUSAL_SCHEMA = "epyc.autokernel.gpu_kernel_attribution_pair_refusal.v1"
SEALED_BUNDLE_SCHEMA = "epyc.autokernel.gpu_source_evidence_bundle.v1"
SHA = re.compile(r"^[0-9a-f]{64}$")
SOURCE_TREE_SCHEMA = "epyc.autokernel.source_tree_identity.v1"
ROCPROF_TIMESTAMP_OUTPUT = "{TIMESTAMP_CSV}"
ROCPROF_OUTPUT_DIRECTORY = "{ROCPROF_OUTPUT_DIRECTORY}"
ROCPROF_OUTPUT_BASENAME = "{ROCPROF_OUTPUT_BASENAME}"
# Retained only to validate already-sealed historical receipts. New governed
# evidence plans are emitted by the deployment factory with ROCPROF_V3_TRACE_ID.
ROCPROF_V1_TRACE_ID = "rocprof-v1-timestamps-v1"
ROCPROF_V3_TRACE_ID = "rocprof-v3-kernel-trace-csv-v1"
ROCPROF_V3_TRANSPORT_POLICY = "require-zero-exit-v1"
ROCPROF_V3_PYTHON = Path("/usr/bin/python3.13")
ROCPROF_V3_COLUMNS = (
    "Kind", "Agent_Id", "Queue_Id", "Kernel_Id", "Kernel_Name",
    "Correlation_Id", "Start_Timestamp", "End_Timestamp",
    "Private_Segment_Size", "Group_Segment_Size",
    "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
    "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z")
ROCPROF_V3_AGENT_COLUMNS = (
    "Node_Id", "Logical_Node_Id", "Agent_Type", "Cpu_Cores_Count",
    "Simd_Count", "Cpu_Core_Id_Base", "Simd_Id_Base", "Max_Waves_Per_Simd",
    "Lds_Size_In_Kb", "Gds_Size_In_Kb", "Num_Gws", "Wave_Front_Size",
    "Num_Xcc", "Cu_Count", "Array_Count", "Num_Shader_Banks",
    "Simd_Arrays_Per_Engine", "Cu_Per_Simd_Array", "Simd_Per_Cu",
    "Max_Slots_Scratch_Cu", "Gfx_Target_Version", "Vendor_Id", "Device_Id",
    "Location_Id", "Domain", "Drm_Render_Minor", "Num_Sdma_Engines",
    "Num_Sdma_Xgmi_Engines", "Num_Sdma_Queues_Per_Engine", "Num_Cp_Queues",
    "Max_Engine_Clk_Ccompute", "Max_Engine_Clk_Fcompute", "Sdma_Fw_Version",
    "Fw_Version", "Capability", "Cu_Per_Engine", "Max_Waves_Per_Cu",
    "Family_Id", "Workgroup_Max_Size", "Grid_Max_Size", "Local_Mem_Size",
    "Hive_Id", "Gpu_Id", "Workgroup_Max_Dim_X", "Workgroup_Max_Dim_Y",
    "Workgroup_Max_Dim_Z", "Grid_Max_Dim_X", "Grid_Max_Dim_Y",
    "Grid_Max_Dim_Z", "Name", "Vendor_Name", "Product_Name", "Model_Name")
PROFILER_MAPPED_ROLES = frozenset({
    "profiler_sdk_library", "profiler_sdk_tool_library",
    "profiler_aqlprofile_library", "profiler_hsa_runtime_library",
    "profiler_register_library",
})
PROFILER_V3_INPUT_ROLES = frozenset({
    "executable", "profiler_wrapper", "profiler_package",
    "profiler_runtime_manifest", "profiler_aqlprofile_manifest",
    # libpci is part of the sealed loader/provenance closure but is not a
    # resident DSO in the governed rocprofv3 KFD child.  Requiring a mapping
    # would reject the exact observed runtime; it remains byte-bound here.
    "profiler_libpci_manifest", "profiler_libpci_library",
    *PROFILER_MAPPED_ROLES,
})


class EvidenceProducerError(RuntimeError):
    """The producer refused to mint a success receipt."""


class RuntimeMapsNotReady(RuntimeError):
    """The owned GPU process has not mapped the complete sealed closure yet.

    This is the sole retryable runtime-map outcome.  The direct executor may
    retry it only while the exact captured child remains alive; every other
    callback failure is terminal and tears the child down.
    """


class SealedEvidenceRefusal(EvidenceProducerError):
    """A scientific stage ended durably without a passing receipt."""

    def __init__(self, message: str, *, receipt_path: str | None = None,
                 receipt_sha256: str | None = None) -> None:
        super().__init__(message)
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256


class CorrectnessParseRefusal(SealedEvidenceRefusal):
    """The authoritative backend-op parser could not prove the targeted run."""


class DispatchAttributionParseRefusal(SealedEvidenceRefusal):
    """The measured profile violated the exact reviewed route contract."""


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


def _c6_mode_from_argv(argv: Sequence[str]) -> str:
    if len(argv) >= 3 and argv[1] == "--mode":
        if argv[2] in {C6_MODE_ORACLE, C6_MODE_COMBINED, C6_MODE_CANDIDATE}:
            return argv[2]
        raise EvidenceProducerError("native C6 argv mode is not oracle/combined/candidate")
    return C6_MODE_COMBINED


def _c6_case_identity_from_argv(argv: Sequence[str]) -> dict[str, Any]:
    """Reduce the sealed native argv to one exact seeded case identity."""
    if not argv or not isinstance(argv[0], str):
        raise EvidenceProducerError("native C6 argv is not the canonical case shape")
    mode = _c6_mode_from_argv(argv)
    tail = argv[3:] if argv[1] == "--mode" else argv[1:]
    expected_keys = (
        "--operation", "--backend", "--type-a", "--m", "--n", "--k",
        "--seed", "--sidecar")
    keys = tuple(tail[index] for index in range(0, len(tail), 2))
    if len(tail) != 16 or keys != expected_keys:
        raise EvidenceProducerError("native C6 argv option order changed")
    values = {tail[index]: tail[index + 1] for index in range(0, len(tail), 2)}
    if (values["--operation"] not in {"MUL_MAT", "RMS_NORM", "FLASH_ATTN_EXT"}
            or not values["--backend"] or not values["--type-a"]
            or values["--sidecar"] != C6_SIDECAR_OUTPUT):
        raise EvidenceProducerError("native C6 argv identity is incomplete")
    try:
        dimensions = {key[2:]: int(values[key]) for key in ("--m", "--n", "--k")}
        seed = int(values["--seed"])
    except ValueError as exc:
        raise EvidenceProducerError("native C6 dimensions/seed are not integers") from exc
    if any(value < 1 for value in dimensions.values()) or seed < 0:
        raise EvidenceProducerError("native C6 dimensions/seed are out of range")
    return {
        "mode": mode,
        "operation": values["--operation"],
        "backend": values["--backend"], "type_a": values["--type-a"],
        **dimensions, "seed": seed,
    }


def _c6_case_path(value: object, label: str) -> Path:
    if (not isinstance(value, str) or not value
            or not Path(value).is_absolute()
            or Path(value) != Path(value).resolve(strict=False)
            or Path(value).is_symlink()):
        raise EvidenceProducerError(f"{label} is not a canonical absolute path")
    return Path(value)


def _c6_candidate_argv_from_argv(
        argv: Sequence[str], *, input_dir: Path, output: Path,
        ready_file: Path, continue_file: Path) -> tuple[str, ...]:
    """Derive one candidate-leg argv deterministically from the oracle argv.

    The candidate never generates inputs: it loads the exact bytes the
    evaluator handed over from the oracle witness.  Every path is injected
    here so the sealed receipt can re-derive the exact argv at reopen time.
    """
    if _c6_mode_from_argv(argv) != C6_MODE_ORACLE:
        raise EvidenceProducerError(
            "candidate argv must derive from an oracle-mode capability")
    if argv.count(C6_SIDECAR_OUTPUT) != 1:
        raise EvidenceProducerError(
            "candidate argv derivation lacks the exact sidecar token")
    tail = argv[3:]
    base = tuple(
        item for pair in (
            (tail[index], tail[index + 1])
            for index in range(0, len(tail), 2))
        if pair[0] != "--sidecar" for item in pair)
    if len(base) != 14 or any(
            key not in {"--operation", "--backend", "--type-a", "--m",
                        "--n", "--k", "--seed"}
            for key in base[::2]):
        raise EvidenceProducerError(
            "candidate argv derivation lost canonical case options")
    for label, path in (("input directory", input_dir),
                        ("output path", output),
                        ("ready path", ready_file),
                        ("continue path", continue_file)):
        _c6_case_path(str(path), label)
    return (str(argv[0]), "--mode", C6_MODE_CANDIDATE, *base,
            "--input-dir", str(input_dir),
            "--output", str(output),
            "--ready-file", str(ready_file),
            "--continue-file", str(continue_file))


def _c6_candidate_paths_from_argv(
        argv: Sequence[str]) -> dict[str, Path]:
    """Recover the four candidate paths from one attested candidate argv."""
    if _c6_mode_from_argv(argv) != C6_MODE_CANDIDATE:
        raise EvidenceProducerError("argv is not a candidate-leg argv")
    expected = ("--operation", "--backend", "--type-a", "--m", "--n", "--k",
                "--seed", "--input-dir", "--output", "--ready-file",
                "--continue-file")
    tail = argv[3:]
    if len(tail) != 22 or tuple(tail[index] for index in range(0, 22, 2)) != expected:
        raise EvidenceProducerError("candidate-leg argv option order changed")
    values = {tail[index]: tail[index + 1] for index in range(0, len(tail), 2)}
    return {
        "input_dir": _c6_case_path(values["--input-dir"], "candidate input directory"),
        "output": _c6_case_path(values["--output"], "candidate output"),
        "ready_file": _c6_case_path(values["--ready-file"], "candidate ready file"),
        "continue_file": _c6_case_path(values["--continue-file"], "candidate continue file"),
    }


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
class C6CorrectnessPlan:
    """Producer-owned native C6 witness and pre-tolerance precision policy."""

    argv: tuple[str, ...]
    inputs: tuple[BoundInputFile, ...]
    precision_contract: Mapping[str, Any]
    precision_equivalence_policy: Mapping[str, Any]
    structural_precision_evidence: Mapping[str, Any]
    semantic_judge_verdicts: Mapping[str, str]

    def __post_init__(self) -> None:
        _argv(self.argv, "C6 correctness argv")
        _c6_case_identity_from_argv(self.argv)
        if (self.argv.count(C6_SIDECAR_OUTPUT) != 1
                or not Path(self.argv[0]).is_absolute()
                or not any(item.role == "executable"
                           and item.path == Path(self.argv[0])
                           for item in self.inputs)):
            raise EvidenceProducerError(
                "C6 correctness must bind one native executable and sidecar")
        if any(not isinstance(item, BoundInputFile) for item in self.inputs):
            raise EvidenceProducerError("C6 correctness inputs must be typed")
        try:
            c6_reward_integrity.PrecisionContract(**dict(self.precision_contract))
            hawkeye_measurement.PrecisionEquivalencePolicy(
                **dict(self.precision_equivalence_policy))
            c6_reward_integrity.StructuralPrecisionEvidence(
                **dict(self.structural_precision_evidence))
            calibration = c6_reward_integrity.calibrate_semantic_judge(
                self.semantic_judge_verdicts)
        except (TypeError, c6_reward_integrity.EvaluatorPolicyError) as exc:
            raise EvidenceProducerError("C6 correctness policy is invalid") from exc
        if calibration.gating:
            raise EvidenceProducerError(
                "semantic judge must remain non-gating until separate ratification")


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
                        try:
                            sampled_identity = self.runtime_maps_sampler(
                                invocation, int(child.pid), sample)
                        except RuntimeMapsNotReady:
                            # A KFD client can become visible before llama-bench
                            # has mapped its model and the sealed HIP/common
                            # closure.  Retry only this typed startup state.
                            sampled_identity = None
                        if sampled_identity is not None:
                            if not isinstance(sampled_identity, Mapping):
                                raise EvidenceProducerError(
                                    "runtime maps sampler returned no typed identity")
                            runtime_maps_identity = sampled_identity
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
        if invocation.runtime_maps_required and runtime_maps_identity is None:
            raise EvidenceProducerError(
                "runtime maps did not prove the sealed arm during child execution")
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
    candidate_structural_exact: tuple[ExactDispatch, ...] = ()
    anchor_structural_exact: tuple[ExactDispatch, ...] = ()
    candidate_forbidden: tuple[ForbiddenDispatch, ...] = ()
    anchor_forbidden: tuple[ForbiddenDispatch, ...] = ()
    invariants: tuple[InvariantDispatch, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_exact or not self.anchor_exact:
            raise EvidenceProducerError("both arms require exact dispatch expectations")
        signatures = [item.signature for group in (
            self.candidate_exact, self.anchor_exact,
            self.candidate_structural_exact, self.anchor_structural_exact,
            self.candidate_forbidden,
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
    c6_correctness: C6CorrectnessPlan | None = None
    # Empty means the legacy single command above.  Portfolio templates may
    # instead seal an ordered multi-invocation correctness contract (FA uses a
    # generic corpus plus a dedicated odd-GQA7 corpus).
    correctness_invocations: tuple[Mapping[str, Any], ...] = ()
    attribution_arm_order_seed_sha256: str = "0" * 64
    attribution_arm_order: tuple[str, str] = ("candidate", "anchor")
    profiler_trace_schema_id: str = ROCPROF_V1_TRACE_ID
    expected_candidate_profiler_dispatch_rows: int | None = None
    expected_anchor_profiler_dispatch_rows: int | None = None
    profiler_transport_policy: str = "require-zero-exit"

    def __post_init__(self) -> None:
        if (not isinstance(self.campaign_id, str) or not self.campaign_id
                or not isinstance(self.device_id, str) or not self.device_id):
            raise EvidenceProducerError("campaign and device identities are required")
        for name in ("manifest_sha256", "model_sha256", "workload_sha256",
                     "runtime_config_sha256", "attribution_arm_order_seed_sha256"):
            _hash(getattr(self, name), name)
        if (not isinstance(self.attribution_arm_order, tuple)
                or len(self.attribution_arm_order) != 2
                or set(self.attribution_arm_order) != {"candidate", "anchor"}):
            raise EvidenceProducerError(
                "attribution arm order must contain candidate and anchor exactly once")
        if self.profiler_trace_schema_id not in {
                ROCPROF_V1_TRACE_ID, ROCPROF_V3_TRACE_ID}:
            raise EvidenceProducerError("profiler trace schema is not reviewed")
        if self.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
            for value in (self.expected_candidate_profiler_dispatch_rows,
                          self.expected_anchor_profiler_dispatch_rows):
                if (isinstance(value, bool) or not isinstance(value, int)
                        or value < 1):
                    raise EvidenceProducerError(
                        "rocprofv3 requires exact positive per-arm dispatch-row counts")
            if self.profiler_transport_policy != ROCPROF_V3_TRANSPORT_POLICY:
                raise EvidenceProducerError("rocprofv3 transport policy is not reviewed")
        elif (self.expected_candidate_profiler_dispatch_rows is not None
              or self.expected_anchor_profiler_dispatch_rows is not None
              or self.profiler_transport_policy != "require-zero-exit"):
            raise EvidenceProducerError("rocprof-v1 cannot claim rocprofv3 transport authority")
        for invocation in self.correctness_invocations:
            if (not isinstance(invocation, Mapping)
                    or not isinstance(invocation.get("invocation_id"), str)
                    or not invocation["invocation_id"]
                    or not isinstance(invocation.get("argv"), (list, tuple))
                    or not invocation["argv"]
                    or isinstance(invocation.get("expected_cases"), bool)
                    or not isinstance(invocation.get("expected_cases"), int)
                    or invocation["expected_cases"] < 1
                    or not isinstance(invocation.get("required_cases", []), list)):
                raise EvidenceProducerError(
                    "correctness invocation contract is malformed")
        if len({row["invocation_id"] for row in self.correctness_invocations}) \
                != len(self.correctness_invocations):
            raise EvidenceProducerError("correctness invocation IDs must be unique")
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
        if self.c6_correctness is not None:
            c6_identity = _c6_case_identity_from_argv(
                self.c6_correctness.argv)
            if (self.correctness_op not in {
                    "MUL_MAT", "RMS_NORM", "FLASH_ATTN_EXT"}
                    or c6_identity["operation"] != self.correctness_op):
                raise EvidenceProducerError(
                    "reviewed native C6 capability differs from correctness operation")
            if (dict(self.correctness_environment).get("LD_LIBRARY_PATH", "").split(":")[0]
                    != str(self.identity_files.candidate.hip_library.path.parent)):
                raise EvidenceProducerError(
                    "native C6 LD_LIBRARY_PATH must load the exact candidate backend first")
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
            if self.profiler_trace_schema_id == ROCPROF_V1_TRACE_ID:
                timestamps = [item for item in inputs if item.role == "timestamp_input"]
                if len(timestamps) != 1:
                    raise EvidenceProducerError(f"{label} requires one sealed timestamp input")
                expected_pair = ("-i", str(timestamps[0].path))
                if not any(tuple(command[index:index + 2]) == expected_pair
                           for index in range(len(command) - 1)):
                    raise EvidenceProducerError(f"{label} does not bind rocprof -i input")
            else:
                roles = {item.role for item in inputs}
                wrappers = [item for item in inputs
                            if item.role == "profiler_wrapper"]
                if (len(wrappers) != 1 or str(wrappers[0].path) != command[1]
                        or not {"profiler_runtime_manifest",
                                "profiler_aqlprofile_manifest",
                                "profiler_libpci_manifest"}.issubset(roles)):
                    raise EvidenceProducerError(
                        f"{label} requires the sealed rocprofv3 wrapper and closures")
                _required_profiler_mapped_files(inputs)
                _validate_rocprofv3_argv(command, target_binary=binary)
            if str(binary) not in command:
                raise EvidenceProducerError(f"{label} does not execute its bound target binary")
        if self.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
            def profiler_identity(inputs: Sequence[BoundInputFile]) -> dict[str, tuple[str, str]]:
                selected = {item.role: (str(item.path), item.sha256) for item in inputs
                            if item.role in PROFILER_V3_INPUT_ROLES}
                if set(selected) != PROFILER_V3_INPUT_ROLES:
                    raise EvidenceProducerError(
                        "rocprofv3 input closure is incomplete")
                return selected
            if profiler_identity(self.candidate_rocprof_inputs) != profiler_identity(
                    self.anchor_rocprof_inputs):
                raise EvidenceProducerError(
                    "rocprofv3 arms do not share one exact profiler closure")
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


def _validate_rocprofv3_argv(argv: tuple[str, ...], *, target_binary: Path) -> None:
    prefix = ("--kernel-trace", "-d", ROCPROF_OUTPUT_DIRECTORY,
              "-o", ROCPROF_OUTPUT_BASENAME, "--output-format", "csv", "--")
    if len(argv) < 3 or Path(argv[0]) != ROCPROF_V3_PYTHON:
        raise EvidenceProducerError("rocprofv3 must use the pinned Python interpreter")
    if tuple(argv[2:2 + len(prefix)]) != prefix:
        raise EvidenceProducerError("rocprofv3 command prefix is not exact kernel-trace CSV")
    try:
        target_index = argv.index(str(target_binary))
    except ValueError as exc:
        raise EvidenceProducerError("rocprofv3 command lacks its bound target") from exc
    if tuple(argv[target_index + 1:]).count("json") != 1:
        raise EvidenceProducerError("rocprofv3 target must emit one JSON result")
    if not any(tuple(argv[index:index + 2]) == ("-o", "json")
               for index in range(target_index + 1, len(argv) - 1)):
        raise EvidenceProducerError("rocprofv3 target does not select JSON output")


def _normalized_rocprof_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize only the producer-owned rocprof ``-o`` output placeholder.

    The plan cannot name a per-operation output directory.  It may use this
    exact token once; the producer substitutes a fresh contained CSV path and
    records the resulting argv.  No other arm-specific argv variation is
    admissible for source-patch reward attribution.
    """
    result = list(argv)
    if ROCPROF_OUTPUT_DIRECTORY in result or ROCPROF_OUTPUT_BASENAME in result:
        if (result.count(ROCPROF_OUTPUT_DIRECTORY) != 1
                or result.count(ROCPROF_OUTPUT_BASENAME) != 1
                or ROCPROF_TIMESTAMP_OUTPUT in result):
            raise EvidenceProducerError("rocprofv3 output authority is ambiguous")
        return tuple(result)
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
    if not output.is_absolute():
        raise EvidenceProducerError("rocprof output substitution requires an absolute path")
    basename = output.stem
    if ROCPROF_OUTPUT_BASENAME in argv:
        if not output.name.endswith("_kernel_trace.csv"):
            raise EvidenceProducerError("rocprofv3 output path lacks kernel-trace suffix")
        basename = output.name[:-len("_kernel_trace.csv")]
    replacements = {
        ROCPROF_TIMESTAMP_OUTPUT: str(output),
        ROCPROF_OUTPUT_DIRECTORY: str(output.parent),
        ROCPROF_OUTPUT_BASENAME: basename,
    }
    return tuple(replacements.get(item, item) for item in argv)


def _receipt_rocprof_template(body: Mapping[str, Any]) -> tuple[str, ...]:
    """Recover the sole producer-owned output placeholder from a receipt."""
    try:
        argv = list(body["command_argv"])
        output = str(body["timestamp_csv_path"])
    except (KeyError, TypeError) as exc:
        raise EvidenceProducerError("attribution receipt lacks command/output binding") from exc
    if body.get("profiler_trace_schema_id") == ROCPROF_V3_TRACE_ID:
        path = Path(output)
        directory = str(path.parent)
        if not path.name.endswith("_kernel_trace.csv"):
            raise EvidenceProducerError("rocprofv3 receipt output suffix changed")
        basename = path.name[:-len("_kernel_trace.csv")]
        if (argv.count(directory) != 1 or argv.count(basename) != 1
                or not any(tuple(argv[index:index + 2]) == ("-d", directory)
                           for index in range(len(argv) - 1))
                or not any(tuple(argv[index:index + 2]) == ("-o", basename)
                           for index in range(len(argv) - 1))):
            raise EvidenceProducerError("rocprofv3 receipt has unbound output authority")
        argv[argv.index(directory)] = ROCPROF_OUTPUT_DIRECTORY
        argv[argv.index(basename)] = ROCPROF_OUTPUT_BASENAME
        return tuple(argv)
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


def _required_profiler_mapped_files(
        inputs: Sequence[BoundInputFile]) -> dict[str, str]:
    selected = [item for item in inputs if item.role in PROFILER_MAPPED_ROLES]
    roles = {item.role for item in selected}
    if (roles != PROFILER_MAPPED_ROLES or len(selected) != len(roles)
            or len({item.path for item in selected}) != len(selected)):
        raise EvidenceProducerError(
            "rocprofv3 mapped DSO closure is incomplete or ambiguous")
    return {str(item.path): item.sha256 for item in selected}


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
        "correctness_invocations": [dict(row) for row in plan.correctness_invocations],
        "c6_correctness": (None if plan.c6_correctness is None else {
            "argv": list(plan.c6_correctness.argv),
            "inputs": [_bound_reference(x) for x in plan.c6_correctness.inputs],
            "precision_contract": dict(plan.c6_correctness.precision_contract),
            "precision_equivalence_policy": dict(
                plan.c6_correctness.precision_equivalence_policy),
            "structural_precision_evidence": dict(
                plan.c6_correctness.structural_precision_evidence),
            "semantic_judge_verdicts": dict(
                plan.c6_correctness.semantic_judge_verdicts),
        }),
        "candidate_rocprof_argv": list(plan.candidate_rocprof_argv),
        "anchor_rocprof_argv": list(plan.anchor_rocprof_argv),
        "profiler_trace_schema_id": plan.profiler_trace_schema_id,
        "expected_candidate_profiler_dispatch_rows": (
            plan.expected_candidate_profiler_dispatch_rows),
        "expected_anchor_profiler_dispatch_rows": (
            plan.expected_anchor_profiler_dispatch_rows),
        "profiler_transport_policy": plan.profiler_transport_policy,
        "attribution_arm_order_seed_sha256": plan.attribution_arm_order_seed_sha256,
        "attribution_arm_order": list(plan.attribution_arm_order),
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
    if plan.c6_correctness is None:
        raise EvidenceProducerError("production evidence plan lacks native C6 correctness")
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
                 + plan.anchor_rocprof_inputs
                 + plan.c6_correctness.inputs):
        _verify_bound(item)
    _verify_executable(
        plan.c6_correctness.argv, plan.c6_correctness.inputs,
        "C6 correctness")
    if plan.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
        manifests = {
            item.path: item for item in (
                plan.candidate_rocprof_inputs + plan.anchor_rocprof_inputs)
            if item.role in {"profiler_runtime_manifest",
                             "profiler_aqlprofile_manifest",
                             "profiler_libpci_manifest"}}
        roles = {item.role for item in manifests.values()}
        if roles != {"profiler_runtime_manifest", "profiler_aqlprofile_manifest",
                     "profiler_libpci_manifest"}:
            raise EvidenceProducerError(
                "rocprofv3 arms do not bind the SDK and dependency closures")
        for manifest in manifests.values():
            _verify_profiler_runtime_manifest(manifest)
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


def profiler_prefix_snapshot(root: Path) -> dict[str, Any]:
    """Hash every file/link in the side-loaded profiler prefix.

    The prefix is intentionally outside the repository and package manager.
    Binding only the wrapper would leave the injected tool, plugin and libpci
    closure mutable after deployment validation.
    """
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise EvidenceProducerError("profiler prefix must be an absolute real directory")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise EvidenceProducerError(
                    f"profiler prefix symlink escapes closure: {relative}") from exc
            if not resolved.is_file() or resolved.is_symlink() \
                    or resolved.stat().st_nlink != 1:
                raise EvidenceProducerError(
                    f"profiler prefix symlink target is not a single-link file: {relative}")
            entries.append({"path": relative, "type": "symlink",
                            "target": os.readlink(path),
                            "target_sha256": _hash_file(
                                resolved, f"profiler symlink target:{relative}"),
                            "target_bytes": resolved.stat().st_size})
        elif path.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif path.is_file():
            if path.stat().st_nlink != 1:
                raise EvidenceProducerError(
                    f"profiler prefix contains a hardlinked file: {relative}")
            entries.append({"path": relative, "type": "file",
                            "sha256": _hash_file(path, f"profiler:{relative}"),
                            "bytes": path.stat().st_size,
                            "mode": path.stat().st_mode & 0o777})
        else:
            raise EvidenceProducerError(
                f"profiler prefix contains a special file: {relative}")
    return {"schema": PROFILER_RUNTIME_SCHEMA, "root": str(root),
            "entries": entries, "entry_count": len(entries),
            "complete_listing": True}


def _verify_profiler_runtime_manifest(bound: BoundInputFile) -> None:
    if bound.role not in {"profiler_runtime_manifest", "profiler_aqlprofile_manifest",
                          "profiler_libpci_manifest"}:
        raise EvidenceProducerError("profiler runtime manifest role changed")
    try:
        body = json.loads(bound.path.read_text(encoding="utf-8"))
        root = Path(str(body["root"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise EvidenceProducerError("profiler runtime manifest is malformed") from exc
    if body != profiler_prefix_snapshot(root):
        raise EvidenceProducerError("profiler runtime closure changed after sealing")
    entries = body.get("entries")
    if not isinstance(entries, list):
        raise EvidenceProducerError("profiler runtime closure listing is malformed")
    paths = {str(row.get("path")) for row in entries
             if isinstance(row, Mapping) and row.get("type") in {"file", "symlink"}}
    if bound.role == "profiler_runtime_manifest":
        required_basenames = {
            "rocprofv3", "librocprofiler-sdk.so.0.4.0",
            "librocprofiler-sdk-tool.so.0.4.0",
        }
        if not required_basenames.issubset({Path(path).name for path in paths}):
            raise EvidenceProducerError(
                "rocprofv3 SDK closure omits its wrapper or injected libraries")
    elif bound.role == "profiler_aqlprofile_manifest":
        if "libhsa-amd-aqlprofile64.so.1.0.60200" not in {
                Path(path).name for path in paths}:
            raise EvidenceProducerError(
                "aqlprofile closure omits the reviewed gfx90a DSO")
    elif bound.role == "profiler_libpci_manifest":
        if "libpciaccess.so.0.11.1" not in {Path(path).name for path in paths}:
            raise EvidenceProducerError(
                "profiler libpci closure omits the reviewed runtime DSO")


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


def _open_claim_boundary_fields(
        opened: Mapping[str, Any], residency: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a phase while its physical/borrowed claim is still held."""
    borrowed = residency.get("device_claim_mode") == "borrowed_outer_reservation"
    return {
        "device_claim_open": dict(opened),
        "device_claim_mode": ("borrowed_outer_reservation" if borrowed
                              else "direct_device_claim"),
        "device_claim_state_at_seal": "held",
    }


def _validate_open_claim_boundary(
        body: Mapping[str, Any], *, plan: GpuSourceEvidencePlan) -> None:
    if (body.get("device_claim_state_at_seal") != "held"
            or body.get("device_claim_mode") not in {
                "direct_device_claim", "borrowed_outer_reservation"}
            or body.get("device_claim_released") is not None
            or body.get("device_claim_borrowed_phase_end") is not None):
        raise EvidenceProducerError("C6 receipt was not sealed under a held claim")
    try:
        opened = device_claim.ClaimReceipt.from_dict(
            body.get("device_claim_open", {}))
    except (TypeError, ValueError) as exc:
        raise EvidenceProducerError(
            "C6 held claim does not satisfy the device-claim schema") from exc
    if (opened.released_at is not None or opened.device_id != plan.device_id
            or opened.campaign_id != plan.campaign_id):
        raise EvidenceProducerError(
            "C6 held claim does not bind the planned device/campaign")


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


def _run_c6_then_targeted_claimed(
    c6_invocation: CommandInvocation, targeted_invocation: CommandInvocation,
    c6_sidecar: Path, *, plan: GpuSourceEvidencePlan,
    executor: CommandExecutor, claim_acquirer: Callable[..., Any],
    claim_verifier: Callable[[Mapping[str, Any]], object], claim_journal: Any,
    claim_timeout_s: float,
) -> tuple[ExecutionCapture, dict[str, Any], dict[str, Any],
           dict[str, Any], Mapping[str, Any]]:
    """Seal/reopen native C6 before targeted correctness in one held claim."""
    invocations = (c6_invocation, targeted_invocation)
    if any(item.kind != "correctness" or item.arm != "candidate"
           for item in invocations):
        raise EvidenceProducerError(
            "claimed correctness sequence requires two candidate commands")
    _verify_plan_files(plan)
    output_paths = [path for invocation in invocations
                    for path in (invocation.stdout_path, invocation.stderr_path)]
    if len(output_paths) != len(set(output_paths)) or any(
            path.exists() or path.is_symlink() for path in output_paths):
        raise EvidenceProducerError("sequence outputs must be fresh unique paths")
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    claim = None
    borrowed_outer = False
    opened = released = None
    targeted_capture: ExecutionCapture | None = None
    c6_receipt: Mapping[str, Any] | None = None
    targeted_residency: dict[str, Any] | None = None
    verified_before = verified_after = False
    failure: BaseException | None = None
    try:
        claim = claim_acquirer(
            plan.device_id,
            purpose="AutoKernel GPU source evidence correctness/candidate",
            campaign_id=plan.campaign_id, journal=claim_journal,
            holder_label="gpu_source_evidence.py", timeout_s=claim_timeout_s,
            max_hold_s=3600.0)
        borrowed_outer = bool(getattr(claim, "borrowed_outer_reservation", False))
        opened = _receipt_dict(claim.receipt(), "opened claim")
        verified_before = _check_result_passed(claim_verifier(opened))
        if not verified_before:
            raise EvidenceProducerError(
                "device claim was not verifiably held before execution")
        c6_capture = executor(c6_invocation)
        if (not isinstance(c6_capture, ExecutionCapture)
                or c6_capture.argv != c6_invocation.argv):
            raise EvidenceProducerError(
                "executor did not attest the exact native C6 argv")
        c6_residency = _residency(c6_capture, plan.device_id)
        c6_residency.update({
            "claim_verified_before": verified_before,
            "claim_verified_after": _check_result_passed(
                claim_verifier(opened)),
            "device_claim_mode": (
                "borrowed_outer_reservation" if borrowed_outer
                else "direct_device_claim"),
            "outer_claim_id": opened["claim_id"] if borrowed_outer else None,
        })
        if c6_residency["claim_verified_after"] is not True:
            raise EvidenceProducerError(
                "device claim was not held after native C6 execution")
        if not c6_sidecar.exists() or c6_sidecar.is_symlink():
            raise EvidenceProducerError(
                "native C6 command did not emit its sidecar")
        c6_receipt = _evaluate_c6_sidecar(
            c6_sidecar, plan, c6_invocation, c6_capture,
            opened, c6_residency)
        _validate_c6_correctness_receipt(c6_receipt, plan)
        _verify_plan_files(plan)
        if not _check_result_passed(claim_verifier(opened)):
            raise EvidenceProducerError(
                "device claim was not held after the sealed C6 boundary")
        targeted_capture = executor(targeted_invocation)
        if (not isinstance(targeted_capture, ExecutionCapture)
                or targeted_capture.argv != targeted_invocation.argv):
            raise EvidenceProducerError(
                "executor did not attest the exact targeted correctness argv")
        verified_after = _check_result_passed(claim_verifier(opened))
        if not verified_after:
            raise EvidenceProducerError(
                "device claim was not verifiably held after correctness sequence")
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
        raise EvidenceProducerError(
            f"correctness sequence execution failed: {failure}") from failure
    if (opened is None or released is None or targeted_capture is None
            or c6_receipt is None):
        raise EvidenceProducerError(
            "claimed correctness sequence did not produce complete evidence")
    _validate_claim_pair(opened, released, plan=plan)
    targeted_residency = _residency(targeted_capture, plan.device_id)
    targeted_residency.update({
        "claim_verified_before": verified_before,
        "claim_verified_after": verified_after,
        "device_claim_mode": (
            "borrowed_outer_reservation" if borrowed_outer
            else "direct_device_claim"),
        "outer_claim_id": opened["claim_id"] if borrowed_outer else None,
    })
    return (targeted_capture, opened, released, targeted_residency,
            c6_receipt)


def _run_c6_oracle_then_targeted_claimed(
    oracle_invocation: CommandInvocation,
    candidate_invocations: tuple[CommandInvocation, ...],
    targeted_invocation: CommandInvocation,
    oracle_sidecar: Path, input_dir: Path, *,
    plan: GpuSourceEvidencePlan, executor: CommandExecutor,
    claim_acquirer: Callable[..., Any],
    claim_verifier: Callable[[Mapping[str, Any]], object], claim_journal: Any,
    claim_timeout_s: float, c6_ready_timeout_s: float = 120.0,
) -> tuple[ExecutionCapture, dict[str, Any], dict[str, Any],
           dict[str, Any], Mapping[str, Any]]:
    """Run the oracle and three candidate legs as distinct confined processes.

    The oracle process generates inputs and the f32/f64 references under one
    held claim.  The evaluator then materializes the exact witness bytes as
    hardlink-safe candidate inputs, and every candidate leg runs as its own
    process whose compute gate the evaluator arms only after the leg proves it
    reached the gate with those exact inputs.  Targeted correctness follows in
    the same held claim; the sealed C6 receipt precedes it.
    """
    invocations = (oracle_invocation, *candidate_invocations,
                   targeted_invocation)
    if (len(candidate_invocations) != 3
            or any(item.kind != "correctness" or item.arm != "candidate"
                   for item in invocations)):
        raise EvidenceProducerError(
            "claimed split C6 requires an oracle, three candidate legs, "
            "and one targeted command")
    _verify_plan_files(plan)
    output_paths = [path for invocation in invocations
                    for path in (invocation.stdout_path, invocation.stderr_path)]
    for invocation in candidate_invocations:
        paths = _c6_candidate_paths_from_argv(invocation.argv)
        output_paths.extend(
            (paths["output"], paths["ready_file"], paths["continue_file"]))
    if len(output_paths) != len(set(output_paths)) or any(
            path.exists() or path.is_symlink() for path in output_paths):
        raise EvidenceProducerError("sequence outputs must be fresh unique paths")
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    claim = None
    borrowed_outer = False
    opened = released = None
    targeted_capture: ExecutionCapture | None = None
    c6_receipt: Mapping[str, Any] | None = None
    targeted_residency: dict[str, Any] | None = None
    verified_before = verified_after = False
    failure: BaseException | None = None
    try:
        claim = claim_acquirer(
            plan.device_id,
            purpose="AutoKernel GPU source evidence correctness/candidate",
            campaign_id=plan.campaign_id, journal=claim_journal,
            holder_label="gpu_source_evidence.py", timeout_s=claim_timeout_s,
            max_hold_s=3600.0)
        borrowed_outer = bool(getattr(claim, "borrowed_outer_reservation", False))
        opened = _receipt_dict(claim.receipt(), "opened claim")
        verified_before = _check_result_passed(claim_verifier(opened))
        if not verified_before:
            raise EvidenceProducerError(
                "device claim was not verifiably held before execution")
        oracle_capture = executor(oracle_invocation)
        if (not isinstance(oracle_capture, ExecutionCapture)
                or oracle_capture.argv != oracle_invocation.argv):
            raise EvidenceProducerError(
                "executor did not attest the exact native C6 oracle argv")
        if oracle_capture.exit_code != 0:
            raise EvidenceProducerError(
                "native C6 oracle command exited nonzero")
        oracle_residency = _residency(oracle_capture, plan.device_id)
        oracle_residency.update({
            "claim_verified_before": verified_before,
            "claim_verified_after": _check_result_passed(
                claim_verifier(opened)),
            "device_claim_mode": (
                "borrowed_outer_reservation" if borrowed_outer
                else "direct_device_claim"),
            "outer_claim_id": opened["claim_id"] if borrowed_outer else None,
        })
        if oracle_residency["claim_verified_after"] is not True:
            raise EvidenceProducerError(
                "device claim was not held after native C6 oracle execution")
        if not oracle_sidecar.exists() or oracle_sidecar.is_symlink():
            raise EvidenceProducerError(
                "native C6 oracle command did not emit its sidecar")
        c6_receipt = _evaluate_c6_oracle_and_candidate_outputs(
            oracle_sidecar, input_dir, oracle_invocation, oracle_capture,
            oracle_residency, candidate_invocations,
            plan=plan, executor=executor, opened=opened,
            c6_ready_timeout_s=c6_ready_timeout_s)
        _validate_c6_correctness_receipt(c6_receipt, plan)
        _verify_plan_files(plan)
        if not _check_result_passed(claim_verifier(opened)):
            raise EvidenceProducerError(
                "device claim was not held after the sealed C6 boundary")
        targeted_capture = executor(targeted_invocation)
        if (not isinstance(targeted_capture, ExecutionCapture)
                or targeted_capture.argv != targeted_invocation.argv):
            raise EvidenceProducerError(
                "executor did not attest the exact targeted correctness argv")
        verified_after = _check_result_passed(claim_verifier(opened))
        if not verified_after:
            raise EvidenceProducerError(
                "device claim was not verifiably held after correctness sequence")
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
        raise EvidenceProducerError(
            f"correctness sequence execution failed: {failure}") from failure
    if (opened is None or released is None or targeted_capture is None
            or c6_receipt is None):
        raise EvidenceProducerError(
            "claimed correctness sequence did not produce complete evidence")
    _validate_claim_pair(opened, released, plan=plan)
    targeted_residency = _residency(targeted_capture, plan.device_id)
    targeted_residency.update({
        "claim_verified_before": verified_before,
        "claim_verified_after": verified_after,
        "device_claim_mode": (
            "borrowed_outer_reservation" if borrowed_outer
            else "direct_device_claim"),
        "outer_claim_id": opened["claim_id"] if borrowed_outer else None,
    })
    return (targeted_capture, opened, released, targeted_residency,
            c6_receipt)


def _materialize_c6_argv(argv: tuple[str, ...], sidecar: Path) -> tuple[str, ...]:
    if not sidecar.is_absolute() or argv.count(C6_SIDECAR_OUTPUT) != 1:
        raise EvidenceProducerError("C6 sidecar substitution is invalid")
    return tuple(str(sidecar) if item == C6_SIDECAR_OUTPUT else item
                 for item in argv)


def _materialize_c6_oracle_argv(argv: tuple[str, ...], sidecar: Path) -> tuple[str, ...]:
    if _c6_mode_from_argv(argv) != C6_MODE_ORACLE:
        raise EvidenceProducerError("C6 oracle substitution requires oracle mode")
    return _materialize_c6_argv(argv, sidecar)


def _decode_f32le_hex(value: object, label: str) -> tuple[bytes, list[float]]:
    if not isinstance(value, str) or len(value) == 0 or len(value) % 8:
        raise EvidenceProducerError(f"{label} is not non-empty f32le hex")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise EvidenceProducerError(f"{label} is not hex") from exc
    return raw, list(struct.unpack("<" + "f" * (len(raw) // 4), raw))


def _decode_f64le_hex(value: object, label: str) -> tuple[bytes, list[float]]:
    if not isinstance(value, str) or len(value) == 0 or len(value) % 16:
        raise EvidenceProducerError(f"{label} is not non-empty f64le hex")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise EvidenceProducerError(f"{label} is not hex") from exc
    return raw, list(struct.unpack("<" + "d" * (len(raw) // 8), raw))


def _validate_c6_sidecar_identity(
        value: object, case_identity: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "schema": C6_SIDECAR_SCHEMA,
        "sequence": ["reference", "candidate-1", "candidate-2", "candidate-3"],
        **{key: case_identity[key] for key in (
            "backend", "operation", "type_a", "m", "n", "k", "seed")},
        "type_b": "f32",
        "output_dtype": "f32",
        "candidate_clone_ids": ["candidate-1", "candidate-2", "candidate-3"],
    }
    payload_keys = {
        "input_witness", "output_elements", "reference_output_f32le_hex",
        "reference_output_f64le_hex",
        "candidate_outputs_f32le_hex"}
    if (not isinstance(value, Mapping)
            or set(value) != set(expected) | payload_keys
            or any(type(value.get(key)) is not type(item)
                   or value.get(key) != item for key, item in expected.items())):
        raise EvidenceProducerError(
            f"{label} identity/ordering/clone contract changed")


def _decode_c6_sidecar_payload(
        value: Mapping[str, Any], case_identity: Mapping[str, Any], *,
        label: str,
) -> tuple[list[bytes], bytes, list[float], bytes, list[float],
           list[bytes], list[list[float]]]:
    witness = value.get("input_witness")
    operation = case_identity["operation"]
    expected_witness = {
        "MUL_MAT": {"weights_hex", "activations_f32le_hex"},
        "RMS_NORM": {"activations_f32le_hex", "scale_f32le_hex"},
        "FLASH_ATTN_EXT": {
            "query_f32le_hex", "key_f16le_hex", "value_f16le_hex"},
    }[operation]
    if not isinstance(witness, Mapping) or set(witness) != expected_witness:
        raise EvidenceProducerError(f"{label} input witness is incomplete")
    ordered_inputs: list[bytes] = []
    for key in sorted(expected_witness):
        encoded = witness[key]
        if (not isinstance(encoded, str) or not encoded
                or len(encoded) % 2):
            raise EvidenceProducerError(f"{label} input witness is not hex")
        try:
            ordered_inputs.append(bytes.fromhex(encoded))
        except ValueError as exc:
            raise EvidenceProducerError(
                f"{label} input witness is not hex") from exc
    by_name = dict(zip(sorted(expected_witness), ordered_inputs))
    m, n, k = (case_identity[name] for name in ("m", "n", "k"))
    if operation == "MUL_MAT":
        lengths_ok = (len(by_name["weights_hex"]) > 0
                      and len(by_name["activations_f32le_hex"]) == k*n*4)
        output_elements = m*n
    elif operation == "RMS_NORM":
        lengths_ok = (case_identity["type_a"] == "f32" and k == 1
                      and len(by_name["activations_f32le_hex"]) == m*n*4
                      and len(by_name["scale_f32le_hex"]) == m*4)
        output_elements = m*n
    else:
        lengths_ok = (case_identity["type_a"] == "f16" and m == 64 and n == 1
                      and len(by_name["query_f32le_hex"]) == m*n*14*4
                      and len(by_name["key_f16le_hex"]) == m*k*2*2
                      and len(by_name["value_f16le_hex"]) == m*k*2*2)
        output_elements = m*n*14
    if not lengths_ok or value.get("output_elements") != output_elements:
        raise EvidenceProducerError(f"{label} input/output shape is inconsistent")
    reference_raw, reference = _decode_f32le_hex(
        value.get("reference_output_f32le_hex"), f"{label} reference")
    reference_f64_raw, reference_f64 = _decode_f64le_hex(
        value.get("reference_output_f64le_hex"),
        f"{label} float64 reference")
    output_values = value.get("candidate_outputs_f32le_hex")
    if not isinstance(output_values, list) or len(output_values) != 3:
        raise EvidenceProducerError(
            f"{label} requires exactly three candidate outputs")
    decoded = [_decode_f32le_hex(item, f"{label} candidate {index}")
               for index, item in enumerate(output_values, 1)]
    candidate_raw = [item[0] for item in decoded]
    candidates = [item[1] for item in decoded]
    if any(len(row) != output_elements
           for row in [reference, reference_f64, *candidates]):
        raise EvidenceProducerError(f"{label} output shape is inconsistent")
    return (ordered_inputs, reference_raw, reference,
            reference_f64_raw, reference_f64, candidate_raw, candidates)


def _validate_c6_oracle_sidecar_identity(
        value: object, case_identity: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "schema": C6_ORACLE_SIDECAR_SCHEMA,
        **{key: case_identity[key] for key in (
            "backend", "operation", "type_a", "m", "n", "k", "seed")},
    }
    payload_keys = {
        "input_witness", "output_elements", "reference_output_f32le_hex",
        "reference_output_f64le_hex"}
    if (not isinstance(value, Mapping)
            or set(value) != set(expected) | payload_keys
            or any(type(value.get(key)) is not type(item)
                   or value.get(key) != item for key, item in expected.items())):
        raise EvidenceProducerError(
            f"{label} oracle identity contract changed")


def _decode_c6_oracle_payload(
        value: Mapping[str, Any], case_identity: Mapping[str, Any], *,
        label: str,
) -> tuple[dict[str, bytes], bytes, list[float], bytes, list[float]]:
    """Decode the oracle phase into by-name inputs plus f32/f64 references."""
    witness = value.get("input_witness")
    operation = case_identity["operation"]
    expected_witness = {
        "MUL_MAT": {"weights_hex", "activations_f32le_hex"},
        "RMS_NORM": {"activations_f32le_hex", "scale_f32le_hex"},
        "FLASH_ATTN_EXT": {
            "query_f32le_hex", "key_f16le_hex", "value_f16le_hex"},
    }[operation]
    if not isinstance(witness, Mapping) or set(witness) != expected_witness:
        raise EvidenceProducerError(f"{label} input witness is incomplete")
    by_name: dict[str, bytes] = {}
    for key in sorted(expected_witness):
        encoded = witness[key]
        if (not isinstance(encoded, str) or not encoded
                or len(encoded) % 2):
            raise EvidenceProducerError(f"{label} input witness is not hex")
        try:
            by_name[key.removesuffix("_hex")] = bytes.fromhex(encoded)
        except ValueError as exc:
            raise EvidenceProducerError(
                f"{label} input witness is not hex") from exc
    m, n, k = (case_identity[name] for name in ("m", "n", "k"))
    if operation == "MUL_MAT":
        lengths_ok = (len(by_name["weights"]) > 0
                      and len(by_name["activations_f32le"]) == k*n*4)
        output_elements = m*n
    elif operation == "RMS_NORM":
        lengths_ok = (case_identity["type_a"] == "f32" and k == 1
                      and len(by_name["activations_f32le"]) == m*n*4
                      and len(by_name["scale_f32le"]) == m*4)
        output_elements = m*n
    else:
        lengths_ok = (case_identity["type_a"] == "f16" and m == 64 and n == 1
                      and len(by_name["query_f32le"]) == m*n*14*4
                      and len(by_name["key_f16le"]) == m*k*2*2
                      and len(by_name["value_f16le"]) == m*k*2*2)
        output_elements = m*n*14
    if not lengths_ok or value.get("output_elements") != output_elements:
        raise EvidenceProducerError(f"{label} input/output shape is inconsistent")
    reference_raw, reference = _decode_f32le_hex(
        value.get("reference_output_f32le_hex"), f"{label} reference")
    reference_f64_raw, reference_f64 = _decode_f64le_hex(
        value.get("reference_output_f64le_hex"),
        f"{label} float64 reference")
    if any(len(row) != output_elements
           for row in (reference, reference_f64)):
        raise EvidenceProducerError(f"{label} output shape is inconsistent")
    return by_name, reference_raw, reference, reference_f64_raw, reference_f64


_C6_INPUT_ORDER = {
    "MUL_MAT": ("weights", "activations_f32le"),
    "RMS_NORM": ("activations_f32le", "scale_f32le"),
    "FLASH_ATTN_EXT": ("query_f32le", "key_f16le", "value_f16le"),
}


def _seal_c6_token_file(path: Path, token: bytes, *, label: str) -> str:
    """Write one handshake token file with hardlink-safe secrecy."""
    if path.exists() or path.is_symlink():
        raise EvidenceProducerError(f"{label} path is not fresh")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        facts = os.fstat(descriptor)
        if (not stat.S_ISREG(facts.st_mode) or facts.st_nlink != 1
                or facts.st_uid != os.geteuid() or facts.st_mode & 0o077):
            raise EvidenceProducerError(f"{label} file is not hardlink-safe")
        os.write(descriptor, token)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _hash_file(path, label)


def _write_c6_input_binding(
        input_dir: Path, by_name: Mapping[str, bytes],
        case_identity: Mapping[str, Any], *, label: str) -> tuple[Path, dict[str, Any]]:
    """Materialize the oracle witness as hardlink-safe candidate inputs.

    Each input is a fresh 0600 regular file created O_EXCL without following
    any final symlink, fsynced, then re-verified by inode identity.  The
    binding manifest seals every per-file digest plus the canonical ordered
    input identity so the candidate's loads are re-derivable at reopen.
    """
    operation = case_identity["operation"]
    order = _C6_INPUT_ORDER.get(operation)
    if order is None or any(name not in by_name for name in order):
        raise EvidenceProducerError(
            f"{label} input witness lacks the canonical input set")
    if (input_dir.is_symlink() or not input_dir.is_absolute()
            or input_dir != input_dir.resolve(strict=False)):
        raise EvidenceProducerError(f"{label} input directory is not canonical")
    input_dir.mkdir(mode=0o700, exist_ok=False)
    files: list[dict[str, Any]] = []
    for name in order:
        path = input_dir / f"{name}.bin"
        digest = _seal_c6_token_file(path, by_name[name], label=f"{label} input {name}")
        files.append({
            "name": name, "path": str(path), "sha256": digest,
            "length": len(by_name[name])})
    manifest = {
        "schema": C6_INPUT_BINDING_SCHEMA,
        "operation": operation,
        "input_dir": str(input_dir),
        "input_identity_sha256": hashlib.sha256(
            b"".join(by_name[name] for name in order)).hexdigest(),
        "files": files,
    }
    manifest_path = input_dir.parent / "c6-input-binding.json"
    manifest_digest = _seal_c6_token_file(
        manifest_path,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        + b"\n", label=f"{label} input binding")
    return manifest_path, {"sha256": manifest_digest, "manifest": manifest}


def _reopen_c6_input_binding(
        manifest_path: Path, by_name: Mapping[str, bytes],
        case_identity: Mapping[str, Any], *, label: str) -> None:
    """Re-verify the materialized inputs and binding against the oracle witness."""
    if (manifest_path.is_symlink() or not manifest_path.is_absolute()
            or manifest_path != manifest_path.resolve(strict=False)):
        raise EvidenceProducerError(f"{label} input binding path is not canonical")
    try:
        value = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError(
            f"{label} input binding is unreadable") from exc
    operation = case_identity["operation"]
    order = _C6_INPUT_ORDER.get(operation)
    if (not isinstance(value, Mapping)
            or not isinstance(value.get("input_dir"), str)
            or not isinstance(value.get("files"), list)
            or any(not isinstance(item, Mapping) for item in value["files"])):
        raise EvidenceProducerError(f"{label} input binding manifest changed")
    input_dir = Path(value["input_dir"])
    if (value.get("schema") != C6_INPUT_BINDING_SCHEMA
            or value.get("operation") != operation
            or input_dir != input_dir.resolve(strict=False)
            or input_dir.is_symlink()
            or value.get("input_identity_sha256") != hashlib.sha256(
                b"".join(by_name[name] for name in order)).hexdigest()
            or [item.get("name") for item in value["files"]] != list(order)):
        raise EvidenceProducerError(
            f"{label} input binding manifest changed")
    for item, name in zip(value["files"], order):
        if (not isinstance(item, Mapping)
                or item.get("path") != str(input_dir / f"{name}.bin")
                or not isinstance(item.get("sha256"), str)
                or item.get("length") != len(by_name[name])):
            raise EvidenceProducerError(
                f"{label} input binding file record changed")
        path = Path(str(item["path"]))
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or _hash_file(path, f"{label} input {name}") != item["sha256"]
                or hashlib.sha256(by_name[name]).hexdigest() != item["sha256"]):
            raise EvidenceProducerError(
                f"{label} input binding file changed")


def _paced_candidate(
        executor: CommandExecutor, invocation: CommandInvocation,
        ready_path: Path, continue_path: Path, *, ready_timeout_s: float = 120.0,
) -> tuple[ExecutionCapture, dict[str, Any]]:
    """Execute one candidate leg under an evaluator-armed continue gate.

    The native candidate writes a ready token only after it has loaded the
    exact handed-over inputs and reached the compute gate; it then waits for
    the continue token.  This wrapper arms the continue token strictly after
    observing the ready token, so every leg journals one coherent event
    stream: launched <= ready <= continue <= completed.  A candidate that
    never reaches the gate fails closed with its own exit/stderr evidence.
    """
    if ready_timeout_s <= 0 or not math.isfinite(ready_timeout_s):
        raise EvidenceProducerError("candidate pacing timeout must be finite")
    if ready_path.exists() or ready_path.is_symlink() \
            or continue_path.exists() or continue_path.is_symlink():
        raise EvidenceProducerError("candidate handshake paths are not fresh")
    launched_ns = time.monotonic_ns()
    result: dict[str, Any] = {}
    failures: list[BaseException] = []

    def run() -> None:
        try:
            result["capture"] = executor(invocation)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            failures.append(exc)

    thread = threading.Thread(target=run, name="c6-paced-candidate", daemon=True)
    thread.start()
    ready_observed_ns: int | None = None
    continue_written_ns: int | None = None
    deadline = launched_ns + int(ready_timeout_s * 1e9)
    while ready_observed_ns is None:
        if time.monotonic_ns() > deadline:
            break
        if not ready_path.is_symlink() and ready_path.is_file():
            try:
                content = ready_path.read_bytes()
            except OSError:
                content = b""
            if content == C6_READY_TOKEN:
                ready_observed_ns = time.monotonic_ns()
                break
        time.sleep(0.01)
    if ready_observed_ns is not None:
        continue_written_ns = time.monotonic_ns()
        _seal_c6_token_file(
            continue_path, C6_CONTINUE_TOKEN, label="candidate continue")
    thread.join(timeout=max(60.0, ready_timeout_s))
    if thread.is_alive():
        raise EvidenceProducerError(
            "candidate pacing thread did not terminate after timeout")
    if failures:
        raise EvidenceProducerError(
            "candidate leg executor failed") from failures[0]
    capture = result.get("capture")
    if not isinstance(capture, ExecutionCapture):
        raise EvidenceProducerError("candidate leg produced no capture")
    completed_ns = time.monotonic_ns()
    if ready_observed_ns is None:
        raise EvidenceProducerError(
            "candidate ready token was not observed within the pacing window")
    if (continue_written_ns is None or not (launched_ns
            <= ready_observed_ns <= continue_written_ns <= completed_ns)):
        raise EvidenceProducerError(
            "candidate handshake event stream is not monotonic")
    return capture, {
        "launched_monotonic_ns": launched_ns,
        "ready_observed_monotonic_ns": ready_observed_ns,
        "continue_written_monotonic_ns": continue_written_ns,
        "completed_monotonic_ns": completed_ns,
    }


def _evaluate_c6_oracle_and_candidate_outputs(
        oracle_sidecar: Path, input_dir: Path,
        oracle_invocation: CommandInvocation, oracle_capture: ExecutionCapture,
        oracle_residency: Mapping[str, Any],
        candidate_invocations: tuple[CommandInvocation, ...], *,
        plan: GpuSourceEvidencePlan, executor: CommandExecutor,
        opened: Mapping[str, Any], c6_ready_timeout_s: float) -> Mapping[str, Any]:
    """Seal one split-mode C6 receipt from oracle and paced candidate legs."""
    c6 = plan.c6_correctness
    if c6 is None:
        raise EvidenceProducerError("production correctness lacks C6 capability")
    if _c6_mode_from_argv(c6.argv) != C6_MODE_ORACLE:
        raise EvidenceProducerError("split C6 evaluation requires oracle mode")
    try:
        raw = oracle_sidecar.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError("native C6 oracle sidecar is unreadable") from exc
    case_identity = _c6_case_identity_from_argv(c6.argv)
    _validate_c6_oracle_sidecar_identity(
        value, case_identity, label="native C6 oracle")
    if plan.correctness_op != case_identity["operation"]:
        raise EvidenceProducerError(
            "native C6 operation differs from targeted correctness")
    by_name, reference_raw, reference, reference_f64_raw, reference_f64 = (
        _decode_c6_oracle_payload(value, case_identity, label="native C6"))
    operation = case_identity["operation"]
    output_elements = (
        case_identity["m"] * case_identity["n"] * 14
        if operation == "FLASH_ATTN_EXT"
        else case_identity["m"] * case_identity["n"])
    manifest_path, input_binding = _write_c6_input_binding(
        input_dir, by_name, case_identity, label="native C6")
    leg_bindings: list[dict[str, Any]] = []
    raw_outputs: list[bytes] = []
    for index, invocation in enumerate(candidate_invocations, 1):
        paths = _c6_candidate_paths_from_argv(invocation.argv)
        if paths["input_dir"] != input_dir:
            raise EvidenceProducerError(
                "candidate leg input directory differs from the handover")
        capture, pacing = _paced_candidate(
            executor, invocation, paths["ready_file"],
            paths["continue_file"], ready_timeout_s=c6_ready_timeout_s)
        if capture.argv != invocation.argv:
            raise EvidenceProducerError(
                "executor did not attest the exact candidate leg argv")
        if capture.exit_code != 0:
            raise EvidenceProducerError(
                f"native C6 candidate leg {index} exited nonzero")
        leg_residency = _residency(capture, plan.device_id)
        if paths["output"].is_symlink() or not paths["output"].is_file() \
                or paths["output"].stat().st_nlink != 1:
            raise EvidenceProducerError(
                f"candidate leg {index} output is not a sealed regular file")
        output_bytes = paths["output"].read_bytes()
        if len(output_bytes) != output_elements * 4:
            raise EvidenceProducerError(
                f"candidate leg {index} output length is inconsistent")
        leg_bindings.append({
            "schema": C6_LEG_BINDING_SCHEMA,
            "leg_index": index,
            "child_pid": capture.child_pid,
            "exit_code": capture.exit_code,
            "argv": list(invocation.argv),
            "argv_sha256": hashlib.sha256(
                json.dumps(list(invocation.argv), sort_keys=True).encode()
            ).hexdigest(),
            "input_dir": str(paths["input_dir"]),
            "stdout_path": str(invocation.stdout_path),
            "stdout_sha256": _hash_file(invocation.stdout_path, f"C6 leg {index} stdout"),
            "stderr_path": str(invocation.stderr_path),
            "stderr_sha256": _hash_file(
                invocation.stderr_path, f"C6 leg {index} stderr",
                allow_empty=True),
            "output_path": str(paths["output"]),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "output_length": len(output_bytes),
            "ready_path": str(paths["ready_file"]),
            "ready_sha256": _hash_file(
                paths["ready_file"], f"C6 leg {index} ready"),
            "continue_path": str(paths["continue_file"]),
            "continue_sha256": _hash_file(
                paths["continue_file"], f"C6 leg {index} continue"),
            "event_stream": pacing,
            "residency": {
                "overlap_sample_count": leg_residency["overlap_sample_count"],
                "kfd_pids": leg_residency["kfd_pids"],
                "max_vram_bytes": leg_residency["max_vram_bytes"],
            },
        })
        raw_outputs.append(output_bytes)
    policy = c6_reward_integrity.PrecisionContract(**dict(c6.precision_contract))
    structural = c6_reward_integrity.StructuralPrecisionEvidence(
        **dict(c6.structural_precision_evidence))
    numerical = [c6_reward_integrity.evaluate_numerics(
        reference, candidate, structural=structural, policy=policy)
        for candidate in [list(struct.unpack(
            f"<{output_elements}f", item)) for item in raw_outputs]]
    precision_policy = hawkeye_measurement.PrecisionEquivalencePolicy(
        **dict(c6.precision_equivalence_policy))
    precision_equivalence = [hawkeye_measurement.evaluate_precision_equivalence(
        reference_f64, candidate, policy=precision_policy,
        observed_output_dtype=structural.output_dtype,
        observed_accumulator_dtype=structural.accumulator_dtype)
        for candidate in [list(struct.unpack(
            f"<{output_elements}f", item)) for item in raw_outputs]]
    determinism = c6_reward_integrity.determinism_from_recorded_outputs(
        raw_outputs)
    semantic = c6_reward_integrity.calibrate_semantic_judge(
        c6.semantic_judge_verdicts)
    if (oracle_capture.exit_code != 0
            or not all(row.correct for row in numerical)
            or not all(row.correct for row in precision_equivalence)
            or not determinism.correct or semantic.gating):
        raise EvidenceProducerError(
            "native C6 split correctness/determinism/semantic policy refused")
    _verify_plan_files(plan)
    dispatch_sha = schemas.content_hash(_expectations(plan))
    ordered_inputs = [by_name[name] for name in _C6_INPUT_ORDER[operation]]
    body = {
        "schema": C6_CORRECTNESS_SCHEMA, "authority": AUTHORITY,
        "status": "complete", "result": "PASS", "non_promotable": True,
        "promotion_claim": False, "campaign_id": plan.campaign_id,
        "device_id": plan.device_id, "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "post_run_compiled_source_sha256": plan.candidate.source_sha256,
        "command_argv": list(oracle_invocation.argv),
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in oracle_invocation.environment]),
        "exit_code": oracle_capture.exit_code,
        "c6_process_mode": "oracle_candidate_split",
        "oracle_output": {
            "stdout_path": str(oracle_invocation.stdout_path),
            "stdout_sha256": _hash_file(
                oracle_invocation.stdout_path, "C6 oracle stdout"),
            "stderr_path": str(oracle_invocation.stderr_path),
            "stderr_sha256": _hash_file(
                oracle_invocation.stderr_path, "C6 oracle stderr",
                allow_empty=True),
        },
        "sidecar": {"path": str(oracle_sidecar),
                    "sha256": hashlib.sha256(raw).hexdigest()},
        "seeded_case_identity": case_identity,
        "input_binding": {
            "path": str(manifest_path), "sha256": input_binding["sha256"],
            "input_identity_sha256":
                input_binding["manifest"]["input_identity_sha256"],
        },
        "per_leg_bindings": leg_bindings,
        "input_identity_sha256": hashlib.sha256(
            b"".join(ordered_inputs)).hexdigest(),
        "reference_output_sha256": hashlib.sha256(reference_raw).hexdigest(),
        "reference_float64_output_sha256": hashlib.sha256(
            reference_f64_raw).hexdigest(),
        "candidate_output_sha256": [hashlib.sha256(row).hexdigest()
                                     for row in raw_outputs],
        "precision_contract": asdict(policy),
        "precision_equivalence_policy": asdict(precision_policy),
        "precision_equivalence": [hawkeye_measurement.serialize_carrier(row)
                                    for row in precision_equivalence],
        "structural_precision_evidence": asdict(structural),
        "numeric_verdicts": [asdict(row) for row in numerical],
        "determinism": json.loads(json.dumps(asdict(determinism))),
        "semantic_judge_calibration": json.loads(json.dumps(asdict(semantic))),
        "semantic_judge_gating": False,
        "native_execution": True, "wrapper_used": False,
        "dispatch_expectations_sha256": dispatch_sha,
        "c6_inputs": [_bound_reference(item) for item in c6.inputs],
        "started_at": oracle_capture.started_at,
        "ended_at": oracle_capture.ended_at,
        **_open_claim_boundary_fields(opened, oracle_residency),
        "residency_witness": dict(oracle_residency),
    }
    return _seal(oracle_sidecar.parent / "c6-receipt.json", body)


def _evaluate_c6_sidecar(
        sidecar: Path, plan: GpuSourceEvidencePlan,
        invocation: CommandInvocation, capture: ExecutionCapture,
        opened: Mapping[str, Any], residency: Mapping[str, Any]) -> Mapping[str, Any]:
    c6 = plan.c6_correctness
    if c6 is None:
        raise EvidenceProducerError("production correctness lacks C6 capability")
    try:
        raw = sidecar.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError("native C6 sidecar is unreadable") from exc
    case_identity = _c6_case_identity_from_argv(c6.argv)
    _validate_c6_sidecar_identity(value, case_identity, label="native C6 sidecar")
    if plan.correctness_op != case_identity["operation"]:
        raise EvidenceProducerError(
            "native C6 operation differs from targeted correctness")
    (input_raw, reference_raw, reference, reference_f64_raw, reference_f64,
     candidate_raw, candidates) = (
        _decode_c6_sidecar_payload(
            value, case_identity, label="native C6"))
    policy = c6_reward_integrity.PrecisionContract(**dict(c6.precision_contract))
    structural = c6_reward_integrity.StructuralPrecisionEvidence(
        **dict(c6.structural_precision_evidence))
    numerical = [c6_reward_integrity.evaluate_numerics(
        reference, candidate, structural=structural, policy=policy)
        for candidate in candidates]
    precision_policy = hawkeye_measurement.PrecisionEquivalencePolicy(
        **dict(c6.precision_equivalence_policy))
    precision_equivalence = [hawkeye_measurement.evaluate_precision_equivalence(
        reference_f64, candidate, policy=precision_policy,
        observed_output_dtype=structural.output_dtype,
        observed_accumulator_dtype=structural.accumulator_dtype)
        for candidate in candidates]
    determinism = c6_reward_integrity.determinism_from_recorded_outputs(
        candidate_raw)
    semantic = c6_reward_integrity.calibrate_semantic_judge(
        c6.semantic_judge_verdicts)
    if (capture.exit_code != 0 or not all(row.correct for row in numerical)
            or not all(row.correct for row in precision_equivalence)
            or not determinism.correct or semantic.gating):
        raise EvidenceProducerError(
            "native C6 correctness/determinism/semantic policy refused candidate")
    # This check intentionally occurs after native execution and before the
    # receipt is sealed.  It reopens the complete file-backed build/source
    # identity plus every C6-linked library; a candidate cannot replace its
    # compiled closure during the run and retain the pre-run identity.
    _verify_plan_files(plan)
    dispatch_sha = schemas.content_hash(_expectations(plan))
    body = {
        "schema": C6_CORRECTNESS_SCHEMA, "authority": AUTHORITY,
        "status": "complete", "result": "PASS", "non_promotable": True,
        "promotion_claim": False, "campaign_id": plan.campaign_id,
        "device_id": plan.device_id, "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "post_run_compiled_source_sha256": plan.candidate.source_sha256,
        "command_argv": list(invocation.argv),
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in invocation.environment]),
        "exit_code": capture.exit_code,
        **_output_hashes(invocation),
        "sidecar": {"path": str(sidecar),
                    "sha256": hashlib.sha256(raw).hexdigest()},
        "seeded_case_identity": case_identity,
        "input_identity_sha256": hashlib.sha256(b"".join(input_raw)).hexdigest(),
        "reference_output_sha256": hashlib.sha256(reference_raw).hexdigest(),
        "reference_float64_output_sha256": hashlib.sha256(
            reference_f64_raw).hexdigest(),
        "candidate_output_sha256": [hashlib.sha256(row).hexdigest()
                                     for row in candidate_raw],
        "precision_contract": asdict(policy),
        "precision_equivalence_policy": asdict(precision_policy),
        "precision_equivalence": [hawkeye_measurement.serialize_carrier(row)
                                    for row in precision_equivalence],
        "structural_precision_evidence": asdict(structural),
        "numeric_verdicts": [asdict(row) for row in numerical],
        "determinism": json.loads(json.dumps(asdict(determinism))),
        "semantic_judge_calibration": json.loads(json.dumps(asdict(semantic))),
        "semantic_judge_gating": False,
        "native_execution": True, "wrapper_used": False,
        "dispatch_expectations_sha256": dispatch_sha,
        "c6_inputs": [_bound_reference(item) for item in c6.inputs],
        "started_at": capture.started_at, "ended_at": capture.ended_at,
        **_open_claim_boundary_fields(opened, residency),
        "residency_witness": dict(residency),
    }
    return _seal(sidecar.parent / "c6-receipt.json", body)


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


def _parse_correctness(
        stdout: str, plan: GpuSourceEvidencePlan,
        invocation_contract: Mapping[str, Any] | None = None,
) -> _CorrectnessResult:
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

    backend = (plan.correctness_backend if invocation_contract is None
               else invocation_contract["backend"])
    operation = (plan.correctness_op if invocation_contract is None
                 else invocation_contract["op"])
    expected = (plan.expected_correctness_cases if invocation_contract is None
                else invocation_contract["expected_cases"])
    targets = tuple(row for row in run.backends if row.name == backend)
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
    if any(case.op != operation for case in compared):
        raise CorrectnessParseRefusal(
            "target correctness backend exercised an unexpected operation")
    if any(not case.passed for case in compared):
        raise CorrectnessParseRefusal(
            "target correctness backend contains a failed case")
    if (len(compared) != expected
            or target.reported_passed != expected
            or target.reported_total != expected):
        raise CorrectnessParseRefusal(
            "correctness did not pass the exact expected case count")
    if invocation_contract is not None:
        required = invocation_contract.get("required_cases", [])
        for requirement in required:
            pattern = requirement.get("params_pattern")
            expected_matches = requirement.get("expected_matches")
            if (not isinstance(pattern, str) or not pattern
                    or isinstance(expected_matches, bool)
                    or not isinstance(expected_matches, int)
                    or expected_matches < 1):
                raise CorrectnessParseRefusal(
                    "required correctness case matcher is malformed")
            matches = [case for case in target.cases
                       if case.op == requirement.get("op")
                       and re.search(pattern, case.params)]
            if len(matches) != expected_matches:
                raise CorrectnessParseRefusal(
                    "required correctness case was absent or duplicated")
            if any(case.status == "not_supported" or not case.passed
                   for case in matches):
                raise CorrectnessParseRefusal(
                    "required correctness case was unsupported or failed")

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
        backend=backend,
        operation=operation,
        passed_cases=expected,
        total_cases=expected,
        skipped_backends=tuple(row.name for row in others),
        backends_passed=run.backends_passed,
        backends_total=run.backends_total,
        overall=run.overall)


def _produce_correctness(
    root: Path, plan: GpuSourceEvidencePlan, executor: CommandExecutor, *,
    claim_acquirer: Callable[..., Any], claim_verifier: Callable[[Mapping[str, Any]], object],
    claim_journal: Any, claim_timeout_s: float, c6_ready_timeout_s: float = 120.0,
) -> Mapping[str, Any]:
    if plan.correctness_invocations:
        return _produce_correctness_invocations(
            root, plan, executor, claim_acquirer=claim_acquirer,
            claim_verifier=claim_verifier, claim_journal=claim_journal,
            claim_timeout_s=claim_timeout_s)
    directory = root / "correctness"
    c6 = plan.c6_correctness
    if c6 is None:
        raise EvidenceProducerError("production correctness lacks C6 capability")
    invocation = CommandInvocation(
        kind="correctness", arm="candidate", argv=plan.correctness_argv,
        stdout_path=(directory / "stdout.txt").resolve(),
        stderr_path=(directory / "stderr.txt").resolve(),
        working_directory=plan.execution_cwd,
        environment=plan.correctness_environment)
    c6_receipt_path = directory / "c6-receipt.json"
    if _c6_mode_from_argv(c6.argv) == C6_MODE_ORACLE:
        oracle_sidecar = (directory / "c6-oracle-sidecar.json").resolve()
        oracle_invocation = CommandInvocation(
            kind="correctness", arm="candidate",
            argv=_materialize_c6_oracle_argv(c6.argv, oracle_sidecar),
            stdout_path=(directory / "c6-oracle-stdout.txt").resolve(),
            stderr_path=(directory / "c6-oracle-stderr.txt").resolve(),
            working_directory=plan.execution_cwd,
            environment=plan.correctness_environment)
        input_dir = (directory / "c6-inputs").resolve()
        candidate_invocations = tuple(
            CommandInvocation(
                kind="correctness", arm="candidate",
                argv=_c6_candidate_argv_from_argv(
                    c6.argv, input_dir=input_dir,
                    output=(directory / f"c6-candidate-{index}-output.bin")
                           .resolve(),
                    ready_file=(directory / f"c6-candidate-{index}-ready")
                               .resolve(),
                    continue_file=(
                        directory / f"c6-candidate-{index}-continue").resolve()),
                stdout_path=(directory / f"c6-candidate-{index}-stdout.txt")
                            .resolve(),
                stderr_path=(directory / f"c6-candidate-{index}-stderr.txt")
                            .resolve(),
                working_directory=plan.execution_cwd,
                environment=plan.correctness_environment)
            for index in (1, 2, 3))
        if c6_receipt_path.exists() or c6_receipt_path.is_symlink():
            try:
                c6_receipt = proofs.load_receipt(
                    c6_receipt_path, schema=C6_CORRECTNESS_SCHEMA)
            except proofs.ProofError as exc:
                raise EvidenceProducerError(
                    "sealed C6 restart boundary is not recoverable") from exc
            _validate_c6_correctness_receipt(c6_receipt, plan)
            capture, opened, released, residency = _run_claimed(
                invocation, plan=plan, executor=executor,
                claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
                claim_journal=claim_journal, claim_timeout_s=claim_timeout_s)
            c6_claim_join = "sealed_c6_restart"
        else:
            capture, opened, released, residency, c6_receipt = \
                _run_c6_oracle_then_targeted_claimed(
                    oracle_invocation, candidate_invocations, invocation,
                    oracle_sidecar, input_dir, plan=plan, executor=executor,
                    claim_acquirer=claim_acquirer,
                    claim_verifier=claim_verifier,
                    claim_journal=claim_journal,
                    claim_timeout_s=claim_timeout_s,
                    c6_ready_timeout_s=c6_ready_timeout_s)
            c6_claim_join = "same_held_claim"
    else:
        c6_sidecar = (directory / "c6-sidecar.json").resolve()
        c6_invocation = CommandInvocation(
            kind="correctness", arm="candidate",
            argv=_materialize_c6_argv(c6.argv, c6_sidecar),
            stdout_path=(directory / "c6-stdout.txt").resolve(),
            stderr_path=(directory / "c6-stderr.txt").resolve(),
            working_directory=plan.execution_cwd,
            environment=plan.correctness_environment)
        if c6_receipt_path.exists() or c6_receipt_path.is_symlink():
            try:
                c6_receipt = proofs.load_receipt(
                    c6_receipt_path, schema=C6_CORRECTNESS_SCHEMA)
            except proofs.ProofError as exc:
                raise EvidenceProducerError(
                    "sealed C6 restart boundary is not recoverable") from exc
            _validate_c6_correctness_receipt(c6_receipt, plan)
            capture, opened, released, residency = _run_claimed(
                invocation, plan=plan, executor=executor,
                claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
                claim_journal=claim_journal, claim_timeout_s=claim_timeout_s)
            c6_claim_join = "sealed_c6_restart"
        else:
            capture, opened, released, residency, c6_receipt = \
                _run_c6_then_targeted_claimed(
                    c6_invocation, invocation, c6_sidecar, plan=plan,
                    executor=executor, claim_acquirer=claim_acquirer,
                    claim_verifier=claim_verifier, claim_journal=claim_journal,
                    claim_timeout_s=claim_timeout_s)
            c6_claim_join = "same_held_claim"
    outputs = _output_hashes(invocation)
    try:
        parsed = _parse_correctness(
            invocation.stdout_path.read_text(encoding="utf-8"), plan)
    except CorrectnessParseRefusal as exc:
        refusal = _seal(directory / "refusal.json", {
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
            "c6_correctness": _reference(c6_receipt),
            "c6_claim_join": c6_claim_join,
            **_claim_boundary_fields(opened, released, residency),
            "residency_witness": residency,
        })
        raise CorrectnessParseRefusal(
            str(exc), receipt_path=str(refusal["path"]),
            receipt_sha256=str(refusal["file_sha256"])) from exc
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
        "c6_correctness": _reference(c6_receipt),
        "c6_claim_join": c6_claim_join,
        **_claim_boundary_fields(opened, released, residency),
        "residency_witness": residency,
    }
    return _seal(directory / "receipt.json", body)


def _validate_correctness_invocation_receipt(
        loaded: Mapping[str, Any], plan: GpuSourceEvidencePlan,
        contract: Mapping[str, Any]) -> None:
    body = loaded["body"]
    expected = {
        "schema": CORRECTNESS_SCHEMA, "authority": AUTHORITY,
        "non_promotable": True, "promotion_claim": False,
        "status": "complete", "result": "PASS",
        "campaign_id": plan.campaign_id, "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "workload_sha256": plan.workload_sha256,
        "invocation_id": contract["invocation_id"],
        "case_set": contract.get("case_set"),
        "command_argv": list(contract["argv"]), "exit_code": 0,
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in _correctness_invocation_environment(
                plan, contract)]),
        "correctness_parser_id": CORRECTNESS_PARSER_ID,
        "correctness_backend": contract["backend"],
        "correctness_op": contract["op"],
        "expected_cases": contract["expected_cases"],
        "passed_cases": contract["expected_cases"],
        "required_cases": list(contract.get("required_cases", [])),
        "exact_case_ok": True,
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError(
            "correctness invocation receipt identity/config/result mismatch")
    c6_reference = body.get("c6_correctness")
    if not isinstance(c6_reference, Mapping):
        raise EvidenceProducerError(
            "correctness invocation lacks sealed C6 receipt")
    c6_loaded = _reload_reference(c6_reference, schema=C6_CORRECTNESS_SCHEMA)
    _validate_c6_correctness_receipt(c6_loaded, plan)
    join = body.get("c6_claim_join")
    if join not in {"same_held_claim", "sealed_c6_restart"}:
        raise EvidenceProducerError(
            "correctness invocation lacks its C6 stage join")
    if (join == "same_held_claim"
            and c6_loaded["body"].get("device_claim_open") !=
                body.get("device_claim_open")):
        raise EvidenceProducerError(
            "correctness invocation did not reuse the held native C6 claim")
    _validate_claim_boundary(body, plan=plan)
    for kind in ("stdout", "stderr"):
        path = Path(str(body.get(f"{kind}_path", "")))
        if _hash_file(path, kind, allow_empty=kind == "stderr") != body.get(
                f"{kind}_sha256"):
            raise EvidenceProducerError(
                f"correctness invocation {kind} bytes changed")
    parsed = _parse_correctness(
        Path(str(body["stdout_path"])).read_text(encoding="utf-8"), plan,
        contract)
    if parsed.summary != body.get("summary"):
        raise EvidenceProducerError("correctness invocation summary changed")
    _validate_residency_witness(
        body.get("residency_witness"), device_id=plan.device_id,
        label=f"correctness {contract['invocation_id']}")


def _correctness_invocation_environment(
        plan: GpuSourceEvidencePlan,
        contract: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    environment = dict(plan.correctness_environment)
    overrides = contract.get("environment_overrides", [])
    if (not isinstance(overrides, list)
            or any(not isinstance(row, list) or len(row) != 2
                   or not all(isinstance(value, str) for value in row)
                   for row in overrides)):
        raise EvidenceProducerError(
            "correctness invocation environment overrides are malformed")
    allowed = {"AUTOKERNEL_CORRECTNESS_CASE_SET"}
    if any(row[0] not in allowed for row in overrides):
        raise EvidenceProducerError(
            "correctness invocation attempted an unreviewed environment override")
    environment.update(dict(overrides))
    return tuple(sorted(environment.items()))


def _load_correctness_invocation_refusal(
        path: Path, plan: GpuSourceEvidencePlan,
        contract: Mapping[str, Any]) -> Mapping[str, Any]:
    loaded = proofs.load_receipt(path, schema=CORRECTNESS_REFUSAL_SCHEMA)
    body = loaded["body"]
    expected = {
        "authority": AUTHORITY, "promotion_claim": False,
        "status": "refused", "classification": "output_parse_refusal",
        "error_type": "CorrectnessParseRefusal",
        "campaign_id": plan.campaign_id, "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "workload_sha256": plan.workload_sha256,
        "invocation_id": contract["invocation_id"],
        "case_set": contract.get("case_set"),
        "command_argv": list(contract["argv"]),
        "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash([
            list(item) for item in _correctness_invocation_environment(
                plan, contract)]),
        "correctness_parser_id": CORRECTNESS_PARSER_ID,
        "correctness_backend": contract["backend"],
        "correctness_op": contract["op"],
        "expected_cases": contract["expected_cases"],
        "required_cases": list(contract.get("required_cases", [])),
    }
    if (any(body.get(key) != value for key, value in expected.items())
            or not isinstance(body.get("reason"), str) or not body["reason"]):
        raise EvidenceProducerError(
            "correctness invocation refusal identity changed")
    c6_reference = body.get("c6_correctness")
    if not isinstance(c6_reference, Mapping):
        raise EvidenceProducerError(
            "correctness invocation refusal lacks sealed C6 receipt")
    c6_loaded = _reload_reference(c6_reference, schema=C6_CORRECTNESS_SCHEMA)
    _validate_c6_correctness_receipt(c6_loaded, plan)
    join = body.get("c6_claim_join")
    if join not in {"same_held_claim", "sealed_c6_restart"}:
        raise EvidenceProducerError(
            "correctness invocation refusal lacks its C6 stage join")
    if (join == "same_held_claim"
            and c6_loaded["body"].get("device_claim_open") !=
                body.get("device_claim_open")):
        raise EvidenceProducerError(
            "correctness invocation refusal did not reuse held native C6 claim")
    _validate_claim_boundary(body, plan=plan)
    for kind in ("stdout", "stderr"):
        if _hash_file(Path(str(body.get(f"{kind}_path", ""))), kind,
                      allow_empty=kind == "stderr") != body.get(f"{kind}_sha256"):
            raise EvidenceProducerError(
                f"correctness invocation refusal {kind} changed")
    try:
        _parse_correctness(
            Path(str(body["stdout_path"])).read_text(encoding="utf-8"),
            plan, contract)
    except CorrectnessParseRefusal as exc:
        if _durable_refusal_reason(exc) != body["reason"]:
            raise EvidenceProducerError(
                "correctness invocation refusal reason changed") from exc
    else:
        raise EvidenceProducerError(
            "correctness invocation refusal now parses as a pass")
    return loaded


def _produce_correctness_invocations(
        root: Path, plan: GpuSourceEvidencePlan, executor: CommandExecutor, *,
        claim_acquirer: Callable[..., Any],
        claim_verifier: Callable[[Mapping[str, Any]], object],
        claim_journal: Any, claim_timeout_s: float) -> Mapping[str, Any]:
    directory = root / "correctness"
    c6 = plan.c6_correctness
    if c6 is None:
        raise EvidenceProducerError(
            "aggregate production correctness lacks C6 capability")
    c6_sidecar = (directory / "c6-sidecar.json").resolve()
    c6_receipt_path = directory / "c6-receipt.json"
    c6_receipt: Mapping[str, Any] | None = None
    c6_claim_join: str | None = None
    if c6_receipt_path.exists() or c6_receipt_path.is_symlink():
        try:
            c6_receipt = proofs.load_receipt(
                c6_receipt_path, schema=C6_CORRECTNESS_SCHEMA)
        except proofs.ProofError as exc:
            raise EvidenceProducerError(
                "sealed aggregate C6 restart boundary is not recoverable") from exc
        _validate_c6_correctness_receipt(c6_receipt, plan)
        c6_claim_join = "sealed_c6_restart"
    if _c6_mode_from_argv(c6.argv) == C6_MODE_ORACLE:
        oracle_sidecar = (directory / "c6-oracle-sidecar.json").resolve()
        input_dir = (directory / "c6-inputs").resolve()
        oracle_invocation = CommandInvocation(
            kind="correctness", arm="candidate",
            argv=_materialize_c6_oracle_argv(c6.argv, oracle_sidecar),
            stdout_path=(directory / "c6-oracle-stdout.txt").resolve(),
            stderr_path=(directory / "c6-oracle-stderr.txt").resolve(),
            working_directory=plan.execution_cwd,
            environment=plan.correctness_environment)
        candidate_invocations = tuple(
            CommandInvocation(
                kind="correctness", arm="candidate",
                argv=_c6_candidate_argv_from_argv(
                    c6.argv, input_dir=input_dir,
                    output=(directory / f"c6-candidate-{index}-output.bin")
                           .resolve(),
                    ready_file=(directory / f"c6-candidate-{index}-ready")
                               .resolve(),
                    continue_file=(
                        directory / f"c6-candidate-{index}-continue").resolve()),
                stdout_path=(directory / f"c6-candidate-{index}-stdout.txt")
                            .resolve(),
                stderr_path=(directory / f"c6-candidate-{index}-stderr.txt")
                            .resolve(),
                working_directory=plan.execution_cwd,
                environment=plan.correctness_environment)
            for index in (1, 2, 3))
    else:
        c6_invocation = CommandInvocation(
            kind="correctness", arm="candidate",
            argv=_materialize_c6_argv(c6.argv, c6_sidecar),
            stdout_path=(directory / "c6-stdout.txt").resolve(),
            stderr_path=(directory / "c6-stderr.txt").resolve(),
            working_directory=plan.execution_cwd,
            environment=plan.correctness_environment)
    references: list[dict[str, Any]] = []
    for invocation_index, contract in enumerate(plan.correctness_invocations):
        invocation_id = str(contract["invocation_id"])
        invocation_dir = directory / invocation_id
        receipt_path = invocation_dir / "receipt.json"
        refusal_path = invocation_dir / "refusal.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            if refusal_path.exists() or refusal_path.is_symlink():
                raise EvidenceProducerError(
                    f"correctness invocation {invocation_id} has contradictory terminals")
            loaded = proofs.load_receipt(receipt_path, schema=CORRECTNESS_SCHEMA)
            _validate_correctness_invocation_receipt(loaded, plan, contract)
            if invocation_index == 0 and c6_receipt is None:
                raise EvidenceProducerError(
                    "aggregate targeted correctness exists without sealed C6")
            references.append(_reference(loaded))
            continue
        if refusal_path.exists() or refusal_path.is_symlink():
            loaded = _load_correctness_invocation_refusal(
                refusal_path, plan, contract)
            raise CorrectnessParseRefusal(
                str(loaded["body"]["reason"]),
                receipt_path=str(loaded["path"]),
                receipt_sha256=str(loaded["file_sha256"]))
        if invocation_dir.exists() or invocation_dir.is_symlink():
            raise EvidenceProducerError(
                f"correctness invocation {invocation_id} is incomplete")
        invocation = CommandInvocation(
            kind="correctness", arm="candidate",
            argv=tuple(contract["argv"]),
            stdout_path=(invocation_dir / "stdout.txt").resolve(),
            stderr_path=(invocation_dir / "stderr.txt").resolve(),
            working_directory=plan.execution_cwd,
            environment=_correctness_invocation_environment(plan, contract))
        if invocation_index == 0 and c6_receipt is None:
            if _c6_mode_from_argv(c6.argv) == C6_MODE_ORACLE:
                capture, opened, released, residency, c6_receipt = (
                    _run_c6_oracle_then_targeted_claimed(
                        oracle_invocation, candidate_invocations, invocation,
                        oracle_sidecar, input_dir, plan=plan, executor=executor,
                        claim_acquirer=claim_acquirer,
                        claim_verifier=claim_verifier,
                        claim_journal=claim_journal,
                        claim_timeout_s=claim_timeout_s))
            else:
                capture, opened, released, residency, c6_receipt = (
                    _run_c6_then_targeted_claimed(
                        c6_invocation, invocation, c6_sidecar, plan=plan,
                        executor=executor, claim_acquirer=claim_acquirer,
                        claim_verifier=claim_verifier,
                        claim_journal=claim_journal,
                        claim_timeout_s=claim_timeout_s))
            c6_claim_join = "same_held_claim"
        else:
            if c6_receipt is None:
                raise EvidenceProducerError(
                    "aggregate targeted correctness cannot precede sealed C6")
            capture, opened, released, residency = _run_claimed(
                invocation, plan=plan, executor=executor,
                claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
                claim_journal=claim_journal, claim_timeout_s=claim_timeout_s)
        outputs = _output_hashes(invocation)
        try:
            parsed = _parse_correctness(
                invocation.stdout_path.read_text(encoding="utf-8"), plan,
                contract)
        except CorrectnessParseRefusal as exc:
            refusal = _seal(refusal_path, {
                "schema": CORRECTNESS_REFUSAL_SCHEMA,
                "authority": AUTHORITY, "promotion_claim": False,
                "status": "refused",
                "classification": "output_parse_refusal",
                "error_type": type(exc).__name__,
                "reason": _durable_refusal_reason(exc),
                "campaign_id": plan.campaign_id,
                "device_id": plan.device_id,
                "manifest_sha256": plan.manifest_sha256,
                "candidate_build_identity": asdict(plan.candidate),
                "workload_sha256": plan.workload_sha256,
                "invocation_id": invocation_id,
                "case_set": contract.get("case_set"),
                "command_argv": list(contract["argv"]),
                "command_cwd": str(plan.execution_cwd),
                "command_environment_sha256": schemas.content_hash([
                    list(item) for item in _correctness_invocation_environment(
                        plan, contract)]),
                "exit_code": capture.exit_code, **outputs,
                "started_at": capture.started_at, "ended_at": capture.ended_at,
                "correctness_parser_id": CORRECTNESS_PARSER_ID,
                "correctness_backend": contract["backend"],
                "correctness_op": contract["op"],
                "expected_cases": contract["expected_cases"],
                "required_cases": list(contract.get("required_cases", [])),
                "c6_correctness": _reference(c6_receipt),
                "c6_claim_join": (
                    c6_claim_join if invocation_index == 0
                    else "sealed_c6_restart"),
                **_claim_boundary_fields(opened, released, residency),
                "residency_witness": residency,
            })
            raise CorrectnessParseRefusal(
                str(exc), receipt_path=str(refusal["path"]),
                receipt_sha256=str(refusal["file_sha256"])) from exc
        if capture.exit_code != 0:
            raise EvidenceProducerError(
                f"correctness invocation {invocation_id} exited nonzero")
        body = {
            "schema": CORRECTNESS_SCHEMA, "authority": AUTHORITY,
            "non_promotable": True, "promotion_claim": False,
            "status": "complete", "result": "PASS",
            "campaign_id": plan.campaign_id, "device_id": plan.device_id,
            "manifest_sha256": plan.manifest_sha256,
            "candidate_build_identity": asdict(plan.candidate),
            "workload_sha256": plan.workload_sha256,
            "invocation_id": invocation_id,
            "case_set": contract.get("case_set"),
            "command_argv": list(contract["argv"]),
            "command_cwd": str(plan.execution_cwd),
            "command_environment_sha256": schemas.content_hash(
                [list(item) for item in _correctness_invocation_environment(
                    plan, contract)]),
            "exit_code": capture.exit_code, **outputs,
            "started_at": capture.started_at, "ended_at": capture.ended_at,
            "summary": parsed.summary,
            "correctness_parser_id": CORRECTNESS_PARSER_ID,
            "correctness_backend": parsed.backend,
            "correctness_op": parsed.operation,
            "expected_cases": contract["expected_cases"],
            "passed_cases": contract["expected_cases"],
            "required_cases": list(contract.get("required_cases", [])),
            "exact_case_ok": True,
            "c6_correctness": _reference(c6_receipt),
            "c6_claim_join": (
                c6_claim_join if invocation_index == 0
                else "sealed_c6_restart"),
            **_claim_boundary_fields(opened, released, residency),
            "residency_witness": residency,
        }
        loaded = _seal(receipt_path, body)
        references.append(_reference(loaded))
    if c6_receipt is None:
        raise EvidenceProducerError(
            "aggregate correctness completed without sealed C6")
    aggregate = {
        "schema": CORRECTNESS_SCHEMA, "authority": AUTHORITY,
        "non_promotable": True, "promotion_claim": False,
        "status": "complete", "result": "PASS",
        "campaign_id": plan.campaign_id, "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "identity_files": _identity_files_reference(plan.identity_files),
        "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x)
                                for x in plan.correctness_inputs],
        "workload_sha256": plan.workload_sha256,
        "command_argv": list(plan.correctness_argv),
        "command_cwd": str(plan.execution_cwd),
        "correctness_parser_id": CORRECTNESS_PARSER_ID,
        "correctness_backend": plan.correctness_backend,
        "correctness_op": plan.correctness_op,
        "expected_cases": plan.expected_correctness_cases,
        "correctness_invocation_contracts": [dict(row)
                                             for row in plan.correctness_invocations],
        "invocations": references,
        "c6_correctness": _reference(c6_receipt),
        "exact_case_ok": True,
    }
    return _seal(directory / "receipt.json", aggregate)


def _integer(row: Mapping[str, str], key: str, *, minimum: int) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceProducerError(f"timestamp CSV has invalid {key}") from exc
    if value < minimum:
        raise EvidenceProducerError(f"timestamp CSV {key} is below {minimum}")
    return value


def _load_dispatches(
        path: Path, *, profiler_trace_schema_id: str = ROCPROF_V1_TRACE_ID,
        expected_rows: int | None = None) -> list[dict[str, Any]]:
    _hash_file(path, "timestamp CSV", allow_empty=False)
    if profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise EvidenceProducerError(
                "kernel trace CSV is not exact UTF-8") from exc
        physical_lines = text.splitlines()
        if (not raw.endswith(b"\n") or raw.endswith((b"\n\n", b"\r\n\r\n"))
                or not physical_lines or any(not line for line in physical_lines)):
            raise EvidenceProducerError(
                "kernel trace CSV must have one trailing newline and no blank physical lines")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
            if tuple(reader.fieldnames or ()) != ROCPROF_V3_COLUMNS:
                raise EvidenceProducerError(
                    "kernel trace CSV lacks the exact rocprofv3 columns")
            dispatches = []
            agent_ids: set[int] = set()
            queue_ids: set[int] = set()
            correlations: set[int] = set()
            for index, row in enumerate(reader):
                if None in row or any(value is None for value in row.values()):
                    raise EvidenceProducerError("kernel trace CSV row shape changed")
                if row["Kind"] != "KERNEL_DISPATCH":
                    raise EvidenceProducerError("kernel trace CSV contains a non-dispatch row")
                kernel = row["Kernel_Name"]
                if not kernel or any(ord(ch) < 0x20 for ch in kernel):
                    raise EvidenceProducerError("kernel trace CSV kernel name is malformed")
                agent_id = _integer(row, "Agent_Id", minimum=0)
                queue_id = _integer(row, "Queue_Id", minimum=0)
                correlation_id = _integer(row, "Correlation_Id", minimum=1)
                agent_ids.add(agent_id); queue_ids.add(queue_id)
                if correlation_id in correlations:
                    raise EvidenceProducerError(
                        "kernel trace correlation IDs are not unique")
                correlations.add(correlation_id)
                workgroup_xyz = tuple(_integer(
                    row, f"Workgroup_Size_{axis}", minimum=1) for axis in "XYZ")
                grid_xyz = tuple(_integer(
                    row, f"Grid_Size_{axis}", minimum=1) for axis in "XYZ")
                workgroup = math.prod(workgroup_xyz)
                grid = math.prod(grid_xyz)
                raw_group_segment = _integer(
                    row, "Group_Segment_Size", minimum=0)
                # rocprofv3 reports the requested group segment while the
                # reviewed gfx90a v1 authority reports its 512-byte allocation.
                # The route contract is deliberately still in allocation
                # bytes, so this mapping is architecture-bound and explicit.
                allocated_lds = (0 if raw_group_segment == 0 else
                                 ((raw_group_segment + 511) // 512) * 512)
                values = {
                    "agent_id": agent_id, "queue_id": queue_id,
                    "kernel_id": _integer(row, "Kernel_Id", minimum=0),
                    "correlation_id": correlation_id,
                    "grid": grid, "grid_xyz": list(grid_xyz),
                    "workgroup": workgroup,
                    "workgroup_xyz": list(workgroup_xyz),
                    "group_segment_size": raw_group_segment,
                    "lds": allocated_lds,
                    "private_segment_size": _integer(
                        row, "Private_Segment_Size", minimum=0),
                    "begin_ns": _integer(row, "Start_Timestamp", minimum=0),
                    "end_ns": _integer(row, "End_Timestamp", minimum=1),
                }
                if (values["end_ns"] <= values["begin_ns"]
                        or values["grid"] % values["workgroup"]):
                    raise EvidenceProducerError(
                        "kernel trace row has invalid duration or non-integral blocks")
                dispatches.append({
                    "index": index, "kernel": kernel, **values,
                    "blocks_per_call": values["grid"] // values["workgroup"],
                })
            if not dispatches:
                raise EvidenceProducerError("kernel trace CSV contains no dispatches")
            if (expected_rows is None or len(dispatches) != expected_rows):
                raise EvidenceProducerError(
                    "kernel trace CSV does not have the exact expected row count")
            if len(agent_ids) != 1:
                raise EvidenceProducerError(
                    "kernel trace does not bind one exact GPU agent")
            if not queue_ids:
                raise EvidenceProducerError("kernel trace contains no queue identity")
            if correlations != set(range(1, expected_rows + 1)):
                raise EvidenceProducerError(
                    "kernel trace correlation IDs are not the exact contiguous domain")
            return dispatches
        if profiler_trace_schema_id != ROCPROF_V1_TRACE_ID:
            raise EvidenceProducerError("profiler trace schema is not reviewed")
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


def _profiler_structural_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent v3 topology; excludes volatile kernel/correlation IDs."""
    counts: dict[tuple[Any, ...], int] = {}
    fields = ("agent_id", "queue_id", "kernel", "private_segment_size",
              "group_segment_size", "workgroup_xyz", "grid_xyz")
    for row in rows:
        try:
            key = tuple(tuple(row[field]) if isinstance(row[field], list)
                        else row[field] for field in fields)
        except KeyError as exc:
            raise EvidenceProducerError(
                "rocprofv3 row lacks a structural fingerprint field") from exc
        counts[key] = counts.get(key, 0) + 1
    canonical = []
    for key, calls in sorted(counts.items(), key=lambda item: repr(item[0])):
        row = dict(zip(fields, key)); row["calls"] = calls
        for field in ("workgroup_xyz", "grid_xyz"):
            row[field] = list(row[field])
        canonical.append(row)
    return schemas.content_hash(canonical)


def _rocprofv3_agent_info_path(timestamp_csv: Path) -> Path:
    suffix = "_kernel_trace.csv"
    if not timestamp_csv.name.endswith(suffix):
        raise EvidenceProducerError("rocprofv3 kernel trace suffix changed")
    return timestamp_csv.with_name(
        f"{timestamp_csv.name[:-len(suffix)]}_agent_info.csv")


def _load_rocprofv3_agent_info(
        path: Path, *, trace_agent_ids: set[int]) -> dict[str, Any]:
    _hash_file(path, "rocprofv3 agent info", allow_empty=False)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceProducerError("rocprofv3 agent info is not UTF-8") from exc
    physical_lines = text.splitlines()
    if (not raw.endswith(b"\n") or raw.endswith((b"\n\n", b"\r\n\r\n"))
            or not physical_lines or any(not line for line in physical_lines)):
        raise EvidenceProducerError(
            "rocprofv3 agent info must have one trailing newline and no blanks")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROCPROF_V3_AGENT_COLUMNS:
            raise EvidenceProducerError(
                "rocprofv3 agent info lacks the exact reviewed columns")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise EvidenceProducerError("rocprofv3 agent info row shape changed")
    gpu = [row for row in rows if row.get("Agent_Type") == "GPU"]
    if len(gpu) != 1:
        raise EvidenceProducerError("rocprofv3 agent info must contain one GPU")
    row = gpu[0]
    try:
        logical_node_id = int(row["Logical_Node_Id"])
        node_id = int(row["Node_Id"])
        gfx_target = int(row["Gfx_Target_Version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceProducerError(
            "rocprofv3 GPU agent identity is malformed") from exc
    if (trace_agent_ids != {logical_node_id} or node_id != logical_node_id
            or gfx_target != 90010 or row.get("Name") != "gfx90a"
            or row.get("Product_Name") != "AMD Instinct MI210"):
        raise EvidenceProducerError(
            "rocprofv3 trace does not bind the reviewed gfx90a MI210 agent")
    return {
        "agent_id": logical_node_id, "node_id": node_id,
        "gfx_target_version": gfx_target, "name": row["Name"],
        "product_name": row["Product_Name"],
        "agent_info_sha256": _hash_file(path, "rocprofv3 agent info"),
    }


def _load_arm_dispatches(
        path: Path, plan: GpuSourceEvidencePlan, *, arm: str) -> list[dict[str, Any]]:
    if arm not in {"candidate", "anchor"}:
        raise EvidenceProducerError("profiler dispatch arm is invalid")
    return _load_dispatches(
        path, profiler_trace_schema_id=plan.profiler_trace_schema_id,
        expected_rows=(plan.expected_candidate_profiler_dispatch_rows
                       if arm == "candidate"
                       else plan.expected_anchor_profiler_dispatch_rows))


def _profile_target_binary(plan: GpuSourceEvidencePlan, arm: str) -> Path:
    if plan.shared_runtime is not None:
        return plan.shared_runtime.measurement_binary.path
    return (plan.identity_files.candidate.binary.path if arm == "candidate"
            else plan.identity_files.anchor.binary.path)


def _parse_profile_completion(
        path: Path, *, expected_csv: Path, plan: GpuSourceEvidencePlan,
        arm: str, argv: Sequence[str]) -> dict[str, Any]:
    """Prove the target completed the exact tg128 JSON cell."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceProducerError("profiler target stdout is not UTF-8") from exc
    try:
        rows = microbench.parse_llama_bench_json(text)
    except (TypeError, microbench.BenchOutputError) as exc:
        raise EvidenceProducerError(
            "profiler target did not emit one strict JSON benchmark payload") from exc
    if len(rows) != 1:
        raise EvidenceProducerError("profiler target must emit exactly one result row")
    target = _profile_target_binary(plan, arm)
    try:
        target_index = tuple(argv).index(str(target))
    except ValueError as exc:
        raise EvidenceProducerError("profiler command lacks the sealed benchmark target") from exc
    target_argv = list(argv[target_index + 1:])
    def after(flag: str) -> str | None:
        if target_argv.count(flag) != 1:
            return None
        index = target_argv.index(flag)
        return target_argv[index + 1] if index + 1 < len(target_argv) else None
    row = rows[0]
    expected = {
        "model_filename": str(plan.identity_files.model.path),
        "n_prompt": 0, "n_gen": 128, "n_threads": 8,
        "n_gpu_layers": 99, "flash_attn": True, "repetitions": 1,
    }
    try:
        argv_expected = {
            "model_filename": after("-m"), "n_prompt": int(after("-p") or ""),
            "n_gen": int(after("-n") or ""), "n_threads": int(after("-t") or ""),
            "n_gpu_layers": int(after("-ngl") or ""),
            "flash_attn": after("-fa") in {"1", "on", "true"},
            "repetitions": int(after("-r") or ""),
        }
    except ValueError as exc:
        raise EvidenceProducerError("profiler target argv is not the sealed tg128 cell") from exc
    if argv_expected != expected or after("-o") != "json":
        raise EvidenceProducerError("profiler target argv is not the exact JSON tg128 cell")
    actual = {
        "model_filename": row.model_filename,
        "n_prompt": row.n_prompt, "n_gen": row.n_gen,
        "n_threads": row.n_threads, "n_gpu_layers": row.n_gpu_layers,
        "flash_attn": row.flash_attn, "repetitions": len(row.samples_ts),
    }
    if actual != expected:
        raise EvidenceProducerError("profiler target result does not match the tg128 argv")
    return {"parser": "microbench.parse_llama_bench_json",
            "result": row.to_dict(), "expected": expected,
            "results_file": str(expected_csv), "complete": True}


def _matching(rows: Sequence[Mapping[str, Any]], pattern: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if re.search(pattern, str(row["kernel"]))]


def _geometry_signature(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[tuple[int, int, int, int], int] = {}
    durations: list[int] = []
    for row in rows:
        key = (int(row["grid"]), int(row["workgroup"]), int(row["lds"]),
               int(row["blocks_per_call"]))
        counts[key] = counts.get(key, 0) + 1
        duration = int(row["end_ns"]) - int(row["begin_ns"])
        if duration <= 0:
            raise EvidenceProducerError("timestamp row has non-positive duration")
        durations.append(duration)
    ordered_durations = sorted(durations)
    return {
        "calls": len(rows),
        "total_duration_ns": sum(durations),
        "duration_ns": durations,
        "duration_statistic": "median_per_dispatch_ns",
        "median_duration_ns": (
            None if not ordered_durations else
            (ordered_durations[len(ordered_durations) // 2]
             if len(ordered_durations) % 2 else
             (ordered_durations[len(ordered_durations) // 2 - 1]
              + ordered_durations[len(ordered_durations) // 2]) / 2)),
        "geometries": [
            {"grid": key[0], "workgroup": key[1], "lds_bytes": key[2],
             "blocks_per_call": key[3], "calls": count}
            for key, count in sorted(counts.items())
        ],
    }


def _reduce_arm(
    rows: Sequence[Mapping[str, Any]], *, exact: Sequence[ExactDispatch],
    structural_exact: Sequence[ExactDispatch] = (),
    forbidden: Sequence[ForbiddenDispatch], invariants: Sequence[InvariantDispatch],
) -> dict[str, Any]:
    def reduce_exact(expectations: Sequence[ExactDispatch], label: str
                     ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for expectation in expectations:
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
            if (geometry["calls"] != expectation.calls
                    or geometry["geometries"] != expected_geometry):
                raise EvidenceProducerError(
                    f"{label} dispatch {expectation.signature} count/geometry mismatch")
            result[expectation.signature] = geometry
        for pattern in {item.kernel_pattern for item in expectations}:
            allowed = {(item.grid, item.workgroup, item.lds_bytes,
                        item.blocks_per_call)
                       for item in expectations if item.kernel_pattern == pattern}
            unexpected = [row for row in _matching(rows, pattern)
                          if (int(row["grid"]), int(row["workgroup"]),
                              int(row["lds"]), int(row["blocks_per_call"]))
                          not in allowed]
            if unexpected:
                raise EvidenceProducerError(
                    f"{label} dispatch matched an unreviewed geometry")
        return result

    exact_result = reduce_exact(exact, "exact")
    structural_result = reduce_exact(structural_exact, "structural")
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
        invariant_result[expectation.signature] = {
            key: value for key, value in _geometry_signature(hits).items()
            if key in {"calls", "geometries"}}
    return {"exact": exact_result, "structural_exact": structural_result,
            "forbidden": forbidden_result,
            "invariants": invariant_result}


def _arm_structural_exact(plan: GpuSourceEvidencePlan, arm: str
                          ) -> tuple[ExactDispatch, ...]:
    return (plan.dispatch.candidate_structural_exact if arm == "candidate"
            else plan.dispatch.anchor_structural_exact)


def _rocprofv3_invocation(
        directory: Path, attempt_number: int, arm: str,
        plan: GpuSourceEvidencePlan) -> CommandInvocation:
    attempt = directory / f"attempt-{attempt_number:02d}" / "raw"
    output_csv = (attempt / "trace_kernel_trace.csv").resolve()
    argv_template = (plan.candidate_rocprof_argv if arm == "candidate"
                     else plan.anchor_rocprof_argv)
    inputs = (plan.candidate_rocprof_inputs if arm == "candidate"
              else plan.anchor_rocprof_inputs)
    return CommandInvocation(
        kind="rocprof", arm=arm,
        argv=_materialize_rocprof_argv(argv_template, output_csv),
        stdout_path=(attempt / "stdout.txt").resolve(),
        stderr_path=(attempt / "stderr.txt").resolve(),
        timestamp_csv_path=output_csv,
        working_directory=plan.execution_cwd,
        environment=(plan.candidate_rocprof_environment if arm == "candidate"
                     else plan.anchor_rocprof_environment),
        runtime_maps_required=plan.shared_runtime is not None,
        runtime_maps_context=(None if plan.shared_runtime is None else {
            "arm": arm, "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
            "model": _bound_reference(plan.identity_files.model),
            "model_sha256": plan.model_sha256, "device_id": plan.device_id,
            "required_profiler_mapped_files":
                _required_profiler_mapped_files(inputs)}))


def _profiler_attempt_expected_outcome(exit_code: int) -> tuple[str, bool]:
    if exit_code == 0:
        return "clean_exit", False
    return "nonzero_exit", False


def _rocprofv3_raw_artifacts(invocation: CommandInvocation) -> list[dict[str, Any]]:
    assert invocation.timestamp_csv_path is not None
    raw = invocation.timestamp_csv_path.parent
    if not invocation.timestamp_csv_path.name.endswith("_kernel_trace.csv"):
        raise EvidenceProducerError("rocprofv3 timestamp output name changed")
    basename = invocation.timestamp_csv_path.name[:-len("_kernel_trace.csv")]
    expected_paths = (
        invocation.stdout_path, invocation.stderr_path,
        invocation.timestamp_csv_path, raw / f"{basename}_agent_info.csv")
    if (any(path.parent != raw for path in expected_paths)
            or len({path.name for path in expected_paths}) != 4):
        raise EvidenceProducerError(
            "rocprofv3 raw output members must share one exact directory")
    expected_files = {path.name for path in expected_paths}
    metadata_name = ".rocprofv3"
    expected_entries = expected_files | {metadata_name}
    directory_flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                       | getattr(os, "O_DIRECTORY", 0)
                       | getattr(os, "O_NOFOLLOW", 0))
    file_flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                  | getattr(os, "O_NOFOLLOW", 0)
                  | getattr(os, "O_NONBLOCK", 0))

    def stable_identity(value: os.stat_result) -> tuple[int, ...]:
        return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def hash_member(directory_fd: int, name: str) -> tuple[str, int]:
        try:
            descriptor = os.open(name, file_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise EvidenceProducerError(
                f"rocprofv3 {name} is not an exact raw closure file") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise EvidenceProducerError(
                    f"rocprofv3 {name} must be a single-link regular file")
            if name != invocation.stderr_path.name and before.st_size == 0:
                raise EvidenceProducerError(f"rocprofv3 {name} must not be empty")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
            if stable_identity(before) != stable_identity(after):
                raise EvidenceProducerError(
                    f"rocprofv3 {name} mutated while sealing raw output")
            return digest.hexdigest(), before.st_size
        finally:
            os.close(descriptor)

    try:
        resolved_raw = raw.resolve(strict=True)
        if raw.is_symlink() or resolved_raw != raw:
            raise EvidenceProducerError(
                "rocprofv3 raw output directory must be a real contained path")
        raw_fd = os.open(raw, directory_flags)
    except OSError as exc:
        raise EvidenceProducerError("rocprofv3 raw output directory is unreadable") from exc
    try:
        raw_before = os.fstat(raw_fd)
        raw_path = os.lstat(raw)
        if (not stat.S_ISDIR(raw_before.st_mode)
                or stable_identity(raw_before) != stable_identity(raw_path)):
            raise EvidenceProducerError(
                "rocprofv3 raw output directory identity changed before sealing")
        actual = set(os.listdir(raw_fd))
        if actual != expected_entries:
            raise EvidenceProducerError(
                "rocprofv3 raw output directory is not the exact two-CSV closure")
        try:
            metadata_fd = os.open(metadata_name, directory_flags, dir_fd=raw_fd)
        except OSError as exc:
            raise EvidenceProducerError(
                "rocprofv3 metadata entry must be a real contained directory") from exc
        try:
            metadata_before = os.fstat(metadata_fd)
            metadata_mode = stat.S_IMODE(metadata_before.st_mode)
            if (not stat.S_ISDIR(metadata_before.st_mode)
                    or metadata_before.st_nlink != 2
                    or metadata_before.st_dev != raw_before.st_dev
                    or metadata_before.st_uid != raw_before.st_uid
                    or metadata_before.st_gid != raw_before.st_gid
                    or metadata_mode & 0o7000
                    or metadata_mode & 0o022
                    or os.path.ismount(raw / metadata_name)
                    or os.listdir(metadata_fd)):
                raise EvidenceProducerError(
                    "rocprofv3 metadata directory must be a safe empty peer")
            metadata_artifact = {
                "name": metadata_name,
                "path": str(raw / metadata_name),
                "kind": "profiler_bookkeeping_directory",
                "scientific_evidence": False,
                "device_major": os.major(metadata_before.st_dev),
                "device_minor": os.minor(metadata_before.st_dev),
                "inode": metadata_before.st_ino,
                "uid": metadata_before.st_uid,
                "gid": metadata_before.st_gid,
                "mode": format(metadata_mode, "04o"),
                "links": metadata_before.st_nlink,
                "entries": 0,
            }
            metadata_artifact["metadata_sha256"] = schemas.content_hash(
                metadata_artifact)
            artifacts = [metadata_artifact]
            for name in sorted(expected_files):
                digest, size = hash_member(raw_fd, name)
                artifacts.append({"name": name, "path": str(raw / name),
                                  "kind": "regular_file",
                                  "scientific_evidence": True,
                                  "sha256": digest, "bytes": size})
            metadata_after = os.fstat(metadata_fd)
            if (stable_identity(metadata_before) != stable_identity(metadata_after)
                    or os.listdir(metadata_fd)
                    or os.path.ismount(raw / metadata_name)):
                raise EvidenceProducerError(
                    "rocprofv3 metadata directory mutated while sealing raw output")
            if set(os.listdir(raw_fd)) != expected_entries:
                raise EvidenceProducerError(
                    "rocprofv3 raw output directory mutated while sealing")
            raw_after = os.fstat(raw_fd)
            if stable_identity(raw_before) != stable_identity(raw_after):
                raise EvidenceProducerError(
                    "rocprofv3 raw output directory mutated while sealing")
            return artifacts
        finally:
            os.close(metadata_fd)
    finally:
        os.close(raw_fd)


def _revalidate_rocprofv3_raw_artifacts(
        invocation: CommandInvocation, sealed: object) -> list[dict[str, Any]]:
    current = _rocprofv3_raw_artifacts(invocation)
    if not isinstance(sealed, list) or sealed != current:
        raise EvidenceProducerError(
            "rocprofv3 raw artifact closure changed after sealing")
    return current


def _seal_profiler_failure_attempt(
        *, directory: Path, attempt_number: int, arm: str,
        plan: GpuSourceEvidencePlan, invocation: CommandInvocation,
        capture: ExecutionCapture, opened: Mapping[str, Any],
        released: Mapping[str, Any], residency: Mapping[str, Any]) -> Mapping[str, Any]:
    artifacts = []
    for path in (invocation.stdout_path, invocation.stderr_path,
                 invocation.timestamp_csv_path):
        if path is not None and path.exists() and not path.is_symlink() and path.is_file():
            artifacts.append({"path": str(path),
                              "sha256": _hash_file(path, "failed profiler output"),
                              "bytes": path.stat().st_size})
    runtime_maps = _validated_runtime_maps_identity(
        capture, plan=plan, arm=arm, residency=residency)
    return _seal(directory / f"attempt-{attempt_number:02d}" / "transport.json", {
        "schema": PROFILER_ATTEMPT_SCHEMA, "authority": AUTHORITY,
        "promotion_claim": False, "status": "refused", "arm": arm,
        "attempt_number": attempt_number, "campaign_id": plan.campaign_id,
        "device_id": plan.device_id, "manifest_sha256": plan.manifest_sha256,
        "build_identity": asdict(plan.candidate if arm == "candidate" else plan.anchor),
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "profiler_trace_schema_id": plan.profiler_trace_schema_id,
        "profiler_transport_policy": plan.profiler_transport_policy,
        "command_argv": list(invocation.argv), "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in invocation.environment]),
        "exit_code": capture.exit_code, "transport_outcome": "nonzero_exit",
        "retry_eligible": False, "evidence_complete": False,
        "raw_artifacts": artifacts,
        "started_at": capture.started_at, "ended_at": capture.ended_at,
        **_claim_boundary_fields(opened, released, residency),
        "residency_witness": residency, "runtime_maps_identity": runtime_maps,
    })


def _seal_profiler_attempt(
        *, directory: Path, attempt_number: int, arm: str,
        plan: GpuSourceEvidencePlan, invocation: CommandInvocation,
        capture: ExecutionCapture, opened: Mapping[str, Any],
        released: Mapping[str, Any], residency: Mapping[str, Any],
        outputs: Mapping[str, Any], dispatches: Sequence[Mapping[str, Any]],
        reduction: Mapping[str, Any], runtime_maps: Mapping[str, Any] | None,
        completion: Mapping[str, Any],
        agent_info: Mapping[str, Any]) -> Mapping[str, Any]:
    outcome, retry_eligible = _profiler_attempt_expected_outcome(capture.exit_code)
    return _seal(directory / f"attempt-{attempt_number:02d}" / "transport.json", {
        "schema": PROFILER_ATTEMPT_SCHEMA, "authority": AUTHORITY,
        "promotion_claim": False, "status": "complete",
        "arm": arm, "attempt_number": attempt_number,
        "campaign_id": plan.campaign_id, "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "build_identity": asdict(plan.candidate if arm == "candidate" else plan.anchor),
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "profiler_trace_schema_id": plan.profiler_trace_schema_id,
        "profiler_transport_policy": plan.profiler_transport_policy,
        "command_argv": list(invocation.argv), "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in invocation.environment]),
        "exit_code": capture.exit_code, "transport_outcome": outcome,
        "retry_eligible": retry_eligible, "evidence_complete": True,
        **outputs, "raw_artifacts": _rocprofv3_raw_artifacts(invocation),
        "dispatch_row_count": len(dispatches),
        "timestamp_reduction_sha256": schemas.content_hash(dispatches),
        "structural_fingerprint_sha256": _profiler_structural_fingerprint(dispatches),
        "exact_dispatch_signatures": reduction["exact"],
        "structural_dispatch_signatures": reduction["structural_exact"],
        "forbidden_dispatch_signatures": reduction["forbidden"],
        "invariant_signatures": reduction["invariants"],
        "benchmark_completion": dict(completion),
        "gpu_agent_identity": dict(agent_info),
        "started_at": capture.started_at, "ended_at": capture.ended_at,
        **_claim_boundary_fields(opened, released, residency),
        "residency_witness": residency, "runtime_maps_identity": runtime_maps,
    })


def _load_profiler_attempt(
        path: Path, *, attempt_number: int, arm: str,
        plan: GpuSourceEvidencePlan) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    loaded = proofs.load_receipt(path, schema=PROFILER_ATTEMPT_SCHEMA)
    body = loaded["body"]
    identity = plan.candidate if arm == "candidate" else plan.anchor
    output = Path(str(body.get("timestamp_csv_path", "")))
    invocation = _rocprofv3_invocation(path.parent.parent, attempt_number, arm, plan)
    if body.get("status") == "refused":
        expected_failure = {
            "authority": AUTHORITY, "promotion_claim": False,
            "status": "refused", "arm": arm, "attempt_number": attempt_number,
            "campaign_id": plan.campaign_id, "device_id": plan.device_id,
            "manifest_sha256": plan.manifest_sha256,
            "build_identity": asdict(identity), "model_sha256": plan.model_sha256,
            "workload_sha256": plan.workload_sha256,
            "runtime_config_sha256": plan.runtime_config_sha256,
            "profiler_trace_schema_id": plan.profiler_trace_schema_id,
            "profiler_transport_policy": plan.profiler_transport_policy,
            "command_argv": list(invocation.argv),
            "command_cwd": str(plan.execution_cwd),
            "command_environment_sha256": schemas.content_hash(
                [list(item) for item in invocation.environment]),
            "transport_outcome": "nonzero_exit", "retry_eligible": False,
            "evidence_complete": False,
        }
        if (any(body.get(key) != value for key, value in expected_failure.items())
                or isinstance(body.get("exit_code"), bool)
                or not isinstance(body.get("exit_code"), int)
                or body["exit_code"] == 0
                or not isinstance(body.get("raw_artifacts"), list)):
            raise EvidenceProducerError("failed profiler transport receipt changed")
        for artifact in body["raw_artifacts"]:
            if (not isinstance(artifact, Mapping)
                    or _hash_file(Path(str(artifact.get("path", ""))),
                                  "failed profiler output") != artifact.get("sha256")):
                raise EvidenceProducerError("failed profiler transport bytes changed")
        _validate_claim_boundary(body, plan=plan)
        _validate_residency_witness(body.get("residency_witness"),
                                    device_id=plan.device_id, label=arm)
        if plan.shared_runtime is not None:
            _validate_runtime_maps_receipt(
                body.get("runtime_maps_identity"), plan=plan, arm=arm,
                residency=body.get("residency_witness"))
        raise EvidenceProducerError(
            f"{arm} rocprofv3 transport previously exited nonzero")
    expected_outcome, retry_eligible = _profiler_attempt_expected_outcome(
        int(body.get("exit_code", -1)))
    expected = {
        "authority": AUTHORITY, "promotion_claim": False, "status": "complete",
        "arm": arm, "attempt_number": attempt_number,
        "campaign_id": plan.campaign_id, "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256, "build_identity": asdict(identity),
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "profiler_trace_schema_id": plan.profiler_trace_schema_id,
        "profiler_transport_policy": plan.profiler_transport_policy,
        "command_argv": list(invocation.argv), "command_cwd": str(plan.execution_cwd),
        "command_environment_sha256": schemas.content_hash(
            [list(item) for item in invocation.environment]),
        "transport_outcome": expected_outcome, "retry_eligible": retry_eligible,
        "evidence_complete": True,
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError("profiler transport attempt identity changed")
    _validate_claim_boundary(body, plan=plan)
    for kind in ("stdout", "stderr", "timestamp_csv"):
        if (_hash_file(Path(str(body.get(f"{kind}_path", ""))), kind,
                       allow_empty=kind == "stderr") != body.get(f"{kind}_sha256")):
            raise EvidenceProducerError(f"profiler transport attempt {kind} changed")
    if output != invocation.timestamp_csv_path:
        raise EvidenceProducerError("profiler transport attempt output path changed")
    dispatches = _load_arm_dispatches(output, plan, arm=arm)
    agent_info = _load_rocprofv3_agent_info(
        _rocprofv3_agent_info_path(output),
        trace_agent_ids={int(row["agent_id"]) for row in dispatches})
    exact = plan.dispatch.candidate_exact if arm == "candidate" else plan.dispatch.anchor_exact
    forbidden = (plan.dispatch.candidate_forbidden if arm == "candidate"
                 else plan.dispatch.anchor_forbidden)
    reduction = _reduce_arm(
        dispatches, exact=exact,
        structural_exact=_arm_structural_exact(plan, arm), forbidden=forbidden,
        invariants=plan.dispatch.invariants)
    completion = _parse_profile_completion(
        Path(str(body["stdout_path"])), expected_csv=output,
        plan=plan, arm=arm, argv=invocation.argv)
    _revalidate_rocprofv3_raw_artifacts(invocation, body.get("raw_artifacts"))
    if (body.get("dispatch_row_count") != len(dispatches)
            or body.get("timestamp_reduction_sha256") != schemas.content_hash(dispatches)
            or body.get("structural_fingerprint_sha256")
            != _profiler_structural_fingerprint(dispatches)
            or body.get("exact_dispatch_signatures") != reduction["exact"]
            or body.get("structural_dispatch_signatures")
               != reduction["structural_exact"]
            or body.get("forbidden_dispatch_signatures") != reduction["forbidden"]
            or body.get("invariant_signatures") != reduction["invariants"]
            or body.get("benchmark_completion") != completion
            or body.get("gpu_agent_identity") != agent_info):
        raise EvidenceProducerError("profiler transport attempt reduction changed")
    _validate_residency_witness(body.get("residency_witness"),
                                device_id=plan.device_id, label=arm)
    if plan.shared_runtime is not None:
        _validate_runtime_maps_receipt(
            body.get("runtime_maps_identity"), plan=plan, arm=arm,
            residency=body.get("residency_witness"))
    return loaded, dispatches


def _execute_profiler_attempt(
        *, directory: Path, attempt_number: int, arm: str,
        plan: GpuSourceEvidencePlan, executor: CommandExecutor,
        claim_acquirer: Callable[..., Any],
        claim_verifier: Callable[[Mapping[str, Any]], object], claim_journal: Any,
        claim_timeout_s: float) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    invocation = _rocprofv3_invocation(directory, attempt_number, arm, plan)
    capture, opened, released, residency = _run_claimed(
        invocation, plan=plan, executor=executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=claim_timeout_s)
    if capture.exit_code != 0:
        _seal_profiler_failure_attempt(
            directory=directory, attempt_number=attempt_number, arm=arm,
            plan=plan, invocation=invocation, capture=capture, opened=opened,
            released=released, residency=residency)
        raise EvidenceProducerError(f"{arm} rocprofv3 command exited nonzero")
    # Map/CSV/completion authority is evaluated before minting a clean
    # transport receipt; no malformed trace or wrong target cell can pass.
    outputs = _output_hashes(invocation)
    assert invocation.timestamp_csv_path is not None
    dispatches = _load_arm_dispatches(invocation.timestamp_csv_path, plan, arm=arm)
    agent_info = _load_rocprofv3_agent_info(
        _rocprofv3_agent_info_path(invocation.timestamp_csv_path),
        trace_agent_ids={int(row["agent_id"]) for row in dispatches})
    exact = plan.dispatch.candidate_exact if arm == "candidate" else plan.dispatch.anchor_exact
    forbidden = (plan.dispatch.candidate_forbidden if arm == "candidate"
                 else plan.dispatch.anchor_forbidden)
    runtime_maps = _validated_runtime_maps_identity(
        capture, plan=plan, arm=arm, residency=residency)
    completion = _parse_profile_completion(
        invocation.stdout_path, expected_csv=invocation.timestamp_csv_path,
        plan=plan, arm=arm, argv=invocation.argv)
    try:
        reduction = _reduce_arm(
            dispatches, exact=exact,
            structural_exact=_arm_structural_exact(plan, arm), forbidden=forbidden,
            invariants=plan.dispatch.invariants)
    except EvidenceProducerError as exc:
        identity = plan.candidate if arm == "candidate" else plan.anchor
        refusal = _seal(directory / "refusal.json", {
            "schema": ATTRIBUTION_REFUSAL_SCHEMA,
            "authority": AUTHORITY, "promotion_claim": False,
            "status": "refused",
            "classification": "attribution_route_falsified",
            "error_type": "DispatchAttributionParseRefusal",
            "reason": _durable_refusal_reason(exc),
            "arm": arm, "campaign_id": plan.campaign_id,
            "device_id": plan.device_id,
            "manifest_sha256": plan.manifest_sha256,
            "build_identity": asdict(identity),
            "model_sha256": plan.model_sha256,
            "workload_sha256": plan.workload_sha256,
            "runtime_config_sha256": plan.runtime_config_sha256,
            "profiler_trace_schema_id": plan.profiler_trace_schema_id,
            "profiler_transport_policy": plan.profiler_transport_policy,
            "expected_profiler_dispatch_rows": (
                plan.expected_candidate_profiler_dispatch_rows
                if arm == "candidate" else
                plan.expected_anchor_profiler_dispatch_rows),
            "transport_outcome": "clean_exit", "retry_eligible": False,
            "command_argv": list(invocation.argv),
            "command_cwd": str(plan.execution_cwd),
            "command_environment_sha256": schemas.content_hash(
                [list(item) for item in invocation.environment]),
            "exit_code": 0, **outputs,
            "raw_artifacts": _rocprofv3_raw_artifacts(invocation),
            "timestamp_reduction_sha256": schemas.content_hash(dispatches),
            "structural_fingerprint_sha256":
                _profiler_structural_fingerprint(dispatches),
            "benchmark_completion": dict(completion),
            "gpu_agent_identity": dict(agent_info),
            "started_at": capture.started_at, "ended_at": capture.ended_at,
            "expectations": _expectations(plan),
            **_claim_boundary_fields(opened, released, residency),
            "residency_witness": residency,
            "runtime_maps_identity": runtime_maps,
        })
        raise DispatchAttributionParseRefusal(
            str(exc), receipt_path=str(refusal["path"]),
            receipt_sha256=str(refusal["file_sha256"])) from exc
    loaded = _seal_profiler_attempt(
        directory=directory, attempt_number=attempt_number, arm=arm,
        plan=plan, invocation=invocation, capture=capture, opened=opened,
        released=released, residency=residency, outputs=outputs,
        dispatches=dispatches, reduction=reduction, runtime_maps=runtime_maps,
        completion=completion, agent_info=agent_info)
    return loaded, dispatches


def _v3_attempt_or_execute(
        *, directory: Path, attempt_number: int, arm: str,
        plan: GpuSourceEvidencePlan, executor: CommandExecutor,
        claim_acquirer: Callable[..., Any],
        claim_verifier: Callable[[Mapping[str, Any]], object], claim_journal: Any,
        claim_timeout_s: float) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    attempt_root = directory / f"attempt-{attempt_number:02d}"
    receipt = attempt_root / "transport.json"
    if receipt.exists() or receipt.is_symlink():
        return _load_profiler_attempt(
            receipt, attempt_number=attempt_number, arm=arm, plan=plan)
    if attempt_root.exists() or attempt_root.is_symlink():
        raise EvidenceProducerError(
            "profiler attempt has raw bytes without a sealed transport outcome")
    return _execute_profiler_attempt(
        directory=directory, attempt_number=attempt_number, arm=arm,
        plan=plan, executor=executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=claim_timeout_s)


def _produce_attribution_arm_v3(
        root: Path, arm: str, plan: GpuSourceEvidencePlan,
        executor: CommandExecutor, *, claim_acquirer: Callable[..., Any],
        claim_verifier: Callable[[Mapping[str, Any]], object], claim_journal: Any,
        claim_timeout_s: float) -> Mapping[str, Any]:
    directory = root / f"attribution-{arm}"
    attempt, dispatches = _v3_attempt_or_execute(
        directory=directory, attempt_number=1, arm=arm, plan=plan,
        executor=executor, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier, claim_journal=claim_journal,
        claim_timeout_s=claim_timeout_s)
    body_attempt = attempt["body"]
    if (body_attempt.get("exit_code") != 0
            or body_attempt.get("transport_outcome") != "clean_exit"):
        raise EvidenceProducerError("rocprofv3 attribution lacks a clean transport")
    identity = plan.candidate if arm == "candidate" else plan.anchor
    inputs = (plan.candidate_rocprof_inputs if arm == "candidate"
              else plan.anchor_rocprof_inputs)
    body = {
        "schema": ATTRIBUTION_SCHEMA, "authority": AUTHORITY,
        "non_promotable": True, "promotion_claim": False,
        "status": "complete", "result": "PASS", "arm": arm,
        "campaign_id": plan.campaign_id, "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "build_identity": asdict(identity),
        "identity_files": _identity_files_reference(plan.identity_files),
        "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
        "execution_policy": _bound_reference(plan.policy),
        "command_input_files": [_bound_reference(x) for x in inputs],
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "profiler_trace_schema_id": plan.profiler_trace_schema_id,
        "profiler_transport_policy": plan.profiler_transport_policy,
        "expected_profiler_dispatch_rows": (
            plan.expected_candidate_profiler_dispatch_rows if arm == "candidate"
            else plan.expected_anchor_profiler_dispatch_rows),
        "command_argv": body_attempt["command_argv"],
        "command_cwd": body_attempt["command_cwd"],
        "command_environment_sha256": body_attempt["command_environment_sha256"],
        "exit_code": 0, "transport_outcome": "clean_exit",
        "transport_attempts": [_reference(attempt)],
        "stdout_path": body_attempt["stdout_path"],
        "stdout_sha256": body_attempt["stdout_sha256"],
        "stderr_path": body_attempt["stderr_path"],
        "stderr_sha256": body_attempt["stderr_sha256"],
        "timestamp_csv_path": body_attempt["timestamp_csv_path"],
        "timestamp_csv_sha256": body_attempt["timestamp_csv_sha256"],
        "raw_artifacts": body_attempt["raw_artifacts"],
        "timestamp_reduction_sha256": body_attempt["timestamp_reduction_sha256"],
        "started_at": body_attempt["started_at"],
        "ended_at": body_attempt["ended_at"], "dispatches": dispatches,
        "exact_dispatch_signatures": body_attempt["exact_dispatch_signatures"],
        "structural_dispatch_signatures":
            body_attempt["structural_dispatch_signatures"],
        "forbidden_dispatch_signatures": body_attempt["forbidden_dispatch_signatures"],
        "invariant_signatures": body_attempt["invariant_signatures"],
        "structural_fingerprint_sha256": body_attempt["structural_fingerprint_sha256"],
        "benchmark_completion": body_attempt["benchmark_completion"],
        "gpu_agent_identity": body_attempt["gpu_agent_identity"],
        **{key: body_attempt[key] for key in (
            "device_claim_open", "device_claim_mode",
            "device_claim_released", "device_claim_borrowed_phase_end")
           if key in body_attempt},
        "residency_witness": body_attempt["residency_witness"],
        "runtime_maps_identity": body_attempt["runtime_maps_identity"],
    }
    return _seal(directory / "receipt.json", body)


def _produce_attribution_arm(
    root: Path, arm: str, plan: GpuSourceEvidencePlan,
    executor: CommandExecutor, *, claim_acquirer: Callable[..., Any],
    claim_verifier: Callable[[Mapping[str, Any]], object], claim_journal: Any,
    claim_timeout_s: float,
) -> Mapping[str, Any]:
    if plan.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
        return _produce_attribution_arm_v3(
            root, arm, plan, executor, claim_acquirer=claim_acquirer,
            claim_verifier=claim_verifier, claim_journal=claim_journal,
            claim_timeout_s=claim_timeout_s)
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
    try:
        reduction = _reduce_arm(
            dispatches, exact=exact,
            structural_exact=_arm_structural_exact(plan, arm), forbidden=forbidden,
            invariants=plan.dispatch.invariants)
    except EvidenceProducerError as exc:
        refusal = _seal(directory / "refusal.json", {
            "schema": ATTRIBUTION_REFUSAL_SCHEMA,
            "authority": AUTHORITY, "promotion_claim": False,
            "status": "refused",
            "classification": "attribution_route_falsified",
            "error_type": "DispatchAttributionParseRefusal",
            "reason": _durable_refusal_reason(exc),
            "arm": arm, "campaign_id": plan.campaign_id,
            "device_id": plan.device_id,
            "manifest_sha256": plan.manifest_sha256,
            "build_identity": asdict(identity),
            "model_sha256": plan.model_sha256,
            "workload_sha256": plan.workload_sha256,
            "runtime_config_sha256": plan.runtime_config_sha256,
            "command_argv": list(argv),
            "command_cwd": str(plan.execution_cwd),
            "command_environment_sha256": schemas.content_hash([
                list(item) for item in (
                    plan.candidate_rocprof_environment if arm == "candidate"
                    else plan.anchor_rocprof_environment)]),
            "exit_code": capture.exit_code, **outputs,
            "started_at": capture.started_at, "ended_at": capture.ended_at,
            "expectations": _expectations(plan),
            **_claim_boundary_fields(opened, released, residency),
            "residency_witness": residency,
        })
        raise DispatchAttributionParseRefusal(
            str(exc), receipt_path=str(refusal["path"]),
            receipt_sha256=str(refusal["file_sha256"])) from exc
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
        "structural_dispatch_signatures": reduction["structural_exact"],
        "forbidden_dispatch_signatures": reduction["forbidden"],
        "invariant_signatures": reduction["invariants"],
        **_claim_boundary_fields(opened, released, residency),
        "residency_witness": residency,
        "runtime_maps_identity": runtime_maps,
    }
    return _seal(directory / "receipt.json", body)


def load_gpu_source_attribution_refusal(
        path: Path, plan: GpuSourceEvidencePlan, *, arm: str) -> Mapping[str, Any]:
    if arm not in {"candidate", "anchor"}:
        raise EvidenceProducerError("attribution refusal arm is invalid")
    _verify_plan_files(plan)
    loaded = proofs.load_receipt(path, schema=ATTRIBUTION_REFUSAL_SCHEMA)
    body = loaded["body"]
    identity = plan.candidate if arm == "candidate" else plan.anchor
    expected = {
        "authority": AUTHORITY, "promotion_claim": False,
        "status": "refused", "classification": "attribution_route_falsified",
        "error_type": "DispatchAttributionParseRefusal", "arm": arm,
        "campaign_id": plan.campaign_id, "device_id": plan.device_id,
        "manifest_sha256": plan.manifest_sha256,
        "build_identity": asdict(identity), "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "expectations": _expectations(plan),
    }
    if plan.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
        expected.update({
            "profiler_trace_schema_id": plan.profiler_trace_schema_id,
            "profiler_transport_policy": plan.profiler_transport_policy,
            "expected_profiler_dispatch_rows": (
                plan.expected_candidate_profiler_dispatch_rows
                if arm == "candidate" else
                plan.expected_anchor_profiler_dispatch_rows),
            "transport_outcome": "clean_exit", "retry_eligible": False,
            "exit_code": 0,
        })
    if (any(body.get(key) != value for key, value in expected.items())
            or not isinstance(body.get("reason"), str) or not body["reason"]):
        raise EvidenceProducerError("attribution refusal identity changed")
    _validate_claim_boundary(body, plan=plan)
    for kind in ("stdout", "stderr", "timestamp_csv"):
        if _hash_file(Path(str(body.get(f"{kind}_path", ""))), kind,
                      allow_empty=kind == "stderr") != body.get(f"{kind}_sha256"):
            raise EvidenceProducerError(f"attribution refusal {kind} changed")
    # Re-derive the exact route failure from the original timestamp bytes.
    # The refusal's self-hash is not allowed to turn a rewritten explanation
    # into scientific evidence.
    timestamp_path = Path(str(body["timestamp_csv_path"]))
    dispatches = _load_arm_dispatches(timestamp_path, plan, arm=arm)
    if plan.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
        invocation = _rocprofv3_invocation(path.parent, 1, arm, plan)
        _revalidate_rocprofv3_raw_artifacts(
            invocation, body.get("raw_artifacts"))
        if (_receipt_rocprof_template(body) != (
                plan.candidate_rocprof_argv if arm == "candidate"
                else plan.anchor_rocprof_argv)
                or body.get("timestamp_reduction_sha256")
                != schemas.content_hash(dispatches)
                or body.get("structural_fingerprint_sha256")
                != _profiler_structural_fingerprint(dispatches)):
            raise EvidenceProducerError(
                "rocprofv3 attribution refusal transport reduction changed")
        agent_info = _load_rocprofv3_agent_info(
            _rocprofv3_agent_info_path(timestamp_path),
            trace_agent_ids={int(row["agent_id"]) for row in dispatches})
        completion = _parse_profile_completion(
            Path(str(body["stdout_path"])), expected_csv=timestamp_path,
            plan=plan, arm=arm, argv=invocation.argv)
        if (body.get("gpu_agent_identity") != agent_info
                or body.get("benchmark_completion") != completion):
            raise EvidenceProducerError(
                "rocprofv3 attribution refusal completion/agent changed")
        _validate_residency_witness(
            body.get("residency_witness"), device_id=plan.device_id,
            label=f"{arm} refusal")
        if plan.shared_runtime is not None:
            _validate_runtime_maps_receipt(
                body.get("runtime_maps_identity"), plan=plan, arm=arm,
                residency=body.get("residency_witness"))
    exact = (plan.dispatch.candidate_exact if arm == "candidate"
             else plan.dispatch.anchor_exact)
    forbidden = (plan.dispatch.candidate_forbidden if arm == "candidate"
                 else plan.dispatch.anchor_forbidden)
    try:
        _reduce_arm(dispatches, exact=exact,
                    structural_exact=_arm_structural_exact(plan, arm),
                    forbidden=forbidden,
                    invariants=plan.dispatch.invariants)
    except EvidenceProducerError as exc:
        if _durable_refusal_reason(exc) != body["reason"]:
            raise EvidenceProducerError(
                "attribution refusal reason differs from timestamp evidence") from exc
    else:
        raise EvidenceProducerError(
            "attribution refusal timestamp now satisfies the route contract")
    return loaded


def _reference(loaded: Mapping[str, Any]) -> dict[str, Any]:
    return {key: loaded[key] for key in ("path", "file_sha256", "native_sha256", "body")}


def _expectations(plan: GpuSourceEvidencePlan) -> dict[str, Any]:
    return {
        "candidate_exact": [asdict(item) for item in plan.dispatch.candidate_exact],
        "anchor_exact": [asdict(item) for item in plan.dispatch.anchor_exact],
        "candidate_structural_exact": [
            asdict(item) for item in plan.dispatch.candidate_structural_exact],
        "anchor_structural_exact": [
            asdict(item) for item in plan.dispatch.anchor_structural_exact],
        "candidate_forbidden": [asdict(item) for item in plan.dispatch.candidate_forbidden],
        "anchor_forbidden": [asdict(item) for item in plan.dispatch.anchor_forbidden],
        "invariants": [asdict(item) for item in plan.dispatch.invariants],
    }


def _exact_duration_comparison(candidate_body: Mapping[str, Any],
                               anchor_body: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce only contract-authorized routes into the decision-bearing pair."""
    candidate_routes = candidate_body.get("exact_dispatch_signatures")
    anchor_routes = anchor_body.get("exact_dispatch_signatures")
    if not isinstance(candidate_routes, Mapping) or not isinstance(anchor_routes, Mapping):
        raise EvidenceProducerError("attribution pair lacks exact route reductions")
    def total(routes: Mapping[str, Any], label: str) -> int:
        values: list[int] = []
        for signature, row in routes.items():
            if (not isinstance(signature, str) or not isinstance(row, Mapping)
                    or isinstance(row.get("total_duration_ns"), bool)
                    or not isinstance(row.get("total_duration_ns"), int)
                    or row["total_duration_ns"] <= 0
                    or isinstance(row.get("calls"), bool)
                    or not isinstance(row.get("calls"), int)
                    or row["calls"] <= 0):
                raise EvidenceProducerError(
                    f"{label} exact route lacks positive sealed duration/calls")
            values.append(row["total_duration_ns"])
        if not values:
            raise EvidenceProducerError(f"{label} attribution has no exact routes")
        return sum(values)
    candidate_total = total(candidate_routes, "candidate")
    anchor_total = total(anchor_routes, "anchor")
    return {
        "candidate_routes": dict(candidate_routes),
        "anchor_routes": dict(anchor_routes),
        "candidate_total_duration_ns": candidate_total,
        "anchor_total_duration_ns": anchor_total,
        "relative_improvement_fraction": (
            anchor_total - candidate_total) / anchor_total,
        "direction": ("improved" if candidate_total < anchor_total else
                      "regressed" if candidate_total > anchor_total else "neutral"),
        "all_candidate_routes_present": True,
        "all_anchor_routes_present": True,
        "statistic": "sum_exact_route_total_duration_ns",
    }


def _structural_signature_projection(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceProducerError("invariant signature reduction is malformed")
    projected: dict[str, Any] = {}
    for signature, row in value.items():
        if (not isinstance(signature, str) or not isinstance(row, Mapping)
                or isinstance(row.get("calls"), bool)
                or not isinstance(row.get("calls"), int)
                or not isinstance(row.get("geometries"), list)):
            raise EvidenceProducerError("invariant structural signature is malformed")
        projected[signature] = {
            "calls": row["calls"], "geometries": row["geometries"]}
    return projected


def _produce_pair(
    root: Path, plan: GpuSourceEvidencePlan, candidate: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> Mapping[str, Any]:
    correctness_loaded = proofs.load_receipt(
        root / "correctness" / "receipt.json", schema=CORRECTNESS_SCHEMA)
    _validate_correctness_body(correctness_loaded["body"], plan)
    c6_reference = correctness_loaded["body"]["c6_correctness"]
    c6_loaded = _reload_reference(c6_reference, schema=C6_CORRECTNESS_SCHEMA)
    _validate_c6_correctness_receipt(c6_loaded, plan)
    candidate_body, anchor_body = candidate["body"], anchor["body"]
    candidate_invariants = _structural_signature_projection(
        candidate_body["invariant_signatures"])
    anchor_invariants = _structural_signature_projection(
        anchor_body["invariant_signatures"])
    if candidate_invariants != anchor_invariants:
        reason = "candidate changed an invariant hot signature"
        refusal = _seal(root / "attribution-pair-refusal.json", {
            "schema": PAIR_REFUSAL_SCHEMA, "authority": AUTHORITY,
            "promotion_claim": False, "status": "refused",
            "classification": "attribution_route_falsified",
            "error_type": "DispatchAttributionParseRefusal",
            "reason": reason, "manifest_sha256": plan.manifest_sha256,
            "model_sha256": plan.model_sha256,
            "workload_sha256": plan.workload_sha256,
            "runtime_config_sha256": plan.runtime_config_sha256,
            "candidate": _reference(candidate), "anchor": _reference(anchor),
            "candidate_invariant_signatures": candidate_invariants,
            "anchor_invariant_signatures": anchor_invariants,
            "expectations": _expectations(plan),
        })
        raise DispatchAttributionParseRefusal(
            reason, receipt_path=str(refusal["path"]),
            receipt_sha256=str(refusal["file_sha256"]))
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
        "correctness_invocations": [dict(row) for row in plan.correctness_invocations],
        "c6_correctness": dict(c6_reference),
        "c6_correctness_plan": {
            "argv": list(plan.c6_correctness.argv),
            "inputs": [_bound_reference(x) for x in plan.c6_correctness.inputs],
            "precision_contract": dict(plan.c6_correctness.precision_contract),
            "precision_equivalence_policy": dict(
                plan.c6_correctness.precision_equivalence_policy),
            "structural_precision_evidence": dict(
                plan.c6_correctness.structural_precision_evidence),
            "semantic_judge_verdicts": dict(
                plan.c6_correctness.semantic_judge_verdicts),
        },
        "candidate_rocprof_inputs": [_bound_reference(x) for x in plan.candidate_rocprof_inputs],
        "anchor_rocprof_inputs": [_bound_reference(x) for x in plan.anchor_rocprof_inputs],
        "required_correctness_argv_paths": [str(x) for x in plan.required_correctness_argv_paths],
        "required_candidate_rocprof_argv_paths": [str(x) for x in plan.required_candidate_rocprof_argv_paths],
        "required_anchor_rocprof_argv_paths": [str(x) for x in plan.required_anchor_rocprof_argv_paths],
        "execution_cwd": str(plan.execution_cwd),
        "correctness_environment": [list(item) for item in plan.correctness_environment],
        "candidate_rocprof_environment": [list(item) for item in plan.candidate_rocprof_environment],
        "anchor_rocprof_environment": [list(item) for item in plan.anchor_rocprof_environment],
        "attribution_arm_order_seed_sha256": plan.attribution_arm_order_seed_sha256,
        "attribution_arm_order": list(plan.attribution_arm_order),
        "profiler_trace_schema_id": plan.profiler_trace_schema_id,
        "expected_candidate_profiler_dispatch_rows": (
            plan.expected_candidate_profiler_dispatch_rows),
        "expected_anchor_profiler_dispatch_rows": (
            plan.expected_anchor_profiler_dispatch_rows),
        "profiler_transport_policy": plan.profiler_transport_policy,
        "expectations": _expectations(plan),
        "candidate": _reference(candidate),
        "anchor": _reference(anchor),
        "invariant_signatures": candidate_invariants,
        "inverse_attribution_proved": True,
        "candidate_runtime_maps_identity": candidate_body.get("runtime_maps_identity"),
        "anchor_runtime_maps_identity": anchor_body.get("runtime_maps_identity"),
        "exact_duration_comparison": _exact_duration_comparison(
            candidate_body, anchor_body),
    }
    return _seal(root / "attribution-pair.json", body)


def _load_gpu_source_attribution_pair_refusal(
        path: Path, plan: GpuSourceEvidencePlan,
        candidate: Mapping[str, Any], anchor: Mapping[str, Any]) -> Mapping[str, Any]:
    loaded = proofs.load_receipt(path, schema=PAIR_REFUSAL_SCHEMA)
    body = loaded["body"]
    candidate_invariants = _structural_signature_projection(
        candidate["body"].get("invariant_signatures"))
    anchor_invariants = _structural_signature_projection(
        anchor["body"].get("invariant_signatures"))
    expected = {
        "authority": AUTHORITY, "promotion_claim": False,
        "status": "refused", "classification": "attribution_route_falsified",
        "error_type": "DispatchAttributionParseRefusal",
        "reason": "candidate changed an invariant hot signature",
        "manifest_sha256": plan.manifest_sha256,
        "model_sha256": plan.model_sha256,
        "workload_sha256": plan.workload_sha256,
        "runtime_config_sha256": plan.runtime_config_sha256,
        "candidate": _reference(candidate), "anchor": _reference(anchor),
        "candidate_invariant_signatures": candidate_invariants,
        "anchor_invariant_signatures": anchor_invariants,
        "expectations": _expectations(plan),
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError(
            "attribution pair refusal identity/reduction changed")
    if candidate_invariants == anchor_invariants:
        raise EvidenceProducerError(
            "attribution pair refusal no longer has invariant drift")
    return loaded


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
    if plan.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
        expected.update({
            "profiler_trace_schema_id": plan.profiler_trace_schema_id,
            "profiler_transport_policy": plan.profiler_transport_policy,
            "expected_profiler_dispatch_rows": (
                plan.expected_candidate_profiler_dispatch_rows
                if arm == "candidate" else
                plan.expected_anchor_profiler_dispatch_rows),
            "transport_outcome": "clean_exit",
        })
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError(f"{arm} attribution receipt identity/config mismatch")
    _validate_claim_boundary(body, plan=plan)
    for kind in ("stdout", "stderr", "timestamp_csv"):
        path = Path(str(body.get(f"{kind}_path", "")))
        if _hash_file(path, kind, allow_empty=kind != "timestamp_csv") != body.get(f"{kind}_sha256"):
            raise EvidenceProducerError(f"{arm} {kind} bytes changed")
    rows = _load_arm_dispatches(
        Path(str(body["timestamp_csv_path"])), plan, arm=arm)
    if rows != body.get("dispatches") or schemas.content_hash(rows) != body.get("timestamp_reduction_sha256"):
        raise EvidenceProducerError(f"{arm} timestamp reduction changed")
    exact = plan.dispatch.candidate_exact if arm == "candidate" else plan.dispatch.anchor_exact
    forbidden = (plan.dispatch.candidate_forbidden if arm == "candidate"
                 else plan.dispatch.anchor_forbidden)
    reduction = _reduce_arm(rows, exact=exact,
                            structural_exact=_arm_structural_exact(plan, arm),
                            forbidden=forbidden,
                            invariants=plan.dispatch.invariants)
    if (body.get("exact_dispatch_signatures") != reduction["exact"]
            or body.get("structural_dispatch_signatures")
               != reduction["structural_exact"]
            or body.get("forbidden_dispatch_signatures") != reduction["forbidden"]
            or body.get("invariant_signatures") != reduction["invariants"]):
        raise EvidenceProducerError(f"{arm} dispatch derivation mismatch")
    _validate_residency_witness(
        body.get("residency_witness"), device_id=plan.device_id, label=arm)
    if plan.shared_runtime is not None:
        _validate_runtime_maps_receipt(body.get("runtime_maps_identity"), plan=plan, arm=arm,
                                       residency=body.get("residency_witness"))
    if plan.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
        attempts = body.get("transport_attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise EvidenceProducerError(f"{arm} rocprofv3 attempt receipt is missing")
        loaded_attempt, attempt_rows = _load_profiler_attempt(
            Path(str(attempts[0].get("path", ""))), attempt_number=1,
            arm=arm, plan=plan)
        if (_reference(loaded_attempt) != attempts[0] or attempt_rows != rows
                or body.get("raw_artifacts") != loaded_attempt["body"].get("raw_artifacts")
                or body.get("structural_fingerprint_sha256")
                != loaded_attempt["body"].get("structural_fingerprint_sha256")
                or body.get("benchmark_completion")
                != loaded_attempt["body"].get("benchmark_completion")
                or body.get("gpu_agent_identity")
                != loaded_attempt["body"].get("gpu_agent_identity")):
            raise EvidenceProducerError(f"{arm} rocprofv3 attempt projection changed")


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
    if plan.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID:
        inputs = (plan.candidate_rocprof_inputs if arm == "candidate"
                  else plan.anchor_rocprof_inputs)
        required_profiler = _required_profiler_mapped_files(inputs)
        mapped = dict(typed.mapped_local_sha256)
        if any(mapped.get(path) != digest
               for path, digest in required_profiler.items()):
            raise EvidenceProducerError(
                f"{arm} runtime maps omit the sealed rocprofv3 DSO closure")
    return typed.to_dict()


def _validate_correctness_body(body: Mapping[str, Any], plan: GpuSourceEvidencePlan) -> None:
    if plan.correctness_invocations:
        expected = {
            "schema": CORRECTNESS_SCHEMA, "authority": AUTHORITY,
            "non_promotable": True, "promotion_claim": False,
            "status": "complete", "result": "PASS",
            "campaign_id": plan.campaign_id, "device_id": plan.device_id,
            "manifest_sha256": plan.manifest_sha256,
            "candidate_build_identity": asdict(plan.candidate),
            "identity_files": _identity_files_reference(plan.identity_files),
            "shared_runtime": _shared_runtime_reference(plan.shared_runtime),
            "execution_policy": _bound_reference(plan.policy),
            "command_input_files": [_bound_reference(x)
                                    for x in plan.correctness_inputs],
            "workload_sha256": plan.workload_sha256,
            "command_argv": list(plan.correctness_argv),
            "command_cwd": str(plan.execution_cwd),
            "correctness_parser_id": CORRECTNESS_PARSER_ID,
            "correctness_backend": plan.correctness_backend,
            "correctness_op": plan.correctness_op,
            "expected_cases": plan.expected_correctness_cases,
            "correctness_invocation_contracts": [dict(row)
                                                 for row in plan.correctness_invocations],
            "exact_case_ok": True,
        }
        if any(body.get(key) != value for key, value in expected.items()):
            raise EvidenceProducerError(
                "aggregate correctness receipt identity/config/result mismatch")
        c6_reference = body.get("c6_correctness")
        if not isinstance(c6_reference, Mapping):
            raise EvidenceProducerError(
                "aggregate correctness lacks sealed C6 receipt")
        c6_loaded = _reload_reference(
            c6_reference, schema=C6_CORRECTNESS_SCHEMA)
        _validate_c6_correctness_receipt(c6_loaded, plan)
        references = body.get("invocations")
        if not isinstance(references, list) or len(references) != len(
                plan.correctness_invocations):
            raise EvidenceProducerError(
                "aggregate correctness receipt has incomplete invocations")
        for reference, contract in zip(references,
                                       plan.correctness_invocations):
            loaded = _reload_reference(reference, schema=CORRECTNESS_SCHEMA)
            _validate_correctness_invocation_receipt(loaded, plan, contract)
        return
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
    c6_reference = body.get("c6_correctness")
    if not isinstance(c6_reference, Mapping):
        raise EvidenceProducerError("targeted correctness lacks sealed C6 receipt")
    c6_loaded = _reload_reference(c6_reference, schema=C6_CORRECTNESS_SCHEMA)
    _validate_c6_correctness_receipt(c6_loaded, plan)
    join = body.get("c6_claim_join")
    if join not in {"same_held_claim", "sealed_c6_restart"}:
        raise EvidenceProducerError("targeted correctness lacks its C6 stage join")
    if (join == "same_held_claim"
            and c6_loaded["body"].get("device_claim_open") !=
                body.get("device_claim_open")):
        raise EvidenceProducerError(
            "targeted correctness did not reuse the held native C6 claim")
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


def _validate_c6_split_receipt_body(
        body: Mapping[str, Any], c6: C6CorrectnessPlan,
        plan: GpuSourceEvidencePlan) -> None:
    """Re-open one split-mode C6 receipt without executing any artifact."""
    sidecar = body.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise EvidenceProducerError("sealed C6 oracle sidecar reference is malformed")
    sidecar_path = Path(str(sidecar.get("path", "")))
    if _hash_file(sidecar_path, "C6 oracle sidecar", allow_empty=False) \
            != sidecar.get("sha256"):
        raise EvidenceProducerError("sealed C6 oracle sidecar changed")
    try:
        value = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError("sealed C6 oracle sidecar is unreadable") from exc
    case_identity = _c6_case_identity_from_argv(c6.argv)
    _validate_c6_oracle_sidecar_identity(
        value, case_identity, label="sealed C6 oracle")
    if plan.correctness_op != case_identity["operation"]:
        raise EvidenceProducerError(
            "sealed C6 operation differs from targeted correctness")
    by_name, ref_raw, reference, ref_f64_raw, reference_f64 = (
        _decode_c6_oracle_payload(value, case_identity, label="sealed C6"))
    operation = case_identity["operation"]
    output_elements = (
        case_identity["m"] * case_identity["n"] * 14
        if operation == "FLASH_ATTN_EXT"
        else case_identity["m"] * case_identity["n"])
    oracle_output = body.get("oracle_output")
    if not isinstance(oracle_output, Mapping):
        raise EvidenceProducerError("sealed C6 oracle output reference is malformed")
    for kind, allow_empty in (("stdout", False), ("stderr", True)):
        path = Path(str(oracle_output.get(f"{kind}_path", "")))
        if _hash_file(path, f"C6 oracle {kind}", allow_empty=allow_empty) \
                != oracle_output.get(f"{kind}_sha256"):
            raise EvidenceProducerError(f"sealed C6 oracle {kind} changed")
    input_binding = body.get("input_binding")
    if (not isinstance(input_binding, Mapping)
            or not isinstance(input_binding.get("path"), str)
            or not isinstance(input_binding.get("sha256"), str)
            or not isinstance(input_binding.get("input_identity_sha256"), str)):
        raise EvidenceProducerError(
            "sealed C6 input binding reference is malformed")
    manifest_path = Path(input_binding["path"])
    if _hash_file(manifest_path, "C6 input binding", allow_empty=False) \
            != input_binding["sha256"]:
        raise EvidenceProducerError("sealed C6 input binding changed")
    _reopen_c6_input_binding(
        manifest_path, by_name, case_identity, label="sealed C6")
    ordered_inputs = [by_name[name] for name in _C6_INPUT_ORDER[operation]]
    if input_binding["input_identity_sha256"] != hashlib.sha256(
            b"".join(ordered_inputs)).hexdigest():
        raise EvidenceProducerError(
            "sealed C6 input binding identity changed")
    per_leg = body.get("per_leg_bindings")
    if not isinstance(per_leg, list) or len(per_leg) != 3:
        raise EvidenceProducerError(
            "sealed C6 requires exactly three candidate leg bindings")
    raw_outputs: list[bytes] = []
    for index, binding in enumerate(per_leg, 1):
        if (not isinstance(binding, Mapping)
                or binding.get("schema") != C6_LEG_BINDING_SCHEMA
                or binding.get("leg_index") != index
                or not isinstance(binding.get("argv"), list)):
            raise EvidenceProducerError(
                f"sealed C6 leg {index} binding is malformed")
        paths = _c6_candidate_paths_from_argv(tuple(binding["argv"]))
        for field, key in (("input directory", "input_dir"),
                           ("output path", "output_path"),
                           ("ready path", "ready_path"),
                           ("continue path", "continue_path")):
            if not isinstance(binding.get(key), str):
                raise EvidenceProducerError(
                    f"sealed C6 leg {index} {field} reference is malformed")
        rebuilt = _c6_candidate_argv_from_argv(
            c6.argv, input_dir=Path(binding["input_dir"]),
            output=Path(binding["output_path"]),
            ready_file=Path(binding["ready_path"]),
            continue_file=Path(binding["continue_path"]))
        if tuple(binding["argv"]) != rebuilt:
            raise EvidenceProducerError(
                f"sealed C6 leg {index} argv differs from deterministic derivation")
        if binding.get("argv_sha256") != hashlib.sha256(
                json.dumps(list(binding["argv"]), sort_keys=True).encode()
        ).hexdigest():
            raise EvidenceProducerError(
                f"sealed C6 leg {index} argv digest changed")
        for kind, allow_empty in (("stdout", False), ("stderr", True)):
            path = Path(str(binding.get(f"{kind}_path", "")))
            if _hash_file(path, f"C6 leg {index} {kind}",
                          allow_empty=allow_empty) \
                    != binding.get(f"{kind}_sha256"):
                raise EvidenceProducerError(
                    f"sealed C6 leg {index} {kind} changed")
        output_path = Path(str(binding.get("output_path", "")))
        if output_path.is_symlink() or not output_path.is_file() \
                or output_path.stat().st_nlink != 1:
            raise EvidenceProducerError(
                f"sealed C6 leg {index} output is not a sealed regular file")
        output_bytes = output_path.read_bytes()
        if (binding.get("output_sha256") != hashlib.sha256(
                output_bytes).hexdigest()
                or binding.get("output_length") != len(output_bytes)
                or len(output_bytes) != output_elements * 4):
            raise EvidenceProducerError(
                f"sealed C6 leg {index} output changed")
        for kind in ("ready", "continue"):
            path = Path(str(binding.get(f"{kind}_path", "")))
            if (path.is_symlink() or not path.is_file()
                    or path.stat().st_nlink != 1
                    or _hash_file(path, f"C6 leg {index} {kind}")
                    != binding.get(f"{kind}_sha256")):
                raise EvidenceProducerError(
                    f"sealed C6 leg {index} {kind} token changed")
        event = binding.get("event_stream")
        if (not isinstance(event, Mapping)
                or not all(isinstance(event.get(key), int)
                           for key in ("launched_monotonic_ns",
                                       "ready_observed_monotonic_ns",
                                       "continue_written_monotonic_ns",
                                       "completed_monotonic_ns"))
                or not (event["launched_monotonic_ns"]
                        <= event["ready_observed_monotonic_ns"]
                        <= event["continue_written_monotonic_ns"]
                        <= event["completed_monotonic_ns"])):
            raise EvidenceProducerError(
                f"sealed C6 leg {index} event stream is not monotonic")
        residency = binding.get("residency")
        child_pid = binding.get("child_pid")
        if (not isinstance(residency, Mapping)
                or not isinstance(residency.get("overlap_sample_count"), int)
                or residency["overlap_sample_count"] < 1
                or not isinstance(residency.get("kfd_pids"), list)
                or not residency["kfd_pids"]
                or any(isinstance(pid, bool) or not isinstance(pid, int)
                       or pid < 1 for pid in residency["kfd_pids"])
                or not isinstance(residency.get("max_vram_bytes"), int)
                or residency["max_vram_bytes"] < 0
                or not isinstance(child_pid, int) or isinstance(child_pid, bool)
                or child_pid < 1
                or child_pid not in residency["kfd_pids"]):
            raise EvidenceProducerError(
                f"sealed C6 leg {index} residency summary is malformed")
        raw_outputs.append(output_bytes)
    policy = c6_reward_integrity.PrecisionContract(**dict(c6.precision_contract))
    structural = c6_reward_integrity.StructuralPrecisionEvidence(
        **dict(c6.structural_precision_evidence))
    candidate_outputs = [list(struct.unpack(
        f"<{output_elements}f", item)) for item in raw_outputs]
    numerical = [c6_reward_integrity.evaluate_numerics(
        reference, candidate, structural=structural, policy=policy)
        for candidate in candidate_outputs]
    precision_policy = hawkeye_measurement.PrecisionEquivalencePolicy(
        **dict(c6.precision_equivalence_policy))
    precision_equivalence = [hawkeye_measurement.evaluate_precision_equivalence(
        reference_f64, candidate, policy=precision_policy,
        observed_output_dtype=structural.output_dtype,
        observed_accumulator_dtype=structural.accumulator_dtype)
        for candidate in candidate_outputs]
    deterministic = c6_reward_integrity.determinism_from_recorded_outputs(
        raw_outputs)
    if (body.get("input_identity_sha256") != hashlib.sha256(
                b"".join(ordered_inputs)).hexdigest()
            or body.get("reference_output_sha256") != hashlib.sha256(ref_raw).hexdigest()
            or body.get("reference_float64_output_sha256") !=
               hashlib.sha256(ref_f64_raw).hexdigest()
            or body.get("candidate_output_sha256") != [
                hashlib.sha256(item).hexdigest() for item in raw_outputs]
            or body.get("numeric_verdicts") != [asdict(item) for item in numerical]
            or body.get("precision_equivalence") != [
                hawkeye_measurement.serialize_carrier(item)
                for item in precision_equivalence]
            or body.get("determinism") != json.loads(json.dumps(asdict(deterministic)))
            or not all(item.correct for item in numerical)
            or not all(item.correct for item in precision_equivalence)
            or not deterministic.correct):
        raise EvidenceProducerError("sealed C6 reductions changed or no longer pass")


def _validate_c6_correctness_receipt(
        loaded: Mapping[str, Any], plan: GpuSourceEvidencePlan) -> None:
    c6 = plan.c6_correctness
    if c6 is None:
        raise EvidenceProducerError("C6 receipt cannot reopen without capability")
    body = loaded["body"]
    expected = {
        "schema": C6_CORRECTNESS_SCHEMA, "authority": AUTHORITY,
        "status": "complete", "result": "PASS", "non_promotable": True,
        "promotion_claim": False, "campaign_id": plan.campaign_id,
        "device_id": plan.device_id, "manifest_sha256": plan.manifest_sha256,
        "candidate_build_identity": asdict(plan.candidate),
        "post_run_compiled_source_sha256": plan.candidate.source_sha256,
        "command_cwd": str(plan.execution_cwd), "exit_code": 0,
        "seeded_case_identity": _c6_case_identity_from_argv(c6.argv),
        "precision_contract": asdict(c6_reward_integrity.PrecisionContract(
            **dict(c6.precision_contract))),
        "precision_equivalence_policy": asdict(
            hawkeye_measurement.PrecisionEquivalencePolicy(
                **dict(c6.precision_equivalence_policy))),
        "structural_precision_evidence": asdict(
            c6_reward_integrity.StructuralPrecisionEvidence(
                **dict(c6.structural_precision_evidence))),
        "semantic_judge_calibration": json.loads(json.dumps(asdict(
            c6_reward_integrity.calibrate_semantic_judge(
                c6.semantic_judge_verdicts)))),
        "semantic_judge_gating": False, "native_execution": True,
        "wrapper_used": False,
        "dispatch_expectations_sha256": schemas.content_hash(_expectations(plan)),
        "c6_inputs": [_bound_reference(item) for item in c6.inputs],
    }
    split_mode = _c6_mode_from_argv(c6.argv) == C6_MODE_ORACLE
    if split_mode:
        expected["c6_process_mode"] = "oracle_candidate_split"
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError("sealed C6 correctness identity/policy changed")
    argv = body.get("command_argv")
    if (not isinstance(argv, list) or len(argv) != len(c6.argv)
            or any(actual != planned for actual, planned in zip(argv, c6.argv)
                   if planned != C6_SIDECAR_OUTPUT)):
        raise EvidenceProducerError("sealed C6 command differs from capability")
    if split_mode:
        _validate_c6_split_receipt_body(body, c6, plan)
        _validate_open_claim_boundary(body, plan=plan)
        _validate_residency_witness(
            body.get("residency_witness"), device_id=plan.device_id, label="C6")
        return
    for kind in ("stdout", "stderr"):
        path = Path(str(body.get(f"{kind}_path", "")))
        if _hash_file(path, f"C6 {kind}", allow_empty=kind == "stderr") \
                != body.get(f"{kind}_sha256"):
            raise EvidenceProducerError(f"sealed C6 {kind} changed")
    sidecar = body.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise EvidenceProducerError("sealed C6 sidecar reference is malformed")
    sidecar_path = Path(str(sidecar.get("path", "")))
    if _hash_file(sidecar_path, "C6 sidecar", allow_empty=False) != sidecar.get("sha256"):
        raise EvidenceProducerError("sealed C6 sidecar changed")
    try:
        value = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProducerError("sealed C6 sidecar is unreadable") from exc
    case_identity = _c6_case_identity_from_argv(c6.argv)
    _validate_c6_sidecar_identity(value, case_identity, label="sealed C6 sidecar")
    if plan.correctness_op != case_identity["operation"]:
        raise EvidenceProducerError(
            "sealed C6 operation differs from targeted correctness")
    (input_raw, ref_raw, reference, ref_f64_raw, reference_f64,
     raw_outputs, candidate_outputs) = (
        _decode_c6_sidecar_payload(
            value, case_identity, label="sealed C6"))
    policy = c6_reward_integrity.PrecisionContract(**dict(c6.precision_contract))
    structural = c6_reward_integrity.StructuralPrecisionEvidence(
        **dict(c6.structural_precision_evidence))
    numerical = [c6_reward_integrity.evaluate_numerics(
        reference, candidate, structural=structural, policy=policy)
        for candidate in candidate_outputs]
    precision_policy = hawkeye_measurement.PrecisionEquivalencePolicy(
        **dict(c6.precision_equivalence_policy))
    precision_equivalence = [hawkeye_measurement.evaluate_precision_equivalence(
        reference_f64, candidate, policy=precision_policy,
        observed_output_dtype=structural.output_dtype,
        observed_accumulator_dtype=structural.accumulator_dtype)
        for candidate in candidate_outputs]
    deterministic = c6_reward_integrity.determinism_from_recorded_outputs(raw_outputs)
    if (body.get("input_identity_sha256") != hashlib.sha256(
                b"".join(input_raw)).hexdigest()
            or body.get("reference_output_sha256") != hashlib.sha256(ref_raw).hexdigest()
            or body.get("reference_float64_output_sha256") !=
               hashlib.sha256(ref_f64_raw).hexdigest()
            or body.get("candidate_output_sha256") != [
                hashlib.sha256(item).hexdigest() for item in raw_outputs]
            or body.get("numeric_verdicts") != [asdict(item) for item in numerical]
            or body.get("precision_equivalence") != [
                hawkeye_measurement.serialize_carrier(item)
                for item in precision_equivalence]
            or body.get("determinism") != json.loads(json.dumps(asdict(deterministic)))
            or not all(item.correct for item in numerical)
            or not all(item.correct for item in precision_equivalence)
            or not deterministic.correct):
        raise EvidenceProducerError("sealed C6 reductions changed or no longer pass")
    _validate_open_claim_boundary(body, plan=plan)
    _validate_residency_witness(
        body.get("residency_witness"), device_id=plan.device_id, label="C6")


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
    c6_reference = body.get("c6_correctness")
    if not isinstance(c6_reference, Mapping):
        raise EvidenceProducerError(
            "correctness parse refusal lacks sealed C6 receipt")
    c6_loaded = _reload_reference(
        c6_reference, schema=C6_CORRECTNESS_SCHEMA)
    _validate_c6_correctness_receipt(c6_loaded, plan)
    join = body.get("c6_claim_join")
    if join not in {"same_held_claim", "sealed_c6_restart"}:
        raise EvidenceProducerError(
            "correctness parse refusal lacks its C6 stage join")
    if (join == "same_held_claim"
            and c6_loaded["body"].get("device_claim_open") !=
                body.get("device_claim_open")):
        raise EvidenceProducerError(
            "correctness parse refusal did not reuse the held native C6 claim")
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


def load_gpu_source_attribution_receipt(
        path: Path, plan: GpuSourceEvidencePlan, *, arm: str) -> Mapping[str, Any]:
    """Re-open one completed attribution arm as an exactly-once stage.

    The raw timestamp CSV is part of the receipt graph, so reusing this stage
    re-runs the authoritative dispatch and duration reduction over the original
    bytes.  It never trusts the presence of a receipt filename alone.
    """
    if arm not in {"candidate", "anchor"}:
        raise EvidenceProducerError("attribution receipt arm is invalid")
    _verify_plan_files(plan)
    try:
        loaded = proofs.load_receipt(path, schema=ATTRIBUTION_SCHEMA)
    except proofs.ProofError as exc:
        raise EvidenceProducerError(
            f"completed {arm} attribution receipt is not durably recoverable") from exc
    _validate_attribution_body(loaded["body"], plan=plan, arm=arm)
    return loaded


def _load_gpu_source_attribution_pair(
        path: Path, plan: GpuSourceEvidencePlan,
        candidate: Mapping[str, Any], anchor: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        loaded = proofs.load_receipt(path, schema=PAIR_SCHEMA)
    except proofs.ProofError as exc:
        raise EvidenceProducerError(
            "completed attribution pair is not durably recoverable") from exc
    body = loaded["body"]
    c6_plan = plan.c6_correctness
    if c6_plan is None:
        raise EvidenceProducerError("attribution pair lacks native C6 plan")
    expected_c6_plan = {
        "argv": list(c6_plan.argv),
        "inputs": [_bound_reference(x) for x in c6_plan.inputs],
        "precision_contract": dict(c6_plan.precision_contract),
        "precision_equivalence_policy": dict(
            c6_plan.precision_equivalence_policy),
        "structural_precision_evidence": dict(
            c6_plan.structural_precision_evidence),
        "semantic_judge_verdicts": dict(c6_plan.semantic_judge_verdicts),
    }
    expected = {
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
        "correctness_invocations": [dict(row) for row in plan.correctness_invocations],
        "c6_correctness_plan": expected_c6_plan,
        "expectations": _expectations(plan),
        "candidate": _reference(candidate),
        "anchor": _reference(anchor),
        "attribution_arm_order_seed_sha256": plan.attribution_arm_order_seed_sha256,
        "attribution_arm_order": list(plan.attribution_arm_order),
        "profiler_trace_schema_id": plan.profiler_trace_schema_id,
        "expected_candidate_profiler_dispatch_rows": (
            plan.expected_candidate_profiler_dispatch_rows),
        "expected_anchor_profiler_dispatch_rows": (
            plan.expected_anchor_profiler_dispatch_rows),
        "profiler_transport_policy": plan.profiler_transport_policy,
        "inverse_attribution_proved": True,
    }
    if any(body.get(key) != value for key, value in expected.items()):
        raise EvidenceProducerError(
            "completed attribution pair identity/contract changed")
    c6_loaded = _reload_reference(
        body.get("c6_correctness", {}), schema=C6_CORRECTNESS_SCHEMA)
    _validate_c6_correctness_receipt(c6_loaded, plan)
    candidate_body, anchor_body = candidate["body"], anchor["body"]
    candidate_invariants = _structural_signature_projection(
        candidate_body.get("invariant_signatures"))
    anchor_invariants = _structural_signature_projection(
        anchor_body.get("invariant_signatures"))
    if (body.get("invariant_signatures") != candidate_invariants
            or candidate_invariants != anchor_invariants):
        raise EvidenceProducerError(
            "completed attribution pair changed an invariant signature")
    if body.get("exact_duration_comparison") != _exact_duration_comparison(
            candidate_body, anchor_body):
        raise EvidenceProducerError(
            "completed attribution pair exact-duration comparison changed")
    return loaded


def _contract_from_dict(value: Mapping[str, Any]) -> DispatchContract:
    try:
        return DispatchContract(
            candidate_exact=tuple(ExactDispatch(**row) for row in value["candidate_exact"]),
            anchor_exact=tuple(ExactDispatch(**row) for row in value["anchor_exact"]),
            candidate_structural_exact=tuple(
                ExactDispatch(**row) for row in value["candidate_structural_exact"]),
            anchor_structural_exact=tuple(
                ExactDispatch(**row) for row in value["anchor_structural_exact"]),
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
            attribution_arm_order_seed_sha256=str(
                pair_body["attribution_arm_order_seed_sha256"]),
            attribution_arm_order=tuple(pair_body["attribution_arm_order"]),
            profiler_trace_schema_id=str(pair_body.get(
                "profiler_trace_schema_id", ROCPROF_V1_TRACE_ID)),
            expected_candidate_profiler_dispatch_rows=pair_body.get(
                "expected_candidate_profiler_dispatch_rows"),
            expected_anchor_profiler_dispatch_rows=pair_body.get(
                "expected_anchor_profiler_dispatch_rows"),
            profiler_transport_policy=str(pair_body.get(
                "profiler_transport_policy", "require-zero-exit")),
            correctness_invocations=tuple(
                dict(row) for row in pair_body.get("correctness_invocations", [])),
            c6_correctness=C6CorrectnessPlan(
                argv=tuple(pair_body["c6_correctness_plan"]["argv"]),
                inputs=tuple(_bound_from_dict(x) for x in
                             pair_body["c6_correctness_plan"]["inputs"]),
                precision_contract=dict(pair_body[
                    "c6_correctness_plan"]["precision_contract"]),
                precision_equivalence_policy=dict(pair_body[
                    "c6_correctness_plan"]["precision_equivalence_policy"]),
                structural_precision_evidence=dict(pair_body[
                    "c6_correctness_plan"]["structural_precision_evidence"]),
                semantic_judge_verdicts=dict(pair_body[
                    "c6_correctness_plan"]["semantic_judge_verdicts"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceProducerError("sealed bundle cannot reconstruct its plan") from exc
    return plan


def produce_gpu_source_evidence(
    *, output_root: Path, plan: GpuSourceEvidencePlan,
    correctness_executor: CommandExecutor, rocprof_executor: CommandExecutor,
    claim_journal: Any, claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    claim_verifier: Callable[[Mapping[str, Any]], object] = _default_claim_verifier,
    claim_timeout_s: float = 300.0, c6_ready_timeout_s: float = 120.0,
) -> proofs.GpuSourceProofBundle:
    """Execute or exactly-once resume the ordered GPU proof stages.

    Every completed stage is recursively revalidated and reused.  Only the
    first incomplete stage may execute.  A directory containing raw output but
    no terminal receipt is deliberately ambiguous: replaying it could perform
    a GPU command twice, so the producer refuses instead.
    """
    root = output_root.resolve()
    if output_root.is_symlink() or (root.exists() and (root.is_symlink()
                                                       or not root.is_dir())):
        raise EvidenceProducerError("output_root must be a real directory")
    if (isinstance(claim_timeout_s, bool) or not isinstance(claim_timeout_s, (int, float))
            or not math.isfinite(claim_timeout_s) or claim_timeout_s < 0):
        raise EvidenceProducerError("claim timeout must be finite and non-negative")
    _verify_plan_files(plan)
    bundle_path = root / "proof-bundle.json"
    if bundle_path.exists() or bundle_path.is_symlink():
        return load_gpu_source_evidence_bundle(bundle_path)
    root.mkdir(parents=True, exist_ok=True)

    correctness_dir = root / "correctness"
    correctness_path = correctness_dir / "receipt.json"
    refusal_path = correctness_dir / "refusal.json"
    later_paths = (
        root / "attribution-candidate", root / "attribution-anchor",
        root / "attribution-pair.json", root / "attribution-pair-refusal.json")
    if correctness_path.exists() or correctness_path.is_symlink():
        if refusal_path.exists() or refusal_path.is_symlink():
            raise EvidenceProducerError(
                "correctness stage has contradictory pass/refusal receipts")
        correctness = load_gpu_source_correctness_receipt(correctness_path, plan)
    elif refusal_path.exists() or refusal_path.is_symlink():
        loaded_refusal = load_gpu_source_correctness_refusal(refusal_path, plan)
        raise CorrectnessParseRefusal(
            str(loaded_refusal["body"]["reason"]),
            receipt_path=str(loaded_refusal["path"]),
            receipt_sha256=str(loaded_refusal["file_sha256"]))
    else:
        sealed_c6_boundary = False
        if (correctness_dir.exists() and correctness_dir.is_dir()
                and not correctness_dir.is_symlink()
                and not plan.correctness_invocations
                and plan.c6_correctness is not None):
            mode = _c6_mode_from_argv(plan.c6_correctness.argv)
            if mode == C6_MODE_ORACLE:
                expected = {"c6-receipt.json", "c6-oracle-sidecar.json",
                            "c6-oracle-stdout.txt", "c6-oracle-stderr.txt",
                            "c6-input-binding.json", "c6-inputs"}
                for index in (1, 2, 3):
                    for suffix in ("stdout.txt", "stderr.txt", "output.bin",
                                   "ready", "continue"):
                        expected.add(f"c6-candidate-{index}-{suffix}")
            else:
                expected = {
                    "c6-receipt.json", "c6-sidecar.json",
                    "c6-stdout.txt", "c6-stderr.txt"}
            sealed_c6_boundary = {
                path.name for path in correctness_dir.iterdir()
            } == expected
        if (((correctness_dir.exists() or correctness_dir.is_symlink())
             and not plan.correctness_invocations and not sealed_c6_boundary)
                or any(path.exists() or path.is_symlink() for path in later_paths)):
            raise EvidenceProducerError(
                "correctness stage is incomplete or later evidence exists out of order")
        correctness = _produce_correctness(
            root, plan, correctness_executor, claim_acquirer=claim_acquirer,
            claim_verifier=claim_verifier, claim_journal=claim_journal,
            claim_timeout_s=float(claim_timeout_s),
            c6_ready_timeout_s=float(c6_ready_timeout_s))

    arm_receipts: dict[str, Mapping[str, Any]] = {}
    for index, arm in enumerate(plan.attribution_arm_order):
        arm_dir = root / f"attribution-{arm}"
        arm_path = arm_dir / "receipt.json"
        arm_refusal_path = arm_dir / "refusal.json"
        if arm_path.exists() or arm_path.is_symlink():
            if arm_refusal_path.exists() or arm_refusal_path.is_symlink():
                raise EvidenceProducerError(
                    f"{arm} attribution has contradictory terminals")
            arm_receipts[arm] = load_gpu_source_attribution_receipt(
                arm_path, plan, arm=arm)
            continue
        if arm_refusal_path.exists() or arm_refusal_path.is_symlink():
            refusal = load_gpu_source_attribution_refusal(
                arm_refusal_path, plan, arm=arm)
            raise DispatchAttributionParseRefusal(
                str(refusal["body"]["reason"]),
                receipt_path=str(refusal["path"]),
                receipt_sha256=str(refusal["file_sha256"]))
        later_arms = plan.attribution_arm_order[index + 1:]
        v3_transport = arm_dir / "attempt-01" / "transport.json"
        if (plan.profiler_trace_schema_id == ROCPROF_V3_TRACE_ID
                and (v3_transport.exists() or v3_transport.is_symlink())):
            if (any((root / f"attribution-{later}").exists()
                    or (root / f"attribution-{later}").is_symlink()
                    for later in later_arms)
                    or (root / "attribution-pair.json").exists()
                    or (root / "attribution-pair-refusal.json").exists()):
                raise EvidenceProducerError(
                    f"{arm} attribution has later evidence out of order")
            arm_receipts[arm] = _produce_attribution_arm(
                root, arm, plan, rocprof_executor,
                claim_acquirer=claim_acquirer,
                claim_verifier=claim_verifier,
                claim_journal=claim_journal,
                claim_timeout_s=float(claim_timeout_s))
            continue
        if (arm_dir.exists() or arm_dir.is_symlink()
                or any((root / f"attribution-{later}").exists()
                       or (root / f"attribution-{later}").is_symlink()
                       for later in later_arms)
                or (root / "attribution-pair.json").exists()
                or (root / "attribution-pair.json").is_symlink()
                or (root / "attribution-pair-refusal.json").exists()
                or (root / "attribution-pair-refusal.json").is_symlink()):
            raise EvidenceProducerError(
                f"{arm} attribution is incomplete or later evidence exists out of order")
        arm_receipts[arm] = _produce_attribution_arm(
            root, arm, plan, rocprof_executor,
            claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
            claim_journal=claim_journal, claim_timeout_s=float(claim_timeout_s))

    candidate = arm_receipts["candidate"]
    anchor = arm_receipts["anchor"]

    pair_path = root / "attribution-pair.json"
    pair_refusal_path = root / "attribution-pair-refusal.json"
    if pair_path.exists() or pair_path.is_symlink():
        if pair_refusal_path.exists() or pair_refusal_path.is_symlink():
            raise EvidenceProducerError(
                "attribution pair has contradictory terminals")
        pair = _load_gpu_source_attribution_pair(
            pair_path, plan, candidate, anchor)
    elif pair_refusal_path.exists() or pair_refusal_path.is_symlink():
        refusal = _load_gpu_source_attribution_pair_refusal(
            pair_refusal_path, plan, candidate, anchor)
        raise DispatchAttributionParseRefusal(
            str(refusal["body"]["reason"]),
            receipt_path=str(refusal["path"]),
            receipt_sha256=str(refusal["file_sha256"]))
    else:
        pair = _produce_pair(root, plan, candidate, anchor)
    bundle = proofs.GpuSourceProofBundle.from_validated_paths(
        manifest_sha256=plan.manifest_sha256, candidate=plan.candidate,
        anchor=plan.anchor, workload_sha256=plan.workload_sha256,
        correctness=_reference(correctness), attribution=_reference(pair))
    _seal(bundle_path, {
        "schema": SEALED_BUNDLE_SCHEMA,
        "authority": AUTHORITY,
        "promotion_claim": False,
        "bundle": bundle.to_dict(),
    })
    # Re-read the complete graph once before returning it to the controller.
    return load_gpu_source_evidence_bundle(bundle_path)


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
    candidate_invariants = _structural_signature_projection(
        candidate["body"]["invariant_signatures"])
    anchor_invariants = _structural_signature_projection(
        anchor["body"]["invariant_signatures"])
    if (candidate_invariants != anchor_invariants
            or pair_body.get("invariant_signatures") != candidate_invariants
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
