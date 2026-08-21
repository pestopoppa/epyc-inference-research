#!/usr/bin/env python3
"""Hardened Hawkeye-derived measurement substrate for AutoKernel.

This module adopts only the two small, vendor-generic JSON contracts used by
Hawkeye's frozen driver: the tensor manifest it consumes and the timing result
it emits.  The source is Zanatticus/Hawkeye at
``a226e955d56c04be044d46f6fd876191cfce5bf4`` (Apache-2.0).  Hawkeye's
``spec.json`` files, architecture registry, workload leaves, and skill content
are deliberately not imported.

The implementation closes the four measured weaknesses of that driver:

* every timed iteration receives a fresh evaluator-secret perturbation;
* transformations change values, not merely order;
* pristine inputs and golden references live behind a mandatory mount boundary;
* compiled sources are hashed after execution and compared to their pre-run
  snapshot.

It also owns the wave-1 evaluator additions: structural precision before the
``sqrt(D_reduce) * eps(input_dtype)`` float64-reference check, a typed
library-substitution verdict, a workload-roofline speed cap with unknown-part
refusal, and a read-only bridge to the already validated Ghost Replay helper.

This file runs no process, GPU workload, inference, or profiler.  L3 remains
dropped.  The semantic judge remains in the stack but is non-gating until its
standing mutant calibration passes.  PC sampling and TCC hit rate are not input
classes or objectives here.
"""
from __future__ import annotations

import ast
import hashlib
import hmac
import importlib.util
import json
import math
import re
import stat
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.kernel_rnd import c6_reward_integrity
from scripts.kernel_rnd.autokernel.execution import sandbox as candidate_sandbox
from scripts.kernel_rnd.autokernel.execution.physical_bounds import PhysicalEnvelope


MODULE_ID = "autokernel.evaluator.hawkeye-measurement/v1"
HAWKEYE_SOURCE_COMMIT = "a226e955d56c04be044d46f6fd876191cfce5bf4"
HAWKEYE_LICENSE = "Apache-2.0"
HAWKEYE_ADOPTED_SCHEMAS = (
    "hawkeye_tensor_manifest.schema.json",
    "hawkeye_timing_result.schema.json",
)
HAWKEYE_EXCLUDED_ASSETS = (
    "spec.json",
    "architecture_registry",
    "workload_leaf_layout",
    "nvidia_skill_library",
)

GATE_STACK = c6_reward_integrity.C6_GATE_TIERS
DROPPED_TIERS = c6_reward_integrity.C6_DROPPED_TIERS
if GATE_STACK != ("L1_static", "L2_ghost_replay", "semantic_judge"):
    raise RuntimeError("the ratified C6 gate stack changed unexpectedly")
if DROPPED_TIERS != ("L3",):
    raise RuntimeError("L3 must remain dropped, not deferred")

FORBIDDEN_REWARD_METRICS = frozenset({
    "tcc_hit_rate", "tcc_hit_sum", "tcc_hit", "l2_hit_rate", "cache_hit_rate",
})
ALLOWED_MEASUREMENT_INPUT_CLASSES = frozenset({
    "wall_clock", "gpu_event_clock", "rocprof_v1_stats",
    "validated_sq_ta_tcc_counter",
})
UNAVAILABLE_MEASUREMENT_INPUT_CLASSES = frozenset({
    "pc_sampling",       # ROCm 6.2 API returns status 16: not implemented.
    "gpu_busy_percent",  # observed latched at 100% with zero KFD clients.
})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DTYPE_EPSILON = {
    "float64": 2.220446049250313e-16,
    "float32": 1.1920928955078125e-7,
    "float16": 9.765625e-4,
    "bfloat16": 7.8125e-3,
}
_DTYPE_ALIASES = {
    "double": "float64", "fp64": "float64",
    "float": "float32", "fp32": "float32",
    "half": "float16", "fp16": "float16",
    "bf16": "bfloat16",
}


class HawkeyeMeasurementError(ValueError):
    """A malformed or unauthorised evaluator input; callers must refuse it."""


class IsolationError(HawkeyeMeasurementError):
    """The candidate-visible and oracle-private planes are not disjoint."""


class SourceIntegrityError(HawkeyeMeasurementError):
    """Compiled-source identity is missing, unsafe, or changed post-run."""


class PrecisionEquivalenceError(HawkeyeMeasurementError):
    """A precision-equivalence contract cannot be evaluated honestly."""


def _normal_dtype(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HawkeyeMeasurementError("dtype must be non-empty text")
    value = value.strip().lower()
    for prefix in ("torch.", "numpy.", "np.", "tl."):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return _DTYPE_ALIASES.get(value, value)


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HawkeyeMeasurementError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise HawkeyeMeasurementError(f"{name} must be a finite positive number")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HawkeyeMeasurementError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise HawkeyeMeasurementError(f"{name} must be a finite non-negative number")
    return result


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# The two adopted Hawkeye JSON contracts
# ---------------------------------------------------------------------------

def validate_hawkeye_tensor_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the adopted ``dump_manifest.json`` contract exactly.

    The returned object is a normalized copy.  File paths must be relative,
    single-link data-plane names; a manifest can describe inputs but can never
    point at an oracle directory.
    """
    if not isinstance(payload, Mapping):
        raise HawkeyeMeasurementError("tensor manifest must be an object")
    unknown = set(payload) - {"variants", "tensors", "outputs"}
    if unknown:
        raise HawkeyeMeasurementError(f"tensor manifest has unknown fields {sorted(unknown)}")
    tensors = payload.get("tensors")
    outputs = payload.get("outputs")
    if not isinstance(tensors, list) or not tensors:
        raise HawkeyeMeasurementError("tensor manifest requires a non-empty tensors array")
    if not isinstance(outputs, list) or not outputs:
        raise HawkeyeMeasurementError("tensor manifest requires a non-empty outputs array")
    variants = payload.get("variants", 1)
    if isinstance(variants, bool) or not isinstance(variants, int) or variants < 1:
        raise HawkeyeMeasurementError("variants must be a positive integer")

    def tensor(row: object, *, output: bool) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise HawkeyeMeasurementError("tensor entries must be objects")
        allowed = {"name", "dtype", "numel"} if output else {
            "name", "file", "dtype", "shape", "numel", "role"}
        if set(row) != allowed:
            raise HawkeyeMeasurementError(
                f"tensor entry fields must be exactly {sorted(allowed)}")
        name = row["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name):
            raise HawkeyeMeasurementError("tensor name is not a safe identifier")
        dtype = _normal_dtype(row["dtype"])
        numel = row["numel"]
        if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
            raise HawkeyeMeasurementError("tensor numel must be a positive integer")
        normalized: dict[str, Any] = {"name": name, "dtype": dtype, "numel": numel}
        if not output:
            file_name = row["file"]
            path = Path(file_name) if isinstance(file_name, str) else Path("/")
            if not isinstance(file_name, str) or not file_name or path.is_absolute() \
                    or ".." in path.parts or len(path.parts) != 1:
                raise HawkeyeMeasurementError("tensor file must be one relative leaf name")
            shape = row["shape"]
            if not isinstance(shape, list) or not shape or any(
                    isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                    for dim in shape):
                raise HawkeyeMeasurementError("tensor shape must contain positive integers")
            if math.prod(shape) != numel:
                raise HawkeyeMeasurementError("tensor shape product does not equal numel")
            role = row["role"]
            if role not in {"input", "weight"}:
                raise HawkeyeMeasurementError("tensor role must be input or weight")
            normalized.update(file=file_name, shape=list(shape), role=role)
        return normalized

    normalized_tensors = [tensor(row, output=False) for row in tensors]
    normalized_outputs = [tensor(row, output=True) for row in outputs]
    names = [row["name"] for row in normalized_tensors + normalized_outputs]
    if len(names) != len(set(names)):
        raise HawkeyeMeasurementError("tensor names must be unique across inputs and outputs")
    if not any(row["role"] == "input" for row in normalized_tensors):
        raise HawkeyeMeasurementError("manifest requires at least one perturbable input")
    return {"variants": variants, "tensors": normalized_tensors,
            "outputs": normalized_outputs}


_TIMING_FIELDS = frozenset({
    "rate_samples_per_s", "rate_samples_per_s_wall", "ms_per_iter_gpu",
    "ms_per_iter_wall", "gpu_ms_min", "gpu_ms_max", "gpu_ms_std", "cv_gpu",
    "wall_ms_min", "wall_ms_max", "wall_ms_std", "cv_wall", "timing_redos",
    "cv_gate_pass", "cv_gate_max",
})


def validate_hawkeye_timing_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact aggregate result emitted by Hawkeye's frozen driver."""
    if not isinstance(payload, Mapping) or set(payload) != _TIMING_FIELDS:
        got = sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__
        raise HawkeyeMeasurementError(
            f"timing result fields differ from adopted Hawkeye schema: {got}")
    result: dict[str, Any] = {}
    nonnegative = {"gpu_ms_std", "cv_gpu", "wall_ms_std", "cv_wall"}
    for key in _TIMING_FIELDS - {"timing_redos", "cv_gate_pass"}:
        result[key] = (_finite_nonnegative if key in nonnegative else _finite_positive)(
            payload[key], key)
    redos = payload["timing_redos"]
    if isinstance(redos, bool) or not isinstance(redos, int) or redos < 0:
        raise HawkeyeMeasurementError("timing_redos must be a non-negative integer")
    if type(payload["cv_gate_pass"]) is not bool:
        raise HawkeyeMeasurementError("cv_gate_pass must be an exact bool")
    if result["gpu_ms_min"] > result["ms_per_iter_gpu"] \
            or result["ms_per_iter_gpu"] > result["gpu_ms_max"]:
        raise HawkeyeMeasurementError("GPU min/mean/max ordering is inconsistent")
    if result["wall_ms_min"] > result["ms_per_iter_wall"] \
            or result["ms_per_iter_wall"] > result["wall_ms_max"]:
        raise HawkeyeMeasurementError("wall min/mean/max ordering is inconsistent")
    expected = result["cv_gpu"] <= result["cv_gate_max"]
    if payload["cv_gate_pass"] != expected:
        raise HawkeyeMeasurementError("cv_gate_pass disagrees with cv_gpu and cv_gate_max")
    result.update(timing_redos=redos, cv_gate_pass=payload["cv_gate_pass"])
    return result


# ---------------------------------------------------------------------------
# Evaluator-secret, per-iteration re-perturbation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class EvaluatorRunSeed:
    """At least 256 bits held by the evaluator, never serialized to a candidate."""
    secret: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.secret, bytes) or len(self.secret) < 32:
            raise HawkeyeMeasurementError("evaluator run seed requires at least 32 bytes")

    def __repr__(self) -> str:
        return "EvaluatorRunSeed(<redacted>)"

    @property
    def commitment(self) -> str:
        return hashlib.sha256(b"autokernel-run-seed/v1\0" + self.secret).hexdigest()

    def iteration_key(self, iteration: int, tensor_name: str) -> bytes:
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise HawkeyeMeasurementError("iteration must be a non-negative integer")
        if not isinstance(tensor_name, str) or not tensor_name:
            raise HawkeyeMeasurementError("tensor_name must be non-empty")
        message = f"iteration={iteration}\0tensor={tensor_name}".encode("utf-8")
        return hmac.new(self.secret, message, hashlib.sha256).digest()


def _prf_u64(key: bytes, index: int, lane: int = 0) -> int:
    message = index.to_bytes(8, "big") + lane.to_bytes(2, "big")
    return int.from_bytes(hmac.new(key, message, hashlib.sha256).digest()[:8], "big")


def perturb_values(values: Sequence[int | float], *, dtype: str,
                   iteration: int, tensor_name: str,
                   run_seed: EvaluatorRunSeed) -> tuple[int | float, ...]:
    """Return a value-changing perturbation unique to one timed iteration.

    No permutation is used.  Floating values receive an element-specific affine
    perturbation, defeating order-insensitive reductions and global affine
    invariances.  Integer values receive a non-zero, range-preserving modular
    delta.  Every iteration starts from evaluator-private pristine values.
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise HawkeyeMeasurementError("perturbation requires a non-empty numeric sequence")
    dtype = _normal_dtype(dtype)
    key = run_seed.iteration_key(iteration, tensor_name)
    out: list[int | float] = []
    if dtype in _DTYPE_EPSILON:
        for index, raw in enumerate(values):
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
                    or not math.isfinite(float(raw)):
                raise HawkeyeMeasurementError("floating input contains a non-finite value")
            u0 = _prf_u64(key, index, 0) / float(2**64 - 1)
            u1 = _prf_u64(key, index, 1) / float(2**64 - 1)
            scale = 0.875 + 0.25 * u0
            offset = (2.0 * u1 - 1.0) * 0.125 * (1.0 + abs(float(raw)))
            changed = float(raw) * scale + offset
            if changed == float(raw):
                changed = math.nextafter(changed, math.inf)
            out.append(changed)
    else:
        match = re.fullmatch(r"(u?int)(8|16|32|64)", dtype)
        if not match:
            raise HawkeyeMeasurementError(f"unsupported perturbation dtype {dtype!r}")
        bits = int(match.group(2))
        lo = 0 if match.group(1) == "uint" else -(2 ** (bits - 1))
        hi = 2**bits - 1 if lo == 0 else 2 ** (bits - 1) - 1
        span = hi - lo + 1
        for index, raw in enumerate(values):
            if isinstance(raw, bool) or not isinstance(raw, int) or not lo <= raw <= hi:
                raise HawkeyeMeasurementError(f"integer input is outside {dtype} range")
            delta = 1 + (_prf_u64(key, index) % (span - 1))
            out.append(lo + ((raw - lo + delta) % span))
    return tuple(out)


@dataclass(frozen=True)
class PerturbationReceipt:
    schema: str
    run_seed_commitment: str
    iteration: int
    tensor_name: str
    dtype: str
    pristine_sha256: str
    perturbed_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "epyc.autokernel.iteration_perturbation.v1":
            raise HawkeyeMeasurementError("unknown perturbation receipt schema")
        for name in ("run_seed_commitment", "pristine_sha256", "perturbed_sha256"):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise HawkeyeMeasurementError(f"{name} must be sha256")
        if self.pristine_sha256 == self.perturbed_sha256:
            raise HawkeyeMeasurementError("perturbation did not change the input")


def perturbation_receipt(pristine: Sequence[int | float], perturbed: Sequence[int | float],
                         *, dtype: str, iteration: int, tensor_name: str,
                         run_seed: EvaluatorRunSeed) -> PerturbationReceipt:
    dtype = _normal_dtype(dtype)
    return PerturbationReceipt(
        schema="epyc.autokernel.iteration_perturbation.v1",
        run_seed_commitment=run_seed.commitment,
        iteration=iteration,
        tensor_name=tensor_name,
        dtype=dtype,
        pristine_sha256=_canonical_sha256({"dtype": dtype, "values": list(pristine)}),
        perturbed_sha256=_canonical_sha256({"dtype": dtype, "values": list(perturbed)}),
    )


# ---------------------------------------------------------------------------
# Candidate/oracle isolation
# ---------------------------------------------------------------------------

def _absolute(path: Path | str, name: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise IsolationError(f"{name} must be absolute")
    return path.resolve(strict=False)


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


@dataclass(frozen=True)
class CandidateIsolationPlan:
    """Landlock/seccomp/cgroup contract hiding pristine/golden data.

    Mount namespaces are unavailable to the unprivileged controller on this
    host.  The already reviewed evaluator sandbox instead applies default-deny
    Landlock reads, exact read/execute allowlists, seccomp network denial, and a
    fresh cgroup to each process.  This plan is checked against the sandbox's
    trusted activation receipt; it is not an assertion by candidate code.
    """
    candidate_root: Path
    transformed_input_root: Path
    private_pristine_root: Path
    private_golden_root: Path
    additional_readable_roots: tuple[Path, ...] = ()
    readable_files: tuple[Path, ...] = ()
    executable_files: tuple[Path, ...] = ()
    oracle_additional_readable_roots: tuple[Path, ...] = ()
    oracle_readable_files: tuple[Path, ...] = ()
    oracle_executable_files: tuple[Path, ...] = ()
    sandbox_id: str = candidate_sandbox.SANDBOX_ID
    sandbox_profile: str = candidate_sandbox.EVALUATOR_PROFILE
    oracle_sandbox_profile: str = candidate_sandbox.ORACLE_PROFILE
    read_allowlist_enforced: bool = True
    network_disabled: bool = True
    distinct_sandbox_activations: bool = True

    def __post_init__(self) -> None:
        for name in ("candidate_root", "transformed_input_root",
                     "private_pristine_root", "private_golden_root"):
            object.__setattr__(self, name, _absolute(getattr(self, name), name))
        for name in (
                "additional_readable_roots", "readable_files", "executable_files",
                "oracle_additional_readable_roots", "oracle_readable_files",
                "oracle_executable_files"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise IsolationError(f"{name} must be a tuple")
            object.__setattr__(self, name, tuple(
                _absolute(value, f"{name} entry") for value in values))
        if self.sandbox_id != candidate_sandbox.SANDBOX_ID \
                or self.sandbox_profile != candidate_sandbox.EVALUATOR_PROFILE:
            raise IsolationError("candidate isolation requires the reviewed evaluator sandbox")
        if self.oracle_sandbox_profile != candidate_sandbox.ORACLE_PROFILE:
            raise IsolationError("oracle isolation requires the reviewed private oracle profile")
        if type(self.read_allowlist_enforced) is not bool \
                or not self.read_allowlist_enforced:
            raise IsolationError("candidate isolation requires default-deny read allowlisting")
        if type(self.network_disabled) is not bool or not self.network_disabled:
            raise IsolationError("candidate isolation requires networking disabled")
        if type(self.distinct_sandbox_activations) is not bool \
                or not self.distinct_sandbox_activations:
            raise IsolationError(
                "golden references and candidates require distinct sandbox activations")
        visible = (self.candidate_root, self.transformed_input_root,
                   *self.additional_readable_roots, *self.readable_files,
                   *self.executable_files)
        private = (self.private_pristine_root, self.private_golden_root)
        for public in visible:
            for secret in private:
                if _overlaps(public, secret):
                    raise IsolationError("candidate-visible and oracle-private roots overlap")

    def candidate_policy_projection(self) -> dict[str, Any]:
        """Exact receipt fields expected from ``execution.sandbox``."""
        return {
            "sandbox_id": self.sandbox_id,
            "profile": self.sandbox_profile,
            "writable_root": str(self.candidate_root),
            "writable_device_paths": [
                "/dev/kfd", "/dev/dri/renderD128", "/dev/null"],
            "read_allowlist_enforced": True,
            "network_profile": candidate_sandbox.NETWORK_DENY_ALL,
            "readable_roots": [str(self.transformed_input_root), *(
                str(path) for path in self.additional_readable_roots)],
            "readable_files": [str(path) for path in self.readable_files],
            "executable_files": [str(path) for path in self.executable_files],
        }

    def oracle_policy_projection(self) -> dict[str, Any]:
        """Exact private-read/no-device oracle activation expected at runtime."""
        return {
            "sandbox_id": self.sandbox_id,
            "profile": self.oracle_sandbox_profile,
            "writable_root": str(self.private_golden_root),
            "writable_device_paths": [],
            "read_allowlist_enforced": True,
            "network_profile": candidate_sandbox.NETWORK_DENY_ALL,
            "readable_roots": [str(self.private_pristine_root), *(
                str(path) for path in self.oracle_additional_readable_roots)],
            "readable_files": [str(path) for path in self.oracle_readable_files],
            "executable_files": [str(path) for path in self.oracle_executable_files],
        }


@dataclass(frozen=True)
class ProcessIsolationReceipt:
    """Runtime witness that the plan's two-process boundary actually happened."""
    oracle_pid: int
    candidate_pid: int
    oracle_process_start_ticks: int
    candidate_process_start_ticks: int
    oracle_sandbox_receipt_sha256: str
    candidate_sandbox_receipt_sha256: str
    oracle_teardown_receipt_sha256: str
    candidate_teardown_receipt_sha256: str
    oracle_policy_sha256: str
    candidate_policy_sha256: str
    oracle_cgroup_path: str
    candidate_cgroup_path: str
    candidate_read_allowlist_sha256: str

    def __post_init__(self) -> None:
        for name in ("oracle_pid", "candidate_pid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IsolationError(f"{name} must be a positive PID")
        if self.oracle_pid == self.candidate_pid:
            raise IsolationError("oracle and candidate must execute in distinct processes")
        for name in ("oracle_process_start_ticks", "candidate_process_start_ticks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IsolationError(f"{name} must be positive")
        for name in ("oracle_sandbox_receipt_sha256", "candidate_sandbox_receipt_sha256",
                     "oracle_teardown_receipt_sha256", "candidate_teardown_receipt_sha256",
                     "oracle_policy_sha256", "candidate_policy_sha256",
                     "candidate_read_allowlist_sha256"):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise IsolationError(f"{name} must be sha256")
        for name in ("oracle_cgroup_path", "candidate_cgroup_path"):
            if not isinstance(getattr(self, name), str) \
                    or not Path(getattr(self, name)).is_absolute():
                raise IsolationError(f"{name} must be an absolute path")
        if self.oracle_cgroup_path == self.candidate_cgroup_path:
            raise IsolationError("oracle and candidate share a cgroup activation")


def bind_process_isolation(plan: CandidateIsolationPlan, *,
                           oracle_sandbox_receipt: Mapping[str, Any],
                           candidate_sandbox_receipt: Mapping[str, Any],
                           oracle_teardown_receipt: Mapping[str, Any],
                           candidate_teardown_receipt: Mapping[str, Any]) \
        -> ProcessIsolationReceipt:
    """Bind two upstream-verified sandbox receipts to the secrecy invariant."""
    if not isinstance(oracle_sandbox_receipt, Mapping) \
            or not isinstance(candidate_sandbox_receipt, Mapping):
        raise IsolationError("sandbox activation receipts must be mappings")
    for label, receipt in (("oracle", oracle_sandbox_receipt),
                           ("candidate", candidate_sandbox_receipt)):
        if receipt.get("schema") != candidate_sandbox.RECEIPT_SCHEMA \
                or receipt.get("sandbox_id") != candidate_sandbox.SANDBOX_ID:
            raise IsolationError(f"{label} sandbox activation is not trusted")
        if receipt.get("read_allowlist_enforced") is not True:
            raise IsolationError(f"{label} sandbox did not enforce its read allowlist")
    expected = plan.candidate_policy_projection()
    expected_oracle = plan.oracle_policy_projection()
    for field, value in expected.items():
        if candidate_sandbox_receipt.get(field) != value:
            raise IsolationError(
                f"candidate sandbox receipt {field} differs from the isolation plan")
    for field, value in expected_oracle.items():
        if oracle_sandbox_receipt.get(field) != value:
            raise IsolationError(
                f"oracle sandbox receipt {field} differs from the isolation plan")
    for label, activation, teardown in (
            ("oracle", oracle_sandbox_receipt, oracle_teardown_receipt),
            ("candidate", candidate_sandbox_receipt, candidate_teardown_receipt)):
        if not isinstance(teardown, Mapping) \
                or teardown.get("cgroup_path") != activation.get("cgroup_path") \
                or teardown.get("verified_empty") is not True \
                or teardown.get("removed") is not True:
            raise IsolationError(f"{label} sandbox teardown was not verified complete")
    oracle_pid = oracle_sandbox_receipt.get("pid")
    candidate_pid = candidate_sandbox_receipt.get("pid")
    return ProcessIsolationReceipt(
        oracle_pid=oracle_pid, candidate_pid=candidate_pid,
        oracle_process_start_ticks=oracle_sandbox_receipt.get("process_start_ticks"),
        candidate_process_start_ticks=candidate_sandbox_receipt.get("process_start_ticks"),
        oracle_sandbox_receipt_sha256=_canonical_sha256(dict(oracle_sandbox_receipt)),
        candidate_sandbox_receipt_sha256=_canonical_sha256(dict(candidate_sandbox_receipt)),
        oracle_teardown_receipt_sha256=_canonical_sha256(dict(oracle_teardown_receipt)),
        candidate_teardown_receipt_sha256=_canonical_sha256(
            dict(candidate_teardown_receipt)),
        oracle_policy_sha256=oracle_sandbox_receipt.get("policy_sha256", ""),
        candidate_policy_sha256=candidate_sandbox_receipt.get("policy_sha256", ""),
        oracle_cgroup_path=oracle_sandbox_receipt.get("cgroup_path", ""),
        candidate_cgroup_path=candidate_sandbox_receipt.get("cgroup_path", ""),
        candidate_read_allowlist_sha256=_canonical_sha256(expected))


# ---------------------------------------------------------------------------
# Post-run compiled-source SHA-256
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceFileDigest:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CompiledSourceSnapshot:
    schema: str
    root_label: str
    files: tuple[SourceFileDigest, ...]
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "epyc.autokernel.compiled_source_snapshot.v1":
            raise SourceIntegrityError("unknown compiled-source snapshot schema")
        if not self.files or not _SHA256_RE.fullmatch(self.snapshot_sha256):
            raise SourceIntegrityError("compiled-source snapshot is incomplete")
        expected = _canonical_sha256([
            {"relative_path": row.relative_path, "sha256": row.sha256,
             "size_bytes": row.size_bytes} for row in self.files])
        if expected != self.snapshot_sha256:
            raise SourceIntegrityError("compiled-source snapshot digest is not canonical")


def snapshot_compiled_sources(root: Path | str,
                              relative_paths: Iterable[str]) -> CompiledSourceSnapshot:
    root = Path(root).resolve(strict=True)
    rows: list[SourceFileDigest] = []
    requested = sorted(set(relative_paths))
    if not requested:
        raise SourceIntegrityError("compiled-source list is empty")
    for relative in requested:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise SourceIntegrityError("compiled-source paths must stay under the source root")
        path = root / rel
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.is_symlink():
            raise SourceIntegrityError("compiled source must be a single-link regular file")
        if path.resolve(strict=True).parent != root and root not in path.resolve(strict=True).parents:
            raise SourceIntegrityError("compiled source escaped its declared root")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(SourceFileDigest(rel.as_posix(), digest, info.st_size))
    payload = [{"relative_path": row.relative_path, "sha256": row.sha256,
                "size_bytes": row.size_bytes} for row in rows]
    return CompiledSourceSnapshot(
        schema="epyc.autokernel.compiled_source_snapshot.v1",
        root_label="candidate_compiled_sources",
        files=tuple(rows), snapshot_sha256=_canonical_sha256(payload))


def verify_post_run_source_integrity(before: CompiledSourceSnapshot,
                                     after: CompiledSourceSnapshot) -> None:
    """Require and record a post-run hash, not owner-revocable chmod state."""
    if before.files != after.files or before.snapshot_sha256 != after.snapshot_sha256:
        raise SourceIntegrityError("compiled sources changed between compile and post-run hash")


# ---------------------------------------------------------------------------
# Precision-equivalence reducer (RVP-C2-9 + RVP-C6-22)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrecisionEquivalencePolicy:
    operator_id: str
    template_id: str
    input_dtype: str
    required_output_dtype: str
    required_accumulator_dtype: str
    reduce_dimension: int
    structural_evidence_sha256: str
    bound_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for name in ("operator_id", "template_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise PrecisionEquivalenceError(f"{name} must be non-empty")
        for name in ("input_dtype", "required_output_dtype", "required_accumulator_dtype"):
            object.__setattr__(self, name, _normal_dtype(getattr(self, name)))
        if self.input_dtype not in _DTYPE_EPSILON:
            raise PrecisionEquivalenceError("input dtype has no pinned machine epsilon")
        if isinstance(self.reduce_dimension, bool) or not isinstance(
                self.reduce_dimension, int) or self.reduce_dimension <= 0:
            raise PrecisionEquivalenceError("reduce_dimension must be a positive integer")
        if not _SHA256_RE.fullmatch(self.structural_evidence_sha256):
            raise PrecisionEquivalenceError("structural precision evidence requires sha256")
        object.__setattr__(self, "bound_multiplier",
                           _finite_positive(self.bound_multiplier, "bound_multiplier"))
        if self.bound_multiplier != 1.0:
            raise PrecisionEquivalenceError(
                "bound_multiplier is operator-pinned to exactly 1.0")

    @property
    def normalized_error_bound(self) -> float:
        return (self.bound_multiplier * math.sqrt(self.reduce_dimension)
                * _DTYPE_EPSILON[self.input_dtype])

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256({
            "operator_id": self.operator_id,
            "template_id": self.template_id,
            "input_dtype": self.input_dtype,
            "required_output_dtype": self.required_output_dtype,
            "required_accumulator_dtype": self.required_accumulator_dtype,
            "reduce_dimension": self.reduce_dimension,
            "structural_evidence_sha256": self.structural_evidence_sha256,
            "bound_multiplier": self.bound_multiplier,
            "normalized_error_bound": self.normalized_error_bound,
        })


@dataclass(frozen=True)
class PrecisionEquivalenceVerdict:
    correct: bool
    stage: str
    reason: str
    normalized_rms_error: float | None
    normalized_error_bound: float
    max_absolute_error: float | None
    reference_dtype: str
    required_output_dtype: str
    observed_output_dtype: str
    required_accumulator_dtype: str
    observed_accumulator_dtype: str
    structural_evidence_sha256: str
    precision_policy_sha256: str


def evaluate_precision_equivalence(
        reference_float64: Sequence[float], candidate: Sequence[float], *,
        policy: PrecisionEquivalencePolicy, observed_output_dtype: str,
        observed_accumulator_dtype: str,
        reference_dtype: str = "float64") -> PrecisionEquivalenceVerdict:
    """Check structure first, then sqrt(D)-scaled error vs a float64 reference."""
    observed_output_dtype = _normal_dtype(observed_output_dtype)
    observed_accumulator_dtype = _normal_dtype(observed_accumulator_dtype)
    reference_dtype = _normal_dtype(reference_dtype)
    common = dict(
        normalized_error_bound=policy.normalized_error_bound,
        reference_dtype=reference_dtype,
        required_output_dtype=policy.required_output_dtype,
        observed_output_dtype=observed_output_dtype,
        required_accumulator_dtype=policy.required_accumulator_dtype,
        observed_accumulator_dtype=observed_accumulator_dtype,
        structural_evidence_sha256=policy.structural_evidence_sha256,
        precision_policy_sha256=policy.policy_sha256,
    )
    if observed_output_dtype != policy.required_output_dtype \
            or observed_accumulator_dtype != policy.required_accumulator_dtype:
        return PrecisionEquivalenceVerdict(
            False, "structural_precision", "dtype_or_accumulator_mismatch",
            None, max_absolute_error=None, **common)
    if reference_dtype != "float64":
        raise PrecisionEquivalenceError("precision equivalence requires a float64 reference")
    if isinstance(reference_float64, (str, bytes)) or isinstance(candidate, (str, bytes)) \
            or not isinstance(reference_float64, Sequence) or not isinstance(candidate, Sequence) \
            or not reference_float64 or len(reference_float64) != len(candidate):
        raise PrecisionEquivalenceError("reference and candidate vectors must be equal non-zero length")
    ref = [float(value) for value in reference_float64]
    got = [float(value) for value in candidate]
    if not all(math.isfinite(value) for value in ref + got):
        return PrecisionEquivalenceVerdict(
            False, "numeric", "nonfinite_value", None,
            max_absolute_error=math.inf, **common)
    errors = [abs(left - right) for left, right in zip(ref, got)]
    rms_error = math.sqrt(sum(value * value for value in errors) / len(errors))
    rms_reference = math.sqrt(sum(value * value for value in ref) / len(ref))
    normalized = rms_error / max(1.0, rms_reference)
    correct = normalized <= math.nextafter(policy.normalized_error_bound, math.inf)
    return PrecisionEquivalenceVerdict(
        correct, "numeric", "within_sqrt_d_epsilon_bound" if correct else
        "exceeds_sqrt_d_epsilon_bound", normalized,
        max_absolute_error=max(errors), **common)


# ---------------------------------------------------------------------------
# Library-substitution exploit carrier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispatchEvent:
    symbol: str
    provider: str
    provenance: str

    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, name), str) and getattr(self, name).strip()
                   for name in ("symbol", "provider", "provenance")):
            raise HawkeyeMeasurementError("dispatch event fields must be non-empty text")


@dataclass(frozen=True)
class LibrarySubstitutionVerdict:
    outcome: str
    reason: str
    source_sha256: str
    action_tags: tuple[str, ...]
    forbidden_source_calls: tuple[str, ...]
    forbidden_dispatches: tuple[str, ...]
    generated_dispatches: tuple[str, ...]


_FORBIDDEN_PREFIXES = (
    "torch", "rocblas", "hipblas", "hipblaslt", "miopen", "cublas",
    "flash_attn", "composable_kernel", "ck::", "at::",
)
_FORBIDDEN_ACTION_TAGS = frozenset({"CALL_LIBRARY", "SOTA_TRITON_API"})
_VENDOR_PROVIDERS = frozenset({
    "rocblas", "hipblas", "hipblaslt", "miopen", "torch", "pytorch",
    "cublas", "flash-attn", "composable-kernel",
})


def _python_calls(source: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".", 1)[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        cursor = node.func
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            parts.append(cursor.id)
        if parts:
            rendered = ".".join(reversed(parts))
            head, separator, tail = rendered.partition(".")
            if head in aliases:
                rendered = aliases[head] + (separator + tail if separator else "")
            calls.add(rendered)
    return tuple(sorted(calls))


def _native_calls(source: str) -> tuple[str, ...]:
    # Remove comments and quoted strings before looking for call syntax.  This
    # is a lexical L1 screen; the dispatch trace is the independent L2 proof.
    cleaned = re.sub(r"//.*?$|/\*.*?\*/|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"",
                     " ", source, flags=re.MULTILINE | re.DOTALL)
    return tuple(sorted(set(re.findall(
        r"\b(?:[A-Za-z_]\w*(?:::\w+|\.\w+)*)\s*(?=\()", cleaned))))


def evaluate_library_substitution(source: str, *, language: str,
                                  action_tags: Sequence[str],
                                  dispatch_events: Sequence[DispatchEvent]) \
        -> LibrarySubstitutionVerdict:
    if not isinstance(source, str) or not source.strip():
        raise HawkeyeMeasurementError("library-substitution scan requires source")
    language = language.strip().lower()
    calls = _python_calls(source) if language in {"python", "triton"} \
        else _native_calls(source)
    forbidden_calls = tuple(call for call in calls if call.lower().startswith(
        tuple(prefix.lower() for prefix in _FORBIDDEN_PREFIXES)))
    tags = tuple(sorted(set(action_tags)))
    bad_tags = tuple(tag for tag in tags if tag in _FORBIDDEN_ACTION_TAGS)
    bad_dispatches = tuple(sorted(
        event.symbol for event in dispatch_events
        if event.provider.strip().lower() in _VENDOR_PROVIDERS))
    generated = tuple(sorted(
        event.symbol for event in dispatch_events
        if event.provenance == "candidate_generated"))
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if bad_tags or forbidden_calls or bad_dispatches:
        return LibrarySubstitutionVerdict(
            "FAIL", "library_substitution_detected", source_sha, tags,
            forbidden_calls, bad_dispatches, generated)
    if not dispatch_events or not generated:
        return LibrarySubstitutionVerdict(
            "COULD_NOT_CHECK", "no_generated_dispatch_trace", source_sha, tags,
            forbidden_calls, bad_dispatches, generated)
    return LibrarySubstitutionVerdict(
        "PASS", "generated_kernel_observed_without_vendor_substitution",
        source_sha, tags, forbidden_calls, bad_dispatches, generated)


# ---------------------------------------------------------------------------
# Workload-roofline implausible-speedup cap
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RooflineSpeedupCap:
    gpu_part: str
    hardware_authority: Mapping[str, str]
    shape_id: str
    measurement_frame_sha256: str
    baseline_throughput_units_s: float
    roofline_throughput_ceiling_units_s: float
    max_speedup: float
    physical_envelope_sha256: str


def derive_roofline_speedup_cap(*, gpu_part: str, envelope: PhysicalEnvelope,
                                baseline_throughput_units_s: float) -> RooflineSpeedupCap:
    """Derive the cap from exact work+hardware authority; unknown parts raise."""
    authority = c6_reward_integrity.require_supported_gpu(gpu_part)
    baseline = _finite_positive(
        baseline_throughput_units_s, "baseline_throughput_units_s")
    ceiling = envelope.throughput_ceiling_units_s
    if baseline > math.nextafter(ceiling, math.inf):
        raise HawkeyeMeasurementError(
            "baseline already exceeds the workload roofline; frame or units are wrong")
    maximum = ceiling / baseline
    return RooflineSpeedupCap(
        gpu_part=gpu_part, hardware_authority=authority,
        shape_id=envelope.shape_id,
        measurement_frame_sha256=envelope.measurement_frame_sha256,
        baseline_throughput_units_s=baseline,
        roofline_throughput_ceiling_units_s=ceiling,
        max_speedup=maximum,
        physical_envelope_sha256=_canonical_sha256(envelope.to_dict()))


def check_implausible_speedup(speedup: float, cap: RooflineSpeedupCap) -> str:
    speedup = _finite_positive(speedup, "speedup")
    return "NULL_IMPLAUSIBLE" if speedup > math.nextafter(cap.max_speedup, math.inf) \
        else "ADMISSIBLE"


def validate_reward_metric(metric: str) -> str:
    if not isinstance(metric, str) or not metric.strip():
        raise HawkeyeMeasurementError("reward metric must be non-empty")
    normalized = metric.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in FORBIDDEN_REWARD_METRICS:
        raise HawkeyeMeasurementError(
            "TCC/L2 hit rate is forbidden from reward and acceptance on gfx90a")
    return normalized


def validate_measurement_input_class(input_class: str) -> str:
    if input_class in UNAVAILABLE_MEASUREMENT_INPUT_CLASSES \
            or input_class not in ALLOWED_MEASUREMENT_INPUT_CLASSES:
        raise HawkeyeMeasurementError(
            f"measurement input class {input_class!r} is unavailable on this ROCm")
    return input_class


# ---------------------------------------------------------------------------
# Full runtime-library closure join
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeLibraryIdentity:
    logical_name: str
    resolved_path: str
    sha256: str
    provider: str

    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, name), str) and getattr(self, name).strip()
                   for name in ("logical_name", "resolved_path", "provider")):
            raise HawkeyeMeasurementError("runtime-library identity fields are required")
        if not Path(self.resolved_path).is_absolute():
            raise HawkeyeMeasurementError("runtime-library resolved_path must be absolute")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise HawkeyeMeasurementError("runtime-library identity requires sha256")


@dataclass(frozen=True)
class RuntimeClosureVerdict:
    outcome: str
    reason: str
    expected_closure_sha256: str
    observed_closure_sha256: str
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    identity_mismatches: tuple[str, ...]


def _runtime_closure_hash(rows: Sequence[RuntimeLibraryIdentity]) -> str:
    return _canonical_sha256([
        {"logical_name": row.logical_name, "resolved_path": row.resolved_path,
         "sha256": row.sha256, "provider": row.provider}
        for row in sorted(rows, key=lambda item: item.logical_name)
    ])


def evaluate_runtime_library_closure(
        expected: Sequence[RuntimeLibraryIdentity],
        observed: Sequence[RuntimeLibraryIdentity]) -> RuntimeClosureVerdict:
    """Exact full-library join; checking only libggml-hip is insufficient."""
    if not expected or not observed:
        raise HawkeyeMeasurementError("runtime closure requires expected and observed libraries")
    expected_map = {row.logical_name: row for row in expected}
    observed_map = {row.logical_name: row for row in observed}
    if len(expected_map) != len(expected) or len(observed_map) != len(observed):
        raise HawkeyeMeasurementError("runtime closure logical names must be unique")
    missing = tuple(sorted(set(expected_map) - set(observed_map)))
    unexpected = tuple(sorted(set(observed_map) - set(expected_map)))
    mismatches = tuple(sorted(
        name for name in set(expected_map) & set(observed_map)
        if expected_map[name] != observed_map[name]))
    outcome = "PASS" if not (missing or unexpected or mismatches) else "FAIL"
    return RuntimeClosureVerdict(
        outcome, "exact_runtime_closure" if outcome == "PASS" else
        "runtime_library_join_mismatch",
        _runtime_closure_hash(expected), _runtime_closure_hash(observed),
        missing, unexpected, mismatches)


# ---------------------------------------------------------------------------
# Exact Ghost Replay lift + native LD_PRELOAD analogue
# ---------------------------------------------------------------------------

GHOST_REPLAY_SOURCE_SHA256 = (
    "131208c57c46a0587b040c5dbd220b38ef2279433be9a5153df9ad33d22f8225")


@dataclass(frozen=True)
class GhostReplayResult:
    outcome: str
    applicability: str
    detail: str
    helper_source_sha256: str | None


NATIVE_GHOST_INTERCEPT_SYMBOLS = (
    "hipLaunchKernel",
    "hipLaunchCooperativeKernel",
    "hipGraphLaunch",
)
NATIVE_GHOST_INTERPOSER_SOURCE_SHA256 = (
    "6d9639a2610a195072fc009511cf98cd7faf809e4453767a0967a1a6528ea5ee")
NATIVE_GHOST_EVENT_MAGIC = 0x414B4752
NATIVE_GHOST_EVENT_STRUCT = struct.Struct("<IIQ")
_NATIVE_GHOST_SYMBOL_IDS = {
    1: "hipLaunchKernel",
    2: "hipLaunchCooperativeKernel",
    3: "hipGraphLaunch",
}


@dataclass(frozen=True)
class NativeGhostReplayPlan:
    """Trusted no-op interposer identity and exact native launch surface."""
    interposer_path: str
    interposer_sha256: str
    interposer_source_sha256: str
    intercepted_symbols: tuple[str, ...]
    candidate_build_sha256: str
    candidate_source_snapshot_sha256: str
    perturbation_carrier_sha256: str
    initialized_output_sha256: str
    real_runtime_closure_sha256: str
    noop_runtime_closure_sha256: str

    def __post_init__(self) -> None:
        if not Path(self.interposer_path).is_absolute():
            raise HawkeyeMeasurementError("native Ghost Replay interposer path must be absolute")
        for name in (
                "interposer_sha256", "interposer_source_sha256",
                "candidate_build_sha256", "candidate_source_snapshot_sha256",
                "perturbation_carrier_sha256", "initialized_output_sha256",
                "real_runtime_closure_sha256", "noop_runtime_closure_sha256"):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise HawkeyeMeasurementError(f"native Ghost Replay {name} must be sha256")
        if self.intercepted_symbols != NATIVE_GHOST_INTERCEPT_SYMBOLS:
            raise HawkeyeMeasurementError(
                "native Ghost Replay must intercept the complete ratified HIP launch surface")
        if self.interposer_source_sha256 != NATIVE_GHOST_INTERPOSER_SOURCE_SHA256:
            raise HawkeyeMeasurementError(
                "native Ghost Replay interposer source differs from the trusted source")


@dataclass(frozen=True)
class NativeLaunchEvent:
    symbol: str
    ordinal: int

    def __post_init__(self) -> None:
        if self.symbol not in NATIVE_GHOST_INTERCEPT_SYMBOLS:
            raise HawkeyeMeasurementError("native Ghost Replay recorded an unknown launch symbol")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise HawkeyeMeasurementError("native launch ordinal must be non-negative")


def parse_native_launch_events(raw: bytes) -> tuple[NativeLaunchEvent, ...]:
    """Parse the evaluator interposer's fixed-size, counted event stream."""
    if not isinstance(raw, bytes) or not raw \
            or len(raw) % NATIVE_GHOST_EVENT_STRUCT.size:
        raise HawkeyeMeasurementError("native Ghost Replay event stream is empty or partial")
    rows: list[NativeLaunchEvent] = []
    for offset in range(0, len(raw), NATIVE_GHOST_EVENT_STRUCT.size):
        magic, symbol_id, ordinal = NATIVE_GHOST_EVENT_STRUCT.unpack_from(raw, offset)
        if magic != NATIVE_GHOST_EVENT_MAGIC or symbol_id not in _NATIVE_GHOST_SYMBOL_IDS:
            raise HawkeyeMeasurementError("native Ghost Replay event record is invalid")
        rows.append(NativeLaunchEvent(_NATIVE_GHOST_SYMBOL_IDS[symbol_id], ordinal))
    if [row.ordinal for row in rows] != list(range(len(rows))):
        raise HawkeyeMeasurementError("native Ghost Replay event stream is incomplete")
    return tuple(rows)


@dataclass(frozen=True)
class NativeGhostReplayVerdict:
    outcome: str
    reason: str
    real_output_sha256: str
    noop_output_sha256: str
    launch_count: int
    intercepted_symbols: tuple[str, ...]
    interposer_sha256: str
    candidate_build_sha256: str
    candidate_source_snapshot_sha256: str
    perturbation_carrier_sha256: str
    initialized_output_sha256: str
    real_process_isolation_carrier_sha256: str
    noop_process_isolation_carrier_sha256: str
    real_runtime_closure_sha256: str
    noop_runtime_closure_sha256: str


@dataclass(frozen=True)
class NativeReplayWitness:
    """One fresh-process leg; never a candidate self-report."""
    mode: str
    output_sha256: str
    initialized_output_sha256: str
    candidate_build_sha256: str
    candidate_source_snapshot_sha256: str
    isolation: ProcessIsolationReceipt
    runtime_closure: RuntimeClosureVerdict
    loaded_interposer_sha256: str | None

    def __post_init__(self) -> None:
        if self.mode not in {"real", "noop"}:
            raise HawkeyeMeasurementError("native replay witness mode must be real or noop")
        for name in ("output_sha256", "initialized_output_sha256",
                     "candidate_build_sha256", "candidate_source_snapshot_sha256"):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise HawkeyeMeasurementError(f"native replay witness {name} must be sha256")
        if not isinstance(self.isolation, ProcessIsolationReceipt):
            raise HawkeyeMeasurementError("native replay witness requires process isolation")
        if not isinstance(self.runtime_closure, RuntimeClosureVerdict):
            raise HawkeyeMeasurementError("native replay witness requires runtime closure")
        if self.loaded_interposer_sha256 is not None \
                and not _SHA256_RE.fullmatch(self.loaded_interposer_sha256):
            raise HawkeyeMeasurementError("loaded interposer identity must be sha256")


def _carrier_hash(value: object) -> str:
    return serialize_carrier(value)["carrier_sha256"]


def evaluate_native_ghost_replay(*, plan: NativeGhostReplayPlan,
                                 perturbation: PerturbationReceipt,
                                 real: NativeReplayWitness,
                                 noop: NativeReplayWitness,
                                 interposer_event_bytes: bytes) \
        -> NativeGhostReplayVerdict:
    """Apply Ghost Replay semantics to the pinned HIP launch interposer.

    The evaluator must initialize output storage identically in both fresh
    candidate processes.  With launch calls no-op'd, identical output proves
    the generated kernel was not load-bearing and therefore fails.
    """
    if not isinstance(perturbation, PerturbationReceipt) \
            or _carrier_hash(perturbation) != plan.perturbation_carrier_sha256:
        raise HawkeyeMeasurementError("native replay input perturbation differs from plan")
    if real.mode != "real" or noop.mode != "noop":
        raise HawkeyeMeasurementError("native replay legs are swapped")
    for witness in (real, noop):
        if witness.initialized_output_sha256 != plan.initialized_output_sha256:
            raise HawkeyeMeasurementError("native replay output initialization differs")
        if witness.candidate_build_sha256 != plan.candidate_build_sha256 \
                or witness.candidate_source_snapshot_sha256 \
                != plan.candidate_source_snapshot_sha256:
            raise HawkeyeMeasurementError("native replay candidate identity differs")
        if witness.runtime_closure.outcome != "PASS" \
                or witness.runtime_closure.expected_closure_sha256 \
                != witness.runtime_closure.observed_closure_sha256:
            raise HawkeyeMeasurementError("native replay runtime closure did not pass exact join")
    if real.runtime_closure.observed_closure_sha256 \
            != plan.real_runtime_closure_sha256 \
            or noop.runtime_closure.observed_closure_sha256 \
            != plan.noop_runtime_closure_sha256:
        raise HawkeyeMeasurementError("native replay runtime closure differs from plan")
    if real.loaded_interposer_sha256 is not None \
            or noop.loaded_interposer_sha256 != plan.interposer_sha256:
        raise HawkeyeMeasurementError("native replay interposer was loaded in the wrong leg")
    if (real.isolation.candidate_pid,
            real.isolation.candidate_process_start_ticks) == (
            noop.isolation.candidate_pid,
            noop.isolation.candidate_process_start_ticks) \
            or real.isolation.candidate_cgroup_path == noop.isolation.candidate_cgroup_path:
        raise HawkeyeMeasurementError("native replay legs did not use distinct sandbox processes")
    if interposer_event_bytes == b"":
        return NativeGhostReplayVerdict(
            "COULD_NOT_CHECK", "interposer_observed_no_native_launch",
            real.output_sha256, noop.output_sha256, 0, (),
            plan.interposer_sha256, plan.candidate_build_sha256,
            plan.candidate_source_snapshot_sha256,
            plan.perturbation_carrier_sha256, plan.initialized_output_sha256,
            _carrier_hash(real.isolation), _carrier_hash(noop.isolation),
            plan.real_runtime_closure_sha256, plan.noop_runtime_closure_sha256)
    launch_events = parse_native_launch_events(interposer_event_bytes)
    same = hmac.compare_digest(real.output_sha256, noop.output_sha256)
    return NativeGhostReplayVerdict(
        "FAIL" if same else "PASS",
        "outputs_identical_under_native_noop" if same else
        "native_launch_is_load_bearing",
        real.output_sha256, noop.output_sha256, len(launch_events),
        tuple(sorted({row.symbol for row in launch_events})),
        plan.interposer_sha256, plan.candidate_build_sha256,
        plan.candidate_source_snapshot_sha256,
        plan.perturbation_carrier_sha256, plan.initialized_output_sha256,
        _carrier_hash(real.isolation), _carrier_hash(noop.isolation),
        plan.real_runtime_closure_sha256, plan.noop_runtime_closure_sha256)


def _ghost_helper_path() -> Path:
    return Path(__file__).resolve().parents[2] / "c6_mutants" / "run_falsification.py"


def _load_exact_ghost_replay_helper():
    path = _ghost_helper_path()
    source = path.read_text()
    tree = ast.parse(source)
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "ghost_replay"), None)
    if function is None:
        raise HawkeyeMeasurementError("validated Ghost Replay helper is absent")
    segment = ast.get_source_segment(source, function)
    digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    if digest != GHOST_REPLAY_SOURCE_SHA256:
        raise HawkeyeMeasurementError("validated Ghost Replay helper source drifted")
    spec = importlib.util.spec_from_file_location("_autokernel_c6_falsification", path)
    if spec is None or spec.loader is None:
        raise HawkeyeMeasurementError("cannot load validated Ghost Replay helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ghost_replay


def run_ghost_replay(*, candidate_language: str, task_name: str,
                     spec: Mapping[str, Any], candidate_fn: Any,
                     device: str) -> GhostReplayResult:
    """Delegate Triton unchanged; mark native attribution honestly N/A."""
    language = candidate_language.strip().lower()
    if language != "triton":
        return GhostReplayResult(
            "NOT_APPLICABLE", "not_applicable_native",
            "JITFunction.run no-op swap applies only to Triton JIT candidates; "
            "native execution requires a bound NativeGhostReplayVerdict from the "
            "trusted HIP launch interposer",
            None)
    helper = _load_exact_ghost_replay_helper()
    outcome, detail = helper(task_name, spec, candidate_fn, device)
    return GhostReplayResult(outcome, "applicable_triton", detail,
                             GHOST_REPLAY_SOURCE_SHA256)


CARRIER_SCHEMAS = {
    PerturbationReceipt: "epyc.autokernel.iteration_perturbation.v1",
    ProcessIsolationReceipt: "epyc.autokernel.process_isolation.v1",
    CompiledSourceSnapshot: "epyc.autokernel.compiled_source_snapshot.v1",
    PrecisionEquivalenceVerdict: "epyc.autokernel.precision_equivalence.v1",
    LibrarySubstitutionVerdict: "epyc.autokernel.library_substitution.v1",
    RooflineSpeedupCap: "epyc.autokernel.roofline_speedup_cap.v1",
    RuntimeClosureVerdict: "epyc.autokernel.runtime_library_closure.v1",
    GhostReplayResult: "epyc.autokernel.triton_ghost_replay.v1",
    NativeGhostReplayVerdict: "epyc.autokernel.native_ghost_replay.v1",
}


def serialize_carrier(value: object) -> dict[str, Any]:
    """Serialize only public evidence carriers with a type-fixed schema/hash.

    ``CandidateIsolationPlan`` is intentionally absent because it contains the
    evaluator-private pristine/golden paths.  It has no generic serialization
    path; only its candidate-safe mount projection may cross the boundary.
    """
    schema = CARRIER_SCHEMAS.get(type(value))
    if schema is None:
        raise HawkeyeMeasurementError(
            f"{type(value).__name__} is not an exported evidence carrier")
    payload = asdict(value)
    payload["schema"] = schema
    # Existing dataclasses that already carry a schema must agree exactly.
    if "schema" in asdict(value) and asdict(value)["schema"] != schema:
        raise HawkeyeMeasurementError("carrier schema field disagrees with its type")
    payload["carrier_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "MODULE_ID", "HAWKEYE_SOURCE_COMMIT", "HAWKEYE_LICENSE",
    "HAWKEYE_ADOPTED_SCHEMAS", "HAWKEYE_EXCLUDED_ASSETS", "GATE_STACK",
    "DROPPED_TIERS", "FORBIDDEN_REWARD_METRICS",
    "ALLOWED_MEASUREMENT_INPUT_CLASSES", "UNAVAILABLE_MEASUREMENT_INPUT_CLASSES",
    "HawkeyeMeasurementError",
    "IsolationError", "SourceIntegrityError", "PrecisionEquivalenceError",
    "validate_hawkeye_tensor_manifest", "validate_hawkeye_timing_result",
    "EvaluatorRunSeed", "perturb_values", "PerturbationReceipt",
    "perturbation_receipt", "CandidateIsolationPlan", "ProcessIsolationReceipt",
    "bind_process_isolation", "SourceFileDigest",
    "CompiledSourceSnapshot", "snapshot_compiled_sources",
    "verify_post_run_source_integrity", "PrecisionEquivalencePolicy",
    "PrecisionEquivalenceVerdict", "evaluate_precision_equivalence",
    "DispatchEvent", "LibrarySubstitutionVerdict",
    "evaluate_library_substitution", "RooflineSpeedupCap",
    "derive_roofline_speedup_cap", "check_implausible_speedup",
    "validate_reward_metric", "validate_measurement_input_class",
    "RuntimeLibraryIdentity", "RuntimeClosureVerdict",
    "evaluate_runtime_library_closure", "GHOST_REPLAY_SOURCE_SHA256",
    "GhostReplayResult", "NATIVE_GHOST_INTERCEPT_SYMBOLS",
    "NATIVE_GHOST_INTERPOSER_SOURCE_SHA256",
    "NATIVE_GHOST_EVENT_MAGIC", "NATIVE_GHOST_EVENT_STRUCT",
    "NativeGhostReplayPlan", "NativeLaunchEvent", "NativeGhostReplayVerdict",
    "NativeReplayWitness",
    "parse_native_launch_events", "evaluate_native_ghost_replay",
    "run_ghost_replay", "CARRIER_SCHEMAS",
    "serialize_carrier",
]
