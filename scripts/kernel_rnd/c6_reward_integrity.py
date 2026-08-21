#!/usr/bin/env python3
"""c6_reward_integrity.py — C6 anti-reward-hacking + provenance harness for the
MI210 auto-kernel authoring loop (Phase 2 of the kernel-R&D loop).

STANDALONE, importable, GPU-free. This module is the loop's owned
differentiator — **C6 (reward integrity)** — plus its provenance layer. It is
NOT wired into kernel_eval.sh / kernel_sweep.sh; the future Phase-2 loop imports
it. The logic is ported from the proven design in the MIT-licensed reference
repo github.com/MrSteeeve/OpenHyra (sandbox.py / provenance.py / stopping.py /
eb.py), adapted to the SOL-ExecBench kernel task/scoring contract.

Discipline it enforces (why each exists):
  * Anti-TOCTOU snapshot — a candidate cannot mutate its artifact after the
    evaluator has looked at it. We SIGKILL the candidate's whole process group
    BEFORE snapshotting, open O_NOFOLLOW, reject symlink / FIFO / non-regular /
    multiply-linked files, cap size, then chmod 0444 an immutable copy.
  * Trusted evaluator — the score is RECOMPUTED by parent-controlled code on the
    immutable snapshot; any self-reported number in the candidate output is
    ignored. A candidate can never grade its own homework.
  * Correctness-gate-BEFORE-latency (lexicographic, mirrors kernel_store's
    `_is_correct`) — `is_correct` MUST pass before any latency / sol_score is
    recorded or ranked. A fast-but-wrong kernel scores nothing.
  * Run-manifest provenance — a sha256 over {sources, task spec, evaluator,
    config}; a resume is REFUSED if any result-affecting input drifted. A
    flock-based single-writer RunLock stops two harnesses sharing one run dir.
  * Evidence-gated stop — an autonomous "stop" is only a REQUEST; it is honored
    only when deterministic guards computed from evaluator RECORDS agree.
    Malformed / empty input can never trigger a stop.
  * Linux sandbox backend — bwrap / unshare + resource.setrlimit (NOT macOS
    Seatbelt). If no sandbox tool works, we FAIL CLOSED (raise) — we never run a
    candidate unsandboxed silently. Availability is probed at import.

KernelEvaluation numbers remain OBSERVATIONs (MEASUREMENT.md) and never gate a
keep/revert/deploy/promote decision.  The separately named AK-PM-12 receipt has
a versioned protocol and gates only write-side skill admission; it cannot
authorize production. The operator alone authorizes any production push.
"""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping

# --- read-only reuse of kernel_store's correctness semantics -----------------
# We import kernel_store purely to reuse `_is_correct` (its lexicographic
# correctness definition for kernel_eval.sh JSONL records). The import has no
# side effects (kernel_store only defines functions/constants at module scope;
# any DB work is guarded behind functions and __main__). We NEVER call its
# mutating entry points and we do not modify any of its symbols.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:  # pragma: no cover - trivial import guard
    import kernel_store as _kernel_store
except Exception:  # pragma: no cover
    _kernel_store = None


# =============================================================================
# Exceptions
# =============================================================================
class SandboxUnavailable(RuntimeError):
    """Raised (fail-closed) when no sandbox backend is available and the caller
    did not explicitly authorize unsandboxed execution."""


class ArtifactRejected(ValueError):
    """Raised when a candidate artifact fails an anti-TOCTOU integrity check."""


class ProvenanceError(RuntimeError):
    """Raised on run-manifest checksum mismatch or resume-drift rejection."""


class RunLockError(RuntimeError):
    """Raised when a run directory is already owned by another writer."""


class EvaluatorPolicyError(ValueError):
    """Raised when a numeric evaluator policy would weaken the pinned gate."""


class UnknownHardwareError(EvaluatorPolicyError):
    """Raised when no exact hardware authority exists for a requested part."""


class AdmissionReceiptError(ValueError):
    """Raised when a write-side admission receipt is malformed or tampered."""


# =============================================================================
# Ratified C6 evaluator policy (RVP-C6-22/23/24)
# =============================================================================
# Source pin: flashinfer-ai/flashinfer-bench at this exact revision.  The pinned
# files are bench/config.py, bench/utils.py, and evaluators/{default,lowbit}.py.
# We deliberately copy the tiny numeric contract rather than importing or
# executing that CUDA-only repository.
FLASHINFER_BENCH_SOURCE_COMMIT = (
    "40e6ca7844b514eb4b1c7edba6d6a7377df57870")
FLASHINFER_DEFAULT_ATOL = 1e-2
FLASHINFER_DEFAULT_RTOL = 1e-2
FLASHINFER_LOWBITS_MATCHED_RATIO = 0.95
FLASHINFER_RELATIVE_EPSILON = 1e-8

# L3 is intentionally absent.  The semantic judge is retained but cannot gate
# until the fixed omission-mutant calibration corpus is rejected in full.
C6_GATE_TIERS = ("L1_static", "L2_ghost_replay", "semantic_judge")
C6_DROPPED_TIERS = ("L3",)
C6_SEMANTIC_CALIBRATION_MUTANTS = (
    "layernorm_no_affine",
    "softmax_no_maxsub",
    "matmul_transpose_no_t",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluatorPolicyError(f"{name} must be non-empty text")
    return value.strip()


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluatorPolicyError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise EvaluatorPolicyError(
            f"{name} must be finite" + (" and > 0" if positive else ""))
    return result


def _normal_dtype(value: object) -> str:
    text = _required_text(value, "dtype").lower()
    for prefix in ("torch.", "tl.", "numpy.", "np."):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


@dataclass(frozen=True)
class PrecisionContract:
    """Operator-owned precision and tolerance contract.

    Output and accumulator dtypes are mandatory and are checked before any
    value tolerance.  Tolerances may match or tighten the pinned
    FlashInfer-Bench defaults, never loosen them.  Low-bit policies must retain
    at least the pinned 0.95 matched ratio.
    """

    required_output_dtype: str
    required_accumulator_dtype: str
    atol: float = FLASHINFER_DEFAULT_ATOL
    rtol: float = FLASHINFER_DEFAULT_RTOL
    required_matched_ratio: float = 1.0
    lowbit: bool = False

    def __post_init__(self):
        object.__setattr__(self, "required_output_dtype",
                           _normal_dtype(self.required_output_dtype))
        object.__setattr__(self, "required_accumulator_dtype",
                           _normal_dtype(self.required_accumulator_dtype))
        atol = _finite_number(self.atol, "atol", positive=True)
        rtol = _finite_number(self.rtol, "rtol", positive=True)
        rho = _finite_number(
            self.required_matched_ratio, "required_matched_ratio", positive=True)
        if atol > FLASHINFER_DEFAULT_ATOL or rtol > FLASHINFER_DEFAULT_RTOL:
            raise EvaluatorPolicyError(
                "per-element bounds must be equal to or tighter than the pinned "
                "FlashInfer-Bench atol=rtol=1e-2 defaults")
        floor = FLASHINFER_LOWBITS_MATCHED_RATIO if self.lowbit else 1.0
        if rho > 1.0 or rho < floor:
            raise EvaluatorPolicyError(
                f"required_matched_ratio must be in [{floor}, 1.0]")
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "required_matched_ratio", rho)

    def allowed_outliers(self, element_count: int) -> int:
        if isinstance(element_count, bool) or not isinstance(element_count, int) \
                or element_count < 0:
            raise EvaluatorPolicyError("element_count must be a non-negative int")
        # A tiny epsilon prevents binary representation of 0.05 from turning an
        # exactly integral budget into the preceding integer.
        return math.floor(
            element_count * (1.0 - self.required_matched_ratio) + 1e-12)


@dataclass(frozen=True)
class StructuralPrecisionEvidence:
    """Trusted static/IR inspection result, never candidate self-report."""

    output_dtype: str
    accumulator_dtype: str
    evidence_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "output_dtype", _normal_dtype(self.output_dtype))
        object.__setattr__(self, "accumulator_dtype",
                           _normal_dtype(self.accumulator_dtype))
        if not isinstance(self.evidence_sha256, str) \
                or not _SHA256_RE.fullmatch(self.evidence_sha256):
            raise EvaluatorPolicyError(
                "structural precision evidence requires an exact sha256")


@dataclass(frozen=True)
class NumericalVerdict:
    correct: bool
    stage: str
    reason: str
    structural_evidence_sha256: str | None = None
    required_output_dtype: str | None = None
    observed_output_dtype: str | None = None
    required_accumulator_dtype: str | None = None
    observed_accumulator_dtype: str | None = None
    total_elements: int = 0
    matched_elements: int = 0
    outlier_elements: int = 0
    allowed_outliers: int = 0
    matched_ratio: float | None = None
    max_absolute_error: float | str | None = None
    max_relative_error: float | str | None = None
    nonfinite_count: int = 0


def _flat_values(value: object) -> list[object]:
    if hasattr(value, "detach") and hasattr(value, "reshape"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        result: list[object] = []
        for item in value:
            result.extend(_flat_values(item))
        return result
    return [value]


def _value_shape(value: object) -> object:
    if hasattr(value, "shape"):
        return tuple(int(item) for item in value.shape)
    if isinstance(value, (list, tuple)):
        return (len(value), tuple(_value_shape(item) for item in value))
    return ()


def evaluate_numerics(reference: object, candidate: object, *,
                      structural: StructuralPrecisionEvidence,
                      policy: PrecisionContract) -> NumericalVerdict:
    """Apply structural precision first, then the pinned numeric predicate.

    An element is an outlier iff BOTH ``abs_error > atol`` AND
    ``rel_error > rtol``.  This is FlashInfer-Bench's exact elementwise AND
    predicate.  The matched-ratio budget is explicit and max errors are always
    recorded once numeric comparison begins.  Any non-finite value refuses the
    candidate outright.
    """
    precision = {
        "structural_evidence_sha256": structural.evidence_sha256,
        "required_output_dtype": policy.required_output_dtype,
        "observed_output_dtype": structural.output_dtype,
        "required_accumulator_dtype": policy.required_accumulator_dtype,
        "observed_accumulator_dtype": structural.accumulator_dtype,
    }
    if structural.output_dtype != policy.required_output_dtype:
        return NumericalVerdict(
            False, "structural", "incorrect_output_dtype", **precision)
    if structural.accumulator_dtype != policy.required_accumulator_dtype:
        return NumericalVerdict(
            False, "structural", "incorrect_accumulator_dtype", **precision)

    if _value_shape(reference) != _value_shape(candidate):
        return NumericalVerdict(
            False, "structural", "incorrect_shape", **precision)
    refs = _flat_values(reference)
    got = _flat_values(candidate)
    if not refs:
        return NumericalVerdict(
            True, "numeric", "matched", **precision, matched_ratio=1.0)
    try:
        refs_f = [float(item) for item in refs]
        got_f = [float(item) for item in got]
    except (TypeError, ValueError, OverflowError):
        return NumericalVerdict(
            False, "numeric", "non_numeric_output", **precision)

    nonfinite = sum(
        1 for ref, out in zip(refs_f, got_f)
        if not math.isfinite(ref) or not math.isfinite(out))
    if nonfinite:
        kind = "nan" if any(math.isnan(item) for item in refs_f + got_f) else "inf"
        return NumericalVerdict(
            False, "numeric", "nonfinite_output", **precision,
            total_elements=len(got_f),
            max_absolute_error=kind, max_relative_error=kind,
            nonfinite_count=nonfinite)

    abs_errors = [abs(out - ref) for ref, out in zip(refs_f, got_f)]
    rel_errors = [
        error / (abs(ref) + FLASHINFER_RELATIVE_EPSILON)
        for error, ref in zip(abs_errors, refs_f)
    ]
    outliers = sum(
        1 for abs_error, rel_error in zip(abs_errors, rel_errors)
        if abs_error > policy.atol and rel_error > policy.rtol)
    total = len(got_f)
    matched = total - outliers
    ratio = matched / total
    budget = policy.allowed_outliers(total)
    correct = ratio >= policy.required_matched_ratio
    return NumericalVerdict(
        correct, "numeric", "matched" if correct else "outlier_budget_exceeded",
        **precision, total_elements=total, matched_elements=matched,
        outlier_elements=outliers, allowed_outliers=budget,
        matched_ratio=ratio, max_absolute_error=max(abs_errors),
        max_relative_error=max(rel_errors), nonfinite_count=0)


def _bitwise_bytes(value: object) -> bytes:
    """Canonical, type-sensitive bytes for exact three-run comparison."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu().contiguous().clone()
        shape = tuple(int(item) for item in tensor.shape)
        return (b"tensor:" + str(tensor.dtype).encode() + b":" +
                repr(shape).encode() + b":" + bytes(tensor.untyped_storage()))
    if isinstance(value, bool):
        return b"bool:1" if value else b"bool:0"
    if isinstance(value, int):
        return b"int:" + str(value).encode()
    if isinstance(value, float):
        return b"float64:" + struct.pack(">d", value)
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, (list, tuple)):
        kind = b"list:" if isinstance(value, list) else b"tuple:"
        pieces = [_bitwise_bytes(item) for item in value]
        return kind + b"".join(
            len(piece).to_bytes(8, "big") + piece for piece in pieces)
    if isinstance(value, Mapping):
        pieces = []
        for key in sorted(value):
            key_blob = _bitwise_bytes(key)
            val_blob = _bitwise_bytes(value[key])
            pieces.append(len(key_blob).to_bytes(8, "big") + key_blob)
            pieces.append(len(val_blob).to_bytes(8, "big") + val_blob)
        return b"map:" + b"".join(pieces)
    raise EvaluatorPolicyError(
        f"unsupported deterministic output type: {type(value).__name__}")


@dataclass(frozen=True)
class DeterminismVerdict:
    correct: bool
    run_count: int
    bitwise_sha256: tuple[str, str, str]
    reason: str


def run_three_bitwise(run_once: Callable[[], object]) -> tuple[
        DeterminismVerdict, object]:
    """Run exactly three times on the same caller-owned input and compare bits."""
    outputs = [run_once() for _ in range(3)]
    digests = tuple(
        hashlib.sha256(_bitwise_bytes(item)).hexdigest() for item in outputs)
    correct = len(set(digests)) == 1
    return (DeterminismVerdict(
        correct, 3, digests,
        "bitwise_identical" if correct else "nondeterministic_output"),
            outputs[0])


class _FallbackReturnRaiser(ast.NodeTransformer):
    def __init__(self):
        self.replaced = 0
        self._inside_handler = 0

    def visit_ExceptHandler(self, node):  # noqa: N802 - ast API spelling
        self._inside_handler += 1
        node = self.generic_visit(node)
        self._inside_handler -= 1
        return node

    def visit_Return(self, node):  # noqa: N802 - ast API spelling
        if not self._inside_handler:
            return node
        self.replaced += 1
        return ast.copy_location(ast.Raise(
            exc=ast.Call(func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                         args=[ast.Constant(
                             value="C6 fallback return disabled for re-run")],
                         keywords=[]),
            cause=None), node)


def replace_fallback_returns_with_raise(source: str) -> tuple[str, int]:
    """Mutation probe: turn every exception-handler return into a hard raise."""
    tree = ast.parse(source)
    transformer = _FallbackReturnRaiser()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", transformer.replaced


@dataclass(frozen=True)
class FallbackProbe:
    correct: bool
    mutated_returns: int
    rerun_passed: bool
    reason: str


def probe_fallback_laundering(
        wrapper_source: str,
        rerun: Callable[[str], bool]) -> FallbackProbe:
    """Disable fallback returns and require the trusted caller's re-run to pass."""
    mutated, count = replace_fallback_returns_with_raise(wrapper_source)
    if count == 0:
        return FallbackProbe(True, 0, True, "no_fallback_return")
    passed = rerun(mutated)
    if not isinstance(passed, bool):
        raise EvaluatorPolicyError("fallback re-run must return an exact bool")
    return FallbackProbe(
        passed, count, passed,
        "fallback_free_rerun_passed" if passed else "fallback_laundering_detected")


@dataclass(frozen=True)
class SemanticJudgeCalibration:
    gating: bool
    rejected_mutants: tuple[str, ...]
    missing_mutants: tuple[str, ...]
    reason: str


def calibrate_semantic_judge(
        verdicts: Mapping[str, str]) -> SemanticJudgeCalibration:
    """Keep the judge non-gating until it rejects all three fixed mutants."""
    if not isinstance(verdicts, Mapping):
        raise EvaluatorPolicyError("semantic judge verdicts must be a mapping")
    required = set(C6_SEMANTIC_CALIBRATION_MUTANTS)
    extras = set(verdicts) - required
    if extras:
        raise EvaluatorPolicyError(
            f"unknown semantic calibration mutants: {sorted(extras)}")
    rejected = tuple(sorted(
        name for name, verdict in verdicts.items() if verdict == "REJECT"))
    invalid = {
        name: verdict for name, verdict in verdicts.items()
        if verdict not in {"REJECT", "ACCEPT"}}
    if invalid:
        raise EvaluatorPolicyError(
            f"semantic judge verdicts must be REJECT or ACCEPT: {invalid}")
    missing = tuple(sorted(required - set(rejected)))
    gating = not missing
    return SemanticJudgeCalibration(
        gating, rejected, missing,
        "calibrated_rejects_all_three" if gating else
        "non_gating_until_all_three_rejected")


SUPPORTED_GPU_PARTS = {
    "gfx90a": {
        "part": "gfx90a",
        "device_family": "AMD Instinct MI200",
        "authority": "operator-ratified-local-hardware/v1",
    },
}


def require_supported_gpu(part: str) -> dict:
    """Return exact part authority; never fabricate a generic fallback spec."""
    key = _required_text(part, "gpu part")
    if key not in SUPPORTED_GPU_PARTS:
        raise UnknownHardwareError(
            f"unknown GPU part {key!r}; refusing instead of estimating hardware")
    return dict(SUPPORTED_GPU_PARTS[key])


# =============================================================================
# AK-PM-12 execution-gated write-side admission
# =============================================================================
ADMISSION_RECEIPT_SCHEMA = "epyc.autokernel.c6_admission_receipt.v1"
ADMISSION_CAPTURE_SCHEMA = "epyc.vidya.autokernel_c6_admission_capture.v1"
_ADMISSION_KEYS = {
    "schema", "task_id", "candidate_commit", "anchor_commit",
    "evaluator_commit", "metric", "metric_direction",
    "first_turn_anchor_latency_ms", "first_turn_candidate_latency_ms",
    "verification_anchor_latency_ms", "verification_candidate_latency_ms",
    "first_turn_correct", "verification_correct",
    "first_turn_speedup", "verification_speedup", "required_speedup",
    "alpha", "beta", "implausible_speedup_cap", "admitted", "reason",
    "reopen_when", "receipt_sha256",
}
_CAPTURE_KEYS = {
    "schema", "source_schema", "receipt_sha256", "task_id",
    "candidate_commit", "anchor_commit", "evaluator_commit", "metric",
    "metric_direction", "value", "unit", "category", "protocol_id",
    "reps", "producer_sha256", "capture_sha256",
}


@dataclass(frozen=True)
class AdmissionPolicy:
    implausible_speedup_cap: float
    alpha: float = 1.2
    beta: float = 1.2

    def __post_init__(self):
        alpha = _finite_number(self.alpha, "alpha", positive=True)
        beta = _finite_number(self.beta, "beta", positive=True)
        cap = _finite_number(
            self.implausible_speedup_cap,
            "implausible_speedup_cap", positive=True)
        if alpha < 1.2 or beta < 1.2:
            raise EvaluatorPolicyError(
                "admission alpha and beta must each be >= 1.2")
        if cap <= max(alpha, beta):
            raise EvaluatorPolicyError(
                "implausible speedup cap must exceed the admission floor")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "implausible_speedup_cap", cap)


def _commit(value: object, name: str) -> str:
    if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
        raise EvaluatorPolicyError(f"{name} must be an exact 40-hex commit")
    return value


def build_admission_receipt(
        *, task_id: str, candidate_commit: str, anchor_commit: str,
        evaluator_commit: str, first_turn_anchor_latency_ms: float,
        first_turn_candidate_latency_ms: float,
        verification_anchor_latency_ms: float,
        verification_candidate_latency_ms: float, first_turn_correct: bool,
        verification_correct: bool, reopen_when: str,
        policy: AdmissionPolicy) -> dict:
    """Seal every admission attempt, including threshold/cap refusals.

    The verification re-run must satisfy ``r_verify >= max(beta,
    alpha*r_1)``.  The raw latency scalars are retained so readers can
    recompute both ratios.  Implausible ratios are recorded but refused.
    """
    task_id = _required_text(task_id, "task_id")
    candidate_commit = _commit(candidate_commit, "candidate_commit")
    anchor_commit = _commit(anchor_commit, "anchor_commit")
    evaluator_commit = _commit(evaluator_commit, "evaluator_commit")
    reopen_when = _required_text(reopen_when, "reopen_when")
    if type(first_turn_correct) is not bool or type(verification_correct) is not bool:
        raise EvaluatorPolicyError("admission correctness fields must be exact bools")
    ft_anchor = _finite_number(
        first_turn_anchor_latency_ms, "first_turn_anchor_latency_ms",
        positive=True)
    ft_candidate = _finite_number(
        first_turn_candidate_latency_ms, "first_turn_candidate_latency_ms",
        positive=True)
    vr_anchor = _finite_number(
        verification_anchor_latency_ms, "verification_anchor_latency_ms",
        positive=True)
    vr_candidate = _finite_number(
        verification_candidate_latency_ms,
        "verification_candidate_latency_ms", positive=True)
    first_speedup = ft_anchor / ft_candidate
    verify_speedup = vr_anchor / vr_candidate
    threshold = max(policy.beta, policy.alpha * first_speedup)
    implausible = (
        first_speedup > policy.implausible_speedup_cap
        or verify_speedup > policy.implausible_speedup_cap)
    correctness_passed = first_turn_correct and verification_correct
    admitted = correctness_passed and not implausible and verify_speedup >= threshold
    reason = (
        "correctness_refused" if not correctness_passed else
        "implausible_speedup_refused" if implausible else
        "verification_threshold_met" if admitted else
        "verification_threshold_not_met")
    receipt = {
        "schema": ADMISSION_RECEIPT_SCHEMA,
        "task_id": task_id,
        "candidate_commit": candidate_commit,
        "anchor_commit": anchor_commit,
        "evaluator_commit": evaluator_commit,
        "metric": "first_turn_latency_ms",
        "metric_direction": "lower_is_better",
        "first_turn_anchor_latency_ms": ft_anchor,
        "first_turn_candidate_latency_ms": ft_candidate,
        "verification_anchor_latency_ms": vr_anchor,
        "verification_candidate_latency_ms": vr_candidate,
        "first_turn_correct": first_turn_correct,
        "verification_correct": verification_correct,
        "first_turn_speedup": first_speedup,
        "verification_speedup": verify_speedup,
        "required_speedup": threshold,
        "alpha": policy.alpha,
        "beta": policy.beta,
        "implausible_speedup_cap": policy.implausible_speedup_cap,
        "admitted": admitted,
        "reason": reason,
        "reopen_when": reopen_when,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def validate_admission_receipt(receipt: Mapping[str, object]) -> dict:
    """Validate exact grammar, self-hash, ratios, disposition, and bindings."""
    if not isinstance(receipt, Mapping) or set(receipt) != _ADMISSION_KEYS:
        raise AdmissionReceiptError("admission receipt has missing or extra fields")
    if receipt.get("schema") != ADMISSION_RECEIPT_SCHEMA:
        raise AdmissionReceiptError("unknown admission receipt schema")
    unsigned = {key: receipt[key] for key in receipt if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != sha256_json(unsigned):
        raise AdmissionReceiptError("admission receipt self-hash mismatch")
    try:
        rebuilt = build_admission_receipt(
            task_id=receipt["task_id"],
            candidate_commit=receipt["candidate_commit"],
            anchor_commit=receipt["anchor_commit"],
            evaluator_commit=receipt["evaluator_commit"],
            first_turn_anchor_latency_ms=receipt["first_turn_anchor_latency_ms"],
            first_turn_candidate_latency_ms=receipt["first_turn_candidate_latency_ms"],
            verification_anchor_latency_ms=
                receipt["verification_anchor_latency_ms"],
            verification_candidate_latency_ms=
                receipt["verification_candidate_latency_ms"],
            first_turn_correct=receipt["first_turn_correct"],
            verification_correct=receipt["verification_correct"],
            reopen_when=receipt["reopen_when"],
            policy=AdmissionPolicy(
                alpha=receipt["alpha"], beta=receipt["beta"],
                implausible_speedup_cap=receipt["implausible_speedup_cap"]),
        )
    except EvaluatorPolicyError as exc:
        raise AdmissionReceiptError(str(exc)) from exc
    if dict(receipt) != rebuilt:
        raise AdmissionReceiptError(
            "admission receipt does not match recomputed measurement disposition")
    return dict(receipt)


def build_admission_claim_capture(
        receipt: Mapping[str, object], *, producer_sha256: str) -> dict:
    """Write-side Vidya capture; the root adapter projects, never re-invents."""
    valid = validate_admission_receipt(receipt)
    if not isinstance(producer_sha256, str) \
            or not _SHA256_RE.fullmatch(producer_sha256):
        raise AdmissionReceiptError("producer_sha256 must be exact")
    capture = {
        "schema": ADMISSION_CAPTURE_SCHEMA,
        "source_schema": ADMISSION_RECEIPT_SCHEMA,
        "receipt_sha256": valid["receipt_sha256"],
        "task_id": valid["task_id"],
        "candidate_commit": valid["candidate_commit"],
        "anchor_commit": valid["anchor_commit"],
        "evaluator_commit": valid["evaluator_commit"],
        "metric": "verification_speedup",
        "metric_direction": "higher_is_better",
        "value": valid["verification_speedup"],
        "unit": "ratio",
        "category": "MEASUREMENT",
        "protocol_id": ADMISSION_RECEIPT_SCHEMA,
        "reps": 1,
        "producer_sha256": producer_sha256,
    }
    capture["capture_sha256"] = sha256_json(capture)
    return capture


def validate_admission_claim_capture(
        capture: Mapping[str, object], receipt: Mapping[str, object]) -> dict:
    valid = validate_admission_receipt(receipt)
    if not isinstance(capture, Mapping) or set(capture) != _CAPTURE_KEYS:
        raise AdmissionReceiptError("admission belief capture has invalid grammar")
    unsigned = {key: capture[key] for key in capture if key != "capture_sha256"}
    if capture.get("capture_sha256") != sha256_json(unsigned):
        raise AdmissionReceiptError("admission belief capture self-hash mismatch")
    expected = build_admission_claim_capture(
        valid, producer_sha256=capture.get("producer_sha256"))
    if dict(capture) != expected:
        raise AdmissionReceiptError("admission belief capture receipt binding mismatch")
    return dict(capture)


class AdmissionReceiptStore:
    """Append-only write-side hook.  This stores evidence; it retrieves none."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, receipt: Mapping[str, object], *,
               producer_sha256: str) -> dict:
        valid = validate_admission_receipt(receipt)
        envelope = {
            "receipt": valid,
            "belief_capture": build_admission_claim_capture(
                valid, producer_sha256=producer_sha256),
        }
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(
                envelope, sort_keys=True, separators=(",", ":"),
                allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return envelope

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    envelope = json.loads(line)
                    if set(envelope) != {"receipt", "belief_capture"}:
                        raise AdmissionReceiptError("envelope grammar mismatch")
                    validate_admission_claim_capture(
                        envelope["belief_capture"], envelope["receipt"])
                except (ValueError, TypeError, KeyError) as exc:
                    raise AdmissionReceiptError(
                        f"invalid admission record at line {line_number}: {exc}") from exc
                records.append(envelope)
        return records


# =============================================================================
# Separable AK-PM-16 and AK-PM-17 + RVP-C4-11 records (no cross-run memory)
# =============================================================================
def select_amdahl_target(family_shares: Mapping[str, float], *,
                         gpu_part: str) -> str:
    """Select the largest measured wall-share after exact GPU admission."""
    require_supported_gpu(gpu_part)
    if not isinstance(family_shares, Mapping) or not family_shares:
        raise EvaluatorPolicyError("family_shares must be a non-empty mapping")
    checked = {}
    for name, share in family_shares.items():
        name = _required_text(name, "family name")
        value = _finite_number(share, f"share for {name}")
        if value < 0 or value > 1:
            raise EvaluatorPolicyError("family shares must lie in [0, 1]")
        checked[name] = value
    return min(checked, key=lambda name: (-checked[name], name))


G15_FROZEN_V9_B128_SHARES = {
    "gather_scatter": 0.18631,
    "recurrent": 0.17464,
    "norm_activation_elementwise": 0.01490,
}


def retrodict_g15_selector() -> dict:
    selected = select_amdahl_target(
        G15_FROZEN_V9_B128_SHARES, gpu_part="gfx90a")
    return {
        "schema": "epyc.autokernel.g15_retrodiction.v1",
        "gpu_part": "gfx90a",
        "frame": "frozen-v9/B128",
        "family_shares": dict(G15_FROZEN_V9_B128_SHARES),
        "selected_family": selected,
        "expected_family": "gather_scatter",
        "selector_validated": selected == "gather_scatter",
    }


@dataclass(frozen=True)
class RoundReflexionRecord:
    """One-round reflection + estimate outcome; deliberately no read memory."""

    round_id: str
    candidate_commit: str
    was_diagnosis_correct: bool
    was_fix_effective: bool
    expected_outcome: str
    actual_outcome: str
    estimated_speedup: float
    achieved_speedup: float
    lessons: tuple[str, ...]
    avoid_patterns: tuple[str, ...]
    try_patterns: tuple[str, ...]

    def __post_init__(self):
        _required_text(self.round_id, "round_id")
        _commit(self.candidate_commit, "candidate_commit")
        if type(self.was_diagnosis_correct) is not bool \
                or type(self.was_fix_effective) is not bool:
            raise EvaluatorPolicyError("round verdicts must be exact bools")
        _required_text(self.expected_outcome, "expected_outcome")
        _required_text(self.actual_outcome, "actual_outcome")
        _finite_number(self.estimated_speedup, "estimated_speedup", positive=True)
        _finite_number(self.achieved_speedup, "achieved_speedup", positive=True)
        for name in ("lessons", "avoid_patterns", "try_patterns"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                    not isinstance(value, str) or not value.strip()
                    for value in values):
                raise EvaluatorPolicyError(
                    f"{name} must be a tuple of non-empty strings")

    def to_dict(self) -> dict:
        estimated = float(self.estimated_speedup)
        achieved = float(self.achieved_speedup)
        return {
            "schema": "epyc.autokernel.round_reflexion.v1",
            "round_id": self.round_id,
            "candidate_commit": self.candidate_commit,
            "was_diagnosis_correct": self.was_diagnosis_correct,
            "was_fix_effective": self.was_fix_effective,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "estimated_speedup": estimated,
            "achieved_speedup": achieved,
            "estimate_error_fraction": (achieved - estimated) / estimated,
            "lessons": list(self.lessons),
            "avoid_patterns": list(self.avoid_patterns),
            "try_patterns": list(self.try_patterns),
        }


# =============================================================================
# SOL-ExecBench task / scoring contract
# =============================================================================
@dataclass(frozen=True)
class KernelTaskSpec:
    """The SOL-ExecBench C5/C6 task + scoring contract.

    The six scoring-core fields are the contract the prompt pins:
      entry_point      e.g. "kernel.py::run" — module file + callable.
      target_hardware  e.g. "MI210/gfx90a".
      dependencies     declared deps (tuple of package names).
      is_correct       HARD GATE — must be True before any score/latency ranks.
      sol_score        speed-of-light score; None until correctness passes.
      latency_ms       measured latency; None until correctness passes.

    is_correct / sol_score / latency_ms are RESULT fields: they are populated
    ONLY by the trusted evaluator (see `trusted_evaluate`), never read from the
    candidate's self-report. On a fresh task spec they are the un-evaluated
    defaults (False / None / None).

    The remaining fields are execution + provenance metadata used by the harness
    (limits, evaluator path, declared source files, free-form config).
    """

    entry_point: str
    target_hardware: str
    dependencies: tuple[str, ...] = ()
    # --- result contract (trusted-evaluator-owned) ---
    is_correct: bool = False
    sol_score: float | None = None
    latency_ms: float | None = None
    # --- execution / provenance metadata ---
    evaluator: str | None = None
    sources: tuple[str, ...] = ()
    config: dict = field(default_factory=dict)
    artifact_name: str = "solution.json"
    timeout_s: int = 60
    max_memory_mb: int = 1024
    max_output_mb: int = 16
    max_artifact_bytes: int = 1024 * 1024
    evaluator_timeout_s: int = 120
    evaluator_max_memory_mb: int = 512

    def entry_module(self) -> str:
        """Return the module-file half of ``entry_point`` (before '::')."""
        return self.entry_point.split("::", 1)[0]

    def entry_callable(self) -> str | None:
        """Return the callable half of ``entry_point`` (after '::'), or None."""
        parts = self.entry_point.split("::", 1)
        return parts[1] if len(parts) == 2 else None

    def scoring_core(self) -> dict:
        """The six-field SOL-ExecBench scoring core as a plain dict."""
        return {
            "entry_point": self.entry_point,
            "target_hardware": self.target_hardware,
            "dependencies": list(self.dependencies),
            "is_correct": self.is_correct,
            "sol_score": self.sol_score,
            "latency_ms": self.latency_ms,
        }

    def with_result(self, is_correct, sol_score, latency_ms) -> "KernelTaskSpec":
        """Return a copy carrying an evaluation result, correctness-gated:
        a non-correct result NEVER carries a score or latency."""
        if not is_correct:
            sol_score = None
            latency_ms = None
        return replace(
            self,
            is_correct=bool(is_correct),
            sol_score=sol_score,
            latency_ms=latency_ms,
        )


@dataclass(frozen=True)
class KernelEvaluation:
    """A trusted-evaluator verdict. Constructed only via `gated(...)`, which
    enforces the correctness-before-latency invariant at the type boundary."""

    is_correct: bool
    sol_score: float | None
    latency_ms: float | None
    status: str  # ok | crash | timeout | rejected | cancelled
    note: str = ""
    metrics: dict = field(default_factory=dict)
    candidate_artifact_sha256: str | None = None

    @classmethod
    def gated(cls, *, is_correct, sol_score, latency_ms, status, note="",
              metrics=None, candidate_artifact_sha256=None) -> "KernelEvaluation":
        """Build a verdict with the hard gate applied: if not is_correct, the
        score and latency are dropped to None no matter what was passed in."""
        if not is_correct:
            sol_score = None
            latency_ms = None
        return cls(
            is_correct=bool(is_correct),
            sol_score=sol_score,
            latency_ms=latency_ms,
            status=status,
            note=note,
            metrics=metrics or {},
            candidate_artifact_sha256=candidate_artifact_sha256,
        )

    def to_record(self, task: KernelTaskSpec | None = None) -> dict:
        rec = asdict(self)
        if task is not None:
            rec["entry_point"] = task.entry_point
            rec["target_hardware"] = task.target_hardware
        rec["observation"] = True
        return rec


# =============================================================================
# Correctness gate (consistent with kernel_store._is_correct)
# =============================================================================
def kernel_eval_is_correct(rec: dict) -> int:
    """Lexicographic correctness for a kernel_eval.sh JSONL record.

    Delegates to kernel_store._is_correct when available (status==OK + full
    test-backend-ops pass + coherent/byte-identical output); otherwise mirrors
    that exact semantics so this module is self-contained if kernel_store cannot
    be imported."""
    if _kernel_store is not None:
        return _kernel_store._is_correct(rec)
    if rec.get("status") != "OK":
        return 0
    corr = rec.get("correctness", {}) or {}
    tbo = corr.get("test_backend_ops", "")
    ok_tbo = False
    if "/" in tbo:
        a = tbo.split("/")[0].strip().split()[-1]
        b = tbo.split("/")[1].strip().split()[0]
        ok_tbo = a.isdigit() and b.isdigit() and a == b
    ok_coh = corr.get("coherence") in ("byte-identical", "coherent")
    return 1 if (ok_tbo and ok_coh) else 0


def is_correct(obj) -> bool:
    """Unified correctness gate for either record shape.

    * A kernel_eval.sh JSONL record (has a 'correctness' block) is judged by
      `kernel_eval_is_correct`.
    * A C6 evaluation (KernelEvaluation, KernelTaskSpec, or a dict with an
      'is_correct' key) is judged by its boolean gate.
    """
    if isinstance(obj, (KernelEvaluation, KernelTaskSpec)):
        return bool(obj.is_correct)
    if isinstance(obj, dict):
        if "correctness" in obj:
            return bool(kernel_eval_is_correct(obj))
        if "is_correct" in obj:
            return bool(obj["is_correct"])
    return False


def _score_of(obj):
    if isinstance(obj, (KernelEvaluation, KernelTaskSpec)):
        return obj.sol_score
    if isinstance(obj, dict):
        return obj.get("sol_score")
    return None


def _latency_of(obj):
    if isinstance(obj, (KernelEvaluation, KernelTaskSpec)):
        return obj.latency_ms
    if isinstance(obj, dict):
        return obj.get("latency_ms")
    return None


def rank_correct_first(evaluations):
    """Rank evaluations lexicographic-correctness-first.

    ONLY correct evaluations with a real sol_score are ever ranked; a
    fast-but-wrong candidate is dropped and can never place. Among the correct,
    higher sol_score wins, ties broken by lower latency_ms. Returns a new list;
    the input is never mutated."""
    ranked = [
        e for e in evaluations
        if is_correct(e) and _score_of(e) is not None
    ]
    ranked.sort(
        key=lambda e: (
            -float(_score_of(e)),
            float(_latency_of(e)) if _latency_of(e) is not None else float("inf"),
        )
    )
    return ranked


# =============================================================================
# Linux sandbox backend (bwrap / unshare + setrlimit) — fail-closed
# =============================================================================
# A tiny in-process wrapper that clamps address space, output-file size and CPU
# seconds via resource.setrlimit, then exec()s the real command. Limits are
# inherited across the subsequent exec of the sandbox tool and the candidate.
LIMIT_WRAPPER = r"""
import os, resource, sys
limits = (
    (resource.RLIMIT_AS, int(sys.argv[1])),
    (resource.RLIMIT_FSIZE, int(sys.argv[2])),
    (resource.RLIMIT_CPU, int(sys.argv[3])),
)
for key, value in limits:
    if value <= 0:
        continue
    try:
        _soft, hard = resource.getrlimit(key)
        target = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(key, (target, target))
    except (OSError, ValueError):
        pass
os.execvp(sys.argv[4], sys.argv[4:])
"""

_ALLOW_ENV = "EPYC_C6_ALLOW_UNSANDBOXED"


def _probe_bwrap():
    exe = shutil.which("bwrap")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--ro-bind", "/", "/", "--dev", "/dev", "true"],
            capture_output=True, timeout=10,
        )
        return exe if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _probe_unshare():
    exe = shutil.which("unshare")
    if not exe:
        return None
    try:
        # Actually attempt the namespace op — a present binary is not enough;
        # user namespaces are frequently disabled (unprivileged containers).
        r = subprocess.run(
            [exe, "--user", "--map-root-user", "--net", "--", "true"],
            capture_output=True, timeout=10,
        )
        return exe if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_sandbox_backend():
    """Probe (not just which()) for a working sandbox backend.

    Returns (name, tool_path) — ('bwrap', ...) preferred, else ('unshare', ...),
    else (None, None). Called once at import; re-callable to re-probe."""
    tool = _probe_bwrap()
    if tool:
        return "bwrap", tool
    tool = _probe_unshare()
    if tool:
        return "unshare", tool
    return None, None


SANDBOX_BACKEND, SANDBOX_TOOL = detect_sandbox_backend()
SANDBOX_AVAILABLE = SANDBOX_BACKEND is not None


def _allow_unsandboxed(explicit):
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(_ALLOW_ENV) == "1"


def build_sandboxed_command(cmd, *, writable_dir, allow_unsandboxed=None):
    """Wrap ``cmd`` (a list) so the candidate runs isolated.

    Isolation strength by backend:
      bwrap    — read-only root, private /dev+/proc+/tmp, one writable bind on
                 ``writable_dir``, all namespaces unshared (network denied).
      unshare  — new user+net+pid namespaces (network denied), mount-proc.
    If NO backend is available this FAILS CLOSED (raises SandboxUnavailable)
    unless the caller passes allow_unsandboxed=True or sets
    EPYC_C6_ALLOW_UNSANDBOXED=1 — appropriate only inside an already-isolated
    container/VM (e.g. this devcontainer or CI)."""
    cmd = list(cmd)
    writable_dir = str(Path(writable_dir).resolve())
    if SANDBOX_BACKEND == "bwrap":
        return [
            SANDBOX_TOOL,
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", writable_dir, writable_dir,
            "--chdir", writable_dir,
            "--unshare-all",
            "--die-with-parent",
            "--",
        ] + cmd
    if SANDBOX_BACKEND == "unshare":
        return [
            SANDBOX_TOOL,
            "--user", "--map-root-user",
            "--net", "--pid", "--fork", "--mount-proc",
            "--",
        ] + cmd
    if _allow_unsandboxed(allow_unsandboxed):
        return cmd
    raise SandboxUnavailable(
        "no working sandbox backend (bwrap/unshare) — refusing to run a "
        "candidate unsandboxed. Set EPYC_C6_ALLOW_UNSANDBOXED=1 ONLY inside an "
        "external container/VM, or install bwrap / enable user namespaces."
    )


def rlimit_wrapped_command(cmd, task: KernelTaskSpec):
    """Prepend the setrlimit exec-wrapper (address space / fsize / cpu)."""
    mem = int(task.max_memory_mb) * 1024 * 1024
    out = int(task.max_output_mb) * 1024 * 1024
    cpu = int(task.timeout_s) + 5
    return [sys.executable, "-c", LIMIT_WRAPPER, str(mem), str(out), str(cpu), *cmd]


# =============================================================================
# Anti-TOCTOU immutable snapshot
# =============================================================================
READ_CHUNK_BYTES = 64 * 1024


def _kill_process_group(proc):
    """SIGKILL the candidate's whole session/process group. Closes the artifact
    mutation race: even descendants deliberately left running after the parent
    exits are gone before we snapshot."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def _read_regular_file(path, max_bytes):
    """Read ONE untrusted artifact without following links or blocking on FIFOs.

    Rejects (raising ArtifactRejected): a symbolic link, a non-regular file
    (FIFO / socket / device), a multiply-linked file (st_nlink != 1), and any
    file over ``max_bytes``. Opens O_NOFOLLOW|O_NONBLOCK — the O_NOFOLLOW is the
    anti-race second line of defense if a symlink is swapped in AFTER the lstat
    check but BEFORE the open."""
    path = Path(path)
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise ArtifactRejected(f"artifact not found: {path}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ArtifactRejected(f"artifact must not be a symbolic link: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        # ELOOP here == a symlink was swapped in after the lstat (TOCTOU).
        raise ArtifactRejected(f"could not safely open artifact: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactRejected(
                f"artifact must be a regular file (got mode {oct(info.st_mode)}): {path}"
            )
        if info.st_nlink != 1:
            raise ArtifactRejected(
                f"artifact must have exactly one hard link (st_nlink="
                f"{info.st_nlink}): {path}"
            )
        if info.st_size > max_bytes:
            raise ArtifactRejected(
                f"artifact exceeds the {max_bytes}-byte limit: {path}"
            )
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ArtifactRejected(
                f"artifact exceeds the {max_bytes}-byte limit: {path}"
            )
        return data
    finally:
        os.close(fd)


def snapshot_candidate_artifact(artifact_path, trusted_dir, *, proc=None,
                                max_bytes=1024 * 1024):
    """Kill the candidate's process group, then copy its validated artifact into
    a fresh parent-controlled directory as an immutable 0444 snapshot.

    The kill happens BEFORE the read so the candidate cannot mutate the file
    between validation and snapshot. Returns (snapshot_path, sha256, data)."""
    _kill_process_group(proc)
    data = _read_regular_file(artifact_path, max_bytes)
    trusted_dir = Path(trusted_dir)
    if trusted_dir.exists():
        shutil.rmtree(trusted_dir)
    trusted_dir.mkdir(parents=True)
    snapshot = trusted_dir / "solution.snapshot.json"
    snapshot.write_bytes(data)
    snapshot.chmod(0o444)
    digest = hashlib.sha256(data).hexdigest()
    return snapshot, digest, data


def trusted_artifact_dir(sandbox_dir):
    """A parent-controlled trusted dir OUTSIDE the candidate's write root."""
    sandbox_dir = Path(sandbox_dir)
    return sandbox_dir.parent / ".c6_trusted" / sandbox_dir.name


# =============================================================================
# Trusted evaluator (recompute score on the snapshot; ignore self-report)
# =============================================================================
def _wait_process(proc, timeout_s):
    started = time.monotonic()
    while True:
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            return "timeout"
        try:
            proc.wait(timeout=min(0.2, remaining))
            return "completed"
        except subprocess.TimeoutExpired:
            pass


def trusted_evaluate(task: KernelTaskSpec, snapshot_path) -> KernelEvaluation:
    """Recompute the verdict with PARENT-controlled evaluator code on the
    immutable snapshot. The candidate's own self-reported score/latency (if any
    inside the artifact) is IGNORED — the harness never reads a number from the
    candidate output; only the evaluator's JSON verdict counts.

    The evaluator is trusted code, so it runs under resource limits only (no
    sandbox), matching OpenHyra's trusted-scoring pattern. Its last stdout line
    must be a JSON object: {"is_correct": bool, "sol_score": <num>?,
    "latency_ms": <num>?, "metrics": {...}?} or {"error": "..."}.

    Correctness gate: even if the evaluator emits a score, a non-correct verdict
    is returned with sol_score=None and latency_ms=None."""
    if not task.evaluator:
        raise ValueError("task.evaluator is required for trusted_evaluate")
    snapshot_path = str(snapshot_path)
    digest = hashlib.sha256(Path(snapshot_path).read_bytes()).hexdigest()

    command = [sys.executable, str(task.evaluator), snapshot_path]
    limited = [
        sys.executable, "-c", LIMIT_WRAPPER,
        str(int(task.evaluator_max_memory_mb) * 1024 * 1024),
        str(int(task.max_output_mb) * 1024 * 1024),
        str(int(task.evaluator_timeout_s) + 5),
        *command,
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(
        limited, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    state = _wait_process(proc, task.evaluator_timeout_s)
    _kill_process_group(proc)  # trusted code must not leave descendants either
    stdout, stderr = proc.communicate()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if state == "timeout":
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="timeout",
            note="evaluator timed out", candidate_artifact_sha256=digest,
        )
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        verdict = json.loads(line)
    except ValueError:
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="crash",
            note=f"evaluator produced no verdict: {stderr.strip()[:300]}",
            candidate_artifact_sha256=digest,
        )
    if "error" in verdict:
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="rejected",
            note=f"evaluator rejected artifact: {verdict['error']}",
            candidate_artifact_sha256=digest,
        )

    correct = bool(verdict.get("is_correct"))
    score = verdict.get("sol_score")
    latency = verdict.get("latency_ms")
    metrics = dict(verdict.get("metrics", {}))
    metrics["evaluator_ms"] = round(elapsed_ms, 4)
    return KernelEvaluation.gated(
        is_correct=correct,
        sol_score=(float(score) if score is not None else None),
        latency_ms=(float(latency) if latency is not None else None),
        status="ok",
        note="",
        metrics=metrics,
        candidate_artifact_sha256=digest,
    )


def evaluate_candidate(candidate_cmd, work_dir, task: KernelTaskSpec, *,
                       env=None, allow_unsandboxed=None) -> KernelEvaluation:
    """End-to-end C6 candidate evaluation:

      1. run the (untrusted) candidate command sandboxed + rlimited in work_dir;
      2. SIGKILL its process group (close the mutation race);
      3. anti-TOCTOU snapshot of its artifact into a trusted dir (immutable);
      4. trusted-evaluator recomputes the verdict on the snapshot;
      5. correctness-gate the result (no score/latency unless is_correct).

    Returns a KernelEvaluation. Never raises for a merely-failing candidate
    (crash/timeout become a non-correct verdict); it DOES raise
    SandboxUnavailable if no sandbox backend and no explicit override."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    artifact = work_dir / task.artifact_name
    if artifact.exists():
        artifact.unlink()

    sandboxed = build_sandboxed_command(
        candidate_cmd, writable_dir=work_dir, allow_unsandboxed=allow_unsandboxed,
    )
    full = rlimit_wrapped_command(sandboxed, task)
    child_env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(work_dir),
        "TMPDIR": str(work_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if env:
        child_env.update(env)

    log_path = work_dir / "run.log"
    with open(log_path, "w") as log_stream:
        proc = subprocess.Popen(
            full, cwd=str(work_dir), env=child_env,
            stdout=log_stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_state = "completed"
        try:
            wait_state = _wait_process(proc, task.timeout_s)
        finally:
            _kill_process_group(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    log_tail = ""
    if log_path.exists():
        log_tail = "\n".join(
            log_path.read_text(errors="replace").splitlines()[-15:]
        )
    if wait_state == "timeout":
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="timeout",
            note=(f"killed candidate process group after {task.timeout_s}s\n"
                  f"{log_tail}").strip(),
        )
    if proc.returncode != 0:
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="crash",
            note=log_tail,
        )

    trusted_dir = trusted_artifact_dir(work_dir)
    try:
        snapshot, digest, _ = snapshot_candidate_artifact(
            artifact, trusted_dir, proc=proc, max_bytes=task.max_artifact_bytes,
        )
    except ArtifactRejected as exc:
        return KernelEvaluation.gated(
            is_correct=False, sol_score=None, latency_ms=None, status="rejected",
            note=(log_tail + f"\nartifact rejected: {exc}").strip(),
        )
    result = trusted_evaluate(task, snapshot)
    return replace(result, candidate_artifact_sha256=digest)


# =============================================================================
# Run-manifest provenance + single-writer RunLock
# =============================================================================
RUN_MANIFEST_SCHEMA = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload):
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build_run_manifest(task: KernelTaskSpec, *, run_id, sources=None,
                       config=None):
    """Build an immutable run manifest: a sha256 over {sources, task spec,
    evaluator, config}. ``sources`` is a {logical_name: path} map of the loop's
    own source files whose content must not drift across a resume.

    A change to ANY hashed input flips manifest_sha256 (and the per-field
    blocks), so `validate_run_manifest` can refuse a poisoned resume."""
    sources = sources or {}
    if task.evaluator is None:
        raise ValueError("task.evaluator is required to build a run manifest")
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id,
        "task": {
            "entry_point": task.entry_point,
            "target_hardware": task.target_hardware,
            "dependencies": list(task.dependencies),
            "artifact_name": task.artifact_name,
            "evaluator_sha256": sha256_file(task.evaluator),
        },
        "source_sha256": {
            name: sha256_file(path) for name, path in sorted(sources.items())
        },
        "config": config or dict(task.config),
        "evaluator_policy": {
            "flashinfer_bench_source_commit":
                FLASHINFER_BENCH_SOURCE_COMMIT,
            "maximum_atol": FLASHINFER_DEFAULT_ATOL,
            "maximum_rtol": FLASHINFER_DEFAULT_RTOL,
            "minimum_lowbit_matched_ratio":
                FLASHINFER_LOWBITS_MATCHED_RATIO,
            "elementwise_outlier_predicate": "abs_gt_atol_AND_rel_gt_rtol",
            "deterministic_runs": 3,
            "gate_tiers": list(C6_GATE_TIERS),
            "dropped_tiers": list(C6_DROPPED_TIERS),
        },
        "limits": {
            "timeout_s": task.timeout_s,
            "max_memory_mb": task.max_memory_mb,
            "max_output_mb": task.max_output_mb,
            "max_artifact_bytes": task.max_artifact_bytes,
            "evaluator_timeout_s": task.evaluator_timeout_s,
            "evaluator_max_memory_mb": task.evaluator_max_memory_mb,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "sandbox_backend": SANDBOX_BACKEND,
        },
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def write_run_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load_run_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise ProvenanceError(
            f"run provenance is missing: {path}; legacy runs cannot be resumed"
        )
    manifest = json.loads(path.read_text())
    expected = manifest.get("manifest_sha256")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if expected != sha256_json(unsigned):
        raise ProvenanceError(f"run provenance checksum mismatch: {path}")
    return manifest


def validate_run_manifest(recorded, current):
    """Refuse to resume when any result-affecting input drifted.

    Compares the result-affecting blocks {task (incl. evaluator sha),
    source_sha256, config, evaluator_policy, limits, environment}. On drift raises
    ProvenanceError naming the drifted field(s)."""
    mismatches = [
        field_name
        for field_name in ("task", "source_sha256", "config",
                           "evaluator_policy", "limits", "environment")
        if recorded.get(field_name) != current.get(field_name)
    ]
    if mismatches:
        raise ProvenanceError(
            "run provenance drift in " + ", ".join(mismatches) +
            "; start a new run_id instead of mixing experiments"
        )
    return recorded


class RunLock:
    """Non-blocking, single-writer flock over one run directory. A second
    holder (this process or another) fails fast rather than corrupting a run."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(self.path, "a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            self.stream.close()
            self.stream = None
            raise RunLockError(
                f"run {self.path.parent.name!r} is already owned by another "
                f"writer"
            ) from exc
        return self

    def release(self):
        if self.stream is None:
            return
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# =============================================================================
# Append-only, fsync'd record store (eb.py port)
# =============================================================================
class RecordStore:
    """All-outcomes append-only JSONL store with per-write fsync — every
    proposed kernel's verdict is durably recorded, win or lose."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> dict:
        with open(self.path, "a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def records(self):
        if not self.path.exists():
            return []
        with open(self.path) as stream:
            return [json.loads(line) for line in stream if line.strip()]


# =============================================================================
# Evidence-gated stop controller (stopping.py port)
# =============================================================================
@dataclass(frozen=True)
class KernelStopPolicy:
    enabled: bool = False
    min_records: int = 3
    min_correct: int = 3
    stop_patience: int = 3          # correct evals since last strict improvement
    meaningful_delta: float = 1e-9

    def __post_init__(self):
        for name in ("min_records", "min_correct", "stop_patience"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.meaningful_delta < 0:
            raise ValueError("meaningful_delta must be >= 0")


@dataclass(frozen=True)
class StopRequest:
    action: str          # "stop" | "continue"
    reason: str = ""


@dataclass(frozen=True)
class StopReview:
    accepted: bool
    reasons: tuple
    evidence: dict

    def to_dict(self):
        return {"accepted": self.accepted, "reasons": list(self.reasons),
                "evidence": self.evidence}


def _record_well_formed(rec) -> bool:
    """A record must be a dict from which correctness is derivable, and if it
    claims correctness it must carry a numeric sol_score. Anything else is
    malformed and can never contribute to a stop."""
    if not isinstance(rec, dict):
        return False
    if "correctness" not in rec and "is_correct" not in rec:
        return False
    if is_correct(rec):
        score = _score_of(rec)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return False
    return True


def stopping_evidence(records, policy: KernelStopPolicy) -> dict:
    """Deterministic evidence derived SOLELY from evaluator records."""
    total = len(records)
    malformed = sum(0 if _record_well_formed(r) else 1 for r in records)
    correct_scores = [
        float(_score_of(r)) for r in records
        if _record_well_formed(r) and is_correct(r) and _score_of(r) is not None
    ]
    running_best = None
    since_improvement = 0
    for score in correct_scores:
        if running_best is None or score > running_best + policy.meaningful_delta:
            running_best = score
            since_improvement = 0
        else:
            since_improvement += 1
    return {
        "total_records": total,
        "malformed_records": malformed,
        "correct_records": len(correct_scores),
        "best_score": running_best,
        "evals_since_meaningful_improvement": since_improvement,
    }


class KernelStopController:
    """Treat an autonomous stop as a REQUEST gated by deterministic evidence.

    The controller never trusts the requester's self-assessment; every guard is
    recomputed from the evaluator RECORDS. Malformed or empty input yields at
    least one blocking reason, so it can never trigger a stop."""

    def __init__(self, policy: KernelStopPolicy):
        self.policy = policy

    def review(self, request: StopRequest, records) -> StopReview:
        evidence = stopping_evidence(records, self.policy)
        reasons = []
        if request.action != "stop":
            reasons.append("not_a_stop_request")
        if not self.policy.enabled:
            reasons.append("stops_disabled")
        if evidence["total_records"] == 0:
            reasons.append("no_records")
        if evidence["malformed_records"]:
            reasons.append("malformed_records")
        if evidence["total_records"] < self.policy.min_records:
            reasons.append("minimum_records_not_met")
        if evidence["correct_records"] < self.policy.min_correct:
            reasons.append("insufficient_correct_records")
        if evidence["evals_since_meaningful_improvement"] < self.policy.stop_patience:
            reasons.append("patience_not_met")
        accepted = request.action == "stop" and not reasons
        return StopReview(accepted, tuple(reasons), evidence)


__all__ = [
    "KernelTaskSpec", "KernelEvaluation",
    "SandboxUnavailable", "ArtifactRejected", "ProvenanceError", "RunLockError",
    "EvaluatorPolicyError", "UnknownHardwareError", "AdmissionReceiptError",
    "FLASHINFER_BENCH_SOURCE_COMMIT", "FLASHINFER_DEFAULT_ATOL",
    "FLASHINFER_DEFAULT_RTOL", "FLASHINFER_LOWBITS_MATCHED_RATIO",
    "C6_GATE_TIERS", "C6_DROPPED_TIERS", "C6_SEMANTIC_CALIBRATION_MUTANTS",
    "PrecisionContract", "StructuralPrecisionEvidence", "NumericalVerdict",
    "evaluate_numerics", "DeterminismVerdict", "run_three_bitwise",
    "FallbackProbe", "replace_fallback_returns_with_raise",
    "probe_fallback_laundering", "SemanticJudgeCalibration",
    "calibrate_semantic_judge", "SUPPORTED_GPU_PARTS",
    "require_supported_gpu", "AdmissionPolicy", "ADMISSION_RECEIPT_SCHEMA",
    "ADMISSION_CAPTURE_SCHEMA", "build_admission_receipt",
    "validate_admission_receipt", "build_admission_claim_capture",
    "validate_admission_claim_capture", "AdmissionReceiptStore",
    "select_amdahl_target", "G15_FROZEN_V9_B128_SHARES",
    "retrodict_g15_selector", "RoundReflexionRecord",
    "is_correct", "kernel_eval_is_correct", "rank_correct_first",
    "detect_sandbox_backend", "build_sandboxed_command", "rlimit_wrapped_command",
    "SANDBOX_BACKEND", "SANDBOX_TOOL", "SANDBOX_AVAILABLE",
    "snapshot_candidate_artifact", "trusted_artifact_dir",
    "trusted_evaluate", "evaluate_candidate",
    "build_run_manifest", "write_run_manifest", "load_run_manifest",
    "validate_run_manifest", "RunLock", "RecordStore",
    "KernelStopPolicy", "StopRequest", "StopReview", "KernelStopController",
    "stopping_evidence",
]
