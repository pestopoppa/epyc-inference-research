"""Prospective C3 vendor-floor, ``fast_p``, and whole-model exit contracts.

This module is deliberately an offline reducer.  It does not launch Apex,
compile a kernel, capture tensors, or run a benchmark.  Numeric results can
enter only through evidence-bound observations supplied by a runner.

The EPYC suite contains two hash-bound C5 references (attention and MoE
dispatch) and one EPYC-native Q4_K dequant+GEMV contract.  The latter is not
misrepresented as a HyRA/C5 artifact: no dequant row exists in the checked-in
C5 corpus.
"""
from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import c5_seed_corpus, schemas


SCHEMA = "epyc.autokernel.c3_epyc_suite.v1"
TARGET_ARCH = "gfx90a"
TARGET_DEVICE = "AMD Instinct MI210"
PINNED_APEX_REVISION = "e06b5d1cd58996a82c5e2897164f760c3b3f87ac"
APEX_PYTHON_OVERLAY = "apex_python_overlay"
EPYC_EXPERIMENTAL_BINARY = "epyc_experimental_binary"

TORCH_ROCM_COMPILE = "torch_rocm_compile"
ROCBLAS = "rocblas"
HIPBLASLT = "hipblaslt"
VENDOR_PROVIDERS = frozenset({TORCH_ROCM_COMPILE, ROCBLAS, HIPBLASLT})
CANDIDATE_PROVIDER = "autokernel_candidate"

SEARCH_EXIT_AUTHORITY = "search_exit_diagnostic_only"
NO_PROMOTION_AUTHORITY = "no_release_or_promotion_authority"

_DEQUANT_CONTRACT = (
    "epyc-native/q4_k-dequant-gemv/v1|ggml_mul_mat|Q4_K|decode|gfx90a|"
    "captured-workload-shape-required"
)
_DEQUANT_CONTRACT_SHA256 = hashlib.sha256(_DEQUANT_CONTRACT.encode()).hexdigest()


class C3ContractError(ValueError):
    """A prospective declaration or completed receipt is not admissible."""


class IdentityMismatch(C3ContractError):
    """Evidence was measured on a different exact surface or implementation."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise C3ContractError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    value = _text(value, label)
    if not schemas.SHA256_RE.fullmatch(value) or schemas.is_placeholder_digest(value):
        raise C3ContractError(f"{label} must be a non-placeholder lowercase SHA-256")
    return value


def _positive_samples(values: Sequence[float], label: str) -> tuple[float, ...]:
    samples = tuple(values)
    if len(samples) < 3:
        raise C3ContractError(f"{label} requires at least three samples")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in samples):
        raise C3ContractError(f"{label} samples must be positive and finite")
    return tuple(float(value) for value in samples)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass(frozen=True)
class EpycOpCase:
    case_id: str
    operator_family: str
    source_kind: str
    source_ref: str
    source_revision: str
    source_artifact_sha256: str
    required_baseline_providers: tuple[str, ...]
    whole_model_exit_required: bool = True

    def __post_init__(self) -> None:
        for name in ("case_id", "operator_family", "source_kind", "source_ref",
                     "source_revision"):
            _text(getattr(self, name), name)
        _sha256(self.source_artifact_sha256, "source_artifact_sha256")
        providers = tuple(self.required_baseline_providers)
        if not providers or len(providers) != len(set(providers)):
            raise C3ContractError("required baseline providers must be non-empty and unique")
        if set(providers) - VENDOR_PROVIDERS:
            raise C3ContractError("EPYC suite baseline plan names a non-vendor provider")
        if not self.whole_model_exit_required:
            raise C3ContractError("every selected EPYC op requires the whole-model exit gate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "operator_family": self.operator_family,
            "source": {
                "kind": self.source_kind,
                "ref": self.source_ref,
                "revision": self.source_revision,
                "artifact_sha256": self.source_artifact_sha256,
            },
            "baseline": {
                "required_providers": list(self.required_baseline_providers),
                "eager_allowed": False,
                "exact_surface_required": True,
            },
            "whole_model_exit_required": self.whole_model_exit_required,
        }


def epyc_op_suite() -> tuple[EpycOpCase, ...]:
    """Return the exact prospective attention/MoE/dequant suite.

    Loading C5 is a provenance check, not performance evidence.  It verifies the
    two upstream rows against the checked-in registry on every call.
    """
    corpus = c5_seed_corpus.load()
    attention, moe = corpus.select(("k228", "k175"))
    expected = {
        "k228": "mla_paged_prefill",
        "k175": "moe_sparse_expert_dispatch",
    }
    for seed in (attention, moe):
        if seed.operator_family != expected[seed.seed_id]:
            raise C3ContractError(f"{seed.seed_id} operator-family identity drifted")
    return (
        EpycOpCase(
            case_id="epyc.attention.mla_paged_prefill.k228",
            operator_family=attention.operator_family,
            source_kind="hyra_c5_reference_only",
            source_ref=f"hyra-sol-execbench/{attention.seed_id}",
            source_revision=corpus.source_revision,
            source_artifact_sha256=attention.artifact_sha256,
            required_baseline_providers=(TORCH_ROCM_COMPILE,),
        ),
        EpycOpCase(
            case_id="epyc.moe.sparse_expert_dispatch.k175",
            operator_family=moe.operator_family,
            source_kind="hyra_c5_reference_only",
            source_ref=f"hyra-sol-execbench/{moe.seed_id}",
            source_revision=corpus.source_revision,
            source_artifact_sha256=moe.artifact_sha256,
            required_baseline_providers=(TORCH_ROCM_COMPILE,),
        ),
        EpycOpCase(
            case_id="epyc.dequant.q4_k_decode_gemv",
            operator_family="q4_k_dequant_gemv",
            source_kind="epyc_native_contract",
            source_ref="epyc-native/q4_k-dequant-gemv/v1",
            source_revision="prospective_contract_v1",
            source_artifact_sha256=_DEQUANT_CONTRACT_SHA256,
            required_baseline_providers=(TORCH_ROCM_COMPILE,),
        ),
    )


@dataclass(frozen=True)
class ExactOpSurface:
    case_id: str
    architecture: str
    device_id: str
    model_sha256: str
    quant: str
    operation: str
    shape: tuple[int, ...]
    dtype: str
    tensor_manifest_sha256: str
    recipe_id: str
    recipe_sha256: str
    harness_build_sha256: str
    factors: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, *, case_id: str, device_id: str, model_sha256: str,
               quant: str, operation: str, shape: Sequence[int], dtype: str,
               tensor_manifest_sha256: str, recipe_id: str, recipe_sha256: str,
               harness_build_sha256: str, factors: Mapping[str, object]) -> "ExactOpSurface":
        normalized_shape = tuple(shape)
        if not normalized_shape or any(isinstance(item, bool) or not isinstance(item, int)
                                       or item <= 0 for item in normalized_shape):
            raise C3ContractError("shape must contain positive integers")
        if not factors:
            raise C3ContractError("all evaluator factors must be explicit")
        normalized_factors = tuple(sorted((str(key), str(value))
                                          for key, value in factors.items()))
        if any(not key or not value or value.lower() == "auto"
               for key, value in normalized_factors):
            raise C3ContractError("surface factors must be named and resolved; auto is forbidden")
        return cls(
            case_id=_text(case_id, "case_id"), architecture=TARGET_ARCH,
            device_id=_text(device_id, "device_id"),
            model_sha256=_sha256(model_sha256, "model_sha256"),
            quant=_text(quant, "quant"), operation=_text(operation, "operation"),
            shape=normalized_shape, dtype=_text(dtype, "dtype"),
            tensor_manifest_sha256=_sha256(
                tensor_manifest_sha256, "tensor_manifest_sha256"),
            recipe_id=_text(recipe_id, "recipe_id"),
            recipe_sha256=_sha256(recipe_sha256, "recipe_sha256"),
            harness_build_sha256=_sha256(
                harness_build_sha256, "harness_build_sha256"),
            factors=normalized_factors,
        )


@dataclass(frozen=True)
class TimingObservation:
    provider: str
    surface: ExactOpSurface
    implementation_sha256: str
    samples_ns: tuple[float, ...]
    evidence_ref: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.provider not in VENDOR_PROVIDERS | {CANDIDATE_PROVIDER}:
            raise C3ContractError(f"unsupported timing provider {self.provider!r}")
        if not isinstance(self.surface, ExactOpSurface):
            raise TypeError("surface must be ExactOpSurface")
        _sha256(self.implementation_sha256, "implementation_sha256")
        object.__setattr__(self, "samples_ns", _positive_samples(
            self.samples_ns, "timing observation"))
        _text(self.evidence_ref, "evidence_ref")
        _sha256(self.evidence_sha256, "evidence_sha256")

    @property
    def median_ns(self) -> float:
        return _median(self.samples_ns)


@dataclass(frozen=True)
class VendorFloor:
    case: EpycOpCase
    surface: ExactOpSurface
    selected: TimingObservation
    compared: tuple[TimingObservation, ...]


def select_vendor_floor(case: EpycOpCase, surface: ExactOpSurface,
                        observations: Sequence[TimingObservation]) -> VendorFloor:
    """Select the fastest required vendor provider on one exact surface."""
    if case.case_id != surface.case_id:
        raise IdentityMismatch("case and exact surface name different tasks")
    matched = tuple(item for item in observations if item.surface == surface)
    if len(matched) != len(observations):
        raise IdentityMismatch("a vendor observation was measured on another exact surface")
    if any(item.provider == CANDIDATE_PROVIDER for item in matched):
        raise C3ContractError("a candidate cannot serve as its own vendor baseline")
    required = set(case.required_baseline_providers)
    observed = {item.provider for item in matched}
    if observed != required or len(matched) != len(required):
        raise C3ContractError(
            f"exact floor requires one observation per provider; required={sorted(required)}, "
            f"observed={sorted(observed)}")
    return VendorFloor(
        case=case, surface=surface,
        selected=min(matched, key=lambda item: item.median_ns),
        compared=tuple(sorted(matched, key=lambda item: item.provider)),
    )


@dataclass(frozen=True)
class FastPGate:
    case_id: str
    p: float
    speedup: float | None
    check: schemas.Check
    baseline_provider: str | None
    baseline_evidence_ref: str | None
    candidate_evidence_ref: str | None
    candidate_implementation_sha256: str | None

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        if (isinstance(self.p, bool) or not isinstance(self.p, (int, float))
                or not math.isfinite(self.p) or self.p < 1.0):
            raise C3ContractError("fast_p gate threshold must be finite and at least 1.0")
        if not isinstance(self.check, schemas.Check):
            raise TypeError("fast_p check must be schemas.Check")
        if self.speedup is not None and (
                isinstance(self.speedup, bool) or not isinstance(self.speedup, (int, float))
                or not math.isfinite(self.speedup) or self.speedup <= 0):
            raise C3ContractError("fast_p speedup must be positive and finite")
        if self.check.outcome == schemas.PASS:
            if self.speedup is None or self.speedup < self.p:
                raise C3ContractError("a PASS fast_p gate must carry speedup >= p")
            if self.baseline_provider not in VENDOR_PROVIDERS:
                raise C3ContractError("a PASS fast_p gate must name its vendor provider")
            _text(self.baseline_evidence_ref, "baseline_evidence_ref")
            _text(self.candidate_evidence_ref, "candidate_evidence_ref")
            _sha256(self.candidate_implementation_sha256,
                    "candidate_implementation_sha256")

    @property
    def admitted(self) -> bool:
        return self.check.outcome == schemas.PASS


def score_fast_p(*, floor: VendorFloor | None,
                 candidate: TimingObservation | None, p: float,
                 correctness: schemas.Check, integrity: schemas.Check) -> FastPGate:
    """Apply correctness/integrity first, then score candidate/vendor latency."""
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 1.0:
        raise C3ContractError("fast_p threshold must be finite and at least 1.0")
    if not isinstance(correctness, schemas.Check) or not isinstance(integrity, schemas.Check):
        raise TypeError("correctness and integrity must be schemas.Check values")
    case_id = floor.case.case_id if floor is not None else (
        candidate.surface.case_id if candidate is not None else "evidence_unavailable")
    prior = schemas.Check.worst_of((correctness, integrity))
    if prior.outcome != schemas.PASS:
        return FastPGate(case_id, float(p), None, prior, None, None, None, None)
    if floor is None or candidate is None:
        return FastPGate(
            case_id, float(p), None,
            schemas.Check(schemas.COULD_NOT_CHECK,
                          ("vendor-floor or candidate timing evidence is absent",)),
            None, None, None, None)
    if candidate.provider != CANDIDATE_PROVIDER:
        raise C3ContractError("candidate observation does not name the candidate provider")
    if candidate.surface != floor.surface:
        raise IdentityMismatch("candidate and vendor floor use different exact surfaces")
    speedup = floor.selected.median_ns / candidate.median_ns
    check = (schemas.Check(schemas.PASS) if speedup >= p else
             schemas.Check(schemas.FAIL,
                           (f"candidate speedup {speedup:.9g} is below fast_{p:g}",)))
    return FastPGate(
        floor.case.case_id, float(p), speedup, check, floor.selected.provider,
        floor.selected.evidence_ref, candidate.evidence_ref,
        candidate.implementation_sha256)


@dataclass(frozen=True)
class FastPSuiteReport:
    p: float
    fast_p: float | None
    admitted_cases: int
    scored_cases: int
    total_cases: int
    cases: tuple[FastPGate, ...]
    authority: str = SEARCH_EXIT_AUTHORITY


def aggregate_fast_p(cases: Sequence[EpycOpCase],
                     gates: Sequence[FastPGate], *, p: float) -> FastPSuiteReport:
    expected = tuple(case.case_id for case in cases)
    by_id = {gate.case_id: gate for gate in gates}
    if len(by_id) != len(gates) or set(by_id) != set(expected):
        raise C3ContractError("fast_p aggregation requires exactly one gate for every suite case")
    ordered = tuple(by_id[case_id] for case_id in expected)
    if any(gate.p != float(p) for gate in ordered):
        raise C3ContractError("fast_p gate thresholds differ inside one suite")
    admitted = sum(gate.admitted for gate in ordered)
    scored = sum(gate.check.outcome != schemas.COULD_NOT_CHECK for gate in ordered)
    fast_p = admitted / len(ordered) if scored == len(ordered) else None
    return FastPSuiteReport(
        float(p), fast_p, admitted, scored, len(ordered), ordered)


@dataclass(frozen=True)
class CapturedWorkload:
    workload_id: str
    model_sha256: str
    tensor_manifest_sha256: str
    capture_receipt_ref: str
    capture_receipt_sha256: str

    def __post_init__(self) -> None:
        _text(self.workload_id, "workload_id")
        _sha256(self.model_sha256, "model_sha256")
        _sha256(self.tensor_manifest_sha256, "tensor_manifest_sha256")
        _text(self.capture_receipt_ref, "capture_receipt_ref")
        _sha256(self.capture_receipt_sha256, "capture_receipt_sha256")


@dataclass(frozen=True)
class WholeModelSurface:
    workload: CapturedWorkload
    architecture: str
    device_id: str
    quant: str
    recipe_id: str
    recipe_sha256: str
    factors: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, *, workload: CapturedWorkload, device_id: str, quant: str,
               recipe_id: str, recipe_sha256: str,
               factors: Mapping[str, object]) -> "WholeModelSurface":
        if not isinstance(workload, CapturedWorkload):
            raise TypeError("workload must be CapturedWorkload")
        normalized = tuple(sorted((str(key), str(value)) for key, value in factors.items()))
        if not normalized or any(not key or not value or value.lower() == "auto"
                                 for key, value in normalized):
            raise C3ContractError("whole-model factors must be explicit and resolved")
        return cls(
            workload=workload, architecture=TARGET_ARCH,
            device_id=_text(device_id, "device_id"), quant=_text(quant, "quant"),
            recipe_id=_text(recipe_id, "recipe_id"),
            recipe_sha256=_sha256(recipe_sha256, "recipe_sha256"),
            factors=normalized)


@dataclass(frozen=True)
class CandidateIntegrationBinding:
    runner_id: str
    runner_revision: str
    patch_bundle_sha256: str
    candidate_source_sha256: str
    candidate_build_sha256: str
    candidate_binary_sha256: str
    receipt_ref: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.runner_id == APEX_PYTHON_OVERLAY:
            if self.runner_revision != PINNED_APEX_REVISION:
                raise C3ContractError(
                    "Apex hot-patch runner is not at the audited pinned revision")
        elif self.runner_id == EPYC_EXPERIMENTAL_BINARY:
            if not schemas.COMMIT_RE.fullmatch(self.runner_revision):
                raise C3ContractError(
                    "EPYC experimental-binary runner requires a full source commit")
        else:
            raise C3ContractError("unknown candidate integration runner")
        for name in ("patch_bundle_sha256", "candidate_source_sha256",
                     "candidate_build_sha256", "candidate_binary_sha256",
                     "receipt_sha256"):
            _sha256(getattr(self, name), name)
        _text(self.receipt_ref, "receipt_ref")


# Compatibility name for callers written against the first prospective draft.
HotPatchBinding = CandidateIntegrationBinding


@dataclass(frozen=True)
class WholeModelObservation:
    arm: str
    surface: WholeModelSurface
    build_sha256: str
    binary_sha256: str
    samples_ns: tuple[float, ...]
    evidence_ref: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in ("unpatched_anchor", "integrated_candidate"):
            raise C3ContractError("unknown whole-model arm")
        if not isinstance(self.surface, WholeModelSurface):
            raise TypeError("surface must be WholeModelSurface")
        _sha256(self.build_sha256, "build_sha256")
        _sha256(self.binary_sha256, "binary_sha256")
        object.__setattr__(self, "samples_ns", _positive_samples(
            self.samples_ns, "whole-model observation"))
        _text(self.evidence_ref, "evidence_ref")
        _sha256(self.evidence_sha256, "evidence_sha256")

    @property
    def median_ns(self) -> float:
        return _median(self.samples_ns)


@dataclass(frozen=True)
class WholeModelExitReport:
    speedup: float | None
    minimum_speedup: float
    check: schemas.Check
    authority: str = SEARCH_EXIT_AUTHORITY
    promotion_authorized: bool = False
    authority_boundary: str = NO_PROMOTION_AUTHORITY


def evaluate_whole_model_exit(*, operator_gate: FastPGate | None,
                              integration: CandidateIntegrationBinding | None,
                              anchor: WholeModelObservation | None,
                              candidate: WholeModelObservation | None,
                              correctness: schemas.Check,
                              integrity: schemas.Check,
                              minimum_speedup: float = 1.0) -> WholeModelExitReport:
    """Evaluate an integrated candidate only on an exact captured workload.

    Missing evidence returns ``COULD_NOT_CHECK``.  Contradictory identity raises:
    absence is expected before an empirical campaign, while mislabeled evidence
    is a record-integrity defect.
    """
    if (isinstance(minimum_speedup, bool) or not isinstance(minimum_speedup, (int, float))
            or not math.isfinite(minimum_speedup) or minimum_speedup < 1.0):
        raise C3ContractError("whole-model minimum speedup must be finite and at least 1.0")
    if not isinstance(correctness, schemas.Check) or not isinstance(integrity, schemas.Check):
        raise TypeError("correctness and integrity must be schemas.Check values")
    prior = schemas.Check.worst_of((correctness, integrity))
    if prior.outcome != schemas.PASS:
        return WholeModelExitReport(None, float(minimum_speedup), prior)
    if operator_gate is None or operator_gate.check.outcome != schemas.PASS:
        reason = ("the exact-surface vendor-floor operator gate has not passed",)
        if operator_gate is not None and operator_gate.check.reasons:
            reason += tuple(operator_gate.check.reasons)
        return WholeModelExitReport(
            None, float(minimum_speedup), schemas.Check(schemas.COULD_NOT_CHECK, reason))
    if integration is None or anchor is None or candidate is None:
        return WholeModelExitReport(
            None, float(minimum_speedup),
            schemas.Check(schemas.COULD_NOT_CHECK,
                          ("candidate integration and matched whole-model observations "
                           "are required",)))
    if operator_gate.candidate_implementation_sha256 != integration.candidate_source_sha256:
        raise IdentityMismatch(
            "operator gate and integration name different candidate source")
    if anchor.arm != "unpatched_anchor" or candidate.arm != "integrated_candidate":
        raise IdentityMismatch("whole-model observations are assigned to the wrong arms")
    if anchor.surface != candidate.surface:
        raise IdentityMismatch("whole-model arms use different captured workload surfaces")
    if candidate.build_sha256 != integration.candidate_build_sha256:
        raise IdentityMismatch("integrated observation names a different candidate build")
    if candidate.binary_sha256 != integration.candidate_binary_sha256:
        raise IdentityMismatch("integrated observation names a different candidate binary")
    speedup = anchor.median_ns / candidate.median_ns
    check = (schemas.Check(schemas.PASS) if speedup >= minimum_speedup else
             schemas.Check(schemas.FAIL,
                           (f"whole-model speedup {speedup:.9g} is below "
                            f"{minimum_speedup:g}",)))
    return WholeModelExitReport(speedup, float(minimum_speedup), check)


@dataclass(frozen=True)
class ExternalArtifactRequirement:
    artifact_id: str
    required_for: str
    description: str
    presence_asserted: bool = False


def external_artifact_requirements() -> tuple[ExternalArtifactRequirement, ...]:
    """Name empirical inputs this offline substrate intentionally cannot create."""
    return (
        ExternalArtifactRequirement(
            "c3_apex_case_mapping", "attention_and_moe_trace_preflight",
            "reviewed c3_apex_case_mapping.v1 binding both exact k228/k175 C5 "
            "artifacts to selected Apex registry entries and hash-bound semantic "
            "equivalence evidence; name similarity is insufficient"),
        ExternalArtifactRequirement(
            "candidate_integration_receipt", "whole_model_exit",
            f"Apex at {PINNED_APEX_REVISION} plus a hash-bound integration receipt "
            "for attention/MoE, or an immutable EPYC experimental-binary binding "
            "for dequant"),
        ExternalArtifactRequirement(
            "captured_epyc_tensor_manifests", "operator_and_whole_model_surfaces",
            "exact attention, MoE-dispatch, and Q4_K dequant workload tensor manifests"),
        ExternalArtifactRequirement(
            "mi210_vendor_and_candidate_timings", "vendor_floor_and_fast_p",
            "matched ROCm vendor and candidate observations on the physical gfx90a device"),
        ExternalArtifactRequirement(
            "matched_whole_model_rebench", "whole_model_exit",
            "unpatched-anchor and integrated-candidate receipts on the same capture"),
    )


def audit_no_execution_paths() -> schemas.Check:
    """Prove this module has no process, network, device, or write-capable call."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_imports = {"os", "subprocess", "socket", "shutil", "signal", "torch", "triton"}
    forbidden_calls = {"run", "Popen", "system", "exec", "eval", "write_text", "write_bytes",
                       "open", "unlink", "remove", "rename", "replace", "mkdir"}
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            findings.extend(name.name for name in node.names
                            if name.name.split(".")[0] in forbidden_imports)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden_imports:
                findings.append(node.module)
        elif isinstance(node, ast.Call):
            name = (node.func.id if isinstance(node.func, ast.Name) else
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if name in forbidden_calls:
                findings.append(name)
    return (schemas.Check(schemas.PASS) if not findings else
            schemas.Check(schemas.FAIL,
                          (f"execution/write-capable paths found: {sorted(set(findings))}",)))


__all__ = [
    "APEX_PYTHON_OVERLAY", "CANDIDATE_PROVIDER", "EPYC_EXPERIMENTAL_BINARY",
    "HIPBLASLT", "NO_PROMOTION_AUTHORITY",
    "PINNED_APEX_REVISION", "ROCBLAS", "SCHEMA", "SEARCH_EXIT_AUTHORITY",
    "TARGET_ARCH", "TARGET_DEVICE", "TORCH_ROCM_COMPILE", "VENDOR_PROVIDERS",
    "C3ContractError", "IdentityMismatch", "EpycOpCase", "ExactOpSurface",
    "TimingObservation", "VendorFloor", "FastPGate", "FastPSuiteReport",
    "CapturedWorkload", "WholeModelSurface", "CandidateIntegrationBinding",
    "HotPatchBinding",
    "WholeModelObservation", "WholeModelExitReport", "ExternalArtifactRequirement",
    "epyc_op_suite", "select_vendor_floor", "score_fast_p", "aggregate_fast_p",
    "evaluate_whole_model_exit", "external_artifact_requirements",
    "audit_no_execution_paths",
]
