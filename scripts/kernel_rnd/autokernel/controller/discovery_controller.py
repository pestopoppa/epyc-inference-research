#!/usr/bin/env python3
"""Candidate-only AutoKernel discovery controller.

The controller deliberately owns only the state machine.  Existing campaign
code owns source mutation, isolated worktrees, build, resource claims, source
proof, dispatch attribution, screening, cleanup, and frozen-tree proof.  This
module never accepts a command from a planner and never turns a screen into a
promotion.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import stat
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from .. import (campaign, cumulative_composition, hypothesis_portfolio, journal,
                preauthored_continuation, schemas, source_candidate)
from ..evaluator import integrity
from . import (claude_fable5_critic_actor, codex_container_actor,
               discovery_telemetry, do_not_repeat, hypotheses)
from . import gpu_source_proofs
from scripts.benchmark import autokernel_progression
from scripts.benchmark import run_autokernel_gpu_discovery as gpu_discovery

SCHEMA = "epyc.autokernel.discovery_controller.v8"
ROSTER_SCHEMA = "epyc.autokernel.discovery_roster.v3"
AUTHORITY = "nonpromotable_candidate_only_discovery"
HASH = __import__("re").compile(r"^[0-9a-f]{64}$")
PORTFOLIO_DNR_CHECK_SCHEMA = "epyc.autokernel.portfolio_exact_dnr_check.v1"
SOL = {"provider": "codex", "model": "gpt-5.6-sol", "effort": "high", "role": "planner"}
FABLE5_CRITIC = {"provider": "claude", "model": "claude-fable-5", "effort": "high", "role": "critic"}


class DiscoveryControllerError(RuntimeError): pass


class PlannerOutputRefusal(DiscoveryControllerError):
    """A safe, bounded refusal of a completed planner's authored artifacts.

    This type is deliberately narrower than ``DiscoveryControllerError``.  It
    may be raised only after the Sol process returned successfully and while
    validating files in its disposable workspace.  Runtime, authentication,
    containment, and later source/build failures must retain their native
    exception types.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # Telemetry is observational.  A telemetry failure must never replace
        # the already-derived, controller-owned planner refusal.
        self.telemetry_status = "not_attempted"
        self.telemetry_failure: dict[str, str] | None = None

    def note_telemetry_failure(self, exc: Exception) -> None:
        self.telemetry_status = "emit_failed"
        self.telemetry_failure = {
            "type": type(exc).__name__,
            "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        }


class PlannerProviderTransient(PlannerOutputRefusal):
    """A retryable provider/API interruption before candidate validation."""


class GovernedStageRefusal(DiscoveryControllerError):
    stage = ""
    disposition = ""
    scientific_budget_spent = False

    def __init__(self, message: str, *, receipt_path: str,
                 receipt_sha256: str) -> None:
        super().__init__(message)
        if (not isinstance(receipt_path, str) or not receipt_path
                or not isinstance(receipt_sha256, str)
                or not HASH.fullmatch(receipt_sha256)):
            raise DiscoveryControllerError(
                "governed stage refusal lacks a sealed receipt")
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256


class SourceApplyRefusal(GovernedStageRefusal):
    stage = "source_apply"
    disposition = "authoring_refused"


class CompileRefusal(GovernedStageRefusal):
    stage = "compile"
    disposition = "authoring_refused"


class CorrectnessRefusal(GovernedStageRefusal):
    stage = "correctness"
    disposition = "correctness_falsified"


class CumulativeCorrectnessRefusal(CorrectnessRefusal):
    """Current full-stack correctness failed after a cumulative pair build."""

    scientific_budget_spent = True

    def __init__(
            self, message: str, *, receipt_path: str, receipt_sha256: str,
            build_pair: cumulative_composition.CumulativeBuildPair,
            correctness: cumulative_composition.FullCorrectness) -> None:
        super().__init__(message, receipt_path=receipt_path,
                         receipt_sha256=receipt_sha256)
        if (not isinstance(build_pair,
                           cumulative_composition.CumulativeBuildPair)
                or not isinstance(correctness,
                                  cumulative_composition.FullCorrectness)):
            raise DiscoveryControllerError(
                "cumulative correctness refusal lacks typed evidence")
        correctness.bind_pair(build_pair)
        if correctness.passed:
            raise DiscoveryControllerError(
                "cumulative correctness refusal claims a passing suite")
        self.build_pair = build_pair
        self.correctness = correctness


class TimedOutputCorrectnessRefusal(CorrectnessRefusal):
    """A measured same-input candidate divergence; science was consumed."""

    scientific_budget_spent = True

    def __init__(self, message: str, *, receipt_path: str,
                 receipt_sha256: str, result_sha256: str,
                 operation_key: str) -> None:
        super().__init__(message, receipt_path=receipt_path,
                         receipt_sha256=receipt_sha256)
        if not isinstance(result_sha256, str) or not HASH.fullmatch(result_sha256):
            raise DiscoveryControllerError(
                "timed-output correctness refusal lacks its native result hash")
        self.result_sha256 = result_sha256
        if (not isinstance(operation_key, str)
                or not HASH.fullmatch(operation_key)):
            raise DiscoveryControllerError(
                "timed-output correctness refusal lacks its operation identity")
        self.operation_key = operation_key


class DispatchAttributionRefusal(GovernedStageRefusal):
    stage = "dispatch_attribution"
    disposition = "attribution_route_falsified"
    scientific_budget_spent = True


class CumulativeAttributionRefusal(DispatchAttributionRefusal):
    """A cumulative build passed correctness but failed route authority."""

    def __init__(
            self, message: str, *, receipt_path: str, receipt_sha256: str,
            build_pair: cumulative_composition.CumulativeBuildPair,
            correctness: cumulative_composition.FullCorrectness) -> None:
        super().__init__(message, receipt_path=receipt_path,
                         receipt_sha256=receipt_sha256)
        if (not isinstance(build_pair,
                           cumulative_composition.CumulativeBuildPair)
                or not isinstance(correctness,
                                  cumulative_composition.FullCorrectness)):
            raise DiscoveryControllerError(
                "cumulative attribution refusal lacks typed evidence")
        correctness.bind_pair(build_pair)
        if not correctness.passed:
            raise DiscoveryControllerError(
                "cumulative attribution refusal lacks passing correctness")
        self.build_pair = build_pair
        self.correctness = correctness


class MeasurementOutputRefusal(GovernedStageRefusal):
    stage = "measurement_output"
    disposition = "measurement_output_refused"


class PrecomputeScreenRefusal(DiscoveryControllerError):
    """Typed adapter refusal proving that no governed operation was started."""


class PostBuildEvidencePlanRefusal(PrecomputeScreenRefusal):
    """A completed build was refused before claim/proof/runner execution.

    The builder's exact terminal remains reusable by its sealed build key; the
    controller may durably classify this screen without treating the operation
    as an ambiguous GPU run.
    """


class ResumableScreenInterruption(DiscoveryControllerError):
    """A pre-run transport failure after durable proof was checkpointed.

    The current candidate and its scientific proof remain inflight.  The
    controller pauses without consuming an iteration so a corrected/reloaded
    runner can resume at the first incomplete stage.
    """


class ScreenInfrastructureAmbiguity(DiscoveryControllerError):
    """A sealed per-arm integrity failure requiring a fresh operation epoch."""

    def __init__(self, message: str, *, receipt_path: str,
                 receipt_sha256: str, operation_key: str) -> None:
        super().__init__(message)
        if (not isinstance(receipt_path, str) or not receipt_path
                or not isinstance(receipt_sha256, str)
                or not HASH.fullmatch(receipt_sha256)
                or not isinstance(operation_key, str)
                or not HASH.fullmatch(operation_key)):
            raise DiscoveryControllerError(
                "screen infrastructure ambiguity lacks sealed authority")
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256
        self.operation_key = operation_key


class ResourceWait(DiscoveryControllerError):
    """A durable pre-executor refusal caused only by resource contention."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str: return hashlib.sha256(_canon(value)).hexdigest()


_Q5_LDS0_ERRATUM_CARRIER = (
    Path(__file__).resolve().parents[1]
    / "q5_lds0_attribution_erratum_v1.json")
_Q5_LDS0_ERRATUM_FILE_SHA256 = (
    "22f23f769bd7e10e24d2c642846fa0b739c5ff03b457c56e374d941f01b60a98")


def _expected_q5_lds0_attribution_erratum() -> dict[str, Any]:
    """Return the one sealed predecessor attribution correction.

    v26 measured the exact preauthored Q5 candidate successfully, but its
    reviewed dispatch contract fabricated half-anchor LDS values (512/256)
    for a one-wave specialization whose static group segment is exactly zero.
    This is an infrastructure-expectation correction, not a scientific result.
    """
    body: dict[str, Any] = {
        "schema": "epyc.autokernel.attribution_expectation_erratum.v1",
        "predecessor_campaign_id": "ak-discovery-03fc1b1230487a35",
        "operation_key":
            "fdfbf8434c361a32cd07d86ac247f61c62f9f840bc3ed8b437053f089e33f837",
        "hypothesis_id": "akh-v2-q5-onewave-preauthored",
        "candidate_semantic_sha256":
            "06973eb2e4f643b76de198d6cae5e2e9f1b915773dafdf5efd08682bf0df2b63",
        "candidate_patch_sha256":
            "f4cc49cd11cdfd93a2d5d2e00e653f503b6a16ce675bfb12c034fbbfae3e7a77",
        "cross_campaign_candidate_sha256":
            "d5671a1dc197e5d0d53f34f9c4d25f640e0e410d6917b3099459bc40064581b2",
        "source_manifest_file_sha256":
            "cb93f92256a828b82a1a780bc5895f317b5ad9b8ffe8e17e3dd03d9d73474d1c",
        "correctness_receipt_file_sha256":
            "800e510a18b2aca292a032e54ec7e4279bfef921e483d2e30b1e5503b59b1a7f",
        "correctness_receipt_sha256":
            "851ce7904290c3a18cc1cf3dafb44471cde5e1058817c18674ca4bf7f1274267",
        "evidence_policy_file_sha256":
            "6aeb9d46a8721d91cd92d5ef4156ec455e7b26cc78d1a569226d1ab5eb2339c8",
        "attribution_refusal_file_sha256":
            "40707008b6fceae9749dfca56253836e07ce51b19eb7fb003377c3340503eb86",
        "attribution_refusal_receipt_sha256":
            "5e4767276dc107c638c92191789cf5d9ac6e58e96d740ae32f313e58f7bec5e3",
        "classification": "attribution_route_falsified",
        "candidate_source_commit":
            "9044b96d072c009709962092c20a3c9f1fbae4ad",
        "candidate_binary_sha256":
            "f14e72ae6c784e56917254c4f315eff33dbd1361da631a35b278f1309d4a025f",
        "candidate_hip_library_sha256":
            "dcd847b61c1f4e55c1a3a785dd07f9bc199d1ed547dc69529cf0302019cbb5a7",
        "anchor_hip_library_sha256":
            "1a8468e83dcf1fdb8e35e57cf182e029618d2af795cc9ca9cf1bfe6a60c95791",
        "profiler_trace_sha256":
            "18341bec0013da91245d279eaaa6113db443802dfbb1ebb20c919126bb69d9e5",
        "reason": (
            "exact dispatch cuda-mmvq-q5-onewave-continuation-v1.anchor.0."
            "candidate-onewave count/geometry mismatch"),
        "invalidated_predecessor_projection": {
            "turn": 1,
            "result_file_sha256":
                "40707008b6fceae9749dfca56253836e07ce51b19eb7fb003377c3340503eb86",
            "removed_effects": [
                "scientific_attempt",
                "attempted_candidate_identity",
                "portfolio_skip",
                "cross_campaign_do_not_repeat",
            ],
            "history_retained": True,
        },
        "stale_candidate_lds_bytes": {
            "cuda-mmvq-q5-onewave-continuation-v1.anchor.0.candidate-onewave": 512,
            "cuda-mmvq-q5-onewave-continuation-v1.anchor.1.candidate-onewave": 512,
            "cuda-mmvq-q5-onewave-continuation-v1.anchor.2.candidate-onewave": 512,
            "cuda-mmvq-q5-onewave-continuation-v1.anchor.3."
            "candidate-structural-excluded": 256,
        },
        "corrected_candidate_lds_bytes": {
            "cuda-mmvq-q5-onewave-continuation-v1.anchor.0.candidate-onewave": 0,
            "cuda-mmvq-q5-onewave-continuation-v1.anchor.1.candidate-onewave": 0,
            "cuda-mmvq-q5-onewave-continuation-v1.anchor.2.candidate-onewave": 0,
            "cuda-mmvq-q5-onewave-continuation-v1.anchor.3."
            "candidate-structural-excluded": 0,
        },
        "compiler_metadata_proof": {
            "schema": "epyc.autokernel.amdgpu_group_segment_proof.v2",
            "llvm_objcopy_sha256":
                "895474f91e7db238db54745673294eea93cd855d93a084ce94200103104b145b",
            "llvm_objcopy_version": (
                "llvm-objcopy, compatible with GNU objcopy\n"
                "AMD LLVM version 18.0.0git\n  Optimized build."),
            "section_extraction_command": [
                "/opt/rocm/llvm/bin/llvm-objcopy",
                "--dump-section=.hip_fatbin=<section-output>",
                "<hip-library>",
            ],
            "clang_offload_bundler_sha256":
                "4a455de48ee5c739f74e26e1979241a8b4e52ea9e57a316186c2425fae615cfe",
            "clang_offload_bundler_version": (
                "AMD clang-offload-bundler version 18.0.0git "
                "(https://github.com/RadeonOpenCompute/llvm-project "
                "roc-6.2.0 24292 "
                "26466ce804ac523b398608f17388eb6d605a3f09)"),
            "llvm_readobj_sha256":
                "3f8e3f02ef3cca007a82490eb204c08f7a714806b41e054d3faad2d2e0e95afd",
            "llvm_readobj_version": (
                "AMD LLVM version 18.0.0git\n  Optimized build."),
            "metadata_command": [
                "/opt/rocm/llvm/bin/llvm-readobj", "--notes",
                "<gfx90a-code-object>",
            ],
            "symbol_command": [
                "/opt/rocm/llvm/bin/llvm-readelf", "-sW",
                "<gfx90a-code-object>",
            ],
            "bundle_parser": {
                "format": "clang_offload_bundle_header_little_endian_v1",
                "container_count": 135,
                "selected_bundle_index": 35,
                "bundle_index_base": 0,
                "selected_target_index": 1,
                "target_index_base": 0,
                "selected_target": "hipv4-amdgcn-amd-amdhsa--gfx90a",
                "payload_offset_within_container": 4096,
                "candidate": {
                    "section_sha256":
                        "2a3f08d4af9fcbab5d1cc8a09adf409f0ae63a29a0b66a498cedc596dae5a7e1",
                    "section_size": 52221816,
                    "container_offset": 5079040,
                    "code_object_size": 1873976,
                },
                "anchor": {
                    "section_sha256":
                        "0f15fd0835b6dbf9908f5104e35f83a64423e80f690530618daca1d41763d106",
                    "section_size": 52225912,
                    "container_offset": 5079040,
                    "code_object_size": 1877048,
                },
            },
            "candidate_code_object_sha256":
                "53c63348f3e1797c6c27a82e887bb0b20649636c725fb04d85af3e2038838bd6",
            "anchor_code_object_sha256":
                "ba878a186026165135705597b1c4966c06c7af6a46a5dd99c3194dc76e7d8ab0",
            "selected_mangled_name_set": [
                (
                    "_ZL13mul_mat_vec_qIL9ggml_type6ELi1ELb0ELb1EEvPKvS2_"
                    "PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_type"
                    "IjLj3EEjjjS8_jjjS8_jjjj"
                ),
                (
                    "_ZL13mul_mat_vec_qIL9ggml_type6ELi1ELb1ELb1EEvPKvS2_"
                    "PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_type"
                    "IjLj3EEjjjS8_jjjS8_jjjj"
                ),
            ],
            "rows": [
                {
                    "mangled_name": (
                        "_ZL13mul_mat_vec_qIL9ggml_type6ELi1ELb1ELb1EEvPKvS2_"
                        "PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_type"
                        "IjLj3EEjjjS8_jjjS8_jjjj"),
                    "candidate_group_segment_fixed_size": 0,
                    "anchor_group_segment_fixed_size": 1024,
                },
                {
                    "mangled_name": (
                        "_ZL13mul_mat_vec_qIL9ggml_type6ELi1ELb0ELb1EEvPKvS2_"
                        "PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_type"
                        "IjLj3EEjjjS8_jjjS8_jjjj"),
                    "candidate_group_segment_fixed_size": 0,
                    "anchor_group_segment_fixed_size": 512,
                },
            ],
        },
        "preserved_evidence": ["source_manifest", "governed_correctness"],
        "scientific_budget_spent": False,
        "do_not_repeat": False,
        "replay_authorized": True,
        "replacement_disposition": "attribution_expectation_invalid",
        "resolution": "unresolved_retry_eligible",
    }
    body["erratum_sha256"] = _sha(body)
    return body


def _q5_lds0_attribution_erratum(
        path: Path = _Q5_LDS0_ERRATUM_CARRIER) -> dict[str, Any]:
    """Load the exact immutable Q5 attribution-expectation correction.

    The self-hash makes the payload internally coherent; the fixed file hash
    and exact expected projection prevent a coherently rewritten carrier from
    authorizing a different predecessor result or replay identity.  Deployment
    initialization copies these same bytes into the sealed bundle and binds
    that copy as an immutable input.
    """
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode)
                    or before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or before.st_mode & 0o022):
                raise DiscoveryControllerError(
                    "Q5 attribution erratum has unsafe file authority")
            handle = os.fdopen(descriptor, "rb")
            descriptor = -1
            with handle:
                raw = handle.read()
                after = os.fstat(handle.fileno())
            if ((before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns, before.st_nlink)
                    != (after.st_dev, after.st_ino, after.st_size,
                        after.st_mtime_ns, after.st_ctime_ns, after.st_nlink)):
                raise DiscoveryControllerError(
                    "Q5 attribution erratum changed while read")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if hashlib.sha256(raw).hexdigest() != _Q5_LDS0_ERRATUM_FILE_SHA256:
            raise DiscoveryControllerError(
                "Q5 attribution erratum file identity changed")

        def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result

        body = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise DiscoveryControllerError(
            "Q5 attribution erratum is not strict immutable JSON") from exc
    expected = _expected_q5_lds0_attribution_erratum()
    if (not isinstance(body, dict)
            or body != expected
            or body.get("erratum_sha256") != _sha({
                key: value for key, value in body.items()
                if key != "erratum_sha256"})
            or raw != _canon(body) + b"\n"):
        raise DiscoveryControllerError(
            "Q5 attribution erratum semantic authority changed")
    return body


def _emit_observational_telemetry(
        telemetry: discovery_telemetry.DiscoveryTelemetry | None,
        *args: Any, failure_sink: list[dict[str, str]] | None = None,
        **kwargs: Any) -> Exception | None:
    """Emit dashboard telemetry without granting it controller authority.

    The durable state machine and actor result remain primary.  Returning the
    telemetry exception lets a typed primary refusal record the visibility
    degradation without allowing it to replace that refusal.
    """
    if telemetry is None:
        return None
    try:
        telemetry.emit(*args, **kwargs)
    except Exception as exc:
        if failure_sink is not None:
            failure_sink.append({
                "event": str(args[1]) if len(args) > 1 else "unknown",
                "operation_key": str(kwargs.get("operation_key", "")),
                "error_type": type(exc).__name__,
                "error_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
            })
        return exc
    return None


def _validated_resource_wait(exc: ResourceWait, operation_key: str) -> dict[str, Any]:
    receipt = dict(exc.receipt)
    required = {
        "admitted": False,
        "phase": "pre_executor_reservation",
        "operation_key": operation_key,
        "promotion_claim": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise DiscoveryControllerError("resource wait does not bind the pre-executor operation")
    path = Path(str(receipt.get("stage_receipt_path", "")))
    expected = receipt.get("stage_receipt_sha256")
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.parent.name != "resource-waits"
            or path.parent.parent.name != operation_key
            or not isinstance(expected, str) or not HASH.fullmatch(expected)):
        raise DiscoveryControllerError("resource wait lacks its durable stage receipt")
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or before.st_nlink != 1 or before.st_mode & 0o022):
                raise DiscoveryControllerError(
                    "resource wait stage receipt has unsafe file authority")
            with os.fdopen(descriptor, "rb") as handle:
                raw = handle.read()
                after = os.fstat(handle.fileno())
            if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_nlink)
                    != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_nlink)):
                raise DiscoveryControllerError("resource wait stage receipt changed while read")
        except BaseException:
            try: os.close(descriptor)
            except OSError: pass
            raise
        if hashlib.sha256(raw).hexdigest() != expected:
            raise DiscoveryControllerError("resource wait stage receipt hash changed")
        stage = json.loads(raw.decode("utf-8", "strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiscoveryControllerError("resource wait stage receipt is unreadable") from error
    stage_required = {
        "schema": "epyc.autokernel.gpu_source_resource_wait.v1",
        "authority": AUTHORITY,
        "promotion_claim": False,
        "operation_key": operation_key,
        "gpu_executor_started": False,
        "proof_root_created": False,
        "runner_plan_created": False,
        "runner_output_created": False,
    }
    if (not isinstance(stage, Mapping)
            or any(stage.get(key) != value for key, value in stage_required.items())
            or stage.get("contention") != {
                key: value for key, value in receipt.items()
                if key not in {"stage_receipt_path", "stage_receipt_sha256"}}
            or stage.get("receipt_sha256") != _sha({
                key: value for key, value in stage.items() if key != "receipt_sha256"})):
        raise DiscoveryControllerError("resource wait stage receipt is not a sealed pre-executor proof")
    return receipt


RESOURCE_WAIT_CHECKPOINT_SCHEMA = \
    "epyc.autokernel.controller_resource_wait_checkpoint.v1"


def _require_safe_resource_wait_recovery(screener: Screener,
                                         inflight: Mapping[str, Any],
                                         wait_receipt: Mapping[str, Any]
                                         | None = None) -> "Recovery":
    reconcile = getattr(screener, "reconcile", None)
    if not callable(reconcile):
        raise DiscoveryControllerError("resource wait lacks reconciliation authority")
    recovery = reconcile(inflight)
    if not isinstance(recovery, Recovery) or recovery.status != "resource_wait":
        raise DiscoveryControllerError(
            "resource wait conflicts with current operation artifacts")
    if (wait_receipt is not None
            and dict(recovery.wait_receipt or {}) != dict(wait_receipt)):
        raise DiscoveryControllerError(
            "resource wait reconciliation mixed stage receipts")
    return recovery


def _resource_wait_pending(inflight: Mapping[str, Any],
                           wait_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically demote one exact post-build inflight operation to pending."""
    source = dict(inflight)
    operation_key = source.get("operation_key")
    if not isinstance(operation_key, str) or not HASH.fullmatch(operation_key):
        raise DiscoveryControllerError(
            "resource wait inflight operation key is malformed")
    prior_lease = source.get("lease")
    if (not isinstance(prior_lease, Mapping)
            or prior_lease.get("admitted") is not True
            or prior_lease.get("operation_key") != operation_key):
        raise DiscoveryControllerError(
            "resource wait lost its admitted inflight lease")
    wait = dict(wait_receipt)
    resume_permit = {**dict(prior_lease), **wait}
    checkpoint = {
        "schema": RESOURCE_WAIT_CHECKPOINT_SCHEMA,
        "authority": AUTHORITY,
        "promotion_claim": False,
        "operation_key": operation_key,
        "inflight": source,
        "inflight_sha256": _sha(source),
        "wait_receipt": wait,
        "wait_receipt_sha256": _sha(wait),
        "resume_permit": resume_permit,
    }
    checkpoint["checkpoint_sha256"] = _sha(checkpoint)
    row = dict(source["row"])
    row.update(status="waiting_resource", lease=wait)
    pending = {
        "row": row,
        "candidate": source["candidate"],
        "authorization": source["authorization"],
        "confirmation": bool(source.get("confirmation")),
        "parent_authorization": source.get("parent_authorization"),
        "infrastructure_retry_epoch": source.get(
            "infrastructure_retry_epoch", 0),
        "resource_wait": checkpoint,
        **_preauthored_pending_fields(row),
    }
    if isinstance(source.get("preauthored_continuation"), Mapping):
        pending["preauthored_continuation"] = dict(
            source["preauthored_continuation"])
    return pending


def _validated_resource_wait_checkpoint(
        pending: Mapping[str, Any], operation_key: str) -> Mapping[str, Any]:
    checkpoint = pending.get("resource_wait")
    required = {
        "schema", "authority", "promotion_claim", "operation_key",
        "inflight", "inflight_sha256", "wait_receipt",
        "wait_receipt_sha256", "resume_permit", "checkpoint_sha256",
    }
    if (not isinstance(checkpoint, Mapping) or set(checkpoint) != required
            or checkpoint.get("schema") != RESOURCE_WAIT_CHECKPOINT_SCHEMA
            or checkpoint.get("authority") != AUTHORITY
            or checkpoint.get("promotion_claim") is not False
            or checkpoint.get("operation_key") != operation_key
            or checkpoint.get("checkpoint_sha256") != _sha({
                key: value for key, value in checkpoint.items()
                if key != "checkpoint_sha256"})):
        raise DiscoveryControllerError(
            "pending resource-wait checkpoint is malformed")
    inflight = checkpoint.get("inflight")
    wait = checkpoint.get("wait_receipt")
    resume_permit = checkpoint.get("resume_permit")
    if (not isinstance(inflight, Mapping)
            or checkpoint.get("inflight_sha256") != _sha(inflight)
            or inflight.get("operation_key") != operation_key
            or not isinstance(wait, Mapping)
            or checkpoint.get("wait_receipt_sha256") != _sha(wait)
            or not isinstance(resume_permit, Mapping)
            or not isinstance(inflight.get("lease"), Mapping)
            or dict(resume_permit) != {
                **dict(inflight["lease"]), **dict(wait)}):
        raise DiscoveryControllerError(
            "pending resource-wait identity changed")
    validated_wait = _validated_resource_wait(
        ResourceWait("reopen durable resource wait", receipt=wait),
        operation_key)
    expected_row = dict(inflight.get("row", {}))
    expected_row.update(status="waiting_resource", lease=validated_wait)
    for key, value in (
            ("candidate", inflight.get("candidate")),
            ("authorization", inflight.get("authorization")),
            ("confirmation", bool(inflight.get("confirmation"))),
            ("parent_authorization", inflight.get("parent_authorization")),
            ("infrastructure_retry_epoch",
             inflight.get("infrastructure_retry_epoch", 0)),
            ("row", expected_row)):
        if pending.get(key) != value:
            raise DiscoveryControllerError(
                f"pending resource-wait {key} identity changed")
    return checkpoint


def _validated_prebuild_resource_wait(
        pending: Mapping[str, Any], operation_key: str) -> Mapping[str, Any]:
    """Accept only the exact claim-free lease refusal written before a build.

    A post-build wait is distinguishable by its ``pre_executor_reservation``
    phase and must carry ``resource_wait``.  Treating a partial post-build
    checkpoint as this older pending form would bypass ``lease.resume`` and
    re-enter the screen through ordinary admission.
    """
    row = pending.get("row")
    if (not isinstance(row, Mapping)
            or row.get("status") != "waiting_resource"):
        raise DiscoveryControllerError(
            "pending resource wait lacks its typed waiting row")
    permit = row.get("lease")
    common = {
        "admitted", "phase", "reason", "operation_key", "promotion_claim",
        "mode", "device_id", "inference_window_lock", "model_sha256",
        "load_admission",
    }
    if not isinstance(permit, Mapping):
        raise DiscoveryControllerError(
            "pending prebuild resource wait lacks a typed lease")
    reason = permit.get("reason")
    allowed = (common | {"foreign_kfd_pids"}
               if isinstance(reason, str) and reason.startswith("foreign_kfd_")
               else common | {"detail"})
    if (set(permit) != allowed
            or permit.get("admitted") is not False
            or permit.get("phase") != "prebuild_probe"
            or permit.get("operation_key") != operation_key
            or permit.get("promotion_claim") is not False
            or permit.get("mode") not in {"cold_overlap", "cold_serialized"}
            or not isinstance(permit.get("device_id"), str)
            or not permit["device_id"]
            or not isinstance(permit.get("inference_window_lock"), str)
            or not permit["inference_window_lock"]
            or not isinstance(permit.get("model_sha256"), str)
            or not HASH.fullmatch(permit["model_sha256"])
            or not isinstance(permit.get("load_admission"), Mapping)
            or not permit["load_admission"]
            or reason not in {
                "device_busy", "foreign_kfd_busy",
                "foreign_kfd_inventory_invalid",
                "foreign_kfd_inventory_unreadable",
            }
            or any(key in permit for key in (
                "stage_receipt_path", "stage_receipt_sha256"))):
        raise DiscoveryControllerError(
            "pending prebuild resource wait is malformed")
    if isinstance(reason, str) and reason.startswith("foreign_kfd_"):
        pids = permit.get("foreign_kfd_pids")
        if (not isinstance(pids, list)
                or pids != sorted(set(pids))
                or any(isinstance(pid, bool) or not isinstance(pid, int)
                       or pid <= 0 for pid in pids)
                or reason == "foreign_kfd_busy" and not pids
                or reason != "foreign_kfd_busy" and pids):
            raise DiscoveryControllerError(
                "pending prebuild KFD inventory is malformed")
    elif not isinstance(permit.get("detail"), str):
        raise DiscoveryControllerError(
            "pending prebuild device contention detail is malformed")
    return permit


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, indent=2) + "\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def sealed_roster() -> dict[str, Any]:
    return {"schema": ROSTER_SCHEMA, "members": [SOL, FABLE5_CRITIC], "claude_members": 1, "member_count": 2}


def _require_roster(value: Mapping[str, Any]) -> None:
    if dict(value) != sealed_roster(): raise DiscoveryControllerError("runtime roster is not exact Sol planner + Claude Fable 5 critic")

def _require_runtime(value: Mapping[str, Any]) -> None:
    required={"kind","docker_path","docker_sha256","image_id","codex_native_sha256","code_mode_host_sha256","ca_certificate_sha256","writable_host_binds","host_network_mode"}
    if set(value) != required or value.get("kind")!="docker_workspace_bind_only" or value.get("host_network_mode")!="docker_bridge" or value.get("writable_host_binds") != ["/workspace"] or not all(isinstance(value.get(k),str) and value[k] for k in required-{"writable_host_binds"}): raise DiscoveryControllerError("Codex runtime attestation is incomplete or unsealed")


def _require_claude_runtime(value: Mapping[str, Any]) -> None:
    """Require a non-secret, byte-bound Fable 5 CLI runtime receipt."""
    required = {"kind", "provider", "model", "effort", "wrapper_path",
                "wrapper_sha256", "argv_policy_sha256", "auth_staging_policy"}
    if (set(value) != required
            or value.get("kind") != "claude_cli_structured_critic"
            or value.get("provider") != FABLE5_CRITIC["provider"]
            or value.get("model") != FABLE5_CRITIC["model"]
            or value.get("effort") != FABLE5_CRITIC["effort"]
            or value.get("auth_staging_policy")
            != claude_fable5_critic_actor.AUTH_STAGING_POLICY
            or not all(isinstance(value.get(key), str) and value[key]
                       for key in required - {"auth_staging_policy"})):
        raise DiscoveryControllerError("Claude critic runtime attestation is incomplete or unsealed")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value: raise DiscoveryControllerError(f"{label} must be non-empty text")
    return value.strip()


@dataclass(frozen=True)
class AuthoringAssignment:
    """Controller-owned identity tuple; the actor may fill content, not authority."""
    campaign_id: str
    proposal_id: str
    candidate_id: str
    production_base_commit: str
    instrument_commit: str
    portfolio_binding: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (not self.campaign_id.startswith("ak-") or not self.proposal_id.startswith("akp-")
                or not self.candidate_id.startswith("akc-")
                or not all(isinstance(value, str) and len(value) == 40
                           and all(ch in "0123456789abcdef" for ch in value)
                           for value in (self.production_base_commit, self.instrument_commit))):
            raise DiscoveryControllerError("invalid controller-owned authoring identity")
        if self.portfolio_binding is not None:
            required = {"portfolio_sha256", "record_sha256", "hypothesis_id",
                        "statement", "falsifier", "mechanism_id", "regime",
                        "target_files", "target_symbols", "target_symbols_by_file",
                        "template_id",
                        "change_class", "decision_policy", "expected_dispatch"}
            value = self.portfolio_binding
            if (not isinstance(value, Mapping) or set(value) != required
                    or not HASH.fullmatch(str(value.get("portfolio_sha256")))
                    or not HASH.fullmatch(str(value.get("record_sha256")))
                    or not all(isinstance(value.get(key), str) and value[key]
                               for key in ("hypothesis_id", "statement", "falsifier",
                                           "mechanism_id", "template_id",
                                           "change_class"))
                    or not isinstance(value.get("regime"), Mapping)
                    or not isinstance(value.get("target_files"), (list, tuple))
                    or not 1 <= len(value["target_files"]) <= 2
                    or not all(isinstance(item, str) and item
                               for item in value["target_files"])
                    or not isinstance(value.get("target_symbols"), (list, tuple))
                    or not value["target_symbols"]
                    or not all(isinstance(item, str) and item
                               for item in value["target_symbols"])
                    or not isinstance(value.get("target_symbols_by_file"), Mapping)
                    or set(value["target_symbols_by_file"]) != set(value["target_files"])
                    or any(not isinstance(symbols, (list, tuple)) or not symbols
                           or not all(isinstance(item, str) and item for item in symbols)
                           for symbols in value["target_symbols_by_file"].values())
                    or set().union(*(set(symbols) for symbols in
                                     value["target_symbols_by_file"].values())) !=
                       set(value["target_symbols"])
                    or not isinstance(value.get("decision_policy"), Mapping)
                    or not isinstance(value.get("expected_dispatch"), (list, tuple))
                    or not 1 <= len(value["expected_dispatch"]) <= 8
                    or not all(isinstance(row, Mapping)
                               and set(row) == {"route_id", "kernel_name", "calls", "grid",
                                               "workgroup", "lds_bytes"}
                               for row in value["expected_dispatch"])):
                raise DiscoveryControllerError("invalid controller-owned portfolio binding")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDispatchExpectation:
    """Planner-authored literal geometry; never a regex, argv, or command."""
    route_id: str
    kernel_name: str
    calls: int
    grid: int
    workgroup: int
    lds_bytes: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.anchor\.[0-9]+", self.route_id):
            raise DiscoveryControllerError("dispatch route id is not deployed authority")
        # The reviewed profilers report complete demangled HIP symbols: v1 adds
        # ``[clone .kd]`` while v3 emits the native undecorated name.  This is
        # still a literal: the deployment factory escapes it before constructing
        # an evidence matcher.  Punctuation is admitted only on function-shaped
        # names; bare regex-like planner strings remain invalid.
        if (not isinstance(self.kernel_name, str)
                or not 1 <= len(self.kernel_name.encode("utf-8")) <= 2048
                or any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in self.kernel_name)
                or (any(ch in self.kernel_name for ch in "*?[]|+\\^$")
                    and "(" not in self.kernel_name)
                or (" " in self.kernel_name and "(" not in self.kernel_name)):
            raise DiscoveryControllerError("dispatch kernel name must be a bounded literal")
        for label, value, maximum in (("calls", self.calls, 10_000_000), ("grid", self.grid, 1 << 31),
                                      ("workgroup", self.workgroup, 4096), ("lds_bytes", self.lds_bytes, 1 << 30)):
            if (isinstance(value, bool) or not isinstance(value, int) or value < 0
                    or value > maximum or (label != "lds_bytes" and value == 0)):
                raise DiscoveryControllerError(f"dispatch {label} is outside reviewed literal bounds")


@dataclass(frozen=True)
class LoadModeRecommendation:
    mode: str
    rationale: str
    example_ids: tuple[str, ...]
    def __post_init__(self) -> None:
        identifier = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
        if (self.mode not in {"cold_overlap", "cold_serialized", "hot_resident"}
                or not isinstance(self.rationale, str) or not self.rationale.strip()
                or len(self.rationale) > 1024 or not isinstance(self.example_ids, tuple)
                or len(self.example_ids) > 8
                or len(set(self.example_ids)) != len(self.example_ids)
                or any(not isinstance(item, str) or not identifier.fullmatch(item)
                       for item in self.example_ids)):
            raise DiscoveryControllerError("load-mode recommendation is malformed or unbounded")
        object.__setattr__(self, "rationale", self.rationale.strip())


@dataclass(frozen=True)
class GpuSourceExperimentIntent:
    """Actor-selected *registry IDs*, never actor-selected commands or regexes."""
    template_id: str
    target_surface: str
    target_symbol: str
    correctness_id: str
    dispatch_id: str
    expected_dispatch: tuple[BoundedDispatchExpectation, ...]
    load_mode_recommendation: LoadModeRecommendation | None = None

    def __post_init__(self) -> None:
        import re
        identifier = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
        for label, value in (("template_id", self.template_id),
                             ("correctness_id", self.correctness_id),
                             ("dispatch_id", self.dispatch_id)):
            if not isinstance(value, str) or not identifier.fullmatch(value):
                raise DiscoveryControllerError(f"experiment intent {label} is not a registry id")
        for label, value in (("target_surface", self.target_surface),
                             ("target_symbol", self.target_symbol)):
            _text(value, f"experiment intent {label}")
        if (not isinstance(self.expected_dispatch, tuple)
                or not 1 <= len(self.expected_dispatch) <= 8
                or not all(isinstance(item, BoundedDispatchExpectation)
                           for item in self.expected_dispatch)
                or len({(item.route_id, item.kernel_name, item.grid,
                         item.workgroup, item.lds_bytes)
                        for item in self.expected_dispatch}) != len(self.expected_dispatch)):
            raise DiscoveryControllerError(
                "experiment intent requires 1..8 distinct bounded literal dispatch expectations")
        if self.load_mode_recommendation is not None and not isinstance(
                self.load_mode_recommendation, LoadModeRecommendation):
            raise DiscoveryControllerError("load-mode recommendation must be typed and immutable")


@dataclass(frozen=True)
class PlannedCandidate:
    hypothesis_id: str
    statement: str
    falsifier: str
    regime: Mapping[str, Any]
    proposal: Mapping[str, Any]
    source_manifest: source_candidate.SourcePatchManifest
    source_manifest_sha256: str
    experiment_intent: GpuSourceExperimentIntent | None = None
    composition_plan: cumulative_composition.CompositionPlan | None = None

    def __post_init__(self) -> None:
        _text(self.hypothesis_id, "hypothesis_id"); _text(self.statement, "statement"); _text(self.falsifier, "falsifier")
        if not self.hypothesis_id.startswith("akh-"): raise DiscoveryControllerError("hypothesis_id must start akh-")
        if not isinstance(self.regime, Mapping) or not isinstance(self.proposal, Mapping): raise DiscoveryControllerError("candidate regime and proposal must be mappings")
        if not isinstance(self.source_manifest, source_candidate.SourcePatchManifest): raise DiscoveryControllerError("candidate requires typed SourcePatchManifest")
        if not HASH.fullmatch(self.source_manifest_sha256): raise DiscoveryControllerError("source manifest hash is required")
        # Planner-owned effect fields are structurally impossible.
        if any("effect" in str(key).lower() or "result" in str(key).lower() for key in self.proposal):
            raise DiscoveryControllerError("planner proposal may not carry measured result fields")
        if (self.composition_plan is not None
                and not isinstance(
                    self.composition_plan,
                    cumulative_composition.CompositionPlan)):
            raise DiscoveryControllerError(
                "candidate cumulative composition plan must be typed")
        if (self.composition_plan is not None
                and self.composition_plan.candidate.accepted[-1].manifest !=
                    self.source_manifest):
            raise DiscoveryControllerError(
                "candidate source manifest differs from cumulative new lever")


@dataclass(frozen=True)
class Critique:
    decision: str
    reason: str
    def __post_init__(self) -> None:
        if self.decision not in {"accept", "reject", "revise"}: raise DiscoveryControllerError("critic decision must be accept, reject, or revise")
        _text(self.reason, "critic reason")


@dataclass(frozen=True)
class SealedScreen:
    receipt_path: str
    result_sha256: str
    effect_fraction: float
    classification: str
    baseline_sha256: str
    source_proof_sha256: str
    dispatch_proof_sha256: str
    exact_attribution_effect_fraction: float | None = None
    target_runtime_effect_fraction: float | None = None
    candidate_only: bool = True
    promotion_claim: bool = False
    stages: tuple[str, ...] = ("materialized", "built", "correctness", "attribution", "screen")
    # A series is one exact patch measured in one exact frame/baseline.  It is
    # deliberately not a hypothesis id: one scientific question can produce
    # several mutually independent source patches.
    series_key: str | None = None
    component_series_keys: tuple[str, ...] = ()
    # Pooled only by the controller after exact-series verification.  Adapter
    # receipts report their individual measured effect; they cannot nominate.
    series_effect_fraction: float | None = None
    build_identity_sha256: str | None = None
    correctness_receipt_sha256: str | None = None
    attribution_receipt_sha256: str | None = None
    graphs_off_receipt_sha256: str | None = None
    graphs_on_receipt_sha256: str | None = None
    composition_build_pair: cumulative_composition.CumulativeBuildPair | None = None
    composition_correctness: cumulative_composition.FullCorrectness | None = None
    composition_comparison: cumulative_composition.IncrementalComparison | None = None
    cumulative_performance: cumulative_composition.CumulativePerformance | None = None
    cumulative_performance_ref: cumulative_composition.CumulativePerformanceRef | None = None

    def __post_init__(self) -> None:
        if self.classification not in {"candidate", "screened_out", "inconclusive", "failed", "top_k_replicated_candidate", "replicated_but_subadditive"}: raise DiscoveryControllerError("unknown screen class")
        if (isinstance(self.effect_fraction, bool)
                or not isinstance(self.effect_fraction, (int, float))
                or not math.isfinite(float(self.effect_fraction))):
            raise DiscoveryControllerError("screen effect must be a finite measured number")
        for label, value in (
                ("exact attribution", self.exact_attribution_effect_fraction),
                ("target runtime", self.target_runtime_effect_fraction)):
            if (value is not None and (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)))):
                raise DiscoveryControllerError(
                    f"{label} effect must be a finite measured number")
        if (self.target_runtime_effect_fraction is not None
                and float(self.effect_fraction)
                != float(self.target_runtime_effect_fraction)):
            raise DiscoveryControllerError(
                "primary screen effect must be the target-runtime effect")
        receipts = (
            self.build_identity_sha256, self.correctness_receipt_sha256,
            self.attribution_receipt_sha256,
            self.graphs_off_receipt_sha256, self.graphs_on_receipt_sha256)
        if any(value is not None for value in receipts):
            if not all(isinstance(value, str) and HASH.fullmatch(value)
                       for value in receipts):
                raise DiscoveryControllerError(
                    "screen replication evidence is incomplete")
        composition = (
            self.composition_build_pair, self.composition_correctness,
            self.composition_comparison, self.cumulative_performance,
            self.cumulative_performance_ref)
        if any(value is not None for value in composition):
            if (not isinstance(self.composition_build_pair,
                               cumulative_composition.CumulativeBuildPair)
                    or not isinstance(self.composition_correctness,
                                      cumulative_composition.FullCorrectness)
                    or not isinstance(self.composition_comparison,
                                      cumulative_composition.IncrementalComparison)
                    or not isinstance(self.cumulative_performance,
                                      cumulative_composition.CumulativePerformance)
                    or not isinstance(self.cumulative_performance_ref,
                                      cumulative_composition.CumulativePerformanceRef)):
                raise DiscoveryControllerError(
                    "screen cumulative evidence is incomplete")
            self.composition_correctness.bind_pair(
                self.composition_build_pair)
            if (self.composition_comparison.operation_key !=
                    self.composition_build_pair.operation_key
                    or self.composition_comparison.build_pair_sha256 !=
                       self.composition_build_pair.pair_sha256
                    or self.composition_comparison.correctness_result_sha256 !=
                       self.composition_correctness.result_sha256):
                raise DiscoveryControllerError(
                    "screen cumulative evidence bindings changed")
            plan_sha256 = self.composition_build_pair.plan_sha256
            if self.cumulative_performance.plan_sha256 != plan_sha256:
                raise DiscoveryControllerError(
                    "screen cumulative performance names another plan")
        if not self.candidate_only or self.promotion_claim:
            raise DiscoveryControllerError(
                "discovery screen must remain nonpromotable")
        if tuple(self.stages) not in {
                ("materialized", "built", "correctness", "attribution", "screen"),
                ("materialized", "built", "correctness", "attribution",
                 "measurement_graphs_off_screen",
                 "target_runtime_graphs_on_screen"),
                ("materialized", "built", "correctness", "attribution")}:
            raise DiscoveryControllerError(
                "screen did not prove the required fail-closed stage order")
        for value in (self.result_sha256, self.baseline_sha256,
                      self.source_proof_sha256, self.dispatch_proof_sha256):
            if not HASH.fullmatch(value):
                raise DiscoveryControllerError(
                    "sealed result requires evidence hashes")
        if self.series_key is not None and not HASH.fullmatch(self.series_key):
            raise DiscoveryControllerError(
                "screen series key must be a sealed hash")
        # JSON recovery naturally turns a tuple into a list; normalize it at
        # the durable boundary, then keep the in-memory receipt immutable.
        if isinstance(self.component_series_keys, list):
            object.__setattr__(self, "component_series_keys",
                               tuple(self.component_series_keys))
        if (not isinstance(self.component_series_keys, tuple)
                or not all(HASH.fullmatch(value)
                           for value in self.component_series_keys)):
            raise DiscoveryControllerError(
                "component series provenance must be sealed hashes")
        if (self.series_effect_fraction is not None
                and (isinstance(self.series_effect_fraction, bool)
                     or not isinstance(self.series_effect_fraction, (int, float))
                     or not math.isfinite(float(self.series_effect_fraction)))):
            raise DiscoveryControllerError(
                "pooled series effect must be finite")


def _sealed_screen_from_dict(value: Mapping[str, Any]) -> SealedScreen:
    if not isinstance(value, Mapping):
        raise DiscoveryControllerError("sealed screen checkpoint is malformed")
    body = dict(value)
    body["stages"] = tuple(body.get("stages", ()))
    body["component_series_keys"] = tuple(
        body.get("component_series_keys", ()))
    constructors = (
        ("composition_build_pair",
         cumulative_composition.CumulativeBuildPair.from_dict),
        ("composition_correctness",
         cumulative_composition.FullCorrectness.from_dict),
        ("composition_comparison",
         cumulative_composition.IncrementalComparison.from_dict),
        ("cumulative_performance",
         cumulative_composition.CumulativePerformance.from_dict),
        ("cumulative_performance_ref",
         cumulative_composition.CumulativePerformanceRef.from_dict),
    )
    try:
        for key, constructor in constructors:
            if body.get(key) is not None:
                body[key] = constructor(body[key])
        return SealedScreen(**body)
    except (TypeError, cumulative_composition.CompositionError) as exc:
        raise DiscoveryControllerError(
            "sealed screen checkpoint evidence is invalid") from exc


class Planner(Protocol):
    def attest(self) -> Mapping[str, Any]: ...
    def plan(self, *, context: Mapping[str, Any], workspace: Path,
             checkpoint_path: Path | None = None) -> PlannedCandidate: ...
    def resume_plan(self, *, context: Mapping[str, Any],
                    workspace: Path, checkpoint_path: Path) -> PlannedCandidate: ...

class Critic(Protocol):
    def attest(self) -> Mapping[str, Any]: ...
    def review(self, candidate: PlannedCandidate, *, context: Mapping[str, Any], workspace: Path) -> Critique: ...

class Lease(Protocol):
    def admit(self, candidate: PlannedCandidate, *, operation_key: str) -> Mapping[str, Any]: ...
    def resume(self, candidate: PlannedCandidate,
               stale_permit: Mapping[str, Any]) -> Mapping[str, Any]: ...

class Screener(Protocol):
    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen: ...
    def reconcile(self, inflight: Mapping[str, Any]) -> "Recovery": ...

@dataclass(frozen=True)
class Recovery:
    status: str
    result: SealedScreen | None = None
    wait_receipt: Mapping[str, Any] | None = None
    def __post_init__(self) -> None:
        if self.status not in {"safe_to_start", "resource_wait", "sealed_result", "ambiguous"}: raise DiscoveryControllerError("unknown recovery status")
        if (self.status == "sealed_result") != isinstance(self.result, SealedScreen): raise DiscoveryControllerError("recovery result binding is invalid")
        if (self.status == "resource_wait") != isinstance(self.wait_receipt, Mapping): raise DiscoveryControllerError("recovery wait binding is invalid")
        if self.wait_receipt is not None:
            object.__setattr__(self, "wait_receipt", dict(self.wait_receipt))


@dataclass(frozen=True)
class ReviewedSourceFile:
    relative_path: str
    sha256: str
    content: bytes

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (path.is_absolute() or path.as_posix() != self.relative_path
                or any(part in {"", ".", ".."} for part in path.parts)
                or not HASH.fullmatch(self.sha256)
                or hashlib.sha256(self.content).hexdigest() != self.sha256):
            raise DiscoveryControllerError("reviewed source file identity is malformed")


@dataclass(frozen=True)
class ReviewedSourcePackage:
    instrument_commit: str
    files: tuple[ReviewedSourceFile, ...]
    package_sha256: str

    def __post_init__(self) -> None:
        if (not re.fullmatch(r"[0-9a-f]{40}", self.instrument_commit)
                or not self.files
                or tuple(sorted(item.relative_path for item in self.files)) != tuple(
                    item.relative_path for item in self.files)
                or len({item.relative_path for item in self.files}) != len(self.files)):
            raise DiscoveryControllerError("reviewed source package is malformed")
        body = {"schema": "epyc.autokernel.reviewed_source_package.v1",
                "instrument_commit": self.instrument_commit,
                "files": [{"relative_path": item.relative_path, "sha256": item.sha256,
                           "workspace_path": f"reviewed-source/{item.relative_path}"}
                          for item in self.files]}
        if self.package_sha256 != _sha(body):
            raise DiscoveryControllerError("reviewed source package hash mismatch")

    def manifest(self) -> dict[str, Any]:
        body = {"schema": "epyc.autokernel.reviewed_source_package.v1",
                "instrument_commit": self.instrument_commit,
                "files": [{"relative_path": item.relative_path, "sha256": item.sha256,
                           "workspace_path": f"reviewed-source/{item.relative_path}"}
                          for item in self.files]}
        return {**body, "package_sha256": self.package_sha256}

    def _manifest_bytes(self) -> bytes:
        return json.dumps(self.manifest(), sort_keys=True, indent=2).encode() + b"\n"

    @staticmethod
    def _require_owned_directory(path: Path, label: str) -> None:
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or path.is_symlink()
                or info.st_uid != os.getuid() or info.st_nlink < 2):
            raise DiscoveryControllerError(f"{label} is not an owned non-symlink directory")

    def revalidate_materialized(self, workspace: Path) -> None:
        self._require_owned_directory(workspace, "actor workspace")
        root = workspace / "reviewed-source"
        self._require_owned_directory(root, "reviewed source root")
        for item in self.files:
            target = root.joinpath(*PurePosixPath(item.relative_path).parts)
            current = root
            for part in PurePosixPath(item.relative_path).parts[:-1]:
                current = current / part
                self._require_owned_directory(current, "reviewed source parent")
            info = target.lstat()
            if (not stat.S_ISREG(info.st_mode) or target.is_symlink()
                    or info.st_uid != os.getuid() or info.st_nlink != 1
                    or hashlib.sha256(target.read_bytes()).hexdigest() != item.sha256):
                raise DiscoveryControllerError("reviewed source bytes changed in actor workspace")
        manifest_path = root / "source-package.json"
        info = manifest_path.lstat()
        if (not stat.S_ISREG(info.st_mode) or manifest_path.is_symlink()
                or info.st_uid != os.getuid() or info.st_nlink != 1
                or manifest_path.read_bytes() != self._manifest_bytes()):
            raise DiscoveryControllerError("reviewed source package manifest changed")

    def materialize(self, workspace: Path) -> Mapping[str, Any]:
        self._require_owned_directory(workspace, "actor workspace")
        root = workspace / "reviewed-source"
        if root.exists() or root.is_symlink():
            raise DiscoveryControllerError("disposable reviewed-source root already exists")
        root.mkdir(mode=0o700)
        for item in self.files:
            target = root.joinpath(*PurePosixPath(item.relative_path).parts)
            target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(item.content); handle.flush(); os.fsync(handle.fileno())
            target.chmod(0o400)
            if (target.is_symlink() or target.stat().st_nlink != 1
                    or hashlib.sha256(target.read_bytes()).hexdigest() != item.sha256):
                raise DiscoveryControllerError("reviewed source materialization changed bytes")
        manifest = self.manifest()
        encoded = self._manifest_bytes()
        manifest_path = root / "source-package.json"
        with manifest_path.open("xb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        manifest_path.chmod(0o400)
        self.revalidate_materialized(workspace)
        return manifest

    def critic_context(self, relative_path: str,
                       symbols: Sequence[str]) -> Mapping[str, Any]:
        matches = [item for item in self.files if item.relative_path == relative_path]
        if len(matches) != 1 or not symbols:
            raise DiscoveryControllerError("critic source preimage is outside reviewed package")
        item = matches[0]
        try:
            lines = item.content.decode("utf-8", "strict").splitlines(keepends=True)
        except UnicodeDecodeError as exc:
            raise DiscoveryControllerError("critic source preimage is not UTF-8") from exc
        ranges: list[tuple[int, int]] = []
        for symbol in symbols:
            indexes = [index for index, line in enumerate(lines) if symbol in line]
            if not indexes:
                raise DiscoveryControllerError(
                    f"critic source preimage lacks selected symbol: {symbol}")
            index = indexes[0]
            ranges.append((max(0, index - 24), min(len(lines), index + 25)))
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        excerpts = []
        total = 0
        for start, end in merged:
            text = "".join(lines[start:end])
            total += len(text.encode("utf-8"))
            excerpts.append({"line_start": start + 1, "line_end": end,
                             "text": text,
                             "sha256": hashlib.sha256(text.encode()).hexdigest()})
        if total > 65536:
            raise DiscoveryControllerError("critic source preimage excerpt exceeds bound")
        value = {"schema": "epyc.autokernel.critic_source_preimage.v1",
                 "relative_path": relative_path, "source_sha256": item.sha256,
                 "symbols": list(symbols), "excerpts": excerpts}
        return {**value, "context_sha256": _sha(value)}


PLANNER_ACTOR_CHECKPOINT_SCHEMA = "epyc.autokernel.planner_actor_checkpoint.v1"


def _planner_artifact_manifest(workspace: Path) -> dict[str, Any]:
    """Hash every actor-owned artifact outside the immutable source package."""
    root_info = workspace.lstat()
    if (workspace.is_symlink() or not stat.S_ISDIR(root_info.st_mode)
            or root_info.st_uid != os.getuid()):
        raise DiscoveryControllerError("planner workspace is not an owned directory")
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    total = 0
    for path in sorted(workspace.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(workspace)
        if relative.parts[0] == "reviewed-source":
            continue
        info = path.lstat()
        if path.is_symlink() or info.st_uid != os.getuid():
            raise DiscoveryControllerError(
                "planner artifact tree contains a symlink or foreign owner")
        if stat.S_ISDIR(info.st_mode):
            directories.append(relative.as_posix())
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise DiscoveryControllerError(
                "planner artifact tree contains a special file or hardlink")
        raw = path.read_bytes()
        total += len(raw)
        files.append({"path": relative.as_posix(), "size": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()})
    if len(files) > 32 or len(directories) > 32 or total > 2 * 1024 * 1024:
        raise DiscoveryControllerError("planner artifact tree exceeds its sealed bound")
    return {"directories": directories, "files": files,
            "total_bytes": total}


def _seal_planner_actor_checkpoint(workspace: Path, checkpoint_path: Path, *,
                                   context: Mapping[str, Any],
                                   result: Mapping[str, Any]) -> Mapping[str, Any]:
    path = checkpoint_path
    if path.parent != workspace.parent or path.name != "actor-result.json":
        raise DiscoveryControllerError(
            "planner actor checkpoint is outside its controller operation")
    ReviewedSourcePackage._require_owned_directory(
        path.parent, "planner operation root")
    if path.exists() or path.is_symlink():
        raise DiscoveryControllerError("planner actor checkpoint already exists")
    body = {
        "schema": PLANNER_ACTOR_CHECKPOINT_SCHEMA,
        "context_sha256": _sha(context),
        "assignment_sha256": _sha(context.get("authoring_assignment")),
        "result": dict(result),
        "artifacts": _planner_artifact_manifest(workspace),
    }
    body["receipt_sha256"] = _sha(body)
    _atomic(path, body)
    return body


def _reopen_planner_actor_checkpoint(workspace: Path, checkpoint_path: Path, *,
                                     context: Mapping[str, Any]) -> Mapping[str, Any]:
    path = checkpoint_path
    if path.parent != workspace.parent or path.name != "actor-result.json":
        raise DiscoveryControllerError(
            "planner actor checkpoint is outside its controller operation")
    ReviewedSourcePackage._require_owned_directory(
        path.parent, "planner operation root")
    if not path.exists():
        raise PlannerOutputRefusal(
            "planner invocation stopped without a completed actor artifact checkpoint")
    info = path.lstat()
    if (path.is_symlink() or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid() or info.st_nlink != 1):
        raise DiscoveryControllerError("planner actor checkpoint file is unsafe")
    checkpoint = _read_object(path, workspace.parent)
    declared = checkpoint.get("receipt_sha256")
    if (checkpoint.get("schema") != PLANNER_ACTOR_CHECKPOINT_SCHEMA
            or not isinstance(declared, str)
            or declared != _sha({key: value for key, value in checkpoint.items()
                                 if key != "receipt_sha256"})
            or checkpoint.get("context_sha256") != _sha(context)
            or checkpoint.get("assignment_sha256") !=
               _sha(context.get("authoring_assignment"))
            or checkpoint.get("artifacts") !=
               _planner_artifact_manifest(workspace)
            or not isinstance(checkpoint.get("result"), Mapping)
            or isinstance(checkpoint["result"].get("returncode"), bool)
            or not isinstance(checkpoint["result"].get("returncode"), int)):
        raise DiscoveryControllerError("planner actor checkpoint identity changed")
    return checkpoint


class CodexPlanner:
    """Concrete Sol actor. It may write only a plan and patch manifest in workspace."""
    def __init__(self, *, wrapper: Path, environment: Mapping[str, str],
                 template_catalog: Mapping[str, Any] | None = None,
                 reviewed_sources: ReviewedSourcePackage | None = None,
                 wrapper_sha256: str | None = None,
                 runtime_identity: Mapping[str, Any] | None = None,
                 actor_launcher_sha256: str | None = None,
                 telemetry: discovery_telemetry.DiscoveryTelemetry | None = None) -> None:
        self.wrapper, self.environment = wrapper, dict(environment)
        self.template_catalog = json.loads(json.dumps(template_catalog or {}, sort_keys=True))
        self.reviewed_sources = reviewed_sources
        self.wrapper_sha256 = wrapper_sha256
        self.runtime_identity = None if runtime_identity is None else dict(runtime_identity)
        self.actor_launcher_sha256 = actor_launcher_sha256
        self.telemetry = telemetry
        self.telemetry_failures: list[dict[str, str]] = []
    def _runtime(self) -> Mapping[str, Any]:
        if self.wrapper_sha256 is not None:
            if self.wrapper.is_symlink() or not self.wrapper.is_file() or hashlib.sha256(self.wrapper.read_bytes()).hexdigest() != self.wrapper_sha256:
                raise DiscoveryControllerError("sealed Codex planner wrapper bytes changed")
        current = codex_container_actor.runtime_identity(self.wrapper)
        if self.runtime_identity is not None and current != self.runtime_identity:
            raise DiscoveryControllerError("sealed Codex planner runtime identity changed")
        if (self.actor_launcher_sha256 is not None
                and hashlib.sha256(Path(codex_container_actor.__file__).resolve().read_bytes()).hexdigest()
                != self.actor_launcher_sha256):
            raise DiscoveryControllerError("sealed Codex planner launcher/argv policy changed")
        return current
    def attest(self) -> Mapping[str, Any]: return {**SOL, "runtime": self._runtime()}

    def _planner_catalog(self) -> dict[str, Any]:
        """Return the actor-visible projection of reviewed template authority.

        Candidate dispatch topology is controller-owned evidence authority.  It
        is useful to the critic and evidence reducer, but exposing its literal
        route/count/geometry cells to the planner invites those cells to be
        copied into ``expected_dispatch`` even though that field is deliberately
        the deployed anchor observation.  Preserve only the named authoring
        strategy and make the ownership boundary executable in the prompt.
        """
        catalog = json.loads(json.dumps(self.template_catalog, sort_keys=True))
        for template in catalog.values():
            if not isinstance(template, dict):
                continue
            semantics = template.get("semantics")
            if not isinstance(semantics, dict):
                continue
            variants = semantics.pop("candidate_dispatch_variants", None)
            if variants is None:
                continue
            if (template.get("template_id") != "cuda-fattn-gqa7-common-v1"
                    or not isinstance(variants, dict)
                    or set(variants) != {"gqa7_bulk_pairs", "gqa7_scalar_tail"}):
                raise DiscoveryControllerError(
                    "planner template candidate strategy authority is malformed")
            semantics["candidate_dispatch_strategy"] = {
                "strategy_id": "gqa7_pair_tail",
                "selection_authority": "controller_owned",
                "expected_dispatch_source":
                    "controller_owned_portfolio_binding.expected_dispatch",
                "instruction": (
                    "Author the bounded six-head pair plus one-head tail source mechanism. "
                    "Do not emit candidate route IDs, call counts, or geometry; the controller "
                    "derives and validates those after authorization."),
            }
        return catalog

    def plan(self, *, context: Mapping[str, Any], workspace: Path,
             checkpoint_path: Path | None = None) -> PlannedCandidate:
        return self._plan(context=context, workspace=workspace, resume=False,
                          checkpoint_path=checkpoint_path)

    def resume_plan(self, *, context: Mapping[str, Any],
                    workspace: Path, checkpoint_path: Path) -> PlannedCandidate:
        return self._plan(context=context, workspace=workspace, resume=True,
                          checkpoint_path=checkpoint_path)

    def _plan(self, *, context: Mapping[str, Any], workspace: Path,
              resume: bool, checkpoint_path: Path | None) -> PlannedCandidate:
        # The model gets a bounded source/profile brief plus a machine contract;
        # it never receives authority to select a campaign, base, executable,
        # argv, profile parser, or evidence regex.
        if resume and self.reviewed_sources is not None:
            self.reviewed_sources.revalidate_materialized(workspace)
            source_package = self.reviewed_sources.manifest()
        else:
            source_package = (None if self.reviewed_sources is None
                              else self.reviewed_sources.materialize(workspace))
        planner_context = context.get("planner_context")
        if (self.reviewed_sources is None or not isinstance(planner_context, Mapping)
                or planner_context.get("reviewed_source_package_sha256")
                != self.reviewed_sources.package_sha256):
            raise DiscoveryControllerError(
                "planner lacks the exact reviewed source preimage authority")
        contract = {
            "plan_json_keys": ["hypothesis_id", "statement", "falsifier", "regime",
                               "proposal", "source_manifest_path", "experiment_intent"],
            "experiment_intent_keys": ["template_id", "target_surface", "target_symbol",
                                       "correctness_id", "dispatch_id", "expected_dispatch",
                                       "load_mode_recommendation"],
            "load_mode_recommendation_keys": ["mode", "rationale", "example_ids"],
            "load_mode_recommendation_semantics": (
                "optional advisory only; it may request safer serialization but cannot author "
                "telemetry, profile facts, bytes, commands, or resource authority"),
            "load_mode_recommendation_modes": ["cold_overlap", "cold_serialized",
                                                "hot_resident"],
            "load_mode_example_ids": sorted({
                str(example.get("id")) for example in
                context.get("admission_policy", {}).get("examples", [])
                if isinstance(example, Mapping) and isinstance(example.get("id"), str)}),
            "expected_dispatch": (
                "array of 1..8 deployed anchor objects copied byte-for-byte from "
                "controller_owned_portfolio_binding.expected_dispatch"),
            "expected_dispatch_rule": (
                "Never substitute predicted candidate subroutes, counts, names, or geometry. "
                "Topology-changing candidate routes are controller-owned and derived only after "
                "the planner's source proposal passes authorization."),
            "expected_dispatch_item_keys": ["route_id", "kernel_name", "calls", "grid", "workgroup", "lds_bytes"],
            "source_manifest_schema": {
                "exact_keys": ["schema", "campaign_id", "proposal_id", "candidate_id",
                    "source_tree", "production_base_commit", "instrument_commit",
                    "change_class", "declared_files", "declared_symbols", "mechanism_id",
                    "patch_sha256", "patch_encoding", "patch_base64"],
                "constants": {"schema": source_candidate.SCHEMA_SOURCE_PATCH,
                              "source_tree": "llama.cpp", "patch_encoding": "base64"},
                "patch_rule": "patch_base64 is strict base64 of a complete UTF-8 unified diff; patch_sha256 hashes the decoded bytes",
                "unified_diff_hunk_rule": (
                    "Every @@ hunk header must contain exact old/new line counts matching its "
                    "body and must end with the reviewed enclosing function symbol from "
                    "declared_symbols for that file. Blank hunk context, a preceding function, "
                    "or a following function is invalid. Before exit, decode patch_base64, "
                    "recount every hunk, and recompute patch_sha256."),
            },
            "proposal_schema": {
                "exact_keys": ["proposal_id", "change_class", "change"],
                "change_exact_keys": ["files_and_symbols", "estimated_diff_size"],
                "files_and_symbols_rule": "sorted file:symbol declarations exactly equal source manifest declarations",
                "estimated_diff_size_rule": (
                    "integer ceiling for actual changed lines in the decoded unified diff; "
                    "actual changed lines are added lines plus removed lines across every hunk"),
            },
            "proposal_requirements": ["proposal_id matches manifest", "change_class matches manifest",
                                       "change.files_and_symbols exactly matches manifest declarations",
                                       "change.estimated_diff_size is positive and is not less than the decoded patch's actual changed-line count"],
            "forbidden": ["commands", "argv", "environment", "measurement results",
                          "campaign/base/instrument selection", "unbounded source reads"],
        }
        assignment = context.get("authoring_assignment")
        if not isinstance(assignment, Mapping):
            raise DiscoveryControllerError("planner context lacks controller-owned authoring assignment")
        binding = assignment.get("portfolio_binding")
        if binding is not None:
            AuthoringAssignment(**assignment)
            example_files = list(binding["target_files"])
            example_file = example_files[0]
            example_symbol = binding["target_symbols"][0]
            example_symbols = list(binding["target_symbols"])
            example_hypothesis = binding["hypothesis_id"]
            example_statement = binding["statement"]
            example_falsifier = binding["falsifier"]
            example_regime = binding["regime"]
            example_template = binding["template_id"]
            example_mechanism = binding["mechanism_id"]
            example_change_class = binding["change_class"]
            example_dispatch = list(binding["expected_dispatch"])
            example_symbols_by_file = {
                relative: list(symbols) for relative, symbols in
                binding["target_symbols_by_file"].items()}
        else:
            example_files = ["ggml/src/ggml-cuda/example.cu"]
            example_file = "ggml/src/ggml-cuda/example.cu"
            example_symbol = "example_symbol"
            example_symbols = [example_symbol]
            example_hypothesis = "akh-example"
            example_statement = "bounded hypothesis"
            example_falsifier = "an exact non-improvement falsifies it"
            example_regime = {"phase": "decode"}
            example_template = "replace-with-reviewed-id"
            example_mechanism = "bounded-example"
            example_change_class = "dispatcher"
            example_symbols_by_file = {example_file: example_symbols}
            example_dispatch = [{"kernel_name": "exact rocprof demangled literal",
                                 "route_id": "replace-with-reviewed-id.anchor.0",
                                 "calls": 1, "grid": 64,
                                 "workgroup": 64, "lds_bytes": 0}]
        example_patch = "".join(
            f"diff --git a/{relative} b/{relative}\n"
            f"--- a/{relative}\n+++ b/{relative}\n"
            f"@@ -1 +1 @@ {example_symbols_by_file[relative][0]}()\n-old\n+new\n"
            for relative in example_files)
        example = {
            "plan.json": {"hypothesis_id": example_hypothesis,
                "statement": example_statement,
                "falsifier": example_falsifier, "regime": example_regime,
                "proposal": {"proposal_id": assignment["proposal_id"], "change_class": example_change_class,
                    "change": {"files_and_symbols": [
                                   f"{relative}:{symbol}"
                                   for relative in example_files
                                   for symbol in example_symbols_by_file[relative]],
                               "estimated_diff_size": 2}},
                "source_manifest_path": "source-patch.json",
                "experiment_intent": {"template_id": example_template,
                    "target_surface": "gpu_decode", "target_symbol": example_symbol,
                    "correctness_id": "backend-ops-hip-v1",
                    "dispatch_id": "decode-tg128-rocprof-v3",
                    "expected_dispatch": example_dispatch}},
            "source-patch.json": {"schema": source_candidate.SCHEMA_SOURCE_PATCH,
                "campaign_id": assignment["campaign_id"], "proposal_id": assignment["proposal_id"],
                "candidate_id": assignment["candidate_id"], "source_tree": "llama.cpp",
                "production_base_commit": assignment["production_base_commit"],
                "instrument_commit": assignment["instrument_commit"], "change_class": example_change_class,
                "declared_files": example_files,
                "declared_symbols": example_symbols_by_file,
                "mechanism_id": example_mechanism,
                "patch_sha256": hashlib.sha256(example_patch.encode()).hexdigest(),
                "patch_encoding": "base64",
                "patch_base64": base64.b64encode(example_patch.encode()).decode("ascii")}}
        prompt = json.dumps({"role": SOL, "context": context,
                             "experiment_template_catalog": self._planner_catalog(),
                             "reviewed_source_package": source_package,
                             "authoring_contract": contract,
                             "controller_owned_portfolio_binding": binding,
                             "structural_example_only": example,
                             "output": "Write plan.json and source-patch.json in workspace."}, sort_keys=True)
        checkpoint_operation_key = (
            checkpoint_path.parent.name if checkpoint_path is not None else None)
        telemetry_operation_key = (
            checkpoint_operation_key
            if isinstance(checkpoint_operation_key, str)
            and HASH.fullmatch(checkpoint_operation_key)
            else _sha({"context": context, "workspace": str(workspace)}))
        self._runtime()
        # Re-emitting on resume is intentional: operation-key idempotence
        # repairs a crash-partial planner projection without duplicating a
        # previously committed start event.
        _emit_observational_telemetry(
            self.telemetry,
            "planner", "planner_started",
            campaign_id=assignment["campaign_id"],
            hypothesis_id=example_hypothesis, provider=SOL["provider"],
            model=SOL["model"], effort=SOL["effort"],
            operation_key=telemetry_operation_key,
            failure_sink=self.telemetry_failures)
        if resume:
            if checkpoint_path is None:
                raise DiscoveryControllerError(
                    "planner resume lacks its controller checkpoint path")
            checkpoint = _reopen_planner_actor_checkpoint(
                workspace, checkpoint_path, context=context)
            result_facts = dict(checkpoint["result"])
            actor_failure_message = None
        else:
            try:
                result = codex_container_actor.run_actor(wrapper=self.wrapper, workspace=workspace, model=SOL["model"], effort=SOL["effort"], prompt=prompt, environment=self.environment,
                    expected_wrapper_sha256=self.wrapper_sha256,
                    expected_runtime_identity=self.runtime_identity,
                    expected_launcher_sha256=self.actor_launcher_sha256)
            except Exception:
                _emit_observational_telemetry(
                        self.telemetry,
                        "planner", "planner_failed",
                        campaign_id=assignment["campaign_id"],
                        hypothesis_id=example_hypothesis, provider=SOL["provider"],
                        model=SOL["model"], effort=SOL["effort"],
                        operation_key=telemetry_operation_key,
                        failure_sink=self.telemetry_failures)
                raise
            result_facts = {
                "returncode": result.returncode,
                "stdout_sha256": hashlib.sha256(
                    getattr(result, "stdout", "").encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    getattr(result, "stderr", "").encode()).hexdigest(),
            }
            actor_failure_message = (
                f"Sol actor failed: {getattr(result, 'stderr', '')[-400:]}"
                if result.returncode else None)
            if checkpoint_path is not None:
                _seal_planner_actor_checkpoint(
                    workspace, checkpoint_path, context=context,
                    result=result_facts)
        if result_facts["returncode"]:
            _emit_observational_telemetry(
                    self.telemetry,
                    "planner", "planner_failed",
                    campaign_id=assignment["campaign_id"],
                    hypothesis_id=example_hypothesis, provider=SOL["provider"],
                    model=SOL["model"], effort=SOL["effort"],
                    result=result_facts,
                    operation_key=telemetry_operation_key,
                    failure_sink=self.telemetry_failures)
            raise PlannerProviderTransient(
                actor_failure_message
                or f"sealed Sol actor invocation failed with return code "
                   f"{result_facts['returncode']}")
        if self.reviewed_sources is not None:
            self.reviewed_sources.revalidate_materialized(workspace)
        try:
            candidate = _load_plan(
                workspace / "plan.json", workspace,
                assignment=AuthoringAssignment(**assignment))
        except PlannerOutputRefusal as exc:
            telemetry_exc = _emit_observational_telemetry(
                self.telemetry, "planner", "planner_refused",
                campaign_id=assignment["campaign_id"],
                hypothesis_id=example_hypothesis,
                provider=SOL["provider"], model=SOL["model"],
                effort=SOL["effort"], operation_key=telemetry_operation_key,
                failure_sink=self.telemetry_failures,
                result={
                    **result_facts,
                    "refusal_type": "planner_output_refusal",
                    "refusal_reason_sha256": hashlib.sha256(
                        str(exc).encode()).hexdigest(),
                })
            if self.telemetry is not None and telemetry_exc is None:
                exc.telemetry_status = "emitted"
            elif telemetry_exc is not None:
                exc.note_telemetry_failure(telemetry_exc)
            raise
        _emit_observational_telemetry(
                self.telemetry,
                "planner", "planner_completed",
                campaign_id=assignment["campaign_id"],
                hypothesis_id=example_hypothesis, provider=SOL["provider"],
                model=SOL["model"], effort=SOL["effort"], result=result_facts,
                operation_key=telemetry_operation_key,
                failure_sink=self.telemetry_failures)
        return candidate


class ClaudeCritic:
    """Concrete Fable 5 critic. It can bind a veto but never alters the candidate."""
    def __init__(self, *, wrapper: Path, environment: Mapping[str, str],
                 template_catalog: Mapping[str, Any] | None = None,
                 reviewed_sources: ReviewedSourcePackage | None = None,
                 wrapper_sha256: str | None = None,
                 runtime_identity: Mapping[str, Any] | None = None,
                 actor_launcher_sha256: str | None = None,
                 telemetry: discovery_telemetry.DiscoveryTelemetry | None = None,
                 auth_root: Path = Path("/home/node/.claude")) -> None:
        self.wrapper, self.environment = wrapper, dict(environment)
        self.template_catalog = json.loads(json.dumps(template_catalog or {}, sort_keys=True))
        self.reviewed_sources = reviewed_sources
        self.wrapper_sha256 = wrapper_sha256
        self.runtime_identity = None if runtime_identity is None else dict(runtime_identity)
        self.actor_launcher_sha256 = actor_launcher_sha256
        self.telemetry = telemetry
        self.telemetry_failures: list[dict[str, str]] = []
        self.auth_root = auth_root
    def _runtime(self) -> Mapping[str, Any]:
        if self.wrapper_sha256 is not None:
            if self.wrapper.is_symlink() or not self.wrapper.is_file() or hashlib.sha256(self.wrapper.read_bytes()).hexdigest() != self.wrapper_sha256:
                raise DiscoveryControllerError("sealed Claude critic wrapper bytes changed")
        current = claude_fable5_critic_actor.runtime_identity(self.wrapper)
        if self.runtime_identity is not None and current != self.runtime_identity:
            raise DiscoveryControllerError("sealed Claude critic runtime identity changed")
        if (self.actor_launcher_sha256 is not None
                and hashlib.sha256(Path(claude_fable5_critic_actor.__file__).resolve().read_bytes()).hexdigest()
                != self.actor_launcher_sha256):
            raise DiscoveryControllerError("sealed Claude critic launcher/argv policy changed")
        return current
    def attest(self) -> Mapping[str, Any]: return {**FABLE5_CRITIC, "runtime": self._runtime()}
    def review(self, candidate: PlannedCandidate, *, context: Mapping[str, Any], workspace: Path) -> Critique:
        manifest = candidate.source_manifest
        if len(manifest.patch_text.encode("utf-8")) > 65536:
            raise DiscoveryControllerError("candidate patch exceeds bounded critic visibility")
        source_context = None
        if self.reviewed_sources is not None:
            if not 1 <= len(manifest.declared_files) <= 2:
                raise DiscoveryControllerError(
                    "critic requires one or two exact reviewed source preimages")
            source_context = [
                self.reviewed_sources.critic_context(
                    relative, manifest.declared_symbols[relative])
                for relative in manifest.declared_files
            ]
        critic_context = {**context, "selected_source_preimage": source_context}
        bindings = {
            "proposal_sha256": _sha(candidate.proposal),
            "source_manifest_sha256": candidate.source_manifest_sha256,
            "candidate_patch_sha256": manifest.patch_sha256,
            "context_sha256": _sha(critic_context),
            "template_catalog_sha256": _sha(self.template_catalog),
        }
        prompt = json.dumps({"role": FABLE5_CRITIC, "context": critic_context,
            "experiment_template_catalog": self.template_catalog, "candidate": {
            "hypothesis_id": candidate.hypothesis_id, "statement": candidate.statement,
            "falsifier": candidate.falsifier, "proposal": candidate.proposal,
            "experiment_intent": asdict(candidate.experiment_intent) if candidate.experiment_intent else None,
            "source_manifest_sha256": candidate.source_manifest_sha256,
            "manifest": {"campaign_id": manifest.campaign_id, "proposal_id": manifest.proposal_id,
                         "candidate_id": manifest.candidate_id,
                         "production_base_commit": manifest.production_base_commit,
                         "instrument_commit": manifest.instrument_commit,
                         "declared_files": list(manifest.declared_files),
                         "declared_symbols": {key: list(value) for key, value in manifest.declared_symbols.items()},
                         "patch_sha256": manifest.patch_sha256, "patch_text": manifest.patch_text}},
            "required_output_bindings": bindings,
            "output": "Return only the strict structured critique; do not edit files or use tools."}, sort_keys=True)
        self._runtime()
        campaign_id = manifest.campaign_id
        telemetry_operation_key = _sha(bindings)
        _emit_observational_telemetry(
                self.telemetry,
                "autokernel", "critic_started", campaign_id=campaign_id,
                hypothesis_id=candidate.hypothesis_id,
                provider=FABLE5_CRITIC["provider"], model=FABLE5_CRITIC["model"],
                effort=FABLE5_CRITIC["effort"],
                operation_key=telemetry_operation_key,
                failure_sink=self.telemetry_failures)
        try:
            result = claude_fable5_critic_actor.run_critic(
                wrapper=self.wrapper, workspace=workspace, prompt=prompt,
                bindings=bindings, environment=self.environment,
                auth_root=self.auth_root,
                expected_wrapper_sha256=self.wrapper_sha256,
                expected_runtime_identity=self.runtime_identity,
                expected_launcher_sha256=self.actor_launcher_sha256)
        except Exception:
            _emit_observational_telemetry(
                    self.telemetry,
                    "autokernel", "critic_failed", campaign_id=campaign_id,
                    hypothesis_id=candidate.hypothesis_id,
                    provider=FABLE5_CRITIC["provider"], model=FABLE5_CRITIC["model"],
                    effort=FABLE5_CRITIC["effort"],
                    operation_key=telemetry_operation_key,
                    failure_sink=self.telemetry_failures)
            raise
        if self.telemetry is not None:
            _emit_observational_telemetry(
                    self.telemetry,
                    "autokernel", "critic_completed", campaign_id=campaign_id,
                    hypothesis_id=candidate.hypothesis_id,
                    provider=FABLE5_CRITIC["provider"], model=FABLE5_CRITIC["model"],
                    effort=FABLE5_CRITIC["effort"],
                    operation_key=telemetry_operation_key,
                    failure_sink=self.telemetry_failures, result={
                        "stdout_sha256": result.stdout_sha256,
                        "stderr_sha256": result.stderr_sha256,
                        "decision": result.decision,
                    })
        return Critique(result.decision, result.reason)


def _read_object(path: Path, root: Path) -> dict[str, Any]:
    try: path.resolve().relative_to(root.resolve())
    except ValueError as exc: raise DiscoveryControllerError("actor artifact escaped workspace") from exc
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise DiscoveryControllerError(f"invalid actor artifact {path.name}") from exc
    if not isinstance(value, dict): raise DiscoveryControllerError("actor artifact must be object")
    return value


def _read_planner_object(path: Path, root: Path) -> dict[str, Any]:
    """Read planner JSON while keeping containment violations terminal."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DiscoveryControllerError("actor artifact escaped workspace") from exc
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerOutputRefusal(f"invalid actor artifact {path.name}") from exc
    if not isinstance(value, dict):
        raise PlannerOutputRefusal("actor artifact must be object")
    return value


_RETRYABLE_PLANNER_SOURCE_ERRORS = (
    "source patch manifest is not strict JSON:",
    "source patch manifest fields must be exactly",
    "source patch manifest schema/encoding is unsupported",
    "patch_base64 is invalid:",
    "patch_sha256 does not match the embedded patch bytes",
    "patch is not strict UTF-8:",
    "patch contains NUL bytes",
    "patch is not an accounted unified diff:",
    "hunk appears before a diff --git header",
    "source patch contains no accounted hunk",
    "patch bytes must end in a newline",
)


def _load_planner_manifest(path: Path) -> source_candidate.SourcePatchManifest:
    """Classify only syntactic carrier/patch defects as retryable output.

    Identity, path/symbol scope, change-class, reward-integrity, and instrument
    policy errors intentionally retain ``SourceCandidateError`` and terminate
    fail closed.
    """
    try:
        return source_candidate.load_source_patch_manifest(path)
    except source_candidate.SourceCandidateError as exc:
        reason = str(exc)
        if any(reason.startswith(prefix) for prefix in
               _RETRYABLE_PLANNER_SOURCE_ERRORS):
            raise PlannerOutputRefusal(
                f"SourceCandidateError: {reason}") from exc
        raise


def _load_plan(path: Path, root: Path, *, assignment: AuthoringAssignment | None = None) -> PlannedCandidate:
    value = _read_planner_object(path, root)
    allowed = {"hypothesis_id", "statement", "falsifier", "regime", "proposal", "source_manifest_path", "experiment_intent"}
    if set(value) not in (allowed, allowed - {"experiment_intent"}): raise PlannerOutputRefusal("planner output schema mismatch")
    intent_raw = value.pop("experiment_intent", None)
    if intent_raw is not None:
        allowed_intent = {"template_id", "target_surface", "target_symbol", "correctness_id", "dispatch_id", "expected_dispatch", "load_mode_recommendation"}
        if not isinstance(intent_raw, Mapping) or set(intent_raw) not in (allowed_intent, allowed_intent - {"load_mode_recommendation"}):
            raise PlannerOutputRefusal("planner experiment intent schema mismatch")
        expected = intent_raw["expected_dispatch"]
        expected_keys = {"route_id", "kernel_name", "calls", "grid", "workgroup", "lds_bytes"}
        if (not isinstance(expected, list) or not 1 <= len(expected) <= 8
                or not all(isinstance(row, Mapping) and set(row) == expected_keys
                           for row in expected)):
            raise PlannerOutputRefusal("planner bounded dispatch schema mismatch")
        try:
            recommendation = intent_raw.get("load_mode_recommendation")
            if recommendation is not None:
                if not isinstance(recommendation, Mapping) or set(recommendation) != {"mode", "rationale", "example_ids"}:
                    raise PlannerOutputRefusal("planner load-mode recommendation schema mismatch")
                recommendation = LoadModeRecommendation(
                    mode=recommendation["mode"], rationale=recommendation["rationale"],
                    example_ids=tuple(recommendation["example_ids"]))
            intent = GpuSourceExperimentIntent(**{**intent_raw,
                "expected_dispatch": tuple(BoundedDispatchExpectation(**row) for row in expected),
                "load_mode_recommendation": recommendation})
        except PlannerOutputRefusal:
            raise
        except (DiscoveryControllerError, TypeError, ValueError) as exc:
            # This is actor-authored plan content, not controller/deployment
            # corruption.  Keep the durable reason bounded and secret-free;
            # telemetry records only its class and digest.
            raise PlannerOutputRefusal(
                "planner experiment intent violates deployed authority") from exc
    else:
        intent = None
    raw_path = Path(_text(value.pop("source_manifest_path"), "source_manifest_path"))
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise DiscoveryControllerError("source manifest path must be a workspace-relative path")
    manifest_path = root / raw_path
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise PlannerOutputRefusal(
            f"invalid actor artifact {manifest_path.name}") from exc
    try:
        resolved_manifest.relative_to(root.resolve())
    except ValueError as exc:
        raise DiscoveryControllerError(
            "source manifest escaped disposable workspace") from exc
    manifest = _load_planner_manifest(resolved_manifest)
    if assignment is not None:
        if (manifest.campaign_id, manifest.proposal_id, manifest.candidate_id,
                manifest.production_base_commit, manifest.instrument_commit) != (
                    assignment.campaign_id, assignment.proposal_id, assignment.candidate_id,
                    assignment.production_base_commit, assignment.instrument_commit):
            raise DiscoveryControllerError("actor attempted to invent campaign/base/instrument identity")
        if value.get("proposal", {}).get("proposal_id") != assignment.proposal_id:
            raise DiscoveryControllerError("actor proposal does not use controller-assigned proposal identity")
        # Bind the actor's proposal to controller-owned identity and the exact
        # manifest file:symbol scope before the critic or any claim can run.
        # The actor may not omit, regroup, or reformat these declarations.
        manifest.bind(
            proposal=value.get("proposal", {}),
            campaign_id=assignment.campaign_id,
            candidate_id=assignment.candidate_id,
            production_base_commit=assignment.production_base_commit,
            instrument_commit=assignment.instrument_commit)
        proposal = value.get("proposal")
        change = proposal.get("change") if isinstance(proposal, Mapping) else None
        estimated = change.get("estimated_diff_size") if isinstance(change, Mapping) else None
        if isinstance(estimated, bool) or not isinstance(estimated, int) or estimated < 1:
            raise PlannerOutputRefusal("planner estimated_diff_size must be a positive integer")
        try:
            actual_changed_lines = integrity.parse_unified_diff(
                manifest.patch_bytes.decode("utf-8")).total_changed
        except (UnicodeDecodeError, integrity.DiffParseError) as exc:
            raise PlannerOutputRefusal(
                "planner patch cannot be counted as a complete UTF-8 unified diff") from exc
        if estimated < actual_changed_lines:
            raise PlannerOutputRefusal(
                "planner estimated_diff_size is smaller than the decoded patch's actual "
                f"changed-line count ({estimated} < {actual_changed_lines})")
    return PlannedCandidate(**value, source_manifest=manifest, source_manifest_sha256=manifest.patch_bundle_sha256,
                            experiment_intent=intent)


class CampaignScreener:
    """Concrete adapter: call the existing candidate-only campaign transaction."""
    def __init__(self, *, spec_factory: Callable[[PlannedCandidate, hypotheses.ClaimAuthorization], campaign.CampaignSpec], ops_factory: Callable[[], Any]) -> None:
        self.spec_factory, self.ops_factory = spec_factory, ops_factory
    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen:
        spec = self.spec_factory(candidate, authorization)
        if not spec.screening_only or spec.source_patch is not candidate.source_manifest or spec.authorization != authorization:
            raise DiscoveryControllerError("campaign adapter must bind typed patch, authorization, and candidate-only screen")
        if spec.source_prerequisite_package is None and spec.fresh_source_prerequisite_plan is None:
            raise DiscoveryControllerError("source candidate requires source correctness and dispatch prerequisite package")
        result = campaign.run_campaign(spec, self.ops_factory())
        return _screen_from_campaign(result)


def _screen_from_campaign(result: campaign.CampaignResult) -> SealedScreen:
    raw = result.to_dict(); report = raw.get("screening_report")
    if not (result.ok and raw.get("state") == "decided" and raw.get("screening_only") is True and isinstance(report, Mapping)):
        raise DiscoveryControllerError("campaign did not produce a sealed candidate-only result")
    required = ("baseline_sha256", "source_prerequisite_package_sha256", "dispatch_attribution_sha256", "result_sha256")
    if not all(isinstance(report.get(key), str) and HASH.fullmatch(report[key]) for key in required):
        raise DiscoveryControllerError("campaign result lacks source proof, exact dispatch proof, baseline, or result hash")
    return SealedScreen(receipt_path=str(report.get("receipt_path", "")), result_sha256=report["result_sha256"], effect_fraction=float(report["median_relative"]), classification=str(report.get("classification", "candidate")), baseline_sha256=report["baseline_sha256"], source_proof_sha256=report["source_prerequisite_package_sha256"], dispatch_proof_sha256=report["dispatch_attribution_sha256"])


@dataclass(frozen=True)
class GpuSourceBuild:
    """A completed isolated build, returned only by a typed source-build seam."""
    anchor_build: Path
    candidate_build: Path
    candidate_identity: gpu_source_proofs.BuildIdentity
    anchor_identity: gpu_source_proofs.BuildIdentity
    measurement_binary: Path | None = None
    common_loader_dir: Path | None = None
    anchor_loader_dir: Path | None = None
    candidate_loader_dir: Path | None = None
    reward_runtime_sha256: str | None = None
    operation_key: str | None = None
    build_key: str | None = None
    materialization_receipt: Path | None = None
    materialization_sha256: str | None = None
    anchor_source_tree_receipt: Path | None = None
    anchor_source_tree_sha256: str | None = None
    candidate_source_tree_receipt: Path | None = None
    candidate_source_tree_sha256: str | None = None
    anchor_correctness_binary: Path | None = None
    anchor_correctness_binary_sha256: str | None = None
    candidate_correctness_binary: Path | None = None
    candidate_correctness_binary_sha256: str | None = None
    anchor_correctness_capability_receipt: Path | None = None
    anchor_correctness_capability_sha256: str | None = None
    candidate_correctness_capability_receipt: Path | None = None
    candidate_correctness_capability_sha256: str | None = None
    teardown_receipt: Path | None = None
    teardown_sha256: str | None = None
    composition_build_pair: cumulative_composition.CumulativeBuildPair | None = None
    composition_production_authority: (
        cumulative_composition.FrozenProductionAuthority | None) = None
    def __post_init__(self) -> None:
        for path in (self.anchor_build, self.candidate_build):
            if not path.is_absolute() or not path.is_dir():
                raise DiscoveryControllerError("GPU source build paths must be existing absolute directories")
        if self.candidate_identity == self.anchor_identity:
            raise DiscoveryControllerError("source screen requires distinct sealed anchor and candidate build identities")
        runtime = (self.measurement_binary, self.common_loader_dir, self.anchor_loader_dir,
                   self.candidate_loader_dir, self.reward_runtime_sha256)
        if any(value is not None for value in runtime):
            if (not all(value is not None for value in runtime)
                    or not isinstance(self.measurement_binary, Path) or not self.measurement_binary.is_file()
                    or not isinstance(self.common_loader_dir, Path) or not self.common_loader_dir.is_dir()
                    or not isinstance(self.anchor_loader_dir, Path) or not self.anchor_loader_dir.is_dir()
                    or not isinstance(self.candidate_loader_dir, Path) or not self.candidate_loader_dir.is_dir()
                    or not isinstance(self.reward_runtime_sha256, str) or not HASH.fullmatch(self.reward_runtime_sha256)):
                raise DiscoveryControllerError("GPU source build has an incomplete shared reward closure")
        if self.operation_key is not None and (not isinstance(self.operation_key, str) or not HASH.fullmatch(self.operation_key)):
            raise DiscoveryControllerError("GPU source build operation key is invalid")
        if self.build_key is not None and (not isinstance(self.build_key, str) or not HASH.fullmatch(self.build_key)):
            raise DiscoveryControllerError("GPU source build cache key is invalid")
        for path, expected, label in ((self.materialization_receipt, self.materialization_sha256, "materialization"),
                                      (self.anchor_source_tree_receipt, self.anchor_source_tree_sha256, "anchor source tree"),
                                      (self.candidate_source_tree_receipt, self.candidate_source_tree_sha256, "candidate source tree"),
                                      (self.anchor_correctness_binary, self.anchor_correctness_binary_sha256, "anchor correctness binary"),
                                      (self.candidate_correctness_binary, self.candidate_correctness_binary_sha256, "candidate correctness binary"),
                                      (self.anchor_correctness_capability_receipt, self.anchor_correctness_capability_sha256, "anchor correctness capability"),
                                      (self.candidate_correctness_capability_receipt, self.candidate_correctness_capability_sha256, "candidate correctness capability"),
                                      (self.teardown_receipt, self.teardown_sha256, "teardown")):
            if (path is None) != (expected is None):
                raise DiscoveryControllerError(f"GPU source build has incomplete {label} receipt")
            if path is not None:
                if (not isinstance(path, Path) or not path.is_absolute() or path.is_symlink()
                        or not path.is_file() or not isinstance(expected, str) or not HASH.fullmatch(expected)):
                    raise DiscoveryControllerError(f"GPU source build has invalid {label} receipt")
                assert isinstance(path, Path) and isinstance(expected, str)
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise DiscoveryControllerError(f"GPU source {label} receipt bytes changed")
        if (self.composition_build_pair is not None
                and not isinstance(
                    self.composition_build_pair,
                    cumulative_composition.CumulativeBuildPair)):
            raise DiscoveryControllerError(
                "GPU source build cumulative pair must be typed")
        if (self.composition_build_pair is None
                and self.composition_production_authority is not None):
            raise DiscoveryControllerError(
                "ordinary GPU source build acquired production authority")
        if (self.composition_production_authority is not None
                and not isinstance(
                    self.composition_production_authority,
                    cumulative_composition.FrozenProductionAuthority)):
            raise DiscoveryControllerError(
                "GPU source build production authority must be typed")


@dataclass(frozen=True)
class ProofReceipt:
    """Hash-bound source or dispatch proof produced before any screen call."""
    path: Path
    sha256: str
    kind: str
    def __post_init__(self) -> None:
        if self.kind not in {"source", "dispatch"} or not self.path.is_absolute() or not self.path.is_file() or not HASH.fullmatch(self.sha256):
            raise DiscoveryControllerError("proof receipt must be an existing typed source/dispatch artifact")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise DiscoveryControllerError("proof receipt bytes differ from its sealed hash")


class GpuSourceScreener:
    """GPU source lane using the existing governed discovery runner.

    This intentionally does not reuse the CPU baseline bank: that bank proves an
    unchanged binary with a parameter delta.  GPU source runs need distinct
    build identities and their own sealed paired receipt.
    """
    def __init__(self, *, build_source: Callable[[PlannedCandidate, hypotheses.ClaimAuthorization, Mapping[str, Any]], GpuSourceBuild],
                 proof_bundle: Callable[[PlannedCandidate, GpuSourceBuild], gpu_source_proofs.GpuSourceProofBundle],
                 args_factory: Callable[[PlannedCandidate, GpuSourceBuild, Mapping[str, Any]], Any],
                 runner_attest: Callable[[], None] = lambda: None) -> None:
        self.build_source, self.proof_bundle, self.args_factory = build_source, proof_bundle, args_factory
        self.runner_attest = runner_attest

    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen:
        try:
            build = self.build_source(candidate, authorization, lease)
        except source_candidate.SourceCandidateError as exc:
            # Source materialization re-derives the committed diff after the
            # critic's review.  A mismatch here is an authoring rejection, not
            # an ambiguous GPU operation: proof production, reservation, and
            # the throughput runner are all strictly downstream of this call.
            # Preserve that ordering as a typed precompute refusal so the
            # controller durably records the failed iteration and advances.
            raise PrecomputeScreenRefusal(
                f"source candidate authoring rejected: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            bundle = self.proof_bundle(candidate, build)
        except CorrectnessRefusal as exc:
            if getattr(candidate, "composition_plan", None) is None:
                raise
            pair = build.composition_build_pair
            if pair is None:
                raise DiscoveryControllerError(
                    "cumulative correctness refusal lacks its build pair") from exc
            pair.bind_plan(candidate.composition_plan)
            correctness = cumulative_composition.FullCorrectness.create(
                pair, suite_id="current-gpu-source-full-correctness-v1",
                cases_sha256=schemas.content_hash({
                    "stage": exc.stage,
                    "receipt_sha256": exc.receipt_sha256}),
                receipt_sha256=exc.receipt_sha256, passed=False)
            raise CumulativeCorrectnessRefusal(
                str(exc), receipt_path=exc.receipt_path,
                receipt_sha256=exc.receipt_sha256,
                build_pair=pair, correctness=correctness) from exc
        if not isinstance(bundle, gpu_source_proofs.GpuSourceProofBundle):
            raise DiscoveryControllerError("GPU source gate did not return a validated proof bundle")
        if bundle.manifest_sha256 != candidate.source_manifest_sha256:
            raise DiscoveryControllerError("GPU proof bundle does not bind the candidate manifest")
        if bundle.candidate != build.candidate_identity or bundle.anchor != build.anchor_identity:
            raise DiscoveryControllerError("GPU proof bundle does not bind both sealed build identities")
        args = self.args_factory(candidate, build, lease)
        target_args = getattr(args, "_target_runtime_args", None)
        if target_args is None:
            raise DiscoveryControllerError(
                "GPU source runner lacks a separate target-runtime stage")
        # The established runner owns KFD/VRAM, device claims, paired samples,
        # and its durable result.  This controller does not spawn a shell.
        if any(getattr(current, "factor", None) != "source_patch"
               or Path(getattr(current, "anchor_build", "")).resolve()
               != build.anchor_build
               or Path(getattr(current, "candidate_build", "")).resolve()
               != build.candidate_build
               for current in (args, target_args)):
            raise DiscoveryControllerError("GPU source runner arguments are not bound to the typed build")
        production_on_args = getattr(args, "_production_graphs_on_args", None)
        production = build.composition_production_authority
        if getattr(candidate, "composition_plan", None) is not None:
            if production is None or production_on_args is None:
                raise DiscoveryControllerError(
                    "cumulative runner lacks frozen-production comparison")
            production.bind_plan(candidate.composition_plan)
            for current, mode in ((production_on_args, "on"),):
                if (getattr(current, "factor", None) !=
                        "cumulative_production"
                        or getattr(current, "runtime_graphs", None) != mode
                        or getattr(current,
                                   "_frozen_production_authority", None) !=
                           production.to_dict()):
                    raise DiscoveryControllerError(
                        "production runner arguments changed comparator authority")
        elif production_on_args is not None:
            raise DiscoveryControllerError(
                "ordinary runner acquired production-comparison stages")
        attribution_body = bundle.attribution.get("body")
        comparison = (attribution_body.get("exact_duration_comparison")
                      if isinstance(attribution_body, Mapping) else None)
        if not isinstance(comparison, Mapping):
            raise DiscoveryControllerError(
                "GPU source proof lacks exact-duration decision evidence")
        exact_effect = float(comparison["relative_improvement_fraction"])
        if (exact_effect <= 0
                and getattr(candidate, "composition_plan", None) is None):
            # A valid neutral/regressed exact-route measurement is a scientific
            # outcome, not a refusal.  It terminates before any whole-model
            # benchmark and therefore cannot consume a target-runtime call.
            result_path = (Path(args.output_dir).resolve().parent /
                           "exact-attribution-outcome.json")
            body = {
                "schema": "epyc.autokernel.exact_attribution_outcome.v1",
                "authority": "nonpromotable_candidate_only_discovery",
                "non_promotable": True, "promotion_claim": False,
                "status": "complete", "classification": "screened_out",
                "manifest_sha256": candidate.source_manifest_sha256,
                "exact_attribution_effect_fraction": exact_effect,
                "target_runtime_executed": False,
                "target_runtime_reason": "nonpositive_exact_duration",
                "dispatch_proof_sha256": bundle.attribution["file_sha256"],
            }
            body["result_sha256"] = schemas.content_hash(body)
            if result_path.exists() or result_path.is_symlink():
                if result_path.is_symlink() or json.loads(
                        result_path.read_text(encoding="utf-8")) != body:
                    raise DiscoveryControllerError(
                        "exact-attribution outcome receipt changed")
            else:
                _atomic(result_path, body)
            return SealedScreen(
                receipt_path=str(result_path),
                result_sha256=body["result_sha256"],
                effect_fraction=exact_effect, classification="screened_out",
                baseline_sha256=schemas.content_hash(
                    comparison.get("anchor_routes", comparison)),
                source_proof_sha256=bundle.correctness["file_sha256"],
                dispatch_proof_sha256=bundle.attribution["file_sha256"],
                exact_attribution_effect_fraction=exact_effect,
                target_runtime_effect_fraction=None,
                stages=("materialized", "built", "correctness", "attribution"))
        def run_stage(current: Any, *, graph_mode: str) -> tuple[Path, Mapping[str, Any]]:
            # Immediately-before-call byte attestation prevents a validated
            # graph from silently executing changed controller/runner bytes.
            self.runner_attest()
            result_path = Path(current.output_dir).resolve() / "result.json"
            if result_path.exists() and not result_path.is_symlink():
                raw = gpu_source_proofs.load_receipt(
                    result_path,
                    schema="epyc.autokernel.gpu_candidate_only_screen.v2")["body"]
            else:
                try:
                    raw = gpu_discovery.run(current)
                except gpu_discovery.TimedOutputInfrastructureAmbiguity as exc:
                    raise ScreenInfrastructureAmbiguity(
                        str(exc), receipt_path=exc.receipt_path,
                        receipt_sha256=exc.receipt_sha256,
                        operation_key=exc.operation_key) from exc
                except gpu_discovery.CandidateCorrectnessDivergence as exc:
                    raise TimedOutputCorrectnessRefusal(
                        str(exc), receipt_path=exc.receipt_path,
                        receipt_sha256=exc.receipt_sha256,
                        result_sha256=exc.result_sha256,
                        operation_key=exc.operation_key) from exc
                except gpu_discovery.MeasurementOutputRefusal as exc:
                    raise MeasurementOutputRefusal(
                        str(exc), receipt_path=exc.receipt_path,
                        receipt_sha256=exc.receipt_sha256) from exc
            raw = gpu_source_proofs.require_result_file(result_path, raw)["body"]
            if not (raw.get("schema") == "epyc.autokernel.gpu_candidate_only_screen.v2"
                    and raw.get("non_promotable") is True
                    and raw.get("promotion_claim") is False
                    and raw.get("hip_residency_proved") is True
                    and raw.get("runtime_graphs") == graph_mode):
                raise DiscoveryControllerError(
                    f"GPU {graph_mode} runner returned an unsealed/non-resident result")
            if (hasattr(current, "_device_claim_acquirer")
                    and raw.get("device_claim_mode") != "borrowed_outer_reservation"):
                raise DiscoveryControllerError(
                    "GPU runner did not bind throughput to the borrowed outer reservation")
            if not hasattr(current, "_device_claim_acquirer"):
                return result_path, raw
            expected_outer = getattr(current, "_expected_outer_claim_id", None)
            opened = raw.get("device_claim_open")
            phase_end = raw.get("device_claim_borrowed_phase_end")
            if (not isinstance(expected_outer, str)
                    or not isinstance(opened, Mapping)
                    or opened.get("claim_id") != expected_outer
                    or not isinstance(phase_end, Mapping)
                    or phase_end.get("schema") !=
                    "epyc.autokernel.borrowed_device_claim_phase.v1"
                    or phase_end.get("outer_claim_id") != expected_outer
                    or phase_end.get("physical_release") is not False
                    or "released_at" in phase_end
                    or raw.get("device_claim_released") is not None):
                raise DiscoveryControllerError(
                    "GPU runner borrowed phase does not bind the exact outer claim")
            governance_path = Path(current.output_dir).resolve() / "live-governance.json"
            try:
                governance = json.loads(governance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DiscoveryControllerError(
                    "GPU runner lacks terminal borrowed-phase governance") from exc
            if (not isinstance(governance, Mapping)
                    or governance.get("status") != "borrowed_phase_ended"
                    or governance.get("device_claim_mode") != "borrowed_outer_reservation"
                    or governance.get("device_claim_open") != opened
                    or governance.get("device_claim_borrowed_phase_end") != phase_end
                    or governance.get("device_claim_released") is not None):
                raise DiscoveryControllerError(
                    "GPU runner terminal governance differs from its borrowed phase")
            return result_path, raw

        graphs_off_path, graphs_off = run_stage(args, graph_mode="off")
        result_path, raw = run_stage(target_args, graph_mode="on")
        production_graphs_on_path = None
        production_graphs_on = None
        if production is not None:
            production_graphs_on_path, production_graphs_on = run_stage(
                production_on_args, graph_mode="on")
        projection = autokernel_progression._gpu_screen(result_path, raw)
        if projection is None: raise DiscoveryControllerError("GPU result failed canonical progression validation")
        target_effect = float(raw["median_relative"])
        graphs_off_effect = float(graphs_off["median_relative"])
        build_identity_sha256 = schemas.content_hash(
            vars(build.candidate_identity))
        graphs_off_file_sha256 = hashlib.sha256(
            graphs_off_path.read_bytes()).hexdigest()
        graphs_on_file_sha256 = hashlib.sha256(
            result_path.read_bytes()).hexdigest()
        composition_correctness = None
        composition_comparison = None
        cumulative_performance = None
        cumulative_performance_ref = None
        if getattr(candidate, "composition_plan", None) is not None:
            pair = build.composition_build_pair
            if pair is None:
                raise DiscoveryControllerError(
                    "cumulative source build lacks its typed build pair")
            pair.bind_plan(candidate.composition_plan)
            correctness_body = bundle.correctness.get("body")
            if not isinstance(correctness_body, Mapping):
                raise DiscoveryControllerError(
                    "cumulative correctness proof body is missing")
            composition_correctness = cumulative_composition.FullCorrectness.create(
                pair, suite_id="current-gpu-source-full-correctness-v1",
                cases_sha256=schemas.content_hash(correctness_body),
                receipt_sha256=bundle.correctness["file_sha256"], passed=True)
            expected_routes = comparison.get("candidate_routes")
            if not isinstance(expected_routes, (list, tuple, Mapping)):
                expected_routes = comparison
            composition_comparison = cumulative_composition.IncrementalComparison.create(
                pair, composition_correctness,
                exact_route_receipt_sha256=bundle.attribution["file_sha256"],
                exact_route_receipt_path=bundle.attribution["path"],
                expected_route_set_sha256=schemas.content_hash(expected_routes),
                graphs_off_receipt_sha256=graphs_off_file_sha256,
                graphs_off_receipt_path=graphs_off_path,
                graphs_on_receipt_sha256=graphs_on_file_sha256,
                graphs_on_receipt_path=result_path,
                target_runtime_frame_sha256=schemas.content_hash({
                    "baseline_sha256": raw["baseline_sha256"],
                    "runtime_graphs": raw["runtime_graphs"],
                    "factor": raw.get("factor"),
                    "technical_workload": raw.get("technical_workload"),
                }),
                exact_route_effect_fraction=exact_effect,
                graphs_off_effect_fraction=graphs_off_effect,
                graphs_on_effect_fraction=target_effect)
            if (production is None or production_graphs_on_path is None
                    or production_graphs_on is None):
                raise DiscoveryControllerError(
                    "cumulative production comparison is incomplete")
            cumulative_performance = \
                cumulative_composition.performance_from_measurements(
                    candidate.composition_plan, pair,
                    composition_correctness, composition_comparison,
                    frozen_production=production,
                    incremental_graphs_off=graphs_off,
                    incremental_graphs_on=raw,
                    production_graphs_on=production_graphs_on,
                    production_graphs_on_receipt_sha256=hashlib.sha256(
                        production_graphs_on_path.read_bytes()).hexdigest(),
                    production_graphs_on_receipt_path=
                        production_graphs_on_path)
            performance_path = getattr(
                args, "_cumulative_performance_path", None)
            if not isinstance(performance_path, str):
                raise DiscoveryControllerError(
                    "cumulative runner lacks performance receipt path")
            cumulative_performance_ref = \
                cumulative_composition.seal_cumulative_performance(
                    Path(performance_path), cumulative_performance)
        return SealedScreen(receipt_path=str(result_path), result_sha256=str(raw["result_sha256"]), effect_fraction=target_effect, classification=str(projection["stage"]), baseline_sha256=str(raw["baseline_sha256"]), source_proof_sha256=bundle.correctness["file_sha256"], dispatch_proof_sha256=bundle.attribution["file_sha256"], exact_attribution_effect_fraction=exact_effect, target_runtime_effect_fraction=target_effect, stages=("materialized", "built", "correctness", "attribution", "measurement_graphs_off_screen", "target_runtime_graphs_on_screen"), build_identity_sha256=build_identity_sha256, correctness_receipt_sha256=bundle.correctness["file_sha256"], attribution_receipt_sha256=bundle.attribution["file_sha256"], graphs_off_receipt_sha256=graphs_off_file_sha256, graphs_on_receipt_sha256=graphs_on_file_sha256, composition_build_pair=build.composition_build_pair, composition_correctness=composition_correctness, composition_comparison=composition_comparison, cumulative_performance=cumulative_performance, cumulative_performance_ref=cumulative_performance_ref)


@dataclass(frozen=True)
class ControllerConfig:
    output_root: Path
    max_iterations: int = 1
    nomination_threshold: float = 0.03
    dry_run: bool = False
    # This is the AutoKernel evidence root that owns the canonical progression
    # projection.  The controller state root is never silently treated as a
    # second evidence tree in live mode.
    evidence_root: Path | None = None
    # Sealed deployment data, never planner-authored prose.  Keeping the hash
    # separately makes a changed profile/source brief a durable-resume refusal.
    planner_context: Mapping[str, Any] | None = None
    planner_context_sha256: str | None = None
    production_base_commit: str | None = None
    instrument_commit: str | None = None
    campaign_id: str = "ak-discovery"
    experiment_template_registry_sha256: str | None = None
    admission_corpus_sha256: str | None = None
    admission_corpus_version: str | None = None
    # The sealed deployment file is the authority for repository paths/refs as
    # well as the hashes separately carried below.  Durable state records this
    # one canonical identity so a resume cannot silently switch checkout roots.
    deployment_identity_sha256: str | None = None
    hypothesis_portfolio: hypothesis_portfolio.Portfolio | None = None
    hypothesis_portfolio_sha256: str | None = None
    carry_forward: Mapping[str, Any] | None = None
    carry_forward_sha256: str | None = None
    preauthored_continuations: Mapping[
        str, preauthored_continuation.PreauthoredContinuation] | None = None
    def __post_init__(self) -> None:
        if (not self.output_root.is_absolute() or not 1 <= self.max_iterations <= 1000
                or isinstance(self.nomination_threshold, bool)
                or not math.isfinite(float(self.nomination_threshold))
                or self.nomination_threshold <= 0
                or self.evidence_root is not None and not self.evidence_root.is_absolute()
                or (self.planner_context is None) != (self.planner_context_sha256 is None)
                or self.planner_context_sha256 is not None and not HASH.fullmatch(self.planner_context_sha256)
                or (self.production_base_commit is None) != (self.instrument_commit is None)
                or self.production_base_commit is not None and not all(
                    isinstance(value, str) and len(value) == 40
                    and all(ch in "0123456789abcdef" for ch in value)
                    for value in (self.production_base_commit, self.instrument_commit))
                or not self.campaign_id.startswith("ak-")
                or self.experiment_template_registry_sha256 is not None and not HASH.fullmatch(self.experiment_template_registry_sha256)
                or self.admission_corpus_sha256 is not None and not HASH.fullmatch(self.admission_corpus_sha256)
                or self.admission_corpus_version is not None and not re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", self.admission_corpus_version)):
            raise DiscoveryControllerError("invalid controller config")
        if self.deployment_identity_sha256 is not None and not HASH.fullmatch(self.deployment_identity_sha256):
            raise DiscoveryControllerError("invalid sealed deployment identity")
        if ((self.hypothesis_portfolio is None) !=
                (self.hypothesis_portfolio_sha256 is None)
                or self.hypothesis_portfolio_sha256 is not None
                and not HASH.fullmatch(self.hypothesis_portfolio_sha256)):
            raise DiscoveryControllerError("invalid sealed hypothesis portfolio authority")
        if (self.hypothesis_portfolio is not None
                and (not isinstance(self.hypothesis_portfolio, hypothesis_portfolio.Portfolio)
                     or self.hypothesis_portfolio.sha256 != self.hypothesis_portfolio_sha256)):
            raise DiscoveryControllerError(
                "controller portfolio must be one loader-validated immutable authority")
        carry_keys = {
            "schema", "predecessor_state_file_sha256",
            "predecessor_journal_file_sha256",
            "predecessor_state_semantic_sha256", "portfolio_outcomes",
            "candidate_semantic_sha256", "candidate_patch_sha256",
            "cross_campaign_candidate_sha256",
            "attribution_expectation_erratum", "carry_forward_sha256",
        }
        expected_outcomes = {
            "akh-v2-q5-type-specific-dequant": "nominated",
            "akh-v2-q8-quantizer-new-mechanism": "retire",
            "akh-v2-fa-gqa7-pair-tail": "bounded_authoring_skip",
            "akh-v2-rms-direct-load-reduction": "bounded_authoring_skip",
        }
        if ((self.carry_forward is None) != (self.carry_forward_sha256 is None)
                or self.carry_forward_sha256 is not None
                and not HASH.fullmatch(self.carry_forward_sha256)
                or self.carry_forward is not None
                and (set(self.carry_forward) != carry_keys
                     or self.carry_forward.get("schema") !=
                     "epyc.autokernel.discovery_carry_forward.v2"
                     or self.carry_forward.get("portfolio_outcomes") !=
                        expected_outcomes
                     or any(not isinstance(self.carry_forward.get(key), str)
                            or not HASH.fullmatch(self.carry_forward[key])
                            for key in (
                                "predecessor_state_file_sha256",
                                "predecessor_journal_file_sha256",
                                "predecessor_state_semantic_sha256"))
                     or any(not isinstance(self.carry_forward.get(key), list)
                            or self.carry_forward[key] !=
                               sorted(set(self.carry_forward[key]))
                            or not self.carry_forward[key]
                            or any(not isinstance(value, str)
                                   or not HASH.fullmatch(value)
                                   for value in self.carry_forward[key])
                            for key in (
                                "candidate_semantic_sha256",
                                "candidate_patch_sha256",
                                "cross_campaign_candidate_sha256"))
                     or tuple(len(self.carry_forward[key]) for key in (
                         "candidate_semantic_sha256", "candidate_patch_sha256",
                         "cross_campaign_candidate_sha256")) != (13, 8, 8)
                     or self.carry_forward.get(
                         "attribution_expectation_erratum") !=
                        _expected_q5_lds0_attribution_erratum()
                     or self.carry_forward.get("carry_forward_sha256") !=
                        self.carry_forward_sha256
                     or _sha({key: value for key, value in self.carry_forward.items()
                              if key != "carry_forward_sha256"}) !=
                        self.carry_forward_sha256)):
            raise DiscoveryControllerError("invalid predecessor carry-forward authority")
        continuations = self.preauthored_continuations
        requires_q5_continuation = bool(
            self.hypothesis_portfolio is not None
            and any(
                isinstance(record, Mapping)
                and record.get("hypothesis_id") ==
                    "akh-v2-q5-onewave-preauthored"
                and isinstance(record.get("current_bundle_eligibility"), Mapping)
                and record["current_bundle_eligibility"].get("eligible") is True
                and tuple(record["current_bundle_eligibility"].get(
                    "template_ids", ())) ==
                    ("cuda-mmvq-q5-onewave-continuation-v1",)
                for record in self.hypothesis_portfolio.hypotheses))
        if ((requires_q5_continuation != (continuations is not None))
                or continuations is not None
                and (not isinstance(continuations, Mapping)
                     or set(continuations) != {"akh-v2-q5-onewave-preauthored"}
                     or any(not isinstance(value,
                                           preauthored_continuation.PreauthoredContinuation)
                            or value.hypothesis_id != key
                            for key, value in continuations.items()))):
            raise DiscoveryControllerError(
                "invalid preauthored continuation authority")
        if (requires_q5_continuation
                and (not isinstance(self.planner_context, Mapping)
                     or self.planner_context.get(
                         "preauthored_continuation_sha256") !=
                        continuations["akh-v2-q5-onewave-preauthored"].sha256
                     or self.planner_context.get(
                         "preauthored_source_backed_diff_sha256") !=
                        continuations[
                            "akh-v2-q5-onewave-preauthored"].source_backed_diff_sha256)):
            raise DiscoveryControllerError(
                "planner context differs from preauthored continuation authority")
        sealed = (self.planner_context_sha256, self.production_base_commit,
                  self.instrument_commit, self.experiment_template_registry_sha256,
                  self.admission_corpus_sha256, self.admission_corpus_version,
                  self.deployment_identity_sha256, self.carry_forward_sha256)
        if (not self.dry_run and any(value is not None for value in sealed)
                and not all(value is not None for value in sealed)):
            raise DiscoveryControllerError("live sealed controller configuration has incomplete deployment authority")


class DurableState:
    def __init__(self, root: Path) -> None:
        self.root=root; self.book=journal.Journal(str(root / "journal")); self.book.initialize(); self.path=root / "state.json"
    def load(self) -> dict[str, Any]:
        if not self.path.exists(): return {"schema": SCHEMA, "authority": AUTHORITY, "roster": sealed_roster(), "iterations": [], "next": 1, "scientific_attempts": 0, "complete": False}
        value=_read_object(self.path, self.root); _require_roster(value.get("roster", {}))
        if value.get("schema") in {
                "epyc.autokernel.discovery_controller.v6",
                "epyc.autokernel.discovery_controller.v7"}:
            raise DiscoveryControllerError(
                "legacy controller state lacks cumulative composition authority; initialize a fresh v8 campaign")
        if value.get("schema") != SCHEMA or value.get("authority") != AUTHORITY: raise DiscoveryControllerError("wrong controller journal")
        declared=value.get("state_sha256")
        if not isinstance(declared,str) or declared != _sha({k:v for k,v in value.items() if k!="state_sha256"}): raise DiscoveryControllerError("durable controller state hash mismatch")
        return value
    def save(self, state: dict[str, Any], phase: str) -> None:
        state["updated_at"]=_now(); state["state_sha256"]=_sha({k:v for k,v in state.items() if k!="state_sha256"}); _atomic(self.path,state)
        self.book.append(journal.KIND_STOP_STATE,{"state":f"discovery_{phase}","controller_state_sha256":state["state_sha256"]})
    def run_lock(self):
        self.root.mkdir(parents=True,exist_ok=True)
        handle=(self.root / "controller.run.lock").open("a+")
        try:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close(); raise DiscoveryControllerError("another discovery controller owns this output root") from exc
        return handle


def _tracker(store: DurableState) -> hypotheses.HypothesisTracker:
    return hypotheses.HypothesisTracker(journal_=store.book, root=str(store.root / "hypotheses"), campaign_id="ak-discovery")


def _memory_block(tracker: hypotheses.HypothesisTracker, turn: int) -> Mapping[str, Any]:
    ledger=do_not_repeat.compile_for_tracker(tracker); return do_not_repeat.planner_round_block(tracker, ledger, round_id=f"discovery-{turn}")


def _ensure_question(tracker: hypotheses.HypothesisTracker, item: PlannedCandidate,
                     portfolio_binding: Mapping[str, Any] | None = None,
                     preauthored_authority: Mapping[str, Any] | None = None) -> None:
    """Open the exact question whose campaign-ledger DNR gate will authorize it.

    Legacy/generic callers retain their original regime verbatim: an old question
    which did not declare a structural mechanism must continue to read
    ``COULD_NOT_CHECK`` rather than acquiring authority retroactively.  A sealed
    portfolio candidate is different.  The controller already owns its exact
    manifest mechanism, so omitting that key would make every new AutoKernel
    authorization structurally incomparable to the campaign ledger.  On this path the
    mechanism is mandatory, controller-derived, and any actor-authored disagreement is
    refused rather than silently overwritten.
    """
    regime = dict(item.regime)
    if portfolio_binding is not None:
        mechanism = portfolio_binding.get("mechanism_id")
        expected_source_mechanism = (
            "q5_0_one_wave_per_output_block"
            if preauthored_authority is not None else mechanism)
        if (not isinstance(mechanism, str) or not HASH.fullmatch(mechanism)
                or item.source_manifest.mechanism_id !=
                   expected_source_mechanism):
            raise DiscoveryControllerError(
                "portfolio candidate lacks its controller-owned structural mechanism")
        declared = regime.get("mechanism")
        if declared is not None and declared != mechanism:
            raise DiscoveryControllerError(
                "portfolio candidate regime disagrees with its controller-owned mechanism")
        regime["mechanism"] = mechanism
    origin = hypotheses.ORIGIN_PLANNER
    author = "gpt-5.6-sol"
    source = {"manifest_sha256": item.source_manifest_sha256}
    if preauthored_authority is not None:
        required = {
            "schema", "hypothesis_id", "carrier_sha256",
            "authoring_turn",
            "source_backed_diff_sha256", "source_manifest_sha256",
            "candidate_semantic_sha256", "cross_campaign_candidate_sha256",
            "origin", "author", "historical_commit",
            "modern_governed_correctness_required", "receipt_sha256",
        }
        if (set(preauthored_authority) != required
                or preauthored_authority.get("schema") !=
                   "epyc.autokernel.preauthored_checkpoint.v1"
                or preauthored_authority.get("hypothesis_id") !=
                   item.hypothesis_id
                or preauthored_authority.get("source_manifest_sha256") !=
                   item.source_manifest_sha256
                or item.proposal.get("proposal_id") !=
                   f"akp-discovery-{preauthored_authority.get('authoring_turn')}"
                or preauthored_authority.get("candidate_semantic_sha256") !=
                   _candidate_semantic_identity(item)
                or preauthored_authority.get(
                    "cross_campaign_candidate_sha256") !=
                   _cross_campaign_candidate_identity(item)
                or preauthored_authority.get("origin") !=
                   hypotheses.ORIGIN_IMPORT
                or preauthored_authority.get("author") !=
                   "reviewed-eb26918-continuation"
                or preauthored_authority.get(
                    "modern_governed_correctness_required") is not True
                or preauthored_authority.get("receipt_sha256") != _sha({
                    key: value for key, value in preauthored_authority.items()
                    if key != "receipt_sha256"})):
            raise DiscoveryControllerError(
                "preauthored hypothesis provenance authority changed")
        origin = hypotheses.ORIGIN_IMPORT
        author = "reviewed-eb26918-continuation"
        source.update({
            "preauthored_continuation_sha256":
                preauthored_authority["carrier_sha256"],
            "historical_commit": preauthored_authority["historical_commit"],
            "source_backed_diff_sha256":
                preauthored_authority["source_backed_diff_sha256"],
        })
    question=hypotheses.Hypothesis(hypothesis_id=item.hypothesis_id, statement=item.statement, falsifier=item.falsifier, origin=origin, author=author, regime=regime, source=source)
    try:
        tracker.open_hypothesis(question)
    except hypotheses.HypothesisAlreadyTracked:
        if (preauthored_authority is not None
                and tracker.get(item.hypothesis_id).hypothesis != question):
            raise DiscoveryControllerError(
                "preauthored hypothesis already exists with different provenance")

def _record_attempt_once(tracker: hypotheses.HypothesisTracker, item: PlannedCandidate, proposal_id: str, result: SealedScreen) -> None:
    ref=f"sha256:{result.result_sha256}"
    for event in tracker.read().events:
        attempt=event.payload.get("attempt") if event.kind==hypotheses.EVENT_ATTEMPTED else None
        if isinstance(attempt,Mapping) and attempt.get("hypothesis_id")==item.hypothesis_id and ref in attempt.get("refs",[]): return
    tracker.note_attempt(item.hypothesis_id,proposal_id=proposal_id,disposition=result.classification,bears_on_falsifier=True,note=f"sealed screen {result.result_sha256}; effect={result.effect_fraction:.9g}",refs=(ref,))


def _portfolio_binding(config: ControllerConfig,
                       record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one eligible scientific question into exact actor authority."""
    if config.hypothesis_portfolio is None or config.hypothesis_portfolio_sha256 is None:
        raise DiscoveryControllerError("controller lacks a sealed hypothesis portfolio")
    target = record.get("target")
    eligibility = record.get("current_bundle_eligibility")
    mechanism = record.get("mechanism")
    policy = record.get("decision_policy")
    falsifiers = record.get("falsifiers")
    if (not isinstance(target, Mapping) or not isinstance(eligibility, Mapping)
            or eligibility.get("eligible") is not True
            or not isinstance(mechanism, Mapping)
            or not isinstance(policy, Mapping)
            or not isinstance(falsifiers, (list, tuple)) or not falsifiers
            or record.get("primary_falsifier") not in falsifiers
            or not isinstance(record.get("regime"), Mapping)):
        raise DiscoveryControllerError("eligible portfolio record is incomplete")
    files = target.get("source_files")
    symbols = target.get("source_symbols")
    templates = eligibility.get("template_ids")
    policy_keys = {"metric", "frame_id", "effect_unit", "continuation_floor_pct",
                   "nomination_floor_pct", "min_replication_effect_pct",
                   "required_replications", "max_replication_spread_pct",
                   "sign_policy", "conflict_policy", "max_distinct_candidates",
                   "terminal_rule"}
    facets = mechanism.get("facets") if isinstance(mechanism, Mapping) else None
    if (not isinstance(files, (list, tuple)) or not 1 <= len(files) <= 2
            or not isinstance(symbols, (list, tuple)) or not symbols
            or not all(isinstance(value, str) and value for value in files + symbols)
            or not isinstance(templates, (list, tuple)) or len(templates) != 1
            or target.get("template_intent") != templates[0]
            or not HASH.fullmatch(str(mechanism.get("fingerprint_sha256")))
            or not isinstance(facets, Mapping)
            or facets.get("change_class") not in schemas.CHANGE_CLASSES
            or facets.get("change_class") == "parameter"
            or not isinstance(policy.get("max_distinct_candidates"), int)
            or isinstance(policy.get("max_distinct_candidates"), bool)
            or not 1 <= policy["max_distinct_candidates"] <= 8
            or set(policy) != policy_keys
            or policy.get("effect_unit") != "relative_percent"
            or policy.get("required_replications") != 2
            or policy.get("sign_policy") != "all_positive"
            or policy.get("conflict_policy") not in {"retire", "retain_inconclusive"}
            or policy.get("terminal_rule") not in {"retire", "retain_inconclusive",
                                                    "needs_review"}):
        raise DiscoveryControllerError(
            "eligible portfolio record is not expressible by one exact reviewed template")
    symbol_authority = (config.planner_context or {}).get(
        "template_symbol_authority", {}).get(templates[0])
    if (not isinstance(symbol_authority, Mapping)
            or set(symbol_authority) != set(files)):
        raise DiscoveryControllerError(
            "eligible portfolio record lacks exact per-file symbol authority")
    symbols_by_file = {
        path: sorted(set(symbols) & set(symbol_authority[path]))
        for path in files
    }
    if (any(not values for values in symbols_by_file.values())
            or set().union(*(set(values) for values in symbols_by_file.values()))
               != set(symbols)
            or any(sum(symbol in values for values in symbols_by_file.values()) != 1
                   for symbol in symbols)):
        raise DiscoveryControllerError(
            "eligible portfolio symbols do not map exactly to reviewed files")
    binding = {
        "portfolio_sha256": config.hypothesis_portfolio_sha256,
        "record_sha256": hypothesis_portfolio.content_sha256(record),
        "hypothesis_id": record.get("hypothesis_id"),
        "statement": record.get("statement"),
        "falsifier": record["primary_falsifier"],
        "mechanism_id": mechanism["fingerprint_sha256"],
        "change_class": facets["change_class"],
        "regime": dict(record["regime"]),
        "target_files": list(files),
        "target_symbols": list(symbols),
        "target_symbols_by_file": symbols_by_file,
        "template_id": templates[0],
        "decision_policy": dict(policy),
    }
    dispatch_authority = (config.planner_context or {}).get(
        "portfolio_dispatch_authority", {})
    rows = dispatch_authority.get(binding["hypothesis_id"])
    if not isinstance(rows, list):
        raise DiscoveryControllerError(
            "eligible portfolio record lacks deployed raw dispatch authority")
    binding["expected_dispatch"] = [dict(row) for row in rows]
    AuthoringAssignment(
        campaign_id="ak-portfolio-validation", proposal_id="akp-portfolio-validation",
        candidate_id="akc-portfolio-validation", production_base_commit="0" * 40,
        instrument_commit="0" * 40, portfolio_binding=binding)
    return binding


def _select_portfolio_binding(state: Mapping[str, Any],
                              config: ControllerConfig) -> dict[str, Any] | None:
    if config.hypothesis_portfolio is None:
        return None
    records = config.hypothesis_portfolio.hypotheses
    eligible = [row for row in records if isinstance(row, Mapping)
                and isinstance(row.get("current_bundle_eligibility"), Mapping)
                and row["current_bundle_eligibility"].get("eligible") is True]
    try:
        eligible.sort(key=lambda row: (int(row["priority"]["rank"]),
                                       str(row["hypothesis_id"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise DiscoveryControllerError("eligible portfolio priority is malformed") from exc
    scheduled: list[tuple[int, int, str, dict[str, Any]]] = []
    authoring_failures = _validate_portfolio_authoring_failures(state)
    for record in eligible:
        binding = _portfolio_binding(config, record)
        if (binding["hypothesis_id"] in state.get("portfolio_terminals", {})
                or binding["hypothesis_id"] in state.get("portfolio_skips", {})
                or binding["hypothesis_id"] in state.get(
                    "portfolio_validations", {})):
            continue
        attempts: set[str] = set()
        for row in state["iterations"]:
            if (not isinstance(row, Mapping)
                    or row.get("portfolio_hypothesis_id") != binding["hypothesis_id"]
                    or not _row_spends_scientific_budget(row)):
                continue
            identity = (row.get("candidate_semantic_sha256")
                        or row.get("source_manifest_sha256"))
            if not isinstance(identity, str) or not HASH.fullmatch(identity):
                raise DiscoveryControllerError(
                    "scientific portfolio row lacks candidate identity")
            attempts.add(identity)
        if len(attempts) >= binding["decision_policy"]["max_distinct_candidates"]:
            continue
        failures = authoring_failures.get(binding["hypothesis_id"], 0)
        if (not isinstance(failures, int) or isinstance(failures, bool)
                or failures < 0):
            raise DiscoveryControllerError(
                "portfolio authoring-failure count is malformed")
        # Opportunity-cost scheduling: give every lower-exposure scientific
        # question a turn before draining another candidate from one family.
        # Rank remains the deterministic evidence/ROI tiebreaker.  Scientific
        # S2 reuses the same semantic candidate identity, while non-scientific
        # authoring failures still represent actor time and therefore yield the
        # lane after one failure.
        exposure = len(attempts) + failures
        scheduled.append((exposure, int(record["priority"]["rank"]),
                          binding["hypothesis_id"], binding))
    if not scheduled:
        return None
    return min(scheduled, key=lambda item: item[:3])[3]


def _derived_portfolio_authoring_failures(
        state: Mapping[str, Any]) -> dict[str, int]:
    authoring_statuses = {
        "planner_refused", "critic_revise", "critic_reject",
        "screen_refused", "authoring_refused", "planner_contract_refused",
        "candidate_semantic_repeat_refused", "authorization_refused",
    }
    iterations = state.get("iterations", [])
    if not isinstance(iterations, list):
        raise DiscoveryControllerError("durable iterations are malformed")
    derived: dict[str, int] = {}
    for row in iterations:
        if (isinstance(row, Mapping)
                and row.get("status") in authoring_statuses
                and isinstance(row.get("portfolio_hypothesis_id"), str)):
            hypothesis_id = row["portfolio_hypothesis_id"]
            derived[hypothesis_id] = derived.get(hypothesis_id, 0) + 1
    return derived


def _validate_portfolio_authoring_failures(
        state: Mapping[str, Any]) -> Mapping[str, int]:
    declared = state.get("portfolio_authoring_failures", {})
    derived = _derived_portfolio_authoring_failures(state)
    if (not isinstance(declared, Mapping) or dict(declared) != derived
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value <= 0 for value in declared.values())):
        raise DiscoveryControllerError(
            "portfolio authoring-failure accounting differs from durable rows")
    return derived


def _validate_portfolio_candidate(
        item: PlannedCandidate, binding: Mapping[str, Any],
        portfolio: hypothesis_portfolio.Portfolio,
        carry_forward: Mapping[str, Any] | None = None) -> None:
    """Refuse any actor attempt to rename or expand a reviewed question."""
    if not isinstance(portfolio, hypothesis_portfolio.Portfolio):
        raise DiscoveryControllerError("portfolio candidate lacks typed portfolio authority")
    intent = item.experiment_intent
    manifest = item.source_manifest
    if (item.hypothesis_id != binding["hypothesis_id"]
            or item.statement != binding["statement"]
            or item.falsifier != binding["falsifier"]
            or dict(item.regime) != dict(binding["regime"])
            or manifest.mechanism_id != (
                "q5_0_one_wave_per_output_block"
                if item.hypothesis_id == "akh-v2-q5-onewave-preauthored"
                else binding["mechanism_id"])
            or manifest.change_class != binding["change_class"]
            or item.proposal.get("change_class") != binding["change_class"]
            or tuple(manifest.declared_files) != tuple(binding["target_files"])
            or {path: sorted(manifest.declared_symbols.get(path, ()))
                for path in binding["target_files"]} !=
               {path: sorted(symbols) for path, symbols in
                binding["target_symbols_by_file"].items()}
            or intent is None
            or intent.template_id != binding["template_id"]
            or intent.target_symbol not in binding["target_symbols"]
            or [asdict(row) for row in intent.expected_dispatch]
               != list(binding["expected_dispatch"])):
        raise DiscoveryControllerError(
            "planner candidate differs from its controller-owned portfolio assignment")
    if carry_forward is not None:
        outcomes = carry_forward.get("portfolio_outcomes")
        exact = carry_forward.get("candidate_semantic_sha256")
        patches = carry_forward.get("candidate_patch_sha256")
        stable = carry_forward.get("cross_campaign_candidate_sha256")
        erratum = carry_forward.get("attribution_expectation_erratum")
        if (not isinstance(outcomes, Mapping)
                or not all(isinstance(values, list) for values in
                           (exact, patches, stable))
                or erratum != _expected_q5_lds0_attribution_erratum()):
            raise DiscoveryControllerError(
                "predecessor carry-forward candidate authority is malformed")
        if item.hypothesis_id in outcomes:
            raise DiscoveryControllerError(
                "planner selected a predecessor-terminal hypothesis")
        semantic = _candidate_semantic_identity(item)
        patch = item.source_manifest.patch_sha256
        cross_campaign = _cross_campaign_candidate_identity(item)
        corrected_q5_retry = (
            item.hypothesis_id == erratum["hypothesis_id"]
            and semantic == erratum["candidate_semantic_sha256"]
            and patch == erratum["candidate_patch_sha256"]
            and cross_campaign == erratum["cross_campaign_candidate_sha256"]
            and erratum["scientific_budget_spent"] is False
            and erratum["do_not_repeat"] is False
            and erratum["replay_authorized"] is True)
        if (not corrected_q5_retry
                and (semantic in exact or patch in patches
                     or cross_campaign in stable)):
            raise DiscoveryControllerError(
                "planner candidate repeats predecessor source semantics")


def _portfolio_exact_dnr_check(config: ControllerConfig, item: PlannedCandidate,
                               binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical, candidate-bound receipt before critic or authorization.

    This is intentionally separate from the campaign ledger.  The portfolio is sealed
    input authority and can answer an exact mechanism/regime question directly; the
    campaign ledger is derived runtime memory and may honestly answer
    ``COULD_NOT_CHECK``.  Conflating the two outcomes makes a portfolio refusal vanish
    into a generic authorization reason on restart.
    """
    portfolio = config.hypothesis_portfolio
    semantic_sha256 = config.hypothesis_portfolio_sha256
    if (not isinstance(portfolio, hypothesis_portfolio.Portfolio)
            or not isinstance(semantic_sha256, str)
            or portfolio.sha256 != semantic_sha256):
        raise DiscoveryControllerError(
            "portfolio exact-DNR check lacks sealed semantic authority")
    mechanism_id = binding.get("mechanism_id")
    expected_source_mechanism = (
        "q5_0_one_wave_per_output_block"
        if item.hypothesis_id == "akh-v2-q5-onewave-preauthored"
        else mechanism_id)
    if (not isinstance(mechanism_id, str) or not HASH.fullmatch(mechanism_id)
            or item.source_manifest.mechanism_id != expected_source_mechanism):
        raise DiscoveryControllerError(
            "portfolio exact-DNR check lacks the controller-owned candidate mechanism")
    regime = dict(item.regime)
    if regime != dict(binding.get("regime") or {}):
        raise DiscoveryControllerError(
            "portfolio exact-DNR check candidate regime differs from assignment")
    matched: list[str] = []
    for index, dnr in enumerate(portfolio.do_not_repeat):
        if not isinstance(dnr, Mapping):
            raise DiscoveryControllerError("portfolio DNR record is malformed")
        dnr_id = dnr.get("dnr_id")
        mechanism = dnr.get("mechanism")
        dnr_regime = dnr.get("regime")
        if (not isinstance(dnr_id, str) or not dnr_id.startswith("dnr-")
                or not isinstance(mechanism, Mapping)
                or not HASH.fullmatch(str(mechanism.get("fingerprint_sha256")))
                or not isinstance(dnr_regime, Mapping)):
            raise DiscoveryControllerError(
                f"portfolio DNR record {index} lacks exact mechanism/regime identity")
        if (mechanism_id == mechanism["fingerprint_sha256"]
                and regime == dict(dnr_regime)):
            matched.append(dnr_id)
    body: dict[str, Any] = {
        "schema": PORTFOLIO_DNR_CHECK_SCHEMA,
        "portfolio_semantic_sha256": semantic_sha256,
        "portfolio_hypothesis_id": binding.get("hypothesis_id"),
        "candidate_source_manifest_sha256": item.source_manifest_sha256,
        "candidate_mechanism_id": mechanism_id,
        "canonical_regime_sha256": schemas.content_hash(regime),
        "matched_dnr_ids": sorted(set(matched)),
        "outcome": schemas.FAIL if matched else schemas.PASS,
    }
    body["receipt_sha256"] = schemas.content_hash(body)
    return body


def _revalidate_portfolio_checkpoint(config: ControllerConfig,
                                     item: PlannedCandidate,
                                     row: Mapping[str, Any]) -> None:
    """Fail closed when a new portfolio checkpoint omitted or changed its DNR receipt."""
    if config.hypothesis_portfolio is None:
        # Legacy generic campaigns never had this receipt.  Their campaign-ledger
        # COULD_NOT_CHECK semantics are preserved rather than rewritten on resume.
        return
    binding = row.get("portfolio_binding")
    if not isinstance(binding, Mapping):
        raise DiscoveryControllerError(
            "portfolio pending candidate lacks controller-owned binding")
    _validate_portfolio_candidate(
        item, binding, config.hypothesis_portfolio, config.carry_forward)
    expected = _portfolio_exact_dnr_check(config, item, binding)
    actual = row.get("portfolio_exact_dnr_check")
    if not isinstance(actual, Mapping) or dict(actual) != expected:
        raise DiscoveryControllerError(
            "portfolio pending candidate DNR receipt is missing or changed")
    if expected["outcome"] != schemas.PASS:
        raise DiscoveryControllerError(
            "portfolio pending candidate exactly matches a sealed DNR")
    semantic = row.get("candidate_semantic_sha256")
    if semantic is not None and semantic != _candidate_semantic_identity(item):
        raise DiscoveryControllerError(
            "portfolio pending candidate semantic identity changed")


def _bind_campaign_ledger_outcome(row: dict[str, Any],
                                  authorization: hypotheses.ClaimAuthorization) -> None:
    """Keep runtime-ledger disposition distinct from the sealed portfolio receipt."""
    expected = authorization.do_not_repeat_outcome
    reasons = list(authorization.do_not_repeat_reasons)
    prior = row.get("campaign_ledger_dnr_outcome")
    prior_reasons = row.get("campaign_ledger_dnr_reasons")
    if prior is not None and (prior != expected or prior_reasons != reasons):
        raise DiscoveryControllerError(
            "campaign-ledger DNR outcome differs from durable authorization")
    row["campaign_ledger_dnr_outcome"] = expected
    row["campaign_ledger_dnr_reasons"] = reasons


def _context(state: Mapping[str, Any], tracker: hypotheses.HypothesisTracker, turn: int,
             config: ControllerConfig,
             portfolio_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    prior = []
    for row in state["iterations"]:
        if not isinstance(row.get("result_sha256"), str):
            continue
        prior.append({key: row.get(key) for key in (
            "result_sha256", "status", "effect_fraction", "series_effect_fraction",
            "source_manifest_sha256", "series_key", "evidence", "statement",
            "falsifier", "experiment_intent", "mechanism_id", "target_surface",
            "target_symbol")})
    assignment = None
    if config.production_base_commit is not None or portfolio_binding is not None:
        assignment = AuthoringAssignment(
            campaign_id=config.campaign_id, proposal_id=f"akp-discovery-{turn}",
            candidate_id=f"akc-discovery-{turn}",
            production_base_commit=config.production_base_commit or "0" * 40,
            instrument_commit=(config.instrument_commit
                               or config.production_base_commit or "0" * 40),
            portfolio_binding=portfolio_binding).to_dict()
    prior_refusals = [
        {key: row.get(key) for key in (
            "turn", "status", "reason", "portfolio_hypothesis_id",
            "context_sha256")}
        for row in state["iterations"] if row.get("status") == "planner_refused"
    ][-8:]
    return {"authority": AUTHORITY, "turn":turn, "roster":sealed_roster(),
            "planner_context": config.planner_context,
            "planner_context_sha256": config.planner_context_sha256,
            "admission_corpus_sha256": config.admission_corpus_sha256,
            "admission_corpus_version": config.admission_corpus_version,
            "deployment_identity_sha256": config.deployment_identity_sha256,
            "hypothesis_portfolio_sha256": config.hypothesis_portfolio_sha256,
            "authoring_assignment": assignment,
            "prior_authoring_refusals": prior_refusals,
            "prior_results": prior, "do_not_repeat":_memory_block(tracker,turn)}


def _pending_item(item: PlannedCandidate) -> dict[str, Any]:
    manifest = item.source_manifest
    raw_manifest=source_candidate.source_patch_manifest_bytes(manifest)
    intent = (None if item.experiment_intent is None else
              json.loads(json.dumps(asdict(item.experiment_intent),
                                    sort_keys=True)))
    composition = (None if item.composition_plan is None else
                   item.composition_plan.to_dict())
    return {"hypothesis_id": item.hypothesis_id, "statement": item.statement,
            "falsifier": item.falsifier, "regime": dict(item.regime),
            "proposal": dict(item.proposal), "source_manifest_sha256": item.source_manifest_sha256,
            "experiment_intent": intent,
            "composition_plan": composition,
            "manifest": {"campaign_id":manifest.campaign_id,"proposal_id":manifest.proposal_id,
                "candidate_id":manifest.candidate_id,"source_tree":manifest.source_tree,
                "production_base_commit":manifest.production_base_commit,"instrument_commit":manifest.instrument_commit,
                "change_class":manifest.change_class,"declared_files":list(manifest.declared_files),
                "declared_symbols":{k:list(v) for k,v in manifest.declared_symbols.items()},
                "mechanism_id":manifest.mechanism_id,"patch_sha256":manifest.patch_sha256,
                "patch_base64":base64.b64encode(manifest.patch_bytes).decode("ascii")},
            "manifest_raw_base64":base64.b64encode(raw_manifest).decode("ascii"),"manifest_file_sha256":hashlib.sha256(raw_manifest).hexdigest(),"patch_bundle_sha256":manifest.patch_bundle_sha256}


def _preauthored_candidate(
        config: ControllerConfig, binding: Mapping[str, Any],
        turn: int) -> PlannedCandidate:
    """Reconstruct the one reviewed Q5 candidate without an actor."""
    hypothesis_id = binding.get("hypothesis_id")
    continuations = config.preauthored_continuations
    if (hypothesis_id != "akh-v2-q5-onewave-preauthored"
            or continuations is None or hypothesis_id not in continuations):
        raise DiscoveryControllerError(
            "portfolio binding is not an authorized preauthored continuation")
    continuation = continuations[hypothesis_id]
    if (continuation.sha256 != config.planner_context.get(
            "preauthored_continuation_sha256")
            or continuation.source_backed_diff_sha256 != hashlib.sha256(
                continuation.source_backed_diff.encode("utf-8")).hexdigest()
            or continuation.mechanism_id !=
               "q5_0_one_wave_per_output_block"
            or binding.get("mechanism_id") !=
               "7d88a2725aab2276202324ecef22f2414a1d893f9081061314622c82a7c28919"
            or continuation.change_class != binding.get("change_class")
            or [continuation.source_file] != binding.get("target_files")
            or list(continuation.declared_symbols) !=
               sorted(binding.get("target_symbols", []))
            or binding.get("target_symbols_by_file") != {
                continuation.source_file: list(continuation.declared_symbols)}
            or continuation.template_id != binding.get("template_id")
            or [dict(row) for row in continuation.expected_dispatch] !=
               binding.get("expected_dispatch")):
        raise DiscoveryControllerError(
            "preauthored continuation differs from its portfolio/config binding")
    proposal_id = f"akp-discovery-{turn}"
    candidate_id = f"akc-discovery-{turn}"
    try:
        manifest = source_candidate.source_backed_source_patch_manifest(
            campaign_id=config.campaign_id, proposal_id=proposal_id,
            candidate_id=candidate_id, source_tree=continuation.source_tree,
            production_base_commit=config.production_base_commit,
            instrument_commit=config.instrument_commit,
            change_class=continuation.change_class,
            declared_files=(continuation.source_file,),
            declared_symbols={
                continuation.source_file: continuation.declared_symbols},
            mechanism_id=continuation.mechanism_id,
            patch_sha256=continuation.patch_sha256,
            patch_bytes=continuation.patch_bytes,
            source_backed_diff=continuation.source_backed_diff)
    except (TypeError, source_candidate.SourceCandidateError) as exc:
        raise DiscoveryControllerError(
            "preauthored source manifest cannot be reconstructed") from exc
    changed_lines = sum(
        1 for line in manifest.patch_text.splitlines()
        if line.startswith(("+", "-"))
        and not line.startswith(("+++", "---")))
    proposal = {
        "proposal_id": proposal_id,
        "change_class": continuation.change_class,
        "change": {
            "files_and_symbols": [
                f"{continuation.source_file}:{symbol}"
                for symbol in continuation.declared_symbols],
            "estimated_diff_size": changed_lines,
        },
        "preauthored_continuation_sha256": continuation.sha256,
    }
    intent = GpuSourceExperimentIntent(
        template_id=continuation.template_id,
        target_surface="gpu_decode", target_symbol="calc_nwarps",
        correctness_id=continuation.correctness_id,
        dispatch_id=continuation.dispatch_id,
        expected_dispatch=tuple(
            BoundedDispatchExpectation(**dict(row))
            for row in continuation.expected_dispatch))
    return PlannedCandidate(
        hypothesis_id=hypothesis_id, statement=binding["statement"],
        falsifier=binding["falsifier"], regime=dict(binding["regime"]),
        proposal=proposal, source_manifest=manifest,
        source_manifest_sha256=manifest.patch_bundle_sha256,
        experiment_intent=intent)


def _preauthored_checkpoint_authority(
        config: ControllerConfig, item: PlannedCandidate) -> dict[str, Any]:
    continuation = config.preauthored_continuations[item.hypothesis_id]
    match = re.fullmatch(r"akp-discovery-([1-9][0-9]*)",
                         str(item.proposal.get("proposal_id")))
    if (match is None or item.source_manifest.candidate_id !=
            f"akc-discovery-{match.group(1)}"):
        raise DiscoveryControllerError(
            "preauthored candidate authoring turn identity changed")
    body = {
        "schema": "epyc.autokernel.preauthored_checkpoint.v1",
        "hypothesis_id": item.hypothesis_id,
        "authoring_turn": int(match.group(1)),
        "carrier_sha256": continuation.sha256,
        "source_backed_diff_sha256": continuation.source_backed_diff_sha256,
        "source_manifest_sha256": item.source_manifest_sha256,
        "candidate_semantic_sha256": _candidate_semantic_identity(item),
        "cross_campaign_candidate_sha256":
            _cross_campaign_candidate_identity(item),
        "origin": hypotheses.ORIGIN_IMPORT,
        "author": "reviewed-eb26918-continuation",
        "historical_commit": continuation.historical_commit,
        "modern_governed_correctness_required": True,
    }
    body["receipt_sha256"] = _sha(body)
    return body


def _restore_pending(
        value: Mapping[str, Any],
        config: ControllerConfig | None = None) -> PlannedCandidate:
    raw=value.get("candidate")
    if not isinstance(raw,Mapping) or not isinstance(raw.get("manifest"),Mapping): raise DiscoveryControllerError("pending candidate is missing sealed manifest")
    preauthored = value.get("preauthored_continuation")
    if preauthored is not None:
        if (config is None or not isinstance(preauthored, Mapping)
                or not isinstance(value.get("row"), Mapping)
                or not isinstance(value["row"].get("portfolio_binding"), Mapping)):
            raise DiscoveryControllerError(
                "preauthored pending candidate lacks sealed controller authority")
        authoring_turn = preauthored.get("authoring_turn")
        if (isinstance(authoring_turn, bool)
                or not isinstance(authoring_turn, int)
                or authoring_turn <= 0):
            raise DiscoveryControllerError(
                "preauthored pending authoring turn is malformed")
        if value["row"].get("authoring_turn") != authoring_turn:
            raise DiscoveryControllerError(
                "preauthored row changed its original authoring turn")
        item = _preauthored_candidate(
            config, value["row"]["portfolio_binding"], authoring_turn)
        composition_raw = raw.get("composition_plan")
        if composition_raw is not None:
            try:
                item = replace(
                    item, composition_plan=
                    cumulative_composition.CompositionPlan.from_dict(
                        composition_raw))
            except cumulative_composition.CompositionError as exc:
                raise DiscoveryControllerError(
                    "preauthored cumulative plan is invalid") from exc
        expected = _preauthored_checkpoint_authority(config, item)
        if dict(preauthored) != expected or _pending_item(item) != dict(raw):
            raise DiscoveryControllerError(
                "preauthored pending candidate identity changed")
        return item
    m=raw["manifest"]
    try:
        manifest=source_candidate.SourcePatchManifest(campaign_id=m["campaign_id"],proposal_id=m["proposal_id"],candidate_id=m["candidate_id"],source_tree=m["source_tree"],production_base_commit=m["production_base_commit"],instrument_commit=m["instrument_commit"],change_class=m["change_class"],declared_files=tuple(m["declared_files"]),declared_symbols={k:tuple(v) for k,v in m["declared_symbols"].items()},mechanism_id=m["mechanism_id"],patch_sha256=m["patch_sha256"],patch_bytes=base64.b64decode(m["patch_base64"],validate=True))
    except (KeyError,TypeError,ValueError,source_candidate.SourceCandidateError) as exc: raise DiscoveryControllerError("pending candidate manifest is invalid") from exc
    try:
        raw_bytes=base64.b64decode(raw.get("manifest_raw_base64",""),validate=True)
    except (TypeError, ValueError) as exc:
        raise DiscoveryControllerError("pending manifest carrier is invalid") from exc
    canonical_bytes=source_candidate.source_patch_manifest_bytes(manifest)
    canonical_sha256=hashlib.sha256(canonical_bytes).hexdigest()
    identities=(canonical_sha256, manifest.patch_bundle_sha256,
                raw.get("manifest_file_sha256"), raw.get("patch_bundle_sha256"),
                raw.get("source_manifest_sha256"))
    if raw_bytes != canonical_bytes or any(value != canonical_sha256 for value in identities):
        raise DiscoveryControllerError("pending manifest identity mismatch")
    intent = raw.get("experiment_intent")
    if intent is not None and not isinstance(intent, Mapping):
        raise DiscoveryControllerError("pending experiment intent is malformed")
    if intent is not None:
        expected = intent.get("expected_dispatch")
        if not isinstance(expected, list) or not expected:
            raise DiscoveryControllerError("pending bounded dispatch is malformed")
        recommendation = intent.get("load_mode_recommendation")
        if recommendation is not None:
            if not isinstance(recommendation, Mapping):
                raise DiscoveryControllerError("pending load-mode recommendation is malformed")
            recommendation = LoadModeRecommendation(
                mode=recommendation.get("mode"), rationale=recommendation.get("rationale"),
                example_ids=tuple(recommendation.get("example_ids", ())))
        intent = {**intent, "expected_dispatch": tuple(
                      BoundedDispatchExpectation(**row) for row in expected),
                  "load_mode_recommendation": recommendation}
    composition_raw = raw.get("composition_plan")
    try:
        composition_plan = (
            None if composition_raw is None else
            cumulative_composition.CompositionPlan.from_dict(composition_raw))
    except cumulative_composition.CompositionError as exc:
        raise DiscoveryControllerError(
            "pending cumulative composition plan is invalid") from exc
    return PlannedCandidate(hypothesis_id=raw["hypothesis_id"],statement=raw["statement"],falsifier=raw["falsifier"],regime=raw["regime"],proposal=raw["proposal"],source_manifest=manifest,source_manifest_sha256=raw["source_manifest_sha256"],experiment_intent=GpuSourceExperimentIntent(**intent) if intent else None,composition_plan=composition_plan)


def _decision_floor(policy: Mapping[str, Any] | None, key: str,
                    fallback: float) -> float:
    if policy is None:
        return fallback
    value = policy.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0 <= float(value) <= 100):
        raise DiscoveryControllerError("portfolio decision policy has an invalid numeric floor")
    return float(value) / 100.0


def _required_replications(policy: Mapping[str, Any] | None) -> int:
    if policy is None:
        return 2
    value = policy.get("required_replications")
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 2 <= value <= 8
            or policy.get("sign_policy") not in {"all_positive", "median_positive"}
            or policy.get("conflict_policy") not in {"retire", "retain_inconclusive"}):
        raise DiscoveryControllerError("portfolio replication policy is malformed")
    return value


def _append_nomination(root: Path, item: PlannedCandidate, result: SealedScreen,
                       threshold: float) -> None:
    # A single screen is discovery evidence, never a nomination.  Only a
    # replicated series that retained a positive pooled classification may be
    # placed in the operator queue.
    if (result.series_effect_fraction is None
            or result.series_effect_fraction < threshold
            or result.classification != "top_k_replicated_candidate"):
        return
    path=root / "promotion-queue.jsonl"; lock=root / "promotion-queue.lock"; key=_sha({"result":result.result_sha256,"manifest":item.source_manifest_sha256})
    row={"schema":"epyc.autokernel.discovery_nomination.v1","idempotency_key":key,"receipt_path":result.receipt_path,"result_sha256":result.result_sha256,"source_manifest_sha256":item.source_manifest_sha256,"effect_fraction":result.effect_fraction,"series_effect_fraction":result.series_effect_fraction,"threshold":threshold,"promotion_claim":False,"operator_decision_required":True,"authority":AUTHORITY}
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("a+") as guard:
        fcntl.flock(guard.fileno(),fcntl.LOCK_EX)
        existing=path.read_text() if path.exists() else ""
        if key in existing: return
        with path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())


def _write_projection(root: Path) -> None:
    # Canonical projection is derived from receipts, not planner text.
    autokernel_progression.export_progression(root=root, output=root / "surface" / "kernel_progression.json")


def classify_screen_series(effects: Sequence[float], *,
                           component_pooled_effects: Sequence[float] = (),
                           continuation_floor: float = 0.0,
                           nomination_floor: float = 0.0,
                           min_replication_effect: float = 0.0,
                           max_replication_spread: float = 0.10,
                           required_replications: int = 2) -> str:
    """Discovery policy classifier; dashboard projection is not authority."""
    if not effects or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in effects):
        raise DiscoveryControllerError("screen series must contain numeric measured effects")
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in (continuation_floor, nomination_floor,
                          min_replication_effect, max_replication_spread))
            or isinstance(required_replications, bool)
            or not isinstance(required_replications, int)
            or not 2 <= required_replications <= 8
            or nomination_floor < continuation_floor):
        raise DiscoveryControllerError("screen series decision policy is malformed")
    if len(effects) == 1:
        return ("candidate" if effects[0] > 0
                and effects[0] >= continuation_floor else "screened_out")
    if min(effects) < 0 < max(effects):
        return "inconclusive"
    # A materially divergent pair is no more rankable than opposite signs.
    # This is the discovery lane's 10 percentage-point spread rule, not a
    # calibration gate; it requests a retest rather than declaring a failure.
    if max(effects) - min(effects) > max_replication_spread:
        return "inconclusive"
    if len(effects) < required_replications:
        return ("candidate" if all(v > 0 and v >= continuation_floor for v in effects)
                else "screened_out")
    pooled = float(statistics.median(effects))
    if any(value < min_replication_effect for value in effects):
        return "screened_out"
    if all(v > 0 for v in effects) and component_pooled_effects and pooled < max(component_pooled_effects):
        return "replicated_but_subadditive"
    if all(v > 0 for v in effects) and pooled >= nomination_floor:
        return "top_k_replicated_candidate"
    return "screened_out"

def _screen_series_key(item: PlannedCandidate, result: SealedScreen) -> str:
    """Return the hash that permits only like-for-like replications to pool."""
    if result.series_key is not None:
        return result.series_key
    # Legacy/replay fakes have no explicitly captured frame.  Their fallback
    # remains conservative: different patch, regime, or immutable baseline is
    # a different series.  Live GPU adapters must populate series_key from the
    # sealed model/workload/runtime frame before returning a SealedScreen.
    return _sha({"source_manifest_sha256": item.source_manifest_sha256,
                 "regime": item.regime,
                 "baseline_sha256": result.baseline_sha256})


def _pooled_component_effects(state: Mapping[str, Any], component_keys: Sequence[str]) -> list[float]:
    values: list[float] = []
    for key in component_keys:
        effects = [float(row["effect_fraction"]) for row in state["iterations"]
                   if row.get("series_key") == key and isinstance(row.get("effect_fraction"), (int, float))]
        if effects:
            values.append(sum(effects) / len(effects))
    return values

def _classified_result(state: Mapping[str, Any], item: PlannedCandidate,
                       result: SealedScreen,
                       decision_policy: Mapping[str, Any] | None = None) -> SealedScreen:
    if getattr(item, "composition_plan", None) is not None:
        comparison = result.composition_comparison
        if comparison is None:
            raise DiscoveryControllerError(
                "cumulative result lacks incremental comparison authority")
        components = tuple(sorted({
            replication.series_key
            for lever in item.composition_plan.anchor.accepted
            for replication in lever.replications}))
        return replace(
            result, classification=comparison.classification,
            series_key=item.composition_plan.candidate.ordered_patch_set_sha256,
            component_series_keys=components,
            series_effect_fraction=result.effect_fraction)
    series_key = _screen_series_key(item, result)
    prior = [float(row["effect_fraction"]) for row in state["iterations"]
             if row.get("series_key") == series_key
             and isinstance(row.get("effect_fraction"), (int, float))]
    # Component provenance is measured/sealed by the adapter.  Planner text
    # cannot name its own component evidence and thereby manufacture a
    # subadditivity claim.
    raw_components = result.component_series_keys
    if not isinstance(raw_components, (list, tuple)) or not all(isinstance(key, str) and HASH.fullmatch(key) for key in raw_components):
        raise DiscoveryControllerError("composition requires exact component series provenance")
    components = tuple(raw_components)
    effects = prior + [result.effect_fraction]
    classification = classify_screen_series(
        effects,
        component_pooled_effects=_pooled_component_effects(state, components),
        continuation_floor=_decision_floor(
            decision_policy, "continuation_floor_pct", 0.0),
        nomination_floor=_decision_floor(
            decision_policy, "nomination_floor_pct", 0.0),
        min_replication_effect=_decision_floor(
            decision_policy, "min_replication_effect_pct", 0.0),
        max_replication_spread=_decision_floor(
            decision_policy, "max_replication_spread_pct", 0.10),
        required_replications=_required_replications(decision_policy),
    )
    dual_effects_present = (result.exact_attribution_effect_fraction is not None
                            or result.target_runtime_effect_fraction is not None)
    if dual_effects_present and (
            result.exact_attribution_effect_fraction is None
            or result.target_runtime_effect_fraction is None
            or result.exact_attribution_effect_fraction <= 0
            or result.target_runtime_effect_fraction <= 0):
        # Route/device-time and target-runtime throughput are conjunctive.  A
        # disagreement is measured evidence, but never a candidate/nomination.
        classification = "inconclusive"
    return replace(
        result, classification=classification, series_key=series_key,
        component_series_keys=components,
        series_effect_fraction=float(statistics.median(effects)))


def _screen_iteration_fields(result: SealedScreen, *, repetition: int) -> dict[str, Any]:
    target_executed = result.target_runtime_effect_fraction is not None
    return {
        "status": result.classification,
        "result_sha256": result.result_sha256,
        "evidence": {"baseline": result.baseline_sha256,
                     "source": result.source_proof_sha256,
                     "dispatch": result.dispatch_proof_sha256},
        "effect_fraction": result.effect_fraction,
        "series_effect_fraction": result.series_effect_fraction,
        "series_key": result.series_key,
        "component_series_keys": list(result.component_series_keys),
        "exact_attribution_effect_fraction":
            result.exact_attribution_effect_fraction,
        "target_runtime_effect_fraction":
            result.target_runtime_effect_fraction,
        "target_runtime_executed": target_executed,
        "target_runtime_reason": (
            None if target_executed else
            "nonpositive_exact_duration"
            if result.exact_attribution_effect_fraction is not None
            and result.exact_attribution_effect_fraction <= 0
            else "not_required_or_unavailable"),
        "stages": list(result.stages),
        "build_identity_sha256": result.build_identity_sha256,
        "correctness_receipt_sha256": result.correctness_receipt_sha256,
        "attribution_receipt_sha256": result.attribution_receipt_sha256,
        "graphs_off_receipt_sha256": result.graphs_off_receipt_sha256,
        "graphs_on_receipt_sha256": result.graphs_on_receipt_sha256,
        "repetition": repetition,
        "scientific_budget_spent": True,
    }


def _row_spends_scientific_budget(row: Mapping[str, Any]) -> bool:
    return (row.get("scientific_budget_spent") is True
            or isinstance(row.get("result_sha256"), str)
            and HASH.fullmatch(row["result_sha256"])
            and isinstance(row.get("evidence"), Mapping))


def _candidate_semantic_identity(item: PlannedCandidate) -> str:
    """Hash source semantics while excluding per-turn envelope identities."""
    manifest = item.source_manifest
    if item.composition_plan is not None:
        return _sha({
            "schema": "epyc.autokernel.cumulative_candidate_semantics.v1",
            "anchor_patch_set_sha256":
                item.composition_plan.anchor.ordered_patch_set_sha256,
            "candidate_patch_set_sha256":
                item.composition_plan.candidate.ordered_patch_set_sha256,
            "new_lever_sha256":
                item.composition_plan.candidate.accepted[-1].lever_sha256,
        })
    return _sha({
        "schema": "epyc.autokernel.candidate_source_semantics.v1",
        "source_tree": manifest.source_tree,
        "production_base_commit": manifest.production_base_commit,
        "instrument_commit": manifest.instrument_commit,
        "change_class": manifest.change_class,
        "declared_files": sorted(manifest.declared_files),
        "declared_symbols": {
            key: sorted(value)
            for key, value in sorted(manifest.declared_symbols.items())},
        "mechanism_id": manifest.mechanism_id,
        "patch_sha256": manifest.patch_sha256,
    })


def _cross_campaign_candidate_identity(item: PlannedCandidate) -> str:
    """Stable candidate identity that deliberately excludes instrument epochs."""
    manifest = item.source_manifest
    if item.composition_plan is not None:
        return _sha({
            "schema": "epyc.autokernel.cross_campaign_composition.v1",
            "production_base_commit": manifest.production_base_commit,
            "anchor_patch_set_sha256":
                item.composition_plan.anchor.ordered_patch_set_sha256,
            "candidate_patch_set_sha256":
                item.composition_plan.candidate.ordered_patch_set_sha256,
        })
    return _sha({
        "schema": "epyc.autokernel.cross_campaign_candidate_semantics.v1",
        "production_base_commit": manifest.production_base_commit,
        "change_class": manifest.change_class,
        "declared_files": sorted(manifest.declared_files),
        "declared_symbols": {
            key: sorted(value)
            for key, value in sorted(manifest.declared_symbols.items())},
        "mechanism_id": manifest.mechanism_id,
        "patch_sha256": manifest.patch_sha256,
    })


def _record_attempted_candidate_identity(
        state: dict[str, Any], row: Mapping[str, Any]) -> None:
    identity = row.get("candidate_semantic_sha256")
    operation_key = row.get("operation_key")
    result_sha256 = row.get("result_sha256")
    hypothesis_id = row.get("portfolio_hypothesis_id")
    if (not all(isinstance(value, str) and HASH.fullmatch(value)
                for value in (identity, operation_key, result_sha256))
            or not isinstance(hypothesis_id, str)
            or not _row_spends_scientific_budget(row)):
        raise DiscoveryControllerError(
            "scientific candidate lacks its semantic attempt identity")
    attempt = {
        "operation_key": operation_key,
        "result_sha256": result_sha256,
        "disposition": row.get("status"),
        "repetition": row.get("repetition", 1),
    }
    state["candidate_semantic_registry_schema"] = (
        "epyc.autokernel.candidate_semantic_registry.v1")
    registry = state.setdefault("attempted_candidate_identities", {})
    prior = registry.get(identity)
    if prior is None:
        prior = {"hypothesis_id": hypothesis_id, "attempts": []}
        registry[identity] = prior
    if (not isinstance(prior, dict)
            or prior.get("hypothesis_id") != hypothesis_id
            or not isinstance(prior.get("attempts"), list)):
        raise DiscoveryControllerError(
            "candidate source semantics already have a different scientific outcome")
    if attempt in prior["attempts"]:
        return
    if any(current.get("operation_key") == operation_key
           for current in prior["attempts"] if isinstance(current, Mapping)):
        raise DiscoveryControllerError(
            "candidate source semantics repeat an operation with changed evidence")
    prior["attempts"].append(attempt)


def _validate_attempted_candidate_identities(state: Mapping[str, Any]) -> None:
    marker = state.get("candidate_semantic_registry_schema")
    registry = state.get("attempted_candidate_identities", {})
    if marker is None and not registry:
        # Historical checkpoints predate semantic retry suppression.  New
        # scientific writes publish the marker and registry atomically.
        return
    if marker != "epyc.autokernel.candidate_semantic_registry.v1":
        raise DiscoveryControllerError(
            "durable candidate semantic registry version is malformed")
    if not isinstance(registry, Mapping):
        raise DiscoveryControllerError(
            "durable candidate semantic attempt registry is malformed")
    derived: dict[str, dict[str, Any]] = {}
    for row in state.get("iterations", []):
        if (not isinstance(row, Mapping)
                or not _row_spends_scientific_budget(row)
                or not isinstance(row.get("portfolio_hypothesis_id"), str)):
            continue
        identity = row.get("candidate_semantic_sha256")
        operation_key = row.get("operation_key")
        result_sha256 = row.get("result_sha256")
        hypothesis_id = row.get("portfolio_hypothesis_id")
        if (not all(isinstance(value, str) and HASH.fullmatch(value)
                    for value in (identity, operation_key, result_sha256))
                or not isinstance(hypothesis_id, str)):
            raise DiscoveryControllerError(
                "durable scientific row lacks candidate semantic identity")
        attempt = {
            "operation_key": operation_key,
            "result_sha256": result_sha256,
            "disposition": row.get("status"),
            "repetition": row.get("repetition", 1),
        }
        entry = derived.setdefault(identity, {
            "hypothesis_id": hypothesis_id, "attempts": []})
        if entry["hypothesis_id"] != hypothesis_id:
            raise DiscoveryControllerError(
                "durable state aliases candidate semantics across hypotheses")
        if (attempt in entry["attempts"]
                or any(current.get("operation_key") == operation_key
                       for current in entry["attempts"])):
            raise DiscoveryControllerError(
                "durable state repeats a candidate semantic operation")
        entry["attempts"].append(attempt)
    if dict(registry) != derived:
        raise DiscoveryControllerError(
            "durable candidate semantic registry differs from scientific rows")


def _require_unattempted_checkpoint(
        state: Mapping[str, Any], row: Mapping[str, Any], *,
        confirmation: bool) -> None:
    if confirmation:
        return
    semantic = row.get("candidate_semantic_sha256")
    registry = state.get("attempted_candidate_identities", {})
    if isinstance(semantic, str) and semantic in registry:
        raise DiscoveryControllerError(
            "durable candidate checkpoint repeats prior source semantics")


def _validate_infrastructure_ambiguities(state: Mapping[str, Any]) -> None:
    events = state.get("infrastructure_ambiguities", [])
    if not isinstance(events, list):
        raise DiscoveryControllerError(
            "durable infrastructure ambiguity ledger is malformed")
    seen: set[str] = set()
    latest: dict[str, int] = {}
    for event in events:
        if (not isinstance(event, Mapping)
                or set(event) != {
                    "schema", "operation_key", "source_manifest_sha256",
                    "candidate_semantic_sha256", "stage_receipt_path",
                    "stage_receipt_sha256", "reason_sha256", "retry_epoch"}
                or event.get("schema") !=
                   "epyc.autokernel.screen_infrastructure_ambiguity.v1"):
            raise DiscoveryControllerError(
                "durable infrastructure ambiguity event is malformed")
        hashes = (event.get("operation_key"),
                  event.get("source_manifest_sha256"),
                  event.get("candidate_semantic_sha256"),
                  event.get("stage_receipt_sha256"),
                  event.get("reason_sha256"))
        epoch = event.get("retry_epoch")
        if (not all(isinstance(value, str) and HASH.fullmatch(value)
                    for value in hashes)
                or not isinstance(event.get("stage_receipt_path"), str)
                or not event["stage_receipt_path"]
                or isinstance(epoch, bool) or not isinstance(epoch, int)
                or epoch < 0 or event["operation_key"] in seen):
            raise DiscoveryControllerError(
                "durable infrastructure ambiguity authority is malformed")
        identity = str(event["candidate_semantic_sha256"])
        expected_epoch = latest.get(identity, -1) + 1
        if epoch != expected_epoch:
            raise DiscoveryControllerError(
                "durable infrastructure retry epochs are not contiguous")
        latest[identity] = epoch
        seen.add(str(event["operation_key"]))
    for label in ("pending", "inflight"):
        holder = state.get(label)
        if not isinstance(holder, Mapping):
            continue
        row = holder.get("row")
        identity = (row.get("candidate_semantic_sha256")
                    if isinstance(row, Mapping) else None)
        epoch = holder.get("infrastructure_retry_epoch", 0)
        if (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
                or epoch > 0 and (not isinstance(identity, str)
                                  or not HASH.fullmatch(identity))):
            raise DiscoveryControllerError(
                f"durable {label} infrastructure retry authority is malformed")
        expected = latest.get(str(identity), -1) + 1
        if ((epoch or identity in latest)
                and holder.get("confirmation") is not True
                and epoch != expected):
            raise DiscoveryControllerError(
                f"durable {label} infrastructure retry epoch changed")
        if label == "inflight" and epoch:
            operation_key = holder.get("operation_key")
            if operation_key in seen:
                raise DiscoveryControllerError(
                    "inflight operation reuses a refused infrastructure epoch")


def _preauthored_pending_fields(
        row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    authority = row.get("preauthored_continuation")
    if authority is None:
        return {}
    if (not isinstance(authority, Mapping)
            or authority.get("schema") !=
               "epyc.autokernel.preauthored_checkpoint.v1"):
        raise DiscoveryControllerError(
            "candidate continuation checkpoint is malformed")
    return {"preauthored_continuation": dict(authority)}


def _record_screen_infrastructure_ambiguity(
        state: dict[str, Any], row: dict[str, Any],
        exc: ScreenInfrastructureAmbiguity) -> None:
    inflight = state.get("inflight")
    if (not isinstance(inflight, Mapping)
            or inflight.get("operation_key") != exc.operation_key):
        raise DiscoveryControllerError(
            "screen infrastructure ambiguity does not bind the inflight operation")
    epoch = inflight.get("infrastructure_retry_epoch", 0)
    semantic = row.get("candidate_semantic_sha256")
    manifest = row.get("source_manifest_sha256")
    if (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
            or not isinstance(semantic, str) or not HASH.fullmatch(semantic)
            or not isinstance(manifest, str) or not HASH.fullmatch(manifest)):
        raise DiscoveryControllerError(
            "screen infrastructure ambiguity lacks candidate retry authority")
    event = {
        "schema": "epyc.autokernel.screen_infrastructure_ambiguity.v1",
        "operation_key": exc.operation_key,
        "source_manifest_sha256": manifest,
        "candidate_semantic_sha256": semantic,
        "stage_receipt_path": exc.receipt_path,
        "stage_receipt_sha256": exc.receipt_sha256,
        "reason_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        "retry_epoch": epoch,
    }
    events = state.setdefault("infrastructure_ambiguities", [])
    events.append(event)
    state.pop("inflight", None)
    state["pending"] = {
        "row": row,
        "candidate": inflight["candidate"],
        "authorization": inflight["authorization"],
        "confirmation": bool(inflight.get("confirmation")),
        "parent_authorization": inflight.get("parent_authorization"),
        "infrastructure_retry_epoch": epoch + 1,
        "prior_operation_key": exc.operation_key,
        **_preauthored_pending_fields(row),
    }
    _validate_infrastructure_ambiguities(state)


def _derived_scientific_attempts(state: Mapping[str, Any]) -> int:
    iterations = state.get("iterations", [])
    if not isinstance(iterations, list):
        raise DiscoveryControllerError("durable iterations are malformed")
    return sum(1 for row in iterations
               if isinstance(row, Mapping)
               and _row_spends_scientific_budget(row))


def _bind_scientific_attempt_counter(state: dict[str, Any]) -> int:
    """Bind the persisted counter to the recursively sealed result rows."""
    derived = _derived_scientific_attempts(state)
    declared = state.get("scientific_attempts")
    if declared is not None and (isinstance(declared, bool)
                                 or declared != derived):
        raise DiscoveryControllerError(
            "durable scientific-attempt counter disagrees with result rows")
    state["scientific_attempts"] = derived
    return derived


def _record_replicated_positive_lever(
        state: dict[str, Any], item: PlannedCandidate,
        result: SealedScreen) \
        -> cumulative_composition.ReplicatedPositiveLever | None:
    """Freeze a controller nomination into executable composition authority.

    Old replay fixtures intentionally lack the five receipt identities.  They
    remain useful for classifier tests, but can never become executable source
    composition authority.
    """
    if (item.composition_plan is not None
            or result.classification != "top_k_replicated_candidate"
            or result.series_key is None):
        return None
    rows = [row for row in state.get("iterations", [])
            if row.get("series_key") == result.series_key
            and row.get("status") in {
                "candidate", "top_k_replicated_candidate"}]
    required = (
        "result_sha256", "series_key", "build_identity_sha256",
        "correctness_receipt_sha256", "attribution_receipt_sha256",
        "graphs_off_receipt_sha256", "graphs_on_receipt_sha256")
    if len(rows) < 2 or any(
            not all(isinstance(row.get(key), str)
                    and HASH.fullmatch(row[key]) for key in required)
            or not isinstance(row.get("effect_fraction"), (int, float))
            or isinstance(row.get("effect_fraction"), bool)
            or not math.isfinite(float(row["effect_fraction"]))
            or float(row["effect_fraction"]) <= 0
            for row in rows):
        return None
    replications = tuple(
        cumulative_composition.IsolatedReplication(
            result_sha256=row["result_sha256"],
            series_key=row["series_key"],
            build_identity_sha256=row["build_identity_sha256"],
            correctness_receipt_sha256=row["correctness_receipt_sha256"],
            attribution_receipt_sha256=row["attribution_receipt_sha256"],
            graphs_off_receipt_sha256=row["graphs_off_receipt_sha256"],
            graphs_on_receipt_sha256=row["graphs_on_receipt_sha256"],
            effect_fraction=float(row["effect_fraction"]))
        for row in rows)
    lever = cumulative_composition.ReplicatedPositiveLever(
        hypothesis_id=item.hypothesis_id,
        cross_campaign_candidate_sha256=
            _cross_campaign_candidate_identity(item),
        manifest=item.source_manifest, replications=replications)
    registry = state.setdefault("replicated_positive_levers", {})
    if not isinstance(registry, dict):
        raise DiscoveryControllerError(
            "durable replicated-positive registry is malformed")
    prior = registry.get(lever.cross_campaign_candidate_sha256)
    if prior is not None and prior != lever.to_dict():
        raise DiscoveryControllerError(
            "replicated-positive lever changed on restart")
    registry[lever.cross_campaign_candidate_sha256] = lever.to_dict()
    state["replicated_positive_lever_schema"] = (
        "epyc.autokernel.replicated_positive_registry.v1")
    return lever


def _validate_replicated_positive_levers(
        state: Mapping[str, Any]) -> None:
    marker = state.get("replicated_positive_lever_schema")
    registry = state.get("replicated_positive_levers", {})
    if marker is None and not registry:
        return
    if (marker != "epyc.autokernel.replicated_positive_registry.v1"
            or not isinstance(registry, Mapping)):
        raise DiscoveryControllerError(
            "durable replicated-positive registry is malformed")
    iterations = state.get("iterations")
    if not isinstance(iterations, list):
        raise DiscoveryControllerError(
            "durable iterations are malformed")
    for key, value in registry.items():
        try:
            lever = cumulative_composition.ReplicatedPositiveLever.from_dict(
                value)
        except cumulative_composition.CompositionError as exc:
            raise DiscoveryControllerError(
                "durable replicated-positive lever is invalid") from exc
        if key != lever.cross_campaign_candidate_sha256:
            raise DiscoveryControllerError(
                "replicated-positive registry key changed")
        for replication in lever.replications:
            matches = []
            for row in iterations:
                effect = row.get("effect_fraction")
                if (isinstance(effect, bool)
                        or not isinstance(effect, (int, float))
                        or not math.isfinite(float(effect))):
                    continue
                if (row.get("result_sha256") == replication.result_sha256
                        and row.get("series_key") == replication.series_key
                        and row.get("build_identity_sha256") ==
                            replication.build_identity_sha256
                        and row.get("correctness_receipt_sha256") ==
                            replication.correctness_receipt_sha256
                        and row.get("attribution_receipt_sha256") ==
                            replication.attribution_receipt_sha256
                        and row.get("graphs_off_receipt_sha256") ==
                            replication.graphs_off_receipt_sha256
                        and row.get("graphs_on_receipt_sha256") ==
                            replication.graphs_on_receipt_sha256
                        and float(effect) == replication.effect_fraction):
                    matches.append(row)
            if len(matches) != 1:
                raise DiscoveryControllerError(
                    "replicated-positive lever differs from scientific rows")


def _composition_ledger(config: ControllerConfig) \
        -> cumulative_composition.CompositionLedger:
    return cumulative_composition.CompositionLedger(
        config.output_root / "cumulative-composition.json")


def _checkpoint_composition_plan(
        holder: object) -> cumulative_composition.CompositionPlan | None:
    if not isinstance(holder, Mapping):
        return None
    candidate = holder.get("candidate")
    if not isinstance(candidate, Mapping):
        return None
    raw = candidate.get("composition_plan")
    if raw is None:
        return None
    try:
        return cumulative_composition.CompositionPlan.from_dict(raw)
    except cumulative_composition.CompositionError as exc:
        raise DiscoveryControllerError(
            "durable cumulative candidate plan is invalid") from exc


def _validate_cumulative_composition_state(
        config: ControllerConfig, state: Mapping[str, Any]) -> None:
    """Join the append-only composition ledger to controller checkpoints.

    A controller checkpoint may precede the lazy ledger begin.  Conversely, a
    ledger terminal may precede the controller's terminal-row save by one
    crash window, but only for the exact currently pending/inflight plan.
    """
    ledger_path = config.output_root / "cumulative-composition.json"
    holders = tuple(
        plan for plan in (
            _checkpoint_composition_plan(state.get("pending")),
            _checkpoint_composition_plan(state.get("inflight")))
        if plan is not None)
    if len(holders) > 1:
        raise DiscoveryControllerError(
            "controller owns multiple cumulative composition checkpoints")
    iteration_rows = [row for row in state.get("iterations", [])
                      if isinstance(row, Mapping)
                      and row.get("composition_terminal_sha256") is not None]
    if not ledger_path.exists() and not ledger_path.is_symlink():
        if iteration_rows:
            raise DiscoveryControllerError(
                "controller composition terminals lack their ledger")
        return
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise DiscoveryControllerError(
            "cumulative composition ledger path is unsafe")
    try:
        ledger_state = _composition_ledger(config).load()
        initial = cumulative_composition.CompositionAuthority.from_dict(
            ledger_state["initial_authority"])
    except cumulative_composition.CompositionError as exc:
        raise DiscoveryControllerError(
            "cumulative composition ledger is invalid") from exc
    production = config.production_base_commit
    instrument = config.instrument_commit
    if (initial.campaign_id != config.campaign_id
            or production is not None
               and initial.production_base_commit != production
            or instrument is not None
               and initial.instrument_commit != instrument):
        raise DiscoveryControllerError(
            "cumulative composition ledger names another deployment")
    terminal_by_operation = {
        row["operation_key"]: row for row in ledger_state["terminals"]}
    if len(terminal_by_operation) != len(ledger_state["terminals"]):
        raise DiscoveryControllerError(
            "cumulative composition terminal operations are duplicated")
    joined: set[str] = set()
    for row in iteration_rows:
        operation = row.get("composition_operation_key")
        terminal = terminal_by_operation.get(operation)
        if (terminal is None
                or row.get("composition_terminal_sha256") !=
                   terminal["terminal_sha256"]
                or row.get("composition_disposition") !=
                   terminal["disposition"]
                or row.get("composition_scientific_budget_spent") !=
                   terminal["scientific_budget_spent"]
                or row.get("composition_promotion_eligible") !=
                   terminal["promotion_eligible"]
                or row.get("composition_promotion_reason") !=
                   terminal["promotion_reason"]
                or row.get("cumulative_performance_result_sha256") !=
                   terminal["cumulative_performance_result_sha256"]):
            raise DiscoveryControllerError(
                "controller composition result differs from its ledger")
        joined.add(str(operation))
    holder = holders[0] if holders else None
    ahead = set(terminal_by_operation) - joined
    if ahead:
        if (len(ahead) != 1 or holder is None
                or holder.operation_key not in ahead
                or terminal_by_operation[holder.operation_key]["plan"] !=
                   holder.to_dict()):
            raise DiscoveryControllerError(
                "composition ledger has an unowned terminal")
    pending = ledger_state["pending"]
    if pending is not None:
        pending_plan = cumulative_composition.CompositionPlan.from_dict(
            pending["plan"])
        if holder is None or pending_plan != holder:
            raise DiscoveryControllerError(
                "composition ledger pending plan lacks controller ownership")
    state_terminal = state.get("cumulative_composition_terminal")
    state_performance = state.get("cumulative_performance")
    if not ledger_state["terminals"]:
        if state_terminal is not None or state_performance is not None:
            raise DiscoveryControllerError(
                "controller cumulative authority lacks a terminal")
    elif not ahead:
        latest = ledger_state["terminals"][-1]
        if state_terminal != latest:
            raise DiscoveryControllerError(
                "controller cumulative terminal envelope changed")
        typed_ref = latest.get("cumulative_performance_ref")
        expected_ref = (
            {"path": typed_ref["path"], "sha256": typed_ref["sha256"]}
            if latest.get("disposition") == "admitted"
            and isinstance(typed_ref, Mapping) else None)
        if state_performance != expected_ref:
            raise DiscoveryControllerError(
                "controller cumulative performance reference changed")
        if expected_ref is not None:
            try:
                reopened, file_sha = \
                    cumulative_composition.load_cumulative_performance(
                        Path(expected_ref["path"]),
                        expected_file_sha256=expected_ref["sha256"])
                expected_performance = \
                    cumulative_composition.CumulativePerformance.from_dict(
                        latest["cumulative_performance"])
            except cumulative_composition.CompositionError as exc:
                raise DiscoveryControllerError(
                    "controller cumulative performance cannot be reopened") \
                    from exc
            if (reopened != expected_performance
                    or file_sha != expected_ref["sha256"]):
                raise DiscoveryControllerError(
                    "controller cumulative performance receipt changed")


def _schedule_cumulative_composition(
        state: dict[str, Any], *, config: ControllerConfig,
        item: PlannedCandidate, result: SealedScreen) -> None:
    if state.get("pending") is not None:
        return
    lever = _record_replicated_positive_lever(state, item, result)
    if lever is None:
        return
    if _derived_scientific_attempts(state) >= config.max_iterations:
        return
    production_commit = (config.production_base_commit
                         or item.source_manifest.production_base_commit)
    instrument_commit = (config.instrument_commit
                         or item.source_manifest.instrument_commit)
    ledger = _composition_ledger(config)
    initial = cumulative_composition.CompositionAuthority(
        campaign_id=config.campaign_id,
        production_base_commit=production_commit,
        instrument_commit=instrument_commit)
    composition_state = ledger.create(
        initial, max_scientific_attempts=config.max_iterations)
    if composition_state["pending"] is not None:
        pending_plan = cumulative_composition.CompositionPlan.from_dict(
            composition_state["pending"]["plan"])
        if (pending_plan.candidate.accepted[-1].lever_sha256 !=
                lever.lever_sha256):
            raise DiscoveryControllerError(
                "another cumulative operation is already pending")
        plan = pending_plan
    else:
        anchor = cumulative_composition.CompositionAuthority.from_dict(
            composition_state["authority"])
        if any(existing.lever_sha256 == lever.lever_sha256
               for existing in anchor.accepted):
            return
        checked = {
            existing.cross_campaign_candidate_sha256
            for existing in anchor.accepted}
        checked.update(
            terminal["cross_campaign_candidate_sha256"]
            for terminal in composition_state["terminals"]
            if terminal["cross_campaign_candidate_sha256"] !=
               lever.cross_campaign_candidate_sha256)
        candidate_authority = anchor.append(lever)
        registry_body = {
            "schema": "epyc.autokernel.composition_dnr_registry.v1",
            "controller_candidate_registry_sha256": _sha(
                state.get("attempted_candidate_identities", {})),
            "checked_cross_campaign_candidate_sha256s": sorted(checked),
        }
        dnr = cumulative_composition.DnrAuthority.pass_for(
            anchor=anchor, candidate=candidate_authority,
            registry_sha256=_sha(registry_body),
            checked_cross_campaign_candidate_sha256s=sorted(checked))
        attempt_id = _sha({
            "schema": "epyc.autokernel.composition_attempt.v1",
            "turn": state["next"], "lever_sha256": lever.lever_sha256,
            "ledger_generation": composition_state["generation"],
        })
        plan = cumulative_composition.CompositionPlan.create(
            anchor=anchor, lever=lever, dnr=dnr, attempt_id=attempt_id)
    cumulative_item = replace(item, composition_plan=plan)
    row = {
        key: value for key, value in state["iterations"][-1].items()
        if key in {
            "hypothesis_id", "statement", "falsifier", "regime",
            "proposal_sha256", "source_manifest_sha256",
            "experiment_intent", "mechanism_id", "target_surface",
            "target_symbol", "context_sha256", "authoring_turn",
            "portfolio_binding",
            "portfolio_record_sha256", "portfolio_decision_policy",
            "portfolio_exact_dnr_check",
            "preauthored_continuation", "hypothesis_origin",
            "hypothesis_author", "historical_correctness_authority",
            "modern_governed_correctness_required"}}
    row.update({
        "turn": state["next"], "status": "cumulative_composition_pending",
        "candidate_semantic_sha256":
            _candidate_semantic_identity(cumulative_item),
        "composition_plan_sha256": plan.plan_sha256,
        "composition_operation_key": plan.operation_key,
        "anchor_patch_set_sha256": plan.anchor.ordered_patch_set_sha256,
        "candidate_patch_set_sha256": plan.candidate.ordered_patch_set_sha256,
        "composition_new_lever_sha256": lever.lever_sha256,
        "composition_hypothesis_id": item.hypothesis_id,
    })
    state["pending"] = {
        "phase": "cumulative_ready", "row": row,
        "candidate": _pending_item(cumulative_item),
        "confirmation": False, "parent_authorization": None,
        **_preauthored_pending_fields(row)}


def _finalize_cumulative_screen(
        config: ControllerConfig, item: PlannedCandidate,
        result: SealedScreen) -> Mapping[str, Any] | None:
    if item.composition_plan is None:
        return None
    pair = result.composition_build_pair
    correctness = result.composition_correctness
    comparison = result.composition_comparison
    performance = result.cumulative_performance
    performance_ref = result.cumulative_performance_ref
    if (pair is None or correctness is None or comparison is None
            or performance is None or performance_ref is None):
        raise DiscoveryControllerError(
            "cumulative screen lacks promotion-authority evidence")
    pair.bind_plan(item.composition_plan)
    performance.bind(item.composition_plan, pair, correctness, comparison)
    reopened, file_sha = cumulative_composition.load_cumulative_performance(
        Path(performance_ref.path),
        expected_file_sha256=performance_ref.sha256)
    if reopened != performance or file_sha != performance_ref.sha256:
        raise DiscoveryControllerError(
            "cumulative performance reference changed before terminalization")
    ledger = _composition_ledger(config)
    existing = [row for row in ledger.load()["terminals"]
                if row["operation_key"] == item.composition_plan.operation_key]
    if existing:
        if (len(existing) != 1
                or existing[0]["build_pair"] != pair.to_dict()
                or existing[0]["correctness"] != correctness.to_dict()
                or existing[0]["comparison"] != comparison.to_dict()
                or existing[0]["cumulative_performance"] !=
                   performance.to_dict()
                or existing[0]["cumulative_performance_ref"] !=
                   performance_ref.to_dict()):
            raise DiscoveryControllerError(
                "cumulative screen differs from its durable terminal")
        return existing[0]
    ledger.begin(item.composition_plan)
    ledger.record_build_pair(pair)
    ledger.record_correctness(correctness)
    ledger.record_comparison(comparison)
    ledger.record_cumulative_performance(performance, performance_ref)
    state = ledger.finalize(item.composition_plan.operation_key)
    matches = [row for row in state["terminals"]
               if row["operation_key"] == item.composition_plan.operation_key]
    if len(matches) != 1:
        raise DiscoveryControllerError(
            "cumulative ledger lacks one exact terminal")
    return matches[0]


def _terminalize_cumulative_refusal(
        config: ControllerConfig, item: PlannedCandidate,
        exc: GovernedStageRefusal) -> Mapping[str, Any] | None:
    if item.composition_plan is None:
        return None
    ledger = _composition_ledger(config)
    existing = [row for row in ledger.load()["terminals"]
                if row["operation_key"] == item.composition_plan.operation_key]
    if existing:
        if len(existing) != 1:
            raise DiscoveryControllerError(
                "cumulative refusal has duplicate durable terminals")
        terminal = existing[0]
        if isinstance(exc, CumulativeCorrectnessRefusal):
            expected = (
                "correctness_rollback", exc.build_pair.to_dict(),
                exc.correctness.to_dict(), None)
        elif isinstance(exc, CumulativeAttributionRefusal):
            expected = (
                "attribution_rollback", exc.build_pair.to_dict(),
                exc.correctness.to_dict(), exc.receipt_sha256)
        elif not exc.scientific_budget_spent:
            expected = (
                "infrastructure_rollback", None, None, exc.receipt_sha256)
        else:
            raise DiscoveryControllerError(
                "scientific cumulative refusal lacks typed incremental evidence")
        observed_receipt = (terminal["attribution_receipt_sha256"]
                            if expected[0] == "attribution_rollback" else
                            terminal["infrastructure_receipt_sha256"])
        if (terminal["disposition"], terminal["build_pair"],
                terminal["correctness"], observed_receipt) != expected:
            raise DiscoveryControllerError(
                "cumulative refusal differs from its durable terminal")
        return terminal
    ledger.begin(item.composition_plan)
    if isinstance(exc, CumulativeCorrectnessRefusal):
        ledger.record_build_pair(exc.build_pair)
        state = ledger.record_correctness(exc.correctness)
    elif isinstance(exc, CumulativeAttributionRefusal):
        ledger.record_build_pair(exc.build_pair)
        ledger.record_correctness(exc.correctness)
        state = ledger.rollback_attribution(
            item.composition_plan.operation_key,
            receipt_sha256=exc.receipt_sha256)
    elif not exc.scientific_budget_spent:
        state = ledger.rollback_infrastructure(
            item.composition_plan.operation_key,
            reason_code=f"governed_{exc.stage}_refusal",
            receipt_sha256=exc.receipt_sha256)
    else:
        raise DiscoveryControllerError(
            "scientific cumulative refusal lacks typed incremental evidence")
    matches = [row for row in state["terminals"]
               if row["operation_key"] == item.composition_plan.operation_key]
    if len(matches) != 1:
        raise DiscoveryControllerError(
            "cumulative refusal did not produce one ledger terminal")
    return matches[0]


def _record_cumulative_infrastructure_ambiguity(
        state: dict[str, Any], *, config: ControllerConfig,
        item: PlannedCandidate, row: dict[str, Any],
        exc: ScreenInfrastructureAmbiguity) -> None:
    plan = item.composition_plan
    if plan is None:
        _record_screen_infrastructure_ambiguity(state, row, exc)
        return
    _record_screen_infrastructure_ambiguity(state, row, exc)
    ledger = _composition_ledger(config)
    ledger.begin(plan)
    terminal_state = ledger.rollback_infrastructure(
        plan.operation_key, reason_code="screen_infrastructure_ambiguity",
        receipt_sha256=exc.receipt_sha256)
    matches = [terminal for terminal in terminal_state["terminals"]
               if terminal["operation_key"] == plan.operation_key]
    if len(matches) != 1:
        raise DiscoveryControllerError(
            "cumulative infrastructure rollback lacks one terminal")
    retry_epoch = state["pending"].get("infrastructure_retry_epoch")
    if (isinstance(retry_epoch, bool)
            or not isinstance(retry_epoch, int) or retry_epoch <= 0):
        raise DiscoveryControllerError(
            "cumulative infrastructure retry epoch is malformed")
    retry_plan = cumulative_composition.CompositionPlan.create(
        anchor=plan.anchor, lever=plan.candidate.accepted[-1], dnr=plan.dnr,
        attempt_id=_sha({
            "schema": "epyc.autokernel.composition_retry_attempt.v1",
            "prior_operation_key": plan.operation_key,
            "retry_epoch": retry_epoch,
            "stage_receipt_sha256": exc.receipt_sha256,
        }))
    retry_item = replace(item, composition_plan=retry_plan)
    retry_row = dict(state["pending"]["row"])
    retry_row.update(
        composition_plan_sha256=retry_plan.plan_sha256,
        composition_operation_key=retry_plan.operation_key,
        anchor_patch_set_sha256=
            retry_plan.anchor.ordered_patch_set_sha256,
        candidate_patch_set_sha256=
            retry_plan.candidate.ordered_patch_set_sha256,
        composition_infrastructure_terminal_sha256=
            matches[0]["terminal_sha256"])
    state["pending"]["row"] = retry_row
    state["pending"]["candidate"] = _pending_item(retry_item)


def _bind_cumulative_terminal_row(
        state: dict[str, Any], row: dict[str, Any] | Mapping[str, Any] | None,
        terminal: Mapping[str, Any] | None = None) -> None:
    # Preserve the private two-argument seam used by historical fixture code;
    # live controller paths always pass state explicitly.
    if terminal is None and isinstance(row, Mapping):
        terminal = row
        row = state
        state = {}
    if terminal is None:
        return
    if not isinstance(row, dict):
        raise DiscoveryControllerError(
            "cumulative terminal row is not mutable")
    row.update(
        composition_disposition=terminal["disposition"],
        composition_terminal_sha256=terminal["terminal_sha256"],
        composition_scientific_budget_spent=
            terminal["scientific_budget_spent"],
        composition_promotion_eligible=terminal["promotion_eligible"],
        composition_promotion_reason=terminal["promotion_reason"],
        cumulative_performance_result_sha256=
            terminal["cumulative_performance_result_sha256"])
    state["cumulative_composition_terminal"] = dict(terminal)
    typed_ref = terminal.get("cumulative_performance_ref")
    state["cumulative_performance"] = (
        {"path": typed_ref["path"], "sha256": typed_ref["sha256"]}
        if terminal.get("disposition") == "admitted"
        and isinstance(typed_ref, Mapping) else None)


def _schedule_replication(state: dict[str, Any], *, item: PlannedCandidate,
                          authorization: hypotheses.ClaimAuthorization,
                          row: Mapping[str, Any], result: SealedScreen,
                          max_iterations: int) -> None:
    """Queue exactly one independent S2 for a positive exact series.

    Replication is not a second planner proposal.  It reuses the sealed patch,
    authorization, frame series key, and critic acceptance, then obtains a new
    resource lease at the next turn.  This supplies the evidence required for
    a nomination without conflating unrelated source patches under the same
    hypothesis.
    """
    if (item.composition_plan is not None
            or result.classification != "candidate"
            or _derived_scientific_attempts(state) >= max_iterations
            or state.get("pending") is not None):
        return
    replica = dict(row)
    replica.update(turn=state["next"], status="replication_pending",
                   replication_of=result.result_sha256,
                   series_key=result.series_key,
                   component_series_keys=list(result.component_series_keys),
                   critic={"decision": "accept",
                           "reason": "independent replication of sealed candidate"})
    state["pending"] = {
        "row": replica,
        "candidate": _pending_item(item),
        # S2 receives a fresh authorization at its own compute boundary.  The
        # original token is retained as provenance only; it is never replayed
        # as permission for a second device claim.
        "confirmation": True,
        "parent_authorization": authorization.to_dict(),
        **_preauthored_pending_fields(replica),
    }


def _apply_portfolio_outcome(state: dict[str, Any], row: dict[str, Any]) -> None:
    policy = row.get("portfolio_decision_policy")
    hypothesis_id = row.get("portfolio_hypothesis_id")
    if not isinstance(policy, Mapping) or not isinstance(hypothesis_id, str):
        return
    terminals = state.setdefault("portfolio_terminals", {})
    status = row.get("status")
    # A scientific candidate budget counts measured, recursively sealed
    # screens only.  Critic/source/authorization refusals have no result or
    # evidence graph and cannot establish the terminal claim "no gain after N
    # candidates" merely by carrying distinct proposed manifest hashes.
    measured = _row_spends_scientific_budget(row)
    if not measured:
        return
    if status == "top_k_replicated_candidate":
        terminals[hypothesis_id] = {"disposition": "nominated",
                                    "policy": dict(policy)}
        row["portfolio_disposition"] = "nominated"
        return
    if status == "inconclusive":
        conflict = policy["conflict_policy"]
        row["portfolio_disposition"] = conflict
        if conflict == "retire":
            terminals[hypothesis_id] = {"disposition": "retire_conflict",
                                        "policy": dict(policy)}
            return
    attempts = {(item.get("candidate_semantic_sha256")
                 or item.get("source_manifest_sha256"))
                for item in state["iterations"]
                if item.get("portfolio_hypothesis_id") == hypothesis_id
                and isinstance((item.get("candidate_semantic_sha256")
                                or item.get("source_manifest_sha256")), str)
                and _row_spends_scientific_budget(item)}
    if len(attempts) >= policy["max_distinct_candidates"]:
        disposition = policy["terminal_rule"]
        terminals[hypothesis_id] = {"disposition": disposition,
                                    "policy": dict(policy)}
        row["portfolio_disposition"] = disposition


def _record_precompute_refusal(state: dict[str, Any], row: dict[str, Any],
                               exc: PrecomputeScreenRefusal) -> None:
    """Commit one proven precompute rejection and consume its iteration."""
    state.pop("inflight", None)
    state.pop("pending", None)
    row.update(status="screen_refused",
               reason=f"{type(exc).__name__}: {exc}")
    state["iterations"].append(row)
    _apply_portfolio_outcome(state, row)
    _note_portfolio_authoring_failure(state, row)
    state["next"] += 1


def _record_governed_stage_refusal(
        state: dict[str, Any], row: dict[str, Any],
        exc: GovernedStageRefusal) -> None:
    """Consume one already-sealed stage terminal without replaying its work."""
    inflight = state.get("inflight")
    if isinstance(exc, TimedOutputCorrectnessRefusal):
        operation_key = (inflight.get("operation_key")
                         if isinstance(inflight, Mapping) else None)
        if (not isinstance(operation_key, str)
                or not HASH.fullmatch(operation_key)
                or operation_key != exc.operation_key):
            raise DiscoveryControllerError(
                "candidate correctness divergence changed its operation identity")
    state.pop("inflight", None)
    state.pop("pending", None)
    row.update(
        status=exc.disposition, reason=str(exc), stage=exc.stage,
        stage_receipt_path=exc.receipt_path,
        stage_receipt_sha256=exc.receipt_sha256,
        scientific_budget_spent=exc.scientific_budget_spent)
    if isinstance(exc, DispatchAttributionRefusal):
        row.update(
            classification="screened_out",
            result_sha256=exc.receipt_sha256,
            evidence={"dispatch_attribution": exc.receipt_sha256})
    if isinstance(exc, CumulativeCorrectnessRefusal):
        row.update(
            classification="screened_out",
            result_sha256=exc.correctness.result_sha256,
            evidence={"full_stack_correctness": exc.receipt_sha256})
    candidate_divergence = isinstance(exc, TimedOutputCorrectnessRefusal)
    if candidate_divergence:
        # This is a typed source-screen result, not an infrastructure refusal.
        # The stage receipt is the immutable result identity; its private raw
        # process captures retain member hashes while this row stays safe for
        # telemetry/dashboard projection.
        row.update(
            classification="screened_out",
            correctness_status="failed",
            result_sha256=exc.result_sha256,
            evidence={"correctness_divergence": exc.receipt_sha256})
        operation_key = exc.operation_key
        row["operation_key"] = operation_key
    state["iterations"].append(row)
    if exc.scientific_budget_spent:
        if isinstance(row.get("portfolio_hypothesis_id"), str):
            _record_attempted_candidate_identity(state, row)
        state["scientific_attempts"] = _derived_scientific_attempts(state)
    if exc.disposition == "authoring_refused":
        _note_portfolio_authoring_failure(state, row)
    hypothesis_id = row.get("portfolio_hypothesis_id")
    terminals = state.setdefault("portfolio_terminals", {})
    if candidate_divergence:
        _apply_portfolio_outcome(state, row)
    elif (isinstance(hypothesis_id, str)
            and exc.disposition == "correctness_falsified"):
        terminals[hypothesis_id] = {
            "disposition": exc.disposition,
            "stage_receipt_path": exc.receipt_path,
            "stage_receipt_sha256": exc.receipt_sha256,
        }
    elif (isinstance(hypothesis_id, str)
          and exc.disposition == "attribution_route_falsified"):
        manifest = row.get("source_manifest_sha256")
        policy = row.get("portfolio_decision_policy")
        if (isinstance(manifest, str) and HASH.fullmatch(manifest)
                and isinstance(policy, Mapping)):
            failures = state.setdefault(
                "portfolio_attribution_failures", {}).setdefault(
                    hypothesis_id, [])
            if manifest not in failures:
                failures.append(manifest)
            budget = policy.get("max_distinct_candidates")
            if (isinstance(budget, int) and not isinstance(budget, bool)
                    and budget > 0 and len(failures) >= budget):
                state.setdefault("portfolio_skips", {})[hypothesis_id] = {
                    "disposition": "bounded_attribution_falsified",
                    "scientific_terminal": False,
                    "distinct_candidate_count": len(failures),
                    "stage_receipt_path": exc.receipt_path,
                    "stage_receipt_sha256": exc.receipt_sha256,
                }
    elif (isinstance(hypothesis_id, str)
          and exc.disposition == "measurement_output_refused"):
        manifest = row.get("source_manifest_sha256")
        policy = row.get("portfolio_decision_policy")
        if (isinstance(manifest, str) and HASH.fullmatch(manifest)
                and isinstance(policy, Mapping)):
            failures = state.setdefault(
                "portfolio_measurement_output_failures", {}).setdefault(
                    hypothesis_id, [])
            if manifest not in failures:
                failures.append(manifest)
            budget = policy.get("max_distinct_candidates")
            if (isinstance(budget, int) and not isinstance(budget, bool)
                    and budget > 0 and len(failures) >= budget):
                state.setdefault("portfolio_skips", {})[hypothesis_id] = {
                    "disposition": "bounded_measurement_output_refused",
                    "scientific_terminal": False,
                    "distinct_candidate_count": len(failures),
                    "stage_receipt_path": exc.receipt_path,
                    "stage_receipt_sha256": exc.receipt_sha256,
                }
    state["next"] += 1


def _note_portfolio_authoring_failure(state: dict[str, Any],
                                      row: Mapping[str, Any]) -> None:
    """Bound repeated non-scientific actor failures without retiring science."""
    hypothesis_id = row.get("portfolio_hypothesis_id")
    if not isinstance(hypothesis_id, str):
        return
    failures = state.setdefault("portfolio_authoring_failures", {})
    prior = failures.get(hypothesis_id, 0)
    if not isinstance(prior, int) or isinstance(prior, bool) or prior < 0:
        raise DiscoveryControllerError(
            "portfolio authoring-failure accounting is malformed")
    count = prior + 1
    failures[hypothesis_id] = count
    if count >= 3:
        state.setdefault("portfolio_skips", {})[hypothesis_id] = {
            "disposition": "bounded_authoring_skip",
            "scientific_terminal": False,
            "failure_count": count,
        }


def _drain_visibility_degradation(
        state: dict[str, Any], actor: object,
        row: dict[str, Any] | None = None) -> list[dict[str, str]]:
    failures = getattr(actor, "telemetry_failures", None)
    if not isinstance(failures, list) or not failures:
        return []
    drained = [dict(item) for item in failures]
    failures.clear()
    durable = state.setdefault("visibility_degraded", [])
    for item in drained:
        if item not in durable:
            durable.append(item)
    if row is not None:
        row["visibility_degraded"] = True
        row_failures = row.setdefault("telemetry_failures", [])
        for item in drained:
            if item not in row_failures:
                row_failures.append(item)
    return drained


def _record_planner_refusal(state: dict[str, Any], *, turn: int,
                            context: Mapping[str, Any],
                            portfolio_binding: Mapping[str, Any] | None,
                            exc: PlannerOutputRefusal) -> None:
    """Persist one non-candidate authoring refusal without spending science budget."""
    row: dict[str, Any] = {
        "turn": turn,
        "status": ("planner_transient" if isinstance(exc, PlannerProviderTransient)
                   else "planner_refused"),
        "reason": str(exc),
        "refusal_type": ("planner_provider_transient"
                         if isinstance(exc, PlannerProviderTransient)
                         else "planner_output_refusal"),
        "scientific_budget_spent": False,
        "context_sha256": _sha(context),
    }
    if not isinstance(exc, PlannerProviderTransient):
        row["telemetry_event"] = "planner_refused"
        row["telemetry_status"] = exc.telemetry_status
        if exc.telemetry_failure is not None:
            row["telemetry_failure"] = dict(exc.telemetry_failure)
    planning = state.pop("planning", None)
    if isinstance(planning, Mapping):
        row["planner_operation_key"] = planning.get("operation_key")
        if isinstance(planning.get("telemetry_recovery"), Mapping):
            row["planner_checkpoint_reused"] = True
            row["telemetry_recovery"] = dict(
                planning["telemetry_recovery"])
    if portfolio_binding is not None:
        row.update(
            hypothesis_id=portfolio_binding["hypothesis_id"],
            statement=portfolio_binding["statement"],
            falsifier=portfolio_binding["falsifier"],
            regime=dict(portfolio_binding["regime"]),
            portfolio_hypothesis_id=portfolio_binding["hypothesis_id"],
            portfolio_binding=dict(portfolio_binding),
            portfolio_record_sha256=portfolio_binding["record_sha256"],
            portfolio_decision_policy=dict(
                portfolio_binding["decision_policy"]),
        )
    state["iterations"].append(row)
    if isinstance(exc, PlannerProviderTransient):
        # Provider/API availability is neither authored output nor a scientific
        # attempt.  Keep the controller turn and portfolio assignment, but use
        # a fresh sealed actor operation on the next pass.
        state["planner_provider_attempt"] = int(
            state.get("planner_provider_attempt", 0)) + 1
    else:
        _note_portfolio_authoring_failure(state, row)
        state["next"] += 1


def _planning_intent(config: ControllerConfig, *, turn: int,
                     context: Mapping[str, Any],
                     portfolio_binding: Mapping[str, Any] | None,
                     provider_attempt: int = 0) -> dict[str, Any]:
    if isinstance(provider_attempt, bool) or provider_attempt < 0:
        raise DiscoveryControllerError("planner provider attempt is invalid")
    context_sha256 = _sha(context)
    operation_key = _sha({
        "schema": "epyc.autokernel.planning_operation.v1",
        "turn": turn,
        "context_sha256": context_sha256,
        "deployment_identity_sha256": config.deployment_identity_sha256,
        "provider_attempt": provider_attempt,
    })
    workspace = (config.output_root / "planner-operations" /
                 operation_key / "workspace")
    return {
        "phase": "intent", "turn": turn,
        "provider_attempt": provider_attempt,
        "operation_key": operation_key,
        "context": dict(context), "context_sha256": context_sha256,
        "portfolio_binding": (None if portfolio_binding is None
                              else dict(portfolio_binding)),
        "workspace": str(workspace),
    }


def _is_legacy_planner_refusal_telemetry_failure(
        planning: Mapping[str, Any]) -> bool:
    """Recognize only the v16 telemetry-schema crash after actor checkpoint.

    The checkpoint and every actor artifact are independently revalidated by
    the caller before this legacy marker may be cleared.
    """
    failure = planning.get("failure")
    return (planning.get("phase") == "actor_entering"
            and isinstance(failure, Mapping)
            and set(failure) == {"type", "message"}
            and failure.get("type") == "TelemetryError"
            and failure.get("message") ==
            "telemetry result contains a non-allowlisted field")


def _require_legacy_planner_success_result(
        checkpoint: Mapping[str, Any]) -> None:
    """Bind the v16 recovery exception to the exact completed actor result."""
    if set(checkpoint) != {
            "schema", "context_sha256", "assignment_sha256", "result",
            "artifacts", "receipt_sha256"}:
        raise DiscoveryControllerError(
            "legacy planner telemetry recovery checkpoint schema changed")
    result = checkpoint.get("result")
    required = {"returncode", "stdout_sha256", "stderr_sha256"}
    if (not isinstance(result, Mapping) or set(result) != required
            or result.get("returncode") != 0
            or any(not isinstance(result.get(key), str)
                   or not HASH.fullmatch(result[key])
                   for key in ("stdout_sha256", "stderr_sha256"))):
        raise DiscoveryControllerError(
            "legacy planner telemetry recovery lacks its exact rc=0 actor result")


def _prepare_planner_workspace(config: ControllerConfig, operation_key: str,
                               workspace: Path) -> bool:
    """Create the exact persistent actor workspace without following links."""
    operations = config.output_root / "planner-operations"
    operation = operations / operation_key
    if workspace != operation / "workspace":
        raise DiscoveryControllerError(
            "durable planner workspace escaped its operation namespace")
    ReviewedSourcePackage._require_owned_directory(
        config.output_root, "controller state root")
    if not operations.exists():
        operations.mkdir(mode=0o700)
    ReviewedSourcePackage._require_owned_directory(
        operations, "planner operations root")
    if not operation.exists():
        operation.mkdir(mode=0o700)
    ReviewedSourcePackage._require_owned_directory(
        operation, "planner operation root")
    if workspace.exists() or workspace.is_symlink():
        ReviewedSourcePackage._require_owned_directory(
            workspace, "planner workspace")
        return False
    workspace.mkdir(mode=0o700)
    ReviewedSourcePackage._require_owned_directory(
        workspace, "planner workspace")
    return True


def _reopen_planning_intent(state: Mapping[str, Any], *,
                            turn: int) -> tuple[dict[str, Any],
                                                Mapping[str, Any] | None]:
    planning = state.get("planning")
    if (not isinstance(planning, Mapping) or planning.get("turn") != turn
            or planning.get("phase") not in {"intent", "actor_entering"}
            or not isinstance(planning.get("context"), Mapping)
            or planning.get("context_sha256") != _sha(planning["context"])
            or not isinstance(planning.get("operation_key"), str)
            or not HASH.fullmatch(planning["operation_key"])):
        raise DiscoveryControllerError("durable planning intent is malformed")
    binding = planning.get("portfolio_binding")
    if binding is not None and not isinstance(binding, Mapping):
        raise DiscoveryControllerError("durable planning portfolio binding is malformed")
    return dict(planning["context"]), binding


def run_controller(config: ControllerConfig, *, planner: Planner, critic: Critic, screener: Screener, lease: Lease) -> dict[str, Any]:
    planner_attestation, critic_attestation = dict(planner.attest()), dict(critic.attest())
    if ({k: planner_attestation.get(k) for k in SOL} != SOL
            or {k: critic_attestation.get(k) for k in FABLE5_CRITIC} != FABLE5_CRITIC
            or not isinstance(planner_attestation.get("runtime"), Mapping)
            or not isinstance(critic_attestation.get("runtime"), Mapping)):
        raise DiscoveryControllerError("actors did not attest the sealed planner/critic runtime identities")
    _require_runtime(planner_attestation["runtime"])
    _require_claude_runtime(critic_attestation["runtime"])
    _require_roster(sealed_roster())
    store=DurableState(config.output_root); lock=store.run_lock()
    try:
        return _run_controller_locked(config,planner=planner,critic=critic,screener=screener,lease=lease,store=store)
    finally:
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN); lock.close()

def _run_controller_locked(config: ControllerConfig, *, planner: Planner, critic: Critic, screener: Screener, lease: Lease, store: DurableState) -> dict[str, Any]:
    state=store.load()
    existing_deployment = state.get("deployment_identity_sha256")
    if (existing_deployment is None and config.deployment_identity_sha256 is not None
            and (state.get("iterations") or state.get("pending") is not None
                 or state.get("inflight") is not None
                 or state.get("planning") is not None)):
        raise DiscoveryControllerError("legacy durable state lacks deployment identity; refusing resume")
    if existing_deployment is not None and existing_deployment != config.deployment_identity_sha256:
        raise DiscoveryControllerError("sealed deployment identity changed; durable discovery cannot resume")
    if existing_deployment is None and config.deployment_identity_sha256 is not None:
        state["deployment_identity_sha256"] = config.deployment_identity_sha256
    existing_context = state.get("planner_context_sha256")
    if existing_context is not None and existing_context != config.planner_context_sha256:
        raise DiscoveryControllerError("sealed planner context changed; durable discovery cannot resume")
    if existing_context is None and config.planner_context_sha256 is not None:
        state["planner_context_sha256"] = config.planner_context_sha256
    existing_templates = state.get("experiment_template_registry_sha256")
    if existing_templates is not None and existing_templates != config.experiment_template_registry_sha256:
        raise DiscoveryControllerError("sealed experiment-template registry changed; durable discovery cannot resume")
    if existing_templates is None and config.experiment_template_registry_sha256 is not None:
        state["experiment_template_registry_sha256"] = config.experiment_template_registry_sha256
    existing_corpus = state.get("admission_corpus_sha256")
    if existing_corpus is not None and existing_corpus != config.admission_corpus_sha256:
        raise DiscoveryControllerError("sealed admission corpus changed; durable discovery cannot resume")
    if existing_corpus is None and config.admission_corpus_sha256 is not None:
        state["admission_corpus_sha256"] = config.admission_corpus_sha256
    existing_corpus_version = state.get("admission_corpus_version")
    if existing_corpus_version is not None and existing_corpus_version != config.admission_corpus_version:
        raise DiscoveryControllerError("sealed admission corpus version changed; durable discovery cannot resume")
    if existing_corpus_version is None and config.admission_corpus_version is not None:
        state["admission_corpus_version"] = config.admission_corpus_version
    existing_portfolio = state.get("hypothesis_portfolio_sha256")
    if existing_portfolio is not None and existing_portfolio != config.hypothesis_portfolio_sha256:
        raise DiscoveryControllerError(
            "sealed hypothesis portfolio changed; durable discovery cannot resume")
    if existing_portfolio is None and config.hypothesis_portfolio_sha256 is not None:
        state["hypothesis_portfolio_sha256"] = config.hypothesis_portfolio_sha256
    existing_carry_forward = state.get("carry_forward_sha256")
    if (existing_carry_forward is not None
            and existing_carry_forward != config.carry_forward_sha256):
        raise DiscoveryControllerError(
            "predecessor carry-forward changed; durable discovery cannot resume")
    if existing_carry_forward is None and config.carry_forward_sha256 is not None:
        if (state.get("iterations") or state.get("pending") is not None
                or state.get("inflight") is not None
                or state.get("planning") is not None):
            raise DiscoveryControllerError(
                "legacy durable state lacks predecessor carry-forward")
        state["carry_forward_sha256"] = config.carry_forward_sha256
    configured_continuation_sha256 = (
        None if config.preauthored_continuations is None else
        config.preauthored_continuations[
            "akh-v2-q5-onewave-preauthored"].sha256)
    configured_source_backed_diff_sha256 = (
        None if config.preauthored_continuations is None else
        config.preauthored_continuations[
            "akh-v2-q5-onewave-preauthored"].source_backed_diff_sha256)
    existing_continuation = state.get("preauthored_continuation_sha256")
    if (existing_continuation is not None
            and existing_continuation != configured_continuation_sha256):
        raise DiscoveryControllerError(
            "preauthored continuation changed; durable discovery cannot resume")
    if (existing_continuation is None
            and configured_continuation_sha256 is not None):
        if (state.get("iterations") or state.get("pending") is not None
                or state.get("inflight") is not None
                or state.get("planning") is not None):
            raise DiscoveryControllerError(
                "legacy durable state lacks preauthored continuation authority")
        state["preauthored_continuation_sha256"] = (
            configured_continuation_sha256)
    existing_source_backed_diff = state.get(
        "preauthored_source_backed_diff_sha256")
    if (existing_source_backed_diff is not None
            and existing_source_backed_diff !=
                configured_source_backed_diff_sha256):
        raise DiscoveryControllerError(
            "preauthored source-backed diff changed; discovery cannot resume")
    if (existing_source_backed_diff is None
            and configured_source_backed_diff_sha256 is not None):
        if (state.get("iterations") or state.get("pending") is not None
                or state.get("inflight") is not None
                or state.get("planning") is not None):
            raise DiscoveryControllerError(
                "legacy durable state lacks preauthored source-backed authority")
        state["preauthored_source_backed_diff_sha256"] = (
            configured_source_backed_diff_sha256)
    _validate_portfolio_authoring_failures(state)
    _validate_attempted_candidate_identities(state)
    _validate_infrastructure_ambiguities(state)
    _validate_replicated_positive_levers(state)
    _validate_cumulative_composition_state(config, state)
    # A completed state is an acknowledged terminal checkpoint.  Re-entering it
    # must be a read, not another executor opportunity or a timestamp rewrite.
    if state["complete"]: return state
    _bind_scientific_attempt_counter(state)
    tracker=_tracker(store)
    if state.get("inflight") is not None:
        precompute_refused = False
        inflight=state["inflight"]; item=_restore_pending({"candidate":inflight["candidate"], "row":inflight.get("row"), "preauthored_continuation":inflight.get("preauthored_continuation")}, config); authorization=hypotheses.ClaimAuthorization.from_dict(inflight["authorization"]); permit=inflight["lease"]
        inflight_row = dict(inflight["row"])
        _revalidate_portfolio_checkpoint(config, item, inflight_row)
        _require_unattempted_checkpoint(
            state, inflight_row,
            confirmation=bool(inflight.get("confirmation")))
        _bind_campaign_ledger_outcome(inflight_row, authorization)
        if (config.hypothesis_portfolio is not None
                and inflight_row != dict(inflight["row"])):
            raise DiscoveryControllerError(
                "inflight DNR outcomes differ from durable authorization")
        if isinstance(inflight.get("result"),Mapping):
            result=_sealed_screen_from_dict(inflight["result"])
        else:
            reconcile=getattr(screener,"reconcile",None)
            if not callable(reconcile): raise DiscoveryControllerError("inflight operation has no reconciliation adapter")
            recovery=reconcile(inflight)
            if not isinstance(recovery,Recovery) or recovery.status == "ambiguous": raise DiscoveryControllerError("inflight operation cannot be safely reconciled")
            if recovery.status == "sealed_result":
                result=recovery.result
            elif recovery.status == "resource_wait":
                wait_receipt = _validated_resource_wait(
                    ResourceWait(
                        "reopen durable resource wait",
                        receipt=dict(recovery.wait_receipt or {})),
                    str(inflight["operation_key"]))
                pending_wait = _resource_wait_pending(
                    inflight, wait_receipt)
                state.pop("inflight", None)
                state["pending"] = pending_wait
                store.save(state, "waiting_resource")
                return state
            else:
                resume=getattr(lease,"resume",None)
                if not callable(resume):
                    raise DiscoveryControllerError("safe inflight recovery lacks resource re-admission")
                fresh_permit=resume(item,permit)
                if not bool(fresh_permit.get("admitted")):
                    row=dict(inflight["row"]); row.update(
                        status="waiting_resource",lease=dict(fresh_permit))
                    state.pop("inflight",None)
                    state["pending"]={
                        "row":row,"candidate":inflight["candidate"],
                        "authorization":inflight["authorization"],
                        "confirmation":bool(inflight.get("confirmation")),
                        "parent_authorization":inflight.get("parent_authorization"),
                        "infrastructure_retry_epoch":inflight.get(
                            "infrastructure_retry_epoch", 0),
                        **_preauthored_pending_fields(row)}
                    store.save(state,"waiting_resource")
                    return state
                fresh_permit={**dict(fresh_permit),
                              "repetition":permit.get("repetition")}
                inflight["lease"]=fresh_permit; permit=fresh_permit
                store.save(state,"pre_screen_reacquired")
                try:
                    result=screener.screen(item,authorization,permit)
                except ResumableScreenInterruption as exc:
                    inflight["interruption"] = {
                        "type": type(exc).__name__, "message": str(exc),
                        "resumable": True,
                    }
                    store.save(state, "screen_resumable_interruption")
                    return state
                except ResourceWait as exc:
                    wait_receipt=_validated_resource_wait(
                        exc,str(inflight["operation_key"]))
                    _require_safe_resource_wait_recovery(
                        screener, inflight, wait_receipt)
                    pending_wait = _resource_wait_pending(
                        inflight, wait_receipt)
                    state.pop("inflight",None)
                    state["pending"] = pending_wait
                    store.save(state,"waiting_resource")
                    return state
                except ScreenInfrastructureAmbiguity as exc:
                    row = dict(inflight["row"])
                    _record_cumulative_infrastructure_ambiguity(
                        state, config=config, item=item, row=row, exc=exc)
                    store.save(state, "screen_infrastructure_ambiguity")
                    return state
                except PrecomputeScreenRefusal as exc:
                    row = dict(inflight["row"])
                    _record_precompute_refusal(state, row, exc)
                    store.save(state, "screen_refused")
                    precompute_refused = True
                except GovernedStageRefusal as exc:
                    row = dict(inflight["row"])
                    _bind_cumulative_terminal_row(
                        state, row, _terminalize_cumulative_refusal(
                            config, item, exc))
                    _record_governed_stage_refusal(state, row, exc)
                    store.save(state, exc.disposition)
                    precompute_refused = True
        if not precompute_refused:
            if not isinstance(result,SealedScreen): raise DiscoveryControllerError("inflight recovery produced no sealed result")
            row=dict(inflight["row"]); policy=row.get("portfolio_decision_policy")
            row["operation_key"] = inflight["operation_key"]
            composition_terminal = _finalize_cumulative_screen(
                config, item, result)
            result=_classified_result(state,item,result,policy); row.update(
                _screen_iteration_fields(
                    result, repetition=int(inflight["lease"].get(
                        "repetition", 2 if inflight.get("confirmation") else 1))))
            if composition_terminal is not None:
                _bind_cumulative_terminal_row(
                    state, row, composition_terminal)
            _record_attempt_once(tracker,item,str(item.proposal.get("proposal_id",row["proposal_sha256"])),result)
            state.pop("inflight",None); state.pop("pending",None); state["iterations"].append(row)
            if isinstance(row.get("portfolio_hypothesis_id"), str):
                _record_attempted_candidate_identity(state, row)
            state["scientific_attempts"]=_derived_scientific_attempts(state); _apply_portfolio_outcome(state,row); state["next"]+=1; _schedule_replication(state,item=item,authorization=authorization,row=row,result=result,max_iterations=config.max_iterations); _schedule_cumulative_composition(state,config=config,item=item,result=result); _append_nomination(config.output_root,item,result,_decision_floor(policy,"nomination_floor_pct",config.nomination_threshold)); _write_projection(config.evidence_root or config.output_root); store.save(state,"recovered_screen")
    while (not state["complete"]
           and (state["scientific_attempts"] < config.max_iterations
                if config.hypothesis_portfolio is not None
                else state["next"] <= config.max_iterations)):
        turn=state["next"]
        pending=state.get("pending")
        planning=state.get("planning")
        if pending is not None and planning is not None:
            raise DiscoveryControllerError(
                "controller cannot own pending candidate and planning intent together")
        if planning is not None:
            context, portfolio_binding = _reopen_planning_intent(
                state, turn=turn)
        else:
            portfolio_binding = (pending.get("row", {}).get("portfolio_binding")
                                 if pending is not None else
                                 _select_portfolio_binding(state, config))
            if (pending is None and config.hypothesis_portfolio is not None
                    and portfolio_binding is None):
                state["complete"] = True
                state["terminal_reason"] = "portfolio_exhausted"
                store.save(state, "portfolio_exhausted")
                break
            if (pending is not None and isinstance(pending.get("context"), Mapping)):
                context = dict(pending["context"])
                if pending.get("context_sha256") != _sha(context):
                    raise DiscoveryControllerError(
                        "pending actor context identity changed")
            else:
                context=_context(state,tracker,turn,config,portfolio_binding)
            if (pending is None and isinstance(portfolio_binding, Mapping)
                    and portfolio_binding.get("hypothesis_id") ==
                        "akh-v2-q5-onewave-preauthored"):
                item = _preauthored_candidate(config, portfolio_binding, turn)
                row = {
                    "turn": turn, "hypothesis_id": item.hypothesis_id,
                    "authoring_turn": turn,
                    "statement": item.statement, "falsifier": item.falsifier,
                    "regime": dict(item.regime),
                    "proposal_sha256": _sha(item.proposal),
                    "source_manifest_sha256": item.source_manifest_sha256,
                    "experiment_intent": asdict(item.experiment_intent),
                    "mechanism_id": item.source_manifest.mechanism_id,
                    "target_surface": item.experiment_intent.target_surface,
                    "target_symbol": item.experiment_intent.target_symbol,
                    "context_sha256": _sha(context),
                    "candidate_semantic_sha256":
                        _candidate_semantic_identity(item),
                    "portfolio_hypothesis_id":
                        portfolio_binding["hypothesis_id"],
                    "portfolio_binding": dict(portfolio_binding),
                    "portfolio_record_sha256":
                        portfolio_binding["record_sha256"],
                    "portfolio_decision_policy": dict(
                        portfolio_binding["decision_policy"]),
                }
                _validate_portfolio_candidate(
                    item, portfolio_binding, config.hypothesis_portfolio,
                    config.carry_forward)
                receipt = _portfolio_exact_dnr_check(
                    config, item, portfolio_binding)
                row["portfolio_exact_dnr_check"] = receipt
                if receipt["outcome"] == schemas.FAIL:
                    row.update(
                        status="portfolio_dnr_refused",
                        reason="candidate exactly repeats sealed portfolio DNR "
                               + ", ".join(receipt["matched_dnr_ids"]))
                    state["iterations"].append(row)
                    state.setdefault("portfolio_terminals", {})[
                        portfolio_binding["hypothesis_id"]] = {
                            "disposition": "portfolio_dnr_refused",
                            "policy": dict(portfolio_binding[
                                "decision_policy"]),
                            "receipt_sha256": receipt["receipt_sha256"],
                        }
                    state["next"] += 1
                    store.save(state, "portfolio_dnr_refused")
                    continue
                semantic = row["candidate_semantic_sha256"]
                if semantic in state.get("attempted_candidate_identities", {}):
                    row.update(
                        status="candidate_semantic_repeat_refused",
                        reason="preauthored source semantics already have a scientific outcome")
                    state["iterations"].append(row)
                    _note_portfolio_authoring_failure(state, row)
                    state["next"] += 1
                    store.save(state, "candidate_semantic_repeat_refused")
                    continue
                authority = _preauthored_checkpoint_authority(config, item)
                row.update(
                    preauthored_continuation=dict(authority),
                    hypothesis_origin=hypotheses.ORIGIN_IMPORT,
                    hypothesis_author="reviewed-eb26918-continuation",
                    historical_correctness_authority="provenance_only",
                    modern_governed_correctness_required=True)
                state["pending"] = {
                    "phase": "preauthored_ready", "row": row,
                    "candidate": _pending_item(item),
                    "preauthored_continuation": dict(authority),
                    "context": dict(context), "context_sha256": _sha(context),
                    "confirmation": False, "parent_authorization": None,
                }
                store.save(state, "preauthored_checkpointed")
                continue
            if pending is None:
                state["planning"] = _planning_intent(
                    config, turn=turn, context=context,
                    portfolio_binding=portfolio_binding,
                    provider_attempt=int(
                        state.get("planner_provider_attempt", 0)))
                store.save(state, "planner_intent")
                planning = state["planning"]
        with tempfile.TemporaryDirectory(prefix=f"ak-discovery-{turn}-", dir=config.output_root) as temp:
            workspace=Path(temp)
            pending_phase = pending.get("phase") if isinstance(pending, Mapping) else None
            if pending is not None and pending_phase == "critic_pending":
                item=_restore_pending(pending, config); row=dict(pending["row"])
                _revalidate_portfolio_checkpoint(config, item, row)
                _require_unattempted_checkpoint(
                    state, row,
                    confirmation=bool(pending.get("confirmation")))
                try:
                    review=critic.review(item,context=context,workspace=workspace)
                except Exception:
                    if _drain_visibility_degradation(state, critic, row):
                        state["pending"]["row"] = row
                        store.save(state, "visibility_degraded")
                    raise
                _drain_visibility_degradation(state, critic, row)
                row["critic"]=asdict(review)
                if review.decision != "accept":
                    row["status"]="critic_"+review.decision
                    state.pop("pending", None); state["iterations"].append(row)
                    _apply_portfolio_outcome(state,row)
                    _note_portfolio_authoring_failure(state, row)
                    state["next"]+=1
                    store.save(state,"critic_refused"); continue
                state["pending"]={
                    "phase":"critic_complete", "row":row,
                    "candidate":pending["candidate"],
                    "context":dict(context), "context_sha256":_sha(context),
                    "confirmation":False, "parent_authorization":None}
                store.save(state,"critic_checkpointed")
                continue
            if pending is not None:
                item=_restore_pending(pending, config); row=dict(pending["row"])
                review = (Critique(
                              "accept",
                              "controller-owned cumulative composition authority")
                          if pending_phase == "cumulative_ready" else
                          Critique("accept", "controller-owned reviewed preauthored continuation")
                          if pending_phase == "preauthored_ready" else
                          Critique(**row["critic"]))
                _revalidate_portfolio_checkpoint(config, item, row)
                _require_unattempted_checkpoint(
                    state, row,
                    confirmation=bool(pending.get("confirmation")))
                if "authorization" in pending:
                    authorization=hypotheses.ClaimAuthorization.from_dict(pending["authorization"])
                    durable_row = dict(row)
                    _bind_campaign_ledger_outcome(row, authorization)
                    if config.hypothesis_portfolio is not None and row != durable_row:
                        raise DiscoveryControllerError(
                            "portfolio pending candidate lacks campaign-ledger DNR outcome")
                elif pending_phase in {
                        "critic_complete", "preauthored_ready",
                        "cumulative_ready"}:
                    authorization=None
                elif pending.get("confirmation") is True:
                    # A positive S1 is not a receipted negative.  Re-consult
                    # DNR and mint the explicit confirmation token before its
                    # own device claim, rather than reusing S1's token.
                    _ensure_question(
                        tracker, item,
                        row.get("portfolio_binding")
                        if isinstance(row.get("portfolio_binding"), Mapping) else None,
                        pending.get("preauthored_continuation")
                        if isinstance(pending.get("preauthored_continuation"), Mapping)
                        else None)
                    authorization=tracker.authorize_claim(item.hypothesis_id,purpose="candidate_only_confirmation",authorized_by="discovery_controller",ledger=do_not_repeat.compile_for_tracker(tracker))
                    _bind_campaign_ledger_outcome(row, authorization)
                else:
                    raise DiscoveryControllerError("pending candidate lacks a sealed authorization")
            else:
                planning = state["planning"]
                planner_workspace=Path(str(planning["workspace"]))
                expected_workspace=(config.output_root / "planner-operations" /
                                    planning["operation_key"] / "workspace")
                if planner_workspace != expected_workspace:
                    raise DiscoveryControllerError(
                        "durable planner workspace escaped its operation namespace")
                checkpoint_path=planner_workspace.parent / "actor-result.json"
                if isinstance(planning.get("failure"), Mapping):
                    if _is_legacy_planner_refusal_telemetry_failure(planning):
                        # This exact historical failure occurred after rc=0 was
                        # checkpointed and while emitting the typed refusal.
                        # Validate the private, single-link actor closure now;
                        # a missing/extra/tampered artifact stays terminal.
                        checkpoint = _reopen_planner_actor_checkpoint(
                            planner_workspace, checkpoint_path,
                            context=context)
                        _require_legacy_planner_success_result(checkpoint)
                        planning.pop("failure")
                        planning["telemetry_recovery"] = {
                            "schema": "epyc.autokernel.planner_telemetry_recovery.v1",
                            "disposition": "resume_checkpoint_and_rederive_refusal",
                        }
                        store.save(state, "planner_telemetry_recovery")
                    else:
                        raise DiscoveryControllerError(
                            "prior planner infrastructure/authority failure remains terminal: "
                            f"{planning['failure'].get('type')}: "
                            f"{planning['failure'].get('message')}")
                if planning["phase"] == "intent":
                    planning["phase"] = "actor_entering"
                    store.save(state, "planner_entering")
                try:
                    workspace_created=_prepare_planner_workspace(
                        config, planning["operation_key"], planner_workspace)
                    if not workspace_created:
                        resume_plan=getattr(planner,"resume_plan",None)
                        if not callable(resume_plan):
                            raise PlannerOutputRefusal(
                                "planner stopped before a reusable actor checkpoint")
                        resume_kwargs={"context":context,"workspace":planner_workspace}
                        if "checkpoint_path" in inspect.signature(resume_plan).parameters:
                            resume_kwargs["checkpoint_path"]=checkpoint_path
                        item=resume_plan(**resume_kwargs)
                    else:
                        plan_kwargs={"context":context,"workspace":planner_workspace}
                        if "checkpoint_path" in inspect.signature(planner.plan).parameters:
                            plan_kwargs["checkpoint_path"]=checkpoint_path
                        item=planner.plan(**plan_kwargs)
                except PlannerOutputRefusal as exc:
                    _record_planner_refusal(
                        state, turn=turn, context=context,
                        portfolio_binding=portfolio_binding, exc=exc)
                    _drain_visibility_degradation(
                        state, planner, state["iterations"][-1])
                    store.save(
                        state,
                        "planner_transient"
                        if isinstance(exc, PlannerProviderTransient)
                        else "planner_refused")
                    continue
                except Exception as exc:
                    state["planning"]["failure"]={
                        "type":type(exc).__name__, "message":str(exc)}
                    _drain_visibility_degradation(
                        state, planner, state["planning"])
                    store.save(state,"planner_terminal_failure")
                    raise
                row={"turn":turn,"hypothesis_id":item.hypothesis_id,"statement":item.statement,
                     "falsifier":item.falsifier,"regime":dict(item.regime),
                     "proposal_sha256":_sha(item.proposal),"source_manifest_sha256":item.source_manifest_sha256,
                     "experiment_intent":asdict(item.experiment_intent) if item.experiment_intent else None,
                     "mechanism_id":item.source_manifest.mechanism_id,
                     "target_surface":item.experiment_intent.target_surface if item.experiment_intent else None,
                     "target_symbol":item.experiment_intent.target_symbol if item.experiment_intent else None,
                     "context_sha256":_sha(context),
                     "candidate_semantic_sha256":_candidate_semantic_identity(item)}
                _drain_visibility_degradation(state, planner, row)
                if portfolio_binding is not None:
                    row.update(portfolio_hypothesis_id=portfolio_binding["hypothesis_id"],
                               portfolio_binding=dict(portfolio_binding),
                               portfolio_record_sha256=portfolio_binding["record_sha256"],
                               portfolio_decision_policy=dict(
                                   portfolio_binding["decision_policy"]))
                    try:
                        _validate_portfolio_candidate(
                            item, portfolio_binding,
                            config.hypothesis_portfolio,
                            config.carry_forward)
                    except DiscoveryControllerError as exc:
                        row.update(status="planner_contract_refused",
                                   reason=str(exc))
                        state.pop("planning", None)
                        state["iterations"].append(row)
                        _note_portfolio_authoring_failure(state, row)
                        state["next"] += 1
                        store.save(state, "planner_contract_refused")
                        continue
                    receipt = _portfolio_exact_dnr_check(
                        config, item, portfolio_binding)
                    row["portfolio_exact_dnr_check"] = receipt
                    if receipt["outcome"] == schemas.FAIL:
                        row.update(
                            status="portfolio_dnr_refused",
                            reason="candidate exactly repeats sealed portfolio DNR "
                                   + ", ".join(receipt["matched_dnr_ids"]))
                        state.pop("planning", None)
                        state["iterations"].append(row)
                        state.setdefault("portfolio_terminals", {})[
                            portfolio_binding["hypothesis_id"]] = {
                                "disposition": "portfolio_dnr_refused",
                                "policy": dict(portfolio_binding["decision_policy"]),
                                "receipt_sha256": receipt["receipt_sha256"],
                            }
                        state["next"] += 1
                        store.save(state, "portfolio_dnr_refused")
                        continue
                    semantic = row["candidate_semantic_sha256"]
                    prior_semantics = state.get(
                        "attempted_candidate_identities", {})
                    if semantic in prior_semantics:
                        row.update(
                            status="candidate_semantic_repeat_refused",
                            reason=("candidate patch/base/mechanism/scope exactly "
                                    "repeats a prior scientific candidate"))
                        state.pop("planning", None)
                        state["iterations"].append(row)
                        _note_portfolio_authoring_failure(state, row)
                        state["next"] += 1
                        store.save(state, "candidate_semantic_repeat_refused")
                        continue
                state.pop("planning", None)
                state["pending"]={
                    "phase":"critic_pending", "row":row,
                    "candidate":_pending_item(item),
                    "context":dict(context), "context_sha256":_sha(context),
                    "confirmation":False, "parent_authorization":None}
                store.save(state,"planner_checkpointed")
                continue
            if review.decision != "accept":
                row["status"]="critic_"+review.decision; state["iterations"].append(row); _apply_portfolio_outcome(state,row); _note_portfolio_authoring_failure(state,row); state["next"]+=1; store.save(state,"critic_refused"); continue
            if pending_phase in {
                    "critic_complete", "preauthored_ready",
                    "cumulative_ready"}:
                _ensure_question(
                    tracker, item,
                    row.get("portfolio_binding")
                    if isinstance(row.get("portfolio_binding"), Mapping) else None,
                    pending.get("preauthored_continuation")
                    if isinstance(pending.get("preauthored_continuation"), Mapping)
                    else None)
                ledger=do_not_repeat.compile_for_tracker(tracker)
                try:
                    authorization=tracker.authorize_claim(
                        item.hypothesis_id,
                        purpose=("cumulative_composition"
                                 if item.composition_plan is not None else
                                 "candidate_only_discovery"),
                        authorized_by="discovery_controller",ledger=ledger)
                    _bind_campaign_ledger_outcome(row, authorization)
                except hypotheses.RepeatsAReceiptedNegative as exc:
                    row.update(campaign_ledger_dnr_outcome=schemas.FAIL,
                               campaign_ledger_dnr_reasons=[str(exc)],
                               status="authorization_refused",reason=str(exc)); state.pop("pending",None); state["iterations"].append(row); _apply_portfolio_outcome(state,row); _note_portfolio_authoring_failure(state,row); state["next"]+=1; store.save(state,"authorization_refused"); continue
                except hypotheses.HypothesisError as exc:
                    row.update(status="authorization_refused",reason=str(exc)); state.pop("pending",None); state["iterations"].append(row); _apply_portfolio_outcome(state,row); _note_portfolio_authoring_failure(state,row); state["next"]+=1; store.save(state,"authorization_refused"); continue
            if config.dry_run:
                # The dry-run still proves exact Sol/Terra actor attestation,
                # plan schema, critic binding, and DNR authorization.  It
                # deliberately never asks for a resource lease or starts a
                # source build, correctness, attribution, or model call.
                row.update(status="dry_run_authorized", authorization=authorization.to_dict())
                state.pop("pending", None); state["iterations"].append(row); _apply_portfolio_outcome(state,row)
                dry_hypothesis = row.get("portfolio_hypothesis_id")
                if isinstance(dry_hypothesis, str):
                    state.setdefault("portfolio_validations", {})[dry_hypothesis] = {
                        "disposition": "dry_run_validated",
                        "scientific_terminal": False,
                    }
                state["next"] += 1
                if (config.hypothesis_portfolio is not None
                        and _select_portfolio_binding(state, config) is None):
                    state["complete"] = True
                    state["terminal_reason"] = "portfolio_exhausted"
                store.save(state, "dry_run_authorized")
                continue
            repetition=2 if pending and pending.get("confirmation") else 1
            retry_epoch = (pending.get("infrastructure_retry_epoch", 0)
                           if isinstance(pending, Mapping) else 0)
            if (isinstance(retry_epoch, bool)
                    or not isinstance(retry_epoch, int) or retry_epoch < 0):
                raise DiscoveryControllerError(
                    "pending infrastructure retry epoch is malformed")
            operation_identity = {"turn":turn,
                                  "manifest":item.source_manifest_sha256,
                                  "authorization":authorization.to_dict(),
                                  "repetition":repetition}
            if item.composition_plan is not None:
                operation_identity["composition_plan_sha256"] = (
                    item.composition_plan.plan_sha256)
            if retry_epoch:
                operation_identity["infrastructure_retry_epoch"] = retry_epoch
            operation_key=_sha(operation_identity)
            row["operation_key"] = operation_key
            has_wait_checkpoint = (isinstance(pending, Mapping)
                                   and "resource_wait" in pending)
            if has_wait_checkpoint:
                checkpoint = _validated_resource_wait_checkpoint(
                    pending, operation_key)
                _require_safe_resource_wait_recovery(
                    screener, checkpoint["inflight"],
                    checkpoint["wait_receipt"])
                resume = getattr(lease, "resume", None)
                if not callable(resume):
                    raise DiscoveryControllerError(
                        "pending resource wait lacks resource re-admission")
                permit = resume(item, checkpoint["resume_permit"])
            else:
                pending_row = (pending.get("row")
                               if isinstance(pending, Mapping) else None)
                if (isinstance(pending_row, Mapping)
                        and pending_row.get("status") == "waiting_resource"):
                    _validated_prebuild_resource_wait(pending, operation_key)
                permit=lease.admit(item, operation_key=operation_key)
            if not bool(permit.get("admitted")):
                # Waiting is durable but is not an experiment and cannot spend an
                # iteration budget.  Planning/critique may continue elsewhere;
                # this exact candidate is retried only after a new lease admits it.
                if not has_wait_checkpoint:
                    row.update(status="waiting_resource",lease=dict(permit)); state["pending"]={"row":row,"candidate":_pending_item(item),"authorization":authorization.to_dict(),"confirmation":bool(pending and pending.get("confirmation")),"parent_authorization":pending.get("parent_authorization") if pending else None,"infrastructure_retry_epoch":retry_epoch, **_preauthored_pending_fields(row)}
                # A post-build wait remains bound to its original admitted
                # lease and durable stage receipt.  A failed re-admission is
                # observational only; preserving the same checkpoint makes
                # any number of fresh supervisor epochs idempotent.
                store.save(state,"waiting_resource"); break
            # The governed GPU adapter owns an operation-key-bound receipt
            # namespace.  It refuses an unkeyed lease, making recovery and
            # result reconciliation refer to the same durable operation.
            if permit.get("operation_key") != operation_key:
                raise DiscoveryControllerError("resource lease did not bind the exact operation key")
            permit={**dict(permit), "repetition":repetition}
            state.pop("pending",None)
            state["inflight"]={"operation_key":operation_key,"row":row,"candidate":_pending_item(item),"authorization":authorization.to_dict(),"lease":dict(permit),"confirmation":bool(pending and pending.get("confirmation")),"parent_authorization":pending.get("parent_authorization") if pending else None,"infrastructure_retry_epoch":retry_epoch, **({"preauthored_continuation":dict(pending["preauthored_continuation"])} if pending and isinstance(pending.get("preauthored_continuation"), Mapping) else {})}
            store.save(state,"pre_screen_intent")
            try: result=screener.screen(item,authorization,permit)
            except ResumableScreenInterruption as exc:
                state["inflight"]["interruption"] = {
                    "type": type(exc).__name__, "message": str(exc),
                    "resumable": True,
                }
                store.save(state, "screen_resumable_interruption")
                break
            except ResourceWait as exc:
                wait_receipt=_validated_resource_wait(exc,operation_key)
                _require_safe_resource_wait_recovery(
                    screener, state["inflight"], wait_receipt)
                pending_wait = _resource_wait_pending(
                    state["inflight"], wait_receipt)
                state.pop("inflight",None)
                state["pending"] = pending_wait
                store.save(state,"waiting_resource")
                break
            except ScreenInfrastructureAmbiguity as exc:
                _record_cumulative_infrastructure_ambiguity(
                    state, config=config, item=item, row=row, exc=exc)
                store.save(state, "screen_infrastructure_ambiguity")
                break
            except PrecomputeScreenRefusal as exc:
                _record_precompute_refusal(state, row, exc)
                store.save(state,"screen_refused"); continue
            except GovernedStageRefusal as exc:
                if item.composition_plan is not None:
                    _bind_cumulative_terminal_row(
                        state, row, _terminalize_cumulative_refusal(
                            config, item, exc))
                _record_governed_stage_refusal(state, row, exc)
                store.save(state, exc.disposition)
                continue
            except Exception as exc:
                # The durable start intent has been written.  An ordinary
                # exception may follow a build, claim, or model invocation, so
                # it is an ambiguous operation until the governed adapter
                # reconciles its operation-key-bound artifacts on restart.
                state["inflight"]["exception"]={"type":type(exc).__name__,"message":str(exc)}
                store.save(state,"screen_ambiguous")
                raise
            state["inflight"]["result"]=asdict(result); store.save(state,"post_screen_result")
            composition_terminal = _finalize_cumulative_screen(
                config, item, result)
            policy=row.get("portfolio_decision_policy")
            result=_classified_result(state,item,result,policy); row.update(
                _screen_iteration_fields(result, repetition=repetition))
            if composition_terminal is not None:
                _bind_cumulative_terminal_row(
                    state, row, composition_terminal)
            # Record the measured disposition before exposing it to the next
            # planner context.  This is the only source of repeat suppression.
            _record_attempt_once(tracker,item,str(item.proposal.get("proposal_id",row["proposal_sha256"])),result)
            state.pop("inflight",None); state.pop("pending",None); state["iterations"].append(row)
            if isinstance(row.get("portfolio_hypothesis_id"), str):
                _record_attempted_candidate_identity(state, row)
            state["scientific_attempts"]=_derived_scientific_attempts(state); _apply_portfolio_outcome(state,row); state["next"]+=1; _schedule_replication(state,item=item,authorization=authorization,row=row,result=result,max_iterations=config.max_iterations); _schedule_cumulative_composition(state,config=config,item=item,result=result); _append_nomination(config.output_root,item,result,_decision_floor(policy,"nomination_floor_pct",config.nomination_threshold)); _write_projection(config.evidence_root or config.output_root); store.save(state,"screened")
    if config.hypothesis_portfolio is not None:
        if (not state.get("complete")
                and state["scientific_attempts"] >= config.max_iterations):
            state["complete"] = True
            state["terminal_reason"] = "scientific_budget_exhausted"
    else:
        state["complete"]=bool(state.get("complete")) or state["next"]>config.max_iterations
    if state["complete"]: state.pop("pending",None)
    store.save(state,"complete" if state["complete"] else "paused"); return state


def build_controller_adapters(*, planner: Planner, critic: Critic, screener: Screener,
                              lease: Lease) -> dict[str, Any]:
    """Bind the four concrete controller seams without accepting shell commands.

    A live factory constructs its governed GPU source screener separately (with
    its proof producer configuration) and supplies that object here.  Keeping
    this top-level boundary object-only prevents a JSON launch manifest from
    becoming an arbitrary command executor.
    """
    parts = {"planner": planner, "critic": critic, "screener": screener, "lease": lease}
    if any(value is None for value in parts.values()):
        raise DiscoveryControllerError("controller adapters must bind every required seam")
    if not callable(getattr(planner, "plan", None)) or not callable(getattr(critic, "review", None)):
        raise DiscoveryControllerError("controller factory did not bind typed planner/critic actors")
    if not callable(getattr(screener, "screen", None)) or not callable(getattr(screener, "reconcile", None)):
        raise DiscoveryControllerError("controller factory did not bind governed screen/reconcile adapter")
    if not callable(getattr(lease, "admit", None)):
        raise DiscoveryControllerError("controller factory did not bind a resource lease adapter")
    return parts


def _load_adapter_config(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).resolve(strict=True)
    if config_path.is_symlink() or not config_path.is_file():
        raise DiscoveryControllerError("adapter config must be a regular file")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiscoveryControllerError("adapter config must be a JSON object") from exc
    if not isinstance(value, Mapping):
        raise DiscoveryControllerError("adapter config must be a JSON object")
    return dict(value)


def _load_factory(reference: str, factory_config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if reference.count(":") != 1:
        raise DiscoveryControllerError("adapter factory must be module:callable")
    module, name = reference.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise DiscoveryControllerError("adapter factory could not be imported") from exc
    if not callable(factory):
        raise DiscoveryControllerError("adapter factory must be callable")
    # Standard factory contract is factory(config: Mapping) -> adapter bundle.
    # A no-argument factory stays useful for narrowly-bound deployment modules.
    try:
        signature = inspect.signature(factory)
        value = factory() if not signature.parameters else factory(dict(factory_config or {}))
    except (TypeError, ValueError) as exc:
        raise DiscoveryControllerError("adapter factory has an unsupported signature") from exc
    if not isinstance(value, Mapping):
        raise DiscoveryControllerError("adapter factory must return mapping")
    try:
        return build_controller_adapters(**dict(value))
    except TypeError as exc:
        raise DiscoveryControllerError("adapter factory must return exactly planner, critic, screener, lease") from exc


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root",required=True); parser.add_argument("--max-iterations",type=int,default=1); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--adapter-factory", required=True); parser.add_argument("--adapter-config"); parser.add_argument("--evidence-root")
    args=parser.parse_args(argv); config=ControllerConfig(Path(args.output_root).resolve(),args.max_iterations,dry_run=args.dry_run,evidence_root=Path(args.evidence_root).resolve() if args.evidence_root else None)
    parts=_load_factory(args.adapter_factory, _load_adapter_config(args.adapter_config)); run_controller(config,planner=parts["planner"],critic=parts["critic"],screener=parts["screener"],lease=parts["lease"]); return 0

if __name__=="__main__": main()
