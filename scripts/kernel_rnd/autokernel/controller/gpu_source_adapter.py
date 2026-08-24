#!/usr/bin/env python3
"""Crash-conservative controller adapter for governed GPU source evidence.

This module is deliberately separate from the controller state machine.  It
turns the evidence producer into the ``GpuSourceScreener`` proof callback and
owns one durable operation directory per controller operation key.  An absent
directory is safe to start, a recursively validated result is recoverable, and
every partial or contradictory state is ambiguous.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .. import schemas
from ..resource import device_claim
from . import discovery_controller as controller
from .. import cumulative_composition
from . import gpu_source_evidence as evidence
from . import gpu_source_proofs
from scripts.benchmark import autokernel_progression

OPERATION_SCHEMA = "epyc.autokernel.gpu_source_operation.v2"
RESULT_SCHEMA = "epyc.autokernel.gpu_source_operation_result.v2"
RESOURCE_WAIT_SCHEMA = "epyc.autokernel.gpu_source_resource_wait.v1"
POSTBUILD_CHECKPOINT_SCHEMA = "epyc.autokernel.gpu_source_postbuild_checkpoint.v1"
RESERVATION_RELEASE_SCHEMA = "epyc.autokernel.gpu_source_reservation_release.v1"
RUNNER_PLAN_SCHEMA = "epyc.autokernel.gpu_source_runner_plan.v2"
AUTHORITY = evidence.AUTHORITY
SHA = re.compile(r"^[0-9a-f]{64}$")


class GpuSourceAdapterError(RuntimeError):
    """The adapter refused an unsafe start or an unprovable recovery."""


@dataclass(frozen=True)
class CompatibleRecovery:
    """Compatibility type used only by controller revisions predating Recovery."""

    status: str
    result: Any | None = None
    wait_receipt: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {
                "safe_to_start", "resource_wait", "sealed_result", "ambiguous"}:
            raise GpuSourceAdapterError("unknown recovery status")
        if (self.status == "sealed_result") != (self.result is not None):
            raise GpuSourceAdapterError("recovery result binding is invalid")
        if (self.status == "resource_wait") != isinstance(
                self.wait_receipt, Mapping):
            raise GpuSourceAdapterError("recovery wait binding is invalid")
        if self.wait_receipt is not None:
            object.__setattr__(self, "wait_receipt", dict(self.wait_receipt))


def _recovery(status: str, result: Any | None = None,
              wait_receipt: Mapping[str, Any] | None = None) -> Any:
    recovery_type = getattr(controller, "Recovery", CompatibleRecovery)
    return recovery_type(
        status=status, result=result, wait_receipt=wait_receipt)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()  # type: ignore[union-attr]
    if not isinstance(value, Mapping):
        raise GpuSourceAdapterError(f"{label} must be a mapping")
    return dict(value)


def _operation_key(value: object) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise GpuSourceAdapterError("operation_key must be an exact SHA-256 digest")
    return value


def _candidate_manifest(value: object) -> str:
    if hasattr(value, "source_manifest_sha256"):
        digest = getattr(value, "source_manifest_sha256")
    elif isinstance(value, Mapping):
        nested = value.get("candidate")
        digest = (nested.get("source_manifest_sha256")
                  if isinstance(nested, Mapping)
                  else value.get("source_manifest_sha256"))
    else:
        digest = None
    if not isinstance(digest, str) or not SHA.fullmatch(digest):
        raise GpuSourceAdapterError("candidate lacks a sealed source manifest identity")
    return digest


def _candidate_composition_plan(
        value: object) -> cumulative_composition.CompositionPlan | None:
    plan = getattr(value, "composition_plan", None)
    if isinstance(value, Mapping):
        nested = value.get("candidate")
        carrier = nested if isinstance(nested, Mapping) else value
        plan = carrier.get("composition_plan")
    if plan is None:
        return None
    try:
        return (plan if isinstance(
            plan, cumulative_composition.CompositionPlan)
            else cumulative_composition.CompositionPlan.from_dict(plan))
    except (TypeError, cumulative_composition.CompositionError) as exc:
        raise GpuSourceAdapterError(
            "candidate cumulative plan is invalid") from exc


def _candidate_composition_plan_sha256(value: object) -> str | None:
    plan = _candidate_composition_plan(value)
    return None if plan is None else plan.plan_sha256


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GpuSourceAdapterError(f"{label} must be a regular non-symlink file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuSourceAdapterError(f"{label} is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise GpuSourceAdapterError(f"{label} must be a JSON object")
    native = raw.get("receipt_sha256")
    if (not isinstance(native, str) or not SHA.fullmatch(native)
            or schemas.content_hash({k: v for k, v in raw.items()
                                     if k != "receipt_sha256"}) != native):
        raise GpuSourceAdapterError(f"{label} self-hash mismatch")
    return raw


def _load_screen_receipt(screen: controller.SealedScreen) -> Mapping[str, Any]:
    path = Path(screen.receipt_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise GpuSourceAdapterError("screen receipt is not an absolute regular file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GpuSourceAdapterError("screen receipt is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise GpuSourceAdapterError("screen receipt must be an object")
    if raw.get("result_sha256") != screen.result_sha256:
        raise GpuSourceAdapterError("screen receipt/result identity mismatch")
    expected = schemas.content_hash({k: v for k, v in raw.items()
                                     if k != "result_sha256"})
    if expected != screen.result_sha256:
        raise GpuSourceAdapterError("screen result native self-hash mismatch")
    if (raw.get("non_promotable") is not True
            or raw.get("promotion_claim") is not False):
        raise GpuSourceAdapterError("screen receipt crossed discovery authority")
    return raw


def _typed_screen(value: object) -> controller.SealedScreen:
    if not isinstance(value, Mapping):
        raise GpuSourceAdapterError("sealed screen payload is malformed")
    try:
        result = controller._sealed_screen_from_dict(value)
    except (TypeError, ValueError, controller.DiscoveryControllerError) as exc:
        raise GpuSourceAdapterError("sealed screen payload is invalid") from exc
    _load_screen_receipt(result)
    return result


def _bind_composition_screen(
        result: controller.SealedScreen,
        plan: cumulative_composition.CompositionPlan,
) -> None:
    pair = result.composition_build_pair
    correctness = result.composition_correctness
    comparison = result.composition_comparison
    performance = result.cumulative_performance
    reference = result.cumulative_performance_ref
    if (pair is None or correctness is None or comparison is None
            or performance is None or reference is None):
        raise GpuSourceAdapterError("cumulative result wrapper is partial")
    try:
        pair.bind_plan(plan)
        correctness.bind_pair(pair)
        performance.bind(plan, pair, correctness, comparison)
        reopened, file_sha = cumulative_composition.load_cumulative_performance(
            Path(reference.path), expected_file_sha256=reference.sha256)
    except cumulative_composition.CompositionError as exc:
        raise GpuSourceAdapterError(
            "cumulative result wrapper changed typed evidence") from exc
    if reopened != performance or file_sha != reference.sha256:
        raise GpuSourceAdapterError(
            "cumulative performance reference changed typed evidence")
    if (comparison.operation_key != plan.operation_key
            or comparison.build_pair_sha256 != pair.pair_sha256
            or comparison.correctness_result_sha256 !=
               correctness.result_sha256):
        raise GpuSourceAdapterError(
            "cumulative result wrapper changed typed evidence")
    try:
        cumulative_composition.commit_result_authority(
            Path(reference.path).resolve().parent)
    except cumulative_composition.CompositionError as exc:
        raise GpuSourceAdapterError(
            "cumulative result authority journal refused") from exc


def _intent_body(*, operation_key: str, candidate: object,
                 authorization: object, lease: Mapping[str, Any]) -> dict[str, Any]:
    authorization_raw = _mapping(authorization, "claim authorization")
    lease_raw = dict(lease)
    if lease_raw.get("operation_key") != operation_key:
        raise GpuSourceAdapterError("lease does not bind the controller operation key")
    return {
        "schema": OPERATION_SCHEMA,
        "authority": AUTHORITY,
        "promotion_claim": False,
        "operation_key": operation_key,
        "manifest_sha256": _candidate_manifest(candidate),
        "composition_plan_sha256":
            _candidate_composition_plan_sha256(candidate),
        "authorization_sha256": schemas.content_hash(authorization_raw),
        "lease_sha256": schemas.content_hash(_lease_identity(lease_raw)),
    }


def _lease_identity(lease: Mapping[str, Any]) -> dict[str, Any]:
    """Stable policy authority; volatile probe receipts are resume evidence."""
    keys = ("operation_key", "repetition", "mode", "device_id",
            "inference_window_lock", "model_sha256", "load_admission",
            "promotion_claim")
    return {key: lease.get(key) for key in keys}


def _inflight_identity(inflight: Mapping[str, Any]) -> dict[str, Any]:
    operation_key = _operation_key(inflight.get("operation_key"))
    candidate = inflight.get("candidate")
    authorization = _mapping(inflight.get("authorization"), "inflight authorization")
    lease = _mapping(inflight.get("lease"), "inflight lease")
    if lease.get("operation_key") != operation_key:
        raise GpuSourceAdapterError("inflight lease does not bind operation_key")
    return {
        "operation_key": operation_key,
        "manifest_sha256": _candidate_manifest(candidate),
        "composition_plan_sha256":
            _candidate_composition_plan_sha256(candidate),
        "authorization_sha256": schemas.content_hash(authorization),
        "lease_sha256": schemas.content_hash(_lease_identity(lease)),
    }


def _safe_wait_receipts(root: Path, identity: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    waits = root / "resource-waits"
    if not waits.exists() and not waits.is_symlink():
        return ()
    if waits.is_symlink() or not waits.is_dir():
        raise GpuSourceAdapterError("resource-wait history is not a real directory")
    wait_stat = waits.stat()
    if wait_stat.st_uid != os.geteuid() or wait_stat.st_mode & 0o077:
        raise GpuSourceAdapterError("resource-wait history is not private to the controller owner")
    rows = []
    for path in sorted(waits.iterdir()):
        if path.is_symlink() or not path.is_file() or not re.fullmatch(r"wait-[0-9]{4}\.json", path.name):
            raise GpuSourceAdapterError("resource-wait history contains an unsafe entry")
        facts = path.stat(follow_symlinks=False)
        if (facts.st_uid != os.geteuid() or facts.st_nlink != 1
                or stat.S_IMODE(facts.st_mode) & 0o022):
            raise GpuSourceAdapterError(
                "resource-wait receipt has unsafe file authority")
        row = _read_json(path, "resource-wait receipt")
        required = {
            "schema": RESOURCE_WAIT_SCHEMA,
            "authority": AUTHORITY,
            "promotion_claim": False,
            "operation_key": identity["operation_key"],
            "manifest_sha256": identity["manifest_sha256"],
            "gpu_executor_started": False,
            "proof_root_created": False,
            "runner_plan_created": False,
            "runner_output_created": False,
        }
        if any(row.get(key) != value for key, value in required.items()):
            raise GpuSourceAdapterError("resource-wait receipt does not prove a pre-executor stop")
        if set(row) != {
                *required, "build_key", "materialization_sha256",
                "contention", "receipt_sha256"}:
            raise GpuSourceAdapterError(
                "resource-wait receipt has an inexact schema")
        contention = row.get("contention")
        if not isinstance(contention, Mapping):
            raise GpuSourceAdapterError("resource-wait receipt lacks typed contention")
        reason = contention.get("reason")
        common = {
            "admitted", "phase", "reason", "device_id", "operation_key",
            "promotion_claim",
        }
        allowed = (common | {"foreign_kfd_pids"}
                   if isinstance(reason, str) and reason.startswith("foreign_kfd_")
                   else common | {"detail"})
        if (set(contention) not in ({*common}, allowed)
                or contention.get("admitted") is not False
                or contention.get("phase") != "pre_executor_reservation"
                or contention.get("operation_key") != identity["operation_key"]
                or contention.get("promotion_claim") is not False
                or not isinstance(contention.get("device_id"), str)
                or not contention["device_id"]
                or reason not in {
                    "device_busy", "foreign_kfd_busy",
                    "foreign_kfd_inventory_invalid",
                    "foreign_kfd_inventory_unreadable",
                }
                or any("claim" in str(key).lower()
                       for key in contention if key != "promotion_claim")):
            raise GpuSourceAdapterError(
                "resource-wait contention is not an exact claim-free boundary")
        if reason.startswith("foreign_kfd_"):
            pids = contention.get("foreign_kfd_pids")
            if (not isinstance(pids, list)
                    or pids != sorted(set(pids))
                    or any(isinstance(pid, bool) or not isinstance(pid, int)
                           or pid <= 0 for pid in pids)
                    or reason == "foreign_kfd_busy" and not pids
                    or reason != "foreign_kfd_busy" and pids):
                raise GpuSourceAdapterError(
                    "resource-wait KFD inventory is malformed")
        elif ("detail" in contention
              and not isinstance(contention.get("detail"), str)):
            raise GpuSourceAdapterError(
                "resource-wait device contention detail is malformed")
        build_key = row.get("build_key")
        materialization = row.get("materialization_sha256")
        if ((build_key is None) != (materialization is None)
                or build_key is not None
                and (not isinstance(build_key, str) or not SHA.fullmatch(build_key)
                     or not isinstance(materialization, str)
                     or not SHA.fullmatch(materialization))):
            raise GpuSourceAdapterError(
                "resource-wait build identity is incomplete")
        rows.append(row)
    return tuple(rows)


_BUILD_PATH_FIELDS = (
    "anchor_build", "candidate_build", "measurement_binary",
    "common_loader_dir", "anchor_loader_dir", "candidate_loader_dir",
    "materialization_receipt", "anchor_source_tree_receipt",
    "candidate_source_tree_receipt", "anchor_correctness_binary",
    "candidate_correctness_binary", "anchor_correctness_capability_receipt",
    "candidate_correctness_capability_receipt", "teardown_receipt",
)
_BUILD_SCALAR_FIELDS = (
    "reward_runtime_sha256", "operation_key", "build_key",
    "materialization_sha256", "anchor_source_tree_sha256",
    "candidate_source_tree_sha256", "anchor_correctness_binary_sha256",
    "candidate_correctness_binary_sha256",
    "anchor_correctness_capability_sha256",
    "candidate_correctness_capability_sha256", "teardown_sha256",
)


def _build_projection(build: controller.GpuSourceBuild) -> dict[str, Any]:
    if not isinstance(build, controller.GpuSourceBuild):
        raise GpuSourceAdapterError("postbuild checkpoint requires a typed build")
    return {
        "candidate_identity": asdict(build.candidate_identity),
        "anchor_identity": asdict(build.anchor_identity),
        **{name: (None if getattr(build, name) is None
                  else str(getattr(build, name)))
           for name in _BUILD_PATH_FIELDS},
        **{name: getattr(build, name) for name in _BUILD_SCALAR_FIELDS},
    }


def _build_from_projection(value: Mapping[str, Any]) -> controller.GpuSourceBuild:
    required = {"candidate_identity", "anchor_identity",
                *_BUILD_PATH_FIELDS, *_BUILD_SCALAR_FIELDS}
    if not isinstance(value, Mapping) or set(value) != required:
        raise GpuSourceAdapterError("postbuild projection has an inexact schema")
    try:
        return controller.GpuSourceBuild(
            candidate_identity=gpu_source_proofs.BuildIdentity(
                **value["candidate_identity"]),
            anchor_identity=gpu_source_proofs.BuildIdentity(
                **value["anchor_identity"]),
            **{name: (None if value[name] is None else Path(value[name]))
               for name in _BUILD_PATH_FIELDS},
            **{name: value[name] for name in _BUILD_SCALAR_FIELDS},
        )
    except (TypeError, ValueError, controller.DiscoveryControllerError,
            gpu_source_proofs.ProofError) as exc:
        raise GpuSourceAdapterError(
            "postbuild projection does not reconstruct a typed build") from exc


def _postbuild_body(*, build: controller.GpuSourceBuild,
                    identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": POSTBUILD_CHECKPOINT_SCHEMA,
        "authority": AUTHORITY,
        "promotion_claim": False,
        "operation_key": identity["operation_key"],
        "manifest_sha256": identity["manifest_sha256"],
        "build": _build_projection(build),
    }


def _validate_postbuild_checkpoint(
        path: Path, identity: Mapping[str, Any], *,
        expected_build: controller.GpuSourceBuild | None = None,
) -> controller.GpuSourceBuild:
    facts = path.stat(follow_symlinks=False)
    if (path.is_symlink() or not path.is_file()
            or facts.st_uid != os.geteuid() or facts.st_nlink != 1
            or stat.S_IMODE(facts.st_mode) & 0o022):
        raise GpuSourceAdapterError(
            "postbuild checkpoint has unsafe file authority")
    row = _read_json(path, "postbuild checkpoint")
    required = {
        "schema", "authority", "promotion_claim", "operation_key",
        "manifest_sha256", "build", "receipt_sha256",
    }
    if (set(row) != required
            or row.get("schema") != POSTBUILD_CHECKPOINT_SCHEMA
            or row.get("authority") != AUTHORITY
            or row.get("promotion_claim") is not False
            or row.get("operation_key") != identity["operation_key"]
            or row.get("manifest_sha256") != identity["manifest_sha256"]):
        raise GpuSourceAdapterError("postbuild checkpoint identity changed")
    build = _build_from_projection(row["build"])
    if build.operation_key not in {None, identity["operation_key"]}:
        raise GpuSourceAdapterError(
            "postbuild checkpoint names another controller operation")
    for name in ("anchor_build", "candidate_build", "common_loader_dir",
                 "anchor_loader_dir", "candidate_loader_dir"):
        current = getattr(build, name)
        if current is not None and current.is_symlink():
            raise GpuSourceAdapterError(
                "postbuild checkpoint contains a symlinked build directory")
    if (expected_build is not None
            and _build_projection(build) != _build_projection(expected_build)):
        raise GpuSourceAdapterError(
            "reopened builder result differs from its postbuild checkpoint")
    return build


def _validate_evidence_policy(path: Path,
                              identity: Mapping[str, Any]) -> None:
    binding = _regular_binding(path, "evidence-policy.json")
    facts = path.stat(follow_symlinks=False)
    if facts.st_uid != os.geteuid() or stat.S_IMODE(facts.st_mode) & 0o022:
        raise GpuSourceAdapterError(
            "evidence policy has unsafe write authority")
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
            raise GpuSourceAdapterError("evidence policy changed while read")
        policy = json.loads(
            raw.decode("utf-8", "strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GpuSourceAdapterError("evidence policy is not strict JSON") from exc
    if (not isinstance(policy, Mapping)
            or policy.get("schema") != evidence.EXECUTION_POLICY_SCHEMA
            or policy.get("manifest_sha256") != identity["manifest_sha256"]):
        raise GpuSourceAdapterError("evidence policy identity changed")
    try:
        candidate = gpu_source_proofs.BuildIdentity(
            **policy["candidate_build_identity"])
        anchor = gpu_source_proofs.BuildIdentity(
            **policy["anchor_build_identity"])
    except (KeyError, TypeError, ValueError, gpu_source_proofs.ProofError) as exc:
        raise GpuSourceAdapterError(
            "evidence policy build identities are malformed") from exc
    if candidate == anchor:
        raise GpuSourceAdapterError("evidence policy reused one build identity")
    references: list[evidence.BoundInputFile] = []
    try:
        for key in ("correctness_inputs", "candidate_rocprof_inputs",
                    "anchor_rocprof_inputs"):
            rows = policy[key]
            if not isinstance(rows, list) or not rows:
                raise TypeError(f"{key} is not a non-empty list")
            references.extend(evidence._bound_from_dict(row) for row in rows)
        shared = policy.get("shared_runtime")
        if shared is not None:
            if not isinstance(shared, Mapping) or set(shared) != {
                    "measurement_binary", "runtime_receipt",
                    "anchor_hip_library", "candidate_hip_library"}:
                raise TypeError("shared runtime is incomplete")
            references.extend(
                evidence._bound_from_dict(shared[key]) for key in sorted(shared))
        for bound in references:
            evidence._verify_bound(bound)
    except (KeyError, TypeError, ValueError,
            evidence.EvidenceProducerError) as exc:
        raise GpuSourceAdapterError(
            "evidence policy bound artifact closure changed") from exc


def _postbuild_wait_root(root: Path, identity: Mapping[str, Any],
                         waits: Sequence[Mapping[str, Any]]) -> bool:
    checkpoint = root / "postbuild-checkpoint.json"
    policy = root / "evidence-policy.json"
    if not waits:
        return False
    if not checkpoint.exists() and not checkpoint.is_symlink():
        return False
    if policy.exists() or policy.is_symlink():
        _validate_evidence_policy(policy, identity)
        if any(row.get("build_key") is None
               or row.get("materialization_sha256") is None for row in waits):
            return False
    build = _validate_postbuild_checkpoint(checkpoint, identity)
    if (build.build_key is not None
            and not policy.exists() and not policy.is_symlink()):
        return False
    return all(
        row.get("build_key") == build.build_key
        and row.get("materialization_sha256") == build.materialization_sha256
        for row in waits)


def _is_resumable_wait_root(root: Path, identity: Mapping[str, Any]) -> bool:
    allowed = {
        "intent.json", "source-manifest.json", "evidence-policy.json",
        "postbuild-checkpoint.json", "resource-waits",
    }
    if any(path.name not in allowed for path in root.iterdir()):
        return False
    manifest = root / "source-manifest.json"
    if manifest.exists() or manifest.is_symlink():
        try:
            binding = _regular_binding(manifest, "source-manifest.json")
        except GpuSourceAdapterError:
            return False
        if binding["sha256"] != identity.get("manifest_sha256"):
            return False
    waits = _safe_wait_receipts(root, identity)
    if waits and not _postbuild_wait_root(root, identity, waits):
        return False
    # Intent plus the canonical prebuild manifest means the process stopped no
    # later than reservation.  With no proof, policy, or runner carrier,
    # re-entering the sealed builder/cache seam cannot repeat a GPU command.
    return True


def _validated_stage_receipt(path: Path, schema: str,
                             identity: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate enough of a terminal stage to prove retry cannot repeat it.

    Full plan-dependent validation happens inside ``produce_gpu_source_evidence``
    after the sealed build and plan have been reconstructed.  Reconcile only
    decides whether it is safe to re-enter that producer.  A self-hashed,
    operation-bound terminal receipt is safe: the producer will either reuse it
    after recursive validation or refuse before invoking the next executor.
    """
    loaded = gpu_source_proofs.load_receipt(path, schema=schema)
    body = loaded["body"]
    if (body.get("authority") != AUTHORITY
            or body.get("promotion_claim") is not False
            or body.get("manifest_sha256") != identity["manifest_sha256"]):
        raise GpuSourceAdapterError("partial proof receipt identity mismatch")
    return loaded


def _validated_runner_plan(
        path: Path, identity: Mapping[str, Any], *,
        composition_plan: cumulative_composition.CompositionPlan | None = None,
) -> tuple[Mapping[str, Any],
           cumulative_composition.CumulativeBuildPair | None,
           cumulative_composition.FullCorrectness | None,
           cumulative_composition.FrozenProductionAuthority | None]:
    """Reopen the pre-run carrier that makes completed output recoverable.

    The pair and full-correctness authority are sealed before either runner
    process starts.  A controller crash after the second result therefore has
    enough immutable evidence to reconstruct the exact typed composition
    screen without rebuilding or spending another scientific attempt.
    """
    body = gpu_source_proofs.load_receipt(
        path, schema=RUNNER_PLAN_SCHEMA)["body"]
    base = {
        "schema", "authority", "promotion_claim", "operation_key",
        "composition_plan_sha256", "composition_build_pair",
        "composition_correctness", "composition_production_authority",
        "receipt_sha256",
    }
    single = {"output_dir"}
    dual = {
        "measurement_graphs_off_output_dir",
        "target_runtime_graphs_on_output_dir",
    }
    cumulative = dual | {
        "production_graphs_on_output_dir",
        "cumulative_performance_path",
        "composition_exact_route_receipt_sha256",
        "composition_expected_route_set_sha256",
        "composition_target_runtime_frame_sha256",
    }
    keys = set(body)
    if (keys not in (base | single, base | dual, base | cumulative)
            or body.get("authority") != AUTHORITY
            or body.get("promotion_claim") is not False
            or body.get("operation_key") != identity["operation_key"]
            or body.get("composition_plan_sha256") !=
               identity.get("composition_plan_sha256")):
        raise GpuSourceAdapterError("GPU runner plan identity mismatch")
    expected_plan_sha = identity.get("composition_plan_sha256")
    pair_raw = body.get("composition_build_pair")
    correctness_raw = body.get("composition_correctness")
    production_raw = body.get("composition_production_authority")
    if expected_plan_sha is None:
        if (pair_raw is not None or correctness_raw is not None
                or production_raw is not None):
            raise GpuSourceAdapterError(
                "ordinary runner acquired cumulative recovery authority")
        return body, None, None, None
    try:
        pair = cumulative_composition.CumulativeBuildPair.from_dict(pair_raw)
        correctness = cumulative_composition.FullCorrectness.from_dict(
            correctness_raw)
        production = cumulative_composition.FrozenProductionAuthority.from_dict(
            production_raw)
        correctness.bind_pair(pair)
        if pair.plan_sha256 != expected_plan_sha or not correctness.passed:
            raise cumulative_composition.CompositionError(
                "runner recovery evidence differs from the composition plan")
        if composition_plan is not None:
            if composition_plan.plan_sha256 != expected_plan_sha:
                raise cumulative_composition.CompositionError(
                    "inflight composition plan identity changed")
            pair.bind_plan(composition_plan)
            production.bind_plan(composition_plan)
    except (TypeError, cumulative_composition.CompositionError) as exc:
        raise GpuSourceAdapterError(
            "GPU runner cumulative recovery evidence is invalid") from exc
    if keys != base | cumulative:
        raise GpuSourceAdapterError(
            "cumulative runner plan lacks frozen-production outputs")
    return body, pair, correctness, production


def _is_resumable_stage_root(
        root: Path, identity: Mapping[str, Any],
        lease: Mapping[str, Any]) -> bool:
    """Return true only for an ordered proof journal with terminal boundaries.

    Raw output without a receipt is deliberately not resumable: the adapter
    cannot know whether the corresponding GPU command completed.  Completed
    receipts, however, are exactly-once checkpoints and must survive controller
    or API reloads.
    """
    allowed_root = {
        "intent.json", "source-manifest.json", "evidence-policy.json",
        "postbuild-checkpoint.json", "resource-waits", "proof",
        "runner-plan.json", "runner",
        "reservation-release.json", "reservation-releases",
    }
    if any(path.name not in allowed_root for path in root.iterdir()):
        return False
    checkpoint = root / "postbuild-checkpoint.json"
    if checkpoint.exists() or checkpoint.is_symlink():
        _validate_postbuild_checkpoint(checkpoint, identity)
    if schemas.content_hash(_lease_identity(lease)) != identity["lease_sha256"]:
        return False
    proof = root / "proof"
    if proof.is_symlink() or not proof.is_dir():
        return False
    allowed_proof = {
        "correctness", "attribution-candidate", "attribution-anchor",
        "attribution-pair.json", "attribution-pair-refusal.json",
        "proof-bundle.json",
    }
    if any(path.name not in allowed_proof for path in proof.iterdir()):
        return False
    try:
        correctness = proof / "correctness"
        if correctness.is_symlink() or not correctness.is_dir():
            return False
        aggregate = correctness / "receipt.json"
        refusal = correctness / "refusal.json"
        if aggregate.exists() or aggregate.is_symlink():
            if refusal.exists() or refusal.is_symlink():
                return False
            _validated_stage_receipt(
                aggregate, evidence.CORRECTNESS_SCHEMA, identity)
        elif refusal.exists() or refusal.is_symlink():
            _validated_stage_receipt(
                refusal, evidence.CORRECTNESS_REFUSAL_SCHEMA, identity)
        else:
            # Multi-invocation correctness may crash after one sub-receipt but
            # before the aggregate.  Every extant child must be terminal.
            children = tuple(correctness.iterdir())
            if not children:
                return False
            for child in children:
                child_receipt = child / "receipt.json"
                child_refusal = child / "refusal.json"
                if child.is_symlink() or not child.is_dir() \
                        or (child_receipt.exists() == child_refusal.exists()) \
                        or (child_receipt.is_symlink() or child_refusal.is_symlink()):
                    return False
                _validated_stage_receipt(
                    child_receipt if child_receipt.exists() else child_refusal,
                    (evidence.CORRECTNESS_SCHEMA if child_receipt.exists()
                     else evidence.CORRECTNESS_REFUSAL_SCHEMA), identity)

        completed_arms = 0
        for arm in ("candidate", "anchor"):
            arm_dir = proof / f"attribution-{arm}"
            if not arm_dir.exists() and not arm_dir.is_symlink():
                continue
            if arm_dir.is_symlink() or not arm_dir.is_dir():
                return False
            receipt = arm_dir / "receipt.json"
            refusal = arm_dir / "refusal.json"
            if receipt.exists() == refusal.exists():
                return False
            _validated_stage_receipt(
                receipt if receipt.exists() else refusal,
                (evidence.ATTRIBUTION_SCHEMA if receipt.exists()
                 else evidence.ATTRIBUTION_REFUSAL_SCHEMA), identity)
            completed_arms += 1

        pair = proof / "attribution-pair.json"
        pair_refusal = proof / "attribution-pair-refusal.json"
        if pair.exists() and pair_refusal.exists():
            return False
        if pair.exists() or pair.is_symlink():
            if completed_arms != 2:
                return False
            _validated_stage_receipt(pair, evidence.PAIR_SCHEMA, identity)
        elif pair_refusal.exists() or pair_refusal.is_symlink():
            if completed_arms != 2:
                return False
            _validated_stage_receipt(
                pair_refusal, evidence.PAIR_REFUSAL_SCHEMA, identity)
        bundle = proof / "proof-bundle.json"
        loaded_bundle = None
        if bundle.exists() or bundle.is_symlink():
            if not pair.exists():
                return False
            loaded_bundle = evidence.load_gpu_source_evidence_bundle(bundle)

        runner_plan = root / "runner-plan.json"
        runner_root = root / "runner"
        partial_claims: list[device_claim.ClaimReceipt] = []
        if (runner_root.exists() or runner_root.is_symlink()) \
                and not (runner_plan.exists() or runner_plan.is_symlink()):
            # The sealed deployment writes its admission carrier immediately
            # before parsing the two runner namespaces.  A parser/process stop
            # may therefore leave this exact pre-plan tree after proof.  It is
            # safe to retry because args_factory revalidates these bytes before
            # it can seal a runner plan or start a process.
            if (not bundle.is_file() or bundle.is_symlink()
                    or runner_root.is_symlink() or not runner_root.is_dir()):
                return False
            repetition = lease.get("repetition")
            decision = lease.get("load_admission")
            if repetition not in {1, 2} or not isinstance(decision, Mapping):
                return False
            entries = tuple(runner_root.iterdir())
            if len(entries) != 1 or entries[0].is_symlink() \
                    or not entries[0].is_dir() \
                    or entries[0].name != f"s{repetition}":
                return False
            carriers = tuple(entries[0].iterdir())
            if (len(carriers) != 1
                    or carriers[0].name != "load-admission-decision.json"
                    or carriers[0].is_symlink() or not carriers[0].is_file()):
                return False
            try:
                binding = _regular_binding(
                    carriers[0],
                    f"runner/s{repetition}/load-admission-decision.json")
                expected = (json.dumps(
                    dict(decision), sort_keys=True, indent=2) + "\n").encode()
            except (GpuSourceAdapterError, OSError, TypeError, ValueError):
                return False
            if (binding["size"] != len(expected)
                    or binding["sha256"] != hashlib.sha256(expected).hexdigest()):
                return False
            releases = _reservation_release_epochs(
                root, identity["operation_key"])
            if not releases or loaded_bundle is None:
                return False
            expected_device = lease.get("device_id")
            if (not isinstance(expected_device, str)
                    or any(body.get("device_claim_released", {}).get(
                        "device_id") != expected_device
                        for body in releases.values())):
                return False
            correctness_body = gpu_source_proofs.load_receipt(
                Path(str(loaded_bundle.correctness["path"])),
                schema=evidence.CORRECTNESS_SCHEMA)["body"]
            pair_body = gpu_source_proofs.load_receipt(
                Path(str(loaded_bundle.attribution["path"])),
                schema=evidence.PAIR_SCHEMA)["body"]
            proof_bodies = [correctness_body]
            for arm in ("candidate", "anchor"):
                reference = pair_body.get(arm)
                if (not isinstance(reference, Mapping)
                        or not isinstance(reference.get("body"), Mapping)):
                    return False
                proof_bodies.append(reference["body"])
            opened_claims = []
            for body in proof_bodies:
                try:
                    opened = device_claim.ClaimReceipt.from_dict(
                        body.get("device_claim_open"))
                except (TypeError, ValueError):
                    return False
                phase_end = body.get("device_claim_borrowed_phase_end")
                if (opened.released_at is not None
                        or body.get("device_claim_mode") !=
                        "borrowed_outer_reservation"
                        or not isinstance(phase_end, Mapping)
                        or phase_end.get("outer_claim_id") != opened.claim_id
                        or phase_end.get("physical_release") is not False):
                    return False
                opened_claims.append(opened)
            canonical_open = opened_claims[0].to_dict()
            if any(opened.to_dict() != canonical_open
                   for opened in opened_claims[1:]):
                return False
            for body in releases.values():
                try:
                    epoch = device_claim.ClaimReceipt.from_dict(
                        body.get("device_claim_released"))
                except (TypeError, ValueError):
                    return False
                if (epoch.device_id != opened_claims[0].device_id
                        or epoch.campaign_id != opened_claims[0].campaign_id):
                    return False
            release_body = releases.get(opened_claims[0].claim_id)
            if not isinstance(release_body, Mapping):
                return False
            try:
                released = device_claim.ClaimReceipt.from_dict(
                    release_body.get("device_claim_released"))
            except (TypeError, ValueError):
                return False
            # The release must be the exact physical terminal of the claim
            # sealed into every completed proof stage.  This binds campaign,
            # device, holder PID/start ticks, boot ID, lock, acquisition and
            # expiry—not merely a mutable probe campaign or a claim-id string.
            if (released.released_at is None
                    or replace(released, released_at=None).to_dict()
                    != canonical_open):
                return False
            return True
        if (runner_plan.exists() or runner_plan.is_symlink()
                or runner_root.exists() or runner_root.is_symlink()):
            if not bundle.is_file() or bundle.is_symlink():
                return False
            plan, _composition_pair, _composition_correctness, \
                _composition_production = \
                _validated_runner_plan(runner_plan, identity)
            output_keys = [
                "measurement_graphs_off_output_dir",
                "target_runtime_graphs_on_output_dir",
            ]
            if _composition_pair is not None:
                output_keys.append("production_graphs_on_output_dir")
            raw_outputs = [plan.get(key) for key in output_keys]
            if not all(isinstance(value, str) for value in raw_outputs):
                return False
            runner_resolved = runner_root.resolve()
            outputs = tuple(Path(str(value)).resolve()
                            for value in raw_outputs)
            if len(set(outputs)) != len(outputs) or any(
                    not output.is_relative_to(runner_resolved)
                    for output in outputs):
                return False
            if runner_root.exists():
                if runner_root.is_symlink() or not runner_root.is_dir():
                    return False
                for entry in runner_root.rglob("*"):
                    if entry.is_symlink():
                        return False
                    if entry.is_file() and not any(
                            entry.resolve().is_relative_to(output)
                            for output in outputs):
                        return False
            stage_status: list[str] = []
            for output, graph_mode in zip(outputs, ("off", "on")):
                if not output.exists() and not output.is_symlink():
                    stage_status.append("absent")
                    continue
                if output.is_symlink() or not output.is_dir():
                    return False
                result = output / "result.json"
                if not result.exists() or result.is_symlink():
                    if controller.gpu_discovery.validate_resumable_output(
                            output, graph_mode=graph_mode):
                        preflight_bytes, _ = (
                            controller.gpu_discovery._capture_file(
                                output / "preflight.json",
                                "adapter resumable preflight"))
                        order = json.loads(
                            preflight_bytes)["arm_order_schedule"]
                        output_claims: list[device_claim.ClaimReceipt] = []
                        for arm in order:
                            process = output / f"process-{arm}/receipt.json"
                            if not process.exists():
                                break
                            process_bytes, _ = (
                                controller.gpu_discovery._capture_file(
                                    process,
                                    "adapter resumable process receipt"))
                            provisional = json.loads(process_bytes)
                            process_body = (
                                controller.gpu_discovery._load_process_capture(
                                    process.parent,
                                    identity=provisional.get("identity"))[
                                        "receipt"])
                            resource = process_body.get("resource_context")
                            if not isinstance(resource, Mapping):
                                return False
                            try:
                                opened = device_claim.ClaimReceipt.from_dict(
                                    resource.get("device_claim_open"))
                            except (TypeError, ValueError):
                                return False
                            partial_claims.append(opened)
                            output_claims.append(opened)
                        governance_bytes, _ = (
                            controller.gpu_discovery._capture_file(
                                output / "live-governance.json",
                                "adapter live governance"))
                        governance = json.loads(governance_bytes)
                        if (not output_claims
                                or governance.get("device_claim_open") !=
                                output_claims[-1].to_dict()
                                or governance.get("status") not in {
                                    "borrowed_phase_ended", "released"}):
                            return False
                        if governance["status"] == "borrowed_phase_ended":
                            phase = governance.get(
                                "device_claim_borrowed_phase_end")
                            if (not isinstance(phase, Mapping)
                                    or phase.get("outer_claim_id") !=
                                    output_claims[-1].claim_id
                                    or phase.get("physical_release") is not False):
                                return False
                        elif not isinstance(
                                governance.get("device_claim_released"), Mapping):
                            return False
                        stage_status.append("partial")
                        continue
                    return False
                body = gpu_source_proofs.load_receipt(
                    result,
                    schema="epyc.autokernel.gpu_candidate_only_screen.v2")["body"]
                if (body.get("runtime_graphs") != graph_mode
                        or body.get("promotion_claim") is not False
                        or body.get("non_promotable") is not True
                        or body.get("hip_residency_proved") is not True):
                    return False
                stage_status.append("complete")
            if (stage_status[1] != "absent"
                    and stage_status[0] != "complete"):
                return False
        if ((root / "reservation-release.json").exists()
                or (root / "reservation-releases").exists()):
            releases = _reservation_release_epochs(
                root, identity["operation_key"])
            for opened in partial_claims:
                release = releases.get(opened.claim_id)
                if not isinstance(release, Mapping):
                    return False
                try:
                    closed = device_claim.ClaimReceipt.from_dict(
                        release.get("device_claim_released"))
                except (TypeError, ValueError):
                    return False
                if (closed.released_at is None
                        or replace(closed, released_at=None).to_dict()
                        != opened.to_dict()):
                    return False
        elif partial_claims:
            return False
        return True
    except (GpuSourceAdapterError, evidence.EvidenceProducerError,
            gpu_source_proofs.ProofError, OSError, TypeError, ValueError,
            KeyError):
        return False


def _git_snapshot_bytes(root: Path, args: Sequence[str], label: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise GpuSourceAdapterError(f"protected root {label} is unreadable")
    return completed.stdout


def _diff_binding(content: bytes) -> dict[str, Any]:
    """Bind exact diff bytes as well as their digest; never normalize a patch."""
    return {"bytes": content, "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def _regular_binding(path: Path, relative: str) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise GpuSourceAdapterError("protected root has unsafe untracked entry") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise GpuSourceAdapterError(
                "protected root has special or hardlinked untracked entry")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_nlink,
                                row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        if identity(before) != identity(after) or size != after.st_size:
            raise GpuSourceAdapterError(
                "protected root untracked entry changed while hashing")
        return {"path": relative, "kind": "file",
                "mode": stat.S_IMODE(after.st_mode), "size": size,
                "sha256": digest.hexdigest()}
    finally:
        os.close(fd)


def _untracked_tree_rows(root: Path, entry: Path) -> list[dict[str, Any]]:
    """Recursively bind one untracked/ignored entry without following anything."""
    rows: list[dict[str, Any]] = []

    def relative(path: Path) -> str:
        try:
            value = path.relative_to(root).as_posix()
            value.encode("utf-8", "strict")
        except (ValueError, UnicodeError) as exc:
            raise GpuSourceAdapterError(
                "protected root untracked path is not normalized beneath root") from exc
        if not value or value.startswith("../") or "/../" in value:
            raise GpuSourceAdapterError(
                "protected root untracked path escaped its root")
        return value

    def visit(path: Path) -> None:
        try:
            before = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise GpuSourceAdapterError(
                "protected root untracked entry is unreadable") from exc
        name = relative(path)
        if stat.S_ISLNK(before.st_mode):
            raise GpuSourceAdapterError("protected root has untracked symlink")
        if stat.S_ISREG(before.st_mode):
            rows.append(_regular_binding(path, name))
            return
        if not stat.S_ISDIR(before.st_mode):
            raise GpuSourceAdapterError("protected root has special untracked entry")
        rows.append({"path": name, "kind": "directory",
                     "mode": stat.S_IMODE(before.st_mode)})
        try:
            children = sorted(path.iterdir(), key=lambda item: item.name.encode("utf-8"))
        except (OSError, UnicodeError) as exc:
            raise GpuSourceAdapterError(
                "protected root untracked directory is unreadable") from exc
        for child in children:
            visit(child)
        after = path.stat(follow_symlinks=False)
        before_identity = (before.st_dev, before.st_ino, before.st_mode,
                           before.st_mtime_ns, before.st_ctime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_mode,
                          after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity:
            raise GpuSourceAdapterError(
                "protected root untracked directory changed while hashing")

    visit(entry)
    return rows


def _untracked_binding(root: Path) -> dict[str, Any]:
    # Normal porcelain reports one root for an untracked directory (including a
    # nested repository/worktree) instead of asking Git to understand its
    # contents. This binds Git's exact visible untracked state; ignored build
    # outputs remain outside protected source authority.
    names: set[str] = set()
    raw = _git_snapshot_bytes(
        root, ("status", "--porcelain=v1", "-z", "--untracked-files=normal"),
        "untracked inventory")
    for item in raw.split(b"\0"):
        if not item.startswith(b"?? "):
            continue
        try:
            names.add(item[3:].decode("utf-8", "strict").rstrip("/"))
        except UnicodeDecodeError as exc:
            raise GpuSourceAdapterError(
                "protected root has a non-UTF-8 untracked path") from exc
    selected: list[str] = []
    for name in sorted(names, key=lambda value: (len(Path(value).parts), value)):
        lexical = Path(name)
        if lexical.is_absolute() or not name or ".." in lexical.parts:
            raise GpuSourceAdapterError("protected root untracked path escaped its root")
        if any(lexical.is_relative_to(Path(parent)) for parent in selected):
            continue
        selected.append(name)
    rows: list[dict[str, Any]] = []
    for name in selected:
        entry = root / name
        # Resolve the parent only. Resolving the entry would follow the very
        # symlink the scanner must reject.
        try:
            entry.parent.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise GpuSourceAdapterError("protected root untracked path escaped its root") from exc
        rows.extend(_untracked_tree_rows(root, entry))
    return {"roots": selected, "items": rows,
            "sha256": schemas.content_hash({"roots": selected, "items": rows})}


def _protected_snapshot(paths: Sequence[Path],
                        files: Sequence[evidence.BoundInputFile]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in paths:
        if raw.is_symlink():
            raise GpuSourceAdapterError("protected root is not a real directory")
        path = raw.resolve()
        if not path.is_dir():
            raise GpuSourceAdapterError("protected root is not a real directory")
        head = _git_snapshot_bytes(path, ("rev-parse", "HEAD"), "HEAD").decode().strip()
        branch = _git_snapshot_bytes(
            path, ("branch", "--show-current"), "branch").decode().strip()
        worktree_diff = _git_snapshot_bytes(
            path, ("diff", "--binary", "--no-ext-diff", "--no-textconv", "--"),
            "working diff")
        index_diff = _git_snapshot_bytes(
            path, ("diff", "--cached", "--binary", "--no-ext-diff",
                   "--no-textconv", "--"), "index diff")
        result[str(path)] = {
            "head": head, "branch": branch,
            # Preexisting tracked/index dirt is state, not a cleanliness veto.
            # Exact diff bytes prevent a hash-only snapshot from concealing what
            # was actually compared before and after the governed operation.
            "working_diff": _diff_binding(worktree_diff),
            "index_diff": _diff_binding(index_diff),
            "untracked": _untracked_binding(path),
        }
    file_rows: dict[str, Any] = {}
    for item in files:
        try:
            evidence._verify_bound(item)
        except evidence.EvidenceProducerError as exc:
            raise GpuSourceAdapterError(
                "protected production artifacts changed") from exc
        file_rows[str(item.path.resolve())] = item.sha256
    result["protected_files"] = file_rows
    return result


def _validate_intent(intent: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    required = {
        "schema": OPERATION_SCHEMA,
        "authority": AUTHORITY,
        "promotion_claim": False,
        **expected,
    }
    if (set(intent) != set(required) | {"receipt_sha256"}
            or any(intent.get(key) != value
                   for key, value in required.items())
            or intent.get("receipt_sha256") != schemas.content_hash(required)):
        raise GpuSourceAdapterError("operation intent differs from inflight identity")


def _series_payload(series: Sequence[controller.SealedScreen], *,
                    current: controller.SealedScreen) -> tuple[list[dict[str, Any]], list[float]]:
    typed = tuple(series)
    if not typed or typed[-1] != current:
        raise GpuSourceAdapterError("receipt series must end with the current measured result")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    effects: list[float] = []
    for screen in typed:
        if not isinstance(screen, controller.SealedScreen):
            raise GpuSourceAdapterError("receipt series contains an untyped result")
        if screen.result_sha256 in seen:
            raise GpuSourceAdapterError("receipt series contains a duplicate result")
        seen.add(screen.result_sha256)
        _load_screen_receipt(screen)
        rows.append(_screen_dict(screen))
        effects.append(float(screen.effect_fraction))
    return rows, effects


def _screen_dict(screen: controller.SealedScreen) -> dict[str, Any]:
    raw = asdict(screen)
    for key in (
            "composition_build_pair", "composition_correctness",
            "composition_comparison", "cumulative_performance",
            "cumulative_performance_ref"):
        typed = getattr(screen, key)
        raw[key] = None if typed is None else typed.to_dict()
    return _json_value(raw)


def _with_series_key(screen: controller.SealedScreen,
                     series_key: str) -> controller.SealedScreen:
    if "series_key" not in screen.__dataclass_fields__:
        return screen
    return replace(screen, series_key=series_key)


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _source_frame(operation_root: Path, result: controller.SealedScreen) -> tuple[str, gpu_source_proofs.GpuSourceProofBundle]:
    bundle = evidence.load_gpu_source_evidence_bundle(
        operation_root / "proof/proof-bundle.json")
    pair = gpu_source_proofs.load_receipt(
        Path(str(bundle.attribution["path"])), schema=evidence.PAIR_SCHEMA)["body"]
    series_key = schemas.content_hash({
        "manifest_sha256": pair["manifest_sha256"],
        "model_sha256": pair["model_sha256"],
        "workload_sha256": pair["workload_sha256"],
        "runtime_config_sha256": pair["runtime_config_sha256"],
        "candidate_build_identity": pair["candidate_build_identity"],
        "anchor_build_identity": pair["anchor_build_identity"],
        # Baseline receipts contain fresh samples/timestamps.  They remain
        # evidence on each result but cannot define progression identity, or
        # S2 would never pool with S1.
        "baseline_frame": {
            "anchor_build_identity": pair["anchor_build_identity"],
            "model_sha256": pair["model_sha256"],
            "workload_sha256": pair["workload_sha256"],
            "runtime_config_sha256": pair["runtime_config_sha256"],
        },
    })
    return series_key, bundle


def _composition_runner_fields(
        candidate: controller.PlannedCandidate,
        build: controller.GpuSourceBuild, operation_root: Path,
        target_runtime_args: Any | None = None,
) -> dict[str, Any]:
    plan = _candidate_composition_plan(candidate)
    pair = build.composition_build_pair
    if plan is None:
        if pair is not None:
            raise GpuSourceAdapterError(
                "ordinary build acquired cumulative recovery authority")
        return {
            "composition_plan_sha256": None,
            "composition_build_pair": None,
            "composition_correctness": None,
            "composition_production_authority": None,
        }
    if pair is None:
        raise GpuSourceAdapterError(
            "cumulative runner plan lacks its typed build pair")
    production = build.composition_production_authority
    if production is None:
        raise GpuSourceAdapterError(
            "cumulative runner plan lacks frozen-production authority")
    try:
        pair.bind_plan(plan)
        production.bind_plan(plan)
        bundle = evidence.load_gpu_source_evidence_bundle(
            operation_root / "proof/proof-bundle.json")
        if (bundle.anchor != pair.anchor.build_identity
                or bundle.candidate != pair.candidate.build_identity):
            raise cumulative_composition.CompositionError(
                "proof bundle build identities differ from cumulative pair")
        correctness_body = bundle.correctness.get("body")
        if not isinstance(correctness_body, Mapping):
            raise cumulative_composition.CompositionError(
                "proof bundle lacks full-correctness body")
        correctness = cumulative_composition.FullCorrectness.create(
            pair, suite_id="current-gpu-source-full-correctness-v1",
            cases_sha256=schemas.content_hash(correctness_body),
            receipt_sha256=str(bundle.correctness["file_sha256"]),
            passed=True)
        attribution = bundle.attribution.get("body")
        exact_comparison = (attribution.get("exact_duration_comparison")
                            if isinstance(attribution, Mapping) else None)
        candidate_routes = (exact_comparison.get("candidate_routes")
                            if isinstance(exact_comparison, Mapping) else None)
        if not isinstance(candidate_routes, Mapping):
            raise cumulative_composition.CompositionError(
                "proof bundle lacks the planned exact route set")
        if target_runtime_args is None:
            raise cumulative_composition.CompositionError(
                "cumulative runner lacks target runtime preflight authority")
        target_preflight = controller.gpu_discovery.preflight(
            target_runtime_args)
        target_frame = \
            cumulative_composition.planned_target_runtime_frame_sha256(
                target_preflight,
                candidate_identity=pair.candidate.build_identity)
    except (KeyError, TypeError, cumulative_composition.CompositionError,
            evidence.EvidenceProducerError, gpu_source_proofs.ProofError) as exc:
        raise GpuSourceAdapterError(
            "cumulative runner recovery evidence could not be sealed") from exc
    return {
        "composition_plan_sha256": plan.plan_sha256,
        "composition_build_pair": pair.to_dict(),
        "composition_correctness": correctness.to_dict(),
        "composition_production_authority": production.to_dict(),
        "composition_exact_route_receipt_sha256":
            str(bundle.attribution["file_sha256"]),
        "composition_expected_route_set_sha256":
            schemas.content_hash(candidate_routes),
        "composition_target_runtime_frame_sha256": target_frame,
    }


def _recover_completed_composition_screen(
        operation_root: Path, identity: Mapping[str, Any],
        runner_plan: Mapping[str, Any],
        plan: cumulative_composition.CompositionPlan,
        pair: cumulative_composition.CumulativeBuildPair,
        correctness: cumulative_composition.FullCorrectness,
        production: cumulative_composition.FrozenProductionAuthority,
) -> controller.SealedScreen:
    """Reconstruct the exact post-run typed screen from durable carriers."""
    off_raw = runner_plan.get("measurement_graphs_off_output_dir")
    on_raw = runner_plan.get("target_runtime_graphs_on_output_dir")
    production_on_raw = runner_plan.get("production_graphs_on_output_dir")
    performance_raw = runner_plan.get("cumulative_performance_path")
    if not all(isinstance(value, str) for value in (
            off_raw, on_raw, production_on_raw,
            performance_raw)):
        raise GpuSourceAdapterError(
            "cumulative runner plan lacks all measured outputs")
    runner_root = (operation_root / "runner").resolve()
    directories = tuple(Path(str(value)).resolve() for value in (
        off_raw, on_raw, production_on_raw))
    off_dir, on_dir, production_on_dir = directories
    performance_path = Path(str(performance_raw)).resolve()
    if (len(set(directories)) != 3
            or any(not path.is_relative_to(runner_root)
                   for path in directories)
            or performance_path !=
               (operation_root / "cumulative-performance.json").resolve()):
        raise GpuSourceAdapterError(
            "cumulative runner output escaped its operation")
    loaded_rows: list[tuple[Path, Mapping[str, Any]]] = []
    for output, graph_mode in (
            (off_dir, "off"), (on_dir, "on"),
            (production_on_dir, "on")):
        result_path = output / "result.json"
        if result_path.is_symlink() or not result_path.is_file():
            raise GpuSourceAdapterError(
                "cumulative completed runner output is partial")
        body = gpu_source_proofs.require_result_file(
            result_path,
            gpu_source_proofs.load_receipt(
                result_path,
                schema="epyc.autokernel.gpu_candidate_only_screen.v2"
            )["body"],
        )["body"]
        if (body.get("runtime_graphs") != graph_mode
                or body.get("promotion_claim") is not False
                or body.get("non_promotable") is not True
                or body.get("hip_residency_proved") is not True):
            raise GpuSourceAdapterError(
                "cumulative completed runner graph mode is invalid")
        loaded_rows.append((result_path, body))
    off_path, graphs_off = loaded_rows[0]
    on_path, graphs_on = loaded_rows[1]
    production_on_path, production_graphs_on = loaded_rows[2]
    projection = autokernel_progression._gpu_screen(on_path, graphs_on)
    if projection is None:
        raise GpuSourceAdapterError(
            "cumulative completed runner failed canonical progression")
    bundle = evidence.load_gpu_source_evidence_bundle(
        operation_root / "proof/proof-bundle.json")
    if (bundle.anchor != pair.anchor.build_identity
            or bundle.candidate != pair.candidate.build_identity):
        raise GpuSourceAdapterError(
            "cumulative proof/build identities changed on recovery")
    correctness_body = bundle.correctness.get("body")
    if not isinstance(correctness_body, Mapping):
        raise GpuSourceAdapterError(
            "cumulative proof lacks its correctness body")
    reconstructed_correctness = cumulative_composition.FullCorrectness.create(
        pair, suite_id="current-gpu-source-full-correctness-v1",
        cases_sha256=schemas.content_hash(correctness_body),
        receipt_sha256=str(bundle.correctness["file_sha256"]), passed=True)
    if reconstructed_correctness != correctness:
        raise GpuSourceAdapterError(
            "cumulative correctness recovery carrier changed")
    attribution = bundle.attribution.get("body")
    comparison_raw = (attribution.get("exact_duration_comparison")
                      if isinstance(attribution, Mapping) else None)
    if not isinstance(comparison_raw, Mapping):
        raise GpuSourceAdapterError(
            "cumulative proof lacks exact-route comparison")
    try:
        exact_effect = float(comparison_raw["relative_improvement_fraction"])
        off_effect = float(graphs_off["median_relative"])
        on_effect = float(graphs_on["median_relative"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GpuSourceAdapterError(
            "cumulative completed runner effects are malformed") from exc
    expected_routes = comparison_raw.get("candidate_routes")
    if not isinstance(expected_routes, (list, tuple, Mapping)):
        expected_routes = comparison_raw
    incremental = cumulative_composition.IncrementalComparison.create(
        pair, correctness,
        exact_route_receipt_sha256=str(bundle.attribution["file_sha256"]),
        exact_route_receipt_path=str(bundle.attribution["path"]),
        expected_route_set_sha256=schemas.content_hash(expected_routes),
        graphs_off_receipt_sha256=hashlib.sha256(
            off_path.read_bytes()).hexdigest(),
        graphs_off_receipt_path=off_path,
        graphs_on_receipt_sha256=hashlib.sha256(
            on_path.read_bytes()).hexdigest(),
        graphs_on_receipt_path=on_path,
        target_runtime_frame_sha256=
            cumulative_composition._target_runtime_frame_sha256(graphs_on),
        exact_route_effect_fraction=exact_effect,
        graphs_off_effect_fraction=off_effect,
        graphs_on_effect_fraction=on_effect)
    performance = cumulative_composition.performance_from_measurements(
        plan, pair, correctness, incremental,
        frozen_production=production,
        incremental_graphs_off=graphs_off,
        incremental_graphs_on=graphs_on,
        production_graphs_on=production_graphs_on,
        production_graphs_on_receipt_sha256=hashlib.sha256(
            production_on_path.read_bytes()).hexdigest(),
        production_graphs_on_receipt_path=production_on_path)
    performance_ref = cumulative_composition.seal_cumulative_performance(
        performance_path, performance)
    recovered = controller.SealedScreen(
        receipt_path=str(on_path),
        result_sha256=str(graphs_on["result_sha256"]),
        effect_fraction=on_effect,
        classification=str(projection["stage"]),
        baseline_sha256=str(graphs_on["baseline_sha256"]),
        source_proof_sha256=str(bundle.correctness["file_sha256"]),
        dispatch_proof_sha256=str(bundle.attribution["file_sha256"]),
        exact_attribution_effect_fraction=exact_effect,
        target_runtime_effect_fraction=on_effect,
        stages=("materialized", "built", "correctness", "attribution",
                "measurement_graphs_off_screen",
                "target_runtime_graphs_on_screen"),
        build_identity_sha256=schemas.content_hash(
            vars(pair.candidate.build_identity)),
        correctness_receipt_sha256=str(bundle.correctness["file_sha256"]),
        attribution_receipt_sha256=str(bundle.attribution["file_sha256"]),
        graphs_off_receipt_sha256=incremental.graphs_off_receipt_sha256,
        graphs_on_receipt_sha256=incremental.graphs_on_receipt_sha256,
        composition_build_pair=pair,
        composition_correctness=correctness,
        composition_comparison=incremental,
        cumulative_performance=performance,
        cumulative_performance_ref=performance_ref)
    series_key, _ = _source_frame(operation_root, recovered)
    recovered = _with_series_key(recovered, series_key)
    _load_screen_receipt(recovered)
    if plan.plan_sha256 != identity.get("composition_plan_sha256"):
        raise GpuSourceAdapterError(
            "recovered cumulative screen names another plan")
    return recovered


def _reservation_release_epochs(
        operation_root: Path, operation_key: str) -> dict[str, Mapping[str, Any]]:
    paths: list[Path] = []
    legacy = operation_root / "reservation-release.json"
    if legacy.exists() or legacy.is_symlink():
        paths.append(legacy)
    journal = operation_root / "reservation-releases"
    if journal.exists() or journal.is_symlink():
        if journal.is_symlink() or not journal.is_dir():
            raise GpuSourceAdapterError(
                "reservation release journal is not a real directory")
        for path in sorted(journal.iterdir()):
            if (path.is_symlink() or not path.is_file()
                    or not re.fullmatch(r"release-[0-9]{4}\.json", path.name)):
                raise GpuSourceAdapterError(
                    "reservation release journal contains an unsafe entry")
            paths.append(path)
    releases: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        body = gpu_source_proofs.load_receipt(
            path, schema=RESERVATION_RELEASE_SCHEMA)["body"]
        if (body.get("authority") != AUTHORITY
                or body.get("promotion_claim") is not False
                or body.get("operation_key") != operation_key):
            raise GpuSourceAdapterError(
                "outer reservation release identity mismatch")
        try:
            released = device_claim.ClaimReceipt.from_dict(
                body.get("device_claim_released"))
        except (TypeError, ValueError) as exc:
            raise GpuSourceAdapterError(
                "outer reservation release receipt is malformed") from exc
        if not released.released_at or released.claim_id in releases:
            raise GpuSourceAdapterError(
                "outer reservation release epoch is incomplete or duplicated")
        releases[released.claim_id] = body
    return releases


def _require_borrowed_proof_claims(
        bundle: gpu_source_proofs.GpuSourceProofBundle,
        outer_opened: Mapping[str, Any], *, operation_root: Path,
        operation_key: str) -> None:
    """Distinguish logical phase releases from the physical outer release."""
    correctness = gpu_source_proofs.load_receipt(
        Path(str(bundle.correctness["path"])), schema=evidence.CORRECTNESS_SCHEMA)["body"]
    pair = gpu_source_proofs.load_receipt(
        Path(str(bundle.attribution["path"])), schema=evidence.PAIR_SCHEMA)["body"]
    bodies = [correctness]
    for arm in ("candidate", "anchor"):
        reference = pair.get(arm)
        if not isinstance(reference, Mapping) or not isinstance(reference.get("body"), Mapping):
            raise GpuSourceAdapterError("borrowed attribution reference is malformed")
        bodies.append(reference["body"])
    claim_id = outer_opened.get("claim_id")
    historical = _reservation_release_epochs(operation_root, operation_key)
    for body in bodies:
        witness = body.get("residency_witness")
        phase_claim = body.get("device_claim_open", {}).get("claim_id")
        if (not isinstance(witness, Mapping)
                or witness.get("device_claim_mode") != "borrowed_outer_reservation"
                or not isinstance(phase_claim, str)
                or witness.get("outer_claim_id") != phase_claim
                or body.get("device_claim_mode") != "borrowed_outer_reservation"
                or body.get("device_claim_released") is not None
                or body.get("device_claim_borrowed_phase_end", {}).get(
                    "outer_claim_id") != phase_claim
                or body.get("device_claim_borrowed_phase_end", {}).get(
                    "physical_release") is not False
                or phase_claim != claim_id and phase_claim not in historical):
            raise GpuSourceAdapterError(
                "proof phase does not explicitly bind the borrowed outer reservation")


class GovernedGpuSourceAdapter:
    """Concrete screen/reconcile adapter with durable operation semantics."""

    def __init__(
        self, *, operations_root: Path,
        build_source: Callable[..., controller.GpuSourceBuild],
        plan_factory: Callable[[controller.PlannedCandidate,
                               controller.GpuSourceBuild], evidence.GpuSourceEvidencePlan],
        args_factory: Callable[..., Any],
        correctness_executor: evidence.CommandExecutor,
        rocprof_executor: evidence.CommandExecutor,
        claim_journal: Any,
        claim_acquirer: Callable[..., Any],
        claim_verifier: Callable[[Mapping[str, Any]], object],
        claim_timeout_s: float,
        reservation_manager: Any | None,
        receipt_series: Callable[[controller.PlannedCandidate,
                                  controller.SealedScreen], Sequence[controller.SealedScreen]],
        protected_roots: Sequence[Path],
        protected_files: Sequence[evidence.BoundInputFile],
        runner_attest: Callable[[], None] = lambda: None,
    ) -> None:
        if not operations_root.is_absolute():
            raise GpuSourceAdapterError("operations_root must be absolute")
        self.operations_root = operations_root
        self.build_source = build_source
        self.plan_factory = plan_factory
        self.args_factory = args_factory
        self.runner_attest = runner_attest
        self.correctness_executor = correctness_executor
        self.rocprof_executor = rocprof_executor
        self.claim_journal = claim_journal
        self.claim_acquirer = claim_acquirer
        self.claim_verifier = claim_verifier
        self.claim_timeout_s = claim_timeout_s
        self.reservation_manager = reservation_manager
        self.receipt_series_loader = receipt_series
        self.protected_roots = tuple(path.resolve() for path in protected_roots)
        if not self.protected_roots:
            raise GpuSourceAdapterError("production protected roots are required")
        self.protected_files = tuple(protected_files)
        if not self.protected_files:
            raise GpuSourceAdapterError("protected production artifacts are required")
        for item in self.protected_files:
            if not isinstance(item, evidence.BoundInputFile):
                raise GpuSourceAdapterError("protected artifact must be a typed bound file")
            if not any(item.path.resolve().is_relative_to(root)
                       for root in self.protected_roots):
                raise GpuSourceAdapterError(
                    "protected artifact must reside below a protected root")

    def _root(self, operation_key: str) -> Path:
        return self.operations_root / _operation_key(operation_key)

    def _proof_bundle(self, operation_root: Path, operation_key: str,
                      reservation_state: dict[str, Any],
                      lease: Mapping[str, Any]) -> Callable[..., gpu_source_proofs.GpuSourceProofBundle]:
        def produce(candidate: controller.PlannedCandidate,
                    build: controller.GpuSourceBuild) -> gpu_source_proofs.GpuSourceProofBundle:
            self.runner_attest()
            plan = self.plan_factory(candidate, build, lease)
            if (not isinstance(plan, evidence.GpuSourceEvidencePlan)
                    or plan.manifest_sha256 != candidate.source_manifest_sha256
                    or plan.candidate != build.candidate_identity
                    or plan.anchor != build.anchor_identity):
                raise GpuSourceAdapterError(
                    "evidence plan does not bind candidate manifest and typed builds")
            for build_path in (build.candidate_build, build.anchor_build):
                for protected in self.protected_roots:
                    try:
                        build_path.resolve().relative_to(protected)
                    except ValueError:
                        continue
                    raise GpuSourceAdapterError(
                        "GPU source evidence may not use a protected production tree")
            claim_acquirer = self.claim_acquirer
            if self.reservation_manager is not None:
                try:
                    opened = self.reservation_manager.reserve(operation_key)
                except controller.ResourceWait as exc:
                    if ((operation_root / "proof").exists()
                            or (operation_root / "runner-plan.json").exists()
                            or (operation_root / "runner").exists()):
                        raise GpuSourceAdapterError(
                            "resource contention surfaced after executor artifacts existed") from exc
                    waits = operation_root / "resource-waits"
                    if waits.exists() or waits.is_symlink():
                        _safe_wait_receipts(operation_root, {
                            "operation_key": operation_key,
                            "manifest_sha256": candidate.source_manifest_sha256})
                    else:
                        waits.mkdir(mode=0o700)
                    sequence = len(tuple(waits.iterdir())) + 1
                    build_key = getattr(build, "build_key", None)
                    materialization = getattr(build, "materialization_sha256", None)
                    receipt_path = waits / f"wait-{sequence:04d}.json"
                    loaded = evidence._seal(receipt_path, {
                        "schema": RESOURCE_WAIT_SCHEMA,
                        "authority": AUTHORITY,
                        "promotion_claim": False,
                        "operation_key": operation_key,
                        "manifest_sha256": candidate.source_manifest_sha256,
                        "build_key": build_key,
                        "materialization_sha256": materialization,
                        "gpu_executor_started": False,
                        "proof_root_created": False,
                        "runner_plan_created": False,
                        "runner_output_created": False,
                        "contention": dict(exc.receipt),
                    })
                    raise controller.ResourceWait(
                        str(exc), receipt={**dict(exc.receipt),
                                           "stage_receipt_path": str(receipt_path.resolve()),
                                           "stage_receipt_sha256": loaded["file_sha256"]}) from exc
                if not isinstance(opened, Mapping):
                    raise GpuSourceAdapterError("outer reservation returned no typed receipt")
                reservation_state["opened"] = dict(opened)
                claim_acquirer = self.reservation_manager.borrower(operation_key)
            self.runner_attest()
            try:
                bundle = evidence.produce_gpu_source_evidence(
                    output_root=operation_root / "proof", plan=plan,
                    correctness_executor=self.correctness_executor,
                    rocprof_executor=self.rocprof_executor,
                    claim_journal=self.claim_journal,
                    claim_acquirer=claim_acquirer,
                    claim_verifier=self.claim_verifier,
                    claim_timeout_s=self.claim_timeout_s)
            except evidence.CorrectnessParseRefusal as exc:
                if not exc.receipt_path or not exc.receipt_sha256:
                    raise GpuSourceAdapterError(
                        "correctness refusal lacks its durable terminal") from exc
                raise controller.CorrectnessRefusal(
                    str(exc), receipt_path=exc.receipt_path,
                    receipt_sha256=exc.receipt_sha256) from exc
            except evidence.DispatchAttributionParseRefusal as exc:
                if not exc.receipt_path or not exc.receipt_sha256:
                    raise GpuSourceAdapterError(
                        "attribution refusal lacks its durable terminal") from exc
                if getattr(candidate, "composition_plan", None) is not None:
                    pair = build.composition_build_pair
                    if pair is None:
                        raise GpuSourceAdapterError(
                            "cumulative attribution refusal lacks its build pair") from exc
                    pair.bind_plan(candidate.composition_plan)
                    correctness = evidence.load_gpu_source_correctness_receipt(
                        operation_root / "proof/correctness/receipt.json", plan)
                    full_correctness = cumulative_composition.FullCorrectness.create(
                        pair,
                        suite_id="current-gpu-source-full-correctness-v1",
                        cases_sha256=schemas.content_hash(correctness["body"]),
                        receipt_sha256=str(correctness["file_sha256"]),
                        passed=True)
                    raise controller.CumulativeAttributionRefusal(
                        str(exc), receipt_path=exc.receipt_path,
                        receipt_sha256=exc.receipt_sha256,
                        build_pair=pair,
                        correctness=full_correctness) from exc
                raise controller.DispatchAttributionRefusal(
                    str(exc), receipt_path=exc.receipt_path,
                    receipt_sha256=exc.receipt_sha256) from exc
            if self.reservation_manager is not None:
                _require_borrowed_proof_claims(
                    bundle, opened, operation_root=operation_root,
                    operation_key=operation_key)
            return bundle
        return produce

    def _build_guarded(self, candidate: Any, authorization: Any,
                       lease: Mapping[str, Any]) -> controller.GpuSourceBuild:
        before = getattr(self, "_active_protected_snapshot", None)
        if not isinstance(before, Mapping):
            raise GpuSourceAdapterError("source build lacks outer production snapshot")
        build = self.build_source(candidate, authorization, lease)
        if _protected_snapshot(self.protected_roots, self.protected_files) != before:
            raise GpuSourceAdapterError("source builder changed protected production tree")
        if not isinstance(build, controller.GpuSourceBuild):
            raise GpuSourceAdapterError("source builder returned no typed GPU build")
        operation_key = _operation_key(lease.get("operation_key"))
        operation_root = self._root(operation_key)
        expected_intent = _intent_body(
            operation_key=operation_key, candidate=candidate,
            authorization=authorization, lease=lease)
        identity = {key: expected_intent[key] for key in (
            "operation_key", "manifest_sha256", "composition_plan_sha256",
            "authorization_sha256", "lease_sha256")}
        _validate_intent(
            _read_json(operation_root / "intent.json", "operation intent"),
            identity)
        checkpoint = operation_root / "postbuild-checkpoint.json"
        if checkpoint.exists() or checkpoint.is_symlink():
            _validate_postbuild_checkpoint(
                checkpoint, identity, expected_build=build)
        else:
            evidence._seal(
                checkpoint, _postbuild_body(build=build, identity=identity))
        waits = _safe_wait_receipts(operation_root, identity)
        if waits:
            latest = waits[-1]
            if (latest.get("build_key") != build.build_key
                    or latest.get("materialization_sha256") !=
                        build.materialization_sha256):
                raise GpuSourceAdapterError(
                    "reopened build differs from its resource-wait identity")
        return build

    def screen(self, candidate: controller.PlannedCandidate, authorization: Any,
               lease: Mapping[str, Any]) -> controller.SealedScreen:
        operation_key = _operation_key(lease.get("operation_key"))
        operation_root = self._root(operation_key)
        intent = _intent_body(
            operation_key=operation_key, candidate=candidate,
            authorization=authorization, lease=lease)
        identity = {key: intent[key] for key in (
            "operation_key", "manifest_sha256", "composition_plan_sha256",
            "authorization_sha256", "lease_sha256")}
        if operation_root.exists() or operation_root.is_symlink():
            if operation_root.is_symlink() or not operation_root.is_dir():
                raise GpuSourceAdapterError("operation root is not a real directory")
            try:
                _validate_intent(
                    _read_json(operation_root / "intent.json", "operation intent"),
                    identity)
            except GpuSourceAdapterError as exc:
                raise GpuSourceAdapterError(
                    "operation already has durable state; reconcile instead of restarting") from exc
            if (not _is_resumable_wait_root(operation_root, identity)
                    and not _is_resumable_stage_root(
                        operation_root, identity, lease)):
                raise GpuSourceAdapterError(
                    "operation already has durable state; reconcile instead of restarting")
        else:
            operation_root.mkdir(mode=0o700, parents=True)
            evidence._seal(operation_root / "intent.json", intent)
        runner_args: dict[str, Any] = {}
        reservation_state: dict[str, Any] = {}

        def contained_args(candidate_: Any, build_: Any, lease_: Mapping[str, Any]) -> Any:
            args = self.args_factory(candidate_, build_, lease_)
            target_args = getattr(args, "_target_runtime_args", None)
            production_on_args = getattr(
                args, "_production_graphs_on_args", None)
            current_args = tuple(row for row in (
                args, target_args, production_on_args)
                if row is not None)
            outputs = tuple(
                Path(getattr(row, "output_dir", "")).resolve()
                for row in current_args)
            is_composition = _candidate_composition_plan(candidate_) is not None
            if is_composition and (target_args is None
                    or production_on_args is None):
                raise GpuSourceAdapterError(
                    "cumulative runner lacks its three measured arms")
            if not is_composition and production_on_args is not None:
                raise GpuSourceAdapterError(
                    "ordinary runner acquired production-comparison arms")
            if is_composition:
                setattr(
                    args, "_cumulative_performance_path",
                    str((operation_root /
                         "cumulative-performance.json").resolve()))
            runner_root = (operation_root / "runner").resolve()
            if len(set(outputs)) != len(outputs):
                raise GpuSourceAdapterError(
                    "GPU runner output directories are not distinct")
            for output in outputs:
                try:
                    output.relative_to(runner_root)
                except ValueError as exc:
                    raise GpuSourceAdapterError(
                        "GPU runner output escaped its operation directory") from exc
                if output.is_symlink():
                    raise GpuSourceAdapterError("GPU runner output is a symlink")
            if self.reservation_manager is not None:
                opened = reservation_state.get("opened")
                if not isinstance(opened, Mapping) or not opened.get("claim_id"):
                    raise GpuSourceAdapterError(
                        "runner arguments were requested before the outer reservation")
                releases = _reservation_release_epochs(
                    operation_root, operation_key)
                for current in current_args:
                    setattr(current, "_device_claim_acquirer",
                            self.reservation_manager.borrower(operation_key))
                    result_path = Path(current.output_dir).resolve() / "result.json"
                    expected_claim = opened["claim_id"]
                    if result_path.exists() and not result_path.is_symlink():
                        completed = gpu_source_proofs.load_receipt(
                            result_path,
                            schema="epyc.autokernel.gpu_candidate_only_screen.v2")["body"]
                        completed_claim = completed.get(
                            "device_claim_open", {}).get("claim_id")
                        if (not isinstance(completed_claim, str)
                                or (completed_claim != opened["claim_id"]
                                    and completed_claim not in releases)):
                            raise GpuSourceAdapterError(
                                "completed runner stage lacks its reservation epoch")
                        expected_claim = completed_claim
                    setattr(current, "_expected_outer_claim_id", expected_claim)
            runner_args["args"] = args
            plan_body = {
                "schema": RUNNER_PLAN_SCHEMA,
                "authority": AUTHORITY,
                "promotion_claim": False,
                "operation_key": operation_key,
                **_composition_runner_fields(
                    candidate_, build_, operation_root,
                    target_runtime_args=target_args),
                **({"measurement_graphs_off_output_dir": str(outputs[0]),
                    "target_runtime_graphs_on_output_dir": str(outputs[1]),
                    **({
                        "production_graphs_on_output_dir": str(outputs[2]),
                        "cumulative_performance_path": str(
                            (operation_root /
                             "cumulative-performance.json").resolve()),
                    } if is_composition else {})}
                   if target_args is not None else {"output_dir": str(outputs[0])}),
            }
            plan_path = operation_root / "runner-plan.json"
            if plan_path.exists() or plan_path.is_symlink():
                loaded = gpu_source_proofs.load_receipt(
                    plan_path,
                    schema=RUNNER_PLAN_SCHEMA)["body"]
                if ({key: value for key, value in loaded.items()
                     if key != "receipt_sha256"} != plan_body):
                    raise GpuSourceAdapterError("runner plan identity changed")
            else:
                evidence._seal(plan_path, plan_body)
            if is_composition:
                try:
                    cumulative_composition.commit_pre_run_authority(
                        operation_root)
                except cumulative_composition.CompositionError as exc:
                    raise GpuSourceAdapterError(
                        "runner pre-run authority journal refused") from exc
            return args

        protected_before = _protected_snapshot(
            self.protected_roots, self.protected_files)
        self._active_protected_snapshot = protected_before
        delegate = controller.GpuSourceScreener(
            build_source=self._build_guarded,
            proof_bundle=self._proof_bundle(
                operation_root, operation_key, reservation_state, lease),
            args_factory=contained_args, runner_attest=self.runner_attest)
        reservation_released = False

        def release_reservation() -> None:
            nonlocal reservation_released
            if reservation_released or self.reservation_manager is None:
                return
            released = self.reservation_manager.release(operation_key)
            reservation_released = True
            if released is not None:
                body = {
                    "schema": RESERVATION_RELEASE_SCHEMA,
                    "authority": AUTHORITY,
                    "promotion_claim": False,
                    "operation_key": operation_key,
                    "device_claim_released": dict(released),
                }
                legacy = operation_root / "reservation-release.json"
                if not legacy.exists() and not legacy.is_symlink():
                    evidence._seal(legacy, body)
                else:
                    existing = _reservation_release_epochs(
                        operation_root, operation_key)
                    claim_id = released.get("claim_id")
                    if claim_id in existing:
                        raise GpuSourceAdapterError(
                            "outer reservation epoch was released twice")
                    journal = operation_root / "reservation-releases"
                    if not journal.exists() and not journal.is_symlink():
                        journal.mkdir(mode=0o700)
                    elif journal.is_symlink() or not journal.is_dir():
                        raise GpuSourceAdapterError(
                            "reservation release journal is unsafe")
                    sequence = len(tuple(journal.iterdir())) + 1
                    evidence._seal(
                        journal / f"release-{sequence:04d}.json", body)
        try:
            result = delegate.screen(candidate, authorization, lease)
            _load_screen_receipt(result)
            composition_plan = _candidate_composition_plan(candidate)
            if composition_plan is not None:
                _bind_composition_screen(result, composition_plan)
            series_key, _bundle = _source_frame(operation_root, result)
            result = _with_series_key(result, series_key)
            rows, effects = _series_payload(
                self.receipt_series_loader(candidate, result), current=result)
            evidence._seal(operation_root / "screen-result.json", {
                "schema": RESULT_SCHEMA,
                "authority": AUTHORITY,
                "promotion_claim": False,
                "operation_key": operation_key,
                "manifest_sha256": candidate.source_manifest_sha256,
                "composition_plan_sha256":
                    _candidate_composition_plan_sha256(candidate),
                "screen": _screen_dict(result),
                "receipt_series": rows,
                "effects": effects,
            })
            release_reservation()
            recovered = self.reconcile({
                "operation_key": operation_key,
                "candidate": {"candidate": {
                    "source_manifest_sha256": candidate.source_manifest_sha256,
                    "composition_plan": (
                        None if getattr(candidate, "composition_plan", None) is None else
                        candidate.composition_plan.to_dict())}},
                "authorization": _mapping(authorization, "claim authorization"),
                "lease": dict(lease),
            })
            if recovered.status != "sealed_result" or recovered.result != result:
                raise GpuSourceAdapterError("freshly sealed operation did not reconcile")
            return result
        finally:
            try:
                try:
                    release_reservation()
                finally:
                    protected_after = _protected_snapshot(
                        self.protected_roots, self.protected_files)
            finally:
                del self._active_protected_snapshot
            if protected_after != protected_before:
                raise GpuSourceAdapterError(
                    "GPU source screen changed protected production tree")

    def reconcile(self, inflight: Mapping[str, Any]) -> Any:
        try:
            self.runner_attest()
            identity = _inflight_identity(inflight)
            composition_plan = _candidate_composition_plan(
                inflight.get("candidate"))
            root = self._root(identity["operation_key"])
            if not root.exists() and not root.is_symlink():
                return _recovery("safe_to_start")
            if root.is_symlink() or not root.is_dir():
                return _recovery("ambiguous")
            intent = _read_json(root / "intent.json", "operation intent")
            _validate_intent(intent, identity)
            result_path = root / "screen-result.json"
            if not result_path.exists() and not result_path.is_symlink():
                runner_plan_path = root / "runner-plan.json"
                if (composition_plan is not None
                        and runner_plan_path.is_file()
                        and not runner_plan_path.is_symlink()):
                    runner_plan, pair, correctness, production = \
                        _validated_runner_plan(
                        runner_plan_path, identity,
                        composition_plan=composition_plan)
                    if (pair is None or correctness is None
                            or production is None):
                        raise GpuSourceAdapterError(
                            "cumulative runner plan lost typed recovery evidence")
                    off_raw = runner_plan.get(
                        "measurement_graphs_off_output_dir")
                    on_raw = runner_plan.get(
                        "target_runtime_graphs_on_output_dir")
                    production_on_raw = runner_plan.get(
                        "production_graphs_on_output_dir")
                    completed = all(
                        isinstance(value, str)
                        and (Path(value).resolve() / "result.json").is_file()
                        and not (Path(value).resolve() / "result.json").is_symlink()
                        for value in (off_raw, on_raw, production_on_raw))
                    if completed:
                        result = _recover_completed_composition_screen(
                            root, identity, runner_plan, composition_plan,
                            pair, correctness, production)
                        rows, effects = _series_payload(
                            (result,), current=result)
                        evidence._seal(result_path, {
                            "schema": RESULT_SCHEMA,
                            "authority": AUTHORITY,
                            "promotion_claim": False,
                            "operation_key": identity["operation_key"],
                            "manifest_sha256": identity["manifest_sha256"],
                            "composition_plan_sha256":
                                identity["composition_plan_sha256"],
                            "screen": _screen_dict(result),
                            "receipt_series": rows,
                            "effects": effects,
                        })
                elif (composition_plan is None
                      and runner_plan_path.is_file()
                      and not runner_plan_path.is_symlink()):
                    runner_plan, pair, correctness, production = \
                        _validated_runner_plan(
                        runner_plan_path, identity)
                    if (pair is not None or correctness is not None
                            or production is not None):
                        raise GpuSourceAdapterError(
                            "ordinary runner plan gained cumulative evidence")
                    output_raw = runner_plan.get("output_dir")
                    if isinstance(output_raw, str):
                        output = Path(output_raw).resolve()
                        if not output.is_relative_to(
                                (root / "runner").resolve()):
                            raise GpuSourceAdapterError(
                                "GPU runner plan escaped operation")
                        durable_path = output / "result.json"
                        if durable_path.is_file() and not durable_path.is_symlink():
                            loaded = gpu_source_proofs.require_result_file(
                                durable_path,
                                gpu_source_proofs.load_receipt(
                                    durable_path,
                                    schema=("epyc.autokernel."
                                            "gpu_candidate_only_screen.v2"))[
                                                "body"])["body"]
                            projection = autokernel_progression._gpu_screen(
                                durable_path, loaded)
                            if (projection is None
                                    or loaded.get("hip_residency_proved") is not True):
                                raise GpuSourceAdapterError(
                                    "durable GPU result failed canonical validation")
                            placeholder = controller.SealedScreen(
                                receipt_path=str(durable_path),
                                result_sha256=str(loaded["result_sha256"]),
                                effect_fraction=float(
                                    loaded["median_relative"]),
                                classification=str(projection["stage"]),
                                baseline_sha256=str(
                                    loaded["baseline_sha256"]),
                                source_proof_sha256="0" * 64,
                                dispatch_proof_sha256="0" * 64)
                            series_key, bundle = _source_frame(
                                root, placeholder)
                            result = _with_series_key(
                                replace(
                                    placeholder,
                                    source_proof_sha256=str(
                                        bundle.correctness["file_sha256"]),
                                    dispatch_proof_sha256=str(
                                        bundle.attribution["file_sha256"])),
                                series_key)
                            rows, effects = _series_payload(
                                (result,), current=result)
                            evidence._seal(result_path, {
                                "schema": RESULT_SCHEMA,
                                "authority": AUTHORITY,
                                "promotion_claim": False,
                                "operation_key": identity["operation_key"],
                                "manifest_sha256": identity["manifest_sha256"],
                                "composition_plan_sha256": None,
                                "screen": _screen_dict(result),
                                "receipt_series": rows,
                                "effects": effects,
                            })
                if _is_resumable_wait_root(root, identity):
                    waits = _safe_wait_receipts(root, identity)
                    if waits:
                        paths = sorted((root / "resource-waits").iterdir())
                        wait_path = paths[-1]
                        return _recovery(
                            "resource_wait",
                            wait_receipt={
                                **dict(waits[-1]["contention"]),
                                "stage_receipt_path": str(wait_path.resolve()),
                                "stage_receipt_sha256": hashlib.sha256(
                                    wait_path.read_bytes()).hexdigest(),
                            })
                    return _recovery("safe_to_start")
                if (not result_path.exists()
                        and _is_resumable_stage_root(
                        root, identity,
                        _mapping(inflight.get("lease"), "inflight lease"))):
                    return _recovery("safe_to_start")
                if not result_path.exists():
                    return _recovery("ambiguous")
            raw = _read_json(result_path, "operation result")
            if self.reservation_manager is not None:
                if not _reservation_release_epochs(
                        root, identity["operation_key"]):
                    raise GpuSourceAdapterError(
                        "outer reservation was not actually released")
            required = {
                "schema": RESULT_SCHEMA,
                "authority": AUTHORITY,
                "promotion_claim": False,
                "operation_key": identity["operation_key"],
                "manifest_sha256": identity["manifest_sha256"],
                "composition_plan_sha256":
                    identity["composition_plan_sha256"],
            }
            if (set(raw) != set(required) | {
                    "screen", "receipt_series", "effects", "receipt_sha256"}
                    or any(raw.get(key) != value
                           for key, value in required.items())
                    or raw.get("receipt_sha256") != schemas.content_hash({
                        key: value for key, value in raw.items()
                        if key != "receipt_sha256"})):
                raise GpuSourceAdapterError("operation result identity mismatch")
            rows = raw.get("receipt_series")
            effects = raw.get("effects")
            if (not isinstance(rows, list) or not rows
                    or not isinstance(effects, list)
                    or len(rows) != len(effects)):
                raise GpuSourceAdapterError("operation result has no receipt series")
            screens = tuple(_typed_screen(row) for row in rows)
            if [float(row.effect_fraction) for row in screens] != effects:
                raise GpuSourceAdapterError("measured effect series differs from receipts")
            result = _typed_screen(raw.get("screen"))
            if screens[-1] != result:
                raise GpuSourceAdapterError("receipt series does not end at sealed result")
            if composition_plan is not None:
                _bind_composition_screen(result, composition_plan)
            # Re-open the producer bundle and all nested receipts on recovery.
            evidence.load_gpu_source_evidence_bundle(root / "proof/proof-bundle.json")
            return _recovery("sealed_result", result)
        except (GpuSourceAdapterError, evidence.EvidenceProducerError,
                gpu_source_proofs.ProofError,
                cumulative_composition.CompositionError,
                OSError, TypeError, ValueError, KeyError):
            return _recovery("ambiguous")

    def receipt_series(self, operation_key: str) -> tuple[controller.SealedScreen, ...]:
        """Expose validated measured replications; dashboard prose has no role."""
        raw = _read_json(self._root(operation_key) / "screen-result.json",
                         "operation result")
        rows = raw.get("receipt_series")
        if not isinstance(rows, list) or not rows:
            raise GpuSourceAdapterError("operation has no sealed receipt series")
        return tuple(_typed_screen(row) for row in rows)

    def effects(self, operation_key: str) -> tuple[float, ...]:
        return tuple(float(row.effect_fraction)
                     for row in self.receipt_series(operation_key))


def build_governed_gpu_source_adapter(
    *, operations_root: Path,
    build_source: Callable[..., controller.GpuSourceBuild],
    plan_factory: Callable[..., evidence.GpuSourceEvidencePlan],
    args_factory: Callable[..., Any],
    correctness_executor: evidence.CommandExecutor,
    rocprof_executor: evidence.CommandExecutor,
    claim_journal: Any,
    claim_acquirer: Callable[..., Any],
    claim_verifier: Callable[[Mapping[str, Any]], object],
    claim_timeout_s: float = 300.0,
    reservation_manager: Any | None = None,
    receipt_series: Callable[[controller.PlannedCandidate,
                              controller.SealedScreen], Sequence[controller.SealedScreen]]
                    = lambda _candidate, current: (current,),
    protected_roots: Sequence[Path] = (),
    protected_files: Sequence[evidence.BoundInputFile] = (),
    runner_attest: Callable[[], None] = lambda: None,
) -> GovernedGpuSourceAdapter:
    """Build the concrete controller adapter without executing any command."""
    return GovernedGpuSourceAdapter(
        operations_root=operations_root, build_source=build_source,
        plan_factory=plan_factory, args_factory=args_factory,
        correctness_executor=correctness_executor,
        rocprof_executor=rocprof_executor, claim_journal=claim_journal,
        claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
        claim_timeout_s=claim_timeout_s, reservation_manager=reservation_manager,
        receipt_series=receipt_series,
        protected_roots=protected_roots, protected_files=protected_files,
        runner_attest=runner_attest)


__all__ = [
    "GpuSourceAdapterError", "CompatibleRecovery", "GovernedGpuSourceAdapter",
    "build_governed_gpu_source_adapter",
]
