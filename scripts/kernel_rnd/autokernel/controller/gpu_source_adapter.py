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
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .. import schemas
from . import discovery_controller as controller
from . import gpu_source_evidence as evidence
from . import gpu_source_proofs
from scripts.benchmark import autokernel_progression

OPERATION_SCHEMA = "epyc.autokernel.gpu_source_operation.v1"
RESULT_SCHEMA = "epyc.autokernel.gpu_source_operation_result.v1"
AUTHORITY = evidence.AUTHORITY
SHA = re.compile(r"^[0-9a-f]{64}$")


class GpuSourceAdapterError(RuntimeError):
    """The adapter refused an unsafe start or an unprovable recovery."""


@dataclass(frozen=True)
class CompatibleRecovery:
    """Compatibility type used only by controller revisions predating Recovery."""

    status: str
    result: Any | None = None

    def __post_init__(self) -> None:
        if self.status not in {"safe_to_start", "sealed_result", "ambiguous"}:
            raise GpuSourceAdapterError("unknown recovery status")
        if (self.status == "sealed_result") != (self.result is not None):
            raise GpuSourceAdapterError("recovery result binding is invalid")


def _recovery(status: str, result: Any | None = None) -> Any:
    recovery_type = getattr(controller, "Recovery", CompatibleRecovery)
    return recovery_type(status=status, result=result)


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
        prepared = dict(value)
        if isinstance(prepared.get("stages"), list):
            prepared["stages"] = tuple(prepared["stages"])
        result = controller.SealedScreen(**prepared)
    except (TypeError, ValueError, controller.DiscoveryControllerError) as exc:
        raise GpuSourceAdapterError("sealed screen payload is invalid") from exc
    _load_screen_receipt(result)
    return result


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
        "authorization_sha256": schemas.content_hash(authorization_raw),
        "lease_sha256": schemas.content_hash(lease_raw),
    }


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
        "authorization_sha256": schemas.content_hash(authorization),
        "lease_sha256": schemas.content_hash(lease),
    }


def _protected_snapshot(paths: Sequence[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in paths:
        path = raw.resolve()
        if not path.is_dir() or path.is_symlink():
            raise GpuSourceAdapterError("protected root is not a real directory")
        completed = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, check=False)
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode or head.returncode:
            raise GpuSourceAdapterError("protected root git identity is unreadable")
        result[str(path)] = {
            "head": head.stdout.strip(),
            "status_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "clean": completed.stdout == "",
        }
    return result


def _validate_intent(intent: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    required = {
        "schema": OPERATION_SCHEMA,
        "authority": AUTHORITY,
        "promotion_claim": False,
        **expected,
    }
    if any(intent.get(key) != value for key, value in required.items()):
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
        "baseline_sha256": result.baseline_sha256,
    })
    return series_key, bundle


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
        receipt_series: Callable[[controller.PlannedCandidate,
                                  controller.SealedScreen], Sequence[controller.SealedScreen]],
        protected_roots: Sequence[Path],
    ) -> None:
        if not operations_root.is_absolute():
            raise GpuSourceAdapterError("operations_root must be absolute")
        self.operations_root = operations_root
        self.build_source = build_source
        self.plan_factory = plan_factory
        self.args_factory = args_factory
        self.correctness_executor = correctness_executor
        self.rocprof_executor = rocprof_executor
        self.claim_journal = claim_journal
        self.claim_acquirer = claim_acquirer
        self.claim_verifier = claim_verifier
        self.claim_timeout_s = claim_timeout_s
        self.receipt_series_loader = receipt_series
        self.protected_roots = tuple(path.resolve() for path in protected_roots)
        if not self.protected_roots:
            raise GpuSourceAdapterError("production protected roots are required")

    def _root(self, operation_key: str) -> Path:
        return self.operations_root / _operation_key(operation_key)

    def _proof_bundle(self, operation_root: Path) -> Callable[..., gpu_source_proofs.GpuSourceProofBundle]:
        def produce(candidate: controller.PlannedCandidate,
                    build: controller.GpuSourceBuild) -> gpu_source_proofs.GpuSourceProofBundle:
            plan = self.plan_factory(candidate, build)
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
            return evidence.produce_gpu_source_evidence(
                output_root=operation_root / "proof", plan=plan,
                correctness_executor=self.correctness_executor,
                rocprof_executor=self.rocprof_executor,
                claim_journal=self.claim_journal,
                claim_acquirer=self.claim_acquirer,
                claim_verifier=self.claim_verifier,
                claim_timeout_s=self.claim_timeout_s)
        return produce

    def _build_guarded(self, candidate: Any, authorization: Any,
                       lease: Mapping[str, Any]) -> controller.GpuSourceBuild:
        before = getattr(self, "_active_protected_snapshot", None)
        if not isinstance(before, Mapping):
            raise GpuSourceAdapterError("source build lacks outer production snapshot")
        build = self.build_source(candidate, authorization, lease)
        if _protected_snapshot(self.protected_roots) != before:
            raise GpuSourceAdapterError("source builder changed protected production tree")
        if not isinstance(build, controller.GpuSourceBuild):
            raise GpuSourceAdapterError("source builder returned no typed GPU build")
        return build

    def screen(self, candidate: controller.PlannedCandidate, authorization: Any,
               lease: Mapping[str, Any]) -> controller.SealedScreen:
        operation_key = _operation_key(lease.get("operation_key"))
        operation_root = self._root(operation_key)
        if operation_root.exists() or operation_root.is_symlink():
            raise GpuSourceAdapterError(
                "operation already has durable state; reconcile instead of restarting")
        operation_root.mkdir(parents=True)
        intent = _intent_body(
            operation_key=operation_key, candidate=candidate,
            authorization=authorization, lease=lease)
        evidence._seal(operation_root / "intent.json", intent)
        runner_args: dict[str, Any] = {}

        def contained_args(candidate_: Any, build_: Any, lease_: Mapping[str, Any]) -> Any:
            args = self.args_factory(candidate_, build_, lease_)
            output = Path(getattr(args, "output_dir", "")).resolve()
            runner_root = (operation_root / "runner").resolve()
            try:
                output.relative_to(runner_root)
            except ValueError as exc:
                raise GpuSourceAdapterError(
                    "GPU runner output escaped its operation directory") from exc
            if output.exists() or output.is_symlink():
                raise GpuSourceAdapterError("GPU runner output must be fresh")
            runner_args["args"] = args
            evidence._seal(operation_root / "runner-plan.json", {
                "schema": "epyc.autokernel.gpu_source_runner_plan.v1",
                "authority": AUTHORITY,
                "promotion_claim": False,
                "operation_key": operation_key,
                "output_dir": str(output),
            })
            return args

        protected_before = _protected_snapshot(self.protected_roots)
        if not all(row["clean"] for row in protected_before.values()):
            raise GpuSourceAdapterError("protected production tree is dirty before screen")
        self._active_protected_snapshot = protected_before
        delegate = controller.GpuSourceScreener(
            build_source=self._build_guarded,
            proof_bundle=self._proof_bundle(operation_root),
            args_factory=contained_args)
        try:
            result = delegate.screen(candidate, authorization, lease)
        finally:
            protected_after = _protected_snapshot(self.protected_roots)
            del self._active_protected_snapshot
            if protected_after != protected_before:
                raise GpuSourceAdapterError(
                    "GPU source screen changed protected production tree")
        _load_screen_receipt(result)
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
            "screen": _screen_dict(result),
            "receipt_series": rows,
            "effects": effects,
        })
        recovered = self.reconcile({
            "operation_key": operation_key,
            "candidate": {"candidate": {
                "source_manifest_sha256": candidate.source_manifest_sha256}},
            "authorization": _mapping(authorization, "claim authorization"),
            "lease": dict(lease),
        })
        if recovered.status != "sealed_result" or recovered.result != result:
            raise GpuSourceAdapterError("freshly sealed operation did not reconcile")
        return result

    def reconcile(self, inflight: Mapping[str, Any]) -> Any:
        try:
            identity = _inflight_identity(inflight)
            root = self._root(identity["operation_key"])
            if not root.exists() and not root.is_symlink():
                return _recovery("safe_to_start")
            if root.is_symlink() or not root.is_dir():
                return _recovery("ambiguous")
            intent = _read_json(root / "intent.json", "operation intent")
            _validate_intent(intent, identity)
            result_path = root / "screen-result.json"
            if not result_path.exists() and not result_path.is_symlink():
                plan = _read_json(root / "runner-plan.json", "GPU runner plan")
                if (plan.get("operation_key") != identity["operation_key"]
                        or plan.get("authority") != AUTHORITY
                        or plan.get("promotion_claim") is not False):
                    raise GpuSourceAdapterError("GPU runner plan identity mismatch")
                output = Path(str(plan.get("output_dir", ""))).resolve()
                try:
                    output.relative_to((root / "runner").resolve())
                except ValueError as exc:
                    raise GpuSourceAdapterError("GPU runner plan escaped operation") from exc
                durable_path = output / "result.json"
                if not durable_path.is_file() or durable_path.is_symlink():
                    return _recovery("ambiguous")
                loaded = gpu_source_proofs.require_result_file(
                    durable_path,
                    gpu_source_proofs.load_receipt(
                        durable_path,
                        schema="epyc.autokernel.gpu_candidate_only_screen.v2")["body"])["body"]
                projection = autokernel_progression._gpu_screen(durable_path, loaded)
                if projection is None or loaded.get("hip_residency_proved") is not True:
                    raise GpuSourceAdapterError("durable GPU result failed canonical validation")
                _series_key, bundle = _source_frame(root, controller.SealedScreen(
                    receipt_path=str(durable_path),
                    result_sha256=str(loaded["result_sha256"]),
                    effect_fraction=float(loaded["median_relative"]),
                    classification=str(projection["stage"]),
                    baseline_sha256=str(loaded["baseline_sha256"]),
                    source_proof_sha256="0" * 64,
                    dispatch_proof_sha256="0" * 64))
                result = controller.SealedScreen(
                    receipt_path=str(durable_path),
                    result_sha256=str(loaded["result_sha256"]),
                    effect_fraction=float(loaded["median_relative"]),
                    classification=str(projection["stage"]),
                    baseline_sha256=str(loaded["baseline_sha256"]),
                    source_proof_sha256=str(bundle.correctness["file_sha256"]),
                    dispatch_proof_sha256=str(bundle.attribution["file_sha256"]))
                result = _with_series_key(result, _series_key)
                rows, effects = _series_payload((result,), current=result)
                evidence._seal(result_path, {
                    "schema": RESULT_SCHEMA,
                    "authority": AUTHORITY,
                    "promotion_claim": False,
                    "operation_key": identity["operation_key"],
                    "manifest_sha256": identity["manifest_sha256"],
                    "screen": _screen_dict(result),
                    "receipt_series": rows,
                    "effects": effects,
                })
            raw = _read_json(result_path, "operation result")
            required = {
                "schema": RESULT_SCHEMA,
                "authority": AUTHORITY,
                "promotion_claim": False,
                "operation_key": identity["operation_key"],
                "manifest_sha256": identity["manifest_sha256"],
            }
            if any(raw.get(key) != value for key, value in required.items()):
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
            # Re-open the producer bundle and all nested receipts on recovery.
            evidence.load_gpu_source_evidence_bundle(root / "proof/proof-bundle.json")
            return _recovery("sealed_result", result)
        except (GpuSourceAdapterError, evidence.EvidenceProducerError,
                gpu_source_proofs.ProofError, OSError, TypeError, ValueError,
                KeyError):
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
    plan_factory: Callable[[controller.PlannedCandidate,
                           controller.GpuSourceBuild], evidence.GpuSourceEvidencePlan],
    args_factory: Callable[..., Any],
    correctness_executor: evidence.CommandExecutor,
    rocprof_executor: evidence.CommandExecutor,
    claim_journal: Any,
    claim_acquirer: Callable[..., Any],
    claim_verifier: Callable[[Mapping[str, Any]], object],
    claim_timeout_s: float = 300.0,
    receipt_series: Callable[[controller.PlannedCandidate,
                              controller.SealedScreen], Sequence[controller.SealedScreen]]
                    = lambda _candidate, current: (current,),
    protected_roots: Sequence[Path] = (),
) -> GovernedGpuSourceAdapter:
    """Build the concrete controller adapter without executing any command."""
    return GovernedGpuSourceAdapter(
        operations_root=operations_root, build_source=build_source,
        plan_factory=plan_factory, args_factory=args_factory,
        correctness_executor=correctness_executor,
        rocprof_executor=rocprof_executor, claim_journal=claim_journal,
        claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
        claim_timeout_s=claim_timeout_s, receipt_series=receipt_series,
        protected_roots=protected_roots)


__all__ = [
    "GpuSourceAdapterError", "CompatibleRecovery", "GovernedGpuSourceAdapter",
    "build_governed_gpu_source_adapter",
]
