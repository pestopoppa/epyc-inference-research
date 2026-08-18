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
from . import gpu_source_evidence as evidence
from . import gpu_source_proofs
from scripts.benchmark import autokernel_progression

OPERATION_SCHEMA = "epyc.autokernel.gpu_source_operation.v1"
RESULT_SCHEMA = "epyc.autokernel.gpu_source_operation_result.v1"
RESOURCE_WAIT_SCHEMA = "epyc.autokernel.gpu_source_resource_wait.v1"
RESERVATION_RELEASE_SCHEMA = "epyc.autokernel.gpu_source_reservation_release.v1"
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
        rows.append(row)
    return tuple(rows)


def _is_resumable_wait_root(root: Path, identity: Mapping[str, Any]) -> bool:
    allowed = {"intent.json", "source-manifest.json", "resource-waits"}
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
    _safe_wait_receipts(root, identity)
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


def _is_resumable_stage_root(root: Path, identity: Mapping[str, Any]) -> bool:
    """Return true only for an ordered proof journal with terminal boundaries.

    Raw output without a receipt is deliberately not resumable: the adapter
    cannot know whether the corresponding GPU command completed.  Completed
    receipts, however, are exactly-once checkpoints and must survive controller
    or API reloads.
    """
    allowed_root = {
        "intent.json", "source-manifest.json", "resource-waits", "proof",
    }
    if any(path.name not in allowed_root for path in root.iterdir()):
        return False
    proof = root / "proof"
    if proof.is_symlink() or not proof.is_dir():
        return False
    allowed_proof = {
        "correctness", "attribution-candidate", "attribution-anchor",
        "attribution-pair.json", "proof-bundle.json",
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
        if pair.exists() or pair.is_symlink():
            if completed_arms != 2:
                return False
            _validated_stage_receipt(pair, evidence.PAIR_SCHEMA, identity)
        bundle = proof / "proof-bundle.json"
        if bundle.exists() or bundle.is_symlink():
            if not pair.exists():
                return False
            evidence.load_gpu_source_evidence_bundle(bundle)
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


def _require_borrowed_proof_claims(
        bundle: gpu_source_proofs.GpuSourceProofBundle,
        outer_opened: Mapping[str, Any]) -> None:
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
    for body in bodies:
        witness = body.get("residency_witness")
        if (not isinstance(witness, Mapping)
                or witness.get("device_claim_mode") != "borrowed_outer_reservation"
                or witness.get("outer_claim_id") != claim_id
                or body.get("device_claim_open", {}).get("claim_id") != claim_id
                or body.get("device_claim_mode") != "borrowed_outer_reservation"
                or body.get("device_claim_released") is not None
                or body.get("device_claim_borrowed_phase_end", {}).get(
                    "outer_claim_id") != claim_id
                or body.get("device_claim_borrowed_phase_end", {}).get(
                    "physical_release") is not False):
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
                raise controller.DispatchAttributionRefusal(
                    str(exc), receipt_path=exc.receipt_path,
                    receipt_sha256=exc.receipt_sha256) from exc
            if self.reservation_manager is not None:
                _require_borrowed_proof_claims(bundle, opened)
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
        return build

    def screen(self, candidate: controller.PlannedCandidate, authorization: Any,
               lease: Mapping[str, Any]) -> controller.SealedScreen:
        operation_key = _operation_key(lease.get("operation_key"))
        operation_root = self._root(operation_key)
        intent = _intent_body(
            operation_key=operation_key, candidate=candidate,
            authorization=authorization, lease=lease)
        identity = {key: intent[key] for key in (
            "operation_key", "manifest_sha256", "authorization_sha256", "lease_sha256")}
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
                    and not _is_resumable_stage_root(operation_root, identity)):
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
            outputs = (Path(getattr(args, "output_dir", "")).resolve(),
                       *((Path(getattr(target_args, "output_dir", "")).resolve(),)
                         if target_args is not None else ()))
            runner_root = (operation_root / "runner").resolve()
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
                setattr(args, "_device_claim_acquirer",
                        self.reservation_manager.borrower(operation_key))
                setattr(args, "_expected_outer_claim_id", opened["claim_id"])
                if target_args is not None:
                    setattr(target_args, "_device_claim_acquirer",
                            self.reservation_manager.borrower(operation_key))
                    setattr(target_args, "_expected_outer_claim_id", opened["claim_id"])
            runner_args["args"] = args
            plan_body = {
                "schema": "epyc.autokernel.gpu_source_runner_plan.v1",
                "authority": AUTHORITY,
                "promotion_claim": False,
                "operation_key": operation_key,
                **({"measurement_graphs_off_output_dir": str(outputs[0]),
                    "target_runtime_graphs_on_output_dir": str(outputs[1])}
                   if target_args is not None else {"output_dir": str(outputs[0])}),
            }
            plan_path = operation_root / "runner-plan.json"
            if plan_path.exists() or plan_path.is_symlink():
                loaded = gpu_source_proofs.load_receipt(
                    plan_path,
                    schema="epyc.autokernel.gpu_source_runner_plan.v1")["body"]
                if loaded != plan_body:
                    raise GpuSourceAdapterError("runner plan identity changed")
            else:
                evidence._seal(plan_path, plan_body)
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
                evidence._seal(operation_root / "reservation-release.json", {
                    "schema": RESERVATION_RELEASE_SCHEMA,
                    "authority": AUTHORITY,
                    "promotion_claim": False,
                    "operation_key": operation_key,
                    "device_claim_released": dict(released),
                })
        try:
            result = delegate.screen(candidate, authorization, lease)
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
            release_reservation()
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
            root = self._root(identity["operation_key"])
            if not root.exists() and not root.is_symlink():
                return _recovery("safe_to_start")
            if root.is_symlink() or not root.is_dir():
                return _recovery("ambiguous")
            intent = _read_json(root / "intent.json", "operation intent")
            _validate_intent(intent, identity)
            result_path = root / "screen-result.json"
            if not result_path.exists() and not result_path.is_symlink():
                if _is_resumable_wait_root(root, identity):
                    return _recovery("safe_to_start")
                if _is_resumable_stage_root(root, identity):
                    return _recovery("safe_to_start")
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
            if self.reservation_manager is not None:
                release = _read_json(
                    root / "reservation-release.json", "outer reservation release")
                if (release.get("schema") != RESERVATION_RELEASE_SCHEMA
                        or release.get("authority") != AUTHORITY
                        or release.get("promotion_claim") is not False
                        or release.get("operation_key") != identity["operation_key"]):
                    raise GpuSourceAdapterError("outer reservation release identity mismatch")
                try:
                    released = device_claim.ClaimReceipt.from_dict(
                        release.get("device_claim_released"))
                except (TypeError, ValueError) as exc:
                    raise GpuSourceAdapterError(
                        "outer reservation release receipt is malformed") from exc
                if not released.released_at:
                    raise GpuSourceAdapterError("outer reservation was not actually released")
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
