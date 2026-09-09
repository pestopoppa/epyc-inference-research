"""Native, held consumer for the pure retention closure.

The public objects in this module are deliberately Python-native capabilities.
There is no ``from_dict`` execution path: serialized retention plans are useful
reports, never deletion authority.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from .. import storage
from . import candidate_manifest as cm
from . import candidate_transactions as ct
from . import retention


class RetentionConsumerRefused(RuntimeError):
    """A native view or held execution cannot prove a safe expiry."""


def inspect_candidate_state(transactions: ct.CandidateTransactions) -> cm.CandidateState:
    """Read the actual candidate transaction owner, never a caller-supplied mapping."""
    if not isinstance(transactions, ct.CandidateTransactions):
        raise TypeError("transactions must be CandidateTransactions")
    snapshot = transactions.inspect()
    expected = {"schema", "initialized", "state", "state_digest",
                "completed_transactions", "historical_validation_receipt",
                "current_evidence_eligibility"}
    if not isinstance(snapshot, Mapping) or set(snapshot) != expected \
            or snapshot["schema"] != ct.POINTER_SCHEMA or snapshot["initialized"] is not True:
        raise RetentionConsumerRefused("native candidate transactions are unavailable/malformed")
    state = cm.CandidateState.from_dict(snapshot["state"])
    encoded = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    if hashlib.sha256(encoded).hexdigest() != snapshot["state_digest"]:
        raise RetentionConsumerRefused("native candidate state digest mismatch")
    return state


@dataclass(frozen=True, slots=True)
class ManifestArtifacts:
    """Artifact membership emitted by the native candidate owner."""

    manifest: cm.CandidateManifest
    artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, cm.CandidateManifest):
            raise TypeError("manifest must be a CandidateManifest")
        object.__setattr__(self, "manifest", self.manifest.validated())
        if not self.artifact_ids or any(not isinstance(item, str) or not item
                                        for item in self.artifact_ids):
            raise RetentionConsumerRefused("manifest artifact IDs must be non-empty text")
        if len(set(self.artifact_ids)) != len(self.artifact_ids):
            raise RetentionConsumerRefused("manifest artifact IDs contain duplicates")


@dataclass(frozen=True, slots=True)
class NativeRoots:
    """The fixed root vocabulary supplied by the controller/native owners."""

    production: tuple[str, ...]
    rollback: tuple[str, ...]
    pending_calibration: tuple[str, ...] = ()
    active_workers: tuple[str, ...] = ()
    launch_intents: tuple[str, ...] = ()
    acquisition_intents: tuple[str, ...] = ()
    integration_intents: tuple[str, ...] = ()
    retained_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.production or not self.rollback:
            raise RetentionConsumerRefused("production and rollback roots are mandatory")
        for name in self.__dataclass_fields__:
            values = getattr(self, name)
            if any(not isinstance(item, str) or not item for item in values):
                raise RetentionConsumerRefused(f"{name} roots must be artifact IDs")
            if len(set(values)) != len(values):
                raise RetentionConsumerRefused(f"{name} roots contain duplicates")


@dataclass(frozen=True, slots=True)
class NativeArtifactIdentity:
    """Git/worktree identity paired with an exact artifact path and digest."""

    artifact_id: str
    path: str
    sha256: str
    branch: str | None
    source: cm.SourceIdentity | None

    def __post_init__(self) -> None:
        if self.source is not None:
            if not isinstance(self.source, cm.SourceIdentity):
                raise TypeError("source must be a SourceIdentity or None")
            object.__setattr__(self, "source",
                               cm.SourceIdentity.from_dict(self.source.to_dict()))
        if (self.branch is None) != (self.source is None):
            raise RetentionConsumerRefused(
                "native source and branch must both be known or both unavailable")
        if self.branch is not None and (not isinstance(self.branch, str)
                                        or not self.branch.strip()):
            raise RetentionConsumerRefused("known native branch must be non-empty")
        if not Path(self.path).is_absolute() or str(Path(self.path)) != self.path:
            raise RetentionConsumerRefused("native artifact path must be absolute and canonical")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise RetentionConsumerRefused("native artifact sha256 must be lowercase SHA-256")
        if not self.artifact_id:
            raise RetentionConsumerRefused("native artifact ID is required")


@dataclass(frozen=True, slots=True)
class NativeRetentionView:
    """One atomic native generation, including candidate and operational roots."""

    snapshot_id: str
    generation: int
    candidate_state: cm.CandidateState
    manifests: tuple[ManifestArtifacts, ...]
    nodes: tuple[retention.ArtifactNode, ...]
    roots: NativeRoots
    identities: tuple[NativeArtifactIdentity, ...]
    uncertain_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_state, cm.CandidateState):
            raise TypeError("candidate_state must be a CandidateState")
        if not isinstance(self.roots, NativeRoots):
            raise TypeError("roots must be NativeRoots")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool) \
                or self.generation < 1:
            raise RetentionConsumerRefused("native generation must be a positive integer")
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise RetentionConsumerRefused("native snapshot_id must be non-empty")
        if any(not isinstance(item, ManifestArtifacts) for item in self.manifests):
            raise TypeError("manifests must contain ManifestArtifacts")
        if any(not isinstance(item, retention.ArtifactNode) for item in self.nodes):
            raise TypeError("nodes must contain ArtifactNode")
        if any(not isinstance(item, NativeArtifactIdentity) for item in self.identities):
            raise TypeError("identities must contain NativeArtifactIdentity")


_ROOT_FIELDS = (
    ("production", "production"), ("rollback", "rollback"),
    ("pending_calibration", "pending_calibration"),
    ("active_workers", "active_worker"), ("launch_intents", "launch_intent"),
    ("acquisition_intents", "launch_intent"),
    ("integration_intents", "integration_intent"),
    ("retained_evidence", "retained_evidence"),
)


def collect_native_snapshot(view: NativeRetentionView) -> retention.RetentionSnapshot:
    """Project an atomic native view into the existing immutable closure contract."""
    if not isinstance(view, NativeRetentionView):
        raise TypeError("view must be a NativeRetentionView, not caller JSON")
    node_ids = {item.artifact_id for item in view.nodes}
    identities = {item.artifact_id: item for item in view.identities}
    if len(identities) != len(view.identities):
        raise RetentionConsumerRefused("duplicate native artifact identities")
    for node in view.nodes:
        if node.path is None:
            continue
        identity = identities.get(node.artifact_id)
        if identity is None or identity.path != node.path:
            raise RetentionConsumerRefused(
                f"path artifact {node.artifact_id} lacks an exact native identity")
        if node.expiry is not None and identity.sha256 != node.expiry.sha256:
            raise RetentionConsumerRefused(
                f"expiry identity for {node.artifact_id} differs from native identity")
        if identity.branch is not None and identity.branch.startswith(
                ("production-consolidated-", "production-speech-")) \
                and node.retention_class == "expirable":
            raise RetentionConsumerRefused(
                f"artifact {node.artifact_id} is on protected branch {identity.branch}")
        if node.retention_class == "expirable" and identity.source is None:
            raise RetentionConsumerRefused(
                f"expirable artifact {node.artifact_id} lacks source ownership")
    by_manifest: dict[str, tuple[str, ...]] = {}
    for binding in view.manifests:
        digest = binding.manifest.manifest_digest
        if digest in by_manifest:
            raise RetentionConsumerRefused(f"duplicate native manifest {digest}")
        if not set(binding.artifact_ids).issubset(node_ids):
            raise RetentionConsumerRefused(f"manifest {digest} names an unknown artifact")
        by_manifest[digest] = binding.artifact_ids

    state = view.candidate_state.validated()
    roots: list[retention.RetentionRoot] = []
    for native_name, root_kind in _ROOT_FIELDS:
        artifact_ids = getattr(view.roots, native_name)
        if artifact_ids:
            roots.append(retention.RetentionRoot(f"native:{native_name}", root_kind,
                                                  artifact_ids))

    def manifest_root(digest: str | None, root_kind: str, root_id: str) -> None:
        if digest is None:
            return
        artifact_ids = by_manifest.get(digest)
        if artifact_ids is None:
            raise RetentionConsumerRefused(
                f"native candidate root {root_id} has no exact manifest artifacts")
        roots.append(retention.RetentionRoot(root_id, root_kind, artifact_ids))

    manifest_root(state.integration_tip, "integration_candidate", "candidate:integration")
    manifest_root(state.validated_candidate, "validated_candidate", "candidate:validated")
    for batch in state.active_batches:
        manifest_root(batch.candidate_manifest_digest, "pending_validation",
                      f"batch:{batch.batch_id}:candidate")
        manifest_root(batch.comparator_manifest_digest, "pending_comparator",
                      f"batch:{batch.batch_id}:comparator")
        if batch.required_loo_keep_ids:
            manifest_root(batch.candidate_manifest_digest, "pending_loo",
                          f"batch:{batch.batch_id}:loo")
    if not set(item for root in roots for item in root.artifact_ids).issubset(node_ids):
        raise RetentionConsumerRefused("a native operational root names an unknown artifact")
    return retention.RetentionSnapshot.create(
        snapshot_id=view.snapshot_id, generation=view.generation, nodes=view.nodes,
        roots=roots, uncertain_scopes=view.uncertain_scopes)


@dataclass(frozen=True, slots=True)
class NativeTombstone:
    event_id: str
    record: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MaintenanceLease:
    """State available only while the controller's exclusion is held."""

    view: NativeRetentionView
    policy: storage.StoragePolicy
    journal: Any
    owner_receipt: str
    tombstones: tuple[NativeTombstone, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.owner_receipt, str) or not self.owner_receipt:
            raise RetentionConsumerRefused("held maintenance owner receipt is required")
        if len(self.tombstones) > 192:
            raise RetentionConsumerRefused(
                "held tombstone history exceeds the bounded 64-artifact selection")


T = TypeVar("T")


class MaintenanceOwner:
    """Fixed capability interface implemented by the primary controller owner."""

    def held(self, operation: Callable[[MaintenanceLease], T]) -> T:
        raise RetentionConsumerRefused(
            "unified retention maintenance/root-exclusion owner is unavailable")


@dataclass(frozen=True, slots=True)
class RetentionJob:
    plan: retention.RetentionPlan
    selected_artifact_ids: tuple[str, ...]
    policy_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, retention.RetentionPlan):
            raise TypeError("plan must be a native RetentionPlan, not caller JSON")
        if not self.selected_artifact_ids or len(self.selected_artifact_ids) > 64:
            raise RetentionConsumerRefused("a retention batch must contain 1..64 artifacts")
        if len(set(self.selected_artifact_ids)) != len(self.selected_artifact_ids):
            raise RetentionConsumerRefused("selected artifact IDs contain duplicates")
        if not set(self.selected_artifact_ids).issubset(self.plan.expirable_ids):
            raise RetentionConsumerRefused("selection is outside the complete retention plan")
        if len(self.policy_digest) != 64 or any(
                char not in "0123456789abcdef" for char in self.policy_digest):
            raise RetentionConsumerRefused("policy_digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class HeldRetentionResult:
    plan_digest: str
    snapshot_generation: int
    owner_receipt: str
    outcomes: tuple[storage.ExpiryOutcome, ...]


def prepare(view: NativeRetentionView, policy: storage.StoragePolicy, *, limit: int = 64,
            now=None) -> tuple[RetentionJob | None, tuple[storage.ExpiryOutcome, ...]]:
    """Return a noncreating dry preview and bounded native job."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 64:
        raise ValueError("limit must be in 1..64")
    snapshot = collect_native_snapshot(view)
    plan = retention.plan_retention(snapshot)
    selected = plan.expirable_ids[:limit]
    previews = retention.plan_expiry_candidates(
        plan, snapshot, policy, artifact_ids=selected, now=now)
    return (RetentionJob(plan, selected, _policy_digest(policy)) if selected else None, previews)


def _policy_digest(policy: storage.StoragePolicy) -> str:
    if not isinstance(policy, storage.StoragePolicy):
        raise TypeError("policy must be a StoragePolicy")
    encoded = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact(node: retention.ArtifactNode) -> storage.ExpirableArtifact:
    assert node.expiry is not None and node.path is not None
    item = node.expiry
    return storage.ExpirableArtifact(
        path=node.path, campaign_id=item.campaign_id, sha256=item.sha256,
        durability_class=item.durability_class, expirable_kind=item.expirable_kind,
        reason=item.reason, rule_id=item.rule_id, actor=item.actor,
        retention_class=node.retention_class, preconditions=dict(item.preconditions),
        declared_size_bytes=node.declared_size_bytes)


def _verify_content(node: retention.ArtifactNode) -> None:
    assert node.path is not None and node.expiry is not None
    path = Path(node.path)
    actual = storage.hash_tree_manifest(path) if path.is_dir() else storage.hash_file(path)
    if actual != node.expiry.sha256:
        raise RetentionConsumerRefused(
            f"artifact {node.artifact_id} content changed after native collection")


def _tombstone_index(records: Sequence[NativeTombstone]) \
        -> dict[str, dict[str, NativeTombstone]]:
    """Validate and key one bounded native history view per held operation."""
    found: dict[str, dict[str, NativeTombstone]] = {}
    for event in records:
        if not isinstance(event, NativeTombstone):
            raise TypeError("tombstones must be NativeTombstone values")
        if not isinstance(event.event_id, str) or not event.event_id:
            raise RetentionConsumerRefused("native tombstone event ID is required")
        record = dict(event.record)
        violations = storage.validate_artifact_tombstone(record)
        if violations:
            raise RetentionConsumerRefused("native journal contains an invalid tombstone")
        states = found.setdefault(record["tombstone_id"], {})
        if record["reclamation_state"] in states:
            raise RetentionConsumerRefused("native journal has duplicate tombstone state")
        states[record["reclamation_state"]] = event
    return found


def _verify_replay(record: Mapping[str, Any], node: retention.ArtifactNode,
                   policy: storage.StoragePolicy) -> None:
    """Bind replay to the complete current descriptor and held policy context."""
    assert node.expiry is not None and node.path is not None
    descriptor = node.expiry
    expected = {
        "campaign_id": descriptor.campaign_id,
        "artifact_path": os.path.realpath(node.path),
        "artifact_sha256": descriptor.sha256,
        "durability_class": descriptor.durability_class,
        "retention_class": node.retention_class,
        "expirable_kind": descriptor.expirable_kind,
        "rule_id": descriptor.rule_id,
        "reason": descriptor.reason,
        "actor": descriptor.actor,
        "preconditions": dict(descriptor.preconditions),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RetentionConsumerRefused(
            f"prior tombstone conflicts with current descriptor for {node.artifact_id}")
    target = Path(expected["artifact_path"])
    if not any(target != Path(root) and target.is_relative_to(Path(root))
               for root in policy.owned_roots):
        raise RetentionConsumerRefused("prior tombstone is outside the held policy roots")


class _ReplaySink:
    def __init__(self, sink: storage.JournalTombstoneSink,
                 existing: Mapping[str, NativeTombstone]):
        self.sink, self.existing = sink, existing

    def append(self, record: Mapping[str, Any]) -> str:
        previous = self.existing.get(record["reclamation_state"])
        if previous is not None:
            before, current = dict(previous.record), dict(record)
            # Time is a presentation field; identity/content and measured bytes must agree.
            before.pop("reclaimed_at", None)
            current.pop("reclaimed_at", None)
            if before != current:
                raise RetentionConsumerRefused("prior tombstone conflicts with current artifact")
            return previous.event_id
        return self.sink.append(record)


def execute(job: RetentionJob, *, owner: MaintenanceOwner | None = None,
            now=None) -> HeldRetentionResult:
    """Execute only through a held native owner; default authority is unavailable."""
    if not isinstance(job, RetentionJob):
        raise TypeError("job must be a native RetentionJob, not caller JSON")
    capability = owner or MaintenanceOwner()

    def under_hold(lease: MaintenanceLease) -> HeldRetentionResult:
        if not isinstance(lease, MaintenanceLease):
            raise TypeError("maintenance owner returned no typed lease")
        snapshot = collect_native_snapshot(lease.view)
        retention.validate_plan_snapshot(job.plan, snapshot)
        if job.policy_digest != _policy_digest(lease.policy):
            raise RetentionConsumerRefused("held storage policy differs from prepared policy")
        nodes = {item.artifact_id: item for item in snapshot.nodes}
        sink = storage.JournalTombstoneSink(lease.journal)
        tombstones = _tombstone_index(lease.tombstones)
        outcomes: list[storage.ExpiryOutcome] = []
        for artifact_id in job.selected_artifact_ids:
            node = nodes[artifact_id]
            descriptor = node.expiry
            assert descriptor is not None and node.path is not None
            expected_id = storage.tombstone_id(
                descriptor.campaign_id, os.path.realpath(node.path), descriptor.sha256,
                descriptor.expirable_kind, descriptor.rule_id)
            previous = tombstones.get(expected_id, {})
            for event in previous.values():
                _verify_replay(event.record, node, lease.policy)
            if "reclaimed" in previous:
                if os.path.lexists(node.path):
                    raise RetentionConsumerRefused("reclaimed tombstone conflicts with live bytes")
                event = previous["reclaimed"]
                record = dict(event.record)
                outcomes.append(storage.ExpiryOutcome(
                    "RECLAIMED", record, record["size_bytes"], record["file_count"],
                    False, (event.event_id,)))
                continue
            if not os.path.lexists(node.path) and "intent" in previous:
                intent = previous["intent"]
                done = dict(intent.record)
                done["reclamation_state"] = "reclaimed"
                done_id = sink.append(done)
                outcomes.append(storage.ExpiryOutcome(
                    "RECLAIMED", done, done["size_bytes"], done["file_count"], False,
                    (intent.event_id, done_id)))
                continue
            # The policy preview is repeated inside the exclusion immediately
            # before mutation; its measured bytes, not the plan estimate, win.
            storage.plan_expiry(_artifact(node), lease.policy, now=now)
            _verify_content(node)
            outcome = storage.expire_artifact(
                _artifact(node), lease.policy, journal=_ReplaySink(sink, previous),
                force=True, now=now)
            outcomes.append(outcome)
        return HeldRetentionResult(job.plan.plan_digest, snapshot.generation,
                                   lease.owner_receipt, tuple(outcomes))

    return capability.held(under_hold)


__all__ = [
    "HeldRetentionResult", "MaintenanceLease", "MaintenanceOwner", "ManifestArtifacts",
    "NativeArtifactIdentity", "NativeRetentionView", "NativeRoots", "NativeTombstone", "RetentionConsumerRefused",
    "RetentionJob", "collect_native_snapshot", "execute", "prepare",
    "inspect_candidate_state",
]
