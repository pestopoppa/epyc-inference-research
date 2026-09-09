"""Execution-thread-owned continuous evidence; configuration is never authority."""
from __future__ import annotations

import builtins
import math
import re
import sys
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Callable, Mapping

from . import evidence_feed, scoped_evidence as evidence
from .journal_feed_owner import DrainLimits, FeedOwnerPending
from .validation_semantic_adapter import _source_bytes

CONFIG_SCHEMA = "epyc.autokernel.standalone_evidence_feed.v1"
ROOT_SOURCES_V1 = (
    "scripts/vidya/adapters/autokernel_unified_arm.py",
    "scripts/vidya/claim_tuple.py", "scripts/vidya/frames.py", "scripts/vidya/ledger.py",
    "scripts/vidya/canonical.py", "scripts/vidya/lattice.py",
)
ROOT_SOURCES_V2 = ROOT_SOURCES_V1 + (
    "scripts/vidya/adapters/autokernel_final_trial.py",
)
ROOT_SOURCES = ROOT_SOURCES_V2
ROOT_PROJECTION_SCHEMA_V1 = "epyc.autokernel.root_feed_projection.v1"
ROOT_PROJECTION_SCHEMA_V2 = "epyc.autokernel.root_feed_projection.v2"


class FeedRuntimeRefused(RuntimeError):
    """Ownership, configuration, or durable evidence is unsafe."""


class FeedNotReady(RuntimeError):
    """A bounded operation made no admissible progress; retry on the same owner."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeedRuntimeRefused(f"{label} must be nonempty text")
    return value


def _root_source_schema(pins: Mapping[str, str]) -> str:
    if not isinstance(pins, Mapping):
        raise FeedRuntimeRefused("installed ROOT requires its exact source closure pins")
    fields = set(pins)
    if fields == set(ROOT_SOURCES_V1):
        schema = ROOT_PROJECTION_SCHEMA_V1
    elif fields == set(ROOT_SOURCES_V2):
        schema = ROOT_PROJECTION_SCHEMA_V2
    else:
        raise FeedRuntimeRefused("installed ROOT requires its exact source closure pins")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in pins.values()):
        raise FeedRuntimeRefused("installed ROOT source digest is malformed")
    return schema


@dataclass(frozen=True)
class FeedConfig:
    binding_id: str
    expected_epoch: str
    source_root: str
    corpus_root: str
    ledger_path: str
    store_root: str
    reader_id: str = evidence_feed.READER_ID
    max_events: int = 32
    max_bytes: int = 4 * 1024 * 1024
    max_seconds: float = 0.25
    max_shards: int = 64
    max_projection_entries: int = 10_000
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise FeedRuntimeRefused("unsupported evidence feed schema")
        for name in ("binding_id", "expected_epoch", "reader_id"):
            _text(getattr(self, name), name)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.reader_id):
            raise FeedRuntimeRefused("reader_id is malformed")
        for name in ("source_root", "corpus_root", "ledger_path", "store_root"):
            if not Path(_text(getattr(self, name), name)).is_absolute():
                raise FeedRuntimeRefused(f"{name} must be an explicit absolute path")
        for name in ("max_events", "max_bytes", "max_shards", "max_projection_entries"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise FeedRuntimeRefused(f"{name} must be a positive integer")
        if self.max_projection_entries > 10_000:
            raise FeedRuntimeRefused("max_projection_entries exceeds 10000")
        if (isinstance(self.max_seconds, bool)
                or not isinstance(self.max_seconds, (float, int))
                or not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise FeedRuntimeRefused("max_seconds must be finite and positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeedConfig:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise FeedRuntimeRefused("feed configuration fields differ")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def limits(self) -> DrainLimits:
        return DrainLimits(max_events=self.max_events, max_bytes=self.max_bytes,
                           max_seconds=self.max_seconds, max_shards=self.max_shards)


@dataclass(frozen=True)
class LoadedFeedProjection:
    adapter: ModuleType
    claim_tuple: ModuleType
    frames: ModuleType
    ledger: ModuleType
    source_sha256: Mapping[str, str]

    @property
    def source_schema(self) -> str:
        return _root_source_schema(self.source_sha256)

    @classmethod
    def load(cls, root: Path, pins: Mapping[str, str]) -> LoadedFeedProjection:
        # Reuse the canonical pinned-loader's bounded stable read. Execute only
        # those captured bytes, with a closed ROOT import closure; no pyc/cache.
        if not isinstance(pins, Mapping):
            raise FeedRuntimeRefused("installed ROOT requires its exact source closure pins")
        pins = dict(pins)
        schema = _root_source_schema(pins)
        sources = {path: _source_bytes(root / path, digest) for path, digest in pins.items()}
        names = {"canonical": "scripts/vidya/canonical.py",
                 "lattice": "scripts/vidya/lattice.py",
                 "claim_tuple": "scripts/vidya/claim_tuple.py",
                 "frames": "scripts/vidya/frames.py",
                 "ledger": "scripts/vidya/ledger.py",
                 "autokernel_unified_arm": "scripts/vidya/adapters/autokernel_unified_arm.py"}
        if schema == ROOT_PROJECTION_SCHEMA_V2:
            names["autokernel_final_trial"] = ROOT_SOURCES_V2[-1]
        modules: dict[str, ModuleType] = {}
        prefix = "_autokernel_feed_" + uuid.uuid4().hex
        # This namespace is retained by the import closure, not ambient module
        # lookup. ROOT's final reader imports the already-loaded arm in reverse.
        adapters = ModuleType(prefix + "_adapters")

        def pinned_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level:
                arm = modules.get("autokernel_unified_arm")
                final = modules.get("autokernel_final_trial")
                if (level == 1 and arm is not None and globals is arm.__dict__
                        and name == "autokernel_final_trial"
                        and fromlist in (("REFERENCE_SCHEMA",), ("validate_final",))):
                    if final is None:
                        raise FeedRuntimeRefused("six-file ROOT closure has no final-v3 helper capability")
                    return final
                if (level == 1 and final is not None and globals is final.__dict__
                        and name == "" and fromlist == ("autokernel_unified_arm",)):
                    return adapters
                raise FeedRuntimeRefused("ROOT relative import is outside the captured closure")
            if level == 0 and name in names:
                if name not in modules:
                    raise FeedRuntimeRefused("pinned ROOT dependency imported before loading")
                return modules[name]
            return builtins.__import__(name, globals, locals, fromlist, level)

        registered = []
        try:
            for name, path in names.items():
                qualified = prefix + "_" + name
                module = ModuleType(qualified)
                module.__file__ = str(root / path)
                module.__dict__["__builtins__"] = dict(vars(builtins), __import__=pinned_import)
                sys.modules[qualified] = module  # dataclasses resolves its own module.
                registered.append(qualified)
                modules[name] = module
                if name in {"autokernel_unified_arm", "autokernel_final_trial"}:
                    module.__package__ = adapters.__name__
                    setattr(adapters, name, module)
                exec(compile(sources[path], str(root / path), "exec", dont_inherit=True),
                     module.__dict__)
            adapter, claim = modules["autokernel_unified_arm"], modules["claim_tuple"]
            if claim.registered().get("autokernel-unified-arm-measurement") is not adapter.project:
                raise FeedRuntimeRefused("pinned canonical projector was not registered")
            return cls(adapter, claim, modules["frames"], modules["ledger"],
                       MappingProxyType(dict(pins)))
        finally:
            for name in registered:
                sys.modules.pop(name, None)


def validate_paths(config: FeedConfig, controller_store: Path) -> None:
    """The reader consumes this controller, with disjoint projection writers."""
    source = Path(config.source_root)
    expected = controller_store / "journal"
    if source != expected.absolute() or source.resolve() != expected.resolve():
        raise FeedRuntimeRefused("feed source must be this controller's Journal root")
    paths = (source, Path(config.corpus_root), Path(config.store_root),
             Path(config.ledger_path).parent)
    resolved = tuple(path.resolve() for path in paths)
    for path in paths:
        for ancestor in (path, *path.parents):
            if ancestor.is_symlink():
                raise FeedRuntimeRefused("feed paths must not traverse symlink aliases")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise FeedRuntimeRefused("feed source/corpus/projection/ledger roots overlap")
    for writer in resolved[2:]:
        store = controller_store.resolve()
        if writer == store or writer in store.parents or store in writer.parents:
            raise FeedRuntimeRefused("feed projection writers overlap the controller store")


@dataclass(frozen=True)
class InstalledFeedBinding:
    """Application-installed source/epoch and optional real finding verifier owner.

    No serialized manifest can create these callables. Without the optional
    complete verifier set, native per-arm ingestion remains ranking-unknown.
    """

    root_repo: Path
    root_source_sha256: Mapping[str, str]
    current_epoch: str
    finding_projector: Callable[..., evidence.Finding | None] | None = None
    scope_verifier: Callable[..., bool | str] | None = None
    use_verifier: Callable[..., bool | str] | None = None
    result_verifier: Callable[..., bool | str] | None = None
    support_rule_identity: str | None = None

    def __post_init__(self) -> None:
        root = Path(self.root_repo)
        if not root.is_absolute():
            raise FeedRuntimeRefused("installed ROOT requires its exact source closure pins")
        if not isinstance(self.root_source_sha256, Mapping):
            raise FeedRuntimeRefused("installed ROOT requires its exact source closure pins")
        pins = dict(self.root_source_sha256)
        _root_source_schema(pins)
        object.__setattr__(self, "root_repo", root)
        object.__setattr__(self, "root_source_sha256", MappingProxyType(pins))
        _text(self.current_epoch, "installed epoch")
        callbacks = (self.finding_projector, self.scope_verifier,
                     self.use_verifier, self.result_verifier)
        if any(item is not None for item in callbacks):
            if not all(callable(item) for item in callbacks):
                raise FeedRuntimeRefused("finding support requires the whole installed verifier set")
            _text(self.support_rule_identity, "installed support rule")
        elif self.support_rule_identity is not None:
            raise FeedRuntimeRefused("arm-only ingestion cannot name an unsupported finding rule")

    def load(self, config: FeedConfig) -> LoadedFeedProjection:
        if config.expected_epoch != self.current_epoch:
            raise FeedRuntimeRefused("feed epoch differs from installed epoch owner")
        return LoadedFeedProjection.load(self.root_repo, self.root_source_sha256)


class FeedEvidenceView:
    """Stable concrete planner interface; it never retains a replaceable cache."""

    def __init__(self, owner: FeedRuntimeOwner) -> None:
        self.owner = owner

    @property
    def current_epoch(self) -> str:
        return self.owner.binding.current_epoch

    def _query(self, dependencies, signatures) -> evidence.ProjectionCompleteness:
        feed = self.owner.current_feed()
        keys = {("dependency", dep) for dep in dependencies}
        keys.update(("signature", signature) for signature in signatures)
        evicted = tuple(key for key in sorted(keys) if feed._db.execute(
            "SELECT 1 FROM evicted WHERE kind = ? AND key = ?", key).fetchone())
        return evidence.ProjectionCompleteness(evicted)

    def planning_evidence(self, claim: evidence.ClaimKey, *,
                          intended_use: str) -> evidence.PlanningEvidence:
        if not self.owner.ready:
            raise FeedNotReady("captured evidence admission frontier is not projected")
        feed = self.owner.current_feed()
        generation = self.owner.generation
        index = feed.index()
        before_fences = index.fence_snapshot()
        signature = evidence._mandatory_signature_digest(claim)
        dependencies = set(claim.dependency_identities)
        for finding in index.findings:
            if (finding.claim_key.digest == claim.digest
                    or evidence._mandatory_signature_digest(finding.claim_key) == signature):
                dependencies.update(finding.claim_key.dependency_identities)
        completeness = self._query(dependencies, (signature,))
        bundle = index.planning_evidence(
            claim, intended_use=intended_use, projection_completeness=completeness)
        if (generation != self.owner.generation or index is not feed.index()
                or index.fence_snapshot() != before_fences):
            raise FeedRuntimeRefused("feed changed during its exact planning query")
        frontier = feed.state["projected_frontier"]
        result_body = bundle.retrieval.to_dict()
        result_body.pop("result_digest")
        result_body["snapshot_frontier"] = frontier
        retrieval = replace(bundle.retrieval, snapshot_frontier=frontier,
                            result_digest=evidence.schemas.content_hash(result_body))
        snapshot = replace(bundle.snapshot, snapshot_frontier=frontier,
            retrieval_result_digest=retrieval.result_digest,
            index_digest=evidence.schemas.content_hash({
                "index": bundle.snapshot.index_digest, "projection_frontier": frontier,
                "admission_frontier": self.owner.last_admitted_frontier,
                "owner_generation": generation}))
        return evidence.PlanningEvidence(retrieval, snapshot)

    def retrieve(self, scope, claim, intended_use, limit=40):
        if dict(scope) != dict(claim.target_scope) or limit != 40:
            raise FeedRuntimeRefused("feed planner query requires exact scope and limit 40")
        return self.planning_evidence(claim, intended_use=intended_use).retrieval

    def proposal_snapshot(self, claim, *, intended_use):
        return self.planning_evidence(claim, intended_use=intended_use).snapshot

    def fences_for(self, proposal: evidence.ProposalSnapshot) -> evidence.LocalFenceSnapshot:
        index = self.owner.current_feed().index()
        fences = index.fence_snapshot()
        completeness = self._query(proposal.dependency_generations, proposal.semantic_fences)
        selected = {key: fences.semantic_fences.get(key, evidence._EMPTY_EVIDENCE_DIGEST)
                    for key in proposal.semantic_fences}
        return replace(fences, available=fences.available and self.owner.ready,
                       semantic_fences=completeness.bind(selected))


class FeedRuntimeOwner:
    """Config-only until first execution-thread drain; never a background worker."""

    def __init__(self, config: FeedConfig, binding: InstalledFeedBinding) -> None:
        if not isinstance(config, FeedConfig) or not isinstance(binding, InstalledFeedBinding):
            raise FeedRuntimeRefused("feed owner requires typed configuration and installed binding")
        if config.expected_epoch != binding.current_epoch:
            raise FeedRuntimeRefused("feed epoch differs from installed epoch owner")
        self.config, self.binding = config, binding
        self.view = FeedEvidenceView(self)
        self._thread: threading.Thread | None = None
        self._feed: evidence_feed.EvidenceFeed | None = None
        self.closed = False
        self.failed: str | None = None
        self.generation = 0
        self.last_snapshot: Mapping[str, Any] | None = None
        self._admission_frontier: int | None = None
        self.last_admitted_frontier: int | None = None
        self.ready = False
        self._observation_at = None
        self._attempted_at = None
        self._observation_error = None
        self._observation_source = None
        self._observation_generation = 0
        self._observation_ready = False
        self._observation_admitted = None

    def _check_thread(self) -> None:
        if self._thread is not None and self._thread is not threading.current_thread():
            raise FeedRuntimeRefused("evidence feed belongs to another execution thread")

    def current_feed(self) -> evidence_feed.EvidenceFeed:
        self._check_thread()
        if self.closed or self.failed or self._feed is None:
            raise FeedRuntimeRefused("evidence feed has no usable current owner")
        return self._feed

    def drain(self) -> Mapping[str, Any]:
        from . import runtime_aggregates as aggregate
        self._check_thread()
        if self.closed or self.failed:
            raise FeedRuntimeRefused(self.failed or "evidence feed is closed")
        self._thread = threading.current_thread()
        self.ready = False
        try:
            self._attempted_at = aggregate.utc_now()
            self._observation_error = None
        except Exception:
            self._observation_error = "diagnostic clock unavailable"
        try:
            if self._feed is None:
                self._feed = self._open()
            result = self._feed.drain_once(self.config.limits())
        except BaseException as exc:
            self._observation_error = str(exc)[:512]
            cause = exc
            while cause is not None:
                if isinstance(cause, (BlockingIOError, TimeoutError,
                                      FeedOwnerPending,
                                      evidence_feed.FeedProjectionPending)):
                    raise FeedNotReady(str(exc) or "native evidence owner is temporarily busy") from exc
                cause = cause.__cause__
            self.failed = f"evidence feed requires recovery: {exc}"
            raise FeedRuntimeRefused(self.failed) from exc
        self.generation += 1
        if self._admission_frontier is None:
            self._admission_frontier = result["source_frontier"]
        self.last_snapshot = MappingProxyType(dict(result,
            admission_frontier=self._admission_frontier))
        if (result.get("proof_pending") or result.get("readiness") == "outage"
                or result["projection_frontier"] < self._admission_frontier):
            self._capture_observation()
            raise FeedNotReady("bounded evidence projection/proof is not ready")
        self.last_admitted_frontier = self._admission_frontier
        self._admission_frontier = None
        self.ready = True
        self._capture_observation()
        return self.last_snapshot

    def _capture_observation(self):
        try:
            from . import runtime_aggregates as aggregate
            stamp = aggregate.utc_now()
            self._observation_source = self.last_snapshot
            self._observation_generation = self.generation
            self._observation_ready = self.ready
            self._observation_admitted = self.last_admitted_frontier
            self._observation_at = self._attempted_at = stamp
            self._observation_error = None
        except Exception:
            self._observation_error = "diagnostic clock unavailable"

    def observation_snapshot(self):
        """Read cached owner facts only, on the existing execution thread."""
        from . import runtime_aggregates as aggregate
        self._check_thread()
        source = self._observation_source
        data = None
        if source is not None and self._observation_at is not None:
            data = {"reader_id": self.config.reader_id, "epoch": self.binding.current_epoch,
                "owner_state": "closed" if self.closed else "failed" if self.failed else
                    "ready" if self._observation_ready else "pending",
                "ready": self._observation_ready and not self.closed and not self.failed,
                "readiness": source["readiness"], "source_frontier": source["source_frontier"],
                "cursor_frontier": source["cursor_frontier"], "projection_frontier": source["projection_frontier"],
                "admission_frontier": source["admission_frontier"],
                "last_admitted_frontier": self._observation_admitted,
                "projection_checksum": source["projection_checksum"],
                "proof_pending": bool(source.get("proof_pending", False)),
                "lag_events": source["lag"], "lag_seconds": None,
                "quarantine_count": source["quarantine_count"], "cached_finding_count": source["finding_count"]}
        reason = self.failed or ("evidence owner closed" if self.closed else
            "captured frontier projected; support is per-query" if self.ready else
            "evidence admission frontier not yet available")
        return aggregate.observation("evidence",
            status="available" if self.ready and data is not None and self._observation_error is None else "unknown",
            reason=reason,
            data=data, observed_at=self._observation_at, attempted_at=self._attempted_at,
            generation=self._observation_generation, error=self._observation_error)

    def _open(self) -> evidence_feed.EvidenceFeed:
        projection = self.binding.load(self.config)
        return evidence_feed.EvidenceFeed(
                source_root=Path(self.config.source_root), corpus_root=Path(self.config.corpus_root),
                ledger_path=Path(self.config.ledger_path), store_root=Path(self.config.store_root),
                root_repo=self.binding.root_repo, current_epoch=self.binding.current_epoch,
                loaded_projection=projection,
                reader_id=self.config.reader_id,
                finding_projector=self.binding.finding_projector,
                scope_verifier=self.binding.scope_verifier, use_verifier=self.binding.use_verifier,
                result_verifier=self.binding.result_verifier,
                support_rule_identity=self.binding.support_rule_identity,
                max_projection_entries=self.config.max_projection_entries)
    def close(self) -> None:
        from . import runtime_aggregates as aggregate
        self._check_thread()
        if self._feed is not None:
            self._feed.close()
            self._feed = None
        self.closed = True
        self.ready = False
        try:
            self._attempted_at = aggregate.utc_now()
        except Exception:
            self._observation_error = "diagnostic clock unavailable"

    def close_if_owner(self) -> bool:
        if self.closed:
            return True
        if self._thread is None or self._thread is threading.current_thread():
            self.close()
            return True
        return False
