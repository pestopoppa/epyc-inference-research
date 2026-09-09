"""Controller-owned retained-artifact catalog projection.

The catalog records declarations and durable owner relationships.  It does not
measure files, decide expiry, hold resources, or delete anything.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any

from . import campaign, candidate_manifest as cm, retention
from . import resolved_recipe as rr, unified_planner
from . import native_model_preparation as model_preparation
from .. import storage


SEED_SCHEMA = "epyc.autokernel.native_retention_catalog_seed.v1"
EVENT_SCHEMA = "epyc.autokernel.native_retention_catalog_event.v1"
EVENT = "INSTALLED"
_TOKEN = object()
_VIEW_TOKEN = object()
MAX_INVENTORY_BYTES = 8 * 1024 * 1024


class NativeRetentionCatalogRefused(ValueError):
    """Catalog input is incomplete, mutable, stale, or foreign."""


@dataclass(frozen=True, slots=True)
class PreparedCatalogView:
    """Opaque result of one external collection against an owner frontier."""

    view: Any
    frontier: tuple[Any, ...]
    view_digest: str
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _VIEW_TOKEN or not isinstance(self.frontier, tuple):
            raise NativeRetentionCatalogRefused("catalog view lacks native preparation")
        _sha(self.view_digest, "catalog view digest")


def _bounded_regular_bytes(path: Path, maximum: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) \
        | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise NativeRetentionCatalogRefused(f"{label} is unavailable") from exc
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > maximum):
            raise NativeRetentionCatalogRefused(
                f"{label} must be a bounded singly-linked regular file")
        data = os.read(fd, maximum + 1)
        after = os.fstat(fd)
        if (len(data) > maximum or len(data) != before.st_size
                or (before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns)):
            raise NativeRetentionCatalogRefused(f"{label} changed while read")
        return data
    finally:
        os.close(fd)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise NativeRetentionCatalogRefused("catalog value is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise NativeRetentionCatalogRefused(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise NativeRetentionCatalogRefused(f"{label} must be non-empty text")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({copy.deepcopy(key): _freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class CatalogArtifact:
    artifact_id: str
    artifact_kind: str
    path: str | None
    sha256: str | None
    identity_kind: str
    dependencies: tuple[str, ...]
    retention_class: str
    root_kinds: tuple[str, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.artifact_id, "artifact_id")
        if self.artifact_kind not in retention.ARTIFACT_KINDS:
            raise NativeRetentionCatalogRefused("catalog artifact kind is unsupported")
        if self.path is not None:
            path = Path(_text(self.path, "artifact path"))
            if not path.is_absolute() or ".." in path.parts or str(path) != self.path:
                raise NativeRetentionCatalogRefused("catalog artifact path is not canonical")
        if self.sha256 is not None:
            _sha(self.sha256, "artifact sha256")
        if self.identity_kind not in {"file_sha256", "directory_pending_measurement"}:
            raise NativeRetentionCatalogRefused("catalog identity kind is unsupported")
        if ((self.identity_kind == "file_sha256") != (self.sha256 is not None)):
            raise NativeRetentionCatalogRefused("catalog identity kind/digest differ")
        deps = tuple(_text(item, "artifact dependency") for item in self.dependencies)
        roots = tuple(_text(item, "artifact root kind") for item in self.root_kinds)
        if len(set(deps)) != len(deps) or len(set(roots)) != len(roots):
            raise NativeRetentionCatalogRefused("catalog dependencies/roots contain duplicates")
        if not set(roots) <= retention.ROOT_KINDS:
            raise NativeRetentionCatalogRefused("catalog root kind is unsupported")
        if self.retention_class not in {
                "permanent_in_repo", "permanent_large", "durable_tracked",
                "durable_untracked", "hash_and_provenance_only", "expirable"}:
            raise NativeRetentionCatalogRefused("catalog retention class is unsupported")
        if self.retention_class == "expirable":
            raise NativeRetentionCatalogRefused(
                "catalog seed cannot create expiry authority")
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise NativeRetentionCatalogRefused("artifact provenance must be non-empty")
        object.__setattr__(self, "dependencies", tuple(sorted(deps)))
        object.__setattr__(self, "root_kinds", tuple(sorted(roots)))
        object.__setattr__(self, "provenance", _freeze(_plain(self.provenance)))

    @classmethod
    def from_dict(cls, value: Any) -> "CatalogArtifact":
        fields = {"artifact_id", "artifact_kind", "path", "sha256", "identity_kind", "dependencies",
                  "retention_class", "root_kinds", "provenance"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise NativeRetentionCatalogRefused("catalog artifact fields differ")
        if not isinstance(value["dependencies"], (list, tuple)) \
                or not isinstance(value["root_kinds"], (list, tuple)):
            raise NativeRetentionCatalogRefused("catalog artifact arrays are malformed")
        return cls(value["artifact_id"], value["artifact_kind"], value["path"],
                   value["sha256"], value["identity_kind"], tuple(value["dependencies"]),
                   value["retention_class"], tuple(value["root_kinds"]),
                   value["provenance"])

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "artifact_kind": self.artifact_kind,
                "path": self.path, "sha256": self.sha256,
                "identity_kind": self.identity_kind,
                "dependencies": list(self.dependencies),
                "retention_class": self.retention_class,
                "root_kinds": list(self.root_kinds),
                "provenance": _plain(self.provenance)}


@dataclass(frozen=True, slots=True)
class NativeRetentionCatalogSeed:
    campaign_id: str
    config_digest: str
    manifest_digest: str
    artifacts: tuple[CatalogArtifact, ...]
    model_inventories: tuple[Mapping[str, Any], ...]
    uncertain_scopes: tuple[str, ...]
    seed_digest: str
    schema: str
    _token: object

    def __post_init__(self) -> None:
        if self.schema != SEED_SCHEMA or self._token is not _TOKEN:
            raise NativeRetentionCatalogRefused("catalog seed lacks native construction")
        _text(self.campaign_id, "campaign_id")
        _sha(self.config_digest, "config_digest")
        _sha(self.manifest_digest, "manifest_digest")
        if not self.artifacts or len({item.artifact_id for item in self.artifacts}) \
                != len(self.artifacts):
            raise NativeRetentionCatalogRefused("catalog artifacts are empty or duplicated")
        try:
            inventories = tuple(_freeze(
                model_preparation.ScheduledModelPreparation.from_dict(
                    item.to_dict() if isinstance(
                        item, model_preparation.ScheduledModelPreparation) else item
                ).to_dict()) for item in self.model_inventories)
        except (TypeError, model_preparation.ModelPreparationRefused) as exc:
            raise NativeRetentionCatalogRefused(
                "catalog model inventory is not an owning typed record") from exc
        uncertain = tuple(sorted(_text(item, "uncertain scope")
                                 for item in self.uncertain_scopes))
        object.__setattr__(self, "model_inventories", inventories)
        object.__setattr__(self, "uncertain_scopes", uncertain)
        _sha(self.seed_digest, "seed_digest")
        if self.seed_digest != _digest(self.body()):
            raise NativeRetentionCatalogRefused("catalog seed digest differs")

    def body(self) -> dict[str, Any]:
        return {"schema": self.schema, "campaign_id": self.campaign_id,
                "config_digest": self.config_digest,
                "manifest_digest": self.manifest_digest,
                "artifacts": [item.to_dict() for item in self.artifacts],
                "model_inventories": [_plain(item) for item in self.model_inventories],
                "uncertain_scopes": list(self.uncertain_scopes)}

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "seed_digest": self.seed_digest}


def _artifact_id(role: str, path: str, sha256: str) -> str:
    return f"native:{role}:{_digest({'path': path, 'sha256': sha256})}"


def build_seed(resolved: campaign.ResolvedCampaign,
               anchors: unified_planner.PreparedRuntimeAnchors,
               *, config_digest: str,
               model_preparations: Mapping[str, Any] | None = None,
               runtime_recipes: Mapping[str, Mapping[str, Any]] | None = None,
               runtime_recipe_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
               artifact_root: Path | None = None) -> NativeRetentionCatalogSeed:
    """Derive static declarations from already validated runtime inputs."""
    if not isinstance(resolved, campaign.ResolvedCampaign):
        raise NativeRetentionCatalogRefused("catalog requires a resolved campaign")
    if (not isinstance(anchors, unified_planner.PreparedRuntimeAnchors)
            or anchors.campaign_id != resolved.campaign_id
            or anchors.manifest_digest != resolved.manifest_digest):
        raise NativeRetentionCatalogRefused("catalog runtime anchors are stale or foreign")
    _sha(config_digest, "config_digest")
    rows: dict[str, CatalogArtifact] = {}

    def add(role: str, item: Any, kind: str, roots: Sequence[str], deps=(), **provenance):
        if item is None:
            return None
        path, digest = str(item.path), item.sha256
        identity = _artifact_id("bytes", path, digest)
        owner = {"role": role, **provenance}
        row = CatalogArtifact(identity, kind, path, digest, "file_sha256", tuple(deps),
                              "permanent_large", tuple(roots),
                              {"owners": [owner]})
        prior = rows.get(identity)
        if prior is None:
            rows[identity] = row
        else:
            owners = [*_plain(prior.provenance)["owners"], owner]
            rows[identity] = CatalogArtifact(
                identity, prior.artifact_kind, path, digest, "file_sha256",
                tuple(sorted(set(prior.dependencies) | set(row.dependencies))),
                prior.retention_class,
                tuple(sorted(set(prior.root_kinds) | set(row.root_kinds))),
                {"owners": owners})
        return identity

    def add_directory(role: str, path_value: str | None, roots: Sequence[str], deps=(),
                      **provenance):
        if path_value is None:
            return None
        path = str(Path(path_value))
        identity = "native:directory:" + _digest({"path": path})
        row = CatalogArtifact(
            identity, "shared_dso_dir" if role == "runpath" else "build_dir",
            path, None, "directory_pending_measurement", tuple(deps),
            "permanent_large", tuple(roots), {"owners": [{"role": role, **provenance}]})
        prior = rows.get(identity)
        if prior is None:
            rows[identity] = row
        else:
            rows[identity] = CatalogArtifact(
                identity, prior.artifact_kind, path, None,
                "directory_pending_measurement",
                tuple(sorted(set(prior.dependencies) | set(row.dependencies))),
                prior.retention_class,
                tuple(sorted(set(prior.root_kinds) | set(row.root_kinds))),
                {"owners": [*_plain(prior.provenance)["owners"],
                            *_plain(row.provenance)["owners"]]})
        return identity

    recipes_by_target: dict[str, dict[str, rr.CanonicalResolvedRecipe]] = {}
    for target_digest, anchor in anchors.recipes.items():
        recipes_by_target[target_digest] = {anchor.snapshot_digest: anchor}
    for target_digest, supplied in (runtime_recipes or {}).items():
        if target_digest not in recipes_by_target or not isinstance(supplied, Mapping):
            raise NativeRetentionCatalogRefused("runtime recipes name an unknown target")
        for execution_digest, value in supplied.items():
            recipe = rr.CanonicalResolvedRecipe.from_dict(
                value.to_dict() if isinstance(value, rr.CanonicalResolvedRecipe) else value)
            if execution_digest != recipe.execution_digest:
                raise NativeRetentionCatalogRefused("runtime recipe key differs")
            prior = recipes_by_target[target_digest].get(recipe.snapshot_digest)
            if prior is not None and prior.to_dict() != recipe.to_dict():
                raise NativeRetentionCatalogRefused("runtime recipe identity conflicts")
            recipes_by_target[target_digest][recipe.snapshot_digest] = recipe
    for target_digest, supplied in (runtime_recipe_snapshots or {}).items():
        if target_digest not in recipes_by_target or not isinstance(supplied, Mapping):
            raise NativeRetentionCatalogRefused(
                "runtime recipe snapshots name an unknown target")
        for snapshot_digest, value in supplied.items():
            recipe = rr.CanonicalResolvedRecipe.from_dict(
                value.to_dict() if isinstance(value, rr.CanonicalResolvedRecipe) else value)
            if snapshot_digest != recipe.snapshot_digest:
                raise NativeRetentionCatalogRefused("runtime recipe snapshot key differs")
            prior = recipes_by_target[target_digest].get(snapshot_digest)
            if prior is not None and prior.to_dict() != recipe.to_dict():
                raise NativeRetentionCatalogRefused("runtime recipe snapshot identity conflicts")
            recipes_by_target[target_digest][snapshot_digest] = recipe
    for target in resolved.targets:
        target_digest = _digest(target.to_dict())
        recipes = recipes_by_target.get(target_digest, {})
        if target.status != "ready" or not recipes:
            continue
        target_roots = ("production",) if "production" in target.enrolled_as else ()
        for recipe in recipes.values():
            owner = {"target_revision_digest": target_digest,
                     "recipe_execution_digest": recipe.execution_digest,
                     "recipe_snapshot_digest": recipe.snapshot_digest}
            model = add("model", recipe.model, "candidate_artifact", target_roots, **owner)
            drafter = add("drafter", recipe.drafter, "candidate_artifact", target_roots,
                          **owner)
            dso_ids = tuple(add("dso", item, "candidate_artifact", target_roots, **owner)
                            for item in recipe.dsos)
            executable = add(
                "executable", recipe.executable, "candidate_artifact", target_roots,
                deps=tuple(x for x in (model, drafter, *dso_ids) if x), **owner)
            add("recipe", target.execution.recipe, "runtime_recipe", target_roots,
                deps=tuple(x for x in (executable, model, drafter, *dso_ids) if x),
                build_dir=recipe.build_dir, runtime_binary_dir=recipe.runtime_binary_dir,
                runtime_ld_paths=list(recipe.runtime_ld_paths), **owner)
            add_directory("build", recipe.build_dir, target_roots, deps=(executable,), **owner)
            for directory in recipe.runtime_ld_paths:
                add_directory("runpath", directory, target_roots, deps=dso_ids, **owner)
        add("baseline", target.baseline, "candidate_artifact", ("rollback",),
            target_revision_digest=target_digest)
    if artifact_root is not None:
        root = str(Path(artifact_root).absolute())
        # A content digest does not exist before the store publishes content.  The root
        # is therefore a dependency declaration only, never a path artifact or expiry.
        identity = "native:artifact-store:" + _digest({"path": root})
        rows[identity] = CatalogArtifact(
            identity, "evidence_record", None, None, "directory_pending_measurement", (), "permanent_large",
            ("retained_evidence",), {"role": "native_artifact_store", "path": root})
    inventories: list[dict[str, Any]] = []
    for target, recipes in sorted((model_preparations or {}).items()):
        if not isinstance(recipes, Mapping):
            raise NativeRetentionCatalogRefused("model preparations must be target mappings")
        for recipe_digest, item in sorted(recipes.items()):
            try:
                prepared = (item if isinstance(item, model_preparation.ScheduledModelPreparation)
                            else model_preparation.ScheduledModelPreparation.from_dict(item))
            except (TypeError, model_preparation.ModelPreparationRefused) as exc:
                raise NativeRetentionCatalogRefused(
                    "model preparation is not an owning typed record") from exc
            row = prepared.to_dict()
            if (prepared.target_revision_digest != target
                    or prepared.recipe_execution_digest != recipe_digest):
                raise NativeRetentionCatalogRefused("model inventory binding differs")
            matching = tuple(recipe for recipe in recipes_by_target.get(target, {}).values()
                             if recipe.execution_digest == recipe_digest)
            if (not matching or any(prepared.entry_path != recipe.model.path
                                    or prepared.entry_sha256 != recipe.model.sha256
                                    for recipe in matching)):
                raise NativeRetentionCatalogRefused(
                    "model inventory entry differs from the enrolled recipe")
            inventories.append(_plain(row))
    uncertain = []
    if any(value is None for _name, value in resolved.source_snapshot):
        uncertain.append("source_snapshot_incomplete")
    for target in resolved.targets:
        target_digest = _digest(target.to_dict())
        recipes = recipes_by_target.get(target_digest, {})
        if target.status != "ready":
            uncertain.append(f"target:{target_digest}:recipe_unready:{target.status}")
        if target.status == "ready" and not recipes:
            uncertain.append(f"target:{target_digest}:runtime_recipe_missing")
        inventory_keys = {(item["target_revision_digest"], item["recipe_execution_digest"])
                          for item in inventories}
        for execution_digest in sorted({item.execution_digest for item in recipes.values()}):
            if (target_digest, execution_digest) not in inventory_keys:
                uncertain.append(
                    f"target:{target_digest}:recipe:{execution_digest}:model_inventory_missing")
            else:
                uncertain.append(
                    f"target:{target_digest}:recipe:{execution_digest}:model_inventory_unverified")
    expected_config = _digest(resolved.to_dict())
    if config_digest != expected_config:
        raise NativeRetentionCatalogRefused("catalog config digest differs from resolved campaign")
    body = {"schema": SEED_SCHEMA, "campaign_id": resolved.campaign_id,
            "config_digest": config_digest, "manifest_digest": resolved.manifest_digest,
            "artifacts": [item.to_dict() for item in sorted(
                rows.values(), key=lambda item: item.artifact_id)],
            "model_inventories": inventories,
            "uncertain_scopes": sorted(uncertain)}
    return NativeRetentionCatalogSeed(
        resolved.campaign_id, config_digest, resolved.manifest_digest,
        tuple(CatalogArtifact.from_dict(item) for item in body["artifacts"]),
        tuple(inventories), tuple(sorted(uncertain)), _digest(body), SEED_SCHEMA, _TOKEN)


def make_install_event(seed: NativeRetentionCatalogSeed, *, config_generation: int,
                       supervisor_incarnation: int) -> dict[str, Any]:
    if not isinstance(seed, NativeRetentionCatalogSeed):
        raise TypeError("seed must be NativeRetentionCatalogSeed")
    if (type(config_generation) is not int or config_generation < 1
            or type(supervisor_incarnation) is not int or supervisor_incarnation < 1):
        raise NativeRetentionCatalogRefused("catalog generation/incarnation is invalid")
    return {"schema": EVENT_SCHEMA, "event": EVENT,
            "campaign_id": seed.campaign_id, "config_digest": seed.config_digest,
            "config_generation": config_generation,
            "supervisor_incarnation": supervisor_incarnation,
            "seed": seed.to_dict(), "seed_digest": seed.seed_digest}


def validate_install_event(value: Any) -> Mapping[str, Any]:
    fields = {"schema", "event", "campaign_id", "config_digest", "config_generation",
              "supervisor_incarnation", "seed", "seed_digest"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise NativeRetentionCatalogRefused("catalog event fields differ")
    if value["schema"] != EVENT_SCHEMA or value["event"] != EVENT:
        raise NativeRetentionCatalogRefused("catalog event schema/type is unsupported")
    if not isinstance(value["seed"], Mapping) or set(value["seed"]) != {
            "schema", "campaign_id", "config_digest", "manifest_digest", "artifacts",
            "model_inventories", "uncertain_scopes", "seed_digest"}:
        raise NativeRetentionCatalogRefused("catalog seed fields differ")
    seed_row = dict(value["seed"])
    supplied = seed_row.pop("seed_digest", None)
    seed = NativeRetentionCatalogSeed(
        seed_row["campaign_id"], seed_row["config_digest"], seed_row["manifest_digest"],
        tuple(CatalogArtifact.from_dict(item) for item in seed_row["artifacts"]),
        tuple(seed_row["model_inventories"]), tuple(seed_row["uncertain_scopes"]),
        supplied, seed_row["schema"], _TOKEN)
    if (value["campaign_id"] != seed.campaign_id
            or value["config_digest"] != seed.config_digest
            or value["seed_digest"] != seed.seed_digest
            or type(value["config_generation"]) is not int
            or value["config_generation"] < 1
            or type(value["supervisor_incarnation"]) is not int
            or value["supervisor_incarnation"] < 1):
        raise NativeRetentionCatalogRefused("catalog event binding differs")
    return _freeze(_plain(value))


def build_native_view(capture: Mapping[str, Any]) -> PreparedCatalogView:
    """Build one exact view outside controller locks from a captured owner frontier."""
    from . import retention_consumer as consumer
    if not isinstance(capture, Mapping) or set(capture) != {
            "seed", "frontier", "candidate_state", "candidate_records",
            "active_workers", "active_acquisitions", "native_records",
            "driver_issued", "driver_settled"}:
        raise NativeRetentionCatalogRefused("catalog capture fields differ")
    seed = capture["seed"]
    if not isinstance(seed, NativeRetentionCatalogSeed):
        raise NativeRetentionCatalogRefused("catalog capture lacks its native seed")
    state = cm.CandidateState.from_dict(capture["candidate_state"])
    manifests: dict[str, cm.CandidateManifest] = {}
    for completed in capture["candidate_records"].values():
        intent = completed.get("intent", {})
        operation = intent.get("operation")
        payload = intent.get("data", {}).get("operation_payload", {})
        keys = {"init": ("manifest",), "integrate": ("previous", "candidate"),
                "start_batch": ("candidate", "comparator"),
                "advance_validated": ("candidate", "comparator")}.get(operation, ())
        for key in keys:
            if key in payload:
                manifest = cm.CandidateManifest.from_dict(payload[key])
                manifests[manifest.manifest_digest] = manifest
    if not manifests:
        raise NativeRetentionCatalogRefused("catalog lacks durable candidate manifests")
    if not any(item.sources for item in manifests.values()):
        raise NativeRetentionCatalogRefused("catalog lacks candidate source provenance")
    nodes: dict[str, retention.ArtifactNode] = {}
    identities: dict[str, consumer.NativeArtifactIdentity] = {}
    roots: dict[str, set[str]] = {name: set() for name in consumer.NativeRoots.__dataclass_fields__}
    root_field = {"production": "production", "rollback": "rollback",
                  "pending_calibration": "pending_calibration",
                  "active_worker": "active_workers", "launch_intent": "launch_intents",
                  "integration_intent": "integration_intents",
                  "retained_evidence": "retained_evidence"}

    def insert(item: CatalogArtifact) -> None:
        digest = item.sha256
        if item.path is not None and item.identity_kind == "directory_pending_measurement":
            digest = storage.hash_tree_manifest(item.path)
        nodes[item.artifact_id] = retention.ArtifactNode(
            item.artifact_id, item.artifact_kind, item.dependencies, item.path,
            item.retention_class, None, None)
        if item.path is not None:
            assert digest is not None
            identities[item.artifact_id] = consumer.NativeArtifactIdentity(
                item.artifact_id, item.path, digest, None, None)
        for kind in item.root_kinds:
            field = root_field.get(kind)
            if field is not None:
                roots[field].add(item.artifact_id)

    for item in seed.artifacts:
        insert(item)
    # Expand the complete bounded model manifest into every declared shard.  This
    # reads only the small inventory, never model bytes.
    for spec in seed.model_inventories:
        identity = spec["inventory_identity"]
        manifest_path = Path(identity["model_manifest"])
        raw = _bounded_regular_bytes(
            manifest_path, MAX_INVENTORY_BYTES, "model inventory")
        if hashlib.sha256(raw).hexdigest() != identity["model_manifest_sha256"]:
            raise NativeRetentionCatalogRefused("model inventory bytes differ")
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise NativeRetentionCatalogRefused("model inventory is not JSON") from exc
        if (not isinstance(manifest, Mapping)
                or set(manifest) != {"schema", "model_path", "files"}
                or manifest["schema"] != "epyc.autokernel.model_identity.v1"
                or not isinstance(manifest["files"], list) or not manifest["files"]):
            raise NativeRetentionCatalogRefused("model inventory schema differs")
        model_root = Path(identity["model_id"])
        if manifest["model_path"] != str(model_root):
            raise NativeRetentionCatalogRefused("model inventory names a different model")
        declared: set[str] = set()
        entry_seen = False
        for member in manifest["files"]:
            if not isinstance(member, Mapping) or set(member) != {"path", "sha256"}:
                raise NativeRetentionCatalogRefused("model inventory member fields differ")
            if not isinstance(member["path"], str) or not member["path"]:
                raise NativeRetentionCatalogRefused("model inventory member path is malformed")
            relative = Path(member["path"])
            if (str(relative) in declared or relative.is_absolute()
                    or ".." in relative.parts):
                raise NativeRetentionCatalogRefused("model inventory member escapes root")
            declared.add(str(relative))
            path = model_root if str(relative) == "." else model_root / relative
            if str(path) == spec["entry_path"]:
                if member["sha256"] != spec["entry_sha256"]:
                    raise NativeRetentionCatalogRefused(
                        "model inventory entry digest differs")
                entry_seen = True
            roots_for_target = tuple(sorted({root for artifact in seed.artifacts
                for owner in artifact.provenance.get("owners", ())
                if owner.get("target_revision_digest") == spec["target_revision_digest"]
                for root in artifact.root_kinds}))
            item = CatalogArtifact(
                _artifact_id("model-shard", str(path), _sha(member["sha256"], "model shard")),
                "candidate_artifact", str(path), member["sha256"], "file_sha256", (),
                "permanent_large", roots_for_target,
                {"owners": [{"role": "model_inventory_member",
                             "target_revision_digest": spec["target_revision_digest"]}]})
            insert(item)
        if not entry_seen:
            raise NativeRetentionCatalogRefused(
                "model inventory omits the enrolled recipe entry")
    by_key = {(item.path, item.sha256): item.artifact_id for item in seed.artifacts}
    bindings = []
    for manifest in manifests.values():
        artifact_ids = []
        for build in manifest.builds:
            for artifact in (build.executable, *build.dsos):
                artifact_id = by_key.get((artifact.path, artifact.sha256))
                if artifact_id is None:
                    artifact_id = _artifact_id("candidate-build", artifact.path, artifact.sha256)
                    insert(CatalogArtifact(
                        artifact_id, "candidate_artifact", artifact.path, artifact.sha256,
                        "file_sha256", (), "permanent_large", (),
                        {"owners": [{"role": artifact.role,
                                     "manifest_digest": manifest.manifest_digest}]}))
                artifact_ids.append(artifact_id)
        if not artifact_ids:
            raise NativeRetentionCatalogRefused("candidate manifest has no catalog artifacts")
        bindings.append(consumer.ManifestArtifacts(manifest, tuple(sorted(set(artifact_ids)))))
    # Durable native capture artifacts are retained evidence roots.
    artifact_store = next((item.provenance.get("path") for item in seed.artifacts
                           if item.provenance.get("role") == "native_artifact_store"), None)
    for measurement_id, entry in capture["native_records"].items():
        artifact = entry.payload["artifact"] if hasattr(entry, "payload") else entry["payload"]["artifact"]
        artifact_id = f"native:evidence:{measurement_id}"
        path = None if artifact_store is None else str(Path(artifact_store) / artifact["locator"])
        insert(CatalogArtifact(
            artifact_id, "evidence_record", path, artifact["sha256"], "file_sha256", (),
            "permanent_large", ("retained_evidence",),
            {"owners": [{"role": "native_capture", "measurement_id": measurement_id}]}))
    active_driver_lineages: set[str] = set()
    for issued in capture["driver_issued"].values():
        transition_id = issued["transition_id"]
        if transition_id in capture["driver_settled"]:
            continue
        active_driver_lineages.add(f"driver:{transition_id}")
        from . import scheduling
        selection = scheduling.Selection.from_dict(issued["selection"])
        if selection.proposal is None:
            raise NativeRetentionCatalogRefused("issued selection lacks its proposal")
        selected = selection.proposal
        stage_digest = selected.digest
        work_by_stage = issued["catalog"].get("work_by_stage_digest", {})
        work = work_by_stage.get(stage_digest) if isinstance(work_by_stage, Mapping) else None
        if not isinstance(work, Mapping):
            raise NativeRetentionCatalogRefused("issued catalog lacks selected work")
        target_digest = selected.target_revision
        recipe_digests: set[str] = set()
        if work.get("kind") == "runtime_comparison":
            proposal_row = work.get("payload", {}).get("proposal", {})
            proposal = unified_planner.UnifiedProposal.from_dict(proposal_row)
            if proposal.runtime_pair is None:
                raise NativeRetentionCatalogRefused("issued runtime work lacks recipes")
            pair = unified_planner.RuntimeArmPair.from_dict(_plain(proposal.runtime_pair))
            recipe_digests = {pair.anchor.execution_digest, pair.candidate.execution_digest}
        matched = []
        for item in seed.artifacts:
            owners = item.provenance.get("owners", ())
            if any(owner.get("target_revision_digest") == target_digest
                   and (not recipe_digests
                        or owner.get("recipe_execution_digest") in recipe_digests)
                   for owner in owners):
                matched.append(item.artifact_id)
        if not matched:
            raise NativeRetentionCatalogRefused(
                "issued work has no exact catalog dependency join")
        roots["launch_intents"].update(matched)
    verified_targets = {item["target_revision_digest"] for item in seed.model_inventories}
    uncertain = [item for item in seed.uncertain_scopes
                 if not (item.endswith(":model_inventory_unverified")
                         and item.split(":", 2)[1] in verified_targets)]
    if capture["native_records"] and artifact_store is None:
        uncertain.append("native_capture_artifact_root_missing")
    active_lineages = {
        row.get("lineage_id") or row.get("data", {}).get("lineage_id")
        for row in (*capture["active_workers"], *capture["active_acquisitions"])
        if isinstance(row, Mapping)
    }
    unmatched = sorted(str(item) for item in active_lineages
                       if item is not None and item not in active_driver_lineages)
    if unmatched or ((capture["active_workers"] or capture["active_acquisitions"])
                     and not active_lineages):
        uncertain.append("active_worker_plan_dependency_join_unavailable")
    view = consumer.NativeRetentionView(
        seed.seed_digest, max(1, int(capture["frontier"][2]) + 1), state,
        tuple(sorted(bindings, key=lambda item: item.manifest.manifest_digest)),
        tuple(sorted(nodes.values(), key=lambda item: item.artifact_id)),
        consumer.NativeRoots(
            tuple(sorted(roots["production"])), tuple(sorted(roots["rollback"])),
            tuple(sorted(roots["pending_calibration"])),
            tuple(sorted(roots["active_workers"])), tuple(sorted(roots["launch_intents"])),
            (), tuple(sorted(roots["integration_intents"])),
            tuple(sorted(roots["retained_evidence"]))),
        tuple(sorted(identities.values(), key=lambda item: item.artifact_id)),
        tuple(sorted(set(uncertain))))
    digest = consumer.collect_native_snapshot(view).snapshot_digest
    return PreparedCatalogView(view, tuple(capture["frontier"]), digest, _VIEW_TOKEN)


__all__ = ["CatalogArtifact", "EVENT_SCHEMA", "NativeRetentionCatalogRefused",
           "PreparedCatalogView",
           "NativeRetentionCatalogSeed", "SEED_SCHEMA", "build_seed",
           "build_native_view", "make_install_event", "validate_install_event"]
