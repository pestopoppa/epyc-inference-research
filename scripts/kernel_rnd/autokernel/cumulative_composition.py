"""Fail-closed authority for cumulative AutoKernel source composition.

This module deliberately owns no compiler, GPU, or controller side effects.  It
defines the immutable records that a later controller integration must satisfy:
an accepted ordered patch stack is the anchor, appending one independently
replicated lever is the candidate, and admission is possible only after a new
full-correctness receipt plus both incremental route and target-runtime results.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import source_candidate
from .controller import gpu_source_proofs


__all__ = (
    "BuildBinding", "CompositionAuthority", "CompositionError",
    "CompositionLedger", "CompositionPlan", "CumulativeBuildPair",
    "DnrAuthority", "FullCorrectness", "IncrementalComparison",
    "IsolatedReplication", "ReplicatedPositiveLever",
)


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
HUNK = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)


class CompositionError(RuntimeError):
    """A composition authority or transition failed closed."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CompositionError("composition value is not canonical JSON") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise CompositionError(f"{label} must be a SHA-256 digest")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not COMMIT.fullmatch(value):
        raise CompositionError(f"{label} must be a Git commit")
    return value


def _require_text(value: Any, label: str, *, prefix: str | None = None) -> str:
    if (not isinstance(value, str) or not value or "\x00" in value
            or (prefix is not None and not value.startswith(prefix))):
        raise CompositionError(f"{label} is invalid")
    return value


def _finite(value: Any, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise CompositionError(f"{label} must be finite")
    return float(value)


def _manifest_dict(manifest: source_candidate.SourcePatchManifest) -> dict[str, Any]:
    if not isinstance(manifest, source_candidate.SourcePatchManifest):
        raise CompositionError("lever source must be a typed source manifest")
    try:
        value = json.loads(
            source_candidate.source_patch_manifest_bytes(manifest).decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompositionError("typed source manifest is not canonical JSON") from exc
    return value


def _manifest_from_dict(value: Mapping[str, Any]) -> source_candidate.SourcePatchManifest:
    required = {
        "schema", "campaign_id", "proposal_id", "candidate_id", "source_tree",
        "production_base_commit", "instrument_commit", "change_class",
        "declared_files", "declared_symbols", "mechanism_id", "patch_sha256",
        "patch_encoding", "patch_base64",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise CompositionError("source manifest projection has an inexact schema")
    if (value.get("schema") != source_candidate.SCHEMA_SOURCE_PATCH
            or value.get("patch_encoding") != "base64"):
        raise CompositionError("source manifest projection has an unsupported carrier")
    try:
        patch = base64.b64decode(value["patch_base64"], validate=True)
        symbols = value["declared_symbols"]
        if not isinstance(symbols, Mapping):
            raise TypeError("declared symbols are not a mapping")
        return source_candidate.SourcePatchManifest(
            campaign_id=value["campaign_id"], proposal_id=value["proposal_id"],
            candidate_id=value["candidate_id"], source_tree=value["source_tree"],
            production_base_commit=value["production_base_commit"],
            instrument_commit=value["instrument_commit"],
            change_class=value["change_class"],
            declared_files=tuple(value["declared_files"]),
            declared_symbols={key: tuple(rows) for key, rows in symbols.items()},
            mechanism_id=value["mechanism_id"], patch_sha256=value["patch_sha256"],
            patch_bytes=patch,
        )
    except (KeyError, TypeError, ValueError, source_candidate.SourceCandidateError) as exc:
        raise CompositionError("source manifest projection is invalid") from exc


@dataclass(frozen=True)
class IsolatedReplication:
    result_sha256: str
    series_key: str
    build_identity_sha256: str
    correctness_receipt_sha256: str
    attribution_receipt_sha256: str
    graphs_off_receipt_sha256: str
    graphs_on_receipt_sha256: str
    effect_fraction: float

    def __post_init__(self) -> None:
        for label in (
            "result_sha256", "series_key", "build_identity_sha256",
            "correctness_receipt_sha256", "attribution_receipt_sha256",
            "graphs_off_receipt_sha256",
            "graphs_on_receipt_sha256",
        ):
            _require_sha(getattr(self, label), label)
        if _finite(self.effect_fraction, "isolated effect") <= 0:
            raise CompositionError("replicated isolated lever must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IsolatedReplication":
        if not isinstance(value, Mapping) or set(value) != {
            "result_sha256", "series_key", "build_identity_sha256",
            "correctness_receipt_sha256", "attribution_receipt_sha256",
            "graphs_off_receipt_sha256",
            "graphs_on_receipt_sha256", "effect_fraction",
        }:
            raise CompositionError("isolated replication has an inexact schema")
        return cls(**dict(value))


@dataclass(frozen=True)
class ReplicatedPositiveLever:
    hypothesis_id: str
    cross_campaign_candidate_sha256: str
    manifest: source_candidate.SourcePatchManifest
    replications: tuple[IsolatedReplication, ...]

    def __post_init__(self) -> None:
        _require_text(self.hypothesis_id, "hypothesis_id", prefix="akh-")
        _require_sha(
            self.cross_campaign_candidate_sha256,
            "cross_campaign_candidate_sha256",
        )
        _manifest_dict(self.manifest)
        if not isinstance(self.replications, tuple) or len(self.replications) < 2:
            raise CompositionError("lever needs at least two isolated replications")
        if any(not isinstance(row, IsolatedReplication) for row in self.replications):
            raise CompositionError("lever replications must be typed")
        series = {row.series_key for row in self.replications}
        results = {row.result_sha256 for row in self.replications}
        builds = {row.build_identity_sha256 for row in self.replications}
        if len(series) != 1:
            raise CompositionError("isolated replications are not one exact series")
        if len(results) != len(self.replications):
            raise CompositionError("isolated replication result was reused")
        # S2 is an independent measurement of the exact same built source, not
        # permission to rebuild different bytes under one scientific series.
        if len(builds) != 1:
            raise CompositionError("isolated replications changed build identity")

    @property
    def manifest_sha256(self) -> str:
        return self.manifest.patch_bundle_sha256

    @property
    def lever_sha256(self) -> str:
        return _sha(self._body())

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.replicated_positive_lever.v2",
            "hypothesis_id": self.hypothesis_id,
            "cross_campaign_candidate_sha256":
                self.cross_campaign_candidate_sha256,
            "manifest": _manifest_dict(self.manifest),
            "manifest_sha256": self.manifest_sha256,
            "isolated_disposition": "top_k_replicated_candidate",
            "replications": [row.to_dict() for row in self.replications],
        }

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        body["lever_sha256"] = self.lever_sha256
        return body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplicatedPositiveLever":
        required = {
            "schema", "hypothesis_id", "cross_campaign_candidate_sha256",
            "manifest", "manifest_sha256", "isolated_disposition",
            "replications", "lever_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("replicated lever has an inexact schema")
        if (value.get("schema") != "epyc.autokernel.replicated_positive_lever.v2"
                or value.get("isolated_disposition") !=
                   "top_k_replicated_candidate"):
            raise CompositionError("replicated lever type/disposition changed")
        rows = value.get("replications")
        if not isinstance(rows, list):
            raise CompositionError("replicated lever rows must be a list")
        lever = cls(
            hypothesis_id=value["hypothesis_id"],
            cross_campaign_candidate_sha256=
                value["cross_campaign_candidate_sha256"],
            manifest=_manifest_from_dict(value["manifest"]),
            replications=tuple(IsolatedReplication.from_dict(row) for row in rows),
        )
        if (value.get("manifest_sha256") != lever.manifest_sha256
                or value.get("lever_sha256") != lever.lever_sha256):
            raise CompositionError("replicated lever identity changed")
        return lever


def _edit_footprint(
        manifest: source_candidate.SourcePatchManifest,
) -> dict[str, tuple[frozenset[int], frozenset[int]]]:
    """Return exact deleted-line positions and insertion anchors by old file."""
    path: str | None = None
    old_line: int | None = None
    deleted: dict[str, set[int]] = {}
    inserted: dict[str, set[int]] = {}
    for line in manifest.patch_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            deleted.setdefault(path, set())
            inserted.setdefault(path, set())
            continue
        match = HUNK.match(line)
        if match:
            if path is None:
                raise CompositionError("patch hunk precedes its path")
            old_line = int(match.group(1))
            continue
        if old_line is None or path is None:
            continue
        if line.startswith("-") and not line.startswith("---"):
            deleted[path].add(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            inserted[path].add(old_line)
        elif line.startswith(" "):
            old_line += 1
        elif line == "\\ No newline at end of file":
            continue
    return {
        key: (frozenset(deleted[key]), frozenset(inserted[key]))
        for key in sorted(deleted)
    }


def _require_compatible(
        accepted: Sequence[ReplicatedPositiveLever],
        proposed: ReplicatedPositiveLever,
) -> None:
    proposed_footprint = _edit_footprint(proposed.manifest)
    for existing in accepted:
        if existing.cross_campaign_candidate_sha256 == \
                proposed.cross_campaign_candidate_sha256:
            raise CompositionError("cross-campaign candidate was already considered")
        if existing.manifest_sha256 == proposed.manifest_sha256:
            raise CompositionError("source manifest was already accepted")
        shared_files = set(existing.manifest.declared_files) & set(
            proposed.manifest.declared_files)
        footprint = _edit_footprint(existing.manifest)
        for path in sorted(shared_files):
            old_symbols = set(existing.manifest.declared_symbols[path])
            new_symbols = set(proposed.manifest.declared_symbols[path])
            if (source_candidate.FILE_SCOPE in old_symbols
                    or source_candidate.FILE_SCOPE in new_symbols
                    or old_symbols & new_symbols):
                raise CompositionError(
                    f"composition patches conflict in declared scope {path}")
            old_deleted, old_inserted = footprint[path]
            new_deleted, new_inserted = proposed_footprint[path]
            if (old_deleted & new_deleted or old_deleted & new_inserted
                    or old_inserted & new_deleted
                    or old_inserted & new_inserted):
                raise CompositionError(
                    f"composition patches overlap old coordinates in {path}")


@dataclass(frozen=True)
class CompositionAuthority:
    campaign_id: str
    production_base_commit: str
    instrument_commit: str
    accepted: tuple[ReplicatedPositiveLever, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.campaign_id, "campaign_id", prefix="ak-")
        _require_commit(self.production_base_commit, "production_base_commit")
        _require_commit(self.instrument_commit, "instrument_commit")
        if not isinstance(self.accepted, tuple):
            raise CompositionError("accepted composition must be an ordered tuple")
        prior: list[ReplicatedPositiveLever] = []
        for lever in self.accepted:
            if not isinstance(lever, ReplicatedPositiveLever):
                raise CompositionError("accepted composition contains an untyped lever")
            manifest = lever.manifest
            if (manifest.production_base_commit != self.production_base_commit
                    or manifest.instrument_commit != self.instrument_commit):
                raise CompositionError("lever belongs to another source era")
            _require_compatible(prior, lever)
            prior.append(lever)

    @property
    def ordered_patch_set_sha256(self) -> str:
        return _sha({
            "schema": "epyc.autokernel.ordered_patch_set.v1",
            "campaign_id": self.campaign_id,
            "production_base_commit": self.production_base_commit,
            "instrument_commit": self.instrument_commit,
            "lever_sha256s": [lever.lever_sha256 for lever in self.accepted],
            "source_manifest_sha256s": [
                lever.manifest_sha256 for lever in self.accepted
            ],
        })

    @property
    def authority_sha256(self) -> str:
        return _sha(self._body())

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.cumulative_composition_authority.v1",
            "campaign_id": self.campaign_id,
            "production_base_commit": self.production_base_commit,
            "instrument_commit": self.instrument_commit,
            "ordered_patch_set_sha256": self.ordered_patch_set_sha256,
            "accepted": [lever.to_dict() for lever in self.accepted],
        }

    def to_dict(self) -> dict[str, Any]:
        body = self._body()
        body["authority_sha256"] = self.authority_sha256
        return body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompositionAuthority":
        required = {
            "schema", "campaign_id", "production_base_commit",
            "instrument_commit", "ordered_patch_set_sha256", "accepted",
            "authority_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("composition authority has an inexact schema")
        if value.get("schema") != "epyc.autokernel.cumulative_composition_authority.v1":
            raise CompositionError("composition authority schema changed")
        accepted = value.get("accepted")
        if not isinstance(accepted, list):
            raise CompositionError("composition accepted set must be ordered JSON")
        authority = cls(
            campaign_id=value["campaign_id"],
            production_base_commit=value["production_base_commit"],
            instrument_commit=value["instrument_commit"],
            accepted=tuple(ReplicatedPositiveLever.from_dict(row) for row in accepted),
        )
        if (value.get("ordered_patch_set_sha256") !=
                authority.ordered_patch_set_sha256
                or value.get("authority_sha256") != authority.authority_sha256):
            raise CompositionError("composition authority identity changed")
        return authority

    def append(self, lever: ReplicatedPositiveLever) -> "CompositionAuthority":
        _require_compatible(self.accepted, lever)
        return CompositionAuthority(
            campaign_id=self.campaign_id,
            production_base_commit=self.production_base_commit,
            instrument_commit=self.instrument_commit,
            accepted=self.accepted + (lever,),
        )


@dataclass(frozen=True)
class DnrAuthority:
    campaign_id: str
    anchor_patch_set_sha256: str
    candidate_patch_set_sha256: str
    proposed_cross_campaign_candidate_sha256: str
    registry_sha256: str
    checked_cross_campaign_candidate_sha256s: tuple[str, ...]
    outcome: str
    receipt_sha256: str

    @classmethod
    def pass_for(
            cls, *, anchor: CompositionAuthority,
            candidate: CompositionAuthority, registry_sha256: str,
            checked_cross_campaign_candidate_sha256s: Sequence[str],
    ) -> "DnrAuthority":
        proposed = candidate.accepted[-1].cross_campaign_candidate_sha256
        body = {
            "schema": "epyc.autokernel.composition_dnr.v1",
            "campaign_id": anchor.campaign_id,
            "anchor_patch_set_sha256": anchor.ordered_patch_set_sha256,
            "candidate_patch_set_sha256": candidate.ordered_patch_set_sha256,
            "proposed_cross_campaign_candidate_sha256": proposed,
            "registry_sha256": _require_sha(registry_sha256, "registry_sha256"),
            "checked_cross_campaign_candidate_sha256s": tuple(sorted(
                set(checked_cross_campaign_candidate_sha256s))),
            "outcome": "PASS",
        }
        return cls(**{key: value for key, value in body.items() if key != "schema"},
                   receipt_sha256=_sha(body))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.composition_dnr.v1",
            "campaign_id": self.campaign_id,
            "anchor_patch_set_sha256": self.anchor_patch_set_sha256,
            "candidate_patch_set_sha256": self.candidate_patch_set_sha256,
            "proposed_cross_campaign_candidate_sha256":
                self.proposed_cross_campaign_candidate_sha256,
            "registry_sha256": self.registry_sha256,
            "checked_cross_campaign_candidate_sha256s":
                list(self.checked_cross_campaign_candidate_sha256s),
            "outcome": self.outcome,
        }

    def __post_init__(self) -> None:
        _require_text(self.campaign_id, "campaign_id", prefix="ak-")
        for label in (
            "anchor_patch_set_sha256", "candidate_patch_set_sha256",
            "proposed_cross_campaign_candidate_sha256", "registry_sha256",
            "receipt_sha256",
        ):
            _require_sha(getattr(self, label), label)
        if (not isinstance(self.checked_cross_campaign_candidate_sha256s, tuple)
                or tuple(sorted(set(self.checked_cross_campaign_candidate_sha256s))) !=
                   self.checked_cross_campaign_candidate_sha256s
                or any(not SHA256.fullmatch(value)
                       for value in self.checked_cross_campaign_candidate_sha256s)):
            raise CompositionError("DNR checked registry is not canonical")
        if (self.outcome != "PASS"
                or self.proposed_cross_campaign_candidate_sha256 in
                   self.checked_cross_campaign_candidate_sha256s
                or self.receipt_sha256 != _sha(self._body())):
            raise CompositionError("composition DNR authority is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DnrAuthority":
        required = {
            "schema", "campaign_id", "anchor_patch_set_sha256",
            "candidate_patch_set_sha256",
            "proposed_cross_campaign_candidate_sha256", "registry_sha256",
            "checked_cross_campaign_candidate_sha256s", "outcome",
            "receipt_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("composition DNR has an inexact schema")
        if value.get("schema") != "epyc.autokernel.composition_dnr.v1":
            raise CompositionError("composition DNR schema changed")
        return cls(
            campaign_id=value["campaign_id"],
            anchor_patch_set_sha256=value["anchor_patch_set_sha256"],
            candidate_patch_set_sha256=value["candidate_patch_set_sha256"],
            proposed_cross_campaign_candidate_sha256=
                value["proposed_cross_campaign_candidate_sha256"],
            registry_sha256=value["registry_sha256"],
            checked_cross_campaign_candidate_sha256s=tuple(
                value["checked_cross_campaign_candidate_sha256s"]),
            outcome=value["outcome"], receipt_sha256=value["receipt_sha256"],
        )


@dataclass(frozen=True)
class CompositionPlan:
    attempt_id: str
    operation_key: str
    anchor: CompositionAuthority
    candidate: CompositionAuthority
    dnr: DnrAuthority
    plan_sha256: str

    @classmethod
    def create(
            cls, *, anchor: CompositionAuthority,
            lever: ReplicatedPositiveLever, dnr: DnrAuthority,
            attempt_id: str,
    ) -> "CompositionPlan":
        _require_sha(attempt_id, "attempt_id")
        candidate = anchor.append(lever)
        _validate_dnr(anchor, candidate, dnr)
        body = cls._body_for(attempt_id, anchor, candidate, dnr)
        operation_key = _sha({
            "schema": "epyc.autokernel.composition_operation.v1",
            "attempt_id": attempt_id,
            "plan_body_sha256": _sha(body),
        })
        return cls(attempt_id, operation_key, anchor, candidate, dnr,
                   _sha({**body, "operation_key": operation_key}))

    @staticmethod
    def _body_for(
            attempt_id: str, anchor: CompositionAuthority,
            candidate: CompositionAuthority, dnr: DnrAuthority,
    ) -> dict[str, Any]:
        lever = candidate.accepted[-1]
        return {
            "schema": "epyc.autokernel.cumulative_composition_plan.v1",
            "attempt_id": attempt_id,
            "anchor_authority": anchor.to_dict(),
            "candidate_authority": candidate.to_dict(),
            "anchor_patch_set_sha256": anchor.ordered_patch_set_sha256,
            "candidate_patch_set_sha256": candidate.ordered_patch_set_sha256,
            "ordered_component_lever_sha256s": [
                row.lever_sha256 for row in candidate.accepted
            ],
            "ordered_source_manifest_sha256s": [
                row.manifest_sha256 for row in candidate.accepted
            ],
            "new_lever_sha256": lever.lever_sha256,
            "isolated_result_sha256s": [
                row.result_sha256 for row in lever.replications
            ],
            "dnr": dnr.to_dict(),
        }

    def __post_init__(self) -> None:
        _require_sha(self.attempt_id, "attempt_id")
        _require_sha(self.operation_key, "operation_key")
        _require_sha(self.plan_sha256, "plan_sha256")
        if (len(self.candidate.accepted) != len(self.anchor.accepted) + 1
                or self.candidate.accepted[:-1] != self.anchor.accepted):
            raise CompositionError("candidate is not anchor plus exactly one lever")
        _validate_dnr(self.anchor, self.candidate, self.dnr)
        body = self._body_for(self.attempt_id, self.anchor, self.candidate, self.dnr)
        expected_operation = _sha({
            "schema": "epyc.autokernel.composition_operation.v1",
            "attempt_id": self.attempt_id,
            "plan_body_sha256": _sha(body),
        })
        if (self.operation_key != expected_operation
                or self.plan_sha256 !=
                   _sha({**body, "operation_key": expected_operation})):
            raise CompositionError("composition plan identity changed")

    def to_dict(self) -> dict[str, Any]:
        body = self._body_for(self.attempt_id, self.anchor, self.candidate, self.dnr)
        return {**body, "operation_key": self.operation_key,
                "plan_sha256": self.plan_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompositionPlan":
        required = {
            "schema", "attempt_id", "operation_key", "anchor_authority",
            "candidate_authority", "anchor_patch_set_sha256",
            "candidate_patch_set_sha256", "ordered_component_lever_sha256s",
            "ordered_source_manifest_sha256s", "new_lever_sha256",
            "isolated_result_sha256s", "dnr", "plan_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("composition plan has an inexact schema")
        if value.get("schema") != "epyc.autokernel.cumulative_composition_plan.v1":
            raise CompositionError("composition plan schema changed")
        plan = cls(
            attempt_id=value["attempt_id"], operation_key=value["operation_key"],
            anchor=CompositionAuthority.from_dict(value["anchor_authority"]),
            candidate=CompositionAuthority.from_dict(value["candidate_authority"]),
            dnr=DnrAuthority.from_dict(value["dnr"]),
            plan_sha256=value["plan_sha256"],
        )
        expected = plan.to_dict()
        if dict(value) != expected:
            raise CompositionError("composition plan projection changed")
        return plan


def _validate_dnr(
        anchor: CompositionAuthority, candidate: CompositionAuthority,
        dnr: DnrAuthority,
) -> None:
    proposed = candidate.accepted[-1].cross_campaign_candidate_sha256
    accepted_ids = {
        lever.cross_campaign_candidate_sha256 for lever in anchor.accepted
    }
    if (dnr.campaign_id != anchor.campaign_id
            or dnr.anchor_patch_set_sha256 != anchor.ordered_patch_set_sha256
            or dnr.candidate_patch_set_sha256 != candidate.ordered_patch_set_sha256
            or dnr.proposed_cross_campaign_candidate_sha256 != proposed
            or not accepted_ids.issubset(
                set(dnr.checked_cross_campaign_candidate_sha256s))):
        raise CompositionError("DNR receipt does not bind the cumulative plan")


@dataclass(frozen=True)
class BuildBinding:
    patch_set_sha256: str
    source_materialization_receipt_sha256: str
    build_identity: gpu_source_proofs.BuildIdentity
    build_identity_sha256: str

    @classmethod
    def create(
            cls, patch_set_sha256: str,
            identity: gpu_source_proofs.BuildIdentity, *,
            source_materialization_receipt_sha256: str,
    ) -> "BuildBinding":
        if not isinstance(identity, gpu_source_proofs.BuildIdentity):
            raise CompositionError("composition build identity must be typed")
        return cls(
            patch_set_sha256, source_materialization_receipt_sha256,
            identity, _sha(asdict(identity)))

    def __post_init__(self) -> None:
        _require_sha(self.patch_set_sha256, "patch_set_sha256")
        _require_sha(self.source_materialization_receipt_sha256,
                     "source_materialization_receipt_sha256")
        _require_commit(self.build_identity.source_commit, "build source_commit")
        if (not isinstance(self.build_identity, gpu_source_proofs.BuildIdentity)
                or self.build_identity_sha256 != _sha(asdict(self.build_identity))):
            raise CompositionError("composition build identity changed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_set_sha256": self.patch_set_sha256,
            "source_materialization_receipt_sha256":
                self.source_materialization_receipt_sha256,
            "build_identity": asdict(self.build_identity),
            "build_identity_sha256": self.build_identity_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BuildBinding":
        if not isinstance(value, Mapping) or set(value) != {
            "patch_set_sha256", "source_materialization_receipt_sha256",
            "build_identity", "build_identity_sha256",
        }:
            raise CompositionError("composition build binding has an inexact schema")
        try:
            identity = gpu_source_proofs.BuildIdentity(**value["build_identity"])
        except (TypeError, ValueError, gpu_source_proofs.ProofError) as exc:
            raise CompositionError("composition build identity is invalid") from exc
        return cls(
            value["patch_set_sha256"],
            value["source_materialization_receipt_sha256"], identity,
            value["build_identity_sha256"])


@dataclass(frozen=True)
class CumulativeBuildPair:
    operation_key: str
    plan_sha256: str
    anchor: BuildBinding
    candidate: BuildBinding
    pair_sha256: str

    @classmethod
    def create(
            cls, plan: CompositionPlan, *, anchor: BuildBinding,
            candidate: BuildBinding,
    ) -> "CumulativeBuildPair":
        body = cls._body_for(plan.operation_key, plan.plan_sha256, anchor, candidate)
        return cls(plan.operation_key, plan.plan_sha256, anchor, candidate,
                   _sha(body))

    @staticmethod
    def _body_for(operation_key: str, plan_sha256: str,
                  anchor: BuildBinding, candidate: BuildBinding) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.cumulative_build_pair.v1",
            "operation_key": operation_key, "plan_sha256": plan_sha256,
            "anchor": anchor.to_dict(), "candidate": candidate.to_dict(),
        }

    def __post_init__(self) -> None:
        for label in ("operation_key", "plan_sha256", "pair_sha256"):
            _require_sha(getattr(self, label), label)
        if self.anchor.build_identity_sha256 == self.candidate.build_identity_sha256:
            raise CompositionError("anchor and candidate reused one build identity")
        if self.pair_sha256 != _sha(self._body_for(
                self.operation_key, self.plan_sha256,
                self.anchor, self.candidate)):
            raise CompositionError("cumulative build pair identity changed")

    def bind_plan(self, plan: CompositionPlan) -> None:
        if (self.operation_key != plan.operation_key
                or self.plan_sha256 != plan.plan_sha256
                or self.anchor.patch_set_sha256 !=
                   plan.anchor.ordered_patch_set_sha256
                or self.candidate.patch_set_sha256 !=
                   plan.candidate.ordered_patch_set_sha256):
            raise CompositionError("build pair does not bind cumulative source stacks")

    def to_dict(self) -> dict[str, Any]:
        return {**self._body_for(self.operation_key, self.plan_sha256,
                                self.anchor, self.candidate),
                "pair_sha256": self.pair_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CumulativeBuildPair":
        required = {"schema", "operation_key", "plan_sha256", "anchor",
                    "candidate", "pair_sha256"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("cumulative build pair has an inexact schema")
        if value.get("schema") != "epyc.autokernel.cumulative_build_pair.v1":
            raise CompositionError("cumulative build pair schema changed")
        return cls(
            operation_key=value["operation_key"],
            plan_sha256=value["plan_sha256"],
            anchor=BuildBinding.from_dict(value["anchor"]),
            candidate=BuildBinding.from_dict(value["candidate"]),
            pair_sha256=value["pair_sha256"],
        )


@dataclass(frozen=True)
class FullCorrectness:
    operation_key: str
    build_pair_sha256: str
    candidate_build_identity_sha256: str
    suite_id: str
    cases_sha256: str
    receipt_sha256: str
    passed: bool
    result_sha256: str

    @classmethod
    def create(
            cls, pair: CumulativeBuildPair, *, suite_id: str,
            cases_sha256: str, receipt_sha256: str, passed: bool,
    ) -> "FullCorrectness":
        body = {
            "schema": "epyc.autokernel.composition_full_correctness.v1",
            "operation_key": pair.operation_key,
            "build_pair_sha256": pair.pair_sha256,
            "candidate_build_identity_sha256":
                pair.candidate.build_identity_sha256,
            "suite_id": _require_text(suite_id, "suite_id"),
            "cases_sha256": _require_sha(cases_sha256, "cases_sha256"),
            "receipt_sha256": _require_sha(receipt_sha256, "receipt_sha256"),
            "passed": passed,
            "current_full_suite": True,
        }
        if not isinstance(passed, bool):
            raise CompositionError("correctness result must be boolean")
        return cls(
            operation_key=pair.operation_key,
            build_pair_sha256=pair.pair_sha256,
            candidate_build_identity_sha256=
                pair.candidate.build_identity_sha256,
            suite_id=suite_id, cases_sha256=cases_sha256,
            receipt_sha256=receipt_sha256, passed=passed,
            result_sha256=_sha(body),
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.composition_full_correctness.v1",
            "operation_key": self.operation_key,
            "build_pair_sha256": self.build_pair_sha256,
            "candidate_build_identity_sha256":
                self.candidate_build_identity_sha256,
            "suite_id": self.suite_id, "cases_sha256": self.cases_sha256,
            "receipt_sha256": self.receipt_sha256, "passed": self.passed,
            "current_full_suite": True,
        }

    def __post_init__(self) -> None:
        for label in (
            "operation_key", "build_pair_sha256",
            "candidate_build_identity_sha256", "cases_sha256",
            "receipt_sha256", "result_sha256",
        ):
            _require_sha(getattr(self, label), label)
        _require_text(self.suite_id, "suite_id")
        if not isinstance(self.passed, bool) or self.result_sha256 != _sha(self._body()):
            raise CompositionError("full correctness identity changed")

    def bind_pair(self, pair: CumulativeBuildPair) -> None:
        if (self.operation_key != pair.operation_key
                or self.build_pair_sha256 != pair.pair_sha256
                or self.candidate_build_identity_sha256 !=
                   pair.candidate.build_identity_sha256):
            raise CompositionError("full correctness does not bind candidate build")

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "result_sha256": self.result_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FullCorrectness":
        required = {
            "schema", "operation_key", "build_pair_sha256",
            "candidate_build_identity_sha256", "suite_id", "cases_sha256",
            "receipt_sha256", "passed", "current_full_suite", "result_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("full correctness has an inexact schema")
        if (value.get("schema") !=
                "epyc.autokernel.composition_full_correctness.v1"
                or value.get("current_full_suite") is not True):
            raise CompositionError("full correctness authority changed")
        return cls(**{key: value[key] for key in (
            "operation_key", "build_pair_sha256",
            "candidate_build_identity_sha256", "suite_id", "cases_sha256",
            "receipt_sha256", "passed", "result_sha256")})


@dataclass(frozen=True)
class IncrementalComparison:
    operation_key: str
    build_pair_sha256: str
    correctness_result_sha256: str
    exact_route_receipt_sha256: str
    expected_route_set_sha256: str
    graphs_off_receipt_sha256: str
    graphs_on_receipt_sha256: str
    target_runtime_frame_sha256: str
    exact_route_effect_fraction: float
    graphs_off_effect_fraction: float
    graphs_on_effect_fraction: float
    classification: str
    result_sha256: str

    @classmethod
    def create(
            cls, pair: CumulativeBuildPair, correctness: FullCorrectness, *,
            exact_route_receipt_sha256: str, graphs_on_receipt_sha256: str,
            graphs_off_receipt_sha256: str,
            expected_route_set_sha256: str,
            target_runtime_frame_sha256: str,
            exact_route_effect_fraction: float,
            graphs_off_effect_fraction: float,
            graphs_on_effect_fraction: float,
    ) -> "IncrementalComparison":
        correctness.bind_pair(pair)
        if not correctness.passed:
            raise CompositionError("failed correctness cannot reach composition measurement")
        route = _finite(exact_route_effect_fraction, "exact-route effect")
        graphs_off = _finite(graphs_off_effect_fraction, "graphs-off effect")
        graphs = _finite(graphs_on_effect_fraction, "graphs-on effect")
        if route > 0 and graphs_off > 0 and graphs > 0:
            classification = "candidate"
        elif route <= 0 and graphs_off <= 0 and graphs <= 0:
            classification = "screened_out"
        else:
            classification = "inconclusive"
        body = {
            "schema": "epyc.autokernel.incremental_composition_comparison.v2",
            "operation_key": pair.operation_key,
            "build_pair_sha256": pair.pair_sha256,
            "correctness_result_sha256": correctness.result_sha256,
            "exact_route_receipt_sha256": _require_sha(
                exact_route_receipt_sha256, "exact_route_receipt_sha256"),
            "expected_route_set_sha256": _require_sha(
                expected_route_set_sha256, "expected_route_set_sha256"),
            "graphs_off_receipt_sha256": _require_sha(
                graphs_off_receipt_sha256, "graphs_off_receipt_sha256"),
            "graphs_on_receipt_sha256": _require_sha(
                graphs_on_receipt_sha256, "graphs_on_receipt_sha256"),
            "target_runtime_frame_sha256": _require_sha(
                target_runtime_frame_sha256, "target_runtime_frame_sha256"),
            "exact_route_effect_fraction": route,
            "graphs_off_effect_fraction": graphs_off,
            "graphs_on_effect_fraction": graphs,
            "classification": classification,
            "exact_route_executed": True,
            "graphs_off_executed": True,
            "graphs_on_executed": True,
        }
        return cls(
            operation_key=pair.operation_key,
            build_pair_sha256=pair.pair_sha256,
            correctness_result_sha256=correctness.result_sha256,
            exact_route_receipt_sha256=exact_route_receipt_sha256,
            expected_route_set_sha256=expected_route_set_sha256,
            graphs_off_receipt_sha256=graphs_off_receipt_sha256,
            graphs_on_receipt_sha256=graphs_on_receipt_sha256,
            target_runtime_frame_sha256=target_runtime_frame_sha256,
            exact_route_effect_fraction=route,
            graphs_off_effect_fraction=graphs_off,
            graphs_on_effect_fraction=graphs,
            classification=classification, result_sha256=_sha(body),
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.incremental_composition_comparison.v2",
            "operation_key": self.operation_key,
            "build_pair_sha256": self.build_pair_sha256,
            "correctness_result_sha256": self.correctness_result_sha256,
            "exact_route_receipt_sha256": self.exact_route_receipt_sha256,
            "expected_route_set_sha256": self.expected_route_set_sha256,
            "graphs_off_receipt_sha256": self.graphs_off_receipt_sha256,
            "graphs_on_receipt_sha256": self.graphs_on_receipt_sha256,
            "target_runtime_frame_sha256": self.target_runtime_frame_sha256,
            "exact_route_effect_fraction": self.exact_route_effect_fraction,
            "graphs_off_effect_fraction": self.graphs_off_effect_fraction,
            "graphs_on_effect_fraction": self.graphs_on_effect_fraction,
            "classification": self.classification,
            "exact_route_executed": True, "graphs_off_executed": True,
            "graphs_on_executed": True,
        }

    def __post_init__(self) -> None:
        for label in (
            "operation_key", "build_pair_sha256", "correctness_result_sha256",
            "exact_route_receipt_sha256", "graphs_on_receipt_sha256",
            "graphs_off_receipt_sha256",
            "expected_route_set_sha256", "target_runtime_frame_sha256",
            "result_sha256",
        ):
            _require_sha(getattr(self, label), label)
        route = _finite(self.exact_route_effect_fraction, "exact-route effect")
        graphs_off = _finite(
            self.graphs_off_effect_fraction, "graphs-off effect")
        graphs = _finite(self.graphs_on_effect_fraction, "graphs-on effect")
        expected = ("candidate" if route > 0 and graphs_off > 0 and graphs > 0
                    else "screened_out"
                    if route <= 0 and graphs_off <= 0 and graphs <= 0
                    else "inconclusive")
        if self.classification != expected or self.result_sha256 != _sha(self._body()):
            raise CompositionError("incremental comparison identity changed")

    @property
    def admissible(self) -> bool:
        return self.classification == "candidate"

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "result_sha256": self.result_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IncrementalComparison":
        required = {
            "schema", "operation_key", "build_pair_sha256",
            "correctness_result_sha256", "exact_route_receipt_sha256",
            "graphs_off_receipt_sha256",
            "expected_route_set_sha256", "graphs_on_receipt_sha256",
            "target_runtime_frame_sha256", "exact_route_effect_fraction",
            "graphs_off_effect_fraction",
            "graphs_on_effect_fraction", "classification",
            "exact_route_executed", "graphs_off_executed",
            "graphs_on_executed", "result_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("incremental comparison has an inexact schema")
        if (value.get("schema") !=
                "epyc.autokernel.incremental_composition_comparison.v2"
                or value.get("exact_route_executed") is not True
                or value.get("graphs_off_executed") is not True
                or value.get("graphs_on_executed") is not True):
            raise CompositionError("incremental comparison authority changed")
        return cls(**{key: value[key] for key in (
            "operation_key", "build_pair_sha256", "correctness_result_sha256",
            "exact_route_receipt_sha256", "expected_route_set_sha256",
            "graphs_off_receipt_sha256",
            "graphs_on_receipt_sha256", "target_runtime_frame_sha256",
            "exact_route_effect_fraction", "graphs_on_effect_fraction",
            "graphs_off_effect_fraction",
            "classification", "result_sha256")})


def _atomic_replace(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_stat = os.lstat(path.parent)
    if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
        raise CompositionError("composition state parent must be a real directory")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise CompositionError("composition temporary state path already exists")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


class CompositionLedger:
    """Atomic, restart-safe state for one cumulative composition campaign."""

    SCHEMA = "epyc.autokernel.cumulative_composition_state.v2"

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def create(
            self, authority: CompositionAuthority, *,
            max_scientific_attempts: int = 10,
    ) -> dict[str, Any]:
        if (isinstance(max_scientific_attempts, bool)
                or not isinstance(max_scientific_attempts, int)
                or max_scientific_attempts <= 0):
            raise CompositionError("composition scientific budget is invalid")
        with self._lock():
            if self.path.exists() or self.path.is_symlink():
                state = self._load_unlocked()
                if (state["initial_authority"] != authority.to_dict()
                        or state["max_scientific_attempts"] !=
                           max_scientific_attempts):
                    raise CompositionError("existing composition state has other authority")
                return state
            state = {
                "schema": self.SCHEMA, "campaign_id": authority.campaign_id,
                "max_scientific_attempts": max_scientific_attempts,
                "initial_authority": authority.to_dict(),
                "authority": authority.to_dict(), "pending": None,
                "terminals": [], "scientific_attempts": 0, "generation": 0,
            }
            return self._write_unlocked(state)

    def load(self) -> dict[str, Any]:
        with self._lock():
            return self._load_unlocked()

    def begin(self, plan: CompositionPlan) -> dict[str, Any]:
        with self._lock():
            state = self._load_unlocked()
            authority = CompositionAuthority.from_dict(state["authority"])
            if plan.anchor != authority:
                raise CompositionError("composition plan anchor is stale")
            if state["pending"] is not None:
                current = CompositionPlan.from_dict(state["pending"]["plan"])
                if current == plan:
                    return state
                raise CompositionError("another cumulative composition is pending")
            if state["scientific_attempts"] >= state["max_scientific_attempts"]:
                raise CompositionError("composition scientific budget is exhausted")
            proposed = plan.candidate.accepted[-1]
            for terminal in state["terminals"]:
                if (terminal["scientific_budget_spent"] is True
                        and terminal["cross_campaign_candidate_sha256"] ==
                            proposed.cross_campaign_candidate_sha256):
                    raise CompositionError("composition would repeat a scientific lever")
            state["pending"] = {
                "stage": "planned", "plan": plan.to_dict(),
                "build_pair": None, "correctness": None, "comparison": None,
            }
            return self._write_unlocked(state)

    def record_build_pair(self, pair: CumulativeBuildPair) -> dict[str, Any]:
        with self._lock():
            state = self._load_unlocked()
            if state["pending"] is None:
                matches = [row for row in state["terminals"]
                           if (row["operation_key"] == pair.operation_key
                               and row["build_pair"] == pair.to_dict())]
                if len(matches) == 1:
                    return state
                raise CompositionError("no cumulative composition is pending")
            state, pending, plan = self._pending(state)
            pair.bind_plan(plan)
            if pending["build_pair"] is not None:
                if pending["build_pair"] == pair.to_dict():
                    return state
                raise CompositionError("composition build pair changed on restart")
            if pending["stage"] != "planned":
                raise CompositionError("composition builds arrived out of order")
            pending["build_pair"] = pair.to_dict()
            pending["stage"] = "built"
            return self._write_unlocked(state)

    def record_correctness(self, correctness: FullCorrectness) -> dict[str, Any]:
        with self._lock():
            state = self._load_unlocked()
            if state["pending"] is None:
                matches = [row for row in state["terminals"]
                           if (row["operation_key"] == correctness.operation_key
                               and row["correctness_result_sha256"] ==
                                   correctness.result_sha256)]
                if len(matches) == 1:
                    return state
                raise CompositionError("no cumulative composition is pending")
            state, pending, plan = self._pending(state)
            if pending["build_pair"] is None:
                raise CompositionError("full correctness cannot precede builds")
            pair = CumulativeBuildPair.from_dict(pending["build_pair"])
            pair.bind_plan(plan)
            correctness.bind_pair(pair)
            if pending["correctness"] is not None:
                if pending["correctness"] == correctness.to_dict():
                    return state
                raise CompositionError("full correctness changed on restart")
            if pending["stage"] != "built":
                raise CompositionError("full correctness arrived out of order")
            pending["correctness"] = correctness.to_dict()
            if correctness.passed:
                pending["stage"] = "correctness_passed"
                return self._write_unlocked(state)
            return self._terminalize(
                state, plan=plan, disposition="correctness_rollback",
                scientific=True, correctness=correctness, comparison=None,
                admitted=None, reason_code="current_full_correctness_failed",
            )

    def record_comparison(
            self, comparison: IncrementalComparison,
    ) -> dict[str, Any]:
        with self._lock():
            state = self._load_unlocked()
            if state["pending"] is None:
                matches = [row for row in state["terminals"]
                           if (row["operation_key"] ==
                               comparison.operation_key
                               and row["comparison"] ==
                               comparison.to_dict())]
                if len(matches) == 1:
                    return state
                raise CompositionError("no cumulative composition is pending")
            state, pending, plan = self._pending(state)
            if pending["build_pair"] is None or pending["correctness"] is None:
                raise CompositionError("incremental comparison cannot skip correctness")
            pair = CumulativeBuildPair.from_dict(pending["build_pair"])
            correctness = FullCorrectness.from_dict(pending["correctness"])
            pair.bind_plan(plan)
            correctness.bind_pair(pair)
            if not correctness.passed:
                raise CompositionError("failed correctness cannot be measured")
            if (comparison.operation_key != plan.operation_key
                    or comparison.build_pair_sha256 != pair.pair_sha256
                    or comparison.correctness_result_sha256 !=
                       correctness.result_sha256):
                raise CompositionError("incremental comparison binds other evidence")
            if pending["comparison"] is not None:
                if pending["comparison"] == comparison.to_dict():
                    return state
                raise CompositionError("incremental comparison changed on restart")
            if pending["stage"] != "correctness_passed":
                raise CompositionError("incremental comparison arrived out of order")
            pending["comparison"] = comparison.to_dict()
            pending["stage"] = "measured"
            return self._write_unlocked(state)

    def finalize(self, operation_key: str) -> dict[str, Any]:
        _require_sha(operation_key, "operation_key")
        with self._lock():
            state = self._load_unlocked()
            if state["pending"] is None:
                matches = [row for row in state["terminals"]
                           if row["operation_key"] == operation_key]
                if len(matches) == 1:
                    return state
                raise CompositionError("composition operation is not pending")
            state, pending, plan = self._pending(state)
            if plan.operation_key != operation_key or pending["stage"] != "measured":
                raise CompositionError("composition is not ready to finalize")
            correctness = FullCorrectness.from_dict(pending["correctness"])
            comparison = IncrementalComparison.from_dict(pending["comparison"])
            if comparison.admissible:
                return self._terminalize(
                    state, plan=plan, disposition="admitted", scientific=True,
                    correctness=correctness, comparison=comparison,
                    admitted=plan.candidate, reason_code="incremental_both_positive",
                )
            return self._terminalize(
                state, plan=plan, disposition="incremental_rollback",
                scientific=True, correctness=correctness, comparison=comparison,
                admitted=None,
                reason_code=f"incremental_{comparison.classification}",
            )

    def rollback_attribution(
            self, operation_key: str, *, receipt_sha256: str,
    ) -> dict[str, Any]:
        """Scientifically reject a stack whose exact route authority failed."""
        _require_sha(operation_key, "operation_key")
        _require_sha(receipt_sha256, "receipt_sha256")
        with self._lock():
            state = self._load_unlocked()
            if state["pending"] is None:
                matches = [row for row in state["terminals"]
                           if (row["operation_key"] == operation_key
                               and row["disposition"] == "attribution_rollback"
                               and row["attribution_receipt_sha256"] ==
                                   receipt_sha256)]
                if len(matches) == 1:
                    return state
                raise CompositionError("no cumulative composition is pending")
            state, pending, plan = self._pending(state)
            if (plan.operation_key != operation_key
                    or pending["stage"] != "correctness_passed"):
                raise CompositionError(
                    "attribution rollback arrived out of order")
            correctness = FullCorrectness.from_dict(pending["correctness"])
            return self._terminalize(
                state, plan=plan, disposition="attribution_rollback",
                scientific=True, correctness=correctness, comparison=None,
                admitted=None, reason_code="exact_route_authority_failed",
                attribution_receipt_sha256=receipt_sha256)

    def rollback_infrastructure(
            self, operation_key: str, *, reason_code: str,
            receipt_sha256: str,
    ) -> dict[str, Any]:
        _require_sha(operation_key, "operation_key")
        _require_text(reason_code, "reason_code")
        _require_sha(receipt_sha256, "receipt_sha256")
        with self._lock():
            state = self._load_unlocked()
            if state["pending"] is None:
                matches = [row for row in state["terminals"]
                           if (row["operation_key"] == operation_key
                               and row["disposition"] == "infrastructure_rollback"
                               and row["reason_code"] == reason_code
                               and row["infrastructure_receipt_sha256"] ==
                                   receipt_sha256)]
                if len(matches) == 1:
                    return state
                raise CompositionError("no cumulative composition is pending")
            state, _pending, plan = self._pending(state)
            if plan.operation_key != operation_key:
                raise CompositionError("infrastructure rollback names another operation")
            return self._terminalize(
                state, plan=plan, disposition="infrastructure_rollback",
                scientific=False, correctness=None, comparison=None,
                admitted=None, reason_code=reason_code,
                infrastructure_receipt_sha256=receipt_sha256,
            )

    class _Lock:
        def __init__(self, path: Path):
            self.path = path
            self.handle: Any = None

        def __enter__(self) -> None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(self.path, flags, 0o600)
            except OSError as exc:
                raise CompositionError("composition lock is unsafe") from exc
            facts = os.fstat(descriptor)
            if (not stat.S_ISREG(facts.st_mode) or facts.st_nlink != 1
                    or facts.st_uid != os.geteuid()
                    or facts.st_mode & 0o022):
                os.close(descriptor)
                raise CompositionError("composition lock identity is unsafe")
            self.handle = os.fdopen(descriptor, "a+")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

        def __exit__(self, *_args: Any) -> None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def _lock(self) -> "CompositionLedger._Lock":
        return self._Lock(self.lock_path)

    def _load_unlocked(self) -> dict[str, Any]:
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags)
            try:
                before = os.fstat(descriptor)
                if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                        or before.st_uid != os.geteuid()
                        or before.st_mode & 0o022
                        or before.st_size > 16 * 1024 * 1024):
                    raise CompositionError("composition state identity is unsafe")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            path_facts = os.lstat(self.path)
            def identity(row: os.stat_result) -> tuple[int, ...]:
                return (
                    row.st_dev, row.st_ino, row.st_uid,
                    stat.S_IFMT(row.st_mode), row.st_nlink, row.st_size,
                    row.st_mtime_ns, row.st_ctime_ns,
                )
            if identity(before) != identity(after) or identity(after) != identity(path_facts):
                raise CompositionError("composition state changed during stable read")
            raw = b"".join(chunks)
            value = json.loads(
                raw.decode("utf-8", "strict"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")),
            )
        except CompositionError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CompositionError("composition state is not strict JSON") from exc
        required = {
            "schema", "campaign_id", "max_scientific_attempts", "authority",
            "initial_authority", "pending", "terminals", "scientific_attempts",
            "generation", "state_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("composition state has an inexact schema")
        state = dict(value)
        if state["schema"] != self.SCHEMA:
            raise CompositionError("composition state schema changed")
        if state["state_sha256"] != _sha({
                key: row for key, row in state.items() if key != "state_sha256"}):
            raise CompositionError("composition state self-hash changed")
        initial = CompositionAuthority.from_dict(state["initial_authority"])
        authority = CompositionAuthority.from_dict(state["authority"])
        if state["campaign_id"] != authority.campaign_id:
            raise CompositionError("composition state campaign changed")
        if state["campaign_id"] != initial.campaign_id:
            raise CompositionError("composition initial campaign changed")
        if (isinstance(state["max_scientific_attempts"], bool)
                or not isinstance(state["max_scientific_attempts"], int)
                or state["max_scientific_attempts"] <= 0
                or isinstance(state["generation"], bool)
                or not isinstance(state["generation"], int)
                or state["generation"] < 0
                or not isinstance(state["terminals"], list)):
            raise CompositionError("composition state counters are malformed")
        terminal_keys: set[str] = set()
        scientific_candidates: set[str] = set()
        derived_authority = initial
        science = 0
        for terminal in state["terminals"]:
            self._validate_terminal(terminal)
            if terminal["operation_key"] in terminal_keys:
                raise CompositionError("composition operation terminal is duplicated")
            terminal_keys.add(terminal["operation_key"])
            science += int(terminal["scientific_budget_spent"])
            if terminal["scientific_budget_spent"]:
                cross_identity = terminal["cross_campaign_candidate_sha256"]
                if cross_identity in scientific_candidates:
                    raise CompositionError(
                        "composition scientific candidate is duplicated")
                scientific_candidates.add(cross_identity)
            plan = CompositionPlan.from_dict(terminal["plan"])
            if plan.anchor != derived_authority:
                raise CompositionError("composition terminal chain has a stale anchor")
            if terminal["disposition"] == "admitted":
                derived_authority = plan.candidate
        if authority != derived_authority:
            raise CompositionError("composition authority differs from terminal chain")
        if state["scientific_attempts"] != science:
            raise CompositionError("composition science count differs from terminals")
        if state["scientific_attempts"] > state["max_scientific_attempts"]:
            raise CompositionError("composition state exceeds its scientific budget")
        if state["pending"] is not None:
            self._validate_pending(state["pending"], authority, terminal_keys)
        return state

    def _write_unlocked(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        state.pop("state_sha256", None)
        state["generation"] = int(state.get("generation", -1)) + 1
        state["scientific_attempts"] = sum(
            int(row["scientific_budget_spent"]) for row in state["terminals"]
        )
        state["state_sha256"] = _sha(state)
        _atomic_replace(self.path, state)
        return self._load_unlocked()

    @staticmethod
    def _pending(
            state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], CompositionPlan]:
        pending = state.get("pending")
        if not isinstance(pending, dict):
            raise CompositionError("no cumulative composition is pending")
        return state, pending, CompositionPlan.from_dict(pending["plan"])

    @staticmethod
    def _validate_pending(
            pending: Mapping[str, Any], authority: CompositionAuthority,
            terminal_keys: set[str],
    ) -> None:
        if not isinstance(pending, Mapping) or set(pending) != {
            "stage", "plan", "build_pair", "correctness", "comparison",
        }:
            raise CompositionError("composition pending state has an inexact schema")
        plan = CompositionPlan.from_dict(pending["plan"])
        if plan.anchor != authority or plan.operation_key in terminal_keys:
            raise CompositionError("composition pending authority is stale")
        stage = pending["stage"]
        allowed = {"planned", "built", "correctness_passed", "measured"}
        if stage not in allowed:
            raise CompositionError("composition pending stage is invalid")
        pair = (None if pending["build_pair"] is None else
                CumulativeBuildPair.from_dict(pending["build_pair"]))
        correctness = (None if pending["correctness"] is None else
                       FullCorrectness.from_dict(pending["correctness"]))
        comparison = (None if pending["comparison"] is None else
                      IncrementalComparison.from_dict(pending["comparison"]))
        expected_presence = {
            "planned": (False, False, False), "built": (True, False, False),
            "correctness_passed": (True, True, False),
            "measured": (True, True, True),
        }
        if tuple(row is not None for row in (pair, correctness, comparison)) != \
                expected_presence[stage]:
            raise CompositionError("composition pending stage/evidence disagree")
        if pair is not None:
            pair.bind_plan(plan)
        if correctness is not None:
            correctness.bind_pair(pair)
            if not correctness.passed:
                raise CompositionError("failed correctness remained pending")
        if comparison is not None and (
                comparison.operation_key != plan.operation_key
                or comparison.build_pair_sha256 != pair.pair_sha256
                or comparison.correctness_result_sha256 !=
                   correctness.result_sha256):
            raise CompositionError("pending comparison evidence changed")

    def _terminalize(
            self, state: dict[str, Any], *, plan: CompositionPlan,
            disposition: str, scientific: bool,
            correctness: FullCorrectness | None,
            comparison: IncrementalComparison | None,
            admitted: CompositionAuthority | None, reason_code: str,
            infrastructure_receipt_sha256: str | None = None,
            attribution_receipt_sha256: str | None = None,
    ) -> dict[str, Any]:
        if scientific and state["scientific_attempts"] >= \
                state["max_scientific_attempts"]:
            raise CompositionError("composition scientific budget is exhausted")
        lever = plan.candidate.accepted[-1]
        body = {
            "schema": "epyc.autokernel.cumulative_composition_terminal.v2",
            "operation_key": plan.operation_key, "plan_sha256": plan.plan_sha256,
            "plan": plan.to_dict(),
            "lever_sha256": lever.lever_sha256,
            "cross_campaign_candidate_sha256":
                lever.cross_campaign_candidate_sha256,
            "isolated_result_sha256s": [
                row.result_sha256 for row in lever.replications
            ],
            "disposition": disposition,
            "scientific_budget_spent": scientific,
            "build_pair": state["pending"].get("build_pair"),
            "correctness": None if correctness is None else correctness.to_dict(),
            "comparison": None if comparison is None else comparison.to_dict(),
            "correctness_result_sha256":
                None if correctness is None else correctness.result_sha256,
            "comparison_result_sha256":
                None if comparison is None else comparison.result_sha256,
            "admitted_authority_sha256":
                None if admitted is None else admitted.authority_sha256,
            "reason_code": _require_text(reason_code, "reason_code"),
            "infrastructure_receipt_sha256": infrastructure_receipt_sha256,
            "attribution_receipt_sha256": attribution_receipt_sha256,
        }
        if infrastructure_receipt_sha256 is not None:
            _require_sha(infrastructure_receipt_sha256,
                         "infrastructure_receipt_sha256")
        if attribution_receipt_sha256 is not None:
            _require_sha(attribution_receipt_sha256,
                         "attribution_receipt_sha256")
        terminal = {**body, "terminal_sha256": _sha(body)}
        if admitted is not None:
            state["authority"] = admitted.to_dict()
        state["terminals"].append(terminal)
        state["pending"] = None
        return self._write_unlocked(state)

    @staticmethod
    def _validate_terminal(value: Mapping[str, Any]) -> None:
        required = {
            "schema", "operation_key", "plan_sha256", "plan", "lever_sha256",
            "cross_campaign_candidate_sha256", "isolated_result_sha256s",
            "disposition", "scientific_budget_spent",
            "build_pair", "correctness", "comparison",
            "correctness_result_sha256", "comparison_result_sha256",
            "admitted_authority_sha256", "reason_code",
            "infrastructure_receipt_sha256", "terminal_sha256",
            "attribution_receipt_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("composition terminal has an inexact schema")
        if value.get("schema") != "epyc.autokernel.cumulative_composition_terminal.v2":
            raise CompositionError("composition terminal schema changed")
        for key in ("operation_key", "plan_sha256", "lever_sha256",
                    "cross_campaign_candidate_sha256"):
            _require_sha(value[key], key)
        plan = CompositionPlan.from_dict(value["plan"])
        if (value["operation_key"] != plan.operation_key
                or value["plan_sha256"] != plan.plan_sha256
                or value["lever_sha256"] !=
                   plan.candidate.accepted[-1].lever_sha256
                or value["cross_campaign_candidate_sha256"] !=
                   plan.candidate.accepted[-1].cross_campaign_candidate_sha256):
            raise CompositionError("composition terminal plan binding changed")
        isolated = value["isolated_result_sha256s"]
        if (not isinstance(isolated, list) or len(isolated) < 2
                or len(set(isolated)) != len(isolated)
                or any(not SHA256.fullmatch(row) for row in isolated)):
            raise CompositionError("terminal lost isolated scientific evidence")
        if isolated != [row.result_sha256
                        for row in plan.candidate.accepted[-1].replications]:
            raise CompositionError("terminal isolated evidence differs from its plan")
        if not isinstance(value["scientific_budget_spent"], bool):
            raise CompositionError("terminal science disposition is malformed")
        for key in ("correctness_result_sha256", "comparison_result_sha256",
                    "admitted_authority_sha256", "infrastructure_receipt_sha256",
                    "attribution_receipt_sha256"):
            if value[key] is not None:
                _require_sha(value[key], key)
        _require_text(value["disposition"], "disposition")
        _require_text(value["reason_code"], "reason_code")
        pair = (None if value["build_pair"] is None else
                CumulativeBuildPair.from_dict(value["build_pair"]))
        correctness = (None if value["correctness"] is None else
                       FullCorrectness.from_dict(value["correctness"]))
        comparison = (None if value["comparison"] is None else
                      IncrementalComparison.from_dict(value["comparison"]))
        if pair is not None:
            pair.bind_plan(plan)
        if correctness is not None:
            if pair is None:
                raise CompositionError("terminal correctness lacks build pair")
            correctness.bind_pair(pair)
        if comparison is not None:
            if correctness is None or pair is None:
                raise CompositionError("terminal comparison lacks prerequisite evidence")
            if (comparison.operation_key != plan.operation_key
                    or comparison.build_pair_sha256 != pair.pair_sha256
                    or comparison.correctness_result_sha256 !=
                       correctness.result_sha256):
                raise CompositionError("terminal comparison binding changed")
        if ((None if correctness is None else correctness.result_sha256) !=
                value["correctness_result_sha256"]
                or (None if comparison is None else comparison.result_sha256) !=
                   value["comparison_result_sha256"]):
            raise CompositionError("terminal evidence hashes changed")
        disposition = value["disposition"]
        shape = (
            value["scientific_budget_spent"],
            value["correctness_result_sha256"] is not None,
            value["comparison_result_sha256"] is not None,
            value["admitted_authority_sha256"] is not None,
            value["infrastructure_receipt_sha256"] is not None,
            value["attribution_receipt_sha256"] is not None,
        )
        expected_shapes = {
            "admitted": (True, True, True, True, False, False),
            "incremental_rollback": (True, True, True, False, False, False),
            "correctness_rollback": (True, True, False, False, False, False),
            "attribution_rollback": (True, True, False, False, False, True),
            "infrastructure_rollback": (False, False, False, False, True, False),
        }
        if expected_shapes.get(disposition) != shape:
            raise CompositionError("composition terminal disposition/evidence disagree")
        if (disposition == "admitted"
                and value["admitted_authority_sha256"] !=
                    plan.candidate.authority_sha256):
            raise CompositionError("admitted authority does not bind terminal plan")
        body = {key: row for key, row in value.items() if key != "terminal_sha256"}
        if value["terminal_sha256"] != _sha(body):
            raise CompositionError("composition terminal identity changed")
