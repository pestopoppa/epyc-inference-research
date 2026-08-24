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
from statistics import median
from typing import Any, Mapping, Sequence

from . import source_candidate
from .controller import gpu_source_evidence, gpu_source_proofs


__all__ = (
    "BuildBinding", "CompositionAuthority", "CompositionError",
    "CompositionLedger", "CompositionPlan", "CumulativeBuildPair",
    "CumulativePerformance", "CumulativePerformanceRef", "DnrAuthority",
    "FrozenProductionAuthority", "FrozenProductionComparator",
    "FullCorrectness", "IncrementalComparison", "MeasurementReceiptRef",
    "IsolatedReplication", "ReplicatedPositiveLever",
)


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
FROZEN_PRODUCTION_BRANCH = "production-consolidated-v9"
FROZEN_PRODUCTION_COMMIT = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
FROZEN_PRODUCTION_SOURCE_SHA256 = \
    "48663678dad691ba046cc3d3e9a70d42046d92602db94704b3707e3fadb82ca8"
FROZEN_BUILD_RECEIPT_SHA256 = \
    "04344a3624247646d0ebd795c12abfa4db48be895ebc8ccc465867917e7da679"
FROZEN_LINKAGE_RECEIPT_SHA256 = \
    "8c4f507a2072fdb170ad4ac7bfa9ca4867bd20023587615b5f960eab9e21580b"
FROZEN_RUNTIME_RECEIPT_SHA256 = \
    "e5374f054d0c1be12fe6d511c32c5e24fcac9fb79bc954c82f42bc28e8822d48"
FROZEN_MEASUREMENT_RECEIPT_SHA256 = \
    "21c396477c1cdcc71dbaffd7452dd43e7bbf5941b1f199c8a5d217da830945ed"
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


def _validate_dnr_history(
        plan: CompositionPlan, prior_cross_campaign_candidates: Sequence[str],
) -> None:
    proposed = plan.candidate.accepted[-1].cross_campaign_candidate_sha256
    expected = {
        lever.cross_campaign_candidate_sha256
        for lever in plan.anchor.accepted
    }
    expected.update(
        candidate for candidate in prior_cross_campaign_candidates
        if candidate != proposed)
    if plan.dnr.checked_cross_campaign_candidate_sha256s != \
            tuple(sorted(expected)):
        raise CompositionError(
            "composition DNR registry omits or invents prior candidates")


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


_MEASUREMENT_ROLES = frozenset({
    "exact_route", "incremental_graphs_off", "incremental_graphs_on",
    "production_graphs_on",
})


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, row in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key}")
        value[key] = row
    return value


_AUTHORITY_JOURNAL = "composition-authority.jsonl"
_AUTHORITY_JOURNAL_SCHEMA = \
    "epyc.autokernel.cumulative_authority_journal_event.v1"


def _authority_journal_path(operation_root: Path) -> Path:
    path = operation_root / _AUTHORITY_JOURNAL
    if (not operation_root.is_absolute()
            or operation_root != operation_root.resolve(strict=False)
            or path != path.resolve(strict=False)):
        raise CompositionError("composition authority journal path is not canonical")
    return path


def _read_authority_journal(operation_root: Path) -> tuple[dict[str, Any], ...]:
    """Read the append-only operation authority chain with strict JSON.

    Runner plans, measurement receipts, performance receipts, and controller
    state are replaceable snapshots.  This journal is the separately appended
    boundary that commits their byte identities before/after execution.
    """
    path = _authority_journal_path(operation_root)
    if not path.exists() and not path.is_symlink():
        return ()
    raw = _stable_receipt_bytes(path)
    if not raw or not raw.endswith(b"\n"):
        raise CompositionError("composition authority journal has a torn tail")
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    for index, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(
                line.decode("utf-8", "strict"), object_pairs_hook=_strict_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CompositionError(
                "composition authority journal is not strict JSON") from exc
        required = {
            "schema", "sequence", "previous_event_sha256", "kind",
            "operation_key", "payload", "event_sha256",
        }
        if (not isinstance(value, dict) or set(value) != required
                or value.get("schema") != _AUTHORITY_JOURNAL_SCHEMA
                or value.get("sequence") != index
                or value.get("previous_event_sha256") != previous
                or value.get("kind") not in {"pre_run", "result"}
                or not isinstance(value.get("payload"), Mapping)):
            raise CompositionError("composition authority journal chain is malformed")
        _require_sha(value.get("operation_key"), "journal operation_key")
        event_sha = _require_sha(value.get("event_sha256"), "journal event_sha256")
        if event_sha != _sha({key: row for key, row in value.items()
                              if key != "event_sha256"}):
            raise CompositionError("composition authority journal hash chain changed")
        previous = event_sha
        rows.append(value)
    return tuple(rows)


def _append_authority_event(
        operation_root: Path, *, kind: str, operation_key: str,
        payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one idempotent authority event; never replace an old event."""
    if kind not in {"pre_run", "result"}:
        raise CompositionError("composition authority journal kind is invalid")
    _require_sha(operation_key, "journal operation_key")
    path = _authority_journal_path(operation_root)
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock, flags, 0o600)
    try:
        facts = os.fstat(descriptor)
        if (not stat.S_ISREG(facts.st_mode) or facts.st_nlink != 1
                or facts.st_uid != os.geteuid() or facts.st_mode & 0o022):
            raise CompositionError("composition authority journal lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        rows = _read_authority_journal(operation_root) \
            if path.exists() else ()
        matches = [row for row in rows if row["kind"] == kind]
        if matches:
            if len(matches) != 1 or matches[0]["operation_key"] != operation_key \
                    or matches[0]["payload"] != dict(payload):
                raise CompositionError(
                    f"composition {kind} authority changed after commitment")
            return matches[0]
        if kind == "result" and (len(rows) != 1 or rows[0]["kind"] != "pre_run"):
            raise CompositionError("composition result lacks one pre-run commitment")
        if kind == "pre_run" and rows:
            raise CompositionError("composition pre-run authority was not first")
        event = {
            "schema": _AUTHORITY_JOURNAL_SCHEMA,
            "sequence": len(rows) + 1,
            "previous_event_sha256": (
                rows[-1]["event_sha256"] if rows else "0" * 64),
            "kind": kind, "operation_key": operation_key,
            "payload": dict(payload),
        }
        event["event_sha256"] = _sha(event)
        encoded = _canonical(event) + b"\n"
        write_flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            write_flags |= os.O_NOFOLLOW
        out = os.open(path, write_flags, 0o600)
        try:
            before = os.fstat(out)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or before.st_uid != os.geteuid()
                    or before.st_mode & 0o022):
                raise CompositionError("composition authority journal is unsafe")
            os.write(out, encoded)
            os.fsync(out)
        finally:
            os.close(out)
        reopened = _read_authority_journal(operation_root)
        if reopened[-1] != event:
            raise CompositionError("composition authority event changed while appending")
        return event
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _stable_receipt_bytes(
        path: Path, *, expected_sha256: str | None = None) -> bytes:
    """Read one immutable evidence file without following a final symlink."""
    if expected_sha256 is not None:
        expected_sha256 = _require_sha(
            expected_sha256, "measurement file sha256")
    if not path.is_absolute() or ".." in path.parts:
        raise CompositionError("measurement receipt path is unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompositionError("measurement receipt is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or before.st_mode & 0o022
                or before.st_size > 16 * 1024 * 1024):
            raise CompositionError("measurement receipt identity is unsafe")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        pathname = os.lstat(path)
    except OSError as exc:
        raise CompositionError("measurement receipt path disappeared") from exc
    identity = lambda row: (
        row.st_dev, row.st_ino, row.st_uid, stat.S_IFMT(row.st_mode),
        row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if (identity(before) != identity(after)
            or identity(after) != identity(pathname)):
        raise CompositionError("measurement receipt changed during stable read")
    raw = b"".join(chunks)
    if (expected_sha256 is not None
            and hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise CompositionError("measurement receipt bytes changed")
    return raw


def _strict_runner_plan(operation_root: Path) -> tuple[dict[str, Any], bytes]:
    """Strict-parse one runner plan and bind its native identity."""
    path = operation_root / "runner-plan.json"
    if path != path.resolve(strict=False):
        raise CompositionError("runner plan path is not canonical")
    raw = _stable_receipt_bytes(path)
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompositionError("runner plan is not strict JSON") from exc
    if (not isinstance(value, dict)
            or value.get("schema") !=
               "epyc.autokernel.gpu_source_runner_plan.v2"
            or value.get("authority") !=
               "nonpromotable_candidate_only_discovery"
            or value.get("promotion_claim") is not False):
        raise CompositionError("runner plan authority changed")
    receipt = _require_sha(value.get("receipt_sha256"),
                           "runner plan receipt_sha256")
    if receipt != _sha({key: row for key, row in value.items()
                        if key != "receipt_sha256"}):
        raise CompositionError("runner plan native identity changed")
    return value, raw


def _runner_measurement_authority_uncommitted(
        operation_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project the runner plan and the bytes the pre-run journal commits."""
    value, raw = _strict_runner_plan(operation_root)
    pair = CumulativeBuildPair.from_dict(value.get("composition_build_pair"))
    correctness = FullCorrectness.from_dict(
        value.get("composition_correctness"))
    correctness.bind_pair(pair)
    production = FrozenProductionAuthority.from_dict(
        value.get("composition_production_authority"))
    authority = {
        "operation_key": _require_sha(
            value.get("operation_key"), "runner plan operation_key"),
        "build_pair_sha256": pair.pair_sha256,
        "correctness_result_sha256": correctness.result_sha256,
        "exact_route_receipt_sha256": _require_sha(
            value.get("composition_exact_route_receipt_sha256"),
            "runner plan exact-route receipt"),
        "expected_route_set_sha256": _require_sha(
            value.get("composition_expected_route_set_sha256"),
            "runner plan expected route set"),
        "target_runtime_frame_sha256": _require_sha(
            value.get("composition_target_runtime_frame_sha256"),
            "runner plan target runtime frame"),
        "frozen_production": production,
    }
    proof_path = operation_root / "proof/proof-bundle.json"
    proof_raw = _stable_receipt_bytes(proof_path)
    payload = {
        "runner_plan_file_sha256": hashlib.sha256(raw).hexdigest(),
        "runner_plan_receipt_sha256": _require_sha(
            value.get("receipt_sha256"), "runner plan receipt_sha256"),
        "proof_bundle_file_sha256": hashlib.sha256(proof_raw).hexdigest(),
        "build_pair_sha256": authority["build_pair_sha256"],
        "correctness_result_sha256": authority["correctness_result_sha256"],
        "exact_route_receipt_sha256":
            authority["exact_route_receipt_sha256"],
        "expected_route_set_sha256": authority["expected_route_set_sha256"],
        "target_runtime_frame_sha256":
            authority["target_runtime_frame_sha256"],
        "frozen_production_authority_sha256": production.authority_sha256,
    }
    return authority, payload


def _strict_screen_result(result_raw: bytes, *, label: str) -> str:
    """Strict-parse one sealed runner result and return its native hash."""
    try:
        result = json.loads(
            result_raw.decode("utf-8", "strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompositionError(f"{label} is not strict JSON") from exc
    if (not isinstance(result, dict)
            or result.get("schema") !=
               "epyc.autokernel.gpu_candidate_only_screen.v2"
            or result.get("promotion_claim") is not False
            or result.get("non_promotable") is not True):
        raise CompositionError(f"{label} authority changed")
    native = _require_sha(result.get("result_sha256"),
                          f"{label} result_sha256")
    if native != _sha({key: row for key, row in result.items()
                       if key != "result_sha256"}):
        raise CompositionError(f"{label} native identity changed")
    return native


def _runner_result_payload_uncommitted(
        operation_root: Path,
) -> dict[str, Any] | None:
    """Project the sealed result bytes the result journal must commit.

    Returns None when the runner plan is not a cumulative three-output plan;
    such a plan can never carry a result commitment.
    """
    value, _ = _strict_runner_plan(operation_root)
    off_raw = value.get("measurement_graphs_off_output_dir")
    on_raw = value.get("target_runtime_graphs_on_output_dir")
    production_raw = value.get("production_graphs_on_output_dir")
    performance_raw = value.get("cumulative_performance_path")
    if not all(isinstance(item, str) for item in (
            off_raw, on_raw, production_raw, performance_raw)):
        return None
    runner_root = (operation_root / "runner").resolve()
    directories = tuple(Path(str(item)).resolve() for item in (
        off_raw, on_raw, production_raw))
    performance_path = Path(str(performance_raw)).resolve()
    if (len(set(directories)) != 3
            or any(not path.is_relative_to(runner_root)
                   for path in directories)
            or performance_path !=
               (operation_root / "cumulative-performance.json").resolve()):
        raise CompositionError(
            "cumulative result output escaped its operation")
    results: dict[str, Any] = {}
    for label, directory in (
            ("graphs_off", directories[0]),
            ("graphs_on", directories[1]),
            ("production_graphs_on", directories[2])):
        result_path = directory / "result.json"
        if result_path.is_symlink() or not result_path.is_file():
            return None
        result_raw = _stable_receipt_bytes(result_path)
        results[label] = {
            "path": str(result_path),
            "file_sha256": hashlib.sha256(result_raw).hexdigest(),
            "result_sha256": _strict_screen_result(
                result_raw, label=f"runner result {label}"),
        }
    if performance_path.is_symlink() or not performance_path.is_file():
        return None
    performance_raw = _stable_receipt_bytes(performance_path)
    try:
        performance = json.loads(
            performance_raw.decode("utf-8", "strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompositionError(
            "cumulative performance receipt is not strict JSON") from exc
    if (not isinstance(performance, dict)
            or performance.get("schema") !=
               "epyc.autokernel.cumulative_performance.v2"
            or performance.get("promotion_authority") is not True):
        raise CompositionError("cumulative performance authority changed")
    performance_native = _require_sha(
        performance.get("result_sha256"),
        "cumulative performance result_sha256")
    if performance_native != _sha(
            {key: row for key, row in performance.items()
             if key != "result_sha256"}):
        raise CompositionError(
            "cumulative performance native identity changed")
    return {
        "runner_plan_file_sha256": hashlib.sha256(
            _stable_receipt_bytes(operation_root / "runner-plan.json")
        ).hexdigest(),
        "results": results,
        "cumulative_performance": {
            "path": str(performance_path),
            "file_sha256": hashlib.sha256(performance_raw).hexdigest(),
            "result_sha256": performance_native,
        },
    }


def commit_pre_run_authority(operation_root: Path) -> dict[str, Any]:
    """Recursively validate proof bytes, then append their pre-run commitment."""
    try:
        gpu_source_evidence.load_gpu_source_evidence_bundle(
            operation_root / "proof/proof-bundle.json")
    except (gpu_source_evidence.EvidenceProducerError,
            gpu_source_proofs.ProofError) as exc:
        raise CompositionError(
            "composition proof bundle failed recursive pre-run reopening") from exc
    authority, payload = _runner_measurement_authority_uncommitted(
        operation_root)
    return _append_authority_event(
        operation_root, kind="pre_run",
        operation_key=authority["operation_key"], payload=payload)


def commit_result_authority(operation_root: Path) -> dict[str, Any]:
    """Append the result commitment after the cumulative runner seals."""
    authority, _ = _runner_measurement_authority_uncommitted(operation_root)
    payload = _runner_result_payload_uncommitted(operation_root)
    if payload is None:
        raise CompositionError(
            "runner plan lacks the cumulative result commitment paths")
    return _append_authority_event(
        operation_root, kind="result",
        operation_key=authority["operation_key"], payload=payload)


def _load_runner_measurement_authority(operation_root: Path) -> dict[str, Any]:
    """Load authority only when the append-only journal still matches.

    The pre-run event is mandatory.  The result event is enforced as soon as
    it exists: in-run creation/binding happens before the result commitment
    is appended, while every later reopen re-derives the result payload from
    the sealed bytes and fails closed unless the journal still agrees.
    """
    authority, payload = _runner_measurement_authority_uncommitted(
        operation_root)
    rows = _read_authority_journal(operation_root)
    matches = [row for row in rows if row["kind"] == "pre_run"]
    if (len(matches) != 1
            or matches[0]["operation_key"] != authority["operation_key"]
            or matches[0]["payload"] != payload):
        raise CompositionError("pre-run authority differs from its journal")
    result_payload = _runner_result_payload_uncommitted(operation_root)
    result_matches = [row for row in rows if row["kind"] == "result"]
    if result_payload is None:
        if result_matches:
            raise CompositionError(
                "result journal exists without derivable result authority")
    elif result_matches and (len(result_matches) != 1
            or result_matches[0]["operation_key"] != authority["operation_key"]
            or result_matches[0]["payload"] != result_payload):
        raise CompositionError("result authority differs from its journal")
    return authority


def _load_exact_proof_bundle_authority(
        operation_root: Path, exact_ref: "MeasurementReceiptRef",
) -> dict[str, Any]:
    """Bind attribution bytes to the separately sealed proof bundle."""
    path = operation_root / "proof/proof-bundle.json"
    if path != path.resolve(strict=False):
        raise CompositionError("proof bundle path is not canonical")
    raw = _stable_receipt_bytes(path)
    try:
        wrapper = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompositionError("proof bundle is not strict JSON") from exc
    if (not isinstance(wrapper, dict)
            or wrapper.get("schema") !=
               "epyc.autokernel.gpu_source_evidence_bundle.v1"
            or wrapper.get("authority") !=
               "nonpromotable_candidate_only_discovery"
            or wrapper.get("promotion_claim") is not False
            or not isinstance(wrapper.get("bundle"), Mapping)):
        raise CompositionError("proof bundle authority changed")
    receipt = _require_sha(wrapper.get("receipt_sha256"),
                           "proof bundle receipt_sha256")
    if receipt != _sha({key: row for key, row in wrapper.items()
                        if key != "receipt_sha256"}):
        raise CompositionError("proof bundle native identity changed")
    bundle_raw = wrapper["bundle"]
    try:
        bundle = gpu_source_proofs.GpuSourceProofBundle(
            manifest_sha256=bundle_raw["manifest_sha256"],
            candidate=_carrier_build_identity(
                bundle_raw.get("candidate"), "proof bundle candidate"),
            anchor=_carrier_build_identity(
                bundle_raw.get("anchor"), "proof bundle anchor"),
            workload_sha256=bundle_raw["workload_sha256"],
            correctness=dict(bundle_raw["correctness"]),
            attribution=dict(bundle_raw["attribution"]),
            bundle_sha256=bundle_raw["bundle_sha256"],
        )
    except (KeyError, TypeError, ValueError,
            gpu_source_proofs.ProofError) as exc:
        raise CompositionError("proof bundle identity is malformed") from exc
    if bundle.to_dict() != bundle_raw:
        raise CompositionError("proof bundle projection changed")
    attribution = bundle.attribution
    exact = exact_ref.load()
    if (attribution.get("path") != exact_ref.path
            or attribution.get("file_sha256") != exact_ref.sha256
            or attribution.get("native_sha256") != exact.get("receipt_sha256")
            or attribution.get("body") != exact):
        raise CompositionError("proof bundle attribution binding changed")
    return {
        "candidate_identity": bundle.candidate,
        "anchor_identity": bundle.anchor,
        "exact_route_receipt_sha256": exact_ref.sha256,
        "expected_route_set_sha256": _exact_route_projection(
            exact)["expected_route_set_sha256"],
    }


def _load_measurement_receipt(reference: "MeasurementReceiptRef") \
        -> dict[str, Any]:
    raw = _stable_receipt_bytes(
        Path(reference.path), expected_sha256=reference.sha256)
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompositionError("measurement receipt is not strict JSON") from exc
    if not isinstance(value, dict):
        raise CompositionError("measurement receipt is not an object")
    native_key = ("receipt_sha256" if reference.role == "exact_route"
                  else "result_sha256")
    if (not _require_sha(value.get(native_key), native_key)
            or value[native_key] != _sha({
                key: row for key, row in value.items()
                if key != native_key})):
        raise CompositionError("measurement receipt native identity changed")
    return value


@dataclass(frozen=True)
class MeasurementReceiptRef:
    """Byte and canonical-location binding for one cumulative measurement."""

    role: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.role not in _MEASUREMENT_ROLES:
            raise CompositionError("measurement receipt role is invalid")
        path = Path(self.path)
        if (not isinstance(self.path, str) or not path.is_absolute()
                or ".." in path.parts):
            raise CompositionError("measurement receipt path is unsafe")
        _require_sha(self.sha256, "measurement receipt sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": "epyc.autokernel.cumulative_measurement_ref.v1",
            "role": self.role, "path": self.path, "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasurementReceiptRef":
        if (not isinstance(value, Mapping)
                or set(value) != {"schema", "role", "path", "sha256"}
                or value.get("schema") !=
                   "epyc.autokernel.cumulative_measurement_ref.v1"):
            raise CompositionError("measurement receipt ref has an inexact schema")
        return cls(
            role=value["role"], path=value["path"], sha256=value["sha256"])

    def canonical_location(self) -> tuple[Path, str | None]:
        path = Path(self.path)
        if self.role == "exact_route":
            if path.name != "attribution-pair.json" \
                    or path.parent.name != "proof":
                raise CompositionError(
                    "exact-route receipt is outside its canonical location")
            root = path.parent.parent
            repetition = None
        else:
            stage_names = {
                "incremental_graphs_off": "measurement-graphs-off",
                "incremental_graphs_on": "target-runtime-graphs-on",
                "production_graphs_on":
                    "cumulative-vs-production-graphs-on",
            }
            stage = path.parent
            repetition_dir = stage.parent
            runner = repetition_dir.parent
            root = runner.parent
            if (path.name != "result.json"
                    or stage.name != stage_names[self.role]
                    or runner.name != "runner"
                    or re.fullmatch(r"s[1-9][0-9]*", repetition_dir.name)
                       is None):
                raise CompositionError(
                    "measurement receipt is outside its canonical location")
            repetition = repetition_dir.name
        if (not root.is_absolute() or root.name == ""
                or path != path.resolve(strict=False)):
            raise CompositionError("measurement receipt path is not canonical")
        return root, repetition

    def load(self) -> dict[str, Any]:
        self.canonical_location()
        return _load_measurement_receipt(self)


_RUNTIME_ROW_FIELDS = (
    "n_threads", "n_batch", "n_ubatch", "use_mmap", "no_op_offload",
    "split_mode", "no_kv_offload", "poll", "n_prompt", "n_gen",
    "flash_attn",
)


def _carrier_build_identity(value: object, label: str) \
        -> gpu_source_proofs.BuildIdentity:
    if not isinstance(value, Mapping):
        raise CompositionError(f"{label} build identity is missing")
    try:
        return gpu_source_proofs.BuildIdentity(**dict(value))
    except (TypeError, ValueError, gpu_source_proofs.ProofError) as exc:
        raise CompositionError(f"{label} build identity is malformed") from exc


def _exact_route_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive every cumulative field duplicated from attribution bytes."""
    if (value.get("schema") !=
            "epyc.autokernel.gpu_kernel_attribution_pair.v2"
            or value.get("authority") !=
               "nonpromotable_candidate_only_discovery"
            or value.get("non_promotable") is not True
            or value.get("promotion_claim") is not False):
        raise CompositionError("exact-route carrier authority changed")
    comparison = value.get("exact_duration_comparison")
    required = {
        "candidate_routes", "anchor_routes", "candidate_total_duration_ns",
        "anchor_total_duration_ns", "relative_improvement_fraction",
        "direction", "all_candidate_routes_present",
        "all_anchor_routes_present", "statistic",
    }
    if not isinstance(comparison, Mapping) or set(comparison) != required:
        raise CompositionError("exact-route comparison is malformed")

    def total(rows: object, label: str) -> int:
        if not isinstance(rows, Mapping) or not rows:
            raise CompositionError(f"{label} exact routes are missing")
        values: list[int] = []
        for signature, row in rows.items():
            if (not isinstance(signature, str) or not signature
                    or not isinstance(row, Mapping)
                    or isinstance(row.get("total_duration_ns"), bool)
                    or not isinstance(row.get("total_duration_ns"), int)
                    or row["total_duration_ns"] <= 0
                    or isinstance(row.get("calls"), bool)
                    or not isinstance(row.get("calls"), int)
                    or row["calls"] <= 0):
                raise CompositionError(f"{label} exact route is malformed")
            values.append(row["total_duration_ns"])
        return sum(values)

    candidate_total = total(comparison["candidate_routes"], "candidate")
    anchor_total = total(comparison["anchor_routes"], "anchor")
    effect = (anchor_total - candidate_total) / anchor_total
    direction = ("improved" if candidate_total < anchor_total else
                 "regressed" if candidate_total > anchor_total else "neutral")
    if (comparison.get("candidate_total_duration_ns") != candidate_total
            or comparison.get("anchor_total_duration_ns") != anchor_total
            or comparison.get("relative_improvement_fraction") != effect
            or comparison.get("direction") != direction
            or comparison.get("all_candidate_routes_present") is not True
            or comparison.get("all_anchor_routes_present") is not True
            or comparison.get("statistic") !=
               "sum_exact_route_total_duration_ns"):
        raise CompositionError("exact-route effect is not derived from routes")
    return {
        "effect_fraction": effect,
        "expected_route_set_sha256": _sha(comparison["candidate_routes"]),
        "candidate_identity": _carrier_build_identity(
            value.get("candidate_build_identity"), "exact-route candidate"),
        "anchor_identity": _carrier_build_identity(
            value.get("anchor_build_identity"), "exact-route anchor"),
        "model_sha256": _require_sha(
            value.get("model_sha256"), "exact-route model_sha256"),
        "workload_sha256": _require_sha(
            value.get("workload_sha256"), "exact-route workload_sha256"),
        "runtime_config_sha256": _require_sha(
            value.get("runtime_config_sha256"),
            "exact-route runtime_config_sha256"),
    }


def _target_runtime_frame_sha256(value: Mapping[str, Any]) -> str:
    """Derive the target frame solely from fields fixed before execution."""
    frame = value.get("frame")
    runs = value.get("candidate_runs")
    identity = _carrier_build_identity(
        value.get("candidate_identity"), "target runtime candidate")
    if (not isinstance(frame, Mapping)
            or not isinstance(runs, list) or len(runs) != 1
            or not isinstance(runs[0], Mapping)
            or not isinstance(runs[0].get("raw_row"), Mapping)):
        raise CompositionError("target runtime frame is malformed")
    raw_row = runs[0]["raw_row"]
    if any(key not in raw_row for key in _RUNTIME_ROW_FIELDS):
        raise CompositionError("target runtime configuration is incomplete")
    return _sha({
        "schema": "epyc.autokernel.target_runtime_frame.v1",
        "runtime_graphs": value.get("runtime_graphs"),
        "factor_name": value.get("sole_factor", {}).get("name")
            if isinstance(value.get("sole_factor"), Mapping) else None,
        "frame": {
            key: frame.get(key) for key in (
                "backend", "recipe", "metric", "metric_direction",
                "n_prompt", "n_gen", "model_sha256", "cpu_list",
                "device", "architecture")
        },
        "runtime": {key: raw_row[key] for key in _RUNTIME_ROW_FIELDS},
        "candidate_identity": asdict(identity),
    })


def planned_target_runtime_frame_sha256(
        value: Mapping[str, Any], *,
        candidate_identity: gpu_source_proofs.BuildIdentity,
) -> str:
    """Derive the same target frame from a sealed preflight plan."""
    if not isinstance(candidate_identity, gpu_source_proofs.BuildIdentity):
        raise CompositionError("planned target candidate identity is untyped")
    required = {
        "frame", "metric", "prompt_tokens", "generation_tokens",
        "model_sha256", "runtime_graphs", "sole_factor",
    }
    if not required.issubset(value):
        raise CompositionError("planned target runtime frame is incomplete")
    runtime = {
        "n_threads": value.get("candidate_threads"),
        "n_batch": value.get("candidate_batch"),
        "n_ubatch": value.get("candidate_ubatch"),
        "use_mmap": value.get("candidate_mmap"),
        "no_op_offload": int(bool(value.get("candidate_no_op_offload"))),
        "split_mode": value.get("candidate_split_mode"),
        "no_kv_offload": value.get("candidate_no_kv_offload"),
        "poll": value.get("candidate_poll"),
        "n_prompt": value.get("prompt_tokens"),
        "n_gen": value.get("generation_tokens"),
        "flash_attn": int(bool(value.get("candidate_flash_attention"))),
    }
    return _sha({
        "schema": "epyc.autokernel.target_runtime_frame.v1",
        "runtime_graphs": value.get("runtime_graphs"),
        "factor_name": value.get("sole_factor", {}).get("name")
            if isinstance(value.get("sole_factor"), Mapping) else None,
        "frame": {
            "backend": "llama_gpu", "recipe": value.get("frame"),
            "metric": value.get("metric"),
            "metric_direction": "higher_better",
            "n_prompt": value.get("prompt_tokens"),
            "n_gen": value.get("generation_tokens"),
            "model_sha256": value.get("model_sha256"),
            "cpu_list": "184-191", "device": "AMD Instinct MI210",
            "architecture": "gfx90a",
        },
        "runtime": runtime,
        "candidate_identity": asdict(candidate_identity),
    })


def _runner_projection(value: Mapping[str, Any], *, graph_mode: str,
                       factor_name: str) -> dict[str, Any]:
    """Re-derive effects, frames, protocol, and identities from one run."""
    if (value.get("schema") !=
            "epyc.autokernel.gpu_candidate_only_screen.v2"
            or value.get("authority") !=
               "nonpromotable_candidate_only_discovery"
            or value.get("non_promotable") is not True
            or value.get("promotion_claim") is not False
            or value.get("hip_residency_proved") is not True
            or value.get("runtime_graphs") != graph_mode
            or not isinstance(value.get("sole_factor"), Mapping)
            or value["sole_factor"].get("name") != factor_name):
        raise CompositionError("runner measurement carrier authority changed")
    frame = value.get("frame")
    required_frame = {
        "backend", "recipe", "metric", "metric_direction",
        "metric_contract", "n_prompt", "n_gen", "model",
        "model_sha256", "source_commit", "cpu_list", "device",
        "architecture",
    }
    metric_contract = frame.get("metric_contract") \
        if isinstance(frame, Mapping) else None
    if (not isinstance(frame, Mapping) or set(frame) != required_frame
            or not isinstance(metric_contract, Mapping)
            or metric_contract.get("graph_mode") not in {
                graph_mode, "disabled_for_integrity"}
            or frame.get("metric_direction") != "higher_better"):
        raise CompositionError("runner metric/frame authority changed")

    def arm_runs(rows: object, label: str) -> dict[str, Any]:
        if (not isinstance(rows, list) or len(rows) != 1
                or not isinstance(rows[0], Mapping)):
            raise CompositionError(f"runner {label} runs are malformed")
        run = rows[0]
        required = {
            "metric", "samples", "metric_contract", "sample_count",
            "raw_row", "reward_binary_sha256", "hip_library_sha256",
            "native_metric_diagnostic", "supervisor",
        }
        if not required.issubset(run):
            raise CompositionError(f"runner {label} run is incomplete")
        samples = run.get("samples")
        if (not isinstance(samples, list) or len(samples) != 9
                or run.get("sample_count") != len(samples)
                or run.get("metric_contract") != metric_contract):
            raise CompositionError(f"runner {label} samples are malformed")
        observed = tuple(
            _finite(row, f"runner {label} sample") for row in samples)
        if any(row <= 0 for row in observed) \
                or _finite(run.get("metric"), f"runner {label} metric") <= 0:
            raise CompositionError(f"runner {label} samples must be positive")
        binary_sha256 = _require_sha(
            run.get("reward_binary_sha256"), f"runner {label} binary")
        hip_sha256 = _require_sha(
            run.get("hip_library_sha256"), f"runner {label} HIP library")
        diagnostic = run.get("native_metric_diagnostic")
        if not isinstance(diagnostic, Mapping):
            raise CompositionError(f"runner {label} native metric is missing")
        native_sha = _require_sha(
            diagnostic.get("receipt_sha256"),
            f"runner {label} native metric receipt")
        if native_sha != _sha({key: row for key, row in diagnostic.items()
                               if key != "receipt_sha256"}):
            raise CompositionError(
                f"runner {label} native metric identity changed")
        supervisor = run.get("supervisor")
        if not isinstance(supervisor, Mapping):
            raise CompositionError(f"runner {label} supervisor is missing")
        _require_sha(supervisor.get("stdout_sha256"),
                     f"runner {label} stdout")
        _require_sha(supervisor.get("stderr_sha256"),
                     f"runner {label} stderr")
        raw_row = run.get("raw_row")
        if (not isinstance(raw_row, Mapping)
                or any(key not in raw_row for key in _RUNTIME_ROW_FIELDS)):
            raise CompositionError(
                f"runner {label} runtime configuration is incomplete")
        return {
            "samples": observed, "raw_row": raw_row,
            "binary_sha256": binary_sha256,
            "hip_library_sha256": hip_sha256,
        }

    anchor_run = arm_runs(value.get("anchor_runs"), "anchor")
    candidate_run = arm_runs(
        value.get("candidate_runs"), "candidate")
    anchor_samples = anchor_run["samples"]
    candidate_samples = candidate_run["samples"]
    if value.get("anchor_samples") != list(anchor_samples) \
            or value.get("candidate_samples") != list(candidate_samples):
        raise CompositionError("runner flattened samples changed")
    center = (float(value["anchor_runs"][0]["metric"])
              if metric_contract.get("schema") ==
                 "epyc.autokernel.serialized_pair_max_metric.v1"
              else sum(anchor_samples) / len(anchor_samples))
    if value.get("baseline_center") != center:
        raise CompositionError("runner baseline center is not derived from anchor runs")
    observed = candidate_samples
    effects = tuple((row - center) / center for row in observed)
    measured = median(effects)
    if (value.get("relative_effects") != list(effects)
            or value.get("median_relative") != measured):
        raise CompositionError("measurement effect is not derived from samples")
    candidate_identity = _carrier_build_identity(
        value.get("candidate_identity"), "runner candidate")
    anchor_identity = _carrier_build_identity(
        value.get("anchor_identity"), "runner anchor")
    if (candidate_run["binary_sha256"] != candidate_identity.binary_sha256
            or candidate_run["hip_library_sha256"] !=
               candidate_identity.hip_library_sha256
            or anchor_run["binary_sha256"] != anchor_identity.binary_sha256
            or anchor_run["hip_library_sha256"] !=
               anchor_identity.hip_library_sha256):
        raise CompositionError("runner native artifacts differ from build identities")
    candidate_raw = candidate_run["raw_row"]
    anchor_raw = anchor_run["raw_row"]
    runtime = {key: candidate_raw[key] for key in _RUNTIME_ROW_FIELDS}
    anchor_runtime = {key: anchor_raw[key] for key in _RUNTIME_ROW_FIELDS}
    if (frame.get("source_commit") != candidate_identity.source_commit
            or anchor_runtime != runtime):
        raise CompositionError("runner metric/frame authority changed")
    if (value.get("candidate_invocations") != 9
            or value.get("anchor_invocations") != 9
            or value.get("candidate_processes") != 1
            or value.get("anchor_processes") != 1):
        raise CompositionError("runner execution cardinality changed")
    model_sha256 = _require_sha(frame.get("model_sha256"), "model_sha256")
    metric = _require_text(frame.get("metric"), "metric")
    workload = {
        "backend": frame["backend"], "recipe": frame["recipe"],
        "n_prompt": frame["n_prompt"], "n_gen": frame["n_gen"],
    }
    runtime_config_sha256 = _sha(runtime)
    protocol = {
        **workload, "model_sha256": model_sha256,
        "metric": metric, "metric_direction": frame["metric_direction"],
        "cpu_list": frame["cpu_list"], "device": frame["device"],
        "architecture": frame["architecture"],
        "runtime_config_sha256": runtime_config_sha256,
        "graphs_mode": graph_mode,
        "candidate_invocations": value["candidate_invocations"],
        "candidate_processes": value["candidate_processes"],
    }
    frame_base = {
        "schema": "epyc.autokernel.measurement_arm_frame.v1",
        "protocol": protocol, "factor_name": factor_name,
    }
    return {
        "effect_fraction": measured,
        "candidate_identity": candidate_identity,
        "anchor_identity": anchor_identity,
        "candidate_frame_sha256": _sha({
            **frame_base, "arm": "candidate",
            "source_commit": candidate_identity.source_commit,
            "build_identity": asdict(candidate_identity),
        }),
        "anchor_frame_sha256": _sha({
            **frame_base, "arm": "anchor",
            "source_commit": anchor_identity.source_commit,
            "build_identity": asdict(anchor_identity),
        }),
        "target_runtime_frame_sha256":
            _target_runtime_frame_sha256(value),
        "protocol_frame_sha256": _sha(protocol),
        "model_sha256": model_sha256,
        "workload_sha256": _sha(workload),
        "runtime_config_sha256": runtime_config_sha256,
        "metric": metric,
        "metric_direction": frame["metric_direction"],
    }


def _runner_effect(value: Mapping[str, Any], *, graph_mode: str,
                   factor_name: str) -> float:
    return _runner_projection(
        value, graph_mode=graph_mode,
        factor_name=factor_name)["effect_fraction"]


@dataclass(frozen=True)
class IncrementalComparison:
    operation_key: str
    build_pair_sha256: str
    correctness_result_sha256: str
    exact_route_receipt_sha256: str
    exact_route_receipt_ref: MeasurementReceiptRef
    expected_route_set_sha256: str
    graphs_off_receipt_sha256: str
    graphs_off_receipt_ref: MeasurementReceiptRef
    graphs_on_receipt_sha256: str
    graphs_on_receipt_ref: MeasurementReceiptRef
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
            exact_route_receipt_path: Path | str,
            graphs_off_receipt_path: Path | str,
            graphs_on_receipt_path: Path | str,
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
        exact_ref = MeasurementReceiptRef(
            role="exact_route", path=str(exact_route_receipt_path),
            sha256=exact_route_receipt_sha256)
        off_ref = MeasurementReceiptRef(
            role="incremental_graphs_off",
            path=str(graphs_off_receipt_path),
            sha256=graphs_off_receipt_sha256)
        on_ref = MeasurementReceiptRef(
            role="incremental_graphs_on",
            path=str(graphs_on_receipt_path),
            sha256=graphs_on_receipt_sha256)
        if route > 0 and graphs_off > 0 and graphs > 0:
            classification = "candidate"
        elif route <= 0 and graphs_off <= 0 and graphs <= 0:
            classification = "screened_out"
        else:
            classification = "inconclusive"
        body = {
            "schema": "epyc.autokernel.incremental_composition_comparison.v3",
            "operation_key": pair.operation_key,
            "build_pair_sha256": pair.pair_sha256,
            "correctness_result_sha256": correctness.result_sha256,
            "exact_route_receipt_sha256": _require_sha(
                exact_route_receipt_sha256, "exact_route_receipt_sha256"),
            "exact_route_receipt_ref": exact_ref.to_dict(),
            "expected_route_set_sha256": _require_sha(
                expected_route_set_sha256, "expected_route_set_sha256"),
            "graphs_off_receipt_sha256": _require_sha(
                graphs_off_receipt_sha256, "graphs_off_receipt_sha256"),
            "graphs_off_receipt_ref": off_ref.to_dict(),
            "graphs_on_receipt_sha256": _require_sha(
                graphs_on_receipt_sha256, "graphs_on_receipt_sha256"),
            "graphs_on_receipt_ref": on_ref.to_dict(),
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
        created = cls(
            operation_key=pair.operation_key,
            build_pair_sha256=pair.pair_sha256,
            correctness_result_sha256=correctness.result_sha256,
            exact_route_receipt_sha256=exact_route_receipt_sha256,
            exact_route_receipt_ref=exact_ref,
            expected_route_set_sha256=expected_route_set_sha256,
            graphs_off_receipt_sha256=graphs_off_receipt_sha256,
            graphs_off_receipt_ref=off_ref,
            graphs_on_receipt_sha256=graphs_on_receipt_sha256,
            graphs_on_receipt_ref=on_ref,
            target_runtime_frame_sha256=target_runtime_frame_sha256,
            exact_route_effect_fraction=route,
            graphs_off_effect_fraction=graphs_off,
            graphs_on_effect_fraction=graphs,
            classification=classification, result_sha256=_sha(body),
        )
        created.bind(pair, correctness)
        return created

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.incremental_composition_comparison.v3",
            "operation_key": self.operation_key,
            "build_pair_sha256": self.build_pair_sha256,
            "correctness_result_sha256": self.correctness_result_sha256,
            "exact_route_receipt_sha256": self.exact_route_receipt_sha256,
            "exact_route_receipt_ref": self.exact_route_receipt_ref.to_dict(),
            "expected_route_set_sha256": self.expected_route_set_sha256,
            "graphs_off_receipt_sha256": self.graphs_off_receipt_sha256,
            "graphs_off_receipt_ref": self.graphs_off_receipt_ref.to_dict(),
            "graphs_on_receipt_sha256": self.graphs_on_receipt_sha256,
            "graphs_on_receipt_ref": self.graphs_on_receipt_ref.to_dict(),
            "target_runtime_frame_sha256": self.target_runtime_frame_sha256,
            "exact_route_effect_fraction": self.exact_route_effect_fraction,
            "graphs_off_effect_fraction": self.graphs_off_effect_fraction,
            "graphs_on_effect_fraction": self.graphs_on_effect_fraction,
            "classification": self.classification,
            "exact_route_executed": True, "graphs_off_executed": True,
            "graphs_on_executed": True,
        }

    def __post_init__(self) -> None:
        if (not isinstance(self.exact_route_receipt_ref, MeasurementReceiptRef)
                or not isinstance(
                    self.graphs_off_receipt_ref, MeasurementReceiptRef)
                or not isinstance(
                    self.graphs_on_receipt_ref, MeasurementReceiptRef)):
            raise CompositionError("incremental measurement refs are untyped")
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
        exact_root, exact_repetition = \
            self.exact_route_receipt_ref.canonical_location()
        off_root, off_repetition = \
            self.graphs_off_receipt_ref.canonical_location()
        on_root, on_repetition = \
            self.graphs_on_receipt_ref.canonical_location()
        if (self.exact_route_receipt_ref.role != "exact_route"
                or self.graphs_off_receipt_ref.role !=
                   "incremental_graphs_off"
                or self.graphs_on_receipt_ref.role !=
                   "incremental_graphs_on"
                or exact_root.name != self.operation_key
                or len({exact_root, off_root, on_root}) != 1
                or exact_repetition is not None
                or off_repetition != on_repetition
                or self.exact_route_receipt_sha256 !=
                   self.exact_route_receipt_ref.sha256
                or self.graphs_off_receipt_sha256 !=
                   self.graphs_off_receipt_ref.sha256
                or self.graphs_on_receipt_sha256 !=
                   self.graphs_on_receipt_ref.sha256):
            raise CompositionError(
                "incremental measurement locations or hashes changed")
        exact = _exact_route_projection(
            self.exact_route_receipt_ref.load())
        off = _runner_projection(
            self.graphs_off_receipt_ref.load(), graph_mode="off",
            factor_name="source_patch")
        on = _runner_projection(
            self.graphs_on_receipt_ref.load(), graph_mode="on",
            factor_name="source_patch")
        authority = _load_runner_measurement_authority(exact_root)
        proof_authority = _load_exact_proof_bundle_authority(
            exact_root, self.exact_route_receipt_ref)
        common = (
            "candidate_identity", "anchor_identity", "model_sha256",
            "workload_sha256", "runtime_config_sha256", "metric",
            "metric_direction",
        )
        expected_authority = {
            "operation_key": self.operation_key,
            "build_pair_sha256": self.build_pair_sha256,
            "correctness_result_sha256": self.correctness_result_sha256,
            "exact_route_receipt_sha256":
                self.exact_route_receipt_sha256,
            "expected_route_set_sha256": self.expected_route_set_sha256,
            "target_runtime_frame_sha256":
                self.target_runtime_frame_sha256,
        }
        if any(authority.get(key) != value
               for key, value in expected_authority.items()):
            raise CompositionError(
                "incremental pre-run measurement authority changed")
        if (proof_authority["candidate_identity"] !=
                exact["candidate_identity"]
                or proof_authority["anchor_identity"] !=
                   exact["anchor_identity"]
                or proof_authority["exact_route_receipt_sha256"] !=
                   self.exact_route_receipt_sha256
                or proof_authority["expected_route_set_sha256"] !=
                   self.expected_route_set_sha256):
            raise CompositionError(
                "incremental proof-bundle authority changed")
        if ((route, graphs_off, graphs) != (
                    exact["effect_fraction"], off["effect_fraction"],
                    on["effect_fraction"])
                or self.expected_route_set_sha256 !=
                   exact["expected_route_set_sha256"]
                or self.target_runtime_frame_sha256 !=
                   on["target_runtime_frame_sha256"]
                or exact["candidate_identity"] != off["candidate_identity"]
                or exact["anchor_identity"] != off["anchor_identity"]
                or exact["model_sha256"] != off["model_sha256"]
                or any(off[field] != on[field] for field in common)
                or self.classification != expected
                or self.result_sha256 != _sha(self._body())):
            raise CompositionError("incremental comparison identity changed")

    def bind(
            self, pair: CumulativeBuildPair,
            correctness: FullCorrectness,
    ) -> None:
        correctness.bind_pair(pair)
        exact = _exact_route_projection(
            self.exact_route_receipt_ref.load())
        off = _runner_projection(
            self.graphs_off_receipt_ref.load(), graph_mode="off",
            factor_name="source_patch")
        on = _runner_projection(
            self.graphs_on_receipt_ref.load(), graph_mode="on",
            factor_name="source_patch")
        authority = _load_runner_measurement_authority(self.operation_root)
        if (self.operation_key != pair.operation_key
                or self.build_pair_sha256 != pair.pair_sha256
                or self.correctness_result_sha256 !=
                   correctness.result_sha256
                or exact["candidate_identity"] != pair.candidate.build_identity
                or exact["anchor_identity"] != pair.anchor.build_identity
                or off["candidate_identity"] != pair.candidate.build_identity
                or off["anchor_identity"] != pair.anchor.build_identity
                or on["candidate_identity"] != pair.candidate.build_identity
                or on["anchor_identity"] != pair.anchor.build_identity):
            raise CompositionError(
                "incremental comparison binds other build evidence")
        if authority["build_pair_sha256"] != pair.pair_sha256 \
                or authority["correctness_result_sha256"] != \
                   correctness.result_sha256:
            raise CompositionError(
                "incremental measurement authority binds other evidence")

    @property
    def operation_root(self) -> Path:
        return self.exact_route_receipt_ref.canonical_location()[0]

    @property
    def repetition(self) -> str:
        repetition = self.graphs_on_receipt_ref.canonical_location()[1]
        if repetition is None:
            raise CompositionError("incremental repetition is unavailable")
        return repetition

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
            "exact_route_receipt_ref",
            "graphs_off_receipt_sha256",
            "graphs_off_receipt_ref",
            "expected_route_set_sha256", "graphs_on_receipt_sha256",
            "graphs_on_receipt_ref",
            "target_runtime_frame_sha256", "exact_route_effect_fraction",
            "graphs_off_effect_fraction",
            "graphs_on_effect_fraction", "classification",
            "exact_route_executed", "graphs_off_executed",
            "graphs_on_executed", "result_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("incremental comparison has an inexact schema")
        if (value.get("schema") !=
                "epyc.autokernel.incremental_composition_comparison.v3"
                or value.get("exact_route_executed") is not True
                or value.get("graphs_off_executed") is not True
                or value.get("graphs_on_executed") is not True):
            raise CompositionError("incremental comparison authority changed")
        kwargs = {key: value[key] for key in (
            "operation_key", "build_pair_sha256", "correctness_result_sha256",
            "exact_route_receipt_sha256", "expected_route_set_sha256",
            "graphs_off_receipt_sha256",
            "graphs_on_receipt_sha256", "target_runtime_frame_sha256",
            "exact_route_effect_fraction", "graphs_on_effect_fraction",
            "graphs_off_effect_fraction",
            "classification", "result_sha256")}
        kwargs["exact_route_receipt_ref"] = MeasurementReceiptRef.from_dict(
            value["exact_route_receipt_ref"])
        kwargs["graphs_off_receipt_ref"] = MeasurementReceiptRef.from_dict(
            value["graphs_off_receipt_ref"])
        kwargs["graphs_on_receipt_ref"] = MeasurementReceiptRef.from_dict(
            value["graphs_on_receipt_ref"])
        return cls(**kwargs)


@dataclass(frozen=True)
class FrozenProductionComparator:
    branch: str
    commit: str
    build_identity: gpu_source_proofs.BuildIdentity
    build_receipt_sha256: str
    linkage_receipt_sha256: str
    runtime_receipt_sha256: str
    runtime_snapshot_sha256: str
    measurement_receipt_sha256: str
    model_sha256: str
    workload_sha256: str
    runtime_config_sha256: str
    observed_workload_sha256: str
    observed_runtime_config_sha256: str
    frame_sha256: str
    graphs_mode: str
    metric: str
    direction: str
    measurement_protocol_sha256: str
    receipt_sha256: str

    @classmethod
    def create(
            cls, *, build_identity: gpu_source_proofs.BuildIdentity,
            build_receipt_sha256: str, linkage_receipt_sha256: str,
            runtime_receipt_sha256: str, runtime_snapshot_sha256: str,
            measurement_receipt_sha256: str, model_sha256: str,
            workload_sha256: str, runtime_config_sha256: str,
            observed_workload_sha256: str,
            observed_runtime_config_sha256: str,
            frame_sha256: str, measurement_protocol_sha256: str,
    ) -> "FrozenProductionComparator":
        values = {
            "branch": FROZEN_PRODUCTION_BRANCH,
            "commit": FROZEN_PRODUCTION_COMMIT,
            "build_identity": build_identity,
            "build_receipt_sha256": build_receipt_sha256,
            "linkage_receipt_sha256": linkage_receipt_sha256,
            "runtime_receipt_sha256": runtime_receipt_sha256,
            "runtime_snapshot_sha256": runtime_snapshot_sha256,
            "measurement_receipt_sha256": measurement_receipt_sha256,
            "model_sha256": model_sha256,
            "workload_sha256": workload_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "observed_workload_sha256": observed_workload_sha256,
            "observed_runtime_config_sha256": observed_runtime_config_sha256,
            "frame_sha256": frame_sha256,
            "graphs_mode": "graphs_on",
            "metric": "tokens_per_second",
            "direction": "higher_is_better",
            "measurement_protocol_sha256": measurement_protocol_sha256,
        }
        body = {
            "schema": "epyc.autokernel.frozen_production_comparator.v2",
            **{key: (asdict(value)
                     if isinstance(value, gpu_source_proofs.BuildIdentity)
                     else value)
               for key, value in values.items()},
        }
        return cls(**values, receipt_sha256=_sha(body))

    def __post_init__(self) -> None:
        if (self.branch != FROZEN_PRODUCTION_BRANCH
                or self.commit != FROZEN_PRODUCTION_COMMIT
                or not isinstance(self.build_identity,
                                  gpu_source_proofs.BuildIdentity)
                or self.build_identity.source_commit != self.commit
                or self.graphs_mode != "graphs_on"
                or self.metric != "tokens_per_second"
                or self.direction != "higher_is_better"):
            raise CompositionError(
                "frozen production comparator protocol is not exact v9")
        for label in (
                "build_receipt_sha256", "linkage_receipt_sha256",
                "runtime_receipt_sha256", "runtime_snapshot_sha256",
                "measurement_receipt_sha256", "model_sha256",
                "workload_sha256", "runtime_config_sha256", "frame_sha256",
                "observed_workload_sha256",
                "observed_runtime_config_sha256",
                "measurement_protocol_sha256", "receipt_sha256"):
            _require_sha(getattr(self, label), label)
        if self.receipt_sha256 != _sha(self._body()):
            raise CompositionError(
                "frozen production comparator self-hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.frozen_production_comparator.v2",
            **{key: (asdict(value)
                     if isinstance(value, gpu_source_proofs.BuildIdentity)
                     else value)
               for key, value in asdict(self).items()
               if key != "receipt_sha256"},
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) \
            -> "FrozenProductionComparator":
        fields = {
            "branch", "commit", "build_identity", "build_receipt_sha256",
            "linkage_receipt_sha256", "runtime_receipt_sha256",
            "runtime_snapshot_sha256", "measurement_receipt_sha256",
            "model_sha256", "workload_sha256", "runtime_config_sha256",
            "observed_workload_sha256", "observed_runtime_config_sha256",
            "frame_sha256", "graphs_mode", "metric", "direction",
            "measurement_protocol_sha256", "receipt_sha256",
        }
        if (not isinstance(value, Mapping)
                or set(value) != fields | {"schema"}
                or value.get("schema") !=
                   "epyc.autokernel.frozen_production_comparator.v2"):
            raise CompositionError(
                "frozen production comparator has an inexact schema")
        try:
            identity = gpu_source_proofs.BuildIdentity(
                **dict(value["build_identity"]))
        except (TypeError, ValueError, gpu_source_proofs.ProofError) as exc:
            raise CompositionError(
                "frozen production comparator build identity is invalid") \
                from exc
        return cls(**{
            **{key: value[key] for key in fields - {"build_identity"}},
            "build_identity": identity,
        })

    def authority(self) -> "FrozenProductionAuthority":
        return FrozenProductionAuthority.create(
            production_commit=self.commit, build_identity=self.build_identity,
            runtime_snapshot_sha256=self.runtime_snapshot_sha256,
            comparator_receipt_sha256=self.receipt_sha256,
            graphs_mode=self.graphs_mode, frame_sha256=self.frame_sha256,
            measurement_protocol_sha256=self.measurement_protocol_sha256,
            measurement_receipt_sha256=self.measurement_receipt_sha256,
            model_sha256=self.model_sha256,
            workload_sha256=self.workload_sha256,
            runtime_config_sha256=self.runtime_config_sha256,
            observed_workload_sha256=self.observed_workload_sha256,
            observed_runtime_config_sha256=
                self.observed_runtime_config_sha256,
            metric=self.metric, direction=self.direction)


@dataclass(frozen=True)
class FrozenProductionAuthority:
    """Exact executable projection of the immutable deployment comparator."""

    production_commit: str
    build_identity: gpu_source_proofs.BuildIdentity
    build_identity_sha256: str
    runtime_snapshot_sha256: str
    comparator_receipt_sha256: str
    graphs_mode: str
    frame_sha256: str
    measurement_protocol_sha256: str
    measurement_receipt_sha256: str
    model_sha256: str
    workload_sha256: str
    runtime_config_sha256: str
    observed_workload_sha256: str
    observed_runtime_config_sha256: str
    metric: str
    direction: str
    authority_sha256: str

    @classmethod
    def create(
            cls, *, production_commit: str,
            build_identity: gpu_source_proofs.BuildIdentity,
            runtime_snapshot_sha256: str,
            comparator_receipt_sha256: str,
            graphs_mode: str,
            frame_sha256: str,
            measurement_protocol_sha256: str,
            measurement_receipt_sha256: str,
            model_sha256: str, workload_sha256: str,
            runtime_config_sha256: str, metric: str, direction: str,
            observed_workload_sha256: str,
            observed_runtime_config_sha256: str,
    ) -> "FrozenProductionAuthority":
        if not isinstance(build_identity, gpu_source_proofs.BuildIdentity):
            raise CompositionError("frozen production build identity must be typed")
        body = cls._body_for(
            production_commit, build_identity, runtime_snapshot_sha256,
            comparator_receipt_sha256, graphs_mode, frame_sha256,
            measurement_protocol_sha256, measurement_receipt_sha256,
            model_sha256, workload_sha256, runtime_config_sha256,
            metric, direction, observed_workload_sha256,
            observed_runtime_config_sha256)
        return cls(
            production_commit=production_commit,
            build_identity=build_identity,
            build_identity_sha256=_sha(asdict(build_identity)),
            runtime_snapshot_sha256=runtime_snapshot_sha256,
            comparator_receipt_sha256=comparator_receipt_sha256,
            graphs_mode=graphs_mode, frame_sha256=frame_sha256,
            measurement_protocol_sha256=measurement_protocol_sha256,
            measurement_receipt_sha256=measurement_receipt_sha256,
            model_sha256=model_sha256, workload_sha256=workload_sha256,
            runtime_config_sha256=runtime_config_sha256,
            observed_workload_sha256=observed_workload_sha256,
            observed_runtime_config_sha256=observed_runtime_config_sha256,
            metric=metric, direction=direction,
            authority_sha256=_sha(body))

    @staticmethod
    def _body_for(
            production_commit: str,
            build_identity: gpu_source_proofs.BuildIdentity,
            runtime_snapshot_sha256: str,
            comparator_receipt_sha256: str,
            graphs_mode: str, frame_sha256: str,
            measurement_protocol_sha256: str,
            measurement_receipt_sha256: str,
            model_sha256: str, workload_sha256: str,
            runtime_config_sha256: str, metric: str, direction: str,
            observed_workload_sha256: str,
            observed_runtime_config_sha256: str,
    ) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.frozen_production_authority.v2",
            "production_commit": _require_commit(
                production_commit, "production_commit"),
            "build_identity": asdict(build_identity),
            "build_identity_sha256": _sha(asdict(build_identity)),
            "runtime_snapshot_sha256": _require_sha(
                runtime_snapshot_sha256, "runtime_snapshot_sha256"),
            "comparator_receipt_sha256": _require_sha(
                comparator_receipt_sha256, "comparator_receipt_sha256"),
            "graphs_mode": graphs_mode,
            "frame_sha256": _require_sha(frame_sha256, "frame_sha256"),
            "measurement_protocol_sha256": _require_sha(
                measurement_protocol_sha256,
                "measurement_protocol_sha256"),
            "measurement_receipt_sha256": _require_sha(
                measurement_receipt_sha256,
                "measurement_receipt_sha256"),
            "model_sha256": _require_sha(model_sha256, "model_sha256"),
            "workload_sha256": _require_sha(
                workload_sha256, "workload_sha256"),
            "runtime_config_sha256": _require_sha(
                runtime_config_sha256, "runtime_config_sha256"),
            "observed_workload_sha256": _require_sha(
                observed_workload_sha256, "observed_workload_sha256"),
            "observed_runtime_config_sha256": _require_sha(
                observed_runtime_config_sha256,
                "observed_runtime_config_sha256"),
            "metric": metric, "direction": direction,
        }

    def __post_init__(self) -> None:
        if not isinstance(self.build_identity, gpu_source_proofs.BuildIdentity):
            raise CompositionError("frozen production build identity must be typed")
        body = self._body_for(
            self.production_commit, self.build_identity,
            self.runtime_snapshot_sha256,
            self.comparator_receipt_sha256, self.graphs_mode,
            self.frame_sha256, self.measurement_protocol_sha256,
            self.measurement_receipt_sha256, self.model_sha256,
            self.workload_sha256, self.runtime_config_sha256,
            self.metric, self.direction, self.observed_workload_sha256,
            self.observed_runtime_config_sha256)
        if (self.production_commit != FROZEN_PRODUCTION_COMMIT
                or self.build_identity.source_commit != self.production_commit
                or self.graphs_mode != "graphs_on"
                or self.metric != "tokens_per_second"
                or self.direction != "higher_is_better"
                or self.build_identity_sha256 !=
                   _sha(asdict(self.build_identity))
                or self.authority_sha256 != _sha(body)):
            raise CompositionError("frozen production authority identity changed")

    def bind_plan(self, plan: CompositionPlan) -> None:
        if self.production_commit != plan.candidate.production_base_commit:
            raise CompositionError(
                "frozen production authority names another source era")

    def to_dict(self) -> dict[str, Any]:
        return {**self._body_for(
            self.production_commit, self.build_identity,
            self.runtime_snapshot_sha256,
            self.comparator_receipt_sha256, self.graphs_mode,
            self.frame_sha256, self.measurement_protocol_sha256,
            self.measurement_receipt_sha256, self.model_sha256,
            self.workload_sha256, self.runtime_config_sha256,
            self.metric, self.direction, self.observed_workload_sha256,
            self.observed_runtime_config_sha256),
            "authority_sha256": self.authority_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenProductionAuthority":
        required = {
            "schema", "production_commit", "build_identity",
            "build_identity_sha256", "runtime_snapshot_sha256",
            "comparator_receipt_sha256", "graphs_mode", "frame_sha256",
            "measurement_protocol_sha256",
            "measurement_receipt_sha256", "model_sha256",
            "workload_sha256", "runtime_config_sha256", "metric",
            "observed_workload_sha256", "observed_runtime_config_sha256",
            "direction",
            "authority_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError(
                "frozen production authority has an inexact schema")
        if value.get("schema") != \
                "epyc.autokernel.frozen_production_authority.v2":
            raise CompositionError("frozen production authority schema changed")
        try:
            identity = gpu_source_proofs.BuildIdentity(
                **dict(value["build_identity"]))
        except (TypeError, ValueError, gpu_source_proofs.ProofError) as exc:
            raise CompositionError(
                "frozen production build identity is invalid") from exc
        return cls(
            production_commit=value["production_commit"],
            build_identity=identity,
            build_identity_sha256=value["build_identity_sha256"],
            runtime_snapshot_sha256=value["runtime_snapshot_sha256"],
            comparator_receipt_sha256=value["comparator_receipt_sha256"],
            graphs_mode=value["graphs_mode"],
            frame_sha256=value["frame_sha256"],
            measurement_protocol_sha256=
                value["measurement_protocol_sha256"],
            measurement_receipt_sha256=value["measurement_receipt_sha256"],
            model_sha256=value["model_sha256"],
            workload_sha256=value["workload_sha256"],
            runtime_config_sha256=value["runtime_config_sha256"],
            observed_workload_sha256=value["observed_workload_sha256"],
            observed_runtime_config_sha256=
                value["observed_runtime_config_sha256"],
            metric=value["metric"], direction=value["direction"],
            authority_sha256=value["authority_sha256"])


_TERMINAL_RECEIPT_EDGE = frozenset({
    "cumulative_performance", "cumulative_performance_ref",
    "cumulative_performance_result_sha256", "terminal_sha256",
})


def _terminal_decision_sha256(value: Mapping[str, Any]) -> str:
    """Hash terminal science/decision while excluding its cyclic receipt edge."""
    return _sha({key: row for key, row in value.items()
                 if key not in _TERMINAL_RECEIPT_EDGE})


@dataclass(frozen=True)
class CumulativePerformance:
    """Promotion authority for an accepted stack versus frozen production.

    The incremental comparison answers whether the newest lever belongs in the
    stack.  This distinct receipt answers whether that resulting stack is
    promotable versus exact frozen production.  Effects are never multiplied,
    added, or otherwise inferred from prior incremental results.
    """

    operation_key: str
    plan_sha256: str
    accepted_authority_sha256: str
    accepted_patch_set_sha256: str
    build_pair_sha256: str
    correctness_result_sha256: str
    incremental_comparison_result_sha256: str
    frozen_production: FrozenProductionAuthority
    model_sha256: str
    workload_sha256: str
    runtime_config_sha256: str
    protocol_frame_sha256: str
    metric: str
    metric_direction: str
    incremental_exact_route_effect_fraction: float
    incremental_graphs_off_effect_fraction: float
    incremental_graphs_on_effect_fraction: float
    cumulative_graphs_on_effect_fraction: float
    incremental_exact_route_receipt_sha256: str
    incremental_exact_route_receipt_ref: MeasurementReceiptRef
    incremental_graphs_off_receipt_sha256: str
    incremental_graphs_off_receipt_ref: MeasurementReceiptRef
    incremental_graphs_on_receipt_sha256: str
    incremental_graphs_on_receipt_ref: MeasurementReceiptRef
    production_graphs_on_receipt_sha256: str
    production_graphs_on_receipt_ref: MeasurementReceiptRef
    incremental_graphs_off_frame_sha256: str
    incremental_graphs_on_frame_sha256: str
    production_graphs_on_frame_sha256: str
    production_graphs_mode: str
    cumulative_classification: str
    promotion_eligible: bool
    promotion_reason: str
    composition_terminal_sha256: str
    result_sha256: str

    @classmethod
    def create(
            cls, plan: CompositionPlan, pair: CumulativeBuildPair,
            correctness: FullCorrectness, incremental: IncrementalComparison,
            *, frozen_production: FrozenProductionAuthority,
            model_sha256: str, workload_sha256: str,
            runtime_config_sha256: str, protocol_frame_sha256: str,
            metric: str, metric_direction: str,
            cumulative_graphs_on_effect_fraction: float,
            production_graphs_on_receipt_sha256: str,
            production_graphs_on_receipt_path: Path | str,
            incremental_graphs_off_frame_sha256: str,
            incremental_graphs_on_frame_sha256: str,
            production_graphs_on_frame_sha256: str,
    ) -> "CumulativePerformance":
        pair.bind_plan(plan)
        correctness.bind_pair(pair)
        incremental.bind(pair, correctness)
        frozen_production.bind_plan(plan)
        if not correctness.passed:
            raise CompositionError(
                "failed correctness cannot reach cumulative performance")
        if metric_direction != "higher_better":
            raise CompositionError(
                "cumulative performance metric direction is unsupported")
        metric = _require_text(metric, "metric")
        production_ref = MeasurementReceiptRef(
            role="production_graphs_on",
            path=str(production_graphs_on_receipt_path),
            sha256=production_graphs_on_receipt_sha256)
        hashes = {
            "model_sha256": model_sha256,
            "workload_sha256": workload_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "protocol_frame_sha256": protocol_frame_sha256,
            "production_graphs_on_receipt_sha256":
                production_graphs_on_receipt_sha256,
            "incremental_graphs_off_frame_sha256":
                incremental_graphs_off_frame_sha256,
            "incremental_graphs_on_frame_sha256":
                incremental_graphs_on_frame_sha256,
            "production_graphs_on_frame_sha256":
                production_graphs_on_frame_sha256,
        }
        for label, value in hashes.items():
            _require_sha(value, label)
        if (frozen_production.graphs_mode != "graphs_on"
                or frozen_production.measurement_protocol_sha256 !=
                   protocol_frame_sha256
                or frozen_production.frame_sha256 !=
                   production_graphs_on_frame_sha256
                or frozen_production.model_sha256 != model_sha256
                or frozen_production.workload_sha256 != workload_sha256
                or frozen_production.runtime_config_sha256 !=
                   runtime_config_sha256
                or frozen_production.metric != "tokens_per_second"
                or metric not in {
                    "decode_tokens_per_s", "prefill_tokens_per_s"}
                or frozen_production.direction != "higher_is_better"
                or metric_direction != "higher_better"):
            raise CompositionError(
                "cumulative production comparator authority changed")
        if len({incremental.graphs_off_receipt_sha256,
                incremental.graphs_on_receipt_sha256,
                production_graphs_on_receipt_sha256}) != 3:
            raise CompositionError(
                "cumulative measurement receipts are not three distinct runs")
        on = _finite(cumulative_graphs_on_effect_fraction,
                     "cumulative graphs-on effect")
        if on > 0:
            classification = "candidate"
        else:
            classification = "screened_out"
        if not incremental.admissible:
            eligible = False
            reason = f"incremental_{incremental.classification}"
        elif classification != "candidate":
            eligible = False
            reason = f"cumulative_{classification}"
        else:
            eligible = True
            reason = "incremental_and_cumulative_positive"
        disposition = ("admitted" if incremental.admissible
                       else "incremental_rollback")
        reason_code = (
            "incremental_admitted_promotion_eligible"
            if incremental.admissible and eligible else
            "incremental_admitted_" + reason
            if incremental.admissible else
            f"incremental_{incremental.classification}")
        terminal_core = {
            "schema": "epyc.autokernel.cumulative_composition_terminal.v3",
            "operation_key": plan.operation_key,
            "plan_sha256": plan.plan_sha256, "plan": plan.to_dict(),
            "lever_sha256": plan.candidate.accepted[-1].lever_sha256,
            "cross_campaign_candidate_sha256":
                plan.candidate.accepted[-1].cross_campaign_candidate_sha256,
            "isolated_result_sha256s": [
                row.result_sha256
                for row in plan.candidate.accepted[-1].replications],
            "disposition": disposition, "scientific_budget_spent": True,
            "build_pair": pair.to_dict(),
            "correctness": correctness.to_dict(),
            "comparison": incremental.to_dict(),
            "correctness_result_sha256": correctness.result_sha256,
            "comparison_result_sha256": incremental.result_sha256,
            "promotion_eligible": eligible, "promotion_reason": reason,
            "admitted_authority_sha256": (
                plan.candidate.authority_sha256
                if disposition == "admitted" else None),
            "reason_code": reason_code,
            "infrastructure_receipt_sha256": None,
            "attribution_receipt_sha256": None,
        }
        values = {
            "operation_key": plan.operation_key,
            "plan_sha256": plan.plan_sha256,
            "accepted_authority_sha256": plan.candidate.authority_sha256,
            "accepted_patch_set_sha256":
                plan.candidate.ordered_patch_set_sha256,
            "build_pair_sha256": pair.pair_sha256,
            "correctness_result_sha256": correctness.result_sha256,
            "incremental_comparison_result_sha256":
                incremental.result_sha256,
            "frozen_production": frozen_production,
            "model_sha256": model_sha256,
            "workload_sha256": workload_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "protocol_frame_sha256": protocol_frame_sha256,
            "metric": metric, "metric_direction": metric_direction,
            "incremental_exact_route_effect_fraction":
                incremental.exact_route_effect_fraction,
            "incremental_graphs_off_effect_fraction":
                incremental.graphs_off_effect_fraction,
            "incremental_graphs_on_effect_fraction":
                incremental.graphs_on_effect_fraction,
            "cumulative_graphs_on_effect_fraction": on,
            "incremental_exact_route_receipt_sha256":
                incremental.exact_route_receipt_sha256,
            "incremental_exact_route_receipt_ref":
                incremental.exact_route_receipt_ref,
            "incremental_graphs_off_receipt_sha256":
                incremental.graphs_off_receipt_sha256,
            "incremental_graphs_off_receipt_ref":
                incremental.graphs_off_receipt_ref,
            "incremental_graphs_on_receipt_sha256":
                incremental.graphs_on_receipt_sha256,
            "incremental_graphs_on_receipt_ref":
                incremental.graphs_on_receipt_ref,
            "production_graphs_on_receipt_sha256":
                production_graphs_on_receipt_sha256,
            "production_graphs_on_receipt_ref": production_ref,
            "incremental_graphs_off_frame_sha256":
                incremental_graphs_off_frame_sha256,
            "incremental_graphs_on_frame_sha256":
                incremental_graphs_on_frame_sha256,
            "production_graphs_on_frame_sha256":
                production_graphs_on_frame_sha256,
            "production_graphs_mode": "on",
            "cumulative_classification": classification,
            "promotion_eligible": eligible,
            "promotion_reason": reason,
            "composition_terminal_sha256": _sha(terminal_core),
        }
        body = cls._body_for(**values)
        return cls(**values, result_sha256=_sha(body))

    @staticmethod
    def _body_for(**values: Any) -> dict[str, Any]:
        body = dict(values)
        frozen = body.get("frozen_production")
        if isinstance(frozen, FrozenProductionAuthority):
            body["frozen_production"] = frozen.to_dict()
        for key in (
                "incremental_exact_route_receipt_ref",
                "incremental_graphs_off_receipt_ref",
                "incremental_graphs_on_receipt_ref",
                "production_graphs_on_receipt_ref"):
            reference = body.get(key)
            if isinstance(reference, MeasurementReceiptRef):
                body[key] = reference.to_dict()
        return {
            "schema": "epyc.autokernel.cumulative_performance.v2",
            "authority": "frozen_production_promotion_gate",
            "promotion_authority": True,
            **body,
        }

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_production, FrozenProductionAuthority):
            raise CompositionError(
                "cumulative performance production authority is untyped")
        references = (
            self.incremental_exact_route_receipt_ref,
            self.incremental_graphs_off_receipt_ref,
            self.incremental_graphs_on_receipt_ref,
            self.production_graphs_on_receipt_ref,
        )
        if any(not isinstance(row, MeasurementReceiptRef)
               for row in references):
            raise CompositionError("cumulative measurement ref is untyped")
        for label in (
            "operation_key", "plan_sha256", "accepted_authority_sha256",
            "accepted_patch_set_sha256", "build_pair_sha256",
            "correctness_result_sha256",
            "incremental_comparison_result_sha256", "model_sha256",
            "workload_sha256", "runtime_config_sha256",
            "protocol_frame_sha256",
            "incremental_exact_route_receipt_sha256",
            "incremental_graphs_off_receipt_sha256",
            "incremental_graphs_on_receipt_sha256",
            "production_graphs_on_receipt_sha256",
            "incremental_graphs_off_frame_sha256",
            "incremental_graphs_on_frame_sha256",
            "production_graphs_on_frame_sha256",
            "composition_terminal_sha256", "result_sha256",
        ):
            _require_sha(getattr(self, label), label)
        _require_text(self.metric, "metric")
        _require_text(self.promotion_reason, "promotion_reason")
        if (self.metric_direction != "higher_better"
                or not isinstance(self.promotion_eligible, bool)):
            raise CompositionError(
                "cumulative performance promotion decision is malformed")
        if (self.production_graphs_mode != "on"
                or self.frozen_production.graphs_mode != "graphs_on"
                or self.frozen_production.measurement_protocol_sha256 !=
                   self.protocol_frame_sha256
                or self.frozen_production.frame_sha256 !=
                   self.production_graphs_on_frame_sha256
                or self.frozen_production.model_sha256 != self.model_sha256
                or self.frozen_production.workload_sha256 !=
                   self.workload_sha256
                or self.frozen_production.runtime_config_sha256 !=
                   self.runtime_config_sha256
                or self.frozen_production.metric != "tokens_per_second"
                or self.metric not in {
                    "decode_tokens_per_s", "prefill_tokens_per_s"}
                or self.frozen_production.direction != "higher_is_better"):
            raise CompositionError(
                "cumulative performance protocol frame changed")
        if len({self.incremental_graphs_off_receipt_sha256,
                self.incremental_graphs_on_receipt_sha256,
                self.production_graphs_on_receipt_sha256}) != 3:
            raise CompositionError(
                "cumulative measurement receipts are not three distinct runs")
        incremental_effects = tuple(_finite(value, label) for value, label in (
            (self.incremental_exact_route_effect_fraction,
             "incremental exact-route effect"),
            (self.incremental_graphs_off_effect_fraction,
             "incremental graphs-off effect"),
            (self.incremental_graphs_on_effect_fraction,
             "incremental graphs-on effect"),
        ))
        on = _finite(self.cumulative_graphs_on_effect_fraction,
                     "cumulative graphs-on effect")
        incremental_roots = tuple(
            reference.canonical_location() for reference in references[:3])
        runner_authority = _load_runner_measurement_authority(
            incremental_roots[0][0])
        exact = _exact_route_projection(
            self.incremental_exact_route_receipt_ref.load())
        off = _runner_projection(
            self.incremental_graphs_off_receipt_ref.load(),
            graph_mode="off", factor_name="source_patch")
        incremental_on = _runner_projection(
            self.incremental_graphs_on_receipt_ref.load(),
            graph_mode="on", factor_name="source_patch")
        production_root, production_repetition = \
            self.production_graphs_on_receipt_ref.canonical_location()
        production = _runner_projection(
            self.production_graphs_on_receipt_ref.load(), graph_mode="on",
            factor_name="cumulative_production")
        derived_incremental = (
            exact["effect_fraction"], off["effect_fraction"],
            incremental_on["effect_fraction"])
        incremental_common = (
            "candidate_identity", "anchor_identity", "model_sha256",
            "workload_sha256", "runtime_config_sha256", "metric",
            "metric_direction",
        )
        if (tuple(reference.role for reference in references) != (
                    "exact_route", "incremental_graphs_off",
                    "incremental_graphs_on", "production_graphs_on")
                or len({row[0] for row in incremental_roots}) != 1
                or incremental_roots[0][0].name != self.operation_key
                or incremental_roots[0][1] is not None
                or incremental_roots[1][1] != incremental_roots[2][1]
                or self.incremental_exact_route_receipt_sha256 !=
                   self.incremental_exact_route_receipt_ref.sha256
                or runner_authority["frozen_production"] !=
                   self.frozen_production
                or self.incremental_graphs_off_receipt_sha256 !=
                   self.incremental_graphs_off_receipt_ref.sha256
                or self.incremental_graphs_on_receipt_sha256 !=
                   self.incremental_graphs_on_receipt_ref.sha256
                or incremental_effects != derived_incremental
                or self.production_graphs_on_receipt_ref.role !=
                "production_graphs_on"
                or production_root != incremental_roots[0][0]
                or production_repetition != incremental_roots[1][1]
                or self.production_graphs_on_receipt_sha256 !=
                   self.production_graphs_on_receipt_ref.sha256
                or on != production["effect_fraction"]
                or any(off[field] != incremental_on[field]
                       for field in incremental_common)
                or exact["candidate_identity"] != off["candidate_identity"]
                or exact["anchor_identity"] != off["anchor_identity"]
                or production["candidate_identity"] !=
                   off["candidate_identity"]
                or production["anchor_identity"] !=
                   self.frozen_production.build_identity
                or exact["model_sha256"] != self.model_sha256
                or exact["workload_sha256"] != self.workload_sha256
                or exact["runtime_config_sha256"] !=
                   self.runtime_config_sha256
                or any(row["model_sha256"] != self.model_sha256
                       for row in (off, incremental_on, production))
                or any(row["workload_sha256"] !=
                       self.frozen_production.observed_workload_sha256
                       for row in (off, incremental_on, production))
                or any(row["runtime_config_sha256"] !=
                       self.frozen_production.observed_runtime_config_sha256
                       for row in (off, incremental_on, production))
                or any(row["metric"] != self.metric
                       or row["metric_direction"] != self.metric_direction
                       for row in (off, incremental_on, production))
                or self.protocol_frame_sha256 !=
                   incremental_on["protocol_frame_sha256"]
                or self.protocol_frame_sha256 !=
                   production["protocol_frame_sha256"]
                or self.incremental_graphs_off_frame_sha256 !=
                   off["candidate_frame_sha256"]
                or self.incremental_graphs_on_frame_sha256 !=
                   incremental_on["candidate_frame_sha256"]
                or self.production_graphs_on_frame_sha256 !=
                   production["anchor_frame_sha256"]):
            raise CompositionError(
                "cumulative measurement carriers changed")
        incremental_class = (
            "candidate" if all(value > 0 for value in incremental_effects)
            else "screened_out" if all(value <= 0 for value in incremental_effects)
            else "inconclusive")
        cumulative_class = (
            "candidate" if on > 0 else "screened_out")
        expected_eligible = (
            incremental_class == "candidate"
            and cumulative_class == "candidate")
        expected_reason = (
            "incremental_and_cumulative_positive"
            if expected_eligible else
            f"incremental_{incremental_class}"
            if incremental_class != "candidate" else
            f"cumulative_{cumulative_class}")
        values = {
            key: getattr(self, key) for key in self.__dataclass_fields__
            if key != "result_sha256"
        }
        if (self.cumulative_classification != cumulative_class
                or self.promotion_eligible != expected_eligible
                or self.promotion_reason != expected_reason
                or self.result_sha256 != _sha(self._body_for(**values))):
            raise CompositionError(
                "cumulative performance identity or decision changed")

    def bind(
            self, plan: CompositionPlan, pair: CumulativeBuildPair,
            correctness: FullCorrectness,
            incremental: IncrementalComparison,
    ) -> None:
        self.frozen_production.bind_plan(plan)
        pair.bind_plan(plan)
        correctness.bind_pair(pair)
        incremental.bind(pair, correctness)
        exact = _exact_route_projection(
            self.incremental_exact_route_receipt_ref.load())
        off = _runner_projection(
            self.incremental_graphs_off_receipt_ref.load(),
            graph_mode="off", factor_name="source_patch")
        incremental_on = _runner_projection(
            self.incremental_graphs_on_receipt_ref.load(),
            graph_mode="on", factor_name="source_patch")
        production = _runner_projection(
            self.production_graphs_on_receipt_ref.load(), graph_mode="on",
            factor_name="cumulative_production")
        if (self.operation_key != plan.operation_key
                or self.plan_sha256 != plan.plan_sha256
                or self.accepted_authority_sha256 !=
                   plan.candidate.authority_sha256
                or self.accepted_patch_set_sha256 !=
                   plan.candidate.ordered_patch_set_sha256
                or self.build_pair_sha256 != pair.pair_sha256
                or self.correctness_result_sha256 !=
                   correctness.result_sha256
                or self.incremental_comparison_result_sha256 !=
                   incremental.result_sha256
                or self.incremental_exact_route_effect_fraction !=
                   incremental.exact_route_effect_fraction
                or self.incremental_graphs_off_effect_fraction !=
                   incremental.graphs_off_effect_fraction
                or self.incremental_graphs_on_effect_fraction !=
                   incremental.graphs_on_effect_fraction
                or self.incremental_exact_route_receipt_sha256 !=
                   incremental.exact_route_receipt_sha256
                or self.incremental_exact_route_receipt_ref !=
                   incremental.exact_route_receipt_ref
                or self.incremental_graphs_off_receipt_sha256 !=
                   incremental.graphs_off_receipt_sha256
                or self.incremental_graphs_off_receipt_ref !=
                   incremental.graphs_off_receipt_ref
                or self.incremental_graphs_on_receipt_sha256 !=
                   incremental.graphs_on_receipt_sha256
                or self.incremental_graphs_on_receipt_ref !=
                   incremental.graphs_on_receipt_ref
                or self.production_graphs_on_receipt_ref.canonical_location()[0]
                   != incremental.operation_root
                or self.production_graphs_on_receipt_ref.canonical_location()[1]
                   != incremental.repetition
                or exact["candidate_identity"] != pair.candidate.build_identity
                or exact["anchor_identity"] != pair.anchor.build_identity
                or any(row["candidate_identity"] !=
                       pair.candidate.build_identity
                       for row in (off, incremental_on, production))
                or any(row["anchor_identity"] != pair.anchor.build_identity
                       for row in (off, incremental_on))
                or production["anchor_identity"] !=
                   self.frozen_production.build_identity):
            raise CompositionError(
                "cumulative performance binds other composition evidence")
        disposition = ("admitted" if incremental.admissible
                       else "incremental_rollback")
        terminal = {
            "schema": "epyc.autokernel.cumulative_composition_terminal.v3",
            "operation_key": plan.operation_key,
            "plan_sha256": plan.plan_sha256, "plan": plan.to_dict(),
            "lever_sha256": plan.candidate.accepted[-1].lever_sha256,
            "cross_campaign_candidate_sha256":
                plan.candidate.accepted[-1].cross_campaign_candidate_sha256,
            "isolated_result_sha256s": [
                row.result_sha256
                for row in plan.candidate.accepted[-1].replications],
            "disposition": disposition, "scientific_budget_spent": True,
            "build_pair": pair.to_dict(),
            "correctness": correctness.to_dict(),
            "comparison": incremental.to_dict(),
            "correctness_result_sha256": correctness.result_sha256,
            "comparison_result_sha256": incremental.result_sha256,
            "promotion_eligible": self.promotion_eligible,
            "promotion_reason": self.promotion_reason,
            "admitted_authority_sha256": (
                plan.candidate.authority_sha256
                if disposition == "admitted" else None),
            "reason_code": (
                "incremental_admitted_promotion_eligible"
                if incremental.admissible and self.promotion_eligible else
                "incremental_admitted_" + self.promotion_reason
                if incremental.admissible else
                f"incremental_{incremental.classification}"),
            "infrastructure_receipt_sha256": None,
            "attribution_receipt_sha256": None,
        }
        if self.composition_terminal_sha256 != _sha(terminal):
            raise CompositionError(
                "cumulative performance terminal decision changed")

    def to_dict(self) -> dict[str, Any]:
        values = {
            key: getattr(self, key) for key in self.__dataclass_fields__
            if key != "result_sha256"
        }
        return {**self._body_for(**values),
                "result_sha256": self.result_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CumulativePerformance":
        fields = {
            "operation_key", "plan_sha256", "accepted_authority_sha256",
            "accepted_patch_set_sha256", "build_pair_sha256",
            "correctness_result_sha256",
            "incremental_comparison_result_sha256", "frozen_production",
            "model_sha256", "workload_sha256", "runtime_config_sha256",
            "protocol_frame_sha256", "metric", "metric_direction",
            "incremental_exact_route_effect_fraction",
            "incremental_graphs_off_effect_fraction",
            "incremental_graphs_on_effect_fraction",
            "cumulative_graphs_on_effect_fraction",
            "incremental_exact_route_receipt_sha256",
            "incremental_exact_route_receipt_ref",
            "incremental_graphs_off_receipt_sha256",
            "incremental_graphs_off_receipt_ref",
            "incremental_graphs_on_receipt_sha256",
            "incremental_graphs_on_receipt_ref",
            "production_graphs_on_receipt_sha256",
            "production_graphs_on_receipt_ref",
            "incremental_graphs_off_frame_sha256",
            "incremental_graphs_on_frame_sha256",
            "production_graphs_on_frame_sha256",
            "production_graphs_mode",
            "cumulative_classification", "promotion_eligible",
            "promotion_reason", "composition_terminal_sha256",
            "result_sha256",
        }
        required = fields | {"schema", "authority", "promotion_authority"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError(
                "cumulative performance has an inexact schema")
        if (value.get("schema") !=
                "epyc.autokernel.cumulative_performance.v2"
                or value.get("authority") !=
                   "frozen_production_promotion_gate"
                or value.get("promotion_authority") is not True):
            raise CompositionError(
                "cumulative performance authority changed")
        kwargs = {key: value[key] for key in fields}
        kwargs["frozen_production"] = FrozenProductionAuthority.from_dict(
            value["frozen_production"])
        for key in (
                "incremental_exact_route_receipt_ref",
                "incremental_graphs_off_receipt_ref",
                "incremental_graphs_on_receipt_ref",
                "production_graphs_on_receipt_ref"):
            kwargs[key] = MeasurementReceiptRef.from_dict(value[key])
        return cls(**kwargs)


@dataclass(frozen=True)
class CumulativePerformanceRef:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        candidate = Path(self.path)
        if (not isinstance(self.path, str) or not candidate.is_absolute()
                or ".." in candidate.parts):
            raise CompositionError(
                "cumulative performance reference path is unsafe")
        _require_sha(self.sha256, "cumulative performance file sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.cumulative_performance_ref.v1",
            "path": self.path, "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CumulativePerformanceRef":
        if (not isinstance(value, Mapping)
                or set(value) != {"schema", "path", "sha256"}
                or value.get("schema") !=
                   "epyc.autokernel.cumulative_performance_ref.v1"):
            raise CompositionError(
                "cumulative performance reference has an inexact schema")
        return cls(path=value["path"], sha256=value["sha256"])


def frozen_production_protocol_binding(
        *, model_sha256: str,
        build_identity: gpu_source_proofs.BuildIdentity,
) -> dict[str, str]:
    """Derive the exact future graphs-on descriptor without measurement."""
    _require_sha(model_sha256, "model_sha256")
    if not isinstance(build_identity, gpu_source_proofs.BuildIdentity):
        raise CompositionError("frozen protocol requires typed build identity")
    workload = {
        "backend": "llama_gpu", "recipe": "tg128-ngl99",
        "n_prompt": 0, "n_gen": 128,
    }
    runtime = {
        "n_threads": 8, "n_batch": 512, "n_ubatch": 512,
        "use_mmap": True, "no_op_offload": 0,
        "split_mode": "layer", "no_kv_offload": False,
        "poll": 50, "n_prompt": 0, "n_gen": 128,
        "flash_attn": 1,
    }
    protocol = {
        **workload, "model_sha256": model_sha256,
        "metric": "decode_tokens_per_s",
        "metric_direction": "higher_better",
        "cpu_list": "184-191", "device": "AMD Instinct MI210",
        "architecture": "gfx90a", "runtime_config_sha256": _sha(runtime),
        "graphs_mode": "on", "candidate_invocations": 9,
        "candidate_processes": 1,
    }
    frame = {
        "schema": "epyc.autokernel.measurement_arm_frame.v1",
        "arm": "anchor", "protocol": protocol,
        "source_commit": build_identity.source_commit,
        "build_identity": asdict(build_identity),
        "factor_name": "cumulative_production",
    }
    return {
        "observed_workload_sha256": _sha(workload),
        "observed_runtime_config_sha256": _sha(runtime),
        "measurement_protocol_sha256": _sha(protocol),
        "frame_sha256": _sha(frame),
    }


def _measurement_descriptor(
        value: Mapping[str, Any], *, graph_mode: str,
        candidate: BuildBinding, anchor_identity: gpu_source_proofs.BuildIdentity,
        factor_name: str,
) -> dict[str, Any]:
    """Project one runner result into its commensurability authority."""
    projection = _runner_projection(
        value, graph_mode=graph_mode, factor_name=factor_name)
    if projection["candidate_identity"] != candidate.build_identity:
        raise CompositionError(
            "cumulative performance candidate build identity changed")
    if projection["anchor_identity"] != anchor_identity:
        raise CompositionError(
            "cumulative performance comparator build identity changed")
    return {
        **projection,
        "frame_sha256": projection["candidate_frame_sha256"],
    }


def performance_from_measurements(
        plan: CompositionPlan, pair: CumulativeBuildPair,
        correctness: FullCorrectness, incremental: IncrementalComparison,
        *, frozen_production: FrozenProductionAuthority,
        incremental_graphs_off: Mapping[str, Any],
        incremental_graphs_on: Mapping[str, Any],
        production_graphs_on: Mapping[str, Any],
        production_graphs_on_receipt_sha256: str,
        production_graphs_on_receipt_path: Path | str,
) -> CumulativePerformance:
    """Create authority from incremental off/on and production graphs-on."""
    pair.bind_plan(plan)
    frozen_production.bind_plan(plan)
    incremental_rows = (
        _measurement_descriptor(
            incremental_graphs_off, graph_mode="off",
            candidate=pair.candidate,
            anchor_identity=pair.anchor.build_identity,
            factor_name="source_patch"),
        _measurement_descriptor(
            incremental_graphs_on, graph_mode="on",
            candidate=pair.candidate,
            anchor_identity=pair.anchor.build_identity,
            factor_name="source_patch"),
    )
    production_row = _measurement_descriptor(
        production_graphs_on, graph_mode="on",
        candidate=pair.candidate,
        anchor_identity=frozen_production.build_identity,
        factor_name="cumulative_production")
    sealed_protocol = frozen_production_protocol_binding(
        model_sha256=frozen_production.model_sha256,
        build_identity=frozen_production.build_identity)
    if (production_row["workload_sha256"] !=
            sealed_protocol["observed_workload_sha256"]
            or frozen_production.observed_workload_sha256 !=
               production_row["workload_sha256"]
            or production_row["runtime_config_sha256"] !=
               sealed_protocol["observed_runtime_config_sha256"]
            or frozen_production.observed_runtime_config_sha256 !=
               production_row["runtime_config_sha256"]
            or production_row["protocol_frame_sha256"] !=
               sealed_protocol["measurement_protocol_sha256"]
            or production_row["anchor_frame_sha256"] !=
               sealed_protocol["frame_sha256"]):
        raise CompositionError(
            "cumulative production measurement differs from sealed protocol")
    common_fields = (
        "model_sha256", "workload_sha256",
        "runtime_config_sha256", "metric", "metric_direction",
    )
    if any(len({row[field] for row in (
            *incremental_rows, production_row)}) != 1
           for field in common_fields):
        raise CompositionError(
            "cumulative production comparison is not protocol matched")
    if (incremental_rows[1]["protocol_frame_sha256"] !=
            production_row["protocol_frame_sha256"]):
        raise CompositionError(
            "cumulative graphs-on measurement protocol changed")
    if (incremental.graphs_off_receipt_sha256 !=
            _require_sha(incremental.graphs_off_receipt_sha256,
                         "incremental graphs-off receipt")
            or incremental.graphs_on_receipt_sha256 !=
               _require_sha(incremental.graphs_on_receipt_sha256,
                            "incremental graphs-on receipt")):
        raise CompositionError("incremental comparison receipt changed")
    shared = incremental_rows[0]
    return CumulativePerformance.create(
        plan, pair, correctness, incremental,
        frozen_production=frozen_production,
        model_sha256=frozen_production.model_sha256,
        workload_sha256=frozen_production.workload_sha256,
        runtime_config_sha256=frozen_production.runtime_config_sha256,
        protocol_frame_sha256=
            incremental_rows[1]["protocol_frame_sha256"],
        metric=shared["metric"],
        metric_direction=shared["metric_direction"],
        cumulative_graphs_on_effect_fraction=
            production_row["effect_fraction"],
        production_graphs_on_receipt_sha256=
            production_graphs_on_receipt_sha256,
        production_graphs_on_receipt_path=
            production_graphs_on_receipt_path,
        incremental_graphs_off_frame_sha256=
            incremental_rows[0]["frame_sha256"],
        incremental_graphs_on_frame_sha256=
            incremental_rows[1]["frame_sha256"],
        production_graphs_on_frame_sha256=
            production_row["anchor_frame_sha256"],
    )


def load_cumulative_performance(
        path: Path, *, expected_file_sha256: str | None = None,
) -> tuple[CumulativePerformance, str]:
    """Stable same-fd reopen of a canonical cumulative-performance receipt."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompositionError(
            "cumulative performance receipt is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or before.st_mode & 0o022
                or before.st_size > 4 * 1024 * 1024):
            raise CompositionError(
                "cumulative performance receipt identity is unsafe")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        pathname = os.lstat(path)
    except OSError as exc:
        raise CompositionError(
            "cumulative performance receipt path disappeared") from exc
    identity = lambda row: (
        row.st_dev, row.st_ino, row.st_uid, stat.S_IFMT(row.st_mode),
        row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(pathname):
        raise CompositionError(
            "cumulative performance receipt changed during stable read")
    raw = b"".join(chunks)
    file_sha = hashlib.sha256(raw).hexdigest()
    if (expected_file_sha256 is not None
            and file_sha != _require_sha(
                expected_file_sha256, "cumulative performance expected hash")):
        raise CompositionError("cumulative performance receipt bytes changed")
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")))
        performance = CumulativePerformance.from_dict(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError,
            TypeError) as exc:
        raise CompositionError(
            "cumulative performance receipt is not strict evidence") from exc
    operation_root = \
        performance.production_graphs_on_receipt_ref.canonical_location()[0]
    if path != operation_root / "cumulative-performance.json":
        raise CompositionError(
            "cumulative performance receipt is outside its canonical operation")
    return performance, file_sha


def seal_cumulative_performance(
        path: Path, performance: CumulativePerformance,
) -> CumulativePerformanceRef:
    operation_root = \
        performance.production_graphs_on_receipt_ref.canonical_location()[0]
    if (not path.is_absolute()
            or path != operation_root / "cumulative-performance.json"):
        raise CompositionError(
            "cumulative performance receipt path is not canonical")
    if path.exists() or path.is_symlink():
        reopened, file_sha = load_cumulative_performance(path)
        if reopened != performance:
            raise CompositionError(
                "cumulative performance changed on restart")
    else:
        _atomic_replace(path, performance.to_dict())
        reopened, file_sha = load_cumulative_performance(path)
        if reopened != performance:
            raise CompositionError(
                "cumulative performance changed while sealing")
    return CumulativePerformanceRef(
        path=str(path.resolve()), sha256=file_sha)


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

    SCHEMA = "epyc.autokernel.cumulative_composition_state.v3"

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
            _validate_dnr_history(
                plan,
                [row["cross_campaign_candidate_sha256"]
                 for row in state["terminals"]])
            for terminal in state["terminals"]:
                if (terminal["scientific_budget_spent"] is True
                        and terminal["cross_campaign_candidate_sha256"] ==
                            proposed.cross_campaign_candidate_sha256):
                    raise CompositionError("composition would repeat a scientific lever")
            state["pending"] = {
                "stage": "planned", "plan": plan.to_dict(),
                "build_pair": None, "correctness": None, "comparison": None,
                "cumulative_performance": None,
                "cumulative_performance_ref": None,
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
                cumulative_performance=None,
                cumulative_performance_ref=None,
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
            comparison.bind(pair, correctness)
            if pending["comparison"] is not None:
                if pending["comparison"] == comparison.to_dict():
                    return state
                raise CompositionError("incremental comparison changed on restart")
            if pending["stage"] != "correctness_passed":
                raise CompositionError("incremental comparison arrived out of order")
            pending["comparison"] = comparison.to_dict()
            pending["stage"] = "incremental_measured"
            return self._write_unlocked(state)

    def record_cumulative_performance(
            self, performance: CumulativePerformance,
            reference: CumulativePerformanceRef,
    ) -> dict[str, Any]:
        with self._lock():
            state = self._load_unlocked()
            if state["pending"] is None:
                matches = [row for row in state["terminals"]
                           if (row["operation_key"] ==
                               performance.operation_key
                               and row["cumulative_performance"] ==
                               performance.to_dict()
                               and row["cumulative_performance_ref"] ==
                               reference.to_dict())]
                if len(matches) == 1:
                    return state
                raise CompositionError("no cumulative composition is pending")
            state, pending, plan = self._pending(state)
            if (pending["build_pair"] is None
                    or pending["correctness"] is None
                    or pending["comparison"] is None):
                raise CompositionError(
                    "cumulative performance cannot skip incremental evidence")
            pair = CumulativeBuildPair.from_dict(pending["build_pair"])
            correctness = FullCorrectness.from_dict(pending["correctness"])
            comparison = IncrementalComparison.from_dict(
                pending["comparison"])
            performance.bind(plan, pair, correctness, comparison)
            reopened, file_sha = load_cumulative_performance(
                Path(reference.path), expected_file_sha256=reference.sha256)
            if reopened != performance or file_sha != reference.sha256:
                raise CompositionError(
                    "cumulative performance reference changed")
            if pending["cumulative_performance"] is not None:
                if (pending["cumulative_performance"] == performance.to_dict()
                        and pending["cumulative_performance_ref"] ==
                            reference.to_dict()):
                    return state
                raise CompositionError(
                    "cumulative performance changed on restart")
            if pending["stage"] != "incremental_measured":
                raise CompositionError(
                    "cumulative performance arrived out of order")
            pending["cumulative_performance"] = performance.to_dict()
            pending["cumulative_performance_ref"] = reference.to_dict()
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
            performance = CumulativePerformance.from_dict(
                pending["cumulative_performance"])
            reference = CumulativePerformanceRef.from_dict(
                pending["cumulative_performance_ref"])
            pair = CumulativeBuildPair.from_dict(pending["build_pair"])
            performance.bind(plan, pair, correctness, comparison)
            if comparison.admissible:
                return self._terminalize(
                    state, plan=plan, disposition="admitted", scientific=True,
                    correctness=correctness, comparison=comparison,
                    cumulative_performance=performance,
                    cumulative_performance_ref=reference,
                    admitted=plan.candidate,
                    reason_code=(
                        "incremental_admitted_promotion_eligible"
                        if performance.promotion_eligible else
                        "incremental_admitted_" +
                        performance.promotion_reason),
                )
            return self._terminalize(
                state, plan=plan, disposition="incremental_rollback",
                scientific=True, correctness=correctness, comparison=comparison,
                cumulative_performance=performance,
                cumulative_performance_ref=reference, admitted=None,
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
                cumulative_performance=None,
                cumulative_performance_ref=None,
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
                cumulative_performance=None,
                cumulative_performance_ref=None,
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
        prior_cross_campaign_candidates: list[str] = []
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
            _validate_dnr_history(plan, prior_cross_campaign_candidates)
            prior_cross_campaign_candidates.append(
                terminal["cross_campaign_candidate_sha256"])
            if terminal["disposition"] == "admitted":
                derived_authority = plan.candidate
        if authority != derived_authority:
            raise CompositionError("composition authority differs from terminal chain")
        if state["scientific_attempts"] != science:
            raise CompositionError("composition science count differs from terminals")
        if state["scientific_attempts"] > state["max_scientific_attempts"]:
            raise CompositionError("composition state exceeds its scientific budget")
        if state["pending"] is not None:
            self._validate_pending(
                state["pending"], authority, terminal_keys,
                prior_cross_campaign_candidates)
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
            prior_cross_campaign_candidates: Sequence[str],
    ) -> None:
        if not isinstance(pending, Mapping) or set(pending) != {
            "stage", "plan", "build_pair", "correctness", "comparison",
            "cumulative_performance", "cumulative_performance_ref",
        }:
            raise CompositionError("composition pending state has an inexact schema")
        plan = CompositionPlan.from_dict(pending["plan"])
        if plan.anchor != authority or plan.operation_key in terminal_keys:
            raise CompositionError("composition pending authority is stale")
        _validate_dnr_history(plan, prior_cross_campaign_candidates)
        stage = pending["stage"]
        allowed = {
            "planned", "built", "correctness_passed",
            "incremental_measured", "measured",
        }
        if stage not in allowed:
            raise CompositionError("composition pending stage is invalid")
        pair = (None if pending["build_pair"] is None else
                CumulativeBuildPair.from_dict(pending["build_pair"]))
        correctness = (None if pending["correctness"] is None else
                       FullCorrectness.from_dict(pending["correctness"]))
        comparison = (None if pending["comparison"] is None else
                      IncrementalComparison.from_dict(pending["comparison"]))
        performance = (
            None if pending["cumulative_performance"] is None else
            CumulativePerformance.from_dict(pending["cumulative_performance"]))
        performance_ref = (
            None if pending["cumulative_performance_ref"] is None else
            CumulativePerformanceRef.from_dict(
                pending["cumulative_performance_ref"]))
        expected_presence = {
            "planned": (False, False, False, False, False),
            "built": (True, False, False, False, False),
            "correctness_passed": (True, True, False, False, False),
            "incremental_measured": (True, True, True, False, False),
            "measured": (True, True, True, True, True),
        }
        if tuple(row is not None for row in (
                pair, correctness, comparison, performance,
                performance_ref)) != \
                expected_presence[stage]:
            raise CompositionError("composition pending stage/evidence disagree")
        if pair is not None:
            pair.bind_plan(plan)
        if correctness is not None:
            correctness.bind_pair(pair)
            if not correctness.passed:
                raise CompositionError("failed correctness remained pending")
        if comparison is not None:
            comparison.bind(pair, correctness)
        if performance is not None:
            performance.bind(plan, pair, correctness, comparison)
            reopened, file_sha = load_cumulative_performance(
                Path(performance_ref.path),
                expected_file_sha256=performance_ref.sha256)
            if reopened != performance or file_sha != performance_ref.sha256:
                raise CompositionError(
                    "pending cumulative performance reference changed")

    def _terminalize(
            self, state: dict[str, Any], *, plan: CompositionPlan,
            disposition: str, scientific: bool,
            correctness: FullCorrectness | None,
            comparison: IncrementalComparison | None,
            cumulative_performance: CumulativePerformance | None,
            cumulative_performance_ref: CumulativePerformanceRef | None,
            admitted: CompositionAuthority | None, reason_code: str,
            infrastructure_receipt_sha256: str | None = None,
            attribution_receipt_sha256: str | None = None,
    ) -> dict[str, Any]:
        if scientific and state["scientific_attempts"] >= \
                state["max_scientific_attempts"]:
            raise CompositionError("composition scientific budget is exhausted")
        lever = plan.candidate.accepted[-1]
        body = {
            "schema": "epyc.autokernel.cumulative_composition_terminal.v3",
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
            "cumulative_performance": (
                None if cumulative_performance is None else
                cumulative_performance.to_dict()),
            "cumulative_performance_ref": (
                None if cumulative_performance_ref is None else
                cumulative_performance_ref.to_dict()),
            "correctness_result_sha256":
                None if correctness is None else correctness.result_sha256,
            "comparison_result_sha256":
                None if comparison is None else comparison.result_sha256,
            "cumulative_performance_result_sha256": (
                None if cumulative_performance is None else
                cumulative_performance.result_sha256),
            "promotion_eligible": (
                False if cumulative_performance is None else
                cumulative_performance.promotion_eligible),
            "promotion_reason": (
                "missing_cumulative_production_comparison"
                if cumulative_performance is None else
                cumulative_performance.promotion_reason),
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
        terminal_sha256 = _terminal_decision_sha256(body)
        if (cumulative_performance is not None
                and cumulative_performance.composition_terminal_sha256 !=
                    terminal_sha256):
            raise CompositionError(
                "cumulative performance names another terminal decision")
        terminal = {**body, "terminal_sha256": terminal_sha256}
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
            "cumulative_performance", "cumulative_performance_ref",
            "correctness_result_sha256", "comparison_result_sha256",
            "cumulative_performance_result_sha256", "promotion_eligible",
            "promotion_reason",
            "admitted_authority_sha256", "reason_code",
            "infrastructure_receipt_sha256", "terminal_sha256",
            "attribution_receipt_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CompositionError("composition terminal has an inexact schema")
        if value.get("schema") != "epyc.autokernel.cumulative_composition_terminal.v3":
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
                    "cumulative_performance_result_sha256",
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
        performance = (
            None if value["cumulative_performance"] is None else
            CumulativePerformance.from_dict(value["cumulative_performance"]))
        performance_ref = (
            None if value["cumulative_performance_ref"] is None else
            CumulativePerformanceRef.from_dict(
                value["cumulative_performance_ref"]))
        if pair is not None:
            pair.bind_plan(plan)
        if correctness is not None:
            if pair is None:
                raise CompositionError("terminal correctness lacks build pair")
            correctness.bind_pair(pair)
        if comparison is not None:
            if correctness is None or pair is None:
                raise CompositionError("terminal comparison lacks prerequisite evidence")
            comparison.bind(pair, correctness)
        if performance is not None:
            if (comparison is None or correctness is None or pair is None
                    or performance_ref is None):
                raise CompositionError(
                    "terminal cumulative performance lacks prerequisites")
            performance.bind(plan, pair, correctness, comparison)
            reopened, file_sha = load_cumulative_performance(
                Path(performance_ref.path),
                expected_file_sha256=performance_ref.sha256)
            if reopened != performance or file_sha != performance_ref.sha256:
                raise CompositionError(
                    "terminal cumulative performance reference changed")
        elif performance_ref is not None:
            raise CompositionError(
                "terminal cumulative performance reference is orphaned")
        if ((None if correctness is None else correctness.result_sha256) !=
                value["correctness_result_sha256"]
                or (None if comparison is None else comparison.result_sha256) !=
                   value["comparison_result_sha256"]
                or (None if performance is None else
                    performance.result_sha256) !=
                   value["cumulative_performance_result_sha256"]):
            raise CompositionError("terminal evidence hashes changed")
        expected_promotion = (
            False if performance is None else performance.promotion_eligible)
        expected_promotion_reason = (
            "missing_cumulative_production_comparison"
            if performance is None else performance.promotion_reason)
        if (value["promotion_eligible"] is not expected_promotion
                or value["promotion_reason"] != expected_promotion_reason):
            raise CompositionError(
                "terminal promotion decision differs from cumulative evidence")
        disposition = value["disposition"]
        shape = (
            value["scientific_budget_spent"],
            value["correctness_result_sha256"] is not None,
            value["comparison_result_sha256"] is not None,
            value["cumulative_performance_result_sha256"] is not None,
            value["admitted_authority_sha256"] is not None,
            value["infrastructure_receipt_sha256"] is not None,
            value["attribution_receipt_sha256"] is not None,
        )
        expected_shapes = {
            "admitted": (True, True, True, True, True, False, False),
            "incremental_rollback": (
                True, True, True, True, False, False, False),
            "correctness_rollback": (
                True, True, False, False, False, False, False),
            "attribution_rollback": (
                True, True, False, False, False, False, True),
            "infrastructure_rollback": (
                False, False, False, False, False, True, False),
        }
        if expected_shapes.get(disposition) != shape:
            raise CompositionError("composition terminal disposition/evidence disagree")
        expected_reason_codes = {
            "admitted": (
                "incremental_admitted_promotion_eligible"
                if performance is not None and performance.promotion_eligible
                else "incremental_admitted_" +
                     (performance.promotion_reason
                      if performance is not None else "missing")),
            "incremental_rollback": (
                "incremental_" + comparison.classification
                if comparison is not None else "incremental_missing"),
            "correctness_rollback": "current_full_correctness_failed",
            "attribution_rollback": "exact_route_authority_failed",
        }
        if (disposition in expected_reason_codes
                and value["reason_code"] != expected_reason_codes[disposition]):
            raise CompositionError(
                "composition terminal reason differs from its decision")
        if (disposition == "admitted"
                and value["admitted_authority_sha256"] !=
                    plan.candidate.authority_sha256):
            raise CompositionError("admitted authority does not bind terminal plan")
        if value["terminal_sha256"] != _terminal_decision_sha256(value):
            raise CompositionError("composition terminal identity changed")
        if (performance is not None
                and performance.composition_terminal_sha256 !=
                    value["terminal_sha256"]):
            raise CompositionError(
                "composition performance names another terminal decision")
