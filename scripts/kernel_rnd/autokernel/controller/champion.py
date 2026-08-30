#!/usr/bin/env python3
"""Deterministic AutoKernel frontier, composition, and champion maintenance.

This is the deliberately lean replacement for the removed AK4 strategy plane.  It
does not choose research ideas, edit a tree, build, benchmark, launch a process, or
prepare a release.  It consumes schema-validated journal records and delegates the
one operation that needs execution -- rebuilding and evaluating a composed source
tree -- through an injected ``CompositionRunner``.

The primary safety property is negative: member results are never combined into a
new result.  A champion can cite only a *combined candidate's* passing T0/T1/T2
events against the current sealed production anchor.  Member performance fields are
not read anywhere in this module.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence

from .. import journal, schemas
from . import build_recipe as recipes
from .shared import ControllerError

__all__ = [
    "ChampionError", "EvidenceRefused", "CompatibilityRefused",
    "CompositionRefused", "AnchorMoved", "LifecycleRole", "AnchorArtifact",
    "AnchorIdentity", "T2Cell", "EvaluatorIdentity", "CompositionEvidence", "BankingVerdict", "CandidateSnapshot",
    "SourceTreeState", "CompatibilityReport", "CompositionRequest",
    "CompositionRunner", "ReanchorRunner", "JournalSnapshot",
    "read_validated_snapshot", "project_source_tree", "compatibility",
    "compatible_groups", "composition_request", "append_idempotent",
    "CompositionReceipt", "champion_build_recipe",
    "promote_composition", "seed_champion", "record_no_champion", "record_rejected_composition",
    "record_anchor_moved", "reanchor_champion", "champion_branch",
]


class ChampionError(ControllerError):
    """Base for fail-closed champion-plane refusals."""


class EvidenceRefused(ChampionError):
    """A journal record is valid syntactically but insufficient for this use."""


class CompatibilityRefused(ChampionError):
    """Candidates cannot safely coexist on one source tree."""


class CompositionRefused(ChampionError):
    """The injected runner did not return a bound, green combined candidate."""


class AnchorMoved(ChampionError):
    """The denominator changed; old comparison evidence has no authority."""


class LifecycleRole(str, Enum):
    """Names that must not collapse into one boolean called ``best``."""

    PROPOSED = "proposed"
    FRONTIER = "frontier"
    BANKED = "banked"
    CHAMPION_MEMBER = "champion_member"
    COMPOSED_CHAMPION = "composed_champion"
    PRODUCTION_INCUMBENT = "production_incumbent"
    SEALED_RELEASE_CANDIDATE = "sealed_release_candidate"


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA_RE = schemas.SHA256_RE
_PREDICATE_LITERAL_RE = re.compile(
    r"^\s*(?P<neg>!)?\s*(?P<name>[A-Za-z_][A-Za-z0-9_.:-]*)"
    r"(?:\s*(?:==|=)\s*(?P<value>[^\s]+))?\s*$"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise EvidenceRefused(f"{name} must be a non-empty NUL-free string")
    return value


def _commit(value: Any, name: str) -> str:
    value = _text(value, name)
    if not _COMMIT_RE.fullmatch(value):
        raise EvidenceRefused(f"{name} must be a 40-character lowercase commit")
    return value


def _sha(value: Any, name: str) -> str:
    value = _text(value, name)
    if not _SHA_RE.fullmatch(value) or schemas.is_placeholder_digest(value):
        raise EvidenceRefused(f"{name} must be a non-placeholder lowercase sha256")
    return value


def _string_tuple(value: Any, name: str, *, sorted_unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise EvidenceRefused(f"{name} must be a list of non-empty strings")
    values = tuple(value)
    if len(set(values)) != len(values):
        raise EvidenceRefused(f"{name} must not contain duplicates")
    if sorted_unique and values != tuple(sorted(values)):
        raise EvidenceRefused(f"{name} must be in canonical sorted order")
    return values


def champion_build_recipe(record: Mapping[str, Any]) -> recipes.BuildRecipe:
    """The build recipe a champion record was built with.

    A champion used to name a source tree and say NOTHING about how it was built,
    which made a build-flag improvement inexpressible as a champion advance: two
    champions could differ only in their compiler defines and be indistinguishable
    on the record. The recipe is the missing arm, and it is the SAME versioned
    object `build_recipe.py` already enforces -- so every flag is named and every
    divergence from production carries its reason, here as well as at the builder.

    Fail-closed: a champion that cannot state its build is refused rather than
    defaulted to the house recipe. Defaulting is exactly how `GGML_HIP_ROCWMMA_FATTN`
    reached production screening as an unset variable.
    """
    if not isinstance(record, Mapping):
        raise EvidenceRefused("champion record must be a mapping")
    raw = record.get("build_recipe")
    if not isinstance(raw, Mapping):
        raise EvidenceRefused("champion carries no build_recipe block")
    if raw.get("schema") != recipes.RECIPE_SCHEMA:
        raise EvidenceRefused(
            f"champion build_recipe is not a {recipes.RECIPE_SCHEMA} record")
    flags = raw.get("flags")
    if not isinstance(flags, list) or not flags:
        raise EvidenceRefused("champion build_recipe names no flags")
    try:
        return recipes.from_flags(_text(raw.get("name"), "build_recipe.name"),
                                  flags, notes=str(raw.get("notes") or ""))
    except recipes.BuildRecipeError as exc:
        raise EvidenceRefused(f"champion build_recipe is invalid: {exc}") from exc


@dataclass(frozen=True, order=True)
class AnchorArtifact:
    """One backend/tool denominator within a sealed source-tree anchor."""

    backend: str
    tool: str
    binary_sha256: str
    linkage_sha256: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise EvidenceRefused(f"unknown anchor backend {self.backend!r}")
        _text(self.tool, "anchor artifact tool")
        _sha(self.binary_sha256, "anchor artifact binary_sha256")
        _sha(self.linkage_sha256, "anchor artifact linkage_sha256")

    def to_dict(self) -> dict:
        return {"backend": self.backend, "tool": self.tool,
                "binary_sha256": self.binary_sha256,
                "linkage_sha256": self.linkage_sha256}


@dataclass(frozen=True)
class AnchorIdentity:
    """The complete sealed denominator for every backend/tool of one tree."""

    source_tree: str
    branch: str
    commit: str
    artifacts: tuple[AnchorArtifact, ...]
    sealed: bool = True

    def __post_init__(self) -> None:
        if self.source_tree not in schemas.SOURCE_TREES:
            raise EvidenceRefused(f"unknown source tree {self.source_tree!r}")
        _text(self.branch, "anchor.branch")
        _commit(self.commit, "anchor.commit")
        if not self.artifacts or tuple(sorted(set(self.artifacts))) != self.artifacts:
            raise EvidenceRefused("anchor.artifacts must be non-empty, unique, canonically sorted")
        expected_backends = {backend for backend, tree in schemas.SOURCE_TREE_BY_BACKEND.items()
                             if tree == self.source_tree}
        actual_backends = {artifact.backend for artifact in self.artifacts}
        if actual_backends != expected_backends:
            raise EvidenceRefused(
                f"anchor artifacts cover {sorted(actual_backends)}, source tree requires "
                f"{sorted(expected_backends)}")
        keys = [(artifact.backend, artifact.tool) for artifact in self.artifacts]
        if len(set(keys)) != len(keys):
            raise EvidenceRefused("anchor artifacts repeat a backend/tool identity")
        if self.sealed is not True:
            raise EvidenceRefused("a champion denominator must be a sealed anchor")

    def to_dict(self) -> dict:
        return {
            "source_tree": self.source_tree,
            "branch": self.branch,
            "commit": self.commit,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "sealed": True,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AnchorIdentity":
        if not isinstance(raw, Mapping):
            raise EvidenceRefused("anchor identity must be a mapping")
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, list):
            raise EvidenceRefused("anchor.artifacts must be a list")
        return cls(
            source_tree=raw.get("source_tree"), branch=raw.get("branch"),
            commit=raw.get("commit"),
            artifacts=tuple(AnchorArtifact(
                item.get("backend"), item.get("tool"), item.get("binary_sha256"),
                item.get("linkage_sha256")) for item in artifacts
                if isinstance(item, Mapping)),
            sealed=raw.get("sealed"),
        )

    def same_denominator(self, other: "AnchorIdentity") -> bool:
        return isinstance(other, AnchorIdentity) and self.to_dict() == other.to_dict()

    def artifact(self, backend: str, tool: str) -> AnchorArtifact:
        matches = [item for item in self.artifacts
                   if item.backend == backend and item.tool == tool]
        if len(matches) != 1:
            raise EvidenceRefused(
                f"anchor has {len(matches)} identities for {backend}/{tool}")
        return matches[0]


@dataclass(frozen=True, order=True)
class T2Cell:
    backend: str
    phase: str
    scope_manifest_sha256: str
    protocol_id: str
    metric: str
    metric_direction: str

    def __post_init__(self) -> None:
        if self.backend not in schemas.BACKENDS:
            raise EvidenceRefused(f"unknown T2 backend {self.backend!r}")
        _text(self.phase, "T2 phase")
        _sha(self.scope_manifest_sha256, "T2 scope manifest")
        _text(self.protocol_id, "T2 protocol id")
        _text(self.metric, "T2 metric")
        if self.metric_direction not in schemas.METRIC_DIRECTIONS:
            raise EvidenceRefused(f"invalid T2 metric direction {self.metric_direction!r}")

    def to_dict(self) -> dict:
        return {"backend": self.backend, "phase": self.phase,
                "scope_manifest_sha256": self.scope_manifest_sha256,
                "protocol_id": self.protocol_id, "metric": self.metric,
                "metric_direction": self.metric_direction}


@dataclass(frozen=True)
class EvaluatorIdentity:
    evaluator_id: str
    bundle_sha256: str
    runtime_source_label_ref: str
    protocol_ids: tuple[str, ...]
    t2_cells: tuple[T2Cell, ...] = ()

    def __post_init__(self) -> None:
        _text(self.evaluator_id, "evaluator.id")
        _sha(self.bundle_sha256, "evaluator.bundle_sha256")
        _text(self.runtime_source_label_ref, "evaluator.runtime_source_label_ref")
        if not self.protocol_ids or tuple(sorted(set(self.protocol_ids))) != self.protocol_ids:
            raise EvidenceRefused("evaluator.protocol_ids must be non-empty, unique, sorted")
        for item in self.protocol_ids:
            _text(item, "evaluator.protocol_ids[]")
        if tuple(sorted(set(self.t2_cells))) != self.t2_cells:
            raise EvidenceRefused("evaluator.t2_cells must be unique and canonically sorted")
        pairs = [(cell.backend, cell.phase) for cell in self.t2_cells]
        if len(set(pairs)) != len(pairs):
            raise EvidenceRefused("evaluator.t2_cells repeats a backend/phase cell")

    def to_dict(self) -> dict:
        return {"id": self.evaluator_id, "bundle_sha256": self.bundle_sha256,
                "runtime_source_label_ref": self.runtime_source_label_ref,
                "protocol_ids": list(self.protocol_ids),
                "t2_cells": [cell.to_dict() for cell in self.t2_cells]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvaluatorIdentity":
        if not isinstance(raw, Mapping):
            raise EvidenceRefused("evaluator identity must be a mapping")
        cells = raw.get("t2_cells")
        if not isinstance(cells, list):
            raise EvidenceRefused("evaluator.t2_cells must be a list")
        parsed: list[T2Cell] = []
        for item in cells:
            if not isinstance(item, Mapping):
                raise EvidenceRefused("evaluator.t2_cells[] must be a mapping")
            parsed.append(T2Cell(
                item.get("backend"), item.get("phase"),
                item.get("scope_manifest_sha256"), item.get("protocol_id"),
                item.get("metric"), item.get("metric_direction")))
        return cls(
            raw.get("id"), raw.get("bundle_sha256"),
            raw.get("runtime_source_label_ref"),
            _string_tuple(raw.get("protocol_ids"), "evaluator.protocol_ids"),
            tuple(parsed))

    def same_runtime(self, other: "EvaluatorIdentity") -> bool:
        return isinstance(other, EvaluatorIdentity) \
            and self.evaluator_id == other.evaluator_id \
            and self.bundle_sha256 == other.bundle_sha256 \
            and self.runtime_source_label_ref == other.runtime_source_label_ref \
            and self.protocol_ids == other.protocol_ids


@dataclass(frozen=True)
class CompositionEvidence:
    """Actual source evidence written by the candidate producer, never a proposal."""

    source_tree: str
    production_base_commit: str
    candidate_source_commit: str
    patch_bundle_sha256: str
    actual_files: tuple[str, ...]
    actual_hunk_ids: tuple[str, ...]
    actual_symbols: tuple[str, ...]
    derived_surface_tokens: tuple[str, ...]
    traced_surface_tokens: tuple[str, ...]
    feature_flag_assignments: tuple[tuple[str, str], ...]
    dispatch_predicates: tuple[str, ...]
    mechanism_id: str
    change_class: str
    evaluator: EvaluatorIdentity

    @classmethod
    def from_candidate(cls, candidate: Mapping[str, Any], *, campaign: Mapping[str, Any]) -> "CompositionEvidence":
        violations = schemas.validate_candidate(candidate)
        if violations:
            raise EvidenceRefused("invalid candidate record: " + "; ".join(violations))
        if not isinstance(campaign, Mapping) or schemas.validate_campaign(campaign):
            raise EvidenceRefused("candidate campaign is missing or invalid")
        raw = candidate.get("composition_evidence")
        if not isinstance(raw, Mapping):
            raise EvidenceRefused("candidate has no validated composition_evidence block")
        flags = raw.get("feature_flag_assignments")
        if not isinstance(flags, Mapping):
            raise EvidenceRefused("composition_evidence.feature_flag_assignments must be a mapping")
        canonical_flags: list[tuple[str, str]] = []
        for key in sorted(flags):
            _text(key, "feature flag name")
            value = flags[key]
            if isinstance(value, (dict, list, tuple, set)) or value is None:
                raise EvidenceRefused(f"feature flag {key!r} must have a canonical scalar value")
            canonical_flags.append((key, schemas.canonical_json(value)))
        evaluator = EvaluatorIdentity(
            evaluator_id=raw.get("evaluator_id"),
            bundle_sha256=raw.get("evaluator_bundle_sha256"),
            runtime_source_label_ref=raw.get("evaluator_runtime_source_label_ref"),
            protocol_ids=_string_tuple(raw.get("protocol_ids"), "composition_evidence.protocol_ids"),
        )
        evidence = cls(
            source_tree=raw.get("source_tree"),
            production_base_commit=_commit(raw.get("production_base_commit"), "composition_evidence.production_base_commit"),
            candidate_source_commit=_commit(raw.get("candidate_source_commit"), "composition_evidence.candidate_source_commit"),
            patch_bundle_sha256=_sha(raw.get("patch_bundle_sha256"), "composition_evidence.patch_bundle_sha256"),
            actual_files=_string_tuple(raw.get("actual_files"), "composition_evidence.actual_files"),
            actual_hunk_ids=_string_tuple(raw.get("actual_hunk_ids"), "composition_evidence.actual_hunk_ids"),
            actual_symbols=_string_tuple(raw.get("actual_symbols"), "composition_evidence.actual_symbols"),
            derived_surface_tokens=_string_tuple(raw.get("derived_surface_tokens"), "composition_evidence.derived_surface_tokens"),
            traced_surface_tokens=_string_tuple(raw.get("traced_surface_tokens"), "composition_evidence.traced_surface_tokens"),
            feature_flag_assignments=tuple(canonical_flags),
            dispatch_predicates=_string_tuple(raw.get("dispatch_predicates"), "composition_evidence.dispatch_predicates"),
            mechanism_id=_text(raw.get("mechanism_id"), "composition_evidence.mechanism_id"),
            change_class=_text(raw.get("change_class"), "composition_evidence.change_class"),
            evaluator=evaluator,
        )
        evidence._cross_check(candidate, campaign)
        return evidence

    def _cross_check(self, candidate: Mapping[str, Any], campaign: Mapping[str, Any]) -> None:
        if self.source_tree not in schemas.SOURCE_TREES or campaign.get("source_tree") != self.source_tree:
            raise EvidenceRefused("composition source tree contradicts the validated campaign")
        ancestry = candidate["ancestry"]
        worktree = candidate["worktree"]
        snapshot = candidate["source_snapshot"]
        evaluator = candidate["evaluator"]
        if ancestry.get("production_base_commit") != self.production_base_commit:
            raise EvidenceRefused("composition production base contradicts candidate.ancestry")
        if campaign["production_anchor"].get("commit") != self.production_base_commit:
            raise EvidenceRefused("composition production base contradicts campaign anchor")
        if worktree.get("source_commit") != self.candidate_source_commit:
            raise EvidenceRefused("composition source commit contradicts candidate.worktree")
        if snapshot.get("patch_bundle_sha256") != self.patch_bundle_sha256:
            raise EvidenceRefused("composition patch digest contradicts candidate.source_snapshot")
        if evaluator.get("id") != self.evaluator.evaluator_id \
                or evaluator.get("bundle_sha256") != self.evaluator.bundle_sha256 \
                or evaluator.get("runtime_source_label_ref") != self.evaluator.runtime_source_label_ref:
            raise EvidenceRefused("composition evaluator contradicts candidate.evaluator")
        if self.change_class not in schemas.CHANGE_CLASSES:
            raise EvidenceRefused(f"unknown composition change_class {self.change_class!r}")
        surface = candidate.get("affected_surface", {})
        if surface.get("reconciled") is not True or not self.derived_surface_tokens or not self.traced_surface_tokens:
            raise EvidenceRefused("composition requires non-empty derived and traced reconciled surfaces")
        if set(self.traced_surface_tokens) - set(self.derived_surface_tokens):
            raise EvidenceRefused("traced surface is not a subset of the derived surface")
        for symbol in self.actual_symbols:
            if ":" not in symbol or symbol.startswith(":") or symbol.endswith(":"):
                raise EvidenceRefused(
                    f"composition symbol {symbol!r} is not canonical path:symbol")

    def flags(self) -> dict[str, str]:
        return dict(self.feature_flag_assignments)

    def to_dict(self) -> dict:
        return {
            "source_tree": self.source_tree,
            "production_base_commit": self.production_base_commit,
            "candidate_source_commit": self.candidate_source_commit,
            "patch_bundle_sha256": self.patch_bundle_sha256,
            "actual_files": list(self.actual_files),
            "actual_hunk_ids": list(self.actual_hunk_ids),
            "actual_symbols": list(self.actual_symbols),
            "derived_surface_tokens": list(self.derived_surface_tokens),
            "traced_surface_tokens": list(self.traced_surface_tokens),
            "feature_flag_assignments": {k: v for k, v in self.feature_flag_assignments},
            "dispatch_predicates": list(self.dispatch_predicates),
            "mechanism_id": self.mechanism_id,
            "change_class": self.change_class,
            "evaluator": self.evaluator.to_dict(),
        }


@dataclass(frozen=True)
class BankingVerdict:
    """Write-side §9.6 disposition; ``status=banked`` alone is not evidence."""

    disposition: str
    qualifying_axis: str
    evaluation_event_ids: tuple[str, ...]
    non_dominated_check_ref: Optional[str]

    @classmethod
    def from_candidate(cls, record: Mapping[str, Any],
                       evaluations: Sequence[Mapping[str, Any]]) -> "BankingVerdict":
        raw = record.get("banking_verdict")
        if not isinstance(raw, Mapping):
            raise EvidenceRefused("banked candidate has no banking_verdict")
        if raw.get("disposition") != "banked":
            raise EvidenceRefused("banking_verdict.disposition is not banked")
        t0 = raw.get("t0")
        sentinels = raw.get("sentinels")
        dispatch = raw.get("real_path_dispatch")
        mechanism = raw.get("mechanism")
        axis_record = raw.get("qualifying_axis")
        if not all(isinstance(item, Mapping)
                   for item in (t0, sentinels, dispatch, mechanism, axis_record)):
            raise EvidenceRefused("banking_verdict gate blocks must be mappings")
        assert isinstance(t0, Mapping) and isinstance(sentinels, Mapping)
        assert isinstance(dispatch, Mapping) and isinstance(mechanism, Mapping)
        assert isinstance(axis_record, Mapping)
        t0_id = _text(t0.get("all_pass_event_id"),
                      "banking_verdict.t0.all_pass_event_id")
        sentinel_ids = _string_tuple(
            sentinels.get("required_all_pass_event_ids"),
            "banking_verdict.sentinels.required_all_pass_event_ids")
        if dispatch.get("resolution") != "confirmed":
            raise EvidenceRefused("real-path dispatch must be confirmed")
        dispatch_id = _text(dispatch.get("event_id"),
                            "banking_verdict.real_path_dispatch.event_id")
        _text(dispatch.get("gate_id"), "banking_verdict.real_path_dispatch.gate_id")
        if mechanism.get("resolution") not in {"confirmed", "explained"}:
            raise EvidenceRefused("mechanism must be confirmed or explained")
        mechanism_id = _text(mechanism.get("event_id"),
                             "banking_verdict.mechanism.event_id")
        _text(mechanism.get("gate_id"), "banking_verdict.mechanism.gate_id")
        axis = axis_record.get("axis")
        allowed_axes = {"throughput", "context_capacity", "vram", "ram",
                        "model_load_time", "run_variance"}
        if axis not in allowed_axes:
            raise EvidenceRefused(f"unknown qualifying axis {axis!r}")
        axis_id = _text(axis_record.get("evaluation_event_id"),
                        "banking_verdict.qualifying_axis.evaluation_event_id")
        non_dominated_ref: Optional[str] = None
        if axis == "throughput":
            if axis_record.get("resolution") != "above_floor":
                raise EvidenceRefused("throughput banking must resolve above_floor")
            values = []
            for key in ("observed_effect", "calibrated_floor",
                        "minimum_detectable_effect"):
                value = axis_record.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(float(value)):
                    raise EvidenceRefused(
                        f"banking_verdict.qualifying_axis.{key} must be finite")
                values.append(float(value))
            observed, floor, mde = values
            if floor < 0 or mde < 0 or observed < max(floor, mde):
                raise EvidenceRefused(
                    "throughput effect must meet both calibrated floor and MDE")
            if axis_record.get("non_dominated") is not None \
                    or axis_record.get("non_dominated_check_ref") is not None:
                raise EvidenceRefused(
                    "throughput banking cannot borrow a non-dominance result")
        else:
            if axis_record.get("resolution") != "non_dominated" \
                    or axis_record.get("non_dominated") is not True:
                raise EvidenceRefused(
                    "an alternate qualifying axis must be explicitly non-dominated")
            non_dominated_ref = _text(
                axis_record.get("non_dominated_check_ref"),
                "banking_verdict.qualifying_axis.non_dominated_check_ref")
        known = {event.get("event_id"): event for event in evaluations}
        required = {t0_id, dispatch_id, mechanism_id, axis_id, *sentinel_ids}
        if required - set(known):
            raise EvidenceRefused(
                f"banking evidence references missing events {sorted(required - set(known))}")
        for event_id in required:
            event = known[event_id]
            if event.get("status") != "pass" or _is_void(event):
                raise EvidenceRefused(f"banking evidence {event_id!r} is not a valid pass")
        if known[t0_id].get("tier") != "T0":
            raise EvidenceRefused("banking T0 reference is not a T0 event")
        return cls("banked", axis, tuple(sorted(required)), non_dominated_ref)


def _is_void(event: Mapping[str, Any]) -> bool:
    flags = event.get("integrity_flags")
    return event.get("status") in {"invalid", "inconclusive", "timeout", "crash", "rejected"} or (
        isinstance(flags, list) and any(isinstance(flag, str) and flag.startswith(schemas.VOID_FLAG_PREFIX) for flag in flags)
    )


def _event_matches_anchor(event: Mapping[str, Any], anchor: AnchorIdentity) -> bool:
    raw = event.get("anchor")
    backend = event.get("backend")
    tool = raw.get("tool") if isinstance(raw, Mapping) else None
    if not isinstance(raw, Mapping) or not isinstance(backend, str) \
            or not isinstance(tool, str) or raw.get("source_commit") != anchor.commit:
        return False
    try:
        artifact = anchor.artifact(backend, tool)
    except EvidenceRefused:
        return False
    return raw.get("binary_sha256") == artifact.binary_sha256 \
        and raw.get("linkage_sha256") == artifact.linkage_sha256


@dataclass(frozen=True)
class CandidateSnapshot:
    record: Mapping[str, Any]
    record_event_id: str
    campaign: Mapping[str, Any]
    evaluations: tuple[Mapping[str, Any], ...]
    evidence: CompositionEvidence
    banking: Optional[BankingVerdict]

    @property
    def candidate_id(self) -> str:
        return self.record["candidate_id"]

    @property
    def parent_candidate_id(self) -> Optional[str]:
        return self.record.get("parent_candidate_id")

    def passing(self, tier: str, anchor: AnchorIdentity) -> tuple[Mapping[str, Any], ...]:
        return tuple(event for event in self.evaluations
                     if event.get("tier") == tier and event.get("status") == "pass"
                     and not _is_void(event) and _event_matches_anchor(event, anchor))

    def frontier_eligible(self, anchor: AnchorIdentity) -> bool:
        if self.record.get("status") != "banked":
            return False
        if self.banking is None or self.banking.disposition != "banked":
            return False
        if self.evidence.production_base_commit != anchor.commit:
            return False
        referenced = set(self.record.get("evaluation_event_ids") or ())
        if not referenced:
            return False
        if any(event.get("event_id") not in referenced for event in self.evaluations):
            return False
        return bool(self.passing("T0", anchor)) and bool(
            tuple(event for event in self.evaluations
                  if event.get("tier") in {"T1", "T1a", "T1b", "T1c"}
                  and event.get("status") == "pass" and not _is_void(event)
                  and _event_matches_anchor(event, anchor)))


@dataclass(frozen=True)
class JournalSnapshot:
    entries: tuple[journal.JournalEntry, ...]
    views: journal.Views


def read_validated_snapshot(book: journal.Journal) -> JournalSnapshot:
    if not isinstance(book, journal.Journal):
        raise TypeError("book must be a journal.Journal")
    entries = tuple(book.read_all())
    views = journal.rebuild_views(entries)
    journal.assert_views_consistent(entries, views)
    for entry in entries:
        if entry.kind in journal.SCHEMA_BOUND_KINDS:
            violations = schemas.validate_record(entry.payload)
            if violations:
                raise EvidenceRefused(f"journal event {entry.event_id} is invalid: " + "; ".join(violations))
    return JournalSnapshot(entries, views)


def _latest_event_id(entries: Sequence[journal.JournalEntry], kind: str, record_id: str) -> str:
    matches = [e for e in entries if e.kind == kind and e.record_id == record_id]
    if not matches:
        raise EvidenceRefused(f"no {kind} journal event resolves {record_id!r}")
    return max(matches, key=lambda e: e.seq).event_id


def _candidate_snapshot(snapshot: JournalSnapshot, candidate_id: str) -> CandidateSnapshot:
    record = snapshot.views.candidates.get(candidate_id)
    if not isinstance(record, Mapping):
        raise EvidenceRefused(f"candidate {candidate_id!r} is missing")
    campaign = snapshot.views.campaigns.get(record.get("campaign_id"))
    if not isinstance(campaign, Mapping):
        raise EvidenceRefused(f"candidate {candidate_id!r} has no campaign record")
    proposal = snapshot.views.proposals.get(record.get("proposal_id"))
    if not isinstance(proposal, Mapping):
        raise EvidenceRefused(f"candidate {candidate_id!r} has no proposal record")
    if proposal.get("schema") == schemas.SCHEMA_PROPOSAL_V4:
        provider_reference = proposal.get("provider_reference")
        if record.get("provider_reference") != provider_reference:
            raise EvidenceRefused(
                f"candidate {candidate_id!r} does not carry its proposal's provider identity")
        target_backend = provider_reference.get("target_backend") \
            if isinstance(provider_reference, Mapping) else None
        if campaign.get("backend") != target_backend:
            raise EvidenceRefused(
                f"candidate {candidate_id!r} provider targets {target_backend!r} but "
                f"campaign backend is {campaign.get('backend')!r}")
    referenced = record.get("evaluation_event_ids")
    if not isinstance(referenced, list):
        raise EvidenceRefused(f"candidate {candidate_id!r} has no evaluation id list")
    evaluations: list[Mapping[str, Any]] = []
    for event_id in referenced:
        event = snapshot.views.evaluations.get(event_id)
        if not isinstance(event, Mapping):
            raise EvidenceRefused(f"candidate {candidate_id!r} references missing evaluation {event_id!r}")
        if event.get("candidate_id") != candidate_id:
            raise EvidenceRefused(f"evaluation {event_id!r} belongs to another candidate")
        evaluations.append(event)
    try:
        banking = BankingVerdict.from_candidate(record, evaluations)
    except EvidenceRefused:
        banking = None
    return CandidateSnapshot(
        record=record,
        record_event_id=_latest_event_id(snapshot.entries, journal.KIND_CANDIDATE_RECORDED, candidate_id),
        campaign=campaign,
        evaluations=tuple(evaluations),
        evidence=CompositionEvidence.from_candidate(record, campaign=campaign),
        banking=banking,
    )


@dataclass(frozen=True)
class SourceTreeState:
    source_tree: str
    incumbent: AnchorIdentity
    proposed: tuple[str, ...]
    banked: tuple[str, ...]
    frontier: tuple[str, ...]
    champion_members: tuple[str, ...]
    composed_champion: Optional[str]
    sealed_release_candidates: tuple[str, ...]
    active_champion: Optional[Mapping[str, Any]]
    candidates: Mapping[str, CandidateSnapshot]

    def roles(self, identity: str) -> frozenset[LifecycleRole]:
        roles: set[LifecycleRole] = set()
        if identity in self.proposed:
            roles.add(LifecycleRole.PROPOSED)
        if identity in self.banked:
            roles.add(LifecycleRole.BANKED)
        if identity in self.frontier:
            roles.add(LifecycleRole.FRONTIER)
        if identity in self.champion_members:
            roles.add(LifecycleRole.CHAMPION_MEMBER)
        if identity == self.composed_champion:
            roles.add(LifecycleRole.COMPOSED_CHAMPION)
        if identity in self.sealed_release_candidates:
            roles.add(LifecycleRole.SEALED_RELEASE_CANDIDATE)
        return frozenset(roles)

    @property
    def incumbent_role(self) -> LifecycleRole:
        return LifecycleRole.PRODUCTION_INCUMBENT


def project_source_tree(snapshot: JournalSnapshot, anchor: AnchorIdentity) -> SourceTreeState:
    campaigns = {cid: value for cid, value in snapshot.views.campaigns.items()
                 if value.get("source_tree") == anchor.source_tree}
    proposals = tuple(sorted(pid for pid, value in snapshot.views.proposals.items()
                             if value.get("campaign_id") in campaigns))
    candidates: dict[str, CandidateSnapshot] = {}
    banked: list[str] = []
    frontier: list[str] = []
    for candidate_id, record in snapshot.views.candidates.items():
        if record.get("campaign_id") not in campaigns:
            continue
        try:
            candidate = _candidate_snapshot(snapshot, candidate_id)
        except EvidenceRefused:
            # Invalid/incomplete candidates stay in the journal but have no frontier
            # authority.  Projection is fail-closed, not globally unavailable.
            continue
        candidates[candidate_id] = candidate
        if record.get("status") == "banked":
            banked.append(candidate_id)
        if candidate.frontier_eligible(anchor):
            frontier.append(candidate_id)
    champion = snapshot.views.champions.get(anchor.source_tree)
    members: tuple[str, ...] = ()
    combined: Optional[str] = None
    if isinstance(champion, Mapping):
        members = tuple(champion.get("member_candidates") or ())
        combined = champion.get("combined_candidate_id")
        # A record anchored elsewhere is retained as history, never treated active.
        raw_anchor = champion.get("anchor")
        try:
            anchored_here = isinstance(raw_anchor, Mapping) \
                and AnchorIdentity.from_dict(raw_anchor).same_denominator(anchor)
        except EvidenceRefused:
            anchored_here = False
        if not anchored_here or champion.get("status") == "anchor_moved":
            champion = None
            members = ()
            combined = None
    sealed: list[str] = []
    for package in snapshot.views.release_packages.values():
        if package.get("source_tree") != anchor.source_tree:
            continue
        raw = package.get("sealed_candidate")
        if isinstance(raw, Mapping) and isinstance(raw.get("candidate_id"), str):
            sealed.append(raw["candidate_id"])
    return SourceTreeState(
        source_tree=anchor.source_tree, incumbent=anchor, proposed=proposals,
        banked=tuple(sorted(banked)), frontier=tuple(sorted(frontier)),
        champion_members=members, composed_champion=combined,
        sealed_release_candidates=tuple(sorted(set(sealed))),
        active_champion=champion, candidates=candidates,
    )


@dataclass(frozen=True)
class CompatibilityReport:
    candidate_ids: tuple[str, ...]
    compatible: bool
    conflicts: tuple[str, ...]
    observations: tuple[str, ...]
    evidence_sha256: str


def _predicate_assignments(predicates: Sequence[str]) -> tuple[dict[str, str], tuple[str, ...]]:
    assignments: dict[str, str] = {}
    opaque: list[str] = []
    for predicate in predicates:
        match = _PREDICATE_LITERAL_RE.fullmatch(predicate)
        if match is None:
            opaque.append(predicate)
            continue
        name = match.group("name")
        value = match.group("value") or ("false" if match.group("neg") else "true")
        if name in assignments and assignments[name] != value:
            raise CompatibilityRefused(
                f"candidate contains contradictory dispatch predicates for {name!r}")
        assignments[name] = value
    return assignments, tuple(opaque)


def compatibility(candidates: Sequence[CandidateSnapshot], *, anchor: AnchorIdentity,
                  evaluator: EvaluatorIdentity) -> CompatibilityReport:
    if not candidates:
        raise CompatibilityRefused("compatibility needs at least one candidate")
    ordered = tuple(sorted(candidates, key=lambda c: c.candidate_id))
    conflicts: set[str] = set()
    observations: set[str] = set()
    seen_files: dict[str, str] = {}
    seen_hunks: dict[str, str] = {}
    seen_symbols: dict[str, str] = {}
    seen_flags: dict[str, tuple[str, str]] = {}
    seen_predicates: dict[str, tuple[str, str]] = {}
    seen_mechanisms: dict[str, str] = {}
    for candidate in ordered:
        cid = candidate.candidate_id
        evidence = candidate.evidence
        if evidence.source_tree != anchor.source_tree:
            conflicts.add(f"cross_tree:{cid}:{evidence.source_tree}!={anchor.source_tree}")
        if evidence.production_base_commit != anchor.commit:
            conflicts.add(f"stale_anchor:{cid}:{evidence.production_base_commit}!={anchor.commit}")
        if not evidence.evaluator.same_runtime(evaluator):
            conflicts.add(f"evaluator_or_protocol_mismatch:{cid}")
        if not candidate.frontier_eligible(anchor):
            conflicts.add(f"not_green_banked_frontier:{cid}")
        for path in evidence.actual_files:
            if path in seen_files:
                # The current evidence contract carries content-derived hunk ids,
                # not ranges.  Two edits to one file therefore cannot prove their
                # ranges disjoint and are conservatively explicit conflicts.
                conflicts.add(f"overlapping_file:{path}:{seen_files[path]}:{cid}")
            else:
                seen_files[path] = cid
        for hunk in evidence.actual_hunk_ids:
            if hunk in seen_hunks:
                conflicts.add(f"overlapping_hunk:{hunk}:{seen_hunks[hunk]}:{cid}")
            else:
                seen_hunks[hunk] = cid
        for symbol in evidence.actual_symbols:
            if symbol in seen_symbols:
                conflicts.add(f"overlapping_symbol:{symbol}:{seen_symbols[symbol]}:{cid}")
            else:
                seen_symbols[symbol] = cid
        for name, value in evidence.feature_flag_assignments:
            prior = seen_flags.get(name)
            if prior is not None and prior[0] != value:
                conflicts.add(f"mutually_exclusive_flag:{name}:{prior[1]}:{cid}")
            else:
                seen_flags[name] = (value, cid)
        parsed_predicates, opaque_predicates = _predicate_assignments(
            evidence.dispatch_predicates)
        if opaque_predicates and len(ordered) > 1:
            conflicts.add(f"opaque_dispatch_uncomposable:{cid}:{','.join(opaque_predicates)}")
        for name, value in parsed_predicates.items():
            prior = seen_predicates.get(name)
            if prior is not None and prior[0] != value:
                conflicts.add(f"mutually_exclusive_dispatch:{name}:{prior[1]}:{cid}")
            else:
                seen_predicates[name] = (value, cid)
        prior_mechanism = seen_mechanisms.get(evidence.mechanism_id)
        if prior_mechanism is not None:
            conflicts.add(f"same_mechanism_replacement:{evidence.mechanism_id}:{prior_mechanism}:{cid}")
        else:
            seen_mechanisms[evidence.mechanism_id] = cid
        overlap = sorted(set(evidence.derived_surface_tokens) & set(evidence.traced_surface_tokens))
        observations.update(f"reconciled_surface:{cid}:{token}" for token in overlap)
    body = {
        "anchor": anchor.to_dict(), "evaluator": evaluator.to_dict(),
        "candidates": [{"candidate_id": c.candidate_id, "evidence": c.evidence.to_dict()}
                       for c in ordered],
        "conflicts": sorted(conflicts), "observations": sorted(observations),
    }
    return CompatibilityReport(
        candidate_ids=tuple(c.candidate_id for c in ordered),
        compatible=not conflicts, conflicts=tuple(sorted(conflicts)),
        observations=tuple(sorted(observations)), evidence_sha256=schemas.content_hash(body),
    )


def compatible_groups(candidates: Sequence[CandidateSnapshot], *, anchor: AnchorIdentity,
                      evaluator: EvaluatorIdentity) -> tuple[tuple[CandidateSnapshot, ...], ...]:
    """Deterministic maximal compatible sets; no ranking and no gain inspection."""
    remaining = list(sorted(candidates, key=lambda c: c.candidate_id))
    groups: list[tuple[CandidateSnapshot, ...]] = []
    while remaining:
        group = [remaining.pop(0)]
        deferred: list[CandidateSnapshot] = []
        for candidate in remaining:
            if compatibility((*group, candidate), anchor=anchor, evaluator=evaluator).compatible:
                group.append(candidate)
            else:
                deferred.append(candidate)
        groups.append(tuple(group))
        remaining = deferred
    return tuple(groups)


def _lineage_order(candidates: Sequence[CandidateSnapshot]) -> tuple[CandidateSnapshot, ...]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    pending = set(by_id)
    ordered: list[CandidateSnapshot] = []
    while pending:
        ready = sorted(cid for cid in pending
                       if by_id[cid].parent_candidate_id not in pending)
        if not ready:
            raise CompositionRefused("candidate parent lineage contains a cycle")
        for cid in ready:
            ordered.append(by_id[cid])
            pending.remove(cid)
    return tuple(ordered)


@dataclass(frozen=True)
class CompositionRequest:
    request_sha256: str
    combined_candidate_id: str
    source_tree: str
    member_candidates: tuple[str, ...]
    member_record_event_ids: tuple[str, ...]
    parent_champion_event_id: Optional[str]
    anchor: AnchorIdentity
    evaluator: EvaluatorIdentity
    required_t2_cells: tuple[T2Cell, ...]
    compatibility_sha256: str
    absorbed_member_candidates: tuple[str, ...] = ()
    release_package_event_id: Optional[str] = None
    mode: str = "compose"
    #: The build the combined tree is compiled with. It is part of the request
    #: SPINE, so two requests over the same members and anchor that differ only in
    #: their defines get different `request_sha256` and `combined_candidate_id`.
    build_recipe: recipes.BuildRecipe = recipes.HOUSE_GPU_RECIPE

    def to_dict(self) -> dict:
        return {
            "request_sha256": self.request_sha256,
            "combined_candidate_id": self.combined_candidate_id,
            "source_tree": self.source_tree,
            "member_candidates": list(self.member_candidates),
            "member_record_event_ids": list(self.member_record_event_ids),
            "parent_champion_event_id": self.parent_champion_event_id,
            "anchor": self.anchor.to_dict(), "evaluator": self.evaluator.to_dict(),
            "required_t2_cells": [cell.to_dict() for cell in self.required_t2_cells],
            "compatibility_sha256": self.compatibility_sha256,
            "absorbed_member_candidates": list(self.absorbed_member_candidates),
            "release_package_event_id": self.release_package_event_id,
            "mode": self.mode,
            "build_recipe": self.build_recipe.to_dict(),
        }


def _required_t2_cells(candidates: Sequence[CandidateSnapshot],
                       evaluator: EvaluatorIdentity) -> tuple[T2Cell, ...]:
    required_pairs: dict[tuple[str, str], tuple[str, str]] = {}
    for candidate in candidates:
        backend = candidate.campaign.get("backend")
        objective = candidate.campaign.get("objective")
        scope = candidate.campaign.get("scope")
        if not isinstance(objective, Mapping) or not isinstance(scope, Mapping):
            raise CompositionRefused("candidate campaign has no objective/scope")
        phases = objective.get("phases")
        protocols = objective.get("protocol_by_phase")
        scope_sha = scope.get("derived_role_manifest_sha256")
        if not isinstance(phases, list) or not isinstance(protocols, Mapping):
            raise CompositionRefused("candidate campaign has no phase protocol matrix")
        for phase in phases:
            pair = (backend, phase)
            identity = (scope_sha, protocols.get(phase))
            if pair in required_pairs and required_pairs[pair] != identity:
                raise CompositionRefused(
                    f"candidate campaigns contradict T2 cell {pair}")
            required_pairs[pair] = identity
    by_pair = {(cell.backend, cell.phase): cell for cell in evaluator.t2_cells}
    missing = sorted(set(required_pairs) - set(by_pair))
    if missing:
        raise CompositionRefused(
            f"predeclared T2 cell coverage mismatch: missing={missing}")
    cells: list[T2Cell] = []
    for pair, (scope_sha, protocol_id) in sorted(required_pairs.items()):
        cell = by_pair[pair]
        if cell.scope_manifest_sha256 != scope_sha or cell.protocol_id != protocol_id:
            raise CompositionRefused(
                f"T2 cell {pair} does not match campaign scope/protocol")
        cells.append(cell)
    if not cells:
        raise CompositionRefused("composition has no required T2 cells")
    return tuple(cells)


def composition_request(candidates: Sequence[CandidateSnapshot], *, anchor: AnchorIdentity,
                        evaluator: EvaluatorIdentity,
                        parent_champion_event_id: Optional[str] = None,
                        mode: str = "compose",
                        build_recipe: Optional[recipes.BuildRecipe] = None
                        ) -> CompositionRequest:
    if mode not in {"compose", "reanchor"}:
        raise ValueError("mode must be compose or reanchor")
    recipe = recipes.HOUSE_GPU_RECIPE if build_recipe is None else build_recipe
    ordered = _lineage_order(candidates)
    report = compatibility(ordered, anchor=anchor, evaluator=evaluator)
    if not report.compatible:
        raise CompatibilityRefused("; ".join(report.conflicts))
    spine = {
        "source_tree": anchor.source_tree,
        "members": [c.candidate_id for c in ordered],
        "member_record_event_ids": [c.record_event_id for c in ordered],
        "parent_champion_event_id": parent_champion_event_id,
        "anchor": anchor.to_dict(), "evaluator": evaluator.to_dict(),
        "required_t2_cells": [cell.to_dict() for cell in _required_t2_cells(ordered, evaluator)],
        "compatibility_sha256": report.evidence_sha256, "mode": mode,
        "build_recipe_sha256": recipe.sha256(),
    }
    digest = schemas.content_hash(spine)
    return CompositionRequest(
        request_sha256=digest, combined_candidate_id=f"akc-composed-{digest[:24]}",
        source_tree=anchor.source_tree,
        member_candidates=tuple(c.candidate_id for c in ordered),
        member_record_event_ids=tuple(c.record_event_id for c in ordered),
        parent_champion_event_id=parent_champion_event_id,
        anchor=anchor, evaluator=evaluator,
        required_t2_cells=_required_t2_cells(ordered, evaluator),
        compatibility_sha256=report.evidence_sha256,
        absorbed_member_candidates=(), release_package_event_id=None,
        mode=mode, build_recipe=recipe,
    )


@dataclass(frozen=True)
class CompositionReceipt:
    """Durable journal envelope refs returned after injected execution."""

    request_event_id: str
    campaign_event_id: str
    proposal_event_id: str
    candidate_event_id: str
    evaluation_event_ids: tuple[str, ...]
    realized_cost: Mapping[str, Any]


class CompositionRunner(Protocol):
    def run_composition(self, book: journal.Journal, request: CompositionRequest,
                        request_event: journal.JournalEntry) -> CompositionReceipt: ...


class ReanchorRunner(Protocol):
    def run_reanchor(self, book: journal.Journal, request: CompositionRequest,
                     request_event: journal.JournalEntry) -> CompositionReceipt: ...


def append_idempotent(book: journal.Journal, kind: str,
                      payload: Mapping[str, Any]) -> journal.JournalEntry:
    """Append once; exact crash-resume duplicates resolve to the fsynced entry."""
    if kind not in journal.KINDS:
        raise ValueError(f"unknown journal kind {kind!r}")
    key = journal.RECORD_ID_KEY_BY_KIND.get(kind)
    record_id = payload.get(key) if key else None
    if record_id is None and kind in journal.NATIVE_KINDS:
        # Native facts without a domain id still get a deterministic content id;
        # the envelope's record_id makes concurrent replay inspectable.
        record_id = f"native-{schemas.content_hash(payload)}"
    canonical = schemas.canonical_json(payload)
    # The scan and append are one transaction.  Without this lock two controller
    # processes can both observe absence and append duplicate campaign work.
    with book.write_lock():
        matches = [entry for entry in book.read_all()
                   if entry.kind == kind
                   and (record_id is None or entry.record_id == record_id)]
        exact = [entry for entry in matches
                 if schemas.canonical_json(entry.payload) == canonical]
        if exact:
            return max(exact, key=lambda entry: entry.seq)
        # A champion has one stable view slot per branch and is intentionally
        # updated as it moves through no-champion, active, and reanchor states.
        if record_id is not None and matches and kind != journal.KIND_CHAMPION_UPDATED:
            raise CompositionRefused(
                f"{kind} identity {record_id!r} already exists with different bytes; "
                "crash recovery may replay an identical record, never rewrite one")
        return book.append(kind, payload, record_id=record_id)


def champion_branch(source_tree: str, anchor_commit: str) -> str:
    safe = source_tree.replace(".", "-")
    return f"ak/champion/{safe}-{anchor_commit[:12]}"


def _resolve_composition_receipt(snapshot: JournalSnapshot,
                                 receipt: CompositionReceipt,
                                 request: CompositionRequest,
                                 request_event: journal.JournalEntry
                                 ) -> tuple[CandidateSnapshot, journal.JournalEntry,
                                            tuple[journal.JournalEntry, ...]]:
    if not isinstance(receipt, CompositionReceipt):
        raise CompositionRefused("composition runner must return CompositionReceipt")
    if receipt.request_event_id != request_event.event_id:
        raise CompositionRefused("composition receipt is bound to another request event")
    by_id = {entry.event_id: entry for entry in snapshot.entries}
    fixed_refs = (
        (receipt.campaign_event_id, journal.KIND_CAMPAIGN_OPENED),
        (receipt.proposal_event_id, journal.KIND_PROPOSAL_RECORDED),
        (receipt.candidate_event_id, journal.KIND_CANDIDATE_RECORDED),
    )
    resolved: list[journal.JournalEntry] = []
    for event_id, kind in fixed_refs:
        entry = by_id.get(event_id)
        if entry is None or entry.kind != kind:
            raise CompositionRefused(
                f"composition receipt {event_id!r} does not resolve to {kind}")
        if entry.seq <= request_event.seq:
            raise CompositionRefused("composition evidence predates its durable request")
        resolved.append(entry)
    campaign_entry, proposal_entry, candidate_entry = resolved
    campaign = campaign_entry.payload
    campaign_violations = schemas.validate_campaign(campaign)
    if campaign_violations:
        raise CompositionRefused("combined campaign is invalid: " + "; ".join(campaign_violations))
    if campaign.get("source_tree") != request.source_tree \
            or campaign.get("production_anchor", {}).get("commit") != request.anchor.commit:
        raise CompositionRefused("combined campaign is not bound to the request tree and anchor")
    proposal = proposal_entry.payload
    proposal_violations = schemas.validate_proposal(proposal)
    if proposal_violations:
        raise CompositionRefused("combined proposal is invalid: " + "; ".join(proposal_violations))
    if proposal.get("campaign_id") != campaign.get("campaign_id"):
        raise CompositionRefused("combined proposal is not bound to the returned campaign")
    candidate = candidate_entry.payload
    violations = schemas.validate_candidate(candidate)
    if violations:
        raise CompositionRefused("combined candidate is invalid: " + "; ".join(violations))
    if candidate.get("candidate_id") != request.combined_candidate_id:
        raise CompositionRefused("combined candidate id is not bound to the request")
    lineage = candidate.get("composition_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("request_sha256") != request.request_sha256 \
            or tuple(lineage.get("member_candidates") or ()) != request.member_candidates \
            or tuple(lineage.get("absorbed_member_candidates") or ()) \
                != request.absorbed_member_candidates \
            or lineage.get("release_package_event_id") \
                != request.release_package_event_id \
            or lineage.get("parent_champion_event_id") != request.parent_champion_event_id \
            or lineage.get("mode") != request.mode:
        raise CompositionRefused("combined candidate carries no exact composition lineage binding")
    if candidate.get("campaign_id") != campaign.get("campaign_id"):
        raise CompositionRefused("combined candidate is not bound to the returned campaign")
    if candidate.get("proposal_id") != proposal.get("proposal_id"):
        raise CompositionRefused("combined candidate is not bound to the returned proposal")
    evidence = CompositionEvidence.from_candidate(candidate, campaign=campaign)
    if evidence.source_tree != request.source_tree or evidence.production_base_commit != request.anchor.commit:
        raise CompositionRefused("combined candidate was built on another tree or anchor")
    event_entries: list[journal.JournalEntry] = []
    events: list[Mapping[str, Any]] = []
    for journal_event_id in receipt.evaluation_event_ids:
        entry = by_id.get(journal_event_id)
        if entry is None or entry.kind != journal.KIND_EVALUATION_EVENT:
            raise CompositionRefused(
                f"composition evaluation ref {journal_event_id!r} does not resolve")
        if entry.seq <= request_event.seq:
            raise CompositionRefused("composition evaluation predates its durable request")
        event = entry.payload
        problems = schemas.validate_evaluation_event(event)
        if problems:
            raise CompositionRefused("combined evaluation is invalid: " + "; ".join(problems))
        if event.get("candidate_id") != request.combined_candidate_id:
            raise CompositionRefused("combined evaluation belongs to another candidate")
        if event.get("status") != "pass" or _is_void(event):
            raise CompositionRefused("combined evaluation is not a valid pass")
        event_evaluator = event.get("evaluator")
        if not isinstance(event_evaluator, Mapping) \
                or event_evaluator.get("id") != request.evaluator.evaluator_id \
                or event_evaluator.get("bundle_sha256") != request.evaluator.bundle_sha256 \
                or event_evaluator.get("runtime_source_label_ref") != request.evaluator.runtime_source_label_ref:
            raise CompositionRefused("combined evaluation used another evaluator bundle")
        protocol_id = (event.get("claim_grammar") or {}).get("protocol_id")
        if protocol_id not in request.evaluator.protocol_ids:
            raise CompositionRefused("combined evaluation used an undeclared protocol")
        if not _event_matches_anchor(event, request.anchor):
            raise CompositionRefused("combined evaluation used another anchor")
        event_entries.append(entry)
        events.append(event)
    tiers = {event.get("tier") for event in events}
    if not {"T0", "T1", "T2"}.issubset(tiers):
        raise CompositionRefused("combined candidate needs its own passing T0, T1, and T2")
    actual_t2: dict[tuple[str, str], T2Cell] = {}
    for event in events:
        if event.get("tier") != "T2":
            continue
        grammar = event.get("claim_grammar") or {}
        cell = T2Cell(
            event.get("backend"), event.get("phase"),
            event.get("scope_manifest_sha256"), grammar.get("protocol_id"),
            grammar.get("metric"), grammar.get("metric_direction"))
        key = (cell.backend, cell.phase)
        if key in actual_t2:
            raise CompositionRefused(f"combined T2 repeats required cell {key}")
        actual_t2[key] = cell
    required_t2 = {(cell.backend, cell.phase): cell
                   for cell in request.required_t2_cells}
    if actual_t2 != required_t2:
        raise CompositionRefused(
            "combined T2 coverage does not exactly equal the predeclared matrix: "
            f"required={sorted(required_t2)}, actual={sorted(actual_t2)}")
    if set(candidate.get("evaluation_event_ids") or ()) != {event["event_id"] for event in events}:
        raise CompositionRefused("combined candidate does not cite exactly the returned evaluations")
    if not isinstance(receipt.realized_cost, Mapping) or not receipt.realized_cost:
        raise CompositionRefused("composition must report its realized rebuild/evaluation cost")
    banking = BankingVerdict.from_candidate(candidate, events)
    return (CandidateSnapshot(
        candidate, candidate_entry.event_id, campaign, tuple(events), evidence,
        banking), candidate_entry, tuple(event_entries))


def _latest_tier(events: Sequence[Mapping[str, Any]], tier: str) -> Mapping[str, Any]:
    matches = [event for event in events if event.get("tier") == tier]
    if not matches:
        raise CompositionRefused(f"combined candidate has no {tier} event")
    return sorted(matches, key=lambda event: (event.get("created_at", ""), event["event_id"]))[-1]


def _readiness(events: Sequence[Mapping[str, Any]],
               required_cells: Sequence[T2Cell]) -> dict:
    by_backend: dict[str, dict[str, Any]] = {}
    signals: list[str] = []
    events_by_cell = {(event.get("backend"), event.get("phase")): event
                      for event in events if event.get("tier") == "T2"}
    for cell in required_cells:
        event = events_by_cell[(cell.backend, cell.phase)]
        performance = event.get("performance") \
            if isinstance(event.get("performance"), Mapping) else {}
        row = {
            "metric": cell.metric, "metric_direction": cell.metric_direction,
            "estimate": performance.get("estimate"),
            "uncertainty": performance.get("uncertainty"),
            "event_id": event.get("event_id"),
            "scope_manifest_sha256": cell.scope_manifest_sha256,
            "protocol_id": cell.protocol_id,
        }
        by_backend.setdefault(cell.backend, {})[cell.phase] = row
        signals.append(
            f"{cell.backend}/{cell.phase} {cell.metric}={row['estimate']!r} "
            f"({cell.metric_direction}) uncertainty="
            f"{schemas.canonical_json(row['uncertainty'])}")
    return {"by_backend": by_backend,
            "reference_signal": "; ".join(signals) + "; versus sealed anchor"}


def _surface_union(candidate: CandidateSnapshot) -> str:
    return schemas.content_hash({
        "derived": list(candidate.evidence.derived_surface_tokens),
        "traced": list(candidate.evidence.traced_surface_tokens),
    })


def _champion_record(request: CompositionRequest, candidate: CandidateSnapshot,
                     candidate_entry: journal.JournalEntry,
                     evaluation_entries: Sequence[journal.JournalEntry],
                     realized_cost: Mapping[str, Any], *, status: str = "active") -> dict:
    by_tier: dict[str, list[journal.JournalEntry]] = {}
    for entry in evaluation_entries:
        by_tier.setdefault(entry.payload["tier"], []).append(entry)
    last_t0 = by_tier["T0"][-1]
    last_t1 = by_tier["T1"][-1]
    last_t2 = by_tier["T2"][-1]
    record = {
        "schema": schemas.SCHEMA_CHAMPION,
        "source_tree": request.source_tree,
        "anchor_commit": request.anchor.commit,
        "branch": champion_branch(request.source_tree, request.anchor.commit),
        "member_candidates": list(request.member_candidates),
        "combined_candidate_id": request.combined_candidate_id,
        "last_t0": {"event_id": last_t0.payload["event_id"], "status": "pass"},
        "last_t1": {"event_id": last_t1.payload["event_id"], "status": "pass"},
        "last_t2": {"event_id": last_t2.payload["event_id"], "status": "pass"},
        "t2_coverage": [
            {"cell": cell.to_dict(),
             "event_id": next(
                 entry.payload["event_id"] for entry in by_tier["T2"]
                 if entry.payload.get("backend") == cell.backend
                 and entry.payload.get("phase") == cell.phase)}
            for cell in request.required_t2_cells
        ],
        "readiness": _readiness(candidate.evaluations, request.required_t2_cells),
        "affected_surface_union_sha256": _surface_union(candidate),
        "storage_gb": candidate.record["storage"]["footprint_gb"],
        "blocking_conditions": [],
        "status": status,
        "anchor": request.anchor.to_dict(),
        "evaluator": request.evaluator.to_dict(),
        # Carried forward from the request, so an advance never silently
        # re-defaults the build the champion is measured on.
        "build_recipe": request.build_recipe.to_dict(),
        "lineage": {
            "membership_order": list(request.member_candidates),
            "parent_champion_event_id": request.parent_champion_event_id,
            "request_sha256": request.request_sha256,
            "mode": request.mode,
            "realized_cost": dict(realized_cost),
        },
        "combined_evidence": {
            "candidate_record_event_id": candidate_entry.event_id,
            "evaluation_journal_event_ids": [entry.event_id for entry in evaluation_entries],
            "evaluation_event_ids": [entry.payload["event_id"] for entry in evaluation_entries],
        },
    }
    violations = schemas.validate_champion(record)
    if violations:
        raise CompositionRefused("constructed champion record is invalid: " + "; ".join(violations))
    return record


def promote_composition(book: journal.Journal, request: CompositionRequest,
                        runner: CompositionRunner, *, snapshot: Optional[JournalSnapshot] = None,
                        reanchor: bool = False) -> journal.JournalEntry:
    """Run/recover composition, append evidence, then atomically-visible champion.

    Each append is idempotent.  A crash after the candidate or any evaluation is
    resumed by replaying the deterministic request; a crash after all evidence but
    before ``CHAMPION_UPDATED`` appends only the missing champion record.
    """
    snapshot = snapshot or read_validated_snapshot(book)
    method = getattr(runner, "run_reanchor" if reanchor else "run_composition", None)
    if not callable(method):
        raise TypeError("runner does not implement the requested composition method")
    request_event = append_idempotent(
        book, journal.KIND_COMPOSITION_REQUESTED, request.to_dict())
    try:
        receipt = method(book, request, request_event)
    except Exception as exc:
        _record_composition_failure(book, request, request_event, exc)
        raise
    try:
        post_run = read_validated_snapshot(book)
        candidate, candidate_entry, evaluation_entries = _resolve_composition_receipt(
            post_run, receipt, request, request_event)
        evaluation_entries = list(evaluation_entries)
        evaluation_entries.sort(
            key=lambda entry: ({"T0": 0, "T1": 1, "T2": 2}.get(
                entry.payload.get("tier"), 9), entry.payload["event_id"]))
        record = _champion_record(
            request, candidate, candidate_entry, evaluation_entries,
            receipt.realized_cost,
            status="reanchored" if reanchor else "active")
        return append_idempotent(book, journal.KIND_CHAMPION_UPDATED, record)
    except Exception as exc:
        _record_composition_failure(book, request, request_event, exc)
        raise


def _record_composition_failure(book: journal.Journal,
                                request: CompositionRequest,
                                request_event: journal.JournalEntry,
                                exc: Exception) -> journal.JournalEntry:
    return append_idempotent(book, journal.KIND_COMPOSITION_FAILED, {
        "request_sha256": request.request_sha256,
        "request_event_id": request_event.event_id,
        "source_tree": request.source_tree,
        "failure_class": type(exc).__name__,
        "failure_detail": str(exc) or type(exc).__name__,
    })


def _empty_champion(anchor: AnchorIdentity, *, status: str, blocking: Sequence[str],
                    detail: Mapping[str, Any],
                    build_recipe: Optional[recipes.BuildRecipe] = None) -> dict:
    recipe = recipes.HOUSE_GPU_RECIPE if build_recipe is None else build_recipe
    record = {
        "schema": schemas.SCHEMA_CHAMPION, "source_tree": anchor.source_tree,
        "anchor_commit": anchor.commit, "branch": champion_branch(anchor.source_tree, anchor.commit),
        "member_candidates": [], "combined_candidate_id": None,
        "last_t0": None, "last_t1": None, "last_t2": None,
        "readiness": {"by_backend": {}, "reference_signal": "no green composed champion"},
        "affected_surface_union_sha256": schemas.content_hash([]), "storage_gb": 0.0,
        "blocking_conditions": list(blocking), "status": status,
        "anchor": anchor.to_dict(), "attempt": dict(detail),
        "build_recipe": recipe.to_dict(),
    }
    violations = schemas.validate_champion(record)
    if violations:
        raise CompositionRefused("constructed empty champion record is invalid: " + "; ".join(violations))
    return record


def seed_champion(book: journal.Journal, anchor: AnchorIdentity, *,
                  reason: str,
                  build_recipe: Optional[recipes.BuildRecipe] = None
                  ) -> journal.JournalEntry:
    """Champion₀: the aggregate EXISTS and currently equals the production anchor.

    This is a different state from :func:`record_no_champion`, and the difference is
    the whole point of the operator's standing requirement that *there is always an
    aggregate candidate ready for promotion gate testing*:

    * ``no_champion``          -- we have nothing, and something is BLOCKING us
      (``NO_GREEN_COMPOSITION``).
    * ``seeded_from_anchor``   -- we have an aggregate; it just has no members yet,
      so it is byte-identical to production. Nothing is blocking it.

    Hence ``blocking_conditions`` is empty here. That is not a weakened gate: a seed
    carries no ``last_t0/t1/t2`` events, so the schema's always-green rule (which
    fires on a tier event whose status is not ``pass``) has nothing to test. The
    champion becomes claim-bearing only once a composition earns its own T0/T1/T2
    against this same anchor, which :func:`promote_composition` still enforces.

    Champion₀ is the anchor PLUS the recipe it is built with (CH-1). "Equals
    production" is a claim about a binary, and a source tree alone does not
    determine one, so the seed states its defines instead of leaving them implied.

    Purity: this module may not build, benchmark or launch anything, so the anchor --
    including its measured per-backend binary and linkage digests -- must be derived
    by the caller (see ``champion_seed.production_anchor``) and handed in already
    sealed.
    """
    return append_idempotent(
        book, journal.KIND_CHAMPION_UPDATED,
        _empty_champion(anchor, status="seeded_from_anchor", blocking=[],
                        detail={"reason": _text(reason, "reason")},
                        build_recipe=build_recipe))


def record_no_champion(book: journal.Journal, anchor: AnchorIdentity, *, reason: str) -> journal.JournalEntry:
    return append_idempotent(book, journal.KIND_CHAMPION_UPDATED,
                             _empty_champion(anchor, status="no_champion", blocking=["NO_GREEN_COMPOSITION"], detail={"reason": _text(reason, "reason")}))


def record_rejected_composition(book: journal.Journal, anchor: AnchorIdentity,
                                report: CompatibilityReport
                                ) -> journal.JournalEntry:
    """Journal an incompatible attempt without replacing the active champion."""
    if report.compatible or not report.conflicts:
        raise CompositionRefused("only an incompatible report may be rejected")
    anchor_sha = schemas.content_hash(anchor.to_dict())
    spine = {
        "source_tree": anchor.source_tree,
        "anchor_sha256": anchor_sha,
        "candidate_ids": list(report.candidate_ids),
        "conflicts": list(report.conflicts),
        "compatibility_sha256": report.evidence_sha256,
    }
    payload = {"attempt_sha256": schemas.content_hash(spine), **spine}
    return append_idempotent(book, journal.KIND_COMPOSITION_REJECTED, payload)


def record_anchor_moved(book: journal.Journal, champion: Mapping[str, Any], *,
                        old_anchor: AnchorIdentity, new_anchor: AnchorIdentity) -> journal.JournalEntry:
    if old_anchor.source_tree != new_anchor.source_tree:
        raise AnchorMoved("an anchor move cannot change source trees")
    try:
        recorded = AnchorIdentity.from_dict(champion.get("anchor"))
    except EvidenceRefused as exc:
        raise AnchorMoved("prior champion has no exact anchor identity") from exc
    if not recorded.same_denominator(old_anchor):
        raise AnchorMoved("supplied old anchor contradicts the prior champion")
    if old_anchor.same_denominator(new_anchor):
        raise AnchorMoved("anchor-moved record requires a changed denominator")
    record = dict(champion)
    record.update({
        "anchor_commit": new_anchor.commit,
        "branch": champion_branch(new_anchor.source_tree, new_anchor.commit),
        "last_t0": None, "last_t1": None, "last_t2": None,
        "readiness": {"by_backend": {}, "reference_signal": "invalidated: production anchor moved; rebuild and re-evaluation required"},
        "blocking_conditions": ["ANCHOR_MOVED", "REANCHOR_PENDING_REMEASURE"],
        "status": "anchor_moved", "anchor": new_anchor.to_dict(),
        "anchor_move": {"old": old_anchor.to_dict(), "new": new_anchor.to_dict(),
                        "superseded_comparison_event_ids": list((champion.get("combined_evidence") or {}).get("evaluation_event_ids") or ())},
    })
    violations = schemas.validate_champion(record)
    if violations:
        raise AnchorMoved("anchor-moved champion record is invalid: " + "; ".join(violations))
    return append_idempotent(book, journal.KIND_CHAMPION_UPDATED, record)


def reanchor_champion(book: journal.Journal, *, prior_champion: Mapping[str, Any],
                      old_anchor: AnchorIdentity, new_anchor: AnchorIdentity,
                      evaluator: EvaluatorIdentity, runner: ReanchorRunner) -> journal.JournalEntry:
    # Read the recipe BEFORE anything is journaled: a champion that cannot state
    # its build cannot be advanced, and failing after the anchor-moved append would
    # leave a champion the loop can neither use nor rebuild.
    recipe = champion_build_recipe(prior_champion)
    moved = record_anchor_moved(book, prior_champion, old_anchor=old_anchor, new_anchor=new_anchor)
    snapshot = read_validated_snapshot(book)
    release_event, absorbed = _matching_release_receipt(snapshot, new_anchor)
    prior_members = tuple(prior_champion.get("member_candidates") or ())
    unknown_absorbed = set(absorbed) - set(prior_members)
    if unknown_absorbed:
        raise AnchorMoved(
            "sealed release receipt names members outside the prior champion: "
            + ", ".join(sorted(unknown_absorbed)))
    remaining_ids = tuple(member for member in prior_members if member not in absorbed)
    if not remaining_ids:
        return append_idempotent(
            book, journal.KIND_CHAMPION_UPDATED,
            _empty_champion(
                new_anchor, status="production_absorbed", blocking=[],
                detail={
                    "reason": "all prior champion members are present in production",
                    "absorbed_member_candidates": list(absorbed),
                    "release_package_event_id": release_event.event_id,
                },
                build_recipe=recipe))
    candidates = [_candidate_snapshot(snapshot, candidate_id)
                  for candidate_id in remaining_ids]
    # Preserve source/patch evidence, but invalidate every old rate comparison.
    # Compatibility is checked on the old source denominator solely to validate
    # that the remaining patches can coexist.  No old performance event is reused.
    ordered = _lineage_order(candidates)
    report = compatibility(ordered, anchor=old_anchor, evaluator=evaluator)
    if not report.compatible:
        raise CompatibilityRefused("reanchor members conflict: " + "; ".join(report.conflicts))
    required_t2 = _required_t2_cells(ordered, evaluator)
    spine = {
        "source_tree": new_anchor.source_tree, "members": [c.candidate_id for c in ordered],
        "member_record_event_ids": [c.record_event_id for c in ordered],
        "parent_champion_event_id": moved.event_id, "anchor": new_anchor.to_dict(),
        "evaluator": evaluator.to_dict(),
        "required_t2_cells": [cell.to_dict() for cell in required_t2],
        "compatibility_sha256": report.evidence_sha256,
        "absorbed_member_candidates": list(absorbed),
        "release_package_event_id": release_event.event_id,
        "mode": "reanchor",
        "build_recipe_sha256": recipe.sha256(),
    }
    digest = schemas.content_hash(spine)
    request = CompositionRequest(
        request_sha256=digest, combined_candidate_id=f"akc-composed-{digest[:24]}",
        source_tree=new_anchor.source_tree, member_candidates=tuple(c.candidate_id for c in ordered),
        member_record_event_ids=tuple(c.record_event_id for c in ordered),
        parent_champion_event_id=moved.event_id, anchor=new_anchor, evaluator=evaluator,
        required_t2_cells=required_t2,
        compatibility_sha256=spine["compatibility_sha256"],
        absorbed_member_candidates=absorbed,
        release_package_event_id=release_event.event_id,
        mode="reanchor", build_recipe=recipe)
    return promote_composition(book, request, runner, snapshot=snapshot, reanchor=True)


def _matching_release_receipt(snapshot: JournalSnapshot,
                              new_anchor: AnchorIdentity
                              ) -> tuple[journal.JournalEntry, tuple[str, ...]]:
    """Resolve the sealed package that became this exact production denominator."""
    matches: list[tuple[journal.JournalEntry, tuple[str, ...]]] = []
    for entry in snapshot.entries:
        if entry.kind != journal.KIND_RELEASE_PACKAGE_PREPARED:
            continue
        package = entry.payload
        if package.get("source_tree") != new_anchor.source_tree:
            continue
        raw_anchor = package.get("production_anchor")
        try:
            exact_anchor = isinstance(raw_anchor, Mapping) \
                and AnchorIdentity.from_dict(raw_anchor).same_denominator(new_anchor)
        except EvidenceRefused:
            exact_anchor = False
        if not exact_anchor:
            continue
        sealed = package.get("sealed_candidate")
        raw_members = sealed.get("member_candidates") \
            if isinstance(sealed, Mapping) else None
        if not isinstance(raw_members, list) or not raw_members \
                or any(not isinstance(item, str) or not item for item in raw_members):
            raise AnchorMoved(
                "matching sealed package has no member_candidates receipt")
        members = tuple(raw_members)
        if len(set(members)) != len(members):
            raise AnchorMoved("matching sealed package repeats a member candidate")
        matches.append((entry, members))
    if not matches:
        raise AnchorMoved(
            "new production anchor has no matching sealed release-package receipt")
    # Multiple receipts are acceptable only when they attest the same membership;
    # choose the latest envelope for the lineage link.
    memberships = {members for _, members in matches}
    if len(memberships) != 1:
        raise AnchorMoved("matching sealed release packages disagree on membership")
    return max(matches, key=lambda item: item[0].seq)
