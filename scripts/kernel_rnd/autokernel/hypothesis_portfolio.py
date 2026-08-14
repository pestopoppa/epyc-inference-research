#!/usr/bin/env python3
"""Strict, read-only loader for the discovery hypothesis portfolio.

The portfolio is strategy input, not measurement or promotion authority.  Its job is
to keep ranked questions, incumbents, negative lessons, and the evidence bytes behind
them in one fail-closed document.  It deliberately does not import the discovery
controller: deployment may later attest the portfolio as an input without making this
module a second execution path.

Intake is append-or-revise through the canonical JSON record contract: bind immutable
evidence paths and hashes, preserve stable record IDs, increment record_version,
declare exact regimes and numeric decision budgets, then run validate with
--verify-evidence.  A validated projection remains strategy input only.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


SCHEMA = "epyc.autokernel.discovery_hypothesis_portfolio.v2"
STATUSES = frozenset({"queued", "candidate_incumbent", "retired", "needs-template"})
MATURITIES = frozenset({
    "design_prior", "characterized", "candidate_authored", "correctness_validated",
    "screened", "candidate_incumbent", "retired", "dirty_diagnostic",
})
FRAME_KINDS = frozenset({"current_bundle", "large_model"})
TEMPORAL_STATUSES = frozenset({"current_v9", "current_v9_dirty_diagnostic"})
PORTABILITY_LEVELS = frozenset({"exact_frame", "low", "medium", "high"})
INTERACTION_KINDS = frozenset({
    "causal_overlap", "composition_candidate", "mutually_exclusive",
    "shared_bottleneck", "subsumes",
})
PRIORITY_TIERS = frozenset({"P0", "P1", "P2", "P3"})
SOURCE_MANIFEST_CHANGE_CLASSES = frozenset({
    "dispatcher", "arithmetic", "layout", "fusion", "moe_scheduling",
    "recurrent", "scheduler_policy", "oracle_port", "core_header",
})
EPISTEMIC_GRADES = frozenset({
    "design_prior", "profile_routing", "dirty_diagnostic", "correctness_only",
    "replicated_candidate_screen", "retired_negative",
})
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
EVIDENCE_AUTHORITIES = frozenset({
    "governed_diagnostic", "candidate_only", "presentation_projection",
    "dirty_diagnostic", "governance_snapshot",
})
DNR_CLASSIFICATIONS = frozenset({
    "measured_negative", "nonreplication", "correctness_failure", "subadditive",
    "sign_conflict", "physics_constraint", "prior_art_already_present",
    "configuration_regression", "low_value",
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
TEMPLATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*-v[1-9][0-9]*$")
ROUTE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*-v[1-9][0-9]*\.anchor\.[0-9]+$")

_TOP_KEYS = frozenset({
    "schema", "corpus_id", "generated_at", "promotion_authority",
    "current_bundle", "evidence", "frames", "hypotheses", "do_not_repeat",
})
_EVIDENCE_KEYS = frozenset({
    "evidence_id", "path", "sha256", "authority", "temporal_status", "claims",
})
_FRAME_KEYS = frozenset({
    "frame_id", "kind", "temporal_status", "production_commit", "device",
    "architecture", "model", "model_path", "model_sha256", "quant", "phase",
    "batch", "generated_tokens", "measurement_binary_sha256",
    "measurement_source_commit", "limitations",
    "measurement_graphs", "target_runtime_graphs", "flash_attention", "authority",
    "evidence_refs", "hotspots",
})
_HOTSPOT_KEYS = frozenset({
    "family", "device_time_share_pct", "calls", "note", "evidence_ref", "extraction",
})
_EXTRACTION_KEYS = frozenset({"method", "selector", "source_artifact_sha256"})
_HYPOTHESIS_KEYS = frozenset({
    "hypothesis_id", "title", "status", "statement", "primary_falsifier", "regime", "target",
    "dispatch_anchors", "mechanism",
    "falsifiers", "evidence_refs", "interactions", "portability", "priority",
    "expected_value", "implementation", "stop_rule", "current_bundle_eligibility",
    "lifecycle", "decision_policy", "epistemic", "record_version", "provenance",
})
_EPISTEMIC_KEYS = frozenset({"grade", "confidence", "limitations"})
_REGIME_KEYS = frozenset({
    "backend", "phase", "batch", "architecture", "model_or_frame", "quant",
    "shape", "measurement_graphs", "target_runtime_graphs",
})
_PROVENANCE_KEYS = frozenset({
    "introduced_at", "introduced_by", "origin", "note", "supersedes",
})
_LIFECYCLE_KEYS = frozenset({
    "maturity", "next_action", "candidate_identity", "diagnostic_identity",
})
_CANDIDATE_KEYS = frozenset({
    "candidate_id", "source_commit", "candidate_patch_sha256", "authority",
})
_DIAGNOSTIC_KEYS = frozenset({"binary_diff_sha256", "authority"})
_TARGET_KEYS = frozenset({
    "frame_ids", "source_files", "source_symbols", "template_intent",
})
_DISPATCH_KEYS = frozenset({
    "frame_id", "signatures", "excluded_signatures", "total_calls",
    "aggregation", "selection", "evidence_ref",
})
_SIGNATURE_KEYS = frozenset({
    "route_id", "kernel_literal", "calls", "grid", "workgroup", "lds_bytes",
})
_MECHANISM_KEYS = frozenset({"facets", "fingerprint_sha256"})
_INTERACTION_KEYS = frozenset({"with", "kind", "rationale"})
_PORTABILITY_KEYS = frozenset({
    "level", "source_frames", "target_frames", "constraints", "required_validation",
})
_PRIORITY_KEYS = frozenset({
    "rank", "tier", "device_time_share_pct_range", "rationale",
})
_EXPECTED_VALUE_KEYS = frozenset({
    "metric", "direction", "expected_relative_gain_pct_range",
    "device_time_ceiling_pct", "device_time_ceiling_frame_id",
    "current_bundle_plausible_gain_ceiling_pct", "basis",
})
_IMPLEMENTATION_KEYS = frozenset({"cost", "risk", "notes"})
_DECISION_POLICY_KEYS = frozenset({
    "metric", "frame_id", "effect_unit", "continuation_floor_pct",
    "nomination_floor_pct", "min_replication_effect_pct",
    "required_replications", "max_replication_spread_pct", "sign_policy", "conflict_policy",
    "max_distinct_candidates", "terminal_rule",
})
_ELIGIBILITY_KEYS = frozenset({
    "eligible", "template_ids", "blocking_conditions", "reason",
})
_DNR_KEYS = frozenset({
    "dnr_id", "title", "enforcement", "classification", "statement", "mechanism", "regime",
    "falsifier_result", "evidence_refs", "reentry_conditions", "record_version",
    "provenance",
})
_BUNDLE_KEYS = frozenset({
    "bundle_id", "frame_id", "template_catalog_version", "template_ids",
    "eligibility_semantics",
})


class PortfolioError(ValueError):
    """The portfolio cannot be trusted or interpreted exactly."""


@dataclass(frozen=True)
class Portfolio:
    """Validated immutable projection of the JSON document."""

    body: Mapping[str, Any]
    sha256: str

    @property
    def hypotheses(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.body["hypotheses"])

    @property
    def do_not_repeat(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.body["do_not_repeat"])

    def eligible_hypotheses(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            row for row in self.hypotheses
            if row["current_bundle_eligibility"]["eligible"]
        )

    def hypothesis(self, hypothesis_id: str) -> Mapping[str, Any]:
        matches = [row for row in self.hypotheses if row["hypothesis_id"] == hypothesis_id]
        if len(matches) != 1:
            raise PortfolioError(f"unknown hypothesis id: {hypothesis_id}")
        return matches[0]

    def eligible_record(self, hypothesis_id: str) -> Mapping[str, Any]:
        row = self.hypothesis(hypothesis_id)
        eligibility = row["current_bundle_eligibility"]
        if not eligibility["eligible"]:
            raise PortfolioError(f"hypothesis is not current-bundle eligible: {hypothesis_id}")
        frame_ids = set(row["target"]["frame_ids"])
        frames = tuple(frame for frame in self.body["frames"] if frame["frame_id"] in frame_ids)
        projection = {
            "record_version": row["record_version"], "provenance": row["provenance"],
            "hypothesis_id": row["hypothesis_id"], "statement": row["statement"],
            "primary_falsifier": row["primary_falsifier"], "falsifiers": row["falsifiers"],
            "regime": row["regime"], "mechanism": row["mechanism"],
            "target": row["target"], "dispatch_anchors": row["dispatch_anchors"],
            "frames": frames, "template_ids": eligibility["template_ids"],
            "stop_rule": row["stop_rule"], "evidence_refs": row["evidence_refs"],
            "maturity": row["lifecycle"]["maturity"],
            "next_action": row["lifecycle"]["next_action"],
            "candidate_identity": row["lifecycle"]["candidate_identity"],
            "diagnostic_identity": row["lifecycle"]["diagnostic_identity"],
            "decision_policy": row["decision_policy"],
            "epistemic": row["epistemic"],
        }
        return _freeze(projection)

    def eligible_projection(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.eligible_record(row["hypothesis_id"])
                     for row in self.eligible_hypotheses())

    def dnr_record(self, dnr_id: str) -> Mapping[str, Any]:
        matches = [row for row in self.do_not_repeat if row["dnr_id"] == dnr_id]
        if len(matches) != 1:
            raise PortfolioError(f"unknown DNR id: {dnr_id}")
        row = matches[0]
        return _freeze({
            "record_version": row["record_version"], "provenance": row["provenance"],
            "dnr_id": row["dnr_id"], "statement": row["statement"],
            "enforcement": row["enforcement"], "classification": row["classification"],
            "mechanism": row["mechanism"],
            "regime": row["regime"], "falsifier_result": row["falsifier_result"],
            "evidence_refs": row["evidence_refs"],
            "reentry_conditions": row["reentry_conditions"],
        })

    def dnr_projection(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.dnr_record(row["dnr_id"]) for row in self.do_not_repeat)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    value = _jsonable(value)
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PortfolioError(f"portfolio is not canonical JSON: {exc}") from exc


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PortfolioError(f"portfolio JSON contains duplicate key: {key}")
        value[key] = item
    return value


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def mechanism_fingerprint(facets: Mapping[str, Any]) -> str:
    if not isinstance(facets, Mapping) or not facets:
        raise PortfolioError("mechanism.facets must be a non-empty object")
    required = {"mechanism", "ops", "files", "symbols", "change_class"}
    if set(facets) != required:
        raise PortfolioError(
            f"mechanism.facets keys must be exactly {sorted(required)}"
        )
    _text(facets["mechanism"], "mechanism.facets.mechanism")
    _text(facets["change_class"], "mechanism.facets.change_class")
    for key in ("ops", "files", "symbols"):
        _text_list(facets[key], f"mechanism.facets.{key}")
    return content_sha256(dict(facets))


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise PortfolioError(f"{label} must be an object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise PortfolioError(f"{label} keys differ; missing={missing}, extra={extra}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PortfolioError(f"{label} must be a non-empty trimmed string")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _text(value, label)
    if not ID_RE.fullmatch(value):
        raise PortfolioError(f"{label} is not a canonical identifier")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PortfolioError(f"{label} must be a lowercase SHA-256")
    return value


def _rfc3339(value: Any, label: str) -> str:
    value = _text(value, label)
    if not RFC3339_RE.fullmatch(value):
        raise PortfolioError(f"{label} must be RFC3339 UTC with Z suffix")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise PortfolioError(f"{label} is not a real RFC3339 timestamp") from exc
    return value


def _validate_provenance(row: Mapping[str, Any], label: str) -> None:
    version = row["record_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PortfolioError(f"{label}.record_version must be positive")
    provenance = row["provenance"]
    _exact_keys(provenance, _PROVENANCE_KEYS, f"{label}.provenance")
    _rfc3339(provenance["introduced_at"], f"{label}.provenance.introduced_at")
    for key in ("introduced_by", "origin", "note"):
        _text(provenance[key], f"{label}.provenance.{key}")
    supersedes = provenance["supersedes"]
    if supersedes is not None:
        _identifier(supersedes, f"{label}.provenance.supersedes")


def _text_list(value: Any, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise PortfolioError(f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} array")
    rows = tuple(_text(item, f"{label}[]") for item in value)
    if len(rows) != len(set(rows)):
        raise PortfolioError(f"{label} contains duplicates")
    return rows


def _refs(value: Any, known: set[str], label: str) -> tuple[str, ...]:
    refs = _text_list(value, label)
    unknown = sorted(set(refs) - known)
    if unknown:
        raise PortfolioError(f"{label} has unknown evidence ids: {unknown}")
    return refs


def _validate_mechanism(value: Any, label: str) -> None:
    _exact_keys(value, _MECHANISM_KEYS, label)
    expected = mechanism_fingerprint(value["facets"])
    actual = _sha(value["fingerprint_sha256"], f"{label}.fingerprint_sha256")
    if actual != expected:
        raise PortfolioError(f"{label} fingerprint does not bind its facets")


def _validate_regime(value: Any, label: str) -> None:
    _exact_keys(value, _REGIME_KEYS, label)
    for key in ("backend", "phase", "architecture", "model_or_frame", "quant", "shape"):
        _text(value[key], f"{label}.{key}")
    if value["backend"] != "hip" or value["phase"] != "decode":
        raise PortfolioError(f"{label} must bind HIP decode")
    batch = value["batch"]
    if not isinstance(batch, (str, int)) or isinstance(batch, bool):
        raise PortfolioError(f"{label}.batch must be an exact integer or named range")
    if not isinstance(value["measurement_graphs"], bool) or not isinstance(
            value["target_runtime_graphs"], bool):
        raise PortfolioError(f"{label} graph fields must be boolean")


def _validate_epistemic(value: Any, label: str) -> None:
    _exact_keys(value, _EPISTEMIC_KEYS, label)
    if value["grade"] not in EPISTEMIC_GRADES:
        raise PortfolioError(f"{label}.grade is invalid")
    if value["confidence"] not in CONFIDENCE_LEVELS:
        raise PortfolioError(f"{label}.confidence is invalid")
    _text_list(value["limitations"], f"{label}.limitations")


def _validate_evidence(rows: Any) -> set[str]:
    if not isinstance(rows, list) or not rows:
        raise PortfolioError("evidence must be a non-empty array")
    ids: set[str] = set()
    for index, row in enumerate(rows):
        label = f"evidence[{index}]"
        _exact_keys(row, _EVIDENCE_KEYS, label)
        evidence_id = _identifier(row["evidence_id"], f"{label}.evidence_id")
        if evidence_id in ids:
            raise PortfolioError(f"duplicate evidence id: {evidence_id}")
        ids.add(evidence_id)
        path = _text(row["path"], f"{label}.path")
        if not Path(path).is_absolute() or ".." in Path(path).parts:
            raise PortfolioError(f"{label}.path must be absolute and traversal-free")
        _sha(row["sha256"], f"{label}.sha256")
        if row["authority"] not in EVIDENCE_AUTHORITIES:
            raise PortfolioError(f"{label}.authority is not recognized")
        if row["temporal_status"] not in TEMPORAL_STATUSES:
            raise PortfolioError(f"{label}.temporal_status is not current v9")
        _text_list(row["claims"], f"{label}.claims")
    return ids


def _validate_frames(rows: Any, evidence_ids: set[str]) -> set[str]:
    if not isinstance(rows, list) or not rows:
        raise PortfolioError("frames must be a non-empty array")
    ids: set[str] = set()
    current = 0
    large = 0
    for index, row in enumerate(rows):
        label = f"frames[{index}]"
        _exact_keys(row, _FRAME_KEYS, label)
        frame_id = _identifier(row["frame_id"], f"{label}.frame_id")
        if frame_id in ids:
            raise PortfolioError(f"duplicate frame id: {frame_id}")
        ids.add(frame_id)
        if row["kind"] not in FRAME_KINDS:
            raise PortfolioError(f"{label}.kind is not recognized")
        current += row["kind"] == "current_bundle"
        large += row["kind"] == "large_model"
        if row["temporal_status"] != "current_v9":
            raise PortfolioError(f"{label} cannot use stale or dirty frame authority")
        if not isinstance(row["production_commit"], str) or not GIT_SHA_RE.fullmatch(row["production_commit"]):
            raise PortfolioError(f"{label}.production_commit must be a full Git SHA")
        for key in ("device", "architecture", "model", "model_path", "quant", "phase", "authority"):
            _text(row[key], f"{label}.{key}")
        if not Path(row["model_path"]).is_absolute():
            raise PortfolioError(f"{label}.model_path must be absolute")
        _sha(row["model_sha256"], f"{label}.model_sha256")
        _sha(row["measurement_binary_sha256"], f"{label}.measurement_binary_sha256")
        measured_commit = row["measurement_source_commit"]
        if measured_commit is not None and (
                not isinstance(measured_commit, str)
                or not re.fullmatch(r"[0-9a-f]{9,40}", measured_commit)):
            raise PortfolioError(f"{label}.measurement_source_commit is invalid")
        _text_list(row["limitations"], f"{label}.limitations")
        if row["phase"] != "decode" or row["batch"] != 1:
            raise PortfolioError(f"{label} is outside the batch-one decode portfolio")
        if not isinstance(row["generated_tokens"], int) or row["generated_tokens"] <= 0:
            raise PortfolioError(f"{label}.generated_tokens must be positive")
        if (row["measurement_graphs"] is not False
                or row["target_runtime_graphs"] is not True
                or not isinstance(row["flash_attention"], bool)):
            raise PortfolioError(f"{label} graph/attention switches must be boolean")
        if "routing" not in row["authority"] or "whole-model" not in row["authority"]:
            raise PortfolioError(f"{label}.authority must limit graphs-off attribution semantics")
        _refs(row["evidence_refs"], evidence_ids, f"{label}.evidence_refs")
        if not isinstance(row["hotspots"], list) or not row["hotspots"]:
            raise PortfolioError(f"{label}.hotspots must be non-empty")
        families: set[str] = set()
        for h_index, hotspot in enumerate(row["hotspots"]):
            h_label = f"{label}.hotspots[{h_index}]"
            _exact_keys(hotspot, _HOTSPOT_KEYS, h_label)
            family = _identifier(hotspot["family"], f"{h_label}.family")
            if family in families:
                raise PortfolioError(f"{label} duplicates hotspot family {family}")
            families.add(family)
            share = hotspot["device_time_share_pct"]
            calls = hotspot["calls"]
            if not isinstance(share, (int, float)) or isinstance(share, bool) or not 0 < share <= 100:
                raise PortfolioError(f"{h_label}.device_time_share_pct is invalid")
            if not isinstance(calls, int) or calls <= 0:
                raise PortfolioError(f"{h_label}.calls must be positive")
            _text(hotspot["note"], f"{h_label}.note")
            if hotspot["evidence_ref"] not in evidence_ids:
                raise PortfolioError(f"{h_label}.evidence_ref is unknown")
            extraction = hotspot["extraction"]
            _exact_keys(extraction, _EXTRACTION_KEYS, f"{h_label}.extraction")
            if extraction["method"] not in {
                    "rocprof_family_classifier_v1", "rocprof_exact_kernel_reducer_v1"}:
                raise PortfolioError(f"{h_label}.extraction.method is invalid")
            _text(extraction["selector"], f"{h_label}.extraction.selector")
            _sha(extraction["source_artifact_sha256"],
                 f"{h_label}.extraction.source_artifact_sha256")
    if current != 1 or large < 1:
        raise PortfolioError("portfolio requires exactly one current-bundle frame and large-model frames")
    return ids


def _validate_bundle(bundle: Any, frame_ids: set[str]) -> None:
    _exact_keys(bundle, _BUNDLE_KEYS, "current_bundle")
    _identifier(bundle["bundle_id"], "current_bundle.bundle_id")
    if bundle["frame_id"] not in frame_ids:
        raise PortfolioError("current_bundle.frame_id is unknown")
    _text(bundle["template_catalog_version"], "current_bundle.template_catalog_version")
    template_ids = set(_text_list(bundle["template_ids"], "current_bundle.template_ids"))
    if any(not TEMPLATE_ID_RE.fullmatch(value) for value in template_ids):
        raise PortfolioError("current_bundle.template_ids contains a non-canonical id")
    _text(bundle["eligibility_semantics"], "current_bundle.eligibility_semantics")


def _validate_hypotheses(
    rows: Any, evidence_ids: set[str], frame_ids: set[str], current_frame: str,
    bundle_template_ids: set[str],
) -> set[str]:
    if not isinstance(rows, list) or not rows:
        raise PortfolioError("hypotheses must be a non-empty array")
    ids: set[str] = set()
    ranks: set[int] = set()
    pending_interactions: list[tuple[str, Mapping[str, Any]]] = []
    statuses: set[str] = set()
    for index, row in enumerate(rows):
        label = f"hypotheses[{index}]"
        _exact_keys(row, _HYPOTHESIS_KEYS, label)
        hypothesis_id = _identifier(row["hypothesis_id"], f"{label}.hypothesis_id")
        if hypothesis_id in ids:
            raise PortfolioError(f"duplicate hypothesis id: {hypothesis_id}")
        ids.add(hypothesis_id)
        _validate_provenance(row, label)
        _text(row["title"], f"{label}.title")
        status = row["status"]
        if status not in STATUSES:
            raise PortfolioError(f"{label}.status is not recognized")
        statuses.add(status)
        _text(row["statement"], f"{label}.statement")
        primary_falsifier = _text(row["primary_falsifier"], f"{label}.primary_falsifier")
        _validate_regime(row["regime"], f"{label}.regime")
        target = row["target"]
        _exact_keys(target, _TARGET_KEYS, f"{label}.target")
        target_frames = set(_text_list(target["frame_ids"], f"{label}.target.frame_ids"))
        if target_frames - frame_ids:
            raise PortfolioError(f"{label}.target references unknown frames")
        _text_list(target["source_files"], f"{label}.target.source_files")
        _text_list(target["source_symbols"], f"{label}.target.source_symbols")
        _text(target["template_intent"], f"{label}.target.template_intent")
        anchors = row["dispatch_anchors"]
        if not isinstance(anchors, list) or not anchors:
            raise PortfolioError(f"{label}.dispatch_anchors must be non-empty")
        anchor_frames: set[str] = set()
        for a_index, anchor in enumerate(anchors):
            a_label = f"{label}.dispatch_anchors[{a_index}]"
            _exact_keys(anchor, _DISPATCH_KEYS, a_label)
            frame_id = anchor["frame_id"]
            if frame_id not in target_frames or frame_id in anchor_frames:
                raise PortfolioError(f"{a_label}.frame_id is unknown, untargeted, or duplicated")
            anchor_frames.add(frame_id)
            signatures = anchor["signatures"]
            excluded = anchor["excluded_signatures"]
            if not isinstance(signatures, list) or not isinstance(excluded, list):
                raise PortfolioError(f"{a_label} signature carriers must be arrays")
            seen_signatures: set[tuple[Any, ...]] = set()
            signature_calls = 0
            for carrier_name, carrier in (
                    ("signatures", signatures), ("excluded_signatures", excluded)):
                for s_index, signature in enumerate(carrier):
                    s_label = f"{a_label}.{carrier_name}[{s_index}]"
                    _exact_keys(signature, _SIGNATURE_KEYS, s_label)
                    if not isinstance(signature["route_id"], str) or not ROUTE_ID_RE.fullmatch(
                            signature["route_id"]):
                        raise PortfolioError(f"{s_label}.route_id is invalid")
                    route_template = signature["route_id"].rsplit(".anchor.", 1)[0]
                    if route_template not in bundle_template_ids:
                        raise PortfolioError(f"{s_label}.route_id names an unknown template")
                    _text(signature["kernel_literal"], f"{s_label}.kernel_literal")
                    for key in ("calls", "grid", "workgroup", "lds_bytes"):
                        value = signature[key]
                        if (not isinstance(value, int) or isinstance(value, bool)
                                or value < (1 if key == "calls" else 0)):
                            raise PortfolioError(f"{s_label}.{key} is invalid")
                    identity = tuple(signature[key] for key in (
                        "route_id", "grid", "workgroup", "lds_bytes"))
                    if identity in seen_signatures:
                        raise PortfolioError(f"{a_label} duplicates an exact dispatch signature")
                    seen_signatures.add(identity)
                    if carrier_name == "signatures":
                        signature_calls += signature["calls"]
            if anchor["aggregation"] not in {
                    "exact_signatures", "family_aggregate", "not_applicable"}:
                raise PortfolioError(f"{a_label}.aggregation is invalid")
            total_calls = anchor["total_calls"]
            if anchor["aggregation"] == "exact_signatures":
                if not signatures or total_calls != signature_calls:
                    raise PortfolioError(f"{a_label} exact signatures do not sum to total_calls")
            elif anchor["aggregation"] == "family_aggregate":
                if signatures or excluded or not isinstance(total_calls, int) or total_calls <= 0:
                    raise PortfolioError(f"{a_label} family aggregate shape is invalid")
            elif signatures or excluded or total_calls is not None:
                raise PortfolioError(f"{a_label} not_applicable anchor must be empty")
            _text(anchor["selection"], f"{a_label}.selection")
            if anchor["evidence_ref"] not in evidence_ids:
                raise PortfolioError(f"{a_label}.evidence_ref is unknown")
        if anchor_frames != target_frames:
            raise PortfolioError(f"{label} needs one dispatch anchor per exact target frame")
        _validate_mechanism(row["mechanism"], f"{label}.mechanism")
        facets = row["mechanism"]["facets"]
        if (set(target["source_files"]) != set(facets["files"])
                or set(target["source_symbols"]) != set(facets["symbols"])):
            raise PortfolioError(f"{label} target surface differs from fingerprinted mechanism")
        falsifiers = _text_list(row["falsifiers"], f"{label}.falsifiers")
        if primary_falsifier not in falsifiers:
            raise PortfolioError(f"{label}.primary_falsifier must be one declared falsifier")
        _refs(row["evidence_refs"], evidence_ids, f"{label}.evidence_refs")
        if not {anchor["evidence_ref"] for anchor in anchors}.issubset(
                set(row["evidence_refs"])):
            raise PortfolioError(f"{label} dispatch evidence must be declared at record level")

        portability = row["portability"]
        _exact_keys(portability, _PORTABILITY_KEYS, f"{label}.portability")
        if portability["level"] not in PORTABILITY_LEVELS:
            raise PortfolioError(f"{label}.portability.level is not recognized")
        sources = set(_text_list(portability["source_frames"], f"{label}.portability.source_frames"))
        targets = set(_text_list(portability["target_frames"], f"{label}.portability.target_frames"))
        if (sources | targets) - frame_ids:
            raise PortfolioError(f"{label}.portability references unknown frames")
        _text_list(portability["constraints"], f"{label}.portability.constraints")
        _text_list(portability["required_validation"], f"{label}.portability.required_validation")

        priority = row["priority"]
        _exact_keys(priority, _PRIORITY_KEYS, f"{label}.priority")
        rank = priority["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0 or rank in ranks:
            raise PortfolioError(f"{label}.priority.rank must be unique and positive")
        ranks.add(rank)
        if priority["tier"] not in PRIORITY_TIERS:
            raise PortfolioError(f"{label}.priority.tier is not recognized")
        share_range = priority["device_time_share_pct_range"]
        if (not isinstance(share_range, list) or len(share_range) != 2
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in share_range)
                or not 0 <= share_range[0] <= share_range[1] <= 100):
            raise PortfolioError(f"{label}.priority.device_time_share_pct_range is invalid")
        _text(priority["rationale"], f"{label}.priority.rationale")

        expected = row["expected_value"]
        _exact_keys(expected, _EXPECTED_VALUE_KEYS, f"{label}.expected_value")
        _text(expected["metric"], f"{label}.expected_value.metric")
        if expected["direction"] != "higher_better":
            raise PortfolioError(f"{label}.expected_value.direction must be higher_better")
        gain = expected["expected_relative_gain_pct_range"]
        if (not isinstance(gain, list) or len(gain) != 2
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in gain)
                or gain[0] > gain[1]):
            raise PortfolioError(f"{label}.expected_value gain range is invalid")
        ceiling = expected["device_time_ceiling_pct"]
        if not isinstance(ceiling, (int, float)) or isinstance(ceiling, bool) or not 0 <= ceiling <= 100:
            raise PortfolioError(f"{label}.expected_value.device_time_ceiling_pct is invalid")
        if expected["device_time_ceiling_frame_id"] not in target_frames:
            raise PortfolioError(f"{label}.expected_value ceiling frame is not targeted")
        current_ceiling = expected["current_bundle_plausible_gain_ceiling_pct"]
        if current_ceiling is not None and (
                not isinstance(current_ceiling, (int, float)) or isinstance(current_ceiling, bool)
                or not 0 <= current_ceiling <= 100):
            raise PortfolioError(f"{label}.expected_value current-bundle ceiling is invalid")
        _text(expected["basis"], f"{label}.expected_value.basis")
        implementation = row["implementation"]
        _exact_keys(implementation, _IMPLEMENTATION_KEYS, f"{label}.implementation")
        if implementation["cost"] not in {"low", "medium", "high"}:
            raise PortfolioError(f"{label}.implementation.cost is invalid")
        if implementation["risk"] not in {"low", "medium", "high"}:
            raise PortfolioError(f"{label}.implementation.risk is invalid")
        _text_list(implementation["notes"], f"{label}.implementation.notes")
        _text_list(row["stop_rule"], f"{label}.stop_rule")
        policy = row["decision_policy"]
        _exact_keys(policy, _DECISION_POLICY_KEYS, f"{label}.decision_policy")
        _text(policy["metric"], f"{label}.decision_policy.metric")
        if policy["effect_unit"] != "relative_percent":
            raise PortfolioError(f"{label}.decision_policy.effect_unit is invalid")
        if policy["frame_id"] not in target_frames:
            raise PortfolioError(f"{label}.decision_policy.frame_id is not a target frame")
        continuation = policy["continuation_floor_pct"]
        nomination = policy["nomination_floor_pct"]
        if (not isinstance(continuation, (int, float)) or isinstance(continuation, bool)
                or not isinstance(nomination, (int, float)) or isinstance(nomination, bool)
                or continuation < 0 or nomination < continuation):
            raise PortfolioError(f"{label}.decision_policy floors are invalid")
        per_replication = policy["min_replication_effect_pct"]
        spread = policy["max_replication_spread_pct"]
        if (not isinstance(per_replication, (int, float)) or isinstance(per_replication, bool)
                or per_replication < 0 or per_replication > nomination
                or not isinstance(spread, (int, float)) or isinstance(spread, bool)
                or spread < 0):
            raise PortfolioError(f"{label}.decision_policy replication bounds are invalid")
        if (not isinstance(policy["required_replications"], int)
                or isinstance(policy["required_replications"], bool)
                or policy["required_replications"] < 1):
            raise PortfolioError(f"{label}.decision_policy required_replications is invalid")
        if policy["sign_policy"] not in {"all_positive", "median_positive"}:
            raise PortfolioError(f"{label}.decision_policy.sign_policy is invalid")
        if policy["conflict_policy"] not in {"retire", "retain_inconclusive"}:
            raise PortfolioError(f"{label}.decision_policy.conflict_policy is invalid")
        if (not isinstance(policy["max_distinct_candidates"], int)
                or isinstance(policy["max_distinct_candidates"], bool)
                or not 1 <= policy["max_distinct_candidates"] <= 8):
            raise PortfolioError(f"{label}.decision_policy.max_distinct_candidates is invalid")
        if policy["terminal_rule"] not in {"retire", "retain_inconclusive", "needs_review"}:
            raise PortfolioError(f"{label}.decision_policy.terminal_rule is invalid")
        _validate_epistemic(row["epistemic"], f"{label}.epistemic")
        lifecycle = row["lifecycle"]
        _exact_keys(lifecycle, _LIFECYCLE_KEYS, f"{label}.lifecycle")
        maturity = lifecycle["maturity"]
        if maturity not in MATURITIES:
            raise PortfolioError(f"{label}.lifecycle.maturity is invalid")
        _text(lifecycle["next_action"], f"{label}.lifecycle.next_action")
        candidate = lifecycle["candidate_identity"]
        diagnostic = lifecycle["diagnostic_identity"]
        if candidate is not None:
            _exact_keys(candidate, _CANDIDATE_KEYS, f"{label}.lifecycle.candidate_identity")
            _identifier(candidate["candidate_id"], f"{label} candidate id")
            if not isinstance(candidate["source_commit"], str) or not GIT_SHA_RE.fullmatch(candidate["source_commit"]):
                raise PortfolioError(f"{label} candidate source_commit must be a full Git SHA")
            if candidate["candidate_patch_sha256"] is not None:
                _sha(candidate["candidate_patch_sha256"], f"{label} candidate patch")
            if candidate["authority"] not in {"source_commit_only", "sealed_patch"}:
                raise PortfolioError(f"{label} candidate authority is invalid")
            if candidate["authority"] == "sealed_patch" and candidate["candidate_patch_sha256"] is None:
                raise PortfolioError(f"{label} sealed patch has no patch SHA-256")
        if maturity == "correctness_validated" and candidate is None:
            raise PortfolioError(f"{label} correctness-validated candidate lacks identity")
        if diagnostic is not None:
            _exact_keys(diagnostic, _DIAGNOSTIC_KEYS, f"{label}.lifecycle.diagnostic_identity")
            _sha(diagnostic["binary_diff_sha256"], f"{label} diagnostic binary diff")
            if diagnostic["authority"] != "dirty_mechanism_only_no_candidate_authority":
                raise PortfolioError(f"{label} diagnostic authority is invalid")
        if maturity == "dirty_diagnostic" and (candidate is not None or diagnostic is None):
            raise PortfolioError(
                f"{label} dirty diagnostic needs mechanism identity and no candidate authority"
            )
        if maturity != "dirty_diagnostic" and diagnostic is not None:
            raise PortfolioError(f"{label} non-diagnostic record carries dirty identity")
        if status == "candidate_incumbent" and maturity != "candidate_incumbent":
            raise PortfolioError(f"{label} incumbent status/maturity disagree")
        if status == "retired" and maturity != "retired":
            raise PortfolioError(f"{label} retired status/maturity disagree")

        eligibility = row["current_bundle_eligibility"]
        _exact_keys(eligibility, _ELIGIBILITY_KEYS, f"{label}.current_bundle_eligibility")
        if not isinstance(eligibility["eligible"], bool):
            raise PortfolioError(f"{label}.current_bundle_eligibility.eligible must be boolean")
        templates = set(_text_list(
            eligibility["template_ids"], f"{label}.current_bundle_eligibility.template_ids",
            allow_empty=True,
        ))
        blockers = _text_list(
            eligibility["blocking_conditions"],
            f"{label}.current_bundle_eligibility.blocking_conditions", allow_empty=True,
        )
        _text(eligibility["reason"], f"{label}.current_bundle_eligibility.reason")
        if eligibility["eligible"]:
            if status != "queued" or not templates or blockers:
                raise PortfolioError(f"{label} eligible entry must be queued, templated, and unblocked")
            if templates - bundle_template_ids or current_frame not in sources:
                raise PortfolioError(f"{label} eligibility is not bound to current bundle")
            if row["mechanism"]["facets"]["change_class"] not in SOURCE_MANIFEST_CHANGE_CLASSES:
                raise PortfolioError(f"{label} eligible change_class is not source-manifest authorable")
        elif status == "needs-template" and not blockers:
            raise PortfolioError(f"{label} needs-template entry must name a blocker")

        interactions = row["interactions"]
        if not isinstance(interactions, list):
            raise PortfolioError(f"{label}.interactions must be an array")
        for i_index, interaction in enumerate(interactions):
            i_label = f"{label}.interactions[{i_index}]"
            _exact_keys(interaction, _INTERACTION_KEYS, i_label)
            _identifier(interaction["with"], f"{i_label}.with")
            if interaction["kind"] not in INTERACTION_KINDS:
                raise PortfolioError(f"{i_label}.kind is not recognized")
            _text(interaction["rationale"], f"{i_label}.rationale")
            pending_interactions.append((hypothesis_id, interaction))
    if statuses != set(STATUSES):
        raise PortfolioError(
            "corpus must exercise queued/candidate_incumbent/retired/needs-template states"
        )
    for source, interaction in pending_interactions:
        if interaction["with"] == source or interaction["with"] not in ids:
            raise PortfolioError(f"{source} has invalid interaction target {interaction['with']}")
    _validate_supersession_graph(
        {row["hypothesis_id"]: row["provenance"]["supersedes"] for row in rows},
        "hypothesis")
    return ids


def _validate_dnr(rows: Any, evidence_ids: set[str]) -> set[str]:
    if not isinstance(rows, list) or not rows:
        raise PortfolioError("do_not_repeat must be a non-empty array")
    ids: set[str] = set()
    fingerprints: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        label = f"do_not_repeat[{index}]"
        _exact_keys(row, _DNR_KEYS, label)
        dnr_id = _identifier(row["dnr_id"], f"{label}.dnr_id")
        if dnr_id in ids:
            raise PortfolioError(f"duplicate DNR id: {dnr_id}")
        ids.add(dnr_id)
        _validate_provenance(row, label)
        _text(row["title"], f"{label}.title")
        if row["enforcement"] != "hard_refusal_exact_mechanism_and_regime":
            raise PortfolioError(f"{label} is not a hard scoped refusal")
        if row["classification"] not in DNR_CLASSIFICATIONS:
            raise PortfolioError(f"{label}.classification is invalid")
        _text(row["statement"], f"{label}.statement")
        _validate_mechanism(row["mechanism"], f"{label}.mechanism")
        _validate_regime(row["regime"], f"{label}.regime")
        _text(row["falsifier_result"], f"{label}.falsifier_result")
        _refs(row["evidence_refs"], evidence_ids, f"{label}.evidence_refs")
        _text_list(row["reentry_conditions"], f"{label}.reentry_conditions")
        identity = (
            row["mechanism"]["fingerprint_sha256"], content_sha256(dict(row["regime"])),
        )
        if identity in fingerprints:
            raise PortfolioError(f"{label} duplicates a mechanism/regime refusal")
        fingerprints.add(identity)
    _validate_supersession_graph(
        {row["dnr_id"]: row["provenance"]["supersedes"] for row in rows}, "DNR")
    return ids


def _validate_supersession_graph(graph: Mapping[str, str | None], label: str) -> None:
    for source, target in graph.items():
        if target is not None and target not in graph:
            raise PortfolioError(f"{label} {source} supersedes unknown id {target}")
        seen = {source}
        cursor = target
        while cursor is not None:
            if cursor in seen:
                raise PortfolioError(f"{label} supersession cycle")
            seen.add(cursor)
            cursor = graph[cursor]


def validate(body: Any) -> Portfolio:
    _exact_keys(body, _TOP_KEYS, "portfolio")
    if body["schema"] != SCHEMA:
        raise PortfolioError(f"schema must be {SCHEMA}")
    _identifier(body["corpus_id"], "corpus_id")
    _rfc3339(body["generated_at"], "generated_at")
    if body["promotion_authority"] is not False:
        raise PortfolioError("portfolio must never claim promotion authority")
    evidence_ids = _validate_evidence(body["evidence"])
    frame_ids = _validate_frames(body["frames"], evidence_ids)
    _validate_bundle(body["current_bundle"], frame_ids)
    current_frame = body["current_bundle"]["frame_id"]
    if next(row for row in body["frames"] if row["frame_id"] == current_frame)["kind"] != "current_bundle":
        raise PortfolioError("current_bundle.frame_id does not name the current frame")
    hypothesis_ids = _validate_hypotheses(
        body["hypotheses"], evidence_ids, frame_ids, current_frame,
        set(body["current_bundle"]["template_ids"]),
    )
    dnr_ids = _validate_dnr(body["do_not_repeat"], evidence_ids)
    if hypothesis_ids & dnr_ids:
        raise PortfolioError("hypothesis and DNR ids overlap")
    dnr_identities = {
        (row["mechanism"]["fingerprint_sha256"], content_sha256(row["regime"]))
        for row in body["do_not_repeat"]
    }
    for row in body["hypotheses"]:
        if row["current_bundle_eligibility"]["eligible"] and (
                row["mechanism"]["fingerprint_sha256"], content_sha256(row["regime"])
        ) in dnr_identities:
            raise PortfolioError(
                f"eligible hypothesis {row['hypothesis_id']} contradicts a hard DNR identity"
            )
    frozen = json.loads(_canonical_bytes(body).decode("utf-8"))
    return Portfolio(_freeze(frozen), content_sha256(frozen))


def _read_pinned(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PortfolioError(f"{label}: cannot open authority: {exc}") from exc
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.geteuid()):
            raise PortfolioError(f"{label}: authority must be an owned single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        path_after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PortfolioError(f"{label}: authority path disappeared: {exc}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (identity_before != identity_after or len(raw) != before.st_size
            or (path_after.st_dev, path_after.st_ino) != (before.st_dev, before.st_ino)):
        raise PortfolioError(f"{label}: authority changed while it was read")
    return raw


def load(path: os.PathLike[str] | str) -> Portfolio:
    source = Path(path)
    raw = _read_pinned(source, "portfolio")
    try:
        body = json.loads(raw, object_pairs_hook=_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortfolioError(f"portfolio is not valid UTF-8 JSON: {exc}") from exc
    return validate(body)


def verify_evidence_files(portfolio: Portfolio, evidence_ids: Iterable[str] | None = None) -> None:
    """Verify selected evidence paths against the bytes bound in the corpus.

    This verifies carrier path identity and bytes, not claims derived from those bytes.
    Each extraction method needs a deterministic source-specific recomputer/test (or a
    separately sealed derived receipt). Loading the checked-in strategy corpus remains
    portable and side-effect free; deployment can call this before sealing a projection.
    """
    rows = {row["evidence_id"]: row for row in portfolio.body["evidence"]}
    selected = set(rows if evidence_ids is None else evidence_ids)
    unknown = sorted(selected - set(rows))
    if unknown:
        raise PortfolioError(f"unknown evidence ids requested: {unknown}")
    for evidence_id in sorted(selected):
        row = rows[evidence_id]
        path = Path(row["path"])
        raw = _read_pinned(path, evidence_id)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != row["sha256"]:
            raise PortfolioError(f"{evidence_id}: evidence SHA-256 mismatch")


def validate_template_authorability(
    portfolio: Portfolio,
    catalog_version: str,
    surfaces: Mapping[str, Mapping[str, Iterable[str]]],
) -> None:
    """Cross-check eligible records against a deployment-owned normalized registry.

    Each surface must expose exact ``source_files``, ``source_symbols``,
    ``change_classes``, ``dispatch_signatures`` and ``excluded_signatures``. The
    portfolio intentionally does not vendor deployment
    authority; the deployment compiler supplies and seals this projection.
    """
    expected_version = portfolio.body["current_bundle"]["template_catalog_version"]
    if catalog_version != expected_version:
        raise PortfolioError("template catalog version does not match portfolio")
    for row in portfolio.eligible_hypotheses():
        templates = row["current_bundle_eligibility"]["template_ids"]
        missing = sorted(set(templates) - set(surfaces))
        if missing:
            raise PortfolioError(f"{row['hypothesis_id']} has unknown deployment templates: {missing}")
        files: set[str] = set()
        symbols: set[str] = set()
        change_classes: set[str] = set()
        dispatch_signatures: set[tuple[str, int, int, int, int]] = set()
        excluded_signatures: set[tuple[str, int, int, int, int]] = set()
        for template_id in templates:
            surface = surfaces[template_id]
            if set(surface) != {
                    "source_files", "source_symbols", "change_classes",
                    "dispatch_signatures", "excluded_signatures"}:
                raise PortfolioError(f"template {template_id} has malformed normalized surface")
            files.update(_text_list(list(surface["source_files"]), f"{template_id}.source_files"))
            symbols.update(_text_list(list(surface["source_symbols"]), f"{template_id}.source_symbols"))
            change_classes.update(_text_list(
                list(surface["change_classes"]), f"{template_id}.change_classes"))
            for carrier_name, target in (
                    ("dispatch_signatures", dispatch_signatures),
                    ("excluded_signatures", excluded_signatures)):
                carrier = surface[carrier_name]
                if not isinstance(carrier, Iterable) or isinstance(carrier, (str, bytes)):
                    raise PortfolioError(f"template {template_id} {carrier_name} is malformed")
                for signature in carrier:
                    if not isinstance(signature, Mapping) or set(signature) != {
                            "route_id", "calls", "grid", "workgroup", "lds_bytes"}:
                        raise PortfolioError(
                            f"template {template_id} {carrier_name} has malformed signature")
                    route_id = signature["route_id"]
                    geometry = tuple(signature[key] for key in (
                        "calls", "grid", "workgroup", "lds_bytes"))
                    if (not isinstance(route_id, str) or not ROUTE_ID_RE.fullmatch(route_id)
                            or not route_id.startswith(f"{template_id}.anchor.")
                            or any(not isinstance(value, int) or isinstance(value, bool)
                                   or value < 0 for value in geometry)):
                        raise PortfolioError(
                            f"template {template_id} {carrier_name} has invalid geometry")
                    target.add((route_id, *geometry))
        if not set(row["target"]["source_files"]).issubset(files):
            raise PortfolioError(f"{row['hypothesis_id']} target files exceed template authority")
        if not set(row["target"]["source_symbols"]).issubset(symbols):
            raise PortfolioError(f"{row['hypothesis_id']} target symbols exceed template authority")
        if row["mechanism"]["facets"]["change_class"] not in change_classes:
            raise PortfolioError(f"{row['hypothesis_id']} change class exceeds template authority")
        current_anchor = next(
            anchor for anchor in row["dispatch_anchors"]
            if anchor["frame_id"] == portfolio.body["current_bundle"]["frame_id"])
        record_dispatch = {
            tuple(signature[key] for key in (
                "route_id", "calls", "grid", "workgroup", "lds_bytes"))
            for signature in current_anchor["signatures"]
        }
        record_excluded = {
            tuple(signature[key] for key in (
                "route_id", "calls", "grid", "workgroup", "lds_bytes"))
            for signature in current_anchor["excluded_signatures"]
        }
        if record_dispatch != dispatch_signatures or record_excluded != excluded_signatures:
            raise PortfolioError(
                f"{row['hypothesis_id']} dispatch geometry differs from template authority")


DEFAULT_PORTFOLIO = Path(__file__).with_name("discovery_hypothesis_portfolio_v2.json")


def main(argv: list[str] | None = None) -> int:
    """Validate a corpus for intake without changing repository or runtime state."""
    parser = argparse.ArgumentParser(description="Validate an AutoKernel hypothesis portfolio")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--verify-evidence", action="store_true")
    summarize_parser = commands.add_parser("summarize")
    summarize_parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        portfolio = load(args.path)
        verify_evidence = bool(getattr(args, "verify_evidence", False))
        if verify_evidence:
            verify_evidence_files(portfolio)
    except PortfolioError as exc:
        parser.exit(2, f"portfolio validation failed: {exc}\n")
    if args.command == "summarize":
        print(json.dumps({
            "schema": portfolio.body["schema"],
            "corpus_id": portfolio.body["corpus_id"],
            "sha256": portfolio.sha256,
            "promotion_authority": portfolio.body["promotion_authority"],
            "eligible_records": _jsonable(portfolio.eligible_projection()),
            "do_not_repeat": _jsonable(portfolio.dnr_projection()),
        }, sort_keys=True, indent=2))
        return 0
    print(json.dumps({
        "corpus_id": portfolio.body["corpus_id"],
        "sha256": portfolio.sha256,
        "hypotheses": len(portfolio.hypotheses),
        "eligible": len(portfolio.eligible_hypotheses()),
        "do_not_repeat": len(portfolio.do_not_repeat),
        "evidence_carriers_verified": verify_evidence,
    }, sort_keys=True))
    return 0


__all__ = [
    "SCHEMA", "STATUSES", "MATURITIES", "Portfolio", "PortfolioError",
    "DEFAULT_PORTFOLIO", "content_sha256", "mechanism_fingerprint", "validate",
    "load", "verify_evidence_files", "validate_template_authorability",
]


if __name__ == "__main__":
    raise SystemExit(main())
