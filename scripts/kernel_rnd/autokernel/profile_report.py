#!/usr/bin/env python3
"""Deterministic offline C4 report over paired hash-bound rocprofv2 traces.

The deterministic pass owns parsing, source-backed matching, tables, and coverage
gaps. A later model may only attach one bounded similarity label and a catalogue
comparison to a row emitted here; it never receives authority to rewrite facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import prior_art


SCHEMA = "epyc.autokernel.c4_profile_report.v1"
JUDGMENT_SCHEMA = "epyc.autokernel.c4_bounded_judgment.v1"
SIMILARITY = ("low", "medium", "high")
STAGES = ("prefill", "decode")
PATTERN_TABLES = ("overlap", "fuse")
MATCH_MODES = ("any", "all")
ATTRIBUTION_MODES = ("graphs_disabled", "lower_fusion")
FORMAL_MODE = "production_optimizations"
CATALOGUE_SCOPES = ("kernel_only", "kernel_and_host")
PROFILER_STATES = ("available", "unavailable", "fallback", "unchecked")
GFX90A_STATES = ("supported", "unsupported", "unchecked")
REQUIRED_PROFILERS = frozenset(("rocprofv2", "rocprof_v1", "omniperf", "rpd"))
DEFAULT_WARMUP_STEPS = 10
DEFAULT_ACTIVE_STEPS = 5
CUMULATIVE_FLOOR = 0.01

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ProfileReportError(ValueError):
    """The offline C4 manifest or its deterministic result is invalid."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileReportError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256_text(value: Any, label: str) -> str:
    value = _text(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise ProfileReportError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ProfileReportError(f"{label} must be a non-empty sequence")
    rendered = tuple(_text(item, f"{label} item") for item in value)
    if len(set(rendered)) != len(rendered):
        raise ProfileReportError(f"{label} must not contain duplicates")
    return rendered


@dataclass(frozen=True)
class CaptureSpec:
    role: str
    stage: str
    attribution_mode: str
    warmup_steps: int
    active_steps: int
    receipt: prior_art.ProfileReceipt

    def __post_init__(self) -> None:
        if self.role not in ("mapping", "formal"):
            raise ProfileReportError("capture role must be mapping or formal")
        if self.stage not in STAGES:
            raise ProfileReportError(f"capture stage must be one of {STAGES}")
        if self.role == "mapping" and self.attribution_mode not in ATTRIBUTION_MODES:
            raise ProfileReportError(
                f"mapping attribution_mode must be one of {ATTRIBUTION_MODES}")
        if self.role == "formal" and self.attribution_mode != FORMAL_MODE:
            raise ProfileReportError(
                f"formal attribution_mode must be {FORMAL_MODE!r}")
        if self.warmup_steps != DEFAULT_WARMUP_STEPS:
            raise ProfileReportError(
                f"capture warmup_steps must be {DEFAULT_WARMUP_STEPS}")
        if self.active_steps != DEFAULT_ACTIVE_STEPS:
            raise ProfileReportError(
                f"capture active_steps must be {DEFAULT_ACTIVE_STEPS}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaptureSpec":
        if not isinstance(payload, Mapping):
            raise ProfileReportError("capture must be an object")
        receipt = payload.get("receipt")
        if not isinstance(receipt, Mapping):
            raise ProfileReportError("capture.receipt must be an object")
        return cls(
            role=_text(payload.get("role"), "capture.role"),
            stage=_text(payload.get("stage"), "capture.stage"),
            attribution_mode=_text(
                payload.get("attribution_mode"), "capture.attribution_mode"),
            warmup_steps=payload.get("warmup_steps"),
            active_steps=payload.get("active_steps"),
            receipt=prior_art.ProfileReceipt(
                corpus_id=receipt.get("corpus_id"),
                workload_id=receipt.get("workload_id"),
                profile_path=receipt.get("profile_path"),
                profile_sha256=receipt.get("profile_sha256"),
                source_commit=receipt.get("source_commit"),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "stage": self.stage,
            "attribution_mode": self.attribution_mode,
            "warmup_steps": self.warmup_steps,
            "active_steps": self.active_steps,
            "receipt": {
                "corpus_id": self.receipt.corpus_id,
                "workload_id": self.receipt.workload_id,
                "profile_path": self.receipt.profile_path,
                "profile_sha256": self.receipt.profile_sha256,
                "source_commit": self.receipt.source_commit,
            },
        }


@dataclass(frozen=True)
class SourcePattern:
    pattern_id: str
    table: str
    kernel_keywords: tuple[str, ...]
    match_mode: str
    source_symbols: tuple[str, ...]
    source_paths: tuple[str, ...]
    reader_should_conclude: str

    def __post_init__(self) -> None:
        _text(self.pattern_id, "pattern_id")
        if self.table not in PATTERN_TABLES:
            raise ProfileReportError(f"pattern table must be one of {PATTERN_TABLES}")
        _string_tuple(self.kernel_keywords, "kernel_keywords")
        if self.match_mode not in MATCH_MODES:
            raise ProfileReportError(f"match_mode must be one of {MATCH_MODES}")
        _string_tuple(self.source_symbols, "source_symbols")
        _string_tuple(self.source_paths, "source_paths")
        _text(self.reader_should_conclude, "reader_should_conclude")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourcePattern":
        return cls(
            pattern_id=_text(payload.get("pattern_id"), "pattern.pattern_id"),
            table=_text(payload.get("table"), "pattern.table"),
            kernel_keywords=_string_tuple(
                payload.get("kernel_keywords"), "pattern.kernel_keywords"),
            match_mode=_text(payload.get("match_mode", "all"), "pattern.match_mode"),
            source_symbols=_string_tuple(
                payload.get("source_symbols"), "pattern.source_symbols"),
            source_paths=_string_tuple(payload.get("source_paths"), "pattern.source_paths"),
            reader_should_conclude=_text(
                payload.get("reader_should_conclude"), "pattern.reader_should_conclude"),
        )

    def matches(self, dispatches: Sequence[prior_art.ProfileDispatch]) -> bool:
        haystack = "\n".join(
            f"{row.kernel_family}\n{row.kernel_name}" for row in dispatches).casefold()
        hits = tuple(keyword.casefold() in haystack for keyword in self.kernel_keywords)
        return all(hits) if self.match_mode == "all" else any(hits)

    def matched_families(self, dispatches: Sequence[prior_art.ProfileDispatch]) -> tuple[str, ...]:
        families = []
        for row in dispatches:
            text = f"{row.kernel_family}\n{row.kernel_name}".casefold()
            if any(keyword.casefold() in text for keyword in self.kernel_keywords):
                families.append(row.kernel_family)
        return tuple(dict.fromkeys(families))


@dataclass(frozen=True)
class ArchitectureBlock:
    block_id: str
    kernel_families: tuple[str, ...]
    kernel_family_aliases: tuple[tuple[str, ...], ...]
    source_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.block_id, "block_id")
        _string_tuple(self.kernel_families, "kernel_families")
        if len(self.kernel_family_aliases) != len(self.kernel_families):
            raise ProfileReportError(
                "architecture kernel_family_aliases must align with kernel_families")
        for index, (canonical, aliases) in enumerate(zip(
                self.kernel_families, self.kernel_family_aliases, strict=True)):
            _string_tuple(aliases, f"architecture kernel_family_aliases[{index}]")
            if canonical not in aliases:
                raise ProfileReportError(
                    "each canonical architecture kernel family must appear in its aliases")
        _string_tuple(self.source_paths, "architecture source_paths")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectureBlock":
        families = _string_tuple(
            payload.get("kernel_families"), "architecture.kernel_families")
        aliases = payload.get("kernel_family_aliases")
        if aliases is None:
            rendered_aliases = tuple((family,) for family in families)
        elif not isinstance(aliases, (list, tuple)):
            raise ProfileReportError(
                "architecture.kernel_family_aliases must be a sequence")
        else:
            rendered_aliases = tuple(
                _string_tuple(row, f"architecture.kernel_family_aliases[{index}]")
                for index, row in enumerate(aliases))
        return cls(
            block_id=_text(payload.get("block_id"), "architecture.block_id"),
            kernel_families=families,
            kernel_family_aliases=rendered_aliases,
            source_paths=_string_tuple(
                payload.get("source_paths"), "architecture.source_paths"),
        )


@dataclass(frozen=True)
class ProfilerCandidate:
    name: str
    state: str
    gfx90a_state: str
    evidence: str

    def __post_init__(self) -> None:
        _text(self.name, "profiler name")
        if self.state not in PROFILER_STATES:
            raise ProfileReportError(f"profiler state must be one of {PROFILER_STATES}")
        if self.gfx90a_state not in GFX90A_STATES:
            raise ProfileReportError(
                f"profiler gfx90a_state must be one of {GFX90A_STATES}")
        _text(self.evidence, "profiler evidence")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProfilerCandidate":
        return cls(
            name=_text(payload.get("name"), "profiler.name"),
            state=_text(payload.get("state"), "profiler.state"),
            gfx90a_state=_text(payload.get("gfx90a_state"), "profiler.gfx90a_state"),
            evidence=_text(payload.get("evidence"), "profiler.evidence"),
        )


@dataclass(frozen=True)
class ReportManifest:
    comparison_id: str
    mapping: CaptureSpec
    formal: CaptureSpec
    source_catalog_sha256: str
    cumulative_floor: float
    catalogue_scope: str
    host_catalog_sha256: str | None
    patterns: tuple[SourcePattern, ...]
    architecture_blocks: tuple[ArchitectureBlock, ...]
    profilers: tuple[ProfilerCandidate, ...]

    def __post_init__(self) -> None:
        _text(self.comparison_id, "comparison_id")
        if self.mapping.role != "mapping" or self.formal.role != "formal":
            raise ProfileReportError("manifest must contain mapping then formal captures")
        if self.mapping.stage != self.formal.stage:
            raise ProfileReportError("mapping and formal captures must be stage-separated equally")
        if self.mapping.receipt.source_commit != self.formal.receipt.source_commit:
            raise ProfileReportError("mapping and formal captures must bind the same source commit")
        if self.mapping.receipt.workload_id != self.formal.receipt.workload_id:
            raise ProfileReportError("mapping and formal captures must bind the same workload")
        if self.mapping.receipt.corpus_id == self.formal.receipt.corpus_id:
            raise ProfileReportError("mapping and formal captures must have distinct corpus ids")
        if self.mapping.receipt.profile_path == self.formal.receipt.profile_path:
            raise ProfileReportError("mapping and formal captures must use distinct trace paths")
        _sha256_text(self.source_catalog_sha256, "source_catalog_sha256")
        if (not isinstance(self.cumulative_floor, (int, float)) or isinstance(
                self.cumulative_floor, bool)
                or float(self.cumulative_floor) != CUMULATIVE_FLOOR):
            raise ProfileReportError(
                f"cumulative_floor must be the reviewed {CUMULATIVE_FLOOR:.0%}")
        if self.catalogue_scope not in CATALOGUE_SCOPES:
            raise ProfileReportError(f"catalogue_scope must be one of {CATALOGUE_SCOPES}")
        if self.catalogue_scope == "kernel_and_host":
            _sha256_text(self.host_catalog_sha256, "host_catalog_sha256")
            if self.host_catalog_sha256 == self.source_catalog_sha256:
                raise ProfileReportError(
                    "host catalogue must be independently hash-bound, not alias the kernel catalogue")
        elif self.host_catalog_sha256 is not None:
            raise ProfileReportError(
                "host_catalog_sha256 is only valid with catalogue_scope=kernel_and_host")
        pattern_ids = [row.pattern_id for row in self.patterns]
        block_ids = [row.block_id for row in self.architecture_blocks]
        if len(set(pattern_ids)) != len(pattern_ids):
            raise ProfileReportError("pattern_id values must be unique")
        if len(set(block_ids)) != len(block_ids):
            raise ProfileReportError("architecture block_id values must be unique")
        profiler_names = [row.name for row in self.profilers]
        if len(set(profiler_names)) != len(profiler_names):
            raise ProfileReportError("profiler names must be unique")
        missing = sorted(REQUIRED_PROFILERS - set(profiler_names))
        if missing:
            raise ProfileReportError(f"profiler registry is missing {missing}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReportManifest":
        if not isinstance(payload, Mapping):
            raise ProfileReportError("manifest must be an object")
        patterns = payload.get("patterns")
        blocks = payload.get("architecture_blocks")
        profilers = payload.get("profilers")
        if not isinstance(patterns, list) or not patterns:
            raise ProfileReportError("manifest.patterns must be a non-empty list")
        if not isinstance(blocks, list) or not blocks:
            raise ProfileReportError("manifest.architecture_blocks must be a non-empty list")
        if not isinstance(profilers, list):
            raise ProfileReportError("manifest.profilers must be a list")
        return cls(
            comparison_id=_text(payload.get("comparison_id"), "comparison_id"),
            mapping=CaptureSpec.from_dict(payload.get("mapping")),
            formal=CaptureSpec.from_dict(payload.get("formal")),
            source_catalog_sha256=_sha256_text(
                payload.get("source_catalog_sha256"), "source_catalog_sha256"),
            cumulative_floor=payload.get("cumulative_floor", CUMULATIVE_FLOOR),
            catalogue_scope=_text(
                payload.get("catalogue_scope"), "catalogue_scope"),
            host_catalog_sha256=payload.get("host_catalog_sha256"),
            patterns=tuple(SourcePattern.from_dict(row) for row in patterns),
            architecture_blocks=tuple(ArchitectureBlock.from_dict(row) for row in blocks),
            profilers=tuple(ProfilerCandidate.from_dict(row) for row in profilers),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "mapping": self.mapping.as_dict(),
            "formal": self.formal.as_dict(),
            "source_catalog_sha256": self.source_catalog_sha256,
            "cumulative_floor": self.cumulative_floor,
            "catalogue_scope": self.catalogue_scope,
            "host_catalog_sha256": self.host_catalog_sha256,
            "patterns": [{
                "pattern_id": row.pattern_id,
                "table": row.table,
                "kernel_keywords": list(row.kernel_keywords),
                "match_mode": row.match_mode,
                "source_symbols": list(row.source_symbols),
                "source_paths": list(row.source_paths),
                "reader_should_conclude": row.reader_should_conclude,
            } for row in self.patterns],
            "architecture_blocks": [{
                "block_id": row.block_id,
                "kernel_families": list(row.kernel_families),
                "kernel_family_aliases": [
                    list(aliases) for aliases in row.kernel_family_aliases],
                "source_paths": list(row.source_paths),
            } for row in self.architecture_blocks],
            "profilers": [{
                "name": row.name,
                "state": row.state,
                "gfx90a_state": row.gfx90a_state,
                "evidence": row.evidence,
            } for row in self.profilers],
        }


@dataclass(frozen=True)
class ProfileReport:
    manifest_sha256: str
    manifest: ReportManifest
    kernel_rows: tuple[Mapping[str, Any], ...]
    overlap_rows: tuple[Mapping[str, Any], ...]
    fuse_rows: tuple[Mapping[str, Any], ...]
    architecture_rows: tuple[Mapping[str, Any], ...]
    coverage_gaps: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        judgment_targets = sorted(
            row["pattern_id"] for row in (*self.overlap_rows, *self.fuse_rows))
        return {
            "schema": SCHEMA,
            "manifest_sha256": self.manifest_sha256,
            "comparison_id": self.manifest.comparison_id,
            "stage": self.manifest.mapping.stage,
            "capture_protocol": {
                "mapping": self.manifest.mapping.as_dict(),
                "formal": self.manifest.formal.as_dict(),
            },
            "catalogue": {
                "source_catalog_sha256": self.manifest.source_catalog_sha256,
                "scope": self.manifest.catalogue_scope,
                "host_catalog_sha256": self.manifest.host_catalog_sha256,
            },
            "cumulative_floor": self.manifest.cumulative_floor,
            "kernel_table": [dict(row) for row in self.kernel_rows],
            "overlap_opportunity_table": [dict(row) for row in self.overlap_rows],
            "fuse_pattern_table": [dict(row) for row in self.fuse_rows],
            "architecture_shape_table": [dict(row) for row in self.architecture_rows],
            "profiler_candidates": [{
                "name": row.name,
                "state": row.state,
                "gfx90a_state": row.gfx90a_state,
                "evidence": row.evidence,
            } for row in self.manifest.profilers],
            "coverage_gaps": list(self.coverage_gaps),
            "bounded_judgment_contract": {
                "schema": JUDGMENT_SCHEMA,
                "allowed_similarity": list(SIMILARITY),
                "targets": judgment_targets,
                "allowed_fields": ["pattern_id", "similarity", "catalogue_comparison"],
            },
        }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_architecture_count(
        haystack: Sequence[str], aliases: Sequence[Sequence[str]]) -> int:
    """Count a reviewed logical sequence while tolerating declared family renames."""
    if not aliases or len(aliases) > len(haystack):
        return 0
    return sum(
        all(haystack[index + offset] in accepted
            for offset, accepted in enumerate(aliases))
        for index in range(len(haystack) - len(aliases) + 1)
    )


def _pattern_rows(patterns: Iterable[SourcePattern],
                  mapping: Sequence[prior_art.ProfileDispatch],
                  formal: Sequence[prior_art.ProfileDispatch],
                  formal_shares: Mapping[str, float], table: str,
                  cumulative_floor: float,
                  ) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for pattern in patterns:
        if pattern.table != table:
            continue
        mapping_match = pattern.matches(mapping)
        formal_match = pattern.matches(formal)
        if not mapping_match and not formal_match:
            continue
        formal_families = pattern.matched_families(formal)
        formal_time_share = sum(
            formal_shares.get(name, 0.0) for name in formal_families)
        if formal_time_share < cumulative_floor:
            continue
        rows.append({
            "pattern_id": pattern.pattern_id,
            "mapping_match": mapping_match,
            "formal_match": formal_match,
            "matched_mapping_families": list(pattern.matched_families(mapping)),
            "matched_formal_families": list(formal_families),
            "formal_time_share": formal_time_share,
            "source_symbols": list(pattern.source_symbols),
            "source_paths": list(pattern.source_paths),
            "reader_should_conclude": pattern.reader_should_conclude,
            "attribution_status": "mapped" if mapping_match else "formal_only_mapping_gap",
        })
    return tuple(sorted(rows, key=lambda row: row["pattern_id"]))


def run_profile_report(mapping_path: str | Path, formal_path: str | Path,
                       manifest: ReportManifest) -> ProfileReport:
    """Run only the deterministic pass over two completed profiler captures."""
    mapping = prior_art.load_rocprof_dispatches(mapping_path, manifest.mapping.receipt)
    formal = prior_art.load_rocprof_dispatches(formal_path, manifest.formal.receipt)
    formal_findings = prior_art.load_rocprof_findings(
        formal_path, manifest.formal.receipt)
    admitted = tuple(
        row for row in formal_findings
        if row.finding.gpu_time_share >= manifest.cumulative_floor
    )
    if not admitted:
        raise ProfileReportError("no formal kernel family clears the cumulative floor")
    kernel_rows = tuple({
        "kernel_family": row.kernel_family,
        "dispatches": row.dispatches,
        "duration_ns": row.duration_ns,
        "gpu_time_share": row.finding.gpu_time_share,
    } for row in sorted(admitted, key=lambda item: (-item.duration_ns, item.kernel_family)))
    formal_shares = {
        row.kernel_family: row.finding.gpu_time_share for row in formal_findings}
    overlap = _pattern_rows(
        manifest.patterns, mapping, formal, formal_shares, "overlap",
        manifest.cumulative_floor)
    fuse = _pattern_rows(
        manifest.patterns, mapping, formal, formal_shares, "fuse",
        manifest.cumulative_floor)
    mapping_families = tuple(row.kernel_family for row in mapping)
    architecture_rows = tuple({
        "block_id": block.block_id,
        "exact_sequence_occurrences": _exact_architecture_count(
            mapping_families, block.kernel_family_aliases),
        "kernel_families": list(block.kernel_families),
        "kernel_family_aliases": [
            list(aliases) for aliases in block.kernel_family_aliases],
        "source_paths": list(block.source_paths),
    } for block in sorted(manifest.architecture_blocks, key=lambda item: item.block_id))
    gaps = []
    if manifest.catalogue_scope == "kernel_only":
        gaps.append(
            "host-only scheduler/event-loop/executor/offload/load-path patterns are out of scope")
    if any(not row["mapping_match"] for row in (*overlap, *fuse)):
        gaps.append("one or more formal patterns lack mapping-trace source attribution")
    manifest_dict = manifest.as_dict()
    return ProfileReport(
        manifest_sha256=_canonical_sha256(manifest_dict),
        manifest=manifest,
        kernel_rows=kernel_rows,
        overlap_rows=overlap,
        fuse_rows=fuse,
        architecture_rows=architecture_rows,
        coverage_gaps=tuple(gaps),
    )


def validate_bounded_judgments(report: ProfileReport,
                               rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the only model-authored fields admitted after deterministic C4."""
    targets = {
        row["pattern_id"] for row in (*report.overlap_rows, *report.fuse_rows)}
    seen = set()
    rendered = []
    allowed = {"pattern_id", "similarity", "catalogue_comparison"}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProfileReportError(f"judgment row {index} must be an object")
        unknown = set(row) - allowed
        if unknown:
            raise ProfileReportError(f"judgment row {index} has unknown fields {sorted(unknown)}")
        pattern_id = _text(row.get("pattern_id"), f"judgment row {index} pattern_id")
        if pattern_id not in targets:
            raise ProfileReportError(f"judgment target {pattern_id!r} was not emitted")
        if pattern_id in seen:
            raise ProfileReportError(f"duplicate judgment target {pattern_id!r}")
        seen.add(pattern_id)
        similarity = _text(row.get("similarity"), f"judgment row {index} similarity")
        if similarity not in SIMILARITY:
            raise ProfileReportError(f"similarity must be one of {SIMILARITY}")
        rendered.append({
            "pattern_id": pattern_id,
            "similarity": similarity,
            "catalogue_comparison": _text(
                row.get("catalogue_comparison"),
                f"judgment row {index} catalogue_comparison"),
        })
    return {
        "schema": JUDGMENT_SCHEMA,
        "profile_report_sha256": _canonical_sha256(report.as_dict()),
        "rows": sorted(rendered, key=lambda row: row["pattern_id"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render deterministic C4 tables from completed rocprofv2 captures")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mapping-profile", required=True)
    parser.add_argument("--formal-profile", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = ReportManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8")))
    report = run_profile_report(
        args.mapping_profile, args.formal_profile, manifest).as_dict()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
