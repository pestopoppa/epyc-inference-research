#!/usr/bin/env python3
"""Deterministic prior-art gate for AutoKernel proposals.

This module performs no search, build, benchmark, or inference.  It turns a
reviewed catalogue into the four outcomes in the owning handoff §20, so buckets
that already have a config/flag/port answer cannot be relabelled as novel by a
planner.  The catalogue is data; this file owns validation and reduction only.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BUCKET_EXISTING_APPLIES = "existing_path_should_apply"
BUCKET_EXISTING_DISABLED = "existing_path_disabled_unsupported_or_regressed"
BUCKET_UPSTREAM_MISSING = "upstream_mainline_or_open_missing_locally"
BUCKET_NOVEL = "genuinely_new_no_catalogue_match"
BUCKETS = (
    BUCKET_EXISTING_APPLIES,
    BUCKET_EXISTING_DISABLED,
    BUCKET_UPSTREAM_MISSING,
    BUCKET_NOVEL,
)

EXIT_BY_BUCKET = {
    BUCKET_EXISTING_APPLIES: "config_or_dispatch_fix",
    BUCKET_EXISTING_DISABLED: "flag_support_or_regression_fix",
    BUCKET_UPSTREAM_MISSING: "port_or_forward_port",
    BUCKET_NOVEL: "research_campaign_eligible",
}

UPSTREAM_STATES = frozenset({"mainline", "in_flight", "local_only"})
LOCAL_STATES = frozenset({"present", "disabled", "unsupported", "regressed", "absent"})
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


class CatalogueError(ValueError):
    pass


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class CatalogueRow:
    pattern: str
    trace_keywords: tuple[str, ...]
    primary_code: tuple[str, ...]
    existing_path: str
    reader_should_conclude: str
    upstream_state: str
    local_state: str
    source_project: str
    source_commit: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "CatalogueRow":
        if not isinstance(row, Mapping):
            raise CatalogueError("catalogue row must be a mapping")
        keywords = row.get("trace_keywords")
        code = row.get("primary_code")
        if not isinstance(keywords, list) or not keywords or not all(
                isinstance(v, str) and v.strip() for v in keywords):
            raise CatalogueError("trace_keywords must be a non-empty string list")
        if not isinstance(code, list) or not code or not all(
                isinstance(v, str) and v.strip() for v in code):
            raise CatalogueError("primary_code must be a non-empty string list")
        upstream = _text(row.get("upstream_state"), "upstream_state")
        local = _text(row.get("local_state"), "local_state")
        commit = _text(row.get("source_commit"), "source_commit")
        if upstream not in UPSTREAM_STATES:
            raise CatalogueError(f"unknown upstream_state {upstream!r}")
        if local not in LOCAL_STATES:
            raise CatalogueError(f"unknown local_state {local!r}")
        if not _COMMIT_RE.fullmatch(commit):
            raise CatalogueError("source_commit must be a pinned hexadecimal commit")
        return cls(
            pattern=_text(row.get("pattern"), "pattern"),
            trace_keywords=tuple(v.strip() for v in keywords),
            primary_code=tuple(v.strip() for v in code),
            existing_path=_text(row.get("existing_path"), "existing_path"),
            reader_should_conclude=_text(
                row.get("reader_should_conclude"), "reader_should_conclude"),
            upstream_state=upstream,
            local_state=local,
            source_project=_text(row.get("source_project"), "source_project"),
            source_commit=commit,
        )


@dataclass(frozen=True)
class ExpectedAbsence:
    flag: str
    state: str
    trace_effect: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ExpectedAbsence":
        return cls(_text(row.get("flag"), "flag"),
                   _text(row.get("state"), "state"),
                   _text(row.get("trace_effect"), "trace_effect"))


@dataclass(frozen=True)
class Catalogue:
    scanned_at: str
    scan_commands: tuple[str, ...]
    searched_trees: tuple[str, ...]
    rows: tuple[CatalogueRow, ...]
    expected_absence: tuple[ExpectedAbsence, ...]

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "Catalogue":
        if not isinstance(doc, Mapping):
            raise CatalogueError("catalogue must be a mapping")
        scanned_at = _text(doc.get("scanned_at"), "scanned_at")
        try:
            datetime.fromisoformat(scanned_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CatalogueError("scanned_at must be ISO-8601") from exc
        commands = doc.get("scan_commands")
        trees = doc.get("searched_trees")
        if not isinstance(commands, list) or not commands:
            raise CatalogueError("scan_commands must be non-empty")
        if not isinstance(trees, list) or not trees:
            raise CatalogueError("searched_trees must be non-empty")
        tree_classes = {str(row).split(":", 1)[0] for row in trees}
        if not {"model", "kernel"}.issubset(tree_classes):
            raise CatalogueError(
                "absence claims require both model: and kernel: trees to be searched")
        rows = tuple(CatalogueRow.from_dict(row) for row in doc.get("rows", ()))
        expected = tuple(ExpectedAbsence.from_dict(row)
                         for row in doc.get("expected_absence", ()))
        if not rows:
            raise CatalogueError("catalogue seed set is empty")
        if not expected:
            raise CatalogueError("expected-absence register must precede catalogue use")
        return cls(scanned_at=scanned_at,
                   scan_commands=tuple(_text(v, "scan_commands[]") for v in commands),
                   searched_trees=tuple(_text(v, "searched_trees[]") for v in trees),
                   rows=rows, expected_absence=expected)


@dataclass(frozen=True)
class Finding:
    finding_id: str
    trace_text: str
    symbols: tuple[str, ...]
    active_flags: Mapping[str, str]
    gpu_time_share: float

    def __post_init__(self) -> None:
        _text(self.finding_id, "finding_id")
        _text(self.trace_text, "trace_text")
        if not isinstance(self.symbols, tuple):
            raise TypeError("symbols must be a tuple")
        if not isinstance(self.active_flags, Mapping):
            raise TypeError("active_flags must be a mapping")
        if isinstance(self.gpu_time_share, bool) or not isinstance(
                self.gpu_time_share, (int, float)) or not math.isfinite(self.gpu_time_share) \
                or not 0 <= self.gpu_time_share <= 1:
            raise ValueError("gpu_time_share must be finite in [0,1]")


@dataclass(frozen=True)
class Classification:
    finding_id: str
    bucket: str
    exit_action: str
    matched_pattern: str | None
    reason: str
    source_commit: str | None


def _matches(finding: Finding, row: CatalogueRow) -> bool:
    trace = finding.trace_text.casefold()
    symbols = {symbol.casefold() for symbol in finding.symbols}
    return (any(keyword.casefold() in trace for keyword in row.trace_keywords)
            or any(code.casefold() in symbols for code in row.primary_code))


def classify(finding: Finding, catalogue: Catalogue) -> Classification:
    matches = [row for row in catalogue.rows if _matches(finding, row)]
    if not matches:
        return Classification(finding.finding_id, BUCKET_NOVEL,
                              EXIT_BY_BUCKET[BUCKET_NOVEL], None,
                              "no reviewed catalogue row matched", None)
    # A more conservative bucket wins when several catalogue rows match: an
    # existing/port answer must not be hidden behind a looser novel-ish row.
    row = sorted(matches, key=lambda item: (
        {"present": 0, "disabled": 1, "unsupported": 1, "regressed": 1,
         "absent": 2}[item.local_state], item.pattern))[0]
    expected = [entry for entry in catalogue.expected_absence
                if finding.active_flags.get(entry.flag) == entry.state]
    if expected or row.local_state in {"disabled", "unsupported", "regressed"}:
        bucket = BUCKET_EXISTING_DISABLED
        reason = (expected[0].trace_effect if expected else row.reader_should_conclude)
    elif row.local_state == "present":
        bucket = BUCKET_EXISTING_APPLIES
        reason = row.reader_should_conclude
    elif row.upstream_state in {"mainline", "in_flight"} and row.local_state == "absent":
        bucket = BUCKET_UPSTREAM_MISSING
        reason = row.reader_should_conclude
    else:
        bucket = BUCKET_NOVEL
        reason = "matched only a local-only absent row; no existing or upstream path resolves it"
    return Classification(finding.finding_id, bucket, EXIT_BY_BUCKET[bucket],
                          row.pattern, reason, row.source_commit)


def proposal_space(finding_rows: Sequence[Finding], catalogue: Catalogue, *,
                   cumulative_floor: float = 0.01) -> tuple[Classification, ...]:
    """Classify only mechanisms whose cumulative observed GPU share clears floor."""
    if not 0 <= cumulative_floor <= 1:
        raise ValueError("cumulative_floor must be in [0,1]")
    grouped: dict[str, list[Finding]] = {}
    for finding in finding_rows:
        result = classify(finding, catalogue)
        key = result.matched_pattern or finding.finding_id
        grouped.setdefault(key, []).append(finding)
    admitted: list[Classification] = []
    for key in sorted(grouped):
        rows = grouped[key]
        if sum(row.gpu_time_share for row in rows) >= cumulative_floor:
            admitted.extend(classify(row, catalogue) for row in rows)
    return tuple(admitted)


def bucket_split(findings: Iterable[Finding], catalogue: Catalogue) -> dict[str, int]:
    counts = {bucket: 0 for bucket in BUCKETS}
    for finding in findings:
        counts[classify(finding, catalogue).bucket] += 1
    return counts


def load_catalogue(path: str | Path | None = None) -> Catalogue:
    source = Path(path) if path is not None else Path(__file__).with_name(
        "prior_art_catalogue.json")
    return Catalogue.from_dict(json.loads(source.read_text(encoding="utf-8")))

