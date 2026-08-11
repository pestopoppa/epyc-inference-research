#!/usr/bin/env python3
"""Deterministic prior-art gate for AutoKernel proposals.

This module performs no search, build, benchmark, or inference.  It turns a
reviewed catalogue into the four outcomes in the owning handoff §20, so buckets
that already have a config/flag/port answer cannot be relabelled as novel by a
planner.  The catalogue is data; this file owns validation and reduction only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
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
TRACE_MATCH_MODES = frozenset({"any", "all"})
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CatalogueError(ValueError):
    pass


class ProfileError(ValueError):
    """A profile cannot support a receipted prior-art classification."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogueError(f"{label} must be a non-empty string")
    return value.strip()


def _profile_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{label} must be a non-empty string")
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
    trace_match: str = "any"

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
        trace_match = _text(row.get("trace_match", "any"), "trace_match")
        if upstream not in UPSTREAM_STATES:
            raise CatalogueError(f"unknown upstream_state {upstream!r}")
        if local not in LOCAL_STATES:
            raise CatalogueError(f"unknown local_state {local!r}")
        if trace_match not in TRACE_MATCH_MODES:
            raise CatalogueError(f"unknown trace_match {trace_match!r}")
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
            trace_match=trace_match,
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
    keyword_hits = tuple(keyword.casefold() in trace for keyword in row.trace_keywords)
    trace_match = (all(keyword_hits) if row.trace_match == "all"
                   else any(keyword_hits))
    return (trace_match
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


@dataclass(frozen=True)
class ProfileReceipt:
    """Identity and provenance for one already-recorded profiler capture."""

    corpus_id: str
    workload_id: str
    profile_path: str
    profile_sha256: str
    source_commit: str

    def __post_init__(self) -> None:
        _profile_text(self.corpus_id, "corpus_id")
        _profile_text(self.workload_id, "workload_id")
        _profile_text(self.profile_path, "profile_path")
        profile_sha256 = _profile_text(self.profile_sha256, "profile_sha256")
        source_commit = _profile_text(self.source_commit, "source_commit")
        if not _SHA256_RE.fullmatch(profile_sha256):
            raise ProfileError("profile_sha256 must be 64 lowercase hexadecimal digits")
        if not _COMMIT_RE.fullmatch(source_commit):
            raise ProfileError("source_commit must be a pinned hexadecimal commit")


@dataclass(frozen=True)
class ProfileFinding:
    """One duration-aggregated kernel family from a receipted profile."""

    finding: Finding
    kernel_family: str
    dispatches: int
    duration_ns: int


@dataclass(frozen=True)
class ProfileDispatch:
    """One ordered dispatch from a hash-bound rocprofv2 capture."""

    dispatch_id: str
    kernel_name: str
    kernel_family: str
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class ScopeReductionReport:
    """AK-DEL-1's durable four-bucket result over real profiler findings."""

    receipt: ProfileReceipt
    catalogue_sha256: str
    cumulative_floor: float
    captured_dispatches: int
    captured_duration_ns: int
    admitted_duration_ns: int
    bucket_counts: Mapping[str, int]
    bucket_duration_ns: Mapping[str, int]
    existing_or_port_dominates: bool
    recommendation: str
    findings: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "epyc.autokernel.scope_reduction_report.v1",
            "receipt": {
                "corpus_id": self.receipt.corpus_id,
                "workload_id": self.receipt.workload_id,
                "profile_path": self.receipt.profile_path,
                "profile_sha256": self.receipt.profile_sha256,
                "source_commit": self.receipt.source_commit,
            },
            "catalogue_sha256": self.catalogue_sha256,
            "cumulative_floor": self.cumulative_floor,
            "captured_dispatches": self.captured_dispatches,
            "captured_duration_ns": self.captured_duration_ns,
            "admitted_duration_ns": self.admitted_duration_ns,
            "bucket_counts": dict(self.bucket_counts),
            "bucket_duration_ns": dict(self.bucket_duration_ns),
            "existing_or_port_dominates": self.existing_or_port_dominates,
            "recommendation": self.recommendation,
            "findings": [dict(row) for row in self.findings],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kernel_family(kernel_name: str) -> str:
    """Reduce a demangled rocprof kernel to its stable callable family."""
    name = _profile_text(kernel_name, "Kernel_Name")
    if name.endswith(" (.kd)"):
        name = name[:-6]
    if name.endswith(".kd"):
        name = name[:-3]
    if name.startswith("void "):
        name = name[5:]
    cut = len(name)
    for marker in ("<", "("):
        position = name.find(marker)
        if position >= 0:
            cut = min(cut, position)
    family = name[:cut].strip()
    if not family:
        raise ProfileError(f"could not derive a kernel family from {kernel_name!r}")
    return family


def load_rocprof_dispatches(path: str | Path,
                            receipt: ProfileReceipt) -> tuple[ProfileDispatch, ...]:
    """Parse ordered rocprofv2 dispatches after verifying the capture hash."""
    source = Path(path)
    if not source.is_file():
        raise ProfileError(f"profile does not exist: {source}")
    observed_hash = _sha256(source)
    if observed_hash != receipt.profile_sha256:
        raise ProfileError(
            f"profile sha256 mismatch: expected {receipt.profile_sha256}, got {observed_hash}")
    seen_ids: set[str] = set()
    required = {"Dispatch_ID", "Kernel_Name", "Start_Timestamp", "End_Timestamp"}
    dispatches = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ProfileError(f"rocprof CSV is missing required columns: {missing}")
        for row_number, row in enumerate(reader, 2):
            dispatch_id = _profile_text(
                row.get("Dispatch_ID"), f"row {row_number} Dispatch_ID")
            if dispatch_id in seen_ids:
                raise ProfileError(f"duplicate Dispatch_ID {dispatch_id!r}")
            seen_ids.add(dispatch_id)
            try:
                start = int(_profile_text(row.get("Start_Timestamp"),
                                          f"row {row_number} Start_Timestamp"))
                end = int(_profile_text(row.get("End_Timestamp"),
                                        f"row {row_number} End_Timestamp"))
            except ValueError as exc:
                raise ProfileError(f"row {row_number} has a non-integer timestamp") from exc
            if end <= start:
                raise ProfileError(
                    f"row {row_number} has non-positive dispatch duration: {start}..{end}")
            kernel_name = _profile_text(
                row.get("Kernel_Name"), f"row {row_number} Kernel_Name")
            dispatches.append(ProfileDispatch(
                dispatch_id=dispatch_id,
                kernel_name=kernel_name,
                kernel_family=_kernel_family(kernel_name),
                start_ns=start,
                end_ns=end,
            ))
    if not dispatches:
        raise ProfileError("rocprof CSV contains no positive-duration dispatches")
    return tuple(dispatches)


def load_rocprof_findings(path: str | Path, receipt: ProfileReceipt, *,
                          active_flags: Mapping[str, str] | None = None,
                          symbol_aliases: Mapping[str, Sequence[str]] | None = None
                          ) -> tuple[ProfileFinding, ...]:
    """Parse and aggregate a rocprofv2 CSV after verifying its content hash.

    This consumes a completed capture.  It launches nothing and deliberately
    treats timestamps as the only timing authority; hardware counter values are
    not substituted for dispatch duration.
    """
    flags = dict(active_flags or {})
    aliases = dict(symbol_aliases or {})
    durations: dict[str, int] = defaultdict(int)
    dispatches: dict[str, int] = defaultdict(int)
    names: dict[str, set[str]] = defaultdict(set)
    ordered = load_rocprof_dispatches(path, receipt)
    for row in ordered:
        durations[row.kernel_family] += row.duration_ns
        dispatches[row.kernel_family] += 1
        names[row.kernel_family].add(row.kernel_name)
    total = sum(durations.values())
    if total <= 0:
        raise ProfileError("rocprof CSV contains no positive-duration dispatches")
    findings = []
    for family in sorted(durations):
        trace_text = " | ".join(sorted(names[family]))
        finding = Finding(
            finding_id=f"{receipt.corpus_id}:{family}",
            trace_text=trace_text,
            symbols=tuple(aliases.get(family, ())),
            active_flags=flags,
            gpu_time_share=durations[family] / total,
        )
        findings.append(ProfileFinding(
            finding=finding,
            kernel_family=family,
            dispatches=dispatches[family],
            duration_ns=durations[family],
        ))
    return tuple(findings)


def run_scope_reduction(profile_path: str | Path, receipt: ProfileReceipt,
                        catalogue_path: str | Path | None = None, *,
                        cumulative_floor: float = 0.01,
                        active_flags: Mapping[str, str] | None = None,
                        symbol_aliases: Mapping[str, Sequence[str]] | None = None,
                        catalogue: Catalogue | None = None) -> ScopeReductionReport:
    """Execute AK-DEL-1 over a completed profile and the reviewed catalogue."""
    if not 0 <= cumulative_floor <= 1:
        raise ValueError("cumulative_floor must be in [0,1]")
    catalogue_file = (Path(catalogue_path) if catalogue_path is not None
                      else Path(__file__).with_name("prior_art_catalogue.json"))
    if catalogue is None:
        catalogue = load_catalogue(catalogue_file)
        catalogue_hash = _sha256(catalogue_file)
    else:
        canonical = json.dumps({
            "rows": [row.__dict__ for row in catalogue.rows],
            "expected_absence": [row.__dict__ for row in catalogue.expected_absence],
            "scanned_at": catalogue.scanned_at,
            "scan_commands": catalogue.scan_commands,
            "searched_trees": catalogue.searched_trees,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        catalogue_hash = hashlib.sha256(canonical).hexdigest()
    parsed = load_rocprof_findings(
        profile_path, receipt, active_flags=active_flags, symbol_aliases=symbol_aliases)
    admitted_ids = {
        row.finding_id for row in proposal_space(
            tuple(row.finding for row in parsed), catalogue,
            cumulative_floor=cumulative_floor)
    }
    admitted = tuple(row for row in parsed if row.finding.finding_id in admitted_ids)
    if not admitted:
        raise ProfileError("no kernel family clears the cumulative wall-share floor")
    counts = {bucket: 0 for bucket in BUCKETS}
    duration_by_bucket = {bucket: 0 for bucket in BUCKETS}
    rendered = []
    for row in admitted:
        result = classify(row.finding, catalogue)
        counts[result.bucket] += 1
        duration_by_bucket[result.bucket] += row.duration_ns
        rendered.append({
            "finding_id": result.finding_id,
            "kernel_family": row.kernel_family,
            "dispatches": row.dispatches,
            "duration_ns": row.duration_ns,
            "captured_time_share": row.finding.gpu_time_share,
            "bucket": result.bucket,
            "exit_action": result.exit_action,
            "matched_pattern": result.matched_pattern,
            "reason": result.reason,
            "source_commit": result.source_commit,
        })
    existing_count = sum(counts[bucket] for bucket in BUCKETS[:-1])
    novel_count = counts[BUCKET_NOVEL]
    dominates = existing_count > novel_count
    recommendation = (
        "expand_catalogue_before_novel_generator"
        if dominates else "retain_novel_generator_scope"
    )
    return ScopeReductionReport(
        receipt=receipt,
        catalogue_sha256=catalogue_hash,
        cumulative_floor=cumulative_floor,
        captured_dispatches=sum(row.dispatches for row in parsed),
        captured_duration_ns=sum(row.duration_ns for row in parsed),
        admitted_duration_ns=sum(row.duration_ns for row in admitted),
        bucket_counts=counts,
        bucket_duration_ns=duration_by_bucket,
        existing_or_port_dominates=dominates,
        recommendation=recommendation,
        findings=tuple(rendered),
    )


def _parse_flag(value: str) -> tuple[str, str]:
    name, separator, state = value.partition("=")
    if not separator or not name.strip() or not state.strip():
        raise argparse.ArgumentTypeError("active flags use NAME=STATE")
    return name.strip(), state.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run AK-DEL-1 over an already-recorded rocprofv2 CSV")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--catalogue", default=str(
        Path(__file__).with_name("prior_art_catalogue.json")))
    parser.add_argument("--cumulative-floor", type=float, default=0.01)
    parser.add_argument("--active-flag", action="append", type=_parse_flag, default=[])
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    receipt = ProfileReceipt(
        corpus_id=args.corpus_id,
        workload_id=args.workload_id,
        profile_path=args.profile,
        profile_sha256=args.profile_sha256,
        source_commit=args.source_commit,
    )
    report = run_scope_reduction(
        args.profile, receipt, args.catalogue,
        cumulative_floor=args.cumulative_floor,
        active_flags=dict(args.active_flag))
    encoded = json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
