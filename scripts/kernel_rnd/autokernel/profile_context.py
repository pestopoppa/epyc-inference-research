"""Hash-bound C4 profile bridge for AutoKernel and external kernel arenas.

The C4 report owns measured facts.  This module turns those facts into two
read-only consumers without granting either consumer authority to rewrite them:

* a compact AutoKernel discovery-context block; and
* a neutral evaluator-observation record suitable for GEAK/AgentKernelArena
  adapters.

Neither record is a verdict or a speed rank.  Both retain the report hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import profile_report


OBSERVATION_SCHEMA = "epyc.autokernel.c4_evaluator_observation.v1"
CONTEXT_SCHEMA = "epyc.autokernel.c4_discovery_context.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ProfileContextError(ValueError):
    """A serialized C4 report cannot safely feed a controller or evaluator."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileContextError(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise ProfileContextError(f"{label} must be a lowercase SHA-256")
    return value


def _ratio(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileContextError(f"{label} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < 0.0 or rendered > 1.0:
        raise ProfileContextError(f"{label} must be finite and in [0, 1]")
    return rendered


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProfileContextError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class KernelMetric:
    family: str
    dispatches: int
    duration_ns: int
    gpu_time_share: float


@dataclass(frozen=True)
class PatternMetric:
    pattern_id: str
    table: str
    formal_time_share: float
    attribution_status: str


@dataclass(frozen=True)
class ArchitectureMetric:
    block_id: str
    exact_sequence_occurrences: int
    kernel_families: tuple[str, ...]


@dataclass(frozen=True)
class C4ProfileContext:
    report_ref: str
    report_sha256: str
    manifest_sha256: str
    comparison_id: str
    stage: str
    source_commit: str
    formal_profile_sha256: str
    kernels: tuple[KernelMetric, ...]
    patterns: tuple[PatternMetric, ...]
    architecture: tuple[ArchitectureMetric, ...]
    coverage_gaps: tuple[str, ...]

    def discovery_context(self) -> dict[str, Any]:
        """Bounded facts for step 1 of the AutoKernel loop.

        Source paths and evaluator implementation names are intentionally absent:
        the authoring prompt needs the mechanism signal, not sealed oracle internals.
        """
        return {
            "schema": CONTEXT_SCHEMA,
            "evidence": {
                "report_ref": self.report_ref,
                "report_sha256": self.report_sha256,
                "formal_profile_sha256": self.formal_profile_sha256,
                "source_commit": self.source_commit,
            },
            "comparison_id": self.comparison_id,
            "stage": self.stage,
            "kernel_wall_share": [{
                "family": row.family,
                "gpu_time_share": row.gpu_time_share,
                "dispatches": row.dispatches,
            } for row in self.kernels],
            "mechanism_opportunities": [{
                "pattern_id": row.pattern_id,
                "kind": row.table,
                "formal_time_share": row.formal_time_share,
                "attribution_status": row.attribution_status,
            } for row in self.patterns],
            "architecture_sequences": [{
                "block_id": row.block_id,
                "exact_sequence_occurrences": row.exact_sequence_occurrences,
                "kernel_families": list(row.kernel_families),
            } for row in self.architecture],
            "coverage_gaps": list(self.coverage_gaps),
        }

    def evaluator_observation(self) -> dict[str, Any]:
        """Framework-neutral metric seam for GEAK/AgentKernelArena adapters."""
        metrics = [{
            "name": "kernel.gpu_time_share",
            "labels": {"kernel_family": row.family, "stage": self.stage},
            "value": row.gpu_time_share,
            "unit": "ratio",
        } for row in self.kernels]
        metrics.extend({
            "name": f"{row.table}.formal_time_share",
            "labels": {"pattern_id": row.pattern_id, "stage": self.stage},
            "value": row.formal_time_share,
            "unit": "ratio",
        } for row in self.patterns)
        return {
            "schema": OBSERVATION_SCHEMA,
            "comparison_id": self.comparison_id,
            "stage": self.stage,
            "source_commit": self.source_commit,
            "evidence": {
                "report_ref": self.report_ref,
                "report_sha256": self.report_sha256,
                "manifest_sha256": self.manifest_sha256,
                "formal_profile_sha256": self.formal_profile_sha256,
            },
            "metrics": metrics,
            "coverage_gaps": list(self.coverage_gaps),
            "authority": "diagnostic_only",
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileContextError(f"{label} must be an object")
    return value


def _rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ProfileContextError(f"{label} must be a list")
    if any(not isinstance(row, Mapping) for row in value):
        raise ProfileContextError(f"{label} rows must be objects")
    return value


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ProfileContextError(f"{label} values must be unique")


def load_profile_context(path: str | Path, *, expected_sha256: str | None = None
                         ) -> C4ProfileContext:
    report_path = Path(path).resolve()
    raw = report_path.read_bytes()
    report_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and report_sha256 != _sha(
            expected_sha256, "expected_sha256"):
        raise ProfileContextError(
            f"report hash mismatch: expected {expected_sha256}, observed {report_sha256}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileContextError("C4 report is not valid UTF-8 JSON") from exc
    root = _mapping(payload, "report")
    if root.get("schema") != profile_report.SCHEMA:
        raise ProfileContextError(
            f"report.schema must be {profile_report.SCHEMA!r}")
    stage = _text(root.get("stage"), "stage")
    if stage not in profile_report.STAGES:
        raise ProfileContextError(f"stage must be one of {profile_report.STAGES}")
    captures = _mapping(root.get("capture_protocol"), "capture_protocol")
    mapping = _mapping(captures.get("mapping"), "capture_protocol.mapping")
    formal = _mapping(captures.get("formal"), "capture_protocol.formal")
    mapping_receipt = _mapping(mapping.get("receipt"), "mapping.receipt")
    formal_receipt = _mapping(formal.get("receipt"), "formal.receipt")
    source_commit = _text(formal_receipt.get("source_commit"), "formal.source_commit")
    if source_commit != _text(mapping_receipt.get("source_commit"),
                              "mapping.source_commit"):
        raise ProfileContextError("mapping and formal source commits differ")

    kernels = []
    for index, row in enumerate(_rows(root.get("kernel_table"), "kernel_table")):
        kernels.append(KernelMetric(
            family=_text(row.get("kernel_family"), f"kernel_table[{index}].family"),
            dispatches=_positive_int(
                row.get("dispatches"), f"kernel_table[{index}].dispatches"),
            duration_ns=_positive_int(
                row.get("duration_ns"), f"kernel_table[{index}].duration_ns"),
            gpu_time_share=_ratio(
                row.get("gpu_time_share"), f"kernel_table[{index}].gpu_time_share"),
        ))
    if not kernels:
        raise ProfileContextError("kernel_table must not be empty")
    _unique([row.family for row in kernels], "kernel family")
    if sum(row.gpu_time_share for row in kernels) > 1.000001:
        raise ProfileContextError("kernel gpu_time_share values sum above one")

    patterns = []
    for table_key, table in (("overlap_opportunity_table", "overlap"),
                             ("fuse_pattern_table", "fuse")):
        for index, row in enumerate(_rows(root.get(table_key), table_key)):
            patterns.append(PatternMetric(
                pattern_id=_text(row.get("pattern_id"),
                                 f"{table_key}[{index}].pattern_id"),
                table=table,
                formal_time_share=_ratio(
                    row.get("formal_time_share"),
                    f"{table_key}[{index}].formal_time_share"),
                attribution_status=_text(
                    row.get("attribution_status"),
                    f"{table_key}[{index}].attribution_status"),
            ))
    _unique([row.pattern_id for row in patterns], "pattern_id")

    architecture = []
    for index, row in enumerate(_rows(
            root.get("architecture_shape_table"), "architecture_shape_table")):
        families = row.get("kernel_families")
        if (not isinstance(families, list) or not families
                or any(not isinstance(item, str) or not item.strip() for item in families)):
            raise ProfileContextError(
                f"architecture_shape_table[{index}].kernel_families must be non-empty strings")
        occurrences = row.get("exact_sequence_occurrences")
        if isinstance(occurrences, bool) or not isinstance(occurrences, int) or occurrences < 0:
            raise ProfileContextError(
                f"architecture_shape_table[{index}].exact_sequence_occurrences "
                "must be a non-negative integer")
        architecture.append(ArchitectureMetric(
            block_id=_text(row.get("block_id"),
                           f"architecture_shape_table[{index}].block_id"),
            exact_sequence_occurrences=occurrences,
            kernel_families=tuple(item.strip() for item in families),
        ))
    _unique([row.block_id for row in architecture], "architecture block_id")

    gaps = root.get("coverage_gaps")
    if (not isinstance(gaps, list)
            or any(not isinstance(item, str) or not item.strip() for item in gaps)):
        raise ProfileContextError("coverage_gaps must be a list of non-empty strings")
    return C4ProfileContext(
        report_ref=str(report_path),
        report_sha256=report_sha256,
        manifest_sha256=_sha(root.get("manifest_sha256"), "manifest_sha256"),
        comparison_id=_text(root.get("comparison_id"), "comparison_id"),
        stage=stage,
        source_commit=source_commit,
        formal_profile_sha256=_sha(
            formal_receipt.get("profile_sha256"), "formal.profile_sha256"),
        kernels=tuple(kernels),
        patterns=tuple(patterns),
        architecture=tuple(architecture),
        coverage_gaps=tuple(item.strip() for item in gaps),
    )
