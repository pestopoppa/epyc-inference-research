"""Fail-closed Apex trace adapter for the INF-48 C3 EPYC cases.

The two C5 records selected by :mod:`c3_epyc_suite` are operator specifications,
not Apex registry identities.  This module therefore never guesses a registry
entry from a similar-looking name.  A separately reviewed mapping artifact must
bind both C5 records to exact Apex entries and include a hash-bound semantic
equivalence artifact before a trace plan can be produced.

The adapter deliberately resolves and validates only the mapped registry rows.
Apex ``e06b5d1`` calls ``find_supported_kernel(validate_files=True)``, which
validates every row and refuses an otherwise present AITER entry when unrelated
vLLM or SGLang source trees are absent.  The pinned registry hash makes global
schema drift detectable without imposing that unrelated-file requirement.

No candidate is authored here.  ``prepare_trace_plan`` performs static and host
preflight only.  ``execute_trace`` requires an explicit inference authorization
argument and calls only Apex's pinned ``run_trace_kernel`` entrypoint.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import yaml

from . import c3_epyc_suite as c3
from . import c3_epyc_tensor_capture as tensor_capture


MAPPING_SCHEMA = "epyc.autokernel.c3_apex_case_mapping.v2"
MAPPING_AUDIT_SCHEMA = "epyc.autokernel.c3_apex_mapping_audit.v1"
SEMANTIC_REVIEW_SCHEMA = "epyc.autokernel.c3_apex_semantic_review.v1"
ARCHITECTURE_REVIEW_SCHEMA = "epyc.autokernel.c3_apex_gfx90a_review.v1"
MODEL_MANIFEST_SCHEMA = "epyc.autokernel.model_identity.v1"
PLAN_SCHEMA = "epyc.autokernel.c3_apex_trace_plan.v2"
CAPTURE_SCHEMA = "epyc.autokernel.c3_apex_capture.v2"

PINNED_APEX_REVISION = c3.PINNED_APEX_REVISION
PINNED_MAGPIE_REVISION = "2a9263833f71755df2a93b466cdd3a9f803fc625"
PINNED_AITER_REVISION = "7890e4be789ac362d3033437d09920ddd5f2891a"
PINNED_APEX_REGISTRY_SHA256 = (
    "72d6529ca945a860abd2ba22dd26bb8dbf8d9c33797327c0e5ce9e11d8047a61"
)
PINNED_AITER_HSA_INVENTORY_SHA256 = (
    "e0503ea08e860b1af7c5f5d0f235ec53310dcdf23932d1b201881533e1cb02dc"
)
PINNED_AITER_HSA_FILE_COUNT = 2867
PINNED_TORCH_VERSION = "2.5.1+rocm6.2"
PINNED_TRITON_VERSION = "3.1.0"
PINNED_HIP_PREFIX = "6.2"
TARGET_ARCH = "gfx90a"

APEX_REGISTRY_RELATIVE = Path("pipeline/kernel_tracing/supported_kernels.yaml")
APEX_RUNNER_ENTRYPOINT = "pipeline.kernel_tracing.runner.run_trace_kernel"
MISSING_MAPPING_ARTIFACT = "c3_apex_case_mapping.v2.json"
DEFAULT_MAPPING_AUDIT = Path(__file__).with_name("c3_apex_mapping_audit.v1.json")
REQUIRED_CAPTURE_OUTPUTS = (
    "trace_result.json",
    "workload_ranges.json",
    "patched_files/patch_manifest.json",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ENTRY_FIELDS = {
    "id", "repo", "kernel_type", "kernel_name", "kernel_file", "trace_mode",
    "patch_strategy",
}
_VALID_REPOS = {"aiter", "vllm", "sglang"}
_VALID_KERNEL_TYPES = {"triton", "hip"}
_VALID_TRACE_MODES = {
    "triton-launch", "aiter-compile-ops", "vllm-custom-op", "sglang-custom-op",
}
_SINGLE_ENTRY = "gfx90a_single_trace"
_ORDERED_COMPOSITE = "ordered_multi_trace_composite"
_COMPONENT_BRANCHES = {"always", "n_le_1350", "n_gt_1350"}
_COMPONENT_STREAMS = {"main", "shared"}
_K228_COMPONENT_IDS = ("mla_paged_prefill",)
_K175_COMPONENT_IDS = (
    "router_projection", "biased_top8_counts_and_ranks",
    "dispatch_n_le_1350", "dispatch_n_gt_1350", "routed_experts",
    "shared_experts", "weighted_undispatch_shared_add", "graph_capture_replay",
)
_K175_COMPONENT_BRANCHES = (
    "always", "always", "n_le_1350", "n_gt_1350", "always", "always",
    "always", "always",
)
_K175_COMPONENT_STREAMS = (
    "main", "main", "main", "main", "main", "shared", "main", "main",
)
_TRACE_CONFIG_FIELDS = {
    "results_dir", "kernel_name", "kernel_file", "kernel_id", "registry_entry",
    "trace_mode", "kernel_type", "patch_strategy", "benchmark_config", "run_cmd",
    "max_records", "sample_rate", "small_tensor_stats", "trace_all", "agent_backend",
    "agent_model", "agent_max_turns", "benchmark_timeout", "docker_image", "framework",
    "dry_run", "repo_root",
}

CASE_REQUIREMENTS: Mapping[str, Mapping[str, str]] = {
    "epyc.attention.mla_paged_prefill.k228": {
        "c5_ref": "hyra-sol-execbench/k228",
        "c5_artifact_sha256": (
            "696b6b0802ba4d2ae371cbdeddee07f5ca2796827805dcc90561205c3f81d83a"
        ),
        "operator_family": "mla_paged_prefill",
    },
    "epyc.moe.sparse_expert_dispatch.k175": {
        "c5_ref": "hyra-sol-execbench/k175",
        "c5_artifact_sha256": (
            "5b28cd02aab7e1ad1b8c48394e220ae398036ce81416e153c7a5e3237ba6f8c3"
        ),
        "operator_family": "moe_sparse_expert_dispatch",
    },
}


class ApexPreflightRefusal(ValueError):
    """The trace target is not sufficiently identified to launch."""


class MissingCaseMapping(ApexPreflightRefusal):
    """No reviewed exact C5-to-Apex mapping artifact exists."""


class StructuralMappingMismatch(MissingCaseMapping):
    """The pinned sources prove that an exact case mapping is not executable."""

    def __init__(self, case_id: str, missing_components: tuple[tuple[str, str], ...]):
        self.case_id = case_id
        self.missing_components = missing_components
        rendered = "; ".join(f"{name}: {detail}"
                             for name, detail in missing_components)
        super().__init__(
            f"{case_id} has a reviewed structural mismatch against pinned "
            f"Apex/AITER on {TARGET_ARCH}: {rendered}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApexPreflightRefusal(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    value = _text(value, label)
    if not _SHA256_RE.fullmatch(value) or len(set(value)) == 1:
        raise ApexPreflightRefusal(f"{label} must be a non-placeholder SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    value = _text(value, label)
    if not _COMMIT_RE.fullmatch(value):
        raise ApexPreflightRefusal(f"{label} must be a full lowercase commit")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ApexPreflightRefusal(
            f"{label} fields differ from schema; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ApexPreflightRefusal(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, json.JSONDecodeError) as exc:
        raise ApexPreflightRefusal(f"cannot read {label} {path}: {exc}") from exc


def _checked_file(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file():
        raise ApexPreflightRefusal(f"{label} is not a file: {path}")
    observed = _sha256_file(path)
    if observed != _sha(expected_sha256, f"{label}.sha256"):
        raise ApexPreflightRefusal(
            f"{label} hash mismatch: expected {expected_sha256}, observed {observed}")
    return path


@dataclass(frozen=True)
class MappingAuditCase:
    case_id: str
    c5_ref: str
    c5_artifact_path: str
    c5_artifact_sha256: str
    disposition: str
    closest_registry_entries: tuple[str, ...]
    component_graph: tuple[str, ...]
    missing_components: tuple[tuple[str, str], ...]
    evidence: tuple[tuple[str, str], ...]

    def refuse(self) -> None:
        raise StructuralMappingMismatch(self.case_id, self.missing_components)


@dataclass(frozen=True)
class MappingAudit:
    artifact_path: Path
    artifact_sha256: str
    cases: tuple[MappingAuditCase, ...]

    def select(self, case_id: str) -> MappingAuditCase:
        matches = [case for case in self.cases if case.case_id == case_id]
        if len(matches) != 1:
            raise ApexPreflightRefusal(
                f"mapping audit does not contain exactly one {case_id} row")
        return matches[0]


def _text_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ApexPreflightRefusal(f"{label} must be a non-empty list")
    return tuple(_text(item, f"{label}[{index}]")
                 for index, item in enumerate(value))


def load_mapping_audit(path: Path = DEFAULT_MAPPING_AUDIT) -> MappingAudit:
    """Load the hash-bound source audit that explains current mapping refusals."""
    path = Path(path)
    document = _read_json(path, "C3 Apex mapping audit")
    _exact_keys(document, {"schema", "audit_date", "authority", "pins", "cases"},
                "mapping audit")
    if document["schema"] != MAPPING_AUDIT_SCHEMA:
        raise ApexPreflightRefusal("unsupported C3 Apex mapping-audit schema")
    _text(document["audit_date"], "mapping audit.audit_date")
    if document["authority"] != (
            "static_source_audit_only_no_runtime_equivalence_or_inference"):
        raise ApexPreflightRefusal("mapping audit overstates its authority")
    pins = _mapping(document["pins"], "mapping audit.pins")
    _exact_keys(pins, {
        "apex_revision", "magpie_revision", "aiter_revision", "registry_sha256",
        "target_architecture", "aiter_hsa_architectures", "aiter_hsa_file_count",
        "aiter_hsa_inventory_sha256",
    }, "mapping audit.pins")
    if pins["apex_revision"] != PINNED_APEX_REVISION \
            or pins["magpie_revision"] != PINNED_MAGPIE_REVISION:
        raise ApexPreflightRefusal("mapping audit names the wrong Apex/Magpie pins")
    if pins["aiter_revision"] != PINNED_AITER_REVISION:
        raise ApexPreflightRefusal("mapping audit names the wrong AITER pin")
    if pins["registry_sha256"] != PINNED_APEX_REGISTRY_SHA256:
        raise ApexPreflightRefusal("mapping audit names the wrong Apex registry hash")
    if pins["aiter_hsa_inventory_sha256"] != PINNED_AITER_HSA_INVENTORY_SHA256:
        raise ApexPreflightRefusal("mapping audit names the wrong AITER HSA inventory hash")
    if pins["target_architecture"] != TARGET_ARCH:
        raise ApexPreflightRefusal("mapping audit names the wrong target architecture")
    if pins["aiter_hsa_architectures"] != ["gfx942", "gfx950"]:
        raise ApexPreflightRefusal("mapping audit AITER HSA architecture inventory drifted")
    if pins["aiter_hsa_file_count"] != PINNED_AITER_HSA_FILE_COUNT:
        raise ApexPreflightRefusal("mapping audit AITER HSA file count drifted")
    rows = document["cases"]
    if not isinstance(rows, list):
        raise ApexPreflightRefusal("mapping audit cases must be a list")
    cases = []
    for index, raw in enumerate(rows):
        label = f"mapping audit.cases[{index}]"
        row = _mapping(raw, label)
        _exact_keys(row, {
            "case_id", "c5_ref", "c5_artifact_path", "c5_artifact_sha256",
            "disposition", "closest_registry_entries", "component_graph",
            "missing_components", "evidence",
        }, label)
        case_id = _text(row["case_id"], f"{label}.case_id")
        expected = CASE_REQUIREMENTS.get(case_id)
        if expected is None or row["c5_ref"] != expected["c5_ref"] \
                or row["c5_artifact_sha256"] != expected["c5_artifact_sha256"]:
            raise ApexPreflightRefusal(f"{label} does not bind the exact C5 artifact")
        if row["disposition"] != "structural_mismatch":
            raise ApexPreflightRefusal(f"{label} overstates mapping executability")
        missing_raw = row["missing_components"]
        if not isinstance(missing_raw, list) or not missing_raw:
            raise ApexPreflightRefusal(f"{label}.missing_components must be non-empty")
        missing = []
        for missing_index, missing_value in enumerate(missing_raw):
            missing_label = f"{label}.missing_components[{missing_index}]"
            component = _mapping(missing_value, missing_label)
            _exact_keys(component, {"id", "detail"}, missing_label)
            missing.append((_text(component["id"], f"{missing_label}.id"),
                            _text(component["detail"], f"{missing_label}.detail")))
        if len({item[0] for item in missing}) != len(missing):
            raise ApexPreflightRefusal(f"{label} repeats a missing-component ID")
        evidence_raw = row["evidence"]
        if not isinstance(evidence_raw, list) or not evidence_raw:
            raise ApexPreflightRefusal(f"{label}.evidence must be non-empty")
        evidence = []
        for evidence_index, evidence_value in enumerate(evidence_raw):
            evidence_label = f"{label}.evidence[{evidence_index}]"
            item = _mapping(evidence_value, evidence_label)
            _exact_keys(item, {"path", "sha256"}, evidence_label)
            evidence.append((_text(item["path"], f"{evidence_label}.path"),
                             _sha(item["sha256"], f"{evidence_label}.sha256")))
        cases.append(MappingAuditCase(
            case_id=case_id,
            c5_ref=_text(row["c5_ref"], f"{label}.c5_ref"),
            c5_artifact_path=_text(
                row["c5_artifact_path"], f"{label}.c5_artifact_path"),
            c5_artifact_sha256=_sha(
                row["c5_artifact_sha256"], f"{label}.c5_artifact_sha256"),
            disposition=row["disposition"],
            closest_registry_entries=_text_list(
                row["closest_registry_entries"], f"{label}.closest_registry_entries"),
            component_graph=_text_list(
                row["component_graph"], f"{label}.component_graph"),
            missing_components=tuple(missing), evidence=tuple(evidence),
        ))
    if {case.case_id for case in cases} != set(CASE_REQUIREMENTS) \
            or len(cases) != len(CASE_REQUIREMENTS):
        raise ApexPreflightRefusal("mapping audit must cover exactly k228 and k175")
    return MappingAudit(path.resolve(), _sha256_file(path), tuple(cases))


@dataclass(frozen=True)
class MappingComponent:
    component_id: str
    order: int
    branch: str
    stream: str
    depends_on: tuple[str, ...]
    kernel_id: str
    source_repo: str
    source_commit: str
    source_file: str
    source_file_sha256: str
    architecture_review_ref: str
    architecture_review_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], label: str) -> "MappingComponent":
        required = {"component_id", "order", "branch", "stream", "depends_on",
                    "kernel_id", "source_repo", "source_commit", "source_file",
                    "source_file_sha256", "architecture_review_ref",
                    "architecture_review_sha256"}
        _exact_keys(value, required, label)
        order = value["order"]
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ApexPreflightRefusal(f"{label}.order must be a non-negative integer")
        depends_on = value["depends_on"]
        if not isinstance(depends_on, list) or any(
                not isinstance(item, str) or not item for item in depends_on):
            raise ApexPreflightRefusal(f"{label}.depends_on must be a string list")
        result = cls(
            component_id=_text(value["component_id"], f"{label}.component_id"),
            order=order, branch=_text(value["branch"], f"{label}.branch"),
            stream=_text(value["stream"], f"{label}.stream"),
            depends_on=tuple(depends_on),
            kernel_id=_text(value["kernel_id"], f"{label}.kernel_id"),
            source_repo=_text(value["source_repo"], f"{label}.source_repo"),
            source_commit=_commit(value["source_commit"], f"{label}.source_commit"),
            source_file=_text(value["source_file"], f"{label}.source_file"),
            source_file_sha256=_sha(value["source_file_sha256"],
                                    f"{label}.source_file_sha256"),
            architecture_review_ref=_text(
                value["architecture_review_ref"], f"{label}.architecture_review_ref"),
            architecture_review_sha256=_sha(
                value["architecture_review_sha256"],
                f"{label}.architecture_review_sha256"),
        )
        if result.branch not in _COMPONENT_BRANCHES:
            raise ApexPreflightRefusal(f"{label} names an unsupported branch")
        if result.stream not in _COMPONENT_STREAMS:
            raise ApexPreflightRefusal(f"{label} names an unsupported stream")
        if result.source_repo not in _VALID_REPOS:
            raise ApexPreflightRefusal(f"{label} names an unsupported Apex source repo")
        return result


@dataclass(frozen=True)
class CaseMapping:
    case_id: str
    c5_ref: str
    c5_artifact_sha256: str
    binding_kind: str
    semantic_binding_ref: str
    semantic_binding_sha256: str
    components: tuple[MappingComponent, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], label: str) -> "CaseMapping":
        required = {"case_id", "c5_ref", "c5_artifact_sha256", "binding_kind",
                    "semantic_binding_ref", "semantic_binding_sha256", "components"}
        _exact_keys(value, required, label)
        raw_components = value["components"]
        if not isinstance(raw_components, list) or not raw_components:
            raise ApexPreflightRefusal(f"{label}.components must be a non-empty list")
        result = cls(
            case_id=_text(value["case_id"], f"{label}.case_id"),
            c5_ref=_text(value["c5_ref"], f"{label}.c5_ref"),
            c5_artifact_sha256=_sha(
                value["c5_artifact_sha256"], f"{label}.c5_artifact_sha256"),
            binding_kind=_text(value["binding_kind"], f"{label}.binding_kind"),
            semantic_binding_ref=_text(
                value["semantic_binding_ref"], f"{label}.semantic_binding_ref"),
            semantic_binding_sha256=_sha(
                value["semantic_binding_sha256"], f"{label}.semantic_binding_sha256"),
            components=tuple(MappingComponent.from_dict(
                _mapping(item, f"{label}.components[{index}]"),
                f"{label}.components[{index}]")
                for index, item in enumerate(raw_components)),
        )
        expected = CASE_REQUIREMENTS.get(result.case_id)
        if expected is None:
            raise ApexPreflightRefusal(f"{label} names an unselected C3 case")
        if result.c5_ref != expected["c5_ref"]:
            raise ApexPreflightRefusal(f"{label} names the wrong C5 record")
        if result.c5_artifact_sha256 != expected["c5_artifact_sha256"]:
            raise ApexPreflightRefusal(f"{label} names the wrong C5 artifact")
        expected_kind = (_ORDERED_COMPOSITE
                         if result.case_id == "epyc.moe.sparse_expert_dispatch.k175"
                         else _SINGLE_ENTRY)
        if result.binding_kind != expected_kind:
            raise ApexPreflightRefusal(
                f"{result.case_id} requires binding_kind {expected_kind}; a similar "
                "single entry cannot stand in for an ordered composite")
        expected_ids = (_K175_COMPONENT_IDS if expected_kind == _ORDERED_COMPOSITE
                        else _K228_COMPONENT_IDS)
        if tuple(item.component_id for item in result.components) != expected_ids \
                or tuple(item.order for item in result.components) != tuple(range(len(expected_ids))):
            raise ApexPreflightRefusal(
                f"{result.case_id} component identity/order differs from the reviewed contract")
        if expected_kind == _ORDERED_COMPOSITE:
            if tuple(item.branch for item in result.components) != _K175_COMPONENT_BRANCHES \
                    or tuple(item.stream for item in result.components) != _K175_COMPONENT_STREAMS:
                raise ApexPreflightRefusal("k175 branch/stream graph differs from the contract")
            if result.components[0].depends_on or result.components[1].depends_on != (
                    "router_projection",):
                raise ApexPreflightRefusal("k175 graph has invalid router/top8 dependencies")
            if result.components[2].depends_on != ("biased_top8_counts_and_ranks",) \
                    or result.components[3].depends_on != ("biased_top8_counts_and_ranks",) \
                    or result.components[4].depends_on != (
                        "dispatch_n_le_1350", "dispatch_n_gt_1350") \
                    or result.components[5].depends_on != (
                        "biased_top8_counts_and_ranks",) \
                    or result.components[6].depends_on != (
                        "routed_experts", "shared_experts") \
                    or result.components[7].depends_on != (
                        "weighted_undispatch_shared_add",):
                raise ApexPreflightRefusal("k175 component dependencies differ from the contract")
        elif result.components[0].branch != "always" \
                or result.components[0].stream != "main" \
                or result.components[0].depends_on:
            raise ApexPreflightRefusal("k228 must be one unconditional gfx90a trace")
        if len({item.kernel_id for item in result.components}) != len(result.components):
            raise ApexPreflightRefusal("mapping repeats a registry kernel_id")
        return result


@dataclass(frozen=True)
class CaseMappingSet:
    artifact_path: Path
    artifact_sha256: str
    registry_sha256: str
    cases: tuple[CaseMapping, ...]

    def select(self, case_id: str) -> CaseMapping:
        matches = [case for case in self.cases if case.case_id == case_id]
        if len(matches) != 1:
            raise ApexPreflightRefusal(f"mapping does not contain exactly one {case_id} row")
        return matches[0]


def load_case_mapping(path: Path, *, case_id: str | None = None,
                      audit_path: Path = DEFAULT_MAPPING_AUDIT) -> CaseMappingSet:
    """Load the separately reviewed mapping; absence is a typed refusal."""
    path = Path(path)
    if not path.is_file():
        if case_id in CASE_REQUIREMENTS:
            load_mapping_audit(audit_path).select(case_id).refuse()
        raise MissingCaseMapping(
            f"missing {MISSING_MAPPING_ARTIFACT}: it must bind both exact C5 records "
            "to Apex registry rows with a hash-bound semantic-equivalence artifact; "
            "kernel-name similarity is not a mapping")
    document = _read_json(path, "case mapping")
    _exact_keys(
        document,
        {"schema", "apex_revision", "magpie_revision", "registry_sha256", "cases"},
        "case mapping",
    )
    if document["schema"] != MAPPING_SCHEMA:
        raise ApexPreflightRefusal("unsupported C3 Apex mapping schema")
    if document["apex_revision"] != PINNED_APEX_REVISION:
        raise ApexPreflightRefusal("case mapping names the wrong Apex revision")
    if document["magpie_revision"] != PINNED_MAGPIE_REVISION:
        raise ApexPreflightRefusal("case mapping names the wrong Magpie revision")
    rows = document["cases"]
    if not isinstance(rows, list):
        raise ApexPreflightRefusal("case mapping cases must be a list")
    cases = tuple(CaseMapping.from_dict(_mapping(row, f"cases[{index}]"),
                                       f"cases[{index}]")
                  for index, row in enumerate(rows))
    if {case.case_id for case in cases} != set(CASE_REQUIREMENTS) \
            or len(cases) != len(CASE_REQUIREMENTS):
        raise ApexPreflightRefusal("case mapping must bind exactly k228 and k175 once each")
    for case in cases:
        binding_path = Path(case.semantic_binding_ref)
        if not binding_path.is_absolute():
            binding_path = path.parent / binding_path
        _checked_file(binding_path, case.semantic_binding_sha256,
                      f"{case.case_id}.semantic_binding")
        semantic = _read_json(binding_path, f"{case.case_id} semantic review")
        _exact_keys(semantic, {"schema", "authority", "review_outcome", "case_id",
                               "c5_ref", "c5_artifact_sha256", "target_architecture",
                               "binding_kind", "component_order",
                               "tensor_manifest_schema"},
                    f"{case.case_id} semantic review")
        expected_semantic = {
            "schema": SEMANTIC_REVIEW_SCHEMA,
            "authority": "reviewed_static_mapping_only_no_correctness_speedup_or_promotion",
            "review_outcome": "accepted_for_trace_identity",
            "case_id": case.case_id, "c5_ref": case.c5_ref,
            "c5_artifact_sha256": case.c5_artifact_sha256,
            "target_architecture": TARGET_ARCH, "binding_kind": case.binding_kind,
            "component_order": [item.component_id for item in case.components],
            "tensor_manifest_schema": tensor_capture.MANIFEST_SCHEMA,
        }
        drift = [key for key, value in expected_semantic.items()
                 if semantic.get(key) != value]
        if drift:
            raise ApexPreflightRefusal(
                f"{case.case_id} semantic review drifted at {drift}")
        for component in case.components:
            review_path = Path(component.architecture_review_ref)
            if not review_path.is_absolute():
                review_path = path.parent / review_path
            _checked_file(review_path, component.architecture_review_sha256,
                          f"{case.case_id}.{component.component_id}.gfx90a_review")
            review = _read_json(
                review_path, f"{case.case_id}.{component.component_id} gfx90a review")
            _exact_keys(review, {"schema", "authority", "review_outcome", "case_id",
                                 "component_id", "target_architecture", "kernel_id",
                                 "source_repo", "source_commit", "source_file",
                                 "source_file_sha256", "evidence"},
                        f"{case.case_id}.{component.component_id} gfx90a review")
            expected_review = {
                "schema": ARCHITECTURE_REVIEW_SCHEMA,
                "authority": "source_and_gfx90a_compatibility_only_no_runtime_performance",
                "review_outcome": "accepted_for_gfx90a_trace",
                "case_id": case.case_id, "component_id": component.component_id,
                "target_architecture": TARGET_ARCH, "kernel_id": component.kernel_id,
                "source_repo": component.source_repo,
                "source_commit": component.source_commit,
                "source_file": component.source_file,
                "source_file_sha256": component.source_file_sha256,
            }
            review_drift = [key for key, value in expected_review.items()
                            if review.get(key) != value]
            evidence = review.get("evidence")
            if review_drift or not isinstance(evidence, list) or not evidence:
                raise ApexPreflightRefusal(
                    f"{case.case_id}.{component.component_id} gfx90a review is incomplete "
                    f"or drifted at {review_drift}")
            for evidence_index, raw in enumerate(evidence):
                evidence_row = _mapping(
                    raw, f"{case.case_id}.{component.component_id}.evidence[{evidence_index}]")
                _exact_keys(evidence_row, {"ref", "sha256"}, "gfx90a review evidence")
                evidence_path = Path(_text(evidence_row["ref"], "gfx90a evidence.ref"))
                if not evidence_path.is_absolute():
                    evidence_path = review_path.parent / evidence_path
                _checked_file(evidence_path, evidence_row["sha256"],
                              "gfx90a review evidence")
    return CaseMappingSet(
        artifact_path=path.resolve(), artifact_sha256=_sha256_file(path),
        registry_sha256=_sha(document["registry_sha256"], "registry_sha256"),
        cases=cases,
    )


@dataclass(frozen=True)
class SelectedRegistryEntry:
    id: str
    repo: str
    kernel_type: str
    kernel_name: str
    kernel_file: str
    trace_mode: str
    patch_strategy: str
    source_commit: str
    source_file_sha256: str

    def as_apex_dict(self) -> dict[str, str]:
        return {
            "id": self.id, "repo": self.repo, "kernel_type": self.kernel_type,
            "kernel_name": self.kernel_name, "kernel_file": self.kernel_file,
            "trace_mode": self.trace_mode, "patch_strategy": self.patch_strategy,
        }


def select_registry_entry(*, apex_root: Path, mappings: CaseMappingSet,
                          component: MappingComponent) -> SelectedRegistryEntry:
    """Validate one selected row without checking unrelated registry files."""
    apex_root = Path(apex_root).resolve()
    registry = (apex_root / APEX_REGISTRY_RELATIVE).resolve()
    if registry.parent != (apex_root / APEX_REGISTRY_RELATIVE).parent.resolve():
        raise ApexPreflightRefusal("Apex registry path escaped the pinned tree")
    _checked_file(registry, mappings.registry_sha256, "Apex supported-kernel registry")
    try:
        document = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ApexPreflightRefusal(f"cannot parse Apex registry: {exc}") from exc
    document = _mapping(document, "Apex registry")
    if document.get("schema_version") != 1:
        raise ApexPreflightRefusal("unsupported Apex registry schema")
    commits = _mapping(document.get("source_commits"), "Apex source_commits")
    rows = document.get("kernels")
    if not isinstance(rows, list):
        raise ApexPreflightRefusal("Apex registry kernels must be a list")
    selected = [row for row in rows
                if isinstance(row, Mapping) and row.get("id") == component.kernel_id]
    if len(selected) != 1:
        raise ApexPreflightRefusal(
            f"Apex registry must contain exactly one {component.kernel_id} row")
    row = selected[0]
    _exact_keys(row, _ENTRY_FIELDS, f"Apex registry {component.kernel_id}")
    normalized = {field: _text(row[field], f"{component.kernel_id}.{field}")
                  for field in _ENTRY_FIELDS}
    if normalized["repo"] not in _VALID_REPOS:
        raise ApexPreflightRefusal("selected Apex row has an unsupported repo")
    if normalized["kernel_type"] not in _VALID_KERNEL_TYPES:
        raise ApexPreflightRefusal("selected Apex row has an unsupported kernel type")
    if normalized["trace_mode"] not in _VALID_TRACE_MODES:
        raise ApexPreflightRefusal("selected Apex row has an unsupported trace mode")
    if normalized["patch_strategy"] != "static":
        raise ApexPreflightRefusal("selected Apex row is not statically patchable")
    if normalized["repo"] != component.source_repo:
        raise ApexPreflightRefusal("mapping and selected Apex row name different repos")
    if normalized["kernel_file"] != component.source_file:
        raise ApexPreflightRefusal("mapping and selected Apex row name different files")
    source_commit = _commit(commits.get(component.source_repo),
                            f"Apex source_commits.{component.source_repo}")
    if source_commit != component.source_commit:
        raise ApexPreflightRefusal("mapping and Apex registry name different source commits")
    source_path = (apex_root / normalized["kernel_file"]).resolve()
    if not source_path.is_relative_to(apex_root):
        raise ApexPreflightRefusal("selected source file escaped the pinned Apex tree")
    _checked_file(source_path, component.source_file_sha256, "selected Apex source file")
    return SelectedRegistryEntry(
        **normalized, source_commit=source_commit,
        source_file_sha256=component.source_file_sha256,
    )


def validate_pinned_runner_interface(apex_root: Path) -> None:
    """Check the exact ``e06b5d1`` dataclass interface without importing Apex."""
    runner_path = Path(apex_root).resolve() / "pipeline/kernel_tracing/runner.py"
    if not runner_path.is_file():
        raise ApexPreflightRefusal("pinned Apex runner.py is missing")
    try:
        tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise ApexPreflightRefusal(f"cannot parse pinned Apex runner interface: {exc}") from exc
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "TraceKernelConfig"]
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "run_trace_kernel"]
    if len(classes) != 1 or len(functions) != 1:
        raise ApexPreflightRefusal("pinned Apex runner entrypoints are missing or ambiguous")
    annotations = {
        node.target.id: ast.unparse(node.annotation)
        for node in classes[0].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    if set(annotations) != _TRACE_CONFIG_FIELDS:
        raise ApexPreflightRefusal("pinned Apex TraceKernelConfig fields drifted")
    for field in ("results_dir", "kernel_file", "repo_root"):
        if annotations[field] != "Path":
            raise ApexPreflightRefusal(
                f"pinned Apex TraceKernelConfig.{field} is no longer a Path")
    function = functions[0]
    if not function.args.args or function.args.args[0].arg != "config":
        raise ApexPreflightRefusal("pinned Apex run_trace_kernel signature drifted")


@dataclass(frozen=True)
class RepositoryIdentity:
    commit: str
    clean: bool

    def __post_init__(self) -> None:
        _commit(self.commit, "repository commit")
        if not isinstance(self.clean, bool):
            raise ApexPreflightRefusal("repository clean flag must be boolean")


@dataclass(frozen=True)
class ToolchainIdentity:
    torch_version: str
    hip_version: str
    triton_version: str

    def assert_pinned(self) -> None:
        if self.torch_version != PINNED_TORCH_VERSION:
            raise ApexPreflightRefusal(
                f"Torch must be {PINNED_TORCH_VERSION}, observed {self.torch_version}")
        if not self.hip_version.startswith(PINNED_HIP_PREFIX):
            raise ApexPreflightRefusal(
                f"Torch HIP runtime must be ROCm {PINNED_HIP_PREFIX}, "
                f"observed {self.hip_version}")
        if self.triton_version != PINNED_TRITON_VERSION:
            raise ApexPreflightRefusal(
                f"Triton must be {PINNED_TRITON_VERSION}, observed {self.triton_version}")


@dataclass(frozen=True)
class EnvironmentIdentity:
    apex: RepositoryIdentity
    magpie: RepositoryIdentity
    selected_source: RepositoryIdentity
    toolchain: ToolchainIdentity
    physical_agents: tuple[str, ...]
    hsa_override_gfx_version: str | None = None

    def assert_pinned(self, source_commit: str) -> None:
        if self.apex.commit != PINNED_APEX_REVISION or not self.apex.clean:
            raise ApexPreflightRefusal("Apex tree is not clean at the pinned revision")
        if self.magpie.commit != PINNED_MAGPIE_REVISION or not self.magpie.clean:
            raise ApexPreflightRefusal("Magpie tree is not clean at the time-matched revision")
        if self.selected_source.commit != source_commit or not self.selected_source.clean:
            raise ApexPreflightRefusal("selected source repo is not clean at the registry commit")
        self.toolchain.assert_pinned()
        if self.hsa_override_gfx_version:
            raise ApexPreflightRefusal("HSA_OVERRIDE_GFX_VERSION would spoof physical gfx90a")
        agents = {agent for agent in self.physical_agents if agent != "gfx000"}
        if agents != {TARGET_ARCH}:
            raise ApexPreflightRefusal(
                f"physical device must resolve only to {TARGET_ARCH}, observed {sorted(agents)}")


@dataclass(frozen=True)
class WorkloadBinding:
    benchmark_config: Path
    benchmark_config_sha256: str
    model_id: str
    model_manifest: Path
    model_manifest_sha256: str
    tensor_capture_receipt: Path
    tensor_capture_receipt_sha256: str
    results_dir: Path

    def validate(self) -> tuple[str, str, Mapping[str, Any]]:
        config_path = _checked_file(Path(self.benchmark_config), self.benchmark_config_sha256,
                                    "benchmark config")
        manifest_path = _checked_file(Path(self.model_manifest), self.model_manifest_sha256,
                                      "model manifest")
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ApexPreflightRefusal(f"cannot parse benchmark config: {exc}") from exc
        config = _mapping(config, "benchmark config")
        model_id = _text(self.model_id, "model_id")
        if config.get("model") != model_id:
            raise ApexPreflightRefusal("benchmark config and model binding name different models")
        framework = _text(config.get("framework"), "benchmark config.framework")
        manifest = _read_json(manifest_path, "model manifest")
        _exact_keys(manifest, {"schema", "model_path", "files"}, "model manifest")
        if manifest["schema"] != MODEL_MANIFEST_SCHEMA:
            raise ApexPreflightRefusal("unsupported model manifest schema")
        model_path = Path(model_id)
        if not model_path.is_absolute() or not model_path.exists():
            raise ApexPreflightRefusal(
                "benchmark model must be an existing absolute local path")
        model_path = model_path.resolve()
        if manifest["model_path"] != str(model_path):
            raise ApexPreflightRefusal("model manifest and benchmark config name different models")
        rows = manifest["files"]
        if not isinstance(rows, list) or not rows:
            raise ApexPreflightRefusal("model manifest files must be a non-empty list")
        declared: dict[str, str] = {}
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"model manifest.files[{index}]")
            _exact_keys(row, {"path", "sha256"}, f"model manifest.files[{index}]")
            relative = _text(row["path"], f"model manifest.files[{index}].path")
            if relative in declared:
                raise ApexPreflightRefusal("model manifest contains duplicate file paths")
            declared[relative] = _sha(
                row["sha256"], f"model manifest.files[{index}].sha256")
        if model_path.is_file():
            actual = {".": model_path}
        elif model_path.is_dir():
            actual = {
                str(path.relative_to(model_path)): path
                for path in model_path.rglob("*") if path.is_file()
            }
        else:
            raise ApexPreflightRefusal("benchmark model path is not a file or directory")
        if set(declared) != set(actual):
            raise ApexPreflightRefusal("model manifest is not a complete exact file inventory")
        for relative, path in actual.items():
            _checked_file(path, declared[relative], f"model file {relative}")
        model_material = {
            "model_path": str(model_path),
            "files": [{"path": relative, "sha256": declared[relative]}
                      for relative in sorted(declared)],
        }
        model_sha256 = hashlib.sha256(_canonical(model_material).encode()).hexdigest()
        capture_path = _checked_file(
            Path(self.tensor_capture_receipt), self.tensor_capture_receipt_sha256,
            "EPYC tensor capture receipt")
        try:
            capture = tensor_capture.load_capture_receipt(capture_path)
        except tensor_capture.TensorCaptureRefusal as exc:
            raise ApexPreflightRefusal(f"EPYC tensor capture receipt refused: {exc}") from exc
        if capture["model_sha256"] != model_sha256:
            raise ApexPreflightRefusal(
                "tensor capture receipt and benchmark config name different models")
        results = Path(self.results_dir)
        if results.exists() and (not results.is_dir() or any(results.iterdir())):
            raise ApexPreflightRefusal("capture results directory must be absent or empty")
        return framework, model_sha256, capture


@dataclass(frozen=True)
class TraceStep:
    component: MappingComponent
    entry: SelectedRegistryEntry

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component.component_id,
            "order": self.component.order,
            "branch": self.component.branch,
            "stream": self.component.stream,
            "depends_on": list(self.component.depends_on),
            "selected_entry": self.entry.as_apex_dict(),
            "selected_source_commit": self.entry.source_commit,
            "selected_source_file_sha256": self.entry.source_file_sha256,
            "architecture_review_sha256": self.component.architecture_review_sha256,
        }


def _trace_step_rows(steps: Sequence[TraceStep]) -> list[dict[str, Any]]:
    """Project only dependencies that exist in the selected captured branch."""
    active = {step.component.component_id for step in steps}
    rows = []
    seen: set[str] = set()
    for step in steps:
        row = step.to_dict()
        row["depends_on"] = [name for name in row["depends_on"] if name in active]
        if any(name not in seen for name in row["depends_on"]):
            raise ApexPreflightRefusal("selected trace graph has a forward or missing dependency")
        rows.append(row)
        seen.add(step.component.component_id)
    return rows


@dataclass(frozen=True)
class ApexTracePlan:
    case: CaseMapping
    mapping_path: Path
    mapping_sha256: str
    registry_sha256: str
    steps: tuple[TraceStep, ...]
    apex_root: Path
    magpie_root: Path
    python_executable: Path
    workload: WorkloadBinding
    framework: str
    model_sha256: str
    plan_sha256: str
    runner_entrypoint: str = APEX_RUNNER_ENTRYPOINT

    @property
    def entry(self) -> SelectedRegistryEntry:
        if len(self.steps) != 1:
            raise ApexPreflightRefusal(
                "ordered composite has no single entry; use runner_configs")
        return self.steps[0].entry

    def _step_results_dir(self, step: TraceStep) -> Path:
        if len(self.steps) == 1:
            return Path(self.workload.results_dir)
        return Path(self.workload.results_dir) / (
            f"{step.component.order:02d}-{step.component.component_id}")

    def runner_configs(self) -> tuple[dict[str, Any], ...]:
        return tuple({
            "results_dir": self._step_results_dir(step),
            "kernel_name": step.entry.kernel_name,
            "kernel_file": (self.apex_root / step.entry.kernel_file).resolve(),
            "kernel_id": step.entry.id,
            "registry_entry": step.entry.as_apex_dict(),
            "trace_mode": step.entry.trace_mode,
            "kernel_type": step.entry.kernel_type,
            "patch_strategy": step.entry.patch_strategy,
            "benchmark_config": str(self.workload.benchmark_config),
            "run_cmd": "", "framework": self.framework,
            "repo_root": self.apex_root, "dry_run": False,
        } for step in self.steps)

    def runner_config(self) -> dict[str, Any]:
        """Compatibility accessor for the one-step k228 plan only."""
        if len(self.steps) != 1:
            raise ApexPreflightRefusal(
                "ordered composite has multiple configs; use runner_configs")
        return self.runner_configs()[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "case_id": self.case.case_id,
            "c5_ref": self.case.c5_ref,
            "c5_artifact_sha256": self.case.c5_artifact_sha256,
            "mapping_path": str(self.mapping_path),
            "mapping_sha256": self.mapping_sha256,
            "apex_revision": PINNED_APEX_REVISION,
            "magpie_revision": PINNED_MAGPIE_REVISION,
            "registry_sha256": self.registry_sha256,
            "trace_steps": _trace_step_rows(self.steps),
            "binding_kind": self.case.binding_kind,
            "semantic_binding_sha256": self.case.semantic_binding_sha256,
            "toolchain": {
                "torch": PINNED_TORCH_VERSION, "hip": PINNED_HIP_PREFIX,
                "triton": PINNED_TRITON_VERSION,
            },
            "architecture": TARGET_ARCH,
            "benchmark_config": str(self.workload.benchmark_config),
            "benchmark_config_sha256": self.workload.benchmark_config_sha256,
            "model_id": self.workload.model_id,
            "model_manifest": str(self.workload.model_manifest),
            "model_manifest_sha256": self.workload.model_manifest_sha256,
            "model_sha256": self.model_sha256,
            "tensor_capture_receipt": str(self.workload.tensor_capture_receipt),
            "tensor_capture_receipt_sha256": self.workload.tensor_capture_receipt_sha256,
            "results_dir": str(self.workload.results_dir),
            "required_capture_outputs": list(REQUIRED_CAPTURE_OUTPUTS),
            "runner_entrypoint": self.runner_entrypoint,
            "plan_sha256": self.plan_sha256,
        }


def _plan_digest(*, case: CaseMapping, mappings: CaseMappingSet,
                 steps: Sequence[TraceStep], workload: WorkloadBinding,
                 framework: str, model_sha256: str, apex_root: Path,
                 magpie_root: Path, python_executable: Path) -> str:
    material = {
        "case_id": case.case_id,
        "binding_kind": case.binding_kind,
        "mapping_sha256": mappings.artifact_sha256,
        "registry_sha256": mappings.registry_sha256,
        "semantic_binding_sha256": case.semantic_binding_sha256,
        "trace_steps": _trace_step_rows(steps),
        "benchmark_config_sha256": workload.benchmark_config_sha256,
        "model_manifest_sha256": workload.model_manifest_sha256,
        "model_sha256": model_sha256,
        "tensor_capture_receipt_sha256": workload.tensor_capture_receipt_sha256,
        "framework": framework,
        "results_dir": str(workload.results_dir),
        "apex_root": str(apex_root),
        "magpie_root": str(magpie_root),
        "python_executable": str(python_executable),
        "toolchain": [PINNED_TORCH_VERSION, PINNED_HIP_PREFIX, PINNED_TRITON_VERSION],
        "architecture": TARGET_ARCH,
    }
    return hashlib.sha256(_canonical(material).encode()).hexdigest()


def prepare_trace_plan(*, case_id: str, mapping_path: Path, apex_root: Path,
                       magpie_root: Path, python_executable: Path,
                       workload: WorkloadBinding,
                       environment: EnvironmentIdentity | None = None) -> ApexTracePlan:
    """Compile a trace plan after all identity and artifact checks pass."""
    mappings = load_case_mapping(mapping_path, case_id=case_id)
    case = mappings.select(case_id)
    validate_pinned_runner_interface(apex_root)
    all_entries = tuple(select_registry_entry(
        apex_root=apex_root, mappings=mappings, component=component)
        for component in case.components)
    source_identities = {(entry.repo, entry.source_commit) for entry in all_entries}
    if len(source_identities) != 1:
        raise ApexPreflightRefusal(
            "one trace plan requires all mapped components from one pinned source tree")
    source_repo, source_commit = next(iter(source_identities))
    if environment is None:
        environment = probe_environment(
            apex_root=apex_root, magpie_root=magpie_root,
            source_repo=source_repo, python_executable=python_executable)
    environment.assert_pinned(source_commit)
    magpie_root = Path(magpie_root).resolve()
    if not (magpie_root / "Magpie/__main__.py").is_file():
        raise ApexPreflightRefusal("pinned Magpie package entrypoint is missing")
    workload = WorkloadBinding(
        benchmark_config=Path(workload.benchmark_config).resolve(),
        benchmark_config_sha256=workload.benchmark_config_sha256,
        model_id=workload.model_id,
        model_manifest=Path(workload.model_manifest).resolve(),
        model_manifest_sha256=workload.model_manifest_sha256,
        tensor_capture_receipt=Path(workload.tensor_capture_receipt).resolve(),
        tensor_capture_receipt_sha256=workload.tensor_capture_receipt_sha256,
        results_dir=Path(workload.results_dir).resolve(),
    )
    framework, model_sha256, capture = workload.validate()
    if capture["case_id"] != case_id:
        raise ApexPreflightRefusal("tensor capture receipt names a different C3 case")
    active = tuple(
        TraceStep(component, entry)
        for component, entry in zip(case.components, all_entries)
        if component.branch in {"always", capture["dispatch_branch"]})
    if not active:
        raise ApexPreflightRefusal("mapping has no trace steps for the captured branch")
    python_executable = Path(python_executable).resolve()
    if not python_executable.is_file():
        raise ApexPreflightRefusal("pinned Python executable does not exist")
    digest = _plan_digest(
        case=case, mappings=mappings, steps=active, workload=workload,
        framework=framework, model_sha256=model_sha256,
        apex_root=Path(apex_root).resolve(), magpie_root=magpie_root,
        python_executable=python_executable,
    )
    return ApexTracePlan(
        case=case, mapping_path=mappings.artifact_path,
        mapping_sha256=mappings.artifact_sha256,
        registry_sha256=mappings.registry_sha256, steps=active,
        apex_root=Path(apex_root).resolve(), magpie_root=magpie_root,
        python_executable=python_executable, workload=workload,
        framework=framework, model_sha256=model_sha256, plan_sha256=digest,
    )


def _command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(tuple(argv), text=True, capture_output=True, check=False)


def _git_identity(root: Path, run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]]) \
        -> RepositoryIdentity:
    head = run(("git", "-C", str(root), "rev-parse", "HEAD"))
    status = run(("git", "-C", str(root), "status", "--porcelain"))
    if head.returncode != 0 or status.returncode != 0:
        raise ApexPreflightRefusal(f"cannot inspect repository identity at {root}")
    return RepositoryIdentity(_text(head.stdout, f"{root}.commit"), not status.stdout.strip())


def probe_environment(*, apex_root: Path, magpie_root: Path, source_repo: str,
                      python_executable: Path,
                      run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _command,
                      environ: Mapping[str, str] | None = None) -> EnvironmentIdentity:
    """Inspect only git/toolchain/device identity; this launches no workload."""
    apex_root = Path(apex_root).resolve()
    magpie_root = Path(magpie_root).resolve()
    source_root = apex_root / "tools" / "rocm" / source_repo
    probe = (
        "import json, torch, triton; "
        "print(json.dumps({'torch': torch.__version__, 'hip': torch.version.hip, "
        "'triton': triton.__version__}, sort_keys=True))"
    )
    toolchain_result = run((str(python_executable), "-c", probe))
    device_result = run(("/opt/rocm/bin/rocm_agent_enumerator",))
    if toolchain_result.returncode != 0 or device_result.returncode != 0:
        raise ApexPreflightRefusal("toolchain or physical-device identity probe failed")
    try:
        toolchain = json.loads(toolchain_result.stdout)
    except json.JSONDecodeError as exc:
        raise ApexPreflightRefusal("toolchain probe did not emit JSON") from exc
    toolchain = _mapping(toolchain, "toolchain probe")
    return EnvironmentIdentity(
        apex=_git_identity(apex_root, run),
        magpie=_git_identity(magpie_root, run),
        selected_source=_git_identity(source_root, run),
        toolchain=ToolchainIdentity(
            _text(toolchain.get("torch"), "toolchain.torch"),
            _text(toolchain.get("hip"), "toolchain.hip"),
            _text(toolchain.get("triton"), "toolchain.triton"),
        ),
        physical_agents=tuple(line.strip() for line in device_result.stdout.splitlines()
                              if line.strip()),
        hsa_override_gfx_version=(os.environ if environ is None else environ).get(
            "HSA_OVERRIDE_GFX_VERSION"),
    )


def execute_trace(plan: ApexTracePlan, *, authorize_inference: bool = False,
                  runner: Callable[[Mapping[str, Any]], Any] | None = None,
                  runtime_environment: EnvironmentIdentity | None = None) -> Any:
    """Run the pinned trace entrypoint only after explicit inference authorization."""
    if not authorize_inference:
        raise ApexPreflightRefusal("trace execution requires explicit inference authorization")
    if Path(sys.executable).resolve() != plan.python_executable:
        raise ApexPreflightRefusal("current Python differs from the preflighted executable")
    if runner is not None and runtime_environment is None:
        raise ApexPreflightRefusal(
            "an injected runner requires an exact runtime_environment identity")
    reproduced = prepare_trace_plan(
        case_id=plan.case.case_id, mapping_path=plan.mapping_path,
        apex_root=plan.apex_root, magpie_root=plan.magpie_root,
        python_executable=plan.python_executable, workload=plan.workload,
        environment=runtime_environment)
    if reproduced.to_dict() != plan.to_dict():
        raise ApexPreflightRefusal("trace plan identity changed after preflight")
    if runner is not None:
        return tuple(runner(config) for config in plan.runner_configs())
    for root in reversed((plan.apex_root, plan.apex_root / "graders",
                          plan.apex_root / "prompts", plan.apex_root / "pipeline")):
        rendered = str(root)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    module = importlib.import_module("kernel_tracing.runner")
    expected_module = (plan.apex_root / "pipeline/kernel_tracing/runner.py").resolve()
    if Path(module.__file__).resolve() != expected_module:
        raise ApexPreflightRefusal("imported Apex runner did not come from the pinned tree")
    old_magpie = os.environ.get("MAGPIE_ROOT")
    old_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["MAGPIE_ROOT"] = str(plan.magpie_root)
    os.environ["PYTHONPATH"] = (
        f"{plan.magpie_root}:{old_pythonpath}" if old_pythonpath else str(plan.magpie_root)
    )
    try:
        results = []
        for runner_config in plan.runner_configs():
            config = module.TraceKernelConfig(**runner_config)
            results.append(module.run_trace_kernel(config))
        return tuple(results)
    finally:
        if old_magpie is None:
            os.environ.pop("MAGPIE_ROOT", None)
        else:
            os.environ["MAGPIE_ROOT"] = old_magpie
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath


def bind_capture_outputs(plan: ApexTracePlan) -> dict[str, Any]:
    """Hash every ordered trace output emitted by pinned Apex."""
    captures = []
    for step in plan.steps:
        root = plan._step_results_dir(step)
        paths = {name: root / name for name in REQUIRED_CAPTURE_OUTPUTS}
        for name, path in paths.items():
            if not path.is_file():
                raise ApexPreflightRefusal(
                    f"capture output is missing for {step.component.component_id}: {name}")
        trace = _read_json(paths["trace_result.json"], "trace result")
        if trace.get("success") is not True or trace.get("kernel_id") != step.entry.id:
            raise ApexPreflightRefusal("trace result did not succeed for the selected entry")
        if trace.get("registry_entry") != step.entry.as_apex_dict():
            raise ApexPreflightRefusal("trace result registry entry differs from the plan")
        ranges = _read_json(paths["workload_ranges.json"], "workload ranges")
        calls = ranges.get("total_calls")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
            raise ApexPreflightRefusal("capture contains no selected-entry calls")
        patch = _read_json(paths["patched_files/patch_manifest.json"], "patch manifest")
        patched = patch.get("patched_files")
        if not isinstance(patched, list) or not any(
                isinstance(row, Mapping)
                and Path(str(row.get("source_file", ""))).resolve()
                == (plan.apex_root / step.entry.kernel_file).resolve()
                for row in patched):
            raise ApexPreflightRefusal("patch manifest does not bind the selected source file")
        captures.append({
            "component_id": step.component.component_id,
            "order": step.component.order,
            "branch": step.component.branch,
            "stream": step.component.stream,
            "kernel_id": step.entry.id,
            "outputs": {name: {"path": str(path), "sha256": _sha256_file(path)}
                        for name, path in paths.items()},
            "total_calls": calls,
        })
    receipt = {
        "schema": CAPTURE_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "case_id": plan.case.case_id,
        "binding_kind": plan.case.binding_kind,
        "tensor_capture_receipt_sha256": plan.workload.tensor_capture_receipt_sha256,
        "traces": captures,
        "authority": "capture_identity_only_no_correctness_speedup_or_promotion",
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt).encode()).hexdigest()
    return receipt


__all__ = [
    "APEX_RUNNER_ENTRYPOINT", "ARCHITECTURE_REVIEW_SCHEMA", "CAPTURE_SCHEMA",
    "CASE_REQUIREMENTS",
    "DEFAULT_MAPPING_AUDIT", "MAPPING_AUDIT_SCHEMA", "MAPPING_SCHEMA",
    "MISSING_MAPPING_ARTIFACT", "MODEL_MANIFEST_SCHEMA",
    "PINNED_AITER_HSA_FILE_COUNT", "PINNED_AITER_HSA_INVENTORY_SHA256",
    "PINNED_AITER_REVISION", "PINNED_APEX_REGISTRY_SHA256",
    "PINNED_APEX_REVISION", "PINNED_HIP_PREFIX", "PINNED_MAGPIE_REVISION",
    "PINNED_TORCH_VERSION", "PINNED_TRITON_VERSION", "PLAN_SCHEMA",
    "REQUIRED_CAPTURE_OUTPUTS", "ApexPreflightRefusal", "ApexTracePlan",
    "CaseMapping", "CaseMappingSet", "EnvironmentIdentity", "MappingAudit",
    "MappingAuditCase", "MissingCaseMapping", "RepositoryIdentity",
    "MappingComponent", "SelectedRegistryEntry", "StructuralMappingMismatch",
    "SEMANTIC_REVIEW_SCHEMA", "ToolchainIdentity", "TraceStep",
    "WorkloadBinding", "bind_capture_outputs", "execute_trace", "load_case_mapping",
    "load_mapping_audit", "prepare_trace_plan", "probe_environment",
    "select_registry_entry",
    "validate_pinned_runner_interface",
]
