"""Fail-closed Apex trace adapter for the INF-48 C3 EPYC cases.

The two C5 records selected by :mod:`c3_epyc_suite` are operator specifications,
not Apex registry identities.  This module therefore never guesses a registry
entry from a similar-looking name.  A separately reviewed mapping artifact must
bind both C5 records to exact Apex entries and include a hash-bound semantic
equivalence artifact before a trace plan can be produced.

The adapter deliberately resolves and validates only the selected registry row.
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


MAPPING_SCHEMA = "epyc.autokernel.c3_apex_case_mapping.v1"
MODEL_MANIFEST_SCHEMA = "epyc.autokernel.model_identity.v1"
PLAN_SCHEMA = "epyc.autokernel.c3_apex_trace_plan.v1"
CAPTURE_SCHEMA = "epyc.autokernel.c3_apex_capture.v1"

PINNED_APEX_REVISION = c3.PINNED_APEX_REVISION
PINNED_MAGPIE_REVISION = "2a9263833f71755df2a93b466cdd3a9f803fc625"
PINNED_TORCH_VERSION = "2.5.1+rocm6.2"
PINNED_TRITON_VERSION = "3.1.0"
PINNED_HIP_PREFIX = "6.2"
TARGET_ARCH = "gfx90a"

APEX_REGISTRY_RELATIVE = Path("pipeline/kernel_tracing/supported_kernels.yaml")
APEX_RUNNER_ENTRYPOINT = "pipeline.kernel_tracing.runner.run_trace_kernel"
MISSING_MAPPING_ARTIFACT = "c3_apex_case_mapping.v1.json"
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
_SINGLE_ENTRY = "single_registry_entry"
_WHOLE_COMPOSITE_ENTRY = "whole_composite_registry_entry"
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
class CaseMapping:
    case_id: str
    c5_ref: str
    c5_artifact_sha256: str
    kernel_id: str
    source_repo: str
    source_commit: str
    source_file: str
    source_file_sha256: str
    semantic_binding_ref: str
    semantic_binding_sha256: str
    binding_kind: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], label: str) -> "CaseMapping":
        required = {
            "case_id", "c5_ref", "c5_artifact_sha256", "kernel_id", "source_repo",
            "source_commit", "source_file", "source_file_sha256",
            "semantic_binding_ref", "semantic_binding_sha256", "binding_kind",
        }
        _exact_keys(value, required, label)
        result = cls(
            case_id=_text(value["case_id"], f"{label}.case_id"),
            c5_ref=_text(value["c5_ref"], f"{label}.c5_ref"),
            c5_artifact_sha256=_sha(
                value["c5_artifact_sha256"], f"{label}.c5_artifact_sha256"),
            kernel_id=_text(value["kernel_id"], f"{label}.kernel_id"),
            source_repo=_text(value["source_repo"], f"{label}.source_repo"),
            source_commit=_commit(value["source_commit"], f"{label}.source_commit"),
            source_file=_text(value["source_file"], f"{label}.source_file"),
            source_file_sha256=_sha(
                value["source_file_sha256"], f"{label}.source_file_sha256"),
            semantic_binding_ref=_text(
                value["semantic_binding_ref"], f"{label}.semantic_binding_ref"),
            semantic_binding_sha256=_sha(
                value["semantic_binding_sha256"],
                f"{label}.semantic_binding_sha256"),
            binding_kind=_text(value["binding_kind"], f"{label}.binding_kind"),
        )
        expected = CASE_REQUIREMENTS.get(result.case_id)
        if expected is None:
            raise ApexPreflightRefusal(f"{label} names an unselected C3 case")
        if result.c5_ref != expected["c5_ref"]:
            raise ApexPreflightRefusal(f"{label} names the wrong C5 record")
        if result.c5_artifact_sha256 != expected["c5_artifact_sha256"]:
            raise ApexPreflightRefusal(f"{label} names the wrong C5 artifact")
        if result.source_repo not in _VALID_REPOS:
            raise ApexPreflightRefusal(f"{label} names an unsupported Apex source repo")
        expected_kind = (_WHOLE_COMPOSITE_ENTRY
                         if result.case_id == "epyc.moe.sparse_expert_dispatch.k175"
                         else _SINGLE_ENTRY)
        if result.binding_kind != expected_kind:
            if result.case_id == "epyc.moe.sparse_expert_dispatch.k175":
                raise ApexPreflightRefusal(
                    "k175 is composite: one component kernel_id is insufficient; "
                    "use a separately reviewed component-graph/multi-trace extension "
                    "or an audited whole-composite registry entry")
            raise ApexPreflightRefusal(
                f"{label}.binding_kind must be {expected_kind}")
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


def load_case_mapping(path: Path) -> CaseMappingSet:
    """Load the separately reviewed mapping; absence is a typed refusal."""
    path = Path(path)
    if not path.is_file():
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
                          case: CaseMapping) -> SelectedRegistryEntry:
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
                if isinstance(row, Mapping) and row.get("id") == case.kernel_id]
    if len(selected) != 1:
        raise ApexPreflightRefusal(
            f"Apex registry must contain exactly one {case.kernel_id} row")
    row = selected[0]
    _exact_keys(row, _ENTRY_FIELDS, f"Apex registry {case.kernel_id}")
    normalized = {field: _text(row[field], f"{case.kernel_id}.{field}")
                  for field in _ENTRY_FIELDS}
    if normalized["repo"] not in _VALID_REPOS:
        raise ApexPreflightRefusal("selected Apex row has an unsupported repo")
    if normalized["kernel_type"] not in _VALID_KERNEL_TYPES:
        raise ApexPreflightRefusal("selected Apex row has an unsupported kernel type")
    if normalized["trace_mode"] not in _VALID_TRACE_MODES:
        raise ApexPreflightRefusal("selected Apex row has an unsupported trace mode")
    if normalized["patch_strategy"] != "static":
        raise ApexPreflightRefusal("selected Apex row is not statically patchable")
    if normalized["repo"] != case.source_repo:
        raise ApexPreflightRefusal("mapping and selected Apex row name different repos")
    if normalized["kernel_file"] != case.source_file:
        raise ApexPreflightRefusal("mapping and selected Apex row name different files")
    source_commit = _commit(commits.get(case.source_repo),
                            f"Apex source_commits.{case.source_repo}")
    if source_commit != case.source_commit:
        raise ApexPreflightRefusal("mapping and Apex registry name different source commits")
    source_path = (apex_root / normalized["kernel_file"]).resolve()
    if not source_path.is_relative_to(apex_root):
        raise ApexPreflightRefusal("selected source file escaped the pinned Apex tree")
    _checked_file(source_path, case.source_file_sha256, "selected Apex source file")
    return SelectedRegistryEntry(
        **normalized, source_commit=source_commit,
        source_file_sha256=case.source_file_sha256,
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
    results_dir: Path

    def validate(self) -> tuple[str, str]:
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
        results = Path(self.results_dir)
        if results.exists() and (not results.is_dir() or any(results.iterdir())):
            raise ApexPreflightRefusal("capture results directory must be absent or empty")
        return framework, model_sha256


@dataclass(frozen=True)
class ApexTracePlan:
    case: CaseMapping
    mapping_sha256: str
    registry_sha256: str
    entry: SelectedRegistryEntry
    apex_root: Path
    magpie_root: Path
    python_executable: Path
    workload: WorkloadBinding
    framework: str
    model_sha256: str
    plan_sha256: str
    runner_entrypoint: str = APEX_RUNNER_ENTRYPOINT

    def runner_config(self) -> dict[str, Any]:
        return {
            # Apex TraceKernelConfig normalizes these fields with Path methods
            # before doing any work.  Keep them as Paths here; only ``to_dict``
            # is a JSON projection.
            "results_dir": Path(self.workload.results_dir),
            "kernel_name": self.entry.kernel_name,
            "kernel_file": (self.apex_root / self.entry.kernel_file).resolve(),
            "kernel_id": self.entry.id,
            "registry_entry": self.entry.as_apex_dict(),
            "trace_mode": self.entry.trace_mode,
            "kernel_type": self.entry.kernel_type,
            "patch_strategy": self.entry.patch_strategy,
            "benchmark_config": str(self.workload.benchmark_config),
            "run_cmd": "",
            "framework": self.framework,
            "repo_root": self.apex_root,
            "dry_run": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "case_id": self.case.case_id,
            "c5_ref": self.case.c5_ref,
            "c5_artifact_sha256": self.case.c5_artifact_sha256,
            "mapping_sha256": self.mapping_sha256,
            "apex_revision": PINNED_APEX_REVISION,
            "magpie_revision": PINNED_MAGPIE_REVISION,
            "registry_sha256": self.registry_sha256,
            "selected_entry": self.entry.as_apex_dict(),
            "selected_source_commit": self.entry.source_commit,
            "selected_source_file_sha256": self.entry.source_file_sha256,
            "binding_kind": self.case.binding_kind,
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
            "results_dir": str(self.workload.results_dir),
            "required_capture_outputs": list(REQUIRED_CAPTURE_OUTPUTS),
            "runner_entrypoint": self.runner_entrypoint,
            "plan_sha256": self.plan_sha256,
        }


def _plan_digest(*, case: CaseMapping, mappings: CaseMappingSet,
                 entry: SelectedRegistryEntry, workload: WorkloadBinding,
                 framework: str, model_sha256: str, apex_root: Path,
                 magpie_root: Path, python_executable: Path) -> str:
    material = {
        "case_id": case.case_id,
        "binding_kind": case.binding_kind,
        "mapping_sha256": mappings.artifact_sha256,
        "registry_sha256": mappings.registry_sha256,
        "entry": entry.as_apex_dict(),
        "source_commit": entry.source_commit,
        "source_file_sha256": entry.source_file_sha256,
        "benchmark_config_sha256": workload.benchmark_config_sha256,
        "model_manifest_sha256": workload.model_manifest_sha256,
        "model_sha256": model_sha256,
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
    mappings = load_case_mapping(mapping_path)
    case = mappings.select(case_id)
    validate_pinned_runner_interface(apex_root)
    entry = select_registry_entry(apex_root=apex_root, mappings=mappings, case=case)
    if environment is None:
        environment = probe_environment(
            apex_root=apex_root, magpie_root=magpie_root,
            source_repo=entry.repo, python_executable=python_executable)
    environment.assert_pinned(entry.source_commit)
    magpie_root = Path(magpie_root).resolve()
    if not (magpie_root / "Magpie/__main__.py").is_file():
        raise ApexPreflightRefusal("pinned Magpie package entrypoint is missing")
    workload = WorkloadBinding(
        benchmark_config=Path(workload.benchmark_config).resolve(),
        benchmark_config_sha256=workload.benchmark_config_sha256,
        model_id=workload.model_id,
        model_manifest=Path(workload.model_manifest).resolve(),
        model_manifest_sha256=workload.model_manifest_sha256,
        results_dir=Path(workload.results_dir).resolve(),
    )
    framework, model_sha256 = workload.validate()
    python_executable = Path(python_executable).resolve()
    if not python_executable.is_file():
        raise ApexPreflightRefusal("pinned Python executable does not exist")
    digest = _plan_digest(
        case=case, mappings=mappings, entry=entry, workload=workload,
        framework=framework, model_sha256=model_sha256,
        apex_root=Path(apex_root).resolve(), magpie_root=magpie_root,
        python_executable=python_executable,
    )
    return ApexTracePlan(
        case=case, mapping_sha256=mappings.artifact_sha256,
        registry_sha256=mappings.registry_sha256, entry=entry,
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
                  runner: Callable[[Mapping[str, Any]], Any] | None = None) -> Any:
    """Run the pinned trace entrypoint only after explicit inference authorization."""
    if not authorize_inference:
        raise ApexPreflightRefusal("trace execution requires explicit inference authorization")
    if Path(sys.executable).resolve() != plan.python_executable:
        raise ApexPreflightRefusal("current Python differs from the preflighted executable")
    if runner is not None:
        return runner(plan.runner_config())
    for root in reversed((plan.apex_root, plan.apex_root / "graders",
                          plan.apex_root / "prompts", plan.apex_root / "pipeline")):
        rendered = str(root)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    module = importlib.import_module("kernel_tracing.runner")
    expected_module = (plan.apex_root / "pipeline/kernel_tracing/runner.py").resolve()
    if Path(module.__file__).resolve() != expected_module:
        raise ApexPreflightRefusal("imported Apex runner did not come from the pinned tree")
    config = module.TraceKernelConfig(**plan.runner_config())
    old_magpie = os.environ.get("MAGPIE_ROOT")
    old_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["MAGPIE_ROOT"] = str(plan.magpie_root)
    os.environ["PYTHONPATH"] = (
        f"{plan.magpie_root}:{old_pythonpath}" if old_pythonpath else str(plan.magpie_root)
    )
    try:
        return module.run_trace_kernel(config)
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
    """Hash and validate the three capture outputs emitted by pinned Apex."""
    root = Path(plan.workload.results_dir)
    paths = {name: root / name for name in REQUIRED_CAPTURE_OUTPUTS}
    for name, path in paths.items():
        if not path.is_file():
            raise ApexPreflightRefusal(f"capture output is missing: {name}")
    trace = _read_json(paths["trace_result.json"], "trace result")
    if trace.get("success") is not True or trace.get("kernel_id") != plan.entry.id:
        raise ApexPreflightRefusal("trace result did not succeed for the selected entry")
    if trace.get("registry_entry") != plan.entry.as_apex_dict():
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
            == (plan.apex_root / plan.entry.kernel_file).resolve()
            for row in patched):
        raise ApexPreflightRefusal("patch manifest does not bind the selected source file")
    receipt = {
        "schema": CAPTURE_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "case_id": plan.case.case_id,
        "kernel_id": plan.entry.id,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "total_calls": calls,
        "authority": "capture_identity_only_no_correctness_speedup_or_promotion",
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt).encode()).hexdigest()
    return receipt


__all__ = [
    "APEX_RUNNER_ENTRYPOINT", "CAPTURE_SCHEMA", "CASE_REQUIREMENTS",
    "MAPPING_SCHEMA", "MISSING_MAPPING_ARTIFACT", "MODEL_MANIFEST_SCHEMA",
    "PINNED_APEX_REVISION", "PINNED_HIP_PREFIX", "PINNED_MAGPIE_REVISION",
    "PINNED_TORCH_VERSION", "PINNED_TRITON_VERSION", "PLAN_SCHEMA",
    "REQUIRED_CAPTURE_OUTPUTS", "ApexPreflightRefusal", "ApexTracePlan",
    "CaseMapping", "CaseMappingSet", "EnvironmentIdentity", "MissingCaseMapping",
    "RepositoryIdentity", "SelectedRegistryEntry", "ToolchainIdentity",
    "WorkloadBinding", "bind_capture_outputs", "execute_trace", "load_case_mapping",
    "prepare_trace_plan", "probe_environment", "select_registry_entry",
    "validate_pinned_runner_interface",
]
