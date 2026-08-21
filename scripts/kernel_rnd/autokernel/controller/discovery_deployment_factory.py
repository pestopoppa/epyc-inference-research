"""Trusted concrete deployment materializer for governed GPU source discovery.

The JSON configuration merely selects IDs.  This module is the one static
bridge from those IDs to typed, registered Python construction seams.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import argparse
import base64
import binascii
import contextlib
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .. import schemas, source_candidate
from .. import (cumulative_composition, hypothesis_portfolio,
                preauthored_continuation)
from ..execution import inference_window, device_sampler, t0_provider, worktree
from ..execution import cpu_region_claim
from ..execution import instrument_integrity
from ..evaluator import integrity
from ..resource import device_claim
from . import discovery_controller as controller
from . import discovery_deployment as deployment
from . import gpu_source_adapter
from . import gpu_source_evidence as evidence
from . import gpu_load_admission
from . import gpu_residency_sampler
from . import codex_container_actor
from . import claude_fable5_critic_actor
from . import discovery_telemetry
from . import discovery_static_registry
from . import discovery_supervisor
from . import discovery_supervisor_secure
from . import gpu_source_proofs
from scripts.benchmark import autokernel_gpu_discovery_beliefs


class DeploymentFactoryError(RuntimeError): pass
_MI210_KFD_PROCS = Path("/sys/class/kfd/kfd/proc")
_ALLOWED_ENV = frozenset({"PATH", "HOME", "CODEX_HOME", "HTTPS_PROXY", "HTTP_PROXY",
                          "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR"})
_SUPERVISED_BUILD_AUTHORITY: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EnvironmentProfile:
    values: Mapping[str, str]
    def __post_init__(self) -> None:
        if not self.values or any(not isinstance(key, str) or not key or not isinstance(value, str)
                                  for key, value in self.values.items()):
            raise DeploymentFactoryError("environment profile must be an exact string mapping")
        if set(self.values) - _ALLOWED_ENV:
            raise DeploymentFactoryError("environment profile contains a non-allowlisted key")


@dataclass(frozen=True)
class SourceBuilderBinding:
    build: Callable[[controller.PlannedCandidate, Any, Mapping[str, Any]], controller.GpuSourceBuild]

@dataclass(frozen=True)
class EvidencePlanBinding:
    build: Callable[[controller.PlannedCandidate, controller.GpuSourceBuild,
                     "ExperimentTemplate", int], evidence.GpuSourceEvidencePlan]

@dataclass(frozen=True)
class RunnerArgsBinding:
    build: Callable[[controller.PlannedCandidate, controller.GpuSourceBuild, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class ExperimentTemplate:
    """Reviewed intent selector; it owns test/dispatch semantics, never actors."""
    template_id: str
    target_surface: str
    target_symbol: str
    correctness_id: str
    dispatch_id: str
    dispatch: evidence.DispatchContract
    allowed_files: frozenset[str]
    allowed_symbols: Mapping[str, frozenset[str]]
    semantics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.allowed_files or set(self.allowed_symbols) != set(self.allowed_files):
            raise DeploymentFactoryError("experiment template requires exact reviewed files/symbols")
        if any(Path(path).suffix not in _GPU_KERNEL_EXTENSIONS or not symbols
               or "<file-scope>" in symbols for path, symbols in self.allowed_symbols.items()):
            raise DeploymentFactoryError("experiment template includes an unsafe kernel scope")
        object.__setattr__(self, "allowed_symbols", MappingProxyType(
            {path: frozenset(symbols) for path, symbols in self.allowed_symbols.items()}))
        object.__setattr__(self, "semantics", MappingProxyType(
            json.loads(json.dumps(dict(self.semantics), sort_keys=True,
                                  ensure_ascii=False, allow_nan=False))))

    def matches(self, intent: controller.GpuSourceExperimentIntent) -> bool:
        return ((self.template_id, self.target_surface,
                 self.correctness_id, self.dispatch_id) == (
                    intent.template_id, intent.target_surface,
                    intent.correctness_id, intent.dispatch_id)
                and any(intent.target_symbol in symbols
                        for symbols in self.allowed_symbols.values()))

    def bind_dispatch(self, intent: controller.GpuSourceExperimentIntent) -> evidence.DispatchContract:
        """Derive an internal escaped matcher from planner literals and reviewed bounds."""
        if not self.matches(intent):
            raise DeploymentFactoryError("dispatch intent does not select this reviewed template")
        expected_rows = intent.expected_dispatch
        bounds = self.semantics.get("dispatch_bounds", {})
        if not isinstance(bounds, Mapping):
            raise DeploymentFactoryError("template dispatch bounds are malformed")
        markers = bounds.get("kernel_name_fragments")
        if (not isinstance(markers, list) or not markers
                or not all(isinstance(value, str) and value for value in markers)):
            raise DeploymentFactoryError("template kernel literal markers are malformed")
        q5_variants = self.semantics.get("preauthored_q5_candidate_dispatch")
        if q5_variants is not None:
            if (self.template_id != "cuda-mmvq-q5-onewave-continuation-v1"
                    or not isinstance(q5_variants, Mapping)
                    or set(q5_variants) != {row.signature for row in
                                            self.dispatch.anchor_exact[:3]}
                    or len(expected_rows) != 3
                    or len(self.dispatch.anchor_exact) != 4):
                raise DeploymentFactoryError(
                    "preauthored Q5 dispatch variants are outside dedicated authority")
            anchors = {row.signature: row for row in self.dispatch.anchor_exact}
            expected_by_route = {row.route_id: row for row in expected_rows}
            selected_ids = {row.signature for row in self.dispatch.anchor_exact[:3]}
            if set(expected_by_route) != selected_ids:
                raise DeploymentFactoryError(
                    "preauthored Q5 routes differ from exact selected anchor authority")
            candidate: list[evidence.ExactDispatch] = []
            selected: list[evidence.ExactDispatch] = []
            for route_id in sorted(selected_ids):
                anchor = anchors[route_id]
                observed = expected_by_route[route_id]
                if ((observed.calls, observed.grid, observed.workgroup,
                     observed.lds_bytes) !=
                    (anchor.calls, anchor.grid, anchor.workgroup,
                     anchor.lds_bytes)
                        or re.fullmatch(anchor.kernel_pattern,
                                        observed.kernel_name) is None):
                    raise DeploymentFactoryError(
                        "preauthored Q5 anchor route identity changed")
                derived_name = observed.kernel_name.replace(
                    "(ggml_type)6, 1, true, true",
                    "(ggml_type)6, 1, true, false")
                if derived_name == observed.kernel_name:
                    raise DeploymentFactoryError(
                        "preauthored Q5 candidate literal cannot be derived")
                variant = q5_variants[route_id]
                expected_variant = {
                    "kernel_name": derived_name, "calls": anchor.calls,
                    "grid": anchor.grid, "workgroup": 64,
                    "lds_bytes": 0,
                }
                if dict(variant) != expected_variant:
                    raise DeploymentFactoryError(
                        "preauthored Q5 candidate geometry changed")
                candidate.append(evidence.ExactDispatch(
                    signature=f"{route_id}.candidate-onewave",
                    kernel_pattern="^" + re.escape(derived_name) + "$",
                    calls=anchor.calls, grid=anchor.grid, workgroup=64,
                    lds_bytes=0,
                    blocks_per_call=anchor.grid // 64))
                selected.append(anchor)
            tail = self.dispatch.anchor_exact[3]
            candidate_tail_pattern = tail.kernel_pattern.replace(
                "false, true", "false, false")
            return evidence.DispatchContract(
                candidate_exact=tuple(candidate), anchor_exact=tuple(selected),
                candidate_structural_exact=(evidence.ExactDispatch(
                    signature=f"{tail.signature}.candidate-structural-excluded",
                    kernel_pattern=candidate_tail_pattern,
                    calls=tail.calls, grid=tail.grid, workgroup=64,
                    lds_bytes=0,
                    blocks_per_call=tail.grid // 64),),
                anchor_structural_exact=(replace(
                    tail, signature=f"{tail.signature}.anchor-structural-excluded"),),
                candidate_forbidden=(evidence.ForbiddenDispatch(
                    signature=f"{tail.signature}.candidate-stale-tail-forbidden",
                    kernel_pattern=tail.kernel_pattern),),
                invariants=())
        variants = self.semantics.get("candidate_dispatch_variants")
        if variants is not None:
            # The only topology-changing template is the reviewed odd-GQA7
            # pair+tail strategy.  The planner binds the exact observed tile
            # and combine rows, but never authors candidate route geometry.
            # The controller derives 7 = 3*2 + 1 and preserves combine.
            if (self.template_id != "cuda-fattn-gqa7-common-v1"
                    or not isinstance(variants, Mapping)
                    or set(variants) != {"gqa7_bulk_pairs", "gqa7_scalar_tail"}
                    or len(expected_rows) != 2
                    or len(self.dispatch.anchor_exact) != 2):
                raise DeploymentFactoryError(
                    "candidate dispatch variants are outside reviewed GQA7 authority")
            anchors = {row.signature: row for row in self.dispatch.anchor_exact}
            expected_by_route = {row.route_id: row for row in expected_rows}
            if set(expected_by_route) != set(anchors):
                raise DeploymentFactoryError(
                    "GQA7 planner routes differ from reviewed anchor authority")
            for route_id, anchor_row in anchors.items():
                observed = expected_by_route[route_id]
                if ((observed.calls, observed.grid, observed.workgroup,
                     observed.lds_bytes) !=
                    (anchor_row.calls, anchor_row.grid, anchor_row.workgroup,
                     anchor_row.lds_bytes)
                        or re.fullmatch(anchor_row.kernel_pattern,
                                        observed.kernel_name) is None):
                    raise DeploymentFactoryError(
                        "GQA7 planner dispatch differs from reviewed anchor authority")
            tile_routes = [row for row in anchors.values()
                           if "flash_attn_tile" in row.kernel_pattern]
            combine_routes = [row for row in anchors.values()
                              if "flash_attn_combine_results" in row.kernel_pattern]
            if len(tile_routes) != 1 or len(combine_routes) != 1:
                raise DeploymentFactoryError(
                    "GQA7 authority lacks exact tile and combine routes")
            anchor = tile_routes[0]
            expected = expected_by_route[anchor.signature]
            combine = combine_routes[0]
            expected_combine = expected_by_route[combine.signature]
            old = "<64, 64, 2, 1, false>"
            new = "<64, 64, 1, 2, false>"
            if expected.kernel_name.count(old) != 1:
                raise DeploymentFactoryError(
                    "GQA7 anchor kernel literal cannot derive reviewed bulk route")
            bulk_name = expected.kernel_name.replace(old, new)
            derived = {
                "gqa7_bulk_pairs": (bulk_name, anchor.grid * 3 // 7, 2),
                "gqa7_scalar_tail": (expected.kernel_name,
                                      anchor.grid // 7, 1),
            }
            if anchor.grid % 7:
                raise DeploymentFactoryError(
                    "GQA7 anchor grid cannot be partitioned into 3 pairs plus tail")
            candidate = []
            for name in ("gqa7_bulk_pairs", "gqa7_scalar_tail"):
                row = variants[name]
                raw_name, grid, ncols2 = derived[name]
                expected_row = {
                    "kernel_name": raw_name.split("(", 1)[0],
                    "calls": anchor.calls,
                    "grid": grid,
                    "workgroup": anchor.workgroup,
                    "lds_bytes": anchor.lds_bytes,
                    "gqa_ratio": 7,
                    "head_size": 64,
                    "ncols2": ncols2,
                }
                if any(row.get(key) != value
                       for key, value in expected_row.items()):
                    raise DeploymentFactoryError(
                        "GQA7 candidate variant differs from relational authority")
                candidate.append(evidence.ExactDispatch(
                    signature=f"{self.template_id}.candidate.{name}",
                    kernel_pattern="^" + re.escape(raw_name) + "$",
                    calls=anchor.calls, grid=grid,
                    workgroup=anchor.workgroup, lds_bytes=anchor.lds_bytes,
                    blocks_per_call=grid // anchor.workgroup))
            candidate.append(evidence.ExactDispatch(
                signature=f"{self.template_id}.candidate.combine_unchanged",
                kernel_pattern="^" + re.escape(expected_combine.kernel_name) + "$",
                calls=combine.calls, grid=combine.grid,
                workgroup=combine.workgroup, lds_bytes=combine.lds_bytes,
                blocks_per_call=combine.grid // combine.workgroup))
            return evidence.DispatchContract(
                candidate_exact=tuple(candidate),
                anchor_exact=self.dispatch.anchor_exact,
                candidate_forbidden=self.dispatch.candidate_forbidden,
                anchor_forbidden=self.dispatch.anchor_forbidden,
                invariants=self.dispatch.invariants)
        candidate = []
        selected_anchors: list[evidence.ExactDispatch] = []
        for index, expected in enumerate(expected_rows):
            anchor = {row.signature: row for row in self.dispatch.anchor_exact}.get(
                expected.route_id)
            if (anchor is None or (anchor.calls, anchor.grid, anchor.workgroup,
                                   anchor.lds_bytes) != (
                    expected.calls, expected.grid, expected.workgroup,
                    expected.lds_bytes)):
                raise DeploymentFactoryError(
                    "planner dispatch route differs from reviewed anchor authority")
            for key, value in (("calls", expected.calls), ("grid", expected.grid),
                               ("workgroup", expected.workgroup), ("lds_bytes", expected.lds_bytes)):
                limit = bounds.get(key)
                if (not isinstance(limit, list) or len(limit) != 2
                        or not all(isinstance(item, int) for item in limit)
                        or not limit[0] <= value <= limit[1]):
                    raise DeploymentFactoryError(f"planner dispatch {key} exceeds reviewed template bounds")
            if not any(marker in expected.kernel_name for marker in markers):
                raise DeploymentFactoryError("planner kernel literal is outside reviewed template families")
            if expected.grid % expected.workgroup:
                raise DeploymentFactoryError("planner dispatch grid must be an exact workgroup multiple")
            candidate.append(evidence.ExactDispatch(
                signature=f"{expected.route_id}.candidate",
                kernel_pattern="^" + re.escape(expected.kernel_name) + "$",
                calls=expected.calls, grid=expected.grid, workgroup=expected.workgroup,
                lds_bytes=expected.lds_bytes,
                blocks_per_call=expected.grid // expected.workgroup))
            selected_anchors.append(anchor)
        selected_signatures = {row.signature for row in selected_anchors}
        structural_only = tuple(evidence.InvariantDispatch(
            signature=f"{row.signature}.structural",
            kernel_pattern=row.kernel_pattern)
            for row in self.dispatch.anchor_exact
            if row.signature not in selected_signatures)
        return evidence.DispatchContract(candidate_exact=tuple(candidate),
            anchor_exact=tuple(selected_anchors),
            candidate_forbidden=self.dispatch.candidate_forbidden,
            anchor_forbidden=self.dispatch.anchor_forbidden,
            invariants=(*self.dispatch.invariants, *structural_only))


@dataclass(frozen=True)
class ExperimentTemplateRegistry:
    version: str
    registry_sha256: str
    templates: Mapping[str, ExperimentTemplate]

    def __post_init__(self) -> None:
        if (not self.version or not self.templates or set(self.templates) != {
                    template.template_id for template in self.templates.values()
                    if isinstance(template, ExperimentTemplate)}):
            raise DeploymentFactoryError("experiment template registry version/hash is invalid")
        frozen_templates = MappingProxyType(dict(self.templates))
        object.__setattr__(self, "templates", frozen_templates)
        body = {"version": self.version, "templates": {
            key: {"template_id": value.template_id, "target_surface": value.target_surface,
                  "target_symbol": value.target_symbol, "correctness_id": value.correctness_id,
                  "dispatch_id": value.dispatch_id,
                  "allowed_files": sorted(value.allowed_files),
                  "allowed_symbols": {path: sorted(symbols) for path, symbols in value.allowed_symbols.items()},
                  "semantics": dict(value.semantics),
                  "dispatch": {"candidate_exact": [vars(row) for row in value.dispatch.candidate_exact],
                               "anchor_exact": [vars(row) for row in value.dispatch.anchor_exact],
                               "candidate_forbidden": [vars(row) for row in value.dispatch.candidate_forbidden],
                               "anchor_forbidden": [vars(row) for row in value.dispatch.anchor_forbidden],
                               "invariants": [vars(row) for row in value.dispatch.invariants]}}
            for key, value in sorted(self.templates.items())}}
        expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if self.registry_sha256 != expected:
            raise DeploymentFactoryError("experiment template registry content hash mismatch")

    def resolve(self, intent: controller.GpuSourceExperimentIntent | None) -> ExperimentTemplate:
        if intent is None:
            raise DeploymentFactoryError("GPU source candidate lacks a typed experiment intent")
        template = self.templates.get(intent.template_id)
        if not isinstance(template, ExperimentTemplate) or not template.matches(intent):
            raise DeploymentFactoryError("candidate intent is not an exact reviewed experiment template")
        return template

@dataclass(frozen=True)
class InferenceWindowLeaseBinding:
    mode: str = "allowed_discovery_noise"
    def make(self, config: deployment.DiscoveryDeployment, *, claim_journal: Any,
             claim_acquirer: Callable[..., Any],
             claim_verifier: Callable[[Mapping[str, Any]], object]) -> "GpuDiscoveryLease":
        if self.mode != "allowed_discovery_noise":
            raise DeploymentFactoryError("GPU discovery lease may only admit allowed discovery noise")
        return GpuDiscoveryLease(
            config=config, mode=self.mode, claim_journal=claim_journal,
            claim_acquirer=claim_acquirer, claim_verifier=claim_verifier)

@dataclass(frozen=True)
class ProductionSnapshotBinding:
    root: Path
    files: tuple[evidence.BoundInputFile, ...]
    runtime_semantics: Mapping[str, Any]
    runtime_semantics_sha256: str
    def __post_init__(self) -> None:
        if (not self.root.is_absolute() or not self.files
                or not all(isinstance(item, evidence.BoundInputFile) for item in self.files)):
            raise DeploymentFactoryError("production snapshot must contain typed frozen artifacts")
        frozen = json.loads(json.dumps(dict(self.runtime_semantics), sort_keys=True))
        if schemas.content_hash(frozen) != self.runtime_semantics_sha256:
            raise DeploymentFactoryError("production runtime closure semantic hash mismatch")
        object.__setattr__(self, "runtime_semantics", MappingProxyType(frozen))

    def revalidate(self) -> None:
        current = _production_runtime_snapshot(self.root)[1]
        if (current != dict(self.runtime_semantics)
                or schemas.content_hash(current) != self.runtime_semantics_sha256):
            raise DeploymentFactoryError("frozen production runtime closure changed")


@dataclass(frozen=True)
class StaticDeploymentGraph:
    """Fully constructed trusted graph plus its durable validation receipt."""
    config: deployment.DiscoveryDeployment
    adapters: Mapping[str, Any]
    registry_ids: Mapping[str, str]
    graph_receipt: Path
    graph_sha256: str


def _execution_module_sources() -> dict[str, tuple[str, Path]]:
    """Return stable repo-logical names plus this process's physical sources."""
    paths = {
        "deployment_factory": Path(__file__),
        "discovery_controller": Path(controller.__file__),
        "hypotheses": Path(controller.hypotheses.__file__),
        "do_not_repeat": Path(controller.do_not_repeat.__file__),
        "discovery_telemetry": Path(discovery_telemetry.__file__),
        "gpu_discovery_runner": Path(controller.gpu_discovery.__file__),
        "gpu_source_adapter": Path(gpu_source_adapter.__file__),
        "discovery_static_registry": Path(discovery_static_registry.__file__),
        "discovery_supervisor": Path(discovery_supervisor.__file__),
        "discovery_supervisor_secure": Path(discovery_supervisor_secure.__file__),
        "discovery_deployment": Path(deployment.__file__),
        "gpu_load_admission": Path(gpu_load_admission.__file__),
        "split_runtime_verifier": Path(
            controller.gpu_discovery.split_runtime_verifier.__file__),
        "inference_window": Path(inference_window.__file__),
        "cpu_region_claim": Path(cpu_region_claim.__file__),
        "worktree": Path(worktree.__file__),
        "source_candidate": Path(source_candidate.__file__),
        "instrument_integrity": Path(instrument_integrity.__file__),
        "t0_provider": Path(t0_provider.__file__),
        "evaluator_integrity": Path(integrity.__file__),
        "gpu_source_evidence": Path(evidence.__file__),
        "gpu_source_proofs": Path(gpu_source_proofs.__file__),
        "gpu_discovery_beliefs": Path(autokernel_gpu_discovery_beliefs.__file__),
        "device_claim": Path(device_claim.__file__),
        "device_sampler": Path(device_sampler.__file__),
        "gpu_residency_sampler": Path(gpu_residency_sampler.__file__),
        "codex_container_actor": Path(codex_container_actor.__file__),
        "claude_fable5_critic_actor": Path(claude_fable5_critic_actor.__file__),
        "hypothesis_portfolio": Path(hypothesis_portfolio.__file__),
        "preauthored_continuation": Path(preauthored_continuation.__file__),
        "cumulative_composition": Path(cumulative_composition.__file__),
    }
    if set(paths) != set(discovery_supervisor.GRAPH_EXECUTION_MODULES):
        raise DeploymentFactoryError("execution module role closure changed")
    return {
        name: (discovery_supervisor.GRAPH_EXECUTION_MODULES[name], path)
        for name, path in paths.items()
    }


def _runtime_module_file(logical_path: str, path: Path, label: str) -> dict[str, Any]:
    """Bind this process's source object without putting its location in the graph."""
    absolute = Path(os.path.abspath(path))
    try:
        before = absolute.lstat()
    except OSError as exc:
        raise DeploymentFactoryError(f"{label} runtime module path is unavailable") from exc
    if (not stat.S_ISREG(before.st_mode) or absolute.is_symlink()
            or before.st_nlink != 1):
        raise DeploymentFactoryError(
            f"{label} runtime module must be a single-link regular non-symlink file")
    digest = _digest_regular(absolute, label)
    try:
        after = absolute.lstat()
    except OSError as exc:
        raise DeploymentFactoryError(f"{label} runtime module path changed") from exc
    stable = ("st_dev", "st_ino", "st_uid", "st_nlink", "st_mode",
              "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise DeploymentFactoryError(f"{label} runtime module object changed")
    return {
        "logical_path": logical_path,
        "path": str(absolute),
        "sha256": digest,
        "dev": after.st_dev,
        "ino": after.st_ino,
        "uid": after.st_uid,
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": after.st_nlink,
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
    }


def _execution_module_runtime_provenance() -> dict[str, dict[str, Any]]:
    return {
        name: _runtime_module_file(logical_path, path, name)
        for name, (logical_path, path) in _execution_module_sources().items()
    }


def _execution_module_identity() -> dict[str, dict[str, str]]:
    runtime = _execution_module_runtime_provenance()
    return {
        name: {"logical_path": row["logical_path"], "sha256": row["sha256"]}
        for name, row in runtime.items()
    }


def _module_attestor(
        expected: Mapping[str, Mapping[str, str]],
        runtime_provenance: Mapping[str, Mapping[str, Any]] | None = None,
        ) -> Callable[[], None]:
    sealed = json.loads(json.dumps(dict(expected), sort_keys=True))
    sealed_runtime = json.loads(json.dumps(
        dict(runtime_provenance or _execution_module_runtime_provenance()),
        sort_keys=True))
    def attest() -> None:
        if _execution_module_identity() != sealed:
            raise DeploymentFactoryError("live execution module bytes changed after graph validation")
        if _execution_module_runtime_provenance() != sealed_runtime:
            raise DeploymentFactoryError(
                "live execution module runtime provenance changed after graph validation")
    return attest


_STATIC_IDS = MappingProxyType({
    "environment_profile": "sealed-codex",
    "source_builder": "gpu-source-v1",
    "evidence_plan": "reviewed-gpu-source-evidence-v1",
    "runner_args": "qwen05b-tg128",
    "experiment_template_registry": "gpu-source-templates-v3",
    "inference_window_lease": "mi210-window-v1",
    "production_snapshot": "llama-v9-artifacts",
})
_LOAD_PROFILE_ID = "mi210-qwen05b-tg128-fallback-only-v1"
_FALLBACK_ONLY_HEADROOM_SENTINEL = 1 << 60
_ROCPROF_V3_SDK = Path("/mnt/raid0/llm/tools/rocprofiler-sdk-6.2.0-66/opt/rocm-6.2.0")
_ROCPROF_V3 = _ROCPROF_V3_SDK / "bin/rocprofv3"
_ROCPROF_V3_SHA256 = "c753449eb635ecb4d8be794e8b66439b200b252c157555920d260df5cbac767a"
_ROCPROF_V3_PACKAGE = Path(
    "/mnt/raid0/llm/tools/rocprofiler-sdk-6.2.0-66/"
    "rocprofiler-sdk_0.4.0-66~20.04_amd64.deb")
_ROCPROF_V3_PACKAGE_SHA256 = "e22b4f30a45c18b9e90fe1abd032c102e1c706d119084d1ca8a48bcd5a1f7baa"
_ROCPROF_V3_PYTHON = evidence.ROCPROF_V3_PYTHON
_ROCPROF_V3_PYTHON_SHA256 = "efb29ce53d36ebaeee80e3aa44fd6c7f9d71bbded5fe1665240b2ed8ecaeee0e"
_ROCPROF_V3_SDK_LIB = _ROCPROF_V3_SDK / "lib/librocprofiler-sdk.so.0.4.0"
_ROCPROF_V3_SDK_LIB_SHA256 = "44d8548b9e31c7ab4ecad3023878d9b9d8bcf62b69350ba3837c01270d45639c"
_ROCPROF_V3_TOOL_LIB = (
    _ROCPROF_V3_SDK / "lib/rocprofiler-sdk/librocprofiler-sdk-tool.so.0.4.0")
_ROCPROF_V3_TOOL_LIB_SHA256 = "5da10b776a105ab4dc013d5bbee606dd74855494224d4448c77e37dbcaa72670"
_ROCPROF_V3_OLD_LIB = Path(
    "/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0/lib")
_ROCPROF_V3_AQL_LIB = _ROCPROF_V3_OLD_LIB / "libhsa-amd-aqlprofile64.so.1.0.60200"
_ROCPROF_V3_AQL_LIB_SHA256 = "2b984d7f29b4477a80a056e4e343815592c4fa4b23623b8bd406ea04ae6797ed"
_ROCPROF_V3_PCI_LIB_DIR = Path(
    "/mnt/raid0/llm/tools/rocm-profilers-6.2/usr/lib/x86_64-linux-gnu")
_ROCPROF_V3_PCI_LIB = _ROCPROF_V3_PCI_LIB_DIR / "libpciaccess.so.0.11.1"
_ROCPROF_V3_PCI_LIB_SHA256 = "9b83c428c743cd3ce54a03d5eb6bc8879d272c2cee51e0b7094364ac8d8f7c8a"
_ROCPROF_V3_HSA_LIB = Path("/opt/rocm/lib/libhsa-runtime64.so.1.14.60200")
_ROCPROF_V3_HSA_LIB_SHA256 = "013887a0ee59a2a088c2b95875cae3f48d2d661b4a6badcaa0541f116619c068"
_ROCPROF_V3_REGISTER_LIB = Path("/opt/rocm/lib/librocprofiler-register.so.0.4.0")
_ROCPROF_V3_REGISTER_LIB_SHA256 = "4f095b333e6f4cb123f4ba2f59850304f51387a254acba89d457a6ff6a76dfc4"
_CORRECTNESS_SUITE_SEED = 2026081301
_INSTRUMENT_PATH = Path("/mnt/raid0/llm/llama.cpp-experimental")
_INSTRUMENT_BRANCH = "codex/autokernel-gqa7-correctness-instrument-20260818"
_INSTRUMENT_COMMIT = "5bbcc5498e4732162356953b7be96a53073a6706"
_INSTRUMENT_DIFF_SHA256 = "87122b4589d434c4275755640fbe2094d07ae4216315345bac16d68bed9703e0"
_INSTRUMENT_TEST_SOURCE_SHA256 = "7571a536ba1305ad078948de2920aea33f9261ab9bb1b5714e55bd485ff335e9"
_READY_CONTINUE_CONTRACT_SHA256 = "1411f5e81c1b0b3db6952523922c672d88a78aaff5945865c9ccc2b4fc5fd99f"
_INSTRUMENT_BENCH_SOURCE_SHA256 = "b118e62cf452aa351a93f864bf4822d157dfc4af309f97b5f64cb6d1f31d2e07"
_INSTRUMENT_BENCH_README_SHA256 = "6429015fe5025d35b65e6271520ea668267910f82922e618b27b80c909cec33f"
_INSTRUMENT_DIFF_PATHS = frozenset({
    "ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp",
    "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp",
    "ggml/src/ggml-cpu/iqk/iqk_quantize.h",
    "ggml/src/ggml-cpu/iqk/iqk_quantize_min.cpp",
    "tests/CMakeLists.txt",
    "tests/test-autokernel-ready-continue-contract.py",
    "tests/test-backend-ops.cpp",
    "tools/llama-bench/README.md",
    "tools/llama-bench/llama-bench.cpp",
})
_TARGET_SOURCE_SHA256 = MappingProxyType({
    "ggml/src/ggml-cuda/fattn.cu": "f6a61657387c153e88bde036e25684b512c7cf078b1d17c7e3b2d31ee73f28d3",
    "ggml/src/ggml-cuda/fattn-common.cuh": "47537d7980d81f7dc9daa18698f5fdbb990ef6b916fa75fed4bc0bbfd1aa08cb",
    "ggml/src/ggml-cuda/fattn-tile.cu": "f57657daf3c5209a32d182bb888ead02b1e806a26273cfa4df8b0a6345ae8247",
    "ggml/src/ggml-cuda/fattn-tile.cuh": "eaa043031cb9574ec4a0018fc0bea25d3cd7d43230bb23b37e63241e3101d9f0",
    "ggml/src/ggml-cuda/mmvq.cu": "15d25d71c945de19e8efc9fbfc6b7e5e66f33bc7635f9dc648d9e1f231ba409e",
    "ggml/src/ggml-cuda/rope.cu": "8286f7b57bb76ab490e05d42cb8262ad886b85a8fdaaef63d6538b7ff06940b2",
    "ggml/src/ggml-cuda/norm.cu": "37e670ad50f8b0c3fb9acaaba54ad520b143b0e73994de5c10f8635e334ff0cd",
    "ggml/src/ggml-cuda/quantize.cu": "9f0074ec27a46a78c4c4709d00163acc35dae772854c290ba9592574d30bd3d9",
    "ggml/src/ggml-cuda/set-rows.cu": "24654fe55234c12b0d4d1e9c78871509fc5348f7e3ff123e146838c861a8c8a9",
    "ggml/src/ggml-cuda/vecdotq.cuh": "c418082b854a33339b99702b10062132595256478d99f5673a81adf403651eb5",
})
_SAFE_ACTOR_ENVIRONMENT = MappingProxyType({
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/home/node",
    "CODEX_HOME": "/home/node/.codex",
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
})
_SAFE_CRITIC_ENVIRONMENT = MappingProxyType({
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
})
_SITE_MODEL = Path(
    "/mnt/raid0/llm/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/"
    "Qwen2.5-Coder-0.5B-Q4_K_M.gguf")
_PROFILE_TRACE_RECEIPT = Path(
    "/mnt/raid0/llm/autokernel/screens/ak-gpu-qwen05b-tg128-rocprof-attribution-20260813/receipt.json")
_PROFILE_TRACE_RECEIPT_SHA256 = "20742be4a69abf5bb70c228660ff0629bf416ed4452c4f69b9765ef74a933cd8"
_PROFILE_TRACE_CSV = Path(
    "/mnt/raid0/llm/autokernel/screens/ak-gpu-qwen05b-tg128-rocprof-attribution-20260813/timestamps.csv")
_PROFILE_TRACE_CSV_SHA256 = "a11bc20a03dfd5ca157990c1766ebbb5edb70a5c036d73d85d806e4f39a222a8"
_PROFILE_V3_TRACE_CSV = Path(
    "/mnt/raid0/llm/autokernel/diagnostics/"
    "v13-rocprofv3-anchor-tg128-20260819/raw/v13_sdk_kernel_trace.csv")
_PROFILE_V3_TRACE_CSV_SHA256 = "fb818d7b135becc5bfd773c1075cbdea91809d1f5c22ed8d8817560678b03c69"
_PROFILE_V3_AGENT_CSV = _PROFILE_V3_TRACE_CSV.with_name("v13_sdk_agent_info.csv")
_PROFILE_V3_AGENT_CSV_SHA256 = "50189a58f15ffb0008e840a8a6d18db1a88f73e3492b686b167d773de6b9323e"
_PORTFOLIO_SEMANTIC_SHA256 = "7ba7dd1c3c246fb22a247d6e24facb5fbe0eaebec8b2eb21635fde20043e8303"
_PORTFOLIO_FILE_SHA256 = "0fab59286577fbb3d5fa2bf527ce0a0890c2f6fed25f81f6e3693518c1bf8e3f"
_PORTFOLIO_CONTRACT_SHA256 = "96f207733e5fc27a722763cf1b3c542f327eb70d41e04b9948aaec086b3facd4"
_V25_STATE_FILE_SHA256 = "7ce6e5561572390e0a1a31ff8a059be3b68c8cfc809a9233c2e22a8ca730ef3c"
_V25_JOURNAL_FILE_SHA256 = "a715dbbf8a8e089ea9e356339ceaf8f007bf6191ee0ea699d445c1560ddc5b69"
_V25_STATE_SEMANTIC_SHA256 = "9d2d58bfa0d7df68107529c5e29b37c978d53efd78803537eb709ffba37ffd64"
_V25_CANDIDATE_SEMANTICS = frozenset({
    "0e8d73812c098b69c2db1bd606db7e464c1e1a7e70941582355687fb8de493dd",
    "239ff98a74859d16121e728688a1b69cd7cfd0d433a1b7b99a240ba11469636f",
    "294a22d78513b9a01300edc0c5de0cf634fa84ed7b102cdd796cdea062fb7151",
    "29cd306bffd15001a8c81ca580a95ab45edd83a2f3c12b406a091c4c088e6a8a",
    "387ef69bae89a278be40083f2bd5c400a3d813f1d91a0d96c5ee1c8833ff16c5",
    "437767ef970ab3b66e6dd329af1d4f39fe04e3a8f9c8bf4919be2271ad21e4d6",
    "7c9ca0be68b3685f120e672965a3a78806cd93be02b381b8c8d4a079fad7d219",
    "93bf487ffe73b39338d56f32e6d57b320cac70365d10e5dc400c38221d1043d6",
    "a8e514d69a62c2eaf38f6e03056f09787a015b47b884bbe31f3997d4dfe3f980",
    "b04756eb0238eb99b6a614e457c4d334939f8e5a19db76ac3bec5bd4b39d862e",
    "e99d0c0fc7e1edfbc21d83e279f52601580cea313de22f7c3f185ebfc578c62c",
    "eb6af5d3196546b100134f85d5e78f168bda9363dab4940b5b61f62368d36f08",
})
_SITE_WINDOW_LOCK = Path("/mnt/raid0/llm/tmp/model-call.lock")
_SITE_ACTOR_WRAPPER = Path(
    "/usr/local/share/npm-global/lib/node_modules/@openai/codex/bin/codex.js")
_SITE_CRITIC_WRAPPER = Path("/home/node/.local/share/claude/versions/2.1.231")
_SITE_CLAUDE_AUTH_ROOT = Path("/home/node/.claude")
_SITE_GOVERNANCE_ROOT = Path("/workspace")
_FROZEN_CLOSURE_MANIFEST = (
    Path(__file__).with_name("frozen_v9_closure_manifest.json"))
_FROZEN_CLOSURE_MANIFEST_FILE_SHA256 = \
    "625c84a582e6e3e706dee1f1ebc2ae6705e8520e701e2de1db959b4b9d1b3fa3"
_FROZEN_CLOSURE_MANIFEST_SHA256 = \
    "e01ea9392bb8300b4d222581da7bb48b03636883d7c8c8a36a2c3acbe1f17da4"


def _digest_regular(path: Path, label: str) -> str:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_uid", "st_nlink", "st_mode",
                  "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in stable):
            raise OSError("file changed while hashing")
    except OSError as exc:
        raise DeploymentFactoryError(
            f"{label} must be a stable regular non-symlink file") from exc
    finally:
        if "fd" in locals():
            os.close(fd)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _read_private_bound_bytes(path: Path, expected_sha256: str,
                              label: str) -> bytes:
    """Read one bundle-owned carrier through a stable no-follow descriptor."""
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_nlink != 1 or before.st_mode & 0o077):
            raise OSError("carrier identity is not private and single-link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise DeploymentFactoryError(
            f"{label} is not a stable private carrier") from exc
    finally:
        if fd is not None:
            os.close(fd)
    keys = ("st_dev", "st_ino", "st_uid", "st_nlink", "st_mode", "st_size",
            "st_mtime_ns", "st_ctime_ns")
    if (any(getattr(before, key) != getattr(after, key) for key in keys)
            or any(getattr(after, key) != getattr(path_after, key) for key in keys)
            or len(raw) != before.st_size
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise DeploymentFactoryError(f"{label} changed while it was read")
    return raw


def _source_manifest_from_canonical_bytes(
        raw: bytes) -> source_candidate.SourcePatchManifest:
    """Parse one already-bound canonical source-manifest carrier."""
    try:
        payload = json.loads(
            raw.decode("utf-8", "strict"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)))
        required = {
            "schema", "campaign_id", "proposal_id", "candidate_id",
            "source_tree", "production_base_commit", "instrument_commit",
            "change_class", "declared_files", "declared_symbols",
            "mechanism_id", "patch_sha256", "patch_encoding", "patch_base64",
        }
        if (not isinstance(payload, Mapping) or set(payload) != required
                or payload["schema"] != source_candidate.SCHEMA_SOURCE_PATCH
                or payload["patch_encoding"] != "base64"
                or not isinstance(payload["declared_symbols"], Mapping)):
            raise ValueError("source manifest grammar changed")
        patch_bytes = base64.b64decode(payload["patch_base64"], validate=True)
        manifest = source_candidate.SourcePatchManifest(
            campaign_id=payload["campaign_id"],
            proposal_id=payload["proposal_id"],
            candidate_id=payload["candidate_id"],
            source_tree=payload["source_tree"],
            production_base_commit=payload["production_base_commit"],
            instrument_commit=payload["instrument_commit"],
            change_class=payload["change_class"],
            declared_files=tuple(payload["declared_files"]),
            declared_symbols={
                key: tuple(value)
                for key, value in payload["declared_symbols"].items()},
            mechanism_id=payload["mechanism_id"],
            patch_sha256=payload["patch_sha256"],
            patch_bytes=patch_bytes)
    except (UnicodeError, ValueError, TypeError, KeyError, binascii.Error,
            json.JSONDecodeError, source_candidate.SourceCandidateError) as exc:
        raise DeploymentFactoryError(
            "v25 source manifest is malformed") from exc
    if source_candidate.source_patch_manifest_bytes(manifest) != raw:
        raise DeploymentFactoryError(
            "v25 source manifest is not canonical")
    return manifest


def _cross_campaign_manifest_identity(
        manifest: source_candidate.SourcePatchManifest) -> str:
    """Recompute the controller's instrument-epoch-independent identity."""
    return schemas.content_hash({
        "schema": "epyc.autokernel.cross_campaign_candidate_semantics.v1",
        "production_base_commit": manifest.production_base_commit,
        "change_class": manifest.change_class,
        "declared_files": sorted(manifest.declared_files),
        "declared_symbols": {
            key: sorted(value)
            for key, value in sorted(manifest.declared_symbols.items())},
        "mechanism_id": manifest.mechanism_id,
        "patch_sha256": manifest.patch_sha256,
    })


def _candidate_manifest_identity(
        manifest: source_candidate.SourcePatchManifest) -> str:
    """Recompute the controller's exact source semantic identity."""
    return schemas.content_hash({
        "schema": "epyc.autokernel.candidate_source_semantics.v1",
        "source_tree": manifest.source_tree,
        "production_base_commit": manifest.production_base_commit,
        "instrument_commit": manifest.instrument_commit,
        "change_class": manifest.change_class,
        "declared_files": sorted(manifest.declared_files),
        "declared_symbols": {
            key: sorted(value)
            for key, value in sorted(manifest.declared_symbols.items())},
        "mechanism_id": manifest.mechanism_id,
        "patch_sha256": manifest.patch_sha256,
    })


def _derive_v25_carry_forward(
        evidence_rows: Mapping[str, Mapping[str, Any]],
        erratum_path: Path) -> Mapping[str, Any]:
    """Derive successor replay authority plus the sealed v26 Q5 erratum."""
    try:
        state_row = evidence_rows["ev-v25-terminal-state"]
        journal_row = evidence_rows["ev-v25-terminal-journal"]
    except KeyError as exc:
        raise DeploymentFactoryError(
            "successor portfolio lacks v25 terminal carry-forward") from exc
    if (state_row.get("sha256") != _V25_STATE_FILE_SHA256
            or journal_row.get("sha256") != _V25_JOURNAL_FILE_SHA256):
        raise DeploymentFactoryError("v25 carry-forward file identity changed")
    try:
        state_raw = _read_private_bound_bytes(
            Path(state_row["path"]), _V25_STATE_FILE_SHA256,
            "v25 terminal state")
        journal_raw = _read_private_bound_bytes(
            Path(journal_row["path"]), _V25_JOURNAL_FILE_SHA256,
            "v25 terminal journal")
        state = json.loads(state_raw.decode("utf-8"),
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ValueError(value)))
        lines = journal_raw.splitlines(keepends=True)
        events = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(value))) for line in lines]
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DeploymentFactoryError("v25 carry-forward is not strict JSON") from exc
    if (not isinstance(state, Mapping)
            or state.get("schema") != "epyc.autokernel.discovery_controller.v5"
            or state.get("authority") != "nonpromotable_candidate_only_discovery"
            or state.get("complete") is not True
            or state.get("terminal_reason") != "portfolio_exhausted"
            or state.get("next") != 15
            or state.get("scientific_attempts") != 5
            or not isinstance(state.get("deployment_identity_sha256"), str)
            or not controller.HASH.fullmatch(
                state["deployment_identity_sha256"])
            or state.get("state_sha256") != _V25_STATE_SEMANTIC_SHA256
            or hashlib.sha256(json.dumps(
                {key: value for key, value in state.items()
                 if key != "state_sha256"},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
               != _V25_STATE_SEMANTIC_SHA256):
        raise DeploymentFactoryError("v25 terminal state semantics changed")
    suffix = events[-3:]
    expected_suffix = (
        (72, "akj-000000000072-9897f8cf5610",
         "discovery_authoring_refused",
         "c7d6d7153be419346dd2f1c31b457ecd2b893460f02b653b288882b8afd2df1d"),
        (73, "akj-000000000073-420cd88678ed",
         "discovery_portfolio_exhausted",
         "2ba9ea65dc150bd99df9f236ee33b5178b8537e042476bb41952d05760df5b49"),
        (74, "akj-000000000074-b0d0bbb12b4f",
         "discovery_complete", _V25_STATE_SEMANTIC_SHA256),
    )
    observed_suffix = tuple(
        (event.get("seq"), event.get("event_id"),
         (event.get("payload") or {}).get("state"),
         (event.get("payload") or {}).get("controller_state_sha256"))
        for event in suffix if isinstance(event, Mapping))
    if (len(events) != 74 or observed_suffix != expected_suffix
            or any((json.dumps(event, sort_keys=True, separators=(",", ":"))
                    + "\n").encode() != line
                   for event, line in zip(events, lines))):
        raise DeploymentFactoryError("v25 terminal journal chain changed")
    terminals = state.get("portfolio_terminals")
    skips = state.get("portfolio_skips")
    outcomes = {
        "akh-v2-q5-type-specific-dequant": "nominated",
        "akh-v2-q8-quantizer-new-mechanism": "retire",
        "akh-v2-fa-gqa7-pair-tail": "bounded_authoring_skip",
        "akh-v2-rms-direct-load-reduction": "bounded_authoring_skip",
    }
    if (not isinstance(terminals, Mapping) or set(terminals) != set(list(outcomes)[:2])
            or {key: terminals[key].get("disposition") for key in terminals}
               != {key: outcomes[key] for key in list(outcomes)[:2]}
            or not isinstance(skips, Mapping) or set(skips) != set(list(outcomes)[2:])
            or any(skips[key] != {"disposition": "bounded_authoring_skip",
                                  "scientific_terminal": False,
                                  "failure_count": 3}
                   for key in skips)):
        raise DeploymentFactoryError("v25 portfolio outcomes changed")
    iterations = state.get("iterations")
    if not isinstance(iterations, list):
        raise DeploymentFactoryError("v25 iterations are malformed")
    semantics = {row.get("candidate_semantic_sha256") for row in iterations
                 if isinstance(row, Mapping)
                 and row.get("candidate_semantic_sha256") is not None}
    if semantics != _V25_CANDIDATE_SEMANTICS:
        raise DeploymentFactoryError("v25 candidate semantic set changed")
    selected = [row for row in iterations
                if isinstance(row, Mapping)
                and row.get("status") in {
                    "candidate", "inconclusive", "authoring_refused"}]
    if ({row.get("turn") for row in selected} != {2, 4, 6, 8, 12, 13, 14}
            or len(selected) != 7):
        raise DeploymentFactoryError(
            "v25 materialized candidate set changed")
    patch_sha256: set[str] = set()
    cross_campaign_sha256: set[str] = set()
    for row in selected:
        turn = row["turn"]
        evidence_id = f"ev-v25-source-manifest-turn{turn:02d}"
        source_row = evidence_rows.get(evidence_id)
        expected_manifest_sha256 = row.get("source_manifest_sha256")
        if (not isinstance(source_row, Mapping)
                or source_row.get("sha256") != expected_manifest_sha256
                or not isinstance(expected_manifest_sha256, str)
                or not isinstance(source_row.get("path"), str)):
            raise DeploymentFactoryError(
                "v25 source manifest does not join its controller row")
        source_raw = _read_private_bound_bytes(
            Path(source_row["path"]), expected_manifest_sha256,
            f"v25 turn {turn} source manifest")
        manifest = _source_manifest_from_canonical_bytes(source_raw)
        if (manifest.campaign_id !=
                f"ak-discovery-{state['deployment_identity_sha256'][:16]}"
                or manifest.production_base_commit !=
                   deployment.FROZEN_PRODUCTION_HEAD
                or manifest.instrument_commit != _INSTRUMENT_COMMIT
                or _candidate_manifest_identity(manifest) !=
                   row.get("candidate_semantic_sha256")):
            raise DeploymentFactoryError(
                "v25 source manifest semantics differ from controller state")
        patch_sha256.add(manifest.patch_sha256)
        cross_campaign_sha256.add(
            _cross_campaign_manifest_identity(manifest))
    if len(patch_sha256) != 7 or len(cross_campaign_sha256) != 7:
        raise DeploymentFactoryError(
            "v25 source manifests do not derive seven distinct candidates")
    erratum = controller._q5_lds0_attribution_erratum(erratum_path)
    # Import the invalid v26 attempt into the replay set, then let the
    # controller exempt only the exact triple bound by the erratum.  This
    # preserves every prior replay prohibition without treating the bad
    # expectation as science or a DNR result.
    semantics.add(erratum["candidate_semantic_sha256"])
    patch_sha256.add(erratum["candidate_patch_sha256"])
    cross_campaign_sha256.add(erratum["cross_campaign_candidate_sha256"])
    body: dict[str, Any] = {
        "schema": "epyc.autokernel.discovery_carry_forward.v2",
        "predecessor_state_file_sha256": _V25_STATE_FILE_SHA256,
        "predecessor_journal_file_sha256": _V25_JOURNAL_FILE_SHA256,
        "predecessor_state_semantic_sha256": _V25_STATE_SEMANTIC_SHA256,
        "portfolio_outcomes": outcomes,
        "candidate_semantic_sha256": sorted(semantics),
        "candidate_patch_sha256": sorted(patch_sha256),
        "cross_campaign_candidate_sha256": sorted(cross_campaign_sha256),
        "attribution_expectation_erratum": erratum,
    }
    body["carry_forward_sha256"] = schemas.content_hash(body)
    return MappingProxyType(body)


def _v25_carry_forward(
        config: deployment.DiscoveryDeployment) -> Mapping[str, Any]:
    """Re-derive and join the immutable carry-forward carrier exactly."""
    derived = _derive_v25_carry_forward(
        config.hypothesis_evidence_manifest.value["evidence"],
        config.q5_lds0_attribution_erratum.path)
    carrier = config.carry_forward.value
    if (dict(carrier) != dict(derived)
            or config.carry_forward.self_sha256 !=
               derived["carry_forward_sha256"]
            or config.carry_forward.semantic_sha256 !=
               derived["carry_forward_sha256"]):
        raise DeploymentFactoryError(
            "immutable carry-forward carrier differs from derived authority")
    return MappingProxyType(dict(carrier))


def _validate_critic_auth_source() -> Mapping[str, Any]:
    """Validate the fixed private carrier without persisting secret bytes/hashes."""
    claude_fable5_critic_actor._credentials(_SITE_CLAUDE_AUTH_ROOT)
    carrier = _SITE_CLAUDE_AUTH_ROOT / ".credentials.json"
    status = carrier.stat()
    return MappingProxyType({
        "policy": claude_fable5_critic_actor.AUTH_STAGING_POLICY,
        "source_root": str(_SITE_CLAUDE_AUTH_ROOT),
        "carrier": carrier.name,
        "owner_uid": status.st_uid,
        "mode": format(status.st_mode & 0o777, "04o"),
        "secret_digest_persisted": False,
        "validated": True,
    })


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise DeploymentFactoryError(f"bundle artifact already differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_artifact(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    raw = (json.dumps(dict(value), sort_keys=True, indent=2) + "\n").encode()
    _atomic_bytes(path, raw)
    return path.resolve(), hashlib.sha256(raw).hexdigest()


def _preauthored_historical_evidence(
        continuation: preauthored_continuation.PreauthoredContinuation,
        evidence_rows: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """Join the exact old receipt/output/binary bytes as provenance only."""
    required_ids = {"ev-q5-onewave-correctness-binary"}
    for row in continuation.historical_receipts:
        required_ids.update({
            row["receipt_evidence_id"], row["stdout_evidence_id"],
            row["stderr_evidence_id"], row["binary_evidence_id"],
        })
    if not required_ids.issubset(evidence_rows):
        raise DeploymentFactoryError(
            "preauthored historical evidence coverage is incomplete")
    joined = []
    receipt_keys = {
        "binary", "binary_sha256", "campaign_id", "command",
        "cpu_overlap_policy", "device_claim_open", "device_claim_released",
        "ended_at", "exact_case", "exact_case_ok", "exit_code",
        "non_promotable", "promotion_claim", "residency_witness", "result",
        "schema", "source_commit", "started_at", "stderr_sha256",
        "stdout_sha256",
    }

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    for expected in continuation.historical_receipts:
        carriers = {
            kind: evidence_rows[expected[f"{kind}_evidence_id"]]
            for kind in ("receipt", "stdout", "stderr", "binary")
        }
        if (carriers["receipt"].get("sha256") != expected["file_sha256"]
                or carriers["stdout"].get("sha256") != expected["stdout_sha256"]
                or carriers["stderr"].get("sha256") != expected["stderr_sha256"]
                or carriers["binary"].get("sha256") != expected["binary_sha256"]):
            raise DeploymentFactoryError(
                "historical correctness carrier hashes do not join")
        carrier_bytes: dict[str, bytes] = {}
        for kind, carrier in carriers.items():
            if not isinstance(carrier.get("path"), str):
                raise DeploymentFactoryError(
                    "historical correctness evidence path is malformed")
            carrier_bytes[kind] = _read_private_bound_bytes(
                Path(carrier["path"]), carrier["sha256"],
                f"historical Q5 {kind}")
        try:
            receipt = json.loads(
                carrier_bytes["receipt"].decode("utf-8", "strict"),
                object_pairs_hook=strict_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DeploymentFactoryError(
                "historical correctness receipt is unreadable") from exc
        command_tail = (["test", "-o", "MUL_MAT", "-b", "ROCm0", "-p",
                         "q5_0", "-j", "1"]
                        if expected["scope"] == "targeted_q5_0" else
                        ["test", "-o", "MUL_MAT", "-b", "ROCm0", "-j", "1"])
        command = receipt.get("command")
        exact_case = ("type_a=q5_0" if expected["scope"] == "targeted_q5_0"
                      else "MUL_MAT(")
        if (not isinstance(receipt, Mapping) or set(receipt) != receipt_keys
                or not isinstance(command, list) or not command
                or Path(str(command[0])).name != "test-backend-ops"
                or command[1:] != command_tail
                or receipt.get("schema") != expected["schema"]
                or receipt.get("result") != expected["result"]
                or receipt.get("source_commit") != expected["source_commit"]
                or receipt.get("binary_sha256") != expected["binary_sha256"]
                or receipt.get("stdout_sha256") != expected["stdout_sha256"]
                or receipt.get("stderr_sha256") != expected["stderr_sha256"]
                or receipt.get("exit_code") != 0
                or receipt.get("exact_case") != exact_case
                or receipt.get("exact_case_ok") is not True
                or receipt.get("non_promotable") is not True
                or receipt.get("promotion_claim") is not False):
            raise DeploymentFactoryError(
                "historical correctness receipt semantics changed")
        joined.append({
            "scope": expected["scope"],
            "receipt_sha256": carriers["receipt"]["sha256"],
            "stdout_sha256": carriers["stdout"]["sha256"],
            "stderr_sha256": carriers["stderr"]["sha256"],
            "binary_sha256": carriers["binary"]["sha256"],
        })
    body = {
        "schema": "epyc.autokernel.preauthored_historical_evidence.v1",
        "authority": "provenance_only_no_current_correctness_waiver",
        "carrier_sha256": continuation.sha256,
        "receipts": sorted(joined, key=lambda row: row["scope"]),
        "modern_governed_correctness_required": True,
    }
    body["receipt_sha256"] = schemas.content_hash(body)
    return body


def _stable_public_bytes(
        path: Path, expected_sha256: str | None, label: str,
) -> bytes:
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_mode & 0o022):
            raise OSError("not a single-link non-writable regular file")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        linked = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise DeploymentFactoryError(f"{label} is not a stable carrier") from exc
    finally:
        if fd is not None:
            os.close(fd)
    fields = ("st_dev", "st_ino", "st_uid", "st_nlink", "st_mode",
              "st_size", "st_mtime_ns", "st_ctime_ns")
    if (any(getattr(before, key) != getattr(after, key) for key in fields)
            or any(getattr(after, key) != getattr(linked, key) for key in fields)
            or (expected_sha256 is not None
                and hashlib.sha256(raw).hexdigest() != expected_sha256)):
        raise DeploymentFactoryError(f"{label} changed while read")
    return raw


def _frozen_v9_closure_manifest(
        path: Path = _FROZEN_CLOSURE_MANIFEST,
) -> dict[str, Any]:
    raw = _stable_public_bytes(
        path, _FROZEN_CLOSURE_MANIFEST_FILE_SHA256,
        "frozen v9 closure manifest")
    try:
        value = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentFactoryError(
            "frozen v9 closure manifest is invalid") from exc
    canonical = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    production_keys = {
        "branch", "commit", "source_archive_sha256", "version"}
    runtime_keys = {
        "llama_bench_sha256", "llama_server_sha256",
        "hip_library_sha256", "linkage_sha256",
        "runtime_snapshot_sha256"}
    provenance_keys = {"path", "file_sha256", "schema", "required"}
    inventory = value.get("runtime_inventory")
    inventory_rows = (
        inventory.get("objects") if isinstance(inventory, dict) else None)
    readelf = (
        inventory.get("readelf") if isinstance(inventory, dict) else None)
    if (not isinstance(value, dict)
            or set(value) != {
                "schema", "production", "runtime", "runtime_inventory", "provenance",
                "manifest_sha256"}
            or value.get("schema") !=
               "epyc.autokernel.frozen_v9_closure_manifest.v2"
            or raw != canonical
            or not isinstance(value.get("production"), dict)
            or set(value["production"]) != production_keys
            or value["production"].get("branch") !=
               deployment.FROZEN_PRODUCTION_BRANCH
            or value["production"].get("commit") !=
               deployment.FROZEN_PRODUCTION_HEAD
            or value["production"].get("version") !=
               "version: 10125 (0db32c06e)"
            or not isinstance(value.get("runtime"), dict)
            or set(value["runtime"]) != runtime_keys
            or not isinstance(inventory, dict)
            or set(inventory) != {"schema", "readelf", "objects"}
            or inventory.get("schema") !=
               "epyc.autokernel.runtime_inventory.v1"
            or not isinstance(readelf, dict)
            or set(readelf) != {"path", "sha256", "version"}
            or readelf.get("path") !=
               "/usr/bin/x86_64-linux-gnu-readelf"
            or not isinstance(readelf.get("version"), str)
            or not readelf["version"]
            or re.fullmatch(r"[0-9a-f]{64}", str(readelf.get("sha256")))
               is None
            or not isinstance(inventory_rows, list)
            or not inventory_rows
            or any(not isinstance(row, dict)
                   or set(row) != {
                       "relative_path", "sha256", "symlink_target"}
                   or not isinstance(row.get("relative_path"), str)
                   or PurePosixPath(row["relative_path"]).is_absolute()
                   or row["relative_path"] !=
                      PurePosixPath(row["relative_path"]).as_posix()
                   or any(part in {"", ".", ".."} for part in
                          PurePosixPath(row["relative_path"]).parts)
                   or PurePosixPath(row["relative_path"]).parts[:2]
                      not in {("build", "bin"), ("build-hip", "bin")}
                   or re.fullmatch(r"[0-9a-f]{64}",
                                   str(row.get("sha256"))) is None
                   or (row.get("symlink_target") is not None
                       and (not isinstance(row["symlink_target"], str)
                            or PurePosixPath(row["symlink_target"]).name !=
                               row["symlink_target"]))
                   for row in inventory_rows)
            or inventory_rows != sorted(
                inventory_rows, key=lambda row: row["relative_path"])
            or len({row["relative_path"] for row in inventory_rows}) !=
               len(inventory_rows)
            or not isinstance(value.get("provenance"), dict)
            or set(value["provenance"]) != {
                "build", "linkage", "runtime", "measurement"}
            or any(not isinstance(row, dict) or set(row) != provenance_keys
                   or not isinstance(row.get("path"), str)
                   or not row["path"] or Path(row["path"]).is_absolute()
                   or not isinstance(row.get("schema"), str)
                   or not row["schema"]
                   or not isinstance(row.get("required"), dict)
                   or not row["required"]
                   for row in value["provenance"].values())
            or any(re.fullmatch(r"[0-9a-f]{64}", str(item)) is None
                   for item in (
                       value["production"]["source_archive_sha256"],
                       *value["runtime"].values(),
                       *(row["file_sha256"]
                         for row in value["provenance"].values()),
                       value["manifest_sha256"]))
            or value["manifest_sha256"] !=
               _FROZEN_CLOSURE_MANIFEST_SHA256
            or value["manifest_sha256"] != schemas.content_hash({
                key: item for key, item in value.items()
                if key != "manifest_sha256"})):
        raise DeploymentFactoryError(
            "frozen v9 closure manifest authority changed")
    return value


def _required_projection_matches(
        observed: object, expected: object,
) -> bool:
    if isinstance(expected, dict):
        return bool(
            isinstance(observed, dict)
            and all(key in observed
                    and _required_projection_matches(observed[key], value)
                    for key, value in expected.items()))
    return type(observed) is type(expected) and observed == expected


def _verify_frozen_v9_provenance(
        manifest: Mapping[str, Any], governance_root: Path,
) -> dict[str, str]:
    result = {}
    for role, authority in manifest["provenance"].items():
        raw = _stable_public_bytes(
            governance_root / authority["path"], authority["file_sha256"],
            f"frozen production {role} provenance")
        try:
            body = json.loads(raw.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeploymentFactoryError(
                f"frozen production {role} provenance is invalid") from exc
        if (not isinstance(body, dict)
                or body.get("schema") != authority["schema"]
                or not _required_projection_matches(
                    body, authority["required"])):
            raise DeploymentFactoryError(
                f"frozen production {role} provenance semantics changed")
        result[role] = authority["file_sha256"]
    return result


def _verify_frozen_v9_runtime_closure(
        manifest: Mapping[str, Any], production_path: Path,
        runtime_semantics: Mapping[str, Any],
) -> None:
    _verify_expected_runtime_inventory(
        production_path, manifest["runtime_inventory"])
    build = production_path / "build-hip"
    try:
        observed = {
            "llama_bench_sha256": _digest_regular(
                build / "bin/llama-bench", "frozen production binary"),
            "llama_server_sha256": _digest_regular(
                build / "bin/llama-server", "frozen production server"),
            "hip_library_sha256": _digest_regular(
                (build / "bin/libggml-hip.so").resolve(strict=True),
                "frozen production HIP library"),
            "linkage_sha256":
                discovery_static_registry._linkage_sha(build),
            "runtime_snapshot_sha256":
                schemas.content_hash(dict(runtime_semantics)),
        }
    except (OSError, discovery_static_registry.StaticRegistryError) as exc:
        raise DeploymentFactoryError(
            "frozen production runtime closure is unavailable") from exc
    if observed != manifest["runtime"]:
        raise DeploymentFactoryError(
            "frozen production runtime closure differs from manifest")


def _frozen_source_archive_sha256(production_path: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(production_path), "archive",
         deployment.FROZEN_PRODUCTION_HEAD),
        check=True, stdin=subprocess.DEVNULL, capture_output=True)
    entries = []
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if member.issym():
                mode, raw = "120000", member.linkname.encode()
            elif member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise DeploymentFactoryError("frozen source archive is incomplete")
                mode, raw = ("100755" if member.mode & 0o111 else "100644"), stream.read()
            else:
                raise DeploymentFactoryError("frozen source archive has unsupported entry")
            entries.append((member.name, mode, hashlib.sha256(raw).hexdigest()))
    manifest = "".join(
        f"{mode}\t{digest}\t{name}\n"
        for name, mode, digest in sorted(entries)).encode()
    return hashlib.sha256(manifest).hexdigest()


def _verify_frozen_v9_source_identity(
        manifest: Mapping[str, Any], production_path: Path,
) -> None:
    try:
        head = subprocess.run(
            ("git", "-C", str(production_path), "rev-parse", "HEAD"),
            check=True, stdin=subprocess.DEVNULL, capture_output=True,
            text=True).stdout.strip()
        branch = subprocess.run(
            ("git", "-C", str(production_path), "symbolic-ref", "--short",
             "HEAD"), check=True, stdin=subprocess.DEVNULL,
            capture_output=True, text=True).stdout.strip()
        source_archive_sha256 = _frozen_source_archive_sha256(production_path)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeploymentFactoryError(
            "frozen v9 source identity observation is unavailable") from exc
    production = manifest["production"]
    if (head != production["commit"]
            or branch != production["branch"]
            or source_archive_sha256 != production["source_archive_sha256"]):
        raise DeploymentFactoryError(
            "frozen v9 source identity differs from manifest")


def _authenticate_frozen_v9_closure(
        manifest: Mapping[str, Any], production_path: Path,
        runtime_semantics: Mapping[str, Any], governance_root: Path,
) -> dict[str, str]:
    """Authenticate the closure without executing any production artifact."""
    _verify_frozen_v9_source_identity(manifest, production_path)
    provenance = _verify_frozen_v9_provenance(manifest, governance_root)
    _verify_frozen_v9_runtime_closure(
        manifest, production_path, runtime_semantics)
    return provenance


def _deployment_workload_body() -> dict[str, Any]:
    return {"schema": "epyc.autokernel.discovery_workload.v1",
            "workload": "decode_tg128", "prompt_tokens": 0,
            "generation_tokens": 128, "calls_per_arm": 9,
            "device_id": "mi210_0", "promotion_claim": False}


def _deployment_runtime_body() -> dict[str, Any]:
    return {"schema": "epyc.autokernel.discovery_runtime.v1",
            "architecture": "gfx90a", "gpu_layers": 99,
            "flash_attention": True, "hip_graphs": True,
            "cpu_list": "184-191", "threads": 8,
            "promotion_claim": False}


def _pretty_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(dict(value), sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()


def _load_frozen_production_comparator(
        path: Path) -> cumulative_composition.FrozenProductionComparator:
    try:
        carrier = _stable_public_bytes(
            path, None, "frozen production comparator")
        raw = json.loads(carrier.decode("utf-8", "strict"))
        comparator = cumulative_composition.FrozenProductionComparator.from_dict(raw)
        canonical = (json.dumps(
            comparator.to_dict(), sort_keys=True, indent=2) + "\n").encode()
        if carrier != canonical:
            raise DeploymentFactoryError(
                "frozen production comparator is not canonical JSON")
        return comparator
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            cumulative_composition.CompositionError) as exc:
        raise DeploymentFactoryError(
            "frozen production comparator receipt is invalid") from exc


def _verify_frozen_production_comparator(
        comparator: cumulative_composition.FrozenProductionComparator,
        production_path: Path, runtime_semantics: Mapping[str, Any],
        *, model_path: Path, workload_sha256: str,
        runtime_config_sha256: str,
        governance_root: Path = _SITE_GOVERNANCE_ROOT,
) -> None:
    """Join the static comparator to the exact live frozen-v9 closure.

    ``source_sha256`` is the integrity-tree digest of ``git archive`` for the
    exact frozen commit.  It intentionally does not hash the serving worktree,
    whose ignored operator tooling is outside the Git/source-build authority.
    The comparator's build/linkage/runtime receipt digests are reopened against
    the reviewed closure manifest and their required ratification semantics.
    Promotion authority also reobserves the exact source, executable, linkage
    and runtime closure instead of trusting current-disk values by themselves.
    """
    manifest = _frozen_v9_closure_manifest()
    _inventory_files, observed_runtime_semantics = \
        _production_runtime_snapshot(
            production_path, closure_manifest=manifest)
    if dict(runtime_semantics) != observed_runtime_semantics:
        raise DeploymentFactoryError(
            "frozen production comparator topology observation changed")
    runtime_semantics = observed_runtime_semantics
    provenance = _authenticate_frozen_v9_closure(
        manifest, production_path, runtime_semantics, governance_root)
    if comparator.runtime_snapshot_sha256 != \
            manifest["runtime"]["runtime_snapshot_sha256"]:
        raise DeploymentFactoryError(
            "frozen production comparator runtime snapshot changed")
    identity = comparator.build_identity
    if (identity.source_commit != deployment.FROZEN_PRODUCTION_HEAD
            or identity.source_sha256 !=
               manifest["production"]["source_archive_sha256"]
            or identity.binary_sha256 !=
               manifest["runtime"]["llama_bench_sha256"]
            or identity.hip_library_sha256 !=
               manifest["runtime"]["hip_library_sha256"]
            or identity.linkage_sha256 !=
               manifest["runtime"]["linkage_sha256"]
            or identity.config_sha256 != comparator.build_receipt_sha256
            or identity.config_sha256 != provenance["build"]):
        raise DeploymentFactoryError(
            "frozen production comparator build/linkage identity changed")
    protocol = cumulative_composition.frozen_production_protocol_binding(
        model_sha256=comparator.model_sha256,
        build_identity=identity)
    model_sha = _digest_regular(model_path, "frozen comparator model")
    expected_receipts = tuple(
        provenance[role]
        for role in ("build", "linkage", "runtime", "measurement"))
    if ((comparator.build_receipt_sha256,
         comparator.linkage_receipt_sha256,
         comparator.runtime_receipt_sha256,
         comparator.measurement_receipt_sha256) != expected_receipts
            or comparator.workload_sha256 != workload_sha256
            or comparator.runtime_config_sha256 != runtime_config_sha256
            or comparator.model_sha256 != model_sha
            or comparator.measurement_protocol_sha256 !=
               protocol["measurement_protocol_sha256"]
            or comparator.observed_workload_sha256 !=
               protocol["observed_workload_sha256"]
            or comparator.observed_runtime_config_sha256 !=
               protocol["observed_runtime_config_sha256"]
            or comparator.frame_sha256 != protocol["frame_sha256"]):
        raise DeploymentFactoryError(
            "frozen production comparator provenance/protocol changed")


def derive_frozen_production_comparator(
        *, production_path: Path = deployment.FROZEN_PRODUCTION_PATH,
        model_path: Path = _SITE_MODEL,
        governance_root: Path = _SITE_GOVERNANCE_ROOT,
) -> cumulative_composition.FrozenProductionComparator:
    """Derive the canonical comparator from real frozen-v9 authority only."""
    manifest = _frozen_v9_closure_manifest()
    snapshot_files, runtime_semantics = _production_runtime_snapshot(
        production_path, closure_manifest=manifest)
    del snapshot_files
    provenance = _authenticate_frozen_v9_closure(
        manifest, production_path, runtime_semantics, governance_root)
    runtime_snapshot_sha = manifest["runtime"]["runtime_snapshot_sha256"]
    identity = gpu_source_proofs.BuildIdentity(
        source_commit=deployment.FROZEN_PRODUCTION_HEAD,
        source_sha256=manifest["production"]["source_archive_sha256"],
        binary_sha256=manifest["runtime"]["llama_bench_sha256"],
        hip_library_sha256=manifest["runtime"]["hip_library_sha256"],
        config_sha256=provenance["build"],
        linkage_sha256=manifest["runtime"]["linkage_sha256"])
    model_sha = _digest_regular(model_path, "frozen comparator model")
    workload_sha = _pretty_json_sha256(_deployment_workload_body())
    runtime_sha = _pretty_json_sha256(_deployment_runtime_body())
    protocol = cumulative_composition.frozen_production_protocol_binding(
        model_sha256=model_sha,
        build_identity=identity)
    comparator = cumulative_composition.FrozenProductionComparator.create(
        build_identity=identity,
        build_receipt_sha256=provenance["build"],
        linkage_receipt_sha256=provenance["linkage"],
        runtime_receipt_sha256=provenance["runtime"],
        runtime_snapshot_sha256=runtime_snapshot_sha,
        measurement_receipt_sha256=provenance["measurement"],
        model_sha256=model_sha, workload_sha256=workload_sha,
        runtime_config_sha256=runtime_sha,
        observed_workload_sha256=protocol["observed_workload_sha256"],
        observed_runtime_config_sha256=
            protocol["observed_runtime_config_sha256"],
        frame_sha256=protocol["frame_sha256"],
        measurement_protocol_sha256=
            protocol["measurement_protocol_sha256"])
    _verify_frozen_production_comparator(
        comparator, production_path, runtime_semantics,
        model_path=model_path.resolve(), workload_sha256=workload_sha,
        runtime_config_sha256=runtime_sha, governance_root=governance_root)
    return comparator


def seal_frozen_production_comparator(
        output: Path, **kwargs: Any,
) -> cumulative_composition.FrozenProductionComparator:
    _validate_comparator_output_path(output)
    comparator = derive_frozen_production_comparator(**kwargs)
    raw = (json.dumps(
        comparator.to_dict(), sort_keys=True, indent=2) + "\n").encode()
    if output.exists() or output.is_symlink():
        reopened = _load_frozen_production_comparator(output)
        if reopened != comparator:
            raise DeploymentFactoryError(
                "existing comparator differs from frozen authority")
        return reopened
    _atomic_bytes(output, raw)
    output.chmod(0o400)
    reopened = _load_frozen_production_comparator(output)
    if reopened != comparator:
        raise DeploymentFactoryError("sealed comparator did not reopen exactly")
    return reopened


def _validate_comparator_output_path(output: Path) -> None:
    if not output.is_absolute() or output != Path(os.path.abspath(output)):
        raise DeploymentFactoryError(
            "comparator output must be an absolute normalized path")
    if ".git" in output.parts:
        raise DeploymentFactoryError(
            "comparator output cannot be inside git metadata")
    production = deployment.FROZEN_PRODUCTION_PATH.resolve(strict=True)
    try:
        output.relative_to(production)
    except ValueError:
        pass
    else:
        raise DeploymentFactoryError(
            "comparator output cannot modify frozen production")
    current = Path(output.anchor)
    for component in output.parts[1:]:
        current /= component
        if not current.exists() and not current.is_symlink():
            continue
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise DeploymentFactoryError(
                    "comparator output cannot traverse an alias")
        except OSError as exc:
            raise DeploymentFactoryError(
                "comparator output ancestry is unavailable") from exc


def initialize_static_deployment_bundle(
        root: Path, *, frozen_production_comparator: Path) -> Path:
    """Emit the one reviewed site bundle; no caller supplies code or argv authority."""
    if not root.is_absolute() or root.is_symlink() or ".." in root.parts:
        raise DeploymentFactoryError("bundle root must be an absolute non-symlink path")
    root.mkdir(parents=True, exist_ok=True)
    comparator = _load_frozen_production_comparator(
        frozen_production_comparator)
    config_dir = root / "config"
    for directory in (config_dir, root / "locks", root / "portfolio-evidence"):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
    try:
        portfolio = hypothesis_portfolio.load(hypothesis_portfolio.DEFAULT_PORTFOLIO)
        hypothesis_portfolio.verify_evidence_files(portfolio)
    except hypothesis_portfolio.PortfolioError as exc:
        raise DeploymentFactoryError("checked-in hypothesis portfolio is not deployable") from exc
    if (portfolio.sha256 != _PORTFOLIO_SEMANTIC_SHA256
            or _digest_regular(hypothesis_portfolio.DEFAULT_PORTFOLIO,
                               "checked-in hypothesis portfolio")
            != _PORTFOLIO_FILE_SHA256):
        raise DeploymentFactoryError("checked-in hypothesis portfolio identity changed")
    portfolio_path = config_dir / "discovery-hypothesis-portfolio-v26.json"
    _atomic_bytes(portfolio_path, hypothesis_portfolio.DEFAULT_PORTFOLIO.read_bytes())
    portfolio_file_sha = _digest_regular(portfolio_path, "hypothesis portfolio")
    contract_source = hypothesis_portfolio.DEFAULT_PORTFOLIO.with_name(
        "HYPOTHESIS_PORTFOLIO_V2.md")
    contract_path = config_dir / "HYPOTHESIS_PORTFOLIO_V2.md"
    _atomic_bytes(contract_path, contract_source.read_bytes())
    contract_sha = _digest_regular(contract_path, "hypothesis portfolio contract")
    if contract_sha != _PORTFOLIO_CONTRACT_SHA256:
        raise DeploymentFactoryError("hypothesis portfolio contract identity changed")
    try:
        continuation = preauthored_continuation.load()
        preauthored_continuation.verify_git_authority(
            continuation, _INSTRUMENT_PATH, _INSTRUMENT_COMMIT)
    except preauthored_continuation.PreauthoredContinuationError as exc:
        raise DeploymentFactoryError(
            "checked-in preauthored continuation is not deployable") from exc
    continuation_path = config_dir / "preauthored-q5-continuation-v1.json"
    _atomic_bytes(
        continuation_path, preauthored_continuation.DEFAULT_CARRIER.read_bytes())
    continuation_path.chmod(0o400)
    continuation_file_sha = _digest_regular(
        continuation_path, "preauthored continuation")
    erratum = controller._q5_lds0_attribution_erratum()
    erratum_path = config_dir / "q5-lds0-attribution-erratum-v1.json"
    _atomic_bytes(
        erratum_path, controller._Q5_LDS0_ERRATUM_CARRIER.read_bytes())
    erratum_path.chmod(0o400)
    erratum_file_sha = _digest_regular(
        erratum_path, "Q5 LDS0 attribution erratum")
    if (erratum_file_sha != controller._Q5_LDS0_ERRATUM_FILE_SHA256
            or controller._q5_lds0_attribution_erratum(erratum_path) != erratum):
        raise DeploymentFactoryError(
            "vendored Q5 LDS0 attribution erratum changed")
    vendored_evidence: dict[str, dict[str, str]] = {}
    for row in portfolio.body["evidence"]:
        evidence_id = row["evidence_id"]
        target = root / "portfolio-evidence" / f"{evidence_id}.bin"
        raw = hypothesis_portfolio.read_evidence_bytes(
            row["path"], f"portfolio evidence {evidence_id}")
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise DeploymentFactoryError(f"portfolio evidence changed: {evidence_id}")
        _atomic_bytes(target, raw)
        target.chmod(0o400)
        vendored_evidence[evidence_id] = {
            "path": str(target.resolve()), "sha256": row["sha256"]}
    historical_q5_evidence = _preauthored_historical_evidence(
        continuation, vendored_evidence)
    evidence_manifest = {
        "schema": deployment.EVIDENCE_MANIFEST_SCHEMA,
        "portfolio_sha256": portfolio.sha256,
        "evidence": vendored_evidence,
    }
    evidence_manifest["manifest_sha256"] = schemas.content_hash(evidence_manifest)
    evidence_manifest_path, evidence_manifest_file_sha = _json_artifact(
        config_dir / "hypothesis-evidence-manifest.json", evidence_manifest)
    carry_forward = _derive_v25_carry_forward(
        evidence_manifest["evidence"], erratum_path)
    carry_forward_path = config_dir / "discovery-carry-forward-v2.json"
    carry_forward_bytes = (
        json.dumps(dict(carry_forward), sort_keys=True,
                   separators=(",", ":")) + "\n").encode()
    _atomic_bytes(carry_forward_path, carry_forward_bytes)
    carry_forward_path.chmod(0o400)
    carry_forward_file_sha = hashlib.sha256(carry_forward_bytes).hexdigest()
    carry_forward_self_sha = carry_forward["carry_forward_sha256"]
    model = _bound(_SITE_MODEL, "model")
    source_plan = _bound(
        Path(vendored_evidence["ev-source-plan-v1"]["path"]),
        "vendored_reviewed_source_plan")
    wrapper_path = _SITE_ACTOR_WRAPPER.resolve(strict=True)
    wrapper = _bound(wrapper_path, "actor_wrapper")
    critic_path = _SITE_CRITIC_WRAPPER.resolve(strict=True)
    critic = _bound(critic_path, "critic_wrapper")
    workload_path, workload_sha = _json_artifact(
        config_dir / "workload.json", _deployment_workload_body())
    runtime_path, runtime_sha = _json_artifact(
        config_dir / "runtime.json", _deployment_runtime_body())
    if (comparator.model_sha256 != model.sha256
            or comparator.workload_sha256 != workload_sha
            or comparator.runtime_config_sha256 != runtime_sha):
        raise DeploymentFactoryError(
            "frozen production comparator names another immutable frame")
    _snapshot_files, snapshot_semantics = _production_runtime_snapshot(
        deployment.FROZEN_PRODUCTION_PATH)
    _verify_frozen_production_comparator(
        comparator, deployment.FROZEN_PRODUCTION_PATH, snapshot_semantics,
        model_path=model.path, workload_sha256=workload_sha,
        runtime_config_sha256=runtime_sha)
    comparator_path = config_dir / "frozen-production-comparator.json"
    _atomic_bytes(
        comparator_path, frozen_production_comparator.read_bytes())
    comparator_path.chmod(0o400)
    comparator_file_sha = _digest_regular(
        comparator_path, "frozen production comparator")
    _fallback_path, fallback_sha = _json_artifact(
        config_dir / "admission-fallback.json", {
            "schema": "epyc.autokernel.gpu_load_admission_fallback.v1",
            "authority": "fallback_only",
            "default_mode": "cold_serialized",
            "overlap_authority": False,
            "telemetry_source": None,
            "headroom_profile_evidence": None,
            "minimum_headroom_bytes_per_s": None,
            "sentinel_value": _FALLBACK_ONLY_HEADROOM_SENTINEL,
            "sentinel_semantics": (
                "deny guard required by the v2 profile shape; not a measured or "
                "reviewed bandwidth threshold"),
            "replacement_requirement": (
                "a new digest-bound site profile and genuine fresh telemetry source "
                "must be reviewed before cold_overlap can be enabled")})
    profile = {
        "profile_id": _LOAD_PROFILE_ID, "model_path": str(model.path),
        "model_sha256": model.sha256, "model_bytes": model.path.stat().st_size,
        "workload": "decode_tg128", "calls_per_arm": 9, "device_id": "mi210_0",
        "cold_load_host_bytes": model.path.stat().st_size,
        "worst_case_loads_per_interval": 18,
        "minimum_headroom_bytes_per_s": _FALLBACK_ONLY_HEADROOM_SENTINEL,
        "telemetry_max_age_ms": 5000, "evidence_sha256": fallback_sha}
    policy = {"schema": gpu_load_admission.POLICY_SCHEMA,
              "version": "mi210-discovery-fallback-only-v1", "profiles": [profile],
              "examples": [
                  {"id": "illustrative-future-fresh-headroom", "polarity": "positive",
                   "facts": {"profile_id": _LOAD_PROFILE_ID, "illustrative_only": True,
                             "genuine_fresh_telemetry": True,
                             "independently_reviewed_profile": True},
                   "missing": [], "mode": "cold_overlap",
                   "rationale": (
                       "illustrative only: a future digest-bound site profile plus genuine "
                       "fresh telemetry could authorize overlap; this bundle does not"),
                   "disqualifiers": [],
                   "counterfactual": "this fallback bundle always supplies no telemetry and serializes",
                   "evidence": [f"sha256:{fallback_sha}"]},
                  {"id": "qwen05b-unknown-headroom", "polarity": "negative",
                   "facts": {"profile_id": _LOAD_PROFILE_ID, "illustrative_only": True,
                             "telemetry_observed": False},
                   "missing": ["fresh_headroom_telemetry"], "mode": "cold_serialized",
                   "rationale": "fallback-only bundle has no bandwidth authority and must serialize",
                   "disqualifiers": ["telemetry_missing", "fallback_only_profile"],
                   "counterfactual": (
                       "replace this bundle with reviewed site evidence and a genuine fresh "
                       "telemetry source before overlap is eligible"),
                   "evidence": [f"sha256:{fallback_sha}"]}]}
    policy["policy_sha256"] = schemas.content_hash(policy)
    policy_path, policy_sha = _json_artifact(config_dir / "admission-policy.json", policy)
    # Only project shares stated exactly by the reviewed attribution receipt.  The
    # source plan records useful fattn/RoPE outcomes, but does not assign either
    # one an exact device-time share that can safely enter planner authority.
    source_rows = (
        ("mmvq", "ggml_cuda_op_mul_mat_vec_q", .274297, "ggml/src/ggml-cuda/mmvq.cu"),
        ("norm", "ggml_cuda_op_rms_norm", .1074, "ggml/src/ggml-cuda/norm.cu"))
    hotspots = []
    for surface, symbol, share, relative in source_rows:
        path = deployment.FROZEN_PRODUCTION_PATH / relative
        text_body = path.read_text(encoding="utf-8")
        line = next((line.strip() for line in text_body.splitlines() if symbol in line), None)
        if not line:
            raise DeploymentFactoryError(f"reviewed planner symbol disappeared: {symbol}")
        hotspots.append({"surface": surface, "symbol": symbol, "share": share,
                         "source_path": str(path), "source_sha256": _digest_regular(path, relative),
                         "source_excerpt": line,
                         "source_excerpt_sha256": hashlib.sha256(line.encode()).hexdigest(),
                         "note": "projected verified source/hotspot fact; no resource authority"})
    templates = _template_registry()
    template_surfaces = _normalized_template_surfaces(templates, portfolio)
    portfolio_dispatch_authority = _portfolio_dispatch_authority(templates, portfolio)
    reviewed_source_body = {
        "schema": "epyc.autokernel.reviewed_source_package.v1",
        "instrument_commit": _INSTRUMENT_COMMIT,
        "files": [{"relative_path": path, "sha256": _TARGET_SOURCE_SHA256[path],
                   "workspace_path": f"reviewed-source/{path}"}
                  for path in sorted(_TARGET_SOURCE_SHA256)],
    }
    hypotheses = portfolio.hypotheses
    context = {"schema": deployment.PLANNER_CONTEXT_SCHEMA,
               "model_sha256": model.sha256, "workload_sha256": workload_sha,
               "runtime_config_sha256": runtime_sha,
               "profile_receipts": [{"path": str(source_plan.path), "sha256": source_plan.sha256}],
               "hotspots": hotspots,
               "source_constraints": {"template_registry": "gpu-source-templates-v4",
                                      "max_reviewed_files_per_candidate": 2,
                                      "excluded_source_plan_fields": ["planner_posture", "current_execution",
                                                                       "max_overlap_bytes", "overlap_policy"]},
               "initial_strategies": [
                   "Do not repeat retired fattn single-column, Q5 four-wave, Q8 vec4, or RMS128 variants.",
                   "Explore a new literal dispatch-bound hypothesis in one reviewed template.",
                   "Treat prior RoPE64 and Q4_K one-wave results as DNR/top-K context, not promotion evidence."],
               "hypothesis_portfolio_sha256": portfolio.sha256,
               "eligible_hypotheses": _plain(portfolio.eligible_projection()),
               "do_not_repeat": _plain(portfolio.dnr_projection()),
               "incumbents": _plain([row for row in hypotheses
                                      if row["status"] == "candidate_incumbent"]),
               "ineligible_hypotheses": _plain([
                   row for row in hypotheses
                   if not row["current_bundle_eligibility"]["eligible"]]),
               "hypothesis_evidence_manifest_sha256": evidence_manifest["manifest_sha256"],
               "hypothesis_evidence": vendored_evidence,
               "reviewed_source_package_sha256": schemas.content_hash(reviewed_source_body),
               "template_registry_sha256": templates.registry_sha256,
               "template_symbol_authority": {
                   template_id: {
                       path: sorted(symbols)
                       for path, symbols in template.allowed_symbols.items()}
                   for template_id, template in sorted(templates.templates.items())},
               "template_surfaces_sha256": schemas.content_hash(template_surfaces),
               "template_surfaces": template_surfaces,
               "portfolio_dispatch_authority": portfolio_dispatch_authority}
    context["preauthored_continuation_sha256"] = continuation.sha256
    context["preauthored_source_backed_diff_sha256"] = (
        continuation.source_backed_diff_sha256)
    context["preauthored_historical_evidence_sha256"] = (
        historical_q5_evidence["receipt_sha256"])
    context["context_sha256"] = schemas.content_hash(context)
    context_path, context_sha = _json_artifact(config_dir / "planner-context.json", context)
    for directory in (root / "state", root / "evidence", root / "operations", root / "builds"):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise DeploymentFactoryError("bundle output root is not a regular directory")
    value = {"schema": deployment.SCHEMA,
             "production": {"path": str(deployment.FROZEN_PRODUCTION_PATH),
                            "branch": deployment.FROZEN_PRODUCTION_BRANCH,
                            "head": deployment.FROZEN_PRODUCTION_HEAD},
             "instrument": {"repo_path": str(_INSTRUMENT_PATH), "branch": _INSTRUMENT_BRANCH,
                            "commit": _INSTRUMENT_COMMIT,
                            "production_ancestor": deployment.FROZEN_PRODUCTION_HEAD},
             "controller": {"state_root": str(root / "state"),
                            "evidence_root": str(root / "evidence"),
                            "operations_root": str(root / "operations"),
                            "build_root": str(root / "builds"),
                            "max_iterations": 10, "nomination_threshold": .03},
             "actors": {"wrapper_path": str(wrapper.path), "wrapper_sha256": wrapper.sha256,
                        "critic_path": str(critic.path), "critic_sha256": critic.sha256,
                        "environment_profile_id": _STATIC_IDS["environment_profile"]},
             "gpu": {"device_id": "mi210_0", "claim_timeout_s": 0,
                     "inference_window_lock": str(_SITE_WINDOW_LOCK),
                     "inference_window_lease_id": _STATIC_IDS["inference_window_lease"]},
             "immutable_inputs": {"model": {"path": str(model.path), "sha256": model.sha256},
                                  "workload": {"path": str(workload_path), "sha256": workload_sha},
                                  "runtime_config": {"path": str(runtime_path), "sha256": runtime_sha},
                                  "frozen_production_comparator": {
                                      "path": str(comparator_path),
                                      "sha256": comparator_file_sha},
                                  "admission_policy": {"path": str(policy_path), "sha256": policy_sha},
                                  "hypothesis_portfolio": {"path": str(portfolio_path.resolve()),
                                                           "sha256": portfolio_file_sha},
                                  "hypothesis_evidence_manifest": {
                                      "path": str(evidence_manifest_path),
                                      "sha256": evidence_manifest_file_sha},
                                  "hypothesis_portfolio_contract": {
                                      "path": str(contract_path.resolve()),
                                      "sha256": contract_sha},
                                  "preauthored_continuation": {
                                      "path": str(continuation_path.resolve()),
                                      "sha256": continuation_file_sha},
                                  "q5_lds0_attribution_erratum": {
                                      "path": str(erratum_path.resolve()),
                                      "sha256": erratum_file_sha},
                                  "carry_forward": {
                                      "path": str(carry_forward_path.resolve()),
                                      "sha256": carry_forward_file_sha,
                                      "self_sha256": carry_forward_self_sha,
                                      "semantic_sha256": carry_forward_self_sha}},
             "planner_context": {"path": str(context_path), "sha256": context_sha},
             "source_plan": {"source_builder_id": _STATIC_IDS["source_builder"],
                             "evidence_plan_id": _STATIC_IDS["evidence_plan"],
                             "runner_args_id": _STATIC_IDS["runner_args"],
                             "experiment_template_registry_id": _STATIC_IDS["experiment_template_registry"],
                             "experiment_template_registry_sha256": static_template_registry_sha256(),
                             "production_snapshot_id": _STATIC_IDS["production_snapshot"]}}
    value["config_sha256"] = schemas.content_hash(value)
    deployment_path, _ = _json_artifact(config_dir / "deployment.json", value)
    return deployment_path


def _bound(path: Path, role: str) -> evidence.BoundInputFile:
    path = path.resolve(strict=True)
    return evidence.BoundInputFile(role=role, path=path,
                                   sha256=_digest_regular(path, role))


def _verify_expected_runtime_inventory(
        root: Path, inventory: Mapping[str, Any],
) -> tuple[tuple[evidence.BoundInputFile, ...], tuple[dict[str, Any], ...]]:
    """Authenticate every expected runtime object without executing a parser."""
    expected = inventory.get("objects") if isinstance(inventory, Mapping) else None
    if not isinstance(expected, list) or not expected:
        raise DeploymentFactoryError("frozen runtime inventory is unavailable")
    observed: list[dict[str, Any]] = []
    files: dict[Path, evidence.BoundInputFile] = {}
    for row in expected:
        relative = row.get("relative_path") if isinstance(row, Mapping) else None
        if not isinstance(relative, str):
            raise DeploymentFactoryError("frozen runtime inventory path is invalid")
        lexical = root / relative
        try:
            facts = lexical.lstat()
            symlink_target = os.readlink(lexical) \
                if stat.S_ISLNK(facts.st_mode) else None
            resolved = lexical.resolve(strict=True)
            if root.resolve(strict=True) not in (resolved, *resolved.parents):
                raise OSError("runtime object escapes production root")
            if (symlink_target is None and not stat.S_ISREG(facts.st_mode)) \
                    or (symlink_target is not None
                        and PurePosixPath(symlink_target).name != symlink_target):
                raise OSError("runtime object is not a local regular file or alias")
            digest = _digest_regular(resolved, f"production runtime {relative}")
        except OSError as exc:
            raise DeploymentFactoryError(
                f"production runtime inventory lacks {relative}") from exc
        observed.append({
            "relative_path": relative,
            "sha256": digest,
            "symlink_target": symlink_target,
        })
        files.setdefault(
            resolved, evidence.BoundInputFile(
                role=f"production-runtime:{relative}", path=resolved,
                sha256=digest))
    if observed != expected:
        raise DeploymentFactoryError(
            "production runtime inventory differs from manifest")
    return (tuple(files[path] for path in sorted(files)),
            tuple(observed))


def _approved_readelf(
        inventory: Mapping[str, Any], *, runner: Any,
) -> Path:
    """Authenticate the exact parser bytes and version before topology use."""
    authority = inventory.get("readelf") \
        if isinstance(inventory, Mapping) else None
    if not isinstance(authority, Mapping):
        raise DeploymentFactoryError("approved readelf authority is unavailable")
    path = Path(str(authority.get("path")))
    try:
        if (not path.is_absolute() or path != path.resolve(strict=True)
                or _digest_regular(path, "approved readelf") !=
                   authority.get("sha256")):
            raise OSError("readelf identity differs")
        completed = runner(
            (str(path), "--version"), check=False,
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                 "PATH": "/usr/bin:/bin"})
        first_line = completed.stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError) as exc:
        raise DeploymentFactoryError("approved readelf is unavailable") from exc
    if completed.returncode or first_line != authority.get("version"):
        raise DeploymentFactoryError("approved readelf version differs")
    return path


def _production_runtime_snapshot(
        root: Path, *, closure_manifest: Mapping[str, Any] | None = None,
        runner: Any | None = None,
) -> tuple[tuple[evidence.BoundInputFile, ...], dict[str, Any]]:
    """Bind raw CPU/HIP inventory before approved topology inspection."""
    manifest = (_frozen_v9_closure_manifest()
                if closure_manifest is None else closure_manifest)
    inventory = manifest.get("runtime_inventory") \
        if isinstance(manifest, Mapping) else None
    if not isinstance(inventory, Mapping):
        raise DeploymentFactoryError("frozen runtime inventory is unavailable")
    files, rows = _verify_expected_runtime_inventory(root, inventory)
    run = subprocess.run if runner is None else runner
    readelf = _approved_readelf(inventory, runner=run)
    by_flavor: dict[str, dict[str, dict[str, Any]]] = {
        "build": {}, "build-hip": {}}
    for row in rows:
        path = PurePosixPath(row["relative_path"])
        by_flavor[path.parts[0]][path.name] = row
    semantics: dict[str, Any] = {
        "production_head": deployment.FROZEN_PRODUCTION_HEAD,
        "closures": {},
    }
    for flavor, required in (
            ("build", frozenset()),
            ("build-hip", frozenset({"libggml-hip.so.0"}))):
        entries = by_flavor[flavor]
        required_names = {"llama-server", "llama-bench", *required}
        if not required_names.issubset(entries):
            raise DeploymentFactoryError(
                f"{flavor} production runtime lacks its reviewed entrypoints")
        directory = root / flavor / "bin"
        topology: dict[str, Any] = {}
        for name, row in sorted(entries.items()):
            resolved = (directory / name).resolve(strict=True)
            completed = run(
                (str(readelf), "-dW", str(resolved)), check=False,
                stdin=subprocess.DEVNULL, capture_output=True, text=True)
            if completed.returncode:
                raise DeploymentFactoryError(
                    f"cannot inspect production runtime object {resolved}")
            needed = sorted(re.findall(
                r"Shared library: \[(.+?)\]", completed.stdout))
            unknown_local = sorted(
                dependency for dependency in needed
                if (directory / dependency).exists()
                and dependency not in entries)
            if unknown_local:
                raise DeploymentFactoryError(
                    f"{flavor} topology contains unbound local objects")
            topology[name] = {
                "resolved_name": resolved.name,
                "sha256": row["sha256"],
                "needed_local": sorted(set(needed) & set(entries)),
                "symlink": row["symlink_target"],
            }
        semantics["closures"][flavor] = {
            "configuration": "cpu" if flavor == "build" else "rocm-gfx90a",
            "entrypoints": ["llama-bench", "llama-server"],
            "topology": topology,
        }
    return files, semantics


def _checked_profiler_bound(path: Path, role: str, expected: str,
                            ) -> evidence.BoundInputFile:
    value = _bound(path, role)
    if value.sha256 != expected:
        raise DeploymentFactoryError(f"fixed rocprofv3 {role} digest changed")
    return value


def _rocprof_v3_policy(config: deployment.DiscoveryDeployment) -> tuple[
        evidence.BoundInputFile, ...]:
    """Seal the official SDK/package and every non-system mapped DSO."""
    python = _checked_profiler_bound(
        _ROCPROF_V3_PYTHON, "executable", _ROCPROF_V3_PYTHON_SHA256)
    wrapper = _checked_profiler_bound(
        _ROCPROF_V3, "profiler_wrapper", _ROCPROF_V3_SHA256)
    package = _checked_profiler_bound(
        _ROCPROF_V3_PACKAGE, "profiler_package", _ROCPROF_V3_PACKAGE_SHA256)
    mapped = (
        _checked_profiler_bound(
            _ROCPROF_V3_SDK_LIB, "profiler_sdk_library",
            _ROCPROF_V3_SDK_LIB_SHA256),
        _checked_profiler_bound(
            _ROCPROF_V3_TOOL_LIB, "profiler_sdk_tool_library",
            _ROCPROF_V3_TOOL_LIB_SHA256),
        _checked_profiler_bound(
            _ROCPROF_V3_AQL_LIB, "profiler_aqlprofile_library",
            _ROCPROF_V3_AQL_LIB_SHA256),
        _checked_profiler_bound(
            _ROCPROF_V3_HSA_LIB, "profiler_hsa_runtime_library",
            _ROCPROF_V3_HSA_LIB_SHA256),
        _checked_profiler_bound(
            _ROCPROF_V3_REGISTER_LIB, "profiler_register_library",
            _ROCPROF_V3_REGISTER_LIB_SHA256),
        _checked_profiler_bound(
            _ROCPROF_V3_PCI_LIB, "profiler_libpci_library",
            _ROCPROF_V3_PCI_LIB_SHA256),
    )
    root = config.operations_root / "config"
    root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for role, prefix, name in (
        ("profiler_runtime_manifest", _ROCPROF_V3_SDK,
         "rocprofv3-sdk-closure.json"),
        ("profiler_aqlprofile_manifest", _ROCPROF_V3_OLD_LIB,
         "rocprofv3-aqlprofile-closure.json"),
        ("profiler_libpci_manifest", _ROCPROF_V3_PCI_LIB_DIR,
         "rocprofv3-libpci-closure.json"),
    ):
        try:
            snapshot = evidence.profiler_prefix_snapshot(prefix.resolve(strict=True))
        except (OSError, evidence.EvidenceProducerError) as exc:
            raise DeploymentFactoryError(
                f"cannot snapshot {role}") from exc
        path, digest = _json_artifact(root / name, snapshot)
        manifests.append(evidence.BoundInputFile(role, path, digest))
    return (python, wrapper, package, *manifests, *mapped)


def _template_registry() -> ExperimentTemplateRegistry:
    if (_digest_regular(_PROFILE_TRACE_RECEIPT, "reviewed profile receipt")
            != _PROFILE_TRACE_RECEIPT_SHA256
            or _digest_regular(_PROFILE_TRACE_CSV, "reviewed profile timestamp CSV")
            != _PROFILE_TRACE_CSV_SHA256
            or _digest_regular(_PROFILE_V3_TRACE_CSV, "reviewed v3 profile trace")
            != _PROFILE_V3_TRACE_CSV_SHA256
            or _digest_regular(_PROFILE_V3_AGENT_CSV, "reviewed v3 agent trace")
            != _PROFILE_V3_AGENT_CSV_SHA256):
        raise DeploymentFactoryError("reviewed real-trace template authority changed")
    mmvq_pattern = lambda type_id, flags: (
        rf"^void mul_mat_vec_q<\(ggml_type\){type_id}, 1, {flags}>\(.*\)$")
    mmvq_anchor = (
        (mmvq_pattern(6, "true, true"), 6063, 57344, 128, 1024),
        (mmvq_pattern(6, "true, true"), 4644, 8192, 128, 1024),
        (mmvq_pattern(6, "true, true"), 3096, 311296, 128, 1024),
        (mmvq_pattern(6, "false, true"), 129, 57344, 128, 512),
        (mmvq_pattern(12, "true, false"), 1548, 114688, 128, 512),
        (mmvq_pattern(14, "true, false"), 1548, 114688, 128, 512),
        (mmvq_pattern(8, "true, true"), 1548, 8192, 256, 6144),
        (mmvq_pattern(8, "false, true"), 129, 9723904, 256, 3072),
    )
    fattn_tile_anchor = ((
        r"^void flash_attn_tile<64, 64, 2, 1, false>\(.*\)$",
        3096, 7168, 64, 5120),)
    fattn_common_anchor = (*fattn_tile_anchor, (
        r"^void flash_attn_combine_results<64>\(.*\)$",
        3096, 896, 64, 512))
    quantize_anchor = (
        (r"^quantize_q8_1\(.*\)$", 15609, 1024, 256, 0),
        (r"^quantize_q8_1\(.*\)$", 3096, 5120, 256, 0),
    )
    rope_anchor = (
        (r"^void rope_neox<true, false, float, __half>\(.*\)$",
         3096, 512, 256, 0),
        (r"^void rope_neox<true, false, float, float>\(.*\)$",
         3096, 3584, 256, 0),
    )
    norm_anchor = ((
        r"^void rms_norm_f32<256, true, false>\(.*\)$",
        6321, 256, 256, 512),)
    set_rows_anchor = ((
        r"^void k_set_rows<float, long, __half>\(.*\)$",
        3096, 256, 256, 0),)
    target_runtime_screen = {
        "stage_id": "target_runtime_graphs_on_screen",
        "workload": "decode_tg128",
        "hip_graphs": True,
        "paired": True,
        "decision_required": True,
        "exact_invocations": 1,
        "resume_without_repeat": True,
        "authority": (
            "whole-model reward direction only; graphs-off attribution remains "
            "a separate route/device-time receipt"),
    }
    decision_evidence = {
        "all_exact_routes_have_duration": True,
        "exact_attribution_gain_required": True,
        "target_runtime_graphs_on_gain_required": True,
        "combination": "conjunction",
        "direction": "lower_exact_duration_and_higher_throughput",
        "short_circuit_graphs_on_when_exact_nonpositive": True,
    }
    stage_fsm = {
        "stages": [
            "correctness", "candidate_attribution", "anchor_attribution",
            "measurement_graphs_off_screen",
            "target_runtime_graphs_on_screen"],
        "crash_after_test_points": [
            "correctness", "candidate_attribution", "anchor_attribution",
            "measurement_graphs_off_screen",
            "target_runtime_graphs_on_screen"],
        "completed_stage_policy": "revalidate_receipt_and_reuse",
        "first_incomplete_stage_policy": "execute_once",
        "reject_identity_drift": True,
        "attribution_arm_order_schedule": {
            "counterbalanced": True,
            "s1": ["candidate", "anchor"],
            "s2": ["anchor", "candidate"],
            "authority": "deployment+manifest keyed; S2 reverses S1",
        },
    }
    gqa7_candidate_variants = {
        "gqa7_bulk_pairs": {
            "kernel_name": "void flash_attn_tile<64, 64, 1, 2, false>",
            "calls": 3096, "grid": 3072, "workgroup": 64,
            "lds_bytes": 5120, "gqa_ratio": 7, "head_size": 64,
            "ncols2": 2,
        },
        "gqa7_scalar_tail": {
            "kernel_name": "void flash_attn_tile<64, 64, 2, 1, false>",
            "calls": 3096, "grid": 1024, "workgroup": 64,
            "lds_bytes": 5120, "gqa_ratio": 7, "head_size": 64,
            "ncols2": 1,
        },
    }
    gqa7_correctness_cases = [
        {"op": "FLASH_ATTN_EXT", "hsk": 64, "hsv": 64,
         "gqa_ratio": 7, "query_tokens": 1, "kv": 128,
         "mask": False, "expected_matches": 1,
         "params_pattern": r"^hsk=64,hsv=64,nh=2,nr23=\[7,1\],kv=128,nb=1,mask=0,"},
        {"op": "FLASH_ATTN_EXT", "hsk": 64, "hsv": 64,
         "gqa_ratio": 7, "query_tokens": 1, "kv": 512,
         "mask": True, "expected_matches": 1,
         "params_pattern": r"^hsk=64,hsv=64,nh=2,nr23=\[7,1\],kv=512,nb=1,mask=1,"},
        {"op": "FLASH_ATTN_EXT", "hsk": 64, "hsv": 64,
         "gqa_ratio": 7, "query_tokens": 1, "kv": 2048,
         "mask": True, "expected_matches": 1,
         "params_pattern": r"^hsk=64,hsv=64,nh=2,nr23=\[7,1\],kv=2048,nb=1,mask=1,"},
    ]
    families = (
        {"id": "cuda-fattn-v2", "path": "ggml/src/ggml-cuda/fattn.cu",
         "primary": "ggml_cuda_get_best_fattn_kernel",
         "symbols": ("ggml_cuda_get_best_fattn_kernel", "ggml_cuda_flash_attn_ext",
            "ggml_cuda_flash_attn_ext_vec", "ggml_cuda_flash_attn_ext_mma_f16",
            "ggml_cuda_flash_attn_ext_supported", "ggml_cuda_flash_attn_ext_get_alloc_size"),
         "markers": ("flash_attn_tile<", "flash_attn_vec<", "flash_attn_combine_results<"),
         "op": "FLASH_ATTN_EXT", "cases": 2868, "anchor": fattn_tile_anchor,
         "replays": ({"trace": "VEC selector", "symbol": "ggml_cuda_get_best_fattn_kernel"},)},
        {"id": "cuda-fattn-tile-v1", "path": "ggml/src/ggml-cuda/fattn-tile.cuh",
         "primary": "ggml_cuda_fattn_tile_get_config_amd",
         "symbols": ("ggml_cuda_fattn_tile_get_config_amd", "ggml_cuda_fattn_tile_get_config_amd_rdna",
            "ggml_cuda_fattn_tile_get_config", "ggml_cuda_fattn_tile_get_nthreads",
            "ggml_cuda_fattn_tile_get_occupancy", "ggml_cuda_fattn_tile_get_nbatch_fa",
            "ggml_cuda_fattn_tile_get_nbatch_K", "flash_attn_tile_load_tile",
            "flash_attn_tile_iter_KQ", "flash_attn_tile_iter", "flash_attn_tile",
            "launch_fattn_tile_switch_ncols1", "launch_fattn_tile_switch_ncols2",
            "ggml_cuda_flash_attn_ext_tile_case", "ggml_cuda_flash_attn_ext_tile"),
         "markers": ("flash_attn_tile<",), "op": "FLASH_ATTN_EXT", "cases": 2868,
         "anchor": fattn_tile_anchor,
         "replays": ({"trace": "D64 Q1 TILE geometry", "symbol": "ggml_cuda_fattn_tile_get_config_amd"},)},
        {"id": "cuda-fattn-tile-entry-v1", "path": "ggml/src/ggml-cuda/fattn-tile.cu",
         "primary": "ggml_cuda_flash_attn_ext_tile", "symbols": ("ggml_cuda_flash_attn_ext_tile",),
         "markers": ("flash_attn_tile<",), "op": "FLASH_ATTN_EXT", "cases": 2868,
         "anchor": fattn_tile_anchor, "replays": ()},
        {"id": "cuda-fattn-common-v1", "path": "ggml/src/ggml-cuda/fattn-common.cuh",
         "primary": "launch_fattn",
         "symbols": ("ggml_cuda_flash_attn_ext_get_f16_extra_data", "vec_dot_fattn_vec_KQ_f16",
            "vec_dot_fattn_vec_KQ_bf16", "vec_dot_fattn_vec_KQ_q4_0", "vec_dot_fattn_vec_KQ_q4_1",
            "vec_dot_fattn_vec_KQ_q5_0", "vec_dot_fattn_vec_KQ_q5_1", "vec_dot_fattn_vec_KQ_q8_0",
            "quantize_q8_1_to_shared", "dequantize_V_f16", "dequantize_V_bf16", "dequantize_V_q4_0",
            "dequantize_V_q4_1", "dequantize_V_q5_0", "dequantize_V_q5_1", "dequantize_V_q8_0",
            "flash_attn_mask_to_KV_max", "flash_attn_stream_k_fixup_uniform",
            "flash_attn_stream_k_fixup_general", "flash_attn_combine_results", "launch_fattn"),
         "markers": ("flash_attn_tile<", "flash_attn_combine_results<"),
         "op": "FLASH_ATTN_EXT", "cases": 2868, "anchor": fattn_common_anchor, "replays": ()},
        {"id": "cuda-fattn-combine-v1", "path": "ggml/src/ggml-cuda/fattn-common.cuh",
         "primary": "flash_attn_combine_results",
         "symbols": ("flash_attn_combine_results",),
         "markers": ("flash_attn_combine_results<",),
         "op": "FLASH_ATTN_EXT", "cases": 2868,
         "anchor": (fattn_common_anchor[1],), "replays": ()},
        {"id": "cuda-fattn-gqa7-common-v1",
         "paths": ("ggml/src/ggml-cuda/fattn-common.cuh",
                   "ggml/src/ggml-cuda/fattn-tile.cuh"),
         "primary": "launch_fattn",
         "symbols_by_path": {
             "ggml/src/ggml-cuda/fattn-common.cuh": ("launch_fattn",),
             "ggml/src/ggml-cuda/fattn-tile.cuh": (
                 "launch_fattn_tile_switch_ncols1",
                 "launch_fattn_tile_switch_ncols2")},
         "markers": ("flash_attn_tile<", "flash_attn_combine_results<"),
         "op": "FLASH_ATTN_EXT", "cases": 2868,
         "anchor": fattn_common_anchor, "replays": (), "gqa7": True},
        {"id": "cuda-mmvq-v2", "path": "ggml/src/ggml-cuda/mmvq.cu",
         "primary": "ggml_cuda_op_mul_mat_vec_q",
         "symbols": ("ggml_cuda_op_mul_mat_vec_q", "ggml_cuda_mul_mat_vec_q",
            "mul_mat_vec_q_switch_type", "mul_mat_vec_q_switch_ncols_dst",
            "mul_mat_vec_q_moe_launch", "mul_mat_vec_q_switch_fusion",
            "mul_mat_vec_q8_0_prefetch_launch"),
         "markers": ("mul_mat_vec_q<",), "op": "MUL_MAT", "cases": 1139,
         "anchor": mmvq_anchor,
         "replays": tuple({"trace": f"{q} exact dispatch", "symbol": "mul_mat_vec_q_switch_type"}
                          for q in ("Q4_K", "Q5_0", "Q6_K"))},
        {"id": "cuda-mmvq-q5-onewave-continuation-v1",
         "path": "ggml/src/ggml-cuda/mmvq.cu",
         "primary": "calc_nwarps",
         "symbols": ("mmvq_parameter_table_id", "get_device_table_id",
                     "calc_nwarps", "calc_rows_per_block"),
         "markers": ("mul_mat_vec_q<",), "op": "MUL_MAT", "cases": 1139,
         "anchor": mmvq_anchor[:4], "replays": (),
         "q5_preauthored": True,
         "planner_target_exclusions": ({
             "kernel_pattern": mmvq_anchor[3][0], "calls": 129,
             "grid": 57344, "workgroup": 128, "lds_bytes": 512,
             "reason": "Q5 false/true tail is structural and excluded from reward"},)},
        {"id": "cuda-vecdotq-v1", "path": "ggml/src/ggml-cuda/vecdotq.cuh",
         "primary": "vec_dot_q5_0_q8_1",
         "symbols": ("get_int_b1", "get_int_b2", "get_int_b4", "get_int_from_table_16", "unpack_ksigns",
            "vec_dot_q4_0_q8_1_impl", "vec_dot_q4_1_q8_1_impl", "vec_dot_q5_0_q8_1_impl",
            "vec_dot_q5_1_q8_1_impl", "vec_dot_q8_0_q8_1_impl", "vec_dot_q8_1_q8_1_impl",
            "vec_dot_q8_0_16_q8_1_impl", "vec_dot_mxfp4_q8_1", "vec_dot_nvfp4_q8_1",
            "vec_dot_q2_K_q8_1_impl_mmvq", "vec_dot_q2_K_q8_1_impl_mmq",
            "vec_dot_q3_K_q8_1_impl_mmvq", "vec_dot_q3_K_q8_1_impl_mmq",
            "vec_dot_q4_K_q8_1_impl_vmmq", "vec_dot_q4_K_q8_1_impl_mmq",
            "vec_dot_q5_K_q8_1_impl_vmmq", "vec_dot_q5_K_q8_1_impl_mmq",
            "vec_dot_q6_K_q8_1_impl_mmvq", "vec_dot_q6_K_q8_1_impl_mmq",
            "vec_dot_q1_0_q8_1", "vec_dot_q4_0_q8_1", "vec_dot_q4_1_q8_1",
            "vec_dot_q5_0_q8_1", "vec_dot_q5_1_q8_1", "vec_dot_q8_0_q8_1",
            "vec_dot_q2_K_q8_1", "vec_dot_q3_K_q8_1", "vec_dot_q4_K_q8_1",
            "vec_dot_q5_K_q8_1", "vec_dot_q6_K_q8_1", "vec_dot_iq2_xxs_q8_1",
            "vec_dot_iq2_xs_q8_1", "vec_dot_iq2_s_q8_1", "vec_dot_iq3_xxs_q8_1",
            "vec_dot_iq3_s_q8_1", "vec_dot_iq1_s_q8_1", "vec_dot_iq1_m_q8_1",
            "vec_dot_iq4_nl_q8_1", "vec_dot_iq4_xs_q8_1"),
         "markers": ("mul_mat_vec_q<",), "op": "MUL_MAT", "cases": 1139,
         "anchor": mmvq_anchor, "replays": (),
         "planner_target_exclusions": ({
             "kernel_pattern": mmvq_anchor[3][0], "calls": 129,
             "grid": 57344, "workgroup": 128, "lds_bytes": 512,
             "reason": "Q5 false/true tail is not the reviewed true/true dequant route"},)},
        {"id": "cuda-vecdotq-q4k-v1", "path": "ggml/src/ggml-cuda/vecdotq.cuh",
         "primary": "vec_dot_q4_K_q8_1",
         "symbols": ("vec_dot_q4_K_q8_1", "vec_dot_q4_K_q8_1_impl_vmmq"),
         "markers": ("mul_mat_vec_q<",), "op": "MUL_MAT", "cases": 1139,
         "anchor": (mmvq_anchor[4],), "replays": ()},
        {"id": "cuda-vecdotq-q6k-v1", "path": "ggml/src/ggml-cuda/vecdotq.cuh",
         "primary": "vec_dot_q6_K_q8_1",
         "symbols": ("vec_dot_q6_K_q8_1", "vec_dot_q6_K_q8_1_impl_mmvq"),
         "markers": ("mul_mat_vec_q<",), "op": "MUL_MAT", "cases": 1139,
         "anchor": (mmvq_anchor[5],), "replays": ()},
        {"id": "cuda-quantize-q8-v1", "path": "ggml/src/ggml-cuda/quantize.cu",
         "primary": "quantize_q8_1",
         "symbols": ("quantize_q8_1", "quantize_mmq_nvfp4", "quantize_mmq_mxfp4",
            "quantize_mmq_q8_1", "quantize_row_q8_1_cuda", "quantize_mmq_q8_1_cuda",
            "quantize_scatter_mmq_q8_1_cuda", "quantize_scatter_mmq_fp4_cuda",
            "quantize_mmq_fp4_cuda"),
         "markers": ("quantize_q8_1",), "op": "MUL_MAT", "cases": 1139,
         "anchor": quantize_anchor,
         "replays": ({"trace": "Q8_1 block128 nonreplication", "symbol": "quantize_q8_1"},)},
        {"id": "cuda-rope-v2", "path": "ggml/src/ggml-cuda/rope.cu",
         "primary": "ggml_cuda_op_rope_impl",
         "symbols": ("ggml_cuda_op_rope_impl", "ggml_cuda_op_rope", "ggml_cuda_op_rope_back",
            "ggml_cuda_op_rope_fused", "rope_norm", "rope_neox", "rope_multi", "rope_vision",
            "rope_norm_cuda", "rope_neox_cuda", "rope_multi_cuda", "rope_vision_cuda"),
         "markers": ("rope_neox<",), "op": "ROPE", "cases": 428,
         "anchor": rope_anchor, "replays": ({"trace": "RoPE64 top-K", "symbol": "ggml_cuda_op_rope_impl"},)},
        {"id": "cuda-norm-v2", "path": "ggml/src/ggml-cuda/norm.cu",
         "primary": "ggml_cuda_op_rms_norm",
         "symbols": ("ggml_cuda_op_norm", "ggml_cuda_op_group_norm", "ggml_cuda_op_rms_norm",
            "ggml_cuda_op_rms_norm_fused", "ggml_cuda_op_rms_norm_fused_add",
            "ggml_cuda_op_rms_norm_back", "ggml_cuda_op_l2_norm", "norm_f32", "group_norm_f32",
            "rms_norm_f32", "rms_norm_back_f32", "l2_norm_f32", "norm_f32_cuda",
            "group_norm_f32_cuda", "rms_norm_f32_cuda", "rms_norm_mul_f32_cuda",
            "rms_norm_back_f32_cuda", "l2_norm_f32_cuda"),
         "markers": ("rms_norm_f32<",), "op": "RMS_NORM", "cases": 21,
         "anchor": norm_anchor, "replays": ({"trace": "RMS128 negative", "symbol": "ggml_cuda_op_rms_norm"},)},
        {"id": "cuda-set-rows-v1", "path": "ggml/src/ggml-cuda/set-rows.cu",
         "primary": "ggml_cuda_op_set_rows",
         "symbols": ("k_set_rows_quant", "set_rows_cuda_quant", "k_set_rows", "set_rows_cuda",
                     "ggml_cuda_op_set_rows"),
         "markers": ("k_set_rows<",), "op": "SET_ROWS", "cases": 655,
         "anchor": set_rows_anchor, "replays": ()},
    )
    templates = {}
    for family in families:
        template_id = family["id"]
        paths = tuple(family.get("paths", (family.get("path"),)))
        if not paths or any(not isinstance(path, str) for path in paths):
            raise DeploymentFactoryError("template family paths are malformed")
        path = paths[0]
        symbols_by_path = family.get(
            "symbols_by_path", {path: family.get("symbols", ())})
        if set(symbols_by_path) != set(paths):
            raise DeploymentFactoryError("template family symbols do not match paths")
        anchor = tuple(evidence.ExactDispatch(
            signature=f"{template_id}.anchor.{index}", kernel_pattern=row[0],
            calls=row[1], grid=row[2], workgroup=row[3], lds_bytes=row[4],
            blocks_per_call=row[2] // row[3])
            for index, row in enumerate(family["anchor"]))
        templates[template_id] = ExperimentTemplate(
            template_id=template_id, target_surface="gpu_decode", target_symbol=family["primary"],
            correctness_id="backend-ops-hip-v1", dispatch_id="decode-tg128-rocprof-v3",
            dispatch=evidence.DispatchContract(candidate_exact=tuple(
                replace(row, signature=row.signature.replace(".anchor.", ".candidate-seed."))
                for row in anchor), anchor_exact=anchor),
            allowed_files=frozenset(paths),
            allowed_symbols={relative: frozenset(symbols)
                             for relative, symbols in symbols_by_path.items()},
            semantics={"workload": "decode_tg128", "calls_per_arm": 9,
                       "load_admission_profile_id": _LOAD_PROFILE_ID,
                       "correctness_op": family["op"],
                       "expected_correctness_cases": family["cases"],
                       "suite_seed": _CORRECTNESS_SUITE_SEED,
                       "test_source_commit": _INSTRUMENT_COMMIT,
                       "test_source_sha256": _INSTRUMENT_TEST_SOURCE_SHA256,
                       "production_instrument_target_sha256_by_file": {
                           relative: _TARGET_SOURCE_SHA256[relative]
                           for relative in paths},
                       "profile_anchor_source": {
                           "receipt": str(_PROFILE_TRACE_RECEIPT),
                           "receipt_sha256": _PROFILE_TRACE_RECEIPT_SHA256,
                           "timestamp_csv": str(_PROFILE_TRACE_CSV),
                           "timestamp_csv_sha256": _PROFILE_TRACE_CSV_SHA256,
                           "v3_kernel_trace": str(_PROFILE_V3_TRACE_CSV),
                           "v3_kernel_trace_sha256": _PROFILE_V3_TRACE_CSV_SHA256,
                           "v3_agent_info": str(_PROFILE_V3_AGENT_CSV),
                           "v3_agent_info_sha256": _PROFILE_V3_AGENT_CSV_SHA256,
                           "cross_profiler_projection_sha256":
                               "8bf84656cd12eecf8e9881fd0f2b6f9f8da7e4485a0a668dcb08065e930fbc54"},
                       "manual_replay_traces": [dict(row, file=path) for row in family["replays"]],
                       "planner_target_exclusions": list(
                           family.get("planner_target_exclusions", ())),
                       **({"preauthored_q5_candidate_dispatch": {
                           f"cuda-mmvq-q5-onewave-continuation-v1.anchor.{index}": {
                               "kernel_name": "", "calls": row[1], "grid": row[2],
                               "workgroup": 64, "lds_bytes": 0}
                           for index, row in enumerate(mmvq_anchor[:3])}}
                          if family.get("q5_preauthored") is True else {}),
                       "target_runtime_screen": target_runtime_screen,
                       "stage_fsm": stage_fsm,
                       "decision_evidence": decision_evidence,
                       **({"candidate_dispatch_variants": gqa7_candidate_variants,
                           "required_correctness_cases": gqa7_correctness_cases,
                           "correctness_invocations": [
                               {"invocation_id": "generic_flash_attn_ext",
                                "case_set": "generic_flash_attn_ext_v1",
                                "expected_cases": 2868,
                                "required_cases": [],
                                "environment_overrides": []},
                               {"invocation_id": "odd_gqa7_d64_q1",
                                "case_set": "odd_gqa7_d64_q1_v1",
                                "expected_cases": len(gqa7_correctness_cases),
                                "required_cases": gqa7_correctness_cases,
                                "environment_overrides": [[
                                    "AUTOKERNEL_CORRECTNESS_CASE_SET",
                                    "odd_gqa7_d64_q1_v1"]]},
                           ],
                           "candidate_dispatch_authority":
                               "derived_from_anchor_by_exact_7_equals_3x2_plus_1_partition"}
                          if family.get("gqa7") is True else {}),
                       "dispatch_bounds": {"calls": [1, 20000], "grid": [64, 16777216],
                                           "workgroup": [64, 1024], "lds_bytes": [0, 131072],
                                           "kernel_name_fragments": list(family["markers"])}})
    provisional = object.__new__(ExperimentTemplateRegistry)
    # Fill exact full demangled Q5 candidate names from the reviewed anchor CSV;
    # the empty placeholders above cannot become actor authority.
    q5_template = templates.get("cuda-mmvq-q5-onewave-continuation-v1")
    if q5_template is not None:
        q5_rows = {}
        for anchor in q5_template.dispatch.anchor_exact[:3]:
            matches = [
                (str(row["kernel"]), int(row["grid"]), int(row["workgroup"]),
                 int(row["lds"]))
                for row in evidence._load_dispatches(
                    _PROFILE_V3_TRACE_CSV,
                    profiler_trace_schema_id=evidence.ROCPROF_V3_TRACE_ID,
                    expected_rows=59925)
                if (re.fullmatch(anchor.kernel_pattern, str(row["kernel"]))
                    and int(row["grid"]) == anchor.grid
                    and int(row["workgroup"]) == anchor.workgroup
                    and int(row["lds"]) == anchor.lds_bytes)]
            names = {row[0] for row in matches}
            if len(names) != 1:
                raise DeploymentFactoryError(
                    "reviewed Q5 anchor lacks one exact native kernel name")
            name = names.pop().replace(
                "(ggml_type)6, 1, true, true",
                "(ggml_type)6, 1, true, false")
            q5_rows[anchor.signature] = {
                "kernel_name": name, "calls": anchor.calls,
                "grid": anchor.grid, "workgroup": 64,
                "lds_bytes": 0}
        semantics = dict(q5_template.semantics)
        semantics["preauthored_q5_candidate_dispatch"] = q5_rows
        templates[q5_template.template_id] = replace(
            q5_template, semantics=semantics)
    object.__setattr__(provisional, "version", "gpu-source-templates-v4")
    object.__setattr__(provisional, "templates", MappingProxyType(templates))
    body = {"version": "gpu-source-templates-v4", "templates": {
        key: {"template_id": value.template_id, "target_surface": value.target_surface,
              "target_symbol": value.target_symbol, "correctness_id": value.correctness_id,
              "dispatch_id": value.dispatch_id, "allowed_files": sorted(value.allowed_files),
              "allowed_symbols": {path: sorted(symbols) for path, symbols in value.allowed_symbols.items()},
              "semantics": dict(value.semantics),
              "dispatch": {"candidate_exact": [vars(row) for row in value.dispatch.candidate_exact],
                           "anchor_exact": [vars(row) for row in value.dispatch.anchor_exact],
                           "candidate_forbidden": [vars(row) for row in value.dispatch.candidate_forbidden],
                           "anchor_forbidden": [vars(row) for row in value.dispatch.anchor_forbidden],
                           "invariants": [vars(row) for row in value.dispatch.invariants]}}
        for key, value in sorted(templates.items())}}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return ExperimentTemplateRegistry("gpu-source-templates-v4", digest, templates)


def _reviewed_source_package(
        config: deployment.DiscoveryDeployment,
        templates: ExperimentTemplateRegistry) -> controller.ReviewedSourcePackage:
    paths = sorted({path for template in templates.templates.values()
                    for path in template.allowed_files})
    files: list[controller.ReviewedSourceFile] = []
    for relative in paths:
        result = subprocess.run(
            ("git", "-C", str(config.instrument_path), "show",
             f"{config.instrument_commit}:{relative}"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        expected = _TARGET_SOURCE_SHA256.get(relative)
        actual = hashlib.sha256(result.stdout).hexdigest()
        if result.returncode or expected is None or actual != expected:
            raise DeploymentFactoryError(
                f"reviewed actor source differs from instrument authority: {relative}")
        files.append(controller.ReviewedSourceFile(relative, actual, result.stdout))
    body = {"schema": "epyc.autokernel.reviewed_source_package.v1",
            "instrument_commit": config.instrument_commit,
            "files": [{"relative_path": item.relative_path, "sha256": item.sha256,
                       "workspace_path": f"reviewed-source/{item.relative_path}"}
                      for item in files]}
    return controller.ReviewedSourcePackage(
        config.instrument_commit, tuple(files), schemas.content_hash(body))


_TEMPLATE_CHANGE_CLASSES = {
    "cuda-fattn-v2": ("dispatcher",),
    "cuda-fattn-tile-v1": ("dispatcher",),
    "cuda-fattn-tile-entry-v1": ("dispatcher",),
    "cuda-fattn-common-v1": ("arithmetic", "fusion"),
    "cuda-fattn-combine-v1": ("arithmetic",),
    "cuda-fattn-gqa7-common-v1": ("dispatcher",),
    "cuda-mmvq-v2": ("arithmetic", "dispatcher"),
    "cuda-mmvq-q5-onewave-continuation-v1": ("dispatcher",),
    "cuda-vecdotq-v1": ("arithmetic",),
    "cuda-vecdotq-q4k-v1": ("arithmetic",),
    "cuda-vecdotq-q6k-v1": ("arithmetic",),
    "cuda-quantize-q8-v1": ("arithmetic",),
    "cuda-rope-v2": ("arithmetic",),
    "cuda-norm-v2": ("arithmetic",),
    "cuda-set-rows-v1": ("arithmetic", "layout"),
}


def _normalized_template_surfaces(
        templates: ExperimentTemplateRegistry,
        portfolio: hypothesis_portfolio.Portfolio) -> dict[str, Mapping[str, Any]]:
    surfaces: dict[str, Mapping[str, Any]] = {}
    for template_id, template in templates.templates.items():
        exact = template.dispatch.anchor_exact
        eligible = [row for row in portfolio.eligible_hypotheses()
                    if template_id in row["current_bundle_eligibility"]["template_ids"]]
        if len(eligible) > 1:
            raise DeploymentFactoryError(
                f"{template_id} has multiple eligible questions; v2 requires one exact "
                "dispatch authority per template")
        selected: set[tuple[str, int, int, int, int]] = set()
        excluded: set[tuple[str, int, int, int, int]] = set()
        for record in eligible:
            current = next(anchor for anchor in record["dispatch_anchors"]
                           if anchor["frame_id"] == portfolio.body["current_bundle"]["frame_id"])
            for carrier, target in ((current["signatures"], selected),
                                    (current["excluded_signatures"], excluded)):
                for row in carrier:
                    route = (row["route_id"], *(row[key] for key in
                              ("calls", "grid", "workgroup", "lds_bytes")))
                    if route in target:
                        raise DeploymentFactoryError(
                            f"portfolio repeats {template_id} dispatch geometry")
                    target.add(route)
        available = {(row.signature, row.calls, row.grid, row.workgroup,
                      row.lds_bytes) for row in exact}
        declared_exclusions: set[tuple[str, int, int, int, int]] = set()
        for excluded_row in template.semantics.get("planner_target_exclusions", []):
            matches = [row for row in exact
                       if (row.kernel_pattern, row.calls, row.grid, row.workgroup,
                           row.lds_bytes) == (
                               excluded_row["kernel_pattern"], excluded_row["calls"],
                               excluded_row["grid"], excluded_row["workgroup"],
                               excluded_row["lds_bytes"])]
            if len(matches) != 1:
                raise DeploymentFactoryError(
                    f"{template_id} exclusion lacks one exact deployed route")
            row = matches[0]
            declared_exclusions.add((row.signature, row.calls, row.grid,
                                     row.workgroup, row.lds_bytes))
        if eligible and (not selected or not selected.issubset(available)
                         or excluded != declared_exclusions
                         or not excluded.issubset(available)
                         or selected & excluded):
            raise DeploymentFactoryError(
                f"portfolio geometry differs from {template_id} reviewed trace authority")
        authorized = selected if eligible else available - declared_exclusions
        dispatch = [{"route_id": route_id, "calls": calls, "grid": grid, "workgroup": workgroup,
                     "lds_bytes": lds_bytes}
                    for route_id, calls, grid, workgroup, lds_bytes in sorted(authorized)]
        surfaces[template_id] = {
            "source_files": sorted(template.allowed_files),
            "source_symbols": sorted({symbol for symbols in
                                      template.allowed_symbols.values()
                                      for symbol in symbols}),
            "change_classes": list(_TEMPLATE_CHANGE_CLASSES[template_id]),
            "dispatch_signatures": dispatch,
            "excluded_signatures": [
                {"route_id": route_id, "calls": calls, "grid": grid, "workgroup": workgroup,
                 "lds_bytes": lds_bytes}
                for route_id, calls, grid, workgroup, lds_bytes in sorted(declared_exclusions)],
        }
    hypothesis_portfolio.validate_template_authorability(
        portfolio, templates.version, surfaces)
    return surfaces


def _portfolio_dispatch_authority(
        templates: ExperimentTemplateRegistry,
        portfolio: hypothesis_portfolio.Portfolio) -> dict[str, list[dict[str, Any]]]:
    """Bind eligible shorthand geometry to exact native rocprofv3 literals."""
    aggregates: dict[tuple[str, int, int, int], int] = {}
    dispatches = evidence._load_dispatches(
        _PROFILE_V3_TRACE_CSV,
        profiler_trace_schema_id=evidence.ROCPROF_V3_TRACE_ID,
        expected_rows=59925)
    evidence._load_rocprofv3_agent_info(
        _PROFILE_V3_AGENT_CSV,
        trace_agent_ids={int(row["agent_id"]) for row in dispatches})
    for row in dispatches:
        identity = (str(row["kernel"]), int(row["grid"]),
                    int(row["workgroup"]), int(row["lds"]))
        aggregates[identity] = aggregates.get(identity, 0) + 1
    result: dict[str, list[dict[str, Any]]] = {}
    for record in portfolio.eligible_hypotheses():
        template_id = record["current_bundle_eligibility"]["template_ids"][0]
        template = templates.templates[template_id]
        current = next(anchor for anchor in record["dispatch_anchors"]
                       if anchor["frame_id"] == portfolio.body["current_bundle"]["frame_id"])
        rows: list[dict[str, Any]] = []
        anchor_by_route = {row.signature: row for row in template.dispatch.anchor_exact}
        for expected in current["signatures"]:
            contract = anchor_by_route.get(expected["route_id"])
            if (contract is None or (contract.calls, contract.grid, contract.workgroup,
                                     contract.lds_bytes) != tuple(
                    expected[key] for key in ("calls", "grid", "workgroup", "lds_bytes"))):
                raise DeploymentFactoryError(
                    f"{record['hypothesis_id']} route differs from deployed template")
            matches = []
            for (name, grid, workgroup, lds_bytes), calls in aggregates.items():
                if ((calls, grid, workgroup, lds_bytes) != tuple(
                        expected[key] for key in ("calls", "grid", "workgroup", "lds_bytes"))):
                    continue
                if re.fullmatch(contract.kernel_pattern, name):
                    matches.append(name)
            if len(matches) != 1:
                raise DeploymentFactoryError(
                    f"{record['hypothesis_id']} geometry lacks one exact raw trace literal")
            rows.append({"route_id": expected["route_id"],
                         "kernel_name": matches[0],
                         **{key: expected[key] for key in
                            ("calls", "grid", "workgroup", "lds_bytes")}})
        if len({tuple(row.items()) for row in rows}) != len(rows):
            raise DeploymentFactoryError(
                f"{record['hypothesis_id']} repeats an exact raw dispatch row")
        result[record["hypothesis_id"]] = rows
    return result


def _target_source_equality_receipt(config: deployment.DiscoveryDeployment) -> tuple[Path, str]:
    rows = {}
    for relative, expected in sorted(_TARGET_SOURCE_SHA256.items()):
        production_bytes = subprocess.run(
            ("git", "-C", str(config.production_path), "show",
             f"{config.production_head}:{relative}"), check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        instrument_bytes = subprocess.run(
            ("git", "-C", str(config.instrument_path), "show",
             f"{config.instrument_commit}:{relative}"), check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        production_sha = hashlib.sha256(production_bytes.stdout).hexdigest()
        instrument_sha = hashlib.sha256(instrument_bytes.stdout).hexdigest()
        if (production_bytes.returncode or instrument_bytes.returncode
                or production_sha != expected or instrument_sha != expected):
            raise DeploymentFactoryError(
                f"production/instrument reviewed target differs: {relative}")
        rows[relative] = {"production_blob_sha256": production_sha,
                          "instrument_blob_sha256": instrument_sha, "equal": True}
    body = {"schema": "epyc.autokernel.instrument_target_equality.v1",
            "production_commit": config.production_head,
            "instrument_commit": config.instrument_commit, "targets": rows}
    body["receipt_sha256"] = schemas.content_hash(body)
    raw = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
    path = config.state_root / "instrument-target-equality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise DeploymentFactoryError("instrument target equality receipt changed")
    else:
        path.write_bytes(raw)
    return path.resolve(), hashlib.sha256(raw).hexdigest()


def _instrument_review_receipt(config: deployment.DiscoveryDeployment) -> tuple[Path, str]:
    """Revalidate the exact reviewed measurement-instrument delta and barrier."""
    if (config.instrument_commit != _INSTRUMENT_COMMIT
            or config.instrument_branch != _INSTRUMENT_BRANCH):
        raise DeploymentFactoryError("deployment selected an unreviewed measurement instrument")
    command = ("git", "-C", str(config.instrument_path))
    diff = subprocess.run(
        (*command, "diff", "--binary", f"{config.production_head}..{config.instrument_commit}"),
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    names = subprocess.run(
        (*command, "diff", "--name-only", "-z",
         f"{config.production_head}..{config.instrument_commit}"),
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    observed_paths = frozenset(
        item.decode("utf-8", "strict") for item in names.stdout.split(b"\0") if item)
    if (diff.returncode or names.returncode
            or hashlib.sha256(diff.stdout).hexdigest() != _INSTRUMENT_DIFF_SHA256
            or observed_paths != _INSTRUMENT_DIFF_PATHS):
        raise DeploymentFactoryError("measurement instrument differs from its reviewed delta")
    expected_blobs = {
        "tests/test-backend-ops.cpp": _INSTRUMENT_TEST_SOURCE_SHA256,
        "tests/test-autokernel-ready-continue-contract.py": _READY_CONTINUE_CONTRACT_SHA256,
        "tools/llama-bench/llama-bench.cpp": _INSTRUMENT_BENCH_SOURCE_SHA256,
        "tools/llama-bench/README.md": _INSTRUMENT_BENCH_README_SHA256,
    }
    blobs: dict[str, str] = {}
    for relative, expected in sorted(expected_blobs.items()):
        result = subprocess.run(
            (*command, "show", f"{config.instrument_commit}:{relative}"),
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        actual = hashlib.sha256(result.stdout).hexdigest()
        if result.returncode or actual != expected:
            raise DeploymentFactoryError(f"reviewed instrument blob changed: {relative}")
        blobs[relative] = actual
    body = {
        "schema": "epyc.autokernel.measurement_instrument_review.v1",
        "production_base_commit": config.production_head,
        "instrument_branch": config.instrument_branch,
        "instrument_commit": config.instrument_commit,
        "instrument_diff_sha256": _INSTRUMENT_DIFF_SHA256,
        "reviewed_diff_paths": sorted(_INSTRUMENT_DIFF_PATHS),
        "reviewed_blobs": blobs,
        "ready_continue_capability": {
            "schema": "epyc.autokernel.ready_continue.v1",
            "source": "tests/test-autokernel-ready-continue-contract.py",
            "source_sha256": _READY_CONTINUE_CONTRACT_SHA256,
            "instrument_commit": _INSTRUMENT_COMMIT,
        },
        "backend_ops_property_capability": {
            "schema": "epyc.autokernel.backend_ops_property_capability_source.v1",
            "source": "tests/test-backend-ops.cpp",
            "source_sha256": _INSTRUMENT_TEST_SOURCE_SHA256,
            "suite_seed": discovery_static_registry._CORRECTNESS_CAPABILITY_SEED,
            "argv_suffix": list(t0_provider.backend_ops_property_self_test_argv(
                "test-backend-ops",
                discovery_static_registry._CORRECTNESS_CAPABILITY_SEED)[1:]),
            "expected_stderr": (
                "AUTOKERNEL_PROPERTY_SELF_TEST suite_seed=2026081301 "
                "sensitivity=1.000 specificity=1.000 planted=5 clean=5\n"),
        },
    }
    body["receipt_sha256"] = schemas.content_hash(body)
    raw = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
    path = config.state_root / "instrument-review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise DeploymentFactoryError("instrument review receipt changed")
    else:
        path.write_bytes(raw)
    return path.resolve(), hashlib.sha256(raw).hexdigest()


def static_template_registry_sha256() -> str:
    """Public value for authoring a sealed config; it grants no constructor authority."""
    return _template_registry().registry_sha256


def _manifest_carrier_bytes(candidate: controller.PlannedCandidate) -> bytes:
    raw = source_candidate.source_patch_manifest_bytes(candidate.source_manifest)
    if hashlib.sha256(raw).hexdigest() != candidate.source_manifest_sha256:
        raise DeploymentFactoryError("candidate manifest canonical carrier hash mismatch")
    return raw


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_uid, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _validate_directory(value: os.stat_result, label: str, *, private: bool) -> None:
    if (not stat.S_ISDIR(value.st_mode) or value.st_uid != os.geteuid()
            or (private and stat.S_IMODE(value.st_mode) & 0o077)):
        qualifier = "private owner directory" if private else "owner directory"
        raise DeploymentFactoryError(f"{label} is not a real {qualifier}")


@contextlib.contextmanager
def _pinned_operation_directory(config: deployment.DiscoveryDeployment,
                                operation_key: str):
    if not isinstance(operation_key, str) or not controller.HASH.fullmatch(operation_key):
        raise DeploymentFactoryError("operation carrier requires an exact operation key")
    operations_root = Path(config.operations_root)
    if not operations_root.is_absolute():
        raise DeploymentFactoryError("operations root must be absolute")
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        root_descriptor = os.open(operations_root, flags)
    except OSError as exc:
        raise DeploymentFactoryError("operations root cannot be pinned") from exc
    operation_descriptor = None
    try:
        root_identity = os.fstat(root_descriptor)
        _validate_directory(root_identity, "operations root", private=False)
        try:
            operation_descriptor = os.open(
                operation_key, flags, dir_fd=root_descriptor)
        except OSError as exc:
            raise DeploymentFactoryError("operation carrier root cannot be pinned") from exc
        operation_identity = os.fstat(operation_descriptor)
        _validate_directory(operation_identity, "operation carrier root", private=True)
        try:
            entry = os.stat(operation_key, dir_fd=root_descriptor,
                            follow_symlinks=False)
        except OSError as exc:
            raise DeploymentFactoryError(
                "operation carrier root entry changed while pinning") from exc
        if _directory_identity(entry) != _directory_identity(operation_identity):
            raise DeploymentFactoryError("operation carrier root entry changed while pinning")
        yield (operations_root, root_descriptor, root_identity,
               operation_descriptor, operation_identity)
    finally:
        if operation_descriptor is not None:
            os.close(operation_descriptor)
        os.close(root_descriptor)


def _verify_operation_chain(operations_root: Path, root_descriptor: int,
                            root_identity: os.stat_result, operation_key: str,
                            operation_descriptor: int,
                            operation_identity: os.stat_result) -> None:
    current_root = os.fstat(root_descriptor)
    current_operation = os.fstat(operation_descriptor)
    try:
        root_entry = os.stat(operations_root, follow_symlinks=False)
        operation_entry = os.stat(
            operation_key, dir_fd=root_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise DeploymentFactoryError("operation carrier parent chain changed") from exc
    if (_directory_identity(current_root) != _directory_identity(root_identity)
            or _directory_identity(current_operation) != _directory_identity(operation_identity)
            or _directory_identity(root_entry) != _directory_identity(root_identity)
            or _directory_identity(operation_entry) != _directory_identity(operation_identity)):
        raise DeploymentFactoryError("operation carrier parent chain changed")


def _operation_carrier_root(config: deployment.DiscoveryDeployment,
                            operation_key: str) -> Path:
    with _pinned_operation_directory(config, operation_key) as pinned:
        operations_root, root_fd, root_identity, operation_fd, operation_identity = pinned
        _verify_operation_chain(
            operations_root, root_fd, root_identity, operation_key,
            operation_fd, operation_identity)
        return operations_root / operation_key


def _read_operation_carrier(descriptor: int, label: str) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o077):
        raise DeploymentFactoryError(
            f"{label} is not a private, single-link regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    size = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
        size += len(block)
    after = os.fstat(descriptor)
    if _file_identity(before) != _file_identity(after) or size != after.st_size:
        raise DeploymentFactoryError(f"{label} changed while it was reopened")
    return b"".join(chunks), after


def _write_operation_carrier(config: deployment.DiscoveryDeployment,
                             operation_key: str, name: str, raw: bytes,
                             label: str) -> Path:
    if name not in {"source-manifest.json", "evidence-policy.json"}:
        raise DeploymentFactoryError("operation carrier name is not allowlisted")
    with _pinned_operation_directory(config, operation_key) as pinned:
        operations_root, root_fd, root_identity, operation_fd, operation_identity = pinned
        flags = (os.O_RDWR | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=operation_fd)
        except FileExistsError:
            try:
                descriptor = os.open(
                    name, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0), dir_fd=operation_fd)
            except OSError as exc:
                raise DeploymentFactoryError(f"{label} cannot be reopened safely") from exc
            created = False
        except OSError as exc:
            raise DeploymentFactoryError(f"{label} cannot be sealed safely") from exc
        else:
            created = True
        try:
            if created:
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise DeploymentFactoryError(f"{label} could not be sealed completely")
                    view = view[written:]
                os.fsync(descriptor)
                os.fsync(operation_fd)
            actual, file_identity = _read_operation_carrier(descriptor, label)
            if actual != raw:
                qualifier = "changed after it was sealed" if created else \
                    "already exists with different bytes"
                raise DeploymentFactoryError(f"{label} {qualifier}")
            try:
                entry = os.stat(name, dir_fd=operation_fd, follow_symlinks=False)
            except OSError as exc:
                raise DeploymentFactoryError(f"{label} directory entry changed") from exc
            if (_file_identity(entry) != _file_identity(file_identity)
                    or entry.st_nlink != 1):
                raise DeploymentFactoryError(f"{label} directory entry changed")
            _verify_operation_chain(
                operations_root, root_fd, root_identity, operation_key,
                operation_fd, operation_identity)
            # This is the final namespace read before returning a path-based
            # binding: it must still name the exact inode held by descriptor.
            try:
                final_entry = os.stat(
                    name, dir_fd=operation_fd, follow_symlinks=False)
            except OSError as exc:
                raise DeploymentFactoryError(
                    f"{label} final directory entry changed") from exc
            if (_file_identity(final_entry) != _file_identity(file_identity)
                    or final_entry.st_nlink != 1):
                raise DeploymentFactoryError(f"{label} final directory entry changed")
            _verify_operation_chain(
                operations_root, root_fd, root_identity, operation_key,
                operation_fd, operation_identity)
            return operations_root / operation_key / name
        finally:
            os.close(descriptor)


def _manifest_file_for_operation(config: deployment.DiscoveryDeployment,
                                 candidate: controller.PlannedCandidate,
                                 operation_key: str) -> evidence.BoundInputFile:
    raw = _manifest_carrier_bytes(candidate)
    path = _write_operation_carrier(
        config, operation_key, "source-manifest.json", raw, "source manifest carrier")
    return evidence.BoundInputFile("manifest", path, candidate.source_manifest_sha256)


def _manifest_file(config: deployment.DiscoveryDeployment,
                   candidate: controller.PlannedCandidate,
                   build: controller.GpuSourceBuild) -> evidence.BoundInputFile:
    if build.operation_key is None:
        raise DeploymentFactoryError("source build lacks an operation key")
    return _manifest_file_for_operation(config, candidate, build.operation_key)


def _evidence_binding(config: deployment.DiscoveryDeployment) -> EvidencePlanBinding:
    profiler_policy = _rocprof_v3_policy(config)
    policy_by_role = {item.role: item for item in profiler_policy}
    python = policy_by_role["executable"]
    profiler = policy_by_role["profiler_wrapper"]
    def build(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild,
              template: ExperimentTemplate, repetition: int = 1) -> evidence.GpuSourceEvidencePlan:
        if build_.materialization_receipt is None or build_.operation_key is None:
            raise DeploymentFactoryError("source build lacks materialization identity")
        identities = discovery_static_registry.evidence_identity_files_for_build(
            build_, manifest=_manifest_file(config, candidate, build_),
            model=evidence.BoundInputFile("model", config.model.path, config.model.sha256),
            workload=evidence.BoundInputFile("workload", config.workload.path,
                                             config.workload.sha256),
            runtime_config=evidence.BoundInputFile(
                "runtime_config", config.runtime_config.path, config.runtime_config.sha256))
        if identities.shared_runtime is None:
            raise DeploymentFactoryError("source evidence lacks a shared reward runtime")
        # Revalidate the profiler closure at the same binding
        # boundary as the source/runtime carriers.
        if any(_digest_regular(item.path, item.role) != item.sha256
               for item in profiler_policy):
            raise DeploymentFactoryError(
                "rocprofv3 policy changed before evidence binding")
        for item in profiler_policy:
            if item.role.endswith("_manifest"):
                try:
                    evidence._verify_profiler_runtime_manifest(item)
                except evidence.EvidenceProducerError as exc:
                    raise DeploymentFactoryError(
                        "rocprofv3 closure changed before evidence binding") from exc
        try:
            correctness_tool, capability_receipt = (
                discovery_static_registry.correctness_capability_files_for_build(
                    build_, arm="candidate"))
        except discovery_static_registry.StaticRegistryError as exc:
            raise DeploymentFactoryError(
                "candidate correctness tool lacks a sealed passing property self-test") from exc
        semantics = template.semantics
        op = semantics.get("correctness_op")
        cases = semantics.get("expected_correctness_cases")
        seed = semantics.get("suite_seed")
        if (not isinstance(op, str) or not isinstance(cases, int)
                or not isinstance(seed, int)):
            raise DeploymentFactoryError("template correctness semantics are malformed")
        correctness_argv = (
            str(correctness_tool.path), "test", "-o", op, "-b", "ROCm0", "-j", "1",
            "--suite-seed", str(seed))
        correctness_invocations: tuple[Mapping[str, Any], ...] = ()
        if template.template_id == "cuda-fattn-gqa7-common-v1":
            required_cases = [dict(row) for row in semantics["required_correctness_cases"]]
            correctness_invocations = (
                {"invocation_id": "generic_flash_attn_ext",
                 "argv": list(correctness_argv), "backend": "ROCm0",
                 "op": op, "case_set": "generic_flash_attn_ext_v1",
                 "expected_cases": cases, "required_cases": []},
                {"invocation_id": "odd_gqa7_d64_q1",
                 "argv": [*correctness_argv, "-p",
                          "hsk=64,hsv=64,nh=2,nr23=[7,1]"],
                 "backend": "ROCm0", "op": op,
                 "case_set": "odd_gqa7_d64_q1_v1",
                 "expected_cases": len(required_cases),
                 "required_cases": required_cases,
                 "environment_overrides": [
                     ["AUTOKERNEL_CORRECTNESS_CASE_SET",
                      "odd_gqa7_d64_q1_v1"]]},
            )
        shared = identities.shared_runtime
        reward_binary = shared.measurement_binary
        profile_argv = (
            str(python.path), str(profiler.path), "--kernel-trace",
            "-d", evidence.ROCPROF_OUTPUT_DIRECTORY,
            "-o", evidence.ROCPROF_OUTPUT_BASENAME,
            "--output-format", "csv", "--",
            "/usr/bin/taskset", "-c", "184-191", str(reward_binary.path),
            "-m", str(config.model.path), "-p", "0", "-n", "128", "-r", "1",
            "-ngl", "99", "-fa", "on", "-t", "8", "-o", "json")
        common_environment = (
            ("GGML_CUDA_DISABLE_GRAPHS", "1"), ("HIP_VISIBLE_DEVICES", "0"),
            ("PATH", f"{_ROCPROF_V3_SDK / 'bin'}:/opt/rocm/bin:/usr/bin:/bin"),
            ("ROCM_PATH", "/opt/rocm"))
        def profile_environment(hip: evidence.BoundInputFile) -> tuple[tuple[str, str], ...]:
            return tuple(sorted((*common_environment, ("LD_LIBRARY_PATH",
                f"{hip.path.parent}:{reward_binary.path.parent}:"
                f"{_ROCPROF_V3_SDK / 'lib'}:{_ROCPROF_V3_OLD_LIB}:"
                f"{_ROCPROF_V3_PCI_LIB_DIR}:/opt/rocm/lib"))))
        carrier_root = _operation_carrier_root(config, build_.operation_key)
        order_seed, order_text = _arm_order_schedule(
            deployment_config_sha256=config.config_sha256,
            source_manifest_sha256=candidate.source_manifest_sha256,
            repetition=repetition)
        attribution_arm_order = tuple(order_text.split(","))
        placeholder = evidence.BoundInputFile(
            "execution_policy", carrier_root / "evidence-policy.json", "0" * 64)
        dispatch = template.bind_dispatch(candidate.experiment_intent)
        candidate_rows, anchor_rows = _expected_rocprofv3_rows(dispatch)
        provisional = evidence.GpuSourceEvidencePlan(
            campaign_id=candidate.source_manifest.campaign_id,
            device_id=config.device_id,
            manifest_sha256=candidate.source_manifest_sha256,
            model_sha256=config.model.sha256,
            workload_sha256=config.workload.sha256,
            runtime_config_sha256=config.runtime_config.sha256,
            candidate=build_.candidate_identity, anchor=build_.anchor_identity,
            correctness_argv=correctness_argv,
            correctness_backend="ROCm0",
            correctness_op=op,
            expected_correctness_cases=cases,
            candidate_rocprof_argv=profile_argv, anchor_rocprof_argv=profile_argv,
            dispatch=dispatch,
            identity_files=identities, policy=placeholder,
            correctness_inputs=(correctness_tool, capability_receipt,
                                identities.candidate.binary,
                                identities.candidate.config, identities.candidate.linkage),
            candidate_rocprof_inputs=(*profiler_policy, reward_binary,
                                      identities.model, identities.workload,
                                      identities.runtime_config),
            anchor_rocprof_inputs=(*profiler_policy, reward_binary,
                                   identities.model, identities.workload,
                                   identities.runtime_config),
            required_correctness_argv_paths=(correctness_tool.path,),
            required_candidate_rocprof_argv_paths=(reward_binary.path, identities.model.path),
            required_anchor_rocprof_argv_paths=(reward_binary.path, identities.model.path),
            execution_cwd=build_.candidate_build.resolve(strict=True),
            correctness_environment=tuple(sorted((
                ("AUTOKERNEL_CORRECTNESS_CASE_SET", ""),
                ("GGML_CUDA_DISABLE_GRAPHS", "1"), ("HIP_VISIBLE_DEVICES", "0"),
                ("LD_LIBRARY_PATH",
                 f"{identities.candidate.hip_library.path.parent}:/opt/rocm/lib"),
                ("PATH", "/opt/rocm/bin:/usr/bin:/bin"), ("ROCM_PATH", "/opt/rocm")))),
            candidate_rocprof_environment=profile_environment(shared.candidate_hip_library),
            anchor_rocprof_environment=profile_environment(shared.anchor_hip_library),
            shared_runtime=shared,
            correctness_invocations=correctness_invocations,
            attribution_arm_order_seed_sha256=order_seed,
            attribution_arm_order=attribution_arm_order,
            profiler_trace_schema_id=evidence.ROCPROF_V3_TRACE_ID,
            expected_candidate_profiler_dispatch_rows=candidate_rows,
            expected_anchor_profiler_dispatch_rows=anchor_rows,
            profiler_transport_policy=evidence.ROCPROF_V3_TRANSPORT_POLICY)
        raw = json.dumps(evidence._policy_payload(provisional), sort_keys=True,
                         separators=(",", ":")).encode()
        policy_path = _write_operation_carrier(
            config, build_.operation_key, "evidence-policy.json", raw,
            "sealed evidence policy")
        policy = evidence.BoundInputFile(
            "execution_policy", policy_path,
            hashlib.sha256(raw).hexdigest())
        return replace(provisional, policy=policy)
    return EvidencePlanBinding(build=build)


def _expected_rocprofv3_rows(
        dispatch: evidence.DispatchContract) -> tuple[int, int]:
    """Derive each arm's exact cardinality from the sealed anchor trace."""
    anchor_rows = 59_925
    candidate_rows = (anchor_rows
                      + sum(row.calls for row in dispatch.candidate_exact)
                      - sum(row.calls for row in dispatch.anchor_exact))
    if candidate_rows < 1:
        raise DeploymentFactoryError(
            "candidate profiler cardinality derivation is invalid")
    return candidate_rows, anchor_rows


def _arm_order_schedule(*, deployment_config_sha256: str,
                        source_manifest_sha256: str,
                        repetition: int) -> tuple[str, str]:
    if (not all(isinstance(value, str) and controller.HASH.fullmatch(value)
                for value in (deployment_config_sha256, source_manifest_sha256))
            or repetition not in {1, 2}):
        raise DeploymentFactoryError("arm-order schedule authority is malformed")
    seed = schemas.content_hash({
        "schema": "epyc.autokernel.discovery_arm_order.v1",
        "deployment_config_sha256": deployment_config_sha256,
        "source_manifest_sha256": source_manifest_sha256})
    anchor_first_s1 = int(seed[-1], 16) % 2 == 0
    anchor_first = anchor_first_s1 if repetition == 1 else not anchor_first_s1
    return seed, ("anchor,candidate" if anchor_first else "candidate,anchor")


def _runner_binding(config: deployment.DiscoveryDeployment) -> RunnerArgsBinding:
    def build(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild,
              permit: Mapping[str, Any]) -> Any:
        operation_key = permit.get("operation_key")
        repetition = permit.get("repetition")
        if operation_key != build_.operation_key or repetition not in {1, 2}:
            raise DeploymentFactoryError("runner operation identity differs from sealed build")
        stage_root = config.operations_root / str(operation_key) / "runner" / f"s{repetition}"
        decision = permit.get("load_admission")
        if not isinstance(decision, Mapping):
            raise DeploymentFactoryError("runner permit lacks sealed load-admission decision")
        corpus = config.admission_policy.corpus
        effective = schemas.content_hash({
            "planner_context_sha256": config.planner_context.value["context_sha256"],
            "admission_policy_sha256": corpus.policy_sha256,
            "admission_policy_version": corpus.version})
        gpu_load_admission.validate_decision_receipt(
            decision, expected_policy_version=corpus.version,
            expected_policy_sha256=corpus.policy_sha256,
            expected_policy_file_sha256=corpus.file_sha256,
            expected_effective_context_sha256=effective)
        decision_path = stage_root / "load-admission-decision.json"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_raw = (json.dumps(dict(decision), sort_keys=True, indent=2) + "\n").encode()
        if decision_path.exists():
            if decision_path.is_symlink() or decision_path.read_bytes() != decision_raw:
                raise DeploymentFactoryError("runner load-admission carrier changed")
        else:
            decision_path.write_bytes(decision_raw)
        schedule_seed, arm_order = _arm_order_schedule(
            deployment_config_sha256=config.config_sha256,
            source_manifest_sha256=candidate.source_manifest_sha256,
            repetition=repetition)
        common_argv = ["--anchor-build", str(build_.anchor_build), "--candidate-build", str(build_.candidate_build),
                "--model", str(config.model.path),
                "--campaign-id", f"ak-discovery-{config.config_sha256[:16]}",
                "--factor", "source_patch", "--calls", "9", "--workload", "decode_tg128",
                "--arm-order-schedule", arm_order,
                "--arm-order-seed-sha256", schedule_seed,
                "--inference-window-lock", str(config.inference_window_lock),
                "--load-admission-decision", str(decision_path),
                "--load-admission-policy", str(config.admission_policy.input.path),
                "--load-admission-policy-sha256", config.admission_policy.input.sha256,
                "--effective-context-sha256", effective, "--device-id", config.device_id,
                "--measurement-binary", str(build_.measurement_binary),
                "--common-loader-dir", str(build_.common_loader_dir),
                "--anchor-loader-dir", str(build_.anchor_loader_dir),
                "--candidate-loader-dir", str(build_.candidate_loader_dir),
                "--cpu-claim-journal", str(config.operations_root / "claims" / "cpu.jsonl"),
                "--device-claim-journal", str(config.operations_root / "claims" / "device.jsonl")]
        try:
            graphs_off = controller.gpu_discovery.parser().parse_args([
                *common_argv, "--output-dir",
                str(stage_root / "measurement-graphs-off"),
                "--runtime-graphs", "off"])
            graphs_on = controller.gpu_discovery.parser().parse_args([
                *common_argv, "--output-dir",
                str(stage_root / "target-runtime-graphs-on"),
                "--runtime-graphs", "on"])
            production_graphs_on = None
            if getattr(candidate, "composition_plan", None) is not None:
                authority = getattr(
                    build_, "composition_production_authority", None)
                if authority is None:
                    raise DeploymentFactoryError(
                        "cumulative build lacks frozen-production authority")
                production_graphs_on = \
                    controller.gpu_discovery.parser().parse_args([
                        "--anchor-build",
                        str(config.production_path / "build-hip"),
                        "--candidate-build", str(build_.candidate_build),
                        "--model", str(config.model.path),
                        "--campaign-id",
                        f"ak-discovery-{config.config_sha256[:16]}",
                        "--factor", "cumulative_production",
                        "--calls", "9", "--workload", "decode_tg128",
                        "--arm-order-schedule", arm_order,
                        "--arm-order-seed-sha256", schedule_seed,
                        "--inference-window-lock",
                        str(config.inference_window_lock),
                        "--load-admission-decision", str(decision_path),
                        "--load-admission-policy",
                        str(config.admission_policy.input.path),
                        "--load-admission-policy-sha256",
                        config.admission_policy.input.sha256,
                        "--effective-context-sha256", effective,
                        "--device-id", config.device_id,
                        "--cpu-claim-journal",
                        str(config.operations_root / "claims" / "cpu.jsonl"),
                        "--device-claim-journal",
                        str(config.operations_root / "claims" / "device.jsonl"),
                        "--output-dir",
                        str(stage_root /
                            "cumulative-vs-production-graphs-on"),
                        "--runtime-graphs", "on"])
        except SystemExit as exc:
            # argparse is allowed to reject an invalid sealed contract, but it
            # must not terminate the unified controller process.  This typed
            # interruption leaves the already-sealed proof operation resumable.
            raise controller.ResumableScreenInterruption(
                f"governed runner argv parser refused with exit {exc.code}") from exc
        sealed_identities = {
            "anchor": dict(build_.anchor_identity.__dict__),
            "candidate": dict(build_.candidate_identity.__dict__),
        }
        for current in (graphs_off, graphs_on):
            setattr(current, "_operation_key", operation_key)
            setattr(current, "_operations_root", str(config.operations_root))
            setattr(current, "_operation_repetition", repetition)
            for arm, identity in sealed_identities.items():
                setattr(current, f"_sealed_{arm}_source_build_identity", identity)
        setattr(graphs_off, "_target_runtime_args", graphs_on)
        if production_graphs_on is not None:
            authority = getattr(
                build_, "composition_production_authority", None)
            assert authority is not None
            setattr(production_graphs_on, "_operation_key", operation_key)
            setattr(production_graphs_on, "_operations_root",
                    str(config.operations_root))
            setattr(production_graphs_on, "_operation_repetition", repetition)
            setattr(production_graphs_on,
                    "_sealed_anchor_source_build_identity",
                    dict(authority.build_identity.__dict__))
            setattr(production_graphs_on,
                    "_sealed_candidate_source_build_identity",
                    dict(build_.candidate_identity.__dict__))
            setattr(production_graphs_on,
                    "_frozen_production_authority", authority.to_dict())
            setattr(graphs_off, "_production_graphs_on_args",
                    production_graphs_on)
        return graphs_off
    return RunnerArgsBinding(build=build)


def _bind_runner_runtime_authority(
        config: deployment.DiscoveryDeployment,
        build_: controller.GpuSourceBuild,
        permit: Mapping[str, Any], result: Any) -> Any:
    """Install lease and source-build authority on both runner namespaces."""
    decision = permit.get("load_admission")
    if not isinstance(decision, Mapping):
        raise DeploymentFactoryError(
            "runner invocation lacks the sealed lease admission decision")
    decision = dict(decision)
    expected_admission = {
        "load_admission_decision": decision,
        "load_admission_policy_version": config.admission_policy.corpus.version,
        "load_admission_policy_sha256": config.admission_policy.corpus.policy_sha256,
        "load_admission_policy_file_sha256": config.admission_policy.corpus.file_sha256,
        "load_admission_effective_context_sha256": decision.get(
            "effective_context_sha256"),
    }
    target = getattr(result, "_target_runtime_args", None)
    if target is None:
        raise DeploymentFactoryError(
            "runner arguments lack target-runtime graphs-on stage")
    expected_build_identities = {
        "anchor": dict(build_.anchor_identity.__dict__),
        "candidate": dict(build_.candidate_identity.__dict__),
    }
    for current in (result, target):
        if getattr(current, "_operation_key", None) != build_.operation_key:
            raise DeploymentFactoryError(
                "runner arguments changed private operation identity")
        if (getattr(current, "_operations_root", None)
                != str(config.operations_root)
                or getattr(current, "_operation_repetition", None)
                != permit.get("repetition")):
            raise DeploymentFactoryError(
                "runner arguments changed private operation namespace")
        for key, value in expected_admission.items():
            existing = getattr(current, key, None)
            if existing is not None and existing != value:
                raise DeploymentFactoryError(
                    f"runner arguments attempted to override {key}")
            try:
                setattr(current, key, value)
            except (AttributeError, TypeError) as exc:
                raise DeploymentFactoryError(
                    "runner arguments cannot carry sealed load admission") from exc
        for arm, identity in expected_build_identities.items():
            if getattr(
                    current,
                    f"_sealed_{arm}_source_build_identity", None) != identity:
                raise DeploymentFactoryError(
                    "runner arguments changed sealed source build identity")
        if (getattr(current, "factor", None) != "source_patch"
                or str(getattr(current, "model", "")) != str(config.model.path)
                or str(getattr(current, "anchor_build", "")) != str(build_.anchor_build)
                or str(getattr(current, "candidate_build", "")) != str(build_.candidate_build)
                or str(getattr(current, "measurement_binary", "")) != str(build_.measurement_binary)
                or str(getattr(current, "common_loader_dir", "")) != str(build_.common_loader_dir)
                or str(getattr(current, "anchor_loader_dir", "")) != str(build_.anchor_loader_dir)
                or str(getattr(current, "candidate_loader_dir", "")) != str(build_.candidate_loader_dir)
                or getattr(current, "promotion_claim", False) is not False
                or str(getattr(current, "inference_window_lock", "")) != str(config.inference_window_lock)
                or getattr(current, "load_admission_decision", None) != decision):
            raise DeploymentFactoryError(
                "runner arguments do not bind source builds/model/window/discovery authority")
    production = getattr(result, "_production_graphs_on_args", None)
    authority = getattr(build_, "composition_production_authority", None)
    if authority is None:
        if production is not None:
            raise DeploymentFactoryError(
                "ordinary runner acquired frozen-production authority")
        return result
    if production is None:
        raise DeploymentFactoryError(
            "cumulative runner lacks frozen-production graphs-on arguments")
    if (getattr(production, "_operation_key", None) != build_.operation_key
            or getattr(production, "_operations_root", None) !=
               str(config.operations_root)
            or getattr(production, "_operation_repetition", None) !=
               permit.get("repetition")
            or getattr(production, "factor", None) !=
               "cumulative_production"
            or getattr(production, "runtime_graphs", None) != "on"
            or str(getattr(production, "model", "")) !=
               str(config.model.path)
            or str(getattr(production, "anchor_build", "")) !=
               str(config.production_path / "build-hip")
            or str(getattr(production, "candidate_build", "")) !=
               str(build_.candidate_build)
            or str(getattr(production, "inference_window_lock", "")) !=
               str(config.inference_window_lock)
            or getattr(production, "promotion_claim", False) is not False
            or getattr(production, "_frozen_production_authority", None) !=
               authority.to_dict()):
        raise DeploymentFactoryError(
            "production runner arguments changed comparator authority")
    for arm, identity in {
            "anchor": dict(authority.build_identity.__dict__),
            "candidate": dict(build_.candidate_identity.__dict__),
    }.items():
        if getattr(
                production,
                f"_sealed_{arm}_source_build_identity", None) != identity:
            raise DeploymentFactoryError(
                "production runner changed sealed build identity")
    for key, value in expected_admission.items():
        existing = getattr(production, key, None)
        if existing is not None and existing != value:
            raise DeploymentFactoryError(
                f"production runner attempted to override {key}")
        try:
            setattr(production, key, value)
        except (AttributeError, TypeError) as exc:
            raise DeploymentFactoryError(
                "production runner cannot carry sealed load admission") from exc
    return result


def _static_registry(config: deployment.DiscoveryDeployment,
                     templates: ExperimentTemplateRegistry) -> Mapping[str, Mapping[str, object]]:
    if config.experiment_template_registry_sha256 != templates.registry_sha256:
        raise DeploymentFactoryError("deployment does not bind the checked-in template registry")
    selected = {"environment_profile": config.environment_profile_id,
                "source_builder": config.source_builder_id, "evidence_plan": config.evidence_plan_id,
                "runner_args": config.runner_args_id,
                "experiment_template_registry": config.experiment_template_registry_id,
                "inference_window_lease": config.inference_window_lease_id,
                "production_snapshot": config.production_snapshot_id}
    if selected != dict(_STATIC_IDS):
        raise DeploymentFactoryError("deployment selected a non-allowlisted constructor ID")
    profiles = [profile for profile in config.admission_policy.corpus.profiles
                if (profile.model_sha256 == config.model.sha256
                    and profile.model_path == str(config.model.path)
                    and profile.model_bytes == config.model.path.stat().st_size
                    and profile.workload == "decode_tg128" and profile.calls_per_arm == 9
                    and profile.device_id == config.device_id)]
    if len(profiles) != 1:
        raise DeploymentFactoryError("sealed admission corpus lacks one exact runner profile")
    environment = EnvironmentProfile(_SAFE_ACTOR_ENVIRONMENT)
    comparator = _load_frozen_production_comparator(
        config.frozen_production_comparator.path)
    snapshot_files, snapshot_semantics = _production_runtime_snapshot(
        config.production_path)
    _verify_frozen_production_comparator(
        comparator, config.production_path, snapshot_semantics,
        model_path=config.model.path,
        workload_sha256=config.workload.sha256,
        runtime_config_sha256=config.runtime_config.sha256)
    source_builder = discovery_static_registry.StaticGpuSourceBuilder(
        production_path=config.production_path,
        production_branch=deployment.FROZEN_PRODUCTION_BRANCH,
        instrument_path=config.instrument_path,
        operations_root=config.operations_root,
        build_root=config.build_root,
        cmake_defines=(("GGML_HIP", "ON"), ("AMDGPU_TARGETS", "gfx90a"),
                       ("GGML_NATIVE", "OFF")),
        composition_production_authority=comparator.authority())
    snapshot = ProductionSnapshotBinding(
        config.production_path, snapshot_files, snapshot_semantics,
        schemas.content_hash(snapshot_semantics))
    return MappingProxyType({
        "environment_profile": MappingProxyType({_STATIC_IDS["environment_profile"]: environment}),
        "source_builder": MappingProxyType({_STATIC_IDS["source_builder"]:
                                               SourceBuilderBinding(source_builder.build)}),
        "evidence_plan": MappingProxyType({_STATIC_IDS["evidence_plan"]: _evidence_binding(config)}),
        "runner_args": MappingProxyType({_STATIC_IDS["runner_args"]: _runner_binding(config)}),
        "experiment_template_registry": MappingProxyType({_STATIC_IDS["experiment_template_registry"]: templates}),
        "inference_window_lease": MappingProxyType({_STATIC_IDS["inference_window_lease"]: InferenceWindowLeaseBinding()}),
        "production_snapshot": MappingProxyType({_STATIC_IDS["production_snapshot"]: snapshot}),
    })


def _seal_graph_receipt(config: deployment.DiscoveryDeployment,
                        planner_runtime: Mapping[str, Any],
                        critic_runtime: Mapping[str, Any],
                        templates: ExperimentTemplateRegistry,
                        target_equality: tuple[Path, str],
                        instrument_review: tuple[Path, str],
                        source_package: controller.ReviewedSourcePackage,
                        execution_modules: Mapping[str, Mapping[str, str]],
                        production_runtime_sha256: str,
                        critic_auth: Mapping[str, Any],
                        carry_forward: Mapping[str, Any]) -> tuple[Path, str]:
    surfaces = _normalized_template_surfaces(
        templates, config.hypothesis_portfolio.value)
    profiler_runtime = [evidence._bound_reference(item)
                        for item in _rocprof_v3_policy(config)]
    historical_q5_evidence = _preauthored_historical_evidence(
        config.preauthored_continuation.value,
        config.hypothesis_evidence_manifest.value["evidence"])
    comparator = _load_frozen_production_comparator(
        config.frozen_production_comparator.path)
    body = {"schema": "epyc.autokernel.static_discovery_graph.v9",
            "authority": "nonpromotable_candidate_only_discovery", "promotion_claim": False,
            "inference_executed": False, "config_sha256": config.config_sha256,
            "registry_ids": dict(_STATIC_IDS), "template_registry_sha256": templates.registry_sha256,
            "template_surfaces": surfaces,
            "template_surfaces_sha256": schemas.content_hash(surfaces),
            "portfolio_dispatch_authority":
                config.planner_context.value["portfolio_dispatch_authority"],
            "portfolio_dispatch_authority_sha256": schemas.content_hash(
                config.planner_context.value["portfolio_dispatch_authority"]),
            "hypothesis_portfolio": {
                "semantic_sha256": config.hypothesis_portfolio.value.sha256,
                "file_sha256": config.hypothesis_portfolio.input.sha256,
                "evidence_manifest_sha256":
                    config.hypothesis_evidence_manifest.value["manifest_sha256"],
                "contract_sha256": config.hypothesis_portfolio_contract.sha256},
            "carry_forward_sha256": carry_forward["carry_forward_sha256"],
            "carry_forward": {
                "schema": deployment.CARRY_FORWARD_SCHEMA,
                "file_sha256": config.carry_forward.input.sha256,
                "self_sha256": config.carry_forward.self_sha256,
                "semantic_sha256": config.carry_forward.semantic_sha256,
            },
            "frozen_production_comparator": {
                "schema": (
                    "epyc.autokernel."
                    "frozen_production_comparator_source.v1"),
                "file_sha256": config.frozen_production_comparator.sha256,
                "receipt_sha256": comparator.receipt_sha256,
            },
            "preauthored_continuation": {
                "schema": preauthored_continuation.SCHEMA,
                "carrier_sha256": config.preauthored_continuation.value.sha256,
                "file_sha256": config.preauthored_continuation.input.sha256,
                "hypothesis_id": config.preauthored_continuation.value.hypothesis_id,
                "template_id": config.preauthored_continuation.value.template_id,
                "patch_sha256": config.preauthored_continuation.value.patch_sha256,
                "source_backed_diff_sha256":
                    config.preauthored_continuation.value.source_backed_diff_sha256,
                "historical_evidence_sha256":
                    historical_q5_evidence["receipt_sha256"],
                "historical_correctness_authority": "provenance_only",
                "modern_governed_correctness_required": True,
            },
            "attribution_expectation_erratum": {
                "schema":
                    "epyc.autokernel.attribution_expectation_erratum_source.v1",
                "erratum_schema": carry_forward[
                    "attribution_expectation_erratum"]["schema"],
                "erratum_sha256": carry_forward[
                    "attribution_expectation_erratum"]["erratum_sha256"],
                "file_sha256": config.q5_lds0_attribution_erratum.sha256,
                "operation_key": carry_forward[
                    "attribution_expectation_erratum"]["operation_key"],
                "attribution_refusal_file_sha256": carry_forward[
                    "attribution_expectation_erratum"][
                        "attribution_refusal_file_sha256"],
                "candidate_semantic_sha256": carry_forward[
                    "attribution_expectation_erratum"][
                        "candidate_semantic_sha256"],
            },
            "reviewed_source_package": source_package.manifest(),
            "profile_trace_authority": {
                "receipt": str(_PROFILE_TRACE_RECEIPT),
                "receipt_sha256": _PROFILE_TRACE_RECEIPT_SHA256,
                "timestamp_csv": str(_PROFILE_TRACE_CSV),
                "timestamp_csv_sha256": _PROFILE_TRACE_CSV_SHA256,
                "v3_kernel_trace": str(_PROFILE_V3_TRACE_CSV),
                "v3_kernel_trace_sha256": _PROFILE_V3_TRACE_CSV_SHA256,
                "v3_agent_info": str(_PROFILE_V3_AGENT_CSV),
                "v3_agent_info_sha256": _PROFILE_V3_AGENT_CSV_SHA256,
                "cross_profiler_projection_sha256":
                    "8bf84656cd12eecf8e9881fd0f2b6f9f8da7e4485a0a668dcb08065e930fbc54"},
            "profiler_runtime_authority": {
                "trace_schema_id": evidence.ROCPROF_V3_TRACE_ID,
                "transport_policy": evidence.ROCPROF_V3_TRANSPORT_POLICY,
                "package": "rocprofiler-sdk 0.4.0-66~20.04 amd64",
                "inputs": profiler_runtime,
                "inputs_sha256": schemas.content_hash(profiler_runtime),
            },
            "admission_policy_sha256": config.admission_policy.value["policy_sha256"],
            "load_admission_profile_id": _LOAD_PROFILE_ID,
            "actor_wrappers": {
                "planner": {"path": str(config.actor_wrapper.path),
                            "sha256": config.actor_wrapper.sha256},
                "critic": {"path": str(config.critic_wrapper.path),
                           "sha256": config.critic_wrapper.sha256}},
            "actor_runtimes": {"planner": dict(planner_runtime),
                               "critic": dict(critic_runtime)},
            "actor_cells": [dict(controller.SOL), dict(controller.FABLE5_CRITIC)],
            "actor_argv_authority": {
                "planner": {"module_id": "codex_container_actor",
                            "module_sha256": execution_modules[
                                "codex_container_actor"]["sha256"],
                            "constructor": "codex_container_actor.build_docker_argv",
                            "image_id": codex_container_actor.CONTAINER_IMAGE_ID},
                "critic": {"module_id": "claude_fable5_critic_actor",
                           "module_sha256": execution_modules[
                               "claude_fable5_critic_actor"]["sha256"],
                           "constructor": "claude_fable5_critic_actor.build_argv",
                           "tools": [], "permission_mode": "plan"}},
            "critic_auth_source": dict(critic_auth),
            "execution_modules": dict(execution_modules),
            "environment_profiles": {"planner": dict(_SAFE_ACTOR_ENVIRONMENT),
                                     "critic": dict(_SAFE_CRITIC_ENVIRONMENT)},
            "source_authority": {
                "production_base_path": str(config.production_path),
                "production_base_commit": config.production_head,
                "instrument_repo_path": str(config.instrument_path),
                "instrument_branch": config.instrument_branch,
                "instrument_commit": config.instrument_commit,
                "instrument_diff_sha256": _INSTRUMENT_DIFF_SHA256,
                "instrument_test_source_sha256": _INSTRUMENT_TEST_SOURCE_SHA256,
                "ready_continue_contract_source_sha256": _READY_CONTINUE_CONTRACT_SHA256},
            "instrument_review": {"path": str(instrument_review[0]),
                                  "sha256": instrument_review[1]},
            "batched_runner": {"processes_per_arm": 1, "calls_per_arm": 9,
                               "arm_order_policy": (
                                   "sha256(deployment_config_sha256+source_manifest_sha256) "
                                   "parity selects S1; S2 is the exact reverse"),
                               "ready_continue_schema": "epyc.autokernel.ready_continue.v1",
                               "instrument_commit": _INSTRUMENT_COMMIT,
                               "contract_source_sha256": _READY_CONTINUE_CONTRACT_SHA256,
                               "early_unlock_enabled": False,
                               "trust_limit": "cooperative_same_uid_not_launch_authority",
                               "safe_fallback": "full_process_cold_serialized_lock",
                               "timed_output_oracle": {
                                   "schema": "epyc.autokernel.timed_output_semantics.v1",
                                   "instrument_commit": _INSTRUMENT_COMMIT,
                                   "enabled_for_source_patch": True,
                                   "independent_of_early_unlock": True,
                                   "input_bank": "same_seed_anchor_candidate",
                                   "within_pair_output": "bitwise_equal",
                                   "cross_arm_output": "sealed_81bf_64bit_hash_equal",
                                   "scored_sample": (
                                       "min(first_tokens_per_s,second_tokens_per_s)"),
                                   "summary": "tokens_per_mean_protected_latency",
                                   "environment": {
                                       "AMD_SERIALIZE_KERNEL": "3",
                                       "AMD_SERIALIZE_COPY": "3",
                                       "GGML_CUDA_DISABLE_GRAPHS": "1"},
                                   "scope": "integrity_discovery_only",
                                   "production_throughput_authority": False}},
            "instrument_target_equality": {"path": str(target_equality[0]),
                                           "sha256": target_equality[1]},
            "production_runtime_snapshot_sha256": production_runtime_sha256,
            "mutable_roots": {
                "state": str(config.state_root),
                "evidence": str(config.evidence_root),
                "operations": str(config.operations_root),
                "build": str(config.build_root),
            },
            "device_id": config.device_id,
            "device_reservation": {
                "prebuild_admission": "nonblocking_acquire_verify_release_probe",
                "postbuild_admission": "nonblocking_operation_scoped_outer_claim",
                "held_across": ["correctness", "candidate_attribution",
                                "anchor_attribution", "throughput"],
                "inner_claim_mode": "borrowed_logical_phase_no_physical_release",
                "nested_physical_acquisitions": False,
                "physical_release_authority": "adapter_terminal_reservation_release",
                "busy_disposition": "durable_pending_exact_candidate_no_iteration",
            },
            "claim_journal": str(config.operations_root / "claims" / "device.jsonl")}
    body["graph_sha256"] = schemas.content_hash(body)
    path = config.state_root / "deployment-graph.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        if path.is_symlink():
            raise DeploymentFactoryError("durable deployment graph is a symlink")
        try:
            durable = json.loads(path.read_bytes())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DeploymentFactoryError("durable deployment graph is malformed") from exc
        if durable.get("schema") in {
                "epyc.autokernel.static_discovery_graph.v4",
                "epyc.autokernel.static_discovery_graph.v5",
                "epyc.autokernel.static_discovery_graph.v6",
                "epyc.autokernel.static_discovery_graph.v7",
                "epyc.autokernel.static_discovery_graph.v8"}:
            raise DeploymentFactoryError(
                "legacy deployment graph cannot authorize successor execution; "
                "initialize a fresh v9 deployment")
        if path.read_bytes() != encoded:
            raise DeploymentFactoryError("durable deployment graph differs from current sealed graph")
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    return path.resolve(), str(body["graph_sha256"])


def build_static_deployment_graph(config: deployment.DiscoveryDeployment) -> StaticDeploymentGraph:
    """Construct the sole live graph.  No registry/executor object is accepted."""
    config.revalidate()
    try:
        preauthored_continuation.verify_git_authority(
            config.preauthored_continuation.value, config.instrument_path,
            config.instrument_commit)
    except preauthored_continuation.PreauthoredContinuationError as exc:
        raise DeploymentFactoryError(
            "preauthored continuation Git authority changed") from exc
    historical_q5_evidence = _preauthored_historical_evidence(
        config.preauthored_continuation.value,
        config.hypothesis_evidence_manifest.value["evidence"])
    if (config.planner_context.value[
            "preauthored_historical_evidence_sha256"]
            != historical_q5_evidence["receipt_sha256"]):
        raise DeploymentFactoryError(
            "planner context historical continuation evidence changed")
    if (config.hypothesis_portfolio.value.sha256 != _PORTFOLIO_SEMANTIC_SHA256
            or config.hypothesis_portfolio.input.sha256 != _PORTFOLIO_FILE_SHA256
            or config.hypothesis_portfolio_contract.sha256
            != _PORTFOLIO_CONTRACT_SHA256):
        raise DeploymentFactoryError(
            "deployment selected an unreviewed hypothesis portfolio authority")
    # Validate the exact predecessor terminal during validation as well as
    # execution.  The graph binds its projection so a graph-only dry run cannot
    # silently omit the cross-campaign replay authority used by the controller.
    carry_forward = _v25_carry_forward(config)
    target_equality = _target_source_equality_receipt(config)
    instrument_review = _instrument_review_receipt(config)
    execution_runtime_provenance = _execution_module_runtime_provenance()
    execution_modules = {
        name: {"logical_path": row["logical_path"], "sha256": row["sha256"]}
        for name, row in execution_runtime_provenance.items()
    }
    templates = _template_registry()
    try:
        hypothesis_portfolio.validate_template_authorability(
            config.hypothesis_portfolio.value, templates.version,
            _normalized_template_surfaces(templates, config.hypothesis_portfolio.value))
    except hypothesis_portfolio.PortfolioError as exc:
        raise DeploymentFactoryError(
            "portfolio exceeds the deployed template registry") from exc
    source_package = (_reviewed_source_package(config, templates)
                      if isinstance(config.instrument_commit, str) else None)
    surfaces = _normalized_template_surfaces(templates, config.hypothesis_portfolio.value)
    symbol_authority = {
        template_id: {
            path: sorted(symbols)
            for path, symbols in template.allowed_symbols.items()}
        for template_id, template in sorted(templates.templates.items())}
    dispatch_authority = _portfolio_dispatch_authority(
        templates, config.hypothesis_portfolio.value)
    if (config.planner_context.value["reviewed_source_package_sha256"]
            != source_package.package_sha256
            or config.planner_context.value["template_registry_sha256"]
            != templates.registry_sha256
            or config.planner_context.value["template_symbol_authority"]
            != symbol_authority
            or config.planner_context.value["template_surfaces"] != surfaces
            or config.planner_context.value["template_surfaces_sha256"]
            != schemas.content_hash(surfaces)
            or config.planner_context.value["portfolio_dispatch_authority"]
            != dispatch_authority):
        raise DeploymentFactoryError(
            "planner context differs from live reviewed source/template authority")
    registry = _static_registry(config, templates)
    production_snapshot = _require(
        registry["production_snapshot"][_STATIC_IDS["production_snapshot"]],
        ProductionSnapshotBinding, "production_snapshot")
    planner_runtime = codex_container_actor.runtime_identity(config.actor_wrapper.path)
    critic_runtime = claude_fable5_critic_actor.runtime_identity(
        config.critic_wrapper.path)
    critic_auth = _validate_critic_auth_source()
    planner_launcher_sha256 = _digest_regular(
        Path(codex_container_actor.__file__).resolve(), "Codex actor launcher")
    critic_launcher_sha256 = _digest_regular(
        Path(claude_fable5_critic_actor.__file__).resolve(),
        "Claude critic launcher")
    sampler = gpu_residency_sampler.Mi210ResidencySampler()
    executor = evidence.SubprocessCommandExecutor(
        residency_sampler=sampler,
        runtime_maps_sampler=discovery_static_registry.runtime_maps_sampler())
    journal = device_claim.ClaimJournal(config.operations_root / "claims" / "device.jsonl")
    adapters = materialize(config, registry, correctness_executor=executor,
                           rocprof_executor=executor, claim_journal=journal,
                           runner_attest=_module_attestor(
                               execution_modules, execution_runtime_provenance))
    # Replace generic actor instances with byte/runtime-pinned equivalents.
    catalog = adapters["planner"].template_catalog
    telemetry = discovery_telemetry.DiscoveryTelemetry(config.operations_root / "live")
    adapters = dict(adapters)
    adapters["planner"] = controller.CodexPlanner(
        wrapper=config.actor_wrapper.path, environment=_SAFE_ACTOR_ENVIRONMENT,
        template_catalog=catalog, reviewed_sources=source_package,
        wrapper_sha256=config.actor_wrapper.sha256,
        runtime_identity=planner_runtime,
        actor_launcher_sha256=planner_launcher_sha256,
        telemetry=telemetry)
    adapters["critic"] = controller.ClaudeCritic(
        wrapper=config.critic_wrapper.path, environment=_SAFE_CRITIC_ENVIRONMENT,
        template_catalog=catalog, reviewed_sources=source_package,
        wrapper_sha256=config.critic_wrapper.sha256,
        runtime_identity=critic_runtime,
        actor_launcher_sha256=critic_launcher_sha256,
        telemetry=telemetry)
    receipt, digest = _seal_graph_receipt(
        config, planner_runtime, critic_runtime, templates,
        target_equality, instrument_review,
        source_package,
        execution_modules,
        production_snapshot.runtime_semantics_sha256,
        critic_auth,
        carry_forward)
    return StaticDeploymentGraph(config=config, adapters=MappingProxyType(adapters),
                                 registry_ids=_STATIC_IDS, graph_receipt=receipt,
                                 graph_sha256=digest)


class GpuDiscoveryLease:
    """Two-phase device admission and operation-scoped execution reservation."""
    def __init__(self, *, config: deployment.DiscoveryDeployment, mode: str,
                 claim_journal: Any, claim_acquirer: Callable[..., Any],
                 claim_verifier: Callable[[Mapping[str, Any]], object],
                 kfd_root: Path = _MI210_KFD_PROCS) -> None:
        if not kfd_root.is_absolute() or ".." in kfd_root.parts:
            raise DeploymentFactoryError("KFD process root must be an absolute path")
        self.config, self.mode = config, mode
        self.claim_journal = claim_journal
        self.claim_acquirer = claim_acquirer
        self.claim_verifier = claim_verifier
        self.kfd_root = kfd_root
        self._active: dict[str, Any] = {}
        self._campaigns: dict[str, str] = {}
        self._pending_probe_release: dict[str, Any] = {}

    def _foreign_kfd_pids(self) -> tuple[tuple[int, ...], str | None]:
        """Read the complete KFD process inventory or return a refusal code."""
        if self.kfd_root.is_symlink() or not self.kfd_root.is_dir():
            return (), "foreign_kfd_inventory_invalid"
        try:
            entries = tuple(self.kfd_root.iterdir())
        except OSError:
            return (), "foreign_kfd_inventory_unreadable"
        pids: list[int] = []
        for entry in entries:
            name = entry.name
            if (not name.isascii() or not name.isdecimal()
                    or name != str(int(name)) or int(name) <= 0
                    or entry.is_symlink() or not entry.is_dir()):
                return (), "foreign_kfd_inventory_invalid"
            pids.append(int(name))
        return tuple(sorted(set(pids))), None

    @staticmethod
    def _receipt(value: object, label: str) -> dict[str, Any]:
        if hasattr(value, "to_dict"):
            value = value.to_dict()  # type: ignore[union-attr]
        if not isinstance(value, Mapping):
            raise DeploymentFactoryError(f"{label} did not produce a device-claim receipt")
        try:
            return device_claim.ClaimReceipt.from_dict(value).to_dict()
        except (TypeError, ValueError) as exc:
            raise DeploymentFactoryError(f"{label} produced a malformed device-claim receipt") from exc

    @staticmethod
    def _passed(value: object) -> bool:
        if isinstance(value, bool):
            return value
        passed = getattr(value, "passed", None)
        if isinstance(passed, bool):
            return passed
        return (getattr(value, "outcome", None) == schemas.PASS
                or getattr(value, "status", None) == schemas.PASS)

    def _claim(self, operation_key: str, *, purpose: str, max_hold_s: float) -> Any:
        if not isinstance(operation_key, str) or not controller.HASH.fullmatch(operation_key):
            raise DeploymentFactoryError("device reservation requires a canonical operation key")
        campaign_id = self._campaigns.get(operation_key)
        if not isinstance(campaign_id, str) or not campaign_id.startswith("ak-"):
            raise DeploymentFactoryError("device reservation lacks the candidate campaign identity")
        return self.claim_acquirer(
            self.config.device_id, purpose=purpose,
            campaign_id=campaign_id,
            journal=self.claim_journal, holder_label="autokernel-discovery-controller",
            timeout_s=0.0, max_hold_s=max_hold_s)

    def _finish_probe_release(self, operation_key: str, claim: Any) -> dict[str, Any]:
        self._pending_probe_release[operation_key] = claim
        opened: dict[str, Any] | None = None
        try:
            opened = self._receipt(claim.receipt(), "admission probe before release")
        except DeploymentFactoryError:
            # Cleanup still proceeds for an acquired handle whose receipt is
            # malformed; the original validation error remains authoritative.
            pass
        try:
            try:
                released = self._receipt(claim.release(), "admission probe release")
            except BaseException:
                released = self._receipt(
                    claim.release(), "retried admission probe release")
        except BaseException:
            # Keep the exact handle: DeviceClaim's durable journal retry is
            # impossible through a reconstructed object.
            raise
        if not released.get("released_at") or getattr(claim, "held", None) is not False:
            raise DeploymentFactoryError("admission probe did not physically release")
        if opened is not None:
            comparable = tuple(key for key in opened if key != "released_at")
            if any(released.get(key) != opened.get(key) for key in comparable):
                raise DeploymentFactoryError("admission probe release changed claim identity")
        self._pending_probe_release.pop(operation_key, None)
        return released

    def admit(self, candidate: controller.PlannedCandidate, *,
              operation_key: str) -> Mapping[str, Any]:
        self.config.revalidate()
        campaign_id = candidate.source_manifest.campaign_id
        existing_campaign = self._campaigns.setdefault(operation_key, campaign_id)
        if existing_campaign != campaign_id:
            raise DeploymentFactoryError("operation key was rebound to another campaign")
        pending_probe = self._pending_probe_release.get(operation_key)
        if pending_probe is not None:
            self._finish_probe_release(operation_key, pending_probe)
        corpus = self.config.admission_policy.corpus
        profiles = [profile for profile in corpus.profiles
                    if (profile.model_path == str(self.config.model.path)
                        and profile.model_sha256 == self.config.model.sha256
                        and profile.device_id == self.config.device_id)]
        if len(profiles) != 1:
            raise DeploymentFactoryError("sealed admission policy has no unique configured model profile")
        profile = profiles[0]
        foreign_kfd_pids, kfd_refusal = self._foreign_kfd_pids()
        actual_bytes = self.config.model.path.stat().st_size
        request = gpu_load_admission.AdmissionRequest(
            effective_context_sha256=schemas.content_hash({
                "planner_context_sha256": self.config.planner_context.value["context_sha256"],
                "admission_policy_sha256": corpus.policy_sha256,
                "admission_policy_version": corpus.version}),
            model_path=str(self.config.model.path), model_sha256=self.config.model.sha256,
            model_bytes=actual_bytes, workload=profile.workload,
            calls_per_arm=profile.calls_per_arm, device_id=self.config.device_id,
            cold_load_host_bytes=profile.cold_load_host_bytes,
            worst_case_loads_per_interval=profile.worst_case_loads_per_interval,
            # This deployment is explicitly fallback-only and deliberately has
            # no telemetry authority.  Enabling overlap requires a new reviewed
            # policy/binding, not an ambient or caller-supplied observation.
            telemetry_observed=False, telemetry_age_ms=None,
            observed_headroom_bytes_per_s=None, telemetry_receipt_sha256=None)
        request = replace(request, foreign_kfd_pids=foreign_kfd_pids)
        advisory = None
        intent = candidate.experiment_intent
        if isinstance(intent, controller.GpuSourceExperimentIntent) and intent.load_mode_recommendation is not None:
            recommendation = intent.load_mode_recommendation
            known_examples = {row.example_id for row in getattr(corpus, "examples", ())}
            if not set(recommendation.example_ids).issubset(known_examples):
                raise DeploymentFactoryError("planner load-mode advisory cites an unknown sealed policy example")
            advisory = recommendation.mode
        decision = gpu_load_admission.arbitrate(corpus, request, actor_recommendation=advisory)
        if decision.mode == "hot_resident":
            raise DeploymentFactoryError("nonpersistent source discovery runner cannot claim hot residency")
        common = {"mode": decision.mode,
                "device_id": self.config.device_id,
                "operation_key": operation_key,
                "inference_window_lock": str(self.config.inference_window_lock),
                "model_sha256": self.config.model.sha256,
                "load_admission": decision.to_dict(),
                "promotion_claim": False}
        if kfd_refusal is not None:
            return {**common, "admitted": False, "phase": "prebuild_probe",
                    "reason": kfd_refusal, "foreign_kfd_pids": []}
        if foreign_kfd_pids:
            return {**common, "admitted": False, "phase": "prebuild_probe",
                    "reason": "foreign_kfd_busy",
                    "foreign_kfd_pids": list(foreign_kfd_pids)}
        try:
            probe = self._claim(
                operation_key, purpose="AutoKernel GPU discovery admission probe",
                max_hold_s=30.0)
        except device_claim.DeviceClaimTimeout as exc:
            return {**common, "admitted": False, "phase": "prebuild_probe",
                    "reason": "device_busy", "detail": str(exc)}
        try:
            opened = self._receipt(probe.receipt(), "admission probe")
            if not self._passed(self.claim_verifier(opened)):
                raise DeploymentFactoryError("admission probe was not verifiably held")
        except BaseException as primary:
            try:
                self._finish_probe_release(operation_key, probe)
            except BaseException as cleanup:
                raise DeploymentFactoryError(
                    f"admission probe validation failed ({primary}); release durability also failed") from cleanup
            raise
        released = self._finish_probe_release(operation_key, probe)
        return {**common, "admitted": True, "phase": "prebuild_probe",
                "device_claim_probe_open": opened,
                "device_claim_probe_released": released}

    def resume(self, candidate: controller.PlannedCandidate,
               stale_permit: Mapping[str, Any]) -> Mapping[str, Any]:
        operation_key = stale_permit.get("operation_key")
        if not isinstance(operation_key, str):
            raise DeploymentFactoryError("stale permit lacks its operation key")
        fresh = self.admit(candidate, operation_key=operation_key)
        foreign_wait = str(stale_permit.get("reason", "")).startswith("foreign_kfd_")
        for key in ("mode", "device_id", "inference_window_lock", "model_sha256",
                    "promotion_claim", "operation_key"):
            if fresh.get(key) != stale_permit.get(key):
                raise DeploymentFactoryError("resumed device admission changed sealed policy authority")
        if not foreign_wait and fresh.get("load_admission") != stale_permit.get("load_admission"):
            raise DeploymentFactoryError("resumed device admission changed sealed policy authority")
        if foreign_wait:
            def stable(value: object) -> dict[str, Any]:
                row = value if isinstance(value, Mapping) else {}
                profile = row.get("profile") if isinstance(row.get("profile"), Mapping) else {}
                request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
                return {"policy_file_sha256": row.get("policy_file_sha256"),
                        "policy_sha256": row.get("policy_sha256"),
                        "policy_version": row.get("policy_version"),
                        "profile_id": profile.get("profile_id"),
                        "profile_model_sha256": profile.get("model_sha256"),
                        "request_context_sha256": request.get("effective_context_sha256"),
                        "request_model_sha256": request.get("model_sha256"),
                        "request_device_id": request.get("device_id"),
                        "promotion_claim": row.get("promotion_claim")}
            if stable(fresh.get("load_admission")) != stable(
                    stale_permit.get("load_admission")):
                raise DeploymentFactoryError(
                    "resumed foreign-KFD wait changed sealed policy authority")
        return fresh

    def reserve(self, operation_key: str) -> Mapping[str, Any]:
        existing = self._active.get(operation_key)
        if existing is not None:
            opened = self._receipt(existing.receipt(), "active reservation")
            if existing.held and self._passed(self.claim_verifier(opened)):
                return opened
            raise DeploymentFactoryError("operation reservation is present but not verifiably held")
        if self._active:
            raise DeploymentFactoryError("one deployment cannot hold two operation reservations")
        def kfd_wait() -> controller.ResourceWait | None:
            foreign, refusal = self._foreign_kfd_pids()
            if refusal is None and not foreign:
                return None
            reason = refusal or "foreign_kfd_busy"
            return controller.ResourceWait(
                "foreign KFD inventory prevents GPU reservation",
                receipt={"admitted": False, "phase": "pre_executor_reservation",
                         "reason": reason, "foreign_kfd_pids": list(foreign),
                         "device_id": self.config.device_id,
                         "operation_key": operation_key,
                         "promotion_claim": False})
        wait = kfd_wait()
        if wait is not None:
            raise wait
        try:
            claim = self._claim(
                operation_key, purpose="AutoKernel GPU source proof and throughput",
                max_hold_s=3600.0)
        except device_claim.DeviceClaimTimeout as exc:
            raise controller.ResourceWait(
                "device became busy after the prebuild probe",
                receipt={"admitted": False, "phase": "pre_executor_reservation",
                         "reason": "device_busy", "detail": str(exc),
                         "device_id": self.config.device_id,
                         "operation_key": operation_key,
                         "promotion_claim": False}) from exc
        # Register cleanup ownership before parsing any caller-controlled
        # receipt or invoking a verifier that may raise.
        self._active[operation_key] = claim
        try:
            opened = self._receipt(claim.receipt(), "operation reservation")
            if not self._passed(self.claim_verifier(opened)):
                raise DeploymentFactoryError("operation reservation was not verifiably held")
            wait = kfd_wait()
            if wait is not None:
                raise wait
        except BaseException as primary:
            try:
                self.release(operation_key)
            except BaseException as cleanup:
                raise DeploymentFactoryError(
                    f"operation reservation validation failed ({primary}); release durability also failed") from cleanup
            raise
        return opened

    def borrower(self, operation_key: str) -> Callable[..., Any]:
        def acquire(device_id: str, **kwargs: Any) -> Any:
            claim = self._active.get(operation_key)
            if (claim is None or not claim.held or device_id != self.config.device_id
                    or kwargs.get("campaign_id") != self._campaigns.get(operation_key)):
                raise DeploymentFactoryError("borrowed device claim lacks its held outer reservation")
            opened = self._receipt(claim.receipt(), "borrowed outer reservation")
            if not self._passed(self.claim_verifier(opened)):
                raise DeploymentFactoryError("borrowed outer reservation is no longer held")
            return _BorrowedDeviceClaim(opened)
        return acquire

    def release(self, operation_key: str) -> Mapping[str, Any] | None:
        claim = self._active.get(operation_key)
        if claim is None:
            return None
        opened: dict[str, Any] | None = None
        try:
            opened = self._receipt(claim.receipt(), "operation reservation before release")
        except DeploymentFactoryError:
            pass
        try:
            released = self._receipt(claim.release(), "operation reservation release")
        except BaseException:
            released = self._receipt(
                claim.release(), "retried operation reservation release")
        if not released.get("released_at") or getattr(claim, "held", None) is not False:
            raise DeploymentFactoryError("operation reservation did not physically release")
        if opened is not None:
            comparable = tuple(key for key in opened if key != "released_at")
            if any(released.get(key) != opened.get(key) for key in comparable):
                raise DeploymentFactoryError("operation reservation release changed claim identity")
        self._active.pop(operation_key, None)
        return released


class _BorrowedDeviceClaim:
    """Logical subclaim whose physical exclusion is owned by the outer claim."""
    borrowed_outer_reservation = True
    def __init__(self, opened: Mapping[str, Any]) -> None:
        self._opened = device_claim.ClaimReceipt.from_dict(opened)
        self._phase_end: dict[str, Any] | None = None

    @property
    def held(self) -> bool:
        return self._phase_end is None

    def receipt(self) -> device_claim.ClaimReceipt:
        return self._opened

    def release(self) -> Mapping[str, Any]:
        if self._phase_end is None:
            ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self._phase_end = {
                "schema": evidence.BORROWED_PHASE_SCHEMA,
                "mode": "borrowed_outer_reservation",
                "outer_claim_id": self._opened.claim_id,
                "device_id": self._opened.device_id,
                "campaign_id": self._opened.campaign_id,
                "phase_ended_at": ended_at,
                "physical_release": False,
            }
        return dict(self._phase_end)


def _require(value: object, type_: type, label: str) -> Any:
    if not isinstance(value, type_): raise DeploymentFactoryError(f"registry entry {label} has wrong typed binding")
    return value

_FORBIDDEN_MUTATION_PREFIXES = ("tools/", "examples/", "scripts/", "tests/", "cmake/",
                                "CMakeLists.txt", "models/", "recipes/", "artifacts/",
                                "benchmark/", "evaluator/", "profiler/")
# The GPU source lane never lets a planner touch generic GGML dispatch, build
# files, tests, or the reward path.  A reviewed HIP kernel surface is the one
# narrowly scoped exception.
_GPU_KERNEL_EXTENSIONS = frozenset({".cu", ".cuh", ".hip", ".hpp"})

def _validate_source_scope(candidate: controller.PlannedCandidate,
                           template: ExperimentTemplate | None = None) -> None:
    manifest = candidate.source_manifest
    if manifest.source_tree != "llama.cpp":
        raise DeploymentFactoryError("discovery source patch must target the allowlisted llama.cpp scope")
    if (template is None
            or not 1 <= len(manifest.declared_files) <= 2
            or set(manifest.declared_files) != set(template.allowed_files)):
        raise DeploymentFactoryError(
            "discovery intent must select the exact reviewed file set")
    for path in manifest.declared_files:
        if (path.startswith(_FORBIDDEN_MUTATION_PREFIXES) or template is None
                or path not in template.allowed_files
                or Path(path).suffix not in _GPU_KERNEL_EXTENSIONS
                or "CMake" in Path(path).name):
            raise DeploymentFactoryError("discovery patch may only modify allowlisted GGML kernel sources")
        symbols = set(manifest.declared_symbols[path])
        if ("<file-scope>" in symbols
                or not symbols.issubset(template.allowed_symbols.get(path, frozenset()))):
            raise DeploymentFactoryError("discovery patch symbols exceed the reviewed kernel template")


def materialize(config: deployment.DiscoveryDeployment, registry: Mapping[str, Mapping[str, object]], *,
                correctness_executor: evidence.CommandExecutor, rocprof_executor: evidence.CommandExecutor,
                claim_journal: device_claim.ClaimJournal,
                claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
                claim_verifier: Callable[[Mapping[str, Any]], object] = device_claim.check_device_claim_held,
                runner_attest: Callable[[], None] = lambda: None,
                receipt_series: Callable[[controller.PlannedCandidate, controller.SealedScreen], tuple[controller.SealedScreen, ...]] = lambda _candidate, current: (current,)
                ) -> dict[str, Any]:
    resolved = deployment.resolve_registry(config, registry)
    env = _require(resolved.environment_profile, EnvironmentProfile, "environment_profile")
    source = _require(resolved.source_builder, SourceBuilderBinding, "source_builder")
    plans = _require(resolved.evidence_plan, EvidencePlanBinding, "evidence_plan")
    runner = _require(resolved.runner_args, RunnerArgsBinding, "runner_args")
    lease_binding = _require(resolved.inference_window_lease, InferenceWindowLeaseBinding, "inference_window_lease")
    snapshot = _require(resolved.production_snapshot, ProductionSnapshotBinding, "production_snapshot")
    templates = _require(resolved.experiment_template_registry, ExperimentTemplateRegistry,
                         "experiment_template_registry")
    if templates.registry_sha256 != config.experiment_template_registry_sha256:
        raise DeploymentFactoryError("experiment template registry differs from sealed deployment digest")
    lease = lease_binding.make(
        config, claim_journal=claim_journal, claim_acquirer=claim_acquirer,
        claim_verifier=claim_verifier)
    # Actors are not dependency-injected: a caller supplied object could attest
    # any model.  The deployment wrapper digest/environment profile are the sole
    # authority for the exact Sol planner and Fable 5 critic identities.
    catalog = {key: {"template_id": template.template_id,
                     "target_surface": template.target_surface,
                     "target_symbol": template.target_symbol,
                     "correctness_id": template.correctness_id,
                     "dispatch_id": template.dispatch_id,
                     "allowed_files": sorted(template.allowed_files),
                     "allowed_symbols": {path: sorted(symbols)
                                         for path, symbols in template.allowed_symbols.items()},
                     "source_workspace_paths": {
                         path: f"reviewed-source/{path}" for path in template.allowed_files},
                     "profile_anchor_dispatch": [vars(row)
                                                 for row in template.dispatch.anchor_exact],
                     "candidate_dispatch_authoring": (
                         "expected_dispatch is the exact deployed rocprofv3 anchor array from the "
                         "controller-owned portfolio binding; never replace it with predicted "
                         "candidate routes. Topology-changing candidate cells are derived and "
                         "validated by the controller after authorization."),
                     "semantics": dict(template.semantics)}
               for key, template in templates.templates.items()}
    source_package = (_reviewed_source_package(config, templates)
                      if isinstance(config.instrument_commit, str) else None)
    planner_runtime = codex_container_actor.runtime_identity(config.actor_wrapper.path)
    critic_runtime = claude_fable5_critic_actor.runtime_identity(
        config.critic_wrapper.path)
    _validate_critic_auth_source()
    planner = controller.CodexPlanner(
        wrapper=config.actor_wrapper.path, environment=env.values,
        template_catalog=catalog, reviewed_sources=source_package,
        wrapper_sha256=config.actor_wrapper.sha256,
        runtime_identity=planner_runtime,
        actor_launcher_sha256=_digest_regular(
            Path(codex_container_actor.__file__).resolve(),
            "Codex actor launcher"))
    critic = controller.ClaudeCritic(wrapper=config.critic_wrapper.path,
                                     environment=_SAFE_CRITIC_ENVIRONMENT,
                                     template_catalog=catalog,
                                     reviewed_sources=source_package,
                                     wrapper_sha256=config.critic_wrapper.sha256,
                                     runtime_identity=critic_runtime,
                                     actor_launcher_sha256=_digest_regular(
                                         Path(claude_fable5_critic_actor.__file__).resolve(),
                                         "Claude critic launcher"))

    def build(candidate: controller.PlannedCandidate, authorization: Any, permit: Mapping[str, Any]):
        config.revalidate()
        runner_attest()
        template = templates.resolve(candidate.experiment_intent)
        _validate_source_scope(candidate, template)
        candidate.source_manifest.bind(
            proposal=candidate.proposal, campaign_id=candidate.source_manifest.campaign_id,
            candidate_id=candidate.source_manifest.candidate_id,
            production_base_commit=config.production_head,
            instrument_commit=config.instrument_commit)
        permit = {**permit, "instrument_branch": config.instrument_branch,
                  "deployment_config_semantic_sha256": config.config_sha256}
        if _SUPERVISED_BUILD_AUTHORITY is not None:
            permit["deployment_config_canonical_sha256"] = (
                _SUPERVISED_BUILD_AUTHORITY[
                    "deployment_config_canonical_sha256"])
            permit["supervised_build_authority"] = json.loads(json.dumps(
                dict(_SUPERVISED_BUILD_AUTHORITY), sort_keys=True))
        snapshot.revalidate()
        operation_key = permit.get("operation_key")
        if not isinstance(operation_key, str):
            raise DeploymentFactoryError("source build permit lacks an operation key")
        # Seal the exact manifest bytes in the adapter-owned operation namespace
        # before an expensive source builder can be entered.  Evidence binding
        # later reopens this same file and refuses any intervening mutation.
        _manifest_file_for_operation(config, candidate, operation_key)
        # Re-open the reviewed source-level capability before entering the
        # expensive builder.  The builder still executes and receipts both
        # exact binaries after compilation; this early gate prevents a known
        # incompatible instrument source from consuming a build transaction.
        _instrument_review_receipt(config)
        return source.build(candidate, authorization, permit)
    def plan(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild,
             permit: Mapping[str, Any]):
        config.revalidate()
        runner_attest()
        # Re-open the builder receipt at every evidence boundary.  In
        # particular an S2/cache path must not inherit S1's root authority.
        from . import discovery_static_registry
        discovery_static_registry.verify_build_authority(
            build_, production_path=config.production_path,
            production_branch=config.production_branch,
            production_commit=config.production_head,
            instrument_path=config.instrument_path,
            instrument_branch=config.instrument_branch,
            instrument_commit=config.instrument_commit)
        template = templates.resolve(candidate.experiment_intent)
        try:
            repetition = permit.get("repetition")
            if repetition not in {1, 2}:
                raise DeploymentFactoryError(
                    "evidence plan lacks its controller-owned repetition")
            result = plans.build(candidate, build_, template, repetition)
            expected_dispatch = template.bind_dispatch(candidate.experiment_intent)
            if (result.dispatch != expected_dispatch
                    or result.model_sha256 != config.model.sha256):
                raise DeploymentFactoryError(
                    "evidence plan does not bind configured model/selected reviewed template")
        except (DeploymentFactoryError,
                discovery_static_registry.StaticRegistryError,
                evidence.EvidenceProducerError) as exc:
            # plan_factory is invoked before the adapter acquires the outer GPU
            # reservation or creates a proof/runner artifact.  Preserve the
            # completed build terminal, but classify this exact boundary as a
            # safe typed refusal rather than an ambiguous operation crash.
            raise controller.PostBuildEvidencePlanRefusal(
                f"post-build evidence plan refused before execution: {exc}") from exc
        return result
    def args(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild, permit: Mapping[str, Any]):
        config.revalidate()
        runner_attest()
        operation_key = permit.get("operation_key")
        if (not isinstance(operation_key, str)
                or build_.operation_key != operation_key):
            raise DeploymentFactoryError(
                "source build operation identity differs from the controller permit")
        if (build_.measurement_binary is None or build_.common_loader_dir is None
                or build_.anchor_loader_dir is None or build_.candidate_loader_dir is None
                or build_.reward_runtime_sha256 is None or build_.operation_key is None
                or build_.build_key is None
                or build_.materialization_receipt is None or build_.materialization_sha256 is None
                or build_.anchor_source_tree_receipt is None or build_.anchor_source_tree_sha256 is None
                or build_.candidate_source_tree_receipt is None or build_.candidate_source_tree_sha256 is None
                or build_.teardown_receipt is None or build_.teardown_sha256 is None):
            raise DeploymentFactoryError("source build lacks sealed runtime/materialization/teardown receipts")
        result = runner.build(candidate, build_, permit)
        return _bind_runner_runtime_authority(config, build_, permit, result)
    screener = gpu_source_adapter.build_governed_gpu_source_adapter(
        operations_root=config.operations_root, build_source=build, plan_factory=plan,
        args_factory=args, correctness_executor=correctness_executor,
        rocprof_executor=rocprof_executor, claim_journal=claim_journal,
        claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
        claim_timeout_s=config.claim_timeout_s, receipt_series=receipt_series,
        reservation_manager=lease,
        protected_roots=(config.production_path, config.instrument_path),
        protected_files=snapshot.files, runner_attest=runner_attest)
    return controller.build_controller_adapters(planner=planner, critic=critic, screener=screener, lease=lease)


def controller_config(config: deployment.DiscoveryDeployment, *, dry_run: bool = False) -> controller.ControllerConfig:
    """The deployment receipt is the sole source of controller configuration."""
    config.revalidate()
    carry_forward = _v25_carry_forward(config)
    preauthored_continuation.verify_git_authority(
        config.preauthored_continuation.value, config.instrument_path,
        config.instrument_commit)
    return controller.ControllerConfig(
        output_root=config.state_root, evidence_root=config.evidence_root,
        max_iterations=config.max_iterations,
        nomination_threshold=config.nomination_threshold, dry_run=dry_run,
        planner_context={**config.planner_context.value,
                         "admission_policy": _plain(config.admission_policy.value)},
        planner_context_sha256=schemas.content_hash({
            "planner_context_sha256": config.planner_context.value["context_sha256"],
            "admission_policy_sha256": config.admission_policy.corpus.policy_sha256,
            "admission_policy_version": config.admission_policy.corpus.version,
            "deployment_identity_sha256": config.config_sha256}),
        production_base_commit=config.production_head,
        instrument_commit=config.instrument_commit,
        experiment_template_registry_sha256=config.experiment_template_registry_sha256,
        admission_corpus_sha256=config.admission_policy.value["policy_sha256"],
        admission_corpus_version=config.admission_policy.corpus.version,
        deployment_identity_sha256=config.config_sha256,
        hypothesis_portfolio=config.hypothesis_portfolio.value,
        hypothesis_portfolio_sha256=config.hypothesis_portfolio.value.sha256,
        carry_forward=carry_forward,
        carry_forward_sha256=carry_forward["carry_forward_sha256"],
        preauthored_continuations={
            config.preauthored_continuation.value.hypothesis_id:
                config.preauthored_continuation.value,
        },
        # The sealed deployment digest, not a caller argument, namespaces all
        # controller/worktree/receipt identities across concurrent deployments.
        campaign_id=f"ak-discovery-{config.config_sha256[:16]}")


def deployment_main(argv: list[str] | None = None) -> int:
    """Config-only launcher; no caller can inject a registry or executor."""
    global _SUPERVISED_BUILD_AUTHORITY
    _SUPERVISED_BUILD_AUTHORITY = None
    parser = argparse.ArgumentParser(description=__doc__)
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--deployment")
    authority.add_argument("--initialize-bundle",
                           help="emit the fixed-site sealed deployment bundle")
    parser.add_argument("--frozen-production-comparator",
                        help="sealed exact-v9 graphs-on comparator receipt")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true",
                       help="alias for validate-only; never calls an actor or hardware")
    parser.add_argument("--supervised-config-fd", type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument("--supervised-authority-fd", type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-runtime-root", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    supervised = (args.supervised_config_fd, args.supervised_authority_fd,
                  args.supervisor_runtime_root)
    if any(value is not None for value in supervised) and \
            not all(value is not None for value in supervised):
        parser.error("supervised config, authority, and runtime root must be paired")
    if args.initialize_bundle:
        if args.validate_only or args.dry_run:
            parser.error("bundle initialization does not accept execution flags")
        if not args.frozen_production_comparator:
            parser.error(
                "bundle initialization requires --frozen-production-comparator")
        result = initialize_static_deployment_bundle(
            Path(args.initialize_bundle),
            frozen_production_comparator=Path(
                args.frozen_production_comparator))
        print(json.dumps({"status": "initialized", "inference_executed": False,
                          "deployment": str(result)}, sort_keys=True))
        return 0
    if args.frozen_production_comparator:
        parser.error(
            "--frozen-production-comparator is only valid with bundle initialization")
    sealed_bytes = None
    if args.supervised_config_fd is not None:
        sealed_bytes, authority = discovery_supervisor.verified_supervised_launch(
            Path(args.supervisor_runtime_root), args.supervised_config_fd,
            args.supervised_authority_fd)
        _SUPERVISED_BUILD_AUTHORITY = authority
        discovery_supervisor.verify_imported_execution_modules(
            Path(args.supervisor_runtime_root),
            _execution_module_runtime_provenance())
    config = deployment.load_deployment_config(
        Path(args.deployment), sealed_bytes=sealed_bytes)
    graph = build_static_deployment_graph(config)
    if args.validate_only or args.dry_run:
        print(json.dumps({"status": "validated", "inference_executed": False,
                          "graph_receipt": str(graph.graph_receipt),
                          "graph_sha256": graph.graph_sha256}, sort_keys=True))
        return 0
    controller.run_controller(controller_config(config), **dict(graph.adapters))
    return 0


def main() -> int:
    return deployment_main()


if __name__ == "__main__":
    raise SystemExit(main())
