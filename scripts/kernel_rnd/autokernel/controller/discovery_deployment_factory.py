"""Trusted concrete deployment materializer for governed GPU source discovery.

The JSON configuration merely selects IDs.  This module is the one static
bridge from those IDs to typed, registered Python construction seams.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .. import schemas
from ..execution import inference_window
from ..resource import device_claim
from . import discovery_controller as controller
from . import discovery_deployment as deployment
from . import gpu_source_adapter
from . import gpu_source_evidence as evidence
from . import gpu_load_admission
from . import gpu_residency_sampler
from . import codex_container_actor
from . import discovery_static_registry


class DeploymentFactoryError(RuntimeError): pass
_ALLOWED_ENV = frozenset({"PATH", "HOME", "CODEX_HOME", "HTTPS_PROXY", "HTTP_PROXY",
                          "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR"})


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
    build: Callable[[controller.PlannedCandidate, controller.GpuSourceBuild, "ExperimentTemplate"], evidence.GpuSourceEvidencePlan]

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
        return (self.template_id, self.target_surface, self.target_symbol,
                self.correctness_id, self.dispatch_id) == (
                    intent.template_id, intent.target_surface, intent.target_symbol,
                    intent.correctness_id, intent.dispatch_id)

    def bind_dispatch(self, intent: controller.GpuSourceExperimentIntent) -> evidence.DispatchContract:
        """Derive an internal escaped matcher from planner literals and reviewed bounds."""
        if not self.matches(intent):
            raise DeploymentFactoryError("dispatch intent does not select this reviewed template")
        expected = intent.expected_dispatch
        bounds = self.semantics.get("dispatch_bounds", {})
        if not isinstance(bounds, Mapping):
            raise DeploymentFactoryError("template dispatch bounds are malformed")
        for key, value in (("calls", expected.calls), ("grid", expected.grid),
                           ("workgroup", expected.workgroup), ("lds_bytes", expected.lds_bytes)):
            limit = bounds.get(key)
            if (not isinstance(limit, list) or len(limit) != 2
                    or not all(isinstance(item, int) for item in limit)
                    or not limit[0] <= value <= limit[1]):
                raise DeploymentFactoryError(f"planner dispatch {key} exceeds reviewed template bounds")
        prefixes = bounds.get("kernel_prefixes")
        if (not isinstance(prefixes, list) or not prefixes
                or not all(isinstance(value, str) and value for value in prefixes)
                or not any(expected.kernel_name.startswith(prefix) for prefix in prefixes)):
            raise DeploymentFactoryError("planner kernel literal is outside reviewed template families")
        if expected.grid % expected.workgroup:
            raise DeploymentFactoryError("planner dispatch grid must be an exact workgroup multiple")
        blocks = expected.grid // expected.workgroup
        candidate = evidence.ExactDispatch(
            signature=f"{self.dispatch_id}.candidate",
            kernel_pattern="^" + re.escape(expected.kernel_name) + "$",
            calls=expected.calls, grid=expected.grid, workgroup=expected.workgroup,
            lds_bytes=expected.lds_bytes, blocks_per_call=blocks)
        return evidence.DispatchContract(candidate_exact=(candidate,),
            anchor_exact=self.dispatch.anchor_exact,
            candidate_forbidden=self.dispatch.candidate_forbidden,
            anchor_forbidden=self.dispatch.anchor_forbidden,
            invariants=self.dispatch.invariants)


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
    def make(self, config: deployment.DiscoveryDeployment) -> "GpuDiscoveryLease":
        if self.mode != "allowed_discovery_noise":
            raise DeploymentFactoryError("GPU discovery lease may only admit allowed discovery noise")
        return GpuDiscoveryLease(config=config, mode=self.mode)

@dataclass(frozen=True)
class ProductionSnapshotBinding:
    files: tuple[evidence.BoundInputFile, ...]
    def __post_init__(self) -> None:
        if not self.files or not all(isinstance(item, evidence.BoundInputFile) for item in self.files):
            raise DeploymentFactoryError("production snapshot must contain typed frozen artifacts")


@dataclass(frozen=True)
class StaticDeploymentGraph:
    """Fully constructed trusted graph plus its durable validation receipt."""
    config: deployment.DiscoveryDeployment
    adapters: Mapping[str, Any]
    registry_ids: Mapping[str, str]
    graph_receipt: Path
    graph_sha256: str


_STATIC_IDS = MappingProxyType({
    "environment_profile": "sealed-codex",
    "source_builder": "gpu-source-v1",
    "evidence_plan": "q5-onewave-v1",
    "runner_args": "qwen05b-tg128",
    "experiment_template_registry": "gpu-source-templates-v1",
    "inference_window_lease": "mi210-window-v1",
    "production_snapshot": "llama-v9-artifacts",
})
_LOAD_PROFILE_ID = "mi210-qwen05b-tg128-18-v1"
_ROCPROF_V1 = Path(
    "/mnt/raid0/llm/autokernel/tools/rocprof6.2-extracted/opt/rocm-6.2.0/bin/rocprof")
_ROCPROF_V1_SHA256 = "585e3e6034e3c0bd9e591f0aa72f6156686680911a0b47ed4ece3c9a8372a4b2"
_ROCPROF_V1_INPUT = b"pmc:\n\ngpu:\nrange:\nkernel:\n"
_ROCPROF_V1_PREFIX = ("--tool-version", "1", "--timestamp", "on",
                       "--ctx-wait", "on", "--heartbeat", "30", "-i")
_SAFE_ACTOR_ENVIRONMENT = MappingProxyType({
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/home/node",
    "CODEX_HOME": "/home/node/.codex",
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
})


def _digest_regular(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise DeploymentFactoryError(f"{label} must be a regular non-symlink file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound(path: Path, role: str) -> evidence.BoundInputFile:
    path = path.resolve(strict=True)
    return evidence.BoundInputFile(role=role, path=path,
                                   sha256=_digest_regular(path, role))


def _rocprof_v1_policy(config: deployment.DiscoveryDeployment) -> tuple[
        evidence.BoundInputFile, evidence.BoundInputFile]:
    profiler = _bound(_ROCPROF_V1, "executable")
    if profiler.sha256 != _ROCPROF_V1_SHA256:
        raise DeploymentFactoryError("fixed rocprof-v1 executable digest changed")
    root = config.operations_root / "config"
    root.mkdir(parents=True, exist_ok=True)
    policy_path = root / "rocprof-v1-timestamps.txt"
    if policy_path.exists():
        if policy_path.is_symlink() or policy_path.read_bytes() != _ROCPROF_V1_INPUT:
            raise DeploymentFactoryError("rocprof-v1 input policy differs from checked-in bytes")
    else:
        policy_path.write_bytes(_ROCPROF_V1_INPUT)
    return profiler, evidence.BoundInputFile(
        "timestamp_input", policy_path.resolve(),
        hashlib.sha256(_ROCPROF_V1_INPUT).hexdigest())


def _template_registry() -> ExperimentTemplateRegistry:
    families = (
        ("cuda-fattn-v1", "ggml/src/ggml-cuda/fattn.cuh",
         "ggml_cuda_flash_attn_ext", ("ggml_cuda_flash_attn_ext",
          "ggml_cuda_flash_attn_ext_supported", "ggml_cuda_flash_attn_ext_get_alloc_size"),
         ("fattn", "flash_attn")),
        ("cuda-mmvq-v1", "ggml/src/ggml-cuda/mmvq.cu",
         "ggml_cuda_op_mul_mat_vec_q", ("ggml_cuda_op_mul_mat_vec_q",
          "ggml_cuda_mul_mat_vec_q", "mul_mat_vec_q_switch_type",
          "mul_mat_vec_q_switch_ncols_dst", "mul_mat_vec_q_moe_launch",
          "mul_mat_vec_q_switch_fusion", "mul_mat_vec_q8_0_prefetch_launch"),
         ("mmvq", "mul_mat_vec")),
        ("cuda-rope-v1", "ggml/src/ggml-cuda/rope.cu",
         "ggml_cuda_op_rope_impl", ("ggml_cuda_op_rope_impl", "ggml_cuda_op_rope",
          "ggml_cuda_op_rope_back", "ggml_cuda_op_rope_fused", "rope_norm",
          "rope_neox", "rope_multi", "rope_vision", "rope_norm_cuda",
          "rope_neox_cuda", "rope_multi_cuda", "rope_vision_cuda"),
         ("rope",)),
        ("cuda-norm-v1", "ggml/src/ggml-cuda/norm.cu",
         "ggml_cuda_op_rms_norm", ("ggml_cuda_op_norm", "ggml_cuda_op_group_norm",
          "ggml_cuda_op_rms_norm", "ggml_cuda_op_rms_norm_fused",
          "ggml_cuda_op_rms_norm_fused_add", "ggml_cuda_op_rms_norm_back",
          "ggml_cuda_op_l2_norm", "norm_f32", "group_norm_f32", "rms_norm_f32",
          "rms_norm_back_f32", "l2_norm_f32", "norm_f32_cuda",
          "group_norm_f32_cuda", "rms_norm_f32_cuda", "rms_norm_mul_f32_cuda",
          "rms_norm_back_f32_cuda", "l2_norm_f32_cuda"),
         ("norm", "rms_norm", "group_norm", "l2_norm")),
    )
    templates = {}
    for template_id, path, symbol, symbols, prefixes in families:
        family_pattern = "^(?:" + "|".join(re.escape(prefix) for prefix in prefixes) + ").*$"
        templates[template_id] = ExperimentTemplate(
            template_id=template_id, target_surface="gpu_decode", target_symbol=symbol,
            correctness_id="backend-ops-hip-v1", dispatch_id="decode-tg128-rocprof-v1",
            dispatch=evidence.DispatchContract(
                candidate_exact=(evidence.ExactDispatch(
                    f"{template_id}.candidate-family", family_pattern, 1, 64, 64, 0, 1),),
                anchor_exact=(evidence.ExactDispatch(
                    f"{template_id}.anchor-family", family_pattern, 1, 64, 64, 0, 1),)),
            allowed_files=frozenset({path}),
            allowed_symbols={path: frozenset(symbols)},
            semantics={"workload": "decode_tg128", "calls_per_arm": 9,
                       "load_admission_profile_id": _LOAD_PROFILE_ID,
                       "dispatch_bounds": {"calls": [1, 4096], "grid": [64, 1048576],
                                           "workgroup": [64, 1024], "lds_bytes": [0, 131072],
                                           "kernel_prefixes": list(prefixes)}})
    provisional = object.__new__(ExperimentTemplateRegistry)
    object.__setattr__(provisional, "version", "gpu-source-templates-v1")
    object.__setattr__(provisional, "templates", MappingProxyType(templates))
    body = {"version": "gpu-source-templates-v1", "templates": {
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
    return ExperimentTemplateRegistry("gpu-source-templates-v1", digest, templates)


def static_template_registry_sha256() -> str:
    """Public value for authoring a sealed config; it grants no constructor authority."""
    return _template_registry().registry_sha256


def _manifest_file(config: deployment.DiscoveryDeployment,
                   candidate: controller.PlannedCandidate,
                   build: controller.GpuSourceBuild) -> evidence.BoundInputFile:
    manifest = candidate.source_manifest
    value = {"schema": "epyc.autokernel.source_patch.v1",
             "campaign_id": manifest.campaign_id, "proposal_id": manifest.proposal_id,
             "candidate_id": manifest.candidate_id, "source_tree": manifest.source_tree,
             "production_base_commit": manifest.production_base_commit,
             "instrument_commit": manifest.instrument_commit,
             "change_class": manifest.change_class,
             "declared_files": list(manifest.declared_files),
             "declared_symbols": {path: list(manifest.declared_symbols[path])
                                  for path in manifest.declared_files},
             "mechanism_id": manifest.mechanism_id, "patch_sha256": manifest.patch_sha256,
             "patch_encoding": "base64",
             "patch_base64": base64.b64encode(manifest.patch_bytes).decode("ascii")}
    raw = schemas.canonical_bytes(value)
    if hashlib.sha256(raw).hexdigest() != candidate.source_manifest_sha256:
        raise DeploymentFactoryError("candidate manifest canonical carrier hash mismatch")
    assert build.operation_key is not None
    path = config.operations_root / "materialization" / build.operation_key / "source-manifest.json"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise DeploymentFactoryError("source manifest carrier already exists with different bytes")
    else:
        path.write_bytes(raw)
    return evidence.BoundInputFile("manifest", path.resolve(), candidate.source_manifest_sha256)


def _evidence_binding(config: deployment.DiscoveryDeployment) -> EvidencePlanBinding:
    profiler, timestamp_input = _rocprof_v1_policy(config)
    def build(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild,
              template: ExperimentTemplate) -> evidence.GpuSourceEvidencePlan:
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
        # Revalidate the profiler and timestamp carriers at the same binding
        # boundary as the source/runtime carriers.
        if (_digest_regular(profiler.path, "rocprof-v1") != profiler.sha256
                or _digest_regular(timestamp_input.path, "rocprof-v1 input")
                != timestamp_input.sha256):
            raise DeploymentFactoryError("rocprof-v1 policy changed before evidence binding")
        # The exact rocprof-v1 executable and input bytes are already resolved
        # and re-hashed above.  The remaining refusal is narrower: the current
        # builder checkpoint does not yet carry an exact reviewed correctness
        # argv/case-count binding for each source family.
        raise DeploymentFactoryError(
            "checked-in exact correctness command policy is unavailable; refusing evidence execution")
    return EvidencePlanBinding(build=build)


def _runner_binding(config: deployment.DiscoveryDeployment) -> RunnerArgsBinding:
    def build(_candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild,
              permit: Mapping[str, Any]) -> Any:
        operation_key = permit.get("operation_key")
        repetition = permit.get("repetition")
        if operation_key != build_.operation_key or repetition not in {1, 2}:
            raise DeploymentFactoryError("runner operation identity differs from sealed build")
        output = config.operations_root / str(operation_key) / "runner" / f"s{repetition}"
        argv = ["--anchor-build", str(build_.anchor_build), "--candidate-build", str(build_.candidate_build),
                "--model", str(config.model.path), "--output-dir", str(output),
                "--campaign-id", f"ak-discovery-{config.config_sha256[:16]}",
                "--factor", "source_patch", "--calls", "9", "--workload", "decode_tg128",
                "--allow-small-model-cpu-overlap", "--inference-window-lock", str(config.inference_window_lock),
                "--load-admission-profile-id", _LOAD_PROFILE_ID, "--device-id", config.device_id,
                "--measurement-binary", str(build_.measurement_binary),
                "--common-loader-dir", str(build_.common_loader_dir),
                "--anchor-loader-dir", str(build_.anchor_loader_dir),
                "--candidate-loader-dir", str(build_.candidate_loader_dir),
                "--cpu-claim-journal", str(config.operations_root / "claims" / "cpu.jsonl"),
                "--device-claim-journal", str(config.operations_root / "claims" / "device.jsonl")]
        return controller.gpu_discovery.parser().parse_args(argv)
    return RunnerArgsBinding(build=build)


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
    site = controller.gpu_discovery.SITE_LOAD_PROFILES[_LOAD_PROFILE_ID]
    if (site.model_sha256 != config.model.sha256
            or site.model_path != str(config.model.path)
            or site.model_bytes != config.model.path.stat().st_size
            or site.device_id != config.device_id):
        raise DeploymentFactoryError("configured model/device differs from checked-in admission profile")
    profiles = config.admission_policy.value.get("profiles")
    if not isinstance(profiles, list) or not any(
            isinstance(row, Mapping) and row.get("id") == _LOAD_PROFILE_ID for row in profiles):
        raise DeploymentFactoryError("sealed admission corpus omits the checked-in load profile")
    environment = EnvironmentProfile(_SAFE_ACTOR_ENVIRONMENT)
    source_builder = discovery_static_registry.StaticGpuSourceBuilder(
        production_path=config.production_path,
        production_branch=deployment.FROZEN_PRODUCTION_BRANCH,
        operations_root=config.operations_root,
        build_root=config.operations_root / "build",
        cmake_defines=(("GGML_HIP", "ON"), ("AMDGPU_TARGETS", "gfx90a"),
                       ("GGML_NATIVE", "OFF")))
    snapshot_paths = (config.production_path / "CMakeLists.txt",
                      config.production_path / "ggml/src/ggml-cuda/unary.cu",
                      config.production_path / "ggml/src/ggml-cuda/mmvq.cu")
    snapshot = ProductionSnapshotBinding(tuple(
        _bound(path, f"production:{path.relative_to(config.production_path)}")
        for path in snapshot_paths))
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
                        runtime: Mapping[str, Any], templates: ExperimentTemplateRegistry) -> tuple[Path, str]:
    launcher_path = Path(codex_container_actor.__file__).resolve(strict=True)
    launcher_sha256 = _digest_regular(launcher_path, "Codex actor launcher")
    body = {"schema": "epyc.autokernel.static_discovery_graph.v1",
            "authority": "nonpromotable_candidate_only_discovery", "promotion_claim": False,
            "inference_executed": False, "config_sha256": config.config_sha256,
            "registry_ids": dict(_STATIC_IDS), "template_registry_sha256": templates.registry_sha256,
            "admission_policy_sha256": config.admission_policy.value["policy_sha256"],
            "load_admission_profile_id": _LOAD_PROFILE_ID,
            "actor_wrapper": {"path": str(config.actor_wrapper.path),
                              "sha256": config.actor_wrapper.sha256},
            "actor_runtime": dict(runtime),
            "actor_cells": [dict(controller.SOL), dict(controller.TERRA)],
            "actor_argv_authority": {"module": str(launcher_path),
                                     "module_sha256": launcher_sha256,
                                     "constructor": "codex_container_actor.build_docker_argv",
                                     "image_id": codex_container_actor.CONTAINER_IMAGE_ID},
            "environment_profile": dict(_SAFE_ACTOR_ENVIRONMENT),
            "device_id": config.device_id,
            "claim_journal": str(config.operations_root / "claims" / "device.jsonl")}
    body["graph_sha256"] = schemas.content_hash(body)
    path = config.state_root / "deployment-graph.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(body, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        if path.is_symlink() or path.read_bytes() != encoded:
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
    templates = _template_registry()
    registry = _static_registry(config, templates)
    runtime = codex_container_actor.runtime_identity(config.actor_wrapper.path)
    launcher_sha256 = _digest_regular(Path(codex_container_actor.__file__).resolve(),
                                      "Codex actor launcher")
    sampler = gpu_residency_sampler.Mi210ResidencySampler()
    executor = evidence.SubprocessCommandExecutor(
        residency_sampler=sampler,
        runtime_maps_sampler=discovery_static_registry.runtime_maps_sampler())
    journal = device_claim.ClaimJournal(config.operations_root / "claims" / "device.jsonl")
    adapters = materialize(config, registry, correctness_executor=executor,
                           rocprof_executor=executor, claim_journal=journal)
    # Replace generic actor instances with byte/runtime-pinned equivalents.
    catalog = adapters["planner"].template_catalog
    adapters = dict(adapters)
    adapters["planner"] = controller.CodexPlanner(
        wrapper=config.actor_wrapper.path, environment=_SAFE_ACTOR_ENVIRONMENT,
        template_catalog=catalog, wrapper_sha256=config.actor_wrapper.sha256,
        runtime_identity=runtime, actor_launcher_sha256=launcher_sha256)
    adapters["critic"] = controller.CodexCritic(
        wrapper=config.actor_wrapper.path, environment=_SAFE_ACTOR_ENVIRONMENT,
        template_catalog=catalog, wrapper_sha256=config.actor_wrapper.sha256,
        runtime_identity=runtime, actor_launcher_sha256=launcher_sha256)
    receipt, digest = _seal_graph_receipt(config, runtime, templates)
    return StaticDeploymentGraph(config=config, adapters=MappingProxyType(adapters),
                                 registry_ids=_STATIC_IDS, graph_receipt=receipt,
                                 graph_sha256=digest)


class GpuDiscoveryLease:
    """Typed admission seam; actual runner calls must receive the same lock path."""
    def __init__(self, *, config: deployment.DiscoveryDeployment, mode: str) -> None:
        self.config, self.mode = config, mode
    def admit(self, candidate: controller.PlannedCandidate) -> Mapping[str, Any]:
        self.config.revalidate()
        corpus = self.config.admission_policy.corpus
        profiles = [profile for profile in corpus.profiles
                    if (profile.model_path == str(self.config.model.path)
                        and profile.model_sha256 == self.config.model.sha256
                        and profile.device_id == self.config.device_id)]
        if len(profiles) != 1:
            raise DeploymentFactoryError("sealed admission policy has no unique configured model profile")
        profile = profiles[0]
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
            # This generic binding deliberately has no mutable telemetry source.
            # The static site adapter can provide one; absence deterministically
            # downgrades to serialized load rather than inventing headroom.
            telemetry_observed=False, telemetry_age_ms=None,
            observed_headroom_bytes_per_s=None, telemetry_receipt_sha256=None)
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
        return {"admitted": True, "mode": decision.mode,
                "device_id": self.config.device_id,
                "inference_window_lock": str(self.config.inference_window_lock),
                "model_sha256": self.config.model.sha256,
                "load_admission": decision.to_dict(),
                "promotion_claim": False}


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
    lease = lease_binding.make(config)
    # Actors are not dependency-injected: a caller supplied object could attest
    # any model.  The deployment wrapper digest/environment profile are the sole
    # authority for the two exact Codex actor identities.
    catalog = {key: {"template_id": template.template_id,
                     "target_surface": template.target_surface,
                     "target_symbol": template.target_symbol,
                     "correctness_id": template.correctness_id,
                     "dispatch_id": template.dispatch_id,
                     "allowed_files": sorted(template.allowed_files),
                     "allowed_symbols": {path: sorted(symbols)
                                         for path, symbols in template.allowed_symbols.items()},
                     "semantics": dict(template.semantics)}
               for key, template in templates.templates.items()}
    planner = controller.CodexPlanner(wrapper=config.actor_wrapper.path,
                                      environment=env.values, template_catalog=catalog)
    critic = controller.CodexCritic(wrapper=config.actor_wrapper.path,
                                    environment=env.values, template_catalog=catalog)

    def build(candidate: controller.PlannedCandidate, authorization: Any, permit: Mapping[str, Any]):
        config.revalidate()
        template = templates.resolve(candidate.experiment_intent)
        _validate_source_scope(candidate, template)
        candidate.source_manifest.bind(
            proposal=candidate.proposal, campaign_id=candidate.source_manifest.campaign_id,
            candidate_id=candidate.source_manifest.candidate_id,
            production_base_commit=config.production_head,
            instrument_commit=config.instrument_commit)
        permit = {**permit, "instrument_branch": config.instrument_branch}
        return source.build(candidate, authorization, permit)
    def plan(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild):
        config.revalidate()
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
        result = plans.build(candidate, build_, template)
        expected_dispatch = template.bind_dispatch(candidate.experiment_intent)
        if result.dispatch != expected_dispatch or result.model_sha256 != config.model.sha256:
            raise DeploymentFactoryError("evidence plan does not bind configured model/selected reviewed template")
        return result
    def args(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild, permit: Mapping[str, Any]):
        config.revalidate()
        if (build_.measurement_binary is None or build_.common_loader_dir is None
                or build_.anchor_loader_dir is None or build_.candidate_loader_dir is None
                or build_.reward_runtime_sha256 is None or build_.operation_key is None
                or build_.materialization_receipt is None or build_.materialization_sha256 is None
                or build_.anchor_source_tree_receipt is None or build_.anchor_source_tree_sha256 is None
                or build_.candidate_source_tree_receipt is None or build_.candidate_source_tree_sha256 is None
                or build_.teardown_receipt is None or build_.teardown_sha256 is None):
            raise DeploymentFactoryError("source build lacks sealed runtime/materialization/teardown receipts")
        result = runner.build(candidate, build_, permit)
        decision = permit.get("load_admission")
        if not isinstance(decision, Mapping):
            raise DeploymentFactoryError("runner invocation lacks the sealed lease admission decision")
        decision = dict(decision)
        expected_admission = {
            "load_admission_decision": decision,
            "load_admission_policy_version": config.admission_policy.corpus.version,
            "load_admission_policy_sha256": config.admission_policy.corpus.policy_sha256,
            "load_admission_policy_file_sha256": config.admission_policy.corpus.file_sha256,
            "load_admission_effective_context_sha256": decision.get("effective_context_sha256"),
        }
        # A trusted static runner binding may construct an argparse.Namespace or
        # another mutable typed args holder, but it never gets to choose the
        # admission frame.  Refuse pre-filled mismatches and install the exact
        # lease receipt for the runner's preflight validator.
        for key, value in expected_admission.items():
            existing = getattr(result, key, None)
            if existing is not None and existing != value:
                raise DeploymentFactoryError(f"runner arguments attempted to override {key}")
            try:
                setattr(result, key, value)
            except (AttributeError, TypeError) as exc:
                raise DeploymentFactoryError("runner arguments cannot carry sealed load admission") from exc
        if (getattr(result, "factor", None) != "source_patch"
                or str(getattr(result, "model", "")) != str(config.model.path)
                or str(getattr(result, "anchor_build", "")) != str(build_.anchor_build)
                or str(getattr(result, "candidate_build", "")) != str(build_.candidate_build)
                or str(getattr(result, "measurement_binary", "")) != str(build_.measurement_binary)
                or str(getattr(result, "common_loader_dir", "")) != str(build_.common_loader_dir)
                or str(getattr(result, "anchor_loader_dir", "")) != str(build_.anchor_loader_dir)
                or str(getattr(result, "candidate_loader_dir", "")) != str(build_.candidate_loader_dir)
                or getattr(result, "promotion_claim", False) is not False
                or str(getattr(result, "inference_window_lock", "")) != str(config.inference_window_lock)
                or getattr(result, "load_admission_decision", None) != decision):
            raise DeploymentFactoryError("runner arguments do not bind source builds/model/window/discovery authority")
        return result
    screener = gpu_source_adapter.build_governed_gpu_source_adapter(
        operations_root=config.operations_root, build_source=build, plan_factory=plan,
        args_factory=args, correctness_executor=correctness_executor,
        rocprof_executor=rocprof_executor, claim_journal=claim_journal,
        claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
        claim_timeout_s=config.claim_timeout_s, receipt_series=receipt_series,
        protected_roots=(config.production_path, config.instrument_path), protected_files=snapshot.files)
    return controller.build_controller_adapters(planner=planner, critic=critic, screener=screener, lease=lease)


def controller_config(config: deployment.DiscoveryDeployment, *, dry_run: bool = False) -> controller.ControllerConfig:
    """The deployment receipt is the sole source of controller configuration."""
    config.revalidate()
    return controller.ControllerConfig(
        output_root=config.state_root, evidence_root=config.evidence_root,
        max_iterations=config.max_iterations,
        nomination_threshold=config.nomination_threshold, dry_run=dry_run,
        planner_context={**config.planner_context.value,
                         "admission_policy": config.admission_policy.value},
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
        # The sealed deployment digest, not a caller argument, namespaces all
        # controller/worktree/receipt identities across concurrent deployments.
        campaign_id=f"ak-discovery-{config.config_sha256[:16]}")


def deployment_main(argv: list[str] | None = None) -> int:
    """Config-only launcher; no caller can inject a registry or executor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true",
                       help="alias for validate-only; never calls an actor or hardware")
    args = parser.parse_args(argv)
    config = deployment.load_deployment_config(Path(args.deployment))
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
