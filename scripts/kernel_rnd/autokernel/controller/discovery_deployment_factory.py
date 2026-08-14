"""Trusted concrete deployment materializer for governed GPU source discovery.

The JSON configuration merely selects IDs.  This module is the one static
bridge from those IDs to typed, registered Python construction seams.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .. import schemas
from ..execution import inference_window, device_sampler
from ..resource import device_claim
from . import discovery_controller as controller
from . import discovery_deployment as deployment
from . import gpu_source_adapter
from . import gpu_source_evidence as evidence
from . import gpu_load_admission
from . import gpu_residency_sampler
from . import codex_container_actor
from . import discovery_static_registry
from . import gpu_source_proofs
from scripts.benchmark import autokernel_gpu_discovery_beliefs


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


def _execution_module_identity() -> dict[str, dict[str, str]]:
    modules = {
        "deployment_factory": Path(__file__).resolve(strict=True),
        "discovery_controller": Path(controller.__file__).resolve(strict=True),
        "gpu_discovery_runner": Path(controller.gpu_discovery.__file__).resolve(strict=True),
        "gpu_source_adapter": Path(gpu_source_adapter.__file__).resolve(strict=True),
        "discovery_static_registry": Path(discovery_static_registry.__file__).resolve(strict=True),
        "gpu_source_evidence": Path(evidence.__file__).resolve(strict=True),
        "gpu_source_proofs": Path(gpu_source_proofs.__file__).resolve(strict=True),
        "gpu_discovery_beliefs": Path(autokernel_gpu_discovery_beliefs.__file__).resolve(strict=True),
        "device_claim": Path(device_claim.__file__).resolve(strict=True),
        "device_sampler": Path(device_sampler.__file__).resolve(strict=True),
        "gpu_residency_sampler": Path(gpu_residency_sampler.__file__).resolve(strict=True),
    }
    return {name: {"path": str(path), "sha256": _digest_regular(path, name)}
            for name, path in modules.items()}


def _module_attestor(expected: Mapping[str, Mapping[str, str]]) -> Callable[[], None]:
    sealed = json.loads(json.dumps(dict(expected), sort_keys=True))
    def attest() -> None:
        if _execution_module_identity() != sealed:
            raise DeploymentFactoryError("live execution module bytes changed after graph validation")
    return attest


_STATIC_IDS = MappingProxyType({
    "environment_profile": "sealed-codex",
    "source_builder": "gpu-source-v1",
    "evidence_plan": "reviewed-gpu-source-evidence-v1",
    "runner_args": "qwen05b-tg128",
    "experiment_template_registry": "gpu-source-templates-v1",
    "inference_window_lease": "mi210-window-v1",
    "production_snapshot": "llama-v9-artifacts",
})
_LOAD_PROFILE_ID = "mi210-qwen05b-tg128-fallback-only-v1"
_FALLBACK_ONLY_HEADROOM_SENTINEL = 1 << 60
_ROCPROF_V1 = Path(
    "/mnt/raid0/llm/autokernel/tools/rocprof6.2-extracted/opt/rocm-6.2.0/bin/rocprof")
_ROCPROF_V1_SHA256 = "585e3e6034e3c0bd9e591f0aa72f6156686680911a0b47ed4ece3c9a8372a4b2"
_ROCPROF_V1_INPUT = b"pmc:\n\ngpu:\nrange:\nkernel:\n"
_ROCPROF_V1_PREFIX = ("--tool-version", "1", "--timestamp", "on",
                       "--ctx-wait", "on", "--heartbeat", "30", "-i")
_CORRECTNESS_SUITE_SEED = 2026081301
_INSTRUMENT_PATH = Path("/mnt/raid0/llm/llama.cpp-experimental")
_INSTRUMENT_BRANCH = "codex/autokernel-ready-continue-instrument-20260814"
_INSTRUMENT_COMMIT = "81bf32f11b4a421880e8f25faec3e4ba872363f0"
_INSTRUMENT_DIFF_SHA256 = "3cf9178fcc00e8c1d3dfc0bfd6086edbff6a6eb6ac528aa4d88b23843b5599c2"
_INSTRUMENT_TEST_SOURCE_SHA256 = "6acd4bf95594d5797a54c912630ec56d3e89fcb3a3a43ca96f95152d77589db4"
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
    "ggml/src/ggml-cuda/mmvq.cu": "15d25d71c945de19e8efc9fbfc6b7e5e66f33bc7635f9dc648d9e1f231ba409e",
    "ggml/src/ggml-cuda/rope.cu": "8286f7b57bb76ab490e05d42cb8262ad886b85a8fdaaef63d6538b7ff06940b2",
    "ggml/src/ggml-cuda/norm.cu": "37e670ad50f8b0c3fb9acaaba54ad520b143b0e73994de5c10f8635e334ff0cd",
})
_SAFE_ACTOR_ENVIRONMENT = MappingProxyType({
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/home/node",
    "CODEX_HOME": "/home/node/.codex",
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
})
_SITE_MODEL = Path(
    "/mnt/raid0/llm/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/"
    "Qwen2.5-Coder-0.5B-Q4_K_M.gguf")
_SITE_SOURCE_PLAN = Path("/mnt/raid0/llm/autokernel/surface/gpu_decode_source_plan.json")
_SITE_WINDOW_LOCK = Path("/mnt/raid0/llm/tmp/model-call.lock")
_SITE_ACTOR_WRAPPER = Path(
    "/usr/local/share/npm-global/lib/node_modules/@openai/codex/bin/codex.js")


def _digest_regular(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise DeploymentFactoryError(f"{label} must be a regular non-symlink file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def initialize_static_deployment_bundle(root: Path) -> Path:
    """Emit the one reviewed site bundle; no caller supplies code or argv authority."""
    if not root.is_absolute() or root.is_symlink() or ".." in root.parts:
        raise DeploymentFactoryError("bundle root must be an absolute non-symlink path")
    root.mkdir(parents=True, exist_ok=True)
    config_dir = root / "config"
    for directory in (config_dir, root / "locks"):
        directory.mkdir(parents=True, exist_ok=True)
    model = _bound(_SITE_MODEL, "model")
    source_plan = _bound(_SITE_SOURCE_PLAN, "reviewed_source_plan")
    wrapper_path = _SITE_ACTOR_WRAPPER.resolve(strict=True)
    wrapper = _bound(wrapper_path, "actor_wrapper")
    workload_path, workload_sha = _json_artifact(config_dir / "workload.json", {
        "schema": "epyc.autokernel.discovery_workload.v1", "workload": "decode_tg128",
        "prompt_tokens": 0, "generation_tokens": 128, "calls_per_arm": 9,
        "device_id": "mi210_0", "promotion_claim": False})
    runtime_path, runtime_sha = _json_artifact(config_dir / "runtime.json", {
        "schema": "epyc.autokernel.discovery_runtime.v1", "architecture": "gfx90a",
        "gpu_layers": 99, "flash_attention": True, "hip_graphs": True,
        "cpu_list": "184-191", "threads": 8, "promotion_claim": False})
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
    context = {"schema": deployment.PLANNER_CONTEXT_SCHEMA,
               "model_sha256": model.sha256, "workload_sha256": workload_sha,
               "runtime_config_sha256": runtime_sha,
               "profile_receipts": [{"path": str(source_plan.path), "sha256": source_plan.sha256}],
               "hotspots": hotspots,
               "source_constraints": {"template_registry": "gpu-source-templates-v1",
                                      "one_reviewed_file_per_candidate": True,
                                      "excluded_source_plan_fields": ["planner_posture", "current_execution",
                                                                       "max_overlap_bytes", "overlap_policy"]},
               "initial_strategies": [
                   "Do not repeat retired fattn single-column, Q5 four-wave, Q8 vec4, or RMS128 variants.",
                   "Explore a new literal dispatch-bound hypothesis in one reviewed template.",
                   "Treat prior RoPE64 and Q4_K one-wave results as DNR/top-K context, not promotion evidence."]}
    context["context_sha256"] = schemas.content_hash(context)
    context_path, context_sha = _json_artifact(config_dir / "planner-context.json", context)
    for directory in (root / "state", root / "evidence", root / "operations"):
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
                            "max_iterations": 100, "nomination_threshold": .03},
             "actors": {"wrapper_path": str(wrapper.path), "wrapper_sha256": wrapper.sha256,
                        "environment_profile_id": _STATIC_IDS["environment_profile"]},
             "gpu": {"device_id": "mi210_0", "claim_timeout_s": 0,
                     "inference_window_lock": str(_SITE_WINDOW_LOCK),
                     "inference_window_lease_id": _STATIC_IDS["inference_window_lease"]},
             "immutable_inputs": {"model": {"path": str(model.path), "sha256": model.sha256},
                                  "workload": {"path": str(workload_path), "sha256": workload_sha},
                                  "runtime_config": {"path": str(runtime_path), "sha256": runtime_sha},
                                  "admission_policy": {"path": str(policy_path), "sha256": policy_sha}},
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


def _production_runtime_snapshot(root: Path) -> tuple[tuple[evidence.BoundInputFile, ...], dict[str, Any]]:
    """Bind CPU/HIP server+bench and their complete local runtime topology."""
    files: dict[Path, evidence.BoundInputFile] = {}
    semantics: dict[str, Any] = {"production_head": deployment.FROZEN_PRODUCTION_HEAD,
                                 "closures": {}}
    for flavor, required in (("build", frozenset()), ("build-hip", frozenset({"libggml-hip.so.0"}))):
        directory = root / flavor / "bin"
        todo = [directory / "llama-server", directory / "llama-bench",
                *(directory / name for name in sorted(required))]
        topology: dict[str, Any] = {}
        seen_names: set[str] = set()
        while todo:
            lexical = todo.pop()
            if lexical.name in seen_names:
                continue
            seen_names.add(lexical.name)
            if not lexical.exists():
                raise DeploymentFactoryError(f"production runtime closure lacks {lexical}")
            resolved = lexical.resolve(strict=True)
            files.setdefault(resolved, _bound(resolved, f"production-runtime:{flavor}:{lexical.name}"))
            completed = subprocess.run(("/usr/bin/readelf", "-dW", str(resolved)),
                                       check=False, stdin=subprocess.DEVNULL,
                                       capture_output=True, text=True)
            if completed.returncode:
                raise DeploymentFactoryError(f"cannot inspect production runtime object {resolved}")
            needed = sorted(re.findall(r"Shared library: \[(.+?)\]", completed.stdout))
            local = []
            for name in needed:
                target = directory / name
                if target.exists():
                    local.append(name); todo.append(target)
            topology[lexical.name] = {
                "resolved_name": resolved.name,
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                "needed_local": local,
                "symlink": os.readlink(lexical) if lexical.is_symlink() else None}
        if not required.issubset(seen_names):
            raise DeploymentFactoryError(f"{flavor} production runtime lacks its reviewed backend")
        semantics["closures"][flavor] = {
            "configuration": "cpu" if flavor == "build" else "rocm-gfx90a",
            "entrypoints": ["llama-bench", "llama-server"],
            "topology": topology}
    return tuple(files[path] for path in sorted(files)), semantics


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
        ("cuda-fattn-v1", "ggml/src/ggml-cuda/fattn.cu",
         "ggml_cuda_get_best_fattn_kernel", ("ggml_cuda_get_best_fattn_kernel",
          "ggml_cuda_flash_attn_ext", "ggml_cuda_flash_attn_ext_vec",
          "ggml_cuda_flash_attn_ext_mma_f16", "ggml_cuda_flash_attn_ext_supported",
          "ggml_cuda_flash_attn_ext_get_alloc_size"),
         ("fattn", "flash_attn"), "FLASH_ATTN_EXT", 2868,
         ({"trace": "VEC selector", "file": "ggml/src/ggml-cuda/fattn.cu",
           "symbol": "ggml_cuda_get_best_fattn_kernel"},
          {"trace": "TILE geometry", "file": "ggml/src/ggml-cuda/fattn.cu",
           "symbol": "ggml_cuda_get_best_fattn_kernel"})),
        ("cuda-mmvq-v1", "ggml/src/ggml-cuda/mmvq.cu",
         "ggml_cuda_op_mul_mat_vec_q", ("ggml_cuda_op_mul_mat_vec_q",
          "ggml_cuda_mul_mat_vec_q", "mul_mat_vec_q_switch_type",
          "mul_mat_vec_q_switch_ncols_dst", "mul_mat_vec_q_moe_launch",
          "mul_mat_vec_q_switch_fusion", "mul_mat_vec_q8_0_prefetch_launch"),
         ("mmvq", "mul_mat_vec"), "MUL_MAT", 1139,
         tuple({"trace": f"{quant} dispatch", "file": "ggml/src/ggml-cuda/mmvq.cu",
                "symbol": "mul_mat_vec_q_switch_type"} for quant in ("Q4", "Q5", "Q6"))),
        ("cuda-rope-v1", "ggml/src/ggml-cuda/rope.cu",
         "ggml_cuda_op_rope_impl", ("ggml_cuda_op_rope_impl", "ggml_cuda_op_rope",
          "ggml_cuda_op_rope_back", "ggml_cuda_op_rope_fused", "rope_norm",
          "rope_neox", "rope_multi", "rope_vision", "rope_norm_cuda",
          "rope_neox_cuda", "rope_multi_cuda", "rope_vision_cuda"),
         ("rope",), "ROPE", 428,
         ({"trace": "ROPE dispatch", "file": "ggml/src/ggml-cuda/rope.cu",
           "symbol": "ggml_cuda_op_rope_impl"},)),
        ("cuda-norm-v1", "ggml/src/ggml-cuda/norm.cu",
         "ggml_cuda_op_rms_norm", ("ggml_cuda_op_norm", "ggml_cuda_op_group_norm",
          "ggml_cuda_op_rms_norm", "ggml_cuda_op_rms_norm_fused",
          "ggml_cuda_op_rms_norm_fused_add", "ggml_cuda_op_rms_norm_back",
          "ggml_cuda_op_l2_norm", "norm_f32", "group_norm_f32", "rms_norm_f32",
          "rms_norm_back_f32", "l2_norm_f32", "norm_f32_cuda",
          "group_norm_f32_cuda", "rms_norm_f32_cuda", "rms_norm_mul_f32_cuda",
          "rms_norm_back_f32_cuda", "l2_norm_f32_cuda"),
         ("norm", "rms_norm", "group_norm", "l2_norm"), "RMS_NORM", 21,
         ({"trace": "RMS_NORM dispatch", "file": "ggml/src/ggml-cuda/norm.cu",
           "symbol": "ggml_cuda_op_rms_norm"},)),
    )
    templates = {}
    for template_id, path, symbol, symbols, prefixes, correctness_op, cases, replays in families:
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
                       "correctness_op": correctness_op,
                       "expected_correctness_cases": cases,
                       "suite_seed": _CORRECTNESS_SUITE_SEED,
                       "test_source_commit": _INSTRUMENT_COMMIT,
                       "test_source_sha256": _INSTRUMENT_TEST_SOURCE_SHA256,
                       "production_instrument_target_sha256": _TARGET_SOURCE_SHA256[path],
                       "manual_replay_traces": list(replays),
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
        correctness_tool = _bound(
            build_.candidate_build / "bin" / "test-backend-ops", "executable")
        # The reviewed instrument, not the planner, owns deterministic-suite
        # capability.  Check the built tool before spending a device claim.
        capability = subprocess.run((str(correctness_tool.path), "--help"),
                                    check=False, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, env={
                                        "HIP_VISIBLE_DEVICES": "0",
                                        "LD_LIBRARY_PATH": (
                                            f"{identities.candidate.hip_library.path.parent}:"
                                            "/opt/rocm/lib"),
                                        "PATH": "/opt/rocm/bin:/usr/bin:/bin",
                                        "ROCM_PATH": "/opt/rocm"})
        if (capability.returncode != 0
                or "--suite-seed <u64>" not in capability.stdout):
            raise DeploymentFactoryError(
                "candidate correctness tool lacks reviewed deterministic suite support")
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
        shared = identities.shared_runtime
        reward_binary = shared.measurement_binary
        profile_argv = (
            str(profiler.path), *_ROCPROF_V1_PREFIX, str(timestamp_input.path),
            "-o", evidence.ROCPROF_TIMESTAMP_OUTPUT,
            "/usr/bin/taskset", "-c", "184-191", str(reward_binary.path),
            "-m", str(config.model.path), "-p", "0", "-n", "128", "-r", "1",
            "-ngl", "99", "-fa", "on", "-t", "8")
        profiler_prefix = _ROCPROF_V1.parents[1]
        common_environment = (
            ("GGML_CUDA_DISABLE_GRAPHS", "1"), ("HIP_VISIBLE_DEVICES", "0"),
            ("PATH", f"{profiler_prefix / 'bin'}:/opt/rocm/bin:/usr/bin:/bin"),
            ("ROCM_PATH", "/opt/rocm"),
            ("ROCP_METRICS", str(profiler_prefix / "lib/rocprofiler/metrics.xml")))
        def profile_environment(hip: evidence.BoundInputFile) -> tuple[tuple[str, str], ...]:
            return tuple(sorted((*common_environment, ("LD_LIBRARY_PATH",
                f"{hip.path.parent}:{reward_binary.path.parent}:{profiler_prefix / 'lib'}:/opt/rocm/lib"))))
        placeholder = evidence.BoundInputFile(
            "execution_policy",
            (config.operations_root / "materialization" / build_.operation_key
             / "evidence-policy.json").resolve(), "0" * 64)
        provisional = evidence.GpuSourceEvidencePlan(
            campaign_id=candidate.source_manifest.campaign_id,
            device_id=config.device_id,
            manifest_sha256=candidate.source_manifest_sha256,
            model_sha256=config.model.sha256,
            workload_sha256=config.workload.sha256,
            runtime_config_sha256=config.runtime_config.sha256,
            candidate=build_.candidate_identity, anchor=build_.anchor_identity,
            correctness_argv=correctness_argv,
            correctness_summary_pattern=(
                rf"(?s)(?P<passed>\d+)/(?P<total>\d+) tests passed.*"
                rf"Backend ROCm0: .*OK.*1/1 backends passed"),
            expected_correctness_cases=cases,
            candidate_rocprof_argv=profile_argv, anchor_rocprof_argv=profile_argv,
            dispatch=template.bind_dispatch(candidate.experiment_intent),
            identity_files=identities, policy=placeholder,
            correctness_inputs=(correctness_tool, identities.candidate.binary,
                                identities.candidate.config, identities.candidate.linkage),
            candidate_rocprof_inputs=(profiler, timestamp_input, reward_binary,
                                      identities.model, identities.workload,
                                      identities.runtime_config),
            anchor_rocprof_inputs=(profiler, timestamp_input, reward_binary,
                                   identities.model, identities.workload,
                                   identities.runtime_config),
            required_correctness_argv_paths=(correctness_tool.path,),
            required_candidate_rocprof_argv_paths=(reward_binary.path, identities.model.path),
            required_anchor_rocprof_argv_paths=(reward_binary.path, identities.model.path),
            execution_cwd=build_.candidate_build.resolve(strict=True),
            correctness_environment=tuple(sorted((
                ("GGML_CUDA_DISABLE_GRAPHS", "1"), ("HIP_VISIBLE_DEVICES", "0"),
                ("LD_LIBRARY_PATH",
                 f"{identities.candidate.hip_library.path.parent}:/opt/rocm/lib"),
                ("PATH", "/opt/rocm/bin:/usr/bin:/bin"), ("ROCM_PATH", "/opt/rocm")))),
            candidate_rocprof_environment=profile_environment(shared.candidate_hip_library),
            anchor_rocprof_environment=profile_environment(shared.anchor_hip_library),
            shared_runtime=shared)
        policy_path = placeholder.path
        raw = json.dumps(evidence._policy_payload(provisional), sort_keys=True,
                         separators=(",", ":")).encode()
        if policy_path.exists():
            if policy_path.is_symlink() or policy_path.read_bytes() != raw:
                raise DeploymentFactoryError("sealed evidence policy changed for operation")
        else:
            policy_path.write_bytes(raw)
        policy = evidence.BoundInputFile(
            "execution_policy", policy_path,
            hashlib.sha256(raw).hexdigest())
        return replace(provisional, policy=policy)
    return EvidencePlanBinding(build=build)


def _runner_binding(config: deployment.DiscoveryDeployment) -> RunnerArgsBinding:
    def build(_candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild,
              permit: Mapping[str, Any]) -> Any:
        operation_key = permit.get("operation_key")
        repetition = permit.get("repetition")
        if operation_key != build_.operation_key or repetition not in {1, 2}:
            raise DeploymentFactoryError("runner operation identity differs from sealed build")
        output = config.operations_root / str(operation_key) / "runner" / f"s{repetition}"
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
        decision_path = output / "load-admission-decision.json"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_raw = (json.dumps(dict(decision), sort_keys=True, indent=2) + "\n").encode()
        if decision_path.exists():
            if decision_path.is_symlink() or decision_path.read_bytes() != decision_raw:
                raise DeploymentFactoryError("runner load-admission carrier changed")
        else:
            decision_path.write_bytes(decision_raw)
        argv = ["--anchor-build", str(build_.anchor_build), "--candidate-build", str(build_.candidate_build),
                "--model", str(config.model.path), "--output-dir", str(output),
                "--campaign-id", f"ak-discovery-{config.config_sha256[:16]}",
                "--factor", "source_patch", "--calls", "9", "--workload", "decode_tg128",
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
    profiles = [profile for profile in config.admission_policy.corpus.profiles
                if (profile.model_sha256 == config.model.sha256
                    and profile.model_path == str(config.model.path)
                    and profile.model_bytes == config.model.path.stat().st_size
                    and profile.workload == "decode_tg128" and profile.calls_per_arm == 9
                    and profile.device_id == config.device_id)]
    if len(profiles) != 1:
        raise DeploymentFactoryError("sealed admission corpus lacks one exact runner profile")
    environment = EnvironmentProfile(_SAFE_ACTOR_ENVIRONMENT)
    source_builder = discovery_static_registry.StaticGpuSourceBuilder(
        production_path=config.production_path,
        production_branch=deployment.FROZEN_PRODUCTION_BRANCH,
        instrument_path=config.instrument_path,
        operations_root=config.operations_root,
        build_root=config.operations_root / "build",
        cmake_defines=(("GGML_HIP", "ON"), ("AMDGPU_TARGETS", "gfx90a"),
                       ("GGML_NATIVE", "OFF")))
    snapshot_files, snapshot_semantics = _production_runtime_snapshot(config.production_path)
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
                        runtime: Mapping[str, Any], templates: ExperimentTemplateRegistry,
                        target_equality: tuple[Path, str],
                        instrument_review: tuple[Path, str],
                        execution_modules: Mapping[str, Mapping[str, str]],
                        production_runtime_sha256: str) -> tuple[Path, str]:
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
            "execution_modules": dict(execution_modules),
            "environment_profile": dict(_SAFE_ACTOR_ENVIRONMENT),
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
                               "ready_continue_schema": "epyc.autokernel.ready_continue.v1",
                               "instrument_commit": _INSTRUMENT_COMMIT,
                               "contract_source_sha256": _READY_CONTINUE_CONTRACT_SHA256,
                               "early_unlock_enabled": False,
                               "trust_limit": "cooperative_same_uid_not_launch_authority",
                               "safe_fallback": "full_process_cold_serialized_lock"},
            "instrument_target_equality": {"path": str(target_equality[0]),
                                           "sha256": target_equality[1]},
            "production_runtime_snapshot_sha256": production_runtime_sha256,
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
    target_equality = _target_source_equality_receipt(config)
    instrument_review = _instrument_review_receipt(config)
    execution_modules = _execution_module_identity()
    templates = _template_registry()
    registry = _static_registry(config, templates)
    production_snapshot = _require(
        registry["production_snapshot"][_STATIC_IDS["production_snapshot"]],
        ProductionSnapshotBinding, "production_snapshot")
    runtime = codex_container_actor.runtime_identity(config.actor_wrapper.path)
    launcher_sha256 = _digest_regular(Path(codex_container_actor.__file__).resolve(),
                                      "Codex actor launcher")
    sampler = gpu_residency_sampler.Mi210ResidencySampler()
    executor = evidence.SubprocessCommandExecutor(
        residency_sampler=sampler,
        runtime_maps_sampler=discovery_static_registry.runtime_maps_sampler())
    journal = device_claim.ClaimJournal(config.operations_root / "claims" / "device.jsonl")
    adapters = materialize(config, registry, correctness_executor=executor,
                           rocprof_executor=executor, claim_journal=journal,
                           runner_attest=_module_attestor(execution_modules))
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
    receipt, digest = _seal_graph_receipt(
        config, runtime, templates, target_equality, instrument_review,
        execution_modules,
        production_snapshot.runtime_semantics_sha256)
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
            # This deployment is explicitly fallback-only and deliberately has
            # no telemetry authority.  Enabling overlap requires a new reviewed
            # policy/binding, not an ambient or caller-supplied observation.
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
    if template is None or len(manifest.declared_files) != 1:
        raise DeploymentFactoryError(
            "discovery intent must select one exact reviewed file")
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
        runner_attest()
        template = templates.resolve(candidate.experiment_intent)
        _validate_source_scope(candidate, template)
        candidate.source_manifest.bind(
            proposal=candidate.proposal, campaign_id=candidate.source_manifest.campaign_id,
            candidate_id=candidate.source_manifest.candidate_id,
            production_base_commit=config.production_head,
            instrument_commit=config.instrument_commit)
        permit = {**permit, "instrument_branch": config.instrument_branch,
                  "deployment_config_sha256": config.config_sha256}
        snapshot.revalidate()
        return source.build(candidate, authorization, permit)
    def plan(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild):
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
        result = plans.build(candidate, build_, template)
        expected_dispatch = template.bind_dispatch(candidate.experiment_intent)
        if result.dispatch != expected_dispatch or result.model_sha256 != config.model.sha256:
            raise DeploymentFactoryError("evidence plan does not bind configured model/selected reviewed template")
        return result
    def args(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild, permit: Mapping[str, Any]):
        config.revalidate()
        runner_attest()
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
        protected_roots=(config.production_path, config.instrument_path),
        protected_files=snapshot.files, runner_attest=runner_attest)
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
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--deployment")
    authority.add_argument("--initialize-bundle",
                           help="emit the fixed-site sealed deployment bundle")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true",
                       help="alias for validate-only; never calls an actor or hardware")
    args = parser.parse_args(argv)
    if args.initialize_bundle:
        if args.validate_only or args.dry_run:
            parser.error("bundle initialization does not accept execution flags")
        result = initialize_static_deployment_bundle(Path(args.initialize_bundle))
        print(json.dumps({"status": "initialized", "inference_executed": False,
                          "deployment": str(result)}, sort_keys=True))
        return 0
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
