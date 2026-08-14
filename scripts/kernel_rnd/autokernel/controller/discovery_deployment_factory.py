"""Trusted concrete deployment materializer for governed GPU source discovery.

The JSON configuration merely selects IDs.  This module is the one static
bridge from those IDs to typed, registered Python construction seams.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
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
        decision = gpu_load_admission.arbitrate(corpus, request)
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
            instrument_commit=config.production_head)
        return source.build(candidate, authorization, permit)
    def plan(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild):
        config.revalidate()
        template = templates.resolve(candidate.experiment_intent)
        result = plans.build(candidate, build_, template)
        if result.dispatch != template.dispatch or result.model_sha256 != config.model.sha256:
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
        if (getattr(result, "factor", None) != "source_patch"
                or str(getattr(result, "model", "")) != str(config.model.path)
                or str(getattr(result, "anchor_build", "")) != str(build_.anchor_build)
                or str(getattr(result, "candidate_build", "")) != str(build_.candidate_build)
                or str(getattr(result, "measurement_binary", "")) != str(build_.measurement_binary)
                or str(getattr(result, "common_loader_dir", "")) != str(build_.common_loader_dir)
                or str(getattr(result, "anchor_loader_dir", "")) != str(build_.anchor_loader_dir)
                or str(getattr(result, "candidate_loader_dir", "")) != str(build_.candidate_loader_dir)
                or getattr(result, "promotion_claim", False) is not False
                or str(getattr(result, "inference_window_lock", "")) != str(config.inference_window_lock)):
            raise DeploymentFactoryError("runner arguments do not bind source builds/model/window/discovery authority")
        return result
    screener = gpu_source_adapter.build_governed_gpu_source_adapter(
        operations_root=config.operations_root, build_source=build, plan_factory=plan,
        args_factory=args, correctness_executor=correctness_executor,
        rocprof_executor=rocprof_executor, claim_journal=claim_journal,
        claim_acquirer=claim_acquirer, claim_verifier=claim_verifier,
        claim_timeout_s=config.claim_timeout_s, receipt_series=receipt_series,
        protected_roots=(config.production_path,), protected_files=snapshot.files)
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
            "admission_policy_version": config.admission_policy.corpus.version}),
        production_base_commit=config.production_head,
        instrument_commit=config.production_head,
        experiment_template_registry_sha256=config.experiment_template_registry_sha256,
        admission_corpus_sha256=config.admission_policy.value["policy_sha256"],
        admission_corpus_version=config.admission_policy.corpus.version,
        # The sealed deployment digest, not a caller argument, namespaces all
        # controller/worktree/receipt identities across concurrent deployments.
        campaign_id=f"ak-discovery-{config.config_sha256[:16]}")


def deployment_main(argv: list[str] | None, *, registry: Mapping[str, Mapping[str, object]],
                    correctness_executor: evidence.CommandExecutor, rocprof_executor: evidence.CommandExecutor,
                    claim_journal: device_claim.ClaimJournal) -> int:
    """Dedicated launcher; refuses generic factory/CLI override authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = deployment.load_deployment_config(Path(args.deployment))
    adapters = materialize(config, registry,
                           correctness_executor=correctness_executor,
                           rocprof_executor=rocprof_executor,
                           claim_journal=claim_journal)
    controller.run_controller(controller_config(config, dry_run=args.dry_run), **adapters)
    return 0
