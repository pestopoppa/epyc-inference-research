"""Trusted concrete deployment materializer for governed GPU source discovery.

The JSON configuration merely selects IDs.  This module is the one static
bridge from those IDs to typed, registered Python construction seams.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Any, Callable, Mapping

from ..execution import inference_window
from ..resource import device_claim
from . import discovery_controller as controller
from . import discovery_deployment as deployment
from . import gpu_source_adapter
from . import gpu_source_evidence as evidence


class DeploymentFactoryError(RuntimeError): pass
_BLOCKED_ENV = frozenset({"LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "DYLD_INSERT_LIBRARIES"})


@dataclass(frozen=True)
class EnvironmentProfile:
    values: Mapping[str, str]
    def __post_init__(self) -> None:
        if not self.values or any(not isinstance(key, str) or not key or not isinstance(value, str)
                                  for key, value in self.values.items()):
            raise DeploymentFactoryError("environment profile must be an exact string mapping")
        if _BLOCKED_ENV.intersection(self.values):
            raise DeploymentFactoryError("environment profile contains a loader injection key")


@dataclass(frozen=True)
class SourceBuilderBinding:
    build: Callable[[controller.PlannedCandidate, Any, Mapping[str, Any]], controller.GpuSourceBuild]

@dataclass(frozen=True)
class EvidencePlanBinding:
    build: Callable[[controller.PlannedCandidate, controller.GpuSourceBuild], evidence.GpuSourceEvidencePlan]

@dataclass(frozen=True)
class RunnerArgsBinding:
    build: Callable[[controller.PlannedCandidate, controller.GpuSourceBuild, Mapping[str, Any]], Any]

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
        try:
            # Admission is non-blocking and does not retain a serializable fd.
            # The registered runner arguments must bind this exact lock for the
            # actual model call, where the producer's windowed runner owns it.
            held = inference_window.InferenceCallWindow(
                self.config.inference_window_lock, timeout_s=0).acquire()
        except inference_window.InferenceWindowTimeout:
            return {"admitted": False, "reason": "inference_window_busy", "mode": self.mode}
        held.release()
        return {"admitted": True, "mode": self.mode,
                "device_id": self.config.device_id,
                "inference_window_lock": str(self.config.inference_window_lock),
                "model_sha256": self.config.model.sha256,
                "small_model_max_bytes": self.config.small_model_max_bytes,
                "promotion_claim": False}


def _require(value: object, type_: type, label: str) -> Any:
    if not isinstance(value, type_): raise DeploymentFactoryError(f"registry entry {label} has wrong typed binding")
    return value

_FORBIDDEN_MUTATION_PREFIXES = ("tools/", "examples/", "scripts/", "tests/", "cmake/", "CMakeLists.txt")
_ALLOWED_KERNEL_PREFIXES = ("ggml/src/", "ggml/include/", "ggml/src/ggml-hip/")

def _validate_source_scope(candidate: controller.PlannedCandidate) -> None:
    manifest = candidate.source_manifest
    if manifest.source_tree != "llama.cpp":
        raise DeploymentFactoryError("discovery source patch must target the allowlisted llama.cpp scope")
    for path in manifest.declared_files:
        if path.startswith(_FORBIDDEN_MUTATION_PREFIXES) or not path.startswith(_ALLOWED_KERNEL_PREFIXES):
            raise DeploymentFactoryError("discovery patch may only modify allowlisted GGML kernel sources")


def materialize(config: deployment.DiscoveryDeployment, registry: Mapping[str, Mapping[str, object]], *,
                planner: controller.Planner, critic: controller.Critic,
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
    if not isinstance(resolved.dispatch_contract, evidence.DispatchContract):
        raise DeploymentFactoryError("registry dispatch contract must be typed")
    lease = lease_binding.make(config)

    def build(candidate: controller.PlannedCandidate, authorization: Any, permit: Mapping[str, Any]):
        config.revalidate()
        _validate_source_scope(candidate)
        return source.build(candidate, authorization, permit)
    def plan(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild):
        config.revalidate()
        result = plans.build(candidate, build_)
        if result.dispatch != resolved.dispatch_contract or result.model_sha256 != config.model.sha256:
            raise DeploymentFactoryError("evidence plan does not bind configured model/dispatch contract")
        return result
    def args(candidate: controller.PlannedCandidate, build_: controller.GpuSourceBuild, permit: Mapping[str, Any]):
        config.revalidate()
        result = runner.build(candidate, build_, permit)
        if (getattr(result, "factor", None) != "source_patch"
                or str(getattr(result, "model", "")) != str(config.model.path)
                or str(getattr(result, "anchor_build", "")) != str(build_.anchor_build)
                or str(getattr(result, "candidate_build", "")) != str(build_.candidate_build)
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
        nomination_threshold=config.nomination_threshold, dry_run=dry_run)


def deployment_main(argv: list[str] | None, *, registry: Mapping[str, Mapping[str, object]],
                    planner: controller.Planner, critic: controller.Critic,
                    correctness_executor: evidence.CommandExecutor, rocprof_executor: evidence.CommandExecutor,
                    claim_journal: device_claim.ClaimJournal) -> int:
    """Dedicated launcher; refuses generic factory/CLI override authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = deployment.load_deployment_config(Path(args.deployment))
    adapters = materialize(config, registry, planner=planner, critic=critic,
                           correctness_executor=correctness_executor,
                           rocprof_executor=rocprof_executor,
                           claim_journal=claim_journal)
    controller.run_controller(controller_config(config, dry_run=args.dry_run), **adapters)
    return 0
