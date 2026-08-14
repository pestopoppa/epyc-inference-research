#!/usr/bin/env python3
"""Declarative, registry-only configuration for GPU source discovery deployment.

This boundary intentionally does *not* import the governed GPU producer.  It
only reads a sealed deployment description and resolves a small set of opaque
IDs against a registry constructed by trusted deployment code.  In particular,
a configuration file cannot name a Python module, callable, argv, or environment
variable.  The later integration factory is consequently the only place that
can connect these already-validated values to executable adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .. import schemas
from . import gpu_load_admission


SCHEMA = "epyc.autokernel.discovery_deployment.v1"
FROZEN_PRODUCTION_PATH = Path("/mnt/raid0/llm/llama.cpp")
FROZEN_PRODUCTION_HEAD = "0db32c06e3e550065b78311a6031ef3dd2c4f27c"
FROZEN_PRODUCTION_BRANCH = "production-consolidated-v9"
ALLOWED_DEVICE_IDS = frozenset({"mi210_0"})
SHA = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class DeploymentConfigError(RuntimeError):
    """The declarative deployment boundary refused an unsealed configuration."""


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DeploymentConfigError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise DeploymentConfigError(f"{label} must be a registry identifier")
    return value


def _digest_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise DeploymentConfigError(f"{label} must be a SHA-256 digest")
    return value


def _absolute(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise DeploymentConfigError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise DeploymentConfigError(f"{label} must be an absolute path")
    absolute = path.absolute()
    resolved = path.resolve(strict=False)
    # `resolve()` before lstat turns a symlink into its target and makes the
    # subsequent check meaningless.  Reject both a final symlink and any
    # symlinked ancestor that changes the lexical authority boundary.
    if path.is_symlink() or absolute != resolved:
        raise DeploymentConfigError(f"{label} must not traverse a symlink")
    return resolved


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _under(left, right) or _under(right, left)


def _digest_file(path: Path, expected: object, label: str) -> str:
    if not isinstance(expected, str) or not SHA.fullmatch(expected):
        raise DeploymentConfigError(f"{label} must carry a SHA-256 digest")
    if path.is_symlink() or not path.is_file():
        raise DeploymentConfigError(f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise DeploymentConfigError(f"{label} bytes do not match declared digest")
    return digest


@dataclass(frozen=True)
class ImmutableInput:
    path: Path
    sha256: str

    def revalidate(self, label: str) -> None:
        _digest_file(self.path, self.sha256, label)


PLANNER_CONTEXT_SCHEMA = "epyc.autokernel.discovery_planner_context.v1"
ADMISSION_POLICY_SCHEMA = gpu_load_admission.POLICY_SCHEMA
_PLANNER_CONTEXT_LIMIT = 512 * 1024


@dataclass(frozen=True)
class PlannerContext:
    """Small, sealed source/profile brief that is safe to hand to both actors."""
    input: ImmutableInput
    value: Mapping[str, Any]

    def revalidate(self) -> None:
        self.input.revalidate("planner_context")


@dataclass(frozen=True)
class AdmissionPolicy:
    input: ImmutableInput
    value: Mapping[str, Any]
    corpus: gpu_load_admission.PolicyCorpus

    def revalidate(self) -> None:
        self.input.revalidate("admission_policy")
        refreshed = gpu_load_admission.load_policy_corpus(
            self.input.path, expected_file_sha256=self.input.sha256)
        if refreshed.policy_sha256 != self.corpus.policy_sha256:
            raise DeploymentConfigError("admission policy semantic identity changed")


def _admission_policy(raw: ImmutableInput, *, model: ImmutableInput,
                      workload: ImmutableInput) -> AdmissionPolicy:
    try:
        corpus = gpu_load_admission.load_policy_corpus(
            raw.path, expected_file_sha256=raw.sha256)
    except gpu_load_admission.AdmissionPolicyError as exc:
        raise DeploymentConfigError("admission policy schema/content mismatch") from exc
    if not any(profile.model_sha256 == model.sha256 for profile in corpus.profiles):
        raise DeploymentConfigError("admission policy does not bind the sealed model")
    return AdmissionPolicy(input=raw, value=corpus.sealed, corpus=corpus)


def _planner_context(value: object, *, model: ImmutableInput,
                     workload: ImmutableInput, runtime_config: ImmutableInput) -> PlannerContext:
    raw = _input(value, "planner_context")
    if raw.path.stat().st_size > _PLANNER_CONTEXT_LIMIT:
        raise DeploymentConfigError("planner_context exceeds its bounded actor input limit")
    try:
        body = json.loads(raw.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentConfigError("planner_context is not JSON") from exc
    required = {"schema", "context_sha256", "model_sha256", "workload_sha256",
                "runtime_config_sha256", "profile_receipts", "hotspots",
                "source_constraints", "initial_strategies"}
    if not isinstance(body, Mapping) or set(body) != required:
        raise DeploymentConfigError("planner_context has an unknown or incomplete schema")
    if body["schema"] != PLANNER_CONTEXT_SCHEMA:
        raise DeploymentConfigError("planner_context schema mismatch")
    expected = schemas.content_hash({key: item for key, item in body.items()
                                     if key != "context_sha256"})
    if body.get("context_sha256") != expected:
        raise DeploymentConfigError("planner_context self-hash mismatch")
    if (body.get("model_sha256"), body.get("workload_sha256"),
            body.get("runtime_config_sha256")) != (model.sha256, workload.sha256,
                                                      runtime_config.sha256):
        raise DeploymentConfigError("planner_context is not bound to sealed model/workload/runtime")
    receipts = body["profile_receipts"]
    hotspots = body["hotspots"]
    if (not isinstance(receipts, list) or len(receipts) > 64
            or not all(isinstance(row, Mapping) and set(row) == {"path", "sha256"}
                       and isinstance(row["path"], str) and SHA.fullmatch(str(row["sha256"]))
                       for row in receipts)):
        raise DeploymentConfigError("planner_context profile receipts are malformed")
    for row in receipts:
        receipt = _absolute(row["path"], "planner_context.profile_receipt.path")
        _digest_file(receipt, row["sha256"], "planner_context.profile_receipt")
    required_hotspot = {"surface", "symbol", "share", "source_path", "source_sha256",
                        "source_excerpt", "source_excerpt_sha256"}
    if (not isinstance(hotspots, list) or len(hotspots) > 128
            or not all(isinstance(row, Mapping) and set(row).issubset(required_hotspot | {"note"})
                       and required_hotspot.issubset(row)
                       and isinstance(row["surface"], str) and isinstance(row["symbol"], str)
                       and isinstance(row["share"], (int, float)) and not isinstance(row["share"], bool)
                       and math.isfinite(float(row["share"]))
                       and isinstance(row["source_excerpt"], str) and len(row["source_excerpt"]) <= 8192
                       and isinstance(row["source_path"], str)
                       and SHA.fullmatch(str(row["source_sha256"]))
                       and SHA.fullmatch(str(row["source_excerpt_sha256"]))
                       for row in hotspots)):
        raise DeploymentConfigError("planner_context hotspots are malformed or unbounded")
    for row in hotspots:
        source = _absolute(row["source_path"], "planner_context.hotspot.source_path")
        if not _under(source, FROZEN_PRODUCTION_PATH.resolve(strict=True)):
            raise DeploymentConfigError("planner_context source excerpt is outside frozen production")
        _digest_file(source, row["source_sha256"], "planner_context.hotspot.source")
        if hashlib.sha256(row["source_excerpt"].encode("utf-8")).hexdigest() != row["source_excerpt_sha256"]:
            raise DeploymentConfigError("planner_context source excerpt hash mismatch")
        if row["source_excerpt"] not in source.read_text(encoding="utf-8"):
            raise DeploymentConfigError("planner_context excerpt does not occur in sealed source")
    if (not isinstance(body["source_constraints"], Mapping)
            or not isinstance(body["initial_strategies"], list)
            or len(body["initial_strategies"]) > 64
            or any(not isinstance(row, str) or not row for row in body["initial_strategies"])):
        raise DeploymentConfigError("planner_context strategies/constraints are malformed")
    return PlannerContext(input=raw, value=dict(body))


@dataclass(frozen=True)
class DiscoveryDeployment:
    config_sha256: str
    production_path: Path
    production_head: str
    state_root: Path
    evidence_root: Path
    operations_root: Path
    max_iterations: int
    nomination_threshold: float
    actor_wrapper: ImmutableInput
    environment_profile_id: str
    device_id: str
    claim_timeout_s: float
    inference_window_lock: Path
    model: ImmutableInput
    workload: ImmutableInput
    runtime_config: ImmutableInput
    policy: ImmutableInput
    admission_policy: AdmissionPolicy
    planner_context: PlannerContext
    source_builder_id: str
    evidence_plan_id: str
    runner_args_id: str
    experiment_template_registry_id: str
    experiment_template_registry_sha256: str
    inference_window_lease_id: str
    production_snapshot_id: str

    def revalidate(self) -> None:
        """Close the parse-to-start TOCTOU gap for every sealed file reference."""
        _verify_production(self.production_path, self.production_head)
        self.actor_wrapper.revalidate("actors.wrapper")
        for label, value in (("model", self.model), ("workload", self.workload),
                             ("runtime_config", self.runtime_config), ("policy", self.policy)):
            value.revalidate(label)
        self.admission_policy.revalidate()
        self.planner_context.revalidate()


@dataclass(frozen=True)
class ResolvedDeployment:
    """Opaque trusted objects selected by immutable registry IDs only."""
    config: DiscoveryDeployment
    environment_profile: object
    source_builder: object
    evidence_plan: object
    runner_args: object
    experiment_template_registry: object
    inference_window_lease: object
    production_snapshot: object


def _input(value: object, label: str) -> ImmutableInput:
    raw = _exact(value, {"path", "sha256"}, label)
    path = _absolute(raw["path"], f"{label}.path")
    return ImmutableInput(path=path, sha256=_digest_file(path, raw["sha256"], label))


def _verify_production(path: Path, declared_head: str) -> None:
    expected = FROZEN_PRODUCTION_PATH.resolve(strict=True)
    if path.resolve(strict=True) != expected or declared_head != FROZEN_PRODUCTION_HEAD:
        raise DeploymentConfigError("production identity is not the exact frozen tree/head")
    def git(*args: str) -> str:
        completed = subprocess.run(("git", "-C", str(expected), *args),
                                   check=False, capture_output=True, text=True)
        if completed.returncode:
            raise DeploymentConfigError("frozen production Git state could not be verified")
        return completed.stdout.strip()
    if git("rev-parse", "HEAD") != FROZEN_PRODUCTION_HEAD:
        raise DeploymentConfigError("frozen production HEAD differs from declared freeze")
    if git("branch", "--show-current") != FROZEN_PRODUCTION_BRANCH:
        raise DeploymentConfigError("frozen production branch differs from declared freeze")
    # Production immutability is the tracked/index contract enforced by the
    # canonical session verifier.  Local untracked analysis metadata is not in
    # the commit and cannot enter the fresh commit-addressed worktrees.
    if git("status", "--porcelain", "--untracked-files=no"):
        raise DeploymentConfigError("frozen production tracked/index state is dirty")


def _validate_root(path: Path, label: str) -> Path:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise DeploymentConfigError(f"{label}.parent must be an existing regular directory")
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise DeploymentConfigError(f"{label} must be an absent or regular directory")
    return path.resolve(strict=False)


def load_deployment_config(path: Path) -> DiscoveryDeployment:
    """Load one sealed JSON configuration without expanding its authority."""
    if path.is_symlink() or not path.is_file():
        raise DeploymentConfigError("deployment configuration must be a regular file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentConfigError("deployment configuration is not JSON") from exc
    top = _exact(raw, {"schema", "config_sha256", "production", "controller", "actors", "gpu",
                       "immutable_inputs", "planner_context", "source_plan"}, "deployment configuration")
    if top["schema"] != SCHEMA:
        raise DeploymentConfigError("deployment configuration schema mismatch")
    try:
        calculated_config_hash = schemas.content_hash(
            {key: value for key, value in top.items() if key != "config_sha256"})
    except (TypeError, ValueError) as exc:
        raise DeploymentConfigError("deployment configuration is not canonicalizable") from exc
    if (not isinstance(top["config_sha256"], str) or not SHA.fullmatch(top["config_sha256"])
            or calculated_config_hash != top["config_sha256"]):
        raise DeploymentConfigError("deployment configuration self-hash mismatch")
    production = _exact(top["production"], {"path", "head"}, "production")
    production_path = _absolute(production["path"], "production.path")
    if production_path.is_symlink() or not production_path.is_dir():
        raise DeploymentConfigError("production.path must be a regular directory")
    if not isinstance(production["head"], str) or not GIT_SHA.fullmatch(production["head"]):
        raise DeploymentConfigError("production.head must be an exact Git SHA")
    _verify_production(production_path, production["head"])
    controller = _exact(top["controller"], {"state_root", "evidence_root",
                                               "operations_root", "max_iterations",
                                               "nomination_threshold"}, "controller")
    roots = {key: _validate_root(_absolute(controller[key], f"controller.{key}"), f"controller.{key}") for key in
             ("state_root", "evidence_root", "operations_root")}
    if any(_overlaps(left, right) for index, left in enumerate(roots.values())
           for right in list(roots.values())[index + 1:]):
        raise DeploymentConfigError("controller roots must not overlap")
    if any(_overlaps(root, production_path) for root in roots.values()):
        raise DeploymentConfigError("controller output roots must not enter frozen production")
    max_iterations = controller["max_iterations"]
    threshold = controller["nomination_threshold"]
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 1000:
        raise DeploymentConfigError("controller.max_iterations is invalid")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold)) or threshold <= 0):
        raise DeploymentConfigError("controller.nomination_threshold is invalid")
    actors = _exact(top["actors"], {"wrapper_path", "wrapper_sha256", "environment_profile_id"}, "actors")
    actor_wrapper = _input({"path": actors["wrapper_path"], "sha256": actors["wrapper_sha256"]}, "actors.wrapper")
    if not os.access(actor_wrapper.path, os.X_OK):
        raise DeploymentConfigError("actors.wrapper_path must be executable")
    environment_profile_id = _identifier(actors["environment_profile_id"], "actors.environment_profile_id")
    gpu = _exact(top["gpu"], {"device_id", "claim_timeout_s", "inference_window_lock",
                                "inference_window_lease_id"}, "gpu")
    device_id = _identifier(gpu["device_id"], "gpu.device_id")
    if device_id not in ALLOWED_DEVICE_IDS:
        raise DeploymentConfigError("gpu.device_id is not an admitted discovery device")
    claim_timeout_s = gpu["claim_timeout_s"]
    if (isinstance(claim_timeout_s, bool) or not isinstance(claim_timeout_s, (int, float))
            or not math.isfinite(float(claim_timeout_s)) or claim_timeout_s < 0):
        raise DeploymentConfigError("gpu.claim_timeout_s is invalid")
    window = _absolute(gpu["inference_window_lock"], "gpu.inference_window_lock")
    if window.parent.is_symlink() or not window.parent.is_dir() or (window.exists() and (window.is_symlink() or not window.is_file())):
        raise DeploymentConfigError("gpu.inference_window_lock parent/file is invalid")
    if _overlaps(window, production_path):
        raise DeploymentConfigError("gpu.inference_window_lock must not enter frozen production")
    inputs = _exact(top["immutable_inputs"], {"model", "workload", "runtime_config", "admission_policy"}, "immutable_inputs")
    source = _exact(top["source_plan"], {"source_builder_id", "evidence_plan_id",
                                           "runner_args_id", "experiment_template_registry_id", "experiment_template_registry_sha256",
                                           "production_snapshot_id"}, "source_plan")
    model = _input(inputs["model"], "model")
    workload = _input(inputs["workload"], "workload")
    runtime_config = _input(inputs["runtime_config"], "runtime_config")
    admission_policy_input = _input(inputs["admission_policy"], "admission_policy")
    admission_policy = _admission_policy(admission_policy_input, model=model, workload=workload)
    planner_context = _planner_context(top["planner_context"], model=model,
                                       workload=workload, runtime_config=runtime_config)
    for label, input_ in (("actors.wrapper", actor_wrapper), ("model", model),
                          ("workload", workload), ("runtime_config", runtime_config),
                          ("admission_policy", admission_policy_input), ("planner_context", planner_context.input)):
        if any(_overlaps(input_.path, protected)
               for protected in (*roots.values(), production_path)):
            raise DeploymentConfigError(
                f"{label} location overlaps a mutable output or frozen production tree")
    return DiscoveryDeployment(
        config_sha256=top["config_sha256"], production_path=production_path, production_head=production["head"],
        state_root=roots["state_root"], evidence_root=roots["evidence_root"],
        operations_root=roots["operations_root"], max_iterations=max_iterations,
        nomination_threshold=float(threshold), actor_wrapper=actor_wrapper,
        environment_profile_id=environment_profile_id, device_id=device_id,
        claim_timeout_s=float(claim_timeout_s), inference_window_lock=window,
        model=model, workload=workload, admission_policy=admission_policy,
        runtime_config=runtime_config, policy=admission_policy_input, planner_context=planner_context,
        source_builder_id=_identifier(source["source_builder_id"], "source_plan.source_builder_id"),
        evidence_plan_id=_identifier(source["evidence_plan_id"], "source_plan.evidence_plan_id"),
        runner_args_id=_identifier(source["runner_args_id"], "source_plan.runner_args_id"),
        experiment_template_registry_id=_identifier(source["experiment_template_registry_id"], "source_plan.experiment_template_registry_id"),
        experiment_template_registry_sha256=_digest_identifier(source["experiment_template_registry_sha256"], "source_plan.experiment_template_registry_sha256"),
        inference_window_lease_id=_identifier(gpu["inference_window_lease_id"], "gpu.inference_window_lease_id"),
        production_snapshot_id=_identifier(source["production_snapshot_id"], "source_plan.production_snapshot_id"),
    )


def resolve_registry(config: DiscoveryDeployment, registry: Mapping[str, Mapping[str, object]]) -> ResolvedDeployment:
    """Resolve IDs from trusted in-process registries; never import arbitrary code."""
    required = {
        "environment_profile": config.environment_profile_id,
        "source_builder": config.source_builder_id,
        "evidence_plan": config.evidence_plan_id,
        "runner_args": config.runner_args_id,
        "experiment_template_registry": config.experiment_template_registry_id,
        "inference_window_lease": config.inference_window_lease_id,
        "production_snapshot": config.production_snapshot_id,
    }
    if set(registry) != set(required):
        raise DeploymentConfigError("registry categories must exactly match the deployment contract")
    config.revalidate()
    values: dict[str, object] = {}
    for kind, identifier in required.items():
        table = registry.get(kind)
        if not isinstance(table, Mapping) or identifier not in table:
            raise DeploymentConfigError(f"registry has no {kind}:{identifier}")
        values[kind] = table[identifier]
    # This module intentionally knows only identifiers and sealed input bytes.
    # Concrete type checks belong to the producer-aware materializer; putting
    # stale structural checks here made its typed registry unreachable.
    return ResolvedDeployment(config=config, **values)


__all__ = ["SCHEMA", "PLANNER_CONTEXT_SCHEMA", "DeploymentConfigError", "ImmutableInput", "PlannerContext", "DiscoveryDeployment",
           "ResolvedDeployment", "load_deployment_config", "resolve_registry"]
