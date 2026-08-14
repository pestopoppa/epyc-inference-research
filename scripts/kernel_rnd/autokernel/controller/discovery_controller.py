#!/usr/bin/env python3
"""Candidate-only AutoKernel discovery controller.

The controller deliberately owns only the state machine.  Existing campaign
code owns source mutation, isolated worktrees, build, resource claims, source
proof, dispatch attribution, screening, cleanup, and frozen-tree proof.  This
module never accepts a command from a planner and never turns a screen into a
promotion.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from .. import campaign, journal, source_candidate
from . import codex_container_actor, do_not_repeat, hypotheses
from . import gpu_source_proofs
from scripts.benchmark import autokernel_progression
from scripts.benchmark import run_autokernel_gpu_discovery as gpu_discovery

SCHEMA = "epyc.autokernel.discovery_controller.v2"
AUTHORITY = "nonpromotable_candidate_only_discovery"
HASH = __import__("re").compile(r"^[0-9a-f]{64}$")
SOL = {"provider": "codex", "model": "gpt-5.6-sol", "effort": "high", "role": "planner"}
TERRA = {"provider": "codex", "model": "gpt-5.6-terra", "effort": "high", "role": "critic"}


class DiscoveryControllerError(RuntimeError): pass


class PrecomputeScreenRefusal(DiscoveryControllerError):
    """Typed adapter refusal proving that no governed operation was started."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str: return hashlib.sha256(_canon(value)).hexdigest()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, indent=2) + "\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def sealed_roster() -> dict[str, Any]:
    return {"schema": "epyc.autokernel.discovery_roster.v2", "members": [SOL, TERRA], "claude_members": 0, "member_count": 2}


def _require_roster(value: Mapping[str, Any]) -> None:
    if dict(value) != sealed_roster(): raise DiscoveryControllerError("runtime roster is not exact Sol planner + Terra critic, 0 Claude")

def _require_runtime(value: Mapping[str, Any]) -> None:
    required={"kind","docker_path","docker_sha256","image_id","codex_native_sha256","code_mode_host_sha256","ca_certificate_sha256","writable_host_binds","host_network_mode"}
    if set(value) != required or value.get("kind")!="docker_workspace_bind_only" or value.get("host_network_mode")!="docker_bridge" or value.get("writable_host_binds") != ["/workspace"] or not all(isinstance(value.get(k),str) and value[k] for k in required-{"writable_host_binds"}): raise DiscoveryControllerError("Codex runtime attestation is incomplete or unsealed")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value: raise DiscoveryControllerError(f"{label} must be non-empty text")
    return value.strip()


@dataclass(frozen=True)
class AuthoringAssignment:
    """Controller-owned identity tuple; the actor may fill content, not authority."""
    campaign_id: str
    proposal_id: str
    candidate_id: str
    production_base_commit: str
    instrument_commit: str

    def __post_init__(self) -> None:
        if (not self.campaign_id.startswith("ak-") or not self.proposal_id.startswith("akp-")
                or not self.candidate_id.startswith("akc-")
                or not all(isinstance(value, str) and len(value) == 40
                           and all(ch in "0123456789abcdef" for ch in value)
                           for value in (self.production_base_commit, self.instrument_commit))):
            raise DiscoveryControllerError("invalid controller-owned authoring identity")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDispatchExpectation:
    """Planner-authored literal geometry; never a regex, argv, or command."""
    kernel_name: str
    calls: int
    grid: int
    workgroup: int
    lds_bytes: int

    def __post_init__(self) -> None:
        import re
        if (not isinstance(self.kernel_name, str) or not re.fullmatch(r"[A-Za-z0-9_:<>.,-]{1,256}", self.kernel_name)
                or any(ch in self.kernel_name for ch in "*?[]()|+\\^$")):
            raise DiscoveryControllerError("dispatch kernel name must be a bounded literal")
        for label, value, maximum in (("calls", self.calls, 10_000_000), ("grid", self.grid, 1 << 31),
                                      ("workgroup", self.workgroup, 4096), ("lds_bytes", self.lds_bytes, 1 << 30)):
            if (isinstance(value, bool) or not isinstance(value, int) or value < 0
                    or value > maximum or (label != "lds_bytes" and value == 0)):
                raise DiscoveryControllerError(f"dispatch {label} is outside reviewed literal bounds")


@dataclass(frozen=True)
class LoadModeRecommendation:
    mode: str
    rationale: str
    example_ids: tuple[str, ...]
    def __post_init__(self) -> None:
        identifier = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
        if (self.mode not in {"cold_overlap", "cold_serialized", "hot_resident"}
                or not isinstance(self.rationale, str) or not self.rationale.strip()
                or len(self.rationale) > 1024 or not isinstance(self.example_ids, tuple)
                or not self.example_ids or len(self.example_ids) > 8
                or len(set(self.example_ids)) != len(self.example_ids)
                or any(not isinstance(item, str) or not identifier.fullmatch(item)
                       for item in self.example_ids)):
            raise DiscoveryControllerError("load-mode recommendation is malformed or unbounded")
        object.__setattr__(self, "rationale", self.rationale.strip())


@dataclass(frozen=True)
class GpuSourceExperimentIntent:
    """Actor-selected *registry IDs*, never actor-selected commands or regexes."""
    template_id: str
    target_surface: str
    target_symbol: str
    correctness_id: str
    dispatch_id: str
    expected_dispatch: BoundedDispatchExpectation
    load_mode_recommendation: LoadModeRecommendation | None = None

    def __post_init__(self) -> None:
        import re
        identifier = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
        for label, value in (("template_id", self.template_id),
                             ("correctness_id", self.correctness_id),
                             ("dispatch_id", self.dispatch_id)):
            if not isinstance(value, str) or not identifier.fullmatch(value):
                raise DiscoveryControllerError(f"experiment intent {label} is not a registry id")
        for label, value in (("target_surface", self.target_surface),
                             ("target_symbol", self.target_symbol)):
            _text(value, f"experiment intent {label}")
        if not isinstance(self.expected_dispatch, BoundedDispatchExpectation):
            raise DiscoveryControllerError("experiment intent requires bounded literal dispatch expectation")
        if self.load_mode_recommendation is not None and not isinstance(
                self.load_mode_recommendation, LoadModeRecommendation):
            raise DiscoveryControllerError("load-mode recommendation must be typed and immutable")


@dataclass(frozen=True)
class PlannedCandidate:
    hypothesis_id: str
    statement: str
    falsifier: str
    regime: Mapping[str, Any]
    proposal: Mapping[str, Any]
    source_manifest: source_candidate.SourcePatchManifest
    source_manifest_sha256: str
    experiment_intent: GpuSourceExperimentIntent | None = None

    def __post_init__(self) -> None:
        _text(self.hypothesis_id, "hypothesis_id"); _text(self.statement, "statement"); _text(self.falsifier, "falsifier")
        if not self.hypothesis_id.startswith("akh-"): raise DiscoveryControllerError("hypothesis_id must start akh-")
        if not isinstance(self.regime, Mapping) or not isinstance(self.proposal, Mapping): raise DiscoveryControllerError("candidate regime and proposal must be mappings")
        if not isinstance(self.source_manifest, source_candidate.SourcePatchManifest): raise DiscoveryControllerError("candidate requires typed SourcePatchManifest")
        if not HASH.fullmatch(self.source_manifest_sha256): raise DiscoveryControllerError("source manifest hash is required")
        # Planner-owned effect fields are structurally impossible.
        if any("effect" in str(key).lower() or "result" in str(key).lower() for key in self.proposal):
            raise DiscoveryControllerError("planner proposal may not carry measured result fields")


@dataclass(frozen=True)
class Critique:
    decision: str
    reason: str
    def __post_init__(self) -> None:
        if self.decision not in {"accept", "reject", "revise"}: raise DiscoveryControllerError("critic decision must be accept, reject, or revise")
        _text(self.reason, "critic reason")


@dataclass(frozen=True)
class SealedScreen:
    receipt_path: str
    result_sha256: str
    effect_fraction: float
    classification: str
    baseline_sha256: str
    source_proof_sha256: str
    dispatch_proof_sha256: str
    candidate_only: bool = True
    promotion_claim: bool = False
    stages: tuple[str, ...] = ("materialized", "built", "correctness", "attribution", "screen")
    # A series is one exact patch measured in one exact frame/baseline.  It is
    # deliberately not a hypothesis id: one scientific question can produce
    # several mutually independent source patches.
    series_key: str | None = None
    component_series_keys: tuple[str, ...] = ()
    # Pooled only by the controller after exact-series verification.  Adapter
    # receipts report their individual measured effect; they cannot nominate.
    series_effect_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.classification not in {"candidate", "screened_out", "inconclusive", "failed", "top_k_replicated_candidate", "replicated_but_subadditive"}: raise DiscoveryControllerError("unknown screen class")
        if (isinstance(self.effect_fraction, bool)
                or not isinstance(self.effect_fraction, (int, float))
                or not math.isfinite(float(self.effect_fraction))):
            raise DiscoveryControllerError("screen effect must be a finite measured number")
        if not self.candidate_only or self.promotion_claim: raise DiscoveryControllerError("discovery screen must remain nonpromotable")
        if tuple(self.stages) != ("materialized", "built", "correctness", "attribution", "screen"):
            raise DiscoveryControllerError("screen did not prove the required fail-closed stage order")
        for value in (self.result_sha256, self.baseline_sha256, self.source_proof_sha256, self.dispatch_proof_sha256):
            if not HASH.fullmatch(value): raise DiscoveryControllerError("sealed result requires evidence hashes")
        if self.series_key is not None and not HASH.fullmatch(self.series_key):
            raise DiscoveryControllerError("screen series key must be a sealed hash")
        # JSON recovery naturally turns a tuple into a list; normalize it at
        # the durable boundary, then keep the in-memory receipt immutable.
        if isinstance(self.component_series_keys, list):
            object.__setattr__(self, "component_series_keys", tuple(self.component_series_keys))
        if not isinstance(self.component_series_keys, tuple) or not all(HASH.fullmatch(value) for value in self.component_series_keys):
            raise DiscoveryControllerError("component series provenance must be sealed hashes")
        if (self.series_effect_fraction is not None
                and (isinstance(self.series_effect_fraction, bool)
                     or not isinstance(self.series_effect_fraction, (int, float))
                     or not math.isfinite(float(self.series_effect_fraction)))):
            raise DiscoveryControllerError("pooled series effect must be finite")


class Planner(Protocol):
    def attest(self) -> Mapping[str, Any]: ...
    def plan(self, *, context: Mapping[str, Any], workspace: Path) -> PlannedCandidate: ...

class Critic(Protocol):
    def attest(self) -> Mapping[str, Any]: ...
    def review(self, candidate: PlannedCandidate, *, context: Mapping[str, Any], workspace: Path) -> Critique: ...

class Lease(Protocol):
    def admit(self, candidate: PlannedCandidate) -> Mapping[str, Any]: ...

class Screener(Protocol):
    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen: ...
    def reconcile(self, inflight: Mapping[str, Any]) -> "Recovery": ...

@dataclass(frozen=True)
class Recovery:
    status: str
    result: SealedScreen | None = None
    def __post_init__(self) -> None:
        if self.status not in {"safe_to_start", "sealed_result", "ambiguous"}: raise DiscoveryControllerError("unknown recovery status")
        if (self.status == "sealed_result") != isinstance(self.result, SealedScreen): raise DiscoveryControllerError("recovery result binding is invalid")


class CodexPlanner:
    """Concrete Sol actor. It may write only a plan and patch manifest in workspace."""
    def __init__(self, *, wrapper: Path, environment: Mapping[str, str],
                 template_catalog: Mapping[str, Any] | None = None) -> None:
        self.wrapper, self.environment = wrapper, dict(environment)
        self.template_catalog = json.loads(json.dumps(template_catalog or {}, sort_keys=True))
    def attest(self) -> Mapping[str, Any]: return {**SOL, "runtime": codex_container_actor.runtime_identity(self.wrapper)}
    def plan(self, *, context: Mapping[str, Any], workspace: Path) -> PlannedCandidate:
        # The model gets a bounded source/profile brief plus a machine contract;
        # it never receives authority to select a campaign, base, executable,
        # argv, profile parser, or evidence regex.
        contract = {
            "plan_json_keys": ["hypothesis_id", "statement", "falsifier", "regime",
                               "proposal", "source_manifest_path", "experiment_intent"],
            "experiment_intent_keys": ["template_id", "target_surface", "target_symbol",
                                       "correctness_id", "dispatch_id", "expected_dispatch",
                                       "load_mode_recommendation"],
            "expected_dispatch_keys": ["kernel_name", "calls", "grid", "workgroup", "lds_bytes"],
            "source_manifest": "epyc.autokernel.source_patch.v1; use deployment-assigned ids/base/instrument only",
            "proposal_requirements": ["proposal_id matches manifest", "change_class matches manifest",
                                       "change.files_and_symbols exactly matches manifest declarations",
                                       "change.estimated_diff_size is positive"],
            "forbidden": ["commands", "argv", "environment", "measurement results",
                          "campaign/base/instrument selection", "unbounded source reads"],
        }
        assignment = context.get("authoring_assignment")
        if not isinstance(assignment, Mapping):
            raise DiscoveryControllerError("planner context lacks controller-owned authoring assignment")
        prompt = json.dumps({"role": SOL, "context": context,
                             "experiment_template_catalog": self.template_catalog,
                             "authoring_contract": contract,
                             "output": "Write plan.json and source-patch.json in workspace."}, sort_keys=True)
        result = codex_container_actor.run_actor(wrapper=self.wrapper, workspace=workspace, model=SOL["model"], effort=SOL["effort"], prompt=prompt, environment=self.environment)
        if result.returncode: raise DiscoveryControllerError(f"Sol actor failed: {result.stderr[-400:]}")
        return _load_plan(workspace / "plan.json", workspace,
                          assignment=AuthoringAssignment(**assignment))


class CodexCritic:
    """Concrete Terra actor. It can bind a veto but never alters the candidate."""
    def __init__(self, *, wrapper: Path, environment: Mapping[str, str],
                 template_catalog: Mapping[str, Any] | None = None) -> None:
        self.wrapper, self.environment = wrapper, dict(environment)
        self.template_catalog = json.loads(json.dumps(template_catalog or {}, sort_keys=True))
    def attest(self) -> Mapping[str, Any]: return {**TERRA, "runtime": codex_container_actor.runtime_identity(self.wrapper)}
    def review(self, candidate: PlannedCandidate, *, context: Mapping[str, Any], workspace: Path) -> Critique:
        manifest = candidate.source_manifest
        if len(manifest.patch_text) > 65536:
            raise DiscoveryControllerError("candidate patch exceeds bounded critic visibility")
        prompt = json.dumps({"role": TERRA, "context": context,
            "experiment_template_catalog": self.template_catalog, "candidate": {
            "hypothesis_id": candidate.hypothesis_id, "statement": candidate.statement,
            "falsifier": candidate.falsifier, "proposal": candidate.proposal,
            "experiment_intent": asdict(candidate.experiment_intent) if candidate.experiment_intent else None,
            "source_manifest_sha256": candidate.source_manifest_sha256,
            "manifest": {"campaign_id": manifest.campaign_id, "proposal_id": manifest.proposal_id,
                         "candidate_id": manifest.candidate_id,
                         "production_base_commit": manifest.production_base_commit,
                         "instrument_commit": manifest.instrument_commit,
                         "declared_files": list(manifest.declared_files),
                         "declared_symbols": {key: list(value) for key, value in manifest.declared_symbols.items()},
                         "patch_sha256": manifest.patch_sha256, "patch_text": manifest.patch_text}},
            "output": "Write critique.json with exactly decision=accept|reject|revise and reason."}, sort_keys=True)
        result = codex_container_actor.run_actor(wrapper=self.wrapper, workspace=workspace, model=TERRA["model"], effort=TERRA["effort"], prompt=prompt, environment=self.environment)
        if result.returncode: raise DiscoveryControllerError(f"Terra actor failed: {result.stderr[-400:]}")
        value = _read_object(workspace / "critique.json", workspace)
        if set(value) != {"decision", "reason"}: raise DiscoveryControllerError("critic output schema mismatch")
        return Critique(**value)


def _read_object(path: Path, root: Path) -> dict[str, Any]:
    try: path.resolve().relative_to(root.resolve())
    except ValueError as exc: raise DiscoveryControllerError("actor artifact escaped workspace") from exc
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise DiscoveryControllerError(f"invalid actor artifact {path.name}") from exc
    if not isinstance(value, dict): raise DiscoveryControllerError("actor artifact must be object")
    return value


def _load_plan(path: Path, root: Path, *, assignment: AuthoringAssignment | None = None) -> PlannedCandidate:
    value = _read_object(path, root)
    allowed = {"hypothesis_id", "statement", "falsifier", "regime", "proposal", "source_manifest_path", "experiment_intent"}
    if set(value) not in (allowed, allowed - {"experiment_intent"}): raise DiscoveryControllerError("planner output schema mismatch")
    intent_raw = value.pop("experiment_intent", None)
    if intent_raw is not None:
        allowed_intent = {"template_id", "target_surface", "target_symbol", "correctness_id", "dispatch_id", "expected_dispatch", "load_mode_recommendation"}
        if not isinstance(intent_raw, Mapping) or set(intent_raw) not in (allowed_intent, allowed_intent - {"load_mode_recommendation"}):
            raise DiscoveryControllerError("planner experiment intent schema mismatch")
        expected = intent_raw["expected_dispatch"]
        if not isinstance(expected, Mapping) or set(expected) != {"kernel_name", "calls", "grid", "workgroup", "lds_bytes"}:
            raise DiscoveryControllerError("planner bounded dispatch schema mismatch")
        recommendation = intent_raw.get("load_mode_recommendation")
        if recommendation is not None:
            if not isinstance(recommendation, Mapping) or set(recommendation) != {"mode", "rationale", "example_ids"}:
                raise DiscoveryControllerError("planner load-mode recommendation schema mismatch")
            recommendation = LoadModeRecommendation(
                mode=recommendation["mode"], rationale=recommendation["rationale"],
                example_ids=tuple(recommendation["example_ids"]))
        intent = GpuSourceExperimentIntent(**{**intent_raw,
            "expected_dispatch": BoundedDispatchExpectation(**expected),
            "load_mode_recommendation": recommendation})
    else:
        intent = None
    raw_path = Path(_text(value.pop("source_manifest_path"), "source_manifest_path"))
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise DiscoveryControllerError("source manifest path must be a workspace-relative path")
    manifest_path = root / raw_path
    try:
        manifest_path.resolve(strict=True).relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise DiscoveryControllerError("source manifest escaped disposable workspace") from exc
    manifest = source_candidate.load_source_patch_manifest(manifest_path)
    if assignment is not None:
        if (manifest.campaign_id, manifest.proposal_id, manifest.candidate_id,
                manifest.production_base_commit, manifest.instrument_commit) != (
                    assignment.campaign_id, assignment.proposal_id, assignment.candidate_id,
                    assignment.production_base_commit, assignment.instrument_commit):
            raise DiscoveryControllerError("actor attempted to invent campaign/base/instrument identity")
        if value.get("proposal", {}).get("proposal_id") != assignment.proposal_id:
            raise DiscoveryControllerError("actor proposal does not use controller-assigned proposal identity")
    return PlannedCandidate(**value, source_manifest=manifest, source_manifest_sha256=manifest.patch_bundle_sha256,
                            experiment_intent=intent)


class CampaignScreener:
    """Concrete adapter: call the existing candidate-only campaign transaction."""
    def __init__(self, *, spec_factory: Callable[[PlannedCandidate, hypotheses.ClaimAuthorization], campaign.CampaignSpec], ops_factory: Callable[[], Any]) -> None:
        self.spec_factory, self.ops_factory = spec_factory, ops_factory
    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen:
        spec = self.spec_factory(candidate, authorization)
        if not spec.screening_only or spec.source_patch is not candidate.source_manifest or spec.authorization != authorization:
            raise DiscoveryControllerError("campaign adapter must bind typed patch, authorization, and candidate-only screen")
        if spec.source_prerequisite_package is None and spec.fresh_source_prerequisite_plan is None:
            raise DiscoveryControllerError("source candidate requires source correctness and dispatch prerequisite package")
        result = campaign.run_campaign(spec, self.ops_factory())
        return _screen_from_campaign(result)


def _screen_from_campaign(result: campaign.CampaignResult) -> SealedScreen:
    raw = result.to_dict(); report = raw.get("screening_report")
    if not (result.ok and raw.get("state") == "decided" and raw.get("screening_only") is True and isinstance(report, Mapping)):
        raise DiscoveryControllerError("campaign did not produce a sealed candidate-only result")
    required = ("baseline_sha256", "source_prerequisite_package_sha256", "dispatch_attribution_sha256", "result_sha256")
    if not all(isinstance(report.get(key), str) and HASH.fullmatch(report[key]) for key in required):
        raise DiscoveryControllerError("campaign result lacks source proof, exact dispatch proof, baseline, or result hash")
    return SealedScreen(receipt_path=str(report.get("receipt_path", "")), result_sha256=report["result_sha256"], effect_fraction=float(report["median_relative"]), classification=str(report.get("classification", "candidate")), baseline_sha256=report["baseline_sha256"], source_proof_sha256=report["source_prerequisite_package_sha256"], dispatch_proof_sha256=report["dispatch_attribution_sha256"])


@dataclass(frozen=True)
class GpuSourceBuild:
    """A completed isolated build, returned only by a typed source-build seam."""
    anchor_build: Path
    candidate_build: Path
    candidate_identity: gpu_source_proofs.BuildIdentity
    anchor_identity: gpu_source_proofs.BuildIdentity
    measurement_binary: Path | None = None
    common_loader_dir: Path | None = None
    anchor_loader_dir: Path | None = None
    candidate_loader_dir: Path | None = None
    reward_runtime_sha256: str | None = None
    operation_key: str | None = None
    build_key: str | None = None
    materialization_receipt: Path | None = None
    materialization_sha256: str | None = None
    anchor_source_tree_receipt: Path | None = None
    anchor_source_tree_sha256: str | None = None
    candidate_source_tree_receipt: Path | None = None
    candidate_source_tree_sha256: str | None = None
    teardown_receipt: Path | None = None
    teardown_sha256: str | None = None
    def __post_init__(self) -> None:
        for path in (self.anchor_build, self.candidate_build):
            if not path.is_absolute() or not path.is_dir():
                raise DiscoveryControllerError("GPU source build paths must be existing absolute directories")
        if self.candidate_identity == self.anchor_identity:
            raise DiscoveryControllerError("source screen requires distinct sealed anchor and candidate build identities")
        runtime = (self.measurement_binary, self.common_loader_dir, self.anchor_loader_dir,
                   self.candidate_loader_dir, self.reward_runtime_sha256)
        if any(value is not None for value in runtime):
            if (not all(value is not None for value in runtime)
                    or not isinstance(self.measurement_binary, Path) or not self.measurement_binary.is_file()
                    or not isinstance(self.common_loader_dir, Path) or not self.common_loader_dir.is_dir()
                    or not isinstance(self.anchor_loader_dir, Path) or not self.anchor_loader_dir.is_dir()
                    or not isinstance(self.candidate_loader_dir, Path) or not self.candidate_loader_dir.is_dir()
                    or not isinstance(self.reward_runtime_sha256, str) or not HASH.fullmatch(self.reward_runtime_sha256)):
                raise DiscoveryControllerError("GPU source build has an incomplete shared reward closure")
        if self.operation_key is not None and (not isinstance(self.operation_key, str) or not HASH.fullmatch(self.operation_key)):
            raise DiscoveryControllerError("GPU source build operation key is invalid")
        if self.build_key is not None and (not isinstance(self.build_key, str) or not HASH.fullmatch(self.build_key)):
            raise DiscoveryControllerError("GPU source build cache key is invalid")
        for path, expected, label in ((self.materialization_receipt, self.materialization_sha256, "materialization"),
                                      (self.anchor_source_tree_receipt, self.anchor_source_tree_sha256, "anchor source tree"),
                                      (self.candidate_source_tree_receipt, self.candidate_source_tree_sha256, "candidate source tree"),
                                      (self.teardown_receipt, self.teardown_sha256, "teardown")):
            if (path is None) != (expected is None):
                raise DiscoveryControllerError(f"GPU source build has incomplete {label} receipt")
            if path is not None:
                if (not isinstance(path, Path) or not path.is_absolute() or path.is_symlink()
                        or not path.is_file() or not isinstance(expected, str) or not HASH.fullmatch(expected)):
                    raise DiscoveryControllerError(f"GPU source build has invalid {label} receipt")
                assert isinstance(path, Path) and isinstance(expected, str)
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise DiscoveryControllerError(f"GPU source {label} receipt bytes changed")


@dataclass(frozen=True)
class ProofReceipt:
    """Hash-bound source or dispatch proof produced before any screen call."""
    path: Path
    sha256: str
    kind: str
    def __post_init__(self) -> None:
        if self.kind not in {"source", "dispatch"} or not self.path.is_absolute() or not self.path.is_file() or not HASH.fullmatch(self.sha256):
            raise DiscoveryControllerError("proof receipt must be an existing typed source/dispatch artifact")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise DiscoveryControllerError("proof receipt bytes differ from its sealed hash")


class GpuSourceScreener:
    """GPU source lane using the existing governed discovery runner.

    This intentionally does not reuse the CPU baseline bank: that bank proves an
    unchanged binary with a parameter delta.  GPU source runs need distinct
    build identities and their own sealed paired receipt.
    """
    def __init__(self, *, build_source: Callable[[PlannedCandidate, hypotheses.ClaimAuthorization, Mapping[str, Any]], GpuSourceBuild],
                 proof_bundle: Callable[[PlannedCandidate, GpuSourceBuild], gpu_source_proofs.GpuSourceProofBundle],
                 args_factory: Callable[[PlannedCandidate, GpuSourceBuild, Mapping[str, Any]], Any]) -> None:
        self.build_source, self.proof_bundle, self.args_factory = build_source, proof_bundle, args_factory

    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen:
        build = self.build_source(candidate, authorization, lease)
        bundle = self.proof_bundle(candidate, build)
        if not isinstance(bundle, gpu_source_proofs.GpuSourceProofBundle):
            raise DiscoveryControllerError("GPU source gate did not return a validated proof bundle")
        if bundle.manifest_sha256 != candidate.source_manifest_sha256:
            raise DiscoveryControllerError("GPU proof bundle does not bind the candidate manifest")
        if bundle.candidate != build.candidate_identity or bundle.anchor != build.anchor_identity:
            raise DiscoveryControllerError("GPU proof bundle does not bind both sealed build identities")
        args = self.args_factory(candidate, build, lease)
        # The established runner owns KFD/VRAM, device claims, paired samples,
        # and its durable result.  This controller does not spawn a shell.
        if getattr(args, "factor", None) != "source_patch" or Path(getattr(args, "anchor_build", "")).resolve() != build.anchor_build or Path(getattr(args, "candidate_build", "")).resolve() != build.candidate_build:
            raise DiscoveryControllerError("GPU source runner arguments are not bound to the typed build")
        raw = gpu_discovery.run(args)
        result_path = Path(args.output_dir).resolve() / "result.json"
        durable = gpu_source_proofs.require_result_file(result_path, raw)["body"]
        raw = durable
        if not (raw.get("schema") == "epyc.autokernel.gpu_candidate_only_screen.v2" and raw.get("non_promotable") is True and raw.get("promotion_claim") is False and raw.get("hip_residency_proved") is True):
            raise DiscoveryControllerError("GPU runner returned an unsealed or non-resident discovery result")
        projection = autokernel_progression._gpu_screen(result_path, raw)
        if projection is None: raise DiscoveryControllerError("GPU result failed canonical progression validation")
        return SealedScreen(receipt_path=str(result_path), result_sha256=str(raw["result_sha256"]), effect_fraction=float(raw["median_relative"]), classification=str(projection["stage"]), baseline_sha256=str(raw["baseline_sha256"]), source_proof_sha256=bundle.correctness["file_sha256"], dispatch_proof_sha256=bundle.attribution["file_sha256"])


@dataclass(frozen=True)
class ControllerConfig:
    output_root: Path
    max_iterations: int = 1
    nomination_threshold: float = 0.03
    dry_run: bool = False
    # This is the AutoKernel evidence root that owns the canonical progression
    # projection.  The controller state root is never silently treated as a
    # second evidence tree in live mode.
    evidence_root: Path | None = None
    # Sealed deployment data, never planner-authored prose.  Keeping the hash
    # separately makes a changed profile/source brief a durable-resume refusal.
    planner_context: Mapping[str, Any] | None = None
    planner_context_sha256: str | None = None
    production_base_commit: str | None = None
    instrument_commit: str | None = None
    campaign_id: str = "ak-discovery"
    experiment_template_registry_sha256: str | None = None
    admission_corpus_sha256: str | None = None
    admission_corpus_version: str | None = None
    # The sealed deployment file is the authority for repository paths/refs as
    # well as the hashes separately carried below.  Durable state records this
    # one canonical identity so a resume cannot silently switch checkout roots.
    deployment_identity_sha256: str | None = None
    def __post_init__(self) -> None:
        if (not self.output_root.is_absolute() or not 1 <= self.max_iterations <= 1000
                or isinstance(self.nomination_threshold, bool)
                or not math.isfinite(float(self.nomination_threshold))
                or self.nomination_threshold <= 0
                or self.evidence_root is not None and not self.evidence_root.is_absolute()
                or (self.planner_context is None) != (self.planner_context_sha256 is None)
                or self.planner_context_sha256 is not None and not HASH.fullmatch(self.planner_context_sha256)
                or (self.production_base_commit is None) != (self.instrument_commit is None)
                or self.production_base_commit is not None and not all(
                    isinstance(value, str) and len(value) == 40
                    and all(ch in "0123456789abcdef" for ch in value)
                    for value in (self.production_base_commit, self.instrument_commit))
                or not self.campaign_id.startswith("ak-")
                or self.experiment_template_registry_sha256 is not None and not HASH.fullmatch(self.experiment_template_registry_sha256)
                or self.admission_corpus_sha256 is not None and not HASH.fullmatch(self.admission_corpus_sha256)
                or self.admission_corpus_version is not None and not re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", self.admission_corpus_version)):
            raise DiscoveryControllerError("invalid controller config")
        if self.deployment_identity_sha256 is not None and not HASH.fullmatch(self.deployment_identity_sha256):
            raise DiscoveryControllerError("invalid sealed deployment identity")
        sealed = (self.planner_context_sha256, self.production_base_commit,
                  self.instrument_commit, self.experiment_template_registry_sha256,
                  self.admission_corpus_sha256, self.admission_corpus_version,
                  self.deployment_identity_sha256)
        if (not self.dry_run and any(value is not None for value in sealed)
                and not all(value is not None for value in sealed)):
            raise DiscoveryControllerError("live sealed controller configuration has incomplete deployment authority")


class DurableState:
    def __init__(self, root: Path) -> None:
        self.root=root; self.book=journal.Journal(str(root / "journal")); self.book.initialize(); self.path=root / "state.json"
    def load(self) -> dict[str, Any]:
        if not self.path.exists(): return {"schema": SCHEMA, "authority": AUTHORITY, "roster": sealed_roster(), "iterations": [], "next": 1, "complete": False}
        value=_read_object(self.path, self.root); _require_roster(value.get("roster", {}))
        if value.get("schema") != SCHEMA or value.get("authority") != AUTHORITY: raise DiscoveryControllerError("wrong controller journal")
        declared=value.get("state_sha256")
        if not isinstance(declared,str) or declared != _sha({k:v for k,v in value.items() if k!="state_sha256"}): raise DiscoveryControllerError("durable controller state hash mismatch")
        return value
    def save(self, state: dict[str, Any], phase: str) -> None:
        state["updated_at"]=_now(); state["state_sha256"]=_sha({k:v for k,v in state.items() if k!="state_sha256"}); _atomic(self.path,state)
        self.book.append(journal.KIND_STOP_STATE,{"state":f"discovery_{phase}","controller_state_sha256":state["state_sha256"]})
    def run_lock(self):
        self.root.mkdir(parents=True,exist_ok=True)
        handle=(self.root / "controller.run.lock").open("a+")
        try:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close(); raise DiscoveryControllerError("another discovery controller owns this output root") from exc
        return handle


def _tracker(store: DurableState) -> hypotheses.HypothesisTracker:
    return hypotheses.HypothesisTracker(journal_=store.book, root=str(store.root / "hypotheses"), campaign_id="ak-discovery")


def _memory_block(tracker: hypotheses.HypothesisTracker, turn: int) -> Mapping[str, Any]:
    ledger=do_not_repeat.compile_for_tracker(tracker); return do_not_repeat.planner_round_block(tracker, ledger, round_id=f"discovery-{turn}")


def _ensure_question(tracker: hypotheses.HypothesisTracker, item: PlannedCandidate) -> None:
    question=hypotheses.Hypothesis(hypothesis_id=item.hypothesis_id, statement=item.statement, falsifier=item.falsifier, origin=hypotheses.ORIGIN_PLANNER, author="gpt-5.6-sol", regime=item.regime, source={"manifest_sha256":item.source_manifest_sha256})
    try: tracker.open_hypothesis(question)
    except hypotheses.HypothesisAlreadyTracked: pass

def _record_attempt_once(tracker: hypotheses.HypothesisTracker, item: PlannedCandidate, proposal_id: str, result: SealedScreen) -> None:
    ref=f"sha256:{result.result_sha256}"
    for event in tracker.read().events:
        attempt=event.payload.get("attempt") if event.kind==hypotheses.EVENT_ATTEMPTED else None
        if isinstance(attempt,Mapping) and attempt.get("hypothesis_id")==item.hypothesis_id and ref in attempt.get("refs",[]): return
    tracker.note_attempt(item.hypothesis_id,proposal_id=proposal_id,disposition=result.classification,bears_on_falsifier=True,note=f"sealed screen {result.result_sha256}; effect={result.effect_fraction:.9g}",refs=(ref,))


def _context(state: Mapping[str, Any], tracker: hypotheses.HypothesisTracker, turn: int,
             config: ControllerConfig) -> dict[str, Any]:
    prior = []
    for row in state["iterations"]:
        if not isinstance(row.get("result_sha256"), str):
            continue
        prior.append({key: row.get(key) for key in (
            "result_sha256", "status", "effect_fraction", "series_effect_fraction",
            "source_manifest_sha256", "series_key", "evidence", "statement",
            "falsifier", "experiment_intent", "mechanism_id", "target_surface",
            "target_symbol")})
    assignment = None
    if config.production_base_commit is not None:
        assignment = AuthoringAssignment(
            campaign_id=config.campaign_id, proposal_id=f"akp-discovery-{turn}",
            candidate_id=f"akc-discovery-{turn}",
            production_base_commit=config.production_base_commit,
            instrument_commit=config.instrument_commit or config.production_base_commit).to_dict()
    return {"authority": AUTHORITY, "turn":turn, "roster":sealed_roster(),
            "planner_context": config.planner_context,
            "planner_context_sha256": config.planner_context_sha256,
            "admission_corpus_sha256": config.admission_corpus_sha256,
            "admission_corpus_version": config.admission_corpus_version,
            "deployment_identity_sha256": config.deployment_identity_sha256,
            "authoring_assignment": assignment,
            "prior_results": prior, "do_not_repeat":_memory_block(tracker,turn)}


def _pending_item(item: PlannedCandidate) -> dict[str, Any]:
    manifest = item.source_manifest
    raw_manifest=json.dumps({"schema":source_candidate.SCHEMA_SOURCE_PATCH,"campaign_id":manifest.campaign_id,"proposal_id":manifest.proposal_id,"candidate_id":manifest.candidate_id,"source_tree":manifest.source_tree,"production_base_commit":manifest.production_base_commit,"instrument_commit":manifest.instrument_commit,"change_class":manifest.change_class,"declared_files":list(manifest.declared_files),"declared_symbols":{k:list(v) for k,v in manifest.declared_symbols.items()},"mechanism_id":manifest.mechanism_id,"patch_sha256":manifest.patch_sha256,"patch_encoding":"base64","patch_base64":base64.b64encode(manifest.patch_bytes).decode("ascii")},sort_keys=True,separators=(",",":")).encode()
    return {"hypothesis_id": item.hypothesis_id, "statement": item.statement,
            "falsifier": item.falsifier, "regime": dict(item.regime),
            "proposal": dict(item.proposal), "source_manifest_sha256": item.source_manifest_sha256,
            "experiment_intent": asdict(item.experiment_intent) if item.experiment_intent else None,
            "manifest": {"campaign_id":manifest.campaign_id,"proposal_id":manifest.proposal_id,
                "candidate_id":manifest.candidate_id,"source_tree":manifest.source_tree,
                "production_base_commit":manifest.production_base_commit,"instrument_commit":manifest.instrument_commit,
                "change_class":manifest.change_class,"declared_files":list(manifest.declared_files),
                "declared_symbols":{k:list(v) for k,v in manifest.declared_symbols.items()},
                "mechanism_id":manifest.mechanism_id,"patch_sha256":manifest.patch_sha256,
                "patch_base64":base64.b64encode(manifest.patch_bytes).decode("ascii")},
            "manifest_raw_base64":base64.b64encode(raw_manifest).decode("ascii"),"manifest_file_sha256":hashlib.sha256(raw_manifest).hexdigest(),"patch_bundle_sha256":manifest.patch_bundle_sha256}


def _restore_pending(value: Mapping[str, Any]) -> PlannedCandidate:
    raw=value.get("candidate")
    if not isinstance(raw,Mapping) or not isinstance(raw.get("manifest"),Mapping): raise DiscoveryControllerError("pending candidate is missing sealed manifest")
    m=raw["manifest"]
    try:
        manifest=source_candidate.SourcePatchManifest(campaign_id=m["campaign_id"],proposal_id=m["proposal_id"],candidate_id=m["candidate_id"],source_tree=m["source_tree"],production_base_commit=m["production_base_commit"],instrument_commit=m["instrument_commit"],change_class=m["change_class"],declared_files=tuple(m["declared_files"]),declared_symbols={k:tuple(v) for k,v in m["declared_symbols"].items()},mechanism_id=m["mechanism_id"],patch_sha256=m["patch_sha256"],patch_bytes=base64.b64decode(m["patch_base64"],validate=True))
    except (KeyError,TypeError,ValueError,source_candidate.SourceCandidateError) as exc: raise DiscoveryControllerError("pending candidate manifest is invalid") from exc
    raw_bytes=base64.b64decode(raw.get("manifest_raw_base64",""),validate=True)
    if hashlib.sha256(raw_bytes).hexdigest()!=raw.get("manifest_file_sha256") or manifest.patch_bundle_sha256!=raw.get("patch_bundle_sha256") or raw.get("source_manifest_sha256")!=manifest.patch_bundle_sha256: raise DiscoveryControllerError("pending manifest identity mismatch")
    intent = raw.get("experiment_intent")
    if intent is not None and not isinstance(intent, Mapping):
        raise DiscoveryControllerError("pending experiment intent is malformed")
    if intent is not None:
        expected = intent.get("expected_dispatch")
        if not isinstance(expected, Mapping):
            raise DiscoveryControllerError("pending bounded dispatch is malformed")
        recommendation = intent.get("load_mode_recommendation")
        if recommendation is not None:
            if not isinstance(recommendation, Mapping):
                raise DiscoveryControllerError("pending load-mode recommendation is malformed")
            recommendation = LoadModeRecommendation(
                mode=recommendation.get("mode"), rationale=recommendation.get("rationale"),
                example_ids=tuple(recommendation.get("example_ids", ())))
        intent = {**intent, "expected_dispatch": BoundedDispatchExpectation(**expected),
                  "load_mode_recommendation": recommendation}
    return PlannedCandidate(hypothesis_id=raw["hypothesis_id"],statement=raw["statement"],falsifier=raw["falsifier"],regime=raw["regime"],proposal=raw["proposal"],source_manifest=manifest,source_manifest_sha256=raw["source_manifest_sha256"],experiment_intent=GpuSourceExperimentIntent(**intent) if intent else None)


def _append_nomination(root: Path, item: PlannedCandidate, result: SealedScreen, threshold: float) -> None:
    # A single screen is discovery evidence, never a nomination.  Only a
    # replicated series that retained a positive pooled classification may be
    # placed in the operator queue.
    if (result.series_effect_fraction is None
            or result.series_effect_fraction < threshold
            or result.classification != "top_k_replicated_candidate"):
        return
    path=root / "promotion-queue.jsonl"; lock=root / "promotion-queue.lock"; key=_sha({"result":result.result_sha256,"manifest":item.source_manifest_sha256})
    row={"schema":"epyc.autokernel.discovery_nomination.v1","idempotency_key":key,"receipt_path":result.receipt_path,"result_sha256":result.result_sha256,"source_manifest_sha256":item.source_manifest_sha256,"effect_fraction":result.effect_fraction,"series_effect_fraction":result.series_effect_fraction,"threshold":threshold,"promotion_claim":False,"operator_decision_required":True,"authority":AUTHORITY}
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("a+") as guard:
        fcntl.flock(guard.fileno(),fcntl.LOCK_EX)
        existing=path.read_text() if path.exists() else ""
        if key in existing: return
        with path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())


def _write_projection(root: Path) -> None:
    # Canonical projection is derived from receipts, not planner text.
    autokernel_progression.export_progression(root=root, output=root / "surface" / "kernel_progression.json")


def classify_screen_series(effects: Sequence[float], *, component_pooled_effects: Sequence[float] = ()) -> str:
    """Discovery policy classifier; dashboard projection is not authority."""
    if not effects or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in effects):
        raise DiscoveryControllerError("screen series must contain numeric measured effects")
    if len(effects) == 1:
        return "screened_out" if effects[0] <= 0 else "candidate"
    if min(effects) < 0 < max(effects):
        return "inconclusive"
    # A materially divergent pair is no more rankable than opposite signs.
    # This is the discovery lane's 10 percentage-point spread rule, not a
    # calibration gate; it requests a retest rather than declaring a failure.
    if max(effects) - min(effects) >= 0.10:
        return "inconclusive"
    if all(v > 0 for v in effects) and component_pooled_effects and (sum(effects) / len(effects)) < max(component_pooled_effects):
        return "replicated_but_subadditive"
    if all(v > 0 for v in effects):
        return "top_k_replicated_candidate"
    return "screened_out"

def _screen_series_key(item: PlannedCandidate, result: SealedScreen) -> str:
    """Return the hash that permits only like-for-like replications to pool."""
    if result.series_key is not None:
        return result.series_key
    # Legacy/replay fakes have no explicitly captured frame.  Their fallback
    # remains conservative: different patch, regime, or immutable baseline is
    # a different series.  Live GPU adapters must populate series_key from the
    # sealed model/workload/runtime frame before returning a SealedScreen.
    return _sha({"source_manifest_sha256": item.source_manifest_sha256,
                 "regime": item.regime,
                 "baseline_sha256": result.baseline_sha256})


def _pooled_component_effects(state: Mapping[str, Any], component_keys: Sequence[str]) -> list[float]:
    values: list[float] = []
    for key in component_keys:
        effects = [float(row["effect_fraction"]) for row in state["iterations"]
                   if row.get("series_key") == key and isinstance(row.get("effect_fraction"), (int, float))]
        if effects:
            values.append(sum(effects) / len(effects))
    return values

def _classified_result(state: Mapping[str, Any], item: PlannedCandidate, result: SealedScreen) -> SealedScreen:
    series_key = _screen_series_key(item, result)
    prior = [float(row["effect_fraction"]) for row in state["iterations"]
             if row.get("series_key") == series_key
             and isinstance(row.get("effect_fraction"), (int, float))]
    # Component provenance is measured/sealed by the adapter.  Planner text
    # cannot name its own component evidence and thereby manufacture a
    # subadditivity claim.
    raw_components = result.component_series_keys
    if not isinstance(raw_components, (list, tuple)) or not all(isinstance(key, str) and HASH.fullmatch(key) for key in raw_components):
        raise DiscoveryControllerError("composition requires exact component series provenance")
    components = tuple(raw_components)
    effects = prior + [result.effect_fraction]
    classification = classify_screen_series(
        effects,
        component_pooled_effects=_pooled_component_effects(state, components),
    )
    return SealedScreen(receipt_path=result.receipt_path,result_sha256=result.result_sha256,effect_fraction=result.effect_fraction,classification=classification,baseline_sha256=result.baseline_sha256,source_proof_sha256=result.source_proof_sha256,dispatch_proof_sha256=result.dispatch_proof_sha256,candidate_only=result.candidate_only,promotion_claim=result.promotion_claim,stages=result.stages,series_key=series_key,component_series_keys=components,series_effect_fraction=float(statistics.median(effects)))


def _schedule_replication(state: dict[str, Any], *, item: PlannedCandidate,
                          authorization: hypotheses.ClaimAuthorization,
                          row: Mapping[str, Any], result: SealedScreen,
                          max_iterations: int) -> None:
    """Queue exactly one independent S2 for a positive exact series.

    Replication is not a second planner proposal.  It reuses the sealed patch,
    authorization, frame series key, and critic acceptance, then obtains a new
    resource lease at the next turn.  This supplies the evidence required for
    a nomination without conflating unrelated source patches under the same
    hypothesis.
    """
    if (result.classification != "candidate" or state["next"] > max_iterations
            or state.get("pending") is not None):
        return
    replica = dict(row)
    replica.update(turn=state["next"], status="replication_pending",
                   replication_of=result.result_sha256,
                   series_key=result.series_key,
                   component_series_keys=list(result.component_series_keys),
                   critic={"decision": "accept",
                           "reason": "independent replication of sealed candidate"})
    state["pending"] = {
        "row": replica,
        "candidate": _pending_item(item),
        # S2 receives a fresh authorization at its own compute boundary.  The
        # original token is retained as provenance only; it is never replayed
        # as permission for a second device claim.
        "confirmation": True,
        "parent_authorization": authorization.to_dict(),
    }


def run_controller(config: ControllerConfig, *, planner: Planner, critic: Critic, screener: Screener, lease: Lease) -> dict[str, Any]:
    planner_attestation, critic_attestation = dict(planner.attest()), dict(critic.attest())
    if ({k: planner_attestation.get(k) for k in SOL} != SOL
            or {k: critic_attestation.get(k) for k in TERRA} != TERRA
            or not isinstance(planner_attestation.get("runtime"), Mapping)
            or not isinstance(critic_attestation.get("runtime"), Mapping)):
        raise DiscoveryControllerError("actors did not attest the sealed Codex runtime identities")
    _require_runtime(planner_attestation["runtime"]); _require_runtime(critic_attestation["runtime"])
    _require_roster({"schema":"epyc.autokernel.discovery_roster.v2","members":[SOL,TERRA],"claude_members":0,"member_count":2})
    store=DurableState(config.output_root); lock=store.run_lock()
    try:
        return _run_controller_locked(config,planner=planner,critic=critic,screener=screener,lease=lease,store=store)
    finally:
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN); lock.close()

def _run_controller_locked(config: ControllerConfig, *, planner: Planner, critic: Critic, screener: Screener, lease: Lease, store: DurableState) -> dict[str, Any]:
    state=store.load()
    existing_deployment = state.get("deployment_identity_sha256")
    if (existing_deployment is None and config.deployment_identity_sha256 is not None
            and (state.get("iterations") or state.get("pending") is not None
                 or state.get("inflight") is not None)):
        raise DiscoveryControllerError("legacy durable state lacks deployment identity; refusing resume")
    if existing_deployment is not None and existing_deployment != config.deployment_identity_sha256:
        raise DiscoveryControllerError("sealed deployment identity changed; durable discovery cannot resume")
    if existing_deployment is None and config.deployment_identity_sha256 is not None:
        state["deployment_identity_sha256"] = config.deployment_identity_sha256
    existing_context = state.get("planner_context_sha256")
    if existing_context is not None and existing_context != config.planner_context_sha256:
        raise DiscoveryControllerError("sealed planner context changed; durable discovery cannot resume")
    if existing_context is None and config.planner_context_sha256 is not None:
        state["planner_context_sha256"] = config.planner_context_sha256
    existing_templates = state.get("experiment_template_registry_sha256")
    if existing_templates is not None and existing_templates != config.experiment_template_registry_sha256:
        raise DiscoveryControllerError("sealed experiment-template registry changed; durable discovery cannot resume")
    if existing_templates is None and config.experiment_template_registry_sha256 is not None:
        state["experiment_template_registry_sha256"] = config.experiment_template_registry_sha256
    existing_corpus = state.get("admission_corpus_sha256")
    if existing_corpus is not None and existing_corpus != config.admission_corpus_sha256:
        raise DiscoveryControllerError("sealed admission corpus changed; durable discovery cannot resume")
    if existing_corpus is None and config.admission_corpus_sha256 is not None:
        state["admission_corpus_sha256"] = config.admission_corpus_sha256
    existing_corpus_version = state.get("admission_corpus_version")
    if existing_corpus_version is not None and existing_corpus_version != config.admission_corpus_version:
        raise DiscoveryControllerError("sealed admission corpus version changed; durable discovery cannot resume")
    if existing_corpus_version is None and config.admission_corpus_version is not None:
        state["admission_corpus_version"] = config.admission_corpus_version
    # A completed state is an acknowledged terminal checkpoint.  Re-entering it
    # must be a read, not another executor opportunity or a timestamp rewrite.
    if state["complete"]: return state
    tracker=_tracker(store)
    if state.get("inflight") is not None:
        inflight=state["inflight"]; item=_restore_pending({"candidate":inflight["candidate"]}); authorization=hypotheses.ClaimAuthorization.from_dict(inflight["authorization"]); permit=inflight["lease"]
        if isinstance(inflight.get("result"),Mapping): result=SealedScreen(**inflight["result"])
        else:
            reconcile=getattr(screener,"reconcile",None)
            if not callable(reconcile): raise DiscoveryControllerError("inflight operation has no reconciliation adapter")
            recovery=reconcile(inflight)
            if not isinstance(recovery,Recovery) or recovery.status == "ambiguous": raise DiscoveryControllerError("inflight operation cannot be safely reconciled")
            result=recovery.result if recovery.status == "sealed_result" else screener.screen(item,authorization,permit)
        if not isinstance(result,SealedScreen): raise DiscoveryControllerError("inflight recovery produced no sealed result")
        result=_classified_result(state,item,result); row=dict(inflight["row"]); row.update(status=result.classification,result_sha256=result.result_sha256,evidence={"baseline":result.baseline_sha256,"source":result.source_proof_sha256,"dispatch":result.dispatch_proof_sha256},effect_fraction=result.effect_fraction,series_effect_fraction=result.series_effect_fraction,series_key=result.series_key,component_series_keys=list(result.component_series_keys))
        _record_attempt_once(tracker,item,str(item.proposal.get("proposal_id",row["proposal_sha256"])),result)
        state.pop("inflight",None); state.pop("pending",None); state["iterations"].append(row); state["next"]+=1; _schedule_replication(state,item=item,authorization=authorization,row=row,result=result,max_iterations=config.max_iterations); _append_nomination(config.output_root,item,result,config.nomination_threshold); _write_projection(config.evidence_root or config.output_root); store.save(state,"recovered_screen")
    while not state["complete"] and state["next"] <= config.max_iterations:
        turn=state["next"]; context=_context(state,tracker,turn,config)
        with tempfile.TemporaryDirectory(prefix=f"ak-discovery-{turn}-", dir=config.output_root) as temp:
            workspace=Path(temp)
            pending=state.get("pending")
            if pending is not None:
                item=_restore_pending(pending); row=dict(pending["row"]); review=Critique(**row["critic"])
                if "authorization" in pending:
                    authorization=hypotheses.ClaimAuthorization.from_dict(pending["authorization"])
                elif pending.get("confirmation") is True:
                    # A positive S1 is not a receipted negative.  Re-consult
                    # DNR and mint the explicit confirmation token before its
                    # own device claim, rather than reusing S1's token.
                    _ensure_question(tracker,item)
                    authorization=tracker.authorize_claim(item.hypothesis_id,purpose="candidate_only_confirmation",authorized_by="discovery_controller",ledger=do_not_repeat.compile_for_tracker(tracker))
                else:
                    raise DiscoveryControllerError("pending candidate lacks a sealed authorization")
            else:
                item=planner.plan(context=context,workspace=workspace)
                review=critic.review(item,context=context,workspace=workspace)
                row={"turn":turn,"hypothesis_id":item.hypothesis_id,"statement":item.statement,
                     "falsifier":item.falsifier,"regime":dict(item.regime),
                     "proposal_sha256":_sha(item.proposal),"source_manifest_sha256":item.source_manifest_sha256,
                     "experiment_intent":asdict(item.experiment_intent) if item.experiment_intent else None,
                     "mechanism_id":item.source_manifest.mechanism_id,
                     "target_surface":item.experiment_intent.target_surface if item.experiment_intent else None,
                     "target_symbol":item.experiment_intent.target_symbol if item.experiment_intent else None,
                     "critic":asdict(review),"context_sha256":_sha(context)}
            if review.decision != "accept":
                row["status"]="critic_"+review.decision; state["iterations"].append(row); state["next"]+=1; store.save(state,"critic_refused"); continue
            if pending is None:
                _ensure_question(tracker,item)
                ledger=do_not_repeat.compile_for_tracker(tracker)
                try: authorization=tracker.authorize_claim(item.hypothesis_id,purpose="candidate_only_discovery",authorized_by="discovery_controller",ledger=ledger)
                except hypotheses.HypothesisError as exc:
                    row.update(status="authorization_refused",reason=str(exc)); state["iterations"].append(row); state["next"]+=1; store.save(state,"authorization_refused"); continue
            if config.dry_run:
                # The dry-run still proves exact Sol/Terra actor attestation,
                # plan schema, critic binding, and DNR authorization.  It
                # deliberately never asks for a resource lease or starts a
                # source build, correctness, attribution, or model call.
                row.update(status="dry_run_authorized", authorization=authorization.to_dict())
                state.pop("pending", None); state["iterations"].append(row); state["next"] += 1
                store.save(state, "dry_run_authorized")
                continue
            permit=lease.admit(item)
            if not bool(permit.get("admitted")):
                # Waiting is durable but is not an experiment and cannot spend an
                # iteration budget.  Planning/critique may continue elsewhere;
                # this exact candidate is retried only after a new lease admits it.
                row.update(status="waiting_resource",lease=dict(permit)); state["pending"]={"row":row,"candidate":_pending_item(item),"authorization":authorization.to_dict(),"confirmation":bool(pending and pending.get("confirmation")),"parent_authorization":pending.get("parent_authorization") if pending else None}; store.save(state,"waiting_resource"); break
            repetition=2 if pending and pending.get("confirmation") else 1
            operation_key=_sha({"turn":turn,"manifest":item.source_manifest_sha256,"authorization":authorization.to_dict(),"repetition":repetition})
            # The governed GPU adapter owns an operation-key-bound receipt
            # namespace.  It refuses an unkeyed lease, making recovery and
            # result reconciliation refer to the same durable operation.
            permit={**dict(permit), "operation_key":operation_key, "repetition":repetition}
            state["inflight"]={"operation_key":operation_key,"row":row,"candidate":_pending_item(item),"authorization":authorization.to_dict(),"lease":dict(permit)}
            store.save(state,"pre_screen_intent")
            try: result=screener.screen(item,authorization,permit)
            except PrecomputeScreenRefusal as exc:
                state.pop("inflight",None); row.update(status="screen_refused",reason=f"{type(exc).__name__}: {exc}"); state["iterations"].append(row); state["next"]+=1; store.save(state,"screen_refused"); continue
            except Exception as exc:
                # The durable start intent has been written.  An ordinary
                # exception may follow a build, claim, or model invocation, so
                # it is an ambiguous operation until the governed adapter
                # reconciles its operation-key-bound artifacts on restart.
                state["inflight"]["exception"]={"type":type(exc).__name__,"message":str(exc)}
                store.save(state,"screen_ambiguous")
                raise
            state["inflight"]["result"]=asdict(result); store.save(state,"post_screen_result")
            result=_classified_result(state,item,result); row.update(status=result.classification,result_sha256=result.result_sha256,evidence={"baseline":result.baseline_sha256,"source":result.source_proof_sha256,"dispatch":result.dispatch_proof_sha256},effect_fraction=result.effect_fraction,series_effect_fraction=result.series_effect_fraction,series_key=result.series_key,component_series_keys=list(result.component_series_keys))
            # Record the measured disposition before exposing it to the next
            # planner context.  This is the only source of repeat suppression.
            _record_attempt_once(tracker,item,str(item.proposal.get("proposal_id",row["proposal_sha256"])),result)
            state.pop("inflight",None); state.pop("pending",None); state["iterations"].append(row); state["next"]+=1; _schedule_replication(state,item=item,authorization=authorization,row=row,result=result,max_iterations=config.max_iterations); _append_nomination(config.output_root,item,result,config.nomination_threshold); _write_projection(config.evidence_root or config.output_root); store.save(state,"screened")
    state["complete"]=state["next"]>config.max_iterations
    if state["complete"]: state.pop("pending",None)
    store.save(state,"complete" if state["complete"] else "paused"); return state


def build_controller_adapters(*, planner: Planner, critic: Critic, screener: Screener,
                              lease: Lease) -> dict[str, Any]:
    """Bind the four concrete controller seams without accepting shell commands.

    A live factory constructs its governed GPU source screener separately (with
    its proof producer configuration) and supplies that object here.  Keeping
    this top-level boundary object-only prevents a JSON launch manifest from
    becoming an arbitrary command executor.
    """
    parts = {"planner": planner, "critic": critic, "screener": screener, "lease": lease}
    if any(value is None for value in parts.values()):
        raise DiscoveryControllerError("controller adapters must bind every required seam")
    if not callable(getattr(planner, "plan", None)) or not callable(getattr(critic, "review", None)):
        raise DiscoveryControllerError("controller factory did not bind typed planner/critic actors")
    if not callable(getattr(screener, "screen", None)) or not callable(getattr(screener, "reconcile", None)):
        raise DiscoveryControllerError("controller factory did not bind governed screen/reconcile adapter")
    if not callable(getattr(lease, "admit", None)):
        raise DiscoveryControllerError("controller factory did not bind a resource lease adapter")
    return parts


def _load_adapter_config(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).resolve(strict=True)
    if config_path.is_symlink() or not config_path.is_file():
        raise DiscoveryControllerError("adapter config must be a regular file")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiscoveryControllerError("adapter config must be a JSON object") from exc
    if not isinstance(value, Mapping):
        raise DiscoveryControllerError("adapter config must be a JSON object")
    return dict(value)


def _load_factory(reference: str, factory_config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if reference.count(":") != 1:
        raise DiscoveryControllerError("adapter factory must be module:callable")
    module, name = reference.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise DiscoveryControllerError("adapter factory could not be imported") from exc
    if not callable(factory):
        raise DiscoveryControllerError("adapter factory must be callable")
    # Standard factory contract is factory(config: Mapping) -> adapter bundle.
    # A no-argument factory stays useful for narrowly-bound deployment modules.
    try:
        signature = inspect.signature(factory)
        value = factory() if not signature.parameters else factory(dict(factory_config or {}))
    except (TypeError, ValueError) as exc:
        raise DiscoveryControllerError("adapter factory has an unsupported signature") from exc
    if not isinstance(value, Mapping):
        raise DiscoveryControllerError("adapter factory must return mapping")
    try:
        return build_controller_adapters(**dict(value))
    except TypeError as exc:
        raise DiscoveryControllerError("adapter factory must return exactly planner, critic, screener, lease") from exc


def main(argv: Sequence[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root",required=True); parser.add_argument("--max-iterations",type=int,default=1); parser.add_argument("--dry-run",action="store_true"); parser.add_argument("--adapter-factory", required=True); parser.add_argument("--adapter-config"); parser.add_argument("--evidence-root")
    args=parser.parse_args(argv); config=ControllerConfig(Path(args.output_root).resolve(),args.max_iterations,dry_run=args.dry_run,evidence_root=Path(args.evidence_root).resolve() if args.evidence_root else None)
    parts=_load_factory(args.adapter_factory, _load_adapter_config(args.adapter_config)); run_controller(config,planner=parts["planner"],critic=parts["critic"],screener=parts["screener"],lease=parts["lease"]); return 0

if __name__=="__main__": main()
