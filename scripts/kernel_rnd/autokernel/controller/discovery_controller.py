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
from pathlib import Path, PurePosixPath
import re
import statistics
import stat
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence

from .. import campaign, hypothesis_portfolio, journal, schemas, source_candidate
from ..evaluator import integrity
from . import (claude_fable5_critic_actor, codex_container_actor,
               discovery_telemetry, do_not_repeat, hypotheses)
from . import gpu_source_proofs
from scripts.benchmark import autokernel_progression
from scripts.benchmark import run_autokernel_gpu_discovery as gpu_discovery

SCHEMA = "epyc.autokernel.discovery_controller.v5"
ROSTER_SCHEMA = "epyc.autokernel.discovery_roster.v3"
AUTHORITY = "nonpromotable_candidate_only_discovery"
HASH = __import__("re").compile(r"^[0-9a-f]{64}$")
PORTFOLIO_DNR_CHECK_SCHEMA = "epyc.autokernel.portfolio_exact_dnr_check.v1"
SOL = {"provider": "codex", "model": "gpt-5.6-sol", "effort": "high", "role": "planner"}
FABLE5_CRITIC = {"provider": "claude", "model": "claude-fable-5", "effort": "high", "role": "critic"}


class DiscoveryControllerError(RuntimeError): pass


class PlannerOutputRefusal(DiscoveryControllerError):
    """A safe, bounded refusal of a completed planner's authored artifacts.

    This type is deliberately narrower than ``DiscoveryControllerError``.  It
    may be raised only after the Sol process returned successfully and while
    validating files in its disposable workspace.  Runtime, authentication,
    containment, and later source/build failures must retain their native
    exception types.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # Telemetry is observational.  A telemetry failure must never replace
        # the already-derived, controller-owned planner refusal.
        self.telemetry_status = "not_attempted"
        self.telemetry_failure: dict[str, str] | None = None

    def note_telemetry_failure(self, exc: Exception) -> None:
        self.telemetry_status = "emit_failed"
        self.telemetry_failure = {
            "type": type(exc).__name__,
            "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
        }


class PlannerProviderTransient(PlannerOutputRefusal):
    """A retryable provider/API interruption before candidate validation."""


class GovernedStageRefusal(DiscoveryControllerError):
    stage = ""
    disposition = ""
    scientific_budget_spent = False

    def __init__(self, message: str, *, receipt_path: str,
                 receipt_sha256: str) -> None:
        super().__init__(message)
        if (not isinstance(receipt_path, str) or not receipt_path
                or not isinstance(receipt_sha256, str)
                or not HASH.fullmatch(receipt_sha256)):
            raise DiscoveryControllerError(
                "governed stage refusal lacks a sealed receipt")
        self.receipt_path = receipt_path
        self.receipt_sha256 = receipt_sha256


class SourceApplyRefusal(GovernedStageRefusal):
    stage = "source_apply"
    disposition = "authoring_refused"


class CompileRefusal(GovernedStageRefusal):
    stage = "compile"
    disposition = "authoring_refused"


class CorrectnessRefusal(GovernedStageRefusal):
    stage = "correctness"
    disposition = "correctness_falsified"


class DispatchAttributionRefusal(GovernedStageRefusal):
    stage = "dispatch_attribution"
    disposition = "attribution_route_falsified"


class PrecomputeScreenRefusal(DiscoveryControllerError):
    """Typed adapter refusal proving that no governed operation was started."""


class PostBuildEvidencePlanRefusal(PrecomputeScreenRefusal):
    """A completed build was refused before claim/proof/runner execution.

    The builder's exact terminal remains reusable by its sealed build key; the
    controller may durably classify this screen without treating the operation
    as an ambiguous GPU run.
    """


class ResumableScreenInterruption(DiscoveryControllerError):
    """A pre-run transport failure after durable proof was checkpointed.

    The current candidate and its scientific proof remain inflight.  The
    controller pauses without consuming an iteration so a corrected/reloaded
    runner can resume at the first incomplete stage.
    """


class ResourceWait(DiscoveryControllerError):
    """A durable pre-executor refusal caused only by resource contention."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str: return hashlib.sha256(_canon(value)).hexdigest()


def _emit_observational_telemetry(
        telemetry: discovery_telemetry.DiscoveryTelemetry | None,
        *args: Any, **kwargs: Any) -> Exception | None:
    """Emit dashboard telemetry without granting it controller authority.

    The durable state machine and actor result remain primary.  Returning the
    telemetry exception lets a typed primary refusal record the visibility
    degradation without allowing it to replace that refusal.
    """
    if telemetry is None:
        return None
    try:
        telemetry.emit(*args, **kwargs)
    except Exception as exc:
        return exc
    return None


def _validated_resource_wait(exc: ResourceWait, operation_key: str) -> dict[str, Any]:
    receipt = dict(exc.receipt)
    required = {
        "admitted": False,
        "phase": "pre_executor_reservation",
        "operation_key": operation_key,
        "promotion_claim": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise DiscoveryControllerError("resource wait does not bind the pre-executor operation")
    path = Path(str(receipt.get("stage_receipt_path", "")))
    expected = receipt.get("stage_receipt_sha256")
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.parent.name != "resource-waits"
            or path.parent.parent.name != operation_key
            or not isinstance(expected, str) or not HASH.fullmatch(expected)):
        raise DiscoveryControllerError("resource wait lacks its durable stage receipt")
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or before.st_nlink != 1 or before.st_mode & 0o022):
                raise DiscoveryControllerError(
                    "resource wait stage receipt has unsafe file authority")
            with os.fdopen(descriptor, "rb") as handle:
                raw = handle.read()
                after = os.fstat(handle.fileno())
            if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_nlink)
                    != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_nlink)):
                raise DiscoveryControllerError("resource wait stage receipt changed while read")
        except BaseException:
            try: os.close(descriptor)
            except OSError: pass
            raise
        if hashlib.sha256(raw).hexdigest() != expected:
            raise DiscoveryControllerError("resource wait stage receipt hash changed")
        stage = json.loads(raw.decode("utf-8", "strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiscoveryControllerError("resource wait stage receipt is unreadable") from error
    stage_required = {
        "schema": "epyc.autokernel.gpu_source_resource_wait.v1",
        "authority": AUTHORITY,
        "promotion_claim": False,
        "operation_key": operation_key,
        "gpu_executor_started": False,
        "proof_root_created": False,
        "runner_plan_created": False,
        "runner_output_created": False,
    }
    if (not isinstance(stage, Mapping)
            or any(stage.get(key) != value for key, value in stage_required.items())
            or stage.get("contention") != {
                key: value for key, value in receipt.items()
                if key not in {"stage_receipt_path", "stage_receipt_sha256"}}
            or stage.get("receipt_sha256") != _sha({
                key: value for key, value in stage.items() if key != "receipt_sha256"})):
        raise DiscoveryControllerError("resource wait stage receipt is not a sealed pre-executor proof")
    return receipt


def _require_safe_resource_wait_recovery(screener: Screener,
                                         inflight: Mapping[str, Any]) -> None:
    reconcile = getattr(screener, "reconcile", None)
    if not callable(reconcile):
        raise DiscoveryControllerError("resource wait lacks reconciliation authority")
    recovery = reconcile(inflight)
    if not isinstance(recovery, Recovery) or recovery.status != "safe_to_start":
        raise DiscoveryControllerError(
            "resource wait conflicts with current operation artifacts")


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
    return {"schema": ROSTER_SCHEMA, "members": [SOL, FABLE5_CRITIC], "claude_members": 1, "member_count": 2}


def _require_roster(value: Mapping[str, Any]) -> None:
    if dict(value) != sealed_roster(): raise DiscoveryControllerError("runtime roster is not exact Sol planner + Claude Fable 5 critic")

def _require_runtime(value: Mapping[str, Any]) -> None:
    required={"kind","docker_path","docker_sha256","image_id","codex_native_sha256","code_mode_host_sha256","ca_certificate_sha256","writable_host_binds","host_network_mode"}
    if set(value) != required or value.get("kind")!="docker_workspace_bind_only" or value.get("host_network_mode")!="docker_bridge" or value.get("writable_host_binds") != ["/workspace"] or not all(isinstance(value.get(k),str) and value[k] for k in required-{"writable_host_binds"}): raise DiscoveryControllerError("Codex runtime attestation is incomplete or unsealed")


def _require_claude_runtime(value: Mapping[str, Any]) -> None:
    """Require a non-secret, byte-bound Fable 5 CLI runtime receipt."""
    required = {"kind", "provider", "model", "effort", "wrapper_path",
                "wrapper_sha256", "argv_policy_sha256", "auth_staging_policy"}
    if (set(value) != required
            or value.get("kind") != "claude_cli_structured_critic"
            or value.get("provider") != FABLE5_CRITIC["provider"]
            or value.get("model") != FABLE5_CRITIC["model"]
            or value.get("effort") != FABLE5_CRITIC["effort"]
            or value.get("auth_staging_policy")
            != claude_fable5_critic_actor.AUTH_STAGING_POLICY
            or not all(isinstance(value.get(key), str) and value[key]
                       for key in required - {"auth_staging_policy"})):
        raise DiscoveryControllerError("Claude critic runtime attestation is incomplete or unsealed")


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
    portfolio_binding: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (not self.campaign_id.startswith("ak-") or not self.proposal_id.startswith("akp-")
                or not self.candidate_id.startswith("akc-")
                or not all(isinstance(value, str) and len(value) == 40
                           and all(ch in "0123456789abcdef" for ch in value)
                           for value in (self.production_base_commit, self.instrument_commit))):
            raise DiscoveryControllerError("invalid controller-owned authoring identity")
        if self.portfolio_binding is not None:
            required = {"portfolio_sha256", "record_sha256", "hypothesis_id",
                        "statement", "falsifier", "mechanism_id", "regime",
                        "target_file", "target_symbols", "template_id",
                        "change_class", "decision_policy", "expected_dispatch"}
            value = self.portfolio_binding
            if (not isinstance(value, Mapping) or set(value) != required
                    or not HASH.fullmatch(str(value.get("portfolio_sha256")))
                    or not HASH.fullmatch(str(value.get("record_sha256")))
                    or not all(isinstance(value.get(key), str) and value[key]
                               for key in ("hypothesis_id", "statement", "falsifier",
                                           "mechanism_id", "target_file", "template_id",
                                           "change_class"))
                    or not isinstance(value.get("regime"), Mapping)
                    or not isinstance(value.get("target_symbols"), (list, tuple))
                    or not value["target_symbols"]
                    or not all(isinstance(item, str) and item
                               for item in value["target_symbols"])
                    or not isinstance(value.get("decision_policy"), Mapping)
                    or not isinstance(value.get("expected_dispatch"), (list, tuple))
                    or not 1 <= len(value["expected_dispatch"]) <= 8
                    or not all(isinstance(row, Mapping)
                               and set(row) == {"route_id", "kernel_name", "calls", "grid",
                                               "workgroup", "lds_bytes"}
                               for row in value["expected_dispatch"])):
                raise DiscoveryControllerError("invalid controller-owned portfolio binding")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedDispatchExpectation:
    """Planner-authored literal geometry; never a regex, argv, or command."""
    route_id: str
    kernel_name: str
    calls: int
    grid: int
    workgroup: int
    lds_bytes: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*\.anchor\.[0-9]+", self.route_id):
            raise DiscoveryControllerError("dispatch route id is not deployed authority")
        # The reviewed profilers report complete demangled HIP symbols: v1 adds
        # ``[clone .kd]`` while v3 emits the native undecorated name.  This is
        # still a literal: the deployment factory escapes it before constructing
        # an evidence matcher.  Punctuation is admitted only on function-shaped
        # names; bare regex-like planner strings remain invalid.
        if (not isinstance(self.kernel_name, str)
                or not 1 <= len(self.kernel_name.encode("utf-8")) <= 2048
                or any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in self.kernel_name)
                or (any(ch in self.kernel_name for ch in "*?[]|+\\^$")
                    and "(" not in self.kernel_name)
                or (" " in self.kernel_name and "(" not in self.kernel_name)):
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
                or len(self.example_ids) > 8
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
    expected_dispatch: tuple[BoundedDispatchExpectation, ...]
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
        if (not isinstance(self.expected_dispatch, tuple)
                or not 1 <= len(self.expected_dispatch) <= 8
                or not all(isinstance(item, BoundedDispatchExpectation)
                           for item in self.expected_dispatch)
                or len({(item.route_id, item.kernel_name, item.grid,
                         item.workgroup, item.lds_bytes)
                        for item in self.expected_dispatch}) != len(self.expected_dispatch)):
            raise DiscoveryControllerError(
                "experiment intent requires 1..8 distinct bounded literal dispatch expectations")
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
    exact_attribution_effect_fraction: float | None = None
    target_runtime_effect_fraction: float | None = None
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
        for label, value in (
                ("exact attribution", self.exact_attribution_effect_fraction),
                ("target runtime", self.target_runtime_effect_fraction)):
            if (value is not None and (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)))):
                raise DiscoveryControllerError(
                    f"{label} effect must be a finite measured number")
        if (self.target_runtime_effect_fraction is not None
                and float(self.effect_fraction)
                != float(self.target_runtime_effect_fraction)):
            raise DiscoveryControllerError(
                "primary screen effect must be the target-runtime effect")
        if not self.candidate_only or self.promotion_claim: raise DiscoveryControllerError("discovery screen must remain nonpromotable")
        if tuple(self.stages) not in {
                ("materialized", "built", "correctness", "attribution", "screen"),
                ("materialized", "built", "correctness", "attribution",
                 "measurement_graphs_off_screen",
                 "target_runtime_graphs_on_screen"),
                ("materialized", "built", "correctness", "attribution")}:
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
    def plan(self, *, context: Mapping[str, Any], workspace: Path,
             checkpoint_path: Path | None = None) -> PlannedCandidate: ...
    def resume_plan(self, *, context: Mapping[str, Any],
                    workspace: Path, checkpoint_path: Path) -> PlannedCandidate: ...

class Critic(Protocol):
    def attest(self) -> Mapping[str, Any]: ...
    def review(self, candidate: PlannedCandidate, *, context: Mapping[str, Any], workspace: Path) -> Critique: ...

class Lease(Protocol):
    def admit(self, candidate: PlannedCandidate, *, operation_key: str) -> Mapping[str, Any]: ...
    def resume(self, candidate: PlannedCandidate,
               stale_permit: Mapping[str, Any]) -> Mapping[str, Any]: ...

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


@dataclass(frozen=True)
class ReviewedSourceFile:
    relative_path: str
    sha256: str
    content: bytes

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (path.is_absolute() or path.as_posix() != self.relative_path
                or any(part in {"", ".", ".."} for part in path.parts)
                or not HASH.fullmatch(self.sha256)
                or hashlib.sha256(self.content).hexdigest() != self.sha256):
            raise DiscoveryControllerError("reviewed source file identity is malformed")


@dataclass(frozen=True)
class ReviewedSourcePackage:
    instrument_commit: str
    files: tuple[ReviewedSourceFile, ...]
    package_sha256: str

    def __post_init__(self) -> None:
        if (not re.fullmatch(r"[0-9a-f]{40}", self.instrument_commit)
                or not self.files
                or tuple(sorted(item.relative_path for item in self.files)) != tuple(
                    item.relative_path for item in self.files)
                or len({item.relative_path for item in self.files}) != len(self.files)):
            raise DiscoveryControllerError("reviewed source package is malformed")
        body = {"schema": "epyc.autokernel.reviewed_source_package.v1",
                "instrument_commit": self.instrument_commit,
                "files": [{"relative_path": item.relative_path, "sha256": item.sha256,
                           "workspace_path": f"reviewed-source/{item.relative_path}"}
                          for item in self.files]}
        if self.package_sha256 != _sha(body):
            raise DiscoveryControllerError("reviewed source package hash mismatch")

    def manifest(self) -> dict[str, Any]:
        body = {"schema": "epyc.autokernel.reviewed_source_package.v1",
                "instrument_commit": self.instrument_commit,
                "files": [{"relative_path": item.relative_path, "sha256": item.sha256,
                           "workspace_path": f"reviewed-source/{item.relative_path}"}
                          for item in self.files]}
        return {**body, "package_sha256": self.package_sha256}

    def _manifest_bytes(self) -> bytes:
        return json.dumps(self.manifest(), sort_keys=True, indent=2).encode() + b"\n"

    @staticmethod
    def _require_owned_directory(path: Path, label: str) -> None:
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or path.is_symlink()
                or info.st_uid != os.getuid() or info.st_nlink < 2):
            raise DiscoveryControllerError(f"{label} is not an owned non-symlink directory")

    def revalidate_materialized(self, workspace: Path) -> None:
        self._require_owned_directory(workspace, "actor workspace")
        root = workspace / "reviewed-source"
        self._require_owned_directory(root, "reviewed source root")
        for item in self.files:
            target = root.joinpath(*PurePosixPath(item.relative_path).parts)
            current = root
            for part in PurePosixPath(item.relative_path).parts[:-1]:
                current = current / part
                self._require_owned_directory(current, "reviewed source parent")
            info = target.lstat()
            if (not stat.S_ISREG(info.st_mode) or target.is_symlink()
                    or info.st_uid != os.getuid() or info.st_nlink != 1
                    or hashlib.sha256(target.read_bytes()).hexdigest() != item.sha256):
                raise DiscoveryControllerError("reviewed source bytes changed in actor workspace")
        manifest_path = root / "source-package.json"
        info = manifest_path.lstat()
        if (not stat.S_ISREG(info.st_mode) or manifest_path.is_symlink()
                or info.st_uid != os.getuid() or info.st_nlink != 1
                or manifest_path.read_bytes() != self._manifest_bytes()):
            raise DiscoveryControllerError("reviewed source package manifest changed")

    def materialize(self, workspace: Path) -> Mapping[str, Any]:
        self._require_owned_directory(workspace, "actor workspace")
        root = workspace / "reviewed-source"
        if root.exists() or root.is_symlink():
            raise DiscoveryControllerError("disposable reviewed-source root already exists")
        root.mkdir(mode=0o700)
        for item in self.files:
            target = root.joinpath(*PurePosixPath(item.relative_path).parts)
            target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(item.content); handle.flush(); os.fsync(handle.fileno())
            target.chmod(0o400)
            if (target.is_symlink() or target.stat().st_nlink != 1
                    or hashlib.sha256(target.read_bytes()).hexdigest() != item.sha256):
                raise DiscoveryControllerError("reviewed source materialization changed bytes")
        manifest = self.manifest()
        encoded = self._manifest_bytes()
        manifest_path = root / "source-package.json"
        with manifest_path.open("xb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        manifest_path.chmod(0o400)
        self.revalidate_materialized(workspace)
        return manifest

    def critic_context(self, relative_path: str,
                       symbols: Sequence[str]) -> Mapping[str, Any]:
        matches = [item for item in self.files if item.relative_path == relative_path]
        if len(matches) != 1 or not symbols:
            raise DiscoveryControllerError("critic source preimage is outside reviewed package")
        item = matches[0]
        try:
            lines = item.content.decode("utf-8", "strict").splitlines(keepends=True)
        except UnicodeDecodeError as exc:
            raise DiscoveryControllerError("critic source preimage is not UTF-8") from exc
        ranges: list[tuple[int, int]] = []
        for symbol in symbols:
            indexes = [index for index, line in enumerate(lines) if symbol in line]
            if not indexes:
                raise DiscoveryControllerError(
                    f"critic source preimage lacks selected symbol: {symbol}")
            index = indexes[0]
            ranges.append((max(0, index - 24), min(len(lines), index + 25)))
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        excerpts = []
        total = 0
        for start, end in merged:
            text = "".join(lines[start:end])
            total += len(text.encode("utf-8"))
            excerpts.append({"line_start": start + 1, "line_end": end,
                             "text": text,
                             "sha256": hashlib.sha256(text.encode()).hexdigest()})
        if total > 65536:
            raise DiscoveryControllerError("critic source preimage excerpt exceeds bound")
        value = {"schema": "epyc.autokernel.critic_source_preimage.v1",
                 "relative_path": relative_path, "source_sha256": item.sha256,
                 "symbols": list(symbols), "excerpts": excerpts}
        return {**value, "context_sha256": _sha(value)}


PLANNER_ACTOR_CHECKPOINT_SCHEMA = "epyc.autokernel.planner_actor_checkpoint.v1"


def _planner_artifact_manifest(workspace: Path) -> dict[str, Any]:
    """Hash every actor-owned artifact outside the immutable source package."""
    root_info = workspace.lstat()
    if (workspace.is_symlink() or not stat.S_ISDIR(root_info.st_mode)
            or root_info.st_uid != os.getuid()):
        raise DiscoveryControllerError("planner workspace is not an owned directory")
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    total = 0
    for path in sorted(workspace.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(workspace)
        if relative.parts[0] == "reviewed-source":
            continue
        info = path.lstat()
        if path.is_symlink() or info.st_uid != os.getuid():
            raise DiscoveryControllerError(
                "planner artifact tree contains a symlink or foreign owner")
        if stat.S_ISDIR(info.st_mode):
            directories.append(relative.as_posix())
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise DiscoveryControllerError(
                "planner artifact tree contains a special file or hardlink")
        raw = path.read_bytes()
        total += len(raw)
        files.append({"path": relative.as_posix(), "size": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()})
    if len(files) > 32 or len(directories) > 32 or total > 2 * 1024 * 1024:
        raise DiscoveryControllerError("planner artifact tree exceeds its sealed bound")
    return {"directories": directories, "files": files,
            "total_bytes": total}


def _seal_planner_actor_checkpoint(workspace: Path, checkpoint_path: Path, *,
                                   context: Mapping[str, Any],
                                   result: Mapping[str, Any]) -> Mapping[str, Any]:
    path = checkpoint_path
    if path.parent != workspace.parent or path.name != "actor-result.json":
        raise DiscoveryControllerError(
            "planner actor checkpoint is outside its controller operation")
    ReviewedSourcePackage._require_owned_directory(
        path.parent, "planner operation root")
    if path.exists() or path.is_symlink():
        raise DiscoveryControllerError("planner actor checkpoint already exists")
    body = {
        "schema": PLANNER_ACTOR_CHECKPOINT_SCHEMA,
        "context_sha256": _sha(context),
        "assignment_sha256": _sha(context.get("authoring_assignment")),
        "result": dict(result),
        "artifacts": _planner_artifact_manifest(workspace),
    }
    body["receipt_sha256"] = _sha(body)
    _atomic(path, body)
    return body


def _reopen_planner_actor_checkpoint(workspace: Path, checkpoint_path: Path, *,
                                     context: Mapping[str, Any]) -> Mapping[str, Any]:
    path = checkpoint_path
    if path.parent != workspace.parent or path.name != "actor-result.json":
        raise DiscoveryControllerError(
            "planner actor checkpoint is outside its controller operation")
    ReviewedSourcePackage._require_owned_directory(
        path.parent, "planner operation root")
    if not path.exists():
        raise PlannerOutputRefusal(
            "planner invocation stopped without a completed actor artifact checkpoint")
    info = path.lstat()
    if (path.is_symlink() or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid() or info.st_nlink != 1):
        raise DiscoveryControllerError("planner actor checkpoint file is unsafe")
    checkpoint = _read_object(path, workspace.parent)
    declared = checkpoint.get("receipt_sha256")
    if (checkpoint.get("schema") != PLANNER_ACTOR_CHECKPOINT_SCHEMA
            or not isinstance(declared, str)
            or declared != _sha({key: value for key, value in checkpoint.items()
                                 if key != "receipt_sha256"})
            or checkpoint.get("context_sha256") != _sha(context)
            or checkpoint.get("assignment_sha256") !=
               _sha(context.get("authoring_assignment"))
            or checkpoint.get("artifacts") !=
               _planner_artifact_manifest(workspace)
            or not isinstance(checkpoint.get("result"), Mapping)
            or isinstance(checkpoint["result"].get("returncode"), bool)
            or not isinstance(checkpoint["result"].get("returncode"), int)):
        raise DiscoveryControllerError("planner actor checkpoint identity changed")
    return checkpoint


class CodexPlanner:
    """Concrete Sol actor. It may write only a plan and patch manifest in workspace."""
    def __init__(self, *, wrapper: Path, environment: Mapping[str, str],
                 template_catalog: Mapping[str, Any] | None = None,
                 reviewed_sources: ReviewedSourcePackage | None = None,
                 wrapper_sha256: str | None = None,
                 runtime_identity: Mapping[str, Any] | None = None,
                 actor_launcher_sha256: str | None = None,
                 telemetry: discovery_telemetry.DiscoveryTelemetry | None = None) -> None:
        self.wrapper, self.environment = wrapper, dict(environment)
        self.template_catalog = json.loads(json.dumps(template_catalog or {}, sort_keys=True))
        self.reviewed_sources = reviewed_sources
        self.wrapper_sha256 = wrapper_sha256
        self.runtime_identity = None if runtime_identity is None else dict(runtime_identity)
        self.actor_launcher_sha256 = actor_launcher_sha256
        self.telemetry = telemetry
    def _runtime(self) -> Mapping[str, Any]:
        if self.wrapper_sha256 is not None:
            if self.wrapper.is_symlink() or not self.wrapper.is_file() or hashlib.sha256(self.wrapper.read_bytes()).hexdigest() != self.wrapper_sha256:
                raise DiscoveryControllerError("sealed Codex planner wrapper bytes changed")
        current = codex_container_actor.runtime_identity(self.wrapper)
        if self.runtime_identity is not None and current != self.runtime_identity:
            raise DiscoveryControllerError("sealed Codex planner runtime identity changed")
        if (self.actor_launcher_sha256 is not None
                and hashlib.sha256(Path(codex_container_actor.__file__).resolve().read_bytes()).hexdigest()
                != self.actor_launcher_sha256):
            raise DiscoveryControllerError("sealed Codex planner launcher/argv policy changed")
        return current
    def attest(self) -> Mapping[str, Any]: return {**SOL, "runtime": self._runtime()}
    def plan(self, *, context: Mapping[str, Any], workspace: Path,
             checkpoint_path: Path | None = None) -> PlannedCandidate:
        return self._plan(context=context, workspace=workspace, resume=False,
                          checkpoint_path=checkpoint_path)

    def resume_plan(self, *, context: Mapping[str, Any],
                    workspace: Path, checkpoint_path: Path) -> PlannedCandidate:
        return self._plan(context=context, workspace=workspace, resume=True,
                          checkpoint_path=checkpoint_path)

    def _plan(self, *, context: Mapping[str, Any], workspace: Path,
              resume: bool, checkpoint_path: Path | None) -> PlannedCandidate:
        # The model gets a bounded source/profile brief plus a machine contract;
        # it never receives authority to select a campaign, base, executable,
        # argv, profile parser, or evidence regex.
        if resume and self.reviewed_sources is not None:
            self.reviewed_sources.revalidate_materialized(workspace)
            source_package = self.reviewed_sources.manifest()
        else:
            source_package = (None if self.reviewed_sources is None
                              else self.reviewed_sources.materialize(workspace))
        planner_context = context.get("planner_context")
        if (self.reviewed_sources is None or not isinstance(planner_context, Mapping)
                or planner_context.get("reviewed_source_package_sha256")
                != self.reviewed_sources.package_sha256):
            raise DiscoveryControllerError(
                "planner lacks the exact reviewed source preimage authority")
        contract = {
            "plan_json_keys": ["hypothesis_id", "statement", "falsifier", "regime",
                               "proposal", "source_manifest_path", "experiment_intent"],
            "experiment_intent_keys": ["template_id", "target_surface", "target_symbol",
                                       "correctness_id", "dispatch_id", "expected_dispatch",
                                       "load_mode_recommendation"],
            "load_mode_recommendation_keys": ["mode", "rationale", "example_ids"],
            "load_mode_recommendation_semantics": (
                "optional advisory only; it may request safer serialization but cannot author "
                "telemetry, profile facts, bytes, commands, or resource authority"),
            "load_mode_recommendation_modes": ["cold_overlap", "cold_serialized",
                                                "hot_resident"],
            "load_mode_example_ids": sorted({
                str(example.get("id")) for example in
                context.get("admission_policy", {}).get("examples", [])
                if isinstance(example, Mapping) and isinstance(example.get("id"), str)}),
            "expected_dispatch": "array of 1..8 exact objects",
            "expected_dispatch_item_keys": ["route_id", "kernel_name", "calls", "grid", "workgroup", "lds_bytes"],
            "source_manifest_schema": {
                "exact_keys": ["schema", "campaign_id", "proposal_id", "candidate_id",
                    "source_tree", "production_base_commit", "instrument_commit",
                    "change_class", "declared_files", "declared_symbols", "mechanism_id",
                    "patch_sha256", "patch_encoding", "patch_base64"],
                "constants": {"schema": source_candidate.SCHEMA_SOURCE_PATCH,
                              "source_tree": "llama.cpp", "patch_encoding": "base64"},
                "patch_rule": "patch_base64 is strict base64 of a complete UTF-8 unified diff; patch_sha256 hashes the decoded bytes",
                "unified_diff_hunk_rule": (
                    "Every @@ hunk header must contain exact old/new line counts matching its "
                    "body and must end with the reviewed enclosing function symbol from "
                    "declared_symbols for that file. Blank hunk context, a preceding function, "
                    "or a following function is invalid. Before exit, decode patch_base64, "
                    "recount every hunk, and recompute patch_sha256."),
            },
            "proposal_schema": {
                "exact_keys": ["proposal_id", "change_class", "change"],
                "change_exact_keys": ["files_and_symbols", "estimated_diff_size"],
                "files_and_symbols_rule": "sorted file:symbol declarations exactly equal source manifest declarations",
                "estimated_diff_size_rule": (
                    "integer ceiling for actual changed lines in the decoded unified diff; "
                    "actual changed lines are added lines plus removed lines across every hunk"),
            },
            "proposal_requirements": ["proposal_id matches manifest", "change_class matches manifest",
                                       "change.files_and_symbols exactly matches manifest declarations",
                                       "change.estimated_diff_size is positive and is not less than the decoded patch's actual changed-line count"],
            "forbidden": ["commands", "argv", "environment", "measurement results",
                          "campaign/base/instrument selection", "unbounded source reads"],
        }
        assignment = context.get("authoring_assignment")
        if not isinstance(assignment, Mapping):
            raise DiscoveryControllerError("planner context lacks controller-owned authoring assignment")
        binding = assignment.get("portfolio_binding")
        if binding is not None:
            AuthoringAssignment(**assignment)
            example_file = binding["target_file"]
            example_symbol = binding["target_symbols"][0]
            example_symbols = list(binding["target_symbols"])
            example_hypothesis = binding["hypothesis_id"]
            example_statement = binding["statement"]
            example_falsifier = binding["falsifier"]
            example_regime = binding["regime"]
            example_template = binding["template_id"]
            example_mechanism = binding["mechanism_id"]
            example_change_class = binding["change_class"]
            example_dispatch = list(binding["expected_dispatch"])
        else:
            example_file = "ggml/src/ggml-cuda/example.cu"
            example_symbol = "example_symbol"
            example_symbols = [example_symbol]
            example_hypothesis = "akh-example"
            example_statement = "bounded hypothesis"
            example_falsifier = "an exact non-improvement falsifies it"
            example_regime = {"phase": "decode"}
            example_template = "replace-with-reviewed-id"
            example_mechanism = "bounded-example"
            example_change_class = "dispatcher"
            example_dispatch = [{"kernel_name": "exact rocprof demangled literal",
                                 "route_id": "replace-with-reviewed-id.anchor.0",
                                 "calls": 1, "grid": 64,
                                 "workgroup": 64, "lds_bytes": 0}]
        example_patch = (f"diff --git a/{example_file} b/{example_file}\n"
                         f"--- a/{example_file}\n+++ b/{example_file}\n"
                         f"@@ -1 +1 @@ {example_symbol}()\n-old\n+new\n")
        example = {
            "plan.json": {"hypothesis_id": example_hypothesis,
                "statement": example_statement,
                "falsifier": example_falsifier, "regime": example_regime,
                "proposal": {"proposal_id": assignment["proposal_id"], "change_class": example_change_class,
                    "change": {"files_and_symbols": [
                                   f"{example_file}:{symbol}"
                                   for symbol in example_symbols],
                               "estimated_diff_size": 2}},
                "source_manifest_path": "source-patch.json",
                "experiment_intent": {"template_id": example_template,
                    "target_surface": "gpu_decode", "target_symbol": example_symbol,
                    "correctness_id": "backend-ops-hip-v1",
                    "dispatch_id": "decode-tg128-rocprof-v3",
                    "expected_dispatch": example_dispatch}},
            "source-patch.json": {"schema": source_candidate.SCHEMA_SOURCE_PATCH,
                "campaign_id": assignment["campaign_id"], "proposal_id": assignment["proposal_id"],
                "candidate_id": assignment["candidate_id"], "source_tree": "llama.cpp",
                "production_base_commit": assignment["production_base_commit"],
                "instrument_commit": assignment["instrument_commit"], "change_class": example_change_class,
                "declared_files": [example_file],
                "declared_symbols": {example_file: example_symbols},
                "mechanism_id": example_mechanism,
                "patch_sha256": hashlib.sha256(example_patch.encode()).hexdigest(),
                "patch_encoding": "base64",
                "patch_base64": base64.b64encode(example_patch.encode()).decode("ascii")}}
        prompt = json.dumps({"role": SOL, "context": context,
                             "experiment_template_catalog": self.template_catalog,
                             "reviewed_source_package": source_package,
                             "authoring_contract": contract,
                             "controller_owned_portfolio_binding": binding,
                             "structural_example_only": example,
                             "output": "Write plan.json and source-patch.json in workspace."}, sort_keys=True)
        self._runtime()
        if not resume:
            _emit_observational_telemetry(
                self.telemetry,
                "planner", "planner_started",
                campaign_id=assignment["campaign_id"],
                hypothesis_id=example_hypothesis, provider=SOL["provider"],
                model=SOL["model"], effort=SOL["effort"])
        if resume:
            if checkpoint_path is None:
                raise DiscoveryControllerError(
                    "planner resume lacks its controller checkpoint path")
            checkpoint = _reopen_planner_actor_checkpoint(
                workspace, checkpoint_path, context=context)
            result_facts = dict(checkpoint["result"])
        else:
            try:
                result = codex_container_actor.run_actor(wrapper=self.wrapper, workspace=workspace, model=SOL["model"], effort=SOL["effort"], prompt=prompt, environment=self.environment,
                    expected_wrapper_sha256=self.wrapper_sha256,
                    expected_runtime_identity=self.runtime_identity,
                    expected_launcher_sha256=self.actor_launcher_sha256)
            except Exception:
                _emit_observational_telemetry(
                        self.telemetry,
                        "planner", "planner_failed",
                        campaign_id=assignment["campaign_id"],
                        hypothesis_id=example_hypothesis, provider=SOL["provider"],
                        model=SOL["model"], effort=SOL["effort"])
                raise
            result_facts = {
                "returncode": result.returncode,
                "stdout_sha256": hashlib.sha256(
                    getattr(result, "stdout", "").encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    getattr(result, "stderr", "").encode()).hexdigest(),
            }
            if checkpoint_path is not None:
                _seal_planner_actor_checkpoint(
                    workspace, checkpoint_path, context=context,
                    result=result_facts)
            if result.returncode:
                _emit_observational_telemetry(
                        self.telemetry,
                        "planner", "planner_failed",
                        campaign_id=assignment["campaign_id"],
                        hypothesis_id=example_hypothesis, provider=SOL["provider"],
                        model=SOL["model"], effort=SOL["effort"], result=result_facts)
                raise PlannerProviderTransient(
                    f"Sol actor failed: {getattr(result, 'stderr', '')[-400:]}")
        if result_facts["returncode"]:
            raise PlannerProviderTransient(
                f"sealed Sol actor invocation failed with return code "
                f"{result_facts['returncode']}")
        if self.reviewed_sources is not None:
            self.reviewed_sources.revalidate_materialized(workspace)
        try:
            candidate = _load_plan(
                workspace / "plan.json", workspace,
                assignment=AuthoringAssignment(**assignment))
        except PlannerOutputRefusal as exc:
            telemetry_exc = _emit_observational_telemetry(
                self.telemetry, "planner", "planner_refused",
                campaign_id=assignment["campaign_id"],
                hypothesis_id=example_hypothesis,
                provider=SOL["provider"], model=SOL["model"],
                effort=SOL["effort"], result={
                    **result_facts,
                    "refusal_type": "planner_output_refusal",
                    "refusal_reason_sha256": hashlib.sha256(
                        str(exc).encode()).hexdigest(),
                })
            if self.telemetry is not None and telemetry_exc is None:
                exc.telemetry_status = "emitted"
            elif telemetry_exc is not None:
                exc.note_telemetry_failure(telemetry_exc)
            raise
        _emit_observational_telemetry(
                self.telemetry,
                "planner", "planner_completed",
                campaign_id=assignment["campaign_id"],
                hypothesis_id=example_hypothesis, provider=SOL["provider"],
                model=SOL["model"], effort=SOL["effort"], result=result_facts)
        return candidate


class ClaudeCritic:
    """Concrete Fable 5 critic. It can bind a veto but never alters the candidate."""
    def __init__(self, *, wrapper: Path, environment: Mapping[str, str],
                 template_catalog: Mapping[str, Any] | None = None,
                 reviewed_sources: ReviewedSourcePackage | None = None,
                 wrapper_sha256: str | None = None,
                 runtime_identity: Mapping[str, Any] | None = None,
                 actor_launcher_sha256: str | None = None,
                 telemetry: discovery_telemetry.DiscoveryTelemetry | None = None,
                 auth_root: Path = Path("/home/node/.claude")) -> None:
        self.wrapper, self.environment = wrapper, dict(environment)
        self.template_catalog = json.loads(json.dumps(template_catalog or {}, sort_keys=True))
        self.reviewed_sources = reviewed_sources
        self.wrapper_sha256 = wrapper_sha256
        self.runtime_identity = None if runtime_identity is None else dict(runtime_identity)
        self.actor_launcher_sha256 = actor_launcher_sha256
        self.telemetry = telemetry
        self.auth_root = auth_root
    def _runtime(self) -> Mapping[str, Any]:
        if self.wrapper_sha256 is not None:
            if self.wrapper.is_symlink() or not self.wrapper.is_file() or hashlib.sha256(self.wrapper.read_bytes()).hexdigest() != self.wrapper_sha256:
                raise DiscoveryControllerError("sealed Claude critic wrapper bytes changed")
        current = claude_fable5_critic_actor.runtime_identity(self.wrapper)
        if self.runtime_identity is not None and current != self.runtime_identity:
            raise DiscoveryControllerError("sealed Claude critic runtime identity changed")
        if (self.actor_launcher_sha256 is not None
                and hashlib.sha256(Path(claude_fable5_critic_actor.__file__).resolve().read_bytes()).hexdigest()
                != self.actor_launcher_sha256):
            raise DiscoveryControllerError("sealed Claude critic launcher/argv policy changed")
        return current
    def attest(self) -> Mapping[str, Any]: return {**FABLE5_CRITIC, "runtime": self._runtime()}
    def review(self, candidate: PlannedCandidate, *, context: Mapping[str, Any], workspace: Path) -> Critique:
        manifest = candidate.source_manifest
        if len(manifest.patch_text.encode("utf-8")) > 65536:
            raise DiscoveryControllerError("candidate patch exceeds bounded critic visibility")
        source_context = None
        if self.reviewed_sources is not None:
            if len(manifest.declared_files) != 1:
                raise DiscoveryControllerError("critic requires one exact reviewed source preimage")
            relative = manifest.declared_files[0]
            source_context = self.reviewed_sources.critic_context(
                relative, manifest.declared_symbols[relative])
        critic_context = {**context, "selected_source_preimage": source_context}
        bindings = {
            "proposal_sha256": _sha(candidate.proposal),
            "source_manifest_sha256": candidate.source_manifest_sha256,
            "candidate_patch_sha256": manifest.patch_sha256,
            "context_sha256": _sha(critic_context),
            "template_catalog_sha256": _sha(self.template_catalog),
        }
        prompt = json.dumps({"role": FABLE5_CRITIC, "context": critic_context,
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
            "required_output_bindings": bindings,
            "output": "Return only the strict structured critique; do not edit files or use tools."}, sort_keys=True)
        self._runtime()
        campaign_id = manifest.campaign_id
        _emit_observational_telemetry(
                self.telemetry,
                "autokernel", "critic_started", campaign_id=campaign_id,
                hypothesis_id=candidate.hypothesis_id,
                provider=FABLE5_CRITIC["provider"], model=FABLE5_CRITIC["model"],
                effort=FABLE5_CRITIC["effort"])
        try:
            result = claude_fable5_critic_actor.run_critic(
                wrapper=self.wrapper, workspace=workspace, prompt=prompt,
                bindings=bindings, environment=self.environment,
                auth_root=self.auth_root,
                expected_wrapper_sha256=self.wrapper_sha256,
                expected_runtime_identity=self.runtime_identity,
                expected_launcher_sha256=self.actor_launcher_sha256)
        except Exception:
            _emit_observational_telemetry(
                    self.telemetry,
                    "autokernel", "critic_failed", campaign_id=campaign_id,
                    hypothesis_id=candidate.hypothesis_id,
                    provider=FABLE5_CRITIC["provider"], model=FABLE5_CRITIC["model"],
                    effort=FABLE5_CRITIC["effort"])
            raise
        if self.telemetry is not None:
            _emit_observational_telemetry(
                    self.telemetry,
                    "autokernel", "critic_completed", campaign_id=campaign_id,
                    hypothesis_id=candidate.hypothesis_id,
                    provider=FABLE5_CRITIC["provider"], model=FABLE5_CRITIC["model"],
                    effort=FABLE5_CRITIC["effort"], result={
                        "stdout_sha256": result.stdout_sha256,
                        "stderr_sha256": result.stderr_sha256,
                        "decision": result.decision,
                    })
        return Critique(result.decision, result.reason)


def _read_object(path: Path, root: Path) -> dict[str, Any]:
    try: path.resolve().relative_to(root.resolve())
    except ValueError as exc: raise DiscoveryControllerError("actor artifact escaped workspace") from exc
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise DiscoveryControllerError(f"invalid actor artifact {path.name}") from exc
    if not isinstance(value, dict): raise DiscoveryControllerError("actor artifact must be object")
    return value


def _read_planner_object(path: Path, root: Path) -> dict[str, Any]:
    """Read planner JSON while keeping containment violations terminal."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DiscoveryControllerError("actor artifact escaped workspace") from exc
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerOutputRefusal(f"invalid actor artifact {path.name}") from exc
    if not isinstance(value, dict):
        raise PlannerOutputRefusal("actor artifact must be object")
    return value


_RETRYABLE_PLANNER_SOURCE_ERRORS = (
    "source patch manifest is not strict JSON:",
    "source patch manifest fields must be exactly",
    "source patch manifest schema/encoding is unsupported",
    "patch_base64 is invalid:",
    "patch_sha256 does not match the embedded patch bytes",
    "patch is not strict UTF-8:",
    "patch contains NUL bytes",
    "patch is not an accounted unified diff:",
    "hunk appears before a diff --git header",
    "source patch contains no accounted hunk",
    "patch bytes must end in a newline",
)


def _load_planner_manifest(path: Path) -> source_candidate.SourcePatchManifest:
    """Classify only syntactic carrier/patch defects as retryable output.

    Identity, path/symbol scope, change-class, reward-integrity, and instrument
    policy errors intentionally retain ``SourceCandidateError`` and terminate
    fail closed.
    """
    try:
        return source_candidate.load_source_patch_manifest(path)
    except source_candidate.SourceCandidateError as exc:
        reason = str(exc)
        if any(reason.startswith(prefix) for prefix in
               _RETRYABLE_PLANNER_SOURCE_ERRORS):
            raise PlannerOutputRefusal(
                f"SourceCandidateError: {reason}") from exc
        raise


def _load_plan(path: Path, root: Path, *, assignment: AuthoringAssignment | None = None) -> PlannedCandidate:
    value = _read_planner_object(path, root)
    allowed = {"hypothesis_id", "statement", "falsifier", "regime", "proposal", "source_manifest_path", "experiment_intent"}
    if set(value) not in (allowed, allowed - {"experiment_intent"}): raise PlannerOutputRefusal("planner output schema mismatch")
    intent_raw = value.pop("experiment_intent", None)
    if intent_raw is not None:
        allowed_intent = {"template_id", "target_surface", "target_symbol", "correctness_id", "dispatch_id", "expected_dispatch", "load_mode_recommendation"}
        if not isinstance(intent_raw, Mapping) or set(intent_raw) not in (allowed_intent, allowed_intent - {"load_mode_recommendation"}):
            raise PlannerOutputRefusal("planner experiment intent schema mismatch")
        expected = intent_raw["expected_dispatch"]
        expected_keys = {"route_id", "kernel_name", "calls", "grid", "workgroup", "lds_bytes"}
        if (not isinstance(expected, list) or not 1 <= len(expected) <= 8
                or not all(isinstance(row, Mapping) and set(row) == expected_keys
                           for row in expected)):
            raise PlannerOutputRefusal("planner bounded dispatch schema mismatch")
        recommendation = intent_raw.get("load_mode_recommendation")
        if recommendation is not None:
            if not isinstance(recommendation, Mapping) or set(recommendation) != {"mode", "rationale", "example_ids"}:
                raise PlannerOutputRefusal("planner load-mode recommendation schema mismatch")
            recommendation = LoadModeRecommendation(
                mode=recommendation["mode"], rationale=recommendation["rationale"],
                example_ids=tuple(recommendation["example_ids"]))
        intent = GpuSourceExperimentIntent(**{**intent_raw,
            "expected_dispatch": tuple(BoundedDispatchExpectation(**row) for row in expected),
            "load_mode_recommendation": recommendation})
    else:
        intent = None
    raw_path = Path(_text(value.pop("source_manifest_path"), "source_manifest_path"))
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise DiscoveryControllerError("source manifest path must be a workspace-relative path")
    manifest_path = root / raw_path
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise PlannerOutputRefusal(
            f"invalid actor artifact {manifest_path.name}") from exc
    try:
        resolved_manifest.relative_to(root.resolve())
    except ValueError as exc:
        raise DiscoveryControllerError(
            "source manifest escaped disposable workspace") from exc
    manifest = _load_planner_manifest(resolved_manifest)
    if assignment is not None:
        if (manifest.campaign_id, manifest.proposal_id, manifest.candidate_id,
                manifest.production_base_commit, manifest.instrument_commit) != (
                    assignment.campaign_id, assignment.proposal_id, assignment.candidate_id,
                    assignment.production_base_commit, assignment.instrument_commit):
            raise DiscoveryControllerError("actor attempted to invent campaign/base/instrument identity")
        if value.get("proposal", {}).get("proposal_id") != assignment.proposal_id:
            raise DiscoveryControllerError("actor proposal does not use controller-assigned proposal identity")
        # Bind the actor's proposal to controller-owned identity and the exact
        # manifest file:symbol scope before the critic or any claim can run.
        # The actor may not omit, regroup, or reformat these declarations.
        manifest.bind(
            proposal=value.get("proposal", {}),
            campaign_id=assignment.campaign_id,
            candidate_id=assignment.candidate_id,
            production_base_commit=assignment.production_base_commit,
            instrument_commit=assignment.instrument_commit)
        proposal = value.get("proposal")
        change = proposal.get("change") if isinstance(proposal, Mapping) else None
        estimated = change.get("estimated_diff_size") if isinstance(change, Mapping) else None
        if isinstance(estimated, bool) or not isinstance(estimated, int) or estimated < 1:
            raise PlannerOutputRefusal("planner estimated_diff_size must be a positive integer")
        try:
            actual_changed_lines = integrity.parse_unified_diff(
                manifest.patch_bytes.decode("utf-8")).total_changed
        except (UnicodeDecodeError, integrity.DiffParseError) as exc:
            raise PlannerOutputRefusal(
                "planner patch cannot be counted as a complete UTF-8 unified diff") from exc
        if estimated < actual_changed_lines:
            raise PlannerOutputRefusal(
                "planner estimated_diff_size is smaller than the decoded patch's actual "
                f"changed-line count ({estimated} < {actual_changed_lines})")
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
    anchor_correctness_binary: Path | None = None
    anchor_correctness_binary_sha256: str | None = None
    candidate_correctness_binary: Path | None = None
    candidate_correctness_binary_sha256: str | None = None
    anchor_correctness_capability_receipt: Path | None = None
    anchor_correctness_capability_sha256: str | None = None
    candidate_correctness_capability_receipt: Path | None = None
    candidate_correctness_capability_sha256: str | None = None
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
                                      (self.anchor_correctness_binary, self.anchor_correctness_binary_sha256, "anchor correctness binary"),
                                      (self.candidate_correctness_binary, self.candidate_correctness_binary_sha256, "candidate correctness binary"),
                                      (self.anchor_correctness_capability_receipt, self.anchor_correctness_capability_sha256, "anchor correctness capability"),
                                      (self.candidate_correctness_capability_receipt, self.candidate_correctness_capability_sha256, "candidate correctness capability"),
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
                 args_factory: Callable[[PlannedCandidate, GpuSourceBuild, Mapping[str, Any]], Any],
                 runner_attest: Callable[[], None] = lambda: None) -> None:
        self.build_source, self.proof_bundle, self.args_factory = build_source, proof_bundle, args_factory
        self.runner_attest = runner_attest

    def screen(self, candidate: PlannedCandidate, authorization: hypotheses.ClaimAuthorization, lease: Mapping[str, Any]) -> SealedScreen:
        try:
            build = self.build_source(candidate, authorization, lease)
        except source_candidate.SourceCandidateError as exc:
            # Source materialization re-derives the committed diff after the
            # critic's review.  A mismatch here is an authoring rejection, not
            # an ambiguous GPU operation: proof production, reservation, and
            # the throughput runner are all strictly downstream of this call.
            # Preserve that ordering as a typed precompute refusal so the
            # controller durably records the failed iteration and advances.
            raise PrecomputeScreenRefusal(
                f"source candidate authoring rejected: {type(exc).__name__}: {exc}"
            ) from exc
        bundle = self.proof_bundle(candidate, build)
        if not isinstance(bundle, gpu_source_proofs.GpuSourceProofBundle):
            raise DiscoveryControllerError("GPU source gate did not return a validated proof bundle")
        if bundle.manifest_sha256 != candidate.source_manifest_sha256:
            raise DiscoveryControllerError("GPU proof bundle does not bind the candidate manifest")
        if bundle.candidate != build.candidate_identity or bundle.anchor != build.anchor_identity:
            raise DiscoveryControllerError("GPU proof bundle does not bind both sealed build identities")
        args = self.args_factory(candidate, build, lease)
        target_args = getattr(args, "_target_runtime_args", None)
        if target_args is None:
            raise DiscoveryControllerError(
                "GPU source runner lacks a separate target-runtime stage")
        # The established runner owns KFD/VRAM, device claims, paired samples,
        # and its durable result.  This controller does not spawn a shell.
        if any(getattr(current, "factor", None) != "source_patch"
               or Path(getattr(current, "anchor_build", "")).resolve()
               != build.anchor_build
               or Path(getattr(current, "candidate_build", "")).resolve()
               != build.candidate_build
               for current in (args, target_args)):
            raise DiscoveryControllerError("GPU source runner arguments are not bound to the typed build")
        attribution_body = bundle.attribution.get("body")
        comparison = (attribution_body.get("exact_duration_comparison")
                      if isinstance(attribution_body, Mapping) else None)
        if not isinstance(comparison, Mapping):
            raise DiscoveryControllerError(
                "GPU source proof lacks exact-duration decision evidence")
        exact_effect = float(comparison["relative_improvement_fraction"])
        if exact_effect <= 0:
            # A valid neutral/regressed exact-route measurement is a scientific
            # outcome, not a refusal.  It terminates before any whole-model
            # benchmark and therefore cannot consume a target-runtime call.
            result_path = (Path(args.output_dir).resolve().parent /
                           "exact-attribution-outcome.json")
            body = {
                "schema": "epyc.autokernel.exact_attribution_outcome.v1",
                "authority": "nonpromotable_candidate_only_discovery",
                "non_promotable": True, "promotion_claim": False,
                "status": "complete", "classification": "screened_out",
                "manifest_sha256": candidate.source_manifest_sha256,
                "exact_attribution_effect_fraction": exact_effect,
                "target_runtime_executed": False,
                "target_runtime_reason": "nonpositive_exact_duration",
                "dispatch_proof_sha256": bundle.attribution["file_sha256"],
            }
            body["result_sha256"] = schemas.content_hash(body)
            if result_path.exists() or result_path.is_symlink():
                if result_path.is_symlink() or json.loads(
                        result_path.read_text(encoding="utf-8")) != body:
                    raise DiscoveryControllerError(
                        "exact-attribution outcome receipt changed")
            else:
                _atomic(result_path, body)
            return SealedScreen(
                receipt_path=str(result_path),
                result_sha256=body["result_sha256"],
                effect_fraction=exact_effect, classification="screened_out",
                baseline_sha256=schemas.content_hash(
                    comparison.get("anchor_routes", comparison)),
                source_proof_sha256=bundle.correctness["file_sha256"],
                dispatch_proof_sha256=bundle.attribution["file_sha256"],
                exact_attribution_effect_fraction=exact_effect,
                target_runtime_effect_fraction=None,
                stages=("materialized", "built", "correctness", "attribution"))
        def run_stage(current: Any, *, graph_mode: str) -> tuple[Path, Mapping[str, Any]]:
            # Immediately-before-call byte attestation prevents a validated
            # graph from silently executing changed controller/runner bytes.
            self.runner_attest()
            result_path = Path(current.output_dir).resolve() / "result.json"
            if result_path.exists() and not result_path.is_symlink():
                raw = gpu_source_proofs.load_receipt(
                    result_path,
                    schema="epyc.autokernel.gpu_candidate_only_screen.v2")["body"]
            else:
                raw = gpu_discovery.run(current)
            raw = gpu_source_proofs.require_result_file(result_path, raw)["body"]
            if not (raw.get("schema") == "epyc.autokernel.gpu_candidate_only_screen.v2"
                    and raw.get("non_promotable") is True
                    and raw.get("promotion_claim") is False
                    and raw.get("hip_residency_proved") is True
                    and raw.get("runtime_graphs") == graph_mode):
                raise DiscoveryControllerError(
                    f"GPU {graph_mode} runner returned an unsealed/non-resident result")
            if (hasattr(current, "_device_claim_acquirer")
                    and raw.get("device_claim_mode") != "borrowed_outer_reservation"):
                raise DiscoveryControllerError(
                    "GPU runner did not bind throughput to the borrowed outer reservation")
            if not hasattr(current, "_device_claim_acquirer"):
                return result_path, raw
            expected_outer = getattr(current, "_expected_outer_claim_id", None)
            opened = raw.get("device_claim_open")
            phase_end = raw.get("device_claim_borrowed_phase_end")
            if (not isinstance(expected_outer, str)
                    or not isinstance(opened, Mapping)
                    or opened.get("claim_id") != expected_outer
                    or not isinstance(phase_end, Mapping)
                    or phase_end.get("schema") !=
                    "epyc.autokernel.borrowed_device_claim_phase.v1"
                    or phase_end.get("outer_claim_id") != expected_outer
                    or phase_end.get("physical_release") is not False
                    or "released_at" in phase_end
                    or raw.get("device_claim_released") is not None):
                raise DiscoveryControllerError(
                    "GPU runner borrowed phase does not bind the exact outer claim")
            governance_path = Path(current.output_dir).resolve() / "live-governance.json"
            try:
                governance = json.loads(governance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DiscoveryControllerError(
                    "GPU runner lacks terminal borrowed-phase governance") from exc
            if (not isinstance(governance, Mapping)
                    or governance.get("status") != "borrowed_phase_ended"
                    or governance.get("device_claim_mode") != "borrowed_outer_reservation"
                    or governance.get("device_claim_open") != opened
                    or governance.get("device_claim_borrowed_phase_end") != phase_end
                    or governance.get("device_claim_released") is not None):
                raise DiscoveryControllerError(
                    "GPU runner terminal governance differs from its borrowed phase")
            return result_path, raw

        _graphs_off_path, _graphs_off = run_stage(args, graph_mode="off")
        result_path, raw = run_stage(target_args, graph_mode="on")
        projection = autokernel_progression._gpu_screen(result_path, raw)
        if projection is None: raise DiscoveryControllerError("GPU result failed canonical progression validation")
        target_effect = float(raw["median_relative"])
        return SealedScreen(receipt_path=str(result_path), result_sha256=str(raw["result_sha256"]), effect_fraction=target_effect, classification=str(projection["stage"]), baseline_sha256=str(raw["baseline_sha256"]), source_proof_sha256=bundle.correctness["file_sha256"], dispatch_proof_sha256=bundle.attribution["file_sha256"], exact_attribution_effect_fraction=exact_effect, target_runtime_effect_fraction=target_effect, stages=("materialized", "built", "correctness", "attribution", "measurement_graphs_off_screen", "target_runtime_graphs_on_screen"))


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
    hypothesis_portfolio: hypothesis_portfolio.Portfolio | None = None
    hypothesis_portfolio_sha256: str | None = None
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
        if ((self.hypothesis_portfolio is None) !=
                (self.hypothesis_portfolio_sha256 is None)
                or self.hypothesis_portfolio_sha256 is not None
                and not HASH.fullmatch(self.hypothesis_portfolio_sha256)):
            raise DiscoveryControllerError("invalid sealed hypothesis portfolio authority")
        if (self.hypothesis_portfolio is not None
                and (not isinstance(self.hypothesis_portfolio, hypothesis_portfolio.Portfolio)
                     or self.hypothesis_portfolio.sha256 != self.hypothesis_portfolio_sha256)):
            raise DiscoveryControllerError(
                "controller portfolio must be one loader-validated immutable authority")
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


def _ensure_question(tracker: hypotheses.HypothesisTracker, item: PlannedCandidate,
                     portfolio_binding: Mapping[str, Any] | None = None) -> None:
    """Open the exact question whose campaign-ledger DNR gate will authorize it.

    Legacy/generic callers retain their original regime verbatim: an old question
    which did not declare a structural mechanism must continue to read
    ``COULD_NOT_CHECK`` rather than acquiring authority retroactively.  A sealed
    portfolio candidate is different.  The controller already owns its exact
    manifest mechanism, so omitting that key would make every new AutoKernel
    authorization structurally incomparable to the campaign ledger.  On this path the
    mechanism is mandatory, controller-derived, and any actor-authored disagreement is
    refused rather than silently overwritten.
    """
    regime = dict(item.regime)
    if portfolio_binding is not None:
        mechanism = item.source_manifest.mechanism_id
        if (not isinstance(mechanism, str) or not HASH.fullmatch(mechanism)
                or mechanism != portfolio_binding.get("mechanism_id")):
            raise DiscoveryControllerError(
                "portfolio candidate lacks its controller-owned structural mechanism")
        declared = regime.get("mechanism")
        if declared is not None and declared != mechanism:
            raise DiscoveryControllerError(
                "portfolio candidate regime disagrees with its controller-owned mechanism")
        regime["mechanism"] = mechanism
    question=hypotheses.Hypothesis(hypothesis_id=item.hypothesis_id, statement=item.statement, falsifier=item.falsifier, origin=hypotheses.ORIGIN_PLANNER, author="gpt-5.6-sol", regime=regime, source={"manifest_sha256":item.source_manifest_sha256})
    try: tracker.open_hypothesis(question)
    except hypotheses.HypothesisAlreadyTracked: pass

def _record_attempt_once(tracker: hypotheses.HypothesisTracker, item: PlannedCandidate, proposal_id: str, result: SealedScreen) -> None:
    ref=f"sha256:{result.result_sha256}"
    for event in tracker.read().events:
        attempt=event.payload.get("attempt") if event.kind==hypotheses.EVENT_ATTEMPTED else None
        if isinstance(attempt,Mapping) and attempt.get("hypothesis_id")==item.hypothesis_id and ref in attempt.get("refs",[]): return
    tracker.note_attempt(item.hypothesis_id,proposal_id=proposal_id,disposition=result.classification,bears_on_falsifier=True,note=f"sealed screen {result.result_sha256}; effect={result.effect_fraction:.9g}",refs=(ref,))


def _portfolio_binding(config: ControllerConfig,
                       record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one eligible scientific question into exact actor authority."""
    if config.hypothesis_portfolio is None or config.hypothesis_portfolio_sha256 is None:
        raise DiscoveryControllerError("controller lacks a sealed hypothesis portfolio")
    target = record.get("target")
    eligibility = record.get("current_bundle_eligibility")
    mechanism = record.get("mechanism")
    policy = record.get("decision_policy")
    falsifiers = record.get("falsifiers")
    if (not isinstance(target, Mapping) or not isinstance(eligibility, Mapping)
            or eligibility.get("eligible") is not True
            or not isinstance(mechanism, Mapping)
            or not isinstance(policy, Mapping)
            or not isinstance(falsifiers, (list, tuple)) or not falsifiers
            or record.get("primary_falsifier") not in falsifiers
            or not isinstance(record.get("regime"), Mapping)):
        raise DiscoveryControllerError("eligible portfolio record is incomplete")
    files = target.get("source_files")
    symbols = target.get("source_symbols")
    templates = eligibility.get("template_ids")
    policy_keys = {"metric", "frame_id", "effect_unit", "continuation_floor_pct",
                   "nomination_floor_pct", "min_replication_effect_pct",
                   "required_replications", "max_replication_spread_pct",
                   "sign_policy", "conflict_policy", "max_distinct_candidates",
                   "terminal_rule"}
    facets = mechanism.get("facets") if isinstance(mechanism, Mapping) else None
    if (not isinstance(files, (list, tuple)) or len(files) != 1
            or not isinstance(symbols, (list, tuple)) or not symbols
            or not all(isinstance(value, str) and value for value in files + symbols)
            or not isinstance(templates, (list, tuple)) or len(templates) != 1
            or target.get("template_intent") != templates[0]
            or not HASH.fullmatch(str(mechanism.get("fingerprint_sha256")))
            or not isinstance(facets, Mapping)
            or facets.get("change_class") not in schemas.CHANGE_CLASSES
            or facets.get("change_class") == "parameter"
            or not isinstance(policy.get("max_distinct_candidates"), int)
            or isinstance(policy.get("max_distinct_candidates"), bool)
            or not 1 <= policy["max_distinct_candidates"] <= 8
            or set(policy) != policy_keys
            or policy.get("effect_unit") != "relative_percent"
            or policy.get("required_replications") != 2
            or policy.get("sign_policy") != "all_positive"
            or policy.get("conflict_policy") not in {"retire", "retain_inconclusive"}
            or policy.get("terminal_rule") not in {"retire", "retain_inconclusive",
                                                    "needs_review"}):
        raise DiscoveryControllerError(
            "eligible portfolio record is not expressible by one exact reviewed template")
    binding = {
        "portfolio_sha256": config.hypothesis_portfolio_sha256,
        "record_sha256": hypothesis_portfolio.content_sha256(record),
        "hypothesis_id": record.get("hypothesis_id"),
        "statement": record.get("statement"),
        "falsifier": record["primary_falsifier"],
        "mechanism_id": mechanism["fingerprint_sha256"],
        "change_class": facets["change_class"],
        "regime": dict(record["regime"]),
        "target_file": files[0],
        "target_symbols": list(symbols),
        "template_id": templates[0],
        "decision_policy": dict(policy),
    }
    dispatch_authority = (config.planner_context or {}).get(
        "portfolio_dispatch_authority", {})
    rows = dispatch_authority.get(binding["hypothesis_id"])
    if not isinstance(rows, list):
        raise DiscoveryControllerError(
            "eligible portfolio record lacks deployed raw dispatch authority")
    binding["expected_dispatch"] = [dict(row) for row in rows]
    AuthoringAssignment(
        campaign_id="ak-portfolio-validation", proposal_id="akp-portfolio-validation",
        candidate_id="akc-portfolio-validation", production_base_commit="0" * 40,
        instrument_commit="0" * 40, portfolio_binding=binding)
    return binding


def _select_portfolio_binding(state: Mapping[str, Any],
                              config: ControllerConfig) -> dict[str, Any] | None:
    if config.hypothesis_portfolio is None:
        return None
    records = config.hypothesis_portfolio.hypotheses
    eligible = [row for row in records if isinstance(row, Mapping)
                and isinstance(row.get("current_bundle_eligibility"), Mapping)
                and row["current_bundle_eligibility"].get("eligible") is True]
    try:
        eligible.sort(key=lambda row: (int(row["priority"]["rank"]),
                                       str(row["hypothesis_id"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise DiscoveryControllerError("eligible portfolio priority is malformed") from exc
    for record in eligible:
        binding = _portfolio_binding(config, record)
        if (binding["hypothesis_id"] in state.get("portfolio_terminals", {})
                or binding["hypothesis_id"] in state.get("portfolio_skips", {})
                or binding["hypothesis_id"] in state.get(
                    "portfolio_validations", {})):
            continue
        attempts = {row.get("source_manifest_sha256") for row in state["iterations"]
                    if row.get("portfolio_hypothesis_id") == binding["hypothesis_id"]
                    and isinstance(row.get("source_manifest_sha256"), str)
                    and isinstance(row.get("result_sha256"), str)
                    and HASH.fullmatch(row["result_sha256"])
                    and isinstance(row.get("evidence"), Mapping)}
        if len(attempts) < binding["decision_policy"]["max_distinct_candidates"]:
            return binding
    return None


def _validate_portfolio_candidate(item: PlannedCandidate, binding: Mapping[str, Any],
                                  portfolio: hypothesis_portfolio.Portfolio) -> None:
    """Refuse any actor attempt to rename or expand a reviewed question."""
    if not isinstance(portfolio, hypothesis_portfolio.Portfolio):
        raise DiscoveryControllerError("portfolio candidate lacks typed portfolio authority")
    intent = item.experiment_intent
    manifest = item.source_manifest
    if (item.hypothesis_id != binding["hypothesis_id"]
            or item.statement != binding["statement"]
            or item.falsifier != binding["falsifier"]
            or dict(item.regime) != dict(binding["regime"])
            or manifest.mechanism_id != binding["mechanism_id"]
            or manifest.change_class != binding["change_class"]
            or item.proposal.get("change_class") != binding["change_class"]
            or tuple(manifest.declared_files) != (binding["target_file"],)
            or set(manifest.declared_symbols.get(binding["target_file"], ())) !=
               set(binding["target_symbols"])
            or intent is None
            or intent.template_id != binding["template_id"]
            or intent.target_symbol not in binding["target_symbols"]
            or [asdict(row) for row in intent.expected_dispatch]
               != list(binding["expected_dispatch"])):
        raise DiscoveryControllerError(
            "planner candidate differs from its controller-owned portfolio assignment")


def _portfolio_exact_dnr_check(config: ControllerConfig, item: PlannedCandidate,
                               binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical, candidate-bound receipt before critic or authorization.

    This is intentionally separate from the campaign ledger.  The portfolio is sealed
    input authority and can answer an exact mechanism/regime question directly; the
    campaign ledger is derived runtime memory and may honestly answer
    ``COULD_NOT_CHECK``.  Conflating the two outcomes makes a portfolio refusal vanish
    into a generic authorization reason on restart.
    """
    portfolio = config.hypothesis_portfolio
    semantic_sha256 = config.hypothesis_portfolio_sha256
    if (not isinstance(portfolio, hypothesis_portfolio.Portfolio)
            or not isinstance(semantic_sha256, str)
            or portfolio.sha256 != semantic_sha256):
        raise DiscoveryControllerError(
            "portfolio exact-DNR check lacks sealed semantic authority")
    mechanism_id = item.source_manifest.mechanism_id
    if (not isinstance(mechanism_id, str) or not HASH.fullmatch(mechanism_id)
            or mechanism_id != binding.get("mechanism_id")):
        raise DiscoveryControllerError(
            "portfolio exact-DNR check lacks the controller-owned candidate mechanism")
    regime = dict(item.regime)
    if regime != dict(binding.get("regime") or {}):
        raise DiscoveryControllerError(
            "portfolio exact-DNR check candidate regime differs from assignment")
    matched: list[str] = []
    for index, dnr in enumerate(portfolio.do_not_repeat):
        if not isinstance(dnr, Mapping):
            raise DiscoveryControllerError("portfolio DNR record is malformed")
        dnr_id = dnr.get("dnr_id")
        mechanism = dnr.get("mechanism")
        dnr_regime = dnr.get("regime")
        if (not isinstance(dnr_id, str) or not dnr_id.startswith("dnr-")
                or not isinstance(mechanism, Mapping)
                or not HASH.fullmatch(str(mechanism.get("fingerprint_sha256")))
                or not isinstance(dnr_regime, Mapping)):
            raise DiscoveryControllerError(
                f"portfolio DNR record {index} lacks exact mechanism/regime identity")
        if (mechanism_id == mechanism["fingerprint_sha256"]
                and regime == dict(dnr_regime)):
            matched.append(dnr_id)
    body: dict[str, Any] = {
        "schema": PORTFOLIO_DNR_CHECK_SCHEMA,
        "portfolio_semantic_sha256": semantic_sha256,
        "portfolio_hypothesis_id": binding.get("hypothesis_id"),
        "candidate_source_manifest_sha256": item.source_manifest_sha256,
        "candidate_mechanism_id": mechanism_id,
        "canonical_regime_sha256": schemas.content_hash(regime),
        "matched_dnr_ids": sorted(set(matched)),
        "outcome": schemas.FAIL if matched else schemas.PASS,
    }
    body["receipt_sha256"] = schemas.content_hash(body)
    return body


def _revalidate_portfolio_checkpoint(config: ControllerConfig,
                                     item: PlannedCandidate,
                                     row: Mapping[str, Any]) -> None:
    """Fail closed when a new portfolio checkpoint omitted or changed its DNR receipt."""
    if config.hypothesis_portfolio is None:
        # Legacy generic campaigns never had this receipt.  Their campaign-ledger
        # COULD_NOT_CHECK semantics are preserved rather than rewritten on resume.
        return
    binding = row.get("portfolio_binding")
    if not isinstance(binding, Mapping):
        raise DiscoveryControllerError(
            "portfolio pending candidate lacks controller-owned binding")
    _validate_portfolio_candidate(item, binding, config.hypothesis_portfolio)
    expected = _portfolio_exact_dnr_check(config, item, binding)
    actual = row.get("portfolio_exact_dnr_check")
    if not isinstance(actual, Mapping) or dict(actual) != expected:
        raise DiscoveryControllerError(
            "portfolio pending candidate DNR receipt is missing or changed")
    if expected["outcome"] != schemas.PASS:
        raise DiscoveryControllerError(
            "portfolio pending candidate exactly matches a sealed DNR")


def _bind_campaign_ledger_outcome(row: dict[str, Any],
                                  authorization: hypotheses.ClaimAuthorization) -> None:
    """Keep runtime-ledger disposition distinct from the sealed portfolio receipt."""
    expected = authorization.do_not_repeat_outcome
    reasons = list(authorization.do_not_repeat_reasons)
    prior = row.get("campaign_ledger_dnr_outcome")
    prior_reasons = row.get("campaign_ledger_dnr_reasons")
    if prior is not None and (prior != expected or prior_reasons != reasons):
        raise DiscoveryControllerError(
            "campaign-ledger DNR outcome differs from durable authorization")
    row["campaign_ledger_dnr_outcome"] = expected
    row["campaign_ledger_dnr_reasons"] = reasons


def _context(state: Mapping[str, Any], tracker: hypotheses.HypothesisTracker, turn: int,
             config: ControllerConfig,
             portfolio_binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    if config.production_base_commit is not None or portfolio_binding is not None:
        assignment = AuthoringAssignment(
            campaign_id=config.campaign_id, proposal_id=f"akp-discovery-{turn}",
            candidate_id=f"akc-discovery-{turn}",
            production_base_commit=config.production_base_commit or "0" * 40,
            instrument_commit=(config.instrument_commit
                               or config.production_base_commit or "0" * 40),
            portfolio_binding=portfolio_binding).to_dict()
    prior_refusals = [
        {key: row.get(key) for key in (
            "turn", "status", "reason", "portfolio_hypothesis_id",
            "context_sha256")}
        for row in state["iterations"] if row.get("status") == "planner_refused"
    ][-8:]
    return {"authority": AUTHORITY, "turn":turn, "roster":sealed_roster(),
            "planner_context": config.planner_context,
            "planner_context_sha256": config.planner_context_sha256,
            "admission_corpus_sha256": config.admission_corpus_sha256,
            "admission_corpus_version": config.admission_corpus_version,
            "deployment_identity_sha256": config.deployment_identity_sha256,
            "hypothesis_portfolio_sha256": config.hypothesis_portfolio_sha256,
            "authoring_assignment": assignment,
            "prior_authoring_refusals": prior_refusals,
            "prior_results": prior, "do_not_repeat":_memory_block(tracker,turn)}


def _pending_item(item: PlannedCandidate) -> dict[str, Any]:
    manifest = item.source_manifest
    raw_manifest=source_candidate.source_patch_manifest_bytes(manifest)
    intent = (None if item.experiment_intent is None else
              json.loads(json.dumps(asdict(item.experiment_intent),
                                    sort_keys=True)))
    return {"hypothesis_id": item.hypothesis_id, "statement": item.statement,
            "falsifier": item.falsifier, "regime": dict(item.regime),
            "proposal": dict(item.proposal), "source_manifest_sha256": item.source_manifest_sha256,
            "experiment_intent": intent,
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
    try:
        raw_bytes=base64.b64decode(raw.get("manifest_raw_base64",""),validate=True)
    except (TypeError, ValueError) as exc:
        raise DiscoveryControllerError("pending manifest carrier is invalid") from exc
    canonical_bytes=source_candidate.source_patch_manifest_bytes(manifest)
    canonical_sha256=hashlib.sha256(canonical_bytes).hexdigest()
    identities=(canonical_sha256, manifest.patch_bundle_sha256,
                raw.get("manifest_file_sha256"), raw.get("patch_bundle_sha256"),
                raw.get("source_manifest_sha256"))
    if raw_bytes != canonical_bytes or any(value != canonical_sha256 for value in identities):
        raise DiscoveryControllerError("pending manifest identity mismatch")
    intent = raw.get("experiment_intent")
    if intent is not None and not isinstance(intent, Mapping):
        raise DiscoveryControllerError("pending experiment intent is malformed")
    if intent is not None:
        expected = intent.get("expected_dispatch")
        if not isinstance(expected, list) or not expected:
            raise DiscoveryControllerError("pending bounded dispatch is malformed")
        recommendation = intent.get("load_mode_recommendation")
        if recommendation is not None:
            if not isinstance(recommendation, Mapping):
                raise DiscoveryControllerError("pending load-mode recommendation is malformed")
            recommendation = LoadModeRecommendation(
                mode=recommendation.get("mode"), rationale=recommendation.get("rationale"),
                example_ids=tuple(recommendation.get("example_ids", ())))
        intent = {**intent, "expected_dispatch": tuple(
                      BoundedDispatchExpectation(**row) for row in expected),
                  "load_mode_recommendation": recommendation}
    return PlannedCandidate(hypothesis_id=raw["hypothesis_id"],statement=raw["statement"],falsifier=raw["falsifier"],regime=raw["regime"],proposal=raw["proposal"],source_manifest=manifest,source_manifest_sha256=raw["source_manifest_sha256"],experiment_intent=GpuSourceExperimentIntent(**intent) if intent else None)


def _decision_floor(policy: Mapping[str, Any] | None, key: str,
                    fallback: float) -> float:
    if policy is None:
        return fallback
    value = policy.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0 <= float(value) <= 100):
        raise DiscoveryControllerError("portfolio decision policy has an invalid numeric floor")
    return float(value) / 100.0


def _required_replications(policy: Mapping[str, Any] | None) -> int:
    if policy is None:
        return 2
    value = policy.get("required_replications")
    if (isinstance(value, bool) or not isinstance(value, int)
            or not 2 <= value <= 8
            or policy.get("sign_policy") not in {"all_positive", "median_positive"}
            or policy.get("conflict_policy") not in {"retire", "retain_inconclusive"}):
        raise DiscoveryControllerError("portfolio replication policy is malformed")
    return value


def _append_nomination(root: Path, item: PlannedCandidate, result: SealedScreen,
                       threshold: float) -> None:
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


def classify_screen_series(effects: Sequence[float], *,
                           component_pooled_effects: Sequence[float] = (),
                           continuation_floor: float = 0.0,
                           nomination_floor: float = 0.0,
                           min_replication_effect: float = 0.0,
                           max_replication_spread: float = 0.10,
                           required_replications: int = 2) -> str:
    """Discovery policy classifier; dashboard projection is not authority."""
    if not effects or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in effects):
        raise DiscoveryControllerError("screen series must contain numeric measured effects")
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in (continuation_floor, nomination_floor,
                          min_replication_effect, max_replication_spread))
            or isinstance(required_replications, bool)
            or not isinstance(required_replications, int)
            or not 2 <= required_replications <= 8
            or nomination_floor < continuation_floor):
        raise DiscoveryControllerError("screen series decision policy is malformed")
    if len(effects) == 1:
        return ("candidate" if effects[0] > 0
                and effects[0] >= continuation_floor else "screened_out")
    if min(effects) < 0 < max(effects):
        return "inconclusive"
    # A materially divergent pair is no more rankable than opposite signs.
    # This is the discovery lane's 10 percentage-point spread rule, not a
    # calibration gate; it requests a retest rather than declaring a failure.
    if max(effects) - min(effects) > max_replication_spread:
        return "inconclusive"
    if len(effects) < required_replications:
        return ("candidate" if all(v > 0 and v >= continuation_floor for v in effects)
                else "screened_out")
    pooled = float(statistics.median(effects))
    if any(value < min_replication_effect for value in effects):
        return "screened_out"
    if all(v > 0 for v in effects) and component_pooled_effects and pooled < max(component_pooled_effects):
        return "replicated_but_subadditive"
    if all(v > 0 for v in effects) and pooled >= nomination_floor:
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

def _classified_result(state: Mapping[str, Any], item: PlannedCandidate,
                       result: SealedScreen,
                       decision_policy: Mapping[str, Any] | None = None) -> SealedScreen:
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
        continuation_floor=_decision_floor(
            decision_policy, "continuation_floor_pct", 0.0),
        nomination_floor=_decision_floor(
            decision_policy, "nomination_floor_pct", 0.0),
        min_replication_effect=_decision_floor(
            decision_policy, "min_replication_effect_pct", 0.0),
        max_replication_spread=_decision_floor(
            decision_policy, "max_replication_spread_pct", 0.10),
        required_replications=_required_replications(decision_policy),
    )
    dual_effects_present = (result.exact_attribution_effect_fraction is not None
                            or result.target_runtime_effect_fraction is not None)
    if dual_effects_present and (
            result.exact_attribution_effect_fraction is None
            or result.target_runtime_effect_fraction is None
            or result.exact_attribution_effect_fraction <= 0
            or result.target_runtime_effect_fraction <= 0):
        # Route/device-time and target-runtime throughput are conjunctive.  A
        # disagreement is measured evidence, but never a candidate/nomination.
        classification = "inconclusive"
    return SealedScreen(receipt_path=result.receipt_path,result_sha256=result.result_sha256,effect_fraction=result.effect_fraction,classification=classification,baseline_sha256=result.baseline_sha256,source_proof_sha256=result.source_proof_sha256,dispatch_proof_sha256=result.dispatch_proof_sha256,exact_attribution_effect_fraction=result.exact_attribution_effect_fraction,target_runtime_effect_fraction=result.target_runtime_effect_fraction,candidate_only=result.candidate_only,promotion_claim=result.promotion_claim,stages=result.stages,series_key=series_key,component_series_keys=components,series_effect_fraction=float(statistics.median(effects)))


def _screen_iteration_fields(result: SealedScreen, *, repetition: int) -> dict[str, Any]:
    target_executed = result.target_runtime_effect_fraction is not None
    return {
        "status": result.classification,
        "result_sha256": result.result_sha256,
        "evidence": {"baseline": result.baseline_sha256,
                     "source": result.source_proof_sha256,
                     "dispatch": result.dispatch_proof_sha256},
        "effect_fraction": result.effect_fraction,
        "series_effect_fraction": result.series_effect_fraction,
        "series_key": result.series_key,
        "component_series_keys": list(result.component_series_keys),
        "exact_attribution_effect_fraction":
            result.exact_attribution_effect_fraction,
        "target_runtime_effect_fraction":
            result.target_runtime_effect_fraction,
        "target_runtime_executed": target_executed,
        "target_runtime_reason": (
            None if target_executed else
            "nonpositive_exact_duration"
            if result.exact_attribution_effect_fraction is not None
            and result.exact_attribution_effect_fraction <= 0
            else "not_required_or_unavailable"),
        "stages": list(result.stages),
        "repetition": repetition,
    }


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


def _apply_portfolio_outcome(state: dict[str, Any], row: dict[str, Any]) -> None:
    policy = row.get("portfolio_decision_policy")
    hypothesis_id = row.get("portfolio_hypothesis_id")
    if not isinstance(policy, Mapping) or not isinstance(hypothesis_id, str):
        return
    terminals = state.setdefault("portfolio_terminals", {})
    status = row.get("status")
    # A scientific candidate budget counts measured, recursively sealed
    # screens only.  Critic/source/authorization refusals have no result or
    # evidence graph and cannot establish the terminal claim "no gain after N
    # candidates" merely by carrying distinct proposed manifest hashes.
    measured = (row.get("scientific_budget_spent") is True
                or isinstance(row.get("result_sha256"), str)
                and HASH.fullmatch(row["result_sha256"])
                and isinstance(row.get("evidence"), Mapping))
    if not measured:
        return
    if status == "top_k_replicated_candidate":
        terminals[hypothesis_id] = {"disposition": "nominated",
                                    "policy": dict(policy)}
        row["portfolio_disposition"] = "nominated"
        return
    if status == "inconclusive":
        conflict = policy["conflict_policy"]
        row["portfolio_disposition"] = conflict
        if conflict == "retire":
            terminals[hypothesis_id] = {"disposition": "retire_conflict",
                                        "policy": dict(policy)}
            return
    attempts = {item.get("source_manifest_sha256") for item in state["iterations"]
                if item.get("portfolio_hypothesis_id") == hypothesis_id
                and isinstance(item.get("source_manifest_sha256"), str)
                and (item.get("scientific_budget_spent") is True
                     or isinstance(item.get("result_sha256"), str)
                     and HASH.fullmatch(item["result_sha256"])
                     and isinstance(item.get("evidence"), Mapping))}
    if len(attempts) >= policy["max_distinct_candidates"]:
        disposition = policy["terminal_rule"]
        terminals[hypothesis_id] = {"disposition": disposition,
                                    "policy": dict(policy)}
        row["portfolio_disposition"] = disposition


def _record_precompute_refusal(state: dict[str, Any], row: dict[str, Any],
                               exc: PrecomputeScreenRefusal) -> None:
    """Commit one proven precompute rejection and consume its iteration."""
    state.pop("inflight", None)
    state.pop("pending", None)
    row.update(status="screen_refused",
               reason=f"{type(exc).__name__}: {exc}")
    state["iterations"].append(row)
    _apply_portfolio_outcome(state, row)
    _note_portfolio_authoring_failure(state, row)
    state["next"] += 1


def _record_governed_stage_refusal(
        state: dict[str, Any], row: dict[str, Any],
        exc: GovernedStageRefusal) -> None:
    """Consume one already-sealed stage terminal without replaying its work."""
    state.pop("inflight", None)
    state.pop("pending", None)
    row.update(
        status=exc.disposition, reason=str(exc), stage=exc.stage,
        stage_receipt_path=exc.receipt_path,
        stage_receipt_sha256=exc.receipt_sha256,
        scientific_budget_spent=exc.scientific_budget_spent)
    state["iterations"].append(row)
    if exc.disposition == "authoring_refused":
        _note_portfolio_authoring_failure(state, row)
    hypothesis_id = row.get("portfolio_hypothesis_id")
    terminals = state.setdefault("portfolio_terminals", {})
    if (isinstance(hypothesis_id, str)
            and exc.disposition == "correctness_falsified"):
        terminals[hypothesis_id] = {
            "disposition": exc.disposition,
            "stage_receipt_path": exc.receipt_path,
            "stage_receipt_sha256": exc.receipt_sha256,
        }
    elif (isinstance(hypothesis_id, str)
          and exc.disposition == "attribution_route_falsified"):
        manifest = row.get("source_manifest_sha256")
        policy = row.get("portfolio_decision_policy")
        if (isinstance(manifest, str) and HASH.fullmatch(manifest)
                and isinstance(policy, Mapping)):
            failures = state.setdefault(
                "portfolio_attribution_failures", {}).setdefault(
                    hypothesis_id, [])
            if manifest not in failures:
                failures.append(manifest)
            budget = policy.get("max_distinct_candidates")
            if (isinstance(budget, int) and not isinstance(budget, bool)
                    and budget > 0 and len(failures) >= budget):
                state.setdefault("portfolio_skips", {})[hypothesis_id] = {
                    "disposition": "bounded_attribution_falsified",
                    "scientific_terminal": False,
                    "distinct_candidate_count": len(failures),
                    "stage_receipt_path": exc.receipt_path,
                    "stage_receipt_sha256": exc.receipt_sha256,
                }
    state["next"] += 1


def _note_portfolio_authoring_failure(state: dict[str, Any],
                                      row: Mapping[str, Any]) -> None:
    """Bound repeated non-scientific actor failures without retiring science."""
    hypothesis_id = row.get("portfolio_hypothesis_id")
    if not isinstance(hypothesis_id, str):
        return
    failures = state.setdefault("portfolio_authoring_failures", {})
    count = int(failures.get(hypothesis_id, 0)) + 1
    failures[hypothesis_id] = count
    if count >= 3:
        state.setdefault("portfolio_skips", {})[hypothesis_id] = {
            "disposition": "bounded_authoring_skip",
            "scientific_terminal": False,
            "failure_count": count,
        }


def _record_planner_refusal(state: dict[str, Any], *, turn: int,
                            context: Mapping[str, Any],
                            portfolio_binding: Mapping[str, Any] | None,
                            exc: PlannerOutputRefusal) -> None:
    """Persist one non-candidate authoring refusal without spending science budget."""
    row: dict[str, Any] = {
        "turn": turn,
        "status": ("planner_transient" if isinstance(exc, PlannerProviderTransient)
                   else "planner_refused"),
        "reason": str(exc),
        "refusal_type": ("planner_provider_transient"
                         if isinstance(exc, PlannerProviderTransient)
                         else "planner_output_refusal"),
        "scientific_budget_spent": False,
        "context_sha256": _sha(context),
    }
    if not isinstance(exc, PlannerProviderTransient):
        row["telemetry_event"] = "planner_refused"
        row["telemetry_status"] = exc.telemetry_status
        if exc.telemetry_failure is not None:
            row["telemetry_failure"] = dict(exc.telemetry_failure)
    planning = state.pop("planning", None)
    if isinstance(planning, Mapping):
        row["planner_operation_key"] = planning.get("operation_key")
        if isinstance(planning.get("telemetry_recovery"), Mapping):
            row["planner_checkpoint_reused"] = True
            row["telemetry_recovery"] = dict(
                planning["telemetry_recovery"])
    if portfolio_binding is not None:
        row.update(
            hypothesis_id=portfolio_binding["hypothesis_id"],
            statement=portfolio_binding["statement"],
            falsifier=portfolio_binding["falsifier"],
            regime=dict(portfolio_binding["regime"]),
            portfolio_hypothesis_id=portfolio_binding["hypothesis_id"],
            portfolio_binding=dict(portfolio_binding),
            portfolio_record_sha256=portfolio_binding["record_sha256"],
            portfolio_decision_policy=dict(
                portfolio_binding["decision_policy"]),
        )
    state["iterations"].append(row)
    if isinstance(exc, PlannerProviderTransient):
        # Provider/API availability is neither authored output nor a scientific
        # attempt.  Keep the controller turn and portfolio assignment, but use
        # a fresh sealed actor operation on the next pass.
        state["planner_provider_attempt"] = int(
            state.get("planner_provider_attempt", 0)) + 1
    else:
        _note_portfolio_authoring_failure(state, row)
        state["next"] += 1


def _planning_intent(config: ControllerConfig, *, turn: int,
                     context: Mapping[str, Any],
                     portfolio_binding: Mapping[str, Any] | None,
                     provider_attempt: int = 0) -> dict[str, Any]:
    if isinstance(provider_attempt, bool) or provider_attempt < 0:
        raise DiscoveryControllerError("planner provider attempt is invalid")
    context_sha256 = _sha(context)
    operation_key = _sha({
        "schema": "epyc.autokernel.planning_operation.v1",
        "turn": turn,
        "context_sha256": context_sha256,
        "deployment_identity_sha256": config.deployment_identity_sha256,
        "provider_attempt": provider_attempt,
    })
    workspace = (config.output_root / "planner-operations" /
                 operation_key / "workspace")
    return {
        "phase": "intent", "turn": turn,
        "provider_attempt": provider_attempt,
        "operation_key": operation_key,
        "context": dict(context), "context_sha256": context_sha256,
        "portfolio_binding": (None if portfolio_binding is None
                              else dict(portfolio_binding)),
        "workspace": str(workspace),
    }


def _is_legacy_planner_refusal_telemetry_failure(
        planning: Mapping[str, Any]) -> bool:
    """Recognize only the v16 telemetry-schema crash after actor checkpoint.

    The checkpoint and every actor artifact are independently revalidated by
    the caller before this legacy marker may be cleared.
    """
    failure = planning.get("failure")
    return (planning.get("phase") == "actor_entering"
            and isinstance(failure, Mapping)
            and set(failure) == {"type", "message"}
            and failure.get("type") == "TelemetryError"
            and failure.get("message") ==
            "telemetry result contains a non-allowlisted field")


def _prepare_planner_workspace(config: ControllerConfig, operation_key: str,
                               workspace: Path) -> bool:
    """Create the exact persistent actor workspace without following links."""
    operations = config.output_root / "planner-operations"
    operation = operations / operation_key
    if workspace != operation / "workspace":
        raise DiscoveryControllerError(
            "durable planner workspace escaped its operation namespace")
    ReviewedSourcePackage._require_owned_directory(
        config.output_root, "controller state root")
    if not operations.exists():
        operations.mkdir(mode=0o700)
    ReviewedSourcePackage._require_owned_directory(
        operations, "planner operations root")
    if not operation.exists():
        operation.mkdir(mode=0o700)
    ReviewedSourcePackage._require_owned_directory(
        operation, "planner operation root")
    if workspace.exists() or workspace.is_symlink():
        ReviewedSourcePackage._require_owned_directory(
            workspace, "planner workspace")
        return False
    workspace.mkdir(mode=0o700)
    ReviewedSourcePackage._require_owned_directory(
        workspace, "planner workspace")
    return True


def _reopen_planning_intent(state: Mapping[str, Any], *,
                            turn: int) -> tuple[dict[str, Any],
                                                Mapping[str, Any] | None]:
    planning = state.get("planning")
    if (not isinstance(planning, Mapping) or planning.get("turn") != turn
            or planning.get("phase") not in {"intent", "actor_entering"}
            or not isinstance(planning.get("context"), Mapping)
            or planning.get("context_sha256") != _sha(planning["context"])
            or not isinstance(planning.get("operation_key"), str)
            or not HASH.fullmatch(planning["operation_key"])):
        raise DiscoveryControllerError("durable planning intent is malformed")
    binding = planning.get("portfolio_binding")
    if binding is not None and not isinstance(binding, Mapping):
        raise DiscoveryControllerError("durable planning portfolio binding is malformed")
    return dict(planning["context"]), binding


def run_controller(config: ControllerConfig, *, planner: Planner, critic: Critic, screener: Screener, lease: Lease) -> dict[str, Any]:
    planner_attestation, critic_attestation = dict(planner.attest()), dict(critic.attest())
    if ({k: planner_attestation.get(k) for k in SOL} != SOL
            or {k: critic_attestation.get(k) for k in FABLE5_CRITIC} != FABLE5_CRITIC
            or not isinstance(planner_attestation.get("runtime"), Mapping)
            or not isinstance(critic_attestation.get("runtime"), Mapping)):
        raise DiscoveryControllerError("actors did not attest the sealed planner/critic runtime identities")
    _require_runtime(planner_attestation["runtime"])
    _require_claude_runtime(critic_attestation["runtime"])
    _require_roster(sealed_roster())
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
                 or state.get("inflight") is not None
                 or state.get("planning") is not None)):
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
    existing_portfolio = state.get("hypothesis_portfolio_sha256")
    if existing_portfolio is not None and existing_portfolio != config.hypothesis_portfolio_sha256:
        raise DiscoveryControllerError(
            "sealed hypothesis portfolio changed; durable discovery cannot resume")
    if existing_portfolio is None and config.hypothesis_portfolio_sha256 is not None:
        state["hypothesis_portfolio_sha256"] = config.hypothesis_portfolio_sha256
    # A completed state is an acknowledged terminal checkpoint.  Re-entering it
    # must be a read, not another executor opportunity or a timestamp rewrite.
    if state["complete"]: return state
    tracker=_tracker(store)
    if state.get("inflight") is not None:
        precompute_refused = False
        inflight=state["inflight"]; item=_restore_pending({"candidate":inflight["candidate"]}); authorization=hypotheses.ClaimAuthorization.from_dict(inflight["authorization"]); permit=inflight["lease"]
        inflight_row = dict(inflight["row"])
        _revalidate_portfolio_checkpoint(config, item, inflight_row)
        _bind_campaign_ledger_outcome(inflight_row, authorization)
        if (config.hypothesis_portfolio is not None
                and inflight_row != dict(inflight["row"])):
            raise DiscoveryControllerError(
                "inflight DNR outcomes differ from durable authorization")
        if isinstance(inflight.get("result"),Mapping): result=SealedScreen(**inflight["result"])
        else:
            reconcile=getattr(screener,"reconcile",None)
            if not callable(reconcile): raise DiscoveryControllerError("inflight operation has no reconciliation adapter")
            recovery=reconcile(inflight)
            if not isinstance(recovery,Recovery) or recovery.status == "ambiguous": raise DiscoveryControllerError("inflight operation cannot be safely reconciled")
            if recovery.status == "sealed_result":
                result=recovery.result
            else:
                resume=getattr(lease,"resume",None)
                if not callable(resume):
                    raise DiscoveryControllerError("safe inflight recovery lacks resource re-admission")
                fresh_permit=resume(item,permit)
                if not bool(fresh_permit.get("admitted")):
                    row=dict(inflight["row"]); row.update(
                        status="waiting_resource",lease=dict(fresh_permit))
                    state.pop("inflight",None)
                    state["pending"]={
                        "row":row,"candidate":inflight["candidate"],
                        "authorization":inflight["authorization"],
                        "confirmation":bool(inflight.get("confirmation")),
                        "parent_authorization":inflight.get("parent_authorization")}
                    store.save(state,"waiting_resource")
                    return state
                fresh_permit={**dict(fresh_permit),
                              "repetition":permit.get("repetition")}
                inflight["lease"]=fresh_permit; permit=fresh_permit
                store.save(state,"pre_screen_reacquired")
                try:
                    result=screener.screen(item,authorization,permit)
                except ResumableScreenInterruption as exc:
                    inflight["interruption"] = {
                        "type": type(exc).__name__, "message": str(exc),
                        "resumable": True,
                    }
                    store.save(state, "screen_resumable_interruption")
                    return state
                except ResourceWait as exc:
                    wait_receipt=_validated_resource_wait(
                        exc,str(inflight["operation_key"]))
                    _require_safe_resource_wait_recovery(screener,inflight)
                    row=dict(inflight["row"]); row.update(
                        status="waiting_resource",lease=wait_receipt)
                    state.pop("inflight",None)
                    state["pending"]={
                        "row":row,"candidate":inflight["candidate"],
                        "authorization":inflight["authorization"],
                        "confirmation":bool(inflight.get("confirmation")),
                        "parent_authorization":inflight.get("parent_authorization")}
                    store.save(state,"waiting_resource")
                    return state
                except PrecomputeScreenRefusal as exc:
                    row = dict(inflight["row"])
                    _record_precompute_refusal(state, row, exc)
                    store.save(state, "screen_refused")
                    precompute_refused = True
                except GovernedStageRefusal as exc:
                    row = dict(inflight["row"])
                    _record_governed_stage_refusal(state, row, exc)
                    store.save(state, exc.disposition)
                    precompute_refused = True
        if not precompute_refused:
            if not isinstance(result,SealedScreen): raise DiscoveryControllerError("inflight recovery produced no sealed result")
            row=dict(inflight["row"]); policy=row.get("portfolio_decision_policy")
            result=_classified_result(state,item,result,policy); row.update(
                _screen_iteration_fields(
                    result, repetition=int(inflight["lease"].get(
                        "repetition", 2 if inflight.get("confirmation") else 1))))
            _record_attempt_once(tracker,item,str(item.proposal.get("proposal_id",row["proposal_sha256"])),result)
            state.pop("inflight",None); state.pop("pending",None); state["iterations"].append(row); _apply_portfolio_outcome(state,row); state["next"]+=1; _schedule_replication(state,item=item,authorization=authorization,row=row,result=result,max_iterations=config.max_iterations); _append_nomination(config.output_root,item,result,_decision_floor(policy,"nomination_floor_pct",config.nomination_threshold)); _write_projection(config.evidence_root or config.output_root); store.save(state,"recovered_screen")
    while not state["complete"] and state["next"] <= config.max_iterations:
        turn=state["next"]
        pending=state.get("pending")
        planning=state.get("planning")
        if pending is not None and planning is not None:
            raise DiscoveryControllerError(
                "controller cannot own pending candidate and planning intent together")
        if planning is not None:
            context, portfolio_binding = _reopen_planning_intent(
                state, turn=turn)
        else:
            portfolio_binding = (pending.get("row", {}).get("portfolio_binding")
                                 if pending is not None else
                                 _select_portfolio_binding(state, config))
            if (pending is None and config.hypothesis_portfolio is not None
                    and portfolio_binding is None):
                state["complete"] = True
                state["terminal_reason"] = "portfolio_exhausted"
                store.save(state, "portfolio_exhausted")
                break
            if (pending is not None and isinstance(pending.get("context"), Mapping)):
                context = dict(pending["context"])
                if pending.get("context_sha256") != _sha(context):
                    raise DiscoveryControllerError(
                        "pending actor context identity changed")
            else:
                context=_context(state,tracker,turn,config,portfolio_binding)
            if pending is None:
                state["planning"] = _planning_intent(
                    config, turn=turn, context=context,
                    portfolio_binding=portfolio_binding,
                    provider_attempt=int(
                        state.get("planner_provider_attempt", 0)))
                store.save(state, "planner_intent")
                planning = state["planning"]
        with tempfile.TemporaryDirectory(prefix=f"ak-discovery-{turn}-", dir=config.output_root) as temp:
            workspace=Path(temp)
            pending_phase = pending.get("phase") if isinstance(pending, Mapping) else None
            if pending is not None and pending_phase == "critic_pending":
                item=_restore_pending(pending); row=dict(pending["row"])
                _revalidate_portfolio_checkpoint(config, item, row)
                review=critic.review(item,context=context,workspace=workspace)
                row["critic"]=asdict(review)
                if review.decision != "accept":
                    row["status"]="critic_"+review.decision
                    state.pop("pending", None); state["iterations"].append(row)
                    _apply_portfolio_outcome(state,row)
                    _note_portfolio_authoring_failure(state, row)
                    state["next"]+=1
                    store.save(state,"critic_refused"); continue
                state["pending"]={
                    "phase":"critic_complete", "row":row,
                    "candidate":pending["candidate"],
                    "context":dict(context), "context_sha256":_sha(context),
                    "confirmation":False, "parent_authorization":None}
                store.save(state,"critic_checkpointed")
                continue
            if pending is not None:
                item=_restore_pending(pending); row=dict(pending["row"]); review=Critique(**row["critic"])
                _revalidate_portfolio_checkpoint(config, item, row)
                if "authorization" in pending:
                    authorization=hypotheses.ClaimAuthorization.from_dict(pending["authorization"])
                    durable_row = dict(row)
                    _bind_campaign_ledger_outcome(row, authorization)
                    if config.hypothesis_portfolio is not None and row != durable_row:
                        raise DiscoveryControllerError(
                            "portfolio pending candidate lacks campaign-ledger DNR outcome")
                elif pending_phase == "critic_complete":
                    authorization=None
                elif pending.get("confirmation") is True:
                    # A positive S1 is not a receipted negative.  Re-consult
                    # DNR and mint the explicit confirmation token before its
                    # own device claim, rather than reusing S1's token.
                    _ensure_question(
                        tracker, item,
                        row.get("portfolio_binding")
                        if isinstance(row.get("portfolio_binding"), Mapping) else None)
                    authorization=tracker.authorize_claim(item.hypothesis_id,purpose="candidate_only_confirmation",authorized_by="discovery_controller",ledger=do_not_repeat.compile_for_tracker(tracker))
                    _bind_campaign_ledger_outcome(row, authorization)
                else:
                    raise DiscoveryControllerError("pending candidate lacks a sealed authorization")
            else:
                planning = state["planning"]
                planner_workspace=Path(str(planning["workspace"]))
                expected_workspace=(config.output_root / "planner-operations" /
                                    planning["operation_key"] / "workspace")
                if planner_workspace != expected_workspace:
                    raise DiscoveryControllerError(
                        "durable planner workspace escaped its operation namespace")
                checkpoint_path=planner_workspace.parent / "actor-result.json"
                if isinstance(planning.get("failure"), Mapping):
                    if _is_legacy_planner_refusal_telemetry_failure(planning):
                        # This exact historical failure occurred after rc=0 was
                        # checkpointed and while emitting the typed refusal.
                        # Validate the private, single-link actor closure now;
                        # a missing/extra/tampered artifact stays terminal.
                        _reopen_planner_actor_checkpoint(
                            planner_workspace, checkpoint_path,
                            context=context)
                        planning.pop("failure")
                        planning["telemetry_recovery"] = {
                            "schema": "epyc.autokernel.planner_telemetry_recovery.v1",
                            "disposition": "resume_checkpoint_and_rederive_refusal",
                        }
                        store.save(state, "planner_telemetry_recovery")
                    else:
                        raise DiscoveryControllerError(
                            "prior planner infrastructure/authority failure remains terminal: "
                            f"{planning['failure'].get('type')}: "
                            f"{planning['failure'].get('message')}")
                if planning["phase"] == "intent":
                    planning["phase"] = "actor_entering"
                    store.save(state, "planner_entering")
                try:
                    workspace_created=_prepare_planner_workspace(
                        config, planning["operation_key"], planner_workspace)
                    if not workspace_created:
                        resume_plan=getattr(planner,"resume_plan",None)
                        if not callable(resume_plan):
                            raise PlannerOutputRefusal(
                                "planner stopped before a reusable actor checkpoint")
                        resume_kwargs={"context":context,"workspace":planner_workspace}
                        if "checkpoint_path" in inspect.signature(resume_plan).parameters:
                            resume_kwargs["checkpoint_path"]=checkpoint_path
                        item=resume_plan(**resume_kwargs)
                    else:
                        plan_kwargs={"context":context,"workspace":planner_workspace}
                        if "checkpoint_path" in inspect.signature(planner.plan).parameters:
                            plan_kwargs["checkpoint_path"]=checkpoint_path
                        item=planner.plan(**plan_kwargs)
                except PlannerOutputRefusal as exc:
                    _record_planner_refusal(
                        state, turn=turn, context=context,
                        portfolio_binding=portfolio_binding, exc=exc)
                    store.save(
                        state,
                        "planner_transient"
                        if isinstance(exc, PlannerProviderTransient)
                        else "planner_refused")
                    continue
                except Exception as exc:
                    state["planning"]["failure"]={
                        "type":type(exc).__name__, "message":str(exc)}
                    store.save(state,"planner_terminal_failure")
                    raise
                row={"turn":turn,"hypothesis_id":item.hypothesis_id,"statement":item.statement,
                     "falsifier":item.falsifier,"regime":dict(item.regime),
                     "proposal_sha256":_sha(item.proposal),"source_manifest_sha256":item.source_manifest_sha256,
                     "experiment_intent":asdict(item.experiment_intent) if item.experiment_intent else None,
                     "mechanism_id":item.source_manifest.mechanism_id,
                     "target_surface":item.experiment_intent.target_surface if item.experiment_intent else None,
                     "target_symbol":item.experiment_intent.target_symbol if item.experiment_intent else None,
                     "context_sha256":_sha(context)}
                if portfolio_binding is not None:
                    row.update(portfolio_hypothesis_id=portfolio_binding["hypothesis_id"],
                               portfolio_binding=dict(portfolio_binding),
                               portfolio_record_sha256=portfolio_binding["record_sha256"],
                               portfolio_decision_policy=dict(
                                   portfolio_binding["decision_policy"]))
                    try:
                        _validate_portfolio_candidate(
                            item, portfolio_binding,
                            config.hypothesis_portfolio)
                    except DiscoveryControllerError as exc:
                        row.update(status="planner_contract_refused",
                                   reason=str(exc))
                        state.pop("planning", None)
                        state["iterations"].append(row)
                        _note_portfolio_authoring_failure(state, row)
                        state["next"] += 1
                        store.save(state, "planner_contract_refused")
                        continue
                    receipt = _portfolio_exact_dnr_check(
                        config, item, portfolio_binding)
                    row["portfolio_exact_dnr_check"] = receipt
                    if receipt["outcome"] == schemas.FAIL:
                        row.update(
                            status="portfolio_dnr_refused",
                            reason="candidate exactly repeats sealed portfolio DNR "
                                   + ", ".join(receipt["matched_dnr_ids"]))
                        state.pop("planning", None)
                        state["iterations"].append(row)
                        state.setdefault("portfolio_terminals", {})[
                            portfolio_binding["hypothesis_id"]] = {
                                "disposition": "portfolio_dnr_refused",
                                "policy": dict(portfolio_binding["decision_policy"]),
                                "receipt_sha256": receipt["receipt_sha256"],
                            }
                        state["next"] += 1
                        store.save(state, "portfolio_dnr_refused")
                        continue
                state.pop("planning", None)
                state["pending"]={
                    "phase":"critic_pending", "row":row,
                    "candidate":_pending_item(item),
                    "context":dict(context), "context_sha256":_sha(context),
                    "confirmation":False, "parent_authorization":None}
                store.save(state,"planner_checkpointed")
                continue
            if review.decision != "accept":
                row["status"]="critic_"+review.decision; state["iterations"].append(row); _apply_portfolio_outcome(state,row); _note_portfolio_authoring_failure(state,row); state["next"]+=1; store.save(state,"critic_refused"); continue
            if pending_phase == "critic_complete":
                _ensure_question(
                    tracker, item,
                    row.get("portfolio_binding")
                    if isinstance(row.get("portfolio_binding"), Mapping) else None)
                ledger=do_not_repeat.compile_for_tracker(tracker)
                try:
                    authorization=tracker.authorize_claim(item.hypothesis_id,purpose="candidate_only_discovery",authorized_by="discovery_controller",ledger=ledger)
                    _bind_campaign_ledger_outcome(row, authorization)
                except hypotheses.RepeatsAReceiptedNegative as exc:
                    row.update(campaign_ledger_dnr_outcome=schemas.FAIL,
                               campaign_ledger_dnr_reasons=[str(exc)],
                               status="authorization_refused",reason=str(exc)); state.pop("pending",None); state["iterations"].append(row); _apply_portfolio_outcome(state,row); _note_portfolio_authoring_failure(state,row); state["next"]+=1; store.save(state,"authorization_refused"); continue
                except hypotheses.HypothesisError as exc:
                    row.update(status="authorization_refused",reason=str(exc)); state.pop("pending",None); state["iterations"].append(row); _apply_portfolio_outcome(state,row); _note_portfolio_authoring_failure(state,row); state["next"]+=1; store.save(state,"authorization_refused"); continue
            if config.dry_run:
                # The dry-run still proves exact Sol/Terra actor attestation,
                # plan schema, critic binding, and DNR authorization.  It
                # deliberately never asks for a resource lease or starts a
                # source build, correctness, attribution, or model call.
                row.update(status="dry_run_authorized", authorization=authorization.to_dict())
                state.pop("pending", None); state["iterations"].append(row); _apply_portfolio_outcome(state,row)
                dry_hypothesis = row.get("portfolio_hypothesis_id")
                if isinstance(dry_hypothesis, str):
                    state.setdefault("portfolio_validations", {})[dry_hypothesis] = {
                        "disposition": "dry_run_validated",
                        "scientific_terminal": False,
                    }
                state["next"] += 1
                if (config.hypothesis_portfolio is not None
                        and _select_portfolio_binding(state, config) is None):
                    state["complete"] = True
                    state["terminal_reason"] = "portfolio_exhausted"
                store.save(state, "dry_run_authorized")
                continue
            repetition=2 if pending and pending.get("confirmation") else 1
            operation_key=_sha({"turn":turn,"manifest":item.source_manifest_sha256,"authorization":authorization.to_dict(),"repetition":repetition})
            permit=lease.admit(item, operation_key=operation_key)
            if not bool(permit.get("admitted")):
                # Waiting is durable but is not an experiment and cannot spend an
                # iteration budget.  Planning/critique may continue elsewhere;
                # this exact candidate is retried only after a new lease admits it.
                row.update(status="waiting_resource",lease=dict(permit)); state["pending"]={"row":row,"candidate":_pending_item(item),"authorization":authorization.to_dict(),"confirmation":bool(pending and pending.get("confirmation")),"parent_authorization":pending.get("parent_authorization") if pending else None}; store.save(state,"waiting_resource"); break
            # The governed GPU adapter owns an operation-key-bound receipt
            # namespace.  It refuses an unkeyed lease, making recovery and
            # result reconciliation refer to the same durable operation.
            if permit.get("operation_key") != operation_key:
                raise DiscoveryControllerError("resource lease did not bind the exact operation key")
            permit={**dict(permit), "repetition":repetition}
            state.pop("pending",None)
            state["inflight"]={"operation_key":operation_key,"row":row,"candidate":_pending_item(item),"authorization":authorization.to_dict(),"lease":dict(permit),"confirmation":bool(pending and pending.get("confirmation")),"parent_authorization":pending.get("parent_authorization") if pending else None}
            store.save(state,"pre_screen_intent")
            try: result=screener.screen(item,authorization,permit)
            except ResumableScreenInterruption as exc:
                state["inflight"]["interruption"] = {
                    "type": type(exc).__name__, "message": str(exc),
                    "resumable": True,
                }
                store.save(state, "screen_resumable_interruption")
                break
            except ResourceWait as exc:
                wait_receipt=_validated_resource_wait(exc,operation_key)
                _require_safe_resource_wait_recovery(screener,state["inflight"])
                state.pop("inflight",None)
                row.update(status="waiting_resource",lease=wait_receipt)
                state["pending"]={"row":row,"candidate":_pending_item(item),"authorization":authorization.to_dict(),"confirmation":bool(pending and pending.get("confirmation")),"parent_authorization":pending.get("parent_authorization") if pending else None}
                store.save(state,"waiting_resource")
                break
            except PrecomputeScreenRefusal as exc:
                _record_precompute_refusal(state, row, exc)
                store.save(state,"screen_refused"); continue
            except GovernedStageRefusal as exc:
                _record_governed_stage_refusal(state, row, exc)
                store.save(state, exc.disposition)
                continue
            except Exception as exc:
                # The durable start intent has been written.  An ordinary
                # exception may follow a build, claim, or model invocation, so
                # it is an ambiguous operation until the governed adapter
                # reconciles its operation-key-bound artifacts on restart.
                state["inflight"]["exception"]={"type":type(exc).__name__,"message":str(exc)}
                store.save(state,"screen_ambiguous")
                raise
            state["inflight"]["result"]=asdict(result); store.save(state,"post_screen_result")
            policy=row.get("portfolio_decision_policy")
            result=_classified_result(state,item,result,policy); row.update(
                _screen_iteration_fields(result, repetition=repetition))
            # Record the measured disposition before exposing it to the next
            # planner context.  This is the only source of repeat suppression.
            _record_attempt_once(tracker,item,str(item.proposal.get("proposal_id",row["proposal_sha256"])),result)
            state.pop("inflight",None); state.pop("pending",None); state["iterations"].append(row); _apply_portfolio_outcome(state,row); state["next"]+=1; _schedule_replication(state,item=item,authorization=authorization,row=row,result=result,max_iterations=config.max_iterations); _append_nomination(config.output_root,item,result,_decision_floor(policy,"nomination_floor_pct",config.nomination_threshold)); _write_projection(config.evidence_root or config.output_root); store.save(state,"screened")
    state["complete"]=bool(state.get("complete")) or state["next"]>config.max_iterations
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
