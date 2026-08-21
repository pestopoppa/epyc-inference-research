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
import stat
import subprocess
from typing import Any, Mapping

from .. import hypothesis_portfolio, preauthored_continuation, schemas
from . import gpu_load_admission


SCHEMA = "epyc.autokernel.discovery_deployment.v6"
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
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


PLANNER_CONTEXT_SCHEMA = "epyc.autokernel.discovery_planner_context.v4"
EVIDENCE_MANIFEST_SCHEMA = "epyc.autokernel.hypothesis_evidence_manifest.v1"
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


@dataclass(frozen=True)
class HypothesisPortfolioInput:
    input: ImmutableInput
    value: hypothesis_portfolio.Portfolio

    def revalidate(self) -> None:
        self.input.revalidate("hypothesis_portfolio")
        try:
            refreshed = hypothesis_portfolio.load(self.input.path)
        except hypothesis_portfolio.PortfolioError as exc:
            raise DeploymentConfigError("hypothesis portfolio schema/content mismatch") from exc
        if refreshed.sha256 != self.value.sha256:
            raise DeploymentConfigError("hypothesis portfolio semantic identity changed")


@dataclass(frozen=True)
class HypothesisEvidenceManifest:
    input: ImmutableInput
    value: Mapping[str, Any]

    def revalidate(self, portfolio: hypothesis_portfolio.Portfolio) -> None:
        self.input.revalidate("hypothesis_evidence_manifest")
        refreshed = _evidence_manifest(self.input, portfolio=portfolio)
        if refreshed.value != self.value:
            raise DeploymentConfigError("hypothesis evidence manifest changed")


@dataclass(frozen=True)
class PreauthoredContinuationInput:
    input: ImmutableInput
    value: preauthored_continuation.PreauthoredContinuation

    def revalidate(self) -> None:
        self.input.revalidate("preauthored_continuation")
        try:
            refreshed = preauthored_continuation.load(self.input.path)
        except preauthored_continuation.PreauthoredContinuationError as exc:
            raise DeploymentConfigError(
                "preauthored continuation changed") from exc
        if refreshed != self.value:
            raise DeploymentConfigError(
                "preauthored continuation semantic identity changed")


CARRY_FORWARD_SCHEMA = "epyc.autokernel.discovery_carry_forward.v2"


@dataclass(frozen=True)
class CarryForwardInput:
    """Canonical predecessor authority read and revalidated through one fd."""

    input: ImmutableInput
    value: Mapping[str, Any]
    self_sha256: str
    semantic_sha256: str

    def revalidate(self) -> None:
        refreshed = _carry_forward({
            "path": str(self.input.path),
            "sha256": self.input.sha256,
            "self_sha256": self.self_sha256,
            "semantic_sha256": self.semantic_sha256,
        })
        if refreshed != self:
            raise DeploymentConfigError(
                "carry_forward immutable authority changed")


def _carry_forward(value: object) -> CarryForwardInput:
    raw = _exact(value, {"path", "sha256", "self_sha256", "semantic_sha256"},
                 "carry_forward")
    path = _absolute(raw["path"], "carry_forward.path")
    file_sha256 = _digest_identifier(raw["sha256"], "carry_forward.sha256")
    self_sha256 = _digest_identifier(
        raw["self_sha256"], "carry_forward.self_sha256")
    semantic_sha256 = _digest_identifier(
        raw["semantic_sha256"], "carry_forward.semantic_sha256")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_nlink != 1 or before.st_mode & 0o022):
            raise DeploymentConfigError(
                "carry_forward has unsafe file authority")
        handle = os.fdopen(descriptor, "rb")
        descriptor = -1
        with handle:
            payload = handle.read()
            after = os.fstat(handle.fileno())
        if ((before.st_dev, before.st_ino, before.st_size,
             before.st_mtime_ns, before.st_ctime_ns, before.st_nlink)
                != (after.st_dev, after.st_ino, after.st_size,
                    after.st_mtime_ns, after.st_ctime_ns, after.st_nlink)):
            raise DeploymentConfigError("carry_forward changed while read")
    except OSError as exc:
        raise DeploymentConfigError("carry_forward is unreadable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != file_sha256:
        raise DeploymentConfigError("carry_forward file identity changed")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result

    try:
        body = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise DeploymentConfigError(
            "carry_forward is not strict JSON") from exc
    keys = {
        "schema", "predecessor_state_file_sha256",
        "predecessor_journal_file_sha256",
        "predecessor_state_semantic_sha256", "portfolio_outcomes",
        "candidate_semantic_sha256", "candidate_patch_sha256",
        "cross_campaign_candidate_sha256",
        "attribution_expectation_erratum", "carry_forward_sha256",
    }
    expected_outcomes = {
        "akh-v2-q5-type-specific-dequant": "nominated",
        "akh-v2-q8-quantizer-new-mechanism": "retire",
        "akh-v2-fa-gqa7-pair-tail": "bounded_authoring_skip",
        "akh-v2-rms-direct-load-reduction": "bounded_authoring_skip",
    }
    calculated = schemas.content_hash({
        key: item for key, item in body.items()
        if key != "carry_forward_sha256"}) if isinstance(body, Mapping) else None
    canonical = ((json.dumps(dict(body), sort_keys=True, separators=(",", ":"))
                  + "\n").encode() if isinstance(body, Mapping) else b"")
    if (not isinstance(body, Mapping) or set(body) != keys
            or body.get("schema") != CARRY_FORWARD_SCHEMA
            or body.get("portfolio_outcomes") != expected_outcomes
            or any(not isinstance(body.get(key), str)
                   or not SHA.fullmatch(body[key])
                   for key in (
                       "predecessor_state_file_sha256",
                       "predecessor_journal_file_sha256",
                       "predecessor_state_semantic_sha256"))
            or any(not isinstance(body.get(key), list)
                   or body[key] != sorted(set(body[key]))
                   or any(not isinstance(item, str) or not SHA.fullmatch(item)
                          for item in body[key])
                   for key in (
                       "candidate_semantic_sha256", "candidate_patch_sha256",
                       "cross_campaign_candidate_sha256"))
            or tuple(len(body[key]) for key in (
                "candidate_semantic_sha256", "candidate_patch_sha256",
                "cross_campaign_candidate_sha256")) != (13, 8, 8)
            or not isinstance(body.get("attribution_expectation_erratum"),
                              Mapping)
            or body.get("carry_forward_sha256") != self_sha256
            or calculated != self_sha256
            or semantic_sha256 != self_sha256
            or payload != canonical):
        raise DeploymentConfigError(
            "carry_forward semantic authority changed")
    return CarryForwardInput(
        ImmutableInput(path, file_sha256), dict(body),
        self_sha256, semantic_sha256)


def _portfolio(raw: ImmutableInput) -> HypothesisPortfolioInput:
    try:
        value = hypothesis_portfolio.load(raw.path)
    except hypothesis_portfolio.PortfolioError as exc:
        raise DeploymentConfigError("hypothesis portfolio schema/content mismatch") from exc
    return HypothesisPortfolioInput(raw, value)


def _evidence_manifest(raw: ImmutableInput, *, portfolio: hypothesis_portfolio.Portfolio
                       ) -> HypothesisEvidenceManifest:
    try:
        body = json.loads(raw.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentConfigError("hypothesis evidence manifest is not JSON") from exc
    required = {"schema", "portfolio_sha256", "evidence", "manifest_sha256"}
    if not isinstance(body, Mapping) or set(body) != required:
        raise DeploymentConfigError("hypothesis evidence manifest schema is incomplete")
    if (body["schema"] != EVIDENCE_MANIFEST_SCHEMA
            or body["portfolio_sha256"] != portfolio.sha256
            or body["manifest_sha256"] != schemas.content_hash(
                {key: value for key, value in body.items() if key != "manifest_sha256"})):
        raise DeploymentConfigError("hypothesis evidence manifest identity mismatch")
    evidence = body["evidence"]
    expected = {row["evidence_id"]: row["sha256"] for row in portfolio.body["evidence"]}
    if not isinstance(evidence, Mapping) or set(evidence) != set(expected):
        raise DeploymentConfigError("hypothesis evidence manifest coverage mismatch")
    for evidence_id, row in evidence.items():
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise DeploymentConfigError("hypothesis evidence manifest row is malformed")
        path = _absolute(row["path"], f"hypothesis evidence {evidence_id}.path")
        expected_root = raw.path.parent.parent / "portfolio-evidence"
        try:
            resolved_evidence_root = expected_root.resolve(strict=True)
            status = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise DeploymentConfigError(
                "hypothesis evidence bundle root/carrier is unavailable") from exc
        if (not _under(path, resolved_evidence_root)
                or status.st_uid != os.geteuid() or status.st_nlink != 1
                or status.st_mode & 0o077):
            raise DeploymentConfigError(
                "hypothesis evidence is not a private bundle-owned carrier")
        if row["sha256"] != expected[evidence_id]:
            raise DeploymentConfigError("hypothesis evidence manifest digest differs from corpus")
        _digest_file(path, row["sha256"], f"hypothesis evidence {evidence_id}")
    return HypothesisEvidenceManifest(raw, dict(body))


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
                     workload: ImmutableInput, runtime_config: ImmutableInput,
                     portfolio: HypothesisPortfolioInput,
                     evidence_manifest: HypothesisEvidenceManifest,
                     continuation: PreauthoredContinuationInput) -> PlannerContext:
    raw = _input(value, "planner_context")
    if raw.path.stat().st_size > _PLANNER_CONTEXT_LIMIT:
        raise DeploymentConfigError("planner_context exceeds its bounded actor input limit")
    try:
        body = json.loads(raw.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentConfigError("planner_context is not JSON") from exc
    required = {"schema", "context_sha256", "model_sha256", "workload_sha256",
                "runtime_config_sha256", "profile_receipts", "hotspots",
                "source_constraints", "initial_strategies",
                "hypothesis_portfolio_sha256", "eligible_hypotheses",
                "do_not_repeat", "incumbents", "ineligible_hypotheses",
                "hypothesis_evidence_manifest_sha256", "hypothesis_evidence",
                "reviewed_source_package_sha256", "template_registry_sha256",
                "template_symbol_authority",
                "template_surfaces_sha256", "template_surfaces",
                "portfolio_dispatch_authority",
                "preauthored_continuation_sha256",
                "preauthored_source_backed_diff_sha256",
                "preauthored_historical_evidence_sha256"}
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
    if body.get("hypothesis_portfolio_sha256") != portfolio.value.sha256:
        raise DeploymentConfigError("planner_context is not bound to the sealed portfolio")
    if body.get("hypothesis_evidence_manifest_sha256") != evidence_manifest.value["manifest_sha256"]:
        raise DeploymentConfigError("planner_context is not bound to vendored portfolio evidence")
    if (body.get("preauthored_continuation_sha256")
            != continuation.value.sha256):
        raise DeploymentConfigError(
            "planner_context preauthored continuation identity mismatch")
    if (body.get("preauthored_source_backed_diff_sha256")
            != continuation.value.source_backed_diff_sha256):
        raise DeploymentConfigError(
            "planner_context preauthored source-backed identity mismatch")
    if not SHA.fullmatch(str(body.get(
            "preauthored_historical_evidence_sha256"))):
        raise DeploymentConfigError(
            "planner_context historical continuation evidence is unsealed")
    if (not all(SHA.fullmatch(str(body.get(key))) for key in (
            "reviewed_source_package_sha256", "template_registry_sha256",
            "template_surfaces_sha256"))
            or schemas.content_hash(body.get("template_surfaces"))
            != body.get("template_surfaces_sha256")):
        raise DeploymentConfigError("planner_context source/template authority is malformed")
    symbol_authority = body.get("template_symbol_authority")
    if (not isinstance(symbol_authority, Mapping) or not symbol_authority
            or any(not isinstance(template_id, str)
                   or not isinstance(by_file, Mapping) or not by_file
                   or any(not isinstance(path, str)
                          or not isinstance(symbols, list) or not symbols
                          or symbols != sorted(set(symbols))
                          or any(not isinstance(symbol, str) or not symbol
                                 for symbol in symbols)
                          for path, symbols in by_file.items())
                   for template_id, by_file in symbol_authority.items())):
        raise DeploymentConfigError(
            "planner_context template symbol authority is malformed")
    if body.get("hypothesis_evidence") != evidence_manifest.value["evidence"]:
        raise DeploymentConfigError("planner_context evidence projection differs from vendored manifest")
    if not all(isinstance(body.get(key), list) for key in (
            "eligible_hypotheses", "do_not_repeat", "incumbents", "ineligible_hypotheses")):
        raise DeploymentConfigError("planner_context portfolio partitions are malformed")
    hypotheses = tuple(portfolio.value.hypotheses)
    expected_partitions = {
        "eligible_hypotheses": _jsonable(portfolio.value.eligible_projection()),
        "do_not_repeat": _jsonable(portfolio.value.dnr_projection()),
        "incumbents": _jsonable(tuple(row for row in hypotheses
                                       if row["status"] == "candidate_incumbent")),
        "ineligible_hypotheses": _jsonable(tuple(row for row in hypotheses
                                                  if not row["current_bundle_eligibility"]["eligible"])),
    }
    if any(body[key] != expected for key, expected in expected_partitions.items()):
        raise DeploymentConfigError("planner_context portfolio projection is not exact")
    dispatch_authority = body["portfolio_dispatch_authority"]
    eligible_ids = {row["hypothesis_id"] for row in portfolio.value.eligible_hypotheses()}
    if (not isinstance(dispatch_authority, Mapping)
            or set(dispatch_authority) != eligible_ids
            or any(not isinstance(rows, list) or not 1 <= len(rows) <= 8
                   or any(not isinstance(row, Mapping)
                          or set(row) != {"route_id", "kernel_name", "calls", "grid", "workgroup", "lds_bytes"}
                          for row in rows)
                   for rows in dispatch_authority.values())):
        raise DeploymentConfigError("planner_context dispatch authority is malformed")
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
    production_branch: str
    production_head: str
    instrument_path: Path
    instrument_branch: str
    instrument_commit: str
    state_root: Path
    evidence_root: Path
    operations_root: Path
    build_root: Path
    max_iterations: int
    nomination_threshold: float
    actor_wrapper: ImmutableInput
    critic_wrapper: ImmutableInput
    environment_profile_id: str
    device_id: str
    claim_timeout_s: float
    inference_window_lock: Path
    model: ImmutableInput
    workload: ImmutableInput
    runtime_config: ImmutableInput
    policy: ImmutableInput
    admission_policy: AdmissionPolicy
    hypothesis_portfolio: HypothesisPortfolioInput
    hypothesis_evidence_manifest: HypothesisEvidenceManifest
    preauthored_continuation: PreauthoredContinuationInput
    q5_lds0_attribution_erratum: ImmutableInput
    carry_forward: CarryForwardInput
    hypothesis_portfolio_contract: ImmutableInput
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
        _verify_production(self.production_path, self.production_branch, self.production_head)
        _verify_instrument(self.instrument_path, self.production_head,
                           self.instrument_branch, self.instrument_commit)
        self.actor_wrapper.revalidate("actors.wrapper")
        self.critic_wrapper.revalidate("actors.critic")
        for label, value in (("model", self.model), ("workload", self.workload),
                             ("runtime_config", self.runtime_config), ("policy", self.policy)):
            value.revalidate(label)
        self.admission_policy.revalidate()
        self.hypothesis_portfolio.revalidate()
        self.hypothesis_evidence_manifest.revalidate(self.hypothesis_portfolio.value)
        self.preauthored_continuation.revalidate()
        self.q5_lds0_attribution_erratum.revalidate(
            "q5_lds0_attribution_erratum")
        self.carry_forward.revalidate()
        self.hypothesis_portfolio_contract.revalidate("hypothesis_portfolio_contract")
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


def _verify_production(path: Path, declared_branch: str, declared_head: str) -> None:
    expected = FROZEN_PRODUCTION_PATH.resolve(strict=True)
    if (path.resolve(strict=True) != expected or declared_head != FROZEN_PRODUCTION_HEAD
            or declared_branch != FROZEN_PRODUCTION_BRANCH):
        raise DeploymentConfigError("production identity is not the exact frozen tree/branch/head")
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
    # The shared production checkout may contain pre-existing untracked local
    # tooling.  It is not authority for experiments and must not be deleted or
    # prettified here.  Tracked/index changes, however, invalidate the freeze.
    if git("status", "--porcelain", "--untracked-files=no"):
        raise DeploymentConfigError("frozen production tree has tracked changes")


def _verify_instrument(path: Path, production_head: str, instrument_branch: str,
                       instrument_commit: str) -> None:
    if (not isinstance(instrument_branch, str) or not instrument_branch
            or len(instrument_branch) > 255 or "\x00" in instrument_branch):
        raise DeploymentConfigError("instrument branch must be an exact identifier")
    if not isinstance(instrument_commit, str) or not GIT_SHA.fullmatch(instrument_commit):
        raise DeploymentConfigError("instrument commit must be an exact Git SHA")
    if path.is_symlink() or not path.is_dir():
        raise DeploymentConfigError("instrument.repo_path must be a regular Git checkout")
    branch_check = subprocess.run(
        ("git", "-C", str(path), "check-ref-format", "--branch", instrument_branch),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False)
    if branch_check.returncode:
        raise DeploymentConfigError("instrument branch must be an exact Git branch name")
    # The measurement instrument is an explicitly approved descendant in a
    # separate experimental repository.  Its checked-out worktree may be dirty;
    # authority is the sealed branch object, never that checkout state.
    check = subprocess.run(("git", "-C", str(path), "merge-base", "--is-ancestor",
                            production_head, instrument_commit), stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
    if check.returncode != 0:
        raise DeploymentConfigError("instrument commit is not a production descendant")
    resolved = subprocess.run(("git", "-C", str(path), "rev-parse",
                               f"refs/heads/{instrument_branch}"),
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              check=False)
    if resolved.returncode or resolved.stdout.strip() != instrument_commit:
        raise DeploymentConfigError("instrument branch does not resolve to the sealed instrument commit")


def _validate_root(path: Path, label: str) -> Path:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise DeploymentConfigError(f"{label}.parent must be an existing regular directory")
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise DeploymentConfigError(f"{label} must be an absent or regular directory")
    return path.resolve(strict=False)


def load_deployment_config(path: Path, *, sealed_bytes: bytes | None = None
                           ) -> DiscoveryDeployment:
    """Load one sealed JSON configuration without expanding its authority."""
    try:
        if sealed_bytes is None:
            if path.is_symlink() or not path.is_file():
                raise DeploymentConfigError(
                    "deployment configuration must be a regular file")
            payload = path.read_bytes()
        else:
            payload = sealed_bytes
        raw = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentConfigError("deployment configuration is not JSON") from exc
    top = _exact(raw, {"schema", "config_sha256", "production", "instrument", "controller", "actors", "gpu",
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
    production = _exact(top["production"], {"path", "branch", "head"}, "production")
    instrument = _exact(top["instrument"], {"repo_path", "branch", "commit",
                                               "production_ancestor"}, "instrument")
    production_path = _absolute(production["path"], "production.path")
    if production_path.is_symlink() or not production_path.is_dir():
        raise DeploymentConfigError("production.path must be a regular directory")
    if not isinstance(production["head"], str) or not GIT_SHA.fullmatch(production["head"]):
        raise DeploymentConfigError("production.head must be an exact Git SHA")
    if instrument["production_ancestor"] != production["head"]:
        raise DeploymentConfigError("instrument.production_ancestor must equal frozen production head")
    instrument_path = _absolute(instrument["repo_path"], "instrument.repo_path")
    if _overlaps(instrument_path, production_path):
        raise DeploymentConfigError("instrument.repo_path must be separate from frozen production")
    _verify_production(production_path, production["branch"], production["head"])
    _verify_instrument(instrument_path, production["head"], instrument["branch"],
                       instrument["commit"])
    controller = _exact(top["controller"], {"state_root", "evidence_root",
                                               "operations_root", "build_root", "max_iterations",
                                               "nomination_threshold"}, "controller")
    roots = {key: _validate_root(_absolute(controller[key], f"controller.{key}"), f"controller.{key}") for key in
             ("state_root", "evidence_root", "operations_root", "build_root")}
    if any(_overlaps(left, right) for index, left in enumerate(roots.values())
           for right in list(roots.values())[index + 1:]):
        raise DeploymentConfigError("controller roots must not overlap")
    if any(_overlaps(root, protected) for root in roots.values()
           for protected in (production_path, instrument_path)):
        raise DeploymentConfigError("controller output roots must not enter production or instrument repositories")
    max_iterations = controller["max_iterations"]
    threshold = controller["nomination_threshold"]
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 1000:
        raise DeploymentConfigError("controller.max_iterations is invalid")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold)) or threshold <= 0):
        raise DeploymentConfigError("controller.nomination_threshold is invalid")
    actors = _exact(top["actors"], {"wrapper_path", "wrapper_sha256",
                                      "critic_path", "critic_sha256",
                                      "environment_profile_id"}, "actors")
    actor_wrapper = _input({"path": actors["wrapper_path"], "sha256": actors["wrapper_sha256"]}, "actors.wrapper")
    if not os.access(actor_wrapper.path, os.X_OK):
        raise DeploymentConfigError("actors.wrapper_path must be executable")
    critic_wrapper = _input({"path": actors["critic_path"], "sha256": actors["critic_sha256"]}, "actors.critic")
    if not os.access(critic_wrapper.path, os.X_OK):
        raise DeploymentConfigError("actors.critic_path must be executable")
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
    inputs = _exact(top["immutable_inputs"], {
        "model", "workload", "runtime_config", "admission_policy",
        "hypothesis_portfolio", "hypothesis_evidence_manifest",
        "hypothesis_portfolio_contract", "preauthored_continuation",
        "q5_lds0_attribution_erratum", "carry_forward"},
        "immutable_inputs")
    source = _exact(top["source_plan"], {"source_builder_id", "evidence_plan_id",
                                           "runner_args_id", "experiment_template_registry_id", "experiment_template_registry_sha256",
                                           "production_snapshot_id"}, "source_plan")
    template_registry_sha256 = _digest_identifier(
        source["experiment_template_registry_sha256"],
        "source_plan.experiment_template_registry_sha256")
    model = _input(inputs["model"], "model")
    workload = _input(inputs["workload"], "workload")
    runtime_config = _input(inputs["runtime_config"], "runtime_config")
    admission_policy_input = _input(inputs["admission_policy"], "admission_policy")
    admission_policy = _admission_policy(admission_policy_input, model=model, workload=workload)
    portfolio_input = _portfolio(_input(inputs["hypothesis_portfolio"],
                                        "hypothesis_portfolio"))
    evidence_manifest = _evidence_manifest(
        _input(inputs["hypothesis_evidence_manifest"], "hypothesis_evidence_manifest"),
        portfolio=portfolio_input.value)
    portfolio_contract = _input(inputs["hypothesis_portfolio_contract"],
                                "hypothesis_portfolio_contract")
    continuation_input = _input(
        inputs["preauthored_continuation"], "preauthored_continuation")
    q5_erratum = _input(
        inputs["q5_lds0_attribution_erratum"],
        "q5_lds0_attribution_erratum")
    carry_forward = _carry_forward(inputs["carry_forward"])
    try:
        continuation = PreauthoredContinuationInput(
            continuation_input,
            preauthored_continuation.load(continuation_input.path))
    except preauthored_continuation.PreauthoredContinuationError as exc:
        raise DeploymentConfigError(
            "preauthored continuation schema/content mismatch") from exc
    planner_context = _planner_context(
        top["planner_context"], model=model, workload=workload,
        runtime_config=runtime_config, portfolio=portfolio_input,
        evidence_manifest=evidence_manifest, continuation=continuation)
    if planner_context.value["template_registry_sha256"] != template_registry_sha256:
        raise DeploymentConfigError(
            "planner_context template registry differs from source plan")
    for label, input_ in (("actors.wrapper", actor_wrapper), ("actors.critic", critic_wrapper),
                          ("model", model),
                          ("workload", workload), ("runtime_config", runtime_config),
                          ("admission_policy", admission_policy_input),
                          ("hypothesis_portfolio", portfolio_input.input),
                          ("hypothesis_evidence_manifest", evidence_manifest.input),
                          ("hypothesis_portfolio_contract", portfolio_contract),
                          ("preauthored_continuation", continuation.input),
                          ("q5_lds0_attribution_erratum", q5_erratum),
                          ("carry_forward", carry_forward.input),
                          ("planner_context", planner_context.input)):
        if any(_overlaps(input_.path, protected)
               for protected in (*roots.values(), production_path, instrument_path)):
            raise DeploymentConfigError(
                f"{label} location overlaps a mutable output or frozen production tree")
    return DiscoveryDeployment(
        config_sha256=top["config_sha256"], production_path=production_path,
        production_branch=production["branch"], production_head=production["head"],
        instrument_path=instrument_path, instrument_branch=instrument["branch"],
        instrument_commit=instrument["commit"],
        state_root=roots["state_root"], evidence_root=roots["evidence_root"],
        operations_root=roots["operations_root"], build_root=roots["build_root"],
        max_iterations=max_iterations,
        nomination_threshold=float(threshold), actor_wrapper=actor_wrapper,
        critic_wrapper=critic_wrapper,
        environment_profile_id=environment_profile_id, device_id=device_id,
        claim_timeout_s=float(claim_timeout_s), inference_window_lock=window,
        model=model, workload=workload, admission_policy=admission_policy,
        runtime_config=runtime_config, policy=admission_policy_input,
        hypothesis_portfolio=portfolio_input,
        hypothesis_evidence_manifest=evidence_manifest,
        preauthored_continuation=continuation,
        q5_lds0_attribution_erratum=q5_erratum,
        carry_forward=carry_forward,
        hypothesis_portfolio_contract=portfolio_contract,
        planner_context=planner_context,
        source_builder_id=_identifier(source["source_builder_id"], "source_plan.source_builder_id"),
        evidence_plan_id=_identifier(source["evidence_plan_id"], "source_plan.evidence_plan_id"),
        runner_args_id=_identifier(source["runner_args_id"], "source_plan.runner_args_id"),
        experiment_template_registry_id=_identifier(source["experiment_template_registry_id"], "source_plan.experiment_template_registry_id"),
        experiment_template_registry_sha256=template_registry_sha256,
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


__all__ = ["SCHEMA", "PLANNER_CONTEXT_SCHEMA", "EVIDENCE_MANIFEST_SCHEMA",
           "DeploymentConfigError", "ImmutableInput", "PlannerContext",
           "HypothesisPortfolioInput", "HypothesisEvidenceManifest",
           "PreauthoredContinuationInput", "CarryForwardInput", "DiscoveryDeployment",
           "ResolvedDeployment", "load_deployment_config", "resolve_registry"]
