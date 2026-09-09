"""Bind actor advice to real source/build inputs; never grant execution authority.

Callers own selected-stage serialization, durable execution admission and recovery.
A selected record/result is data, not a capability. Source mutation additionally
requires the existing private Worktree; build delegation requires the caller's
already-owned runner. This module creates no launcher, WAL, grant or build proof.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import json
import math
import weakref
from typing import Any, Mapping

from .. import source_candidate as source
from ..controller.discovery_controller import AuthoringAssignment
from ..evaluator import integrity
from ..execution import worktree
from . import actor_preparation as actor, campaign, campaign_control
from . import lifecycle_observation, unified_driver as driver, unified_planner

SCHEMA = "epyc.autokernel.bound_source_build_preparation.v1"
BUILD_SCHEMA_V2 = "epyc.autokernel.bound_source_build_preparation.v2"
_MATERIALIZED_SOURCE_TOKEN = object()
_MATERIALIZED_SOURCE_ISSUED = weakref.WeakKeyDictionary()


class PreparationBindingRefused(RuntimeError):
    pass


def _bytes(value: Any) -> bytes:
    return driver._canonical(driver._thaw(value))


def _manifest(value: Mapping[str, Any]) -> source.SourcePatchManifest:
    # The native constructor owns patch/scope/reward policy; no path is re-opened.
    if (set(value) != {"schema", "campaign_id", "proposal_id", "candidate_id", "source_tree",
                      "production_base_commit", "instrument_commit", "change_class",
                      "declared_files", "declared_symbols", "mechanism_id", "patch_sha256",
                      "patch_encoding", "patch_base64"}
            or value["schema"] != source.SCHEMA_SOURCE_PATCH
            or value["patch_encoding"] != "base64"):
        raise PreparationBindingRefused("authored manifest carrier differs")
    return source.SourcePatchManifest(
        campaign_id=value["campaign_id"], proposal_id=value["proposal_id"],
        candidate_id=value["candidate_id"], source_tree=value["source_tree"],
        production_base_commit=value["production_base_commit"],
        instrument_commit=value["instrument_commit"], change_class=value["change_class"],
        declared_files=tuple(value["declared_files"]),
        declared_symbols={key: tuple(item) for key, item in value["declared_symbols"].items()},
        mechanism_id=value["mechanism_id"], patch_sha256=value["patch_sha256"],
        patch_bytes=base64.b64decode(value["patch_base64"], validate=True))


def _common(selected, result, resolved, kind):
    if not isinstance(selected, driver.SelectedActorWork):
        raise PreparationBindingRefused("selected actor work must be typed")
    selected = driver.SelectedActorWork.from_dict(selected.to_dict())
    resolved = campaign.ResolvedCampaign.from_dict(resolved.to_dict())
    request = selected.actor_request
    if (request.actor_kind != kind or not isinstance(result, actor.PreparationResult)
            or result.status != "proposed" or result.proposed_output is None
            or result.failure_class is not None or result.retry_after is not None
            or result.request_digest != actor._digest(request.to_dict())
            or result.target_revision_digest != request.proposal.target_revision_digest
            or not result.actor_profile_digest):
        raise PreparationBindingRefused("accepted advice differs from selected actor request")
    driver._sha(result.actor_profile_digest, "accepted actor identity")
    if (selected.controller_binding["campaign_id"] != resolved.campaign_id
            or selected.controller_binding["config_digest"]
               != campaign_control.resolved_config_digest(resolved)
            or request.proposal.target_revision_digest not in {
                unified_planner._target_digest(target) for target in resolved.targets}):
        raise PreparationBindingRefused("selected advice differs from resolved campaign/target")
    advice = actor._validate_output(kind, _bytes(result.proposed_output).decode())
    return {"schema": SCHEMA, "kind": kind, "selected_work": selected.to_dict(),
            "resolved_campaign": resolved.to_dict(), "advice": advice,
            "actor_profile_digest": result.actor_profile_digest,
            "execution_authorized": False}


@dataclass(frozen=True)
class BoundSourcePreparation:
    """Canonical bytes own the input; native mutable mappings never escape."""

    canonical: bytes

    def __post_init__(self):
        if not isinstance(self.canonical, bytes):
            raise PreparationBindingRefused("source carrier must own immutable bytes")
        row = json.loads(self.canonical)
        if _bytes(row) != self.canonical:
            raise PreparationBindingRefused("source preparation is not canonical")
        _validate_source(row)

    @property
    def digest(self):
        return hashlib.sha256(self.canonical).hexdigest()

    def to_dict(self):
        return json.loads(self.canonical)


@dataclass(frozen=True, eq=False)
class MaterializedSourceCapability(Mapping[str, Any]):
    """Same-process proof that the owned source operation made these bytes."""

    canonical: bytes
    bound_digest: str
    target_revision_digest: str
    controller_binding: Mapping[str, Any]
    source_worktree: worktree.Worktree
    _token: object

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __post_init__(self):
        if (self._token is not _MATERIALIZED_SOURCE_TOKEN
                or not isinstance(self.canonical, bytes)
                or not isinstance(self.source_worktree, worktree.Worktree)):
            raise PreparationBindingRefused(
                "materialized source capability lacks its native issuer")
        row = json.loads(self.canonical)
        if (_bytes(row) != self.canonical
                or row.get("schema") != "epyc.autokernel.source_preparation_result.v1"
                or row.get("preparation_digest") != self.bound_digest):
            raise PreparationBindingRefused("materialized source receipt differs")
        driver._sha(self.target_revision_digest, "materialized source target")
        object.__setattr__(self, "controller_binding",
                           driver._freeze(driver._thaw(self.controller_binding)))

    def __getitem__(self, key):
        return driver._freeze(json.loads(self.canonical)[key])

    def __iter__(self):
        return iter(json.loads(self.canonical))

    def __len__(self):
        return len(json.loads(self.canonical))

    def to_dict(self):
        return json.loads(self.canonical)


def _validate_materialized_source(capability):
    if not isinstance(capability, MaterializedSourceCapability):
        raise PreparationBindingRefused(
            "candidate build lacks its exact materialized source capability")
    expected = _MATERIALIZED_SOURCE_ISSUED.get(capability)
    actual = (capability.canonical, capability.bound_digest,
              capability.target_revision_digest,
              driver._thaw(capability.controller_binding),
              capability.source_worktree)
    if expected is None or actual != expected:
        raise PreparationBindingRefused(
            "materialized source capability is not an original owner issuance")


def _validate_source(row):
    if set(row) != {"schema", "kind", "selected_work", "resolved_campaign", "advice",
                    "actor_profile_digest", "execution_authorized", "assignment",
                    "authored_manifest", "legacy_proposal"}:
        raise PreparationBindingRefused("source preparation fields differ")
    selected = _validate_common_row(row, "source")
    assignment = AuthoringAssignment(**row["assignment"])
    if assignment.portfolio_binding is not None:
        raise PreparationBindingRefused(
            "portfolio assignments require the existing full discovery-plan binding")
    manifest = _manifest(row["authored_manifest"])
    proposal = row["legacy_proposal"]
    manifest.bind(proposal=proposal, campaign_id=assignment.campaign_id,
                  candidate_id=assignment.candidate_id,
                  production_base_commit=assignment.production_base_commit,
                  instrument_commit=assignment.instrument_commit)
    resolved = campaign.ResolvedCampaign.from_dict(row["resolved_campaign"])
    advice = row["advice"]
    estimate = proposal.get("change", {}).get("estimated_diff_size")
    if (assignment.campaign_id != resolved.campaign_id
            or assignment.proposal_id != manifest.proposal_id
            or driver._pinned_source_revision(resolved) != manifest.instrument_commit
            or manifest.mechanism_id != selected.actor_request.proposal.mechanism_id
            or advice["mechanism"] != manifest.mechanism_id
            or manifest.declared_files != (advice["target_surface"],)
            or advice["target_symbol"] not in manifest.declared_symbols[advice["target_surface"]]
            or type(estimate) is not int or estimate < 1):
        raise PreparationBindingRefused("source advice/assignment/immutable artifact differs")
    if estimate < integrity.parse_unified_diff(manifest.patch_text).total_changed:
        raise PreparationBindingRefused("source patch exceeds the assigned diff-size bound")
    return manifest, proposal


def _validate_common_row(row, kind, *, schemas=(SCHEMA,)):
    if (row["schema"] not in schemas or row["kind"] != kind
            or row["execution_authorized"] is not False):
        raise PreparationBindingRefused("preparation schema/kind/authority differs")
    selected = driver.SelectedActorWork.from_dict(row["selected_work"])
    resolved = campaign.ResolvedCampaign.from_dict(row["resolved_campaign"])
    result = actor.PreparationResult(
        "proposed", actor._digest(selected.actor_request.to_dict()),
        selected.actor_request.proposal.target_revision_digest,
        row["actor_profile_digest"], row["advice"])
    _common(selected, result, resolved, kind)
    return selected


def bind_source_preparation(*, selected_actor_work, preparation_result, resolved_campaign,
                            assignment: AuthoringAssignment,
                            authored_manifest: source.SourcePatchManifest,
                            legacy_proposal: Mapping[str, Any]) -> BoundSourcePreparation:
    if not isinstance(assignment, AuthoringAssignment):
        raise PreparationBindingRefused("source assignment must be controller-owned typed input")
    if not isinstance(authored_manifest, source.SourcePatchManifest):
        raise PreparationBindingRefused("source preparation needs actual authored patch bytes")
    row = _common(selected_actor_work, preparation_result, resolved_campaign, "source")
    row.update(assignment=assignment.to_dict(),
               authored_manifest=json.loads(source.source_patch_manifest_bytes(authored_manifest)),
               legacy_proposal=json.loads(_bytes(legacy_proposal)))
    return BoundSourcePreparation(_bytes(row))


def _current_owner(selected, campaign_driver):
    if (not isinstance(campaign_driver, driver.UnifiedCampaignDriver)
            or not isinstance(campaign_driver.controller, campaign_control.CampaignController)):
        raise PreparationBindingRefused("preparation requires the actual current driver/controller")
    # Re-open the actual issued catalog through the public driver API. A caller
    # cannot substitute a self-consistent request/cache key inside selected advice.
    outcome = driver.DriverOutcome("intent_recorded", ("revalidate selected actor preparation",),
                                   selected.transition_id,
                                   selected.selection.to_dict())
    current = campaign_driver.materialize_actor(outcome)
    if current.to_dict() != selected.to_dict():
        raise PreparationBindingRefused("preparation differs from current selected actor work")


def materialize_source(bound: BoundSourcePreparation, *, campaign_driver,
                       actor_worktree: worktree.Worktree) -> MaterializedSourceCapability:
    """Apply only through an explicitly supplied existing private Worktree.

    The caller must serialize its selected preparation operation and own recovery.
    This preflight is not an atomic admission/restart lease or a build grant.
    """
    if not isinstance(bound, BoundSourcePreparation):
        raise PreparationBindingRefused("source preparation must be bound")
    row = bound.to_dict()
    manifest, proposal = _validate_source(row)
    selected = driver.SelectedActorWork.from_dict(row["selected_work"])
    _current_owner(selected, campaign_driver)
    if (not isinstance(actor_worktree, worktree.Worktree)
            or not actor_worktree.is_ancestor(
                manifest.production_base_commit, manifest.instrument_commit)):
        raise PreparationBindingRefused("source instrument does not descend from the pinned base")
    applied = source.apply_source_candidate(manifest, proposal=proposal, actor=actor_worktree)
    if (actor_worktree.head_commit() != applied.candidate_commit
            or not actor_worktree.is_clean()):
        raise PreparationBindingRefused(
            "materialized candidate commit/worktree is not exact and clean")
    snapshot = actor_worktree.snapshot_digest()
    receipt = {"schema": "epyc.autokernel.source_preparation_result.v1",
               "preparation_digest": bound.digest, "source_commit": applied.candidate_commit,
               "source_tree_digest": snapshot.sha256,
               "patch_bundle_sha256": manifest.patch_bundle_sha256,
               "diff_sha256": hashlib.sha256(applied.diff_text.encode()).hexdigest(),
               "actual_files": list(applied.actual_files),
               "actual_symbols": list(applied.actual_symbols),
               "actual_hunk_ids": list(applied.actual_hunk_ids),
               "diff_policy_checks": {name: {"outcome": check.outcome,
                                              "reasons": list(check.reasons)}
                                      for name, check in applied.diff_evidence.checks},
               "mutation_receipt": applied.mutation_receipt,
               "build_status": "pending", "execution_authorized": False}
    selected = driver.SelectedActorWork.from_dict(row["selected_work"])
    capability = MaterializedSourceCapability(
        _bytes(receipt), bound.digest,
        selected.actor_request.proposal.target_revision_digest,
        selected.controller_binding, actor_worktree, _MATERIALIZED_SOURCE_TOKEN)
    _MATERIALIZED_SOURCE_ISSUED[capability] = (
        capability.canonical, capability.bound_digest,
        capability.target_revision_digest,
        driver._thaw(capability.controller_binding), actor_worktree)
    return capability


@dataclass(frozen=True)
class BoundBuildPreparation:
    canonical: bytes
    plan: worktree.BuildPlan
    source_capability: MaterializedSourceCapability | None = None

    def __post_init__(self):
        if not isinstance(self.canonical, bytes):
            raise PreparationBindingRefused("build carrier must own immutable bytes")
        if not isinstance(self.plan, worktree.BuildPlan):
            raise PreparationBindingRefused("build preparation requires the existing BuildPlan")
        # Reconstruct both dataclasses; detach caller's tuple-compatible containers.
        plan = replace(self.plan, parallelism=replace(self.plan.parallelism),
                       targets=tuple(self.plan.targets),
                       cmake_defines=tuple(tuple(item) for item in self.plan.cmake_defines))
        if any(not isinstance(value, str) for entry in plan.cmake_defines for value in entry):
            raise PreparationBindingRefused("build definitions must be immutable string pairs")
        if (plan.parallelism.load_average_cap is not None
                and not math.isfinite(plan.parallelism.load_average_cap)):
            raise PreparationBindingRefused("explicit build load limit must be finite")
        row = json.loads(self.canonical)
        if (_bytes(row) != self.canonical
                or set(row) != {"schema", "kind", "selected_work", "resolved_campaign", "advice",
                                "actor_profile_digest", "execution_authorized",
                                "source_commit", "source_tree_digest", "recipe_digest", "build_plan"}
                or row["build_plan"] != plan.to_dict()):
            raise PreparationBindingRefused("build preparation plan/carrier differs")
        _validate_common_row(row, "build_recipe", schemas=(SCHEMA, BUILD_SCHEMA_V2))
        driver._sha(row["source_tree_digest"], "build source tree digest")
        resolved = campaign.ResolvedCampaign.from_dict(row["resolved_campaign"])
        cpus = (lifecycle_observation.parse_cpu_list(plan.parallelism.cpu_list)
                if plan.parallelism.cpu_list is not None else ())
        selected = driver.SelectedActorWork.from_dict(row["selected_work"])
        pinned_source = row["source_commit"] == driver._pinned_source_revision(resolved)
        capability = self.source_capability
        if capability is not None:
            _validate_materialized_source(capability)
        candidate_invalid = (not pinned_source and (
                row["schema"] != BUILD_SCHEMA_V2
                or not isinstance(capability, MaterializedSourceCapability)
                or capability.source_worktree.path.path != plan.source_root.path
                or capability["source_commit"] != row["source_commit"]
                or capability["source_tree_digest"] != row["source_tree_digest"]
                or capability.target_revision_digest
                   != selected.actor_request.proposal.target_revision_digest
                or driver._thaw(capability.controller_binding)
                   != driver._thaw(selected.controller_binding)))
        if ((pinned_source and (capability is not None or row["schema"] != SCHEMA))
                or candidate_invalid):
            raise PreparationBindingRefused(
                "candidate build lacks its exact materialized source capability")
        if (plan.parallelism.jobs > resolved.resources.build_jobs
                or not cpus or not set(cpus) <= set(resolved.resources.cpu_logical)
                or row["advice"]["build_system"] != "cmake"
                or row["advice"]["configured_options"] != [
                    f"-D{name}={value}" for name, value in plan.effective_defines]
                or len({name for name, _ in plan.cmake_defines}) != len(plan.cmake_defines)
                or row["recipe_digest"] != driver._digest(plan.to_dict())):
            raise PreparationBindingRefused("build advice differs from explicit recipe/source")
        object.__setattr__(self, "plan", plan)

    @property
    def digest(self):
        return hashlib.sha256(self.canonical).hexdigest()

    def to_dict(self):
        return json.loads(self.canonical)


def _source_snapshot(plan, source_worktree, source_commit, source_capability=None):
    if (not isinstance(source_worktree, worktree.Worktree)
            or (source_worktree.branch is not None
                and (not isinstance(source_capability, MaterializedSourceCapability)
                     or source_capability.source_worktree is not source_worktree))
            or source_worktree.path.path != plan.source_root.path
            or source_worktree.head_commit() != source_commit
            or not source_worktree.is_clean()):
        raise PreparationBindingRefused("build needs the exact clean detached source snapshot")
    return source_worktree.snapshot_digest().sha256


def bind_build_preparation(*, selected_actor_work, preparation_result, resolved_campaign,
                           source_commit: str, build_plan: worktree.BuildPlan,
                           source_worktree: worktree.Worktree,
                           materialized_source: MaterializedSourceCapability | None = None,
                           ) -> BoundBuildPreparation:
    if not isinstance(build_plan, worktree.BuildPlan):
        raise PreparationBindingRefused("an explicit typed build plan is required")
    row = _common(selected_actor_work, preparation_result, resolved_campaign, "build_recipe")
    if materialized_source is not None:
        _validate_materialized_source(materialized_source)
        row["schema"] = BUILD_SCHEMA_V2
    row.update(source_commit=source_commit,
               source_tree_digest=_source_snapshot(
                   build_plan, source_worktree, source_commit, materialized_source),
               recipe_digest=driver._digest(build_plan.to_dict()),
               build_plan=build_plan.to_dict())
    return BoundBuildPreparation(_bytes(row), build_plan, materialized_source)


def delegate_build(bound: BoundBuildPreparation, *, campaign_driver, runner, source_worktree,
                   log_path: str, configure_timeout_s: float, build_timeout_s: float,
                   env: Mapping[str, str], sandbox_cgroup_root: str):
    """Forward exact native run_build arguments to an explicitly owned runner.

    There is deliberately no default runner, invented permit, or success wrapper.
    Its original return/exception is preserved; build_identity remains upstream.
    """
    if not isinstance(bound, BoundBuildPreparation) or not callable(runner):
        raise PreparationBindingRefused("bound build and owned runner are required")
    # Validate again at the delegation boundary; never synthesize missing budgets.
    checked = BoundBuildPreparation(bound.canonical, bound.plan, bound.source_capability)
    selected = driver.SelectedActorWork.from_dict(checked.to_dict()["selected_work"])
    _current_owner(selected, campaign_driver)
    if _source_snapshot(checked.plan, source_worktree, checked.to_dict()["source_commit"],
                        checked.source_capability) \
            != checked.to_dict()["source_tree_digest"]:
        raise PreparationBindingRefused("build snapshot changed before delegation")
    for value in (configure_timeout_s, build_timeout_s):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value <= 0:
            raise PreparationBindingRefused("build deadlines must be finite and positive")
    resources = campaign.ResolvedCampaign.from_dict(
        checked.to_dict()["resolved_campaign"]).resources
    if configure_timeout_s + build_timeout_s > resources.build_timeout_s:
        raise PreparationBindingRefused("build deadlines exceed the campaign build budget")
    if (not isinstance(env, Mapping)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items())
            or not isinstance(log_path, str) or not log_path.startswith("/")
            or not isinstance(sandbox_cgroup_root, str)
            or not sandbox_cgroup_root.startswith("/")):
        raise PreparationBindingRefused("explicit build environment/log/containment is required")
    return runner(checked.plan, log_path=log_path, configure_timeout_s=configure_timeout_s,
                  build_timeout_s=build_timeout_s, env=dict(env),
                  require_fresh_build_dir=True, sandbox_cgroup_root=sandbox_cgroup_root)
