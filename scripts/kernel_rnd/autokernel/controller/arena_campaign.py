"""Fail-closed matched controller campaign for INF-03.

``arena_adapter`` proves that one command can cross the AgentKernelArena seam.
It intentionally does not prove that the command implements the controller
whose name appears in an ``ArenaTask``.  This module closes that bookkeeping
gap for the prospective MI210 comparison:

* the primary panel is exactly all seven registered authoring controllers plus
  the measured starting state (eight arms total);
* every controller cell receives the same task and 2 h / 8 h / 32 h wall-time
  checkpoints;
* ARGUS is included because it is registered, but remains unavailable until
  code availability, licence, gfx90a reachability, and an adapter are established; and
* execution is all-or-nothing.  A missing implementation produces a durable
  refusal receipt before any controller or GPU command can start.

The repository carries ``claude_codex_actor_critic`` plus governed adapters for
the pinned upstream EvoEngineer-Full, KernelFoundry, K-Search, Xe-Forge, and
GEAK-v1 implementations. ARGUS remains an exact refusal; a similarly named
command does not count as its implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

from . import (
    arena_adapter,
    claude_codex_actor_critic,
    evoengineer_arena,
    geak_v1_arena,
    kernelfoundry_arena,
    k_search_arena,
    xe_forge_arena,
)
from ..evaluator import rebench_scoring


CAMPAIGN_SCHEMA = "epyc.autokernel.arena_controller_campaign.v1"
AUDIT_SCHEMA = "epyc.autokernel.arena_controller_campaign_audit.v1"
AVAILABLE_SOURCE_AUDIT_SCHEMA = (
    "epyc.autokernel.arena_available_source_campaign_audit.v2")
BASELINE_ARM_ID = "starting_state_baseline"
PRIMARY_CONTROLLER_IDS = (
    "claude_codex_actor_critic",
    "evoengineer",
    "kernelfoundry",
    "k_search",
    "xe_forge",
    "geak_v1",
    "argus",
)
PRIMARY_PANEL_IDS = (BASELINE_ARM_ID, *PRIMARY_CONTROLLER_IDS)
AVAILABLE_SOURCE_PANEL_IDS = (
    BASELINE_ARM_ID, "claude_codex_actor_critic", "evoengineer", "kernelfoundry",
    "k_search", "xe_forge", "geak_v1")
AVAILABLE_SOURCE_EXCLUDED_IDS = ("argus",)
DISCOVERY_ONLY_CONTROLLER_IDS: tuple[str, ...] = ()
MATCHED_BUDGET_HOURS = rebench_scoring.DEFAULT_BUDGET_HOURS
IMPLEMENTATION_MODULE = Path(__file__).resolve()
REPOSITORY_ROOT = IMPLEMENTATION_MODULE.parents[4]
IN_TREE_SOURCE_ROOT = "repository://."
VENDOR_SOURCE_SCHEME = "vendor://"
DEFAULT_VENDOR_SOURCE_ROOT = Path(
    "/mnt/raid0/llm/autokernel/vendor/arena-controllers")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,95}")


class ArenaCampaignError(ValueError):
    """The matched panel is malformed, incomplete, or unsafe to execute."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArenaCampaignError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    digest = _text(value, label)
    if not _SHA256_RE.fullmatch(digest):
        raise ArenaCampaignError(f"{label} must be a lowercase SHA-256")
    return digest


def _positive_number(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ArenaCampaignError(f"{label} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class TaskArtifact:
    task_id: str
    relative_root: str
    file_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.task_id):
            raise ArenaCampaignError(f"invalid task_id {self.task_id!r}")
        root = Path(_text(self.relative_root, "task.relative_root"))
        if root.is_absolute() or ".." in root.parts:
            raise ArenaCampaignError("task.relative_root must stay Arena-relative")
        if not isinstance(self.file_sha256, Mapping) or not self.file_sha256:
            raise ArenaCampaignError("task.file_sha256 must be a non-empty object")
        for relative, digest in self.file_sha256.items():
            path = Path(_text(relative, "task artifact path"))
            if path.is_absolute() or ".." in path.parts:
                raise ArenaCampaignError("task artifact paths must stay task-relative")
            _sha256(digest, f"task artifact {relative!r}")


@dataclass(frozen=True)
class ArmImplementation:
    arm_id: str
    availability: str
    adapter_kind: str
    missing_artifacts: tuple[str, ...]
    argv: tuple[str, ...] = ()
    source_root: str | None = None
    source_commit: str | None = None
    entrypoint_path: str | None = None
    entrypoint_sha256: str | None = None
    model_ids: tuple[str, ...] = ()
    required_clis: tuple[str, ...] = ()
    upstream_source_root: str | None = None
    upstream_source_commit: str | None = None
    upstream_entrypoint_path: str | None = None
    upstream_entrypoint_sha256: str | None = None
    upstream_license_path: str | None = None
    upstream_license_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.arm_id not in PRIMARY_PANEL_IDS:
            raise ArenaCampaignError(f"unknown primary arm {self.arm_id!r}")
        if self.availability not in {"ready", "missing"}:
            raise ArenaCampaignError(
                f"{self.arm_id}: availability must be ready or missing")
        _text(self.adapter_kind, f"{self.arm_id}.adapter_kind")
        if (not isinstance(self.missing_artifacts, tuple)
                or any(not isinstance(item, str) or not item.strip()
                       for item in self.missing_artifacts)):
            raise ArenaCampaignError(
                f"{self.arm_id}: missing_artifacts must contain non-empty strings")
        if self.availability == "ready":
            if self.missing_artifacts:
                raise ArenaCampaignError(
                    f"{self.arm_id}: a ready arm cannot declare missing artifacts")
            if self.arm_id != BASELINE_ARM_ID and not self.argv:
                raise ArenaCampaignError(
                    f"{self.arm_id}: a ready controller requires an argv")
            if self.arm_id != BASELINE_ARM_ID:
                for field in ("source_root", "source_commit", "entrypoint_path",
                              "entrypoint_sha256"):
                    if getattr(self, field) is None:
                        raise ArenaCampaignError(
                            f"{self.arm_id}: a ready controller requires {field}")
                if not self.model_ids:
                    raise ArenaCampaignError(
                        f"{self.arm_id}: a ready controller requires explicit model_ids")
        elif not self.missing_artifacts:
            raise ArenaCampaignError(
                f"{self.arm_id}: a missing arm must name exact missing artifacts")
        if any(not isinstance(part, str) or not part for part in self.argv):
            raise ArenaCampaignError(f"{self.arm_id}: argv must contain non-empty strings")
        if any(not isinstance(model, str) or not model.strip() for model in self.model_ids):
            raise ArenaCampaignError(f"{self.arm_id}: model_ids must be non-empty strings")
        if (not isinstance(self.required_clis, tuple)
                or any(not isinstance(name, str) or not re.fullmatch(
                    r"[a-z][a-z0-9_-]{1,31}", name) for name in self.required_clis)
                or len(set(self.required_clis)) != len(self.required_clis)):
            raise ArenaCampaignError(
                f"{self.arm_id}: required_clis must be unique executable names")
        if self.source_commit is not None and not re.fullmatch(
                r"[0-9a-f]{40}", self.source_commit):
            raise ArenaCampaignError(
                f"{self.arm_id}: source_commit must be a full lowercase SHA-1")
        if self.entrypoint_sha256 is not None:
            _sha256(self.entrypoint_sha256, f"{self.arm_id}.entrypoint_sha256")
        upstream_fields = (
            "upstream_source_root", "upstream_source_commit",
            "upstream_entrypoint_path", "upstream_entrypoint_sha256",
            "upstream_license_path", "upstream_license_sha256",
        )
        upstream_values = [getattr(self, field) for field in upstream_fields]
        if any(value is not None for value in upstream_values):
            if any(value is None for value in upstream_values):
                raise ArenaCampaignError(
                    f"{self.arm_id}: upstream source identity must be complete")
            if not str(self.upstream_source_root).startswith(VENDOR_SOURCE_SCHEME):
                raise ArenaCampaignError(
                    f"{self.arm_id}: upstream source must use vendor://")
            if not re.fullmatch(r"[0-9a-f]{40}", str(self.upstream_source_commit)):
                raise ArenaCampaignError(
                    f"{self.arm_id}: upstream_source_commit must be a full SHA-1")
            _sha256(self.upstream_entrypoint_sha256,
                    f"{self.arm_id}.upstream_entrypoint_sha256")
            _sha256(self.upstream_license_sha256,
                    f"{self.arm_id}.upstream_license_sha256")
        if (self.availability == "missing"
                and self.arm_id == evoengineer_arena.CONTROLLER_ID
                and any(value is not None for value in upstream_values)):
            if self.adapter_kind != evoengineer_arena.PENDING_ADAPTER_KIND:
                raise ArenaCampaignError(
                    "the pending EvoEngineer arm must name its admitted policy seam")
            expected_upstream = (
                "vendor://evotoolkit", evoengineer_arena.SOURCE_COMMIT,
                evoengineer_arena.UPSTREAM_ENTRYPOINT,
                evoengineer_arena.EXPECTED_SOURCE_SHA256[
                    evoengineer_arena.UPSTREAM_ENTRYPOINT],
                "LICENSE", evoengineer_arena.EXPECTED_SOURCE_SHA256["LICENSE"],
            )
            if tuple(upstream_values) != expected_upstream:
                raise ArenaCampaignError(
                    "the pending EvoEngineer arm must bind the exact paper-era source")
        if (self.availability == "ready"
                and self.arm_id == evoengineer_arena.CONTROLLER_ID):
            expected_tail = evoengineer_arena.campaign_argv("python3")[1:]
            if self.adapter_kind != evoengineer_arena.ADAPTER_KIND:
                raise ArenaCampaignError(
                    "evoengineer requires its exact Full Arena adapter")
            if len(self.argv) < 2 or self.argv[1:] != expected_tail:
                raise ArenaCampaignError(
                    "evoengineer argv differs from its pinned executable")
            if self.entrypoint_path != evoengineer_arena.ENTRYPOINT_RELATIVE:
                raise ArenaCampaignError(
                    "evoengineer entrypoint differs from its implementation")
            if self.model_ids != evoengineer_arena.PINNED_MODEL_IDS:
                raise ArenaCampaignError(
                    "evoengineer model_ids differ from exact model/effort pins")
            if self.required_clis != evoengineer_arena.REQUIRED_CLIS:
                raise ArenaCampaignError("evoengineer requires the Codex CLI")
            expected_upstream = (
                "vendor://evoengineer", evoengineer_arena.SOURCE_COMMIT,
                evoengineer_arena.UPSTREAM_ENTRYPOINT,
                evoengineer_arena.EXPECTED_SOURCE_SHA256[
                    evoengineer_arena.UPSTREAM_ENTRYPOINT],
                "LICENSE", evoengineer_arena.EXPECTED_SOURCE_SHA256["LICENSE"],
            )
            if tuple(upstream_values) != expected_upstream:
                raise ArenaCampaignError(
                    "evoengineer must bind the exact paper-era source")
        if self.availability == "ready" and self.arm_id == claude_codex_actor_critic.CONTROLLER_ID:
            expected_tail = claude_codex_actor_critic.campaign_argv("python3")[1:]
            if self.adapter_kind != "agentkernelarena_three_arg_v1":
                raise ArenaCampaignError(
                    "claude_codex_actor_critic requires its exact three-argument adapter")
            if len(self.argv) < 2 or self.argv[1:] != expected_tail:
                raise ArenaCampaignError(
                    "claude_codex_actor_critic argv differs from its pinned executable")
            if self.entrypoint_path != claude_codex_actor_critic.ENTRYPOINT_RELATIVE:
                raise ArenaCampaignError(
                    "claude_codex_actor_critic entrypoint differs from its implementation")
            if self.model_ids != claude_codex_actor_critic.PINNED_MODEL_IDS:
                raise ArenaCampaignError(
                    "claude_codex_actor_critic model_ids differ from exact model/effort pins")
            if self.required_clis != claude_codex_actor_critic.REQUIRED_CLIS:
                raise ArenaCampaignError(
                    "claude_codex_actor_critic requires both installed CLIs")
        if self.availability == "ready" and self.arm_id == k_search_arena.CONTROLLER_ID:
            expected_tail = k_search_arena.campaign_argv("python3")[1:]
            if self.adapter_kind != "k_search_world_model_arena_v1":
                raise ArenaCampaignError(
                    "k_search requires its world-model Arena adapter")
            if len(self.argv) < 2 or self.argv[1:] != expected_tail:
                raise ArenaCampaignError(
                    "k_search argv differs from its pinned executable")
            if self.entrypoint_path != k_search_arena.ENTRYPOINT_RELATIVE:
                raise ArenaCampaignError(
                    "k_search entrypoint differs from its implementation")
            if self.model_ids != k_search_arena.PINNED_MODEL_IDS:
                raise ArenaCampaignError(
                    "k_search model_ids differ from exact model/effort pins")
            if self.required_clis != k_search_arena.REQUIRED_CLIS:
                raise ArenaCampaignError("k_search requires the Codex CLI")
            if self.upstream_source_commit != k_search_arena.SOURCE_COMMIT:
                raise ArenaCampaignError("k_search upstream source pin drifted")
        if (self.availability == "ready"
                and self.arm_id == kernelfoundry_arena.CONTROLLER_ID):
            expected_tail = kernelfoundry_arena.campaign_argv("python3")[1:]
            if self.adapter_kind != "kernelfoundry_map_elites_arena_v1":
                raise ArenaCampaignError(
                    "kernelfoundry requires its MAP-Elites Arena adapter")
            if len(self.argv) < 2 or self.argv[1:] != expected_tail:
                raise ArenaCampaignError(
                    "kernelfoundry argv differs from its pinned executable")
            if self.entrypoint_path != kernelfoundry_arena.ENTRYPOINT_RELATIVE:
                raise ArenaCampaignError(
                    "kernelfoundry entrypoint differs from its implementation")
            if self.model_ids != kernelfoundry_arena.PINNED_MODEL_IDS:
                raise ArenaCampaignError(
                    "kernelfoundry model_ids differ from exact model/effort pins")
            if self.required_clis != kernelfoundry_arena.REQUIRED_CLIS:
                raise ArenaCampaignError("kernelfoundry requires the Codex CLI")
            if self.upstream_source_commit != kernelfoundry_arena.SOURCE_COMMIT:
                raise ArenaCampaignError(
                    "kernelfoundry upstream source pin drifted")
        if self.availability == "ready" and self.arm_id == geak_v1_arena.CONTROLLER_ID:
            expected_tail = geak_v1_arena.campaign_argv("python3")[1:]
            if self.adapter_kind != "geak_v1_optimagent_arena_v1":
                raise ArenaCampaignError(
                    "geak_v1 requires its OptimAgent Arena adapter")
            if len(self.argv) < 2 or self.argv[1:] != expected_tail:
                raise ArenaCampaignError(
                    "geak_v1 argv differs from its pinned executable")
            if self.entrypoint_path != geak_v1_arena.ENTRYPOINT_RELATIVE:
                raise ArenaCampaignError(
                    "geak_v1 entrypoint differs from its implementation")
            if self.model_ids != geak_v1_arena.PINNED_MODEL_IDS:
                raise ArenaCampaignError(
                    "geak_v1 model_ids differ from exact model/effort pins")
            if self.required_clis != geak_v1_arena.REQUIRED_CLIS:
                raise ArenaCampaignError("geak_v1 requires the Codex CLI")
            if self.upstream_source_commit != geak_v1_arena.SOURCE_COMMIT:
                raise ArenaCampaignError("geak_v1 upstream source pin drifted")
        if self.availability == "ready" and self.arm_id == "xe_forge":
            expected_tail = xe_forge_arena.campaign_argv("python3")[1:]
            if self.adapter_kind != "xe_forge_linear_cover_arena_v1":
                raise ArenaCampaignError(
                    "xe_forge requires its linear-CoVeR Arena adapter")
            if len(self.argv) < 2 or self.argv[1:] != expected_tail:
                raise ArenaCampaignError(
                    "xe_forge argv differs from its pinned executable")
            if self.entrypoint_path != xe_forge_arena.ENTRYPOINT_RELATIVE:
                raise ArenaCampaignError(
                    "xe_forge entrypoint differs from its implementation")
            if self.model_ids != xe_forge_arena.PINNED_MODEL_IDS:
                raise ArenaCampaignError(
                    "xe_forge model_ids differ from exact model/effort pins")
            if self.required_clis != xe_forge_arena.REQUIRED_CLIS:
                raise ArenaCampaignError("xe_forge requires the Codex CLI")
            if self.upstream_source_commit != xe_forge_arena.SOURCE_COMMIT:
                raise ArenaCampaignError("xe_forge upstream source pin drifted")


@dataclass(frozen=True)
class CampaignSpec:
    config_path: str
    config_sha256: str
    campaign_id: str
    target_gpu_model: str
    target_gfx_arch: str
    budget_hours: tuple[float, ...]
    tasks: tuple[TaskArtifact, ...]
    arms: tuple[ArmImplementation, ...]
    out_of_panel_registered: tuple[str, ...]

    def __post_init__(self) -> None:
        config = Path(self.config_path)
        if not config.is_absolute() or not config.is_file():
            raise ArenaCampaignError("config_path must be an existing absolute file")
        _sha256(self.config_sha256, "config_sha256")
        if not _ID_RE.fullmatch(self.campaign_id):
            raise ArenaCampaignError(f"invalid campaign_id {self.campaign_id!r}")
        if self.target_gpu_model != arena_adapter.TARGET_GPU_MODEL:
            raise ArenaCampaignError("campaign target must be the physical MI210")
        if self.target_gfx_arch != arena_adapter.TARGET_GFX_ARCH:
            raise ArenaCampaignError("campaign architecture must be gfx90a")
        if self.budget_hours != MATCHED_BUDGET_HOURS:
            raise ArenaCampaignError(
                f"matched checkpoints must be exactly {MATCHED_BUDGET_HOURS}")
        if not self.tasks:
            raise ArenaCampaignError("campaign requires at least one exact task")
        observed = tuple(arm.arm_id for arm in self.arms)
        if observed != PRIMARY_PANEL_IDS:
            raise ArenaCampaignError(
                f"primary panel must be exactly {PRIMARY_PANEL_IDS}; observed {observed}")
        if self.out_of_panel_registered != DISCOVERY_ONLY_CONTROLLER_IDS:
            raise ArenaCampaignError(
                "no registered controller may remain outside the eight-arm panel")


def load_spec(path: str | Path) -> CampaignSpec:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaCampaignError(f"cannot read campaign config: {source}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != CAMPAIGN_SCHEMA:
        raise ArenaCampaignError(f"campaign schema must be {CAMPAIGN_SCHEMA}")
    budget = payload.get("budget")
    if not isinstance(budget, Mapping):
        raise ArenaCampaignError("budget must be an object")
    if budget.get("match_axis") != "elapsed_wall_time":
        raise ArenaCampaignError("the only admitted match axis is elapsed_wall_time")
    checkpoints = tuple(
        _positive_number(value, "budget.checkpoint_hours")
        for value in budget.get("checkpoint_hours", ()))
    maximum = _positive_number(
        budget.get("maximum_wall_hours_per_controller_task"),
        "budget.maximum_wall_hours_per_controller_task")
    if not checkpoints or maximum != checkpoints[-1]:
        raise ArenaCampaignError("maximum wall hours must equal the last checkpoint")
    if budget.get("concurrent_gpu_cells") != 1:
        raise ArenaCampaignError("one MI210 permits exactly one GPU cell at a time")
    task_rows = payload.get("tasks")
    arm_rows = payload.get("arms")
    if not isinstance(task_rows, list) or not isinstance(arm_rows, list):
        raise ArenaCampaignError("tasks and arms must be arrays")
    tasks = tuple(TaskArtifact(
        task_id=_text(row.get("task_id"), "task_id"),
        relative_root=_text(row.get("relative_root"), "task.relative_root"),
        file_sha256=dict(row.get("file_sha256", {})),
    ) for row in task_rows if isinstance(row, Mapping))
    if len(tasks) != len(task_rows):
        raise ArenaCampaignError("every task row must be an object")
    arms = tuple(ArmImplementation(
        arm_id=_text(row.get("arm_id"), "arm_id"),
        availability=_text(row.get("availability"), "arm.availability"),
        adapter_kind=_text(row.get("adapter_kind"), "arm.adapter_kind"),
        missing_artifacts=tuple(row.get("missing_artifacts", ())),
        argv=tuple(row.get("argv", ())),
        source_root=row.get("source_root"),
        source_commit=row.get("source_commit"),
        entrypoint_path=row.get("entrypoint_path"),
        entrypoint_sha256=row.get("entrypoint_sha256"),
        model_ids=tuple(row.get("model_ids", ())),
        required_clis=tuple(row.get("required_clis", ())),
        upstream_source_root=row.get("upstream_source_root"),
        upstream_source_commit=row.get("upstream_source_commit"),
        upstream_entrypoint_path=row.get("upstream_entrypoint_path"),
        upstream_entrypoint_sha256=row.get("upstream_entrypoint_sha256"),
        upstream_license_path=row.get("upstream_license_path"),
        upstream_license_sha256=row.get("upstream_license_sha256"),
    ) for row in arm_rows if isinstance(row, Mapping))
    if len(arms) != len(arm_rows):
        raise ArenaCampaignError("every arm row must be an object")
    return CampaignSpec(
        config_path=str(source),
        config_sha256=_sha256_file(source),
        campaign_id=_text(payload.get("campaign_id"), "campaign_id"),
        target_gpu_model=_text(payload.get("target_gpu_model"), "target_gpu_model"),
        target_gfx_arch=_text(payload.get("target_gfx_arch"), "target_gfx_arch"),
        budget_hours=checkpoints,
        tasks=tasks,
        arms=arms,
        out_of_panel_registered=tuple(payload.get("out_of_panel_registered", ())),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_audit(arena_root: Path, task: TaskArtifact) -> dict[str, Any]:
    root = arena_root / task.relative_root
    failures: list[str] = []
    observed: dict[str, str | None] = {}
    if not root.is_dir():
        failures.append(f"missing task root: {task.relative_root}")
    for relative, expected in sorted(task.file_sha256.items()):
        path = root / relative
        digest = _sha256_file(path) if path.is_file() else None
        observed[relative] = digest
        if digest != expected:
            failures.append(
                f"{task.relative_root}/{relative}: expected {expected}, observed {digest}")
    return {
        "task_id": task.task_id,
        "relative_root": task.relative_root,
        "ready": not failures,
        "observed_file_sha256": observed,
        "failures": failures,
    }


def _resolve_source_root(value: str | None) -> tuple[Path, bool]:
    if value == IN_TREE_SOURCE_ROOT:
        return REPOSITORY_ROOT, True
    if isinstance(value, str) and value.startswith(VENDOR_SOURCE_SCHEME):
        relative = Path(value.removeprefix(VENDOR_SOURCE_SCHEME))
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                or len(relative.parts) != 1):
            raise ArenaCampaignError("vendor source locator must name one checkout")
        base = Path(os.environ.get(
            "AUTOKERNEL_ARENA_CONTROLLER_ROOT", str(DEFAULT_VENDOR_SOURCE_ROOT)))
        return (base / relative).resolve(), False
    return Path(value or "").resolve(), False


def _upstream_source_audit(arm: ArmImplementation) -> tuple[dict[str, Any] | None,
                                                             list[str]]:
    if arm.upstream_source_root is None:
        return None, []
    failures: list[str] = []
    source, in_tree = _resolve_source_root(arm.upstream_source_root)
    if in_tree:
        failures.append("upstream controller source cannot alias the adapter tree")
    paths = {
        "entrypoint": (arm.upstream_entrypoint_path,
                       arm.upstream_entrypoint_sha256),
        "license": (arm.upstream_license_path, arm.upstream_license_sha256),
    }
    observed_commit = None
    observed_dirty = None
    observed_files: dict[str, Any] = {}
    if not source.is_dir():
        failures.append(f"upstream source root does not exist: {source}")
    else:
        try:
            commit_run = subprocess.run(
                ("git", "-C", str(source), "rev-parse", "HEAD"),
                capture_output=True, text=True, check=False, timeout=30)
            status_run = subprocess.run(
                ("git", "-C", str(source), "status", "--porcelain=v1",
                 "--untracked-files=all"),
                capture_output=True, text=True, check=False, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"upstream source identity command failed: {exc}")
        else:
            if commit_run.returncode != 0 or status_run.returncode != 0:
                failures.append("upstream source is not a readable git checkout")
            else:
                observed_commit = commit_run.stdout.strip()
                observed_dirty = bool(status_run.stdout.strip())
                if observed_commit != arm.upstream_source_commit:
                    failures.append(
                        f"upstream source expected {arm.upstream_source_commit}, "
                        f"observed {observed_commit}")
                if observed_dirty:
                    failures.append("upstream source checkout is not clean")
        for label, (relative, expected) in paths.items():
            path = (source / str(relative or "")).resolve()
            try:
                path.relative_to(source)
            except ValueError:
                failures.append(f"upstream {label} escapes its source checkout")
                observed = None
            else:
                observed = _sha256_file(path) if path.is_file() else None
            observed_files[label] = {
                "path": relative, "expected_sha256": expected,
                "observed_sha256": observed,
            }
            if observed != expected:
                failures.append(
                    f"upstream {label} expected {expected}, observed {observed}")
    return {
        "root": str(source),
        "expected_commit": arm.upstream_source_commit,
        "observed_commit": observed_commit,
        "clean": False if observed_dirty else (
            True if observed_dirty is not None else None),
        "files": observed_files,
    }, failures


def _implementation_audit(arm: ArmImplementation) -> dict[str, Any]:
    failures = list(arm.missing_artifacts)
    executable_path: str | None = None
    executable_sha256: str | None = None
    source_identity: dict[str, Any] | None = None
    upstream_source_identity: dict[str, Any] | None = None
    cli_identities: list[dict[str, Any]] = []
    if arm.availability == "ready" and arm.arm_id != BASELINE_ARM_ID:
        executable = shutil.which(arm.argv[0])
        if executable is None:
            failures.append(f"controller executable not found: {arm.argv[0]}")
        else:
            resolved = Path(executable).resolve()
            executable_path = str(resolved)
            executable_sha256 = _sha256_file(resolved)
        for name in arm.required_clis:
            cli = shutil.which(name)
            if cli is None:
                failures.append(f"required controller CLI not found: {name}")
                cli_identities.append({"name": name, "available": False})
            else:
                resolved_cli = Path(cli).resolve()
                if not resolved_cli.is_file() or not os.access(resolved_cli, os.X_OK):
                    failures.append(f"required controller CLI is not executable: {name}")
                    cli_identities.append({"name": name, "available": False})
                else:
                    cli_identities.append({
                        "name": name, "available": True, "path": str(resolved_cli),
                        "sha256": _sha256_file(resolved_cli),
                    })
        source, in_tree = _resolve_source_root(arm.source_root)
        entrypoint = (source / (arm.entrypoint_path or "")).resolve()
        try:
            entrypoint.relative_to(source)
        except ValueError:
            failures.append("controller entrypoint escapes its source checkout")
        observed_commit = None
        observed_dirty = None
        pin_relation = None
        pinned_entrypoint_sha256 = None
        if not source.is_dir():
            failures.append(f"controller source root does not exist: {source}")
        else:
            try:
                commit_run = subprocess.run(
                    ("git", "-C", str(source), "rev-parse", "HEAD"),
                    capture_output=True, text=True, check=False, timeout=30)
                status_run = subprocess.run(
                    ("git", "-C", str(source), "status", "--porcelain=v1",
                     "--untracked-files=all"),
                    capture_output=True, text=True, check=False, timeout=30)
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(f"controller source identity command failed: {exc}")
            else:
                if commit_run.returncode != 0 or status_run.returncode != 0:
                    failures.append("controller source is not a readable git checkout")
                else:
                    observed_commit = commit_run.stdout.strip()
                    observed_dirty = bool(status_run.stdout.strip())
                    if in_tree:
                        ancestor_run = subprocess.run(
                            ("git", "-C", str(source), "merge-base", "--is-ancestor",
                             str(arm.source_commit), observed_commit),
                            capture_output=True, text=True, check=False, timeout=30)
                        pin_relation = "ancestor" if ancestor_run.returncode == 0 else None
                        if pin_relation is None:
                            failures.append(
                                f"controller source pin {arm.source_commit} is not an "
                                f"ancestor of {observed_commit}")
                        pinned_run = subprocess.run(
                            ("git", "-C", str(source), "show",
                             f"{arm.source_commit}:{arm.entrypoint_path}"),
                            capture_output=True, check=False, timeout=30)
                        if pinned_run.returncode == 0:
                            pinned_entrypoint_sha256 = hashlib.sha256(
                                pinned_run.stdout).hexdigest()
                        if pinned_entrypoint_sha256 != arm.entrypoint_sha256:
                            failures.append(
                                "controller entrypoint digest is not present at its source pin")
                    elif observed_commit != arm.source_commit:
                        failures.append(
                            f"controller source expected {arm.source_commit}, "
                            f"observed {observed_commit}")
                    else:
                        pin_relation = "exact"
                    if observed_dirty:
                        failures.append("controller source checkout is not clean")
        observed_entrypoint = _sha256_file(entrypoint) if entrypoint.is_file() else None
        if observed_entrypoint != arm.entrypoint_sha256:
            failures.append(
                f"controller entrypoint expected {arm.entrypoint_sha256}, "
                f"observed {observed_entrypoint}")
        source_identity = {
            "root": str(source),
            "expected_commit": arm.source_commit,
            "observed_commit": observed_commit,
            "clean": False if observed_dirty else (True if observed_dirty is not None else None),
            "entrypoint_path": arm.entrypoint_path,
            "expected_entrypoint_sha256": arm.entrypoint_sha256,
            "observed_entrypoint_sha256": observed_entrypoint,
            "pinned_entrypoint_sha256": pinned_entrypoint_sha256,
            "pin_relation": pin_relation,
        }
        upstream_source_identity, upstream_failures = _upstream_source_audit(arm)
        failures.extend(upstream_failures)
    return {
        "arm_id": arm.arm_id,
        "adapter_kind": arm.adapter_kind,
        "declared_availability": arm.availability,
        "executable": not failures,
        "argv": list(arm.argv),
        "executable_path": executable_path,
        "executable_sha256": executable_sha256,
        "source_identity": source_identity,
        "upstream_source_identity": upstream_source_identity,
        "model_ids": list(arm.model_ids),
        "required_cli_identities": cli_identities,
        "missing_artifacts": failures,
    }


def _receipt_hash(receipt: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(receipt), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cli_inventory() -> list[dict[str, Any]]:
    """Record locally visible upstream CLIs without invoking any CLI."""
    rows = []
    for name in ("claude", "codex", "cursor", "geak"):
        executable = shutil.which(name)
        digest = None
        resolved = None
        if executable is not None:
            path = Path(executable).resolve()
            resolved = str(path)
            digest = _sha256_file(path)
        rows.append({
            "name": name,
            "available": executable is not None,
            "path": resolved,
            "sha256": digest,
            "version": None,
            "version_probe_executed": False,
            "implementation_coverage_implied": False,
        })
    return rows


def audit_campaign(
    spec: CampaignSpec, *, arena_root: str | Path, geak_root: str | Path,
    enumerator: str = "/opt/rocm/bin/rocm_agent_enumerator",
    inspect_hardware: bool = True,
) -> dict[str, Any]:
    """Audit the complete panel without launching a controller or GPU kernel."""
    arena_path = Path(arena_root).resolve()
    geak_path = Path(geak_root).resolve()
    failures: list[str] = []
    config_path = Path(spec.config_path)
    observed_config_sha256 = (
        _sha256_file(config_path) if config_path.is_file() else None)
    if observed_config_sha256 != spec.config_sha256:
        failures.append(
            f"campaign config expected {spec.config_sha256}, "
            f"observed {observed_config_sha256}")
    implementation_sha256 = _sha256_file(IMPLEMENTATION_MODULE)
    sources: dict[str, Any] = {}
    for key, root, pin in (
        ("agent_kernel_arena", arena_path, arena_adapter.AGENT_KERNEL_ARENA_PIN),
        ("geak_v1", geak_path, arena_adapter.GEAK_V1_PIN),
    ):
        try:
            sources[key] = arena_adapter.inspect_vendor_source(root, pin)
        except arena_adapter.ArenaAdapterError as exc:
            failures.append(str(exc))
            sources[key] = {"ready": False, "error": str(exc)}
    expected_registered = set(PRIMARY_CONTROLLER_IDS) | set(DISCOVERY_ONLY_CONTROLLER_IDS)
    observed_registered = set(arena_adapter.CONTROLLERS)
    if observed_registered != expected_registered:
        failures.append(
            "controller registry drift: expected "
            f"{sorted(expected_registered)}, observed {sorted(observed_registered)}")
    hardware: dict[str, Any]
    if inspect_hardware:
        try:
            hardware = arena_adapter.detect_gfx_arch(enumerator)
        except arena_adapter.ArenaAdapterError as exc:
            failures.append(str(exc))
            hardware = {"ready": False, "error": str(exc)}
    else:
        failures.append("physical gfx90a inspection was skipped")
        hardware = {
            "ready": None,
            "inspection_skipped": True,
            "target_gpu_model": spec.target_gpu_model,
            "target_gfx_arch": spec.target_gfx_arch,
        }
    task_rows = [_task_audit(arena_path, task) for task in spec.tasks]
    arm_rows = [_implementation_audit(arm) for arm in spec.arms]
    for row in task_rows:
        failures.extend(row["failures"])
    executable_arms = sum(bool(row["executable"]) for row in arm_rows)
    if executable_arms != len(PRIMARY_PANEL_IDS):
        failures.append(
            f"primary panel implementation coverage is {executable_arms}/"
            f"{len(PRIMARY_PANEL_IDS)}; partial panels are forbidden")
    for row in arm_rows:
        failures.extend(
            f"{row['arm_id']}: {item}" for item in row["missing_artifacts"])
    receipt: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "campaign_id": spec.campaign_id,
        "status": "ready" if not failures else "refused",
        "authority": "diagnostic_only",
        "target": {
            "gpu_model": spec.target_gpu_model,
            "gfx_arch": spec.target_gfx_arch,
        },
        "execution_identity": {
            "config_path": str(config_path),
            "config_sha256": spec.config_sha256,
            "observed_config_sha256": observed_config_sha256,
            "implementation_module": str(IMPLEMENTATION_MODULE),
            "implementation_module_sha256": implementation_sha256,
        },
        "matched_budget": {
            "match_axis": "elapsed_wall_time",
            "checkpoint_hours": list(spec.budget_hours),
            "maximum_wall_hours_per_controller_task": spec.budget_hours[-1],
            "concurrent_gpu_cells": 1,
        },
        "panel": {
            "arm_count": len(PRIMARY_PANEL_IDS),
            "baseline_arm_id": BASELINE_ARM_ID,
            "baseline_semantics": "arena_measured_starting_state_no_authoring",
            "primary_arm_ids": list(PRIMARY_PANEL_IDS),
            "registered_out_of_panel": list(spec.out_of_panel_registered),
            "executable_arm_count": executable_arms,
            "arms": arm_rows,
        },
        "tasks": task_rows,
        "sources": sources,
        "hardware": hardware,
        "host_cli_inventory": _cli_inventory(),
        "refusal_reasons": failures,
        "constraints": {
            "all_or_nothing_execution": True,
            "partial_results_rankable": False,
            "controller_or_gpu_command_executed": False,
            "argus_result_transfer": False,
        },
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return receipt


def audit_available_source_campaign(
    spec: CampaignSpec, *, arena_root: str | Path, geak_root: str | Path,
    enumerator: str = "/opt/rocm/bin/rocm_agent_enumerator",
    inspect_hardware: bool = True,
) -> dict[str, Any]:
    """Audit a separately labelled seven-arm available-source diagnostic panel."""
    full = audit_campaign(
        spec, arena_root=arena_root, geak_root=geak_root,
        enumerator=enumerator, inspect_hardware=inspect_hardware)
    arm_rows = {row["arm_id"]: row for row in full["panel"]["arms"]}
    selected = [arm_rows[arm_id] for arm_id in AVAILABLE_SOURCE_PANEL_IDS]
    excluded = [arm_rows[arm_id] for arm_id in AVAILABLE_SOURCE_EXCLUDED_IDS]
    failures: list[str] = []
    if tuple(arm_rows) != PRIMARY_PANEL_IDS:
        failures.append("parent eight-arm ordering drifted")
    if not all(row["executable"] for row in selected):
        failures.append("one or more available-source arms are not executable")
    if any(row["executable"] for row in excluded):
        failures.append(
            "an externally excluded arm became executable; refresh the panel")
    expected_full_refusals = {
        (f"primary panel implementation coverage is "
         f"{len(AVAILABLE_SOURCE_PANEL_IDS)}/{len(PRIMARY_PANEL_IDS)}; "
         "partial panels are forbidden")}
    for row in excluded:
        expected_full_refusals.update(
            f"{row['arm_id']}: {item}" for item in row["missing_artifacts"])
    unexpected = sorted(
        set(full["refusal_reasons"]) - expected_full_refusals)
    absent = sorted(
        expected_full_refusals - set(full["refusal_reasons"]))
    if unexpected:
        failures.extend(f"parent audit: {item}" for item in unexpected)
    if absent:
        failures.extend(f"expected parent refusal absent: {item}" for item in absent)
    campaign_id = f"{spec.campaign_id}-available-source-seven-arm-v2"
    receipt: dict[str, Any] = {
        "schema": AVAILABLE_SOURCE_AUDIT_SCHEMA,
        "campaign_id": campaign_id,
        "status": "ready" if not failures else "refused",
        "authority": "availability_conditioned_diagnostic_only",
        "parent_eight_arm_campaign": {
            "campaign_id": spec.campaign_id,
            "audit_status": full["status"],
            "audit_receipt_sha256": full["receipt_sha256"],
            "full_panel_claim_permitted": False,
        },
        "execution_identity": full["execution_identity"],
        "target": full["target"],
        "matched_budget": full["matched_budget"],
        "panel": {
            "arm_count": len(AVAILABLE_SOURCE_PANEL_IDS),
            "baseline_arm_id": BASELINE_ARM_ID,
            "primary_arm_ids": list(AVAILABLE_SOURCE_PANEL_IDS),
            "executable_arm_count": sum(
                bool(row["executable"]) for row in selected),
            "arms": selected,
            "externally_excluded_arms": excluded,
        },
        "tasks": full["tasks"],
        "sources": full["sources"],
        "hardware": full["hardware"],
        "host_cli_inventory": full["host_cli_inventory"],
        "refusal_reasons": failures,
        "constraints": {
            "availability_conditioned_only": True,
            "full_eight_arm_result_implied": False,
            "partial_full_panel_results_rankable": False,
            "controller_or_gpu_command_executed": False,
            "promotion_authority": False,
        },
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return receipt


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    rendered = json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


@dataclass(frozen=True)
class CampaignCellRequest:
    """One matched cell request delivered to the governed Arena executor."""

    arm: ArmImplementation
    task: TaskArtifact
    is_starting_state_baseline: bool
    checkpoint_hours: tuple[float, ...]
    maximum_wall_hours: float

    def __post_init__(self) -> None:
        if not isinstance(self.arm, ArmImplementation):
            raise TypeError("cell arm must be an ArmImplementation")
        if not isinstance(self.task, TaskArtifact):
            raise TypeError("cell task must be a TaskArtifact")
        if not isinstance(self.is_starting_state_baseline, bool):
            raise TypeError("is_starting_state_baseline must be boolean")
        expected_baseline = self.arm.arm_id == BASELINE_ARM_ID
        if self.is_starting_state_baseline != expected_baseline:
            raise ArenaCampaignError("cell baseline semantics disagree with its arm")
        if expected_baseline:
            if self.checkpoint_hours or self.maximum_wall_hours != 0.0:
                raise ArenaCampaignError(
                    "starting-state baseline receives no authoring checkpoints or wall budget")
        else:
            if self.checkpoint_hours != MATCHED_BUDGET_HOURS:
                raise ArenaCampaignError(
                    f"controller cell must receive every checkpoint {MATCHED_BUDGET_HOURS}")
            if self.maximum_wall_hours != MATCHED_BUDGET_HOURS[-1]:
                raise ArenaCampaignError(
                    "controller cell ceiling must equal the final matched checkpoint")


def execute_campaign(
    spec: CampaignSpec, audit: Mapping[str, Any], *,
    run_cell: Callable[["CampaignCellRequest"], Any],
) -> list[Any]:
    """Execute a complete preflighted matrix through an injected governed cell runner.

    The actual AgentKernelArena runner supplies ``run_cell``.  Keeping that process
    boundary explicit prevents this development module from duplicating the vendor
    evaluator or silently turning a metadata registration into an implementation.
    """
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("campaign_id") != spec.campaign_id:
        raise ArenaCampaignError("audit does not bind this campaign")
    if audit.get("status") != "ready":
        raise ArenaCampaignError("campaign audit refused; no cell may execute")
    identity = audit.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise ArenaCampaignError("audit lacks execution identity")
    current_config_sha256 = _sha256_file(Path(spec.config_path))
    if (identity.get("config_sha256") != spec.config_sha256
            or identity.get("observed_config_sha256") != current_config_sha256
            or current_config_sha256 != spec.config_sha256):
        raise ArenaCampaignError("campaign config changed after audit")
    current_module_sha256 = _sha256_file(IMPLEMENTATION_MODULE)
    if identity.get("implementation_module_sha256") != current_module_sha256:
        raise ArenaCampaignError("campaign implementation module changed after audit")
    if not callable(run_cell):
        raise TypeError("run_cell must be callable")
    results = []
    for task in spec.tasks:
        for arm in spec.arms:
            is_baseline = arm.arm_id == BASELINE_ARM_ID
            request = CampaignCellRequest(
                arm=arm,
                task=task,
                is_starting_state_baseline=is_baseline,
                checkpoint_hours=() if is_baseline else spec.budget_hours,
                maximum_wall_hours=0.0 if is_baseline else spec.budget_hours[-1],
            )
            results.append(run_cell(request))
    return results


def execute_available_source_campaign(
    spec: CampaignSpec, audit: Mapping[str, Any], *,
    run_cell: Callable[["CampaignCellRequest"], Any],
) -> list[Any]:
    """Execute only the explicitly labelled available-source seven-arm panel."""
    expected_id = f"{spec.campaign_id}-available-source-seven-arm-v2"
    if (audit.get("schema") != AVAILABLE_SOURCE_AUDIT_SCHEMA
            or audit.get("campaign_id") != expected_id):
        raise ArenaCampaignError("audit does not bind the available-source panel")
    if audit.get("status") != "ready":
        raise ArenaCampaignError(
            "available-source campaign audit refused; no cell may execute")
    identity = audit.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise ArenaCampaignError("available-source audit lacks execution identity")
    if (_sha256_file(Path(spec.config_path)) != spec.config_sha256
            or identity.get("config_sha256") != spec.config_sha256
            or identity.get("implementation_module_sha256")
            != _sha256_file(IMPLEMENTATION_MODULE)):
        raise ArenaCampaignError(
            "available-source execution identity changed after audit")
    arms = {arm.arm_id: arm for arm in spec.arms}
    results = []
    for task in spec.tasks:
        for arm_id in AVAILABLE_SOURCE_PANEL_IDS:
            arm = arms[arm_id]
            baseline = arm_id == BASELINE_ARM_ID
            results.append(run_cell(CampaignCellRequest(
                arm=arm, task=task, is_starting_state_baseline=baseline,
                checkpoint_hours=() if baseline else spec.budget_hours,
                maximum_wall_hours=(0.0 if baseline
                                    else spec.budget_hours[-1]))))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arena-root", required=True)
    parser.add_argument("--geak-root", required=True)
    parser.add_argument("--enumerator", default="/opt/rocm/bin/rocm_agent_enumerator")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--skip-hardware-inspection", action="store_true",
        help="development-only: retain hardware as unchecked and never report ready",
    )
    args = parser.parse_args(argv)
    spec = load_spec(args.config)
    receipt = audit_campaign(
        spec,
        arena_root=args.arena_root,
        geak_root=args.geak_root,
        enumerator=args.enumerator,
        inspect_hardware=not args.skip_hardware_inspection,
    )
    output = write_receipt(args.output, receipt)
    print(json.dumps({
        "output": str(output),
        "status": receipt["status"],
        "executable_arm_count": receipt["panel"]["executable_arm_count"],
        "arm_count": receipt["panel"]["arm_count"],
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))
    return 0 if receipt["status"] == "ready" else 3


__all__ = [
    "AUDIT_SCHEMA", "AVAILABLE_SOURCE_AUDIT_SCHEMA",
    "AVAILABLE_SOURCE_EXCLUDED_IDS", "AVAILABLE_SOURCE_PANEL_IDS",
    "BASELINE_ARM_ID", "CAMPAIGN_SCHEMA",
    "DISCOVERY_ONLY_CONTROLLER_IDS", "MATCHED_BUDGET_HOURS",
    "PRIMARY_CONTROLLER_IDS", "PRIMARY_PANEL_IDS", "ArenaCampaignError",
    "ArmImplementation", "CampaignCellRequest", "CampaignSpec", "TaskArtifact",
    "audit_available_source_campaign", "audit_campaign",
    "execute_available_source_campaign", "execute_campaign", "load_spec",
    "write_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
