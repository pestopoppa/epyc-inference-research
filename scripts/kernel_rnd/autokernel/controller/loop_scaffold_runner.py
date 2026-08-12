#!/usr/bin/env python3
"""Governed SAME-MODEL scaffold authoring seam for AK-LE-3.

This module turns the already-preregistered ``ScaffoldArm`` factorial into
disposable, workspace-writing cells.  It deliberately does not select tasks,
models, champions, winners, or releases.  A caller must pin all of those inputs
before compilation, and AgentKernelArena remains the only evaluator.

Importing this module performs no filesystem, process, model, compiler,
evaluator, GPU, campaign, ranking, champion, or release action.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from . import authoring_contract
from . import arena_adapter
from . import arena_cell_runner
from . import codex_container_actor
from . import loop_experiments as experiments
from ..resource import device_claim


COMPILE_INPUT_SCHEMA = "epyc.autokernel.ak_le_3_scaffold_compile_input.v1"
MANIFEST_SCHEMA = "epyc.autokernel.ak_le_3_scaffold_manifest.v1"
PANEL_SCHEMA = "epyc.autokernel.ak_le_3_scaffold_panel.v1"
CHECKPOINT_SCHEMA = "epyc.autokernel.ak_le_3_role_checkpoint.v1"
EVALUATION_SCHEMA = "epyc.autokernel.ak_le_3_arena_evaluation.v1"
AUTHORITY = "diagnostic_scaffold_observation_only"
REQUIRED_ACTOR_BOUNDARY = "disposable_worktree_single_writable_bind_v1"
EVALUATOR_BOUNDARY = "agentkernelarena_centralized_evaluator_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EVALUATOR_BRIDGE = Path(__file__).with_name("arena_scaffold_evaluator.py")
EXTERNAL_PREREQUISITE = (
    "A real observation requires operator-selected exact source/task/context/champion "
    "inputs, the clean pinned AgentKernelArena checkout, Codex credentials for both "
    "reviewed model cells, and an exclusive mi210_0 device claim. This implementation "
    "session was explicitly forbidden to run model inference or GPU evaluation, so it "
    "produces no real AK-LE-3 observation."
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,95}")
_PRODUCTION_TREES = tuple(Path(value).resolve() for value in (
    "/mnt/raid0/llm/llama.cpp", "/mnt/raid0/llm/whisper.cpp",
    "/mnt/raid0/llm/qwentts.cpp"))


class ScaffoldRunnerError(RuntimeError):
    """A scaffold cell is unpinned, confounded, unsafe, or authority-seeking."""


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ScaffoldRunnerError(f"{label} must be non-empty text without NUL")
    return value.strip()


def _sha(value: object, label: str) -> str:
    value = _text(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise ScaffoldRunnerError(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: object, label: str) -> str:
    value = _text(value, label)
    if not _COMMIT_RE.fullmatch(value):
        raise ScaffoldRunnerError(f"{label} must be a full lowercase commit")
    return value


def _safe_relative(value: object, label: str) -> str:
    raw = _text(value, label)
    path = PurePosixPath(raw)
    if (path.is_absolute() or path.as_posix() != raw or ".." in path.parts
            or "." in path.parts or raw.startswith("-")):
        raise ScaffoldRunnerError(f"{label} must be a normalized relative path")
    return raw


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args), capture_output=True, text=True,
        check=False)
    if result.returncode != 0:
        raise ScaffoldRunnerError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _assert_new_absolute(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.exists() or candidate.is_symlink():
        raise ScaffoldRunnerError(f"{label} must be a new absolute path")
    return candidate


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(
        dict(payload), indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _tree_state(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            continue
        if path.is_symlink():
            rows[relative] = f"symlink:{os.readlink(path)}"
        elif path.is_dir():
            rows[relative] = "directory"
        elif path.is_file():
            rows[relative] = f"file:{_file_sha(path)}"
        else:
            rows[relative] = "special"
    return rows


def _changed(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted(
        path for path in set(before) | set(after)
        if before.get(path) != after.get(path)))


def _path_is_allowed(path: str, scopes: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    return any(candidate == PurePosixPath(scope)
               or PurePosixPath(scope) in candidate.parents for scope in scopes)


def _tree_digest(repo: Path, commit: str) -> str:
    listing = _git(repo, "ls-tree", "-r", "--full-tree", commit)
    return hashlib.sha256((listing + "\n").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourcePin:
    """Exact non-production repository snapshot used for every arm."""

    repository: str
    base_commit: str
    base_tree_sha256: str

    def __post_init__(self) -> None:
        repo = Path(_text(self.repository, "source repository")).resolve()
        if not repo.is_absolute() or not repo.is_dir() or repo.is_symlink():
            raise ScaffoldRunnerError("source repository must be an absolute non-symlink git tree")
        if repo in _PRODUCTION_TREES:
            raise ScaffoldRunnerError("production kernel trees can never be AK-LE-3 actor sources")
        _commit(self.base_commit, "source base_commit")
        _sha(self.base_tree_sha256, "source base_tree_sha256")
        if _git(repo, "rev-parse", self.base_commit) != self.base_commit:
            raise ScaffoldRunnerError("source base commit is not present exactly")
        if _tree_digest(repo, self.base_commit) != self.base_tree_sha256:
            raise ScaffoldRunnerError("source tree identity drifted")
        object.__setattr__(self, "repository", str(repo))

    def to_dict(self) -> dict[str, str]:
        return {"repository": self.repository, "base_commit": self.base_commit,
                "base_tree_sha256": self.base_tree_sha256}


@dataclass(frozen=True)
class ActorPin:
    """Exact in-repository Codex single-writable-bind launcher identity."""

    model_id: str
    quant_id: str
    effort: str
    python_executable: str
    python_sha256: str
    codex_wrapper: str
    codex_wrapper_sha256: str
    launcher_sha256: str
    runtime_identity_sha256: str
    boundary: str = REQUIRED_ACTOR_BOUNDARY

    def __post_init__(self) -> None:
        for name in ("model_id", "quant_id", "effort"):
            _text(getattr(self, name), f"actor {name}")
        if (self.model_id not in codex_container_actor.SUPPORTED_MODELS
                or self.quant_id != "provider-native"
                or self.effort != codex_container_actor.SUPPORTED_EFFORT):
            raise ScaffoldRunnerError("actor model/quant/effort is not a reviewed Codex cell")
        python = Path(_text(self.python_executable, "actor Python"))
        wrapper = Path(_text(self.codex_wrapper, "Codex wrapper"))
        if (not python.is_absolute() or not python.is_file() or python.is_symlink()
                or _file_sha(python) != _sha(self.python_sha256, "actor Python SHA-256")):
            raise ScaffoldRunnerError("actor Python identity drifted")
        if (not wrapper.is_absolute() or not wrapper.is_file() or wrapper.is_symlink()
                or _file_sha(wrapper) != _sha(
                    self.codex_wrapper_sha256, "Codex wrapper SHA-256")):
            raise ScaffoldRunnerError("Codex wrapper identity drifted")
        launcher = Path(codex_container_actor.__file__).resolve()
        if _file_sha(launcher) != _sha(self.launcher_sha256, "actor launcher SHA-256"):
            raise ScaffoldRunnerError("trusted container actor module identity drifted")
        runtime = codex_container_actor.runtime_identity(wrapper)
        if _digest(runtime) != _sha(
                self.runtime_identity_sha256, "actor runtime identity SHA-256"):
            raise ScaffoldRunnerError("container actor runtime identity drifted")
        if (runtime.get("kind") != "docker_workspace_bind_only"
                or runtime.get("writable_host_binds") != ["/workspace"]
                or runtime.get("image_id") != codex_container_actor.CONTAINER_IMAGE_ID
                or self.boundary != REQUIRED_ACTOR_BOUNDARY):
            raise ScaffoldRunnerError("actor runtime does not enforce one writable workspace bind")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "codex_wrapper", str(wrapper))

    @property
    def cell_key(self) -> tuple[str, str, str]:
        return self.model_id, self.quant_id, self.effort

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in (
            "model_id", "quant_id", "effort", "python_executable",
            "python_sha256", "codex_wrapper",
            "codex_wrapper_sha256", "launcher_sha256",
            "runtime_identity_sha256", "boundary")}


@dataclass(frozen=True)
class ArenaEvaluatorPin:
    """Exact existing AgentKernelArena evaluator and selected task surface."""

    arena_root: str
    arena_commit: str
    evaluator_relative_path: str
    evaluator_sha256: str
    task_relative_root: str
    task_tree_sha256: str
    task_config_sha256: str
    python_executable: str
    python_sha256: str
    package_identity_sha256: str
    repository_root: str
    bridge_sha256: str
    boundary: str = EVALUATOR_BOUNDARY

    def __post_init__(self) -> None:
        root = Path(_text(self.arena_root, "Arena root")).resolve()
        if not root.is_dir() or root.is_symlink():
            raise ScaffoldRunnerError("Arena root must be an exact non-symlink checkout")
        if _git(root, "rev-parse", "HEAD") != _commit(
                self.arena_commit, "Arena commit"):
            raise ScaffoldRunnerError("Arena checkout commit drifted")
        if self.arena_commit != arena_adapter.AGENT_KERNEL_ARENA_PIN.commit:
            raise ScaffoldRunnerError("Arena commit is not the reviewed paper pin")
        if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ScaffoldRunnerError("Arena checkout must be clean")
        evaluator = root / _safe_relative(
            self.evaluator_relative_path, "evaluator relative path")
        if not evaluator.is_file() or _file_sha(evaluator) != _sha(
                self.evaluator_sha256, "evaluator SHA-256"):
            raise ScaffoldRunnerError("Arena evaluator source identity drifted")
        task_root = root / _safe_relative(self.task_relative_root, "task relative root")
        config = task_root / "config.yaml"
        if not task_root.is_dir() or not config.is_file():
            raise ScaffoldRunnerError("pinned Arena task/config is unavailable")
        if _tree_state(task_root) and _digest(_tree_state(task_root)) != _sha(
                self.task_tree_sha256, "task tree SHA-256"):
            raise ScaffoldRunnerError("Arena task tree identity drifted")
        if _file_sha(config) != _sha(self.task_config_sha256, "task config SHA-256"):
            raise ScaffoldRunnerError("Arena task config identity drifted")
        expected_python = arena_cell_runner._evaluator_python_identity()
        python = Path(_text(self.python_executable, "evaluator Python"))
        if not python.is_absolute() or not python.is_file() or _file_sha(python) != _sha(
                self.python_sha256, "evaluator Python SHA-256"):
            raise ScaffoldRunnerError("evaluator Python identity drifted")
        if (str(python) != expected_python["path"]
                or self.python_sha256 != expected_python["sha256"]
                or _digest(expected_python["packages"]) != _sha(
                    self.package_identity_sha256,
                    "evaluator package identity SHA-256")):
            raise ScaffoldRunnerError("evaluator Python/package pin is not the Arena pin")
        repository = Path(_text(self.repository_root, "repository root")).resolve()
        if repository != REPOSITORY_ROOT:
            raise ScaffoldRunnerError("evaluator repository import root drifted")
        if _file_sha(EVALUATOR_BRIDGE) != _sha(
                self.bridge_sha256, "evaluator bridge SHA-256"):
            raise ScaffoldRunnerError("evaluator bridge module identity drifted")
        if self.boundary != EVALUATOR_BOUNDARY:
            raise ScaffoldRunnerError("evaluation must cross AgentKernelArena")
        object.__setattr__(self, "arena_root", str(root))
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "repository_root", str(repository))

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in (
            "arena_root", "arena_commit", "evaluator_relative_path",
            "evaluator_sha256", "task_relative_root", "task_tree_sha256",
            "task_config_sha256", "python_executable", "python_sha256",
            "package_identity_sha256", "repository_root", "bridge_sha256",
            "boundary")}


@dataclass(frozen=True)
class ProcessCapture:
    argv: tuple[str, ...]
    pid: int
    process_group_id: int
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    started_at: str
    finished_at: str
    elapsed_wall_seconds: float
    group_members_at_start: tuple[int, ...]
    group_members_after_reap: tuple[int, ...]


CommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str], str, float], ProcessCapture]
EvaluatorRunner = Callable[
    [Mapping[str, Any], Path, Mapping[str, str], float], tuple[ProcessCapture, Mapping[str, Any]]]


def _live_group_members(group_id: int) -> tuple[int, ...]:
    rows: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2:].split()
            if int(fields[2]) == group_id and fields[0] != "Z":
                rows.append(int(entry.name))
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    return tuple(sorted(rows))


def _terminate_group(group_id: int) -> None:
    if group_id <= 1 or group_id == os.getpgrp():
        raise ScaffoldRunnerError("refusing unsafe process-group teardown")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not _live_group_members(group_id):
            return
        try:
            os.killpg(group_id, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and _live_group_members(group_id):
            time.sleep(0.02)
    survivors = _live_group_members(group_id)
    if survivors:
        raise ScaffoldRunnerError(f"captured process group survived teardown: {survivors}")


def _run_process(argv: Sequence[str], cwd: Path, environment: Mapping[str, str],
                 prompt: str, timeout_seconds: float) -> ProcessCapture:
    if (not argv or any(not isinstance(part, str) or not part for part in argv)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ScaffoldRunnerError("captured command and timeout must be bounded")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(argv), cwd=cwd, env=dict(environment), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
    except OSError as exc:
        raise ScaffoldRunnerError(f"could not start captured command: {argv[0]}") from exc
    pid = process.pid
    group = os.getpgid(pid)
    initial = _live_group_members(group)
    stdout = stderr = ""
    timed_out = False
    try:
        stdout, stderr = process.communicate(input=prompt, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_group(group)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if _live_group_members(group):
            _terminate_group(group)
        if process.poll() is None:
            stdout, stderr = process.communicate(timeout=5)
    ended = time.monotonic()
    survivors = _live_group_members(group)
    if survivors:
        raise ScaffoldRunnerError(f"captured process group was not reaped: {survivors}")
    return ProcessCapture(
        tuple(argv), pid, group, int(process.returncode), stdout, stderr,
        timed_out, started_at, datetime.now(timezone.utc).isoformat(),
        ended - started, initial, survivors)


def _actor_argv(pin: ActorPin, *, workspace: Path, role: str,
                wall_seconds: float) -> tuple[str, ...]:
    return (
        pin.python_executable, "-m",
        codex_container_actor.EXECUTABLE_MODULE,
        "--codex-wrapper", pin.codex_wrapper,
        "--workspace", str(workspace), "--model", pin.model_id,
        "--effort", pin.effort)


def _exact_mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ScaffoldRunnerError(f"{label} must contain exactly {sorted(keys)}")
    return dict(value)


def _contract_from_manifest(value: object) -> experiments.ExperimentContract:
    row = _exact_mapping(value, {
        "schema", "experiment_id", "authority", "fixed", "planner_arms",
        "direction_predictions", "scaffold_arms", "prior_hypothesis_sha256",
        "prefilter", "constraints", "contract_sha256",
    }, "experiment contract")
    fixed_row = _exact_mapping(row["fixed"], {
        "champion", "retrieval_context_sha256", "propose_prompt",
        "propose_prompt_sha256", "selected_task",
    }, "fixed prompt frame")
    champion = _exact_mapping(fixed_row["champion"], {"ref", "sha256"}, "champion")
    task = _exact_mapping(
        fixed_row["selected_task"], {"ref", "task", "task_sha256"},
        "selected task")
    planners = tuple(experiments.PlannerArm(**_exact_mapping(item, {
        "cell_id", "model_id", "quant_id", "effort", "target_context_mode",
    }, "planner arm")) for item in row["planner_arms"])
    predictions = tuple(experiments.DirectionPrediction(**_exact_mapping(item, {
        "model_id", "quant_id", "direction", "rationale",
    }, "direction prediction")) for item in row["direction_predictions"])
    scaffolds = []
    for item in row["scaffold_arms"]:
        arm = _exact_mapping(item, {
            "cell_id", "model_id", "quant_id", "effort", "scaffold", "roles",
            "wall_seconds",
        }, "scaffold arm")
        roles = tuple(experiments.RoleBudget(**_exact_mapping(role, {
            "role", "wall_seconds", "instruction", "instruction_sha256",
        }, "role budget")) for role in arm.pop("roles"))
        claimed_wall = arm.pop("wall_seconds")
        scaffold = experiments.ScaffoldArm(roles=roles, **arm)
        if scaffold.wall_seconds != claimed_wall:
            raise ScaffoldRunnerError("scaffold wall_seconds is not derived from its roles")
        scaffolds.append(scaffold)
    prefilter = _exact_mapping(row["prefilter"], {"ref", "sha256"}, "prefilter")
    contract = experiments.ExperimentContract(
        row["experiment_id"],
        experiments.FixedPromptFrame(
            experiments.ArtifactPin(**champion),
            fixed_row["retrieval_context_sha256"], fixed_row["propose_prompt"],
            fixed_row["propose_prompt_sha256"],
            experiments.SelectedTaskArtifact(**task)),
        planners, predictions, tuple(scaffolds),
        tuple(row["prior_hypothesis_sha256"]),
        experiments.ArtifactPin(**prefilter))
    input_body = dict(row)
    input_claim = input_body.pop("contract_sha256")
    if _digest(input_body) != _sha(input_claim, "input contract SHA-256"):
        raise ScaffoldRunnerError("input experiment contract SHA-256 does not verify")
    reproduced = contract.to_manifest()
    reproduced.pop("contract_sha256")
    if reproduced != input_body:
        raise ScaffoldRunnerError("experiment contract does not reproduce exactly")
    return contract


def _context_from_input(value: object) -> authoring_contract.PricedContext:
    row = _exact_mapping(value, {"round_id", "budget", "items"}, "priced context")
    budget = authoring_contract.ContextBudget(**_exact_mapping(row["budget"], {
        "max_total_tokens", "max_item_tokens", "max_items",
    }, "context budget"))
    items = tuple(authoring_contract.ContextItem(**_exact_mapping(item, {
        "source_ref", "purpose", "content", "bulk_read",
    }, "context item")) for item in row["items"])
    return authoring_contract.price_context(
        round_id=row["round_id"], budget=budget, items=items)


def compile_manifest_input(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one strict reviewed JSON input into the write-once run manifest."""
    row = _exact_mapping(value, {
        "schema", "experiment_contract", "context", "source", "actors",
        "evaluator", "allowed_write_paths",
    }, "compile input")
    if row["schema"] != COMPILE_INPUT_SCHEMA:
        raise ScaffoldRunnerError("compile input schema drifted")
    if not isinstance(row["actors"], list) or not isinstance(
            row["allowed_write_paths"], list):
        raise ScaffoldRunnerError("actors and allowed_write_paths must be JSON lists")
    return compile_manifest(
        _contract_from_manifest(row["experiment_contract"]),
        context=_context_from_input(row["context"]),
        source=SourcePin(**_exact_mapping(row["source"], {
            "repository", "base_commit", "base_tree_sha256",
        }, "source pin")),
        actors=tuple(ActorPin(**_exact_mapping(actor, {
            "model_id", "quant_id", "effort", "python_executable",
            "python_sha256", "codex_wrapper", "codex_wrapper_sha256",
            "launcher_sha256", "runtime_identity_sha256", "boundary",
        }, "actor pin")) for actor in row["actors"]),
        evaluator=ArenaEvaluatorPin(**_exact_mapping(row["evaluator"], {
            "arena_root", "arena_commit", "evaluator_relative_path",
            "evaluator_sha256", "task_relative_root", "task_tree_sha256",
            "task_config_sha256", "python_executable", "python_sha256",
            "package_identity_sha256", "repository_root", "bridge_sha256",
            "boundary",
        }, "evaluator pin")),
        allowed_write_paths=tuple(row["allowed_write_paths"]))


def compile_manifest(
    contract: experiments.ExperimentContract, *,
    context: authoring_contract.PricedContext,
    source: SourcePin,
    actors: Sequence[ActorPin],
    evaluator: ArenaEvaluatorPin,
    allowed_write_paths: Sequence[str],
) -> dict[str, Any]:
    """Compile the exact matched scaffold panel without executing anything."""
    if experiments.context_sha256(context) != contract.fixed.retrieval_context_sha256:
        raise ScaffoldRunnerError("retrieval context differs from selected context pin")
    scopes = tuple(sorted({_safe_relative(value, "allowed write path")
                           for value in allowed_write_paths}))
    if not scopes:
        raise ScaffoldRunnerError("at least one exact actor write path is required")
    actor_map = {pin.cell_key: pin for pin in actors}
    if len(actor_map) != len(tuple(actors)):
        raise ScaffoldRunnerError("actor pins must be unique by model/quant/effort")
    expected = {arm.model_quant_effort for arm in contract.scaffold_arms}
    if set(actor_map) != expected:
        raise ScaffoldRunnerError("actor pins must exactly cover scaffold model cells")
    if {key[0] for key in expected} != set(codex_container_actor.SUPPORTED_MODELS):
        raise ScaffoldRunnerError(
            "AK-LE-3 must independently vary exactly the two reviewed Codex models")
    cells = []
    for arm in contract.scaffold_arms:
        pin = actor_map[arm.model_quant_effort]
        roles = []
        for stage in arm.roles:
            prompt = experiments.render_scaffold_prompt(
                contract, arm.cell_id, stage.role, context=context)
            roles.append({
                **stage.to_dict(), "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "argv_shape": list(_actor_argv(
                    pin, workspace=Path("{candidate_worktree}"), role=stage.role,
                    wall_seconds=stage.wall_seconds)),
            })
        cells.append({
            "cell_id": arm.cell_id, "model_id": arm.model_id,
            "quant_id": arm.quant_id, "effort": arm.effort,
            "scaffold": arm.scaffold, "wall_seconds": arm.wall_seconds,
            "actor": pin.to_dict(), "roles": roles,
        })
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA, "authority": AUTHORITY,
        "experiment_id": contract.experiment_id,
        "experiment_contract": contract.to_manifest(),
        "selected_pins": {
            "task": contract.fixed.selected_task.to_dict(),
            "context_sha256": experiments.context_sha256(context),
            "champion": contract.fixed.champion.to_dict(),
            "source": source.to_dict(), "evaluator": evaluator.to_dict(),
        },
        "allowed_write_paths": list(scopes), "cells": cells,
        "external_prerequisite": EXTERNAL_PREREQUISITE,
        "constraints": {
            "same_model_within_scaffold_pair": True,
            "model_and_scaffold_independently_varied": True,
            "wall_time_matched": True,
            "fresh_baseline_and_candidate_worktrees": True,
            "actor_write_boundary": REQUIRED_ACTOR_BOUNDARY,
            "evaluation_boundary": EVALUATOR_BOUNDARY,
            "campaign_authority": False, "ranking_authority": False,
            "champion_authority": False, "release_authority": False,
            "model_or_evaluator_invoked_by_compiler": False,
        },
    }
    payload["manifest_sha256"] = _digest(payload)
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(manifest))
    claimed = payload.pop("manifest_sha256", None)
    if _digest(payload) != _sha(claimed, "manifest SHA-256"):
        raise ScaffoldRunnerError("manifest SHA-256 does not verify")
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("authority") != AUTHORITY:
        raise ScaffoldRunnerError("manifest schema or authority drifted")
    constraints = payload.get("constraints")
    forbidden = ("campaign_authority", "ranking_authority",
                 "champion_authority", "release_authority")
    if (not isinstance(constraints, Mapping)
            or any(constraints.get(key) is not False for key in forbidden)
            or constraints.get("actor_write_boundary") != REQUIRED_ACTOR_BOUNDARY
            or constraints.get("evaluation_boundary") != EVALUATOR_BOUNDARY):
        raise ScaffoldRunnerError("manifest requests forbidden authority or boundary")
    contract = payload.get("experiment_contract")
    if not isinstance(contract, Mapping):
        raise ScaffoldRunnerError("manifest lacks the experiment contract")
    contract_body = dict(contract)
    contract_sha = contract_body.pop("contract_sha256", None)
    if _digest(contract_body) != _sha(contract_sha, "contract SHA-256"):
        raise ScaffoldRunnerError("embedded experiment contract does not verify")
    pins = payload.get("selected_pins")
    if not isinstance(pins, Mapping):
        raise ScaffoldRunnerError("selected pins are absent")
    SourcePin(**pins["source"])
    ArenaEvaluatorPin(**pins["evaluator"])
    if pins.get("task") != contract.get("fixed", {}).get("selected_task"):
        raise ScaffoldRunnerError("selected task pin drifted")
    if pins.get("champion") != contract.get("fixed", {}).get("champion"):
        raise ScaffoldRunnerError("selected champion pin drifted")
    if pins.get("context_sha256") != contract.get("fixed", {}).get(
            "retrieval_context_sha256"):
        raise ScaffoldRunnerError("selected context pin drifted")
    scopes = tuple(_safe_relative(value, "allowed write path")
                   for value in payload.get("allowed_write_paths", ()))
    if not scopes:
        raise ScaffoldRunnerError("manifest has no actor write scope")
    contract_arms = {row["cell_id"]: row for row in contract.get("scaffold_arms", ())}
    cells = payload.get("cells")
    if not isinstance(cells, list) or set(contract_arms) != {
            row.get("cell_id") for row in cells if isinstance(row, Mapping)}:
        raise ScaffoldRunnerError("cells do not exactly cover scaffold arms")
    totals: set[float] = set()
    pairs: dict[tuple[str, str, str], set[str]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("cell_id") not in contract_arms:
            raise ScaffoldRunnerError("scaffold cell is malformed")
        arm = contract_arms[cell["cell_id"]]
        for key in ("model_id", "quant_id", "effort", "scaffold", "wall_seconds"):
            if cell.get(key) != arm.get(key):
                raise ScaffoldRunnerError(f"scaffold cell {key} drifted")
        pin = ActorPin(**cell["actor"])
        if pin.cell_key != (cell["model_id"], cell["quant_id"], cell["effort"]):
            raise ScaffoldRunnerError("actor pin differs from scaffold model cell")
        roles = cell.get("roles")
        if not isinstance(roles, list) or len(roles) != len(arm["roles"]):
            raise ScaffoldRunnerError("scaffold role coverage drifted")
        for role, planned in zip(roles, arm["roles"]):
            if any(role.get(key) != planned.get(key) for key in (
                    "role", "wall_seconds", "instruction", "instruction_sha256")):
                raise ScaffoldRunnerError("scaffold role pin drifted")
            prompt = role.get("prompt")
            if not isinstance(prompt, str) or hashlib.sha256(
                    prompt.encode("utf-8")).hexdigest() != role.get("prompt_sha256"):
                raise ScaffoldRunnerError("scaffold prompt identity drifted")
        totals.add(float(cell["wall_seconds"]))
        pairs.setdefault(pin.cell_key, set()).add(cell["scaffold"])
    if len(totals) != 1 or any(value != {
            experiments.SCAFFOLD_DIRECT, experiments.SCAFFOLD_SPLIT}
            for value in pairs.values()):
        raise ScaffoldRunnerError("scaffold factorial is not same-model and wall matched")
    payload["manifest_sha256"] = claimed
    return payload


def _create_worktree(source: SourcePin, destination: Path) -> dict[str, Any]:
    _assert_new_absolute(destination, "worktree")
    repo = Path(source.repository)
    result = subprocess.run(
        ("git", "-C", str(repo), "worktree", "add", "--detach", str(destination),
         source.base_commit), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ScaffoldRunnerError(f"could not create disposable worktree: {result.stderr.strip()}")
    try:
        observed = _git(destination, "rev-parse", "HEAD")
        if (observed != source.base_commit
                or _tree_digest(destination, observed) != source.base_tree_sha256):
            raise ScaffoldRunnerError("fresh disposable worktree identity drifted")
    except Exception:
        subprocess.run(
            ("git", "-C", str(repo), "worktree", "remove", "--force",
             str(destination)), capture_output=True, text=True, check=False)
        raise
    return {"argv": ["git", "-C", str(repo), "worktree", "add", "--detach",
                     str(destination), source.base_commit],
            "commit": observed, "tree_sha256": source.base_tree_sha256}


def _remove_worktree(source: SourcePin, destination: Path) -> dict[str, Any]:
    """Remove exactly one worktree created by this cell and prove it is gone."""
    destination = destination.resolve()
    common = Path(_git(destination, "rev-parse", "--git-common-dir")).resolve()
    expected = Path(source.repository, ".git").resolve()
    if common != expected:
        raise ScaffoldRunnerError("refusing to remove a worktree with foreign git identity")
    argv = ("git", "-C", source.repository, "worktree", "remove", "--force",
            str(destination))
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ScaffoldRunnerError(
            f"could not remove captured disposable worktree: {result.stderr.strip()}")
    if destination.exists() or destination.is_symlink():
        raise ScaffoldRunnerError("captured disposable worktree survived removal")
    listed = _git(Path(source.repository), "worktree", "list", "--porcelain")
    if str(destination) in listed:
        raise ScaffoldRunnerError("removed worktree remains registered")
    return {"argv": list(argv), "removed": True, "path": str(destination)}


def _capture_dict(capture: ProcessCapture) -> dict[str, Any]:
    return {
        "argv": list(capture.argv), "pid": capture.pid,
        "process_group_id": capture.process_group_id,
        "group_members_at_start": list(capture.group_members_at_start),
        "group_members_after_reap": list(capture.group_members_after_reap),
        "returncode": capture.returncode, "timed_out": capture.timed_out,
        "started_at": capture.started_at, "finished_at": capture.finished_at,
        "elapsed_wall_seconds": capture.elapsed_wall_seconds,
    }


def _seal_checkpoint(
    *, cell_root: Path, ordinal: int, cell_id: str, role: Mapping[str, Any],
    capture: ProcessCapture, before: Mapping[str, str], after: Mapping[str, str],
    allowed_write_paths: Sequence[str], workspace: Path, base_commit: str,
) -> dict[str, Any]:
    changed = _changed(before, after)
    unauthorized = tuple(path for path in changed
                         if not _path_is_allowed(path, allowed_write_paths))
    unsafe = tuple(path for path in changed if after.get(path, "").startswith(
        ("symlink:", "special")))
    diff = _git(workspace, "diff", "--binary", "--no-ext-diff", "--")
    observed_head = _git(workspace, "rev-parse", "HEAD")
    checkpoint: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA, "cell_id": cell_id,
        "role": role["role"], "role_ordinal": ordinal,
        "planned_wall_seconds": role["wall_seconds"],
        "process": _capture_dict(capture), "changed_paths": list(changed),
        "unauthorized_paths": list(unauthorized), "unsafe_paths": list(unsafe),
        "write_scope_passed": not unauthorized and not unsafe,
        "tree_before_sha256": _digest(before), "tree_after_sha256": _digest(after),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "base_commit": base_commit, "observed_head": observed_head,
        "authority": AUTHORITY,
    }
    checkpoint["checkpoint_sha256"] = _digest(checkpoint)
    role_root = cell_root / "checkpoints" / f"{ordinal:02d}-{role['role']}"
    role_root.mkdir(parents=True, exist_ok=False)
    _atomic_write(role_root / "stdout.txt", capture.stdout.encode("utf-8"))
    _atomic_write(role_root / "stderr.txt", capture.stderr.encode("utf-8"))
    _atomic_write(role_root / "workspace.diff", diff.encode("utf-8"))
    checkpoint["stdout_sha256"] = _file_sha(role_root / "stdout.txt")
    checkpoint["stderr_sha256"] = _file_sha(role_root / "stderr.txt")
    # Re-seal after artifact identities have entered the checkpoint.
    checkpoint.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = _digest(checkpoint)
    _atomic_json(role_root / "checkpoint.json", checkpoint)
    if observed_head != base_commit:
        raise ScaffoldRunnerError(
            f"actor changed disposable worktree HEAD for {cell_id}")
    if unauthorized or unsafe:
        raise ScaffoldRunnerError(
            f"actor write-scope audit failed for {cell_id}: "
            f"unauthorized={unauthorized}, unsafe={unsafe}")
    if capture.timed_out or capture.returncode != 0:
        raise ScaffoldRunnerError(
            f"actor role failed for {cell_id}/{role['role']}: "
            f"timeout={capture.timed_out}, returncode={capture.returncode}")
    if capture.elapsed_wall_seconds > float(role["wall_seconds"]) + 1.0:
        raise ScaffoldRunnerError("actor exceeded role wall-time budget")
    return checkpoint


def _default_evaluator_runner(
    request: Mapping[str, Any], cell_root: Path, environment: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[ProcessCapture, Mapping[str, Any]]:
    request_path = cell_root / "arena-evaluator-request.json"
    output_path = cell_root / "arena-evaluator-result.json"
    _atomic_json(request_path, request)
    evaluator = request["evaluator"]
    repository_root = Path(evaluator["repository_root"]).resolve()
    if (repository_root != REPOSITORY_ROOT
            or _file_sha(EVALUATOR_BRIDGE) != evaluator["bridge_sha256"]):
        raise ScaffoldRunnerError("evaluator bridge import identity drifted before launch")
    argv = (
        evaluator["python_executable"], "-m",
        "scripts.kernel_rnd.autokernel.controller.arena_scaffold_evaluator",
        "--request", str(request_path), "--output", str(output_path))
    child_environment = dict(environment)
    child_environment.update({
        "PYTHONPATH": str(repository_root),
        "HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    capture = _run_process(
        argv, repository_root, child_environment, "", timeout_seconds)
    if capture.timed_out or capture.returncode != 0:
        raise ScaffoldRunnerError("AgentKernelArena evaluator process failed")
    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScaffoldRunnerError("AgentKernelArena evaluator emitted no valid result") from exc
    if not isinstance(result, Mapping):
        raise ScaffoldRunnerError("AgentKernelArena evaluator result must be an object")
    return capture, result


def run_manifest(
    manifest: Mapping[str, Any], *, output_root: str | Path,
    environment: Mapping[str, str] | None = None,
    actor_runner: CommandRunner = _run_process,
    evaluator_runner: EvaluatorRunner = _default_evaluator_runner,
    fixture_mode: bool = False,
    claim_journal: str | Path | None = None,
) -> dict[str, Any]:
    """Execute and seal the complete scaffold factorial in fresh worktrees."""
    payload = validate_manifest(manifest)
    root = _assert_new_absolute(output_root, "output_root")
    injected = actor_runner is not _run_process or evaluator_runner is not _default_evaluator_runner
    if injected and fixture_mode is not True:
        raise ScaffoldRunnerError("injected runners require explicit fixture_mode")
    if not injected and fixture_mode:
        raise ScaffoldRunnerError("fixture_mode cannot label the real execution boundary")
    journal_path = None
    if not fixture_mode:
        if claim_journal is None:
            raise ScaffoldRunnerError(
                "real Arena evaluation requires an explicit device claim journal")
        journal_path = Path(claim_journal)
        if not journal_path.is_absolute():
            raise ScaffoldRunnerError("device claim journal must be absolute")
    root.mkdir(parents=True)
    _atomic_json(root / "manifest.json", payload)
    env = dict(os.environ if environment is None else environment)
    source = SourcePin(**payload["selected_pins"]["source"])
    evaluator = ArenaEvaluatorPin(**payload["selected_pins"]["evaluator"])
    completed: list[dict[str, Any]] = []
    try:
        for ordinal, cell in enumerate(payload["cells"], 1):
            cell_root = root / "cells" / f"{ordinal:03d}-{cell['cell_id']}"
            cell_root.mkdir(parents=True)
            baseline = cell_root / "baseline-worktree"
            candidate = cell_root / "candidate-worktree"
            created: list[Path] = []
            cleanup: dict[str, Any] = {}
            try:
                creation = {"baseline": _create_worktree(source, baseline)}
                created.append(baseline)
                creation["candidate"] = _create_worktree(source, candidate)
                created.append(candidate)
                _atomic_json(cell_root / "worktree-creation.json", creation)
                pin = ActorPin(**cell["actor"])
                checkpoints = []
                for role_ordinal, role in enumerate(cell["roles"], 1):
                    before = _tree_state(candidate)
                    argv = _actor_argv(
                        pin, workspace=candidate, role=role["role"],
                        wall_seconds=float(role["wall_seconds"]))
                    if (_file_sha(Path(argv[0])) != pin.python_sha256
                            or _file_sha(Path(codex_container_actor.__file__))
                            != pin.launcher_sha256):
                        raise ScaffoldRunnerError("container actor identity drifted before role")
                    actor_env = dict(env)
                    actor_env["PYTHONPATH"] = str(REPOSITORY_ROOT)
                    capture = actor_runner(
                        argv, candidate, actor_env, role["prompt"],
                        float(role["wall_seconds"]))
                    if capture.argv != argv or capture.pid <= 1 \
                            or capture.process_group_id <= 1:
                        raise ScaffoldRunnerError("actor process identity was not captured exactly")
                    after = _tree_state(candidate)
                    checkpoints.append(_seal_checkpoint(
                        cell_root=cell_root, ordinal=role_ordinal,
                        cell_id=cell["cell_id"], role=role, capture=capture,
                        before=before, after=after,
                        allowed_write_paths=payload["allowed_write_paths"],
                        workspace=candidate, base_commit=source.base_commit))
                    if _git(baseline, "status", "--porcelain=v1", "--untracked-files=all"):
                        raise ScaffoldRunnerError("actor modified the isolated baseline worktree")
                request = {
                    "schema": EVALUATION_SCHEMA, "authority": AUTHORITY,
                    "cell_id": cell["cell_id"], "baseline_workspace": str(baseline),
                    "candidate_workspace": str(candidate), "source": source.to_dict(),
                    "evaluator": evaluator.to_dict(),
                    "constraints": {
                        "agentkernelarena_is_only_evaluator": True,
                        "actor_reported_performance_admitted": False,
                        "campaign_authority": False, "ranking_authority": False,
                        "champion_authority": False, "release_authority": False,
                    },
                }
                claim_open = claim_released = None
                claim = None
                try:
                    if not fixture_mode:
                        claim = device_claim.acquire_device_claim(
                            "mi210_0", purpose=f"AK-LE-3 Arena {cell['cell_id']}",
                            campaign_id=payload["experiment_id"],
                            journal=device_claim.ClaimJournal(journal_path),
                            holder_label="loop_scaffold_runner.py", timeout_s=0,
                            max_hold_s=max(420.0, float(cell["wall_seconds"]) + 120.0))
                        claim_open = claim.receipt().to_dict()
                    evaluation_capture, evaluation = evaluator_runner(
                        request, cell_root, env, max(300.0, float(cell["wall_seconds"])))
                finally:
                    if claim is not None:
                        claim_released = claim.release().to_dict()
                if (evaluation_capture.pid <= 1 or evaluation_capture.process_group_id <= 1
                        or evaluation_capture.group_members_after_reap):
                    raise ScaffoldRunnerError("evaluator process group was not captured and reaped")
                if (evaluation.get("schema") != EVALUATION_SCHEMA
                        or evaluation.get("authority") != AUTHORITY
                        or evaluation.get("cell_id") != cell["cell_id"]):
                    raise ScaffoldRunnerError("AgentKernelArena evaluation identity drifted")
                if any(evaluation.get(key) for key in (
                        "campaign_authority", "ranking_authority",
                        "champion_authority", "release_authority")):
                    raise ScaffoldRunnerError("evaluation attempted to acquire forbidden authority")
                _atomic_json(cell_root / "arena-evaluation.json", dict(evaluation))
                cell_receipt = {
                    "cell_id": cell["cell_id"], "model_id": cell["model_id"],
                    "quant_id": cell["quant_id"], "effort": cell["effort"],
                    "scaffold": cell["scaffold"],
                    "capture_mode": "fixture" if fixture_mode else "measured",
                    "planned_wall_seconds": cell["wall_seconds"],
                    "observed_actor_wall_seconds": sum(
                        row["process"]["elapsed_wall_seconds"] for row in checkpoints),
                    "checkpoints": checkpoints,
                    "evaluation_process": _capture_dict(evaluation_capture),
                    "evaluation_sha256": _file_sha(cell_root / "arena-evaluation.json"),
                    "device_claim_open": claim_open,
                    "device_claim_released": claim_released,
                    "authority": AUTHORITY if not fixture_mode else "fixture_only",
                }
                cell_receipt["cell_receipt_sha256"] = _digest(cell_receipt)
                _atomic_json(cell_root / "cell-receipt.json", cell_receipt)
                completed.append(cell_receipt)
            finally:
                cleanup_errors = []
                for worktree in reversed(created):
                    try:
                        cleanup[worktree.name] = _remove_worktree(source, worktree)
                    except Exception as cleanup_error:
                        cleanup_errors.append(f"{worktree}: {cleanup_error}")
                cleanup["all_removed"] = not cleanup_errors
                cleanup["errors"] = cleanup_errors
                _atomic_json(cell_root / "worktree-cleanup.json", cleanup)
                if cleanup_errors:
                    raise ScaffoldRunnerError(
                        f"disposable worktree cleanup failed: {cleanup_errors}")
    except Exception as exc:
        failure = {
            "schema": PANEL_SCHEMA, "status": "failed", "authority": AUTHORITY,
            "manifest_sha256": payload["manifest_sha256"],
            "completed_cell_ids": [row["cell_id"] for row in completed],
            "error_type": type(exc).__name__, "error": str(exc),
            "constraints": {"rankable": False, "promotable": False},
        }
        failure["panel_sha256"] = _digest(failure)
        _atomic_json(root / "panel.json", failure)
        raise
    panel = {
        "schema": PANEL_SCHEMA, "status": "complete",
        "authority": AUTHORITY if not fixture_mode else "fixture_only",
        "capture_mode": "fixture" if fixture_mode else "measured",
        "experiment_id": payload["experiment_id"],
        "manifest_sha256": payload["manifest_sha256"], "cells": completed,
        "constraints": {
            "same_model_within_scaffold_pair": True,
            "wall_time_matched_by_plan": True,
            "centralized_agentkernelarena_evaluation": True,
            "empirical_observation": not fixture_mode,
            "campaign_authority": False, "ranking_authority": False,
            "champion_authority": False, "release_authority": False,
        },
    }
    panel["panel_sha256"] = _digest(panel)
    _atomic_json(root / "panel.json", panel)
    return panel


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser(
        "compile", help="compile one reviewed JSON input without executing a cell")
    compile_parser.add_argument("--input", required=True)
    compile_parser.add_argument("--output", required=True)
    run_parser = commands.add_parser(
        "run", help="execute one exact compiled four-cell manifest")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--claim-journal", required=True)
    args = parser.parse_args(argv)
    input_path = Path(args.input if args.command == "compile" else args.manifest)
    label = "compile input" if args.command == "compile" else "manifest"
    if not input_path.is_absolute() or not input_path.is_file():
        raise ScaffoldRunnerError(f"{label} must be an existing absolute JSON file")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScaffoldRunnerError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ScaffoldRunnerError(f"{label} must be a JSON object")
    if args.command == "compile":
        output = _assert_new_absolute(args.output, "manifest output")
        manifest = compile_manifest_input(payload)
        _atomic_json(output, manifest)
        print(json.dumps({
            "status": "compiled", "experiment_id": manifest["experiment_id"],
            "manifest": str(output),
            "manifest_sha256": manifest["manifest_sha256"],
        }, sort_keys=True))
        return 0
    panel = run_manifest(
        payload, output_root=args.output_root, claim_journal=args.claim_journal)
    print(json.dumps({
        "status": panel["status"], "experiment_id": panel["experiment_id"],
        "output_root": str(Path(args.output_root).resolve()),
        "panel_sha256": panel["panel_sha256"],
    }, sort_keys=True))
    return 0


__all__ = [
    "REQUIRED_ACTOR_BOUNDARY", "AUTHORITY", "CHECKPOINT_SCHEMA", "COMPILE_INPUT_SCHEMA",
    "EVALUATION_SCHEMA",
    "EVALUATOR_BOUNDARY", "EXTERNAL_PREREQUISITE", "MANIFEST_SCHEMA",
    "PANEL_SCHEMA", "ActorPin", "ArenaEvaluatorPin", "ProcessCapture",
    "ScaffoldRunnerError", "SourcePin", "compile_manifest",
    "compile_manifest_input", "run_manifest",
    "validate_manifest", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
