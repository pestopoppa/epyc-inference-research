#!/usr/bin/env python3
"""Governed, observation-only execution bridge for AK-LE-1/2.

The compiler consumes an :class:`loop_experiments.ExperimentContract` and emits
a hash-bound planner manifest.  The runner executes only those predeclared
planner cells through captured Claude or Codex CLI process groups, preserving
the exact prompt, stdout, stderr, last message, timing, argv, and parsed output.

AK-LE-3 is deliberately refused.  The existing AgentKernelArena boundary owns
GPU evaluation and fixed controller/checkpoint semantics; it does not expose a
same-model direct-vs-split authoring seam.  Pretending otherwise would turn
different model roles and evaluator paths into a scaffold comparison.

Importing this module performs no filesystem, process, model, evaluator, GPU,
campaign, ranking, champion, or release action.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from . import authoring_contract
from . import loop_experiments as experiments


MANIFEST_SCHEMA = "epyc.autokernel.loop_experiment_execution_manifest.v1"
PANEL_SCHEMA = "epyc.autokernel.loop_experiment_planner_panel.v1"
RAW_OBSERVATION_SCHEMA = "epyc.autokernel.loop_experiment_raw_planner.v1"
AUTHORITY = "observe_only_no_campaign_ranking_champion_or_release_authority"
PROVIDERS = frozenset({"claude", "codex"})
SCAFFOLD_GAP = (
    "AK-LE-3 requires a governed same-model direct-implement versus "
    "implement-then-exploit authoring/evaluation seam. Existing Arena runners "
    "fix controller roles and GPU checkpoint semantics, while the reviewed "
    "Claude/Codex boundary fixes Claude as read-only planner/critic and Codex as "
    "the workspace-writing actor; neither can produce the matched scaffold panel."
)
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,95}")


class LoopRunnerError(RuntimeError):
    """Execution would be unbound, unsafe, incomplete, or authority-seeking."""


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _bytes_sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise LoopRunnerError(f"{label} must be non-empty text without NUL")
    return value.strip()


def _sha(value: object, label: str) -> str:
    value = _text(value, label)
    if not _SHA_RE.fullmatch(value):
        raise LoopRunnerError(f"{label} must be a lowercase SHA-256")
    return value


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
        if path.is_symlink():
            rows[relative] = f"symlink:{os.readlink(path)}"
        elif path.is_dir():
            rows[relative] = "directory"
        elif path.is_file():
            rows[relative] = f"file:{_file_sha(path)}"
        else:
            rows[relative] = "special"
    return rows


@dataclass(frozen=True)
class ModelCellPin:
    """One exact contract model/quant/effort cell bound to one CLI binary."""

    provider: str
    model_id: str
    quant_id: str
    effort: str
    executable: str
    executable_sha256: str

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise LoopRunnerError("model provider must be claude or codex")
        for name in ("model_id", "quant_id", "effort"):
            _text(getattr(self, name), name)
        executable = Path(_text(self.executable, "CLI executable"))
        if not executable.is_absolute() or not executable.is_file():
            raise LoopRunnerError("CLI executable must be an existing absolute file")
        if executable != executable.resolve():
            raise LoopRunnerError("CLI executable path must be canonical")
        if executable.is_symlink() or not os.access(executable, os.X_OK):
            raise LoopRunnerError("CLI executable must be non-symlink and executable")
        _sha(self.executable_sha256, "CLI executable SHA-256")
        if _file_sha(executable) != self.executable_sha256:
            raise LoopRunnerError("CLI executable identity does not match its pin")

    @property
    def cell_key(self) -> tuple[str, str, str]:
        return self.model_id, self.quant_id, self.effort

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider, "model_id": self.model_id,
            "quant_id": self.quant_id, "effort": self.effort,
            "executable": self.executable,
            "executable_sha256": self.executable_sha256,
        }


def resolve_model_pin(*, provider: str, model_id: str, quant_id: str,
                      effort: str, environment: Mapping[str, str]) -> ModelCellPin:
    """Resolve and hash a CLI without executing it."""
    if provider not in PROVIDERS:
        raise LoopRunnerError("model provider must be claude or codex")
    executable = shutil.which(provider, path=environment.get("PATH"))
    if executable is None:
        raise LoopRunnerError(f"required planner CLI not found: {provider}")
    resolved = Path(executable).resolve()
    return ModelCellPin(
        provider, model_id, quant_id, effort, str(resolved), _file_sha(resolved))


def _argv_template(pin: ModelCellPin) -> tuple[str, ...]:
    if pin.provider == "claude":
        return (
            pin.executable, "--print", "--output-format", "json",
            "--model", pin.model_id, "--effort", pin.effort,
            "--permission-mode", "plan", "--disallowedTools",
            "Bash,Edit,Write,NotebookEdit",
        )
    return (
        pin.executable, "exec", "--model", pin.model_id,
        "--config", f'model_reasoning_effort="{pin.effort}"',
        "--config", 'approval_policy="never"', "--sandbox", "read-only",
        "--ephemeral", "--ignore-user-config", "--ignore-rules",
        "--skip-git-repo-check", "--cd", "{cell_root}",
        "--output-last-message", "{last_message}", "-",
    )


def _observation_instruction() -> str:
    return (
        "\n\nReturn exactly one JSON object and no prose. The object must have "
        f"schema={RAW_OBSERVATION_SCHEMA!r}, an explicit termination equal to "
        "already_optimized, budget_exhausted, or "
        "search_exhausted, and hypotheses as an array. Every hypothesis must "
        "contain exactly mechanism, target_surface, falsifiable_counter, and "
        "predicted_direction as non-empty strings. Do not report prefilter "
        "survival: the hash-pinned external prefilter is applied after capture."
    )


def compile_planner_manifest(
    contract: experiments.ExperimentContract, *,
    context: authoring_contract.PricedContext,
    target_lines: Mapping[str, str],
    model_pins: Sequence[ModelCellPin],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Compile all AK-LE-1/2 cells; no subprocess or model is invoked."""
    if not isinstance(contract, experiments.ExperimentContract):
        raise TypeError("contract must be an ExperimentContract")
    if not isinstance(context, authoring_contract.PricedContext):
        raise TypeError("context must be a PricedContext")
    if (isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise LoopRunnerError("cell timeout must be positive and finite")
    if not isinstance(target_lines, Mapping):
        raise TypeError("target_lines must be a mapping")
    rendered_ids = {
        arm.cell_id for arm in contract.planner_arms
        if arm.target_context_mode == experiments.TARGET_RENDERED}
    if set(target_lines) != rendered_ids:
        raise LoopRunnerError(
            "target_lines must cover every rendered planner cell exactly once")
    pins = tuple(model_pins)
    pin_map = {pin.cell_key: pin for pin in pins}
    if len(pin_map) != len(pins):
        raise LoopRunnerError("model cell pins must be unique")
    required = {
        (arm.model_id, arm.quant_id, arm.effort) for arm in contract.planner_arms}
    if set(pin_map) != required:
        raise LoopRunnerError("model cell pins must cover every planner model cell")

    cells: list[dict[str, Any]] = []
    for arm in contract.planner_arms:
        target = (target_lines[arm.cell_id]
                  if arm.target_context_mode == experiments.TARGET_RENDERED else None)
        rendered = experiments.render_planner_prompt(
            contract, arm.cell_id, context=context, target_line=target)
        prompt = rendered + _observation_instruction()
        pin = pin_map[(arm.model_id, arm.quant_id, arm.effort)]
        cells.append({
            **arm.to_dict(), "provider": pin.provider,
            "prompt": prompt, "prompt_sha256": _bytes_sha(prompt.encode("utf-8")),
            "cli": pin.to_dict(), "argv_template": list(_argv_template(pin)),
            "timeout_seconds": float(timeout_seconds),
        })
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "authority": AUTHORITY,
        "scope": "ak-le-1-2-planner-only",
        "experiment_id": contract.experiment_id,
        "experiment_contract_sha256": contract.to_manifest()["contract_sha256"],
        "experiment_contract": contract.to_manifest(),
        "retrieval_context_sha256": experiments.context_sha256(context),
        "prefilter": contract.prefilter.to_dict(),
        "cells": cells,
        "constraints": {
            "planner_workspace_write_access": False,
            "scaffold_execution_supported": False,
            "campaign_1_authority": False, "ranking_authority": False,
            "champion_authority": False, "release_authority": False,
            "model_or_kernel_invoked_by_compiler": False,
        },
        "scaffold_gap": SCAFFOLD_GAP,
    }
    payload["manifest_sha256"] = _digest(payload)
    return payload


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Validate and atomically persist a compiled execution manifest."""
    payload = validate_manifest(manifest)
    target = Path(path)
    if not target.is_absolute() or target.exists():
        raise LoopRunnerError("manifest output must be a new absolute path")
    _atomic_json(target, payload)


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise LoopRunnerError("execution manifest must be an object")
    payload = dict(manifest)
    claimed = payload.pop("manifest_sha256", None)
    _sha(claimed, "manifest_sha256")
    if _digest(payload) != claimed:
        raise LoopRunnerError("execution manifest SHA-256 does not verify")
    if set(payload) != {
            "schema", "authority", "scope", "experiment_id",
            "experiment_contract_sha256", "experiment_contract",
            "retrieval_context_sha256", "prefilter", "cells", "constraints",
            "scaffold_gap"}:
        raise LoopRunnerError("execution manifest has unknown or missing fields")
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("authority") != AUTHORITY:
        raise LoopRunnerError("execution manifest schema or authority drifted")
    if payload.get("scope") != "ak-le-1-2-planner-only":
        raise LoopRunnerError("execution manifest may contain planner cells only")
    constraints = payload.get("constraints")
    if (not isinstance(constraints, Mapping)
            or set(constraints) != {
                "planner_workspace_write_access", "scaffold_execution_supported",
                "campaign_1_authority", "ranking_authority", "champion_authority",
                "release_authority", "model_or_kernel_invoked_by_compiler"}
            or any(constraints.get(key) is not False for key in constraints)):
        raise LoopRunnerError("execution manifest requests forbidden authority")
    if payload.get("scaffold_gap") != SCAFFOLD_GAP:
        raise LoopRunnerError("execution manifest obscures the AK-LE-3 design gap")
    contract = payload.get("experiment_contract")
    if not isinstance(contract, dict):
        raise LoopRunnerError("execution manifest lacks its experiment contract")
    contract_body = dict(contract)
    contract_sha = contract_body.pop("contract_sha256", None)
    _sha(contract_sha, "experiment contract SHA-256")
    if (_digest(contract_body) != contract_sha
            or contract_sha != payload.get("experiment_contract_sha256")):
        raise LoopRunnerError("embedded experiment contract SHA-256 does not verify")
    if contract.get("experiment_id") != payload.get("experiment_id"):
        raise LoopRunnerError("execution manifest experiment identity drifted")
    fixed = contract.get("fixed")
    if (not isinstance(fixed, dict)
            or fixed.get("retrieval_context_sha256") != payload.get(
                "retrieval_context_sha256")):
        raise LoopRunnerError("execution manifest retrieval context identity drifted")
    if contract.get("prefilter") != payload.get("prefilter"):
        raise LoopRunnerError("execution manifest prefilter identity drifted")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise LoopRunnerError("execution manifest has no planner cells")
    ids: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
            "cell_id", "model_id", "quant_id", "effort", "target_context_mode",
            "provider", "prompt", "prompt_sha256", "cli", "argv_template",
            "timeout_seconds",
        }:
            raise LoopRunnerError("planner cell has unknown or missing fields")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not _ID_RE.fullmatch(cell_id) or cell_id in ids:
            raise LoopRunnerError("planner cell identifiers must be unique and safe")
        ids.add(cell_id)
        prompt = cell.get("prompt")
        if not isinstance(prompt, str) or _bytes_sha(prompt.encode("utf-8")) != cell.get(
                "prompt_sha256"):
            raise LoopRunnerError("planner prompt SHA-256 does not verify")
        cli = cell.get("cli")
        if not isinstance(cli, dict):
            raise LoopRunnerError("planner CLI identity must be an object")
        pin = ModelCellPin(**cli)
        if (pin.provider, pin.model_id, pin.quant_id, pin.effort) != (
                cell.get("provider"), cell.get("model_id"), cell.get("quant_id"),
                cell.get("effort")):
            raise LoopRunnerError("planner cell and CLI pin differ")
        if tuple(cell.get("argv_template", ())) != _argv_template(pin):
            raise LoopRunnerError("planner argv template differs from its exact CLI pin")
        timeout = cell.get("timeout_seconds")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise LoopRunnerError("planner timeout must be positive and finite")
    contract_arms = contract.get("planner_arms")
    if (not isinstance(contract_arms, list)
            or [{key: cell[key] for key in (
                    "cell_id", "model_id", "quant_id", "effort",
                    "target_context_mode")} for cell in cells] != contract_arms):
        raise LoopRunnerError("planner cells differ from the embedded experiment contract")
    payload["manifest_sha256"] = claimed
    return payload


@dataclass(frozen=True)
class ProcessCapture:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    result_text: str
    timed_out: bool
    started_at: str
    finished_at: str
    elapsed_wall_seconds: float


CommandRunner = Callable[
    [Sequence[str], Path, Mapping[str, str], str, float, Path | None], ProcessCapture]


def _live_group_members(process_group_id: int) -> tuple[int, ...]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2:].split()
            state, group = fields[0], int(fields[2])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
        if group == process_group_id and state != "Z":
            members.append(int(entry.name))
    return tuple(sorted(members))


def _terminate_group(process_group_id: int) -> None:
    if process_group_id <= 1 or process_group_id == os.getpgrp():
        raise LoopRunnerError("refusing an unsafe process-group target")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not _live_group_members(process_group_id):
            return
        try:
            os.killpg(process_group_id, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and _live_group_members(process_group_id):
            time.sleep(0.02)
    survivors = _live_group_members(process_group_id)
    if survivors:
        raise LoopRunnerError(f"planner process group survived teardown: {survivors}")


def _run_process(argv: Sequence[str], cwd: Path, environment: Mapping[str, str],
                 prompt: str, timeout_seconds: float,
                 result_path: Path | None) -> ProcessCapture:
    """Run and reap one exact captured process group."""
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise LoopRunnerError("planner argv must be non-empty strings")
    started_dt = datetime.now(timezone.utc)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(argv), cwd=cwd, env=dict(environment), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
    except OSError as exc:
        raise LoopRunnerError(f"could not start planner CLI: {argv[0]}") from exc
    timed_out = False
    stdout = stderr = ""
    try:
        stdout, stderr = process.communicate(input=prompt, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_group(process.pid)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if _live_group_members(process.pid):
            _terminate_group(process.pid)
        if process.poll() is None:
            stdout, stderr = process.communicate(timeout=5)
    finished = time.monotonic()
    finished_dt = datetime.now(timezone.utc)
    result = stdout
    if result_path is not None:
        if not result_path.is_file() or result_path.is_symlink():
            result = ""
        else:
            result = result_path.read_text(encoding="utf-8")
    return ProcessCapture(
        tuple(argv), int(process.returncode), stdout, stderr, result, timed_out,
        started_dt.isoformat(), finished_dt.isoformat(), finished - started)


def _unwrap_result(raw: str, provider: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LoopRunnerError(f"{provider} planner emitted malformed JSON") from exc
    if provider == "claude" and isinstance(payload, dict) and isinstance(
            payload.get("result"), str):
        try:
            payload = json.loads(payload["result"])
        except json.JSONDecodeError as exc:
            raise LoopRunnerError("Claude result wrapper contains malformed JSON") from exc
    if not isinstance(payload, dict):
        raise LoopRunnerError("planner observation must be one JSON object")
    return payload


def parse_raw_observation(raw: str, *, provider: str,
                          expected_cell_id: str) -> dict[str, Any]:
    payload = _unwrap_result(raw, provider)
    if set(payload) != {"schema", "termination", "hypotheses"}:
        raise LoopRunnerError("planner observation has unknown or missing fields")
    if payload.get("schema") != RAW_OBSERVATION_SCHEMA:
        raise LoopRunnerError("planner observation schema drifted")
    if not isinstance(expected_cell_id, str) or not _ID_RE.fullmatch(expected_cell_id):
        raise LoopRunnerError("trusted runner supplied an invalid cell identity")
    if payload.get("termination") not in experiments.TERMINATIONS:
        raise LoopRunnerError("planner termination is not explicit")
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise LoopRunnerError("planner hypotheses must be an array")
    parsed = []
    required = {"mechanism", "target_surface", "falsifiable_counter",
                "predicted_direction"}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict) or set(hypothesis) != required:
            raise LoopRunnerError("planner hypothesis has unknown or missing fields")
        parsed.append({key: _text(hypothesis[key], key) for key in sorted(required)})
    return {**payload, "cell_id": expected_cell_id, "hypotheses": parsed}


def _bound_argv(cell: Mapping[str, Any], cell_root: Path) -> tuple[str, ...]:
    last_message = cell_root / "last-message.json"
    return tuple(str(part).replace("{cell_root}", str(cell_root)).replace(
        "{last_message}", str(last_message)) for part in cell["argv_template"])


def run_planner_manifest(
    manifest: Mapping[str, Any], *, output_root: str | Path,
    environment: Mapping[str, str] | None = None,
    runner: CommandRunner = _run_process,
) -> dict[str, Any]:
    """Execute a complete precompiled planner panel and durably seal its evidence."""
    payload = validate_manifest(manifest)
    root = Path(output_root)
    if not root.is_absolute() or root.exists() or root.is_symlink():
        raise LoopRunnerError("output_root must be a new absolute path")
    if not callable(runner):
        raise TypeError("runner must be callable")
    root.mkdir(parents=True)
    _atomic_json(root / "manifest.json", payload)
    env = dict(os.environ if environment is None else environment)
    observations: list[dict[str, Any]] = []
    try:
        for ordinal, cell in enumerate(payload["cells"], 1):
            cell_root = root / f"{ordinal:04d}-{cell['cell_id']}"
            cell_root.mkdir()
            prompt = cell["prompt"]
            _atomic_write(cell_root / "prompt.txt", prompt.encode("utf-8"))
            argv = _bound_argv(cell, cell_root)
            if _file_sha(Path(argv[0])) != cell["cli"]["executable_sha256"]:
                raise LoopRunnerError(
                    f"planner CLI identity drifted before cell: {cell['cell_id']}")
            result_path = (cell_root / "last-message.json"
                           if cell["provider"] == "codex" else None)
            before_process = _tree_state(cell_root)
            capture = runner(
                argv, cell_root, env, prompt, cell["timeout_seconds"], result_path)
            if capture.argv != argv:
                raise LoopRunnerError("captured argv differs from the predeclared command")
            if (not math.isfinite(capture.elapsed_wall_seconds)
                    or capture.elapsed_wall_seconds <= 0):
                raise LoopRunnerError("captured elapsed wall time must be positive and finite")
            after_process = _tree_state(cell_root)
            allowed_process_writes = ({"last-message.json"}
                                      if cell["provider"] == "codex" else set())
            changed = {
                path for path in set(before_process) | set(after_process)
                if before_process.get(path) != after_process.get(path)}
            _atomic_write(cell_root / "stdout.txt", capture.stdout.encode("utf-8"))
            _atomic_write(cell_root / "stderr.txt", capture.stderr.encode("utf-8"))
            _atomic_write(cell_root / "result.txt", capture.result_text.encode("utf-8"))
            if cell["provider"] == "claude" and capture.result_text != capture.stdout:
                raise LoopRunnerError("Claude parsed result differs from captured stdout")
            if (result_path is not None and result_path.is_file()
                    and _file_sha(result_path) != _file_sha(cell_root / "result.txt")):
                raise LoopRunnerError("Codex parsed result differs from last-message bytes")
            event = {
                "cell_id": cell["cell_id"], "provider": cell["provider"],
                "model_id": cell["model_id"], "quant_id": cell["quant_id"],
                "effort": cell["effort"], "argv": list(capture.argv),
                "cli_executable_sha256": cell["cli"]["executable_sha256"],
                "prompt_sha256": _file_sha(cell_root / "prompt.txt"),
                "stdout_sha256": _file_sha(cell_root / "stdout.txt"),
                "stderr_sha256": _file_sha(cell_root / "stderr.txt"),
                "result_sha256": _file_sha(cell_root / "result.txt"),
                "returncode": capture.returncode, "timed_out": capture.timed_out,
                "started_at": capture.started_at, "finished_at": capture.finished_at,
                "elapsed_wall_seconds": capture.elapsed_wall_seconds,
                "status": "captured",
            }
            unauthorized = sorted(changed - allowed_process_writes)
            if unauthorized:
                event["undeclared_process_writes"] = unauthorized
                event["status"] = "rejected"
                _atomic_json(cell_root / "event.json", event)
                raise LoopRunnerError(
                    f"read-only planner changed undeclared paths: {unauthorized}")
            if capture.timed_out:
                event["status"] = "rejected"
                _atomic_json(cell_root / "event.json", event)
                raise LoopRunnerError(f"planner cell timed out: {cell['cell_id']}")
            if capture.returncode != 0:
                event["status"] = "rejected"
                _atomic_json(cell_root / "event.json", event)
                raise LoopRunnerError(
                    f"planner cell exited {capture.returncode}: {cell['cell_id']}")
            _atomic_json(cell_root / "event.json", event)
            observation = parse_raw_observation(
                capture.result_text, provider=cell["provider"],
                expected_cell_id=cell["cell_id"])
            _atomic_json(cell_root / "observation.json", observation)
            event["observation_sha256"] = _file_sha(cell_root / "observation.json")
            event["status"] = "parsed"
            _atomic_json(cell_root / "event.json", event)
            observations.append({**event, "observation": observation})
    except Exception as exc:
        failure = {
            "schema": PANEL_SCHEMA, "status": "failed", "authority": AUTHORITY,
            "manifest_sha256": payload["manifest_sha256"],
            "completed_cell_ids": [row["cell_id"] for row in observations],
            "error_type": type(exc).__name__, "error": str(exc),
        }
        failure["panel_sha256"] = _digest(failure)
        _atomic_json(root / "panel.json", failure)
        raise
    panel: dict[str, Any] = {
        "schema": PANEL_SCHEMA, "status": "complete", "authority": AUTHORITY,
        "experiment_id": payload["experiment_id"],
        "experiment_contract_sha256": payload["experiment_contract_sha256"],
        "manifest_sha256": payload["manifest_sha256"],
        "capture_mode": "measured_model_output",
        "observations": observations,
        "constraints": {
            "external_prefilter_applied": False,
            "scaffold_observations_present": False,
            "campaign_1_authority": False, "ranking_authority": False,
            "champion_authority": False, "release_authority": False,
        },
        "next_required_step": (
            "Apply the manifest-pinned external prefilter, bind its evidence, and "
            "materialize loop_experiments.PlannerObservation values."),
    }
    panel["panel_sha256"] = _digest(panel)
    _atomic_json(root / "panel.json", panel)
    return panel


def materialize_planner_observation(
    raw: Mapping[str, Any], *, survived_prefilter: Sequence[bool],
    elapsed_wall_seconds: float, evidence_sha256: str, provider: str,
) -> experiments.PlannerObservation:
    """Bind separately obtained prefilter results to one captured raw observation."""
    if not isinstance(raw, Mapping):
        raise TypeError("raw observation must be a mapping")
    expected_cell_id = str(raw.get("cell_id", ""))
    model_payload = {key: value for key, value in raw.items() if key != "cell_id"}
    parsed = parse_raw_observation(
        json.dumps(model_payload), provider=provider,
        expected_cell_id=expected_cell_id)
    survived = tuple(survived_prefilter)
    if len(survived) != len(parsed["hypotheses"]) or any(
            not isinstance(value, bool) for value in survived):
        raise LoopRunnerError(
            "prefilter results must provide one boolean per captured hypothesis")
    hypotheses = tuple(experiments.HypothesisObservation(
        mechanism=row["mechanism"], target_surface=row["target_surface"],
        falsifiable_counter=row["falsifiable_counter"],
        predicted_direction=row["predicted_direction"],
        survived_prefilter=passed,
    ) for row, passed in zip(parsed["hypotheses"], survived))
    return experiments.PlannerObservation(
        parsed["cell_id"], parsed["termination"], hypotheses,
        elapsed_wall_seconds, _sha(evidence_sha256, "prefilter evidence SHA-256"))


def run_scaffold_manifest(*_args: Any, **_kwargs: Any) -> None:
    """Fail closed until a matched governed AK-LE-3 authoring seam exists."""
    raise LoopRunnerError(SCAFFOLD_GAP)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise LoopRunnerError("manifest must be an existing absolute file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoopRunnerError("manifest file contains malformed JSON") from exc
    panel = run_planner_manifest(manifest, output_root=args.output_root)
    print(json.dumps(panel, sort_keys=True))
    return 0


__all__ = [
    "AUTHORITY", "MANIFEST_SCHEMA", "PANEL_SCHEMA", "RAW_OBSERVATION_SCHEMA",
    "SCAFFOLD_GAP", "LoopRunnerError", "ModelCellPin", "ProcessCapture",
    "compile_planner_manifest", "materialize_planner_observation",
    "parse_raw_observation", "resolve_model_pin", "run_planner_manifest",
    "run_scaffold_manifest", "validate_manifest", "write_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
