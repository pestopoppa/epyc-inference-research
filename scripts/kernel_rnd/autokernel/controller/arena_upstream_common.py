#!/usr/bin/env python3
"""Shared governed substrate for licensed upstream Arena controller ports.

The upstream project owns the proposal/search algorithm.  This module supplies
only the two host-specific dependencies those algorithms need: a deadline-bound
text model and the pinned AgentKernelArena evaluator over one isolated task
workspace.  Importing it performs no model, compiler, evaluator, or GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

from . import arena_adapter
from .arena_cell_runner import _terminate_captured_process_group


MODEL_ID = "gpt-5.6-sol"
MODEL_EFFORT = "high"
PINNED_MODEL_IDS = (f"{MODEL_ID}:{MODEL_EFFORT}:upstream-controller",)
REQUIRED_CLIS = ("codex",)
ARTIFACT_DIRNAME = ".autokernel-upstream-controller"
_SAFE_FILE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{0,255}")


class UpstreamControllerError(RuntimeError):
    """A licensed controller port cannot preserve its declared evidence bounds."""


class ControllerBudgetExpired(UpstreamControllerError):
    """The matched controller wall-time budget has been exhausted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def workspace_root(value: str | Path) -> Path:
    root = Path(value)
    root = (Path.cwd() / root).resolve() if not root.is_absolute() else root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise UpstreamControllerError(
            "workspace must be an existing non-symlink directory")
    return root


@dataclass(frozen=True)
class ControllerBudget:
    checkpoint_hours: float
    timeout_seconds: int
    reserve_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (isinstance(self.checkpoint_hours, bool)
                or not isinstance(self.checkpoint_hours, (int, float))
                or float(self.checkpoint_hours) not in (2.0, 8.0, 32.0)):
            raise UpstreamControllerError(
                "checkpoint_hours must be one of 2, 8, or 32")
        expected = int(float(self.checkpoint_hours) * 3600)
        if self.timeout_seconds != expected:
            raise UpstreamControllerError(
                "timeout_seconds must equal the matched checkpoint")
        if (not math.isfinite(self.reserve_seconds)
                or not 0 <= self.reserve_seconds < self.timeout_seconds):
            raise UpstreamControllerError("reserve_seconds is invalid")


class CodexTextModel:
    """OpenAI-compatible text surface backed by a read-only Codex CLI call."""

    def __init__(
        self, *, workspace: Path, budget: ControllerBudget,
        environment: Mapping[str, str] | None = None,
        monotonic: Any = time.monotonic,
    ):
        self.workspace = workspace_root(workspace)
        self.budget = budget
        self.environment = arena_adapter.architecture_environment(
            os.environ if environment is None else environment)
        executable = shutil.which("codex", path=self.environment.get("PATH"))
        if executable is None:
            raise UpstreamControllerError("required controller CLI not found: codex")
        self.executable = Path(executable).resolve()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise UpstreamControllerError("resolved Codex CLI is not executable")
        self.cli_sha256 = _sha256_file(self.executable)
        self._monotonic = monotonic
        self._deadline = monotonic() + budget.timeout_seconds
        self.artifact_root = self.workspace / ARTIFACT_DIRNAME
        if self.artifact_root.exists():
            raise UpstreamControllerError(
                f"controller artifact root already exists: {self.artifact_root}")
        self.artifact_root.mkdir()
        self._calls: list[dict[str, Any]] = []

        responses = SimpleNamespace(create=self._responses_create)
        completions = SimpleNamespace(create=self._chat_create)
        self.openai_compat = SimpleNamespace(
            responses=responses,
            chat=SimpleNamespace(completions=completions),
        )

    def remaining_seconds(self) -> float:
        return max(0.0, self._deadline - self._monotonic())

    def _argv(self, output: Path) -> tuple[str, ...]:
        return (
            str(self.executable), "exec", "--model", MODEL_ID,
            "--config", f'model_reasoning_effort="{MODEL_EFFORT}"',
            "--config", 'approval_policy="never"',
            "--sandbox", "read-only", "--ephemeral", "--ignore-user-config",
            "--ignore-rules", "--skip-git-repo-check", "--cd",
            str(self.workspace), "--output-last-message", str(output), "-",
        )

    def call(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise UpstreamControllerError("model prompt must be non-empty")
        remaining = self.remaining_seconds() - self.budget.reserve_seconds
        if remaining <= 0:
            raise ControllerBudgetExpired("controller checkpoint reached")
        ordinal = len(self._calls) + 1
        output = self.artifact_root / f"{ordinal:04d}-model-output.txt"
        stderr_path = self.artifact_root / f"{ordinal:04d}-model-stderr.txt"
        process = subprocess.Popen(
            self._argv(output), cwd=self.workspace, env=self.environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        timed_out = False
        stdout = ""
        stderr = ""
        try:
            stdout, stderr = process.communicate(input=prompt, timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            cleanup_error: Exception | None = None
            try:
                _terminate_captured_process_group(process.pid)
            except Exception as exc:  # reap the exact child before surfacing teardown
                cleanup_error = exc
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate(timeout=5)
            if cleanup_error is not None:
                raise cleanup_error
        stderr_path.write_text(stderr, encoding="utf-8")
        event = {
            "ordinal": ordinal,
            "model": MODEL_ID,
            "effort": MODEL_EFFORT,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "argv": list(self._argv(output)),
            "returncode": process.returncode,
            "timed_out": timed_out,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": _sha256_file(stderr_path),
        }
        self._calls.append(event)
        _atomic_json(self.artifact_root / "transcript.json", {"calls": self._calls})
        if timed_out:
            raise ControllerBudgetExpired("Codex call reached the campaign checkpoint")
        if process.returncode != 0:
            raise UpstreamControllerError(
                f"Codex CLI exited {process.returncode}: {stderr[-500:]}")
        if not output.is_file():
            raise UpstreamControllerError("Codex CLI emitted no last-message artifact")
        result = output.read_text(encoding="utf-8").strip()
        if not result:
            raise UpstreamControllerError("Codex CLI emitted an empty last message")
        event["output_sha256"] = _sha256_file(output)
        _atomic_json(self.artifact_root / "transcript.json", {"calls": self._calls})
        return result

    def _responses_create(self, **kwargs: Any) -> Any:
        prompt = kwargs.get("input")
        return SimpleNamespace(output_text=self.call(str(prompt or "")))

    def _chat_create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not messages:
            raise UpstreamControllerError("chat request has no messages")
        prompt = "\n\n".join(
            str(row.get("content", "")) for row in messages
            if isinstance(row, Mapping))
        content = self.call(prompt)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def identity(self) -> dict[str, Any]:
        return {
            "cli": "codex", "path": str(self.executable),
            "sha256": self.cli_sha256, "model": MODEL_ID,
            "effort": MODEL_EFFORT, "call_count": len(self._calls),
        }


@dataclass(frozen=True)
class EvaluationRecord:
    passed: bool
    latency_ms: float | None
    speedup: float | None
    log_excerpt: str
    raw: Mapping[str, Any]

    @property
    def score(self) -> float:
        if not self.passed:
            return -1.0
        if self.speedup is not None and self.speedup > 0:
            return self.speedup
        if self.latency_ms is not None and self.latency_ms > 0:
            return 1.0 / self.latency_ms
        return -1.0


class ArenaWorkspaceEvaluator:
    """Evaluate candidate files only through the pinned Arena implementation."""

    def __init__(self, *, workspace: Path, arena_root: Path):
        self.workspace = workspace_root(workspace)
        self.arena_root = Path(arena_root).resolve()
        if not self.arena_root.is_dir():
            raise UpstreamControllerError("arena_root must be an existing directory")
        config_path = self.workspace / "config.yaml"
        if not config_path.is_file():
            raise UpstreamControllerError("Arena workspace lacks config.yaml")
        sys.path.insert(0, str(self.arena_root))
        try:
            import yaml  # type: ignore[import-not-found]
            from src import evaluator as vendor_evaluator  # type: ignore[import-not-found]
        except ImportError as exc:
            raise UpstreamControllerError(
                "cannot import pinned AgentKernelArena evaluator") from exc
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise UpstreamControllerError("Arena task config must be an object")
        self.config = config
        self.vendor = vendor_evaluator
        self.log_path = self.workspace / ARTIFACT_DIRNAME / "arena-evaluator.log"
        self.log_path.parent.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"autokernel.upstream.{os.getpid()}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.addHandler(logging.FileHandler(self.log_path, encoding="utf-8"))
        self.source_paths = self._discover_source_paths()
        self._starting = {
            path: (self.workspace / path).read_bytes() for path in self.source_paths}
        passed, error = self.vendor.evaluate_compilation(
            self.workspace, self.config, self.logger, None)
        if not passed:
            raise UpstreamControllerError(
                f"Arena starting task does not compile: {error}")
        self.baseline_cases = self.vendor.measure_baseline(
            self.workspace, self.config, self.logger, None)
        self.best_files = dict(self._starting)
        self.best_score = 1.0
        self.last_record: EvaluationRecord | None = None
        self.evaluation_count = 0

    def _discover_source_paths(self) -> tuple[str, ...]:
        declared = self.config.get("source_file_path")
        rows: list[str] = []
        if isinstance(declared, str) and declared.strip():
            rows = [declared.strip()]
        elif isinstance(declared, list):
            rows = [str(value).strip() for value in declared if value]
        if not rows:
            targets = self.config.get("target_kernel_functions")
            names = ([str(value) for value in targets]
                     if isinstance(targets, list) else [str(targets or "")])
            candidates = []
            for path in sorted(self.workspace.glob("*.py")):
                text = path.read_text(encoding="utf-8", errors="replace")
                if names and all(re.search(
                        rf"\bdef\s+{re.escape(name)}\s*\(", text)
                        for name in names if name):
                    candidates.append(path.name)
            if len(candidates) != 1:
                raise UpstreamControllerError(
                    f"could not uniquely discover Arena source file: {candidates}")
            rows = candidates
        clean = []
        for value in rows:
            if not _SAFE_FILE.fullmatch(value):
                raise UpstreamControllerError("Arena source path is unsafe")
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise UpstreamControllerError("Arena source path escapes workspace")
            path = (self.workspace / Path(*relative.parts)).resolve()
            try:
                path.relative_to(self.workspace)
            except ValueError as exc:
                raise UpstreamControllerError(
                    "Arena source path escapes workspace") from exc
            if not path.is_file() or path.is_symlink():
                raise UpstreamControllerError(
                    f"Arena source file is missing or symlinked: {value}")
            clean.append(relative.as_posix())
        return tuple(clean)

    def definition(self, prompt: str) -> str:
        sources = "\n\n".join(
            f"### COMPLETE FILE {path}\n"
            f"{(self.workspace / path).read_text(encoding='utf-8')}"
            for path in self.source_paths)
        return (
            f"{prompt.strip()}\n\n"
            "Return complete replacement contents for every named source file; do not "
            "return a function body or patch fragment. Preserve the public function "
            "signatures and test harness. The target is AMD MI210 gfx90a, wave64.\n\n"
            f"{sources}")

    def _materialize(self, files: Mapping[str, bytes]) -> None:
        for relative, content in files.items():
            if relative not in self.source_paths:
                raise UpstreamControllerError(
                    f"candidate attempted undeclared source path {relative}")
            path = self.workspace / relative
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

    def evaluate(self, files: Mapping[str, str]) -> EvaluationRecord:
        if set(files) != set(self.source_paths):
            raise UpstreamControllerError(
                f"candidate files must be exactly {list(self.source_paths)}")
        candidate = {
            path: text.encode("utf-8") for path, text in files.items()
            if isinstance(text, str) and text.strip()}
        if set(candidate) != set(self.source_paths):
            raise UpstreamControllerError("candidate source content must be non-empty")
        self._materialize(candidate)
        self.evaluation_count += 1
        raw = self.vendor.evaluate_kernel(
            self.workspace, self.config, self.baseline_cases, self.logger, None)
        if not isinstance(raw, Mapping):
            raise UpstreamControllerError("Arena evaluator returned a non-object")
        passed = bool(raw.get("pass_compilation") and raw.get("pass_correctness")
                      and int(raw.get("valid_optimized_cases", 0)) > 0)
        latency = raw.get("best_optimized_execution_time")
        speedup = raw.get("average_speedup")
        latency_value = (float(latency) if isinstance(latency, (int, float))
                         and float(latency) > 0 else None)
        speedup_value = (float(speedup) if isinstance(speedup, (int, float))
                         and float(speedup) > 0 else None)
        errors = [str(raw.get(key) or "") for key in (
            "compilation_error_message", "correctness_error_message")]
        record = EvaluationRecord(
            passed=passed, latency_ms=latency_value, speedup=speedup_value,
            log_excerpt="\n".join(value for value in errors if value)[-2000:],
            raw=dict(raw))
        if record.score > self.best_score:
            self.best_score = record.score
            self.best_files = dict(candidate)
        self._materialize(self.best_files)
        self.last_record = record
        return record

    def materialize_best(self) -> None:
        self._materialize(self.best_files)

    def receipt_fields(self) -> dict[str, Any]:
        return {
            "arena_root": str(self.arena_root),
            "source_paths": list(self.source_paths),
            "evaluation_count": self.evaluation_count,
            "best_score": self.best_score,
            "best_source_sha256": {
                path: hashlib.sha256(content).hexdigest()
                for path, content in sorted(self.best_files.items())},
            "evaluator_log_sha256": _sha256_file(self.log_path),
        }


def build_controller_receipt(
    *, controller_id: str, source_root: Path, source_commit: str,
    entrypoint: Path, model: CodexTextModel,
    evaluator: ArenaWorkspaceEvaluator, stop_reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "epyc.autokernel.upstream_arena_controller.v1",
        "status": "complete",
        "authority": "whole_agent_task_only",
        "controller_id": controller_id,
        "stop_reason": stop_reason,
        "source": {
            "root": str(source_root.resolve()), "commit": source_commit,
            "entrypoint": str(entrypoint.resolve()),
            "entrypoint_sha256": _sha256_file(entrypoint.resolve()),
        },
        "model": model.identity(),
        "evaluation": evaluator.receipt_fields(),
        "constraints": {
            "upstream_search_algorithm_retained": True,
            "centralized_arena_evaluator": True,
            "agent_reported_performance_admitted": False,
            "promotion_authority": False,
        },
        **({"extra": dict(extra)} if extra else {}),
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    _atomic_json(model.artifact_root / "receipt.json", payload)
    return payload


__all__ = [
    "ARTIFACT_DIRNAME", "MODEL_EFFORT", "MODEL_ID", "PINNED_MODEL_IDS",
    "REQUIRED_CLIS", "ArenaWorkspaceEvaluator", "CodexTextModel",
    "ControllerBudget", "ControllerBudgetExpired", "EvaluationRecord",
    "UpstreamControllerError", "build_controller_receipt", "workspace_root",
]
