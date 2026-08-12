#!/usr/bin/env python3
"""Bounded Claude-planner/critic and Codex-actor controller for INF-03.

The controller is deliberately a thin orchestration boundary.  AgentKernelArena
continues to own task setup and evaluation; this module owns exact model pins,
workspace confinement, checkpoint termination, and hash-bound transcripts.  It
does not call either CLI at import time or during preflight.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from . import arena_adapter, codex_container_actor


CONTROLLER_ID = "claude_codex_actor_critic"
RECEIPT_SCHEMA = "epyc.autokernel.claude_codex_actor_critic.v1"
PROPOSAL_SCHEMA = "epyc.autokernel.actor_critic_proposal.v1"
CRITIQUE_SCHEMA = "epyc.autokernel.actor_critic_critique.v1"
PROPOSAL_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema": {"const": PROPOSAL_SCHEMA},
        "proposal_id": {
            "type": "string", "pattern": r"^[a-z][a-z0-9_.-]{2,95}$",
        },
        "candidate_path": {"type": "string", "minLength": 1},
        "actor_instruction": {"type": "string", "minLength": 1},
    },
    "required": ["schema", "proposal_id", "candidate_path", "actor_instruction"],
}
CLAUDE_MODEL = "claude-opus-5"
CLAUDE_EFFORT = "high"
CODEX_MODEL = "gpt-5.6-sol"
CODEX_EFFORT = "high"
CAMPAIGN_CHECKPOINT_HOURS = (2.0, 8.0, 32.0)
PINNED_MODEL_IDS = (
    f"{CLAUDE_MODEL}:{CLAUDE_EFFORT}:planner+critic",
    f"{CODEX_MODEL}:{CODEX_EFFORT}:actor",
)
REQUIRED_CLIS = ("claude", "codex")
ARTIFACT_DIRNAME = ".autokernel-claude-codex"
ENTRYPOINT_RELATIVE = (
    "scripts/kernel_rnd/autokernel/controller/claude_codex_actor_critic.py")
EXECUTABLE_MODULE = (
    "scripts.kernel_rnd.autokernel.controller.claude_codex_actor_critic")
_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,95}")
_JSON_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?\s*\n(?P<body>.*?)\n```\s*\Z", re.DOTALL)


class ActorCriticError(ValueError):
    """The controller request or an agent result is unsafe or malformed."""


@dataclass(frozen=True)
class ControllerConfig:
    claude_model: str
    claude_effort: str
    codex_model: str
    codex_effort: str
    checkpoint_hours: float
    timeout_seconds: int
    max_iterations: int

    def __post_init__(self) -> None:
        exact = {
            "claude_model": CLAUDE_MODEL,
            "claude_effort": CLAUDE_EFFORT,
            "codex_model": CODEX_MODEL,
            "codex_effort": CODEX_EFFORT,
        }
        for field, expected in exact.items():
            if getattr(self, field) != expected:
                raise ActorCriticError(f"{field} must be pinned to {expected!r}")
        if (isinstance(self.checkpoint_hours, bool)
                or not isinstance(self.checkpoint_hours, (int, float))
                or not math.isfinite(self.checkpoint_hours)
                or float(self.checkpoint_hours) not in CAMPAIGN_CHECKPOINT_HOURS):
            raise ActorCriticError(
                f"checkpoint_hours must be one of {CAMPAIGN_CHECKPOINT_HOURS}")
        expected_timeout = int(float(self.checkpoint_hours) * 3600)
        if (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, int)
                or self.timeout_seconds != expected_timeout):
            raise ActorCriticError(
                f"timeout_seconds must equal the {self.checkpoint_hours:g} h "
                f"campaign checkpoint ({expected_timeout})")
        if (isinstance(self.max_iterations, bool)
                or not isinstance(self.max_iterations, int)
                or not 1 <= self.max_iterations <= 256):
            raise ActorCriticError("max_iterations must be an integer in [1, 256]")

    @classmethod
    def from_mapping(cls, payload: object) -> "ControllerConfig":
        if not isinstance(payload, Mapping):
            raise ActorCriticError(
                f"eval_config.{CONTROLLER_ID} must be an object")
        required = {
            "claude_model", "claude_effort", "codex_model", "codex_effort",
            "checkpoint_hours", "timeout_seconds", "max_iterations",
        }
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        if missing:
            raise ActorCriticError(f"controller config is missing {missing}")
        if extra:
            raise ActorCriticError(f"controller config has unknown keys {extra}")
        return cls(**{key: payload[key] for key in required})


@dataclass(frozen=True)
class ProcessCapture:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str], str, float],
                         ProcessCapture]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(encoded.encode("utf-8"))


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


def _run_process(argv: Sequence[str], cwd: Path, env: Mapping[str, str],
                 input_text: str, timeout_seconds: float) -> ProcessCapture:
    """Run one child; launcher-side cgroup teardown owns timeout descendants."""
    if (not argv or any(not isinstance(part, str) or not part for part in argv)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ActorCriticError("process argv and timeout must be bounded and non-empty")
    try:
        process = subprocess.Popen(
            list(argv), cwd=cwd, env=dict(env), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise ActorCriticError(f"could not start controller CLI: {argv[0]}") from exc
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout_seconds)
        return ProcessCapture(tuple(argv), process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        partial_out = _timeout_text(exc.stdout)
        partial_err = _timeout_text(exc.stderr)
        _defer_timed_out_child(process)
        return ProcessCapture(
            tuple(argv), process.poll(), partial_out, partial_err, True)
    except BaseException:
        _defer_timed_out_child(process)
        raise


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def _defer_timed_out_child(process: subprocess.Popen[str]) -> None:
    """Close controller pipes; the launcher's cgroup verifier owns teardown."""
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()
    threading.Thread(
        target=process.wait,
        name=f"autokernel-actor-critic-child-{process.pid}",
        daemon=True,
    ).start()


def resolve_cli_identities(environment: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Resolve and hash both installed CLIs without executing either one."""
    path_value = environment.get("PATH")
    rows: dict[str, dict[str, str]] = {}
    for name in REQUIRED_CLIS:
        executable = shutil.which(name, path=path_value)
        if executable is None:
            raise ActorCriticError(f"required controller CLI not found: {name}")
        resolved = Path(executable).resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ActorCriticError(f"required controller CLI is not executable: {name}")
        rows[name] = {"path": str(resolved), "sha256": _sha256_file(resolved)}
    return rows


def _workspace_root(workspace: str | Path) -> Path:
    source = Path(workspace)
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    else:
        source = source.resolve()
    if not source.is_dir() or source.is_symlink():
        raise ActorCriticError("workspace must be an existing non-symlink directory")
    return source


def _relative_candidate(workspace: Path, value: object) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ActorCriticError("proposal.candidate_path must be a non-empty string")
    workspace = workspace.resolve()
    supplied = PurePosixPath(value.strip())
    if supplied.is_absolute():
        lexical_candidate = Path(supplied)
    else:
        if ".." in supplied.parts or "." in supplied.parts:
            raise ActorCriticError("proposal candidate escapes the Arena workspace")
        lexical_candidate = workspace / Path(*supplied.parts)
    if lexical_candidate.is_symlink():
        raise ActorCriticError(
            "proposal candidate must name an existing non-symlink workspace file")
    candidate = lexical_candidate.resolve()
    try:
        relative = candidate.relative_to(workspace)
    except ValueError as exc:
        raise ActorCriticError("proposal candidate escapes the Arena workspace") from exc
    if not candidate.is_file():
        raise ActorCriticError(
            "proposal candidate must name an existing non-symlink workspace file")
    return relative.as_posix(), candidate


def _workspace_manifest(workspace: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    artifact_root = workspace / ARTIFACT_DIRNAME
    for path in sorted(workspace.rglob("*")):
        try:
            path.relative_to(artifact_root)
        except ValueError:
            pass
        else:
            continue
        relative = path.relative_to(workspace).as_posix()
        if path.is_symlink():
            target = path.resolve()
            try:
                target.relative_to(workspace)
            except ValueError as exc:
                raise ActorCriticError(
                    f"workspace symlink escapes isolation: {relative}") from exc
            rows[relative] = _sha256_bytes(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            rows[relative] = _sha256_file(path)
    return rows


def _changed_paths(before: Mapping[str, str], after: Mapping[str, str]) -> set[str]:
    return {key for key in set(before) | set(after) if before.get(key) != after.get(key)}


def _parse_json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ActorCriticError(f"{label} emitted malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ActorCriticError(f"{label} must emit one JSON object")
    # Claude's --json-schema result wrapper carries the provider-validated
    # object in structured_output. Older captured fixtures and compatible
    # launchers may carry the exact object in result or provide it directly.
    if isinstance(payload.get("structured_output"), dict):
        payload = payload["structured_output"]
    elif set(payload) >= {"result"} and isinstance(payload["result"], str):
        inner = payload["result"]
        try:
            payload = json.loads(inner)
        except json.JSONDecodeError as direct_error:
            fenced = _JSON_FENCE_RE.fullmatch(inner)
            if fenced is None:
                raise ActorCriticError(
                    f"{label}.result emitted malformed JSON") from direct_error
            try:
                payload = json.loads(fenced.group("body"))
            except json.JSONDecodeError as fenced_error:
                raise ActorCriticError(
                    f"{label}.result emitted malformed JSON") from fenced_error
        if not isinstance(payload, dict):
            raise ActorCriticError(f"{label}.result must be one JSON object")
    return payload


def parse_proposal(raw: str, workspace: Path) -> dict[str, Any]:
    payload = _parse_json_object(raw, "planner")
    required = {"schema", "proposal_id", "candidate_path", "actor_instruction"}
    if set(payload) != required or payload.get("schema") != PROPOSAL_SCHEMA:
        raise ActorCriticError("planner proposal has unknown/missing fields or schema")
    proposal_id = payload.get("proposal_id")
    instruction = payload.get("actor_instruction")
    if not isinstance(proposal_id, str) or not _ID_RE.fullmatch(proposal_id):
        raise ActorCriticError("proposal_id is invalid")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ActorCriticError("proposal actor_instruction must be non-empty")
    relative, candidate = _relative_candidate(workspace, payload["candidate_path"])
    return {
        **payload,
        "candidate_path": relative,
        "candidate_abspath": str(candidate),
    }


def parse_critique(raw: str, proposal_id: str) -> dict[str, Any]:
    payload = _parse_json_object(raw, "critic")
    required = {"schema", "proposal_id", "decision", "reason"}
    if set(payload) != required or payload.get("schema") != CRITIQUE_SCHEMA:
        raise ActorCriticError("critic result has unknown/missing fields or schema")
    if payload.get("proposal_id") != proposal_id:
        raise ActorCriticError("critic result does not bind the proposal")
    if payload.get("decision") not in {"accept", "revise", "stop"}:
        raise ActorCriticError("critic decision must be accept, revise, or stop")
    if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
        raise ActorCriticError("critic reason must be non-empty")
    return payload


class ArtifactJournal:
    def __init__(self, workspace: Path):
        self.root = workspace / ARTIFACT_DIRNAME
        if self.root.exists() and self.root.is_symlink():
            raise ActorCriticError("controller artifact directory may not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.transcript = self.root / "transcript.jsonl"
        if self.transcript.exists():
            raise ActorCriticError("controller artifact transcript already exists")
        self._events: list[dict[str, Any]] = []

    def record(self, role: str, iteration: int, prompt: str,
               capture: ProcessCapture) -> None:
        ordinal = len(self._events) + 1
        stem = f"{ordinal:04d}-{role}-{iteration:03d}"
        stdout_path = self.root / f"{stem}.stdout"
        stderr_path = self.root / f"{stem}.stderr"
        _atomic_write(stdout_path, capture.stdout.encode("utf-8"))
        _atomic_write(stderr_path, capture.stderr.encode("utf-8"))
        event = {
            "ordinal": ordinal,
            "role": role,
            "iteration": iteration,
            "argv": list(capture.argv),
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "returncode": capture.returncode,
            "timed_out": capture.timed_out,
            "stdout": {"path": stdout_path.name, "sha256": _sha256_file(stdout_path)},
            "stderr": {"path": stderr_path.name, "sha256": _sha256_file(stderr_path)},
        }
        self._events.append(event)
        rendered = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in self._events)
        _atomic_write(self.transcript, rendered)

    def receipt(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        artifact_hashes = {
            path.relative_to(self.root).as_posix(): _sha256_file(path)
            for path in sorted(self.root.iterdir()) if path.is_file()
        }
        result = {**dict(payload), "artifact_sha256": artifact_hashes}
        result["receipt_sha256"] = _canonical_sha256(result)
        _atomic_write(
            self.root / "receipt.json",
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        return result


def _critique_json_schema(proposal_id: str) -> dict[str, Any]:
    """Return the critic schema with the current proposal binding enforced upstream."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema": {"const": CRITIQUE_SCHEMA},
            "proposal_id": {"const": proposal_id},
            "decision": {"enum": ["accept", "revise", "stop"]},
            "reason": {"type": "string", "minLength": 1},
        },
        "required": ["schema", "proposal_id", "decision", "reason"],
    }


def _claude_argv(identity: Mapping[str, str], config: ControllerConfig,
                 output_schema: Mapping[str, Any]) -> tuple[str, ...]:
    schema_text = json.dumps(
        dict(output_schema), sort_keys=True, separators=(",", ":"))
    return (
        identity["path"], "--print", "--output-format", "json",
        "--json-schema", schema_text,
        "--model", config.claude_model, "--effort", config.claude_effort,
        "--permission-mode", "plan", "--disallowedTools", "Bash,Edit,Write,NotebookEdit",
    )


def _codex_argv(identity: Mapping[str, str], config: ControllerConfig,
                workspace: Path) -> tuple[str, ...]:
    return (
        sys.executable, "-m", codex_container_actor.EXECUTABLE_MODULE,
        "--codex-wrapper", identity["path"], "--workspace", str(workspace),
        "--model", config.codex_model, "--effort", config.codex_effort,
    )


def campaign_argv(executable: str = "python3") -> tuple[str, ...]:
    """Return the exact maximum-checkpoint stdin executable bound by INF-03."""
    if not isinstance(executable, str) or not executable:
        raise ActorCriticError("campaign executable must be non-empty")
    return (
        executable, "-m", EXECUTABLE_MODULE,
        "--claude-model", CLAUDE_MODEL, "--claude-effort", CLAUDE_EFFORT,
        "--codex-model", CODEX_MODEL, "--codex-effort", CODEX_EFFORT,
        "--checkpoint-hours", "32", "--timeout-seconds", "115200",
        "--max-iterations", "64", "--workspace", ".",
    )


def run_controller(
    *, prompt: str, workspace: str | Path, config: ControllerConfig,
    environment: Mapping[str, str] | None = None,
    runner: CommandRunner = _run_process,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run the bounded actor/critic loop and return its hash-bound receipt."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ActorCriticError("Arena prompt must be non-empty")
    if not isinstance(config, ControllerConfig):
        raise TypeError("config must be ControllerConfig")
    if not callable(runner) or not callable(monotonic):
        raise TypeError("runner and monotonic must be callable")
    root = _workspace_root(workspace)
    env = arena_adapter.architecture_environment(
        os.environ if environment is None else environment)
    cli = resolve_cli_identities(env)
    initial = _workspace_manifest(root)
    journal = ArtifactJournal(root)
    started = monotonic()
    deadline = started + config.timeout_seconds
    proposals: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    stop_reason = "max_iterations"

    for iteration in range(1, config.max_iterations + 1):
        remaining = deadline - monotonic()
        if remaining <= 0:
            stop_reason = "campaign_checkpoint"
            break
        planner_prompt = (
            f"{prompt}\n\nYou are the planner for iteration {iteration}. Return only JSON "
            f"with schema {PROPOSAL_SCHEMA}, proposal_id, candidate_path, and "
            "actor_instruction. candidate_path must name one existing non-symlink "
            "file under the supplied Arena workspace; use a workspace-relative "
            "path when possible, although an exact contained absolute path is "
            "accepted. Do not edit files."
        )
        before_planner = _workspace_manifest(root)
        capture = runner(
            _claude_argv(cli["claude"], config, PROPOSAL_JSON_SCHEMA),
            root, env, planner_prompt, remaining)
        journal.record("planner", iteration, planner_prompt, capture)
        if _workspace_manifest(root) != before_planner:
            raise ActorCriticError("planner changed the isolated Arena workspace")
        if capture.timed_out:
            stop_reason = "campaign_checkpoint"
            break
        if capture.returncode != 0:
            raise ActorCriticError(f"planner CLI exited {capture.returncode}")
        proposal = parse_proposal(capture.stdout, root)
        proposals.append({key: value for key, value in proposal.items()
                          if key != "candidate_abspath"})
        candidate_path = Path(proposal["candidate_abspath"])
        before_actor = _workspace_manifest(root)
        before_sha = _sha256_file(candidate_path)
        actor_prompt = (
            f"{prompt}\n\nYou are the actor. Implement only proposal "
            f"{proposal['proposal_id']} in {proposal['candidate_path']}. "
            f"Do not modify any other workspace path.\n\n"
            f"{proposal['actor_instruction']}"
        )
        remaining = deadline - monotonic()
        if remaining <= 0:
            stop_reason = "campaign_checkpoint"
            break
        capture = runner(
            _codex_argv(cli["codex"], config, root), root, env, actor_prompt, remaining)
        journal.record("actor", iteration, actor_prompt, capture)
        after_actor = _workspace_manifest(root)
        changed = _changed_paths(before_actor, after_actor)
        if changed - {proposal["candidate_path"]}:
            raise ActorCriticError(
                f"actor changed paths outside its candidate: {sorted(changed)}")
        if capture.timed_out:
            stop_reason = "campaign_checkpoint"
            break
        if capture.returncode != 0:
            raise ActorCriticError(f"actor CLI exited {capture.returncode}")
        after_sha = _sha256_file(candidate_path)
        candidate_rows.append({
            "iteration": iteration,
            "proposal_id": proposal["proposal_id"],
            "path": proposal["candidate_path"],
            "before_sha256": before_sha,
            "after_sha256": after_sha,
        })
        critic_prompt = (
            f"{prompt}\n\nYou are the critic. Review proposal "
            f"{proposal['proposal_id']} for {proposal['candidate_path']}. The candidate "
            f"changed from SHA-256 {before_sha} to {after_sha}. Return only JSON with "
            f"schema {CRITIQUE_SCHEMA}. The object must contain exactly these four "
            "fields and no others: schema, proposal_id (the same value), decision "
            "(accept, revise, or stop), and a non-empty reason. Do not edit files."
        )
        remaining = deadline - monotonic()
        if remaining <= 0:
            stop_reason = "campaign_checkpoint"
            break
        capture = runner(
            _claude_argv(
                cli["claude"], config,
                _critique_json_schema(proposal["proposal_id"])),
            root, env, critic_prompt, remaining)
        journal.record("critic", iteration, critic_prompt, capture)
        if _workspace_manifest(root) != after_actor:
            raise ActorCriticError("critic changed the isolated Arena workspace")
        if capture.timed_out:
            stop_reason = "campaign_checkpoint"
            break
        if capture.returncode != 0:
            raise ActorCriticError(f"critic CLI exited {capture.returncode}")
        critique = parse_critique(capture.stdout, proposal["proposal_id"])
        if critique["decision"] in {"accept", "stop"}:
            stop_reason = f"critic_{critique['decision']}"
            break

    final = _workspace_manifest(root)
    receipt = journal.receipt({
        "schema": RECEIPT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "authority": "whole_agent_task_only",
        "status": "complete",
        "stop_reason": stop_reason,
        "workspace": str(root),
        "workspace_initial_sha256": _canonical_sha256(initial),
        "workspace_final_sha256": _canonical_sha256(final),
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "checkpoint_hours": float(config.checkpoint_hours),
        "timeout_seconds": config.timeout_seconds,
        "max_iterations": config.max_iterations,
        "models": {
            "planner_critic": {"cli": "claude", "model": config.claude_model,
                               "effort": config.claude_effort, **cli["claude"]},
            "actor": {"cli": "codex", "model": config.codex_model,
                      "effort": config.codex_effort, **cli["codex"]},
        },
        "proposal_sha256": [_canonical_sha256(row) for row in proposals],
        "candidate_artifacts": candidate_rows,
        "constraints": {
            "workspace_only": True,
            "planner_critic_write_access": False,
            "actor_sandbox": "docker_workspace_bind_only",
            "actor_runtime": codex_container_actor.runtime_identity(
                Path(cli["codex"]["path"])),
            "promotion_authority": False,
            "model_or_kernel_invoked_by_preflight": False,
        },
    })
    return receipt


def register_agentkernelarena_launcher(
    register_agent: Callable[[str], Callable[[Callable[..., str]], Callable[..., str]]],
    prompt_builder: Callable[[Mapping[str, Any], str, str], str],
    *, runner: CommandRunner = _run_process,
    environment: Mapping[str, str] | None = None,
) -> Callable[[Mapping[str, Any], str, str], str]:
    """Register the exact ``(eval_config, task_config_dir, workspace)`` launcher."""
    if not callable(register_agent) or not callable(prompt_builder):
        raise TypeError("register_agent and prompt_builder must be callable")

    @register_agent(CONTROLLER_ID)
    def arena_launcher(eval_config: Mapping[str, Any], task_config_dir: str,
                       workspace: str) -> str:
        if not isinstance(eval_config, Mapping):
            raise ActorCriticError("AgentKernelArena eval_config must be an object")
        config = ControllerConfig.from_mapping(eval_config.get(CONTROLLER_ID))
        prompt = prompt_builder(eval_config, task_config_dir, workspace)
        receipt = run_controller(
            prompt=prompt, workspace=workspace, config=config, runner=runner,
            environment=environment)
        return json.dumps(receipt, sort_keys=True)

    return arena_launcher


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-model", required=True)
    parser.add_argument("--claude-effort", required=True)
    parser.add_argument("--codex-model", required=True)
    parser.add_argument("--codex-effort", required=True)
    parser.add_argument("--checkpoint-hours", required=True, type=float)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    prompt = sys.stdin.read()
    config = ControllerConfig(
        claude_model=args.claude_model, claude_effort=args.claude_effort,
        codex_model=args.codex_model, codex_effort=args.codex_effort,
        checkpoint_hours=args.checkpoint_hours, timeout_seconds=args.timeout_seconds,
        max_iterations=args.max_iterations,
    )
    receipt = run_controller(prompt=prompt, workspace=args.workspace, config=config)
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "ARTIFACT_DIRNAME", "CAMPAIGN_CHECKPOINT_HOURS", "CLAUDE_EFFORT",
    "CLAUDE_MODEL", "CODEX_EFFORT", "CODEX_MODEL", "CONTROLLER_ID",
    "CRITIQUE_SCHEMA", "ENTRYPOINT_RELATIVE", "EXECUTABLE_MODULE",
    "PINNED_MODEL_IDS", "PROPOSAL_JSON_SCHEMA", "PROPOSAL_SCHEMA",
    "RECEIPT_SCHEMA", "REQUIRED_CLIS",
    "ActorCriticError", "ControllerConfig", "ProcessCapture", "campaign_argv",
    "parse_critique", "parse_proposal", "register_agentkernelarena_launcher",
    "resolve_cli_identities", "run_controller",
]


if __name__ == "__main__":
    raise SystemExit(main())
