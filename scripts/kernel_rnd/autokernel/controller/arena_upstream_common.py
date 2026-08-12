#!/usr/bin/env python3
"""Shared governed substrate for licensed upstream Arena controller ports.

The upstream project owns the proposal/search algorithm.  This module supplies
only the two host-specific dependencies those algorithms need: a deadline-bound
text model and the pinned AgentKernelArena evaluator over one isolated task
workspace.  Importing it performs no model, compiler, evaluator, or GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from . import arena_adapter
from ..execution import sandbox
MODEL_ID = "gpt-5.6-sol"
MODEL_EFFORT = "high"
PINNED_MODEL_IDS = (f"{MODEL_ID}:{MODEL_EFFORT}:upstream-controller",)
REQUIRED_CLIS = ("codex",)
ARTIFACT_DIRNAME = ".autokernel-upstream-controller"
BROKER_SOCKET_ENV = "AUTOKERNEL_ARENA_EVALUATION_BROKER_SOCKET"
BROKER_TOKEN_ENV = "AUTOKERNEL_ARENA_EVALUATION_BROKER_TOKEN"
BROKER_OWNER_PID_ENV = "AUTOKERNEL_ARENA_EVALUATION_BROKER_OWNER_PID"
ARENA_SOURCE_PATHS_ENV = "AUTOKERNEL_ARENA_SOURCE_PATHS_JSON"
BROKER_REQUEST_SCHEMA = "epyc.autokernel.arena_controller_evaluation_request.v1"
BROKER_RESULT_SCHEMA = "epyc.autokernel.arena_controller_evaluation.v1"
MODEL_BROKER_REQUEST_SCHEMA = "epyc.autokernel.arena_model_request.v1"
MODEL_BROKER_RESULT_SCHEMA = "epyc.autokernel.arena_model_result.v1"
_MAX_BROKER_MESSAGE_BYTES = 16 * 1024 * 1024
_SAFE_FILE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{0,255}")


class UpstreamControllerError(RuntimeError):
    """A licensed controller port cannot preserve its declared evidence bounds."""


class ControllerBudgetExpired(UpstreamControllerError):
    """The matched controller wall-time budget has been exhausted."""


_MODEL_BROKER_IO_LOCK = threading.Lock()


class ModelBrokerClient:
    """Authenticated model-inference client over the wrapper-inherited stream."""

    def __init__(self, environment: Mapping[str, str]):
        inherited_fd = environment.get(sandbox.BROKER_FD_ENV)
        token = environment.get(BROKER_TOKEN_ENV)
        owner = environment.get(BROKER_OWNER_PID_ENV)
        workspace = environment.get("AUTOKERNEL_CONTROLLER_WORKSPACE")
        if not inherited_fd or not token or not owner or not workspace:
            raise UpstreamControllerError(
                "controller lacks its inherited model broker identity")
        try:
            descriptor = int(inherited_fd)
            self.stream = socket.socket(fileno=os.dup(descriptor))
            self.owner_pid = int(owner)
        except (OSError, ValueError) as exc:
            raise UpstreamControllerError(
                "controller inherited model broker descriptor is invalid") from exc
        self.stream.set_inheritable(False)
        self.token = token
        self.workspace = workspace
        self._ordinal = 0

    def close(self) -> None:
        self.stream.close()

    def call(
        self, *, kind: str, argv: Sequence[str], prompt: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if (kind not in {"claude_json", "codex_actor", "codex_text"}
                or not isinstance(prompt, str) or not prompt.strip()
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
                or not argv or any(not isinstance(row, str) or not row for row in argv)):
            raise UpstreamControllerError("model broker request is invalid")
        self._ordinal += 1
        request = {
            "schema": MODEL_BROKER_REQUEST_SCHEMA,
            "token": self.token,
            "owner_pid": self.owner_pid,
            "workspace": self.workspace,
            "model_call_ordinal": self._ordinal,
            "kind": kind,
            "argv": list(argv),
            "prompt": prompt,
            "timeout_seconds": float(timeout_seconds),
        }
        encoded = json.dumps(
            request, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > _MAX_BROKER_MESSAGE_BYTES:
            raise UpstreamControllerError("model broker request exceeds its size limit")
        with _MODEL_BROKER_IO_LOCK:
            try:
                _write_all(self.stream, struct.pack("!Q", len(encoded)) + encoded)
                length = struct.unpack("!Q", _recv_exact(self.stream, 8))[0]
                if length > _MAX_BROKER_MESSAGE_BYTES:
                    raise UpstreamControllerError(
                        "model broker response exceeds its size limit")
                response = json.loads(_recv_exact(self.stream, length))
            except (OSError, json.JSONDecodeError, struct.error) as exc:
                raise UpstreamControllerError("model broker IPC failed") from exc
        if isinstance(response, Mapping) and response.get("status") == "error":
            raise UpstreamControllerError(
                "model broker failed: " + str(response.get("error")))
        claimed = response.get("receipt_sha256") if isinstance(response, dict) else None
        without_hash = ({key: value for key, value in response.items()
                         if key != "receipt_sha256"}
                        if isinstance(response, dict) else {})
        if (not isinstance(response, dict)
                or response.get("schema") != MODEL_BROKER_RESULT_SCHEMA
                or claimed != _canonical_sha256(without_hash)
                or response.get("model_call_ordinal") != self._ordinal
                or response.get("kind") != kind
                or response.get("workspace") != self.workspace
                or response.get("prompt_sha256")
                != hashlib.sha256(prompt.encode()).hexdigest()):
            raise UpstreamControllerError("model broker response identity is invalid")
        return response


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _recv_exact(stream: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise UpstreamControllerError("Arena broker closed a partial message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(stream: socket.socket, payload: bytes) -> None:
    """Write to the preconnected broker without a blocked destination syscall."""
    remaining = memoryview(payload)
    while remaining:
        try:
            written = os.write(stream.fileno(), remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("broker stream write returned no progress")
        remaining = remaining[written:]


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


def _assert_gpu_devices_inaccessible() -> None:
    """Prove controller deliberation cannot open either ROCm device surface."""
    devices = (Path("/dev/kfd"), Path("/dev/dri/renderD128"))
    denied = {errno.EACCES, errno.EPERM}
    for device in devices:
        for flags in (os.O_RDONLY, os.O_RDWR):
            try:
                descriptor = os.open(device, flags)
            except OSError as exc:
                if exc.errno not in denied:
                    raise UpstreamControllerError(
                        f"controller device-isolation probe failed: {device}: "
                        f"errno={exc.errno}") from exc
            else:
                os.close(descriptor)
                raise UpstreamControllerError(
                    f"controller is not device-isolated: opened {device}")


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def _defer_timed_out_child(process: subprocess.Popen[str]) -> None:
    """Release pipes and let the parent controller cgroup own final teardown.

    A sandboxed controller cannot inspect ``/proc`` or send signals.  The
    launcher-side lifecycle verifier owns the controller cgroup and removes
    any descendants after the controller exits.  A daemon waiter keeps the
    ``Popen`` object live long enough to reap a child that exits on its own,
    without delaying controller exit when the cgroup must do the cleanup.
    """
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()
    threading.Thread(
        target=process.wait,
        name=f"autokernel-controller-child-{process.pid}",
        daemon=True,
    ).start()


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
        source_environment = os.environ if environment is None else environment
        self.environment = arena_adapter.architecture_environment(source_environment)
        self._model_broker = (
            ModelBrokerClient(source_environment)
            if source_environment.get(sandbox.BROKER_FD_ENV) else None)
        self.environment.update({
            "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
        })
        for key in (BROKER_SOCKET_ENV, BROKER_TOKEN_ENV, BROKER_OWNER_PID_ENV,
                    sandbox.BROKER_FD_ENV):
            self.environment.pop(key, None)
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
        model_broker = getattr(self, "_model_broker", None)
        if model_broker is not None:
            brokered = model_broker.call(
                kind="codex_text", argv=self._argv(output), prompt=prompt,
                timeout_seconds=remaining)
            returncode = brokered.get("returncode")
            timed_out = bool(brokered.get("timed_out"))
            stdout = str(brokered.get("stdout", ""))
            stderr = str(brokered.get("stderr", ""))
        else:
            process = subprocess.Popen(
                self._argv(output), cwd=self.workspace, env=self.environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True)
            timed_out = False
            stdout = ""
            stderr = ""
            try:
                stdout, stderr = process.communicate(input=prompt, timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = _timeout_text(exc.stdout)
                stderr = _timeout_text(exc.stderr)
                _defer_timed_out_child(process)
            except BaseException:
                _defer_timed_out_child(process)
                raise
            returncode = process.returncode
        stderr_path.write_text(stderr, encoding="utf-8")
        event = {
            "ordinal": ordinal,
            "model": MODEL_ID,
            "effort": MODEL_EFFORT,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "argv": list(self._argv(output)),
            "returncode": returncode,
            "timed_out": timed_out,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": _sha256_file(stderr_path),
        }
        self._calls.append(event)
        _atomic_json(self.artifact_root / "transcript.json", {"calls": self._calls})
        if timed_out:
            raise ControllerBudgetExpired("Codex call reached the campaign checkpoint")
        if returncode != 0:
            raise UpstreamControllerError(
                f"Codex CLI exited {returncode}: {stderr[-500:]}")
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
    """Evaluate exact candidate files only through the parent Arena broker."""

    def __init__(
        self, *, workspace: Path, arena_root: Path,
        source_paths: Sequence[str],
    ):
        self.workspace = workspace_root(workspace)
        _assert_gpu_devices_inaccessible()
        self.arena_root = Path(arena_root).resolve()
        if not self.arena_root.is_dir():
            raise UpstreamControllerError("arena_root must be an existing directory")
        config_path = self.workspace / "config.yaml"
        if not config_path.is_file():
            raise UpstreamControllerError("Arena workspace lacks config.yaml")
        self.log_path = self.workspace / ARTIFACT_DIRNAME / "arena-evaluator.log"
        self.log_path.parent.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"autokernel.upstream.{os.getpid()}")
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.addHandler(logging.FileHandler(self.log_path, encoding="utf-8"))
        self.source_paths = self._admit_source_paths(source_paths)
        self._starting = {
            path: (self.workspace / path).read_bytes() for path in self.source_paths}
        broker_socket = os.environ.get(BROKER_SOCKET_ENV)
        broker_token = os.environ.get(BROKER_TOKEN_ENV)
        broker_owner = os.environ.get(BROKER_OWNER_PID_ENV)
        if not broker_socket or not broker_token or not broker_owner:
            raise UpstreamControllerError(
                "Arena evaluator lacks its governed GPU evaluation broker")
        self.broker_socket = Path(broker_socket).resolve()
        if not self.broker_socket.is_socket():
            raise UpstreamControllerError(
                "Arena evaluator broker socket is absent or unsafe")
        self._broker_token = broker_token
        try:
            self._broker_owner_pid = int(broker_owner)
        except ValueError as exc:
            raise UpstreamControllerError(
                "Arena evaluator broker owner identity is invalid") from exc
        self.best_files = dict(self._starting)
        self.best_score = 1.0
        self.last_record: EvaluationRecord | None = None
        self.evaluation_count = 0
        self.broker_receipts: list[dict[str, Any]] = []
        inherited_fd = os.environ.get(sandbox.BROKER_FD_ENV)
        self._broker_stream: socket.socket | None = None
        if inherited_fd is not None:
            try:
                descriptor = int(inherited_fd)
                self._broker_stream = socket.socket(fileno=os.dup(descriptor))
            except (OSError, ValueError) as exc:
                raise UpstreamControllerError(
                    "Arena evaluator inherited broker descriptor is invalid") from exc
            finally:
                os.environ.pop(sandbox.BROKER_FD_ENV, None)
        # Upstream controllers may generate candidates concurrently, but the
        # Arena evaluator owns one mutable copied workspace and one physical
        # GPU. Serialize the materialize/evaluate/restore transaction while
        # leaving proposal generation under the upstream controller's policy.
        self._evaluation_lock = threading.Lock()

    def _admit_source_paths(self, declared: Sequence[str]) -> tuple[str, ...]:
        if isinstance(declared, (str, bytes)) or not declared:
            raise UpstreamControllerError(
                "Arena source paths must be a non-empty parent declaration")
        rows = [value.strip() for value in declared if isinstance(value, str)]
        if len(rows) != len(declared) or not all(rows) or len(set(rows)) != len(rows):
            raise UpstreamControllerError(
                "Arena source paths contain an invalid or duplicate declaration")
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
        with self._evaluation_lock:
            return self._evaluate_serialized(files)

    def _evaluate_serialized(self, files: Mapping[str, str]) -> EvaluationRecord:
        if set(files) != set(self.source_paths):
            raise UpstreamControllerError(
                f"candidate files must be exactly {list(self.source_paths)}")
        candidate = {
            path: text.encode("utf-8") for path, text in files.items()
            if isinstance(text, str) and text.strip()}
        if set(candidate) != set(self.source_paths):
            raise UpstreamControllerError("candidate source content must be non-empty")
        self.evaluation_count += 1
        raw = self._brokered_evaluation(self.evaluation_count, candidate)
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

    def _brokered_evaluation(
        self, ordinal: int, candidate: Mapping[str, bytes],
    ) -> Mapping[str, Any]:
        """Ask the parent worker to own materialization, claim, and evaluation."""
        root = self.workspace / ARTIFACT_DIRNAME / "brokered-evaluations"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / f"{ordinal:04d}-result.json"
        request: dict[str, Any] = {
            "schema": BROKER_REQUEST_SCHEMA,
            "token": self._broker_token,
            "owner_pid": self._broker_owner_pid,
            "workspace": str(self.workspace),
            "evaluation_ordinal": ordinal,
            "source_files": {
                path: content.decode("utf-8") for path, content in candidate.items()},
        }
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > _MAX_BROKER_MESSAGE_BYTES:
            raise UpstreamControllerError("Arena broker request exceeds its size limit")
        try:
            with _MODEL_BROKER_IO_LOCK:
                broker_stream = getattr(self, "_broker_stream", None)
                if broker_stream is None:
                    client_context = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    client_context.connect(str(self.broker_socket))
                    close_after = True
                else:
                    client_context = broker_stream
                    close_after = False
                client = client_context
                try:
                    server_pid, server_uid, _ = struct.unpack(
                        "3i", client.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if server_pid != self._broker_owner_pid \
                            or server_uid != os.getuid():
                        raise UpstreamControllerError(
                            "Arena broker server identity is invalid")
                    _write_all(client, struct.pack("!Q", len(encoded)) + encoded)
                    header = _recv_exact(client, 8)
                    length = struct.unpack("!Q", header)[0]
                    if length > _MAX_BROKER_MESSAGE_BYTES:
                        raise UpstreamControllerError(
                            "Arena broker response exceeds its size limit")
                    receipt = json.loads(_recv_exact(client, length))
                finally:
                    if close_after:
                        client.close()
        except (OSError, json.JSONDecodeError, struct.error) as exc:
            raise UpstreamControllerError(
                "Arena evaluation broker IPC failed") from exc
        if isinstance(receipt, Mapping) and receipt.get("status") == "error":
            raise UpstreamControllerError(
                "Arena evaluation broker failed: " + str(receipt.get("error")))
        claimed = receipt.get("receipt_sha256") if isinstance(receipt, dict) else None
        without_hash = ({key: value for key, value in receipt.items()
                         if key != "receipt_sha256"}
                        if isinstance(receipt, dict) else {})
        if (not isinstance(receipt, dict)
                or receipt.get("schema") != BROKER_RESULT_SCHEMA
                or claimed != _canonical_sha256(without_hash)
                or receipt.get("evaluation_ordinal") != ordinal
                or receipt.get("workspace") != str(self.workspace)
                or receipt.get("source_sha256") != {
                    path: hashlib.sha256(content).hexdigest()
                    for path, content in candidate.items()}
                or not isinstance(receipt.get("evaluation"), Mapping)):
            raise UpstreamControllerError(
                "Arena evaluation broker receipt identity is invalid")
        self.broker_receipts.append({
            "evaluation_ordinal": ordinal,
            "receipt_sha256": claimed,
            "path": str(output_path.relative_to(self.workspace)),
        })
        _atomic_json(output_path, receipt)
        return dict(receipt["evaluation"])

    def materialize_best(self) -> None:
        self._materialize(self.best_files)

    def receipt_fields(self) -> dict[str, Any]:
        return {
            "arena_root": str(self.arena_root),
            "source_paths": list(self.source_paths),
            "evaluation_count": self.evaluation_count,
            "brokered_evaluation_count": len(self.broker_receipts),
            "broker_receipts": list(self.broker_receipts),
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
    "ARTIFACT_DIRNAME", "ARENA_SOURCE_PATHS_ENV", "BROKER_SOCKET_ENV", "BROKER_TOKEN_ENV",
    "BROKER_OWNER_PID_ENV", "BROKER_REQUEST_SCHEMA", "BROKER_RESULT_SCHEMA",
    "MODEL_BROKER_REQUEST_SCHEMA", "MODEL_BROKER_RESULT_SCHEMA",
    "MODEL_EFFORT", "MODEL_ID", "PINNED_MODEL_IDS",
    "REQUIRED_CLIS", "ArenaWorkspaceEvaluator", "CodexTextModel",
    "ModelBrokerClient",
    "ControllerBudget", "ControllerBudgetExpired", "EvaluationRecord",
    "UpstreamControllerError", "build_controller_receipt", "workspace_root",
]
