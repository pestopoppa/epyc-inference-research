#!/usr/bin/env python3
"""Governed AgentKernelArena execution bridge for the INF-03 campaign.

The campaign driver deliberately stops at a typed ``run_cell`` seam.  This
module supplies the concrete implementation without patching either pinned
vendor checkout.  Each non-baseline campaign cell becomes three independent
2 h / 8 h / 32 h runs from the same hash-bound task.  The MI210 is claimed only
for the two centralized vendor measurement windows.  It is deliberately free
during remote controller/model deliberation.

Importing this module performs no model, compiler, evaluator, or GPU work.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from . import (arena_adapter, arena_campaign, arena_controller_sandbox,
               arena_evaluator_child,
               arena_roundtrip, arena_upstream_common)
from ..execution import device_sampler, sandbox
from ..resource import device_claim


RUNNER_SCHEMA = "epyc.autokernel.arena_cell_runner.v3"
CHECKPOINT_SCHEMA = "epyc.autokernel.arena_checkpoint.v2"
AGGREGATE_SCHEMA = "epyc.autokernel.arena_campaign_execution.v2"
RUN_MANIFEST_SCHEMA = "epyc.autokernel.arena_campaign_run_manifest.v2"
LEGACY_RUN_MANIFEST_SCHEMA = "epyc.autokernel.arena_campaign_run_manifest.v1"
VALIDATION_SCHEMA = "epyc.autokernel.arena_campaign_validation.v1"
MEASUREMENT_WINDOW_SCHEMA = "epyc.autokernel.arena_gpu_measurement_window.v1"
IMPLEMENTATION_MODULE = Path(__file__).resolve()
REPOSITORY_ROOT = IMPLEMENTATION_MODULE.parents[4]
DEFAULT_CLAIM_JOURNAL = "/mnt/raid0/llm/ak-claims/device.jsonl"
DEFAULT_DEVICE_ID = "mi210_0"
EVALUATION_RESERVE_SECONDS = 7200
CONTROLLER_ACTIVATION_RECEIPT = "controller-sandbox-activation.json"
CONTROLLER_TEARDOWN_RECEIPT = "controller-sandbox-teardown.json"
EVALUATOR_PYTHON = Path(
    "/mnt/raid0/llm/tools/geak-v1-rocm62-py312/bin/python")
EVALUATOR_PYTHON_SHA256 = (
    "9544d2a29138833e6177d45dbc57468d37710b5080c901fbb579d53f251cdd6f")
EVALUATOR_PACKAGE_VERSIONS = {
    "pytest": "9.1.1",
    "torch": "2.5.1+rocm6.2",
    "triton": "3.1.0",
}
_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,95}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ArenaCellRunnerError(RuntimeError):
    """A campaign cell cannot be executed with its declared evidence bounds."""


class ArenaCampaignInterrupted(ArenaCellRunnerError):
    """The campaign received a graceful termination request."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArenaCellRunnerError(
            f"{label} is absent or not a regular non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaCellRunnerError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ArenaCellRunnerError(f"{label} must be a JSON object")
    return payload


def _verify_self_hash(payload: Mapping[str, Any], label: str) -> None:
    claimed = payload.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
        raise ArenaCellRunnerError(f"{label} lacks its internal SHA-256")
    without_hash = {key: value for key, value in payload.items()
                    if key != "receipt_sha256"}
    if _canonical_sha256(without_hash) != claimed:
        raise ArenaCellRunnerError(f"{label} internal SHA-256 does not verify")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@lru_cache(maxsize=1)
def _evaluator_python_identity() -> dict[str, Any]:
    """Verify the exact ROCm evaluator interpreter without launching a GPU kernel."""
    if not EVALUATOR_PYTHON.is_file() or not os.access(EVALUATOR_PYTHON, os.X_OK):
        raise ArenaCellRunnerError("pinned Arena evaluator Python is unavailable")
    observed_sha256 = _sha256_file(EVALUATOR_PYTHON)
    if observed_sha256 != EVALUATOR_PYTHON_SHA256:
        raise ArenaCellRunnerError(
            "pinned Arena evaluator Python binary identity drifted")
    probe = (
        "import json,pytest,torch,triton; "
        "print(json.dumps({'pytest':pytest.__version__,"
        "'torch':torch.__version__,'triton':triton.__version__},sort_keys=True))"
    )
    environment = dict(os.environ)
    environment.update({
        "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
        "CUDA_VISIBLE_DEVICES": "",
    })
    result = subprocess.run(
        (str(EVALUATOR_PYTHON), "-c", probe), capture_output=True, text=True,
        check=False, timeout=30, env=environment)
    try:
        packages = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ArenaCellRunnerError(
            "Arena evaluator Python package probe emitted invalid JSON") from exc
    if result.returncode != 0 or packages != EVALUATOR_PACKAGE_VERSIONS:
        raise ArenaCellRunnerError(
            f"Arena evaluator package identity drifted: {packages!r}")
    return {
        "path": str(EVALUATOR_PYTHON),
        "resolved_path": str(EVALUATOR_PYTHON.resolve()),
        "sha256": observed_sha256,
        "packages": packages,
    }


def _declared_evaluator_python_identity() -> dict[str, Any]:
    """Return the pinned evaluator identity without importing its packages."""
    return {
        "path": str(EVALUATOR_PYTHON),
        "resolved_path": str(EVALUATOR_PYTHON.resolve()),
        "sha256": EVALUATOR_PYTHON_SHA256,
        "packages": dict(EVALUATOR_PACKAGE_VERSIONS),
    }


def _assert_worker_evaluator_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    expected = _evaluator_python_identity()
    if request.get("evaluator_python") != expected:
        raise ArenaCellRunnerError("worker request evaluator Python identity drifted")
    if Path(sys.executable).resolve() != EVALUATOR_PYTHON.resolve():
        raise ArenaCellRunnerError(
            "Arena worker must run under the pinned ROCm evaluator Python")
    return expected


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ArenaCellRunnerError(f"{label} is not a safe campaign identifier")
    return value


def _path_id(value: str, label: str) -> str:
    """Return a dot-free, collision-resistant component for one governed ID.

    AgentKernelArena's pinned ``test_add_kernel.py`` derives a cache name with
    ``__file__.replace('.', '_')``.  A dotted parent component therefore moves
    its write into a sibling directory.  Human-readable normalization alone is
    not injective, so retain a short digest of the native ID as the collision
    witness.
    """
    native = _safe_id(value, label)
    readable = re.sub(r"[^a-z0-9_-]", "_", native)
    digest = hashlib.sha256(native.encode("utf-8")).hexdigest()[:12]
    component = f"{readable}-{digest}"
    if "." in component or not re.fullmatch(r"[a-z0-9_-]+", component):
        raise ArenaCellRunnerError(f"{label} did not normalize to a safe path ID")
    return component


def _assert_dot_safe_directory_path(path: Path, label: str) -> None:
    """Refuse a worker root whose directory components change under dot rewrite."""
    dotted = [part for part in path.resolve().parts if "." in part]
    if dotted:
        raise ArenaCellRunnerError(
            f"{label} has dot-bearing directory components unsafe for the pinned "
            f"Arena filename transform: {dotted}")


def _assert_contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArenaCellRunnerError(f"{label} escapes its governed root") from exc
    return resolved


def _outside_cell_manifest(cell_root: Path) -> dict[str, str]:
    """Hash every governed sibling object while excluding the active cell."""
    cells_root = cell_root.parent.resolve()
    active = cell_root.resolve()
    if active.parent != cells_root:
        raise ArenaCellRunnerError("cell_root is not a direct child of cells root")
    rows: dict[str, str] = {}
    for path in sorted(cells_root.rglob("*")):
        try:
            path.resolve().relative_to(active)
        except ValueError:
            pass
        else:
            continue
        relative = path.relative_to(cells_root).as_posix()
        if path.is_symlink():
            rows[relative] = f"symlink:{os.readlink(path)}"
        elif path.is_file():
            rows[relative] = f"file:{_sha256_file(path)}"
        elif path.is_dir():
            rows[relative] = "directory"
        else:
            rows[relative] = "special"
    return rows


def _assert_worker_tree_contained(cell_root: Path) -> None:
    """Prove the worker left only ordinary objects inside its exact cell root."""
    resolved_root = cell_root.resolve()
    workspace = resolved_root / "workspace"
    if workspace.exists() and (workspace.is_symlink() or not workspace.is_dir()):
        raise ArenaCellRunnerError("Arena workspace is not an exact regular directory")
    for path in resolved_root.rglob("*"):
        if path.is_symlink():
            raise ArenaCellRunnerError(
                f"Arena worker left a symlink in its cell: {path.relative_to(resolved_root)}")
        _assert_contained(path, resolved_root, "worker artifact")
        if not path.is_file() and not path.is_dir():
            raise ArenaCellRunnerError(
                f"Arena worker left a special file in its cell: "
                f"{path.relative_to(resolved_root)}")


@contextmanager
def _graceful_campaign_signals():
    """Turn TERM/INT into an exception so claim and worker finally blocks run."""
    watched = (signal.SIGTERM, signal.SIGINT)
    previous = {sig: signal.getsignal(sig) for sig in watched}
    received: list[int] = []

    def interrupt(signum: int, _frame: object) -> None:
        if received:
            return
        received.append(signum)
        # Protect the finite worker teardown and claim journal append from a
        # duplicate polite signal. SIGKILL remains the operator's hard stop.
        for watched_signal in watched:
            signal.signal(watched_signal, signal.SIG_IGN)
        raise ArenaCampaignInterrupted(
            f"campaign interrupted by {signal.Signals(signum).name}")

    try:
        for watched_signal in watched:
            signal.signal(watched_signal, interrupt)
        yield
    finally:
        for watched_signal, handler in previous.items():
            signal.signal(watched_signal, handler)


@contextmanager
def _defer_campaign_signals():
    """Defer polite termination across claim acquisition's return boundary."""
    watched = {signal.SIGTERM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        yield
    finally:
        # A pending signal is delivered here, after the caller has assigned the
        # returned claim handle, so its enclosing finally can journal release.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


@contextmanager
def _gpu_visibility(visible_device: str):
    """Expose the MI210 only for one already-claimed measurement call."""
    keys = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = visible_device
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_gpu_measurement_window(
    *, request: Mapping[str, Any], cell_root: Path, ordinal: int, phase: str,
    action: Callable[[], Any],
    window_path: Path | None = None,
    claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    sampler_factory: Callable[..., Any] = device_sampler.RocmSmiSampler,
) -> tuple[Any, dict[str, Any]]:
    """Run one vendor measurement under an exact, durable claim window.

    Acquisition, sampling, GPU visibility, action, release, and receipt
    publication are one signal-safe unit.  In particular, callers cannot carry
    this claim across controller/model deliberation.
    """
    if phase not in {
        "vendor_baseline", "centralized_final_evaluation",
        "controller_intermediate_evaluation",
    }:
        raise ArenaCellRunnerError(f"unsupported GPU measurement phase: {phase}")
    campaign_id = _safe_id(str(request.get("campaign_id")), "campaign_id")
    attempt_id_raw = request.get("attempt_id")
    attempt_id = (
        _safe_id(str(attempt_id_raw), "attempt_id")
        if attempt_id_raw is not None else None)
    claim_campaign_id = _safe_id(
        str(request.get("claim_campaign_id", campaign_id)),
        "claim_campaign_id")
    if attempt_id is not None and claim_campaign_id != attempt_id:
        raise ArenaCellRunnerError(
            "measurement claim scope does not match the campaign attempt")
    arm = request.get("arm")
    task = request.get("task")
    if not isinstance(arm, Mapping) or not isinstance(task, Mapping):
        raise ArenaCellRunnerError("measurement window lacks task or arm identity")
    arm_id = _safe_id(str(arm.get("arm_id")), "arm_id")
    task_id = _safe_id(str(task.get("task_id")), "task_id")
    visible_device = str(request.get("visible_device"))
    claim_journal = str(request.get("claim_journal"))
    claim_timeout = request.get("claim_timeout_seconds")
    if (isinstance(claim_timeout, bool)
            or not isinstance(claim_timeout, (int, float))
            or not math.isfinite(claim_timeout) or claim_timeout < 0):
        raise ArenaCellRunnerError("worker claim timeout is invalid")

    if window_path is None:
        window_root = cell_root / "measurement-windows"
        window_path = window_root / f"{ordinal:02d}-{phase}.json"
    else:
        window_path = _assert_contained(
            window_path, cell_root, "measurement window receipt")
    if window_path.exists():
        raise ArenaCellRunnerError(f"measurement window receipt already exists: {window_path}")
    started_at = _utc_now()
    claim = None
    sampler = None
    opened = None
    released = None
    sampling = None
    result: Any = None
    failure: BaseException | None = None
    failure_traceback = None
    try:
        with _defer_campaign_signals():
            claim = claim_acquirer(
                DEFAULT_DEVICE_ID,
                purpose=f"INF-03 {phase} {arm_id} {task_id}",
                campaign_id=claim_campaign_id,
                journal=device_claim.ClaimJournal(claim_journal),
                holder_label="arena_cell_runner.py:measurement-window",
                timeout_s=float(claim_timeout),
                max_hold_s=EVALUATION_RESERVE_SECONDS + 120.0,
            )
        opened = claim.receipt().to_dict()
        sampler = sampler_factory(
            device_index=int(visible_device), interval_s=0.250).start()
        with _gpu_visibility(visible_device):
            result = action()
    except BaseException as exc:
        failure = exc
        failure_traceback = exc.__traceback__
    try:
        if sampler is not None:
            sampling = sampler.stop().to_dict()
    except BaseException as exc:
        if failure is None:
            failure = exc
            failure_traceback = exc.__traceback__
    try:
        if claim is not None:
            released = claim.release().to_dict()
    except BaseException as exc:
        if failure is None:
            failure = exc
            failure_traceback = exc.__traceback__

    receipt = _self_hash({
        "schema": MEASUREMENT_WINDOW_SCHEMA,
        "campaign_id": campaign_id,
        **({"attempt_id": attempt_id} if attempt_id is not None else {}),
        "claim_campaign_id": claim_campaign_id,
        "task_id": task_id,
        "arm_id": arm_id,
        "checkpoint_hours": request.get("checkpoint_hours"),
        "phase": phase,
        "ordinal": ordinal,
        "status": "complete" if failure is None else "failed",
        "started_at": started_at,
        "ended_at": _utc_now(),
        "device_claim_open": opened,
        "device_claim_released": released,
        "device_sampling": sampling,
        "gpu_action_executed_only_while_claim_held": True,
        "failure": None if failure is None else {
            "type": type(failure).__name__, "message": str(failure)},
    })
    _atomic_json(window_path, receipt)
    if failure is not None:
        raise failure.with_traceback(failure_traceback)
    if opened is None or released is None or sampling is None:
        raise ArenaCellRunnerError("GPU measurement window lacks complete claim evidence")
    return result, receipt


def _load_preflight(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaCellRunnerError(f"cannot read preflight receipt: {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != arena_adapter.PREFLIGHT_SCHEMA:
        raise ArenaCellRunnerError("preflight receipt has the wrong schema")
    claimed = payload.get("receipt_sha256")
    without_hash = {key: value for key, value in payload.items()
                    if key != "receipt_sha256"}
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed):
        raise ArenaCellRunnerError("preflight receipt lacks its internal SHA-256")
    if _canonical_sha256(without_hash) != claimed:
        raise ArenaCellRunnerError("preflight receipt internal SHA-256 does not verify")
    hardware = payload.get("hardware")
    if not isinstance(hardware, Mapping) or hardware.get("target_gfx_arch") != "gfx90a":
        raise ArenaCellRunnerError("preflight does not bind the physical gfx90a target")
    return payload, _sha256_file(source)


def _live_process_group_members(process_group_id: int) -> tuple[int, ...]:
    """Return live Linux processes in one exact, already-captured group."""
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2:].split()
            state = fields[0]
            group = int(fields[2])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
        if group == process_group_id and state != "Z":
            members.append(int(entry.name))
    return tuple(sorted(members))


def _terminate_captured_process_group(
    process_group_id: int, *, grace_seconds: float = 5.0,
) -> None:
    """Terminate and verify one process group captured from ``Popen.pid``."""
    if process_group_id <= 1 or process_group_id == os.getpgrp():
        raise ArenaCellRunnerError("refusing an unsafe process-group target")
    members = _live_process_group_members(process_group_id)
    if not members:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process_group_id, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            members = _live_process_group_members(process_group_id)
            if not members:
                return
            time.sleep(0.05)
    members = _live_process_group_members(process_group_id)
    if members:
        raise ArenaCellRunnerError(
            f"Arena worker process group {process_group_id} survived teardown: "
            f"{list(members)}")


@dataclass(frozen=True)
class RunnerConfig:
    campaign_id: str
    arena_root: str
    preflight_path: str
    output_root: str
    attempt_id: str | None = None
    claim_journal: str = DEFAULT_CLAIM_JOURNAL
    claim_timeout_seconds: float = 0.0
    device_id: str = DEFAULT_DEVICE_ID
    visible_device: str = "0"
    expected_runner_sha256: str | None = None
    config_path: str | None = None
    expected_config_sha256: str | None = None
    expected_campaign_module_sha256: str | None = None
    geak_root: str | None = None
    expected_vendor_sources: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _safe_id(self.campaign_id, "campaign_id")
        if self.attempt_id is not None:
            _safe_id(self.attempt_id, "attempt_id")
        arena = Path(self.arena_root)
        preflight = Path(self.preflight_path)
        output = Path(self.output_root)
        if not arena.is_absolute() or not arena.is_dir():
            raise ArenaCellRunnerError("arena_root must be an existing absolute directory")
        if not preflight.is_absolute() or not preflight.is_file():
            raise ArenaCellRunnerError("preflight_path must be an existing absolute file")
        if not output.is_absolute():
            raise ArenaCellRunnerError("output_root must be absolute")
        if output.exists() and (not output.is_dir() or output.is_symlink()):
            raise ArenaCellRunnerError(
                "output_root must be absent or an existing non-symlink directory")
        if self.expected_runner_sha256 is not None:
            if (not isinstance(self.expected_runner_sha256, str)
                    or not _SHA256_RE.fullmatch(self.expected_runner_sha256)):
                raise ArenaCellRunnerError(
                    "expected_runner_sha256 must be a lowercase SHA-256")
        if (self.config_path is None) != (self.expected_config_sha256 is None):
            raise ArenaCellRunnerError(
                "config_path and expected_config_sha256 must be supplied together")
        if self.config_path is not None:
            config_path = Path(self.config_path)
            if not config_path.is_absolute() or not config_path.is_file():
                raise ArenaCellRunnerError("config_path must be an existing absolute file")
            _sha256_text = self.expected_config_sha256
            if (not isinstance(_sha256_text, str)
                    or not _SHA256_RE.fullmatch(_sha256_text)):
                raise ArenaCellRunnerError(
                    "expected_config_sha256 must be a lowercase SHA-256")
        if self.expected_campaign_module_sha256 is not None:
            if not _SHA256_RE.fullmatch(self.expected_campaign_module_sha256):
                raise ArenaCellRunnerError(
                    "expected_campaign_module_sha256 must be a lowercase SHA-256")
        if (self.geak_root is None) != (self.expected_vendor_sources is None):
            raise ArenaCellRunnerError(
                "geak_root and expected_vendor_sources must be supplied together")
        if self.geak_root is not None:
            geak = Path(self.geak_root)
            if not geak.is_absolute() or not geak.is_dir():
                raise ArenaCellRunnerError("geak_root must be an existing absolute directory")
            if not isinstance(self.expected_vendor_sources, Mapping):
                raise ArenaCellRunnerError("expected_vendor_sources must be an object")
        if (isinstance(self.claim_timeout_seconds, bool)
                or not isinstance(self.claim_timeout_seconds, (int, float))
                or not math.isfinite(self.claim_timeout_seconds)
                or self.claim_timeout_seconds < 0):
            raise ArenaCellRunnerError("claim_timeout_seconds must be finite and non-negative")
        if self.device_id != DEFAULT_DEVICE_ID or self.visible_device != "0":
            raise ArenaCellRunnerError("the INF-03 campaign is pinned to MI210 device zero")


WorkerRunner = Callable[[Mapping[str, Any], float], Mapping[str, Any]]


class GovernedArenaCellRunner:
    """Callable concrete implementation of ``arena_campaign.run_cell``."""

    def __init__(self, config: RunnerConfig, *, worker: WorkerRunner | None = None):
        if not isinstance(config, RunnerConfig):
            raise TypeError("config must be a RunnerConfig")
        self.config = config
        self.arena_root = Path(config.arena_root).resolve()
        self.output_root = Path(config.output_root).resolve()
        self.preflight, self.preflight_file_sha256 = _load_preflight(
            config.preflight_path)
        self.evaluator_python = _evaluator_python_identity()
        self.expected_runner_sha256 = (
            config.expected_runner_sha256 or _sha256_file(IMPLEMENTATION_MODULE))
        self.expected_campaign_module_sha256 = (
            config.expected_campaign_module_sha256
            or _sha256_file(arena_campaign.IMPLEMENTATION_MODULE))
        self._worker = worker or self._run_worker_subprocess
        self._ordinal = 0
        self._cell_ordinal = 0
        self.resumed_checkpoints = 0
        self.executed_checkpoints = 0
        self._assert_static_identities()

    @property
    def attempt_id(self) -> str | None:
        return self.config.attempt_id

    @property
    def claim_campaign_id(self) -> str:
        # Legacy direct callers and v1 manifests used the logical campaign id
        # as the claim scope.  Every v2 campaign supplies the run-directory
        # attempt id instead, so journals cannot conflate separate attempts.
        return self.config.attempt_id or self.config.campaign_id

    def __call__(self, request: arena_campaign.CampaignCellRequest) -> dict[str, Any]:
        if not isinstance(request, arena_campaign.CampaignCellRequest):
            raise TypeError("request must be a CampaignCellRequest")
        self._cell_ordinal += 1
        if request.is_starting_state_baseline:
            runs = [self._run_checkpoint(request, checkpoint_hours=None)]
        else:
            runs = [self._run_checkpoint(request, checkpoint_hours=hours)
                    for hours in request.checkpoint_hours]
        receipt = _self_hash({
            "schema": RUNNER_SCHEMA,
            "authority": "whole_agent_task_only",
            "campaign_id": self.config.campaign_id,
            **({"attempt_id": self.attempt_id}
               if self.attempt_id is not None else {}),
            "task_id": request.task.task_id,
            "arm_id": request.arm.arm_id,
            "baseline": request.is_starting_state_baseline,
            "checkpoint_hours": list(request.checkpoint_hours),
            "runs": runs,
            "constraints": {
                "independent_checkpoint_workspaces": True,
                "mi210_claimed_only_for_measurement_windows": True,
                "controller_deliberation_holds_no_gpu_claim": True,
                "dot_free_collision_bound_cell_paths": True,
                "post_worker_sibling_manifest_verified": True,
                "promotion_authority": False,
            },
        })
        receipt_path = (
            self.output_root / "cell-receipts" /
            f"{self._cell_ordinal:03d}-{request.task.task_id}-{request.arm.arm_id}.json"
        )
        if receipt_path.exists():
            observed = _load_json_object(receipt_path, "cell receipt")
            _verify_self_hash(observed, "cell receipt")
            if observed != receipt:
                raise ArenaCellRunnerError(
                    f"completed cell receipt drifted: {receipt_path}")
        else:
            _atomic_json(receipt_path, receipt)
        return receipt

    def _run_checkpoint(
        self, request: arena_campaign.CampaignCellRequest,
        *, checkpoint_hours: float | None,
    ) -> dict[str, Any]:
        self._assert_static_identities()
        task_audit = arena_campaign._task_audit(self.arena_root, request.task)
        if not task_audit["ready"]:
            raise ArenaCellRunnerError(
                f"task identity drifted after campaign audit: {task_audit['failures']}")
        arm_audit = arena_campaign._implementation_audit(request.arm)
        if not arm_audit["executable"]:
            raise ArenaCellRunnerError(
                f"arm identity drifted after campaign audit: "
                f"{arm_audit['missing_artifacts']}")
        self._ordinal += 1
        checkpoint_name = "baseline" if checkpoint_hours is None else f"{checkpoint_hours:g}h"
        cell_id = (
            f"{self._ordinal:03d}-{_path_id(request.task.task_id, 'task_id')}-"
            f"{_path_id(request.arm.arm_id, 'arm_id')}-"
            f"{checkpoint_name}"
        )
        cell_root = self.output_root / "cells" / cell_id
        _assert_dot_safe_directory_path(cell_root, "cell_root")
        if cell_root.exists():
            restored = self._restore_checkpoint(
                cell_root, request=request, checkpoint_hours=checkpoint_hours)
            if restored is not None:
                self.resumed_checkpoints += 1
                return restored
            self._abandon_partial_checkpoint(cell_root)
        cell_root.mkdir(parents=True)
        self.executed_checkpoints += 1
        started_at = _utc_now()
        worker_request = self._worker_request(
            request, checkpoint_hours=checkpoint_hours, cell_root=cell_root)
        _atomic_json(cell_root / "worker-request.json", worker_request)
        outside_before = _outside_cell_manifest(cell_root)
        max_runtime = ((checkpoint_hours or 0.0) * 3600
                       + EVALUATION_RESERVE_SECONDS)
        worker_result: Mapping[str, Any] | None = None
        try:
            worker_result = self._worker(worker_request, max_runtime)
        finally:
            _assert_worker_tree_contained(cell_root)
            outside_after = _outside_cell_manifest(cell_root)
            if outside_after != outside_before:
                added = sorted(outside_after.keys() - outside_before.keys())
                removed = sorted(outside_before.keys() - outside_after.keys())
                changed = sorted(
                    key for key in outside_before.keys() & outside_after.keys()
                    if outside_before[key] != outside_after[key])
                raise ArenaCellRunnerError(
                    "Arena worker wrote outside its exact cell root; "
                    f"added={added}, removed={removed}, changed={changed}")
        if worker_result is None:
            raise ArenaCellRunnerError("Arena worker completed without a durable result")
        self._assert_static_identities()
        completed_task_audit = arena_campaign._task_audit(self.arena_root, request.task)
        completed_arm_audit = arena_campaign._implementation_audit(request.arm)
        if not completed_task_audit["ready"] or not completed_arm_audit["executable"]:
            raise ArenaCellRunnerError(
                "task or controller source identity changed during checkpoint execution")
        if worker_result.get("schema") != CHECKPOINT_SCHEMA:
            raise ArenaCellRunnerError("Arena worker returned the wrong checkpoint schema")
        artifact_map = worker_result.get("artifacts")
        if not isinstance(artifact_map, Mapping) or not artifact_map:
            raise ArenaCellRunnerError("Arena worker returned no hash-bound artifacts")
        controller_sandbox_execution = worker_result.get(
            "controller_sandbox_execution")
        if request.is_starting_state_baseline:
            if controller_sandbox_execution is not None:
                raise ArenaCellRunnerError(
                    "starting-state baseline carries controller sandbox evidence")
        else:
            _validate_controller_sandbox_execution(
                controller_sandbox_execution, cell_root=cell_root,
                expected=worker_result)
        self._verify_measurement_windows(
            worker_result.get("measurement_windows"), cell_root=cell_root,
            expected=worker_result)
        belief_receipt = None
        if not request.is_starting_state_baseline:
            evaluation = worker_result.get("evaluation")
            if not isinstance(evaluation, Mapping):
                raise ArenaCellRunnerError("Arena worker omitted centralized evaluation")
            timing_total = 1
            timing_passed = int(int(evaluation.get("valid_optimized_cases", 0)) > 0)
            belief_receipt = arena_roundtrip.build_receipt(
                campaign_id=self.config.campaign_id,
                attempt_id=self.attempt_id,
                claim_campaign_id=(self.claim_campaign_id
                                   if self.attempt_id is not None else None),
                task_id=request.task.task_id,
                controller_id=request.arm.arm_id,
                started_at=started_at,
                ended_at=_utc_now(),
                correctness=arena_roundtrip.ScoredCount(
                    passed=int(bool(evaluation.get("pass_correctness"))), total=1,
                    reps_basis="one centralized AgentKernelArena task evaluation"),
                timing_validity=arena_roundtrip.ScoredCount(
                    passed=timing_passed, total=timing_total,
                    reps_basis=("one centralized AgentKernelArena timing phase; pass "
                                "requires at least one valid optimized timing case")),
                preflight_locator=str(Path(self.config.preflight_path).resolve()),
                preflight_sha256=self.preflight["receipt_sha256"],
                source={
                    "arm_source_root": request.arm.source_root,
                    "arm_source_commit": request.arm.source_commit,
                    "entrypoint_path": request.arm.entrypoint_path,
                    "entrypoint_sha256": request.arm.entrypoint_sha256,
                    "model_ids": list(request.arm.model_ids),
                    "checkpoint_hours": checkpoint_hours,
                },
                artifacts={str(key): str(value)
                           for key, value in artifact_map.items()},
            )
            _atomic_json(cell_root / "belief-receipt.json", belief_receipt)
        receipt = _self_hash({
            **dict(worker_result),
            "started_at": started_at,
            "ended_at": _utc_now(),
            "preflight": {
                "path": str(Path(self.config.preflight_path).resolve()),
                "file_sha256": self.preflight_file_sha256,
                "receipt_sha256": self.preflight["receipt_sha256"],
            },
            "runner": {
                "path": str(IMPLEMENTATION_MODULE),
                "sha256": self.expected_runner_sha256,
            },
            "write_containment": {
                "exact_cell_root": str(cell_root.resolve()),
                "dot_free_directory_path": True,
                "cell_tree_symlink_free": True,
                "sibling_manifest_unchanged": True,
            },
            "belief_receipt": belief_receipt,
        })
        _atomic_json(cell_root / "checkpoint-receipt.json", receipt)
        return receipt

    def _assert_static_identities(self) -> None:
        if _sha256_file(IMPLEMENTATION_MODULE) != self.expected_runner_sha256:
            raise ArenaCellRunnerError("Arena runner identity changed during execution")
        if (_sha256_file(arena_campaign.IMPLEMENTATION_MODULE)
                != self.expected_campaign_module_sha256):
            raise ArenaCellRunnerError(
                "Arena campaign module identity changed during execution")
        if _sha256_file(Path(self.config.preflight_path)) != self.preflight_file_sha256:
            raise ArenaCellRunnerError("Arena preflight identity changed during execution")
        if self.config.config_path is not None:
            if (_sha256_file(Path(self.config.config_path))
                    != self.config.expected_config_sha256):
                raise ArenaCellRunnerError(
                    "Arena campaign config identity changed during execution")
        if self.config.expected_vendor_sources is not None:
            observed_sources = {
                "agent_kernel_arena": arena_adapter.inspect_vendor_source(
                    self.arena_root, arena_adapter.AGENT_KERNEL_ARENA_PIN),
                "geak_v1": arena_adapter.inspect_vendor_source(
                    Path(str(self.config.geak_root)), arena_adapter.GEAK_V1_PIN),
            }
            if observed_sources != self.config.expected_vendor_sources:
                raise ArenaCellRunnerError(
                    "Arena vendor source identity changed during execution")

    def _restore_checkpoint(
        self, cell_root: Path, *, request: arena_campaign.CampaignCellRequest,
        checkpoint_hours: float | None,
    ) -> dict[str, Any] | None:
        """Return one exact complete checkpoint; never reuse partial evidence."""
        if not cell_root.is_dir() or cell_root.is_symlink():
            raise ArenaCellRunnerError(
                f"checkpoint output is not a non-symlink directory: {cell_root}")
        receipt_path = cell_root / "checkpoint-receipt.json"
        if not receipt_path.exists():
            return None
        receipt = _load_json_object(receipt_path, "checkpoint receipt")
        _verify_self_hash(receipt, "checkpoint receipt")
        expected_fields = {
            "schema": CHECKPOINT_SCHEMA,
            "campaign_id": self.config.campaign_id,
            "task_id": request.task.task_id,
            "arm_id": request.arm.arm_id,
            "baseline": request.is_starting_state_baseline,
            "checkpoint_hours": checkpoint_hours,
        }
        for field, expected in expected_fields.items():
            if receipt.get(field) != expected:
                raise ArenaCellRunnerError(
                    f"checkpoint receipt {field} drifted in {cell_root}")
        runner = receipt.get("runner")
        if (not isinstance(runner, Mapping)
                or runner.get("path") != str(IMPLEMENTATION_MODULE)
                or runner.get("sha256") != self.expected_runner_sha256):
            raise ArenaCellRunnerError("checkpoint runner identity drifted")
        expected_preflight = {
            "path": str(Path(self.config.preflight_path).resolve()),
            "file_sha256": self.preflight_file_sha256,
            "receipt_sha256": self.preflight["receipt_sha256"],
        }
        if receipt.get("preflight") != expected_preflight:
            raise ArenaCellRunnerError("checkpoint preflight identity drifted")
        self._verify_measurement_windows(
            receipt.get("measurement_windows"), cell_root=cell_root,
            expected=receipt)
        worker_request = _load_json_object(
            cell_root / "worker-request.json", "worker request")
        expected_request = self._worker_request(
            request, checkpoint_hours=checkpoint_hours, cell_root=cell_root)
        if _canonical_sha256(worker_request) != _canonical_sha256(expected_request):
            raise ArenaCellRunnerError("completed checkpoint request identity drifted")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ArenaCellRunnerError("completed checkpoint has no artifact manifest")
        for relative, expected_digest in artifacts.items():
            if not isinstance(relative, str) or not isinstance(expected_digest, str):
                raise ArenaCellRunnerError("checkpoint artifact manifest is malformed")
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ArenaCellRunnerError("checkpoint artifact path escapes its cell")
            artifact = cell_root / path
            if artifact.is_symlink() or not artifact.is_file():
                raise ArenaCellRunnerError(
                    f"checkpoint artifact is absent or unsafe: {relative}")
            if _sha256_file(artifact) != expected_digest:
                raise ArenaCellRunnerError(
                    f"checkpoint artifact digest drifted: {relative}")
        belief = receipt.get("belief_receipt")
        if request.is_starting_state_baseline:
            if belief is not None:
                raise ArenaCellRunnerError("baseline checkpoint carries a belief receipt")
            if receipt.get("controller_sandbox_execution") is not None:
                raise ArenaCellRunnerError(
                    "baseline checkpoint carries controller sandbox evidence")
        else:
            _validate_controller_sandbox_execution(
                receipt.get("controller_sandbox_execution"),
                cell_root=cell_root, expected=receipt)
            if not isinstance(belief, Mapping):
                raise ArenaCellRunnerError("controller checkpoint lacks its belief receipt")
            self._verify_belief_receipt(
                belief, checkpoint=receipt, request=request,
                artifacts=artifacts)
            persisted_belief = _load_json_object(
                cell_root / "belief-receipt.json", "persisted belief receipt")
            if persisted_belief != belief:
                raise ArenaCellRunnerError("persisted belief receipt drifted")
        return receipt

    def _verify_released_claim(
        self, receipt: Mapping[str, Any], *, expected_claim_campaign_id: str,
    ) -> None:
        """Verify the claim pair embedded in one measurement-window receipt."""
        opened = receipt.get("device_claim_open")
        released = receipt.get("device_claim_released")
        if not isinstance(opened, Mapping) or not isinstance(released, Mapping):
            raise ArenaCellRunnerError("checkpoint lacks device claim receipts")
        try:
            opened_receipt = device_claim.ClaimReceipt.from_dict(opened)
            released_receipt = device_claim.ClaimReceipt.from_dict(released)
        except (TypeError, ValueError) as exc:
            raise ArenaCellRunnerError(
                "checkpoint device claim receipt is malformed") from exc
        exact_fields = ("schema", "claim_id", "device_id", "campaign_id",
                        "acquired_at")
        if any(opened.get(field) != released.get(field) for field in exact_fields):
            raise ArenaCellRunnerError("opened and released device claims disagree")
        if (released_receipt.schema != device_claim.RECEIPT_SCHEMA
                or released_receipt.device_id != self.config.device_id
                or released_receipt.campaign_id != expected_claim_campaign_id
                or not isinstance(released_receipt.released_at, str)
                or not released_receipt.released_at
                or opened_receipt.released_at is not None):
            raise ArenaCellRunnerError("checkpoint device claim was not cleanly released")

    def _verify_measurement_windows(
        self, windows: object, *, cell_root: Path,
        expected: Mapping[str, Any],
    ) -> None:
        if not isinstance(windows, list) or len(windows) != 2:
            raise ArenaCellRunnerError(
                "checkpoint must carry exactly two GPU measurement windows")
        phases = ("vendor_baseline", "centralized_final_evaluation")
        expected_attempt = expected.get("attempt_id")
        expected_claim_scope = str(expected.get(
            "claim_campaign_id", expected.get("campaign_id")))
        claim_ids: list[str] = []
        for ordinal, (window, phase) in enumerate(zip(windows, phases), start=1):
            if not isinstance(window, Mapping):
                raise ArenaCellRunnerError("GPU measurement window is malformed")
            _verify_self_hash(window, "GPU measurement window")
            if any(window.get(key) != expected.get(key)
                   for key in ("campaign_id", "task_id", "arm_id")):
                raise ArenaCellRunnerError(
                    "GPU measurement window disagrees with its checkpoint identity")
            if expected_attempt is not None:
                if (window.get("attempt_id") != expected_attempt
                        or window.get("claim_campaign_id") != expected_claim_scope
                        or window.get("checkpoint_hours")
                        != expected.get("checkpoint_hours")):
                    raise ArenaCellRunnerError(
                        "GPU measurement window disagrees with its attempt or budget")
            elif (("attempt_id" in window and window.get("attempt_id") is not None)
                  or ("claim_campaign_id" in window
                      and window.get("claim_campaign_id") != expected_claim_scope)
                  or ("checkpoint_hours" in window
                      and window.get("checkpoint_hours")
                      != expected.get("checkpoint_hours"))):
                raise ArenaCellRunnerError(
                    "legacy GPU measurement window carries inconsistent identity")
            if (window.get("schema") != MEASUREMENT_WINDOW_SCHEMA
                    or window.get("phase") != phase
                    or window.get("ordinal") != ordinal
                    or window.get("status") != "complete"
                    or window.get("gpu_action_executed_only_while_claim_held") is not True
                    or not isinstance(window.get("device_sampling"), Mapping)):
                raise ArenaCellRunnerError(
                    "GPU measurement window identity or evidence is incomplete")
            sampling = window.get("device_sampling")
            if (isinstance(sampling.get("sample_count"), bool)
                    or not isinstance(sampling.get("sample_count"), int)
                    or sampling.get("sample_count") < 1):
                raise ArenaCellRunnerError(
                    "GPU measurement window has no numeric samples")
            self._verify_released_claim(
                window, expected_claim_campaign_id=expected_claim_scope)
            if phase == "centralized_final_evaluation" \
                    and expected.get("baseline") is False:
                _validate_evaluator_execution(
                    window.get("evaluator_execution_receipt"),
                    expected_workspace=cell_root / "final-evaluation-workspace",
                    expected_phase=phase, expected_identity=expected,
                    persisted_path=(cell_root / "final-evaluator-evidence"
                                    / "execution-receipt.json"),
                    expected_evaluation=expected["evaluation"],
                    expected_baseline_receipt_sha256=str(
                        windows[0]["receipt_sha256"]),
                    arena_root=self.arena_root)
            claim_ids.append(str(window["device_claim_open"]["claim_id"]))
            persisted = _load_json_object(
                cell_root / "measurement-windows" / f"{ordinal:02d}-{phase}.json",
                "persisted GPU measurement window")
            if persisted != window:
                raise ArenaCellRunnerError(
                    "persisted GPU measurement window receipt drifted")
        if len(set(claim_ids)) != 2:
            raise ArenaCellRunnerError(
                "baseline and final evaluation must use distinct device claims")

    def _verify_belief_receipt(
        self, belief: Mapping[str, Any], *, checkpoint: Mapping[str, Any],
        request: arena_campaign.CampaignCellRequest,
        artifacts: Mapping[str, Any],
    ) -> None:
        _verify_self_hash(belief, "belief receipt")
        if (belief.get("schema") != arena_roundtrip.SCHEMA
                or belief.get("producer_id") != arena_roundtrip.PRODUCER_ID
                or belief.get("campaign_id") != checkpoint.get("campaign_id")
                or belief.get("task") != {
                    "task_id": request.task.task_id,
                    "controller_id": request.arm.arm_id}
                or not isinstance(belief.get("source"), Mapping)
                or belief["source"].get("checkpoint_hours")
                != checkpoint.get("checkpoint_hours")):
            raise ArenaCellRunnerError(
                "belief receipt disagrees with its checkpoint identity")
        expected_attempt = checkpoint.get("attempt_id")
        if expected_attempt is not None:
            if (belief.get("attempt_id") != expected_attempt
                    or belief.get("claim_campaign_id")
                    != checkpoint.get("claim_campaign_id")):
                raise ArenaCellRunnerError(
                    "belief receipt disagrees with its campaign attempt")
        elif "attempt_id" in belief or "claim_campaign_id" in belief:
            raise ArenaCellRunnerError(
                "legacy belief receipt carries an unexpected attempt identity")
        expected_artifacts = [
            {"path": str(path), "sha256": str(digest)}
            for path, digest in sorted(artifacts.items())]
        if belief.get("artifacts") != expected_artifacts:
            raise ArenaCellRunnerError(
                "belief receipt artifact identity disagrees with its checkpoint")
        measurements = belief.get("belief_measurements")
        expected_ids = {
            "arena_correctness_pass_rate",
            "arena_timing_harness_validity_rate",
        }
        if (not isinstance(measurements, list) or len(measurements) != 2
                or {row.get("measurement_id") for row in measurements
                    if isinstance(row, Mapping)} != expected_ids
                or any(not isinstance(row, Mapping)
                       or not isinstance(row.get("extra"), Mapping)
                       or row["extra"].get("task_id") != request.task.task_id
                       or row["extra"].get("controller_id") != request.arm.arm_id
                       for row in measurements)):
            raise ArenaCellRunnerError(
                "belief receipt measurements disagree with their checkpoint")

    def _abandon_partial_checkpoint(self, cell_root: Path) -> None:
        abandoned = self.output_root / "abandoned"
        abandoned.mkdir(parents=True, exist_ok=True)
        suffix = 1
        while True:
            destination = abandoned / f"{cell_root.name}.attempt-{suffix:03d}"
            if not destination.exists():
                break
            suffix += 1
        os.replace(cell_root, destination)
        _fsync_directory(cell_root.parent)
        _fsync_directory(abandoned)

    def _worker_request(
        self, request: arena_campaign.CampaignCellRequest,
        *, checkpoint_hours: float | None, cell_root: Path,
    ) -> dict[str, Any]:
        arm_audit = arena_campaign._implementation_audit(request.arm)
        if not arm_audit["executable"]:
            raise ArenaCellRunnerError(
                "cannot construct worker request from a non-executable arm audit")
        return {
            "schema": CHECKPOINT_SCHEMA,
            "campaign_id": self.config.campaign_id,
            **({"attempt_id": self.attempt_id}
               if self.attempt_id is not None else {}),
            "claim_campaign_id": self.claim_campaign_id,
            "arena_root": str(self.arena_root),
            "repository_root": str(REPOSITORY_ROOT),
            "cell_root": str(cell_root),
            "task": asdict(request.task),
            "arm": asdict(request.arm),
            "arm_audit": arm_audit,
            "baseline": request.is_starting_state_baseline,
            "checkpoint_hours": checkpoint_hours,
            "visible_device": self.config.visible_device,
            "claim_journal": str(Path(self.config.claim_journal).resolve()),
            "claim_timeout_seconds": float(self.config.claim_timeout_seconds),
            "evaluator_python": self.evaluator_python,
        }

    @staticmethod
    def _run_worker_subprocess(
        request: Mapping[str, Any], timeout_seconds: float,
    ) -> Mapping[str, Any]:
        cell_root = Path(str(request["cell_root"]))
        output = cell_root / "worker-result.json"
        command = _worker_command(cell_root, output)
        env = arena_adapter.architecture_environment(os.environ)
        env.update({
            "HIP_VISIBLE_DEVICES": str(request["visible_device"]),
            "ROCR_VISIBLE_DEVICES": str(request["visible_device"]),
            "CUDA_VISIBLE_DEVICES": str(request["visible_device"]),
            "PYTHONPATH": str(request["repository_root"]),
        })
        process = subprocess.Popen(
            command, cwd=str(request["repository_root"]), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        timed_out = False
        stdout = ""
        stderr = ""
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            cleanup_error: Exception | None = None
            try:
                _terminate_captured_process_group(process.pid)
            except Exception as exc:  # preserve the teardown finding after reaping
                cleanup_error = exc
            if process.poll() is None:
                process.kill()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as exc:
                cleanup_error = ArenaCellRunnerError(
                    "Arena worker remained unreapable after group teardown")
                cleanup_error.__cause__ = exc
            if cleanup_error is not None:
                raise cleanup_error
        (cell_root / "worker.stdout").write_text(stdout, encoding="utf-8")
        (cell_root / "worker.stderr").write_text(stderr, encoding="utf-8")
        if timed_out:
            raise ArenaCellRunnerError(
                f"Arena worker exceeded its {timeout_seconds:g}s ceiling")
        if process.returncode != 0:
            raise ArenaCellRunnerError(
                f"Arena worker exited {process.returncode}: {stderr[-1000:]}")
        try:
            result = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArenaCellRunnerError("Arena worker did not emit valid JSON") from exc
        if not isinstance(result, dict):
            raise ArenaCellRunnerError("Arena worker result must be an object")
        return result


def _controller_argv(
    arm: Mapping[str, Any], checkpoint_hours: float,
    *, executable_path: str | None = None,
) -> tuple[str, ...]:
    raw = arm.get("argv")
    if not isinstance(raw, list) or not raw or any(not isinstance(x, str) for x in raw):
        raise ArenaCellRunnerError("controller arm lacks a valid argv")
    argv = list(raw)
    if executable_path is not None:
        executable = Path(executable_path)
        if (not executable.is_absolute() or executable.is_symlink()
                or not executable.is_file() or not os.access(executable, os.X_OK)):
            raise ArenaCellRunnerError(
                "audited controller executable is not an exact executable file")
        argv[0] = str(executable)
    for flag, value in (
        ("--checkpoint-hours", f"{checkpoint_hours:g}"),
        ("--timeout-seconds", str(int(checkpoint_hours * 3600))),
    ):
        try:
            index = argv.index(flag)
        except ValueError as exc:
            raise ArenaCellRunnerError(
                f"controller argv must expose the matched {flag} seam") from exc
        if index + 1 >= len(argv):
            raise ArenaCellRunnerError(f"controller argv has no value after {flag}")
        argv[index + 1] = value
    return tuple(argv)


def _worker_command(cell_root: Path, output: Path) -> tuple[str, ...]:
    return (
        str(EVALUATOR_PYTHON), "-m",
        "scripts.kernel_rnd.autokernel.controller.arena_cell_runner",
        "--worker-request", str(cell_root / "worker-request.json"),
        "--worker-output", str(output),
    )


def _copy_task(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ArenaCellRunnerError("Arena task root must be a non-symlink directory")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ArenaCellRunnerError(
                f"Arena task contains a symlink: {path.relative_to(source)}")
    shutil.copytree(source, destination)


def _declared_task_sources(config: Mapping[str, Any]) -> tuple[str, ...]:
    declared = config.get("source_file_path")
    rows = ([declared] if isinstance(declared, str)
            else declared if isinstance(declared, list) else [])
    if (not rows or any(not isinstance(row, str) or not row.strip()
                        for row in rows)):
        raise ArenaCellRunnerError(
            "brokered Arena tasks must declare source_file_path")
    paths = tuple(sorted(row.strip() for row in rows))
    if any(Path(row).is_absolute() or ".." in Path(row).parts for row in paths):
        raise ArenaCellRunnerError("brokered Arena source path is unsafe")
    return paths


@dataclass(frozen=True)
class EvaluatorChildResult:
    result: Mapping[str, Any]
    pid: int
    process_start_ticks: int
    process_group_id: int
    session_id: int
    activation_receipt: Mapping[str, Any]
    teardown_receipt: Mapping[str, Any]
    stdout_sha256: str
    stderr_sha256: str


class SandboxedEvaluatorRunner:
    """Run one Arena evaluation in a fresh deny-network GPU sandbox."""

    DEVICE_PATHS = ("/dev/kfd", "/dev/dri/renderD128")

    def __init__(self, *, arena_root: Path):
        self.arena_root = arena_root.resolve()

    @staticmethod
    def _environment(evaluation_root: Path, arena_root: Path) -> dict[str, str]:
        """Return the fixed startup environment admitted by the read policy."""
        return {
            "PATH": "/opt/rocm/bin:/usr/bin:/bin",
            "PYTHONPATH": str(arena_root),
            "HOME": str(evaluation_root), "TMPDIR": str(evaluation_root),
            "XDG_CACHE_HOME": str(evaluation_root / ".cache"),
            "TRITON_CACHE_DIR": str(evaluation_root / ".triton"),
            "TORCH_EXTENSIONS_DIR": str(evaluation_root / ".torch-extensions"),
            "HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
            # The evaluator read policy intentionally excludes /dev.  CPython
            # otherwise opens /dev/urandom during preinitialization before the
            # child can emit its activation-bound result.
            "PYTHONHASHSEED": "0",
        }

    @staticmethod
    def _readable_roots() -> tuple[str, ...]:
        candidates = (
            EVALUATOR_PYTHON.resolve().parents[1], EVALUATOR_PYTHON.parents[1],
            Path("/opt/rocm"), Path("/usr/bin"), Path("/usr/lib"),
            Path("/usr/share"), Path("/usr/include"),
            Path("/sys/devices/virtual/kfd/kfd/topology"),
            Path("/sys/devices/system/node"), Path("/sys/devices/system/cpu"),
            Path("/sys/class/drm/renderD128/device").resolve(),
        )
        return tuple(dict.fromkeys(str(path.resolve()) for path in candidates
                                   if path.exists()))

    def run(
        self, *, request: Mapping[str, Any], evaluation_root: Path,
        evidence_root: Path, timeout_s: float,
        cancel_event: threading.Event | None = None,
    ) -> EvaluatorChildResult:
        if evaluation_root.parent != evidence_root.parent:
            raise ArenaCellRunnerError("evaluator root/evidence ownership drifted")
        request_path = evaluation_root / "evaluator-request.json"
        _atomic_json(request_path, request)
        activation_path = evidence_root / "sandbox-activation.json"
        stdout_path = evidence_root / "evaluator.stdout"
        stderr_path = evidence_root / "evaluator.stderr"
        policy = sandbox.SandboxPolicy(
            writable_root=str(evaluation_root),
            writable_device_paths=self.DEVICE_PATHS,
            profile=sandbox.EVALUATOR_PROFILE,
            readable_roots=(*self._readable_roots(), str(self.arena_root / "src")),
            readable_files=("/etc/ld.so.cache",
                            str(arena_evaluator_child.__file__)),
            token=f"eval{secrets.token_hex(8)}")
        child_argv = (
            str(EVALUATOR_PYTHON), str(arena_evaluator_child.__file__),
            "--request", str(request_path))
        spawn_argv = policy.wrap(child_argv, receipt_path=str(activation_path))
        environment = self._environment(evaluation_root, self.arena_root)
        for path in (".cache", ".triton", ".torch-extensions"):
            (evaluation_root / path).mkdir()
        process: subprocess.Popen[str] | None = None
        timed_out = False
        cleanup_error: Exception | None = None
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, \
                stderr_path.open("w", encoding="utf-8") as stderr_handle:
            try:
                process = subprocess.Popen(
                    spawn_argv, cwd=evaluation_root, env=environment,
                    stdin=subprocess.DEVNULL, stdout=stdout_handle,
                    stderr=stderr_handle, text=True, close_fds=True,
                    start_new_session=True)
                pid = process.pid
                stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
                start_ticks = int(stat_text[stat_text.rfind(")") + 2:].split()[19])
                pgid, sid = os.getpgid(pid), os.getsid(pid)
                if pid != pgid or pid != sid:
                    raise ArenaCellRunnerError(
                        "evaluator child lacks an exact owned session")
                deadline = time.monotonic() + timeout_s
                while process.poll() is None:
                    if cancel_event is not None and cancel_event.wait(0.05):
                        timed_out = True
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(0.05)
            finally:
                if process is not None and (timed_out or process.poll() is None):
                    try:
                        _terminate_captured_process_group(process.pid)
                    except Exception as exc:
                        cleanup_error = exc
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
        if process is None:
            raise ArenaCellRunnerError("evaluator child did not start")
        activation: Mapping[str, Any] | None = None
        teardown: Mapping[str, Any] | None = None
        try:
            activation = sandbox.read_receipt(activation_path)
            sandbox.verify_receipt(
                activation, policy=policy, pid=process.pid, argv=child_argv)
            if activation.get("process_start_ticks") != start_ticks:
                raise ArenaCellRunnerError("evaluator child PID identity drifted")
        finally:
            if policy.cgroup_path(process.pid).exists():
                teardown = sandbox.cleanup_cgroup(policy, process.pid)
        if cleanup_error is not None:
            raise cleanup_error
        if timed_out:
            reason = "cancelled" if cancel_event is not None \
                and cancel_event.is_set() else "timed out"
            raise ArenaCellRunnerError(f"evaluator child {reason}")
        if process.returncode != 0:
            raise ArenaCellRunnerError(
                "evaluator child failed: " + stderr_path.read_text(
                    encoding="utf-8", errors="replace")[-1000:])
        try:
            output = json.loads(stdout_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArenaCellRunnerError("evaluator child emitted invalid JSON") from exc
        if not isinstance(output, Mapping):
            raise ArenaCellRunnerError("evaluator child result is not an object")
        arena_evaluator_child.verify_self_hash(output, "evaluator child result")
        if (output.get("schema") != arena_evaluator_child.RESULT_SCHEMA
                or output.get("request_receipt_sha256")
                != request.get("receipt_sha256")):
            raise ArenaCellRunnerError("evaluator child result identity drifted")
        assert activation is not None and teardown is not None
        return EvaluatorChildResult(
            result=output, pid=process.pid, process_start_ticks=start_ticks,
            process_group_id=pgid, session_id=sid,
            activation_receipt=activation, teardown_receipt=teardown,
            stdout_sha256=_sha256_file(stdout_path),
            stderr_sha256=_sha256_file(stderr_path))


def _run_sandboxed_arena_evaluation(
    *, evaluator_runner_factory: Callable[..., Any], arena_root: Path,
    evaluation_root: Path, evidence_root: Path,
    identity: Mapping[str, Any], evaluator_python: Mapping[str, Any],
    baseline_document: Mapping[str, Any], baseline_receipt_sha256: str,
    ordinal: int, timeout_s: float, cancel_event: threading.Event,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    runtime_root = evaluation_root.with_name(
        f"{ordinal:04d}-evaluator-runtime")
    runtime_root.mkdir(mode=0o700)
    shutil.copytree(arena_root / "src", runtime_root / "src")
    runner = evaluator_runner_factory(arena_root=runtime_root)
    request = _self_hash({
        "schema": arena_evaluator_child.REQUEST_SCHEMA,
        **dict(identity), "evaluation_ordinal": ordinal,
        "workspace": str(evaluation_root),
        "config_sha256": _sha256_file(evaluation_root / "config.yaml"),
        "arena_root": str(runtime_root),
        "vendor_evaluator_sha256": _sha256_file(
            runtime_root / "src" / "evaluator.py"),
        "evaluator_python": dict(evaluator_python),
        "baseline_cases": dict(baseline_document),
        "outer_baseline_receipt_sha256": baseline_receipt_sha256,
        "authority": "parent_claimed_sandboxed_evaluator_only",
    })
    try:
        child = runner.run(
            request=request, evaluation_root=evaluation_root,
            evidence_root=evidence_root, timeout_s=timeout_s,
            cancel_event=cancel_event)
    finally:
        shutil.rmtree(runtime_root)
    child_result = child.result
    if (child_result.get("baseline_cases_sha256")
            != baseline_document["receipt_sha256"]
            or child_result.get("outer_baseline_receipt_sha256")
            != baseline_receipt_sha256):
        raise ArenaCellRunnerError("evaluator child baseline identity drifted")
    _atomic_json(evidence_root / "evaluator-result.json", child_result)
    execution = _self_hash({
        "schema": "epyc.autokernel.arena_evaluator_execution.v1",
        "request_receipt_sha256": request["receipt_sha256"],
        "result_receipt_sha256": child_result["receipt_sha256"],
        "pid": child.pid, "process_start_ticks": child.process_start_ticks,
        "process_group_id": child.process_group_id,
        "session_id": child.session_id,
        "activation_receipt": dict(child.activation_receipt),
        "teardown_receipt": dict(child.teardown_receipt),
        "stdout_sha256": child.stdout_sha256,
        "stderr_sha256": child.stderr_sha256,
    })
    _atomic_json(evidence_root / "execution-receipt.json", execution)
    return dict(child_result["evaluation"]), execution


def _artifact_hashes(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows[path.relative_to(root).as_posix()] = _sha256_file(path)
    if not rows:
        raise ArenaCellRunnerError("checkpoint produced no artifacts")
    return rows


def _validate_evaluator_execution(
    execution: object, *, expected_workspace: Path, expected_phase: str,
    expected_identity: Mapping[str, Any], persisted_path: Path,
    expected_evaluation: Mapping[str, Any],
    expected_baseline_receipt_sha256: str, arena_root: Path,
) -> None:
    """Validate one candidate evaluator's process/sandbox evidence chain."""
    if not isinstance(execution, Mapping):
        raise ArenaCellRunnerError("candidate evaluation lacks sandbox evidence")
    _verify_self_hash(execution, "evaluator execution receipt")
    if execution.get("schema") != "epyc.autokernel.arena_evaluator_execution.v1":
        raise ArenaCellRunnerError("evaluator execution receipt schema drifted")
    pid = execution.get("pid")
    start_ticks = execution.get("process_start_ticks")
    if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1
            or isinstance(start_ticks, bool) or not isinstance(start_ticks, int)
            or start_ticks <= 0
            or execution.get("process_group_id") != pid
            or execution.get("session_id") != pid):
        raise ArenaCellRunnerError("evaluator process ownership is invalid")
    for field in ("request_receipt_sha256", "result_receipt_sha256",
                  "stdout_sha256", "stderr_sha256"):
        if not _SHA256_RE.fullmatch(str(execution.get(field))):
            raise ArenaCellRunnerError(f"evaluator execution {field} is invalid")
    activation = execution.get("activation_receipt")
    teardown = execution.get("teardown_receipt")
    if not isinstance(activation, Mapping) or not isinstance(teardown, Mapping):
        raise ArenaCellRunnerError("evaluator sandbox lifecycle evidence is absent")
    workspace = expected_workspace.resolve()
    required_syscalls = {
        "connect", "socket", "process_vm_readv", "process_vm_writev",
        "io_uring_setup", "io_uring_enter", "io_uring_register",
        "pidfd_getfd", "process_madvise",
    }
    readable_roots = activation.get("readable_roots")
    if (activation.get("profile") != sandbox.EVALUATOR_PROFILE
            or activation.get("pid") != pid
            or activation.get("process_start_ticks") != start_ticks
            or Path(str(activation.get("writable_root"))).resolve() != workspace
            or set(activation.get("writable_device_paths", ()))
            != set(SandboxedEvaluatorRunner.DEVICE_PATHS)
            or activation.get("read_allowlist_enforced") is not True
            or not isinstance(readable_roots, list)
            or any(Path(str(root)).resolve() in {Path("/"), Path("/proc")}
                   for root in readable_roots)
            or activation.get("network_profile") != sandbox.NETWORK_DENY_ALL
            or activation.get("outbound_socket_families") != []
            or activation.get("unix_socket_creation_denied") is not True
            or activation.get("broker_socket_path") is not None
            or activation.get("broker_fd_inherited") is not False
            or activation.get("broker_peer") is not None
            or not required_syscalls.issubset(set(
                activation.get("blocked_syscalls", ())))):
        raise ArenaCellRunnerError("evaluator sandbox activation is invalid")
    if (teardown.get("cgroup_path") != activation.get("cgroup_path")
            or teardown.get("verified_empty") is not True
            or teardown.get("removed") is not True):
        raise ArenaCellRunnerError("evaluator sandbox teardown is incomplete")
    request = _load_json_object(
        workspace / "evaluator-request.json", "evaluator child request")
    _verify_self_hash(request, "evaluator child request")
    baseline = request.get("baseline_cases")
    if not isinstance(baseline, Mapping):
        raise ArenaCellRunnerError("evaluator baseline serialization is absent")
    try:
        arena_evaluator_child.verify_self_hash(baseline, "baseline cases")
    except arena_evaluator_child.EvaluatorChildError as exc:
        raise ArenaCellRunnerError(
            "evaluator baseline serialization drifted") from exc
    vendor_path = arena_root.resolve() / "src" / "evaluator.py"
    if (request.get("schema") != arena_evaluator_child.REQUEST_SCHEMA
            or request.get("receipt_sha256")
            != execution.get("request_receipt_sha256")
            or request.get("workspace") != str(workspace)
            or request.get("phase") != expected_phase
            or request.get("config_sha256")
            != _sha256_file(workspace / "config.yaml")
            or request.get("vendor_evaluator_sha256")
            != _sha256_file(vendor_path)
            or request.get("evaluator_python")
            != _declared_evaluator_python_identity()
            or request.get("outer_baseline_receipt_sha256")
            != expected_baseline_receipt_sha256
            or any(request.get(key) != expected_identity.get(key) for key in (
                "campaign_id", "attempt_id", "claim_campaign_id", "task_id",
                "arm_id", "checkpoint_hours"))):
        raise ArenaCellRunnerError("evaluator child request identity drifted")
    evidence_root = persisted_path.parent
    stdout_path = evidence_root / "evaluator.stdout"
    stderr_path = evidence_root / "evaluator.stderr"
    if (_sha256_file(stdout_path) != execution.get("stdout_sha256")
            or _sha256_file(stderr_path) != execution.get("stderr_sha256")):
        raise ArenaCellRunnerError("evaluator output identity drifted")
    result = _load_json_object(
        evidence_root / "evaluator-result.json", "evaluator child result")
    try:
        arena_evaluator_child.verify_self_hash(result, "evaluator child result")
    except arena_evaluator_child.EvaluatorChildError as exc:
        raise ArenaCellRunnerError("evaluator child result drifted") from exc
    try:
        stdout_result = json.loads(stdout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaCellRunnerError("evaluator stdout is not its JSON result") from exc
    if (result != stdout_result
            or result.get("schema") != arena_evaluator_child.RESULT_SCHEMA
            or result.get("receipt_sha256")
            != execution.get("result_receipt_sha256")
            or result.get("request_receipt_sha256")
            != request.get("receipt_sha256")
            or result.get("baseline_cases_sha256")
            != baseline.get("receipt_sha256")
            or result.get("outer_baseline_receipt_sha256")
            != expected_baseline_receipt_sha256
            or result.get("evaluation") != expected_evaluation):
        raise ArenaCellRunnerError("evaluator result/evaluation identity drifted")
    persisted = _load_json_object(persisted_path, "evaluator execution receipt")
    if persisted != execution:
        raise ArenaCellRunnerError("persisted evaluator execution receipt drifted")


def _controller_runtime_allowlist(
    *, request: Mapping[str, Any], arm: Mapping[str, Any], workspace: Path,
    cell_root: Path, arena_root: Path, repository_root: Path,
) -> arena_controller_sandbox.RuntimeAllowlist:
    """Reify the parent-audited arm into one exact controller runtime."""
    audit = request.get("arm_audit")
    if not isinstance(audit, Mapping) or audit.get("arm_id") != arm.get("arm_id"):
        raise ArenaCellRunnerError("worker request lacks its exact arm audit")
    if audit.get("executable") is not True:
        raise ArenaCellRunnerError("worker arm audit is not executable")
    executable_raw = audit.get("executable_path")
    if not isinstance(executable_raw, str):
        raise ArenaCellRunnerError("worker arm audit lacks an executable path")
    executable = Path(executable_raw)
    if (not executable.is_absolute() or executable.is_symlink()
            or not executable.is_file()
            or _sha256_file(executable) != audit.get("executable_sha256")):
        raise ArenaCellRunnerError("audited controller executable identity drifted")
    source = audit.get("source_identity")
    if not isinstance(source, Mapping) or source.get("clean") is not True:
        raise ArenaCellRunnerError("audited controller source is absent or dirty")
    source_root = Path(str(source.get("root"))).resolve()
    entrypoint_relative = source.get("entrypoint_path")
    if not isinstance(entrypoint_relative, str):
        raise ArenaCellRunnerError("audited controller entrypoint is absent")
    entrypoint = (source_root / entrypoint_relative).resolve()
    scripts_root = (repository_root / "scripts").resolve()
    try:
        entrypoint.relative_to(scripts_root)
    except ValueError as exc:
        raise ArenaCellRunnerError(
            "in-tree controller entrypoint escaped the scripts source root") from exc
    if (_sha256_file(entrypoint) != source.get("observed_entrypoint_sha256")
            or source.get("observed_entrypoint_sha256")
            != source.get("expected_entrypoint_sha256")):
        raise ArenaCellRunnerError("audited controller entrypoint identity drifted")
    cli_rows = audit.get("required_cli_identities")
    if not isinstance(cli_rows, list):
        raise ArenaCellRunnerError("worker arm audit lacks CLI identities")
    cli: dict[str, Path] = {}
    for row in cli_rows:
        if (not isinstance(row, Mapping) or row.get("available") is not True
                or not isinstance(row.get("name"), str)
                or not isinstance(row.get("path"), str)):
            raise ArenaCellRunnerError("audited controller CLI is unavailable")
        path = Path(str(row["path"]))
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or _sha256_file(path) != row.get("sha256")):
            raise ArenaCellRunnerError("audited controller CLI identity drifted")
        cli[str(row["name"])] = path
    if "codex" not in cli:
        raise ArenaCellRunnerError("controller runtime lacks its audited Codex CLI")
    node_raw = shutil.which("node")
    if node_raw is None:
        raise ArenaCellRunnerError("controller runtime lacks Node")
    node = Path(node_raw).resolve(strict=True)
    codex_auth = Path("/home/node/.codex/auth.json")
    codex_config = Path("/home/node/.codex/config.toml")
    ca_file = Path("/etc/ssl/certs/ca-certificates.crt")
    exact_read_files = [codex_config]
    extra_clis: list[Path] = []
    if "claude" in cli:
        extra_clis.append(cli["claude"])
        exact_read_files.extend((
            Path("/home/node/.claude/.credentials.json"),
            Path("/home/node/.claude/.claude.json"),
        ))
    upstream = audit.get("upstream_source_identity")
    source_roots = [scripts_root, arena_root]
    if upstream is not None:
        if not isinstance(upstream, Mapping) or upstream.get("clean") is not True:
            raise ArenaCellRunnerError("audited upstream controller source is invalid")
        upstream_root = Path(str(upstream.get("root"))).resolve()
        source_roots.append(upstream_root)
        files = upstream.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ArenaCellRunnerError(
                "audited upstream controller file identities are absent")
        for row in files.values():
            if (not isinstance(row, Mapping)
                    or not isinstance(row.get("path"), str)
                    or not isinstance(row.get("observed_sha256"), str)
                    or row.get("observed_sha256") != row.get("expected_sha256")):
                raise ArenaCellRunnerError(
                    "audited upstream controller file identity is invalid")
            path = (upstream_root / str(row["path"])).resolve()
            if _sha256_file(path) != row["observed_sha256"]:
                raise ArenaCellRunnerError(
                    "audited upstream controller source identity drifted")
    return arena_controller_sandbox.discover_runtime_allowlist(
        workspace=workspace, python_executable=executable,
        controller_source_roots=tuple(source_roots),
        controller_entrypoint=entrypoint, repository_module_roots=(),
        codex_cli=cli["codex"], node_executable=node,
        codex_auth=codex_auth, ca_files=(ca_file,),
        additional_cli_executables=tuple(extra_clis),
        additional_cli_read_files=tuple(exact_read_files),
        forbidden_roots=(cell_root.parent.parent,),
    )


def _controller_process_start_ticks(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        ticks = int(text[text.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError) as exc:
        raise ArenaCellRunnerError(
            f"cannot bind controller broker process identity: {exc}") from exc
    if ticks <= 0:
        raise ArenaCellRunnerError("controller broker process start time is invalid")
    return ticks


def _controller_sandbox_execution(
    *, invocation: arena_controller_sandbox.ControllerSandboxInvocation,
    cell_root: Path,
) -> dict[str, Any]:
    teardown = invocation.verify_and_teardown(
        cell_root / CONTROLLER_TEARDOWN_RECEIPT)
    activation = sandbox.read_receipt(cell_root / CONTROLLER_ACTIVATION_RECEIPT)
    runtime = {
        "readable_roots": list(invocation.runtime.readable_roots),
        "readable_files": list(invocation.runtime.readable_files),
        "executable_files": list(invocation.runtime.executable_files),
        "identities": dict(invocation.runtime.identities),
        "sha256": invocation.runtime.sha256,
    }
    execution = _self_hash({
        "schema": "epyc.autokernel.arena_controller_sandbox_execution.v1",
        "pid": invocation.pid,
        "policy_sha256": invocation.policy.policy_sha256,
        "runtime_allowlist": runtime,
        "activation_receipt": activation,
        "teardown_receipt": teardown,
    })
    _atomic_json(cell_root / "controller-sandbox-execution.json", execution)
    return execution


def _launch_isolated_controller(
    *, prepared: arena_adapter.PreparedArenaTask, argv: Sequence[str],
    timeout_seconds: int, broker: "_ControllerEvaluationBroker",
    invocation: arena_controller_sandbox.ControllerSandboxInvocation,
    cell_root: Path,
) -> tuple[str, dict[str, Any]]:
    """Launch one controller and always empty/remove its exact cgroup."""
    output: str | None = None
    launch_error: BaseException | None = None
    try:
        def started(pid: int) -> None:
            invocation.process_started(pid)
            broker.register_controller(pid)

        output = arena_adapter.launch(
            prepared, argv, timeout_seconds=timeout_seconds,
            command_prefix=invocation.command_prefix,
            process_started=started)
    except BaseException as exc:
        launch_error = exc
    cleanup_error: BaseException | None = None
    execution: dict[str, Any] | None = None
    if invocation.pid is not None:
        try:
            execution = _controller_sandbox_execution(
                invocation=invocation, cell_root=cell_root)
        except BaseException as exc:
            cleanup_error = exc
    elif launch_error is None:
        cleanup_error = ArenaCellRunnerError(
            "controller launch returned without capturing its PID")
    if cleanup_error is not None:
        if launch_error is not None:
            raise ArenaCellRunnerError(
                "controller launch failed and sandbox teardown also failed: "
                f"launch={launch_error}; teardown={cleanup_error}") from cleanup_error
        raise cleanup_error
    if launch_error is not None:
        raise launch_error
    assert output is not None and execution is not None
    return output, execution


def _validate_controller_sandbox_execution(
    execution: object, *, cell_root: Path, expected: Mapping[str, Any],
) -> None:
    if not isinstance(execution, Mapping):
        raise ArenaCellRunnerError("controller checkpoint lacks sandbox evidence")
    _verify_self_hash(execution, "controller sandbox execution receipt")
    if execution.get("schema") != \
            "epyc.autokernel.arena_controller_sandbox_execution.v1":
        raise ArenaCellRunnerError("controller sandbox execution schema drifted")
    activation = execution.get("activation_receipt")
    teardown = execution.get("teardown_receipt")
    runtime = execution.get("runtime_allowlist")
    if not all(isinstance(row, Mapping) for row in (activation, teardown, runtime)):
        raise ArenaCellRunnerError("controller sandbox lifecycle evidence is malformed")
    assert isinstance(activation, Mapping)
    assert isinstance(teardown, Mapping)
    assert isinstance(runtime, Mapping)
    runtime_without_hash = {
        key: runtime.get(key) for key in (
            "readable_roots", "readable_files", "executable_files", "identities")}
    if _canonical_sha256(runtime_without_hash) != runtime.get("sha256"):
        raise ArenaCellRunnerError("controller runtime allowlist hash drifted")
    identities = runtime.get("identities")
    if not isinstance(identities, Mapping) or not identities:
        raise ArenaCellRunnerError("controller runtime identities are absent")
    for raw_path, expected_sha256 in identities.items():
        if (not isinstance(raw_path, str)
                or not _SHA256_RE.fullmatch(str(expected_sha256))):
            raise ArenaCellRunnerError("controller runtime identity is malformed")
        path = Path(raw_path)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or _sha256_file(path) != expected_sha256):
            raise ArenaCellRunnerError("controller runtime identity drifted")
    if (activation.get("profile") != sandbox.CONTROLLER_PROFILE
            or activation.get("writable_device_paths") != []
            or activation.get("read_allowlist_enforced") is not True
            or activation.get("broker_socket_path") is None
            or activation.get("broker_fd_inherited") is not True
            or not isinstance(activation.get("broker_peer"), Mapping)
            or activation.get("network_profile") != sandbox.NETWORK_OUTBOUND_CLIENT
            or Path(str(activation.get("writable_root"))).resolve()
            != (cell_root / "workspace").resolve()
            or activation.get("policy_sha256") != execution.get("policy_sha256")
            or teardown.get("policy_sha256") != execution.get("policy_sha256")
            or teardown.get("runtime_allowlist_sha256") != runtime.get("sha256")):
        raise ArenaCellRunnerError("controller sandbox activation is invalid")
    teardown_state = teardown.get("teardown")
    teardown_without_hash = {
        key: value for key, value in teardown.items() if key != "receipt_sha256"}
    activation_path = cell_root / CONTROLLER_ACTIVATION_RECEIPT
    teardown_path = cell_root / CONTROLLER_TEARDOWN_RECEIPT
    if (teardown.get("schema")
            != arena_controller_sandbox.TEARDOWN_SCHEMA
            or teardown.get("receipt_sha256")
            != _canonical_sha256(teardown_without_hash)
            or teardown.get("pid") != activation.get("pid")
            or teardown.get("process_start_ticks")
            != activation.get("process_start_ticks")
            or teardown.get("activation_receipt") != str(activation_path)
            or teardown.get("activation_receipt_sha256")
            != _sha256_file(activation_path)
            or not isinstance(teardown_state, Mapping)
            or teardown_state.get("cgroup_path") != activation.get("cgroup_path")
            or teardown_state.get("verified_empty") is not True
            or teardown_state.get("removed") is not True):
        raise ArenaCellRunnerError("controller sandbox teardown is incomplete")
    if runtime.get("readable_roots") != activation.get("readable_roots") \
            or runtime.get("readable_files") != activation.get("readable_files") \
            or runtime.get("executable_files") != activation.get("executable_files"):
        raise ArenaCellRunnerError("controller runtime and activation disagree")
    chain = expected.get("broker_evaluation_chain")
    if (not isinstance(chain, Mapping)
            or chain.get("controller_sandbox_execution_receipt_sha256")
            != execution.get("receipt_sha256")):
        raise ArenaCellRunnerError(
            "controller broker chain is not bound to sandbox execution")
    persisted = _load_json_object(
        cell_root / "controller-sandbox-execution.json",
        "controller sandbox execution receipt")
    if persisted != execution:
        raise ArenaCellRunnerError("persisted controller sandbox evidence drifted")
    if (sandbox.read_receipt(activation_path) != activation
            or _load_json_object(teardown_path, "controller teardown receipt") != teardown):
        raise ArenaCellRunnerError("controller sandbox lifecycle files drifted")


def _recv_exact(stream: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    while length:
        chunk = stream.recv(length)
        if not chunk:
            raise ArenaCellRunnerError("broker peer closed a partial message")
        chunks.append(chunk)
        length -= len(chunk)
    return b"".join(chunks)


class _ControllerEvaluationBroker:
    """Parent-worker-owned, serialized candidate evaluation service."""

    def __init__(
        self, *, request: Mapping[str, Any], workspace: Path, cell_root: Path,
        source_paths: Sequence[str],
        evaluate: Callable[[int, Path, threading.Event],
                           tuple[Mapping[str, Any], Mapping[str, Any]]],
        baseline_receipt_sha256: str,
    ):
        self.request, self.workspace, self.cell_root = request, workspace, cell_root
        self.template = cell_root / "controller-evaluation-template"
        _copy_task(workspace, self.template)
        self.source_paths = tuple(sorted(source_paths))
        self.evaluate = evaluate
        self.baseline_receipt_sha256 = baseline_receipt_sha256
        self.owner_pid = os.getpid()
        self.token = secrets.token_hex(32)
        runtime_parent = Path(os.environ.get(
            "AUTOKERNEL_BROKER_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        if not runtime_parent.is_dir():
            runtime_parent = Path("/tmp")
        self.runtime_dir = Path(tempfile.mkdtemp(
            prefix="akb-", dir=str(runtime_parent))).resolve()
        os.chmod(self.runtime_dir, 0o700)
        self.socket_path = self.runtime_dir / "broker.sock"
        if len(os.fsencode(self.socket_path)) >= 108:
            self.runtime_dir.rmdir()
            raise ArenaCellRunnerError("controller broker socket path is too long")
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._stop = threading.Event()
        self._ordinal = 0
        self._controller_pid: int | None = None
        self._controller_starttime: str | None = None
        self._controller_registered = threading.Event()
        self._previous_receipt_sha256: str | None = None
        self._thread: threading.Thread | None = None
        self._active_peer: socket.socket | None = None

    def __enter__(self) -> "_ControllerEvaluationBroker":
        if self.socket_path.exists():
            raise ArenaCellRunnerError("controller broker socket already exists")
        self._server.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self._server.listen(1)
        self._server.settimeout(0.2)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def environment(self) -> dict[str, str]:
        return {
            arena_upstream_common.BROKER_SOCKET_ENV: str(self.socket_path),
            arena_upstream_common.BROKER_TOKEN_ENV: self.token,
            arena_upstream_common.BROKER_OWNER_PID_ENV: str(self.owner_pid),
        }

    def register_controller(self, pid: int) -> None:
        if self._controller_pid is not None:
            raise ArenaCellRunnerError("controller broker PID was already registered")
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        self._controller_pid = pid
        self._controller_starttime = stat[stat.rfind(")") + 2:].split()[19]
        self._controller_registered.set()

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._controller_registered.set()
        if self._active_peer is not None:
            try:
                self._active_peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as wake:
                wake.connect(str(self.socket_path))
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                raise ArenaCellRunnerError("controller broker did not stop")
        self._server.close()
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.runtime_dir.rmdir()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                peer, _ = self._server.accept()
            except TimeoutError:
                continue
            with peer:
                self._active_peer = peer
                try:
                    self._handle_connection(peer)
                except Exception as exc:  # response is diagnostic, never authority
                    try:
                        self._send(peer, {"status": "error", "error": str(exc)})
                    except OSError:
                        pass
                finally:
                    self._active_peer = None

    @staticmethod
    def _send(peer: socket.socket, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        peer.sendall(struct.pack("!Q", len(encoded)) + encoded)

    def _handle_connection(self, peer: socket.socket) -> None:
        peer_pid, peer_uid, _ = struct.unpack(
            "3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if not self._controller_registered.wait(timeout=10) or self._stop.is_set():
            raise ArenaCellRunnerError("controller broker registration timed out")
        if (peer_uid != os.getuid() or self._controller_pid is None
                or peer_pid != self._controller_pid):
            raise ArenaCellRunnerError("controller broker rejected peer identity")
        stat = Path(f"/proc/{peer_pid}/stat").read_text(encoding="utf-8")
        if stat[stat.rfind(")") + 2:].split()[19] != self._controller_starttime:
            raise ArenaCellRunnerError("controller broker rejected PID reuse")
        while not self._stop.is_set():
            try:
                self._handle_frame(peer)
            except ArenaCellRunnerError as exc:
                if "partial message" in str(exc):
                    return
                raise

    def _handle_frame(self, peer: socket.socket) -> None:
        length = struct.unpack("!Q", _recv_exact(peer, 8))[0]
        if length > 16 * 1024 * 1024:
            raise ArenaCellRunnerError("controller broker request is too large")
        payload = json.loads(_recv_exact(peer, length))
        next_ordinal = self._ordinal + 1
        if (not isinstance(payload, dict)
                or payload.get("schema") != arena_upstream_common.BROKER_REQUEST_SCHEMA
                or not secrets.compare_digest(str(payload.get("token")), self.token)
                or payload.get("owner_pid") != self.owner_pid
                or payload.get("workspace") != str(self.workspace)
                or payload.get("evaluation_ordinal") != next_ordinal):
            raise ArenaCellRunnerError("controller broker request identity is invalid")
        self._ordinal = next_ordinal
        sources = payload.get("source_files")
        if not isinstance(sources, dict) or tuple(sorted(sources)) != self.source_paths:
            raise ArenaCellRunnerError("controller candidate source set is invalid")
        if sum(len(value.encode()) for value in sources.values()
               if isinstance(value, str)) > 8 * 1024 * 1024:
            raise ArenaCellRunnerError("controller candidate exceeds total byte limit")
        evaluation_root = (
            self.cell_root / "controller-evaluation-windows"
            / f"{self._ordinal:04d}-workspace")
        _copy_task(self.template, evaluation_root)
        hashes: dict[str, str] = {}
        for relative, text in sources.items():
            if (not isinstance(text, str) or not text.strip()
                    or len(text.encode()) > 2 * 1024 * 1024):
                raise ArenaCellRunnerError("controller candidate source is empty")
            target = _assert_contained(
                evaluation_root / relative, evaluation_root, "controller candidate")
            if target.is_symlink() or not target.is_file():
                raise ArenaCellRunnerError("controller candidate target is unsafe")
            target.write_text(text, encoding="utf-8")
            hashes[relative] = _sha256_file(target)
        cancel = threading.Event()
        outcome: list[Any] = []

        def invoke() -> None:
            try:
                outcome.append(self.evaluate(self._ordinal, evaluation_root, cancel))
            except BaseException as exc:
                outcome.append(exc)

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        disconnected = False
        while worker.is_alive():
            if self._stop.wait(0.05):
                cancel.set()
            try:
                if peer.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b"":
                    disconnected = True
                    cancel.set()
            except BlockingIOError:
                pass
            except OSError:
                disconnected = True
                cancel.set()
        worker.join()
        if not outcome:
            raise ArenaCellRunnerError("controller evaluation produced no outcome")
        if isinstance(outcome[0], BaseException):
            raise outcome[0]
        if disconnected:
            raise ArenaCellRunnerError("controller disconnected during evaluation")
        evaluation, window = outcome[0]
        receipt = _self_hash({
            "schema": arena_upstream_common.BROKER_RESULT_SCHEMA,
            "campaign_id": self.request["campaign_id"],
            **({"attempt_id": self.request["attempt_id"]}
               if self.request.get("attempt_id") is not None else {}),
            "claim_campaign_id": self.request.get(
                "claim_campaign_id", self.request["campaign_id"]),
            "task_id": self.request["task"]["task_id"],
            "arm_id": self.request["arm"]["arm_id"],
            "checkpoint_hours": self.request["checkpoint_hours"],
            "evaluation_ordinal": self._ordinal,
            "workspace": str(self.workspace), "evaluation_root": str(evaluation_root),
            "source_sha256": hashes,
            "baseline_receipt_sha256": self.baseline_receipt_sha256,
            "evaluation": dict(evaluation),
            "measurement_window": dict(window),
            "previous_receipt_sha256": self._previous_receipt_sha256,
            "authority": "controller_feedback_only",
        })
        _atomic_json(
            self.cell_root / "controller-evaluation-windows"
            / f"{self._ordinal:04d}-result.json", receipt)
        self._previous_receipt_sha256 = str(receipt["receipt_sha256"])
        self._send(peer, receipt)


def _run_worker_impl(
    request: Mapping[str, Any], *,
    claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    sampler_factory: Callable[..., Any] = device_sampler.RocmSmiSampler,
    evaluator_runner_factory: Callable[..., Any] = SandboxedEvaluatorRunner,
) -> dict[str, Any]:
    """Implement one checkpoint behind :func:`run_worker` cleanup."""
    if request.get("schema") != CHECKPOINT_SCHEMA:
        raise ArenaCellRunnerError("worker request has the wrong schema")
    evaluator_python = _assert_worker_evaluator_identity(request)
    campaign_id = _safe_id(str(request.get("campaign_id")), "campaign_id")
    attempt_id_raw = request.get("attempt_id")
    attempt_id = (
        _safe_id(str(attempt_id_raw), "attempt_id")
        if attempt_id_raw is not None else None)
    claim_campaign_id = _safe_id(
        str(request.get("claim_campaign_id", campaign_id)),
        "claim_campaign_id")
    if attempt_id is not None and claim_campaign_id != attempt_id:
        raise ArenaCellRunnerError(
            "worker claim scope does not match the campaign attempt")
    arena_root = Path(str(request.get("arena_root"))).resolve()
    repository_root = Path(str(request.get("repository_root"))).resolve()
    cell_root = Path(str(request.get("cell_root"))).resolve()
    _assert_contained(cell_root, cell_root.parent.parent, "cell_root")
    _assert_dot_safe_directory_path(cell_root, "cell_root")
    if repository_root != REPOSITORY_ROOT:
        raise ArenaCellRunnerError("worker repository identity drifted")
    task = request.get("task")
    arm = request.get("arm")
    if not isinstance(task, Mapping) or not isinstance(arm, Mapping):
        raise ArenaCellRunnerError("worker task and arm must be objects")
    task_id = _safe_id(str(task.get("task_id")), "task_id")
    arm_id = _safe_id(str(arm.get("arm_id")), "arm_id")
    relative_root = Path(str(task.get("relative_root")))
    if relative_root.is_absolute() or ".." in relative_root.parts:
        raise ArenaCellRunnerError("worker task path escapes Arena")
    task_root = _assert_contained(arena_root / relative_root, arena_root, "task root")
    workspace = cell_root / "workspace"
    _copy_task(task_root, workspace)
    config_path = workspace / "config.yaml"
    if not config_path.is_file():
        raise ArenaCellRunnerError("Arena task has no config.yaml")

    # The pinned vendor modules are loaded only inside this isolated worker.
    sys.path.insert(0, str(arena_root))
    try:
        import yaml  # type: ignore[import-not-found]
        from src import evaluator as vendor_evaluator  # type: ignore[import-not-found]
        from src import prompt_builder as vendor_prompt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ArenaCellRunnerError("cannot import pinned AgentKernelArena evaluator") from exc
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(task_config, dict):
        raise ArenaCellRunnerError("Arena task config must be an object")
    baseline = bool(request.get("baseline"))
    controller_runtime: arena_controller_sandbox.RuntimeAllowlist | None = None
    if not baseline:
        checkpoint = request.get("checkpoint_hours")
        if (isinstance(checkpoint, bool) or not isinstance(checkpoint, (int, float))
                or float(checkpoint) not in arena_campaign.MATCHED_BUDGET_HOURS):
            raise ArenaCellRunnerError("worker checkpoint is not a matched budget")
        # Resolve and hash the full controller runtime before compilation and
        # before the first GPU claim.  An incomplete isolation closure must not
        # leave a misleading partial baseline.
        controller_runtime = _controller_runtime_allowlist(
            request=request, arm=arm, workspace=workspace,
            cell_root=cell_root, arena_root=arena_root,
            repository_root=repository_root)
    elif request.get("checkpoint_hours") is not None:
        raise ArenaCellRunnerError(
            "starting-state baseline cannot have a checkpoint budget")

    log_path = cell_root / "arena.log"
    logger = logging.getLogger(f"autokernel.arena.{task_id}.{arm_id}")
    for existing_handler in logger.handlers[:]:
        existing_handler.close()
        logger.removeHandler(existing_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    logger.addHandler(file_handler)
    environment = arena_adapter.architecture_environment(os.environ)
    environment.update({
        "HIP_VISIBLE_DEVICES": str(request.get("visible_device")),
        "ROCR_VISIBLE_DEVICES": str(request.get("visible_device")),
        "CUDA_VISIBLE_DEVICES": str(request.get("visible_device")),
    })
    os.environ.update(environment)

    pass_compilation, compile_error = vendor_evaluator.evaluate_compilation(
        workspace, task_config, logger, None)
    if not pass_compilation:
        raise ArenaCellRunnerError(f"starting task does not compile: {compile_error}")
    baseline_cases, baseline_window = _run_gpu_measurement_window(
        request=request, cell_root=cell_root, ordinal=1,
        phase="vendor_baseline",
        action=lambda: vendor_evaluator.measure_baseline(
            workspace, task_config, logger, None),
        claim_acquirer=claim_acquirer, sampler_factory=sampler_factory)
    baseline_document = arena_evaluator_child.serialize_baseline_cases(
        baseline_cases)
    controller_stdout_sha256 = None
    broker_chain = None
    controller_sandbox_execution = None
    if not baseline:
        checkpoint = request.get("checkpoint_hours")
        raw_prompt = vendor_prompt.prompt_builder(
            str(config_path), workspace,
            {"target_gpu_model": arena_adapter.TARGET_GPU_MODEL}, logger)
        controller_environment = dict(environment)
        controller_environment.update({
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "CODEX_HOME": "/home/node/.codex",
            "CLAUDE_CONFIG_DIR": "/home/node/.claude",
        })
        source_paths = _declared_task_sources(task_config)

        def broker_evaluate(
            ordinal: int, evaluation_root: Path, cancel_event: threading.Event,
        ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
            evidence_root = evaluation_root.with_name(
                f"{ordinal:04d}-evaluator-evidence")
            evidence_root.mkdir(mode=0o700)
            identity = {
                "campaign_id": campaign_id,
                **({"attempt_id": attempt_id} if attempt_id is not None else {}),
                "claim_campaign_id": claim_campaign_id,
                "task_id": task_id, "arm_id": arm_id,
                "checkpoint_hours": checkpoint,
                "phase": "controller_intermediate_evaluation",
            }

            def action() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
                return _run_sandboxed_arena_evaluation(
                    evaluator_runner_factory=evaluator_runner_factory,
                    arena_root=arena_root, evaluation_root=evaluation_root,
                    evidence_root=evidence_root, identity=identity,
                    evaluator_python=evaluator_python,
                    baseline_document=baseline_document,
                    baseline_receipt_sha256=baseline_window["receipt_sha256"],
                    ordinal=ordinal,
                    timeout_s=min(3600.0, float(checkpoint) * 3600),
                    cancel_event=cancel_event)

            child_payload, window = _run_gpu_measurement_window(
                request=request, cell_root=cell_root, ordinal=ordinal,
                phase="controller_intermediate_evaluation", action=action,
                window_path=(cell_root / "controller-evaluation-windows"
                             / f"{ordinal:04d}-measurement.json"),
                claim_acquirer=claim_acquirer, sampler_factory=sampler_factory)
            evaluation, execution = child_payload
            window = _self_hash({
                **{key: value for key, value in window.items()
                   if key != "receipt_sha256"},
                "evaluator_execution_receipt": execution,
            })
            _atomic_json(
                cell_root / "controller-evaluation-windows"
                / f"{ordinal:04d}-measurement.json", window)
            return evaluation, window

        broker = _ControllerEvaluationBroker(
            request=request, workspace=workspace, cell_root=cell_root,
            source_paths=source_paths, evaluate=broker_evaluate,
            baseline_receipt_sha256=baseline_window["receipt_sha256"])
        prepared = arena_adapter.prepare_task(arena_adapter.ArenaTask(
            task_id=task_id,
            task_prompt=raw_prompt,
            workspace=str(workspace),
            controller_id=arm_id,
            round_id=f"{claim_campaign_id}-{checkpoint:g}h",
            actual_gfx_arch=arena_adapter.TARGET_GFX_ARCH,
        ), base_environment=controller_environment)
        # This is the deliberately unclaimed gap.  The controller receives a
        # GPU-blind environment and may spend its remote-model budget while
        # another host tenant uses the MI210.
        with broker:
            argv = _controller_argv(
                arm, float(checkpoint),
                executable_path=str(request["arm_audit"]["executable_path"]))
            assert controller_runtime is not None
            invocation = arena_controller_sandbox.prepare_controller_sandbox(
                workspace=workspace,
                receipt_path=cell_root / CONTROLLER_ACTIVATION_RECEIPT,
                expected_argv=argv, runtime=controller_runtime,
                broker_socket_path=broker.socket_path,
                broker_peer_pid=broker.owner_pid,
                broker_peer_start_ticks=_controller_process_start_ticks(
                    broker.owner_pid))
            controller_environment.update(invocation.environment_overrides)
            controller_environment.update(broker.environment())
            prepared = arena_adapter.prepare_task(
                prepared.task, base_environment=controller_environment)
            stdout, controller_sandbox_execution = _launch_isolated_controller(
                prepared=prepared, argv=argv,
                timeout_seconds=int(float(checkpoint) * 3600),
                broker=broker, invocation=invocation, cell_root=cell_root)
        controller_output = cell_root / "controller.stdout"
        controller_output.write_text(stdout, encoding="utf-8")
        controller_stdout_sha256 = _sha256_file(controller_output)
        try:
            controller_receipt = json.loads(stdout.strip().splitlines()[-1])
            best_hashes = controller_receipt["evaluation"]["best_source_sha256"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ArenaCellRunnerError(
                "controller did not identify its broker-evaluated selection") from exc
        selected = None
        for result_path in sorted((
                cell_root / "controller-evaluation-windows").glob("*-result.json")):
            candidate_receipt = _load_json_object(
                result_path, "broker evaluation receipt")
            _verify_self_hash(candidate_receipt, "broker evaluation receipt")
            if candidate_receipt.get("source_sha256") == best_hashes:
                selected = candidate_receipt
        if selected is None:
            raise ArenaCellRunnerError(
                "controller selection does not name broker-stored candidate bytes")
        selected_root = Path(str(selected["evaluation_root"])).resolve()
        for relative in source_paths:
            source = _assert_contained(
                selected_root / relative, selected_root, "selected candidate")
            target = _assert_contained(
                workspace / relative, workspace, "selected candidate target")
            shutil.copyfile(source, target)
        broker_chain = {
            "evaluation_count": broker._ordinal,
            "terminal_receipt_sha256": broker._previous_receipt_sha256,
            "selected_receipt_sha256": selected["receipt_sha256"],
            "source_paths": list(source_paths),
            "baseline_receipt_sha256": baseline_window["receipt_sha256"],
            "controller_sandbox_execution_receipt_sha256":
                controller_sandbox_execution["receipt_sha256"],
        }

    result_workspace = workspace
    if baseline:
        evaluation, evaluation_window = _run_gpu_measurement_window(
            request=request, cell_root=cell_root, ordinal=2,
            phase="centralized_final_evaluation",
            action=lambda: vendor_evaluator.evaluate_kernel(
                workspace, task_config, baseline_cases, logger, None),
            claim_acquirer=claim_acquirer, sampler_factory=sampler_factory)
    else:
        assert broker_chain is not None
        result_workspace = cell_root / "final-evaluation-workspace"
        _copy_task(broker.template, result_workspace)
        for relative in source_paths:
            source = _assert_contained(
                workspace / relative, workspace, "selected candidate")
            target = _assert_contained(
                result_workspace / relative, result_workspace,
                "selected candidate final target")
            shutil.copyfile(source, target)
        final_evidence_root = cell_root / "final-evaluator-evidence"
        final_evidence_root.mkdir(mode=0o700)
        final_cancel = threading.Event()
        final_ordinal = broker._ordinal + 1

        def final_action() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
            return _run_sandboxed_arena_evaluation(
                evaluator_runner_factory=evaluator_runner_factory,
                arena_root=arena_root, evaluation_root=result_workspace,
                evidence_root=final_evidence_root,
                identity={
                    "campaign_id": campaign_id,
                    **({"attempt_id": attempt_id}
                       if attempt_id is not None else {}),
                    "claim_campaign_id": claim_campaign_id,
                    "task_id": task_id, "arm_id": arm_id,
                    "checkpoint_hours": request.get("checkpoint_hours"),
                    "phase": "centralized_final_evaluation",
                },
                evaluator_python=evaluator_python,
                baseline_document=baseline_document,
                baseline_receipt_sha256=baseline_window["receipt_sha256"],
                ordinal=final_ordinal,
                timeout_s=EVALUATION_RESERVE_SECONDS,
                cancel_event=final_cancel)

        final_payload, evaluation_window = _run_gpu_measurement_window(
            request=request, cell_root=cell_root, ordinal=2,
            phase="centralized_final_evaluation", action=final_action,
            claim_acquirer=claim_acquirer, sampler_factory=sampler_factory)
        evaluation, execution = final_payload
        evaluation_window = _self_hash({
            **{key: value for key, value in evaluation_window.items()
               if key != "receipt_sha256"},
            "evaluator_execution_receipt": execution,
        })
        _atomic_json(
            cell_root / "measurement-windows"
            / "02-centralized_final_evaluation.json", evaluation_window)
    vendor_evaluator.write_task_result(
        result_workspace, evaluation, baseline_cases, task_id, arm_id, logger,
        create_plots=False)
    artifacts = _artifact_hashes(cell_root)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "authority": "whole_agent_task_only",
        "campaign_id": campaign_id,
        **({"attempt_id": attempt_id} if attempt_id is not None else {}),
        "claim_campaign_id": claim_campaign_id,
        "task_id": task_id,
        "arm_id": arm_id,
        "baseline": baseline,
        "checkpoint_hours": request.get("checkpoint_hours"),
        "evaluation": (
            dict(evaluation) if not baseline else {
                "pass_compilation": bool(evaluation.get("pass_compilation")),
                "pass_correctness": bool(evaluation.get("pass_correctness")),
                "valid_baseline_cases": int(
                    evaluation.get("valid_baseline_cases", 0)),
                "valid_optimized_cases": int(
                    evaluation.get("valid_optimized_cases", 0)),
                "average_speedup": float(evaluation.get("average_speedup", 0.0)),
            }),
        "controller_stdout_sha256": controller_stdout_sha256,
        "controller_sandbox_execution": controller_sandbox_execution,
        "broker_evaluation_chain": broker_chain,
        "measurement_windows": [baseline_window, evaluation_window],
        "artifacts": artifacts,
        "constraints": {
            "starting_state_copied_fresh": True,
            "centralized_vendor_evaluator": True,
            "evaluator_python": evaluator_python,
            "agent_reported_performance_admitted": False,
            "controller_deliberation_holds_no_gpu_claim": True,
            "controller_environment_gpu_blind": True,
            "promotion_authority": False,
        },
    }


def run_worker(
    request: Mapping[str, Any], *,
    claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    sampler_factory: Callable[..., Any] = device_sampler.RocmSmiSampler,
    evaluator_runner_factory: Callable[..., Any] = SandboxedEvaluatorRunner,
) -> dict[str, Any]:
    """Execute one checkpoint and close its invocation-owned log handler."""
    logger: logging.Logger | None = None
    try:
        task = request.get("task")
        arm = request.get("arm")
        if isinstance(task, Mapping) and isinstance(arm, Mapping):
            logger = logging.getLogger(
                f"autokernel.arena.{task.get('task_id')}.{arm.get('arm_id')}")
        return _run_worker_impl(
            request, claim_acquirer=claim_acquirer,
            sampler_factory=sampler_factory,
            evaluator_runner_factory=evaluator_runner_factory)
    finally:
        if logger is not None:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


def _run_manifest(
    args: argparse.Namespace, spec: arena_campaign.CampaignSpec,
    audit: Mapping[str, Any], *, available_source: bool,
) -> dict[str, Any]:
    preflight, preflight_file_sha256 = _load_preflight(args.preflight)
    selected_arms = (
        arena_campaign.AVAILABLE_SOURCE_PANEL_IDS
        if available_source else arena_campaign.PRIMARY_PANEL_IDS)
    output_root = Path(args.output_root).resolve()
    attempt_id = _safe_id(output_root.name, "attempt_id")
    return _self_hash({
        "schema": RUN_MANIFEST_SCHEMA,
        "campaign_id": audit["campaign_id"],
        "attempt_id": attempt_id,
        "attempt_root": str(output_root),
        "claim_campaign_id": attempt_id,
        "available_source": available_source,
        "authority": audit["authority"],
        "audit_receipt_sha256": audit["receipt_sha256"],
        "audit_schema": audit["schema"],
        "config": {
            "path": str(Path(spec.config_path).resolve()),
            "sha256": spec.config_sha256,
        },
        "preflight": {
            "path": str(Path(args.preflight).resolve()),
            "file_sha256": preflight_file_sha256,
            "receipt_sha256": preflight["receipt_sha256"],
        },
        "sources": {
            "arena_root": str(Path(args.arena_root).resolve()),
            "geak_root": str(Path(args.geak_root).resolve()),
            "audit_sources": audit["sources"],
            "controller_arms": audit["panel"]["arms"],
        },
        "runner": {
            "path": str(IMPLEMENTATION_MODULE),
            "sha256": _sha256_file(IMPLEMENTATION_MODULE),
        },
        "matrix": {
            "task_ids": [task.task_id for task in spec.tasks],
            "arm_ids": list(selected_arms),
            "checkpoint_hours": list(spec.budget_hours),
            "ordering": "task_then_declared_arm_then_checkpoint",
        },
        "constraints": {
            "resume_exact_complete_checkpoints_only": True,
            "partial_or_inflight_work_reused": False,
            "tampered_completed_work_reused": False,
            "dot_free_collision_bound_cell_paths": True,
            "post_worker_sibling_manifest_required": True,
            "sigterm_sigint_unwind_claims": True,
            "mi210_claimed_only_for_vendor_measurements": True,
            "controller_deliberation_holds_no_gpu_claim": True,
            "controller_environment_gpu_blind": True,
            "partial_results_rankable": False,
            "aggregate_atomic_after_complete_matrix_only": True,
            "promotion_authority": False,
        },
    })


def _prepare_campaign_root(
    output_root: Path, *, audit: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> bool:
    """Create or verify one immutable campaign root; return True on resume."""
    if not output_root.exists():
        output_root.mkdir(parents=True)
        arena_campaign.write_receipt(output_root / "audit.json", audit)
        if manifest is not None:
            _atomic_json(output_root / "campaign-manifest.json", manifest)
        return False
    if not output_root.is_dir() or output_root.is_symlink():
        raise ArenaCellRunnerError(
            "output_root must be an existing non-symlink campaign directory")
    stored_audit = _load_json_object(output_root / "audit.json", "stored audit")
    _verify_self_hash(stored_audit, "stored audit")
    if stored_audit != audit:
        raise ArenaCellRunnerError(
            "live audit/config/source identity differs from the stored campaign audit")
    if manifest is None:
        raise ArenaCellRunnerError("a refused campaign directory cannot be resumed")
    stored_manifest = _load_json_object(
        output_root / "campaign-manifest.json", "campaign manifest")
    _verify_self_hash(stored_manifest, "campaign manifest")
    if stored_manifest != manifest:
        raise ArenaCellRunnerError(
            "campaign manifest/config/source/runner identity drifted")
    return True


def _publish_or_verify_aggregate(
    path: Path, aggregate: Mapping[str, Any],
) -> None:
    if path.exists():
        observed = _load_json_object(path, "campaign aggregate")
        _verify_self_hash(observed, "campaign aggregate")
        if observed != aggregate:
            raise ArenaCellRunnerError("completed campaign aggregate drifted")
        return
    _atomic_json(path, aggregate)


def _validate_broker_chain(
    checkpoint: Mapping[str, Any], *, cell_root: Path, claim_scope: str,
    arena_root: Path,
) -> None:
    chain = checkpoint.get("broker_evaluation_chain")
    if not isinstance(chain, Mapping):
        raise ArenaCellRunnerError(
            "controller checkpoint lacks its broker evaluation chain")
    count = chain.get("evaluation_count")
    sources = chain.get("source_paths")
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or not isinstance(sources, list) or not sources):
        raise ArenaCellRunnerError("broker evaluation chain is malformed")
    result_paths = sorted((
        cell_root / "controller-evaluation-windows").glob("*-result.json"))
    window_paths = sorted((
        cell_root / "controller-evaluation-windows").glob("*-measurement.json"))
    if len(result_paths) != count or len(window_paths) != count:
        raise ArenaCellRunnerError("broker evaluation chain has orphaned evidence")
    previous = None
    hashes: set[str] = set()
    for ordinal, (result_path, window_path) in enumerate(
            zip(result_paths, window_paths), 1):
        result = _load_json_object(result_path, "broker evaluation receipt")
        _verify_self_hash(result, "broker evaluation receipt")
        if (result.get("schema") != arena_upstream_common.BROKER_RESULT_SCHEMA
                or result.get("evaluation_ordinal") != ordinal
                or result.get("previous_receipt_sha256") != previous
                or any(result.get(key) != checkpoint.get(key) for key in (
                    "campaign_id", "task_id", "arm_id", "checkpoint_hours"))
                or result.get("claim_campaign_id") != claim_scope
                or (checkpoint.get("attempt_id") is not None and
                    result.get("attempt_id") != checkpoint.get("attempt_id"))):
            raise ArenaCellRunnerError("broker evaluation semantic identity drifted")
        if (result.get("baseline_receipt_sha256")
                != chain.get("baseline_receipt_sha256")):
            raise ArenaCellRunnerError("broker baseline identity drifted")
        source_hashes = result.get("source_sha256")
        if (not isinstance(source_hashes, Mapping)
                or sorted(source_hashes) != sorted(sources)
                or any(not _SHA256_RE.fullmatch(str(value))
                       for value in source_hashes.values())):
            raise ArenaCellRunnerError("broker candidate source identity is invalid")
        window = result.get("measurement_window")
        if not isinstance(window, Mapping):
            raise ArenaCellRunnerError("broker evaluation lacks measurement evidence")
        _verify_self_hash(window, "broker measurement window")
        persisted = _load_json_object(window_path, "persisted broker measurement")
        if persisted != window:
            raise ArenaCellRunnerError("persisted broker measurement drifted")
        if (window.get("phase") != "controller_intermediate_evaluation"
                or window.get("ordinal") != ordinal
                or window.get("status") != "complete"
                or any(window.get(key) != checkpoint.get(key) for key in (
                    "campaign_id", "task_id", "arm_id", "checkpoint_hours"))
                or window.get("claim_campaign_id") != claim_scope):
            raise ArenaCellRunnerError("broker measurement semantic identity drifted")
        evaluation_root = Path(str(result.get("evaluation_root"))).resolve()
        expected_evaluation_root = (
            cell_root / "controller-evaluation-windows"
            / f"{ordinal:04d}-workspace").resolve()
        if evaluation_root != expected_evaluation_root:
            raise ArenaCellRunnerError("broker evaluation root identity drifted")
        _validate_evaluator_execution(
            window.get("evaluator_execution_receipt"),
            expected_workspace=evaluation_root,
            expected_phase="controller_intermediate_evaluation",
            expected_identity=checkpoint,
            persisted_path=evaluation_root.with_name(
                f"{ordinal:04d}-evaluator-evidence") / "execution-receipt.json",
            expected_evaluation=result["evaluation"],
            expected_baseline_receipt_sha256=str(
                chain["baseline_receipt_sha256"]), arena_root=arena_root)
        opened, released = window.get("device_claim_open"), window.get(
            "device_claim_released")
        if (not isinstance(opened, Mapping) or not isinstance(released, Mapping)
                or opened.get("claim_id") != released.get("claim_id")
                or opened.get("campaign_id") != claim_scope
                or released.get("campaign_id") != claim_scope
                or opened.get("released_at") is not None
                or not released.get("released_at")):
            raise ArenaCellRunnerError("broker measurement claim pair is invalid")
        previous = str(result["receipt_sha256"])
        if previous in hashes:
            raise ArenaCellRunnerError("broker evaluation receipt is duplicated")
        hashes.add(previous)
    if (chain.get("terminal_receipt_sha256") != previous
            or chain.get("selected_receipt_sha256") not in hashes
            or chain.get("baseline_receipt_sha256") != checkpoint.get(
                "measurement_windows", [{}])[0].get("receipt_sha256")):
        raise ArenaCellRunnerError("broker chain terminal or selection is invalid")


def validate_campaign_receipts(output_root: str | Path) -> dict[str, Any]:
    """Validate durable campaign evidence without probing hardware or resuming work."""
    root = Path(output_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ArenaCellRunnerError("validation root must be a campaign directory")
    audit = _load_json_object(root / "audit.json", "stored audit")
    _verify_self_hash(audit, "stored audit")
    manifest = _load_json_object(
        root / "campaign-manifest.json", "campaign manifest")
    _verify_self_hash(manifest, "campaign manifest")
    if manifest.get("schema") not in {
            RUN_MANIFEST_SCHEMA, LEGACY_RUN_MANIFEST_SCHEMA}:
        raise ArenaCellRunnerError("campaign manifest schema is unsupported")
    campaign_id = str(manifest.get("campaign_id"))
    if audit.get("campaign_id") != campaign_id:
        raise ArenaCellRunnerError("audit and manifest campaign identities disagree")
    attempt_id = manifest.get("attempt_id")
    claim_scope = str(manifest.get("claim_campaign_id", campaign_id))
    if manifest.get("schema") == RUN_MANIFEST_SCHEMA:
        if (attempt_id != root.name or claim_scope != attempt_id
                or manifest.get("attempt_root") != str(root)):
            raise ArenaCellRunnerError("campaign attempt identity is invalid")
    checkpoints: list[dict[str, Any]] = []
    cells_root = root / "execution" / "cells"
    for receipt_path in sorted(cells_root.glob("*/checkpoint-receipt.json")):
        receipt = _load_json_object(receipt_path, "checkpoint receipt")
        _verify_self_hash(receipt, "checkpoint receipt")
        if (receipt.get("schema") != CHECKPOINT_SCHEMA
                or receipt.get("campaign_id") != campaign_id):
            raise ArenaCellRunnerError("checkpoint campaign identity drifted")
        if attempt_id is not None and (
                receipt.get("attempt_id") != attempt_id
                or receipt.get("claim_campaign_id") != claim_scope):
            raise ArenaCellRunnerError("checkpoint attempt identity drifted")
        cell_root = receipt_path.parent
        windows = receipt.get("measurement_windows")
        if not isinstance(windows, list) or len(windows) != 2:
            raise ArenaCellRunnerError("checkpoint measurement windows are incomplete")
        for ordinal, (window, phase) in enumerate(zip(
                windows, ("vendor_baseline", "centralized_final_evaluation")), 1):
            if not isinstance(window, Mapping):
                raise ArenaCellRunnerError("checkpoint measurement window is malformed")
            _verify_self_hash(window, "GPU measurement window")
            if (window.get("phase") != phase or window.get("ordinal") != ordinal
                    or any(window.get(key) != receipt.get(key)
                           for key in ("campaign_id", "task_id", "arm_id"))
                    or ("checkpoint_hours" in window and window.get(
                        "checkpoint_hours") != receipt.get("checkpoint_hours"))
                    or window.get("status") != "complete"):
                raise ArenaCellRunnerError(
                    "GPU measurement window semantic identity drifted")
            if attempt_id is not None and (
                    window.get("attempt_id") != attempt_id
                    or window.get("claim_campaign_id") != claim_scope):
                raise ArenaCellRunnerError("GPU window attempt identity drifted")
            opened, released = window.get("device_claim_open"), window.get(
                "device_claim_released")
            if not isinstance(opened, Mapping) or not isinstance(released, Mapping):
                raise ArenaCellRunnerError("GPU window lacks claim receipts")
            if (opened.get("claim_id") != released.get("claim_id")
                    or opened.get("campaign_id") != claim_scope
                    or released.get("campaign_id") != claim_scope
                    or opened.get("released_at") is not None
                    or not released.get("released_at")):
                raise ArenaCellRunnerError("GPU window claim pair is invalid")
            persisted = _load_json_object(
                cell_root / "measurement-windows" / f"{ordinal:02d}-{phase}.json",
                "persisted GPU measurement window")
            if persisted != window:
                raise ArenaCellRunnerError("persisted GPU window drifted")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ArenaCellRunnerError("checkpoint artifact manifest is missing")
        for relative, digest in artifacts.items():
            path = Path(str(relative))
            artifact = cell_root / path
            if (path.is_absolute() or ".." in path.parts or artifact.is_symlink()
                    or not artifact.is_file() or _sha256_file(artifact) != digest):
                raise ArenaCellRunnerError("checkpoint artifact identity drifted")
        belief = receipt.get("belief_receipt")
        if receipt.get("baseline"):
            if belief is not None:
                raise ArenaCellRunnerError("baseline carries a belief receipt")
        else:
            sources = manifest.get("sources")
            if not isinstance(sources, Mapping) \
                    or not isinstance(sources.get("arena_root"), str):
                raise ArenaCellRunnerError(
                    "campaign manifest lacks pinned Arena source root")
            _validate_broker_chain(
                receipt, cell_root=cell_root, claim_scope=claim_scope,
                arena_root=Path(sources["arena_root"]))
            if not isinstance(belief, Mapping):
                raise ArenaCellRunnerError("controller checkpoint lacks belief evidence")
            _verify_self_hash(belief, "belief receipt")
            if (belief.get("campaign_id") != campaign_id
                    or belief.get("task") != {
                        "task_id": receipt.get("task_id"),
                        "controller_id": receipt.get("arm_id")}
                    or belief.get("source", {}).get("checkpoint_hours")
                    != receipt.get("checkpoint_hours")):
                raise ArenaCellRunnerError("belief receipt semantic identity drifted")
            if attempt_id is not None and (
                    belief.get("attempt_id") != attempt_id
                    or belief.get("claim_campaign_id") != claim_scope):
                raise ArenaCellRunnerError("belief attempt identity drifted")
            if _load_json_object(cell_root / "belief-receipt.json",
                                 "persisted belief receipt") != belief:
                raise ArenaCellRunnerError("persisted belief receipt drifted")
        checkpoints.append(receipt)
    cell_receipts = []
    checkpoint_hashes = {
        str(receipt["receipt_sha256"]): receipt for receipt in checkpoints}
    referenced_checkpoints: set[str] = set()
    for path in sorted((root / "execution" / "cell-receipts").glob("*.json")):
        receipt = _load_json_object(path, "cell receipt")
        _verify_self_hash(receipt, "cell receipt")
        if receipt.get("campaign_id") != campaign_id:
            raise ArenaCellRunnerError("cell campaign identity drifted")
        if attempt_id is not None and receipt.get("attempt_id") != attempt_id:
            raise ArenaCellRunnerError("cell attempt identity drifted")
        runs = receipt.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ArenaCellRunnerError("cell receipt has no checkpoint runs")
        for run in runs:
            digest = run.get("receipt_sha256") if isinstance(run, Mapping) else None
            durable = checkpoint_hashes.get(str(digest))
            if (durable is None or durable != run
                    or run.get("task_id") != receipt.get("task_id")
                    or run.get("arm_id") != receipt.get("arm_id")):
                raise ArenaCellRunnerError(
                    "cell receipt disagrees with its durable checkpoints")
            if str(digest) in referenced_checkpoints:
                raise ArenaCellRunnerError("checkpoint is referenced by multiple cells")
            referenced_checkpoints.add(str(digest))
        cell_receipts.append(receipt)
    aggregate_path = root / "execution-receipt.json"
    complete = aggregate_path.is_file()
    aggregate_hash = None
    if complete:
        aggregate = _load_json_object(aggregate_path, "campaign aggregate")
        _verify_self_hash(aggregate, "campaign aggregate")
        if aggregate.get("campaign_id") != campaign_id or (
                attempt_id is not None and aggregate.get("attempt_id") != attempt_id):
            raise ArenaCellRunnerError("aggregate campaign attempt drifted")
        if aggregate.get("cells") != cell_receipts:
            raise ArenaCellRunnerError("aggregate and durable cell receipts disagree")
        if referenced_checkpoints != set(checkpoint_hashes):
            raise ArenaCellRunnerError(
                "complete aggregate omits durable checkpoint evidence")
        aggregate_hash = aggregate["receipt_sha256"]
    return _self_hash({
        "schema": VALIDATION_SCHEMA, "status": "valid_complete" if complete
        else "valid_partial", "campaign_id": campaign_id,
        **({"attempt_id": attempt_id} if attempt_id is not None else {}),
        "validated_checkpoint_count": len(checkpoints),
        "validated_cell_count": len(cell_receipts),
        "aggregate_receipt_sha256": aggregate_hash,
        "hardware_or_controller_executed": False,
    })


def _execute_from_cli(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    spec = arena_campaign.load_spec(args.config)
    available_source = bool(getattr(args, "available_source", False))
    audit_function = (
        arena_campaign.audit_available_source_campaign
        if available_source else arena_campaign.audit_campaign)
    audit = audit_function(
        spec, arena_root=args.arena_root, geak_root=args.geak_root,
        enumerator=args.enumerator)
    output_root = Path(args.output_root).resolve()
    manifest = None
    if audit["status"] == "ready":
        manifest = _run_manifest(
            args, spec, audit, available_source=available_source)
    _prepare_campaign_root(output_root, audit=audit, manifest=manifest)
    if audit["status"] != "ready":
        return 3, {
            "schema": AGGREGATE_SCHEMA,
            "campaign_id": audit["campaign_id"],
            "status": "refused",
            "audit": str(output_root / "audit.json"),
            "controller_or_gpu_command_executed": False,
        }
    # RunnerConfig requires a not-yet-existing root so cell artifacts cannot mix
    # with the audit or any predecessor campaign.
    cells_root = output_root / "execution"
    runner = GovernedArenaCellRunner(RunnerConfig(
        campaign_id=audit["campaign_id"],
        attempt_id=manifest["attempt_id"],
        arena_root=str(Path(args.arena_root).resolve()),
        preflight_path=str(Path(args.preflight).resolve()),
        output_root=str(cells_root),
        claim_journal=args.claim_journal,
        claim_timeout_seconds=args.claim_timeout_seconds,
        expected_runner_sha256=manifest["runner"]["sha256"],
        config_path=str(Path(spec.config_path).resolve()),
        expected_config_sha256=spec.config_sha256,
        expected_campaign_module_sha256=(
            audit["execution_identity"]["implementation_module_sha256"]),
        geak_root=str(Path(args.geak_root).resolve()),
        expected_vendor_sources=audit["sources"],
    ))
    execute_function = (
        arena_campaign.execute_available_source_campaign
        if available_source else arena_campaign.execute_campaign)
    cells = execute_function(spec, audit, run_cell=runner)
    aggregate = _self_hash({
        "schema": AGGREGATE_SCHEMA,
        "campaign_id": audit["campaign_id"],
        "attempt_id": manifest["attempt_id"],
        "claim_campaign_id": manifest["claim_campaign_id"],
        "status": "complete",
        "authority": "whole_agent_task_only",
        "audit": str(output_root / "audit.json"),
        "audit_receipt_sha256": audit["receipt_sha256"],
        "campaign_manifest": str(output_root / "campaign-manifest.json"),
        "campaign_manifest_receipt_sha256": manifest["receipt_sha256"],
        "cells": cells,
        "constraints": {
            "partial_results_rankable": False,
            "availability_conditioned_only": available_source,
            "full_eight_arm_result_implied": False,
            "promotion_authority": False},
    })
    _publish_or_verify_aggregate(output_root / "execution-receipt.json", aggregate)
    return 0, aggregate


def execute_from_cli(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Execute a campaign with graceful TERM/INT claim cleanup."""
    with _graceful_campaign_signals():
        return _execute_from_cli(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-request")
    parser.add_argument("--worker-output")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--arena-root")
    parser.add_argument("--geak-root")
    parser.add_argument("--preflight")
    parser.add_argument("--output-root")
    parser.add_argument("--enumerator", default="/opt/rocm/bin/rocm_agent_enumerator")
    parser.add_argument("--claim-journal", default=DEFAULT_CLAIM_JOURNAL)
    parser.add_argument("--claim-timeout-seconds", type=float, default=0.0)
    parser.add_argument(
        "--available-source", action="store_true",
        help=("run the separately labelled six-arm available-source panel; "
              "never implies completion of the fixed eight-arm campaign"))
    args = parser.parse_args(argv)
    if args.validate_only:
        if not args.output_root:
            parser.error("--validate-only requires --output-root")
        disallowed = (args.worker_request, args.worker_output, args.config,
                      args.arena_root, args.geak_root, args.preflight)
        if any(value is not None for value in disallowed):
            parser.error("--validate-only cannot be combined with execution options")
        try:
            result = validate_campaign_receipts(args.output_root)
        except ArenaCellRunnerError as exc:
            print(json.dumps({
                "schema": VALIDATION_SCHEMA, "status": "invalid",
                "output_root": str(Path(args.output_root).resolve()),
                "error": str(exc), "hardware_or_controller_executed": False,
            }, sort_keys=True))
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.worker_request or args.worker_output:
        if not args.worker_request or not args.worker_output:
            parser.error("worker mode requires both --worker-request and --worker-output")
        request_path = Path(args.worker_request).resolve()
        output_path = Path(args.worker_output).resolve()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, Mapping):
            raise ArenaCellRunnerError("worker request must be a JSON object")
        declared_cell = Path(str(request.get("cell_root"))).resolve()
        if request_path.parent != declared_cell or output_path.parent != declared_cell:
            raise ArenaCellRunnerError(
                "worker request and output must stay in the declared cell root")
        with _graceful_campaign_signals():
            result = run_worker(request)
        _atomic_json(output_path, result)
        return 0
    required = ("config", "arena_root", "geak_root", "preflight", "output_root")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"campaign mode is missing: {', '.join(missing)}")
    status, receipt = execute_from_cli(args)
    print(json.dumps({
        "status": receipt["status"],
        "campaign_id": receipt["campaign_id"],
        "output_root": str(Path(args.output_root).resolve()),
    }, sort_keys=True))
    return status


__all__ = [
    "AGGREGATE_SCHEMA", "CHECKPOINT_SCHEMA", "MEASUREMENT_WINDOW_SCHEMA",
    "RUNNER_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "ArenaCampaignInterrupted", "ArenaCellRunnerError",
    "GovernedArenaCellRunner", "RunnerConfig",
    "execute_from_cli", "run_worker", "validate_campaign_receipts",
]


if __name__ == "__main__":
    raise SystemExit(main())
