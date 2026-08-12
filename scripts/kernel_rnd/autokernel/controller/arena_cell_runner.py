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
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from . import arena_adapter, arena_campaign, arena_roundtrip
from ..execution import device_sampler
from ..resource import device_claim


RUNNER_SCHEMA = "epyc.autokernel.arena_cell_runner.v3"
CHECKPOINT_SCHEMA = "epyc.autokernel.arena_checkpoint.v2"
AGGREGATE_SCHEMA = "epyc.autokernel.arena_campaign_execution.v2"
RUN_MANIFEST_SCHEMA = "epyc.autokernel.arena_campaign_run_manifest.v1"
MEASUREMENT_WINDOW_SCHEMA = "epyc.autokernel.arena_gpu_measurement_window.v1"
IMPLEMENTATION_MODULE = Path(__file__).resolve()
REPOSITORY_ROOT = IMPLEMENTATION_MODULE.parents[4]
DEFAULT_CLAIM_JOURNAL = "/mnt/raid0/llm/ak-claims/device.jsonl"
DEFAULT_DEVICE_ID = "mi210_0"
EVALUATION_RESERVE_SECONDS = 7200
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
    claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    sampler_factory: Callable[..., Any] = device_sampler.RocmSmiSampler,
) -> tuple[Any, dict[str, Any]]:
    """Run one vendor measurement under an exact, durable claim window.

    Acquisition, sampling, GPU visibility, action, release, and receipt
    publication are one signal-safe unit.  In particular, callers cannot carry
    this claim across controller/model deliberation.
    """
    if phase not in {"vendor_baseline", "centralized_final_evaluation"}:
        raise ArenaCellRunnerError(f"unsupported GPU measurement phase: {phase}")
    campaign_id = _safe_id(str(request.get("campaign_id")), "campaign_id")
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

    window_root = cell_root / "measurement-windows"
    window_path = window_root / f"{ordinal:02d}-{phase}.json"
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
                campaign_id=campaign_id,
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
        "task_id": task_id,
        "arm_id": arm_id,
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
        self._verify_measurement_windows(
            worker_result.get("measurement_windows"), cell_root=cell_root)
        belief_receipt = None
        if not request.is_starting_state_baseline:
            evaluation = worker_result.get("evaluation")
            if not isinstance(evaluation, Mapping):
                raise ArenaCellRunnerError("Arena worker omitted centralized evaluation")
            timing_total = 1
            timing_passed = int(int(evaluation.get("valid_optimized_cases", 0)) > 0)
            belief_receipt = arena_roundtrip.build_receipt(
                campaign_id=self.config.campaign_id,
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
            receipt.get("measurement_windows"), cell_root=cell_root)
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
        else:
            if not isinstance(belief, Mapping):
                raise ArenaCellRunnerError("controller checkpoint lacks its belief receipt")
            _verify_self_hash(belief, "belief receipt")
            persisted_belief = _load_json_object(
                cell_root / "belief-receipt.json", "persisted belief receipt")
            if persisted_belief != belief:
                raise ArenaCellRunnerError("persisted belief receipt drifted")
        return receipt

    def _verify_released_claim(self, receipt: Mapping[str, Any]) -> None:
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
                or released_receipt.campaign_id != self.config.campaign_id
                or not isinstance(released_receipt.released_at, str)
                or not released_receipt.released_at
                or opened_receipt.released_at is not None):
            raise ArenaCellRunnerError("checkpoint device claim was not cleanly released")

    def _verify_measurement_windows(self, windows: object, *, cell_root: Path) -> None:
        if not isinstance(windows, list) or len(windows) != 2:
            raise ArenaCellRunnerError(
                "checkpoint must carry exactly two GPU measurement windows")
        phases = ("vendor_baseline", "centralized_final_evaluation")
        claim_ids: list[str] = []
        for ordinal, (window, phase) in enumerate(zip(windows, phases), start=1):
            if not isinstance(window, Mapping):
                raise ArenaCellRunnerError("GPU measurement window is malformed")
            _verify_self_hash(window, "GPU measurement window")
            if (window.get("schema") != MEASUREMENT_WINDOW_SCHEMA
                    or window.get("phase") != phase
                    or window.get("ordinal") != ordinal
                    or window.get("status") != "complete"
                    or window.get("gpu_action_executed_only_while_claim_held") is not True
                    or not isinstance(window.get("device_sampling"), Mapping)):
                raise ArenaCellRunnerError(
                    "GPU measurement window identity or evidence is incomplete")
            self._verify_released_claim(window)
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
        return {
            "schema": CHECKPOINT_SCHEMA,
            "campaign_id": self.config.campaign_id,
            "arena_root": str(self.arena_root),
            "repository_root": str(REPOSITORY_ROOT),
            "cell_root": str(cell_root),
            "task": asdict(request.task),
            "arm": asdict(request.arm),
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


def _controller_argv(arm: Mapping[str, Any], checkpoint_hours: float) -> tuple[str, ...]:
    raw = arm.get("argv")
    if not isinstance(raw, list) or not raw or any(not isinstance(x, str) for x in raw):
        raise ArenaCellRunnerError("controller arm lacks a valid argv")
    argv = list(raw)
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


def _artifact_hashes(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows[path.relative_to(root).as_posix()] = _sha256_file(path)
    if not rows:
        raise ArenaCellRunnerError("checkpoint produced no artifacts")
    return rows


def run_worker(
    request: Mapping[str, Any], *,
    claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    sampler_factory: Callable[..., Any] = device_sampler.RocmSmiSampler,
) -> dict[str, Any]:
    """Execute one checkpoint, claiming only its two GPU measurements."""
    if request.get("schema") != CHECKPOINT_SCHEMA:
        raise ArenaCellRunnerError("worker request has the wrong schema")
    evaluator_python = _assert_worker_evaluator_identity(request)
    campaign_id = _safe_id(str(request.get("campaign_id")), "campaign_id")
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

    log_path = cell_root / "arena.log"
    logger = logging.getLogger(f"autokernel.arena.{task_id}.{arm_id}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
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
    baseline = bool(request.get("baseline"))
    controller_stdout_sha256 = None
    if not baseline:
        checkpoint = request.get("checkpoint_hours")
        if (isinstance(checkpoint, bool) or not isinstance(checkpoint, (int, float))
                or float(checkpoint) not in arena_campaign.MATCHED_BUDGET_HOURS):
            raise ArenaCellRunnerError("worker checkpoint is not a matched budget")
        raw_prompt = vendor_prompt.prompt_builder(
            str(config_path), workspace,
            {"target_gpu_model": arena_adapter.TARGET_GPU_MODEL}, logger)
        controller_environment = dict(environment)
        controller_environment.update({
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
        })
        prepared = arena_adapter.prepare_task(arena_adapter.ArenaTask(
            task_id=task_id,
            task_prompt=raw_prompt,
            workspace=str(workspace),
            controller_id=arm_id,
            round_id=f"{campaign_id}-{checkpoint:g}h",
            actual_gfx_arch=arena_adapter.TARGET_GFX_ARCH,
        ), base_environment=controller_environment)
        # This is the deliberately unclaimed gap.  The controller receives a
        # GPU-blind environment and may spend its remote-model budget while
        # another host tenant uses the MI210.
        stdout = arena_adapter.launch(
            prepared, _controller_argv(arm, float(checkpoint)),
            timeout_seconds=int(float(checkpoint) * 3600))
        controller_output = cell_root / "controller.stdout"
        controller_output.write_text(stdout, encoding="utf-8")
        controller_stdout_sha256 = _sha256_file(controller_output)
    elif request.get("checkpoint_hours") is not None:
        raise ArenaCellRunnerError("starting-state baseline cannot have a checkpoint budget")

    evaluation, evaluation_window = _run_gpu_measurement_window(
        request=request, cell_root=cell_root, ordinal=2,
        phase="centralized_final_evaluation",
        action=lambda: vendor_evaluator.evaluate_kernel(
            workspace, task_config, baseline_cases, logger, None),
        claim_acquirer=claim_acquirer, sampler_factory=sampler_factory)
    vendor_evaluator.write_task_result(
        workspace, evaluation, baseline_cases, task_id, arm_id, logger,
        create_plots=False)
    artifacts = _artifact_hashes(cell_root)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "authority": "whole_agent_task_only",
        "campaign_id": campaign_id,
        "task_id": task_id,
        "arm_id": arm_id,
        "baseline": baseline,
        "checkpoint_hours": request.get("checkpoint_hours"),
        "evaluation": {
            "pass_compilation": bool(evaluation.get("pass_compilation")),
            "pass_correctness": bool(evaluation.get("pass_correctness")),
            "valid_baseline_cases": int(evaluation.get("valid_baseline_cases", 0)),
            "valid_optimized_cases": int(evaluation.get("valid_optimized_cases", 0)),
            "average_speedup": float(evaluation.get("average_speedup", 0.0)),
        },
        "controller_stdout_sha256": controller_stdout_sha256,
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


def _run_manifest(
    args: argparse.Namespace, spec: arena_campaign.CampaignSpec,
    audit: Mapping[str, Any], *, available_source: bool,
) -> dict[str, Any]:
    preflight, preflight_file_sha256 = _load_preflight(args.preflight)
    selected_arms = (
        arena_campaign.AVAILABLE_SOURCE_PANEL_IDS
        if available_source else arena_campaign.PRIMARY_PANEL_IDS)
    return _self_hash({
        "schema": RUN_MANIFEST_SCHEMA,
        "campaign_id": audit["campaign_id"],
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
    "execute_from_cli", "run_worker",
]


if __name__ == "__main__":
    raise SystemExit(main())
