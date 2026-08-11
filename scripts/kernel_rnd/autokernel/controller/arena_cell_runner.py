#!/usr/bin/env python3
"""Governed AgentKernelArena execution bridge for the INF-03 campaign.

The campaign driver deliberately stops at a typed ``run_cell`` seam.  This
module supplies the concrete implementation without patching either pinned
vendor checkout.  Each non-baseline campaign cell becomes three independent
2 h / 8 h / 32 h runs from the same hash-bound task, and every GPU subprocess
runs beneath AutoKernel's cross-process MI210 claim.

Importing this module performs no model, compiler, evaluator, or GPU work.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from typing import Any, Callable, Mapping, Sequence

from . import arena_adapter, arena_campaign, arena_roundtrip
from ..execution import device_sampler
from ..resource import device_claim


RUNNER_SCHEMA = "epyc.autokernel.arena_cell_runner.v1"
CHECKPOINT_SCHEMA = "epyc.autokernel.arena_checkpoint.v1"
AGGREGATE_SCHEMA = "epyc.autokernel.arena_campaign_execution.v1"
IMPLEMENTATION_MODULE = Path(__file__).resolve()
REPOSITORY_ROOT = IMPLEMENTATION_MODULE.parents[4]
DEFAULT_CLAIM_JOURNAL = "/mnt/raid0/llm/ak-claims/device.jsonl"
DEFAULT_DEVICE_ID = "mi210_0"
EVALUATION_RESERVE_SECONDS = 7200
_ID_RE = re.compile(r"[a-z][a-z0-9_.-]{2,95}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ArenaCellRunnerError(RuntimeError):
    """A campaign cell cannot be executed with its declared evidence bounds."""


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
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ArenaCellRunnerError(f"{label} is not a safe campaign identifier")
    return value


def _assert_contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArenaCellRunnerError(f"{label} escapes its governed root") from exc
    return resolved


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
        if output.exists():
            raise ArenaCellRunnerError("output_root must not already exist")
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

    def __init__(self, config: RunnerConfig, *, worker: WorkerRunner | None = None,
                 claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
                 sampler_factory: Callable[..., Any] = device_sampler.RocmSmiSampler):
        if not isinstance(config, RunnerConfig):
            raise TypeError("config must be a RunnerConfig")
        self.config = config
        self.arena_root = Path(config.arena_root).resolve()
        self.output_root = Path(config.output_root).resolve()
        self.preflight, self.preflight_file_sha256 = _load_preflight(
            config.preflight_path)
        self._worker = worker or self._run_worker_subprocess
        self._claim_acquirer = claim_acquirer
        self._sampler_factory = sampler_factory
        self._ordinal = 0

    def __call__(self, request: arena_campaign.CampaignCellRequest) -> dict[str, Any]:
        if not isinstance(request, arena_campaign.CampaignCellRequest):
            raise TypeError("request must be a CampaignCellRequest")
        if request.is_starting_state_baseline:
            runs = [self._run_checkpoint(request, checkpoint_hours=None)]
        else:
            runs = [self._run_checkpoint(request, checkpoint_hours=hours)
                    for hours in request.checkpoint_hours]
        return _self_hash({
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
                "one_mi210_cell_at_a_time": True,
                "promotion_authority": False,
            },
        })

    def _run_checkpoint(
        self, request: arena_campaign.CampaignCellRequest,
        *, checkpoint_hours: float | None,
    ) -> dict[str, Any]:
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
            f"{self._ordinal:03d}-{request.task.task_id}-{request.arm.arm_id}-"
            f"{checkpoint_name}"
        )
        cell_root = self.output_root / "cells" / cell_id
        if cell_root.exists():
            raise ArenaCellRunnerError(f"cell output already exists: {cell_root}")
        cell_root.mkdir(parents=True)
        started_at = _utc_now()
        worker_request = self._worker_request(
            request, checkpoint_hours=checkpoint_hours, cell_root=cell_root)
        _atomic_json(cell_root / "worker-request.json", worker_request)
        max_runtime = ((checkpoint_hours or 0.0) * 3600
                       + EVALUATION_RESERVE_SECONDS)
        claim = self._claim_acquirer(
            self.config.device_id,
            purpose=(f"INF-03 Arena {request.arm.arm_id} {request.task.task_id} "
                     f"{checkpoint_name}"),
            campaign_id=self.config.campaign_id,
            journal=device_claim.ClaimJournal(self.config.claim_journal),
            holder_label="arena_cell_runner.py",
            timeout_s=float(self.config.claim_timeout_seconds),
            max_hold_s=max_runtime + 120.0,
        )
        opened = claim.receipt().to_dict()
        sampler = None
        sampling = None
        worker_result: Mapping[str, Any] | None = None
        try:
            sampler = self._sampler_factory(
                device_index=int(self.config.visible_device), interval_s=0.250).start()
            worker_result = self._worker(worker_request, max_runtime)
        finally:
            try:
                if sampler is not None:
                    sampling = sampler.stop().to_dict()
            finally:
                released = claim.release().to_dict()
        if worker_result is None or sampling is None:
            raise ArenaCellRunnerError("Arena worker completed without durable result or sampling")
        if worker_result.get("schema") != CHECKPOINT_SCHEMA:
            raise ArenaCellRunnerError("Arena worker returned the wrong checkpoint schema")
        artifact_map = worker_result.get("artifacts")
        if not isinstance(artifact_map, Mapping) or not artifact_map:
            raise ArenaCellRunnerError("Arena worker returned no hash-bound artifacts")
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
        receipt = _self_hash({
            **dict(worker_result),
            "started_at": started_at,
            "ended_at": _utc_now(),
            "device_claim_open": opened,
            "device_claim_released": released,
            "device_sampling": sampling,
            "preflight": {
                "path": str(Path(self.config.preflight_path).resolve()),
                "file_sha256": self.preflight_file_sha256,
                "receipt_sha256": self.preflight["receipt_sha256"],
            },
            "runner": {
                "path": str(IMPLEMENTATION_MODULE),
                "sha256": _sha256_file(IMPLEMENTATION_MODULE),
            },
            "belief_receipt": belief_receipt,
        })
        _atomic_json(cell_root / "checkpoint-receipt.json", receipt)
        return receipt

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
        }

    @staticmethod
    def _run_worker_subprocess(
        request: Mapping[str, Any], timeout_seconds: float,
    ) -> Mapping[str, Any]:
        cell_root = Path(str(request["cell_root"]))
        output = cell_root / "worker-result.json"
        command = (
            sys.executable, "-m",
            "scripts.kernel_rnd.autokernel.controller.arena_cell_runner",
            "--worker-request", str(cell_root / "worker-request.json"),
            "--worker-output", str(output),
        )
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
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate(timeout=5)
            raise ArenaCellRunnerError(
                f"Arena worker exceeded its {timeout_seconds:g}s ceiling")
        (cell_root / "worker.stdout").write_text(stdout, encoding="utf-8")
        (cell_root / "worker.stderr").write_text(stderr, encoding="utf-8")
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


def run_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one already-claimed baseline or controller checkpoint."""
    if request.get("schema") != CHECKPOINT_SCHEMA:
        raise ArenaCellRunnerError("worker request has the wrong schema")
    campaign_id = _safe_id(str(request.get("campaign_id")), "campaign_id")
    arena_root = Path(str(request.get("arena_root"))).resolve()
    repository_root = Path(str(request.get("repository_root"))).resolve()
    cell_root = Path(str(request.get("cell_root"))).resolve()
    _assert_contained(cell_root, cell_root.parent.parent, "cell_root")
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
    baseline_cases = vendor_evaluator.measure_baseline(
        workspace, task_config, logger, None)
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
        prepared = arena_adapter.prepare_task(arena_adapter.ArenaTask(
            task_id=task_id,
            task_prompt=raw_prompt,
            workspace=str(workspace),
            controller_id=arm_id,
            round_id=f"{campaign_id}-{checkpoint:g}h",
            actual_gfx_arch=arena_adapter.TARGET_GFX_ARCH,
        ), base_environment=environment)
        stdout = arena_adapter.launch(
            prepared, _controller_argv(arm, float(checkpoint)),
            timeout_seconds=int(float(checkpoint) * 3600))
        controller_output = cell_root / "controller.stdout"
        controller_output.write_text(stdout, encoding="utf-8")
        controller_stdout_sha256 = _sha256_file(controller_output)
    elif request.get("checkpoint_hours") is not None:
        raise ArenaCellRunnerError("starting-state baseline cannot have a checkpoint budget")

    evaluation = vendor_evaluator.evaluate_kernel(
        workspace, task_config, baseline_cases, logger, None)
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
        "artifacts": artifacts,
        "constraints": {
            "starting_state_copied_fresh": True,
            "centralized_vendor_evaluator": True,
            "agent_reported_performance_admitted": False,
            "promotion_authority": False,
        },
    }


def execute_from_cli(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    spec = arena_campaign.load_spec(args.config)
    audit = arena_campaign.audit_campaign(
        spec, arena_root=args.arena_root, geak_root=args.geak_root,
        enumerator=args.enumerator)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    arena_campaign.write_receipt(output_root / "audit.json", audit)
    if audit["status"] != "ready":
        return 3, {
            "schema": AGGREGATE_SCHEMA,
            "campaign_id": spec.campaign_id,
            "status": "refused",
            "audit": str(output_root / "audit.json"),
            "controller_or_gpu_command_executed": False,
        }
    # RunnerConfig requires a not-yet-existing root so cell artifacts cannot mix
    # with the audit or any predecessor campaign.
    cells_root = output_root / "execution"
    runner = GovernedArenaCellRunner(RunnerConfig(
        campaign_id=spec.campaign_id,
        arena_root=str(Path(args.arena_root).resolve()),
        preflight_path=str(Path(args.preflight).resolve()),
        output_root=str(cells_root),
        claim_journal=args.claim_journal,
        claim_timeout_seconds=args.claim_timeout_seconds,
    ))
    cells = arena_campaign.execute_campaign(spec, audit, run_cell=runner)
    aggregate = _self_hash({
        "schema": AGGREGATE_SCHEMA,
        "campaign_id": spec.campaign_id,
        "status": "complete",
        "authority": "whole_agent_task_only",
        "audit": str(output_root / "audit.json"),
        "audit_receipt_sha256": audit["receipt_sha256"],
        "cells": cells,
        "constraints": {"partial_results_rankable": False,
                        "promotion_authority": False},
    })
    _atomic_json(output_root / "execution-receipt.json", aggregate)
    return 0, aggregate


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
    "AGGREGATE_SCHEMA", "CHECKPOINT_SCHEMA", "RUNNER_SCHEMA",
    "ArenaCellRunnerError", "GovernedArenaCellRunner", "RunnerConfig",
    "execute_from_cli", "run_worker",
]


if __name__ == "__main__":
    raise SystemExit(main())
