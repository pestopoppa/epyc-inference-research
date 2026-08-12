#!/usr/bin/env python3
"""Governed raw-HIP authoring seam for the physical MI210.

This is deliberately narrower than the Triton controller campaign. It accepts
only a real ``torch2hip`` task from the exact Apache-2.0 AgentKernelArena pin,
copies that task into a fresh campaign workspace, installs one separately
hash-bound HIP candidate, and calls the same centralized vendor evaluator used
by the controller arena. The output is an observation-only round-trip receipt;
it is never a performance claim or production-kernel candidate.

Compilation is performed GPU-blind. The PyTorch baseline and the centralized
correctness/timing evaluation each acquire and release their own MI210 claim.
No claim spans source authoring, controller deliberation, or CPU compilation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
import sys
from typing import Any, Callable, Mapping, Sequence

from . import arena_adapter
from ..execution import device_sampler
from ..resource import device_claim


SCHEMA = "epyc.autokernel.hip_authoring_roundtrip.v1"
PRODUCER_ID = "autokernel.controller.hip_authoring_arm/v1"
TASK_AUDIT_SCHEMA = "epyc.autokernel.hip_authoring_task_audit.v1"
WINDOW_SCHEMA = "epyc.autokernel.hip_authoring_gpu_window.v1"
AUTHORITY = "observation_only"
TARGET_DEVICE_ID = "mi210_0"
TARGET_GFX_ARCH = "gfx90a"
TARGET_GPU_MODEL = "MI210"
DEFAULT_ARENA_ROOT = Path("/mnt/raid0/llm/autokernel/vendor/agent-kernel-arena")
DEFAULT_CLAIM_JOURNAL = Path("/mnt/raid0/llm/ak-claims/device.jsonl")
DEFAULT_VISIBLE_DEVICE = "0"
DEFAULT_TASK = "torch2hip/gpumode/16636_SiLU"
_TASK_ID_RE = re.compile(r"torch2hip/[a-z0-9_-]+/[A-Za-z0-9_-]+")


class HipAuthoringError(RuntimeError):
    """A raw-HIP task or measurement cannot satisfy the governed seam."""


class HipAuthoringInterrupted(HipAuthoringError):
    """A polite signal interrupted a round trip after cleanup was armed."""


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


def _self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


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


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HipAuthoringError(f"{label} is not a regular non-symlink file: {path}")
    return path


def _contained(root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise HipAuthoringError(f"{label} escapes its governed root")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise HipAuthoringError(f"{label} escapes its governed root") from exc
    return resolved


def _task_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HipAuthoringError(
                f"vendor task contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _sha256_file(path)
        elif not path.is_dir():
            raise HipAuthoringError(
                f"vendor task contains a special file: {path.relative_to(root)}")
    if not files:
        raise HipAuthoringError("vendor task contains no files")
    return files


def toolchain_identity() -> dict[str, Any]:
    """Bind the exact evaluator Python, Ninja, and HIP compiler before a run."""
    # Keep the environment entrypoint rather than resolving its Python symlink:
    # the environment-local ``bin/ninja`` lives beside that entrypoint, while
    # resolving Python leads to the shared base interpreter in another tree.
    python_entrypoint = Path(os.path.abspath(sys.executable))
    python = _regular_file(python_entrypoint.resolve(), "evaluator Python")
    ninja = python_entrypoint.parent / "ninja"
    if not ninja.is_file() or not os.access(ninja, os.X_OK):
        raise HipAuthoringError(
            "pinned evaluator environment lacks executable ninja; "
            "AgentKernelArena requirements.txt declares it")
    hipcc_name = shutil.which("hipcc")
    if hipcc_name is None:
        raise HipAuthoringError("hipcc is unavailable on PATH")
    hipcc = _regular_file(Path(hipcc_name).resolve(), "hipcc")

    def version(argv: Sequence[str], label: str) -> str:
        try:
            result = subprocess.run(
                tuple(argv), capture_output=True, text=True, check=False, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HipAuthoringError(f"{label} version probe failed") from exc
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0 or not output:
            raise HipAuthoringError(f"{label} version probe failed")
        return output

    return {
        "evaluator_python": {"entrypoint": str(python_entrypoint),
                             "resolved_path": str(python),
                             "sha256": _sha256_file(python)},
        "ninja": {"path": str(ninja), "sha256": _sha256_file(ninja),
                  "version": version((str(ninja), "--version"), "ninja")},
        "hipcc": {"path": str(hipcc), "sha256": _sha256_file(hipcc),
                  "version": version((str(hipcc), "--version"), "hipcc")},
    }


@dataclass(frozen=True)
class HipTaskAudit:
    task_id: str
    task_root: str
    target_file: str
    source_files: tuple[str, ...]
    target_functions: tuple[str, ...]
    file_sha256: Mapping[str, str]
    vendor_identity: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_AUDIT_SCHEMA,
            "task_id": self.task_id,
            "task_root": self.task_root,
            "task_type": "torch2hip",
            "target_file": self.target_file,
            "source_files": list(self.source_files),
            "target_functions": list(self.target_functions),
            "file_sha256": dict(self.file_sha256),
            "vendor_identity": dict(self.vendor_identity),
            "target": {"gpu_model": TARGET_GPU_MODEL, "gfx_arch": TARGET_GFX_ARCH},
            "constraints": {
                "torch2hip_namesake_substitution_forbidden": True,
                "vendor_task_is_starting_spec_not_performance_evidence": True,
                "production_tree_touched": False,
            },
        }


def audit_task(arena_root: str | Path, task_id: str) -> HipTaskAudit:
    """Bind one true Torch2HIP task to the exact licensed Arena checkout."""
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise HipAuthoringError(
            "task_id must be an exact torch2hip/<suite>/<task> locator")
    root = Path(arena_root).resolve()
    vendor = arena_adapter.inspect_vendor_source(
        root, arena_adapter.AGENT_KERNEL_ARENA_PIN)
    task_root = _contained(root, f"tasks/{task_id}", "task root")
    if not task_root.is_dir() or task_root.is_symlink():
        raise HipAuthoringError(f"Torch2HIP task root is unavailable: {task_root}")
    config_path = _regular_file(task_root / "config.yaml", "task config")
    try:
        import yaml  # type: ignore[import-not-found]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HipAuthoringError("Torch2HIP task config is unreadable") from exc
    if not isinstance(config, Mapping) or config.get("task_type") != "torch2hip":
        raise HipAuthoringError("task config is not a real task_type=torch2hip task")
    target_file = str(config.get("target_file_path", ""))
    target = _contained(task_root, target_file, "target_file_path")
    if target.suffix != ".hip":
        raise HipAuthoringError("Torch2HIP target must be a .hip source file")
    sources = config.get("source_file_path")
    functions = config.get("target_kernel_functions")
    commands = tuple(config.get(name) for name in (
        "compile_command", "correctness_command", "performance_command"))
    if (not isinstance(sources, list) or not sources
            or any(not isinstance(item, str) for item in sources)):
        raise HipAuthoringError("Torch2HIP task has no source specification")
    if (not isinstance(functions, list) or not functions
            or any(not isinstance(item, str) or not item for item in functions)):
        raise HipAuthoringError("Torch2HIP task has no target functions")
    if any(not isinstance(command, list) or not command for command in commands):
        raise HipAuthoringError(
            "Torch2HIP task must expose compile, correctness, and performance commands")
    for source in sources:
        _regular_file(_contained(task_root, source, "source_file_path"), "task source")
    return HipTaskAudit(
        task_id=task_id, task_root=str(task_root), target_file=target_file,
        source_files=tuple(sources), target_functions=tuple(functions),
        file_sha256=_task_files(task_root), vendor_identity=vendor)


@contextmanager
def _gpu_visibility(device: str):
    keys = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = device
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _graceful_signals():
    """Convert TERM/INT into an exception so the active claim is released."""
    watched = (signal.SIGTERM, signal.SIGINT)
    previous = {item: signal.getsignal(item) for item in watched}

    def interrupt(signum: int, _frame: object) -> None:
        for item in watched:
            signal.signal(item, signal.SIG_IGN)
        raise HipAuthoringInterrupted(
            f"HIP round trip interrupted by {signal.Signals(signum).name}")

    try:
        for item in watched:
            signal.signal(item, interrupt)
        yield
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


@contextmanager
def _defer_campaign_signals():
    """Close the acquisition race before a claim is assigned locally."""
    watched = {signal.SIGTERM, signal.SIGINT}
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _measurement_window(
    *, phase: str, task_id: str, campaign_id: str, output_root: Path,
    claim_journal: Path, visible_device: str, claim_timeout_s: float,
    action: Callable[[], Any],
    claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    sampler_factory: Callable[..., Any] = device_sampler.RocmSmiSampler,
) -> tuple[Any, dict[str, Any]]:
    if phase not in {
        "vendor_baseline", "centralized_final_evaluation",
        "sealed_correctness", "exact_provider_timing",
    }:
        raise HipAuthoringError(f"unsupported HIP measurement phase: {phase}")
    window_path = output_root / "measurement-windows" / f"{phase}.json"
    if window_path.exists():
        raise HipAuthoringError(f"measurement window already exists: {window_path}")
    claim = None
    sampler = None
    opened = None
    released = None
    sampling = None
    result: Any = None
    failure: BaseException | None = None
    started_at = _utc_now()
    try:
        with _defer_campaign_signals():
            claim = claim_acquirer(
                TARGET_DEVICE_ID,
                purpose=f"AutoKernel HIP arm {phase} {task_id}",
                campaign_id=campaign_id,
                journal=device_claim.ClaimJournal(claim_journal),
                holder_label="hip_authoring_arm.py:measurement-window",
                timeout_s=claim_timeout_s,
                max_hold_s=3720.0,
            )
        opened = claim.receipt().to_dict()
        sampler = sampler_factory(
            device_index=int(visible_device), interval_s=0.250).start()
        with _gpu_visibility(visible_device):
            result = action()
    except BaseException as exc:
        failure = exc
    try:
        if sampler is not None:
            sampling = sampler.stop().to_dict()
    except BaseException as exc:
        failure = failure or exc
    try:
        if claim is not None:
            released = claim.release().to_dict()
    except BaseException as exc:
        failure = failure or exc
    receipt = _self_hash({
        "schema": WINDOW_SCHEMA,
        "phase": phase,
        "task_id": task_id,
        "campaign_id": campaign_id,
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
        raise failure
    return result, receipt


def _load_vendor_evaluator(arena_root: Path) -> Any:
    sys.path.insert(0, str(arena_root))
    try:
        from src import evaluator as vendor_evaluator  # type: ignore[import-not-found]
    except ImportError as exc:
        raise HipAuthoringError("cannot import pinned AgentKernelArena evaluator") from exc
    finally:
        if sys.path[0] == str(arena_root):
            sys.path.pop(0)
    return vendor_evaluator


def _fresh_workspace(output_root: Path, audit: HipTaskAudit,
                     candidate_source: Path) -> tuple[Path, str]:
    if output_root.exists():
        raise HipAuthoringError("output_root already exists; HIP round trips never resume in place")
    output_root.mkdir(parents=True)
    workspace = output_root / "workspace"
    shutil.copytree(audit.task_root, workspace)
    candidate = _regular_file(candidate_source.resolve(), "candidate source")
    if candidate.suffix != ".hip" or candidate.stat().st_size == 0:
        raise HipAuthoringError("candidate source must be a non-empty .hip file")
    target = _contained(workspace, audit.target_file, "workspace target")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, target)
    return workspace, _sha256_file(candidate)


def _public_correctness_cases(workspace: Path, correct: bool) -> int:
    """Read the vendor-produced case count; never infer it from timing rows."""
    path = _regular_file(
        workspace / "build/correctness_report.json", "correctness report")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HipAuthoringError("correctness report is unreadable") from exc
    if not isinstance(report, Mapping):
        raise HipAuthoringError("correctness report is not an object")
    cases = report.get("cases_run")
    if (isinstance(cases, bool) or not isinstance(cases, int) or cases < 1
            or (report.get("status") == "ok") != correct):
        raise HipAuthoringError(
            "correctness report status/case count disagrees with centralized evaluation")
    return cases


def _belief_measurement(*, measurement_id: str, metric: str, claim: str,
                        passed: int, total: int, role: str,
                        candidate_sha256: str) -> dict[str, Any]:
    if total < 1 or not 0 <= passed <= total:
        raise HipAuthoringError("belief measurement counts are invalid")
    return {
        "measurement_id": measurement_id,
        "metric": metric,
        "value": passed / total,
        "unit": "fraction",
        "metric_direction": "higher_better",
        "category": "CANDIDATE",
        "claim": claim,
        "reps": total,
        "reps_basis": "scored_public_cases",
        "extra": {
            "measurement_role": role,
            "passed": passed,
            "total": total,
            "candidate_source_sha256": candidate_sha256,
            "authority": AUTHORITY,
        },
    }


def run_roundtrip(
    *, arena_root: str | Path, task_id: str, candidate_source: str | Path,
    output_root: str | Path, campaign_id: str, claim_journal: str | Path,
    visible_device: str = DEFAULT_VISIBLE_DEVICE, claim_timeout_s: float = 3600.0,
    arch_detector: Callable[..., Mapping[str, Any]] = arena_adapter.detect_gfx_arch,
    claim_acquirer: Callable[..., Any] = device_claim.acquire_device_claim,
    sampler_factory: Callable[..., Any] = device_sampler.RocmSmiSampler,
) -> dict[str, Any]:
    """Compile, check, and time one raw-HIP candidate on gfx90a."""
    started_at = _utc_now()
    if not campaign_id or not re.fullmatch(r"[a-z][a-z0-9_.-]{2,95}", campaign_id):
        raise HipAuthoringError("campaign_id is not a safe governed identifier")
    if (isinstance(claim_timeout_s, bool) or not isinstance(claim_timeout_s, (int, float))
            or not math.isfinite(claim_timeout_s) or claim_timeout_s < 0):
        raise HipAuthoringError("claim_timeout_s must be finite and non-negative")
    arena = Path(arena_root).resolve()
    task = audit_task(arena, task_id)
    toolchain = toolchain_identity()
    hardware = dict(arch_detector())
    if hardware.get("architectures") != [TARGET_GFX_ARCH]:
        raise HipAuthoringError("physical hardware audit did not resolve exactly gfx90a")
    architecture_environment = arena_adapter.architecture_environment(os.environ)
    os.environ.update(architecture_environment)
    root = Path(output_root).resolve()
    workspace, candidate_sha256 = _fresh_workspace(
        root, task, Path(candidate_source))
    try:
        import yaml  # type: ignore[import-not-found]
        config = yaml.safe_load((workspace / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HipAuthoringError("copied task config is unreadable") from exc
    if not isinstance(config, dict):
        raise HipAuthoringError("copied task config is not an object")
    vendor = _load_vendor_evaluator(arena)

    with _gpu_visibility(""):
        compiled, compile_error = vendor.evaluate_compilation(
            workspace, config, None, None)
    if not compiled:
        receipt = _self_hash({
            "schema": SCHEMA, "authority": AUTHORITY,
            "campaign_id": campaign_id, "status": "compile_failed",
            "started_at": started_at, "ended_at": _utc_now(),
            "producer": {"producer_id": PRODUCER_ID,
                         "path": "scripts/kernel_rnd/autokernel/controller/hip_authoring_arm.py",
                         "sha256": _sha256_file(Path(__file__).resolve())},
            "task": task.to_dict(), "hardware": hardware, "toolchain": toolchain,
            "candidate": {"source": str(Path(candidate_source).resolve()),
                          "sha256": candidate_sha256},
            "evaluation": {"compiled": False, "correct": False,
                           "diagnostic_speedup_vs_torch_eager": None,
                           "diagnostics": compile_error,
                           "integrity_flags": ["compile_failed"]},
            "measurement_windows": [],
            "constraints": {"performance_claim": False,
                            "production_tree_touched": False},
        })
        _atomic_json(root / "receipt.json", receipt)
        return receipt

    baseline, baseline_window = _measurement_window(
        phase="vendor_baseline", task_id=task_id, campaign_id=campaign_id,
        output_root=root, claim_journal=Path(claim_journal),
        visible_device=visible_device, claim_timeout_s=float(claim_timeout_s),
        action=lambda: vendor.measure_baseline(workspace, config, None, None),
        claim_acquirer=claim_acquirer, sampler_factory=sampler_factory)
    evaluation, final_window = _measurement_window(
        phase="centralized_final_evaluation", task_id=task_id,
        campaign_id=campaign_id, output_root=root,
        claim_journal=Path(claim_journal), visible_device=visible_device,
        claim_timeout_s=float(claim_timeout_s),
        action=lambda: vendor.evaluate_kernel(
            workspace, config, baseline, None, None),
        claim_acquirer=claim_acquirer, sampler_factory=sampler_factory)
    if not isinstance(evaluation, Mapping):
        raise HipAuthoringError("centralized evaluator did not return an object")
    compiled = bool(evaluation.get("pass_compilation"))
    correct = bool(evaluation.get("pass_correctness"))
    baseline_cases = int(evaluation.get("valid_baseline_cases", 0))
    optimized_cases = int(evaluation.get("valid_optimized_cases", 0))
    correctness_cases = _public_correctness_cases(workspace, correct)
    speedup = float(evaluation.get("average_speedup", 0.0))
    integrity_flags = ["public_shapes_only", "honest_vendor_baseline_not_bound"]
    if baseline_cases < 1 or optimized_cases < 1 or not math.isfinite(speedup) or speedup <= 0:
        integrity_flags.append("timing_incomplete")
    receipt = _self_hash({
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "campaign_id": campaign_id,
        "status": "complete" if compiled and correct else "evaluation_failed",
        "started_at": started_at,
        "ended_at": _utc_now(),
        "producer": {
            "producer_id": PRODUCER_ID,
            "path": "scripts/kernel_rnd/autokernel/controller/hip_authoring_arm.py",
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "task": task.to_dict(),
        "hardware": hardware,
        "toolchain": toolchain,
        "candidate": {"source": str(Path(candidate_source).resolve()),
                      "sha256": candidate_sha256},
        "evaluation": {
            "compiled": compiled,
            "correct": correct,
            "public_correctness_cases": correctness_cases,
            "valid_baseline_cases": baseline_cases,
            "diagnostic_speedup_vs_torch_eager": speedup if speedup > 0 else None,
            "speedup_rankable": False,
            "profiler_metrics": {},
            "diagnostics": "",
            "integrity_flags": integrity_flags,
        },
        "measurement_windows": [baseline_window, final_window],
        "belief_measurements": [
            _belief_measurement(
                measurement_id="hip_public_correctness_pass_rate",
                metric="autokernel_hip_public_correctness_pass_rate",
                claim="Fraction of scored public Torch2HIP correctness cases that passed",
                passed=correctness_cases if correct else 0,
                total=correctness_cases,
                role="raw_hip_authoring_correctness",
                candidate_sha256=candidate_sha256),
            _belief_measurement(
                measurement_id="hip_timing_harness_validity_rate",
                metric="autokernel_hip_timing_harness_validity_rate",
                claim="Fraction of scored Torch2HIP timing cases admitted as valid",
                passed=optimized_cases,
                total=correctness_cases,
                role="raw_hip_timing_validity",
                candidate_sha256=candidate_sha256),
        ],
        "constraints": {
            "performance_claim": False,
            "promotion_authority": False,
            "candidate_is_not_llama_cpp_patch": True,
            "production_tree_touched": False,
            "frozen_kernel_built": False,
            "torch2hip_namesake_substitution_forbidden": True,
        },
    })
    _atomic_json(root / "receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-root", default=str(DEFAULT_ARENA_ROOT))
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--claim-journal", default=str(DEFAULT_CLAIM_JOURNAL))
    parser.add_argument("--visible-device", default=DEFAULT_VISIBLE_DEVICE)
    parser.add_argument("--claim-timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    with _graceful_signals():
        receipt = run_roundtrip(
            arena_root=args.arena_root, task_id=args.task_id,
            candidate_source=args.candidate_source, output_root=args.output_root,
            campaign_id=args.campaign_id, claim_journal=args.claim_journal,
            visible_device=args.visible_device,
            claim_timeout_s=args.claim_timeout_seconds)
    print(json.dumps({
        "status": receipt["status"], "campaign_id": receipt["campaign_id"],
        "receipt_sha256": receipt["receipt_sha256"],
        "output_root": str(Path(args.output_root).resolve()),
    }, sort_keys=True))
    return 0 if receipt["status"] == "complete" else 2


__all__ = [
    "AUTHORITY", "HipAuthoringError", "HipAuthoringInterrupted", "HipTaskAudit",
    "PRODUCER_ID", "SCHEMA",
    "TASK_AUDIT_SCHEMA", "WINDOW_SCHEMA", "audit_task", "run_roundtrip",
    "toolchain_identity",
]


if __name__ == "__main__":
    raise SystemExit(main())
