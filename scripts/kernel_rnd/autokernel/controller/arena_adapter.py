"""Fail-closed AgentKernelArena/GEAK adapter contract for the MI210.

The arena owns task isolation and scoring.  AutoKernel owns prompt hygiene,
priced context, evidence binding, and the exact gfx90a execution contract.  This
module joins those seams without importing either vendor project into the
campaign path and without assigning a performance verdict.

The paper-era source pins are deliberate.  A later checkout is a new input,
not an invisible upgrade:

* AgentKernelArena ``2dbbf1d3f676b948c04c339de50516fe80ed4ab9``;
* GEAK v1 ``4ffba15a55f250816598b4e27eb56ca40a699cea``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import authoring_contract


PREFLIGHT_SCHEMA = "epyc.autokernel.geak_arena_preflight.v1"
TARGET_GPU_MODEL = "MI210"
TARGET_GFX_ARCH = "gfx90a"
ARCH_ENV_KEYS = ("PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS")
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_CONTROLLER_RE = re.compile(r"[a-z][a-z0-9_]{2,63}")


class ArenaAdapterError(ValueError):
    """A vendor source, hardware identity, or adapter request is unsafe."""


@dataclass(frozen=True)
class VendorPin:
    name: str
    commit: str
    license_path: str
    required_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ArenaAdapterError("vendor pin name must be non-empty")
        if not _COMMIT_RE.fullmatch(self.commit):
            raise ArenaAdapterError(f"{self.name}: commit must be a full lowercase SHA-1")
        paths = (self.license_path, *self.required_paths)
        if any(Path(value).is_absolute() or ".." in Path(value).parts for value in paths):
            raise ArenaAdapterError(f"{self.name}: source paths must stay repository-relative")


AGENT_KERNEL_ARENA_PIN = VendorPin(
    name="AgentKernelArena",
    commit="2dbbf1d3f676b948c04c339de50516fe80ed4ab9",
    license_path="LICENSE",
    required_paths=(
        "agents/__init__.py",
        "src/module_registration.py",
        "agents/codex/launch_agent.py",
        "agents/geak_v3_triton/launch_agent.py",
        "src/evaluator.py",
        "src/prompts/cheatsheet/default_cheatsheet.yaml",
    ),
)
GEAK_V1_PIN = VendorPin(
    name="GEAK-v1",
    commit="4ffba15a55f250816598b4e27eb56ca40a699cea",
    license_path="LICENSE.md",
    required_paths=(
        "src/agents/OptimAgent_ROCm.py",
        "src/dataloaders/ROCm.py",
        "src/main_optimagent_ROCm.py",
        "src/configs/rocm_optimagent_config.yaml",
    ),
)


@dataclass(frozen=True)
class ControllerSpec:
    controller_id: str
    family: str
    roles: tuple[str, ...]
    evidence_scope: str = "whole_agent_task_only"

    def __post_init__(self) -> None:
        if not _CONTROLLER_RE.fullmatch(self.controller_id):
            raise ArenaAdapterError(f"invalid controller id {self.controller_id!r}")
        if not self.family.strip() or not self.roles:
            raise ArenaAdapterError("controller family and roles must be non-empty")
        allowed = {"planner", "actor", "critic", "exploit"}
        if set(self.roles) - allowed:
            raise ArenaAdapterError(f"{self.controller_id}: unknown authoring role")
        if self.evidence_scope != "whole_agent_task_only":
            raise ArenaAdapterError("arena adapters may carry only whole-agent task evidence")


CONTROLLERS = {
    row.controller_id: row for row in (
        ControllerSpec("claude_codex_actor_critic", "actor_critic",
                       ("planner", "critic", "actor", "exploit")),
        ControllerSpec("evoengineer", "evolutionary", ("planner", "actor", "critic")),
        ControllerSpec("kernelfoundry", "map_elites", ("planner", "actor", "critic")),
        ControllerSpec("k_search", "world_model_tree", ("planner", "actor", "critic")),
        ControllerSpec("xe_forge", "linear_cover", ("planner", "actor", "critic")),
        ControllerSpec("geak_v1", "geak_optimagent", ("planner", "actor", "critic")),
        ControllerSpec("argus", "agentic_gpu_optimization",
                       ("planner", "actor", "critic", "exploit")),
    )
}


@dataclass(frozen=True)
class ArenaTask:
    task_id: str
    task_prompt: str
    workspace: str
    controller_id: str
    round_id: str
    actual_gfx_arch: str
    c4_report_path: str | None = None
    c4_report_sha256: str | None = None
    c5_seed_ids: tuple[str, ...] = ()
    datatype_target_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for label in ("task_id", "task_prompt", "round_id"):
            if not str(getattr(self, label)).strip():
                raise ArenaAdapterError(f"{label} must be non-empty")
        if self.controller_id not in CONTROLLERS:
            raise ArenaAdapterError(
                f"unknown controller {self.controller_id!r}; registered={sorted(CONTROLLERS)}")
        workspace = Path(self.workspace)
        if not workspace.is_absolute() or not workspace.is_dir():
            raise ArenaAdapterError("workspace must be an existing absolute directory")
        if self.actual_gfx_arch != TARGET_GFX_ARCH:
            raise ArenaAdapterError(
                f"MI210 arena cell requires {TARGET_GFX_ARCH}, observed {self.actual_gfx_arch!r}")
        if (self.c4_report_path is None) != (self.c4_report_sha256 is None):
            raise ArenaAdapterError("C4 report path and SHA-256 must be supplied together")
        if (not isinstance(self.c5_seed_ids, tuple)
                or any(not isinstance(seed_id, str) or not seed_id
                       for seed_id in self.c5_seed_ids)):
            raise ArenaAdapterError("C5 seed ids must be a tuple of non-empty strings")
        if (not isinstance(self.datatype_target_ids, tuple)
                or any(not isinstance(target_id, str) or not target_id
                       for target_id in self.datatype_target_ids)):
            raise ArenaAdapterError(
                "datatype target ids must be a tuple of non-empty strings")


@dataclass(frozen=True)
class PreparedArenaTask:
    task: ArenaTask
    prompt: str
    prompt_sha256: str
    environment: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(argv: Sequence[str], *, cwd: Path | None = None,
         env: Mapping[str, str] | None = None, timeout: int = 30,
         input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise ArenaAdapterError("argv must contain non-empty strings")
    try:
        return subprocess.run(
            list(argv), cwd=cwd, env=None if env is None else dict(env),
            input=input_text, capture_output=True, text=True, check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArenaAdapterError(f"command failed to start or finish: {argv[0]}") from exc


def _git(root: Path, *args: str) -> str:
    result = _run(("git", "-C", str(root), *args))
    if result.returncode != 0:
        raise ArenaAdapterError(
            f"git {' '.join(args)} failed for {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def inspect_vendor_source(root: str | Path, pin: VendorPin) -> dict[str, Any]:
    """Prove one vendor checkout is the exact clean source named by the handoff."""
    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise ArenaAdapterError(f"{pin.name}: source root does not exist: {source_root}")
    observed = _git(source_root, "rev-parse", "HEAD")
    if observed != pin.commit:
        raise ArenaAdapterError(
            f"{pin.name}: expected commit {pin.commit}, observed {observed}")
    dirty = _git(source_root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ArenaAdapterError(f"{pin.name}: source checkout is not clean")
    identities: dict[str, str] = {}
    for relative in (pin.license_path, *pin.required_paths):
        path = source_root / relative
        if not path.is_file():
            raise ArenaAdapterError(f"{pin.name}: required source file missing: {relative}")
        identities[relative] = _sha256_file(path)
    return {
        "name": pin.name,
        "root": str(source_root),
        "commit": observed,
        "clean": True,
        "license": {"path": pin.license_path,
                    "sha256": identities[pin.license_path]},
        "required_file_sha256": identities,
    }


def detect_gfx_arch(enumerator: str = "/opt/rocm/bin/rocm_agent_enumerator") -> dict[str, Any]:
    """Detect the physical GPU architecture; CPU agents and gfx overrides do not count."""
    executable = Path(enumerator).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ArenaAdapterError(f"ROCm agent enumerator is not executable: {executable}")
    result = _run((str(executable), "-name"))
    if result.returncode != 0:
        raise ArenaAdapterError(
            f"ROCm agent enumeration failed: {result.stderr.strip()}")
    raw = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    architectures = tuple(sorted({line.split(":", 1)[0] for line in raw
                                  if line.startswith("gfx")}))
    if architectures != (TARGET_GFX_ARCH,):
        raise ArenaAdapterError(
            f"expected exactly one architecture family {TARGET_GFX_ARCH}, "
            f"observed {architectures}")
    return {
        "enumerator": str(executable),
        "enumerator_sha256": _sha256_file(executable),
        "raw_agents": list(raw),
        "architectures": list(architectures),
        "target_gpu_model": TARGET_GPU_MODEL,
        "target_gfx_arch": TARGET_GFX_ARCH,
    }


def validate_arena_registry(arena_root: str | Path) -> dict[str, Any]:
    """Exercise the pinned decorator and enumerate vendor-loadable controller types."""
    root = Path(arena_root).resolve()
    probe = (
        "import json\n"
        "from agents import AGENT_REGISTRY, register_agent\n"
        "from src.module_registration import AgentType\n"
        "@register_agent('epyc_contract_probe')\n"
        "def probe(*args): return 'ok'\n"
        "assert AGENT_REGISTRY['epyc_contract_probe'] is probe\n"
        "print(json.dumps({'registered': sorted(AGENT_REGISTRY), "
        "'agent_types': sorted(x.value for x in AgentType)}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    result = _run((sys.executable, "-c", probe), cwd=root, env=env)
    if result.returncode != 0:
        raise ArenaAdapterError(
            f"AgentKernelArena registry probe failed: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ArenaAdapterError("AgentKernelArena registry probe emitted invalid JSON") from exc
    required = {"codex", "geak_v3_triton"}
    missing = sorted(required - set(payload.get("agent_types", ())))
    if missing:
        raise ArenaAdapterError(f"AgentKernelArena registry is missing {missing}")
    # The decorator accepts an external adapter, while AgentType cannot select it
    # until the vendor enum/import dispatch is overlaid.  This is an explicit
    # integration fact, not something callers are allowed to discover mid-run.
    payload["external_decorator_registration"] = "epyc_contract_probe" in payload["registered"]
    payload["external_type_dispatch_present"] = "epyc_autokernel" in payload["agent_types"]
    payload["overlay_required"] = not payload["external_type_dispatch_present"]
    return payload


def architecture_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    override = env.get("HSA_OVERRIDE_GFX_VERSION")
    if override:
        raise ArenaAdapterError(
            "HSA_OVERRIDE_GFX_VERSION would falsify the physical gfx90a identity")
    for key in ARCH_ENV_KEYS:
        existing = env.get(key)
        if existing and existing != TARGET_GFX_ARCH:
            raise ArenaAdapterError(
                f"{key} conflicts with MI210 target: {existing!r}")
        env[key] = TARGET_GFX_ARCH
    # Arena executes controllers from a fresh copied task workspace.  In-tree
    # controller modules therefore need an explicit import root; relying on the
    # parent process's cwd made the real worker fail before its first model call.
    existing_pythonpath = [
        value for value in env.get("PYTHONPATH", "").split(os.pathsep) if value
    ]
    repository = str(REPOSITORY_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [repository, *(value for value in existing_pythonpath if value != repository)]
    )
    return env


def prepare_task(task: ArenaTask, *, base_environment: Mapping[str, str] | None = None,
                 max_context_tokens: int = 4096) -> PreparedArenaTask:
    """Render the one prompt and environment admissible to an MI210 arena cell."""
    architecture = authoring_contract.ContextItem(
        source_ref="hardware://mi210/gfx90a",
        purpose="bind compilation and optimization advice to the physical CDNA2 target",
        content=(
            "Target AMD Instinct MI210, CDNA2 gfx90a, wavefront 64. "
            "Compile every HIP/Triton artifact for gfx90a. Treat any gfx942/gfx950 "
            "number or instruction as non-transferable until measured on this card."
        ),
    )
    items = [architecture]
    if task.c4_report_path is not None:
        items.append(authoring_contract.c4_profile_context_item(
            task.c4_report_path, expected_sha256=task.c4_report_sha256 or ""))
    if task.c5_seed_ids:
        from .. import c5_seed_corpus

        items.append(c5_seed_corpus.seed_context_item(task.c5_seed_ids))
    if task.datatype_target_ids:
        from .. import datatype_targets

        items.append(datatype_targets.target_context_item(task.datatype_target_ids))
    priced = authoring_contract.price_context(
        round_id=task.round_id,
        budget=authoring_contract.ContextBudget(
            max_total_tokens=max_context_tokens,
            max_item_tokens=max_context_tokens,
            max_items=4,
        ),
        items=items,
    )
    prompt = authoring_contract.assemble_authoring_prompt(
        role="actor", task=task.task_prompt, context=priced)
    return PreparedArenaTask(
        task=task,
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        environment=architecture_environment(base_environment),
    )


def launch(
    prepared: PreparedArenaTask, argv: Sequence[str], *, timeout_seconds: int,
    command_prefix: Sequence[str] = (),
    process_started: Callable[[int], None] | None = None,
) -> str:
    """Launch one registered controller through stdin; shell interpolation is forbidden."""
    if not isinstance(prepared, PreparedArenaTask):
        raise TypeError("prepared must be a PreparedArenaTask")
    if isinstance(timeout_seconds, bool) or timeout_seconds < 1:
        raise ArenaAdapterError("timeout_seconds must be a positive integer")
    if not argv:
        raise ArenaAdapterError("controller argv must not be empty")
    executable = shutil.which(argv[0], path=prepared.environment.get("PATH"))
    if executable is None:
        raise ArenaAdapterError(f"controller executable not found: {argv[0]}")
    if any(not isinstance(part, str) or not part for part in command_prefix):
        raise ArenaAdapterError("command_prefix must contain non-empty strings")
    bound = (*command_prefix, executable, *argv[1:])
    try:
        process = subprocess.Popen(
            bound, cwd=Path(prepared.task.workspace), env=prepared.environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        if process_started is not None:
            process_started(process.pid)
        stdout, stderr = process.communicate(
            input=prepared.prompt, timeout=timeout_seconds)
    except BaseException as exc:
        if "process" in locals() and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if "process" in locals():
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ArenaAdapterError("controller failed to start or finish") from exc
    if process.returncode != 0:
        diagnostic = stderr.strip()[-4096:]
        raise ArenaAdapterError(
            f"controller {prepared.task.controller_id} exited {process.returncode}: "
            f"{diagnostic}")
    return stdout


def register_agentkernelarena_adapter(
    register_agent: Callable[[str], Callable[[Callable[..., str]], Callable[..., str]]],
    prompt_builder: Callable[[Mapping[str, Any], str, str], str],
) -> Callable[[Mapping[str, Any], str, str], str]:
    """Register the exact three-argument launcher AgentKernelArena expects.

    The tiny vendor-side module supplies its own ``register_agent`` decorator
    and a wrapper around its standard prompt builder.  All EPYC-specific
    validation stays here and is independently testable.
    """
    if not callable(register_agent) or not callable(prompt_builder):
        raise TypeError("register_agent and prompt_builder must be callable")

    @register_agent("epyc_autokernel")
    def arena_launcher(eval_config: Mapping[str, Any], task_config_dir: str,
                       workspace: str) -> str:
        if not isinstance(eval_config, Mapping):
            raise ArenaAdapterError("AgentKernelArena eval_config must be an object")
        config = eval_config.get("epyc_autokernel")
        if not isinstance(config, Mapping):
            raise ArenaAdapterError("eval_config.epyc_autokernel must be an object")
        argv = config.get("argv")
        if (not isinstance(argv, (list, tuple)) or not argv
                or any(not isinstance(part, str) or not part for part in argv)):
            raise ArenaAdapterError("epyc_autokernel.argv must be non-empty strings")
        controller_id = config.get("controller_id")
        if not isinstance(controller_id, str):
            raise ArenaAdapterError("epyc_autokernel.controller_id must be a string")
        timeout = config.get("timeout_seconds", 600)
        if isinstance(timeout, bool) or not isinstance(timeout, int):
            raise ArenaAdapterError("epyc_autokernel.timeout_seconds must be an integer")
        prompt = prompt_builder(eval_config, task_config_dir, workspace)
        seed_ids = config.get("c5_seed_ids", ())
        if (not isinstance(seed_ids, (list, tuple))
                or any(not isinstance(seed_id, str) or not seed_id
                       for seed_id in seed_ids)):
            raise ArenaAdapterError(
                "epyc_autokernel.c5_seed_ids must be non-empty strings")
        datatype_target_ids = config.get("datatype_target_ids", ())
        if (not isinstance(datatype_target_ids, (list, tuple))
                or any(not isinstance(target_id, str) or not target_id
                       for target_id in datatype_target_ids)):
            raise ArenaAdapterError(
                "epyc_autokernel.datatype_target_ids must be non-empty strings")
        hardware = detect_gfx_arch(str(config.get(
            "enumerator", "/opt/rocm/bin/rocm_agent_enumerator")))
        task = ArenaTask(
            task_id=str(config.get("task_id") or Path(task_config_dir).name),
            task_prompt=prompt,
            workspace=str(Path(workspace).resolve()),
            controller_id=controller_id,
            round_id=str(config.get("round_id") or "arena-round-1"),
            actual_gfx_arch=hardware["target_gfx_arch"],
            c4_report_path=config.get("c4_report_path"),
            c4_report_sha256=config.get("c4_report_sha256"),
            c5_seed_ids=tuple(seed_ids),
            datatype_target_ids=tuple(datatype_target_ids),
        )
        return launch(prepare_task(task), tuple(argv), timeout_seconds=timeout)

    return arena_launcher


def run_preflight(*, arena_root: str | Path, geak_root: str | Path,
                  enumerator: str = "/opt/rocm/bin/rocm_agent_enumerator") -> dict[str, Any]:
    """Return a diagnostic-only, hash-bound readiness receipt; run no model or kernel."""
    arena = inspect_vendor_source(arena_root, AGENT_KERNEL_ARENA_PIN)
    geak = inspect_vendor_source(geak_root, GEAK_V1_PIN)
    hardware = detect_gfx_arch(enumerator)
    registry = validate_arena_registry(arena_root)
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "authority": "diagnostic_only",
        "sources": {"agent_kernel_arena": arena, "geak_v1": geak},
        "hardware": hardware,
        "registry": registry,
        "registered_controller_ids": sorted(CONTROLLERS),
        "constraints": {
            "target_gpu_model": TARGET_GPU_MODEL,
            "target_gfx_arch": TARGET_GFX_ARCH,
            "arch_environment": {key: TARGET_GFX_ARCH for key in ARCH_ENV_KEYS},
            "vendor_performance_numbers_transfer": False,
            "model_or_kernel_executed": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return receipt


def write_preflight_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically persist the diagnostic receipt at an explicitly named path."""
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-root", required=True)
    parser.add_argument("--geak-root", required=True)
    parser.add_argument("--enumerator", default="/opt/rocm/bin/rocm_agent_enumerator")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = run_preflight(
        arena_root=args.arena_root, geak_root=args.geak_root,
        enumerator=args.enumerator)
    output = write_preflight_receipt(args.output, receipt)
    print(json.dumps({
        "output": str(output), "receipt_sha256": receipt["receipt_sha256"],
        "target_gfx_arch": receipt["hardware"]["target_gfx_arch"],
        "overlay_required": receipt["registry"]["overlay_required"],
    }, sort_keys=True))
    return 0


__all__ = [
    "AGENT_KERNEL_ARENA_PIN", "GEAK_V1_PIN", "CONTROLLERS", "TARGET_GPU_MODEL",
    "TARGET_GFX_ARCH", "ArenaAdapterError", "ArenaTask", "PreparedArenaTask",
    "architecture_environment", "detect_gfx_arch", "inspect_vendor_source", "launch",
    "prepare_task", "register_agentkernelarena_adapter", "run_preflight",
    "validate_arena_registry",
]


if __name__ == "__main__":
    raise SystemExit(main())
