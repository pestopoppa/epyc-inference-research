#!/usr/bin/env python3
"""Licensed GEAK-v1 OptimAgent_ROCm port for governed AgentKernelArena cells."""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import sys
from typing import Any, Callable, Mapping, Sequence

from . import arena_adapter, arena_upstream_common as common


CONTROLLER_ID = "geak_v1"
SOURCE_COMMIT = "4ffba15a55f250816598b4e27eb56ca40a699cea"
SOURCE_PIN = arena_adapter.VendorPin(
    name="GEAK-v1", commit=SOURCE_COMMIT, license_path="LICENSE.md",
    required_paths=(
        "src/agents/OptimAgent_ROCm.py",
        "src/agents/reflexion_oneshot.py",
        "src/dataloaders/TB_eval/train_crawl.json",
    ),
)
DEFAULT_SOURCE_ROOT = Path(
    "/mnt/raid0/llm/autokernel/vendor/arena-controllers/geak-v1")
RUNTIME_PYTHON = Path(
    "/mnt/raid0/llm/tools/geak-v1-rocm62-py312/bin/python")
ENTRYPOINT_RELATIVE = (
    "scripts/kernel_rnd/autokernel/controller/geak_v1_arena.py")
EXECUTABLE_MODULE = "scripts.kernel_rnd.autokernel.controller.geak_v1_arena"
UPSTREAM_ENTRYPOINT = "src/agents/OptimAgent_ROCm.py"
PINNED_MODEL_IDS = common.PINNED_MODEL_IDS
REQUIRED_CLIS = common.REQUIRED_CLIS
_SAFE_RUNTIME_ROOT = re.compile(r"[A-Za-z0-9_./-]+")
_GEAK_TOP_LEVEL_PACKAGES = (
    "agents", "dataloaders", "memories", "models", "prompts", "retrievers",
    "utils")


class GeakArenaError(common.UpstreamControllerError):
    """GEAK-v1 cannot preserve the governed Arena execution contract."""


class GeakTextModel:
    """Expose GEAK's ``model.generate`` interface over the fixed text model."""

    def __init__(self, model: common.CodexTextModel):
        self.model = model

    def generate(self, messages: list[Mapping[str, Any]], **kwargs: Any) -> str:
        del kwargs
        prompt = "\n\n".join(
            str(row.get("content", "")) for row in messages
            if isinstance(row, Mapping))
        return self.model.call(prompt)


class GeakArenaDataset:
    """GEAK ROCm dataset surface whose tests are exclusively Arena-owned."""

    rocm_tests = True

    def __init__(
        self, *, prompt: str, evaluator: common.ArenaWorkspaceEvaluator,
        runtime_root: Path,
    ):
        if len(evaluator.source_paths) != 1:
            raise GeakArenaError("the GEAK-v1 port requires one Arena source file")
        self.evaluator = evaluator
        self.runtime_root = runtime_root.resolve()
        if (not _SAFE_RUNTIME_ROOT.fullmatch(str(self.runtime_root))
                or not self.runtime_root.is_relative_to(evaluator.workspace)):
            raise GeakArenaError("GEAK runtime root must be a safe workspace child")
        self.runtime_root.mkdir(parents=True, exist_ok=False)
        self.log_root = self.runtime_root / "output"
        self.filename = evaluator.source_paths[0]
        source = (evaluator.workspace / self.filename).read_text(encoding="utf-8")
        targets = evaluator.config.get("target_kernel_functions")
        if not isinstance(targets, list) or len(targets) != 1 or not targets[0]:
            raise GeakArenaError("the GEAK-v1 port requires one target kernel symbol")
        self.problem_states = [SimpleNamespace(
            instruction=evaluator.definition(prompt), label=source,
            filename=Path(self.filename).name,
            target_kernel_name=str(targets[0]), test_code="",
            solution=source, pass_call=False, pass_exe=False,
        )]
        self._last: common.EvaluationRecord | None = None

    def __len__(self) -> int:
        return 1

    def _bounded_dir(self, value: str, label: str) -> Path:
        path = Path(value)
        resolved = (self.log_root / path).resolve() if not path.is_absolute() else path.resolve()
        if not resolved.is_relative_to(self.runtime_root):
            raise GeakArenaError(f"GEAK {label} escapes its safe runtime root")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def test_opt_correctness(
        self, code: str, filename: str, tmp_dir: str = "temp",
        save_scripts: bool = True, exe_dir: str = "pass_exe",
    ) -> tuple[bool, bool, str, str, str, str]:
        del tmp_dir
        if filename != Path(self.filename).name:
            raise GeakArenaError("GEAK candidate changed the Arena filename")
        record = self.evaluator.evaluate({self.filename: code})
        self._last = record
        raw = record.raw
        compiled = bool(raw.get("pass_compilation"))
        correct = bool(raw.get("pass_correctness"))
        error = record.log_excerpt or "None"
        if correct and save_scripts:
            output = self._bounded_dir(exe_dir, "correctness output") / filename
            output.write_text(code, encoding="utf-8")
        return compiled, correct, "", error, "", error

    def run_perf_evaluation(self, exec_folder: str,
                            gen_perf_folder: str) -> dict[str, dict[str, float]]:
        self._bounded_dir(exec_folder, "performance input")
        self._bounded_dir(gen_perf_folder, "performance output")
        if self._last is None or not self._last.passed:
            return {}
        speedup = self._last.speedup or 0.0
        # OptimAgent_ROCm sorts ``ms`` ascending but deliberately selects the
        # final element and describes it as speedup. Supplying higher-is-better
        # Arena speedup here normalizes that upstream naming/direction defect.
        return {Path(self.filename).name: {
            "ms": speedup,
            "efficiency": speedup,
        }}

    def write_file(self, file_path: str, start_idx: int = 0,
                   datalen: int | None = None) -> None:
        del start_idx, datalen
        output = Path(file_path).resolve()
        if not output.is_relative_to(self.runtime_root):
            raise GeakArenaError("GEAK output escapes its safe runtime root")
        output.parent.mkdir(parents=True, exist_ok=True)
        row = self.problem_states[0]
        output.write_text(json.dumps({
            "instruction": row.instruction, "label": row.label,
            "file": row.filename, "target_kernel_name": row.target_kernel_name,
            "predict": row.solution or "",
        }, sort_keys=True) + "\n", encoding="utf-8")


def _is_geak_top_level_module(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.")
               for prefix in _GEAK_TOP_LEVEL_PACKAGES)


class _GeakSourceFinder(importlib.abc.MetaPathFinder):
    """Resolve GEAK's PEP-420 packages before Arena's regular ``agents``."""

    def __init__(self, source_root: Path):
        self.source_root = (source_root / "src").resolve()

    def find_spec(self, fullname: str, path: Any = None,
                  target: Any = None) -> Any:
        del path, target
        if not _is_geak_top_level_module(fullname):
            return None
        relative = Path(*fullname.split("."))
        package = self.source_root / relative
        module = package.with_suffix(".py")
        if module.is_file():
            return importlib.util.spec_from_file_location(fullname, module)
        if not package.is_dir():
            return None
        initializer = package / "__init__.py"
        if initializer.is_file():
            return importlib.util.spec_from_file_location(
                fullname, initializer,
                submodule_search_locations=[str(package)])
        spec = importlib.machinery.ModuleSpec(
            fullname, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package)]
        return spec


def _import_optim_agent_isolated(source_root: Path) -> Any:
    """Load GEAK's generic packages without poisoning Arena's namespaces.

    Both vendors publish a top-level package named ``agents``.  The controller
    process imports AgentKernelArena first, so a normal GEAK import resolves the
    already-cached Arena package.  Temporarily swap only GEAK's declared generic
    package families, retain the loaded class graph, then restore the exact
    pre-existing module objects before any evaluation begins.
    """
    source_path = str((source_root / "src").resolve())
    saved = {
        name: module for name, module in tuple(sys.modules.items())
        if _is_geak_top_level_module(name)
    }
    for name in saved:
        sys.modules.pop(name, None)
    finder = _GeakSourceFinder(source_root)
    sys.meta_path.insert(0, finder)
    sys.path.insert(0, source_path)
    try:
        from agents.OptimAgent_ROCm import OptimAgent  # type: ignore[import-not-found]
    except ImportError as exc:
        raise GeakArenaError("cannot import pinned GEAK-v1 OptimAgent_ROCm") from exc
    finally:
        for name in tuple(sys.modules):
            if _is_geak_top_level_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        try:
            sys.path.remove(source_path)
        except ValueError:
            pass
    return OptimAgent


def _load_upstream(source_root: Path) -> Any:
    arena_adapter.inspect_vendor_source(source_root, SOURCE_PIN)
    return _import_optim_agent_isolated(source_root)


def run_controller(
    *, prompt: str, workspace: str | Path, arena_root: str | Path,
    source_root: str | Path, budget: common.ControllerBudget,
    max_iterations: int, ancestor_num: int = 2,
    model_factory: Callable[..., Any] = common.CodexTextModel,
    evaluator_factory: Callable[..., Any] = common.ArenaWorkspaceEvaluator,
    upstream_loader: Callable[[Path], Any] = _load_upstream,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise GeakArenaError("Arena prompt must be non-empty")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or not 1 <= max_iterations <= 256):
        raise GeakArenaError("max_iterations must be in [1, 256]")
    if (isinstance(ancestor_num, bool) or not isinstance(ancestor_num, int)
            or not 1 <= ancestor_num <= 16):
        raise GeakArenaError("ancestor_num must be in [1, 16]")
    root = common.workspace_root(workspace)
    source = Path(source_root).resolve()
    model = model_factory(workspace=root, budget=budget)
    evaluator = evaluator_factory(
        workspace=root, arena_root=Path(arena_root).resolve())
    runtime_root = root / common.ARTIFACT_DIRNAME / "geak-runtime"
    dataset = GeakArenaDataset(
        prompt=prompt, evaluator=evaluator, runtime_root=runtime_root)
    optim_agent = upstream_loader(source)
    corpus = source / "src/dataloaders/TB_eval/train_crawl.json"
    agent = optim_agent(
        model=GeakTextModel(model), dataset=dataset, corpus_path=str(corpus),
        max_perf_debug_num=5, mem_file=None)
    output_path = runtime_root / "output.jsonl"
    previous_cwd = Path.cwd()
    stop_reason = "upstream_complete"
    try:
        os.chdir(runtime_root)
        agent.run(
            output_path=str(output_path), multi_thread=False, datalen=1,
            iteration_num=max_iterations, temperature=0,
            ancestor_num=ancestor_num, start_idx=0, gpu_id=0, start_iter=0)
    except common.ControllerBudgetExpired:
        stop_reason = "campaign_checkpoint"
    finally:
        os.chdir(previous_cwd)
    evaluator.materialize_best()
    return common.build_controller_receipt(
        controller_id=CONTROLLER_ID, source_root=source,
        source_commit=SOURCE_COMMIT, entrypoint=source / UPSTREAM_ENTRYPOINT,
        model=model, evaluator=evaluator, stop_reason=stop_reason,
        extra={
            "upstream_callable": "OptimAgent.run",
            "max_iterations": max_iterations,
            "ancestor_num": ancestor_num,
            "performance_direction_normalization": (
                "Arena higher-is-better speedup supplied to upstream field named ms; "
                "upstream selects the largest value"),
            "safe_runtime_root": str(runtime_root),
        },
    )


def campaign_argv(executable: str | None = None) -> tuple[str, ...]:
    python = str(RUNTIME_PYTHON if executable is None else executable)
    source_root = Path(os.environ.get(
        "AUTOKERNEL_ARENA_CONTROLLER_ROOT",
        str(DEFAULT_SOURCE_ROOT.parent))) / DEFAULT_SOURCE_ROOT.name
    return (
        python, "-m", EXECUTABLE_MODULE,
        "--model", common.MODEL_ID, "--effort", common.MODEL_EFFORT,
        "--checkpoint-hours", "32", "--timeout-seconds", "115200",
        "--max-iterations", "64", "--ancestor-num", "2",
        "--workspace", ".",
        "--arena-root", os.environ.get(
            "AUTOKERNEL_AGENT_KERNEL_ARENA_ROOT",
            "/mnt/raid0/llm/autokernel/vendor/agent-kernel-arena"),
        "--source-root", str(source_root),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--checkpoint-hours", required=True, type=float)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--ancestor-num", required=True, type=int)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--arena-root", required=True)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args(argv)
    if args.model != common.MODEL_ID or args.effort != common.MODEL_EFFORT:
        parser.error("model and effort must match the fixed campaign pins")
    receipt = run_controller(
        prompt=sys.stdin.read(), workspace=args.workspace,
        arena_root=args.arena_root, source_root=args.source_root,
        budget=common.ControllerBudget(
            args.checkpoint_hours, args.timeout_seconds),
        max_iterations=args.max_iterations, ancestor_num=args.ancestor_num)
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "CONTROLLER_ID", "DEFAULT_SOURCE_ROOT", "ENTRYPOINT_RELATIVE",
    "EXECUTABLE_MODULE", "GeakArenaDataset", "GeakArenaError",
    "GeakTextModel", "PINNED_MODEL_IDS", "REQUIRED_CLIS", "RUNTIME_PYTHON",
    "SOURCE_COMMIT", "SOURCE_PIN", "UPSTREAM_ENTRYPOINT", "campaign_argv",
    "run_controller",
]


if __name__ == "__main__":
    raise SystemExit(main())
