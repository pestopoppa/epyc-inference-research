#!/usr/bin/env python3
"""Licensed K-Search world-model/tree controller port for AgentKernelArena."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Callable, Sequence

from . import arena_adapter, arena_upstream_common as common


CONTROLLER_ID = "k_search"
SOURCE_COMMIT = "53c8fab9a5e8fab2c86610d24fbec5067f90e115"
SOURCE_PIN = arena_adapter.VendorPin(
    name="K-Search", commit=SOURCE_COMMIT, license_path="LICENSE",
    required_paths=(
        "k_search/kernel_generators/kernel_generator_world_model.py",
        "k_search/kernel_generators/world_model_manager.py",
        "k_search/tasks/task_base.py",
    ),
)
DEFAULT_SOURCE_ROOT = Path(
    "/mnt/raid0/llm/autokernel/vendor/arena-controllers/k-search")
ENTRYPOINT_RELATIVE = (
    "scripts/kernel_rnd/autokernel/controller/k_search_arena.py")
EXECUTABLE_MODULE = (
    "scripts.kernel_rnd.autokernel.controller.k_search_arena")
UPSTREAM_ENTRYPOINT = "k_search/kernel_generators/kernel_generator_world_model.py"
PINNED_MODEL_IDS = common.PINNED_MODEL_IDS
REQUIRED_CLIS = common.REQUIRED_CLIS


class KSearchArenaError(common.UpstreamControllerError):
    """K-Search cannot run faithfully against the governed Arena seam."""


class KSearchArenaTask:
    """The exact K-Search Task protocol backed by AgentKernelArena evaluation."""

    def __init__(self, *, prompt: str, evaluator: common.ArenaWorkspaceEvaluator,
                 types_module: Any):
        self.name = evaluator.workspace.name
        self._prompt = evaluator.definition(prompt)
        self._evaluator = evaluator
        self._types = types_module
        self._last: common.EvaluationRecord | None = None

    def get_definition_text(self, language: str | None = None) -> str:
        del language
        return self._prompt

    def get_solution(self, solution_name: str) -> None:
        del solution_name
        return None

    def make_solution_from_generated_code(
        self, *, cleaned_code: Any, raw_code: Any, round_num: int,
        model_name: str, target_gpu: str, language: str,
    ) -> Any:
        del target_gpu
        if len(self._evaluator.source_paths) != 1:
            raise KSearchArenaError(
                "the current K-Search port requires one Arena source file")
        content = cleaned_code if isinstance(cleaned_code, str) else raw_code
        if not isinstance(content, str) or not content.strip():
            raise KSearchArenaError("K-Search generated no complete source file")
        relative = self._evaluator.source_paths[0]
        return self._types.Solution(
            name=f"k_search_{self.name}_r{round_num}", definition=self.name,
            author=model_name,
            spec=self._types.BuildSpec(
                language=self._types.SupportedLanguages.TRITON,
                target_hardware=[arena_adapter.TARGET_GPU_MODEL],
                entry_point=f"{relative}::{self._target_symbol()}",
            ),
            sources=[self._types.SourceFile(path=relative, content=content)],
            description=(f"K-Search {language} candidate for physical "
                         f"{arena_adapter.TARGET_GFX_ARCH}"),
        )

    def _target_symbol(self) -> str:
        targets = self._evaluator.config.get("target_kernel_functions")
        if not isinstance(targets, list) or len(targets) != 1 or not targets[0]:
            raise KSearchArenaError(
                "the current K-Search port requires one target kernel symbol")
        return str(targets[0])

    def run_benchmark(
        self, *, solution: Any, config: Any = None,
        dump_traces: bool = False, round_num: int | None = None,
    ) -> Any:
        del config, dump_traces, round_num
        files = {str(row.path): str(row.content) for row in solution.sources}
        record = self._evaluator.evaluate(files)
        self._last = record
        return self._eval_result(record)

    def _eval_result(self, record: common.EvaluationRecord) -> Any:
        return self._types.EvalResult(
            status="passed" if record.passed else "failed",
            latency_ms=record.latency_ms,
            reference_latency_ms=None,
            mean_vs_baseline_factor=record.speedup,
            speedup_factor=record.speedup,
            log_excerpt=record.log_excerpt,
            metrics={"score": record.score, "score_name": "arena_speedup"},
        )

    def seed_eval_for_base_solution(
        self, *, base_solution: Any, config: Any = None,
    ) -> Any:
        del config
        return self.run_benchmark(solution=base_solution)

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        del language
        return str(raw or "")

    def get_config_for_logging(self) -> dict[str, Any]:
        return {
            "backend": "AgentKernelArena", "target_gpu": "MI210",
            "target_gfx_arch": "gfx90a",
            "source_paths": list(self._evaluator.source_paths),
        }

    def get_baseline_targets_text(self) -> str:
        return "- AgentKernelArena starting-state score: 1.0x"

    def get_per_task_requirement_text(
        self, language: str, target_gpu: str, phase: str,
    ) -> str:
        del language, target_gpu
        return (
            f"Phase {phase}: preserve the complete Arena source file and target "
            "physical MI210 gfx90a wave64. Performance is admitted only from the "
            "centralized Arena evaluator.")

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return "" if self._last is None else self._last.log_excerpt

    def has_last_round_feedback_trace(self) -> bool:
        return bool(self._last and self._last.log_excerpt)

    def get_last_round_passed_count(self) -> int:
        return int(bool(self._last and self._last.passed))

    def get_last_round_total_workloads(self) -> int:
        return int(self._last is not None)

    def run_final_evaluation(
        self, *, solutions: list[Any], config: Any = None,
        dump_traces: bool = False, workload_limit: int | None = None,
    ) -> dict[str, Any]:
        del config, dump_traces, workload_limit
        rows = [self.run_benchmark(solution=solution).to_dict()
                for solution in solutions]
        return {"results": rows}


def _load_upstream(source_root: Path, model: common.CodexTextModel) -> tuple[Any, Any]:
    arena_adapter.inspect_vendor_source(source_root, SOURCE_PIN)
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: model.openai_compat
    sys.modules["openai"] = fake_openai
    sys.path.insert(0, str(source_root))
    try:
        from k_search.kernel_generators.kernel_generator_world_model import (  # type: ignore[import-not-found]
            WorldModelKernelGeneratorWithBaseline,
        )
        from k_search.tasks import task_base  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KSearchArenaError("cannot import the pinned K-Search controller") from exc
    return WorldModelKernelGeneratorWithBaseline, task_base


def run_controller(
    *, prompt: str, workspace: str | Path, arena_root: str | Path,
    source_root: str | Path, budget: common.ControllerBudget, max_rounds: int,
    model_factory: Callable[..., Any] = common.CodexTextModel,
    evaluator_factory: Callable[..., Any] = common.ArenaWorkspaceEvaluator,
    upstream_loader: Callable[..., tuple[Any, Any]] = _load_upstream,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise KSearchArenaError("Arena prompt must be non-empty")
    if (isinstance(max_rounds, bool) or not isinstance(max_rounds, int)
            or not 1 <= max_rounds <= 256):
        raise KSearchArenaError("max_rounds must be in [1, 256]")
    root = common.workspace_root(workspace)
    source = Path(source_root).resolve()
    model = model_factory(workspace=root, budget=budget)
    evaluator = evaluator_factory(
        workspace=root, arena_root=Path(arena_root).resolve())
    generator_type, types_module = upstream_loader(source, model)
    task = KSearchArenaTask(
        prompt=prompt, evaluator=evaluator, types_module=types_module)
    generator = generator_type(
        model_name=common.MODEL_ID, language="triton",
        target_gpu="AMD Instinct MI210 gfx90a wave64", api_key="not-used",
        reasoning_effort=common.MODEL_EFFORT,
        artifacts_dir=str(model.artifact_root / "k-search-artifacts"),
    )
    stop_reason = "upstream_complete"
    result = None
    try:
        result = generator.generate(
            task=task, max_opt_rounds=max_rounds,
            wm_stagnation_window=min(5, max_rounds),
            num_debug_and_improve_rounds=min(5, max_rounds),
        )
    except common.ControllerBudgetExpired:
        stop_reason = "campaign_checkpoint"
    evaluator.materialize_best()
    entrypoint = source / UPSTREAM_ENTRYPOINT
    return common.build_controller_receipt(
        controller_id=CONTROLLER_ID, source_root=source,
        source_commit=SOURCE_COMMIT, entrypoint=entrypoint, model=model,
        evaluator=evaluator, stop_reason=stop_reason,
        extra={
            "upstream_callable": (
                "WorldModelKernelGeneratorWithBaseline.generate"),
            "max_rounds": max_rounds,
            "returned_solution": bool(result is not None),
        },
    )


def campaign_argv(executable: str = "python3") -> tuple[str, ...]:
    source_root = Path(os.environ.get(
        "AUTOKERNEL_ARENA_CONTROLLER_ROOT",
        str(DEFAULT_SOURCE_ROOT.parent))) / DEFAULT_SOURCE_ROOT.name
    return (
        executable, "-m", EXECUTABLE_MODULE,
        "--model", common.MODEL_ID, "--effort", common.MODEL_EFFORT,
        "--checkpoint-hours", "32", "--timeout-seconds", "115200",
        "--max-rounds", "64", "--workspace", ".",
        "--arena-root", os.environ.get(
            "AUTOKERNEL_AGENT_KERNEL_ARENA_ROOT",
            "/mnt/raid0/llm/tmp/inf03-vendor-inspect-UqTLqw/AgentKernelArena"),
        "--source-root", str(source_root),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--checkpoint-hours", required=True, type=float)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--max-rounds", required=True, type=int)
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
        max_rounds=args.max_rounds,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "CONTROLLER_ID", "DEFAULT_SOURCE_ROOT", "ENTRYPOINT_RELATIVE",
    "EXECUTABLE_MODULE", "KSearchArenaError", "KSearchArenaTask",
    "PINNED_MODEL_IDS", "REQUIRED_CLIS", "SOURCE_COMMIT", "SOURCE_PIN",
    "UPSTREAM_ENTRYPOINT", "campaign_argv", "run_controller",
]


if __name__ == "__main__":
    raise SystemExit(main())
