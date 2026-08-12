#!/usr/bin/env python3
"""Paper-faithful EvoEngineer-Full controller for AgentKernelArena.

This module admits the exact MIT-licensed, paper-era EvoToolkit source and
defines the two host-specific bridges its unchanged ``EvoEngineer.run`` loop
will consume: a text model and a one-file AgentKernelArena task/interface.

Every candidate is evaluated only through the parent-owned broker behind
``ArenaWorkspaceEvaluator.evaluate(files)``.  Importing this module performs no
model, evaluator, compiler, or GPU work; the campaign runner independently
enforces controller and evaluator process isolation before launching it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from . import arena_adapter
from . import arena_upstream_common as common


CONTROLLER_ID = "evoengineer"
VARIANT = "EvoEngineer-Full"
SOURCE_REPOSITORY = "https://github.com/pgg3/evotoolkit"
SOURCE_RELEASE = "data-v1.0.0"
SOURCE_COMMIT = "1649715a975b9022c84b5279c88aaef0b73b28dc"
SOURCE_RELEASE_ASSET = "rtx4090_cu12_4_py311_torch_2_4_0.json"
SOURCE_RELEASE_ASSET_SHA256 = (
    "acd55a66b4e26cc657b107bf13da0f37efd486c4910291839a0299c5335fdc77")
UPSTREAM_ENTRYPOINT = "src/evotoolkit/evo_method/evoengineer/evoengineer.py"
UPSTREAM_CONFIG = "src/evotoolkit/evo_method/evoengineer/run_config.py"
UPSTREAM_INTERFACE = (
    "src/evotoolkit/core/method_interface/evoengineer_interface.py")
UPSTREAM_FULL_CUDA_INTERFACE = (
    "src/evotoolkit/task/cuda_engineering/method_interface/"
    "evoengineer_full_interface.py")
UPSTREAM_SOLUTION = "src/evotoolkit/core/solution.py"
EXPECTED_SOURCE_SHA256: Mapping[str, str] = {
    "LICENSE": "3a18133891b736252655b83391edfef51bd52aa317198fcc4374eb5f16e99de3",
    UPSTREAM_ENTRYPOINT:
        "28c56fbeb8663c9084734c8682dea39df4a539e1680eee782a8553046963e50d",
    UPSTREAM_CONFIG:
        "d85062ed6bca45323c04e9227032627a0abc9d0d88b90ed529035be949b23e3d",
    UPSTREAM_INTERFACE:
        "79a8ecc978154c0e66b4f9efd4d9080040746bf62f901fdab508447e68e358da",
    UPSTREAM_FULL_CUDA_INTERFACE:
        "79f553b1b7cc4bb6a8e583eeb0976840881d2f1d01ccfb739e67126ce5544d04",
    UPSTREAM_SOLUTION:
        "c77161f880528b4acb9dbfd778dbc1fcdb05ae2029f4f6632c1213c3d9c59bac",
}
SOURCE_PIN = arena_adapter.VendorPin(
    name="EvoEngineer (EvoToolkit paper release)", commit=SOURCE_COMMIT,
    license_path="LICENSE",
    required_paths=tuple(path for path in EXPECTED_SOURCE_SHA256
                         if path != "LICENSE"),
)
DEFAULT_SOURCE_ROOT = Path(
    "/mnt/raid0/llm/autokernel/vendor/arena-controllers/evoengineer")
RUNTIME_PYTHON = Path(
    "/mnt/raid0/llm/tools/geak-v1-rocm62-py312/bin/python")
ENTRYPOINT_RELATIVE = (
    "scripts/kernel_rnd/autokernel/controller/evoengineer_arena.py")
EXECUTABLE_MODULE = "scripts.kernel_rnd.autokernel.controller.evoengineer_arena"
ADAPTER_KIND = "evoengineer_full_arena_v1"
# Kept until the campaign manifest is advanced in its self-pinning follow-up
# commit; the current manifest must remain loadable at every intermediate tip.
PENDING_ADAPTER_KIND = "evoengineer_full_arena_pending_v1"
PINNED_MODEL_IDS = common.PINNED_MODEL_IDS
REQUIRED_CLIS = common.REQUIRED_CLIS

# The named paper arm is explicit.  Generic "EvoEngineer" would be ambiguous
# among the upstream Free, Insight, and Full interfaces.
MAX_GENERATIONS = 10
MAX_SAMPLE_NUMS = 45
POPULATION_SIZE = 4
NUM_SAMPLERS = 4
NUM_EVALUATORS = 4
INIT_OPERATORS = (("init", 0),)
OFFSPRING_OPERATORS = (("crossover", 2), ("mutation", 1))

class EvoEngineerArenaError(common.UpstreamControllerError):
    """The EvoEngineer source or policy seam is not exactly admissible."""


@dataclass(frozen=True)
class UpstreamTypes:
    """Types imported from the exact source checkout after admission."""

    controller_type: Any
    config_type: Any
    operator_type: Any
    solution_type: Any
    evaluation_result_type: Any
    full_interface_type: Any | None = None


def inspect_source(source_root: str | Path) -> dict[str, Any]:
    """Verify commit, cleanliness, licence, and policy-bearing source bytes."""
    try:
        receipt = arena_adapter.inspect_vendor_source(source_root, SOURCE_PIN)
    except arena_adapter.ArenaAdapterError as exc:
        raise EvoEngineerArenaError(str(exc)) from exc
    observed = receipt["required_file_sha256"]
    failures = [
        f"{path}: expected {expected}, observed {observed.get(path)}"
        for path, expected in EXPECTED_SOURCE_SHA256.items()
        if observed.get(path) != expected
    ]
    if failures:
        raise EvoEngineerArenaError(
            "EvoEngineer paper-source digest mismatch: " + "; ".join(failures))
    return {
        **receipt,
        "repository": SOURCE_REPOSITORY,
        "release": SOURCE_RELEASE,
        "release_asset": SOURCE_RELEASE_ASSET,
        "release_asset_sha256": SOURCE_RELEASE_ASSET_SHA256,
        "variant": VARIANT,
        "paper_policy": policy_identity(),
    }


def load_upstream(source_root: str | Path) -> UpstreamTypes:
    """Import only after the exact historical source passes admission."""
    source = Path(source_root).resolve()
    inspect_source(source)
    package_root = source / "src"
    sys.path.insert(0, str(package_root))
    try:
        from evotoolkit.core import EvaluationResult, Operator, Solution
        from evotoolkit.evo_method.evoengineer.evoengineer import EvoEngineer
        from evotoolkit.evo_method.evoengineer.run_config import EvoEngineerConfig
    except ImportError as exc:
        raise EvoEngineerArenaError(
            "cannot import the admitted EvoEngineer paper source") from exc
    return UpstreamTypes(
        controller_type=EvoEngineer, config_type=EvoEngineerConfig,
        operator_type=Operator, solution_type=Solution,
        evaluation_result_type=EvaluationResult)


def policy_identity() -> dict[str, Any]:
    """Return the fully declared named arm; this is not an execution receipt."""
    return {
        "variant": VARIANT,
        "upstream_callable": "EvoEngineer.run",
        "max_generations": MAX_GENERATIONS,
        "max_sample_nums": MAX_SAMPLE_NUMS,
        "population_size": POPULATION_SIZE,
        "num_samplers": NUM_SAMPLERS,
        "num_evaluators": NUM_EVALUATORS,
        "init_operators": [list(row) for row in INIT_OPERATORS],
        "offspring_operators": [list(row) for row in OFFSPRING_OPERATORS],
        "selection": "upstream rank-probability selection",
        "population_management": "upstream best-valid elite trim",
        "insight_sampling": "upstream random sample of up to three thoughts",
        "score_direction": "higher_is_better",
    }


class EvoEngineerTextModel:
    """Upstream ``get_response`` protocol over the governed text model."""

    def __init__(self, model: Any):
        self.model = model

    def get_response(self, messages: Sequence[Mapping[str, Any]]) -> tuple[str, dict]:
        if not isinstance(messages, Sequence) or not messages:
            raise EvoEngineerArenaError("EvoEngineer supplied no model messages")
        prompt = "\n\n".join(
            f"{row.get('role', 'user')}: {row.get('content', '')}"
            for row in messages if isinstance(row, Mapping))
        if not prompt.strip():
            raise EvoEngineerArenaError("EvoEngineer supplied an empty model prompt")
        response = self.model.call(prompt)
        return response, {
            "model": common.MODEL_ID,
            "effort": common.MODEL_EFFORT,
            "transport": "governed_codex_text_model",
        }


class EvoEngineerArenaTask:
    """One-file, higher-is-better Arena task consumed by EvoEngineer."""

    def __init__(self, *, prompt: str, evaluator: Any, types: UpstreamTypes):
        if not isinstance(prompt, str) or not prompt.strip():
            raise EvoEngineerArenaError("Arena prompt must be non-empty")
        if len(evaluator.source_paths) != 1:
            raise EvoEngineerArenaError(
                "the pending EvoEngineer port requires one Arena source file")
        self.evaluator = evaluator
        self.types = types
        self.source_path = evaluator.source_paths[0]
        self.prompt = evaluator.definition(prompt)
        self.task_info = {
            "task_type": "AgentKernelArena",
            "target_gpu": arena_adapter.TARGET_GPU_MODEL,
            "target_gfx_arch": arena_adapter.TARGET_GFX_ARCH,
            "source_path": self.source_path,
        }

    def get_base_task_description(self) -> str:
        return self.prompt

    def make_init_sol_wo_other_info(self) -> Any:
        source = (self.evaluator.workspace / self.source_path).read_text(
            encoding="utf-8")
        solution = self.types.solution_type(source)
        solution.evaluation_res = self.evaluate_code(source)
        return solution

    def evaluate_code(self, candidate_code: str) -> Any:
        if not isinstance(candidate_code, str) or not candidate_code.strip():
            return self.types.evaluation_result_type(
                False, None, {"error": "empty complete-source candidate"})
        record = self.evaluator.evaluate({self.source_path: candidate_code})
        score = record.speedup if record.passed else None
        return self.types.evaluation_result_type(
            bool(record.passed), score, {
                "arena_score": record.score,
                "speedup": record.speedup,
                "latency_ms": record.latency_ms,
                "prof_string": record.log_excerpt,
                "evidence_authority": "AgentKernelArena",
            })


class EvoEngineerFullArenaInterface:
    """AMD task/prompt boundary retaining EvoEngineer-Full policy operators."""

    valid_require = 2

    def __init__(self, *, task: EvoEngineerArenaTask, types: UpstreamTypes):
        self.task = task
        self.types = types

    def make_init_sol(self) -> Any:
        solution = self.task.make_init_sol_wo_other_info()
        solution.other_info = {"name": "Baseline", "thought": "Baseline"}
        return solution

    def get_init_operators(self) -> list[Any]:
        return [self.types.operator_type(name, size)
                for name, size in INIT_OPERATORS]

    def get_offspring_operators(self) -> list[Any]:
        return [self.types.operator_type(name, size)
                for name, size in OFFSPRING_OPERATORS]

    @staticmethod
    def _solution_block(label: str, solution: Any) -> str:
        info = solution.other_info or {}
        result = solution.evaluation_res
        return (
            f"### {label}\n"
            f"Name: {info.get('name', 'unnamed')}\n"
            f"Arena speedup: {getattr(result, 'score', None)}\n"
            f"Approach: {info.get('thought', '')}\n"
            f"```triton\n{solution.sol_string}\n```")

    def get_operator_prompt(
        self, operator_name: str, selected_individuals: Sequence[Any],
        current_best_sol: Any, random_thoughts: Sequence[str], **kwargs: Any,
    ) -> list[dict[str, str]]:
        del kwargs
        expected = dict((*INIT_OPERATORS, *OFFSPRING_OPERATORS))
        if operator_name not in expected:
            raise EvoEngineerArenaError(f"unknown operator: {operator_name}")
        if len(selected_individuals) < expected[operator_name]:
            raise EvoEngineerArenaError(
                f"{operator_name} requires {expected[operator_name]} selected parents")
        if current_best_sol is None:
            current_best_sol = self.make_init_sol()
        sections = [
            "# AMD GFX90A KERNEL OPTIMIZATION TASK",
            self.task.get_base_task_description(),
            self._solution_block("CURRENT BEST", current_best_sol),
        ]
        if selected_individuals:
            sections.append("\n\n".join(
                self._solution_block(f"PARENT {index}", solution)
                for index, solution in enumerate(selected_individuals, 1)))
        if random_thoughts:
            sections.append("## SEARCH INSIGHTS\n" + "\n".join(
                f"- {thought}" for thought in random_thoughts[:3]))
        sections.append(
            f"## {operator_name.upper()}\n"
            "Return one complete replacement for the source file in a ```triton "
            "block. Target AMD Instinct MI210 gfx90a wave64. Preserve imports, "
            "public signatures, and the embedded test harness. Do not emit a "
            "patch or partial function. AgentKernelArena is the only authority "
            "for compilation, correctness, and timing.\n\n"
            "Response format:\nname: descriptive_name\n"
            "code:\n```triton\n[complete source file]\n```\n"
            "thought: optimization rationale")
        return [{"role": "user", "content": "\n\n".join(sections)}]

    def parse_response(self, response: str) -> Any:
        """Retain the upstream Full parser's four ordered fallbacks."""
        content = response.strip() if isinstance(response, str) else ""
        if not content:
            return self.types.solution_type("")

        def any_code_block(text: str) -> str:
            language = re.search(
                r"```(?:triton|cpp|c\+\+|cuda|python)\n(.*?)```",
                text, re.DOTALL | re.IGNORECASE)
            if language:
                return language.group(1).strip()
            generic = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
            if generic:
                return generic.group(1).strip()
            section = re.search(
                r"code:\s*\n*(.*?)(?=\n(?:thought|$))", text,
                re.DOTALL | re.IGNORECASE)
            if section:
                code = re.sub(
                    r"^```[^\n]*\n?", "", section.group(1).strip())
                return re.sub(r"\n?```\s*$", "", code).strip()
            return ""

        name_match = re.search(
            r"^name:\s*([^\n\r]+?)(?:\n|\r|$)", content,
            re.MULTILINE | re.IGNORECASE)
        code_match = re.search(
            r"code:\s*\n*```(?:triton|cpp|c\+\+|cuda|python)?\n(.*?)```",
            content, re.DOTALL | re.IGNORECASE)
        thought_match = re.search(
            r"thought:\s*(.*?)$", content, re.DOTALL | re.IGNORECASE)
        if code_match:
            return self.types.solution_type(
                code_match.group(1).strip(), other_info={
                    "name": (name_match.group(1).strip()
                             if name_match else "extracted"),
                    "thought": (thought_match.group(1).strip()
                                if thought_match else ""),
                })
        flexible_name = re.search(
            r"(?:name|Name|NAME)\s*:?\s*([^\n\r]+)", content,
            re.IGNORECASE)
        flexible_thought = re.search(
            r"(?:thought|Thought|THOUGHT)\s*:?\s*(.*?)(?=\n(?:name|code)|$)",
            content, re.DOTALL | re.IGNORECASE)
        flexible_code = any_code_block(content)
        if flexible_code:
            return self.types.solution_type(
                flexible_code, other_info={
                    "name": (flexible_name.group(1).strip()
                             if flexible_name else "extracted"),
                    "thought": (flexible_thought.group(1).strip()
                                if flexible_thought else ""),
                })
        fallback_code = any_code_block(content)
        if fallback_code:
            return self.types.solution_type(
                fallback_code, other_info={
                    "name": "extracted", "thought": "Fallback parsing"})
        return self.types.solution_type(
            content, other_info={"name": "raw", "thought": "Failed to parse"})


def build_controller(
    *, prompt: str, evaluator: Any, model: Any, output_path: str | Path,
    types: UpstreamTypes,
) -> Any:
    """Assemble the exact upstream controller without running its search loop."""
    output = Path(output_path).resolve()
    if not output.is_dir() or output.is_symlink():
        raise EvoEngineerArenaError(
            "EvoEngineer output_path must be an existing non-symlink directory")
    task = EvoEngineerArenaTask(prompt=prompt, evaluator=evaluator, types=types)
    interface = EvoEngineerFullArenaInterface(task=task, types=types)
    config = types.config_type(
        interface=interface, output_path=str(output),
        running_llm=EvoEngineerTextModel(model), verbose=False,
        max_generations=MAX_GENERATIONS, max_sample_nums=MAX_SAMPLE_NUMS,
        pop_size=POPULATION_SIZE, num_samplers=NUM_SAMPLERS,
        num_evaluators=NUM_EVALUATORS)
    return types.controller_type(config)


def run_controller(
    *, prompt: str, workspace: str | Path, arena_root: str | Path,
    source_root: str | Path, budget: common.ControllerBudget,
    model_factory: Callable[..., Any] = common.CodexTextModel,
    evaluator_factory: Callable[..., Any] = common.ArenaWorkspaceEvaluator,
    upstream_loader: Callable[[str | Path], UpstreamTypes] = load_upstream,
) -> dict[str, Any]:
    """Run the exact upstream ``EvoEngineer.run`` policy over broker feedback."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise EvoEngineerArenaError("Arena prompt must be non-empty")
    root = common.workspace_root(workspace)
    source = Path(source_root).resolve()
    source_receipt = inspect_source(source)
    model = model_factory(workspace=root, budget=budget)
    evaluator = evaluator_factory(
        workspace=root, arena_root=Path(arena_root).resolve(),
        source_paths=common.declared_arena_source_paths())
    output = root / common.ARTIFACT_DIRNAME / "evoengineer-runtime"
    output.mkdir(parents=True, exist_ok=False)
    controller = build_controller(
        prompt=prompt, evaluator=evaluator, model=model, output_path=output,
        types=upstream_loader(source))
    stop_reason = "upstream_complete"
    try:
        controller.run()
    except common.ControllerBudgetExpired:
        stop_reason = "campaign_checkpoint"
    evaluator.materialize_best()
    state = getattr(controller, "run_state_dict", None)
    return common.build_controller_receipt(
        controller_id=CONTROLLER_ID, source_root=source,
        source_commit=SOURCE_COMMIT, entrypoint=source / UPSTREAM_ENTRYPOINT,
        model=model, evaluator=evaluator, stop_reason=stop_reason,
        extra={
            "upstream_callable": "EvoEngineer.run",
            "variant": VARIANT,
            "policy": policy_identity(),
            "source_admission": source_receipt,
            "generation": getattr(state, "generation", None),
            "sample_count": getattr(state, "tot_sample_nums", None),
            "safe_runtime_root": str(output),
        })


def campaign_argv(executable: str | None = None) -> tuple[str, ...]:
    python = str(RUNTIME_PYTHON if executable is None else executable)
    source_root = Path(os.environ.get(
        "AUTOKERNEL_ARENA_CONTROLLER_ROOT",
        str(DEFAULT_SOURCE_ROOT.parent))) / DEFAULT_SOURCE_ROOT.name
    return (
        python, "-m", EXECUTABLE_MODULE,
        "--model", common.MODEL_ID, "--effort", common.MODEL_EFFORT,
        "--checkpoint-hours", "32", "--timeout-seconds", "115200",
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
            args.checkpoint_hours, args.timeout_seconds))
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "ADAPTER_KIND", "CONTROLLER_ID", "DEFAULT_SOURCE_ROOT", "ENTRYPOINT_RELATIVE",
    "EXECUTABLE_MODULE",
    "EXPECTED_SOURCE_SHA256", "EvoEngineerArenaError",
    "EvoEngineerArenaTask", "EvoEngineerFullArenaInterface",
    "EvoEngineerTextModel", "INIT_OPERATORS", "MAX_GENERATIONS",
    "MAX_SAMPLE_NUMS", "NUM_EVALUATORS", "NUM_SAMPLERS",
    "OFFSPRING_OPERATORS", "PENDING_ADAPTER_KIND", "PINNED_MODEL_IDS", "POPULATION_SIZE",
    "REQUIRED_CLIS", "RUNTIME_PYTHON", "SOURCE_COMMIT", "SOURCE_PIN", "SOURCE_RELEASE",
    "SOURCE_RELEASE_ASSET_SHA256", "UPSTREAM_ENTRYPOINT", "UpstreamTypes",
    "VARIANT", "build_controller", "campaign_argv", "inspect_source",
    "load_upstream", "policy_identity", "run_controller",
]


if __name__ == "__main__":
    raise SystemExit(main())
