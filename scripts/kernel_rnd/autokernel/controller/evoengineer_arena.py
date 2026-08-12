#!/usr/bin/env python3
"""Paper-faithful EvoEngineer-Full policy seam for AgentKernelArena.

This module admits the exact MIT-licensed, paper-era EvoToolkit source and
defines the two host-specific bridges its unchanged ``EvoEngineer.run`` loop
will consume: a text model and a one-file AgentKernelArena task/interface.

It deliberately has no CLI or campaign launcher.  The shared Arena runtime
must first provide claim-scoped intermediate evaluation through
``ArenaWorkspaceEvaluator.evaluate(files)``.  Until then the campaign arm is
``missing`` and importing this module performs no model, evaluator, compiler,
or GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

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
    "/mnt/raid0/llm/autokernel/vendor/arena-controllers/evotoolkit")
ENTRYPOINT_RELATIVE = (
    "scripts/kernel_rnd/autokernel/controller/evoengineer_arena.py")
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

PENDING_RUNTIME_DEPENDENCIES = (
    "an exact clean vendor checkout at the admitted paper-era commit",
    "a parent-worker AF_UNIX evaluation broker behind "
    "ArenaWorkspaceEvaluator.evaluate(files)",
    "launcher-enforced controller device isolation with no controller-side "
    "vendor measure/evaluate path",
    "a hash-bound campaign launcher and receipt integration validated after "
    "the evaluation broker lands",
)


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
    full_interface_type: Any


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
        from evotoolkit.task.cuda_engineering.method_interface.evoengineer_full_interface import (
            EvoEngineerFullCudaInterface,
        )
    except ImportError as exc:
        raise EvoEngineerArenaError(
            "cannot import the admitted EvoEngineer paper source") from exc
    return UpstreamTypes(
        controller_type=EvoEngineer, config_type=EvoEngineerConfig,
        operator_type=Operator, solution_type=Solution,
        evaluation_result_type=EvaluationResult,
        full_interface_type=EvoEngineerFullCudaInterface)


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


def execution_refusal() -> None:
    """Fail closed until the shared broker and launcher are integrated."""
    raise EvoEngineerArenaError(
        "EvoEngineer is source-admitted but not executable: "
        + "; ".join(PENDING_RUNTIME_DEPENDENCIES))


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
        # Parsing is policy-bearing and target-independent, so retain the exact
        # upstream Full parser instead of reconstructing it in the AMD port.
        self._upstream_parser = object.__new__(types.full_interface_type)

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
        return self._upstream_parser.parse_response(response)


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


__all__ = [
    "CONTROLLER_ID", "DEFAULT_SOURCE_ROOT", "ENTRYPOINT_RELATIVE",
    "EXPECTED_SOURCE_SHA256", "EvoEngineerArenaError",
    "EvoEngineerArenaTask", "EvoEngineerFullArenaInterface",
    "EvoEngineerTextModel", "INIT_OPERATORS", "MAX_GENERATIONS",
    "MAX_SAMPLE_NUMS", "NUM_EVALUATORS", "NUM_SAMPLERS",
    "OFFSPRING_OPERATORS", "PENDING_ADAPTER_KIND",
    "PENDING_RUNTIME_DEPENDENCIES", "PINNED_MODEL_IDS", "POPULATION_SIZE",
    "REQUIRED_CLIS", "SOURCE_COMMIT", "SOURCE_PIN", "SOURCE_RELEASE",
    "SOURCE_RELEASE_ASSET_SHA256", "UPSTREAM_ENTRYPOINT", "UpstreamTypes",
    "VARIANT", "build_controller", "execution_refusal", "inspect_source",
    "load_upstream", "policy_identity",
]
