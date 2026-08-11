#!/usr/bin/env python3
"""Run pinned KernelFoundry MAP-Elites/QD-gradient on AgentKernelArena.

The pinned upstream ``Controller.run_single`` owns proposal branching, static
feature coordinates, island migration, elite selection, and gradient-informed
sampling.  This gfx90a port replaces only its model and evaluator dependencies:
GPT-5.6 Sol/high proposes complete Triton files and AgentKernelArena is the sole
authority for compilation, correctness, and timing.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Callable, Sequence

from . import arena_adapter
from . import arena_upstream_common as common

CONTROLLER_ID = "kernelfoundry"
SOURCE_COMMIT = "1c053e02383d12937f144923bcc1faa82fa7788f"
SOURCE_PIN = arena_adapter.VendorPin(
    name="KernelFoundry", commit=SOURCE_COMMIT, license_path="LICENSE",
    required_paths=(
        "kernelfoundry/algorithm/controller.py",
        "kernelfoundry/algorithm/evolve_database_optimization_aware.py",
        "kernelfoundry/algorithm/qd_gradient.py",
        "kernelfoundry/algorithm/schemas.py",
    ),
)
DEFAULT_SOURCE_ROOT = Path(
    "/mnt/raid0/llm/autokernel/vendor/arena-controllers/kernelfoundry")
RUNTIME_PYTHON = Path(
    "/mnt/raid0/llm/tools/geak-v1-rocm62-py312/bin/python")
ENTRYPOINT_RELATIVE = (
    "scripts/kernel_rnd/autokernel/controller/kernelfoundry_arena.py")
EXECUTABLE_MODULE = (
    "scripts.kernel_rnd.autokernel.controller.kernelfoundry_arena")
UPSTREAM_ENTRYPOINT = "kernelfoundry/algorithm/controller.py"
PINNED_MODEL_IDS = common.PINNED_MODEL_IDS
REQUIRED_CLIS = common.REQUIRED_CLIS

_MI210_SPECS = {
    "GPU Architecture": "AMD CDNA2 gfx90a",
    "GPU Memory": "64 GiB HBM2e",
    "Memory Bandwidth": "1.6 TB/s nominal",
    "Compute Units": 104,
    "Wavefront Size": 64,
    "LDS per Compute Unit": "64 KiB",
}


class KernelFoundryArenaError(common.UpstreamControllerError):
    """KernelFoundry cannot preserve its governed gfx90a port contract."""


@dataclass(frozen=True)
class _Upstream:
    hydra: Any
    omega_conf: Any
    controller_type: Any
    answer_processor_type: Any
    feedback_helper_type: Any
    prompt_constructor_type: Any
    eval_result_type: Any
    program_type: Any
    database_type: Any
    classifier_type: Any
    triton_pattern_dicts: tuple[dict, ...]
    setup_logging: Callable[..., Any]
    gpu_specs_module: Any


class KernelFoundryTextServer:
    """KernelFoundry inference-server protocol over the governed Codex model."""

    def __init__(self, model: common.CodexTextModel):
        self.model = model
        self.model_id = common.MODEL_ID
        self._lock = threading.Lock()
        self.expired = False

    def __call__(self, messages, return_model_info=False, **kwargs):
        del kwargs
        if not isinstance(messages, list) or not messages:
            raise KernelFoundryArenaError("KernelFoundry supplied no model messages")
        prompt = "\n\n".join(
            f"{row.get('role', 'user')}: {row.get('content', '')}"
            for row in messages if isinstance(row, dict))
        try:
            # Upstream submits MAP-Elites branches concurrently.  The shared
            # transcript ordinal is intentionally serialized and deterministic.
            with self._lock:
                output = self.model.call(prompt)
        except common.ControllerBudgetExpired:
            self.expired = True
            raise
        outputs = [output]
        return (outputs, [self.model_id]) if return_model_info else outputs


class KernelFoundryArenaTask:
    """Minimal Task protocol consumed by the inherited Controller loop."""

    def __init__(self, *, prompt: str,
                 evaluator: common.ArenaWorkspaceEvaluator):
        if len(evaluator.source_paths) != 1:
            raise KernelFoundryArenaError(
                "the current KernelFoundry port requires one Arena source file")
        self.evaluator = evaluator
        self.source_path = evaluator.source_paths[0]
        source = (evaluator.workspace / self.source_path).read_text(
            encoding="utf-8")
        self.blocks = {
            "REFERENCE": {self.source_path: evaluator.definition(prompt)},
            "EVOLVE": {self.source_path: source},
            "USER_INSTRUCTIONS": {self.source_path: (
                "Return the complete source file in a ```triton block. Target "
                "AMD Instinct MI210 gfx90a wave64. Preserve imports, public "
                "signatures, and the embedded test harness. AgentKernelArena "
                "is the only correctness and timing authority.")},
        }
        self.config: Any = {}
        self.has_build_step = False
        self.has_reference_build_step = False
        self.test_result_reference = None
        self.build_result_reference = None


def _load_upstream(source_root: Path) -> _Upstream:
    arena_adapter.inspect_vendor_source(source_root, SOURCE_PIN)
    sys.path.insert(0, str(source_root))
    try:
        import hydra
        from omegaconf import OmegaConf
        from kernelfoundry.algorithm.answer_processor import AnswerProcessor
        from kernelfoundry.algorithm.controller import Controller, setup_logging
        from kernelfoundry.algorithm.prompts.feedback_llm import FeedbackHelper
        from kernelfoundry.algorithm.prompts.prompt_constructor import PromptConstructor
        from kernelfoundry.algorithm.schemas import EvalResult
        from kernelfoundry.algorithm.schemas import Program
        from kernelfoundry.algorithm.evolve_database_optimization_aware import (
            OptimizationAwareDatabase,
            OptimizationFeatureClassifier,
            TRITON_COMPUTE_OPT_PATTERNS,
            TRITON_MEMORY_OPT_PATTERNS,
            TRITON_PARALLELISM_OPT_PATTERNS,
        )
        from kernelfoundry.eval_pipeline.utils import gpu_specs
    except ImportError as exc:
        raise KernelFoundryArenaError("cannot import pinned KernelFoundry") from exc
    return _Upstream(
        hydra=hydra, omega_conf=OmegaConf, controller_type=Controller,
        answer_processor_type=AnswerProcessor,
        feedback_helper_type=FeedbackHelper,
        prompt_constructor_type=PromptConstructor,
        eval_result_type=EvalResult, program_type=Program,
        database_type=OptimizationAwareDatabase,
        classifier_type=OptimizationFeatureClassifier,
        triton_pattern_dicts=(
            TRITON_MEMORY_OPT_PATTERNS, TRITON_COMPUTE_OPT_PATTERNS,
            TRITON_PARALLELISM_OPT_PATTERNS),
        setup_logging=setup_logging,
        gpu_specs_module=gpu_specs)


@contextmanager
def _install_triton_pattern_cache(upstream: _Upstream):
    """Make the pinned classifier's declared Triton patterns executable."""
    classifier = upstream.classifier_type
    old_cache = classifier._compiled_cache
    old_initialized = classifier._cache_initialized
    classifier._ensure_patterns_compiled()
    cache = dict(classifier._compiled_cache)
    required: set[str] = set()
    compiled = 0
    try:
        for pattern_dict in upstream.triton_pattern_dicts:
            for categories in pattern_dict.values():
                for spec in categories.values():
                    for pattern in spec.get("patterns", ()):
                        required.add(pattern)
                        if cache.get(pattern) is not None:
                            continue
                        try:
                            cache[pattern] = re.compile(
                                pattern, re.IGNORECASE | re.MULTILINE)
                        except re.error as exc:
                            raise KernelFoundryArenaError(
                                f"invalid pinned Triton feature pattern: {exc}") from exc
                        compiled += 1
        classifier._compiled_cache = cache
        classifier._cache_initialized = True
        unresolved = sorted(
            pattern for pattern in required if cache.get(pattern) is None)
        if unresolved:
            raise KernelFoundryArenaError(
                f"{len(unresolved)} pinned Triton patterns are unresolved")
        yield compiled
    finally:
        classifier._compiled_cache = old_cache
        classifier._cache_initialized = old_initialized


def _build_config(
    upstream: _Upstream, *, source: Path, artifact_root: Path,
    max_iterations: int, branches_per_iteration: int,
) -> Any:
    with upstream.hydra.initialize_config_dir(
            config_dir=str(source / "configs"), version_base=None):
        config = upstream.hydra.compose(config_name="run")
    config.job_name = "agentkernelarena-mi210"
    config.task_name = "agentkernelarena-task"
    config.task_origin = "AgentKernelArena"
    config.language = "triton"
    config.gpu_arch = "MI210"
    config.max_iters = max_iterations
    config.branches_per_iteration = branches_per_iteration
    config.stop_once_correct = False
    config.start_from_best = False
    config.kernels_iter_0_path = None
    config.use_queue = False
    config.use_container = False
    config.has_build_step = False
    config.has_reference_build_step = False
    config.store_generated_kernels_in_db = False
    config.use_feedback_llm = False
    config.use_docs_via_keywords = False
    config.postprocess_code = False
    config.logdir = str(artifact_root / "kernelfoundry-runs")
    config.prompt.diff_format = False
    config.prompt.include_hardware_specs = True
    config.prompt.meta_prompting.enabled = False
    upstream.omega_conf.set_struct(config.database.config, False)
    config.database.config.in_memory = True
    config.database.config.db_path = None
    config.database.config.artifacts_base_path = str(
        artifact_root / "kernelfoundry-map-elites")
    config.database.config.use_gradient_tracking = True
    config.database.config.gradient_sampling_weight = 0.3
    config.database.config.gradient_config = upstream.omega_conf.to_container(
        config.gradient_config, resolve=True)
    return config


def _make_controller(
    upstream: _Upstream, *, config: Any,
    server: KernelFoundryTextServer,
    evaluator: common.ArenaWorkspaceEvaluator,
) -> Any:
    class ArenaProgramDatabase:
        """Bind inherited add() calls to upstream QD transition tracking."""

        def __init__(self, database_config):
            self.inner = upstream.database_type(database_config)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def add(self, program, island_id=None, force_add=False,
                iteration=None):
            parent = (self.inner.programs.get(program.parent_id)
                      if program.parent_id else None)
            if parent is not None:
                return self.inner.add_with_parent(
                    child=program, parent=parent, island_id=island_id,
                    force_add=force_add, iteration=iteration,
                    mutation_hint=program.metadata.get("changes"))
            return self.inner.add(
                program, island_id=island_id, force_add=force_add,
                iteration=iteration)

    class ArenaController(upstream.controller_type):
        def __init__(self):
            self.config = config
            self.job_id = None
            self.task_id = None
            self.failures = 0
            self.max_failures = int(config.get("max_failures", 5))
            self.evolve_mode = config.branches_per_iteration > 1
            if not self.evolve_mode:
                raise KernelFoundryArenaError(
                    "KernelFoundry Arena port requires MAP-Elites branching")
            self.resource_allocator = None
            base_logdir = Path(config.logdir)
            base_logdir.mkdir(parents=True, exist_ok=True)
            upstream.setup_logging(str(base_logdir), logging_level=logging.INFO)
            self.answer_processor = upstream.answer_processor_type(
                language=config.language,
                diff_format=config.prompt.diff_format,
                postprocess_code=config.postprocess_code,
                postprocess_code_config=config.postprocess_code_config)
            self.reference_language = "complete AgentKernelArena Triton task"
            self.llm_server = server
            self.feedback_helper = upstream.feedback_helper_type(
                use_feedback_llm=False, language=config.language,
                server_config=config.feedback_llm_config,
                use_docs_via_keywords=False)
            upstream.gpu_specs_module.ARCH_TO_NAME["MI210"] = (
                "AMD Instinct MI210")
            upstream.gpu_specs_module.ARCH_TO_SPECS["MI210"] = dict(
                _MI210_SPECS)
            self.prompt_constructor = upstream.prompt_constructor_type(
                config.language, config.gpu_arch, config.prompt,
                reference_language=self.reference_language,
                mode=config.mode, use_feedback_llm=False)
            self.program_database = ArenaProgramDatabase(
                config.database.config)
            config.logdir = str(base_logdir / config.job_name)
            Path(config.logdir).mkdir(parents=True, exist_ok=True)
            self.setup_prompt_evolution()

        def evaluate_single(self, program, problem_logger, version=0):
            if not isinstance(program.code, str) or not program.code.strip():
                record = None
            else:
                record = evaluator.evaluate(
                    {evaluator.source_paths[0]: program.code})
            if record is None:
                result = upstream.eval_result_type(
                    compiled=False, correctness=False, perf_score=0,
                    eval_log="KernelFoundry produced no complete source file")
            else:
                compiled = bool(record.raw.get("pass_compilation"))
                correct = bool(record.raw.get("pass_correctness"))
                result = upstream.eval_result_type(
                    compiled=compiled, correctness=correct,
                    perf_score=5 if record.passed else (2 if compiled else 0),
                    runtime=(record.latency_ms * 1000.0
                             if record.latency_ms else -1.0),
                    runtime_improvement=(record.speedup
                                         if record.passed and record.speedup
                                         else -1.0),
                    runtime_stats={"arena": {
                        "speedup": record.speedup,
                        "latency_ms": record.latency_ms}},
                    eval_log=(record.log_excerpt or
                              "AgentKernelArena evaluation completed"),
                    metadata={"arena_score": record.score,
                              "arena_evidence_only": True})
            artifact_path = Path(
                f"{problem_logger.stdout_path_part}_v{version}.txt")
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(result.eval_log, encoding="utf-8")
            return result, program.task

    return ArenaController()


def run_controller(
    *, prompt: str, workspace: str | Path, arena_root: str | Path,
    source_root: str | Path, budget: common.ControllerBudget,
    max_iterations: int, branches_per_iteration: int = 4,
    model_factory: Callable[..., Any] = common.CodexTextModel,
    evaluator_factory: Callable[..., Any] = common.ArenaWorkspaceEvaluator,
    upstream_loader: Callable[[Path], _Upstream] = _load_upstream,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise KernelFoundryArenaError("Arena prompt must be non-empty")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or not 1 <= max_iterations <= 256):
        raise KernelFoundryArenaError("max_iterations must be in [1, 256]")
    if (isinstance(branches_per_iteration, bool)
            or not isinstance(branches_per_iteration, int)
            or not 2 <= branches_per_iteration <= 16):
        raise KernelFoundryArenaError(
            "branches_per_iteration must be in [2, 16]")
    root = common.workspace_root(workspace)
    source = Path(source_root).resolve()
    model = model_factory(workspace=root, budget=budget)
    evaluator = evaluator_factory(
        workspace=root, arena_root=Path(arena_root).resolve())
    upstream = upstream_loader(source)
    server = KernelFoundryTextServer(model)
    config = _build_config(
        upstream, source=source, artifact_root=model.artifact_root,
        max_iterations=max_iterations,
        branches_per_iteration=branches_per_iteration)
    stop_reason = "upstream_complete"
    programs = None
    best_id = None
    with _install_triton_pattern_cache(upstream) as patterns_added:
        controller = _make_controller(
            upstream, config=config, server=server, evaluator=evaluator)
        task = KernelFoundryArenaTask(prompt=prompt, evaluator=evaluator)
        try:
            programs, best_id = controller.run_single(task)
        except Exception:
            if server.expired:
                stop_reason = "campaign_checkpoint"
            else:
                raise
    evaluator.materialize_best()
    database = controller.program_database
    transition_stats = database.get_transition_statistics()
    return common.build_controller_receipt(
        controller_id=CONTROLLER_ID, source_root=source,
        source_commit=SOURCE_COMMIT, entrypoint=source / UPSTREAM_ENTRYPOINT,
        model=model, evaluator=evaluator, stop_reason=stop_reason,
        extra={
            "upstream_callable": "Controller.run_single",
            "upstream_search": "optimization-aware MAP-Elites with QD gradient",
            "max_iterations": max_iterations,
            "branches_per_iteration": branches_per_iteration,
            "maximum_model_calls": max_iterations * branches_per_iteration,
            "map_elites_program_count": len(getattr(database, "programs", {})),
            "map_elites_occupied_cells": len(getattr(database, "feature_map", {})),
            "triton_patterns_added_to_cache": patterns_added,
            "qd_gradient_enabled": bool(getattr(
                database, "_use_gradient_tracking", False)),
            "qd_transition_count": int(
                transition_stats.get("total_transitions", 0)),
            "qd_unique_directions": int(
                transition_stats.get("unique_directions_observed", 0)),
            "returned_program_count": len(programs or {}),
            "returned_best_program_id": best_id,
            "gfx90a_port": True,
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
        "--max-iterations", "16", "--branches-per-iteration", "4",
        "--workspace", ".",
        "--arena-root", os.environ.get(
            "AUTOKERNEL_AGENT_KERNEL_ARENA_ROOT",
            "/mnt/raid0/llm/autokernel/vendor/agent-kernel-arena"),
        "--source-root", str(source_root))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--checkpoint-hours", required=True, type=float)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--branches-per-iteration", required=True, type=int)
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
        max_iterations=args.max_iterations,
        branches_per_iteration=args.branches_per_iteration)
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "CONTROLLER_ID", "DEFAULT_SOURCE_ROOT", "ENTRYPOINT_RELATIVE",
    "EXECUTABLE_MODULE", "KernelFoundryArenaError", "KernelFoundryArenaTask",
    "KernelFoundryTextServer", "PINNED_MODEL_IDS", "REQUIRED_CLIS",
    "RUNTIME_PYTHON", "SOURCE_COMMIT", "SOURCE_PIN", "UPSTREAM_ENTRYPOINT",
    "campaign_argv", "run_controller",
]


if __name__ == "__main__":
    raise SystemExit(main())
