#!/usr/bin/env python3
"""Run pinned Xe-Forge linear CoVeR against governed AgentKernelArena.

This is an explicit AMD MI210/gfx90a port.  Xe-Forge retains ownership of
analysis, planning, and its CoVeR optimization loop; all compilation,
correctness, and timing decisions cross the shared Arena evaluator seam.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from . import arena_adapter
from . import arena_upstream_common as common

CONTROLLER_ID = "xe_forge_gfx90a_linear_cover"
SOURCE_COMMIT = "4dcb5080b0f56d0b655ec8c8c9509b8e3ba0382c"
SOURCE_PIN = arena_adapter.VendorPin(
    name="Xe-Forge", commit=SOURCE_COMMIT, license_path="LICENSE",
    required_paths=(
        "src/xe_forge/engines/dspy_engine.py",
        "src/xe_forge/pipeline.py",
        "src/xe_forge/core/executor.py",
        "src/xe_forge/agents/optimizer_agent.py",
    ),
)
DEFAULT_SOURCE_ROOT = Path(
    "/mnt/raid0/llm/autokernel/vendor/arena-controllers/xe-forge")
RUNTIME_PYTHON = Path(
    "/mnt/raid0/llm/tools/geak-v1-rocm62-py312/bin/python")
ENTRYPOINT_RELATIVE = (
    "scripts/kernel_rnd/autokernel/controller/xe_forge_arena.py")
EXECUTABLE_MODULE = (
    "scripts.kernel_rnd.autokernel.controller.xe_forge_arena")
UPSTREAM_ENTRYPOINT = "src/xe_forge/engines/dspy_engine.py"
PINNED_MODEL_IDS = common.PINNED_MODEL_IDS
REQUIRED_CLIS = common.REQUIRED_CLIS

# Xe-Forge gates all executor calls on a truthy KernelBench shape list.  Arena
# owns the real workload shapes, so this marker only opens the upstream seam;
# it is explicitly excluded from model guidance and never creates tensors.
_ARENA_SHAPE_MARKER = [(1,)]
_AMD_GUIDANCE = """
This is an AMD gfx90a port of Xe-Forge. Target the physical AMD Instinct MI210
with wavefront size 64. The AgentKernelArena source file and its embedded tests
are the complete task authority. Ignore any Intel XPU, NVIDIA CUDA, SM,
compute-capability, warp-32, GRF, TMA, or placeholder input-shape guidance
inherited from upstream prompts. Use ROCm Triton constructs supported on gfx90a;
preserve the complete file, every public signature, imports, and test harness.
Never replace the target Triton kernel with a vendor-library call. The
compile_and_verify tool is the sole authority for compilation, correctness,
and performance.
""".strip()


class XeForgeArenaError(common.UpstreamControllerError):
    """Xe-Forge cannot preserve its governed gfx90a port contract."""


@dataclass(frozen=True)
class _Upstream:
    dspy: Any
    config: Any
    pipeline_module: Any
    engine_type: Any
    pipeline_type: Any
    analyzer_type: Any
    optimizer_type: Any
    executor_type: Any
    execution_result_type: Any
    comparison_result_type: Any
    success_message: str
    extract_code: Callable[[Any], str]


class XeForgeCodexLM:
    """Factory namespace for a DSPy BaseLM backed by the governed Codex model."""

    @staticmethod
    def build(dspy_module: Any, model: common.CodexTextModel) -> Any:
        class CodexLM(dspy_module.BaseLM):
            def __init__(self) -> None:
                super().__init__(
                    model=f"openai/{common.MODEL_ID}", model_type="chat",
                    temperature=0.0, max_tokens=32768, cache=False)

            def forward(self, prompt=None, messages=None, **kwargs):
                del kwargs
                if messages:
                    rendered = "\n\n".join(
                        f"{row.get('role', 'user')}: {row.get('content', '')}"
                        for row in messages if isinstance(row, dict))
                else:
                    rendered = str(prompt or "")
                content = model.call(rendered)
                return SimpleNamespace(
                    id=("autokernel-xe-forge-"
                        f"{int(model.identity().get('call_count', 0))}"),
                    model=common.MODEL_ID,
                    choices=[SimpleNamespace(
                        index=0,
                        message=SimpleNamespace(
                            role="assistant", content=content),
                        finish_reason="stop")],
                    usage={"prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0},
                    _hidden_params={})

        return CodexLM()


def _source_text(evaluator: common.ArenaWorkspaceEvaluator) -> str:
    if len(evaluator.source_paths) != 1:
        raise XeForgeArenaError(
            "the current Xe-Forge port requires one Arena source file")
    return (evaluator.workspace / evaluator.source_paths[0]).read_text(
        encoding="utf-8")


def _make_executor(upstream: _Upstream,
                   evaluator: common.ArenaWorkspaceEvaluator) -> Any:
    starting = _source_text(evaluator)
    starting_sha = hashlib.sha256(starting.encode()).hexdigest()

    class ArenaKernelBenchExecutor(upstream.executor_type):
        """KernelBench-shaped adapter whose only backend is AgentKernelArena."""

        def __init__(self) -> None:
            self.device = "cuda"  # PyTorch's stable ROCm device spelling.
            self.require_correctness = True
            self.rtol = 1e-2
            self.atol = 1e-5
            self._records: dict[str, common.EvaluationRecord] = {}

        @staticmethod
        def _digest(code: str) -> str:
            return hashlib.sha256(code.encode()).hexdigest()

        def _record(self, code: str) -> common.EvaluationRecord | None:
            digest = self._digest(code)
            if digest == starting_sha:
                return None
            if digest not in self._records:
                self._records[digest] = evaluator.evaluate(
                    {evaluator.source_paths[0]: code})
            return self._records[digest]

        def execute(self, kernel_code: str, *args, **kwargs):
            del args, kwargs
            record = self._record(kernel_code)
            if record is None:
                return upstream.execution_result_type(
                    success=True, execution_time_ms=1.0,
                    output_correct=True)
            normalized_ms = (
                1.0 / record.speedup if record.passed and record.speedup else None)
            return upstream.execution_result_type(
                success=record.passed, execution_time_ms=normalized_ms,
                output_correct=record.passed,
                error_message=None if record.passed else record.log_excerpt)

        def compare_kernels(self, original_code: str, optimized_code: str,
                            *args, **kwargs):
            del original_code, args, kwargs
            record = self._record(optimized_code)
            if record is None:
                return upstream.comparison_result_type(
                    original_time_us=1000.0, optimized_time_us=1000.0,
                    speedup=1.0)
            speedup = record.speedup or 0.0
            optimized_us = 1000.0 / speedup if speedup > 0 else float("inf")
            return upstream.comparison_result_type(
                original_time_us=1000.0,
                optimized_time_us=optimized_us,
                speedup=speedup,
                optimized_correct=record.passed,
                is_slower=record.passed and speedup < 1.0,
                feedback_message=record.log_excerpt,
            )

    return ArenaKernelBenchExecutor()


def _make_pipeline(upstream: _Upstream, model: common.CodexTextModel) -> Any:
    class ArenaOptimizerAgent(upstream.optimizer_type):
        def _create_verify_tool(
            self, original_code, kernel_name, input_shapes, flop, dtype=None,
            init_args=None, skip_speedup_check=False, stage=None,
            baseline_ms=None, spec_dims=None, input_dtypes=None,
        ):
            del input_shapes, flop, dtype, init_args, stage, baseline_ms
            del spec_dims, input_dtypes
            last_accepted = {"comparison": None}

            def compile_and_verify(optimized_code):
                code = upstream.extract_code(
                    optimized_code.code if hasattr(optimized_code, "code")
                    else str(optimized_code))
                try:
                    ast.parse(code)
                except SyntaxError as exc:
                    return f"SYNTAX ERROR at line {exc.lineno}: {exc.msg}"
                required = (
                    ("import triton" in code or "from triton" in code,
                     "MISSING: import triton"),
                    ("@triton.jit" in code, "MISSING: @triton.jit decorator"),
                )
                for valid, error in required:
                    if not valid:
                        return error
                original_targets = set(re.findall(
                    r"@triton\.jit[\s\S]{0,300}?def\s+(\w+)\s*\(",
                    original_code))
                missing = sorted(name for name in original_targets
                                 if not re.search(rf"\bdef\s+{re.escape(name)}\s*\(", code))
                if missing:
                    return f"MISSING: original Triton target(s) {missing}"
                try:
                    comparison = self.executor.compare_kernels(
                        original_code=original_code, optimized_code=code,
                        kernel_name=kernel_name)
                except Exception as exc:
                    return f"RUNTIME ERROR: {exc}"
                if not comparison.optimized_correct:
                    return comparison.feedback_message or "Optimized kernel failed."
                if comparison.is_slower and not skip_speedup_check:
                    return (comparison.feedback_message or
                            "PERFORMANCE REGRESSION: candidate is slower")
                last_accepted["comparison"] = comparison
                return upstream.success_message

            return (upstream.dspy.Tool(
                func=compile_and_verify, name="compile_and_verify",
                desc=("Evaluate the complete source only with AgentKernelArena; "
                      f'returns "{upstream.success_message}" on success.')),
                    last_accepted)

    class ArenaXeForgePipeline(upstream.pipeline_type):
        def _setup_llm(self):
            upstream.dspy.configure(
                lm=XeForgeCodexLM.build(upstream.dspy, model),
                warn_on_type_mismatch=False)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.analyzer = upstream.analyzer_type(
                knowledge_base=self.knowledge_base,
                dsl=self.config.device_config.dsl,
                extra_instructions=_AMD_GUIDANCE)
            self.optimizer = ArenaOptimizerAgent(
                executor=self.executor, validator=self.validator,
                max_iterations=self.config.agent.max_iterations,
                knowledge_base=self.knowledge_base,
                dsl=self.config.device_config.dsl,
                extra_instructions=_AMD_GUIDANCE)
            self.coordinator = None

    return ArenaXeForgePipeline


def _load_upstream(source_root: Path) -> _Upstream:
    arena_adapter.inspect_vendor_source(source_root, SOURCE_PIN)
    sys.path.insert(0, str(source_root / "src"))
    try:
        import dspy
        import xe_forge.config as config_module
        import xe_forge.pipeline as pipeline_module
        from xe_forge.agents import AnalyzerAgent, OptimizerAgent
        from xe_forge.agents.optimizer_agent import _extract_code_from_response
        from xe_forge.agents.utils import SUCCESS_MESSAGE
        from xe_forge.core.executor import ComparisonResult, KernelBenchExecutor
        from xe_forge.engines.dspy_engine import DSPyEngine
        from xe_forge.models import ExecutionResult
        from xe_forge.pipeline import XeForgePipeline
    except ImportError as exc:
        raise XeForgeArenaError("cannot import pinned Xe-Forge") from exc
    return _Upstream(
        dspy=dspy, config=config_module, pipeline_module=pipeline_module,
        engine_type=DSPyEngine, pipeline_type=XeForgePipeline,
        analyzer_type=AnalyzerAgent, optimizer_type=OptimizerAgent,
        executor_type=KernelBenchExecutor,
        execution_result_type=ExecutionResult,
        comparison_result_type=ComparisonResult,
        success_message=SUCCESS_MESSAGE,
        extract_code=_extract_code_from_response)


def _config(upstream: _Upstream, artifact_root: Path, max_iterations: int) -> Any:
    cfg = upstream.config.Config()
    cfg.llm.model = f"openai/{common.MODEL_ID}"
    cfg.llm.temperature = 0.0
    cfg.llm.max_tokens = 32768
    cfg.agent.strategy = "cover"
    cfg.agent.max_iterations = max_iterations
    cfg.device_config = upstream.config.CUDAConfig(
        device="cuda", dsl="triton", default_num_warps=4,
        default_num_stages=2, preferred_tile_m=128,
        preferred_tile_n=128, preferred_tile_k=32)
    cfg.knowledge.enabled = False
    cfg.logging.log_dir = str(artifact_root / "xe-forge-logs")
    cfg.logging.kernel_dir = str(artifact_root / "xe-forge-kernels")
    cfg.logging.save_intermediate = True
    cfg.trial.enabled = False
    cfg.profiler.vtune_enabled = False
    return cfg


def run_controller(
    *, prompt: str, workspace: str | Path, arena_root: str | Path,
    source_root: str | Path, budget: common.ControllerBudget,
    max_iterations: int,
    model_factory: Callable[..., Any] = common.CodexTextModel,
    evaluator_factory: Callable[..., Any] = common.ArenaWorkspaceEvaluator,
    upstream_loader: Callable[[Path], _Upstream] = _load_upstream,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise XeForgeArenaError("Arena prompt must be non-empty")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or not 1 <= max_iterations <= 256):
        raise XeForgeArenaError("max_iterations must be in [1, 256]")
    root = common.workspace_root(workspace)
    source = Path(source_root).resolve()
    model = model_factory(workspace=root, budget=budget)
    evaluator = evaluator_factory(
        workspace=root, arena_root=Path(arena_root).resolve())
    upstream = upstream_loader(source)
    executor = _make_executor(upstream, evaluator)
    pipeline_type = _make_pipeline(upstream, model)
    previous_pipeline = upstream.pipeline_module.XeForgePipeline
    stop_reason = "upstream_complete"
    result = None
    try:
        upstream.pipeline_module.XeForgePipeline = pipeline_type
        engine = upstream.engine_type(
            _config(upstream, model.artifact_root, max_iterations),
            executor=executor)
        result = engine.optimize(
            kernel_code=_source_text(evaluator),
            reference_code=evaluator.definition(prompt),
            kernel_name=str(evaluator.config.get(
                "target_kernel_functions", ["kernel"])[0]),
            input_shapes=_ARENA_SHAPE_MARKER)
    except common.ControllerBudgetExpired:
        stop_reason = "campaign_checkpoint"
    finally:
        upstream.pipeline_module.XeForgePipeline = previous_pipeline
    evaluator.materialize_best()
    return common.build_controller_receipt(
        controller_id=CONTROLLER_ID, source_root=source,
        source_commit=SOURCE_COMMIT, entrypoint=source / UPSTREAM_ENTRYPOINT,
        model=model, evaluator=evaluator, stop_reason=stop_reason,
        extra={
            "upstream_callable": "DSPyEngine.optimize",
            "upstream_strategy": "linear CoVeR",
            "gfx90a_port": True,
            "max_iterations": max_iterations,
            "arena_shape_marker_not_evidence": list(_ARENA_SHAPE_MARKER),
            "returned_optimized_code": bool(
                result is not None and getattr(result, "optimized_code", None)),
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
        "--max-iterations", "64", "--workspace", ".",
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
        max_iterations=args.max_iterations)
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "CONTROLLER_ID", "DEFAULT_SOURCE_ROOT", "ENTRYPOINT_RELATIVE",
    "EXECUTABLE_MODULE", "PINNED_MODEL_IDS", "REQUIRED_CLIS",
    "RUNTIME_PYTHON", "SOURCE_COMMIT", "SOURCE_PIN", "UPSTREAM_ENTRYPOINT",
    "XeForgeArenaError", "XeForgeCodexLM", "campaign_argv", "run_controller",
]


if __name__ == "__main__":
    raise SystemExit(main())
