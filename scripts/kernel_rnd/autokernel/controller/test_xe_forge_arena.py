#!/usr/bin/env python3
"""Tests for the licensed Xe-Forge gfx90a AgentKernelArena port."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from . import arena_upstream_common as common
from . import xe_forge_arena as X


class FakeEvaluator:
    def __init__(self, *, workspace: Path, arena_root: Path):
        self.workspace = workspace
        self.arena_root = arena_root
        self.source_paths = ("kernel.py",)
        self.config = {"target_kernel_functions": ["kernel"]}
        self.seen = []
        self.materialized = False

    def definition(self, prompt):
        return prompt + "\ncomplete kernel.py"

    def evaluate(self, files):
        self.seen.append(files)
        return common.EvaluationRecord(
            passed=True, latency_ms=0.5, speedup=2.0,
            log_excerpt="Arena accepted", raw={"pass_correctness": True})

    def materialize_best(self):
        self.materialized = True

    def receipt_fields(self):
        return {"evaluation_count": len(self.seen), "best_score": 2.0}


class FakeModel:
    def __init__(self, *, workspace, budget):
        self.workspace = workspace
        self.budget = budget
        self.artifact_root = workspace / common.ARTIFACT_DIRNAME
        self.artifact_root.mkdir()
        self.prompts = []

    def call(self, prompt):
        self.prompts.append(prompt)
        return "fixture response"

    def identity(self):
        return {"cli": "fixture", "model": common.MODEL_ID,
                "call_count": len(self.prompts)}


@dataclass
class FakeComparison:
    original_time_us: float
    optimized_time_us: float
    speedup: float
    original_tflops: float | None = None
    optimized_tflops: float | None = None
    original_correct: bool = True
    optimized_correct: bool = True
    is_slower: bool = False
    feedback_message: str = ""


class FakeExecutionResult(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FakeExecutor:
    pass


class FakeTool:
    def __init__(self, *, func, name, desc):
        self.func = func
        self.name = name
        self.desc = desc


class FakeBaseLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeDspy:
    BaseLM = FakeBaseLM
    Tool = FakeTool
    configured = None

    @classmethod
    def configure(cls, **kwargs):
        cls.configured = kwargs


class FakeAnalyzer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOptimizer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakePipeline:
    def __init__(self, *, config, executor, trial_manager=None, profiler=None):
        del trial_manager, profiler
        self.config = config
        self.executor = executor
        self.validator = None
        self.knowledge_base = None
        self._setup_llm()


class FakeConfig:
    def __init__(self):
        self.llm = SimpleNamespace()
        self.agent = SimpleNamespace()
        self.device_config = SimpleNamespace(dsl="triton")
        self.knowledge = SimpleNamespace(enabled=False)
        self.logging = SimpleNamespace()
        self.trial = SimpleNamespace(enabled=False)
        self.profiler = SimpleNamespace(vtune_enabled=False)


class FakeCUDAConfig(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class XeForgeArenaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "kernel.py").write_text(
            "import triton\n@triton.jit\ndef kernel(x):\n    return x\n",
            encoding="utf-8")
        self.arena = self.root / "arena"
        self.arena.mkdir()
        self.source = self.root / "xe-forge"
        entrypoint = self.source / X.UPSTREAM_ENTRYPOINT
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("# fixture DSPyEngine\n", encoding="utf-8")

    def fake_upstream(self):
        pipeline_module = SimpleNamespace(XeForgePipeline=FakePipeline)

        class FakeEngine:
            calls = 0

            def __init__(engine_self, config, executor=None):
                engine_self.config = config
                engine_self.executor = executor

            def optimize(engine_self, **kwargs):
                del engine_self
                FakeEngine.calls += 1
                pipeline = pipeline_module.XeForgePipeline(
                    config=kwargs.pop("config", None) or config_ref[0],
                    executor=executor_ref[0])
                tool, accepted = pipeline.optimizer._create_verify_tool(
                    kwargs["kernel_code"], kwargs["kernel_name"],
                    kwargs["input_shapes"], None)
                candidate = kwargs["kernel_code"].replace("return x", "return x + 1")
                self.assertEqual(tool.func(candidate), "SUCCESS")
                self.assertIsNotNone(accepted["comparison"])
                return SimpleNamespace(optimized_code=candidate)

        config_ref = [None]
        executor_ref = [None]

        class CapturingEngine(FakeEngine):
            def __init__(engine_self, config, executor=None):
                config_ref[0] = config
                executor_ref[0] = executor
                super().__init__(config, executor)

        return X._Upstream(
            dspy=FakeDspy,
            config=SimpleNamespace(Config=FakeConfig, CUDAConfig=FakeCUDAConfig),
            pipeline_module=pipeline_module, engine_type=CapturingEngine,
            pipeline_type=FakePipeline, analyzer_type=FakeAnalyzer,
            optimizer_type=FakeOptimizer, executor_type=FakeExecutor,
            execution_result_type=FakeExecutionResult,
            comparison_result_type=FakeComparison,
            success_message="SUCCESS", extract_code=lambda value: str(value))

    def test_upstream_engine_and_cover_tool_cross_only_arena_seam(self):
        upstream = self.fake_upstream()
        receipt = X.run_controller(
            prompt="optimize", workspace=self.workspace, arena_root=self.arena,
            source_root=self.source,
            budget=common.ControllerBudget(2.0, 7200), max_iterations=3,
            model_factory=FakeModel, evaluator_factory=FakeEvaluator,
            upstream_loader=lambda source: upstream)
        self.assertEqual(receipt["controller_id"], X.CONTROLLER_ID)
        self.assertEqual(receipt["extra"]["upstream_callable"],
                         "DSPyEngine.optimize")
        self.assertEqual(receipt["evaluation"]["evaluation_count"], 1)
        self.assertTrue(receipt["extra"]["gfx90a_port"])
        self.assertIs(upstream.pipeline_module.XeForgePipeline, FakePipeline)
        persisted = json.loads((
            self.workspace / common.ARTIFACT_DIRNAME / "receipt.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(persisted["receipt_sha256"], receipt["receipt_sha256"])

    def test_codex_lm_and_campaign_argv_keep_exact_identity(self):
        model = FakeModel(
            workspace=self.workspace,
            budget=common.ControllerBudget(2.0, 7200))
        lm = X.XeForgeCodexLM.build(FakeDspy, model)
        response = lm.forward(messages=[{"role": "user", "content": "prompt"}])
        self.assertEqual(response.model, common.MODEL_ID)
        self.assertEqual(model.prompts, ["user: prompt"])
        argv = X.campaign_argv("python-fixture")
        self.assertEqual(argv[0], "python-fixture")
        self.assertEqual(argv[argv.index("--checkpoint-hours") + 1], "32")
        self.assertEqual(argv[argv.index("--timeout-seconds") + 1], "115200")

    def test_invalid_budget_and_multi_source_fail_closed(self):
        with self.assertRaisesRegex(X.XeForgeArenaError, r"\[1, 256\]"):
            X.run_controller(
                prompt="optimize", workspace=self.workspace,
                arena_root=self.arena, source_root=self.source,
                budget=common.ControllerBudget(2.0, 7200), max_iterations=0,
                model_factory=FakeModel, evaluator_factory=FakeEvaluator,
                upstream_loader=lambda source: self.fake_upstream())
        evaluator = FakeEvaluator(workspace=self.workspace, arena_root=self.arena)
        evaluator.source_paths = ("a.py", "b.py")
        with self.assertRaisesRegex(X.XeForgeArenaError, "one Arena source"):
            X._source_text(evaluator)


if __name__ == "__main__":
    unittest.main()
