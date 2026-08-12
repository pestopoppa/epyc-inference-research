#!/usr/bin/env python3
"""Tests for the licensed K-Search AgentKernelArena port."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from . import arena_upstream_common as common
from . import k_search_arena as K


class Language(str, Enum):
    TRITON = "triton"


@dataclass
class SourceFile:
    path: str
    content: str


@dataclass
class BuildSpec:
    language: Language
    target_hardware: list[str]
    entry_point: str


@dataclass
class Solution:
    name: str
    definition: str
    author: str
    spec: BuildSpec
    sources: list[SourceFile]
    description: str


@dataclass
class EvalResult:
    status: str
    latency_ms: float | None
    reference_latency_ms: float | None
    mean_vs_baseline_factor: float | None
    speedup_factor: float | None
    log_excerpt: str
    metrics: dict

    def to_dict(self):
        return vars(self)


TYPES = SimpleNamespace(
    Solution=Solution, SourceFile=SourceFile, BuildSpec=BuildSpec,
    SupportedLanguages=Language, EvalResult=EvalResult)


class FakeEvaluator:
    def __init__(self, *, workspace: Path, arena_root: Path, source_paths=("kernel.py",)):
        self.workspace = workspace
        self.arena_root = arena_root
        self.source_paths = tuple(source_paths)
        self.config = {"target_kernel_functions": ["kernel"]}
        self.best = False
        self.seen = []

    def definition(self, prompt: str) -> str:
        return prompt + "\ncomplete kernel.py"

    def evaluate(self, files):
        self.seen.append(files)
        return common.EvaluationRecord(
            passed=True, latency_ms=0.5, speedup=2.0, log_excerpt="ok",
            raw={"pass_correctness": True})

    def materialize_best(self):
        self.best = True

    def receipt_fields(self):
        return {"evaluation_count": len(self.seen), "best_score": 2.0}


class FakeModel:
    def __init__(self, *, workspace: Path, budget: common.ControllerBudget):
        self.workspace = workspace
        self.budget = budget
        self.artifact_root = workspace / common.ARTIFACT_DIRNAME
        self.artifact_root.mkdir()
        self.openai_compat = object()

    def identity(self):
        return {"cli": "fixture", "model": common.MODEL_ID, "call_count": 1}


class FakeGenerator:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).last = self

    def generate(self, *, task, max_opt_rounds, wm_stagnation_window,
                 num_debug_and_improve_rounds):
        self.args = (max_opt_rounds, wm_stagnation_window,
                     num_debug_and_improve_rounds)
        solution = task.make_solution_from_generated_code(
            cleaned_code="def kernel():\n    return 1\n", raw_code="ignored",
            round_num=1, model_name=common.MODEL_ID,
            target_gpu="MI210", language="triton")
        result = task.run_benchmark(solution=solution, round_num=1)
        if result.status != "passed" or result.metrics["score"] != 2.0:
            raise AssertionError("Arena result did not cross the upstream Task seam")
        return solution


class KSearchArenaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source_env = mock.patch.dict(os.environ, {
            common.ARENA_SOURCE_PATHS_ENV: json.dumps(["kernel.py"])})
        self.source_env.start()
        self.addCleanup(self.source_env.stop)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "kernel.py").write_text(
            "def kernel():\n    return 0\n", encoding="utf-8")
        self.arena = self.root / "arena"
        self.arena.mkdir()
        self.source = self.root / "k-search"
        entrypoint = self.source / K.UPSTREAM_ENTRYPOINT
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("# pinned fixture\n", encoding="utf-8")

    @staticmethod
    def loader(source, model):
        del source, model
        return FakeGenerator, TYPES

    def test_task_maps_complete_source_and_arena_result_to_upstream_types(self):
        evaluator = FakeEvaluator(workspace=self.workspace, arena_root=self.arena)
        task = K.KSearchArenaTask(
            prompt="optimize", evaluator=evaluator, types_module=TYPES)
        solution = task.make_solution_from_generated_code(
            cleaned_code="def kernel():\n    return 2\n", raw_code="ignored",
            round_num=3, model_name="fixture", target_gpu="MI210",
            language="triton")
        self.assertEqual(solution.sources[0].path, "kernel.py")
        result = task.run_benchmark(solution=solution)
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.mean_vs_baseline_factor, 2.0)
        self.assertEqual(result.metrics["score_name"], "arena_speedup")

    def test_real_upstream_generate_callable_runs_through_injected_task_seam(self):
        receipt = K.run_controller(
            prompt="optimize", workspace=self.workspace, arena_root=self.arena,
            source_root=self.source,
            budget=common.ControllerBudget(2.0, 7200), max_rounds=7,
            model_factory=FakeModel, evaluator_factory=FakeEvaluator,
            upstream_loader=self.loader)
        self.assertEqual(receipt["controller_id"], K.CONTROLLER_ID)
        self.assertEqual(receipt["stop_reason"], "upstream_complete")
        self.assertTrue(receipt["constraints"]["upstream_search_algorithm_retained"])
        self.assertEqual(FakeGenerator.last.args, (7, 5, 5))
        persisted = json.loads((
            self.workspace / common.ARTIFACT_DIRNAME / "receipt.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(persisted["receipt_sha256"], receipt["receipt_sha256"])

    def test_campaign_argv_exposes_only_the_two_matched_budget_values(self):
        argv = K.campaign_argv("python-fixture")
        self.assertEqual(argv[0], "python-fixture")
        self.assertEqual(argv[argv.index("--checkpoint-hours") + 1], "32")
        self.assertEqual(argv[argv.index("--timeout-seconds") + 1], "115200")
        self.assertEqual(argv[argv.index("--model") + 1], common.MODEL_ID)

    def test_budget_and_multi_source_contracts_fail_closed(self):
        with self.assertRaisesRegex(common.UpstreamControllerError, "equal"):
            common.ControllerBudget(2.0, 7199)
        evaluator = FakeEvaluator(workspace=self.workspace, arena_root=self.arena)
        evaluator.source_paths = ("a.py", "b.py")
        task = K.KSearchArenaTask(
            prompt="optimize", evaluator=evaluator, types_module=TYPES)
        with self.assertRaisesRegex(K.KSearchArenaError, "one Arena source"):
            task.make_solution_from_generated_code(
                cleaned_code="code", raw_code="code", round_num=1,
                model_name="fixture", target_gpu="MI210", language="triton")


if __name__ == "__main__":
    unittest.main()
