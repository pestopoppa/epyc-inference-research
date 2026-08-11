#!/usr/bin/env python3
"""Tests for the licensed GEAK-v1 AgentKernelArena port."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

from . import arena_upstream_common as common
from . import geak_v1_arena as G


class FakeEvaluator:
    def __init__(self, *, workspace: Path, arena_root: Path):
        self.workspace = workspace
        self.arena_root = arena_root
        self.source_paths = ("kernel.py",)
        self.config = {"target_kernel_functions": ["kernel"]}
        self.seen = []
        self.materialized = False

    def definition(self, prompt):
        return prompt + "\nreturn a complete file"

    def evaluate(self, files):
        self.seen.append(files)
        return common.EvaluationRecord(
            passed=True, latency_ms=0.25, speedup=1.5, log_excerpt="ok",
            raw={"pass_compilation": True, "pass_correctness": True})

    def materialize_best(self):
        self.materialized = True

    def receipt_fields(self):
        return {"evaluation_count": len(self.seen), "best_score": 1.5}


class FakeModel:
    def __init__(self, *, workspace, budget):
        self.workspace = workspace
        self.budget = budget
        self.artifact_root = workspace / common.ARTIFACT_DIRNAME
        self.artifact_root.mkdir()
        self.prompts = []

    def call(self, prompt):
        self.prompts.append(prompt)
        return '{"thought":"fixture","code":"def kernel():\\n    return 1"}'

    def identity(self):
        return {"cli": "fixture", "model": common.MODEL_ID,
                "call_count": len(self.prompts)}


class FakeOptimAgent:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).last = self

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        dataset = self.kwargs["dataset"]
        filename = dataset.problem_states[0].filename
        result = dataset.test_opt_correctness(
            "def kernel():\n    return 2\n", filename,
            exe_dir="pass_exe")
        if result[:2] != (True, True):
            raise AssertionError("Arena correctness did not cross GEAK dataset seam")
        perf = dataset.run_perf_evaluation("pass_exe", "perf_results")
        if perf[filename]["ms"] != 1.5:
            raise AssertionError("higher-is-better direction was not normalized")


class GeakArenaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "kernel.py").write_text(
            "def kernel():\n    return 0\n", encoding="utf-8")
        self.arena = self.root / "arena"
        self.arena.mkdir()
        self.source = self.root / "geak-v1"
        entrypoint = self.source / G.UPSTREAM_ENTRYPOINT
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("# fixture OptimAgent\n", encoding="utf-8")
        corpus = self.source / "src/dataloaders/TB_eval/train_crawl.json"
        corpus.parent.mkdir(parents=True)
        corpus.write_text("[]\n", encoding="utf-8")

    def test_dataset_projects_arena_correctness_and_normalizes_speedup(self):
        evaluator = FakeEvaluator(workspace=self.workspace, arena_root=self.arena)
        dataset = G.GeakArenaDataset(
            prompt="optimize", evaluator=evaluator,
            runtime_root=self.workspace / "geak-safe")
        filename = dataset.problem_states[0].filename
        result = dataset.test_opt_correctness(
            "def kernel():\n    return 2\n", filename, exe_dir="pass_exe")
        self.assertEqual(result[:2], (True, True))
        self.assertTrue((dataset.log_root / "pass_exe" / filename).is_file())
        perf = dataset.run_perf_evaluation("pass_exe", "perf_results")
        self.assertEqual(perf[filename], {"ms": 1.5, "efficiency": 1.5})

    def test_upstream_import_restores_preloaded_arena_agents_namespace(self):
        source = self.root / "namespace-fixture"
        agents = source / "src" / "agents"
        agents.mkdir(parents=True)
        (agents / "__init__.py").write_text("", encoding="utf-8")
        (agents / "helper.py").write_text("TOKEN = 'geak'\n", encoding="utf-8")
        (agents / "OptimAgent_ROCm.py").write_text(
            "from agents.helper import TOKEN\n"
            "class OptimAgent:\n"
            "    marker = TOKEN\n",
            encoding="utf-8",
        )
        sentinel = ModuleType("agents")
        sentinel.__path__ = []
        previous = {
            name: module for name, module in tuple(sys.modules.items())
            if name == "agents" or name.startswith("agents.")
        }
        for name in previous:
            sys.modules.pop(name, None)
        sys.modules["agents"] = sentinel
        try:
            optim_agent = G._import_optim_agent_isolated(source)
            self.assertEqual(optim_agent.marker, "geak")
            self.assertIs(sys.modules["agents"], sentinel)
            self.assertNotIn("agents.OptimAgent_ROCm", sys.modules)
            self.assertNotIn("agents.helper", sys.modules)
        finally:
            for name in tuple(sys.modules):
                if name == "agents" or name.startswith("agents."):
                    sys.modules.pop(name, None)
            sys.modules.update(previous)

    def test_upstream_optimagent_run_uses_safe_runtime_and_arena_dataset(self):
        receipt = G.run_controller(
            prompt="optimize", workspace=self.workspace, arena_root=self.arena,
            source_root=self.source,
            budget=common.ControllerBudget(2.0, 7200), max_iterations=3,
            model_factory=FakeModel, evaluator_factory=FakeEvaluator,
            upstream_loader=lambda source: FakeOptimAgent)
        self.assertEqual(receipt["controller_id"], G.CONTROLLER_ID)
        self.assertEqual(receipt["stop_reason"], "upstream_complete")
        self.assertEqual(FakeOptimAgent.last.run_kwargs["iteration_num"], 3)
        self.assertFalse(FakeOptimAgent.last.run_kwargs["multi_thread"])
        persisted = json.loads((
            self.workspace / common.ARTIFACT_DIRNAME / "receipt.json"
        ).read_text(encoding="utf-8"))
        self.assertIn("direction_normalization", json.dumps(persisted))

    def test_text_model_and_campaign_argv_preserve_exact_model_and_budget(self):
        model = FakeModel(
            workspace=self.workspace,
            budget=common.ControllerBudget(2.0, 7200))
        result = G.GeakTextModel(model).generate([{"content": "prompt"}])
        self.assertIn("thought", result)
        argv = G.campaign_argv("python-fixture")
        self.assertEqual(argv[0], "python-fixture")
        self.assertEqual(argv[argv.index("--checkpoint-hours") + 1], "32")
        self.assertEqual(argv[argv.index("--timeout-seconds") + 1], "115200")

    def test_runtime_root_escape_is_refused(self):
        evaluator = FakeEvaluator(workspace=self.workspace, arena_root=self.arena)
        with self.assertRaisesRegex(G.GeakArenaError, "workspace child"):
            G.GeakArenaDataset(
                prompt="optimize", evaluator=evaluator,
                runtime_root=self.root / "outside")


if __name__ == "__main__":
    unittest.main()
