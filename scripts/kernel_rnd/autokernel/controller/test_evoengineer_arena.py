#!/usr/bin/env python3
"""CPU-only unit tests for the executable EvoEngineer Arena controller."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from . import arena_upstream_common as common
from . import evoengineer_arena as E


class FixtureOperator:
    def __init__(self, name, selection_size):
        self.name = name
        self.selection_size = selection_size


class FixtureEvaluationResult:
    def __init__(self, valid, score, additional_info):
        self.valid = valid
        self.score = score
        self.additional_info = additional_info


class FixtureSolution:
    def __init__(self, sol_string, other_info=None, evaluation_res=None):
        self.sol_string = sol_string
        self.other_info = other_info
        self.evaluation_res = evaluation_res


class FixtureFullInterface:
    def parse_response(self, response):
        code = re.search(r"```triton\n(.*?)```", response, re.DOTALL)
        name = re.search(r"^name:\s*([^\n]+)", response, re.MULTILINE)
        thought = re.search(r"^thought:\s*(.*?)$", response,
                            re.MULTILINE | re.DOTALL)
        return FixtureSolution(
            code.group(1).strip() if code else "",
            other_info={
                "name": name.group(1).strip() if name else "extracted",
                "thought": thought.group(1).strip() if thought else "",
            })


class FixtureConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def task(self):
        return self.interface.task


class FixtureController:
    def __init__(self, config):
        self.config = config
        self.run_state_dict = type(
            "RunState", (), {"generation": 3, "tot_sample_nums": 9})()

    def run(self):
        self.config.task.evaluate_code("candidate")


TYPES = E.UpstreamTypes(
    controller_type=FixtureController, config_type=FixtureConfig,
    operator_type=FixtureOperator,
    solution_type=FixtureSolution,
    evaluation_result_type=FixtureEvaluationResult,
    full_interface_type=FixtureFullInterface)


class FixtureEvaluator:
    last = None

    def __init__(self, workspace, arena_root=None, *, source_paths=("kernel.py",)):
        type(self).last = self
        self.workspace = workspace
        self.arena_root = arena_root
        self.source_paths = tuple(source_paths)
        self.calls = []
        self.materialized = False

    def definition(self, prompt):
        return f"Arena definition: {prompt}"

    def evaluate(self, files):
        self.calls.append(dict(files))
        source = files["kernel.py"]
        passed = "invalid" not in source
        speedup = 1.25 if "candidate" in source else 1.0
        return common.EvaluationRecord(
            passed=passed, latency_ms=0.8 if passed else None,
            speedup=speedup if passed else None,
            log_excerpt="fixture Arena feedback",
            raw={"pass_compilation": passed, "pass_correctness": passed})

    def materialize_best(self):
        self.materialized = True

    def receipt_fields(self):
        return {"brokered_evaluation_count": len(self.calls)}


class FixtureModel:
    def __init__(self, workspace=None, budget=None):
        self.workspace = workspace
        self.budget = budget
        self.prompts = []
        self.artifact_root = workspace / common.ARTIFACT_DIRNAME if workspace else None

    def call(self, prompt):
        self.prompts.append(prompt)
        return "name: candidate\ncode:\n```triton\ncandidate\n```\nthought: wave64"

    def identity(self):
        return {"model": common.MODEL_ID, "effort": common.MODEL_EFFORT}


class EvoEngineerArenaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        (self.workspace / "kernel.py").write_text(
            "baseline", encoding="utf-8")
        self.source_env = mock.patch.dict(os.environ, {
            common.ARENA_SOURCE_PATHS_ENV: json.dumps(["kernel.py"])})
        self.source_env.start()
        self.addCleanup(self.source_env.stop)
        self.evaluator = FixtureEvaluator(self.workspace)
        self.task = E.EvoEngineerArenaTask(
            prompt="optimize", evaluator=self.evaluator, types=TYPES)
        self.interface = E.EvoEngineerFullArenaInterface(
            task=self.task, types=TYPES)

    def test_exact_paper_source_and_named_full_policy_are_pinned(self):
        self.assertEqual(
            E.SOURCE_COMMIT,
            "1649715a975b9022c84b5279c88aaef0b73b28dc")
        self.assertEqual(
            E.EXPECTED_SOURCE_SHA256["LICENSE"],
            "3a18133891b736252655b83391edfef51bd52aa317198fcc4374eb5f16e99de3")
        policy = E.policy_identity()
        self.assertEqual(policy["variant"], "EvoEngineer-Full")
        self.assertEqual(policy["population_size"], 4)
        self.assertEqual(policy["max_generations"], 10)
        self.assertEqual(policy["max_sample_nums"], 45)
        self.assertEqual(policy["init_operators"], [["init", 0]])
        self.assertEqual(
            policy["offspring_operators"],
            [["crossover", 2], ["mutation", 1]])

    def test_source_admission_checks_every_policy_bearing_digest(self):
        receipt = {
            "name": "fixture", "commit": E.SOURCE_COMMIT, "clean": True,
            "required_file_sha256": dict(E.EXPECTED_SOURCE_SHA256),
        }
        with mock.patch.object(
                E.arena_adapter, "inspect_vendor_source", return_value=receipt):
            admitted = E.inspect_source(self.workspace)
        self.assertEqual(admitted["variant"], E.VARIANT)
        self.assertEqual(
            admitted["release_asset_sha256"], E.SOURCE_RELEASE_ASSET_SHA256)

        bad = {**receipt, "required_file_sha256": {
            **E.EXPECTED_SOURCE_SHA256, "LICENSE": "0" * 64}}
        with (
            mock.patch.object(
                E.arena_adapter, "inspect_vendor_source", return_value=bad),
            self.assertRaisesRegex(E.EvoEngineerArenaError, "digest mismatch"),
        ):
            E.inspect_source(self.workspace)

    def test_full_interface_retains_operator_selection_sizes(self):
        init = self.interface.get_init_operators()
        offspring = self.interface.get_offspring_operators()
        self.assertEqual([(row.name, row.selection_size) for row in init],
                         [("init", 0)])
        self.assertEqual(
            [(row.name, row.selection_size) for row in offspring],
            [("crossover", 2), ("mutation", 1)])
        self.assertEqual(self.interface.valid_require, 2)

    def test_arena_feedback_is_the_only_candidate_score(self):
        baseline = self.interface.make_init_sol()
        candidate = self.task.evaluate_code("candidate")
        invalid = self.task.evaluate_code("invalid")
        self.assertTrue(baseline.evaluation_res.valid)
        self.assertEqual(baseline.evaluation_res.score, 1.0)
        self.assertTrue(candidate.valid)
        self.assertEqual(candidate.score, 1.25)
        self.assertFalse(invalid.valid)
        self.assertIsNone(invalid.score)
        self.assertEqual(len(self.evaluator.calls), 3)

    def test_amd_prompt_preserves_full_crossover_and_insight_context(self):
        best = self.interface.make_init_sol()
        parents = [
            FixtureSolution(
                f"candidate-{index}",
                other_info={"name": f"p{index}", "thought": f"idea-{index}"},
                evaluation_res=FixtureEvaluationResult(True, 1.1 + index, {}))
            for index in range(2)
        ]
        messages = self.interface.get_operator_prompt(
            "crossover", parents, best, ["one", "two", "three", "four"])
        prompt = messages[0]["content"]
        self.assertIn("AMD Instinct MI210 gfx90a wave64", prompt)
        self.assertIn("candidate-0", prompt)
        self.assertIn("candidate-1", prompt)
        self.assertIn("- three", prompt)
        self.assertNotIn("- four", prompt)
        self.assertIn("complete replacement", prompt)
        self.assertNotIn("PYBIND11_MODULE", prompt)

    def test_model_and_parser_bridge_complete_source_without_execution(self):
        model = FixtureModel()
        bridge = E.EvoEngineerTextModel(model)
        response, usage = bridge.get_response(
            [{"role": "user", "content": "optimize"}])
        solution = self.interface.parse_response(response)
        self.assertEqual(solution.sol_string, "candidate")
        self.assertEqual(solution.other_info["name"], "candidate")
        self.assertEqual(solution.other_info["thought"], "wave64")
        self.assertEqual(usage["model"], common.MODEL_ID)
        self.assertEqual(len(model.prompts), 1)

    def test_parser_retains_unfenced_code_section_fallback(self):
        solution = self.interface.parse_response(
            "name: unfenced\ncode:\ncandidate\nthought: wave64")
        self.assertEqual(solution.sol_string, "candidate")
        self.assertEqual(solution.other_info["name"], "unfenced")

    def test_builder_assembles_exact_full_parameters_but_does_not_run(self):
        model = FixtureModel()
        controller = E.build_controller(
            prompt="optimize", evaluator=self.evaluator, model=model,
            output_path=self.workspace, types=TYPES)
        config = controller.config
        self.assertEqual(config.max_generations, 10)
        self.assertEqual(config.max_sample_nums, 45)
        self.assertEqual(config.pop_size, 4)
        self.assertEqual(config.num_samplers, 4)
        self.assertEqual(config.num_evaluators, 4)
        self.assertIsInstance(config.interface, E.EvoEngineerFullArenaInterface)
        self.assertIsInstance(config.running_llm, E.EvoEngineerTextModel)
        self.assertEqual(self.evaluator.calls, [])
        self.assertEqual(model.prompts, [])

    def test_run_controller_executes_upstream_loop_only_through_evaluator(self):
        arena_root = self.workspace / "arena"
        arena_root.mkdir()
        source_root = self.workspace / "source"
        source_root.mkdir()
        entrypoint = source_root / E.UPSTREAM_ENTRYPOINT
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text("fixture", encoding="utf-8")
        source_receipt = {
            "commit": E.SOURCE_COMMIT,
            "required_file_sha256": dict(E.EXPECTED_SOURCE_SHA256),
        }
        with (
            mock.patch.object(E, "inspect_source", return_value=source_receipt),
            mock.patch.object(common, "_atomic_json"),
        ):
            receipt = E.run_controller(
                prompt="optimize", workspace=self.workspace,
                arena_root=arena_root, source_root=source_root,
                budget=common.ControllerBudget(2, 7200),
                model_factory=FixtureModel,
                evaluator_factory=FixtureEvaluator,
                upstream_loader=lambda unused: TYPES)
        self.assertEqual(receipt["controller_id"], E.CONTROLLER_ID)
        self.assertEqual(receipt["source"]["commit"], E.SOURCE_COMMIT)
        self.assertEqual(receipt["evaluation"]["brokered_evaluation_count"], 1)
        self.assertEqual(receipt["extra"]["policy"], E.policy_identity())
        self.assertEqual(receipt["extra"]["generation"], 3)
        self.assertEqual(receipt["extra"]["sample_count"], 9)
        self.assertEqual(FixtureEvaluator.last.source_paths, ("kernel.py",))

    def test_campaign_argv_is_fully_pinned(self):
        argv = E.campaign_argv("/fixed/python")
        self.assertEqual(argv[:3], (
            "/fixed/python", "-m", E.EXECUTABLE_MODULE))
        self.assertEqual(argv[argv.index("--model") + 1], common.MODEL_ID)
        self.assertEqual(argv[argv.index("--effort") + 1], common.MODEL_EFFORT)
        self.assertEqual(argv[argv.index("--checkpoint-hours") + 1], "32")

    @unittest.skipUnless(E.DEFAULT_SOURCE_ROOT.is_dir(),
                         "exact admitted EvoEngineer checkout is absent")
    def test_exact_upstream_run_completes_full_policy_with_fixture_bridges(self):
        """Exercise the real historical loop without model, GPU, or inference."""
        output = self.workspace / "exact-upstream-run"
        output.mkdir()
        model = FixtureModel(self.workspace, common.ControllerBudget(2, 7200))
        evaluator = FixtureEvaluator(self.workspace)
        controller = E.build_controller(
            prompt="optimize", evaluator=evaluator, model=model,
            output_path=output, types=E.load_upstream(E.DEFAULT_SOURCE_ROOT))
        controller.run()
        self.assertTrue(controller.run_state_dict.is_done)
        self.assertEqual(controller.run_state_dict.generation, E.MAX_GENERATIONS)
        self.assertEqual(controller.run_state_dict.tot_sample_nums, 40)
        self.assertEqual(len(model.prompts), 40)
        self.assertEqual(len(evaluator.calls), 41)


if __name__ == "__main__":
    unittest.main()
