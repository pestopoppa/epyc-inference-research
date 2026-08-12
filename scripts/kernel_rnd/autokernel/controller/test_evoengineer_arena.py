#!/usr/bin/env python3
"""CPU-only unit tests for the pending EvoEngineer Arena policy seam."""

from __future__ import annotations

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


class FixtureController:
    def __init__(self, config):
        self.config = config


TYPES = E.UpstreamTypes(
    controller_type=FixtureController, config_type=FixtureConfig,
    operator_type=FixtureOperator,
    solution_type=FixtureSolution,
    evaluation_result_type=FixtureEvaluationResult,
    full_interface_type=FixtureFullInterface)


class FixtureEvaluator:
    def __init__(self, workspace):
        self.workspace = workspace
        self.source_paths = ("kernel.py",)
        self.calls = []

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


class FixtureModel:
    def __init__(self):
        self.prompts = []

    def call(self, prompt):
        self.prompts.append(prompt)
        return "name: candidate\ncode:\n```triton\ncandidate\n```\nthought: wave64"


class EvoEngineerArenaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        (self.workspace / "kernel.py").write_text(
            "baseline", encoding="utf-8")
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

    def test_runtime_remains_explicitly_refused(self):
        with self.assertRaisesRegex(
                E.EvoEngineerArenaError, "source-admitted but not executable"):
            E.execution_refusal()
        self.assertIn(
            "ArenaWorkspaceEvaluator.evaluate(files)",
            " ".join(E.PENDING_RUNTIME_DEPENDENCIES))

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


if __name__ == "__main__":
    unittest.main()
