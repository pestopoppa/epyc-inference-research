from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest

from . import authoring_contract as A
from . import loop_experiment_runner as R
from . import loop_experiments as L


def sha(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def context() -> A.PricedContext:
    return A.price_context(
        round_id="ak-le-runner-round",
        budget=A.ContextBudget(max_total_tokens=128, max_item_tokens=128, max_items=1),
        items=(A.ContextItem(
            source_ref="profile://fixed", purpose="fixed retrieval",
            content="unpack control dominates"),))


def role(name: str, seconds: int) -> L.RoleBudget:
    instruction = f"Run the fixed {name} stage."
    return L.RoleBudget(name, seconds, instruction, sha(instruction))


def contract(ctx: A.PricedContext) -> L.ExperimentContract:
    propose = "PROPOSE one falsifiable kernel hypothesis; do not implement it yet."
    selected = "Implement the independently selected fixed hypothesis."
    planners = tuple(L.PlannerArm(
        f"plan-{model}-{effort}-{mode}", model, "provider-native", effort,
        L.TARGET_ABSENT if mode == "control" else L.TARGET_RENDERED)
        for model in ("claude-opus-5", "gpt-5.6-sol")
        for effort in ("high", "xhigh") for mode in ("control", "target"))
    predictions = tuple(L.DirectionPrediction(
        model, "provider-native", "higher_effort_increases_search_persistence",
        "Higher effort should delay an already-optimized stop.")
        for model in ("claude-opus-5", "gpt-5.6-sol"))
    scaffolds = tuple(arm for model in ("claude-opus-5", "gpt-5.6-sol")
                      for arm in (
        L.ScaffoldArm(
            f"scaffold-{model}-direct", model, "provider-native", "high",
            L.SCAFFOLD_DIRECT, (role("implement", 120),)),
        L.ScaffoldArm(
            f"scaffold-{model}-split", model, "provider-native", "high",
            L.SCAFFOLD_SPLIT, (role("implement", 60), role("exploit", 60))),
    ))
    return L.ExperimentContract(
        "ak-le-runner-fixture-v1",
        L.FixedPromptFrame(
            L.ArtifactPin("champion://fixed", sha("champion")),
            L.context_sha256(ctx), propose, sha(propose),
            L.SelectedTaskArtifact("hypothesis://fixed", selected, sha(selected))),
        planners, predictions, scaffolds, (),
        L.ArtifactPin("prefilter://fixed", sha("prefilter")))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("claude", "codex"):
            path = self.bin / name
            path.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        self.ctx = context()
        self.spec = contract(self.ctx)
        self.pins = tuple(R.ModelCellPin(
            "claude" if arm.model_id.startswith("claude") else "codex",
            arm.model_id, arm.quant_id, arm.effort,
            str(self.bin / ("claude" if arm.model_id.startswith("claude") else "codex")),
            sha((self.bin / ("claude" if arm.model_id.startswith("claude") else "codex")).read_bytes()))
            for arm in self.spec.planner_arms[::2])
        self.targets = {
            arm.cell_id: "PROXIMATE AUTHORING TARGET: recover one bounded rung"
            for arm in self.spec.planner_arms
            if arm.target_context_mode == L.TARGET_RENDERED}

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        return R.compile_planner_manifest(
            self.spec, context=self.ctx, target_lines=self.targets,
            model_pins=self.pins, timeout_seconds=30)


class ManifestTest(Fixture):
    def test_manifest_covers_exact_factorial_cells_without_execution_authority(self):
        manifest = self.manifest()
        self.assertEqual(len(manifest["cells"]), 8)
        self.assertEqual(manifest["authority"], R.AUTHORITY)
        self.assertFalse(manifest["constraints"]["scaffold_execution_supported"])
        self.assertFalse(manifest["constraints"]["model_or_kernel_invoked_by_compiler"])
        self.assertEqual(manifest, R.validate_manifest(manifest))

    def test_cli_argv_are_exact_read_only_model_and_effort_cells(self):
        manifest = self.manifest()
        claude = next(row for row in manifest["cells"] if row["provider"] == "claude")
        codex = next(row for row in manifest["cells"] if row["provider"] == "codex")
        self.assertIn("--permission-mode", claude["argv_template"])
        self.assertIn("Bash,Edit,Write,NotebookEdit", claude["argv_template"])
        schema_index = claude["argv_template"].index("--json-schema")
        self.assertEqual(
            json.loads(claude["argv_template"][schema_index + 1]),
            R._RAW_OBSERVATION_JSON_SCHEMA)
        effort_index = claude["argv_template"].index("--effort")
        self.assertEqual(claude["argv_template"][effort_index + 1], claude["effort"])
        self.assertIn("read-only", codex["argv_template"])
        self.assertIn(f'model_reasoning_effort="{codex["effort"]}"',
                      codex["argv_template"])
        self.assertNotIn("danger-full-access", codex["argv_template"])

    def test_target_value_occurs_only_in_rendered_prompt_cells(self):
        manifest = self.manifest()
        for cell in manifest["cells"]:
            present = "recover one bounded rung" in cell["prompt"]
            self.assertEqual(present, cell["target_context_mode"] == L.TARGET_RENDERED)
        control = next(row for row in manifest["cells"]
                       if row["target_context_mode"] == L.TARGET_ABSENT)
        target = next(row for row in manifest["cells"]
                      if (row["model_id"], row["quant_id"], row["effort"]) ==
                      (control["model_id"], control["quant_id"], control["effort"])
                      and row["target_context_mode"] == L.TARGET_RENDERED)
        self.assertEqual(
            target["prompt"].replace(
                "\nPROXIMATE AUTHORING TARGET: recover one bounded rung", ""),
            control["prompt"])

    def test_missing_target_pin_or_cli_drift_refuses(self):
        with self.assertRaisesRegex(R.LoopRunnerError, "every rendered"):
            R.compile_planner_manifest(
                self.spec, context=self.ctx,
                target_lines=dict(list(self.targets.items())[1:]),
                model_pins=self.pins, timeout_seconds=30)
        manifest = self.manifest()
        manifest["cells"][0]["argv_template"].append("--unsafe")
        manifest["manifest_sha256"] = R._digest({
            key: value for key, value in manifest.items() if key != "manifest_sha256"})
        with self.assertRaisesRegex(R.LoopRunnerError, "argv template"):
            R.validate_manifest(manifest)

    def test_manifest_hash_and_binary_hash_are_fail_closed(self):
        manifest = self.manifest()
        manifest["cells"][0]["prompt"] += " drift"
        with self.assertRaisesRegex(R.LoopRunnerError, "manifest SHA"):
            R.validate_manifest(manifest)
        with self.assertRaisesRegex(R.LoopRunnerError, "identity"):
            replace(self.pins[0], executable_sha256=sha("wrong"))

    def test_write_manifest_requires_new_absolute_path(self):
        path = self.root / "manifest.json"
        R.write_manifest(path, self.manifest())
        self.assertEqual(json.loads(path.read_text()), self.manifest())
        with self.assertRaisesRegex(R.LoopRunnerError, "new absolute"):
            R.write_manifest(path, self.manifest())


def raw(_cell_id: str) -> str:
    return json.dumps({
        "schema": R.RAW_OBSERVATION_SCHEMA,
        "termination": "already_optimized",
        "hypotheses": [{
            "mechanism": "branchless unpack", "target_surface": "Q4_K decode",
            "falsifiable_counter": "VALU per wave falls",
            "predicted_direction": "lower",
        }],
    })


class FakeRunner:
    def __init__(self, *, malformed: bool = False, timed_out: bool = False,
                 mutate: bool = False):
        self.calls = []
        self.malformed = malformed
        self.timed_out = timed_out
        self.mutate = mutate

    def __call__(self, argv, cwd, environment, prompt, timeout, result_path):
        cell_id = cwd.name.split("-", 1)[1]
        result = "not json" if self.malformed else raw(cell_id)
        if result_path is not None:
            result_path.write_text(result, encoding="utf-8")
        if self.mutate:
            (cwd / "unauthorized.txt").write_text("mutation", encoding="utf-8")
        self.calls.append((tuple(argv), cwd, prompt, timeout, result_path))
        stdout = "event stream" if result_path is not None else result
        return R.ProcessCapture(
            tuple(argv), 0, stdout, "diagnostic", result,
            self.timed_out, "2026-08-12T00:00:00+00:00",
            "2026-08-12T00:00:01+00:00", 1.0)


class RunnerTest(Fixture):
    def test_claude_structured_output_wrapper_is_admitted_without_markdown_repair(self):
        expected = json.loads(raw("ignored"))
        wrapper = json.dumps({
            "type": "result", "subtype": "success",
            "structured_output": expected,
            "result": "```json\\nnot admitted\\n```",
        })
        parsed = R.parse_raw_observation(
            wrapper, provider="claude", expected_cell_id="plan-claude-high-control")
        self.assertEqual(parsed["schema"], R.RAW_OBSERVATION_SCHEMA)
        self.assertEqual(parsed["cell_id"], "plan-claude-high-control")

    def test_complete_panel_retains_all_hash_bound_artifacts_and_observations(self):
        fake = FakeRunner()
        output = self.root / "panel"
        panel = R.run_planner_manifest(
            self.manifest(), output_root=output, environment={"PATH": str(self.bin)},
            runner=fake)
        self.assertEqual(panel["status"], "complete")
        self.assertEqual(len(panel["observations"]), 8)
        self.assertFalse(panel["constraints"]["external_prefilter_applied"])
        for ordinal, row in enumerate(panel["observations"], 1):
            cell = output / f"{ordinal:04d}-{row['cell_id']}"
            for name in ("prompt.txt", "stdout.txt", "stderr.txt", "result.txt",
                         "observation.json", "event.json"):
                self.assertTrue((cell / name).is_file())
            self.assertEqual(row["prompt_sha256"], sha((cell / "prompt.txt").read_bytes()))
        self.assertEqual(json.loads((output / "panel.json").read_text()), panel)

    def test_codex_result_path_and_claude_stdout_boundary_are_predeclared(self):
        fake = FakeRunner()
        R.run_planner_manifest(
            self.manifest(), output_root=self.root / "panel", runner=fake)
        for argv, _, _, _, result_path in fake.calls:
            if Path(argv[0]).name == "codex":
                self.assertIsNotNone(result_path)
                self.assertIn("--output-last-message", argv)
            else:
                self.assertIsNone(result_path)

    def test_real_captured_process_boundary_with_fake_clis_only(self):
        fake_cli = """#!/usr/bin/env python3
import json
from pathlib import Path
import sys
prompt = sys.stdin.read()
result = json.dumps({
    "schema": "epyc.autokernel.loop_experiment_raw_planner.v1",
    "termination": "search_exhausted",
    "hypotheses": [],
})
if "--output-last-message" in sys.argv:
    output = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
    output.write_text(result, encoding="utf-8")
    print(json.dumps({"type": "turn.completed"}))
else:
    print(json.dumps({"result": result}))
"""
        for name in ("claude", "codex"):
            executable = self.bin / name
            executable.write_text(fake_cli, encoding="utf-8")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        pins = tuple(R.resolve_model_pin(
            provider="claude" if arm.model_id.startswith("claude") else "codex",
            model_id=arm.model_id, quant_id=arm.quant_id, effort=arm.effort,
            environment={"PATH": str(self.bin)})
            for arm in self.spec.planner_arms[::2])
        manifest = R.compile_planner_manifest(
            self.spec, context=self.ctx, target_lines=self.targets,
            model_pins=pins, timeout_seconds=10)
        panel = R.run_planner_manifest(
            manifest, output_root=self.root / "subprocess-panel",
            environment={"PATH": "/usr/bin:/bin"})
        self.assertEqual(panel["status"], "complete")
        self.assertTrue(all(
            row["observation"]["termination"] == "search_exhausted"
            for row in panel["observations"]))

    def test_malformed_or_timed_out_cell_fails_with_durable_terminal_record(self):
        for name, fake, message in (
                ("malformed", FakeRunner(malformed=True), "malformed JSON"),
                ("timeout", FakeRunner(timed_out=True), "timed out")):
            output = self.root / name
            with self.subTest(name=name), self.assertRaisesRegex(R.LoopRunnerError, message):
                R.run_planner_manifest(self.manifest(), output_root=output, runner=fake)
            terminal = json.loads((output / "panel.json").read_text())
            self.assertEqual(terminal["status"], "failed")
            self.assertTrue((next(output.glob("0001-*")) / "event.json").is_file())

    def test_existing_or_relative_output_root_refuses_before_any_call(self):
        fake = FakeRunner()
        with self.assertRaisesRegex(R.LoopRunnerError, "new absolute"):
            R.run_planner_manifest(self.manifest(), output_root="relative", runner=fake)
        self.assertEqual(fake.calls, [])

    def test_read_only_boundary_refuses_undeclared_cli_writes(self):
        output = self.root / "mutated"
        with self.assertRaisesRegex(R.LoopRunnerError, "undeclared paths"):
            R.run_planner_manifest(
                self.manifest(), output_root=output, runner=FakeRunner(mutate=True))
        terminal = json.loads((output / "panel.json").read_text())
        self.assertEqual(terminal["status"], "failed")

    def test_strict_observation_rejects_unknown_fields_and_binds_trusted_cell(self):
        payload = json.loads(raw("plan-a"))
        payload["authority"] = "rank"
        with self.assertRaisesRegex(R.LoopRunnerError, "unknown"):
            R.parse_raw_observation(
                json.dumps(payload), provider="codex", expected_cell_id="plan-a")
        parsed = R.parse_raw_observation(
            raw("ignored"), provider="codex", expected_cell_id="plan-a")
        self.assertEqual(parsed["cell_id"], "plan-a")
        self.assertNotIn("plan-a", raw("ignored"))

    def test_claude_wrapper_is_parsed_strictly(self):
        wrapped = json.dumps({"result": raw("plan-a"), "usage": {"input": 1}})
        parsed = R.parse_raw_observation(
            wrapped, provider="claude", expected_cell_id="plan-a")
        self.assertEqual(parsed["termination"], "already_optimized")

    def test_external_prefilter_binding_materializes_existing_contract_type(self):
        parsed = R.parse_raw_observation(
            raw("plan-a"), provider="codex", expected_cell_id="plan-a")
        observation = R.materialize_planner_observation(
            parsed, survived_prefilter=(True,), elapsed_wall_seconds=2,
            evidence_sha256=sha("prefilter evidence"), provider="codex")
        self.assertIsInstance(observation, L.PlannerObservation)
        self.assertTrue(observation.hypotheses[0].survived_prefilter)
        with self.assertRaisesRegex(R.LoopRunnerError, "one boolean"):
            R.materialize_planner_observation(
                parsed, survived_prefilter=(), elapsed_wall_seconds=2,
                evidence_sha256=sha("prefilter evidence"), provider="codex")

    def test_scaffold_execution_routes_to_the_governed_writer_seam(self):
        with self.assertRaisesRegex(Exception, "manifest SHA-256"):
            R.run_scaffold_manifest(
                {"manifest": "unbound"}, output_root="/disposable/output")
        self.assertIn("loop_scaffold_runner", R.SCAFFOLD_GAP)


if __name__ == "__main__":
    unittest.main()
