from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from . import authoring_contract as A
from . import codex_container_actor as C
from . import loop_experiments as L
from . import loop_scaffold_runner as R


def sha(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args), capture_output=True, text=True,
        check=True)
    return result.stdout.strip()


def commit_all(repo: Path) -> str:
    git(repo, "add", ".")
    git(repo, "-c", "user.name=AKLE Test", "-c", "user.email=akle@example.invalid",
        "commit", "-m", "fixture")
    return git(repo, "rev-parse", "HEAD")


def context() -> A.PricedContext:
    return A.price_context(
        round_id="ak-le-3-fixed",
        budget=A.ContextBudget(max_total_tokens=128, max_item_tokens=128, max_items=1),
        items=(A.ContextItem(
            source_ref="profile://fixed", purpose="fixed context",
            content="The selected source surface is exact."),))


def role(name: str, seconds: int) -> L.RoleBudget:
    instruction = f"Perform the fixed {name} stage."
    return L.RoleBudget(name, seconds, instruction, sha(instruction))


def contract(ctx: A.PricedContext) -> L.ExperimentContract:
    propose = "PROPOSE a bounded hypothesis."
    selected = "Implement exactly the selected source change."
    planners = tuple(L.PlannerArm(
        f"plan-{model}-{effort}-{mode}", model, "native", effort,
        L.TARGET_ABSENT if mode == "control" else L.TARGET_RENDERED)
        for model in C.SUPPORTED_MODELS for effort in ("high", "xhigh")
        for mode in ("control", "target"))
    predictions = tuple(L.DirectionPrediction(
        model, "native", "higher_effort_increases_search_persistence", "fixed")
        for model in C.SUPPORTED_MODELS)
    scaffolds = tuple(arm for model in C.SUPPORTED_MODELS for arm in (
        L.ScaffoldArm(f"scaffold-{model}-direct", model, "provider-native", "high",
                      L.SCAFFOLD_DIRECT, (role("implement", 4),)),
        L.ScaffoldArm(f"scaffold-{model}-split", model, "provider-native", "high",
                      L.SCAFFOLD_SPLIT,
                      (role("implement", 2), role("exploit", 2))),
    ))
    return L.ExperimentContract(
        "ak-le-3-fixture-v1",
        L.FixedPromptFrame(
            L.ArtifactPin("champion://exact", sha("champion")),
            L.context_sha256(ctx), propose, sha(propose),
            L.SelectedTaskArtifact("task://exact", selected, sha(selected))),
        planners, predictions, scaffolds, (),
        L.ArtifactPin("prefilter://exact", sha("prefilter")))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_repo = self.root / "source"
        self.source_repo.mkdir()
        git(self.source_repo, "init", "-q")
        (self.source_repo / "src").mkdir()
        (self.source_repo / "src" / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.source_commit = commit_all(self.source_repo)
        self.source = R.SourcePin(
            str(self.source_repo), self.source_commit,
            R._tree_digest(self.source_repo, self.source_commit))

        self.arena = Path(
            "/mnt/raid0/llm/tmp/inf03-vendor-inspect-UqTLqw/AgentKernelArena")
        task_relative = "tasks/hip2hip/gpumode/SimpleMatmulModule"
        task = self.arena / task_relative
        expected_python = R.arena_cell_runner._evaluator_python_identity()
        python = Path(expected_python["path"])
        self.evaluator = R.ArenaEvaluatorPin(
            str(self.arena), R.arena_adapter.AGENT_KERNEL_ARENA_PIN.commit,
            "src/evaluator.py",
            sha((self.arena / "src" / "evaluator.py").read_bytes()),
            task_relative, R._digest(R._tree_state(task)),
            sha((task / "config.yaml").read_bytes()), str(python),
            expected_python["sha256"], R._digest(expected_python["packages"]),
            str(R.REPOSITORY_ROOT), sha(R.EVALUATOR_BRIDGE.read_bytes()))

        self.wrapper = Path(subprocess.run(
            ("bash", "-lc", "command -v codex"), capture_output=True,
            text=True, check=True).stdout.strip()).resolve()
        runtime = C.runtime_identity(self.wrapper)
        actor_python = python.resolve()
        self.ctx = context()
        self.spec = contract(self.ctx)
        self.actors = tuple(R.ActorPin(
            model, "provider-native", "high", str(actor_python),
            sha(actor_python.read_bytes()),
            str(self.wrapper), sha(self.wrapper.read_bytes()),
            sha(Path(C.__file__).read_bytes()), R._digest(runtime))
            for model in C.SUPPORTED_MODELS)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        return R.compile_manifest(
            self.spec, context=self.ctx, source=self.source, actors=self.actors,
            evaluator=self.evaluator, allowed_write_paths=("src/kernel.py",))

    def compile_input(self):
        return {
            "schema": R.COMPILE_INPUT_SCHEMA,
            "experiment_contract": self.spec.to_manifest(),
            "context": {
                "round_id": self.ctx.round_id,
                "budget": {
                    "max_total_tokens": self.ctx.budget.max_total_tokens,
                    "max_item_tokens": self.ctx.budget.max_item_tokens,
                    "max_items": self.ctx.budget.max_items,
                },
                "items": [{
                    "source_ref": item.source_ref, "purpose": item.purpose,
                    "content": item.content, "bulk_read": item.bulk_read,
                } for item in self.ctx.items],
            },
            "source": self.source.to_dict(),
            "actors": [actor.to_dict() for actor in self.actors],
            "evaluator": self.evaluator.to_dict(),
            "allowed_write_paths": ["src/kernel.py"],
        }


class ManifestTest(Fixture):
    def test_strict_json_compile_input_reproduces_manifest(self):
        compiled = R.compile_manifest_input(self.compile_input())
        self.assertEqual(compiled, R.validate_manifest(compiled))
        self.assertEqual(compiled["experiment_id"], self.spec.experiment_id)
        self.assertEqual(
            [cell["cell_id"] for cell in compiled["cells"]],
            [arm.cell_id for arm in self.spec.scaffold_arms])
        malformed = self.compile_input()
        malformed["ignored"] = True
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "must contain exactly"):
            R.compile_manifest_input(malformed)

    def test_manifest_is_exact_factorial_and_has_no_decision_authority(self):
        manifest = self.manifest()
        self.assertEqual(len(manifest["cells"]), 4)
        self.assertEqual(manifest, R.validate_manifest(manifest))
        constraints = manifest["constraints"]
        self.assertTrue(constraints["same_model_within_scaffold_pair"])
        self.assertTrue(constraints["model_and_scaffold_independently_varied"])
        self.assertTrue(constraints["wall_time_matched"])
        for key in ("campaign_authority", "ranking_authority",
                    "champion_authority", "release_authority"):
            self.assertFalse(constraints[key])

    def test_exact_task_context_champion_source_actor_and_evaluator_are_bound(self):
        manifest = self.manifest()
        pins = manifest["selected_pins"]
        self.assertEqual(pins["task"], self.spec.fixed.selected_task.to_dict())
        self.assertEqual(pins["champion"], self.spec.fixed.champion.to_dict())
        self.assertEqual(pins["context_sha256"], L.context_sha256(self.ctx))
        self.assertEqual(pins["source"], self.source.to_dict())
        self.assertEqual(pins["evaluator"], self.evaluator.to_dict())
        self.assertTrue(all(cell["actor"]["boundary"] == R.REQUIRED_ACTOR_BOUNDARY
                            for cell in manifest["cells"]))

    def test_tampering_or_missing_model_cell_refuses(self):
        manifest = self.manifest()
        manifest["cells"][0]["wall_seconds"] = 5.0
        manifest["manifest_sha256"] = R._digest({
            key: value for key, value in manifest.items() if key != "manifest_sha256"})
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "wall_seconds drifted"):
            R.validate_manifest(manifest)
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "exactly cover"):
            R.compile_manifest(
                self.spec, context=self.ctx, source=self.source,
                actors=self.actors[:-1], evaluator=self.evaluator,
                allowed_write_paths=("src/kernel.py",))

    def test_production_source_and_broad_or_escaping_scope_refuse(self):
        production = R._PRODUCTION_TREES[0]
        if production.is_dir():
            with self.assertRaisesRegex(R.ScaffoldRunnerError, "production"):
                R.SourcePin(str(production), "0" * 40, "0" * 64)
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "normalized"):
            R.compile_manifest(
                self.spec, context=self.ctx, source=self.source, actors=self.actors,
                evaluator=self.evaluator, allowed_write_paths=("../escape",))


class FakeActor:
    def __init__(self, unauthorized: bool = False):
        self.ordinal = 0
        self.unauthorized = unauthorized

    def __call__(self, argv, cwd, environment, prompt, timeout):
        self.ordinal += 1
        if environment.get("PYTHONPATH") != str(R.REPOSITORY_ROOT):
            raise AssertionError("actor import root was not pinned")
        target = cwd / ("outside.txt" if self.unauthorized else "src/kernel.py")
        target.write_text(
            target.read_text(encoding="utf-8") + f"# role {self.ordinal}\n"
            if target.exists() else "unauthorized\n", encoding="utf-8")
        return R.ProcessCapture(
            tuple(argv), 1000 + self.ordinal, 1000 + self.ordinal, 0,
            "actor stdout", "actor stderr", False,
            "2026-08-12T00:00:00+00:00", "2026-08-12T00:00:01+00:00",
            min(1.0, timeout), (1000 + self.ordinal,), ())


def fake_evaluator(request, cell_root, environment, timeout):
    cell = request["cell_id"]
    result = {
        "schema": R.EVALUATION_SCHEMA, "authority": R.AUTHORITY,
        "cell_id": cell, "pass_compilation": True, "pass_correctness": True,
        "valid_baseline_cases": 4, "valid_optimized_cases": 4,
        "average_speedup": 1.01,
        "campaign_authority": False, "ranking_authority": False,
        "champion_authority": False, "release_authority": False,
    }
    capture = R.ProcessCapture(
        ("arena-evaluator",), 9000, 9000, 0, "", "", False,
        "2026-08-12T00:00:00+00:00", "2026-08-12T00:00:01+00:00",
        1.0, (9000,), ())
    return capture, result


class ExecutionTest(Fixture):
    def test_complete_panel_uses_fresh_worktrees_and_seals_every_role(self):
        output = self.root / "output"
        panel = R.run_manifest(
            self.manifest(), output_root=output, actor_runner=FakeActor(),
            evaluator_runner=fake_evaluator, fixture_mode=True)
        self.assertEqual(panel["status"], "complete")
        self.assertEqual(len(panel["cells"]), 4)
        for cell in panel["cells"]:
            root = next((output / "cells").glob(f"*-{cell['cell_id']}"))
            self.assertFalse((root / "baseline-worktree").exists())
            self.assertFalse((root / "candidate-worktree").exists())
            cleanup = json.loads((root / "worktree-cleanup.json").read_text())
            self.assertTrue(cleanup["all_removed"])
            self.assertTrue((root / "arena-evaluation.json").is_file())
            for checkpoint in cell["checkpoints"]:
                self.assertTrue(checkpoint["write_scope_passed"])
                self.assertGreater(checkpoint["process"]["pid"], 1)
                self.assertEqual(checkpoint["process"]["group_members_after_reap"], [])
        self.assertEqual(json.loads((output / "panel.json").read_text()), panel)
        self.assertNotIn("belief_measurements", panel)

    def test_prospective_beliefs_preserve_cells_and_same_model_effects(self):
        panel = R.run_manifest(
            self.manifest(), output_root=self.root / "belief-fixture",
            actor_runner=FakeActor(), evaluator_runner=fake_evaluator,
            fixture_mode=True)
        for ordinal, cell in enumerate(panel["cells"], 1):
            common = {
                "schema": "epyc.autokernel.device_claim_receipt.v1",
                "claim_id": f"claim-{ordinal}", "device_id": "mi210_0",
                "campaign_id": self.manifest()["experiment_id"],
                "acquired_at": "2026-08-12T00:00:00+00:00", "state": "held",
            }
            cell["device_claim_open"] = {**common, "released_at": None}
            cell["device_claim_released"] = {
                **common, "released_at": "2026-08-12T00:00:02+00:00"}
            unsigned = dict(cell)
            unsigned.pop("cell_receipt_sha256", None)
            cell["cell_receipt_sha256"] = R._digest(unsigned)
        rows = R._belief_measurements(self.manifest(), panel["cells"])
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            len({row["measurement_id"] for row in rows}), 6)
        effects = [row for row in rows if row["metric"].startswith(
            "implement_then_exploit_over_direct")]
        self.assertEqual(len(effects), 2)
        self.assertTrue(all(row["value"] == 1.0 for row in effects))
        self.assertTrue(all(row["extra"]["diagnostic_only"] for row in rows))
        self.assertTrue(all(not row["extra"]["ranking_authority"] for row in rows))

    def test_belief_capture_refuses_unmatched_case_basis(self):
        panel = R.run_manifest(
            self.manifest(), output_root=self.root / "belief-mismatch",
            actor_runner=FakeActor(), evaluator_runner=fake_evaluator,
            fixture_mode=True)
        for ordinal, cell in enumerate(panel["cells"], 1):
            common = {
                "claim_id": f"claim-{ordinal}", "device_id": "mi210_0",
                "campaign_id": self.manifest()["experiment_id"],
                "acquired_at": "2026-08-12T00:00:00+00:00", "state": "held",
            }
            cell["device_claim_open"] = {**common, "released_at": None}
            cell["device_claim_released"] = {
                **common, "released_at": "2026-08-12T00:00:02+00:00"}
            unsigned = dict(cell)
            unsigned.pop("cell_receipt_sha256", None)
            cell["cell_receipt_sha256"] = R._digest(unsigned)
        panel["cells"][1]["evaluation"]["valid_baseline_cases"] = 3
        panel["cells"][1]["evaluation"]["valid_optimized_cases"] = 3
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "different scored-case basis"):
            R._belief_measurements(self.manifest(), panel["cells"])

    def test_undeclared_write_fails_closed_with_terminal_panel(self):
        output = self.root / "rejected"
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "write-scope audit"):
            R.run_manifest(
                self.manifest(), output_root=output,
                actor_runner=FakeActor(unauthorized=True),
                evaluator_runner=fake_evaluator, fixture_mode=True)
        terminal = json.loads((output / "panel.json").read_text())
        self.assertEqual(terminal["status"], "failed")
        self.assertFalse(terminal["constraints"]["rankable"])
        cell_root = next((output / "cells").iterdir())
        self.assertTrue(json.loads(
            (cell_root / "worktree-cleanup.json").read_text())["all_removed"])

    def test_captured_process_records_exact_pid_group_and_reaps(self):
        executable = self.root / "short.py"
        executable.write_text(
            "#!/usr/bin/env python3\nimport sys\nprint(sys.stdin.read())\n",
            encoding="utf-8")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        capture = R._run_process(
            (str(executable),), self.root, dict(os.environ), "hello", 5)
        self.assertEqual(capture.stdout.strip(), "hello")
        self.assertEqual(capture.pid, capture.process_group_id)
        self.assertEqual(capture.group_members_after_reap, ())

    def test_external_prerequisite_is_explicit_not_fake_observation(self):
        manifest = self.manifest()
        self.assertEqual(manifest["external_prerequisite"], R.EXTERNAL_PREREQUISITE)
        self.assertIn("explicitly forbidden to run model inference", R.EXTERNAL_PREREQUISITE)
        self.assertNotIn("observations", manifest)

    def test_injected_runner_without_fixture_label_refuses_before_output(self):
        output = self.root / "not-fixture"
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "explicit fixture_mode"):
            R.run_manifest(
                self.manifest(), output_root=output, actor_runner=FakeActor(),
                evaluator_runner=fake_evaluator)
        self.assertFalse(output.exists())

    def test_real_boundary_requires_device_claim_before_creating_output(self):
        output = self.root / "no-claim"
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "device claim journal"):
            R.run_manifest(self.manifest(), output_root=output)
        self.assertFalse(output.exists())

    def test_evaluator_bridge_pins_clean_pythonpath_and_module_digest(self):
        cell = self.root / "eval-cell"
        cell.mkdir()
        request = {
            "evaluator": self.evaluator.to_dict(), "cell_id": "scaffold-test",
        }
        observed = {}

        def captured(argv, cwd, environment, prompt, timeout):
            observed.update({"argv": argv, "cwd": cwd, "env": environment})
            (cell / "arena-evaluator-result.json").write_text(json.dumps({
                "schema": R.EVALUATION_SCHEMA, "authority": R.AUTHORITY,
                "cell_id": "scaffold-test"}), encoding="utf-8")
            return R.ProcessCapture(
                tuple(argv), 88, 88, 0, "", "", False, "start", "end", 1, (88,), ())

        with mock.patch.object(R, "_run_process", side_effect=captured):
            _, result = R._default_evaluator_runner(
                request, cell, {"PYTHONPATH": "/attacker"}, 10)
        self.assertEqual(observed["cwd"], R.REPOSITORY_ROOT)
        self.assertEqual(observed["env"]["PYTHONPATH"], str(R.REPOSITORY_ROOT))
        self.assertEqual(observed["env"]["HIP_VISIBLE_DEVICES"], "0")
        self.assertIn("scripts.kernel_rnd.autokernel.controller.arena_scaffold_evaluator",
                      observed["argv"])
        self.assertEqual(result["cell_id"], "scaffold-test")
        mutated = dict(self.evaluator.to_dict())
        mutated["bridge_sha256"] = "0" * 64
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "bridge module identity"):
            R.ArenaEvaluatorPin(**mutated)

    def test_arbitrary_actor_launcher_or_model_cannot_self_attest_confinement(self):
        trusted = self.actors[0].to_dict()
        trusted["model_id"] = "unreviewed-model"
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "reviewed Codex cell"):
            R.ActorPin(**trusted)
        trusted = self.actors[0].to_dict()
        trusted["launcher_sha256"] = "0" * 64
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "module identity"):
            R.ActorPin(**trusted)


if __name__ == "__main__":
    unittest.main()
