from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

from . import authoring_contract as A
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
        for model in ("model-a", "model-b") for effort in ("high", "xhigh")
        for mode in ("control", "target"))
    predictions = tuple(L.DirectionPrediction(
        model, "native", "higher_effort_increases_search_persistence", "fixed")
        for model in ("model-a", "model-b"))
    scaffolds = tuple(arm for model in ("model-a", "model-b") for arm in (
        L.ScaffoldArm(f"scaffold-{model}-direct", model, "native", "high",
                      L.SCAFFOLD_DIRECT, (role("implement", 4),)),
        L.ScaffoldArm(f"scaffold-{model}-split", model, "native", "high",
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

        self.arena = self.root / "arena"
        self.arena.mkdir()
        git(self.arena, "init", "-q")
        (self.arena / "src").mkdir()
        (self.arena / "src" / "evaluator.py").write_text("# exact evaluator\n", encoding="utf-8")
        task = self.arena / "tasks" / "fixed"
        task.mkdir(parents=True)
        (task / "config.yaml").write_text("task: fixed\n", encoding="utf-8")
        self.arena_commit = commit_all(self.arena)
        python = Path("/usr/bin/python3")
        self.evaluator = R.ArenaEvaluatorPin(
            str(self.arena), self.arena_commit, "src/evaluator.py",
            sha((self.arena / "src" / "evaluator.py").read_bytes()),
            "tasks/fixed", R._digest(R._tree_state(task)),
            sha((task / "config.yaml").read_bytes()), str(python),
            sha(python.read_bytes()), sha("pytest=x;torch=y;triton=z"))

        self.bin = self.root / "actor"
        self.bin.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        self.bin.chmod(self.bin.stat().st_mode | stat.S_IXUSR)
        self.ctx = context()
        self.spec = contract(self.ctx)
        self.actors = tuple(R.ActorPin(
            model, "native", "high", str(self.bin), sha(self.bin.read_bytes()),
            sha(f"runtime:{model}")) for model in ("model-a", "model-b"))

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        return R.compile_manifest(
            self.spec, context=self.ctx, source=self.source, actors=self.actors,
            evaluator=self.evaluator, allowed_write_paths=("src/kernel.py",))


class ManifestTest(Fixture):
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
        self.assertTrue(all(cell["actor"]["boundary"] == R.ACTOR_BOUNDARY
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
            evaluator_runner=fake_evaluator)
        self.assertEqual(panel["status"], "complete")
        self.assertEqual(len(panel["cells"]), 4)
        for cell in panel["cells"]:
            root = next((output / "cells").glob(f"*-{cell['cell_id']}"))
            self.assertEqual(git(root / "baseline-worktree", "status", "--porcelain=v1"), "")
            self.assertNotEqual(git(root / "candidate-worktree", "status", "--porcelain=v1"), "")
            self.assertTrue((root / "arena-evaluation.json").is_file())
            for checkpoint in cell["checkpoints"]:
                self.assertTrue(checkpoint["write_scope_passed"])
                self.assertGreater(checkpoint["process"]["pid"], 1)
                self.assertEqual(checkpoint["process"]["group_members_after_reap"], [])
        self.assertEqual(json.loads((output / "panel.json").read_text()), panel)

    def test_undeclared_write_fails_closed_with_terminal_panel(self):
        output = self.root / "rejected"
        with self.assertRaisesRegex(R.ScaffoldRunnerError, "write-scope audit"):
            R.run_manifest(
                self.manifest(), output_root=output,
                actor_runner=FakeActor(unauthorized=True),
                evaluator_runner=fake_evaluator)
        terminal = json.loads((output / "panel.json").read_text())
        self.assertEqual(terminal["status"], "failed")
        self.assertFalse(terminal["constraints"]["rankable"])

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
        self.assertIn("No such multi-model actor-launcher set", R.EXTERNAL_PREREQUISITE)
        self.assertNotIn("observations", manifest)


if __name__ == "__main__":
    unittest.main()
