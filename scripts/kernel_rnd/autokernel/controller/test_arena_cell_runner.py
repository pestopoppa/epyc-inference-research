#!/usr/bin/env python3
"""Tests for the governed AgentKernelArena campaign execution bridge."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from . import arena_adapter as A
from . import arena_campaign as C
from . import arena_cell_runner as R
from . import k_search_arena as KS


def canonical_sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class FakeReceipt:
    def __init__(self, phase: str):
        self.phase = phase

    def to_dict(self):
        row = {
            "schema": "epyc.autokernel.device_claim_receipt.v1",
            "claim_id": "akd-fixture0000001",
            "device_id": "mi210_0",
            "campaign_id": "fixture-campaign-v1",
            "acquired_at": "2026-08-11T00:00:00Z",
        }
        if self.phase == "released":
            row["released_at"] = "2026-08-11T00:00:01Z"
        return row


class FakeClaim:
    def __init__(self):
        self.released = False

    def receipt(self):
        return FakeReceipt("opened")

    def release(self):
        self.released = True
        return FakeReceipt("released")


class FakeSampling:
    def to_dict(self):
        return {"sample_count": 2, "interval_s": 0.25}


class FakeSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        return self

    def stop(self):
        return FakeSampling()


class ArenaCellRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.arena = self.root / "arena"
        task = self.arena / "tasks" / "fixture"
        task.mkdir(parents=True)
        (task / "config.yaml").write_text("task_type: fixture\n", encoding="utf-8")
        self.preflight_path = self.root / "preflight.json"
        preflight = {
            "schema": A.PREFLIGHT_SCHEMA,
            "authority": "diagnostic_only",
            "hardware": {"target_gfx_arch": "gfx90a", "target_gpu_model": "MI210"},
        }
        preflight["receipt_sha256"] = canonical_sha(preflight)
        self.preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
        self.output = self.root / "output"
        self.task = C.TaskArtifact(
            task_id="fixture.task", relative_root="tasks/fixture",
            file_sha256={"config.yaml": hashlib.sha256(
                (task / "config.yaml").read_bytes()).hexdigest()},
        )

    def arm(self, arm_id: str) -> C.ArmImplementation:
        if arm_id == C.BASELINE_ARM_ID:
            return C.ArmImplementation(
                arm_id, "ready", "arena_measure_baseline", ())
        if arm_id == KS.CONTROLLER_ID:
            return C.ArmImplementation(
                arm_id, "ready", "k_search_world_model_arena_v1", (),
                argv=KS.campaign_argv(sys.executable),
                source_root=str(self.root), source_commit="a" * 40,
                entrypoint_path=KS.ENTRYPOINT_RELATIVE,
                entrypoint_sha256="b" * 64,
                model_ids=KS.PINNED_MODEL_IDS, required_clis=KS.REQUIRED_CLIS,
                upstream_source_root="vendor://k-search",
                upstream_source_commit=KS.SOURCE_COMMIT,
                upstream_entrypoint_path=KS.UPSTREAM_ENTRYPOINT,
                upstream_entrypoint_sha256="c" * 64,
                upstream_license_path="LICENSE",
                upstream_license_sha256="d" * 64,
            )
        return C.ArmImplementation(
            arm_id, "ready", "stdin_workspace_v1", (),
            argv=(sys.executable, "controller.py", "--checkpoint-hours", "32",
                  "--timeout-seconds", "115200"),
            source_root=str(self.root), source_commit="a" * 40,
            entrypoint_path="controller.py", entrypoint_sha256="b" * 64,
            model_ids=("fixture-model",),
        )

    def config(self) -> R.RunnerConfig:
        return R.RunnerConfig(
            campaign_id="fixture-campaign-v1",
            arena_root=str(self.arena.resolve()),
            preflight_path=str(self.preflight_path.resolve()),
            output_root=str(self.output.resolve()),
            claim_journal=str((self.root / "claim.jsonl").resolve()),
        )

    @staticmethod
    def worker(request, timeout):
        cell_root = Path(request["cell_root"])
        artifact = cell_root / "fixture.txt"
        artifact.write_text("evidence\n", encoding="utf-8")
        return {
            "schema": R.CHECKPOINT_SCHEMA,
            "authority": "whole_agent_task_only",
            "campaign_id": request["campaign_id"],
            "task_id": request["task"]["task_id"],
            "arm_id": request["arm"]["arm_id"],
            "baseline": request["baseline"],
            "checkpoint_hours": request["checkpoint_hours"],
            "evaluation": {
                "pass_compilation": True, "pass_correctness": True,
                "valid_baseline_cases": 3, "valid_optimized_cases": 3,
                "average_speedup": 1.1,
            },
            "artifacts": {"fixture.txt": hashlib.sha256(
                artifact.read_bytes()).hexdigest()},
        }

    def runner(self, *, claim_acquirer=None, sampler_factory=FakeSampler):
        kwargs = {"worker": self.worker, "sampler_factory": sampler_factory}
        if claim_acquirer is not None:
            kwargs["claim_acquirer"] = claim_acquirer
        return R.GovernedArenaCellRunner(self.config(), **kwargs)

    def test_baseline_runs_once_without_a_budget_or_belief_measurement(self):
        claim = FakeClaim()
        runner = self.runner(claim_acquirer=lambda *args, **kwargs: claim)
        request = C.CampaignCellRequest(
            arm=self.arm(C.BASELINE_ARM_ID), task=self.task,
            is_starting_state_baseline=True, checkpoint_hours=(),
            maximum_wall_hours=0.0)
        receipt = runner(request)
        self.assertEqual(receipt["checkpoint_hours"], [])
        self.assertEqual(len(receipt["runs"]), 1)
        self.assertIsNone(receipt["runs"][0]["belief_receipt"])
        self.assertTrue(claim.released)

    def test_controller_runs_three_fresh_matched_checkpoints_and_emits_beliefs(self):
        claims = []

        def acquire(*args, **kwargs):
            claim = FakeClaim()
            claims.append((claim, kwargs))
            return claim

        runner = self.runner(claim_acquirer=acquire)
        request = C.CampaignCellRequest(
            arm=self.arm("k_search"), task=self.task,
            is_starting_state_baseline=False,
            checkpoint_hours=C.MATCHED_BUDGET_HOURS,
            maximum_wall_hours=32.0)
        with mock.patch.object(
            C, "_implementation_audit",
            return_value={"executable": True, "missing_artifacts": []},
        ):
            receipt = runner(request)
        self.assertEqual(
            [row["checkpoint_hours"] for row in receipt["runs"]],
            [2.0, 8.0, 32.0])
        self.assertEqual(len(claims), 3)
        self.assertTrue(all(claim.released for claim, _ in claims))
        for run in receipt["runs"]:
            belief = run["belief_receipt"]
            self.assertEqual(belief["status"], "pass")
            self.assertEqual(len(belief["belief_measurements"]), 2)
            self.assertEqual(
                belief["belief_measurements"][0]["extra"]["controller_id"],
                "k_search")
        belief_files = sorted(self.output.glob("cells/*/belief-receipt.json"))
        self.assertEqual(len(belief_files), 3)
        for belief_file in belief_files:
            persisted = json.loads(belief_file.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["schema"], "epyc.autokernel.geak_arena_roundtrip.v1")
            self.assertEqual(
                persisted["receipt_sha256"],
                canonical_sha({key: value for key, value in persisted.items()
                               if key != "receipt_sha256"}))

    def test_identity_drift_refuses_before_claim_or_worker(self):
        acquire = mock.Mock()
        worker = mock.Mock()
        runner = R.GovernedArenaCellRunner(
            self.config(), worker=worker, claim_acquirer=acquire,
            sampler_factory=FakeSampler)
        request = C.CampaignCellRequest(
            arm=self.arm("k_search"), task=self.task,
            is_starting_state_baseline=False,
            checkpoint_hours=C.MATCHED_BUDGET_HOURS,
            maximum_wall_hours=32.0)
        with (
            mock.patch.object(
                C, "_implementation_audit",
                return_value={"executable": False,
                              "missing_artifacts": ["source checkout drifted"]}),
            self.assertRaisesRegex(R.ArenaCellRunnerError, "identity drifted"),
        ):
            runner(request)
        acquire.assert_not_called()
        worker.assert_not_called()

    def test_sampler_failure_still_releases_the_device_claim(self):
        claim = FakeClaim()

        class BrokenSampler(FakeSampler):
            def stop(self):
                raise RuntimeError("sampler failed")

        runner = self.runner(
            claim_acquirer=lambda *args, **kwargs: claim,
            sampler_factory=BrokenSampler)
        request = C.CampaignCellRequest(
            arm=self.arm(C.BASELINE_ARM_ID), task=self.task,
            is_starting_state_baseline=True, checkpoint_hours=(),
            maximum_wall_hours=0.0)
        with self.assertRaisesRegex(RuntimeError, "sampler failed"):
            runner(request)
        self.assertTrue(claim.released)

    def test_preflight_tamper_and_existing_output_fail_before_execution(self):
        payload = json.loads(self.preflight_path.read_text())
        payload["hardware"]["target_gfx_arch"] = "gfx942"
        self.preflight_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "does not verify"):
            self.runner()
        self.preflight_path.unlink()

        valid = {"schema": A.PREFLIGHT_SCHEMA, "hardware": {
            "target_gfx_arch": "gfx90a"}}
        valid["receipt_sha256"] = canonical_sha(valid)
        self.preflight_path.write_text(json.dumps(valid), encoding="utf-8")
        self.output.mkdir()
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "must not already exist"):
            self.config()

    def test_checkpoint_rewrites_only_declared_budget_flags(self):
        arm = self.arm("k_search")
        argv = R._controller_argv({"argv": list(arm.argv)}, 2.0)
        self.assertEqual(argv[argv.index("--checkpoint-hours") + 1], "2")
        self.assertEqual(argv[argv.index("--timeout-seconds") + 1], "7200")

    def test_worker_subprocess_uses_the_pinned_rocm_evaluator_python(self):
        cell = self.root / "cells" / "001"
        output = cell / "worker-result.json"
        command = R._worker_command(cell, output)
        self.assertEqual(command[0], str(R.EVALUATOR_PYTHON))
        identity = R._evaluator_python_identity()
        self.assertEqual(identity["sha256"], R.EVALUATOR_PYTHON_SHA256)
        self.assertEqual(identity["packages"], R.EVALUATOR_PACKAGE_VERSIONS)
        with mock.patch.object(R.sys, "executable", "/usr/bin/false"):
            with self.assertRaisesRegex(R.ArenaCellRunnerError, "must run under"):
                R._assert_worker_evaluator_identity({"evaluator_python": identity})

    def test_worker_uses_fresh_copy_and_centralized_vendor_evaluator(self):
        cell_root = self.root / "cells" / "001-fixture"
        cell_root.mkdir(parents=True)
        task_config = self.arena / "tasks" / "fixture" / "config.yaml"
        task_config.write_text(
            "task_type: hip2hip\nsource_file_path: [kernel.hip]\n"
            "target_kernel_functions: [kernel]\ncompile_command: [true]\n"
            "correctness_command: [true]\nperformance_command: [true]\n",
            encoding="utf-8")
        (task_config.parent / "kernel.hip").write_text("// original\n", encoding="utf-8")
        evaluator = types.ModuleType("src.evaluator")
        evaluator.evaluate_compilation = mock.Mock(return_value=(True, None))
        evaluator.measure_baseline = mock.Mock(return_value=[object(), object()])
        evaluator.evaluate_kernel = mock.Mock(return_value={
            "pass_compilation": True, "pass_correctness": True,
            "valid_baseline_cases": 2, "valid_optimized_cases": 2,
            "average_speedup": 1.05})

        def write_task_result(workspace, *args, **kwargs):
            (workspace / "task_result.yaml").write_text("pass: true\n", encoding="utf-8")

        evaluator.write_task_result = write_task_result
        prompt_builder = types.ModuleType("src.prompt_builder")
        prompt_builder.prompt_builder = mock.Mock(return_value="Optimize the kernel.")
        src = types.ModuleType("src")
        src.evaluator = evaluator
        src.prompt_builder = prompt_builder
        request = {
            "schema": R.CHECKPOINT_SCHEMA,
            "campaign_id": "fixture-campaign-v1",
            "arena_root": str(self.arena.resolve()),
            "repository_root": str(R.REPOSITORY_ROOT),
            "cell_root": str(cell_root.resolve()),
            "task": {"task_id": "fixture.task", "relative_root": "tasks/fixture",
                     "file_sha256": {}},
            "arm": {"arm_id": "k_search", "argv": [sys.executable, "controller.py",
                    "--checkpoint-hours", "32", "--timeout-seconds", "115200"],
                    "source_root": str(self.root), "source_commit": "a" * 40,
                    "entrypoint_path": "controller.py", "entrypoint_sha256": "b" * 64,
                    "model_ids": ["fixture"]},
            "baseline": False,
            "checkpoint_hours": 2.0,
            "visible_device": "0",
            "evaluator_python": R._evaluator_python_identity(),
        }
        with (
            mock.patch.dict(sys.modules, {
                "src": src, "src.evaluator": evaluator,
                "src.prompt_builder": prompt_builder,
            }),
            mock.patch.object(R.arena_adapter, "launch", return_value="controller receipt"),
            mock.patch.object(
                R, "_assert_worker_evaluator_identity",
                return_value=R._evaluator_python_identity()),
        ):
            receipt = R.run_worker(request)
        self.assertTrue(receipt["evaluation"]["pass_correctness"])
        self.assertEqual(receipt["checkpoint_hours"], 2.0)
        self.assertTrue((cell_root / "workspace" / "task_result.yaml").is_file())
        self.assertTrue((cell_root / "controller.stdout").is_file())
        evaluator.evaluate_kernel.assert_called_once()
        self.assertEqual(
            receipt["constraints"]["evaluator_python"]["sha256"],
            R.EVALUATOR_PYTHON_SHA256)

    def test_exact_group_teardown_kills_a_planted_descendant(self):
        child_pid_path = self.root / "descendant.pid"
        script = (
            "import pathlib, subprocess, sys; "
            "child=subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(30)'], stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL); "
            f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script], start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            process.wait(timeout=5)
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            self.assertIn(child_pid, R._live_process_group_members(process.pid))
            R._terminate_captured_process_group(process.pid, grace_seconds=1.0)
            self.assertNotIn(child_pid, R._live_process_group_members(process.pid))
        finally:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    unittest.main()
