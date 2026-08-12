#!/usr/bin/env python3
"""Tests for the governed AgentKernelArena campaign execution bridge."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
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
    def __init__(self, phase: str, claim_id: str = "akd-fixture0000001"):
        self.phase = phase
        self.claim_id = claim_id

    def to_dict(self):
        row = {
            "schema": "epyc.autokernel.device_claim_receipt.v1",
            "claim_id": self.claim_id,
            "device_id": "mi210_0",
            "lock_path": "/tmp/gpu_device.mi210_0.lock",
            "state": "held",
            "holder_pid": 1234,
            "holder_start_ticks": 5678,
            "holder_boot_id": "00000000-0000-0000-0000-000000000000",
            "host": "fixture-host",
            "holder_label": "fixture",
            "purpose": "fixture test",
            "campaign_id": "fixture-campaign-v1",
            "acquired_at": "2026-08-11T00:00:00Z",
            "expires_at": None,
            "released_at": None,
            "reclaimed_from": None,
        }
        if self.phase == "released":
            row["released_at"] = "2026-08-11T00:00:01Z"
        return row


class FakeClaim:
    def __init__(self, claim_id: str = "akd-fixture0000001"):
        self.released = False
        self.claim_id = claim_id

    def receipt(self):
        return FakeReceipt("opened", self.claim_id)

    def release(self):
        self.released = True
        return FakeReceipt("released", self.claim_id)


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

    def window_request(self, cell_root: Path) -> dict:
        return {
            "campaign_id": "fixture-campaign-v1",
            "task": {"task_id": "fixture.task"},
            "arm": {"arm_id": "k_search"},
            "visible_device": "0",
            "claim_journal": str((self.root / "claim.jsonl").resolve()),
            "claim_timeout_seconds": 0.0,
            "cell_root": str(cell_root),
        }

    @staticmethod
    def worker(request, timeout):
        cell_root = Path(request["cell_root"])
        artifact = cell_root / "fixture.txt"
        artifact.write_text("evidence\n", encoding="utf-8")
        windows = []
        for ordinal, phase in enumerate(
            ("vendor_baseline", "centralized_final_evaluation"), start=1,
        ):
            claim_id = f"akd-fixture000000{ordinal}"
            window = {
                "schema": R.MEASUREMENT_WINDOW_SCHEMA,
                "campaign_id": request["campaign_id"],
                "task_id": request["task"]["task_id"],
                "arm_id": request["arm"]["arm_id"],
                "phase": phase,
                "ordinal": ordinal,
                "status": "complete",
                "started_at": "2026-08-11T00:00:00Z",
                "ended_at": "2026-08-11T00:00:01Z",
                "device_claim_open": FakeReceipt("opened", claim_id).to_dict(),
                "device_claim_released": FakeReceipt("released", claim_id).to_dict(),
                "device_sampling": FakeSampling().to_dict(),
                "gpu_action_executed_only_while_claim_held": True,
                "failure": None,
            }
            window["receipt_sha256"] = canonical_sha(window)
            path = cell_root / "measurement-windows" / f"{ordinal:02d}-{phase}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(window), encoding="utf-8")
            windows.append(window)
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
            "measurement_windows": windows,
            "artifacts": {"fixture.txt": hashlib.sha256(
                artifact.read_bytes()).hexdigest()},
        }

    def runner(self):
        return R.GovernedArenaCellRunner(self.config(), worker=self.worker)

    def test_baseline_runs_once_without_a_budget_or_belief_measurement(self):
        runner = self.runner()
        request = C.CampaignCellRequest(
            arm=self.arm(C.BASELINE_ARM_ID), task=self.task,
            is_starting_state_baseline=True, checkpoint_hours=(),
            maximum_wall_hours=0.0)
        receipt = runner(request)
        self.assertEqual(receipt["checkpoint_hours"], [])
        self.assertEqual(len(receipt["runs"]), 1)
        self.assertIsNone(receipt["runs"][0]["belief_receipt"])
        self.assertEqual(len(receipt["runs"][0]["measurement_windows"]), 2)

    def test_controller_runs_three_fresh_matched_checkpoints_and_emits_beliefs(self):
        runner = self.runner()
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
        for run in receipt["runs"]:
            self.assertEqual(len(run["measurement_windows"]), 2)
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
        runner = R.GovernedArenaCellRunner(self.config(), worker=worker)
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
        worker.assert_not_called()

    def test_sampler_failure_still_releases_the_device_claim(self):
        claim = FakeClaim()

        class BrokenSampler(FakeSampler):
            def stop(self):
                raise RuntimeError("sampler failed")

        cell = self.root / "cells" / "window"
        cell.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "sampler failed"):
            R._run_gpu_measurement_window(
                request=self.window_request(cell), cell_root=cell, ordinal=1,
                phase="vendor_baseline", action=lambda: "measured",
                claim_acquirer=lambda *args, **kwargs: claim,
                sampler_factory=BrokenSampler)
        self.assertTrue(claim.released)
        persisted = json.loads(next((cell / "measurement-windows").glob("*.json")).read_text())
        self.assertEqual(persisted["status"], "failed")

    def test_sigterm_unwinds_worker_and_journals_claim_release_boundary(self):
        claim = FakeClaim()

        def interrupted_action():
            os.kill(os.getpid(), signal.SIGTERM)
            self.fail("SIGTERM handler did not interrupt the measurement")

        cell = self.root / "cells" / "window"
        cell.mkdir(parents=True)
        with (
            R._graceful_campaign_signals(),
            self.assertRaisesRegex(R.ArenaCampaignInterrupted, "SIGTERM"),
        ):
            R._run_gpu_measurement_window(
                request=self.window_request(cell), cell_root=cell, ordinal=1,
                phase="vendor_baseline", action=interrupted_action,
                claim_acquirer=lambda *args, **kwargs: claim,
                sampler_factory=FakeSampler)
        self.assertTrue(claim.released)
        persisted = json.loads(next((cell / "measurement-windows").glob("*.json")).read_text())
        self.assertIsNotNone(persisted["device_claim_released"]["released_at"])

    def test_sigterm_during_claim_acquisition_is_deferred_until_handle_assignment(self):
        claim = FakeClaim()

        def acquire(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            return claim

        cell = self.root / "cells" / "window"
        cell.mkdir(parents=True)
        with (
            R._graceful_campaign_signals(),
            self.assertRaisesRegex(R.ArenaCampaignInterrupted, "SIGTERM"),
        ):
            R._run_gpu_measurement_window(
                request=self.window_request(cell), cell_root=cell, ordinal=1,
                phase="vendor_baseline", action=lambda: "unreachable",
                claim_acquirer=acquire, sampler_factory=FakeSampler)
        self.assertTrue(claim.released)
        persisted = json.loads(next((cell / "measurement-windows").glob("*.json")).read_text())
        self.assertIsNotNone(persisted["device_claim_released"]["released_at"])

    def test_dotted_task_id_gets_collision_bound_dot_free_cell_path(self):
        native = "instruction2triton.rocmbench.test_add_kernel"
        normalized = R._path_id(native, "task_id")
        self.assertNotIn(".", normalized)
        self.assertRegex(normalized, r"^[a-z0-9_-]+-[0-9a-f]{12}$")
        self.assertNotEqual(
            normalized, R._path_id(native.replace(".", "_"), "task_id"))

        cell_root = self.root / "cells" / f"001-{normalized}-baseline"
        workspace = cell_root / "workspace"
        workspace.mkdir(parents=True)
        source = workspace / "test_add_kernel.py"
        source.write_text("# fixture\n", encoding="utf-8")
        # Exact transform pinned in AgentKernelArena's test_add_kernel.py.
        transformed = Path(str(source).replace(".", "_") + ".pt")
        transformed.write_text("cache\n", encoding="utf-8")
        self.assertEqual(transformed.parent, workspace)
        self.assertEqual(
            transformed.relative_to(cell_root).as_posix(),
            "workspace/test_add_kernel_py.pt")
        R._assert_worker_tree_contained(cell_root)

    def test_sibling_write_is_detected_even_when_worker_raises(self):
        def escaping_worker(request, timeout):
            cell_root = Path(request["cell_root"])
            escaped = cell_root.with_name(cell_root.name + "-escaped")
            escaped.mkdir()
            (escaped / "test_add_kernel_py.pt").write_text(
                "escaped\n", encoding="utf-8")
            raise RuntimeError("worker failed after escape")

        runner = R.GovernedArenaCellRunner(self.config(), worker=escaping_worker)
        request = C.CampaignCellRequest(
            arm=self.arm(C.BASELINE_ARM_ID), task=self.task,
            is_starting_state_baseline=True, checkpoint_hours=(),
            maximum_wall_hours=0.0)
        with self.assertRaisesRegex(
                R.ArenaCellRunnerError, "wrote outside its exact cell root"):
            runner(request)
        self.assertTrue(next(self.output.glob("cells/*-escaped")).is_dir())
        self.assertEqual(
            list(self.output.glob("cells/*/checkpoint-receipt.json")), [])

    def test_preflight_tamper_and_existing_non_directory_fail_before_execution(self):
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
        self.output.write_text("not a campaign directory\n", encoding="utf-8")
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "non-symlink directory"):
            self.config()

    def test_resume_skips_complete_checkpoint_and_replaces_only_partial_attempt(self):
        calls: list[float | None] = []

        def interrupted_worker(request, timeout):
            calls.append(request["checkpoint_hours"])
            if request["checkpoint_hours"] == 8.0:
                Path(request["cell_root"], "partial.txt").write_text(
                    "interrupted\n", encoding="utf-8")
                raise RuntimeError("planted interruption")
            return self.worker(request, timeout)

        request = C.CampaignCellRequest(
            arm=self.arm("k_search"), task=self.task,
            is_starting_state_baseline=False,
            checkpoint_hours=C.MATCHED_BUDGET_HOURS,
            maximum_wall_hours=32.0)
        first = R.GovernedArenaCellRunner(
            self.config(), worker=interrupted_worker)
        with (
            mock.patch.object(
                C, "_implementation_audit",
                return_value={"executable": True, "missing_artifacts": []}),
            self.assertRaisesRegex(RuntimeError, "planted interruption"),
        ):
            first(request)
        self.assertEqual(calls, [2.0, 8.0])

        resumed_calls: list[float | None] = []

        def resumed_worker(worker_request, timeout):
            resumed_calls.append(worker_request["checkpoint_hours"])
            return self.worker(worker_request, timeout)

        resumed = R.GovernedArenaCellRunner(
            self.config(), worker=resumed_worker)
        with mock.patch.object(
            C, "_implementation_audit",
            return_value={"executable": True, "missing_artifacts": []},
        ):
            receipt = resumed(request)
        self.assertEqual(resumed_calls, [8.0, 32.0])
        self.assertEqual(resumed.resumed_checkpoints, 1)
        self.assertEqual(resumed.executed_checkpoints, 2)
        abandoned = list((self.output / "abandoned").glob("*8h.attempt-001"))
        self.assertEqual(len(abandoned), 1)
        self.assertTrue((abandoned[0] / "partial.txt").is_file())
        self.assertEqual(
            [row["checkpoint_hours"] for row in receipt["runs"]],
            [2.0, 8.0, 32.0])

    def test_tampered_complete_checkpoint_refuses_without_claim_or_worker(self):
        request = C.CampaignCellRequest(
            arm=self.arm(C.BASELINE_ARM_ID), task=self.task,
            is_starting_state_baseline=True, checkpoint_hours=(),
            maximum_wall_hours=0.0)
        first = self.runner()
        first(request)
        checkpoint = next(self.output.glob("cells/*/checkpoint-receipt.json"))
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        payload["measurement_windows"][0]["device_claim_released"]["released_at"] = None
        payload["measurement_windows"][0]["receipt_sha256"] = canonical_sha({
            key: value for key, value in payload["measurement_windows"][0].items()
            if key != "receipt_sha256"})
        payload["receipt_sha256"] = canonical_sha({
            key: value for key, value in payload.items()
            if key != "receipt_sha256"})
        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
        worker = mock.Mock()
        resumed = R.GovernedArenaCellRunner(self.config(), worker=worker)
        with self.assertRaisesRegex(
                R.ArenaCellRunnerError, "not cleanly released"):
            resumed(request)
        worker.assert_not_called()

    def test_artifact_mutation_refuses_instead_of_rerunning_complete_checkpoint(self):
        request = C.CampaignCellRequest(
            arm=self.arm(C.BASELINE_ARM_ID), task=self.task,
            is_starting_state_baseline=True, checkpoint_hours=(),
            maximum_wall_hours=0.0)
        self.runner()(request)
        artifact = next(self.output.glob("cells/*/fixture.txt"))
        artifact.write_text("mutated\n", encoding="utf-8")
        worker = mock.Mock()
        resumed = R.GovernedArenaCellRunner(self.config(), worker=worker)
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "artifact digest drifted"):
            resumed(request)
        worker.assert_not_called()

    def test_source_mutation_during_worker_prevents_completed_receipt(self):
        def mutating_worker(request, timeout):
            result = self.worker(request, timeout)
            (self.arena / "tasks" / "fixture" / "config.yaml").write_text(
                "task_type: mutated\n", encoding="utf-8")
            return result

        runner = R.GovernedArenaCellRunner(
            self.config(), worker=mutating_worker)
        request = C.CampaignCellRequest(
            arm=self.arm(C.BASELINE_ARM_ID), task=self.task,
            is_starting_state_baseline=True, checkpoint_hours=(),
            maximum_wall_hours=0.0)
        with self.assertRaisesRegex(
                R.ArenaCellRunnerError, "changed during checkpoint"):
            runner(request)
        self.assertEqual(
            list(self.output.glob("cells/*/checkpoint-receipt.json")), [])

    def test_campaign_root_resume_requires_exact_audit_and_manifest(self):
        audit = R._self_hash({"schema": "audit.fixture", "campaign_id": "fixture"})
        manifest = R._self_hash({
            "schema": R.RUN_MANIFEST_SCHEMA, "campaign_id": "fixture"})
        campaign_root = self.root / "campaign"
        self.assertFalse(R._prepare_campaign_root(
            campaign_root, audit=audit, manifest=manifest))
        self.assertTrue(R._prepare_campaign_root(
            campaign_root, audit=audit, manifest=manifest))
        changed = R._self_hash({
            "schema": R.RUN_MANIFEST_SCHEMA, "campaign_id": "other"})
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "identity drifted"):
            R._prepare_campaign_root(
                campaign_root, audit=audit, manifest=changed)

    def test_aggregate_is_absent_until_atomic_complete_publication(self):
        path = self.root / "campaign" / "execution-receipt.json"
        path.parent.mkdir()
        complete = R._self_hash({
            "schema": R.AGGREGATE_SCHEMA,
            "campaign_id": "fixture-campaign-v1",
            "status": "complete",
            "cells": [],
        })
        self.assertFalse(path.exists())
        R._publish_or_verify_aggregate(path, complete)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), complete)
        R._publish_or_verify_aggregate(path, complete)
        tampered = dict(complete)
        tampered["status"] = "partial"
        tampered["receipt_sha256"] = canonical_sha({
            key: value for key, value in tampered.items()
            if key != "receipt_sha256"})
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "aggregate drifted"):
            R._publish_or_verify_aggregate(path, complete)

    def test_cli_fault_keeps_partial_matrix_non_aggregate_and_exact_retry_publishes(self):
        config_path = self.root / "campaign.json"
        config_path.write_text("{}\n", encoding="utf-8")
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        spec = types.SimpleNamespace(
            config_path=str(config_path.resolve()), config_sha256=config_sha,
            tasks=(self.task,), budget_hours=C.MATCHED_BUDGET_HOURS)
        audit = R._self_hash({
            "schema": C.AVAILABLE_SOURCE_AUDIT_SCHEMA,
            "campaign_id": "fixture-campaign-v1",
            "status": "ready",
            "authority": "availability_conditioned_diagnostic_only",
            "execution_identity": {
                "implementation_module_sha256": hashlib.sha256(
                    C.IMPLEMENTATION_MODULE.read_bytes()).hexdigest()},
            "sources": {"agent_kernel_arena": {}, "geak_v1": {}},
            "panel": {"arms": []},
        })
        output = self.root / "cli-campaign"
        args = types.SimpleNamespace(
            config=str(config_path), arena_root=str(self.arena.resolve()),
            geak_root=str(self.arena.resolve()), preflight=str(self.preflight_path),
            output_root=str(output), enumerator="fixture-enumerator",
            claim_journal=str(self.root / "claims.jsonl"),
            claim_timeout_seconds=0.0, available_source=True)
        fake_runner = mock.Mock()
        with (
            mock.patch.object(R.arena_campaign, "load_spec", return_value=spec),
            mock.patch.object(
                R.arena_campaign, "audit_available_source_campaign",
                return_value=audit),
            mock.patch.object(R, "GovernedArenaCellRunner", return_value=fake_runner),
            mock.patch.object(
                R.arena_campaign, "execute_available_source_campaign",
                side_effect=RuntimeError("planted campaign interruption")),
            self.assertRaisesRegex(RuntimeError, "planted campaign interruption"),
        ):
            R.execute_from_cli(args)
        self.assertTrue((output / "audit.json").is_file())
        self.assertTrue((output / "campaign-manifest.json").is_file())
        self.assertFalse((output / "execution-receipt.json").exists())

        cells = [R._self_hash({"schema": R.RUNNER_SCHEMA, "cell": "complete"})]
        with (
            mock.patch.object(R.arena_campaign, "load_spec", return_value=spec),
            mock.patch.object(
                R.arena_campaign, "audit_available_source_campaign",
                return_value=audit),
            mock.patch.object(R, "GovernedArenaCellRunner", return_value=fake_runner),
            mock.patch.object(
                R.arena_campaign, "execute_available_source_campaign",
                return_value=cells),
        ):
            status, aggregate = R.execute_from_cli(args)
        self.assertEqual(status, 0)
        self.assertEqual(aggregate["status"], "complete")
        self.assertEqual(aggregate["cells"], cells)
        self.assertTrue((output / "execution-receipt.json").is_file())

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
            "claim_journal": str((self.root / "claims.jsonl").resolve()),
            "claim_timeout_seconds": 0.0,
            "evaluator_python": R._evaluator_python_identity(),
        }
        claims: list[FakeClaim] = []
        broker_held = False

        class BrokerClaim(FakeClaim):
            def release(inner_self):
                nonlocal broker_held
                receipt = super().release()
                broker_held = False
                return receipt

        def acquire(*args, **kwargs):
            nonlocal broker_held
            self.assertFalse(broker_held, "device broker was still held")
            broker_held = True
            claim = BrokerClaim(f"akd-fixture000000{len(claims) + 1}")
            claims.append(claim)
            return claim

        def controller_launch(prepared, *args, **kwargs):
            self.assertTrue(claims[0].released)
            self.assertEqual(prepared.environment["HIP_VISIBLE_DEVICES"], "")
            self.assertEqual(prepared.environment["ROCR_VISIBLE_DEVICES"], "")
            # A second tenant can acquire the device during this gap because
            # AutoKernel holds no claim on behalf of the remote model.
            gap_claim = acquire(purpose="other governed tenant")
            gap_claim.release()
            return "controller receipt"

        with (
            mock.patch.dict(sys.modules, {
                "src": src, "src.evaluator": evaluator,
                "src.prompt_builder": prompt_builder,
            }),
            mock.patch.object(R.arena_adapter, "launch", side_effect=controller_launch),
            mock.patch.object(
                R, "_assert_worker_evaluator_identity",
                return_value=R._evaluator_python_identity()),
        ):
            receipt = R.run_worker(
                request, claim_acquirer=acquire, sampler_factory=FakeSampler)
        self.assertTrue(receipt["evaluation"]["pass_correctness"])
        self.assertEqual(receipt["checkpoint_hours"], 2.0)
        self.assertTrue((cell_root / "workspace" / "task_result.yaml").is_file())
        self.assertTrue((cell_root / "controller.stdout").is_file())
        evaluator.evaluate_kernel.assert_called_once()
        self.assertEqual(len(claims), 3)
        self.assertTrue(all(claim.released for claim in claims))
        self.assertEqual(
            [window["phase"] for window in receipt["measurement_windows"]],
            ["vendor_baseline", "centralized_final_evaluation"])
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
