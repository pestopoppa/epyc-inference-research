#!/usr/bin/env python3
"""Tests for INF-03's matched, all-or-nothing controller campaign."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import arena_campaign as C
from . import claude_codex_actor_critic as AC


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "arena_campaign_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArenaCampaignTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.arena = self.root / "arena"
        self.geak = self.root / "geak"
        self.arena.mkdir()
        self.geak.mkdir()
        task = self.arena / "tasks" / "one"
        task.mkdir(parents=True)
        (task / "config.yaml").write_text("task_type: fixture\n", encoding="utf-8")
        self.config_source = self.root / "campaign.json"
        self.config_source.write_text('{"fixture":true}\n', encoding="utf-8")
        self.source = self.root / "controller-source"
        self.source.mkdir()
        self.entrypoint = self.source / AC.ENTRYPOINT_RELATIVE
        self.entrypoint.parent.mkdir(parents=True)
        self.entrypoint.write_text("raise SystemExit(0)\n", encoding="utf-8")
        for command in (
            ("git", "init", "-q"),
            ("git", "add", "."),
            ("git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
             "commit", "-qm", "fixture"),
        ):
            subprocess.run(command, cwd=self.source, check=True)
        self.source_commit = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=self.source, check=True,
            capture_output=True, text=True).stdout.strip()

    def ready_spec(self) -> C.CampaignSpec:
        task_path = self.arena / "tasks" / "one" / "config.yaml"
        arms = []
        for arm_id in C.PRIMARY_PANEL_IDS:
            is_actor_critic = arm_id == AC.CONTROLLER_ID
            argv = (() if arm_id == C.BASELINE_ARM_ID else
                    (AC.campaign_argv(sys.executable) if is_actor_critic else
                     (sys.executable, "-c", "pass")))
            arms.append(C.ArmImplementation(
                arm_id=arm_id,
                availability="ready",
                adapter_kind=("arena_measure_baseline" if arm_id == C.BASELINE_ARM_ID
                              else ("agentkernelarena_three_arg_v1"
                                    if is_actor_critic else "stdin_workspace_v1")),
                missing_artifacts=(),
                argv=argv,
                source_root=(None if arm_id == C.BASELINE_ARM_ID else str(self.source)),
                source_commit=(None if arm_id == C.BASELINE_ARM_ID
                               else self.source_commit),
                entrypoint_path=(None if arm_id == C.BASELINE_ARM_ID
                                 else AC.ENTRYPOINT_RELATIVE),
                entrypoint_sha256=(None if arm_id == C.BASELINE_ARM_ID
                                   else sha(self.entrypoint)),
                model_ids=(() if arm_id == C.BASELINE_ARM_ID else
                           (AC.PINNED_MODEL_IDS if is_actor_critic else
                            ("fixture-model",))),
                required_clis=(AC.REQUIRED_CLIS if is_actor_critic else ()),
            ))
        return C.CampaignSpec(
            config_path=str(self.config_source.resolve()),
            config_sha256=sha(self.config_source),
            campaign_id="fixture-campaign-v1",
            target_gpu_model="MI210",
            target_gfx_arch="gfx90a",
            budget_hours=(2.0, 8.0, 32.0),
            tasks=(C.TaskArtifact(
                "fixture.task", "tasks/one", {"config.yaml": sha(task_path)}),),
            arms=tuple(arms),
            out_of_panel_registered=(),
        )

    @staticmethod
    def source_receipt(name: str) -> dict:
        return {"name": name, "commit": "a" * 40, "clean": True}

    def audit(self, spec: C.CampaignSpec) -> dict:
        with (
            mock.patch.object(
                C.arena_adapter, "inspect_vendor_source",
                side_effect=(self.source_receipt("arena"), self.source_receipt("geak")),
            ),
            mock.patch.object(
                C.arena_adapter, "detect_gfx_arch",
                return_value={"target_gpu_model": "MI210", "target_gfx_arch": "gfx90a"},
            ),
        ):
            return C.audit_campaign(spec, arena_root=self.arena, geak_root=self.geak)

    def test_repository_config_predeclares_eight_arms_and_exact_rebench_budgets(self):
        spec = C.load_spec(CONFIG)
        self.assertEqual(tuple(arm.arm_id for arm in spec.arms), C.PRIMARY_PANEL_IDS)
        self.assertEqual(len(spec.arms), 8)
        self.assertEqual(spec.budget_hours, (2.0, 8.0, 32.0))
        self.assertEqual(spec.arms[0].adapter_kind, "arena_measure_baseline")
        self.assertEqual(spec.out_of_panel_registered, ())
        self.assertEqual(spec.arms[-1].arm_id, "argus")

    def test_repository_config_names_missing_implementations_instead_of_argv_aliases(self):
        spec = C.load_spec(CONFIG)
        self.assertEqual(spec.arms[0].availability, "ready")
        actor = spec.arms[1]
        self.assertEqual(actor.arm_id, AC.CONTROLLER_ID)
        self.assertEqual(actor.availability, "ready")
        self.assertEqual(actor.adapter_kind, "agentkernelarena_three_arg_v1")
        self.assertEqual(actor.argv, AC.campaign_argv("python3"))
        self.assertEqual(actor.source_root, C.IN_TREE_SOURCE_ROOT)
        self.assertEqual(actor.entrypoint_path, AC.ENTRYPOINT_RELATIVE)
        self.assertEqual(actor.model_ids, AC.PINNED_MODEL_IDS)
        self.assertEqual(actor.required_clis, AC.REQUIRED_CLIS)
        self.assertEqual(actor.missing_artifacts, ())
        pinned = subprocess.run(
            ("git", "-C", str(C.REPOSITORY_ROOT), "show",
             f"{actor.source_commit}:{actor.entrypoint_path}"),
            capture_output=True, check=True)
        self.assertEqual(hashlib.sha256(pinned.stdout).hexdigest(),
                         actor.entrypoint_sha256)
        subprocess.run(
            ("git", "-C", str(C.REPOSITORY_ROOT), "merge-base", "--is-ancestor",
             actor.source_commit, "HEAD"), check=True)
        for arm in spec.arms[2:]:
            self.assertEqual(arm.availability, "missing")
            self.assertFalse(arm.argv)
            self.assertGreaterEqual(len(arm.missing_artifacts), 3)
        geak = next(arm for arm in spec.arms if arm.arm_id == "geak_v1")
        self.assertIn("TB-eval-smol", " ".join(geak.missing_artifacts))
        self.assertIn("dvue-aoai-001-gpt-4.1", " ".join(geak.missing_artifacts))
        argus = spec.arms[-1]
        self.assertIn("licensed", " ".join(argus.missing_artifacts))
        self.assertIn("gfx90a", " ".join(argus.missing_artifacts))

    def test_controller_coverage_is_two_only_when_both_clis_are_present(self):
        actor = C.ArmImplementation(
            arm_id=AC.CONTROLLER_ID,
            availability="ready",
            adapter_kind="agentkernelarena_three_arg_v1",
            missing_artifacts=(),
            argv=AC.campaign_argv(sys.executable),
            source_root=str(self.source),
            source_commit=self.source_commit,
            entrypoint_path=AC.ENTRYPOINT_RELATIVE,
            entrypoint_sha256=sha(self.entrypoint),
            model_ids=AC.PINNED_MODEL_IDS,
            required_clis=AC.REQUIRED_CLIS,
        )
        original_which = C.shutil.which

        def all_present(name):
            if name in {"claude", "codex"}:
                return sys.executable
            return original_which(name)

        with mock.patch.object(C.shutil, "which", side_effect=all_present):
            row = C._implementation_audit(actor)
        self.assertTrue(row["executable"])
        self.assertEqual(
            [item["name"] for item in row["required_cli_identities"]],
            ["claude", "codex"],
        )

        def codex_missing(name):
            if name == "claude":
                return sys.executable
            if name == "codex":
                return None
            return original_which(name)

        with mock.patch.object(C.shutil, "which", side_effect=codex_missing):
            row = C._implementation_audit(actor)
        self.assertFalse(row["executable"])
        self.assertTrue(any("not found: codex" in item
                            for item in row["missing_artifacts"]))

    def test_panel_cardinality_order_budget_and_registered_coverage_fail_closed(self):
        spec = self.ready_spec()
        with self.assertRaisesRegex(C.ArenaCampaignError, "primary panel"):
            C.CampaignSpec(**{**spec.__dict__, "arms": tuple(reversed(spec.arms))})
        with self.assertRaisesRegex(C.ArenaCampaignError, "checkpoints"):
            C.CampaignSpec(**{**spec.__dict__, "budget_hours": (2.0,)})
        with self.assertRaisesRegex(C.ArenaCampaignError, "no registered controller"):
            C.CampaignSpec(**{
                **spec.__dict__, "out_of_panel_registered": ("argus",)})

    def test_missing_arm_must_name_exact_artifact_and_ready_arm_needs_argv(self):
        with self.assertRaisesRegex(C.ArenaCampaignError, "must name exact"):
            C.ArmImplementation(
                "evoengineer", "missing", "stdin_workspace_v1", ())
        with self.assertRaisesRegex(C.ArenaCampaignError, "requires an argv"):
            C.ArmImplementation(
                "evoengineer", "ready", "stdin_workspace_v1", ())
        with self.assertRaisesRegex(C.ArenaCampaignError, "source_root"):
            C.ArmImplementation(
                "evoengineer", "ready", "stdin_workspace_v1", (),
                argv=(sys.executable, "-c", "pass"))

    def test_complete_fixture_audits_ready_and_binds_executable_hashes(self):
        receipt = self.audit(self.ready_spec())
        self.assertEqual(receipt["status"], "ready")
        self.assertEqual(receipt["panel"]["executable_arm_count"], 8)
        self.assertEqual(receipt["panel"]["baseline_arm_id"], "starting_state_baseline")
        self.assertEqual(receipt["execution_identity"]["config_sha256"],
                         sha(self.config_source))
        self.assertEqual(
            receipt["execution_identity"]["implementation_module_sha256"],
            sha(C.IMPLEMENTATION_MODULE),
        )
        controller_rows = receipt["panel"]["arms"][1:]
        self.assertTrue(all(row["executable_sha256"] for row in controller_rows))
        self.assertTrue(all(row["source_identity"]["clean"] for row in controller_rows))
        self.assertEqual(controller_rows[0]["model_ids"], list(AC.PINNED_MODEL_IDS))
        self.assertEqual(
            [row["model_ids"] for row in controller_rows[1:]],
            [["fixture-model"]] * 6,
        )
        self.assertEqual(
            [row["name"] for row in controller_rows[0]["required_cli_identities"]],
            list(AC.REQUIRED_CLIS),
        )
        self.assertFalse(receipt["constraints"]["controller_or_gpu_command_executed"])
        self.assertEqual(
            [row["name"] for row in receipt["host_cli_inventory"]],
            ["claude", "codex", "cursor", "geak"],
        )
        self.assertTrue(all(not row["implementation_coverage_implied"]
                            for row in receipt["host_cli_inventory"]))

    def test_incomplete_panel_refuses_before_any_cell_executes(self):
        spec = self.ready_spec()
        arms = list(spec.arms)
        arms[2] = C.ArmImplementation(
            "evoengineer", "missing", "stdin_workspace_v1",
            ("pinned EvoEngineer checkout",))
        incomplete = C.CampaignSpec(**{**spec.__dict__, "arms": tuple(arms)})
        receipt = self.audit(incomplete)
        self.assertEqual(receipt["status"], "refused")
        self.assertEqual(receipt["panel"]["executable_arm_count"], 7)
        self.assertTrue(any("7/8" in reason for reason in receipt["refusal_reasons"]))
        runner = mock.Mock()
        with self.assertRaisesRegex(C.ArenaCampaignError, "no cell may execute"):
            C.execute_campaign(incomplete, receipt, run_cell=runner)
        runner.assert_not_called()

    def test_task_hash_drift_refuses_the_whole_panel(self):
        spec = self.ready_spec()
        (self.arena / "tasks" / "one" / "config.yaml").write_text(
            "task_type: moved\n", encoding="utf-8")
        receipt = self.audit(spec)
        self.assertEqual(receipt["status"], "refused")
        self.assertFalse(receipt["tasks"][0]["ready"])
        self.assertTrue(any("expected" in reason for reason in receipt["refusal_reasons"]))

    def test_registry_drift_refuses_instead_of_changing_campaign_cardinality(self):
        spec = self.ready_spec()
        with (
            mock.patch.object(
                C.arena_adapter, "inspect_vendor_source",
                side_effect=(self.source_receipt("arena"), self.source_receipt("geak")),
            ),
            mock.patch.object(C.arena_adapter, "detect_gfx_arch", return_value={}),
            mock.patch.object(C.arena_adapter, "CONTROLLERS", {
                key: value for key, value in C.arena_adapter.CONTROLLERS.items()
                if key != "argus"
            }),
        ):
            receipt = C.audit_campaign(spec, arena_root=self.arena, geak_root=self.geak)
        self.assertEqual(receipt["status"], "refused")
        self.assertTrue(any("registry drift" in reason for reason in receipt["refusal_reasons"]))

    def test_skipped_hardware_inspection_is_never_ready(self):
        spec = self.ready_spec()
        with mock.patch.object(
            C.arena_adapter, "inspect_vendor_source",
            side_effect=(self.source_receipt("arena"), self.source_receipt("geak")),
        ):
            receipt = C.audit_campaign(
                spec, arena_root=self.arena, geak_root=self.geak,
                inspect_hardware=False)
        self.assertEqual(receipt["status"], "refused")
        self.assertTrue(any("inspection was skipped" in reason
                            for reason in receipt["refusal_reasons"]))

    def test_ready_execution_order_is_task_then_predeclared_arm_with_matched_ceiling(self):
        spec = self.ready_spec()
        receipt = self.audit(spec)
        calls = []

        def runner(request):
            calls.append(request)
            return request.arm.arm_id

        result = C.execute_campaign(spec, receipt, run_cell=runner)
        self.assertEqual(result, list(C.PRIMARY_PANEL_IDS))
        self.assertEqual([row.arm.arm_id for row in calls], list(C.PRIMARY_PANEL_IDS))
        self.assertTrue(calls[0].is_starting_state_baseline)
        self.assertEqual(calls[0].checkpoint_hours, ())
        self.assertEqual(calls[0].maximum_wall_hours, 0.0)
        for request in calls[1:]:
            self.assertFalse(request.is_starting_state_baseline)
            self.assertEqual(request.checkpoint_hours, (2.0, 8.0, 32.0))
            self.assertEqual(request.maximum_wall_hours, 32.0)

    def test_ready_audit_cannot_replay_after_config_or_driver_identity_changes(self):
        spec = self.ready_spec()
        receipt = self.audit(spec)
        self.config_source.write_text('{"fixture":false}\n', encoding="utf-8")
        runner = mock.Mock()
        with self.assertRaisesRegex(C.ArenaCampaignError, "config changed"):
            C.execute_campaign(spec, receipt, run_cell=runner)
        runner.assert_not_called()

        self.config_source.write_text('{"fixture":true}\n', encoding="utf-8")
        receipt = self.audit(spec)
        changed_driver = json.loads(json.dumps(receipt))
        changed_driver["execution_identity"]["implementation_module_sha256"] = "f" * 64
        with self.assertRaisesRegex(C.ArenaCampaignError, "module changed"):
            C.execute_campaign(spec, changed_driver, run_cell=runner)
        runner.assert_not_called()

    def test_cell_request_refuses_dropped_checkpoint_and_baseline_budget(self):
        spec = self.ready_spec()
        with self.assertRaisesRegex(C.ArenaCampaignError, "every checkpoint"):
            C.CampaignCellRequest(
                arm=spec.arms[1], task=spec.tasks[0],
                is_starting_state_baseline=False,
                checkpoint_hours=(32.0,), maximum_wall_hours=32.0)
        with self.assertRaisesRegex(C.ArenaCampaignError, "no authoring checkpoints"):
            C.CampaignCellRequest(
                arm=spec.arms[0], task=spec.tasks[0],
                is_starting_state_baseline=True,
                checkpoint_hours=(2.0, 8.0, 32.0), maximum_wall_hours=32.0)

    def test_receipt_writer_is_atomic_and_hash_stable(self):
        receipt = self.audit(self.ready_spec())
        second = self.audit(self.ready_spec())
        self.assertEqual(receipt["receipt_sha256"], second["receipt_sha256"])
        output = self.root / "receipts" / "audit.json"
        C.write_receipt(output, receipt)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), receipt)
        self.assertEqual(list(output.parent.glob(".audit.json.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
