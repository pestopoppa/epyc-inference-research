#!/usr/bin/env python3
"""Tests for the governed AgentKernelArena campaign execution bridge."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from . import arena_adapter as A
from . import arena_campaign as C
from . import arena_cell_runner as R
from . import arena_upstream_common as U
from . import k_search_arena as KS


def canonical_sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def fake_evaluator_execution(
    workspace: Path, evidence_root: Path, identity: dict,
    phase: str, baseline_receipt_sha256: str,
) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "config.yaml"
    if not config_path.exists():
        config_path.write_text("task_type: fixture\n", encoding="utf-8")
    baseline = {"schema": R.arena_evaluator_child.BASELINE_SCHEMA, "cases": []}
    baseline["receipt_sha256"] = canonical_sha(baseline)
    request = {
        "schema": R.arena_evaluator_child.REQUEST_SCHEMA,
        **identity, "workspace": str(workspace.resolve()), "phase": phase,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "vendor_evaluator_sha256": hashlib.sha256(
            (Path(identity["arena_root"]) / "src" / "evaluator.py").read_bytes()
        ).hexdigest(),
        "evaluator_python": R._declared_evaluator_python_identity(),
        "baseline_cases": baseline,
        "outer_baseline_receipt_sha256": baseline_receipt_sha256,
    }
    request["receipt_sha256"] = canonical_sha(request)
    (workspace / "evaluator-request.json").write_text(
        json.dumps(request), encoding="utf-8")
    activation = {
        "profile": R.sandbox.EVALUATOR_PROFILE, "pid": 12345,
        "process_start_ticks": 67890,
        "writable_root": str(workspace.resolve()),
        "writable_device_paths": ["/dev/kfd", "/dev/dri/renderD128", "/dev/null"],
        "read_allowlist_enforced": True, "readable_roots": ["/usr/lib"],
        "network_profile": R.sandbox.NETWORK_DENY_ALL,
        "outbound_socket_families": [], "unix_socket_creation_denied": True,
        "broker_socket_path": None, "broker_fd_inherited": False,
        "broker_peer": None, "cgroup_path": "/sys/fs/cgroup/autokernel-fixture",
        "blocked_syscalls": [
            "connect", "socket", "process_vm_readv", "process_vm_writev",
            "io_uring_setup", "io_uring_enter", "io_uring_register",
            "pidfd_getfd", "process_madvise"],
    }
    evaluation = {
        "pass_compilation": True, "pass_correctness": True,
        "valid_baseline_cases": 3, "valid_optimized_cases": 3,
        "average_speedup": 1.1,
    }
    result = {
        "schema": R.arena_evaluator_child.RESULT_SCHEMA,
        "request_receipt_sha256": request["receipt_sha256"],
        "baseline_cases_sha256": baseline["receipt_sha256"],
        "outer_baseline_receipt_sha256": baseline_receipt_sha256,
        "evaluation": evaluation,
    }
    result["receipt_sha256"] = canonical_sha(result)
    stdout = json.dumps(result, sort_keys=True) + "\n"
    (evidence_root / "evaluator.stdout").write_text(stdout, encoding="utf-8")
    (evidence_root / "evaluator.stderr").write_text("", encoding="utf-8")
    (evidence_root / "evaluator-result.json").write_text(
        json.dumps(result), encoding="utf-8")
    execution = {
        "schema": "epyc.autokernel.arena_evaluator_execution.v1",
        "request_receipt_sha256": request["receipt_sha256"],
        "result_receipt_sha256": result["receipt_sha256"],
        "pid": 12345, "process_start_ticks": 67890,
        "process_group_id": 12345, "session_id": 12345,
        "activation_receipt": activation,
        "teardown_receipt": {
            "cgroup_path": activation["cgroup_path"],
            "verified_empty": True, "removed": True,
            "descendants_killed": False},
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }
    execution["receipt_sha256"] = canonical_sha(execution)
    (evidence_root / "execution-receipt.json").write_text(
        json.dumps(execution), encoding="utf-8")
    return execution


def fake_controller_sandbox_execution(cell_root: Path) -> dict:
    workspace = (cell_root / "workspace").resolve()
    readable_roots = [str(Path(sys.executable).resolve().parent)]
    runtime = {
        "readable_roots": readable_roots, "readable_files": [],
        "executable_files": [str(Path(sys.executable).resolve())],
        "identities": {str(Path(sys.executable).resolve()):
                       R._sha256_file(Path(sys.executable).resolve())},
    }
    runtime["sha256"] = canonical_sha(runtime)
    policy_sha = "b" * 64
    cgroup = "/sys/fs/cgroup/autokernel-fixture-controller"
    blocked = sorted(R.sandbox._network_policy(
        R.sandbox.CONTROLLER_PROFILE)[0])
    activation = {
        "schema": R.sandbox.RECEIPT_SCHEMA,
        "sandbox_id": R.sandbox.SANDBOX_ID,
        "pid": 12346, "process_start_ticks": 67891, "euid": 1000,
        "landlock_abi": 3, "landlock_write_rights": 0,
        "landlock_handled_rights": 0, "seccomp_sha256": "c" * 64,
        "blocked_syscalls": blocked, "writable_root": str(workspace),
        "writable_device_paths": [], "cgroup_path": cgroup,
        "resource_limits": {}, "argv_sha256": "d" * 64,
        "read_allowlist_enforced": True,
        "readable_roots": readable_roots, "readable_files": [],
        "executable_files": [str(Path(sys.executable).resolve())],
        "profile": R.sandbox.CONTROLLER_PROFILE,
        "network_profile": R.sandbox.NETWORK_OUTBOUND_CLIENT,
        "outbound_socket_families": ["AF_INET", "AF_INET6"],
        "server_socket_operations_denied": ["bind", "listen", "accept", "accept4"],
        "unix_socket_creation_denied": True,
        "broker_socket_path": "/tmp/fake-broker.sock",
        "broker_fd_inherited": True,
        "broker_peer": {"pid": os.getpid(), "start_ticks": 1,
                        "uid": os.getuid(), "gid": os.getgid()},
        "policy_sha256": policy_sha,
    }
    activation_path = cell_root / R.CONTROLLER_ACTIVATION_RECEIPT
    activation_path.write_text(json.dumps(activation), encoding="utf-8")
    teardown = {
        "schema": R.arena_controller_sandbox.TEARDOWN_SCHEMA,
        "pid": 12346, "process_start_ticks": 67891,
        "policy_sha256": policy_sha,
        "runtime_allowlist_sha256": runtime["sha256"],
        "activation_receipt": str(activation_path),
        "activation_receipt_sha256": R._sha256_file(activation_path),
        "teardown": {"cgroup_path": cgroup, "verified_empty": True,
                     "removed": True, "descendants_killed": False},
    }
    teardown["receipt_sha256"] = canonical_sha(teardown)
    (cell_root / R.CONTROLLER_TEARDOWN_RECEIPT).write_text(
        json.dumps(teardown), encoding="utf-8")
    execution = {
        "schema": "epyc.autokernel.arena_controller_sandbox_execution.v1",
        "pid": 12346, "policy_sha256": policy_sha,
        "runtime_allowlist": runtime,
        "activation_receipt": activation, "teardown_receipt": teardown,
    }
    execution["receipt_sha256"] = canonical_sha(execution)
    (cell_root / "controller-sandbox-execution.json").write_text(
        json.dumps(execution), encoding="utf-8")
    return execution


class FakeReceipt:
    def __init__(self, phase: str, claim_id: str = "akd-fixture0000001",
                 campaign_id: str = "fixture-campaign-v1"):
        self.phase = phase
        self.claim_id = claim_id
        self.campaign_id = campaign_id

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
            "campaign_id": self.campaign_id,
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
        (self.arena / "src").mkdir()
        (self.arena / "src" / "evaluator.py").write_text(
            "# pinned fixture evaluator\n", encoding="utf-8")
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
                **({"attempt_id": request["attempt_id"]}
                   if request.get("attempt_id") is not None else {}),
                "claim_campaign_id": request.get(
                    "claim_campaign_id", request["campaign_id"]),
                "task_id": request["task"]["task_id"],
                "arm_id": request["arm"]["arm_id"],
                "checkpoint_hours": request["checkpoint_hours"],
                "phase": phase,
                "ordinal": ordinal,
                "status": "complete",
                "started_at": "2026-08-11T00:00:00Z",
                "ended_at": "2026-08-11T00:00:01Z",
                "device_claim_open": FakeReceipt(
                    "opened", claim_id, request.get(
                        "claim_campaign_id", request["campaign_id"])).to_dict(),
                "device_claim_released": FakeReceipt(
                    "released", claim_id, request.get(
                        "claim_campaign_id", request["campaign_id"])).to_dict(),
                "device_sampling": FakeSampling().to_dict(),
                "gpu_action_executed_only_while_claim_held": True,
                "failure": None,
            }
            if phase == "centralized_final_evaluation" \
                    and not request["baseline"]:
                identity = {
                    "campaign_id": request["campaign_id"],
                    **({"attempt_id": request["attempt_id"]}
                       if request.get("attempt_id") is not None else {}),
                    "claim_campaign_id": request.get(
                        "claim_campaign_id", request["campaign_id"]),
                    "task_id": request["task"]["task_id"],
                    "arm_id": request["arm"]["arm_id"],
                    "checkpoint_hours": request["checkpoint_hours"],
                    "arena_root": str(Path(request["arena_root"]).resolve()),
                }
                window["evaluator_execution_receipt"] = fake_evaluator_execution(
                    cell_root / "final-evaluation-workspace",
                    cell_root / "final-evaluator-evidence", identity, phase,
                    windows[0]["receipt_sha256"])
            window["receipt_sha256"] = canonical_sha(window)
            path = cell_root / "measurement-windows" / f"{ordinal:02d}-{phase}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(window), encoding="utf-8")
            windows.append(window)
        controller_execution = (
            None if request["baseline"]
            else fake_controller_sandbox_execution(cell_root))
        result = {
            "schema": R.CHECKPOINT_SCHEMA,
            "authority": "whole_agent_task_only",
            "campaign_id": request["campaign_id"],
            **({"attempt_id": request["attempt_id"]}
               if request.get("attempt_id") is not None else {}),
            "claim_campaign_id": request.get(
                "claim_campaign_id", request["campaign_id"]),
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
            "controller_sandbox_execution": controller_execution,
            "artifacts": {"fixture.txt": hashlib.sha256(
                artifact.read_bytes()).hexdigest()},
        }
        if controller_execution is not None:
            result["broker_evaluation_chain"] = {
                "controller_sandbox_execution_receipt_sha256":
                    controller_execution["receipt_sha256"]}
        return result

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

    def test_diagnostic_pilot_runs_one_governed_checkpoint_without_authority(self):
        runner = self.runner()
        argv = list(self.arm("k_search").argv)
        argv[argv.index("--max-rounds") + 1] = "1"
        request = R.DiagnosticPilotCellRequest(
            arm=self.arm("k_search"), task=self.task, checkpoint_hours=2.0,
            controller_argv=tuple(argv))
        with mock.patch.object(
            C, "_implementation_audit",
            return_value={"executable": True, "missing_artifacts": []},
        ):
            receipt = runner.run_diagnostic_pilot(request)
        self.assertEqual(receipt["schema"], R.DIAGNOSTIC_PILOT_SCHEMA)
        self.assertEqual(receipt["authority"],
                         "compatibility_only_no_ranking_or_promotion_authority")
        self.assertEqual(receipt["checkpoint_hours"], 2.0)
        self.assertEqual(receipt["checkpoint"]["checkpoint_hours"], 2.0)
        self.assertEqual(receipt["controller_argv"][
            receipt["controller_argv"].index("--max-rounds") + 1], "1")
        self.assertFalse(receipt["constraints"]["matched_campaign_result_implied"])
        self.assertFalse(receipt["constraints"]["belief_update_authority"])
        self.assertEqual(
            receipt["checkpoint_receipt_sha256"],
            receipt["checkpoint"]["receipt_sha256"])
        self.assertTrue((self.output / "diagnostic-pilot-receipt.json").is_file())

    def test_diagnostic_pilot_refuses_baseline_and_nonmatched_budget(self):
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "ready controller"):
            R.DiagnosticPilotCellRequest(
                arm=self.arm(C.BASELINE_ARM_ID), task=self.task)
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "matched budget"):
            R.DiagnosticPilotCellRequest(
                arm=self.arm("k_search"), task=self.task,
                checkpoint_hours=0.01)
        argv = list(self.arm("k_search").argv)
        argv[argv.index("--max-rounds") + 1] = "2"
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "may only set"):
            R.DiagnosticPilotCellRequest(
                arm=self.arm("k_search"), task=self.task,
                controller_argv=tuple(argv))

    def test_instruction_task_source_is_uniquely_discovered_from_kernel_target(self):
        workspace = self.root / "instruction-workspace"
        workspace.mkdir()
        (workspace / "helper.py").write_text("def helper(): pass\n", encoding="utf-8")
        (workspace / "kernel.py").write_text(
            "def add_kernel(x):\n    return x\n", encoding="utf-8")
        self.assertEqual(
            R._declared_task_sources({
                "source_file_path": [],
                "target_kernel_functions": ["add_kernel"],
            }, workspace),
            ("kernel.py",))

    def test_instruction_task_source_discovery_refuses_ambiguity(self):
        workspace = self.root / "ambiguous-instruction-workspace"
        workspace.mkdir()
        for name in ("one.py", "two.py"):
            (workspace / name).write_text(
                "def add_kernel(x):\n    return x\n", encoding="utf-8")
        with self.assertRaisesRegex(
                R.ArenaCellRunnerError, "uniquely discover"):
            R._declared_task_sources({
                "source_file_path": [],
                "target_kernel_functions": ["add_kernel"],
            }, workspace)

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

    def test_validate_only_is_read_only_and_rejects_semantic_window_swap(self):
        campaign = self.root / "attempt-r1"
        execution = campaign / "execution"
        campaign.mkdir()
        audit = R._self_hash({
            "schema": "audit.fixture", "campaign_id": "fixture-campaign-v1"})
        manifest = R._self_hash({
            "schema": R.RUN_MANIFEST_SCHEMA,
            "campaign_id": "fixture-campaign-v1", "attempt_id": campaign.name,
            "attempt_root": str(campaign.resolve()),
            "claim_campaign_id": campaign.name})
        (campaign / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
        (campaign / "campaign-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        config = R.RunnerConfig(
            campaign_id="fixture-campaign-v1", attempt_id=campaign.name,
            arena_root=str(self.arena.resolve()),
            preflight_path=str(self.preflight_path.resolve()),
            output_root=str(execution.resolve()),
            claim_journal=str((self.root / "claim.jsonl").resolve()))
        runner = R.GovernedArenaCellRunner(config, worker=self.worker)
        request = C.CampaignCellRequest(
            arm=self.arm(C.BASELINE_ARM_ID), task=self.task,
            is_starting_state_baseline=True, checkpoint_hours=(),
            maximum_wall_hours=0.0)
        runner(request)
        before = {path: path.read_bytes() for path in campaign.rglob("*")
                  if path.is_file()}
        result = R.validate_campaign_receipts(campaign)
        self.assertEqual(result["status"], "valid_partial")
        self.assertEqual(result["validated_checkpoint_count"], 1)
        self.assertEqual(before, {path: path.read_bytes()
                                 for path in campaign.rglob("*") if path.is_file()})

        checkpoint_path = next(execution.glob("cells/*/checkpoint-receipt.json"))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["measurement_windows"][0]["task_id"] = "other.task"
        checkpoint["measurement_windows"][0]["receipt_sha256"] = canonical_sha({
            key: value for key, value in checkpoint["measurement_windows"][0].items()
            if key != "receipt_sha256"})
        persisted = checkpoint_path.parent / "measurement-windows" / \
            "01-vendor_baseline.json"
        persisted.write_text(
            json.dumps(checkpoint["measurement_windows"][0]), encoding="utf-8")
        checkpoint["receipt_sha256"] = canonical_sha({
            key: value for key, value in checkpoint.items()
            if key != "receipt_sha256"})
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        with self.assertRaisesRegex(
                R.ArenaCellRunnerError, "semantic identity"):
            R.validate_campaign_receipts(campaign)

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
        self.assertEqual(
            argv[2], "kernel_rnd.autokernel.controller.k_search_arena")

    def test_controller_argv_replaces_only_with_exact_audited_executable(self):
        arm = self.arm("k_search")
        argv = R._controller_argv(
            {"argv": list(arm.argv)}, 2.0,
            executable_path=str(Path(sys.executable).resolve()))
        self.assertEqual(argv[0], str(Path(sys.executable).resolve()))
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "exact executable"):
            R._controller_argv(
                {"argv": list(arm.argv)}, 2.0,
                executable_path=str(self.root / "missing-python"))

    def test_evaluator_startup_is_deterministic_without_device_randomness(self):
        workspace = self.root / "evaluator-workspace"
        environment = R.SandboxedEvaluatorRunner._environment(
            workspace, self.arena)
        self.assertEqual(environment["PYTHONHASHSEED"], "0")
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertNotIn("LD_LIBRARY_PATH", environment)

    def test_evaluator_policy_admits_ephemeral_package_parent_not_proc(self):
        runner = R.SandboxedEvaluatorRunner(arena_root=self.arena)
        roots = (*runner._readable_roots(), str(runner.arena_root))
        self.assertIn(str(self.arena.resolve()), roots)
        self.assertNotIn(str((self.arena / "src").resolve()), roots)
        self.assertNotIn("/proc", roots)
        self.assertIn("/usr/libexec", roots)

    def test_parent_broker_is_short_private_fresh_and_hash_chained(self):
        cell = self.root / "broker-cell"
        workspace = cell / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "config.yaml").write_text(
            "source_file_path: [kernel.hip]\n", encoding="utf-8")
        (workspace / "kernel.hip").write_text("// original\n", encoding="utf-8")
        seen = []

        def evaluate(ordinal, evaluation_root, cancel_event):
            self.assertFalse(cancel_event.is_set())
            seen.append((ordinal, (evaluation_root / "kernel.hip").read_text()))
            claim_id = f"akd-broker0000000{ordinal}"
            identity = {
                "campaign_id": "fixture-campaign-v1", "attempt_id": "attempt-r1",
                "claim_campaign_id": "attempt-r1", "task_id": "fixture.task",
                "arm_id": "kernelfoundry", "checkpoint_hours": 2.0,
                "arena_root": str(self.arena.resolve())}
            execution = fake_evaluator_execution(
                evaluation_root,
                evaluation_root.with_name(f"{ordinal:04d}-evaluator-evidence"),
                identity, "controller_intermediate_evaluation", "b" * 64)
            window = R._self_hash({
                "schema": R.MEASUREMENT_WINDOW_SCHEMA,
                **identity,
                "phase": "controller_intermediate_evaluation", "ordinal": ordinal,
                "status": "complete", "started_at": "2026-08-12T00:00:00Z",
                "ended_at": "2026-08-12T00:00:01Z",
                "device_claim_open": FakeReceipt(
                    "opened", claim_id, "attempt-r1").to_dict(),
                "device_claim_released": FakeReceipt(
                    "released", claim_id, "attempt-r1").to_dict(),
                "device_sampling": {"sample_count": 2},
                "gpu_action_executed_only_while_claim_held": True,
                "failure": None,
                "evaluator_execution_receipt": execution})
            _path = cell / "controller-evaluation-windows" / \
                f"{ordinal:04d}-measurement.json"
            _path.parent.mkdir(parents=True, exist_ok=True)
            _path.write_text(json.dumps(window), encoding="utf-8")
            return ({"pass_compilation": True, "pass_correctness": True,
                     "valid_baseline_cases": 3, "valid_optimized_cases": 3,
                     "average_speedup": 1.1},
                    window)

        request = {
            "campaign_id": "fixture-campaign-v1", "attempt_id": "attempt-r1",
            "claim_campaign_id": "attempt-r1",
            "task": {"task_id": "fixture.task"},
            "arm": {"arm_id": "kernelfoundry"}, "checkpoint_hours": 2.0}
        broker = R._ControllerEvaluationBroker(
            request=request, workspace=workspace, cell_root=cell,
            source_paths=("kernel.hip",), evaluate=evaluate,
            baseline_receipt_sha256="b" * 64)
        evaluator = object.__new__(U.ArenaWorkspaceEvaluator)
        evaluator.workspace = workspace
        evaluator.broker_receipts = []
        with broker:
            self.assertLess(len(os.fsencode(broker.socket_path)), 108)
            self.assertEqual(broker.runtime_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(broker.socket_path.stat().st_mode & 0o777, 0o600)
            broker.register_controller(os.getpid())
            evaluator.broker_socket = broker.socket_path
            evaluator._broker_token = broker.token
            evaluator._broker_owner_pid = os.getpid()
            first = evaluator._brokered_evaluation(
                1, {"kernel.hip": b"// first\n"})
            second = evaluator._brokered_evaluation(
                2, {"kernel.hip": b"// second\n"})
        self.assertEqual(first["average_speedup"], 1.1)
        self.assertTrue(second["pass_correctness"])
        self.assertEqual((workspace / "kernel.hip").read_text(), "// original\n")
        self.assertEqual((broker.template / "kernel.hip").read_text(), "// original\n")
        receipts = [json.loads(path.read_text()) for path in sorted(
            (cell / "controller-evaluation-windows").glob("*-result.json"))]
        self.assertIsNone(receipts[0]["previous_receipt_sha256"])
        self.assertEqual(receipts[1]["previous_receipt_sha256"],
                         receipts[0]["receipt_sha256"])
        self.assertEqual(seen, [(1, "// first\n"), (2, "// second\n")])
        self.assertFalse(broker.runtime_dir.exists())
        checkpoint = {
            "campaign_id": "fixture-campaign-v1", "attempt_id": "attempt-r1",
            "claim_campaign_id": "attempt-r1", "task_id": "fixture.task",
            "arm_id": "kernelfoundry", "checkpoint_hours": 2.0,
            "broker_evaluation_chain": {
                "evaluation_count": 2,
                "terminal_receipt_sha256": receipts[-1]["receipt_sha256"],
                "selected_receipt_sha256": receipts[0]["receipt_sha256"],
                "source_paths": ["kernel.hip"]}}
        checkpoint["broker_evaluation_chain"]["baseline_receipt_sha256"] = "b" * 64
        checkpoint["measurement_windows"] = [{"receipt_sha256": "b" * 64}]
        R._validate_broker_chain(
            checkpoint, cell_root=cell, claim_scope="attempt-r1",
            arena_root=self.arena)
        second_path = cell / "controller-evaluation-windows" / "0002-result.json"
        broken = json.loads(second_path.read_text())
        broken["previous_receipt_sha256"] = "f" * 64
        broken["receipt_sha256"] = canonical_sha({
            key: value for key, value in broken.items() if key != "receipt_sha256"})
        second_path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "semantic identity"):
            R._validate_broker_chain(
                checkpoint, cell_root=cell, claim_scope="attempt-r1",
                arena_root=self.arena)

    def test_parent_broker_rejects_bad_token_before_evaluation(self):
        cell = self.root / "broker-reject"
        workspace = cell / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "config.yaml").write_text("x: y\n", encoding="utf-8")
        (workspace / "kernel.hip").write_text("// original\n", encoding="utf-8")
        evaluated = mock.Mock()
        broker = R._ControllerEvaluationBroker(
            request={"campaign_id": "fixture-campaign-v1",
                     "task": {"task_id": "fixture.task"},
                     "arm": {"arm_id": "k_search"}, "checkpoint_hours": 2.0},
            workspace=workspace, cell_root=cell, source_paths=("kernel.hip",),
            evaluate=evaluated, baseline_receipt_sha256="b" * 64)
        evaluator = object.__new__(U.ArenaWorkspaceEvaluator)
        evaluator.workspace = workspace
        evaluator.broker_receipts = []
        with broker:
            broker.register_controller(os.getpid())
            evaluator.broker_socket = broker.socket_path
            evaluator._broker_token = "wrong-token"
            evaluator._broker_owner_pid = os.getpid()
            with self.assertRaisesRegex(U.UpstreamControllerError, "broker failed"):
                evaluator._brokered_evaluation(
                    1, {"kernel.hip": b"// candidate\n"})
        evaluated.assert_not_called()

    def test_evaluator_execution_validation_rehashes_stdout_and_vendor_source(self):
        workspace = self.root / "validated-evaluator-workspace"
        evidence = self.root / "validated-evaluator-evidence"
        identity = {
            "campaign_id": "fixture-campaign-v1", "attempt_id": "attempt-r1",
            "claim_campaign_id": "attempt-r1", "task_id": "fixture.task",
            "arm_id": "k_search", "checkpoint_hours": 2.0,
            "arena_root": str(self.arena.resolve()),
        }
        execution = fake_evaluator_execution(
            workspace, evidence, identity, "centralized_final_evaluation",
            "c" * 64)
        evaluation = json.loads(
            (evidence / "evaluator-result.json").read_text())["evaluation"]
        R._validate_evaluator_execution(
            execution, expected_workspace=workspace,
            expected_phase="centralized_final_evaluation",
            expected_identity=identity,
            persisted_path=evidence / "execution-receipt.json",
            expected_evaluation=evaluation,
            expected_baseline_receipt_sha256="c" * 64,
            arena_root=self.arena)
        (evidence / "evaluator.stdout").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "output identity"):
            R._validate_evaluator_execution(
                execution, expected_workspace=workspace,
                expected_phase="centralized_final_evaluation",
                expected_identity=identity,
                persisted_path=evidence / "execution-receipt.json",
                expected_evaluation=evaluation,
                expected_baseline_receipt_sha256="c" * 64,
                arena_root=self.arena)

    def test_broker_queues_pre_registration_and_reuses_one_stream(self):
        cell = self.root / "broker-race"
        workspace = cell / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "config.yaml").write_text("x: y\n", encoding="utf-8")
        (workspace / "kernel.hip").write_text("// original\n", encoding="utf-8")
        calls = []

        def evaluate(ordinal, evaluation_root, cancel_event):
            calls.append(ordinal)
            self.assertFalse(cancel_event.is_set())
            return ({"pass_compilation": True, "pass_correctness": True,
                     "valid_optimized_cases": 1, "average_speedup": 1.0},
                    {"receipt_sha256": f"{ordinal:064x}"})

        request = {"campaign_id": "fixture-campaign-v1",
                   "task": {"task_id": "fixture.task"},
                   "arm": {"arm_id": "kernelfoundry"},
                   "checkpoint_hours": 2.0}
        broker = R._ControllerEvaluationBroker(
            request=request, workspace=workspace, cell_root=cell,
            source_paths=("kernel.hip",), evaluate=evaluate,
            baseline_receipt_sha256="b" * 64)

        def send(stream, ordinal):
            payload = json.dumps({
                "schema": U.BROKER_REQUEST_SCHEMA, "token": broker.token,
                "owner_pid": os.getpid(), "workspace": str(workspace),
                "evaluation_ordinal": ordinal,
                "source_files": {"kernel.hip": f"// {ordinal}\n"}},
                sort_keys=True, separators=(",", ":")).encode()
            stream.sendall(struct.pack("!Q", len(payload)) + payload)
            size = struct.unpack("!Q", U._recv_exact(stream, 8))[0]
            return json.loads(U._recv_exact(stream, size))

        with broker:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(broker.socket_path))
            output = []
            thread = threading.Thread(target=lambda: output.append(send(client, 1)))
            thread.start()
            time.sleep(0.1)
            self.assertTrue(thread.is_alive())
            self.assertEqual(calls, [])
            broker.register_controller(os.getpid())
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(output[0]["evaluation_ordinal"], 1)
            self.assertEqual(send(client, 2)["evaluation_ordinal"], 2)
            client.close()
        self.assertEqual(calls, [1, 2])

    def test_broker_client_rejects_replacement_server_pid(self):
        evaluator = object.__new__(U.ArenaWorkspaceEvaluator)
        evaluator.workspace = self.root
        evaluator.broker_receipts = []
        evaluator.broker_socket = self.root / "replacement.sock"
        evaluator._broker_token = "token"
        evaluator._broker_owner_pid = os.getpid() + 100000
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(evaluator.broker_socket))
        server.listen(1)
        thread = threading.Thread(target=lambda: server.accept()[0].close())
        thread.start()
        try:
            with self.assertRaisesRegex(U.UpstreamControllerError, "server identity"):
                evaluator._brokered_evaluation(1, {"x.py": b"pass\n"})
        finally:
            thread.join(timeout=2)
            server.close()

    def test_candidate_evaluation_failure_releases_short_claim(self):
        claim = FakeClaim()
        cell = self.root / "cells" / "failed-candidate"
        cell.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "candidate failed"):
            R._run_gpu_measurement_window(
                request=self.window_request(cell), cell_root=cell, ordinal=1,
                phase="controller_intermediate_evaluation",
                action=lambda: (_ for _ in ()).throw(RuntimeError("candidate failed")),
                claim_acquirer=lambda *args, **kwargs: claim,
                sampler_factory=FakeSampler)
        self.assertTrue(claim.released)

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
        baseline_type = types.SimpleNamespace
        evaluator.measure_baseline = mock.Mock(return_value=[
            baseline_type(test_case_id="case-1", shape=[1],
                          execution_time_ms=1.0, metadata={}),
            baseline_type(test_case_id="case-2", shape=[2],
                          execution_time_ms=2.0, metadata={}),
        ])
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
            "baseline": True,
            "checkpoint_hours": None,
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

        with (
            mock.patch.dict(sys.modules, {
                "src": src, "src.evaluator": evaluator,
                "src.prompt_builder": prompt_builder,
            }),
            mock.patch.object(
                R, "_assert_worker_evaluator_identity",
                return_value=R._evaluator_python_identity()),
        ):
            receipt = R.run_worker(
                request, claim_acquirer=acquire, sampler_factory=FakeSampler)
        self.assertTrue(receipt["evaluation"]["pass_correctness"])
        self.assertIsNone(receipt["checkpoint_hours"])
        self.assertTrue((cell_root / "workspace" / "task_result.yaml").is_file())
        self.assertFalse((cell_root / "controller.stdout").exists())
        evaluator.evaluate_kernel.assert_called_once()
        self.assertEqual(len(claims), 2)
        self.assertTrue(all(claim.released for claim in claims))
        self.assertEqual(
            [window["phase"] for window in receipt["measurement_windows"]],
            ["vendor_baseline", "centralized_final_evaluation"])
        self.assertEqual(
            receipt["constraints"]["evaluator_python"]["sha256"],
            R.EVALUATOR_PYTHON_SHA256)

        refused = dict(request)
        refused["baseline"] = False
        refused["checkpoint_hours"] = 2.0
        refused["cell_root"] = str((self.root / "cells" / "002-refused").resolve())
        Path(refused["cell_root"]).mkdir()
        evaluator.evaluate_compilation.reset_mock()
        evaluator.measure_baseline.reset_mock()
        evaluator.evaluate_kernel.reset_mock()
        claims.clear()
        with (
            mock.patch.dict(sys.modules, {
                "src": src, "src.evaluator": evaluator,
                "src.prompt_builder": prompt_builder}),
            mock.patch.object(
                R, "_assert_worker_evaluator_identity",
                return_value=R._evaluator_python_identity()),
            self.assertRaisesRegex(R.ArenaCellRunnerError, "exact arm audit"),
        ):
            R.run_worker(
                refused, claim_acquirer=acquire, sampler_factory=FakeSampler)
        evaluator.evaluate_compilation.assert_not_called()
        evaluator.measure_baseline.assert_not_called()
        evaluator.evaluate_kernel.assert_not_called()
        self.assertEqual(claims, [])

    def test_nonbaseline_routes_intermediate_and_final_evaluation_only_to_child(self):
        cell_root = self.root / "cells" / "003-brokered"
        cell_root.mkdir(parents=True)
        task_config = self.arena / "tasks" / "fixture" / "config.yaml"
        task_config.write_text(
            "task_type: hip2hip\nsource_file_path: [kernel.hip]\n"
            "target_kernel_functions: [kernel]\ncompile_command: [true]\n"
            "correctness_command: [true]\nperformance_command: [true]\n",
            encoding="utf-8")
        (task_config.parent / "kernel.hip").write_text(
            "// original\n", encoding="utf-8")
        evaluator = types.ModuleType("src.evaluator")
        evaluator.evaluate_compilation = mock.Mock(return_value=(True, None))
        evaluator.measure_baseline = mock.Mock(return_value=[
            types.SimpleNamespace(test_case_id="case-1", shape=[1],
                                  execution_time_ms=1.0, metadata={})])
        evaluator.evaluate_kernel = mock.Mock(
            side_effect=AssertionError("candidate evaluator ran in parent"))
        evaluator.write_task_result = lambda workspace, *args, **kwargs: (
            workspace / "task_result.yaml").write_text(
                "pass: true\n", encoding="utf-8")
        prompt_builder = types.ModuleType("src.prompt_builder")
        prompt_builder.prompt_builder = mock.Mock(return_value="Optimize.")
        src = types.ModuleType("src")
        src.evaluator = evaluator
        src.prompt_builder = prompt_builder
        request = {
            "schema": R.CHECKPOINT_SCHEMA,
            "campaign_id": "fixture-campaign-v1", "attempt_id": "attempt-r1",
            "claim_campaign_id": "attempt-r1",
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
            "baseline": False, "checkpoint_hours": 2.0, "visible_device": "0",
            "claim_journal": str((self.root / "claims.jsonl").resolve()),
            "claim_timeout_seconds": 0.0,
            "evaluator_python": R._evaluator_python_identity(),
            "arm_audit": {
                "arm_id": "k_search", "executable": True,
                "executable_path": str(Path(sys.executable).resolve()),
                "executable_sha256": R._sha256_file(Path(sys.executable).resolve()),
            },
        }
        claims: list[FakeClaim] = []
        child_sources: list[str] = []

        def acquire(*args, **kwargs):
            claim = FakeClaim(f"akd-child00000000{len(claims) + 1}")
            claim.receipt = lambda claim=claim: FakeReceipt(
                "opened", claim.claim_id, "attempt-r1")
            claim.release = lambda claim=claim: FakeReceipt(
                "released", claim.claim_id, "attempt-r1")
            claims.append(claim)
            return claim

        class FakeChildRunner:
            def __init__(inner_self, *, arena_root):
                inner_self.arena_root = arena_root

            def run(inner_self, *, request, evaluation_root, evidence_root,
                    timeout_s, cancel_event):
                child_sources.append((evaluation_root / "kernel.hip").read_text())
                evaluation = {
                    "average_speedup": 1.25,
                    "best_optimized_execution_time": 0.8,
                    "compilation_error_message": None,
                    "correctness_error_message": None,
                    "pass_compilation": True, "pass_correctness": True,
                    "valid_baseline_cases": 1, "valid_optimized_cases": 1,
                }
                result = R.arena_evaluator_child.self_hash({
                    "schema": R.arena_evaluator_child.RESULT_SCHEMA,
                    "request_receipt_sha256": request["receipt_sha256"],
                    "baseline_cases_sha256": request["baseline_cases"][
                        "receipt_sha256"],
                    "outer_baseline_receipt_sha256": request[
                        "outer_baseline_receipt_sha256"],
                    "evaluation": evaluation})
                stdout = json.dumps(result, sort_keys=True) + "\n"
                (evidence_root / "evaluator.stdout").write_text(
                    stdout, encoding="utf-8")
                (evidence_root / "evaluator.stderr").write_text(
                    "", encoding="utf-8")
                return R.EvaluatorChildResult(
                    result=result, pid=22222, process_start_ticks=33333,
                    process_group_id=22222, session_id=22222,
                    activation_receipt={"fixture": True},
                    teardown_receipt={"verified_empty": True, "removed": True},
                    stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                    stderr_sha256=hashlib.sha256(b"").hexdigest())

        def fake_launch(prepared, argv, *, timeout_seconds, command_prefix,
                        process_started):
            self.assertEqual(
                prepared.environment["PYTHONPATH"],
                str(R.REPOSITORY_ROOT / "scripts"))
            process_started(os.getpid())
            broker_client = object.__new__(U.ArenaWorkspaceEvaluator)
            broker_client.workspace = Path(prepared.task.workspace)
            broker_client.broker_receipts = []
            broker_client.broker_socket = Path(
                prepared.environment[U.BROKER_SOCKET_ENV])
            broker_client._broker_token = prepared.environment[U.BROKER_TOKEN_ENV]
            broker_client._broker_owner_pid = int(
                prepared.environment[U.BROKER_OWNER_PID_ENV])
            candidate = "// candidate\n"
            broker_client._brokered_evaluation(
                1, {"kernel.hip": candidate.encode()})
            digest = hashlib.sha256(candidate.encode()).hexdigest()
            return json.dumps({"evaluation": {
                "best_source_sha256": {"kernel.hip": digest}}}) + "\n"

        def fake_isolated_launch(*, prepared, argv, timeout_seconds, broker,
                                 invocation, cell_root):
            stdout = fake_launch(
                prepared, argv, timeout_seconds=timeout_seconds,
                command_prefix=(), process_started=broker.register_controller)
            return stdout, fake_controller_sandbox_execution(cell_root)

        with (
            mock.patch.dict(sys.modules, {
                "src": src, "src.evaluator": evaluator,
                "src.prompt_builder": prompt_builder}),
            mock.patch.object(R, "_assert_worker_evaluator_identity",
                              return_value=R._evaluator_python_identity()),
            mock.patch.object(R, "_controller_runtime_allowlist",
                              return_value=mock.sentinel.runtime),
            mock.patch.object(
                R.arena_controller_sandbox, "prepare_controller_sandbox",
                return_value=types.SimpleNamespace(environment_overrides={})),
            mock.patch.object(R, "_launch_isolated_controller",
                              side_effect=fake_isolated_launch),
        ):
            receipt = R.run_worker(
                request, claim_acquirer=acquire, sampler_factory=FakeSampler,
                evaluator_runner_factory=FakeChildRunner)
        evaluator.evaluate_kernel.assert_not_called()
        self.assertEqual(child_sources, ["// candidate\n", "// candidate\n"])
        self.assertEqual(len(claims), 3)
        self.assertEqual(receipt["evaluation"]["average_speedup"], 1.25)
        self.assertIn(
            "evaluator_execution_receipt",
            receipt["measurement_windows"][1])

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
