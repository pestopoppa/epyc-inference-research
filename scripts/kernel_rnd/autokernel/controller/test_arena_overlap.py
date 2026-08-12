#!/usr/bin/env python3
"""Adversarial tests for governed AutoKernel controller-lane overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from . import arena_campaign as C
from . import arena_cell_runner as R


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArenaOverlapTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.arena = self.root / "arena"
        self.geak = self.root / "geak"
        self.arena.mkdir()
        self.geak.mkdir()
        self.preflight = self.root / "preflight.json"
        self.preflight.write_text("{}\n", encoding="utf-8")
        self.config_path = self.root / "campaign.json"
        self.config_path.write_text("{}\n", encoding="utf-8")
        self.cpus = tuple(sorted(os.sched_getaffinity(0)))
        if len(self.cpus) < 2:
            self.skipTest("overlap tests require two available CPUs")

    @staticmethod
    def request(index: int, *, baseline: bool = False):
        task = C.TaskArtifact(
            task_id=f"fixture.task{index}", relative_root=f"tasks/{index}",
            file_sha256={"config.yaml": "a" * 64})
        arm_ids = {1: "k_search", 2: "geak_v1"}
        # The scheduler test exercises only immutable identity/order. Construct
        # a typed arm without invoking each real controller's pin-heavy policy.
        arm = object.__new__(C.ArmImplementation)
        object.__setattr__(
            arm, "arm_id", C.BASELINE_ARM_ID if baseline else arm_ids[index])
        return C.CampaignCellRequest(
            arm=arm, task=task, is_starting_state_baseline=baseline,
            checkpoint_hours=() if baseline else C.MATCHED_BUDGET_HOURS,
            maximum_wall_hours=0.0 if baseline else C.MATCHED_BUDGET_HOURS[-1])

    def schedule(self):
        rows = []
        for index, baseline in ((0, True), (1, False), (2, False)):
            request = self.request(index, baseline=baseline)
            rows.append({
                "schedule_index": index,
                "lane_name": f"lane-{index:04d}-fixture-task{index}-{request.arm.arm_id}",
                "task_id": request.task.task_id,
                "arm_id": request.arm.arm_id,
                "baseline": baseline,
                "checkpoint_names": ["baseline"] if baseline else ["2h", "8h", "32h"],
                "request": request,
            })
        return rows

    def execution_inputs(self, *, width: int = 2):
        schedule = self.schedule()
        output = self.root / "attempt-r1"
        args = types.SimpleNamespace(
            output_root=str(output), arena_root=str(self.arena),
            geak_root=str(self.geak), preflight=str(self.preflight))
        manifest = {
            "attempt_id": output.name,
            "available_source": True,
            "claim_journal": str(self.root / "claim.jsonl"),
            "claim_lock_path": str(R.device_claim.device_lock_path(
                R.DEFAULT_DEVICE_ID).resolve()),
            "claim_timeout_seconds": 10.0,
            "runner": {"sha256": "d" * 64,
                       "taskset_sha256": sha(R.TASKSET_EXECUTABLE)},
            "matrix": {"lanes": [
                {key: value for key, value in row.items() if key != "request"}
                for row in schedule]},
            "overlap": {
                "controller_width": width,
                "controller_cpu_set": list(self.cpus[:-1]),
                "evaluator_cpu_set": [self.cpus[-1]],
            },
        }
        spec = types.SimpleNamespace(
            config_path=str(self.config_path), config_sha256=sha(self.config_path))
        audit = {
            "campaign_id": "fixture-campaign-v1",
            "execution_identity": {"implementation_module_sha256": "e" * 64},
            "sources": {},
        }
        return schedule, args, spec, audit, manifest

    def test_reverse_future_completion_retains_schedule_order_and_observes_overlap(self):
        schedule, args, spec, audit, manifest = self.execution_inputs()

        class FakeRunner:
            def __init__(inner, config, *, cancel_event, overlap_tracker):
                inner.config = config
                inner.cancel = cancel_event
                inner.tracker = overlap_tracker

            def __call__(inner, request):
                if inner.config.schedule_index == 0:
                    return {"task_id": request.task.task_id, "arm_id": request.arm.arm_id,
                            "schedule_index": 0}
                with inner.tracker.active():
                    time.sleep(0.10 if inner.config.schedule_index == 1 else 0.02)
                return {"task_id": request.task.task_id, "arm_id": request.arm.arm_id,
                        "schedule_index": inner.config.schedule_index}

        with mock.patch.object(R, "_campaign_schedule", return_value=schedule), \
                mock.patch.object(R, "GovernedArenaCellRunner", FakeRunner):
            lease_fd = os.open(self.preflight, os.O_RDONLY)
            try:
                results, observed = R._execute_v3_schedule(
                    args, spec, audit, manifest, lease_fd=lease_fd)
            finally:
                os.close(lease_fd)
        self.assertEqual([row["schedule_index"] for row in results], [0, 1, 2])
        self.assertEqual(observed["observed_peak_live_controller_workers"], 2)
        self.assertTrue(observed["controller_overlap_observed"])

    def test_first_lane_failure_cancels_peer(self):
        schedule, args, spec, audit, manifest = self.execution_inputs()
        peer_cancelled = threading.Event()

        class FakeRunner:
            def __init__(inner, config, *, cancel_event, overlap_tracker):
                inner.config, inner.cancel = config, cancel_event

            def __call__(inner, request):
                if inner.config.schedule_index == 0:
                    return {"task_id": request.task.task_id, "arm_id": request.arm.arm_id}
                if inner.config.schedule_index == 1:
                    time.sleep(0.02)
                    raise RuntimeError("planted first failure")
                if inner.cancel.wait(2):
                    peer_cancelled.set()
                    raise R.ArenaCellRunnerError("cancelled")
                raise AssertionError("peer did not receive cancellation")

        with mock.patch.object(R, "_campaign_schedule", return_value=schedule), \
                mock.patch.object(R, "GovernedArenaCellRunner", FakeRunner), \
                self.assertRaisesRegex(R.ArenaCellRunnerError, "lane 1 failed"):
            lease_fd = os.open(self.preflight, os.O_RDONLY)
            try:
                R._execute_v3_schedule(
                    args, spec, audit, manifest, lease_fd=lease_fd)
            finally:
                os.close(lease_fd)
        self.assertTrue(peer_cancelled.is_set())

    def test_attempt_lease_refuses_duplicate_until_inherited_descriptor_closes(self):
        output = self.root / "attempt-r1"
        with R._attempt_lease(output) as (descriptor, _path):
            inherited = os.dup(descriptor)
            with self.assertRaisesRegex(R.ArenaCellRunnerError, "already live"):
                with R._attempt_lease(output):
                    pass
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "already live"):
            with R._attempt_lease(output):
                pass
        os.close(inherited)
        with R._attempt_lease(output):
            pass

    def test_claim_object_preflight_is_durable_and_releases_without_gpu_action(self):
        output = self.root / "attempt-r1"
        output.mkdir()
        manifest = {
            "attempt_id": output.name, "claim_campaign_id": output.name,
            "claim_lock_path": str(self.root / "gpu.lock"),
            "claim_journal": str(self.root / "claims.jsonl"),
            "claim_timeout_seconds": 10.0,
        }

        class Receipt:
            def __init__(inner, released):
                inner.released = released

            def to_dict(inner):
                return {"claim_id": "fixture", "released_at": (
                    "2026-08-12T00:00:01Z" if inner.released else None)}

        class Claim:
            def receipt(inner):
                return Receipt(False)

            def release(inner):
                return Receipt(True)

        revocation = self.root / "revoke.json"
        acquire = mock.Mock(return_value=Claim())
        with mock.patch.object(R.device_claim, "acquire_device_claim", acquire), \
                mock.patch.object(
                    R.device_claim, "revocation_path", return_value=revocation):
            first = R._prepare_lane_claim_objects(output, manifest)
            second = R._prepare_lane_claim_objects(output, manifest)
        self.assertEqual(first, second)
        self.assertEqual(acquire.call_count, 1)
        self.assertFalse(first["gpu_action_executed"])
        self.assertIsNotNone(first["device_claim_released"]["released_at"])

    def test_worker_cancel_terminates_exact_captured_group(self):
        cell = self.root / "cell"
        cell.mkdir()
        pid_path = cell / "pid"
        script = (
            "import os,pathlib,time;"
            f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()));"
            "time.sleep(30)")
        cancel = threading.Event()
        owner = types.SimpleNamespace(_cancel_event=cancel)
        request = {
            "cell_root": str(cell), "repository_root": str(self.root),
            "visible_device": "0", "controller_cpu_set": [self.cpus[0]],
        }
        timer = threading.Timer(0.15, cancel.set)
        timer.start()
        try:
            with mock.patch.object(
                    R, "_worker_command", return_value=(sys.executable, "-c", script)), \
                    self.assertRaisesRegex(R.ArenaCellRunnerError, "cancelled"):
                R.GovernedArenaCellRunner._run_worker_subprocess(owner, request, 5)
        finally:
            timer.cancel()
        pid = int(pid_path.read_text(encoding="utf-8"))
        self.assertFalse(Path(f"/proc/{pid}").exists())

    def v3_shape(self):
        root = self.root / "attempt-r1"
        lane_name = "lane-0000-fixture-task-arena-baseline"
        cell_name = (
            f"001-{R._path_id('fixture.task', 'task_id')}-"
            f"{R._path_id(C.BASELINE_ARM_ID, 'arm_id')}-baseline")
        cell = root / "execution" / "lanes" / lane_name / "cells" / cell_name
        receipts = cell.parents[1] / "cell-receipts"
        cell.mkdir(parents=True)
        receipts.mkdir()
        (cell / "checkpoint-receipt.json").write_text("{}", encoding="utf-8")
        (receipts / f"001-fixture.task-{C.BASELINE_ARM_ID}.json").write_text(
            "{}", encoding="utf-8")
        manifest = {
            "claim_timeout_seconds": 10.0,
            "claim_journal": str(self.root / "claims.jsonl"),
            "claim_lock_path": str(self.root / "device.lock"),
            "runner": {"taskset_path": str(R.TASKSET_EXECUTABLE),
                       "taskset_sha256": sha(R.TASKSET_EXECUTABLE)},
            "attempt_lease_path": str(
                (root.parent / f".{root.name}.{R.ATTEMPT_LEASE_NAME}").resolve()),
            "matrix": {"checkpoint_hours": [2.0, 8.0, 32.0], "lanes": [{
                "schedule_index": 0, "lane_name": lane_name,
                "task_id": "fixture.task", "arm_id": C.BASELINE_ARM_ID,
                "baseline": True, "checkpoint_names": ["baseline"]}]},
            "overlap": {"controller_width": 2,
                        "controller_cpu_set": [self.cpus[0]],
                        "evaluator_cpu_set": [self.cpus[-1]],
                        "aa_calibration": True, "aa_receipt": None,
                        "concurrent_results_rankable": False},
            "worker_timeout": {
                "formula": "checkpoint_hours*3600 + 2*claim_timeout_seconds + evaluation_reserve_seconds",
                "evaluation_reserve_seconds": R.EVALUATION_RESERVE_SECONDS,
                "maximum_seconds": 32 * 3600 + 20 + R.EVALUATION_RESERVE_SECONDS},
        }
        return root, manifest

    def test_v3_lane_validation_rejects_extra_missing_symlink_and_multi_cell(self):
        root, manifest = self.v3_shape()
        checkpoints, cells = R._validate_v3_lane_shape(root, manifest)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(len(cells), 1)
        lane = root / "execution" / "lanes" / manifest["matrix"]["lanes"][0]["lane_name"]
        extra = lane / "peer-write"
        extra.write_text("escape", encoding="utf-8")
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "extra object"):
            R._validate_v3_lane_shape(root, manifest)
        extra.unlink()
        second = lane / "cells" / "002-fixture-task-arena-baseline-baseline"
        second.mkdir()
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "cell shape"):
            R._validate_v3_lane_shape(root, manifest)
        second.rmdir()
        checkpoint = next((lane / "cells").glob("*/checkpoint-receipt.json"))
        checkpoint.unlink()
        checkpoint.symlink_to(self.preflight)
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "missing checkpoint"):
            R._validate_v3_lane_shape(root, manifest)

    def test_landlock_worker_boundary_denies_peer_lane_write(self):
        own = self.root / "own"
        peer = self.root / "peer"
        shared = self.root / "shared"
        journal = self.root / "claim.jsonl"
        own.mkdir()
        peer.mkdir()
        shared.mkdir()
        journal.write_text("", encoding="utf-8")
        script = """
import pathlib, sys
from scripts.kernel_rnd.autokernel.execution import sandbox
own, peer, shared, journal = map(pathlib.Path, sys.argv[1:])
sandbox.install_landlock(
    str(own), additional_writable_roots=(str(shared),),
    writable_files=(str(journal),))
(own / 'allowed').write_text('ok')
(shared / 'allowed').write_text('ok')
with journal.open('a') as handle:
    handle.write('ok\\n')
try:
    (peer / 'denied').write_text('bad')
except PermissionError:
    raise SystemExit(0)
raise SystemExit(9)
"""
        completed = subprocess.run(
            (sys.executable, "-c", script, str(own), str(peer), str(shared),
             str(journal)),
            cwd=Path(__file__).parents[4],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTHONPATH": str(Path(__file__).parents[4])},
            capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((own / "allowed").is_file())
        self.assertTrue((shared / "allowed").is_file())
        self.assertEqual(journal.read_text(encoding="utf-8"), "ok\n")
        self.assertFalse((peer / "denied").exists())

    def test_width_one_never_claims_observed_overlap(self):
        tracker = R.OverlapTracker(1)
        with tracker.active():
            pass
        fields = tracker.receipt_fields()
        self.assertEqual(fields["observed_peak_live_controller_workers"], 1)
        self.assertFalse(fields["controller_overlap_observed"])

    def test_schedule_and_lane_identities_are_exactly_restart_deterministic(self):
        spec = C.load_spec(
            Path(__file__).with_name("arena_campaign_v1.json"))
        first = R._campaign_schedule(spec, available_source=True)
        second = R._campaign_schedule(spec, available_source=True)
        def project(rows):
            return [
                {key: value for key, value in row.items() if key != "request"}
                for row in rows]
        self.assertEqual(project(first), project(second))
        names = [row["lane_name"] for row in first]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            [row["schedule_index"] for row in first], list(range(len(first))))
        self.assertTrue(all(
            row["checkpoint_names"] == ["baseline"] if row["baseline"]
            else row["checkpoint_names"] == ["2h", "8h", "32h"]
            for row in first))

    def test_aa_receipt_must_match_width_cpu_geometry_and_noise_gate(self):
        controller_cpus = (self.cpus[0],)
        evaluator_cpus = (self.cpus[-1],)
        receipt = R._self_hash({
            "schema": R.OVERLAP_AA_SCHEMA, "status": "pass",
            "controller_width": 2,
            "controller_cpu_set": list(controller_cpus),
            "evaluator_cpu_set": list(evaluator_cpus),
            "observed_peak_live_controller_workers": 2,
            "predeclared_noise_bound_pct": 3.0,
            "observed_max_noise_pct": 1.0,
            "within_predeclared_noise_bound": True,
        })
        path = self.root / "aa.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        loaded, loaded_path = R._load_overlap_aa(
            str(path), width=2, controller_cpus=controller_cpus,
            evaluator_cpus=evaluator_cpus)
        self.assertEqual(loaded, receipt)
        self.assertEqual(loaded_path, str(path.resolve()))
        receipt["within_predeclared_noise_bound"] = False
        receipt["receipt_sha256"] = R._canonical_sha256({
            key: value for key, value in receipt.items()
            if key != "receipt_sha256"})
        path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "does not authorize"):
            R._load_overlap_aa(
                str(path), width=2, controller_cpus=controller_cpus,
                evaluator_cpus=evaluator_cpus)

    def test_concurrent_manifest_requires_positive_wait_and_aa_or_calibration(self):
        args = argparse.Namespace(
            output_root=str(self.root / "attempt-r1"),
            arena_root=str(self.arena), geak_root=str(self.geak),
            preflight=str(self.preflight), claim_journal=str(self.root / "claims"),
            claim_timeout_seconds=0.0, controller_width=2,
            controller_cpus=str(self.cpus[0]), evaluator_cpus=str(self.cpus[-1]),
            overlap_aa_receipt=None, overlap_aa_calibration=True)
        spec = types.SimpleNamespace(
            config_path=str(self.config_path), config_sha256=sha(self.config_path),
            tasks=(), arms=(), budget_hours=C.MATCHED_BUDGET_HOURS)
        audit = {"campaign_id": "fixture", "authority": "diagnostic",
                 "receipt_sha256": "a" * 64, "schema": "fixture",
                 "sources": {}, "panel": {"arms": []}}
        preflight = R._self_hash({
            "schema": R.arena_adapter.PREFLIGHT_SCHEMA,
            "hardware": {"target_gfx_arch": "gfx90a"}})
        self.preflight.write_text(json.dumps(preflight), encoding="utf-8")
        with self.assertRaisesRegex(R.ArenaCellRunnerError, "positive claim timeout"):
            R._run_manifest(args, spec, audit, available_source=True)
        args.claim_timeout_seconds = 10.0
        manifest = R._run_manifest(args, spec, audit, available_source=True)
        self.assertFalse(manifest["overlap"]["concurrent_results_rankable"])
        self.assertIn("2*claim_timeout_seconds", manifest["worker_timeout"]["formula"])


if __name__ == "__main__":
    unittest.main()
