from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import arena_adapter
from . import hip_authoring_arm as H


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeReceipt:
    def __init__(self, claim_id: str, released: bool = False):
        self.claim_id = claim_id
        self.released = released

    def to_dict(self):
        return {"claim_id": self.claim_id,
                "released_at": "done" if self.released else None}


class FakeClaim:
    def __init__(self, claim_id: str):
        self.claim_id = claim_id
        self.released = False

    def receipt(self):
        return FakeReceipt(self.claim_id, self.released)

    def release(self):
        self.released = True
        return FakeReceipt(self.claim_id, True)


class FakeSampling:
    def to_dict(self):
        return {"samples": [{"gpu_use_percent": 100.0}]}


class FakeSampler:
    def start(self):
        return self

    def stop(self):
        return FakeSampling()


class HipAuthoringArmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.arena = self.root / "arena"
        task = self.arena / "tasks/torch2hip/gpumode/16636_SiLU"
        (task / "hip").mkdir(parents=True)
        (task / "source.py").write_text("def get_inputs(): pass\n")
        (task / "hip/candidate.hip").write_text("")
        (task / "config.yaml").write_text(
            "task_type: torch2hip\n"
            "source_file_path: [source.py]\n"
            "target_kernel_functions: [forward]\n"
            "target_file_path: hip/candidate.hip\n"
            "compile_command: [compile]\n"
            "correctness_command: [correct]\n"
            "performance_command: [perf]\n")
        (self.arena / "LICENSE").write_text("Apache-2.0\n")
        self.candidate = self.root / "silu.hip"
        self.candidate.write_text("// non-empty candidate\n")
        self.vendor = {
            "name": "AgentKernelArena", "commit": arena_adapter.AGENT_KERNEL_ARENA_PIN.commit,
            "clean": True, "license": {"path": "LICENSE", "sha256": "0" * 64},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self):
        with mock.patch.object(H.arena_adapter, "inspect_vendor_source",
                               return_value=self.vendor):
            return H.audit_task(self.arena, "torch2hip/gpumode/16636_SiLU")

    def test_audit_refuses_a_triton_namesake(self):
        with self.assertRaisesRegex(H.HipAuthoringError, "exact torch2hip"):
            H.audit_task(self.arena, "instruction2triton/gpumode/16636_SiLU")

    def test_audit_binds_every_task_file_and_true_type(self):
        audit = self.audit()
        self.assertEqual(audit.target_file, "hip/candidate.hip")
        self.assertEqual(audit.target_functions, ("forward",))
        self.assertEqual(set(audit.file_sha256), {
            "config.yaml", "hip/candidate.hip", "source.py"})
        self.assertEqual(audit.to_dict()["task_type"], "torch2hip")

    def test_measurement_window_releases_on_failure(self):
        claims = []

        def acquire(*args, **kwargs):
            claim = FakeClaim("akd-window")
            claims.append(claim)
            return claim

        with self.assertRaisesRegex(RuntimeError, "boom"):
            H._measurement_window(
                phase="vendor_baseline", task_id="torch2hip/gpumode/16636_SiLU",
                campaign_id="ak-hip-test", output_root=self.root / "window",
                claim_journal=self.root / "claim.jsonl", visible_device="0",
                claim_timeout_s=0.0, action=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                claim_acquirer=acquire, sampler_factory=lambda **kwargs: FakeSampler())
        self.assertTrue(claims[0].released)
        receipt = json.loads((self.root / "window/measurement-windows/vendor_baseline.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["device_claim_released"]["released_at"], "done")

    def test_roundtrip_is_observation_only_and_uses_two_distinct_claims(self):
        claims = []

        def acquire(*args, **kwargs):
            claim = FakeClaim(f"akd-{len(claims)}")
            claims.append(claim)
            return claim

        class Vendor:
            @staticmethod
            def evaluate_compilation(*args):
                return True, None

            @staticmethod
            def measure_baseline(*args):
                return [object()]

            @staticmethod
            def evaluate_kernel(*args):
                return {"pass_compilation": True, "pass_correctness": True,
                        "valid_baseline_cases": 11, "valid_optimized_cases": 11,
                        "average_speedup": 1.25}

        with (mock.patch.object(H.arena_adapter, "inspect_vendor_source",
                                return_value=self.vendor),
              mock.patch.object(H, "_load_vendor_evaluator", return_value=Vendor),
              mock.patch.object(H, "toolchain_identity",
                                return_value={"ninja": {"version": "1.13.0"}}),
              mock.patch.object(H, "_public_correctness_cases", return_value=11)):
            receipt = H.run_roundtrip(
                arena_root=self.arena, task_id="torch2hip/gpumode/16636_SiLU",
                candidate_source=self.candidate, output_root=self.root / "run",
                campaign_id="ak-hip-test", claim_journal=self.root / "claim.jsonl",
                arch_detector=lambda: {"architectures": ["gfx90a"]},
                claim_acquirer=acquire,
                sampler_factory=lambda **kwargs: FakeSampler())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["authority"], "observation_only")
        self.assertEqual(receipt["producer"]["producer_id"], H.PRODUCER_ID)
        self.assertTrue(receipt["started_at"].endswith("Z"))
        self.assertTrue(receipt["ended_at"].endswith("Z"))
        self.assertFalse(receipt["evaluation"]["speedup_rankable"])
        self.assertIn("honest_vendor_baseline_not_bound",
                      receipt["evaluation"]["integrity_flags"])
        self.assertEqual(
            [row["measurement_id"] for row in receipt["belief_measurements"]],
            ["hip_public_correctness_pass_rate", "hip_timing_harness_validity_rate"])
        self.assertTrue(all(row["reps"] == 11 for row in receipt["belief_measurements"]))
        self.assertEqual([claim.claim_id for claim in claims], ["akd-0", "akd-1"])
        self.assertTrue(all(claim.released for claim in claims))
        self.assertEqual(sha(self.root / "run/workspace/hip/candidate.hip"),
                         sha(self.candidate))
        without_hash = {key: value for key, value in receipt.items()
                        if key != "receipt_sha256"}
        self.assertEqual(receipt["receipt_sha256"], H._canonical_sha256(without_hash))

    def test_existing_output_root_is_a_refusal(self):
        output = self.root / "exists"
        output.mkdir()
        with self.assertRaisesRegex(H.HipAuthoringError, "already exists"):
            H._fresh_workspace(output, self.audit(), self.candidate)

    def test_polite_signal_becomes_a_cleanup_exception(self):
        import signal

        with H._graceful_signals():
            with self.assertRaisesRegex(H.HipAuthoringInterrupted, "SIGTERM"):
                signal.raise_signal(signal.SIGTERM)

    def test_claim_assignment_is_signal_deferred(self):
        states = []

        def acquire(*args, **kwargs):
            states.append(H.signal.pthread_sigmask(H.signal.SIG_BLOCK, set()))
            return FakeClaim("akd-deferred")

        with mock.patch.object(H.signal, "pthread_sigmask",
                               wraps=H.signal.pthread_sigmask):
            result, _ = H._measurement_window(
                phase="vendor_baseline",
                task_id="torch2hip/gpumode/16636_SiLU",
                campaign_id="ak-hip-signal-test",
                output_root=self.root / "signal-window",
                claim_journal=self.root / "claim.jsonl",
                visible_device="0", claim_timeout_s=0.0,
                action=lambda: "ok", claim_acquirer=acquire,
                sampler_factory=lambda **kwargs: FakeSampler())
        self.assertEqual(result, "ok")
        self.assertIn(H.signal.SIGTERM, states[0])
        self.assertIn(H.signal.SIGINT, states[0])


if __name__ == "__main__":
    unittest.main()
