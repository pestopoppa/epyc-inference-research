#!/usr/bin/env python3
"""Focused tests for the process-only AutoKernel fault rehearsal.

The one real smoke uses only private Python children and a fake device id below
an exact temporary directory on ``/mnt/raid0``.  It never touches a GPU, model,
benchmark, kernel tree, stack process, or live claim root.
"""

from __future__ import annotations

import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path

_KERNEL_RND = str(Path(__file__).resolve().parents[1])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import fault_rehearsal as fr  # noqa: E402


class _ScriptedAdapter:
    """Injected process adapter; no OS process or signal is involved."""

    def __init__(self, waits):
        self.waits = list(waits)
        self.signals = []
        self.alive = True

    def spawn(self, argv, stdout_path, stderr_path):  # pragma: no cover - unused seam
        raise AssertionError("spawn not expected")

    def is_alive(self, process):
        return self.alive

    def wait(self, process, timeout_s):
        result = self.waits.pop(0)
        if result is not None:
            self.alive = False
        return result

    def signal_group(self, process, signal_number):
        self.signals.append(signal_number)

    def verify_dead(self, process):
        return not self.alive


def _fake_process() -> fr.OwnedProcess:
    identity = fr.ProcessIdentity(
        pid=1234,
        pgid=1234,
        start_ticks=5678,
        boot_id="fixture-boot",
        argv=("python", "fixture"),
        argv_sha256="0" * 64,
    )
    return fr.OwnedProcess(identity, object(), Path("stdout"), Path("stderr"))


def _dependency_receipt() -> dict:
    crash = _fake_process().identity.to_dict()
    restart = {**crash, "pid": 2345, "pgid": 2345, "start_ticks": 6789}
    holder = {**crash, "pid": 3456, "pgid": 3456, "start_ticks": 7890}
    receipt = {
        "schema": fr.RECEIPT_SCHEMA,
        "capture_mode": fr.CAPTURE_MODE,
        "campaign_id": "ak-fault-rehearsal-dependency-fixture",
        "status": "PASS",
        "started_at_epoch_s": 1.0,
        "completed_at_epoch_s": 2.0,
        "environment": {
            "source_tree": {
                "root": "/source", "branch": "fixture", "commit": "a" * 40,
            },
            "producer_path": "scripts/kernel_rnd/autokernel/fault_rehearsal.py",
            "producer_sha256": "b" * 64,
        },
        "authority": fr._authority_boundary(),
        "live_claim_root_touched": False,
        "process_selection": "captured_children_only_no_name_pattern_scan",
        "legs": [
            {
                "name": "durable_journal_crash_restart_replay", "status": "PASS",
                "crash_process": crash, "restart_process": restart,
            },
            {
                "name": "resource_revocation_non_preemption", "status": "PASS",
                "teardown": {"identity": holder},
            },
            {"name": "hash_bound_artifact_tamper_refusal", "status": "PASS"},
        ],
    }
    receipt["dependency_evidence"] = fr._dependency_evidence_rows(receipt)
    receipt["receipt_sha256"] = fr._sha256_bytes(fr._canonical_bytes(receipt))
    return receipt


class TestInjectedProcessAdapter(unittest.TestCase):
    def test_term_then_kill_only_after_term_timeout(self):
        adapter = _ScriptedAdapter([None, -signal.SIGKILL])
        receipt = fr.terminate_owned_process(
            _fake_process(), adapter, term_grace_s=0.01, kill_grace_s=0.01
        )
        self.assertEqual(adapter.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertEqual(receipt["actions"], ["SIGTERM", "SIGKILL"])
        self.assertTrue(receipt["verified_dead"])

    def test_successful_term_does_not_send_kill(self):
        adapter = _ScriptedAdapter([-signal.SIGTERM])
        receipt = fr.terminate_owned_process(_fake_process(), adapter)
        self.assertEqual(adapter.signals, [signal.SIGTERM])
        self.assertEqual(receipt["actions"], ["SIGTERM"])


class TestTamperBoundary(unittest.TestCase):
    def test_changed_bytes_are_refused(self):
        with tempfile.TemporaryDirectory(
            prefix="ak_fault_hash_", dir="/mnt/raid0/llm/tmp"
        ) as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"before")
            expected = fr._sha256_bytes(b"before")
            self.assertEqual(fr.read_hash_bound_artifact(path, expected), b"before")
            path.write_bytes(b"after")
            with self.assertRaises(fr.TamperRefusal):
                fr.read_hash_bound_artifact(path, expected)


class TestDependencyEvidenceSeam(unittest.TestCase):
    def test_three_legs_share_one_run_support_key_and_are_not_measurements_or_witnesses(self):
        receipt = _dependency_receipt()
        rows = receipt["dependency_evidence"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["support_key"] for row in rows}), 1)
        self.assertEqual(len({row["evidence_id"] for row in rows}), 3)
        self.assertTrue(all(row["support_scope"] == "rehearsal_run" for row in rows))
        self.assertTrue(all(row["run_status"] == "PASS" for row in rows))
        self.assertTrue(all(row["performance_measurement"] is False for row in rows))
        self.assertTrue(all(row["corroborating_witness"] is False for row in rows))
        self.assertTrue(all(row["belief_measurement_emitted"] is False for row in rows))
        self.assertEqual([len(row["process_identities"]) for row in rows], [2, 1, 0])
        self.assertEqual(fr.validate_receipt(receipt), [])

    def test_dependency_row_tamper_fails_closed(self):
        receipt = _dependency_receipt()
        receipt["dependency_evidence"][0]["support_key"] = "per-leg-is-forbidden"
        violations = fr.validate_receipt(receipt)
        self.assertTrue(any("rehearsal-run key" in item for item in violations))
        self.assertTrue(any("receipt_sha256" in item for item in violations))


class TestRealProcessSmoke(unittest.TestCase):
    def test_complete_rehearsal_is_real_process_only_and_atomic(self):
        with tempfile.TemporaryDirectory(
            prefix="ak_fault_smoke_", dir="/mnt/raid0/llm/tmp"
        ) as temporary:
            target = Path(temporary) / "published"
            receipt = fr.run_fault_rehearsal(target, campaign_id="ak-fault-rehearsal-unit-smoke")
            self.assertEqual(receipt["status"], "PASS", receipt)
            self.assertEqual(receipt["capture_mode"], fr.CAPTURE_MODE)
            self.assertEqual([leg["status"] for leg in receipt["legs"]], ["PASS", "PASS", "PASS"])
            self.assertFalse(receipt["live_claim_root_touched"])
            self.assertTrue(all(value is False for value in receipt["authority"].values()))
            source_tree = receipt["environment"]["source_tree"]
            self.assertRegex(source_tree["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(source_tree["branch"])
            self.assertTrue((target / "receipt.json").is_file())
            disk = json.loads((target / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(fr.validate_receipt(disk), [])
            claimed_hash = disk.pop("receipt_sha256")
            self.assertEqual(claimed_hash, fr._sha256_bytes(fr._canonical_bytes(disk)))
            resource_leg = receipt["legs"][1]
            self.assertEqual(resource_leg["compliance_outcome_while_alive"], "FAIL")
            self.assertTrue(resource_leg["holder_alive_after_deadline"])
            self.assertTrue(resource_leg["teardown"]["verified_dead"])

            with self.assertRaises(FileExistsError):
                fr.run_fault_rehearsal(target)

    def test_receipt_validator_rejects_changed_leg_bytes(self):
        with tempfile.TemporaryDirectory(
            prefix="ak_fault_receipt_", dir="/mnt/raid0/llm/tmp"
        ) as temporary:
            target = Path(temporary) / "published"
            receipt = fr.run_fault_rehearsal(
                target, campaign_id="ak-fault-rehearsal-validator-smoke"
            )
            receipt["legs"][0]["status"] = "FAIL"
            violations = fr.validate_receipt(receipt)
            self.assertTrue(any("status must derive" in item for item in violations))
            self.assertTrue(any("receipt_sha256" in item for item in violations))


if __name__ == "__main__":
    unittest.main()
