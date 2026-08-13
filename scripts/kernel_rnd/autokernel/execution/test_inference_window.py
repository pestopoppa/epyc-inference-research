import fcntl
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from dataclasses import dataclass

from . import inference_window as window
from . import cpu_region_claim


def _hold_until_exit(path: str, ready) -> None:
    lease = window.InferenceCallWindow(path).acquire()
    ready.send(True)
    ready.close()
    # Exit without an explicit release: the kernel must recover the flock.
    os._exit(0)


class InferenceCallWindowTests(unittest.TestCase):

    def test_two_open_descriptions_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.lock"
            with window.InferenceCallWindow(path).hold():
                with self.assertRaises(window.InferenceWindowTimeout):
                    window.InferenceCallWindow(path, timeout_s=0).acquire()

    def test_dead_holder_releases_without_stale_reclamation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.lock"
            receiver, sender = multiprocessing.Pipe(duplex=False)
            process = multiprocessing.Process(
                target=_hold_until_exit, args=(str(path), sender))
            process.start()
            self.assertTrue(receiver.recv())
            process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)
            lease = window.InferenceCallWindow(path, timeout_s=1).acquire()
            self.assertTrue(lease.held)
            lease.release()

    def test_release_is_idempotent_and_lock_file_is_never_unlinked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.lock"
            lease = window.InferenceCallWindow(path).acquire()
            lease.release()
            lease.release()
            self.assertTrue(path.is_file())
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def test_windowed_spawner_holds_only_during_delegate_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.lock"
            observations = []

            class Delegate:
                spawner_id = "subprocess/v1"

                def run(self, argv, env, *, timeout_s, cwd=None):
                    contender = window.InferenceCallWindow(path, timeout_s=0)
                    try:
                        contender.acquire()
                    except window.InferenceWindowTimeout:
                        observations.append("held")
                    return (argv, env, timeout_s, cwd)

            wrapped = window.WindowedSpawner(
                Delegate(), window.InferenceCallWindow(path))
            self.assertEqual(wrapped.spawner_id, "subprocess/v1")
            self.assertEqual(wrapped.run(["tool"], {"A": "1"}, timeout_s=3),
                             (["tool"], {"A": "1"}, 3, None))
            self.assertEqual(observations, ["held"])
            with window.InferenceCallWindow(path, timeout_s=0).hold():
                pass

    def test_windowed_spawner_attaches_a_released_per_call_receipt(self):
        @dataclass(frozen=True)
        class Result:
            value: str
            inference_window_receipt: dict | None = None

        class Delegate:
            spawner_id = "subprocess/v1"

            def run(self, argv, env, *, timeout_s, cwd=None):
                return Result("measured")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.lock"
            result = window.WindowedSpawner(
                Delegate(), window.InferenceCallWindow(path)).run(
                    ["tool"], {}, timeout_s=3)
        receipt = result.inference_window_receipt
        self.assertEqual(receipt["schema"],
                         "epyc.autokernel.inference_call_window.v1")
        self.assertEqual(receipt["lock_path"], str(path))
        self.assertEqual(receipt["scope"], "model_load_and_inference_only")
        self.assertTrue(receipt["released"])
        self.assertGreaterEqual(receipt["waited_s"], 0.0)
        self.assertGreaterEqual(receipt["held_s"], 0.0)

    def test_gpu_helpers_can_borrow_only_windowed_live_control_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "regions"
            journal = cpu_region_claim.RegionClaimJournal(Path(tmp) / "claims.jsonl")
            claim = cpu_region_claim.acquire_cpu_region_claim(
                "0-95", purpose="AutoKernel five-control calibration block test",
                campaign_id="ak-controls-test", journal=journal,
                role=window.WINDOWED_CPU_ROLE, timeout_s=0, max_hold_s=60,
                lock_root=root)
            try:
                borrowed = window.borrow_windowed_cpu_coverage(
                    "184-191", lock_root=root)
                self.assertTrue(borrowed.borrowed)
                self.assertEqual(borrowed.claim_id, claim.claim_id)
                borrowed.validate()
            finally:
                claim.release()
            with self.assertRaisesRegex(RuntimeError, "not held"):
                borrowed.validate()

    def test_gpu_helpers_can_borrow_windowed_strict_campaign_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "regions"
            journal = cpu_region_claim.RegionClaimJournal(Path(tmp) / "claims.jsonl")
            claim = cpu_region_claim.acquire_cpu_region_claim(
                "0-95", purpose="AutoKernel campaign ak-decode / akc-decode",
                campaign_id="ak-iqk-v9-decode-test", journal=journal,
                role=window.WINDOWED_CPU_ROLE, timeout_s=0, max_hold_s=60,
                lock_root=root)
            try:
                borrowed = window.borrow_windowed_cpu_coverage(
                    "184-191", lock_root=root)
                self.assertEqual(borrowed.campaign_id, "ak-iqk-v9-decode-test")
                borrowed.validate()
            finally:
                claim.release()

    def test_legacy_autokernel_claim_is_never_borrowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "regions"
            journal = cpu_region_claim.RegionClaimJournal(Path(tmp) / "claims.jsonl")
            claim = cpu_region_claim.acquire_cpu_region_claim(
                "0-95", purpose="AutoKernel five-control calibration block test",
                campaign_id="ak-controls-test", journal=journal,
                role="autokernel", timeout_s=0, max_hold_s=60, lock_root=root)
            try:
                with self.assertRaisesRegex(RuntimeError, "not held|unreadable"):
                    window.borrow_windowed_cpu_coverage("184-191", lock_root=root)
            finally:
                claim.release()

    def test_cpu_claim_release_waits_for_call_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.lock"
            observations = []

            class Claim:
                def release(self):
                    try:
                        window.InferenceCallWindow(path, timeout_s=0).acquire()
                    except window.InferenceWindowTimeout:
                        observations.append("release-under-window")

            with window.ReleaseUnderWindow(
                    Claim(), window.InferenceCallWindow(path)):
                pass
            self.assertEqual(observations, ["release-under-window"])


if __name__ == "__main__":
    unittest.main()
