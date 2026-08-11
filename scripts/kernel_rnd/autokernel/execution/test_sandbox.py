#!/usr/bin/env python3
"""C6 confinement tests.  These execute only tiny shell probes, never inference."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import microbench
from . import sandbox as S
from . import t0_provider


class SandboxKernelProbeTest(unittest.TestCase):
    def _run(self, policy: S.SandboxPolicy, command: list[str]):
        evaluator = Path(policy.writable_root).parent / "evaluator"
        evaluator.mkdir(exist_ok=True)
        receipt = evaluator / "receipt.json"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.Popen(
            policy.wrap(command, receipt_path=str(receipt)),
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
        stdout, stderr = proc.communicate(timeout=10)
        document = S.read_receipt(receipt)
        teardown = S.cleanup_cgroup(policy, proc.pid)
        return proc.returncode, stdout, stderr, document, teardown

    def test_active_policy_allows_only_its_writable_tree_and_denies_signalling(self):
        with tempfile.TemporaryDirectory(prefix="ak-c6-root-") as outer, \
                tempfile.TemporaryDirectory(prefix="ak-c6-outside-") as outside:
            root = Path(outer, "candidate")
            root.mkdir()
            Path(outer, "evaluator").mkdir()
            policy = S.SandboxPolicy(str(root), token="unittest1")
            code = (
                f"echo inside > {root}/inside; "
                f"echo escape > {outside}/escape; "
                f"echo forged > {outer}/evaluator/receipt.json; "
                "kill -0 1"
            )
            rc, _out, err, receipt, teardown = self._run(
                policy, ["/bin/sh", "-c", code])
            self.assertNotEqual(rc, 0)
            self.assertEqual(Path(root, "inside").read_text().strip(), "inside")
            self.assertFalse(Path(outside, "escape").exists())
            self.assertIn("Permission denied", err)
            self.assertIn("Operation not permitted", err)
            self.assertGreaterEqual(receipt["landlock_abi"], 1)
            self.assertNotEqual(receipt["euid"], 0)
            self.assertIn("kill", receipt["blocked_syscalls"])
            self.assertEqual(receipt["schema"], "epyc.autokernel.sandbox_receipt.v1")
            self.assertTrue(teardown["verified_empty"])
            self.assertTrue(teardown["removed"])

    def test_cgroup_teardown_kills_a_descendant_the_top_process_left_behind(self):
        with tempfile.TemporaryDirectory(prefix="ak-c6-root-") as outer:
            root = Path(outer, "candidate")
            root.mkdir()
            Path(outer, "evaluator").mkdir()
            policy = S.SandboxPolicy(str(root), token="unittest2")
            rc, _out, _err, _receipt, teardown = self._run(
                policy, [sys.executable, "-c",
                         "import os,time\n"
                         "if os.fork(): os._exit(0)\n"
                         "os.close(1); os.close(2); os.setsid(); time.sleep(30)"])
            self.assertEqual(rc, 0)
            self.assertTrue(teardown["descendants_killed"])
            self.assertTrue(teardown["verified_empty"])

    def test_root_identity_is_refused_before_a_command_exists(self):
        with tempfile.TemporaryDirectory(prefix="ak-c6-root-") as root, \
                mock.patch.object(S.os, "geteuid", return_value=0):
            with self.assertRaisesRegex(S.SandboxError, "root"):
                S.SandboxPolicy(root)

    def test_receipt_cannot_be_placed_in_candidate_writable_state(self):
        with tempfile.TemporaryDirectory(prefix="ak-c6-root-") as root:
            policy = S.SandboxPolicy(root, token="receipt1")
            with self.assertRaisesRegex(S.SandboxError, "evaluator-owned"):
                policy.wrap(["/bin/true"], receipt_path=str(Path(root, "receipt.json")))


class LiveRunnerWiringTest(unittest.TestCase):
    def test_microbench_spawner_returns_activation_and_teardown_receipts(self):
        with tempfile.TemporaryDirectory(prefix="ak-c6-root-") as root:
            policy = S.SandboxPolicy(root, token="microbench1")
            spawner = microbench.SubprocessSpawner(
                workdir_root=root, sandbox_policy=policy)
            result = spawner.run(
                ["/bin/sh", "-c", "printf sandboxed"],
                {"PATH": os.environ["PATH"]}, timeout_s=5)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "sandboxed")
            self.assertEqual(result.sandbox_receipt["pid"], result.pid)
            self.assertTrue(result.sandbox_teardown["verified_empty"])
            self.assertFalse(policy.cgroup_path(result.pid).exists())

    def test_t0_runner_returns_activation_and_teardown_receipts(self):
        with tempfile.TemporaryDirectory(prefix="ak-c6-root-") as root:
            policy = S.SandboxPolicy(root, token="t0runner1")
            runner = t0_provider.SubprocessRunner(sandbox_policy=policy)
            result = runner.run(
                ["/bin/sh", "-c", "printf t0"],
                env={"PATH": os.environ["PATH"]}, cwd=root, timeout_s=5)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout, "t0")
            self.assertEqual(result.sandbox_receipt["pid"],
                             int(Path(result.sandbox_receipt["cgroup_path"]).name.split("-")[1]))
            self.assertTrue(result.sandbox_teardown["verified_empty"])

    def test_sandboxed_spawner_refuses_a_workdir_outside_its_write_tree(self):
        with tempfile.TemporaryDirectory(prefix="ak-c6-root-") as root, \
                tempfile.TemporaryDirectory(prefix="ak-c6-other-") as other:
            policy = S.SandboxPolicy(root, token="workdir1")
            with self.assertRaisesRegex(S.SandboxError, "outside"):
                microbench.SubprocessSpawner(
                    workdir_root=other, sandbox_policy=policy)


if __name__ == "__main__":
    unittest.main()
