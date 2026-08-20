"""Hardware-free acceptance tests for the durable discovery supervisor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from . import discovery_deployment_factory as F
from . import discovery_supervisor as S


class SupervisorContractTests(unittest.TestCase):
    def test_factory_attests_supervisor_as_an_execution_module(self):
        identity = F._execution_module_identity()
        self.assertEqual(identity["discovery_supervisor"], {
            "path": str(Path(S.__file__).resolve(strict=True)),
            "sha256": F._digest_regular(
                Path(S.__file__).resolve(strict=True), "discovery_supervisor"),
        })

    def test_validate_only_argv_has_no_alternate_execution_seam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployment = root / "deployment.json"
            deployment.write_text("{}\n", encoding="utf-8")
            runtime = root / "runtime"
            spec = S._new_spec(
                runtime_root=runtime, deployment=deployment,
                validate_only=True, canary=None, max_restarts=0,
                restart_delay=1.0, term_grace=2.0, kill_grace=1.0,
            )
            self.assertEqual(spec.child_argv(), (
                str(Path(sys.executable).resolve()), "-m", S.FACTORY_MODULE,
                "--deployment", str(deployment.resolve()), "--validate-only",
            ))
            self.assertEqual(set(spec.body["execution_modules"]),
                             {"supervisor", "deployment_factory"})
            self.assertNotIn("command", spec.body)
            self.assertNotIn("environment", spec.body)

    def test_execution_module_drift_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = S._new_spec(
                runtime_root=root / "runtime", deployment=None,
                validate_only=True,
                canary={"hold_seconds": 1.0, "exit_code": 0,
                        "spawn_descendant": False},
                max_restarts=0, restart_delay=0.0,
                term_grace=1.0, kill_grace=1.0,
            )
            original = S._file_sha256

            def changed(path: Path) -> str:
                if Path(path).resolve() == Path(S.__file__).resolve():
                    return "0" * 64
                return original(path)

            with mock.patch.object(S, "_file_sha256", side_effect=changed), \
                    self.assertRaisesRegex(
                        S.SupervisorError, "execution module bytes changed"):
                spec.verify_execution_modules()

    def test_deployment_restart_is_zero_until_typed_reconciliation_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deployment = root / "deployment.json"
            deployment.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                    S.SupervisorError, "offline reconciliation receipt"):
                S._new_spec(
                    runtime_root=root / "runtime", deployment=deployment,
                    validate_only=False, canary=None, max_restarts=1,
                    restart_delay=0.0, term_grace=1.0, kill_grace=1.0,
                )

    def test_status_and_stop_refuse_unbound_identity_before_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runtime"
            spec = S._new_spec(
                runtime_root=root, deployment=None, validate_only=True,
                canary={"hold_seconds": 1.0, "exit_code": 0,
                        "spawn_descendant": False},
                max_restarts=0, restart_delay=0.0,
                term_grace=1.0, kill_grace=1.0,
            )
            S._persist_spec(root / "launch-spec.json", spec)
            identity = S._write_identity(
                root / "identity.json", spec, state="starting",
                supervisor=S._process_identity(os.getpid()), child=None,
                restarts=0,
            )
            identity["spec_sha256"] = "0" * 64
            S._atomic_json(root / "identity.json", identity)
            with mock.patch.object(
                    S.signal, "pidfd_send_signal",
                    side_effect=AssertionError("signal attempted")), \
                    self.assertRaisesRegex(
                        S.SupervisorError, "not bound to this launch spec"):
                S.stop_supervisor(root)

    def test_stop_refuses_nonprivate_ledger_before_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runtime"
            spec = S._new_spec(
                runtime_root=root, deployment=None, validate_only=True,
                canary={"hold_seconds": 1.0, "exit_code": 0,
                        "spawn_descendant": False},
                max_restarts=0, restart_delay=0.0,
                term_grace=1.0, kill_grace=1.0,
            )
            S._persist_spec(root / "launch-spec.json", spec)
            supervisor = S._process_identity(os.getpid())
            ledger = S.DeathLedger(root / "death-ledger.jsonl")
            ledger.append("supervisor_started", {
                "spec_sha256": spec.sha256,
                "session_name": spec.session_name,
                "supervisor": supervisor,
            })
            S._write_identity(
                root / "identity.json", spec, state="starting",
                supervisor=supervisor, child=None, restarts=0,
            )
            alias = root / "death-ledger.alias"
            os.link(root / "death-ledger.jsonl", alias)
            with mock.patch.object(
                    S.signal, "pidfd_send_signal",
                    side_effect=AssertionError("signal attempted")), \
                    self.assertRaisesRegex(S.SupervisorError, "single-link"):
                S.stop_supervisor(root)
            alias.unlink()

    def test_private_state_rejects_mode_hardlink_and_noncanonical_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            S._atomic_json(path, {"ok": True})
            os.chmod(path, 0o640)
            with self.assertRaisesRegex(S.SupervisorError, "mode-0600"):
                S._read_json(path)
            os.chmod(path, 0o600)
            alias = path.with_name("alias.json")
            os.link(path, alias)
            with self.assertRaisesRegex(S.SupervisorError, "single-link"):
                S._read_json(path)
            alias.unlink()
            path.write_text('{"ok": true}\n', encoding="utf-8")
            os.chmod(path, 0o600)
            with self.assertRaisesRegex(S.SupervisorError, "canonically encoded"):
                S._read_json(path)

    def test_identity_requires_boot_startticks_and_host_namespace(self):
        identity = S._process_identity(os.getpid())
        self.assertEqual(S._identity_liveness(identity)[0], "live")
        for field in ("start_ticks", "boot_id", "host",
                      "host_id_source", "host_id_sha256"):
            broken = dict(identity)
            broken.pop(field)
            self.assertEqual(S._identity_liveness(broken)[0], "unknown", field)
        recycled = dict(identity)
        recycled["start_ticks"] += 1
        self.assertEqual(S._identity_liveness(recycled)[0], "dead")


@unittest.skipUnless(shutil.which("tmux"), "tmux is required")
class DetachedCanaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "runtime"

    def tearDown(self):
        if self.root.exists():
            try:
                status = S._status_payload(self.root)
                if status.get("status") == "live":
                    S.stop_supervisor(self.root, timeout=8.0)
            except (S.SupervisorError, FileNotFoundError):
                pass
        self.temporary.cleanup()

    def _launcher(self, *extra: str) -> tuple[int, subprocess.CompletedProcess[str]]:
        process = subprocess.Popen(
            (str(Path(sys.executable).resolve()), "-m", S.SUPERVISOR_MODULE,
             "canary", "--runtime-root", str(self.root), *extra),
            cwd=S._REPO_ROOT, stdin=subprocess.DEVNULL,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        launcher_pid = process.pid
        stdout, stderr = process.communicate(timeout=15.0)
        return launcher_pid, subprocess.CompletedProcess(
            process.args, process.returncode, stdout, stderr)

    def _wait_identity(self, predicate, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            if (self.root / "identity.json").exists():
                last = S._read_json(self.root / "identity.json")
                if predicate(last):
                    return last
            time.sleep(0.05)
        self.fail(f"identity did not reach expected state; last={last!r}")

    def _ledger(self):
        path = self.root / "death-ledger.jsonl"
        return [json.loads(line) for line in path.read_text(
            encoding="utf-8").splitlines() if line]

    def test_canary_survives_launcher_completion_and_is_singleton(self):
        launcher_pid, launcher = self._launcher(
            "--hold-seconds", "3", "--max-restarts", "0",
            "--restart-delay", "0", "--term-grace", "0.5",
            "--kill-grace", "0.5",
        )
        self.assertEqual(launcher.returncode, 0, launcher.stderr)
        receipt = json.loads(launcher.stdout)
        self.assertEqual(receipt["launch_result"], "started")
        identity = self._wait_identity(lambda row: row.get("child") is not None)
        supervisor_pid = identity["supervisor"]["pid"]
        self.assertEqual(S._identity_liveness(identity["supervisor"])[0], "live")
        self.assertIsNone(S._read_start_ticks(launcher_pid))
        self.assertNotEqual(supervisor_pid, launcher_pid)
        self.assertTrue(S._tmux_has_session(identity["session_name"]))

        _duplicate_pid, duplicate = self._launcher(
            "--hold-seconds", "3", "--max-restarts", "0",
            "--restart-delay", "0", "--term-grace", "0.5",
            "--kill-grace", "0.5",
        )
        self.assertEqual(duplicate.returncode, 0, duplicate.stderr)
        duplicate_receipt = json.loads(duplicate.stdout)
        self.assertEqual(duplicate_receipt["launch_result"], "already_running")
        self.assertEqual(
            duplicate_receipt["identity"]["supervisor"]["pid"], supervisor_pid)

        terminal = self._wait_identity(lambda row: row.get("state") == "stopped")
        self.assertEqual(terminal["exit_code"], 0)
        self.assertEqual(S._identity_liveness(identity["supervisor"])[0], "dead")
        self.assertFalse(S._tmux_has_session(identity["session_name"]))
        self.assertEqual([row["event"] for row in self._ledger()], [
            "supervisor_started", "child_started", "child_exited",
            "supervisor_stopped",
        ])
        for path in self.root.iterdir():
            self.assertFalse(path.is_symlink())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path)

    def test_bounded_restart_policy_exhausts_exactly(self):
        _launcher_pid, launcher = self._launcher(
            "--hold-seconds", "0.2", "--exit-code", "7",
            "--max-restarts", "2", "--restart-delay", "0",
            "--term-grace", "0.2", "--kill-grace", "0.2",
        )
        self.assertEqual(launcher.returncode, 0, launcher.stderr)
        terminal = self._wait_identity(lambda row: row.get("state") == "stopped")
        self.assertEqual(terminal["restart_count"], 2)
        self.assertEqual(terminal["exit_code"], 7)
        events = [row["event"] for row in self._ledger()]
        self.assertEqual(events.count("child_started"), 3)
        self.assertEqual(events.count("child_exited"), 3)
        self.assertEqual(events.count("restart_scheduled"), 2)
        self.assertEqual(events.count("restarts_exhausted"), 1)

    def test_stop_forwards_signal_and_cleans_owned_descendant_group(self):
        _launcher_pid, launcher = self._launcher(
            "--hold-seconds", "30", "--spawn-descendant",
            "--max-restarts", "0", "--restart-delay", "0",
            "--term-grace", "0.5", "--kill-grace", "1",
        )
        self.assertEqual(launcher.returncode, 0, launcher.stderr)
        identity = self._wait_identity(lambda row: row.get("child") is not None)
        stdout = self.root / "controller.stdout.log"
        deadline = time.monotonic() + 5.0
        canary = None
        while time.monotonic() < deadline:
            lines = stdout.read_text(encoding="utf-8").splitlines()
            if lines:
                canary = json.loads(lines[0])
                break
            time.sleep(0.05)
        self.assertIsNotNone(canary)
        child_pid = identity["child"]["pid"]
        descendant_pid = canary["descendant_pid"]
        stopped = S.stop_supervisor(self.root, timeout=8.0)
        self.assertEqual(stopped["stop_result"], "stopped")
        self.assertIsNone(S._read_start_ticks(child_pid))
        self.assertIsNone(S._read_start_ticks(descendant_pid))
        events = self._ledger()
        forwarded = [row for row in events if row["event"] == "signal_forwarded"]
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(forwarded[0]["payload"]["signal"], 15)
        self.assertEqual(events[-1]["event"], "supervisor_stopped")


if __name__ == "__main__":
    unittest.main()
