"""Hardware-free acceptance tests for the immutable discovery supervisor."""

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
from . import discovery_supervisor_secure as R


CANARY = {"hold_seconds": 1.0, "exit_code": 0, "spawn_descendant": False}


class ImmutableAuthorityTests(unittest.TestCase):
    def _spec(self, root: Path, **overrides):
        values = dict(
            runtime_root=root,
            deployment=None,
            validate_only=True,
            canary=CANARY,
            max_restarts=0,
            restart_delay=0.0,
            term_grace=0.2,
            kill_grace=0.5,
        )
        values.update(overrides)
        return S._new_spec(**values)

    def test_factory_attests_supervisor_and_secure_runtime_modules(self):
        identity = F._execution_module_identity()
        for name, module in (("discovery_supervisor", S), ("discovery_supervisor_secure", R)):
            path = Path(module.__file__).resolve(strict=True)
            self.assertEqual(
                identity[name], {"path": str(path), "sha256": F._digest_regular(path, name)}
            )

    def test_closure_excludes_pyc_and_binds_exact_factory_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(Path(temporary) / "runtime")
            closure = Path(spec.body["execution_closure"]["path"])
            self.assertFalse(any(path.suffix == ".pyc" for path in closure.rglob("*")))
            self.assertFalse(any(path.name == "__pycache__" for path in closure.rglob("*")))
            for directory in [closure, *(p for p in closure.rglob("*") if p.is_dir())]:
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o500)
            factory = Path(spec.body["execution_modules"]["deployment_factory"]["path"])
            self.assertEqual(
                S._file_sha256(factory),
                spec.body["execution_modules"]["deployment_factory"]["sha256"],
            )

    def test_pyc_injection_is_refused_before_supervision(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(Path(temporary) / "runtime")
            closure = Path(spec.body["execution_closure"]["path"])
            os.chmod(closure, 0o700)
            injected = closure / "injected.pyc"
            injected.write_bytes(b"bytecode")
            with self.assertRaisesRegex(S.SupervisorError, "bytecode"):
                S._verify_execution_closure(spec)

    def test_stable_open_refuses_path_swap_and_never_follows_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"trusted")
            alias = root / "alias"
            alias.symlink_to(target)
            with self.assertRaises(OSError):
                R.open_stable(alias)
            original_open = os.open
            replacement = root / "replacement"
            replacement.write_bytes(b"changed")

            def swapped(path, flags, *args, **kwargs):
                if Path(path) == target:
                    target.rename(root / "old")
                    replacement.rename(target)
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(R.os, "open", side_effect=swapped):
                fd, raw, _identity = R.open_stable(target)
                os.close(fd)
            self.assertEqual(raw, b"changed")
            # The selected object, not the pre-open pathname observation, wins.

    def test_runtime_root_symlink_swap_is_refused_by_pinned_dirfd(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            path = parent / "runtime"
            runtime = R.RuntimeRoot.create_or_open(path)
            moved = parent / "moved"
            path.rename(moved)
            path.symlink_to(moved)
            try:
                with self.assertRaisesRegex(
                    R.SecureRuntimeError, "runtime root object identity changed"
                ):
                    runtime.verify()
            finally:
                runtime.close()

    def test_config_canonical_bytes_and_exact_object_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            deployment = parent / "deployment.json"
            deployment.write_text('{\n  "b": 2, "a": 1\n}\n', encoding="utf-8")
            spec = self._spec(parent / "runtime", deployment=deployment, canary=None)
            root = R.RuntimeRoot.create_or_open(spec.runtime_root)
            try:
                fd = root.open_leaf("deployment-config.json", os.O_RDONLY)
                try:
                    raw = S._validate_config_fd(spec, fd)
                    self.assertEqual(raw, b'{"a":1,"b":2}\n')
                    os.fchmod(fd, 0o400)
                    with self.assertRaisesRegex(S.SupervisorError, "differs from launch spec"):
                        S._validate_config_fd(spec, fd)
                finally:
                    os.close(fd)
            finally:
                root.close()

    def test_deployment_restart_requires_typed_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            deployment = parent / "deployment.json"
            deployment.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(S.SupervisorError, "typed reconciliation"):
                self._spec(
                    parent / "runtime",
                    deployment=deployment,
                    canary=None,
                    validate_only=False,
                    max_restarts=1,
                )

    def test_ledger_refuses_torn_hash_and_out_of_order_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = R.RuntimeRoot.create_or_open(Path(temporary) / "runtime")
            try:
                ledger = S.DeathLedger(root)
                with self.assertRaisesRegex(S.SupervisorError, "invalid child_started transition"):
                    ledger.append(
                        "child_started",
                        {
                            "restart_count": 0,
                            "child": {},
                            "stdout": "x",
                            "stderr": "y",
                            "cgroup": "z",
                        },
                    )
                fd = root.open_append("death-ledger.jsonl")
                os.write(fd, b'{"torn":')
                os.fsync(fd)
                os.close(fd)
                with self.assertRaisesRegex(S.SupervisorError, "torn record"):
                    S.DeathLedger(root)
            finally:
                root.close()

    def test_ledger_reloads_same_locked_fd_before_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = R.RuntimeRoot.create_or_open(Path(temporary) / "runtime")
            try:
                ledger = S.DeathLedger(root)
                supervisor = S._process_identity(os.getpid())
                tmux = {
                    "session_id": "$1",
                    "pane_id": "%1",
                    "pane_pid": os.getpid(),
                    "pane_start_ticks": supervisor["start_ticks"],
                }
                ledger.append(
                    "supervisor_started",
                    {
                        "spec_sha256": "0" * 64,
                        "session_name": "ak-test",
                        "supervisor": supervisor,
                        "tmux": tmux,
                    },
                )
                stale = S.DeathLedger(root)
                ledger.append(
                    "supervisor_stopped",
                    {
                        "exit_code": 0,
                        "restart_count": 0,
                        "stop_signal": None,
                        "supervisor": supervisor,
                    },
                )
                appended = stale.append(
                    "supervisor_started",
                    {
                        "spec_sha256": "1" * 64,
                        "session_name": "ak-test",
                        "supervisor": supervisor,
                        "tmux": tmux,
                    },
                )
                self.assertEqual(appended["sequence"], 3)
                self.assertEqual(appended["previous_sha256"], ledger.records[-1]["record_sha256"])
            finally:
                root.close()

    def test_tmux_swap_is_refused_before_signal(self):
        supervisor = S._process_identity(os.getpid())
        expected = {
            "session_id": "$1",
            "pane_id": "%1",
            "pane_pid": os.getpid(),
            "pane_start_ticks": supervisor["start_ticks"],
        }
        swapped = {**expected, "session_id": "$2"}
        with (
            mock.patch.object(S, "_tmux_binding", return_value=swapped),
            self.assertRaisesRegex(S.SupervisorError, "not bound"),
        ):
            S._validate_tmux_binding(expected, supervisor, "ak-test")

    def test_source_has_no_process_group_signal_fallback(self):
        source = Path(S.__file__).read_text(encoding="utf-8")
        self.assertNotIn("killpg", source)
        self.assertNotIn("start_new_session", source)


@unittest.skipUnless(
    shutil.which("tmux") and Path("/sys/fs/cgroup/cgroup.kill").exists(),
    "tmux and cgroup v2 are required",
)
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
            except (S.SupervisorError, R.SecureRuntimeError, FileNotFoundError):
                pass
        self.temporary.cleanup()

    def _launcher(self, *extra: str) -> tuple[int, subprocess.CompletedProcess[str]]:
        process = subprocess.Popen(
            (
                str(Path(sys.executable).resolve()),
                "-B",
                "-m",
                S.SUPERVISOR_MODULE,
                "canary",
                "--runtime-root",
                str(self.root),
                *extra,
            ),
            cwd=S._REPO_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdin=subprocess.DEVNULL,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        launcher_pid = process.pid
        stdout, stderr = process.communicate(timeout=30.0)
        return launcher_pid, subprocess.CompletedProcess(
            process.args, process.returncode, stdout, stderr
        )

    def _wait_identity(self, predicate, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            path = self.root / "identity.json"
            if path.exists():
                last = json.loads(path.read_text())
                if predicate(last):
                    return last
            time.sleep(0.05)
        self.fail(f"identity did not reach expected state; last={last!r}")

    def _ledger(self):
        return [
            json.loads(line) for line in (self.root / "death-ledger.jsonl").read_text().splitlines()
        ]

    def test_canary_survives_launcher_and_binds_tmux_pane(self):
        launcher_pid, launcher = self._launcher(
            "--hold-seconds",
            "2",
            "--max-restarts",
            "0",
            "--restart-delay",
            "0",
            "--term-grace",
            "0.2",
            "--kill-grace",
            "0.5",
        )
        self.assertEqual(launcher.returncode, 0, launcher.stderr)
        receipt = json.loads(launcher.stdout)
        self.assertEqual(receipt["launch_result"], "started")
        identity = receipt["identity"]
        self.assertIsNone(S._read_start_ticks(launcher_pid))
        self.assertEqual(identity["tmux"]["pane_pid"], identity["supervisor"]["pid"])
        terminal = self._wait_identity(lambda row: row["state"] == "stopped")
        self.assertEqual(terminal["exit_code"], 0)

    def test_bounded_canary_restart_is_exact(self):
        _pid, launcher = self._launcher(
            "--hold-seconds",
            "0.2",
            "--exit-code",
            "7",
            "--max-restarts",
            "2",
            "--restart-delay",
            "0",
            "--term-grace",
            "0.1",
            "--kill-grace",
            "0.5",
        )
        self.assertEqual(launcher.returncode, 0, launcher.stderr)
        terminal = self._wait_identity(lambda row: row["state"] == "stopped")
        self.assertEqual((terminal["restart_count"], terminal["exit_code"]), (2, 7))
        events = [row["event"] for row in self._ledger()]
        self.assertEqual(events.count("child_started"), 3)
        self.assertEqual(events.count("restart_scheduled"), 2)

    def test_forced_stop_cleans_leader_and_descendant_without_killpg(self):
        _pid, launcher = self._launcher(
            "--hold-seconds",
            "30",
            "--spawn-descendant",
            "--max-restarts",
            "0",
            "--restart-delay",
            "0",
            "--term-grace",
            "0.2",
            "--kill-grace",
            "1",
        )
        self.assertEqual(launcher.returncode, 0, launcher.stderr)
        identity = self._wait_identity(lambda row: row.get("child") is not None)
        stdout = self.root / "controller.stdout.log"
        canary = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if stdout.exists() and stdout.read_text().strip():
                canary = json.loads(stdout.read_text().splitlines()[0])
                break
            time.sleep(0.05)
        self.assertIsNotNone(canary)
        stopped = S.stop_supervisor(self.root, timeout=8.0)
        self.assertEqual(stopped["stop_result"], "stopped")
        self.assertIsNone(S._read_start_ticks(identity["child"]["pid"]))
        self.assertIsNone(S._read_start_ticks(canary["descendant_pid"]))
        self.assertFalse(
            Path(
                json.loads((self.root / "launch-spec.json").read_text())["cgroup"]["base"],
                json.loads((self.root / "launch-spec.json").read_text())["cgroup"]["name"],
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
