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
from .test_discovery_deployment_factory import frozen_production_comparator


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
                identity[name], {
                    "logical_path": (
                        "scripts/kernel_rnd/autokernel/controller/"
                        f"{path.name}"),
                    "sha256": F._digest_regular(path, name),
                }
            )

    def test_live_module_provenance_refuses_identical_user_owned_source_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            spec = self._spec(runtime)
            root = R.RuntimeRoot.create_or_open(runtime)
            try:
                S._persist_spec(root, spec)
            finally:
                root.close()
            with self.assertRaisesRegex(
                    S.SupervisorError,
                    "supervisor/factory execution module bytes changed|escaped sealed closure"):
                S.verify_imported_execution_modules(
                    runtime, F._execution_module_runtime_provenance())

    def test_live_module_set_refuses_missing_extra_and_wrong_logical_path(self):
        expected = {
            "runner": {"logical_path": "scripts/runner.py", "sha256": "a" * 64}}
        good = {"runner": {
            **expected["runner"], "path": "/closure/scripts/runner.py"}}
        S._validate_imported_module_set(expected, good)
        cases = (
            {},
            {**good, "extra": {**good["runner"]}},
            {"runner": {**good["runner"],
                        "logical_path": "scripts/other.py"}},
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaisesRegex(
                    S.SupervisorError, "empty|differs from launch authority"):
                S._validate_imported_module_set(expected, case)

    def test_closure_excludes_pyc_and_binds_exact_factory_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(Path(temporary) / "runtime")
            closure = Path(spec.body["execution_closure"]["path"])
            self.assertFalse(any(path.suffix == ".pyc" for path in closure.rglob("*")))
            self.assertFalse(any(path.name == "__pycache__" for path in closure.rglob("*")))
            for directory in [closure, *(p for p in closure.rglob("*") if p.is_dir())]:
                self.assertEqual(directory.stat().st_uid, 0)
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o555)
            for path in (path for path in closure.rglob("*") if path.is_file()):
                self.assertEqual(path.stat().st_uid, 0)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            factory = Path(spec.body["execution_modules"]["deployment_factory"]["path"])
            self.assertEqual(
                S._file_sha256(factory),
                spec.body["execution_modules"]["deployment_factory"]["sha256"],
            )

    def test_same_uid_cannot_replace_factory_before_import_or_execute_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self._spec(Path(temporary) / "runtime")
            closure = Path(spec.body["execution_closure"]["path"])
            factory = (
                closure / "scripts/kernel_rnd/autokernel/controller/discovery_deployment_factory.py"
            )
            marker = Path(temporary) / "injected-top-level-executed"
            replacement = Path(temporary) / "replacement.py"
            replacement.write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('EXECUTED')\n",
                encoding="utf-8",
            )
            original_sha256 = S._file_sha256(factory)
            program = """
import importlib, os, pathlib, sys
factory, replacement = map(pathlib.Path, sys.argv[1:3])
results = []
try:
    os.chmod(factory, 0o644)
    factory.write_text(replacement.read_text())
    results.append('overwrite-succeeded')
except OSError:
    results.append('overwrite-refused')
try:
    os.replace(replacement, factory)
    results.append('replace-succeeded')
except OSError:
    results.append('replace-refused')
importlib.import_module(
    'scripts.kernel_rnd.autokernel.controller.discovery_deployment_factory')
print(','.join(results))
"""
            result = subprocess.run(
                (
                    str(Path(sys.executable).resolve()),
                    "-B",
                    "-c",
                    program,
                    str(factory),
                    str(replacement),
                ),
                cwd="/",
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONSAFEPATH": "1",
                    "PYTHONPATH": str(closure),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "overwrite-refused,replace-refused")
            self.assertFalse(marker.exists())
            self.assertEqual(S._file_sha256(factory), original_sha256)

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
            semantic_sha256 = S._content_hash({"a": 1, "b": 2})
            deployment.write_text(
                '{\n  "b": 2, "a": 1, "config_sha256": "'
                + semantic_sha256 + '"\n}\n',
                encoding="utf-8")
            spec = self._spec(parent / "runtime", deployment=deployment, canary=None)
            root = R.RuntimeRoot.create_or_open(spec.runtime_root)
            try:
                fd = root.open_leaf("deployment-config.json", os.O_RDONLY)
                try:
                    raw = S._validate_config_fd(spec, fd)
                    self.assertEqual(
                        raw,
                        ('{"a":1,"b":2,"config_sha256":"'
                         + semantic_sha256 + '"}\n').encode())
                    self.assertEqual(
                        spec.body["deployment_config"]["semantic_sha256"],
                        semantic_sha256)
                    self.assertNotEqual(
                        spec.body["deployment_config"]["canonical_sha256"],
                        spec.body["deployment_config"]["semantic_sha256"])
                    confused_body = json.loads(json.dumps(spec.body))
                    confused_body["deployment_config"]["semantic_sha256"] = (
                        confused_body["deployment_config"]["canonical_sha256"])
                    with self.assertRaisesRegex(
                            S.SupervisorError, "semantic identity"):
                        S._validate_config_fd(S.LaunchSpec(confused_body), fd)
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
            deployment.write_text(
                json.dumps({"config_sha256": S._content_hash({})}) + "\n",
                encoding="utf-8")
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

    def test_public_status_and_stop_cli_require_both_bytecode_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = (
                str(Path(sys.executable).resolve()),
                "-m",
                S.SUPERVISOR_MODULE,
                "status",
                "--runtime-root",
                str(Path(temporary) / "runtime"),
            )
            environment = dict(os.environ)
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            refusal_cases = (
                (argv, environment),
                ((argv[0], "-B", *argv[1:]), environment),
                (argv, {**environment, "PYTHONDONTWRITEBYTECODE": "1"}),
                (
                    (*argv[:3], "stop", *argv[4:]),
                    environment,
                ),
            )
            for command, command_environment in refusal_cases:
                refused = subprocess.run(
                    command,
                    cwd=S._REPO_ROOT,
                    env=command_environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(refused.returncode, 2, command)
                self.assertIn("requires python -B", refused.stderr)
            accepted = subprocess.run(
                (argv[0], "-B", *argv[1:]),
                cwd=S._REPO_ROOT,
                env={**environment, "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(json.loads(accepted.stdout)["status"], "absent")


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

    def test_direct_validate_graph_reopens_byte_identically_from_root_closure(self):
        bundle = Path(self.temporary.name) / "deployment"
        deployment = F.initialize_static_deployment_bundle(
            bundle, frozen_production_comparator=
            frozen_production_comparator(
                Path(self.temporary.name) / "authority"))
        command = (
            str(Path(sys.executable).resolve()), "-B", "-m",
            F.__name__, "--deployment", str(deployment), "--validate-only")
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        direct_payloads = []
        for _ in range(2):
            result = subprocess.run(
                command, cwd=S._REPO_ROOT, env=environment,
                text=True, capture_output=True, check=False, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            direct_payloads.append(json.loads(result.stdout))
        self.assertEqual(direct_payloads[0], direct_payloads[1])
        graph_path = Path(direct_payloads[0]["graph_receipt"])
        direct_bytes = graph_path.read_bytes()
        direct_sha = direct_payloads[0]["graph_sha256"]

        spec = S._new_spec(
            runtime_root=self.root, deployment=deployment,
            validate_only=True, canary=None, max_restarts=0,
            restart_delay=0.0, term_grace=0.2, kill_grace=1.0)
        launched = S.start_detached(spec, start_timeout=20.0)
        self.assertEqual(launched["launch_result"], "started")
        terminal = self._wait_identity(lambda row: row["state"] == "stopped", timeout=60)
        self.assertEqual(terminal["exit_code"], 0)
        self.assertEqual(graph_path.read_bytes(), direct_bytes)
        supervised = json.loads((self.root / "controller.stdout.log").read_text())
        self.assertEqual(supervised["graph_sha256"], direct_sha)
        self.assertEqual(supervised["graph_receipt"], str(graph_path.resolve()))
        self.assertEqual(list((bundle / "operations" / "claims").iterdir()), [])

        legacy = json.loads(direct_bytes)
        legacy["schema"] = "epyc.autokernel.static_discovery_graph.v4"
        graph_path.write_text(json.dumps(legacy, sort_keys=True, indent=2) + "\n")
        refused = subprocess.run(
            command, cwd=S._REPO_ROOT, env=environment,
            text=True, capture_output=True, check=False, timeout=60)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn(
            "legacy deployment graph cannot authorize successor execution",
            refused.stderr)

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
        children = [row for row in self._ledger()
                    if row["event"] == "child_started"]
        cgroups = [row["payload"]["cgroup"] for row in children]
        self.assertEqual(len({row["path"] for row in cgroups}), 3)
        self.assertTrue(all(set(row) == {
            "path", "dev", "ino", "uid", "nlink", "mode"} for row in cgroups))
        self.assertTrue(all(not Path(row["path"]).exists() for row in cgroups))
        exited = [row for row in self._ledger() if row["event"] == "child_exited"]
        self.assertTrue(all("cgroup.remove" in row["payload"]["cleanup_actions"]
                            for row in exited))

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
        child_rows = [row for row in self._ledger()
                      if row["event"] == "child_started"]
        self.assertTrue(child_rows)
        self.assertTrue(all(
            not Path(row["payload"]["cgroup"]["path"]).exists()
            for row in child_rows))


if __name__ == "__main__":
    unittest.main()
