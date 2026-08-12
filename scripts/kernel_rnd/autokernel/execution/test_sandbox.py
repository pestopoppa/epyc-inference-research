#!/usr/bin/env python3
"""C6 confinement tests.  These execute only tiny shell probes, never inference."""
from __future__ import annotations

import errno
import os
import json
import socket
import subprocess
import sys
import tempfile
import threading
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
        env = dict(
            os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        proc = subprocess.Popen(
            policy.wrap(command, receipt_path=str(receipt)),
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
        stdout, stderr = proc.communicate(timeout=10)
        document = S.read_receipt(receipt)
        teardown = S.cleanup_cgroup(policy, proc.pid)
        return proc.returncode, stdout, stderr, document, teardown

    @staticmethod
    def _controller_read_surface(allowed_file: Path) -> tuple[tuple[str, ...],
                                                               tuple[str, ...]]:
        roots = [str(Path(sys.base_prefix).resolve())]
        system_lib = Path("/usr/lib").resolve()
        if str(system_lib) not in roots:
            roots.append(str(system_lib))
        files = [str(allowed_file)]
        loader_cache = Path("/etc/ld.so.cache")
        if loader_cache.is_file():
            files.append(str(loader_cache))
        return tuple(roots), tuple(files)

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
            self.assertEqual(receipt["schema"], S.RECEIPT_SCHEMA)
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

    def test_host_delegation_is_the_default_when_present(self):
        with mock.patch.dict(S.os.environ, {}, clear=True), \
                mock.patch.object(S.os.path, "isdir", return_value=True):
            self.assertEqual(S.default_cgroup_root(), S.HOST_CGROUP_ROOT)

    def test_explicit_cgroup_root_must_be_absolute(self):
        with mock.patch.dict(
                S.os.environ, {S.CGROUP_ROOT_ENV: "relative/cgroup"}, clear=True):
            with self.assertRaisesRegex(S.SandboxError, "absolute"):
                S.default_cgroup_root()

    def test_explicit_cgroup_root_overrides_host_default(self):
        with mock.patch.dict(
                S.os.environ, {S.CGROUP_ROOT_ENV: "/delegated/cgroup"}, clear=True):
            self.assertEqual(S.default_cgroup_root(), "/delegated/cgroup")

    def test_evaluator_profile_is_read_restricted_gpu_exact_and_network_denied(self):
        with tempfile.TemporaryDirectory(prefix="ak-eval-profile-") as outer:
            root = Path(outer, "candidate")
            evidence = Path(outer, "evidence")
            root.mkdir()
            evidence.mkdir()
            policy = S.SandboxPolicy(
                str(root), token="evalprofile1", profile=S.EVALUATOR_PROFILE,
                readable_roots=("/usr/bin", "/usr/lib"),
                readable_files=("/etc/ld.so.cache", "/dev/urandom"),
                writable_device_paths=("/dev/kfd", "/dev/dri/renderD128", "/dev/null"))
            probe = (
                "import errno,os,socket; "
                "fds=[os.open(p,os.O_RDWR) for p in "
                "('/dev/kfd','/dev/dri/renderD128','/dev/null')]; "
                "[os.close(fd) for fd in fds]; "
                "random_fd=os.open('/dev/urandom',os.O_RDONLY); "
                "assert len(os.read(random_fd,8)) == 8; os.close(random_fd); "
                "denied=[]; "
                "\ntry: open('/etc/passwd').close()\n"
                "except PermissionError: denied.append('read')\n"
                "try: socket.socket()\n"
                "except PermissionError: denied.append('socket')\n"
                "assert denied == ['read','socket']; print('isolated',end='')")
            command = ["/usr/bin/python3", "-c", probe]
            receipt_path = evidence / "activation.json"
            process = subprocess.Popen(
                policy.wrap(command, receipt_path=str(receipt_path)),
                cwd=root, env={"PATH": "/usr/bin:/bin", "HOME": str(root)},
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, close_fds=True,
                start_new_session=True)
            stdout, stderr = process.communicate(timeout=10)
            receipt = S.read_receipt(receipt_path)
            S.verify_receipt(receipt, policy=policy, pid=process.pid, argv=command)
            teardown = S.cleanup_cgroup(policy, process.pid)
            self.assertEqual((process.returncode, stdout, stderr), (0, "isolated", ""))
            self.assertEqual(receipt["profile"], S.EVALUATOR_PROFILE)
            self.assertEqual(receipt["network_profile"], S.NETWORK_DENY_ALL)
            self.assertTrue(receipt["read_allowlist_enforced"])
            self.assertEqual(set(receipt["writable_device_paths"]),
                    {"/dev/kfd", "/dev/dri/renderD128", "/dev/null"})
            self.assertIn("/dev/urandom", receipt["readable_files"])
            self.assertTrue(teardown["verified_empty"])

    def test_evaluator_profile_rejects_partial_devices_and_broker_identity(self):
        with tempfile.TemporaryDirectory(prefix="ak-eval-profile-") as root:
            with self.assertRaisesRegex(S.SandboxError, "exact MI210 pair and /dev/null"):
                S.SandboxPolicy(
                    root, profile=S.EVALUATOR_PROFILE,
                    readable_roots=("/usr/bin",),
                    writable_device_paths=("/dev/kfd",))
            with self.assertRaisesRegex(S.SandboxError, "cannot name a broker peer"):
                S.SandboxPolicy(
                    root, profile=S.EVALUATOR_PROFILE,
                    readable_roots=("/usr/bin",), broker_peer_pid=123,
                    broker_peer_start_ticks=456,
                    writable_device_paths=("/dev/kfd", "/dev/dri/renderD128", "/dev/null"))

    def test_controller_profile_enforces_read_devices_signals_and_client_network(self):
        with tempfile.TemporaryDirectory(prefix="ak-controller-profile-") as outer:
            root = Path(outer)
            controller = root / "controller"
            controller.mkdir()
            evaluator = root / "evaluator"
            evaluator.mkdir()
            campaign = root / "campaign"
            campaign.mkdir()
            secret = campaign / "evidence.json"
            secret.write_text("campaign-secret", encoding="utf-8")
            allowed = root / "allowed.txt"
            allowed.write_text("allowed", encoding="utf-8")
            broker_path = root / "broker.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(broker_path))
            server.listen(1)
            accepted = threading.Event()
            tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_server.bind(("127.0.0.1", 0))
            tcp_server.listen(1)
            tcp_port = tcp_server.getsockname()[1]
            tcp_accepted = threading.Event()

            def serve():
                connection, _ = server.accept()
                accepted.set()
                with connection:
                    connection.sendall(b"broker-ok")

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()

            def serve_tcp():
                connection, _ = tcp_server.accept()
                tcp_accepted.set()
                with connection:
                    connection.sendall(b"tcp-ok")

            tcp_thread = threading.Thread(target=serve_tcp, daemon=True)
            tcp_thread.start()
            roots, files = self._controller_read_surface(allowed)
            controller_python = str(Path(sys.executable).resolve())
            policy = S.SandboxPolicy(
                str(controller), token="controller1",
                profile=S.CONTROLLER_PROFILE,
                readable_roots=roots, readable_files=files,
                broker_socket_path=str(broker_path),
                broker_peer_pid=os.getpid(),
                broker_peer_start_ticks=S._process_start_ticks())
            code = "\n".join((
                "import errno,json,os,socket",
                "def probe_open(path):",
                "    try:",
                "        with open(path, 'rb') as handle: return ['allowed', handle.read().decode(errors='replace')]",
                "    except OSError as exc: return ['denied', exc.errno]",
                "def probe_call(call):",
                "    try: call(); return ['allowed', None]",
                "    except OSError as exc: return ['denied', exc.errno]",
                "broker = socket.socket(fileno=int(os.environ['EPYC_AUTOKERNEL_BROKER_FD']))",
                "inet = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
                "client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
                f"client.connect(('127.0.0.1', {tcp_port}))",
                "render = {path: probe_open(path) for path in __import__('glob').glob('/dev/dri/renderD*')}",
                "result = {",
                f" 'allowed_file': probe_open({str(allowed)!r}),",
                f" 'campaign_sibling': probe_open({str(secret)!r}),",
                " 'kfd': probe_open('/dev/kfd') if os.path.exists('/dev/kfd') else ['absent', None],",
                " 'signal_parent': probe_call(lambda: os.kill(os.getppid(), 0)),",
                " 'inet_socket_created': True,",
                " 'inet_connect': client.recv(32).decode(),",
                " 'inet_bind': probe_call(lambda: inet.bind(('127.0.0.1', 0))),",
                " 'inet_listen': probe_call(lambda: inet.listen(1)),",
                " 'inet_accept': probe_call(lambda: inet.accept()),",
                " 'unix_socket': probe_call(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)),",
                " 'unnamed_socketpair': probe_call(lambda: socket.socketpair()),",
                " 'netlink_socket': probe_call(lambda: socket.socket(socket.AF_NETLINK, socket.SOCK_RAW)),",
                " 'render_nodes': render,",
                " 'broker': broker.recv(32).decode(),",
                "}",
                "print(json.dumps(result, sort_keys=True))",
            ))
            try:
                rc, stdout, stderr, receipt, teardown = self._run(
                    policy, [controller_python, "-c", code])
            finally:
                server.close()
                tcp_server.close()
            self.assertEqual(rc, 0, stderr)
            self.assertTrue(accepted.wait(timeout=2))
            self.assertTrue(tcp_accepted.wait(timeout=2))
            result = json.loads(stdout)
            self.assertEqual(result["allowed_file"], ["allowed", "allowed"])
            self.assertEqual(result["campaign_sibling"], ["denied", errno.EACCES])
            if result["kfd"][0] != "absent":
                self.assertEqual(result["kfd"], ["denied", errno.EACCES])
            self.assertTrue(all(
                row == ["denied", errno.EACCES]
                for row in result["render_nodes"].values()))
            self.assertEqual(result["signal_parent"], ["denied", errno.EPERM])
            self.assertTrue(result["inet_socket_created"])
            self.assertEqual(result["inet_connect"], "tcp-ok")
            self.assertEqual(result["inet_bind"], ["allowed", None])
            self.assertEqual(result["inet_listen"], ["denied", errno.EPERM])
            self.assertEqual(result["inet_accept"], ["denied", errno.EPERM])
            self.assertEqual(result["unix_socket"], ["denied", errno.EPERM])
            self.assertEqual(result["unnamed_socketpair"], ["allowed", None])
            self.assertEqual(result["netlink_socket"], ["denied", errno.EPERM])
            self.assertEqual(result["broker"], "broker-ok")
            self.assertEqual(receipt["network_profile"], S.NETWORK_OUTBOUND_CLIENT)
            self.assertEqual(receipt["outbound_socket_families"],
                             ["AF_INET", "AF_INET6"])
            self.assertEqual(receipt["server_socket_operations_denied"],
                             ["listen", "accept", "accept4"])
            self.assertTrue(receipt["unix_socket_creation_denied"])
            self.assertTrue(receipt["read_allowlist_enforced"])
            self.assertEqual(receipt["readable_files"], list(files))
            self.assertEqual(receipt["broker_socket_path"], str(broker_path))
            self.assertEqual(receipt["policy_sha256"], policy.policy_sha256)
            S.verify_receipt(
                receipt, policy=policy, pid=receipt["pid"],
                argv=[controller_python, "-c", code])
            self.assertTrue(teardown["verified_empty"])

    def test_controller_runtime_reads_are_exact_and_urandom_is_read_only(self):
        with tempfile.TemporaryDirectory(prefix="ak-controller-runtime-") as outer:
            root = Path(outer)
            controller = root / "controller"
            controller.mkdir()
            allowed = root / "allowed.txt"
            allowed.write_text("allowed", encoding="utf-8")
            broker_path = root / "broker.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(broker_path))
            server.listen(1)
            roots, files = self._controller_read_surface(allowed)
            policy = S.SandboxPolicy(
                str(controller), token="controllerruntime1",
                profile=S.CONTROLLER_PROFILE,
                readable_roots=roots,
                readable_files=(*files, *S.CONTROLLER_RUNTIME_READ_FILES),
                broker_socket_path=str(broker_path),
                broker_peer_pid=os.getpid(),
                broker_peer_start_ticks=S._process_start_ticks())
            code = (
                "import errno,json,os; "
                "stat=open('/proc/self/stat').read(); "
                "fd=os.open('/dev/urandom',os.O_RDONLY); "
                "random=os.read(fd,8); os.close(fd); "
                "denied=None; "
                "\ntry: os.open('/dev/urandom',os.O_WRONLY)\n"
                "except OSError as exc: denied=exc.errno\n"
                "print(json.dumps({'pid':int(stat.split()[0]),"
                "'random_bytes':len(random),'write_errno':denied}))")
            try:
                rc, stdout, stderr, receipt, teardown = self._run(
                    policy, [str(Path(sys.executable).resolve()), "-c", code])
            finally:
                server.close()
            self.assertEqual(rc, 0, stderr)
            result = json.loads(stdout)
            self.assertEqual(result["pid"], receipt["pid"])
            self.assertEqual(result["random_bytes"], 8)
            self.assertEqual(result["write_errno"], errno.EACCES)
            self.assertEqual(
                receipt["readable_files"],
                [*files, *S.CONTROLLER_RUNTIME_READ_FILES])
            self.assertTrue(teardown["verified_empty"])

    def test_controller_receipt_policy_digest_detects_allowlist_drift(self):
        with tempfile.TemporaryDirectory(prefix="ak-controller-policy-") as outer:
            root = Path(outer)
            controller = root / "controller"
            controller.mkdir()
            allowed = root / "allowed"
            allowed.write_text("one", encoding="utf-8")
            other = root / "other"
            other.write_text("two", encoding="utf-8")
            broker_path = root / "broker.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(broker_path))
            roots, files = self._controller_read_surface(allowed)
            first = S.SandboxPolicy(
                str(controller), token="digest1", profile=S.CONTROLLER_PROFILE,
                readable_roots=roots, readable_files=files,
                broker_socket_path=str(broker_path),
                broker_peer_pid=os.getpid(),
                broker_peer_start_ticks=S._process_start_ticks())
            _roots, other_files = self._controller_read_surface(other)
            second = S.SandboxPolicy(
                str(controller), token="digest1", profile=S.CONTROLLER_PROFILE,
                readable_roots=roots, readable_files=other_files,
                broker_socket_path=str(broker_path),
                broker_peer_pid=os.getpid(),
                broker_peer_start_ticks=S._process_start_ticks())
            server.close()
            self.assertNotEqual(first.policy_sha256, second.policy_sha256)

    def test_controller_broker_peer_mismatch_refuses_before_exec(self):
        with tempfile.TemporaryDirectory(prefix="ak-controller-peer-") as outer:
            root = Path(outer)
            controller = root / "controller"
            controller.mkdir()
            allowed = root / "allowed"
            allowed.write_text("one", encoding="utf-8")
            broker_path = root / "broker.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(broker_path))
            server.listen(1)
            roots, files = self._controller_read_surface(allowed)
            policy = S.SandboxPolicy(
                str(controller), token="wrongpeer1",
                profile=S.CONTROLLER_PROFILE,
                readable_roots=roots, readable_files=files,
                broker_socket_path=str(broker_path),
                broker_peer_pid=os.getpid(),
                broker_peer_start_ticks=S._process_start_ticks() + 1)
            evaluator = root / "evaluator"
            evaluator.mkdir()
            receipt = evaluator / "receipt.json"
            proc = subprocess.Popen(
                policy.wrap(["/bin/true"], receipt_path=str(receipt)),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True)
            connection, _ = server.accept()
            connection.close()
            server.close()
            _stdout, stderr = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 125)
            self.assertIn("broker peer identity changed", stderr)
            self.assertEqual(receipt.read_text(encoding="utf-8"), "")
            teardown = S.cleanup_cgroup(policy, proc.pid)
            self.assertTrue(teardown["verified_empty"])

    def test_controller_read_roots_cannot_admit_device_subtrees(self):
        if not Path("/dev/dri").is_dir():
            self.skipTest("host has no /dev/dri subtree")
        with tempfile.TemporaryDirectory(prefix="ak-controller-devroot-") as outer:
            root = Path(outer)
            controller = root / "controller"
            controller.mkdir()
            broker_path = root / "broker.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(broker_path))
            with self.assertRaisesRegex(S.SandboxError, "expose host devices"):
                S.SandboxPolicy(
                    str(controller), token="devroot1",
                    profile=S.CONTROLLER_PROFILE,
                    readable_roots=("/dev/dri",),
                    broker_socket_path=str(broker_path),
                    broker_peer_pid=os.getpid(),
                    broker_peer_start_ticks=S._process_start_ticks())
            server.close()

    def test_default_profile_retains_deny_all_network_and_unrestricted_reads(self):
        with tempfile.TemporaryDirectory(prefix="ak-default-profile-") as outer:
            root = Path(outer)
            writable = root / "candidate"
            writable.mkdir()
            outside = root / "outside.txt"
            outside.write_text("still-readable", encoding="utf-8")
            policy = S.SandboxPolicy(str(writable), token="defaultprofile1")
            code = (
                "import pathlib,socket; "
                f"print(pathlib.Path({str(outside)!r}).read_text()); "
                "socket.socket(socket.AF_INET, socket.SOCK_STREAM)")
            rc, stdout, _stderr, receipt, teardown = self._run(
                policy, [sys.executable, "-c", code])
            self.assertNotEqual(rc, 0)
            self.assertIn("still-readable", stdout)
            self.assertEqual(receipt["profile"], S.DEFAULT_PROFILE)
            self.assertEqual(receipt["network_profile"], S.NETWORK_DENY_ALL)
            self.assertFalse(receipt["read_allowlist_enforced"])
            self.assertTrue(teardown["verified_empty"])


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
