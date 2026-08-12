#!/usr/bin/env python3
"""Controller-sandbox integration tests; tiny local processes, never inference."""
from __future__ import annotations

import errno
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from . import arena_adapter
from . import arena_controller_sandbox as C
from ..execution import sandbox


def _start_ticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    return int(text[text.rfind(")") + 2:].split()[19])


class ControllerSandboxContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="arena-controller-sandbox-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_workspace = self.root / "task-source"
        self.source_workspace.mkdir()
        (self.source_workspace / "task.py").write_text(
            "VALUE = 1\n", encoding="utf-8")
        self.workspace = self.root / "controller-workspace"
        self.copy_receipt = C.copy_controller_workspace(
            self.source_workspace, self.workspace)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.forbidden = self.root / "campaign-sibling"
        self.forbidden.mkdir()
        self.secret = self.forbidden / "secret.json"
        self.secret.write_text("campaign-secret", encoding="utf-8")
        self.module_root = Path(__file__).resolve().parents[3]
        self.python = Path(sys.executable).resolve(strict=True)
        self.fake_codex_root = self.root / "fake-codex-package"
        self.fake_codex_bin = self.fake_codex_root / "bin"
        self.fake_codex_bin.mkdir(parents=True)
        self.fake_codex = self.fake_codex_bin / "codex.js"
        self.fake_codex.write_text(
            "#!/usr/bin/env node\n// fake; never executed\n", encoding="utf-8")
        self.fake_node = self.root / "fake-node"
        shutil.copy2(self.python, self.fake_node)
        self.fake_node.chmod(self.fake_node.stat().st_mode | stat.S_IXUSR)
        self.fake_node_sibling = self.root / "fake-node-sibling"
        shutil.copy2(self.python, self.fake_node_sibling)
        self.fake_node_sibling.chmod(
            self.fake_node_sibling.stat().st_mode | stat.S_IXUSR)
        self.auth = self.root / "auth.json"
        self.auth.write_text("{}\n", encoding="utf-8")
        self.ca = self.root / "ca.pem"
        self.ca.write_text("test-ca\n", encoding="utf-8")

    def runtime(self) -> C.RuntimeAllowlist:
        return C.discover_runtime_allowlist(
            workspace=self.workspace, python_executable=self.python,
            controller_source_roots=(self.module_root,),
            controller_entrypoint=Path(__file__).resolve(),
            repository_module_roots=(), codex_cli=self.fake_codex,
            node_executable=self.fake_node, codex_auth=self.auth,
            ca_files=(self.ca,), forbidden_roots=(self.forbidden,),
        )

    def test_runtime_allowlist_rejects_broad_symlink_duplicate_device_and_state(self):
        runtime = self.runtime()
        self.assertNotIn("/", runtime.readable_roots)
        self.assertNotIn("/dev", runtime.readable_roots)
        self.assertIn(str(self.module_root), runtime.readable_roots)
        self.assertIn(str(self.python), runtime.executable_files)
        self.assertIn(str(self.fake_node), runtime.executable_files)
        self.assertIn(str(Path("/usr/bin/env").resolve(strict=True)),
                      runtime.executable_files)
        self.assertNotIn(str(self.fake_node_sibling), runtime.executable_files)
        self.assertEqual(runtime.identities[str(self.fake_codex)],
                         C._sha256_file(self.fake_codex))
        import _ctypes
        if getattr(_ctypes, "__file__", None) is not None:
            self.assertTrue(any(
                Path(path).name.startswith("libffi.so")
                for path in runtime.readable_files))

        symlink = self.root / "python-link"
        symlink.symlink_to(self.python)
        variants = (
            {"controller_source_roots": ("/",)},
            {"controller_source_roots": (self.module_root, self.module_root)},
            {"controller_source_roots": ("/dev",)},
            {"python_executable": symlink},
        )
        base = dict(
            workspace=self.workspace, python_executable=self.python,
            controller_source_roots=(self.module_root,),
            controller_entrypoint=Path(__file__).resolve(),
            repository_module_roots=(), codex_cli=self.fake_codex,
            node_executable=self.fake_node, codex_auth=self.auth,
            ca_files=(self.ca,), forbidden_roots=(self.forbidden,),
        )
        for change in variants:
            with self.subTest(change=change), self.assertRaises(C.ControllerSandboxError):
                C.discover_runtime_allowlist(**dict(base, **change))

        forged = replace(runtime, readable_roots=("/",))
        broker_path = self.root / "forged-broker.sock"
        broker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        broker.bind(str(broker_path))
        self.addCleanup(broker.close)
        with self.assertRaisesRegex(C.ControllerSandboxError, "too broad"):
            C.prepare_controller_sandbox(
                workspace=self.workspace,
                receipt_path=self.evidence / "forged-activation.json",
                expected_argv=(str(self.python), "-c", "pass"),
                runtime=forged, broker_socket_path=broker_path,
                broker_peer_pid=os.getpid(),
                broker_peer_start_ticks=_start_ticks(os.getpid()))

        campaign = Path("/mnt/raid0/llm/autokernel/campaigns")
        if campaign.is_dir() and not campaign.is_symlink():
            with self.assertRaisesRegex(C.ControllerSandboxError, "campaign/evidence"):
                C.discover_runtime_allowlist(
                    **dict(base, controller_source_roots=(campaign,)))

    def test_copy_workspace_rejects_symlink_and_binds_regular_files(self):
        self.assertEqual(self.copy_receipt["files"], {
            "task.py": C._sha256_file(self.source_workspace / "task.py")})
        source = self.root / "source-with-link"
        source.mkdir()
        (source / "link").symlink_to(self.secret)
        with self.assertRaisesRegex(C.ControllerSandboxError, "symlink"):
            C.copy_controller_workspace(source, self.root / "bad-copy")
        with self.assertRaisesRegex(C.ControllerSandboxError, "must not overlap"):
            C.copy_controller_workspace(
                self.source_workspace, self.source_workspace / "nested-copy")

    def test_live_controller_queues_peer_until_pid_registration_and_drains_descendant(self):
        runtime = self.runtime()
        broker_path = self.root / "broker.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(broker_path))
        server.listen(1)
        self.addCleanup(server.close)
        accepted = threading.Event()
        registered = threading.Event()
        broker_result: dict[str, object] = {}

        code = "\n".join((
            "import errno,json,os,socket,subprocess,sys,time",
            "def read(path):",
            "  try:",
            "    with open(path,'rb') as h: return ['allowed',h.read().decode(errors='replace')]",
            "  except OSError as exc: return ['denied',exc.errno]",
            "def execute(path):",
            "  try: return ['allowed',subprocess.run([path,'-c','pass']).returncode]",
            "  except OSError as exc: return ['denied',exc.errno]",
            "def probe_device(path):",
            "  result=[]",
            "  for flags in (os.O_RDONLY,os.O_RDWR):",
            "    try:",
            "      descriptor=os.open(path,flags); os.close(descriptor); result.append(['allowed',None])",
            "    except OSError as exc: result.append(['denied',exc.errno])",
            "  return result",
            "broker=socket.socket(fileno=int(os.environ['EPYC_AUTOKERNEL_BROKER_FD']))",
            "child_pid=os.fork()",
            "if child_pid == 0:",
            "  os.close(0); os.close(1); os.close(2); time.sleep(60); os._exit(0)",
            "result={",
            f" 'workspace':read({str(self.workspace / 'task.py')!r}),",
            f" 'campaign_sibling':read({str(self.secret)!r}),",
            f" 'sibling_executable':execute({str(self.fake_node_sibling)!r}),",
            " 'null':probe_device('/dev/null'),",
            " 'kfd':probe_device('/dev/kfd'),",
            " 'renderD128':probe_device('/dev/dri/renderD128'),",
            " 'broker':broker.recv(32).decode(),",
            " 'peer':list(broker.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12)),",
            " 'child_pid':child_pid}",
            "print(json.dumps(result,sort_keys=True),flush=True)",
        ))
        argv = (str(self.python), "-c", code)
        invocation = C.prepare_controller_sandbox(
            workspace=self.workspace,
            receipt_path=self.evidence / "activation.json",
            expected_argv=argv, runtime=runtime,
            broker_socket_path=broker_path, broker_peer_pid=os.getpid(),
            broker_peer_start_ticks=_start_ticks(os.getpid()),
        )

        def broker() -> None:
            connection, _ = server.accept()
            with connection:
                raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                peer_pid = int.from_bytes(raw[:4], sys.byteorder, signed=True)
                broker_result["peer_pid"] = peer_pid
                broker_result["accepted_before_registration"] = invocation.pid is None
                accepted.set()
                if not registered.wait(timeout=5):
                    broker_result["error"] = "registration timeout"
                    return
                broker_result["registered_pid"] = invocation.pid
                broker_result["registered_start_ticks"] = _start_ticks(peer_pid)
                connection.sendall(b"broker-ok")

        thread = threading.Thread(target=broker, daemon=True)
        thread.start()
        prepared = arena_adapter.prepare_task(
            arena_adapter.ArenaTask(
                task_id="tiny/controller", task_prompt="Return controller probe.",
                workspace=str(self.workspace), controller_id="kernelfoundry",
                round_id="sandbox-test", actual_gfx_arch="gfx90a"),
            base_environment={
                "PATH": os.environ["PATH"], "PYTHONPATH": "",
                **invocation.environment_overrides})

        def started(pid: int) -> None:
            self.assertTrue(accepted.wait(timeout=5))
            invocation.process_started(pid)
            registered.set()

        output = arena_adapter.launch(
            prepared, argv, timeout_seconds=10,
            command_prefix=invocation.command_prefix,
            process_started=started)
        thread.join(timeout=5)
        self.assertNotIn("error", broker_result)
        self.assertTrue(broker_result["accepted_before_registration"])
        self.assertEqual(broker_result["peer_pid"], invocation.pid)
        self.assertEqual(broker_result["registered_pid"], invocation.pid)
        result = json.loads(output)
        self.assertEqual(result["workspace"], ["allowed", "VALUE = 1\n"])
        self.assertEqual(result["campaign_sibling"], ["denied", errno.EACCES])
        self.assertEqual(result["sibling_executable"], ["denied", errno.EACCES])
        self.assertEqual(result["null"], [["allowed", None]] * 2)
        self.assertEqual(result["kfd"], [["denied", errno.EACCES]] * 2)
        self.assertEqual(result["renderD128"], [["denied", errno.EACCES]] * 2)
        self.assertEqual(result["broker"], "broker-ok")
        teardown = invocation.verify_and_teardown(
            self.evidence / "teardown.json")
        activation = sandbox.read_receipt(self.evidence / "activation.json")
        self.assertEqual(
            activation["executable_files"], list(runtime.executable_files))
        drift_policy = replace(
            invocation.policy,
            executable_files=(str(self.python), str(self.fake_node_sibling)))
        self.assertNotEqual(
            invocation.policy.policy_sha256, drift_policy.policy_sha256)
        with self.assertRaisesRegex(sandbox.SandboxError, "executable_files"):
            sandbox.verify_receipt(
                activation, policy=drift_policy, pid=invocation.pid,
                argv=argv)
        self.assertTrue(teardown["teardown"]["verified_empty"])
        self.assertTrue(teardown["teardown"]["removed"])
        self.assertTrue(teardown["teardown"]["descendants_killed"])
        for _ in range(50):
            if not Path(f"/proc/{result['child_pid']}").exists():
                break
            time.sleep(0.02)
        self.assertFalse(Path(f"/proc/{result['child_pid']}").exists())

    def test_wrong_broker_peer_refuses_without_activation_receipt(self):
        runtime = self.runtime()
        broker_path = self.root / "wrong-broker.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(broker_path))
        server.listen(1)
        self.addCleanup(server.close)
        invocation = C.prepare_controller_sandbox(
            workspace=self.workspace,
            receipt_path=self.evidence / "wrong-activation.json",
            expected_argv=(str(self.python), "-I", "-c", "print('never')"),
            runtime=runtime, broker_socket_path=broker_path,
            broker_peer_pid=os.getpid(),
            broker_peer_start_ticks=_start_ticks(os.getpid()) + 1)
        prepared = arena_adapter.prepare_task(
            arena_adapter.ArenaTask(
                task_id="tiny/wrong-peer", task_prompt="Probe.",
                workspace=str(self.workspace), controller_id="kernelfoundry",
                round_id="sandbox-test", actual_gfx_arch="gfx90a"),
            base_environment={
                "PATH": os.environ["PATH"], "PYTHONPATH": "",
                **invocation.environment_overrides})

        accepted = threading.Event()
        def accept() -> None:
            connection, _ = server.accept()
            accepted.set()
            connection.close()
        thread = threading.Thread(target=accept, daemon=True)
        thread.start()
        with self.assertRaisesRegex(arena_adapter.ArenaAdapterError, "exited 125"):
            arena_adapter.launch(
                prepared, invocation.expected_argv, timeout_seconds=10,
                command_prefix=invocation.command_prefix,
                process_started=invocation.process_started)
        self.assertTrue(accepted.wait(timeout=2))
        self.assertEqual((self.evidence / "wrong-activation.json").read_text(), "")
        teardown = sandbox.cleanup_cgroup(invocation.policy, invocation.pid)
        self.assertTrue(teardown["verified_empty"])


if __name__ == "__main__":
    unittest.main()
