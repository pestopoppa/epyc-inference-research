from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import tempfile
import textwrap
import unittest
from unittest import mock

from . import claude_fable5_critic_actor as C


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


BINDINGS = {key: _sha(key) for key in C.BINDING_KEYS}


class ClaudeFable5CriticActorTests(unittest.TestCase):
    def _layout(self, temporary: str) -> tuple[Path, Path]:
        root = Path(temporary)
        workspace = root / "workspace"
        auth = root / "auth"
        workspace.mkdir(mode=0o700)
        auth.mkdir(mode=0o700)
        credential = auth / ".credentials.json"
        credential.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "test-secret"}}),
            encoding="utf-8",
        )
        credential.chmod(0o600)
        return workspace, auth

    def _wrapper(self, root: Path, body: str) -> Path:
        wrapper = root / "fake-claude"
        wrapper.write_text(
            "#!/usr/bin/env python3\n" + textwrap.dedent(body),
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        return wrapper

    def _success_wrapper(self, root: Path, *, payload_edit: str = "") -> Path:
        return self._wrapper(root, f"""
            import json
            import os
            from pathlib import Path
            import sys

            def stat_mode(path):
                return oct(path.stat().st_mode & 0o777)

            argv = sys.argv[1:]
            schema = json.loads(argv[argv.index('--json-schema') + 1])
            payload = {{
                'decision': 'accept',
                'reason': 'All bound inputs are internally consistent.',
            }}
            for key in {C.BINDING_KEYS!r}:
                payload[key] = schema['properties'][key]['const']
            {payload_edit}
            config = Path(os.environ['CLAUDE_CONFIG_DIR'])
            observed = {{
                'argv': argv,
                'wrapper_path': sys.argv[0],
                'wrapper_mode': stat_mode(Path(sys.argv[0])),
                'config_mode': stat_mode(config),
                'credentials_mode': stat_mode(config / '.credentials.json'),
                'state': json.loads((config / '.claude.json').read_text()),
                'mcp': json.loads((config / 'empty-mcp.json').read_text()),
                'credential_present': (config / '.credentials.json').is_file(),
                'ambient_claude': sorted(
                    key for key in os.environ
                    if key.startswith('CLAUDE_') and key not in {{
                        'CLAUDE_CONFIG_DIR', 'CLAUDE_CODE_DISABLE_AGENT_VIEW',
                        'CLAUDE_CODE_DISABLE_WORKFLOWS',
                        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'}}),
            }}
            Path('observed.json').write_text(json.dumps(observed))
            print(json.dumps({{'type': 'result', 'structured_output': payload}}))
        """)

    def test_runtime_identity_is_exact_non_secret_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = self._wrapper(Path(temporary), "print('{}')\n")
            identity = C.runtime_identity(wrapper)
        self.assertEqual(set(identity), {
            "kind", "provider", "model", "effort", "wrapper_path",
            "wrapper_sha256", "argv_policy_sha256", "auth_staging_policy",
        })
        self.assertEqual(identity["kind"], "claude_cli_structured_critic")
        self.assertEqual(identity["provider"], "claude")
        self.assertEqual(identity["model"], "claude-fable-5")
        self.assertEqual(identity["effort"], "high")
        self.assertEqual(
            identity["auth_staging_policy"],
            "ephemeral_0600_copy_no_secret_receipt",
        )
        self.assertNotIn("credential", json.dumps(identity).lower())
        self.assertNotIn("auth_root", identity)

    def test_exact_argv_disables_tools_mcp_customizations_and_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = self._wrapper(root, "print('{}')\n")
            config = root / "config"
            config.mkdir(mode=0o700)
            (config / "empty-mcp.json").write_bytes(C._EMPTY_MCP)
            (config / "empty-mcp.json").chmod(0o600)
            argv = C.build_argv(
                wrapper=wrapper, config_dir=config, bindings=BINDINGS)
        self.assertEqual(argv[0], str(wrapper))
        self.assertEqual(argv[argv.index("--model") + 1], "claude-fable-5")
        self.assertEqual(argv[argv.index("--effort") + 1], "high")
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
        self.assertIn("--strict-mcp-config", argv)
        self.assertIn("--no-session-persistence", argv)
        self.assertIn("--safe-mode", argv)
        self.assertIn("--disable-slash-commands", argv)
        start = argv.index("--disallowedTools")
        self.assertEqual(
            argv[start + 1:start + 5], ("Bash", "Edit", "Write", "NotebookEdit"))
        schema = json.loads(argv[argv.index("--json-schema") + 1])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(C.RESULT_KEYS))
        for key, value in BINDINGS.items():
            self.assertEqual(schema["properties"][key], {"const": value})

    def test_success_stages_scrubbed_auth_and_verifies_all_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace, auth = self._layout(temporary)
            wrapper = self._success_wrapper(Path(temporary))
            identity = C.runtime_identity(wrapper)
            launcher_sha = hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()
            result = C.run_critic(
                wrapper=wrapper, workspace=workspace, prompt="Review this full patch.",
                bindings=BINDINGS,
                environment={
                    "HOME": str(Path(temporary)), "PATH": os.environ["PATH"],
                    "CLAUDE_PROJECT_DIR": "/must/not/pass",
                    "ANTHROPIC_API_KEY": "must-not-pass",
                },
                auth_root=auth,
                expected_wrapper_sha256=identity["wrapper_sha256"],
                expected_runtime_identity=identity,
                expected_launcher_sha256=launcher_sha,
                timeout_seconds=5,
            )
            observed = json.loads((workspace / "observed.json").read_text())
            staged = list(workspace.glob(".autokernel-fable5-*"))
        self.assertEqual(result.decision, "accept")
        self.assertEqual(result.binding_map(), BINDINGS)
        self.assertEqual(observed["config_mode"], "0o700")
        self.assertEqual(observed["credentials_mode"], "0o600")
        self.assertEqual(observed["wrapper_mode"], "0o500")
        self.assertEqual(Path(observed["wrapper_path"]).name, "claude")
        self.assertNotEqual(observed["wrapper_path"], str(wrapper))
        self.assertEqual(observed["state"], {"hasCompletedOnboarding": True})
        self.assertEqual(observed["mcp"], {"mcpServers": {}})
        self.assertTrue(observed["credential_present"])
        self.assertEqual(observed["ambient_claude"], [])
        self.assertEqual(staged, [])

    def test_wrong_or_extra_structured_binding_is_rejected(self) -> None:
        cases = (
            "payload['context_sha256'] = '0' * 64",
            "payload['unexpected'] = 'field'",
            "payload.pop('reason')",
        )
        for edit in cases:
            with self.subTest(edit=edit), tempfile.TemporaryDirectory() as temporary:
                workspace, auth = self._layout(temporary)
                wrapper = self._success_wrapper(Path(temporary), payload_edit=edit)
                with self.assertRaisesRegex(C.ClaudeFable5CriticError, "structured output|echo exact"):
                    C.run_critic(
                        wrapper=wrapper, workspace=workspace, prompt="review",
                        bindings=BINDINGS,
                        environment={"HOME": temporary, "PATH": os.environ["PATH"]},
                        auth_root=auth, timeout_seconds=5,
                    )

    def test_non_json_or_unstructured_stdout_is_rejected(self) -> None:
        bodies = (
            "print('```json\\n{}\\n```')\n",
            "print('{}')\n",
            "print('{} {}')\n",
        )
        for body in bodies:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as temporary:
                workspace, auth = self._layout(temporary)
                wrapper = self._wrapper(Path(temporary), body)
                with self.assertRaises(C.ClaudeFable5CriticError):
                    C.run_critic(
                        wrapper=wrapper, workspace=workspace, prompt="review",
                        bindings=BINDINGS,
                        environment={"HOME": temporary}, auth_root=auth,
                        timeout_seconds=5,
                    )

    def test_nonzero_failure_summary_is_classified_without_error_text(self) -> None:
        secret = "operator-secret-error-detail"
        stdout = json.dumps({
            "type": "result", "subtype": "error_during_execution",
            "is_error": True, "result": f"Usage limit reached: {secret}",
        })
        summary = C._failure_summary(stdout, "")
        self.assertIn("category=usage_limit", summary)
        self.assertIn("subtype='error_during_execution'", summary)
        self.assertIn("envelope_keys=['is_error', 'result', 'subtype', 'type']", summary)
        self.assertNotIn(secret, summary)
        self.assertNotIn("Usage limit reached", summary)

    def test_unsafe_auth_mode_and_symlink_refuse_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace, auth = self._layout(temporary)
            marker = Path(temporary) / "spawned"
            wrapper = self._wrapper(
                Path(temporary), f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
            (auth / ".credentials.json").chmod(0o644)
            with self.assertRaisesRegex(C.ClaudeFable5CriticError, "mode is unsafe"):
                C.run_critic(
                    wrapper=wrapper, workspace=workspace, prompt="review",
                    bindings=BINDINGS,
                    environment={"HOME": temporary}, auth_root=auth,
                )
            self.assertFalse(marker.exists())
            (auth / ".credentials.json").unlink()
            target = Path(temporary) / "credential-target"
            target.write_text(json.dumps({"claudeAiOauth": {"token": "x"}}))
            target.chmod(0o600)
            (auth / ".credentials.json").symlink_to(target)
            with self.assertRaises(C.ClaudeFable5CriticError):
                C.run_critic(
                    wrapper=wrapper, workspace=workspace, prompt="review",
                    bindings=BINDINGS,
                    environment={"HOME": temporary}, auth_root=auth,
                )
            self.assertFalse(marker.exists())

    def test_timeout_uses_term_then_kill_and_proves_group_death(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace, auth = self._layout(temporary)
            pid_file = workspace / "pids.json"
            wrapper = self._wrapper(Path(temporary), """
                import json
                import os
                from pathlib import Path
                import signal
                import subprocess
                import sys
                import time
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                child = subprocess.Popen([
                    sys.executable, '-c',
                    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
                Path('pids.json').write_text(json.dumps([os.getpid(), child.pid]))
                while True:
                    time.sleep(1)
            """)
            signals: list[signal.Signals] = []
            original = os.killpg

            def recording(group_id: int, sig: signal.Signals) -> None:
                signals.append(sig)
                original(group_id, sig)

            with mock.patch.object(C.os, "killpg", side_effect=recording):
                with self.assertRaises(C.ClaudeFable5CriticTimeout):
                    C.run_critic(
                        wrapper=wrapper, workspace=workspace, prompt="review",
                        bindings=BINDINGS,
                        environment={"HOME": temporary, "PATH": os.environ["PATH"]},
                        auth_root=auth, timeout_seconds=0.15,
                        terminate_grace_seconds=0.1,
                    )
            pids = json.loads(pid_file.read_text())
            self.assertEqual(signals[:2], [signal.SIGTERM, signal.SIGKILL])
            for pid in pids:
                if Path(f"/proc/{pid}/stat").exists():
                    state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
                    self.assertEqual(state, "Z")
            self.assertEqual(list(workspace.glob(".autokernel-fable5-*")), [])

    def test_teardown_failure_surfaces_over_primary_process_error(self) -> None:
        fake = mock.Mock()
        fake.pid = 100001
        fake.communicate.side_effect = KeyboardInterrupt("primary")
        cleanup = C.ClaudeFable5CriticError("death proof failed")
        with mock.patch.object(C.subprocess, "Popen", return_value=fake), mock.patch.object(
                C, "_destroy_group", side_effect=cleanup):
            with self.assertRaisesRegex(C.ClaudeFable5CriticError, "death proof failed") as raised:
                C._run_process(
                    argv=("/bin/false",), cwd=Path("/tmp"), environment={},
                    prompt="x", timeout_seconds=1, terminate_grace_seconds=1,
                    capture_root=Path("/tmp"),
                )
        self.assertIsInstance(raised.exception.__cause__, KeyboardInterrupt)

    def test_original_wrapper_replacement_cannot_change_executed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace, auth = self._layout(temporary)
            wrapper = self._success_wrapper(Path(temporary))
            expected_sha = hashlib.sha256(wrapper.read_bytes()).hexdigest()
            payload = {
                "decision": "accept", "reason": "exact staged bytes",
                **BINDINGS,
            }
            envelope = json.dumps({"structured_output": payload})
            observed: dict[str, object] = {}

            def replace_before_exec(**kwargs: object) -> tuple[int, str, str]:
                argv = kwargs["argv"]
                assert isinstance(argv, tuple)
                staged = Path(argv[0])
                observed["staged"] = staged
                observed["digest"] = hashlib.sha256(staged.read_bytes()).hexdigest()
                observed["mode"] = stat.S_IMODE(staged.stat().st_mode)
                wrapper.write_text("#!/usr/bin/env python3\nraise SystemExit(99)\n")
                wrapper.chmod(0o700)
                return 0, envelope, ""

            with mock.patch.object(C, "_run_process", side_effect=replace_before_exec):
                with self.assertRaisesRegex(
                        C.ClaudeFable5CriticError, "changed during execution"):
                    C.run_critic(
                        wrapper=wrapper, workspace=workspace, prompt="review",
                        bindings=BINDINGS,
                        environment={"HOME": temporary}, auth_root=auth,
                    )
            self.assertNotEqual(observed["staged"], wrapper)
            self.assertEqual(observed["digest"], expected_sha)
            self.assertEqual(observed["mode"], 0o500)

    def test_launcher_mutation_during_staging_refuses_before_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace, auth = self._layout(temporary)
            wrapper = self._success_wrapper(Path(temporary))
            launcher_sha = C._launcher_sha256()
            with mock.patch.object(
                    C, "_launcher_sha256",
                    side_effect=(launcher_sha, "0" * 64)), mock.patch.object(
                        C, "_run_process") as run:
                with self.assertRaisesRegex(
                        C.ClaudeFable5CriticError, "changed during staging"):
                    C.run_critic(
                        wrapper=wrapper, workspace=workspace, prompt="review",
                        bindings=BINDINGS,
                        environment={"HOME": temporary}, auth_root=auth,
                    )
            run.assert_not_called()

    def test_launcher_is_rechecked_immediately_at_popen_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace, auth = self._layout(temporary)
            wrapper = self._success_wrapper(Path(temporary))
            launcher_sha = C._launcher_sha256()
            with mock.patch.object(
                    C, "_launcher_sha256",
                    side_effect=(launcher_sha, launcher_sha, "0" * 64)), \
                    mock.patch.object(C.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(
                        C.ClaudeFable5CriticError, "spawn boundary"):
                    C.run_critic(
                        wrapper=wrapper, workspace=workspace, prompt="review",
                        bindings=BINDINGS,
                        environment={"HOME": temporary}, auth_root=auth,
                    )
            popen.assert_not_called()

    def test_staged_executable_is_rechecked_at_popen_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace, auth = self._layout(temporary)
            wrapper = self._success_wrapper(Path(temporary))
            wrapper_sha = hashlib.sha256(wrapper.read_bytes()).hexdigest()
            with mock.patch.object(
                    C, "_executable_sha256",
                    side_effect=(wrapper_sha, "0" * 64)), \
                    mock.patch.object(C.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(
                        C.ClaudeFable5CriticError,
                        "staged Claude executable changed.*spawn boundary"):
                    C.run_critic(
                        wrapper=wrapper, workspace=workspace, prompt="review",
                        bindings=BINDINGS,
                        environment={"HOME": temporary}, auth_root=auth,
                    )
            popen.assert_not_called()

    def test_output_files_are_hard_capped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = self._wrapper(
                root,
                f"import sys\nsys.stdout.write('x' * {C.MAX_STDOUT_BYTES + 8192})\n",
            )
            returncode, stdout, _stderr = C._run_process(
                argv=(str(wrapper),), cwd=root,
                environment={"PATH": os.environ["PATH"]}, prompt="x",
                timeout_seconds=5, terminate_grace_seconds=1,
                capture_root=root,
            )
            self.assertNotEqual(returncode, 0)
            self.assertLessEqual(len(stdout.encode()), C.MAX_STDOUT_BYTES)

    def test_runtime_and_launcher_tamper_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace, auth = self._layout(temporary)
            wrapper = self._success_wrapper(Path(temporary))
            identity = C.runtime_identity(wrapper)
            wrapper.write_text("#!/usr/bin/env python3\nprint('{}')\n")
            wrapper.chmod(0o700)
            with self.assertRaisesRegex(C.ClaudeFable5CriticError, "wrapper bytes changed"):
                C.run_critic(
                    wrapper=wrapper, workspace=workspace, prompt="review",
                    bindings=BINDINGS,
                    environment={"HOME": temporary}, auth_root=auth,
                    expected_wrapper_sha256=identity["wrapper_sha256"],
                )
            with self.assertRaisesRegex(C.ClaudeFable5CriticError, "launcher"):
                C.run_critic(
                    wrapper=wrapper, workspace=workspace, prompt="review",
                    bindings=BINDINGS,
                    environment={"HOME": temporary}, auth_root=auth,
                    expected_launcher_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
