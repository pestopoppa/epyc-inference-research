#!/usr/bin/env python3
"""Tests for the opencode DeepSeek V4 Flash backup critic actor."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import opencode_deepseek_critic_actor as C


def _h(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _bindings() -> dict[str, str]:
    return {key: _h(key) for key in C.BINDING_KEYS}


def _event_stream(decision: str = "accept", reason: str = "ok",
                  bindings: dict[str, str] | None = None,
                  text: str | None = None) -> bytes:
    bindings = bindings or _bindings()
    critique = {"decision": decision, "reason": reason, **bindings}
    if text is not None:
        body = text
    else:
        body = json.dumps(critique, sort_keys=True)
    return ("\n".join((
        json.dumps({"type": "event", "subtype": "session.initialized"}),
        json.dumps({"type": "message", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": body}]}}),
        json.dumps({"type": "done"}),
    )) + "\n").encode()


class OpenCodeCriticActorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        home = self.root / "home"
        (home / ".local/share/opencode").mkdir(parents=True)
        (home / ".config/opencode").mkdir(parents=True)
        auth = home / ".local/share/opencode/auth.json"
        auth.write_bytes(b'{"provider":"deepseek","key":"test"}\n')
        auth.chmod(0o600)
        (home / ".config/opencode/opencode.jsonc").write_text(
            '{"model":"deepseek/deepseek-v4-flash"}\n')
        self.wrapper = self.root / "opencode.exe"
        self.wrapper.write_bytes(b"#!/bin/sh\necho opencode\n")
        self.wrapper.chmod(0o755)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.identity = C.runtime_identity(self.wrapper)
        self._home_patch = mock.patch("pathlib.Path.home",
                                      return_value=home)
        self._home_patch.start()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self._tmp.cleanup()

    def _run(self, *, stdout: bytes = b"", stderr: bytes = b"", code: int = 0):
        calls: list[tuple] = []
        staged: dict[str, object] = {}

        def fake_run_process(**kwargs):
            calls.append(kwargs)
            env = kwargs["environment"]
            data = Path(env["XDG_DATA_HOME"]) / "opencode"
            staged["auth"] = (data / "auth.json").read_bytes()
            staged["auth_mode"] = stat.S_IMODE((data / "auth.json").stat().st_mode)
            config = Path(env["XDG_CONFIG_HOME"]) / "opencode"
            staged["config_exists"] = (config / "opencode.jsonc").is_file()
            return stdout, stderr, code

        with mock.patch.object(C, "_run_process", side_effect=fake_run_process):
            result = C.run_critic(
                wrapper=self.wrapper, workspace=self.workspace,
                prompt='{"role":"critic"}', bindings=_bindings(),
                environment={"PATH": "/usr/bin:/bin"},
                expected_wrapper_sha256=self.identity["wrapper_sha256"],
                expected_runtime_identity=self.identity)
        return result, calls, staged

    def test_happy_path_binds_everything_and_stages_private_auth(self):
        result, calls, staged = self._run(stdout=_event_stream())
        self.assertEqual(staged["auth"], b'{"provider":"deepseek","key":"test"}\n')
        self.assertEqual(staged["auth_mode"], 0o600)
        self.assertTrue(staged["config_exists"])
        self.assertEqual(result.decision, "accept")
        self.assertEqual(result.reason, "ok")
        self.assertEqual(result.binding_map(), _bindings())
        self.assertEqual(result.wrapper_sha256,
                         self.identity["wrapper_sha256"])
        self.assertEqual(result.argv_sha256, C._sha256_bytes(
            C._canonical_bytes(list(calls[0]["argv"]))))
        self.assertEqual(result.stdout_sha256,
                         C._sha256_bytes(_event_stream()))
        argv = calls[0]["argv"]
        self.assertEqual(argv[0], str(self.wrapper))
        self.assertEqual(argv[1], "run")
        self.assertIn("--model", argv)
        self.assertEqual(argv[argv.index("--model") + 1], C.MODEL)
        self.assertIn("--format", argv)
        self.assertEqual(argv[argv.index("--format") + 1], "json")
        self.assertIn("--pure", argv)
        env = calls[0]["environment"]
        self.assertTrue(
            Path(env["XDG_DATA_HOME"]).is_relative_to(self.workspace))
        self.assertTrue(
            Path(env["XDG_CONFIG_HOME"]).is_relative_to(self.workspace))
        self.assertFalse(self.wrapper.is_symlink())

    def test_binding_mismatch_is_refused(self):
        wrong = dict(_bindings())
        wrong["context_sha256"] = _h("other")
        with mock.patch.object(C, "_run_process", return_value=(
                _event_stream(bindings=wrong), b"", 0)):
            with self.assertRaisesRegex(C.OpenCodeCriticError,
                                        "binding context_sha256 differs"):
                C.run_critic(
                    wrapper=self.wrapper, workspace=self.workspace,
                    prompt="p", bindings=_bindings(),
                    environment={"PATH": "/usr/bin:/bin"})

    def test_wrong_decision_is_refused(self):
        with mock.patch.object(C, "_run_process", return_value=(
                _event_stream(decision="maybe"), b"", 0)):
            with self.assertRaisesRegex(C.OpenCodeCriticError,
                                        "critique shape changed"):
                C.run_critic(
                    wrapper=self.wrapper, workspace=self.workspace,
                    prompt="p", bindings=_bindings(),
                    environment={"PATH": "/usr/bin:/bin"})

    def test_nonzero_exit_is_refused(self):
        with mock.patch.object(C, "_run_process", return_value=(b"", b"boom", 1)):
            with self.assertRaisesRegex(C.OpenCodeCriticError, "exited nonzero"):
                C.run_critic(
                    wrapper=self.wrapper, workspace=self.workspace,
                    prompt="p", bindings=_bindings(),
                    environment={"PATH": "/usr/bin:/bin"})

    def test_timeout_is_refused_and_group_destroyed(self):
        def fake_run_process(**kwargs):
            raise C.OpenCodeCriticTimeout("timed out")
        with mock.patch.object(C, "_run_process", side_effect=fake_run_process):
            with self.assertRaises(C.OpenCodeCriticTimeout):
                C.run_critic(
                    wrapper=self.wrapper, workspace=self.workspace,
                    prompt="p", bindings=_bindings(),
                    environment={"PATH": "/usr/bin:/bin"})

    def test_missing_audio_auth_is_refused(self):
        (self.root / "home/.local/share/opencode/auth.json").unlink()
        with self.assertRaisesRegex(C.OpenCodeCriticError, "auth is unavailable"):
            C.run_critic(
                wrapper=self.wrapper, workspace=self.workspace,
                prompt="p", bindings=_bindings(),
                environment={"PATH": "/usr/bin:/bin"})

    def test_wrapper_drift_is_refused_before_any_run(self):
        with mock.patch.object(C, "_run_process", return_value=(b"", b"", 0)):
            with self.assertRaisesRegex(C.OpenCodeCriticError,
                                        "wrapper bytes changed"):
                C.run_critic(
                    wrapper=self.wrapper, workspace=self.workspace,
                    prompt="p", bindings=_bindings(),
                    environment={"PATH": "/usr/bin:/bin"},
                    expected_wrapper_sha256="0" * 64)

    def test_prose_framed_critique_is_extracted(self):
        text = "Here is the critique:\n" + json.dumps(
            {"decision": "revise", "reason": "re", **_bindings()},
            sort_keys=True) + "\nend"
        result, _, _ = self._run(stdout=_event_stream(text=text))
        self.assertEqual(result.decision, "revise")

    def test_argv_policy_is_sealed_and_stable(self):
        self.assertTrue(C.ARGV_POLICY_SHA256)
        self.assertEqual(C._ARGV_POLICY["model"], C.MODEL)
        self.assertEqual(C._ARGV_POLICY["input"], "positional_message")
        self.assertIs(C._ARGV_POLICY["pure"], True)
        identity = C.runtime_identity(self.wrapper)
        self.assertEqual(identity["kind"], C.RUNTIME_KIND)
        self.assertEqual(identity["argv_policy_sha256"],
                         C.ARGV_POLICY_SHA256)
        self.assertEqual(identity["auth_staging_policy"],
                         C.AUTH_STAGING_POLICY)

    def test_prompt_argv_bound_is_enforced(self):
        with self.assertRaisesRegex(C.OpenCodeCriticError, "argv bound"):
            C.build_argv(wrapper=self.wrapper,
                         prompt="x" * (C.MAX_PROMPT_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
