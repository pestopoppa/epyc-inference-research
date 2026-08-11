#!/usr/bin/env python3
"""Fail-closed tests for INF-03's first executable controller arm."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest

from . import claude_codex_actor_critic as A


def canonical_sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FakeRunner:
    def __init__(self, workspace: Path, *, planner_path: str = "kernel.py",
                 malformed_planner: bool = False, change_extra: bool = False,
                 timeout_role: str | None = None):
        self.workspace = workspace
        self.planner_path = planner_path
        self.malformed_planner = malformed_planner
        self.change_extra = change_extra
        self.timeout_role = timeout_role
        self.calls = []

    def __call__(self, argv, cwd, env, input_text, timeout_seconds):
        self.calls.append((tuple(argv), cwd, dict(env), input_text, timeout_seconds))
        executable = Path(argv[0]).name
        if executable == "claude" and "You are the planner" in input_text:
            if self.timeout_role == "planner":
                return A.ProcessCapture(tuple(argv), -15, "", "", True)
            if self.malformed_planner:
                stdout = "not-json"
            else:
                proposal = {
                    "schema": A.PROPOSAL_SCHEMA,
                    "proposal_id": "proposal-001",
                    "candidate_path": self.planner_path,
                    "actor_instruction": "Replace the fixture with the bounded candidate.",
                }
                stdout = json.dumps({"result": json.dumps(proposal)})
            return A.ProcessCapture(tuple(argv), 0, stdout, "")
        if executable == "codex":
            if self.timeout_role == "actor":
                return A.ProcessCapture(tuple(argv), -15, "", "", True)
            (self.workspace / "kernel.py").write_text(
                "def kernel():\n    return 2\n", encoding="utf-8")
            if self.change_extra:
                (self.workspace / "escaped.py").write_text("changed\n", encoding="utf-8")
            return A.ProcessCapture(tuple(argv), 0, '{"type":"turn.completed"}\n', "")
        if executable == "claude" and "You are the critic" in input_text:
            critique = {
                "schema": A.CRITIQUE_SCHEMA,
                "proposal_id": "proposal-001",
                "decision": "accept",
                "reason": "The candidate is ready for Arena evaluation.",
            }
            return A.ProcessCapture(
                tuple(argv), 0, json.dumps({"result": json.dumps(critique)}), "")
        raise AssertionError(f"unexpected command: {argv}")


class ActorCriticControllerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in A.REQUIRED_CLIS:
            path = self.bin / name
            path.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def workspace(self, name: str = "workspace") -> Path:
        root = self.root / name
        root.mkdir()
        (root / "kernel.py").write_text(
            "def kernel():\n    return 1\n", encoding="utf-8")
        (root / "escaped.py").write_text("original\n", encoding="utf-8")
        return root

    def environment(self):
        return {"PATH": str(self.bin)}

    @staticmethod
    def config(**overrides) -> A.ControllerConfig:
        payload = dict(
            claude_model=A.CLAUDE_MODEL,
            claude_effort=A.CLAUDE_EFFORT,
            codex_model=A.CODEX_MODEL,
            codex_effort=A.CODEX_EFFORT,
            checkpoint_hours=2.0,
            timeout_seconds=7200,
            max_iterations=3,
        )
        payload.update(overrides)
        return A.ControllerConfig(**payload)

    def test_model_and_effort_pins_are_exact_and_missing_pin_refuses(self):
        with self.assertRaisesRegex(A.ActorCriticError, "claude_model"):
            self.config(claude_model="opus")
        with self.assertRaisesRegex(A.ActorCriticError, "codex_effort"):
            self.config(codex_effort="medium")
        complete = self.config().__dict__
        with self.assertRaisesRegex(A.ActorCriticError, "missing.*codex_model"):
            A.ControllerConfig.from_mapping({
                key: value for key, value in complete.items() if key != "codex_model"})

    def test_timeout_is_required_and_exactly_one_campaign_checkpoint(self):
        with self.assertRaisesRegex(A.ActorCriticError, "timeout_seconds"):
            self.config(timeout_seconds=7199)
        with self.assertRaisesRegex(A.ActorCriticError, "checkpoint_hours"):
            self.config(checkpoint_hours=3.0, timeout_seconds=10800)
        payload = self.config().__dict__
        with self.assertRaisesRegex(A.ActorCriticError, "missing.*timeout_seconds"):
            A.ControllerConfig.from_mapping({
                key: value for key, value in payload.items()
                if key != "timeout_seconds"})

    def test_missing_installed_cli_refuses_without_invocation(self):
        empty = self.root / "empty-bin"
        empty.mkdir()
        with self.assertRaisesRegex(A.ActorCriticError, "not found: claude"):
            A.resolve_cli_identities({"PATH": str(empty)})
        (empty / "claude").write_text("not executable", encoding="utf-8")
        with self.assertRaisesRegex(A.ActorCriticError, "not found: claude"):
            A.resolve_cli_identities({"PATH": str(empty)})

    def test_success_binds_three_roles_workspace_and_every_artifact_hash(self):
        workspace = self.workspace()
        runner = FakeRunner(workspace)
        receipt = A.run_controller(
            prompt="Optimize the public Arena kernel.", workspace=workspace,
            config=self.config(), environment=self.environment(), runner=runner)
        self.assertEqual(receipt["stop_reason"], "critic_accept")
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual([Path(call[0][0]).name for call in runner.calls],
                         ["claude", "codex", "claude"])
        planner_argv, actor_argv, critic_argv = (
            call[0] for call in runner.calls)
        for argv in (planner_argv, critic_argv):
            self.assertIn(A.CLAUDE_MODEL, argv)
            self.assertIn(A.CLAUDE_EFFORT, argv)
            self.assertIn("plan", argv)
        self.assertIn(A.CODEX_MODEL, actor_argv)
        self.assertIn(f'model_reasoning_effort="{A.CODEX_EFFORT}"', actor_argv)
        self.assertIn("workspace-write", actor_argv)
        self.assertNotIn("dangerously-bypass-approvals-and-sandbox", actor_argv)
        self.assertEqual(receipt["candidate_artifacts"][0]["path"], "kernel.py")
        artifacts = workspace / A.ARTIFACT_DIRNAME
        for relative, digest in receipt["artifact_sha256"].items():
            self.assertEqual(
                hashlib.sha256((artifacts / relative).read_bytes()).hexdigest(), digest)
        without_self = {key: value for key, value in receipt.items()
                        if key != "receipt_sha256"}
        self.assertEqual(receipt["receipt_sha256"], canonical_sha(without_self))
        self.assertEqual(
            json.loads((artifacts / "receipt.json").read_text(encoding="utf-8")),
            receipt)

    def test_malformed_proposal_and_candidate_escape_fail_closed(self):
        malformed_workspace = self.workspace("malformed")
        with self.assertRaisesRegex(A.ActorCriticError, "malformed JSON"):
            A.run_controller(
                prompt="task", workspace=malformed_workspace, config=self.config(),
                environment=self.environment(),
                runner=FakeRunner(malformed_workspace, malformed_planner=True))
        escape_workspace = self.workspace("escape")
        with self.assertRaisesRegex(A.ActorCriticError, "escapes"):
            A.run_controller(
                prompt="task", workspace=escape_workspace, config=self.config(),
                environment=self.environment(),
                runner=FakeRunner(escape_workspace, planner_path="../outside.py"))

    def test_actor_cannot_change_a_second_workspace_path(self):
        workspace = self.workspace()
        with self.assertRaisesRegex(A.ActorCriticError, "outside its candidate"):
            A.run_controller(
                prompt="task", workspace=workspace, config=self.config(),
                environment=self.environment(),
                runner=FakeRunner(workspace, change_extra=True))

    def test_checkpoint_timeout_returns_without_parsing_partial_agent_output(self):
        workspace = self.workspace()
        receipt = A.run_controller(
            prompt="task", workspace=workspace, config=self.config(),
            environment=self.environment(),
            runner=FakeRunner(workspace, timeout_role="planner"))
        self.assertEqual(receipt["stop_reason"], "campaign_checkpoint")
        self.assertEqual(receipt["proposal_sha256"], [])
        transcript = (workspace / A.ARTIFACT_DIRNAME / "transcript.jsonl")
        event = json.loads(transcript.read_text(encoding="utf-8").strip())
        self.assertTrue(event["timed_out"])

    def test_process_timeout_kills_only_the_captured_fake_process_group(self):
        sleeper = self.root / "sleeper"
        sleeper.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
        sleeper.chmod(sleeper.stat().st_mode | stat.S_IXUSR)
        started = time.monotonic()
        capture = A._run_process(
            (str(sleeper),), self.root, os.environ, "", 0.05)
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(capture.timed_out)
        self.assertNotEqual(capture.returncode, 0)

    def test_registered_launcher_has_exact_agentkernelarena_three_arg_shape(self):
        workspace = self.workspace()
        registry = {}

        def register(name):
            def decorate(function):
                registry[name] = function
                return function
            return decorate

        runner = FakeRunner(workspace)
        launcher = A.register_agentkernelarena_launcher(
            register, lambda config, task_dir, path: "Arena prompt",
            runner=runner, environment=self.environment())
        self.assertIs(registry[A.CONTROLLER_ID], launcher)
        self.assertEqual(
            list(inspect.signature(launcher).parameters),
            ["eval_config", "task_config_dir", "workspace"],
        )
        rendered = launcher(
            {A.CONTROLLER_ID: self.config().__dict__},
            str(self.root / "task-config"), str(workspace))
        receipt = json.loads(rendered)
        self.assertEqual(receipt["controller_id"], A.CONTROLLER_ID)
        self.assertEqual(receipt["stop_reason"], "critic_accept")


if __name__ == "__main__":
    unittest.main()
