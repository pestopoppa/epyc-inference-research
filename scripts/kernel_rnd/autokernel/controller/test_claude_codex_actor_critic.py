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
from unittest import mock

from . import claude_codex_actor_critic as A


def canonical_sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FakeRunner:
    def __init__(self, workspace: Path, *, planner_path: str = "kernel.py",
                 malformed_planner: bool = False, change_extra: bool = False,
                 timeout_role: str | None = None,
                 planner_extra_fields: bool = False,
                 critic_decisions: tuple[str, ...] = ("accept",)):
        self.workspace = workspace
        self.planner_path = planner_path
        self.malformed_planner = malformed_planner
        self.change_extra = change_extra
        self.timeout_role = timeout_role
        self.planner_extra_fields = planner_extra_fields
        self.critic_decisions = critic_decisions
        self.planner_count = 0
        self.critic_count = 0
        self.last_proposal_id = "proposal-001"
        self.calls = []

    def __call__(self, argv, cwd, env, input_text, timeout_seconds):
        self.calls.append((tuple(argv), cwd, dict(env), input_text, timeout_seconds))
        executable = Path(argv[0]).name
        if executable == "claude" and "You are the planner" in input_text:
            self.planner_count += 1
            self.last_proposal_id = f"proposal-{self.planner_count:03d}"
            if self.timeout_role == "planner":
                return A.ProcessCapture(tuple(argv), -15, "", "", True)
            if self.malformed_planner:
                stdout = "not-json"
            else:
                proposal = {
                    "schema": A.PROPOSAL_SCHEMA,
                    "proposal_id": self.last_proposal_id,
                    "candidate_path": self.planner_path,
                    "actor_instruction": "Replace the fixture with the bounded candidate.",
                }
                if self.planner_extra_fields:
                    proposal.update({
                        "iteration": 1,
                        "target": {"hardware": "MI210"},
                        "diagnosis": {"current_state": "block-pointer load"},
                        "hypothesis": "Raw pointers may improve bandwidth.",
                        "constraints": ["Edit one file."],
                        "success_criteria": {"correctness": "tests pass"},
                        "risks": ["Measurement noise."],
                    })
                stdout = json.dumps({
                    "structured_output": proposal,
                    "result": "provider-rendered structured result",
                })
            return A.ProcessCapture(tuple(argv), 0, stdout, "")
        if A.codex_container_actor.EXECUTABLE_MODULE in argv:
            if self.timeout_role == "actor":
                return A.ProcessCapture(tuple(argv), -15, "", "", True)
            (self.workspace / "kernel.py").write_text(
                "def kernel():\n    return 2\n", encoding="utf-8")
            if self.change_extra:
                (self.workspace / "escaped.py").write_text("changed\n", encoding="utf-8")
            return A.ProcessCapture(tuple(argv), 0, '{"type":"turn.completed"}\n', "")
        if executable == "claude" and "You are the critic" in input_text:
            decision = self.critic_decisions[min(
                self.critic_count, len(self.critic_decisions) - 1)]
            self.critic_count += 1
            critique = {
                "schema": A.CRITIQUE_SCHEMA,
                "proposal_id": self.last_proposal_id,
                "decision": decision,
                "reason": f"Measured iteration {self.critic_count} requires {decision}.",
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
        native = self.root / A.codex_container_actor.CODEX_NATIVE_RELATIVE
        native.parent.mkdir(parents=True)
        for path in (native, native.with_name(
                A.codex_container_actor.CODE_MODE_HOST_NAME)):
            path.write_text("fake static executable\n", encoding="utf-8")
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
        self.assertEqual(Path(runner.calls[0][0][0]).name, "claude")
        self.assertEqual(Path(runner.calls[2][0][0]).name, "claude")
        self.assertIn(A.codex_container_actor.EXECUTABLE_MODULE,
                      runner.calls[1][0])
        planner_argv, actor_argv, critic_argv = (
            call[0] for call in runner.calls)
        for argv in (planner_argv, critic_argv):
            self.assertIn(A.CLAUDE_MODEL, argv)
            self.assertIn(A.CLAUDE_EFFORT, argv)
            self.assertIn("plan", argv)
            self.assertIn("--json-schema", argv)
        planner_schema = json.loads(
            planner_argv[planner_argv.index("--json-schema") + 1])
        critic_schema = json.loads(
            critic_argv[critic_argv.index("--json-schema") + 1])
        self.assertEqual(planner_schema, A.PROPOSAL_JSON_SCHEMA)
        self.assertFalse(planner_schema["additionalProperties"])
        self.assertFalse(critic_schema["additionalProperties"])
        self.assertEqual(
            critic_schema["properties"]["proposal_id"],
            {"const": "proposal-001"})
        self.assertIn(A.CODEX_MODEL, actor_argv)
        self.assertIn(A.CODEX_EFFORT, actor_argv)
        self.assertNotIn("dangerously-bypass-approvals-and-sandbox", actor_argv)
        self.assertIn("exactly these four fields and no others", runner.calls[2][3])
        self.assertEqual(receipt["candidate_artifacts"][0]["path"], "kernel.py")
        self.assertEqual(
            receipt["constraints"]["actor_sandbox"],
            "docker_workspace_bind_only")
        self.assertEqual(
            receipt["constraints"]["actor_runtime"]["image_id"],
            A.codex_container_actor.CONTAINER_IMAGE_ID)
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

    def test_revision_feedback_is_bound_and_supplied_to_the_next_planner(self):
        workspace = self.workspace("feedback")
        runner = FakeRunner(
            workspace, critic_decisions=("revise", "stop"))
        receipt = A.run_controller(
            prompt="Optimize with measured feedback.", workspace=workspace,
            config=self.config(), environment=self.environment(), runner=runner)
        planner_prompts = [call[3] for call in runner.calls
                           if "You are the planner" in call[3]]
        self.assertEqual(len(planner_prompts), 2)
        self.assertNotIn("Governed prior-iteration memory", planner_prompts[0])
        self.assertIn("Governed prior-iteration memory", planner_prompts[1])
        self.assertIn("proposal-001", planner_prompts[1])
        self.assertIn("Measured iteration 1 requires revise.", planner_prompts[1])
        self.assertEqual(receipt["stop_reason"], "critic_stop")
        self.assertEqual(len(receipt["feedback_memory"]), 2)
        self.assertEqual(
            receipt["feedback_memory_sha256"],
            canonical_sha(receipt["feedback_memory"]))

    def test_feedback_row_carries_governed_measurement_and_bounded_reason(self):
        reason = "r" * (A.MAX_FEEDBACK_REASON_CHARS + 17)
        measured = A.arena_upstream_common.EvaluationRecord(
            passed=True, latency_ms=0.02, speedup=0.99, log_excerpt="",
            raw={
                "pass_compilation": True, "pass_correctness": True,
                "valid_baseline_cases": 4, "valid_optimized_cases": 4,
                "average_speedup": 0.99,
                "best_optimized_execution_time": 0.02,
            })
        row = A._feedback_row(
            iteration=1,
            proposal={"proposal_id": "proposal-001", "candidate_path": "kernel.py"},
            candidate_sha256="a" * 64, measured=measured,
            critique={"decision": "revise", "reason": reason})
        self.assertEqual(row["arena_measurement"]["average_speedup"], 0.99)
        self.assertEqual(row["critic"]["reason"], reason)
        self.assertEqual(
            len(row["critic"]["reason_for_next_planner"]),
            A.MAX_FEEDBACK_REASON_CHARS)

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

    def test_live_extra_field_planner_shape_still_fails_closed(self):
        workspace = self.workspace("extra-fields")
        with self.assertRaisesRegex(
                A.ActorCriticError, "unknown/missing fields or schema"):
            A.run_controller(
                prompt="task", workspace=workspace, config=self.config(),
                environment=self.environment(),
                runner=FakeRunner(workspace, planner_extra_fields=True))

    def test_contained_absolute_candidate_is_normalized_but_escape_refuses(self):
        workspace = self.workspace("absolute")
        proposal = {
            "schema": A.PROPOSAL_SCHEMA,
            "proposal_id": "proposal-absolute",
            "candidate_path": str(workspace / "kernel.py"),
            "actor_instruction": "Implement the bounded candidate.",
        }
        parsed = A.parse_proposal(json.dumps(proposal), workspace)
        self.assertEqual(parsed["candidate_path"], "kernel.py")
        self.assertEqual(parsed["candidate_abspath"], str(workspace / "kernel.py"))

        proposal["candidate_path"] = str(self.root / "outside.py")
        (self.root / "outside.py").write_text("outside\n", encoding="utf-8")
        with self.assertRaisesRegex(A.ActorCriticError, "escapes"):
            A.parse_proposal(json.dumps(proposal), workspace)

    def test_candidate_symlink_is_rejected_even_when_target_is_contained(self):
        workspace = self.workspace("symlink")
        (workspace / "kernel-link.py").symlink_to("kernel.py")
        proposal = {
            "schema": A.PROPOSAL_SCHEMA,
            "proposal_id": "proposal-symlink",
            "candidate_path": "kernel-link.py",
            "actor_instruction": "Implement the bounded candidate.",
        }
        with self.assertRaisesRegex(A.ActorCriticError, "non-symlink"):
            A.parse_proposal(json.dumps(proposal), workspace)

    def test_semantic_manifest_excludes_only_reserved_controller_state(self):
        workspace = self.workspace("reserved-state")
        initial = A._workspace_manifest(workspace)
        for name in A.CONTROL_PLANE_DIRNAMES:
            root = workspace / name
            root.mkdir(exist_ok=True)
            (root / "session.json").write_text("one\n", encoding="utf-8")
        self.assertEqual(A._workspace_manifest(workspace), initial)
        (workspace / ".autokernel-controller-claude-config"
         / "session.json").write_text("two\n", encoding="utf-8")
        self.assertEqual(A._workspace_manifest(workspace), initial)
        (workspace / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
        self.assertNotEqual(A._workspace_manifest(workspace), initial)

        proposal = {
            "schema": A.PROPOSAL_SCHEMA,
            "proposal_id": "proposal-reserved",
            "candidate_path": ".autokernel-controller-claude-config/session.json",
            "actor_instruction": "Never admit control-plane state as a candidate.",
        }
        with self.assertRaisesRegex(A.ActorCriticError, "reserved"):
            A.parse_proposal(json.dumps(proposal), workspace)

    def test_claude_result_accepts_only_an_exact_single_json_fence(self):
        workspace = self.workspace("fenced")
        proposal = {
            "schema": A.PROPOSAL_SCHEMA,
            "proposal_id": "proposal-fenced",
            "candidate_path": "kernel.py",
            "actor_instruction": "Implement the bounded candidate.",
        }
        fenced = json.dumps({
            "result": f"```json\n{json.dumps(proposal)}\n```",
        })
        parsed = A.parse_proposal(fenced, workspace)
        self.assertEqual(parsed["proposal_id"], "proposal-fenced")
        with self.assertRaisesRegex(A.ActorCriticError, "malformed JSON"):
            A.parse_proposal(json.dumps({
                "result": f"preface\n```json\n{json.dumps(proposal)}\n```",
            }), workspace)
        with self.assertRaisesRegex(A.ActorCriticError, "malformed JSON"):
            A.parse_proposal(json.dumps({
                "result": "```json\n{}\n```\n```json\n{}\n```",
            }), workspace)

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

    def test_process_timeout_defers_to_parent_cgroup_without_signals(self):
        sleeper = self.root / "sleeper"
        sleeper.write_text("#!/bin/sh\nsleep 0.2\n", encoding="utf-8")
        sleeper.chmod(sleeper.stat().st_mode | stat.S_IXUSR)
        started = time.monotonic()
        with mock.patch.object(
                os, "killpg", side_effect=AssertionError("signal attempted")):
            capture = A._run_process(
                (str(sleeper),), self.root, os.environ, "", 0.02)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(capture.timed_out)
        self.assertIsNone(capture.returncode)

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
