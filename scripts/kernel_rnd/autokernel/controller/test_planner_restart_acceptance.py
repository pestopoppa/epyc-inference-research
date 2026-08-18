"""Red acceptance gate for planner refusal and restart-safe actor checkpoints.

This module is intentionally committed against ``43e59899`` before the product
change.  It exercises only disposable directories and patched actor calls: no
build, inference, lease, screen, or GPU path is reachable.
"""
from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .. import hypothesis_portfolio
from . import discovery_controller as D


H = "a" * 64
RUNTIME = {
    "kind": "docker_workspace_bind_only",
    "docker_path": "/docker",
    "docker_sha256": H,
    "image_id": "image",
    "codex_native_sha256": H,
    "code_mode_host_sha256": H,
    "ca_certificate_sha256": H,
    "writable_host_binds": ["/workspace"],
    "host_network_mode": "docker_bridge",
}
CLAUDE_RUNTIME = {
    "kind": "claude_cli_structured_critic",
    "provider": "claude",
    "model": "claude-fable-5",
    "effort": "high",
    "wrapper_path": "/sealed/claude",
    "wrapper_sha256": H,
    "argv_policy_sha256": H,
    "auth_staging_policy":
        "ephemeral_0600_copy_atomic_oauth_rotation_sync_no_secret_receipt",
}


class SimulatedProcessExit(SystemExit):
    """An interruption outside the controller's ordinary Exception boundary."""


class GenericPlannerFailure(RuntimeError):
    """An unclassified failure that must never become a planner refusal."""


class CountingCritic:
    def __init__(self, decision: str = "reject", *, interrupt: bool = False):
        self.decision = decision
        self.interrupt = interrupt
        self.calls = 0

    def attest(self):
        return {**D.FABLE5_CRITIC, "runtime": CLAUDE_RUNTIME}

    def review(self, *_args, **_kwargs):
        self.calls += 1
        if self.interrupt:
            raise SimulatedProcessExit("after valid planner checkpoint")
        return D.Critique(self.decision, "bounded acceptance fixture")


class NeverLease:
    def __init__(self):
        self.calls = 0

    def admit(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("planner acceptance gate reached a resource lease")


class NeverScreen:
    def __init__(self):
        self.calls = 0

    def screen(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("planner acceptance gate reached a screen")

    def reconcile(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("planner acceptance gate reached reconciliation")


class RefusingPlanner:
    """Typed-refusal fake used only after actor-output classification is tested."""

    def __init__(self, exception_factory):
        self.exception_factory = exception_factory
        self.contexts = []

    def attest(self):
        return {**D.SOL, "runtime": RUNTIME}

    def plan(self, *, context, workspace):
        self.contexts.append(context)
        raise self.exception_factory()


class PlannerRestartAcceptance(unittest.TestCase):
    q5_hypothesis_id = "akh-q5-0-packed-load"
    q5_file = "ggml/src/ggml-cuda/vecdotq.cuh"
    q5_symbol = "vec_dot_q5_0_q8_1"

    def portfolio_record(self):
        return {
            "hypothesis_id": self.q5_hypothesis_id,
            "statement": "reuse exact Q5_0 packed loads",
            "primary_falsifier": "replicated decode gain is below the sealed floor",
            "falsifiers": ["replicated decode gain is below the sealed floor"],
            "regime": {
                "frame_id": "qwen05",
                "architecture": "gfx90a",
                "phase": "decode",
            },
            "target": {
                "source_files": [self.q5_file],
                "source_symbols": [self.q5_symbol],
                "template_intent": "cuda-q5-0-v1",
            },
            "mechanism": {
                "fingerprint_sha256": H,
                "facets": {"change_class": "arithmetic"},
            },
            "priority": {"rank": 1},
            "current_bundle_eligibility": {
                "eligible": True,
                "template_ids": ["cuda-q5-0-v1"],
            },
            "decision_policy": {
                "frame_id": "qwen05",
                "continuation_floor_pct": 0.4,
                "nomination_floor_pct": 0.8,
                "required_replications": 2,
                "sign_policy": "all_positive",
                "conflict_policy": "retain_inconclusive",
                "max_distinct_candidates": 1,
                "terminal_rule": "retire",
                "metric": "decode_tokens_per_s",
                "effect_unit": "relative_percent",
                "min_replication_effect_pct": 0.0,
                "max_replication_spread_pct": 1.0,
            },
        }

    def source_package(self):
        content = (
            b"template <typename T>\n"
            b"void vec_dot_q5_0_q8_1(T * x) { x[0] = T{}; }\n"
        )
        digest = hashlib.sha256(content).hexdigest()
        body = {
            "schema": "epyc.autokernel.reviewed_source_package.v1",
            "instrument_commit": "1" * 40,
            "files": [{
                "relative_path": self.q5_file,
                "sha256": digest,
                "workspace_path": f"reviewed-source/{self.q5_file}",
            }],
        }
        return D.ReviewedSourcePackage(
            "1" * 40,
            (D.ReviewedSourceFile(self.q5_file, digest, content),),
            D._sha(body),
        )

    def config(self, root: Path, *, max_iterations: int = 1):
        record = self.portfolio_record()
        portfolio = hypothesis_portfolio.Portfolio(
            {"hypotheses": [record], "frames": [], "do_not_repeat": []},
            "f" * 64,
        )
        package = self.source_package()
        context = {
            "reviewed_source_package_sha256": package.package_sha256,
            "portfolio_dispatch_authority": {
                self.q5_hypothesis_id: [{
                    "route_id": "cuda-q5-0-v1.anchor.0",
                    "kernel_name": self.q5_symbol,
                    "calls": 18705,
                    "grid": 1024,
                    "workgroup": 256,
                    "lds_bytes": 0,
                }],
            },
        }
        config = D.ControllerConfig(
            root / "controller",
            max_iterations,
            dry_run=True,
            planner_context=context,
            planner_context_sha256="e" * 64,
            production_base_commit="0" * 40,
            instrument_commit="1" * 40,
            hypothesis_portfolio=portfolio,
            hypothesis_portfolio_sha256="f" * 64,
        )
        return config, package

    def planner(self, root: Path, package):
        wrapper = root / "codex"
        wrapper.write_bytes(b"codex")
        wrapper.chmod(0o700)
        return D.CodexPlanner(
            wrapper=wrapper,
            environment={"PATH": "/usr/bin"},
            reviewed_sources=package,
        )

    def write_actor_output(self, kwargs, *, hunk_before_header: bool = False,
                           change_campaign_authority: bool = False):
        prompt = json.loads(kwargs["prompt"])
        assignment = prompt["context"]["authoring_assignment"]
        binding = assignment["portfolio_binding"]
        workspace = Path(kwargs["workspace"])
        if hunk_before_header:
            patch_bytes = (
                f"@@ -1 +1 @@ {self.q5_symbol}()\n-old\n+new\n"
            ).encode()
        else:
            patch_bytes = (
                f"diff --git a/{self.q5_file} b/{self.q5_file}\n"
                f"--- a/{self.q5_file}\n"
                f"+++ b/{self.q5_file}\n"
                f"@@ -1 +1 @@ {self.q5_symbol}()\n-old\n+new\n"
            ).encode()
        manifest = {
            "schema": D.source_candidate.SCHEMA_SOURCE_PATCH,
            "campaign_id": assignment["campaign_id"],
            "proposal_id": assignment["proposal_id"],
            "candidate_id": assignment["candidate_id"],
            "source_tree": "llama.cpp",
            "production_base_commit": assignment["production_base_commit"],
            "instrument_commit": assignment["instrument_commit"],
            "change_class": binding["change_class"],
            "declared_files": [binding["target_file"]],
            "declared_symbols": {
                binding["target_file"]: list(binding["target_symbols"]),
            },
            "mechanism_id": binding["mechanism_id"],
            "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
            "patch_encoding": "base64",
            "patch_base64": base64.b64encode(patch_bytes).decode("ascii"),
        }
        if change_campaign_authority:
            manifest["campaign_id"] = "ak-actor-invented"
        plan = {
            "hypothesis_id": binding["hypothesis_id"],
            "statement": binding["statement"],
            "falsifier": binding["falsifier"],
            "regime": binding["regime"],
            "proposal": {
                "proposal_id": assignment["proposal_id"],
                "change_class": binding["change_class"],
                "change": {
                    "files_and_symbols": [
                        f"{binding['target_file']}:{symbol}"
                        for symbol in binding["target_symbols"]
                    ],
                    "estimated_diff_size": 2,
                },
            },
            "source_manifest_path": "source-patch.json",
            "experiment_intent": {
                "template_id": binding["template_id"],
                "target_surface": "gpu_decode",
                "target_symbol": binding["target_symbols"][0],
                "correctness_id": "backend-ops-hip-v1",
                "dispatch_id": "decode-tg128-rocprof-v1",
                "expected_dispatch": binding["expected_dispatch"],
            },
        }
        (workspace / "source-patch.json").write_text(json.dumps(manifest))
        (workspace / "plan.json").write_text(json.dumps(plan))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def test_live_hunk_before_header_is_typed_planner_output_refusal(self):
        refusal = getattr(D, "PlannerOutputRefusal", type(
            "MissingPlannerOutputRefusal", (D.DiscoveryControllerError,), {}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, package = self.config(root)
            planner = self.planner(root, package)
            binding = D._select_portfolio_binding({"iterations": []}, config)
            context = D._context(
                {"iterations": []},
                D._tracker(D.DurableState(config.output_root)),
                1,
                config,
                binding,
            )
            workspace = root / "planner-workspace"
            workspace.mkdir()
            with patch.object(
                    D.codex_container_actor, "runtime_identity",
                    return_value=RUNTIME), patch.object(
                    D.codex_container_actor, "run_actor",
                    side_effect=lambda **kwargs: self.write_actor_output(
                        kwargs, hunk_before_header=True)):
                with self.assertRaisesRegex(
                        refusal, "hunk header outside a file section"):
                    planner.plan(context=context, workspace=workspace)

    def test_live_identity_authority_violation_is_not_a_planner_refusal(self):
        refusal = getattr(D, "PlannerOutputRefusal", None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, package = self.config(root)
            planner = self.planner(root, package)
            binding = D._select_portfolio_binding({"iterations": []}, config)
            context = D._context(
                {"iterations": []},
                D._tracker(D.DurableState(config.output_root)),
                1,
                config,
                binding,
            )
            workspace = root / "planner-workspace"
            workspace.mkdir()
            with patch.object(
                    D.codex_container_actor, "runtime_identity",
                    return_value=RUNTIME), patch.object(
                    D.codex_container_actor, "run_actor",
                    side_effect=lambda **kwargs: self.write_actor_output(
                        kwargs, change_campaign_authority=True)):
                with self.assertRaisesRegex(
                        D.DiscoveryControllerError,
                        "actor attempted to invent campaign/base/instrument identity") as caught:
                    planner.plan(context=context, workspace=workspace)
        if refusal is not None:
            self.assertNotIsInstance(caught.exception, refusal)

    def test_refusals_are_durable_bounded_and_do_not_spend_q5_candidate_budget(self):
        refusal = getattr(D, "PlannerOutputRefusal", D.DiscoveryControllerError)
        planner = RefusingPlanner(
            lambda: refusal("SourceCandidateError: hunk appears before a diff --git header"))
        critic = CountingCritic()
        lease, screen = NeverLease(), NeverScreen()
        with tempfile.TemporaryDirectory() as temp:
            config, _package = self.config(Path(temp), max_iterations=3)
            result = D.run_controller(
                config, planner=planner, critic=critic,
                screener=screen, lease=lease)
            events = D.DurableState(config.output_root).book.read_all()

        self.assertTrue(result["complete"])
        self.assertEqual(result["next"], 4)
        self.assertEqual(len(planner.contexts), 3)
        self.assertEqual(
            [row["status"] for row in result["iterations"]],
            ["planner_refused"] * 3,
        )
        self.assertEqual(
            [row["portfolio_hypothesis_id"] for row in result["iterations"]],
            [self.q5_hypothesis_id] * 3,
        )
        self.assertTrue(all(
            "source_manifest_sha256" not in row
            for row in result["iterations"]))
        self.assertEqual(
            [context["turn"] for context in planner.contexts], [1, 2, 3])
        self.assertTrue(all(
            context["authoring_assignment"]["portfolio_binding"]["hypothesis_id"]
            == self.q5_hypothesis_id for context in planner.contexts))
        self.assertEqual(planner.contexts[0]["prior_authoring_refusals"], [])
        self.assertEqual(
            [row["turn"] for row in
             planner.contexts[2]["prior_authoring_refusals"]], [1, 2])
        self.assertEqual((critic.calls, lease.calls, screen.calls), (0, 0, 0))
        self.assertNotIn("planning", result)
        self.assertNotIn("pending", result)
        self.assertNotIn("inflight", result)
        stop_states = [event.payload["state"] for event in events
                       if event.kind == D.journal.KIND_STOP_STATE]
        self.assertEqual(stop_states.count("discovery_planner_refused"), 3)

    def test_restart_after_refusal_starts_next_turn_not_old_turn(self):
        refusal = getattr(D, "PlannerOutputRefusal", D.DiscoveryControllerError)
        first = RefusingPlanner(lambda: refusal("malformed Q5 hunk"))
        critic, lease, screen = CountingCritic(), NeverLease(), NeverScreen()
        with tempfile.TemporaryDirectory() as temp:
            config, _package = self.config(Path(temp), max_iterations=2)
            real_save = D.DurableState.save

            def stop_after_refusal(store, state, phase):
                real_save(store, state, phase)
                if phase == "planner_refused":
                    raise SimulatedProcessExit("operator stopped after refusal")

            with patch.object(D.DurableState, "save", new=stop_after_refusal):
                with self.assertRaises(SimulatedProcessExit):
                    D.run_controller(
                        config, planner=first, critic=critic,
                        screener=screen, lease=lease)
            checkpoint = D.DurableState(config.output_root).load()
            self.assertEqual(checkpoint["next"], 2)
            self.assertEqual(
                [row["status"] for row in checkpoint["iterations"]],
                ["planner_refused"],
            )

            second = RefusingPlanner(lambda: refusal("second malformed Q5 hunk"))
            result = D.run_controller(
                config, planner=second, critic=critic,
                screener=screen, lease=lease)

        self.assertEqual([context["turn"] for context in first.contexts], [1])
        self.assertEqual([context["turn"] for context in second.contexts], [2])
        self.assertEqual(
            second.contexts[0]["prior_authoring_refusals"][0]["turn"], 1)
        self.assertEqual(
            [row["status"] for row in result["iterations"]],
            ["planner_refused", "planner_refused"],
        )
        self.assertEqual((critic.calls, lease.calls, screen.calls), (0, 0, 0))

    def test_generic_planner_error_is_not_swallowed(self):
        planner = RefusingPlanner(lambda: GenericPlannerFailure("runtime broke"))
        with tempfile.TemporaryDirectory() as temp:
            config, _package = self.config(Path(temp))
            with self.assertRaisesRegex(GenericPlannerFailure, "runtime broke"):
                D.run_controller(
                    config, planner=planner, critic=CountingCritic(),
                    screener=NeverScreen(), lease=NeverLease())
            state = D.DurableState(config.output_root).load()
        self.assertFalse(any(row.get("status") == "planner_refused"
                             for row in state["iterations"]))

    def test_source_authority_error_remains_raw_and_terminal(self):
        planner = RefusingPlanner(lambda: D.source_candidate.SourceCandidateError(
            "manifest campaign/proposal/candidate identity does not match controller authority"))
        with tempfile.TemporaryDirectory() as temp:
            config, _package = self.config(Path(temp))
            with self.assertRaisesRegex(
                    D.source_candidate.SourceCandidateError,
                    "does not match controller authority"):
                D.run_controller(
                    config, planner=planner, critic=CountingCritic(),
                    screener=NeverScreen(), lease=NeverLease())
            state = D.DurableState(config.output_root).load()
        self.assertFalse(any(row.get("status") == "planner_refused"
                             for row in state["iterations"]))

    def test_actor_rc0_checkpoint_resumes_validation_without_sol_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, package = self.config(root)
            planner = self.planner(root, package)
            actor_calls = []

            def actor(**kwargs):
                actor_calls.append(kwargs)
                return self.write_actor_output(kwargs)

            with patch.object(
                    D.codex_container_actor, "runtime_identity",
                    return_value=RUNTIME), patch.object(
                    D.codex_container_actor, "run_actor", side_effect=actor), \
                    patch.object(D, "_load_plan",
                                 side_effect=SimulatedProcessExit(
                                     "after actor rc0 before validation")):
                with self.assertRaises(SimulatedProcessExit):
                    D.run_controller(
                        config, planner=planner, critic=CountingCritic(),
                        screener=NeverScreen(), lease=NeverLease())

            state = D.DurableState(config.output_root).load()
            self.assertEqual(state["planning"]["phase"], "actor_entering")
            workspace = Path(state["planning"]["workspace"])
            actor_checkpoint = workspace.parent / "actor-result.json"
            self.assertTrue(actor_checkpoint.is_file())
            self.assertEqual(
                json.loads(actor_checkpoint.read_text())["schema"],
                "epyc.autokernel.planner_actor_checkpoint.v1",
            )

            critic = CountingCritic("reject")
            with patch.object(
                    D.codex_container_actor, "runtime_identity",
                    return_value=RUNTIME), patch.object(
                    D.codex_container_actor, "run_actor",
                    side_effect=AssertionError("Sol replayed after rc0 checkpoint")):
                result = D.run_controller(
                    config, planner=planner, critic=critic,
                    screener=NeverScreen(), lease=NeverLease())

        self.assertEqual(len(actor_calls), 1)
        self.assertEqual(critic.calls, 1)
        self.assertEqual(result["iterations"][0]["status"], "critic_reject")

    def test_valid_plan_checkpoint_skips_sol_on_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, package = self.config(root)
            planner = self.planner(root, package)
            actor_calls = []

            def actor(**kwargs):
                actor_calls.append(kwargs)
                return self.write_actor_output(kwargs)

            interrupting_critic = CountingCritic(interrupt=True)
            with patch.object(
                    D.codex_container_actor, "runtime_identity",
                    return_value=RUNTIME), patch.object(
                    D.codex_container_actor, "run_actor", side_effect=actor):
                with self.assertRaises(SimulatedProcessExit):
                    D.run_controller(
                        config, planner=planner, critic=interrupting_critic,
                        screener=NeverScreen(), lease=NeverLease())

            checkpoint = D.DurableState(config.output_root).load()
            self.assertEqual(checkpoint["pending"]["phase"], "critic_pending")
            self.assertNotIn("planning", checkpoint)

            critic = CountingCritic("reject")
            with patch.object(
                    D.codex_container_actor, "runtime_identity",
                    return_value=RUNTIME), patch.object(
                    D.codex_container_actor, "run_actor",
                    side_effect=AssertionError("Sol replayed after valid-plan checkpoint")):
                result = D.run_controller(
                    config, planner=planner, critic=critic,
                    screener=NeverScreen(), lease=NeverLease())

        self.assertEqual(len(actor_calls), 1)
        self.assertEqual((interrupting_critic.calls, critic.calls), (1, 1))
        self.assertEqual(result["iterations"][0]["status"], "critic_reject")

    def test_accepted_critic_checkpoint_skips_both_actors_on_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, package = self.config(root)
            planner = self.planner(root, package)
            actor_calls = []

            def actor(**kwargs):
                actor_calls.append(kwargs)
                return self.write_actor_output(kwargs)

            accepting_critic = CountingCritic("accept")
            with patch.object(
                    D.codex_container_actor, "runtime_identity",
                    return_value=RUNTIME), patch.object(
                    D.codex_container_actor, "run_actor", side_effect=actor), \
                    patch.object(D, "_ensure_question",
                                 side_effect=SimulatedProcessExit(
                                     "after accepted critic checkpoint")):
                with self.assertRaises(SimulatedProcessExit):
                    D.run_controller(
                        config, planner=planner, critic=accepting_critic,
                        screener=NeverScreen(), lease=NeverLease())

            checkpoint = D.DurableState(config.output_root).load()
            self.assertEqual(checkpoint["pending"]["phase"], "critic_complete")
            self.assertEqual(
                checkpoint["pending"]["row"]["critic"]["decision"], "accept")

            critic = CountingCritic("accept")
            with patch.object(
                    D.codex_container_actor, "runtime_identity",
                    return_value=RUNTIME), patch.object(
                    D.codex_container_actor, "run_actor",
                    side_effect=AssertionError(
                        "Sol replayed after accepted-critic checkpoint")):
                result = D.run_controller(
                    config, planner=planner, critic=critic,
                    screener=NeverScreen(), lease=NeverLease())

        self.assertEqual(len(actor_calls), 1)
        self.assertEqual(accepting_critic.calls, 1)
        self.assertEqual(critic.calls, 0)
        self.assertEqual(result["iterations"][0]["status"], "dry_run_authorized")


if __name__ == "__main__":
    unittest.main()
