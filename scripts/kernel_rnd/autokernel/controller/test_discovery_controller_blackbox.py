"""No-hardware process-boundary acceptance tests for autonomous discovery.

These tests intentionally exercise only public controller seams with fake compute.
They are a launch gate: a red test means the live adapter must remain disabled.
"""

from __future__ import annotations

import json
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.benchmark import run_autokernel_gpu_discovery as gpu_reward
from scripts.kernel_rnd.autokernel.controller import discovery_controller as D
from scripts.kernel_rnd.autokernel.controller import gpu_source_adapter as A
from scripts.kernel_rnd.autokernel.controller import gpu_source_evidence as E


H = "a" * 64
RUNTIME={"kind":"docker_workspace_bind_only","docker_path":"/docker","docker_sha256":H,"image_id":"image","codex_native_sha256":H,"code_mode_host_sha256":H,"ca_certificate_sha256":H,"writable_host_binds":["/workspace"],"host_network_mode":"docker_bridge"}
CLAUDE_RUNTIME={"kind":"claude_cli_structured_critic","provider":"claude","model":"claude-fable-5","effort":"high","wrapper_path":"/sealed/claude","wrapper_sha256":H,"argv_policy_sha256":H,"auth_staging_policy":"ephemeral_0600_copy_atomic_oauth_rotation_sync_no_secret_receipt"}


class Manifest:
    def __init__(self, **values):
        defaults = {
            "campaign_id": "ak-blackbox",
            "proposal_id": "akp-blackbox",
            "candidate_id": "akc-blackbox",
            "source_tree": "llama.cpp",
            "production_base_commit": "0" * 40,
            "instrument_commit": "1" * 40,
            "change_class": "source",
            "declared_files": ("ggml/src/ggml.c",),
            "declared_symbols": {"ggml/src/ggml.c": ("<file-scope>",)},
            "mechanism_id": "blackbox",
            "patch_bytes": b"diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c\n@@ -1 +1 @@\n-x\n+y\n",
        }
        defaults["patch_sha256"] = "0" * 64
        defaults.update(values)
        for key, value in defaults.items():
            setattr(self, key, value)

    @property
    def patch_bundle_sha256(self):
        return H


class Planner:
    def __init__(self):
        self.calls = 0

    def attest(self):
        return {**D.SOL, "runtime": RUNTIME}

    def plan(self, *, context, workspace):
        self.calls += 1
        return D.PlannedCandidate(
            "akh-blackbox",
            "bounded source hypothesis",
            "no throughput improvement",
            {"backend": "gpu", "phase": "decode"},
            {"proposal_id": "akp-blackbox"},
            Manifest(),
            H,
        )


class Critic:
    def __init__(self):
        self.calls = 0

    def attest(self):
        return {**D.FABLE5_CRITIC, "runtime": CLAUDE_RUNTIME}

    def review(self, candidate, *, context, workspace):
        self.calls += 1
        return D.Critique("accept", "bounded")


class Lease:
    def __init__(self, decisions=(True,)):
        self.decisions = iter(decisions)

    def admit(self, candidate, *, operation_key):
        admitted = next(self.decisions)
        return {"admitted": admitted, "mode": "allowed_discovery_noise",
                "operation_key": operation_key}

    def resume(self, candidate, permit):
        return self.admit(candidate, operation_key=permit["operation_key"])


class Screen:
    def __init__(self, effect=0.04):
        self.calls = 0
        self.effect = effect
        self.items = []

    def screen(self, item, authorization, lease):
        self.calls += 1
        self.items.append(item)
        return D.SealedScreen(
            "result.json", H, self.effect, "candidate", H, H, H
        )
    def reconcile(self, inflight):
        return D.Recovery("safe_to_start")

    def reconcile(self, inflight):
        return D.Recovery("safe_to_start")


class SeriesScreen(Screen):
    def __init__(self, effects):
        super().__init__()
        self.effects = iter(effects)

    def screen(self, item, authorization, lease):
        self.calls += 1
        self.items.append(item)
        return D.SealedScreen(
            "result.json", chr(96 + self.calls) * 64, next(self.effects),
            "candidate", H, H, H,
        )


class ProcessCrash(BaseException):
    pass


class OrdinaryAfterStart(RuntimeError):
    pass


class CrashBeforeRunner(Screen):
    def __init__(self):
        super().__init__()
        self.entries = 0

    def screen(self, item, authorization, lease):
        self.entries += 1
        if self.entries == 1:
            raise ProcessCrash("before fake runner")
        return super().screen(item, authorization, lease)


class CrashThenPrecomputeRefusal(Screen):
    def __init__(self):
        super().__init__()
        self.entries = 0

    def screen(self, item, authorization, lease):
        self.entries += 1
        if self.entries == 1:
            raise ProcessCrash("before fake runner")
        raise D.PrecomputeScreenRefusal(
            "source candidate authoring rejected on safe restart")


class CleanupFailureOverridesRefusal(Screen):
    def screen(self, item, authorization, lease):
        try:
            raise D.PrecomputeScreenRefusal("authoring rejected")
        finally:
            raise RuntimeError("cleanup durability failed")

    def reconcile(self, inflight):
        return D.Recovery("ambiguous")


class ExceptionAfterStart(Screen):
    def __init__(self):
        super().__init__()
        self.entries = 0

    def screen(self, item, authorization, lease):
        self.entries += 1
        if self.entries == 1:
            raise OrdinaryAfterStart("adapter lost result after operation start")
        return super().screen(item, authorization, lease)


class CrashAfterRunner(Screen):
    def __init__(self, durable_result: Path):
        super().__init__()
        self.durable_result = durable_result
        self.compute_calls = 0

    def screen(self, item, authorization, lease):
        if self.durable_result.exists():
            return super().screen(item, authorization, lease)
        self.compute_calls += 1
        self.durable_result.write_text(json.dumps({"result_sha256": H}))
        raise ProcessCrash("after fake runner")
    def reconcile(self, inflight):
        if self.durable_result.exists():
            return D.Recovery("sealed_result", D.SealedScreen("result.json",H,self.effect,"candidate",H,H,H))
        return D.Recovery("safe_to_start")

    def reconcile(self, inflight):
        if self.durable_result.exists():
            return D.Recovery(
                "sealed_result",
                D.SealedScreen(
                    "result.json", H, self.effect, "candidate", H, H, H
                ),
            )
        return D.Recovery("safe_to_start")


class AmbiguousRecovery(Screen):
    def screen(self, item, authorization, lease):
        self.calls += 1
        if self.calls == 1:
            raise ProcessCrash("unknown whether fake runner started")
        return super().screen(item, authorization, lease)

    def reconcile(self, inflight):
        return None


class RecoveredResult(Screen):
    def reconcile(self, inflight):
        return D.Recovery(
            "sealed_result",
            D.SealedScreen(
                "result.json", H, self.effect, "candidate", H, H, H
            ),
        )


class PrecomputeRefusal(Screen):
    def screen(self, item, authorization, lease):
        self.calls += 1
        raise D.PrecomputeScreenRefusal("bounded follow-on refusal")


class BlackBoxLaunchGate(unittest.TestCase):
    def config(self, root: Path):
        return D.ControllerConfig(root / "out", max_iterations=1)

    def run_twice(self, root, planner, critic, screener, lease):
        first = D.run_controller(
            self.config(root), planner=planner, critic=critic,
            screener=screener, lease=lease,
        )
        second = D.run_controller(
            self.config(root), planner=planner, critic=critic,
            screener=screener, lease=lease,
        )
        return first, second

    def test_launch_gate_refuses_semantically_encoded_phase_switch(self):
        """Opaque source state may pass static review but cannot pass timed output.

        The live path runs correctness in ``test-backend-ops`` and later runs
        attribution/reward in fresh ``llama-bench`` processes.  Fresh process
        state does not neutralize a call-sequence switch: each executable has a
        distinct, repeatable call sequence.  The deliberately opaque counter
        below avoids every lexical phase/call/timing token.  The candidate
        returns the same sealed 81bf 64-bit output hashes but shifts all work
        into the first member; serialized pair-max scoring proves that the
        fast second member earns no gain.  This is authority over that sealed
        instrument hash contract, not a new cryptographic collision claim.
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            relative = "ggml/src/ggml-cuda/vecdotq.cuh"
            symbol = "vec_dot_q5_0_q8_1_impl"
            patch_bytes = (
                f"diff --git a/{relative} b/{relative}\n"
                f"--- a/{relative}\n+++ b/{relative}\n"
                f"@@ -1,1 +1,3 @@ {symbol}\n"
                "-old\n"
                "+static unsigned x = 0;\n"
                "+if (++x > 10) return;\n"
                "+new\n"
            ).encode()
            assignment = D.AuthoringAssignment(
                "ak-blackbox", "akp-blackbox", "akc-blackbox", "0" * 40, "1" * 40)
            manifest = {
                "schema": D.source_candidate.SCHEMA_SOURCE_PATCH,
                "campaign_id": assignment.campaign_id,
                "proposal_id": assignment.proposal_id,
                "candidate_id": assignment.candidate_id,
                "source_tree": "llama.cpp",
                "production_base_commit": assignment.production_base_commit,
                "instrument_commit": assignment.instrument_commit,
                "change_class": "arithmetic",
                "declared_files": [relative],
                "declared_symbols": {relative: [symbol]},
                "mechanism_id": "opaque-sequence-switch",
                "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
                "patch_encoding": "base64",
                "patch_base64": base64.b64encode(patch_bytes).decode(),
            }
            (root / "source-patch.json").write_text(json.dumps(manifest))
            plan = {
                "hypothesis_id": "akh-blackbox-phase-switch",
                "statement": "opaque sequence state changes timed behavior",
                "falsifier": "timed outputs remain oracle-equivalent",
                "regime": {"phase": "decode"},
                "proposal": {
                    "proposal_id": assignment.proposal_id,
                    "change_class": "arithmetic",
                    "change": {
                        "files_and_symbols": [f"{relative}:{symbol}"],
                        "estimated_diff_size": 4,
                    },
                },
                "source_manifest_path": "source-patch.json",
            }
            (root / "plan.json").write_text(json.dumps(plan))
            loaded = D._load_plan(root / "plan.json", root, assignment=assignment)
            self.assertEqual(loaded.source_manifest.mechanism_id,
                             "opaque-sequence-switch")

            def semantics(*, first_ns: list[int], second_ts: list[float]) -> dict:
                input_hashes = ["a000000000000001", "a000000000000002",
                                "a000000000000003"]
                output_hashes = ["b000000000000065", "b000000000000066",
                                 "b000000000000067"]
                row = {
                    "autokernel_hardened": True,
                    "autokernel_output_invariant": True,
                    "autokernel_hybrid_ab_complete": True,
                    "autokernel_thread_set_stable": True,
                    "autokernel_escape_checks_complete": True,
                    "autokernel_input_working_set_bytes": 12288,
                    "autokernel_device_sync_mode": "hip_full_device",
                    "autokernel_input_hashes": ",".join(input_hashes),
                    "autokernel_output_hashes": ",".join(
                        f"{value}/{value}" for value in output_hashes),
                    "autokernel_input_addresses": "0x1/0x2,0x3/0x4,0x5/0x6",
                    "autokernel_context_addresses": "0x7/0x8,0x9/0xa,0xb/0xc",
                    "autokernel_unsynchronized_samples_ns": ",".join(
                        str(value) for value in first_ns),
                    "autokernel_thread_set_hashes": ",".join([(
                        "00000000000000aa/00000000000000aa/"
                        "00000000000000aa/00000000000000aa")] * 3),
                    "samples_ts": second_ts,
                }
                return gpu_reward._validate_timed_output_semantics(
                    row, repetitions=3, seed=8613, tokens_per_repetition=128,
                    serialization_env={
                        "AMD_SERIALIZE_KERNEL": "3", "AMD_SERIALIZE_COPY": "3",
                        "GGML_CUDA_DISABLE_GRAPHS": "1"})

            anchor_semantics = semantics(
                first_ns=[1000, 1000, 1000],
                second_ts=[128_000_000.0, 128_000_000.0, 128_000_000.0])
            switched_semantics = semantics(
                first_ns=[1000, 1000, 1000],
                second_ts=[128_000_000_000.0] * 3)
            oracle = gpu_reward._validate_cross_arm_timed_outputs(
                {"timed_output_semantics": anchor_semantics},
                {"timed_output_semantics": switched_semantics})
            self.assertTrue(oracle["bitwise_equal"])
            self.assertEqual(switched_semantics["second_samples_ns"], [1, 1, 1])
            self.assertEqual(switched_semantics["protected_samples_ns"],
                             anchor_semantics["protected_samples_ns"])
            self.assertEqual(switched_semantics["protected_samples_ts"],
                             anchor_semantics["protected_samples_ts"])

    def test_pending_resume_reuses_exact_candidate_without_replanning(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic, screen = Planner(), Critic(), Screen()
            first, second = self.run_twice(
                root, planner, critic, screen, Lease((False, True))
            )
            self.assertIn("pending", first)
            self.assertTrue(second["complete"])
            self.assertEqual(planner.calls, 1)
            self.assertEqual(critic.calls, 1)
            self.assertEqual(screen.calls, 1)
            self.assertEqual(screen.items[0].source_manifest.patch_bytes,
                             Manifest().patch_bytes)

    def test_state_tamper_is_refused_before_pending_resume(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic = Planner(), Critic()
            D.run_controller(
                self.config(root), planner=planner, critic=critic,
                screener=Screen(), lease=Lease((False,)),
            )
            state_path = root / "out" / "state.json"
            state = json.loads(state_path.read_text())
            state["pending"]["candidate"]["proposal"]["proposal_id"] = "tampered"
            state_path.write_text(json.dumps(state))
            with self.assertRaises(D.DiscoveryControllerError):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=Screen(), lease=Lease((True,)),
                )

    def test_real_planner_manifest_identity_survives_pending_roundtrip(self):
        patch_bytes = (
            b"diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c\n"
            b"--- a/ggml/src/ggml.c\n+++ b/ggml/src/ggml.c\n"
            b"@@ -1 +1 @@\n-x\n+y\n"
        )
        manifest = {
            "schema": D.source_candidate.SCHEMA_SOURCE_PATCH,
            "campaign_id": "ak-blackbox",
            "proposal_id": "akp-blackbox",
            "candidate_id": "akc-blackbox",
            "source_tree": "llama.cpp",
            "production_base_commit": "0" * 40,
            "instrument_commit": "1" * 40,
            "change_class": "fusion",
            "declared_files": ["ggml/src/ggml.c"],
            "declared_symbols": {"ggml/src/ggml.c": ["<file-scope>"]},
            "mechanism_id": "blackbox",
            "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
            "patch_encoding": "base64",
            "patch_base64": base64.b64encode(patch_bytes).decode("ascii"),
        }
        plan = {
            "hypothesis_id": "akh-blackbox",
            "statement": "bounded source hypothesis",
            "falsifier": "no throughput improvement",
            "regime": {"backend": "gpu", "phase": "decode"},
            "proposal": {"proposal_id": "akp-blackbox"},
            "source_manifest_path": "source-patch.json",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source-patch.json").write_text(json.dumps(manifest))
            (root / "plan.json").write_text(json.dumps(plan))
            item = D._load_plan(root / "plan.json", root)
            restored = D._restore_pending({"candidate": D._pending_item(item)})
            self.assertEqual(restored.source_manifest_sha256,
                             item.source_manifest.patch_bundle_sha256)
            self.assertEqual(restored.source_manifest.patch_bytes, patch_bytes)

    def test_real_planner_multirow_dispatch_and_fallback_advice_load(self):
        """Regression for the first live Sol artifact rejected before Fable."""
        patch_bytes = (
            b"diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c\n"
            b"--- a/ggml/src/ggml.c\n+++ b/ggml/src/ggml.c\n"
            b"@@ -1 +1 @@ vec_dot_q5_0_q8_1\n-x\n+y\n"
        )
        assignment = D.AuthoringAssignment(
            "ak-inaugural", "akp-inaugural", "akc-inaugural", "0" * 40, "1" * 40)
        symbol = "vec_dot_q5_0_q8_1"
        symbols = {"ggml/src/ggml.c": [symbol]}
        manifest = {
            "schema": D.source_candidate.SCHEMA_SOURCE_PATCH,
            "campaign_id": assignment.campaign_id,
            "proposal_id": assignment.proposal_id,
            "candidate_id": assignment.candidate_id,
            "source_tree": "llama.cpp",
            "production_base_commit": assignment.production_base_commit,
            "instrument_commit": assignment.instrument_commit,
            "change_class": "fusion",
            "declared_files": ["ggml/src/ggml.c"],
            "declared_symbols": symbols,
            "mechanism_id": "inaugural-live-artifact",
            "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
            "patch_encoding": "base64",
            "patch_base64": base64.b64encode(patch_bytes).decode("ascii"),
        }
        kernel = "void mul_mat_vec_q<(ggml_type)6>(void const*) [clone .kd]"
        rows = [
            {"route_id": f"cuda-vecdotq-v1.anchor.{index}",
             "kernel_name": kernel, "calls": calls, "grid": grid,
             "workgroup": 128, "lds_bytes": 1024}
            for index, (calls, grid) in enumerate(
                ((6063, 57344), (4644, 8192), (3096, 311296)))
        ]
        plan = {
            "hypothesis_id": "akh-v2-q5-type-specific-dequant",
            "statement": "bounded Q5 dequant hypothesis",
            "falsifier": "the exact type-6 kernel does not improve",
            "regime": {"backend": "hip", "phase": "decode"},
            "proposal": {"proposal_id": assignment.proposal_id,
                         "change_class": "fusion",
                         "change": {"files_and_symbols": [
                                        f"ggml/src/ggml.c:{symbol}"],
                                    "estimated_diff_size": 2}},
            "source_manifest_path": "source-patch.json",
            "experiment_intent": {
                "template_id": "cuda-vecdotq-v1",
                "target_surface": "gpu_decode",
                "target_symbol": "vec_dot_q5_0_q8_1",
                "correctness_id": "backend-ops-hip-v1",
                "dispatch_id": "decode-tg128-rocprof-v1",
                "expected_dispatch": rows,
                "load_mode_recommendation": {
                    "mode": "cold_serialized",
                    "rationale": "No reviewed overlap authority exists.",
                    "example_ids": [],
                },
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source-patch.json").write_text(json.dumps(manifest))
            (root / "plan.json").write_text(json.dumps(plan))
            item = D._load_plan(root / "plan.json", root, assignment=assignment)
        self.assertEqual(
            [row.route_id for row in item.experiment_intent.expected_dispatch],
            [row["route_id"] for row in rows])
        self.assertEqual(item.experiment_intent.load_mode_recommendation.example_ids, ())

    def test_actor_cannot_invent_assigned_campaign_or_base_identity(self):
        patch_bytes = (
            b"diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c\n"
            b"--- a/ggml/src/ggml.c\n+++ b/ggml/src/ggml.c\n@@ -1 +1 @@\n-x\n+y\n")
        assignment = D.AuthoringAssignment("ak-assigned", "akp-assigned", "akc-assigned",
                                           "0" * 40, "1" * 40)
        manifest = {"schema": D.source_candidate.SCHEMA_SOURCE_PATCH,
                    "campaign_id": assignment.campaign_id, "proposal_id": assignment.proposal_id,
                    "candidate_id": assignment.candidate_id, "source_tree": "llama.cpp",
                    "production_base_commit": assignment.production_base_commit,
                    "instrument_commit": assignment.instrument_commit, "change_class": "fusion",
                    "declared_files": ["ggml/src/ggml.c"],
                    "declared_symbols": {"ggml/src/ggml.c": ["<file-scope>"]},
                    "mechanism_id": "bounded", "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
                    "patch_encoding": "base64", "patch_base64": base64.b64encode(patch_bytes).decode("ascii")}
        plan = {"hypothesis_id": "akh-assigned", "statement": "s", "falsifier": "f",
                "regime": {}, "proposal": {"proposal_id": assignment.proposal_id,
                    "change_class": "fusion", "change": {
                        "files_and_symbols": ["ggml/src/ggml.c:<file-scope>"],
                        "estimated_diff_size": 2}},
                "source_manifest_path": "source-patch.json"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "source-patch.json").write_text(json.dumps(manifest)); (root / "plan.json").write_text(json.dumps(plan))
            self.assertEqual(D._load_plan(root / "plan.json", root, assignment=assignment).source_manifest.candidate_id, assignment.candidate_id)
            manifest["candidate_id"] = "akc-invented"; (root / "source-patch.json").write_text(json.dumps(manifest))
            with self.assertRaises(D.DiscoveryControllerError):
                D._load_plan(root / "plan.json", root, assignment=assignment)

    def test_process_crash_before_runner_resumes_without_replanning(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic, screen = Planner(), Critic(), CrashBeforeRunner()
            with self.assertRaises(ProcessCrash):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=screen, lease=Lease((True,)),
                )
            completed = D.run_controller(
                self.config(root), planner=planner, critic=critic,
                screener=screen, lease=Lease((True,)),
            )
            self.assertTrue(completed["complete"])
            self.assertEqual((planner.calls, critic.calls, screen.calls), (1, 1, 1))

    def test_safe_restart_precompute_refusal_is_durable_without_replanning(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic, screen = Planner(), Critic(), CrashThenPrecomputeRefusal()
            with self.assertRaises(ProcessCrash):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=screen, lease=Lease((True,)))
            crashed = json.loads((root / "out" / "state.json").read_text())
            self.assertEqual(crashed["next"], 1)
            self.assertIn("inflight", crashed)

            completed = D.run_controller(
                self.config(root), planner=planner, critic=critic,
                screener=screen, lease=Lease((True,)))

            self.assertTrue(completed["complete"])
            self.assertEqual(
                json.loads((root / "out" / "state.json").read_text()), completed)
            self.assertEqual((planner.calls, critic.calls, screen.entries), (1, 1, 2))
            self.assertEqual(len(completed["iterations"]), 1)
            self.assertEqual(completed["iterations"][0]["status"], "screen_refused")
            self.assertIn("safe restart", completed["iterations"][0]["reason"])
            self.assertNotIn("inflight", completed)
            self.assertNotIn("pending", completed)
            again = D.run_controller(
                self.config(root), planner=planner, critic=critic,
                screener=screen, lease=Lease((True,)))
            self.assertEqual(again, completed)
            self.assertEqual((planner.calls, critic.calls, screen.entries), (1, 1, 2))

    def test_cleanup_failure_overrides_refusal_and_keeps_ambiguous_inflight(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic = Planner(), Critic()
            screen = CleanupFailureOverridesRefusal()
            with self.assertRaisesRegex(RuntimeError, "cleanup durability failed"):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=screen, lease=Lease((True,)))
            state = json.loads((root / "out" / "state.json").read_text())
            self.assertEqual(state["next"], 1)
            self.assertEqual(state["iterations"], [])
            self.assertEqual(
                state["inflight"]["exception"],
                {"type": "RuntimeError", "message": "cleanup durability failed"})
            with self.assertRaisesRegex(
                    D.DiscoveryControllerError, "cannot be safely reconciled"):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=screen, lease=Lease((True,)))
            self.assertEqual((planner.calls, critic.calls), (1, 1))

    def test_recovery_busy_demotes_exact_inflight_to_pending_without_replanning(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic, screen = Planner(), Critic(), CrashBeforeRunner()
            with self.assertRaises(ProcessCrash):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=screen, lease=Lease((True,)))
            waiting = D.run_controller(
                self.config(root), planner=planner, critic=critic,
                screener=screen, lease=Lease((False,)))
            self.assertEqual(waiting["pending"]["row"]["status"], "waiting_resource")
            self.assertEqual(waiting["next"], 1)
            self.assertEqual((planner.calls, critic.calls, screen.entries), (1, 1, 1))
            completed = D.run_controller(
                self.config(root), planner=planner, critic=critic,
                screener=screen, lease=Lease((True,)))
            self.assertTrue(completed["complete"])
            self.assertEqual((planner.calls, critic.calls, screen.entries), (1, 1, 2))

    def test_ordinary_post_start_exception_requires_reconcile_before_retry(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic, screen = Planner(), Critic(), ExceptionAfterStart()
            with self.assertRaises(OrdinaryAfterStart):
                D.run_controller(self.config(root), planner=planner, critic=critic,
                                 screener=screen, lease=Lease((True,)))
            state = json.loads((root / "out" / "state.json").read_text())
            self.assertIn("inflight", state)
            self.assertEqual(state["inflight"]["exception"]["type"], "OrdinaryAfterStart")
            resumed = D.run_controller(self.config(root), planner=planner, critic=critic,
                                       screener=screen, lease=Lease((True,)))
            self.assertTrue(resumed["complete"])
            self.assertEqual((planner.calls, critic.calls, screen.calls), (1, 1, 1))

    def test_source_authoring_rejection_is_nonfatal_and_never_starts_gpu_work(self):
        """A post-critic committed-diff rejection is a failed iteration.

        This reproduces the live ``SourceCandidateError`` after the controller
        has written ``pre_screen_intent``.  The governed adapter must unwind
        its reservation seam, expose a typed precompute refusal, and let the
        controller plan the next iteration without proof or runner activity.
        """
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"), \
                patch.object(A, "_protected_snapshot", return_value={"sealed": True}), \
                patch.object(D.gpu_discovery, "run",
                             side_effect=AssertionError("GPU runner must not start")) as runner:
            root = Path(temp)
            protected = root / "protected"
            protected.mkdir()
            artifact = protected / "frozen.bin"
            artifact.write_bytes(b"frozen")
            bound = E.BoundInputFile(
                "production_artifact", artifact.resolve(),
                hashlib.sha256(artifact.read_bytes()).hexdigest())

            class Reservation:
                def __init__(self):
                    self.reserve_calls = 0
                    self.release_calls = 0
                    self.active = set()

                def reserve(self, operation_key):
                    self.reserve_calls += 1
                    self.active.add(operation_key)
                    raise AssertionError("GPU reservation must not start")

                def release(self, operation_key):
                    self.release_calls += 1
                    self.active.discard(operation_key)
                    return None

                def borrower(self, operation_key):
                    raise AssertionError("GPU borrower must not be constructed")

            reservation = Reservation()
            proof_calls = []

            def reject_after_materialization(*_args):
                raise D.source_candidate.SourceCandidateError(
                    "committed diff in 'ggml/src/ggml-cuda/vecdotq.cuh' "
                    "derives undeclared symbols ['<file-scope>']")

            def no_proof(*_args):
                proof_calls.append("proof")
                raise AssertionError("GPU proof must not start")

            governed = A.GovernedGpuSourceAdapter(
                operations_root=(root / "operations").resolve(),
                build_source=reject_after_materialization,
                plan_factory=no_proof,
                args_factory=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("runner arguments must not be built")),
                correctness_executor=no_proof, rocprof_executor=no_proof,
                claim_journal=object(), claim_acquirer=no_proof,
                claim_verifier=lambda _receipt: True, claim_timeout_s=0.0,
                reservation_manager=reservation,
                receipt_series=lambda *_args: (),
                protected_roots=(protected,), protected_files=(bound,))

            class RejectOnceThenRefuse:
                def __init__(self):
                    self.calls = 0
                    self.follow_on = PrecomputeRefusal()

                def screen(self, item, authorization, lease):
                    self.calls += 1
                    if self.calls == 1:
                        return governed.screen(item, authorization, lease)
                    return self.follow_on.screen(item, authorization, lease)

            class ReleasedAdmissionProbe(Lease):
                def __init__(self):
                    super().__init__((True, True))
                    self.probes = 0
                    self.held = 0

                def admit(self, candidate, *, operation_key):
                    self.probes += 1
                    self.held += 1
                    try:
                        return super().admit(candidate, operation_key=operation_key)
                    finally:
                        self.held -= 1

            planner, critic, screen = Planner(), Critic(), RejectOnceThenRefuse()
            admission = ReleasedAdmissionProbe()
            result = D.run_controller(
                D.ControllerConfig(root / "out", max_iterations=2),
                planner=planner, critic=critic, screener=screen,
                lease=admission)

            self.assertTrue(result["complete"])
            self.assertEqual(
                json.loads((root / "out" / "state.json").read_text()), result)
            self.assertEqual((planner.calls, critic.calls, screen.calls), (2, 2, 2))
            self.assertEqual(
                [row["status"] for row in result["iterations"]],
                ["screen_refused", "screen_refused"])
            self.assertIn("SourceCandidateError", result["iterations"][0]["reason"])
            self.assertNotIn("inflight", result)
            self.assertNotIn("pending", result)
            self.assertEqual(proof_calls, [])
            runner.assert_not_called()
            self.assertEqual(reservation.reserve_calls, 0)
            self.assertEqual(reservation.release_calls, 1)
            self.assertEqual(reservation.active, set())
            self.assertEqual((admission.probes, admission.held), (2, 0))
            operation_roots = tuple((root / "operations").iterdir())
            self.assertEqual(len(operation_roots), 1)
            self.assertEqual(
                {path.name for path in operation_roots[0].iterdir()}, {"intent.json"})

    def test_process_crash_after_runner_does_not_repeat_compute(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic = Planner(), Critic()
            screen = CrashAfterRunner(root / "fake-result.json")
            with self.assertRaises(ProcessCrash):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=screen, lease=Lease((True,)),
                )
            completed = D.run_controller(
                self.config(root), planner=planner, critic=critic,
                screener=screen, lease=Lease((True,)),
            )
            self.assertTrue(completed["complete"])
            self.assertEqual(screen.compute_calls, 1)
            self.assertEqual(planner.calls, 1)
            queue = root / "out" / "promotion-queue.jsonl"
            self.assertFalse(queue.exists(), "one positive screen is not a nomination")

    def test_ambiguous_inflight_recovery_refuses_duplicate_compute(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic, screen = Planner(), Critic(), AmbiguousRecovery()
            with self.assertRaises(ProcessCrash):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=screen, lease=Lease((True,)),
                )
            with self.assertRaises(D.DiscoveryControllerError):
                D.run_controller(
                    self.config(root), planner=planner, critic=critic,
                    screener=screen, lease=Lease((True,)),
                )
            self.assertEqual(screen.calls, 1)

    def test_post_result_crash_records_dnr_attempt_exactly_once(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic, screen = Planner(), Critic(), RecoveredResult()
            with patch.object(D, "_append_nomination", side_effect=ProcessCrash):
                with self.assertRaises(ProcessCrash):
                    D.run_controller(
                        self.config(root), planner=planner, critic=critic,
                        screener=screen, lease=Lease((True,)),
                    )
            completed = D.run_controller(
                self.config(root), planner=planner, critic=critic,
                screener=screen, lease=Lease((True,)),
            )
            self.assertTrue(completed["complete"])
            tracked = D._tracker(D.DurableState(root / "out")).get("akh-blackbox")
            self.assertEqual(len(tracked.attempts), 1)

    def test_test_only_runtime_attestation_is_not_accepted_in_live_code(self):
        with self.assertRaises(D.DiscoveryControllerError):
            D._require_runtime({"wrapper_sha256": H})

    def test_controller_run_lock_refuses_a_second_owner_before_planning(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = D.DurableState(root / "out")
            held = store.run_lock()
            planner, critic = Planner(), Critic()
            try:
                with self.assertRaises(D.DiscoveryControllerError):
                    D.run_controller(
                        self.config(root), planner=planner, critic=critic,
                        screener=Screen(), lease=Lease((True,)),
                    )
            finally:
                held.close()
            self.assertEqual(planner.calls, 0)
            self.assertEqual(critic.calls, 0)

    def test_pooled_classifier_never_combines_distinct_source_manifests(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            first = D.PlannedCandidate(
                "akh-shared-question", "one question with two different patches",
                "no throughput improvement", {"backend": "gpu", "phase": "decode"},
                {"proposal_id": "akp-patch-1"}, Manifest(proposal_id="akp-patch-1"), "a" * 64,
            )
            second = D.PlannedCandidate(
                "akh-shared-question", "one question with two different patches",
                "no throughput improvement", {"backend": "gpu", "phase": "decode"},
                {"proposal_id": "akp-patch-2"}, Manifest(proposal_id="akp-patch-2"), "b" * 64,
            )
            initial = D.SealedScreen("result-a.json", "a" * 64, 0.01, "candidate", H, H, H)
            prior = D._classified_result({"iterations": []}, first, initial)
            next_result = D._classified_result(
                {"iterations": [{"series_key": prior.series_key, "effect_fraction": 0.01}]},
                second, D.SealedScreen("result-b.json", "b" * 64, 0.02, "candidate", H, H, H),
            )
            self.assertEqual(
                next_result.classification, "candidate",
            )

    def test_positive_candidate_schedules_one_s2_without_replanning(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(D.source_candidate, "SourcePatchManifest", Manifest), \
                patch.object(D, "_write_projection"):
            root = Path(temp)
            planner, critic = Planner(), Critic()
            screen = SeriesScreen((0.04, 0.03))
            result = D.run_controller(
                D.ControllerConfig(root / "out", max_iterations=2),
                planner=planner, critic=critic, screener=screen,
                lease=Lease((True, True)),
            )
            self.assertEqual((planner.calls, critic.calls, screen.calls), (1, 1, 2))
            self.assertEqual(screen.items[0].source_manifest_sha256,
                             screen.items[1].source_manifest_sha256)
            self.assertEqual(
                [row["status"] for row in result["iterations"]],
                ["candidate", "top_k_replicated_candidate"],
            )
            tracked = D._tracker(D.DurableState(root / "out")).get("akh-blackbox")
            self.assertEqual(len(tracked.claim_authorizations), 2)
            queue = root / "out" / "promotion-queue.jsonl"
            self.assertEqual(len(queue.read_text().splitlines()), 1)

    def test_pooled_classifier_handles_sign_conflict_and_subadditive_stack(self):
        self.assertEqual(D.classify_screen_series([0.01]), "candidate")
        self.assertEqual(
            D.classify_screen_series([0.01, 0.02]),
            "top_k_replicated_candidate",
        )
        self.assertEqual(D.classify_screen_series([0.01, -0.01]), "inconclusive")
        self.assertEqual(
            D.classify_screen_series(
                [0.021687370691388094, 0.003313242012254847],
                component_pooled_effects=[0.013465889, 0.01012618],
            ),
            "replicated_but_subadditive",
        )
        for classification in (
                "top_k_replicated_candidate", "replicated_but_subadditive"):
            with self.subTest(classification=classification):
                D.SealedScreen(
                    "result.json", H, 0.01, classification, H, H, H
                )


if __name__ == "__main__":
    unittest.main()
