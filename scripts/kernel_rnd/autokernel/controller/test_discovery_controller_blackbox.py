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

from scripts.kernel_rnd.autokernel.controller import discovery_controller as D


H = "a" * 64
RUNTIME={"kind":"docker_workspace_bind_only","docker_path":"/docker","docker_sha256":H,"image_id":"image","codex_native_sha256":H,"code_mode_host_sha256":H,"ca_certificate_sha256":H,"writable_host_binds":["/workspace"],"host_network_mode":"docker_bridge"}


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
        return {**D.TERRA, "runtime": RUNTIME}

    def review(self, candidate, *, context, workspace):
        self.calls += 1
        return D.Critique("accept", "bounded")


class Lease:
    def __init__(self, decisions=(True,)):
        self.decisions = iter(decisions)

    def admit(self, candidate):
        admitted = next(self.decisions)
        return {"admitted": admitted, "mode": "allowed_discovery_noise"}


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
                "regime": {}, "proposal": {"proposal_id": assignment.proposal_id},
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
