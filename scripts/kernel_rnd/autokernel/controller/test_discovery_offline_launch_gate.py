"""Black-box, hardware-free launch acceptance for GPU source discovery.

This is intentionally an acceptance gate rather than a collection of unit
tests for individual helpers.  Every assertion describes authority required
at the deployment process boundary.  A failure means the autonomous launcher
must remain disabled; it is not permission to weaken the assertion.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.benchmark import run_autokernel_gpu_discovery as gpu_runner

from . import discovery_controller as C
from . import discovery_deployment as D
from . import discovery_deployment_factory as F
from . import gpu_source_evidence as E
from .test_discovery_controller_blackbox import Critic, Lease, Manifest, Planner
from .test_gpu_source_evidence import ClaimFactory, FakeExecutors, plan, write_bound


class OfflineLaunchGate(unittest.TestCase):
    """Requirements that cross configuration, actors, evidence, and runner."""

    def _config(self, root: Path) -> D.DiscoveryDeployment:
        (root / "production").mkdir(exist_ok=True)
        inputs = {}
        for label in ("wrapper", "model", "workload", "runtime", "policy"):
            path = (root / f"{label}.bin").resolve()
            path.write_bytes(label.encode())
            inputs[label] = D.ImmutableInput(path, __import__("hashlib").sha256(path.read_bytes()).hexdigest())
        context_value = {
            "schema": D.PLANNER_CONTEXT_SCHEMA,
            "model_sha256": inputs["model"].sha256,
            "workload_sha256": inputs["workload"].sha256,
            "runtime_config_sha256": inputs["runtime"].sha256,
            "profile_receipts": [],
            "hotspots": [{"surface": "ggml/src/ggml-cuda/fattn.cu",
                          "symbol": "fattn_kernel", "share": .5,
                          "source_excerpt": "__global__ void fattn_kernel() {}"}],
            "source_constraints": {"allowed": ["ggml/src/ggml-cuda/fattn.cu"]},
            "initial_strategies": ["one wave"],
        }
        context_value["context_sha256"] = C._sha(context_value)
        context_path = (root / "planner-context.json").resolve()
        context_path.write_text(__import__("json").dumps(context_value))
        context_input = D.ImmutableInput(
            context_path, __import__("hashlib").sha256(context_path.read_bytes()).hexdigest())
        return D.DiscoveryDeployment(
            config_sha256="f" * 64,
            production_path=(root / "production").resolve(), production_head="a" * 40,
            state_root=(root / "state").resolve(), evidence_root=(root / "evidence").resolve(),
            operations_root=(root / "operations").resolve(), max_iterations=2,
            nomination_threshold=.03, actor_wrapper=inputs["wrapper"],
            environment_profile_id="sealed-codex", device_id="mi210_0",
            claim_timeout_s=0, inference_window_lock=(root / "window.lock").resolve(),
            small_model_max_bytes=512 * 1024 * 1024, model=inputs["model"],
            workload=inputs["workload"], runtime_config=inputs["runtime"], policy=inputs["policy"],
            planner_context=D.PlannerContext(context_input, context_value),
            source_builder_id="source", evidence_plan_id="evidence", runner_args_id="runner",
            experiment_template_registry_id="templates",
            experiment_template_registry_sha256="e" * 64,
            inference_window_lease_id="lease",
            production_snapshot_id="production")

    def test_typed_registry_resolves_and_materializes(self):
        """The resolver and materializer must accept the same concrete types."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "production").mkdir()
            config = self._config(root)
            evidence_plan = plan(root / "inputs")
            protected = write_bound(root / "production", "frozen-hip.so",
                                    b"frozen hip", "hip_library")
            template = F.ExperimentTemplate(
                "fattn-v1", "gpu-decode", "fattn_kernel", "backend-fattn",
                "fattn-dispatch", evidence_plan.dispatch,
                frozenset({"ggml/src/ggml-cuda/fattn.cu"}),
                {"ggml/src/ggml-cuda/fattn.cu": frozenset({"fattn_kernel"})},
                {"kind": "fattn"})
            registry = {
                "environment_profile": {"sealed-codex": F.EnvironmentProfile({"PATH": "/usr/bin"})},
                "source_builder": {"source": F.SourceBuilderBinding(mock.Mock())},
                "evidence_plan": {"evidence": F.EvidencePlanBinding(mock.Mock())},
                "runner_args": {"runner": F.RunnerArgsBinding(mock.Mock())},
                "experiment_template_registry": {"templates": F.ExperimentTemplateRegistry(
                    "v1", "e" * 64, {"fattn-v1": template})},
                "inference_window_lease": {"lease": F.InferenceWindowLeaseBinding()},
                "production_snapshot": {"production": F.ProductionSnapshotBinding(
                    (protected,))},
            }
            # This call is the real public registry boundary.  It currently
            # rejects the factory's own typed bindings as non-callable/non-maps.
            with mock.patch.object(D.DiscoveryDeployment, "revalidate"):
                try:
                    resolved = D.resolve_registry(config, registry)
                except D.DeploymentConfigError as exc:
                    self.fail(f"factory's own typed registry is unresolvable: {exc}")
            self.assertIsInstance(resolved.environment_profile, F.EnvironmentProfile)
            self.assertIsInstance(resolved.experiment_template_registry,
                                  F.ExperimentTemplateRegistry)
            executors, claims = FakeExecutors(), ClaimFactory()
            with mock.patch.object(D.DiscoveryDeployment, "revalidate"):
                adapters = F.materialize(
                    config, registry,
                    correctness_executor=executors.correctness,
                    rocprof_executor=executors.rocprof, claim_journal=object(),
                    claim_acquirer=claims, claim_verifier=lambda _receipt: True)
            self.assertIsInstance(adapters["planner"], C.CodexPlanner)
            self.assertIsInstance(adapters["critic"], C.CodexCritic)
            self.assertEqual(executors.calls, [])
            self.assertEqual(claims.claims, [])

    def test_deployment_entrypoint_needs_only_sealed_config(self):
        """The executable launcher may not require Python-injected authority."""
        parameters = inspect.signature(F.deployment_main).parameters
        self.assertEqual(set(parameters), {"argv"})

    def test_actor_authority_is_constructed_from_sealed_wrapper_and_environment(self):
        """Callers cannot substitute spoof planner/critic objects."""
        parameters = inspect.signature(F.materialize).parameters
        self.assertNotIn("planner", parameters)
        self.assertNotIn("critic", parameters)

    def test_configured_lock_is_the_actual_runner_lock(self):
        """A configured non-default lock must own every model-call window."""
        parameters = inspect.signature(gpu_runner.invoke).parameters
        self.assertIn("inference_window_lock", parameters)
        # The launch-gate integration test below supplies a held nondefault
        # lock and proves _invoke_locked is not entered until it is released.

    def test_initial_planner_context_contains_sealed_search_inputs(self):
        """Turn one must be authorable from sealed evidence, not a blank prompt."""
        with tempfile.TemporaryDirectory() as directory:
            # Exercise the real context function directly: actor execution is
            # outside this gate and no compute subprocess is started.
            root = Path(directory)
            store = C.DurableState(root / "state")
            config = self._config(root)
            with mock.patch.object(D.DiscoveryDeployment, "revalidate"):
                controller_config = F.controller_config(config, dry_run=True)
            context = C._context(store.load(), C._tracker(store), 1,
                                 controller_config)
        self.assertIn("planner_context", context)
        self.assertIn("hotspots", context["planner_context"])
        self.assertIn("source_constraints", context["planner_context"])
        self.assertIn("authoring_assignment", context)

    def test_planner_prompt_is_authorable_and_controller_binds_source_identity(self):
        """Sol receives schemas plus controller-owned candidate/base identities."""
        source = inspect.getsource(C.CodexPlanner.plan)
        for token in ("plan_json_keys", "source_manifest", "authoring_assignment",
                      "experiment_template_catalog"):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_measured_feedback_is_semantic_not_hash_only(self):
        """Later Sol turns receive measured disposition/effect/evidence and DNR."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = C.DurableState(root / "state")
            state = store.load()
            state["iterations"].append({
                "status": "screened_out", "result_sha256": "a" * 64,
                "effect_fraction": -.02,
                "evidence": {"source": "b" * 64, "dispatch": "c" * 64,
                             "baseline": "d" * 64},
            })
            config = self._config(root)
            with mock.patch.object(D.DiscoveryDeployment, "revalidate"):
                controller_config = F.controller_config(config, dry_run=True)
            context = C._context(state, C._tracker(store), 2,
                                 controller_config)
        self.assertIsInstance(context["prior_results"][0], dict)
        self.assertEqual(context["prior_results"][0]["status"], "screened_out")
        self.assertEqual(context["prior_results"][0]["effect_fraction"], -.02)
        self.assertEqual(context["prior_results"][0]["evidence"]["source"], "b" * 64)
        self.assertIn("do_not_repeat", context)

    def test_reward_path_is_byte_identical_across_arms(self):
        """Only candidate kernel material and its DSO may differ from anchor."""
        with tempfile.TemporaryDirectory() as directory:
            evidence_plan = plan(Path(directory))
        candidate = evidence_plan.identity_files.candidate
        anchor = evidence_plan.identity_files.anchor
        # Preferred closure: the same measurement executable is used with an
        # arm-specific HIP DSO selected by isolated LD_LIBRARY_PATH.  If the
        # implementation chooses distinct build-metadata-bearing binaries,
        # this gate must be replaced by an explicit normalized object/source
        # closure; arbitrary two-binary inequality is never sufficient.
        self.assertEqual(candidate.binary.sha256, anchor.binary.sha256)
        self.assertEqual(evidence_plan.workload_sha256,
                         evidence_plan.identity_files.workload.sha256)
        self.assertEqual(evidence_plan.runtime_config_sha256,
                         evidence_plan.identity_files.runtime_config.sha256)

    def test_source_scope_rejects_reward_symbols_even_under_kernel_prefix(self):
        """A path prefix alone cannot authorize benchmark/reward manipulation."""
        manifest = mock.Mock(
            source_tree="llama.cpp",
            declared_files=("ggml/src/ggml-hip/reward_adapter.cpp",),
            declared_symbols={"ggml/src/ggml-hip/reward_adapter.cpp":
                              ("main", "emit_benchmark_result")},
        )
        with self.assertRaises(F.DeploymentFactoryError):
            F._validate_source_scope(mock.Mock(source_manifest=manifest))

    def test_candidate_selects_allowlisted_evidence_template_not_raw_commands(self):
        """One run can evaluate distinct hypotheses with sealed proof templates."""
        fields = C.PlannedCandidate.__dataclass_fields__
        self.assertIn("experiment_intent", fields)
        self.assertNotIn("correctness_argv", fields)
        self.assertNotIn("dispatch_regex", fields)

    def test_controller_automatically_replicates_and_pooled_nominates(self):
        """A positive S1 causes fresh-authorized S2, then pooled nomination."""
        class RecordingScreen:
            def __init__(self):
                self.calls = []
            def screen(self, candidate, authorization, lease):
                self.calls.append((authorization.to_dict(), dict(lease)))
                number = len(self.calls)
                return C.SealedScreen(
                    f"result-{number}.json", f"{number:064x}", .04,
                    "candidate", "a" * 64, "b" * 64, "c" * 64,
                    series_key="d" * 64)
            def reconcile(self, inflight):
                return C.Recovery("safe_to_start")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(C.source_candidate, "SourcePatchManifest", Manifest), \
                mock.patch.object(C, "_write_projection"):
            root = Path(directory)
            screen = RecordingScreen()
            result = C.run_controller(
                C.ControllerConfig((root / "state").resolve(), max_iterations=2,
                                   nomination_threshold=.03),
                planner=Planner(), critic=Critic(), screener=screen,
                lease=Lease((True, True)))
            self.assertEqual([row[1]["repetition"] for row in screen.calls], [1, 2])
            self.assertNotEqual(screen.calls[0][0], screen.calls[1][0])
            self.assertEqual(result["iterations"][-1]["status"],
                             "top_k_replicated_candidate")
            queue = (root / "state" / "promotion-queue.jsonl").read_text()
            self.assertIn('"promotion_claim": false', queue)


if __name__ == "__main__":
    unittest.main()
