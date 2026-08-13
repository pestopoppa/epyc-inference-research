"""Black-box, hardware-free launch acceptance for GPU source discovery.

This is intentionally an acceptance gate rather than a collection of unit
tests for individual helpers.  Every assertion describes authority required
at the deployment process boundary.  A failure means the autonomous launcher
must remain disabled; it is not permission to weaken the assertion.
"""
from __future__ import annotations

import inspect
import argparse
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
from . import test_discovery_deployment as deployment_tests
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
            template_body = {"version": "v1", "templates": {"fattn-v1": {
                "template_id": template.template_id,
                "target_surface": template.target_surface,
                "target_symbol": template.target_symbol,
                "correctness_id": template.correctness_id,
                "dispatch_id": template.dispatch_id,
                "allowed_files": sorted(template.allowed_files),
                "allowed_symbols": {path: sorted(symbols)
                                    for path, symbols in template.allowed_symbols.items()},
                "semantics": dict(template.semantics),
                "dispatch": repr(template.dispatch),
            }}}
            template_sha = __import__("hashlib").sha256(
                __import__("json").dumps(template_body, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
            object.__setattr__(config, "experiment_template_registry_sha256",
                               template_sha)
            registry = {
                "environment_profile": {"sealed-codex": F.EnvironmentProfile({"PATH": "/usr/bin"})},
                "source_builder": {"source": F.SourceBuilderBinding(mock.Mock())},
                "evidence_plan": {"evidence": F.EvidencePlanBinding(mock.Mock())},
                "runner_args": {"runner": F.RunnerArgsBinding(mock.Mock())},
                "experiment_template_registry": {"templates": F.ExperimentTemplateRegistry(
                    "v1", template_sha, {"fattn-v1": template})},
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

    def test_overlap_mode_binds_distinct_configured_lock_and_small_model_receipt(self):
        """Permitted CPU overlap remains capped, nonpromotable, and receipted."""
        parameters = inspect.signature(gpu_runner.invoke).parameters
        self.assertIn("inference_window_lock", parameters)
        self.assertEqual(gpu_runner.SMALL_MODEL_OVERLAP_MAX_BYTES,
                         512 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "small.gguf"
            model.write_bytes(b"small-model")
            configured = root / "gpu-discovery.lock"
            with mock.patch.object(gpu_runner.cpu_region_claim,
                                      "inspect_region_claims",
                                      return_value={"regions": {}}), \
                    mock.patch.object(gpu_runner, "_invoke_locked",
                                      return_value={"hip_residency_proved": True}):
                receipt = gpu_runner.invoke(
                    build=root, model=model, seed=1, baseline_vram=0,
                    flash_attention=True, campaign_id="ak-offline",
                    cpu_journal=object(), allow_small_model_cpu_overlap=True,
                    inference_window_lock=configured)
        self.assertIsNone(receipt["inference_call_window"])
        overlap = receipt["cpu_coverage"]
        self.assertEqual(overlap["cpu_overlap_policy"], "allowed_discovery_noise")
        self.assertLessEqual(overlap["model_size_bytes"], 512 * 1024 * 1024)
        self.assertFalse(overlap["promotion_claim"])
        self.assertEqual(gpu_runner.DEVICE_ID, "mi210_0")

    def test_deployment_refuses_overlap_cap_above_ratified_ceiling(self):
        """A sealed config cannot expand allowed-noise overlap beyond 512 MiB."""
        with tempfile.TemporaryDirectory() as directory:
            helper = deployment_tests.DeploymentConfigTests()
            path, raw = helper.config(Path(directory))
            raw["gpu"]["small_model_max_bytes"] = 512 * 1024 * 1024 + 1
            deployment_tests.seal(raw)
            path.write_text(__import__("json").dumps(raw))
            with self.assertRaises(D.DeploymentConfigError), \
                    mock.patch.object(D, "_verify_production"):
                D.load_deployment_config(path)

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

    def test_reward_path_uses_one_binary_and_distinct_arm_hip_dsos(self):
        """Only candidate kernel material and its DSO may differ from anchor."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builds = []
            for label, hip in (("anchor", b"anchor-hip"),
                               ("candidate", b"candidate-hip")):
                build = root / label
                (build / "bin").mkdir(parents=True)
                (build / "CMakeCache.txt").write_text(
                    "GGML_HIP_ROCWMMA_FATTN:BOOL=ON\n"
                    "GGML_HIP_MMQ_MFMA:BOOL=OFF\n"
                    "GGML_HIP_GRAPHS:BOOL=ON\n")
                binary = build / "bin" / "llama-bench"
                binary.write_bytes(b"shared-reward-binary")
                binary.chmod(0o755)
                (build / "bin" / "libggml-hip.so").write_bytes(hip)
                builds.append(build)
            model = root / "model.gguf"; model.write_bytes(b"model")
            def commit(argv, **_kwargs):
                return mock.Mock(returncode=0,
                                 stdout=("a" if str(builds[0]) in argv else "b") * 40)
            args = argparse.Namespace(
                model=str(model), anchor_build=str(builds[0]),
                candidate_build=str(builds[1]), factor="source_patch",
                campaign_id="ak-offline", calls=3, workload="prefill_pp512",
                allow_small_model_cpu_overlap=True,
                measurement_binary=str(builds[0] / "bin" / "llama-bench"),
                anchor_loader_dir=str(builds[0] / "bin"),
                candidate_loader_dir=str(builds[1] / "bin"),
                small_model_max_bytes=512 * 1024 * 1024,
                device_id="mi210_0", inference_window_lock=None)
            with mock.patch.object(gpu_runner.subprocess, "run", side_effect=commit):
                sealed = gpu_runner.preflight(args)
        closure = sealed["runtime_arms"]
        self.assertEqual(closure["reward_closure"],
                         "shared_anchor_binary_per_arm_hip_dso")
        self.assertNotEqual(closure["anchor_hip_sha256"],
                            closure["candidate_hip_sha256"])

    def test_complete_runtime_closure_proves_each_arm_loaded_intended_hip(self):
        """SONAME topology/targets close $ORIGIN fallback and DSO substitution."""
        with tempfile.TemporaryDirectory() as directory:
            evidence_plan = plan(Path(directory))
        closure = getattr(evidence_plan, "runtime_closure", None)
        self.assertIsInstance(closure, dict)
        self.assertEqual(set(closure), {"anchor", "candidate", "common"})
        self.assertEqual(closure["anchor"]["loaded_hip_sha256"],
                         evidence_plan.anchor.hip_library_sha256)
        self.assertEqual(closure["candidate"]["loaded_hip_sha256"],
                         evidence_plan.candidate.hip_library_sha256)
        self.assertEqual(closure["anchor"]["regular_target_hashes"],
                         closure["candidate"]["regular_target_hashes"])
        self.assertEqual(closure["anchor"]["soname_topology"],
                         closure["candidate"]["soname_topology"])

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
