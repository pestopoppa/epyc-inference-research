"""Black-box, hardware-free launch acceptance for GPU source discovery.

This is intentionally an acceptance gate rather than a collection of unit
tests for individual helpers.  Every assertion describes authority required
at the deployment process boundary.  A failure means the autonomous launcher
must remain disabled; it is not permission to weaken the assertion.
"""
from __future__ import annotations

import inspect
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.benchmark import run_autokernel_gpu_discovery as gpu_runner

from . import discovery_controller as C
from . import discovery_deployment as D
from . import discovery_deployment_factory as F
from . import discovery_static_registry as S
from . import gpu_source_evidence as E
from . import gpu_residency_sampler as R
from . import split_runtime_verifier as V
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
                "dispatch": {
                    "candidate_exact": [vars(row) for row in template.dispatch.candidate_exact],
                    "anchor_exact": [vars(row) for row in template.dispatch.anchor_exact],
                    "candidate_forbidden": [vars(row) for row in template.dispatch.candidate_forbidden],
                    "anchor_forbidden": [vars(row) for row in template.dispatch.anchor_forbidden],
                    "invariants": [vars(row) for row in template.dispatch.invariants],
                },
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

    def test_overlap_mode_receipts_bandwidth_duty_cycle_not_model_size(self):
        """Overlap authority is rolling cold-load bytes, not a size threshold."""
        parameters = inspect.signature(gpu_runner.invoke).parameters
        self.assertIn("inference_window_lock", parameters)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "small.gguf"
            model.write_bytes(b"small-model")
            configured = root / "gpu-discovery.lock"
            transfer = gpu_runner.host_transfer_admission(
                bytes_per_cold_load=model.stat().st_size, cold_loads=1,
                interval_s=60, host_bandwidth_bytes_s=10**9,
                conservative_fraction=.1, site_policy_allows_overlap=True,
                observed_headroom=True)
            with mock.patch.object(gpu_runner.cpu_region_claim,
                                      "inspect_region_claims",
                                      return_value={"regions": {}}), \
                    mock.patch.object(gpu_runner, "_invoke_locked",
                                      return_value={"hip_residency_proved": True}):
                receipt = gpu_runner.invoke(
                    build=root, model=model, seed=1, baseline_vram=0,
                    flash_attention=True, campaign_id="ak-offline",
                    cpu_journal=object(), allow_small_model_cpu_overlap=True,
                    sealed_load_decision=transfer,
                    inference_window_lock=configured)
        self.assertIsNone(receipt["inference_call_window"])
        coverage = receipt["cpu_coverage"]
        self.assertEqual(coverage["cpu_overlap_policy"], "allowed_discovery_noise")
        overlap = coverage["host_transfer"]
        for key in ("policy_version", "mode", "inputs", "reason",
                    "lock_interval", "residency_transition"):
            self.assertIn(key, overlap)
        self.assertIn(overlap["mode"],
                      {"hot_resident", "cold_overlap", "cold_serialized"})
        self.assertFalse(coverage["promotion_claim"])
        self.assertEqual(gpu_runner.DEVICE_ID, "mi210_0")

    def test_three_mode_admission_defaults_unknown_or_excess_to_serialized(self):
        """Overlap is earned by all predicates; missing data never guesses."""
        decide = getattr(gpu_runner, "host_transfer_admission")
        common = dict(interval_s=60, host_bandwidth_bytes_s=100_000,
                      conservative_fraction=.1)
        overlap = decide(bytes_per_cold_load=400, cold_loads=18,
                         site_policy_allows_overlap=True,
                         observed_headroom=True, hot_resident=False, **common)
        excess = decide(bytes_per_cold_load=40_000, cold_loads=18,
                         site_policy_allows_overlap=True,
                         observed_headroom=True, hot_resident=False, **common)
        unknown = decide(bytes_per_cold_load=400, cold_loads=18,
                         site_policy_allows_overlap=False,
                         observed_headroom=False, hot_resident=False, **common)
        hot = decide(bytes_per_cold_load=1, cold_loads=1,
                     site_policy_allows_overlap=False,
                     observed_headroom=False, hot_resident=True,
                     resident_identity="a" * 64,
                     expected_identity="a" * 64, **common)
        self.assertEqual(overlap["mode"], "cold_overlap")
        self.assertEqual(excess["mode"], "cold_serialized")
        self.assertEqual(unknown["mode"], "cold_serialized")
        self.assertIn(hot["mode"], {"hot_resident", "cold_serialized"})
        for row in (overlap, excess, unknown, hot):
            self.assertTrue({"policy_version", "mode", "inputs", "reason",
                             "lock_interval", "residency_transition"}.issubset(row))

    def test_large_load_serializes_only_load_then_reuses_hot_residency(self):
        """Large loads release the CPU lock before repeated hot GPU calls."""
        source = inspect.getsource(gpu_runner.run)
        self.assertIn("load_phase_window", source)
        self.assertIn("hot_gpu_residency", source)
        self.assertIn("unexpected_reload", source)
        self.assertIn("residency_changed", source)

    def test_current_nonpersistent_runner_never_claims_hot_resident(self):
        """Hot mode needs a persistent residency witness the current loop lacks."""
        source = inspect.getsource(gpu_runner.run)
        self.assertNotIn('"mode": "hot_resident"', source)

    def test_sealed_admission_corpus_is_context_and_state_provenance(self):
        """Actors may cite examples, but their version/hash remain controller-owned."""
        fields = C.ControllerConfig.__dataclass_fields__
        self.assertIn("admission_corpus_version", fields)
        self.assertIn("admission_corpus_sha256", fields)
        context_source = inspect.getsource(C._context)
        state_source = inspect.getsource(C.DurableState.save)
        self.assertIn("admission_corpus_version", context_source)
        self.assertIn("admission_corpus_sha256", context_source)
        self.assertIn("admission_corpus_sha256", state_source)

    def test_wrong_actor_overlap_recommendation_is_deterministically_downgraded(self):
        """Negative examples are facts, not prose an actor can vote past."""
        keys = {"facts", "missing", "mode", "rationale", "disqualifiers",
                "counterfactual", "evidence"}
        corpus = {
            "version": "admission-corpus-v1", "sha256": "c" * 64,
            "examples": [
                {"facts": {"profile": "exact-reviewed-tg128"}, "missing": [],
                 "mode": "cold_overlap", "rationale": "reviewed profile",
                 "disqualifiers": [], "counterfactual": "higher cadence serializes",
                 "evidence": ["sha256:" + "a" * 64]},
                *[{"facts": {"case": case}, "missing": (["headroom"] if case == "unknown" else []),
                   "mode": "cold_serialized", "rationale": case,
                   "disqualifiers": [case], "counterfactual": "exact reviewed facts",
                   "evidence": ["sha256:" + "b" * 64]}
                  for case in ("high_cadence", "unknown", "large", "hot_mismatch", "foreign_kfd")],
            ],
        }
        self.assertTrue(all(set(row) == keys for row in corpus["examples"]))
        decide = getattr(gpu_runner, "host_transfer_admission")
        self.assertIn("admission_corpus", inspect.signature(decide).parameters)
        for case in ("high_cadence", "unknown", "large", "hot_mismatch", "foreign_kfd"):
            result = decide(admission_corpus=corpus, observed_case=case,
                            actor_recommendation="cold_overlap")
            with self.subTest(case=case):
                self.assertIn(result["mode"], {"cold_serialized", "refused"})
                self.assertNotEqual(result.get("authorized_by"), "actor")

    def test_planner_can_author_distinct_literal_dispatch_expectations(self):
        """One template supports genuine hypotheses without actor-authored regex/argv."""
        expectation = getattr(C, "BoundedDispatchExpectation", None)
        self.assertIsNotNone(expectation)
        first = expectation(kernel_name="fattn_vec_f32", calls=2,
                            grid=(128, 1, 1), workgroup=(64, 1, 1), lds_bytes=0)
        second = expectation(kernel_name="fattn_vec_f32_v2", calls=1,
                             grid=(64, 1, 1), workgroup=(64, 1, 1), lds_bytes=256)
        self.assertNotEqual(first, second)
        self.assertIn("expected_dispatch",
                      C.GpuSourceExperimentIntent.__dataclass_fields__)
        critic_source = inspect.getsource(C.CodexCritic.review)
        self.assertIn("expected_dispatch", critic_source)
        materializer_source = inspect.getsource(F.materialize)
        self.assertIn("expected_dispatch", materializer_source)
        self.assertNotIn("actor_dispatch_regex", materializer_source)

    def test_literal_dispatch_expectation_refuses_meta_and_out_of_range(self):
        """Kernel names and geometry are bounded literals, never patterns."""
        expectation = getattr(C, "BoundedDispatchExpectation", None)
        self.assertIsNotNone(expectation)
        for kernel in ("*", ".*", "kernel[0]", "kernel;exec", "kernel $(x)"):
            with self.subTest(kernel=kernel), self.assertRaises(Exception):
                expectation(kernel_name=kernel, calls=1, grid=(1, 1, 1),
                            workgroup=(1, 1, 1), lds_bytes=0)
        for bad in (0, -1, 2**31):
            with self.subTest(calls=bad), self.assertRaises(Exception):
                expectation(kernel_name="kernel_v1", calls=bad, grid=(1, 1, 1),
                            workgroup=(1, 1, 1), lds_bytes=0)

    def test_planner_is_not_a_canned_finite_experiment_queue(self):
        """Deployment templates bound authority, not a finite patch inventory."""
        source = inspect.getsource(C.CodexPlanner.plan)
        self.assertIn("patch", source)
        self.assertIn("expected_dispatch", source)
        self.assertNotIn("canned_candidates", source)
        self.assertNotIn("finite_experiment_queue", source)

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
                versioned = build / "bin" / "libggml-hip.so.0.16.0"
                versioned.write_bytes(hip)
                (build / "bin" / "libggml-hip.so.0").symlink_to(versioned.name)
                (build / "bin" / "libggml-hip.so").symlink_to("libggml-hip.so.0")
                builds.append(build)
            model = root / "model.gguf"; model.write_bytes(b"model")
            common = root / "common-runtime"; common.mkdir()
            def commit(argv, **_kwargs):
                return mock.Mock(returncode=0,
                                 stdout=("a" if str(builds[0]) in argv else "b") * 40)
            args = argparse.Namespace(
                model=str(model), anchor_build=str(builds[0]),
                candidate_build=str(builds[1]), factor="source_patch",
                campaign_id="ak-offline", calls=3, workload="prefill_pp512",
                allow_small_model_cpu_overlap=True,
                measurement_binary=str(builds[0] / "bin" / "llama-bench"),
                common_loader_dir=str(common),
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
            root = Path(directory) / "runtime"
            common, anchor, candidate = (root / name for name in
                                          ("common", "anchor-hip", "candidate-hip"))
            for path in (common, anchor, candidate):
                path.mkdir(parents=True)
            (common / "llama-bench").write_bytes(b"reward")
            (common / "llama-bench").chmod(0o755)
            (common / "libllama-bench-impl.so").write_bytes(b"bench")
            families = ("libllama-common.so", "libllama.so", "libggml.so",
                        "libggml-base.so", "libggml-cpu.so")
            for directory_path, stems in ((common, families),
                                          (anchor, ("libggml-hip.so",)),
                                          (candidate, ("libggml-hip.so",))):
                for stem in stems:
                    version = directory_path / f"{stem}.0.16.0"
                    version.write_bytes((directory_path.name + stem).encode())
                    (directory_path / f"{stem}.0").symlink_to(version.name)
                    (directory_path / stem).symlink_to(f"{stem}.0")

            def elf(path: Path) -> V.ElfIdentity:
                if path.name == "llama-bench":
                    return V.ElfIdentity(None, ("libllama-bench-impl.so",), ("$ORIGIN",))
                if path.name.startswith("libggml-hip.so."):
                    return V.ElfIdentity("libggml-hip.so.0", ("libggml-base.so.0",),
                                         ("/opt/rocm/lib",))
                soname = ("libllama-bench-impl.so" if path.name == "libllama-bench-impl.so"
                          else path.name.rsplit(".0.", 1)[0] + ".0")
                return V.ElfIdentity(soname, ("libc.so.6",), ("$ORIGIN",))

            manifest = V.verify_split_runtime(root, elf_reader=elf)
            model = Path(directory) / "model.gguf"; model.write_bytes(b"model")
            model_sha = hashlib.sha256(model.read_bytes()).hexdigest()

            def maps(arm: str) -> str:
                paths = {path.resolve() for path in common.iterdir() if path.is_file()}
                hip_dir = anchor if arm == "anchor" else candidate
                paths |= {(hip_dir / "libggml-hip.so.0").resolve(), model.resolve()}
                return "\n".join(f"7f00-7f01 r-xp 0 00:00 1 {path}" for path in paths)

            anchor_maps = V.verify_runtime_maps(
                manifest, arm="anchor", maps_text=maps("anchor"), model_path=model,
                model_sha256=model_sha, device_id="mi210_0", kfd_pid=1,
                boot_id="boot", process_start_ticks=10)
            candidate_maps = V.verify_runtime_maps(
                manifest, arm="candidate", maps_text=maps("candidate"), model_path=model,
                model_sha256=model_sha, device_id="mi210_0", kfd_pid=2,
                boot_id="boot", process_start_ticks=11)
            V.validate_arm_pair(anchor_maps, candidate_maps)
        self.assertEqual(anchor_maps.reward_binary_sha256,
                         candidate_maps.reward_binary_sha256)
        self.assertNotEqual(anchor_maps.hip_library_sha256,
                            candidate_maps.hip_library_sha256)

    def test_source_patch_has_no_legacy_runtime_closure_fallback(self):
        """Every source patch must supply one reward binary and two HIP dirs."""
        source = inspect.getsource(gpu_runner.preflight)
        self.assertNotIn('and getattr(args, "measurement_binary", None)', source)
        self.assertIn("libggml-hip.so.0", source)
        self.assertIn("sealed shared reward runtime closure", source)
        for field in ("measurement_binary", "common_loader_dir",
                      "anchor_loader_dir", "candidate_loader_dir"):
            self.assertIn(field, source)

    def test_static_builder_always_tears_down_owned_worktrees(self):
        """Success and failure both end in governed teardown receipts."""
        source = inspect.getsource(S.StaticGpuSourceBuilder.build)
        self.assertIn("finally", source)
        self.assertIn("teardown_worktree", source)
        self.assertIn("source_materialization_teardown", source)
        self.assertIn("receipt_sha256", source)

    def test_static_builder_validates_build_results_and_required_targets(self):
        """A failed/missing bench or correctness target never becomes evidence."""
        source = inspect.getsource(S.StaticGpuSourceBuilder.build)
        self.assertIn("test-backend-ops", source)
        self.assertIn("result.succeeded", source)
        self.assertIn("log_disagrees_with_exit_code", source)
        self.assertIn("built_targets", source)

    def test_runtime_paths_are_operation_key_scoped_for_s1_and_s2(self):
        """Independent replications cannot collide in a campaign-only path."""
        source = inspect.getsource(S.StaticGpuSourceBuilder.build)
        self.assertIn('_permit.get("operation_key")', source)
        self.assertIn('"runtime" / operation_key', source)

    def test_empty_kfd_never_becomes_residency_from_aggregate_vram(self):
        """A VRAM delta without the captured child KFD PID proves nothing."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kfd = root / "kfd"; kfd.mkdir()
            vram = root / "vram"; vram.write_text("1048576")
            proc = root / "proc"; proc.mkdir()
            sample = R.Mi210ResidencySampler(
                kfd_root=kfd, vram_path=vram, proc_root=proc)(123)
        self.assertEqual(sample.kfd_pids, (123,))
        self.assertEqual(sample.vram_bytes, 0)

    def test_build_logs_are_operation_key_scoped_and_exclusive(self):
        """S1/S2 cannot append to or overwrite one campaign-named build log."""
        source = inspect.getsource(S.StaticGpuSourceBuilder.build)
        self.assertIn("operation_key", source)
        self.assertIn('f"{ident}.log"', source)
        self.assertIn('open(log, "w"', inspect.getsource(S.worktree.run_build))

    def test_teardown_attempts_every_tree_and_seals_all_outcomes(self):
        """One teardown failure cannot skip the remaining owned worktrees."""
        source = inspect.getsource(S.StaticGpuSourceBuilder.build)
        self.assertIn("receipts", source)
        self.assertIn("teardown_errors", source)
        self.assertIn("receipt_sha256", source)
        self.assertIn("for", source[source.find("finally"):])

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
