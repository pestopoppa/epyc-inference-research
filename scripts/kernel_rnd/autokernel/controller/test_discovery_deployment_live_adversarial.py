"""No-hardware adversarial gates for the concrete discovery deployment.

This module intentionally remains runnable against an integration checkpoint.
If the concrete factory is absent on an older fixture-only base, the behavioral
class is skipped; the JSON launch contract is still mandatory there.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from . import discovery_deployment as D

try:
    from . import discovery_controller as C
    from . import discovery_deployment_factory as F
    from . import gpu_source_evidence as E
except ImportError:  # fixture-only acceptance branches predate the live factory
    C = None
    F = None
    E = None


H = "a" * 64


@unittest.skipIf(
    C is None or F is None or E is None,
    "concrete deployment not present on fixture base",
)
class ConcreteDeploymentAdversarialTests(unittest.TestCase):
    def dispatch(self):
        return E.DispatchContract(
            candidate_exact=(E.ExactDispatch("candidate", r"^candidate$", 1, 64, 64, 0, 1),),
            anchor_exact=(E.ExactDispatch("anchor", r"^anchor$", 1, 128, 128, 512, 1),),
        )

    def typed_registry(self, protected_file: Path):
        return {
            "environment_profile": {
                "env": F.EnvironmentProfile({"PATH": "/usr/bin"}),
            },
            "source_builder": {
                "source": F.SourceBuilderBinding(lambda *_args: None),
            },
            "evidence_plan": {
                "plan": F.EvidencePlanBinding(lambda *_args: None),
            },
            "runner_args": {
                "runner": F.RunnerArgsBinding(lambda *_args: None),
            },
            "dispatch_contract": {"dispatch": self.dispatch()},
            "inference_window_lease": {
                "lease": F.InferenceWindowLeaseBinding(),
            },
            "production_snapshot": {
                "snapshot": F.ProductionSnapshotBinding((
                    E.BoundInputFile("production_identity", protected_file, H),
                )),
            },
        }

    def config_stub(self):
        value = SimpleNamespace(
            environment_profile_id="env",
            source_builder_id="source",
            evidence_plan_id="plan",
            runner_args_id="runner",
            dispatch_contract_id="dispatch",
            inference_window_lease_id="lease",
            production_snapshot_id="snapshot",
        )
        value.revalidate = mock.Mock()
        return value

    def test_typed_registry_resolution_and_materialization_have_nonempty_intersection(self):
        with tempfile.TemporaryDirectory() as temp:
            protected = Path(temp) / "identity"
            protected.write_bytes(b"identity")
            registry = self.typed_registry(protected)
            # This is the exact type family required by materialize().  Registry
            # resolution must not retain a contradictory legacy Mapping/callable
            # predicate that makes the launch path impossible.
            resolved = D.resolve_registry(self.config_stub(), registry)
            self.assertIsInstance(resolved.environment_profile, F.EnvironmentProfile)
            self.assertIsInstance(resolved.source_builder, F.SourceBuilderBinding)
            self.assertIsInstance(resolved.evidence_plan, F.EvidencePlanBinding)
            self.assertIsInstance(resolved.runner_args, F.RunnerArgsBinding)
            self.assertIsInstance(resolved.dispatch_contract, E.DispatchContract)
            self.assertIsInstance(resolved.inference_window_lease, F.InferenceWindowLeaseBinding)
            self.assertIsInstance(resolved.production_snapshot, F.ProductionSnapshotBinding)

    def test_live_entrypoint_has_no_required_object_injection_authority(self):
        signature = inspect.signature(F.deployment_main)
        required = {
            name for name, parameter in signature.parameters.items()
            if name != "argv" and parameter.default is inspect.Parameter.empty
        }
        self.assertEqual(
            required,
            set(),
            "live launcher must construct its trusted registry, actors, executors, and journals",
        )

    def test_live_materializer_does_not_accept_arbitrary_actor_objects(self):
        parameters = inspect.signature(F.materialize).parameters
        self.assertNotIn("planner", parameters)
        self.assertNotIn("critic", parameters)

    def test_environment_is_an_allowlist_not_a_short_loader_denylist(self):
        for key in (
            "LD_PRELOAD", "LD_AUDIT", "GCONV_PATH", "PYTHONPATH",
            "PYTHONHOME", "BASH_ENV", "ENV", "PERL5OPT", "RUBYOPT",
        ):
            with self.subTest(key=key), self.assertRaises(F.DeploymentFactoryError):
                F.EnvironmentProfile({"PATH": "/usr/bin", key: "attacker-controlled"})

    def test_source_scope_is_gpu_kernel_only(self):
        def candidate(path: str):
            manifest = SimpleNamespace(
                source_tree="llama.cpp",
                declared_files=(path,),
                declared_symbols={path: ("kernel_symbol",)},
            )
            return SimpleNamespace(source_manifest=manifest)

        for path in (
            "ggml/src/ggml.c",
            "ggml/src/ggml-backend.cpp",
            "ggml/include/ggml.h",
            "tools/llama-bench/llama-bench.cpp",
            "CMakeLists.txt",
        ):
            with self.subTest(path=path), self.assertRaises(F.DeploymentFactoryError):
                F._validate_source_scope(candidate(path))
        F._validate_source_scope(candidate("ggml/src/ggml-cuda/fattn.cu"))

    def test_runner_parser_exposes_every_factory_checked_runtime_field(self):
        from scripts.benchmark import run_autokernel_gpu_discovery as runner

        destinations = {action.dest for action in runner.parser()._actions}
        self.assertTrue(
            {"inference_window_lock", "small_model_max_bytes", "device_id"}
            <= destinations,
            "factory must not validate synthetic Namespace fields ignored by the runner",
        )

    def test_controller_context_is_authorable_and_carries_measured_feedback(self):
        state = {
            "iterations": [{
                "result_sha256": "b" * 64,
                "effect_fraction": 0.031,
                "status": "top_k_replicated_candidate",
                "source_proof_sha256": "c" * 64,
                "dispatch_proof_sha256": "d" * 64,
            }],
        }
        with mock.patch.object(C, "_memory_block", return_value={"attempts": []}):
            context = C._context(state, object(), 2)
        self.assertIn("initial_context", context)
        self.assertIn("measured_feedback", context)
        self.assertIn("authoring_contract", context)
        initial = context["initial_context"]
        self.assertTrue(initial.get("workload_sha256"))
        self.assertTrue(initial.get("profile_sha256"))
        self.assertTrue(initial.get("source_excerpts"))
        feedback = context["measured_feedback"]
        self.assertEqual(feedback[0]["effect_fraction"], 0.031)
        self.assertEqual(feedback[0]["classification"], "top_k_replicated_candidate")
        self.assertEqual(feedback[0]["result_sha256"], "b" * 64)
        authoring = context["authoring_contract"]
        self.assertTrue({
            "campaign_id", "proposal_id", "candidate_id",
            "production_base_commit", "instrument_commit",
        } <= set(authoring["assigned_ids"]))
        self.assertTrue(authoring["output_schema"]["required_fields"])
        self.assertTrue(authoring["source_excerpts"])

    def test_deployment_selects_a_versioned_template_registry_not_one_fixed_contract(self):
        deployment_fields = getattr(D.DiscoveryDeployment, "__dataclass_fields__", {})
        self.assertIn("experiment_template_registry_id", deployment_fields)
        self.assertNotIn("dispatch_contract_id", deployment_fields)
        candidate_fields = getattr(C.PlannedCandidate, "__dataclass_fields__", {})
        self.assertIn("experiment_intent", candidate_fields)

    def test_two_candidates_can_select_distinct_sealed_templates_and_hotspots(self):
        registry_type = getattr(F, "ExperimentTemplateRegistry", None)
        intent_type = getattr(F, "ExperimentIntent", None)
        self.assertIsNotNone(registry_type)
        self.assertIsNotNone(intent_type)
        templates = registry_type.fixture_for_tests((
            {
                "template_id": "flash-d64",
                "allowed_files": ("ggml/src/ggml-cuda/fattn.cu",),
                "allowed_symbols": ("ggml_cuda_get_best_fattn_kernel",),
            },
            {
                "template_id": "q6-onewave",
                "allowed_files": ("ggml/src/ggml-cuda/mmvq.cu",),
                "allowed_symbols": ("calc_nwarps",),
            },
        ))
        first = intent_type(
            template_id="flash-d64",
            hotspot="flash_attn_ext",
            declared_files=("ggml/src/ggml-cuda/fattn.cu",),
            declared_symbols=("ggml_cuda_get_best_fattn_kernel",),
        )
        second = intent_type(
            template_id="q6-onewave",
            hotspot="mul_mat_vec_q6_k",
            declared_files=("ggml/src/ggml-cuda/mmvq.cu",),
            declared_symbols=("calc_nwarps",),
        )
        first_bound = templates.bind(first)
        second_bound = templates.bind(second)
        self.assertNotEqual(first_bound.intent_sha256, second_bound.intent_sha256)
        self.assertNotEqual(first_bound.dispatch, second_bound.dispatch)
        self.assertEqual(first_bound.template_id, "flash-d64")
        self.assertEqual(second_bound.template_id, "q6-onewave")


if __name__ == "__main__":
    unittest.main()
