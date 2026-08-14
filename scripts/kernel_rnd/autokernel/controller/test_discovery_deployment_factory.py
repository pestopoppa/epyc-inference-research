from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile

from . import discovery_deployment_factory as F
from . import discovery_controller as C


def template(path="ggml/src/ggml-cuda/fattn.cu", symbol="fattn_kernel"):
    return F.ExperimentTemplate("fattn-v1", "gpu_decode", symbol, "backend-fattn",
                                "fattn-dispatch", mock.Mock(), frozenset({path}),
                                {path: frozenset({symbol})}, {"kind": "fattn"})


class DeploymentFactoryTests(unittest.TestCase):
    def test_environment_rejects_loader_injection(self):
        for key in ("LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "DYLD_INSERT_LIBRARIES"):
            with self.subTest(key=key):
                with self.assertRaises(F.DeploymentFactoryError):
                    F.EnvironmentProfile({"PATH": "/usr/bin", key: "bad"})

    def test_source_scope_refuses_reward_and_toolchain_mutations(self):
        class Manifest:
            source_tree = "llama.cpp"
            def __init__(self, paths):
                self.declared_files = paths
                self.declared_symbols = {path: ("fattn_kernel",) for path in paths}
        for path in ("tools/llama-bench/llama-bench.cpp", "CMakeLists.txt",
                     "cmake/toolchain.cmake", "scripts/parse.py", "tests/test.cpp",
                     "ggml/src/ggml.c"):
            candidate = mock.Mock(source_manifest=Manifest((path,)))
            with self.subTest(path=path), self.assertRaises(F.DeploymentFactoryError):
                F._validate_source_scope(candidate, template())
        F._validate_source_scope(mock.Mock(
            source_manifest=Manifest(("ggml/src/ggml-cuda/fattn.cu",))), template())

    def test_controller_config_has_no_cli_override_authority(self):
        context = {"context_sha256": "a" * 64}
        config = mock.Mock(state_root=Path("/state"), evidence_root=Path("/evidence"),
                           max_iterations=2, nomination_threshold=.03,
                           planner_context=mock.Mock(value=context), production_branch="production-consolidated-v9",
                           production_head="b" * 40,
                           instrument_branch="measurement-instrument",
                           instrument_commit="c" * 40,
                           config_sha256="c" * 64,
                           experiment_template_registry_sha256="d" * 64)
        config.admission_policy = SimpleNamespace(
            value={"policy_sha256": "e" * 64, "examples": [], "profiles": []},
            corpus=SimpleNamespace(policy_sha256="e" * 64, version="test-v2"))
        config.revalidate = mock.Mock()
        result = F.controller_config(config, dry_run=True)
        self.assertEqual((result.output_root, result.evidence_root,
                          result.max_iterations, result.nomination_threshold,
                          result.dry_run, result.planner_context_sha256, result.production_base_commit),
                         (Path("/state"), Path("/evidence"), 2, .03, True,
                          F.schemas.content_hash({"planner_context_sha256": "a" * 64,
                                                  "admission_policy_sha256": "e" * 64,
                                                  "admission_policy_version": "test-v2",
                                                  "deployment_identity_sha256": "c" * 64}), "b" * 40))
        self.assertEqual(result.deployment_identity_sha256, "c" * 64)
        config.revalidate.assert_called_once()

    def test_window_lease_uses_sealed_arbiter_and_never_probes_cpu_lock(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"; model.write_bytes(b"model")
            profile = SimpleNamespace(model_path=str(model), model_sha256="a" * 64,
                                      device_id="mi210_0", workload="tg128", calls_per_arm=9,
                                      cold_load_host_bytes=4, worst_case_loads_per_interval=18)
            config = mock.Mock(inference_window_lock="/lock", device_id="mi210_0",
                               model=SimpleNamespace(path=model, sha256="a" * 64),
                               admission_policy=SimpleNamespace(corpus=SimpleNamespace(
                                   profiles=(profile,), policy_sha256="b" * 64, version="test-v2")),
                               planner_context=SimpleNamespace(value={"context_sha256": "c" * 64}))
            config.revalidate = mock.Mock()
            decision = SimpleNamespace(mode="cold_serialized", to_dict=lambda: {"decision_sha256": "d" * 64})
            with mock.patch.object(F.gpu_load_admission, "arbitrate", return_value=decision), \
                 mock.patch.object(F.inference_window.InferenceCallWindow, "acquire",
                                   side_effect=AssertionError("lease must not invent a CPU lock probe")):
                admitted = F.GpuDiscoveryLease(config=config, mode="allowed_discovery_noise").admit(mock.Mock())
        self.assertTrue(admitted["admitted"])
        self.assertEqual(admitted["mode"], "cold_serialized")
        self.assertEqual(admitted["load_admission"], {"decision_sha256": "d" * 64})

    def test_materialized_builder_preserves_each_operation_and_binds_deployment_authority(self):
        """The build cache key must never replace the controller operation key."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            protected = root / "protected"; protected.mkdir()
            artifact = protected / "artifact"; artifact.write_bytes(b"frozen")
            bound = F.evidence.BoundInputFile(
                "production_artifact", artifact,
                hashlib.sha256(artifact.read_bytes()).hexdigest())
            calls = []
            source = F.SourceBuilderBinding(
                lambda _candidate, _authorization, permit: calls.append(dict(permit)) or dict(permit))
            templates = mock.Mock(spec=F.ExperimentTemplateRegistry)
            templates.registry_sha256 = "e" * 64
            templates.templates = {}
            templates.resolve.return_value = mock.sentinel.template
            resolved = SimpleNamespace(
                environment_profile=F.EnvironmentProfile({"PATH": "/usr/bin"}),
                source_builder=source,
                evidence_plan=F.EvidencePlanBinding(mock.Mock()),
                runner_args=F.RunnerArgsBinding(mock.Mock()),
                experiment_template_registry=templates,
                inference_window_lease=F.InferenceWindowLeaseBinding(),
                production_snapshot=F.ProductionSnapshotBinding((bound,)))
            config = mock.Mock(
                config_sha256="d" * 64, experiment_template_registry_sha256="e" * 64,
                actor_wrapper=SimpleNamespace(path=Path("/sealed/codex-wrapper")),
                production_path=protected, instrument_path=protected,
                claim_timeout_s=0.0, instrument_branch="measurement-instrument")
            config.revalidate = mock.Mock()
            candidate = mock.Mock(experiment_intent=mock.sentinel.intent,
                                  source_manifest=mock.Mock())
            adapters = {}
            def adapter_factory(**kwargs):
                adapters.update(kwargs)
                return mock.sentinel.screener
            with mock.patch.object(F.deployment, "resolve_registry", return_value=resolved), \
                 mock.patch.object(F, "_validate_source_scope"), \
                 mock.patch.object(F.gpu_source_adapter, "build_governed_gpu_source_adapter",
                                   side_effect=adapter_factory), \
                 mock.patch.object(F.controller, "build_controller_adapters",
                                   side_effect=lambda **kwargs: kwargs):
                F.materialize(config, {}, correctness_executor=mock.Mock(),
                              rocprof_executor=mock.Mock(), claim_journal=mock.Mock())
                build = adapters["build_source"]
                first = build(candidate, object(), {"operation_key": "1" * 64})
                second = build(candidate, object(), {"operation_key": "2" * 64})
            self.assertEqual(first["operation_key"], "1" * 64)
            self.assertEqual(second["operation_key"], "2" * 64)
            self.assertEqual([row["deployment_config_sha256"] for row in calls],
                             [config.config_sha256, config.config_sha256])
            self.assertEqual([row["instrument_branch"] for row in calls],
                             [config.instrument_branch, config.instrument_branch])
            with self.assertRaisesRegex(F.DeploymentFactoryError, "operation identity"):
                adapters["args_factory"](
                    candidate, mock.Mock(operation_key="1" * 64),
                    {"operation_key": "2" * 64})


if __name__ == "__main__":
    unittest.main()
