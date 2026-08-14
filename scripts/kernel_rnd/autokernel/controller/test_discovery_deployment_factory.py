from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

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
                           planner_context=mock.Mock(value=context), production_head="b" * 40,
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
                                                  "admission_policy_version": "test-v2"}), "b" * 40))
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


if __name__ == "__main__":
    unittest.main()
