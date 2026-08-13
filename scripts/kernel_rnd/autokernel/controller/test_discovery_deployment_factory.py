from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path

from . import discovery_deployment_factory as F
from . import discovery_controller as C


class DeploymentFactoryTests(unittest.TestCase):
    def test_environment_rejects_loader_injection(self):
        for key in ("LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "DYLD_INSERT_LIBRARIES"):
            with self.subTest(key=key):
                with self.assertRaises(F.DeploymentFactoryError):
                    F.EnvironmentProfile({"PATH": "/usr/bin", key: "bad"})

    def test_source_scope_refuses_reward_and_toolchain_mutations(self):
        class Manifest:
            source_tree = "llama.cpp"
            def __init__(self, paths): self.declared_files = paths
        for path in ("tools/llama-bench/llama-bench.cpp", "CMakeLists.txt",
                     "cmake/toolchain.cmake", "scripts/parse.py", "tests/test.cpp",
                     "ggml/src/ggml.c"):
            candidate = mock.Mock(source_manifest=Manifest((path,)))
            if path.startswith("ggml/src/"):
                F._validate_source_scope(candidate)
            else:
                with self.subTest(path=path), self.assertRaises(F.DeploymentFactoryError):
                    F._validate_source_scope(candidate)

    def test_controller_config_has_no_cli_override_authority(self):
        config = mock.Mock(state_root=Path("/state"), evidence_root=Path("/evidence"),
                           max_iterations=2, nomination_threshold=.03)
        config.revalidate = mock.Mock()
        result = F.controller_config(config, dry_run=True)
        self.assertEqual((result.output_root, result.evidence_root,
                          result.max_iterations, result.nomination_threshold,
                          result.dry_run), (Path("/state"), Path("/evidence"), 2, .03, True))
        config.revalidate.assert_called_once()

    def test_window_lease_refuses_busy_and_binds_discovery_metadata(self):
        config = mock.Mock(inference_window_lock="/lock", device_id="mi210_0",
                           model=mock.Mock(sha256="a" * 64), small_model_max_bytes=1)
        config.revalidate = mock.Mock()
        with mock.patch.object(F.inference_window.InferenceCallWindow, "acquire",
                               side_effect=F.inference_window.InferenceWindowTimeout("busy")):
            denied = F.GpuDiscoveryLease(config=config, mode="allowed_discovery_noise").admit(mock.Mock())
        self.assertFalse(denied["admitted"])


if __name__ == "__main__":
    unittest.main()
