"""Static contract tests for the live gfx90a LDS runner."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/benchmark/run_autokernel_lds_solver.py"
PROBE = ROOT / "scripts/benchmark/autokernel_lds_probe.cpp"


class LdsRunnerContractTest(unittest.TestCase):
    def test_runner_owns_claim_sampling_and_exact_gfx90a_build(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("acquire_device_claim", text)
        self.assertIn("RocmSmiSampler", text)
        self.assertIn('"--offload-arch=gfx90a"', text)
        self.assertIn("assert_source_identity", text)
        self.assertIn("diagnostic_only", text)

    def test_probe_is_self_contained_and_arch_refusing(self):
        text = PROBE.read_text(encoding="utf-8")
        self.assertIn("ds_read_b128", text)
        self.assertIn('arch.rfind("gfx90a", 0)', text)
        self.assertNotIn("kittens.cuh", text)
        self.assertNotIn("torch", text.casefold())

    def test_runner_imports_without_touching_gpu(self):
        spec = importlib.util.spec_from_file_location("lds_runner_test", RUNNER)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.PROBE_SOURCE, PROBE)

    def test_runner_writes_prospective_belief_measurement(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"belief_measurements": [{', text)
        self.assertIn('"metric": "cdna3_swizzle_topology_mismatch"', text)
        self.assertIn('"reps_basis": "scored:complete bank-and-phase solver repetitions"', text)


if __name__ == "__main__":
    unittest.main()
