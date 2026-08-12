from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "run_autokernel_wgm_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_autokernel_wgm_sweep", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class WgmSweepTests(unittest.TestCase):
    def test_logical_mapping_is_a_bijection(self) -> None:
        expected = {(m, n) for m in range(64) for n in range(64)}
        for factor in runner.FACTORS:
            with self.subTest(factor=factor):
                observed = {
                    runner.logical_mapping(factor, linear)
                    for linear in range(64 * 64)
                }
                self.assertEqual(observed, expected)

    def test_parse_samples_refuses_unbalanced_capture(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "raw.jsonl"
            rows = [{"type": "header", "correctness": "bit_exact"}]
            rows.extend(
                {"type": "sample", "factor": factor, "elapsed_ms": 1.0}
                for factor in runner.FACTORS[:-1]
            )
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(RuntimeError, "expected 6 samples"):
                runner.parse_samples(path, rounds=1)

    def test_summary_uses_lower_is_better(self) -> None:
        rows = []
        for factor in runner.FACTORS:
            value = 5.0 if factor == 8 else 10.0 + factor
            rows.extend(
                {"round": round_id, "factor": factor, "elapsed_ms": value}
                for round_id in range(3)
            )
        result = runner.summarize(rows)
        self.assertEqual(result["direction"], "lower_is_better")
        self.assertEqual(result["best_factor"], 8)
        self.assertEqual(result["factors"]["8"]["relative_to_none_pct"], 100.0)
        self.assertEqual(result["factors"]["8"]["bootstrap_best_frequency"], 1.0)

    def test_runner_owns_governance_and_proxy_labels(self) -> None:
        text = MODULE_PATH.read_text()
        self.assertIn("acquire_device_claim", text)
        self.assertIn("RocmSmiSampler", text)
        self.assertIn("rocprofv2", text)
        self.assertIn('"diagnostic_only"', text)
        self.assertIn('"standalone_l2_tile_reuse_proxy_not_mmq"', text)
        source = runner.SOURCE.read_text()
        self.assertIn("refusing non-gfx90a/wave64", source)
        self.assertNotIn("chiplet", source.lower())


if __name__ == "__main__":
    unittest.main()
