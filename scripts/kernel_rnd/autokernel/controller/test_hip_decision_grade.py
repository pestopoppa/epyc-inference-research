from __future__ import annotations

from array import array
import hashlib
from pathlib import Path
import tempfile
import unittest

from . import hip_decision_grade as H


class HipDecisionGradeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_suite_is_replayable_hostile_and_not_generated_until_called(self):
        first, expected_a = H._create_suite(self.work, "a" * 64)
        self.assertEqual(len(first["cases"]), 24)
        self.assertEqual({row["distribution"] for row in first["cases"]}, {
            "baseline", "alternating", "sparse_outlier", "cancellation"})
        self.assertTrue(first["suite_seed_generated_after_candidate_seal"])
        self.assertEqual(first["timing_blocks"], 20)
        self.assertEqual(first["repetitions_per_arm"], 30_000)
        self.assertEqual(len(expected_a), 24)
        replay_work = self.root / "replay"
        replay_work.mkdir()
        replay, expected_b = H._create_suite(replay_work, "a" * 64)
        self.assertEqual(
            [row["input_sha256"] for row in first["cases"]],
            [row["input_sha256"] for row in replay["cases"]])
        self.assertEqual(expected_a, expected_b)

    def test_host_double_silu_is_stable_at_extremes(self):
        self.assertEqual(H._silu_host_double(1000.0), 1000.0)
        self.assertEqual(H._silu_host_double(-1000.0), -0.0)
        self.assertAlmostEqual(H._silu_host_double(0.0), 0.0)

    def test_static_c6_scan_accepts_candidate_and_detects_environment_probe(self):
        source = self.root / "candidate.hip"
        source.write_text("void forward() { int value = 1; }\n")
        self.assertTrue(H._scan_candidate(source)["clean"])
        source.write_text("void forward() { auto p = getenv(\"MODE\"); }\n")
        scan = H._scan_candidate(source)
        self.assertFalse(scan["clean"])
        self.assertTrue(scan["environment_probe_findings"])

    def test_correctness_reducer_requires_both_poisons_bitwise_equal(self):
        inputs = self.work / "inputs"
        outputs = self.work / "outputs"
        inputs.mkdir()
        outputs.mkdir()
        values = [0.0, 1.0, -1.0]
        H._write_f32(inputs / "x.f32", values)
        oracle = tuple(H._silu_host_double(v) for v in H._read_f32(inputs / "x.f32"))
        H._write_f32(outputs / "x-a.f32", oracle)
        H._write_f32(outputs / "x-b.f32", oracle)
        specification = {"cases": [{"case_id": "x"}]}
        child = {"cases": [{
            "case_id": "x", "input_file_unchanged": True,
            "device_input_unchanged": True,
            "output_a": "outputs/x-a.f32", "output_b": "outputs/x-b.f32",
        }]}
        reduced = H._reduce_correctness(
            work=self.work, specification=specification,
            expected={"x": oracle}, child=child)
        self.assertTrue(reduced["all_passed"])
        H._write_f32(outputs / "x-b.f32", [0.0, 0.0, 0.0])
        reduced = H._reduce_correctness(
            work=self.work, specification=specification,
            expected={"x": oracle}, child=child)
        self.assertFalse(reduced["all_passed"])

    def test_timing_reducer_uses_anytime_valid_bundle_construction(self):
        child = {
            "provider": {"provider_id": "torch_rocm_compile"},
            "repetitions_per_arm": 30_000,
            "blocks": [{
                "block_index": index,
                "order": "candidate_first" if index % 2 else "anchor_first",
                "candidate_ns": 80.0, "anchor_ns": 100.0,
                "candidate_measured_duration_ns": 300_000_000.0,
                "anchor_measured_duration_ns": 300_000_000.0,
            } for index in range(20)],
        }
        result = H._reduce_timing(child)
        self.assertTrue(result["candidate_beats_exact_provider"])
        self.assertEqual(result["e_process"]["construction_id"], H.CONSTRUCTION_ID)
        self.assertEqual(result["e_process"]["threshold"], 20.0)

    def test_sub_floor_timing_cannot_be_rankable(self):
        child = {
            "provider": {"provider_id": "torch_rocm_compile"},
            "repetitions_per_arm": 200,
            "blocks": [{
                "block_index": index, "order": "anchor_first",
                "candidate_ns": 80.0, "anchor_ns": 100.0,
                "candidate_measured_duration_ns": 2_000_000.0,
                "anchor_measured_duration_ns": 2_000_000.0,
            } for index in range(20)],
        }
        result = H._reduce_timing(child)
        self.assertFalse(result["candidate_beats_exact_provider"])
        self.assertEqual(
            result["ranked_duration_admission"]["all_arms_passed"], False)
        self.assertEqual(
            result["ranked_duration_admission"]["checks"][0]["outcome"],
            "COULD_NOT_CHECK")


if __name__ == "__main__":
    unittest.main()
