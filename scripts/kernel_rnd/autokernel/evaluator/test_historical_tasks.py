from __future__ import annotations

import unittest

from .. import schemas
from . import historical_tasks as H


def descriptor(**changes):
    value = {
        "schema": H.SCHEMA, "task_id": "gdn-bf16-b32-v1",
        "source_repo": "/mnt/raid0/llm/llama.cpp-experimental",
        "parent_commit": "7c28056b78d939f7b34c2a4350ea26a88442d08c",
        "expert_commit": "496e2f098a332d3f40fe9b5e7ff56f9efc1d52b6",
        "expert_authored_at": "2026-07-04T23:46:28Z",
        "holdout_mode": H.SEALED_HOLDOUT,
        "actor_source_commit": "7c28056b78d939f7b34c2a4350ea26a88442d08c",
        "expert_visibility": "sealed_until_terminal",
        "model_path": "/models/model.gguf", "model_sha256": "a" * 64,
        "benchmark_argv": ["llama-batched-bench", "-m", "{model}", "-c", "8192",
                           "-b", "2048", "-ub", "512", "-ngl", "99", "-fa", "0",
                           "-npp", "128", "-ntg", "128", "-npl", "32",
                           "--output-format", "jsonl"],
        "metric_id": "speed_tg", "metric_direction": "higher_is_better",
        "minimum_repeats": 3, "evidence_refs": ["git://expert"],
        "historical_command_recovered": False,
    }
    value.update(changes)
    return value


class TestHistoricalDescriptor(unittest.TestCase):
    def test_sealed_exact_surface_constructs_and_hashes(self):
        item = H.HistoricalTaskDescriptor.from_dict(descriptor())
        self.assertEqual(item.actor_source_commit, item.parent_commit)
        self.assertEqual(len(item.canonical_sha256()), 64)

    def test_expert_visibility_and_parent_source_are_load_bearing(self):
        with self.assertRaisesRegex(ValueError, "pre-optimization parent"):
            H.HistoricalTaskDescriptor.from_dict(
                descriptor(actor_source_commit="496e2f098a332d3f40fe9b5e7ff56f9efc1d52b6"))
        with self.assertRaisesRegex(ValueError, "sealed until"):
            H.HistoricalTaskDescriptor.from_dict(
                descriptor(expert_visibility="visible"))

    def test_missing_exact_surface_or_recovered_command_claim_refuses(self):
        argv = descriptor()["benchmark_argv"][:-2]
        with self.assertRaisesRegex(ValueError, "full exact surface"):
            H.HistoricalTaskDescriptor.from_dict(descriptor(benchmark_argv=argv))
        with self.assertRaisesRegex(ValueError, "original command was not recovered"):
            H.HistoricalTaskDescriptor.from_dict(
                descriptor(historical_command_recovered=True))


class TestExpertCeiling(unittest.TestCase):
    def test_candidate_reports_both_floor_and_ceiling(self):
        report = H.score_expert_ceiling(
            baseline_samples=(100.0, 101.0, 99.0),
            expert_samples=(120.0, 121.0, 119.0),
            candidate_samples=(110.0, 111.0, 109.0), minimum_repeats=3)
        self.assertEqual(report.check.outcome, schemas.PASS)
        self.assertAlmostEqual(report.expert_gain_pct, 20.0)
        self.assertAlmostEqual(report.candidate_gain_pct, 10.0)
        self.assertAlmostEqual(report.expert_fraction_recovered, 0.5)

    def test_archive_without_candidate_cannot_claim_expert_scoring(self):
        report = H.score_expert_ceiling(
            baseline_samples=(100.0, 101.0, 99.0),
            expert_samples=(120.0, 121.0, 119.0),
            candidate_samples=None, minimum_repeats=3)
        self.assertEqual(report.check.outcome, schemas.COULD_NOT_CHECK)

    def test_non_improving_human_patch_fails(self):
        report = H.score_expert_ceiling(
            baseline_samples=(100.0, 101.0, 99.0),
            expert_samples=(90.0, 91.0, 89.0),
            candidate_samples=(95.0, 96.0, 94.0), minimum_repeats=3)
        self.assertEqual(report.check.outcome, schemas.FAIL)

    def test_short_or_nonpositive_samples_refuse(self):
        with self.assertRaisesRegex(ValueError, "baseline requires"):
            H.score_expert_ceiling(
                baseline_samples=(100.0, 101.0), expert_samples=(120.0,) * 3,
                candidate_samples=None, minimum_repeats=3)


if __name__ == "__main__":
    unittest.main()
