"""M2 scorer-profile and predecessor evidence tests; no model or GPU needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.benchmark import v11_fa4b_quality_m2 as m


class QualityM2Tests(unittest.TestCase):
    def test_actual_failed_corpus_item_has_native_oracle(self):
        base = Path("/mnt/raid0/llm/tmp/codex-ni-20261027-fa4-sixkey-diagnostic/model-quality-row-prep")
        old = m.read_json(base / "MANIFEST.json")
        item = old["scored"][100]
        self.assertEqual(item["id"], "leetcode_find-the-index-of-the-first-occurrence-in-a-string")
        self.assertEqual(item["scoring_config"]["entry_point"], "strStr")
        self.assertEqual(len(item["scoring_config"]["entry_point_cases"]), 2)
        self.assertEqual(m.reference_errors(old["scored"], m.read_json(m.ORACLE)), {})
        from answer_scoring import score_response_or_error
        self.assertEqual(score_response_or_error("def strStr(haystack, needle): return haystack.find(needle)",
                                                 item["expected"], item), (None, "no_reference"))
        native = m._native()
        correct = "def strStr(haystack, needle):\n    return haystack.find(needle)"
        mutated = "def strStr(haystack, needle):\n    return -1"
        self.assertEqual(native.score_answer_or_error(correct, item["expected"], "code_execution",
                                                      item["scoring_config"]), (True, None))
        self.assertEqual(native.score_answer_or_error(mutated, item["expected"], "code_execution",
                                                      item["scoring_config"]), (False, None))

    def test_missing_reference_classified_before_serving(self):
        base = Path("/mnt/raid0/llm/tmp/codex-ni-20261027-fa4-sixkey-diagnostic/model-quality-row-prep")
        old = m.read_json(base / "MANIFEST.json")
        item = copy.deepcopy(old["scored"][100])
        item["scoring_config"]["entry_point_cases"] = []
        errors = m.reference_errors([item], m.read_json(m.ORACLE))
        self.assertEqual(errors[item["id"]], "no_reference:entry_point_cases")

    def test_native_code_scoring_and_undecidable_record(self):
        base = Path("/mnt/raid0/llm/tmp/codex-ni-20261027-fa4-sixkey-diagnostic/model-quality-row-prep")
        q = m.read_json(base / "MANIFEST.json")["scored"][100]
        response = {"choices": [{"finish_reason": "length", "message": {"content":
                    "def strStr(haystack, needle):\n    return haystack.find(needle)"}}]}
        result = m.parse_response("score", response, q)
        self.assertIs(result["correct"], True)
        self.assertIsNone(result["undecidable_reason"])
        self.assertTrue(result["truncated"])
        with patch.object(m._native(), "score_answer_or_error", return_value=(None, "scoring_unavailable: timeout")):
            outcome = m.parse_response("score", response, q)
        self.assertIsNone(outcome["correct"])
        self.assertEqual(outcome["undecidable_reason"], "scoring_unavailable: timeout")

    def test_predecessor_manifest_and_first_hundred_records_still_validate(self):
        base = Path("/mnt/raid0/llm/tmp/codex-ni-20261027-fa4-sixkey-diagnostic/model-quality-row-prep")
        old_path = base / "MANIFEST.json"
        old = m.read_json(old_path)
        self.assertEqual(m.file_sha(old_path),
                         "7cec33127e4162b89610d4e8b66eb362fb289e3fde1dadbc20e2834d36f13a51")
        self.assertEqual(old["seal_sha256"], "646a43a17d5138de2402db1bac6f6443a25f63e7886572a81dff3d2b5dd4a590")
        m.m1.validate_manifest(old)
        with tempfile.TemporaryDirectory(dir=base / "tmp") as d:
            result = m.freeze_successor(old_path, base / "out", Path(d) / "M2.json")
            self.assertEqual(len(result["predecessor"]["links"]), 100)
            m.validate_manifest(result)
            q = old["scored"][99]
            row = m.historical_row(result, "score", "ON", "q8_0", "off", q)
            self.assertEqual(row["index"], 99)
            self.assertEqual(row["manifest_sha256"], old["seal_sha256"])
            self.assertIsNone(m.historical_row(result, "score", "OFF", "q8_0", "off", q))
            poisoned = copy.deepcopy(result)
            poisoned["predecessor"]["links"][q["id"]]["sha256"] = "0" * 64
            poisoned["seal_sha256"] = m.digest({k: v for k, v in poisoned.items() if k != "seal_sha256"})
            with self.assertRaisesRegex(ValueError, "historical record file drift"):
                m.validate_manifest(poisoned)

    def test_full_paired_gate_refuses_one_undecidable_without_false_denominator(self):
        ids = [f"q{i:03d}" for i in range(200)]
        scored = {arm: {qid: {"correct": True, "undecidable_reason": None} for qid in ids}
                  for arm in m.ARMS}
        full = m.paired_quality(scored, ids, 42)
        self.assertEqual(full["n_decidable_pairs"], 200)
        self.assertEqual(full["ci95"], [0.0, 0.0])
        self.assertTrue(full["agreement"])
        scored["OFF"][ids[17]] = {"correct": None, "undecidable_reason": "scoring_unavailable: checker error"}
        reduced = m.paired_quality(scored, ids, 42)
        self.assertEqual(reduced["n_decidable_pairs"], 199)
        self.assertEqual(reduced["status"], "INCONCLUSIVE_INSUFFICIENT_DECIDABLE_PAIRS")
        self.assertIsNone(reduced["agreement"])
        self.assertNotIn("ci95", reduced)
        self.assertEqual(reduced["undecidable"]["OFF"][ids[17]], "scoring_unavailable: checker error")


if __name__ == "__main__":
    unittest.main()
