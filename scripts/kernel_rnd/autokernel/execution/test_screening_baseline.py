import json
import tempfile
import unittest
from pathlib import Path

from . import screening_baseline as bank


class ScreeningBaselineBankTest(unittest.TestCase):
    def test_one_bank_admits_many_candidate_screens_without_new_anchors(self):
        value = bank.BaselineBank({"recipe": "decode", "model": "m", "instrument": "i"},
                                  (100.0, 101.0, 99.0), 100.0)
        for _ in range(8):
            value.admit({"recipe": "decode", "model": "m", "instrument": "i"})
        self.assertEqual(len(value.anchor_samples), 3)

    def test_frame_drift_and_closed_sentinel_refuse(self):
        value = bank.BaselineBank({"recipe": "decode"}, (1.0, 1.0), 1.0)
        with self.assertRaisesRegex(bank.BaselineBankError, "frame differs"):
            value.admit({"recipe": "prefill"})
        closed = bank.BaselineBank({"recipe": "decode"}, (1.0, 1.0), 1.0, 1.0)
        with self.assertRaisesRegex(bank.BaselineBankError, "closed"):
            closed.admit({"recipe": "decode"})

    def test_hash_tamper_refuses(self):
        value = bank.BaselineBank({"recipe": "decode"}, (1.0, 1.0), 1.0).to_dict()
        value["anchor_samples"] = [9.0, 9.0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(bank.BaselineBankError, "hash"):
                bank.load(path)

    def test_nomination_reports_uncertainty_not_a_keep(self):
        value = bank.BaselineBank({"recipe": "decode"}, (100.0, 102.0), 100.0)
        report = value.nominate((104.0, 105.0, 106.0))
        self.assertEqual(report["nomination"], "top_k_candidate_only_not_a_keep")
        self.assertIn("nonpromotable", report["uncertainty"])


if __name__ == "__main__":
    unittest.main()
