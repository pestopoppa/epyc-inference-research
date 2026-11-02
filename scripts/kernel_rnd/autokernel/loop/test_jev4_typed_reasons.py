"""JEV-4: typed resume ineligibility + critic asked/answered/unavailable record."""

from __future__ import annotations

import unittest

from autokernel.loop import loop, resume


class IneligibleCodeTest(unittest.TestCase):
    def test_every_return_is_typed_and_still_a_string(self):
        r = resume.Ineligible(resume.IneligibleCode.NO_RETAINED_PATCH, "no retained patch")
        self.assertIsInstance(r, str)
        self.assertEqual(r, "no retained patch")
        self.assertIn("retained", r)
        self.assertIs(r.code, resume.IneligibleCode.NO_RETAINED_PATCH)
        self.assertEqual(r.to_dict(), {"code": "no_retained_patch", "text": "no retained patch"})
        self.assertEqual(resume.ineligible_code(r), "no_retained_patch")
        self.assertIsNone(resume.ineligible_code(None))

    def test_no_untyped_return_in_ineligible_reason(self):
        import inspect
        src = inspect.getsource(resume.ineligible_reason)
        returns = [l.strip() for l in src.splitlines() if l.strip().startswith("return ")]
        self.assertTrue(returns)
        for line in returns:
            self.assertTrue(line == "return None" or line.startswith("return Ineligible("), line)

    def test_enum_is_closed(self):
        self.assertEqual(len(resume.IneligibleCode), 9)


class CriticCallRecordTest(unittest.TestCase):
    def test_unavailable_critic_is_recorded_not_a_verdict(self):
        call = {"critic": "critic:patch", "state": "unavailable", "attempts": 1, "error": "x"}
        row = loop.Outcome("planner_transient", None, ["x"], critic_call=call).to_attempt()
        self.assertEqual(row["critic_call"], call)
        self.assertIsNone(row.get("refusal_gate"))

    def test_absent_when_no_critic_failed(self):
        row = loop.Outcome("planner_transient", None, ["x"]).to_attempt()
        self.assertNotIn("critic_call", row)


if __name__ == "__main__":
    unittest.main()
