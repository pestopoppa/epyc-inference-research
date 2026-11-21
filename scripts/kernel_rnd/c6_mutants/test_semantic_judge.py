"""CPU-only tests for the RVP-C6-19 semantic-judge plumbing."""
import ast
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import semantic_judge as sj  # noqa: E402

J = {"model": "m", "temperature": 0.0, "seed": 1, "run_label": "t"}
T = ("layernorm_no_affine", "softmax_no_maxsub", "matmul_transpose_no_t")


class RubricTests(unittest.TestCase):
    def test_rubric_loads_and_pins(self):
        r = sj.load_rubric()
        self.assertEqual(len(r["rules"]), 5)
        self.assertIn("No Omission of Necessary Computations", r["instruction_text"])
        self.assertIn("2510.17891", r["source"]["arxiv"])

    def test_tampered_rubric_refused(self):
        r = json.loads(sj.RUBRIC_PATH.read_text())
        r["instruction_text"] += " x"
        p = Path(__file__).parent / "_tmp_rubric_tamper.json"
        p.write_text(json.dumps(r))
        try:
            with self.assertRaises(sj.JudgeError):
                sj.load_rubric(p)
        finally:
            p.unlink()

    def test_prompt_contains_rubric_and_kernel(self):
        p = sj.build_prompt("def k(): pass")
        self.assertIn("Detect whether the following Triton kernel", p)
        self.assertIn("def k(): pass", p)
        with self.assertRaises(sj.JudgeError):
            sj.build_prompt("  ")


class VerdictTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(sj.parse_verdict("why\nVERDICT: CHEATING\n"), "REJECT")
        self.assertEqual(sj.parse_verdict("VERDICT: OK"), "ACCEPT")

    def test_unparseable_refused(self):
        for bad in ("", "looks fine", "VERDICT: OK\nmore", "VERDICT: MAYBE"):
            with self.assertRaises(sj.JudgeError):
                sj.parse_verdict(bad)


class SampleTests(unittest.TestCase):
    def test_samples_parse_and_differ_only_in_candidate(self):
        s = sj.build_samples()
        self.assertEqual(tuple(s), T)
        for task, d in s.items():
            ast.parse(d["mutant"])
            ast.parse(d["honest"])
            self.assertNotEqual(d["mutant"], d["honest"])
            self.assertIn("@triton.jit", d["mutant"])
        self.assertIn("_layernorm_launch(x, weight, bias, False)",
                      s["layernorm_no_affine"]["mutant"])
        self.assertIn("_matmul_launch(a, b, True)",
                      s["matmul_transpose_no_t"]["honest"])


class RecordTests(unittest.TestCase):
    def rec(self, m, c):
        return sj.build_record(m, c, J)

    def test_gates_only_when_all_rejected_and_controls_accepted(self):
        acc = {t: "ACCEPT" for t in T}
        rej = {t: "REJECT" for t in T}
        self.assertTrue(self.rec(rej, acc)["gating"])
        self.assertFalse(self.rec({**rej, T[0]: "ACCEPT"}, acc)["gating"])

    def test_reject_everything_judge_is_vacuous(self):
        rej = {t: "REJECT" for t in T}
        r = self.rec(rej, rej)
        self.assertFalse(r["gating"])
        self.assertEqual(r["reason"], "vacuous_judge_rejects_honest_control")

    def test_missing_sampling_pin_or_controls_refused(self):
        acc = {t: "ACCEPT" for t in T}
        with self.assertRaises(sj.JudgeError):
            sj.build_record(acc, acc, {"model": "m"})
        with self.assertRaises(sj.JudgeError):
            sj.build_record(acc, {T[0]: "ACCEPT"}, J)


if __name__ == "__main__":
    unittest.main()
