"""AKX-ALL-31: planted-control numerics gates (CPU/host only)."""
import unittest

from autokernel.loop import numerics_gates as ng


class NumericsGateTests(unittest.TestCase):
    def test_clean_passes_both(self):
        self.assertTrue(ng.non_finite_gate(ng.CLEAN).passed)
        self.assertTrue(ng.fp32_accum_gate(ng.CLEAN).passed)

    def test_nan_dropping_fmax_control_is_red(self):
        v = ng.non_finite_gate(ng.CONTROL_NAN_DROP)
        self.assertFalse(v.passed)
        self.assertIn("non_finite_class_mismatch", v.reason)

    def test_half_precision_dot_control_is_red(self):
        v = ng.fp32_accum_gate(ng.CONTROL_HALF_DOT)
        self.assertFalse(v.passed)
        self.assertIn("accumulator_not_exact", v.reason)

    def test_inf_to_nan_and_inf_sign_flip_mutants_are_red(self):
        flip = ng.Op("flip", lambda rows: [-v if v == ng.PINF else v for v in ng.CLEAN.max_rows(rows)],
                     ng.CLEAN.dot_rows)
        self.assertFalse(ng.non_finite_gate(flip).passed)
        clamp = ng.Op("clamp", lambda rows: [3.4e38 if v == ng.PINF else v for v in ng.CLEAN.max_rows(rows)],
                      ng.CLEAN.dot_rows)
        self.assertFalse(ng.non_finite_gate(clamp).passed)

    def test_bf16_dot_mutant_is_red(self):
        bf = ng.Op("bf", ng.CLEAN.max_rows,
                   lambda A, B: [ng._replay_dot(a, b, ng.to_bf16) for a, b in zip(A, B)])
        self.assertFalse(ng.fp32_accum_gate(bf).passed)

    def test_power_proof_holds_and_weak_inputs_refuse(self):
        A, B = ng.accum_inputs()
        proof = ng.accumulator_power_proof(A, B)
        self.assertTrue(proof["ok"])
        self.assertGreaterEqual(proof["fp16_min_ratio"], 100)
        self.assertGreaterEqual(proof["bf16_min_ratio"], 100)
        v = ng.fp32_accum_gate(ng.CLEAN, [[1.0, 2.0]], [[1.0, 1.0]])
        self.assertFalse(v.passed)
        self.assertIn("lack_power", v.reason)

    def test_controls_gate_green_and_fails_when_a_control_passes(self):
        self.assertTrue(ng.controls_gate().passed)
        saved = ng.CONTROL_NAN_DROP
        try:
            ng.CONTROL_NAN_DROP = ng.CLEAN          # a control that passes
            v = ng.controls_gate()
            self.assertFalse(v.passed)
            self.assertIn("planted_control_not_red", v.reason)
            self.assertEqual([x.passed for x in ng.chain_verdicts(ng.CLEAN, flag=True)], [False])
        finally:
            ng.CONTROL_NAN_DROP = saved

    def test_opt_in_default_off(self):
        self.assertFalse(ng.NUMERICS_GATES_DEFAULT)
        self.assertEqual(ng.chain_verdicts(ng.CONTROL_NAN_DROP, environ={}), [])
        self.assertTrue(ng.enabled(environ={ng.ENV_FLAG: "1"}))
        self.assertFalse(ng.enabled(flag=False, environ={ng.ENV_FLAG: "1"}))

    def test_chain_enabled(self):
        self.assertTrue(all(v.passed for v in ng.chain_verdicts(ng.CLEAN, flag=True)))
        res = ng.chain_verdicts(ng.CONTROL_NAN_DROP, flag=True)
        self.assertFalse(all(v.passed for v in res))
        self.assertFalse(ng.chain_verdicts(None, flag=True)[-1].passed)

    def test_receipts(self):
        import json
        text = "log " + ng.RECEIPT_PREFIX + json.dumps(ng.make_receipt(ng.CLEAN)) + "\n"
        (r,) = ng.parse_receipts(text)
        self.assertTrue(all(v.passed for v in ng.verdict_from_receipt(r)))
        (b,) = ng.parse_receipts(ng.RECEIPT_PREFIX + json.dumps(ng.make_receipt(ng.CONTROL_NAN_DROP)))
        self.assertFalse(all(v.passed for v in ng.verdict_from_receipt(b)))
        self.assertFalse(ng.verdict_from_receipt(None)[0].passed)


if __name__ == "__main__":
    unittest.main()
