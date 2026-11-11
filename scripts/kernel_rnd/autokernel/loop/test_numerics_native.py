"""AKX-ALL-31.NATIVE: host-side judge of the native receipts (no native binary needed)."""
import json
import os
import tempfile
import math
import unittest

from autokernel.loop import numerics_gates as ng
from autokernel.loop import numerics_native as nn


def _line(variant, op_max, op_dot, backend="CPU"):
    r = ng.make_receipt(ng.Op("x", op_max, op_dot))
    r.update(variant=variant, backend=backend)
    return ng.RECEIPT_PREFIX + json.dumps(r)


def _clamp_max(rows):  # what the native clamp control does: inf becomes finite (clamp feeds POOL_2D)
    return nn.GGML_POOL_MAX_OP.max_rows([[max(min(v, 3e38), -3e38) if v == v else v for v in r] for r in rows])


def _good():
    return "\n".join([
        _line("incumbent", nn.GGML_POOL_MAX_OP.max_rows, ng.CLEAN.dot_rows),
        _line("control_clamp_max", _clamp_max, ng.CLEAN.dot_rows),
        _line("control_f16_dot", nn.GGML_POOL_MAX_OP.max_rows, ng.CONTROL_HALF_DOT.dot_rows)]) + "\n"


class NativeJudge(unittest.TestCase):
    def test_good_native_output_passes(self):
        res = nn.judge("noise\n" + _good())
        self.assertTrue(res["ok"], res["problems"])

    def test_missing_receipt_is_failure(self):
        res = nn.judge("")
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["problems"]), 3)

    def test_incumbent_nan_drop_fails(self):
        bad = _good().replace(_line("incumbent", nn.GGML_POOL_MAX_OP.max_rows, ng.CLEAN.dot_rows),
                              _line("incumbent", ng.CONTROL_NAN_DROP.max_rows, ng.CLEAN.dot_rows))
        res = nn.judge(bad)
        self.assertFalse(res["ok"])
        self.assertTrue(any("incumbent_failed_numerics_non_finite" in p for p in res["problems"]))

    def test_control_that_passes_fails_the_row(self):
        txt = "\n".join([_line("incumbent", nn.GGML_POOL_MAX_OP.max_rows, ng.CLEAN.dot_rows),
                         _line("control_clamp_max", nn.GGML_POOL_MAX_OP.max_rows, ng.CLEAN.dot_rows),
                         _line("control_f16_dot", nn.GGML_POOL_MAX_OP.max_rows, ng.CONTROL_HALF_DOT.dot_rows)])
        res = nn.judge(txt)
        self.assertFalse(res["ok"])
        self.assertIn("control_not_red: control_clamp_max passed numerics_non_finite", res["problems"])

    def test_observed_c10_cpu_receipt_passes(self):
        # the 2026-11-11 NATIVE_FAIL incumbent receipt (C10 7f1bd044b CPU), verbatim max_rows
        obs = [6, 6, 6, "nan", "inf", 6, "inf", "inf", -3.40282347e+38]
        r = {"schema": ng.SCHEMA, "variant": "incumbent", "backend": "CPU", "max_rows": obs,
             "dot_rows": ng.CLEAN.dot_rows(*ng.accum_inputs())}
        vs = ng.verdict_from_receipt(r, want_max=nn.ggml_pool_max_reference())
        self.assertTrue(all(v.passed for v in vs), [v.reason for v in vs])
        # ...and the same receipt is (correctly) NOT IEEE
        self.assertFalse(ng.verdict_from_receipt(r)[0].passed)

    def test_ggml_reference_semantics_and_deviations(self):
        self.assertEqual(nn.ggml_pool_max_reference(),
                         ["finite", "finite", "finite", "nan", "+inf", "finite", "+inf", "+inf", "finite"])
        self.assertEqual([d["row"] for d in nn.ieee_deviations()], [1, 2, 7, 8])
        self.assertEqual(nn.judge("")["ieee_deviations"][0]["row"], 1)

    def test_gpu_reference_drops_every_nan(self):
        self.assertEqual(nn.ggml_pool_max_reference(backend="HIP"),
                         ["finite", "finite", "finite", "finite", "+inf", "finite", "+inf", "+inf", "finite"])
        nan, ninf = float("nan"), float("-inf")
        self.assertEqual(nn.gpu_pool_max_row([nan, nan]), -nn.FLT_MAX)
        self.assertEqual(nn.gpu_pool_max_row([ninf, ninf]), -nn.FLT_MAX)
        self.assertEqual(nn.gpu_pool_max_row([1.0, nan]), 1.0)
        self.assertTrue(math.isnan(nn.pool_max_row_for("CPU", [1.0, nan])))
        self.assertEqual(nn.pool_max_row_for("ROCm0", [1.0, nan]), 1.0)

    def test_gpu_receipt_judged_with_gpu_reference(self):
        mx = [nn.gpu_pool_max_row(r) for r in ng.non_finite_inputs()]
        r = {"schema": ng.SCHEMA, "variant": "incumbent", "backend": "HIP", "max_rows": mx,
             "dot_rows": ng.CLEAN.dot_rows(*ng.accum_inputs())}
        vs = ng.verdict_from_receipt(r, want_max=nn.ggml_pool_max_reference(backend="HIP"))
        self.assertTrue(all(v.passed for v in vs), [v.reason for v in vs])
        r["backend"] = "CPU"  # same numbers are wrong under the CPU reference
        self.assertFalse(ng.verdict_from_receipt(r, want_max=nn.ggml_pool_max_reference(backend="CPU"))[0].passed)

    def test_nan_dropping_beyond_ggml_semantics_still_fails(self):
        # a kernel that ALSO drops the tail NaN (row 3) deviates from the ggml reference
        mx = nn.GGML_POOL_MAX_OP.max_rows(ng.non_finite_inputs())
        mx[3] = 6.0
        res = nn.judge(_good().replace(_line("incumbent", nn.GGML_POOL_MAX_OP.max_rows, ng.CLEAN.dot_rows),
                                       _line("incumbent", lambda rows: mx, ng.CLEAN.dot_rows)))
        self.assertFalse(res["ok"])

    def test_control_f16_dot_only_red_on_accum(self):
        res = nn.judge(_good())
        g = {x["gate"]: x["passed"] for x in res["variants"]["control_f16_dot"]}
        self.assertEqual(g, {ng.GATE_NON_FINITE: True, ng.GATE_FP32_ACCUM: False})

    def test_inputs_file_round_trips_header_and_specials(self):
        t = nn.inputs_text().split("\n")
        self.assertEqual(t[0], "MAX 9 64")
        self.assertIn("nan", t[2].split())
        self.assertIn("-inf", t[9].split())
        self.assertEqual(t[10], "DOT 8 4096")
        self.assertEqual(len(t), 1 + 9 + 1 + 16 + 1)

    def test_write_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "in.txt")
            nn.write_inputs(p)
            self.assertGreater(os.path.getsize(p), 1000)


if __name__ == "__main__":
    unittest.main()
