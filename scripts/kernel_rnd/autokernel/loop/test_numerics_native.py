"""AKX-ALL-31.NATIVE: host-side judge of the native receipts (no native binary needed)."""
import json
import os
import tempfile
import unittest

from autokernel.loop import numerics_gates as ng
from autokernel.loop import numerics_native as nn


def _line(variant, op_max, op_dot, backend="CPU"):
    r = ng.make_receipt(ng.Op("x", op_max, op_dot))
    r.update(variant=variant, backend=backend)
    return ng.RECEIPT_PREFIX + json.dumps(r)


def _clamp_max(rows):  # what the native clamp control does: inf becomes finite
    return [max(min(v, 3e38), -3e38) for v in ng.CLEAN.max_rows(rows)]


def _good():
    return "\n".join([
        _line("incumbent", ng.CLEAN.max_rows, ng.CLEAN.dot_rows),
        _line("control_clamp_max", _clamp_max, ng.CLEAN.dot_rows),
        _line("control_f16_dot", ng.CLEAN.max_rows, ng.CONTROL_HALF_DOT.dot_rows)]) + "\n"


class NativeJudge(unittest.TestCase):
    def test_good_native_output_passes(self):
        res = nn.judge("noise\n" + _good())
        self.assertTrue(res["ok"], res["problems"])

    def test_missing_receipt_is_failure(self):
        res = nn.judge("")
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["problems"]), 3)

    def test_incumbent_nan_drop_fails(self):
        bad = _good().replace(_line("incumbent", ng.CLEAN.max_rows, ng.CLEAN.dot_rows),
                              _line("incumbent", ng.CONTROL_NAN_DROP.max_rows, ng.CLEAN.dot_rows))
        res = nn.judge(bad)
        self.assertFalse(res["ok"])
        self.assertTrue(any("incumbent_failed_numerics_non_finite" in p for p in res["problems"]))

    def test_control_that_passes_fails_the_row(self):
        txt = "\n".join([_line("incumbent", ng.CLEAN.max_rows, ng.CLEAN.dot_rows),
                         _line("control_clamp_max", ng.CLEAN.max_rows, ng.CLEAN.dot_rows),
                         _line("control_f16_dot", ng.CLEAN.max_rows, ng.CONTROL_HALF_DOT.dot_rows)])
        res = nn.judge(txt)
        self.assertFalse(res["ok"])
        self.assertIn("control_not_red: control_clamp_max passed numerics_non_finite", res["problems"])

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
