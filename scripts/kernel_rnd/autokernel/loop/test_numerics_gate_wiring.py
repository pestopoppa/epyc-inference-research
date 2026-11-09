"""AKX-ALL-31.WIRE: the numerics chain through the real `op_correctness` path."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import gates, guard_bands, numerics_gates as NG

CONSOLE = "Backend 1/1: CPU\n  3/3 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n"


def receipt_line(op):
    return NG.RECEIPT_PREFIX + json.dumps(NG.make_receipt(op), allow_nan=False) + "\n"


class Wiring(unittest.TestCase):
    def run_gate(self, extra="", flag=None):
        env = {k: v for k, v in os.environ.items() if k != NG.ENV_FLAG}
        if flag is not None:
            env[NG.ENV_FLAG] = flag
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "bin" / "test-backend-ops"
            binary.parent.mkdir()
            binary.write_text("")
            done = subprocess.CompletedProcess([], 0, stdout=CONSOLE + extra, stderr="")
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(gates.subprocess, "run", return_value=done):
                return gates.op_correctness(Path(tmp), op="MUL_MAT", backend="CPU")

    def test_default_off_ignores_absent_receipts(self):
        self.assertTrue(self.run_gate().passed)

    def test_default_off_does_not_need_numpy(self):
        with mock.patch.dict("sys.modules", {"numpy": None}):
            self.assertTrue(self.run_gate().passed)

    def test_a_planted_nan_dropping_candidate_is_refused(self):
        verdict = self.run_gate(receipt_line(NG.CONTROL_NAN_DROP), flag="1")
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.gate, "correctness")
        self.assertIn(NG.GATE_NON_FINITE, verdict.reason)

    def test_a_half_precision_dot_is_refused(self):
        verdict = self.run_gate(receipt_line(NG.CONTROL_HALF_DOT), flag="1")
        self.assertFalse(verdict.passed)
        self.assertIn(NG.GATE_FP32_ACCUM, verdict.reason)

    def test_the_clean_candidate_passes(self):
        self.assertTrue(self.run_gate(receipt_line(NG.CLEAN), flag="1").passed)

    def test_a_missing_receipt_is_an_oracle_fault_not_a_patch_failure(self):
        verdict = self.run_gate(flag="1")
        self.assertEqual((verdict.passed, verdict.gate), (False, "oracle_unavailable"))

    def test_a_passing_control_voids_the_gate(self):
        red = gates.Verdict(NG.GATE_CONTROLS, False, "planted_control_not_red: x")
        with mock.patch.object(NG, "controls_gate", return_value=red):
            verdict = self.run_gate(receipt_line(NG.CLEAN), flag="1")
        self.assertEqual((verdict.passed, verdict.gate), (False, "oracle_unavailable"))

    def test_a_touched_guard_band_is_refused_when_the_chain_is_on(self):
        band = {"schema": guard_bands.SCHEMA, "views": [{"name": "dst", "extent": 8, "offset": 0}],
                "violations": [{f: 0 for f in guard_bands.Violation.__dataclass_fields__}]}
        extra = receipt_line(NG.CLEAN) + guard_bands.RECEIPT_PREFIX + json.dumps(band) + "\n"
        verdict = self.run_gate(extra, flag="1")
        self.assertEqual((verdict.passed, verdict.gate), (False, "correctness"))
        self.assertIn(guard_bands.GATE, verdict.reason)


if __name__ == "__main__":
    unittest.main()
