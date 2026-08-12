import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ..evaluator import statistics
from . import live_controls


class InstrumentSelection(unittest.TestCase):

    def tearDown(self):
        importlib.reload(live_controls)

    def test_clean_instrument_worktree_can_be_selected_explicitly(self):
        with mock.patch.dict(os.environ, {
                "AUTOKERNEL_INSTRUMENT_ROOT": "/tmp/ak-clean-instrument"}, clear=False):
            module = importlib.reload(live_controls)
        self.assertEqual(str(module.INSTRUMENT_ROOT), "/tmp/ak-clean-instrument")
        self.assertEqual(
            str(module.INSTRUMENT_BINARY),
            "/tmp/ak-clean-instrument/build-v9-cpu/bin/llama-bench")

    def test_binary_override_is_separate_and_explicit(self):
        with mock.patch.dict(os.environ, {
                "AUTOKERNEL_INSTRUMENT_ROOT": "/tmp/ak-clean-instrument",
                "AUTOKERNEL_INSTRUMENT_BINARY": "/tmp/ak-build/bin/llama-bench",
        }, clear=False):
            module = importlib.reload(live_controls)
        self.assertEqual(str(module.INSTRUMENT_BINARY), "/tmp/ak-build/bin/llama-bench")


class InstrumentCapability(unittest.TestCase):

    def test_requires_every_binding_hardening_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-bench"
            binary.write_bytes(b"launcher")
            library = Path(tmp) / "libllama-bench-impl.so"
            library.write_bytes(b"\0".join(live_controls.REQUIRED_HARDENING_RECEIPTS))
            check = live_controls._instrument_receipt_capability(binary)
        self.assertEqual(check.outcome, "PASS")

    def test_names_missing_receipts_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-bench"
            binary.write_bytes(b"autokernel_hybrid_ab_complete")
            check = live_controls._instrument_receipt_capability(binary)
        self.assertEqual(check.outcome, "FAIL")
        self.assertIn("autokernel_device_sync_mode", check.reasons[0])


class ControlEffectReachability(unittest.TestCase):

    def test_predeclared_control_window_crosses_where_base_segment_cannot(self):
        rule = live_controls._control_stopping_rule()
        self.assertEqual(rule.max_total_blocks(5), 10)
        construction = statistics.select_construction(
            "sign_martingale_predictable_lambda/v1")
        base = statistics.run_e_process(
            (0.08,) * 5, construction=construction,
            hypothesis=statistics.HYPOTHESIS_IMPROVEMENT,
            margin=0.0, threshold=10.0)
        full = statistics.run_e_process(
            (0.08,) * rule.max_total_blocks(5), construction=construction,
            hypothesis=statistics.HYPOTHESIS_IMPROVEMENT,
            margin=0.0, threshold=10.0)
        self.assertLess(base.e_running_max, 10.0)
        self.assertGreaterEqual(full.e_running_max, 10.0)
        self.assertEqual(full.first_crossing_block, 7)


if __name__ == "__main__":
    unittest.main()
