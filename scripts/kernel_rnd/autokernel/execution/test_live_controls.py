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


class CampaignIdentity(unittest.TestCase):

    def test_distinct_campaigns_have_distinct_seeds_and_windows(self):
        first = live_controls.LiveCampaignIdentity(
            "ak-controls-v9-first", "/tmp/ak-controls-v9-first")
        second = live_controls.LiveCampaignIdentity(
            "ak-controls-v9-second", "/tmp/ak-controls-v9-second")
        self.assertNotEqual(first.campaign_seed, second.campaign_seed)
        self.assertNotEqual(first.window_id, second.window_id)
        self.assertTrue(first.window_id.startswith("akw-"))

    def test_evidence_reference_must_be_absolute(self):
        with self.assertRaisesRegex(ValueError, "absolute durable path"):
            live_controls.LiveCampaignIdentity(
                "ak-controls-v9-relative", "data/ak-controls-v9-relative")

    def test_parser_requires_and_preserves_fresh_identity(self):
        args = live_controls.build_parser().parse_args([
            "--campaign-id", "ak-controls-v9-parser",
            "--output", "/tmp/ak-controls-v9-parser",
        ])
        self.assertEqual(args.campaign_id, "ak-controls-v9-parser")
        self.assertEqual(args.output, Path("/tmp/ak-controls-v9-parser"))

    def test_parser_makes_live_and_existing_composition_exclusive(self):
        with self.assertRaises(SystemExit):
            live_controls.build_parser().parse_args([
                "--campaign-id", "ak-controls-v9-parser",
                "--output", "/tmp/ak-controls-v9-parser",
                "--execute", "--evaluate-existing",
            ])

    def test_current_source_claim_names_v9_and_exact_commit(self):
        reason = live_controls.CURRENT_SOURCE_CORRECTNESS_REASON
        self.assertIn("frozen v9 source", reason)
        self.assertIn(live_controls.PRODUCTION_COMMIT, reason)
        self.assertNotIn("frozen v8 source", reason)

    def test_existing_composition_cannot_overwrite_a_terminal_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                live_controls.evaluate_existing(
                    root, campaign_id="ak-controls-v9-terminal")


class RecordedCompositionMaterial(unittest.TestCase):

    def test_canonical_paired_block_round_trips(self):
        block = statistics.PairedBlock(
            block_index=3, unit_id="fixture:pp512", stratum="selection",
            order="candidate_first", anchor_samples=(1.0,),
            candidate_samples=(1.1,), measured_at="2026-08-12T00:00:00+00:00")
        self.assertEqual(
            live_controls._paired_block_from_raw(block.to_list()), block)

    def test_noncanonical_paired_block_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nine-field"):
            live_controls._paired_block_from_raw([0, "too-short"])


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
