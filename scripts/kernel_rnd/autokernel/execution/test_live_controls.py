import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ..evaluator import statistics
from . import live_controls
from .. import campaign, schemas


class InstrumentSelection(unittest.TestCase):

    def test_missing_instrument_tools_are_built_as_complete_t0_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bindir = root / "build-v9-cpu" / "bin"
            bindir.mkdir(parents=True)
            def build(_argv, **_kwargs):
                for name in live_controls.INSTRUMENT_BUILD_TARGETS:
                    path = bindir / name
                    path.touch()
                    path.chmod(0o755)
            with mock.patch.object(live_controls, "INSTRUMENT_ROOT", root), \
                 mock.patch.object(live_controls, "INSTRUMENT_BINARY",
                                   bindir / "llama-bench"), \
                 mock.patch.object(live_controls.subprocess, "run", side_effect=build) as run:
                live_controls._ensure_instrument_build()
            run.assert_called_once_with(
                ["/usr/bin/cmake", "--build", str(root / "build-v9-cpu"), "--target",
                 "llama-completion", "llama-bench", "test-backend-ops", "-j", "64"],
                cwd=str(root), check=True)

    def tearDown(self):
        importlib.reload(live_controls)

    def test_clean_instrument_worktree_can_be_selected_explicitly(self):
        with mock.patch.dict(os.environ, {
                "AUTOKERNEL_INSTRUMENT_ROOT": "/tmp/ak-clean-instrument"}, clear=False):
            module = importlib.reload(live_controls)
        self.assertEqual(str(module.INSTRUMENT_ROOT), "/tmp/ak-clean-instrument")
        self.assertEqual(
            str(module.INSTRUMENT_BINARY),
            "/tmp/ak-clean-instrument/build-ak-t0-cpu-f744cc220/bin/llama-bench")

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


class CalibrationFrame(unittest.TestCase):

    def test_calibration_is_explicitly_iqk_off_at_campaign_repetition_count(self):
        """A/A must govern the same off-baseline that campaigns compare."""
        self.assertEqual(live_controls.CALIBRATION_REPS, 1)
        self.assertEqual(live_controls.CONTROL_ARM_IQK["aa_calibration"], ("0", "0"))
        self.assertEqual(
            live_controls.CONTROL_ARM_IQK["neutral_calibration"], ("0", "0"))
        self.assertEqual(
            live_controls._params(prompt=live_controls.PROMPT_TOKENS)["reps"], 1)

    def test_declaration_precommits_the_calibration_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = live_controls.LiveCampaignIdentity("ak-controls-frame", str(root))
            live_controls._write_declaration(
                root, identity=identity, instrument_sha="1" * 64,
                copy_sha="1" * 64, instrument_linkage="2" * 64,
                copy_linkage="2" * 64, toolchain_manifest_sha256="3" * 64,
                sealed_binding=live_controls.recipes.ToolBinding(
                    binary="/sealed/llama-bench", source_root="/sealed",
                    library_path="/sealed"))
            declaration = json.loads(
                (root / "campaign_declaration.json").read_text(encoding="utf-8"))
            source = json.loads(
                (root / "runtime-source-label.json").read_text(encoding="utf-8"))
        self.assertEqual(declaration["calibration_frame"], {
            "recipe_id": live_controls.RECIPE_ID,
            "prompt_tokens": live_controls.PROMPT_TOKENS,
            "reps": 1,
            "candidate_ggml_iqk": "0",
            "anchor_ggml_iqk": "0",
        })
        self.assertEqual(declaration["source_sha256"], source["source_sha256"])

    def test_declaration_precommits_a_non_ranked_campaign_length_anchor_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = live_controls.LiveCampaignIdentity("ak-controls-anchor-window", str(root))
            live_controls._write_declaration(
                root, identity=identity, instrument_sha="1" * 64,
                copy_sha="1" * 64, instrument_linkage="2" * 64,
                copy_linkage="2" * 64, toolchain_manifest_sha256="3" * 64,
                sealed_binding=live_controls.recipes.ToolBinding(
                    binary="/sealed/llama-bench", source_root="/sealed",
                    library_path="/sealed"))
            declaration = json.loads(
                (root / "campaign_declaration.json").read_text(encoding="utf-8"))
        self.assertEqual(declaration["anchor_motion_window_blocks"], 15)
        self.assertEqual(declaration["anchor_motion_settling"], {
            "schema": "epyc.autokernel.anchor_motion_settling.v1",
            "kind": "non_ranked_post_work_quiet",
            "quiet_barrier_s": campaign.POST_T0_QUIET_BARRIER_S,
            "required_samples": campaign.POST_T0_QUIET_SAMPLES,
            "sample_interval_s": campaign.POST_T0_QUIET_SAMPLE_INTERVAL_S,
        })

    def test_measurement_plan_carries_the_declared_baseline_frame(self):
        """The executor must not silently fall back to recipe defaults."""
        class Run:
            complete = True

            @staticmethod
            def raw_vector():
                return {"fixture": True}

        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = live_controls.recipes.ToolBinding(
                binary="/sealed/llama-bench", source_root="/sealed", library_path="/sealed")
            worktree = live_controls.recipes.ToolBinding(
                binary="/worktree/llama-bench", source_root="/worktree", library_path="/worktree")
            anchor = live_controls.api.AnchorIdentity(
                source_commit="f" * 40, binary_sha256="1" * 64,
                linkage_sha256="2" * 64, tool="llama-bench")
            with mock.patch.object(
                    live_controls.microbench, "MicrobenchPlan",
                    side_effect=lambda **kwargs: captured.update(kwargs) or object()), \
                 mock.patch.object(
                    live_controls.microbench, "CpuRegionClaimAdapter"), \
                 mock.patch.object(
                    live_controls.microbench, "MicrobenchRunner") as runner:
                runner.return_value.run.return_value = Run()
                live_controls._measure(
                    label="aa_calibration", blocks=1, claim=object(),
                    candidate_binding=binding, anchor_binding=worktree, anchor=anchor,
                    candidate_iqk="0", anchor_iqk="0", output_root=root,
                    host_state=mock.Mock(), identity=live_controls.LiveCampaignIdentity(
                        "ak-controls-frame-run", str(root)))
        self.assertEqual(captured["params"]["reps"], 1)
        self.assertEqual(captured["candidate_param_overrides"], {"ggml_iqk": "0"})
        self.assertEqual(captured["anchor_param_overrides"], {"ggml_iqk": "0"})
        self.assertEqual(captured["candidate_binding"], binding)
        self.assertEqual(captured["anchor_binding"], binding)


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


class ProspectiveBeliefReceipt(unittest.TestCase):

    class _Result:
        may_rank = True

        def to_dict(self):
            controls = (
                "positive", "neutral", "degraded_negative", "aa",
                "historical_win_replay")
            return {
                "panel_result": {
                    "marker": "5/5", "may_rank": True,
                    "halts_campaign": False, "voids_window": False,
                    "outcomes": [{
                        "control_id": control, "ordinal": index,
                        "outcome": "PASS", "disposition": "satisfied",
                    } for index, control in enumerate(controls, 1)],
                    "observations": [{
                        "control_id": control, "abs_effect_count": 15,
                        "effect_resolution": (
                            "improvement" if control in {
                                "positive", "historical_win_replay"}
                            else "below_noise_floor"),
                    } for control in controls],
                },
            }

    class _Claim:
        def to_dict(self):
            return {
                "schema": "epyc.autokernel.cpu_region_claim_receipt.v1",
                "claim_id": "akc-live-claim", "campaign_id": "ak-live-next",
                "cpu_list": live_controls.CPU_LIST,
                "acquired_at": "2026-08-12T02:00:00+00:00",
                "released_at": "2026-08-12T02:10:00+00:00",
            }

    def _fixture(self, root: Path, *, prospective: bool = True):
        runtime_body = {
            "schema": "epyc.autokernel.runtime_source_label.v1",
            "production_source_commit": live_controls.PRODUCTION_COMMIT,
            "measurement_instrument_commit": live_controls.INSTRUMENT_COMMIT,
            "measurement_binary_sha256": "1" * 64,
            "copied_binary_sha256": "1" * 64,
            "measurement_linkage_sha256": "2" * 64,
            "copied_linkage_sha256": "2" * 64,
            "binary_copy_exact": True,
        }
        declaration = {
            "schema": "epyc.autokernel.live_control_campaign_declaration.v1",
            "campaign_id": "ak-live-next", "model": "/models/tiny.gguf",
            "model_sha256": "3" * 64,
        }
        if prospective:
            declaration["belief_capture_schema"] = live_controls.BELIEF_RECEIPT_SCHEMA
        for name, value in (
                ("campaign_declaration.json", declaration),
                ("runtime-source-label.json", {
                    **runtime_body, "source_sha256": schemas.content_hash(runtime_body)}),
                ("control_sweep.json", self._Result().to_dict())):
            (root / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
        for label in (
                "positive", "neutral_calibration", "negative_committed_cell",
                "negative_wrong_cell", "aa_calibration", "historical_win_replay"):
            path = root / "raw" / f"{label}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"label": label}) + "\n", encoding="utf-8")

    def test_future_writer_emits_five_self_hashed_identity_bound_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            receipt = live_controls._build_belief_receipt(
                root, identity=live_controls.LiveCampaignIdentity(
                    "ak-live-next", str(root)), result=self._Result(),
                claim_receipt=self._Claim(),
                measured_at="2026-08-12T02:10:00+00:00")
        self.assertIsNotNone(receipt)
        self.assertEqual(len(receipt["belief_measurements"]), 5)
        self.assertEqual(receipt["native_verdict"]["marker"], "5/5")
        for row in receipt["belief_measurements"]:
            self.assertEqual(row["protocol_id"], "P-AK-SEARCH-1/v1")
            self.assertEqual(row["reps_basis"], "scored:paired live-control blocks")
            self.assertEqual(
                row["extra"]["evidence_sha256"],
                schemas.content_hash(row["extra"]["evidence_basis"]))
            unsigned = dict(row)
            stored = unsigned.pop("measurement_sha256")
            self.assertEqual(stored, schemas.content_hash(unsigned))
        unsigned_receipt = dict(receipt)
        stored_receipt = unsigned_receipt.pop("receipt_sha256")
        self.assertEqual(stored_receipt, schemas.content_hash(unsigned_receipt))

    def test_pre_hook_declaration_is_not_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root, prospective=False)
            receipt = live_controls._build_belief_receipt(
                root, identity=live_controls.LiveCampaignIdentity(
                    "ak-live-next", str(root)), result=self._Result(),
                claim_receipt=self._Claim(),
                measured_at="2026-08-12T02:10:00+00:00")
        self.assertIsNone(receipt)


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
