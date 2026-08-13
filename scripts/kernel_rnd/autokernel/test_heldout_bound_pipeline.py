"""No-inference tests for the decode -> heldout prefill -> archive seam."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from . import heldout_bound_pipeline as H
from . import offline_least_commitment as L
from . import test_prepare_iqk_matched_pair as pair_test
from . import test_least_commitment_receipts as receipt_test


class HeldoutBoundPipelineTest(unittest.TestCase):
    def setUp(self):
        self.fixture = pair_test.MatchedPairPreparationTest(methodName="runTest")
        self.fixture.setUp()
        for name in (
                "root", "intervention", "intervention_plan", "control_plan"):
            setattr(self, name, getattr(self.fixture, name))
        self.manifest = self.fixture.manifest
        self._heldout = self.fixture._heldout

    def tearDown(self):
        self.fixture.tearDown()

    def pipeline_manifest(self) -> dict:
        pair = self.manifest()
        measurements = {}
        for role, raw in (("intervention", self.intervention_plan),
                          ("control", self.control_plan)):
            receipt = json.loads(self._heldout(raw).read_text(encoding="utf-8"))
            record = receipt["measurement_record"]
            measurements[role] = {
                "receipt_id": f"aklc-fresh-heldout-{role}",
                "measurement": {key: record[key] for key in (
                    "journal_root", "campaign_id", "proposal_id",
                    "completion_event_id")},
            }
            pair[role]["heldout_outcome"] = None
            pair[role]["evidence_stage"] = "heldout_bound"
        result = {
            "schema": H.SCHEMA,
            "pair_manifest": pair,
            "fixed_proposals": {},
            "heldout_measurements": measurements,
            "nominal_khz": 2_500_000,
            "archive": {
                "archive_id": "ak-heldout-pipeline-test",
                "created_at": "2026-08-13T00:00:00Z",
                "diagnostic_directions": {
                    name: ("lower" if name in {
                        "unsupported_scope_width", "k_rho", "raw_impurity"
                    } else "higher") for name in L.DIAGNOSTICS},
                "outcome_weights": {
                    "heldout_regime_transfer": 0.5,
                    "falsifier_resolution": 0.5,
                },
                "output_dir": str(self.root / "archive-output"),
                "report_output": str(self.root / "ap-wm-report.json"),
            },
        }
        fixed = H._target_proposals(pair)
        for role, proposal in zip(("intervention", "control"), fixed):
            path = self.root / f"fixed-{role}-proposal.json"
            path.write_text(json.dumps(proposal), encoding="utf-8")
            result["fixed_proposals"][role] = {
                "path": str(path), "sha256": H.pair._sha256(path)}
        return result

    def test_two_real_decode_receipts_publish_fresh_heldout_pair(self):
        result = H.prepare(self.pipeline_manifest())
        self.assertFalse(result["inference_started"])
        self.assertFalse(result["campaign_executed"])
        for role in ("intervention", "control"):
            root = Path(result["pair_result"]["outputs"][role]["path"])
            plan = json.loads((root / "least-commitment-capture-plan.json").read_text())
            self.assertEqual(plan["evidence_stage"], "heldout_bound")
            self.assertIsNotNone(plan["heldout_outcome_receipt"])
            self.assertTrue((root / "least-commitment-heldout-outcome.json").is_file())
            self.assertIn("--execute", result["commands"][role]["execute"])
            self.assertNotIn("--execute", result["commands"][role]["dry_run"])
        rows = result["archive_template"]["rows"]
        self.assertEqual([row["completion_event_id"] for row in rows], [None, None])
        self.assertEqual(rows[1]["matched_control_id"], rows[0]["proposal_id"])

    def test_bootstrap_or_unbound_template_refuses_before_campaign_dirs(self):
        raw = self.pipeline_manifest()
        raw["pair_manifest"]["control"]["evidence_stage"] = "bootstrap"
        with self.assertRaisesRegex(H.HeldoutPipelineError, "null heldout_bound"):
            H.prepare(raw)
        for role in ("intervention", "control"):
            self.assertFalse(Path(raw["pair_manifest"][role]["output_dir"]).exists())

    def test_missing_or_null_decode_receipt_refuses_before_campaign_dirs(self):
        raw = self.pipeline_manifest()
        raw["heldout_measurements"]["control"] = None
        with self.assertRaisesRegex(H.HeldoutPipelineError, "fields must be exactly"):
            H.prepare(raw)
        for role in ("intervention", "control"):
            self.assertFalse(Path(raw["pair_manifest"][role]["output_dir"]).exists())

    def test_fixed_proposal_hash_or_record_mismatch_refuses_before_dirs(self):
        raw = self.pipeline_manifest()
        raw["fixed_proposals"]["control"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(H.HeldoutPipelineError, "file SHA-256 differs"):
            H.prepare(raw)
        for role in ("intervention", "control"):
            self.assertFalse(Path(raw["pair_manifest"][role]["output_dir"]).exists())

    def test_same_regime_measurement_refuses_before_campaign_dirs(self):
        raw = self.pipeline_manifest()
        raw["heldout_measurements"]["control"] = copy.deepcopy(
            raw["heldout_measurements"]["intervention"])
        # The intervention decode proposal is not the exact A/A parameter
        # surface required by the control target.
        with self.assertRaisesRegex(Exception, "parameter surface differs"):
            H.prepare(raw)
        for role in ("intervention", "control"):
            self.assertFalse(Path(raw["pair_manifest"][role]["output_dir"]).exists())

    def test_archive_refuses_until_both_prefill_campaigns_are_decided(self):
        result = H.prepare(self.pipeline_manifest())
        with self.assertRaisesRegex(H.HeldoutPipelineError, "DECIDED terminal"):
            H.archive(result)
        self.assertFalse(Path(result["archive_template"]["output_dir"]).exists())
        self.assertFalse(Path(result["archive_template"]["report_output"]).exists())

    def test_archive_and_observe_only_report_run_from_two_decided_rows(self):
        result = H.prepare(self.pipeline_manifest())
        fixture = receipt_test.ReceiptProjectionTest(methodName="runTest")
        fixture.setUp()
        try:
            result["archive_template"]["rows"] = [{
                key: row[key] for key in (
                    "journal_root", "campaign_id", "proposal_id",
                    "completion_event_id", "matched_control_id")
            } for row in fixture.rows]
            result["result_sha256"] = H.schemas.content_hash({
                key: value for key, value in result.items()
                if key != "result_sha256"})
            archived = H.archive(result)
            report = json.loads(Path(archived["report"]["path"]).read_text())
            self.assertFalse(archived["live_authority"])
            self.assertEqual(archived["planner"], {
                "invoked": False,
                "status": "roster_not_invoked_for_observe_only",
                "roster": None,
            })
            self.assertEqual(report["evidence_label"], "real")
            self.assertEqual(report["matched_validation"]["status"], "PASS")
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
