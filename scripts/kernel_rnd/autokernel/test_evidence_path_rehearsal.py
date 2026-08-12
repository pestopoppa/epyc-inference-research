"""Architecture-only producer-chain rehearsal; launches no inference."""

from __future__ import annotations

import copy
import unittest

from . import evidence_path_rehearsal as R
from . import least_commitment_capture as C
from .test_campaign import MODEL
from .test_least_commitment_capture import plan, proposal


class EvidencePathRehearsalTest(unittest.TestCase):
    def setUp(self):
        self.intervention = proposal()
        self.control = C.make_iqk_control_proposal(
            self.intervention, campaign_id="ak-iqk-control-20260812",
            proposal_id="akp-20260812-1000")
        intervention_raw = plan(self.intervention)
        self.intervention_plan = C.from_mapping(
            intervention_raw, proposal=self.intervention,
            campaign_id=self.intervention["campaign_id"],
            candidate_id=intervention_raw["candidate_id"])
        control_raw = plan(self.control, role="control", matched_control=None)
        control_raw["plan_sha256"] = C.plan_sha256(control_raw)
        self.control_plan = C.from_mapping(
            control_raw, proposal=self.control,
            campaign_id=self.control["campaign_id"],
            candidate_id=control_raw["candidate_id"])

    def test_every_ap_wm_and_release_stage_has_a_callable_producer(self):
        report = R.producer_manifest(
            intervention_proposal=self.intervention,
            intervention_plan=self.intervention_plan,
            control_proposal=self.control, control_plan=self.control_plan)
        self.assertFalse(report["inference_started"])
        self.assertFalse(report["live_authority"])
        for name in C.DIAGNOSTICS:
            self.assertIn(f"diagnostic.{name}", report["field_producers"])
        for name in C.OUTCOME_REDUCERS:
            self.assertIn(f"outcome.{name}", report["field_producers"])
        for stage in ("matched_control_join", "archive", "ap_wm_report",
                      "champion", "readiness", "t3", "package"):
            self.assertIn(stage, report["field_producers"])

    def test_a_second_changed_factor_is_refused(self):
        raw = copy.deepcopy(self.intervention_plan.raw)
        raw["factors"]["threads"] = 192
        raw["plan_sha256"] = C.plan_sha256(raw)
        changed = C.from_mapping(
            raw, proposal=self.intervention,
            campaign_id=self.intervention["campaign_id"],
            candidate_id=raw["candidate_id"])
        with self.assertRaisesRegex(R.RehearsalError, "pair changes"):
            R.producer_manifest(
                intervention_proposal=self.intervention,
                intervention_plan=changed,
                control_proposal=self.control, control_plan=self.control_plan)

    def test_current_campaign_cli_contract_is_one_json_document(self):
        contract = R.verify_campaign_json_contract(MODEL)
        self.assertEqual(contract["stdout"], "exactly_one_json_document")
        self.assertEqual(contract["trace"], "stderr")
        self.assertEqual(contract["state"], "dry_run_composed")
        self.assertFalse(contract["executed"])


if __name__ == "__main__":
    unittest.main()
