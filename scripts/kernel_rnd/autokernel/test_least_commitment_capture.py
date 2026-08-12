"""No-inference tests for the prospective AK-WM-2 campaign capture."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from . import least_commitment_capture as C
from .test_schemas import _proposal, _sha


def proposal() -> dict:
    value = _proposal()
    value.update({
        "proposal_id": "akp-20260812-1001",
        "campaign_id": "ak-iqk-intervention-20260812",
        "campaign_kind": "config", "change_class": "parameter",
    })
    value["target"]["regimes"] = ["prefill"]
    value["change"]["parameter_surface"] = {
        "candidate": {"ggml_iqk": "1"}, "anchor": {"ggml_iqk": "0"}}
    return value


def plan(value: dict, *, role: str = "intervention",
         matched_control: str | None = "akp-20260812-1000") -> dict:
    diagnostics = {name: float(index + 1) / 10.0
                   for index, name in enumerate(C.DIAGNOSTICS)}
    diagnostics["information_gain"] = value["expected_information_gain"]
    receipts = {name: {"receipt_id": f"rcpt-{name}", "sha256": _sha(name)}
                for name in C.DIAGNOSTICS}
    raw = {
        "schema": C.SCHEMA, "capture_id": "aklc-20260812-1001",
        "campaign_id": value["campaign_id"], "candidate_id": "akc-20260812-1001",
        "proposal_id": value["proposal_id"], "role": role,
        "matched_control_proposal_id": matched_control,
        "candidate_frame_id": "iqk-cpu-prefill-v9", "regime": "prefill",
        "surface": "mul_mat", "intervention_id": "ggml-iqk-1",
        "changed_factor": "ggml_iqk",
        "factors": {"ggml_iqk": value["change"]["parameter_surface"]["candidate"]["ggml_iqk"],
                    "threads": 96},
        "diagnostics": diagnostics,
        "recodings": {
            fixture_id: copy.deepcopy(diagnostics)
            for fixture_id in value["representation_contract"][
                "semantics_preserving_recoding_fixture_ids"]},
        "diagnostic_source_receipts": receipts,
        "outcome_reducers": dict(C.OUTCOME_REDUCERS),
        "capture_mode": "measured",
    }
    raw["plan_sha256"] = C.plan_sha256(raw)
    return raw


class CapturePlanTest(unittest.TestCase):
    def test_measured_decision_produces_every_projector_field(self):
        proposal_record = proposal()
        capture = C.from_mapping(
            plan(proposal_record), proposal=proposal_record,
            campaign_id=proposal_record["campaign_id"],
            candidate_id="akc-20260812-1001")
        block = C.materialize(
            capture,
            decision=SimpleNamespace(median_relative=0.06, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01))
        self.assertEqual(block["schema"], C.BLOCK_SCHEMA)
        self.assertEqual(set(block["diagnostics"]), set(C.DIAGNOSTICS))
        self.assertEqual(block["outcome"], {
            "heldout_regime_transfer": 0.06,
            "falsifier_resolution": 0.03,
            "noise_floor": 0.01,
        })

    def test_post_result_diagnostic_mutation_breaks_plan_hash(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        raw["diagnostics"]["k_rho"] += 1.0
        with self.assertRaisesRegex(C.CapturePlanError, "plan_sha256"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_information_gain_cannot_disagree_with_proposal(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        raw["diagnostics"]["information_gain"] = 0.9
        raw["plan_sha256"] = C.plan_sha256(raw)
        with self.assertRaisesRegex(C.CapturePlanError, "expected_information_gain"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_control_proposal_is_exact_a_a_and_schema_valid(self):
        intervention = proposal()
        control = C.make_iqk_control_proposal(
            intervention, campaign_id="ak-iqk-control-20260812",
            proposal_id="akp-20260812-1000")
        self.assertEqual(control["change"]["parameter_surface"], {
            "candidate": {"ggml_iqk": "0"}, "anchor": {"ggml_iqk": "0"}})
        control_plan = plan(control, role="control", matched_control=None)
        control_plan.update({
            "campaign_id": control["campaign_id"], "proposal_id": control["proposal_id"],
        })
        control_plan["plan_sha256"] = C.plan_sha256(control_plan)
        parsed = C.from_mapping(
            control_plan, proposal=control, campaign_id=control["campaign_id"],
            candidate_id="akc-20260812-1001")
        self.assertEqual(parsed.role, "control")


if __name__ == "__main__":
    unittest.main()
