"""No-inference tests for the prospective AK-WM-2 campaign capture."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from . import least_commitment_capture as C
from .test_schemas import _proposal


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


def diagnostic_source(value: dict) -> dict:
    cell = {
        "cell_id": "prefill/mul_mat", "demand_weight": 1.0,
        "supported": True, "compatible": True,
        "report_mass": {"iqk_on": 0.8, "iqk_off": 0.2},
        "regret_margin": 0.5,
    }
    fixtures = value["representation_contract"][
        "semantics_preserving_recoding_fixture_ids"]
    return {
        "schema": C.SOURCE_SCHEMA, "authority": "prospective_observe_only",
        "receipt_id": f"aklc-source-{value['proposal_id']}",
        "proposal_sha256": C.schemas.content_hash(value),
        "representation_frame_sha256": value["representation_contract"][
            "frame_sha256"],
        "candidate_frame_id": "iqk-cpu-prefill-v9",
        "do_not_repeat_match_ids": [],
        "quotients": {"canonical": [cell], **{
            fixture_id: [copy.deepcopy(cell)] for fixture_id in fixtures}},
    }


def plan(value: dict, *, role: str = "intervention",
         matched_control: str | None = "akp-20260812-1000") -> dict:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    with handle:
        json.dump(diagnostic_source(value), handle)
    binding = C.source_binding(Path(handle.name))
    source = json.loads(Path(handle.name).read_text(encoding="utf-8"))
    diagnostics, recodings = C.derive_diagnostics(
        source, proposal=value, candidate_frame_id="iqk-cpu-prefill-v9")
    receipts = {name: dict(binding) for name in C.DIAGNOSTICS}
    raw = {
        "schema": C.SCHEMA, "capture_id": "aklc-20260812-1001",
        "campaign_id": value["campaign_id"], "candidate_id": "akc-20260812-1001",
        "matched_experiment_id": "akm-iqk-20260812-0001",
        "proposal_id": value["proposal_id"], "role": role,
        "matched_control_proposal_id": matched_control,
        "candidate_frame_id": "iqk-cpu-prefill-v9", "regime": "prefill",
        "surface": "mul_mat", "intervention_id": "ggml-iqk-1",
        "changed_factor": "ggml_iqk",
        "factors": {"ggml_iqk": value["change"]["parameter_surface"]["candidate"]["ggml_iqk"],
                    "threads": 96},
        "diagnostics": diagnostics,
        "recodings": recodings,
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
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
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

    def test_source_bytes_are_resolved_and_mechanically_reduced(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        source_path = Path(next(iter(
            raw["diagnostic_source_receipts"].values()))["path"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["quotients"]["canonical"][0]["compatible"] = False
        source_path.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(C.CapturePlanError, "source SHA-256 differs"):
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
