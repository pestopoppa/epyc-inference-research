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
    is_control = value["change"]["parameter_surface"]["candidate"]["ggml_iqk"] \
        == value["change"]["parameter_surface"]["anchor"]["ggml_iqk"]
    cell = {
        "cell_id": "prefill/mul_mat", "demand_weight": 1.0,
        "supported": True, "compatible": True,
        "report_mass": ({"aa_equal": 0.5, "aa_alternate": 0.5}
                        if is_control else {"iqk_on": 0.8, "iqk_off": 0.2}),
        "regret_margin": 0.0 if is_control else 0.5,
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
    heldout = {
        "schema": C.HELDOUT_SCHEMA,
        "authority": "observe_only_measurement",
        "receipt_id": f"aklc-heldout-{value['proposal_id']}",
        "proposal_id": value["proposal_id"],
        "proposal_sha256": C.schemas.content_hash(value),
        "candidate_frame_id": "iqk-cpu-prefill-v9",
        "regime": "decode", "surface": "mul_mat",
        "metric": "tokens_per_second", "metric_direction": "higher",
        "relative_effect": 0.02 if role == "intervention" else 0.0,
        "measurement_record_sha256": C.schemas.content_hash({
            "proposal_id": value["proposal_id"], "regime": "decode"}),
        "capture_mode": "measured",
    }
    heldout_handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False)
    with heldout_handle:
        json.dump(heldout, heldout_handle)
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
        "heldout_outcome_receipt": C.source_binding(Path(heldout_handle.name)),
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
            decision=SimpleNamespace(
                keep=True, median_relative=0.06, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
        self.assertEqual(block["schema"], C.BLOCK_SCHEMA)
        self.assertEqual(set(block["diagnostics"]), set(C.DIAGNOSTICS))
        self.assertEqual(block["outcome"], {
            "heldout_regime_transfer": 0.02,
            "falsifier_resolution": 0.03,
            "noise_floor": 0.01,
        })
        self.assertFalse(block["falsifier"]["triggered"])

    def test_missing_heldout_input_is_refused(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        raw.pop("heldout_outcome_receipt")
        raw["plan_sha256"] = C.plan_sha256(raw)
        with self.assertRaisesRegex(C.CapturePlanError, "fields must be exactly"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_target_regime_cannot_masquerade_as_heldout(self):
        proposal_record = proposal()
        raw = plan(proposal_record)
        binding = raw["heldout_outcome_receipt"]
        path = Path(binding["path"])
        source = json.loads(path.read_text(encoding="utf-8"))
        source["regime"] = "prefill"
        path.write_text(json.dumps(source), encoding="utf-8")
        raw["heldout_outcome_receipt"] = C.source_binding(path)
        raw["plan_sha256"] = C.plan_sha256(raw)
        with self.assertRaisesRegex(C.CapturePlanError, "outside.*target regimes"):
            C.from_mapping(raw, proposal=proposal_record,
                           campaign_id=proposal_record["campaign_id"],
                           candidate_id="akc-20260812-1001")

    def test_control_falsifier_is_keep_or_effect_above_noise(self):
        control = C.make_iqk_control_proposal(
            proposal(), campaign_id="ak-iqk-control-20260812",
            proposal_id="akp-20260812-1000")
        raw = plan(control, role="control", matched_control=None)
        capture = C.from_mapping(
            raw, proposal=control, campaign_id=control["campaign_id"],
            candidate_id=raw["candidate_id"])
        kept = C.materialize(
            capture,
            decision=SimpleNamespace(
                keep=True, median_relative=0.001, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
        self.assertTrue(kept["falsifier"]["triggered"])
        self.assertTrue(kept["falsifier"]["predicates"]["decision_triggered"])
        noisy = C.materialize(
            capture,
            decision=SimpleNamespace(
                keep=False, median_relative=-0.02, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
        self.assertTrue(noisy["falsifier"]["triggered"])
        self.assertTrue(noisy["falsifier"]["predicates"]["noise_exceeded"])
        clean = C.materialize(
            capture,
            decision=SimpleNamespace(
                keep=False, median_relative=0.005, contribution_floor=0.03),
            calibration=SimpleNamespace(noise_floor_phi=0.01),
            executed_factors=capture.raw["factors"])
        self.assertFalse(clean["falsifier"]["triggered"])
        self.assertEqual(clean["outcome"]["falsifier_resolution"], 0.005)

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
