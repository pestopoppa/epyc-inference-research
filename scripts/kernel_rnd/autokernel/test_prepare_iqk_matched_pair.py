"""No-inference tests for deterministic matched IQK pair preparation."""

from __future__ import annotations

import json
import copy
from pathlib import Path
import tempfile
import unittest

from . import campaign, least_commitment_capture as C
from . import prepare_iqk_matched_pair as P
from .execution import physical_bounds
from .test_campaign import write_calibration_bundle
from .test_least_commitment_capture import plan, proposal


class MatchedPairPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            dir="/mnt/raid0/llm/autokernel")
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / "fixture.gguf"
        self.model.write_bytes(b"not-a-real-model; preparation hashes only\n")
        self.calibration = write_calibration_bundle(self.root / "calibration")
        self.intervention = proposal()
        self.intervention["provider_reference"]["target_backend"] = \
            campaign.BACKEND_CPU
        P._rebind_provider_reference(self.intervention, self.calibration)
        self.proposal_path = self.root / "proposal-v4.json"
        self.proposal_path.write_text(
            json.dumps(self.intervention), encoding="utf-8")
        self.control = C.make_iqk_control_proposal(
            self.intervention, campaign_id="ak-iqk-control-20260812",
            proposal_id="akp-20260812-1000")
        base = campaign.CampaignSpec(
            campaign_id=self.intervention["campaign_id"],
            candidate_id="akc-20260812-1001",
            candidate_ref="registered:ggml_iqk", model=str(self.model),
            proposal=self.intervention,
            calibration=campaign.load_calibration_bundle(self.calibration))
        envelope = physical_bounds.PhysicalEnvelope(
            shape_id=base.measurement_unit_id, delivered_unit="token",
            flops_per_unit=1.0, bytes_per_unit=1.0,
            peak_compute_flops_s=1e15, peak_memory_bytes_s=1e15,
            measurement_frame_sha256=physical_bounds.measurement_frame_sha256(
                base.recipe_id, base.bench_params),
            work_derivation_ref="fixture", hardware_peak_ref="fixture")
        self.envelope_path = self.root / "physical-envelope-template.json"
        self.envelope_path.write_text(
            json.dumps(envelope.to_dict()), encoding="utf-8")
        intervention_factors = base.matched_factor_frame_for(
            "akm-iqk-20260812-0001", physical_envelope=envelope)
        control_factors = copy.deepcopy(intervention_factors)
        control_factors["ggml_iqk"] = "0"
        self.intervention_plan = plan(
            self.intervention, factors_override=intervention_factors)
        self.control_plan = plan(
            self.control, role="control", matched_control=None,
            factors_override=control_factors)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _source(raw: dict) -> Path:
        return Path(next(iter(raw["diagnostic_source_receipts"].values()))["path"])

    @staticmethod
    def _heldout(raw: dict) -> Path:
        return Path(raw["heldout_outcome_receipt"]["path"])

    def manifest(self) -> dict:
        return {
            "schema": P.SCHEMA,
            "matched_experiment_id": "akm-iqk-20260812-0001",
            "model": str(self.model),
            "calibration_bundle": str(self.calibration),
            "physical_envelope_template": str(self.envelope_path),
            "intervention_proposal": str(self.proposal_path),
            "intervention_campaign_id": self.intervention["campaign_id"],
            "intervention_proposal_id": self.intervention["proposal_id"],
            "control_proposal_id": self.control["proposal_id"],
            "blocks": 10, "reps": 5,
            "intervention": {
                "campaign_id": self.intervention["campaign_id"],
                "candidate_id": self.intervention_plan["candidate_id"],
                "capture_id": "aklc-prepared-intervention",
                "intervention_id": "iqk-enabled",
                "diagnostic_source": str(self._source(self.intervention_plan)),
                "evidence_stage": "heldout_bound",
                "heldout_outcome": str(self._heldout(self.intervention_plan)),
                "output_dir": str(self.root / "prepared-intervention"),
            },
            "control": {
                "campaign_id": self.control["campaign_id"],
                "candidate_id": self.control_plan["candidate_id"],
                "capture_id": "aklc-prepared-control",
                "intervention_id": "iqk-aa-control",
                "diagnostic_source": str(self._source(self.control_plan)),
                "evidence_stage": "heldout_bound",
                "heldout_outcome": str(self._heldout(self.control_plan)),
                "output_dir": str(self.root / "prepared-control"),
            },
        }

    def test_pair_is_complete_exactly_one_factor_and_non_executing(self):
        result = P.prepare(self.manifest())
        self.assertEqual(result["sole_changed_factor"], "ggml_iqk")
        self.assertFalse(result["inference_started"])
        self.assertFalse(result["campaign_executed"])
        frames = []
        for name in ("intervention", "control"):
            output = Path(result["outputs"][name]["path"])
            self.assertEqual({path.name for path in output.iterdir()}, {
                "proposal-v4.json", "least-commitment-diagnostic-source.json",
                "least-commitment-heldout-outcome.json",
                "least-commitment-capture-plan.json", "physical-envelope.json",
                "hypotheses.json"})
            hypothesis_store = json.loads((output / "hypotheses.json").read_text())
            self.assertEqual(hypothesis_store["schema"],
                             "epyc.autokernel.operator_hypotheses.v1")
            self.assertEqual(len(hypothesis_store["hypotheses"]), 1)
            self.assertEqual(
                hypothesis_store["hypotheses"][0]["statement"],
                (self.intervention if name == "intervention" else self.control)["hypothesis"])
            entry = hypothesis_store["hypotheses"][0]
            expected_prefix = ("akh-iqk-v9-known-real-" if name == "intervention"
                               else "akh-iqk-v9-aa-control-")
            self.assertEqual(entry["hypothesis_id"], expected_prefix +
                             self.manifest()[name]["candidate_id"].rsplit("-", 1)[-1])
            expected_falsifier = (P.HYPOTHESIS_FALSIFIER if name == "intervention"
                                  else P.AA_CONTROL_FALSIFIER)
            self.assertEqual(entry["falsifier"], expected_falsifier)
            raw = json.loads((output / "least-commitment-capture-plan.json").read_text())
            self.assertEqual(raw["schema"], C.SCHEMA)
            frames.append(raw["factors"])
        changed = [key for key in frames[0] if frames[0][key] != frames[1][key]]
        self.assertEqual(changed, ["ggml_iqk"])

    def test_provider_is_rebound_to_current_calibration_anchor(self):
        result = P.prepare(self.manifest())
        proposal_out = json.loads(
            (Path(result["outputs"]["intervention"]["path"]) /
             "proposal-v4.json").read_text())
        provider = proposal_out["provider_reference"]
        anchor = json.loads(
            (self.calibration / "runtime-source-label.json").read_text())
        self.assertEqual(provider["source_commit"], campaign.MEASUREMENT_COMMIT)
        self.assertEqual(provider["source_commit"],
                         anchor["measurement_instrument_commit"])
        self.assertEqual(provider["artifact_sha256"],
                         anchor["measurement_binary_sha256"])
        self.assertEqual(provider["linkage_manifest_sha256"],
                         anchor["measurement_linkage_sha256"])
        self.assertEqual(provider["toolchain_manifest_sha256"],
                         anchor["measurement_toolchain_manifest_sha256"])

    def test_existing_output_refuses_before_mutation(self):
        manifest = self.manifest()
        Path(manifest["control"]["output_dir"]).mkdir()
        with self.assertRaisesRegex(P.PreparationError, "must be new"):
            P.prepare(manifest)
        self.assertFalse(Path(manifest["intervention"]["output_dir"]).exists())

    def test_same_manifest_rebuilds_byte_identical_pair(self):
        manifest = self.manifest()
        first = P.prepare(manifest)
        for role in ("intervention", "control"):
            original = Path(manifest[role]["output_dir"])
            original.rename(self.root / f"first-{role}")
        second = P.prepare(manifest)
        self.assertEqual(first["input_sources"], second["input_sources"])
        self.assertEqual(first["input_manifest_sha256"],
                         second["input_manifest_sha256"])
        self.assertEqual(first["producer_sha256"], second["producer_sha256"])
        for role in ("intervention", "control"):
            first_dir = self.root / f"first-{role}"
            second_dir = Path(manifest[role]["output_dir"])
            self.assertEqual(
                {path.name: P._sha256(path) for path in first_dir.iterdir()},
                {path.name: P._sha256(path) for path in second_dir.iterdir()},
            )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
