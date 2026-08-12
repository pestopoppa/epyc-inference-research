"""Architecture-only producer-chain rehearsal; launches no inference."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from . import evidence_path_rehearsal as R
from . import least_commitment_capture as C
from . import schemas
from .test_campaign import MODEL
from .test_least_commitment_capture import plan, proposal


class EvidencePathRehearsalTest(unittest.TestCase):
    def setUp(self):
        self.intervention = proposal()
        self.intervention["provider_reference"]["target_backend"] = "llama_cpu"
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

    def test_copied_control_diagnostic_semantics_are_refused(self):
        raw = copy.deepcopy(self.control_plan.raw)
        source_binding = next(iter(
            self.intervention_plan.raw["diagnostic_source_receipts"].values()))
        source = json.loads(Path(source_binding["path"]).read_text(encoding="utf-8"))
        source["receipt_id"] = "aklc-source-independent-name-only"
        source["proposal_sha256"] = schemas.content_hash(self.control)
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as handle:
            json.dump(source, handle)
            path = Path(handle.name)
        binding = C.source_binding(path)
        raw["diagnostic_source_receipts"] = {
            name: dict(binding) for name in C.DIAGNOSTICS}
        diagnostics, recodings = C.derive_diagnostics(
            source, proposal=self.control,
            candidate_frame_id=raw["candidate_frame_id"])
        raw["diagnostics"] = diagnostics
        raw["recodings"] = recodings
        raw["plan_sha256"] = C.plan_sha256(raw)
        copied = C.from_mapping(
            raw, proposal=self.control, campaign_id=self.control["campaign_id"],
            candidate_id=raw["candidate_id"])
        with self.assertRaisesRegex(R.RehearsalError, "semantics are identical"):
            R.producer_manifest(
                intervention_proposal=self.intervention,
                intervention_plan=self.intervention_plan,
                control_proposal=self.control, control_plan=copied)

    def test_current_campaign_cli_contract_is_one_json_document(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proposal_path = root / "proposal-v4.json"
            proposal_path.write_text(
                json.dumps(self.intervention), encoding="utf-8")
            contract = R.verify_campaign_json_contract(MODEL, campaign_args=(
                "--campaign-id", self.intervention["campaign_id"],
                "--candidate-id", self.intervention_plan.raw["candidate_id"],
                "--proposal-manifest", str(proposal_path),
                "--backend", "llama_cpu",
            ))
        self.assertEqual(contract["stdout"], "exactly_one_json_document")
        self.assertEqual(contract["trace"], "stderr")
        self.assertEqual(contract["state"], "dry_run_composed")
        self.assertFalse(contract["executed"])

    def test_durable_contract_regression_uses_current_proposal_schema(self):
        self.assertEqual(self.intervention["schema"], schemas.SCHEMA_PROPOSAL)
        self.assertEqual(self.intervention["schema"], schemas.SCHEMA_PROPOSAL_V4)
        self.assertFalse(schemas.validate_proposal(self.intervention))
        self.assertEqual(
            self.intervention["provider_reference"]["kind"], "llama_source")
        self.assertEqual(
            self.intervention["provider_reference"]["target_backend"], "llama_cpu")

    def test_hypothesis_store_resolves_the_exact_proposal_statement(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "hypotheses.json"
            path.write_text(json.dumps({
                "schema": "epyc.autokernel.operator_hypotheses.v1",
                "hypotheses": [{
                    "hypothesis_id": "akh-evidence-path-rehearsal",
                    "statement": self.intervention["hypothesis"],
                    "falsifier": "The accepted paired result misses its effect floor.",
                    "author": "operator",
                }],
            }), encoding="utf-8")
            binding = R.verify_hypothesis_store(
                path, hypothesis_id="akh-evidence-path-rehearsal",
                proposal=self.intervention)
            self.assertEqual(binding["store_path"], str(path.resolve()))
            self.assertEqual(len(binding["store_sha256"]), 64)
            changed = copy.deepcopy(self.intervention)
            changed["hypothesis"] = "a different question"
            with self.assertRaisesRegex(R.RehearsalError, "differs"):
                R.verify_hypothesis_store(
                    path, hypothesis_id="akh-evidence-path-rehearsal",
                    proposal=changed)


if __name__ == "__main__":
    unittest.main()
