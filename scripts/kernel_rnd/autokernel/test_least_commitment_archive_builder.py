#!/usr/bin/env python3
"""Regression tests for the strict AK-WM-2a real-archive projection."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from . import least_commitment_archive_builder as B
from . import offline_least_commitment as L
from . import schemas as S


def _contract() -> dict:
    contract = {
        "vocabulary": {
            "regimes": ["prefill"], "surfaces": ["mul_mat"],
            "outcomes": ["heldout_regime_transfer", "falsifier_resolution"],
            "contradictions": ["known_win_does_not_reproduce"],
        },
        "vocabulary_source_receipts": ["rcpt-vocabulary-real-1"],
        "considered_alternatives": ["iqk_on", "iqk_off", "source_patch"],
        "excluded_alternatives": [{
            "alternative_id": "source_patch", "reason": "not one factor",
            "source_receipt_id": "rcpt-exclusion-real-1",
        }],
        "empirical_demand": {
            "receipt_id": "rcpt-demand-real-1",
            "weights_sha256": S.content_hash({"prefill": 1.0}),
        },
        "abstraction_construction_cost": {
            "value": 1.0, "unit": "typed_facts", "receipt_id": "rcpt-cost-real-1",
        },
        "canonical_encoding": {
            "encoding_id": "ak-representation-json/v1",
            "schema_sha256": S.content_hash({"schema": "ak-representation-json/v1"}),
        },
        "semantics_preserving_recoding_fixture_ids": ["rename-v1"],
    }
    contract["frame_sha256"] = S.representation_frame_sha256(contract)
    return contract


def _diagnostics(score: float) -> dict:
    return {
        "unsupported_scope_width": 2.0 - score,
        "compatible_future_mass": score,
        "k_rho": 2.0 - score,
        "information_gain": score,
        "novelty": score,
        "raw_impurity": 2.0 - score,
        "weighted_minority": score,
    }


def _completed(proposal_id: str, event_id: str) -> B.CompletedProposal:
    proposal = {
        "proposal_id": proposal_id,
        "representation_contract": _contract(),
    }
    result = {
        "campaign_id": f"ak-{proposal_id}",
        "spec": {"recipe_id": "t1b.llama_cpu.llama_bench_prefill.v1"},
        "decision": {"keep": proposal_id.endswith("intervention")},
    }
    return B.CompletedProposal(
        proposal=proposal, proposal_event_id=f"proposal-{event_id}",
        result=result, completion_event_id=event_id)


def _diagnostic(completed: B.CompletedProposal, score: float) -> dict:
    contract = completed.proposal["representation_contract"]
    values = _diagnostics(score)
    return {
        "schema": B.DIAGNOSTIC_SCHEMA,
        "proposal_id": completed.proposal["proposal_id"],
        "proposal_sha256": completed.proposal_sha256,
        "representation_frame_sha256": contract["frame_sha256"],
        "empirical_demand_weights_sha256": contract["empirical_demand"][
            "weights_sha256"],
        "diagnostics": values,
        "recodings": {"rename-v1": copy.deepcopy(values)},
    }


def _outcome(completed: B.CompletedProposal, value: float) -> dict:
    contract = completed.proposal["representation_contract"]
    return {
        "schema": B.OUTCOME_SCHEMA,
        "proposal_id": completed.proposal["proposal_id"],
        "completion_event_id": completed.completion_event_id,
        "campaign_result_sha256": completed.result_sha256,
        "candidate_frame_id": "candidate-frame-real-v1",
        "representation_frame_sha256": contract["frame_sha256"],
        "empirical_demand_weights_sha256": contract["empirical_demand"][
            "weights_sha256"],
        "metric": "prefill_tokens_per_s",
        "metric_direction": "higher_better",
        "regime": "prefill", "surface": "mul_mat",
        "intervention_id": f"factor-{completed.proposal['proposal_id']}",
        "changed_factor": "ggml_iqk",
        "outcome": {
            "heldout_regime_transfer": value,
            "falsifier_resolution": value,
            "noise_floor": 0.01,
        },
    }


class ArchiveBuilderTest(unittest.TestCase):

    def setUp(self):
        self.control = _completed("akp-control", "akj-control")
        self.intervention = _completed("akp-intervention", "akj-intervention")
        self.receipts = {
            "control-diagnostic": _diagnostic(self.control, 0.5),
            "control-outcome": _outcome(self.control, 0.0),
            "intervention-diagnostic": _diagnostic(self.intervention, 1.0),
            "intervention-outcome": _outcome(self.intervention, 0.2),
            "match": {
                "schema": B.MATCH_SCHEMA,
                "intervention_proposal_id": "akp-intervention",
                "control_proposal_id": "akp-control",
                "intervention_completion_event_id": "akj-intervention",
                "control_completion_event_id": "akj-control",
                "candidate_frame_id": "candidate-frame-real-v1",
                "regime": "prefill", "surface": "mul_mat",
                "changed_factor": "ggml_iqk", "one_factor": True,
            },
        }
        contract = self.control.proposal["representation_contract"]
        directions = {
            name: ("lower" if name in {
                "unsupported_scope_width", "k_rho", "raw_impurity"} else "higher")
            for name in L.DIAGNOSTICS
        }
        self.manifest = {
            "schema": B.BUILD_SCHEMA,
            "archive_id": "ak-real-archive-test",
            "created_at": "2026-08-12T00:00:00+00:00",
            "candidate_frame_id": "candidate-frame-real-v1",
            "representation_frame_sha256": contract["frame_sha256"],
            "empirical_demand_weights_sha256": contract["empirical_demand"][
                "weights_sha256"],
            "metric_direction": "higher_better",
            "diagnostic_directions": directions,
            "outcome_weights": {
                "heldout_regime_transfer": 0.5,
                "falsifier_resolution": 0.5,
            },
            "rows": [
                {
                    "proposal_id": "akp-control",
                    "diagnostic_receipt": {"key": "control-diagnostic"},
                    "outcome_receipt": {"key": "control-outcome"},
                },
                {
                    "proposal_id": "akp-intervention",
                    "matched_control_id": "akp-control",
                    "diagnostic_receipt": {"key": "intervention-diagnostic"},
                    "outcome_receipt": {"key": "intervention-outcome"},
                    "matched_intervention_receipt": {"key": "match"},
                },
            ],
        }

    def completed(self, row):
        return (self.control if row["proposal_id"] == "akp-control"
                else self.intervention)

    def bound(self, binding, **_kwargs):
        key = binding["key"]
        return self.receipts[key], {"path": f"/evidence/{key}.json", "sha256": "a" * 64}

    def build(self):
        with mock.patch.object(B, "_completed_proposal", side_effect=self.completed), \
                mock.patch.object(B, "_bound_receipt", side_effect=self.bound):
            return B.build_archive(self.manifest)

    def test_real_join_projects_a_protocol_valid_observe_only_archive(self):
        archive = self.build()
        self.assertEqual(L.validate_archive(archive), [])
        self.assertEqual(len(archive["rows"]), 2)
        self.assertEqual(archive["rows"][1]["matched_control_id"], "akp-control")
        self.assertEqual(
            archive["rows"][1]["source_receipts"]["campaign_result_sha256"],
            self.intervention.result_sha256)

    def test_mismatched_demand_frame_is_refused(self):
        self.receipts["intervention-outcome"] = copy.deepcopy(
            self.receipts["intervention-outcome"])
        self.receipts["intervention-outcome"][
            "empirical_demand_weights_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "demand_frame"):
            self.build()

    def test_a_control_cannot_claim_a_match_receipt(self):
        self.manifest["rows"][0]["matched_intervention_receipt"] = {"key": "match"}
        with self.assertRaisesRegex(ValueError, "control row"):
            self.build()


if __name__ == "__main__":
    unittest.main()
