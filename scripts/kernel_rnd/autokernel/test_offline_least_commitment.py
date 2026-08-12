#!/usr/bin/env python3
"""Pure regression tests for the AP-WM-1 observe-only archive evaluator."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

try:
    from . import offline_least_commitment as L
    from . import schemas as S
except ImportError:
    import offline_least_commitment as L
    import schemas as S


def _contract() -> dict:
    contract = {
        "vocabulary": {
            "regimes": ["decode", "prefill"],
            "surfaces": ["mul_mat", "rms_norm"],
            "outcomes": ["heldout_regime_transfer", "falsifier_resolution"],
            "contradictions": ["counter_did_not_move", "heldout_regression"],
        },
        "vocabulary_source_receipts": ["rcpt-vocabulary-offline-1"],
        "considered_alternatives": ["dispatcher", "fusion", "no_change", "backend_rewrite"],
        "excluded_alternatives": [
            {
                "alternative_id": "backend_rewrite",
                "reason": "not a one-factor intervention",
                "source_receipt_id": "rcpt-exclusion-offline-1",
            }
        ],
        "empirical_demand": {
            "receipt_id": "rcpt-demand-offline-1",
            "weights_sha256": S.content_hash({"decode": 0.6, "prefill": 0.4}),
        },
        "abstraction_construction_cost": {
            "value": 7,
            "unit": "typed_facts",
            "receipt_id": "rcpt-cost-offline-1",
        },
        "canonical_encoding": {
            "encoding_id": "ak-representation-json/v1",
            "schema_sha256": S.content_hash({"schema": "ak-representation-json/v1"}),
        },
        "semantics_preserving_recoding_fixture_ids": ["rename-v1", "permutation-v1"],
    }
    contract["frame_sha256"] = S.representation_frame_sha256(contract)
    return contract


def _diagnostics(score: float) -> dict[str, float]:
    return {
        "unsupported_scope_width": 10.0 - score,
        "compatible_future_mass": score,
        "k_rho": 10.0 - score,
        "information_gain": score,
        "novelty": score,
        "raw_impurity": 10.0 - score,
        "weighted_minority": score,
    }


def _row(
    proposal_id: str,
    regime: str,
    surface: str,
    score: float,
    outcome: float,
    *,
    control: str | None = None,
) -> dict:
    diagnostics = _diagnostics(score)
    return {
        "proposal_id": proposal_id,
        "completion_event_id": f"complete-{proposal_id}",
        "candidate_frame_id": "candidate-frame-fixture-v1",
        "regime": regime,
        "surface": surface,
        "intervention_id": f"intervention-{proposal_id}",
        "changed_factor": "dispatch_predicate",
        "matched_control_id": control,
        "representation_contract": _contract(),
        "diagnostics": diagnostics,
        "outcome": {
            "heldout_regime_transfer": outcome,
            "falsifier_resolution": outcome,
            "noise_floor": 0.01,
        },
        "recodings": {
            "rename-v1": copy.deepcopy(diagnostics),
            "permutation-v1": copy.deepcopy(diagnostics),
        },
    }


def _archive() -> dict:
    rows = []
    for regime, surface, prefix in (
        ("decode", "mul_mat", "d"),
        ("prefill", "rms_norm", "p"),
    ):
        control = f"akp-{prefix}-control"
        rows.append(_row(control, regime, surface, 1.0, 0.0))
        rows.extend(
            [
                _row(f"akp-{prefix}-a", regime, surface, 3.0, 0.30, control=control),
                _row(f"akp-{prefix}-b", regime, surface, 2.0, 0.15, control=control),
                _row(f"akp-{prefix}-c", regime, surface, 0.5, -0.10, control=control),
            ]
        )
    return {
        "schema": L.ARCHIVE_SCHEMA,
        "archive_id": "ak-wm-fixture-20260805",
        "created_at": "2026-08-05T00:00:00+00:00",
        "protocol_id": L.PROTOCOL_ID,
        "authority": L.AUTHORITY,
        "candidate_frame_id": "candidate-frame-fixture-v1",
        "diagnostic_directions": {
            "unsupported_scope_width": "lower",
            "compatible_future_mass": "higher",
            "k_rho": "lower",
            "information_gain": "higher",
            "novelty": "higher",
            "raw_impurity": "lower",
            "weighted_minority": "higher",
        },
        "outcome_weights": {
            "heldout_regime_transfer": 0.5,
            "falsifier_resolution": 0.5,
        },
        "rows": rows,
    }


class OfflineLeastCommitmentTest(unittest.TestCase):
    def test_matched_fixture_emits_full_observe_only_report(self):
        archive = _archive()
        self.assertEqual(L.validate_archive(archive), [])
        report = L.evaluate_archive(archive)
        self.assertEqual(report["pair_count"], 6)
        self.assertFalse(report["live_authority"])
        self.assertEqual(report["evidence_label"], "fixture_or_unlabelled")
        self.assertEqual(report["power"]["status"], "adequately_powered")
        self.assertEqual(report["matched_validation"]["status"], "PASS")
        self.assertEqual(len(report["pair_noise_floors"]), 6)
        self.assertEqual(report["recommendation"], "retain_simpler_baseline")
        self.assertEqual(
            set(report["by_regime_surface"]),
            {
                "decode::mul_mat",
                "prefill::rms_norm",
            },
        )
        self.assertEqual(set(report["by_regime"]), {"decode", "prefill"})
        self.assertEqual(set(report["by_surface"]), {"mul_mat", "rms_norm"})
        for metric in L.DIAGNOSTICS:
            stats = report["overall"][metric]
            self.assertEqual(stats["effective_pairs"], 6)
            self.assertEqual(stats["conditional_predictive_value"], 1.0)
            self.assertEqual(stats["mean_sign_error"], 0.0)
            self.assertEqual(
                [item["kendall_tau"] for item in stats["recoding_stability"]],
                [1.0, 1.0],
            )

    def test_cross_representation_archive_is_refused(self):
        archive = _archive()
        contract = archive["rows"][-1]["representation_contract"]
        contract["empirical_demand"]["receipt_id"] = "rcpt-other-demand"
        contract["frame_sha256"] = S.representation_frame_sha256(contract)
        errors = L.validate_archive(archive)
        self.assertTrue(any("frames differ" in error for error in errors), errors)

    def test_unmatched_archive_is_refused(self):
        archive = _archive()
        for row in archive["rows"]:
            row["matched_control_id"] = None
        errors = L.validate_archive(archive)
        self.assertTrue(any("one-factor" in error for error in errors), errors)

    def test_live_authority_label_is_refused(self):
        archive = _archive()
        archive["authority"] = "selector"
        self.assertIn("authority: must be 'observe_only'", L.validate_archive(archive))

    def test_underpowered_archive_is_explicit(self):
        archive = _archive()
        archive["rows"] = archive["rows"][:2]
        report = L.evaluate_archive(archive)
        self.assertEqual(report["power"]["status"], "underpowered")
        self.assertEqual(report["recommendation"], "underpowered_retain_observe_only")

    def test_synthetic_fixture_cannot_emit_real_label(self):
        archive = _archive()
        with self.assertRaisesRegex(ValueError, "real provenance"):
            L.evaluate_archive(archive, projection={}, real_label=True)

    def test_cli_requires_projection_result(self):
        with tempfile.TemporaryDirectory() as temp:
            archive_path = Path(temp) / "archive.json"
            archive_path.write_text(json.dumps(_archive()), encoding="utf-8")
            with self.assertRaises(SystemExit):
                L.main([str(archive_path)])


if __name__ == "__main__":
    unittest.main()
