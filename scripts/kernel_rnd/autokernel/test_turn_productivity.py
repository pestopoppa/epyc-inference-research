#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import unittest

from scripts.kernel_rnd.autokernel import schemas
from scripts.kernel_rnd.autokernel import turn_productivity as P
from scripts.kernel_rnd.autokernel.evaluator import api


CAMPAIGN = "ak-productivity-20260811"
STAMP = "2026-08-11T00:00:00+00:00"


def calibration(*, accepted=True) -> api.CalibrationOutputs:
    return api.CalibrationOutputs(
        backend="llama_cpu", phase="decode", cell_class="turn_productivity",
        noise_floor_phi=0.01, b_min_blocks=5,
        alpha_sel=0.5, alpha_conf=0.25,
        anchor_gate_band=(0.98, 1.02), accepted=accepted,
        solve_order_recorded=api.CALIBRATION_SOLVE_ORDER,
        samples_ref="data/aa/productivity.json",
        e_process_construction_id="sign_martingale_fixed_lambda/v1")


def rule() -> P.TurnBudgetRule:
    return P.TurnBudgetRule(
        rule_id="ak-x-6/v1",
        floor=P.ContributionFloor(
            campaign_id=CAMPAIGN, relative_speedup=0.03,
            rationale_ref="campaign-manifest#contribution-floor",
            committed_at=STAMP),
        calibration=P.ProductivityCalibration.from_outputs(
            calibration(), stratum=api.STRATUM_SELECTION))


def observation(sequence, *, proposal, turn, before, correct, speedup,
                task="mul_mat") -> P.TurnObservation:
    return P.TurnObservation(
        sequence=sequence, campaign_id=CAMPAIGN, proposal_id=proposal,
        turn=turn, task_id=task, previous_correct=before, correct=correct,
        relative_speedup=speedup, evidence_ref=f"events/{sequence}.json",
        measured_at=STAMP)


class TurnObservationTest(unittest.TestCase):
    def test_the_split_is_derived_from_the_state_transition(self):
        self.assertEqual(
            observation(0, proposal="a", turn=1, before=False,
                        correct=True, speedup=0.01).turn_class,
            P.RESCUED)
        self.assertEqual(
            observation(0, proposal="a", turn=1, before=True,
                        correct=True, speedup=0.04).turn_class,
            P.PERSISTENT)
        self.assertEqual(
            observation(0, proposal="a", turn=1, before=False,
                        correct=False, speedup=None).turn_class,
            P.FAILED)
        self.assertEqual(
            observation(0, proposal="a", turn=1, before=True,
                        correct=False, speedup=None).turn_class,
            P.REGRESSED)

    def test_an_incorrect_turn_cannot_carry_a_speed_rank(self):
        with self.assertRaisesRegex(P.ProductivityError, "must not carry"):
            observation(0, proposal="a", turn=1, before=False,
                        correct=False, speedup=0.5)

    def test_a_correct_turn_must_record_speedup(self):
        with self.assertRaisesRegex(P.ProductivityError, "needs finite"):
            observation(0, proposal="a", turn=1, before=False,
                        correct=True, speedup=None)


class ArchiveTest(unittest.TestCase):
    def test_append_order_and_state_continuity_are_verified(self):
        rows = (
            observation(0, proposal="a", turn=1, before=False,
                        correct=False, speedup=None),
            observation(1, proposal="a", turn=2, before=False,
                        correct=True, speedup=0.01),
        )
        archive = P.ProductivityArchive(CAMPAIGN, rows)
        self.assertEqual([row.turn_class for row in archive.observations],
                         [P.FAILED, P.RESCUED])
        self.assertRegex(archive.content_hash, r"^[0-9a-f]{64}$")

    def test_sequence_rewrite_is_refused(self):
        with self.assertRaisesRegex(P.ArchiveError, "append order"):
            P.ProductivityArchive(CAMPAIGN, (
                observation(1, proposal="a", turn=1, before=False,
                            correct=False, speedup=None),))

    def test_previous_correct_must_match_the_prior_turn(self):
        with self.assertRaisesRegex(P.ArchiveError, "previous_correct"):
            P.ProductivityArchive(CAMPAIGN, (
                observation(0, proposal="a", turn=1, before=False,
                            correct=True, speedup=0.01),
                observation(1, proposal="a", turn=2, before=False,
                            correct=True, speedup=0.01),
            ))


class CalibrationTest(unittest.TestCase):
    def test_threshold_and_construction_come_from_campaign_calibration(self):
        source = calibration()
        derived = P.ProductivityCalibration.from_outputs(
            source, stratum=api.STRATUM_SELECTION)
        self.assertEqual(derived.threshold, source.threshold_for(api.STRATUM_SELECTION))
        self.assertEqual(derived.construction_id, source.e_process_construction_id)
        self.assertEqual(derived.samples_ref, source.samples_ref)
        self.assertRegex(derived.calibration_sha256, r"^[0-9a-f]{64}$")

    def test_rejected_calibration_cannot_license_the_rule(self):
        with self.assertRaisesRegex(P.ProductivityError, "rejected"):
            P.ProductivityCalibration.from_outputs(
                calibration(accepted=False), stratum=api.STRATUM_SELECTION)


class TurnBudgetTest(unittest.TestCase):
    @staticmethod
    def archive(*, persistent=12, rescued=12, latest_persistent=False):
        rows = []
        for i in range(persistent):
            rows.append(observation(
                len(rows), proposal=f"persistent-{i}", turn=1,
                before=True, correct=True, speedup=0.08))
        for i in range(rescued):
            rows.append(observation(
                len(rows), proposal=f"rescued-{i}", turn=2,
                before=False, correct=True, speedup=0.005))
        if latest_persistent:
            rows.append(observation(
                len(rows), proposal="late-persistent", turn=2,
                before=True, correct=True, speedup=0.08))
        return P.ProductivityArchive(CAMPAIGN, tuple(rows))

    def evaluate(self, archive, *, active_rule=None, commitment=None):
        active_rule = rule() if active_rule is None else active_rule
        commitment = (P.TurnBudgetCommitment.commit(
            active_rule, committed_at=STAMP) if commitment is None else commitment)
        return P.evaluate_turn_budget(
            archive, rule=active_rule, commitment=commitment)

    def test_only_rescued_below_floor_after_persistent_above_becomes_repair_only(self):
        result = self.evaluate(self.archive())
        self.assertEqual(result.decision, P.REPAIR_ONLY)
        self.assertTrue(result.rescued_below_floor.crossed)
        self.assertTrue(result.persistent_above_floor.crossed)
        self.assertEqual(set(result.latest_admitted_classes), {P.RESCUED})
        schemas.canonical_bytes(result.to_dict())

    def test_one_point_below_the_floor_cannot_stop_the_loop(self):
        result = self.evaluate(self.archive(persistent=1, rescued=1))
        self.assertEqual(result.decision, P.CONTINUE_REFINEMENT)
        self.assertFalse(result.rescued_below_floor.crossed)
        self.assertIn("insufficient evidence", result.reasons[-1])

    def test_a_persistent_admission_in_the_latest_turn_keeps_search_live(self):
        result = self.evaluate(self.archive(latest_persistent=True))
        self.assertEqual(result.decision, P.CONTINUE_REFINEMENT)
        self.assertIn(P.PERSISTENT, result.latest_admitted_classes)

    def test_no_persistent_reference_never_shortens_the_budget(self):
        result = self.evaluate(self.archive(persistent=0))
        self.assertEqual(result.decision, P.CONTINUE_REFINEMENT)
        self.assertIsNone(result.persistent_above_floor)

    def test_mutating_the_floor_after_commitment_is_refused(self):
        original = rule()
        commitment = P.TurnBudgetCommitment.commit(original, committed_at=STAMP)
        changed = dataclasses.replace(
            original,
            floor=dataclasses.replace(original.floor, relative_speedup=0.04))
        with self.assertRaises(P.TurnBudgetRuleMutated):
            self.evaluate(self.archive(), active_rule=changed, commitment=commitment)


if __name__ == "__main__":
    unittest.main()
