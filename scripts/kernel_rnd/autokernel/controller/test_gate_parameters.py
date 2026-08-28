"""A moving gate must be visible in the record, next to the results it moved.

Every `keep` in this project's history came after `drift_bound` was widened 6x
(0.0308 -> 0.1850), and r41/r43 were accepted at drift EXCEEDING r29's, which had
been rejected. Nothing in 107,707 lines of tests caught it, because no record ever
compared one run's thresholds to the previous run's.
"""
import unittest

from autokernel.controller import gate_parameters as gp


def _gate(**values):
    return {"schema": gp.GATE_SCHEMA, "values": values}


class Snapshot(unittest.TestCase):

    def test_the_threshold_is_always_captured(self):
        snap = gp.snapshot(nomination_threshold=0.03, decision_policy=None)
        self.assertEqual(snap["values"]["nomination_threshold"], 0.03)

    def test_policy_floors_are_captured_when_present(self):
        snap = gp.snapshot(nomination_threshold=0.03, decision_policy={
            "nomination_floor_pct": 3.0, "required_replications": 2,
            "irrelevant_key": "ignored"})
        self.assertEqual(snap["values"]["nomination_floor_pct"], 3.0)
        self.assertEqual(snap["values"]["required_replications"], 2)
        self.assertNotIn("irrelevant_key", snap["values"])


class Diff(unittest.TestCase):

    def test_a_loosened_threshold_is_labelled_a_widening(self):
        changes = gp.diff(_gate(nomination_threshold=0.03),
                          _gate(nomination_threshold=0.005))
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["direction"], "WIDENED")
        self.assertAlmostEqual(changes[0]["ratio"], 0.005 / 0.03)
        self.assertEqual(gp.widenings(changes), changes)

    def test_a_tightened_threshold_is_not_a_widening(self):
        changes = gp.diff(_gate(nomination_threshold=0.03),
                          _gate(nomination_threshold=0.05))
        self.assertEqual(changes[0]["direction"], "tightened")
        self.assertEqual(gp.widenings(changes), [])

    def test_the_historical_six_fold_widening_is_reported_with_its_ratio(self):
        """The r37 -> r38 transition, as it would now appear on the record."""
        changes = gp.diff(_gate(nomination_threshold=0.0308,
                                required_replications=10),
                          _gate(nomination_threshold=0.1850,
                                required_replications=15))
        by_name = {change["parameter"]: change for change in changes}
        # The bound rose, which for a floor means it got HARDER, not easier -- the
        # historical drift_bound is a ceiling, so its own direction is recorded by
        # whichever list it belongs to. What matters here is that the change is
        # reported at all, with its magnitude.
        self.assertAlmostEqual(by_name["nomination_threshold"]["ratio"],
                               0.1850 / 0.0308, places=6)
        self.assertAlmostEqual(by_name["required_replications"]["ratio"], 1.5)

    def test_relaxing_a_replication_requirement_is_a_widening(self):
        changes = gp.diff(_gate(required_replications=3),
                          _gate(required_replications=1))
        self.assertEqual(changes[0]["direction"], "WIDENED")

    def test_raising_an_allowance_ceiling_is_a_widening(self):
        changes = gp.diff(_gate(max_replication_spread_pct=10.0),
                          _gate(max_replication_spread_pct=50.0))
        self.assertEqual(changes[0]["direction"], "WIDENED")

    def test_an_unchanged_gate_reports_nothing(self):
        self.assertEqual(gp.diff(_gate(nomination_threshold=0.03),
                                 _gate(nomination_threshold=0.03)), [])

    def test_a_newly_introduced_parameter_is_reported(self):
        changes = gp.diff(_gate(nomination_threshold=0.03),
                          _gate(nomination_threshold=0.03, required_replications=2))
        self.assertEqual([c["parameter"] for c in changes], ["required_replications"])
        self.assertIsNone(changes[0]["before"])

    def test_a_non_numeric_change_is_reported_without_a_ratio(self):
        changes = gp.diff(_gate(sign_policy="strict"), _gate(sign_policy="lenient"))
        self.assertEqual(changes[0]["direction"], "changed")
        self.assertNotIn("ratio", changes[0])

    def test_a_first_run_with_no_predecessor_reports_nothing(self):
        self.assertEqual(gp.diff(None, _gate(nomination_threshold=0.03)), [])

    def test_a_zero_before_does_not_divide(self):
        changes = gp.diff(_gate(nomination_threshold=0.0),
                          _gate(nomination_threshold=0.03))
        self.assertNotIn("ratio", changes[0])

    def test_booleans_are_never_treated_as_numbers(self):
        changes = gp.diff(_gate(terminal_rule=True), _gate(terminal_rule=False))
        self.assertEqual(changes[0]["direction"], "changed")
        self.assertNotIn("ratio", changes[0])


if __name__ == "__main__":
    unittest.main()
