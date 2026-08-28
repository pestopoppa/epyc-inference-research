"""AK-VIS-2: a critic revision is not an authoring failure.

`_note_portfolio_authoring_failure` charged every non-scientific turn outcome to one
3-strike budget, `critic_revise` included. A critic asking for a revision is the
mechanism by which a proposal is supposed to improve, so counting it there retires
hypotheses for being reviewed.

Measured on v33: one `critic_revise` plus two `authoring_refused` reached
`bounded_authoring_skip`, and hypothesis `akh-v2-q5-type-specific-dequant` was skipped
for the campaign without ever being tested.

Revisions still need a bound -- a planner/critic pair could otherwise ping-pong on one
hypothesis for a whole campaign -- so they get their own larger budget rather than
none. A critic REJECT stays an authoring failure: the critic judged the proposal
unsound, not merely improvable.
"""
from __future__ import annotations

import unittest

from . import discovery_controller as D

HYP = "akh-v2-q5-type-specific-dequant"


def _row(status: str) -> dict:
    return {"status": status, "portfolio_hypothesis_id": HYP}


class CriticRevisionBudgetTests(unittest.TestCase):

    def test_v33_sequence_no_longer_retires_the_hypothesis(self):
        """The exact v33 sequence: revise, refused, refused."""
        state: dict = {}
        for status in ("critic_revise", "authoring_refused", "authoring_refused"):
            D._note_portfolio_authoring_failure(state, _row(status))
        self.assertEqual(state["portfolio_authoring_failures"][HYP], 2,
                         "only the two genuine authoring failures should count")
        self.assertEqual(state.get("portfolio_critic_revisions", {}).get(HYP), 1)
        self.assertNotIn(HYP, state.get("portfolio_skips", {}),
                         "v33's sequence must no longer skip the hypothesis")

    def test_authoring_failures_still_bound_at_their_budget(self):
        """The original protection must survive: repeated bad diffs still stop."""
        state: dict = {}
        for _ in range(D.AUTHORING_FAILURE_BUDGET):
            D._note_portfolio_authoring_failure(state, _row("authoring_refused"))
        skip = state["portfolio_skips"][HYP]
        self.assertEqual(skip["disposition"], "bounded_authoring_skip")
        self.assertIs(skip["scientific_terminal"], False,
                      "a skip is campaign-scoped, never a scientific retirement")

    def test_revisions_are_bounded_too_just_more_generously(self):
        """Not unbounded: a planner/critic pair must not ping-pong forever."""
        state: dict = {}
        for _ in range(D.CRITIC_REVISION_BUDGET):
            D._note_portfolio_authoring_failure(state, _row("critic_revise"))
        skip = state["portfolio_skips"][HYP]
        self.assertEqual(skip["disposition"], "bounded_critic_revision_skip")
        self.assertIs(skip["scientific_terminal"], False)
        self.assertGreater(D.CRITIC_REVISION_BUDGET, D.AUTHORING_FAILURE_BUDGET,
                           "revisions are ordinary iteration and need more room")

    def test_critic_reject_still_counts_as_an_authoring_failure(self):
        """A reject is a judgement that the proposal is unsound, not improvable."""
        state: dict = {}
        D._note_portfolio_authoring_failure(state, _row("critic_reject"))
        self.assertEqual(state["portfolio_authoring_failures"][HYP], 1)
        self.assertNotIn(HYP, state.get("portfolio_critic_revisions", {}))

    def test_the_two_budgets_do_not_share_a_counter(self):
        state: dict = {}
        D._note_portfolio_authoring_failure(state, _row("critic_revise"))
        D._note_portfolio_authoring_failure(state, _row("authoring_refused"))
        self.assertEqual(state["portfolio_critic_revisions"][HYP], 1)
        self.assertEqual(state["portfolio_authoring_failures"][HYP], 1)


if __name__ == "__main__":
    unittest.main()
