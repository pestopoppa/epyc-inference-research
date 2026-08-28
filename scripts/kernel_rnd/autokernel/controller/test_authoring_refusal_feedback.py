"""The planner must be told WHY its authored diff was refused.

Regression test for a defect measured on campaign v33: `prior_authoring_refusals`
filtered on `status == "planner_refused"`, a status written almost never, while the
status actually produced by a rejected diff is `authoring_refused`. Across campaigns
v28-v34 the tally was 22 `authoring_refused` against 1 `planner_refused`, so the
planner was told nothing for 22 of 23 authoring failures and re-derived rejected work
blind.

The v33 evidence is reproduced verbatim below: two turns, two different
`operation_key`s, one byte-identical refusal reason, on the same hypothesis -- which
then consumed the 3-strike `bounded_authoring_skip` and retired the hypothesis for the
campaign without ever testing it.
"""
from __future__ import annotations

import unittest

from . import discovery_controller as D


#: Verbatim from v33 (gpu-discovery-champion-v33), turns 2 and 3.
V33_REASON = ("committed diff in 'ggml/src/ggml-cuda/vecdotq.cuh' derives undeclared "
              "symbols ['<file-scope>']")


def _rows() -> list[dict]:
    return [
        {"turn": 1, "status": "planner_transient",
         "reason": "actor transport failure: CodexContainerError: timed out",
         "portfolio_hypothesis_id": "akh-v2-q5-type-specific-dequant"},
        {"turn": 1, "status": "critic_revise",
         "portfolio_hypothesis_id": "akh-v2-q5-type-specific-dequant"},
        {"turn": 2, "status": "authoring_refused", "reason": V33_REASON,
         "portfolio_hypothesis_id": "akh-v2-q5-type-specific-dequant",
         "operation_key": "203269626b9cc5563451083b279a5019f888c613954f8cde5f2a7e0fcf4fd1ec"},
        {"turn": 3, "status": "authoring_refused", "reason": V33_REASON,
         "portfolio_hypothesis_id": "akh-v2-q5-type-specific-dequant",
         "operation_key": "bac604b84224c7bf82800f2ff0278d3acea0bf79a997d614fa1c24f336076a56"},
    ]


def _fed_back(rows: list[dict]) -> list[dict]:
    """The projection `_planner_round` performs when building the prompt."""
    return [
        {key: row.get(key) for key in (
            "turn", "status", "reason", "portfolio_hypothesis_id", "context_sha256")}
        for row in rows if row.get("status") in D.AUTHORING_REFUSAL_STATUSES
    ][-8:]


class AuthoringRefusalFeedbackTests(unittest.TestCase):

    def test_authoring_refused_reaches_the_planner(self):
        """The regression itself: this status carries the diagnostic and was dropped."""
        fed = _fed_back(_rows())
        self.assertEqual([r["turn"] for r in fed], [2, 3])
        for row in fed:
            self.assertEqual(row["reason"], V33_REASON,
                             "the planner must receive the REASON, not just the fact")

    def test_the_old_filter_would_have_fed_back_nothing(self):
        """Pins why this went unnoticed: the old predicate matches none of v33's rows."""
        old = [r for r in _rows() if r.get("status") == "planner_refused"]
        self.assertEqual(old, [], "v33 produced no planner_refused row at all")

    def test_transient_and_critic_rows_are_not_authoring_refusals(self):
        """A provider outage is not authored output; a critic revision is not a refusal
        of the diff's validity. Neither belongs in this feedback channel."""
        for status in ("planner_transient", "critic_revise", "critic_reject",
                       "candidate", "inconclusive"):
            self.assertNotIn(status, D.AUTHORING_REFUSAL_STATUSES, status)

    def test_every_authoring_failure_status_is_covered(self):
        """The feedback set must cover the statuses the controller really writes for a
        rejected diff; a status missing here is silent lost feedback, which is exactly
        the defect this test exists for."""
        for status in ("authoring_refused", "authorization_refused",
                       "planner_contract_refused", "candidate_semantic_repeat_refused",
                       "portfolio_dnr_refused", "planner_refused"):
            self.assertIn(status, D.AUTHORING_REFUSAL_STATUSES, status)


if __name__ == "__main__":
    unittest.main()
