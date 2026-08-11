"""autokernel.controller — the hypothesis plane.

WHAT IS LEFT HERE, AND WHY
--------------------------
This package used to hold the AK4 planner/critic/controller plane: `composition`,
`context`, `critic`, `fingerprint`, `guards`, `oracles`, `planner`, `selection`
and `state_machine` — about 20,000 lines encoding research *strategy* in Python.
It was removed on 2026-08-04 and is recoverable from the tag
`autokernel-preserve-20260804`.

The reason is not that strategy-in-code is wrong in principle. It is that
`selection.py` encoded 22 rejection codes and **zero domain knowledge about what
makes an EPYC or MI210 kernel fast**, and `campaign.py`'s real import closure
reached none of this plane — the loop was drivable without any of it. Against
`karpathy/autoresearch`, whose equivalent control surface is 114 lines of prose in
a `program.md` a human edits, 20,000 lines of scaffolding around an empty centre
is the wrong trade. Ours is `../program.md`.

WHAT SURVIVES, AND WHY IT IS NOT STRATEGY
-----------------------------------------
`hypotheses` and `do_not_repeat` are MEMORY, not strategy. They exist because the
operator asked for one specific thing — *"I want to be able to drop in hypotheses,
but ultimately it's the agents that have to iterate on ideas / modifications /
tweaks to find what works"* — and because a loop that cannot tell "tried and
failed" from "never tried" re-runs dead ideas until someone notices.

  * `hypotheses` — the operator-editable store, `Hypothesis`, `Attempt`, and the
    adoption transfer. A falsifier is optional when the operator writes an entry
    and mandatory before a claim is spent on it, so the discipline that stops an
    agent chasing a dead idea does not become a barrier to writing one down.
    Adoption REMOVES the entry from the store, because a hypothesis an agent has
    picked up is the agents'. Journal first, then remove: a crash leaves a
    detectable duplicate and never a loss.
  * `do_not_repeat` — the ledger `check_do_not_repeat()` had always consumed and
    that nothing had ever built. A negative result against an anchor that has
    since moved is a SUPERSEDED_FACT, not a MATCHED_NEGATIVE; a measurement taken
    under contention is CONFOUNDED_RESULT and closes nothing.

Neither decides what to try next. That is the agent's job, per the operator, and
that is the whole point of the removal above.

`authoring_contract` is the deliberate off-campaign-path adapter seam. It
provides prompt hygiene, priced context, reversible log compaction, and
structured external-number validation without selecting work or calling a
model, so it does not restore the deleted strategy plane.

ON THE CAMPAIGN PATH SINCE 2026-08-04
-------------------------------------
`campaign.py` imports both. It had not, and that was the defect:
`hypotheses.claim_for_hypothesis` calls itself *"The ONLY route from a hypothesis
to a resource claim"* and had ZERO non-test callers, because the driver acquired
its region claim directly. The driver is what SPENDS the claim, so
`campaign.py --hypothesis` now acquires through the gate, and `do_not_repeat`
comes with it — `authorize_claim(ledger=…)` has no default and a token carrying
no verdict is refused at the door, so no spendable token exists without a real
ledger.

Note what this does NOT change: neither module ranks anything, and neither is
consulted about what to try. The campaign path reaches the gate that decides
whether a claim may be spent on a question — not a strategy plane.

Governing instrument: `epyc-root/measurement/protocols/kernel-research.md`
(Annex K, **P-AK-SEARCH-1**, RATIFIED 2026-08-03).
"""
from __future__ import annotations

from . import do_not_repeat, hypotheses

__all__ = ["do_not_repeat", "hypotheses"]
