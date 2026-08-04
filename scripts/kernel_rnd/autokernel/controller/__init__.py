"""autokernel.controller — the AK4 planner/critic/controller plane.

WHY THIS PACKAGE EXISTS
-----------------------
AK1–AK3 built the substrate: versioned contracts (`schemas`), an append-only
sharded journal (`journal`), durability and quota classes (`storage`), acquired
resource claims (`resource`), and a trusted tiered evaluator (`evaluator`).
None of them WALKS the loop. This package does, and it is the half where the
project's two most expensive AutoPilot scars live:

  * **a control that was requested but never verified.** AutoPilot's pause was a
    silent no-op for months because the run state was cached in memory and
    written back over the operator's change. Design §4 invariant 19 answers it:
    every operator control is *acknowledged in the journal, latched on disk, and
    re-read from disk at the top of each iteration under the write lock*, and an
    unacked control is a HARD failure rather than a slow one. `state_machine`
    makes the cached-state shape structurally impossible — no object here holds a
    `ControlLatch` as attribute state, and `audit_no_cached_control_state()`
    proves it rather than asserting it in prose.
  * **a restart that came up empty with nothing objecting.** 232 trials and about
    16 days of compute vanished because a rebuilt derived view disagreed with the
    record and nothing refused to start. §8.2 step 10 answers it: BOOTSTRAP
    asserts journal/derived-view consistency and REFUSES on disagreement, with a
    deliberate-rebase escape that must state its reason on the record.

AUTHORITY
---------
The LLM proposes and interprets; this package DISPOSES. Every gate, every stop
condition and every state transition is decided by deterministic code from
journaled records (design §8.1 *"The LLM produces proposals and interpretations;
a deterministic controller disposes gates and stop conditions"*, §8.10 *"The LLM
may request a stop. The controller owns disposition from records."*, AK-D4).
Any place an LLM output could decide a transition is a defect.

Governing instrument: `epyc-root/measurement/protocols/kernel-research.md`
(Annex K, **P-AK-SEARCH-1**, RATIFIED 2026-08-03). Nothing here freezes, cuts
over, writes a production tree, or emits a claim: the loop's release-side job
ends at a release package that a human executes (§1.3, invariant 5). T3/T4 are
release instruments owned by AK5; `state_machine` declares the
`SEAL -> T3_RELEASE_GATE -> PACKAGE` seam and REFUSES the tier when no AK5
runner is wired.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md`, phase
AK4 (§6.1, §6.3, §6.5, §8.1–§8.10, §12, §17, §19).
"""
from __future__ import annotations

from . import (
    composition,
    context,
    critic,
    do_not_repeat,
    fingerprint,
    guards,
    hypotheses,
    oracles,
    planner,
    selection,
    state_machine,
)

#: All eleven, not the three that happened to exist when this file was written.
#: The six modules built in parallel each landed after it and none re-exported
#: itself, so `import autokernel.controller` bound a package that could not reach
#: most of its own plane. Importing them here is also what makes the cross-module
#: agreements (`oracles`, `fingerprint`, the shared closure vocabulary) resolve at
#: package import rather than at whichever consumer happened to be imported first.
__all__ = [
    "composition", "context", "critic", "do_not_repeat", "fingerprint", "guards",
    "hypotheses", "oracles", "planner", "selection", "state_machine",
]
