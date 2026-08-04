"""The two things the hypothesis plane needs from the plane that was removed.

WHY THIS MODULE EXISTS — and it is the whole refactoring lesson in one file
--------------------------------------------------------------------------
On 2026-08-04 the AK4 strategy plane (`state_machine`, `planner`, `critic`,
`selection`, `composition`, `context`, `guards`, `oracles`, `fingerprint` —
~20,000 lines) was removed; `hypotheses` and `do_not_repeat` were kept, because
the operator asked for a hypothesis drop-in and because a loop that cannot tell
"tried and failed" from "never tried" re-runs dead ideas forever.

The removal broke, and the break is instructive. The two survivors reached into
the removed plane for exactly **two small things**:

  * `state_machine.ControllerError` — a two-line base exception
  * `fingerprint.selection_block()` — a four-line accessor

Twenty thousand lines could not be removed because of six lines buried inside
them. That is what the refactor review means by concerns being orthogonal to the
module tree: `schemas.py` is the only module every plane may import, it holds
record shapes rather than behaviour, so anything needing a base class or a shared
accessor had nowhere to live and ended up wherever it was first written.

The fix is not to restore the plane. It is to put the six lines where a shared
concern belongs.

CORRECTION, 2026-08-04
----------------------
This module first shipped claiming `canonical_items()` and `mechanism_facets()`
had no caller and were therefore left behind. That was wrong: it was concluded
from a grep of NON-TEST code. `test_do_not_repeat` calls `mechanism_facets` to
assert that the ledger's identity extraction agrees with the package's existing
answer to "where does a proposal's structural identity live" — which is exactly
the cross-module agreement worth keeping, and exactly the kind of check that gets
silently dropped when a module is removed. They are here now.

`proposal_fingerprint` came with them because it is the reason the pair exists:
one algorithm, so a skip recorded by two different paths counts as one concept.
"""
from __future__ import annotations

from typing import Any, Mapping

from .. import schemas

__all__ = ["ControllerError", "LEDGER_DIMENSIONS", "SELECTION_BLOCK_KEY",
           "canonical_items", "mechanism_facets", "mechanism_fingerprint",
           "proposal_fingerprint", "selection_block"]


class ControllerError(Exception):
    """Base for every refusal the controller plane raises.

    Verbatim from the removed `state_machine.py:123`. Kept as a distinct class
    rather than folded into `Exception` so a driver can still catch controller
    refusals separately from programming errors — which is what
    `hypotheses.HypothesisError` extends it for.
    """


#: The key §7.1 puts the selection block under. Verbatim from the removed
#: `fingerprint.py`; a second spelling of this string is a suppression that
#: silently never matches.
SELECTION_BLOCK_KEY = "selection"


def selection_block(proposal: Mapping[str, Any]) -> Mapping[str, Any]:
    """The `selection` block, or an empty mapping. Never raises on a partial draft.

    Verbatim from the removed `fingerprint.py:66`. Tolerance of a partial draft is
    load-bearing rather than lax: the caller that most needs to look up a prior
    attempt is the one holding a REJECTED draft, and raising here would deny the
    do-not-repeat check to precisely the proposals it exists to catch.
    """
    block = proposal.get(SELECTION_BLOCK_KEY)
    return block if isinstance(block, Mapping) else {}


#: The regime dimensions a do-not-repeat match may key on. Verbatim from the
#: removed `selection.py:427`, and it was in the wrong module there: it describes
#: what the LEDGER keys on, and `do_not_repeat` is the only surviving consumer.
#:
#: §19.2 states the reason it must exist at all, and it is the most expensive
#: lesson in the ledger: *"'Do not repeat' without regime identity is dangerous
#: because this project repeatedly observes SIGN CHANGES across architecture,
#: substrate, batch, context, and quant."* A suppression that ignores the regime
#: does not merely over-reject — it rejects an idea that would have WON here on
#: the evidence that it lost somewhere else.
LEDGER_DIMENSIONS = frozenset({
    "backend", "phase", "regimes", "shapes", "models", "ops", "quant", "batch",
    "context", "change_class", "hierarchy_layer", "architecture", "substrate",
})


def canonical_items(values: Any) -> list:
    """Canonical ITEMS for one facet, whether it arrived as a scalar or a set.

    A scalar goes through `canonical_json` exactly as a collection member does.
    Returning the bare `str(values)` for a scalar would put the two sides of every
    ledger comparison in different encodings — `"dispatcher"` against
    `'"dispatcher"'` — so a suppression keyed on a scalar dimension would silently
    never match, and `1` would compare equal to `"1"`.
    """
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        text = values.decode("utf-8", "replace") if isinstance(values, bytes) else values
        return [schemas.canonical_json(text)]
    if isinstance(values, Mapping):
        return sorted(schemas.canonical_json({k: values[k]}) for k in values)
    if isinstance(values, (list, tuple, set, frozenset)):
        return sorted(schemas.canonical_json(v) for v in values)
    return [schemas.canonical_json(values)]


def mechanism_facets(proposal: Mapping[str, Any]) -> dict:
    """The STRUCTURAL identity of what a proposal proposes.

    Prose is excluded on purpose. `hypothesis`, `narrative` and
    `change.conceptual_change` are the fields a planner rewords between attempts,
    and a fingerprint that included them would let attempt 119 look novel — which
    is exactly the AutoPilot failure §8.4 cites. What survives is the closed
    vocabulary and the sorted structural sets.

    Tolerant of a partially-built manifest on purpose: the caller that most needs
    a fingerprint is the one holding a REJECTED draft.
    """
    if not isinstance(proposal, Mapping):
        raise TypeError("proposal must be a mapping")
    block = selection_block(proposal)
    target = proposal.get("target") if isinstance(proposal.get("target"), Mapping) else {}
    mech = proposal.get("mechanism_prediction")
    mech = mech if isinstance(mech, Mapping) else {}
    change = proposal.get("change") if isinstance(proposal.get("change"), Mapping) else {}
    identity = block.get("regime_identity")
    return {
        "mechanism": block.get("mechanism"),
        "hierarchy_layer": block.get("hierarchy_layer"),
        "change_class": proposal.get("change_class"),
        "campaign_kind": proposal.get("campaign_kind"),
        "bottleneck_before": mech.get("bottleneck_before"),
        "counters": canonical_items(sorted(mech.get("expected_counter_changes") or {})),
        "ops": canonical_items(target.get("ops")),
        "regimes": canonical_items(target.get("regimes")),
        "shapes": canonical_items(target.get("shapes")),
        "models": canonical_items(target.get("models")),
        "symbols": canonical_items(change.get("files_and_symbols")),
        "regime_identity": (
            {k: sorted(canonical_items(v)) for k, v in sorted(identity.items())}
            if isinstance(identity, Mapping) else None
        ),
    }


def mechanism_fingerprint(facets: Mapping[str, Any]) -> str:
    """A stable content hash over the structural facets. Deterministic by
    construction: `schemas.canonical_json` sorts keys and refuses NaN."""
    if not isinstance(facets, Mapping):
        raise TypeError("facets must be a mapping")
    return schemas.content_hash({k: facets[k] for k in sorted(facets)})


def proposal_fingerprint(proposal: Mapping[str, Any]) -> str:
    """The value that goes in `PROPOSAL_SKIPPED.payload["fingerprint"]`. One
    algorithm, so a skip recorded by the planner adapter and a skip recorded by
    the screener are counted as the same concept."""
    return mechanism_fingerprint(mechanism_facets(proposal))
