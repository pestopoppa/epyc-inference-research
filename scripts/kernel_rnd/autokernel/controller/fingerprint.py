"""fingerprint.py — the ONE identity a filtered proposal is journaled under (§8.4).

WHY THIS MODULE EXISTS
----------------------
§8.4 says a filtered proposal is journaled *"with its reason, fingerprinted"*,
that *"a repeated fingerprint auto-blacklists"*, and §8.10 makes a RUN of
repeated fingerprints the evidence for `PLANNER_DEGRADED`. All three read one
journal field: `PROPOSAL_SKIPPED.payload["fingerprint"]`.

`planner.py` and `selection.py` were built in parallel and each shipped a
function called `proposal_fingerprint`, computing DIFFERENT digests over the same
manifest, both documented as that field:

  * `planner`'s hashed `change.conceptual_change` — free prose — so rewording the
    sentence minted a new fingerprint. That is exactly attempt 119 looking novel,
    the AutoPilot failure §8.4 cites by name.
  * `selection`'s excluded prose and hashed the structural facets.

Nothing in either module could see the disagreement, because each was consistent
with itself. End to end it was load-bearing: the planner adapter fingerprints a
rejected draft (`ProposalRejected.fingerprint`), the screener fingerprints a
filtered one, both land in the same field, and `selection.read_skip_history()`
counts them in ONE dict against a threshold of two. Two skips of one concept
counted 1 + 1, the auto-blacklist never fired, and the degradation run was
computed over a key the record did not use.

So the algorithm lives here once. It is deliberately the PROSE-FREE one:
`hypothesis`, `narrative` and `change.conceptual_change` are the fields a planner
rewords between attempts, and a blacklist a reworder can walk around is not a
blacklist. What survives is the closed vocabulary and the sorted structural sets.

WHAT THIS MODULE IS NOT
-----------------------
It decides nothing and journals nothing. It imports only `schemas`, so both
`planner` (which `selection` imports) and `selection` can depend on it without a
cycle — the reason the algorithm could not simply live in either of them.

Owning design: `epyc-root/handoffs/active/autokernel-research-loop.md` §8.4,
§8.10, §19.2.
"""
from __future__ import annotations

from typing import Any, Mapping

from .. import schemas

__all__ = [
    "SELECTION_BLOCK_KEY", "FACET_KEYS", "canonical_items", "selection_block",
    "mechanism_facets", "mechanism_fingerprint", "proposal_fingerprint",
]

#: §7.1's planner-authored selection block, where the mechanism label, the
#: hierarchy layer and the regime identity live.
SELECTION_BLOCK_KEY = "selection"

#: Every facet the fingerprint covers. Enumerated so a test can assert the SET
#: rather than a digest: a digest test tells you the algorithm changed, this tells
#: you what it now covers.
FACET_KEYS = (
    "mechanism", "hierarchy_layer", "change_class", "campaign_kind",
    "bottleneck_before", "counters", "ops", "regimes", "shapes", "models",
    "symbols", "regime_identity",
)


def selection_block(proposal: Mapping[str, Any]) -> Mapping[str, Any]:
    """The `selection` block, or an empty mapping. Never raises on a partial draft."""
    block = proposal.get(SELECTION_BLOCK_KEY)
    return block if isinstance(block, Mapping) else {}


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
