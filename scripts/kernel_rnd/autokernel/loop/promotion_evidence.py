"""S3-AKU-07: promotion evidence from held-out launches only.

The promotion record's compounded gain (and any headline) is computed ONLY from fresh
launches that are disjoint from every A/B pair that decided a keep, against the frozen
production binary pinned by commit + recipe hash. Pair ids are recorded on both sides so
disjointness is checkable. A computation that reuses a keep-deciding pair REFUSES: the
selection data would be reported as if it were an independent estimate (ERA separates
selection from report, intake-1370#record). S3-AKU-03 sets how MANY launches; this sets
WHICH.
"""
from __future__ import annotations

import math
import re
from typing import Mapping, Sequence

from . import goodhart_count

SCHEMA = "epyc.autokernel.promotion_evidence.v1"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class PromotionEvidenceRefusal(ValueError):
    """The promotion computation was refused; the message names the reason."""


def compound_from_heldout(*, launches: Sequence[Mapping], deciding_pair_ids: Sequence[str],
                          production_commit: str, recipe_hash: str, min_launches: int = 1,
                          goodhart: Mapping | None = None) -> dict:
    """Compounded gain over held-out launches.

    Each launch: {"pair_id": str, "effect": float (candidate/production - 1),
    "production_commit": str, "recipe_hash": str}. Gain is the product of (1+effect)
    minus one. Refuses on any overlap with `deciding_pair_ids`, a duplicate launch pair,
    a launch pinned to a different production binary/recipe, a non-finite effect, or
    fewer than `min_launches` launches.

    S3-AKU-13: `goodhart` is the `goodhart_count.count_goodharted` result for the same
    bundle's leave-one-out arms. When given it is validated and recorded under the
    additive `goodharted_count`/`goodhart` fields; it never changes the gain. When
    omitted the record carries `goodharted_count: None` (not measured), never 0.
    """
    if goodhart is not None:
        if (not isinstance(goodhart, Mapping) or goodhart.get("schema") != goodhart_count.SCHEMA
                or type(goodhart.get("goodharted_count")) is not int
                or type(goodhart.get("keeps")) is not int
                or not 0 <= goodhart["goodharted_count"] <= goodhart["keeps"]
                or len(goodhart.get("goodharted_keep_ids") or ()) != goodhart["goodharted_count"]):
            raise PromotionEvidenceRefusal("goodhart count is not a valid goodhart_count.v1 result")
    if not _COMMIT.match(str(production_commit)) or not _HASH.match(str(recipe_hash)):
        raise PromotionEvidenceRefusal("production binary must be pinned by 40-hex commit and 64-hex recipe hash")
    if not deciding_pair_ids:
        raise PromotionEvidenceRefusal("deciding pair ids are required so disjointness is checkable")
    deciding = set(deciding_pair_ids)
    ids = [str(l.get("pair_id") or "") for l in launches]
    if any(not i for i in ids):
        raise PromotionEvidenceRefusal("every launch must carry a pair_id")
    if len(set(ids)) != len(ids):
        raise PromotionEvidenceRefusal("duplicate launch pair_id")
    reused = sorted(set(ids) & deciding)
    if reused:
        raise PromotionEvidenceRefusal(f"launch reuses keep-deciding pair(s): {reused}")
    if len(ids) < min_launches:
        raise PromotionEvidenceRefusal(f"{len(ids)} held-out launches < required {min_launches}")
    gain = 1.0
    for l in launches:
        if l.get("production_commit") != production_commit or l.get("recipe_hash") != recipe_hash:
            raise PromotionEvidenceRefusal(f"launch {l['pair_id']} is not pinned to the frozen production binary/recipe")
        e = l.get("effect")
        if isinstance(e, bool) or not isinstance(e, (int, float)) or not math.isfinite(e) or e <= -1:
            raise PromotionEvidenceRefusal(f"launch {l['pair_id']} has an invalid effect {e!r}")
        gain *= 1.0 + e
    return {"schema": SCHEMA, "production_commit": production_commit, "recipe_hash": recipe_hash,
            "heldout_pair_ids": ids, "deciding_pair_ids": sorted(deciding),
            "launches": len(ids), "compounded_gain": gain - 1.0,
            "goodharted_count": None if goodhart is None else goodhart["goodharted_count"],
            "goodhart": None if goodhart is None else dict(goodhart)}
