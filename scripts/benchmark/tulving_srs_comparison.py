"""SRS comparison eligibility only; does not compute metrics or project claims."""
from __future__ import annotations
import math
BINS = ("0", "1", "2", "3-5", "6+")
def comparison_basis(summary: dict) -> tuple:
    if summary.get("scorer_version") != 2:
        raise ValueError("comparison requires scorer version 2")
    if summary.get("gold_binding") != "chapter_set_v1":
        raise ValueError("comparison requires explicit chapter-set gold binding")
    basis = summary.get("simple_recall_bin_basis")
    if basis not in ("nb_events", "nb_gt_fallback"):
        raise ValueError("comparison refuses missing or mixed bin basis")
    bins = summary.get("simple_recall_bins")
    if not isinstance(bins, dict) or set(bins) != set(BINS):
        raise ValueError("comparison requires all five declared bin rows")
    active = []
    for label in BINS:
        bucket = bins[label]
        if not isinstance(bucket, dict) or type(bucket.get("count")) is not int or bucket["count"] < 0:
            raise ValueError("invalid bin count")
        value = bucket.get("avg_f1")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid bin mean")
        if bucket["count"]:
            active.append(label)
    total = summary.get("simple_recall_questions")
    if type(total) is not int or total != sum(bucket["count"] for bucket in bins.values()) or not active:
        raise ValueError("empty or inconsistent Simple Recall population")
    return summary["scorer_version"], summary["gold_binding"], basis, tuple(active)
def require_comparable_srs(left: dict, right: dict) -> tuple:
    a, b = comparison_basis(left), comparison_basis(right)
    if a != b:
        raise ValueError("SRS quantities differ: scorer, gold-binding rule, bin basis or populated bin set")
    return a
