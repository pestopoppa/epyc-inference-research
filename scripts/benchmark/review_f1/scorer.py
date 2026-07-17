"""Clean-room review-finding F1 scorer for the EV-13 suite.

Micro-averaged Precision / Recall / F1 of code-review findings against a
human-curated golden set, with DETERMINISTIC criterion/location matching.

This is a clean-room re-implementation of the Factory ``eval_common.py``
scoring rules. The upstream harness is UNLICENSED and is NOT vendored here;
only the (open) methodology is reproduced. Semantics per the 2026-06-03
deep-dive (``research/factory-ai-harvest-2026-06-03.md``, Part 4):

  * TP = a *scored* golden finding matched by >=1 reviewer finding.
    Dedup is BY GOLDEN INDEX: each golden counts at most once.
  * FP = a reviewer finding matching no golden.
  * FN = a scored golden finding never matched.
  * precision = tp/(tp+fp), recall = tp/(tp+fn), f1 = 2PR/(P+R).
  * MICRO-averaged: pool tp/fp/fn across ALL PRs, then compute P/R/F once.
  * LOAD-BEARING RULE: low-severity golden comments count as NEITHER
    TP/FP/FN. A low-severity golden is never an FN; a reviewer finding
    whose *only* match is a low-severity golden is neutral (NOT an FP).
  * Stability protocol: Mean-F1 + population StdDev over >=3 runs.

Matching here is deterministic (criterion + location) so the build/test leg
needs NO inference. The semantic LLM-as-judge matcher + the EV-6 judge-swap
ablation are separate, inference-gated manifest entries; this module is what
they will feed. Absolute F1 is INTERNAL-ONLY and is NOT comparable to the
Factory leaderboard (our diff+context CPU review vs their agentic whole-repo
review).
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple

LOW_SEVERITY = "low"


class Counts(NamedTuple):
    tp: int
    fp: int
    fn: int


def _norm(text: Any) -> str:
    return str(text or "").strip().lower()


def _severity(finding: dict) -> str:
    return _norm(finding.get("severity"))


def _loc_match(a: dict | None, b: dict | None) -> bool:
    """Deterministic location overlap.

    Location-agnostic when either side omits a location (criterion decides).
    Otherwise require same file and, when both give line ranges, an interval
    overlap. A missing end defaults to its own start (single-line finding).
    """
    if a is None or b is None:
        return True
    if a.get("file") != b.get("file"):
        return False
    a_start, b_start = a.get("line_start"), b.get("line_start")
    if a_start is None or b_start is None:
        return True
    a_end = a.get("line_end") if a.get("line_end") is not None else a_start
    b_end = b.get("line_end") if b.get("line_end") is not None else b_start
    return a_start <= b_end and b_start <= a_end


def _matches(finding: dict, golden: dict) -> bool:
    return _norm(finding.get("criterion")) == _norm(golden.get("criterion")) and _loc_match(
        finding.get("location"), golden.get("location")
    )


def score_pr(golden_findings: list[dict], reviewer_findings: list[dict]) -> Counts:
    """Count TP/FP/FN for a single PR under the low-severity-neither rule."""
    scored = [g for g in golden_findings if _severity(g) != LOW_SEVERITY]
    low = [g for g in golden_findings if _severity(g) == LOW_SEVERITY]
    matched = [False] * len(scored)
    tp = fp = 0
    for finding in reviewer_findings:
        hit = None
        for i, golden in enumerate(scored):
            if not matched[i] and _matches(finding, golden):
                hit = i
                break
        if hit is not None:
            matched[hit] = True
            tp += 1
            continue
        # A reviewer finding whose only match is a low-severity golden is neutral.
        if any(_matches(finding, golden) for golden in low):
            continue
        fp += 1
    fn = matched.count(False)
    return Counts(tp=tp, fp=fp, fn=fn)


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def micro_average(pr_counts: Iterable[Counts]) -> dict[str, float | int]:
    """Micro-average: pool TP/FP/FN across PRs, then compute P/R/F once."""
    tp = fp = fn = 0
    for c in pr_counts:
        tp += c.tp
        fp += c.fp
        fn += c.fn
    return prf(tp, fp, fn)


def score_run(cases: list[dict], run_findings: dict[str, list[dict]]) -> dict[str, float | int]:
    """Score one run. ``cases`` carry ``case_id`` + ``golden_findings``;
    ``run_findings`` maps case_id -> reviewer findings for this run."""
    pr_counts = [
        score_pr(case["golden_findings"], run_findings.get(case["case_id"], [])) for case in cases
    ]
    return micro_average(pr_counts)


def aggregate_runs(cases: list[dict], runs: list[dict[str, list[dict]]]) -> dict[str, Any]:
    """Mean-F1 + population StdDev over >=3 runs (the stability protocol)."""
    per_run = [score_run(cases, rf) for rf in runs]
    f1s = [r["f1"] for r in per_run]
    n = len(f1s)
    mean_f1 = sum(f1s) / n if n else 0.0
    std_f1 = (sum((x - mean_f1) ** 2 for x in f1s) / n) ** 0.5 if n else 0.0
    return {
        "n_runs": n,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "per_run": per_run,
        "protocol_ok": n >= 3,
        "note": "internal-only F1; not comparable to the Factory leaderboard",
    }
