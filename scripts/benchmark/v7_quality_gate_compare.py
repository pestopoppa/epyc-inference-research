#!/usr/bin/env python3
"""V7 quality-gate comparator: per-suite accuracy regression check.

Prevents the PPL-only gaming seen in the Gemma Challenge (lossy submission
held PPL but lost 15 GPQA-Diamond / 40 MMLU-Pro points). Any v7+ kernel
candidate must pass MMLU-Pro + GPQA-Diamond before promotion.

Inputs are two JSON files with per-suite eval results. Both must have the
same suites. The baseline is measured on the current production kernel
(v6); the candidate is measured on the experimental kernel (v7).

Output is a Markdown report + an exit code (0 PASS, 1 FAIL).

Gate criteria (default):
  - Each suite: candidate accuracy >= baseline accuracy - regression_threshold
  - Default regression_threshold = 0.05 (5 percentage points)
  - Both mmlu_pro AND gpqa must pass

Usage:
    v7_quality_gate_compare.py --baseline PATH --candidate PATH --output PATH

Override the gate with --regression-threshold if you want to tighten or
loosen. Default 0.05 is the production threshold for v7 promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-suite regression check
# ---------------------------------------------------------------------------


def check_suite(
    baseline_acc: float,
    candidate_acc: float,
    baseline_n: int,
    candidate_n: int,
    regression_threshold: float,
    min_n: int = 50,
) -> tuple[bool, str]:
    """Check a single suite for regression.

    Returns (pass, explanation).

    Regression is flagged when:
      candidate_acc < baseline_acc - regression_threshold

    Additionally, if the candidate has fewer than `min_n` questions, the
    result is flagged as insufficient evidence (advisory, not blocking).
    """
    delta = candidate_acc - baseline_acc

    if candidate_n < min_n:
        return (
            False,
            f"INSUFFICIENT: only {candidate_n} questions (need >= {min_n}); "
            f"accuracy {candidate_acc:.1%} vs baseline {baseline_acc:.1%} "
            f"(delta {delta:+.1%})",
        )

    if delta < -regression_threshold:
        return (
            False,
            f"REGRESSION: {candidate_acc:.1%} vs baseline {baseline_acc:.1%} "
            f"(delta {delta:+.1%}, threshold -{regression_threshold:.1%})",
        )

    return (
        True,
        f"OK: {candidate_acc:.1%} vs baseline {baseline_acc:.1%} "
        f"(delta {delta:+.1%})",
    )


# ---------------------------------------------------------------------------
# Comparison loop + verdict
# ---------------------------------------------------------------------------


def compare(
    baseline: dict,
    candidate: dict,
    regression_threshold: float,
    min_n: int = 50,
) -> tuple[list[dict], dict, bool, str]:
    """Per-suite comparison. Returns (rows, summary, passed, verdict_text).

    Each suite in both JSON files is compared. A suite is defined as
    regression if candidate accuracy < baseline accuracy - threshold.

    The gate passes only if ALL suites pass.
    """
    baseline_suites = {s["suite"]: s for s in baseline.get("suites", [])}
    candidate_suites = {s["suite"]: s for s in candidate.get("suites", [])}

    all_suites = sorted(set(baseline_suites) | set(candidate_suites))

    rows: list[dict] = []
    n_pass = 0
    n_fail = 0
    n_missing = 0

    for suite in all_suites:
        bl = baseline_suites.get(suite)
        cand = candidate_suites.get(suite)

        if bl is None:
            rows.append({
                "suite": suite,
                "baseline_acc": None,
                "candidate_acc": None,
                "baseline_n": None,
                "candidate_n": None,
                "delta": None,
                "pass": False,
                "status": "missing from baseline",
            })
            n_missing += 1
            continue

        if cand is None:
            rows.append({
                "suite": suite,
                "baseline_acc": bl.get("accuracy"),
                "candidate_acc": None,
                "baseline_n": bl.get("n"),
                "candidate_n": None,
                "delta": None,
                "pass": False,
                "status": "missing from candidate",
            })
            n_fail += 1
            continue

        bl_acc = float(bl.get("accuracy", 0))
        cand_acc = float(cand.get("accuracy", 0))
        bl_n = int(bl.get("n", 0))
        cand_n = int(cand.get("n", 0))

        suite_pass, explanation = check_suite(
            bl_acc, cand_acc, bl_n, cand_n,
            regression_threshold, min_n,
        )

        delta = cand_acc - bl_acc

        rows.append({
            "suite": suite,
            "baseline_acc": bl_acc,
            "candidate_acc": cand_acc,
            "baseline_n": bl_n,
            "candidate_n": cand_n,
            "delta": delta,
            "pass": suite_pass,
            "status": explanation,
        })

        if suite_pass:
            n_pass += 1
        else:
            n_fail += 1

    summary = {
        "n_suites": len(all_suites),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_missing": n_missing,
        "regression_threshold": regression_threshold,
        "min_n": min_n,
    }

    if n_fail > 0 or n_missing > 0:
        passed = False
        verdict = (
            f"FAIL: {n_fail} suite(s) with regression/insufficient evidence, "
            f"{n_missing} missing. {n_pass}/{summary['n_suites']} passed."
        )
    else:
        passed = True
        verdict = (
            f"PASS: all {n_pass}/{summary['n_suites']} suites within "
            f"regression threshold (-{regression_threshold:.1%})."
        )

    return rows, summary, passed, verdict


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_markdown(
    rows: list[dict],
    summary: dict,
    verdict_text: str,
    baseline_meta: dict,
    candidate_meta: dict,
) -> str:
    lines: list[str] = []
    lines.append("# V7 Kernel Quality-Gate Report")
    lines.append("")
    lines.append(f"**Verdict**: {verdict_text}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(
        f"- Baseline kernel: `{baseline_meta.get('kernel','?')}` "
        f"(`{baseline_meta.get('binary','?')}`)"
    )
    lines.append(
        f"- Candidate kernel: `{candidate_meta.get('kernel','?')}` "
        f"(`{candidate_meta.get('binary','?')}`)"
    )
    lines.append(f"- Model(s): `{candidate_meta.get('models','?')}`")
    lines.append(f"- Regression threshold: -{summary['regression_threshold']:.1%}")
    lines.append(f"- Min questions per suite: {summary['min_n']}")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append("| Suite | Baseline Acc | Candidate Acc | Delta | Verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for row in rows:
        bl_disp = f"{row['baseline_acc']:.1%}" if row.get("baseline_acc") is not None else "—"
        ca_disp = f"{row['candidate_acc']:.1%}" if row.get("candidate_acc") is not None else "—"
        delta_disp = f"{row['delta']:+.1%}" if row.get("delta") is not None else "—"
        tick = "✓" if row["pass"] else "✗"
        lines.append(
            f"| {row['suite']} | {bl_disp} | {ca_disp} | {delta_disp} | "
            f"{tick} {row['status']} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Suites evaluated: {summary['n_suites']}")
    lines.append(f"- Passed: {summary['n_pass']}")
    lines.append(f"- Failed: {summary['n_fail']}")
    lines.append(f"- Missing: {summary['n_missing']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="V7 quality-gate comparator: per-suite accuracy regression",
    )
    p.add_argument(
        "--baseline", required=True, type=Path,
        help="Baseline JSON (production kernel, e.g. v6)",
    )
    p.add_argument(
        "--candidate", required=True, type=Path,
        help="Candidate JSON (experimental kernel, e.g. v7)",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output markdown report path",
    )
    p.add_argument(
        "--regression-threshold", type=float, default=0.05,
        help="Max allowed regression per suite (default: 0.05 = 5pp)",
    )
    p.add_argument(
        "--min-n", type=int, default=50,
        help="Minimum questions per suite for binding verdict (default: 50)",
    )
    args = p.parse_args()

    with args.baseline.open() as f:
        baseline = json.load(f)
    with args.candidate.open() as f:
        candidate = json.load(f)

    rows, summary, passed, verdict_text = compare(
        baseline, candidate, args.regression_threshold, args.min_n,
    )

    report = render_markdown(
        rows, summary, verdict_text,
        baseline.get("meta", {}), candidate.get("meta", {}),
    )
    args.output.write_text(report)
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
