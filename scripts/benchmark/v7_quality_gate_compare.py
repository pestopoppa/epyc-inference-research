#!/usr/bin/env python3
"""V7 quality-gate comparator: per-suite accuracy regression check.

Prevents the PPL-only gaming seen in the Gemma Challenge (lossy submission
held PPL but lost 15 GPQA-Diamond / 40 MMLU-Pro points). Any v7+ kernel
candidate must pass MMLU-Pro + GPQA-Diamond before promotion.

Inputs are two JSON files with per-suite eval results. Both must have the
same suites. The baseline is measured on the current production kernel
(v6); the candidate is measured on the experimental kernel (v7).

Output is a Markdown report + an exit code.

CJ-8 (2026-09-07): THE VERDICT IS THREE-VALUED, and the exit code is 0/1/2.

Before this, three different events all produced ``pass: False`` and exit 1:

  * a real REGRESSION (candidate lost accuracy) — a decision about the kernel;
  * INSUFFICIENT evidence (``candidate_n < min_n``) — no decision at all, and
    the docstring of ``check_suite`` claimed it was "advisory, not blocking"
    while the code blocked on it;
  * a suite MISSING from either file — no decision at all.

They have different remedies (revert the kernel / run more questions / fix the
join) and a two-valued gate cannot tell an operator which. Undecidable inputs are
now ``out-of-coverage`` with a cause code and **exit 2**.

EXIT 2 IS STILL BLOCKING. ``run_v9_quality_gate.sh`` is ``set -e``, so a non-zero
exit aborts the promotion exactly as before. Nothing that used to block now
promotes; the blocking status is merely NAMED.

The one behaviour change in the other direction is deliberate and is a bug fix:
``bl.get("accuracy", 0)`` used to read a MISSING baseline accuracy as 0.0, which
made ``delta = cand_acc - 0.0`` positive and **silently passed the suite**. A
corrupt baseline file disarmed the very gate that exists to stop a quality-losing
kernel from promoting. An absent accuracy is now ``out-of-coverage``/``absent``
and blocks.

  0 = every suite decided and within threshold
  1 = at least one suite DECIDED against the candidate (regression)
  2 = nothing was decided against the candidate, but at least one suite could
      not be decided at all

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
import importlib.util
import json
import sys
from pathlib import Path

# The shared CJ-8 vocabulary, loaded by path: scripts/benchmark/ is not a
# package, and this module is invoked as a script from a promotion shell script.
_GVV_PATH = Path(__file__).resolve().parent / "gate_verdict_vocab.py"
_GVV_SPEC = importlib.util.spec_from_file_location("gate_verdict_vocab", _GVV_PATH)
assert _GVV_SPEC is not None and _GVV_SPEC.loader is not None
gvv = importlib.util.module_from_spec(_GVV_SPEC)
sys.modules.setdefault("gate_verdict_vocab", gvv)
_GVV_SPEC.loader.exec_module(gvv)

PASS = gvv.VERDICT_PASS
FAIL = gvv.VERDICT_FAIL
OUT_OF_COVERAGE = gvv.VERDICT_OUT_OF_COVERAGE

#: 0 pass / 1 fail / 2 could-not-check.
EXIT_PASS = gvv.EXIT_PASS
EXIT_FAIL = gvv.EXIT_FAIL
EXIT_OUT_OF_COVERAGE = gvv.EXIT_OUT_OF_COVERAGE


# ---------------------------------------------------------------------------
# Per-suite regression check
# ---------------------------------------------------------------------------


def check_suite(
    baseline_acc: float | None,
    candidate_acc: float | None,
    baseline_n: int | None,
    candidate_n: int | None,
    regression_threshold: float,
    min_n: int = 50,
) -> tuple[str, str | None, str]:
    """Check a single suite for regression.

    Returns ``(verdict, cause, explanation)`` where ``verdict`` is one of
    ``pass`` / ``fail`` / ``out-of-coverage`` and ``cause`` is a CJ-8 cause code,
    mandatory on ``out-of-coverage`` and ``None`` on the two decided verdicts.

    A REGRESSION is a decision about the candidate:
      candidate_acc < baseline_acc - regression_threshold

    Everything else that stops this function reaching that comparison is NOT a
    decision about the candidate and must not be spelled as one:

    * an ABSENT accuracy on either side — the file did not carry the number;
    * fewer than ``min_n`` questions — the run is too thin to rule either way.

    Both keep BLOCKING (the caller exits non-zero on them); they simply stop
    claiming the candidate lost accuracy.
    """
    if baseline_acc is None or candidate_acc is None:
        which = "baseline" if baseline_acc is None else "candidate"
        return (
            OUT_OF_COVERAGE,
            gvv.CAUSE_ABSENT,
            f"NOT ASSESSABLE: the {which} record carries no 'accuracy' field, so "
            f"no comparison was performed. Previously a missing accuracy read as "
            f"0.0 — on the BASELINE side that made the delta positive and passed "
            f"the suite, disarming the gate.",
        )

    delta = candidate_acc - baseline_acc

    if candidate_n is None:
        return (
            OUT_OF_COVERAGE,
            gvv.CAUSE_ABSENT,
            f"NOT ASSESSABLE: the candidate record carries no 'n', so the "
            f"evidence floor (>= {min_n}) could not be evaluated.",
        )

    if candidate_n < min_n:
        return (
            OUT_OF_COVERAGE,
            gvv.CAUSE_INSUFFICIENT_COVERAGE,
            f"INSUFFICIENT: only {candidate_n} questions (need >= {min_n}); "
            f"accuracy {candidate_acc:.1%} vs baseline {baseline_acc:.1%} "
            f"(delta {delta:+.1%}). This is NOT a regression finding — the run is "
            f"too thin to rule either way.",
        )

    if delta < -regression_threshold:
        return (
            FAIL,
            None,
            f"REGRESSION: {candidate_acc:.1%} vs baseline {baseline_acc:.1%} "
            f"(delta {delta:+.1%}, threshold -{regression_threshold:.1%})",
        )

    return (
        PASS,
        None,
        f"OK: {candidate_acc:.1%} vs baseline {baseline_acc:.1%} "
        f"(delta {delta:+.1%})",
    )


def _opt_float(record: dict, key: str) -> float | None:
    """Read a numeric field, returning None when it is ABSENT or unreadable.

    Deliberately NOT ``float(record.get(key, 0))``: a suite that never reported
    a number and a suite that genuinely measured 0.0 are different events, and
    collapsing them is how a corrupt file reads as a confident measurement.
    """
    if key not in record or record[key] is None:
        return None
    try:
        return float(record[key])
    except (TypeError, ValueError):
        return None


def _opt_int(record: dict, key: str) -> int | None:
    value = _opt_float(record, key)
    return None if value is None else int(value)


# ---------------------------------------------------------------------------
# Comparison loop + verdict
# ---------------------------------------------------------------------------


def compare(
    baseline: dict,
    candidate: dict,
    regression_threshold: float,
    min_n: int = 50,
) -> tuple[list[dict], dict, str, str]:
    """Per-suite comparison. Returns ``(rows, summary, verdict, verdict_text)``.

    ``verdict`` is the CJ-8 three-valued fold over the per-suite verdicts:

    * a DECIDED ``fail`` outranks everything -- if any suite ran and rejected the
      candidate, that is the finding and the undecided remainder is noise beside
      it;
    * otherwise, any undecided suite makes the WHOLE comparison undecided
      (``out-of-coverage``), because a gate that could not read part of its
      surface has not cleared the candidate on that part;
    * only an all-decided, all-within-threshold comparison is ``pass``.

    Note the ordering: undecided never becomes ``pass``. The gate still blocks.
    """
    baseline_suites = {s["suite"]: s for s in baseline.get("suites", [])}
    candidate_suites = {s["suite"]: s for s in candidate.get("suites", [])}

    all_suites = sorted(set(baseline_suites) | set(candidate_suites))

    rows: list[dict] = []
    n_pass = 0
    n_fail = 0
    n_out_of_coverage = 0
    by_cause: dict[str, int] = {}

    def _record(row: dict) -> None:
        nonlocal n_pass, n_fail, n_out_of_coverage
        rows.append(row)
        if row["verdict"] == PASS:
            n_pass += 1
        elif row["verdict"] == FAIL:
            n_fail += 1
        else:
            n_out_of_coverage += 1
            by_cause[row["cause"]] = by_cause.get(row["cause"], 0) + 1

    for suite in all_suites:
        bl = baseline_suites.get(suite)
        cand = candidate_suites.get(suite)

        # CJ-8. A suite present on only one side was never COMPARED. That says
        # nothing about the candidate's quality; it says the two runs cover
        # different surfaces. It still blocks -- it is simply no longer counted
        # as a suite the candidate failed.
        if bl is None or cand is None:
            which = "baseline" if bl is None else "candidate"
            present = cand if bl is None else bl
            _record({
                "suite": suite,
                "baseline_acc": None if bl is None else _opt_float(bl, "accuracy"),
                "candidate_acc": None if cand is None else _opt_float(cand, "accuracy"),
                "baseline_n": None if bl is None else _opt_int(bl, "n"),
                "candidate_n": None if cand is None else _opt_int(cand, "n"),
                "delta": None,
                "verdict": OUT_OF_COVERAGE,
                "cause": gvv.CAUSE_ABSENT,
                "pass": False,
                "status": (
                    f"NOT ASSESSABLE: suite missing from {which}; the two runs "
                    f"do not cover the same surface, so no comparison exists "
                    f"(n on the present side: {_opt_int(present, 'n')})"
                ),
            })
            continue

        bl_acc = _opt_float(bl, "accuracy")
        cand_acc = _opt_float(cand, "accuracy")
        bl_n = _opt_int(bl, "n")
        cand_n = _opt_int(cand, "n")

        verdict, cause, explanation = check_suite(
            bl_acc, cand_acc, bl_n, cand_n,
            regression_threshold, min_n,
        )

        delta = None if (bl_acc is None or cand_acc is None) else cand_acc - bl_acc

        _record({
            "suite": suite,
            "baseline_acc": bl_acc,
            "candidate_acc": cand_acc,
            "baseline_n": bl_n,
            "candidate_n": cand_n,
            "delta": delta,
            "verdict": verdict,
            "cause": cause,
            # `pass` is retained for readers that only ever asked "may this
            # promote", and it answers no for BOTH a regression and an
            # undecidable suite. The three-valued split rides `verdict`.
            "pass": verdict == PASS,
            "status": explanation,
        })

    summary = {
        "n_suites": len(all_suites),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_out_of_coverage": n_out_of_coverage,
        "out_of_coverage_by_cause": dict(sorted(by_cause.items())),
        # Retained key: it used to mean "suites absent from the baseline". It now
        # means every suite that could not be decided, of which that is one kind.
        "n_missing": n_out_of_coverage,
        "regression_threshold": regression_threshold,
        "min_n": min_n,
        "resolved_coverage": (
            (n_pass + n_fail) / len(all_suites) if all_suites else None
        ),
    }

    # A DECIDED fail outranks an undecided suite. Only when nothing was decided
    # against the candidate does the undecided mass set the verdict -- and it
    # sets it to `out-of-coverage`, never to `pass`.
    if not all_suites:
        gate_verdict = OUT_OF_COVERAGE
        verdict_text = (
            "NOT ASSESSABLE: neither file declared any suite, so the comparison "
            "asserted NOTHING. A gate that asserted nothing has not passed -- it "
            "has not run."
        )
    elif n_fail > 0:
        gate_verdict = FAIL
        verdict_text = (
            f"FAIL: {n_fail} suite(s) regressed beyond the "
            f"-{regression_threshold:.1%} threshold. "
            f"{n_pass}/{summary['n_suites']} passed, "
            f"{n_out_of_coverage} not decided."
        )
    elif n_out_of_coverage > 0:
        top = ", ".join(f"{c}={n}" for c, n in sorted(by_cause.items())) or "none"
        gate_verdict = OUT_OF_COVERAGE
        verdict_text = (
            f"NOT ASSESSABLE: no suite regressed, but {n_out_of_coverage} of "
            f"{summary['n_suites']} could not be decided (by cause: {top}). "
            f"This BLOCKS promotion -- it is not a pass. Resolved coverage "
            f"{summary['resolved_coverage']:.1%}."
        )
    else:
        gate_verdict = PASS
        verdict_text = (
            f"PASS: all {n_pass}/{summary['n_suites']} suites within "
            f"regression threshold (-{regression_threshold:.1%})."
        )

    return rows, summary, gate_verdict, verdict_text


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
        # Three glyphs, not two: an undecided suite must not wear the same mark
        # as a suite the candidate lost.
        tick = {PASS: "✓", FAIL: "✗"}.get(row["verdict"], "?")
        lines.append(
            f"| {row['suite']} | {bl_disp} | {ca_disp} | {delta_disp} | "
            f"{tick} {row['status']} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Suites evaluated: {summary['n_suites']}")
    lines.append(f"- Passed: {summary['n_pass']}")
    lines.append(f"- Failed (decided regression): {summary['n_fail']}")
    lines.append(f"- Not decided (out-of-coverage): {summary['n_out_of_coverage']}")
    if summary["out_of_coverage_by_cause"]:
        causes = ", ".join(
            f"{c}={n}" for c, n in summary["out_of_coverage_by_cause"].items()
        )
        lines.append(f"  - by cause: {causes}")
    coverage = summary.get("resolved_coverage")
    lines.append(
        "- Resolved coverage: "
        + ("undefined (no suites)" if coverage is None else f"{coverage:.1%}")
    )
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

    rows, summary, verdict, verdict_text = compare(
        baseline, candidate, args.regression_threshold, args.min_n,
    )

    report = render_markdown(
        rows, summary, verdict_text,
        baseline.get("meta", {}), candidate.get("meta", {}),
    )
    args.output.write_text(report)
    print(report)
    # 0 pass / 1 fail / 2 could-not-check. BOTH non-zero codes block: a wrapper
    # testing `!= 0` (run_v9_quality_gate.sh is `set -e`) is unaffected, and one
    # that wants to tell a regression from an unreadable input can now do so.
    return gvv.EXIT_BY_VERDICT[verdict]


if __name__ == "__main__":
    sys.exit(main())
