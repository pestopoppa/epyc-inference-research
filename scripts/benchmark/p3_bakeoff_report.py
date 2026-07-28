#!/usr/bin/env python3
"""Paired bake-off report for P3 (deterministic replay/transform only).

Compares two arms on one duty from banked capture/score artifacts.
Primary statistic: exact two-sided McNemar on paired per-question
correctness (pairing discipline is enforced -- mismatched id sets fail
closed).  Secondary: paired token economics.  Output feeds the P3-2
tenancy decision package; it authorizes NOTHING by itself (D3).

Inputs per suite:
- livecodebench_hard: the capture pq.jsonl per arm (executable-oracle
  ``correct`` recorded at capture time).
- swebench_oracle: the pinned swebench harness report JSON per arm
  (``resolved_ids`` authoritative); capture pq.jsonl supplies token
  economics.  The FG-1 hard-core subset gets a descriptive breakdown.
- p3_cocritic_v1: critic_score.json per arm (p3_bakeoff_critic_score.py);
  paired metric is ``verdict_correct`` (declared estimand).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_bakeoff_common import (  # noqa: E402
    CRITIC_SUITE,
    REPORT_SCHEMA_VERSION,
    load_jsonl,
    mcnemar_exact,
    sha256_file,
    write_json,
)


def correctness_from_capture(pq_path: Path, suite: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in load_jsonl(pq_path):
        if row.get("suite") != suite:
            continue
        out[row["id"]] = bool(row.get("correct"))
    return out


def correctness_from_swe_report(report_path: Path,
                                expected_ids: set[str]) -> dict[str, bool]:
    """Read a swebench harness report; resolved_ids is authoritative."""
    doc = json.loads(report_path.read_text())
    if "resolved_ids" not in doc:
        raise ValueError(f"{report_path}: no resolved_ids key -- not a "
                         "swebench harness report")
    resolved = set(doc["resolved_ids"])
    unknown = sorted(resolved - expected_ids)
    if unknown:
        raise ValueError(f"{report_path}: resolved ids outside the pinned "
                         f"instance set: {unknown[:5]}")
    return {iid: (iid in resolved) for iid in sorted(expected_ids)}


def correctness_from_critic_score(score_path: Path) -> dict[str, bool]:
    doc = json.loads(score_path.read_text())
    return {r["id"]: bool(r["verdict_correct"]) for r in doc["per_row"]}


def tokens_from_capture(pq_path: Path, suite: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in load_jsonl(pq_path):
        if row.get("suite") == suite:
            out[row["id"]] = int(row.get("completion_tokens", 0))
    return out


def paired_compare(a: dict[str, bool], b: dict[str, bool],
                   *, label_a: str, label_b: str,
                   subset: list[str] | None = None) -> dict:
    """Exact McNemar over the paired id set; fail-closed on id mismatch."""
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))[:5]
        only_b = sorted(set(b) - set(a))[:5]
        raise ValueError(
            "pairing violation: id sets differ "
            f"(only-{label_a}: {only_a}, only-{label_b}: {only_b})"
        )
    ids = sorted(a) if subset is None else sorted(set(subset) & set(a))
    if subset is not None and len(ids) != len(subset):
        raise ValueError("subset ids missing from the paired set")
    b_only = sum(1 for i in ids if a[i] and not b[i])
    c_only = sum(1 for i in ids if b[i] and not a[i])
    n = len(ids)
    return {
        "n_pairs": n,
        f"solved_{label_a}": sum(a[i] for i in ids),
        f"solved_{label_b}": sum(b[i] for i in ids),
        "discordant": {f"{label_a}_only": b_only, f"{label_b}_only": c_only},
        "mcnemar_exact_p_two_sided": mcnemar_exact(b_only, c_only),
        "ids_solved_only_by": {
            label_a: sorted(i for i in ids if a[i] and not b[i]),
            label_b: sorted(i for i in ids if b[i] and not a[i]),
        },
    }


def token_economics(tokens: dict[str, int], solved: dict[str, bool]) -> dict:
    values = sorted(tokens.values())
    n_solved = sum(1 for i in tokens if solved.get(i))
    total = sum(values)
    return {
        "n": len(values),
        "median_completion_tokens": statistics.median(values) if values else None,
        "total_completion_tokens": total,
        "tokens_per_solved": round(total / n_solved, 1) if n_solved else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--suite", required=True,
                   choices=["livecodebench_hard", "swebench_oracle", CRITIC_SUITE])
    p.add_argument("--label-a", required=True, help="Arm key A (e.g. stock27b)")
    p.add_argument("--label-b", required=True, help="Arm key B (e.g. ff27b)")
    p.add_argument("--capture-a", type=Path, default=None,
                   help="pq.jsonl for arm A (lcb correctness / swe tokens)")
    p.add_argument("--capture-b", type=Path, default=None)
    p.add_argument("--swe-report-a", type=Path, default=None,
                   help="swebench harness report JSON for arm A")
    p.add_argument("--swe-report-b", type=Path, default=None)
    p.add_argument("--critic-score-a", type=Path, default=None,
                   help="critic_score.json for arm A")
    p.add_argument("--critic-score-b", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    la, lb = args.label_a, args.label_b
    suite = args.suite
    inputs: dict[str, dict] = {}
    tokens_a: dict[str, int] = {}
    tokens_b: dict[str, int] = {}
    extra: dict = {}

    if suite == "livecodebench_hard":
        if not (args.capture_a and args.capture_b):
            p.error("livecodebench_hard requires --capture-a/--capture-b")
        a = correctness_from_capture(args.capture_a, suite)
        b = correctness_from_capture(args.capture_b, suite)
        tokens_a = tokens_from_capture(args.capture_a, suite)
        tokens_b = tokens_from_capture(args.capture_b, suite)
        inputs = {la: {"capture": str(args.capture_a),
                       "sha256": sha256_file(args.capture_a)},
                  lb: {"capture": str(args.capture_b),
                       "sha256": sha256_file(args.capture_b)}}
    elif suite == "swebench_oracle":
        if not (args.swe_report_a and args.swe_report_b):
            p.error("swebench_oracle requires --swe-report-a/--swe-report-b")
        spec = manifest["duties"]["coder"]["suites"]["swebench_oracle"]
        pinned_ids = {
            q["id"] for q in json.loads(
                Path(spec["questions_file"]["path"]).read_text())
        }
        a = correctness_from_swe_report(args.swe_report_a, pinned_ids)
        b = correctness_from_swe_report(args.swe_report_b, pinned_ids)
        inputs = {la: {"swe_report": str(args.swe_report_a),
                       "sha256": sha256_file(args.swe_report_a)},
                  lb: {"swe_report": str(args.swe_report_b),
                       "sha256": sha256_file(args.swe_report_b)}}
        if args.capture_a and args.capture_b:
            tokens_a = tokens_from_capture(args.capture_a, suite)
            tokens_b = tokens_from_capture(args.capture_b, suite)
        hard_core = spec["hard_core_tag"]["ids"]
        extra["hard_core_fg1"] = {
            "ids": hard_core,
            "note": "descriptive only (14 instances unsolved by all six FG-1 arms)",
            "comparison": paired_compare(a, b, label_a=la, label_b=lb,
                                         subset=hard_core),
        }
    else:  # cocritic
        if not (args.critic_score_a and args.critic_score_b):
            p.error(f"{CRITIC_SUITE} requires --critic-score-a/--critic-score-b")
        a = correctness_from_critic_score(args.critic_score_a)
        b = correctness_from_critic_score(args.critic_score_b)
        inputs = {la: {"critic_score": str(args.critic_score_a),
                       "sha256": sha256_file(args.critic_score_a)},
                  lb: {"critic_score": str(args.critic_score_b),
                       "sha256": sha256_file(args.critic_score_b)}}
        extra["calibration"] = {
            la: json.loads(args.critic_score_a.read_text())["summary"],
            lb: json.loads(args.critic_score_b.read_text())["summary"],
        }

    comparison = paired_compare(a, b, label_a=la, label_b=lb)
    if tokens_a and tokens_b:
        extra["token_economics"] = {
            la: token_economics(tokens_a, a),
            lb: token_economics(tokens_b, b),
        }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "grade": "observation",
        "suite": suite,
        "arms": {la: manifest["arms"].get(la, {}).get("label", la),
                 lb: manifest["arms"].get(lb, {}).get("label", lb)},
        "manifest": {"path": str(args.manifest),
                     "sha256": sha256_file(args.manifest)},
        "inputs": inputs,
        "comparison": comparison,
        **extra,
        "statistical_plan": manifest["statistical_plan"],
        "not_authorized": manifest["invariants"]["not_authorized"],
    }
    write_json(args.output, report, sort_keys=False)

    md = args.output.with_suffix(".md")
    c = comparison
    lines = [
        f"# P3 bake-off — {suite}: {la} vs {lb}",
        "",
        f"Generated {report['generated_utc']} · grade: OBSERVATION · "
        "feeds P3-2 decision package only",
        "",
        f"- pairs: {c['n_pairs']}; solved {la}: {c[f'solved_{la}']}, "
        f"{lb}: {c[f'solved_{lb}']}",
        f"- discordants: {la}-only {c['discordant'][f'{la}_only']}, "
        f"{lb}-only {c['discordant'][f'{lb}_only']}; "
        f"exact McNemar p = {c['mcnemar_exact_p_two_sided']:.4g}",
    ]
    if "token_economics" in extra:
        for lbl in (la, lb):
            te = extra["token_economics"][lbl]
            lines.append(
                f"- tokens {lbl}: median {te['median_completion_tokens']}, "
                f"tokens/solved {te['tokens_per_solved']}"
            )
    if "hard_core_fg1" in extra:
        hc = extra["hard_core_fg1"]["comparison"]
        lines.append(
            f"- FG-1 hard-core (n={hc['n_pairs']}, descriptive): solved "
            f"{la}: {hc[f'solved_{la}']}, {lb}: {hc[f'solved_{lb}']}"
        )
    if "calibration" in extra:
        for lbl in (la, lb):
            s = extra["calibration"][lbl]
            lines.append(
                f"- calibration {lbl}: FA={s['fa_rate']}, FR={s['fr_rate']}, "
                f"kappa={s['kappa_committed']}, "
                f"noncommittal={s['noncommittal_rate']}, "
                f"parse_fail={s['parse_failure_rate']}, "
                f"prevalence(correct)={s['prevalence_gold_correct']}"
            )
    lines += ["", f"> {report['not_authorized']}", ""]
    md.write_text("\n".join(lines))
    print(f"[report] written {args.output} and {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
