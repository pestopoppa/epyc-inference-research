#!/usr/bin/env python3
"""Paired analysis for the architect-model-selection bench.

Reads the per-question JSONL each arm writes and reports, per suite:
  * per-arm accuracy (avg@k), restricted to the seeds EVERY arm completed so
    arms stopped at different k stay comparable;
  * pairwise deltas with an exact McNemar test when k==1, and a paired
    bootstrap CI over per-question mean scores when k>1 (McNemar needs binary
    paired outcomes, which avg@k scores are not);
  * failure classification (truncated / empty / unextracted) so a budget or
    parsing failure is never silently read as a reasoning failure.

Usage: architect_bench_analyze.py <runs_dir> [--suite S]
       [--slow-crr-baseline ARM]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

SLOW_CRR_MIN_SOLVED = 30


def slow_crr(candidate_rows: list[dict], baseline_rows: list[dict]) -> dict:
    """Describe how often a correct candidate answer is no faster than baseline.

    The unit is one matched (suite, question id, seed) output. Only candidate
    outputs explicitly marked ``correct is True`` enter the denominator. For
    every such output the baseline row and both positive finite ``decode_tok_s``
    values must be present; otherwise the conditional is reported unavailable
    rather than silently shrinking its denominator.

    This is a descriptive conditional diagnostic, not a score or gate. The
    handoff's existing 30-correct-output floor controls whether the fraction is
    reportable.
    """
    result = {
        "status": "below_minimum", "solved": 0, "no_faster": None,
        "fraction": None, "reason": None, "unmatched": 0,
        "invalid_candidate_speed": 0, "invalid_baseline_speed": 0,
        "duplicate_candidate_identity": 0, "duplicate_baseline_identity": 0,
    }
    baseline_by_key: dict[tuple[str, str, object], list[dict]] = defaultdict(list)
    for row in baseline_rows:
        try:
            key = (str(row["suite"]), str(row["id"]), row["seed"])
            hash(key)
        except (KeyError, TypeError):
            return {**result, "status": "incomplete", "reason": "baseline identity missing",
                    "unmatched": 1}
        baseline_by_key[key].append(row)

    seen_candidate: set[tuple[str, str, object]] = set()
    candidate_by_key: dict[tuple[str, str, object], dict] = {}
    invalid_correctness = 0
    no_faster = 0
    for row in candidate_rows:
        try:
            key = (str(row["suite"]), str(row["id"]), row["seed"])
            hash(key)
        except (KeyError, TypeError):
            return {**result, "status": "incomplete", "reason": "candidate identity missing",
                    "unmatched": 1}
        if key in seen_candidate:
            result["duplicate_candidate_identity"] += 1
            continue
        seen_candidate.add(key)
        if type(row.get("correct")) is not bool:
            invalid_correctness += 1
            continue
        if row["correct"] is not True:
            continue
        candidate_by_key[key] = row

    solved = len(candidate_by_key)
    result["solved"] = solved
    if invalid_correctness or result["duplicate_candidate_identity"]:
        result.update(status="incomplete", reason=(
            f"candidate rows invalid: correctness={invalid_correctness}, "
            f"duplicate_identity={result['duplicate_candidate_identity']}"))
        return result

    for key, row in candidate_by_key.items():
        matches = baseline_by_key.get(key, [])
        if len(matches) != 1:
            if len(matches) > 1:
                result["duplicate_baseline_identity"] += 1
            else:
                result["unmatched"] += 1
            continue
        candidate_speed = row.get("decode_tok_s")
        baseline_speed = matches[0].get("decode_tok_s")
        try:
            candidate_speed_valid = (
                not isinstance(candidate_speed, bool)
                and isinstance(candidate_speed, (int, float))
                and math.isfinite(candidate_speed) and candidate_speed > 0)
        except (OverflowError, TypeError, ValueError):
            candidate_speed_valid = False
        if not candidate_speed_valid:
            result["invalid_candidate_speed"] += 1
            continue
        try:
            baseline_speed_valid = (
                not isinstance(baseline_speed, bool)
                and isinstance(baseline_speed, (int, float))
                and math.isfinite(baseline_speed) and baseline_speed > 0)
        except (OverflowError, TypeError, ValueError):
            baseline_speed_valid = False
        if not baseline_speed_valid:
            result["invalid_baseline_speed"] += 1
            continue
        if candidate_speed <= baseline_speed:
            no_faster += 1

    if any(result[name] for name in (
            "unmatched", "invalid_candidate_speed", "invalid_baseline_speed",
            "duplicate_baseline_identity")):
        result.update(status="incomplete", reason=(
            f"matched-speed coverage incomplete: unmatched={result['unmatched']}, "
            f"duplicate_baseline={result['duplicate_baseline_identity']}, "
            f"invalid_candidate_speed={result['invalid_candidate_speed']}, "
            f"invalid_baseline_speed={result['invalid_baseline_speed']}"))
        return result
    if solved < SLOW_CRR_MIN_SOLVED:
        return result
    result.update(status="reported", no_faster=no_faster, fraction=no_faster / solved)
    return result


def load(runs_dir: Path) -> dict:
    """suite -> arm -> list[record]"""
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for pq in sorted(runs_dir.glob("*/*/per_question.jsonl")):
        # Always prefer a re-scored file when present: arms may have run under
        # different scorer versions, and mixing them would compare arms on
        # different rules. Re-scoring normalises every arm to one scorer.
        rescored = pq.with_name("per_question.rescored.jsonl")
        if rescored.exists():
            pq = rescored
        arm = pq.parent.name
        for line in pq.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[r.get("suite") or pq.parent.parent.name][arm].append(r)
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant pairs."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(pairs: list[tuple[float, float]], iters: int = 10000,
                 seed: int = 12345) -> tuple[float, float, float]:
    """Paired bootstrap over questions -> (mean delta, lo95, hi95)."""
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    obs = sum(x - y for x, y in pairs) / n
    deltas = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            x, y = pairs[rng.randrange(n)]
            s += x - y
        deltas.append(s / n)
    deltas.sort()
    return obs, deltas[int(0.025 * iters)], deltas[int(0.975 * iters)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--suite", default=None)
    ap.add_argument(
        "--slow-crr-baseline", default=None, metavar="ARM",
        help=("also report the descriptive Slow/CRR conditional against ARM: "
              "among correct candidate outputs, share with decode_tok_s <= ARM; "
              "reported only at 30 or more correct outputs"),
    )
    args = ap.parse_args()

    data = load(args.runs_dir)
    for suite, arms in sorted(data.items()):
        if args.suite and suite != args.suite:
            continue
        print(f"\n{'='*74}\nSUITE: {suite}\n{'='*74}")

        # Comparable slice = seeds every arm actually completed.
        seed_sets = {a: {r["seed"] for r in rs} for a, rs in arms.items()}
        common = set.intersection(*seed_sets.values()) if seed_sets else set()
        # ...and questions every arm attempted at those seeds.
        qid_sets = {a: {r["id"] for r in rs if r["seed"] in common}
                    for a, rs in arms.items()}
        common_q = set.intersection(*qid_sets.values()) if qid_sets else set()
        k = len(common)
        print(f"matched slice: k={k} seed(s) {sorted(common)}, "
              f"{len(common_q)} questions")
        for a in sorted(arms):
            extra = sorted(seed_sets[a] - common)
            if extra:
                print(f"  note: {a} also has seeds {extra} (excluded to keep arms matched)")

        # per-arm score + failure classification
        scores: dict[str, dict[str, float]] = {}
        print(f"\n{'arm':32s} {'acc(avg@k)':>11s} {'n':>6s} {'trunc':>6s} "
              f"{'empty':>6s} {'noparse':>8s} {'medtok':>7s}")
        for a in sorted(arms):
            rs = [r for r in arms[a] if r["seed"] in common and r["id"] in common_q]
            per_q: dict[str, list] = defaultdict(list)
            for r in rs:
                per_q[r["id"]].append(r)
            scores[a] = {q: sum(x["correct"] for x in v) / len(v)
                         for q, v in per_q.items()}
            acc = sum(scores[a].values()) / len(scores[a]) if scores[a] else 0.0
            trunc = sum(1 for r in rs if r.get("truncated"))
            empty = sum(1 for r in rs if r.get("empty_response"))
            noparse = sum(1 for r in rs if not r.get("extracted"))
            toks = sorted(r.get("completion_tokens", 0) for r in rs)
            med = toks[len(toks)//2] if toks else 0
            print(f"{a:32s} {acc:>10.1%} {len(rs):>6d} {trunc:>6d} "
                  f"{empty:>6d} {noparse:>8d} {med:>7d}")

        # pairwise
        names = sorted(scores)
        if len(names) > 1:
            print("\npairwise (row - col):")
            for i, a in enumerate(names):
                for b in names[i+1:]:
                    qs = sorted(set(scores[a]) & set(scores[b]))
                    pairs = [(scores[a][q], scores[b][q]) for q in qs]
                    delta = sum(x - y for x, y in pairs) / len(pairs) if pairs else 0
                    if k == 1:
                        bb = sum(1 for x, y in pairs if x > y)
                        cc = sum(1 for x, y in pairs if x < y)
                        p = mcnemar_exact(bb, cc)
                        print(f"  {a} - {b}: {delta:+.1%}  "
                              f"McNemar b={bb} c={cc} p={p:.4f}"
                              f"{'  *' if p < 0.05 else ''}")
                    else:
                        d, lo, hi = bootstrap_ci(pairs)
                        sig = "" if lo <= 0 <= hi else "  *"
                        print(f"  {a} - {b}: {d:+.1%}  "
                              f"paired-bootstrap 95% CI [{lo:+.1%}, {hi:+.1%}]{sig}")

        if args.slow_crr_baseline:
            baseline = args.slow_crr_baseline
            print(f"\nSlow/CRR conditional (correct candidate outputs with decode_tok_s "
                  f"<= {baseline}; descriptive only; minimum {SLOW_CRR_MIN_SOLVED}):")
            if baseline not in arms:
                print(f"  unavailable: baseline arm {baseline!r} is absent from suite")
                continue
            # This diagnostic has its own candidate/baseline matched set. Do not
            # reuse the all-arm accuracy cohort above: a third arm or a missing
            # baseline row must not silently remove correct candidate outputs.
            for candidate in sorted(arms):
                if candidate == baseline:
                    continue
                metric = slow_crr(arms[candidate], arms[baseline])
                if metric["status"] == "reported":
                    print(f"  {candidate}: {metric['no_faster']}/{metric['solved']} = "
                          f"{metric['fraction']:.1%}")
                elif metric["status"] == "below_minimum":
                    print(f"  {candidate}: not reported ({metric['solved']} correct; "
                          f"requires {SLOW_CRR_MIN_SOLVED})")
                else:
                    print(f"  {candidate}: unavailable ({metric['reason']}; "
                          f"{metric['solved']} correct outputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
