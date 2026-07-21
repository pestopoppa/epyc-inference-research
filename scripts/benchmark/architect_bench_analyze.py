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
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
