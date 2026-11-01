#!/usr/bin/env python3
"""E-6: does budget-capped <think> beat both unlimited-think and no-think?

For each (model, budget) arm, report accuracy, non-termination rate, and mean
tokens, next to the two baselines from the R2d ablation (think off / think on
unlimited) on the SAME pinned items -- so the question "did force-closing the
think block recover the non-termination tail?" is answered paired.
"""
from __future__ import annotations
import argparse
import json
from math import comb
from pathlib import Path


def load(p: Path) -> dict:
    f = p / "per_question.jsonl"
    if not f.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def acc(d, qs):
    return sum(d[q]["correct"] for q in qs) / len(qs) if qs else 0.0


def nonterm(d, qs):
    return sum(1 for q in qs if d[q].get("empty_content_with_reasoning")) / len(qs) if qs else 0.0


def mtok(d, qs):
    return sum(d[q]["completion_tokens"] for q in qs) / len(qs) if qs else 0.0


def mcnemar(on, off, qs):
    b = sum(1 for q in qs if on[q]["correct"] and not off[q]["correct"])
    c = sum(1 for q in qs if off[q]["correct"] and not on[q]["correct"])
    n = b + c
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return b, c, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("art", type=Path, help="architect-bench-gpu-* artifact dir")
    args = ap.parse_args()
    e6 = args.art / "e6_reasoning_budget"
    abl = args.art / "ablation_thinking"

    # model label -> {baseline dirs}
    models = {
        "A4_frontdoor_35b_a3b": ("A4_frontdoor_35b_a3b_thinkoff", "A4_frontdoor_35b_a3b_thinkon"),
        "A1_architect_122b_iq2": ("A1_architect_122b_iq2_thinkoff", "A1_architect_122b_iq2_thinkon"),
    }
    for model, (offd, ond) in models.items():
        off = load(abl / offd)
        on = load(abl / ond)
        budgets = sorted(e6.glob(f"{model}_budget*")) if e6.exists() else []
        print(f"\n{'='*78}\n{model}\n{'='*78}")
        print(f"{'arm':22s} {'acc':>7s} {'nonterm':>8s} {'meantok':>8s}  vs-off(paired)")
        if off:
            qs = sorted(off)
            print(f"{'think off (baseline)':22s} {acc(off,qs):>6.1%} {nonterm(off,qs):>7.0%} "
                  f"{mtok(off,qs):>8.0f}  —")
        if on:
            qs = sorted(set(on) & set(off)) if off else sorted(on)
            b, c, p = mcnemar(on, off, qs) if off else (0, 0, 1.0)
            print(f"{'think on (unlimited)':22s} {acc(on,qs):>6.1%} {nonterm(on,qs):>7.0%} "
                  f"{mtok(on,qs):>8.0f}  {acc(on,qs)-acc(off,qs):+.1%} b={b} c={c} p={p:.3f}")
        for bd in budgets:
            d = load(bd)
            if not d:
                print(f"{bd.name.split('_budget')[-1]+' (running)':22s} (no data yet)")
                continue
            qs = sorted(set(d) & set(off)) if off else sorted(d)
            label = f"budget {bd.name.split('_budget')[-1]}"
            if off:
                b, c, p = mcnemar(d, off, qs)
                print(f"{label:22s} {acc(d,qs):>6.1%} {nonterm(d,qs):>7.0%} {mtok(d,qs):>8.0f}  "
                      f"{acc(d,qs)-acc(off,qs):+.1%} b={b} c={c} p={p:.3f}"
                      f"{'  *' if p<0.05 else ''}")
            else:
                print(f"{label:22s} {acc(d,qs):>6.1%} {nonterm(d,qs):>7.0%} {mtok(d,qs):>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
