#!/usr/bin/env python3
"""Paired thinking-ON vs thinking-OFF analysis.

Reports accuracy, the degenerate-loop signature (empty content while
reasoning_content is populated), truncation, and median completion tokens --
because enabling thinking is a quality-per-latency decision, not quality alone.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def load(d: Path) -> dict:
    f = d / "per_question.jsonl"
    if not f.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ablation_dir", type=Path)
    args = ap.parse_args()

    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for d in sorted(args.ablation_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.endswith("_thinkon"):
            groups[d.name[:-8]]["on"] = d
        elif d.name.endswith("_thinkoff"):
            groups[d.name[:-9]]["off"] = d

    for label, arms in sorted(groups.items()):
        print(f"\n{'='*78}\n{label}\n{'='*78}")
        data = {m: load(p) for m, p in arms.items()}
        for mode in ("off", "on"):
            rows = list(data.get(mode, {}).values())
            if not rows:
                print(f"  think={mode}: (no data yet)")
                continue
            n = len(rows)
            toks = sorted(r["completion_tokens"] for r in rows)
            print(f"  think={mode:3s} n={n:3d} acc={sum(r['correct'] for r in rows)/n:6.1%} "
                  f"med_tok={toks[len(toks)//2]:6d} p90_tok={toks[int(0.9*(n-1))]:6d} "
                  f"trunc={sum(r['truncated'] for r in rows):3d} "
                  f"emptyC={sum(r.get('empty_content_with_reasoning', 0) for r in rows):3d} "
                  f"noparse={sum(1 for r in rows if not r['extracted']):3d}")
        if len(data) == 2 and all(data.values()):
            common = sorted(set(data["on"]) & set(data["off"]))
            if common:
                b = sum(1 for q in common if data["on"][q]["correct"] and not data["off"][q]["correct"])
                c = sum(1 for q in common if data["off"][q]["correct"] and not data["on"][q]["correct"])
                accon = sum(data["on"][q]["correct"] for q in common) / len(common)
                accoff = sum(data["off"][q]["correct"] for q in common) / len(common)
                p = mcnemar_exact(b, c)
                ton = sum(data["on"][q]["completion_tokens"] for q in common) / len(common)
                toff = sum(data["off"][q]["completion_tokens"] for q in common) / len(common)
                print(f"\n  PAIRED n={len(common)}: on-off = {accon-accoff:+.1%} "
                      f"(on {accon:.1%} vs off {accoff:.1%}); McNemar b={b} c={c} p={p:.4f}"
                      f"{'  *' if p < 0.05 else '  (n.s.)'}")
                print(f"  token cost: mean {toff:.0f} -> {ton:.0f} "
                      f"({ton/toff:.2f}x output tokens with thinking on)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
