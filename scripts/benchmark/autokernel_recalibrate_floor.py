#!/usr/bin/env python3
"""Recalibrate the effect floor from an A/A campaign, and re-adjudicate what it changes.

    python3 scripts/benchmark/autokernel_recalibrate_floor.py \
        --campaign /mnt/raid0/llm/autokernel/loop-memory/aa-campaign/aa-campaign.json \
        --store /mnt/raid0/llm/autokernel/loop-memory [--apply]

WHY THE OLD FLOOR WAS WRONG
---------------------------
`bench.MEASURED_FLOOR_PCT` was derived as p95 of |median effect| over EXHAUSTIVE
SUBSETS OF ONE FIXED 20-pair sample. Subsets of a fixed sample cannot exceed that
sample's own observed tail, so the construction is bounded below the true p95 by
construction -- it measures how much that particular sample varies internally, not how
much a fresh measurement varies.

The A/A campaign fixes the flaw at the source: it BOOTSTRAPS over fresh pairs whose
true effect is exactly zero, so every non-zero number it produces is estimator error.
That is what a floor is supposed to be.

WHAT THIS DOES
--------------
Reports the bootstrap floor beside the enforced one, flags every pair count where the
enforced bar sits BELOW the instrument's real resolution, and re-adjudicates every
recorded comparison so the cost of the change is visible before it is made. A keep that
does not survive recalibration was never a keep; better to find that here than to ship
it.

`--apply` rewrites the table in `bench.py`. Without it, nothing is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))

from autokernel.loop import bench                                   # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--surface", default="tg128")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    report = json.loads(args.campaign.read_text(encoding="utf-8"))
    boot = {int(k): v for k, v in (report.get("bootstrap_floor_pct") or {}).items()}
    if not boot:
        print("REFUSED: campaign carries no bootstrap_floor_pct", file=sys.stderr)
        return 1

    print(f"surface {args.surface}   (this campaign measured ONLY this surface; the "
          f"other surface's floor is untouched and remains uncalibrated)\n")
    print(f"{'pairs':>6} {'enforced':>10} {'bootstrap':>10}   verdict")
    enforced_table = dict(bench.MEASURED_FLOOR_PCT[args.surface])
    proposed = dict(enforced_table)
    for k in sorted(boot):
        old = enforced_table.get(k)
        new = boot[k]
        if old is None:
            verdict = "new row"
        elif new > old:
            verdict = f"RAISE — enforced bar was BELOW resolution by {new - old:.3f} pp"
        else:
            verdict = "enforced bar already conservative; keeping the higher value"
        proposed[k] = max(new, old) if old is not None else new
        print(f"{k:>6} {('%.3f' % old) if old is not None else '   —':>10} "
              f"{new:>9.3f}   {verdict}")

    # The A/A effect estimates themselves are a direct sample of estimator error on a
    # known-zero truth. Report them: if any exceeds the proposed floor, the floor is
    # still too low no matter what the bootstrap says.
    print("\nA/A effect estimates (true effect is 0 by construction):")
    worst = 0.0
    for cond in report.get("conditions", []):
        e = abs(cond.get("effect_pct", 0.0))
        worst = max(worst, e)
        print(f"  {cond.get('condition','?'):>12}: {cond.get('effect_pct', 0.0):+.3f}%")
    ref = proposed.get(report.get("pairs", 20))
    if ref is not None and worst > ref:
        print(f"  WARNING: the largest A/A error ({worst:.3f}%) EXCEEDS the proposed "
              f"{report.get('pairs')}-pair floor ({ref:.3f}%). A whole-run effect on "
              f"identical code is the most direct floor estimate there is; the "
              f"bootstrap is resampling WITHIN a run and cannot see run-level shifts.")

    # ---- re-adjudicate every recorded comparison ---------------------------
    print("\nre-adjudicating recorded measurements under the proposed floor:")
    db = args.store / "experiments.db"
    rows = [json.loads(p) for (p,) in sqlite3.connect(db).execute(
        "SELECT payload FROM experiments ORDER BY recorded_at")]
    changed = 0
    for d in rows:
        cm = d.get("comparison") or {}
        if not cm.get("anchor_samples"):
            continue
        pairs = cm.get("pairs") or 9
        old_floor = cm.get("noise_floor_pct") or 0.0
        new_floor = proposed.get(pairs, old_floor)
        effect = cm.get("effect_pct", 0.0)
        was = abs(effect) > old_floor
        now = abs(effect) > new_floor
        if was != now:
            changed += 1
            print(f"  {(d.get('mechanism_id') or '?')[:38]:38} {effect:+7.3f}%  "
                  f"floor {old_floor:.3f} -> {new_floor:.3f}  "
                  f"{'CLEARS -> no longer clears' if was else 'now clears'}")
    print(f"  {changed} verdict change(s)")

    if not args.apply:
        print("\nDRY RUN — pass --apply to rewrite bench.MEASURED_FLOOR_PCT.")
        return 0

    source = Path(bench.__file__)
    text = source.read_text(encoding="utf-8")
    row = ", ".join(f"{k}: {proposed[k]}" for k in sorted(proposed))
    pattern = re.compile(rf'("{args.surface}": )\{{[^}}]*\}}')
    if not pattern.search(text):
        print("REFUSED: could not locate the table row to rewrite", file=sys.stderr)
        return 1
    source.write_text(pattern.sub(rf'\g<1>{{{row}}}', text, count=1), encoding="utf-8")
    print(f"\nrewrote {args.surface} row in {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
