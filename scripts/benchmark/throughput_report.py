#!/usr/bin/env python3
"""Aggregate + per-request throughput for a batched eval run.

Works from what any run already persists (per_question.jsonl completion_tokens +
result.json elapsed_s), so it covers runs made before per-request timings were
captured. If a result carries a `throughput` block (newer runs) it uses that;
otherwise it reconstructs aggregate = Σcompletion_tokens / wall.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path


def report(run_dir: Path, concurrency_hint: dict[str, int]):
    rows = []
    for rj in sorted(run_dir.glob("*/result.json")):
        arm = rj.parent.name
        meta = json.loads(rj.read_text())
        suite = meta["suites"][0]
        wall = None
        tp = suite.get("throughput")
        pq = rj.parent / "per_question.jsonl"
        recs = [json.loads(l) for l in pq.read_text().splitlines() if l.strip()] if pq.exists() else []
        comp = sum(r.get("completion_tokens", 0) for r in recs)
        if tp:  # newer runs: authoritative
            wall = tp.get("wall_s")
            conc = tp.get("concurrency")
            agg = tp.get("aggregate_decode_tok_s")
        else:  # reconstruct from elapsed_s
            wall = meta["meta"].get("elapsed_s")
            conc = concurrency_hint.get(arm, "?")
            agg = round(comp / wall, 1) if wall else 0
        per_req = [r["decode_tok_s"] for r in recs if r.get("decode_tok_s")]
        med_req = round(statistics.median(per_req), 1) if per_req else None
        rows.append((arm, conc, suite["n"], comp, wall, agg, med_req))
    print(f"{'arm':30s} {'conc':>4s} {'n':>4s} {'gen_tok':>9s} {'wall_s':>8s} "
          f"{'AGG_tok/s':>9s} {'med_req_t/s':>11s}")
    for arm, conc, n, comp, wall, agg, med in rows:
        print(f"{arm:30s} {str(conc):>4s} {n:>4d} {comp:>9d} {str(wall):>8s} "
              f"{agg:>9} {str(med) if med else '-':>11s}")
    print("\nAGG_tok/s = aggregate decode throughput (Σ generated / wall, all slots).")
    print("med_req_t/s = median per-request decode speed (single-slot view; blank if not captured).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="runs/<suite>/ dir")
    ap.add_argument("--conc", nargs="*", default=[],
                    help="arm=concurrency hints for older runs, e.g. A1_...=14")
    args = ap.parse_args()
    hint = dict(x.split("=") for x in args.conc) if args.conc else {}
    hint = {k: int(v) for k, v in hint.items()}
    report(args.run_dir, hint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
