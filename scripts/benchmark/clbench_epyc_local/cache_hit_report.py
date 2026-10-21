#!/usr/bin/env python3
"""CME-7 post-run check: did the local server reuse the append-only prefix? (stdlib only; reads traces, no model)

  python3 -I cache_hit_report.py <traces_dir> [--min-hit 0.95] [--expect-runs 3 --expect-hands 5] [--out report.json]

CME-5 sizing rule (predeclared 2026-10-21, queue-builders-3): the acceptance number is NOT cached/input over all
calls, because the first call of every run is cold by construction and early prompts are mostly new suffix, so a
short ``quick_test`` would fail 95% by arithmetic alone. The gated number is over the PREFIX-REUSABLE tokens:

    for call k >= 2 of a run:  reusable_k = min(input_k, input_{k-1} + output_{k-1})
    hit_ratio = sum(cached_k) / sum(reusable_k)            gate: >= --min-hit (default 0.95)

The raw ``sum(cached)/sum(input)`` is reported beside it, never instead. A call whose ``cached_input_tokens`` is
missing makes the report INVALID (a null counter is "not captured", never zero). Exit 0 = complete and gate met,
1 = gate missed, 2 = invalid/incomplete (wrong run or hand count, missing counters).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def run_calls(trace: dict) -> list[dict]:
    calls = []
    for it in trace.get("interactions") or []:
        for ev in ((it.get("usage") or {}).get("respond") or {}).get("events") or []:
            calls.append(ev)
    return calls


def analyse(traces: list[dict], min_hit: float = 0.95, expect_runs: int | None = None, expect_hands: int | None = None) -> dict:
    problems, per_run = [], []
    num = den = raw_c = raw_i = 0
    for n, tr in enumerate(traces):
        calls = run_calls(tr)
        hands = len(tr.get("instance_outcomes") or [])
        if expect_hands is not None and hands != expect_hands:
            problems.append(f"run {n}: {hands} completed hands, expected {expect_hands}")
        if not calls:
            problems.append(f"run {n}: no usage events")
            continue
        rc = rr = 0
        for k, ev in enumerate(calls):
            c, i = ev.get("cached_input_tokens"), ev.get("input_tokens")
            if c is None or i is None:
                problems.append(f"run {n} call {k}: cached/input counter not captured")
                continue
            raw_c += c; raw_i += i
            if k >= 1:
                prev = calls[k - 1]
                reusable = min(i, (prev.get("input_tokens") or 0) + (prev.get("output_tokens") or 0))
                num += c; den += reusable; rc += c; rr += reusable
        per_run.append({"run": n, "calls": len(calls), "hands": hands, "hit_ratio": round(rc / rr, 4) if rr else None})
    if expect_runs is not None and len(traces) != expect_runs:
        problems.append(f"{len(traces)} runs found, expected {expect_runs}")
    ratio = num / den if den else None
    rep = {"schema": "epyc.clbench_cache_hit.v1", "min_hit": min_hit, "hit_ratio_reusable": None if ratio is None else round(ratio, 4),
           "raw_cached_over_input": round(raw_c / raw_i, 4) if raw_i else None, "reusable_tokens": den, "cached_tokens": num,
           "per_run": per_run, "problems": problems}
    rep["verdict"] = "INVALID" if problems or ratio is None else ("PASS" if ratio >= min_hit else "MISS")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces_dir")
    ap.add_argument("--min-hit", type=float, default=0.95)
    ap.add_argument("--expect-runs", type=int)
    ap.add_argument("--expect-hands", type=int)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(str(Path(a.traces_dir) / "run_*.json")))
    rep = analyse([json.load(open(f)) for f in files], a.min_hit, a.expect_runs, a.expect_hands)
    rep["trace_files"] = files
    text = json.dumps(rep, indent=1)
    (Path(a.out).write_text(text + "\n") if a.out else print(text))
    return {"PASS": 0, "MISS": 1, "INVALID": 2}[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
