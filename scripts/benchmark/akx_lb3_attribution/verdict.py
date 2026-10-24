#!/usr/bin/env python3
"""LB3.2 verdict over the arms' profile.json files (rule.py; PREDECLARATION.md). Pure arithmetic.

usage: verdict.py --g OUT/G-t48 --m OUT/M-t48 [--g-report OUT/G-t96 --m-report OUT/M-t96] --steps OUT/steps.txt
                  --out VERDICT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from akx_lb3_attribution import rule  # noqa: E402
else:
    from . import rule


def load(prefix: str):
    p = Path(prefix + ".profile.json")
    return json.loads(p.read_text()) if p.exists() else None


def steps_ok(steps_path: str, labels) -> bool:
    p = Path(steps_path)
    if not p.exists():
        return False
    ok = {l.split()[0] for l in p.read_text().splitlines() if len(l.split()) > 1 and l.split()[1] == "OK"}
    return all(l in ok for l in labels)


def one(g_prefix: str, m_prefix: str, steps_path: str) -> dict:
    g, m = load(g_prefix), load(m_prefix)
    if g is None or m is None:
        return dict(verdict="INVALID", why=[f"profile missing: {'G' if g is None else 'M'}"])
    labels = [Path(g_prefix).name, Path(m_prefix).name]
    return rule.decide(g["rows"], m["rows"], {r["idx"]: r for r in m["rows"]},
                       steps_ok=steps_ok(steps_path, labels), eval_truncated=bool(m.get("eval_file_truncated")),
                       g_perf={r["idx"]: r for r in g["rows"]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", required=True)
    ap.add_argument("--m", required=True)
    ap.add_argument("--g-report")
    ap.add_argument("--m-report")
    ap.add_argument("--steps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = dict(item="akx-lb3-2-attribution", class_="correctness (within-process ratios; G/M cross-arm overhead gate)",
               gating=one(a.g, a.m, a.steps))
    res["verdict"] = res["gating"]["verdict"]
    if a.g_report and a.m_report and load(a.g_report) and load(a.m_report):
        res["reported_t96"] = one(a.g_report, a.m_report, a.steps)
    Path(a.out).write_text(json.dumps(res, indent=1))
    g = res["gating"]
    print(json.dumps(dict(verdict=res["verdict"], max_F_N2to5=g.get("max_F_N2to5"), max_S_N2to5=g.get("max_S_N2to5"),
                          seed_class=g.get("seed_class"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
