#!/usr/bin/env python3
"""Post-process ONE arm of the LB3.2 run into <prefix>.profile.json (pure: reads files, starts nothing).

    analyze.py --arm G --prefix OUT/G-t48      reads .lines.tsv .control.tsv .jsonl .perfscript.txt
    analyze.py --arm M --prefix OUT/M-t48      reads .lines.tsv .jsonl .evals.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from akx_lb3_attribution import evalprof, perfagg, phases  # noqa: E402
else:
    from . import evalprof, perfagg, phases


def read_lines_tsv(path: str):
    out = []
    with open(path, errors="replace") as f:
        for raw in f:
            t, _, text = raw.rstrip("\n").partition("\t")
            try:
                out.append((float(t), text))
            except ValueError:
                continue
    return out


def read_control_tsv(path: str):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for raw in p.read_text().splitlines():
        f = raw.split("\t")
        if len(f) >= 3:
            out.append((float(f[0]), f[1], float(f[2]) if f[2] not in ("", "None") else None))
    return out


def read_jsonl(path: str):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def analyze_g(prefix: str, guard_s: float = 0.002) -> dict:
    lines = read_lines_tsv(prefix + ".lines.tsv")
    rows = read_jsonl(prefix + ".jsonl")
    samples = {i: r.get("samples_ns") for i, r in enumerate(rows, start=1)}
    segs = phases.segments(lines)
    reps = phases.timed_reps(segs, samples, guard_s=guard_s, control_log=read_control_tsv(prefix + ".control.tsv"))
    agg = perfagg.aggregate(perfagg.file_source(prefix + ".perfscript.txt"), reps)
    out_rows = []
    for i, r in enumerate(rows, start=1):
        a = agg.get(i, {})
        out_rows.append(dict(idx=i, n_prompt=r.get("n_prompt"), n_depth=r.get("n_depth"), n_threads=r.get("n_threads"),
                             avg_ns=r.get("avg_ns"), stddev_ns=r.get("stddev_ns"), samples=a.get("samples", 0),
                             share=a.get("share", {}), active_tids=a.get("active_tids", {}),
                             reps_used=a.get("reps_used", 0), reps_excluded=a.get("reps_excluded", []),
                             restore_trim_ms=a.get("restore_trim_ms", [])))
    return dict(arm="G", guard_s=guard_s, rows=out_rows)


def analyze_m(prefix: str) -> dict:
    lines = read_lines_tsv(prefix + ".lines.tsv")
    rows = read_jsonl(prefix + ".jsonl")
    samples = {i: r.get("samples_ns") for i, r in enumerate(rows, start=1)}
    segs = phases.segments(lines)
    with open(prefix + ".evals.tsv", errors="replace") as f:
        ef = evalprof.parse_eval_file(f)
    reps = evalprof.attribute(segs, ef, samples)
    roll = evalprof.rollup(reps)
    by_cell = {(r.get("n_prompt"), r.get("n_depth")): i for i, r in enumerate(rows, start=1)}
    tops = {}
    for (n, d), i in by_cell.items():
        lo = by_cell.get((n, 512))
        if d == 65536 and lo is not None:
            tops[str(n)] = evalprof.top_nodes(ef, [r for r in reps if r.test == i], [r for r in reps if r.test == lo])
    out_rows = []
    for i, r in enumerate(rows, start=1):
        out_rows.append(dict(idx=i, n_prompt=r.get("n_prompt"), n_depth=r.get("n_depth"), n_threads=r.get("n_threads"),
                             avg_ns=r.get("avg_ns"), stddev_ns=r.get("stddev_ns"), **roll.get(i, {})))
    return dict(arm="M", eval_file_truncated=ef.truncated, n_evals=len(ef.evals), n_shapes=len(ef.shapes),
                rows=out_rows, top_depth_growing_nodes=tops)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("G", "M"), required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--guard-ms", type=float, default=2.0)
    a = ap.parse_args(argv)
    prof = analyze_g(a.prefix, a.guard_ms / 1e3) if a.arm == "G" else analyze_m(a.prefix)
    Path(a.prefix + ".profile.json").write_text(json.dumps(prof, indent=1))
    print(json.dumps(dict(arm=a.arm, rows=len(prof["rows"]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
