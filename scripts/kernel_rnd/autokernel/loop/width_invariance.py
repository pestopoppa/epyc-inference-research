#!/usr/bin/env python3
"""AKX-ALL-26 width-invariance op probe: driver, first-fail-k table and prediction scoring.

The probe itself is ``width_invariance_probe.cpp`` (one process per route/threads/op/type group, one JSONL
record per cell).  This driver

* ``build``  compiles the probe against a ggml build dir,
* ``run``    executes the group plan (resumable; a group that exits non-zero leaves its cells MISSING, which the
             table reports as ERR - never omitted),
* ``table``  folds the JSONL into the first-fail-k table per op/type/route and scores the frozen predicted-fail
             list (confirmed / falsified / void), and checks the harness controls.

Run the probe only inside a held region-lock claim (CPU work rule); this script does not take the claim.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

KQ = ["Q8_0", "Q4_K", "Q3_K", "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_XS"]
FA_TYPES = ["F16", "Q8_0", "Q4_0"]
KMAX = 9
NKV = [2048, 7168, 16384, 65536]

MM_SHAPES = {"D27-5120": 5120, "D27-17408": 17408, "D27-6144": 6144, "M2048": 2048, "M512": 512, "F2560": 2560, "F640": 640}
MM_SMALL = ["M2048", "M512"]
MMID_SHAPES = {"M2048": 2048, "M512": 512, "F2560": 2560, "F640": 640}
MMID_MODES = ["rand", "shared", "disjoint"]
BLCK = {"Q8_0": 32, "Q4_0": 32, "F16": 1, "F32": 1}  # everything else is 256
GDN_SHAPES = ["Q38-27B-Hv48-scalar", "A3B-Hv32-scalar", "Q38-27B-Hv48-kda"]
CTL_SHAPES = ["ctl-positive-width-dependent", "ctl-negative-width-independent", "ctl-bitflip-injection"]

ROUTES = {
    "cpu": {"GGML_IQK": "1"},
    "cpu-native": {},
    "cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]": {"GGML_IQK": "1", "GGML_IQK_Q8_0": "1"},
    "cpu[GGML_IQK=1,GGML_ROWEXACT_N=9]": {"GGML_IQK": "1", "GGML_ROWEXACT_N": "9"},
}
SCRUB = ["GGML_IQK", "GGML_IQK_Q8_0", "GGML_IQK_DEQUANT", "GGML_ROWEXACT_N", "GGML_FA_SPLIT_KV", "GGML_QSPLIT", "GGML_QSPLIT_MIN"]


def plan():
    """(route, nth, op, type) groups."""
    g = []
    r = "cpu"
    g += [(r, 24, "ctl", "-"), (r, 24, "gdn", "F32")]
    g += [(r, 24, "fa", t) for t in FA_TYPES]
    g += [(r, 24, "mm", t) for t in KQ] + [(r, 24, "mmid", t) for t in KQ]
    r = "cpu-native"
    g += [(r, 24, "mm", t) for t in KQ] + [(r, 24, "mmid", t) for t in KQ]
    r = "cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]"
    g += [(r, 24, "mm", "Q8_0"), (r, 24, "mmid", "Q8_0")]
    r = "cpu[GGML_IQK=1,GGML_ROWEXACT_N=9]"
    g += [(r, 24, "mm", t) for t in KQ]
    r = "cpu"
    g += [(r, 1, "gdn", "F32")] + [(r, 1, "fa", t) for t in FA_TYPES]
    g += [(r, 1, "mm", t) for t in KQ] + [(r, 1, "mmid", t) for t in KQ]
    return g


def gname(g):
    route, nth, op, typ = g
    return f"{route.replace('[', '_').replace(']', '_').replace(',', '_').replace('=', '-')}__nth{nth}__{op}__{typ}"


def build(args):
    inc = Path(args.ggml_src) / "ggml" / "include"
    bin_dir = Path(args.build_dir) / "bin"
    src = Path(__file__).with_name("width_invariance_probe.cpp")
    cmd = ["c++", "-std=c++17", "-O2", "-I", str(inc), "-o", args.probe, str(src), "-L", str(bin_dir),
           "-lggml", "-lggml-cpu", "-lggml-base", f"-Wl,-rpath,{bin_dir}", "-pthread"]
    print(" ".join(cmd))
    return subprocess.call(cmd)


def run_group(args, g, out_dir: Path, log):
    route, nth, op, typ = g
    out = out_dir / (gname(g) + ".jsonl")
    done = out_dir / (gname(g) + ".done")
    if done.exists():
        return
    if out.exists():
        out.unlink()
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update(ROUTES[route])
    bin_dir = str(Path(args.build_dir) / "bin")
    env["LD_LIBRARY_PATH"] = bin_dir
    cmd = [args.probe, "--op", op, "--type", typ, "--route", route, "--nth", str(nth), "--reps", "3", "--kmax", str(KMAX),
           "--seed", str(args.seed), "--out", str(out)]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    rec = {"group": gname(g), "rc": p.returncode, "seconds": round(dt, 1), "stderr_tail": p.stderr[-600:]}
    log.write(json.dumps(rec) + "\n")
    log.flush()
    print(rec["group"], "rc", p.returncode, f"{dt:.1f}s", flush=True)
    done.write_text(json.dumps(rec))


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = plan()
    if args.only:
        groups = [g for g in groups if args.only in gname(g)]
    with open(out_dir / "run_groups.log", "a") as log:
        for g in groups:
            run_group(args, g, out_dir, log)
    return 0


# ------------------------------------------------------------------------------------------ expected grid
def expected_cells():
    """(route, nth, op, type, shape, mode) keys that a complete run must contain (k=1..9 each)."""
    exp = []
    for route, nth, op, typ in plan():
        if op == "mm":
            for s in MM_SHAPES:
                if nth == 1 and s not in MM_SMALL:
                    continue
                exp.append((route, nth, "MUL_MAT", typ, s, "-"))
        elif op == "mmid":
            for s in MMID_SHAPES:
                if nth == 1 and s not in MM_SMALL:
                    continue
                for m in MMID_MODES:
                    exp.append((route, nth, "MUL_MAT_ID", typ, s, m))
        elif op == "fa":
            for n in NKV:
                exp.append((route, nth, "FLASH_ATTN_EXT", typ, str(n), "-"))
        elif op == "gdn":
            for s in GDN_SHAPES:
                exp.append((route, nth, "GATED_DELTA_NET", "F32", s, "-"))
        elif op == "ctl":
            for s in CTL_SHAPES:
                exp.append((route, nth, "CONTROL", "F32", s, "-"))
    return exp


def load_cells(out_dir: Path):
    cells = defaultdict(dict)    # key -> k -> record
    headers = []
    for f in sorted(out_dir.glob("*.jsonl")):
        if f.name in ("cells.jsonl",):
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["schema"] == "akx-all-26.group.v1":
                headers.append(r)
                continue
            key = (r["route"], r["nth"], r["op"], r["type"].upper(), r["shape"], r["mode"])  # ggml_type_name() is lower-case
            cells[key][r["k"]] = r
    return cells, headers


def summarize(per_k):
    """per_k: {k: record} -> (pattern, first_fail_k or None, status)"""
    pat = ""
    first = None
    bad = False
    for k in range(1, KMAX + 1):
        r = per_k.get(k)
        if r is None:
            pat += "?"
            bad = True
            continue
        v = r["verdict"]
        pat += {"PASS": "P", "FAIL": "F", "NONDET": "N", "ERR": "E", "UNSUPPORTED": "U"}.get(v, "?")
        if v == "FAIL" and first is None:
            first = k
        if v in ("NONDET", "ERR"):
            bad = True
    if set(pat) == {"U"}:
        return pat, None, "UNSUPPORTED"
    if "N" in pat:
        return pat, first, "VOID_AA"
    if "E" in pat or "?" in pat:
        return pat, first, "ERR"
    return pat, first, "FAIL" if first else "PASS"


def cell_label(status, first):
    return {"PASS": "PASS", "FAIL": f"F@{first}", "UNSUPPORTED": "UNSUP", "VOID_AA": "VOID(A/A)", "ERR": "ERR"}[status]


# ------------------------------------------------------------------------------------------ predictions
def load_frozen(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def frozen_selectors(rows):
    """Map each frozen CPU row to selectors over measured groups."""
    sel = []
    route_of = {"cpu": "cpu", "cpu-native": "cpu-native", "cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]": "cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]"}
    for r in rows:
        b = r["backend"]
        if b not in route_of:
            continue
        i = r["id"]
        pred = r["predicted_verdict"]
        pk = r["predicted_first_fail_k"]
        opn = r["op"]
        typ = r["type"]
        if opn in ("MUL_MAT", "MUL_MAT_ID"):
            if i.endswith("-F640") and pred == "UNSUPPORTED":
                sel.append(dict(id=i, route=route_of[b], nth=24, op=opn, type=typ, shape="F640", mode=None, pred="UNSUPPORTED", src="frozen"))
            else:
                sel.append(dict(id=i, route=route_of[b], nth=24, op=opn, type=typ, shape=None, mode="rand" if opn == "MUL_MAT_ID" else None,
                                pred="PASS" if pred == "PASS" else pk, src="frozen"))
        elif opn == "FLASH_ATTN_EXT":
            sel.append(dict(id=i, route="cpu", nth=24, op=opn, type=typ, shape=i.rsplit("-", 1)[1], mode=None,
                            pred="PASS" if pred == "PASS" else pk, src="frozen"))
        elif opn == "GATED_DELTA_NET":
            sel.append(dict(id=i, route="cpu", nth=24, op=opn, type="F32", shape=None, mode=None, pred="PASS", src="frozen", shapes=GDN_SHAPES[:2]))
    return sel


def addendum_selectors():
    a = []
    def add(i, **kw):
        a.append(dict(id=i, src="addendum", **kw))
    for t in ("IQ2_XXS", "IQ2_XS", "IQ3_S"):
        add(f"ADD-CPU-MMID-{t}-shared", route="cpu", nth=24, op="MUL_MAT_ID", type=t, shape=None, mode="shared", pred=2)
        add(f"ADD-CPU-MMID-{t}-disjoint", route="cpu", nth=24, op="MUL_MAT_ID", type=t, shape=None, mode="disjoint", pred="PASS")
    add("ADD-CPU-MMID-IQ3_XXS-shared", route="cpu", nth=24, op="MUL_MAT_ID", type="IQ3_XXS", shape=None, mode="shared", pred=2)
    add("ADD-CPU-MMID-IQ3_XXS-disjoint", route="cpu", nth=24, op="MUL_MAT_ID", type="IQ3_XXS", shape=None, mode="disjoint", pred=2)
    for t in ("Q8_0", "Q4_K", "Q3_K", "Q6_K", "IQ4_XS"):
        for m in ("shared", "disjoint"):
            add(f"ADD-CPU-MMID-{t}-{m}", route="cpu", nth=24, op="MUL_MAT_ID", type=t, shape=None, mode=m, pred="PASS")
    add("ADD-CPU-FA-F16-nth1", route="cpu", nth=1, op="FLASH_ATTN_EXT", type="F16", shape=None, mode=None, pred="PASS")
    for t in ("Q8_0", "Q4_0"):
        add(f"ADD-CPU-FA-{t}-nth1", route="cpu", nth=1, op="FLASH_ATTN_EXT", type=t, shape=None, mode=None, pred="PASS")
    for nth in (24, 1):
        add(f"ADD-CPU-GDN-all-shapes-nth{nth}", route="cpu", nth=nth, op="GATED_DELTA_NET", type="F32", shape=None, mode=None, pred="PASS")
    add("ADD-CPU-MM-Q8_0-nth1", route="cpu", nth=1, op="MUL_MAT", type="Q8_0", shape=None, mode=None, pred=2)
    for t in ("IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S"):
        add(f"ADD-CPU-MM-{t}-nth1", route="cpu", nth=1, op="MUL_MAT", type=t, shape=None, mode=None, pred=2)
    rx = "cpu[GGML_IQK=1,GGML_ROWEXACT_N=9]"
    for t in KQ:
        add(f"ADD-CPU-MM-{t}-rowexact9", route=rx, nth=24, op="MUL_MAT", type=t, shape=None, mode=None, pred="PASS")
    add("ADD-CTL-positive", route="cpu", nth=24, op="CONTROL", type="F32", shape="ctl-positive-width-dependent", mode=None, pred=2)
    add("ADD-CTL-negative", route="cpu", nth=24, op="CONTROL", type="F32", shape="ctl-negative-width-independent", mode=None, pred="PASS")
    add("ADD-CTL-bitflip", route="cpu", nth=24, op="CONTROL", type="F32", shape="ctl-bitflip-injection", mode=None, pred=1)
    return a


def score(selectors, cells):
    out = []
    for s in selectors:
        got = []
        for key, per_k in cells.items():
            route, nth, op, typ, shape, mode = key
            if route != s["route"] or nth != s["nth"] or op != s["op"] or typ != s["type"]:
                continue
            if s.get("shape") is not None and shape != s["shape"]:
                continue
            if s.get("shapes") is not None and shape not in s["shapes"]:
                continue
            if s.get("mode") is not None and mode != s["mode"]:
                continue
            pat, first, status = summarize(per_k)
            got.append((shape, mode, pat, first, status))
        if not got:
            out.append({**s, "result": "NOT_MEASURED", "detail": []})
            continue
        pred = s["pred"]
        verdicts = []
        for shape, mode, pat, first, status in got:
            if pred == "UNSUPPORTED":
                ok = status == "UNSUPPORTED"
            elif status in ("VOID_AA", "ERR"):
                ok = None
            elif status == "UNSUPPORTED":
                continue            # a shape that cannot be constructed is not part of an 'ALL shapes' prediction
            elif pred == "PASS":
                ok = status == "PASS"
            else:
                ok = status == "FAIL" and first == pred
            verdicts.append((shape, mode, pat, first, status, ok))
        if not verdicts:
            out.append({**s, "result": "NOT_MEASURED", "detail": []})
            continue
        if any(v[5] is False for v in verdicts):
            res = "falsified"
        elif any(v[5] is None for v in verdicts):
            res = "void"
        else:
            res = "confirmed"
        out.append({**s, "result": res,
                    "detail": [{"shape": v[0], "mode": v[1], "pattern": v[2], "first_fail_k": v[3], "status": v[4], "match": v[5]} for v in verdicts]})
    return out


# ------------------------------------------------------------------------------------------ table
def render_tables(cells):
    lines = []
    by = defaultdict(lambda: defaultdict(dict))   # (op, route, nth, mode) -> type -> shape -> label
    for key, per_k in cells.items():
        route, nth, op, typ, shape, mode = key
        pat, first, status = summarize(per_k)
        by[(op, route, nth, mode)][typ][shape] = cell_label(status, first) + ("" if status in ("UNSUPPORTED",) else "")
    for (op, route, nth, mode), types in sorted(by.items()):
        shapes = sorted({s for t in types.values() for s in t}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
        lines.append(f"\n### {op}  route=`{route}`  nth={nth}" + (f"  ids-mode={mode}" if mode != "-" else ""))
        lines.append("| type | " + " | ".join(shapes) + " |")
        lines.append("|---|" + "---|" * len(shapes))
        for t in sorted(types, key=lambda x: KQ.index(x) if x in KQ else 99):
            lines.append(f"| {t} | " + " | ".join(types[t].get(s, "MISSING") for s in shapes) + " |")
    return "\n".join(lines)


def table(args):
    out_dir = Path(args.out_dir)
    cells, headers = load_cells(out_dir)
    # merge every cell into one JSONL (the evidence file)
    merged = out_dir / "cells.jsonl"
    with open(merged, "w") as f:
        for key in sorted(cells):
            for k in sorted(cells[key]):
                f.write(json.dumps(cells[key][k], sort_keys=True) + "\n")
    # expected-grid coverage: missing groups are ERR rows, never omissions
    missing = []
    for key in expected_cells():
        per_k = cells.get(key, {})
        if len(per_k) != KMAX:
            missing.append({"key": list(key), "have_k": sorted(per_k)})
    nondet = [list(k) + [kk] for k, per in cells.items() for kk, r in per.items() if r["verdict"] == "NONDET"]
    errs = [list(k) + [kk] for k, per in cells.items() for kk, r in per.items() if r["verdict"] == "ERR"]
    n_cells = sum(len(p) for p in cells.values())
    n_aa_unclean = sum(1 for p in cells.values() for r in p.values() if r["aa_clean"] is False)
    n_with_reps = sum(1 for p in cells.values() for r in p.values() if r["aa_clean"] is not None)

    frozen_rows = load_frozen(Path(args.frozen))
    scored = score(frozen_selectors(frozen_rows) + addendum_selectors(), cells)
    summ = defaultdict(int)
    for s in scored:
        summ[(s["src"], s["result"])] += 1

    ctl = {}
    for sh in CTL_SHAPES:
        per = cells.get(("cpu", 24, "CONTROL", "F32", sh, "-"), {})
        ctl[sh] = summarize(per)[0] if per else "MISSING"
    ctl_ok = (ctl.get(CTL_SHAPES[0], "")[:1] == "P" and set(ctl.get(CTL_SHAPES[0], "P")[1:]) == {"F"}
              and ctl.get(CTL_SHAPES[1]) == "P" * KMAX and ctl.get(CTL_SHAPES[2]) == "F" * KMAX)

    report = {
        "schema": "akx-all-26.cpu-summary.v1",
        "n_cells": n_cells, "n_groups": len(cells), "n_group_headers": len(headers),
        "n_cells_with_reps": n_with_reps, "n_aa_unclean": n_aa_unclean,
        "missing_groups": missing, "nondet_cells": nondet, "err_cells": errs,
        "controls": ctl, "controls_ok": ctl_ok,
        "scoring_counts": {f"{a}/{b}": n for (a, b), n in sorted(summ.items())},
        "scoring": scored,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    md = [f"# AKX-ALL-26 CPU first-fail-k tables", "",
          f"cells={n_cells} groups={len(cells)} A/A-unclean cells={n_aa_unclean}/{n_with_reps} missing groups={len(missing)} err cells={len(errs)}",
          f"controls: {json.dumps(ctl)} controls_ok={ctl_ok}", "", "Legend: PASS = bit-equal for k=1..9; F@N = first width N whose batched bytes differ from the serial ones; UNSUP = cannot be constructed.",
          render_tables(cells), "", "## Prediction scoring", "", "| id | source | selector | predicted | result | measured |", "|---|---|---|---|---|---|"]
    for s in scored:
        det = "; ".join(f"{d['shape']}{('/' + d['mode']) if d['mode'] not in (None, '-') else ''}:{d['pattern']}" for d in s["detail"][:12])
        sel = f"{s['route']} nth{s['nth']} {s['op']} {s['type']}" + (f" {s['shape']}" if s.get("shape") else "") + (f" {s['mode']}" if s.get("mode") else "")
        md.append(f"| {s['id']} | {s['src']} | {sel} | {s['pred']} | {s['result']} | {det} |")
    (out_dir / "first_fail_tables.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("scoring", "missing_groups", "nondet_cells", "err_cells")}, indent=1))
    print("missing groups:", len(missing), "nondet:", len(nondet), "err:", len(errs))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ggml-src", required=True)
    b.add_argument("--build-dir", required=True)
    b.add_argument("--probe", required=True)
    r = sub.add_parser("run")
    r.add_argument("--probe", required=True)
    r.add_argument("--build-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--only", default="")
    t = sub.add_parser("table")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--frozen", required=True)
    a = ap.parse_args()
    return {"build": build, "run": run, "table": table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
