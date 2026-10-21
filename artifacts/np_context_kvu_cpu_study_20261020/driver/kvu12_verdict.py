#!/usr/bin/env python3
"""KVU-12.4 predeclared verdict (PLAN section 10; PROPOSED-by-workspace-f8 2026-10-21, pending owner/operator ratification BEFORE the window).
Pure arithmetic over summary.jsonl rows; starts nothing.  Usage:
  kvu12_verdict.py --anchors A_start/summary.jsonl A_mid/summary.jsonl A_end/summary.jsonl \
                   --split-a S1/summary.jsonl --unified U/summary.jsonl --split-a2 S2/summary.jsonl [--json OUT]
Metric directions: decode tok/s HIGHER = better; TTFT seconds LOWER = better.  Unit of the floor: percent of the anchor cell's median,
unit = ANCHOR LAUNCH (n = 3 launches; each launch median over its waves; per-stream n = 8 clients x decode waves is reported alongside)."""
import argparse, json, statistics, sys

def rows(path):
    return [json.loads(x) for x in open(path) if x.strip()]

def cell(path, want_cell=None):
    rs = [r for r in rows(path) if r.get("cell") and r.get("result") in ("OK", "ERRORS", "REFUSED_OVERSIZE")]
    if want_cell:
        rs = [r for r in rs if r["cell"] == want_cell]
    if not rs:
        raise SystemExit(f"INVALID: no cell row in {path}")
    return rs[-1]

def spread_pct(v):
    m = statistics.mean(v)
    return (max(v) - min(v)) / m * 100.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", nargs=3, required=True); ap.add_argument("--split-a", required=True)
    ap.add_argument("--unified", required=True); ap.add_argument("--split-a2", required=True)
    ap.add_argument("--min-pct", type=float, default=3.0); ap.add_argument("--k", type=float, default=2.0); ap.add_argument("--json")
    a = ap.parse_args()
    inv = []
    anc = [cell(p, "t8short_independent") for p in a.anchors]
    A, U, A2 = cell(a.split_a, "64k_depth"), cell(a.unified, "64k_depth"), cell(a.split_a2, "64k_depth")
    for nm, r in (("anchor_start", anc[0]), ("anchor_mid", anc[1]), ("anchor_end", anc[2]), ("split_A", A), ("unified_B", U), ("split_A2", A2)):
        if r.get("result") != "OK":
            inv.append(f"{nm} result={r.get('result')}")
        if r.get("validation_problems"):
            inv.append(f"{nm} validation_problems={r['validation_problems']}")
    for nm, r in (("split_A", A), ("unified_B", U), ("split_A2", A2)):
        if (r.get("min_headroom_tok") is None) or r["min_headroom_tok"] < 0:
            inv.append(f"{nm} min_headroom_tok={r.get('min_headroom_tok')}")
    if len({r.get("binary_sha256") for r in anc + [A, U, A2]}) != 1:
        inv.append("binary sha256 differs across cells")
    out = dict(proposed_by="workspace-f8 2026-10-21", status="PROPOSED-until-ratified", invalid=inv)
    fl = {}
    for key in ("decode_tps_sum_med", "decode_tps_stream_med", "ttft_s_med"):
        fl[key] = round(spread_pct([r[key] for r in anc]), 2)
    aba = {key: round(abs(A[key] - A2[key]) / statistics.mean([A[key], A2[key]]) * 100.0, 2) for key in ("decode_tps_sum_med", "decode_tps_stream_med", "ttft_s_med")}
    out["anchor_floor_pct"] = fl; out["aba_split_spread_pct"] = aba
    out["floor_unit"] = "percent of the anchor-cell median; unit = anchor launch (n=3 launches at start/mid/end; each = median over its waves; per-stream n = 8 clients x decode waves within each launch)"
    thr = {k_: round(max(a.k * max(fl[k_], aba[k_]), a.min_pct), 2) for k_ in fl}
    out["threshold_pct"] = thr
    gap = {}
    for key in fl:
        base = statistics.mean([A[key], A2[key]])
        gap[key] = round((U[key] - base) / base * 100.0, 2)
    out["gap_unified_vs_mean_split_pct"] = gap
    ds, dp, tt = gap["decode_tps_sum_med"], gap["decode_tps_stream_med"], gap["ttft_s_med"]
    t_s, t_p, t_t = thr["decode_tps_sum_med"], thr["decode_tps_stream_med"], thr["ttft_s_med"]
    if inv:
        verdict = "INVALID"
    elif ds > t_s and dp > t_p and tt <= t_t:
        verdict = "UNIFIED_BETTER"
    elif ds < -t_s and dp < -t_p:
        verdict = "SPLIT_BETTER"
    elif abs(ds) <= t_s or abs(dp) <= t_p:
        verdict = "NO_DIFFERENCE"
    else:
        verdict = "MIXED"
    out["verdict"] = verdict
    print(json.dumps(out, indent=1))
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1) + "\n")
    sys.exit(0 if verdict != "INVALID" else 2)

if __name__ == "__main__":
    main()
