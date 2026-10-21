#!/usr/bin/env python3
"""Analysis for the GPU -> CPU timing contamination experiment (pre-registered).

Pre-registration: epyc-root docs/design/gpu-cpu-noise-contribution-prereg-20261021.md
Input: <session>/reps.jsonl (schema epyc.gpu_cpu_noise.rep.v1, written by gpu_cpu_noise_driver.py).

Per GPU condition k (vs G0 in the same blocks) and metric (tg = decode, primary; pp = prefill):
  BIAS   b_k = median(k reps)/median(G0 reps) - 1  (the matched_process_v2 estimator shape),
         95% CI = union of the block-bootstrap percentile CI (blocks resampled, pairing kept)
         and the t-CI of the per-block log ratios. Decisions use the union (the wider one).
  NOISE  process unit: pooled within-run variance of log(rep value), VR_k = s2_k / s2_G0
         (F-CI, plus block-bootstrap CI); sigma_add = sqrt(max(0, s2_k - s2_G0)), in %.
         arm unit (within one llama-bench process): pooled variance of log(samples_ts).
  A/A    the two G0 runs of each block (validity gate V1) and the INF-70 continuity figure
         "A/A pair p95" (non-overlapping consecutive rep pairs within runs).
Decision table per timing class (DG / AK / IND) exactly as pre-registered; see DECISION below.
numpy only; scipy (if importable) adds the parametric t/F intervals.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as _st  # optional
except Exception:  # pragma: no cover
    _st = None

CONDS = ["G1", "G2", "G3", "G4", "S8"]
METRICS = ["tg", "pp"]
N_BOOT = 20000
# ---- DECISION (pre-registered; do not edit after the session starts) ----
AK_FLOOR_SD_PCT = 2.793          # matched_process_v2 serving floor, process unit (INF-70 RETEST-1, MEASUREMENT FLOOR-UNIT-1)
WIDEN_MAX = 0.10                 # a class tolerates <=10% widening of its own floor
ADD_FRAC = math.sqrt((1 + WIDEN_MAX) ** 2 - 1)   # = 0.458: sigma_add <= 0.458 * sigma_class
CLASSES = {
    "DG":  {"bias_bound_pct": 0.5, "vr_proc_u95_max": 2.0, "vr_arm_u95_max": 1.5},
    "AK":  {"bias_bound_pct": 1.0, "sigma_add_u95_max_pct": ADD_FRAC * AK_FLOOR_SD_PCT},
    "IND": {"large_bias_pct": 5.0},
}
AA_BIAS_MAX_PCT = 0.5            # V1
VALID_FRAC_MIN = 0.8             # V2


def load(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in rows:
        body = json.dumps({k: v for k, v in r.items() if k != "content_sha256"}, sort_keys=True).encode()
        r["_hash_ok"] = hashlib.sha256(body).hexdigest() == r.get("content_sha256")
        out.append(r)
    return out


def rep_value(r: dict, m: str) -> float | None:
    d = r.get(m) or {}
    s = d.get("samples_ts") or []
    return float(np.mean(s)) if s else None


def organize(rows: list[dict], m: str, include_invalid: bool = False):
    """-> runs[(block, run, cond)] = list of (log value, log samples); retries replace failures."""
    runs: dict = defaultdict(list)
    meas = [r for r in rows if r.get("phase") == "measure" and r["_hash_ok"]]
    by_id = {}
    for r in meas:
        base = r["rep_id"].replace("-retry", "")
        ok = r.get("valid") or (include_invalid and rep_value(r, m) is not None)
        if ok and rep_value(r, m) is not None:
            by_id[base] = r
    for r in by_id.values():
        s = [math.log(x) for x in (r[m].get("samples_ts") or []) if x and x > 0]
        runs[(r["block"], r["run"], r["cond"])].append((math.log(rep_value(r, m)), s))
    return runs


def pooled_var(groups: list[list[float]]) -> tuple[float, int]:
    ss = df = 0.0
    for g in groups:
        if len(g) >= 2:
            a = np.asarray(g)
            ss += float(((a - a.mean()) ** 2).sum())
            df += len(g) - 1
    return (ss / df if df else float("nan")), int(df)


def complete_blocks(runs, cond: str) -> list[int]:
    blocks = sorted({b for (b, _, c) in runs})
    return [b for b in blocks if any(c == cond for (bb, _, c) in runs if bb == b)
            and any(c == "G0" for (bb, _, c) in runs if bb == b)]


def block_data(runs, cond: str, blocks: list[int]):
    cand = {b: [v for (bb, _, c), reps in runs.items() if bb == b and c == cond for v, _ in reps] for b in blocks}
    anch = {b: [v for (bb, _, c), reps in runs.items() if bb == b and c == "G0" for v, _ in reps] for b in blocks}
    return cand, anch


def bias_estimate(cand: dict, anch: dict, blocks: list[int], rng) -> dict:
    def est(bs):
        c = np.concatenate([cand[b] for b in bs])
        a = np.concatenate([anch[b] for b in bs])
        return math.exp(np.median(c) - np.median(a)) - 1
    if len(blocks) < 2:
        return {"n_blocks": len(blocks)}
    point = est(blocks)
    pick = rng.integers(0, len(blocks), size=(N_BOOT, len(blocks)))
    boots = np.array([est([blocks[i] for i in row]) for row in pick])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lr = np.array([np.median(cand[b]) - np.median(anch[b]) for b in blocks])
    out = {"n_blocks": len(blocks), "bias_pct": 100 * point, "boot_ci_pct": [100 * lo, 100 * hi],
           "boot_p95_abs_pct": 100 * float(np.percentile(np.abs(boots), 95)),
           "mean_log_ratio_pct": 100 * (math.exp(lr.mean()) - 1)}
    if _st is not None and len(lr) >= 2:
        h = _st.t.ppf(0.975, len(lr) - 1) * lr.std(ddof=1) / math.sqrt(len(lr))
        out["t_ci_pct"] = [100 * (math.exp(lr.mean() - h) - 1), 100 * (math.exp(lr.mean() + h) - 1)]
    cis = [out["boot_ci_pct"]] + ([out["t_ci_pct"]] if "t_ci_pct" in out else [])
    out["ci_pct"] = [min(c[0] for c in cis), max(c[1] for c in cis)]
    return out


def noise_estimate(runs, cond: str, rng) -> dict:
    def groups(c, unit):
        if unit == "process":
            return {k: [v for v, _ in reps] for k, reps in runs.items() if k[2] == c}
        return {(k, i): s for k, reps in runs.items() if k[2] == c for i, (_, s) in enumerate(reps)}
    out = {}
    for unit in ("process", "arm"):
        gk, g0 = groups(cond, unit), groups("G0", unit)
        vk, dfk = pooled_var(list(gk.values()))
        v0, df0 = pooled_var(list(g0.values()))
        if not (dfk and df0) or not v0 > 0:
            out[unit] = {"df_k": dfk, "df_0": df0}
            continue
        vr = vk / v0
        u = {"unit": unit, "sd_k_pct": 100 * math.sqrt(vk), "sd_0_pct": 100 * math.sqrt(v0), "df_k": dfk,
             "df_0": df0, "vr": vr, "sigma_add_pct": 100 * math.sqrt(max(0.0, vk - v0))}
        # block bootstrap: resample blocks; per-block sums of squares / df make it a vector sum
        keyb = (lambda k: k[0]) if unit == "process" else (lambda k: k[0][0])
        bl = sorted({keyb(k) for k in list(gk) + list(g0)})
        pos = {b: i for i, b in enumerate(bl)}
        ss = np.zeros((2, len(bl)))
        df = np.zeros((2, len(bl)))
        for row, g in ((0, gk), (1, g0)):
            for k, vals in g.items():
                if len(vals) >= 2:
                    a = np.asarray(vals)
                    ss[row, pos[keyb(k)]] += ((a - a.mean()) ** 2).sum()
                    df[row, pos[keyb(k)]] += len(vals) - 1
        pick = rng.integers(0, len(bl), size=(N_BOOT, len(bl)))
        S, F = ss[:, pick].sum(axis=2), df[:, pick].sum(axis=2)
        ok = (F[0] > 0) & (F[1] > 0)
        a_, z_ = S[0][ok] / F[0][ok], S[1][ok] / F[1][ok]
        ok2 = z_ > 0
        vrs = list(a_[ok2] / z_[ok2])
        adds = list(100 * np.sqrt(np.maximum(0.0, a_[ok2] - z_[ok2])))
        if vrs:
            u["vr_boot_ci"] = [float(np.percentile(vrs, 2.5)), float(np.percentile(vrs, 97.5))]
            u["sigma_add_boot_ci_pct"] = [float(np.percentile(adds, 2.5)), float(np.percentile(adds, 97.5))]
        if _st is not None:
            u["vr_f_ci"] = [vr / _st.f.ppf(0.975, dfk, df0), vr / _st.f.ppf(0.025, dfk, df0)]
            u["sigma_add_f_ci_pct"] = [100 * math.sqrt(max(0.0, v0 * (x - 1))) for x in u["vr_f_ci"]]
        cis = [u[k] for k in ("vr_boot_ci", "vr_f_ci") if k in u]
        if cis:
            u["vr_ci"] = [min(c[0] for c in cis), max(c[1] for c in cis)]
        cis = [u[k] for k in ("sigma_add_boot_ci_pct", "sigma_add_f_ci_pct") if k in u]
        if cis:
            u["sigma_add_ci_pct"] = [min(c[0] for c in cis), max(c[1] for c in cis)]
        out[unit] = u
    return out


def pair_p95(runs, cond: str) -> dict:
    diffs = []
    for k, reps in runs.items():
        if k[2] != cond:
            continue
        v = [math.exp(x) for x, _ in reps]
        for i in range(0, len(v) - 1, 2):
            diffs.append(abs(v[i] - v[i + 1]) / ((v[i] + v[i + 1]) / 2))
    if not diffs:
        return {"n_pairs": 0}
    d = np.asarray(diffs) * 100
    return {"n_pairs": len(d), "p95_pct": float(np.percentile(d, 95)), "sd_pct": float(d.std(ddof=1)) if len(d) > 1 else None,
            "quotable": len(d) >= 20, "unit": "process", "note": "INF-70 hot-harness A/A pair p95 shape"}


def aa_check(runs, rng) -> dict:
    by_block = defaultdict(list)
    for (b, j, c), reps in runs.items():
        if c == "G0":
            by_block[b].append((j, [v for v, _ in reps]))
    pairs = [(sorted(v)[0][1], sorted(v)[1][1]) for b, v in sorted(by_block.items()) if len(v) >= 2]
    if len(pairs) < 2:
        return {"n_blocks": len(pairs), "pass": None}
    cand = {i: np.asarray(p[1]) for i, p in enumerate(pairs)}
    anch = {i: np.asarray(p[0]) for i, p in enumerate(pairs)}
    res = bias_estimate(cand, anch, list(range(len(pairs))), rng)
    lo, hi = res["ci_pct"]
    res["pass"] = (lo <= 0 <= hi) and abs(res["bias_pct"]) <= AA_BIAS_MAX_PCT
    return res


def verdict(cls: str, bias: dict, noise: dict) -> dict:
    c = CLASSES[cls]
    if "ci_pct" not in bias:
        return {"verdict": "UNTESTED", "why": ["no paired blocks"]}
    lo, hi = bias["ci_pct"]
    why, bad, undecided = [], False, False
    if cls == "IND":
        big = abs(bias["bias_pct"]) > c["large_bias_pct"]
        return {"verdict": "SHAREABLE-LABELLED" + ("(LARGE-BIAS)" if big else ""),
                "why": [f"label every rep with the GPU job; measured bias {bias['bias_pct']:+.2f}%"]}
    bb = c["bias_bound_pct"]
    if min(abs(lo), abs(hi)) > bb and (lo > 0 or hi < 0):
        bad = True
        why.append(f"bias CI [{lo:+.2f},{hi:+.2f}]% lies outside +-{bb}%")
    elif lo < -bb or hi > bb:
        undecided = True
        why.append(f"bias CI [{lo:+.2f},{hi:+.2f}]% not inside +-{bb}%")
    if cls == "DG":
        for unit, key in (("process", "vr_proc_u95_max"), ("arm", "vr_arm_u95_max")):
            u = noise.get(unit, {})
            if "vr_ci" not in u:
                undecided = True
                why.append(f"{unit} VR not estimable")
                continue
            if u["vr_ci"][0] >= c[key]:
                bad = True
                why.append(f"{unit} VR lower CI {u['vr_ci'][0]:.2f} >= {c[key]}")
            elif u["vr_ci"][1] >= c[key]:
                undecided = True
                why.append(f"{unit} VR upper CI {u['vr_ci'][1]:.2f} >= {c[key]}")
    if cls == "AK":
        u = noise.get("process", {})
        lim = c["sigma_add_u95_max_pct"]
        if "sigma_add_ci_pct" not in u:
            undecided = True
            why.append("sigma_add not estimable")
        elif u["sigma_add_ci_pct"][0] > lim:
            bad = True
            why.append(f"sigma_add lower CI {u['sigma_add_ci_pct'][0]:.2f}% > {lim:.2f}%")
        elif u["sigma_add_ci_pct"][1] > lim:
            undecided = True
            why.append(f"sigma_add upper CI {u['sigma_add_ci_pct'][1]:.2f}% > {lim:.2f}%")
    v = "NOT-SHAREABLE" if bad else ("UNDECIDED" if undecided else "SHAREABLE")
    return {"verdict": v, "why": why}


def worst(a: str, b: str) -> str:
    order = ["NOT-SHAREABLE", "UNTESTED", "UNDECIDED", "SHAREABLE"]
    ka = next(i for i, o in enumerate(order) if a.startswith(o))
    kb = next(i for i, o in enumerate(order) if b.startswith(o))
    return a if ka <= kb else b


def gpu_side(sess: str, rows: list[dict]) -> dict:
    """Reverse direction (CPU bench -> GPU throughput): client requests overlapping a rep vs not."""
    spans = [(r["start"], r["end"]) for r in rows if r.get("phase") == "measure"]
    out = {}
    for f in sorted(glob.glob(os.path.join(sess, "gpu_client.*.jsonl"))):
        mode = f.split("gpu_client.")[1].split(".")[0]
        for line in open(f):
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = "predicted_per_second" if mode == "decode" else "prompt_per_second"
            if q.get(key) is None:
                continue
            inside = any(s <= q["t"] <= e for s, e in spans)
            out.setdefault(mode, {"in_rep": [], "between": []})["in_rep" if inside else "between"].append(q[key])
    return {m: {k: {"n": len(v), "median": float(np.median(v)) if v else None} for k, v in d.items()}
            for m, d in out.items()}


def analyze(sess: str, include_invalid: bool = False, seed: int = 20261021) -> dict:
    rows = load(os.path.join(sess, "reps.jsonl"))
    rng = np.random.default_rng(seed)
    rep = {"session": os.path.basename(sess.rstrip("/")), "n_rows": len(rows),
           "n_hash_bad": sum(1 for r in rows if not r["_hash_ok"]), "include_invalid": include_invalid,
           "classes": CLASSES, "ak_floor_sd_pct": AK_FLOOR_SD_PCT, "metrics": {}}
    meas = [r for r in rows if r.get("phase") == "measure"]
    rep["valid_frac"] = {c: (round(sum(1 for r in meas if r["cond"] == c and r.get("valid")) /
                                   max(1, sum(1 for r in meas if r["cond"] == c)), 3))
                         for c in ["G0"] + CONDS}
    rep["invalid_reasons"] = sorted({w for r in meas for w in (r.get("why_invalid") or [])})
    for m in METRICS:
        runs = organize(rows, m, include_invalid)
        mm = {"aa": aa_check(runs, rng), "pair_p95": {c: pair_p95(runs, c) for c in ["G0"] + CONDS}, "cond": {}}
        for c in CONDS:
            blocks = complete_blocks(runs, c)
            cand, anch = block_data(runs, c, blocks)
            cand = {b: np.asarray(v) for b, v in cand.items() if v}
            anch = {b: np.asarray(v) for b, v in anch.items() if v}
            blocks = [b for b in blocks if b in cand and b in anch]
            bias = bias_estimate(cand, anch, blocks, rng)
            noise = noise_estimate(runs, c, rng)
            mm["cond"][c] = {"bias": bias, "noise": noise,
                             "verdict": {cls: verdict(cls, bias, noise) for cls in CLASSES}}
        if all(k in mm["cond"].get("G4", {}).get("bias", {}) for k in ("bias_pct",)) and \
                "bias_pct" in mm["cond"].get("G2", {}).get("bias", {}):
            mm["mechanism"] = {"G4_minus_G2_pct": mm["cond"]["G4"]["bias"]["bias_pct"] - mm["cond"]["G2"]["bias"]["bias_pct"],
                               "S8_bias_pct": mm["cond"].get("S8", {}).get("bias", {}).get("bias_pct"),
                               "note": "G4-G2 = host-thread placement (node 1 local vs node 3 remote); S8 = SMT-only"}
        rep["metrics"][m] = mm
    # validity gates and the final table (tg AND pp must both pass: worse of the two)
    v1 = rep["metrics"]["tg"]["aa"].get("pass")
    rep["gates"] = {"V1_aa_tg": v1, "V1_aa_pp": rep["metrics"]["pp"]["aa"].get("pass"),
                    "V2_valid_frac": {c: f >= VALID_FRAC_MIN for c, f in rep["valid_frac"].items()}}
    table = {}
    for c in CONDS:
        table[c] = {}
        for cls in CLASSES:
            vt = rep["metrics"]["tg"]["cond"][c]["verdict"][cls]["verdict"]
            vp = rep["metrics"]["pp"]["cond"][c]["verdict"][cls]["verdict"]
            v = worst(vt, vp)
            if v1 is False:
                v = "WITHHELD(V1 A/A drift)"
            elif not rep["gates"]["V2_valid_frac"].get(c, False) or not rep["gates"]["V2_valid_frac"].get("G0", False):
                v = "UNTESTED(V2 positive controls)"
            table[c][cls] = v
    rep["decision_table"] = table
    rep["gpu_side"] = gpu_side(sess, rows)
    return rep


def render_md(r: dict) -> str:
    L = [f"# GPU -> CPU timing contamination, session {r['session']}", "",
         f"rows {r['n_rows']} (hash-bad {r['n_hash_bad']}); include_invalid={r['include_invalid']}; "
         f"valid fraction {r['valid_frac']}", "", "## Decision table (tg AND pp; worse of the two)", "",
         "| cond | DG (champion baselines) | AK (keep gates 1-3%) | IND (indicative) |", "|---|---|---|---|"]
    for c, d in r["decision_table"].items():
        L.append(f"| {c} | {d['DG']} | {d['AK']} | {d['IND']} |")
    for m, mm in r["metrics"].items():
        L += ["", f"## {m} ({'decode' if m == 'tg' else 'prefill'}), unit = process unless stated", "",
              f"A/A (G0 vs G0, same blocks): {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in mm['aa'].items() if k != 'boot_p95_abs_pct'})}",
              "", "| cond | n blk | bias % [95% CI] | VR proc [CI] | sigma_add % [CI] | VR arm [CI] | pair p95 % (n) |",
              "|---|---|---|---|---|---|---|"]
        for c in ["G0"] + CONDS:
            pp = mm["pair_p95"][c]
            p95 = f"{pp['p95_pct']:.2f} ({pp['n_pairs']})" if pp.get("n_pairs") else "-"
            if c == "G0":
                L.append(f"| G0 | - | reference | 1 | 0 | 1 | {p95} |")
                continue
            d = mm["cond"][c]
            b, np_, na = d["bias"], d["noise"].get("process", {}), d["noise"].get("arm", {})
            f = lambda x: "-" if x is None else f"{x:.2f}"
            ci = lambda x: "" if not x else f" [{x[0]:.2f},{x[1]:.2f}]"
            L.append(f"| {c} | {b.get('n_blocks', 0)} | {f(b.get('bias_pct'))}{ci(b.get('ci_pct'))} | "
                     f"{f(np_.get('vr'))}{ci(np_.get('vr_ci'))} | {f(np_.get('sigma_add_pct'))}{ci(np_.get('sigma_add_ci_pct'))} | "
                     f"{f(na.get('vr'))}{ci(na.get('vr_ci'))} | {p95} |")
        if mm.get("mechanism"):
            L += ["", f"Mechanism: {json.dumps(mm['mechanism'])}"]
    L += ["", f"GPU side (reverse direction): {json.dumps(r['gpu_side'])}", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("--include-invalid", action="store_true", help="sensitivity analysis (not the primary)")
    ap.add_argument("--out", default=None, help="write ANALYSIS.json + ANALYSIS.md here (default: session dir)")
    a = ap.parse_args()
    r = analyze(a.session_dir, a.include_invalid)
    out = a.out or a.session_dir
    tag = "-sensitivity" if a.include_invalid else ""
    with open(os.path.join(out, f"ANALYSIS{tag}.json"), "w") as fh:
        json.dump(r, fh, indent=1, default=float)
    with open(os.path.join(out, f"ANALYSIS{tag}.md"), "w") as fh:
        fh.write(render_md(r))
    print(render_md(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
