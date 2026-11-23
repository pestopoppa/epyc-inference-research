#!/usr/bin/env python3
"""Offline served-T re-test of spec-decode decisions parked on greedy evidence (PREDECL.md, sealed first).
Pure stdlib, deterministic (seeded). Recorded data only; no inference.
Usage: greedy_retest.py --out DIR
"""
import argparse, hashlib, json, math, os, random, statistics as st

DICE = "/mnt/raid0/llm/tmp/ak-greedy-retest-20261123/dice_ref/RESULT.json"  # git show 03c100462
CAP = "/mnt/raid0/llm/tmp/ak-dice-capture-a3b-mtp-cpu-20261123-r2/out/capture/cap-L01.jsonl"
UNITS = "/mnt/raid0/llm/tmp/ak-dice-capture-a3b-mtp-cpu-20261123-r2/out/results.jsonl"
PHASE = "/mnt/raid0/llm/tmp/ak-dtree-phase-gpu-20261122-r2/out/results.jsonl"
STOCH = "/mnt/raid0/llm/tmp/ak-dtree-stoch-27b-gpu-20261121/out/RESULT.json"
C10 = "/mnt/raid0/llm/tmp/c10-verify-slope-20261109/out/RESULT.json"
CAFE1 = "/workspace/artifacts/compute/cafe1-replay-20261101/ANALYSIS.json"
CAFE2 = "/mnt/raid0/llm/tmp/inf50-runs-20261107/cafe2-verify-cost-native-r2/out/RESULT.json"
DEPTH = "/mnt/raid0/llm/tmp/ak-q38fn-cpu-draft-depth-20261121-r2/out/RESULT.json"
TS = [0.1, 0.3, 1.0]
GPU_FLOOR = 1.0084
B = 2000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def q(xs, a):
    xs = sorted(xs)
    i = a * (len(xs) - 1)
    lo = int(math.floor(i))
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


def dice_cell(d, T, W, key="V2/V0"):
    for r in d["table_holdout_prompts_32_63"]:
        if r["T"] == T and r["W"] == W:
            pt, lo, hi, _ = r[key]
            return pt, (hi - lo) / 2 / 1.96
    raise KeyError((T, W))


# ------------------------------------------------------------------ item 1
def item1(d):
    pairs = [json.loads(l) for l in open(PHASE)]
    per = []
    for p in pairs:
        row = {}
        for arm in ("L@1", "T12@1", "T16@1"):
            a = p["arms"][arm]
            tau = st.mean(x["mean_len"] for x in a["accept"])
            row[arm] = {"tau": tau, "tps": a["tps"], "ms": 1000.0 * tau / a["tps"]}
        per.append(row)
    rng = random.Random(20261123)
    out = {"pairs": len(per), "L_ms_mean": st.mean(r["L@1"]["ms"] for r in per), "cells": []}
    for W, arm in ((12, "T12@1"), (16, "T16@1")):
        cost = [r[arm]["ms"] / r["L@1"]["ms"] for r in per]
        tau_g = [r[arm]["tau"] / r["L@1"]["tau"] for r in per]
        g01, s01 = dice_cell(d, 0.1, W)
        for T in TS:
            gT, sT = dice_cell(d, T, W)
            res = {"T": T, "W": W, "cost_ratio": st.mean(cost), "tau_greedy_27b": st.mean(tau_g),
                   "dice_V2V0": gT, "dice_drift_vs_T0.1": gT / g01}
            for name in ("P1_transfer", "P2_absolute"):
                bs = []
                for _ in range(B):
                    idx = [rng.randrange(len(per)) for _ in per]
                    c = st.mean(cost[i] for i in idx)
                    dT = rng.gauss(gT, sT)
                    if name == "P1_transfer":
                        t = st.mean(tau_g[i] for i in idx) * dT / rng.gauss(g01, s01)
                    else:
                        t = dT
                    bs.append(t / c)
                if name == "P1_transfer":
                    pt = st.mean(tau_g) * gT / g01 / st.mean(cost)
                else:
                    pt = gT / st.mean(cost)
                res[name] = {"S": pt, "ci95": [q(bs, 0.025), q(bs, 0.975)]}
            best = max((res["P1_transfer"], res["P2_absolute"]), key=lambda x: x["S"])
            lo, hi = best["ci95"]
            v = "PROJECTED_WIN" if lo > GPU_FLOOR else ("PROJECTED_LOSS" if hi < 1.0 else "PROJECTED_PARITY")
            res["verdict_basis"] = "more favourable of P1/P2"
            res["verdict"] = v
            tau_best = best["S"] * st.mean(cost)
            res["required_tree_ms"] = out["L_ms_mean"] * tau_best
            res["measured_tree_ms"] = st.mean(r[arm]["ms"] for r in per)
            out["cells"].append(res)
    s = json.load(open(STOCH))["summary"]
    out["crosscheck_measured_T1"] = {"A_auto": s["arms"]["A"]["speedup_mean"], "A_ci": s["arms"]["A"]["speedup_ci95"],
                                     "T12": s["arms"]["T12"]["speedup_mean"], "T12_ci": s["arms"]["T12"]["speedup_ci95"],
                                     "mean_len": s["mean_len_mean"], "ms_per_round": s["ms_per_round_mean"]}
    for c in out["cells"]:
        if c["T"] == 1.0 and c["verdict"] == "PROJECTED_WIN":
            c["verdict"] = "PROJECTED_LOSS (measurement overrides: DTREE-STOCH T=1 lost)"
    return out


# ------------------------------------------------------------------ item 2
def item2(d):
    c10 = json.load(open(C10))
    s = c10["s"]["short"]
    sp, slo, shi = s["s_ms_per_row"], s["ci95_lo"], s["ci95_hi"]
    tb = c10["slopes_27b"]["short"]["abs_ms_mean"]["breadth"]["8"]
    cells = []
    for W in (12, 16):
        dM = W - 8
        req = {k: 1.05 * (1 + v * dM / tb) for k, v in (("s", sp), ("s_lo", slo), ("s_hi", shi))}
        for T in TS:
            g, se = dice_cell(d, T, W)
            lo, hi = g - 1.96 * se, g + 1.96 * se
            v = "CLEARS" if lo > req["s_hi"] else ("FAILS" if hi < req["s_lo"] else "MARGINAL")
            cells.append({"T": T, "W": W, "required_tau_ratio": req, "dice_V2V0": [g, lo, hi],
                          "margin_pt": g - req["s"], "verdict": v})
    # the brief's 1.17 figure: what dM reproduces it
    return {"s": [sp, slo, shi], "t_base_ms_M8": tb, "cells": cells,
            "s_needed_for_1.05_at_dice_T": {str(T): {str(W): (dice_cell(d, T, W)[0] / 1.05 - 1) * tb / (W - 8)
                                                    for W in (12, 16)} for T in TS}}


# ------------------------------------------------------------------ capture pass (items 3, 4)
def load_capture():
    units = [json.loads(l) for l in open(UNITS)]
    rounds = []
    with open(CAP, "rb") as f:
        for u in units:
            f.seek(u["off0"])
            blob = f.read(u["off1"] - u["off0"]).decode()
            for line in blob.splitlines():
                r = json.loads(line)
                if r.get("k") != "round":
                    continue
                rows = r["rows"]
                nrow = 0
                for x in rows:
                    if x is None:
                        break
                    nrow += 1
                L = max(0, min(r["D"], len(r["chain"]), len(r["mtp_top"]), nrow - 1))
                pc, qc = [], []
                for j in range(L):
                    ps = {int(c): float(p) for c, p in rows[j][1]}
                    s = sum(ps.values()) or 1.0
                    pc.append(ps.get(r["chain"][j], 0.0) / s)
                    qd = {int(t[0]): float(t[2]) for t in r["mtp_top"][j]}
                    qc.append(qd.get(r["chain"][j], 0.0))
                real = 0
                for j, t in enumerate(r["acc"][:-1] if r.get("acc") else []):
                    if j < len(r["chain"]) and t == r["chain"][j]:
                        real += 1
                    else:
                        break
                rounds.append({"T": u["T"], "prompt": u["prompt_idx"], "unit": u["unit_id"], "L": L, "pc": pc,
                               "qc": qc, "real": real, "ndraft": r.get("n_draft", L)})
    return rounds


def item3(rounds):
    a = json.load(open(CAFE1))
    fl = a["floors_pct"]
    resc = {}
    for k, c in a["candidate_results"].items():
        g = c["gain_pct"]
        ok = g["aggregate"] >= 1.0 + fl["aggregate"] and all(g[x] >= -(1.0 + fl[x]) for x in ("code", "rag", "prose"))
        resc[k] = {"p_min": c["p_min"], "gain_pct": g, "identity": f"{c['matched_token_identity']}/{c['matched_total']}",
                   "throughput_only": "THROUGHPUT_PASS" if ok else "THROUGHPUT_FAIL"}
    cafe2 = json.load(open(CAFE2))["stats"]["median_ns"]
    cost = {int(k): v / 1e6 for k, v in cafe2.items()}  # ms per verify of n rows
    proj = {}
    for T in TS:
        R = [r for r in rounds if r["T"] == T]
        proj[str(T)] = {}
        for pm in (0.0, 0.5, 0.75, 0.9):
            tok, ms, rows_ = 0.0, 0.0, 0.0
            for r in R:
                k = 0
                while k < r["L"] and r["qc"][k] >= pm:
                    k += 1
                e, pr = 0.0, 1.0
                for j in range(k):
                    pr *= r["pc"][j]
                    e += pr
                tok += 1 + e
                rows_ += k + 1
                ms += cost[k + 1]
            proj[str(T)][str(pm)] = {"tok_per_pass": tok / len(R), "rows_per_round": rows_ / len(R),
                                     "verify_ms_per_round": ms / len(R), "tok_per_verify_s": 1000 * tok / ms}
        base = proj[str(T)]["0.0"]["tok_per_verify_s"]
        for pm in ("0.5", "0.75", "0.9"):
            proj[str(T)][pm]["ratio_vs_pmin0"] = proj[str(T)][pm]["tok_per_verify_s"] / base
    return {"cafe1_temperature": 0, "floors_pct": fl, "rescored": resc, "cafe2_verify_ms_by_rows": cost,
            "served_T_projection_verify_only": proj,
            "note": "projection prices only target verify (CAFE-2 r2 medians, 1 prompt, 24t); MTP draft cost per depth "
                    "not on disk and is saved by p-min too, so ratios are a lower bound on the p-min benefit"}


def item4(rounds):
    acc = {}
    for T in TS:
        R = [r for r in rounds if r["T"] == T]
        num = [0.0] * 5
        cnt = [0] * 5
        for r in R:
            pr = 1.0
            for j in range(r["L"]):
                num[j] += pr * r["pc"][j]
                cnt[j] += 1
                pr *= r["pc"][j]
        # conditional a_j = E[prod_{i<=j}] / E[prod_{i<j}] over rounds with L > j
        cond, marg = [], []
        for j in range(4):
            Rj = [r for r in R if r["L"] > j]
            m1 = st.mean(math.prod(r["pc"][:j + 1]) for r in Rj)
            m0 = st.mean(math.prod(r["pc"][:j]) for r in Rj)
            cond.append(m1 / m0)
            marg.append(m1)
        acc[str(T)] = {"cond_accept": cond, "marginal_reach": marg, "n_rounds": len(R)}
    a4 = {T: acc[T]["cond_accept"][3] for T in acc}
    g = json.load(open(DEPTH))["phase1_D1K"]
    n4, n6 = g["4"], g["6"]
    proj = {}
    for T in ("0.3", "1.0"):
        f = a4[T] / a4["0.1"]
        # n6 gain over n4 attributed to the extra depths 5,6: scale the tau increment by f^1, f^2 (avg 1.5)
        dtau = n6["emit_per_round"] - n4["emit_per_round"]
        tau6 = n4["emit_per_round"] + dtau * (f + f * f) / 2
        ratio = (tau6 / n6["ms_per_round"]) / (n4["emit_per_round"] / n4["ms_per_round"])
        proj[T] = {"a4_ratio_vs_T0.1": f, "n6_over_n4_projected": ratio,
                   "measured_greedy_n6_over_n4": n6["ratio_geomean"], "crosses_1.0415": ratio > 1.0415}
    # hyp 18
    req = {}
    for r in rounds:
        req.setdefault((r["T"], r["prompt"]), []).append(r)
    h18 = {}
    for T in TS:
        means, trips, healthy, minw = [], 0, 0, []
        for (t, p), R in req.items():
            if t != T:
                continue
            dr = sum(x["L"] for x in R) or 1
            m = sum(x["real"] for x in R) / dr
            means.append(m)
            wmin = 1.0
            i = 0
            while i < len(R):
                tok, a_, d_ = 0, 0, 0
                j = i
                while j < len(R) and tok < 64:
                    tok += R[j]["real"] + 1
                    a_ += R[j]["real"]
                    d_ += R[j]["L"]
                    j += 1
                if tok >= 64 and d_:
                    wmin = min(wmin, a_ / d_)
                i += 1
            minw.append(wmin)
            if m >= 0.5:
                healthy += 1
                if wmin < 0.2:
                    trips += 1
        h18[str(T)] = {"requests": len(means), "mean_accept": st.mean(means),
                       "share_below_0.5": sum(m < 0.5 for m in means) / len(means),
                       "healthy": healthy, "healthy_with_64tok_window_below_0.2": trips,
                       "min_window_accept_p05": q(minw, 0.05), "min_window_accept_min": min(minw)}
    moves = any(h18[str(T)]["healthy_with_64tok_window_below_0.2"] > 0 for T in TS) or \
        (h18["1.0"]["share_below_0.5"] - h18["0.1"]["share_below_0.5"]) > 0.10
    return {"per_depth": acc, "q38fn_depth_projection": proj, "hyp18": h18, "hyp18_threshold_moves": moves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(DICE))
    rounds = load_capture()
    res = {"task": "greedy-parked spec-decode re-test at served T (offline)", "at": "2026-11-23",
           "predecl_sha256": sha("/mnt/raid0/llm/tmp/ak-greedy-retest-20261123/PREDECL.md"),
           "inputs_sha256": {p: sha(p) for p in (DICE, CAP, PHASE, STOCH, C10, CAFE1, CAFE2, DEPTH)},
           "item1_dtree": item1(d), "item2_akm_c10": item2(d), "item3_cafe1": item3(rounds),
           "item4_depth_hyp18": item4(rounds)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "RESULT.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:200])


if __name__ == "__main__":
    main()
