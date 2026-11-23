#!/usr/bin/env python3
"""SPEC-COUPLE-MTP-1 offline bound: MTP sample-and-compare vs argmax draft (V0) against
routing MTP through the rejection path with q = softmax(logit/T) over the MTP top-10 (R).

Usage: couple_mtp_replay.py <capture.jsonl> <results.jsonl> <out.json>
Capture schema dice.v1 (A3B MTP, D=4 captured, truncated to served D=2).

V0 per round: E = sum_k prod_{j<k} p_j(chain_j)            (dice_replay.v0_analytic)
R  per round: E = sum_k prod_{j<k} (1 - TV(q_j, p_j))       (per-position single-draft ceiling)
R is OPTIMISTIC: position j>0 reuses the target/drafter rows conditioned on the argmax prefix,
even when R would have accepted a different token at j-1 (the capture cannot condition on it).
tokens/pass = 1 + E. Ratio R/V0 = sum(1+E_R)/sum(1+E_V0), CI = prompt-cluster bootstrap (2000).
Also reports the box metric: mean per-position gap 1 - TV(q,p) - p(argmax q), over all served
positions (j < 2) and at j = 0 only.
"""
import json, math, random, sys

D = 2
TOPK = 10


def norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items() if v > 0} if s > 0 else {}


def softmax_top(top, T):
    top = top[:TOPK]
    m = max(float(t[1]) for t in top)
    return norm({int(t[0]): math.exp((float(t[1]) - m) / T) for t in top})


def tv(q, p):
    keys = set(q) | set(p)
    return 0.5 * sum(abs(q.get(k, 0.0) - p.get(k, 0.0)) for k in keys)


def main():
    cap, units_path, out_path = sys.argv[1:4]
    U = [json.loads(l) for l in open(units_path)]
    U = [u for u in U if "off0" in u]
    per = {}  # (T, prompt) -> accumulators
    for u in U:
        with open(cap, "rb") as f:
            f.seek(u["off0"]); blob = f.read(u["off1"] - u["off0"])
        a = per.setdefault((u["T"], u["prompt_idx"]), dict(n=0, v0=0.0, r=0.0, gap=0.0, npos=0, gap0=0.0, n0=0,
                                                            argmax_mismatch=0))
        for line in blob.decode().splitlines():
            rec = json.loads(line)
            if rec.get("k") != "round" or not rec.get("mtp_top"):
                continue
            rows = rec["rows"]; nrow = 0
            for r in rows:
                if r is None:
                    break
                nrow += 1
            L = max(0, min(rec["D"], len(rec["chain"]), len(rec["mtp_top"]), nrow - 1, D))
            chain = rec["chain"][:L]
            ev0, er, pv0, pr = 0.0, 0.0, 1.0, 1.0
            for j in range(L):
                p = norm({int(c): float(x) for c, x in rows[j][1]})
                q = softmax_top(rec["mtp_top"][j], rec["T"])
                if int(rec["mtp_top"][j][0][0]) != chain[j]:
                    a["argmax_mismatch"] += 1
                pa = p.get(chain[j], 0.0)
                acc_r = 1.0 - tv(q, p)
                pv0 *= pa; ev0 += pv0
                pr *= acc_r; er += pr
                g = acc_r - pa
                a["gap"] += g; a["npos"] += 1
                if j == 0:
                    a["gap0"] += g; a["n0"] += 1
            a["n"] += 1; a["v0"] += 1.0 + ev0; a["r"] += 1.0 + er
    rng = random.Random(1123)
    res = {"temps": {}}
    Ts = sorted({k[0] for k in per})
    tot_rounds = sum(v["n"] for v in per.values())
    for T in Ts:
        for split, sel in (("calibration", lambda i: i < 32), ("holdout", lambda i: i >= 32)):
            ps = [v for (t, i), v in per.items() if t == T and sel(i) and v["n"] > 0]
            S = lambda xs, k: sum(x[k] for x in xs)
            ratio = S(ps, "r") / S(ps, "v0")
            boots = []
            for _ in range(2000):
                b = [rng.choice(ps) for _ in ps]
                boots.append(S(b, "r") / S(b, "v0"))
            boots.sort()
            res["temps"].setdefault(str(T), {})[split] = {
                "prompts": len(ps), "rounds": S(ps, "n"),
                "round_share_of_all_T": round(S(ps, "n") / tot_rounds, 4),
                "tok_per_pass_V0": round(S(ps, "v0") / S(ps, "n"), 4),
                "tok_per_pass_R": round(S(ps, "r") / S(ps, "n"), 4),
                "ratio_R_V0": round(ratio, 4),
                "ci95": [round(boots[49], 4), round(boots[1949], 4)],
                "gap_all_pos": round(S(ps, "gap") / max(1, S(ps, "npos")), 4),
                "gap_pos0": round(S(ps, "gap0") / max(1, S(ps, "n0")), 4),
                "argmax_mismatch": S(ps, "argmax_mismatch"),
            }
    h = [res["temps"][str(T)]["holdout"] for T in Ts]
    res["futility_ci_upper_lt_1.02_all_T"] = all(x["ci95"][1] < 1.02 for x in h)
    res["box_stop_gap_lt_1pct_all_T"] = all(x["gap_all_pos"] < 0.01 for x in h)
    res["box_build_gap_ge_2pct_any_T"] = any(x["gap_all_pos"] >= 0.02 for x in h)
    json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
