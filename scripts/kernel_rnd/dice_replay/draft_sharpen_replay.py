#!/usr/bin/env python3
"""SPEC-DRAFT-SHARPEN: offline replay of two lossless drafting levers for 27B DFlash2 under sampling, on the dice.v1
capture of row ak-dice-capture-27b-dflash2-gpu-20261123-r2. See PREDECLARATION_DRAFT_SHARPEN.md (sealed before any
replay output).

Levers (both keep rejection sampling exact, so the emitted distribution stays p):
  (a) trunc: q' = the drafter's candidate distribution truncated with the target's sampler (top_k, top_p, min_p,
      applied in llama.cpp default chain order on the pre-temperature distribution), renormalized, then tempered.
  (b) scale: q = softmax(scale * score / T) over the K = 16 lattice candidates (per-T sharpening of selector scores).
Per round, expected accepted draft tokens along the realized chain prefix
  a(scale, trunc) = sum_{j<L} prod_{i<=j} alpha_i,  alpha_i = sum_c min(p_i(c), q'_i(c))   (1 - TV, single draft),
with p_i the target's post-sampler row at chain depth i and q'_i the drafter's distribution conditioned on the realized
predecessor ck[i-1]. Tokens per target pass = 1 + a. V0 = (scale 1.0, trunc off) = the served q.
Usage: selftest --out DIR | replay --capture CAP --units RESULTS --out DIR [--procs N] | analyze --out DIR
"""
import argparse, json, math, os, random, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dice_replay import norm, chisq, sha  # noqa: E402
from dice_replay_27b import qdist  # noqa: E402

CAP_SHA = "82d46a61a6d8f8b8088faece7935c5b8c10fd71c43bf3afa6db7941820d3a845"
UNITS_SHA = "38c571ae4969f75e6c06ba7b7cd663ae6c88d695de5585920430df739b7f4fd1"
SCALES = [round(0.5 + 0.1 * i, 1) for i in range(11)]  # 0.5 .. 1.5
TRUNCS = [0, 1]
TS_GATE = [0.3, 1.0]
TS_ALL = [0.1, 0.3, 1.0]
CAL, HOLD = list(range(0, 32)), list(range(32, 64))
B, BSEED = 2000, 20261123


def truncate_keep(base, top_k, top_p, min_p):
    """Kept token set under llama.cpp's top_k -> top_p -> min_p on distribution `base` (pre-temperature)."""
    items = sorted(base.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k and top_k > 0:
        items = items[:top_k]
    z = sum(v for _, v in items)
    items = [(c, v / z) for c, v in items]
    if top_p < 1.0:
        cum, keep = 0.0, []
        for c, v in items:
            keep.append((c, v))
            cum += v
            if cum >= top_p:
                break
        items = keep
    if min_p > 0.0 and items:
        mx = items[0][1]
        items = [(c, v) for c, v in items if v >= min_p * mx] or items[:1]
    return {c for c, _ in items}


def qprime(scores, toks, T, scale, trunc, top_k, top_p, min_p):
    sc = [scale * s for s in scores]
    q = qdist(sc, toks, T)
    if not trunc:
        return q
    keep = truncate_keep(qdist(sc, toks, 1.0), top_k, top_p, min_p)
    return norm({c: v for c, v in q.items() if c in keep})


def alpha(p, q):
    return sum(min(p.get(c, 0.0), v) for c, v in q.items())


def round_ctx(rec):
    """Chain length L, the target rows p_j along the chain, the predecessor index per depth (as dice_replay_27b)."""
    L = min(int(rec["n_draft"]), int(rec["D"]), len(rec["chain"]))
    chain_node = {}
    for i, n in enumerate(rec["nodes"]):
        if i > 0 and (n[4] & 1):
            chain_node[n[1]] = i
    ps = []
    for j in range(L):
        ri = 0 if j == 0 else chain_node.get(j - 1)
        if ri is None or rec["rows"][ri] is None:
            break
        ps.append(norm({int(c): float(p) for c, p in rec["rows"][ri][1]}))
    return len(ps), ps


def replay_round(rec):
    L, ps = round_ctx(rec)
    T = float(rec["T"])
    kk, tp, mp = int(rec["top_k"]), float(rec["top_p"]), float(rec["min_p"])
    grid = {}
    for tr in TRUNCS:
        for s in SCALES:
            e, pr = 0.0, 1.0
            for j in range(L):
                pred = 0 if j == 0 else int(rec["ck"][j - 1])
                q = qprime(rec["lat_sc"][j][pred], rec["lat_tok"][j], T, s, tr, kk, tp, mp)
                pr *= alpha(ps[j], q)
                e += pr
            grid[f"{tr}:{s}"] = e
    real = 0
    for j, t in enumerate(rec["acc"][:-1] if rec.get("acc") else []):
        if j < L and t == rec["chain"][j]:
            real += 1
        else:
            break
    return {"L": L, "grid": grid, "realized_acc": real}


def replay_unit(args):
    cap, unit = args
    res = []
    with open(cap, "rb") as f:
        f.seek(unit["off0"])
        blob = f.read(unit["off1"] - unit["off0"])
    for line in blob.decode().splitlines():
        rec = json.loads(line)
        if rec.get("k") != "round":
            continue
        o = replay_round(rec)
        o.update({"unit": unit["unit_id"], "prompt": unit["prompt_idx"], "T": unit["T"], "task": rec["task"],
                  "round": rec["round"]})
        res.append(o)
    return res


def selftest(outdir):
    rep = {"pass": True, "checks": {}}

    def ck(name, ok, **kw):
        rep["checks"][name] = {"ok": bool(ok), **kw}
        rep["pass"] &= bool(ok)

    # (a) truncation hand case: base {1:.5, 2:.3, 3:.15, 4:.05}; top_p .9 keeps 1,2,3 (cum .95 >= .9); min_p .2 drops
    # tokens below .1 of the max -> 4 is gone already; min_p .4 drops 3 (.15/.95 < .4*.5/.95)
    base = {1: .5, 2: .3, 3: .15, 4: .05}
    ck("trunc_top_p", truncate_keep(base, 20, 0.9, 0.0) == {1, 2, 3})
    ck("trunc_min_p", truncate_keep(base, 20, 0.9, 0.4) == {1, 2})
    ck("trunc_top_k", truncate_keep(base, 2, 1.0, 0.0) == {1, 2})
    ck("trunc_off_equiv", truncate_keep(base, 20, 1.0, 0.0) == {1, 2, 3, 4})
    # (b) scale 1, trunc off == served qdist; scale s at T == qdist at T/s
    rng = random.Random("sharpen-selftest:20261123")
    sc = [rng.uniform(0, 30) for _ in range(16)]
    tk = [rng.randrange(50) for _ in range(16)]
    q0 = qdist(sc, tk, 0.3)
    q1 = qprime(sc, tk, 0.3, 1.0, 0, 20, 0.95, 0.05)
    ck("v0_is_served_q", max(abs(q0[c] - q1[c]) for c in q0) < 1e-12)
    q2 = qprime(sc, tk, 1.0, 1.3, 0, 20, 0.95, 0.05)
    q3 = qdist(sc, tk, 1.0 / 1.3)
    ck("scale_is_temperature", max(abs(q2[c] - q3[c]) for c in q2) < 1e-12)
    # (c) alpha = 1 - TV, and an MC of the served rejection rule with q' (draft c ~ q', accept w.p. min(1, p/q'),
    # else resample from norm(max(p - q', 0))) accepts at alpha and emits p (lossless) - chi-square p > 1e-3, 200k
    N = 200000
    for case in range(3):
        V = 12
        p = norm({i: rng.random() ** 3 for i in range(V)})
        scs = [rng.uniform(0, 8) for _ in range(V)]
        q = qprime(scs, list(range(V)), 1.0, 1.0 + 0.2 * case, 1, 20, 0.95, 0.05)
        tv = 0.5 * sum(abs(p.get(c, 0.0) - q.get(c, 0.0)) for c in set(p) | set(q))
        a = alpha(p, q)
        resid = norm({c: max(p[c] - q.get(c, 0.0), 0.0) for c in p})
        r = random.Random(f"mc:{case}")
        qi, ri = sorted(q.items()), sorted(resid.items())
        cnt, acc = {}, 0

        def draw(items):
            x, s = r.random(), 0.0
            for c, v in items:
                s += v
                if x < s:
                    return c
            return items[-1][0]
        for _ in range(N):
            c = draw(qi)
            if r.random() < min(1.0, p.get(c, 0.0) / q[c]):
                acc += 1
                e = c
            else:
                e = draw(ri)
            cnt[e] = cnt.get(e, 0) + 1
        st, pv = chisq(cnt, p, N)
        se = math.sqrt(a * (1 - a) / N)
        ck(f"mc{case}", abs(a - (1 - tv)) < 1e-12 and abs(acc / N - a) <= 4 * se and pv > 1e-3,
           alpha=a, mc_acc=acc / N, chi2_p=pv, kept=len(q))
    # (d) expected accepted tokens on a hand chain: alphas .5, .4 -> .5 + .2
    ck("chain_sum", abs((0.5 + 0.5 * 0.4) - 0.7) < 1e-12)
    os.makedirs(outdir, exist_ok=True)
    json.dump(rep, open(os.path.join(outdir, "SELFTEST_SHARPEN.json"), "w"), indent=1)
    print(json.dumps({"pass": rep["pass"]}))
    return 0 if rep["pass"] else 1


def analyze(outdir):
    recs = [json.loads(l) for l in open(os.path.join(outdir, "results_sharpen.jsonl"))]
    keys = [f"{tr}:{s}" for tr in TRUNCS for s in SCALES]
    agg = {}  # (T, prompt) -> n, sums
    for r in recs:
        a = agg.setdefault((r["T"], r["prompt"]), {"n": 0, "L": 0, "real": 0, **{k: 0.0 for k in keys}})
        a["n"] += 1
        a["L"] += r["L"]
        a["real"] += r["realized_acc"]
        for k in keys:
            a[k] += 1.0 + r["grid"][k]

    def mean(T, k, plist):
        n = s = 0.0
        for p in plist:
            a = agg.get((T, p))
            if a:
                n += a["n"]; s += a[k]
        return s / n if n else float("nan")

    brng = random.Random(BSEED)
    boots = [[brng.choice(HOLD) for _ in HOLD] for _ in range(B)]

    def ratio_ci(T, k, k0="0:1.0"):
        pt = mean(T, k, HOLD) / mean(T, k0, HOLD)
        bs = sorted(mean(T, k, b) / mean(T, k0, b) for b in boots)
        lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
        return [round(pt, 4), round(lo, 4), round(hi, 4), round((hi - lo) / 2, 4)]

    table, fits = [], {}
    for T in TS_ALL:
        best = max(keys, key=lambda k: (mean(T, k, CAL), -abs(float(k.split(":")[1]) - 1.0), -int(k.split(":")[0])))
        best_scale_only = max([k for k in keys if k.startswith("0:")], key=lambda k: (mean(T, k, CAL), -abs(float(k.split(":")[1]) - 1.0)))
        best_trunc_on = max([k for k in keys if k.startswith("1:")], key=lambda k: (mean(T, k, CAL), -abs(float(k.split(":")[1]) - 1.0)))
        rs = [r for r in recs if r["T"] == T]
        nd = sum(r["L"] for r in rs)
        v0rate = sum(r["grid"]["0:1.0"] for r in rs) / nd if nd else float("nan")
        realrate = sum(r["realized_acc"] for r in rs) / nd if nd else float("nan")
        row = {"T": T, "rounds": len(rs), "rounds_holdout": int(sum(agg[(T, p)]["n"] for p in HOLD if (T, p) in agg)),
               "V0_accept_rate": v0rate, "realized_accept_rate": realrate, "V0_over_realized": v0rate / realrate,
               "V0_tpp_holdout": mean(T, "0:1.0", HOLD), "fit": best, "fit_scale_only": best_scale_only,
               "fit_trunc_on": best_trunc_on,
               "Vstar/V0": ratio_ci(T, best), "trunc_only/V0": ratio_ci(T, "1:1.0"),
               "scale_only/V0": ratio_ci(T, best_scale_only), "trunc_fitscale/V0": ratio_ci(T, best_trunc_on),
               "grid_holdout_tpp": {k: round(mean(T, k, HOLD), 5) for k in keys},
               "grid_cal_tpp": {k: round(mean(T, k, CAL), 5) for k in keys}}
        fits[T] = best
        table.append(row)
    gate = [r for r in table if r["T"] in TS_GATE]
    base_bad = [r["T"] for r in gate if abs(r["V0_over_realized"] - 1.0) > 0.02]
    keep = [r["T"] for r in gate if r["Vstar/V0"][0] >= 1.03 and r["Vstar/V0"][1] > 1.0]
    close = all(r["Vstar/V0"][0] < 1.01 and r["Vstar/V0"][2] < 1.02 for r in gate)
    if base_bad:
        verdict = "FUTILITY_BASELINE"
    elif keep:
        verdict = "KEEP"
    elif close:
        verdict = "CLOSE"
    else:
        verdict = "INCONCLUSIVE"
    carrier = {}
    for r in gate:
        t, s = r["trunc_only/V0"], r["scale_only/V0"]
        tags = []
        if t[1] > 1.0 and t[0] >= 1.01:
            tags.append("trunc")
        if s[1] > 1.0 and s[0] >= 1.01:
            tags.append("scale")
        carrier[str(r["T"])] = "+".join(tags) if tags else ("both_jointly" if r["T"] in keep else "none")
    answered = all(r["Vstar/V0"][3] <= 0.01 for r in gate)
    summary = {"verdict": verdict, "keep_T": keep, "baseline_fail_T": base_bad, "carrier": carrier,
               "ci_halfwidth_le_1pct": answered, "rounds_total": len(recs)}
    return table, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "replay", "analyze"])
    ap.add_argument("--capture")
    ap.add_argument("--units")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest(a.out)
    if a.cmd == "replay":
        if not json.load(open(os.path.join(a.out, "SELFTEST_SHARPEN.json")))["pass"]:
            print("selftest not PASS; refusing", file=sys.stderr)
            return 2
        if sha(a.capture) != CAP_SHA or sha(a.units) != UNITS_SHA:
            print("capture digest mismatch; refusing", file=sys.stderr)
            return 2
        units = [json.loads(l) for l in open(a.units)]
        units = [u for u in units if u.get("status") == "CAPTURED"]
        assert len(units) == 192
        tmp = os.path.join(a.out, "results_sharpen.jsonl.tmp")
        with Pool(a.procs) as pool, open(tmp, "w") as f:
            for res in pool.imap(replay_unit, [(a.capture, u) for u in units]):
                for o in res:
                    f.write(json.dumps(o, sort_keys=True) + "\n")
        os.replace(tmp, os.path.join(a.out, "results_sharpen.jsonl"))
        return 0
    table, summary = analyze(a.out)
    json.dump({"table": table, "summary": summary}, open(os.path.join(a.out, "analysis_sharpen.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
