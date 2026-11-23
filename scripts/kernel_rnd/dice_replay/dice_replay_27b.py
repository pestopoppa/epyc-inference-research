#!/usr/bin/env python3
"""SPEC-DTREE-DICE step 2b, 27B DFlash2 half: offline replay on the dice.v1 DFlash2 tree capture (row
ak-dice-capture-27b-dflash2-gpu-20261123-r2). See PREDECLARATION_27B.md (sealed before any replay output).

Differences from the frozen A3B replay (dice_replay.py @ 03c100462, whose primitives are imported unchanged):
- V0 is the SERVED DFlash2 T>0 accept: rejection sampling with the drafter's own distributions
  q_j = softmax(lattice score(j, pred = ck[j-1]) / T) over the K = 16 lattice candidates (common/speculative.cpp DFlash2
  draft dists; common/sampling.cpp dists overload), accept chain_j w.p. min(1, p_j(c)/q_j(c)), at the served depth
  L = n_draft (spec-draft-n-max 7, n_draft < 7 on capped rounds).
- V1 recursive rejection WITHOUT replacement where the chain token is the FIRST draw from q (so V1 with no siblings ==
  V0 exactly) and m_j further siblings are drawn without replacement from q minus the chain (leaves).
- V2t: target-sample walk on the CAPTURED probability-greedy tree (pr ranks 1..W, full subtrees, exact analytic).
Usage: selftest --out DIR | replay --capture CAP --units RESULTS --out DIR [--procs N] [--draws R] | analyze --out DIR
"""
import argparse, json, math, os, random, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dice_replay import (gumbel, norm, blind_shape, select, walk, v2_analytic, draw_noise, dice_sorted,  # noqa: E402
                         chisq, sha)

CAP_SHA = "82d46a61a6d8f8b8088faece7935c5b8c10fd71c43bf3afa6db7941820d3a845"
UNITS_SHA = "38c571ae4969f75e6c06ba7b7cd663ae6c88d695de5585920430df739b7f4fd1"
WS = [4, 8, 12, 16]
RHOS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
TS = [0.1, 0.3, 1.0]
# break-even tokens/pass ratio vs the served chain (8 rows) = measured 27B tree round-cost ratio (implied_round_cost_ratio,
# ak-dtree-batch-27b-gpu-20261120-r4 RESULT.json) at W + 1 columns (anchor incl.), linear between T8/T10/T12/T16,
# extrapolated past T16 with the T12->T16 slope; W4 (5 columns < chain's 8) floored at 1.0.
BREAK_EVEN = {4: 1.0, 8: 1.1013, 12: 1.2557, 16: 1.3396}
BREAK_EVEN_ALT_HOSTGAP = {4: 1.0, 8: 1.079, 12: 1.079, 16: 1.141}  # the A3B-replay table (optimistic), reported only


def qdist(scores, toks, T):
    mx = max(scores)
    w = {}
    for s, t in zip(scores, toks):
        w[int(t)] = w.get(int(t), 0.0) + math.exp((s - mx) / T)
    return norm(w)


def rrs_q(p, q, chain_tok, m, trng, drng):
    """Recursive rejection without replacement; chain_tok is the first draw from q (None: no chain at this depth).
    Returns (accepted token or None, emitted token)."""
    r = dict(p)
    pool = dict(q)
    drafts = []
    if chain_tok is not None:
        drafts.append(chain_tok)
    rest = norm({c: v for c, v in q.items() if c != chain_tok})
    keys = sorted(((math.log(v) + gumbel(drng), c) for c, v in sorted(rest.items())), reverse=True)
    drafts += [c for _, c in keys[:m]]
    for s in drafts:
        qs_ = pool.get(s, 0.0)
        if qs_ <= 0:
            continue
        if trng.random() < min(1.0, r.get(s, 0.0) / qs_):
            return s, s
        ks = set(r) | set(pool)
        rr = {k: max(r.get(k, 0.0) - pool.get(k, 0.0), 0.0) for k in ks}
        if sum(rr.values()) <= 1e-15:
            break
        r = norm(rr)
        pool.pop(s, None)
        pool = norm(pool)
        if not pool:
            break
    x, acc = trng.random(), 0.0
    items = sorted(r.items())
    for c, v in items:
        acc += v
        if x < acc:
            return None, c
    return None, items[-1][0]


def v0_rej(chain, ps, qs, L):
    e, pr = 0.0, 1.0
    for j in range(L):
        qc = qs[j].get(chain[j], 0.0)
        pr *= min(1.0, ps[j].get(chain[j], 0.0) / qc) if qc > 0 else 0.0
        e += pr
    return e


def v1_mc(S, chain, ps, qs, trng, drng):
    acc = 0
    for j in range(len(chain)):
        if not S[j]:
            break
        has_chain = chain[j] in S[j]
        m = len(S[j]) - (1 if has_chain else 0)
        a, _ = rrs_q(ps[j], qs[j], chain[j] if has_chain else None, m, trng, drng)
        if a is None:
            break
        acc += 1
        if a != chain[j]:
            break
    return acc


def v2t_analytic(rec, W, L):
    """Exact expected accepted tokens of a target-sample walk on the captured pr-tree prefix of W draft nodes."""
    nodes, rows = rec["nodes"], rec["rows"]
    have = {n[5] for n in nodes}
    if any(k not in have for k in range(1, W + 1)):
        return None
    sel = {i for i, n in enumerate(nodes) if 1 <= n[5] <= W}
    pd = {}

    def prow(i):
        if i not in pd:
            pd[i] = norm({int(c): float(p) for c, p in rows[i][1]}) if rows[i] is not None else None
        return pd[i]

    reach = {0: 1.0}  # node index -> probability the walk reaches (accepts) it
    e = 0.0
    order = sorted(sel, key=lambda i: nodes[i][1])
    for i in order:
        par, depth, tok = nodes[i][0], nodes[i][1], nodes[i][3]
        if depth >= L or par not in reach:
            continue
        pp = prow(par)
        if pp is None:
            continue
        pr = reach[par] * pp.get(tok, 0.0)
        if pr > 0:
            reach[i] = reach.get(i, 0.0) + pr
            e += pr
    return e


def round_inputs(rec):
    L = min(int(rec["n_draft"]), int(rec["D"]), len(rec["chain"]))
    chain = [int(c) for c in rec["chain"][:L]]
    T = float(rec["T"])
    qs = []
    for j in range(L):
        pred = 0 if j == 0 else int(rec["ck"][j - 1])
        qs.append(qdist(rec["lat_sc"][j][pred], rec["lat_tok"][j], T))
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
    L = len(ps)
    return chain[:L], ps, qs[:L], L


def replay_round(rec, R):
    chain, ps, qs, L = round_inputs(rec)
    out = {"L": L, "n_draft": int(rec["n_draft"]), "capped": int(rec["n_draft"]) < int(rec["D"]), "W": {}}
    out["V0"] = v0_rej(chain, ps, qs, L)
    out["V0_mc"] = 0.0
    shapes = {W: blind_shape(chain, qs, W) for W in WS}
    acc = {W: {"V1": 0, "V2mc": 0, "V3": [0] * len(RHOS)} for W in WS}
    key = f"{rec['task']}:{rec['round']}:{rec['T']}"
    v0mc = 0
    for d in range(R):
        trng = random.Random(f"tgt:{key}:{d}")
        gs, tgt = draw_noise(ps, qs, L, trng)
        for W in WS:
            acc[W]["V2mc"] += walk(shapes[W], chain, tgt)
        for ri, rho in enumerate(RHOS):
            cs = dice_sorted(chain, qs, gs, rho, L)
            for W in WS:
                acc[W]["V3"][ri] += walk(select(cs, chain, W), chain, tgt)
        urng = random.Random(f"rrs:{key}:{d}")
        drng = random.Random(f"drf:{key}:{d}")
        for W in WS:
            acc[W]["V1"] += v1_mc(shapes[W], chain, ps, qs, urng, drng)
        v0mc += v1_mc([[c] for c in chain], chain, ps, qs, random.Random(f"v0:{key}:{d}"), random.Random(f"v0d:{key}:{d}"))
    out["V0_mc"] = v0mc / R
    for W in WS:
        out["W"][str(W)] = {"V1": acc[W]["V1"] / R, "V2": v2_analytic(shapes[W], chain, ps), "V2mc": acc[W]["V2mc"] / R,
                            "V3": [a / R for a in acc[W]["V3"]], "V2t": v2t_analytic(rec, W, L),
                            "shape": [len(s) for s in shapes[W]]}
    real = 0
    for j, t in enumerate(rec["acc"][:-1] if rec.get("acc") else []):
        if j < len(rec["chain"]) and t == rec["chain"][j]:
            real += 1
        else:
            break
    out["realized_acc"] = real
    return out


def replay_unit(args):
    cap, unit, R = args
    res = []
    with open(cap, "rb") as f:
        f.seek(unit["off0"])
        blob = f.read(unit["off1"] - unit["off0"])
    for line in blob.decode().splitlines():
        rec = json.loads(line)
        if rec.get("k") != "round":
            continue
        o = replay_round(rec, R)
        o.update({"unit": unit["unit_id"], "prompt": unit["prompt_idx"], "T": unit["T"], "task": rec["task"], "round": rec["round"]})
        res.append(o)
    return res


def selftest(outdir):
    rng = random.Random("selftest27b:20261123")
    rep = {"cases": [], "pass": True}
    N = 200000
    for case in range(3):
        V = 12
        p = norm({i: rng.random() ** 3 for i in range(V)})
        q = dict(p) if case == 2 else norm({i: rng.random() ** 2 for i in range(V)})
        cnt = {"V1m0": {}, "V1m3": {}}
        a0 = 0
        for n in range(N):
            trng = random.Random(f"st:{case}:{n}")
            drng = random.Random(f"sd:{case}:{n}")
            # the chain token is a draw from q (served DFlash2 samples its chain)
            x, c, ct = trng.random(), 0.0, None
            for t, v in sorted(q.items()):
                c += v
                if x < c:
                    ct = t
                    break
            ct = ct if ct is not None else max(q)
            a, e = rrs_q(p, q, ct, 0, trng, drng)
            cnt["V1m0"][e] = cnt["V1m0"].get(e, 0) + 1
            a0 += a is not None
            a, e = rrs_q(p, q, ct, 3, trng, drng)
            cnt["V1m3"][e] = cnt["V1m3"].get(e, 0) + 1
        cr = {"case": case, "q_eq_p": case == 2}
        for k, v in cnt.items():
            st, pv = chisq(v, p, N)
            cr[k] = {"chi2": st, "p": pv, "ok": pv > 1e-3}
            rep["pass"] &= pv > 1e-3
        exp0 = sum(min(p.get(t, 0.0), qv) for t, qv in q.items())  # 1 - TV
        se = math.sqrt(exp0 * (1 - exp0) / N) + 1e-12
        cr["V1m0_acc"], cr["V0_analytic_1mTV"] = a0 / N, exp0
        cr["V1m0_eq_V0_within_3se"] = abs(a0 / N - exp0) <= 3 * se
        rep["pass"] &= cr["V1m0_eq_V0_within_3se"]
        rep["cases"].append(cr)
    # V2t analytic on a hand tree: anchor -> {a, b}, a -> {c}; p0 = {a:.5, b:.3, z:.2}, pa = {c:.6, y:.4}
    rec = {"nodes": [[-1, -1, -1, -1, 0, 0, 0, 0], [0, 0, 0, 1, 7, 1, 1, 1], [0, 0, 1, 2, 6, 2, 2, 2], [1, 1, 0, 3, 7, 3, 3, 3]],
           "rows": [[[], [[1, .5], [2, .3], [9, .2]]], [[], [[3, .6], [8, .4]]], [[], [[5, 1.0]]], None]}
    v = v2t_analytic(rec, 3, 7)
    rep["v2t_hand"] = [v, 0.5 + 0.3 + 0.5 * 0.6]
    rep["pass"] &= abs(v - 1.1) < 1e-12
    os.makedirs(outdir, exist_ok=True)
    json.dump(rep, open(os.path.join(outdir, "SELFTEST_27B.json"), "w"), indent=1)
    print(json.dumps({"pass": rep["pass"]}))
    return 0 if rep["pass"] else 1


def analyze(outdir):
    recs = [json.loads(l) for l in open(os.path.join(outdir, "results_27b.jsonl"))]
    agg = {}
    for r in recs:
        for W, v in r["W"].items():
            k = (r["T"], int(W), r["prompt"])
            a = agg.setdefault(k, {"n": 0, "nt": 0, "V0": 0.0, "V1": 0.0, "V2": 0.0, "V2mc": 0.0, "V2t": 0.0,
                                   "V0t": 0.0, "V3": [0.0] * len(RHOS)})
            a["n"] += 1
            a["V0"] += 1.0 + r["V0"]
            for arm in ("V1", "V2", "V2mc"):
                a[arm] += 1.0 + v[arm]
            if v["V2t"] is not None:
                a["nt"] += 1
                a["V2t"] += 1.0 + v["V2t"]
                a["V0t"] += 1.0 + r["V0"]
            for i, x in enumerate(v["V3"]):
                a["V3"][i] += 1.0 + x
    cal, hold = range(0, 32), list(range(32, 64))
    fit = {(T, W): max(range(len(RHOS)), key=lambda i: (sum(agg[(T, W, p)]["V3"][i] for p in cal if (T, W, p) in agg), -i))
           for T in TS for W in WS}
    brng = random.Random(20261123)
    B = 2000
    boots = [[brng.choice(hold) for _ in hold] for _ in range(B)]

    def tot(T, W, arm, plist):
        n = s = 0.0
        for p in plist:
            a = agg.get((T, W, p))
            if not a:
                continue
            if arm in ("V2t", "V0t"):
                n += a["nt"]; s += a[arm]
            else:
                n += a["n"]; s += a["V3"][fit[(T, W)]] if arm == "V3" else a[arm]
        return s / n if n else float("nan")

    table = []
    for T in TS:
        for W in WS:
            row = {"T": T, "W": W, "rho_fit": RHOS[fit[(T, W)]],
                   "rounds_holdout": int(sum(agg[(T, W, p)]["n"] for p in hold if (T, W, p) in agg)),
                   "rounds_holdout_v2t": int(sum(agg[(T, W, p)]["nt"] for p in hold if (T, W, p) in agg))}
            for arm in ("V0", "V1", "V2", "V2mc", "V3", "V2t", "V0t"):
                row[arm] = tot(T, W, arm, hold)
            for num, den in (("V1", "V0"), ("V2", "V0"), ("V3", "V0"), ("V3", "V1"), ("V2t", "V0t")):
                pt = row[num] / row[den]
                bs = sorted(tot(T, W, num, b) / tot(T, W, den, b) for b in boots)
                lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
                row[f"{num}/{den}"] = [round(pt, 4), round(lo, 4), round(hi, 4), round((hi - lo) / 2, 4)]
            row["break_even"] = BREAK_EVEN[W]
            row["break_even_alt_hostgap"] = BREAK_EVEN_ALT_HOSTGAP[W]
            table.append(row)
    served = [r for r in table if r["T"] in (0.3, 1.0)]
    v1win = [[r["T"], r["W"]] for r in served if r["V1/V0"][0] >= r["break_even"] and r["V1/V0"][1] > 1.0]
    v1win_alt = [[r["T"], r["W"]] for r in served if r["V1/V0"][0] >= r["break_even_alt_hostgap"] and r["V1/V0"][1] > 1.0]
    v1par = [[r["T"], r["W"]] for r in served if r["V1/V0"][1] <= r["break_even"] <= r["V1/V0"][2]]
    verdict = "PUBLISH_V2_ROW" if v1win else ("PARITY" if v1par else "PROJECTED_LOSS")
    val = {}
    for T in TS:
        rs = [r for r in recs if r["T"] == T]
        nd = sum(r["L"] for r in rs)
        val[str(T)] = {"rounds": len(rs), "drafted": nd, "V0_accept_rate": sum(r["V0"] for r in rs) / nd,
                       "V0mc_accept_rate": sum(r["V0_mc"] for r in rs) / nd,
                       "realized_accept_rate": sum(r["realized_acc"] for r in rs) / nd,
                       "capped_rounds": sum(r["capped"] for r in rs)}
        val[str(T)]["V0_over_realized"] = val[str(T)]["V0_accept_rate"] / val[str(T)]["realized_accept_rate"]
    measured = {"0.3": [0.47, 0.4647], "1.0": [0.43, 0.4374]}  # ak-dtree-sampled A arm; ak-specdist-trunc-ab C19 A arm L256
    for T, m in measured.items():
        val[T]["measured_served_A"] = m
        val[T]["V0_over_measured"] = [val[T]["V0_accept_rate"] / x for x in m]
    summary = {"verdict": verdict, "v1_cells_clearing_break_even": v1win, "v1_cells_clearing_alt_hostgap": v1win_alt,
               "v1_parity_cells": v1par, "v0_validation": val, "rounds_total": len(recs)}
    return table, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "replay", "analyze"])
    ap.add_argument("--capture")
    ap.add_argument("--units")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--draws", type=int, default=64)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest(a.out)
    if a.cmd == "replay":
        if not json.load(open(os.path.join(a.out, "SELFTEST_27B.json")))["pass"]:
            print("selftest not PASS; refusing", file=sys.stderr)
            return 2
        if sha(a.capture) != CAP_SHA or sha(a.units) != UNITS_SHA:
            print("capture digest mismatch; refusing", file=sys.stderr)
            return 2
        units = [json.loads(l) for l in open(a.units)]
        units = [u for u in units if u.get("status") == "CAPTURED"]
        assert len(units) == 192
        with Pool(a.procs) as pool, open(os.path.join(a.out, "results_27b.jsonl"), "w") as f:
            for res in pool.imap(replay_unit, [(a.capture, u, a.draws) for u in units]):
                for o in res:
                    f.write(json.dumps(o, sort_keys=True) + "\n")
        return 0
    table, summary = analyze(a.out)
    json.dump({"table": table, "summary": summary}, open(os.path.join(a.out, "analysis_27b.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
