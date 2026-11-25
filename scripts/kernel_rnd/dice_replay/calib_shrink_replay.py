#!/usr/bin/env python3
"""DTREE-CALIB-SHRINK-REPLAY (TreeSpark, intake-2975): does an edge-acceptance Platt calibration (a stop threshold for tree
growth) and a load-adaptive shrink to the chain beat the current tree width rule on 27B DFlash2? Offline replay of the dice.v1
capture of row ak-dice-capture-27b-dflash2-gpu-20261123-r2 (the only per-edge data that exists: the C21 official tree row
ak-c21-tree-official-gpu-20261124 logs per-request aggregates, no per-edge dump). See PREDECLARATION_DTREE_CALIB_SHRINK.md
(sealed before any replay output).

Per round (target rows only where the capture holds them, as in headroom_replay.py; tokens per pass = 1 + exact expected
accepted tokens under V1t, the lossless target-sample walk of a deterministic tree):
  chain     the served chain, V0 (dice_replay_27b.v0_rej), cost 1.0.
  fixed10/16  the current builder (dice_replay.blind_shape spine with sibling leaves) at W = 10 / 16.
  auto      PROXY of the C21 AUTO width rule (its rule text was not found in the box or handoffs): best-first on the drafter's
            RAW q with the value stop rule below. Isolates what calibration adds.
  platt     same, on p_hat = sigmoid(a*logit(q) + b), (a, b) fitted per T on calibration prompts 0-31 from greedy-match edge
            labels (edge label 1 iff child == argmax of the target row at its parent, edges whose ancestors all matched).
  shrink    platt grown with the load-priced stop rule, then the chain wins whenever the tree's predicted tokens per cost
            does not beat the chain's (the price ladder below).
Value stop rule (no tuned threshold): adding a node with predicted reach r lifts tokens/cost iff
r > R * (cost(n+1) - cost(n)), R = current predicted (1 + sum reach) / cost(n). The first node that fails ends growth; W <= 16.
Cost: f(W) = measured 27B round-cost ratio at W draft nodes (dice_replay_27b.BREAK_EVEN: 4 -> 1.0, 8 -> 1.1013, 12 -> 1.2557,
16 -> 1.3396, linear between, 1.0 below 4); load price ladder cost_m(W) = 1 + m * (f(W) - 1), m in {1, 2, 3} (m = 1 is the
measured np1 table; m = 2, 3 are SEALED HYPOTHESES for np2 / np4 verify, not measurements).
Usage: selftest --out DIR [--capture CAP --units UNITS] | fit --capture CAP --units UNITS --out DIR
       | replay --capture CAP --units UNITS --out DIR [--procs N] | analyze --out DIR
"""
import argparse, heapq, json, math, os, random, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dice_replay import norm, blind_shape, sha  # noqa: E402
from dice_replay_27b import qdist, round_inputs, v0_rej, BREAK_EVEN  # noqa: E402
from headroom_replay import eval_tree, eval_path_tree, row_paths, lattice, build_ceiling  # noqa: E402

CAP_SHA = "82d46a61a6d8f8b8088faece7935c5b8c10fd71c43bf3afa6db7941820d3a845"
UNITS_SHA = "38c571ae4969f75e6c06ba7b7cd663ae6c88d695de5585920430df739b7f4fd1"
TS_GATE = [0.3, 1.0]
TS_ALL = [0.1, 0.3, 1.0]
CAL, HOLD = list(range(0, 32)), list(range(32, 64))
MS = [1, 2, 3]
NMAX = 16
KCHILD = 4
B, BSEED = 2000, 20261126
WIN_PT, WIN_LO = 1.01, 1.0
PAPER_AB = (0.67, -0.46)
BLIND = ["fixed10", "fixed16", "auto", "platt"]


# ---------------------------------------------------------------- cost
def f_cost(w):
    pts = sorted(BREAK_EVEN.items())
    if w <= pts[0][0]:
        return 1.0
    for (w0, c0), (w1, c1) in zip(pts, pts[1:]):
        if w <= w1:
            return c0 + (c1 - c0) * (w - w0) / (w1 - w0)
    (w0, c0), (w1, c1) = pts[-2], pts[-1]
    return c1 + (c1 - c0) * (w - w1) / (w1 - w0)


def cost_m(w, m):
    return 1.0 + m * (f_cost(w) - 1.0)


def logit(q):
    q = min(max(q, 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


def phat(q, ab):
    z = ab[0] * logit(q) + ab[1]
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------- Platt fit (Newton / IRLS, 2 parameters)
def _sig(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def fit_platt(xs, ys, iters=100, lam=1e-3):
    """Damped Newton on the logistic loss with a tiny L2 pull toward (a, b) = (1, 0) so separable or tiny samples stay finite."""
    a, b = 1.0, 0.0
    for _ in range(iters):
        g0, g1 = lam * (a - 1.0), lam * b
        h00 = h11 = lam
        h01 = 0.0
        for x, y in zip(xs, ys):
            p = _sig(a * x + b)
            w = p * (1 - p) + 1e-9
            g0 += (p - y) * x
            g1 += (p - y)
            h00 += w * x * x
            h01 += w * x
            h11 += w
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-18:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        sc = min(1.0, 2.0 / max(abs(da), abs(db), 1e-12))
        a -= sc * da
        b -= sc * db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


# ---------------------------------------------------------------- edges for the fit
def round_edges(rec):
    """(logit q, greedy-match label) for the top-KCHILD drafter candidates under every captured node whose ancestors all
    matched the target argmax."""
    T = float(rec["T"])
    nodes, rows = rec["nodes"], rec["rows"]
    q_at, _ = lattice(rec, T)
    L = min(int(rec["n_draft"]), int(rec["D"]))
    ok = {}
    amax = {}
    out = []
    for i, n in enumerate(nodes):
        row = norm({int(c): float(p) for c, p in rows[i][1]}) if rows[i] is not None else None
        amax[i] = max(row.items(), key=lambda kv: (kv[1], -kv[0]))[0] if row else None
        if i == 0:
            ok[i] = True
        else:
            par = n[0]
            ok[i] = bool(ok.get(par)) and amax.get(par) is not None and int(n[3]) == amax[par]
        d = 0 if i == 0 else int(n[1]) + 1
        if not ok[i] or amax[i] is None or d >= L:
            continue
        q = q_at(d, 0 if d == 0 else int(n[2]))
        for t, qv in sorted(q.items(), key=lambda kv: (-kv[1], kv[0]))[:KCHILD]:
            out.append((logit(qv), 1 if t == amax[i] else 0))
    return out


def edges_unit(args):
    cap, unit = args
    with open(cap, "rb") as f:
        f.seek(unit["off0"])
        blob = f.read(unit["off1"] - unit["off0"])
    res = []
    for line in blob.decode().splitlines():
        rec = json.loads(line)
        if rec.get("k") == "round":
            res.extend(round_edges(rec))
    return float(unit["T"]), res


def fit_all(capture, units, procs=4):
    cal = [u for u in units if int(u["prompt_idx"]) in CAL]
    by_T = {T: ([], []) for T in TS_ALL}
    with Pool(procs) as pool:
        for T, es in pool.imap(edges_unit, [(capture, u) for u in cal]):
            for x, y in es:
                by_T[T][0].append(x)
                by_T[T][1].append(y)
    fit = {}
    for T, (xs, ys) in by_T.items():
        a, b = fit_platt(xs, ys)
        fit[str(T)] = {"a": a, "b": b, "n_edges": len(xs), "pos_rate": (sum(ys) / len(ys)) if ys else None,
                       "ok": bool(xs) and math.isfinite(a) and math.isfinite(b) and a > 0}
    return fit


# ---------------------------------------------------------------- tree growth
def grow(rows, q_at, first_idx, L, score_ab, m, k, stop=True, nmax=NMAX):
    """Best-first on cumulative edge score (raw q when score_ab is None, else p_hat); value stop rule priced at load m.
    Returns (selected [(path, kidx)], predicted sum of reach)."""
    sel, heap, cnt, psum = [], [], 0, 0.0

    def push(par, par_k, d, cum):
        nonlocal cnt
        if rows.get(par) is None or d >= L:
            return
        q = q_at(d, 0 if d == 0 else par_k)
        for t, qv in sorted(q.items(), key=lambda kv: (-kv[1], kv[0]))[:k]:
            if qv > 0:
                e = qv if score_ab is None else phat(qv, score_ab)
                heapq.heappush(heap, (-(cum * e), d, cnt, par, t, cum * e))
                cnt += 1

    push((), 0, 0, 1.0)
    while heap and len(sel) < nmax:
        _, d, _, par, t, r = heapq.heappop(heap)
        if stop:
            n = len(sel)
            R = (1.0 + psum) / cost_m(n, m)
            if r <= R * (cost_m(n + 1, m) - cost_m(n, m)):
                break
        path = par + (t,)
        kidx = first_idx[d].get(t, 0)
        sel.append((path, kidx))
        psum += r
        push(path, kidx, d + 1, r)
    return sel, psum


def tree_tokens(sel, rows, q_at, L):
    kidx = {(): 0}
    for p, k in sel:
        kidx[p] = k

    def qpar(path):
        d = len(path)
        return q_at(d, 0 if d == 0 else kidx.get(path, 0)) if d < L else {}
    t, _ = eval_path_tree([p for p, _ in sel], rows, qpar)
    return 1.0 + t


def cur_tokens(chain, ps, qs, L, W):
    S = blind_shape(chain, qs, W)
    nodes, par = [], -1
    for j in range(L):
        if not S[j]:
            break
        order = ([chain[j]] if chain[j] in S[j] else []) + [c for c in S[j] if c != chain[j]]
        ids = {}
        for c in order:
            nodes.append((par, c))
            ids[c] = len(nodes) - 1
        if chain[j] in ids:
            par = ids[chain[j]]
        else:
            break
    spine = {}
    pp = -1
    for j in range(L):
        spine[pp] = j
        hit = [i for i, (pa, c) in enumerate(nodes) if pa == pp and c == chain[j]]
        if not hit:
            break
        pp = hit[0]
    t, _ = eval_tree(nodes, lambda pid: ps[spine[pid]] if pid in spine else {}, lambda pid: qs[spine[pid]] if pid in spine else {})
    return 1.0 + t, len(nodes)


def replay_round(rec, ab):
    chain, ps, qs, L = round_inputs(rec)
    T = float(rec["T"])
    rows = row_paths(rec)
    q_at, first_idx = lattice(rec, T)
    out = {"L": L, "V0": v0_rej(chain, ps, qs, L)}
    real = 0
    for j, t in enumerate(rec["acc"][:-1] if rec.get("acc") else []):
        if j < len(rec["chain"]) and t == rec["chain"][j]:
            real += 1
        else:
            break
    out["realized_acc"] = real
    arms = {}
    for W in (10, 16):
        tk, n = cur_tokens(chain, ps, qs, L, W)
        arms[f"fixed{W}"] = [tk, n]
    sel, _ = grow(rows, q_at, first_idx, L, None, 1, KCHILD)
    arms["auto"] = [tree_tokens(sel, rows, q_at, L), len(sel)]
    sel, _ = grow(rows, q_at, first_idx, L, ab, 1, KCHILD)
    arms["platt"] = [tree_tokens(sel, rows, q_at, L), len(sel)]
    shrink = {}
    for m in MS:
        sel, psum = grow(rows, q_at, first_idx, L, ab, m, KCHILD)
        shrink[str(m)] = [tree_tokens(sel, rows, q_at, L), len(sel), 1.0 + psum]
    arms["shrink"] = shrink
    # predicted chain value: the single best path (k = 1), no stop rule, depth <= 7
    csel, cpsum = grow(rows, q_at, first_idx, L, ab, 1, 1, stop=False, nmax=7)
    out["pchain"] = 1.0 + cpsum
    out["arms"] = arms
    return out


def replay_unit(args):
    cap, unit, fit = args
    ab = (fit[str(float(unit["T"]))]["a"], fit[str(float(unit["T"]))]["b"])
    with open(cap, "rb") as f:
        f.seek(unit["off0"])
        blob = f.read(unit["off1"] - unit["off0"])
    res = []
    for line in blob.decode().splitlines():
        rec = json.loads(line)
        if rec.get("k") != "round":
            continue
        o = replay_round(rec, ab)
        o.update({"unit": unit["unit_id"], "prompt": unit["prompt_idx"], "T": unit["T"], "task": rec["task"], "round": rec["round"]})
        res.append(o)
    return res


# ---------------------------------------------------------------- analysis
def round_metric(r, arm, m):
    """(tokens, cost) of one round for an arm under price m."""
    if arm == "chain":
        return 1.0 + r["V0"], 1.0
    if arm in ("fixed10", "fixed16", "auto", "platt"):
        tk, n = r["arms"][arm]
        return tk, cost_m(n, m)
    tk, n, pred = r["arms"]["shrink"][str(m)]
    if pred / cost_m(n, m) >= r["pchain"] / 1.0:
        return tk, cost_m(n, m)
    return 1.0 + r["V0"], 1.0


def analyze(outdir):
    recs = [json.loads(l) for l in open(os.path.join(outdir, "results_calib.jsonl"))]
    arms = ["chain"] + BLIND + ["shrink"]
    agg = {}
    for r in recs:
        a = agg.setdefault((r["T"], r["prompt"]), {"n": 0, "L": 0, "real": 0, "V0": 0.0})
        a["n"] += 1
        a["L"] += r["L"]
        a["real"] += r["realized_acc"]
        a["V0"] += r["V0"]
        for m in MS:
            for arm in arms:
                tk, c = round_metric(r, arm, m)
                s = a.setdefault((arm, m), [0.0, 0.0])
                s[0] += tk
                s[1] += c
    brng = random.Random(BSEED)
    boots = [[brng.choice(HOLD) for _ in HOLD] for _ in range(B)]

    def msum(T, plist, arm, m):
        tk = c = 0.0
        for p in plist:
            a = agg.get((T, p))
            if a:
                tk += a[(arm, m)][0]
                c += a[(arm, m)][1]
        return tk / c if c else float("nan")

    def ratio(T, m, num, den):
        pt = msum(T, HOLD, num, m) / msum(T, HOLD, den, m)
        bs = sorted(msum(T, b, num, m) / msum(T, b, den, m) for b in boots)
        lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
        return [round(pt, 4), round(lo, 4), round(hi, 4)]

    table = []
    for T in TS_ALL:
        for m in MS:
            best_blind = max(BLIND, key=lambda arm: msum(T, CAL, arm, m))   # frozen on calibration prompts
            row = {"T": T, "m": m, "best_blind": best_blind,
                   "tokens_per_cost": {arm: round(msum(T, HOLD, arm, m), 4) for arm in arms},
                   "platt/auto": ratio(T, m, "platt", "auto"),
                   "shrink/best_blind": ratio(T, m, "shrink", best_blind),
                   "shrink/chain": ratio(T, m, "shrink", "chain"),
                   "platt/chain": ratio(T, m, "platt", "chain"),
                   "auto/chain": ratio(T, m, "auto", "chain"),
                   "fixed16/chain": ratio(T, m, "fixed16", "chain")}
            table.append(row)
    fit = json.load(open(os.path.join(outdir, "fit.json")))
    fails, val = [], {}
    for T in TS_ALL:
        rs = [r for r in recs if r["T"] == T]
        nd = sum(r["L"] for r in rs)
        rate, real = sum(r["V0"] for r in rs) / nd, sum(r["realized_acc"] for r in rs) / nd
        val[str(T)] = {"V0_accept_rate": rate, "realized_accept_rate": real, "V0_over_realized": rate / real}
        if T in TS_GATE and abs(rate / real - 1.0) > 0.02:
            fails.append(T)
    fit_bad = [T for T in TS_GATE if not fit[str(T)]["ok"]]
    gate = [r for r in table if r["T"] in TS_GATE]
    calib = [[r["T"], r["m"]] for r in gate if r["m"] == 1 and r["platt/auto"][0] >= WIN_PT and r["platt/auto"][1] > WIN_LO]
    shr = [[r["T"], r["m"]] for r in gate if r["m"] >= 2 and r["shrink/best_blind"][0] >= WIN_PT and r["shrink/best_blind"][1] > WIN_LO]
    closed = all(r["platt/auto"][2] < WIN_PT for r in gate if r["m"] == 1) and \
        all(r["shrink/best_blind"][2] < WIN_PT for r in gate if r["m"] >= 2)
    if fails:
        verdict = "FUTILITY_BASELINE"
    elif fit_bad:
        verdict = "FUTILITY_FIT"
    elif calib and shr:
        verdict = "FILE_CALIB_AND_SHRINK"
    elif calib:
        verdict = "FILE_CALIB"
    elif shr:
        verdict = "FILE_SHRINK"
    elif closed:
        verdict = "CLOSE"
    else:
        verdict = "INCONCLUSIVE"
    summary = {"verdict": verdict, "baseline_fail_T": fails, "fit_bad_T": fit_bad, "v0_validation": val, "platt_fit": fit,
               "paper_ab": PAPER_AB, "calib_cells": calib, "shrink_cells": shr, "rounds_total": len(recs)}
    return table, summary


# ---------------------------------------------------------------- selftest
def selftest(outdir, capture=None, units=None):
    rep = {"cases": {}, "pass": True}

    def chk(name, ok, detail=None):
        rep["cases"][name] = {"ok": bool(ok), "detail": detail}
        rep["pass"] &= bool(ok)

    chk("cost_table", abs(f_cost(8) - 1.1013) < 1e-9 and f_cost(3) == 1.0 and abs(f_cost(10) - (1.1013 + 1.2557) / 2) < 1e-9)
    chk("cost_ladder", abs(cost_m(16, 3) - (1 + 3 * 0.3396)) < 1e-9 and cost_m(4, 3) == 1.0)
    chk("phat_hand", abs(phat(0.5, (0.67, -0.46)) - 1 / (1 + math.exp(0.46))) < 1e-12)
    rng = random.Random(7)
    xs, ys = [], []
    for _ in range(20000):
        x = rng.uniform(-5, 3)
        xs.append(x)
        ys.append(1 if rng.random() < 1 / (1 + math.exp(-(0.67 * x - 0.46))) else 0)
    a, b = fit_platt(xs, ys)
    chk("platt_recovers_params", abs(a - 0.67) < 0.05 and abs(b + 0.46) < 0.08, [a, b])
    q_at = lambda d, pred: {1: .6, 2: .3, 3: .1} if d == 0 else {5: .5, 6: .5}  # noqa: E731
    rows = {(): {1: .6, 2: .4}, (1,): {5: .5, 6: .5}, (2,): {5: .5, 6: .5}}
    fi = [{1: 0, 2: 1, 3: 2}, {5: 0, 6: 1}]
    sel, ps_ = grow(rows, q_at, fi, 3, None, 1, 2, stop=False, nmax=6)
    chk("grow_nested_eligible", len(sel) == 6 and all(rows.get(p[:-1]) is not None for p, _ in sel), [p for p, _ in sel])
    sel_all, _ = grow(rows, q_at, fi, 3, None, 1, 2, stop=False, nmax=16)
    sel_stop, _ = grow(rows, q_at, fi, 3, None, 3, 2, stop=True, nmax=16)
    chk("stop_rule_shrinks_with_price", len(sel_stop) <= len(sel_all) and len(sel_stop) < 16, [len(sel_stop), len(sel_all)])
    chk("price_monotone", len(grow(rows, q_at, fi, 3, None, 3, 2)[0]) <= len(grow(rows, q_at, fi, 3, None, 1, 2)[0]))
    if capture and units and os.path.exists(capture):
        us = [json.loads(l) for l in open(units)]
        us = [u for u in us if u.get("status") == "CAPTURED"]
        fit = {str(T): {"a": PAPER_AB[0], "b": PAPER_AB[1]} for T in TS_ALL}
        res = replay_unit((capture, us[0], fit))
        edges = edges_unit((capture, us[0]))[1]
        okc = okn = okf = True
        for r in res:
            for arm in BLIND:
                okn &= 0 <= r["arms"][arm][1] <= 16
            for m in MS:
                sh = r["arms"]["shrink"][str(m)]
                okn &= 0 <= sh[1] <= 16
                tk, c = round_metric(r, "shrink", m)
                okf &= tk >= 1.0 and c >= 1.0
            okc &= r["arms"]["platt"][0] >= 1.0 and r["pchain"] >= 1.0
        chk("real_unit0_rounds", len(res) > 0 and okn and okf and okc, {"rounds": len(res)})
        chk("real_unit0_edges", len(edges) > 0 and all(y in (0, 1) for _, y in edges), {"edges": len(edges)})
        # a unit-0 fit must be finite
        a0, b0 = fit_platt([x for x, _ in edges], [y for _, y in edges])
        chk("real_unit0_fit_finite", math.isfinite(a0) and math.isfinite(b0), [a0, b0])
    os.makedirs(outdir, exist_ok=True)
    json.dump(rep, open(os.path.join(outdir, "SELFTEST_CALIB.json"), "w"), indent=1)
    print(json.dumps({"pass": rep["pass"]}))
    return 0 if rep["pass"] else 1


def load_units(path):
    us = [json.loads(l) for l in open(path)]
    return [u for u in us if u.get("status") == "CAPTURED"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "fit", "replay", "analyze"])
    ap.add_argument("--capture")
    ap.add_argument("--units")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest(a.out, a.capture, a.units)
    if a.cmd in ("fit", "replay"):
        st = os.path.join(a.out, "SELFTEST_CALIB.json")
        if not os.path.exists(st) or not json.load(open(st))["pass"]:
            print("selftest not PASS; refusing", file=sys.stderr)
            return 2
        if sha(a.capture) != CAP_SHA or sha(a.units) != UNITS_SHA:
            print("capture digest mismatch; refusing", file=sys.stderr)
            return 2
    if a.cmd == "fit":
        fit = fit_all(a.capture, load_units(a.units), a.procs)
        json.dump(fit, open(os.path.join(a.out, "fit.json"), "w"), indent=1)
        print(json.dumps(fit))
        return 0
    if a.cmd == "replay":
        fit = json.load(open(os.path.join(a.out, "fit.json")))
        units = load_units(a.units)
        tmp = os.path.join(a.out, "results_calib.jsonl.tmp")
        with Pool(a.procs) as pool, open(tmp, "w") as f:
            for res in pool.imap(replay_unit, [(a.capture, u, fit) for u in units]):
                for o in res:
                    f.write(json.dumps(o, sort_keys=True) + "\n")
        os.replace(tmp, os.path.join(a.out, "results_calib.jsonl"))
        return 0
    table, summary = analyze(a.out)
    json.dump({"table": table, "summary": summary}, open(os.path.join(a.out, "analysis_calib.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
