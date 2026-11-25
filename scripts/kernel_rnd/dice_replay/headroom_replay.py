#!/usr/bin/env python3
"""DTREE-HEADROOM-REPLAY: how much tree headroom is left on 27B DFlash2? Offline CPU replay of the dice.v1 capture of row
ak-dice-capture-27b-dflash2-gpu-20261123-r2. See PREDECLARATION_DTREE_HEADROOM.md (sealed before any replay output).

Per round (sampled targets, T0.3 and T1.0 gating, T0.1 reported), tokens per verify pass = 1 + E[accepted draft tokens]:
  V0       the served chain: realized sampled chain, accept min(1, p/q) per depth (dice_replay_27b.v0_rej, exact).
  ceiling  speed-of-light: the best N-node tree chosen with the TARGET's own distributions (oracle drafter), N in {8,12,16}.
           Nodes are taken in order of cumulative target probability (reach); a node can be expanded only where the capture
           holds the target row for its token path (anchor, the whole realized chain, and the captured 32-column tree).
           Optimal for sum-of-reach because reach is non-increasing along a path. Bound is over CAPTURED rows (stated).
  pando    Pando-style best-first tree: children per node = top-k by the drafter's own q (DFlash2 lattice, conditioned on
           the parent's lattice index), nodes taken in order of cumulative draft probability, pruned to N nodes, k fitted
           on prompts 0-31 per (T, N). Same expandability rule as the ceiling (the tree must be evaluable on captured rows).
  cur      the current builder: dice_replay.blind_shape (spine with sibling leaves, probability-greedy) at W = N.
Two verifies are reported for every tree:
  V1t  (GATING)    lossless verify of a deterministic tree: the target's sample walks the tree, a node's child is accepted
                   iff the sample equals it, so reach(child) = reach(parent) * p_parent(child); exact, analytic.
  V1q  (secondary) ordered recursive rejection with the drafter's q pool (dice_replay_27b.rrs_q convention, exact analytic).
                   With deterministic children it is NOT lossless (it can exceed the oracle ceiling); reported only.
Usage: selftest --out DIR [--capture CAP --units UNITS] | replay --capture CAP --units UNITS --out DIR [--procs N]
       | analyze --out DIR
"""
import argparse, heapq, json, math, os, random, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dice_replay import norm, blind_shape, sha  # noqa: E402
from dice_replay_27b import qdist, round_inputs, v0_rej, rrs_q  # noqa: E402

CAP_SHA = "82d46a61a6d8f8b8088faece7935c5b8c10fd71c43bf3afa6db7941820d3a845"
UNITS_SHA = "38c571ae4969f75e6c06ba7b7cd663ae6c88d695de5585920430df739b7f4fd1"
NS = [8, 12, 16]
KS = [1, 2, 3, 4, 6, 8]
TS_GATE = [0.3, 1.0]
TS_ALL = [0.1, 0.3, 1.0]
CAL, HOLD = list(range(0, 32)), list(range(32, 64))
B, BSEED = 2000, 20261125
CEIL_CLOSE, PANDO_CLOSE_UP, PANDO_FILE_PT = 1.10, 1.02, 1.03


# ---------------------------------------------------------------- tree evaluation (exact, analytic)
def eval_tree(nodes, p_of, q_of):
    """nodes: list of (parent_id, token); parent_id -1 = anchor; parents precede children. p_of(pid) / q_of(pid): the
    target row / drafter pool at node pid (-1 anchor). Returns (V1t, V1q) = expected accepted draft tokens."""
    kids = {}
    for i, (par, tok) in enumerate(nodes):
        kids.setdefault(par, []).append(i)
    reach_t = {-1: 1.0}
    reach_q = {-1: 1.0}
    et = eq = 0.0
    stack = [-1]
    while stack:
        par = stack.pop()
        ch = kids.get(par)
        if not ch:
            continue
        p = p_of(par)
        q = q_of(par)
        # V1t: target-sample walk
        for i in ch:
            pt = p.get(nodes[i][1], 0.0) if p else 0.0
            r = reach_t[par] * pt
            reach_t[i] = r
            et += r
        # V1q: ordered recursive rejection, children by q descending (ties by token), pool = q
        order = sorted(ch, key=lambda i: (-q.get(nodes[i][1], 0.0), nodes[i][1])) if q else list(ch)
        rem = reach_q[par]
        r_dist = dict(p) if p else {}
        pool = dict(q) if q else {}
        for i in order:
            s = nodes[i][1]
            qs_ = pool.get(s, 0.0)
            if qs_ <= 0 or not r_dist:
                reach_q[i] = 0.0
                continue
            a = min(1.0, r_dist.get(s, 0.0) / qs_)
            reach_q[i] = rem * a
            eq += rem * a
            rem *= (1.0 - a)
            ks = set(r_dist) | set(pool)
            rr = {k: max(r_dist.get(k, 0.0) - pool.get(k, 0.0), 0.0) for k in ks}
            if sum(rr.values()) <= 1e-15:
                r_dist = {}
            else:
                r_dist = norm(rr)
            pool.pop(s, None)
            pool = norm(pool)
            if not pool:
                r_dist = {}
        for i in ch:
            stack.append(i)
    return et, eq


# ---------------------------------------------------------------- captured rows by token path
def row_paths(rec):
    """token-path tuple -> normalized target row (post-sampler) for every captured node that has one."""
    nodes, rows = rec["nodes"], rec["rows"]
    path = {}
    out = {}
    for i, n in enumerate(nodes):
        par = n[0]
        path[i] = () if i == 0 else path.get(par, ()) + (int(n[3]),)
        if rows[i] is not None:
            out[path[i]] = norm({int(c): float(p) for c, p in rows[i][1]})
    return out


def lattice(rec, T):
    """per depth: (tokens list, cache of qdist by predecessor index)."""
    toks = rec["lat_tok"]
    sc = rec["lat_sc"]
    cache = {}

    def q_at(depth, pred):
        k = (depth, pred)
        if k not in cache:
            cache[k] = qdist(sc[depth][pred], toks[depth], T)
        return cache[k]

    first_idx = []
    for d in range(len(toks)):
        fi = {}
        for k, t in enumerate(toks[d]):
            fi.setdefault(int(t), k)
        first_idx.append(fi)
    return q_at, first_idx


# ---------------------------------------------------------------- builders
def build_ceiling(rows, L, nmax):
    """Oracle: best-first by cumulative TARGET probability. Returns the selected nodes (path, reach) in order."""
    sel = []
    heap = []
    cnt = 0
    root = rows.get(())
    if root:
        for t, p in root.items():
            if p > 0:
                heapq.heappush(heap, (-p, 0, t, cnt, ()))
                cnt += 1
    while heap and len(sel) < nmax:
        negr, d, t, _, par = heapq.heappop(heap)
        path = par + (t,)
        sel.append((path, -negr))
        row = rows.get(path)
        if row and d + 1 < L:
            for t2, p2 in row.items():
                if p2 > 0:
                    heapq.heappush(heap, (negr * p2, d + 1, t2, cnt, path))
                    cnt += 1
    return sel


def build_pando(rows, q_at, first_idx, L, k, nmax):
    """Best-first by cumulative DRAFT probability, top-k children per node by q, nodes eligible only under a captured
    parent row. Returns list of (path, kidx_at_depth, parent_path) in selection order."""
    sel = []
    heap = []
    cnt = 0

    def push(par, par_k, d, logq):
        nonlocal cnt
        if rows.get(par) is None or d >= L:
            return
        q = q_at(d, 0 if d == 0 else par_k)
        top = sorted(q.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        for t, qv in top:
            if qv > 0:
                heapq.heappush(heap, (-(logq + math.log(qv)), d, cnt, par, t))
                cnt += 1

    push((), 0, 0, 0.0)
    while heap and len(sel) < nmax:
        nl, d, _, par, t = heapq.heappop(heap)
        path = par + (t,)
        kidx = first_idx[d].get(t, 0)
        sel.append((path, kidx, par))
        push(path, kidx, d + 1, -nl)
    return sel


def eval_path_tree(sel_paths, rows, q_of_parent):
    """sel_paths: ordered list of node paths (parents first). Returns (V1t, V1q)."""
    idx = {(): -1}
    nodes = []
    for path in sel_paths:
        idx[path] = len(nodes)
        nodes.append((idx[path[:-1]], path[-1]))
    inv = {v: k for k, v in idx.items()}
    return eval_tree(nodes, lambda pid: rows.get(inv[pid]), lambda pid: q_of_parent(inv[pid]))


def replay_round(rec):
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
    # ceiling
    cs = build_ceiling(rows, L, max(NS))
    out["ceil"] = {str(n): sum(r for _, r in cs[:n]) for n in NS}
    out["ceil_nodes"] = len(cs)
    # pando; q pool at a parent path = drafter's distribution at that depth given the parent's lattice index
    pando = {}
    for k in KS:
        sel = build_pando(rows, q_at, first_idx, L, k, max(NS))
        kidx = {(): 0}
        for path, ki, _ in sel:
            kidx[path] = ki

        def qpar(path, kidx=kidx):
            d = len(path)
            return q_at(d, 0 if d == 0 else kidx.get(path, 0)) if d < L else {}
        pando[str(k)] = {}
        for n in NS:
            t, q = eval_path_tree([s[0] for s in sel[:n]], rows, qpar)
            pando[str(k)][str(n)] = [t, q, len(sel[:n])]
    out["pando"] = pando
    # current builder (spine with sibling leaves) at W = N, on the chain rows ps / pools qs
    cur = {}
    for W in NS:
        S = blind_shape(chain, qs, W)
        nodes, pid_of_depth = [], {}
        par = -1
        spine_ok = True
        for j in range(L):
            if not S[j] or not spine_ok:
                break
            ids = {}
            order = ([chain[j]] if chain[j] in S[j] else []) + [c for c in S[j] if c != chain[j]]
            for c in order:
                nodes.append((par, c))
                ids[c] = len(nodes) - 1
            if chain[j] in ids:
                par = ids[chain[j]]
            else:
                spine_ok = False
        spine_parent = {}
        pp = -1
        for j in range(L):
            spine_parent[pp] = j
            hit = [i for i, (pa, c) in enumerate(nodes) if pa == pp and c == chain[j]]
            if not hit:
                break
            pp = hit[0]
        t, q = eval_tree(nodes, lambda pid: ps[spine_parent[pid]] if pid in spine_parent else {},
                         lambda pid: qs[spine_parent[pid]] if pid in spine_parent else {})
        cur[str(W)] = [t, q, len(nodes)]
    out["cur"] = cur
    return out


def replay_unit(args):
    cap, unit = args
    with open(cap, "rb") as f:
        f.seek(unit["off0"])
        blob = f.read(unit["off1"] - unit["off0"])
    res = []
    for line in blob.decode().splitlines():
        rec = json.loads(line)
        if rec.get("k") != "round":
            continue
        o = replay_round(rec)
        o.update({"unit": unit["unit_id"], "prompt": unit["prompt_idx"], "T": unit["T"], "task": rec["task"],
                  "round": rec["round"]})
        res.append(o)
    return res


# ---------------------------------------------------------------- selftest (hand cases; optional real-capture invariants)
def selftest(outdir, capture=None, units=None):
    rep = {"cases": {}, "pass": True}

    def chk(name, ok, detail=None):
        rep["cases"][name] = {"ok": bool(ok), "detail": detail}
        rep["pass"] &= bool(ok)

    # (1) ordered recursive rejection, hand numbers: p=(a .2, b .5, z .3), q=(a .7, b .1, z .2), children a then b
    p = {"a": .2, "b": .5, "z": .3}
    q = {"a": .7, "b": .1, "z": .2}
    t, qq = eval_tree([(-1, "a"), (-1, "b")], lambda pid: p, lambda pid: q)
    pa = .2 / .7
    chk("rrs_hand_q", abs(qq - (pa + (1 - pa) * 1.0)) < 1e-12, qq)
    chk("target_walk_hand_t", abs(t - .7) < 1e-12, t)
    # (2) two-level: a -> c ; p_root a .5 b .3; p_a c .6; target walk = .5 + .3 + .5*.6
    rows = {(): {"a": .5, "b": .3, "z": .2}, ("a",): {"c": .6, "y": .4}}
    t, _ = eval_path_tree([("a",), ("b",), ("a", "c")], rows, lambda path: {"a": .5, "b": .3, "z": .2, "c": .6, "y": .4})
    chk("two_level_walk", abs(t - 1.1) < 1e-12, t)
    # (3) single-child chain under V1q == min(1,p/q) product (served V0)
    pr = [{"a": .5, "x": .5}, {"b": .9, "x": .1}]
    qr = [{"a": .4, "x": .6}, {"b": .95, "x": .05}]
    chain = ["a", "b"]
    _, qv = eval_tree([(-1, "a"), (0, "b")], lambda pid: pr[pid + 1], lambda pid: qr[pid + 1])
    chk("chain_equals_v0", abs(qv - v0_rej(chain, pr, qr, 2)) < 1e-12, [qv, v0_rej(chain, pr, qr, 2)])
    # (4) ceiling: best N by reach, optimal and closed
    rows = {(): {"a": .5, "b": .3, "z": .2}, ("a",): {"c": .6, "y": .4}, ("b",): {"c": .1, "y": .9}}
    cs = build_ceiling(rows, 3, 4)
    chk("ceiling_order", [r for _, r in cs] == sorted([r for _, r in cs], reverse=True) and
        abs(sum(r for _, r in cs) - (.5 + .3 + .3 + .27)) < 1e-12, [(p_, round(r, 4)) for p_, r in cs])
    # (5) pando on a hand lattice stays eligible-only: no node under a parent without a captured row
    q_at = lambda d, pred: {1: .7, 2: .3} if d == 0 else {5: .6, 6: .4}  # noqa: E731
    rows2 = {(): {1: .6, 2: .4}, (1,): {5: .5, 6: .5}}
    sel = build_pando(rows2, q_at, [{1: 0, 2: 1}, {5: 0, 6: 1}], 3, 2, 10)
    paths = [s[0] for s in sel]
    chk("pando_eligible_only", all(len(p_) <= 2 for p_ in paths) and (2, 5) not in paths and (1, 5) in paths, paths)
    # (6) real-capture invariants on the first captured unit (no replay output is produced)
    if capture and units and os.path.exists(capture):
        us = [json.loads(l) for l in open(units)]
        us = [u for u in us if u.get("status") == "CAPTURED"]
        res = replay_unit((capture, us[0]))
        ok_ord = ok_ceil = ok_mono = ok_k1 = True
        for r in res:
            for n in NS:
                ok_mono &= r["ceil"][str(n)] <= r["ceil"][str(n + 4 if n < 16 else 16)] + 1e-12
                for k in KS:
                    ok_ceil &= r["pando"][str(k)][str(n)][0] <= r["ceil"][str(n)] + 1e-9
                ok_ceil &= r["cur"][str(n)][0] <= r["ceil"][str(n)] + 1e-9
            ok_k1 &= r["pando"]["1"]["8"][2] <= 8
            ok_ord &= r["V0"] >= 0.0
        chk("real_unit0_ceiling_dominates_V1t", ok_ceil, {"rounds": len(res)})
        chk("real_unit0_ceiling_monotone_in_N", ok_mono)
        chk("real_unit0_node_counts", ok_k1 and ok_ord)
    os.makedirs(outdir, exist_ok=True)
    json.dump(rep, open(os.path.join(outdir, "SELFTEST_HEADROOM.json"), "w"), indent=1)
    print(json.dumps({"pass": rep["pass"]}))
    return 0 if rep["pass"] else 1


# ---------------------------------------------------------------- analysis
def analyze(outdir):
    recs = [json.loads(l) for l in open(os.path.join(outdir, "results_headroom.jsonl"))]
    agg = {}
    for r in recs:
        a = agg.setdefault((r["T"], r["prompt"]), {"n": 0, "L": 0, "real": 0, "V0": 0.0, "ceil": {}, "pando": {}, "cur": {}})
        a["n"] += 1
        a["L"] += r["L"]
        a["real"] += r["realized_acc"]
        a["V0"] += 1.0 + r["V0"]
        for n in NS:
            s = str(n)
            a["ceil"][s] = a["ceil"].get(s, 0.0) + 1.0 + r["ceil"][s]
            for w in ("t", "q"):
                i = 0 if w == "t" else 1
                a["cur"][(s, w)] = a["cur"].get((s, w), 0.0) + 1.0 + r["cur"][s][i]
                for k in KS:
                    key = (str(k), s, w)
                    a["pando"][key] = a["pando"].get(key, 0.0) + 1.0 + r["pando"][str(k)][s][i]

    def tot(T, plist, getter):
        n = s = 0.0
        for p in plist:
            a = agg.get((T, p))
            if a:
                n += a["n"]
                s += getter(a)
        return s / n if n else float("nan")

    brng = random.Random(BSEED)
    boots = [[brng.choice(HOLD) for _ in HOLD] for _ in range(B)]

    def ratio(T, num, den=lambda a: a["V0"]):
        pt = tot(T, HOLD, num) / tot(T, HOLD, den)
        bs = sorted(tot(T, b, num) / tot(T, b, den) for b in boots)
        lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
        return [round(pt, 4), round(lo, 4), round(hi, 4), round((hi - lo) / 2, 4)]

    table = []
    for T in TS_ALL:
        for n in NS:
            s = str(n)
            kfit = max(KS, key=lambda k: (tot(T, CAL, lambda a: a["pando"][(str(k), s, "t")]), -k))
            row = {"T": T, "N": n, "k_fit": kfit,
                   "ceiling/V0": ratio(T, lambda a: a["ceil"][s]),
                   "pando_V1t/V0": ratio(T, lambda a: a["pando"][(str(kfit), s, "t")]),
                   "pando_V1q/V0": ratio(T, lambda a: a["pando"][(str(kfit), s, "q")]),
                   "cur_V1t/V0": ratio(T, lambda a: a["cur"][(s, "t")]),
                   "cur_V1q/V0": ratio(T, lambda a: a["cur"][(s, "q")]),
                   "greedy_chain_V1t/V0": ratio(T, lambda a: a["pando"][("1", s, "t")]),
                   "pando_V1t/cur_V1t": ratio(T, lambda a: a["pando"][(str(kfit), s, "t")],
                                             lambda a: a["cur"][(s, "t")]),
                   "ceiling/pando_V1t": ratio(T, lambda a: a["ceil"][s], lambda a: a["pando"][(str(kfit), s, "t")]),
                   "holdout_rounds": int(sum(agg[(T, p)]["n"] for p in HOLD if (T, p) in agg))}
            table.append(row)
    val = {}
    fail = []
    for T in TS_ALL:
        rs = [r for r in recs if r["T"] == T]
        nd = sum(r["L"] for r in rs)
        rate = sum(r["V0"] for r in rs) / nd
        real = sum(r["realized_acc"] for r in rs) / nd
        val[str(T)] = {"rounds": len(rs), "V0_accept_rate": rate, "realized_accept_rate": real,
                       "V0_over_realized": rate / real}
        if T in TS_GATE and abs(rate / real - 1.0) > 0.02:
            fail.append(T)
    gate = [r for r in table if r["T"] in TS_GATE]
    ceil16 = {r["T"]: r["ceiling/V0"] for r in gate if r["N"] == 16}
    close_ceiling = all(v[0] < CEIL_CLOSE for v in ceil16.values()) and len(ceil16) == len(TS_GATE)
    close_pando = all(r["pando_V1t/V0"][2] < PANDO_CLOSE_UP for r in gate)
    file_cells = [[r["T"], r["N"]] for r in gate if r["pando_V1t/V0"][0] >= PANDO_FILE_PT and r["pando_V1t/V0"][1] > 1.0]
    if fail:
        verdict = "FUTILITY_BASELINE"
    elif file_cells:
        verdict = "FILE_GPU_ROW"
    elif close_ceiling or close_pando:
        verdict = "CLOSE_TREE_SHAPE"
    else:
        verdict = "INCONCLUSIVE"
    summary = {"verdict": verdict, "baseline_fail_T": fail, "v0_validation": val, "rounds_total": len(recs),
               "ceiling_over_V0_N16": ceil16, "close_by_ceiling": close_ceiling, "close_by_pando": close_pando,
               "file_cells": file_cells,
               "ci_halfwidth_le_1pct": {f"{r['T']}/N{r['N']}": r["pando_V1t/V0"][3] <= 0.01 for r in gate},
               "greedy_chain_bias_flag": {str(r["T"]): r["greedy_chain_V1t/V0"][0] for r in gate if r["N"] == 8},
               "v1q_exceeds_ceiling_cells": [[r["T"], r["N"]] for r in gate if r["pando_V1q/V0"][0] > r["ceiling/V0"][0]]}
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
        return selftest(a.out, a.capture, a.units)
    if a.cmd == "replay":
        st = os.path.join(a.out, "SELFTEST_HEADROOM.json")
        if not os.path.exists(st) or not json.load(open(st))["pass"]:
            print("selftest not PASS; refusing", file=sys.stderr)
            return 2
        if sha(a.capture) != CAP_SHA or sha(a.units) != UNITS_SHA:
            print("capture digest mismatch; refusing", file=sys.stderr)
            return 2
        units = [json.loads(l) for l in open(a.units)]
        units = [u for u in units if u.get("status") == "CAPTURED"]
        assert len(units) == 192
        tmp = os.path.join(a.out, "results_headroom.jsonl.tmp")
        with Pool(a.procs) as pool, open(tmp, "w") as f:
            for res in pool.imap(replay_unit, [(a.capture, u) for u in units]):
                for o in res:
                    f.write(json.dumps(o, sort_keys=True) + "\n")
        os.replace(tmp, os.path.join(a.out, "results_headroom.jsonl"))
        return 0
    table, summary = analyze(a.out)
    json.dump({"table": table, "summary": summary}, open(os.path.join(a.out, "analysis_headroom.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
