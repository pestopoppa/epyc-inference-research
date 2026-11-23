#!/usr/bin/env python3
"""SPEC-DTREE-DICE step 2b: offline replay of sampled-tree verification arms on a dice.v1 MTP chain capture.

Arms V0 (served sample-and-compare chain), V1 (recursive rejection without replacement, draw-blind shape),
V2 (Gumbel-max walk, draw-blind shape), V3 (DICE draw-informed selection, rho = sigma/alpha grid).
Tree family = chain spine with leaf siblings (the only family a chain capture can replay). See PREDECLARATION.md.

Usage:
  dice_replay.py selftest --out DIR
  dice_replay.py replay --capture CAP --units RESULTS_JSONL --out DIR [--procs N] [--draws R]
  dice_replay.py analyze --out DIR
Pure stdlib, deterministic (keyed random.Random seeds).
"""
import argparse, hashlib, json, math, os, random, sys
from multiprocessing import Pool

CAP_SHA = "00d810a82ba86172ef5860a3ed4055a3817cf66004b18ca88808beddddd74bb7"
UNITS_SHA = "ff058e74f82e5b4f4c75c4562d5dba95299b7c052af1e6b1ad56f6ff80fbd6ea"
WS = [1, 4, 8, 12, 16]
RHOS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
TS = [0.1, 0.3, 1.0]
BREAK_EVEN = {1: 1.0, 4: 1.0, 8: 1.079, 12: 1.079, 16: 1.141}
U_EPS = 2.0 ** -53


def gumbel(rng):
    u = min(max(rng.random(), U_EPS), 1.0 - U_EPS)
    return -math.log(-math.log(u))


def norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items() if v > 0} if s > 0 else {}


# ---------------------------------------------------------------- per-round primitives
def blind_shape(chain, qs, W):
    """Probability-greedy best-first over spine-with-leaves under raw q. Returns per-depth list of node ids
    (chain first if selected) for depths 0..L-1."""
    L = len(chain)
    cand = []
    pref = 0.0
    for j in range(L):
        for c, qv in qs[j].items():
            cand.append((pref + math.log(qv), j, 0 if c == chain[j] else 1, c))
        qc = qs[j].get(chain[j], 0.0)
        if qc <= 0:
            break
        pref += math.log(qc)
    cand.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    return select(cand, chain, W)


def select(cand_sorted, chain, W):
    """Take nodes in order, skipping any whose parent (chain token at depth j-1) is not yet selected."""
    L = len(chain)
    S = [[] for _ in range(L)]
    spine = 0  # number of chain nodes selected (depth 0..spine-1)
    n = 0
    pending = list(cand_sorted)
    progress = True
    while n < W and progress:
        progress = False
        rest = []
        for item in pending:
            if n >= W:
                rest.append(item)
                continue
            j, c = item[1], item[3]
            if j <= spine:
                S[j].append(c)
                n += 1
                if c == chain[j] and j == spine:
                    spine += 1
                progress = True
            else:
                rest.append(item)
        pending = rest
        if progress:
            pending.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    return S


def walk(S, chain, tgt):
    acc = 0
    for j in range(len(chain)):
        if tgt[j] in S[j]:
            acc += 1
            if tgt[j] != chain[j]:
                break
        else:
            break
    return acc


def v0_analytic(chain, ps, W):
    e, pr = 0.0, 1.0
    for j in range(min(W, len(chain))):
        pr *= ps[j].get(chain[j], 0.0)
        e += pr
    return e


def v2_analytic(S, chain, ps):
    e, pr = 0.0, 1.0
    for j in range(len(chain)):
        if not S[j]:
            break
        e += pr * sum(ps[j].get(c, 0.0) for c in S[j])
        if chain[j] not in S[j]:
            break
        pr *= ps[j].get(chain[j], 0.0)
    return e


def rrs_depth(p, q, chain_tok, m, trng, drng):
    """One depth of V1. Returns (accepted_token or None, emitted_token). Chain token point-mass first, then m siblings
    drawn without replacement from q minus chain, recursive rejection."""
    r = dict(p)
    if trng.random() < r.get(chain_tok, 0.0):
        return chain_tok, chain_tok
    r.pop(chain_tok, None)
    r = norm(r)
    pool = norm({c: v for c, v in q.items() if c != chain_tok})
    # draw m siblings without replacement (Gumbel-top-k on log q, drafter-own noise, fixed token order)
    keys = sorted(((math.log(v) + gumbel(drng), c) for c, v in sorted(pool.items())), reverse=True)
    sib = [c for _, c in keys[:m]]
    for s in sib:
        qs_ = pool.get(s, 0.0)
        if qs_ <= 0:
            continue
        if trng.random() < min(1.0, r.get(s, 0.0) / qs_):
            return s, s
        keys_ = set(r) | set(pool)
        rr = {k: max(r.get(k, 0.0) - pool.get(k, 0.0), 0.0) for k in keys_}
        if sum(rr.values()) <= 1e-15:
            break
        r = norm(rr)
        pool.pop(s, None)
        pool = norm(pool)
        if not pool:
            break
    # all rejected: emit from residual
    x, acc = trng.random(), 0.0
    items = sorted(r.items())
    for c, v in items:
        acc += v
        if x < acc:
            return None, c
    return None, items[-1][0]


def v1_mc(S, chain, ps, qs, trng, drng):
    acc = 0
    for j in range(len(chain)):
        if not S[j]:
            break
        has_chain = chain[j] in S[j]
        m = len(S[j]) - (1 if has_chain else 0)
        if not has_chain:  # shape without chain node at this depth: treat all as sampled siblings (chain excluded never)
            a, _ = rrs_depth(ps[j], qs[j], -1, m, trng, drng)
        else:
            a, _ = rrs_depth(ps[j], qs[j], chain[j], m, trng, drng)
        if a is None:
            break
        acc += 1
        if a != chain[j]:
            break
    return acc


def draw_noise(ps, qs, L, rng):
    gs, tgt = [], []
    for j in range(L):
        toks = sorted(set(ps[j]) | set(qs[j]))
        g = {t: gumbel(rng) for t in toks}
        gs.append(g)
        tgt.append(max(ps[j], key=lambda c: (math.log(ps[j][c]) + g[c], -c)))
    return gs, tgt


def dice_sorted(chain, qs, gs, rho, L):
    cand = []
    pref = 0.0
    for j in range(L):
        sc = {c: math.log(v) + rho * gs[j][c] for c, v in qs[j].items()}
        mx = max(sc.values())
        for c, v in sc.items():
            cand.append((pref + v - mx, j, 0 if c == chain[j] else 1, c))
        if chain[j] not in sc:
            break
        pref += sc[chain[j]] - mx
    cand.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    return cand


def replay_round(rec, R):
    """Returns per-W dict of expectations (accepted draft tokens, bonus excluded)."""
    rows = rec["rows"]
    nrow = 0
    for r in rows:
        if r is None:
            break
        nrow += 1
    L = max(0, min(rec["D"], len(rec["chain"]), len(rec["mtp_top"]), nrow - 1))
    chain = rec["chain"][:L]
    ps = [norm({int(c): float(p) for c, p in rows[j][1]}) for j in range(L)]
    qs = [norm({int(t[0]): float(t[2]) for t in rec["mtp_top"][j] if float(t[2]) > 0}) for j in range(L)]
    for j in range(L):
        if not qs[j]:
            qs[j] = {chain[j]: 1.0}
        if chain[j] not in qs[j]:  # chain token must be a drafter candidate
            qs[j][chain[j]] = min(qs[j].values()) * 1e-3
            qs[j] = norm(qs[j])
    chain_is_argmax = all(max(qs[j], key=qs[j].get) == chain[j] for j in range(L))
    out = {"L": L, "chain_is_argmax": chain_is_argmax, "W": {}}
    shapes = {W: blind_shape(chain, qs, W) for W in WS}
    acc_v1 = {W: 0 for W in WS}
    acc_v2 = {W: 0 for W in WS}
    acc_v3 = {W: [0] * len(RHOS) for W in WS}
    key = f"{rec['task']}:{rec['round']}:{rec['T']}"
    for d in range(R):
        trng = random.Random(f"tgt:{key}:{d}")
        gs, tgt = draw_noise(ps, qs, L, trng)
        for W in WS:
            acc_v2[W] += walk(shapes[W], chain, tgt)
        for ri, rho in enumerate(RHOS):
            cs = dice_sorted(chain, qs, gs, rho, L)
            for W in WS:
                acc_v3[W][ri] += walk(select(cs, chain, W), chain, tgt)
        urng = random.Random(f"rrs:{key}:{d}")
        drng = random.Random(f"drf:{key}:{d}")
        for W in WS:
            acc_v1[W] += v1_mc(shapes[W], chain, ps, qs, urng, drng)
    for W in WS:
        out["W"][str(W)] = {
            "V0": v0_analytic(chain, ps, W),
            "V0s": v0_analytic(chain, ps, L),
            "V1": acc_v1[W] / R,
            "V2": v2_analytic(shapes[W], chain, ps),
            "V2mc": acc_v2[W] / R,
            "V3": [a / R for a in acc_v3[W]],
            "shape": [len(s) for s in shapes[W]],
        }
    # realized served outcome check (accepted draft tokens of the actual round)
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
        o.update({"unit": unit["unit_id"], "prompt": unit["prompt_idx"], "T": unit["T"], "task": rec["task"],
                  "round": rec["round"]})
        res.append(o)
    return res


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- statistics
def gammq(a, x):
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(10000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chisq(counts, p, N):
    stat, k = 0.0, 0
    for c, pv in p.items():
        e = N * pv
        if e < 5:
            continue
        stat += (counts.get(c, 0) - e) ** 2 / e
        k += 1
    return stat, gammq((k - 1) / 2.0, stat / 2.0)


# ---------------------------------------------------------------- selftest (synthetic, known answers)
def selftest(outdir):
    rng = random.Random("selftest:20261123")
    report = {"cases": [], "pass": True}
    N = 200000
    for case in range(3):
        V = 12
        p = norm({i: rng.random() ** 3 for i in range(V)})
        q = dict(p) if case == 2 else norm({i: rng.random() ** 2 for i in range(V)})
        chain_tok = max(q, key=q.get)
        S = sorted(q, key=lambda c: -q[c])[:4]  # chain + 3 siblings
        counts = {k: {} for k in ("V0", "V1", "V2", "V3")}
        accs = {k: 0 for k in counts}
        for n in range(N):
            trng = random.Random(f"st:{case}:{n}")
            drng = random.Random(f"sd:{case}:{n}")
            gs, tgt = draw_noise([p], [q], 1, trng)
            t = tgt[0]
            counts["V0"][t] = counts["V0"].get(t, 0) + 1
            accs["V0"] += t == chain_tok
            counts["V2"][t] = counts["V2"].get(t, 0) + 1
            accs["V2"] += t in S
            cs = dice_sorted([chain_tok], [q], gs, 1.0, 1)
            S3 = select(cs, [chain_tok], 1)[0]
            counts["V3"][t] = counts["V3"].get(t, 0) + 1
            accs["V3"] += t in S3
            a, e = rrs_depth(p, q, chain_tok, 3, trng, drng)
            counts["V1"][e] = counts["V1"].get(e, 0) + 1
            accs["V1"] += a is not None
        cr = {"case": case, "q_eq_p": case == 2, "N": N}
        for k in counts:
            st, pv = chisq(counts[k], p, N)
            cr[k] = {"chi2": st, "p": pv, "acc": accs[k] / N, "ok": pv > 1e-3}
            report["pass"] &= pv > 1e-3
        pS = sum(p[c] for c in S)
        se = math.sqrt(pS * (1 - pS) / N)
        cr["V2_analytic"] = pS
        cr["V2_within_3se"] = abs(accs["V2"] / N - pS) <= 3 * se + 1e-12
        report["pass"] &= cr["V2_within_3se"]
        if case == 2:
            cr["V3_qeqp_W1_acc_is_1"] = accs["V3"] == N
            report["pass"] &= cr["V3_qeqp_W1_acc_is_1"]
        report["cases"].append(cr)
    # multi-depth shape/select sanity: select never takes an orphan
    chain = [1, 2, 3, 4]
    qs = [norm({i: rng.random() for i in range(10)}) for _ in range(4)]
    for j in range(4):
        qs[j][chain[j]] = 10.0
        qs[j] = norm(qs[j])
    for W in WS:
        S = blind_shape(chain, qs, W)
        ok = sum(map(len, S)) == W and all((not S[j]) or chain[j - 1] in S[j - 1] for j in range(1, 4))
        report.setdefault("shape_ok", []).append([W, [len(s) for s in S], ok])
        report["pass"] &= ok
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "SELFTEST.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"pass": report["pass"]}))
    return 0 if report["pass"] else 1


# ---------------------------------------------------------------- analysis
def analyze(outdir):
    recs = [json.loads(l) for l in open(os.path.join(outdir, "results.jsonl"))]
    # per prompt per T per W sums
    agg = {}
    for r in recs:
        for W, v in r["W"].items():
            k = (r["T"], int(W), r["prompt"])
            a = agg.setdefault(k, {"n": 0, "V0": 0.0, "V0s": 0.0, "V1": 0.0, "V2": 0.0, "V2mc": 0.0,
                                   "V3": [0.0] * len(RHOS)})
            a["n"] += 1
            for arm in ("V0", "V0s", "V1", "V2", "V2mc"):
                a[arm] += 1.0 + v[arm]
            for i, x in enumerate(v["V3"]):
                a["V3"][i] += 1.0 + x
    cal = range(0, 32)
    hold = list(range(32, 64))
    fit = {}
    for T in TS:
        for W in WS:
            best = max(range(len(RHOS)), key=lambda i: (sum(agg[(T, W, p)]["V3"][i] for p in cal if (T, W, p) in agg), -i))
            fit[(T, W)] = best
    brng = random.Random(20261123)
    B = 2000
    boots = [[brng.choice(hold) for _ in hold] for _ in range(B)]
    table = []

    def tot(T, W, arm, plist):
        n = s = 0.0
        for p in plist:
            a = agg.get((T, W, p))
            if not a:
                continue
            n += a["n"]
            s += a["V3"][fit[(T, W)]] if arm == "V3" else a[arm]
        return s / n if n else float("nan")

    for T in TS:
        for W in WS:
            row = {"T": T, "W": W, "rho_fit": RHOS[fit[(T, W)]],
                   "rounds_holdout": int(sum(agg[(T, W, p)]["n"] for p in hold if (T, W, p) in agg))}
            for arm in ("V0", "V0s", "V1", "V2", "V2mc", "V3"):
                row[arm] = tot(T, W, arm, hold)
            for num, den in (("V1", "V0"), ("V2", "V0"), ("V3", "V0"), ("V3", "V1"), ("V3", "V2"), ("V3", "V0s"),
                             ("V1", "V0s")):
                pt = row[num] / row[den]
                bs = sorted(tot(T, W, num, b) / tot(T, W, den, b) for b in boots)
                lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B) - 1]
                row[f"{num}/{den}"] = [pt, lo, hi, (hi - lo) / 2]
            row["break_even"] = BREAK_EVEN[W]
            table.append(row)
    fut_v31 = all(r["V3/V1"][0] < 1.02 for r in table)
    fut_v30 = all(r["V3/V0"][0] < r["break_even"] for r in table)
    gpu = [r for r in table if r["V3/V0"][0] >= r["break_even"] and r["V3/V0"][1] > 1.0 and r["V3/V1"][0] >= 1.02]
    rrs = [r for r in table if r["V1/V0"][0] >= r["break_even"] and r["V1/V0"][1] > 1.0]
    if gpu:
        verdict = "FILE_GPU_ROW"
    elif rrs:
        verdict = "FILE_RRS_ROW"
    else:
        verdict = "CLOSE_FUTILITY"
    hw = {str(T): max(r["V3/V1"][3] for r in table if r["T"] == T) for T in TS}
    realized = {}
    for r in recs:
        x = realized.setdefault(r["T"], [0, 0.0, 0])
        x[0] += r["realized_acc"]
        x[1] += r["W"]["4"]["V0s"]
        x[2] += 1
    summary = {
        "futility_clause_V3_over_V1_lt_1.02_everywhere": fut_v31,
        "futility_clause_V3_over_V0_below_break_even_everywhere": fut_v30,
        "futility_fired": fut_v31 and fut_v30,
        "gpu_cells": [[r["T"], r["W"]] for r in gpu],
        "rrs_cells": [[r["T"], r["W"]] for r in rrs],
        "verdict": verdict,
        "stop_on_answer_halfwidth_V3_over_V1_max_over_W": hw,
        "stop_on_answer_met_T0.3": hw["0.3"] <= 0.01,
        "chain_is_argmax_frac": sum(r["chain_is_argmax"] for r in recs) / len(recs),
        "served_check_realized_vs_analytic_acc_per_round": {str(T): [v[0] / v[2], v[1] / v[2], v[2]] for T, v in realized.items()},
        "rounds_total": len(recs),
    }
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
        st = json.load(open(os.path.join(a.out, "SELFTEST.json")))
        if not st["pass"]:
            print("selftest not PASS; refusing", file=sys.stderr)
            return 2
        if sha(a.capture) != CAP_SHA or sha(a.units) != UNITS_SHA:
            print("capture digest mismatch; refusing", file=sys.stderr)
            return 2
        units = [json.loads(l) for l in open(a.units)]
        assert len(units) == 192 and all(u["status"] == "CAPTURED" for u in units)
        with Pool(a.procs) as pool, open(os.path.join(a.out, "results.jsonl"), "w") as f:
            for res in pool.imap(replay_unit, [(a.capture, u, a.draws) for u in units]):
                for o in res:
                    f.write(json.dumps(o, sort_keys=True) + "\n")
        return 0
    table, summary = analyze(a.out)
    json.dump({"table": table, "summary": summary}, open(os.path.join(a.out, "analysis.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
