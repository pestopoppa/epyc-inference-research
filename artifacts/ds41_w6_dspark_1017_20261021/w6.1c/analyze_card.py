"""DS41-W6.1c analysis: DSpark alpha and block-2/3 speedup per sampling arm (card T=1.0/top_p=1.0, stoch T=0.7, greedy identity).

usage: analyze_card.py <runs_dir> <out_json>      (runs_dir holds p<N>_plain/ and p<N>_dspark/, each with rows.json + DONE)

Only COMPLETE pairs (both arms DONE, same threads/cpus) count. Alpha = accepted/drafted over VALID dspark rows. Speedup of block b
in pair p = (sum predicted_n / sum predicted_s over dspark rows of that block) / (same over the plain arm's rows of the same mode).
CIs: bootstrap over (pair, prompt) cells (10,000 resamples, seed 20261021), reported beside the per-pair values.
Every invalid row is counted. The sampling triple of each mode is read from the rows and must be consistent.
Timing here is INDICATIVE (shared host, not a quiet window): no champion-baseline claim may be built on it.
"""
import json, os, random, sys, glob

runs, outp = sys.argv[1], sys.argv[2]
random.seed(20261021)
NPRED = 128


def load(label):
    f = os.path.join(runs, label, "rows.json")
    return json.load(open(f)) if os.path.exists(os.path.join(runs, label, "DONE")) and os.path.exists(f) else None


def arm_meta(label):
    try:
        return json.load(open(os.path.join(runs, label, "arm.json")))
    except Exception:
        return {}


pairs = {}
for d in sorted(glob.glob(os.path.join(runs, "p[0-9]*_plain"))):
    p = os.path.basename(d)[1:].split("_")[0]
    a, b = load(f"p{p}_plain"), load(f"p{p}_dspark")
    if not a or not b:
        continue
    ma, mb = arm_meta(f"p{p}_plain"), arm_meta(f"p{p}_dspark")
    if (ma.get("threads"), ma.get("cpus")) != (mb.get("threads"), mb.get("cpus")):
        continue
    pairs[p] = {"plain": a, "dspark": b, "threads": ma.get("threads"), "cpus": ma.get("cpus")}


def sel(rows, mode, nmax=None):
    return [r for r in rows if r["mode"] == mode and (nmax is None or r.get("n_max") == nmax)]


def tput(rows):
    n = sum((r.get("predicted_n") or 0) for r in rows if r.get("valid"))
    t = sum(((r.get("predicted_ms") or 0) / 1000.0) for r in rows if r.get("valid"))
    return (n / t) if t else float("nan")


def q(b, x):
    b = sorted(b)
    return b[int(x * (len(b) - 1))]


res = {"schema": "ds41.w61c.v1", "n_complete_pairs": len(pairs), "pairs": sorted(pairs, key=int),
       "threads": sorted({v["threads"] for v in pairs.values() if v["threads"]}), "claim_tuple": {}, "modes": {}, "invalid_rows": {}, "greedy_identity": None,
       "timing_caveat": "indicative host (not a quiet window); speedups are plain-vs-dspark ratios inside one pair on the same cpus/threads"}
for p, v in pairs.items():
    for kind in ("plain", "dspark"):
        bad = [r for r in v[kind] if not r.get("valid")]
        res["invalid_rows"][f"p{p}_{kind}"] = {"invalid": len(bad), "total": len(v[kind]), "reasons": sorted({x for r in bad for x in r.get("invalid_reasons", [])})}

for mode in ("card", "stoch"):
    trip = {json.dumps(r["sampling"], sort_keys=True) for v in pairs.values() for k in ("plain", "dspark") for r in sel(v[k], mode)}
    res["claim_tuple"][mode] = {"sampling_triples_seen": sorted(trip), "consistent": len(trip) <= 1}
    res["modes"][mode] = {}
    for nmax in (2, 3):
        cells = []   # (pair, prompt, acc, drafted, spec_n, spec_t, plain_n, plain_t)
        perpair = {}
        for p, v in pairs.items():
            D = {r["prompt"]: r for r in sel(v["dspark"], mode, nmax)}
            Pl = {r["prompt"]: r for r in sel(v["plain"], mode)}
            for i in sorted(set(D) & set(Pl)):
                d, pl = D[i], Pl[i]
                if not (d.get("valid") and pl.get("valid")):
                    continue
                cells.append((p, i, d["draft_n_accepted"], d["draft_n"], d["predicted_n"], d["predicted_ms"] / 1000.0, pl["predicted_n"], pl["predicted_ms"] / 1000.0))
            cs = [c for c in cells if c[0] == p]
            if cs:
                perpair[p] = {"speedup": (sum(c[4] for c in cs) / sum(c[5] for c in cs)) / (sum(c[6] for c in cs) / sum(c[7] for c in cs)),
                              "alpha": sum(c[2] for c in cs) / max(1, sum(c[3] for c in cs)), "dspark_tok_s": sum(c[4] for c in cs) / sum(c[5] for c in cs),
                              "plain_tok_s": sum(c[6] for c in cs) / sum(c[7] for c in cs)}
        if not cells:
            res["modes"][mode][str(nmax)] = {"n_cells": 0}
            continue

        def stat(cs):
            sp = (sum(c[4] for c in cs) / sum(c[5] for c in cs)) / (sum(c[6] for c in cs) / sum(c[7] for c in cs))
            return sum(c[2] for c in cs) / max(1, sum(c[3] for c in cs)), sp
        a0, s0 = stat(cells)
        keys = sorted({(c[0], c[1]) for c in cells})
        byk = {}
        for c in cells:
            byk.setdefault((c[0], c[1]), []).append(c)
        ba, bs = [], []
        for _ in range(10000):
            cs = [c for k in (random.choice(keys) for _ in keys) for c in byk[k]]
            a, s = stat(cs)
            ba.append(a); bs.append(s)
        res["modes"][mode][str(nmax)] = {"n_cells": len(cells), "alpha": a0, "alpha_ci95": [q(ba, .025), q(ba, .975)],
                                         "speedup": s0, "speedup_ci95": [q(bs, .025), q(bs, .975)], "per_pair": perpair,
                                         "accepted": sum(c[2] for c in cells), "drafted": sum(c[3] for c in cells)}

# greedy identity (pair 1): dspark greedy text must equal plain greedy text per prompt
gi = {}
for p, v in pairs.items():
    P = {r["prompt"]: r.get("text_sha") for r in sel(v["plain"], "greedy") if r.get("valid")}
    D = {r["prompt"]: r.get("text_sha") for r in sel(v["dspark"], "greedy") if r.get("valid")}
    common = sorted(set(P) & set(D))
    if common:
        gi[p] = {"compared": len(common), "identical": sum(1 for i in common if P[i] == D[i])}
res["greedy_identity"] = gi or None
res["verdict"] = "ok" if len(pairs) >= 5 else f"insufficient_pairs ({len(pairs)}/5): indicative only"
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps(res, indent=1))
