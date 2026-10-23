#!/usr/bin/env python3
"""Expert-locality analyses on epyc.moe.routing_trace.v1 shards: adjacent-token reuse (EOR / IR_t) against chance,
LRU stack-distance (reuse distance) hit-rate curves, and expert co-activation structure against a rescaled-independence null.

* IR_t: |E_t n E_{t-1}| / K between adjacent tokens of one window; chance = K / n_experts (RT-2 rider).
* Reuse distance: per window and layer, an expert's stack distance at a token is the number of DISTINCT experts selected since
  its last use (state before the current token). A fully-associative LRU cache of k experts per layer hits iff distance < k;
  first use in a window is a compulsory miss. Within-window only (cold start every window); a cache persisting across
  windows is a different question and is not measured here.
* Co-activation: pair counts over experts selected in the same token; expected counts = rescaled independence (marginals
  preserved, total pair mass preserved); excess = sum max(obs-exp, 0) / total, with the Poisson noise floor of the same
  statistic beside it so a small sample is not read as structure.

Usage: locality_report.py RESULTS_DIR --out OUT_DIR [--domains ...] [--lru-windows 64] [--lru-layers 0,6,12,...] [--seed 7]
CPU analysis only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_io as tio  # noqa: E402

CACHE_SIZES = (8, 16, 32, 64, 128, 256)


def ir_by_phase(t: tio.Trace, l: int) -> dict:
    E, K = int(t.n_experts[l]), int(t.top_k[l])
    ids = t.layer_ids(l)
    same = t.win[1:] == t.win[:-1]
    out = {"chance": K / E}
    for name, ph in (("prefill", 0), ("decode", 1)):
        sel = same & (t.phase[1:] == ph)
        n = int(sel.sum())
        if not n:
            continue
        a, b = ids[1:][sel].astype(np.int64), ids[:-1][sel].astype(np.int64)
        ma = np.zeros((n, E), bool)
        mb = np.zeros((n, E), bool)
        r = np.arange(n)[:, None]
        ma[r, a] = True
        mb[r, b] = True
        ir = (ma & mb).sum(1) / K
        out[name] = {"pairs": n, "ir_mean": float(ir.mean()), "ratio_to_chance": float(ir.mean() / (K / E))}
    return out


def lru_curve(t: tio.Trace, l: int, win_ids: np.ndarray, sizes=CACHE_SIZES) -> dict:
    E, K = int(t.n_experts[l]), int(t.top_k[l])
    ids_all = t.layer_ids(l)
    hits = {ph: np.zeros(len(sizes)) for ph in (0, 1)}
    tot = {0: 0, 1: 0}
    cold = {0: 0, 1: 0}
    dists = {0: [], 1: []}
    for w in win_ids:
        m = np.flatnonzero(t.win == w)
        if m.size == 0:
            continue
        ids, ph = ids_all[m].astype(np.int64), t.phase[m]
        last = np.full(E, -1, np.int64)
        for i in range(ids.shape[0]):
            sel = ids[i]
            ls = last[sel]
            seen = ls >= 0
            p = int(ph[i])
            tot[p] += K
            cold[p] += int((~seen).sum())
            if seen.any():
                d = (last[None, :] > ls[seen][:, None]).sum(1)
                dists[p].append(d)
                for si, k in enumerate(sizes):
                    hits[p][si] += int((d < k).sum())
            last[sel] = i
    out = {}
    for name, p in (("prefill", 0), ("decode", 1)):
        if not tot[p]:
            continue
        d = np.concatenate(dists[p]) if dists[p] else np.zeros(0)
        out[name] = {"selections": tot[p], "compulsory_miss_rate": cold[p] / tot[p],
                     "hit_rate": {str(k): float(hits[p][i] / tot[p]) for i, k in enumerate(sizes)},
                     "uniform_chance_hit_rate": {str(k): float(min(1.0, k / E)) for k in sizes},
                     "median_stack_distance": float(np.median(d)) if d.size else None,
                     "p90_stack_distance": float(np.percentile(d, 90)) if d.size else None}
    return out


def coactivation(t: tio.Trace, l: int, rng: np.random.Generator, min_exp: float = 5.0) -> dict:
    E, K = int(t.n_experts[l]), int(t.top_k[l])
    ids = np.sort(t.layer_ids(l).astype(np.int64), axis=1)
    N = ids.shape[0]
    ii, jj = zip(*itertools.combinations(range(K), 2))
    pair = ids[:, list(ii)] * E + ids[:, list(jj)]
    C = np.bincount(pair.ravel(), minlength=E * E).reshape(E, E).astype(np.float64)
    iu = np.triu_indices(E, 1)
    obs = C[iu]
    c = np.bincount(ids.ravel(), minlength=E).astype(np.float64)
    q = c / N
    ex = np.outer(q, q)[iu]
    ex = ex * (obs.sum() / ex.sum())

    def stats(o):
        tot = o.sum()
        top = max(1, int(0.01 * o.size))
        return (float(np.maximum(o - ex, 0).sum() / tot), float(np.sort(o)[::-1][:top].sum() / tot))

    exc, conc = stats(obs)
    nexc, nconc = [], []
    for _ in range(3):
        n_o = rng.poisson(ex).astype(np.float64)
        a, b = stats(n_o)
        nexc.append(a)
        nconc.append(b)
    ok = ex >= min_exp
    lift = np.where(ok, obs / np.maximum(ex, 1e-12), 0.0)
    top = np.argsort(-lift)[:5]
    return {"tokens": int(N), "excess_fraction": exc, "excess_fraction_noise": float(np.mean(nexc)),
            "top1pct_pair_mass": conc, "top1pct_pair_mass_noise": float(np.mean(nconc)),
            "pairs_with_expected_ge_min": int(ok.sum()),
            "max_lift": float(lift.max()), "lift_p99": float(np.percentile(lift[ok], 99)) if ok.any() else None,
            "frac_pairs_lift_ge_2": float((lift[ok] >= 2).mean()) if ok.any() else None,
            "top_pairs": [{"a": int(iu[0][i]), "b": int(iu[1][i]), "obs": int(obs[i]), "exp": float(ex[i]), "lift": float(lift[i])} for i in top]}


def analyse(results_dir, domains, lru_windows: int, lru_layers: list, seed: int, skip_coact: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    out: dict = {"domains": {}}
    for dom in domains:
        t = tio.load_domain(results_dir, dom)
        L = t.L
        d: dict = {"ir": [], "lru": {}, "coact": []}
        for l in range(L):
            r = ir_by_phase(t, l)
            r["layer"] = l
            d["ir"].append(r)
        wins = np.unique(t.win)[:lru_windows]
        for l in lru_layers:
            d["lru"][str(l)] = lru_curve(t, l, wins)
        if not skip_coact:
            for l in range(L):
                r = coactivation(t, l, rng)
                r["layer"] = l
                d["coact"].append(r)
        out["domains"][dom] = d
        out["domains"][dom]["lru_windows"] = int(wins.size)
    return out


def to_markdown(r: dict, title: str) -> str:
    L = ["# " + title, ""]
    L += ["## Adjacent-token reuse IR_t (mean over layers; ratio to chance K/E)", "", "| workload | phase | IR mean | chance | ratio | layer min..max IR |", "|---|---|---|---|---|---|"]
    for dom, d in r["domains"].items():
        for ph in ("prefill", "decode"):
            v = [x[ph]["ir_mean"] for x in d["ir"] if ph in x]
            if v:
                ch = d["ir"][0]["chance"]
                L.append(f"| {dom} | {ph} | {np.mean(v):.4f} | {ch:.4f} | {np.mean(v) / ch:.1f}x | {min(v):.3f}..{max(v):.3f} |")
    L += ["", "## LRU hit rate vs cache size (experts per layer; mean over the sampled layers; within-window)", "",
          "| workload | phase | " + " | ".join(f"k={k}" for k in CACHE_SIZES) + " | compulsory miss | median dist |", "|---|---|" + "---|" * (len(CACHE_SIZES) + 2)]
    for dom, d in r["domains"].items():
        for ph in ("prefill", "decode"):
            rows = [v[ph] for v in d["lru"].values() if ph in v]
            if not rows:
                continue
            hr = [np.mean([x["hit_rate"][str(k)] for x in rows]) for k in CACHE_SIZES]
            L.append(f"| {dom} | {ph} | " + " | ".join(f"{h:.3f}" for h in hr) + f" | {np.mean([x['compulsory_miss_rate'] for x in rows]):.3f} | {np.mean([x['median_stack_distance'] for x in rows]):.0f} |")
    E = None
    L += ["", "## Co-activation (mean over layers)", "", "| workload | excess fraction | noise floor | top-1% pair mass | noise floor | max lift (exp>=5) | frac pairs lift>=2 |", "|---|---|---|---|---|---|---|"]
    for dom, d in r["domains"].items():
        c = d["coact"]
        if c:
            L.append(f"| {dom} | {np.mean([x['excess_fraction'] for x in c]):.4f} | {np.mean([x['excess_fraction_noise'] for x in c]):.4f} | "
                     f"{np.mean([x['top1pct_pair_mass'] for x in c]):.4f} | {np.mean([x['top1pct_pair_mass_noise'] for x in c]):.4f} | "
                     f"{np.mean([x['max_lift'] for x in c]):.1f} | {np.mean([x['frac_pairs_lift_ge_2'] or 0 for x in c]):.4f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="")
    ap.add_argument("--lru-windows", type=int, default=64)
    ap.add_argument("--lru-layers", default="0,6,12,18,24,30,36,42,47")
    ap.add_argument("--skip-coact", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--title", default="Expert locality report")
    a = ap.parse_args(argv)
    doms = [d for d in a.domains.split(",") if d] or sorted(tio.discover(a.results_dir))
    r = analyse(a.results_dir, doms, a.lru_windows, [int(x) for x in a.lru_layers.split(",") if x], a.seed, a.skip_coact)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "locality_report.json").write_text(json.dumps(r) + "\n")
    md = to_markdown(r, a.title)
    (out / "locality_report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
