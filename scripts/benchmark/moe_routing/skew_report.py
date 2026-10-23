#!/usr/bin/env python3
"""RT-5 expert-skew report + routing entropy + cross-workload overlap, from epyc.moe.routing_trace.v1 shards.

Per layer and workload: share of expert selections landing on the top 10% / 25% most-selected experts (in-sample, and
held-out: ranking learned on even windows, scored on odd windows), the uniform-routing null for the same selection count
(multinomial Monte Carlo, so a small decode sample is not read as skew), normalised entropy, effective expert count, Gini,
never-selected experts. Cross-workload: top-set Jaccard and Spearman of per-expert counts between domains, with the
random-set chance and the within-domain even/odd ceiling.

The RT-5 gate (CAFE-7 trigger) is top-10% share >= 0.60 on >= 2 workloads. The verdict block applies it literally and states
which model the traces came from; a trace of another model is a PROXY, not the RT-5 answer for the frontdoor model.

Usage: skew_report.py RESULTS_DIR --out OUT_DIR [--domains code,prose,reasoning] [--gate 0.60]
CPU analysis only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_io as tio  # noqa: E402


def top_n(frac: float, E: int) -> int:
    return max(1, int(math.ceil(frac * E - 1e-9)))


def share_top(counts: np.ndarray, n: int) -> float:
    tot = counts.sum()
    return float(np.sort(counts)[::-1][:n].sum() / tot) if tot else float("nan")


def heldout_share(c_train: np.ndarray, c_test: np.ndarray, n: int) -> float:
    top = np.argsort(-c_train, kind="stable")[:n]
    tot = c_test.sum()
    return float(c_test[top].sum() / tot) if tot else float("nan")


def entropy_stats(c: np.ndarray) -> dict:
    p = c / c.sum()
    nz = p[p > 0]
    h = float(-(nz * np.log(nz)).sum())
    s = np.sort(c).astype(np.float64)
    n = s.size
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum()))
    return {"entropy_norm": h / math.log(c.size), "effective_experts": math.exp(h), "gini": gini,
            "never_selected": int((c == 0).sum()), "top1_share": float(s[-1] / s.sum())}


def balance_stats(c: np.ndarray) -> dict:
    """RT-5.1 load-balance statistics of one layer's per-expert selection counts (intake-2195#record definitions, as used by the
    loss-free / ID / Quantile balancing papers): MaxVio = (max - mean) / mean, MinVio = (mean - min) / mean over expert loads,
    inactive-expert ratio = share of experts that received no selection. Chance for a uniform router with this many selections is
    reported by the caller (null)."""
    m = float(c.mean())
    if m == 0:
        return {"maxvio": float("nan"), "minvio": float("nan"), "inactive_ratio": float("nan")}
    return {"maxvio": float((c.max() - m) / m), "minvio": float((m - c.min()) / m), "inactive_ratio": float((c == 0).mean())}


def null_balance(E: int, n_rows: int, K: int, trials: int = 60, seed: int = 9) -> dict:
    rng = np.random.default_rng(seed)
    out = [balance_stats(rng.binomial(max(1, n_rows), K / E, size=E)) for _ in range(trials)]
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def null_top_share(E: int, n_sel: int, K: int, n: int, trials: int = 200, seed: int = 7) -> float:
    """Expected top-n share when every row picks K DISTINCT experts uniformly (rows = n_sel // K)."""
    rng = np.random.default_rng(seed)
    rows = max(1, n_sel // K)
    out = []
    for _ in range(trials):
        # K distinct uniform picks per row == argpartition of random keys; per-expert count is Binomial(rows, K/E)
        c = rng.binomial(rows, K / E, size=E)
        out.append(np.sort(c)[::-1][:n].sum() / max(1, c.sum()))
    return float(np.mean(out))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        r = np.empty(x.size)
        order = np.argsort(x, kind="stable")
        r[order] = np.arange(x.size)
        # average ties
        u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(u.size)
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def jaccard_chance(E: int, n: int) -> float:
    x = n * n / E
    return x / (2 * n - x)


def subsets(t: tio.Trace) -> dict:
    return {"prefill": t.phase == 0, "decode": t.phase == 1}


def analyse(results_dir, domains, gate: float, fracs=(0.10, 0.25)) -> dict:
    data = {d: tio.load_domain(results_dir, d) for d in domains}
    t0 = next(iter(data.values()))
    L = t0.L
    out: dict = {"arch": t0.arch, "n_experts": sorted(set(t0.n_experts.tolist())), "top_k": sorted(set(t0.top_k.tolist())),
                 "layers": L, "domains": {}, "overlap": {}, "gate": gate}
    counts: dict = {}   # (domain, phase, half) -> [L][E]
    for dom, t in data.items():
        out["domains"][dom] = {}
        for ph, m in subsets(t).items():
            if not m.any():
                continue
            even = m & (t.win % 2 == 0)
            odd = m & (t.win % 2 == 1)
            per_layer = []
            cs_all, cs_even, cs_odd = [], [], []
            for l in range(L):
                E, K = int(t.n_experts[l]), int(t.top_k[l])
                ids = t.layer_ids(l)
                c = tio.layer_counts(ids[m], E)
                ce, co = tio.layer_counts(ids[even], E), tio.layer_counts(ids[odd], E)
                cs_all.append(c), cs_even.append(ce), cs_odd.append(co)
                row = {"layer": l, "n_experts": E, "top_k": K, "selections": int(c.sum())}
                for f in fracs:
                    n = top_n(f, E)
                    key = f"top{int(round(f * 100))}"
                    row[key + "_n"] = n
                    row[key + "_chance"] = n / E
                    row[key + "_in_sample"] = share_top(c, n)
                    row[key + "_null"] = null_top_share(E, int(c.sum()), K, n, trials=60)
                    if ce.sum() and co.sum():
                        row[key + "_heldout"] = 0.5 * (heldout_share(ce, co, n) + heldout_share(co, ce, n))
                row.update(entropy_stats(c))
                row.update({"stream_" + k: v for k, v in balance_stats(c).items()})
                row.update({"stream_null_" + k: v for k, v in null_balance(E, int(m.sum()), K, trials=20).items()})
                if ph == "prefill":
                    sub = t.select(m)
                    blk, nb, _ = tio.blocks(sub, 512)
                    if nb:
                        X = tio.layer_loads(sub.layer_ids(l), blk, nb, E)
                        bs = [balance_stats(X[i]) for i in range(nb)]
                        row.update({"chunk512_" + k: float(np.mean([b[k] for b in bs])) for k in bs[0]})
                        row.update({"chunk512_null_" + k: v for k, v in null_balance(E, 512, K, trials=20).items()})
                per_layer.append(row)
            counts[(dom, ph)] = (cs_all, cs_even, cs_odd)
            summ: dict = {"windows": int(np.unique(t.win[m]).size), "rows": int(m.sum())}
            for f in fracs:
                key = f"top{int(round(f * 100))}"
                for k in ("in_sample", "heldout", "null", "chance"):
                    v = [r.get(f"{key}_{k}") for r in per_layer if r.get(f"{key}_{k}") is not None]
                    if v:
                        summ[f"{key}_{k}_mean"] = float(np.mean(v))
                        if k in ("in_sample", "heldout"):
                            summ[f"{key}_{k}_min"] = float(np.min(v))
                            summ[f"{key}_{k}_max"] = float(np.max(v))
                ins = [r[f"{key}_in_sample"] for r in per_layer]
                summ[f"{key}_layers_ge_gate"] = int(sum(x >= gate for x in ins))
            for k in ("stream_maxvio", "stream_minvio", "stream_inactive_ratio", "stream_null_maxvio", "stream_null_minvio", "stream_null_inactive_ratio",
                      "chunk512_maxvio", "chunk512_minvio", "chunk512_inactive_ratio", "chunk512_null_maxvio", "chunk512_null_minvio", "chunk512_null_inactive_ratio"):
                v_ = [r[k] for r in per_layer if k in r]
                if v_:
                    summ[k + "_mean"] = float(np.mean(v_))
                    if k in ("stream_maxvio", "chunk512_maxvio"):
                        summ[k + "_max_layer"] = float(np.max(v_))
            for k in ("entropy_norm", "effective_experts", "gini", "never_selected", "top1_share"):
                summ[k + "_mean"] = float(np.mean([r[k] for r in per_layer]))
            out["domains"][dom][ph] = {"summary": summ, "per_layer": per_layer}
    # cross-workload overlap on prefill counts (the only phase with enough data in every domain)
    doms = list(data)
    n10 = {l: top_n(0.10, int(t0.n_experts[l])) for l in range(L)}
    for ph in ("prefill", "decode"):
        pairs = {}
        for i in range(len(doms)):
            for j in range(i + 1, len(doms)):
                a, b = doms[i], doms[j]
                if (a, ph) not in counts or (b, ph) not in counts:
                    continue
                ca, cb = counts[(a, ph)][0], counts[(b, ph)][0]
                jac = [jaccard(np.argsort(-ca[l], kind="stable")[:n10[l]], np.argsort(-cb[l], kind="stable")[:n10[l]]) for l in range(L)]
                rho = [spearman(ca[l].astype(float), cb[l].astype(float)) for l in range(L)]
                pairs[f"{a}|{b}"] = {"top10_jaccard_mean": float(np.mean(jac)), "top10_jaccard_min": float(np.min(jac)),
                                     "top10_jaccard_max": float(np.max(jac)),
                                     "spearman_mean": float(np.nanmean(rho)), "spearman_min": float(np.nanmin(rho))}
        ceil = []
        for d in doms:
            if (d, ph) in counts:
                _, ce, co = counts[(d, ph)]
                ceil += [jaccard(np.argsort(-ce[l], kind="stable")[:n10[l]], np.argsort(-co[l], kind="stable")[:n10[l]]) for l in range(L)]
        out["overlap"][ph] = {"pairs": pairs, "within_domain_even_odd_jaccard_mean": float(np.mean(ceil)) if ceil else None,
                              "random_set_jaccard": float(np.mean([jaccard_chance(int(t0.n_experts[l]), n10[l]) for l in range(L)]))}
    # RT-5 verdict, applied literally to whatever model the traces came from
    v: dict = {"rule": f"top-10% share >= {gate} on >= 2 workloads", "per_workload": {}}
    for dom in doms:
        for ph in ("prefill", "decode"):
            s = out["domains"][dom].get(ph, {}).get("summary")
            if s:
                v["per_workload"][f"{dom}/{ph}"] = {"top10_mean_over_layers": s["top10_in_sample_mean"],
                                                    "top10_heldout_mean": s.get("top10_heldout_mean"),
                                                    "layers_meeting_gate": s["top10_layers_ge_gate"], "layers": L}
    for ph in ("prefill", "decode"):
        met = [d for d in doms if ph in out["domains"][d] and out["domains"][d][ph]["summary"]["top10_in_sample_mean"] >= gate]
        v[f"{ph}_workloads_meeting_gate_by_layer_mean"] = met
        v[f"{ph}_gate_met"] = len(met) >= 2
    out["rt5_verdict"] = v
    return out


def to_markdown(r: dict, title: str) -> str:
    L = ["# " + title, "", f"arch `{r['arch']}`, experts {r['n_experts']}, top-k {r['top_k']}, layers {r['layers']}; gate = top-10% share >= {r['gate']}.", "",
         "| workload | phase | rows | top-10% in-sample | top-10% held-out | top-10% null | top-25% in-sample | top-25% null | layers>=gate | H/ln E | eff. experts | gini | never |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for dom, ph in r["domains"].items():
        for p, d in ph.items():
            s = d["summary"]
            L.append(f"| {dom} | {p} | {s['rows']} | {s['top10_in_sample_mean']:.4f} [{s['top10_in_sample_min']:.3f}-{s['top10_in_sample_max']:.3f}] | "
                     f"{s.get('top10_heldout_mean', float('nan')):.4f} | {s['top10_null_mean']:.4f} | {s['top25_in_sample_mean']:.4f} | {s['top25_null_mean']:.4f} | "
                     f"{s['top10_layers_ge_gate']}/{r['layers']} | {s['entropy_norm_mean']:.4f} | {s['effective_experts_mean']:.1f} | {s['gini_mean']:.3f} | {s['never_selected_mean']:.1f} |")
    L += ["", "## Cross-workload overlap (top-10% expert sets, mean over layers)", ""]
    for ph, o in r["overlap"].items():
        L.append(f"* {ph}: within-domain even/odd ceiling Jaccard {o['within_domain_even_odd_jaccard_mean']}, random-set chance {o['random_set_jaccard']:.4f}")
        for k, v in o["pairs"].items():
            L.append(f"  * {k}: Jaccard {v['top10_jaccard_mean']:.4f} (min {v['top10_jaccard_min']:.3f}, max {v['top10_jaccard_max']:.3f}), Spearman {v['spearman_mean']:.4f} (min {v['spearman_min']:.3f})")
    L += ["", "## RT-5.1 load-balance statistics (mean over layers; null = uniform router with the same number of rows)", "",
          "| workload | phase | MaxVio stream | null | MinVio stream | null | inactive stream | null | MaxVio per-512-chunk | null | inactive per-512-chunk | null |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for dom, ph in r["domains"].items():
        for p, d in ph.items():
            s = d["summary"]
            g = lambda k: f"{s[k]:.3f}" if k in s else ""
            L.append(f"| {dom} | {p} | {g('stream_maxvio_mean')} (worst layer {s['stream_maxvio_max_layer']:.2f}) | {g('stream_null_maxvio_mean')} | {g('stream_minvio_mean')} | {g('stream_null_minvio_mean')} | "
                     f"{g('stream_inactive_ratio_mean')} | {g('stream_null_inactive_ratio_mean')} | {g('chunk512_maxvio_mean')} | {g('chunk512_null_maxvio_mean')} | {g('chunk512_inactive_ratio_mean')} | {g('chunk512_null_inactive_ratio_mean')} |")
    v = r["rt5_verdict"]
    L += ["", "## RT-5 gate", "", f"Rule: {v['rule']}.", ""]
    for ph in ("prefill", "decode"):
        L.append(f"* {ph}: workloads meeting the gate by layer-mean share = {v[ph + '_workloads_meeting_gate_by_layer_mean']} -> gate met: **{v[ph + '_gate_met']}**")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="")
    ap.add_argument("--gate", type=float, default=0.60)
    ap.add_argument("--title", default="RT-5 expert-skew report")
    a = ap.parse_args(argv)
    doms = [d for d in a.domains.split(",") if d] or sorted(tio.discover(a.results_dir))
    r = analyse(a.results_dir, doms, a.gate)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "skew_report.json").write_text(json.dumps(r, indent=1) + "\n")
    md = to_markdown(r, a.title)
    (out / "skew_report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
