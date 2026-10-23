#!/usr/bin/env python3
"""RT-7: residency hit-rate curve h(k) and stationarity of the hot-expert set, offline from RT-1 traces.

h(k): a STATIC ranking learned on workload A (prefill traces of A) fixes which experts are resident; k experts per layer on
average, allocated greedily across layers (the k*L globally highest per-layer selection shares); h is the share of workload B's
top-K selections those residents cover. Reported for B = prefill and B = decode, for every ordered pair A->B (A != B is the
"learned on domain A, scored on domain B" case), plus same-domain references: A prefill -> A decode, A even windows -> A odd
windows, a uniform per-layer allocation, an in-sample oracle for B (upper bound), and the random-residency chance k/E.

Stationarity: Jaccard of the top-n expert sets between consecutive blocks of 1024 rows (prefill: two windows; decode: the
decode rows concatenated), against the random-set chance and the within-block random-half ceiling, and the share of block
t+1's selections covered by block t's top-n set.

Gate (RT-7): open HIP MXFP4 expert-cache work (CAFE-7) only if h(53) >= 0.30 on >= 2 workloads (and the HIP resident MXFP4 MoE
kernel check passes); h < 0.23 closes it. The verdict block applies this literally for the traced model; k is also reported
rescaled to the same FRACTION of the expert pool, because 53 of 384 (DS41) is not 53 of 512. A trace of another model is a PROXY.

Usage: residency_curve.py RESULTS_DIR --out OUT_DIR [--domains ...] [--ks 39,53,63] [--ref-experts 384]
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

CURVE_KS = (8, 16, 32, 39, 53, 63, 84, 128, 192, 256)


def counts_by_layer(t: tio.Trace, mask: np.ndarray) -> np.ndarray:
    """[L, E] selection counts (E = max over layers; layers are uniform in the RT-1 fleet traces)."""
    E = int(t.n_experts.max())
    out = np.zeros((t.L, E), np.int64)
    for l in range(t.L):
        out[l, : int(t.n_experts[l])] = tio.layer_counts(t.layer_ids(l)[mask], int(t.n_experts[l]))
    return out


def resident_set(cnt_a: np.ndarray, k: float, mode: str = "greedy") -> np.ndarray:
    """Boolean [L, E] residency mask holding k experts per layer on average."""
    L, E = cnt_a.shape
    tot = cnt_a.sum(1, keepdims=True).astype(np.float64)
    p = cnt_a / np.maximum(tot, 1)
    res = np.zeros((L, E), bool)
    if mode == "uniform":
        top = np.argsort(-p, axis=1, kind="stable")[:, : int(round(k))]
        np.put_along_axis(res, top, True, axis=1)
        return res
    budget = int(round(k * L))
    order = np.argsort(-p.ravel(), kind="stable")[:budget]
    res.ravel()[order] = True
    return res


def hit_rate(res: np.ndarray, cnt_b: np.ndarray) -> float:
    tot = cnt_b.sum()
    return float(cnt_b[res].sum() / tot) if tot else float("nan")


def block_stability(t: tio.Trace, mask: np.ndarray, n: int, rows_per_block: int = 1024, rng=None) -> dict:
    sub = t.select(mask)
    nb = sub.R // rows_per_block
    if nb < 2:
        return {"blocks": int(nb)}
    sub = sub.select(np.arange(sub.R) < nb * rows_per_block)
    blk = (np.arange(sub.R) // rows_per_block).astype(np.int64)
    rng = rng or np.random.default_rng(7)
    jac, cov, ceil_, oracle_cov = [], [], [], []
    for l in range(sub.L):
        E = int(sub.n_experts[l])
        x = sub.layer_ids(l)
        X = tio.layer_loads(x, blk, nb, E).astype(np.float64)
        tops = np.argsort(-X, axis=1, kind="stable")[:, :n]
        sets = np.zeros((nb, E), bool)
        np.put_along_axis(sets, tops, True, axis=1)
        inter = (sets[1:] & sets[:-1]).sum(1)
        jac.append(float(np.mean(inter / (2 * n - inter))))
        cov.append(float(np.mean([X[i + 1][sets[i]].sum() / X[i + 1].sum() for i in range(nb - 1)])))
        oracle_cov.append(float(np.mean([X[i + 1][sets[i + 1]].sum() / X[i + 1].sum() for i in range(nb - 1)])))
        # ceiling: random split of one block into two halves
        c_ = []
        for b in range(min(nb, 8)):
            rows = np.flatnonzero(blk == b)
            rng.shuffle(rows)
            h1, h2 = rows[: rows.size // 2], rows[rows.size // 2:]
            a = tio.layer_counts(x[h1], E)
            c = tio.layer_counts(x[h2], E)
            sa, sc = set(np.argsort(-a, kind="stable")[:n].tolist()), set(np.argsort(-c, kind="stable")[:n].tolist())
            c_.append(len(sa & sc) / len(sa | sc))
        ceil_.append(float(np.mean(c_)))
    En = int(sub.n_experts[0])
    x_ = n * n / En
    return {"blocks": int(nb), "rows_per_block": rows_per_block, "n": n,
            "consecutive_top_jaccard_mean": float(np.mean(jac)), "consecutive_top_jaccard_min_layer": float(np.min(jac)),
            "random_set_jaccard": x_ / (2 * n - x_), "within_block_half_split_ceiling": float(np.mean(ceil_)),
            "next_block_coverage_by_prev_top": float(np.mean(cov)), "next_block_coverage_oracle": float(np.mean(oracle_cov))}


def analyse(results_dir, domains, ks, ref_experts: int, gate_k: int = 53) -> dict:
    data = {d: tio.load_domain(results_dir, d) for d in domains}
    t0 = next(iter(data.values()))
    E = int(t0.n_experts.max())
    scale = E / ref_experts
    kp = sorted({int(math.ceil(k * scale)) for k in ks})
    kp = sorted(set(kp) | {int(math.ceil(gate_k * scale))})
    all_ks = sorted(set(CURVE_KS) | set(ks) | set(kp) | {gate_k})
    cnt = {}
    for d, t in data.items():
        cnt[(d, "prefill")] = counts_by_layer(t, t.phase == 0)
        cnt[(d, "decode")] = counts_by_layer(t, t.phase == 1)
        cnt[(d, "even")] = counts_by_layer(t, (t.phase == 0) & (t.win % 2 == 0))
        cnt[(d, "odd")] = counts_by_layer(t, (t.phase == 0) & (t.win % 2 == 1))
    out: dict = {"arch": t0.arch, "n_experts": E, "top_k": sorted(set(t0.top_k.tolist())), "layers": t0.L, "ks": ks,
                 "ks_same_fraction_of_pool": kp, "ref_experts": ref_experts, "curves": {}, "stationarity": {}}
    for B in domains:
        for phase in ("prefill", "decode"):
            cb = cnt[(B, phase)]
            rec: dict = {"selections": int(cb.sum()), "chance": {str(k): k / E for k in all_ks}, "learned_on": {}}
            for A in domains:
                if A == B and phase == "prefill":
                    ca, cbt, name = cnt[(A, "even")], cnt[(A, "odd")], f"{A}_even->{B}_odd(prefill)"
                    rec["learned_on"][name] = {str(k): hit_rate(resident_set(ca, k), cbt) for k in all_ks}
                    continue
                ca = cnt[(A, "prefill")]
                key = A if A != B else f"{A}(same-domain prefill->decode)"
                rec["learned_on"][key] = {str(k): hit_rate(resident_set(ca, k), cb) for k in all_ks}
            rec["uniform_alloc_learned_on_same_domain_prefill"] = {str(k): hit_rate(resident_set(cnt[(B, "prefill")], k, "uniform"), cb) for k in all_ks}
            rec["oracle_in_sample"] = {str(k): hit_rate(resident_set(cb, k), cb) for k in all_ks}
            cross = [v for a, v in rec["learned_on"].items() if a in domains and a != B]
            rec["cross_domain_worst"] = {str(k): min(v[str(k)] for v in cross) for k in all_ks} if cross else {}
            rec["cross_domain_mean"] = {str(k): float(np.mean([v[str(k)] for v in cross])) for k in all_ks} if cross else {}
            out["curves"][f"{B}/{phase}"] = rec
    n_s = [gate_k, int(math.ceil(gate_k * scale))]
    for d, t in data.items():
        out["stationarity"][d] = {}
        for ph, m in (("prefill", t.phase == 0), ("decode", t.phase == 1)):
            out["stationarity"][d][ph] = {str(n): block_stability(t, m, n) for n in sorted(set(n_s))}
    # verdict
    v: dict = {"rule": "open CAFE-7 only if h(53) >= 0.30 on >= 2 workloads; h < 0.23 closes it", "model_is_proxy": True, "by_k": {}}
    for k in sorted(set([gate_k, int(math.ceil(gate_k * scale))])):
        rows = {}
        for ph in ("prefill", "decode"):
            def same(B):
                lo = out["curves"][f"{B}/{ph}"]["learned_on"]
                key = f"{B}_even->{B}_odd(prefill)" if ph == "prefill" else f"{B}(same-domain prefill->decode)"
                return lo[key][str(k)]
            rows[ph] = {B: {"cross_domain_worst": out["curves"][f"{B}/{ph}"]["cross_domain_worst"][str(k)],
                            "cross_domain_mean": out["curves"][f"{B}/{ph}"]["cross_domain_mean"][str(k)],
                            "same_domain_heldout": same(B),
                            "chance": k / E} for B in domains}
            rows[ph]["workloads_ge_0.30_same_domain"] = [B for B in domains if rows[ph][B]["same_domain_heldout"] >= 0.30]
            w = [B for B in domains if rows[ph][B]["cross_domain_worst"] >= 0.30]
            m = [B for B in domains if rows[ph][B]["cross_domain_mean"] >= 0.30]
            rows[ph]["workloads_ge_0.30_worst_case"] = w
            rows[ph]["workloads_ge_0.30_mean_over_sources"] = m
            rows[ph]["workloads_lt_0.23_worst_case"] = [B for B in domains if rows[ph][B]["cross_domain_worst"] < 0.23]
        v["by_k"][str(k)] = rows
    out["rt7_verdict"] = v
    return out


def to_markdown(r: dict, title: str) -> str:
    E = r["n_experts"]
    L = ["# " + title, "", f"arch `{r['arch']}`, {E} experts, top-k {r['top_k']}, {r['layers']} layers. Static residency learned on A (prefill), greedy cross-layer allocation, scored on B. "
         f"k rescaled to the same fraction of the pool as {r['ref_experts']}-expert DS41: {r['ks_same_fraction_of_pool']}.", ""]
    shown = [39, 53, 63] + [k for k in r["ks_same_fraction_of_pool"] if k not in (39, 53, 63)] + [128, 256]
    for phase in ("prefill", "decode"):
        L += [f"## B = {phase}", "", "| B | learned on | " + " | ".join(f"h({k})" for k in shown) + " |", "|---|---|" + "---|" * len(shown)]
        for key, rec in r["curves"].items():
            B, ph = key.split("/")
            if ph != phase:
                continue
            for a, h in rec["learned_on"].items():
                L.append(f"| {B} | {a} | " + " | ".join(f"{h[str(k)]:.3f}" for k in shown) + " |")
            L.append(f"| {B} | *oracle in-sample (bound)* | " + " | ".join(f"{rec['oracle_in_sample'][str(k)]:.3f}" for k in shown) + " |")
            L.append(f"| {B} | *chance k/E* | " + " | ".join(f"{rec['chance'][str(k)]:.3f}" for k in shown) + " |")
        L.append("")
    L += ["## Stationarity (top-n expert set, consecutive 1024-row blocks)", "", "| workload | phase | n | blocks | consecutive Jaccard | random-set | half-split ceiling | next-block coverage | oracle coverage |", "|---|---|---|---|---|---|---|---|---|"]
    for d, ph in r["stationarity"].items():
        for p, ns in ph.items():
            for n, s in ns.items():
                if s.get("blocks", 0) < 2:
                    L.append(f"| {d} | {p} | {n} | {s.get('blocks')} | (too few blocks) | | | | |")
                    continue
                L.append(f"| {d} | {p} | {n} | {s['blocks']} | {s['consecutive_top_jaccard_mean']:.3f} | {s['random_set_jaccard']:.3f} | {s['within_block_half_split_ceiling']:.3f} | {s['next_block_coverage_by_prev_top']:.3f} | {s['next_block_coverage_oracle']:.3f} |")
    L += ["", "## RT-7 gate (applied literally; PROXY model)", ""]
    for k, rows in r["rt7_verdict"]["by_k"].items():
        for ph, x in rows.items():
            L.append(f"* k={k} {ph}: same-domain held-out >= 0.30 on {x['workloads_ge_0.30_same_domain']}; worst-case-source >= 0.30 on {x['workloads_ge_0.30_worst_case']}; mean-over-sources >= 0.30 on {x['workloads_ge_0.30_mean_over_sources']}; worst-case < 0.23 on {x['workloads_lt_0.23_worst_case']}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="")
    ap.add_argument("--ks", default="39,53,63")
    ap.add_argument("--ref-experts", type=int, default=384)
    ap.add_argument("--title", default="RT-7 residency hit-rate curve")
    a = ap.parse_args(argv)
    doms = [d for d in a.domains.split(",") if d] or sorted(tio.discover(a.results_dir))
    r = analyse(a.results_dir, doms, [int(x) for x in a.ks.split(",")], a.ref_experts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "residency_curve.json").write_text(json.dumps(r, indent=1) + "\n")
    md = to_markdown(r, a.title)
    (out / "residency_curve.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
