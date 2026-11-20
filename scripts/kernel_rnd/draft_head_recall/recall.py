#!/usr/bin/env python3
"""DRAFT-HEAD-APPROX-RECALL offline evaluator (numpy only; run with `python -I` in a numpy venv).

Input: an act_dump --head directory (head_hidden [d,N] f32, head_w.f32 [d,V] f32, head_topk_ids/val [K,N]).
Reference: the full borrowed head recomputed in f32 from the dump (logits = H @ W^T). Sanity gate (box rule): the
recomputed argmax must reproduce the model's own argmax (head_topk_ids row 0) on >= 99.9% of rows, else the head is VOID.

Arms (no training):
  svd   truncated-SVD pre-score  P = (H V_r)(W V_r)^T  , exact logits on the top-C rows of P
        MACs/token = r*d + V*r + C*d
  clus  K-means over the head rows (2048 clusters), router = H . centroid, exact logits on the members of the top-P clusters
        MACs/token = n_clusters*d + |members(top-P)|*d   (members counted per token)
  Full head MACs/token = V*d.  removed = 1 - arm/full.
Metrics per arm: recall@1 of the full-head argmax (overall, lowest top-1-margin quartile), top-16 containment.
PASS (box): recall@1 >= 99.5% overall AND >= 98% in the low-margin quartile AND removed >= 79%.
Head verdict: VOID (sanity) | PASS (any arm passes) | FAIL.  futility = no arm with removed >= 79% reaches recall@1 >= 95%.
"""
import json
import os
import sys
import time

import numpy as np

RECALL_ALL, RECALL_LOWQ, MIN_REMOVED, SANITY_MIN, FUTILITY_RECALL = 0.995, 0.98, 0.79, 0.999, 0.95
SVD_R = (128, 256, 384, 512)
SVD_C = (1024, 4096, 16384)
N_CLUSTERS = 2048
CLUS_P = (16, 32, 64)
TOPK_REF = 16
CHUNK = 256


def load_dump(d):
    m = json.load(open(os.path.join(d, "manifest.json")))
    by = {t["key"]: t for t in m["tensors"]}

    def arr(key, dt):
        t = by[key]
        ne = t["ne"]
        a = np.fromfile(os.path.join(d, t["file"]), dtype=dt)
        return a.reshape(ne[1], ne[0])  # ne0 fastest -> (rows, ne0)

    H = arr("head_hidden", np.float32)          # (N, d)
    W = arr("head_w.f32", np.float32)           # (V, d)
    ids = arr("head_topk_ids", np.int32)        # (N, K)
    return H, W, ids, m


def kmeans(W, k, iters=8, seed=0, chunk=32768):
    """Lloyd K-means on the rows of W (Euclidean); returns centroids (k,d) f32 and labels (V,)."""
    rng = np.random.default_rng(seed)
    V = W.shape[0]
    k = min(k, V)
    C = W[rng.choice(V, k, replace=False)].copy()
    lab = np.zeros(V, np.int32)
    for _ in range(iters):
        half = 0.5 * np.einsum("ij,ij->i", C, C)
        for s in range(0, V, chunk):
            lab[s:s + chunk] = np.argmax(W[s:s + chunk] @ C.T - half, axis=1)
        cnt = np.bincount(lab, minlength=k)
        nz = cnt > 0
        order = np.argsort(lab, kind="stable")
        starts = np.concatenate(([0], np.cumsum(cnt)[:-1]))[nz]
        sums = np.add.reduceat(W[order], starts, axis=0, dtype=np.float64)  # one row per non-empty cluster, ascending label
        C[nz] = (sums / cnt[nz, None]).astype(np.float32)
        if (~nz).any():  # re-seed empty clusters
            C[~nz] = W[rng.choice(V, int((~nz).sum()), replace=False)]
    half = 0.5 * np.einsum("ij,ij->i", C, C)
    for s in range(0, V, chunk):
        lab[s:s + chunk] = np.argmax(W[s:s + chunk] @ C.T - half, axis=1)
    return C, lab


def svd_basis(W, rmax):
    G = np.zeros((W.shape[1], W.shape[1]), np.float64)
    for s in range(0, W.shape[0], 32768):
        Wc = W[s:s + 32768].astype(np.float64)
        G += Wc.T @ Wc
    ev, vec = np.linalg.eigh(G)
    order = np.argsort(ev)[::-1][:rmax]
    return vec[:, order].astype(np.float32)  # (d, rmax)


def evaluate(H, W, ids, svd_r=SVD_R, svd_c=SVD_C, n_clusters=N_CLUSTERS, clus_p=CLUS_P, kmeans_iters=8, seed=0, log=print):
    N, d = H.shape
    V = W.shape[0]
    full_macs = V * d
    t0 = time.time()
    ref = np.empty(N, np.int64)
    margin = np.empty(N, np.float32)
    top16 = np.empty((N, min(TOPK_REF, V)), np.int64)
    for s in range(0, N, CHUNK):
        L = H[s:s + CHUNK] @ W.T
        k = top16.shape[1]
        part = np.argpartition(-L, k - 1, axis=1)[:, :k]
        pv = np.take_along_axis(L, part, 1)
        o = np.argsort(-pv, axis=1, kind="stable")
        top16[s:s + CHUNK] = np.take_along_axis(part, o, 1)
        sv = np.take_along_axis(pv, o, 1)
        ref[s:s + CHUNK] = top16[s:s + CHUNK, 0]
        margin[s:s + CHUNK] = sv[:, 0] - sv[:, 1]
    sanity = float(np.mean(ref == ids[:, 0]))
    lowq = margin <= np.quantile(margin, 0.25)
    log(f"full head recomputed: sanity {sanity:.5f} ({time.time() - t0:.1f}s)")
    arms = []

    def summarize(name, params, hit1, cont16, macs):
        removed = 1.0 - float(np.mean(macs)) / full_macs
        r_all, r_low = float(np.mean(hit1)), float(np.mean(hit1[lowq]))
        a = {"arm": name, **params, "recall1": r_all, "recall1_lowq": r_low, "top16_containment": float(np.mean(cont16)),
             "macs_removed": removed, "pass": bool(r_all >= RECALL_ALL and r_low >= RECALL_LOWQ and removed >= MIN_REMOVED)}
        arms.append(a)
        log(json.dumps(a))

    svd_r = [r for r in svd_r if r < d]
    if svd_r:
        B = svd_basis(W, max(svd_r))
        for r in svd_r:
            Br = B[:, :r]
            A = W @ Br                                  # (V, r)
            cs = [c for c in svd_c if c < V]
            hit = {c: np.zeros(N, bool) for c in cs}
            cont = {c: np.zeros(N) for c in cs}
            for s in range(0, N, CHUNK):
                P = (H[s:s + CHUNK] @ Br) @ A.T          # (B, V)
                part = np.partition(-P, [c - 1 for c in cs], axis=1)
                rows = np.arange(P.shape[0])
                pa = P[rows, ref[s:s + CHUNK]]
                p16 = np.take_along_axis(P, top16[s:s + CHUNK], 1)
                for c in cs:
                    thr = -part[:, c - 1]
                    hit[c][s:s + CHUNK] = pa >= thr
                    cont[c][s:s + CHUNK] = np.mean(p16 >= thr[:, None], axis=1)
            for c in cs:
                summarize("svd", {"rank": r, "C": c}, hit[c], cont[c], np.full(N, r * d + V * r + c * d, np.float64))
    if n_clusters and n_clusters < V:
        t1 = time.time()
        Cn, lab = kmeans(W, n_clusters, kmeans_iters, seed)
        sizes = np.bincount(lab, minlength=Cn.shape[0])
        log(f"kmeans k={Cn.shape[0]} iters={kmeans_iters} ({time.time() - t1:.1f}s) size min/med/max {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")
        for p in [p for p in clus_p if p < Cn.shape[0]]:
            hit = np.zeros(N, bool)
            cont = np.zeros(N)
            macs = np.zeros(N)
            for s in range(0, N, CHUNK):
                S = H[s:s + CHUNK] @ Cn.T
                top = np.argpartition(-S, p - 1, axis=1)[:, :p]
                sel = np.zeros(S.shape, bool)
                np.put_along_axis(sel, top, True, 1)
                rows = np.arange(S.shape[0])
                hit[s:s + CHUNK] = sel[rows, lab[ref[s:s + CHUNK]]]
                cont[s:s + CHUNK] = np.mean(np.take_along_axis(sel, lab[top16[s:s + CHUNK]], 1), axis=1)
                macs[s:s + CHUNK] = (Cn.shape[0] + sizes[top].sum(axis=1)) * d
            summarize("clus", {"clusters": int(Cn.shape[0]), "top_p": p}, hit, cont, macs)
    elig = [a for a in arms if a["macs_removed"] >= MIN_REMOVED]
    best = max(elig, key=lambda a: (a["pass"], a["recall1"], a["macs_removed"]), default=None)
    if sanity < SANITY_MIN:
        verdict = f"VOID(sanity {sanity:.4f} < {SANITY_MIN})"
    elif any(a["pass"] for a in arms):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    futility = not any(a["recall1"] >= FUTILITY_RECALL for a in elig)
    return {"verdict": verdict, "sanity_argmax_match": sanity, "n_rows": N, "d": d, "vocab": V, "full_macs_per_token": full_macs,
            "lowq_margin_threshold": float(np.quantile(margin, 0.25)), "arms": arms, "best_eligible_arm": best, "futility": futility,
            "rule": {"recall1_all": RECALL_ALL, "recall1_lowq": RECALL_LOWQ, "min_macs_removed": MIN_REMOVED, "sanity_min": SANITY_MIN,
                     "futility_recall1": FUTILITY_RECALL},
            "elapsed_s": round(time.time() - t0, 1)}


def main(argv):
    if len(argv) < 3:
        print("usage: recall.py <act_dump --head dir> <out.json> [--kmeans-iters N]", file=sys.stderr)
        return 1
    it = int(argv[argv.index("--kmeans-iters") + 1]) if "--kmeans-iters" in argv else 8
    H, W, ids, m = load_dump(argv[1])
    print(f"loaded {argv[1]}: H {H.shape} W {W.shape} ids {ids.shape}", flush=True)
    r = evaluate(H, W, ids, kmeans_iters=it, log=lambda s: print(s, flush=True))
    r["dump"] = argv[1]
    r["model"] = m.get("model")
    tmp = argv[2] + ".tmp"
    json.dump(r, open(tmp, "w"), indent=1)
    os.replace(tmp, argv[2])
    print(json.dumps({"verdict": r["verdict"], "best": r["best_eligible_arm"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
