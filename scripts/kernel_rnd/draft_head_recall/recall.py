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

AMENDMENT A1 (2026-11-21, before the r2 rerun; r1q0 came back VOID at all-rows sanity 0.99805):
  Faithful exact head.  The C14 CPU path never multiplies f32 x f32: with GGML_IQK=1 and a 4096-row batch it quantizes
  the activations and (for Q6_K) re-quantizes the weight before the integer dot products (iqk_dispatch.cpp
  iqk_activation_type, iqk_mul_mat.cpp is_dequant_better, iqk_gemm_kquants.cpp iqk_convert_q6_k_q8_0_r8,
  iqk_quantize_min.cpp quantize_row_q8_2_x4).  `--arith` reproduces that arithmetic:
    f32       H @ W^T (r1 behaviour)
    iqk_q6k   Q6_K head, Ny >= 64: weight -> Q8_0 per 32 (fp16 scale = maxabs/127, codes rint(127 w/maxabs)),
              activation -> Q8_2_X4 (bf16 scale = bf16(maxabs/127), codes rint(x / d), saturated to int8)
    iqk_q8_0  Q8_0 head (AVX512-VNNI): weight Q8_0 -> Q8_0_R8 lossless repack, activation -> Q8_2_X4_H
              (ggml quantize_row_q8_0: codes rint(x * 127/maxabs), fp16 d)
  Because every scale is per 32-block, sum_b dA_b dW_b <qA_b,qW_b> == dequant(qA) @ dequant(qW)^T exactly (only the
  f32 accumulation order differs), so the faithful head is an f32 GEMM of the dequantized operands.  Reference logits,
  recall and every arm are computed on these operands.
  Sanity (still proves the dump): rows whose MODEL top-2 margin (head_topk_val[0]-[1]) exceeds SANITY_EPS logits must
  reproduce the model argmax on >= 99.9%, AND those rows must be >= 99% of all rows (the filter may not hide a broken
  dump).  Reported: all-rows agreement, margin-filtered agreement, excluded-row fraction, max |top-1 logit| diff.
"""
import json
import os
import sys
import time

import numpy as np

RECALL_ALL, RECALL_LOWQ, MIN_REMOVED, SANITY_MIN, FUTILITY_RECALL = 0.995, 0.98, 0.79, 0.999, 0.95
SANITY_EPS, SANITY_MIN_KEPT = 1e-2, 0.99
ARITH = ("f32", "iqk_q6k", "iqk_q8_0")
QB = 32
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
    vals = arr("head_topk_val", np.float32) if "head_topk_val" in by else None
    return H, W, ids, m, vals


def _bf16(x):
    """f32 -> bf16 -> f32, round-to-nearest-even (GGML_FP32_TO_BF16 for finite inputs)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (u.astype(np.uint32) << 16).view(np.float32)


def act_codes(X, mode):
    """(codes int8-valued f32 (N,d/32,32), scales (N,d/32,1)) of the iqk activation quantizer. d % 32 == 0."""
    N, d = X.shape
    B = X.reshape(N, d // QB, QB).astype(np.float32)
    mx = np.max(np.abs(B), axis=2, keepdims=True)
    if mode == "iqk_q6k":      # quantize_row_q8_2_x4: d = bf16(max/127), q = rint(x * (1/d))
        dq = _bf16(mx / np.float32(127))
        inv = np.where(dq > 0, np.float32(1) / np.where(dq > 0, dq, 1), 0).astype(np.float32)
    elif mode == "iqk_q8_0":   # quantize_row_q8_0 (x86): fp16 d = max/127, q = rint(x * 127/max)
        dq = (mx / np.float32(127)).astype(np.float16).astype(np.float32)
        inv = np.where(mx > 0, np.float32(127) / np.where(mx > 0, mx, 1), 0).astype(np.float32)
    else:
        raise ValueError(mode)
    return np.clip(np.rint(B * inv), -128, 127), dq


def quant_act(X, mode):
    """Dequantized activations as the iqk kernel sees them."""
    if mode == "f32":
        return X
    q, dq = act_codes(X, mode)
    return (q * dq).reshape(X.shape).astype(np.float32)


def weight_codes(W, mode):
    """(codes, fp16 scales) of iqk_convert_q6_k_q8_0_r8 for iqk_q6k (per-32 maxabs/127)."""
    B = W.reshape(W.shape[0], -1, QB)
    mx = np.max(np.abs(B), axis=2, keepdims=True) / np.float32(127)
    inv = np.where(mx > 1e-30, np.float32(1) / np.where(mx > 1e-30, mx, 1), 0).astype(np.float32)
    return np.clip(np.rint(B * inv), -127, 127), mx.astype(np.float16).astype(np.float32)


def quant_weight(W, mode, chunk=16384):
    """Effective head weight. iqk_q6k: iqk_convert_q6_k_q8_0_r8 (per-32 Q8_0 with fp16 scale); others unchanged."""
    if mode != "iqk_q6k":
        return W
    V, d = W.shape
    out = np.empty_like(W)
    for s in range(0, V, chunk):
        q, sc = weight_codes(W[s:s + chunk], mode)
        out[s:s + chunk] = (q * sc).reshape(-1, d)
    return out


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


def evaluate(H, W, ids, vals=None, arith="f32", svd_r=SVD_R, svd_c=SVD_C, n_clusters=N_CLUSTERS, clus_p=CLUS_P, kmeans_iters=8, seed=0, log=print):
    if arith not in ARITH:
        raise ValueError(f"arith {arith!r} not in {ARITH}")
    H, W = quant_act(H, arith), quant_weight(W, arith)
    N, d = H.shape
    V = W.shape[0]
    full_macs = V * d
    t0 = time.time()
    ref = np.empty(N, np.int64)
    top1v = np.empty(N, np.float32)
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
        top1v[s:s + CHUNK] = sv[:, 0]
    agree = ref == ids[:, 0]
    sanity_all = float(np.mean(agree))
    if vals is not None:
        model_margin = vals[:, 0] - vals[:, 1]
        keep = model_margin > SANITY_EPS
        max_top1_diff = float(np.max(np.abs(top1v - vals[:, 0])))
    else:  # no model logits: fall back to the recomputed margin
        keep = margin > SANITY_EPS
        max_top1_diff = None
    kept = float(np.mean(keep))
    sanity = float(np.mean(agree[keep])) if keep.any() else 0.0
    sanity_ok = sanity >= SANITY_MIN and kept >= SANITY_MIN_KEPT
    lowq = margin <= np.quantile(margin, 0.25)
    log(f"full head recomputed ({arith}): sanity all-rows {sanity_all:.5f}, margin>{SANITY_EPS} {sanity:.5f} on {kept:.4f} of rows, "
        f"max|top1 diff| {max_top1_diff} ({time.time() - t0:.1f}s)")
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
    if not sanity_ok:
        verdict = f"VOID(sanity margin>{SANITY_EPS} {sanity:.4f} (need {SANITY_MIN}) on {kept:.4f} of rows (need {SANITY_MIN_KEPT}))"
    elif any(a["pass"] for a in arms):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    futility = not any(a["recall1"] >= FUTILITY_RECALL for a in elig)
    return {"verdict": verdict, "arith": arith, "sanity_argmax_match": sanity, "sanity_argmax_match_all_rows": sanity_all,
            "sanity_rows_kept": kept, "sanity_eps": SANITY_EPS, "max_top1_logit_diff": max_top1_diff, "n_rows": N, "d": d, "vocab": V, "full_macs_per_token": full_macs,
            "lowq_margin_threshold": float(np.quantile(margin, 0.25)), "arms": arms, "best_eligible_arm": best, "futility": futility,
            "rule": {"recall1_all": RECALL_ALL, "recall1_lowq": RECALL_LOWQ, "min_macs_removed": MIN_REMOVED, "sanity_min": SANITY_MIN, "sanity_eps": SANITY_EPS, "sanity_min_kept": SANITY_MIN_KEPT,
                     "futility_recall1": FUTILITY_RECALL},
            "elapsed_s": round(time.time() - t0, 1)}


def main(argv):
    if len(argv) < 3:
        print("usage: recall.py <act_dump --head dir> <out.json> [--kmeans-iters N] [--arith f32|iqk_q6k|iqk_q8_0]", file=sys.stderr)
        return 1
    it = int(argv[argv.index("--kmeans-iters") + 1]) if "--kmeans-iters" in argv else 8
    arith = argv[argv.index("--arith") + 1] if "--arith" in argv else "f32"
    H, W, ids, m, vals = load_dump(argv[1])
    print(f"loaded {argv[1]}: H {H.shape} W {W.shape} ids {ids.shape}", flush=True)
    r = evaluate(H, W, ids, vals, arith=arith, kmeans_iters=it, log=lambda s: print(s, flush=True))
    r["dump"] = argv[1]
    r["model"] = m.get("model")
    tmp = argv[2] + ".tmp"
    json.dump(r, open(tmp, "w"), indent=1)
    os.replace(tmp, argv[2])
    print(json.dumps({"verdict": r["verdict"], "best": r["best_eligible_arm"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
