#!/usr/bin/env python3
"""DRAFT-HEAD-SVD-CPU offline factors for the Q38FN shared head (output.weight, borrowed by qwen4exp graph_mtp).

Arithmetic = DRAFT-HEAD-APPROX-RECALL r2q0 amendment A1 (recall.py --arith iqk_q6k, copied here verbatim):
  W_eff = iqk_convert_q6_k_q8_0_r8(dequant Q6_K)   (per-32 Q8_0, fp16 scale)
  B     = top-r right singular vectors of W_eff     (recall.svd_basis: eigh of W_eff^T W_eff in f64)   (d, r)
  A     = W_eff @ B                                                                                       (V, r)
Runtime pre-score: P = A (B^T h).  GGUF layout (ggml ne): mtp_svd.v = B^T as ne [d, r] F32; mtp_svd.a = A as ne [r, V].
A's storage type is chosen by the recall rule on the r2q0 dump (H 4096 x 2560): Q8_0 if recall@1 >= 0.995 AND low-margin
quartile >= 0.98 at C, else F16 (F32 reference reported too).  Pre-score activation as the runtime sees it: h f32 (V is F32),
t = B^T h then t -> f16 (F16 A) or t -> Q8_0 (Q8_0 A, ggml quantize_row_q8_0).
usage: make_svd_factors.py <dump dir> <out dir> [rank 256] [C 16384]
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import recall as R  # noqa: E402
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

QB = 32


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def q8_0_dequant(X):
    """ggml quantize_row_q8_0 reference round-trip along the last axis (fp16 d = max/127, q = round(x/d))."""
    s = X.shape
    B = X.reshape(-1, QB).astype(np.float32)
    mx = np.max(np.abs(B), axis=1, keepdims=True)
    d = (mx / np.float32(127)).astype(np.float32)
    inv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, 1), 0)
    q = np.clip(np.round(B * inv), -128, 127)
    return (q * d.astype(np.float16).astype(np.float32)).reshape(s).astype(np.float32)


def recall_at(Hraw, Hq, W_eff, B, A, C, tmode):
    N = Hraw.shape[0]
    ref = np.empty(N, np.int64)
    margin = np.empty(N, np.float32)
    hit = np.zeros(N, bool)
    CH = 256
    for s in range(0, N, CH):
        L = Hq[s:s + CH] @ W_eff.T
        p2 = np.argpartition(-L, 1, axis=1)[:, :2]
        v2 = np.take_along_axis(L, p2, 1)
        o = np.argsort(-v2, axis=1)
        ref[s:s + CH] = np.take_along_axis(p2, o, 1)[:, 0]
        sv = np.take_along_axis(v2, o, 1)
        margin[s:s + CH] = sv[:, 0] - sv[:, 1]
        t = Hraw[s:s + CH] @ B
        if tmode == "f16":
            t = t.astype(np.float16).astype(np.float32)
        elif tmode == "q8_0":
            t = q8_0_dequant(t)
        P = t @ A.T
        thr = -np.partition(-P, C - 1, axis=1)[:, C - 1]
        hit[s:s + CH] = P[np.arange(P.shape[0]), ref[s:s + CH]] >= thr
    lowq = margin <= np.quantile(margin, 0.25)
    return {"recall1": float(hit.mean()), "recall1_lowq": float(hit[lowq].mean())}


def main(argv):
    dump, out = argv[1], argv[2]
    r = int(argv[3]) if len(argv) > 3 else 256
    C = int(argv[4]) if len(argv) > 4 else 16384
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    H, W, ids, man, vals = R.load_dump(dump)
    V, d = W.shape
    W_eff = R.quant_weight(W, "iqk_q6k")
    del W
    Hq = R.quant_act(H, "iqk_q6k")
    B = R.svd_basis(W_eff, r)                      # (d, r) f32
    A = (W_eff @ B).astype(np.float32)             # (V, r)
    print(f"factors r={r} V={V} d={d} ({time.time() - t0:.1f}s)", flush=True)
    arms = {}
    for name, Aq, tm in (("F32", A, "f32"), ("F16", A.astype(np.float16).astype(np.float32), "f16"), ("Q8_0", q8_0_dequant(A), "q8_0")):
        arms[name] = recall_at(H, Hq, W_eff, B, Aq, C, tm)
        print(name, arms[name], f"({time.time() - t0:.1f}s)", flush=True)
    ok = lambda x: x["recall1"] >= 0.995 and x["recall1_lowq"] >= 0.98  # noqa: E731
    atype = "Q8_0" if ok(arms["Q8_0"]) else "F16"
    path = os.path.join(out, f"q38fn-mtp-head-svd-r{r}-{atype.lower()}.gguf")
    w = gguf.GGUFWriter(path, "mtp_svd")
    w.add_uint32("mtp_svd.rank", r)
    w.add_uint32("mtp_svd.c", C)
    w.add_uint32("mtp_svd.n_embd", d)
    w.add_uint32("mtp_svd.n_vocab", V)
    w.add_string("mtp_svd.source_model", man["model"])
    w.add_string("mtp_svd.source_tensor", "output.weight")
    w.add_string("mtp_svd.arith", "iqk_q6k (DRAFT-HEAD-APPROX-RECALL A1): W_eff = Q6_K -> per-32 Q8_0; B = eigh(W_eff^T W_eff) top-r; A = W_eff B")
    w.add_tensor("mtp_svd.v", np.ascontiguousarray(B.T))               # numpy (r, d) -> ne [d, r]
    if atype == "Q8_0":
        qa = quants.quantize(np.ascontiguousarray(A), gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor("mtp_svd.a", qa, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    else:
        w.add_tensor("mtp_svd.a", np.ascontiguousarray(A.astype(np.float16)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    res = {"path": path, "sha256": sha256(path), "rank": r, "C": C, "a_type": atype, "n_embd": d, "n_vocab": V,
           "recall_rule": "recall1 >= 0.995 and recall1_lowq >= 0.98 at C (A1 iqk_q6k reference, 4096 dump rows)",
           "recall": arms, "source_dump": dump, "source_model": man["model"],
           "head_w_f32_sha256": sha256(os.path.join(dump, "head_w.f32.bin")),
           "recall_py_sha256": sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall.py")),
           "wall_s": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(out, "FACTORS.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
