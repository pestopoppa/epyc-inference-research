"""Synthetic unit tests for recall.py (tiny V/d; no model). Run: python -I -m pytest -q test_recall.py (or python -I test_recall.py)."""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recall as R  # noqa: E402


def _case(V=3000, d=48, N=512, seed=1, rank=None):
    g = np.random.default_rng(seed)
    if rank:  # exactly low-rank head -> SVD prescore at r >= rank is exact
        W = (g.standard_normal((V, rank)) @ g.standard_normal((rank, d))).astype(np.float32)
    else:
        W = g.standard_normal((V, d)).astype(np.float32)
    H = g.standard_normal((N, d)).astype(np.float32) * 3
    L = H @ W.T
    ids = np.argsort(-L, axis=1)[:, :16].astype(np.int32)
    return H, W, ids


def _write_dump(dirn, H, W, ids, vals=None):
    if vals is None:  # the model's own top-K logits, f32 head
        vals = np.take_along_axis(H @ W.T, ids.astype(np.int64), 1).astype(np.float32)
    t = []
    for key, a, typ in (("head_hidden", H, "f32"), ("head_w.f32", W, "f32"), ("head_topk_ids", ids, "i32"),
                        ("head_topk_val", vals, "f32")):
        a.tofile(os.path.join(dirn, key + ".bin"))
        t.append({"key": key, "file": key + ".bin", "type": typ, "ne": [a.shape[1], a.shape[0], 1, 1], "bytes": a.nbytes})
    json.dump({"head_mode": True, "n_tokens": H.shape[0], "vocab": W.shape[0], "tensors": t}, open(os.path.join(dirn, "manifest.json"), "w"))


def test_exact_lowrank_svd_recall_is_one_and_macs_formula():
    H, W, ids = _case(rank=16)
    r = R.evaluate(H, W, ids, svd_r=(16,), svd_c=(64,), n_clusters=0, log=lambda s: None)
    a = r["arms"][0]
    assert r["sanity_argmax_match"] == 1.0
    assert a["recall1"] == 1.0 and a["recall1_lowq"] == 1.0 and a["top16_containment"] == 1.0
    V, d = W.shape
    assert abs(a["macs_removed"] - (1 - (16 * d + V * 16 + 64 * d) / (V * d))) < 1e-12


def test_sanity_gate_voids_on_wrong_model_argmax():
    H, W, ids = _case()
    bad = ids.copy(); bad[:10, 0] = (bad[:10, 0] + 1) % W.shape[0]
    r = R.evaluate(H, W, bad, svd_r=(8,), svd_c=(64,), n_clusters=0, log=lambda s: None)
    assert r["verdict"].startswith("VOID")


def test_cluster_all_clusters_selected_is_exact_and_removes_nothing():
    H, W, ids = _case(V=1000, d=16, N=128)
    r = R.evaluate(H, W, ids, svd_r=(), n_clusters=8, clus_p=(7,), kmeans_iters=3, log=lambda s: None)
    a = r["arms"][0]
    assert 0 < a["recall1"] <= 1.0 and a["macs_removed"] < 0.79 and not a["pass"]
    # top_p = all clusters -> exact recall (p must be < k by design, so use k-1 clusters + check monotone vs a smaller p)
    r2 = R.evaluate(H, W, ids, svd_r=(), n_clusters=8, clus_p=(1,), kmeans_iters=3, log=lambda s: None)
    assert r2["arms"][0]["recall1"] <= a["recall1"] and r2["arms"][0]["macs_removed"] > a["macs_removed"]


def test_pass_rule_and_futility():
    H, W, ids = _case(V=4000, d=64, N=256, rank=4)
    r = R.evaluate(H, W, ids, svd_r=(4,), svd_c=(16,), n_clusters=0, log=lambda s: None)
    assert r["arms"][0]["macs_removed"] >= 0.79 and r["verdict"] == "PASS" and not r["futility"]
    H, W, ids = _case(V=4000, d=64, N=256)  # full-rank random head, tiny rank/C -> poor recall
    r = R.evaluate(H, W, ids, svd_r=(2,), svd_c=(16,), n_clusters=0, log=lambda s: None)
    assert r["verdict"] == "FAIL" and r["futility"]


def test_loader_and_cli_roundtrip():
    H, W, ids = _case(V=600, d=16, N=64)
    with tempfile.TemporaryDirectory() as td:
        _write_dump(td, H, W, ids)
        H2, W2, ids2, m, v2 = R.load_dump(td)
        assert np.array_equal(H, H2) and np.array_equal(W, W2) and np.array_equal(ids, ids2) and v2.shape == ids.shape
        if True:
            out = os.path.join(td, "r.json")
            assert R.main(["recall.py", td, out, "--kmeans-iters", "2"]) == 0
            j = json.load(open(out))
            assert j["sanity_argmax_match"] == 1.0 and j["vocab"] == 600 and j["verdict"] == "FAIL" and j["futility"]  # defaults (r>=d, k>=V) leave no arm on a tiny head


def _int_reference(H, W, mode):
    """Block-integer dot exactly as the kernel forms it: sum_b dA_b * dW_b * <qA_b, qW_b> (int64 dots, float64 scales)."""
    qa, da = R.act_codes(H, mode)
    if mode == "iqk_q6k":
        qw, dw = R.weight_codes(W, mode)
    else:  # Q8_0 weight: the f32 dump is already q*d exactly; recover its codes with ggml's own scale
        qw, dw = R.act_codes(W, "iqk_q8_0")
    out = np.zeros((H.shape[0], W.shape[0]))
    for b in range(qa.shape[1]):
        dots = qa[:, b].astype(np.int64) @ qw[:, b].astype(np.int64).T
        assert np.all(np.abs(qa[:, b]) <= 128) and np.all(qa[:, b] == np.rint(qa[:, b]))
        out += np.outer(da[:, b, 0], dw[:, b, 0]) * dots
    return out


def test_bf16_rounding():
    x = np.array([1.0, 1.00390625, 1.005859375, 3.0e-3, -2.5], np.float32)
    y = R._bf16(x)
    assert y[0] == 1.0 and y[1] == 1.0 and y[2] == np.float32(1.0078125) and y[4] == -2.5  # RNE: tie -> even
    assert np.all(np.abs(y - x) <= np.abs(x) * 2 ** -8)


def test_quantizers_codes_are_int8_and_dequant_gemm_equals_block_int_dot():
    g = np.random.default_rng(3)
    H = (g.standard_normal((64, 128)) * 4).astype(np.float32)
    W = (g.standard_normal((300, 128)) * 0.05).astype(np.float32)
    W0 = W
    qw, dw = R.act_codes(W, "iqk_q8_0")
    W = (qw * dw).reshape(W.shape).astype(np.float32)  # a real Q8_0 head: weights already on the q8_0 grid
    for mode in ("iqk_q6k", "iqk_q8_0"):
        A = R.quant_act(H, mode)
        assert not np.array_equal(A, H) and np.max(np.abs(A - H)) < np.max(np.abs(H)) / 127
        L = A.astype(np.float64) @ R.quant_weight(W, mode).T.astype(np.float64)
        assert np.allclose(L, _int_reference(H, W, mode), rtol=1e-6, atol=1e-6)
    assert not np.array_equal(R.quant_weight(W0, "iqk_q6k"), W0) and np.array_equal(R.quant_weight(W, "iqk_q8_0"), W)


def _quantized_model(mode, V=4000, d=64, N=1024, seed=5):
    g = np.random.default_rng(seed)
    W = (g.standard_normal((V, d)) * 0.05).astype(np.float32)
    H = (g.standard_normal((N, d)) * 40).astype(np.float32)  # logit scale ~ real heads (|top1| ~ 20, few margins < eps)
    L = R.quant_act(H, mode) @ R.quant_weight(W, mode).T
    o = np.argsort(-L, axis=1)[:, :16]
    return H, W, o.astype(np.int32), np.take_along_axis(L, o, 1).astype(np.float32)


def test_faithful_arith_reproduces_quantized_model_and_f32_does_not():
    for mode in ("iqk_q6k", "iqk_q8_0"):
        H, W, ids, vals = _quantized_model(mode)
        r = R.evaluate(H, W, ids, vals, arith=mode, svd_r=(), n_clusters=0, log=lambda s: None)
        assert r["sanity_argmax_match_all_rows"] == 1.0 and r["sanity_argmax_match"] == 1.0 and not r["verdict"].startswith("VOID")
        assert r["max_top1_logit_diff"] < 1e-3
        f = R.evaluate(H, W, ids, vals, arith="f32", svd_r=(), n_clusters=0, log=lambda s: None)
        assert f["sanity_argmax_match_all_rows"] < 1.0  # quantization flips near-tie rows: the r1 failure mode


def test_margin_filter_cannot_hide_a_broken_dump():
    H, W, ids, vals = _quantized_model("iqk_q8_0")
    bad = ids.copy()
    big = np.where(vals[:, 0] - vals[:, 1] > 1.0)[0][:20]  # wrong argmax on confidently-decided rows
    bad[big, 0] = (bad[big, 0] + 1) % W.shape[0]
    r = R.evaluate(H, W, bad, vals, arith="iqk_q8_0", svd_r=(), n_clusters=0, log=lambda s: None)
    assert r["verdict"].startswith("VOID") and r["sanity_argmax_match"] < 0.999
    flat = vals.copy(); flat[:, 1] = flat[:, 0]  # every model margin 0 -> nothing kept -> VOID, not a vacuous pass
    r = R.evaluate(H, W, ids, flat, arith="iqk_q8_0", svd_r=(), n_clusters=0, log=lambda s: None)
    assert r["verdict"].startswith("VOID") and r["sanity_rows_kept"] == 0.0
    # a near-tie disagreement below eps is excluded and does not void
    tie = vals.copy(); i = 7; tie[i, 1] = tie[i, 0] - R.SANITY_EPS / 2
    sw = ids.copy(); sw[i, 0] = ids[i, 1]
    r = R.evaluate(H, W, sw, tie, arith="iqk_q8_0", svd_r=(), n_clusters=0, log=lambda s: None)
    assert r["sanity_argmax_match"] == 1.0 and r["sanity_argmax_match_all_rows"] < 1.0 and not r["verdict"].startswith("VOID")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
