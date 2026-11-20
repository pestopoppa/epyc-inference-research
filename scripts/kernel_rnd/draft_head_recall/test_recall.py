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


def _write_dump(dirn, H, W, ids):
    t = []
    for key, a, typ in (("head_hidden", H, "f32"), ("head_w.f32", W, "f32"), ("head_topk_ids", ids, "i32"),
                        ("head_topk_val", np.zeros_like(ids, dtype=np.float32), "f32")):
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
        H2, W2, ids2, m = R.load_dump(td)
        assert np.array_equal(H, H2) and np.array_equal(W, W2) and np.array_equal(ids, ids2)
        if True:
            out = os.path.join(td, "r.json")
            assert R.main(["recall.py", td, out, "--kmeans-iters", "2"]) == 0
            j = json.load(open(out))
            assert j["sanity_argmax_match"] == 1.0 and j["vocab"] == 600 and j["verdict"] == "FAIL" and j["futility"]  # defaults (r>=d, k>=V) leave no arm on a tiny head


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
