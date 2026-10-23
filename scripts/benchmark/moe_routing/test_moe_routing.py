"""Synthetic-trace tests for the RT-5/6/7 analysis scripts. No model, no inference: every trace is generated here."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import locality_report as lr  # noqa: E402
import placement_replay as pr  # noqa: E402
import residency_curve as rc  # noqa: E402
import skew_report as sk  # noqa: E402
import trace_io as tio  # noqa: E402
import validate_traces as vt  # noqa: E402

E, K, L = 16, 2, 2


def write_shard(root: Path, domain: str, part: str, first_window: int, windows, probs=None, seed=0, n_pre=8, n_dec=4, repeat=False):
    """windows: list of window-local sample ids. probs: [L][E] sampling distributions (default uniform).
    repeat=True makes every token of a window pick the same experts (IR_t == 1)."""
    rng = np.random.default_rng(seed)
    d = root / domain / part / f"shard-{first_window:06d}"
    d.mkdir(parents=True)
    rows, ids, samples = [], [], []
    for w in windows:
        start = len(rows)
        base = [rng.choice(E, K, replace=False, p=None if probs is None else probs[l]) for l in range(L)]
        n = n_pre + (n_dec if part == "decode" else 0)
        for p in range(n):
            ph = 0 if p < n_pre else 1
            step = 0 if ph == 0 else p - n_pre + 1
            blk = n_pre if ph == 0 else 1
            sel = []
            for l in range(L):
                s = base[l] if repeat else rng.choice(E, K, replace=False, p=None if probs is None else probs[l])
                sel.append(np.asarray(s, np.int16))
            rows.append((w, p, step, blk, ph, 255))
            ids.append(np.stack(sel))
        samples.append({"sample_id": w, "domain": domain, "start_row": start, "n_rows": len(rows) - start})
    ra = np.array(rows, dtype=tio.ROW_DTYPE)
    ia = np.stack(ids).astype("<i2")
    (d / "routing_trace.rows.bin").write_bytes(ra.tobytes())
    (d / "routing_trace.ids.i16").write_bytes(ia.tobytes())
    man = {"schema": tio.SCHEMA_ID, "model": "synthetic", "arch": "synth", "batch_size": n_pre,
           "layers": [{"layer": l, "n_experts": E, "top_k": K, "id_dtype": "int16"} for l in range(L)],
           "samples": samples, "columns": ["sample_id", "pos", "phase", "step", "spec_block", "spec_accepted"],
           "ids": {"dtype": "int16", "shape_note": "x", "sha256": tio.sha256_file(d / "routing_trace.ids.i16")},
           "rows_sha256": tio.sha256_file(d / "routing_trace.rows.bin")}
    (d / "routing_trace.json").write_text(json.dumps(man))
    (d / "routing_trace.provenance.json").write_text(json.dumps({"domain": domain, "first_window": first_window}))
    # tool-side IR the independent way (python loops)
    eor = {"layers": []}
    for l in range(L):
        num, pairs = 0.0, 0
        for i in range(1, ra.size):
            if ra["sample_id"][i] == ra["sample_id"][i - 1] and ra["phase"][i] == 0:
                num += len(set(ia[i, l].tolist()) & set(ia[i - 1, l].tolist())) / K
                pairs += 1
        eor["layers"].append({"prefill": {"pairs": pairs, "ir_mean": num / pairs}})
    (d / "routing_trace.eor.json").write_text(json.dumps(eor))
    (d / "VERIFIED").write_text("ok")
    return d


@pytest.fixture()
def uniform_root(tmp_path):
    write_shard(tmp_path, "uni", "decode", 0, list(range(4)), seed=1)
    write_shard(tmp_path, "uni", "bulk", 4, list(range(60)), seed=2)
    return tmp_path


def skewed_probs(hot=2, mass=0.9):
    p = np.full(E, (1 - mass) / (E - hot))
    p[:hot] = mass / hot
    return [p.copy() for _ in range(L)]


# ------------------------------------------------------------------ trace_io
def test_load_domain_geometry_and_window_ids(uniform_root):
    t = tio.load_domain(uniform_root, "uni", verify=True)
    assert t.R == 4 * 12 + 60 * 8 and t.L == L and t.ids.shape[2] == K
    assert sorted(set(t.win.tolist())) == list(range(64))
    assert int((t.phase == 1).sum()) == 16


def test_hash_mismatch_is_refused(tmp_path):
    d = write_shard(tmp_path, "x", "bulk", 0, [0, 1])
    (d / "routing_trace.ids.i16").write_bytes(b"\0" * (d / "routing_trace.ids.i16").stat().st_size)
    with pytest.raises(ValueError):
        tio.load_shard(d, verify=True)


def test_blocks_drop_incomplete_tail_and_never_cross_windows(uniform_root):
    t = tio.load_domain(uniform_root, "uni")
    dec = t.select(t.phase == 1)             # 4 windows x 4 decode rows
    blk, nb, win_of = tio.blocks(dec, 3)
    assert nb == 4 and (blk >= 0).sum() == 12 and (blk < 0).sum() == 4
    assert win_of.tolist() == [0, 1, 2, 3]
    blk1, nb1, _ = tio.blocks(dec, 1)
    assert nb1 == 16
    pre = t.select(t.phase == 0)
    _, nbp, _ = tio.blocks(pre, 8)
    assert nbp == 64                          # one chunk per window


def test_layer_loads_counts_every_selection(uniform_root):
    t = tio.load_domain(uniform_root, "uni")
    blk, nb, _ = tio.blocks(t.select(t.phase == 0), 8)
    X = tio.layer_loads(t.select(t.phase == 0).layer_ids(0), blk, nb, E)
    assert X.shape == (64, E) and (X.sum(1) == 8 * K).all()


# ------------------------------------------------------------------ validate_traces
def test_validate_accepts_clean_and_rejects_missing_window(tmp_path):
    write_shard(tmp_path, "d", "bulk", 0, list(range(6)))
    assert vt.validate(tmp_path, 6, True, 8)["problems"] == []
    bad = tmp_path / "bad"
    write_shard(bad, "d", "bulk", 0, [0, 1, 2, 4, 5])      # window 3 missing
    probs = vt.validate(bad, 6, False, 8)["problems"]
    assert any("windows are not exactly" in p for p in probs)


# ------------------------------------------------------------------ skew (RT-5)
def test_skew_uniform_is_near_chance_and_planted_skew_is_detected(tmp_path):
    write_shard(tmp_path, "uni", "bulk", 0, list(range(400)), seed=3, n_pre=8)
    write_shard(tmp_path, "hot", "bulk", 0, list(range(400)), probs=skewed_probs(2, 0.9), seed=4, n_pre=8)
    r = sk.analyse(tmp_path, ["uni", "hot"], gate=0.6)
    n10 = sk.top_n(0.10, E)                        # 2 of 16
    uni = r["domains"]["uni"]["prefill"]["summary"]
    hot = r["domains"]["hot"]["prefill"]["summary"]
    assert abs(uni["top10_in_sample_mean"] - n10 / E) < 0.04
    assert abs(uni["top10_heldout_mean"] - n10 / E) < 0.05          # held-out removes the selection bias
    assert uni["top10_in_sample_mean"] - uni["top10_null_mean"] < 0.02
    assert hot["top10_in_sample_mean"] > 0.8 and hot["top10_heldout_mean"] > 0.8
    assert hot["entropy_norm_mean"] < uni["entropy_norm_mean"] - 0.2
    assert r["rt5_verdict"]["prefill_workloads_meeting_gate_by_layer_mean"] == ["hot"]
    assert r["rt5_verdict"]["prefill_gate_met"] is False           # needs >= 2 workloads


def test_skew_gate_met_needs_two_workloads(tmp_path):
    for i, d in enumerate(("a", "b")):
        write_shard(tmp_path, d, "bulk", 0, list(range(300)), probs=skewed_probs(2, 0.9), seed=10 + i)
    assert sk.analyse(tmp_path, ["a", "b"], gate=0.6)["rt5_verdict"]["prefill_gate_met"] is True


def test_cross_workload_overlap_identical_vs_disjoint_hot_sets(tmp_path):
    pa = skewed_probs(2, 0.9)
    pb = [np.roll(p, 8) for p in pa]               # hot experts moved to 8, 9
    write_shard(tmp_path, "a", "bulk", 0, list(range(300)), probs=pa, seed=1)
    write_shard(tmp_path, "a2", "bulk", 0, list(range(300)), probs=pa, seed=2)
    write_shard(tmp_path, "b", "bulk", 0, list(range(300)), probs=pb, seed=3)
    o = sk.analyse(tmp_path, ["a", "a2", "b"], 0.6)["overlap"]["prefill"]["pairs"]
    assert o["a|a2"]["top10_jaccard_mean"] == 1.0
    assert o["a|b"]["top10_jaccard_mean"] == 0.0


def test_spearman_and_jaccard_edges():
    assert sk.spearman(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)
    assert sk.spearman(np.arange(5.0), -np.arange(5.0)) == pytest.approx(-1.0)
    assert sk.jaccard(np.array([1, 2]), np.array([2, 3])) == pytest.approx(1 / 3)


# ------------------------------------------------------------------ locality
def test_ir_is_one_when_tokens_repeat_and_near_chance_when_random(tmp_path):
    write_shard(tmp_path, "rep", "bulk", 0, list(range(20)), repeat=True)
    write_shard(tmp_path, "rnd", "bulk", 0, list(range(400)), seed=5)
    assert lr.ir_by_phase(tio.load_domain(tmp_path, "rep"), 0)["prefill"]["ir_mean"] == 1.0
    ir = lr.ir_by_phase(tio.load_domain(tmp_path, "rnd"), 0)["prefill"]
    assert abs(ir["ir_mean"] - K / E) < 0.02


def test_lru_stack_distance_on_a_hand_built_sequence():
    # one window, K=1-like behaviour via top_k=2 rows: A=(0,1) B=(2,3) A again -> experts 0,1 each have 2 distinct experts since last use
    ids = np.array([[0, 1], [2, 3], [0, 1], [0, 1]], np.int16)[:, None, :].repeat(L, 1)
    t = tio.Trace("x", "synth", np.full(L, E), np.full(L, K), np.zeros(4, np.int64), np.arange(4), np.zeros(4, np.int64),
                  np.zeros(4, np.int8), ids, [])
    c = lr.lru_curve(t, 0, np.array([0]), sizes=(2, 3))["prefill"]
    # 8 selections: tokens 0,1 cold (4); token 2 dist 2 (x2); token 3 dist 0 (x2)
    assert c["compulsory_miss_rate"] == 0.5
    assert c["hit_rate"]["2"] == pytest.approx(2 / 8)       # only token 3 (dist 0 < 2)
    assert c["hit_rate"]["3"] == pytest.approx(4 / 8)       # token 2 (dist 2 < 3) joins


def test_coactivation_detects_planted_pair_and_noise_floor_is_small(tmp_path):
    # experts (0,1) always together, everything else random -> huge lift for pair (0,1)
    rng = np.random.default_rng(0)
    d = tmp_path / "c" / "bulk" / "shard-000000"
    d.mkdir(parents=True)
    n = 6000
    ids = np.zeros((n, L, K), np.int16)
    for i in range(n):
        for l in range(L):
            ids[i, l] = [0, 1] if rng.random() < 0.3 else rng.choice(np.arange(2, E), K, replace=False)
    rows = np.zeros(n, tio.ROW_DTYPE)
    rows["pos"] = np.arange(n)
    (d / "routing_trace.rows.bin").write_bytes(rows.tobytes())
    (d / "routing_trace.ids.i16").write_bytes(ids.astype("<i2").tobytes())
    (d / "routing_trace.json").write_text(json.dumps({"schema": tio.SCHEMA_ID, "arch": "s", "layers": [
        {"layer": l, "n_experts": E, "top_k": K, "id_dtype": "int16"} for l in range(L)], "samples": [], "ids": {}, "rows_sha256": ""}))
    t = tio.load_shard(d)
    c = lr.coactivation(t, 0, np.random.default_rng(1), min_exp=1.0)
    assert (c["top_pairs"][0]["a"], c["top_pairs"][0]["b"]) == (0, 1)
    assert c["top_pairs"][0]["lift"] > 4
    assert c["excess_fraction"] > 2 * c["excess_fraction_noise"]


# ------------------------------------------------------------------ residency (RT-7)
def test_resident_set_greedy_crosses_layers_and_hit_rate_is_exact():
    cnt = np.array([[90, 5, 3, 2], [30, 30, 20, 20]])        # layer 0 concentrated, layer 1 flat
    res = rc.resident_set(cnt, k=1.5, mode="greedy")           # budget 3 over two layers
    assert res.sum() == 3 and res[0, 0]                                           # the 0.9 expert is taken first
    assert rc.hit_rate(res, cnt) == pytest.approx(cnt[res].sum() / cnt.sum())
    uni = rc.resident_set(cnt, k=1, mode="uniform")
    assert uni.sum(1).tolist() == [1, 1]


def test_residency_transfers_same_hot_set_but_not_a_moved_one(tmp_path):
    pa = skewed_probs(2, 0.9)
    pb = [np.roll(p, 8) for p in pa]
    write_shard(tmp_path, "a", "decode", 0, list(range(6)), probs=pa, seed=1, n_dec=40)
    write_shard(tmp_path, "a", "bulk", 6, list(range(200)), probs=pa, seed=2)
    write_shard(tmp_path, "b", "decode", 0, list(range(6)), probs=pb, seed=3, n_dec=40)
    write_shard(tmp_path, "b", "bulk", 6, list(range(200)), probs=pb, seed=4)
    r = rc.analyse(tmp_path, ["a", "b"], [2], ref_experts=E, gate_k=2)
    c = r["curves"]["a/prefill"]
    assert c["learned_on"]["a_even->a_odd(prefill)"]["2"] > 0.85
    assert c["learned_on"]["b"]["2"] < 0.15
    assert c["oracle_in_sample"]["2"] >= c["learned_on"]["a_even->a_odd(prefill)"]["2"] - 1e-9
    assert r["rt7_verdict"]["by_k"]["2"]["prefill"]["workloads_ge_0.30_same_domain"] == ["a", "b"]
    assert r["rt7_verdict"]["by_k"]["2"]["prefill"]["workloads_ge_0.30_worst_case"] == []


# ------------------------------------------------------------------ placement (RT-6)
def test_waterfill_integer_and_fractional():
    assert waterfill_sum(np.array([0.0, 0.0]), 3) == 3
    assert pr.waterfill(np.array([5.0, 0.0]), 2).tolist() == [0.0, 2.0]
    base = np.array([0.0, 2.0, 10.0])
    a = pr.waterfill(base, 40)
    assert a.sum() == pytest.approx(40)
    assert (base + a) == pytest.approx(np.full(3, 52 / 3))


def waterfill_sum(base, c):
    return pr.waterfill(base, c).sum()


def test_eplb_replicates_the_hot_expert_and_packs_equal_cardinality():
    w = np.array([100.0, 1, 1, 1, 1, 1, 1, 1])
    bins = pr.place_eplb(w, B=4, R=2)
    assert len(bins[0]) == 3 and sum(len(b) for b in bins) == 10
    sizes = np.bincount([b for bb in bins for b in bb], minlength=4)
    assert sizes.max() == math.ceil(10 / 4)


def test_deepinfer_peak_objective_separates_experts_that_peak_together():
    # experts 0 and 1 peak in the same steps; experts 2 and 3 peak in other steps. Same aggregate load for all four.
    Wm = np.array([[10, 10, 0, 0]] * 4 + [[0, 0, 10, 10]] * 4, float)
    bins = pr.place_deepinfer(Wm, B=2, R=0)
    assert bins[0][0] != bins[1][0] and bins[2][0] != bins[3][0]


def test_eval_static_balanced_vs_stacked_and_replica_dispatch():
    X = np.zeros((1, 4))
    X[0, :2] = 5
    assert pr.eval_static(X, [[0], [0], [1], [1]], 2)[0] == pytest.approx(2.0)      # both hot on bin 0
    assert pr.eval_static(X, [[0], [1], [0], [1]], 2)[0] == pytest.approx(1.0)
    X2 = np.array([[4.0, 0, 0, 0]])
    assert pr.eval_static(X2, [[0, 1], [0], [1], [1]], 2)[0] == pytest.approx(1.0)  # one hot expert split over two replicas


def test_lp_floor_matches_hand_solution_and_never_beats_the_integrality_free_mean():
    X = np.array([[4.0, 2.0]])
    r = pr.lp_floor(X, [[0, 1], [0]], B=2)[0]
    assert r == pytest.approx(1.0)                   # t* = 3 = mean -> expert 0 gives 3 units to bin 1? total 6, 2 bins, t*=3
    r2 = pr.lp_floor(X, [[0], [0]], B=2)[0]
    assert r2 == pytest.approx(2.0)                  # no replica: all 6 on bin 0
    rng = np.random.default_rng(0)
    Xr = rng.integers(0, 4, (30, 8)).astype(float)
    pl = [[int(rng.integers(0, 4))] + ([int(rng.integers(0, 4))] if e < 3 else []) for e in range(8)]
    lp = pr.lp_floor(Xr, pl, 4)
    assert (lp >= 1.0 - 1e-6).all()
    assert (lp <= pr.eval_static(Xr, pl, 4) + 1e-6).all()      # fractional <= integral greedy dispatch


def test_oracle_and_moonep_reach_the_integrality_floor_with_enough_budget():
    home = np.array([0, 0, 1, 1])
    e_act, c_act = np.array([0, 1]), np.array([4, 2])          # 6 units all on bin 0
    mx, used = pr.oracle_step(c_act, e_act, home, B=2, R=8)
    assert mx == 3 and used == 1                                # move 3 units of expert 0 to bin 1
    mx0, used0 = pr.oracle_step(c_act, e_act, home, B=2, R=0)
    assert mx0 == 6 and used0 == 0                              # no replica budget -> no change
    assert pr.moonep_step(c_act, e_act, home, B=2) == 1


def test_integrality_floor_and_uniform_null():
    assert pr.integrality_floor(1, 10, 4) == pytest.approx(1.2)
    assert pr.integrality_floor(3, 10, 12) == pytest.approx(1.2)
    assert pr.null_uniform_mc(1, 4, 1) == pytest.approx(1.0)
    assert pr.null_uniform_mc(1, 10, 4) > pr.integrality_floor(1, 10, 4)


def test_end_to_end_replay_on_planted_skew(tmp_path):
    write_shard(tmp_path, "s", "decode", 0, list(range(24)), probs=skewed_probs(4, 0.8), seed=1, n_dec=32)
    write_shard(tmp_path, "s", "bulk", 24, list(range(488)), probs=skewed_probs(4, 0.8), seed=2)
    r = pr.replay_domain((str(tmp_path), "s", [4], 0.25, "same", 1.65, 0, 3))
    c = r["configs"]["B4/decode/T1"]
    a = c["arms"]
    assert a["eplb_R4"]["max_over_mean"] < a["contiguous"]["max_over_mean"] and a["eplb_R4@random"]["max_over_mean"] >= a["eplb_R4"]["max_over_mean"] - 1e-9
    assert a["oracle_R4"]["max_over_mean"] <= a["eplb_R4"]["max_over_mean"] + 1e-9
    assert a["integrality_floor"]["max_over_mean"] <= a["oracle_R4"]["max_over_mean"] + 1e-9
    assert a["lp_floor@eplb_R4"]["max_over_mean"] <= a["eplb_R4"]["max_over_mean"] + 1e-6
    assert set(c["decision"]) >= {"best_practical", "best_bound", "ratio_practical", "ratio_bound"}
    assert "B4/prefill/T8" in r["configs"] and "B4/decode/T3" in r["configs"]


# ------------------------------------------------------------------ RT-5.1 balance statistics
def test_balance_stats_hand_values():
    b = sk.balance_stats(np.array([6, 2, 0, 0]))          # mean 2
    assert b["maxvio"] == pytest.approx(2.0) and b["minvio"] == pytest.approx(1.0) and b["inactive_ratio"] == 0.5
    u = sk.balance_stats(np.array([3, 3, 3, 3]))
    assert u == {"maxvio": 0.0, "minvio": 0.0, "inactive_ratio": 0.0}


def test_balance_stats_separate_skewed_from_uniform_routing(tmp_path):
    write_shard(tmp_path, "uni", "bulk", 0, list(range(200)), seed=3, n_pre=8)
    write_shard(tmp_path, "hot", "bulk", 0, list(range(200)), probs=skewed_probs(2, 0.9), seed=4, n_pre=8)
    r = sk.analyse(tmp_path, ["uni", "hot"], gate=0.6)
    u, h = (r["domains"][d]["prefill"]["summary"] for d in ("uni", "hot"))
    assert h["stream_maxvio_mean"] > 3 * u["stream_maxvio_mean"]
    assert h["stream_inactive_ratio_mean"] >= u["stream_inactive_ratio_mean"]
    assert abs(u["stream_maxvio_mean"] - u["stream_null_maxvio_mean"]) < 0.1     # uniform routing sits on its own null
    assert "chunk512_maxvio_mean" not in u or u["chunk512_maxvio_mean"] >= 0     # 8-row windows hold no 512-row chunk: stat absent, no crash


def test_random_dispatch_is_stateless_and_never_beats_least_loaded_in_expectation():
    X = np.zeros((400, 4))
    X[:, 0] = 6                                              # one hot expert, 6 units a step
    pl = [[0, 1], [0], [1], [1]]
    ideal = pr.eval_static(X, pl, 2)
    rnd = pr.eval_static(X, pl, 2, np.random.default_rng(0))
    assert ideal.mean() == pytest.approx(1.0)                # 3 + 3
    assert rnd.mean() > ideal.mean() + 0.1                   # 6 coin flips are rarely 3/3
    assert (rnd >= 1.0 - 1e-9).all()
