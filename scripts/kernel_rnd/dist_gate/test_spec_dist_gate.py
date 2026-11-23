"""Tests for spec_dist_gate. Run: python3 -m pytest scripts/kernel_rnd/dist_gate -q  (CPU, seconds)."""
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_dist_gate as g  # noqa: E402

V = 30


def make_dist(rng, confident):
    logits = {t: rng.gauss(0, 1.0) for t in range(V)}
    if confident:
        logits[rng.randrange(V)] += 6.0
    return logits


def sample(probs, rng, power=1.0):
    items = sorted(probs.items())
    w = [p ** power for _, p in items]
    z = sum(w)
    x = rng.random() * z
    for (t, _), wi in zip(items, w):
        x -= wi
        if x <= 0:
            return t
    return items[-1][0]


def corpus(n_req=48, n_tok=100, seed=0, power=1.0, keep=None, sampler=None):
    rng = random.Random(seed)
    reqs = []
    for q in range(n_req):
        req = []
        for i in range(n_tok):
            logits = make_dist(rng, confident=rng.random() < 0.5)
            probs = g.post_sampler_probs(logits, **(sampler or {}))
            tok = sample(probs, rng, power)
            rec = {"probs": probs, "token": tok, "key": (q, i)}
            if keep is None or keep(rec):
                req.append(rec)
        reqs.append(req)
    return reqs


def confident(rec):
    return max(rec["probs"].values()) >= 0.5


def test_sampler_chain():
    logits = {0: 3.0, 1: 2.0, 2: 1.0, 3: -5.0}
    p = g.post_sampler_probs(logits, temperature=1.0, top_k=2)
    assert set(p) == {0, 1} and abs(sum(p.values()) - 1) < 1e-12
    p = g.post_sampler_probs(logits, min_p=0.2)
    assert 3 not in p and 1 in p
    full = g._softmax(logits)
    assert (2 in p) == (full[2] >= 0.2 * full[0])
    p = g.post_sampler_probs(logits, top_p=0.5)
    assert set(p) == {0}
    p_hot = g.post_sampler_probs(logits, temperature=2.0)
    p_cold = g.post_sampler_probs(logits, temperature=0.5)
    assert p_hot[0] < g.post_sampler_probs(logits)[0] < p_cold[0]
    assert g.post_sampler_probs(logits, temperature=0) == {0: 1.0}


def test_pit_uniform_with_ties():
    rng = random.Random(3)
    probs = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # all tied: PIT must still be uniform
    u = [g.randomized_pit(probs, sample(probs, rng), rng) for _ in range(5000)]
    assert g.ks_uniform(u)[0] > 1e-3
    assert g.randomized_pit(probs, 99, rng) is None


@pytest.mark.parametrize("sampler", [{}, {"temperature": 0.3, "top_k": 40, "top_p": 0.95, "min_p": 0.05},
                                     {"temperature": 4.0, "top_p": 0.95}])
def test_exact_sampler_passes(sampler):
    arm = corpus(seed=1, sampler=sampler)
    ref = corpus(seed=2, sampler=sampler)
    r = g.gate(arm, ref, match_keys=False)
    assert r["verdict"] == "PASS", r["reasons"]


def test_biased_sampler_fails():
    arm = corpus(seed=1, power=1.3)  # sharpened: over-samples the top token
    ref = corpus(seed=2)
    r = g.gate(arm, ref)
    assert r["verdict"] == "FAIL"
    assert r["ks_p"] <= g.THRESHOLDS["ks_fail_p"]


def test_biased_sampler_fails_on_ks_alone():
    assert g.gate(corpus(seed=5, power=0.7))["verdict"] == "FAIL"


def test_out_of_support_fails():
    arm = corpus(seed=1)
    arm[0][0] = dict(arm[0][0], token=10_000)
    assert g.gate(arm)["verdict"] == "FAIL"


def test_selection_bias_needs_matching():
    # The arm is an EXACT sampler that only emits at confident positions (p-min style selection).
    arm = corpus(seed=1, keep=confident)
    ref = corpus(seed=2)
    unmatched = g.gate(arm, ref)
    assert unmatched["verdict"] == "FAIL", "unmatched chi2 should be fooled by selection"
    assert unmatched["ks_p"] > g.THRESHOLDS["ks_fail_p"]  # the KS itself is not fooled
    matched = g.gate(arm, ref, select=confident)
    assert matched["verdict"] == "PASS", matched["reasons"]


def test_selection_matching_by_key():
    rng_ref = corpus(seed=7)
    # arm: same positions (shared keys) re-sampled exactly, then thinned to confident ones
    arm = [[dict(r, token=sample(r["probs"], random.Random(hash(r["key"]) & 0xffff))) for r in q if confident(r)]
           for q in rng_ref]
    assert g.gate(arm, corpus(seed=8))["verdict"] == "FAIL"
    assert g.gate(arm, rng_ref, match_keys=True)["verdict"] == "PASS"


def test_inconclusive_small_and_uncalibrated_ref():
    assert g.gate(corpus(n_req=4, n_tok=50, seed=1))["verdict"] == "INCONCLUSIVE"
    r = g.gate(corpus(seed=1), corpus(seed=2, power=1.5))
    assert r["verdict"] in ("INCONCLUSIVE", "FAIL") and any("reference" in x for x in r["reasons"])


def test_perm_test_null_is_calibrated():
    ps = [g.perm_test_requests(corpus(n_req=12, n_tok=40, seed=s), corpus(n_req=12, n_tok=40, seed=100 + s),
                               n_perm=199, seed=s)[1] for s in range(10)]
    assert min(ps) > 0.005


def test_chi2_sf_matches_known():
    assert abs(g._chi2_sf(3.841, 1) - 0.05) < 1e-3
    assert abs(g._chi2_sf(18.307, 10) - 0.05) < 1e-3


# --------------------------------------------------------------------- SPEC-ID-GATE-1 helpers
def test_margin_class():
    assert [g.margin_class(m) for m in (0.0, 0.05, 0.1, 0.3, 0.5, 0.51)] == \
        ["exact0", "le0.1", "le0.1", "le0.5", "le0.5", "gt0.5"]
    with pytest.raises(ValueError):
        g.margin_class(-1)


def test_classify_divergence():
    c = g.classify_divergence
    assert c(0.05, identity_floor_zero=False)["verdict"] == "NOISE"
    assert c(0.4, identity_floor_zero=False, tau=0.22)["verdict"] == "DEFECT_CANDIDATE"
    assert c(0.8, identity_floor_zero=False)["verdict"] == "DEFECT_CANDIDATE"
    assert c(0.01, identity_floor_zero=True)["verdict"] == "DEFECT_CANDIDATE"
    assert c(0.01, identity_floor_zero=False, degen_after=True)["verdict"] == "DEGEN_AFTER_TIE"
    assert c(0.01, identity_floor_zero=False, degen_after=True, control_degen=True)["verdict"] == "NOISE"
    assert c(None, identity_floor_zero=False)["verdict"] == "UNCLASSIFIABLE"


def test_paired_count_test():
    aa = [1, 2, 1, 1, 2, 1, 1, 2]
    same = g.paired_count_test([2, 1, 1, 2, 1, 1, 2, 1], aa)
    assert same["p_one_sided"] > 0.1
    worse = g.paired_count_test([4, 5, 4, 3, 5, 4, 4, 5], aa)
    assert worse["p_one_sided"] < 0.01 and worse["ci95"][0] > 0
    big = g.paired_count_test([3] * 20, [1] * 20)
    assert big["p_one_sided"] < 0.001


def _to_server_form(reqs):
    """Project probs-dict records onto the llama-server pit_probs form (pit_less, pit_tie, rank)."""
    out = []
    for q in reqs:
        o = []
        for r in q:
            p = r["probs"]
            px = p.get(r["token"], 0.0)
            z = sum(p.values())
            less = sum(v for v in p.values() if v < px) / z if px > 0 else -1.0
            tie = sum(v for v in p.values() if v == px) / z if px > 0 else -1.0
            o.append({"pit_less": less, "pit_tie": tie, "rank": g.token_rank(p, r["token"])})
        out.append(o)
    return out


def test_server_form_records():
    exact = _to_server_form(corpus(seed=1, sampler={"temperature": 1.0, "top_p": 0.95}))
    ref = _to_server_form(corpus(seed=2, sampler={"temperature": 1.0, "top_p": 0.95}))
    assert g.gate(exact, ref)["verdict"] == "PASS"
    biased = _to_server_form(corpus(seed=3, power=1.3))
    assert g.gate(biased, _to_server_form(corpus(seed=4)))["verdict"] == "FAIL"
    oos = _to_server_form(corpus(seed=5))
    oos[0][0] = {"pit_less": -1.0, "pit_tie": -1.0, "rank": 20}
    assert g.gate(oos)["verdict"] == "FAIL"
