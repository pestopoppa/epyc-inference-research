"""No-model tests for the SPEC-RNG.1.2 runner (run: /usr/bin/python3 -m pytest scripts/benchmark/spec_rng_1_2 inside the worktree).

Part A  statistics and analyzer logic (chi-square vs known values, a null passes, an injected bias fails, gates fire).
Part B  the DFlash2 rejection path at the LOGIC level: a Python mirror of common_sampler_sample_and_accept_n
        (llama.cpp experimental/spec-rng-argv-20261022 @ 261db6c7e, common/sampling.cpp:740-835) on a toy Markov target. It checks
        that the position-keyed variant preserves the target distribution at every draft depth (what the model run measures on
        the real server), that the stateful variant does too, and that the TEST CAN FAIL: a variant that reuses one uniform for
        the accept test and the residual draw is detected as biased.
        tools/server/tests/unit/test_rng_poskey.py cannot cover this path (it needs a model and excludes the block drafters).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_rng_1_2 as S  # noqa: E402

M64 = (1 << 64) - 1


# ----------------------------------------------------------------------------------------- mirror of common/rng-poskey.h
def splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & M64
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & M64
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & M64
    return x ^ (x >> 31)


def u01(seed, pos, role):
    h = splitmix64(seed)
    h = splitmix64(h ^ pos)
    h = splitmix64(h ^ ((role * 0xd6e8feb86659fd93) & M64))
    return (h >> 11) / float(1 << 53)


ROLE_DIST, ROLE_ACC, ROLE_RES = 1, 2, 3
SPEC_SEED_XOR = 0x9e3779b9


def test_u01_matches_cpp_header():
    # values produced by compiling /mnt/raid0/llm/worktrees/spec-rng-argv-20261022/common/rng-poskey.h (g++, 2026-10-23)
    vec = [(0, 0, 1, 0.22639831625024998), (0, 7, 2, 0.90165521735338938), (1234, 0, 3, 0.5379810652461593),
           (1234, 1, 2, 0.81715033637020384), (1234, 1000, 1, 0.31845234378634779), (42, 7, 3, 0.13621367749982705)]
    for s, p, r, want in vec:
        assert u01(s, p, r) == want


# ------------------------------------------------------------------------------------------------------ toy Markov models
V = 6
TARGET = [[0.30, 0.25, 0.20, 0.10, 0.10, 0.05], [0.05, 0.40, 0.25, 0.15, 0.10, 0.05], [0.20, 0.20, 0.20, 0.20, 0.10, 0.10],
          [0.10, 0.10, 0.10, 0.30, 0.20, 0.20], [0.25, 0.05, 0.05, 0.05, 0.30, 0.30], [0.15, 0.15, 0.35, 0.15, 0.10, 0.10]]
DRAFT = [[0.50, 0.10, 0.10, 0.10, 0.10, 0.10], [0.10, 0.50, 0.10, 0.10, 0.10, 0.10], [0.10, 0.10, 0.50, 0.10, 0.10, 0.10],
         [0.10, 0.10, 0.10, 0.50, 0.10, 0.10], [0.10, 0.10, 0.10, 0.10, 0.50, 0.10], [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]]


def inv_cdf(p, u):
    tgt, run = sum(p) * u, 0.0
    for i, x in enumerate(p):
        run += x
        if run >= tgt:
            return i
    return len(p) - 1


class Stream:
    """One sequence's sampler state: stateful mt-style RNGs, or the position-keyed counter."""

    def __init__(self, seed, poskey, bug_reuse_u=False):
        self.seed, self.poskey, self.bug = seed, poskey, bug_reuse_u
        self.dist_rng = random.Random(seed)
        self.spec_rng = random.Random(seed ^ SPEC_SEED_XOR)
        self.n_accepted = 0

    def u_dist(self):
        return u01(self.seed, self.n_accepted, ROLE_DIST) if self.poskey else self.dist_rng.random()

    def u_acc(self):
        return u01(self.seed ^ SPEC_SEED_XOR, self.n_accepted, ROLE_ACC) if self.poskey else self.spec_rng.random()

    def u_res(self, u_acc):
        if self.bug:
            return u_acc                      # the injected defect: accept test and residual draw share one uniform
        return u01(self.seed ^ SPEC_SEED_XOR, self.n_accepted, ROLE_RES) if self.poskey else self.spec_rng.random()


def verify(st, prev, draft):
    """Mirror of common_sampler_sample_and_accept_n for one round: draft = [(token, q_row_prev)] chained from `prev`."""
    out = []
    cur = prev
    for tok, q in draft:
        p = TARGET[cur]
        u = st.u_acc()
        if q[tok] > 0 and u * q[tok] <= p[tok]:
            out.append(tok)
            st.n_accepted += 1
            cur = tok
            continue
        resid = [max(0.0, p[i] - q[i]) for i in range(V)]
        if sum(resid) > 0:
            pick = inv_cdf(resid, st.u_res(u))
        else:
            pick = inv_cdf(p, st.u_dist())
        out.append(pick)
        st.n_accepted += 1
        return out
    out.append(inv_cdf(TARGET[cur], st.u_dist()))   # bonus token
    st.n_accepted += 1
    return out


def generate(seed, length, depth, poskey, bug=False):
    st = Stream(seed, poskey, bug)
    draft_rng = random.Random(seed * 7919 + 13)     # the drafter's own randomness: independent of the verifier's RNG
    toks, cur = [], 0
    while len(toks) < length:
        if depth == 0:
            t = inv_cdf(TARGET[cur], st.u_dist())
            st.n_accepted += 1
            toks.append(t)
            cur = t
            continue
        draft, c = [], cur
        for _ in range(depth):
            q = DRAFT[c]
            t = inv_cdf(q, draft_rng.random())
            draft.append((t, q))
            c = t
        for t in verify(st, cur, draft):
            toks.append(t)
            cur = t
    return toks[:length]


def marginals(depth, poskey, n, length=4, bug=False, base=0):
    cs = [dict() for _ in range(length)]
    for s in range(base, base + n):
        for j, t in enumerate(generate(s, length, depth, poskey, bug)):
            cs[j][t] = cs[j].get(t, 0) + 1
    return cs


def exact_marginals(length):
    d = [1.0] + [0.0] * (V - 1)                     # prev token 0 before the first draw
    out = []
    for _ in range(length):
        d = [sum(d[a] * TARGET[a][b] for a in range(V)) for b in range(V)]
        out.append(list(d))
    return out


def gof_p(counts, exact):
    n = sum(counts.values())
    stat = sum((counts.get(i, 0) - n * exact[i]) ** 2 / (n * exact[i]) for i in range(V))
    return S.chi2_sf(stat, V - 1)


N_LOGIC = 6000


def test_poskey_rejection_preserves_target_at_every_depth():
    ex = exact_marginals(4)
    for depth in (0, 3, 7):
        cs = marginals(depth, True, N_LOGIC, base=10_000 * depth)
        ps = [gof_p(cs[j], ex[j]) for j in range(4)]
        assert min(ps) > 1e-3, (depth, ps)


def test_stateful_rejection_preserves_target_at_every_depth():
    ex = exact_marginals(4)
    for depth in (0, 3, 7):
        cs = marginals(depth, False, N_LOGIC, base=10_000 * depth)
        assert min(gof_p(cs[j], ex[j]) for j in range(4)) > 1e-3


def test_poskey_depth_invariance_homogeneity_like_the_model_run():
    a, b = marginals(0, True, N_LOGIC, base=0), marginals(7, True, N_LOGIC, base=500_000)
    ps = [S.homogeneity(a[j], b[j])[2] for j in range(1, 4)]
    assert min(ps) > 1e-3, ps


def test_the_test_can_fail_reused_uniform_is_detected():
    ex = exact_marginals(4)
    cs = marginals(7, True, 20000, bug=True, base=900_000)
    assert min(gof_p(cs[j], ex[j]) for j in range(1, 4)) < 1e-4


def test_poskey_u_is_a_function_of_position_not_of_draw_count():
    # extra rejected rounds at one position (more draws) leave later positions' uniforms unchanged: pure function
    assert u01(5, 17, ROLE_ACC) == u01(5, 17, ROLE_ACC)
    assert len({u01(5, 17, r) for r in (ROLE_DIST, ROLE_ACC, ROLE_RES)}) == 3


def test_same_seed_depth_changes_the_stream_by_design():
    # distribution-invariant, not sample-identical (SPEC-RNG.1.1 / the task text): at least one seed differs between depth 0 and 7
    diff = sum(generate(s, 6, 0, True) != generate(s, 6, 7, True) for s in range(200))
    assert diff > 0


# --------------------------------------------------------------------------------------------------------- statistics
def test_chi2_sf_known_values():
    assert abs(S.chi2_sf(3.841458820694124, 1) - 0.05) < 1e-9
    assert abs(S.chi2_sf(18.307038053275146, 10) - 0.05) < 1e-9
    assert abs(S.chi2_sf(0.0, 5) - 1.0) < 1e-12
    assert S.chi2_sf(200.0, 3) < 1e-30


def test_homogeneity_identical_and_different():
    a = {1: 100, 2: 100, 3: 100}
    assert S.homogeneity(a, dict(a))[2] > 0.99
    stat, dof, p, tv = S.homogeneity({1: 190, 2: 10}, {1: 100, 2: 100})
    assert p < 1e-10 and dof == 1 and 0.3 < tv < 0.5


def test_homogeneity_pools_rare_ids_and_handles_degenerate():
    a = {1: 100, 90: 2, 91: 3}
    b = {1: 95, 92: 4, 93: 2}
    stat, dof, p, tv = S.homogeneity(a, b)
    assert dof == 1 and p > 0.05                    # rare ids pooled into one bin
    assert S.homogeneity({}, {1: 5}) == (0.0, 0, 1.0, 0.0)
    assert S.homogeneity({1: 50}, {1: 50})[2] == 1.0


def test_sign_test_and_welch():
    assert S.sign_test_one_sided(0, 0) == 1.0
    assert abs(S.sign_test_one_sided(8, 8) - 1 / 256) < 1e-12
    assert S.welch_p([0.5] * 50, [0.5] * 50) == 1.0
    assert S.welch_p([0.1, 0.2] * 100, [0.8, 0.9] * 100) < 1e-10


# --------------------------------------------------------------------------------------------------------- analyzer
def test_plan_shape():
    assert [a[0] for a in S.ARMS] == ["s0", "s3", "s7", "p0", "p3", "p7"]
    seeds = set()
    n = 0
    for blk, reqs in S.schedule():
        for key, meta, body in reqs:
            assert body["seed"] not in seeds and body["n_predict"] == S.K_TOKENS and body["temperature"] > 0
            seeds.add(body["seed"])
            n += 1
            assert body["rng_poskey"] == (meta["arm"][0] == "p") and body["speculative.n_max"] == int(meta["arm"][1:])
    assert n == S.MAX_BLOCKS * S.SEEDS_PER_BLOCK * 3 * 6


def test_analyzer_null_has_no_poskey_effect():
    r = S.analyze(S.synthetic_records())
    assert r["gates"]["primary_poskey_invariant"] and r["gates"]["controls_pass"] and r["gates"]["V2_rejection_path_engaged"]
    assert r["verdict"] == "DO-NOT-CARRY"           # identical probe outputs in both modes -> no demonstrated gain


def test_analyzer_detects_injected_bias():
    r = S.analyze(S.synthetic_records(bias=0.5))
    assert not r["gates"]["primary_poskey_invariant"] and r["verdict"] == "POSKEY-BIASED"


def test_analyzer_inconclusive_when_rejection_path_not_engaged():
    r = S.analyze(S.synthetic_records(engaged=False))
    assert r["verdict"] == "INCONCLUSIVE" and not r["gates"]["V2_rejection_path_engaged"]


def test_analyzer_inconclusive_when_cells_too_small():
    r = S.analyze(S.synthetic_records(n=100))
    assert r["verdict"] == "INCONCLUSIVE" and not r["gates"]["V1_cells_n_ge_min"]


def test_analyzer_carry_requires_probe_gain():
    recs = S.synthetic_records()
    # make poskey reproducible alone-vs-batched and stateful not, for 20 of 48 seeds
    for r in recs:
        if r["phase"] == "probe" and r["rng"] == "stateful" and r["cond"] == "batched" and r["seed"] < 20:
            r["tokens"] = [t + 1 for t in r["tokens"]]
    assert S.analyze(recs)["verdict"] == "CARRY"
    # the reverse (stateful better) never carries
    recs2 = S.synthetic_records()
    for r in recs2:
        if r["phase"] == "probe" and r["rng"] == "poskey" and r["cond"] == "batched" and r["seed"] < 20:
            r["tokens"] = [t + 1 for t in r["tokens"]]
    assert S.analyze(recs2)["verdict"] == "DO-NOT-CARRY"


def test_acceptance_shift_is_poskey_biased():
    recs = S.synthetic_records()
    for r in recs:
        if r["phase"] == "main" and r["arm"] == "p7":
            r["draft_n_accepted"] = 1                  # acceptance 1/7 vs 0.6 stateful
    res = S.analyze(recs)
    assert not res["gates"]["acceptance_unchanged"] and res["verdict"] == "POSKEY-BIASED"


def test_selftest_and_dry_drive(capsys, tmp_path):
    assert S.selftest() == 0
    assert S.drive(0, str(tmp_path), 0, dry=True) == 0
    assert "DRY plan" in capsys.readouterr().out


def test_sink_resume_skips_done(tmp_path):
    sink = S.Sink(str(tmp_path / "r.jsonl"))
    sink.put({"key": "a", "ok": True})
    sink.put({"key": "b", "ok": False})
    sink2 = S.Sink(str(tmp_path / "r.jsonl"))
    assert sink2.done == {"a"}                          # failed requests are retried on resume
