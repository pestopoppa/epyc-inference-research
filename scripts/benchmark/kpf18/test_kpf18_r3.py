"""Server-free tests for kpf18r3 (item 186, KPF-18-S1.R3).

Nothing here starts a server, touches the GPU or reads a model. The steps run against test_kpf18_s1r.FakeServer (the
fork build's HTTP surface: KPF-11 fork, own-slot reuse with checkpoint rollback, the --cache-ram save-on-reslot /
load-into-slot rule, logit_bias). The rules are tested on hand-made inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_kpf18_arms as TA  # noqa: F401  (loads the gate as a module: TA.GATE)
import test_kpf18_s1r as TS
import kpf18_arms as A
import kpf18_s1r as S
import kpf18_r3 as R

GATE = TA.GATE


# ================================================================ wiring
def test_registered_as_item_186_with_resume_regex():
    assert A.ARM_CMDS["kpf18r3"] == [186] and 186 in A.ARM_ITEMS and A.TASK_OF[186] == "KPF-18-S1.R3"
    assert 186 in GATE.ITEMS and 186 in A.BUILDERS and 186 in A.PREFLIGHTS
    assert set(A.RULES[186]) == {"common", "long6", "item2q"}
    assert A.rules_sha256([186]) and A.rules_sha256([186]) not in (A.rules_sha256([185]), A.rules_sha256([184]))
    assert GATE._ARM_STEP_RE.match("186 (3) long6 x") and GATE._ARM_STEP_RE.match("185 (4) x")
    assert not GATE._ARM_STEP_RE.match("187 x")
    assert A.ARM_CMDS["kpf18s1r"] == [185]                 # the remainder command is untouched


def _args(tmp_path, **over):
    a = TS.mk_args(tmp_path, **over)
    a.r3_steps = over.get("r3_steps", ",".join(R.STEP_KEYS))
    return a


def test_plan_order_estimates_and_resume_keys(tmp_path):
    ctx = GATE.Ctx(_args(tmp_path), GATE.PromptSet.from_spec(), TA.GPU_ARGV, ["numactl"], {})
    steps = A.BUILDERS[186](ctx, GATE)
    d = [s.desc for s in steps]
    assert "linkage + launch" in d[0] and "SIGTERM own pid" in d[-1] and sum("linkage + launch" in x for x in d) == 1
    body = d[1:-1]
    assert len(body) == 2 and body[0].startswith("186 (3) long6") and body[1].startswith("186 (2) item2q g3.s2")
    est = {s.desc.split(" ")[2]: s.est_s for s in steps[1:-1]}
    # long6: 5 x 60k prefills at the measured 120 s + 6 x 256 positions x 0.12 s (+ small terms)
    assert 5 * 120 + 6 * 256 * 0.12 <= est["long6"] < 5 * 120 + 6 * 256 * 0.12 + 60
    assert 120 < est["item2q"] < 900
    skip = GATE.resume_plan(d, lambda k: k == 1)           # long6 done -> skipped, item2q and the launch re-run
    assert skip == [False, True, False, False]
    steps2 = A.BUILDERS[186](ctx, GATE)
    assert [GATE.step_key(186, k, s.desc) for k, s in enumerate(steps)] == \
           [GATE.step_key(186, k, s.desc) for k, s in enumerate(steps2)]


def test_step_subset_and_preflight_problems(tmp_path):
    ctx = GATE.Ctx(_args(tmp_path, r3_steps="long6,bogus"), GATE.PromptSet.from_spec(), TA.GPU_ARGV, ["numactl"], {})
    probs, notes = A.PREFLIGHTS[186](ctx, GATE)
    assert any("bogus" in p for p in probs) and any("predeclaration" in p for p in probs)
    assert any("--s1r-forced-seq" in p for p in probs)
    steps = A.BUILDERS[186](ctx, GATE)
    assert [s.desc.split(" ")[2] for s in steps[1:-1]] == ["long6"]


# ================================================================ rules on hand-made inputs
T, NT, S0 = 60000, 60512, 512
GOOD_FIRST = {"child": {"n_fork_tokens": T, "fork_src_kind": "checkpoint", "fork_src_slot": 0, "prompt_n": S0, "cache_n": T},
              **{c: {"cache_n": 0, "prompt_n": NT, "n_fork_tokens": 0} for c in R.COLD_ARMS},
              "sameshape": {"cache_n": T, "prompt_n": S0, "n_fork_tokens": 0}}


def _lp(offsets: dict, n=64):
    return {a: [-1.0 - off + 0.001 * (i % 5) for i in range(n)] for a, off in offsets.items()}


def test_arm_validity():
    for a in R.JUDGED + (R.SS,):
        assert R.arm_validity(a, GOOD_FIRST[a], T=T, n_task=NT, S0=S0) == []
    assert R.arm_validity("child", {**GOOD_FIRST["child"], "n_fork_tokens": 0}, T=T, n_task=NT, S0=S0)
    assert R.arm_validity("child", {**GOOD_FIRST["child"], "fork_src_slot": 2}, T=T, n_task=NT, S0=S0)
    assert R.arm_validity("coldC", {"cache_n": T, "prompt_n": S0}, T=T, n_task=NT, S0=S0)        # a hit is not cold
    assert R.arm_validity("sameshape", {"cache_n": 0, "prompt_n": NT}, T=T, n_task=NT, S0=S0)     # recompute is not a hit
    assert R.arm_validity("sameshape", {"cache_n": T, "prompt_n": S0, "n_fork_tokens": T}, T=T, n_task=NT, S0=S0)


def test_long6_six_pair_floor_satisfies_what_one_pair_failed():
    # the remainder's numbers: child-A 0.02925, child-B 0.02988, A-B 0.02746 (one pair: NOT_SATISFIED by 9%)
    lp = _lp({"child": 0.0, "coldA": 0.02925, "coldB": 0.02988, "coldC": 0.02990, "coldD": 0.0, "sameshape": 0.0})
    r = R.eval_long6(lp, GOOD_FIRST, T=T, n_task=NT, S0=S0)
    assert r["verdict"] == "SATISFIED" and r["predeclared_rule_satisfied"] is True
    assert len(r["d"]) == 10 and r["n_common_valid"] == 64
    assert r["floor_pair"] in ("coldC|coldD", "coldB|coldD") and r["floor_used"] == pytest.approx(0.02990)
    assert r["worst_child_vs_cold"] == pytest.approx(0.02990) and r["worst_pair"] == "child|coldC"
    assert r["pcache_reading"]["label"] == "FORK_MATCHES_PCACHE_EXACTLY"


def test_long6_not_satisfied_and_pcache_labels():
    lp = _lp({"child": 0.05, "coldA": 0.0, "coldB": 0.01, "coldC": 0.0, "coldD": 0.01, "sameshape": 0.049})
    r = R.eval_long6(lp, GOOD_FIRST, T=T, n_task=NT, S0=S0)
    assert r["verdict"] == "NOT_SATISFIED" and r["ratio_worst_over_floor"] > 1
    pr = r["pcache_reading"]
    assert pr["label"] == "FORK_CLOSER_TO_PCACHE_THAN_EVERY_COLD" and pr["weight"].startswith("zero")
    lp[R.SS] = [x - 0.028 for x in lp["coldA"]]            # child-ss 0.022; cold-ss in [0.018, 0.028]
    assert R.eval_long6(lp, GOOD_FIRST, T=T, n_task=NT, S0=S0)["pcache_reading"]["label"] == "FORK_WITHIN_COLD_VS_PCACHE_RANGE"
    lp[R.SS] = [x + 0.5 for x in lp["coldA"]]              # child-ss 0.55 > every cold-ss (0.50, 0.51)
    assert R.eval_long6(lp, GOOD_FIRST, T=T, n_task=NT, S0=S0)["pcache_reading"]["label"] == "FORK_OUTSIDE_COLD_VS_PCACHE_RANGE"
    bad_ss = {**GOOD_FIRST, "sameshape": {"cache_n": 0, "prompt_n": NT}}
    r2 = R.eval_long6(lp, bad_ss, T=T, n_task=NT, S0=S0)
    assert r2["verdict"] == "NOT_SATISFIED" and r2["pcache_reading"]["label"] == "PCACHE_NOT_ESTABLISHED"


def test_long6_floor_is_lifted_invalid_and_abstain():
    same = _lp({a: 0.0 for a in R.JUDGED + (R.SS,)})
    r = R.eval_long6(same, GOOD_FIRST, T=T, n_task=NT, S0=S0)
    assert r["floor_raw"] == 0.0 and r["floor_used"] == A.FLOOR_EPS and r["verdict"] == "SATISFIED"
    bad = {**GOOD_FIRST, "coldB": {"cache_n": 59999, "prompt_n": 513}}
    r = R.eval_long6(same, bad, T=T, n_task=NT, S0=S0)
    assert r["verdict"] == "INVALID" and r["arm_invalid_reasons"]["coldB"] and r["d"] is None
    sparse = {a: [v if i < 10 else None for i, v in enumerate(xs)] for a, xs in same.items()}
    r = R.eval_long6(sparse, GOOD_FIRST, T=T, n_task=NT, S0=S0)
    assert r["verdict"] == "ABSTAIN" and r["n_common_valid"] == 10
    nofork = {**GOOD_FIRST, "child": {**GOOD_FIRST["child"], "n_fork_tokens": 0}}
    assert R.eval_long6(same, nofork, T=T, n_task=NT, S0=S0)["verdict"] == "INVALID"
    # an invalid sameshape never blocks the rule
    assert R.eval_long6(same, {**GOOD_FIRST, "sameshape": {}}, T=T, n_task=NT, S0=S0)["verdict"] == "SATISFIED"


def test_cross_load_reading_is_recorded_only():
    now = _lp({"child": 0.0, "coldA": 0.0, "coldB": 0.0})
    base = _lp({"child": 0.0, "coldA": 0.01, "coldB": 0.0})
    base["coldB"][3] = None
    x = R.cross_load(now, base)
    assert x["pairs"]["child"]["mean_abs_dlogprob"] == 0.0 and x["pairs"]["coldA"]["mean_abs_dlogprob"] == pytest.approx(0.01)
    assert x["pairs"]["coldB"]["n_positions"] == 63 and "never pooled" in x["weight"]


def _cmp(prefix=256, maxabs=0.0, kld=0.0, ident=False, top1=True):
    return {"top1": top1, "prefix": prefix, "n_cmp": 256, "maxabs_union": maxabs, "maxabs_intersection": maxabs, "kld": kld,
            "tokens_identical": prefix == 256, "identical": ident, "len_a": 256, "len_b": 256}


def _q(name, prefix, valid=True):
    return {"name": name, "valid": valid, "invalid_reasons": [] if valid else ["Q_not_prompt_cache_load_at_T:0,36512,0"],
            "q_vs_own": _cmp(prefix=prefix, maxabs=0.03, kld=0.01)}


def test_item2q_verdicts():
    qs = [_q("Q1", 24), _q("Q2", 3), _q("Q3", 70)]
    fo = _cmp(prefix=5, maxabs=0.02, kld=0.005)
    r = R.eval_item2q(fo, [], qs, 256)
    assert r["verdict"] == "G3S2_WITHIN_PLACEMENT_FLOOR" and r["n_q_valid"] == 3
    assert r["q_vs_own_prefixes"] == {"Q1": 24, "Q2": 3, "Q3": 70} and r["judge"]["aa_floor"]["prefix_min"] == 3
    r = R.eval_item2q(fo, [], [_q("Q1", 24), _q("Q2", 30), _q("Q3", 70)], 256)
    assert r["verdict"] == "G3S2_OUTSIDE_PLACEMENT_FLOOR" and r["judge"]["n_prefix_ok"] == 0
    r = R.eval_item2q(fo, [], [_q("Q1", 24), _q("Q2", 3, valid=False), _q("Q3", 70)], 256)
    assert r["verdict"] == "G3S2_CONTROL_NOT_ESTABLISHED" and r["q_invalid"]["Q2"]
    r = R.eval_item2q(fo, ["F_not_forked_at_T:0,none,-1"], qs, 256)
    assert r["verdict"] == "G3S2_INVALID" and "fork_vs_own" in r and "n_q_valid" in r


def test_q_and_fo_validity():
    ok = {"http": 200, "tokens": [1] * 8, "timings": {"cache_n": 32000, "prompt_n": 4000, "n_fork_tokens": 0}}
    assert R.q_validity(ok, T=32000, S=4000, n=8) == []
    assert R.q_validity({**ok, "timings": {"cache_n": 0, "prompt_n": 36000}}, T=32000, S=4000, n=8)
    assert R.q_validity({**ok, "tokens": [1] * 7}, T=32000, S=4000, n=8)
    F = {"http": 200, "tokens": [1] * 8, "timings": {"n_fork_tokens": 32000, "fork_src_kind": "checkpoint", "fork_src_slot": 0}}
    O = {"http": 200, "tokens": [1] * 8, "timings": {"cache_n": 32000, "n_fork_tokens": 0}}
    assert R.fo_validity(F, O, T=32000, S=4000, n=8) == []
    assert R.fo_validity({**F, "timings": {}}, O, T=32000, S=4000, n=8)


def test_missing_result_line_is_a_fail():
    full = {"item3_r3": {k: 0 for k in ("verdict", "floor_used", "worst_child_vs_cold", "n_common_valid", "arm_valid", "d",
                                        "pcache_reading")},
            "item2_g3s2": {k: 0 for k in ("verdict", "n_q_valid", "q_vs_own_prefixes", "fork_vs_own")}, "run_valid": {}}
    assert R.missing_lines(full) == []
    del full["item3_r3"]["d"]
    del full["run_valid"]
    assert R.missing_lines(full) == ["item3_r3.d", "run_valid"]


# ================================================================ steps end to end on the fake server
def mk_env(tmp_path, **over):
    ctx, env = TS.mk_run_env(tmp_path, **over)
    ctx.args.r3_steps = ",".join(R.STEP_KEYS)
    return ctx, env


def _records(ctx):
    return [json.loads(p.read_text()) for p in sorted((ctx.out / "records").glob("*.json"))]


def test_steps_end_to_end_on_the_fake_server_and_verdict(tmp_path):
    ctx, env = mk_env(tmp_path, s1r_tf_positions=20, n_predict=12)
    R.run_long6(env)
    R.run_item2q(env)
    recs = _records(ctx)
    assert len(recs) == 2
    sha = S.sha256_path(ctx.args.s1r_predeclaration)
    for r in recs:
        assert r["schema"] == "epyc.kpf.run.v1" and r["kpf18_item"] == 186 and r["rows_sha256"]
        assert r["r3"]["step_complete"] and r["r3"]["predeclaration_sha256"] == sha and "argv" in r
    lg, i2 = (r["r3"] for r in recs)
    assert lg["fresh_load"] is True and i2["fresh_load"] is False
    res = lg["result"]
    assert all(res["arm_valid"].values()), res["arm_invalid_reasons"]       # the fake's prompt cache really moved the source
    assert res["first"]["sameshape"]["cache_n"] == 60000 and res["first"]["coldD"]["cache_n"] == 0
    assert res["n_common_valid"] == 20 and len(res["d"]) == 10 and res["verdict"] in ("SATISFIED", "NOT_SATISFIED")
    assert res["occupancy_before"]["coldC"]["2"] == 0 and res["occupancy_before"]["coldC"]["3"] == 0
    assert res["occupancy_before"]["coldD"]["3"] > 0          # coldD prefills beside coldC (coldB's occupancy)
    q = i2["result"]
    assert q["verdict"] in ("G3S2_WITHIN_PLACEMENT_FLOOR", "G3S2_OUTSIDE_PLACEMENT_FLOOR") and q["n_q_valid"] == 3, q["q_invalid"]
    assert [x["resident_slots"] for x in q["q"]] == [[0, 1], [0], []] and len(q["q_vs_q"]) == 3
    assert q["q"][0]["occupancy_before"]["1"] > 0 and q["q"][2]["occupancy_before"]["0"] == 0
    rv = {"valid": True, "graphs": "ON", "vram_trip": None, "gate_rc": 0}
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"s1r": {"result": {"lp": res["lp"]}}}))
    v = R.verdict(ctx.out, sha, run_valid=rv, base_long_record=str(base))
    assert v["verdict"] == "COMPLETE" and v["missing_result_lines"] == []
    assert v["item3_r3"]["cross_load_graphs_off_vs_on"]["pairs"]["coldA"]["mean_abs_dlogprob"] == 0.0
    assert R.verdict(ctx.out, sha, run_valid={**rv, "valid": False})["verdict"] == "RUN_INVALID"
    assert R.verdict(ctx.out, sha)["verdict"] == "FAIL_MISSING_RESULT_LINE"           # no run_valid line
    v2 = R.verdict(ctx.out, "0" * 64, run_valid=rv)
    assert v2["verdict"] == "INCOMPLETE" and len(v2["excluded_predeclaration_mismatch"]) == 2
    # the remainder's own verdict reader never picks R3 records up
    assert S.verdict(ctx.out, sha)["records_seen"] == 0


def test_partial_run_is_incomplete_and_resume_reads_latest(tmp_path):
    ctx, env = mk_env(tmp_path, s1r_tf_positions=20, n_predict=12)
    R.run_item2q(env)
    sha = S.sha256_path(ctx.args.s1r_predeclaration)
    v = R.verdict(ctx.out, sha, run_valid={"valid": True})
    assert v["verdict"] == "INCOMPLETE" and v["item_verdicts"]["item3_r3"] == "INCOMPLETE"
    assert v["missing_result_lines"] == []


def test_sameshape_without_a_cached_source_is_recorded_invalid(tmp_path):
    ctx, env = mk_env(tmp_path, s1r_tf_positions=20)
    fake = ctx.client
    real = fake._cache_path

    def no_cache(slot, ids):                                # a server whose --cache-ram never loads anything
        cur = fake.s[slot]["tokens"]
        if cur and A.lcp(cur, ids) / len(cur) < 0.5:
            fake.s[slot] = {"tokens": [], "ckpts": [], "fork": 0}
            return None
        return real(slot, ids)
    fake._cache_path = no_cache
    R.run_long6(env)
    res = _records(ctx)[0]["r3"]["result"]
    assert res["arm_valid"]["sameshape"] is False and res["pcache_reading"]["label"] == "PCACHE_NOT_ESTABLISHED"
    assert res["verdict"] in ("SATISFIED", "NOT_SATISFIED")   # the rule does not depend on the control


def test_budget_guard_refuses_long6_when_it_does_not_fit(tmp_path):
    ctx, env = mk_env(tmp_path, budget_min=5.0)
    steps = R.body_186(ctx, GATE, env)
    with pytest.raises(GATE.Abort, match="not started"):
        steps[0].fn()
    assert not list((ctx.out / "records").glob("*.json")) if (ctx.out / "records").exists() else True


def test_stop_file_refuses_a_new_r3_step(tmp_path, monkeypatch):
    ctx, env = mk_env(tmp_path)
    stop = tmp_path / "STOP"
    stop.write_text("")
    monkeypatch.setenv("KPF18_S1R_STOPFILE", str(stop))
    with pytest.raises(GATE.Abort, match="STOP file"):
        R.body_186(ctx, GATE, env)[1].fn()


def test_predeclared_forced_file_and_launcher_predeclaration_when_present():
    d = Path("/mnt/raid0/llm/tmp/f8-launchers-20261024/kpf18-s1-r3")
    f = d / "forced_seq_183p4.json"
    if not f.exists():
        pytest.skip("launcher directory not on this host")
    doc, probs = S.load_forced(f, S.FORCED_SHA256)
    assert probs == [] and doc["prompt"]["ids_sha256"].startswith("74ee91a9899677a3")
    assert "FAIL_MISSING_RESULT_LINE" in (d / "PREDECLARATION.md").read_text()
