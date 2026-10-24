"""Server-free tests for kpf18r4 (item 187, KPF-18-S1.R4).

Nothing here starts a server, touches the GPU or reads a model. The step runs against test_kpf18_s1r.FakeServer (the
fork build's HTTP surface: KPF-11 fork, own-slot reuse with checkpoint rollback, the --cache-ram rule, logit_bias). The
rule is tested on hand-made inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_kpf18_arms as TA  # noqa: F401  (loads the gate as a module: TA.GATE)
import test_kpf18_s1r as TS
import kpf18_arms as A
import kpf18_s1r as S
import kpf18_r4 as R

GATE = TA.GATE


# ================================================================ wiring
def test_registered_as_item_187_with_resume_regex():
    assert A.ARM_CMDS["kpf18r4"] == [187] and 187 in A.ARM_ITEMS and A.TASK_OF[187] == "KPF-18-S1.R4"
    assert 187 in GATE.ITEMS and 187 in A.BUILDERS and 187 in A.PREFLIGHTS
    assert set(A.RULES[187]) == {"common", "r4"}
    assert A.rules_sha256([187]) and A.rules_sha256([187]) not in (A.rules_sha256([186]), A.rules_sha256([185]))
    assert GATE._ARM_STEP_RE.match("187 (3) r4 x") and GATE._ARM_STEP_RE.match("186 (3) long6 x")
    assert not GATE._ARM_STEP_RE.match("188 x")
    assert A.ARM_CMDS["kpf18r3"] == [186]                  # the R3 command is untouched


def _args(tmp_path, **over):
    a = TS.mk_args(tmp_path, **over)
    a.r4_steps = over.get("r4_steps", ",".join(R.STEP_KEYS))
    return a


def test_plan_order_estimate_and_resume_keys(tmp_path):
    ctx = GATE.Ctx(_args(tmp_path), GATE.PromptSet.from_spec(), TA.GPU_ARGV, ["numactl"], {})
    steps = A.BUILDERS[187](ctx, GATE)
    d = [s.desc for s in steps]
    assert "linkage + launch" in d[0] and "SIGTERM own pid" in d[-1] and sum("linkage + launch" in x for x in d) == 1
    body = d[1:-1]
    assert len(body) == 1 and body[0].startswith("187 (3) r4")
    est = steps[1].est_s
    # 5 x 60k prefills (source, coldE, coldA, coldB, decoy) at the measured 120 s + 6 x 256 positions x 0.12 s
    # (+ three 512-token fork prefills, overheads)
    assert 5 * 120 + 6 * 256 * 0.12 <= est < 5 * 120 + 6 * 256 * 0.12 + 60
    assert GATE.resume_plan(d, lambda k: k == 1) == [False, True, False]
    steps2 = A.BUILDERS[187](ctx, GATE)
    assert [GATE.step_key(187, k, s.desc) for k, s in enumerate(steps)] == \
           [GATE.step_key(187, k, s.desc) for k, s in enumerate(steps2)]


def test_step_subset_and_preflight_problems(tmp_path):
    ctx = GATE.Ctx(_args(tmp_path, r4_steps="r4,bogus"), GATE.PromptSet.from_spec(), TA.GPU_ARGV, ["numactl"], {})
    probs, notes = A.PREFLIGHTS[187](ctx, GATE)
    assert any("bogus" in p for p in probs) and any("predeclaration" in p for p in probs)
    assert any("--s1r-forced-seq" in p for p in probs)
    ctx = GATE.Ctx(_args(tmp_path, r4_steps=""), GATE.PromptSet.from_spec(), TA.GPU_ARGV, ["numactl"], {})
    assert len(A.BUILDERS[187](ctx, GATE)) == 2            # launch + stop only


# ================================================================ rules on hand-made inputs
T, NT, S0 = 60000, 60512, 512
FORKED = {"n_fork_tokens": T, "fork_src_kind": "checkpoint", "fork_src_slot": 0, "prompt_n": S0, "cache_n": T}
COLD = {"cache_n": 0, "prompt_n": NT, "n_fork_tokens": 0}
GOOD_FIRST = {**{a: dict(FORKED) for a in R.FORK_ARMS}, **{c: dict(COLD) for c in R.COLD_ARMS}}


def _occ(res, n=60767):
    o = {str(i): None for i in range(4)}
    for s in res:
        o[str(s)] = NT if s == 0 else n
    return o


GOOD_OCC = {a: _occ(r) for a, r in R.RESIDENTS.items()}


def _lp(offsets: dict, n=64):
    return {a: [-1.0 - off + 0.001 * (i % 5) for i in range(n)] for a, off in offsets.items()}


GOOD_DECOY = {"n": 60767, "lcp_with_P": 7, "first": {"cache_n": 0, "prompt_n": 60767, "n_fork_tokens": 0}}


def _ev(lp, first=GOOD_FIRST, occ=GOOD_OCC, decoy=GOOD_DECOY):
    return R.eval_r4(lp, first, occ, T=T, n_task=NT, S0=S0, decoy=decoy, n_min=1024)


def test_decoy_validity_only_touches_childX():
    assert R.decoy_validity(GOOD_DECOY, GOOD_OCC["childX"], n_min=1024) == []
    assert R.decoy_validity({**GOOD_DECOY, "lcp_with_P": 60000}, GOOD_OCC["childX"], n_min=1024)   # could be a fork source
    assert R.decoy_validity({**GOOD_DECOY, "first": {"cache_n": 0, "prompt_n": 5}}, GOOD_OCC["childX"], n_min=1024)
    assert R.decoy_validity(GOOD_DECOY, _occ((0, 1), n=60000), n_min=1024)                    # resident, wrong length
    assert R.decoy_validity(None, GOOD_OCC["childX"], n_min=1024) == ["decoy_missing"]
    same = _lp({a: 0.0 for a in R.ALL_ARMS})
    r = _ev(same, decoy=None)
    assert r["verdict"] == "OCCUPANCY_NUMERICS" and r["arm_valid"]["childX"] is False
    assert r["factorial_reading"]["label"] == "CHILDX_NOT_ESTABLISHED"


def test_plan_matches_the_predeclared_order_slots_and_residents():
    assert [p[0] for p in R.PLAN] == ["child", "coldE", "child2", "coldA", "coldB", "childX"]
    assert R.SLOTS == {"child": 1, "coldE": 1, "child2": 1, "coldA": 2, "coldB": 3, "childX": 2}
    assert R.RESIDENTS["coldE"] == R.RESIDENTS["child"] == R.RESIDENTS["child2"] == (0,)
    assert R.RESIDENTS["childX"] == R.RESIDENTS["coldA"] == (0, 1) and R.RESIDENTS["coldB"] == (0, 1, 2)
    assert dict((p[0], p[3]) for p in R.PLAN) == {"child": (), "coldE": (1,), "child2": (1,), "coldA": (), "coldB": (),
                                                  "childX": (1, 2, 3)}
    assert R.DECOY_BEFORE == "childX" and R.DECOY_SLOT == 1 and R.DECOY_GROUP != 5
    assert set(R.JUDGED) == {"child", "child2", "coldA", "coldB", "coldE"} and R.CX not in R.JUDGED


def test_arm_validity_timings_and_residents():
    for a in R.ALL_ARMS:
        assert R.arm_validity(a, GOOD_FIRST[a], GOOD_OCC[a], T=T, n_task=NT, S0=S0) == []
    assert R.arm_validity("child2", {**FORKED, "n_fork_tokens": 0}, GOOD_OCC["child2"], T=T, n_task=NT, S0=S0)
    assert R.arm_validity("childX", {**FORKED, "fork_src_slot": 1}, GOOD_OCC["childX"], T=T, n_task=NT, S0=S0)
    assert R.arm_validity("coldE", {"cache_n": T, "prompt_n": S0}, GOOD_OCC["coldE"], T=T, n_task=NT, S0=S0)  # a hit is not cold
    # coldE run beside a second resident sequence is R3's coldA occupancy, not the child's: invalid
    bad = R.arm_validity("coldE", COLD, _occ((0, 2)), T=T, n_task=NT, S0=S0)
    assert bad and "residents" in bad[0]
    assert R.arm_validity("coldA", COLD, None, T=T, n_task=NT, S0=S0)                  # no /slots snapshot -> invalid
    assert R.residents({"0": 60512, "1": 0, "2": None, "3": 7}, 3) == [0]             # own slot and empty slots excluded


def test_occupancy_numerics():
    # fork A/A exact, child-vs-coldE inside the A-B floor
    lp = _lp({"child": 0.0, "child2": 0.0, "coldE": 0.01, "coldA": 0.0, "coldB": 0.0275, "childX": 0.0})
    r = _ev(lp)
    assert r["verdict"] == "OCCUPANCY_NUMERICS" and r["predeclared_rule_satisfied"] is True
    assert r["d_child_vs_child2"] == 0.0 and r["child_vs_child2_exact_equal"] == 64
    assert r["floor_used"] == pytest.approx(0.0275) and r["d_child_vs_coldE"] == pytest.approx(0.01)
    assert len(r["d"]) == 10 and r["n_common_valid"] == 64 and r["floor_pair"] == "coldA|coldB"
    assert r["floor_matches_r3"] is False and r["r3_floor_raw"] == pytest.approx(0.027456215011625318)
    assert r["factorial_reading"]["label"] == "CHILDX_RECORDED"


def test_fork_path_specific_between_and_not_reproducible():
    # child-vs-coldE above every cold pair (A-B 0.0275, A-E 0.001, B-E 0.0265)
    lp = _lp({"child": 0.04, "child2": 0.04, "coldE": 0.001, "coldA": 0.0, "coldB": 0.0275, "childX": 0.04})
    r = _ev(lp)
    assert r["verdict"] == "FORK_PATH_SPECIFIC" and r["cold_max"] == pytest.approx(0.0275)
    assert r["d_child_vs_coldE"] == pytest.approx(0.039) and r["ratio_dCE_over_floor"] > 1
    # coldE is its own occupancy class far from A and B: cold_max widens past floor, dCE sits between them
    lp = _lp({"child": 0.04, "child2": 0.04, "coldE": 0.07, "coldA": 0.0, "coldB": 0.0275, "childX": 0.0})
    r = _ev(lp)
    assert r["verdict"] == "BETWEEN_FLOOR_AND_COLD_SPREAD" and r["floor_used"] < r["d_child_vs_coldE"] <= r["cold_max"]
    assert r["cold_max_pair"] == "coldA|coldE"
    # any difference between child and child2 decides first, even when dCE is inside the floor
    lp = _lp({"child": 0.0, "child2": 0.0, "coldE": 0.0, "coldA": 0.0, "coldB": 0.0275, "childX": 0.0})
    lp["child2"][7] += 1e-6
    r = _ev(lp)
    assert r["verdict"] == "FORK_PATH_NOT_REPRODUCIBLE" and r["d_child_vs_child2"] > 0
    assert r["predeclared_rule_satisfied"] is False


def test_floor_lift_invalid_and_abstain():
    same = _lp({a: 0.0 for a in R.ALL_ARMS})
    r = _ev(same)
    assert r["floor_raw"] == 0.0 and r["floor_used"] == A.FLOOR_EPS and r["verdict"] == "OCCUPANCY_NUMERICS"
    bad = {**GOOD_FIRST, "coldB": {"cache_n": 59999, "prompt_n": 513}}
    r = _ev(same, first=bad)
    assert r["verdict"] == "INVALID" and r["arm_invalid_reasons"]["coldB"] and r["d"] is None
    for k in ("floor_used", "cold_max", "d_child_vs_coldE", "d_child_vs_child2"):
        assert k in r and r[k] is None
    r = _ev(same, occ={**GOOD_OCC, "coldE": _occ((0, 2))})
    assert r["verdict"] == "INVALID" and r["arm_invalid_reasons"]["coldE"]
    sparse = {a: [v if i < 10 else None for i, v in enumerate(xs)] for a, xs in same.items()}
    r = _ev(sparse)
    assert r["verdict"] == "ABSTAIN" and r["n_common_valid"] == 10
    assert R.eval_r4(same, GOOD_FIRST, GOOD_OCC, T=T, n_task=40000, S0=S0)["verdict"] in ("ABSTAIN", "INVALID")
    # an invalid childX never blocks the rule; its reading is then not established
    r = _ev(same, first={**GOOD_FIRST, "childX": {}})
    assert r["verdict"] == "OCCUPANCY_NUMERICS" and r["factorial_reading"]["label"] == "CHILDX_NOT_ESTABLISHED"
    assert r["arm_valid"]["childX"] is False


def test_factorial_reading_cells():
    lp = _lp({"child": 0.03, "child2": 0.03, "coldE": 0.0, "coldA": 0.0, "coldB": 0.0275, "childX": 0.02})
    f = _ev(lp)["factorial_reading"]
    c = f["cells"]
    assert c["fork_vs_cold_at_child_occupancy"]["mean_abs_dlogprob"] == pytest.approx(0.03)
    assert c["fork_vs_cold_at_coldA_occupancy"]["mean_abs_dlogprob"] == pytest.approx(0.02)
    assert c["fork_across_occupancy"]["mean_abs_dlogprob"] == pytest.approx(0.01)
    assert c["cold_across_occupancy"]["n_exact_equal"] == 64 and f["childX_vs_coldA_within_floor"] is True
    assert f["weight"].startswith("zero")


def test_cross_load_reading_is_recorded_only():
    now = _lp({"child": 0.0, "coldA": 0.0, "coldB": 0.0})
    assert R.cross_load(now, now)["reproduces_r3_exactly"] is True
    base = _lp({"child": 0.0, "coldA": 0.01, "coldB": 0.0})
    base["coldB"][3] = None
    x = R.cross_load(now, base)
    assert x["pairs"]["child"]["mean_abs_dlogprob"] == 0.0 and x["pairs"]["coldA"]["mean_abs_dlogprob"] == pytest.approx(0.01)
    assert x["pairs"]["coldB"]["n_positions"] == 63 and x["reproduces_r3_exactly"] is False and "never pooled" in x["weight"]


def test_missing_result_line_is_a_fail():
    full = {"item3_r4": {k[1]: 0 for k in R.RESULT_LINES if k[0] == "item3_r4"}, "run_valid": {}}
    assert R.missing_lines(full) == []
    del full["item3_r4"]["occupancy_before"]
    del full["run_valid"]
    assert R.missing_lines(full) == ["item3_r4.occupancy_before", "run_valid"]


# ================================================================ the step end to end on the fake server
def mk_env(tmp_path, **over):
    ctx, env = TS.mk_run_env(tmp_path, **over)
    ctx.args.r4_steps = ",".join(R.STEP_KEYS)
    return ctx, env


def _records(ctx):
    return [json.loads(p.read_text()) for p in sorted((ctx.out / "records").glob("*.json"))]


def test_step_end_to_end_on_the_fake_server_and_verdict(tmp_path):
    ctx, env = mk_env(tmp_path, s1r_tf_positions=20)
    R.run_r4(env)
    recs = _records(ctx)
    assert len(recs) == 1
    sha = S.sha256_path(ctx.args.s1r_predeclaration)
    r = recs[0]
    assert r["schema"] == "epyc.kpf.run.v1" and r["kpf18_item"] == 187 and r["rows_sha256"] and "argv" in r
    x = r["r4"]
    assert x["step_complete"] and x["predeclaration_sha256"] == sha and x["fresh_load"] is True
    res = x["result"]
    assert all(res["arm_valid"].values()), res["arm_invalid_reasons"]
    assert res["residents_seen"] == {a: list(v) for a, v in R.RESIDENTS.items()}
    assert res["first"]["coldE"]["cache_n"] == 0 and res["first"]["child2"]["n_fork_tokens"] == 60000
    assert res["first"]["childX"]["fork_src_slot"] == 0 and res["occupancy_before"]["childX"]["3"] == 0
    dc = res["decoy"]
    assert dc["lcp_with_P"] < 1024 and dc["first"]["cache_n"] == 0 and dc["first"]["prompt_n"] == dc["n"]
    assert res["occupancy_before"]["childX"]["1"] == dc["n"] == res["occupancy_before"]["coldA"]["1"]   # same cells as coldA's
    assert res["factorial_reading"]["label"] == "CHILDX_RECORDED"
    assert res["n_common_valid"] == 20 and len(res["d"]) == 10
    assert res["verdict"] in ("OCCUPANCY_NUMERICS", "FORK_PATH_SPECIFIC", "FORK_PATH_NOT_REPRODUCIBLE",
                              "BETWEEN_FLOOR_AND_COLD_SPREAD")
    rv = {"valid": True, "graphs": "ON", "vram_trip": None, "gate_rc": 0}
    r3 = tmp_path / "r3.json"
    r3.write_text(json.dumps({"r3": {"result": {"lp": res["lp"]}}}))
    v = R.verdict(ctx.out, sha, run_valid=rv, r3_record=str(r3))
    assert v["verdict"] == "COMPLETE" and v["missing_result_lines"] == []
    assert v["item3_r4"]["cross_load_vs_r3"]["reproduces_r3_exactly"] is True
    assert R.verdict(ctx.out, sha, run_valid={**rv, "valid": False}, r3_record=str(r3))["verdict"] == "RUN_INVALID"
    assert R.verdict(ctx.out, sha, r3_record=str(r3))["verdict"] == "FAIL_MISSING_RESULT_LINE"      # no run_valid line
    v2 = R.verdict(ctx.out, "0" * 64, run_valid=rv)
    assert v2["verdict"] == "INCOMPLETE" and len(v2["excluded_predeclaration_mismatch"]) == 1
    # the R3 and remainder verdict readers never pick R4 records up
    import kpf18_r3 as R3
    assert R3.verdict(ctx.out, sha)["records_seen"] == 0 and S.verdict(ctx.out, sha)["records_seen"] == 0


def test_no_records_is_incomplete(tmp_path):
    v = R.verdict(tmp_path, "a" * 64, run_valid={"valid": True})
    assert v["verdict"] == "INCOMPLETE" and v["item_verdicts"] == {"item3_r4": "INCOMPLETE"} and v["missing_result_lines"] == []


def test_budget_guard_refuses_r4_when_it_does_not_fit(tmp_path):
    ctx, env = mk_env(tmp_path, budget_min=5.0)
    steps = R.body_187(ctx, GATE, env)
    with pytest.raises(GATE.Abort, match="not started"):
        steps[0].fn()
    assert not list((ctx.out / "records").glob("*.json")) if (ctx.out / "records").exists() else True


def test_stop_file_refuses_the_r4_step(tmp_path, monkeypatch):
    ctx, env = mk_env(tmp_path)
    stop = tmp_path / "STOP"
    stop.write_text("")
    monkeypatch.setenv("KPF18_S1R_STOPFILE", str(stop))
    with pytest.raises(GATE.Abort, match="STOP file"):
        R.body_187(ctx, GATE, env)[0].fn()


def test_launcher_predeclaration_and_forced_file_when_present():
    d = Path("/mnt/raid0/llm/tmp/f8-launchers-20261024/kpf18-s1-r4")
    f = d / "forced_seq_183p4.json"
    if not f.exists():
        pytest.skip("launcher directory not on this host")
    doc, probs = S.load_forced(f, S.FORCED_SHA256)
    assert probs == [] and doc["prompt"]["ids_sha256"].startswith("74ee91a9899677a3")
    txt = (d / "PREDECLARATION.md").read_text()
    for w in ("OCCUPANCY_NUMERICS", "FORK_PATH_SPECIFIC", "FORK_PATH_NOT_REPRODUCIBLE", "BETWEEN_FLOOR_AND_COLD_SPREAD",
              "FAIL_MISSING_RESULT_LINE"):
        assert w in txt
    for path in R.RESULT_LINES:
        assert "`" + ".".join(path) + "`" in txt, path
