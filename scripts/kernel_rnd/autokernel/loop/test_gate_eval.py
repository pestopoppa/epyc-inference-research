"""gate_eval: deterministic RESULT.json -> verdict (AK parallel loop P1)."""
from __future__ import annotations

import json

from autokernel.loop import gate_eval as ge, gate_eval_replay as rp

POLICY = ge.load_policy()
OK = {"rc": "0", "reason": "complete"}


def ev(result, **kw):
    kw.setdefault("ended", OK)
    kw.setdefault("policy", POLICY)
    return ge.evaluate(result, **kw)


def test_policy_file_is_the_27b_anchored_gate():
    gpu = {b["model"]: b for b in POLICY["backends"]["gpu"]}
    assert gpu["q38_27b"]["max_mean_kld"] == 0.003 and gpu["q38_27b"]["max_abs_dppl_pct"] == 0.5
    assert gpu["q38fn"]["min_same_top_pct"] == 93.0


def test_graph_kld_pass_on_c17_shape():
    r = {"at": "2026-11-21T10:00:00Z", "verdict": "PASS", "kld": {
        "q27b": {"mean_kld": -0.0, "ppl_rel": 4.4e-07},
        "q38fn_info_only": {"same_top1_pct": 94.3, "mean_kld": 0.026}}}
    v = ev(r, row="c17-fold-gpu-20261121-r3")
    assert v["verdict"] == "PASS" and v["filed_verdict"] == "pass"
    assert v["gate_id"] == "graph_kld.gpu.q38_27b"
    assert {c["metric"] for c in v["checks"]} == {"mean_kld", "abs_dppl_pct", "same_top_pct"}


def test_policy_fail_overrides_declared_pass_and_names_threshold():
    r = {"at": "2026-11-21", "verdict": "PASS", "gates": {"27B": {"mean_kld": 0.0041, "ppl_rel": 0.001}}}
    v = ev(r, row="x-cpu-20261121")
    assert v["verdict"] == "FAIL" and v["gate_id"] == "graph_kld.cpu.q38_27b"
    bad = [c for c in v["checks"] if not c["passed"]][0]
    assert bad["metric"] == "mean_kld" and bad["threshold"] == 0.003 and bad["observed"] == 0.0041


def test_q38fn_top1_floor_and_fraction_normalisation():
    r = {"at": "2026-11-21", "verdict": "KEEP", "gates": {"Q38FN": {"top1_agree": 0.92}}}
    v = ev(r, row="y-gpu-20261121")
    assert v["verdict"] == "FAIL" and v["checks"][0]["observed"] == 92.0


def test_policy_not_applied_before_its_ruling_date():
    r = {"at": "2026-11-16", "verdict": "PASS", "gates": {"27B": {"mean_kld": 0.02}}}
    v = ev(r, row="z-gpu-20261116")
    assert v["verdict"] == "PASS" and v["checks"] == []


def test_anchor_and_aa_blocks_are_not_the_candidate():
    r = {"at": "2026-11-21", "verdict": "PASS", "kld": {"aa_27b": {"mean_kld": 0.5}, "anchor_27b": {"mean_kld": 0.5}}}
    assert ev(r, row="w-gpu-20261121")["verdict"] == "PASS"


def test_row_geomean_speed_floor():
    r = {"at": "2026-11-21", "verdict": "PASS", "geomean": 0.990, "rule": {"geomean_min": 0.995}}
    v = ev(r, row="s-cpu-20261121")
    assert v["verdict"] == "FAIL" and v["gate_id"] == "speed.row_geomean_floor"


def test_bench_floor_noise_matches_run_noise_floor_rule():
    floors = ge._constants()["floors"]
    assert ge.speed_floor_pct(floors, "tg128", 5) == 2.422      # measured row dominates
    assert ge.speed_floor_pct(floors, "pp512", 9) == round(2.175 / 3, 4)  # sigma/sqrt(n) dominates
    sp = {"surface": "tg128", "pairs": 5, "effect_pct": 1.0, "model": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"}
    v = ev({"verdict": "KEEP", "speed": sp}, row="b-gpu")
    assert v["verdict"] == "NOISE" and v["filed_verdict"] == "fail"
    sp["model"] = "other.gguf"     # floor is the 1.5B's calibration only: no decisive speed gate
    assert ev({"verdict": "KEEP", "speed": sp}, row="b-gpu")["verdict"] == "PASS"


def test_run_end_and_lifecycle_layers():
    assert ev(None)["verdict"] == "FUTILITY"
    assert ev({"verdict": "PASS"}, ended={"rc": "75", "reason": "stopped"})["filed_verdict"] == "inconclusive"
    assert ev({"verdict": "FAIL(crash)"}, ended={"rc": "75", "reason": "stopped"})["verdict"] == "FAIL"
    assert ev({"verdict": "PASS"}, ended={"rc": "0", "reason": "direct-run"})["verdict"] == "PASS"
    assert ev({"verdict": "PASS", "complete": False})["verdict"] == "FUTILITY"
    assert ev({}, lifecycle={"superseded_by": "x-r2"})["filed_verdict"] == "SUPERSEDED"
    assert ev({}, lifecycle={"withdrawn": "owner"})["filed_verdict"] == "WITHDRAWN"


def test_declared_vocabulary():
    c = ge.classify_token
    assert c("KEEP_NEUTRAL")[0] == "PASS" and c("ANSWERED")[0] == "PASS"
    assert c("DROP(no gain)")[0] == "FAIL" and c("NO_GAIN")[0] == "FAIL"
    assert c("FUTILITY_BELOW_BASELINE")[0] == "FAIL"
    assert c("FUTILITY_RESIDENCY")[0] == "FUTILITY" and c("FAIL_FOREIGN_LOAD")[0] == "FUTILITY"
    assert c("DROP(diagnostic: NOT_ENGAGED)")[0] == "FUTILITY"
    assert c("FAIL(visits invalid/missing: [...])")[0] == "FUTILITY"
    assert c("FAIL(KLD gate: mean KLD 0.025 > 0.01; not measured on 2)")[0] == "FAIL"
    assert c("DETM_pp-1550-LOSS_1500-LOSS")[0] == "FAIL"
    assert c("NOT_JUDGEABLE")[0] == "FUTILITY" and c("")[0] is None


def test_no_declared_verdict():
    assert ev({"complete": True, "census": 3})["verdict"] == "PASS"
    assert ev({"ok": False})["verdict"] == "FAIL"
    assert ev({"x": 1})["verdict"] == "FUTILITY"


def test_evaluate_is_pure_and_cli_writes_verdict(tmp_path, capsys):
    r = {"at": "2026-11-21", "verdict": "PASS", "gates": {"27B": {"mean_kld": 0.001, "ppl_rel": 0.0}}}
    assert ev(r, row="p-gpu-20261121") == ev(json.loads(json.dumps(r)), row="p-gpu-20261121")
    res = tmp_path / "RESULT.json"
    res.write_text(json.dumps(r))
    end = tmp_path / "x.ended"
    end.write_text("rc=0 time=2026-11-21T00:00:00Z reason=complete\n")
    out = tmp_path / "verdict.json"
    assert ge.main([str(res), "--ended", str(end), "--row", "p-gpu-20261121", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["filed_verdict"] == "pass"


def test_replay_lifecycle_rules():
    ids = {"a-20261120-r2", "b-20261120-r3q0"}
    assert rp.successor_by_name("a-20261120", ids) == "a-20261120-r2"
    assert rp.successor_by_name("b-20261120-r2q0", ids) == "b-20261120-r3q0"
    assert rp.lifecycle({"id": "a-20261120"}, OK, {}, True, ids) == {}   # completed: keeps its verdict
    assert rp.lifecycle({"id": "a-20261120"}, {"rc": "75", "reason": "stopped"}, {}, False, ids) == \
        {"superseded_by": "a-20261120-r2"}
    assert rp.lifecycle({"id": "c"}, {"rc": "75", "reason": "operator_said_drop"}, {}, False, set()) == \
        {"withdrawn": "operator_said_drop"}
    assert rp.norm("BIT_EXACT_PASS") == "pass" and rp.norm("superseded") == "SUPERSEDED"
