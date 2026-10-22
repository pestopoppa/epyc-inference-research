import json

import pytest

from scripts.speech.stt_bakeoff import fitting_screen as fs


def items(k, n, unp=0, inelig=0):
    out = [{"item_id": f"a{i}", "eligible": True, "decision": "accept_ref"} for i in range(k)]
    out += [{"item_id": f"h{i}", "eligible": True, "decision": "accept_hyp"} for i in range(n - k - unp)]
    out += [{"item_id": f"u{i}", "eligible": True, "decision": "unparseable"} for i in range(unp)]
    out += [{"item_id": f"x{i}", "eligible": False, "decision": "accept_ref"} for i in range(inelig)]
    return out


def test_threshold_boundary():
    assert fs.label_for(0.10) is None
    assert fs.label_for(0.1001) == fs.LABEL_UNTRUSTWORTHY
    r = fs.screen_record("a", items(10, 100), model_revision="r", panel="p", predictions_sha256="0" * 64)
    assert r["accept_ref"] == 0.10 and r["label"] is None
    r = fs.screen_record("a", items(11, 100), model_revision="r", panel="p", predictions_sha256="0" * 64)
    assert r["label"] == fs.LABEL_UNTRUSTWORTHY


def test_unparseable_in_denominator_and_ineligible_excluded():
    r = fs.accept_ref(items(10, 100, unp=20, inelig=7))
    assert r["n_eligible"] == 100 and r["n_unparseable"] == 20 and r["accept_ref"] == 0.10


def test_parse_bad_lines_become_unparseable():
    got = fs.parse_decisions(['{"item_id":"1","eligible":true,"decision":"accept_ref"}', "not json",
                              '{"item_id":"3","eligible":true,"decision":"bogus"}', ""])
    assert [g["decision"] for g in got] == ["accept_ref", "unparseable", "unparseable"]
    assert fs.accept_ref(got)["n_eligible"] == 3


def test_non_en_unmeasured():
    for lang in ("it", "de", "fr"):
        r = fs.screen_record("a", items(50, 100), model_revision="r", panel="p",
                             predictions_sha256="0" * 64, language=lang)
        assert r["status"] == "unmeasured" and r["accept_ref"] is None and r["label"] is None


def test_wilson_known_values():
    lo, hi = fs.wilson_interval(0, 31)
    assert lo == 0.0 and hi == pytest.approx(0.1102, abs=5e-4)  # matches tool README [0.000, 0.110]
    lo, hi = fs.wilson_interval(192, 512)
    assert (lo, hi) == pytest.approx((0.3342, 0.4177), abs=2e-3)
    assert fs.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = fs.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.7225, abs=1e-3)


def test_record_fields_and_no_gate():
    r = fs.screen_record("whisper-large-v3", items(1, 50), model_revision="rev", panel="abc123",
                         predictions_sha256="1" * 64)
    for k in ("arm", "model_revision", "probe", "accept_ref", "accept_ref_ci", "n_eligible", "n_unparseable",
              "panel", "predictions_sha256", "language", "stratum", "metric_direction", "gates"):
        assert k in r
    assert r["gates"] is False and r["is_control_baseline"] is True
    assert "higher accept-ref = more fitting" in r["metric_direction"]


def test_plan_builds_argv_executes_nothing(tmp_path):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    p1.write_text("x")
    p2.write_text("y")
    plan = fs.build_screen_plan(["a", "b"], {"a": p1, "b": p2}, "deadbeef", panel_models=["m1", "m2", "m3"])
    assert plan["executed"] is False
    assert plan["arms"]["a"]["predictions_sha256"] == fs.sha256_file(p1) != plan["arms"]["b"]["predictions_sha256"]
    assert plan["ref_disagreement_argv"][:2] == ["benchmark-optimization", "ref-disagreement"]
    assert "m1,m2,m3" in plan["ref_disagreement_argv"]
    assert plan["tool_commit"] == fs.BENCHMARK_OPTIMIZATION_COMMIT
    json.dumps(plan)
    with pytest.raises(KeyError):
        fs.build_screen_plan(["zzz"], {}, "c")


def test_module_imports_no_subprocess():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(fs))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert not mods & {"subprocess", "os", "multiprocessing", "socket"}
