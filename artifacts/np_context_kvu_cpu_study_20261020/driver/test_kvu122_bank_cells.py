"""Server-free tests for kvu122_bank_cells.py (rule arithmetic, marker proof, plan, budget gating). Written 2026-10-22; run in a granted window:
  python3 -m pytest -q driver/test_kvu122_bank_cells.py        (no model, no server; sub-second)"""
import copy, json, sys, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvu122_bank_cells as K          # noqa: E402
import kvu_ckpt_arms as A              # noqa: E402


def good_gate(mode, ratio=1.0):
    live = dict(prompt_n=7808, draft_n=300, draft_acc=150, predicted_n=256, decode_tps=20.0, err=None)
    rest = dict(prompt_n=128, cache_n=7680, draft_n=300, draft_acc=int(150 * ratio), predicted_n=256, decode_tps=20.0 * ratio, err=None)
    g = dict(mode=mode, D=7680, extend=dict(prompt_n=1024, expected_prompt_n=1024, ok=True))
    for s in "AB":
        g[f"live{s}"], g[f"restored{s}"], g[f"identical{s}"] = dict(live), dict(rest), True
    return g


def args(tmp, **kw):
    d = dict(out=str(tmp), cpu_list="48-71,144-167", threads=24, port=18972, lens="8k", waves=3, n_cells=4, skip_control=False, stop_file="",
             deadline_epoch=0, hard_end_epoch=0)
    d.update(kw)
    return SimpleNamespace(**d)


def test_rule_passes_when_restored_matches_live():
    v = K.gate_verdict([good_gate("split"), good_gate("unified")])
    assert v["passed"] and not v["failed"]


def test_rule_fails_on_acceptance_and_on_decode_and_on_identity():
    g = good_gate("split", ratio=0.6)
    v = K.gate_verdict([g, good_gate("unified")])
    assert not v["passed"] and any("acceptance ratio" in f for f in v["failed"]) and any("decode ratio" in f for f in v["failed"])
    g2 = good_gate("split"); g2["identicalB"] = False
    assert any("gate (a)" in f for f in K.gate_verdict([g2])["failed"])
    g3 = good_gate("split"); g3["extend"]["ok"] = False
    assert any("gate (c)" in f for f in K.gate_verdict([g3])["failed"])


def test_boundary_values_are_inclusive():
    g = good_gate("split"); g["restoredA"]["draft_acc"] = 135; g["restoredA"]["decode_tps"] = 19.0     # exactly 0.9 and 0.95
    v = K.gate_verdict([g])
    assert not [f for f in v["failed"] if f.startswith("split/A")]


def test_missing_decode_tps_is_not_a_pass():
    g = good_gate("split")
    for s in "AB":
        del g[f"live{s}"]["decode_tps"]
    assert any("not captured" in f for f in K.gate_verdict([g])["failed"])


def test_v10_study_record_fails_the_rule_on_acceptance():
    gates = json.loads(K.STUDY_V10_GATES.read_text())
    v = K.gate_verdict(gates)
    assert not v["passed"]
    assert all(r["acc_ratio"] < 0.7 for r in v["rows"]), [r["acc_ratio"] for r in v["rows"]]       # PLAN.md: 0.55-0.65


def _write_cand(tmp, gates, markers):
    d = Path(tmp) / "gates_cand"; (d / "gate_split_np2").mkdir(parents=True)
    (d / "gates.json").write_text(json.dumps(gates))
    (d / "gate_split_np2" / "server.log").write_text(markers)


def test_verdict_requires_the_draft_state_path_to_have_run(tmp_path):
    gates = [good_gate("split"), good_gate("unified")]
    _write_cand(tmp_path, gates, "nothing relevant\n")
    v = K.compute_verdict(args(tmp_path, skip_control=True))
    assert not v["candidate_passes"] and not v["proceed"] and any("vacuous" in f for f in v["candidate"]["failed"])
    _write_cand(tmp_path, gates, "slot 0 appended the draft state (1.0 MiB)\nslot 0 restored the draft state (1.0 MiB)\n")
    v = K.compute_verdict(args(tmp_path, skip_control=True))
    assert v["candidate_passes"] and v["control_discriminates"] and v["proceed"]


def test_non_discriminating_control_blocks_proceeding(tmp_path):
    _write_cand(tmp_path, [good_gate("split")], "appended the draft state\nrestored the draft state\n")
    ctrl = Path(tmp_path) / "gates_control"; ctrl.mkdir()
    (ctrl / "gates.json").write_text(json.dumps([good_gate("split"), good_gate("unified")]))     # control that does NOT show the limitation
    v = K.compute_verdict(args(tmp_path))
    assert v["candidate_passes"] and not v["control_discriminates"] and not v["proceed"]


def test_plan_stage_names_estimates_and_commands(tmp_path):
    a = args(tmp_path)
    st = K.build_stages(a)
    names = [s.name for s in st]
    assert names[:2] == ["gates_control", "gates_cand"]
    assert {"bank_unified", "bank_split_np4", "bank_split_np2", "cells_unified_np4", "cells_split_np4", "cells_unified_np2", "cells_split_np2"} <= set(names)
    assert all(s.est_raw_s > 0 for s in st)
    cells = [s for s in st if s.name.startswith("cells_")]
    assert all("--from-bank" in s.argv and "--binary" in s.argv and str(K.CAND_DIR / "llama-server") in s.argv for s in cells)
    gates = [s for s in st if s.name.startswith("gates_")]
    assert str(K.CTRL_DIR / "llama-server") in gates[0].argv and str(K.CAND_DIR / "llama-server") in gates[1].argv
    # 24 threads: gates about 40-50 min each pair at the D4 curve (FILLERS 45 min for the gate cells)
    g_all = sum(s.est_raw_s for s in gates) / 60
    assert 20 <= g_all <= 90, g_all
    assert [s.name for s in K.build_stages(args(tmp_path, skip_control=True))][0] == "gates_cand"
    assert len([s for s in K.build_stages(args(tmp_path, n_cells=2)) if s.name.startswith("cells_")]) == 2


def test_stage_not_started_when_it_cannot_finish(tmp_path):
    a = args(tmp_path)
    st = K.build_stages(a)[0]
    budget = A.Budget("", int(time.time()) + 300)            # 5 minutes left: a ~25 min stage must not start
    rc = K.run_stage(st, a, {}, A.Calibrator(), budget)
    assert rc == 9 and not (Path(st.out) / "stage.log").exists()


def test_done_stage_is_skipped_and_stop_file_blocks(tmp_path):
    a = args(tmp_path)
    st = K.build_stages(a)[0]
    assert K.run_stage(st, a, {st.name: dict(rc=0)}, A.Calibrator(), A.Budget("", 0)) == 0
    stop = tmp_path / "STOP"; stop.write_text("x")
    assert K.run_stage(st, a, {}, A.Calibrator(), A.Budget(str(stop), 0)) == 9


def test_production_port_refused():
    try:
        K.main(["plan", "--out", "/tmp/x", "--port", "8083"])
    except SystemExit as e:
        assert "production port" in str(e)
    else:
        raise AssertionError("port 8083 accepted")
