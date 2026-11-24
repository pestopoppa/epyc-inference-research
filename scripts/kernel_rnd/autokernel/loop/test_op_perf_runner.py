"""RSL-R3 live op-perf runner: parse, plan, floors, weights, every decision path in test mode (no binary runs)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from autokernel.loop import op_perf_runner as opr
from autokernel.loop import op_perf_screen as ops
from autokernel.loop import served_shape_cases as ssc
from autokernel.loop import stage_bodies as sb

GDN = Path(__file__).resolve().parents[4] / "data/k28_gdn_perf/k28-phase0-op-rerun-20260720T102526Z/gated_delta_net_perf.console.txt"
N = [1, 3, 4, 8, 16]
SPEC27 = {"key": "27b-q8_0|mi210", "gguf": ssc.DENSE_PROFILES["q38_27b"][0], "lane": "gpu", "n": N, "model_role": "qwen38_27b_q8_local", "device": "mi210"}
FN = ssc.LANE_PROFILES["q38fn"].model
SPECFN = {"key": "q38fn-ud-iq4xs-cpu|cpu", "gguf": FN, "lane": "cpu", "n": [1, 3], "model_role": "qwen38_flash_next_ud_iq4xs_local", "device": "cpu"}
BUILD = Path("/nonexistent-build")


def cells27(fam="MUL_MAT", **dims):
    return ops.grid({"op_family": fam, "dims": dims}, ssc.lane_served_shapes("q38_27b"), N)


def test_parse_real_perf_console_and_mul_mat_format():
    rows = opr.parse_perf(GDN.read_text())   # VERBATIM `test-backend-ops perf` console from this host (k28 filed row)
    assert len(rows) == 12 and rows[0][0] == "GATED_DELTA_NET" and rows[0][2] == 6.33 and rows[3][2] == 152.90
    cell = cells27(k=[5120], m=[1024])[0]   # MUL_MAT lines use the flops branch of print_test_result (same C++ printf)
    line = f"  MUL_MAT({opr.vars_prefix(cell, 'Q8_0')}):      1234 runs -  12.34 us/run -  10.49 MFLOP/run - \033[1;34m  1.20 TFLOPS\033[0m"
    assert opr.parse_perf(line)[0][0] == "MUL_MAT" and opr.parse_perf(line)[0][2] == 12.34


def test_grid_matches_ggufs_times_nset_and_cells_are_sum_over_types():
    assert len(opr.full_grid(SPEC27)) == 9 * len(N)
    fn = ops.grid({"op_family": "MUL_MAT_ID", "dims": {"k": [2560]}}, ssc.lane_served_shapes("q38fn"), [1])
    plan = opr.plan_cells(SPECFN, fn)
    assert all(len(rx) >= 3 for _, rx in plan)   # gate_up: IQ3_S, IQ4_XS, plus the operator-listed Q6_K extra
    c, rxs = plan[0]
    lines = [("MUL_MAT_ID", opr.vars_prefix(c, t) + ",max_nmse=0.0005", 10.0 * (i + 1)) for i, (t, _) in enumerate(rxs)]
    assert opr.cell_us(plan, lines) == {c["key"]: sum(10.0 * (i + 1) for i in range(len(rxs)))}
    assert opr.cell_us(plan, lines[:-1]) == {}                       # one type missing: the cell is unmeasured, not partial
    with pytest.raises(opr.RunnerRefused):                            # two lines for one case: ambiguous, refuse
        opr.cell_us(plan, lines + [lines[0]])


def _run(outcome, floors="ones", pairs=5, cells=None, spec=SPEC27):
    cells = cells or cells27(k=[5120], m=[1024, 6144])
    r = opr.synth_runner(cells, spec, BUILD / "a" / "bin" / "test-backend-ops", outcome)
    fl = {c["key"]: 1.0 for c in cells} if floors == "ones" else floors
    res = opr.run(BUILD / "a", BUILD / "c", cells, spec, pairs=pairs, runner=r, live=False, floors=fl,
                  census={"records": {}})
    for c in res["cells"]:
        assert len(c["anchor"]) == len(c["candidate"]) == (0 if outcome == "missing-line" and c["key"] == cells[0]["key"] else max(pairs, ops.MIN_PAIRS))
    return res, cells


def _decide(res, weights=None):
    return ops.decide(res["cells"], weights or {c["key"]: 1.0 for c in res["cells"]})


def test_pass_regress_unmeasured_and_missing_floor_paths():
    res, cells = _run("pass")
    assert _decide(res)["screen"] == ops.PASS
    res, _ = _run("regress")
    d = _decide(res)
    assert d["screen"] == ops.REJECT and d["regressed"] == [cells[0]["key"]]
    assert ops.polar_result(d)["polar"] == "FAIL"
    res, _ = _run("missing-line")
    d = _decide(res)
    assert d["screen"] == ops.UNMEASURED and cells[0]["key"] in d["unmeasured"]
    res, _ = _run("pass", floors={})                                  # no floor file: every cell unmeasured -> harness futility
    d = _decide(res)
    assert d["screen"] == ops.UNMEASURED and "no paired floor" in d["reason"]
    assert "polar" not in ops.polar_result(d) and ops.polar_result(d)["verdict"].startswith("FUTILITY_HARNESS_INSTRUMENT")
    res, _ = _run("pass", pairs=3)                                    # fewer than MIN_PAIRS is raised to MIN_PAIRS
    assert res["pairs"] == ops.MIN_PAIRS


@pytest.mark.parametrize("outcome", ["bad-rc", "no-backend"])
def test_runner_refuses_bad_exit_and_wrong_backend(outcome):
    with pytest.raises(opr.RunnerRefused) as e:
        _run(outcome)
    assert e.value.token == "FUTILITY_HARNESS_INSTRUMENT"


def test_arm_order_alternates_abba():
    seen = []
    cells = cells27(k=[5120], m=[1024])
    inner = opr.synth_runner(cells, SPEC27, BUILD / "a" / "bin" / "test-backend-ops")
    opr.run(BUILD / "a", BUILD / "c", cells, SPEC27, pairs=5, live=False, floors={}, census={"records": {}},
            runner=lambda argv, **kw: (seen.append("A" if "/a/" in " ".join(argv) else "C"), inner(argv, **kw))[1])
    assert "".join(seen) == "ACCAACCAAC"[: len(seen)] and len(seen) == 10   # one MUL_MAT run per arm per round


def test_live_refuses_a_binary_without_the_perf_case_set(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    (b / "test-backend-ops").write_bytes(b"no served shapes here")
    with pytest.raises(opr.RunnerRefused) as e:                       # raised BEFORE anything is launched
        opr.run(tmp_path, tmp_path, cells27(k=[5120], m=[1024]), SPEC27, store=tmp_path, census={"records": {}})
    assert "lacks" in str(e.value)
    with pytest.raises(sb.StageRefused):
        sb._live_op_perf({"path_spec": SPEC27, "test_registry": {}, "serving_path": SPEC27["key"]}, tmp_path, tmp_path, [])


def test_calibrate_writes_floor_file_and_run_reads_it(tmp_path):
    cells = opr.full_grid(SPEC27)
    r = opr.synth_runner(cells, SPEC27, BUILD / "a" / "bin" / "test-backend-ops")
    with pytest.raises(opr.RunnerRefused):
        opr.calibrate(BUILD / "a", SPEC27, source_row="r", pairs=10, store=tmp_path, runner=r, live=False)   # < 24 pairs
    p = opr.calibrate(BUILD / "a", SPEC27, source_row="row-1", store=tmp_path, runner=r, live=False)
    rec = json.loads(p.read_text())
    assert p == opr.floor_path(tmp_path, SPEC27["key"]) and rec["schema"] == opr.FLOOR_SCHEMA and rec["n_pairs"] == 24
    assert set(rec["cells"]) == {c["key"] for c in cells} and rec["backend"] == "ROCm0" and rec["unit"] == opr.bench.FLOOR_UNIT
    one = next(iter(rec["cells"].values()))
    assert set(one["floor_pct"]) == {"5", "9", "20"} and len(one["anchor_us"]) == len(one["candidate_us"]) == 24
    fl = opr.load_floors(tmp_path, SPEC27, "ROCm0", 5)
    assert len(fl) == len(cells) and all(v > 0 for v in fl.values())
    assert opr.load_floors(tmp_path, SPEC27, "CPU", 5) == {} and opr.load_floors(tmp_path, SPEC27, "ROCm0", 7) == {}
    rec["n_pairs"] = 5
    p.write_text(json.dumps(rec))
    assert opr.load_floors(tmp_path, SPEC27, "ROCm0", 5) == {}        # an under-sampled floor never gates


def test_weights_exact_shape_median_and_none():
    rec = lambda shape, n, calls, share: {"model": "27b-q8_0", "shape": shape, "n": n, "calls_per_round": calls, "share_of_round": share}
    census = {"records": {"a": rec("1024x5120", 1, 32, 0.01), "b": rec("1024x5120", 4, 32, 0.02), "c": rec("1024x5120", 8, 32, 0.04)}}
    cells = cells27(k=[5120], m=[1024, 6144])
    w, basis = opr.weights(cells, SPEC27, census)
    k = lambda n: next(c["key"] for c in cells if c["n"] == n and c["dims"]["m"] == 1024)
    assert w[k(1)] == pytest.approx(0.32) and basis[k(1)] == "exact"
    assert w[k(3)] == pytest.approx(0.64) and basis[k(3)] == "shape-median"   # median of 0.32, 0.64, 1.28
    assert not any(c["key"] in w for c in cells if c["dims"]["m"] == 6144)    # no census row: unweighted -> unmeasured


def test_cli_test_mode_runs_no_binary(tmp_path, capsys):
    out = tmp_path / "o.json"
    assert opr.main(["--path", "27b-q8_0|mi210", "--anchor-build", str(BUILD / "a"), "--candidate-build", str(BUILD / "c"),
                     "--test-mode", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["decision"]["screening_only"] is True and res["decision"]["screen"] in (ops.PASS, ops.UNMEASURED)
    assert opr.main(["--path", "27b-q8_0|mi210", "--anchor-build", str(BUILD / "a"), "--test-mode", "--outcome", "bad-rc",
                     "--out", str(out)]) == 0 and json.loads(out.read_text())["verdict"].startswith("FUTILITY_HARNESS_INSTRUMENT")
