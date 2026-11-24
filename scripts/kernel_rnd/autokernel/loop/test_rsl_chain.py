"""RSL-R3 (op-perf stage wiring), R6 (per-path timing cells), R8 (per-path quality policy): all fixture/--test-mode, no binary."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from autokernel.loop import census_coverage as cc
from autokernel.loop import gate_eval as ge
from autokernel.loop import served_shape_cases as ssc
from autokernel.loop import stage_bodies as sb

BASE, PATCH = "a" * 40, "b" * 40
REF = "ak/champion/llama-cpp-ffc1bac82eec"
M = "/mnt/raid0/llm/models/"
K27, A3B = "27b-q8_0|mi210", "a3b-q8_0-cpu|cpu"
A3B_GPU, FN = "a3b-q8_0|mi210", "q38fn-ud-iq4xs-cpu|cpu"
POLICY = ge.load_policy()


def _host(role, dev, gguf, port, dm, slots):
    e = {"model_role": role, "model": gguf, "port": port, "slots": slots,
         "acceleration": {"spec_type": "draft-mtp", "draft_max": dm}}
    return {**e, "device": dev} if dev else e


REG = {"server_mode": {  # today's served shape (registry host names, GGUFs, draft_max, slots); DS41 has no host
    "architect_critic": _host("qwen38_27b_q8_local", "ROCm0", M + "Qwen3.8-27B-Q8_0.gguf", 8083, 7, 4),
    "frontdoor": _host("qwen36_35b_a3b_mtp_q8_local", None, M + "Qwen3.6-35B-A3B-MTP-Q8_0.gguf", 8070, 4, 4),
    "architect_general": _host("qwen38_flash_next_ud_iq4xs_local", None, ssc.LANE_PROFILES["q38fn"].model, 8074, 4, 1),
    "worker": {"alias_of": "frontdoor", "model_role": "qwen36_35b_a3b_mtp_q8_local"}}}


def cand(tmp_path, path=K27, floors=(K27,), **kw):
    c = {"id": "cand-x-bbbbbbbbb", "base_sha": BASE, "patch_sha": PATCH, "champion_ref": REF, "build_root": str(tmp_path),
         "test_refs": {REF: BASE}, "test_parents": {PATCH: BASE}, "stage_results": {}, "serving_path": path,
         "test_registry": REG, "test_floor_paths": list(floors),
         "shape_class": {"op_family": "MUL_MAT", "dims": {"k": [5120], "m": [1024, 6144]}}}
    c.update(kw)
    return c


def run(c, tmp_path, stages, lane_of=lambda st, spec: "cpu" if st == "build" else spec["lane"] if st in sb.PATH_STAGES else "gpu"):
    prim, spec, out = sb.test_primitives(c), sb.path_specs(REG)[c["serving_path"]], {}
    for st in stages:
        env = {}
        r = sb.run_stage(st, c, lane=lane_of(st, spec), prim=prim, test_mode=True, env=env, cpus={0})
        (tmp_path / f"{st}.json").write_text(json.dumps(r))
        c["stage_results"][st] = str(tmp_path / f"{st}.json")
        out[st] = (r, env)
    return out


# ---- R3 ------------------------------------------------------------------------------------------------------

def _gguf_2d_weights(path):
    """Header-only GGUF read: {tensor name: (ne0, ne1, type id)} for 2-D tensors (no numpy, no tensor data)."""
    with open(path, "rb") as f:
        r = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        rs = lambda: f.read(r("<Q")[0]).decode("utf-8", "replace")
        assert f.read(4) == b"GGUF"
        r("<I"); nt, nk = r("<Q")[0], r("<Q")[0]
        fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def val(t):
            if t == 8:
                return rs()
            if t == 9:
                et, n = r("<I")[0], r("<Q")[0]
                for _ in range(n):
                    val(et)
                return None
            return f.read(fixed[t])
        for _ in range(nk):
            rs(); val(r("<I")[0])
        out = {}
        for _ in range(nt):
            name, nd = rs(), r("<I")[0]
            dims = r("<" + "Q" * nd); ty = r("<I")[0]; r("<Q")
            if nd == 2:
                out[name] = (*dims, ty)
        return out


@pytest.mark.skipif(not os.path.exists(ssc.DENSE_PROFILES["q38_27b"][0]), reason="27B GGUF not on this host")
def test_dense_lane_matches_gguf_header():
    w = _gguf_2d_weights(ssc.DENSE_PROFILES["q38_27b"][0])
    got = {(k, m) for n, (k, m, ty) in w.items() if ".nextn." not in n and not n.startswith("token_embd") and ty == 8}
    assert got == {(s.k, s.m) for s in ssc.lane_served_shapes("q38_27b")} and len(got) == 9
    assert "q38_27b" not in ssc.LANE_PROFILES and all("q38_27b" not in s.name for s in ssc.SERVED_SHAPES)
    assert ssc.dense_lane_for_model(ssc.DENSE_PROFILES["q38_27b"][0]) == "q38_27b"
    assert ssc.lane_for_model(ssc.DENSE_PROFILES["q38_27b"][0]) is None      # gates' lane lookup is unchanged


def test_op_perf_27b_grid_is_gguf_shapes_times_registry_n_set(tmp_path):
    c = cand(tmp_path, shape_class={"op_family": "MUL_MAT", "dims": {}})
    out = run(c, tmp_path, ("build", "op-perf"))
    cells = {x["key"] for x in out["op-perf"][0]["op_perf_screen"]["cells"]}
    n_set = sb.path_specs(REG)[K27]["n"]
    assert n_set == [1, 2, 3, 4, 5, 6, 7, 8]                                 # registry draft_max 7 -> n 1..8
    assert cells == {f"{s.name}@n{n}" for s in ssc.lane_served_shapes("q38_27b") for n in n_set}


def test_op_perf_screen_pass_is_never_a_keep_or_gate_verdict(tmp_path):
    out = run(cand(tmp_path), tmp_path, ("build", "op-perf"))
    r = out["op-perf"][0]
    assert r["screen_pass"] and r["screening_only"] and "polar" not in r and r["stage"] == "op-perf"
    assert r["verdict"].startswith("FUTILITY_SCREEN_ONLY_NO_KEEP")           # check_complete.py prefix contract holds
    v = ge.evaluate(r, ended={"rc": "0", "reason": "complete"}, row="akc-ak-x-op-perf-20261124")
    assert v["verdict"] == "FUTILITY" and v["filed_verdict"] == "inconclusive" and not v["policy_decided"]


def test_op_perf_fold_check_ignores_the_screen_and_a_pass_cannot_stand_in_for_gates(tmp_path):
    c = cand(tmp_path, test_outcomes={"ab-timing": "fail"})
    out = run(c, tmp_path, ("build", "oracle-kld", "op-perf", "ab-timing", "fold-check"))
    assert out["op-perf"][0]["screen_pass"] and "MISSING_INPUT" in out["fold-check"][0]["verdict"]


def test_op_perf_one_cell_regression_fails_and_unmeasured_is_harness(tmp_path):
    r = run(cand(tmp_path, test_outcomes={"op-perf": "fail"}), tmp_path, ("build", "op-perf"))["op-perf"][0]
    assert r["polar"] == "FAIL" and r["verdict"].startswith("FAIL_OP_PERF_REGRESSION") and not r.get("screen_pass")
    r = run(cand(tmp_path, test_outcomes={"op-perf": "harness"}), tmp_path, ("build", "op-perf"))["op-perf"][0]
    assert r["verdict"].startswith("FUTILITY_HARNESS_INSTRUMENT") and "polar" not in r


def test_op_perf_needs_shape_class_and_a_served_shape_lane(tmp_path):
    r = run(cand(tmp_path, shape_class=None), tmp_path, ("build", "op-perf"))["op-perf"][0]
    assert "MISSING_INPUT" in r["verdict"]
    r = run(cand(tmp_path, shape_class={"op_family": "MUL_MAT", "dims": {"bogus": [1]}}), tmp_path,
            ("build", "op-perf"))["op-perf"][0]
    assert "FUTILITY_HARNESS_INPUT" in r["verdict"]
    r = run(cand(tmp_path, path=A3B), tmp_path, ("build", "op-perf"))["op-perf"][0]   # no A3B lane profile: ungated
    assert r["ungated"] == A3B and "polar" not in r


def test_live_op_perf_runner_refuses_instead_of_inventing_cells():
    with pytest.raises(sb.StageRefused):
        sb._live_op_perf({}, Path("/a"), Path("/b"), [])


# ---- R6 ------------------------------------------------------------------------------------------------------

def test_path_specs_come_from_served_set_not_constants():
    specs = sb.path_specs(REG)
    assert set(specs) == {K27, FN, A3B, A3B_GPU} == {f"{m}|{d}" for m, d in cc.served_set(REG)}
    assert not any("ds41" in k.lower() for k in specs)
    assert (specs[K27]["device"], specs[K27]["lane"], specs[K27]["gguf"]) == ("mi210", "gpu", M + "Qwen3.8-27B-Q8_0.gguf")
    assert (specs[A3B]["lane"], specs[A3B]["status"], specs[A3B]["serving"]["port"]) == ("cpu", "live", 8070)
    assert specs[FN]["serving"]["port"] == 8074 and specs[FN]["gguf"] == ssc.LANE_PROFILES["q38fn"].model
    assert specs[A3B_GPU]["status"] == "planned" and specs[A3B_GPU]["serving"] is None       # no argv invented
    assert specs[A3B_GPU]["gguf"] == specs[A3B]["gguf"]
    assert not hasattr(sb, "TIMING_MODEL") and not hasattr(sb, "TIMING_SURFACE")


def test_timing_uses_the_paths_own_gguf_and_floor(tmp_path):
    c = cand(tmp_path, path=A3B, floors=(A3B,))
    r = run(c, tmp_path, ("build", "ab-timing"))["ab-timing"][0]
    assert r["polar"] == "PASS" and r["speed"]["model"] == M + "Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
    assert r["timing_path"]["device"] == "cpu" and r["timing_path"]["host"] == "frontdoor"


def test_path_without_a_timing_cell_is_ungated_and_cannot_be_kept(tmp_path):
    c = cand(tmp_path, path=FN, floors=(K27,))                                  # Q38FN has no A/A floor cell
    out = run(c, tmp_path, ("build", "oracle-kld", "ab-timing", "fold-check"))
    r = out["ab-timing"][0]
    assert r["ungated"] == FN and "ungated" in r["verdict"] and "polar" not in r
    assert "MISSING_INPUT" in out["fold-check"][0]["verdict"]
    rep = sb.coverage_report(REG, POLICY, c)
    assert rep[FN]["timing"] == rep[A3B]["timing"] == rep[A3B_GPU]["timing"] == "ungated" and rep[K27]["timing"] == "cell"


def test_cpu_timing_never_runs_in_a_gpu_grant(tmp_path):
    c = cand(tmp_path, path=FN, floors=(FN,))
    run(c, tmp_path, ("build",))
    for st in ("ab-timing", "op-perf"):
        env = {}
        r = sb.run_stage(st, c, lane="gpu", test_mode=True, env=env, prim=sb.test_primitives(c))
        assert "WRONG_GRANT" in r["verdict"] and "topology" in r["verdict"] and env["HIP_VISIBLE_DEVICES"] == "-1"
    env = {}
    ok = sb.run_stage("ab-timing", c, lane="cpu", test_mode=True, env=env, prim=sb.test_primitives(c))
    assert ok["polar"] == "PASS" and env["HIP_VISIBLE_DEVICES"] == "-1"          # the cpu lane never sees the MI210
    env = {}                                                                     # and the MI210 path still needs the gpu lane
    r = sb.run_stage("ab-timing", cand(tmp_path), lane="cpu", test_mode=True, env=env)
    assert "WRONG_GRANT" in r["verdict"]
    env = {}
    sb.guard_device("ab-timing", "gpu", env, "gpu")
    assert "HIP_VISIBLE_DEVICES" not in env


def test_unserved_path_is_refused(tmp_path):
    r = sb.run_stage("ab-timing", cand(tmp_path, path="ds41-mxfp4|cpu"), lane="cpu", test_mode=True, env={})
    assert "MISSING_INPUT" in r["verdict"] and "polar" not in r


def test_live_cpu_timing_runner_refuses_without_the_served_topology():
    with pytest.raises(sb.StageRefused):
        sb._live_timing({"path_spec": sb.path_specs(REG)[FN]}, Path("/a"), Path("/b"))


# ---- R8 ------------------------------------------------------------------------------------------------------

def test_every_served_path_has_a_policy_entry():
    missing = [k for k in sb.path_specs(REG) if k not in POLICY["paths"]]
    assert missing == [], f"served paths with no graph_kld_policy entry: {missing}"


def test_new_served_path_without_a_policy_entry_is_quality_ungated(tmp_path):
    reg = json.loads(json.dumps(REG))
    reg["server_mode"]["worker_new"] = _host("newmodel_q8_local", None, M + "New-Q8_0.gguf", 8099, 3, 1)
    missing = [k for k in sb.path_specs(reg) if k not in POLICY["paths"]]
    assert missing == ["newmodel_q8_local|cpu"]                                   # the coverage test above would fail here
    assert sb.quality_status(missing[0], POLICY) == {"path": missing[0], "status": "quality-ungated",
                                                     "reason": "no-policy-entry"}


def test_a3b_entries_have_null_floor_and_report_quality_ungated(tmp_path):
    for k in (A3B, A3B_GPU):
        e = POLICY["paths"][k]
        assert e["model"] == "a3b" and e["floor"] is None and e["status"] == "needs-floor-row"
        assert not any(c["model"] == "a3b" for b in POLICY["backends"].values() for c in b)   # no number was invented
        assert sb.quality_status(k, POLICY)["status"] == "quality-ungated"
    assert POLICY["paths"][A3B_GPU]["planned"] is True
    assert sb.quality_status(K27, POLICY)["status"] == sb.quality_status(FN, POLICY)["status"] == "gated"
    out = run(cand(tmp_path, path=A3B), tmp_path, ("build", "oracle-kld"))["oracle-kld"][0]
    assert "polar" not in out and out["verdict"].startswith("FUTILITY_QUALITY_UNGATED (a3b-q8_0-cpu|cpu needs-floor-row")
    assert out["quality"]["status"] == "quality-ungated"
    v = ge.evaluate(out, ended={"rc": "0", "reason": "complete"}, row="akc-ak-x-oracle-kld-20261124")
    assert v["verdict"] == "FUTILITY" and v["filed_verdict"] == "inconclusive"
    gated = run(cand(tmp_path), tmp_path, ("build", "oracle-kld"))["oracle-kld"][0]
    assert gated["quality"]["status"] == "gated" and "quality-ungated" not in gated["verdict"]


def test_policy_still_loads_for_the_gate_and_a3b_adds_no_clause():
    from autokernel.loop import gate_graph_kld_policy as gp
    for be in ("gpu", "cpu"):
        pol = gp.load_graph_kld_policy(be)
        assert pol is not None and {c.model for c in pol.clauses} == {"q38_27b", "q38fn"}


@pytest.mark.parametrize("path", [A3B, A3B_GPU])
def test_quality_ungated_candidate_cannot_fold(tmp_path, path):
    """oracle-kld would PASS on the 27B/Q38FN clauses, ab-timing has a cell and passes: fold-check still refuses."""
    c = cand(tmp_path, path=path, floors=(path,))
    out = run(c, tmp_path, ("build", "oracle-kld", "ab-timing", "fold-check"),
              lane_of=lambda st, spec: "cpu" if st == "build" else "gpu" if st != "ab-timing" else spec["lane"])
    assert out["ab-timing"][0]["polar"] == "PASS" and "polar" not in out["oracle-kld"][0]
    fc = out["fold-check"][0]
    assert "MISSING_INPUT" in fc["verdict"] and "oracle-kld" in fc["verdict"] and "polar" not in fc
