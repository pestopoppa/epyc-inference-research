"""AKPL-P3 stage bodies: polar RESULTs gate_eval files with zero LLM turns, and the hard rules refuse, never act."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autokernel.loop import gate_eval as ge
from autokernel.loop import stage_bodies as sb

BASE, PATCH, OTHER = "a" * 40, "b" * 40, "c" * 40
REF = "ak/champion/llama-cpp-ffc1bac82eec"


def cand(tmp_path, **kw):
    c = {"id": "cand-x-bbbbbbbbb", "base_sha": BASE, "patch_sha": PATCH, "hypothesis_id": "AK-X",
         "predicted_effect": "+2%", "champion_ref": REF, "build_root": str(tmp_path),
         "test_refs": {REF: BASE}, "test_parents": {PATCH: BASE}, "stage_results": {},
         "shape_class": {"op_family": "MUL_MAT", "dims": {"k": [5120], "m": [1024, 6144]}}}
    c.update(kw)
    return c


def run_all(c, tmp_path, stages=sb.STAGES, lineage=None):
    prim = sb.test_primitives(c, lineage_log=lineage)
    out = {}
    for st in stages:
        lane = "cpu" if st == "build" else "gpu"
        r = sb.run_stage(st, c, lane=lane, prim=prim, test_mode=True, env={}, cpus={0, 1})
        p = tmp_path / f"{st}.json"
        p.write_text(json.dumps(r))
        c["stage_results"][st] = str(p)
        out[st] = r
    return out, prim


def filed(r, row):
    v = ge.evaluate(r, ended={"rc": "0", "reason": "complete"}, row=row)
    return v["verdict"], ge.policy_decided(v)


def test_happy_chain_files_zero_llm_and_lands_keep(tmp_path):
    c = cand(tmp_path)
    log = []
    out, prim = run_all(c, tmp_path, lineage=log)
    gated = [s for s in sb.STAGES if s != "op-perf"]   # op-perf is a screen: it never carries a polar PASS (RSL-R3)
    assert [out[s]["polar"] for s in gated] == ["PASS"] * 4 and "polar" not in out["op-perf"]
    for st, r in {s: out[s] for s in gated}.items():
        verdict, decided = filed(r, f"akc-ak-x-{st}-20261121")
        assert verdict == "PASS" and decided, (st, verdict)
    assert prim.paths["refs"][REF] == PATCH
    assert out["fold-check"]["verdict"] == "KEEP_LANDED"
    assert log and log[0]["kind"] == "keep" and log[0]["keep_sha"] == PATCH   # P4 lineage generate on KEEP


@pytest.mark.parametrize("stage,expect", [("build", "FAIL"), ("oracle-kld", "FAIL"), ("ab-timing", "FAIL")])
def test_candidate_failures_are_policy_decided_fail(tmp_path, stage, expect):
    c = cand(tmp_path, test_outcomes={stage: "fail"})
    stages = sb.STAGES[:sb.STAGES.index(stage) + 1]
    out, _ = run_all(c, tmp_path, stages)
    verdict, decided = filed(out[stage], f"akc-ak-x-{stage}-20261121")
    assert verdict == expect and decided


def test_kld_fail_is_l2_policy_gate(tmp_path):
    c = cand(tmp_path, test_outcomes={"oracle-kld": "fail"})
    out, _ = run_all(c, tmp_path, ("build", "oracle-kld"))
    v = ge.evaluate(out["oracle-kld"], ended={"rc": "0"}, row="akc-x-oracle-kld-20261121")
    assert v["gate_id"] == "graph_kld.gpu.q38_27b" and v["verdict"] == "FAIL"


def test_harness_build_is_futility_not_fail(tmp_path):
    c = cand(tmp_path, test_outcomes={"build": "harness"})
    out, _ = run_all(c, tmp_path, ("build",))
    assert out["build"]["verdict"].startswith("FUTILITY_HARNESS_BUILD_ENV")
    assert filed(out["build"], "akc-x-build-20261121")[0] == "FUTILITY"


def test_stale_base_never_lands(tmp_path):
    c = cand(tmp_path, test_refs={REF: OTHER})          # champion advanced to OTHER after the build on BASE
    out, prim = run_all(c, tmp_path)
    fc = out["fold-check"]
    assert fc["verdict"].startswith("FUTILITY_HARNESS_STALE_BASE") and "polar" not in fc
    assert fc["stale_base"]["action"] == "rebase-and-recheck"
    assert prim.paths["refs"][REF] == OTHER


def test_patch_not_built_on_base_never_lands(tmp_path):
    c = cand(tmp_path, test_parents={PATCH: OTHER})
    out, prim = run_all(c, tmp_path)
    assert "SOURCE_MISMATCH" in out["fold-check"]["verdict"]
    assert prim.paths["refs"][REF] == BASE


def test_fold_check_refuses_without_upstream_pass(tmp_path):
    c = cand(tmp_path, test_outcomes={"ab-timing": "fail"})
    out, prim = run_all(c, tmp_path)
    assert "MISSING_INPUT" in out["fold-check"]["verdict"]
    assert prim.paths["refs"][REF] == BASE


def test_build_result_must_name_the_patch(tmp_path):
    c = cand(tmp_path)
    run_all(c, tmp_path, ("build",))
    b = json.loads(Path(c["stage_results"]["build"]).read_text())
    b["built_sha"] = OTHER
    Path(c["stage_results"]["build"]).write_text(json.dumps(b))
    r = sb.run_stage("oracle-kld", c, lane="gpu", test_mode=True, env={})
    assert "SOURCE_MISMATCH" in r["verdict"]


def test_production_refs_and_tree_refused():
    with pytest.raises(sb.StageRefused):
        sb.guard_ref("production-consolidated-v10")
    with pytest.raises(sb.StageRefused):
        sb.guard_ref("refs/heads/master")
    sb.guard_ref(REF)
    sb.guard_ref("llama.cpp-experimental-c18-composite-20261121")
    with pytest.raises(sb.StageRefused):
        sb.guard_tree(sb.CANONICAL_TREE)
    with pytest.raises(sb.StageRefused):
        sb.guard_tree(Path("/tmp"), git=lambda *a: "production-consolidated-v10")


def test_production_ref_in_candidate_refused_by_fold_check(tmp_path):
    c = cand(tmp_path, champion_ref="production-consolidated-v10")
    out, _ = run_all(c, tmp_path)
    assert "PRODUCTION_REF" in out["fold-check"]["verdict"]


def test_hip_hidden_outside_gpu_grant():
    env = {}
    sb.guard_device("build", "cpu", env)
    assert env["HIP_VISIBLE_DEVICES"] == "-1"
    env = {}
    sb.guard_device("ab-timing", "gpu", env)
    assert "HIP_VISIBLE_DEVICES" not in env
    r = sb.run_stage("ab-timing", {"id": "x"}, lane="cpu", test_mode=True, env=(e := {}))
    assert "WRONG_GRANT" in r["verdict"] and e["HIP_VISIBLE_DEVICES"] == "-1"


def test_build_cpus_only_the_grant():
    assert sb.guard_build_cpus(set(range(24, 48)), test_mode=False) == "24-47"
    with pytest.raises(sb.StageRefused):
        sb.guard_build_cpus(set(range(72, 96)), test_mode=False)          # q3 = MI210 host cores
    with pytest.raises(sb.StageRefused):
        sb.guard_build_cpus(set(range(0, 192)), test_mode=False)          # whole host: not under a grant


def test_fixture_repo_cas_is_real_compare_and_swap(tmp_path):
    repo = tmp_path / "r"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    g = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a],
                                  capture_output=True, text=True, check=True).stdout.strip()
    (repo / "f").write_text("1")
    g("add", "f"); g("commit", "-qm", "base")
    base = g("rev-parse", "HEAD")
    g("branch", REF)
    (repo / "f").write_text("2")
    g("commit", "-qam", "patch")
    patch = g("rev-parse", "HEAD")
    c = cand(tmp_path, base_sha=base, patch_sha=patch)
    c.pop("test_refs"); c.pop("test_parents")
    prim = sb.test_primitives(c, fixture_repo=repo)
    for st in sb.STAGES:
        r = sb.run_stage(st, c, lane="cpu" if st == "build" else "gpu", prim=prim, test_mode=True, env={}, cpus={0})
        p = tmp_path / f"{st}.json"
        p.write_text(json.dumps(r))
        c["stage_results"][st] = str(p)
    assert r["verdict"] == "KEEP_LANDED"
    assert g("rev-parse", REF) == patch


def test_cli_writes_result(tmp_path, monkeypatch):
    # main() mutates os.environ (guard_device); register the keys so monkeypatch restores them (CTC-2-TI)
    for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k)
    c = cand(tmp_path)
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(c))
    out = tmp_path / "out" / "RESULT.json"
    assert sb.main(["--stage", "build", "--candidate", str(cp), "--out", str(out), "--lane", "cpu",
                    "--row", "akc-x-build-20261121", "--test-mode"]) == 0
    r = json.loads(out.read_text())
    assert r["verdict"].startswith(("PASS", "KEEP", "FAIL", "FUTILITY"))   # check_complete.py contract
    assert r["polar"] == "PASS" and r["row"] == "akc-x-build-20261121" and r["complete"] is True


def test_trigger_lineage_runs_generate(tmp_path):
    seen = []
    r = sb.trigger_lineage({"kind": "composite", "llama_repo": "/x"}, out_root=tmp_path,
                           run=lambda argv: seen.append(argv) or 0)
    assert seen and seen[0][0] == "generate" and "--repo" in seen[0]
    assert r["lineage_rc"] == 0 and (Path(r["lineage_out"]) / "event.json").exists()


def test_cli_does_not_leak_device_env(tmp_path, monkeypatch):
    """CTC-2-TI regression: the CLI's os.environ mutation must be undone after the test."""
    import os
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    test_cli_writes_result(tmp_path, monkeypatch)
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "-1"   # set during the test ...
    monkeypatch.undo()                                      # ... and removed by teardown
    assert "HIP_VISIBLE_DEVICES" not in os.environ
