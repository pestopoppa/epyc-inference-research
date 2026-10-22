"""File-scoped tests for the GPU->CPU noise experiment driver + analysis (no inference)."""
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_gpu_cpu_noise as A  # noqa: E402
import gpu_cpu_noise_driver as D  # noqa: E402

A.N_BOOT = 2000

REAL_BENCH_JSON = "/mnt/raid0/llm/tmp/ds41-w5-20261021/runs-aba/bench-B-1017.json"


def test_schedule_balanced_and_deterministic():
    s1 = D.make_schedule("seed-x", 10, 5)
    s2 = D.make_schedule("seed-x", 10, 5)
    assert s1 == s2
    assert D.make_schedule("seed-y", 10, 5) != s1
    for b in range(10):
        conds = sorted(r["cond"] for r in s1 if r["block"] == b)
        exp = sorted(D.BASE_BLOCK + (["S8"] if b % 2 == 1 else []))
        assert conds == exp
    assert all(len(r["reps"]) == 5 for r in s1)
    assert len({rid for r in s1 for rid in r["reps"]}) == sum(len(r["reps"]) for r in s1)


def test_parse_bench_jsonl_against_real_llama_bench_output():
    if os.path.exists(REAL_BENCH_JSON):
        real = json.load(open(REAL_BENCH_JSON))
    else:  # same field shape as llama-bench -o json
        real = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 87.9, "samples_ts": [87.8, 88.0]},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": 16.7, "samples_ts": [16.6, 16.7]}]
    noise = ["[region-lock] held regions q0,q1,q2,q3", "| model | size |", "Placement proof: /x/p.tsv (exit 0)"]
    text = "\n".join(noise[:2] + [json.dumps(r) for r in real] + noise[2:])
    res = D.parse_bench_jsonl(text)
    assert set(res) >= {"pp", "tg"}
    assert res["pp"]["samples_ts"] == real[0]["samples_ts"]
    assert res["tg"]["n_gen"] == real[1]["n_gen"]


def test_gpu_server_argv_matches_receipt_shape():
    q3 = D.gpu_server_argv("q3", 18961)
    n1 = D.gpu_server_argv("n1", 18961)
    assert q3[:4] == ["taskset", "-c", "184-191", "numactl"] and "--membind=3" in q3
    assert n1[2] == "136-143" and "--membind=1" in n1
    for flag in ("--kv-unified", "draft-dflash", "-ngl", "--device-draft", "--no-mmap"):
        assert flag in q3
    assert q3[q3.index("-md") + 1].endswith("Qwen3.8-27B-DFlash2-Q8_0.gguf")


def test_self_hash_detects_mutation(tmp_path):
    row = D.self_hash({"a": 1, "b": [1, 2]})
    p = tmp_path / "reps.jsonl"
    bad = dict(row, a=2)
    p.write_text(json.dumps(row) + "\n" + json.dumps(bad) + "\n")
    rows = A.load(str(p))
    assert rows[0]["_hash_ok"] and not rows[1]["_hash_ok"]


def _synth(tmp_path, effects, blocks=10, run_len=5, sd=0.003, seed=1, drift=0.002):
    """effects: cond -> (bias fraction, sd multiplier)."""
    rng = random.Random(seed)
    sched = D.make_schedule("synthetic", blocks, run_len)
    rows, step = [], 0
    for r in sched:
        bias, mult = effects.get(r["cond"], (0.0, 1.0))
        for k, rid in enumerate(r["reps"]):
            step += 1
            base = 50.0 * (1 + drift * step / 100)
            tg = base * (1 + bias) * math.exp(rng.gauss(0, sd * mult))
            pp = 10 * base * (1 + bias / 2) * math.exp(rng.gauss(0, sd * mult))
            samp = lambda v: [v * math.exp(rng.gauss(0, sd * mult)) for _ in range(3)]
            row = {"schema": D.SCHEMA_REP, "rep_id": rid, "phase": "measure", "cond": r["cond"],
                   "block": r["block"], "run": r["run"], "idx": k, "valid": True, "why_invalid": [],
                   "start": f"2026-10-22T00:{step // 60:02d}:{step % 60:02d}Z",
                   "end": f"2026-10-22T00:{step // 60:02d}:{step % 60:02d}Z",
                   "tg": {"samples_ts": samp(tg)}, "pp": {"samples_ts": samp(pp)}}
            rows.append(D.self_hash(row))
    d = tmp_path / "sess"
    d.mkdir()
    (d / "reps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    return str(d)


def test_analysis_recovers_null(tmp_path):
    sess = _synth(tmp_path, {})
    r = A.analyze(sess)
    assert r["metrics"]["tg"]["aa"]["pass"] is True
    for c in A.CONDS:
        b = r["metrics"]["tg"]["cond"][c]["bias"]
        assert b["ci_pct"][0] <= 0.3 and b["ci_pct"][1] >= -0.3
        assert r["decision_table"][c]["IND"].startswith("SHAREABLE")
        assert r["decision_table"][c]["AK"] == "SHAREABLE", r["decision_table"][c]


def test_analysis_detects_bias_and_noise(tmp_path):
    sess = _synth(tmp_path, {"G2": (-0.05, 1.0), "G3": (0.0, 3.0), "G1": (0.0, 1.0)})
    r = A.analyze(sess)
    g2 = r["metrics"]["tg"]["cond"]["G2"]
    assert -6.0 < g2["bias"]["bias_pct"] < -4.0 and g2["bias"]["ci_pct"][1] < -1.0
    assert r["decision_table"]["G2"]["DG"] == "NOT-SHAREABLE"
    assert r["decision_table"]["G2"]["AK"] == "NOT-SHAREABLE"
    g3 = r["metrics"]["tg"]["cond"]["G3"]["noise"]["process"]
    assert g3["vr"] > 4 and g3["vr_ci"][0] > 2.0
    assert r["decision_table"]["G3"]["DG"] == "NOT-SHAREABLE"
    md = A.render_md(r)
    assert "Decision table" in md and "| G2 |" in md


def test_v2_gate_marks_untested(tmp_path):
    sess = _synth(tmp_path, {})
    p = os.path.join(sess, "reps.jsonl")
    rows = [json.loads(l) for l in open(p)]
    out = []
    for x in rows:
        if x["cond"] == "G4":
            x = dict(x, valid=False, why_invalid=["GPU load did not fire"])
            x.pop("content_sha256")
            x = D.self_hash(x)
        out.append(x)
    open(p, "w").write("".join(json.dumps(x) + "\n" for x in out))
    r = A.analyze(sess)
    assert r["decision_table"]["G4"]["AK"].startswith("UNTESTED")


def test_affinity_within_handles_omp_bound_threads():
    assert D.affinity_within(["184", "191", "184-191"], "184-191")
    assert not D.affinity_within(["0-95"], "184-191")
    assert not D.affinity_within([], "184-191")


# ---------------------------------------------------------------- W2-Q38FN (addendum PREREG.ADDENDUM-W2-Q38FN.md)
def test_w2_schedule_is_abba_and_g3_is_opt_in():
    s = D.make_schedule_w2(5, 4)
    assert len(s) == 20 and all(len(r["reps"]) == 4 for r in s)
    for b in range(5):
        assert [r["cond"] for r in s if r["block"] == b] == ["G0", "G2", "G2", "G0"]
    assert "G3" not in {r["cond"] for r in s}
    g3 = D.make_schedule_w2(2, 3, with_g3=True)
    assert [r["cond"] for r in g3 if r["block"] == 0] == ["G0", "G2", "G3", "G3", "G2", "G0"]
    assert len({rid for r in s for rid in r["reps"]}) == 80


def test_q38fn_argv_is_the_production_recipe():
    argv = D.q38fn_server_argv(18962, "/x/kv")
    assert argv[:5] == ["taskset", "-c", "0-95", "numactl", "--interleave=all"]
    assert argv[5] == D.Q38FN_BINDIR + "/llama-server" and D.Q38FN_BINDIR.endswith("cpu-20260921-ffc1bac82/bin")
    srv = argv[5:]                                   # server flags only (taskset also has a -c)
    flag = lambda f: srv[srv.index(f) + 1]  # noqa: E731
    assert flag("-m") == D.Q38FN_SHARDS[0] and flag("-md") == D.Q38FN_MTP
    assert (flag("-np"), flag("-c"), flag("-t"), flag("-ub")) == ("1", "262144", "96", "2048")
    assert (flag("-ctk"), flag("-ctv"), flag("--flash-attn"), flag("--cache-ram")) == ("f16", "f16", "on", "32768")
    assert (flag("--spec-type"), flag("--spec-draft-n-max"), flag("--draft-p-min")) == ("draft-mtp", "4", "0.5")
    assert (flag("--reasoning"), flag("--device"), flag("--device-draft"), flag("--port")) == ("off", "none", "none", "18962")
    for f in ("--jinja", "--mlock", "--no-mmap"):
        assert f in argv
    assert "--fa" not in argv and "-ngl" not in argv


def test_q38fn_launch_holds_its_own_claim_without_gpu_quiet():
    env = D.q38fn_env({"PATH": "/usr/bin", "GGML_NOHUGEPAGE_PROCESS": "1", "OMP_WAIT_POLICY": "passive"})
    assert env["LD_LIBRARY_PATH"].split(":")[0] == D.Q38FN_BINDIR
    assert (env["GGML_IQK"], env["OMP_WAIT_POLICY"], env["KMP_BLOCKTIME"]) == ("1", "active", "10")
    assert "GGML_NOHUGEPAGE_PROCESS" not in env   # production does not run the recipe env (SSU-F11)
    la = D.q38fn_launch_argv(18962, "/x/kv", "tag", env)
    assert la[:4] == [D.REGION_LOCK, "run", "--cpu-list", "0-95"]
    assert "--gpu-quiet" not in la              # hold-once: the session already holds gpu-quiet exclusive
    cmd = la[-1]
    assert la[la.index("--") + 1:la.index("--") + 3] == ["bash", "-c"]
    assert "numa_evict.py --target-gib 40" in cmd and "exec env " in cmd and "region-lock" not in cmd
    assert cmd.index("numa_evict.py") < cmd.index("llama-server")


def test_w2_request_bodies_are_seeded_and_cache_proof():
    a, b = D.w2_prefill_body("s", 1), D.w2_prefill_body("s", 2)
    assert a["prompt"] != b["prompt"] and a["prompt"].startswith("[1] ") and a["cache_prompt"] is False
    assert D.w2_prefill_body("s", 1) == a and len(a["prompt"].split()) >= D.W2_PREFILL_WORDS
    d = D.w2_decode_body("s")
    assert d == D.w2_decode_body("s") and d["n_predict"] == 256 and d["ignore_eos"] and d["cache_prompt"] is False
    assert D.w2_decode_body("t") != d


def test_w2_controls_flag_cache_reuse_short_decode_and_lost_server():
    good = {"cpu_server": {"ok_after_rep": True}, "prompt_n_ref": 4000,
            "pp": {"prompt_n": [4010]}, "tg": {"predicted_n": [256, 256]}}
    assert D.w2_controls(good) == []
    assert any("cache" in w for w in D.w2_controls(dict(good, pp={"prompt_n": [120]})))
    assert any("predicted_n" in w for w in D.w2_controls(dict(good, tg={"predicted_n": [256, 31]})))
    assert any("server" in w for w in D.w2_controls(dict(good, cpu_server={"ok_after_rep": False})))


# seed 5: the null A/A passes with and without scipy (a "CI contains 0" gate on two metrics fails ~10% of
# null seeds by construction; these tests check the decision logic, not interval coverage).
def _synth_w2(tmp_path, tg_bias, pp_bias=-0.02, sd=0.004, blocks=5, run_len=4, seed=5, aa_shift=0.0):
    rng = random.Random(seed)
    rows = []
    for r in D.make_schedule_w2(blocks, run_len):
        g2 = r["cond"] == "G2"
        for k, rid in enumerate(r["reps"]):
            shift = aa_shift if (r["cond"] == "G0" and r["run"] == 3) else 0.0
            tg = 40.0 * (1 + (tg_bias if g2 else 0) + shift)
            pp = 230.0 * (1 + (pp_bias if g2 else 0) + shift)
            row = {"schema": D.SCHEMA_REP, "workload": "w2-q38fn", "rep_id": rid, "phase": "measure",
                   "cond": r["cond"], "block": r["block"], "run": r["run"], "idx": k, "valid": True,
                   "why_invalid": [], "start": "2026-10-22T00:00:00Z", "end": "2026-10-22T00:00:01Z",
                   "tg": {"samples_ts": [tg * math.exp(rng.gauss(0, sd)) for _ in range(2)],
                          "draft_n_accepted": [82, 82]},
                   "pp": {"samples_ts": [pp * math.exp(rng.gauss(0, sd))]}}
            rows.append(D.self_hash(row))
    d = tmp_path / f"w2-{tg_bias}-{aa_shift}"
    d.mkdir()
    (d / "reps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    return str(d)


def test_w2_analysis_corun_ok_and_transfer_consistent(tmp_path):
    r = A.analyze(_synth_w2(tmp_path, -0.01))
    assert r["workload"] == "w2-q38fn" and r["conds"] == ["G2"] and list(r["decision_table"]) == ["G2"]
    w = r["w2"]["cond"]["G2"]
    assert r["w2"]["gates"]["V1_aa"] is True
    assert w["corun"].startswith("CO-RUN-OK"), w
    assert w["w1_transfer"] == "CONSISTENT", w
    assert -1.6 < w["correction_note"]["decode_bias_pct"] < -0.4
    md = A.render_md(r)
    assert "W2-Q38FN predeclared decision" in md and "| G2 |" in md and "| G1 |" not in md


def test_w2_analysis_big_slowdown_is_not_ok_and_contradicts_w1(tmp_path):
    w = A.analyze(_synth_w2(tmp_path, -0.08))["w2"]["cond"]["G2"]
    assert w["corun"] == "CO-RUN-NOT-OK" and w["w1_transfer"] == "CONTRADICTED", w


def test_w2_aa_drift_withholds(tmp_path):
    w = A.analyze(_synth_w2(tmp_path, -0.01, aa_shift=0.03))["w2"]
    assert w["gates"]["V1_aa"] is False and w["cond"]["G2"]["corun"].startswith("WITHHELD")


def test_w2_stub_session_end_to_end(tmp_path):
    """Stub mode only: stub GPU + stub Q38FN HTTP servers spawned and reaped by the driver itself; no inference,
    no region-lock claim, no model load."""
    import subprocess
    run = tmp_path / "run"
    run.mkdir()
    drv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_cpu_noise_driver.py")
    cp = subprocess.run([sys.executable, drv, "stub", "--run-dir", str(run), "--session", "S", "--workload", "w2-q38fn",
                         "--blocks", "2", "--blocks-min", "1", "--run-len", "2", "--warm-cpu", "1", "--warm-gpu-s", "0",
                         "--leadin-min-s", "0", "--port", "18975", "--cpu-port", "18976", "--max-hours", "0.2"],
                        capture_output=True, text=True, timeout=300)
    sess = run / "runs" / "S"
    end = json.loads((sess / "END.json").read_text())
    assert cp.returncode == 0 and end["rc"] == 0 and end["workload"] == "w2-q38fn", cp.stdout[-2000:]
    rows = [json.loads(x) for x in (sess / "reps.jsonl").read_text().splitlines()]
    meas = [x for x in rows if x["phase"] == "measure"]
    assert len(meas) == 2 * 4 * 2 and all(x["workload"] == "w2-q38fn" and x["valid"] for x in meas)
    assert all(len(x["tg"]["samples_ts"]) == 2 and len(x["pp"]["samples_ts"]) == 1 for x in meas)
    assert rows[0]["prompt_n_ref"] is None and meas[0]["prompt_n_ref"] > 3000
    steps = [json.loads(x)["step"] for x in (sess / "timeline.jsonl").read_text().splitlines()]
    assert steps.index("cpu_server_up") < steps.index("rep") and "cpu_server_down" in steps
    rec = json.loads((sess / "RUN_RECORD.json").read_text())
    assert rec["workload"] == "w2-q38fn" and rec["w2"]["decodes_per_rep"] == 2
    for line in (sess / "children.pids").read_text().splitlines():  # nothing we spawned survives
        pid, st = line.split()[:2]
        assert not D.alive(int(pid), int(st)), line
    r = A.analyze(str(sess))
    assert r["w2"]["cond"]["G2"]["n_blocks"] == 2
