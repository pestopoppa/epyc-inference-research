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
