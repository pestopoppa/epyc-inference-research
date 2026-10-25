"""Tests for the R23-64 CPU decode thread sweep (plan, argv builder, analysis, client, provenance).

Run:  cd <repo> && /usr/bin/python3 -m pytest scripts/benchmark/cpu_thread_sweep -q
No test starts a llama-server, loads a model or touches the page cache.
"""
import hashlib
import http.server
import json
import math
import os
import random
import subprocess
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_sweep as A  # noqa: E402
import sweep_lib as S      # noqa: E402

R = S.R
BIN = "/fake/build/bin"


# ---------------------------------------------------------------------------
# plan / rotated order
# ---------------------------------------------------------------------------
class TestPlan:
    def test_three_arm_rotation_is_the_specified_one(self):
        names = lambda r: [a.name for a in S.rotated_order(r)]  # noqa: E731
        assert names(1) == ["t48", "t64", "t96"]
        assert names(2) == ["t64", "t96", "t48"]
        assert names(3) == ["t96", "t48", "t64"]
        assert [names(r) for r in (4, 5, 6)] == [names(1), names(2), names(3)]

    def test_each_arm_once_per_round_and_six_launches_per_arm(self):
        p = S.plan(6)
        assert len(p) == 18
        for r in range(1, 7):
            assert sorted(l.arm.name for l in p if l.round_no == r) == ["t48", "t64", "t96"]
        for arm in ("t48", "t64", "t96"):
            assert sum(1 for l in p if l.arm.name == arm) == 6

    def test_no_arm_owns_a_slot(self):
        slots = {a: set() for a in ("t48", "t64", "t96")}
        for r in range(1, 4):
            for pos, arm in enumerate(S.rotated_order(r)):
                slots[arm.name].add(pos)
        assert all(v == {0, 1, 2} for v in slots.values())

    def test_masked_arm_is_opt_in(self):
        assert "t48m" not in {l.arm.name for l in S.plan(6)}
        p = S.plan(6, include_masked=True)
        assert len(p) == 24
        m = [l for l in p if l.arm.name == "t48m"]
        assert len(m) == 6 and all(l.arm.cpus == "0-47" and l.arm.threads == 48 for l in m)
        for r in range(1, 7):
            assert len(S.rotated_order(r, True)) == 4

    def test_more_rounds_continue_the_rotation(self):
        assert [l.label for l in S.plan(7)][:18] == [l.label for l in S.plan(6)]
        assert [a.name for a in S.rotated_order(7)] == ["t48", "t64", "t96"]

    def test_labels_round_trip(self):
        for l in S.plan(6, True):
            assert S.parse_label(l.label) == (l.arm.name, l.round_no)
        with pytest.raises(ValueError):
            S.parse_label("t48-r1")
        with pytest.raises(ValueError):
            S.rotated_order(0)


# ---------------------------------------------------------------------------
# argv / env builder
# ---------------------------------------------------------------------------
class TestArgv:
    @pytest.mark.parametrize("arm", S.arm_set(True), ids=lambda a: a.name)
    def test_dash_t_and_mask_per_arm(self, arm):
        cmd = S.serve_argv(BIN, arm.threads, arm.cpus)
        assert cmd[:5] == ["taskset", "-c", arm.cpus, "numactl", "--interleave=all"]
        assert cmd.count("-t") == 1 and cmd[cmd.index("-t") + 1] == str(arm.threads)
        assert cmd[5] == f"{BIN}/llama-server"

    def test_only_t_and_mask_differ_between_arms(self):
        base = S.serve_argv(BIN, 48, "0-95")
        for t in (64, 96):
            other = S.serve_argv(BIN, t, "0-95")
            diff = [(a, b) for a, b in zip(base, other) if a != b]
            assert diff == [("48", str(t))]
        masked = S.serve_argv(BIN, 48, "0-47")
        assert [(a, b) for a, b in zip(base, masked) if a != b] == [("0-95", "0-47")]

    def test_context_edit_does_not_clobber_the_taskset_mask(self):
        # regression: a bare cmd.index("-c") finds taskset's -c first and overwrites the cpu list
        cmd = S.serve_argv(BIN, 48, "0-95", ctx=4096)
        assert cmd[:3] == ["taskset", "-c", "0-95"]
        assert cmd[cmd.index("-c", 5) + 1] == "4096"

    def test_recipe_flags_present(self):
        cmd = S.serve_argv(BIN, 64)
        j = " ".join(cmd)
        for needle in ("--no-webui", "-np 1", "-c 8192", "--no-mmap", "-lv 4", "-fa on", "-ctk f16", "-ctv f16",
                       "--spec-type draft-mtp", "--spec-draft-n-max 4", "--spec-draft-p-min 0.5"):
            assert needle in j, needle
        assert "--fa " not in j and "--fa" not in cmd          # the flag form that does not exist
        assert cmd[cmd.index("-m") + 1] == R.TRUNK_GGUF
        assert cmd[cmd.index("-md") + 1] == R.MTP_HEAD_GGUF
        assert "-ub" not in cmd                                  # unset by default (recipe), opt-in only
        assert S.serve_argv(BIN, 64, ubatch=2048)[-2:] == ["-ub", "2048"]

    def test_trunk_is_the_production_ud_iq4_xs_not_the_retired_uniform_file(self):
        cmd = S.serve_argv(BIN, 48)
        assert "UD-IQ4_XS" in cmd[cmd.index("-m") + 1]
        assert "uniform" not in " ".join(cmd)

    def test_env_is_the_recipe_and_inherits_nothing(self):
        env = S.serve_env(BIN)
        assert env["LD_LIBRARY_PATH"] == BIN                     # the build's own dir, nothing inherited
        assert env["GGML_NOHUGEPAGE_PROCESS"] == "1"             # CHAMP-2 shim
        assert env["GGML_FUSED_DECODE_OFF"] == "1" and env["GGML_FA_SPLIT_KV"] == "0" and env["GGML_IQK"] == "1"
        assert (env["OMP_PROC_BIND"], env["OMP_PLACES"], env["OMP_WAIT_POLICY"], env["OMP_DYNAMIC"]) == \
            ("spread", "cores", "active", "false")
        assert "GGML_KNOB_FILE" not in env and "GGML_NOHUGEPAGE" not in env   # the madvise knob stays unset
        assert set(env) == {"LD_LIBRARY_PATH", "PATH", "HOME", "GGML_NOHUGEPAGE_PROCESS", "GGML_FUSED_DECODE_OFF",
                            "GGML_FA_SPLIT_KV", "GGML_IQK", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "OMP_DYNAMIC"}

    def test_refuses_a_quantised_kv_or_missing_mtp(self, monkeypatch):
        monkeypatch.setattr(R, "KV_CACHE_FLAGS", ["-ctk", "q8_0", "-ctv", "q8_0"])
        with pytest.raises(R.RecipeViolation):
            S.serve_argv(BIN, 48)

    def test_model_files_cover_every_shard_and_the_head(self):
        f = S.model_files()
        assert len(f) == 4 and f[0] == R.TRUNK_GGUF and f[-1] == R.MTP_HEAD_GGUF
        assert [p for p, _b, _s in R.TRUNK_SHARDS] == f[:3]

    def test_ram_figure_is_computed_from_the_files(self):
        files_gib = (sum(b for _p, b, _s in R.TRUNK_SHARDS) + R.MTP_HEAD_BYTES) / 2**30
        assert 89.0 < files_gib < 90.5
        assert S.ram_gib() == math.ceil(files_gib + 0.26 + S.RAM_OVERHEAD_GIB) == 92

    def test_cli_serve_files_and_plan(self, tmp_path):
        out = tmp_path / "t96_r1"
        r = subprocess.run([sys.executable, str(HERE / "sweep_lib.py"), "serve-files", "--bindir", BIN, "--threads", "96",
                            "--cpus", "0-95", "--out-prefix", str(out)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        argv = Path(str(out) + ".argv").read_text().splitlines()
        assert argv[argv.index("-t") + 1] == "96"
        assert "LD_LIBRARY_PATH=" + BIN in Path(str(out) + ".env").read_text().splitlines()
        r = subprocess.run([sys.executable, str(HERE / "sweep_lib.py"), "plan", "--rounds", "3"], capture_output=True, text=True)
        assert [ln.split("\t")[2] for ln in r.stdout.splitlines()[:3]] == ["t48", "t64", "t96"]


class TestKnobProof:
    def _fake_build(self, tmp_path, knobs):
        for name, words in knobs.items():
            (tmp_path / name).write_bytes(b"\0".join(w.encode() for w in words) + b"\0")
        return str(tmp_path)

    def test_passes_when_every_required_knob_is_compiled_in_and_no_knob_page(self, tmp_path):
        b = self._fake_build(tmp_path, {"libggml-cpu.so.0": ["GGML_IQK", "GGML_FA_SPLIT_KV"],
                                        "libllama.so.0": ["GGML_FUSED_DECODE_OFF"],
                                        "libllama-common.so.0": ["GGML_NOHUGEPAGE_PROCESS"]})
        assert S.knob_proof(b)["ok"]

    def test_a_knob_absent_from_the_binary_is_a_refusal_not_a_null(self, tmp_path):
        b = self._fake_build(tmp_path, {"libggml-cpu.so.0": ["GGML_IQK", "GGML_FA_SPLIT_KV"],
                                        "libllama.so.0": ["GGML_FUSED_DECODE_OFF"]})
        with pytest.raises(R.RecipeViolation, match="GGML_NOHUGEPAGE_PROCESS"):
            S.knob_proof(b)

    def test_knob_page_build_refuses_no_knobs_mode(self, tmp_path):
        b = self._fake_build(tmp_path, {"libggml-cpu.so.0": ["GGML_IQK", "GGML_FA_SPLIT_KV", "GGML_KNOB_FILE"],
                                        "libllama.so.0": ["GGML_FUSED_DECODE_OFF"],
                                        "libllama-common.so.0": ["GGML_NOHUGEPAGE_PROCESS"]})
        with pytest.raises(R.RecipeViolation, match="GGML_KNOB_FILE"):
            S.knob_proof(b)
        assert S.knob_proof(b, no_knobs=False)["ok"]


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _rows(rate, n_prompts=24, ntok=200, short=0):
    """n_prompts rows at exactly `rate` t/s; `short` of them have pred_n < 16 (below the floor, other rate)."""
    rows = []
    for k in range(n_prompts):
        if k < short:
            rows.append(dict(seq=k, id=f"p{k}", pred_n=5, pred_ms=5 / 1.0 * 1000, verdict="COHERENT"))   # 1 t/s decoy
        else:
            rows.append(dict(seq=k, id=f"p{k}", pred_n=ntok, pred_ms=ntok / rate * 1000.0, verdict="COHERENT"))
    return rows


def _write_launch(d, lab, rate, *, screens="clean", n_prompts=24, own=None, **kw):
    (d / f"{lab}.rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows(rate, n_prompts, **kw)))
    own = own or int(lab[1:].split("_")[0].rstrip("m"))
    if screens == "none":
        return
    foreign = {"foreign_pct": 40.0, "top": [{"pid": 1, "comm": "x", "pct": 40.0, "allowed": "0-191"}]}
    busy = own + 1.0
    if screens == "dropped_cpu":
        foreign = {"foreign_pct": 900.0, "top": [{"pid": 1, "comm": "burst", "pct": 900.0, "allowed": "0-191"}]}
    if screens == "dropped_host":
        busy = own + 20.0
    (d / f"{lab}.foreign.jsonl").write_text(json.dumps({"meta": "bench_logical_cpus"}) + "\n" +
                                            "".join(json.dumps(dict(foreign, t=i)) + "\n" for i in range(5)))
    (d / f"{lab}.host.jsonl").write_text("".join(json.dumps(dict(busy_cores=busy, pgpgin_kb=10, pgpgout_kb=10,
                                                                   pgmajfault=0, loadavg="3.0")) + "\n" for _ in range(5)))


def _fill(d, rates, **kw):
    """rates: {arm: [per-launch rate...]} -> launches t<arm>_r1.."""
    for arm, vals in rates.items():
        for i, v in enumerate(vals, 1):
            _write_launch(d, S.label(arm, i), v, **kw)


class TestAnalysis:
    def test_launch_rate_is_token_weighted_with_the_16_token_floor(self):
        rows = _rows(40.0, n_prompts=4)
        rows[0] = dict(pred_n=100, pred_ms=1000.0)           # 100 t/s
        rows[1] = dict(pred_n=300, pred_ms=3000.0)           # 100 t/s
        rows[2] = dict(pred_n=200, pred_ms=5000.0)           # 40 t/s
        rows[3] = dict(pred_n=15, pred_ms=10.0)              # below the floor: must not count (1500 t/s decoy)
        tps, n_dec, toks = A.launch_rate(rows)
        assert (n_dec, toks) == (3, 600)
        assert tps == pytest.approx(600 / 9.0)               # NOT the mean of per-prompt rates (80)

    def test_floor_boundary_is_inclusive_at_16(self):
        assert A.launch_rate([dict(pred_n=16, pred_ms=1000.0)])[1] == 1
        assert A.launch_rate([dict(pred_n=15, pred_ms=1000.0)]) == (None, 0, 0)

    def test_per_arm_stats_match_hand_computation(self, tmp_path):
        vals = {"t48": [30.0, 30.3, 29.7, 30.15, 29.85, 30.0], "t64": [31.0] * 6, "t96": [28.0] * 6}
        _fill(tmp_path, vals)
        res = A.analyze(str(tmp_path), reps=200)
        s = res["arms"]["t48"]
        assert s["n"] == 6
        mean = sum(vals["t48"]) / 6
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals["t48"]) / 5)
        assert s["mean"] == pytest.approx(mean) and s["sd"] == pytest.approx(sd)
        half = 2.570582 * sd / math.sqrt(6)                   # Student t, df 5 -- not the z value 1.96
        assert s["ci95"][0] == pytest.approx(mean - half, rel=1e-5) and s["ci95"][1] == pytest.approx(mean + half, rel=1e-5)
        assert res["arms"]["t64"]["sd"] == pytest.approx(0.0, abs=1e-9)
        assert res["unit"] == "launch"

    def test_t_quantiles(self):
        assert A.t_ppf(0.975, 5) == pytest.approx(2.570582, abs=1e-5)
        assert A.t_ppf(0.975, 10) == pytest.approx(2.228139, abs=1e-5)
        assert A.t_ppf(0.80, 10) == pytest.approx(0.879058, abs=1e-5)
        assert A.t_ppf(0.025, 5) == pytest.approx(-2.570582, abs=1e-5)

    def test_ratio_and_bootstrap_ci_recover_a_planted_effect(self, tmp_path):
        rng = random.Random(7)
        _fill(tmp_path, {"t48": [30.0 * (1 + rng.gauss(0, 0.005)) for _ in range(6)],
                         "t64": [33.0 * (1 + rng.gauss(0, 0.005)) for _ in range(6)],      # +10%
                         "t96": [30.0 * (1 + rng.gauss(0, 0.005)) for _ in range(6)]})     # null
        res = A.analyze(str(tmp_path), reps=4000)
        p64, p96 = res["vs_control"]["t64"], res["vs_control"]["t96"]
        assert p64["effect_pct"] == pytest.approx(10.0, abs=1.0)
        assert p64["boot_ci95_pct"][0] < p64["effect_pct"] < p64["boot_ci95_pct"][1]
        assert p64["boot_ci95_pct"][0] > 7.0 and p64["ci_excludes_zero"] and p64["exceeds_mde_observed"]
        assert p96["boot_ci95_pct"][0] < 0 < p96["boot_ci95_pct"][1] and not p96["ci_excludes_zero"]

    def test_bootstrap_is_deterministic_for_a_seed(self):
        a, c = [30.0, 30.2, 29.9, 30.1], [29.0, 29.2, 28.9, 29.1]
        assert A.bootstrap_effect_pct(a, c, 500, 1) == A.bootstrap_effect_pct(a, c, 500, 1)
        assert A.bootstrap_effect_pct(a, c, 500, 1) != A.bootstrap_effect_pct(a, c, 500, 2)

    def test_mde_formula(self):
        # sd 0.609% of 30 t/s, 6 v 6, alpha .05 two-sided, power .8, df 10: (2.228139 + 0.879058) * sd * sqrt(1/3)
        sd = 0.00609 * 30.0
        expected = 100 * (2.228139 + 0.879058) * sd * math.sqrt(1 / 3) / 30.0
        assert A.mde_pct(sd, 6, 6, 30.0) == pytest.approx(expected, rel=1e-4)
        assert expected == pytest.approx(1.0925, abs=1e-3)
        assert A.mde_pct(sd, 1, 1, 30.0) is None            # df < 1: no MDE, not a made-up one

    def test_mde_reported_both_observed_and_floor(self, tmp_path):
        _fill(tmp_path, {"t48": [30.0, 30.2, 29.8, 30.1, 29.9, 30.0], "t64": [30.5, 30.6, 30.4, 30.55, 30.45, 30.5]})
        res = A.analyze(str(tmp_path), floor=A.Floor(0.609, "launch", 6), reps=200)
        p = res["vs_control"]["t64"]
        assert p["mde_observed_pct"] is not None and p["mde_floor_pct"] == pytest.approx(1.0925, abs=0.02)
        assert res["floor"] == dict(sd_pct=0.609, unit="launch", n=6)

    # ---- FLOOR-UNIT-1 ----
    @pytest.mark.parametrize("unit", ["arm", "ARM", "within-session", "", None])
    def test_refuses_a_floor_of_the_wrong_or_missing_unit(self, tmp_path, unit):
        _fill(tmp_path, {"t48": [30.0, 30.1, 29.9], "t64": [31.0, 31.1, 30.9]})
        with pytest.raises(A.UnitRefusal):
            A.analyze(str(tmp_path), floor=A.Floor(0.501, unit, 24))

    @pytest.mark.parametrize("unit", ["launch", "process", "session", "LAUNCH"])
    def test_accepts_a_launch_unit_floor(self, tmp_path, unit):
        _fill(tmp_path, {"t48": [30.0, 30.1, 29.9], "t64": [31.0, 31.1, 30.9]})
        assert A.analyze(str(tmp_path), floor=A.Floor(0.609, unit, 6), reps=100)["floor"]["unit"] == unit

    def test_cli_refuses_with_exit_2_on_the_arm_floor_and_prints_nothing_else(self, tmp_path):
        _fill(tmp_path, {"t48": [30.0, 30.1, 29.9], "t64": [31.0, 31.1, 30.9]})
        base = [sys.executable, str(HERE / "analyze_sweep.py"), "--runs", str(tmp_path), "--bootstrap", "100"]
        r = subprocess.run(base + ["--floor-sd-pct", "0.501", "--floor-unit", "arm"], capture_output=True, text=True)
        assert r.returncode == 2 and "FLOOR-UNIT-1" in r.stderr and r.stdout == ""
        r = subprocess.run(base + ["--floor-sd-pct", "0.501"], capture_output=True, text=True)       # unit missing
        assert r.returncode == 2 and r.stdout == ""
        r = subprocess.run(base + ["--floor-sd-pct", "0.609", "--floor-unit", "launch", "--floor-n", "6"], capture_output=True, text=True)
        assert r.returncode == 0 and "unit=LAUNCH" in r.stdout and "t64" in r.stdout

    # ---- screens / completeness / controls ----
    def test_dropped_launches_never_enter_a_statistic(self, tmp_path):
        _fill(tmp_path, {"t48": [30.0, 30.1, 29.9, 30.0]})
        _write_launch(tmp_path, "t48_r5", 5.0, screens="dropped_cpu")            # would wreck the mean if it counted
        _write_launch(tmp_path, "t48_r6", 99.0, screens="dropped_host")
        _fill(tmp_path, {"t64": [31.0, 31.1, 30.9]})
        res = A.analyze(str(tmp_path), reps=100)
        assert res["arms"]["t48"]["n"] == 4 and res["arms"]["t48"]["max"] < 31
        assert {(e["label"], e["status"]) for e in res["excluded"]} == {("t48_r5", "DROPPED"), ("t48_r6", "DROPPED")}

    def test_own_threads_are_charged_to_the_arm_not_to_foreign_load(self, tmp_path):
        # a t96 launch whose host shows 97 busy cores is CLEAN; the same 97 on a t48 launch is a 49-core intruder
        _write_launch(tmp_path, "t96_r1", 30.0, own=96)
        _write_launch(tmp_path, "t48_r1", 30.0, own=96)                           # busy 97 against -t 48
        recs = {r.label: r.status for r in A.load_launches(str(tmp_path))}
        assert recs == {"t96_r1": "valid", "t48_r1": "DROPPED"}

    def test_no_sampler_records_excludes_unless_asked(self, tmp_path):
        _fill(tmp_path, {"t48": [30.0, 30.1, 29.9], "t64": [31.0, 31.1, 30.9]}, screens="none")
        with pytest.raises(A.UnitRefusal):                                          # nothing valid -> control missing
            A.analyze(str(tmp_path))
        res = A.analyze(str(tmp_path), allow_unscreened=True, reps=100)
        assert res["arms"]["t48"]["n"] == 3

    def test_incomplete_launch_excluded(self, tmp_path):
        _fill(tmp_path, {"t48": [30.0, 30.1, 29.9], "t64": [31.0, 31.1, 30.9]})
        _write_launch(tmp_path, "t64_r4", 50.0, n_prompts=10)
        res = A.analyze(str(tmp_path), reps=100)
        assert res["arms"]["t64"]["n"] == 3
        assert ("t64_r4", "incomplete") in {(e["label"], e["status"]) for e in res["excluded"]}

    def test_http_error_row_makes_a_launch_incomplete(self, tmp_path):
        _write_launch(tmp_path, "t48_r1", 30.0)
        rows = [json.loads(x) for x in (tmp_path / "t48_r1.rows.jsonl").read_text().splitlines()]
        rows[3] = dict(seq=3, id="p3", verdict="HTTP-ERROR")
        (tmp_path / "t48_r1.rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert A.load_launches(str(tmp_path))[0].status == "incomplete"

    def test_refuses_without_a_control_arm_with_two_launches(self, tmp_path):
        _fill(tmp_path, {"t48": [30.0], "t64": [31.0, 31.1]})
        with pytest.raises(A.UnitRefusal, match="control"):
            A.analyze(str(tmp_path))

    def test_masked_arm_is_a_comparison_like_any_other(self, tmp_path):
        _fill(tmp_path, {"t48": [30.0, 30.1, 29.9, 30.0], "t48m": [27.0, 27.1, 26.9, 27.0]})
        p = A.analyze(str(tmp_path), reps=500)["vs_control"]["t48m"]
        assert p["effect_pct"] == pytest.approx(-10.0, abs=0.5) and p["ci_excludes_zero"]

    def test_render_lists_every_arm_and_exclusion(self, tmp_path):
        _fill(tmp_path, {"t48": [30.0, 30.1, 29.9], "t64": [31.0, 31.1, 30.9]})
        _write_launch(tmp_path, "t64_r9", 9.0, screens="dropped_cpu")
        txt = A.render(A.analyze(str(tmp_path), reps=100))
        assert "t48" in txt and "t64" in txt and "EXCLUDED" in txt and "t64_r9" in txt and "MDE" in txt


# ---------------------------------------------------------------------------
# client (against an in-process stub, no model)
# ---------------------------------------------------------------------------
class _Stub(http.server.BaseHTTPRequestHandler):
    mode = "ok"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        if _Stub.mode == "fail" and "hi" != body["messages"][0]["content"]:
            self.send_response(500)
            self.end_headers()
            return
        text = "The quick brown fox jumps over the lazy dog. " * 6
        out = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 50, "completion_tokens": 100},
               "timings": {"predicted_n": 100, "predicted_ms": 2000.0, "predicted_per_second": 50.0,
                           "prompt_n": 50, "prompt_ms": 100.0, "prompt_per_second": 500.0, "draft_n": 80, "draft_n_accepted": 60}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()


class TestClient:
    def test_complete_launch_renames_partial_and_feeds_the_analysis(self, stub, tmp_path):
        _Stub.mode = "ok"
        r = subprocess.run([sys.executable, str(HERE / "client.py"), "t48_r1", str(stub), "--out", str(tmp_path), "--timeout", "20"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert (tmp_path / "t48_r1.rows.jsonl").exists() and not (tmp_path / "t48_r1.rows.jsonl.partial").exists()
        rows = [json.loads(x) for x in (tmp_path / "t48_r1.rows.jsonl").read_text().splitlines()]
        assert len(rows) == 24 and {r_["cls"] for r_ in rows} == {"coding", "reasoning", "general"}
        assert A.launch_rate(rows)[0] == pytest.approx(50.0)

    def test_failed_launch_never_becomes_a_rows_jsonl(self, stub, tmp_path):
        _Stub.mode = "fail"
        r = subprocess.run([sys.executable, str(HERE / "client.py"), "t48_r2", str(stub), "--out", str(tmp_path), "--timeout", "20"],
                           capture_output=True, text=True)
        assert r.returncode == 3
        assert not (tmp_path / "t48_r2.rows.jsonl").exists()           # so a resume re-runs it
        assert (tmp_path / "t48_r2.rows.failed.jsonl").exists()

    def test_request_body_is_the_inf70_production_request(self):
        import client as C
        b = C.mkbody("x")
        assert b == {"messages": [{"role": "user", "content": "x"}], "max_tokens": 200, "cache_prompt": False,
                     "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.0}
        items = C.load_items()
        assert len(items) == 24 and len({i["id"] for i in items}) == 24


# ---------------------------------------------------------------------------
# shell, provenance
# ---------------------------------------------------------------------------
class TestShellAndProvenance:
    @pytest.mark.parametrize("script", ["session.sh", "run_sweep.sh", "affinity_lib.sh"])
    def test_bash_syntax(self, script):
        assert subprocess.run(["bash", "-n", str(HERE / script)], capture_output=True).returncode == 0

    def test_session_captures_live_affinity_before_meta_and_teardown(self, tmp_path):
        source = (HERE / "session.sh").read_text()
        capture = 'AFF=$(aff_check "$SPID" "$CPUS")'
        meta = '$LIB meta --out'
        teardown = '\nstop_server\n'
        assert capture in source
        assert 'live_affinity="$AFF"' in source
        assert source.index(capture) < source.index(meta) < source.index(teardown)

        self._fake_proc(tmp_path, 314, "0-3", ["0", "1-2", "3"])
        command = (f'set -u; . "{HERE}/affinity_lib.sh"; SPID=314; CPUS=0-3; PROC_ROOT="{tmp_path}"; '
                   f'{capture}; printf "live_affinity=%s\\n" "$AFF"')
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "live_affinity=threads=3 union_cpus=4 process_mask=0-3 within 0-3"

    @staticmethod
    def _fake_proc(root, pid, proc_mask, thread_masks):
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "status").write_text(f"Name:\tx\nCpus_allowed_list:\t{proc_mask}\n")
        for i, m in enumerate(thread_masks):
            t = d / "task" / str(pid + i)
            t.mkdir(parents=True)
            (t / "status").write_text(f"Name:\tx\nCpus_allowed_list:\t{m}\n")

    def _aff(self, root, pid, want):
        r = subprocess.run(["bash", "-c", f'. "{HERE}/affinity_lib.sh"; PROC_ROOT="{root}" aff_check {pid} {want}'],
                           capture_output=True, text=True)
        return r.returncode, r.stdout

    def test_affinity_main_thread_on_place0_is_ok(self, tmp_path):
        # the 2026-10-24 false rc=68: spread/cores binds the master to cpu 0, workers cover the rest
        self._fake_proc(tmp_path, 100, "0-95", ["0", "1", "2", "95"])
        assert self._aff(tmp_path, 100, "0-95")[0] == 0

    def test_affinity_missing_taskset_fails(self, tmp_path):
        self._fake_proc(tmp_path, 101, "0-191", ["0-191", "0-191"])
        assert self._aff(tmp_path, 101, "0-95")[0] == 1

    def test_affinity_one_thread_escaping_fails(self, tmp_path):
        self._fake_proc(tmp_path, 102, "0-95", ["0", "1", "96"])
        assert self._aff(tmp_path, 102, "0-95")[0] == 1

    def test_affinity_no_threads_is_not_a_pass(self, tmp_path):
        self._fake_proc(tmp_path, 103, "0-95", [])
        assert self._aff(tmp_path, 103, "0-95")[0] == 2

    def test_affinity_subset_arm_mask(self, tmp_path):
        self._fake_proc(tmp_path, 104, "0-47", ["0", "8-9", "47"])
        assert self._aff(tmp_path, 104, "0-47")[0] == 0

    def test_driver_dry_run_prints_the_rotated_plan_and_launches_nothing(self, tmp_path):
        env = dict(os.environ, CHAMP=BIN, OUT=str(tmp_path))
        r = subprocess.run(["bash", str(HERE / "run_sweep.sh"), "--rounds", "3", "--dry-run"], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        order = [ln.split("would launch ")[1].split()[0] for ln in r.stdout.splitlines() if "would launch" in ln]
        assert order == ["t48_r1", "t64_r1", "t96_r1", "t64_r2", "t96_r2", "t48_r2", "t96_r3", "t48_r3", "t64_r3"]
        assert list(tmp_path.iterdir()) == []

    def test_driver_resume_skips_existing_rows_and_honours_the_window(self, tmp_path):
        (tmp_path / "t48_r1.rows.jsonl").write_text("{}\n")
        env = dict(os.environ, CHAMP=BIN, OUT=str(tmp_path), FILLER_DEADLINE=str(int(__import__("time").time()) + 100))
        r = subprocess.run(["bash", str(HERE / "run_sweep.sh"), "--rounds", "1"], env=env, capture_output=True, text=True)
        assert r.returncode == 0
        assert "t48_r1 done (resume skip)" in r.stdout
        assert "window exhausted" in r.stdout and "t64_r1" in r.stdout          # 100 s left < 300 s needed: nothing started
        assert not (tmp_path / "t64_r1.timeline").exists()

    def test_driver_honours_the_stop_file(self, tmp_path):
        stop = tmp_path / "STOP"
        stop.write_text("")
        env = dict(os.environ, CHAMP=BIN, OUT=str(tmp_path / "o"), FILLER_STOPFILE=str(stop))
        r = subprocess.run(["bash", str(HERE / "run_sweep.sh"), "--rounds", "1"], env=env, capture_output=True, text=True)
        assert r.returncode == 0 and "stop file: t48_r1 not started" in r.stdout

    def test_vendored_copies_match_the_recorded_provenance(self):
        prov = (HERE / "PROVENANCE.txt").read_text().splitlines()
        checked = 0
        for i, ln in enumerate(prov):
            if ln.startswith("IDENTICAL"):
                name = ln.split()[1]
                want = prov[i + 1].split("sha256")[1].strip()
                assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == want, name
                checked += 1
        assert checked == 3
