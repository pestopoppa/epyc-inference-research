"""AK-AMDAHL-PRUNE + AKX-ALL-16 serving shares. Hermetic: synthetic traces and the
packaged data file only; no GPU, no server, no profiler."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import amdahl, loop, pipeline
from . import serving_shares as ss

HEADER = ("Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n")


def _trace(path: Path, rows):
    path.write_text(HEADER + "".join(f'KERNEL_DISPATCH,"{n}",{s},{e}\n' for n, s, e in rows))
    return path


# ------------------------------------------------------------------ the math
def test_ceilings_match_amdahl():
    assert amdahl.ceiling_zero(0.5) == pytest.approx(1.0)
    assert amdahl.ceiling_zero(0.03) == pytest.approx(0.03 / 0.97)
    assert amdahl.ceiling_at(0.5, 2.0) == pytest.approx(1 / 0.75 - 1)
    assert amdahl.ceiling_at(0.2, 1e12) == pytest.approx(amdahl.ceiling_zero(0.2))
    with pytest.raises(ValueError):
        amdahl.ceiling_zero(1.0)


def test_verdicts():
    assert amdahl.assess(None)["verdict"] == "unknown"
    assert amdahl.assess(0.02)["verdict"] == "below_bar_hard"          # 2.04% < 3%
    assert amdahl.assess(0.04)["verdict"] == "below_bar_realistic"     # 4.2% / 2.0% at 2x
    assert amdahl.assess(0.10)["verdict"] == "clears"
    assert amdahl.assess(0.04, speedup=10)["verdict"] == "clears"


# ------------------------------------------------------------------ serving shares
def test_build_post_prefill_window_and_validate(tmp_path):
    trace = _trace(tmp_path / "t.csv", [
        ("Cijk_prefill", 0, 1000),                 # prefill: excluded
        ("mul_mat_vec_q<(ggml_type)8, 1>", 1100, 1800),
        ("quantize_q8_1<false>", 1800, 2000),
        ("rms_norm_f32<1024>", 2000, 2100)])
    entry = ss.build([trace], rule="post_prefill", model="m", mode="verify")
    shares = {row["kernel"]: row["share"] for row in entry["kernels"]}
    assert shares["mul_mat_vec_q<(ggml_type)8, 1>"] == pytest.approx(0.7)
    assert "Cijk_prefill" not in shares
    assert entry["sources"][0]["sha256"] and entry["busy_ns"] == 1000
    all_entry = ss.build([trace], rule="all", model="m", mode="all")
    assert any(r["kernel"] == "Cijk_prefill" for r in all_entry["kernels"])
    body = {"schema": ss.SCHEMA, "version": 1, "generated": "x", "hardware": "h",
            "profiles": [entry]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(body))
    loaded = ss.load(path)
    profile = ss.get_profile(loaded, "m/verify")
    assert ss.share_for(profile, "mul_mat_vec_q") == pytest.approx(0.7)
    assert ss.share_for(profile, "mul_mat_vec_q|rms_norm_f32") == pytest.approx(0.8)
    assert ss.share_for(profile, "nothing_here") is None


def test_validate_refuses_bad_files(tmp_path):
    with pytest.raises(ss.SharesInvalid):
        ss.validate({"schema": "wrong"})
    base = {"schema": ss.SCHEMA, "version": 1, "generated": "x", "hardware": "h"}
    with pytest.raises(ss.SharesInvalid):          # shares must sum to 1
        ss.validate({**base, "profiles": [{"model": "m", "mode": "x", "window_rule": "all",
                     "sources": [{"path": "p"}], "kernels": [{"kernel": "k", "share": 0.5}]}]})
    with pytest.raises(ss.SharesInvalid):          # no traces, no data
        ss.validate({**base, "profiles": [{"model": "m", "mode": "x", "window_rule": "all",
                     "sources": [], "kernels": [{"kernel": "k", "share": 1.0}]}]})
    with pytest.raises(ss.SharesInvalid):
        ss.load(tmp_path / "missing.json")


def test_packaged_file_matches_the_hotspot_scan():
    """research/deep-dives/mi210-next-hotspots-20261118.md shares, recomputed from traces."""
    shares = ss.load()
    def fam(pid):
        return dict(ss.family_shares(ss.get_profile(shares, pid)))
    assert fam("27b-q8_0/dflash2_verify_d8192")["mul_mat_vec_q"] == pytest.approx(0.72, abs=0.01)
    assert fam("27b-q8_0/dflash2_verify_d131072")["mul_mat_vec_q"] == pytest.approx(0.54, abs=0.01)
    assert fam("27b-q8_0/dflash2_verify_d131072")["gv_fattn_kernel"] == pytest.approx(0.27, abs=0.01)
    a3b = fam("a3b-q8_0/mtp_verify_n4")
    assert a3b["mul_mat_vec_q"] == pytest.approx(0.269, abs=0.005)
    assert a3b["mul_mat_f"] == pytest.approx(0.134, abs=0.005)
    assert a3b["mul_mat_vec_q_moe"] == pytest.approx(0.132, abs=0.005)
    assert a3b["k_bin_bcast"] == pytest.approx(0.070, abs=0.005)


# ------------------------------------------------------------------ assessor
def _assessor(mode, tmp_path, pid="27b-q8_0/dflash2_verify_d32768"):
    shares = ss.load()
    return amdahl.Assessor(mode=mode, profile=ss.get_profile(shares, pid), profile_id=pid,
                           log_path=tmp_path / amdahl.LOG_NAME)


def test_advisory_logs_but_never_demotes(tmp_path):
    a = _assessor("advisory", tmp_path)
    row = a.record("quantize_q8_1", mechanism_id="akm-fold-q8")
    assert row["verdict"] == "below_bar_hard" and row["demoted"] is False
    assert a.demote(SimpleNamespace(target_symbol="quantize_q8_1", mechanism_id="x")) is False
    logged = [json.loads(l) for l in (tmp_path / amdahl.LOG_NAME).read_text().splitlines()]
    assert logged[0]["mechanism_id"] == "akm-fold-q8" and "ceiling_zero" in logged[0]


def test_demote_mode_and_unknown_is_never_demoted(tmp_path):
    a = _assessor("demote", tmp_path)
    assert a.demote(SimpleNamespace(target_symbol="quantize_q8_1", mechanism_id="x"))
    assert not a.demote(SimpleNamespace(target_symbol="mul_mat_vec_q", mechanism_id="y"))
    assert not a.demote(SimpleNamespace(target_symbol="host_launcher_xyz", mechanism_id="z"))
    none = amdahl.Assessor(mode="demote", profile=None, profile_id=None)
    assert not none.demote(SimpleNamespace(target_symbol="quantize_q8_1", mechanism_id="x"))
    assert amdahl.Assessor(mode="off", profile=None, profile_id=None).record("x") is None


def test_planner_view_renders(tmp_path):
    view = _assessor("advisory", tmp_path).planner_view()
    assert view["status"] == "observed" and view["kernels"][0]["kernel"] == "mul_mat_vec_q"
    text = "\n".join(amdahl.render(view))
    assert "End-to-end ceilings" in text and "keep gate itself is unchanged" in text
    assert amdahl.render({"status": "off"}) == []
    assert "unavailable" in "\n".join(amdahl.render(
        amdahl.Assessor(mode="advisory", profile=None, profile_id=None).planner_view()))


def test_from_args_default_is_advisory(tmp_path):
    import argparse
    parser = argparse.ArgumentParser()
    amdahl.add_arguments(parser)
    args = parser.parse_args(["--serving-shares-profile", "a3b-q8_0/mtp_verify_n4"])
    assert args.amdahl_prune == "advisory"
    a = amdahl.from_args(args, store=tmp_path)
    assert a.profile is not None and a.log_path == tmp_path / amdahl.LOG_NAME
    with pytest.raises(ss.SharesInvalid):
        amdahl.from_args(parser.parse_args(["--serving-shares-profile", "no/such"]), store=None)


# ------------------------------------------------------------------ replay (acceptance)
def test_replay_2026_11_18_candidates(tmp_path):
    """Today's GPU candidates against their served profiles.

    Killed by hand for Amdahl: folding q8_1 into MMVQ (target quantize_q8_1) on the 27B
    DFlash2 verify -- the ceiling flags it up front. The kept-direction candidate L1
    (C16-TOPK-STAGE, the A3B router chain) clears its ceiling, so no keep is demoted.
    CPU candidates (paired gate/up, Q38FN top-k) have no GPU profile: unknown, untouched."""
    shares = ss.load()
    q27 = ss.get_profile(shares, "27b-q8_0/dflash2_verify_d32768")
    a3b = ss.get_profile(shares, "a3b-q8_0/mtp_verify_n4")
    assert amdahl.demoted(amdahl.assess(ss.share_for(q27, "quantize_q8_1")))
    chain = "soft_max_f32|k_argsort_f32_i32|k_get_rows_float|reduce_rows_f32|op_clamp_kernel"
    keep = amdahl.assess(ss.share_for(a3b, chain))
    assert keep["verdict"] == "clears", keep
    assert amdahl.assess(ss.share_for(q27, "ggml_cpu_gate_up_pair"))["verdict"] == "unknown"


# ------------------------------------------------------------------ tail wiring
def _hyp(mechanism, symbol):
    return loop.Hypothesis(mechanism_id=mechanism, statement="s", falsifier="f",
                           target_surface="ggml/src/ggml-cuda/x.cu", target_symbol=symbol)


def test_tail_times_demoted_candidates_last(tmp_path):
    a = _assessor("demote", tmp_path)
    head = "0" * 40
    tail = pipeline.SerializedTail(lambda: head, priority=lambda h: None, demote=a.demote)
    admitted, release = [], threading.Event()

    def holder():
        with tail.session(head, hypothesis=_hyp("holder", "none")):
            release.wait(10)

    def waiter(h):
        with tail.session(head, hypothesis=h):
            admitted.append(h.mechanism_id)

    threads = [threading.Thread(target=holder)]
    threads[0].start()
    deadline = time.monotonic() + 5
    while not tail._admitted and time.monotonic() < deadline:
        time.sleep(0.005)
    arrivals = [("low", "quantize_q8_1"), ("unknown", "host_fn"), ("big", "mul_mat_vec_q")]
    for index, (m, s) in enumerate(arrivals):
        t = threading.Thread(target=waiter, args=(_hyp(m, s),))
        threads.append(t)
        t.start()
        while len(tail._waiting) < index + 1 and time.monotonic() < deadline:
            time.sleep(0.005)
    release.set()
    for t in threads:
        t.join(10)
    assert admitted == ["unknown", "big", "low"]
    low = next(r for r in tail.admission_log if r["mechanism_id"] == "low")
    assert low["amdahl_demoted"] is True


def test_failing_demote_hook_is_not_a_failed_attempt():
    def boom(_h):
        raise RuntimeError("x")
    tail = pipeline.SerializedTail(lambda: "0" * 40, demote=boom)
    with tail.session("0" * 40, hypothesis=_hyp("a", "x")):
        pass
    assert tail.admission_log[0]["amdahl_demoted"] is False


# ------------------------------------------------------------------ planner-context canary
def test_assembled_planner_context_shows_the_ceilings():
    """The ceiling table reaches the context the planner actually assembles
    (`actors.render_context`), built through run.py's own assessor helper."""
    import argparse
    from . import actors, run
    parser = argparse.ArgumentParser()
    amdahl.add_arguments(parser)
    args = parser.parse_args(["--serving-shares-profile", "27b-q8_0/dflash2_verify_d32768"])
    args.store = None
    ctx = {"gpu_serving_profile": {"status": "observed", "windows": {}},
           "amdahl_ceilings": run._amdahl_assessor(args).planner_view(), "kernel_hotspots": []}
    text = actors.render_context(ctx)
    assert "End-to-end ceilings by kernel (serving profile `27b-q8_0/dflash2_verify_d32768`)" in text
    assert "| below_bar_hard | `quantize_q8_1` |" in text
    assert "End-to-end ceilings" not in actors.render_context(
        {"gpu_serving_profile": {"status": "observed", "windows": {}}, "kernel_hotspots": []})
