from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.kernel_rnd.gpu_signals import c4_feed as f  # noqa: E402
from scripts.kernel_rnd.gpu_signals import counters as c  # noqa: E402
from scripts.kernel_rnd.gpu_signals import isa_claims as isa  # noqa: E402
from scripts.kernel_rnd.gpu_signals import race_eqv as re_  # noqa: E402

STATS = ('"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n'
         '"mul_mat_q",4,4000,1000.0,80.0\n"quantize_q8_1",4,1000,250.0,20.0\n')


# ---- C4-5
def test_stats_argv_is_v1_and_parser_roundtrips(tmp_path):
    argv = c.stats_argv("rocprof", tmp_path / "p.csv", ["bin", "-m", "x"])
    assert argv[:5] == ["rocprof", "--tool-version", "1", "--stats", "-o"] and argv[-3:] == ["bin", "-m", "x"]
    with pytest.raises(c.SignalError):
        c.stats_argv("rocprof", tmp_path / "p.txt", ["b"])
    rows = c.parse_stats_csv(STATS)
    assert rows[0].name == "mul_mat_q" and rows[0].total_ns == 4000 and rows[1].percentage == 20.0


def test_parser_rejects_foreign_header_and_bad_rows():
    with pytest.raises(c.SignalError):
        c.parse_stats_csv("Kernel_Name,Start_Timestamp\nx,1\n")      # rocprofv2/v3 shape
    with pytest.raises(c.SignalError):
        c.parse_stats_csv(STATS.replace("4,4000", "0,4000"))
    with pytest.raises(c.SignalError):
        c.parse_stats_csv(STATS.split("\n")[0] + "\n")


def test_find_stats_csv_ignores_hip_hsa_tables(tmp_path):
    (tmp_path / "p.hip_stats.csv").write_text("x")
    (tmp_path / "p.hsa_stats.csv").write_text("x")
    with pytest.raises(c.SignalError):
        c.find_stats_csv(tmp_path)
    (tmp_path / "p.stats.csv").write_text(STATS)
    assert c.find_stats_csv(tmp_path).name == "p.stats.csv"


def test_profile_with_fake_runner(tmp_path):
    seen = {}

    def runner(argv, **kw):
        seen["argv"] = argv
        Path(argv[argv.index("-o") + 1]).with_suffix(".stats.csv").write_text(STATS)
        return type("R", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    rows = c.profile(["bin"], tmp_path, rocprof="rocprof", runner=runner)
    assert [r.name for r in rows] == ["mul_mat_q", "quantize_q8_1"]
    bad = lambda argv, **kw: type("R", (), {"returncode": 3, "stderr": "boom", "stdout": ""})()  # noqa: E731
    with pytest.raises(c.SignalError):
        c.profile(["bin"], tmp_path / "b", rocprof="rocprof", runner=bad)


# ---- C4-6 / C4-7 against the in-repo validated listing
def test_listing_is_465_and_reference_fully_enumerated():
    listing = c.load_listing()
    assert len(listing) == 465
    r = c.cross_check(c.load_reference_partition(), listing)
    assert (r["reference_passes"], r["reference_total"], r["reference_unique"]) == (6, 77, 74)
    assert r["absent_from_validated"] == []
    assert set(r["at_risk_in_reference"]) == set(c.AT_RISK)
    assert all(p["legal"] for p in r["pass_legality"])


def test_cross_check_flags_absent_and_overfull_pass():
    listing = c.load_listing()
    r = c.cross_check([["SQ_WAVES", "NOT_A_COUNTER"]], listing)
    assert r["absent_from_validated"] == ["NOT_A_COUNTER"]
    sq = [n for n, i in listing.items() if i.block == "SQ"][:9]
    assert not c.pass_legality(sq, listing)["legal"]


def test_hawkeye_nine_enumerated_one_pass_collection_unconfirmed():
    h = c.hawkeye_presence(c.load_listing())
    assert h["absent"] == [] and len(h["enumerated"]) == 9
    assert h["pass_legality"]["legal"] and h["collection_confirmed"] is False


# ---- CTR-1
def test_declared_counters_all_enumerated():
    assert set(c.declared_counters()) <= set(c.load_listing())


def test_missing_counter_is_absent_not_zero_and_citations_checked():
    block = c.planner_block([c.CounterReading("SQ_WAVES", 10.0), c.CounterReading("FETCH_SIZE", None),
                             c.CounterReading("SQ_INSTS_LDS", 0.0)])
    assert "FETCH_SIZE" in block["absent"] and "FETCH_SIZE" not in block["present"]
    assert block["present"]["SQ_INSTS_LDS"] == 0.0          # a real zero stays present
    res = c.validate_idea_citations([{"cites": ["SQ_WAVES"]}, {"cites": []}, {"cites": ["FETCH_SIZE"]},
                                     {"cites": ["MADE_UP"]}], block)
    assert [r["ok"] for r in res] == [True, False, False, False]


# ---- C4-8
def test_context_two_slots_no_xcd_and_curated():
    ctx = f.hardware_context({"SQ_WAVES": 1.0})
    f.validate_context(ctx)
    assert ctx["arch_constants"]["cus"] == 104 and ctx["arch_constants"]["xcds"] == 0
    assert ctx["block_scheduling_policy"]["xcd_remap"] == "identity"
    with pytest.raises(c.SignalError):
        f.hardware_context({f"C{i}": 1.0 for i in range(13)})
    bad = {"arch_constants": {"xcds": 0}, "block_scheduling_policy": {"xcd_remap": "swizzle"}}
    with pytest.raises(c.SignalError):
        f.validate_context(bad)
    with pytest.raises(c.SignalError):
        f.validate_context({"arch_constants": {}})


# ---- C4-9 / C4-11
def test_telemetry_fold_prune_and_estimate_error(tmp_path):
    ev = [f.make_event(f"t{i}", "R-hot", True, speedup=1.0, estimated_speedup=1.1) for i in range(5)]
    ev += [f.make_event(f"t{i}", "R-good", True, speedup=2.0, estimated_speedup=1.9) for i in range(5)]
    ev += [f.make_event("t9", "R-good", False)]
    out = f.fold(ev)
    assert out["base_hit_rate"] == 0.5 and out["rules"]["R-good"]["hit_rate"] == 1.0
    assert out["rules"]["R-hot"]["hit_rate"] == 0.0 and out["prune_candidates"] == ["R-hot"]
    assert out["rules"]["R-good"]["tasks_covered"] == 5 and out["rules"]["R-good"]["fires"] == 5
    assert out["rules"]["R-hot"]["estimate_error"]["mean"] == pytest.approx(0.1)
    p = tmp_path / "e.jsonl"
    for e in ev:
        f.append_event(p, e)
    assert f.fold(f.load_events(p)) == out


def test_fired_rule_requires_estimate_and_worst_error_reported():
    with pytest.raises(c.SignalError):
        f.make_event("t", "R", True, speedup=1.2)
    with pytest.raises(c.SignalError):
        f.make_event("t", "R", False, speedup=1.2)
    with pytest.raises(c.SignalError):
        f.make_event("t", "R", True, speedup=-1, estimated_speedup=1.2)
    e = f.estimate_error([f.make_event("a", "R", True, 2.0, 2.0), f.make_event("b", "R", True, 1.0, 1.42)])
    assert e["worst"] == pytest.approx(0.42) and e["mean"] == pytest.approx(0.21)
    assert f.fold([f.make_event("a", "R", True, None, 1.5)])["rules"]["R"]["hit_rate"] is None


# ---- C5-ISA-1
SYM = "_Z6kernelv"
def _build_dis():
    rows = [("s_mov_b64 s[2:3], s[0:1]", 0x1000), ("buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen", 0x1004),
            ("ds_write_b128 v5, v[0:3]", 0x1008), ("v_mfma_f32_16x16x16f16 v[8:11], v[0:1], v[2:3], v[8:11]", 0x100C),
            ("s_endpgm", 0x1010)]
    out = ["%016x <%s>:" % (0x1000, SYM)]
    out += ["\t%-58s// %s: DEADBEEF" % (t, format(a, "X").rjust(12, "0")) for t, a in rows]
    out += ["%016x <other>:" % 0x2000,
            "\t%-58s// %s: DEADBEEF" % ("v_dot2_f32_f16 v0, v1, v2, v3", format(0x2000, "X").rjust(12, "0"))]
    return "\n".join(out) + "\n"


DIS = _build_dis()


def _insts():
    return isa.ga.split_disassembly(DIS, {SYM})[SYM]


def test_isa_archive_is_content_addressed_and_isolated(tmp_path):
    rec = isa.archive(DIS, SYM, tmp_path, "a" * 64)
    assert (tmp_path / rec["isa_file"]).read_text().count("v_dot") == 0
    assert rec["mechanism_counts"]["mfma"] == 1 and rec["mechanism_counts"]["lds_staging"] == 1
    assert isa.archive(DIS, SYM, tmp_path, "a" * 64)["isa_sha256"] == rec["isa_sha256"]
    with pytest.raises(c.SignalError):
        isa.archive(DIS, "nope", tmp_path, "a" * 64)
    with pytest.raises(c.SignalError):
        isa.archive(DIS, SYM, tmp_path, "xyz")


def test_claims_static_dynamic_and_refusals():
    ok = isa.check_claims(["mfma", "lds_staging"], _insts())
    assert ok["accepted"] and {r["status"] for r in ok["claims"]} == {"static_only"}
    absent = isa.check_claims(["mfma", "dot"], _insts())
    assert not absent["accepted"] and absent["claims"][1]["status"] == "refused_absent_in_isa"
    zero = isa.check_claims(["mfma"], _insts(), [c.CounterReading("SQ_INSTS_MFMA", 0.0)])
    assert zero["claims"][0]["status"] == "refused_counter_zero" and not zero["accepted"]
    live = isa.check_claims(["mfma"], _insts(), [c.CounterReading("SQ_INSTS_MFMA", 5.0)])
    assert live["claims"][0]["status"] == "confirmed_dynamic"
    none = isa.check_claims(["mfma"], _insts(), [c.CounterReading("SQ_INSTS_MFMA", None)])
    assert none["claims"][0]["status"] == "static_only"      # None is not zero
    with pytest.raises(c.SignalError):
        isa.check_claims(["tensor_core"], _insts())


def test_lds_dma_is_distinguished_from_plain_buffer_load():
    d = DIS.replace("buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen", "buffer_load_dword v4, s[8:11], 0 offen lds")
    i = isa.ga.split_disassembly(d, {SYM})[SYM]
    assert isa.check_claims(["lds_dma"], i)["accepted"] and not isa.check_claims(["buffer_load"], i)["accepted"]


def test_claim_counters_are_enumerated():
    listing = c.load_listing()
    assert all(ctr in listing for _, ctr in isa.MECHANISMS.values() if ctr)


# ---- ATT-1
def test_mfma_pass_table_and_efficiency():
    assert re_.mfma_passes("v_mfma_f32_16x16x16f16") == 8 and re_.mfma_passes("v_mfma_i32_32x32x8i8") == 16
    assert re_.mfma_passes("v_mfma_f32_4x4x4f16") == 2 and re_.mfma_passes("v_mfma_f64_16x16x4f64") == 8
    assert re_.mfma_passes("v_mfma_f64_4x4x4f64") == 4
    ops = ["v_mfma_f32_16x16x16f16"] * 2 + ["v_add_u32"]
    assert re_.mfma_cycles(ops) == 64 and re_.mfma_efficiency(ops, 128) == 0.5
    with pytest.raises(c.SignalError):
        re_.mfma_passes("v_add_u32")
    with pytest.raises(c.SignalError):
        re_.mfma_efficiency(ops, 0)


# ---- manifest + RACE-1
def test_manifest_shape():
    m = re_.load_manifest()
    assert len(m["race1"]["mutants"]) == 5 and len(m["race1"]["isa_only_mutants"]) == 2
    assert len(m["eqv1"]["mutants"]) == 8 and m["race1"]["runs_per_mutant"] == 20


def _mut(passes, base=10, mut=9, waves=4):
    return {"gate_pass": [True] * passes + [False] * (20 - passes), "barriers_base": base, "barriers_mut": mut,
            "waves_per_wg": waves}


def test_race1_decision_rule():
    ok = [True] * 20
    d = re_.race1_decide({"a": _mut(0), "b": _mut(1), "w": _mut(20, waves=1), "p": _mut(0, mut=10)}, ok)
    assert d["mutants"]["a"]["status"] == "caught" and d["mutants"]["b"]["status"] == "miss"
    assert d["mutants"]["w"]["status"] == "excluded_single_wave"
    assert d["mutants"]["p"]["status"] == "invalid_no_structural_proof"
    assert d["measured_race_misses"] == ["b"] and d["decision"] == "gate_blind_to_races"
    assert re_.race1_decide({"a": _mut(0)}, ok)["decision"] == "gate_sees_all_tested_races"
    assert re_.race1_decide({"a": _mut(0)}, [True] * 19 + [False])["decision"] == "INVALID_UNMUTATED_FAILS"
    with pytest.raises(c.SignalError):
        re_.race1_decide({"a": _mut(0)}, [True] * 19)                # short list refused, not padded


def test_race2_m1_acceptance():
    kw = dict(prod_findings=0, mutant_flagged={"m1": True, "m2": True}, required_mutants=["m1", "m2"],
              isa_only_flagged={"i1": True, "i2": True}, seconds_per_kernel_instance=599)
    assert re_.race2_m1_accept(**kw)["accepted"]
    assert not re_.race2_m1_accept(**{**kw, "prod_findings": 1})["accepted"]
    assert re_.race2_m1_accept(**{**kw, "mutant_flagged": {"m1": True}})["unflagged"] == ["m2"]
    assert not re_.race2_m1_accept(**{**kw, "isa_only_flagged": {"i1": True}})["accepted"]
    assert not re_.race2_m1_accept(**{**kw, "seconds_per_kernel_instance": 601})["accepted"]


# ---- EQV-1
def test_eqv1_decision_rule():
    ok5 = [True] * 5
    caught = [False] * 5
    surv = {"m1": caught, "m2": [False, False, True, False, False], "cA": caught, "cB": caught}
    d = re_.eqv1_decide(surv, ["cA", "cB"], ok5)
    assert d["decision"] == "open_equivalence_milestone_for_subclass" and d["survivors"] == ["m2"]
    assert re_.eqv1_decide({**surv, "m2": caught}, ["cA", "cB"], ok5)["decision"] == \
        "close_equivalence_port_with_catch_table"
    assert re_.eqv1_decide({**surv, "cA": ok5}, ["cA", "cB"], ok5)["decision"] == "INVALID_HARNESS"
    assert re_.eqv1_decide(surv, ["cA", "cB"], [True] * 4 + [False])["decision"] == "INVALID_HARNESS"
