"""CPU-only tests for race2_executor on synthetic gfx90a snippets (fixture-format .s)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import race2_executor as rx  # noqa: E402


def S(*lines, name="_Zk"):
    out, a = [f"; {name}"], 0x1000
    for ln in lines:
        op, _, args = ln.partition(" ")
        out.append(f"{a:x}\t{op}\t{args}")
        a += 4
    return "\n".join(out) + "\n"


def run(text, **kw):
    return rx.analyze(rx.parse_s(text)[1], **kw)


def test_write_barrier_read_is_clean():
    r = run(S("ds_write_b32 v0, v1", "s_waitcnt lgkmcnt(0)", "s_barrier", "ds_read_b32 v2, v0", "s_endpgm"))
    assert r["verdict"] == "CLEAN" and r["n_lds"] == 2


def test_deleted_barrier_is_finding():
    r = run(S("ds_write_b32 v0, v1", "s_waitcnt lgkmcnt(0)", "ds_read_b32 v2, v0", "s_endpgm"))
    assert r["verdict"] == "FINDING" and r["findings"][0]["kinds"] == "W->R"


def test_barrier_before_completion_does_not_order():
    r = run(S("ds_write_b32 v0, v1", "s_barrier", "ds_read_b32 v2, v0", "s_endpgm"))
    assert r["verdict"] == "FINDING"


def test_async_lds_needs_vmcnt0():
    ok = run(S("buffer_load_dword v0, s[0:3], 0 offen lds", "s_waitcnt vmcnt(0)", "s_barrier", "ds_read_b32 v1, v2", "s_endpgm"))
    bad = run(S("buffer_load_dword v0, s[0:3], 0 offen lds", "s_waitcnt vmcnt(1)", "s_barrier", "ds_read_b32 v1, v2", "s_endpgm"))
    assert ok["verdict"] == "CLEAN" and bad["verdict"] == "FINDING"


def test_partial_lgkm_with_smem_is_conservative():
    # smem may return out of order: lgkmcnt(1) proves nothing about the ds write
    r = run(S("s_load_dword s0, s[4:5], 0x0", "ds_write_b32 v0, v1", "s_waitcnt lgkmcnt(1)", "s_barrier",
              "ds_read_b32 v2, v0", "s_endpgm"))
    assert r["verdict"] == "FINDING"
    r2 = run(S("ds_write_b32 v0, v1", "ds_read_b32 v3, v4", "s_waitcnt lgkmcnt(1)", "s_barrier", "s_endpgm"))
    # read-after-write in one epoch is already a finding; the point is no crash and in-order completion
    assert r2["verdict"] == "FINDING"


def test_loop_back_edge_war():
    # loop: read; barrier missing before next iteration's write -> W after R across waves
    r = run(S("ds_write_b32 v0, v1", "s_waitcnt lgkmcnt(0)", "s_barrier", "ds_read_b32 v2, v0",
              "s_waitcnt lgkmcnt(0)", "s_cbranch_scc1 65530", "s_endpgm"))
    # branch at 0x1014: target = 0x1018 + 4*(-6) = 0x1000 (the write)
    assert r["verdict"] == "FINDING" and any(f["kinds"] == "R->W" for f in r["findings"])
    r = run(S("ds_write_b32 v0, v1", "s_waitcnt lgkmcnt(0)", "s_barrier", "ds_read_b32 v2, v0",
              "s_waitcnt lgkmcnt(0)", "s_barrier", "s_cbranch_scc1 65529", "s_endpgm"))
    assert r["verdict"] == "CLEAN"


def test_ignored_and_atomics_and_rr():
    assert run(S("ds_bpermute_b32 v0, v1, v2", "ds_bpermute_b32 v0, v1, v2", "s_endpgm"))["verdict"] == "CLEAN"
    assert run(S("ds_add_u32 v0, v1", "ds_add_u32 v0, v2", "s_endpgm"))["verdict"] == "CLEAN"
    assert run(S("ds_read_b32 v0, v1", "ds_read_b32 v2, v3", "s_endpgm"))["verdict"] == "CLEAN"


def test_fail_closed():
    assert run(S("s_setpc_b64 s[0:1]"))["verdict"] == "UNKNOWN"
    assert run(S("s_branch 100", "s_endpgm"))["verdict"] == "UNKNOWN"
    assert run(S("ds_gws_init v0", "s_endpgm"))["verdict"] == "UNKNOWN"
    assert run(S("s_waitcnt garbage", "s_endpgm"))["verdict"] == "UNKNOWN"


def test_divergent_paths_join():
    # one path has a barrier, the other skips it -> may-analysis keeps the write in the window
    r = run(S("ds_write_b32 v0, v1", "s_waitcnt lgkmcnt(0)", "s_cbranch_execz 1", "s_barrier",
              "ds_read_b32 v2, v0", "s_endpgm"))
    assert r["verdict"] == "FINDING"


def test_single_wave_na(tmp_path):
    p = tmp_path / "k.s"
    p.write_text(S("ds_write_b32 v0, v1", "ds_read_b32 v2, v0", "s_endpgm"))
    assert rx.analyze_file(p, waves_per_wg=[1])["verdict"] == "N/A"
    r = rx.analyze_file(p, waves_per_wg=[4])
    assert r["verdict"] == "FINDING" and r["kernel"] == "_Zk" and len(r["disassembly_sha256"]) == 64


def test_real_format_line():
    name, ins = rx.parse_s("; _ZL25k\n883f0\ts_barrier\t\n883f4\ts_cbranch_scc1\t22\n")
    assert name == "_ZL25k" and ins[0].op == "s_barrier" and ins[1].args == "22"
    assert rx._simm16("65467") == -69


def _rec(k, v, s=0.1):
    return {"kernel": k, "disassembly_sha256": k * 4, "verdict": v, "seconds": s}


def test_decide_bar():
    fx = {f"F{i}": [_rec(f"k{i}", "FINDING")] for i in range(7)}
    ctl = [_rec("b1", "CLEAN"), _rec("b2", "CLEAN"), _rec("b3", "N/A")]
    d = rx.decide(fx, ctl)
    assert d["verdict"] == "EXECUTOR_M1_BAR_MET" and d["sensitivity"] == "7/7" and d["specificity"] == "2/2"
    d = rx.decide(fx, ctl + [_rec("b4", "UNKNOWN")])
    assert d["verdict"] == "EXECUTOR_M1_BAR_MISSED" and d["controls_failed"] == ["b4"]
    fx2 = dict(fx, F0=[_rec("k0", "CLEAN"), _rec("kx", "UNKNOWN")])
    d = rx.decide(fx2, ctl)
    assert d["undetected"] == ["F0"] and d["sensitivity"] == "6/7"
    d = rx.decide(fx, ctl[:2] + [_rec("slow", "CLEAN", 601)])
    assert not d["checks"]["within_time_budget"]


def test_loop_with_unwaited_smem_and_ds_converges():
    # s_load and ds_read issued every iteration with no lgkmcnt(0) inside the loop must still reach a fixpoint
    r = run(S("s_load_dword s0, s[4:5], 0x0", "ds_read_b32 v2, v0", "s_cbranch_scc1 65533", "s_waitcnt lgkmcnt(0)",
              "s_endpgm"))
    assert r["verdict"] == "CLEAN"
