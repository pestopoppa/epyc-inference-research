"""Tests for mmvq_hoist_audit (INF37-LB1).

Fixtures are llvm-objdump text, so the suite needs neither ROCm nor a GPU.  The Q8_0 ncols=2 hot
loop in ``samples/mmvq_q8_0_n2.dis.txt`` is the real v10 ``mul_mat_vec_q<Q8_0,2>`` body
(symbol shortened to ``k``): 2 rows x 2 columns, weights loaded once per row.  One integration
test runs against the frozen production library when it and ROCm's llvm-objdump are present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx90a_isa_audit as ga  # noqa: E402
import mmvq_hoist_audit as mh  # noqa: E402

HERE = Path(__file__).resolve().parent
LIB = Path("/mnt/raid0/llm/kernels/production/gpu/libggml-hip.so.0.16.0")


def _insts(text: str, name: str = "k"):
    return ga.split_disassembly(text, {name})[name]


def test_real_q8_0_n2_loop_weights_are_per_row_not_per_column():
    insts = _insts((HERE / "samples/mmvq_q8_0_n2.dis.txt").read_text())
    rep = mh.hot_loop_report(insts, "k")
    # 2 rows x (1 ushort d + 2 dword qs) = 6; 2 columns x (1 dword ds + 1 dwordx2 qs) = 4
    assert rep["weight_loads"] == 6
    assert rep["act_loads"] == 4
    assert rep["other_loads"] == 0 and rep["mixed_loads"] == 0
    assert rep["dot"] == 8
    assert rep["weight_load_ops"].count("global_load_ushort") == 2
    assert rep["internal_branches"] == 0


def test_parse_regs_ranges_and_scalars():
    assert mh.parse_regs("v[4:7]") == {("v", 4), ("v", 5), ("v", 6), ("v", 7)}
    assert mh.parse_regs("s9") == {("s", 9)}
    assert mh.parse_regs("vcc") == set()
    assert mh.parse_regs("0x480") == set()


def test_split_operands_drops_modifiers():
    assert mh.split_operands("v[14:15], v[14:15], off offset:4") == ["v[14:15]", "v[14:15]", "off"]
    assert mh.split_operands("v[16:17], v[20:21], v[18:19] op_sel_hi:[1,0]") == [
        "v[16:17]", "v[20:21]", "v[18:19]"]


def _synthetic(loads_per_col: int, ncols: int, weight_loads: int) -> str:
    """kernarg vx -> s[20:21], vy -> s[22:23]; one loop of weight + ncols*loads_per_col y loads."""
    lines = ["0000 <k>:",
             "\ts_load_dwordx4 s[20:23], s[4:5], 0x0       // 0000: 00000000",
             "\ts_waitcnt lgkmcnt(0)                       // 0008: 00000000",
             "\tv_mov_b32_e32 v2, s20                      // 0010: 00000000",
             "\tv_mov_b32_e32 v3, s21                      // 0014: 00000000",
             "\tv_mov_b32_e32 v4, s22                      // 0018: 00000000",
             "\tv_mov_b32_e32 v5, s23                      // 001C: 00000000"]
    addr = 0x20
    header = addr
    for i in range(weight_loads):
        lines.append(f"\tglobal_load_dword v{10 + i}, v[2:3], off offset:{4 * i}   // {addr:04X}: 0")
        addr += 8
    for c in range(ncols):
        for i in range(loads_per_col):
            lines.append(f"\tglobal_load_dword v{40 + c * 8 + i}, v[4:5], off offset:{36 * c + 4 * i}"
                         f"   // {addr:04X}: 0")
            addr += 8
    lines.append(f"\tv_dot4c_i32_i8_e32 v60, v10, v40            // {addr:04X}: 0")
    addr += 8
    lines.append(f"\ts_cbranch_execnz 0                          // {addr:04X}: 0 <k+0x{header - 0x0:x}>")
    addr += 4
    lines.append(f"\ts_endpgm                                    // {addr:04X}: 0")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("ncols", [1, 4, 8])
def test_taint_separates_weight_from_activation_loads(ncols):
    rep = mh.hot_loop_report(_insts(_synthetic(loads_per_col=2, ncols=ncols, weight_loads=3)), "k")
    assert rep["weight_loads"] == 3
    assert rep["act_loads"] == 2 * ncols
    assert rep["other_loads"] == 0


def test_carry_registers_do_not_cross_contaminate_pointer_chains():
    # vx chain and vy chain interleave on vcc; the Y load must stay pure Y.
    text = "\n".join([
        "0000 <k>:",
        "\ts_load_dwordx4 s[20:23], s[4:5], 0x0       // 0000: 0",
        "\ts_waitcnt lgkmcnt(0)                       // 0008: 0",
        "\tv_add_co_u32_e32 v2, vcc, s20, v1          // 0010: 0",
        "\tv_add_co_u32_e32 v4, vcc, s22, v1          // 0018: 0",
        "\tv_addc_co_u32_e32 v3, vcc, 0, v7, vcc      // 0020: 0",
        "\tglobal_load_dword v10, v[2:3], off          // 0028: 0",
        "\tglobal_load_dword v11, v[4:5], off          // 0030: 0",
        "\tv_dot4c_i32_i8_e32 v12, v10, v11            // 0038: 0",
        "\ts_cbranch_execnz 0                          // 0040: 0 <k+0x28>",
        "\ts_endpgm                                    // 0044: 0", ""])
    rep = mh.hot_loop_report(_insts(text), "k")
    assert (rep["weight_loads"], rep["act_loads"], rep["mixed_loads"]) == (1, 1, 0)


def test_rows_per_thread_matches_v10_table():
    assert mh.rows_per_thread(8, 1, small_k=False) == 1
    assert mh.rows_per_thread(8, 1, small_k=True, eight_warp=True) == 4
    assert mh.rows_per_thread(8, 4, small_k=False) == 4
    assert mh.rows_per_thread(8, 4, small_k=True) == 2
    assert mh.rows_per_thread(12, 8, small_k=False) == 2
    assert mh.rows_per_thread(12, 1, small_k=True) == 1  # halfwave: one row per thread
    assert mh.rows_per_thread(23, 1, small_k=True) == 2


def _row(t, n, w_ops, act, r=2, sk=False, ew=False):
    return {"type": t, "ncols": n, "small_k": sk, "eight_warp": ew, "rows_per_thread": r,
            "weight_load_ops": w_ops, "weight_loads": len(w_ops), "act_loads": act,
            "act_loads_per_col": act / n, "w_loads_per_row": len(w_ops) / r,
            "dot_per_unit": 4.0, "internal_branches": 0}


def test_verdict_hoisted_vs_partly_repeated():
    hoisted = [_row("Q8_0", n, ["global_load_dword"] * 4 + ["global_load_ushort"] * 2, 2 * n)
               for n in (1, 2, 8)]
    mh.flag_dot_unit(hoisted)
    assert mh.verdict(hoisted, "Q8_0")["verdict"] == "hoisted"
    rep = [_row("Q4_K", n, ["global_load_dword"] * 6 + ["global_load_ushort"] * (6 * n), 6 * n)
           for n in (1, 2, 3, 8)]
    mh.flag_dot_unit(rep)
    v = mh.verdict(rep, "Q4_K")
    assert v["verdict"] == "partly_repeated_per_column"
    assert v["repeated_ops"] == ["global_load_ushort"]


def test_overcounted_cell_is_flagged_and_excluded():
    rows = [_row("Q4_K", 1, ["global_load_dword"] * 3, 6, r=1),
            _row("Q4_K", 4, ["global_load_dword"] * 6, 36)]  # 9 y loads/col vs 6 at ncols=1
    mh.flag_dot_unit(rows)
    assert rows[1]["load_overcount"] is True
    assert mh.verdict(rows, "Q4_K")["verdict"] == "no_data"


@pytest.mark.skipif(not (LIB.exists() and (ga.LLVM_BIN / "llvm-objdump").exists()),
                    reason="production HIP library or ROCm llvm-objdump absent")
def test_production_library_q8_0_and_iq4_xs_are_hoisted_and_q4_k_is_not():
    doc = mh.audit_library(LIB)
    by = {(v["type"], v["small_k"]): v for v in map(lambda t: mh.verdict(doc["rows"], t[0], t[1]),
                                                   [("Q8_0", False), ("IQ4_XS", False), ("Q4_K", False)])}
    assert by[("Q8_0", False)]["verdict"] == "hoisted"
    assert by[("IQ4_XS", False)]["verdict"] == "hoisted"
    assert by[("Q4_K", False)]["verdict"] == "partly_repeated_per_column"
    assert not any(r["dot_unit_mismatch"] for r in doc["rows"] if r["type"] in ("Q8_0", "IQ4_XS"))
