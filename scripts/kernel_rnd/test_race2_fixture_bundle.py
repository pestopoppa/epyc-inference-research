"""Tests for race2_fixture_bundle (CPU only; the store test is skipped when the mutant store is absent)."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import race2_fixture_bundle as rb  # noqa: E402

HDR = ('"Kind","Agent_Id","Queue_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp",'
       '"Private_Segment_Size","Group_Segment_Size","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z",'
       '"Grid_Size_X","Grid_Size_Y","Grid_Size_Z"\n')


def _row(name, priv, lds, wg, grid):
    return f'"KERNEL_DISPATCH",4,1,8,"{name}",1,0,1,{priv},{lds},{wg[0]},{wg[1]},{wg[2]},{grid[0]},{grid[1]},{grid[2]}\n'


def test_read_kernel_trace_folds_and_filters(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(HDR + _row("_Zk", 0, 4096, (256, 1, 1), (1000, 2, 1)) * 3 + _row("_Zk", 0, 4096, (256, 1, 1), (512, 1, 1))
                 + _row("other", 0, 0, (64, 1, 1), (64, 1, 1)))
    g = rb.read_kernel_trace(p, {"_Zk"})
    assert list(g) == ["_Zk"]
    rows = {tuple(r["grid"]): r for r in g["_Zk"]}
    assert rows[(1000, 2, 1)]["dispatches"] == 3
    assert rows[(1000, 2, 1)]["n_workgroups"] == [4, 2, 1] and rows[(1000, 2, 1)]["n_workgroups_total"] == 8
    assert rows[(1000, 2, 1)]["waves_per_wg"] == 4 and rows[(512, 1, 1)]["dispatches"] == 1


def test_geometry_delta():
    a = {"k": [{"grid": [1, 1, 1], "workgroup": [64, 1, 1], "lds_bytes": 0, "dispatches": 5}]}
    b = {"k": [{"grid": [1, 1, 1], "workgroup": [64, 1, 1], "lds_bytes": 0, "dispatches": 2}]}
    c = {"k": [{"grid": [2, 1, 1], "workgroup": [64, 1, 1], "lds_bytes": 0, "dispatches": 2}]}
    assert rb.geometry_delta(a, b, ["k"])["k"]["same_geometry"] is True
    assert rb.geometry_delta(a, c, ["k"])["k"]["same_geometry"] is False
    assert rb.geometry_delta(a, {}, ["k"])["k"] == {"in_baseline": True, "in_mutant": False, "same_geometry": False}


STORE = Path("/mnt/raid0/llm/kernels/builds/MUTANT-race-eqv-r3-20261109-36f024341")


@pytest.mark.skipif(not (STORE / "R1-MMVQ-LDSDMA/BUILD-OK.json").exists() or not Path("/opt/rocm/llvm/bin/llvm-objdump").exists(),
                    reason="mutant store or ROCm llvm absent")
def test_store_barrier_drop_matches_build_proof(tmp_path):
    b = json.loads((STORE / "R1-MMVQ-LDSDMA/BUILD-OK.json").read_text())
    ks = set(b["isa_proof"]["kernels"])
    base = rb.kernel_isa(STORE / "BASELINE/bin/libggml-hip.so", ks)
    mut = rb.kernel_isa(STORE / "R1-MMVQ-LDSDMA/bin/libggml-hip.so", ks, tmp_path / "isa")
    assert set(base) == ks == set(mut)
    for k in ks:
        assert mut[k]["sync"].get("s_barrier", 0) < base[k]["sync"].get("s_barrier", 0)
        assert Path(mut[k]["disassembly"]).stat().st_size > 0
    e = rb.bundle_entry("R1-MMVQ-LDSDMA", b, {}, mut, base, {})
    assert all(v["barrier_dropped"] for v in e["kernels"].values()) and e["launched"] is False
