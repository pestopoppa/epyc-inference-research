"""CPU-only tests for the launch-capture host side (the C shim itself needs a GPU run)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import race2_launch_capture as lc  # noqa: E402

NOTES = """Displaying notes found in: .note
  AMDGPU               0x00063053	NT_AMDGPU_METADATA (AMDGPU Metadata)
    AMDGPU Metadata:
        ---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           8
        .value_kind:     hidden_global_offset_x
    .kernarg_segment_size: 272
    .name:           _Zk1
  - .kernarg_segment_size: 0
    .name:           _Zk2
amdhsa.target:   amdgcn-amd-amdhsa--gfx90a
...
"""



def test_argtable_skips_hidden():
    t = lc.argtable(NOTES).splitlines()
    assert t == ["_Zk1 272 2 0:8 8:4", "_Zk2 0 0"]


def test_read_capture_folds_and_tolerates_torn_line(tmp_path):
    p = tmp_path / "c.jsonl"
    r = {"k": "_Zk1", "grid": [4, 1, 1], "block": [256, 1, 1], "shmem": 0, "kernarg": "00" * 16}
    p.write_text(json.dumps(r) + "\n" + json.dumps(r) + "\n" + json.dumps(dict(r, grid=[8, 1, 1])) + "\n{\"k\":")
    c = lc.read_capture(p)
    assert [l["dispatches"] for l in c["_Zk1"]] == [2, 1] and c["_Zk1"][0]["waves_per_wg"] == 4
    plan = lc.replay_plan(c, "_Zk1")
    assert plan[0]["gridDim"] == [4, 1, 1] and plan[0]["kernarg"] == bytes(16)


def test_replay_refuses_missing_kernarg(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"k": "_Zk", "grid": [1, 1, 1], "block": [64, 1, 1], "shmem": 0, "kernarg": None}) + "\n")
    with pytest.raises(ValueError):
        lc.replay_plan(lc.read_capture(p), "_Zk")


def test_shim_source_present():
    s = lc.SHIM_SOURCE.read_text()
    assert "hipLaunchKernel" in s and "RACE2_CAPTURE_ARGTABLE" in s and "RTLD_NEXT" in s
