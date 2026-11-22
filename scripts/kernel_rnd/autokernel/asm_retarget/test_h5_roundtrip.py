import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5_roundtrip as h5  # noqa: E402

SRC = Path("/mnt/raid0/llm/tmp/dive-dtree-intake-20261120/c/aiter/hsa/gfx942/fmha_v3_fwd/MI300")
REF = Path("/mnt/raid0/llm/kernels/src/llama-268cfdf8a-specstale1/ggml/src/ggml-hip/fa-asm-gfx90a")
LLVM = Path("/opt/rocm/llvm/bin/llvm-mc")


def test_pins_are_well_formed():
    for src, ref in h5.PINNED.values():
        assert len(src) == len(ref) == 64 and src != ref


def test_missing_inputs_do_not_pass(tmp_path):
    doc = h5.roundtrip(tmp_path / "nope", tmp_path / "nope", tmp_path / "w")
    assert doc["ok"] is False and all(not r["bit_identical"] for r in doc["rows"])


@pytest.mark.skipif(not (SRC.is_dir() and REF.is_dir() and LLVM.exists()),
                    reason="AITER gfx942 inputs / H5 reference / ROCm llvm absent")
def test_h5_round_trips_bit_identically(tmp_path):
    doc = h5.roundtrip(SRC, REF, tmp_path)
    assert doc["ok"], doc
    assert {r["kernel"] for r in doc["rows"]} == set(h5.PINNED)
