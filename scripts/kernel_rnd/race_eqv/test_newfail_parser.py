"""Pure-python checks of the r8 "Failing tests:" parser and baseline gate (no GPU, no logs)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_signals import newfail as nf

LOG = ("  MUL_MAT(type_a=f16,m=1): OK\n"
       "AK_VALUE_V1 foo completed=3 | test failed\n"
       "Failing tests:\n  MUL_MAT(type_a=f16,m=2)\n  GATED_DELTA_NET(type=f32)\n\n"
       "10/12 tests passed\n")


def test_failing_list_parsed():
    p = nf.parse_log(LOG)
    assert (p["passed"], p["total"]) == (10, 12)
    assert p["n_fail"] >= 2
    assert any("MUL_MAT(type_a=f16,m=2)" in k for k in p["fail_counts"])


def test_case_key_drops_bound():
    assert nf.case_key("OP(a=1,max_nmse=1e-7)") == "OP(a=1)"


def test_small_baseline_unusable():
    ok, why = nf.baseline_usable({"complete": True, "total": 5, "passed": 5, "consistent": True})
    assert not ok
