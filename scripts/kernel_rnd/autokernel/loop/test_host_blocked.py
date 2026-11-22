import pytest

from . import host_blocked as hb
from .gates import Verdict


@pytest.mark.parametrize("verdict,case", [
    (Verdict("correctness", False, "test-backend-ops failed", "hipErrorNoDevice: no ROCm-capable device"), "gpu_absent"),
    (Verdict("correctness", False, "residency check failed: weights not resident"), "residency_failed"),
    (Verdict("compile", False, "build failed", "c++: fatal error: Killed signal terminated program cc1plus"), "oom"),
    (Verdict("compile", False, "build failed", "No space left on device"), "host_resource"),
])
def test_host_faults_are_blocked(verdict, case):
    out = hb.classify_verdict(verdict)
    assert out["class"] == hb.BLOCKED and out["case"] == case


@pytest.mark.parametrize("verdict", [
    Verdict("compile", False, "build failed", "error: 'foo' was not declared in this scope"),
    Verdict("reference_comparison", False, "wrong output: max err 3.2e-1"),
    Verdict("op_scope", False, "out of memory route not admitted"),  # rule gate never blocked
])
def test_candidate_faults_stay_failed(verdict):
    assert hb.classify_verdict(verdict)["class"] == hb.FAILED


def test_passed_verdict_refuses():
    with pytest.raises(ValueError):
        hb.classify_verdict(Verdict("compile", True))
