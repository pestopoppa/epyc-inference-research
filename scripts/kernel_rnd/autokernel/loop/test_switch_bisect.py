"""Kill-switch bisect state machine, incl. the C7 Q38FN replay and negative controls."""
import pytest

from . import runtime_arms as ra
from .experiment_plan import PlanValidationError

SW = {
    "TOPK_RADIX": {"env": {"GGML_CUDA_TOPK_RADIX": "0"}, "witnesses": ["radix"], "suspicion": 1},
    "HC_RMS_FUSE": {"env": {"GGML_CUDA_HC_RMS_FUSE": "0"}, "suspicion": 0, "breadth": 1},
    "HC_FUSE": {"env": {"GGML_CUDA_HC_FUSE": "0"}, "witnesses": ["hc"], "suspicion": 0, "breadth": 2},
    "Q8_WVSK": {"env": {"GGML_HIP_Q8_WVSK": "0"}, "witnesses": ["q8wv"], "suspicion": 9},
}


def _b(**kw):
    return ra.switch_bisect(SW, tol=0.005, repro_min=0.05, default_witnesses=["hc"], **kw)


def test_c7_replay_culprit_hc_fuse_after_one_single():
    b = _b()
    # HC_FUSE ranked first here (suspicion 0, narrow-first would put HC_RMS_FUSE before it)
    b.fold("default", 0.1264, ["hc", "radix"])
    assert b.next_arm()["arm"] == "all_off"
    all_off = b.next_arm()
    assert all_off["env"]["GGML_CUDA_HC_FUSE"] == "0" and "hc" in all_off["must_absent"]
    b.fold("all_off", 0.0012)
    first = b.next_arm()["arm"]
    assert first == "HC_RMS_FUSE"            # narrow before broad
    assert b.fold(first, 0.126) is None
    assert b.fold("HC_FUSE", 0.0013) == "CULPRIT(HC_FUSE)"
    assert b.next_arm() is None and b.detail["not_run"]  # stop on answer


def test_c7_replay_when_hc_fuse_ranked_first():
    sw = {k: dict(v) for k, v in SW.items()}
    sw["HC_FUSE"]["suspicion"] = -1
    b = ra.switch_bisect(sw, tol=0.005, repro_min=0.05, default_witnesses=["hc"])
    b.fold("default", 0.126, ["hc"]); b.fold("all_off", 0.001)
    assert b.fold(b.next_arm()["arm"], 0.001) == "CULPRIT(HC_FUSE)"
    assert len(b.units) == 3


def test_negative_not_reproduced_stops():
    b = _b()
    assert b.fold("default", 0.01, ["hc"]) == "NOT_REPRODUCED"
    assert b.next_arm() is None


def test_negative_not_switchable_stops():
    b = _b(); b.fold("default", 0.126, ["hc"])
    assert b.fold("all_off", 0.126) == "NOT_SWITCHABLE" and b.next_arm() is None


def test_vacuous_default_witness_voids_then_inconclusive():
    b = _b()
    assert b.fold("default", 0.126, []) is None          # default did not log hc: void
    assert "default" not in b.units and b.next_arm()["arm"] == "default"
    assert b.fold("default", 0.126, []) == "INCONCLUSIVE"


def test_off_arm_logging_its_witness_voids():
    b = _b(); b.fold("default", 0.126, ["hc"])
    assert b.fold("all_off", 0.0, ["hc"]) is None        # switch off but witness still logged
    assert "all_off" not in b.units


def test_invalid_unit_voids_and_wrong_arm_refused():
    b = _b()
    assert b.fold("default", None, ["hc"], valid=False) is None
    with pytest.raises(PlanValidationError):
        b.fold("all_off", 0.0)


def test_multi_cause_when_no_single_restores():
    b = _b(); b.fold("default", 0.126, ["hc"]); b.fold("all_off", 0.0)
    v = None
    while b.next_arm():
        v = b.fold(b.next_arm()["arm"], 0.1)
    assert v == "MULTI_CAUSE"
