"""Unit tests for the INF06 exact-order lane (pure Python, no inference)."""
import json

import pytest

from . import exact_lane as el

PROMPTS = ["p0", "p1"]


def streams(n=40, base=100):
    return {(p, a): list(range(base, base + n)) for p in PROMPTS for a in el.ARMS}


def test_identical_streams_pass():
    v = el.judge_streams(streams(), streams(), PROMPTS)
    assert v.passed and v.cells == 4 and v.tokens_compared == 160


def test_one_differing_token_fails():
    cand = streams()
    cand[("p1", "drafted")][17] += 1
    v = el.judge_streams(streams(), cand, PROMPTS)
    assert not v.passed
    assert v.first_divergence == {"prompt": "p1", "arm": "drafted", "index": 17,
                                  "incumbent": 117, "candidate": 118}


@pytest.mark.parametrize("n", [0, 5])
def test_empty_or_short_stream_fails(n):
    v = el.judge_streams(streams(n), streams(n), PROMPTS)
    assert not v.passed and "no vacuous pass" in v.reason


def test_length_mismatch_fails():
    cand = streams()
    cand[("p0", "plain")].append(7)
    assert not el.judge_streams(streams(), cand, PROMPTS).passed


def test_missing_drafted_arm_fails():
    cand = {k: v for k, v in streams().items() if k[1] != "drafted"}
    v = el.judge_streams(streams(), cand, PROMPTS)
    assert not v.passed and ("candidate", "p0", "drafted") in v.missing


def test_drafted_arm_cannot_be_dropped_from_arms():
    v = el.judge_streams(streams(), streams(), PROMPTS, arms=("plain",))
    assert not v.passed


def test_no_prompts_fails():
    assert not el.judge_streams({}, {}, []).passed


def test_non_int_tokens_fail():
    cand = streams()
    cand[("p0", "plain")] = ["a"] * 40
    assert not el.judge_streams(streams(), cand, PROMPTS).passed


def test_lane_default_is_tolerance():
    assert el.candidate_lane({}) == "tolerance"
    assert el.candidate_lane(None) == "tolerance"
    assert el.candidate_lane({"lane": None}) == "tolerance"
    assert el.gate_plan({}) == el.LANE_GATES["tolerance"]
    assert "quality_eval" in el.gate_plan({})


def test_config_distinguishes_lanes():
    assert "quality_eval" not in el.gate_plan({"lane": "exact"})
    assert "exact_token_identity" in el.gate_plan({"lane": "exact"})


def test_unknown_lane_rejected():
    with pytest.raises(el.LaneError):
        el.candidate_lane({"lane": "fast"})


def test_routing():
    ok = el.judge_streams(streams(), streams(), PROMPTS)
    bad = el.judge_streams(streams(), streams(base=1), PROMPTS)
    exact = {"lane": "exact"}
    assert el.route_after({}, ok) == "quality_eval"          # tolerance unchanged
    assert el.route_after(exact, ok) == "speed_abba"           # quality skipped
    assert el.route_after(exact, bad) == "reject"
    assert el.route_after(exact, bad, on_mismatch="tolerance") == "quality_eval"
    assert el.route_after(exact, None) == "reject"             # no verdict != pass


def test_request_body_is_greedy_fixed_seed_ignore_eos():
    b = json.loads(el.request_body("x", n_predict=64, seed=1))
    assert b["temperature"] == 0 and b["top_k"] == 1 and b["seed"] == 1
    assert b["ignore_eos"] is True and b["return_tokens"] is True
    assert b["cache_prompt"] is False


# ---- loop dispatch wiring (lane_gate in loop.iterate) ----------------------------

from . import gates as _gates  # noqa: E402
from . import loop as _loop  # noqa: E402


def _hyp(**kw):
    return _loop.Hypothesis(mechanism_id="m", statement="s", falsifier="f",
                            target_surface="cpu", target_symbol="x", **kw)


class _Gate:
    def __init__(self, result):
        self.result, self.calls = result, []

    def __call__(self, hypothesis, paths):
        self.calls.append((hypothesis, paths))
        return self.result


def test_tolerance_dispatch_is_unchanged():
    result = (False, [_gates.Verdict("compile", False, "boom")])
    gate = _Gate(result)
    judge_calls = []
    d = el.lane_gate(gate, lambda h, p: judge_calls.append(1))
    for h in (_hyp(), _hyp(lane="tolerance")):
        out = d(h, ("a.cpp",))
        assert out is result                      # the very same object, untouched
    assert [c[1] for c in gate.calls] == [("a.cpp",), ("a.cpp",)]
    assert judge_calls == []


def test_absent_lane_keeps_hypothesis_row_shape():
    import json as _json
    row = _hyp().to_dict()
    assert "lane" not in row
    assert _json.dumps(row, sort_keys=True) == _json.dumps(
        {"mechanism_id": "m", "statement": "s", "falsifier": "f",
         "target_surface": "cpu", "target_symbol": "x"}, sort_keys=True)
    assert _hyp(lane="exact").to_dict()["lane"] == "exact"


def test_exact_pass_skips_quality_gate():
    gate = _Gate((True, []))
    ok = el.judge_streams(streams(), streams(), PROMPTS)
    passed, verdicts = el.lane_gate(gate, lambda h, p: ok)(_hyp(lane="exact"), ("a",))
    assert passed and gate.calls == []
    assert [v.gate for v in verdicts] == ["exact_token_identity"]


def test_exact_fail_rejects_without_quality_gate():
    gate = _Gate((True, []))
    bad = el.judge_streams(streams(), streams(base=1), PROMPTS)
    passed, verdicts = el.lane_gate(gate, lambda h, p: bad)(_hyp(lane="exact"), ("a",))
    assert not passed and gate.calls == []
    assert verdicts[0].gate == "exact_token_identity" and not verdicts[0].passed


def test_exact_fail_reroute_to_tolerance_runs_quality_gate():
    q = _gates.Verdict("quality", True, "")
    gate = _Gate((True, [q]))
    bad = el.judge_streams(streams(), streams(base=1), PROMPTS)
    passed, verdicts = el.lane_gate(gate, lambda h, p: bad, on_mismatch="tolerance")(
        _hyp(lane="exact"), ("a",))
    assert passed and len(gate.calls) == 1 and verdicts[-1] is q
    assert not verdicts[0].passed


@pytest.mark.parametrize("judge", [None, lambda h, p: (_ for _ in ()).throw(RuntimeError("x")),
                                   lambda h, p: "PASS"])
def test_exact_without_valid_verdict_rejects(judge):
    gate = _Gate((True, []))
    passed, _ = el.lane_gate(gate, judge)(_hyp(lane="exact"), ())
    assert not passed and gate.calls == []


def test_iterate_wires_lane_gate():
    import inspect
    src = inspect.getsource(_loop.iterate)
    assert "exact_lane.lane_gate(gate, exact_judge" in src
    assert inspect.signature(_loop.iterate).parameters["exact_judge"].default is None
