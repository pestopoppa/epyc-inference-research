import importlib.util
import json
import sys
from pathlib import Path

_S = importlib.util.spec_from_file_location("cache_hit_report", Path(__file__).resolve().parents[1] / "cache_hit_report.py")
M = importlib.util.module_from_spec(_S); sys.modules["cache_hit_report"] = M; _S.loader.exec_module(M)


def trace(calls, hands=None):
    evs = [{"input_tokens": i, "output_tokens": o, "cached_input_tokens": c} for i, o, c in calls]
    return {"interactions": [{"usage": {"respond": {"events": [e]}}} for e in evs],
            "instance_outcomes": [{}] * (hands if hands is not None else len(calls))}


def test_first_call_cold_and_new_suffix_do_not_count_against_gate():
    # call2 reuses all of call1 (100+10=110 of 200): cached 110 -> 100% of reusable although 55% of input
    t = trace([(100, 10, 0), (200, 10, 110), (300, 10, 210)])
    r = M.analyse([t])
    assert r["verdict"] == "PASS" and r["hit_ratio_reusable"] == 1.0 and r["raw_cached_over_input"] < 0.7


def test_miss_when_prefix_not_reused():
    r = M.analyse([trace([(100, 10, 0), (200, 10, 20), (300, 10, 30)])])
    assert r["verdict"] == "MISS" and r["hit_ratio_reusable"] < 0.95


def test_missing_counter_and_wrong_counts_are_invalid_not_zero():
    t = trace([(100, 10, 0), (200, 10, 110)])
    t["interactions"][1]["usage"]["respond"]["events"][0]["cached_input_tokens"] = None
    assert M.analyse([t])["verdict"] == "INVALID"
    assert M.analyse([trace([(100, 10, 0)], hands=4)], expect_hands=5)["verdict"] == "INVALID"
    assert M.analyse([], expect_runs=3)["verdict"] == "INVALID"
