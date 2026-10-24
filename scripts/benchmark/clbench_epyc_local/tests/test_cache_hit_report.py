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


def test_hybrid_basis_does_not_count_generated_tokens_and_cme5_basis_still_available():
    # perfect prompt-prefix reuse (cached == previous input) on a model that cannot reuse its own output
    t = trace([(100, 50, 0), (200, 50, 100), (300, 50, 200)])
    r = M.analyse([t])
    assert r["hit_ratio_prompt_prefix"] == 1.0 and r["verdict"] == "PASS" and r["hit_ratio_reusable"] < 0.95
    assert r["missed_split"] == {"generated_tokens_unreusable": 100, "history_rewind": 0, "prompt_prefix_lost": 0}
    assert M.analyse([t], basis="cme5")["verdict"] == "MISS"


def test_recorded_cme7_poker_smoke_fixture_numbers_and_split():
    """First lines of the 2026-10-22T13:12Z smoke (run 0 calls 3-5 plus the retry rewind): the lost part is one whole user message."""
    t = trace([(457, 2048, 0), (2607, 207, 0), (870, 186, 0), (1273, 223, 505), (1720, 222, 897), (2346, 162, 1337)])
    r = M.analyse([t])
    assert r["verdict"] == "MISS" and r["hit_ratio_prompt_prefix"] < 0.95
    sp = r["missed_split"]
    assert sp["history_rewind"] == 870 and sp["generated_tokens_unreusable"] > 0 and sp["prompt_prefix_lost"] > 0
    assert sum(sp.values()) == r["reusable_tokens"] - r["cached_tokens"]


def test_missing_counter_and_wrong_counts_are_invalid_not_zero():
    t = trace([(100, 10, 0), (200, 10, 110)])
    t["interactions"][1]["usage"]["respond"]["events"][0]["cached_input_tokens"] = None
    assert M.analyse([t])["verdict"] == "INVALID"
    assert M.analyse([trace([(100, 10, 0)], hands=4)], expect_hands=5)["verdict"] == "INVALID"
    assert M.analyse([], expect_runs=3)["verdict"] == "INVALID"


# --- VB-CLBENCH-CACHE-WIRE: --belief-out emit (writer lives in epyc-root, EPYC_ROOT) ---------------------------

import os  # noqa: E402

import pytest  # noqa: E402

_ROOT = os.environ.get("EPYC_ROOT", "")
_IDENT = ["--served-model", "Qwen3.6-35B-A3B-Q4_K_M", "--recipe", "cme7-poker-smoke", "--np", "1", "--ctx", "65536",
          "--slots", "1", "--kernel-build", "10303"]


def _traces(tmp_path, calls):
    d = tmp_path / "traces"
    d.mkdir()
    (d / "run_0000.json").write_text(json.dumps(trace(calls)))
    return d


def test_belief_out_requires_identity_flags_and_out(tmp_path, capsys):
    d = _traces(tmp_path, [(100, 50, 0), (200, 50, 100)])
    with pytest.raises(SystemExit) as e:
        M.main([str(d), "--belief-out", str(tmp_path / "b"), "--out", str(tmp_path / "r.json")])
    assert e.value.code == 2 and "--served-model" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        M.main([str(d), "--belief-out", str(tmp_path / "b"), *_IDENT])  # no --out


def test_no_flag_stays_stdlib_only(tmp_path, capsys):
    d = _traces(tmp_path, [(100, 50, 0), (200, 50, 100)])
    sys.modules.pop("belief_capture", None)
    assert M.main([str(d)]) == 0 and "belief_capture" not in sys.modules


@pytest.mark.skipif(not _ROOT, reason="EPYC_ROOT not set")
def test_belief_out_writes_sidecar(tmp_path):
    d = _traces(tmp_path, [(100, 50, 0), (200, 50, 100), (300, 50, 200)])
    rc = M.main([str(d), "--out", str(tmp_path / "r.json"), "--belief-out", str(tmp_path / "b"), *_IDENT])
    assert rc == 0
    side = tmp_path / "b" / "clbench_cache_hit.beliefs.jsonl"
    rows = [json.loads(l) for l in side.read_text().splitlines()]
    assert len(rows) == 5 and {r["extra"]["identity"]["research_sha"] for r in rows} and rows[0]["protocol_id"] == ""


def test_capture_failure_exit_code_still_writes_out(tmp_path, monkeypatch):
    d = _traces(tmp_path, [(100, 50, 0), (200, 50, 100)])
    monkeypatch.delenv("EPYC_ROOT", raising=False)
    out = tmp_path / "r.json"
    rc = M.main([str(d), "--out", str(out), "--belief-out", str(tmp_path / "b"), *_IDENT])
    assert rc == 3 and json.loads(out.read_text())["schema"] == "epyc.clbench_cache_hit.v1"
    assert not (tmp_path / "b" / "clbench_cache_hit.beliefs.jsonl").exists()


def test_driver_failure_code_is_not_masked_by_capture_failure(tmp_path, monkeypatch):
    d = _traces(tmp_path, [(100, 50, 0), (200, 10, 20)])  # MISS -> rc 1
    monkeypatch.delenv("EPYC_ROOT", raising=False)
    assert M.main([str(d), "--out", str(tmp_path / "r.json"), "--belief-out", str(tmp_path / "b"), *_IDENT]) == 1
